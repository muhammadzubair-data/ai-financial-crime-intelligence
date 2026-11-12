"""Fraud-ring and network detection with NetworkX.

The idea, in three steps:

1. *Suspects* are accounts whose own evidence (model score plus rule hits) is already notable.
2. Suspects are linked when a flagged transfer passes between them, or when they have used
   the same device. A non-suspect is pulled in only if flagged transfers connect it to two or
   more suspects (this recovers quiet collectors while keeping one-off scam victims out).
3. Each connected group of three or more accounts is reported as a candidate network, with
   roles inferred from who pays whom.

Nothing here uses the labels. ``evaluate_networks`` compares the result with the injected
ground truth afterwards.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .. import config as C

MAX_DEVICE_FANOUT = 30      # a "device" seen on more accounts than this is treated as shared infrastructure
NET_COLS = ["detected_network_id", "n_accounts", "n_transfer_links", "n_device_links", "flagged_internal_amount",
            "exposure", "first_ts", "last_ts", "density", "key_account", "patterns", "inferred_type", "mean_score",
            "max_score"]
MEM_COLS = ["detected_network_id", "account_id", "role", "betweenness", "degree", "in_amount", "out_amount", "exit_amount"]
EDGE_COLS = ["detected_network_id", "src", "dst", "kind", "n_txns", "amount"]


def _infer_type(patterns: set[str], n_device_links: int) -> str:
    if "CIRCULAR" in patterns:
        return "Circular transfers"
    if "STRUCTURING" in patterns:
        return "Structuring"
    if "FAN_IN" in patterns or n_device_links > 0:
        return "Mule ring"
    if "RAPID_MOVEMENT" in patterns:
        return "Layering chain"
    return "Linked suspicious accounts"


def detect_networks(tx: pd.DataFrame, base: pd.DataFrame, flagged_ids: set, hits: pd.DataFrame, seed_threshold: float):
    """tx: transactions in the window. base: per-account evidence indexed by account_id (column ``base``)."""
    seeds = set(base.index[base.base >= seed_threshold])
    fl = tx[tx.txn_id.isin(flagged_ids)]
    internal = fl[(fl.src_account >= 0) & (fl.dst_account >= 0)]
    pair = (internal.groupby(["src_account", "dst_account"]).agg(n_txns=("amount", "size"), amount=("amount", "sum"))
            .reset_index())

    # accounts tied to two or more suspects by flagged transfers
    a = pd.concat([pair.rename(columns={"src_account": "x", "dst_account": "y"})[["x", "y"]],
                   pair.rename(columns={"dst_account": "x", "src_account": "y"})[["x", "y"]]])
    a = a[a.y.isin(seeds) & ~a.x.isin(seeds)]
    extra = set(a.groupby("x").y.nunique().loc[lambda s: s >= 2].index)
    nodes = seeds | extra

    g = nx.Graph()
    g.add_nodes_from(nodes)
    p = pair[pair.src_account.isin(nodes) & pair.dst_account.isin(nodes)]
    for s, d, n, amt in p.itertuples(index=False):
        if g.has_edge(s, d):
            g[s][d]["n_txns"] += n
            g[s][d]["amount"] += amt
        else:
            g.add_edge(s, d, kind="transfer", n_txns=int(n), amount=float(amt))
    # shared devices among suspects
    dv = tx[tx.device_id >= 0]
    dv = dv.assign(account=np.where(dv.src_account >= 0, dv.src_account, dv.dst_account))
    dv = dv[dv.account.isin(nodes)].drop_duplicates(["device_id", "account"])
    n_dev = 0
    for _, grp in dv.groupby("device_id"):
        accts = list(grp.account)
        if 2 <= len(accts) <= MAX_DEVICE_FANOUT:
            for other in accts[1:]:
                if not g.has_edge(accts[0], other):
                    g.add_edge(accts[0], other, kind="device", n_txns=0, amount=0.0)
                    n_dev += 1
                else:
                    g[accts[0]][other]["device"] = True

    acct_rules = hits.groupby("account_id").rule.agg(set).to_dict() if len(hits) else {}
    ext_out = fl[(fl.src_account >= 0) & (fl.dst_account < 0)].groupby("src_account").amount.sum()
    comps = [c for c in nx.connected_components(g) if len(c) >= C.NETWORK_MIN_SIZE]
    nets, members, edges = [], [], []
    for comp in comps:
        sub = g.subgraph(comp)
        dg = nx.DiGraph()
        dg.add_nodes_from(comp)
        pp = p[p.src_account.isin(comp) & p.dst_account.isin(comp)]
        dg.add_weighted_edges_from(pp[["src_account", "dst_account", "amount"]].itertuples(index=False))
        btw = nx.betweenness_centrality(dg) if len(comp) > 2 else {n: 0.0 for n in comp}
        inflow, outflow = dict(dg.in_degree(weight="weight")), dict(dg.out_degree(weight="weight"))
        involved = fl[fl.src_account.isin(comp) | fl.dst_account.isin(comp)]
        patterns = set().union(*[acct_rules.get(n, set()) for n in comp])
        n_dev_links = sum(1 for _, _, d in sub.edges(data=True) if d["kind"] == "device")
        scores = base.base.reindex(list(comp)).fillna(0)
        key = max(comp, key=lambda n: (btw.get(n, 0), inflow.get(n, 0)))
        nets.append([0, len(comp), sub.number_of_edges() - n_dev_links, n_dev_links, float(pp.amount.sum()),
                     float(involved.amount.sum()), involved.ts.min(), involved.ts.max(), float(nx.density(sub)), int(key),
                     sorted(patterns), _infer_type(patterns, n_dev_links), float(scores.mean()), float(scores.max()),
                     comp])
        for n in comp:
            ex = float(ext_out.get(n, 0.0))
            if ex > 0 and ex >= 0.3 * (ex + inflow.get(n, 0) + outflow.get(n, 0)):     # most of its flagged money leaves the bank
                role = "Exit point"
            elif inflow.get(n, 0) > 1.5 * outflow.get(n, 0) and dg.in_degree(n) >= 2:
                role = "Collector"
            elif outflow.get(n, 0) > 0 and inflow.get(n, 0) > 0:
                role = "Pass-through"
            elif outflow.get(n, 0) > 0:
                role = "Feeder"
            else:
                role = "Linked"
            members.append([comp, int(n), role, float(btw.get(n, 0)), int(sub.degree(n)), float(inflow.get(n, 0)),
                            float(outflow.get(n, 0)), ex])
        for s, d, n, amt in pp.itertuples(index=False):
            edges.append([comp, int(s), int(d), "transfer", int(n), float(amt)])
        for u, v, d in sub.edges(data=True):
            if d["kind"] == "device" or d.get("device"):
                edges.append([comp, int(u), int(v), "device", 0, 0.0])

    # number networks by exposure, largest first
    order = sorted(range(len(nets)), key=lambda i: -nets[i][5])
    ids = {id(nets[i][-1]): k + 1 for k, i in enumerate(order)}
    for row in nets:
        row[0] = ids[id(row[-1])]
    nets_df = pd.DataFrame([r[:-1] for r in nets], columns=NET_COLS).sort_values("detected_network_id").reset_index(drop=True)
    mem_df = pd.DataFrame([[ids[id(m[0])]] + m[1:] for m in members], columns=MEM_COLS)
    edge_df = pd.DataFrame([[ids[id(e[0])]] + e[1:] for e in edges], columns=EDGE_COLS)
    return nets_df, mem_df, edge_df


def evaluate_networks(nets: pd.DataFrame, mem: pd.DataFrame, tx: pd.DataFrame, true_members: pd.DataFrame,
                      true_nets: pd.DataFrame, alerted: set | None = None):
    """Compare detected networks with the injected ones that were active in the window."""
    sus = tx[tx.is_suspicious == 1]
    active = sus.groupby("network_id").size().loc[lambda s: s >= 3].index
    touched = set(sus.src_account) | set(sus.dst_account)
    tm = true_members[true_members.network_id.isin(active) & true_members.account_id.isin(touched)]
    truth = tm.groupby("network_id").account_id.agg(set)
    acct_true = true_members.set_index("account_id").network_id
    det = mem.groupby("detected_network_id").account_id.agg(set) if len(mem) else pd.Series(dtype=object)

    rows = []
    for nid, accts in truth.items():
        best, best_id = 0.0, -1
        for did, d in det.items():
            share = len(accts & d) / len(accts)
            if share > best:
                best, best_id = share, did
        rows.append({"network_id": int(nid), "active_members": len(accts), "best_detected_network": int(best_id),
                     "member_recall": best, "recovered": best >= 0.5,
                     "touched_by_alert": bool(alerted and accts & alerted)})
    per_true = pd.DataFrame(rows).merge(true_nets[["network_id", "typology", "stealth"]], on="network_id", how="left") \
        if rows else pd.DataFrame(columns=["network_id", "typology", "stealth", "recovered", "touched_by_alert", "member_recall"])

    purity, dominant = [], []
    for did, d in det.items():
        t = pd.Series(list(d)).map(acct_true)
        purity.append(float(t.notna().mean()))
        dominant.append(int(t.mode().iloc[0]) if t.notna().any() else -1)
    quality = pd.DataFrame({"detected_network_id": list(det.index), "purity": purity, "true_network_id": dominant})
    summary = {
        "true_networks_active": int(len(per_true)),
        "recovered": int(per_true.recovered.sum()) if len(per_true) else 0,
        "recovery_rate": float(per_true.recovered.mean()) if len(per_true) else 0.0,
        "touched_by_alert_rate": float(per_true.touched_by_alert.mean()) if len(per_true) else 0.0,
        "mean_member_recall": float(per_true.member_recall.mean()) if len(per_true) else 0.0,
        "detected_networks": int(len(det)),
        "detected_mostly_true": int((quality.purity >= 0.5).sum()),
        "mean_purity": float(quality.purity.mean()) if len(quality) else 0.0,
        "by_typology": (per_true.groupby("typology").recovered.agg(["size", "sum", "mean"]).reset_index()
                        .rename(columns={"size": "active", "sum": "recovered", "mean": "rate"}).to_dict("records"))
        if len(per_true) else [],
        "by_stealth": (per_true.groupby("stealth").recovered.agg(["size", "sum", "mean"]).reset_index()
                       .rename(columns={"size": "active", "sum": "recovered", "mean": "rate"}).to_dict("records"))
        if len(per_true) else [],
    }
    return summary, per_true, quality
