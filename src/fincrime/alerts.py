"""Account-level risk: combine model scores, rule hits and network membership into an alert queue.

    risk = 0.55 x model evidence + 0.25 x rule evidence + 0.20 x network evidence      (weights in config.py)

    model evidence    mean of the account's three highest transaction scores in the window
                      (an account needs repeated suspicious activity to score highly; one odd
                      payment, such as a scam victim's, is not enough)
    rule evidence     1 - product(1 - weight) over the distinct AML rules the account triggered
    network evidence  average evidence of the other accounts in the same detected network

The alert threshold is chosen on the validation window (Aug-Sep) and then applied, unchanged,
to the monitoring window (Oct-Dec).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .graph.network import detect_networks, evaluate_networks
from .models.train import rules_baseline_score

SEED_THRESHOLD = 0.30
TOP_K = [100, 250, 500, 1000]


def load_scored(data_dir: Path) -> pd.DataFrame:
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    return tx.merge(pd.read_parquet(data_dir / "scores.parquet"), on="txn_id")


def _sides(tx: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Long form: one row per (transaction, internal account involved)."""
    s = tx.loc[tx.src_account >= 0, ["src_account"] + cols].rename(columns={"src_account": "account_id"})
    d = tx.loc[tx.dst_account >= 0, ["dst_account"] + cols].rename(columns={"dst_account": "account_id"})
    return pd.concat([s, d], ignore_index=True)


def base_scores(tx: pd.DataFrame, hits: pd.DataFrame, flagged_ids: set) -> pd.DataFrame:
    long = _sides(tx, ["txn_id", "score", "amount"])
    rank = long.groupby("account_id").score.rank(method="first", ascending=False)
    top3 = long[rank <= 3].groupby("account_id").score.sum() / 3      # accounts with fewer than 3 transactions are padded with zeros
    out = pd.DataFrame({"ml": top3, "n_txns": long.groupby("account_id").size()})
    fl = long[long.txn_id.isin(flagged_ids)]
    out["n_flagged"] = fl.groupby("account_id").size()
    out["exposure"] = fl.groupby("account_id").amount.sum()
    w = C.RULE_WEIGHTS
    if len(hits):
        rules = hits.groupby("account_id").rule.agg(lambda r: sorted(set(r)))
        out["rules"] = rules.map(lambda r: 1 - float(np.prod([1 - w[x] for x in r])))
        out["rule_list"] = rules
    else:
        out["rules"], out["rule_list"] = 0.0, None
    out = out.fillna({"n_flagged": 0, "exposure": 0.0, "rules": 0.0})
    out["rule_list"] = [r if isinstance(r, list) else [] for r in out.rule_list]
    wm, wr = C.ACCOUNT_WEIGHTS["ml"], C.ACCOUNT_WEIGHTS["rules"]
    out["base"] = (wm * out.ml + wr * out.rules) / (wm + wr)
    out.index.name = "account_id"
    return out


def score_window(tx_all: pd.DataFrame, hits_all: pd.DataFrame, txn_threshold: float, start, end):
    """Score every account active in [start, end) and detect networks. Returns (accounts, nets, members, edges, flagged)."""
    tx = tx_all[(tx_all.ts >= start) & (tx_all.ts < end)]
    hits = hits_all[(hits_all.end_ts >= start) & (hits_all.end_ts < end)]
    in_window = set(tx.txn_id)
    rule_txns = {t for ids in hits.txn_ids for t in ids if t in in_window}
    flagged = set(tx.txn_id[tx.score >= txn_threshold]) | rule_txns
    acc = base_scores(tx, hits, flagged)
    nets, mem, edges = detect_networks(tx, acc, flagged, hits, SEED_THRESHOLD)
    acc["detected_network_id"] = mem.set_index("account_id").detected_network_id.reindex(acc.index).fillna(0).astype(int)
    # network evidence: the average evidence of the other members
    acc["network"] = 0.0
    if len(mem):
        m = mem[["detected_network_id", "account_id"]].merge(acc[["base"]], left_on="account_id", right_index=True)
        g = m.groupby("detected_network_id").base
        others = (g.transform("sum") - m.base) / (g.transform("size") - 1)
        acc.loc[m.account_id.to_numpy(), "network"] = others.clip(0, 1).to_numpy()
    w = C.ACCOUNT_WEIGHTS
    acc["risk"] = w["ml"] * acc.ml + w["rules"] * acc.rules + w["network"] * acc.network
    return acc, nets, mem, edges, flagged, tx, hits


def window_truth(tx: pd.DataFrame, true_members: pd.DataFrame) -> set:
    """Network member accounts that actually moved suspicious money inside the window."""
    sus = tx[tx.is_suspicious == 1]
    return (set(sus.src_account) | set(sus.dst_account)) & set(true_members.account_id)


def choose_alert_threshold(acc: pd.DataFrame, truth: set) -> float:
    y = acc.index.isin(truth)
    best, best_f1 = 0.30, -1.0
    for t in np.arange(0.15, 0.4501, 0.01):
        pred = (acc.risk >= t).to_numpy()
        tp = (pred & y).sum()
        p, r = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best, best_f1 = float(round(t, 2)), f1
    return best


def priority(risk: pd.Series) -> pd.Series:
    out = pd.Series("Medium", index=risk.index)
    for name, cut in reversed(C.PRIORITY_BANDS):
        out[risk >= cut] = name
    return out


def effectiveness(acc: pd.DataFrame, tx: pd.DataFrame, truth: set, true_members: pd.DataFrame, n_days: int) -> list[dict]:
    """What an investigator gets from working the top-K accounts under different ranking strategies."""
    sus = tx[tx.is_suspicious == 1]
    acct_net = true_members.set_index("account_id").network_id
    active_nets = set(sus.network_id.unique())
    total_amt = sus.amount.sum()
    base = tx.assign(flag=rules_baseline_score_tx(tx))
    static = _sides(base[base.flag], ["amount"]).groupby("account_id").amount.agg(["size", "sum"])
    static_rank = (static["size"] + static["sum"] / 1e9).reindex(acc.index).fillna(0)
    rankings = {
        "Hybrid risk score (model + rules + network)": acc.risk,
        "Model only": acc.ml,
        "AML rules only": acc.rules + acc.exposure / 1e12,
        "Static baseline (10k+ / high-risk corridor rule)": static_rank,
    }
    rows = []
    for name, s in rankings.items():
        order = s.sort_values(ascending=False, kind="stable").index
        for k in TOP_K:
            if k > len(order):
                continue
            top = set(order[:k])
            hit = top & truth
            nets = set(acct_net.reindex(list(hit)).dropna().astype(int))
            covered = sus[sus.src_account.isin(top) | sus.dst_account.isin(top)].amount.sum()
            rows.append({"strategy": name, "accounts_reviewed": k, "per_week": round(k / (n_days / 7), 1),
                         "precision": len(hit) / k, "recall": len(hit) / max(len(truth), 1),
                         "networks_reached": len(nets & active_nets) / max(len(active_nets), 1),
                         "suspicious_value_covered": float(covered / total_amt) if total_amt else 0.0})
    return rows


def rules_baseline_score_tx(tx: pd.DataFrame) -> pd.Series:
    """The static baseline rule evaluated on raw transactions (no engineered features needed)."""
    f = pd.DataFrame({"amount": tx.amount, "log_amount": np.log1p(tx.amount), "txn_type": tx.txn_type,
                      "is_high_risk_geo": tx.ext_country.isin(C.HIGH_RISK).astype(int),
                      "just_below_threshold": ((tx.amount >= C.STRUCT_LOW) & (tx.amount < C.CASH_REPORT_THRESHOLD)).astype(int)})
    return pd.Series(rules_baseline_score(f) >= 1, index=tx.index)


def alert_times(acc: pd.DataFrame, tx: pd.DataFrame, hits: pd.DataFrame, alert_thr: float) -> pd.Series:
    """When each alerted account would first have crossed the alert threshold.

    Replays the account's window transactions in time order and recomputes model and rule evidence
    as it accumulates. Network evidence is left out of the replay, so the result is conservative;
    an account that only crosses the line with network evidence is stamped at its last flagged event.
    """
    wm, wr = C.ACCOUNT_WEIGHTS["ml"], C.ACCOUNT_WEIGHTS["rules"]
    alerted = acc.index[acc.alert]
    long = _sides(tx, ["ts", "score"])
    long = long[long.account_id.isin(alerted)].sort_values("ts")
    by_acct = {a: g for a, g in long.groupby("account_id")}
    rule_events = {a: sorted(zip(g.end_ts, g.rule)) for a, g in hits[hits.account_id.isin(alerted)].groupby("account_id")}
    out = {}
    for a in alerted:
        g = by_acct.get(a)
        events = [(t, "txn", sc) for t, sc in zip(g.ts, g.score)] if g is not None else []
        events += [(t, "rule", r) for t, r in rule_events.get(a, [])]
        events.sort(key=lambda e: e[0])
        top, rules, when = [], set(), None
        for t, kind, val in events:
            if kind == "txn":
                top = sorted(top + [val], reverse=True)[:3]
            else:
                rules.add(val)
            risk = wm * sum(top) / 3 + wr * (1 - float(np.prod([1 - C.RULE_WEIGHTS[r] for r in rules])))
            if risk >= alert_thr:
                when = t
                break
        out[a] = when if when is not None else (events[-1][0] if events else pd.NaT)
    return pd.Series(out, dtype="datetime64[us]")


def time_to_detect(tx: pd.DataFrame, alert_ts: pd.Series, true_members: pd.DataFrame, true_nets: pd.DataFrame, start) -> dict:
    """Hours from a network's first suspicious transaction to the first alert on one of its members.

    Only networks that began inside the window are counted, so earlier activity cannot flatter the number.
    """
    sus = tx[tx.is_suspicious == 1]
    first_sus = sus.groupby("network_id").ts.min()
    began = set(true_nets.network_id[true_nets.first_ts >= start]) & set(first_sus.index)
    m = true_members[true_members.account_id.isin(alert_ts.index)]
    first_alert = m.assign(ts=m.account_id.map(alert_ts)).groupby("network_id").ts.min()
    d = ((first_alert - first_sus).dropna().dt.total_seconds() / 3600)
    d = d[d.index.isin(began)]
    return {"networks_started_in_window": int(len(began)), "networks_alerted": int(len(d)),
            "median_hours": float(d.median()) if len(d) else None, "p75_hours": float(d.quantile(0.75)) if len(d) else None,
            "within_48h": float((d <= 48).mean()) if len(d) else None}
