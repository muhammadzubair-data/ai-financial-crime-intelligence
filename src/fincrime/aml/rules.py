"""AML pattern monitoring: four transparent, label-free scenario rules.

    STRUCTURING      repeated cash deposits just under the reporting threshold
    RAPID_MOVEMENT   money in and (almost) the same money out within 24 hours
    FAN_IN           many first-time senders paying one account in a short period
    CIRCULAR         funds that travel round a loop of accounts and return to the originator

Each rule returns *instances*: one row per account per episode, with the transactions that
triggered it. Thresholds live in ``config.py``. The rules never look at the labels; labels are
only used afterwards, in ``evaluate_rules``, to report how often each rule is right.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from .. import config as C

HIT_COLS = ["rule", "account_id", "start_ts", "end_ts", "n_txns", "amount", "detail", "txn_ids", "linked_accounts"]
EPISODE = """
    , g AS (SELECT *, CASE WHEN ts - lag(ts) OVER (PARTITION BY account ORDER BY ts) <= INTERVAL {gap} DAYS
                           THEN 0 ELSE 1 END AS new_ep FROM f)
    , e AS (SELECT *, sum(new_ep) OVER (PARTITION BY account ORDER BY ts) AS ep FROM g)
"""


def _connect(data_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{data_dir / 'transactions.parquet'}')")
    con.execute(f"CREATE VIEW acct AS SELECT * FROM read_parquet('{data_dir / 'accounts.parquet'}')")
    return con


def structuring(con) -> pd.DataFrame:
    df = con.execute(f"""
        WITH f AS (SELECT dst_account AS account, ts, amount, txn_id FROM tx
                   WHERE txn_type = 'cash_deposit' AND amount >= {C.STRUCT_LOW} AND amount < {C.CASH_REPORT_THRESHOLD})
        {EPISODE.format(gap=C.STRUCT_WINDOW_DAYS)}
        SELECT account AS account_id, min(ts) AS start_ts, max(ts) AS end_ts, count(*) AS n_txns,
               sum(amount) AS amount, list(txn_id ORDER BY ts) AS txn_ids
        FROM e GROUP BY account, ep
        HAVING count(*) >= {C.STRUCT_MIN_DEPOSITS} AND sum(amount) >= {C.STRUCT_MIN_TOTAL}
    """).df()
    df["rule"] = "STRUCTURING"
    df["detail"] = [f"{n} cash deposits between {C.STRUCT_LOW:,} and {C.CASH_REPORT_THRESHOLD:,} totalling ${a:,.0f}"
                    for n, a in zip(df.n_txns, df.amount)]
    df["linked_accounts"] = [[] for _ in range(len(df))]
    return df


def rapid_movement(con) -> pd.DataFrame:
    lo, hi = C.RAPID_RATIO
    df = con.execute(f"""
        WITH ledger AS (
            SELECT txn_id, ts, src_account AS account, 'D' AS dir, amount, txn_type FROM tx WHERE src_account >= 0
            UNION ALL
            SELECT txn_id, ts, dst_account, 'C', amount, txn_type FROM tx WHERE dst_account >= 0),
        r AS (
            SELECT *,
                coalesce(sum(CASE WHEN dir = 'C' THEN amount END) OVER w, 0) AS cr_24h,
                coalesce(sum(CASE WHEN dir = 'D' THEN amount END) OVER w, 0) AS db_24h
            FROM ledger
            WINDOW w AS (PARTITION BY account ORDER BY ts
                         RANGE BETWEEN INTERVAL {C.RAPID_WINDOW_HOURS} HOURS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW)),
        f AS (
            SELECT account, ts, amount, txn_id, cr_24h FROM r
            WHERE dir = 'D' AND txn_type IN ('transfer', 'wire_out', 'cash_withdrawal')
              AND amount + db_24h >= {C.RAPID_MIN_AMOUNT} AND cr_24h >= {C.RAPID_MIN_AMOUNT}
              AND (amount + db_24h) / cr_24h BETWEEN {lo} AND {hi})
        {EPISODE.format(gap=3)}
        SELECT account AS account_id, min(ts) AS start_ts, max(ts) AS end_ts, count(*) AS n_txns,
               sum(amount) AS amount, max(cr_24h) AS inflow, list(txn_id ORDER BY ts) AS txn_ids
        FROM e GROUP BY account, ep
    """).df()
    df["rule"] = "RAPID_MOVEMENT"
    df["detail"] = [f"${a:,.0f} sent out within {C.RAPID_WINDOW_HOURS}h of receiving ${i:,.0f}"
                    for a, i in zip(df.amount, df.inflow)]
    df["linked_accounts"] = [[] for _ in range(len(df))]
    return df.drop(columns="inflow")


def fan_in(con) -> pd.DataFrame:
    r, b = C.FANIN_MIN_SENDERS["retail"], C.FANIN_MIN_SENDERS["business"]
    df = con.execute(f"""
        WITH p AS (
            SELECT dst_account AS account, src_account AS sender, ts, amount, txn_id,
                   row_number() OVER (PARTITION BY src_account, dst_account ORDER BY ts, txn_id) AS k
            FROM tx WHERE txn_type = 'transfer' AND src_account >= 0 AND dst_account >= 0),
        n AS (
            SELECT p.*, a.customer_type,
                   count(*) OVER w AS senders, sum(amount) OVER w AS total
            FROM p JOIN acct a ON a.account_id = p.account WHERE k = 1
            WINDOW w AS (PARTITION BY account ORDER BY ts RANGE BETWEEN INTERVAL {C.FANIN_WINDOW_DAYS} DAYS PRECEDING AND CURRENT ROW)),
        hot AS (SELECT account, ts FROM n
                WHERE total >= {C.FANIN_MIN_TOTAL} AND senders >= CASE WHEN customer_type = 'business' THEN {b} ELSE {r} END),
        f AS (  -- every first-time payment inside a window that ended in a trigger
            SELECT DISTINCT n.account, n.ts, n.amount, n.txn_id, n.sender FROM n JOIN hot h
            ON n.account = h.account AND n.ts <= h.ts AND n.ts >= h.ts - INTERVAL {C.FANIN_WINDOW_DAYS} DAYS)
        {EPISODE.format(gap=C.FANIN_WINDOW_DAYS)}
        SELECT account AS account_id, min(ts) AS start_ts, max(ts) AS end_ts, count(*) AS n_txns,
               sum(amount) AS amount, list(txn_id ORDER BY ts) AS txn_ids, list(DISTINCT sender) AS linked_accounts
        FROM e GROUP BY account, ep
    """).df()
    df["rule"] = "FAN_IN"
    df["detail"] = [f"{n} first-time senders paid in ${a:,.0f} within {C.FANIN_WINDOW_DAYS} days"
                    for n, a in zip(df.n_txns, df.amount)]
    return df


def circular(con) -> pd.DataFrame:
    """Time-respecting cycle search on internal transfers.

    A cycle is a chain A -> B -> ... -> A in which every hop happens after the previous one
    (within CIRC_MAX_HOP_HOURS), carries roughly the same amount, and the whole loop closes
    within CIRC_MAX_TOTAL_DAYS. NetworkX narrows the search to strongly connected components
    first, because an account outside one cannot be on any cycle.
    """
    e = con.execute(f"""
        SELECT txn_id, epoch(ts) AS t, ts, src_account AS u, dst_account AS v, amount FROM tx
        WHERE txn_type = 'transfer' AND src_account >= 0 AND dst_account >= 0 AND amount >= {C.CIRC_MIN_AMOUNT}
        ORDER BY ts
    """).df()
    if e.empty:
        return pd.DataFrame(columns=HIT_COLS)
    g = nx.DiGraph()
    g.add_edges_from(zip(e.u.to_numpy(), e.v.to_numpy()))
    keep = set().union(*[c for c in nx.strongly_connected_components(g) if len(c) >= 3] or [set()])
    e = e[e.u.isin(keep) & e.v.isin(keep)].reset_index(drop=True)
    out_t, out_e = {}, {}
    for i, (u, t) in enumerate(zip(e.u.to_numpy(), e.t.to_numpy())):
        out_t.setdefault(u, []).append(t)
        out_e.setdefault(u, []).append(i)
    t_arr, v_arr, a_arr = e.t.to_numpy(), e.v.to_numpy(), e.amount.to_numpy()
    hop, total = C.CIRC_MAX_HOP_HOURS * 3600, C.CIRC_MAX_TOTAL_DAYS * 86400
    lo, hi = C.CIRC_AMOUNT_BAND
    cycles, seen = [], set()

    def walk(path, nodes, origin, t_start):
        last = path[-1]
        u = v_arr[last]
        ts_list = out_t.get(u)
        if not ts_list:
            return
        j = bisect_right(ts_list, t_arr[last])
        while j < len(ts_list) and ts_list[j] <= t_arr[last] + hop and ts_list[j] <= t_start + total:
            k = out_e[u][j]
            j += 1
            if not (lo * a_arr[last] <= a_arr[k] <= hi * a_arr[last]):
                continue
            nxt = v_arr[k]
            if nxt == origin:
                if len(path) + 1 >= 3:
                    key = frozenset(path + [k])
                    if key not in seen:
                        seen.add(key)
                        cycles.append(path + [k])
            elif nxt not in nodes and len(path) + 1 < C.CIRC_MAX_LEN:
                walk(path + [k], nodes | {nxt}, origin, t_start)

    for i in range(len(e)):
        walk([i], {e.u[i], e.v[i]}, e.u[i], t_arr[i])

    rows = []
    for cyc in cycles:
        c = e.iloc[cyc]
        accounts = [int(a) for a in c.u]
        for a in accounts:
            rows.append(("CIRCULAR", a, c.ts.min(), c.ts.max(), len(cyc), float(c.amount.iloc[0]),
                         f"${c.amount.iloc[0]:,.0f} passed round {len(cyc)} accounts and returned to the originator "
                         f"in {(c.t.max() - c.t.min()) / 3600:.0f}h",
                         [int(x) for x in c.txn_id], [x for x in accounts if x != a]))
    return pd.DataFrame(rows, columns=HIT_COLS)


def run_rules(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or C.DATA_DIR)
    con = _connect(data_dir)
    parts = [f(con)[HIT_COLS] for f in (structuring, rapid_movement, fan_in, circular)]
    con.close()
    hits = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
        else pd.DataFrame(columns=HIT_COLS)
    hits["txn_ids"] = [[int(x) for x in v] for v in hits.txn_ids]
    hits["linked_accounts"] = [[int(x) for x in v] for v in hits.linked_accounts]
    hits["account_id"] = hits.account_id.astype("int64")
    hits["n_txns"] = hits.n_txns.astype("int64")
    hits.insert(0, "hit_id", np.arange(len(hits)))
    # ground truth, for evaluation only: an instance is "true" when most of its transactions are labelled suspicious
    tx = pd.read_parquet(data_dir / "transactions.parquet", columns=["txn_id", "is_suspicious"])
    sus = set(tx.txn_id[tx.is_suspicious == 1])
    hits["is_true"] = [np.mean([t in sus for t in ids]) >= 0.5 for ids in hits.txn_ids]
    hits.to_parquet(data_dir / "rule_hits.parquet", index=False)
    return hits


def evaluate_rules(hits: pd.DataFrame, data_dir: Path) -> list[dict]:
    """Precision of each rule, and the share of injected networks of the matching typology it touches."""
    members = pd.read_parquet(data_dir / "network_members.parquet")
    nets = pd.read_parquet(data_dir / "networks.parquet")
    acct_net = members.set_index("account_id").network_id
    targets = {"STRUCTURING": ["structuring"], "CIRCULAR": ["circular"], "RAPID_MOVEMENT": ["layering", "mule_ring"],
               "FAN_IN": ["mule_ring"]}
    out = []
    for rule, typs in targets.items():
        h = hits[hits.rule == rule]
        true_hits = h[h.is_true]
        found = set(true_hits.account_id.map(acct_net).dropna().astype(int))
        pool = nets[nets.typology.isin(typs)]
        out.append({"rule": rule, "label": C.RULE_LABELS[rule], "instances": int(len(h)),
                    "accounts": int(h.account_id.nunique()), "true_instances": int(len(true_hits)),
                    "precision": float(len(true_hits) / len(h)) if len(h) else 0.0,
                    "target_typologies": typs, "networks_in_scope": int(len(pool)),
                    "networks_touched": int(pool.network_id.isin(found).sum()),
                    "network_coverage": float(pool.network_id.isin(found).mean()) if len(pool) else 0.0})
    return out


if __name__ == "__main__":
    h = run_rules()
    print(pd.DataFrame(evaluate_rules(h, C.DATA_DIR)).drop(columns=["label", "target_typologies"]).to_string())
