import json

import pandas as pd

from fincrime import config as C
from fincrime.alerts import base_scores, priority
from fincrime.graph.network import detect_networks


def test_risk_formula_and_alert_threshold(run, artifacts):
    acc = artifacts["accounts"]
    w = C.ACCOUNT_WEIGHTS
    expected = w["ml"] * acc.ml + w["rules"] * acc.rules + w["network"] * acc.network
    assert (acc.risk - expected).abs().max() < 1e-9
    assert acc.risk.between(0, 1).all()
    thr = run["summary"]["alert_threshold"]
    assert (acc.alert == (acc.risk >= thr)).all()
    assert set(artifacts["alerts"].account_id) == set(acc.account_id[acc.alert])


def test_alerts_are_explained_and_ordered(artifacts):
    al = artifacts["alerts"]
    assert al.risk.is_monotonic_decreasing and al.alert_id.is_unique
    assert all(len(json.loads(r)) >= 1 for r in al.reasons), "every alert carries at least one reason"
    assert al.alert_ts.notna().all()
    assert set(al.priority) <= {"Critical", "High", "Medium"}


def test_alert_quality_on_ground_truth(run):
    a = run["metrics"]["accounts"]
    assert a["precision"] > 0.6 and a["recall"] > 0.4
    assert a["pr_auc_hybrid"] > a["pr_auc_rules_only"]


def test_priority_bands():
    p = priority(pd.Series([0.2, 0.55, 0.9]))
    assert list(p) == ["Medium", "High", "Critical"]


def test_networks_are_consistent(artifacts):
    nets, mem, edges = artifacts["networks"], artifacts["network_members"], artifacts["network_edges"]
    assert (nets.n_accounts >= C.NETWORK_MIN_SIZE).all()
    assert (mem.groupby("detected_network_id").size() == nets.set_index("detected_network_id").n_accounts).all()
    assert not mem.account_id.duplicated().any()
    members = mem.groupby("detected_network_id").account_id.agg(set)
    for e in edges.itertuples():
        assert {e.src, e.dst} <= members[e.detected_network_id]


def test_detects_a_planted_ring_and_leaves_a_victim_out():
    """Three suspects passing flagged money to each other form a network; a one-off payer does not join it."""
    ts = pd.Timestamp("2025-10-05")
    rows = [(1, 10, 11, 900.0, 0.9), (2, 11, 12, 850.0, 0.9), (3, 10, 12, 500.0, 0.9), (4, 11, 12, 300.0, 0.9),
            (5, 10, 11, 700.0, 0.9), (6, 99, 10, 400.0, 0.9), (7, 50, 51, 20.0, 0.01)]
    tx = pd.DataFrame(rows, columns=["txn_id", "src_account", "dst_account", "amount", "score"])
    tx["ts"], tx["device_id"] = ts, -1
    hits = pd.DataFrame(columns=["account_id", "rule", "end_ts", "txn_ids"])
    flagged = set(tx.txn_id[tx.score > 0.5])
    acc = base_scores(tx, hits, flagged)
    nets, mem, _ = detect_networks(tx, acc, flagged, hits, seed_threshold=0.30)
    assert len(nets) == 1 and set(mem.account_id) == {10, 11, 12}
    assert acc.base[99] < 0.30, "a single flagged payment is not enough to make an account a suspect"
