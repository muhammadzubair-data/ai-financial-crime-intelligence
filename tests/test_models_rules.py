import json

import lightgbm as lgb
import pandas as pd

from fincrime import config as C
from fincrime.aml.rules import circular
from fincrime.models.train import pick_threshold, ranking_metrics


def test_model_beats_every_baseline(run):
    m = run["metrics"]["transaction_model"]["models"]
    assert m["lightgbm"]["pr_auc"] > 0.6
    for base in ("isolation_forest", "logistic_baseline", "rules_baseline"):
        assert m["lightgbm"]["pr_auc"] > m[base]["pr_auc"]


def test_split_is_out_of_time(run, data_dir):
    sp = run["metrics"]["transaction_model"]["split"]
    assert sp["train"][1] < sp["valid"][0] <= sp["valid"][1] < sp["test"][0]
    scores = pd.read_parquet(data_dir / "scores.parquet")
    assert scores.score.between(0, 1).all() and scores.anomaly_pct.between(0, 100).all()


def test_saved_model_reproduces_scores(run, data_dir):
    meta = json.loads((C.MODEL_DIR / "feature_meta.json").read_text())
    booster = lgb.Booster(model_file=str(C.MODEL_DIR / "lightgbm_txn_risk.txt"))
    f = pd.read_parquet(data_dir / "features.parquet").tail(500)
    for c, cats in meta["categorical"].items():
        f[c] = pd.Categorical(f[c], categories=cats)
    saved = pd.read_parquet(data_dir / "scores.parquet").set_index("txn_id").score.loc[f.txn_id]
    assert abs(booster.predict(f[meta["features"]]) - saved.to_numpy()).max() < 1e-6


def test_metric_helpers():
    import numpy as np
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .1])
    assert pick_threshold(y, s, 1.0) == 0.9
    m = ranking_metrics(y, s, n_days=1)
    assert 0.8 < m["pr_auc"] < 0.9 and m["roc_auc"] > 0.9


def test_rule_hits_respect_thresholds(artifacts, tx):
    hits = artifacts["rule_hits"]
    assert set(hits.rule) <= set(C.RULE_LABELS)
    amounts = tx.set_index("txn_id").amount
    types = tx.set_index("txn_id").txn_type
    st = hits[hits.rule == "STRUCTURING"]
    assert (st.n_txns >= C.STRUCT_MIN_DEPOSITS).all() and (st.amount >= C.STRUCT_MIN_TOTAL).all()
    for ids in st.txn_ids:
        assert amounts.loc[list(ids)].between(C.STRUCT_LOW, C.CASH_REPORT_THRESHOLD, inclusive="left").all()
        assert (types.loc[list(ids)] == "cash_deposit").all()


def test_circular_hits_are_real_cycles(artifacts, tx):
    hits = artifacts["rule_hits"]
    t = tx.set_index("txn_id")
    cyc = hits[hits.rule == "CIRCULAR"]
    assert len(cyc) > 0
    for ids in cyc.txn_ids.head(40):
        hop = t.loc[list(ids)]
        assert hop.ts.is_monotonic_increasing and len(hop) >= 3
        assert list(hop.dst_account[:-1]) == list(hop.src_account[1:]), "each hop starts where the last one ended"
        assert hop.dst_account.iloc[-1] == hop.src_account.iloc[0], "the loop closes"


def test_circular_finds_a_planted_loop_and_ignores_a_chain():
    import duckdb
    rows = [(1, "2025-03-01 10:00", 1, 2, 5000.0), (2, "2025-03-01 15:00", 2, 3, 4900.0), (3, "2025-03-02 09:00", 3, 1, 4800.0),
            (4, "2025-03-05 10:00", 7, 8, 5000.0), (5, "2025-03-05 12:00", 8, 9, 4900.0)]
    tx = pd.DataFrame(rows, columns=["txn_id", "ts", "src_account", "dst_account", "amount"])
    tx["ts"], tx["txn_type"] = pd.to_datetime(tx.ts), "transfer"
    con = duckdb.connect()
    con.register("tx", tx)
    hits = circular(con)
    assert set(hits.account_id) == {1, 2, 3} and hits.txn_ids.map(sorted).iloc[0] == [1, 2, 3]
