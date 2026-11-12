import numpy as np
import pandas as pd

from fincrime.features.build import FEATURES


def test_one_row_per_transaction_with_all_features(data_dir, tx):
    f = pd.read_parquet(data_dir / "features.parquet")
    assert len(f) == len(tx) and f.txn_id.is_unique
    assert set(FEATURES) <= set(f.columns)


def test_features_only_look_backwards(data_dir, tx):
    """Recompute two rolling features by hand for a sample and compare: they must use earlier rows only."""
    f = pd.read_parquet(data_dir / "features.parquet").set_index("txn_id")
    rng = np.random.default_rng(0)
    senders = tx[tx.src_account >= 0]
    for txn_id in rng.choice(senders.txn_id, 60, replace=False):
        row = tx.loc[tx.txn_id == txn_id].iloc[0]
        a, t = row.src_account, row.ts
        hist = tx[((tx.src_account == a) | (tx.dst_account == a)) & (tx.ts <= t) & (tx.ts >= t - pd.Timedelta(hours=24))
                  & (tx.txn_id != txn_id)]
        credits = hist[hist.dst_account == a].amount.sum()
        assert f.s_n_24h[txn_id] == len(hist)
        assert abs(f.s_cr_sum_24h[txn_id] - credits) < 0.01


def test_first_transaction_has_no_history(data_dir):
    f = pd.read_parquet(data_dir / "features.parquet")
    first = f[f.s_n_prior == 0]
    assert (first.s_n_7d == 0).all() and first.s_secs_since_prev.isna().all()
