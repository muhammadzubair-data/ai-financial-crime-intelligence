import pandas as pd

from fincrime import config as C
from fincrime.data.generate import Generator


def test_scale_and_coverage(run, tx):
    s = run["summary"]
    scale = C.SCALES["small"]
    assert s["n_accounts"] == scale.n_accounts
    assert s["n_networks"] == scale.n_networks
    assert tx.ts.min() >= C.START and tx.ts.max() < C.END
    assert tx.ts.dt.month.nunique() == 12
    assert tx.txn_id.is_unique and tx.ts.is_monotonic_increasing


def test_generation_is_reproducible():
    a = Generator(C.SCALES["small"], seed=7).run()[1]
    b = Generator(C.SCALES["small"], seed=7).run()[1]
    pd.testing.assert_frame_equal(a, b)


def test_labels_match_ground_truth_tables(data_dir, tx):
    nets = pd.read_parquet(data_dir / "networks.parquet")
    members = pd.read_parquet(data_dir / "network_members.parquet")
    sus = tx[tx.is_suspicious == 1]
    assert set(sus.network_id) <= set(nets.network_id)
    assert (tx[tx.is_suspicious == 0].network_id == -1).all()
    assert set(nets.typology) == set(C.TYPOLOGY_LABELS)
    assert not members.account_id.duplicated().any(), "an account belongs to at most one network"
    # every suspicious transaction involves at least one member of its network
    m = members.groupby("network_id").account_id.agg(set)
    ok = [bool({r.src_account, r.dst_account} & m[r.network_id]) for r in sus.itertuples()]
    assert all(ok)


def test_amounts_and_parties_are_valid(tx):
    assert (tx.amount > 0).all()
    assert ((tx.src_account >= 0) | (tx.dst_account >= 0)).all(), "every transaction touches a customer account"
    assert (tx.src_account != tx.dst_account).all()
    assert set(tx.ext_country) <= {C.DOMESTIC, *C.FOREIGN}


def test_no_transfer_before_account_opening(data_dir, tx):
    opened = pd.read_parquet(data_dir / "accounts.parquet").set_index("account_id").open_date
    internal = tx[tx.dst_account >= 0]
    assert (internal.ts >= internal.dst_account.map(opened)).all()
