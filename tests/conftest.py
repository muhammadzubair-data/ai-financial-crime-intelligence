"""Runs the whole pipeline once at the small scale; every test reads from that run."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="fincrime_test_"))
# set before fincrime.config is imported, so the tests never touch the real data or artifacts
os.environ["FINCRIME_DATA_DIR"] = str(_TMP / "data")
os.environ["FINCRIME_ARTIFACT_DIR"] = str(_TMP / "artifacts")
os.environ["FINCRIME_CASE_DB"] = str(_TMP / "cases.db")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

import warnings  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def run():
    from fincrime.pipeline import run as run_pipeline
    return run_pipeline("small", verbose=False)


@pytest.fixture(scope="session")
def data_dir(run):
    return _TMP / "data"


@pytest.fixture(scope="session")
def artifacts(run):
    d = _TMP / "artifacts"
    return {p.stem: pd.read_parquet(p) for p in d.glob("*.parquet")}


@pytest.fixture(scope="session")
def tx(data_dir):
    return pd.read_parquet(data_dir / "transactions.parquet")
