"""Central configuration: paths, scale presets, time windows and thresholds.

Everything a reviewer might want to change lives here, so no number is buried in the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FINCRIME_DATA_DIR", ROOT / "data"))            # full generated data (git-ignored)
ARTIFACT_DIR = Path(os.environ.get("FINCRIME_ARTIFACT_DIR", ROOT / "artifacts"))  # compact outputs the app reads
MODEL_DIR = ARTIFACT_DIR / "models"
CASE_DB = Path(os.environ.get("FINCRIME_CASE_DB", ARTIFACT_DIR / "cases.db"))


# --------------------------------------------------------------------------- scale presets
@dataclass(frozen=True)
class Scale:
    name: str
    n_accounts: int
    target_txns: int          # approximate number of legitimate transactions
    n_mule_rings: int
    n_circular: int
    n_structuring: int
    n_layering: int

    @property
    def n_networks(self) -> int:
        return self.n_mule_rings + self.n_circular + self.n_structuring + self.n_layering


SCALES = {
    # full  = the portfolio demonstration scale
    "full": Scale("full", 100_000, 1_120_000, 180, 120, 110, 110),
    # medium = quick local iteration (about a minute end to end)
    "medium": Scale("medium", 20_000, 224_000, 36, 24, 22, 22),
    # small = unit tests and CI
    "small": Scale("small", 3_000, 34_000, 14, 10, 8, 8),
}
SEED = 20250115

# --------------------------------------------------------------------------- calendar
START = pd.Timestamp("2025-01-01")
N_DAYS = 365
END = START + pd.Timedelta(days=N_DAYS)
# Out-of-time design: the model never sees the monitoring window during training.
TRAIN_END = pd.Timestamp("2025-08-01")     # train:      Jan - Jul
VALID_END = pd.Timestamp("2025-10-01")     # validation: Aug - Sep (early stopping, threshold choice)
# monitoring window (test): Oct - Dec. Every alert shown in the app comes from this window.

# --------------------------------------------------------------------------- jurisdictions
# Fictional codes (the ISO "user-assigned" X* range) so no real country is labelled high risk.
DOMESTIC = "DOM"
FOREIGN = [f"X{c}" for c in "ABCDEFGHIJKLMNOPQRST"]
HIGH_RISK = ["XQ", "XR", "XS", "XT"]

# --------------------------------------------------------------------------- AML rule thresholds
CASH_REPORT_THRESHOLD = 10_000        # reporting threshold that structuring tries to stay under
STRUCT_LOW = 7_000                    # "just under the threshold" band starts here
STRUCT_MIN_DEPOSITS = 3
STRUCT_MIN_TOTAL = 20_000
STRUCT_WINDOW_DAYS = 7
RAPID_MIN_AMOUNT = 3_000              # minimum outflow to consider
RAPID_WINDOW_HOURS = 24
RAPID_RATIO = (0.80, 1.05)            # outflow as a share of the inflow received in the window
FANIN_WINDOW_DAYS = 7
FANIN_MIN_SENDERS = {"retail": 4, "business": 15}   # first-time senders within the window
FANIN_MIN_TOTAL = 3_000
CIRC_MIN_AMOUNT = 1_000
CIRC_MAX_HOP_HOURS = 48
CIRC_MAX_TOTAL_DAYS = 7
CIRC_AMOUNT_BAND = (0.80, 1.02)       # each hop must carry 80-102% of the previous hop
CIRC_MAX_LEN = 6

# --------------------------------------------------------------------------- scoring
TXN_TARGET_PRECISION = 0.90           # transaction alert threshold is the lowest score reaching this on validation
ACCOUNT_WEIGHTS = {"ml": 0.55, "rules": 0.25, "network": 0.20}
RULE_WEIGHTS = {"CIRCULAR": 0.60, "STRUCTURING": 0.50, "RAPID_MOVEMENT": 0.45, "FAN_IN": 0.40}
PRIORITY_BANDS = [("Critical", 0.70), ("High", 0.50), ("Medium", 0.0)]   # minimum account risk for each band
NETWORK_MIN_SIZE = 3

RULE_LABELS = {
    "STRUCTURING": "Structuring (cash deposits just under the reporting threshold)",
    "RAPID_MOVEMENT": "Rapid movement of funds (in and out within 24 hours)",
    "CIRCULAR": "Circular transfers (funds return to the originator)",
    "FAN_IN": "Fan-in (many first-time senders in a short period)",
}
TYPOLOGY_LABELS = {
    "mule_ring": "Mule ring",
    "circular": "Circular transfers",
    "structuring": "Structuring",
    "layering": "Layering chain",
}


def acct_label(account_id: int) -> str:
    """Display form of an internal account id."""
    return f"AC-{int(account_id):06d}"
