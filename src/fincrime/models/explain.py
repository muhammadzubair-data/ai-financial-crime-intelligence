"""SHAP explanations for the transaction risk model, and plain-language reason text."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build import CATEGORICAL, FEATURES

TOP_K = 6

# feature -> (label, sentence template). {v} is the feature value, already formatted.
FEATURE_TEXT = {
    "amount": ("Transaction amount", "Amount of ${v}"),
    "log_amount": ("Transaction amount", "Size of the transaction"),
    "hour": ("Hour of day", "Made at {v}:00"),
    "dow": ("Day of week", "Day of the week"),
    "is_night": ("Night-time", "Made at night"),
    "is_weekend": ("Weekend", "Made at the weekend"),
    "is_international": ("International", "International counterparty"),
    "is_high_risk_geo": ("High-risk jurisdiction", "Counterparty in a high-risk jurisdiction"),
    "just_below_threshold": ("Just under reporting threshold", "Amount sits just under the 10,000 reporting threshold"),
    "is_round": ("Round amount", "Round amount"),
    "pair_prior": ("Earlier payments between this pair", "{v} earlier payments between these two parties"),
    "dev_accounts": ("Accounts on this device", "Device has been used by {v} different accounts"),
    "s_new_device": ("New device for sender", "Sender used a device not seen on the account before"),
    "s_passthrough": ("Sender: outflow vs 24h inflow", "Sender is passing on {v}x of what it received in the last 24 hours"),
    "s_out_in_72h": ("Sender: 72h outflow vs inflow", "Sender has moved out {v}x of its 72-hour inflow"),
    "s_amount_z": ("Sender: amount vs own history", "Amount is {v} standard deviations from the sender's 30-day norm"),
    "s_amount_to_income": ("Sender: amount vs declared income", "Amount is {v}x the sender's declared monthly income"),
    "s_age_days": ("Sender: account age (days)", "Sender account is {v} days old"),
    "d_age_days": ("Receiver: account age (days)", "Receiving account is {v} days old"),
    "d_amount_z": ("Receiver: amount vs own history", "Amount is {v} standard deviations from the receiver's 30-day norm"),
    "d_amount_to_income": ("Receiver: amount vs declared income", "Amount is {v}x the receiver's declared monthly income"),
    "txn_type": ("Transaction type", "Transaction type: {v}"),
    "channel": ("Channel", "Channel: {v}"),
    "s_type": ("Sender customer type", "Sender is a {v} customer"),
    "d_type": ("Receiver customer type", "Receiver is a {v} customer"),
    "s_kyc": ("Sender KYC risk rating", "Sender KYC rating: {v}"),
    "d_kyc": ("Receiver KYC risk rating", "Receiver KYC rating: {v}"),
}
_SIDE = {
    "n_24h": ("transactions in last 24h", "{who} had {v} other transactions in the last 24 hours"),
    "n_7d": ("transactions in last 7 days", "{who} had {v} other transactions in the last 7 days"),
    "cr_sum_24h": ("money in, last 24h", "{who} received ${v} in the last 24 hours"),
    "db_sum_24h": ("money out, last 24h", "{who} sent ${v} in the last 24 hours"),
    "cr_sum_72h": ("money in, last 72h", "{who} received ${v} in the last 72 hours"),
    "db_sum_72h": ("money out, last 72h", "{who} sent ${v} in the last 72 hours"),
    "cr_cnt_7d": ("credits in last 7 days", "{who} received {v} credits in the last 7 days"),
    "db_cnt_7d": ("debits in last 7 days", "{who} made {v} debits in the last 7 days"),
    "senders_7d": ("distinct senders, 7 days", "{who} was paid by {v} different parties in the last 7 days"),
    "receivers_7d": ("distinct payees, 7 days", "{who} paid {v} different parties in the last 7 days"),
    "sub10k_cash_7d": ("cash deposits just under 10k, 7 days", "{who} made {v} cash deposits just under 10,000 in the last 7 days"),
    "avg_30d": ("average amount, 30 days", "{who}'s average transaction over 30 days is ${v}"),
    "std_30d": ("amount variability, 30 days", "{who}'s 30-day amount variability"),
    "n_30d": ("transactions in last 30 days", "{who} had {v} transactions in the last 30 days"),
    "secs_since_prev": ("time since previous transaction", "{who}'s previous transaction was {v} earlier"),
    "secs_since_credit": ("time since last money in", "{who} last received money {v} earlier"),
    "n_prior": ("transactions on record", "{who} has {v} earlier transactions on record"),
    "dev_share": ("accounts sharing a device", "{who} has used a device shared by {v} accounts"),
}
for _p, _who in (("s", "Sender"), ("d", "Receiver")):
    for _k, (_lab, _txt) in _SIDE.items():
        FEATURE_TEXT[f"{_p}_{_k}"] = (f"{_who}: {_lab}", _txt.replace("{who}", _who))


def feature_label(feature: str) -> str:
    return FEATURE_TEXT.get(feature, (feature, ""))[0]


def _fmt(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, str):
        return value.replace("_", " ")
    if feature.endswith(("secs_since_prev", "secs_since_credit")):
        h = float(value) / 3600
        return f"{h * 60:.0f} minutes" if h < 1 else (f"{h:.1f} hours" if h < 48 else f"{h / 24:.0f} days")
    if "sum" in feature or "avg" in feature or feature == "amount":
        return f"{float(value):,.0f}"
    v = float(value)
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"


def reason_text(feature: str, value) -> str:
    """One plain-language sentence for a feature that pushed a score up."""
    if feature == "pair_prior" and value == 0:
        return "First payment between these two parties"
    template = FEATURE_TEXT.get(feature, (feature, feature))[1]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return feature_label(feature) + " (no history)"
    return template.replace("{v}", _fmt(feature, value))


def explain_transactions(model, rows: pd.DataFrame, background: pd.DataFrame):
    """Return (per-transaction top contributions, global importance, method used).

    Uses the SHAP library's TreeExplainer. If SHAP is unavailable, falls back to LightGBM's own
    ``pred_contrib`` output, which is the same TreeSHAP algorithm.
    """
    def contributions(x: pd.DataFrame) -> np.ndarray:
        if len(x) == 0:
            return np.zeros((0, len(FEATURES)))
        try:
            import shap
            vals = shap.TreeExplainer(model).shap_values(x[FEATURES])
            if isinstance(vals, list):
                vals = vals[-1]
            vals = np.asarray(vals)
            return vals[:, :, -1] if vals.ndim == 3 else vals
        except Exception:
            return model.booster_.predict(x[FEATURES], pred_contrib=True)[:, :-1]

    try:
        import shap  # noqa: F401
        method = "shap.TreeExplainer"
    except Exception:
        method = "lightgbm pred_contrib (TreeSHAP)"

    sv = contributions(rows)
    recs = []
    if len(rows):
        top = np.argsort(-sv, axis=1)[:, :TOP_K]
        ids = rows.txn_id.to_numpy()
        cols = {f: rows[f].astype(object).to_numpy() if f in CATEGORICAL else rows[f].to_numpy() for f in FEATURES}
        for i in range(len(rows)):
            for rank, j in enumerate(top[i]):
                if sv[i, j] <= 0:
                    break
                f = FEATURES[j]
                v = cols[f][i]
                recs.append((ids[i], rank + 1, f, None if isinstance(v, str) else float(v), v if isinstance(v, str) else None,
                             float(sv[i, j]), reason_text(f, v)))
    shap_txn = pd.DataFrame(recs, columns=["txn_id", "rank", "feature", "value", "value_text", "shap", "reason"])
    g = np.abs(contributions(background)).mean(axis=0) if len(background) else np.zeros(len(FEATURES))
    shap_global = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": g})
                   .assign(label=lambda d: d.feature.map(feature_label))
                   .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return shap_txn, shap_global, method
