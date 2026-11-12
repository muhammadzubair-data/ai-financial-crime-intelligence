"""Transaction risk models and their evaluation.

Models
    lightgbm            gradient-boosted trees on all behavioural features (the production scorer)
    isolation_forest    unsupervised anomaly score, no labels used
    logistic_baseline   logistic regression on a handful of raw transaction fields
    rules_baseline      the kind of static rule a bank starts with: large amounts and high-risk corridors

Split (out of time)
    train Jan-Jul, validation Aug-Sep, monitoring window Oct-Dec.
    All reported numbers are for the monitoring window, which no model saw while fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as C
from ..features.build import CATEGORICAL, FEATURES, NUMERIC

ALERT_BUDGETS = [25, 50, 100, 200]          # transaction alerts per day an investigations team might handle
BASE_COLS = ["log_amount", "hour", "is_night", "is_weekend", "is_international", "is_high_risk_geo",
             "just_below_threshold", "is_round"]


def load_features(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(data_dir / "features.parquet")
    cats = {}
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
        cats[c] = [str(v) for v in df[c].cat.categories]
    df["split"] = np.where(df.ts < C.TRAIN_END, "train", np.where(df.ts < C.VALID_END, "valid", "test"))
    return df, cats


def rules_baseline_score(df: pd.DataFrame) -> np.ndarray:
    """Static rule: 10k+ transactions, 3k+ to a high-risk jurisdiction, or cash just under the threshold."""
    flag = ((df.amount >= C.CASH_REPORT_THRESHOLD)
            | ((df.is_high_risk_geo == 1) & (df.amount >= 3_000))
            | ((df.txn_type == "cash_deposit") & (df.just_below_threshold == 1)))
    return flag.to_numpy().astype(float) + df.log_amount.to_numpy() / 100.0   # amount only breaks ties


def _base_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[BASE_COLS].astype(float).copy()
    for t in ["transfer", "wire_out", "wire_in", "cash_deposit", "cash_withdrawal"]:
        x[f"type_{t}"] = (df.txn_type == t).astype(float)
    return x


def ranking_metrics(y: np.ndarray, score: np.ndarray, n_days: int) -> dict:
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    out = {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score)),
           "budgets": {}}
    for per_day in ALERT_BUDGETS:
        k = min(per_day * n_days, len(y))
        out["budgets"][str(per_day)] = {"alerts": int(k), "precision": float(hits[k - 1] / k),
                                        "recall": float(hits[k - 1] / max(y.sum(), 1))}
    return out


def threshold_metrics(y, score, thr, n_days) -> dict:
    pred = score >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": float(thr), "alerts": int(pred.sum()), "alerts_per_day": round(float(pred.sum() / n_days), 1),
            "true_positives": tp, "false_positives": fp, "missed": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9)}


def pick_threshold(y, score, target_precision) -> float:
    """Lowest score whose precision on validation reaches the target; best F1 if the target is unreachable."""
    p, r, t = precision_recall_curve(y, score)
    ok = np.flatnonzero(p[:-1] >= target_precision)
    if len(ok):
        return float(t[ok[0]])
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    return float(t[int(np.argmax(f1))])


def workload_curve(y, score, n_days, points=60) -> list[dict]:
    qs = np.unique(np.quantile(score, 1 - np.geomspace(1e-4, 0.2, points)))
    return [{k: v for k, v in threshold_metrics(y, score, q, n_days).items()
             if k in ("threshold", "alerts_per_day", "precision", "recall")} for q in qs[::-1]]


def train(data_dir: Path | None = None, artifact_dir: Path | None = None, seed: int = C.SEED) -> dict:
    data_dir, artifact_dir = Path(data_dir or C.DATA_DIR), Path(artifact_dir or C.ARTIFACT_DIR)
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    df, cats = load_features(data_dir)
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    y_tr, y_va, y_te = (d.is_suspicious.to_numpy() for d in (tr, va, te))
    n_days = max(int((te.ts.max() - te.ts.min()).days) + 1, 1)
    rng = np.random.default_rng(seed)

    # ---------------- LightGBM
    model = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(tr[FEATURES], y_tr, eval_X=va[FEATURES], eval_y=y_va, eval_metric="average_precision",
              callbacks=[lgb.early_stopping(60, verbose=False)])
    score = model.predict_proba(df[FEATURES])[:, 1]
    df["score"] = score
    thr = pick_threshold(y_va, score[(df.split == "valid").to_numpy()], C.TXN_TARGET_PRECISION)

    # ---------------- baselines
    sample = np.concatenate([np.flatnonzero(y_tr == 1),
                             rng.choice(np.flatnonzero(y_tr == 0), min(300_000, int((y_tr == 0).sum())), replace=False)])
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
    logit.fit(_base_matrix(tr.iloc[sample]), y_tr[sample])
    iso_cols = NUMERIC
    fit_rows = rng.choice(len(tr), min(200_000, len(tr)), replace=False)
    iso = IsolationForest(n_estimators=200, max_samples=512, random_state=seed, n_jobs=-1)
    iso.fit(tr.iloc[fit_rows][iso_cols].fillna(-1).astype("float32"))
    anomaly = -iso.score_samples(df[iso_cols].fillna(-1).astype("float32"))
    # express the anomaly score as a percentile of the training distribution (0-100, higher = more unusual)
    ref = np.sort(anomaly[(df.split == "train").to_numpy()])
    df["anomaly_pct"] = np.searchsorted(ref, anomaly) / len(ref) * 100

    te_mask = (df.split == "test").to_numpy()
    scores_te = {"lightgbm": score[te_mask], "isolation_forest": anomaly[te_mask],
                 "logistic_baseline": logit.predict_proba(_base_matrix(te))[:, 1],
                 "rules_baseline": rules_baseline_score(te)}
    metrics = {name: ranking_metrics(y_te, s, n_days) for name, s in scores_te.items()}
    metrics["lightgbm"]["at_threshold"] = threshold_metrics(y_te, scores_te["lightgbm"], thr, n_days)
    rule_flag = scores_te["rules_baseline"] >= 1
    metrics["rules_baseline"]["at_threshold"] = threshold_metrics(y_te, rule_flag.astype(float), 0.5, n_days)

    # recall of the operating threshold by typology and for stealth networks
    te2 = df.loc[te_mask, ["typology", "network_id", "score"]].assign(hit=lambda d: d.score >= thr)
    nets = pd.read_parquet(data_dir / "networks.parquet").set_index("network_id")
    sus = te2[te2.network_id >= 0].assign(stealth=lambda d: d.network_id.map(nets.stealth))
    by_typ = sus.groupby("typology").hit.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "recall"})
    by_stealth = sus.groupby("stealth").hit.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "recall"})

    # ---------------- explanations (SHAP)
    from .explain import explain_transactions
    flagged = df[te_mask & (score >= thr)]
    explain_rows = flagged if len(flagged) <= 40_000 else flagged.nlargest(40_000, "score")
    background = df[te_mask].sample(min(5_000, int(te_mask.sum())), random_state=seed)
    shap_txn, shap_global, method = explain_transactions(model, explain_rows, background)
    shap_txn.to_parquet(artifact_dir / "shap_txn.parquet", index=False)
    shap_global.to_parquet(artifact_dir / "shap_global.parquet", index=False)

    # ---------------- persist
    model.booster_.save_model(str(model_dir / "lightgbm_txn_risk.txt"))
    joblib.dump(iso, model_dir / "isolation_forest.joblib", compress=3)
    joblib.dump(logit, model_dir / "logistic_baseline.joblib")
    (model_dir / "feature_meta.json").write_text(json.dumps(
        {"features": FEATURES, "categorical": cats, "isolation_forest_features": iso_cols, "threshold": thr}, indent=1))
    df[["txn_id", "split", "score", "anomaly_pct"]].to_parquet(data_dir / "scores.parquet", index=False)

    out = {
        "split": {"train": [str(tr.ts.min().date()), str(tr.ts.max().date()), len(tr), int(y_tr.sum())],
                  "valid": [str(va.ts.min().date()), str(va.ts.max().date()), len(va), int(y_va.sum())],
                  "test": [str(te.ts.min().date()), str(te.ts.max().date()), len(te), int(y_te.sum())]},
        "window_days": n_days, "base_rate_test": float(y_te.mean()), "threshold": thr,
        "target_precision": C.TXN_TARGET_PRECISION, "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "models": metrics, "recall_by_typology": by_typ.reset_index().to_dict("records"),
        "recall_by_stealth": by_stealth.reset_index().to_dict("records"),
        "workload_curve": workload_curve(y_te, scores_te["lightgbm"], n_days), "explain_method": method,
    }
    return out


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: v for k, v in m.items() if k != "workload_curve"}, indent=1))
