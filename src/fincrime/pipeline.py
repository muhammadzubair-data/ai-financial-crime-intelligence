"""End-to-end pipeline: generate -> features -> models -> AML rules -> alerts and networks -> app artifacts.

    python -m fincrime.pipeline --scale full      # 100k accounts, 1.6M transactions (about 6 minutes on 2 cores)
    python -m fincrime.pipeline --scale small     # 3k accounts, used by the tests and CI
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import alerts as A
from . import config as C
from .aml.rules import evaluate_rules, run_rules
from .data.generate import generate
from .features.build import build_features
from .graph.network import evaluate_networks
from .models.train import train


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


GENERIC_DRIVERS = {"txn_type", "channel", "hour", "dow", "log_amount", "amount", "s_type", "d_type", "is_weekend"}


def _reasons(alerts: pd.DataFrame, hits: pd.DataFrame, nets: pd.DataFrame, shap_txn: pd.DataFrame, tx: pd.DataFrame) -> pd.Series:
    """Up to six plain-language reasons per alerted account: rule hits, network membership, model drivers."""
    out = {a: [] for a in alerts.index}
    for a, rule, detail, s, e in hits[hits.account_id.isin(out)].sort_values("amount", ascending=False)[
            ["account_id", "rule", "detail", "start_ts", "end_ts"]].itertuples(index=False):
        label = C.RULE_LABELS[rule].split(" (")[0]
        if sum(r.startswith(label) for r in out[a]) == 0:
            when = f"{s:%d %b}" if s.date() == e.date() else f"{s:%d %b} to {e:%d %b}"
            out[a].append(f"{label}: {detail} ({when})")
    net = nets.set_index("detected_network_id") if len(nets) else None
    for a, nid in alerts.detected_network_id.items():
        if nid > 0:
            out[a].append(f"Linked to detected network N-{nid:03d} ({net.n_accounts[nid]} accounts, {net.inferred_type[nid].lower()})")
    if len(shap_txn):
        sides = A._sides(tx[tx.txn_id.isin(shap_txn.txn_id)], ["txn_id"])
        sides = sides[sides.account_id.isin(out)].merge(shap_txn, on="txn_id")
        # leave out drivers that say little on their own, and keep each remaining driver once per account
        sides = sides[~sides.feature.isin(GENERIC_DRIVERS)]
        sides = sides.assign(topic=sides.feature.where(~sides.feature.str.contains("dev"), "device"))
        sides = sides.sort_values("shap", ascending=False).drop_duplicates(["account_id", "topic"])
        for a, grp in sides.groupby("account_id"):
            out[a] += list(grp.reason.head(3))
    return pd.Series({a: json.dumps(r[:6]) for a, r in out.items()})


def run(scale: str = "full", data_dir: Path | None = None, artifact_dir: Path | None = None, verbose: bool = True,
        reuse: bool = False) -> dict:
    """reuse=True skips generation, features and training when a previous run of the same scale is on disk."""
    data_dir, artifact_dir = Path(data_dir or C.DATA_DIR), Path(artifact_dir or C.ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    cache = data_dir / "stage_cache.json"
    if reuse and cache.exists() and json.loads(cache.read_text())["info"]["scale"] == scale:
        cached = json.loads(cache.read_text())
        info, model_metrics = cached["info"], cached["model_metrics"]
        log("reusing generated data, features and trained models")
    else:
        info = generate(scale, data_dir)
        log(f"generated {info['n_transactions']:,} transactions for {info['n_accounts']:,} accounts, {info['n_networks']} networks")
        build_features(data_dir)
        log("features built")
        model_metrics = train(data_dir, artifact_dir)
        log(f"models trained (LightGBM PR-AUC {model_metrics['models']['lightgbm']['pr_auc']:.3f})")
        cache.write_text(json.dumps({"info": info, "model_metrics": model_metrics}, default=_json_default))
    thr = model_metrics["threshold"]
    hits_all = run_rules(data_dir)
    rule_eval = evaluate_rules(hits_all, data_dir)
    log(f"AML rules: {len(hits_all):,} pattern instances")

    tx_all = A.load_scored(data_dir)
    accounts = pd.read_parquet(data_dir / "accounts.parquet")
    true_members = pd.read_parquet(data_dir / "network_members.parquet")
    true_nets = pd.read_parquet(data_dir / "networks.parquet")

    # alert threshold from the validation window, applied unchanged to the monitoring window
    v_acc, *_rest, v_tx, _ = A.score_window(tx_all, hits_all, thr, C.TRAIN_END, C.VALID_END)
    alert_thr = A.choose_alert_threshold(v_acc, A.window_truth(v_tx, true_members))
    acc, nets, mem, edges, flagged, tx, hits = A.score_window(tx_all, hits_all, thr, C.VALID_END, C.END)
    truth = A.window_truth(tx, true_members)
    n_days = model_metrics["window_days"]
    acc["alert"] = acc.risk >= alert_thr
    acc["priority"] = np.where(acc.alert, A.priority(acc.risk), "")
    alerted = set(acc.index[acc.alert])
    alert_ts = A.alert_times(acc, tx, hits, alert_thr)
    log(f"account scoring: {len(alerted):,} alerts at threshold {alert_thr:.2f}, {len(nets)} networks detected")

    # ------------------------------------------------------------------ evaluation
    y = acc.index.isin(truth)
    tp = int((acc.alert & y).sum())
    account_eval = {
        "alert_threshold": alert_thr, "alerts": len(alerted), "alerts_per_week": round(len(alerted) / (n_days / 7), 1),
        "suspicious_accounts_active": len(truth), "true_positives": tp,
        "precision": tp / max(len(alerted), 1), "recall": tp / max(len(truth), 1),
        "pr_auc_hybrid": float(average_precision_score(y, acc.risk)), "pr_auc_model_only": float(average_precision_score(y, acc.ml)),
        "pr_auc_rules_only": float(average_precision_score(y, acc.rules)),
        "by_priority": acc[acc.alert].assign(true=y[acc.alert.to_numpy()]).groupby("priority").true.agg(["size", "mean"])
        .rename(columns={"size": "alerts", "mean": "precision"}).reset_index().to_dict("records"),
    }
    net_summary, per_true, quality = evaluate_networks(nets, mem, tx, true_members, true_nets, alerted)
    metrics = {
        "transaction_model": model_metrics, "rules": rule_eval, "accounts": account_eval, "networks": net_summary,
        "effectiveness": A.effectiveness(acc, tx, truth, true_members, n_days),
        "time_to_detect": A.time_to_detect(tx, alert_ts, true_members, true_nets, C.VALID_END),
    }

    # ------------------------------------------------------------------ artifacts for the app
    if not (data_dir / "shap_txn_all.parquet").exists() or not reuse:
        pd.read_parquet(artifact_dir / "shap_txn.parquet").to_parquet(data_dir / "shap_txn_all.parquet", index=False)
    shap_txn = pd.read_parquet(data_dir / "shap_txn_all.parquet")
    tm = true_members.merge(true_nets[["network_id", "typology", "stealth"]], on="network_id")
    acct_out = accounts.merge(acc.drop(columns=["base"]), left_on="account_id", right_index=True, how="left")
    acct_out = acct_out.merge(tm.rename(columns={"network_id": "true_network_id", "role": "true_role",
                                                 "typology": "true_typology", "stealth": "true_stealth"}),
                              on="account_id", how="left")
    acct_out["true_network_id"] = acct_out.true_network_id.fillna(-1).astype(int)
    acct_out["true_active_in_window"] = acct_out.account_id.isin(truth)
    for c, v in {"ml": 0.0, "rules": 0.0, "network": 0.0, "risk": 0.0, "n_txns": 0, "n_flagged": 0, "exposure": 0.0,
                 "detected_network_id": 0, "alert": False, "priority": ""}.items():
        acct_out[c] = acct_out[c].fillna(v)
    acct_out["rule_list"] = [", ".join(r) if isinstance(r, (list, np.ndarray)) else "" for r in acct_out.rule_list]
    acct_out = acct_out.astype({"n_txns": int, "n_flagged": int, "detected_network_id": int, "alert": bool})
    acct_out.to_parquet(artifact_dir / "accounts.parquet", index=False)

    al = acc[acc.alert].sort_values("risk", ascending=False).copy()
    fl_tx = tx[tx.txn_id.isin(flagged)]
    first = A._sides(fl_tx, ["ts"]).groupby("account_id").ts.agg(["min", "max"])
    al["first_flag_ts"], al["last_flag_ts"] = first["min"].reindex(al.index), first["max"].reindex(al.index)
    al["alert_ts"] = alert_ts.reindex(al.index)
    al["reasons"] = _reasons(al, hits, nets, shap_txn, tx)
    al["rule_list"] = [", ".join(r) for r in al.rule_list]
    al.insert(0, "alert_id", [f"ALT-{i + 1:05d}" for i in range(len(al))])
    al.reset_index().drop(columns=["base"]).to_parquet(artifact_dir / "alerts.parquet", index=False)

    interest = alerted | set(mem.account_id) | set(acc.index[acc.risk >= alert_thr / 2])
    keep = tx_all[tx_all.src_account.isin(interest) | tx_all.dst_account.isin(interest)].copy()
    keep["in_window"] = keep.ts >= C.VALID_END
    keep["flagged"] = keep.txn_id.isin(flagged)
    keep.loc[~keep.in_window, "score"] = np.nan      # scores before the window are in-sample, so they are not shown
    keep.drop(columns=["split"]).to_parquet(artifact_dir / "transactions.parquet", index=False)

    d = tx_all.assign(date=tx_all.ts.dt.floor("D"), flagged=tx_all.txn_id.isin(flagged))
    daily = d.groupby("date").agg(n_txns=("amount", "size"), volume=("amount", "sum"), n_flagged=("flagged", "sum"),
                                  n_suspicious=("is_suspicious", "sum")).reset_index()
    daily["flagged_amount"] = d[d.flagged].groupby("date").amount.sum().reindex(daily.date).fillna(0).to_numpy()
    daily["suspicious_amount"] = d[d.is_suspicious == 1].groupby("date").amount.sum().reindex(daily.date).fillna(0).to_numpy()
    daily["in_window"] = daily.date >= C.VALID_END
    daily.to_parquet(artifact_dir / "daily.parquet", index=False)
    w = d[d.ts >= C.VALID_END]
    (w.groupby("txn_type").agg(n_txns=("amount", "size"), volume=("amount", "sum"), n_flagged=("flagged", "sum"))
     .reset_index().assign(flagged_amount=lambda t: t.txn_type.map(w[w.flagged].groupby("txn_type").amount.sum()).fillna(0))
     .to_parquet(artifact_dir / "type_mix.parquet", index=False))

    hits_all.assign(in_window=hits_all.end_ts >= C.VALID_END).to_parquet(artifact_dir / "rule_hits.parquet", index=False)
    nets = nets.merge(quality, on="detected_network_id", how="left") if len(nets) else nets.assign(purity=[], true_network_id=[])
    nets["true_typology"] = nets.true_network_id.map(true_nets.set_index("network_id").typology).fillna("none")
    nets["patterns"] = [", ".join(p) for p in nets.patterns]
    nets.to_parquet(artifact_dir / "networks.parquet", index=False)
    mem.to_parquet(artifact_dir / "network_members.parquet", index=False)
    edges.to_parquet(artifact_dir / "network_edges.parquet", index=False)
    per_true.to_parquet(artifact_dir / "true_networks_window.parquet", index=False)
    shap_txn[shap_txn.txn_id.isin(keep.txn_id)].to_parquet(artifact_dir / "shap_txn.parquet", index=False)

    summary = {
        **info, "synthetic": True, "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "period": [str(C.START.date()), str((C.END - pd.Timedelta(days=1)).date())],
        "train_period": [str(C.START.date()), str((C.TRAIN_END - pd.Timedelta(days=1)).date())],
        "validation_period": [str(C.TRAIN_END.date()), str((C.VALID_END - pd.Timedelta(days=1)).date())],
        "monitoring_window": [str(C.VALID_END.date()), str((C.END - pd.Timedelta(days=1)).date())],
        "window_days": n_days, "window_transactions": int(len(tx)), "window_volume": float(tx.amount.sum()),
        "txn_threshold": thr, "alert_threshold": alert_thr,
        "flagged_transactions": int(len(flagged)), "flagged_amount": float(fl_tx.amount.sum()),
        "alerts": len(alerted), "alert_exposure": float(fl_tx[fl_tx.src_account.isin(alerted) | fl_tx.dst_account.isin(alerted)].amount.sum()),
        "detected_networks": int(len(nets)), "networks_by_typology": true_nets.typology.value_counts().to_dict(),
        "stealth_networks": int(true_nets.stealth.sum()), "runtime_seconds": round(time.time() - t0, 1),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_json_default))
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    log("artifacts written")
    return {"summary": summary, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=list(C.SCALES), default="full")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--reuse", action="store_true", help="skip generation, features and training if already on disk")
    a = ap.parse_args()
    out = run(a.scale, a.data_dir, a.artifact_dir, reuse=a.reuse)
    m, s = out["metrics"], out["summary"]
    print(json.dumps({"alerts": s["alerts"], "alert_threshold": s["alert_threshold"], "accounts": {k: v for k, v in m["accounts"].items() if k != "by_priority"},
                      "networks": {k: v for k, v in m["networks"].items() if not k.startswith("by_")},
                      "time_to_detect": m["time_to_detect"]}, indent=1, default=_json_default))


if __name__ == "__main__":
    main()
