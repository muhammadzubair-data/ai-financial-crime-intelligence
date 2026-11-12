# Model card: transaction risk scorer

| | |
|---|---|
| Model | LightGBM binary classifier (`artifacts/models/lightgbm_txn_risk.txt`) |
| Output | Score between 0 and 1 per transaction; higher means more like the suspicious transactions seen in training |
| Trained on | Synthetic transactions, 1 January to 31 July 2025 (927,615 rows, 1.28% suspicious) |
| Validated on | 1 August to 30 September 2025 (283,381 rows) |
| Tested on | 1 October to 31 December 2025 (451,533 rows, 1.02% suspicious) |
| Features | 63, listed in `artifacts/models/feature_meta.json` |
| Operating threshold | 0.130 |

## Intended use

A demonstration of how a transaction-monitoring model fits into an investigation workflow. It ranks transactions and accounts for human review.

## Not intended for

Use on real customers. The model has only seen simulated behaviour. It must not be used to block payments, close accounts or report anyone.

## Performance on the test window

| Metric | Value |
|---|---|
| PR-AUC | 0.957 |
| ROC-AUC | 0.999 |
| Precision at threshold | 85% |
| Recall at threshold | 92% |
| Alerts per day at threshold | 54 |
| Recall, standard networks | 95% |
| Recall, stealth networks | 81% |

## Known weaknesses

- Suspicious activity follows four scripted typologies. Anything outside them is unlikely to be caught by this model; the Isolation Forest score is there as a partial safety net.
- Stealth networks are missed more often.
- Precision fell from 90% on validation to 85% on test at the same threshold, so thresholds need monitoring over time.
- Scores are not calibrated probabilities.
- No fairness analysis has been done. The synthetic customers have no demographic attributes; customer type, KYC rating and declared income are model inputs and would need review before any real use.

## Other models in the repository

- `isolation_forest.joblib`: unsupervised anomaly score, reported as a percentile.
- `logistic_baseline.joblib`: baseline for comparison only.
