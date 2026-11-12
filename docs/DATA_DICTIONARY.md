# Data dictionary

## Generated data (`data/`, git-ignored)

### accounts.parquet

| Column | Meaning |
|---|---|
| account_id | Integer id, shown in the app as `AC-000123` |
| customer_type | retail or business |
| segment | Customer segment |
| kyc_risk | low, medium or high |
| annual_income | Declared annual income (retail) or turnover (business) |
| open_date | Account opening date |

### transactions.parquet

| Column | Meaning |
|---|---|
| txn_id | Sequential id in time order |
| ts | Timestamp |
| txn_type | card_payment, bill_payment, transfer, incoming_transfer, cash_deposit, cash_withdrawal, wire_out, wire_in |
| channel | pos, online, mobile, web, branch, atm, ach, wire |
| amount | Amount in USD |
| src_account | Paying customer account, or -1 if the payer is external or cash |
| dst_account | Receiving customer account, or -1 if the payee is external or cash |
| ext_party | External counterparty id (merchant, biller, employer or external bank account), or -1 |
| ext_country | `DOM` for domestic, or a fictional jurisdiction code `XA` to `XT` (`XQ` to `XT` are the higher-risk ones) |
| device_id | Device used for digital channels, or -1 |
| is_suspicious | Ground truth label: 1 if the transaction belongs to an injected network |
| typology | mule_ring, circular, structuring, layering or none |
| network_id | Injected network id, or -1 |

### networks.parquet and network_members.parquet

Ground truth. One row per injected network (typology, stealth flag, size, dates, total amount) and one row per member account with its role (mule, collector, cycle_member, smurf, beneficiary, layer). Scam victims pay into mule rings but are not members.

### features.parquet, scores.parquet, rule_hits.parquet

Model features per transaction, model outputs (`score`, `anomaly_pct`, `split`), and AML rule instances.

## App artifacts (`artifacts/`)

| File | Contents |
|---|---|
| summary.json | Dataset facts, periods, thresholds, headline counts |
| metrics.json | All evaluation results |
| accounts.parquet | Every account with risk components, alert flag, priority, detected network and ground truth |
| alerts.parquet | The alert queue with reasons and alert time |
| transactions.parquet | Twelve months of transactions for alerted and near-threshold accounts; scores only inside the monitoring window |
| daily.parquet, type_mix.parquet | Daily and per-type aggregates |
| rule_hits.parquet | AML pattern instances for the full year |
| networks.parquet, network_members.parquet, network_edges.parquet | Detected networks |
| true_networks_window.parquet | Injected networks active in the window and whether each was recovered |
| shap_txn.parquet, shap_global.parquet | Per-transaction and global explanations |
| models/ | Trained LightGBM, Isolation Forest, logistic baseline and feature metadata |
| cases.db | Created by the app when the first case is opened (git-ignored) |
