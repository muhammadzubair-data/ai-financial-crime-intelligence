# Methodology

## 1. Synthetic data

`src/fincrime/data/generate.py` builds a year of bank activity from a fixed random seed.

**Legitimate activity**: card and bill payments, person-to-person transfers within small contact groups, salaries and rent, cash deposits and withdrawals, domestic and international wires.

**Legitimate look-alikes**, added so the problem is not trivial:

| Behaviour | Resembles |
|---|---|
| Cash-heavy businesses depositing 5,500 to 9,950 many times a year | Structuring |
| Individuals with bursts of large cash deposits (tips, private sales) | Structuring |
| Businesses paying 5 to 40 employees on the same day | Fan-out |
| Marketplace sellers paid by many first-time senders | Mule accounts, fan-in |
| Group organisers who collect from several people and pay it all out | Mule accounts, rapid movement |
| House or car purchases: a large sum in, most of it out within days | Layering |
| Monthly remittances abroad, half to higher-risk jurisdictions | High-risk corridor |
| Three flatmates passing similar amounts round in a circle | Circular transfers |
| New accounts funded with a lump sum and spent quickly | Mule accounts |
| Households sharing one device | Fraud ring device sharing |

**Suspicious networks** (520 at full scale):

| Typology | Count | Mechanics |
|---|---|---|
| Mule ring | 180 | Scam victims pay 4 to 12 mule accounts, which forward to one or two collectors, which cash out by wire or ATM. Ring members often share devices. |
| Circular transfers | 120 | 3 to 6 accounts pass a sum round a closed loop several times, each hop slightly smaller. |
| Structuring | 110 | 2 to 6 "smurf" accounts deposit cash just under 10,000 and consolidate it with a beneficiary, who wires it out. |
| Layering chain | 110 | A large inbound wire is passed along 3 to 7 accounts within hours, then wired out. |

One network in five is "stealth": amounts are a quarter to a third of the usual size and movement is 3 to 10 times slower. In stealth structuring the deposits sit well below the band the rule watches.

One member in five is a business account, and many members are established accounts, not new ones. Network accounts also carry ordinary legitimate activity.

## 2. Out-of-time evaluation

| Period | Months | Use |
|---|---|---|
| Train | January to July | Fit the models |
| Validation | August to September | Early stopping, transaction threshold, account alert threshold |
| Monitoring window | October to December | Every reported result and every alert in the app |

Networks active in the monitoring window are different networks from those in training. Features only use information from before each transaction (`tests/test_features.py` recomputes a sample by hand to check).

## 3. Transaction model

LightGBM, 63 leaves, learning rate 0.05, early stopping on validation average precision. Sixty-three features: transaction facts, sender-side behaviour (`s_`), receiver-side behaviour (`d_`) and customer attributes.

**Threshold.** The lowest score that reaches 90% precision on the validation months. On the monitoring window that threshold gives 85% precision, so there is some drift between the two periods.

**Baselines.**

- *Static rule*: 10,000 or more, or 3,000 or more to a high-risk jurisdiction, or a cash deposit just under 10,000.
- *Logistic regression* on raw transaction fields only (amount, time, type, geography).
- *Isolation Forest* on the numeric features, no labels.

**Metrics.** PR-AUC (the base rate is about 1%, so ROC-AUC flatters every model), precision and recall at fixed alert budgets, and recall by typology and by stealth.

## 4. AML scenario rules

Defined in `src/fincrime/aml/rules.py`, thresholds in `config.py`. Rules produce instances (one per account per episode) with the transactions that triggered them. They never see labels.

The circular rule searches for chains A to B to ... to A where each hop follows the previous one within 48 hours, carries 80% to 102% of the previous amount, and the loop closes within 7 days.

## 5. Account risk

```
risk = 0.55 x model evidence + 0.25 x rule evidence + 0.20 x network evidence
```

- **Model evidence**: mean of the account's three highest transaction scores in the window (padded with zeros).
- **Rule evidence**: `1 - product(1 - w)` over the distinct rules triggered (weights: circular 0.60, structuring 0.50, rapid movement 0.45, fan-in 0.40).
- **Network evidence**: average model-plus-rule evidence of the other accounts in the same detected network.

The alert threshold maximises F1 on the validation window and is applied unchanged. Priority bands: Critical at 0.70 and above, High at 0.50, Medium below that.

The weights are a transparent starting point, not a fitted optimum. On this data the model term carries almost all of the ranking power.

## 6. Network detection

1. *Suspects*: accounts whose model-plus-rule evidence is 0.30 or more.
2. *Links*: flagged transfers between suspects, and devices used by more than one suspect. A non-suspect joins only if flagged transfers tie it to two or more suspects.
3. *Networks*: connected groups of three or more.

Roles come from money flow: exit point (most flagged money leaves the bank), collector, pass-through, feeder. The network "type" shown in the app is a heuristic from the patterns its members triggered; it matches the true typology for about four in five of the real networks.

**Evaluation.** An injected network counts as recovered when at least half of its members that were active in the window sit in a single detected network.

## 7. Investigation effectiveness

For the top K accounts under each ranking strategy: precision, share of suspicious accounts found, share of networks reached, and share of suspicious value covered.

**Time to detect** replays each alerted account's transactions in order and records when its score first crosses the threshold, using only evidence available at that moment. Network evidence is left out of the replay, which makes the figure conservative.

## 8. Explanations

SHAP TreeExplainer values for each flagged transaction. The top positive contributions are stored and mapped to sentences in `models/explain.py`. Generic drivers such as transaction type are kept in charts but left out of the short reason list on alerts.
