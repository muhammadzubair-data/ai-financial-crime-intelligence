"""Model Performance: how the system compares with simpler baselines, and what the numbers do not show."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQUA, BLUE, ORANGE, SLATE, page, show, style
from fincrime import config as C

data = page("Model Performance and Method",
            "All results are for the monitoring window, which no model saw during training.")
m, s = data["metrics"], data["summary"]
tm = m["transaction_model"]
NAMES = {"lightgbm": "LightGBM (this system)", "isolation_forest": "Isolation Forest (no labels)",
         "logistic_baseline": "Logistic regression baseline", "rules_baseline": "Static rule baseline"}

st.warning("These results come from synthetic data, where suspicious behaviour follows four scripted typologies. "
           "Real financial crime is messier and labels are incomplete, so performance on real data would be lower. "
           "Read the numbers as evidence that the pipeline works end to end, not as a forecast.", icon=":material/warning:")

st.subheader("Evaluation design")
sp = pd.DataFrame([{"Period": k.title().replace("Test", "Monitoring window (test)"), "From": v[0], "To": v[1],
                    "Transactions": v[2], "Suspicious": v[3], "Suspicious share": v[3] / v[2]}
                   for k, v in tm["split"].items()])
st.dataframe(sp, hide_index=True, width="stretch",
             column_config={"Suspicious share": st.column_config.NumberColumn(format="percent")})

st.subheader("Transaction scoring: model against baselines")
rows = []
for key, name in NAMES.items():
    r = tm["models"][key]
    rows.append({"Model": name, "PR-AUC": r["pr_auc"], "ROC-AUC": r["roc_auc"],
                 **{f"{k.title()} at {b} alerts/day": r["budgets"][b][k] for b in ("50", "100") for k in ("precision", "recall")}})
cmp = pd.DataFrame(rows)
c1, c2 = st.columns([2, 3])
with c1:
    d = cmp.sort_values("PR-AUC")
    fig = go.Figure(go.Bar(x=d["PR-AUC"], y=d.Model, orientation="h", text=d["PR-AUC"].map("{:.2f}".format),
                           textposition="outside", marker_color=[BLUE if "this system" in n else SLATE for n in d.Model],
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.1], title="PR-AUC (area under the precision-recall curve)")
    show(style(fig, 280))
    st.caption(f"A random ranking would score {tm['base_rate_test']:.3f}, the share of suspicious transactions.")
with c2:
    st.dataframe(cmp, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in cmp.columns if c != "Model"})
    op, rb = tm["models"]["lightgbm"]["at_threshold"], tm["models"]["rules_baseline"]["at_threshold"]
    st.markdown(f"**At the operating threshold** the model raises **{op['alerts_per_day']:.0f} alerts a day** at "
                f"**{op['precision']:.0%} precision** and **{op['recall']:.0%} recall**. The static rule raises "
                f"{rb['alerts_per_day']:.0f} a day at {rb['precision']:.0%} precision and {rb['recall']:.0%} recall.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Recall by typology")
    t = pd.DataFrame(tm["recall_by_typology"])
    t["typology"] = t.typology.map(C.TYPOLOGY_LABELS)
    t.columns = ["Typology", "Suspicious transactions", "Recall"]
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"Recall": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
with c2:
    st.subheader("Recall on stealth networks")
    t = pd.DataFrame(tm["recall_by_stealth"])
    t["stealth"] = t.stealth.map({True: "Stealth", False: "Standard"})
    t.columns = ["Network style", "Suspicious transactions", "Recall"]
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"Recall": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    st.caption("Stealth networks move smaller amounts much more slowly.")

st.subheader("Investigation effectiveness: what a team gets from reviewing the top accounts")
ef = pd.DataFrame(m["effectiveness"])
ac = m["accounts"]
metric = st.radio("Measure", ["recall", "precision", "networks_reached", "suspicious_value_covered"], horizontal=True,
                  format_func={"recall": "Suspicious accounts found", "precision": "Precision",
                               "networks_reached": "Networks reached", "suspicious_value_covered": "Suspicious value covered"}.get)
fig = go.Figure()
for (name, grp), color in zip(ef.groupby("strategy", sort=False), [BLUE, AQUA, ORANGE, SLATE]):
    fig.add_scatter(x=grp.accounts_reviewed, y=grp[metric], name=name, mode="lines+markers", line=dict(color=color, width=2),
                    marker=dict(size=8), hovertemplate="%{x} accounts: %{y:.0%}")
fig.update_xaxes(title=f"Accounts reviewed over the {s['window_days']}-day window", type="category")
fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
fig.update_layout(hovermode="x unified")
show(style(fig, 340, legend=True))
st.caption("Where the hybrid and model-only lines coincide, only one is visible.")
st.markdown(f"""
- The alert queue holds **{ac['alerts']:,} accounts** ({ac['alerts_per_week']:.0f} a week). **{ac['precision']:.0%}** are true
  network members and they cover **{ac['recall']:.0%}** of the {ac['suspicious_accounts_active']:,} suspicious accounts active in the window.
- Account ranking quality (PR-AUC): hybrid score **{ac['pr_auc_hybrid']:.3f}**, model alone **{ac['pr_auc_model_only']:.3f}**,
  rules alone **{ac['pr_auc_rules_only']:.3f}**. On this data the rules and network terms do not improve ranking over the model alone.
  Their value is in the explanation and in grouping accounts into networks.""")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Network recovery")
    n = m["networks"]
    st.markdown(f"""
- Injected networks active in the window: **{n['true_networks_active']}**
- Recovered (half or more of the active members in one detected network): **{n['recovered']} ({n['recovery_rate']:.0%})**
- Touched by at least one alert: **{n['touched_by_alert_rate']:.0%}**
- Detected networks that are mostly real: **{n['detected_mostly_true']} of {n['detected_networks']}**""")
    bs = pd.DataFrame(n["by_stealth"])
    if len(bs):
        bs["stealth"] = bs.stealth.map({True: "Stealth", False: "Standard"})
        bs.columns = ["Network style", "Active", "Recovered", "Recovery rate"]
        st.dataframe(bs, hide_index=True, column_config={"Recovery rate": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="percent")})
with c2:
    st.subheader("Time to detect")
    t = m["time_to_detect"]
    if t["median_hours"] is not None:
        st.markdown(f"""
- Networks that began inside the window: **{t['networks_started_in_window']}**, of which **{t['networks_alerted']}** were alerted
- Median time from a network's first suspicious transaction to its first alert: **{t['median_hours']:.0f} hours**
- Three quarters were alerted within **{t['p75_hours']:.0f} hours**; **{t['within_48h']:.0%}** within 48 hours""")
        st.caption("Alert times are replayed in order, using only the evidence available at each moment.")
    else:
        st.caption("No network started inside the window.")

st.subheader("Limitations")
st.markdown(f"""
- **Synthetic data.** Suspicious networks follow four scripted typologies, {s['stealth_networks']} of {s['n_networks']} in a harder
  "stealth" variant. The model is tested on networks it never saw, but they come from the same generator as the training ones.
- **Perfect labels.** Every suspicious transaction is labelled. A real bank only has labels for cases someone investigated.
- **No public benchmark yet.** The plan allowed for a public fraud benchmark alongside the synthetic data. It is not included in this version.
- **Scores are not calibrated probabilities** and thresholds were set for this dataset.
- **Decision support only.** Nothing here is a finding of wrongdoing; an investigator makes the decision.""")
