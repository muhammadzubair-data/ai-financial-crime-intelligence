"""AML Pattern Monitoring: scenario rules for structuring, rapid movement, circular transfers and fan-in."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import RULE_COLOR, RULE_SHORT, acct, page, reveal, show, style, usd
from fincrime import config as C

data = page("AML Pattern Monitoring",
            "Four transparent scenario rules run over every transaction. They need no training data and no labels.")
hits, alerts = data["rule_hits"], data["alerts"]
quality = pd.DataFrame(data["metrics"]["rules"]).set_index("rule")
win = hits[hits.in_window]
alerted = set(alerts.account_id)

k = st.columns(4)
for col, rule in zip(k, RULE_SHORT):
    h = win[win.rule == rule]
    col.metric(RULE_SHORT[rule], f"{len(h):,} instances", f"{h.account_id.nunique():,} accounts", delta_color="off")

st.subheader("Pattern instances per week (full year)")
wk = hits.assign(week=hits.end_ts.dt.to_period("W").dt.start_time).groupby(["week", "rule"]).size().unstack(fill_value=0)
fig = go.Figure()
for rule in RULE_SHORT:
    if rule in wk:
        fig.add_scatter(x=wk.index, y=wk[rule], name=RULE_SHORT[rule], line=dict(color=RULE_COLOR[rule], width=2),
                        hovertemplate="%{y} instances")
fig.add_vrect(x0=data["summary"]["monitoring_window"][0], x1=data["summary"]["monitoring_window"][1], fillcolor="#2a78d6",
              opacity=0.06, line_width=0, annotation_text="Monitoring window", annotation_position="top left")
fig.update_layout(hovermode="x unified")
show(style(fig, 320, legend=True))

st.subheader("Instances in the monitoring window")
c1, c2, c3 = st.columns([2, 1, 1])
rule = c1.selectbox("Pattern", list(RULE_SHORT), format_func=C.RULE_LABELS.get)
min_amt = c2.number_input("Minimum amount ($)", min_value=0, value=0, step=1000)
only_alerted = c3.toggle("Alerted accounts only", value=False,
                         help="A rule hit on its own does not raise an alert; it adds to the account's risk score.")
h = win[(win.rule == rule) & (win.amount >= min_amt)]
if only_alerted:
    h = h[h.account_id.isin(alerted)]
t = pd.DataFrame({"Account": h.account_id.map(acct), "From": h.start_ts, "To": h.end_ts, "Transactions": h.n_txns,
                  "Amount": h.amount, "What was seen": h.detail, "Alerted": h.account_id.isin(alerted),
                  "Linked accounts": h.linked_accounts.map(lambda v: ", ".join(acct(x) for x in list(v)[:6]))})
if reveal():
    t["Truly suspicious"] = h.is_true.to_numpy()
st.caption(f"{len(t):,} instances, {usd(h.amount.sum())} in total. Largest first.")
st.dataframe(t.sort_values("Amount", ascending=False).head(1000), hide_index=True, width="stretch", height=380,
             column_config={"From": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                            "To": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                            "Amount": st.column_config.NumberColumn(format="dollar")})

left, right = st.columns(2)
with left:
    st.subheader("Rule definitions")
    lo, hi = C.RAPID_RATIO
    st.markdown(f"""
- **Structuring:** {C.STRUCT_MIN_DEPOSITS} or more cash deposits between \\${C.STRUCT_LOW:,} and \\${C.CASH_REPORT_THRESHOLD:,}
  on one account, each within {C.STRUCT_WINDOW_DAYS} days of the last, totalling \\${C.STRUCT_MIN_TOTAL:,} or more.
- **Rapid movement:** \\${C.RAPID_MIN_AMOUNT:,} or more leaves an account within {C.RAPID_WINDOW_HOURS} hours of a similar
  amount arriving (outflow between {lo:.0%} and {hi:.0%} of the inflow).
- **Fan-in:** {C.FANIN_MIN_SENDERS['retail']} or more first-time senders (or {C.FANIN_MIN_SENDERS['business']} for a business)
  pay one account \\${C.FANIN_MIN_TOTAL:,} or more within {C.FANIN_WINDOW_DAYS} days.
- **Circular transfers:** money of \\${C.CIRC_MIN_AMOUNT:,} or more passes through 3 to {C.CIRC_MAX_LEN} accounts and returns
  to the first, each hop within {C.CIRC_MAX_HOP_HOURS} hours and carrying a similar amount.

All thresholds are in `src/fincrime/config.py`.""")
with right:
    st.subheader("How often is each rule right?")
    st.caption("Measured over the full year against the synthetic ground truth. Legitimate look-alikes (cash-heavy "
               "businesses, marketplace sellers, flat-share loops) are in the data on purpose, so rules alone are noisy.")
    q = pd.DataFrame({"Pattern": [RULE_SHORT[r] for r in quality.index], "Instances": quality.instances,
                      "Precision": quality.precision, "Networks it should catch": quality.networks_in_scope,
                      "Share of those touched": quality.network_coverage})
    st.dataframe(q, hide_index=True, width="stretch",
                 column_config={"Precision": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                                "Share of those touched": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
