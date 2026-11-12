"""AI Transaction Risk Scoring: the model's alert queue and the workload trade-off."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQUA, BLUE, RED, SLATE, acct, page, reveal, show, style
from fincrime.models.explain import feature_label
from fincrime.reports.case_report import party_label

data = page("Transaction Risk Scoring",
            "Every transaction in the monitoring window is scored by a model that was trained only on earlier months.")
m = data["metrics"]["transaction_model"]
tx, shap_txn = data["transactions"], data["shap_txn"]
curve = pd.DataFrame(m["workload_curve"]).sort_values("alerts_per_day")
op = m["models"]["lightgbm"]["at_threshold"]

st.subheader("How many alerts can the team handle?")
st.caption("Move the slider to see what a different daily alert volume would mean. Figures are measured on the "
           "monitoring window against the synthetic ground truth.")
c1, c2 = st.columns([1, 2])
with c1:
    options = [round(v, 1) for v in curve.alerts_per_day]
    nearest = min(options, key=lambda v: abs(v - op["alerts_per_day"]))
    pick = st.select_slider("Transaction alerts per day", options=options, value=nearest)
    row = curve.iloc[int(np.argmin(np.abs(curve.alerts_per_day - pick)))]
    a, b = st.columns(2)
    a.metric("Precision", f"{row.precision:.0%}", help="Share of alerts that are truly suspicious.")
    b.metric("Recall", f"{row.recall:.0%}", help="Share of suspicious transactions that are alerted.")
    st.metric("Score threshold", f"{row.threshold:.3f}")
    st.caption(f"In use: threshold {op['threshold']:.3f}, {op['alerts_per_day']:.0f} alerts a day, "
               f"{op['precision']:.0%} precision, {op['recall']:.0%} recall. It was chosen on the validation months "
               f"as the lowest score reaching {m['target_precision']:.0%} precision there.")
with c2:
    fig = go.Figure()
    fig.add_scatter(x=curve.alerts_per_day, y=curve.precision, name="Precision", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=curve.alerts_per_day, y=curve.recall, name="Recall", line=dict(color=AQUA, width=2))
    fig.add_vline(x=pick, line=dict(color=SLATE, dash="dot"))
    fig.update_xaxes(type="log", title="Transaction alerts per day (log scale)")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.03])
    fig.update_traces(hovertemplate="%{x:.0f} alerts/day: %{y:.0%}")
    fig.update_layout(hovermode="x unified")
    show(style(fig, 330, legend=True))

st.subheader("Highest-risk transactions")
win = tx[tx.in_window & tx.score.notna()].copy()
f1, f2, f3 = st.columns([2, 1, 1])
types = f1.multiselect("Transaction type", sorted(win.txn_type.unique()), format_func=lambda t: t.replace("_", " "))
min_amt = f2.number_input("Minimum amount ($)", min_value=0, value=0, step=500)
only = f3.toggle("Above threshold only", value=True)
q = win[(win.amount >= min_amt)]
if types:
    q = q[q.txn_type.isin(types)]
if only:
    q = q[q.score >= op["threshold"]]
q = q.sort_values("score", ascending=False).head(500)
table = pd.DataFrame({
    "Txn": q.txn_id, "Time": q.ts, "Type": q.txn_type.str.replace("_", " "), "Amount": q.amount,
    "From": [party_label(a, e, c) if a >= 0 or t in ("wire_in", "incoming_transfer") else "Cash"
             for a, e, c, t in zip(q.src_account, q.ext_party, q.ext_country, q.txn_type)],
    "To": [party_label(a, e, c) if a >= 0 or t not in ("cash_withdrawal",) else "Cash"
           for a, e, c, t in zip(q.dst_account, q.ext_party, q.ext_country, q.txn_type)],
    "Risk score": q.score, "Unusualness": q.anomaly_pct,
})
if reveal():
    table["Truly suspicious"] = (q.is_suspicious == 1).to_numpy()
st.caption(f"Showing {len(table):,} transactions. Select a row to see why it scored the way it did. "
           "Unusualness is the Isolation Forest percentile: how rare this transaction's behaviour is, without using any labels.")
event = st.dataframe(table, hide_index=True, width="stretch", height=330, on_select="rerun", selection_mode="single-row",
                     column_config={"Time": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                                    "Amount": st.column_config.NumberColumn(format="dollar"),
                                    "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "Unusualness": st.column_config.NumberColumn(format="%.0f", help="Percentile, 0 to 100")})
rows = event.selection.rows if event and event.selection else []
if len(table):
    sel = q.iloc[rows[0] if rows else 0]
    st.subheader(f"Why transaction {sel.txn_id} scored {sel.score:.2f}")
    d = shap_txn[shap_txn.txn_id == sel.txn_id].sort_values("shap")
    if len(d):
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Bar(x=d.shap, y=d.feature.map(feature_label), orientation="h", marker_color=RED,
                                   hovertemplate="%{y}: +%{x:.2f}<extra></extra>"))
            fig.update_xaxes(title="Contribution to the risk score (SHAP, log-odds)")
            show(style(fig, 280))
        with c2:
            st.markdown("**In plain words**")
            for r in d.sort_values("shap", ascending=False).reason:
                st.markdown(f"- {r}")
            for label, a_id in (("Sender", sel.src_account), ("Receiver", sel.dst_account)):
                if a_id >= 0:
                    st.caption(f"{label}: {acct(a_id)}")
    else:
        st.caption("This transaction is below the model threshold, so no explanation was stored for it.")
