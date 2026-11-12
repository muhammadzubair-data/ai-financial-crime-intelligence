"""Explainable Risk and Investigation Reports: what the model relies on, and exportable case files."""
import io
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ALERT_COLUMNS, BLUE, acct, alert_table, case_store, page, show, style
from fincrime.reports.case_report import build_case_report, render_html, render_markdown, to_json

data = page("Explainable Risk and Investigation Reports",
            "What the model pays attention to overall, and structured case reports ready for analyst review.")
alerts, sg = data["alerts"], data["shap_global"]
store = case_store()
analyst = st.session_state.get("analyst", "analyst")

left, right = st.columns([3, 2])
with left:
    st.subheader("What the risk model relies on most")
    top = sg.head(15).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(x=top.mean_abs_shap, y=top.label, orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Average influence on the score (mean absolute SHAP value)")
    show(style(fig, 460))
with right:
    st.subheader("How to read this")
    st.markdown(f"""
The chart ranks the model's inputs by how much they move the risk score on average, across a random sample of
transactions from the monitoring window.

Each flagged transaction also gets its own explanation: the inputs that pushed *that* score up, turned into
plain sentences. Those appear on the Transaction Risk page, in the Workbench and in every case report.

Explanations are computed with `{data['metrics']['transaction_model']['explain_method']}`. They describe what the
model used, which is not the same as proving what happened. An investigator still has to check the evidence.""")

st.subheader("Case report")
if alerts.empty:
    st.warning("There are no alerts to report on.")
    st.stop()
labels = {int(r.account_id): f"{r.alert_id}  |  {acct(r.account_id)}  |  {r.priority}  |  risk {r.risk:.2f}" for r in alerts.itertuples()}
account_id = st.selectbox("Alert", list(labels), format_func=labels.get)
case = store.by_account(account_id)
report = build_case_report(account_id, data, case, store.notes(case["case_id"]) if case else None, analyst)
name = f"case_report_{acct(account_id)}"
b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
b1.download_button("Download HTML", render_html(report), f"{name}.html", "text/html", type="primary")
b2.download_button("Download Markdown", render_markdown(report), f"{name}.md", "text/markdown")
b3.download_button("Download JSON", to_json(report), f"{name}.json", "application/json")
with st.expander("Preview", expanded=True):
    st.markdown(render_markdown(report))

st.subheader("Bulk export")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Alert queue**")
    st.caption(f"All {len(alerts):,} alerts with scores, exposure and reasons.")
    export = alert_table(alerts).assign(Reasons=alerts.reasons.to_numpy())
    st.download_button("Download alert queue (CSV)", export.to_csv(index=False), "alert_queue.csv", "text/csv")
with c2:
    st.markdown("**Case reports for the top alerts**")
    n = st.number_input("How many", min_value=1, max_value=min(50, len(alerts)), value=min(10, len(alerts)))
    if st.button("Build ZIP of reports"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for a_id in alerts.account_id.head(int(n)):
                c = store.by_account(int(a_id))
                r = build_case_report(int(a_id), data, c, store.notes(c["case_id"]) if c else None, analyst)
                z.writestr(f"case_report_{acct(a_id)}.html", render_html(r))
        st.download_button("Download ZIP", buf.getvalue(), "case_reports.zip", "application/zip")

with st.expander("Alert queue preview"):
    st.dataframe(alert_table(alerts.head(200)), hide_index=True, width="stretch", column_config=ALERT_COLUMNS)
