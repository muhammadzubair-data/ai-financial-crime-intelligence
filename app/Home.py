"""Financial Crime Command Center: what needs attention right now."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (ALERT_COLUMNS, AQUA, BLUE, PRIORITY_COLOR, RULE_COLOR, RULE_SHORT, alert_table, case_store, page,
                    show, style, usd)

data = page("Financial Crime Command Center",
            "Suspicious activity, high-risk accounts, exposure and investigation workload for the monitoring window.")
s, alerts, daily, nets = data["summary"], data["alerts"], data["daily"], data["networks"]
cases = case_store().list()
weeks = s["window_days"] / 7

k = st.columns(6)
k[0].metric("Open alerts", f"{len(alerts):,}", help="Accounts whose risk score is at or above the alert threshold.")
k[1].metric("Critical", f"{(alerts.priority == 'Critical').sum():,}")
k[2].metric("Flagged txns", f"{s['flagged_transactions']:,}",
            help="Transactions above the model threshold or inside an AML pattern, out of "
                 f"{s['window_transactions']:,} in the window.")
k[3].metric("Exposure", usd(s["alert_exposure"]), help="Value of flagged transactions involving alerted accounts.")
k[4].metric("Networks", f"{len(nets):,}")
k[5].metric("Alerts / week", f"{len(alerts) / weeks:,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Flagged transactions per day")
    w = daily[daily.in_window]
    fig = go.Figure(go.Bar(x=w.date, y=w.n_flagged, marker_color=BLUE, hovertemplate="%{x|%d %b}: %{y} flagged<extra></extra>"))
    show(style(fig, 300))
with right:
    st.subheader("Alerts raised per week, by priority")
    a = alerts.assign(week=alerts.alert_ts.dt.to_period("W").dt.start_time)
    wk = a.groupby(["week", "priority"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for p in ["Critical", "High", "Medium"]:
        if p in wk:
            fig.add_bar(x=wk.index, y=wk[p], name=p, marker_color=PRIORITY_COLOR[p],
                        hovertemplate="Week of %{x|%d %b}: %{y} " + p.lower() + "<extra></extra>")
    fig.update_layout(barmode="stack")
    show(style(fig, 300, legend=True))
    st.caption("The first weeks are busiest because networks already active before 1 October are picked up as soon as monitoring starts.")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Exposure by detected network type")
    t = nets.groupby("inferred_type").agg(exposure=("exposure", "sum"), n=("exposure", "size")).sort_values("exposure")
    fig = go.Figure(go.Bar(x=t.exposure, y=t.index, orientation="h", marker_color=AQUA, customdata=t.n,
                           hovertemplate="%{y}: $%{x:,.0f} across %{customdata} networks<extra></extra>"))
    show(style(fig, 280, money_axis="x"))
with c2:
    st.subheader("Alerted accounts by AML pattern")
    counts = {r: int(alerts.rule_list.str.contains(r).sum()) for r in RULE_SHORT}
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1]))
    fig = go.Figure(go.Bar(x=list(counts.values()), y=[RULE_SHORT[r] for r in counts], orientation="h",
                           marker_color=[RULE_COLOR[r] for r in counts], hovertemplate="%{y}: %{x} accounts<extra></extra>"))
    show(style(fig, 280))
    st.caption(f"{(alerts.rule_list == '').sum():,} alerts come from the model alone, with no rule hit.")
with c3:
    st.subheader("Investigation workload")
    opened = set(cases.account_id) if len(cases) else set()
    status = cases.status.value_counts() if len(cases) else pd.Series(dtype=int)
    rows = [("Alerts without a case", int((~alerts.account_id.isin(opened)).sum()))]
    rows += [(f"Cases: {name.lower()}", int(status.get(name, 0))) for name in ["Open", "In review", "Escalated", "Closed"]]
    wl = pd.DataFrame(rows, columns=["Stage", "Count"])
    st.dataframe(wl, hide_index=True, width="stretch",
                 column_config={"Count": st.column_config.ProgressColumn("Count", min_value=0, max_value=max(len(alerts), 1),
                                                                         format="%d")})
    st.caption("Cases are opened and decided in the Investigation Workbench. Decisions are stored in a local SQLite file.")

st.subheader("Priority queue")
st.caption("The 15 highest-risk alerts. Open the Investigation Workbench to review any of them.")
st.dataframe(alert_table(alerts.head(15)), hide_index=True, width="stretch", column_config=ALERT_COLUMNS)
