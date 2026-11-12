"""AI Investigation Workbench: review one account end to end and record the decision."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQUA, BLUE, PRIORITY_COLOR, RED, RULE_SHORT, SLATE, acct, case_store, network_figure, page, reveal, show, style, usd
from fincrime.cases.store import DISPOSITIONS, STATUSES
from fincrime.models.explain import feature_label
from fincrime.reports.case_report import account_view, build_case_report, render_html, render_markdown, to_json

data = page("Investigation Workbench",
            "Pick an alert, examine the account's activity and evidence, then record your decision.")
alerts, accounts, tx = data["alerts"], data["accounts"], data["transactions"]
store = case_store()
analyst = st.session_state.get("analyst", "analyst")

if alerts.empty:
    st.warning("There are no alerts in this window.")
    st.stop()

# ------------------------------------------------------------------ choose an alert
c1, c2 = st.columns([2, 3])
prio = c1.multiselect("Priority", ["Critical", "High", "Medium"], default=["Critical", "High", "Medium"])
pool = alerts[alerts.priority.isin(prio)] if prio else alerts
labels = {int(r.account_id): f"{r.alert_id}  |  {acct(r.account_id)}  |  {r.priority}  |  risk {r.risk:.2f}  |  {usd(r.exposure)}"
          for r in pool.itertuples()}
if not labels:
    st.warning("No alerts match the filter.")
    st.stop()
account_id = c2.selectbox("Alert", list(labels), format_func=labels.get)
alert = alerts[alerts.account_id == account_id].iloc[0]
a = accounts.set_index("account_id").loc[account_id]
view = account_view(tx, account_id)
win = view[view.in_window]
case = store.by_account(account_id)

# ------------------------------------------------------------------ header
st.markdown(f"### {acct(account_id)} &nbsp; <span style='background:{PRIORITY_COLOR[alert.priority]};color:white;"
            f"padding:2px 10px;border-radius:10px;font-size:0.8rem'>{alert.priority}</span>"
            + (f" &nbsp; <span style='color:#5f6b7a;font-size:0.85rem'>Case #{case['case_id']}: {case['status']}, {case['disposition']}</span>"
               if case else ""), unsafe_allow_html=True)
k = st.columns(6)
k[0].metric("Risk score", f"{alert.risk:.2f}", help=f"Alert threshold {data['summary']['alert_threshold']:.2f}")
k[1].metric("Exposure", usd(alert.exposure))
k[2].metric("Flagged txns", f"{int(alert.n_flagged)} of {len(win)}")
k[3].metric("Customer", a.customer_type.title())
k[4].metric("Account age", f"{(pd.Timestamp(data['summary']['monitoring_window'][1]) - a.open_date).days:,} days")
k[5].metric("KYC rating", a.kyc_risk.title())

left, right = st.columns([3, 2])
with left:
    st.subheader("Why this account was flagged")
    for r in json.loads(alert.reasons):
        st.markdown(f"- {r}")
with right:
    st.subheader("Where the score comes from")
    parts = pd.DataFrame({"Evidence": ["Model", "AML rules", "Network"], "Weight": ["55%", "25%", "20%"],
                          "Strength": [alert.ml, alert.rules, alert.network]})
    st.dataframe(parts, hide_index=True, width="stretch",
                 column_config={"Strength": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.caption("Model evidence is the average of the account's three highest transaction scores, so one odd payment is not enough.")
    if reveal():
        if a.true_network_id >= 0:
            st.success(f"Ground truth: member of injected network {int(a.true_network_id)} "
                       f"({str(a.true_typology).replace('_', ' ')}, role: {str(a.true_role).replace('_', ' ')}).")
        else:
            st.error("Ground truth: this account is not part of any injected network (false positive).")

tabs = st.tabs(["Timeline", "Transactions", "Network and counterparties", "Evidence", "Decision and notes", "Case report"])

# ------------------------------------------------------------------ timeline
with tabs[0]:
    span = st.radio("Period", ["Monitoring window", "Full 12 months"], horizontal=True, label_visibility="collapsed")
    v = win if span == "Monitoring window" else view
    fig = go.Figure()
    for name, mask, color, symbol in (("Money in", (v.direction == "In") & ~v.flagged, AQUA, "circle"),
                                      ("Money out", (v.direction == "Out") & ~v.flagged, BLUE, "circle"),
                                      ("Flagged, in", (v.direction == "In") & v.flagged, RED, "triangle-up"),
                                      ("Flagged, out", (v.direction == "Out") & v.flagged, RED, "triangle-down")):
        d = v[mask]
        fig.add_scatter(x=d.ts, y=d.signed_amount, mode="markers", name=name,
                        marker=dict(color=color, size=10 if "Flagged" in name else 7, symbol=symbol,
                                    line=dict(width=1, color="white")),
                        customdata=list(zip(d.counterparty, d.txn_type.str.replace("_", " "), d.amount)),
                        hovertemplate="%{x|%d %b %H:%M}<br>%{customdata[1]} $%{customdata[2]:,.2f}<br>%{customdata[0]}<extra></extra>")
    fig.update_yaxes(title="Money in (+) and out (-)")
    show(style(fig, 380, legend=True, money_axis="y"))
    bal = v.sort_values("ts").assign(net=lambda d: d.signed_amount.cumsum())
    fig = go.Figure(go.Scatter(x=bal.ts, y=bal.net, mode="lines", line=dict(color=SLATE, width=2, shape="hv"),
                               hovertemplate="%{x|%d %b %H:%M}: $%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(title="Cumulative net flow")
    show(style(fig, 200, money_axis="y"))
    st.caption("Money that arrives and leaves again quickly shows as a net flow that keeps returning to the same level.")

# ------------------------------------------------------------------ transactions
with tabs[1]:
    t = pd.DataFrame({"Time": view.ts, "Type": view.txn_type.str.replace("_", " "), "Direction": view.direction,
                      "Counterparty": view.counterparty, "Amount": view.amount, "Channel": view.channel,
                      "Flagged": view.flagged, "Risk score": view.score, "Unusualness": view.anomaly_pct})
    if reveal():
        t["Truly suspicious"] = (view.is_suspicious == 1).to_numpy()
    only = st.toggle("Flagged only", value=False)
    t = t[t.Flagged] if only else t
    st.dataframe(t.sort_values("Time", ascending=False), hide_index=True, width="stretch", height=420,
                 column_config={"Time": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
                                "Amount": st.column_config.NumberColumn(format="dollar"),
                                "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "Unusualness": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Risk scores are shown for the monitoring window only. Earlier months were used to train the model, "
               "so scores there would be flattering.")

# ------------------------------------------------------------------ network
with tabs[2]:
    nid = int(alert.detected_network_id)
    g, c = st.columns([3, 2])
    with g:
        if nid > 0:
            m = data["network_members"]
            m = m[m.detected_network_id == nid].merge(accounts[["account_id", "risk"]], on="account_id")
            e = data["network_edges"]
            net = data["networks"].set_index("detected_network_id").loc[nid]
            st.markdown(f"**Detected network N-{nid:03d}** ({net.inferred_type}, {int(net.n_accounts)} accounts, "
                        f"{usd(net.exposure)} exposure). The outlined node is this account.")
            show(network_figure(m, e[e.detected_network_id == nid], highlight=account_id, height=420))
        else:
            # ego view: the account and its main counterparties in the window
            top = (win[win.counterparty_account >= 0].groupby("counterparty_account")
                   .agg(amount=("amount", "sum"), out=("signed_amount", lambda x: (x < 0).sum())).nlargest(12, "amount").reset_index())
            nodes = pd.DataFrame({"account_id": [account_id] + list(top.counterparty_account)})
            nodes = nodes.merge(accounts[["account_id", "risk"]], on="account_id", how="left").fillna({"risk": 0})
            nodes["role"] = ["Linked"] + ["Collector" if o else "Feeder" for o in top.out]
            edges = pd.DataFrame({"src": [account_id if o else cp for o, cp in zip(top.out, top.counterparty_account)],
                                  "dst": [cp if o else account_id for o, cp in zip(top.out, top.counterparty_account)],
                                  "kind": "transfer", "amount": top.amount})
            st.markdown("**Not part of a detected network.** Showing this account and its main internal counterparties.")
            show(network_figure(nodes, edges, highlight=account_id, height=420))
    with c:
        st.markdown("**Main counterparties in the window**")
        cp = (win.groupby("counterparty").agg(Transactions=("amount", "size"),
                                              money_in=("signed_amount", lambda x: x[x > 0].sum()),
                                              money_out=("signed_amount", lambda x: -x[x < 0].sum()),
                                              Flagged=("flagged", "sum")).reset_index()
              .rename(columns={"counterparty": "Counterparty", "money_in": "Money in", "money_out": "Money out"}))
        cp = cp.assign(total=cp["Money in"] + cp["Money out"]).sort_values("total", ascending=False).drop(columns="total")
        st.dataframe(cp.head(15), hide_index=True, width="stretch", height=420,
                     column_config={c_: st.column_config.NumberColumn(format="dollar") for c_ in ("Money in", "Money out")})

# ------------------------------------------------------------------ evidence
with tabs[3]:
    hits = data["rule_hits"]
    hits = hits[(hits.account_id == account_id) & hits.in_window]
    st.markdown("**AML patterns triggered**")
    if len(hits):
        h = pd.DataFrame({"Pattern": hits.rule.map(RULE_SHORT), "From": hits.start_ts, "To": hits.end_ts,
                          "Transactions": hits.n_txns, "Amount": hits.amount, "What was seen": hits.detail})
        st.dataframe(h.sort_values("From"), hide_index=True, width="stretch",
                     column_config={"From": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                                    "To": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                                    "Amount": st.column_config.NumberColumn(format="dollar")})
    else:
        st.caption("No scenario rule fired for this account. The alert comes from the model and network evidence.")
    st.markdown("**What drives the model score on this account's flagged transactions**")
    sv = data["shap_txn"]
    sv = sv[sv.txn_id.isin(win.txn_id[win.flagged])]
    if len(sv):
        agg = sv.groupby("feature").shap.agg(["sum", "size"]).nlargest(10, "sum").sort_values("sum")
        fig = go.Figure(go.Bar(x=agg["sum"], y=[feature_label(f) for f in agg.index], orientation="h", marker_color=RED,
                               customdata=agg["size"], hovertemplate="%{y}: +%{x:.1f} across %{customdata} transactions<extra></extra>"))
        fig.update_xaxes(title="Total contribution to risk (SHAP, summed over flagged transactions)")
        show(style(fig, 340))
    else:
        st.caption("No model explanations are stored for this account.")

# ------------------------------------------------------------------ decision
with tabs[4]:
    if case is None:
        st.markdown("No case has been opened for this alert yet.")
        if st.button("Open a case", type="primary"):
            store.open_case(account_id, alert.alert_id, alert.priority, float(alert.risk), actor=analyst)
            st.rerun()
    else:
        f, hcol = st.columns([2, 3])
        with f:
            with st.form("decision"):
                status = st.selectbox("Status", STATUSES, index=STATUSES.index(case["status"]))
                disposition = st.selectbox("Decision", DISPOSITIONS, index=DISPOSITIONS.index(case["disposition"]))
                assigned = st.text_input("Assigned to", value=case["assigned_to"] or analyst)
                note = st.text_area("Add a note", placeholder="What you checked and what you concluded")
                if st.form_submit_button("Save", type="primary"):
                    store.update(case["case_id"], actor=analyst, status=status, disposition=disposition, assigned_to=assigned)
                    if note.strip():
                        store.add_note(case["case_id"], analyst, note)
                    st.rerun()
        with hcol:
            st.markdown("**Notes**")
            notes = store.notes(case["case_id"])
            if len(notes):
                for n in notes.itertuples():
                    st.markdown(f"- {n.created_at} ({n.author}): {n.body}")
            else:
                st.caption("No notes yet.")
            st.markdown("**Audit trail**")
            st.dataframe(store.history(case["case_id"]), hide_index=True, width="stretch", height=200)

# ------------------------------------------------------------------ report
with tabs[5]:
    report = build_case_report(account_id, data, case, store.notes(case["case_id"]) if case else None, analyst)
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    name = f"case_report_{acct(account_id)}"
    b1.download_button("Download HTML", render_html(report), f"{name}.html", "text/html")
    b2.download_button("Download Markdown", render_markdown(report), f"{name}.md", "text/markdown")
    b3.download_button("Download JSON", to_json(report), f"{name}.json", "application/json")
    with st.container(border=True):
        st.markdown(render_markdown(report))
