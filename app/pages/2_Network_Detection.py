"""Fraud Ring and Network Detection: groups of accounts linked by flagged transfers and shared devices."""
import pandas as pd
import streamlit as st

from common import acct, network_figure, page, reveal, show, usd
from fincrime import config as C

data = page("Fraud Ring and Network Detection",
            "Suspect accounts are linked when flagged money passes between them or when they share a device. "
            "Each connected group of three or more is a candidate network.")
nets, mem, edges, accounts = data["networks"], data["network_members"], data["network_edges"], data["accounts"]
ev = data["metrics"]["networks"]

if nets.empty:
    st.warning("No networks were detected in this window.")
    st.stop()

k = st.columns(5)
k[0].metric("Networks detected", f"{len(nets):,}")
k[1].metric("Accounts in networks", f"{len(mem):,}")
k[2].metric("Total exposure", usd(nets.exposure.sum()))
k[3].metric("Largest network", f"{int(nets.n_accounts.max())} accounts")
k[4].metric("With shared devices", f"{(nets.n_device_links > 0).sum():,}")

st.subheader("Detected networks")
table = pd.DataFrame({
    "Network": nets.detected_network_id.map(lambda n: f"N-{n:03d}"), "Type": nets.inferred_type,
    "Accounts": nets.n_accounts, "Exposure": nets.exposure, "Transfer links": nets.n_transfer_links,
    "Device links": nets.n_device_links, "Key account": nets.key_account.map(acct),
    "Patterns": nets.patterns.map(lambda p: ", ".join(C.RULE_LABELS[x].split(" (")[0] for x in p.split(", ")) if p else ""),
    "First seen": nets.first_ts, "Last seen": nets.last_ts,
})
if reveal():
    table["Purity (truth)"] = nets.purity
st.caption("Sorted by exposure. Select a row to draw the network.")
event = st.dataframe(table, hide_index=True, width="stretch", height=280, on_select="rerun", selection_mode="single-row",
                     column_config={"Exposure": st.column_config.NumberColumn(format="dollar"),
                                    "First seen": st.column_config.DateColumn(format="D MMM"),
                                    "Last seen": st.column_config.DateColumn(format="D MMM"),
                                    "Purity (truth)": st.column_config.ProgressColumn(
                                        min_value=0, max_value=1, format="percent",
                                        help="Share of members that really belong to an injected network.")})
rows = event.selection.rows if event and event.selection else []
net = nets.iloc[rows[0] if rows else 0]
nid = int(net.detected_network_id)
m = mem[mem.detected_network_id == nid].merge(accounts[["account_id", "risk", "customer_type", "priority", "open_date",
                                                         "true_network_id", "true_role"]], on="account_id")
e = edges[edges.detected_network_id == nid]

st.subheader(f"Network N-{nid:03d}: {net.inferred_type}")
g, side = st.columns([3, 2])
with g:
    show(network_figure(m, e, highlight=int(net.key_account)))
    st.caption("Arrows are flagged transfers (thicker means more money). Dotted lines join accounts that used the same "
               "device. The outlined node is the key account: the one most of the money has to pass through.")
with side:
    a, b = st.columns(2)
    a.metric("Accounts", int(net.n_accounts))
    b.metric("Exposure", usd(net.exposure))
    a.metric("Moved internally", usd(net.flagged_internal_amount))
    b.metric("Sent out of bank", usd(m.exit_amount.sum()))
    st.markdown(f"**Active:** {net.first_ts:%d %b} to {net.last_ts:%d %b %Y}  \n"
                f"**Patterns:** {', '.join(C.RULE_LABELS[p].split(' (')[0] for p in net.patterns.split(', ')) if net.patterns else 'none'}  \n"
                f"**Key account:** {acct(net.key_account)}")
    if reveal():
        truth = ("no injected network" if net.true_network_id < 0 else
                 f"injected network {int(net.true_network_id)} ({C.TYPOLOGY_LABELS.get(net.true_typology, 'n/a').lower()})")
        st.markdown(f"**Ground truth:** {net.purity:.0%} of members belong to an injected network; mostly {truth}.")

st.subheader("Members")
mt = pd.DataFrame({"Account": m.account_id.map(acct), "Inferred role": m.role, "Risk": m.risk, "Priority": m.priority,
                   "Customer": m.customer_type.str.title(), "Opened": m.open_date, "Received in network": m.in_amount,
                   "Sent in network": m.out_amount, "Sent out of bank": m.exit_amount, "Links": m.degree})
if reveal():
    mt["True role"] = m.true_role.fillna("not in a network").str.replace("_", " ")
st.dataframe(mt.sort_values("Risk", ascending=False), hide_index=True, width="stretch",
             column_config={"Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                            "Opened": st.column_config.DateColumn(format="D MMM YYYY"),
                            **{c: st.column_config.NumberColumn(format="dollar")
                               for c in ("Received in network", "Sent in network", "Sent out of bank")}})

with st.expander("How well does network detection work? (measured against the synthetic ground truth)"):
    a, b, c, d = st.columns(4)
    a.metric("Injected networks active", ev["true_networks_active"])
    b.metric("Recovered", f"{ev['recovery_rate']:.0%}", help="At least half of the active members ended up in one detected network.")
    c.metric("Touched by an alert", f"{ev['touched_by_alert_rate']:.0%}", help="At least one member was alerted.")
    d.metric("Detected networks that are real", f"{ev['detected_mostly_true']} of {ev['detected_networks']}",
             help="At least half of the members belong to an injected network.")
    bt = pd.DataFrame(ev["by_typology"])
    if len(bt):
        bt["typology"] = bt.typology.map(C.TYPOLOGY_LABELS)
        bt.columns = ["Typology", "Active in window", "Recovered", "Recovery rate"]
        st.dataframe(bt, hide_index=True, column_config={"Recovery rate": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="percent")})
