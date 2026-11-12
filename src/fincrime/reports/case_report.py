"""Structured case reports for analyst review.

``build_case_report`` gathers everything known about one account into a list of sections.
``render_markdown`` and ``render_html`` turn the same sections into a file an analyst can save,
and ``to_json`` gives the machine-readable version.
"""
from __future__ import annotations

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config as C
from ..data.generate import OFF_BILLER, OFF_EMPLOYER, OFF_EXTBANK

DISCLAIMER = ("SYNTHETIC DATA. This report was produced from a simulated dataset for demonstration. It describes no "
              "real person, account or transaction. Scores are decision support for a trained investigator and are "
              "not a finding of wrongdoing.")


def party_label(account: int, ext: int, country: str = "") -> str:
    """Readable name for the other side of a transaction."""
    if account is not None and account >= 0:
        return C.acct_label(account)
    if ext is None or ext < 0:
        return "Cash"
    kind = "MER" if ext < OFF_BILLER else "BIL" if ext < OFF_EMPLOYER else "EMP" if ext < OFF_EXTBANK else "EXT"
    tail = f" ({country})" if country and country != C.DOMESTIC else ""
    return f"{kind}-{int(ext):05d}{tail}"


def account_view(tx: pd.DataFrame, account_id: int) -> pd.DataFrame:
    """Transactions as they appear on one account's statement (direction and counterparty relative to it)."""
    t = tx[(tx.src_account == account_id) | (tx.dst_account == account_id)].sort_values("ts").copy()
    out = t.src_account == account_id
    t["direction"] = np.where(out, "Out", "In")
    other = np.where(out, t.dst_account, t.src_account)
    t["counterparty"] = [party_label(o, e, c) for o, e, c in zip(other, t.ext_party, t.ext_country)]
    t["counterparty_account"] = other
    t["signed_amount"] = np.where(out, -t.amount, t.amount)
    return t


def money(x) -> str:
    return f"${x:,.0f}"


def build_case_report(account_id: int, data: dict, case: dict | None = None, notes: pd.DataFrame | None = None,
                      analyst: str = "") -> dict:
    """data: the app artifacts (accounts, alerts, transactions, rule_hits, networks, network_members, shap_txn, summary)."""
    acct = data["accounts"].set_index("account_id").loc[account_id]
    alerts = data["alerts"]
    alert = alerts[alerts.account_id == account_id]
    alert = alert.iloc[0] if len(alert) else None
    view = account_view(data["transactions"], account_id)
    win = view[view.in_window]
    flagged = win[win.flagged].sort_values("score", ascending=False)
    hits = data["rule_hits"]
    hits = hits[(hits.account_id == account_id) & hits.in_window]
    s = data["summary"]
    sections = []

    sections.append(("Subject", "kv", [
        ("Account", C.acct_label(account_id)), ("Customer type", acct.customer_type.title()),
        ("Segment", str(acct.segment).replace("_", " ").title()), ("KYC risk rating", acct.kyc_risk.title()),
        ("Account opened", f"{acct.open_date:%d %b %Y}"), ("Declared annual income / turnover", money(acct.annual_income)),
    ]))
    sections.append(("Risk summary", "kv", [
        ("Monitoring window", f"{s['monitoring_window'][0]} to {s['monitoring_window'][1]}"),
        ("Account risk score", f"{acct.risk:.2f} (alert threshold {s['alert_threshold']:.2f})"),
        ("Priority", acct.priority or "Below alert threshold"),
        ("Alert", f"{alert.alert_id}, raised {alert.alert_ts:%d %b %Y %H:%M}" if alert is not None and pd.notna(alert.alert_ts) else "None"),
        ("Model evidence", f"{acct.ml:.2f}"), ("Rule evidence", f"{acct.rules:.2f}"), ("Network evidence", f"{acct.network:.2f}"),
        ("Flagged transactions", f"{len(flagged)} of {len(win)} in the window"),
        ("Value of flagged transactions", money(flagged.amount.sum())),
    ]))
    reasons = json.loads(alert.reasons) if alert is not None else []
    sections.append(("Why this account was flagged", "list", reasons or ["No alert was raised for this account."]))

    if len(hits):
        h = hits.sort_values("start_ts")[["rule", "start_ts", "end_ts", "n_txns", "amount", "detail"]].copy()
        h["rule"] = h.rule.map(lambda r: C.RULE_LABELS[r].split(" (")[0])
        h["start_ts"], h["end_ts"] = h.start_ts.dt.strftime("%d %b %Y"), h.end_ts.dt.strftime("%d %b %Y")
        h["amount"] = h.amount.map(money)
        h.columns = ["Pattern", "From", "To", "Transactions", "Amount", "Detail"]
        sections.append(("AML patterns triggered", "table", h))
    else:
        sections.append(("AML patterns triggered", "text", "No scenario rule fired for this account in the window."))

    nid = int(acct.detected_network_id)
    if nid > 0:
        net = data["networks"].set_index("detected_network_id").loc[nid]
        mem = data["network_members"]
        mem = mem[mem.detected_network_id == nid]
        me = mem[mem.account_id == account_id].iloc[0]
        others = ", ".join(C.acct_label(a) for a in mem.account_id if a != account_id)
        sections.append(("Network context", "kv", [
            ("Detected network", f"N-{nid:03d} ({net.inferred_type})"), ("Accounts in network", str(int(net.n_accounts))),
            ("Role of this account", me.role), ("Key account", C.acct_label(net.key_account)),
            ("Links", f"{int(net.n_transfer_links)} flagged transfer links, {int(net.n_device_links)} shared-device links"),
            ("Network exposure", money(net.exposure)), ("Other members", others),
        ]))
    else:
        sections.append(("Network context", "text", "This account is not part of a detected network."))

    if len(flagged):
        drivers = data["shap_txn"]
        top = drivers[drivers["rank"] == 1].set_index("txn_id").reason
        f = flagged.head(15)
        table = pd.DataFrame({
            "Date": f.ts.dt.strftime("%d %b %Y %H:%M"), "Type": f.txn_type.str.replace("_", " "), "Direction": f.direction,
            "Counterparty": f.counterparty, "Amount": f.amount.map(lambda x: f"${x:,.2f}"),
            "Score": f.score.map(lambda x: f"{x:.2f}"), "Main driver": f.txn_id.map(top).fillna("Rule pattern"),
        })
        sections.append((f"Flagged transactions (top {len(table)} of {len(flagged)})", "table", table))

    if len(win):
        cp = (win.groupby("counterparty").agg(n=("amount", "size"), money_in=("signed_amount", lambda x: x[x > 0].sum()),
                                              money_out=("signed_amount", lambda x: -x[x < 0].sum()))
              .assign(total=lambda d: d.money_in + d.money_out).sort_values("total", ascending=False).head(8).reset_index())
        cp = pd.DataFrame({"Counterparty": cp.counterparty, "Transactions": cp.n, "Money in": cp.money_in.map(money),
                           "Money out": cp.money_out.map(money)})
        sections.append(("Activity in the window", "kv", [
            ("Transactions", str(len(win))), ("Money in", money(win.signed_amount[win.signed_amount > 0].sum())),
            ("Money out", money(-win.signed_amount[win.signed_amount < 0].sum())),
            ("Distinct counterparties", str(win.counterparty.nunique())),
        ]))
        sections.append(("Main counterparties", "table", cp))

    if case:
        sections.append(("Investigation record", "kv", [
            ("Case", f"#{case['case_id']}"), ("Status", case["status"]), ("Decision", case["disposition"]),
            ("Assigned to", case["assigned_to"] or "Unassigned"), ("Opened", case["opened_at"]), ("Last updated", case["updated_at"]),
        ]))
        if notes is not None and len(notes):
            sections.append(("Investigator notes", "list", [f"{r.created_at} ({r.author}): {r.body}" for r in notes.itertuples()]))
    else:
        sections.append(("Investigation record", "text", "No case has been opened for this account."))

    sections.append(("Method", "text",
                     f"Transactions are scored by a gradient-boosted model trained on {s['train_period'][0]} to "
                     f"{s['train_period'][1]} and applied to a later period it never saw. Account risk combines the model "
                     f"score ({C.ACCOUNT_WEIGHTS['ml']:.0%}), scenario rules ({C.ACCOUNT_WEIGHTS['rules']:.0%}) and network "
                     f"evidence ({C.ACCOUNT_WEIGHTS['network']:.0%}). Drivers are SHAP contributions."))
    return {"title": f"Case report: {C.acct_label(account_id)}", "account_id": int(account_id),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "analyst": analyst, "disclaimer": DISCLAIMER,
            "sections": sections}


# ---------------------------------------------------------------------- renderers
def render_markdown(report: dict) -> str:
    out = [f"# {report['title']}", "", f"Generated {report['generated_at']}" + (f" by {report['analyst']}" if report["analyst"] else ""),
           "", f"> {report['disclaimer']}", ""]
    for title, kind, body in report["sections"]:
        out += [f"## {title}", ""]
        if kind == "kv":
            out += [f"- **{k}:** {v}" for k, v in body]
        elif kind == "list":
            out += [f"- {x}" for x in body]
        elif kind == "table":
            cols = list(body.columns)
            out += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            out += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in body.itertuples(index=False)]
        else:
            out.append(str(body))
        out.append("")
    return "\n".join(out)


CSS = """
body{font-family:Arial,Helvetica,sans-serif;color:#1f2933;max-width:960px;margin:32px auto;padding:0 20px;line-height:1.45}
h1{color:#17324d;font-size:24px;margin-bottom:4px} h2{color:#17324d;font-size:16px;border-bottom:1px solid #d5d9de;padding-bottom:4px;margin-top:28px}
.meta{color:#5f6b7a;font-size:13px} .note{background:#fff7e0;border-left:4px solid #e8912d;padding:10px 14px;font-size:13px;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13px} th{background:#17324d;color:#fff;text-align:left;padding:6px 8px}
td{padding:6px 8px;border-bottom:1px solid #e3e6ea;vertical-align:top} dl{display:grid;grid-template-columns:260px 1fr;gap:4px 16px;font-size:14px;margin:0}
dt{color:#5f6b7a} dd{margin:0} li{margin-bottom:4px;font-size:14px} p{font-size:14px}
@media print{body{margin:0}}
"""


def render_html(report: dict) -> str:
    e = html.escape
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{e(report['title'])}</title><style>{CSS}</style></head><body>",
           f"<h1>{e(report['title'])}</h1>",
           f"<div class='meta'>Generated {e(report['generated_at'])}" + (f" by {e(report['analyst'])}" if report["analyst"] else "") + "</div>",
           f"<div class='note'>{e(report['disclaimer'])}</div>"]
    for title, kind, body in report["sections"]:
        out.append(f"<h2>{e(title)}</h2>")
        if kind == "kv":
            out.append("<dl>" + "".join(f"<dt>{e(str(k))}</dt><dd>{e(str(v))}</dd>" for k, v in body) + "</dl>")
        elif kind == "list":
            out.append("<ul>" + "".join(f"<li>{e(str(x))}</li>" for x in body) + "</ul>")
        elif kind == "table":
            head = "".join(f"<th>{e(str(c))}</th>" for c in body.columns)
            rows = "".join("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in row) + "</tr>" for row in body.itertuples(index=False))
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        else:
            out.append(f"<p>{e(str(body))}</p>")
    out.append("</body></html>")
    return "\n".join(out)


def to_json(report: dict) -> str:
    def conv(kind, body):
        if kind == "kv":
            return dict(body)
        if kind == "table":
            return body.to_dict("records")
        return body
    return json.dumps({**{k: v for k, v in report.items() if k != "sections"},
                       "sections": {title: conv(kind, body) for title, kind, body in report["sections"]}}, indent=1, default=str)
