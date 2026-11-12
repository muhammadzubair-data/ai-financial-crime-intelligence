"""Shared helpers for the Streamlit pages: data loading, page chrome, formatting and chart styling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fincrime import config as C  # noqa: E402
from fincrime.cases.store import CaseStore  # noqa: E402

# one colour per job, used the same way on every page
BLUE, AQUA, RED, ORANGE, SLATE, VIOLET = "#2a78d6", "#1baf7a", "#d64545", "#e8912d", "#6b7a8c", "#7a5cc7"
INK, MUTED, GRID = "#1f2933", "#5f6b7a", "#e6e8eb"
PRIORITY_COLOR = {"Critical": RED, "High": ORANGE, "Medium": SLATE}
RULE_COLOR = {"STRUCTURING": BLUE, "RAPID_MOVEMENT": AQUA, "CIRCULAR": VIOLET, "FAN_IN": ORANGE}
ROLE_COLOR = {"Exit point": RED, "Collector": ORANGE, "Pass-through": BLUE, "Feeder": AQUA, "Linked": SLATE}
RULE_SHORT = {k: v.split(" (")[0] for k, v in C.RULE_LABELS.items()}
FILES = ["accounts", "alerts", "transactions", "daily", "type_mix", "rule_hits", "networks", "network_members",
         "network_edges", "true_networks_window", "shap_txn", "shap_global"]


@st.cache_data(show_spinner="Loading data")
def load(artifact_dir: str = str(C.ARTIFACT_DIR)) -> dict:
    d = Path(artifact_dir)
    data = {name: pd.read_parquet(d / f"{name}.parquet") for name in FILES}
    data["summary"] = json.loads((d / "summary.json").read_text())
    data["metrics"] = json.loads((d / "metrics.json").read_text())
    return data


@st.cache_resource
def case_store() -> CaseStore:
    return CaseStore(C.CASE_DB)


def page(title: str, caption: str = "") -> dict:
    """Standard page setup. Returns the loaded data, or stops with instructions if the pipeline has not run."""
    st.set_page_config(page_title=f"{title} | Financial Crime Intelligence", page_icon=":material/policy:", layout="wide")
    if not (C.ARTIFACT_DIR / "summary.json").exists():
        st.title(title)
        st.error("No data found. Run the pipeline first:  `python -m fincrime.pipeline --scale full`")
        st.stop()
    data = load()
    s = data["summary"]
    st.markdown("""<style>
        .block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1400px}
        [data-testid="stMetric"]{background:#f5f7f9;border-radius:8px;padding:12px 16px;border-left:4px solid #2a78d6}
        [data-testid="stMetricLabel"] p{font-size:0.78rem;color:#5f6b7a;text-transform:uppercase;letter-spacing:.03em}
        [data-testid="stMetricValue"]{font-size:1.55rem;color:#17324d}
        h1{color:#17324d;font-size:1.9rem !important;padding-bottom:0}
        h3{color:#17324d;font-size:1.05rem !important;margin-top:.4rem}
        </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**Financial Crime Intelligence**")
        st.caption(f"Monitoring window  \n{s['monitoring_window'][0]} to {s['monitoring_window'][1]}")
        st.caption(f"{s['n_accounts']:,} accounts  \n{s['n_transactions']:,} transactions (12 months)")
        st.toggle("Show synthetic ground truth", key="reveal", value=st.session_state.get("reveal", False),
                  help="The data is simulated, so the true answer is known. Switch on to see it next to the system's output.")
        st.text_input("Analyst name", key="analyst", value=st.session_state.get("analyst", "Demo Analyst"))
    st.title(title)
    if caption:
        st.caption(caption)
    st.info("Synthetic data: every account, transaction and network here is simulated. Nothing describes a real person or bank.",
            icon=":material/science:")
    return data


def reveal() -> bool:
    return bool(st.session_state.get("reveal", False))


def acct(a) -> str:
    return C.acct_label(a)


def usd(x, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    if abs(x) >= 1e6 and decimals == 0:
        return f"${x / 1e6:,.2f}M"
    return f"${x:,.{decimals}f}"


def style(fig: go.Figure, height: int = 320, legend: bool = False, money_axis: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=8, r=8, t=28, b=8), showlegend=legend,
                      font=dict(color=INK, size=12), hoverlabel=dict(font_size=12), bargap=0.25,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None))
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c9d0", title_font=dict(color=MUTED), tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="#c3c9d0", title_font=dict(color=MUTED),
                     tickfont=dict(color=MUTED))
    if money_axis == "y":
        fig.update_yaxes(tickformat="$~s")
    elif money_axis == "x":
        fig.update_xaxes(tickformat="$~s")
    return fig


def show(fig: go.Figure, **kwargs) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kwargs)


def network_figure(nodes: pd.DataFrame, edges: pd.DataFrame, highlight: int | None = None, height: int = 480) -> go.Figure:
    """nodes: account_id, role, risk. edges: src, dst, kind ('transfer' or 'device'), amount."""
    g = nx.Graph()
    g.add_nodes_from(nodes.account_id)
    for e in edges.itertuples():
        if e.src in g and e.dst in g and e.src != e.dst:
            g.add_edge(e.src, e.dst)
    pos = nx.kamada_kawai_layout(g) if g.number_of_edges() else nx.circular_layout(g)
    fig = go.Figure()
    tr = edges[edges.kind == "transfer"]
    max_amt = max(tr.amount.max(), 1) if len(tr) else 1
    annotations = []
    for e in tr.itertuples():
        if e.src not in pos or e.dst not in pos or e.src == e.dst:
            continue
        (x0, y0), (x1, y1) = pos[e.src], pos[e.dst]
        annotations.append(dict(ax=x0, ay=y0, x=x1, y=y1, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                arrowhead=2, arrowsize=1, arrowwidth=1 + 3 * e.amount / max_amt, arrowcolor="#9aa5b1",
                                standoff=11, startstandoff=11))
    dv = edges[edges.kind == "device"]
    xs, ys = [], []
    for e in dv.itertuples():
        if e.src in pos and e.dst in pos:
            xs += [pos[e.src][0], pos[e.dst][0], None]
            ys += [pos[e.src][1], pos[e.dst][1], None]
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=VIOLET, width=1.5, dash="dot"),
                                 name="Shared device", hoverinfo="skip"))
    for role, grp in nodes.groupby("role", sort=False):
        fig.add_trace(go.Scatter(
            x=[pos[a][0] for a in grp.account_id], y=[pos[a][1] for a in grp.account_id], mode="markers+text", name=role,
            text=[acct(a)[-4:] for a in grp.account_id], textposition="top center", textfont=dict(size=10, color=MUTED),
            marker=dict(size=[26 if a == highlight else 18 for a in grp.account_id], color=ROLE_COLOR.get(role, SLATE),
                        line=dict(width=[3 if a == highlight else 1.5 for a in grp.account_id],
                                  color=[INK if a == highlight else "white" for a in grp.account_id])),
            customdata=[[acct(a), r] for a, r in zip(grp.account_id, grp.risk)],
            hovertemplate="%{customdata[0]}<br>" + role + "<br>Risk %{customdata[1]:.2f}<extra></extra>"))
    style(fig, height, legend=True)
    fig.update_layout(annotations=annotations, legend=dict(orientation="h", y=-0.02, yanchor="top", x=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, zeroline=False)
    return fig


def alert_table(alerts: pd.DataFrame) -> pd.DataFrame:
    """Display columns for an alert queue."""
    out = pd.DataFrame({
        "Alert": alerts.alert_id, "Account": alerts.account_id.map(acct), "Priority": alerts.priority,
        "Risk": alerts.risk, "Raised": alerts.alert_ts, "Exposure": alerts.exposure,
        "Flagged txns": alerts.n_flagged.astype(int), "Patterns": alerts.rule_list.map(
            lambda r: ", ".join(RULE_SHORT.get(x, x) for x in r.split(", ")) if r else ""),
        "Network": alerts.detected_network_id.map(lambda n: f"N-{n:03d}" if n > 0 else ""),
    })
    return out


ALERT_COLUMNS = {
    "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
    "Raised": st.column_config.DatetimeColumn("Raised", format="D MMM, HH:mm"),
    "Exposure": st.column_config.NumberColumn("Exposure", format="dollar"),
}
