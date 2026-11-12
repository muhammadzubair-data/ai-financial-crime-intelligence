"""Builds the README illustrations in images/ from artifacts/metrics.json and summary.json.

    python scripts/make_readme_graphics.py

Charts are drawn with Matplotlib from the saved evaluation results, so every number in a
graphic is the number the pipeline produced. The banner and diagrams are hand-written SVG.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
M = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
S = json.loads((ROOT / "artifacts" / "summary.json").read_text())
NAVY, BLUE, AQUA, RED, ORANGE, VIOLET, SLATE, INK, MUTED, GRID, PALE = (
    "#17324d", "#2a78d6", "#1baf7a", "#d64545", "#e8912d", "#7a5cc7", "#6b7a8c", "#1f2933", "#5f6b7a", "#e6e8eb", "#f5f7f9")
FONT = "font-family='Segoe UI,Helvetica,Arial,sans-serif'"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c9d0", "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left", "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 170,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
pct = mtick.PercentFormatter(1, decimals=0)
tm, acc, net = M["transaction_model"], M["accounts"], M["networks"]
op = tm["models"]["lightgbm"]["at_threshold"]


# ---------------------------------------------------------------------------------- SVG helpers
def svg(name, w, h, body):
    (IMG / name).write_text(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' "
                            f"role='img'>{body}</svg>")


def text(x, y, s, size=14, fill=INK, weight="normal", anchor="start", extra=""):
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' font-weight='{weight}' text-anchor='{anchor}' {FONT} {extra}>{s}</text>"


# ---------------------------------------------------------------------------------- banner
def banner():
    import math
    import random
    rnd = random.Random(7)
    w, h = 1280, 360
    b = [f"<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#0f2438'/>"
         f"<stop offset='1' stop-color='#1d4468'/></linearGradient></defs><rect width='{w}' height='{h}' rx='18' fill='url(#g)'/>"]
    # a transaction network on the right: quiet accounts, with one ring picked out
    nodes = [(rnd.uniform(760, 1240), rnd.uniform(40, 320)) for _ in range(34)]
    ring = [(1010 + 78 * math.cos(a), 180 + 78 * math.sin(a)) for a in [i * 2 * math.pi / 6 + 0.3 for i in range(6)]]
    for i, (x, y) in enumerate(nodes):
        for x2, y2 in sorted(nodes, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[1:3]:
            b.append(f"<line x1='{x:.0f}' y1='{y:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='#5f86ad' stroke-opacity='.35'/>")
    for x, y in nodes:
        if min((x - rx) ** 2 + (y - ry) ** 2 for rx, ry in ring) > 900:
            b.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='5' fill='#8fb3d6' fill-opacity='.75'/>")
    for i, (x, y) in enumerate(ring):
        x2, y2 = ring[(i + 1) % 6]
        b.append(f"<line x1='{x:.0f}' y1='{y:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='{RED}' stroke-width='2.5'/>")
        b.append(f"<line x1='{x:.0f}' y1='{y:.0f}' x2='1010' y2='180' stroke='{ORANGE}' stroke-width='1.5' stroke-dasharray='4 4'/>")
    for x, y in ring:
        b.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='9' fill='{RED}' stroke='white' stroke-width='2'/>")
    b.append(f"<circle cx='1010' cy='180' r='13' fill='{ORANGE}' stroke='white' stroke-width='2.5'/>")
    b.append(text(64, 92, "PROJECT 015", 15, "#8fc1f2", "bold", extra="letter-spacing='3'"))
    b.append(text(64, 150, "AI Financial Crime", 46, "white", "bold"))
    b.append(text(64, 204, "Intelligence Platform", 46, "white", "bold"))
    b.append(text(64, 248, "Fraud detection, transaction monitoring and investigation analytics", 19, "#c9dcee"))
    x = 64
    for label in ["LightGBM", "NetworkX", "DuckDB", "SHAP", "Streamlit"]:
        wd = 22 + 9.2 * len(label)
        b.append(f"<rect x='{x}' y='282' width='{wd:.0f}' height='32' rx='16' fill='white' fill-opacity='.12' stroke='white' stroke-opacity='.3'/>")
        b.append(text(x + wd / 2, 303, label, 14, "white", "600", "middle"))
        x += wd + 12
    svg("banner.svg", w, h, "".join(b))


# ---------------------------------------------------------------------------------- headline numbers
def stat_cards():
    cards = [(f"{S['n_accounts']:,}", "customer accounts", BLUE), (f"{S['n_transactions'] / 1e6:.2f}M", "transactions, 12 months", BLUE),
             (f"{S['n_networks']}", "injected fraud networks", VIOLET), (f"{tm['models']['lightgbm']['pr_auc']:.2f}", "PR-AUC, unseen months", AQUA),
             (f"{net['recovery_rate']:.0%}", "networks recovered", AQUA), (f"{M['time_to_detect']['median_hours']:.0f} h", "median time to alert", ORANGE)]
    w, h, cw, gap = 1280, 150, 196, 20.8
    b = [f"<rect width='{w}' height='{h}' rx='14' fill='white'/>"]
    for i, (big, small, color) in enumerate(cards):
        x = i * (cw + gap)
        b.append(f"<rect x='{x:.1f}' y='8' width='{cw}' height='134' rx='12' fill='{PALE}' stroke='#dfe3e8'/>")
        b.append(f"<rect x='{x:.1f}' y='8' width='6' height='134' rx='3' fill='{color}'/>")
        b.append(text(x + cw / 2 + 3, 78, big, 38, NAVY, "bold", "middle"))
        b.append(text(x + cw / 2 + 3, 112, small, 14, MUTED, "normal", "middle"))
    svg("stats.svg", w, h, "".join(b))


# ---------------------------------------------------------------------------------- architecture
def architecture():
    w, h = 1280, 470
    b = [f"<rect width='{w}' height='{h}' rx='14' fill='white' stroke='#dfe3e8'/>",
         "<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
         f"<path d='M0 0L10 5L0 10z' fill='{SLATE}'/></marker></defs>",
         text(32, 44, "How the platform works", 20, NAVY, "bold")]
    steps = [("1", "Generate", "Synthetic accounts,", "transactions, networks", "NumPy", SLATE),
             ("2", "Features", "Sender and receiver", "behaviour, look-back only", "DuckDB", BLUE),
             ("3", "Score", "Risk model, anomaly", "score, explanations", "LightGBM, SHAP", BLUE),
             ("4", "Monitor", "Structuring, rapid, fan-in,", "circular transfer rules", "DuckDB, NetworkX", VIOLET),
             ("5", "Connect", "Account risk, alert queue,", "fraud-ring detection", "NetworkX", ORANGE),
             ("6", "Investigate", "Workbench, decisions,", "case reports", "Streamlit, SQLite", AQUA)]
    bw, gap, y = 180, 28, 78
    for i, (n, title, l1, l2, tech, color) in enumerate(steps):
        x = 32 + i * (bw + gap)
        b.append(f"<rect x='{x}' y='{y}' width='{bw}' height='150' rx='12' fill='{PALE}' stroke='{color}' stroke-width='2'/>")
        b.append(f"<circle cx='{x + 26}' cy='{y + 28}' r='15' fill='{color}'/>" + text(x + 26, y + 33, n, 15, "white", "bold", "middle"))
        b.append(text(x + 50, y + 34, title, 17, NAVY, "bold"))
        b.append(text(x + 14, y + 70, l1, 12.5, INK) + text(x + 14, y + 88, l2, 12.5, INK))
        b.append(f"<rect x='{x + 12}' y='{y + 108}' width='{bw - 24}' height='26' rx='13' fill='white' stroke='#d5d9de'/>")
        b.append(text(x + bw / 2, y + 126, tech, 12, MUTED, "600", "middle"))
        if i < 5:
            b.append(f"<line x1='{x + bw + 3}' y1='{y + 75}' x2='{x + bw + gap - 4}' y2='{y + 75}' stroke='{SLATE}' stroke-width='2' marker-end='url(#a)'/>")
    # time split
    b.append(text(32, 282, "Out-of-time evaluation: the model is judged on months it never saw", 15, NAVY, "bold"))
    x0, x1, yy = 32, 1248, 300
    month = (x1 - x0) / 12
    for lab, a, z, color, sub in (("TRAIN", 0, 7, BLUE, "January to July"), ("VALIDATE", 7, 9, VIOLET, "Aug to Sep"),
                                  ("MONITORING WINDOW (TEST)", 9, 12, AQUA, "October to December: every alert in the app")):
        b.append(f"<rect x='{x0 + a * month + 2:.0f}' y='{yy}' width='{(z - a) * month - 4:.0f}' height='52' rx='8' fill='{color}'/>")
        b.append(text(x0 + (a + z) / 2 * month, yy + 23, lab, 13, "white", "bold", "middle", "letter-spacing='1'"))
        b.append(text(x0 + (a + z) / 2 * month, yy + 41, sub, 12, "white", "normal", "middle"))
    for i, mname in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()):
        b.append(text(x0 + (i + .5) * month, yy + 72, mname, 12, MUTED, "normal", "middle"))
    # risk formula
    b.append(text(32, 412, "Account risk", 15, NAVY, "bold"))
    x = 150
    for lab, wt, color in (("model evidence", "55%", BLUE), ("rule evidence", "25%", VIOLET), ("network evidence", "20%", ORANGE)):
        ww = 250
        b.append(f"<rect x='{x}' y='390' width='{ww}' height='36' rx='18' fill='{color}' fill-opacity='.12' stroke='{color}'/>")
        b.append(text(x + 18, 414, wt, 15, color, "bold") + text(x + 64, 414, lab, 14, INK))
        x += ww + 14
        if lab != "network evidence":
            b.append(text(x - 7, 414, "+", 18, MUTED, "bold", "middle"))
    b.append(text(32, 452, "Model evidence is the average of an account's three highest transaction scores, so a single odd payment does not raise an alert.", 12.5, MUTED))
    svg("architecture.svg", w, h, "".join(b))


# ---------------------------------------------------------------------------------- typologies
def typologies():
    import math
    w, h = 1280, 330
    b = [f"<rect width='{w}' height='{h}' rx='14' fill='white' stroke='#dfe3e8'/>",
         "<defs><marker id='t' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'>"
         f"<path d='M0 0L10 5L0 10z' fill='{SLATE}'/></marker></defs>",
         text(32, 44, "The four suspicious typologies in the data", 20, NAVY, "bold")]

    def node(x, y, color, r=11):
        return f"<circle cx='{x:.0f}' cy='{y:.0f}' r='{r}' fill='{color}' stroke='white' stroke-width='2'/>"

    def arrow(x1, y1, x2, y2, pad=14):
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        return (f"<line x1='{x1 + ux * pad:.0f}' y1='{y1 + uy * pad:.0f}' x2='{x2 - ux * pad:.0f}' y2='{y2 - uy * pad:.0f}' "
                f"stroke='{SLATE}' stroke-width='1.8' marker-end='url(#t)'/>")

    cw = 296
    counts = S["networks_by_typology"]
    panels = [("Mule ring", counts["mule_ring"], "Victims pay mules, mules forward", "to a collector who cashes out"),
              ("Circular transfers", counts["circular"], "Money goes round a loop and", "comes back to where it started"),
              ("Structuring", counts["structuring"], "Cash deposits just under 10,000,", "consolidated and wired out"),
              ("Layering chain", counts["layering"], "A large wire is passed along", "a chain within hours")]
    for i, (title, n, l1, l2) in enumerate(panels):
        x = 32 + i * (cw + 10.6)
        cx, cy = x + cw / 2, 150
        b.append(f"<rect x='{x:.0f}' y='64' width='{cw}' height='246' rx='12' fill='{PALE}'/>")
        if i == 0:
            v = [(x + 40, 100 + k * 34) for k in range(4)]
            m = [(x + 130, 115 + k * 36) for k in range(3)]
            c, out = (x + 210, 150), (x + 268, 150)
            for k, p in enumerate(v):
                b.append(arrow(*p, *m[min(k, 2)]))
            for p in m:
                b.append(arrow(*p, *c))
            b.append(arrow(*c, *out, 16))
            b += [node(*p, "#9aa5b1", 8) for p in v] + [node(*p, RED) for p in m] + [node(*c, ORANGE, 14)]
            b.append(text(out[0], out[1] + 5, "$", 20, AQUA, "bold", "middle"))
        elif i == 1:
            pts = [(cx + 62 * math.cos(a), cy + 56 * math.sin(a)) for a in [k * 2 * math.pi / 5 - 1.57 for k in range(5)]]
            for k, p in enumerate(pts):
                b.append(arrow(*p, *pts[(k + 1) % 5]))
            b += [node(*p, VIOLET) for p in pts]
        elif i == 2:
            sm = [(x + 110, 100 + k * 34) for k in range(4)]
            ben, out = (x + 205, 150), (x + 268, 150)
            for p in sm:
                b.append(text(x + 40, p[1] + 5, "9,400", 13, MUTED, "600", "middle") + arrow(x + 62, p[1], *p, 4))
                b.append(arrow(*p, *ben))
            b.append(arrow(*ben, *out, 16))
            b += [node(*p, BLUE) for p in sm] + [node(*ben, ORANGE, 14)]
            b.append(text(out[0], out[1] + 5, "$", 20, AQUA, "bold", "middle"))
        else:
            pts = [(x + 62 + k * 44, cy + (18 if k % 2 else -18)) for k in range(5)]
            b.append(text(x + 24, pts[0][1] + 5, "$", 20, AQUA, "bold", "middle") + arrow(x + 30, pts[0][1], *pts[0], 6))
            for k in range(4):
                b.append(arrow(*pts[k], *pts[k + 1]))
            b.append(arrow(*pts[-1], x + 276, pts[-1][1], 14) + text(x + 280, pts[-1][1] + 5, "$", 20, AQUA, "bold", "middle"))
            b += [node(*p, AQUA) for p in pts]
        b.append(text(x + 18, 262, title, 16, NAVY, "bold") + text(x + cw - 18, 262, f"{n} networks", 13, MUTED, "600", "end"))
        b.append(text(x + 18, 282, l1, 12.5, INK) + text(x + 18, 299, l2, 12.5, INK))
    svg("typologies.svg", w, h, "".join(b))


# ---------------------------------------------------------------------------------- charts
def chart_models():
    names = {"lightgbm": "LightGBM (this system)", "isolation_forest": "Isolation Forest (no labels)",
             "logistic_baseline": "Logistic regression", "rules_baseline": "Static rule"}
    keys = ["rules_baseline", "logistic_baseline", "isolation_forest", "lightgbm"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (title, get) in zip(axes, [("PR-AUC", lambda k: tm["models"][k]["pr_auc"]),
                                       ("Precision at 50 alerts a day", lambda k: tm["models"][k]["budgets"]["50"]["precision"]),
                                       ("Recall at 50 alerts a day", lambda k: tm["models"][k]["budgets"]["50"]["recall"])]):
        vals = [get(k) for k in keys]
        bars = ax.barh([names[k] for k in keys], vals, color=[BLUE if k == "lightgbm" else "#aab4c0" for k in keys], height=0.62)
        ax.bar_label(bars, labels=[f"{v:.2f}" if title == "PR-AUC" else f"{v:.0%}" for v in vals], padding=4, color=INK, fontsize=11)
        ax.set_title(title)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Transaction scoring on October to December, months the model never saw", x=0.01, ha="left", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(IMG / "chart_model_comparison.png")
    plt.close(fig)


def chart_effectiveness():
    ef = pd.DataFrame(M["effectiveness"])
    short = {"Hybrid risk score (model + rules + network)": ("Hybrid score", BLUE), "Model only": ("Model only", AQUA),
             "AML rules only": ("AML rules only", ORANGE), "Static baseline (10k+ / high-risk corridor rule)": ("Static rule", SLATE)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, [("recall", "Suspicious accounts found"), ("suspicious_value_covered", "Suspicious money covered")]):
        for name, grp in ef.groupby("strategy", sort=False):
            lab, color = short[name]
            style = dict(ls="--", lw=2) if lab == "Model only" else dict(lw=2.4)
            ax.plot(range(len(grp)), grp[col], marker="o", ms=7, color=color, label=lab, mec="white", mew=1.5, **style)
            ax.annotate(f"{grp[col].iloc[-1]:.0%}", (len(grp) - 1, grp[col].iloc[-1]), xytext=(8, -4), textcoords="offset points", color=color, fontsize=10)
        ax.set_xticks(range(len(grp)), [f"{k:,}" for k in grp.accounts_reviewed])
        ax.set_xlim(-0.2, len(grp) - 0.55)
        ax.set_title(title)
        ax.set_xlabel(f"Accounts reviewed in the {S['window_days']}-day window")
        ax.yaxis.set_major_formatter(pct)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle("Investigation effectiveness: what a team gets for the same review effort", x=0.01, ha="left", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(IMG / "chart_investigation_effectiveness.png")
    plt.close(fig)


def chart_networks():
    labels = {"mule_ring": "Mule ring", "circular": "Circular", "structuring": "Structuring", "layering": "Layering"}
    bt = pd.DataFrame(net["by_typology"])
    bs = pd.DataFrame(net["by_stealth"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    x = range(len(bt))
    ax.bar(x, bt.active, color="#d5dbe2", width=0.62, label="Active in the window")
    bars = ax.bar(x, bt.recovered, color=AQUA, width=0.62, label="Recovered")
    ax.bar_label(bars, labels=[f"{r} of {a}" for r, a in zip(bt.recovered, bt.active)], padding=-18, color="white", fontweight="bold")
    ax.set_xticks(list(x), [labels[t] for t in bt.typology])
    ax.set_title(f"Networks recovered by typology ({net['recovered']} of {net['true_networks_active']} overall)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    rec = {bool(r["stealth"]): r["recall"] for r in tm["recall_by_stealth"]}
    groups = ["Standard networks", "Stealth networks"]
    v1 = [rec[False], rec[True]]
    v2 = [float(bs[bs.stealth == s].rate.iloc[0]) for s in (False, True)]
    b1 = ax.bar([i - 0.2 for i in range(2)], v1, width=0.38, color=BLUE, label="Suspicious transactions flagged")
    b2 = ax.bar([i + 0.2 for i in range(2)], v2, width=0.38, color=AQUA, label="Networks recovered")
    ax.bar_label(b1, labels=[f"{v:.0%}" for v in v1], padding=3)
    ax.bar_label(b2, labels=[f"{v:.0%}" for v in v2], padding=3)
    ax.set_xticks(range(2), groups)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Stealth networks are the weak spot")
    ax.legend(frameon=False, loc="upper center", fontsize=9.5, ncol=2, columnspacing=1)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(IMG / "chart_network_recovery.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (banner, stat_cards, architecture, typologies, chart_models, chart_effectiveness, chart_networks):
        f()
    print(sorted(p.name for p in IMG.glob("*.svg")) + sorted(p.name for p in IMG.glob("chart_*.png")))
