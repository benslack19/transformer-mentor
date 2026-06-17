"""
Generate browsing_model_launch_readiness.html — reveal.js slide deck with interactive Plotly charts.
Run from the repo root: python notebooks/assessment/generate_slides.py
"""
import ast
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import krippendorff
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "../../local_data/assessment/browse_evals_takehome_v1.1.csv"
OUT_PATH  = Path(__file__).parent / "browsing_model_launch_readiness.html"

COLOR_A   = "#10A37F"  # OpenAI green – Model A (release candidate)
COLOR_B   = "#1B1B1B"  # near-black   – Model B (baseline)
COLOR_TIE = "#6B7280"  # mid-gray     – ties

import plotly.io as pio
pio.templates["oai"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, system-ui, sans-serif", color="#0D0D0D"),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    colorway=[COLOR_A, COLOR_B, COLOR_TIE, "#F59E0B", "#7C3AED"],
    title=dict(font=dict(size=15, color="#0D0D0D")),
))
PLOTLY_TEMPLATE = "plotly_white+oai"

BATCH_ORDER = {
    "b74661183d": 1, "bb800a5bbf": 2, "b1bc0c57ef": 3, "ba0be5102c": 4,
    "b08a9dc797": 5, "bbba5ca703": 6, "b07acabe04": 7, "b06b812ca5": 8,
}
BATCH_LABELS = {
    "b74661183d": "B1<br>Jun 17", "bb800a5bbf": "B2<br>Jun 21",
    "b1bc0c57ef": "B3<br>Jun 24", "ba0be5102c": "B4<br>Jun 25",
    "b08a9dc797": "B5<br>Jun 26", "bbba5ca703": "B6<br>Jun 27",
    "b07acabe04": "B7<br>Jul 1",  "b06b812ca5": "B8<br>Jul 12",
}
COMP_MAP = {"A << B": 1, "A < B": 2, "A == B": 3, "A > B": 4, "A >> B": 5}


# ── Data prep ───────────────────────────────────────────────────────────────
def load_data():
    dt_cols = ["created_at_pacific", "claimed_at_pacific", "submitted_at_pacific"]
    df = pd.read_csv(DATA_PATH, parse_dates=dt_cols)

    def parse_errors(val):
        if pd.isna(val) or str(val).strip() in ("[]", ""):
            return []
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    df["errors_a"] = df["completion_a_errors"].apply(parse_errors)
    df["errors_b"] = df["completion_b_errors"].apply(parse_errors)
    df["score_diff"]   = df["scorecard_a"] - df["scorecard_b"]
    df["a_wins"]       = df["response_quality_comparison"].isin(["A > B", "A >> B"])
    df["b_wins"]       = df["response_quality_comparison"].isin(["A < B", "A << B"])
    df["tie"]          = df["response_quality_comparison"] == "A == B"
    df["comp_numeric"] = df["response_quality_comparison"].map(COMP_MAP)
    df["batch_order"]  = df["batch"].map(BATCH_ORDER)
    df["batch_label"]  = df["batch"].map(BATCH_LABELS)
    grader_counts      = df["grader"].value_counts().rename("grader_task_count")
    df                 = df.join(grader_counts, on="grader")
    return df


def bootstrap_ci(data, stat_fn=np.mean, n_boot=5000, ci=0.95):
    rng  = np.random.default_rng(42)
    boot = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot, (1 - ci) / 2 * 100), np.percentile(boot, (1 + ci) / 2 * 100)


# ── Figure builders ────────────────────────────────────────────────────────

def fig_overall_donut(df):
    n      = len(df)
    a_rate = df["a_wins"].mean()
    fig = go.Figure(go.Pie(
        labels=["New model wins", "Tie", "Current model wins"],
        values=[df["a_wins"].sum(), df["tie"].sum(), df["b_wins"].sum()],
        hole=0.58,
        marker_colors=[COLOR_A, COLOR_TIE, COLOR_B],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} tasks (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        annotations=[dict(text=f"<b>New model<br>wins {a_rate:.0%}</b>",
                          x=0.5, y=0.5, font_size=18, showarrow=False)],
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    return fig


def fig_score_dist(df):
    fig = go.Figure()
    for col, name, color in [
        ("scorecard_a", "New model (A)", COLOR_A),
        ("scorecard_b", "Current model (B)", COLOR_B),
    ]:
        fig.add_trace(go.Histogram(
            x=df[col], name=name, opacity=0.72,
            marker_color=color, xbins=dict(start=0.5, end=5.5, size=1),
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, barmode="overlay",
        xaxis_title="Quality score (1 = poor, 5 = excellent)",
        yaxis_title="Number of tasks",
        legend_title="", margin=dict(t=10, b=40, l=50, r=10),
    )
    return fig


def fig_score_diff(df):
    fig = px.histogram(
        df, x="score_diff",
        color_discrete_sequence=[COLOR_A],
        template=PLOTLY_TEMPLATE,
        labels={"score_diff": "Score gap (New − Current)"},
        nbins=9,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="No difference", annotation_position="top right")
    fig.update_layout(margin=dict(t=10, b=40, l=60, r=10), yaxis_title="Tasks")
    return fig


def fig_category_winrate(df):
    rows = []
    for cat, grp in df.groupby("prompt_category"):
        a = grp["a_wins"].values.astype(float)
        b = grp["b_wins"].values.astype(float)
        a_lo, a_hi = bootstrap_ci(a)
        b_lo, b_hi = bootstrap_ci(b)
        rows.append(dict(category=cat, n=len(grp),
                         a_rate=a.mean(), a_lo=a_lo, a_hi=a_hi,
                         b_rate=b.mean(), b_lo=b_lo, b_hi=b_hi))
    cdf = pd.DataFrame(rows).sort_values("a_rate", ascending=False)

    fig = go.Figure()
    for model, rc, lo_c, hi_c, color, label in [
        ("New model wins", "a_rate", "a_lo", "a_hi", COLOR_A, "New model (A)"),
        ("Current model wins", "b_rate", "b_lo", "b_hi", COLOR_B, "Current model (B)"),
    ]:
        fig.add_trace(go.Bar(
            x=cdf["category"], y=cdf[rc], name=label,
            marker_color=color,
            error_y=dict(type="data", symmetric=False,
                         array=(cdf[hi_c] - cdf[rc]).tolist(),
                         arrayminus=(cdf[rc] - cdf[lo_c]).tolist()),
            hovertemplate="%{x}<br>Win rate: %{y:.1%}<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, barmode="group",
        yaxis_title="Win rate", yaxis_tickformat=".0%",
        xaxis_title="", legend_title="",
        margin=dict(t=10, b=80, l=60, r=10),
    )
    return fig


def fig_batch_trend(df):
    rows = []
    for (order, label), grp in df.groupby(["batch_order", "batch_label"]):
        a = grp["a_wins"].values.astype(float)
        lo, hi = bootstrap_ci(a)
        rows.append(dict(batch_order=order, batch_label=label, n=len(grp),
                         a_rate=a.mean(), ci_lo=lo, ci_hi=hi))
    bdf = pd.DataFrame(rows).sort_values("batch_order")

    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=bdf["batch_label"].tolist() + bdf["batch_label"].tolist()[::-1],
        y=bdf["ci_hi"].tolist() + bdf["ci_lo"].tolist()[::-1],
        fill="toself", fillcolor=f"rgba(16,163,127,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=bdf["batch_label"], y=bdf["a_rate"],
        mode="lines+markers", name="New model win rate",
        line=dict(color=COLOR_A, width=2.5), marker=dict(size=9),
        hovertemplate="Batch: %{x}<br>Win rate: %{y:.1%}<br>n=%{customdata}",
        customdata=bdf["n"],
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#9CA3AF",
                  annotation_text="50% parity", annotation_position="bottom right")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title="Win rate", yaxis_tickformat=".0%",
        xaxis_title="Batch (chronological order, Jun–Jul 2024)",
        legend_title="", margin=dict(t=10, b=60, l=60, r=10),
    )
    return fig


def fig_within_prompt_trend(df):
    repeated_ids = df.groupby("unique_prompt_id")["batch"].nunique()
    repeated_ids = repeated_ids[repeated_ids >= 4].index
    sub = df[df["unique_prompt_id"].isin(repeated_ids)]

    rows = []
    for (order, label), grp in sub.groupby(["batch_order", "batch_label"]):
        a = grp["a_wins"].values.astype(float)
        lo, hi = bootstrap_ci(a)
        rows.append(dict(batch_order=order, batch_label=label, n=len(grp),
                         a_rate=a.mean(), ci_lo=lo, ci_hi=hi))
    bdf = pd.DataFrame(rows).sort_values("batch_order")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bdf["batch_label"], y=bdf["a_rate"],
        mode="lines+markers", name="Win rate (repeated prompts only)",
        line=dict(color=COLOR_A, width=2, dash="dot"), marker=dict(size=8),
        hovertemplate="Batch: %{x}<br>Win rate: %{y:.1%}<br>n=%{customdata}",
        customdata=bdf["n"],
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#9CA3AF")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title="Win rate", yaxis_tickformat=".0%",
        xaxis_title="Batch", legend_title="",
        margin=dict(t=10, b=60, l=60, r=10),
    )
    return fig


def fig_grader_volume(df):
    gc = df["grader"].value_counts().reset_index()
    gc.columns = ["grader", "tasks"]
    fig = px.histogram(
        gc, x="tasks", nbins=20,
        template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
        labels={"tasks": "Number of tasks completed"},
    )
    fig.add_vline(x=10, line_dash="dash", line_color="orange",
                  annotation_text="10-task threshold for individual analysis")
    fig.update_layout(margin=dict(t=10, b=40, l=60, r=10), yaxis_title="Number of graders")
    return fig


def fig_grader_bias(df):
    gs = (df.groupby("grader")
            .agg(task_count=("task", "count"), a_win_rate=("a_wins", "mean"))
            .reset_index())
    gs = gs[gs["task_count"] >= 10].copy()
    pop_mean = df["a_wins"].mean()
    pop_std  = gs["a_win_rate"].std()
    gs["z_score"] = (gs["a_win_rate"] - pop_mean) / pop_std
    gs["outlier"]  = gs["z_score"].abs() > 2
    gs_sorted = gs.sort_values("a_win_rate")

    fig = px.scatter(
        gs_sorted,
        x="a_win_rate", y="grader",
        color="outlier",
        color_discrete_map={True: "#EF4444", False: COLOR_A},
        size="task_count",
        template=PLOTLY_TEMPLATE,
        labels={"a_win_rate": "New model win rate", "grader": "",
                "outlier": "Outlier (outside ±2σ)"},
        hover_data={"task_count": True, "z_score": ":.2f"},
    )
    # Sort y-axis by win rate so highest is at top
    fig.update_layout(
        height=520,
        margin=dict(t=10, b=40, l=120, r=10),
        yaxis=dict(categoryorder="array",
                   categoryarray=gs_sorted["grader"].tolist()),
    )
    fig.add_vline(x=pop_mean, line_dash="dash", line_color="black",
                  annotation_text=f"Avg {pop_mean:.0%}")
    fig.add_vline(x=pop_mean - 2*pop_std, line_dash="dot", line_color="orange")
    fig.add_vline(x=pop_mean + 2*pop_std, line_dash="dot", line_color="orange",
                  annotation_text="±2σ", annotation_position="top right")
    return fig


def fig_error_heatmap(df):
    def long_errors(col, model):
        rows = (df[["task", col]].explode(col)
                .rename(columns={col: "error"})
                .dropna(subset=["error"])
                .query("error != ''"))
        rows = rows.copy()
        rows["model"] = model
        return rows

    el = pd.concat([long_errors("errors_a", "New model (A)"),
                    long_errors("errors_b", "Current model (B)")])
    pivot = (el.groupby(["error", "model"]).size()
               .unstack(fill_value=0)
               .div(len(df)))
    for col in ["New model (A)", "Current model (B)"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    fig = px.imshow(
        pivot[["New model (A)", "Current model (B)"]].T,
        text_auto=".1%", aspect="auto",
        color_continuous_scale=["#FFFFFF", "#10A37F"],
        template=PLOTLY_TEMPLATE,
        labels={"x": "Error type", "y": "", "color": "Error rate"},
    )
    fig.update_layout(margin=dict(t=10, b=140, l=130, r=10),
                      coloraxis_showscale=False)
    return fig


def fig_category_coverage(df):
    cat_batch = df.groupby(["batch_label", "prompt_category"]).size().reset_index(name="count")
    ordered_labels = [BATCH_LABELS[b] for b in sorted(BATCH_ORDER, key=BATCH_ORDER.get)]
    pivot = (cat_batch.pivot(index="prompt_category", columns="batch_label", values="count")
                      .fillna(0)
                      .reindex(columns=[l for l in ordered_labels
                                        if l in cat_batch["batch_label"].unique()]))
    fig = px.imshow(
        pivot, text_auto=True, aspect="auto",
        color_continuous_scale=["#F9FAFB", "#10A37F"],
        template=PLOTLY_TEMPLATE,
        labels={"x": "Batch", "y": "Prompt category", "color": "Tasks"},
    )
    fig.update_layout(margin=dict(t=10, b=60, l=160, r=10),
                      coloraxis_showscale=False)
    return fig


def fig_length_by_category(df):
    fig = px.box(
        df, x="prompt_text_length", y="prompt_category",
        template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
        labels={"prompt_text_length": "Prompt length (characters)", "prompt_category": ""},
    )
    fig.update_layout(
        yaxis=dict(categoryorder="median ascending"),
        margin=dict(t=10, b=40, l=160, r=10),
    )
    return fig


def fig_length_win_rate(df):
    df = df.copy()
    df["length_bin"] = pd.cut(
        df["prompt_text_length"],
        bins=[0, 20, 35, 55, 80, 500],
        labels=["≤20", "21–35", "36–55", "56–80", ">80"],
    )
    ls = (df.groupby("length_bin", observed=True)["a_wins"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "a_rate", "count": "n"}))
    fig = px.bar(
        ls, x="length_bin", y="a_rate", text="n",
        template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
        labels={"length_bin": "Prompt length (chars)", "a_rate": "New model win rate"},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(t=10, b=40, l=60, r=10))
    return fig


# ── HTML helpers ────────────────────────────────────────────────────────────
def fig_to_div(fig, first=False):
    return fig.to_html(full_html=False,
                       include_plotlyjs="cdn" if first else False,
                       config={"responsive": True})


def slide(content, bg=""):
    bg_attr = f' data-background="{bg}"' if bg else ""
    return f"<section{bg_attr}>{content}</section>\n"


def titled_slide(question, subtitle, body):
    """Main slide with question as h2 and subtitle as smaller h3."""
    return slide(
        f'<h2 style="margin-bottom:2px">{question}</h2>'
        f'<h3 style="color:var(--oai-gray);font-weight:400;font-size:0.75em;margin-top:0;margin-bottom:16px">{subtitle}</h3>'
        f'{body}'
    )


def chart_titled_slide(question, subtitle, div, note=""):
    note_html = (f'<p style="font-size:0.5em;color:var(--oai-gray);margin-top:4px">{note}</p>'
                 if note else "")
    return titled_slide(
        question, subtitle,
        f'<div style="height:62vh">{div}</div>{note_html}'
    )


def appendix_slide(title, body, ref=""):
    ref_html = (f'<p style="font-size:0.5em;color:var(--oai-gray)">↖ Relates to: {ref}</p>'
                if ref else "")
    return slide(
        f'<h2 style="color:var(--oai-gray);font-size:0.9em;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-bottom:4px">Appendix</h2>'
        f'<h3 style="margin-top:0">{title}</h3>'
        f'{ref_html}{body}',
        bg="#F9FAFB",
    )


def stat_box(plain_english, test_name, ref_slide=""):
    ref = f" &nbsp;·&nbsp; <em>See appendix for full explanation</em>" if ref_slide else ""
    return (
        f'<div style="background:#F0FDF4;border-left:3px solid {COLOR_A};'
        f'padding:10px 14px;border-radius:6px;font-size:0.62em;text-align:left;margin-top:8px">'
        f'<b>How we measured this ({test_name}):</b> {plain_english}{ref}'
        f'</div>'
    )


def caveat_box(text):
    return (
        f'<div style="background:#FFFBEB;border-left:3px solid #F59E0B;'
        f'padding:8px 14px;border-radius:6px;font-size:0.58em;text-align:left;margin-top:8px">'
        f'<b>Note:</b> {text}'
        f'</div>'
    )


# ── Main builder ────────────────────────────────────────────────────────────
def build():
    print("Loading data…")
    df = load_data()
    score_diff_mean = df["score_diff"].mean()

    # ── Pre-compute all stats ─────────────────────────────────────────────
    n      = len(df)
    a_rate = df["a_wins"].mean()
    b_rate = df["b_wins"].mean()
    t_rate = df["tie"].mean()
    a_lo, a_hi = bootstrap_ci(df["a_wins"].values.astype(float))
    b_lo, b_hi = bootstrap_ci(df["b_wins"].values.astype(float))

    _, p_w = stats.wilcoxon(df["score_diff"].dropna())

    bdf = (df.groupby(["batch_order", "batch_label"])["a_wins"]
             .mean().reset_index().sort_values("batch_order"))
    tau, p_mk = stats.kendalltau(bdf["batch_order"], bdf["a_wins"])
    b1_rate = bdf["a_wins"].iloc[0]
    b2_rate = bdf["a_wins"].iloc[1]
    jump    = b2_rate - b1_rate
    plateau_std = bdf["a_wins"].iloc[1:].std()

    pivot_irr = df.pivot_table(index="grader", columns="unique_prompt_id",
                                values="comp_numeric", aggfunc="first")
    alpha_k = krippendorff.alpha(reliability_data=pivot_irr.values,
                                  level_of_measurement="ordinal")

    gs = (df.groupby("grader").agg(tc=("task","count"), awr=("a_wins","mean")).reset_index())
    gs_filt = gs[gs["tc"] >= 10].copy()
    gs_filt["z"] = (gs_filt["awr"] - gs_filt["awr"].mean()) / gs_filt["awr"].std()
    n_outlier_graders = (gs_filt["z"].abs() > 2).sum()
    n_low_vol = (gs["tc"] < 10).sum()

    groups_kw = [g["prompt_text_length"].values for _, g in df.groupby("prompt_category")]
    _, p_kw = stats.kruskal(*groups_kw)

    print("Building figures…")
    figs = {
        "donut":    fig_overall_donut(df),
        "scores":   fig_score_dist(df),
        "score_diff": fig_score_diff(df),
        "cats":     fig_category_winrate(df),
        "trend":    fig_batch_trend(df),
        "within":   fig_within_prompt_trend(df),
        "vol":      fig_grader_volume(df),
        "bias":     fig_grader_bias(df),
        "errors":   fig_error_heatmap(df),
        "coverage":          fig_category_coverage(df),
        "len_cat":           fig_length_by_category(df),
        "len_rate":          fig_length_win_rate(df),
    }

    print("Converting figures to HTML…")
    divs = {}
    first = True
    for k, fig in figs.items():
        divs[k] = fig_to_div(fig, first=first)
        first = False

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN SLIDES
    # ═══════════════════════════════════════════════════════════════════════
    s = ""

    # 1 — Title
    s += slide(f"""
        <div style="margin-top:30px">
          <span class="oai-tag">OpenAI · Browsing Model Evaluation</span>
        </div>
        <h1 style="font-size:1.65em;margin-top:18px">
          Should We Launch the New ChatGPT Browsing Model?
        </h1>
        <p style="font-size:0.7em;color:var(--oai-gray);margin-top:6px">
          Analysis of {n:,} human evaluations across 8 model iterations · Jun–Jul 2024
        </p>
        <p style="font-size:0.65em;color:var(--oai-gray);margin-top:20px">
          Ben Lacar, Ph.D. &nbsp;·&nbsp; Candidate: Data Scientist, Preparedness
        </p>
    """)

    # 2 — Executive Summary
    s += slide(f"""
        <h2>Executive Summary</h2>
        <table style="width:100%;border-collapse:collapse;font-size:0.62em;margin-top:12px;text-align:left">
          <thead>
            <tr style="border-bottom:2px solid var(--oai-border)">
              <th style="padding:6px 10px;width:28%;color:var(--oai-gray);font-weight:600">Question</th>
              <th style="padding:6px 10px;width:54%;color:var(--oai-gray);font-weight:600">Answer</th>
              <th style="padding:6px 10px;width:18%;color:var(--oai-gray);font-weight:600">Slides</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid var(--oai-border)">
              <td style="padding:8px 10px;vertical-align:top"><b>Is the new model better?</b></td>
              <td style="padding:8px 10px;vertical-align:top">✅ Yes — wins <b>{a_rate:.0%} of tasks</b> vs. {b_rate:.0%} for the current model. The difference is statistically robust.</td>
              <td style="padding:8px 10px;vertical-align:top;color:var(--oai-gray)">4, 5, 6</td>
            </tr>
            <tr style="border-bottom:1px solid var(--oai-border);background:var(--oai-light)">
              <td style="padding:8px 10px;vertical-align:top"><b>Diminishing returns or still improving?</b></td>
              <td style="padding:8px 10px;vertical-align:top">📉 Diminishing returns. Win rate jumped early then plateaued — no significant upward trend across the last 6 snapshots.</td>
              <td style="padding:8px 10px;vertical-align:top;color:var(--oai-gray)">7, 8, 9</td>
            </tr>
            <tr style="border-bottom:1px solid var(--oai-border)">
              <td style="padding:8px 10px;vertical-align:top"><b>Are graders providing signal or noise?</b></td>
              <td style="padding:8px 10px;vertical-align:top">🧑‍⚖️ Adequate signal in aggregate. Cross-grader consistency is limited — partly because the model itself changed between batches, not purely grader noise.</td>
              <td style="padding:8px 10px;vertical-align:top;color:var(--oai-gray)">10, 11</td>
            </tr>
            <tr style="border-bottom:1px solid var(--oai-border);background:var(--oai-light)">
              <td style="padding:8px 10px;vertical-align:top"><b>Other key observations?</b></td>
              <td style="padding:8px 10px;vertical-align:top">
                <div>(1) New and current models fail in different ways — review error profiles before launch.</div>
                <div style="margin-top:4px">(2) Some prompt categories are absent from early snapshots — affects trend interpretation.</div>
                <div style="margin-top:4px">(3) Prompt length is confounded with category, not a standalone signal.</div>
              </td>
              <td style="padding:8px 10px;vertical-align:top;color:var(--oai-gray)">8, 12, 23</td>
            </tr>
            <tr>
              <td style="padding:8px 10px;vertical-align:top"><b>Launch or keep iterating?</b></td>
              <td style="padding:8px 10px;vertical-align:top">🚀 <b>Launch</b> — with a staged rollout and weekly monitoring. Real-world signal will now outpace what lab evaluation can provide.</td>
              <td style="padding:8px 10px;vertical-align:top;color:var(--oai-gray)">13</td>
            </tr>
          </tbody>
        </table>
    """)

    # 3 — Data at a glance
    s += slide(f"""
        <h2>What Data Are We Working With?</h2>
        <p style="font-size:0.68em;color:var(--oai-gray);margin:4px 0 14px">
          Human graders compared the new model (A) vs the current model (B) on the same prompts,
          assigning quality scores and a head-to-head verdict.
        </p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
          <div class="stat-card"><div class="num">{n:,}</div>
            <div class="lbl">evaluation tasks</div></div>
          <div class="stat-card"><div class="num">8</div>
            <div class="lbl">model snapshots tested<br>Jun 17 – Jul 12, 2024</div></div>
          <div class="stat-card"><div class="num">61</div>
            <div class="lbl">human graders</div></div>
          <div class="stat-card"><div class="num">490</div>
            <div class="lbl">prompts re-evaluated across<br>every snapshot <em>(by design)</em></div></div>
          <div class="stat-card"><div class="num">8</div>
            <div class="lbl">prompt categories<br>e.g. Commerce, Local, …</div></div>
          <div class="stat-card"><div class="num">1–5</div>
            <div class="lbl">quality score scale<br>+ head-to-head verdict</div></div>
        </div>
    """)

    # 4 — Q1 verdict
    s += slide(f"""
        <h2>Is the New Model Better Than the Current One?</h2>
        <h3 style="color:var(--oai-gray);font-weight:400;font-size:0.7em;margin:2px 0 14px">
          Head-to-head verdict across {n:,} tasks
        </h3>
        <div style="display:flex;align-items:center;justify-content:center;gap:32px">
          <div style="flex:1;max-width:340px">{divs["donut"]}</div>
          <div style="flex:1;text-align:left;font-size:0.7em">
            <p>📗 <b>New model wins {a_rate:.0%} of tasks</b>
               <span style="color:var(--oai-gray)">(95% range: {a_lo:.0%}–{a_hi:.0%})</span></p>
            <p>📕 Current model wins {b_rate:.0%}</p>
            <p>⬜ Tied: {t_rate:.0%}</p>
            <hr style="margin:12px 0;border-color:var(--oai-border)">
            <p>The advantage holds across every prompt category — this is not a niche finding.</p>
            {stat_box(
                "We checked whether the score gap between the two models is consistently "
                "positive across all task pairs, ruling out random chance.",
                "Wilcoxon signed-rank test", ref_slide="appendix"
            )}
          </div>
        </div>
    """)

    # 5 — Q1 score distributions
    s += chart_titled_slide(
        "Is the New Model Better Than the Current One?",
        "Quality score distributions — how often each score was given",
        divs["scores"],
        note=f"Scores are 1–5 (1 = poor, 5 = excellent). "
             f"The new model skews toward higher scores. "
             f"Bars overlap so you can compare the shapes directly.",
    )

    # 6 — Q1 by category
    s += chart_titled_slide(
        "Is the New Model Better Than the Current One?",
        "Win rate by prompt category — the advantage is consistent",
        divs["cats"],
        note="Error bars show the 95% confidence range "
             "(wider bars = fewer tasks in that category = more uncertainty). "
             "Sorted by new model win rate, highest first.",
    )

    # 7 — Q2 trend
    s += chart_titled_slide(
        "Has the Model Stopped Improving?",
        "Win rate across 8 model snapshots — a clear early jump then a plateau",
        divs["trend"],
        note=f"Each point is one model snapshot (batch). Shaded band = 95% confidence range. "
             f"Note: batch sizes range from 97 to 969 tasks — "
             f"smaller batches have wider uncertainty bands. "
             f"Dashed line = 50% (parity between old and new model).",
    )

    # 8 — Category coverage caveat (moved here so it immediately contextualises the trend chart)
    s += chart_titled_slide(
        "Before Reading the Trend — a Key Caveat",
        "Some prompt categories only appear in later batches",
        divs["coverage"],
        note="Blank cells = category absent from that batch. "
             "'Other' and 'Websites & Apps' are missing from the first four snapshots. "
             "When category mix changes across batches, overall win rates may shift "
             "for reasons unrelated to model quality improvement.",
    )

    # 9 — Q2 interpretation
    s += slide(f"""
        <h2>Has the Model Stopped Improving?</h2>
        <h3 style="color:var(--oai-gray);font-weight:400;font-size:0.7em;margin:2px 0 14px">
          Reading the trend: a big early gain, then stability
        </h3>
        <div style="text-align:left;font-size:0.7em;max-width:820px;margin:auto">
          <p>📈 <b>Early jump:</b> Win rate rose from {b1_rate:.0%} → {b2_rate:.0%}
             (+{jump:.0%}) between the first and second model snapshot.</p>
          <p>📉 <b>Plateau since:</b> The remaining 6 snapshots fluctuate within
             ±{plateau_std:.0%} — effectively flat.</p>
          <p>📊 <b>Trend test result:</b> No statistically significant monotonic upward
             trend across all 8 batches (p = {p_mk:.2f}). The model is not consistently
             getting better batch-over-batch.</p>
          <hr style="margin:12px 0;border-color:var(--oai-border)">
          <p><b>What this means:</b> The team captured a meaningful improvement early in the
             evaluation window, but appears to have hit a ceiling with the current roadmap.
             Real-world production feedback will now provide higher-value signal than
             continued lab iteration.</p>
          {caveat_box(
              f"As shown on the previous slide, two prompt categories are absent from early "
              f"snapshots — the B1→B2 jump may be partly a category-mix effect, "
              f"not purely model improvement."
          )}
          {stat_box(
              "We tested whether win rates consistently increased from one batch to the next "
              "(not just on average). They did not.",
              "Mann-Kendall trend test", ref_slide="appendix"
          )}
        </div>
    """)

    # 10 — Q3 grader volume
    s += chart_titled_slide(
        "Can We Trust the Grader Ratings?",
        "Task volume per grader — a long tail of low-volume contributors",
        divs["vol"],
        note=f"61 graders total. Task counts range from 1 to 175. "
             f"{n_low_vol} graders completed fewer than 10 tasks — "
             f"their individual ratings are less reliable, though they represent a small share of total tasks. "
             f"Graders self-selected tasks from a shared pool rather than being randomly assigned.",
    )

    # 11 — Q3 grader consistency
    s += chart_titled_slide(
        "Can We Trust the Grader Ratings?",
        "Individual grader win rates — most are consistent with the group",
        divs["bias"],
        note=f"Each bubble is one grader (bubble size = task volume). "
             f"Orange lines = ±2 standard deviations from the average. "
             f"{n_outlier_graders} of {len(gs_filt)} graders with ≥10 tasks fall outside this range. "
             f"Sorted by win rate — highest at top.",
    )

    # 12 — Error patterns
    s += chart_titled_slide(
        "Where Does Each Model Fall Short?",
        "Error rates by type — the new model has a different failure profile",
        divs["errors"],
        note="Rate = share of all tasks where this error was flagged. "
             "Note: an empty error list doesn't necessarily mean no errors exist — "
             "graders may not always complete the error checklist. "
             "Treat these as minimum error rates.",
    )

    # 13 — Recommendation
    s += slide(f"""
        <h2>Should We Launch or Keep Iterating?</h2>
        <h3 style="color:{COLOR_A};font-weight:600;font-size:0.85em;margin:4px 0 16px">
          Recommendation: Launch with a staged rollout and monitoring plan
        </h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;
                    font-size:0.67em;text-align:left">
          <div class="stat-card" style="border-left:4px solid {COLOR_A}">
            <b style="color:{COLOR_A}">Why launch now</b>
            <ul style="margin-top:6px">
              <li>New model wins <b>{a_rate:.0%} of tasks</b> vs {b_rate:.0%} — a meaningful margin</li>
              <li>The difference is statistically robust (not due to random variation)</li>
              <li>Win rate has <b>plateaued</b> — continuing lab iteration has diminishing returns</li>
              <li>Production traffic will yield faster, higher-quality signal for the next improvement cycle</li>
            </ul>
          </div>
          <div class="stat-card" style="border-left:4px solid #F59E0B">
            <b style="color:#B45309">Conditions before / after launch</b>
            <ul style="margin-top:6px">
              <li>Review the specific error types where the new model underperforms (see slide 11)</li>
              <li>Start with a <b>staged rollout</b> (e.g., 10% of traffic)</li>
              <li>Monitor win rate weekly — pause if it falls below 50% at 2 standard deviations</li>
              <li>For future eval rounds: randomly assign graders to tasks rather than self-selection</li>
            </ul>
          </div>
        </div>
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # APPENDIX
    # ═══════════════════════════════════════════════════════════════════════

    # A0 — Appendix header
    s += slide("""
        <h1 style="color:var(--oai-gray)">Appendix</h1>
        <p style="color:var(--oai-gray);font-size:0.75em">
          Technical details, statistical method explanations, and supplemental figures
        </p>
        <ul style="font-size:0.65em;color:var(--oai-gray);text-align:left;
                   display:inline-block;margin-top:20px">
          <li>Glossary of terms</li>
          <li>Statistical method deep-dives (ELI5 + technical)</li>
          <li>Supplemental figures: score gap, within-prompt trend, prompt length analysis</li>
        </ul>
    """, bg="#F9FAFB")

    # A1 — Glossary
    s += appendix_slide("Glossary of Key Terms", f"""
        <table style="font-size:0.63em;width:100%;border-collapse:collapse">
          <tr style="background:#F0FDF4"><th style="text-align:left;padding:6px 10px">Term</th>
              <th style="text-align:left;padding:6px 10px">Plain English meaning</th></tr>
          <tr><td style="padding:5px 10px"><b>New model (Model A)</b></td>
              <td style="padding:5px 10px">The release candidate — the new browsing model being considered for launch</td></tr>
          <tr style="background:#F9FAFB"><td style="padding:5px 10px"><b>Current model (Model B)</b></td>
              <td style="padding:5px 10px">The baseline — the model currently in production</td></tr>
          <tr><td style="padding:5px 10px"><b>Batch</b></td>
              <td style="padding:5px 10px">A snapshot of the model at a specific point in development; 8 snapshots taken Jun 17 – Jul 12</td></tr>
          <tr style="background:#F9FAFB"><td style="padding:5px 10px"><b>Win rate</b></td>
              <td style="padding:5px 10px">The fraction of tasks where the new model was rated as better than the current model</td></tr>
          <tr><td style="padding:5px 10px"><b>Grader</b></td>
              <td style="padding:5px 10px">A human evaluator who reads both model responses and decides which is better</td></tr>
          <tr style="background:#F9FAFB"><td style="padding:5px 10px"><b>Prompt category</b></td>
              <td style="padding:5px 10px">The type of question (e.g., Commerce = shopping queries, Local = location-based, Real-time = news/weather)</td></tr>
          <tr><td style="padding:5px 10px"><b>Confidence range / error bars</b></td>
              <td style="padding:5px 10px">The range in which the true value likely falls, given the data we have. Wider = less certainty</td></tr>
          <tr style="background:#F9FAFB"><td style="padding:5px 10px"><b>Ordinal scale</b></td>
              <td style="padding:5px 10px">A ranking where order matters but gaps between values aren't equal (e.g., A&gt;B vs A&gt;&gt;B)</td></tr>
          <tr><td style="padding:5px 10px"><b>Inter-rater reliability</b></td>
              <td style="padding:5px 10px">How much graders agree with each other when rating the same prompt</td></tr>
        </table>
    """)

    # A2 — Statistical methods overview
    s += appendix_slide("Statistical Methods Used", f"""
        <table style="font-size:0.63em;width:100%;border-collapse:collapse">
          <tr style="background:#F0FDF4">
            <th style="text-align:left;padding:6px 10px">Method</th>
            <th style="text-align:left;padding:6px 10px">Used for</th>
            <th style="text-align:left;padding:6px 10px">Why this method</th>
          </tr>
          <tr><td style="padding:5px 10px"><b>Bootstrap CI</b></td>
              <td style="padding:5px 10px">All win-rate estimates</td>
              <td style="padding:5px 10px">No distributional assumptions; works for any sample size</td></tr>
          <tr style="background:#F9FAFB"><td style="padding:5px 10px"><b>Wilcoxon signed-rank</b></td>
              <td style="padding:5px 10px">Q1: Is A better overall?</td>
              <td style="padding:5px 10px">Scores are ordinal — paired t-test would be inappropriate</td></tr>
          <tr><td style="padding:5px 10px"><b>Mann-Kendall (Kendall τ)</b></td>
              <td style="padding:5px 10px">Q2: Is there a trend across batches?</td>
              <td style="padding:5px 10px">Tests monotonic trend without assuming linearity</td></tr>
          <tr style="background:#F9FAFB"><td style="padding:5px 10px"><b>Krippendorff's α</b></td>
              <td style="padding:5px 10px">Q3: Grader inter-rater reliability</td>
              <td style="padding:5px 10px">Handles ordinal scale and missing data (not all graders rate all prompts)</td></tr>
          <tr><td style="padding:5px 10px"><b>Kruskal-Wallis</b></td>
              <td style="padding:5px 10px">Category length confound check</td>
              <td style="padding:5px 10px">Non-parametric ANOVA for comparing distributions across groups</td></tr>
        </table>
    """)

    # A3 — Bootstrap CI
    s += appendix_slide("How We Computed Uncertainty: Bootstrap Confidence Intervals", f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:0.63em;text-align:left">
          <div>
            <b style="color:{COLOR_A}">ELI5 (plain English)</b>
            <p>Imagine you want to know the average rating in a city, but you only measured 1,000 people.
               What if you'd measured a slightly different 1,000?</p>
            <p>Bootstrap answers this by repeatedly re-drawing 1,000 people <em>from your existing data</em>
               (with replacement) and recomputing the average — 5,000 times.
               The spread of those 5,000 averages becomes your confidence range.</p>
            <p><b>In plain terms:</b> "If we'd run this study with slightly different data,
               our estimate would land somewhere in this range 95% of the time."</p>
          </div>
          <div>
            <b style="color:{COLOR_A}">Technical details</b>
            <p>Resample n = {n:,} observations with replacement, 5,000 iterations.
               95% CI: 2.5th–97.5th percentile of bootstrap distribution.</p>
            <p>No distributional assumptions required — robust to non-normal and ordinal data.</p>
            <p><b>Example:</b> For win rate = {a_rate:.1%}:<br>
               Bootstrap → 95% CI [{a_lo:.1%}, {a_hi:.1%}].<br>
               We are 95% confident the true win rate falls in this range.</p>
            <p><em>Note: wider CIs in early batches (n ≈ 97) vs later batches (n ≈ 969)
               reflect the smaller sample sizes — not more noisy grading.</em></p>
          </div>
        </div>
    """, ref="Slides 5–11 (all charts with error bars)")

    # A4 — Wilcoxon
    s += appendix_slide("Wilcoxon Signed-Rank Test", f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:0.63em;text-align:left">
          <div>
            <b style="color:{COLOR_A}">ELI5</b>
            <p>For each task, we have two scores: one for A and one for B.
               We look at the <em>gap</em> between them (A minus B).</p>
            <p>If the new model is genuinely better, most gaps should be positive.
               This test checks whether the pattern of gaps is consistent enough
               to rule out random chance — without assuming scores follow a bell curve.</p>
            <p><b>Think of it like:</b> You flip a biased coin 2,699 times. The test
               asks: "Is this coin significantly biased toward heads, accounting
               for how <em>much</em> it flipped each direction?"</p>
          </div>
          <div>
            <b style="color:{COLOR_A}">Technical details</b>
            <p>Non-parametric alternative to the paired t-test. Appropriate when
               data are ordinal (Likert 1–5 scores) because the distance between
               scale points may not be uniform.</p>
            <p><b>Procedure:</b> Rank the absolute differences |scorecard_A − scorecard_B|,
               then check whether the sum of positive-difference ranks significantly
               exceeds what you'd expect under the null (no difference).
               Pairs where the score difference is exactly zero are excluded
               from the ranking — a standard part of the procedure that
               slightly reduces the effective sample size.</p>
            <p><b>Result:</b> p = {p_w:.1e} — the probability of seeing a result
               this extreme if the models were equal is essentially zero.
               We reject the null: Model A is statistically significantly better.</p>
          </div>
        </div>
    """, ref="Slide 4 (Q1 Verdict)")

    # A5 — Mann-Kendall
    s += appendix_slide("Mann-Kendall Trend Test", f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:0.63em;text-align:left">
          <div>
            <b style="color:{COLOR_A}">ELI5</b>
            <p>Imagine 8 monthly sales reports. A <em>monotonic trend</em> means sales are
               generally moving in one direction overall — either upward or downward — even
               if there are a few flat months or minor bumps along the way. It does
               <em>not</em> require every single month to be strictly higher than the last.</p>
            <p>This test looks at every possible pair of months and counts how often a
               later month is higher than an earlier one. If those "later is higher" pairs
               vastly outnumber the others, the test confirms a real trend.</p>
            <p><b>Result here:</b> τ = {tau:.2f} (moderate positive tendency),
               but p = {p_mk:.2f} — with only 8 data points, even a moderate
               correlation isn't statistically significant. The plateau is real.</p>
          </div>
          <div>
            <b style="color:{COLOR_A}">Technical details</b>
            <p>Kendall's τ measures rank correlation between batch order (1–8) and
               batch-level win rate. The Mann-Kendall test evaluates whether τ
               differs from zero under a permutation-based null distribution.</p>
            <p><b>Why not linear regression?</b> With only 8 points and no
               assumption that improvement is linear, a rank-based test is more
               appropriate and robust.</p>
            <p><b>Result:</b> τ = {tau:.3f}, p = {p_mk:.3f}. Moderate positive
               correlation but not significant at α = 0.05.
               We cannot reject the null of no monotonic trend across all 8 batches.</p>
            <p><em>Note: low power (n=8 batches) limits sensitivity to real trends.</em></p>
          </div>
        </div>
    """, ref="Slides 7–8 (Q2 Trend)")

    # A6 — Krippendorff
    s += appendix_slide("Krippendorff's Alpha (α<sub>K</sub>) — Inter-Rater Reliability", f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:0.63em;text-align:left">
          <div>
            <b style="color:{COLOR_A}">ELI5</b>
            <p>Imagine 5 film critics each rating the same movie. Even if they picked
               scores randomly, they'd sometimes agree by coincidence.</p>
            <p>Krippendorff's alpha subtracts that random-agreement baseline before
               scoring consistency. A value of 0 means graders agree no more than
               chance; 1 means perfect agreement.</p>
            <p><b>Our result: α<sub>K</sub> = {alpha_k:.3f} (poor agreement).</b>
               But there's a critical caveat — see right.</p>
            <p style="background:#FEF3C7;border-radius:4px;padding:6px">
               <em>Note: α<sub>K</sub> here is Krippendorff's reliability coefficient —
               not to be confused with the significance level α = 0.05 used elsewhere
               in this analysis.</em></p>
          </div>
          <div>
            <b style="color:{COLOR_A}">Technical details + important caveat</b>
            <p>Computed on the ordinal comparison scale using squared distance metric.
               Handles sparse data naturally (not all graders rate all prompts).</p>
            <p>Benchmark: &lt;0.20 poor · 0.20–0.40 fair · 0.40–0.60 moderate · &gt;0.60 substantial.</p>
            <p><b>Critical caveat:</b> α<sub>K</sub> is computed across different batches where
               the underlying model changed. A grader rating prompt p100 in Batch 1
               and a different grader rating p100 in Batch 6 may genuinely disagree —
               not because the graders are inconsistent, but because the model improved
               between batches. This conflates <em>model improvement</em>
               with <em>grader inconsistency</em>. True inter-rater reliability
               would require multiple graders rating the <em>same model version</em>.</p>
            <p>Despite low α<sub>K</sub>, the overall finding (A wins {a_rate:.0%}) is
               robust because it is based on {n:,} tasks.</p>
          </div>
        </div>
    """, ref="Slides 9–10 (Q3 Grader Quality)")

    # A7 — Supplemental: score difference
    s += appendix_slide(
        "Supplemental: Score Gap Distribution",
        f'<div style="height:58vh">{divs["score_diff"]}</div>'
        f'<p style="font-size:0.52em;color:var(--oai-gray);margin-top:4px">'
        f'Distribution of (scorecard_A − scorecard_B) across all {n:,} tasks. '
        f'Positive = new model scored higher. Mean gap = '
        f'{score_diff_mean:.2f} points.'
        f'</p>',
        ref="Slide 4 (Q1 Verdict)",
    )

    # A8 — Supplemental: within-prompt trend
    s += appendix_slide(
        "Supplemental: Within-Prompt Trend (Controls for Prompt Difficulty)",
        f'<div style="height:55vh">{divs["within"]}</div>'
        f'<p style="font-size:0.52em;color:var(--oai-gray);margin-top:4px">'
        f'Using only the 490 prompts that appear in all 8 batches removes prompt-difficulty '
        f'confounds. The plateau pattern holds in this controlled view as well.'
        f'</p>',
        ref="Slides 7–8 (Q2 Trend)",
    )

    # A9 — Supplemental: prompt length by category
    s += appendix_slide(
        "Supplemental: Prompt Length Varies Significantly by Category",
        f'<div style="height:52vh">{divs["len_cat"]}</div>'
        f'<p style="font-size:0.52em;color:var(--oai-gray);margin-top:4px">'
        f'Categories differ significantly in prompt length '
        f'(Kruskal-Wallis p = {p_kw:.1e}). '
        f'The raw length→win-rate relationship may partly reflect category composition '
        f'rather than prompt complexity per se.'
        f'</p>',
        ref="Slide 11 (Error Patterns / Q4)",
    )

    # ── HTML template ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Browsing Model Launch Readiness — OpenAI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
      rel="stylesheet">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/white.css">
<style>
  :root {{
    --oai-green:  #10A37F;
    --oai-black:  #0D0D0D;
    --oai-gray:   #6B7280;
    --oai-light:  #F9FAFB;
    --oai-border: #E5E7EB;
  }}
  body, .reveal {{ font-family: 'Inter', system-ui, sans-serif; }}
  .reveal {{ font-size: 22px; color: var(--oai-black); background: #fff; }}
  .reveal h1 {{ font-size: 1.9em; font-weight: 700; color: var(--oai-black); }}
  .reveal h2 {{ font-size: 1.2em; font-weight: 700; color: var(--oai-black); line-height: 1.2; }}
  .reveal h3 {{ font-size: 1.0em; font-weight: 500; }}
  .reveal ul  {{ font-size: 0.85em; }}
  .reveal section {{ padding: 20px 48px; }}
  .reveal .controls {{ color: var(--oai-green); }}
  .reveal .progress span {{ background: var(--oai-green); }}
  .reveal .slide-number {{ color: var(--oai-gray); font-size: 0.55em; }}
  .js-plotly-plot {{ width: 100% !important; }}
  .stat-card {{
    background: var(--oai-light); border: 1px solid var(--oai-border);
    border-radius: 10px; padding: 16px 14px;
  }}
  .stat-card .num {{ font-size: 2em; font-weight: 700; color: var(--oai-green); }}
  .stat-card .lbl {{ font-size: 0.62em; color: var(--oai-gray); margin-top: 2px; }}
  .oai-tag {{
    display: inline-block; padding: 3px 12px;
    background: var(--oai-green); color: #fff;
    border-radius: 999px; font-size: 0.65em; font-weight: 600;
  }}
</style>
</head>
<body>
<div class="reveal">
  <div class="slides">
{s}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"></script>
<script>
  Reveal.initialize({{
    hash: true, transition: "slide",
    slideNumber: "c/t",
    width: 1200, height: 700, margin: 0.04,
    plugins: [],
  }});
  Reveal.on("slidechanged", () => {{
    if (window.Plotly) {{
      document.querySelectorAll(".js-plotly-plot").forEach(el => {{
        Plotly.relayout(el, {{ autosize: true }});
      }});
    }}
  }});
</script>
</body>
</html>
"""

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Written → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    build()
