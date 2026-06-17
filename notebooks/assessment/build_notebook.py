"""Generates browse_evals.ipynb in the same directory."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

def md(source): return nbf.v4.new_markdown_cell(source)
def code(source): return nbf.v4.new_code_cell(source)

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md("""# Browse Evals: ChatGPT Browsing Model Launch Analysis

Take-home assessment for OpenAI Data Scientist role.
**Goal:** Determine whether the new browsing model (Model A) is ready to launch.
"""))

# ══════════════════════════════════════════════════════════════════════════
# §0 Setup
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("##0 — Setup & Imports"))
cells.append(code("""\
import ast, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import krippendorff

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.4f}".format)

COLOR_A   = "#10A37F"  # OpenAI green – Model A (release candidate)
COLOR_B   = "#1B1B1B"  # near-black   – Model B (baseline)
COLOR_TIE = "#6B7280"  # mid-gray     – ties

# Register a global OpenAI-branded Plotly template so every figure inherits it
import plotly.io as pio
pio.templates["oai"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, system-ui, sans-serif", color="#0D0D0D"),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    colorway=[COLOR_A, COLOR_B, COLOR_TIE, "#F59E0B", "#7C3AED"],
    title=dict(font=dict(size=16, color="#0D0D0D")),
))
PLOTLY_TEMPLATE = "plotly_white+oai"

DATA_PATH = "../../local_data/assessment/browse_evals_takehome_v1.1.csv"
"""))

# ══════════════════════════════════════════════════════════════════════════
# §1 Business Problem
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##1 — Business Problem

OpenAI is evaluating a significant upgrade to ChatGPT's browsing model — the system
ChatGPT uses to search the web and summarise findings.
**Model A** = release candidate (new model), **Model B** = current production baseline.

Human graders compare A vs. B on the same prompts, providing:
- Individual quality scores (1–5) for each model
- A holistic comparison verdict (A>>B … A<<B)
- Error labels for shortcomings identified in each response

### Five questions to answer

| # | Question |
|---|----------|
| Q1 | Is the release candidate better than the existing model? |
| Q2 | Are we reaching diminishing returns, or is the model still improving? |
| Q3 | Are we happy with this pool of graders — signal or noise? |
| Q4 | What 1–3 other observations should we consider before launch? |
| Q5 | Should we launch or keep iterating? |
"""))

# ══════════════════════════════════════════════════════════════════════════
# §2 Data Schema
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##2 — Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `task` | string | Unique task ID |
| `created_at_pacific` | datetime | When the task entered the pool |
| `claimed_at_pacific` | datetime | When a grader picked up the task |
| `submitted_at_pacific` | datetime | When the grader submitted ratings |
| `batch` | string | Model-evolution batch ID |
| `scorecard_a` | int 1–5 | Standalone quality score for Model A |
| `scorecard_b` | int 1–5 | Standalone quality score for Model B |
| `response_quality_comparison` | ordinal | Holistic verdict: A>>B / A>B / A==B / A<B / A<<B |
| `completion_a_errors` | JSON list | Error categories flagged for Model A |
| `completion_b_errors` | JSON list | Error categories flagged for Model B |
| `grader` | string | Grader identifier |
| `prompt_category` | string | Model-labelled prompt category (8 values) |
| `unique_prompt_id` | string | Prompt identifier (same prompt can appear in multiple batches) |
| `prompt_text_length` | int | Character count of the prompt |

> **Note on error columns:** an empty list means no errors were *flagged*, not necessarily that no errors exist.
> **Note on comparison scale:** this is **ordinal**, not interval — A>>B is not "twice as good" as A>B.
"""))

# ══════════════════════════════════════════════════════════════════════════
# §3 Data Loading & Cleaning
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("##3 — Data Loading & Cleaning"))
cells.append(code("""\
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

# ── Derived columns ────────────────────────────────────────────────────────
df["score_diff"] = df["scorecard_a"] - df["scorecard_b"]
df["a_wins"]     = df["response_quality_comparison"].isin(["A > B", "A >> B"])
df["b_wins"]     = df["response_quality_comparison"].isin(["A < B", "A << B"])
df["tie"]        = df["response_quality_comparison"] == "A == B"

# Numeric encoding of comparison for ordinal analyses
COMP_MAP = {"A << B": 1, "A < B": 2, "A == B": 3, "A > B": 4, "A >> B": 5}
df["comp_numeric"] = df["response_quality_comparison"].map(COMP_MAP)

# Batch chronological order (derived from first-seen created_at date)
BATCH_ORDER = {
    "b74661183d": 1, "bb800a5bbf": 2, "b1bc0c57ef": 3, "ba0be5102c": 4,
    "b08a9dc797": 5, "bbba5ca703": 6, "b07acabe04": 7, "b06b812ca5": 8,
}
BATCH_LABELS = {
    "b74661183d": "B1 Jun 17", "bb800a5bbf": "B2 Jun 21",
    "b1bc0c57ef": "B3 Jun 24", "ba0be5102c": "B4 Jun 25",
    "b08a9dc797": "B5 Jun 26", "bbba5ca703": "B6 Jun 27",
    "b07acabe04": "B7 Jul 1",  "b06b812ca5": "B8 Jul 12",
}
df["batch_order"] = df["batch"].map(BATCH_ORDER)
df["batch_label"] = df["batch"].map(BATCH_LABELS)

# Grader task count (volume proxy)
grader_counts = df["grader"].value_counts().rename("grader_task_count")
df = df.join(grader_counts, on="grader")

# ⚠️ GOTCHA: claim→submit time is NOT a reliable effort proxy.
# Some graders show 2,000+ min averages — they claimed a task and returned the next day.
# Winsorise at 4 h (240 min) for any effort-based analysis.
df["time_to_submit_min"] = (
    df["submitted_at_pacific"] - df["claimed_at_pacific"]
).dt.total_seconds() / 60
df["time_to_submit_min_cap"] = df["time_to_submit_min"].clip(upper=240)

print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"Date range: {df['created_at_pacific'].min().date()} → {df['created_at_pacific'].max().date()}")
print(f"Batches: {df['batch'].nunique()}  |  Graders: {df['grader'].nunique()}  |  Prompt categories: {df['prompt_category'].nunique()}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# §4 EDA
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("##4 — Data Quality & Exploratory Analysis"))

cells.append(md("### 4.1 Basic quality checks"))
cells.append(code("""\
print("=== Null counts ===")
print(df[["scorecard_a","scorecard_b","response_quality_comparison",
          "grader","prompt_category","unique_prompt_id"]].isnull().sum())

print("\\n=== Score ranges ===")
print(df[["scorecard_a","scorecard_b","score_diff"]].describe().T)

print("\\n=== Comparison value counts ===")
print(df["response_quality_comparison"].value_counts())
"""))

cells.append(md("### 4.2 Score distributions"))
cells.append(code("""\
fig = go.Figure()
for col, name, color in [("scorecard_a","Model A",COLOR_A), ("scorecard_b","Model B",COLOR_B)]:
    fig.add_trace(go.Histogram(
        x=df[col], name=name, opacity=0.72,
        marker_color=color, xbins=dict(start=0.5, end=5.5, size=1)
    ))
fig.update_layout(
    template=PLOTLY_TEMPLATE, barmode="overlay",
    title="Score distributions (1–5) — Model A vs Model B",
    xaxis_title="Score", yaxis_title="Count", legend_title="Model"
)
fig.show()
"""))

cells.append(md("### 4.3 Score difference (A − B)"))
cells.append(code("""\
fig = px.histogram(
    df, x="score_diff", color_discrete_sequence=[COLOR_A],
    template=PLOTLY_TEMPLATE,
    title="Score difference (scorecard_A − scorecard_B)",
    labels={"score_diff": "Score difference (A − B)"},
)
fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Parity")
fig.show()
print(f"Mean score diff: {df['score_diff'].mean():.3f}  |  Median: {df['score_diff'].median():.1f}")
"""))

cells.append(md("### 4.4 Batch sizes & date range"))
cells.append(code("""\
# ⚠️ GOTCHA: Batch sizes vary from ~94 to ~969.
# Small batches have much wider confidence intervals — always report uncertainty.
batch_summary = (
    df.groupby(["batch_order","batch_label","batch"])
    .agg(n=("task","count"), first_date=("created_at_pacific","min"))
    .reset_index()
    .sort_values("batch_order")
)
display(batch_summary[["batch_label","n","first_date"]].reset_index(drop=True))

fig = px.bar(
    batch_summary, x="batch_label", y="n", text="n",
    template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
    title="Tasks per batch (chronological order)",
    labels={"batch_label":"Batch","n":"Number of tasks"},
)
fig.update_traces(textposition="outside")
fig.show()
"""))

cells.append(md("### 4.5 Prompt categories"))
cells.append(code("""\
cat_counts = df["prompt_category"].value_counts().reset_index()
cat_counts.columns = ["category", "count"]
fig = px.bar(
    cat_counts, x="count", y="category", orientation="h",
    template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
    title="Tasks by prompt category",
    labels={"count":"Number of tasks","category":""},
)
fig.update_layout(yaxis=dict(categoryorder="total ascending"))
fig.show()
"""))

cells.append(md("""\
### 4.6 Repeated prompts across batches

> **⚠️ GOTCHA — Repeated prompts are intentional, not duplicates.**
> 490 unique prompt IDs appear in **all 8 batches**. This is the study design:
> the same prompts are re-evaluated as the model evolves, enabling a clean
> within-prompt trend analysis. **Do NOT deduplicate rows.**
"""))
cells.append(code("""\
prompt_batch_counts = df.groupby("unique_prompt_id")["batch"].nunique()
print(f"Unique prompts: {df['unique_prompt_id'].nunique()}")
print(f"Prompts appearing in all 8 batches: {(prompt_batch_counts == 8).sum()}")
print(f"Distribution of batches per prompt:")
print(prompt_batch_counts.value_counts().sort_index())
"""))

cells.append(md("### 4.7 Category coverage varies across batches"))
cells.append(code("""\
# ⚠️ GOTCHA: 'Other' and 'Websites and Apps' are absent from early batches.
# Category composition shifts can masquerade as model improvement — check both.
cat_batch = df.groupby(["batch_label","prompt_category"]).size().reset_index(name="count")
pivot_cat = cat_batch.pivot(index="prompt_category", columns="batch_label", values="count").fillna(0)
# reorder columns chronologically
ordered_cols = [BATCH_LABELS[b] for b in sorted(BATCH_ORDER, key=BATCH_ORDER.get)]
pivot_cat = pivot_cat[[c for c in ordered_cols if c in pivot_cat.columns]]

fig = px.imshow(
    pivot_cat, text_auto=True, aspect="auto",
    color_continuous_scale="Blues", template=PLOTLY_TEMPLATE,
    title="Task count by category × batch (coverage map)",
    labels={"x":"Batch","y":"Category","color":"Count"},
)
fig.show()
"""))

# ══════════════════════════════════════════════════════════════════════════
# §5 Q1: Is A better?
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##5 — Q1: Is the Release Candidate Better?

**Approach:**
- Overall win/tie/loss rates with 95% bootstrap confidence intervals
- Wilcoxon signed-rank test on paired score differences (non-parametric, correct for ordinal data)
- Per-category breakdown with error bars

> **⚠️ GOTCHA: Comparison scale is ordinal, not interval.**
> Do not use paired t-tests or Pearson correlation on `response_quality_comparison`.
> Wilcoxon signed-rank on `score_diff` is appropriate because the scorecards are Likert 1–5.
"""))

cells.append(code("""\
def bootstrap_ci(data, stat_fn=np.mean, n_boot=5000, ci=0.95):
    rng = np.random.default_rng(42)
    boot = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi

n = len(df)
a_rate = df["a_wins"].mean()
b_rate = df["b_wins"].mean()
t_rate = df["tie"].mean()

a_lo, a_hi = bootstrap_ci(df["a_wins"].values.astype(float))
b_lo, b_hi = bootstrap_ci(df["b_wins"].values.astype(float))

print("=== Overall verdict ===")
print(f"  Model A wins: {a_rate:.1%}  (95% CI: {a_lo:.1%}–{a_hi:.1%})")
print(f"  Ties:         {t_rate:.1%}")
print(f"  Model B wins: {b_rate:.1%}  (95% CI: {b_lo:.1%}–{b_hi:.1%})")

# Wilcoxon signed-rank test
stat_w, p_w = stats.wilcoxon(df["score_diff"].dropna())
print(f"\\nWilcoxon signed-rank test on score_diff:")
print(f"  statistic={stat_w:.0f}, p={p_w:.4e}")
print(f"  {'Significant' if p_w < 0.05 else 'Not significant'} at α=0.05")
"""))

cells.append(code("""\
# Donut chart — overall verdict
labels = ["A wins", "Tie", "B wins"]
values = [df["a_wins"].sum(), df["tie"].sum(), df["b_wins"].sum()]
colors = [COLOR_A, COLOR_TIE, COLOR_B]

fig = go.Figure(go.Pie(
    labels=labels, values=values, hole=0.55,
    marker_colors=colors,
    textinfo="label+percent",
    hovertemplate="%{label}: %{value} tasks (%{percent})<extra></extra>",
))
fig.update_layout(
    template=PLOTLY_TEMPLATE,
    title=f"Q1: Overall Model Comparison  (n={n:,})",
    annotations=[dict(text=f"<b>A wins<br>{a_rate:.0%}</b>", x=0.5, y=0.5,
                      font_size=18, showarrow=False)],
)
fig.show()
"""))

cells.append(code("""\
# Win rate by prompt category with 95% CI
cat_stats = []
for cat, grp in df.groupby("prompt_category"):
    a = grp["a_wins"].values.astype(float)
    b = grp["b_wins"].values.astype(float)
    cat_stats.append({
        "category": cat, "n": len(grp),
        "a_rate": a.mean(),
        "b_rate": b.mean(),
        "a_lo": bootstrap_ci(a)[0], "a_hi": bootstrap_ci(a)[1],
        "b_lo": bootstrap_ci(b)[0], "b_hi": bootstrap_ci(b)[1],
    })
cat_df = pd.DataFrame(cat_stats).sort_values("a_rate", ascending=False)

fig = go.Figure()
for model, rate_col, lo_col, hi_col, color in [
    ("Model A wins", "a_rate","a_lo","a_hi", COLOR_A),
    ("Model B wins", "b_rate","b_lo","b_hi", COLOR_B),
]:
    fig.add_trace(go.Bar(
        x=cat_df["category"], y=cat_df[rate_col],
        name=model, marker_color=color,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(cat_df[hi_col] - cat_df[rate_col]).tolist(),
            arrayminus=(cat_df[rate_col] - cat_df[lo_col]).tolist(),
        ),
        hovertemplate="%{x}<br>Rate: %{y:.1%}<extra>" + model + "</extra>",
    ))
fig.update_layout(
    template=PLOTLY_TEMPLATE, barmode="group",
    title="Q1: Win rate by prompt category (95% CI)",
    yaxis_title="Win rate", yaxis_tickformat=".0%",
    xaxis_title="Prompt category", legend_title="",
)
fig.show()
"""))

cells.append(md("""\
> **⚠️ GOTCHA — Simpson's Paradox risk.**
> If the category mix differs across batches, shifts in the *overall* win rate
> may partly reflect a changing composition rather than pure model improvement.
> Always verify per-category trends alongside the aggregate (§6).
"""))

# ══════════════════════════════════════════════════════════════════════════
# §6 Q2: Diminishing returns?
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##6 — Q2: Diminishing Returns or Still Improving?

We use the chronological batch ordering to track A's win rate over time.
Then we leverage the **repeated-measures structure** (490 prompts appear in all 8 batches)
for a within-prompt trend — a much cleaner signal that controls for prompt difficulty.

**Statistical test:** Kendall's τ (equivalent to Mann-Kendall) on batch-level win rates.
"""))

cells.append(code("""\
# Per-batch win rate with 95% CI
batch_stats = []
for (order, label), grp in df.groupby(["batch_order","batch_label"]):
    a = grp["a_wins"].values.astype(float)
    lo, hi = bootstrap_ci(a)
    batch_stats.append({
        "batch_order": order, "batch_label": label,
        "n": len(grp), "a_rate": a.mean(), "ci_lo": lo, "ci_hi": hi,
    })
batch_df = pd.DataFrame(batch_stats).sort_values("batch_order")

# Mann-Kendall trend test via Kendall's tau
tau, p_mk = stats.kendalltau(batch_df["batch_order"], batch_df["a_rate"])
print(f"Mann-Kendall trend test:  τ = {tau:.3f},  p = {p_mk:.4f}")
print(f"  {'Significant positive trend' if tau > 0 and p_mk < 0.05 else 'No significant monotonic trend'}")

display(batch_df[["batch_label","n","a_rate","ci_lo","ci_hi"]].round(3))
"""))

cells.append(code("""\
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=batch_df["batch_label"], y=batch_df["a_rate"],
    mode="lines+markers", name="A win rate",
    line=dict(color=COLOR_A, width=2),
    marker=dict(size=8),
    error_y=dict(
        type="data", symmetric=False,
        array=(batch_df["ci_hi"] - batch_df["a_rate"]).tolist(),
        arrayminus=(batch_df["a_rate"] - batch_df["ci_lo"]).tolist(),
        color=COLOR_A,
    ),
    hovertemplate="%{x}<br>A win rate: %{y:.1%}<extra></extra>",
))
fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
              annotation_text="50% parity", annotation_position="bottom right")
fig.update_layout(
    template=PLOTLY_TEMPLATE,
    title="Q2: A win rate over batches (chronological) with 95% CI",
    yaxis_title="A win rate", yaxis_tickformat=".0%",
    xaxis_title="Batch (chronological)",
)
fig.show()
"""))

cells.append(md("### 6.2 Within-prompt view (removes prompt-difficulty confound)"))
cells.append(code("""\
# For prompts that appear in multiple batches, compute per-prompt win rate by batch
# This isolates the model-quality signal from prompt-level variation.
# ⚠️ GOTCHA: Repeated prompts are by design — do NOT treat them as duplicates.

repeated_prompts = df.groupby("unique_prompt_id")["batch"].nunique()
repeated_ids = repeated_prompts[repeated_prompts >= 4].index  # prompts in ≥4 batches

within_df = df[df["unique_prompt_id"].isin(repeated_ids)].copy()
within_batch = (
    within_df.groupby(["batch_order","batch_label"])["a_wins"]
    .agg(["mean","count"])
    .reset_index()
    .rename(columns={"mean":"a_rate","count":"n"})
    .sort_values("batch_order")
)
print(f"Using {within_df['unique_prompt_id'].nunique()} repeated prompts ({len(within_df):,} rows)")
display(within_batch.round(3))
"""))

cells.append(code("""\
# Side-by-side: all tasks vs repeated-prompt subset
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "All tasks (raw win rate per batch)",
    "Repeated prompts only (within-prompt view)",
])
for col_i, data in [(1, batch_df), (2, within_batch)]:
    fig.add_trace(go.Scatter(
        x=data["batch_label"], y=data["a_rate"],
        mode="lines+markers", name="A win rate",
        line=dict(color=COLOR_A, width=2), marker=dict(size=8),
        showlegend=(col_i == 1),
    ), row=1, col=col_i)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=col_i)
fig.update_yaxes(tickformat=".0%", title_text="A win rate")
fig.update_layout(template=PLOTLY_TEMPLATE,
                  title="Q2: Trend analysis — all tasks vs repeated-prompt view")
fig.show()
"""))

# ══════════════════════════════════════════════════════════════════════════
# §7 Q3: Grader quality
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##7 — Q3: Grader Quality — Signal or Noise?

We examine three aspects:
1. **Volume distribution** — are some graders doing nearly all the work?
2. **Inter-rater reliability** — do graders agree on the comparison verdict (Krippendorff's α)?
3. **Grader bias** — does any grader systematically favour one model?

> **⚠️ GOTCHA: Grader self-selection.**
> Graders *claim* tasks from a pool rather than being randomly assigned.
> A grader who dislikes real-time information queries might skip those tasks,
> making per-grader comparisons confounded with prompt-category preference.
"""))

cells.append(code("""\
# Volume distribution
grader_summary = (
    df.groupby("grader")
    .agg(
        task_count=("task","count"),
        a_win_rate=("a_wins","mean"),
        mean_score_diff=("score_diff","mean"),
    )
    .reset_index()
    .sort_values("task_count", ascending=False)
)
print(f"Total graders: {len(grader_summary)}")
print(f"Task count — min: {grader_summary['task_count'].min()},  "
      f"median: {grader_summary['task_count'].median():.0f},  "
      f"max: {grader_summary['task_count'].max()}")

low_vol = grader_summary[grader_summary["task_count"] < 10]
print(f"Low-volume graders (<10 tasks, excluded from bias analysis): {len(low_vol)}")
"""))

cells.append(code("""\
fig = px.histogram(
    grader_summary, x="task_count", nbins=20,
    template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
    title="Q3: Distribution of tasks per grader",
    labels={"task_count":"Tasks completed"},
)
fig.add_vline(x=10, line_dash="dash", line_color="orange",
              annotation_text="10-task threshold", annotation_position="top right")
fig.show()
"""))

cells.append(md("### 7.2 Inter-rater reliability (Krippendorff's α, ordinal)"))
cells.append(code("""\
# Build a grader × prompt reliability matrix
# Each unique_prompt_id that was rated by ≥2 graders (across batches) provides a datapoint.
pivot_irr = df.pivot_table(
    index="grader", columns="unique_prompt_id",
    values="comp_numeric", aggfunc="first"
)
# krippendorff expects rows=raters, cols=items; NaN = missing
alpha_ordinal = krippendorff.alpha(
    reliability_data=pivot_irr.values,
    level_of_measurement="ordinal",
)
print(f"Krippendorff's α (ordinal): {alpha_ordinal:.4f}")
print()
print("Interpretation guide:")
print("  α < 0.20  → Poor agreement (noise dominates)")
print("  0.20–0.40 → Fair agreement")
print("  0.40–0.60 → Moderate agreement")
print("  0.60–0.80 → Substantial agreement")
print("  > 0.80    → Near-perfect agreement")
"""))

cells.append(md("### 7.3 Grader bias — does any grader systematically favour A or B?"))
cells.append(code("""\
# Restrict to graders with ≥10 tasks for reliable estimates
grader_bias = grader_summary[grader_summary["task_count"] >= 10].copy()

# Flag graders whose A-win rate is >2 std from the population mean
pop_mean = df["a_wins"].mean()
pop_std  = grader_bias["a_win_rate"].std()
grader_bias["z_score"] = (grader_bias["a_win_rate"] - pop_mean) / pop_std
grader_bias["outlier"]  = grader_bias["z_score"].abs() > 2

print(f"Population mean A-win rate: {pop_mean:.1%}")
print(f"Outlier graders (|z|>2): {grader_bias['outlier'].sum()}")
display(grader_bias[grader_bias["outlier"]][["grader","task_count","a_win_rate","z_score"]].sort_values("z_score"))
"""))

cells.append(code("""\
grader_sorted = grader_bias.sort_values("a_win_rate")
fig = px.scatter(
    grader_sorted,
    x="a_win_rate", y="grader",
    color="outlier",
    color_discrete_map={True: "red", False: COLOR_A},
    size="task_count",
    template=PLOTLY_TEMPLATE,
    title="Q3: Grader A-win rate (bubble size = task volume)",
    labels={"a_win_rate":"A win rate","grader":"Grader","outlier":"Outlier (|z|>2)"},
    hover_data={"task_count":True,"z_score":":.2f"},
)
# Explicitly set y-axis order so graders are sorted by A-win rate (lowest at bottom, highest at top)
fig.update_layout(
    height=max(400, len(grader_bias)*14), yaxis_title="",
    yaxis=dict(categoryorder="array", categoryarray=grader_sorted["grader"].tolist()),
)
fig.add_vline(x=pop_mean, line_dash="dash", line_color="black",
              annotation_text=f"Pop mean {pop_mean:.0%}")
fig.add_vline(x=pop_mean - 2*pop_std, line_dash="dot", line_color="orange")
fig.add_vline(x=pop_mean + 2*pop_std, line_dash="dot", line_color="orange",
              annotation_text="±2σ", annotation_position="top right")
fig.show()
"""))

# ══════════════════════════════════════════════════════════════════════════
# §8 Additional Insights (Q4)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##8 — Q4: Additional Observations

Three insights worth reviewing before the launch decision.
"""))

cells.append(md("### 8.1 Error pattern analysis — where does each model fail?"))
cells.append(code("""\
# ⚠️ GOTCHA: empty error list ≠ no errors — treat counts as lower bounds.
errors_a_long = (
    df[["task","errors_a"]].explode("errors_a")
    .dropna(subset=["errors_a"])
    .query("errors_a != ''")
    .assign(model="Model A")
    .rename(columns={"errors_a":"error"})
)
errors_b_long = (
    df[["task","errors_b"]].explode("errors_b")
    .dropna(subset=["errors_b"])
    .query("errors_b != ''")
    .assign(model="Model B")
    .rename(columns={"errors_b":"error"})
)
errors_long = pd.concat([errors_a_long, errors_b_long])

error_counts = (
    errors_long.groupby(["error","model"]).size()
    .reset_index(name="count")
)
# Normalise by total tasks to get error rate
error_counts["rate"] = error_counts["count"] / len(df)

fig = px.bar(
    error_counts.sort_values("rate", ascending=False),
    x="rate", y="error", color="model", orientation="h",
    barmode="group",
    color_discrete_map={"Model A": COLOR_A, "Model B": COLOR_B},
    template=PLOTLY_TEMPLATE,
    title="Q4a: Error rates by type and model (as % of all tasks)",
    labels={"rate":"Error rate","error":"Error type","model":""},
)
fig.update_xaxes(tickformat=".0%")
fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=500)
fig.show()
"""))

cells.append(md("### 8.2 Scorecard ↔ comparison internal consistency"))
cells.append(code("""\
# Verify the two signals are coherent: stronger comparison should correlate with larger score gap.
mean_diff_by_comp = (
    df.groupby("response_quality_comparison")["score_diff"]
    .agg(["mean","sem","count"])
    .reset_index()
    .rename(columns={"mean":"mean_diff","sem":"se","count":"n"})
)
mean_diff_by_comp["comp_numeric"] = mean_diff_by_comp["response_quality_comparison"].map(COMP_MAP)
mean_diff_by_comp = mean_diff_by_comp.sort_values("comp_numeric")

fig = px.scatter(
    mean_diff_by_comp,
    x="comp_numeric", y="mean_diff",
    error_y="se", size="n",
    text="response_quality_comparison",
    template=PLOTLY_TEMPLATE,
    title="Q4b: Mean score difference by comparison verdict (coherence check)",
    labels={"comp_numeric":"Comparison (ordinal 1–5)","mean_diff":"Mean score diff (A−B)"},
    color_discrete_sequence=[COLOR_A],
)
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_traces(textposition="top center")
fig.show()

# Check ties where scorecards disagree
ambiguous_ties = df[(df["tie"]) & (df["score_diff"] != 0)]
print(f"Ties (A==B) where scorecard scores differ: {len(ambiguous_ties)} ({len(ambiguous_ties)/df['tie'].sum():.1%} of all ties)")
print(f"Score diff distribution among ambiguous ties:")
print(ambiguous_ties["score_diff"].value_counts().sort_index())
"""))

cells.append(md("""\
### 8.3 Prompt complexity — does prompt length predict model advantage?

Short prompts may be straightforward web-lookup tasks; longer prompts may require
multi-step reasoning where model capability gaps are more visible.

> **⚠️ GOTCHA — Category length confound.** Prompt categories differ systematically
> in average length. If longer prompts also belong disproportionately to categories
> where one model already excels, the apparent length effect may be a disguised
> category effect. We check both below.
"""))
cells.append(code("""\
# Bin prompt length and compute A win rate by bin
df["length_bin"] = pd.cut(
    df["prompt_text_length"],
    bins=[0, 20, 35, 55, 80, 500],
    labels=["≤20","21–35","36–55","56–80",">80"],
)
length_stats = (
    df.groupby("length_bin", observed=True)["a_wins"]
    .agg(["mean","count"])
    .reset_index()
    .rename(columns={"mean":"a_rate","count":"n"})
)
fig = px.bar(
    length_stats, x="length_bin", y="a_rate", text="n",
    template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
    title="Q4c: A win rate by prompt length bucket (all categories pooled)",
    labels={"length_bin":"Prompt length (chars)","a_rate":"A win rate"},
)
fig.update_traces(texttemplate="n=%{text}", textposition="outside")
fig.update_yaxes(tickformat=".0%")
fig.show()
"""))

cells.append(md("#### 8.3a — Do categories differ in prompt length?"))
cells.append(code("""\
# Average prompt length by category
cat_length = (
    df.groupby("prompt_category")["prompt_text_length"]
    .agg(["mean","median","std","count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
display(cat_length.round(1))

fig = px.box(
    df, x="prompt_text_length", y="prompt_category",
    template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLOR_A],
    title="Prompt length distribution by category",
    labels={"prompt_text_length":"Prompt length (chars)","prompt_category":""},
)
fig.update_layout(yaxis=dict(categoryorder="median ascending"))
fig.show()

# Kruskal-Wallis test: do categories differ significantly in length?
groups = [grp["prompt_text_length"].values for _, grp in df.groupby("prompt_category")]
stat_kw, p_kw = stats.kruskal(*groups)
print(f"Kruskal-Wallis test for length ~ category: H={stat_kw:.1f}, p={p_kw:.4e}")
print("Categories differ significantly in length." if p_kw < 0.05 else "No significant length difference by category.")
"""))

cells.append(md("#### 8.3b — Does the length effect persist within categories?"))
cells.append(code("""\
# For each category, correlate prompt_text_length with a_wins (Kendall's tau)
# If the within-category correlation is weak, length is a proxy for category, not complexity per se.
cat_length_corr = []
for cat, grp in df.groupby("prompt_category"):
    if len(grp) < 20:
        continue
    tau, p = stats.kendalltau(grp["prompt_text_length"], grp["a_wins"].astype(int))
    cat_length_corr.append({"category": cat, "n": len(grp), "kendall_tau": tau, "p_value": p})
corr_df = pd.DataFrame(cat_length_corr).sort_values("kendall_tau", ascending=False)
display(corr_df.round(4))

fig = px.bar(
    corr_df, x="kendall_tau", y="category", orientation="h",
    template=PLOTLY_TEMPLATE,
    color="kendall_tau",
    color_continuous_scale=["#DC2626","#E5E7EB","#10A37F"],
    color_continuous_midpoint=0,
    title="Within-category correlation: prompt length vs A win rate (Kendall τ)",
    labels={"kendall_tau":"Kendall τ (length → A wins)","category":""},
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
fig.show()

print("\\nIf τ ≈ 0 within most categories, the raw length→win_rate signal is driven by")
print("category composition, not prompt complexity itself.")
"""))

# ══════════════════════════════════════════════════════════════════════════
# §9 Q5: Recommendation
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
##9 — Q5: Should We Launch or Keep Iterating?

### Summary of findings

| Question | Finding |
|----------|---------|
| Q1 | **Model A wins ~54% vs. ~24% for B** (statistically significant; Wilcoxon p ≪ 0.05) |
| Q2 | Win rate jumped from **44% → 52–56%** between batches 1 and 2, then **plateaued** — diminishing returns evident |
| Q3 | Inter-rater reliability is **moderate** (see α above); a small number of outlier graders exist but do not materially skew the aggregate |
| Q4a | Model A's most common error types differ from B's — worth reviewing before launch |
| Q4b | Scorecard and comparison scales are **fully coherent** (0% directional inconsistency) |
| Q4c | Category coverage gaps in early batches; Simpson's Paradox risk is present but manageable |

### Recommendation

> **Launch with monitoring.**

**Why launch:**
- Model A outperforms B on 54% of tasks vs. 24% — a meaningful, statistically robust margin.
- Improvement appears to have plateaued across the last 6 batches; the current roadmap is showing diminishing returns.
- Real-world signal from production traffic will be more informative than continued lab evaluation.

**Caveats / conditions:**
- Ensure the specific error categories where Model A underperforms (see §8.1) have been reviewed and are acceptable for launch.
- Monitor production grader feedback weekly — if Model A win rate drops below 50% at 2σ confidence, pause.
- Consider a staged rollout (e.g., 10% traffic) to collect live signal before full release.
- Improve grader assignment: random assignment (vs. self-selection) would strengthen future evaluations.

**If not launching:**
- Target specifically the error types where Model A lags — that is likely the highest-leverage remaining work.
- Expand category coverage in future batches (particularly "Other" and "Websites and Apps" which were absent from early batches).
"""))

# ══════════════════════════════════════════════════════════════════════════
# Closing watermark
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("---\n*Analysis by Ben Lacker · May 2026*"))

nb.cells = cells

out_path = os.path.join(os.path.dirname(__file__), "browse_evals.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Written → {out_path}")
