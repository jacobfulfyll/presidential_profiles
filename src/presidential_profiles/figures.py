"""All chart output. Static PNGs (matplotlib) + interactive HTML (plotly).

Colors and chart chrome follow a validated palette: categorical hues are
assigned in fixed slot order, magnitude uses a single blue ramp, grids and
axes are recessive, and party colors follow political convention (drawn from
the same validated slots).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "outputs" / "figures"
HTML_DIR = REPO_ROOT / "outputs" / "interactive"

# --- palette tokens (light mode) -------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PARTY_COLORS = {
    "Democratic": "#2a78d6",
    "Republican": "#e34948",
    "Democratic-Republican": "#1baf7a",
    "Whig": "#eda100",
    "Federalist": "#4a3aa7",
    "Unaffiliated": "#898781",
}

BLUES_CMAP = LinearSegmentedColormap.from_list("pp_blues", BLUE_RAMP)


def _theme():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": 600,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def _yearly_rate(stats: pd.DataFrame, count_col: str, window: int = 5,
                 min_tokens: int = 20_000) -> pd.Series:
    """Rolling per-10k rate by year, computed on window totals. Years whose
    window holds fewer than min_tokens are masked (sparse early corpus)."""
    g = stats.groupby("year")
    counts = g[count_col].sum()
    toks = g["n_tokens"].sum()
    years = pd.RangeIndex(int(stats["year"].min()), int(stats["year"].max()) + 1)
    counts = counts.reindex(years, fill_value=0)
    toks = toks.reindex(years, fill_value=0)
    csum = counts.rolling(window, center=True, min_periods=1).sum()
    tsum = toks.rolling(window, center=True, min_periods=1).sum()
    rate = csum / tsum * 10_000
    rate[tsum < min_tokens] = np.nan
    return rate


def modal_verbs(stats: pd.DataFrame):
    """The retreat of 'shall' and the language of obligation, 1789-2026."""
    _theme()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    modals = ["shall", "will", "must", "should"]
    for i, m in enumerate(modals):
        rate = _yearly_rate(stats, f"modal_{m}")
        ax.plot(rate.index, rate.values, color=SERIES[i], label=m)
        last = rate.dropna()
        ax.annotate(
            m,
            (last.index[-1], last.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK2,
            fontsize=9,
            va="center",
        )
    ax.set_title("Modal verbs in presidential speech, 1789–2026")
    ax.set_ylabel("uses per 10,000 words (5-yr rolling)")
    ax.set_xlim(1786, 2042)
    ax.legend(loc="upper right")
    _save(fig, "modal_verbs.png")


def pronouns(stats: pd.DataFrame):
    """First-person singular vs plural over time."""
    _theme()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, col) in enumerate([("we / us / our", "we_count"), ("I / me / my", "i_count")]):
        rate = _yearly_rate(stats, col)
        ax.plot(rate.index, rate.values, color=SERIES[i], label=label)
        last = rate.dropna()
        ax.annotate(
            label.split(" ")[0],
            (last.index[-1], last.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK2,
            fontsize=9,
            va="center",
        )
    ax.set_title("First-person pronouns: the collective vs the individual voice")
    ax.set_ylabel("uses per 10,000 words (5-yr rolling)")
    ax.set_xlim(1786, 2042)
    ax.legend(loc="upper left")
    _save(fig, "pronouns.png")


def readability(stats: pd.DataFrame):
    """Flesch-Kincaid grade level per speech with decade median."""
    _theme()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(
        stats["year"],
        stats["fk_grade"].clip(0, 30),
        s=10,
        color=BLUE_RAMP[1],
        alpha=0.45,
        linewidths=0,
    )
    med = (stats.groupby("year")["fk_grade"].median()
           .reindex(pd.RangeIndex(int(stats["year"].min()), int(stats["year"].max()) + 1))
           .rolling(7, center=True, min_periods=3).median())
    ax.plot(med.index, med.values, color=BLUE_RAMP[5], label="rolling median (7 yr)")
    ax.set_title("Reading level of presidential speeches, 1789–2026")
    ax.set_ylabel("Flesch–Kincaid grade level")
    ax.set_ylim(0, 30)
    ax.legend(loc="upper right")
    _save(fig, "readability.png")


def topics_small_multiples(doc_topics: pd.DataFrame, topic_terms: dict):
    """Mean topic share by decade, one panel per topic, single hue."""
    _theme()
    keys = [k for k in doc_topics.columns if k.startswith("topic_")]
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True)
    for ax, key in zip(axes.flat, keys):
        share = doc_topics.groupby("decade")[key].mean() * 100
        ax.fill_between(share.index, share.values, color=BLUE_RAMP[2], alpha=0.55)
        ax.plot(share.index, share.values, color=BLUE_RAMP[4], linewidth=1.6)
        ax.set_title(" · ".join(topic_terms[key][:3]), fontsize=9.5)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    fig.suptitle("Topic prevalence by decade (NMF, 12 topics)", fontsize=13, fontweight=600)
    fig.supylabel("mean share of speech (%)", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0.01, 0, 1, 0.98))
    _save(fig, "topics_by_decade.png")


def keyword_small_multiples(trends: pd.DataFrame):
    """Rate per 10k words by decade, one panel per term group."""
    _theme()
    terms = list(trends["term"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
    for ax, term in zip(axes.flat, terms):
        sub = trends[trends["term"] == term]
        ax.fill_between(sub["period"], sub["rate"], color=BLUE_RAMP[2], alpha=0.55)
        ax.plot(sub["period"], sub["rate"], color=BLUE_RAMP[4], linewidth=1.6)
        ax.set_title(term, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    fig.suptitle("Keyword usage per 10,000 words, by decade", fontsize=13, fontweight=600)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, "keyword_trends.png")


def distinctive_terms_chart(scores: pd.DataFrame):
    """Two-panel bar chart of era-distinctive vocabulary (z-scored log-odds)."""
    _theme()
    eras = list(scores["era"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, era in zip(axes, eras):
        sub = scores[scores["era"] == era].copy()
        sub["mag"] = sub["z"].abs()
        sub = sub.sort_values("mag")
        ax.barh(sub["term"], sub["mag"], color=BLUE_RAMP[4], height=0.62)
        ax.set_title(f"Distinctive of {era}", fontsize=11)
        ax.set_xlabel("log-odds z-score", fontsize=9)
        ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "What changed: era-distinctive vocabulary in modern presidential speech",
        fontsize=13,
        fontweight=600,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "distinctive_terms.png")


def president_map(emb: pd.DataFrame):
    """PCA map of president mean embeddings — PNG and interactive HTML."""
    _theme()
    fig, ax = plt.subplots(figsize=(12, 9))
    for party, color in PARTY_COLORS.items():
        sub = emb[emb["party"] == party]
        ax.scatter(sub["pc1"], sub["pc2"], s=46, color=color, label=party, zorder=3)
    texts = [
        ax.text(row["pc1"], row["pc2"], row["president"], fontsize=7, color=INK2)
        for _, row in emb.iterrows()
    ]
    adjust_text(texts, ax=ax, expand=(1.2, 1.6), force_text=(0.3, 0.6))
    ax.set_title("Who sounds like whom: presidents mapped by speech embeddings (PCA)")
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend(loc="best")
    _save(fig, "president_map.png")

    # Interactive twin with hover detail.
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    figp = go.Figure()
    for party, color in PARTY_COLORS.items():
        sub = emb[emb["party"] == party]
        figp.add_trace(
            go.Scatter(
                x=sub["pc1"],
                y=sub["pc2"],
                mode="markers+text",
                text=sub["president"],
                textposition="top center",
                textfont=dict(size=9, color=INK2),
                marker=dict(size=11, color=color),
                name=party,
                customdata=np.stack([sub["first_year"], sub["n_speeches"]], axis=-1),
                hovertemplate=(
                    "<b>%{text}</b><br>" + party
                    + "<br>first speech: %{customdata[0]}"
                    + "<br>speeches in corpus: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    figp.update_layout(
        title="Who sounds like whom: presidents mapped by speech embeddings (PCA)",
        template="simple_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=INK),
        xaxis=dict(title="principal component 1", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="principal component 2", gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=760,
    )
    path = HTML_DIR / "president_map.html"
    figp.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def similarity_heatmap(sim: pd.DataFrame):
    """Chronological president-by-president cosine similarity."""
    _theme()
    names = list(sim.index)
    m = sim.to_numpy().copy()
    lo = np.percentile(m[~np.eye(len(m), dtype=bool)], 2)
    fig, ax = plt.subplots(figsize=(12.5, 11))
    im = ax.imshow(m, cmap=BLUES_CMAP, vmin=lo, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(names)), names, fontsize=7.5, color=INK2)
    ax.grid(visible=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    cbar.outline.set_visible(False)
    ax.set_title("Rhetorical similarity between presidents (chronological order)")
    _save(fig, "similarity_heatmap.png")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    figp = go.Figure(
        go.Heatmap(
            z=m,
            x=names,
            y=names,
            colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)],
            zmin=lo,
            zmax=1.0,
            hovertemplate="%{y} × %{x}<br>similarity: %{z:.3f}<extra></extra>",
            colorbar=dict(title="cosine"),
        )
    )
    figp.update_layout(
        title="Rhetorical similarity between presidents (chronological order)",
        template="simple_white",
        paper_bgcolor=SURFACE,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=INK),
        height=860,
        width=960,
        yaxis=dict(autorange="reversed"),
    )
    path = HTML_DIR / "similarity_heatmap.html"
    figp.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
