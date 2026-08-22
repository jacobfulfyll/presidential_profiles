"""Validated Issues V2 view models, directory, detail pages, and evidence exports."""

from dataclasses import dataclass, replace
import hashlib
import html as html_mod
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .figures import BASELINE, BLUE_RAMP, GRID, INK, INK2, MUTED, SURFACE
from .html_safety import json_for_script
from .issues_assets import ISSUES_CSS, ISSUES_JS
from .llm_annotations import corpus_fingerprint
from .expansion_site import NAV_CSS, nav
from .profiles import (
    _EXTRA_ANCHORS,
    _pick_sentence,
    miller_speech_url,
    public_display_name,
    slug,
)
from .site_style import FONT, PAGE_CSS
from . import ai_labels, bands, indices, issues, metrics, topic_quality

# Bands are drawn UNDER the trend line at low opacity: the 16-panel
# small-multiples grid on the dashboard reuses these traces, and a heavier fill
# turns it into mush. The old `fill="tozeroy"` area under the line is gone —
# two overlapping fills read as one shape and the reader cannot tell which edge
# is the estimate and which is the interval.
BAND_FILL = "rgba(109, 167, 236, 0.22)"
CAUTION_LINE = "rgba(109, 167, 236, 0.55)"

# The hollow marker for `interval_unresolvable` cells. It is drawn UNFILLED
# against the page, which is the whole signal: a solid dot on the line says "we
# measured this", a ring says "we measured the share but could not resolve an
# interval around it". Sized above the president dots (5px on the issue page,
# 4px in the grid) so it does not read as one of them.
UNRESOLVED_RING = "rgba(109, 167, 236, 0.95)"

# THE definition of the chart x window, for every issue surface. `site.X_RANGE`
# is derived from it rather than restating the pair — the same two numbers in two
# modules is how the lower bound comes to be widened in one place and not the
# other, which is exactly the defect `x_range_covering` exists to fix.
X_MIN, X_MAX = 1786, 2029


def x_range_covering(bands_frames) -> list[int]:
    """`[X_MIN, X_MAX]`, widened DOWNWARD to cover every plotted x.

    The lower bound is a floor to be widened; the upper bound is a fixed crop
    and is deliberately NOT adaptive (nothing in this corpus reaches 2029, and a
    stray future year should be visibly out of range rather than silently
    restyle every panel's axis).

    Widening matters because Surface A plots the recovered 1785 bucket one year
    below `X_MIN`: a fixed lower bound clipped the exact period this whole
    change exists to stop hiding — band, marker and hover all outside the axis.

    One implementation, used by the single-issue figure and by both dashboard
    grids. It lived twice, differently, for one round; `int()` is here because
    the shipped `x` column is float64 (era rows carry midpoints) and a bare
    min() would emit `1785.0` as an axis bound.

    Args:
        bands_frames: Iterable of `bands.parquet` slices or None.
    """
    lows = [int(b["x"].min()) for b in bands_frames
            if b is not None and not b.empty]
    return [min([X_MIN, *lows]), X_MAX]

# Minimum paragraphs before one president's own share inside one period is
# worth plotting as a dot. Below this a dot is a coin flip wearing a portrait.
# Measured on the shipped corpus: of the 100 (president, period) cells, this
# drops 9 and keeps 91. No president disappears entirely — all 45 keep at least
# one dot — so the floor thins the overlay rather than silently removing anyone
# from their own chart.
MIN_DOT_PARAGRAPHS = 25

ISSUE_VIEW_SCHEMA = "issue-page-v2"
OWNER_MIN_SPEECHES = indices.PRESIDENT_PERCENTILE_MIN_SPEECHES
DEFAULT_FINE_BUCKET = 20
FINE_TOPIC_COLORS = (
    "#275d8c", "#c06b35", "#60936a", "#8b5f96", "#9b7424",
    "#397d79", "#9b4e50", "#697c9f", "#805f48", "#63713d",
    "#9a6280", "#3f7b9b",
)
METHOD_CAVEATS = {
    "Education": (
        "The frozen AI taxonomy has no standalone Education topic. Its exploratory "
        "crosswalk reconstructs this issue through opportunity, jobs, and civil-rights "
        "topics, so it is especially diffuse across the two methods."
    ),
    "Money & banking": (
        "The frozen AI taxonomy emphasizes coinage and banking crises and has no "
        "standalone modern monetary-policy topic. Modern AI-topic comparisons are "
        "therefore incomplete."
    ),
}


def issue_slug(name: str) -> str:
    return slug(name.replace("&", "and"))


def president_period_dots(pl: pd.DataFrame, name: str,
                          min_paragraphs: int = MIN_DOT_PARAGRAPHS,
                          period_years: int = bands.PERIOD_YEARS) -> pd.DataFrame:
    """One dot per (president, period) — their own share inside that period.

    Replaces the previous one-dot-per-president-at-term-midpoint overlay, which
    could only ever say "this president, on average, somewhere in here". A
    president spanning several periods now gets a point in each, so the dots and
    the trend line are on the same time axis instead of merely near it.

    `(president, period)` cells thinner than `min_paragraphs` are dropped rather
    than plotted, so a dot never carries less evidence than the band it sits on.
    At the default that drops 9 of 100 cells and no president entirely; the
    chart prose states the floor, because "one dot per president per period"
    would otherwise imply a completeness the overlay does not have.
    """
    d = pl.assign(period=(pl["year"] // period_years) * period_years)
    g = d.groupby(["president", "period"], sort=True)
    out = g.agg(y=(name, "mean"), n=(name, "size")).reset_index()
    out = out[out["n"] >= min_paragraphs]
    return pd.DataFrame({
        "x": out["period"].to_numpy(),
        "y": (out["y"] * 100).to_numpy(),
        "name": out["president"].to_numpy(),
        "n": out["n"].to_numpy(),
    })


def band_y_top(band: pd.DataFrame | None, headroom: float = 1.05) -> float | None:
    """Panel y-ceiling, in percent — scaled by the TRUSTWORTHY content only.

    A 2-speech period's bootstrap interval is legitimately enormous: 1785's
    upper bound on "Religion & values" is 66.7% against a series that never
    exceeds 14.3%. Letting plotly autoscale to that flattens 240 years of real
    trend into a hairline so one un-trustworthy cell can be drawn whole.

    So the axis is scaled to cover every point estimate plus the full interval
    of every `ci_status == "ok"` period, and a cautioned period's band is
    allowed to run off the top of the panel — which reads as exactly what it is
    ("this period's uncertainty exceeds the chart"). Nothing is clipped in the
    DATA: `bands.parquet` and the hover tooltip both carry the true bound.
    Across the 16 CorEx issue panels the site actually renders
    (`topic_quality.display_issues` yields 15 anchored issues plus
    `Discovered 5`), this rule lets exactly ONE overflow: Religion & values,
    whose 1785 upper bound is 66.7% against a series maximum of 14.3%.

    Returns None when there is nothing to scale to, leaving plotly's autorange.
    """
    if band is None or band.empty:
        return None
    ok = band[band["ci_status"] == "ok"]
    tops = [band["point"].max()]
    if not ok.empty and ok["hi"].notna().any():
        tops.append(ok["hi"].max())
    top = float(np.nanmax(tops))
    return top * 100 * headroom if top > 0 else None


def band_traces(band: pd.DataFrame | None) -> list[go.Scatter]:
    """The shaded interval, as one `toself` polygon per CONTIGUOUS run.

    Rows with no published interval (`ci_status == "suppressed_n_floor"`) are
    not merely dropped from the vertex list — a single `toself` polygon closes
    straight across such a gap, which draws a band over periods that have none.
    That is visual interpolation, and it is exactly what `connectgaps=False` on
    the solid line already refuses to do. So the frame is split into maximal
    runs of consecutive periods that DO have an interval, and each run gets its
    own polygon; the gap between them is left empty.

    A run of one period has no area, so it is emitted as a vertical whisker
    instead — otherwise a lone period's interval would silently render as
    nothing at all.

    Precisely: runs are maximal runs of consecutive ROWS, which are the same
    thing as consecutive periods only because `bands.series_band` returns a
    period-complete frame ordered by `period_order` — every period in the
    surface gets a row, intervals absent or not. A series-sparse table would
    make two non-adjacent periods adjacent rows and bridge a real gap. That
    invariant is asserted directly — over every series on both surfaces, on
    consecutiveness rather than on a row count — by
    `TestSeriesBandIsPeriodComplete` in `tests/test_band_charts.py`.
    """
    if band is None or band.empty:
        return []
    b = band.reset_index(drop=True)
    have = (b["lo"].notna() & b["hi"].notna()).to_numpy()
    if not have.any():
        return []
    run_id = np.cumsum(~have)          # constant within a run of True
    traces = []
    for _, run in b[have].groupby(run_id[have], sort=True):
        x = run["x"].tolist()
        lo = (run["lo"] * 100).tolist()
        hi = (run["hi"] * 100).tolist()
        if len(x) == 1:
            traces.append(go.Scatter(
                x=[x[0], x[0]], y=[lo[0], hi[0]], mode="lines",
                line=dict(color=BAND_FILL, width=6),
                hoverinfo="skip", showlegend=False,
            ))
            continue
        traces.append(go.Scatter(
            x=x + x[::-1], y=hi + lo[::-1],
            mode="lines", fill="toself", fillcolor=BAND_FILL,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
    return traces


def band_label(b: pd.DataFrame) -> np.ndarray:
    """The interval, as tooltip text — including when there is no interval.

    Formatted in Python rather than by a plotly `:.1f`, because a null bound
    renders as the literal string `nan` through a numeric format directive and
    "95% band nan–nan" is worse than useless. A row is null for two different
    reasons and the tip names which: `interval_unresolvable` (the bootstrap ran
    and could not resolve a width — see `bands._unresolvable_interval`) and the
    `suppressed_n_floor` case (no bootstrap ran at all). Conflating them would
    put the wrong cause in front of the reader.
    """
    out = []
    for lo, hi, unresolvable, n in zip(
        b["lo"], b["hi"], unresolved_mask(b), b["n_speeches"]
    ):
        if bool(unresolvable):
            out.append(f"not resolvable — {int(n)} speeches agree exactly, "
                       "so the bootstrap has no width to report")
        elif pd.isna(lo) or pd.isna(hi):
            out.append("not published — too few speeches to bootstrap")
        else:
            out.append(f"{lo * 100:.1f}–{hi * 100:.1f}")
    return np.array(out, dtype=object)


def unresolved_mask(band: pd.DataFrame) -> pd.Series:
    """`interval_unresolvable` as a plain boolean mask.

    A null flag reads as False — a slice that lost the column's values means
    "nothing is KNOWN to be unresolvable", never "everything is". A slice
    missing the column ENTIRELY reads the same way, so a `bands.parquet`
    predating the column degrades to "no rings, no ring caption" instead of
    crashing the whole page. (`bands.load_bands` raises on that schema long
    before it gets here; this is the hand-built-frame path.)

    Defined once because THREE callers must agree on it: the trace builder that
    draws the rings, `band_label`'s tooltip text, and `write_issue_pages`'s gate
    on the caption that explains them. A page promising a marker its chart does
    not draw is the same species of small lie as the marker's absence, so the
    three must not be able to drift — nor disagree about how defensive to be on
    one column, which is how a "legacy frames do not crash" contract comes to
    hold for one trace and fail on the next one in the same loop.
    """
    if "interval_unresolvable" not in band:
        return pd.Series(False, index=band.index, dtype=bool)
    return band["interval_unresolvable"].fillna(False).astype(bool)


def unresolved_traces(band: pd.DataFrame | None,
                      size: float = 9,
                      unit: str = "% of eligible paragraphs",
                      suffix: str = "") -> list[go.Scatter]:
    """Hollow rings at cells whose bootstrap could not resolve an interval.

    **This, not the suppression, is the reader-facing half of the change.**
    Nulling `lo`/`hi` on a zero-width cell removes a band that already occupied
    zero pixels: on its own it changes literally nothing on the page, and the
    cell would go on reading as a confidently measured value. The ring is the
    positive signal — it says "the share is real, the interval is not
    knowable" without requiring a hover, which touch devices do not have.

    `cliponaxis=False` is load-bearing, not cosmetic. Every flagged cell in this
    corpus has `point == 0` (they are flagged precisely because every speech in
    the period agreed at zero), so the ring sits exactly on the axis floor and
    would otherwise be drawn half-cropped by the plot edge — the same "a visual
    check cannot see a point that is outside the axis" failure that hid the
    whole 1785 period once already.

    `unit` and `suffix` are threaded exactly as `line_traces` threads them, and
    the defaults reproduce today's strings rather than shortening them. A
    tooltip is published prose: hardcoding "% of paragraphs" here would make
    this the one trace in the loop that cannot follow a caller onto a different
    measure, and omitting `suffix` makes it the one tooltip in a 16-panel grid
    that does not name its own panel — in a small-multiples figure that is the
    difference between a reader knowing which issue they are hovering and not.

    Returns an empty list when nothing is flagged, so a caller adds no trace
    rather than an invisible one.
    """
    if band is None or band.empty:
        return []
    flagged = band[unresolved_mask(band)]
    if flagged.empty:
        return []
    y = (flagged["point"] * 100).to_numpy()
    return [go.Scatter(
        x=flagged["x"].to_numpy(), y=y, mode="markers", showlegend=False,
        cliponaxis=False,
        marker=dict(symbol="circle-open", size=size,
                    color=UNRESOLVED_RING,
                    line=dict(color=UNRESOLVED_RING, width=1.8)),
        customdata=np.stack([
            flagged["n_speeches"].to_numpy(),
            flagged["n_paragraphs"].to_numpy(),
        ], axis=-1),
        hovertemplate="%{y:.1f}" + unit + " — no interval<br>"
                      "%{customdata[0]} speeches, %{customdata[1]} paragraphs: "
                      "too few to resolve one<extra>" + suffix + "</extra>",
    )]


def _band_hover(b: pd.DataFrame, unit: str,
                show_components: bool) -> tuple[np.ndarray, str]:
    """Customdata + template putting the interval and its trust gate in the tip.

    `ci_status` and (on the LLM surface) `ci_components` are the machine-readable
    trust gate on every band row; putting them in the tooltip is what stops a
    reader treating a `low_cluster_caution` interval as an `ok` one.
    """
    cols = [
        band_label(b),
        b["n_speeches"].to_numpy(),
        b["ci_status"].to_numpy(),
    ]
    tpl = ("%{y:.1f}" + unit
           + "<br>95% band %{customdata[0]}"
           + "<br>%{customdata[1]} speeches · %{text} paragraphs"
           + " · %{customdata[2]}")
    if show_components:
        cols.append(b["ci_components"].to_numpy())
        tpl += " · %{customdata[3]}"
    return np.stack(cols, axis=-1), tpl


def line_traces(band: pd.DataFrame, unit: str = "% of eligible paragraphs",
                width: float = 2.4, suffix: str = "",
                show_components: bool = False) -> list[go.Scatter]:
    """Trend line, split so thin periods are visibly provisional.

    The charts used to apply a hard `counts >= 40` mask, which deleted thin
    periods outright — presenting absence of data as absence of interest. The
    mask is now a rendering distinction: every period is drawn, but only
    `ci_status == "ok"` periods get the solid line. Thin ones show as a dashed
    underlay carrying an enormous band. Corpus-wide this recovers exactly one
    period (1785: 14 paragraphs, 2 speeches) — it is a correctness fix, not a
    large data recovery.
    """
    custom, tpl = _band_hover(band, unit, show_components)
    y = (band["point"] * 100).to_numpy()
    solid = np.where(band["ci_status"].to_numpy() == "ok", y, np.nan)
    return [
        go.Scatter(
            x=band["x"], y=y, mode="lines", showlegend=False,
            line=dict(color=CAUTION_LINE, width=width * 0.75, dash="dot"),
            text=band["n_paragraphs"], customdata=custom,
            hovertemplate=tpl + "<extra>" + suffix + "</extra>",
        ),
        go.Scatter(
            x=band["x"], y=solid, mode="lines", showlegend=False,
            connectgaps=False, line=dict(color=BLUE_RAMP[4], width=width),
            hoverinfo="skip",
        ),
    ]


def fig_issue_timeline(pl: pd.DataFrame, name: str, label: str,
                       pres_dots: pd.DataFrame,
                       band: pd.DataFrame | None = None) -> go.Figure:
    """One issue's 240 years, with its speech-clustered sampling band.

    `band` is the `bands.parquet` slice for this issue, or None when the table
    has not been built — in which case the line renders unbanded rather than
    with an invented interval. CorEx labels come from a deterministic model with
    no annotator in the loop, so these intervals are `sampling_only` by
    construction and permanently: annotator disagreement is not applicable here,
    not merely absent (see `bands.py`).
    """
    fig = go.Figure()
    for trace in band_traces(band):
        fig.add_trace(trace)
    if band is None:
        d = pl.assign(period=(pl["year"] // bands.PERIOD_YEARS) * bands.PERIOD_YEARS)
        counts = d.groupby("period").size()
        share = (d.groupby("period")[name].mean() * 100)
        share = share[counts >= 40]
        fig.add_trace(go.Scatter(
            x=share.index, y=share.values, mode="lines", showlegend=False,
            line=dict(color=BLUE_RAMP[4], width=2.4),
            hovertemplate="%{y:.1f}% of eligible paragraphs<extra></extra>",
        ))
    else:
        for trace in line_traces(band):
            fig.add_trace(trace)
        # After the line, before the president dots: the ring must sit on top of
        # the dotted trend it annotates, but under nothing that would hide it.
        for trace in unresolved_traces(band):
            fig.add_trace(trace)
    fig.add_trace(go.Scatter(
        x=pres_dots["x"], y=pres_dots["y"], mode="markers", showlegend=False,
        marker=dict(color=BLUE_RAMP[5], size=5, opacity=0.45),
        customdata=np.stack([pres_dots["name"], pres_dots["n"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b>: %{y:.1f}% of their eligible paragraphs "
                      "in this period (%{customdata[1]})<extra></extra>",
    ))
    x_lo, x_hi = x_range_covering([band])
    top = band_y_top(band)
    if top is not None and len(pres_dots):
        # Dots are observed shares, never estimates — a president who spent 30%
        # of a period on one issue must not be cropped out of their own chart.
        top = max(top, float(pres_dots["y"].max()) * 1.05)
    fig.update_layout(
        template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=56, r=24, t=24, b=44), height=380,
        xaxis=dict(range=[x_lo, x_hi], gridcolor=GRID, linecolor=BASELINE,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title="share of eligible paragraphs (%)",
                   gridcolor=GRID, linecolor=BASELINE, rangemode="tozero",
                   range=None if top is None else [0, top],
                   tickfont=dict(color=MUTED, size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# Issues Page V2: one validated evidence model for every public representation
# ---------------------------------------------------------------------------


def _python_scalar(value):
    """Return JSON/CSV-safe Python scalars, with missing values as ``None``."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _stable_topic_color(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return FINE_TOPIC_COLORS[int.from_bytes(digest[:2], "big") % len(FINE_TOPIC_COLORS)]


def _supported_period(rows: list[dict] | tuple[dict, ...]) -> dict:
    supported = [
        row for row in rows
        if row.get("ci_status") == "ok" and row.get("point") is not None
    ]
    if not supported:
        raise ValueError("issue trend has no supported five-year period")
    # Highest point, then earliest declared period order. This tie break is part
    # of the public route contract, not a pandas implementation accident.
    return dict(min(
        supported,
        key=lambda row: (-float(row["point"]), int(row["period_order"])),
    ))


@dataclass(frozen=True)
class IssuePageViewModel:
    """Validated evidence shared by one directory card, detail page and exports."""

    source_issue: str
    label: str
    slug: str
    order: int
    source_kind: str
    source_badge: str
    source_note: str
    trend_rows: tuple[dict, ...]
    highest_supported_period: dict
    president_period_rows: tuple[dict, ...]
    owners: tuple[dict, ...]
    excerpts: tuple[dict, ...]
    fine_topics: tuple[dict, ...]
    fine_topic_rows: tuple[dict, ...]
    initial_topic_names: tuple[str, ...]
    provenance: dict
    previous_issue: dict | None = None
    next_issue: dict | None = None

    @property
    def has_unresolved(self) -> bool:
        return any(bool(row.get("interval_unresolvable")) for row in self.trend_rows)

    def validate(self, *, require_complete: bool = True) -> None:
        if not self.source_issue or not self.label:
            raise ValueError("issue identity and public label must be nonempty")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.slug):
            raise ValueError(f"invalid issue slug {self.slug!r}")
        if self.slug != issue_slug(self.label):
            raise ValueError(f"{self.label}: route slug drifted from its stable label")
        if self.source_kind not in {"anchored_corex", "surfaced_discovered_corex"}:
            raise ValueError(f"invalid issue source kind {self.source_kind!r}")
        if not self.trend_rows:
            raise ValueError(f"{self.label}: broad trend is empty")
        expected_peak = _supported_period(self.trend_rows)
        peak_fields = ("period_start", "period_end", "point", "ci_status")
        if any(
            self.highest_supported_period.get(field) != expected_peak.get(field)
            for field in peak_fields
        ):
            raise ValueError(f"{self.label}: highest-supported period drifted")
        period_orders = [int(row["period_order"]) for row in self.trend_rows]
        if period_orders != sorted(period_orders) or len(period_orders) != len(set(period_orders)):
            raise ValueError(f"{self.label}: broad periods must be unique and ordered")
        for row in self.trend_rows:
            if row.get("interval_unresolvable") and any(
                row.get(field) is not None for field in ("lo_sampling", "hi_sampling", "lo", "hi")
            ):
                raise ValueError(f"{self.label}: unresolved interval published numeric bounds")
        prior_share = float("inf")
        for rank, owner in enumerate(self.owners, 1):
            if int(owner["rank"]) != rank:
                raise ValueError(f"{self.label}: owner ranks are not contiguous")
            if int(owner["n_speeches"]) < OWNER_MIN_SPEECHES:
                raise ValueError(f"{self.label}: ineligible president entered owner ranking")
            denominator = int(owner["denominator"])
            numerator = int(owner["numerator"])
            expected_share = numerator / denominator * 100
            if not np.isclose(float(owner["share_percent"]), expected_share):
                raise ValueError(f"{self.label}: owner share/support mismatch")
            if float(owner["share_percent"]) > prior_share + 1e-12:
                raise ValueError(f"{self.label}: owners are not sorted by share")
            prior_share = float(owner["share_percent"])
        if len(self.owners) > 8:
            raise ValueError(f"{self.label}: more than eight owners")
        receipts = [(row.get("doc_name"), row.get("para_idx")) for row in self.excerpts]
        if len(receipts) != len(set(receipts)):
            raise ValueError(f"{self.label}: excerpt receipts must be unique")
        for excerpt in self.excerpts:
            if excerpt.get("url") != miller_speech_url(str(excerpt.get("doc_name") or "")):
                raise ValueError(f"{self.label}: excerpt URL is not canonical")
        if require_complete:
            roles = [row.get("window_role") for row in self.excerpts]
            if roles != ["early", "highest_supported", "recent"]:
                raise ValueError(f"{self.label}: expected early/supported/recent excerpts")
            supported = self.excerpts[1]
            if (
                int(supported["window_start"]) != int(self.highest_supported_period["period_start"])
                or int(supported["window_end"]) != int(self.highest_supported_period["period_end"])
            ):
                raise ValueError(f"{self.label}: supported excerpt window drifted")
        topic_names = [topic["name"] for topic in self.fine_topics]
        if require_complete and not topic_names:
            raise ValueError(f"{self.label}: no mapped fine topics")
        if topic_names != list(self.initial_topic_names) or len(topic_names) != len(set(topic_names)):
            raise ValueError(f"{self.label}: initial fine-topic selection is incomplete")
        valid_topics = set(topic_names)
        observed_topic_buckets: set[tuple[str, int]] = set()
        for row in self.fine_topic_rows:
            if row["topic"] not in valid_topics:
                raise ValueError(f"{self.label}: fine row references an unmapped topic")
            denominator = int(row["denominator"])
            numerator = int(row["numerator"])
            if denominator <= 0 or numerator < 0:
                raise ValueError(f"{self.label}: invalid fine-topic support")
            if not np.isclose(float(row["share_percent"]), numerator / denominator * 100):
                raise ValueError(f"{self.label}: fine-topic share/support mismatch")
            expected_partial = (
                int(row["requested_start"]) != int(row["observed_start"])
                or int(row["requested_end"]) != int(row["observed_end"])
            )
            if bool(row["partial_period"]) != expected_partial:
                raise ValueError(f"{self.label}: fine-topic partial-period flag drifted")
            if row.get("interval_status") != "not_estimated":
                raise ValueError(f"{self.label}: fine-topic intervals must not be implied")
            observed_topic_buckets.add((str(row["topic"]), int(row["bucket_years"])))
        if require_complete and observed_topic_buckets != {
            (topic, bucket) for topic in valid_topics for bucket in (10, 20)
        }:
            raise ValueError(f"{self.label}: incomplete 10/20-year fine-topic rows")
        fingerprint = self.provenance.get("corpus_fingerprint", {})
        if not {"n_speeches", "n_paragraphs", "doc_name_sha256"}.issubset(fingerprint):
            raise ValueError(f"{self.label}: incomplete corpus provenance")


def validate_issue_view_models(
    views: list[IssuePageViewModel] | tuple[IssuePageViewModel, ...],
    *,
    expected_source_issues: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Validate the whole route set before the first generated file is touched."""
    for view in views:
        view.validate(require_complete=True)
    if expected_source_issues is not None:
        expected = list(expected_source_issues)
        observed = [view.source_issue for view in views]
        if observed != expected:
            raise ValueError(f"issue route order drifted: expected={expected}; observed={observed}")
    if len(views) != 16:
        raise ValueError(f"expected exactly 16 issue pages, observed {len(views)}")
    slugs = [view.slug for view in views]
    if len(slugs) != len(set(slugs)):
        raise ValueError("issue slugs must be unique")
    labels = [view.label for view in views]
    if len(labels) != len(set(labels)):
        raise ValueError("public issue labels must be unique")


def _band_rows(band: pd.DataFrame, source_issue: str) -> tuple[dict, ...]:
    rows: list[dict] = []
    for record in band.to_dict("records"):
        row = {column: _python_scalar(record.get(column)) for column in bands.BANDS_COLUMNS}
        row["source_issue"] = source_issue
        row["period_label"] = f'{int(row["period_start"])}\u2013{int(row["period_end"])}'
        row["share_percent"] = float(row["point"]) * 100
        row["lo_sampling_percent"] = (
            None if row["lo_sampling"] is None else float(row["lo_sampling"]) * 100
        )
        row["hi_sampling_percent"] = (
            None if row["hi_sampling"] is None else float(row["hi_sampling"]) * 100
        )
        row["lo_percent"] = None if row["lo"] is None else float(row["lo"]) * 100
        row["hi_percent"] = None if row["hi"] is None else float(row["hi"]) * 100
        row["disagreement_half_width_percent"] = (
            None
            if row["disagreement_half_width"] is None
            else float(row["disagreement_half_width"]) * 100
        )
        rows.append(row)
    return tuple(rows)


def _president_period_evidence(labels: pd.DataFrame, source_issue: str) -> tuple[dict, ...]:
    data = labels[["doc_name", "president", "year", source_issue]].copy()
    data["period_start"] = (data["year"] // bands.PERIOD_YEARS) * bands.PERIOD_YEARS
    grouped = (
        data.groupby(["president", "period_start"], sort=True)
        .agg(
            numerator=(source_issue, "sum"),
            denominator=(source_issue, "size"),
            n_speeches=("doc_name", "nunique"),
        )
        .reset_index()
    )
    output = []
    for row in grouped.itertuples(index=False):
        denominator = int(row.denominator)
        numerator = int(row.numerator)
        output.append({
            "president": str(row.president),
            "display_name": public_display_name(str(row.president)),
            "profile_slug": slug(str(row.president)),
            "period_start": int(row.period_start),
            "period_end": int(row.period_start) + bands.PERIOD_YEARS - 1,
            "period_label": f"{int(row.period_start)}\u2013{int(row.period_start) + bands.PERIOD_YEARS - 1}",
            "numerator": numerator,
            "denominator": denominator,
            "share_percent": numerator / denominator * 100,
            "n_speeches": int(row.n_speeches),
            "dot_included": denominator >= MIN_DOT_PARAGRAPHS,
            "dot_status": "included" if denominator >= MIN_DOT_PARAGRAPHS else "below_25_paragraph_floor",
        })
    return tuple(output)


def _owner_evidence(labels: pd.DataFrame, source_issue: str) -> tuple[dict, ...]:
    grouped = (
        labels.groupby("president", sort=True)
        .agg(
            numerator=(source_issue, "sum"),
            denominator=(source_issue, "size"),
            n_speeches=("doc_name", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_speeches"] >= OWNER_MIN_SPEECHES].copy()
    grouped["share_percent"] = grouped["numerator"] / grouped["denominator"] * 100
    grouped = grouped.sort_values(
        ["share_percent", "president"], ascending=[False, True], kind="mergesort"
    ).head(8)
    return tuple({
        "rank": rank,
        "president": str(row.president),
        "display_name": public_display_name(str(row.president)),
        "profile_slug": slug(str(row.president)),
        "numerator": int(row.numerator),
        "denominator": int(row.denominator),
        "share_percent": float(row.share_percent),
        "n_speeches": int(row.n_speeches),
        "eligibility_floor": OWNER_MIN_SPEECHES,
    } for rank, row in enumerate(grouped.itertuples(index=False), 1))


def _issue_quotes(
    merged: pd.DataFrame,
    name: str,
    anchors: list[str],
    titles: pd.DataFrame,
    supported_period: dict | None = None,
) -> list[dict]:
    """Select keyed early/supported/recent excerpts without inventing moments."""
    issue_rows = merged[merged[name].astype(bool)].copy()
    if issue_rows.empty:
        return []
    if supported_period is None:
        # Compatibility seam for small hand-built frames. Production always
        # supplies the canonical supported band row.
        peak_start = int((issue_rows["year"] // bands.PERIOD_YEARS * bands.PERIOD_YEARS)
                         .value_counts().sort_index().idxmax())
        supported_period = {
            "period_start": peak_start,
            "period_end": peak_start + bands.PERIOD_YEARS - 1,
        }
    earliest = int(issue_rows["year"].min())
    latest = int(issue_rows["year"].max())
    windows = [
        ("early", "Early window", earliest, earliest + 40),
        (
            "highest_supported",
            "Highest-supported window",
            int(supported_period["period_start"]),
            int(supported_period["period_end"]),
        ),
        ("recent", "Recent window", latest - 15, latest),
    ]
    anchor_pattern = "|".join(rf"\b{re.escape(term)}\w*" for term in anchors)
    excerpts: list[dict] = []
    seen_quotes: set[str] = set()
    seen_receipts: set[tuple[str, int]] = set()
    for role, label, start, end in windows:
        candidates = issue_rows[issue_rows["year"].between(start, end)].copy()
        if candidates.empty:
            continue
        candidates["anchor_hits"] = candidates["text"].str.lower().str.count(anchor_pattern)
        candidates = candidates.sort_values(
            ["anchor_hits", "year", "doc_name", "para_idx"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        chosen = None
        for row in candidates.itertuples(index=False):
            quote = _pick_sentence([str(row.text)], anchors, [])
            receipt = (str(row.doc_name), int(row.para_idx))
            if not quote or quote in seen_quotes or receipt in seen_receipts:
                continue
            chosen = (row, quote, receipt)
            break
        if chosen is None:
            continue
        row, quote, receipt = chosen
        seen_quotes.add(quote)
        seen_receipts.add(receipt)
        title = titles.loc[str(row.doc_name)]
        speech_title = str(title["title"])
        excerpts.append({
            "window_role": role,
            "label": label,
            "window_start": int(start),
            "window_end": int(end),
            "quote": quote,
            "doc_name": str(row.doc_name),
            "para_idx": int(row.para_idx),
            "president": str(title["president"]),
            "display_name": public_display_name(str(title["president"])),
            "title": speech_title,
            "year": int(title["year"]),
            "cite": (
                f"{public_display_name(str(title['president']))}, "
                f"{speech_title.split(':', 1)[-1].strip()}, {int(title['year'])}"
            ),
            "url": miller_speech_url(str(row.doc_name)),
        })
    return excerpts


def _fine_topic_rows(ai_data: dict, topic_names: list[str]) -> tuple[dict, ...]:
    paragraphs = ai_data["paragraphs"][["doc_name", "para_idx", "year"]].copy()
    paragraph_key = ["doc_name", "para_idx"]
    if paragraphs.duplicated(paragraph_key).any():
        raise ValueError("AI paragraph evidence keys must be unique")
    assignments = ai_data["topic_assignments"][paragraph_key + ["topic"]].copy()
    assignment_key = paragraph_key + ["topic"]
    if assignments.duplicated(assignment_key).any():
        raise ValueError("AI topic assignment keys must be unique")
    # Year belongs to the keyed paragraph receipt. Never trust the convenience
    # copy carried on an assignment row: a stale or foreign assignment must not
    # be able to change a numerator under an unchanged denominator/fingerprint.
    assignments = assignments.merge(
        paragraphs,
        on=paragraph_key,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not (assignments["_merge"] == "both").all():
        raise RuntimeError("AI topic assignments contain keys outside the paragraph corpus")
    assignments = assignments.drop(columns="_merge")
    output: list[dict] = []
    for bucket_years in (10, 20):
        paragraph_frame = paragraphs.copy()
        assignment_frame = assignments.copy()
        paragraph_frame["requested_start"] = (
            paragraph_frame["year"] // bucket_years
        ) * bucket_years
        assignment_frame["requested_start"] = (
            assignment_frame["year"] // bucket_years
        ) * bucket_years
        observed = paragraph_frame.groupby("requested_start").agg(
            observed_start=("year", "min"),
            observed_end=("year", "max"),
            denominator=("year", "size"),
        )
        numerator = (
            assignment_frame.groupby(["requested_start", "topic"], observed=True)
            .size()
        )
        for topic in topic_names:
            for requested_start, meta in observed.iterrows():
                requested_start = int(requested_start)
                requested_end = requested_start + bucket_years - 1
                observed_start = int(meta["observed_start"])
                observed_end = int(meta["observed_end"])
                denominator = int(meta["denominator"])
                count = int(numerator.get((requested_start, topic), 0))
                partial = observed_start != requested_start or observed_end != requested_end
                output.append({
                    "bucket_years": bucket_years,
                    "topic": topic,
                    "requested_start": requested_start,
                    "requested_end": requested_end,
                    "observed_start": observed_start,
                    "observed_end": observed_end,
                    "period_label": f"{observed_start}\u2013{observed_end}",
                    "numerator": count,
                    "denominator": denominator,
                    "share_percent": count / denominator * 100,
                    "partial_period": partial,
                    "support_status": "low_support" if denominator < 20 else "descriptive",
                    "interval_status": "not_estimated",
                })
    return tuple(output)


def _issue_provenance(ai_data: dict, active_fingerprint: dict) -> dict:
    band_meta = json.loads(bands.BANDS_META_PATH.read_text())
    crosswalk_document = json.loads(ai_labels.CROSSWALK_PATH.read_text())
    taxonomy_provenance = ai_data["taxonomy"].get("provenance", {})
    fingerprint = dict(band_meta.get("corpus_fingerprint", {}))
    taxonomy_fingerprint = taxonomy_provenance.get("corpus_fingerprint")
    if fingerprint != active_fingerprint:
        raise ValueError("active corpus does not match the CorEx bands fingerprint")
    if not isinstance(taxonomy_fingerprint, dict):
        raise ValueError("AI taxonomy is missing its required corpus fingerprint")
    if taxonomy_fingerprint != active_fingerprint:
        raise ValueError("active corpus does not match the AI taxonomy fingerprint")
    crosswalk_provenance = crosswalk_document.get("provenance", {})
    crosswalk_check = crosswalk_provenance.get("check", {})
    if (
        crosswalk_provenance.get("run_id") != taxonomy_provenance.get("run_id")
        or crosswalk_check.get("complete") is not True
        or int(crosswalk_check.get("n_pages_reconstructable") or 0) != 16
    ):
        raise ValueError("fine-topic crosswalk provenance is incomplete or mismatched")
    surface = band_meta.get("surfaces", {}).get(bands.COREX_SURFACE, {})
    return {
        "corpus_fingerprint": fingerprint,
        "broad_instrument": {
            "artifact": "data/bands.parquet",
            "metadata": "data/bands_meta.json",
            "surface": bands.COREX_SURFACE,
            "labels": surface.get("labels"),
            "grain": surface.get("grain"),
            "ci_components": surface.get("ci_components"),
            "unit": band_meta.get("units"),
            "cluster_unit": band_meta.get("bootstrap", {}).get("cluster_unit"),
            "n_draws": band_meta.get("bootstrap", {}).get("n_draws"),
        },
        "fine_instrument": {
            "taxonomy_artifact": "data/llm_annotations/taxonomy_v1.json",
            "crosswalk_artifact": "data/llm_annotations/crosswalk_v1.json",
            "run_id": taxonomy_provenance.get("run_id"),
            "date": taxonomy_provenance.get("date"),
            "prompt_version": taxonomy_provenance.get("prompt_version"),
            "proposal_model": taxonomy_provenance.get("proposal_model"),
            "coverage_model": taxonomy_provenance.get("coverage_model"),
            "crosswalk": crosswalk_provenance,
            "annotation_models": ai_data.get("manifests", {}).get("primary", {}).get("models", []),
        },
    }


def build_issue_view_models(
    issue_df: pd.DataFrame,
    issue_meta: dict,
    scores: pd.DataFrame,
    ai_data: dict | None = None,
    *,
    paragraphs: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    speeches: pd.DataFrame | None = None,
    band_table: pd.DataFrame | None = None,
) -> tuple[IssuePageViewModel, ...]:
    """Build and validate all 16 routes without writing generated output."""
    from .fetch import PARAGRAPHS_PATH

    paragraphs = (
        pd.read_parquet(PARAGRAPHS_PATH) if paragraphs is None else paragraphs.copy()
    ).reset_index(drop=True)
    paragraph_key_columns = ["doc_name", "para_idx"]
    if paragraphs.duplicated(paragraph_key_columns).any():
        raise ValueError("paragraph corpus keys must be unique")
    labels = (
        pd.read_parquet(issues.PARA_LABELS_PATH) if labels is None else labels.copy()
    ).reset_index(drop=True)
    merged = paragraphs.merge(
        labels, on=["doc_name", "para_idx"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(paragraphs) or len(merged) != len(labels):
        raise RuntimeError("paragraphs and labels key sets diverge - rerun the pipeline")
    speeches = (
        pd.read_parquet(issues.DATA_DIR / "speeches.parquet")
        if speeches is None else speeches.copy()
    )
    if speeches["doc_name"].duplicated().any():
        raise ValueError("speech metadata doc_name keys must be unique")
    active_fingerprint = corpus_fingerprint(speeches=speeches, paragraphs=paragraphs)
    titles = speeches.set_index("doc_name")[["president", "title", "year"]]
    band_table = bands.load_bands() if band_table is None else band_table.copy()
    if band_table is None:
        raise FileNotFoundError("Issues V2 requires the validated bands artifact")
    ai_data = ai_labels.build_ai_data() if ai_data is None else ai_data
    ai_paragraphs = ai_data["paragraphs"]
    if ai_paragraphs.duplicated(paragraph_key_columns).any():
        raise ValueError("AI paragraph evidence keys must be unique")
    paragraph_keys = set(map(
        tuple,
        paragraphs[paragraph_key_columns].itertuples(index=False, name=None),
    ))
    ai_keys = set(map(
        tuple,
        ai_paragraphs[paragraph_key_columns].itertuples(index=False, name=None),
    ))
    if paragraph_keys != ai_keys:
        raise RuntimeError("CorEx and AI paragraph key sets diverge")
    ai_years = ai_paragraphs[paragraph_key_columns + ["year"]].merge(
        labels[paragraph_key_columns + ["year"]],
        on=paragraph_key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_ai", "_corex"),
    )
    if not (ai_years["year_ai"].to_numpy() == ai_years["year_corex"].to_numpy()).all():
        raise RuntimeError("CorEx and AI paragraph years diverge on keyed receipts")
    crosswalk = ai_labels.issue_crosswalk(ai_data)
    provenance = _issue_provenance(ai_data, active_fingerprint)
    names = topic_quality.load_names()
    display = topic_quality.display_issues(issue_meta["issues"], names)
    mapped_by_label = {
        topic_quality.display_name(source_issue, names): crosswalk.get(
            topic_quality.display_name(source_issue, names), []
        )
        for source_issue in display
    }
    all_topic_names = list(dict.fromkeys(
        str(topic["name"])
        for mapped in mapped_by_label.values()
        for topic in mapped
    ))
    all_fine_rows = _fine_topic_rows(ai_data, all_topic_names)
    anchors_all = {**issues.ISSUE_ANCHORS, **_EXTRA_ANCHORS}
    anchored = set(issue_meta["issues"])

    # Check the passed president table against the keyed paragraph evidence.
    issue_index = issue_df.set_index("president") if "president" in issue_df else issue_df
    for source_issue in display:
        if f"share_{source_issue}" not in issue_index:
            raise ValueError(f"issues_president is missing share_{source_issue}")
    if "n_speeches" in scores:
        ineligible = scores[scores["n_speeches"] < OWNER_MIN_SPEECHES]
        if not ineligible.empty:
            raise ValueError("the eligible president score frame contains sparse presidents")

    drafts: list[IssuePageViewModel] = []
    for order, source_issue in enumerate(display):
        label = topic_quality.display_name(source_issue, names)
        route_slug = issue_slug(label)
        band = bands.series_band(band_table, bands.COREX_SURFACE, source_issue)
        if band is None:
            raise ValueError(f"{label}: missing complete CorEx band series")
        trend_rows = _band_rows(band, source_issue)
        highest = _supported_period(trend_rows)
        owners = _owner_evidence(labels, source_issue)
        # Frozen per-president table parity is checked here, while the public
        # model retains exact integer support from the keyed label table.
        for owner in owners:
            expected = float(issue_index.loc[owner["president"], f"share_{source_issue}"]) * 100
            if not np.isclose(expected, owner["share_percent"]):
                raise ValueError(f"{label}: owner evidence disagrees with issues_president")
        excerpts = tuple(_issue_quotes(
            merged,
            source_issue,
            anchors_all[source_issue],
            titles,
            highest,
        ))
        mapped = mapped_by_label[label]
        fine_topics = tuple({
            "name": str(topic["name"]),
            "level1": str(topic["level1"]),
            "definition": str(topic["definition"]),
            "color": _stable_topic_color(str(topic["name"])),
            "source_status": "exploratory_frozen_ai_label",
        } for topic in mapped)
        topic_names = tuple(topic["name"] for topic in fine_topics)
        topic_name_set = set(topic_names)
        source_kind = (
            "anchored_corex" if source_issue in anchored else "surfaced_discovered_corex"
        )
        drafts.append(IssuePageViewModel(
            source_issue=source_issue,
            label=label,
            slug=route_slug,
            order=order,
            source_kind=source_kind,
            source_badge=(
                "Anchored CorEx axis" if source_kind == "anchored_corex"
                else "Surfaced CorEx theme"
            ),
            source_note=(
                "One of the 15 seeded issue axes in the deterministic CorEx model."
                if source_kind == "anchored_corex"
                else "The one unseeded CorEx theme cleared for public display; its broad trend is not an AI label."
            ),
            trend_rows=trend_rows,
            highest_supported_period=highest,
            president_period_rows=_president_period_evidence(labels, source_issue),
            owners=owners,
            excerpts=excerpts,
            fine_topics=fine_topics,
            fine_topic_rows=tuple(
                row for row in all_fine_rows if row["topic"] in topic_name_set
            ),
            initial_topic_names=topic_names,
            provenance=provenance,
        ))

    views = []
    for index, draft in enumerate(drafts):
        previous_issue = None if index == 0 else {
            "slug": drafts[index - 1].slug,
            "label": drafts[index - 1].label,
        }
        next_issue = None if index == len(drafts) - 1 else {
            "slug": drafts[index + 1].slug,
            "label": drafts[index + 1].label,
        }
        views.append(replace(draft, previous_issue=previous_issue, next_issue=next_issue))
    validate_issue_view_models(views, expected_source_issues=display)
    return tuple(views)


def _figure_from_view(view: IssuePageViewModel) -> go.Figure:
    band = pd.DataFrame(view.trend_rows)
    included = [row for row in view.president_period_rows if row["dot_included"]]
    dots = pd.DataFrame({
        "x": [row["period_start"] for row in included],
        "y": [row["share_percent"] for row in included],
        "name": [row["display_name"] for row in included],
        "n": [row["denominator"] for row in included],
    })
    return fig_issue_timeline(
        pd.DataFrame(), view.source_issue, view.label, dots, band
    )


def issue_view_payload(view: IssuePageViewModel, figure: go.Figure | None = None) -> dict:
    """Public JSON shard consumed by both charts and parity tests."""
    figure = _figure_from_view(view) if figure is None else figure
    return {
        "schema": ISSUE_VIEW_SCHEMA,
        "source_issue": view.source_issue,
        "label": view.label,
        "slug": view.slug,
        "unit": "share of eligible paragraphs",
        "highest_supported_period": view.highest_supported_period,
        "trend_rows": list(view.trend_rows),
        "trend_figure": json.loads(pio.to_json(figure)),
        "president_period_rows": list(view.president_period_rows),
        "owners": list(view.owners),
        "excerpts": list(view.excerpts),
        "fine_topics": list(view.fine_topics),
        "fine_topic_rows": list(view.fine_topic_rows),
        "fine_topic_colors": {topic["name"]: topic["color"] for topic in view.fine_topics},
        "initial_topic_names": list(view.initial_topic_names),
        "provenance": view.provenance,
    }


def trend_export_frame(view: IssuePageViewModel) -> pd.DataFrame:
    columns = [
        "source_issue", "period_kind", "period", "period_order", "period_start",
        "period_end", "x", "share_percent", "lo_sampling_percent",
        "hi_sampling_percent", "lo_percent", "hi_percent", "n_paragraphs",
        "n_speeches", "n_paired_paragraphs", "ci_status",
        "interval_unresolvable", "disagreement_band_applied",
        "disagreement_half_width_percent", "ci_components",
        "disagreement_status", "agreement_source",
    ]
    frame = pd.DataFrame(view.trend_rows).reindex(columns=columns)
    frame.insert(1, "unit", "share of eligible paragraphs")
    return frame


def president_period_export_frame(view: IssuePageViewModel) -> pd.DataFrame:
    columns = [
        "president", "display_name", "period_start", "period_end", "numerator",
        "denominator", "share_percent", "n_speeches", "dot_included", "dot_status",
    ]
    frame = pd.DataFrame(view.president_period_rows).reindex(columns=columns)
    frame.insert(0, "unit", "share of eligible paragraphs")
    frame.insert(0, "source_issue", view.source_issue)
    return frame


def fine_topic_export_frame(view: IssuePageViewModel) -> pd.DataFrame:
    columns = [
        "bucket_years", "topic", "requested_start", "requested_end", "observed_start",
        "observed_end", "numerator", "denominator", "share_percent", "partial_period",
        "support_status", "interval_status",
    ]
    frame = pd.DataFrame(view.fine_topic_rows).reindex(columns=columns)
    frame.insert(0, "unit", "share of eligible paragraphs")
    frame.insert(0, "source_issue", view.source_issue)
    return frame


def _format_percent(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _format_interval(row: dict) -> str:
    if row.get("interval_unresolvable"):
        return "Not resolvable"
    if row.get("lo_percent") is None or row.get("hi_percent") is None:
        return "Not published"
    return f'{_format_percent(row["lo_percent"], 1)}\u2013{_format_percent(row["hi_percent"], 1)}'


def _legacy_view(
    label: str,
    owners,
    quotes: list[dict],
    ai_topics: list[dict] | None,
    ai_topic_series: dict[str, dict] | None,
) -> IssuePageViewModel:
    owner_rows = []
    if isinstance(owners, pd.DataFrame) and "share" in owners:
        for rank, (president, row) in enumerate(owners.iterrows(), 1):
            share = float(row["share"])
            owner_rows.append({
                "rank": rank,
                "president": str(president),
                "display_name": public_display_name(str(president)),
                "profile_slug": slug(str(president)),
                "numerator": int(round(share)),
                "denominator": 100,
                "share_percent": share,
                "n_speeches": OWNER_MIN_SPEECHES,
                "eligibility_floor": OWNER_MIN_SPEECHES,
            })
    excerpt_rows = []
    for index, quote in enumerate(quotes or []):
        excerpt_rows.append({
            "window_role": str(quote.get("window_role", "selected")),
            "label": str(quote.get("label", "Selected excerpt")),
            "window_start": int(quote.get("window_start", 0)),
            "window_end": int(quote.get("window_end", 0)),
            "quote": str(quote.get("quote", "")),
            "doc_name": str(quote.get("doc_name", f"legacy-{index}")),
            "para_idx": int(quote.get("para_idx", index)),
            "president": str(quote.get("president", "")),
            "display_name": str(quote.get("display_name", "")),
            "title": str(quote.get("title", "")),
            "year": int(quote.get("year", 0)),
            "cite": str(quote.get("cite", "")),
            "url": str(quote.get("url", "")),
        })
    fine_topics = tuple({
        "name": str(topic.get("name", "")),
        "level1": str(topic.get("level1", "")),
        "definition": str(topic.get("definition", "")),
        "color": _stable_topic_color(str(topic.get("name", ""))),
        "source_status": "exploratory_frozen_ai_label",
    } for topic in (ai_topics or []))
    fine_rows = []
    for bucket, topics in (ai_topic_series or {}).items():
        for topic, series in topics.items():
            for start, value, denominator in zip(
                series.get("x", []), series.get("v", []), series.get("n", [])
            ):
                bucket_years = int(bucket)
                fine_rows.append({
                    "bucket_years": bucket_years,
                    "topic": topic,
                    "requested_start": int(start),
                    "requested_end": int(start) + bucket_years - 1,
                    "observed_start": int(start),
                    "observed_end": int(start) + bucket_years - 1,
                    "period_label": f"{int(start)}\u2013{int(start) + bucket_years - 1}",
                    "numerator": int(round(float(value) / 100 * int(denominator))),
                    "denominator": int(denominator),
                    "share_percent": float(value),
                    "partial_period": False,
                    "support_status": "descriptive",
                    "interval_status": "not_estimated",
                })
    return IssuePageViewModel(
        source_issue=label,
        label=label,
        slug=issue_slug(label) or "issue",
        order=0,
        source_kind="anchored_corex",
        source_badge="Anchored CorEx axis",
        source_note="Deterministic CorEx issue axis.",
        trend_rows=(),
        highest_supported_period={
            "period_start": 0, "period_end": 0, "point": 0.0,
            "share_percent": 0.0, "n_speeches": 0, "n_paragraphs": 0,
            "ci_status": "ok",
        },
        president_period_rows=(),
        owners=tuple(owner_rows),
        excerpts=tuple(excerpt_rows),
        fine_topics=fine_topics,
        fine_topic_rows=tuple(fine_rows),
        initial_topic_names=tuple(topic["name"] for topic in fine_topics),
        provenance={
            "corpus_fingerprint": {
                "n_speeches": 0, "n_paragraphs": 0, "doc_name_sha256": "legacy"
            },
            "broad_instrument": {},
            "fine_instrument": {},
        },
    )


def _broad_table(view: IssuePageViewModel) -> str:
    body = []
    for row in view.trend_rows:
        status_class = (
            "status-unresolved" if row.get("interval_unresolvable")
            else "status-ok" if row.get("ci_status") == "ok"
            else "status-caution"
        )
        body.append(
            "<tr>"
            f'<th scope="row">{html_mod.escape(str(row["period_label"]))}</th>'
            f'<td>{_format_percent(row.get("share_percent"))}</td>'
            f'<td>{html_mod.escape(_format_interval(row))}</td>'
            f'<td>{int(row.get("n_speeches") or 0):,}</td>'
            f'<td>{int(row.get("n_paragraphs") or 0):,}</td>'
            f'<td class="{status_class}">{html_mod.escape(str(row.get("ci_status") or ""))}</td>'
            f'<td>{"Yes" if row.get("interval_unresolvable") else "No"}</td>'
            "</tr>"
        )
    if not body:
        body.append('<tr><td colspan="7">No broad trend rows were supplied.</td></tr>')
    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Exact broad trend values; scroll horizontally if needed">'
        '<table class="evidence-table" id="broad-trend-table">'
        '<caption>One canonical row per five-year period. Percentages are shares of eligible paragraphs.</caption>'
        '<thead><tr><th>Period</th><th>Share</th><th>95% band</th><th>Speeches</th>'
        '<th>Paragraphs</th><th>Status</th><th>Unresolved</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _president_period_table(view: IssuePageViewModel) -> str:
    included = [row for row in view.president_period_rows if row.get("dot_included")]
    body = "".join(
        "<tr>"
        f'<th scope="row">{html_mod.escape(str(row["display_name"]))}</th>'
        f'<td>{html_mod.escape(str(row["period_label"]))}</td>'
        f'<td>{int(row["numerator"]):,}</td><td>{int(row["denominator"]):,}</td>'
        f'<td>{_format_percent(row["share_percent"])}</td>'
        f'<td>{int(row["n_speeches"]):,}</td>'
        "</tr>" for row in included
    )
    if not body:
        body = '<tr><td colspan="6">No president-period dots were supplied.</td></tr>'
    return (
        '<h3>President-period overlay values</h3>'
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Exact president period values; scroll horizontally if needed">'
        '<table class="evidence-table"><caption>Rows plotted as dots; cells below 25 eligible paragraphs remain in the CSV with an exclusion status.</caption>'
        '<thead><tr><th>President</th><th>Period</th><th>Issue paragraphs</th>'
        '<th>Eligible paragraphs</th><th>Share</th><th>Speeches</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def _fine_table(view: IssuePageViewModel, bucket_years: int = DEFAULT_FINE_BUCKET) -> str:
    initial = set(view.initial_topic_names)
    rows = [
        row for row in view.fine_topic_rows
        if int(row["bucket_years"]) == bucket_years and row["topic"] in initial
    ]
    body = "".join(
        "<tr>"
        f'<th scope="row">{html_mod.escape(str(row["topic"]))}</th>'
        f'<td>{int(row["requested_start"])}\u2013{int(row["requested_end"])}</td>'
        f'<td>{int(row["observed_start"])}\u2013{int(row["observed_end"])}</td>'
        f'<td>{int(row["numerator"]):,}</td><td>{int(row["denominator"]):,}</td>'
        f'<td>{_format_percent(row["share_percent"])}</td>'
        f'<td>{"Partial observed bucket" if row["partial_period"] else "Complete observed bucket"}</td>'
        f'<td class="{"status-caution" if row["support_status"] == "low_support" else "status-ok"}">'
        f'{"Low support" if row["support_status"] == "low_support" else "Descriptive"}</td>'
        "</tr>" for row in rows
    )
    if not body:
        body = '<tr><td colspan="8">Select one or more fine topics to inspect exact values.</td></tr>'
    return (
        '<div class="table-scroll" tabindex="0" role="region" '
        'aria-label="Exact fine topic values; scroll horizontally if needed">'
        '<table class="evidence-table" id="fine-topic-table">'
        '<caption>Initially selected topics in 20-year descriptive buckets.</caption>'
        '<thead><tr><th>Topic</th><th>Requested range</th><th>Observed range</th>'
        '<th>Topic paragraphs</th><th>Eligible paragraphs</th><th>Share</th>'
        f'<th>Edge status</th><th>Support status</th></tr></thead><tbody>{body}</tbody></table></div>'
    )


def _owner_rows(view: IssuePageViewModel) -> str:
    if not view.owners:
        return '<p class="dim">No eligible owner rows were supplied.</p>'
    rows = []
    for owner in view.owners:
        name = html_mod.escape(str(owner["display_name"]))
        profile = html_mod.escape(str(owner["profile_slug"]), quote=True)
        width = max(0.0, min(100.0, float(owner["share_percent"])))
        rows.append(f'''<li class="owner-row">
<a class="owner-person" href="../presidents/{profile}.html"><img src="../portraits/{profile}.png"
  alt="" loading="lazy"><span>{name}<small>{int(owner["n_speeches"]):,} eligible speeches</small></span></a>
<span class="owner-track" aria-hidden="true"><span style="width:{width:.6f}%"></span></span>
<span class="owner-value"><strong>{_format_percent(owner["share_percent"])}</strong>
<small>{int(owner["numerator"]):,}/{int(owner["denominator"]):,} paragraphs</small></span></li>''')
    return '<div class="owner-scale" aria-hidden="true"><span>0%</span><span>100%</span></div>' \
        f'<ol class="owner-list">{"".join(rows)}</ol>'


def _excerpt_cards(view: IssuePageViewModel) -> str:
    if not view.excerpts:
        return '<p class="dim">No keyed excerpts were supplied.</p>'
    cards = []
    for excerpt in view.excerpts:
        label = html_mod.escape(str(excerpt["label"]))
        quote = html_mod.escape(str(excerpt["quote"]))
        cite = html_mod.escape(str(excerpt["cite"]))
        url = html_mod.escape(str(excerpt["url"]), quote=True)
        doc_name = html_mod.escape(str(excerpt["doc_name"]))
        cards.append(f'''<blockquote class="issue-excerpt">
<span class="excerpt-label">{label} · {int(excerpt["window_start"])}\u2013{int(excerpt["window_end"])}</span>
<p>“{quote}”</p><cite><a href="{url}" target="_blank" rel="noopener">{cite}</a>
<span class="receipt">{doc_name} · paragraph {int(excerpt["para_idx"])}</span></cite></blockquote>''')
    return f'<div class="excerpt-grid">{"".join(cards)}</div>'


def _fine_definitions(view: IssuePageViewModel) -> str:
    return "".join(
        f'''<article class="fine-definition" style="--topic-color:{html_mod.escape(topic["color"], quote=True)}">
<span>{html_mod.escape(topic["level1"])}</span><h3>{html_mod.escape(topic["name"])}</h3>
<p>{html_mod.escape(topic["definition"])}</p></article>'''
        for topic in view.fine_topics
    )


def _topic_controls(view: IssuePageViewModel) -> str:
    return "".join(
        f'''<label style="--topic-color:{html_mod.escape(topic["color"], quote=True)}">
<input type="checkbox" value="{html_mod.escape(topic["name"], quote=True)}" checked disabled>
{html_mod.escape(topic["name"])}</label>'''
        for topic in view.fine_topics
    )


def _adjacent_navigation(view: IssuePageViewModel) -> str:
    previous = view.previous_issue
    following = view.next_issue
    previous_html = (
        '<span></span>' if previous is None else
        f'''<a href="{html_mod.escape(previous["slug"], quote=True)}.html"><span>Previous issue</span>
<strong>← {html_mod.escape(previous["label"])}</strong></a>'''
    )
    next_html = (
        '<span></span>' if following is None else
        f'''<a class="next" href="{html_mod.escape(following["slug"], quote=True)}.html"><span>Next issue</span>
<strong>{html_mod.escape(following["label"])} →</strong></a>'''
    )
    return f'<nav class="issue-adjacent" aria-label="Adjacent issues">{previous_html}{next_html}</nav>'


def render_issue(
    view_or_label: IssuePageViewModel | str,
    owners=None,
    fig: go.Figure | None = None,
    quotes: list[dict] | None = None,
    has_unresolved: bool = False,
    ai_topics: list[dict] | None = None,
    chart_download: str | None = None,
    ai_topic_series: dict[str, dict] | None = None,
    ai_download: str | None = None,
    *,
    view_data_url: str | None = None,
) -> str:
    """Render one V2 detail page; retain the small legacy test seam."""
    production_view = isinstance(view_or_label, IssuePageViewModel)
    if production_view:
        view = view_or_label
        view.validate(require_complete=True)
        figure = _figure_from_view(view)
        unresolved = view.has_unresolved
    else:
        view = _legacy_view(
            str(view_or_label), owners, quotes or [], ai_topics, ai_topic_series
        )
        figure = go.Figure() if fig is None else fig
        unresolved = bool(has_unresolved)
    payload = issue_view_payload(view, figure)
    if view_data_url:
        data_markup = ""
        data_attribute = html_mod.escape(view_data_url, quote=True)
    else:
        # Preserve Plotly's own safe slash escaping inside the larger safe JSON
        # object. This keeps direct-render security tests representative without
        # duplicating an executable JavaScript assignment.
        placeholder = "__ISSUES_V2_PLOTLY_FIGURE__"
        inline_payload = dict(payload)
        inline_payload["trend_figure"] = placeholder
        encoded = json_for_script(inline_payload).replace(
            json.dumps(placeholder), pio.to_json(figure)
        )
        data_markup = f'<script type="application/json" id="issue-inline-data">{encoded}</script>'
        data_attribute = ""

    label = html_mod.escape(view.label)
    label_lower = html_mod.escape(view.label.lower())
    peak = view.highest_supported_period
    peak_label = (
        f'{int(peak.get("period_start", 0))}\u2013{int(peak.get("period_end", 0))}'
        if view.trend_rows else "Not supplied"
    )
    peak_support = (
        f'{int(peak.get("n_speeches") or 0):,} speeches · '
        f'{int(peak.get("n_paragraphs") or 0):,} paragraphs'
        if view.trend_rows else "Direct-render compatibility view"
    )
    top_owner = view.owners[0] if view.owners else None
    ring_note = (
        " A hollow ring marks a point whose interval could not be resolved at all — "
        "every speech in that period agreed exactly, on too few speeches for the "
        "agreement to mean anything. Its share is plotted; its uncertainty is unknown, "
        "which is not the same as small. "
    ) if unresolved else ""
    ring_attribute = (
        ' data-unresolved-symbol="circle-open"' if production_view and unresolved else ""
    )
    source_class = "anchored" if view.source_kind == "anchored_corex" else "surfaced"
    topic_count = len(view.initial_topic_names)
    initial_fine_rows = [
        row for row in view.fine_topic_rows
        if int(row["bucket_years"]) == DEFAULT_FINE_BUCKET
        and row["topic"] in set(view.initial_topic_names)
    ]
    initial_periods = {row["period_label"] for row in initial_fine_rows}
    initial_partial_periods = {
        row["period_label"] for row in initial_fine_rows if row["partial_period"]
    }
    initial_low_support_periods = {
        row["period_label"]
        for row in initial_fine_rows
        if row["support_status"] == "low_support"
    }
    initial_mode = (
        "Grouped bars"
        if topic_count <= 3 and topic_count * len(initial_periods) <= 48
        else "Heatmap"
    )
    caveat = METHOD_CAVEATS.get(view.label)
    caveat_html = (
        f'<p class="method-caveat"><strong>Cross-method caveat.</strong> {html_mod.escape(caveat)}</p>'
        if caveat else ""
    )
    broad_summary = (
        f"The highest supported five-year period is {peak_label} at "
        f"{_format_percent(peak.get('share_percent'))}, supported by {peak_support}. "
        "Cautioned periods remain visible rather than being treated as zero."
        if view.trend_rows else
        "Interactive trend preview. Exact values remain available below when scripts are unavailable."
    )
    fine_summary = (
        f"All {topic_count} mapped fine {('topic is' if topic_count == 1 else 'topics are')} "
        "selected initially in 20-year descriptive buckets. Edge labels use years actually observed; "
        f"{len(initial_low_support_periods)} low-support periods are flagged."
    )
    fingerprint = view.provenance.get("corpus_fingerprint", {})
    broad_provenance = view.provenance.get("broad_instrument", {})
    fine_provenance = view.provenance.get("fine_instrument", {})
    chart_download = chart_download or f"../data/issues/{view.slug}-broad-trend.csv"
    president_download = f"../data/issues/{view.slug}-president-periods.csv"
    ai_download = ai_download or f"../data/issues/{view.slug}-ai-topics.csv"
    owner_fact = (
        f'<strong>{html_mod.escape(top_owner["display_name"])}</strong>'
        f'<small>{_format_percent(top_owner["share_percent"])} · '
        f'{int(top_owner["n_speeches"]):,} speeches</small>'
        if top_owner else '<strong>Not supplied</strong><small>Eligible ranking</small>'
    )
    topic_word = "topic" if topic_count == 1 else "topics"

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{label} - Presidential Profiles</title>
<style>{PAGE_CSS}\n{NAV_CSS}</style>
<link rel="stylesheet" href="../assets/issues-v2.css">
<script src="../assets/plotly-3.0.1.min.js" defer></script>
<script src="../assets/issues-v2.js" defer></script></head>
<body class="issues-page" data-generated="issues-v2" data-issue-page data-issue-view="{data_attribute}">
<a class="skip-link" href="#main-content">Skip to issue evidence</a>
{nav("../", "issues/index.html")}
<header class="issues-hero"><nav class="breadcrumbs" aria-label="Breadcrumb">
<a href="../index.html">Story</a><span aria-hidden="true">/</span>
<a href="index.html">Issues</a><span aria-hidden="true">/</span><span aria-current="page">{label}</span>
</nav><p class="issues-eyebrow">Issue profile</p><h1>{label}</h1>
<p class="sub">Share of presidential speech about {label_lower}. Broad historical evidence,
selected corpus excerpts, eligible-president emphasis, and a distinct exploratory fine-topic view.</p>
<span class="source-badge {source_class}">{html_mod.escape(view.source_badge)}</span>
<p class="dim">{html_mod.escape(view.source_note)}</p>
<div class="issue-facts"><div class="issue-fact"><span>Highest supported period</span>
<strong>{peak_label} · {_format_percent(peak.get("share_percent"))}</strong><small>{peak_support}</small></div>
<div class="issue-fact"><span>Highest eligible president</span>{owner_fact}</div>
<div class="issue-fact"><span>Exploratory fine crosswalk</span><strong>{topic_count} {topic_word}</strong>
<small>All selected initially · 10/20-year descriptive buckets</small></div></div></header>
<nav class="issue-local-nav" aria-label="On this issue"><ul>
<li><a href="#broad-trend">Broad trend</a></li><li><a href="#selected-excerpts">Selected excerpts</a></li>
<li><a href="#president-emphasis">President emphasis</a></li><li><a href="#fine-topics">Fine topics</a></li>
<li><a href="#data-method">Data & method</a></li></ul></nav>
<main id="main-content">
<section id="broad-trend"><p class="section-kicker">Deterministic broad axis</p><h2>Broad trend</h2>
<p>Each point is the share of eligible paragraphs assigned to this broad CorEx issue in a
five-year period. The shaded 95% sampling band resamples whole speeches; a dotted line marks
periods too thin to trust.{ring_note}Each dot is one president's own share inside one 5-year
period when at least {MIN_DOT_PARAGRAPHS} eligible paragraphs support that cell.</p>
<figure class="issue-figure" aria-labelledby="broad-trend-title" aria-describedby="broad-chart-summary"{ring_attribute}>
<h3 id="broad-trend-title">Share of eligible paragraphs across five-year periods</h3>
<p class="chart-summary" id="broad-chart-summary">{html_mod.escape(broad_summary)}</p>
<div class="chart-scroll" tabindex="0" role="region" aria-label="Interactive broad trend; scroll horizontally on narrow screens">
<div class="chart" id="issue-trend-chart" role="img" aria-labelledby="broad-trend-title" aria-describedby="broad-chart-summary">
<p class="chart-loading">Loading the interactive broad trend. Exact values are available below.</p></div></div>
<p class="scroll-cue">On a narrow screen, swipe or use the arrow keys in the chart region to inspect the full timeline.</p>
<noscript><p>The interactive trend needs JavaScript. The exact table below contains every period and status.</p></noscript>
<figcaption>Solid segments meet the support rule; dotted segments are cautioned. Hollow rings mark unresolved intervals only.</figcaption></figure>
{metrics.lesson_html("confidence_interval")}
<details class="exact-values"><summary>Inspect the evidence</summary>{_broad_table(view)}
{_president_period_table(view)}<div class="download-links">
<a href="{html_mod.escape(chart_download, quote=True)}" download>Download broad-trend CSV</a>
<a href="{html_mod.escape(president_download, quote=True)}" download>Download president-period CSV</a></div></details></section>

<section id="selected-excerpts"><p class="section-kicker">Keyed source receipts</p><h2>Selected corpus excerpts</h2>
<p>These are representative passages selected within an early window, the same highest-supported
five-year window used above, and a recent window. They are examples from the corpus, not claims
that any speech caused the trend.</p>{_excerpt_cards(view)}</section>

<section id="president-emphasis"><p class="section-kicker">Eligible-president comparison</p>
<h2>Presidents who emphasized it most</h2><p><strong>Top eight eligible presidents.</strong>
A president needs at least {OWNER_MIN_SPEECHES} corpus speeches. Bars use an absolute 0–100% scale;
the printed fraction is issue paragraphs divided by all eligible paragraphs for that president.</p>
{_owner_rows(view)}</section>

<section id="fine-topics"><p class="section-kicker">Exploratory frozen AI labels</p>
<h2>Finer AI topics inside this broad issue</h2>
<p class="ai-note">The broad trend above remains the independent deterministic CorEx instrument.
This many-to-many crosswalk uses frozen AI paragraph labels to inspect finer subjects; shares need
not sum to 100%. Ten- and twenty-year buckets are descriptive and have no interval at this grain;
buckets below 20 eligible paragraphs are retained and visibly flagged as low support.
<a href="../label-models.html">Compare the two instruments →</a></p>{caveat_html}
<div class="fine-definitions">{_fine_definitions(view)}</div>
<noscript><p>Topic controls require JavaScript and remain disabled. The exact table retains every initially selected 20-year value.</p></noscript>
<div class="fine-controls"><div><strong>Choose fine topics</strong><p class="dim">All mapped topics start selected.</p></div>
<div class="fine-selects"><label>Bucket size<select id="fine-topic-bucket" disabled>
<option value="20" selected>20 years</option><option value="10">10 years</option></select></label>
<label>View<select id="fine-topic-mode" disabled><option value="auto" selected>Auto</option>
<option value="bars">Grouped bars</option><option value="heatmap">Heatmap</option></select></label></div></div>
<details class="topic-picker"><summary>Topic selection <span id="fine-topic-count">{topic_count} {topic_word} selected</span></summary>
<div class="topic-picker-actions"><button type="button" id="fine-topic-all" disabled>Select all</button>
<button type="button" id="fine-topic-clear" disabled>Clear</button></div>
<fieldset class="topic-controls" id="fine-topic-controls"><legend>Fine topics to display</legend>
{_topic_controls(view)}</fieldset></details>
<p class="fine-status" id="fine-topic-status" role="status" aria-live="polite" aria-atomic="true">
{topic_count} {topic_word} selected · 20-year descriptive buckets · Auto → {initial_mode} ·
{len(initial_partial_periods)} partial edge {"period" if len(initial_partial_periods) == 1 else "periods"} ·
{len(initial_low_support_periods)} low-support {"period" if len(initial_low_support_periods) == 1 else "periods"}.</p>
<figure class="issue-figure fine-figure" aria-labelledby="fine-chart-title" aria-describedby="fine-chart-summary">
<h3 id="fine-chart-title">Exploratory fine-topic shares</h3><p class="chart-summary" id="fine-chart-summary">{html_mod.escape(fine_summary)}</p>
<div class="chart-scroll" tabindex="0" role="region" aria-label="Interactive fine topic chart; scroll horizontally on narrow screens">
<div class="chart" id="fine-topic-chart" role="img" aria-labelledby="fine-chart-title" aria-describedby="fine-chart-summary">
<p class="chart-loading">The fine-topic chart initializes when this section approaches the viewport.</p></div></div>
<p class="scroll-cue">On a narrow screen, swipe or use the arrow keys in the chart region.</p>
<noscript><p>The interactive fine-topic view needs JavaScript. All initially selected 20-year values remain in the exact table.</p></noscript>
<figcaption>Observed edge labels, such as 2020–2026, state the years actually present instead of implying complete future decades.</figcaption></figure>
{metrics.lesson_html("paragraph_share")}
<details class="exact-values"><summary>Inspect the evidence</summary>{_fine_table(view)}
<div class="download-links"><a href="{html_mod.escape(ai_download, quote=True)}" download>Download fine-topic CSV</a></div></details></section>

<section id="data-method"><p class="section-kicker">Provenance and limits</p><h2>Data and method</h2>
<p>All public percentages use the same unit: <strong>share of eligible paragraphs</strong>.
The broad axis and fine-topic crosswalk remain separate instruments with separate uncertainty status.</p>
<div class="provenance-grid"><article class="provenance-card"><h3>Broad CorEx evidence</h3>
<p>{html_mod.escape(str(broad_provenance.get("labels") or "Deterministic CorEx labels"))} ·
{html_mod.escape(str(broad_provenance.get("grain") or "5-year periods"))} ·
{html_mod.escape(str(broad_provenance.get("ci_components") or "sampling only"))}.
Intervals cluster on {html_mod.escape(str(broad_provenance.get("cluster_unit") or "speech"))}.</p></article>
<article class="provenance-card"><h3>Exploratory AI crosswalk</h3><p>Frozen taxonomy
{html_mod.escape(str(fine_provenance.get("run_id") or "not supplied"))} · prompt
{html_mod.escape(str(fine_provenance.get("prompt_version") or "not supplied"))} · models
{html_mod.escape(", ".join(fine_provenance.get("annotation_models") or []) or "not supplied")}.</p></article>
<article class="provenance-card"><h3>Active corpus receipt</h3><p>
{int(fingerprint.get("n_speeches") or 0):,} speeches · {int(fingerprint.get("n_paragraphs") or 0):,} paragraphs ·
<code>{html_mod.escape(str(fingerprint.get("doc_name_sha256") or "not supplied"))}</code>.</p></article>
<article class="provenance-card"><h3>Source corpus</h3><p>Miller Center presidential-speech corpus.
<a href="../methodology.html">Read the full methodology and limitations →</a></p></article></div>
{_adjacent_navigation(view)}</section></main>
<footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
University of Virginia</a>. Generated from validated local artifacts.</p>
<p class="footer-feedback"><a href="../feedback.html">Feedback and accuracy →</a></p></footer>
{data_markup}</body></html>'''


def render_issue_index(entries: list[IssuePageViewModel] | list[dict]) -> str:
    cards = []
    for entry in entries:
        if isinstance(entry, IssuePageViewModel):
            peak = entry.highest_supported_period
            top = entry.owners[0] if entry.owners else None
            source_badge = entry.source_badge
            source_class = "anchored" if entry.source_kind == "anchored_corex" else "surfaced"
            peak_label = f'{int(peak["period_start"])}\u2013{int(peak["period_end"])}'
            peak_detail = (
                f'{_format_percent(peak["share_percent"])} · '
                f'{int(peak["n_speeches"]):,} speeches · {int(peak["n_paragraphs"]):,} paragraphs'
            )
            top_detail = (
                f'{html_mod.escape(top["display_name"])} · {_format_percent(top["share_percent"])} · '
                f'{int(top["n_speeches"]):,} speeches · {int(top["numerator"]):,}/{int(top["denominator"]):,} paragraphs'
                if top else "No eligible owner supplied"
            )
            n_topics = len(entry.initial_topic_names)
            route_slug = entry.slug
            label = entry.label
        else:
            source_badge = str(entry.get("source_badge", "Anchored CorEx axis"))
            source_class = "anchored"
            peak_start = int(entry.get("peak", 0))
            peak_label = f"{peak_start}\u2013{peak_start + 4}"
            peak_detail = "Support not supplied"
            top_detail = html_mod.escape(str(entry.get("top", "Not supplied")))
            n_topics = int(entry.get("n_ai_topics", 0))
            route_slug = str(entry["slug"])
            label = str(entry["label"])
        topic_word = "topic" if n_topics == 1 else "topics"
        cards.append(f'''<li><article class="directory-card">
<span class="source-badge {source_class}">{html_mod.escape(source_badge)}</span>
<a href="{html_mod.escape(route_slug, quote=True)}.html"><h2>{html_mod.escape(label)}</h2></a>
<template><div class="name">{html_mod.escape(label)}</div></template><dl>
<div><dt>Highest supported period</dt><dd>{peak_label} · {peak_detail}</dd></div>
<div><dt>Highest eligible president</dt><dd>{top_detail}</dd></div>
<div><dt>Exploratory fine crosswalk</dt><dd>{n_topics} finer AI {topic_word}</dd></div>
</dl></article></li>''')
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Issues - Presidential Profiles</title><style>{PAGE_CSS}\n{NAV_CSS}</style>
<link rel="stylesheet" href="../assets/issues-v2.css"></head>
<body class="issues-page" data-generated="issues-v2"><a class="skip-link" href="#main-content">Skip to issues</a>
{nav("../", "issues/index.html")}
<header class="issues-hero"><nav class="breadcrumbs" aria-label="Breadcrumb">
<a href="../index.html">Story</a><span aria-hidden="true">/</span><span aria-current="page">Issues</span></nav>
<p class="issues-eyebrow">Directory</p><h1>Issue profiles</h1>
<p class="sub">Fifteen anchored issue axes plus one surfaced discovered theme, in the model's
canonical order. Each profile uses one supported-period contract for its summary, chart,
selected excerpts, exact tables, and downloads.</p></header>
<main class="directory-main" id="main-content"><h2>All 16 issues</h2>
<p>Broad trends are deterministic CorEx paragraph shares. Finer AI topics are a separate,
exploratory many-to-many crosswalk and are labeled as such.</p>
<ol class="issue-directory">{"".join(cards)}</ol></main>
<footer><p>Data: <a href="https://data.millercenter.org">Miller Center of Public Affairs,
University of Virginia</a>. <a href="../methodology.html">Methods and provenance</a>.</p>
<p class="footer-feedback"><a href="../feedback.html">Feedback and accuracy →</a></p></footer>
</body></html>'''


def _prune_stale_issue_outputs(
    out_dir: Path,
    data_dir: Path,
    *,
    expected_pages: set[str],
    expected_data: set[str],
    current_slugs: set[str],
) -> None:
    """Remove only outputs previously declared as Issues-owned."""
    manifest_path = data_dir / "manifest.json"
    prior_pages: set[str] = set()
    prior_data: set[str] = set()
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
            prior_pages = set(prior.get("issue_pages", []))
            prior_data = set(prior.get("data_files", []))
        except (json.JSONDecodeError, TypeError):
            # A malformed manifest is validated elsewhere; do not broaden a
            # cleanup because its ownership declaration cannot be trusted.
            prior_pages = set()
            prior_data = set()

    def owned_name(name: str) -> bool:
        return bool(name) and Path(name).name == name

    for name in sorted(prior_pages - expected_pages):
        path = out_dir / name
        if owned_name(name) and path.is_file():
            path.unlink()
    for name in sorted(prior_data - expected_data):
        path = data_dir / name
        if owned_name(name) and path.is_file():
            path.unlink()

    # V2 pages carry an explicit ownership marker, so a route removed between
    # manifests is still safe to prune. Unmarked sentinels are left untouched.
    for path in out_dir.glob("*.html"):
        if path.name in expected_pages:
            continue
        try:
            owned = 'data-generated="issues-v2"' in path.read_text()
        except UnicodeDecodeError:
            owned = False
        if owned:
            path.unlink()

    # The one legacy artifact V2 deliberately retires was a Plotly geometry
    # CSV named exactly after each still-canonical slug.
    for route_slug in current_slugs:
        legacy = data_dir / f"{route_slug}.csv"
        if legacy.is_file():
            legacy.unlink()


def write_issue_pages(
    site_dir,
    issue_df: pd.DataFrame,
    issue_meta: dict,
    scores: pd.DataFrame,
    faces: dict,
    ai_data: dict | None = None,
) -> None:
    """Validate and prepare every Issues V2 representation before writing."""
    del faces  # Portrait paths are shared static assets; data URIs are unnecessary here.
    site_dir = Path(site_dir)
    views = build_issue_view_models(issue_df, issue_meta, scores, ai_data)

    # Prepare every representation before mutating docs/. A bad label, figure,
    # JSON value or export schema therefore fails the build without leaving a
    # half-new directory beside half-old evidence.
    page_text: dict[str, str] = {}
    data_text: dict[str, str] = {}
    csv_text: dict[str, str] = {}
    for view in views:
        figure = _figure_from_view(view)
        has_unresolved = pd.Series(
            [row["interval_unresolvable"] for row in view.trend_rows], dtype=bool
        ).any()
        payload = issue_view_payload(view, figure)
        data_name = f"{view.slug}.json"
        data_text[data_name] = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, indent=2
        ) + "\n"
        csv_text[f"{view.slug}-broad-trend.csv"] = trend_export_frame(view).to_csv(index=False)
        csv_text[f"{view.slug}-president-periods.csv"] = (
            president_period_export_frame(view).to_csv(index=False)
        )
        csv_text[f"{view.slug}-ai-topics.csv"] = fine_topic_export_frame(view).to_csv(index=False)
        page_text[f"{view.slug}.html"] = render_issue(
            view,
            has_unresolved=has_unresolved,
            chart_download=f"../data/issues/{view.slug}-broad-trend.csv",
            ai_download=f"../data/issues/{view.slug}-ai-topics.csv",
            view_data_url=f"../data/issues/{view.slug}.json",
        )
    page_text["index.html"] = render_issue_index(list(views))

    out_dir = site_dir / "issues"
    data_dir = site_dir / "data" / "issues"
    asset_dir = site_dir / "assets"
    expected_pages = set(page_text)
    expected_data = set(data_text) | set(csv_text) | {"manifest.json"}
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "issues-v2.css").write_text(ISSUES_CSS.strip() + "\n")
    (asset_dir / "issues-v2.js").write_text(ISSUES_JS.strip() + "\n")
    for name, text in page_text.items():
        (out_dir / name).write_text(text)
    for name, text in data_text.items():
        (data_dir / name).write_text(text)
    for name, text in csv_text.items():
        (data_dir / name).write_text(text)
    # Commit current outputs before removing anything. If a filesystem write
    # fails, the previous ownership manifest and every stale route remain in
    # place rather than leaving a deletion-only half update.
    _prune_stale_issue_outputs(
        out_dir,
        data_dir,
        expected_pages=expected_pages,
        expected_data=expected_data,
        current_slugs={view.slug for view in views},
    )
    manifest = {
        "schema": "issues-v2-output-manifest",
        "issue_pages": sorted(expected_pages),
        "data_files": sorted(expected_data),
        "slugs": [view.slug for view in views],
        "corpus_fingerprint": views[0].provenance["corpus_fingerprint"],
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"  wrote {len(views)} validated Issues V2 pages + directory")
