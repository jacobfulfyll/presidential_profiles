"""The rendering layer for `data/bands.parquet` — `issues_site` + `site`.

The band table is only half the feature; the other half is drawing it without
telling a lie the data does not support. Four things carry that weight:

1. **Polygon run-splitting.** A single `toself` polygon closes straight across a
   period with no interval, painting a band over periods that have none — the
   visual interpolation `connectgaps=False` already refuses on the solid line.
   `band_traces` splits the frame into maximal runs of periods that DO have an
   interval. Every boundary case is pinned directly (leading gap, trailing gap,
   two adjacent gaps, an isolated singleton, all-present), because
   `np.cumsum(~have)` is the kind of index arithmetic that is right for four
   shapes and wrong for the fifth.

2. **A one-period run must still be visible.** It has no area, so it is drawn
   as a vertical whisker. Emitting a zero-area polygon would render as nothing
   at all — the same "absence of data drawn as absence of interest" this task
   set out to remove.

3. **The panel ceiling.** 1785's upper bound on one issue is 66.7% against a
   series that never exceeds 14.3%. Autoscaling to it flattens 240 years of real
   trend into a hairline. The axis is scaled by trustworthy content only and the
   cautioned band is allowed to overflow — but nothing is clipped in the DATA.

4. **Derived prose.** `llm_divergence_sentence` is computed from the band table
   at render time rather than typed into `SECTIONS`, so the superlative cannot
   outlive the numbers behind it. Tested by feeding it a table and watching the
   sentence follow.

Everything here runs on hand-authored frames; no figure is written to disk and
no page is built from the real corpus.

--------------------------------------------------------------------------
Fixture convention — READ THIS BEFORE ADDING A TEST
--------------------------------------------------------------------------

`_band`, `_llm_table` and `TestPresidentPeriodDots._labels` all fill
unspecified fields from ``r.get(key, <one scalar>)``. That keeps the
hand-computed oracles readable, but it has one consequence, and it has already
produced five separate defects across this stage's two test files:

    **A field a test does not explicitly vary is IDENTICAL on every row.**

So the rule for any new test is:

    **No helper default may be shared by two rows the test distinguishes.**

If the claim is "this period is flagged and that one is not", or "the upper
edge is traced left-to-right and the lower edge back", then every field
carrying that distinction must be passed explicitly per row.
`test_all_periods_present_is_exactly_one_polygon` is the worked example: it
varies `lo` and `hi` across all four periods precisely because a flat band's
lower edge could be traced in either direction and the polygon would look
identical, leaving `lo[::-1]` unasserted.

Do NOT fix this by making the helpers emit per-row-varying defaults — the
oracles are hand-computed against the flat ones. Vary explicitly, at the call
site, and say in the test why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from presidential_profiles import bands as B
from presidential_profiles import issues_site as S
from presidential_profiles import site


def _band(rows: list[dict]) -> pd.DataFrame:
    """A `bands.parquet` slice: ``{"x", "lo", "hi"}`` with NaN bounds meaning
    "no published interval". `ci_status` defaults to "ok", or to
    `suppressed_n_floor` when the bounds are absent.

    Unspecified fields take one shared default across every row — see the
    module docstring's fixture convention before relying on one.
    """
    out = []
    for r in rows:
        lo, hi = r.get("lo", np.nan), r.get("hi", np.nan)
        missing = pd.isna(lo) or pd.isna(hi)
        out.append({
            "x": r["x"],
            "period": str(r["x"]),
            "period_order": r["x"],
            "point": r.get("point", 0.10),
            "lo": lo,
            "hi": hi,
            "lo_sampling": lo,
            "hi_sampling": hi,
            "n_speeches": r.get("n_speeches", 12),
            "ci_status": r.get(
                "ci_status", "suppressed_n_floor" if missing else "ok"),
            "ci_components": r.get("ci_components", "sampling_only"),
        })
    return pd.DataFrame(out)


def _polygons(traces: list[go.Scatter]) -> list[go.Scatter]:
    return [t for t in traces if t.fill == "toself"]


def _xs(trace: go.Scatter) -> list:
    return list(trace.x)


# --------------------------------------------------------------------------- #
# 1. polygon run-splitting
# --------------------------------------------------------------------------- #
class TestBandTraceRunSplitting:
    def test_all_periods_present_is_exactly_one_polygon(self):
        """Today's real-data case for every issue but the thinnest: one closed
        shape, hi across then lo back.

        The bounds VARY across periods on purpose. With a flat lo and hi the
        lower edge could be traced in either direction and the polygon would
        look identical — the test would pass while `lo[::-1]` was gone and the
        closed shape crossed over itself.
        """
        band = _band([
            {"x": 1900, "lo": 0.10, "hi": 0.20},
            {"x": 1905, "lo": 0.11, "hi": 0.24},
            {"x": 1910, "lo": 0.12, "hi": 0.28},
            {"x": 1915, "lo": 0.13, "hi": 0.32},
        ])
        traces = S.band_traces(band)
        assert len(traces) == 1
        (poly,) = traces
        assert poly.fill == "toself"
        assert _xs(poly) == [1900, 1905, 1910, 1915, 1915, 1910, 1905, 1900]
        # upper edge left-to-right, then the lower edge back — percent, not
        # fractions, and each y must pair with the x beside it.
        assert list(poly.y) == pytest.approx(
            [20.0, 24.0, 28.0, 32.0, 13.0, 12.0, 11.0, 10.0])
        for x, y in zip(_xs(poly), poly.y):
            row = band[band["x"] == x].iloc[0]
            assert y in (pytest.approx(row["lo"] * 100), pytest.approx(row["hi"] * 100))

    def test_a_leading_gap_starts_the_polygon_after_it(self):
        band = _band([
            {"x": 1785},
            {"x": 1790, "lo": 0.1, "hi": 0.2},
            {"x": 1795, "lo": 0.1, "hi": 0.2},
            {"x": 1800, "lo": 0.1, "hi": 0.2},
        ])
        (poly,) = S.band_traces(band)
        assert 1785 not in _xs(poly)
        assert min(_xs(poly)) == 1790

    def test_a_trailing_gap_ends_the_polygon_before_it(self):
        band = _band([
            {"x": 1990, "lo": 0.1, "hi": 0.2},
            {"x": 1995, "lo": 0.1, "hi": 0.2},
            {"x": 2000, "lo": 0.1, "hi": 0.2},
            {"x": 2005},
        ])
        (poly,) = S.band_traces(band)
        assert 2005 not in _xs(poly)
        assert max(_xs(poly)) == 2000

    def test_two_adjacent_gaps_split_the_band_in_two_and_nothing_spans_them(self):
        """The defect this replaced, stated directly: a single polygon would
        close across the hole and paint a band over two periods that have
        none."""
        band = _band([
            {"x": 1900, "lo": 0.1, "hi": 0.2},
            {"x": 1905, "lo": 0.1, "hi": 0.2},
            {"x": 1910},
            {"x": 1915},
            {"x": 1920, "lo": 0.3, "hi": 0.4},
            {"x": 1925, "lo": 0.3, "hi": 0.4},
        ])
        polys = _polygons(S.band_traces(band))
        assert len(polys) == 2
        assert set(_xs(polys[0])) == {1900, 1905}
        assert set(_xs(polys[1])) == {1920, 1925}
        for poly in polys:
            assert 1910 not in _xs(poly) and 1915 not in _xs(poly)
            # and no shape merely straddles the hole without a vertex in it
            assert not (min(_xs(poly)) < 1910 < max(_xs(poly)))

    def test_a_single_gap_also_splits_the_band(self):
        band = _band([
            {"x": 1900, "lo": 0.1, "hi": 0.2},
            {"x": 1905},
            {"x": 1910, "lo": 0.1, "hi": 0.2},
            {"x": 1915, "lo": 0.1, "hi": 0.2},
        ])
        traces = S.band_traces(band)
        assert len(traces) == 2
        assert not any(1905 in _xs(t) for t in traces)

    def test_an_isolated_period_is_drawn_as_a_whisker_not_a_zero_area_polygon(self):
        """A run of one has no area. Emitted as a polygon it would render as
        nothing at all, silently deleting the interval of exactly the period
        most in need of one."""
        band = _band([
            {"x": 1785, "lo": 0.05, "hi": 0.60},
            {"x": 1790},
            {"x": 1795, "lo": 0.10, "hi": 0.20},
            {"x": 1800, "lo": 0.10, "hi": 0.20},
        ])
        traces = S.band_traces(band)
        assert len(traces) == 2
        whisker, poly = traces
        assert whisker.fill is None
        assert _xs(whisker) == [1785, 1785]
        assert list(whisker.y) == [5.0, 60.0]
        assert whisker.line.width and whisker.line.width > 0
        assert poly.fill == "toself"

    def test_two_isolated_periods_each_get_their_own_whisker(self):
        band = _band([
            {"x": 1785, "lo": 0.05, "hi": 0.6},
            {"x": 1790},
            {"x": 1795, "lo": 0.1, "hi": 0.5},
            {"x": 1800},
        ])
        traces = S.band_traces(band)
        assert len(traces) == 2
        assert all(t.fill is None and len(set(_xs(t))) == 1 for t in traces)

    def test_a_frame_with_no_intervals_at_all_draws_nothing(self):
        assert S.band_traces(_band([{"x": 1785}, {"x": 1790}])) == []

    def test_absent_and_empty_tables_draw_nothing(self):
        assert S.band_traces(None) == []
        assert S.band_traces(_band([])) == []

    def test_a_half_missing_bound_counts_as_no_interval(self):
        """`lo` present with `hi` NaN is not half a band — it is no band."""
        band = _band([
            {"x": 1900, "lo": 0.1, "hi": 0.2},
            {"x": 1905, "lo": 0.1, "hi": np.nan},
            {"x": 1910, "lo": 0.1, "hi": 0.2},
        ])
        traces = S.band_traces(band)
        assert len(traces) == 2
        assert not any(1905 in _xs(t) for t in traces)

    def test_row_order_in_the_frame_does_not_change_the_runs(self):
        """`band_traces` reindexes; `series_band` is what guarantees x-order.
        A frame arriving in period order must draw the same shape whatever its
        positional index happens to be."""
        rows = [{"x": 1900, "lo": 0.1, "hi": 0.2}, {"x": 1905},
                {"x": 1910, "lo": 0.1, "hi": 0.2}]
        shifted = _band(rows)
        shifted.index = [7, 8, 9]
        shape = [_xs(t) for t in S.band_traces(shifted)]
        # Anchored absolutely as well as against the unshifted frame: an
        # invariance test that only compares one call of a function to another
        # call of the same function passes when BOTH are wrong.
        assert shape == [[1900, 1900], [1910, 1910]]
        assert shape == [_xs(t) for t in S.band_traces(_band(rows))]

    def test_traces_never_appear_in_the_legend_or_the_hover(self):
        """The band is context for the line, not a series of its own."""
        band = _band([{"x": 1900, "lo": 0.1, "hi": 0.2},
                      {"x": 1905, "lo": 0.1, "hi": 0.2}])
        traces = S.band_traces(band)
        assert len(traces) == 1, "count first — an empty list satisfies any for-loop"
        for trace in traces:
            assert trace.showlegend is False
            assert trace.hoverinfo == "skip"


# --------------------------------------------------------------------------- #
# 2. the panel ceiling
# --------------------------------------------------------------------------- #
class TestBandYTop:
    def test_ceiling_ignores_the_interval_of_an_untrustworthy_period(self):
        band = _band([
            {"x": 1785, "lo": 0.0, "hi": 0.667, "point": 0.05,
             "ci_status": "low_cluster_caution"},
            {"x": 1900, "lo": 0.08, "hi": 0.12, "point": 0.10},
        ])
        top = S.band_y_top(band)
        assert top == pytest.approx(12.0 * 1.05)
        assert top < 66.7, "a 2-speech interval must not set the axis"

    def test_ceiling_still_covers_every_point_estimate(self):
        """A cautioned period's POINT is real data; only its interval is
        allowed off the top of the panel."""
        band = _band([
            {"x": 1785, "lo": 0.0, "hi": 0.9, "point": 0.30,
             "ci_status": "low_cluster_caution"},
            {"x": 1900, "lo": 0.08, "hi": 0.12, "point": 0.10},
        ])
        assert S.band_y_top(band) == pytest.approx(30.0 * 1.05)

    def test_no_trustworthy_rows_falls_back_to_the_points(self):
        band = _band([{"x": 1785, "lo": 0.0, "hi": 0.9, "point": 0.2,
                       "ci_status": "low_cluster_caution"}])
        assert S.band_y_top(band) == pytest.approx(20.0 * 1.05)

    def test_absent_or_all_zero_input_leaves_plotly_to_autorange(self):
        assert S.band_y_top(None) is None
        assert S.band_y_top(_band([])) is None
        assert S.band_y_top(_band([{"x": 1900, "lo": 0.0, "hi": 0.0,
                                    "point": 0.0}])) is None


# --------------------------------------------------------------------------- #
# 3. the trend line and its hover
# --------------------------------------------------------------------------- #
class TestLineTraces:
    @pytest.fixture
    def band(self) -> pd.DataFrame:
        return _band([
            {"x": 1785, "lo": 0.0, "hi": 0.6, "point": 0.20, "n_speeches": 2,
             "ci_status": "low_cluster_caution"},
            {"x": 1900, "lo": 0.08, "hi": 0.12, "point": 0.10, "n_speeches": 40},
            {"x": 1905, "lo": 0.09, "hi": 0.13, "point": 0.11, "n_speeches": 44},
        ])

    def test_the_solid_line_is_broken_at_periods_too_thin_to_trust(self, band):
        """The `>= 40` mask became a rendering distinction: every period is
        drawn, but only an `ok` one gets the solid line."""
        dotted, solid = S.line_traces(band)
        assert list(dotted.y) == [20.0, 10.0, 11.0]
        assert np.isnan(solid.y[0])
        assert list(solid.y[1:]) == [10.0, 11.0]
        assert solid.connectgaps is False
        assert dotted.line.dash == "dot"

    def test_the_hover_carries_the_interval_and_the_trust_gate(self, band):
        """`ci_status` is the machine-readable trust gate; putting it in the tip
        is what stops a reader treating a cautioned interval as an `ok` one."""
        dotted, _ = S.line_traces(band)
        assert list(dotted.customdata[0]) == [0.0, 60.0, 2, "low_cluster_caution"]
        assert "95% band" in dotted.hovertemplate
        assert "customdata[3]" in dotted.hovertemplate

    def test_components_are_shown_only_where_they_vary(self, band):
        """Surface A is permanently `sampling_only`, so printing it on every
        CorEx tip would be noise; Surface B's tip must say which components its
        interval contains."""
        plain, _ = S.line_traces(band)
        assert plain.customdata.shape[1] == 4
        withc, _ = S.line_traces(band, show_components=True)
        assert withc.customdata.shape[1] == 5
        assert list(withc.customdata[:, 4]) == ["sampling_only"] * 3
        assert "customdata[4]" in withc.hovertemplate


# --------------------------------------------------------------------------- #
# 4. per-president-per-period dots
# --------------------------------------------------------------------------- #
class TestPresidentPeriodDots:
    def _labels(self, specs: list[dict]) -> pd.DataFrame:
        """A stand-in for `paragraph_issues.parquet`.

        Unspecified fields take one shared default across every row — see the
        module docstring's fixture convention before relying on one. `doc`
        defaults to the president's own name, so two specs for one president
        collapse into ONE speech unless a test gives them distinct `doc`s.
        """
        rows = []
        for spec in specs:
            for i in range(spec["n"]):
                rows.append({
                    "doc_name": spec.get("doc", spec["president"]),
                    "para_idx": i,
                    "president": spec["president"],
                    "year": spec["year"],
                    "war": bool(i < spec.get("hits", 0)),
                })
        return pd.DataFrame(rows)

    def test_a_president_spanning_two_periods_gets_a_dot_in_each(self):
        """The upgrade: the old overlay could only say "this president, on
        average, somewhere in here" by plotting one dot at the term midpoint."""
        pl = self._labels([
            {"president": "Lincoln", "year": 1861, "n": 40, "hits": 10},
            {"president": "Lincoln", "year": 1865, "n": 40, "hits": 20},
        ])
        dots = S.president_period_dots(pl, "war")
        assert list(dots["x"]) == [1860, 1865]
        assert list(dots["y"]) == [25.0, 50.0]
        assert set(dots["name"]) == {"Lincoln"}

    def test_a_thin_president_period_cell_is_dropped_not_plotted(self):
        """A dot must never carry less evidence than the band it sits on."""
        assert S.MIN_DOT_PARAGRAPHS == 25
        pl = self._labels([
            {"president": "Garfield", "year": 1881, "n": 24, "hits": 24},
            {"president": "Arthur", "year": 1882, "n": 25, "hits": 5},
        ])
        dots = S.president_period_dots(pl, "war")
        assert list(dots["name"]) == ["Arthur"]
        assert list(dots["n"]) == [25]

    def test_two_presidents_inside_one_period_get_one_dot_each(self):
        pl = self._labels([
            {"president": "Hoover", "year": 1932, "n": 40, "hits": 4},
            {"president": "Roosevelt", "year": 1933, "n": 40, "hits": 36},
        ])
        dots = S.president_period_dots(pl, "war")
        assert len(dots) == 2
        assert set(dots["x"]) == {1930}
        # Keyed by president, not sorted: `sorted(dots["y"])` would pass just as
        # happily if the two presidents' shares were attached to each other's
        # names, which is the one thing a per-president dot must never do.
        assert dict(zip(dots["name"], dots["y"])) == {"Hoover": 10.0,
                                                      "Roosevelt": 90.0}

    def test_periods_use_the_same_five_year_grain_as_the_band(self):
        assert S.president_period_dots.__defaults__[-1] == B.PERIOD_YEARS


# --------------------------------------------------------------------------- #
# 5. the derived divergence sentence
# --------------------------------------------------------------------------- #
def _llm_table(rows: list[dict]) -> pd.DataFrame:
    """A Surface B slice for the prose/figure tests.

    Unspecified fields take one shared default across every row — see the
    module docstring's fixture convention before relying on one. `point` and
    `n_paragraphs` matter most here: the panel ranking is paragraph-weighted
    corpus share, so a test about ranking must vary BOTH explicitly.
    """
    return pd.DataFrame([
        {
            "surface": r.get("surface", B.LLM_SURFACE),
            "series": r["series"],
            "period": r["period"],
            "point": r.get("point", 0.1),
            "n_paragraphs": r.get("n_paragraphs", 100),
            "disagreement_half_width": r.get("hw", 0.01),
        }
        for r in rows
    ])


class TestLlmDivergenceSentence:
    def test_the_superlatives_follow_the_table_it_is_given(self):
        table = _llm_table([
            {"series": "Trade", "period": "The founding", "hw": 0.05},
            {"series": "War", "period": "The founding", "hw": 0.03},
            {"series": "Trade", "period": "Expansion", "hw": 0.005},
            {"series": "War", "period": "Expansion", "hw": 0.005},
        ])
        sentence = site.llm_divergence_sentence(table)
        # The sentence reports the FULL annotator-to-annotator gap, which is
        # TWICE the stored half-width: mean hw (0.05 + 0.03) / 2 = 0.04, so the
        # printed gap is 8.00 pp; Expansion's (0.005 + 0.005) / 2 = 0.005 -> 1.00.
        assert "diverge most in The founding — by 8.00 percentage points" in sentence
        assert "closest on Expansion (1.00)" in sentence
        assert "across all 2 topics" in sentence

    def test_the_printed_number_is_the_full_gap_not_the_half_width(self):
        """Regression pin for a real defect: the sentence first printed the mean
        HALF-width while describing it as the gap between the two readers,
        leaving a reader off by exactly 2x on the one quantity this section
        exists to expose. The factor is pinned here rather than trusted to the
        wording, and the halving must still be named so the stored column and
        the printed number cannot be confused."""
        hw = 0.0125
        table = _llm_table([
            {"series": "Trade", "period": "The founding", "hw": hw},
            {"series": "Trade", "period": "Expansion", "hw": 0.001},
        ])
        sentence = site.llm_divergence_sentence(table)
        assert f"{hw * 200:.2f} percentage points" in sentence      # 2.50, not 1.25
        assert f"{hw * 100:.2f} percentage points" not in sentence
        assert "half of which widens each side of the band" in sentence

    def test_swapping_the_data_swaps_the_sentence(self):
        """The claim is derived, not typed into `SECTIONS`: a table where the
        divergence runs the other way must produce the opposite superlative."""
        flipped = _llm_table([
            {"series": "Trade", "period": "The founding", "hw": 0.001},
            {"series": "Trade", "period": "Expansion", "hw": 0.090},
        ])
        sentence = site.llm_divergence_sentence(flipped)
        assert "diverge most in Expansion" in sentence
        assert "closest on The founding" in sentence

    def test_corex_rows_are_excluded_from_the_aggregation(self):
        """Surface A has no annotator and no half-width; letting its rows in
        would make the sentence describe a mixture of two labeling systems."""
        table = _llm_table([
            {"series": "Trade", "period": "The founding", "hw": 0.05},
            {"series": "Trade", "period": "Expansion", "hw": 0.01},
        ])
        contaminated = pd.concat([table, _llm_table([
            {"surface": B.COREX_SURFACE, "series": "war", "period": "1785",
             "hw": 0.99},
        ])], ignore_index=True)
        assert site.llm_divergence_sentence(table) == \
               site.llm_divergence_sentence(contaminated)

    def test_era_labels_are_html_escaped(self):
        """The sentence is interpolated straight into `index.html`.
        `escape-display-names-in-rendered-html` is the backlogged case where
        this was NOT done; it must not be repeated on a new surface."""
        table = _llm_table([
            {"series": "Trade", "period": "<b>Boom</b> & Bust", "hw": 0.05},
            {"series": "Trade", "period": "Quiet", "hw": 0.01},
        ])
        sentence = site.llm_divergence_sentence(table)
        assert "<b>" not in sentence
        assert "&lt;b&gt;Boom&lt;/b&gt; &amp; Bust" in sentence

    def test_a_table_with_no_measured_half_widths_says_nothing(self):
        """Better an omitted sentence than a superlative over an empty set."""
        table = _llm_table([
            {"series": "Trade", "period": "The founding", "hw": np.nan},
        ])
        assert site.llm_divergence_sentence(table) == ""

    def test_the_aggregation_is_named_in_the_sentence(self):
        """A superlative whose aggregation is unstated is not checkable: mean,
        max and share-weighted rankings need not agree."""
        table = _llm_table([
            {"series": "Trade", "period": "The founding", "hw": 0.05},
            {"series": "Trade", "period": "Expansion", "hw": 0.01},
        ])
        sentence = site.llm_divergence_sentence(table)
        assert "Averaging" in sentence
        assert "percentage points of paragraph share per topic" in sentence


# --------------------------------------------------------------------------- #
# 6. figure assembly and prose threading
# --------------------------------------------------------------------------- #
class TestLlmTopicFigure:
    def test_panels_are_the_largest_topics_by_corpus_wide_share(self):
        """The selection is a property of the data, not a curated list — so it
        must actually rank by the quantity it claims to (CLAUDE.md's "a
        'largest N' table must be sorted by the quantity it ranks")."""
        rows = []
        for topic, share in [("Big", 0.4), ("Middle", 0.2), ("Small", 0.01)]:
            for era in ("The founding", "Expansion"):
                rows.append({"series": topic, "period": era, "point": share,
                             "hw": 0.01, "n_paragraphs": 100})
        table = _llm_table(rows)
        for col, val in [("x", 1800.0), ("lo", 0.1), ("hi", 0.3),
                         ("lo_sampling", 0.1), ("hi_sampling", 0.3),
                         ("n_speeches", 10), ("ci_status", "ok"),
                         ("ci_components", "sampling+annotator_disagreement")]:
            table[col] = val
        table["period_order"] = table["period"].map(
            {"The founding": 0, "Expansion": 1})
        fig = site.fig_llm_topic_bands(table, top_n=2, rows=1, cols=2)
        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["Big", "Middle"]

    def test_a_weighted_topic_outranks_a_topic_that_only_spiked_once(self):
        """Corpus-wide share is paragraph-weighted, so a topic that was large in
        one tiny era does not outrank one that was moderate throughout."""
        table = _llm_table([
            {"series": "Steady", "period": "The founding", "point": 0.30,
             "n_paragraphs": 1000},
            {"series": "Steady", "period": "Expansion", "point": 0.30,
             "n_paragraphs": 1000},
            {"series": "Spike", "period": "The founding", "point": 0.95,
             "n_paragraphs": 10},
            {"series": "Spike", "period": "Expansion", "point": 0.0,
             "n_paragraphs": 1000},
        ])
        for col, val in [("x", 1800.0), ("lo", 0.1), ("hi", 0.3),
                         ("lo_sampling", 0.1), ("hi_sampling", 0.3),
                         ("n_speeches", 10), ("ci_status", "ok"),
                         ("ci_components", "sampling+annotator_disagreement")]:
            table[col] = val
        table["period_order"] = table["period"].map(
            {"The founding": 0, "Expansion": 1})
        fig = site.fig_llm_topic_bands(table, top_n=1, rows=1, cols=1)
        assert [a.text for a in fig.layout.annotations] == ["Steady"]


class TestIssuesDecadeFallback:
    """`fig_issues_decade(band_table=None)` — the unbanded fallback branch.

    Live and reachable today: `site.main` passes whatever `bands.load_bands()`
    returned, and that is `None` on any checkout where `data/bands.parquet` has
    not been built. The branch preserves the OLD behaviour — the `>= 40` hard
    mask that drops thin periods outright — so the two branches disagree about
    1785 by design, and that disagreement is the whole feature. Testing one
    without the other would leave the claim "the mask became a rendering
    distinction" unasserted at the figure level.

    The fixture turned out to be small: the rewritten function reads only the
    label frame, the issue names and the band table (the `issue_df`/`scores`
    parameters it inherited went unread and have since been dropped), and
    `topic_quality.surfaced_discovered()` contributes exactly one extra column,
    so a synthetic frame with three boolean columns drives both branches through
    the real `display_issues` / `DISCOVERED_LABELS` plumbing.
    """

    @pytest.fixture
    def labels(self) -> pd.DataFrame:
        from presidential_profiles import topic_quality

        discovered = topic_quality.surfaced_discovered()
        assert discovered, "fixture assumes at least one surfaced discovered topic"
        cols = ["war", "trade", *discovered]
        # Every issue gets its own share AND every period gets its own share, so
        # the y-value assertions below cannot pass by comparing a number to
        # itself, and a panel that got its NEIGHBOUR's series is visible. The
        # module docstring's fixture convention, applied across two axes at once:
        # a flat `i % 2` here would make all three panels and both periods
        # identical and leave the label -> series pairing unasserted.
        col_offset = {c: j for j, c in enumerate(cols)}
        rows = []
        specs = [
            # 1785: 2 speeches, 14 paragraphs — under MIN_PERIOD_PARAGRAPHS
            ("thin-a", 1785, "Washington", 7, 3),
            ("thin-b", 1786, "Washington", 7, 3),
        ] + [
            # 1900 gets 5 hits per 20-paragraph speech, 1905 gets 15.
            (f"d{k}", 1900 + (k % 2) * 5, "McKinley", 20, 5 + (k % 2) * 10)
            for k in range(6)
        ]
        for doc, year, president, n, hits in specs:
            for i in range(n):
                rows.append({
                    "doc_name": doc, "para_idx": i, "year": year,
                    "president": president,
                    **{c: bool(i < hits + col_offset[c]) for c in cols},
                })
        out = pd.DataFrame(rows)
        for c in cols:
            out[c] = out[c].astype(bool)
        return out

    def _fallback_lines(self, fig) -> list:
        return [t for t in fig.data if t.mode == "lines" and t.fill == "tozeroy"]

    def test_the_fallback_still_applies_the_old_forty_paragraph_hard_mask(self, labels):
        """The mask, and the SHARES the surviving periods carry.

        The y-values matter more here than the trace count. A real checkout has
        `data/bands.parquet`, so `site.main` always takes the banded path and
        the `band_table is None` branch never executes during a site rebuild —
        it lies outside both of this stage's runtime proofs (the parquet digest
        and the byte-identical `docs/` rebuild). These assertions are the only
        thing covering it, so they pin what `share.loc[share.index.isin(valid)]`
        actually selects, not merely which x's survive.

        Expected: each period pools three 20-paragraph speeches (60 paragraphs),
        1900 at 5 hits per speech and 1905 at 15, plus one extra hit per speech
        for each successive issue column.
        """
        fig = site.fig_issues_decade(labels, ["war", "trade"], band_table=None)
        lines = self._fallback_lines(fig)
        assert len(lines) == 3
        expected = [(25.0, 75.0), (30.0, 80.0), (35.0, 85.0)]
        for line, (at_1900, at_1905) in zip(lines, expected):
            assert 1785 not in list(line.x), "the unbanded branch must still drop it"
            # Absolute, and in order: the panel's series must line up with the
            # panel's title, which a set-membership check cannot see.
            assert list(line.x) == [1900, 1905]
            assert list(line.y) == pytest.approx([at_1900, at_1905])

    def test_the_banded_branch_draws_the_period_the_fallback_drops(self, labels):
        """The feature, at the figure level: same frame, same function, and the
        thin period appears only once the band table exists."""
        band_table = B.corex_bands(labels)
        fig = site.fig_issues_decade(labels, ["war", "trade"],
                                     band_table=band_table)
        dotted = [t for t in fig.data
                  if t.mode == "lines" and t.line.dash == "dot"]
        assert len(dotted) == 3
        for line in dotted:
            assert 1785 in list(line.x)
        assert not self._fallback_lines(fig), "the fallback trace must not be drawn"

    def test_both_branches_title_their_panels_with_the_display_labels(self, labels):
        """A discovered topic is titled by its hand-edited display name in both
        branches — the plumbing most likely to be dropped when a branch is
        rewritten."""
        from presidential_profiles import profiles_site, topic_quality

        expected = [profiles_site.DISCOVERED_LABELS.get(n, n) for n in
                    topic_quality.display_issues(["war", "trade"])]
        assert expected[:2] == ["war", "trade"]
        assert expected[2] != topic_quality.surfaced_discovered()[0], (
            "fixture assumes the discovered topic has a display name to map to"
        )
        for band_table in (None, B.corex_bands(labels)):
            fig = site.fig_issues_decade(labels, ["war", "trade"],
                                         band_table=band_table)
            titles = [a.text for a in fig.layout.annotations]
            assert titles == expected

    def test_both_branches_overlay_per_president_per_period_dots(self, labels):
        """The old fallback plotted one dot per president at their term
        midpoint; both branches now use `president_period_dots`, so a dot must
        land on a period boundary rather than between two of them."""
        for band_table in (None, B.corex_bands(labels)):
            fig = site.fig_issues_decade(labels, ["war", "trade"],
                                         band_table=band_table)
            markers = [t for t in fig.data if t.mode == "markers"]
            assert len(markers) == 3, "one dot overlay per panel"
            for trace in markers:
                # Counted, not merely present: an overlay of ZERO dots is still
                # a marker trace, so `assert markers` alone would pass with the
                # dots silently emptied.
                assert len(trace.x) == 2, "McKinley's two periods, both dotted"
                assert set(trace.x) == {1900, 1905}
                assert set(np.ravel(trace.customdata)) == {"McKinley"}, (
                    "Washington's 14-paragraph 1785 cell is under "
                    "MIN_DOT_PARAGRAPHS and must be dropped, not plotted"
                )


class TestProseExtraThreading:
    STATS = {"speeches": 1057, "words": 12_000_000, "presidents": 45,
             "start": 1789, "end": 2026}

    def _figs(self, *keys) -> dict:
        return {k: go.Figure(layout=dict(height=400)) for k in keys}

    @pytest.mark.parametrize("inline", [False, True])
    def test_the_derived_sentence_reaches_the_llm_topics_section(self, inline):
        """Both `build_html` call sites pass `prose_extra`, including the
        `--inline` self-contained variant — the parameter most likely to be
        silently dropped, since it is written second and rarely rebuilt."""
        html = site.build_html(
            self._figs("llm_topics", "issues"), self.STATS, {}, inline=inline,
            prose_extra={"llm_topics": "DERIVED SENTENCE."},
        )
        llm_section = html.split('<section id="llm_topics">')[1].split("</section>")[0]
        assert "DERIVED SENTENCE." in llm_section

    def test_the_sentence_does_not_leak_into_any_other_section(self):
        html = site.build_html(
            self._figs("llm_topics", "issues", "keywords"), self.STATS, {},
            inline=False, prose_extra={"llm_topics": "DERIVED SENTENCE."},
        )
        assert html.count("DERIVED SENTENCE.") == 1
        others = html.split('<section id="llm_topics">')[0] + \
            html.split('<section id="llm_topics">')[1].split("</section>", 1)[1]
        assert "DERIVED SENTENCE." not in others

    def test_omitting_prose_extra_entirely_still_builds(self):
        """The `bands.parquet`-absent path: no derived sentence, and the
        section's own prose survives intact."""
        html = site.build_html(self._figs("issues"), self.STATS, {}, inline=False)
        assert '<section id="issues">' in html
        assert "36,000 paragraph-sized chunks" in html

    def test_a_section_with_no_figure_is_omitted_rather_than_left_empty(self):
        """Prose describing a chart that is not there is worse than no section.
        `llm_topics` is the live case — it needs `data/bands.parquet`."""
        html = site.build_html(self._figs("issues"), self.STATS, {}, inline=False)
        assert '<section id="llm_topics">' not in html
        assert "the AI's doubt is part of the answer" not in html

    def test_an_extra_for_an_omitted_section_does_not_resurrect_it(self):
        html = site.build_html(
            self._figs("issues"), self.STATS, {}, inline=False,
            prose_extra={"llm_topics": "DERIVED SENTENCE."},
        )
        assert "DERIVED SENTENCE." not in html


# --------------------------------------------------------------------------- #
# 8. defects found in review — one assertion each would have caught them
# --------------------------------------------------------------------------- #
class TestPerSeriesBandMiss:
    """`_banded_small_multiples` must survive a series with no band row.

    `bands.series_band` returns None for a series absent from the table — the
    seam's PARTIAL failure mode, which a stale `bands.parquet` predating a CorEx
    refit produces. Whole-file absence already degrades gracefully and schema
    drift already raises a rebuild message; this case used to die as an opaque
    `TypeError: 'NoneType' object is not subscriptable` inside `_band_hover`,
    several frames below the thing that was actually wrong.
    """

    def test_a_none_band_does_not_crash_the_grid(self):
        fig = site._banded_small_multiples(
            [("X", None)], rows=1, cols=1, height=200, unit="%")
        assert fig.data == ()

    def test_an_empty_band_does_not_crash_the_grid(self):
        fig = site._banded_small_multiples(
            [("X", _band([]))], rows=1, cols=1, height=200, unit="%")
        assert fig.data == ()

    def test_a_missing_band_still_renders_that_panels_dots(self):
        """The panel loses its line, not its president attribution — a stale
        band table must not silently delete data that does not depend on it."""
        dots = {"X": pd.DataFrame({"x": [1900], "y": [4.0], "name": ["Taft"]})}
        fig = site._banded_small_multiples(
            [("X", None)], rows=1, cols=1, height=200, unit="%", dots=dots)
        assert [t.mode for t in fig.data] == ["markers"]
        assert list(fig.data[0].x) == [1900]

    def test_a_present_band_is_unaffected_by_the_guard(self):
        """Fail-safe check: the skip must trigger on absence only. If it also
        fired on a real band the grid would silently render empty panels."""
        fig = site._banded_small_multiples(
            [("X", _band([{"x": 1900, "lo": 0.1, "hi": 0.2},
                          {"x": 1905, "lo": 0.1, "hi": 0.2}]))],
            rows=1, cols=1, height=200, unit="%")
        assert len(fig.data) > 1


class TestXRangeCoversEveryPlottedPoint:
    """The x domain is a floor to widen, never a window to crop to.

    `X_RANGE` starts at 1786 and Surface A plots the recovered 1785 bucket, so a
    fixed lower bound clipped the one period this whole change exists to stop
    hiding — band, marker and hover all outside the axis, on the dashboard only.
    """

    def test_the_domain_extends_below_the_earliest_plotted_x(self):
        panels = [("X", _band([{"x": 1785, "lo": 0.0, "hi": 0.6},
                               {"x": 1790, "lo": 0.1, "hi": 0.2}]))]
        assert site._x_range_covering(panels)[0] == 1785

    def test_a_late_starting_series_does_not_shrink_the_domain(self):
        """Widening only: a chart whose data starts in 1900 keeps the historical
        window, so panels in one grid still share an axis."""
        panels = [("X", _band([{"x": 1900, "lo": 0.1, "hi": 0.2}]))]
        assert site._x_range_covering(panels) == site.X_RANGE

    def test_missing_and_empty_panels_are_ignored(self):
        assert site._x_range_covering([("X", None)]) == site.X_RANGE
        assert site._x_range_covering([("X", _band([]))]) == site.X_RANGE

    def test_the_real_figure_plots_no_point_outside_its_own_axis(self):
        """The end-to-end form of the defect, against the shipped parquet: every
        x the figure draws must fall inside every panel's declared range."""
        table = B.load_bands()
        if table is None:
            pytest.skip("data/bands.parquet not built")
        corex = table[table["surface"] == B.COREX_SURFACE]
        panels = [(s, B.series_band(table, B.COREX_SURFACE, s))
                  for s in corex["series"].unique()[:3]]
        lo, hi = site._x_range_covering(panels)
        assert lo <= corex["x"].min()
        assert hi >= corex["x"].max()


class TestDotsUnitDescribesWhatIsPlotted:
    """The dots became per (president, PERIOD); the hover must say so.

    The issue pages were updated when the dots changed and the dashboard was
    not, so `Lincoln: 12.3% of their paragraphs` read as a whole-presidency
    share — a different quantity from the one plotted. Both call sites now share
    one constant, which is what stops the two renderings drifting again.
    """

    def test_the_shared_constant_names_the_period(self):
        assert "in this period" in site.DOTS_UNIT

    def test_both_grid_branches_use_it(self):
        dots = {"X": pd.DataFrame({"x": [1900], "y": [4.0], "name": ["Taft"]})}
        banded = site._banded_small_multiples(
            [("X", _band([{"x": 1900, "lo": 0.1, "hi": 0.2},
                          {"x": 1905, "lo": 0.1, "hi": 0.2}]))],
            rows=1, cols=1, height=200, unit="%", dots=dots,
            dots_unit=site.DOTS_UNIT)
        plain = site._small_multiples(
            [("X", pd.Series([4.0], index=[1900]))], rows=1, cols=1, height=200,
            hovertemplate="%{y:.1f}%", dots=dots, dots_unit=site.DOTS_UNIT)
        for fig in (banded, plain):
            markers = [t for t in fig.data if t.mode == "markers"]
            assert markers, "fixture must produce a dot trace"
            assert "in this period" in markers[0].hovertemplate

    def test_the_issue_page_hover_agrees_with_the_dashboard(self):
        """Same function, two surfaces: they must describe one quantity."""
        pl = pd.DataFrame({
            "doc_name": ["a"] * 30, "para_idx": range(30), "year": [1900] * 30,
            "president": ["Taft"] * 30, "war": [True] * 15 + [False] * 15,
        })
        fig = S.fig_issue_timeline(pl, "war", "War",
                                   S.president_period_dots(pl, "war"))
        markers = [t for t in fig.data if t.mode == "markers"]
        assert "in this period" in markers[0].hovertemplate


class TestWrapPanelTitle:
    """`_wrap_panel_title` — the only new function that shipped untested.

    A 50-character topic name was clipped mid-word by plotly ("… & Media
    Attac"), which reads as a rendering bug rather than a shortened label.
    """

    def test_a_short_title_is_returned_unchanged(self):
        assert site._wrap_panel_title("Cold War", 34) == "Cold War"

    def test_a_long_title_wraps_at_a_word_boundary(self):
        out = site._wrap_panel_title(
            "Partisan Combat, Press Conferences & Media Attacks", 34)
        assert out == "Partisan Combat, Press Conferences<br>& Media Attacks"
        assert all(len(line) <= 34 for line in out.split("<br>"))

    def test_no_content_is_lost_while_it_still_fits(self):
        title = "Treaties, Diplomacy & International Arbitration"
        assert site._wrap_panel_title(title, 34).replace("<br>", " ") == title

    def test_overflow_past_the_last_line_truncates_with_an_ellipsis(self):
        out = site._wrap_panel_title("aa bb cc dd ee ff gg hh", 5, max_lines=2)
        assert out.count("<br>") == 1
        assert out.endswith("…")
        assert all(len(line) <= 5 for line in out.split("<br>"))

    def test_a_single_unbreakable_word_is_never_dropped(self):
        assert site._wrap_panel_title("Supercalifragilistic", 5).startswith("Super")

    def test_the_hover_keeps_the_untruncated_name(self):
        """Only the DISPLAYED title is shortened; the tooltip must not be."""
        long_title = "Partisan Combat, Press Conferences & Media Attacks"
        fig = site._banded_small_multiples(
            [(long_title, _band([{"x": 1900, "lo": 0.1, "hi": 0.2},
                                 {"x": 1905, "lo": 0.1, "hi": 0.2}]))],
            rows=1, cols=1, height=200, unit="%")
        assert any(long_title in (t.hovertemplate or "") for t in fig.data)
        assert "<br>" in fig.layout.annotations[0].text


class TestCoverageSentence:
    """B-1: the published prose claimed the whole corpus was double-annotated."""

    @pytest.fixture
    def coverage(self, monkeypatch):
        """The governing number must be checked against the REAL artifacts.

        `conftest.redirect_annotation_dirs` is autouse and points the annotation
        directory at tmp_path (a write-side guard); this test is read-only and
        the whole claim is about the shipped tables, so it points it back. A
        synthetic fixture here would assert only that the f-string interpolates.
        """
        from presidential_profiles import llm_annotations as ann
        from presidential_profiles.corpus import DATA_DIR

        real = DATA_DIR / "llm_annotations"
        if not (real / f"{B.SECONDARY_ANNOTATIONS}.parquet").exists():
            pytest.skip("secondary annotation table not present")
        monkeypatch.setattr(ann, "ANNOTATIONS_DIR", real)
        return B.paired_coverage()

    def test_it_states_the_sample_not_the_corpus(self, coverage):
        c = coverage
        sentence = site.llm_coverage_sentence()
        assert f"{c['n_paired_paragraphs']:,}" in sentence
        assert f"{c['n_corpus_paragraphs']:,}" in sentence
        assert f"{c['n_paired_speeches']}" in sentence
        # The governing fact: it is a sample, and a minority of the corpus.
        assert c["paragraph_fraction"] < 0.5
        assert f"{c['paragraph_fraction']:.1%}" in sentence

    def test_it_names_the_transfer_assumption(self, coverage):
        assert "assumed" in site.llm_coverage_sentence()

    def test_the_rendered_section_never_claims_every_paragraph_was_reread(self):
        """The defect verbatim: 'every paragraph ... then read again'. The
        subject of 'read again' must not be the whole corpus."""
        prose = next(p for k, _, _, p in site.SECTIONS if k == "llm_topics")
        # The defect verbatim: "every paragraph" was the subject of "read
        # again". Pinning the exact join is what makes this fail if the two
        # clauses are ever spliced back together.
        assert "corpus itself, then read again" not in prose
        assert "sample" in prose
        # ... and the corrected subject must actually be there.
        assert "sample</em> of those speeches" in prose


class TestMetaCoverageFiguresAreNotDrifting:
    """`bands_meta.json` states the sample's size as text, not as a computed
    field — `write_bands` is deliberately a pure table-to-parquet write and does
    not reach for the annotation parquets. That trade is only safe if the
    literals are pinned against a re-derivation, which is this.
    """

    @pytest.fixture
    def coverage(self, monkeypatch):
        from presidential_profiles import llm_annotations as ann
        from presidential_profiles.corpus import DATA_DIR

        real = DATA_DIR / "llm_annotations"
        if not (real / f"{B.SECONDARY_ANNOTATIONS}.parquet").exists():
            pytest.skip("secondary annotation table not present")
        monkeypatch.setattr(ann, "ANNOTATIONS_DIR", real)
        return B.paired_coverage()

    @pytest.fixture
    def caveat(self):
        import json
        from presidential_profiles.corpus import DATA_DIR

        path = DATA_DIR / "bands_meta.json"
        if not path.exists():
            pytest.skip("data/bands_meta.json not built")
        meta = json.loads(path.read_text())
        return meta["surfaces"][B.LLM_SURFACE]["annotator_disagreement"][
            "caveats"]["transfer_assumption"]

    def test_every_count_in_the_transfer_caveat_re_derives_IN_CONTEXT(
        self, caveat, coverage
    ):
        """Containment alone is too weak for a pin that stands in for a
        computed field: "266 speeches holding 262 of the corpus's 36229
        paragraphs" contains every right number attached to the wrong noun and
        would pass. Each number is matched against the words around it."""
        import re

        for pattern in (
            rf"{coverage['n_paired_speeches']}\s+speeches",
            rf"of the {coverage['n_sampled_speeches']} agreement_sample",
            rf"holding {coverage['n_paired_paragraphs']} of the corpus's "
            rf"{coverage['n_corpus_paragraphs']}\s+paragraphs",
        ):
            assert re.search(pattern, caveat), f"no in-context match for {pattern!r}"

    def test_the_numbers_are_not_merely_present_somewhere(self, caveat, coverage):
        """Fail-safe on the matcher itself: transposing two values must break
        the assertions above. If it does not, they are not doing their job."""
        import re

        swapped = caveat.replace(
            str(coverage["n_paired_speeches"]), "<X>"
        ).replace(str(coverage["n_sampled_speeches"]),
                  str(coverage["n_paired_speeches"])).replace(
            "<X>", str(coverage["n_sampled_speeches"]))
        assert not re.search(
            rf"{coverage['n_paired_speeches']}\s+speeches", swapped)

    def test_the_stated_fraction_re_derives(self, caveat, coverage):
        assert f"{coverage['paragraph_fraction']:.1%}" in caveat

    def test_the_caveat_names_the_recomputation_entry_point(self, caveat):
        """So a reader who doubts the literals knows where to check them."""
        assert "paired_coverage()" in caveat


class TestUnpinnedMetaProseIsNowPinned:
    """MJ-1's actual lesson. `transfer_assumption` had a re-derivation pin and
    stayed correct; `differs_from_combat.reason` and `de_emphasis_is_two_sided`
    were new prose with no pin, and the one that drifted was the unpinned one —
    it shipped "the corpus median is ~30 speeches", against a true median of 20,
    in the sentence written to justify why this module's trust gate diverges
    from combat's. A floor of 20 sitting BELOW a median of 30 would mark well
    under half the record provisional, so the wrong premise contradicted the
    conclusion it was offered to support.
    """

    @pytest.fixture
    def meta(self):
        import json
        from presidential_profiles.corpus import DATA_DIR

        path = DATA_DIR / "bands_meta.json"
        if not path.exists():
            pytest.skip("data/bands_meta.json not built")
        return json.loads(path.read_text())["ci_status"]

    @pytest.fixture
    def periods(self):
        table = B.load_bands()
        if table is None:
            pytest.skip("data/bands.parquet not built")
        corex = table[table["surface"] == B.COREX_SURFACE]
        return corex.drop_duplicates("period")[["period", "n_speeches"]]

    def test_the_combat_thresholds_it_quotes_are_the_real_ones(self, meta):
        from presidential_profiles import combat

        d = meta["differs_from_combat"]
        assert d["combat_min_clusters_for_ci"] == combat.MIN_CLUSTERS_FOR_CI
        assert d["combat_low_cluster_caution"] == combat.LOW_CLUSTER_CAUTION

    def test_the_quoted_median_is_the_real_median(self, meta, periods):
        median = int(periods["n_speeches"].median())
        assert f"median is {median} speeches" in meta["differs_from_combat"]["reason"]

    def test_the_quoted_share_of_provisional_periods_re_derives(self, meta, periods):
        from presidential_profiles import combat

        below = int((periods["n_speeches"] < combat.LOW_CLUSTER_CAUTION).sum())
        reason = meta["differs_from_combat"]["reason"]
        assert f"mark {below} of the {len(periods)} periods" in reason

    def test_the_rationale_is_not_self_defeating(self, meta, periods):
        """The premise must actually support the conclusion: combat's floor has
        to sit at or above the median for "roughly half the record" to follow."""
        from presidential_profiles import combat

        assert combat.LOW_CLUSTER_CAUTION >= periods["n_speeches"].median()

    def test_the_two_sided_claim_names_both_periods(self, meta, periods):
        """Recovery and de-emphasis are different sets; the claim states both."""
        claim = meta["de_emphasis_is_two_sided"]
        recovered = periods[periods["n_speeches"] < B.MIN_CLUSTERS_FOR_CI + 1]
        demoted = periods[(periods["n_speeches"] >= B.MIN_CLUSTERS_FOR_CI + 1)
                          & (periods["n_speeches"] < B.LOW_CLUSTER_CAUTION)]
        assert len(recovered) == 1 and len(demoted) == 1
        assert recovered.iloc[0]["period"] in claim
        assert demoted.iloc[0]["period"] in claim
        assert str(int(demoted.iloc[0]["n_speeches"])) in claim


class TestSeriesBandIsPeriodComplete:
    """MN-5: `band_traces` splits on consecutive ROWS and calls them periods.

    That equivalence rests on `series_band` returning a period-complete frame in
    `period_order`, and the docstring claimed the invariant was test-asserted
    when the only nearby assertion was one series' row count against a magic
    literal. This is the assertion the sentence was describing — over EVERY
    series on both surfaces, because a single series proves nothing about the
    table, and asserting consecutiveness rather than a count.
    """

    @pytest.fixture
    def table(self):
        t = B.load_bands()
        if t is None:
            pytest.skip("data/bands.parquet not built")
        return t

    @pytest.mark.parametrize("surface", [B.COREX_SURFACE, B.LLM_SURFACE])
    def test_every_series_carries_every_period_of_its_surface(self, table, surface):
        expected = sorted(table[table["surface"] == surface]["period_order"].unique())
        for series in table[table["surface"] == surface]["series"].unique():
            got = list(B.series_band(table, surface, series)["period_order"])
            assert got == expected, f"{surface}/{series} is period-sparse"

    def test_row_adjacency_therefore_means_period_adjacency(self, table):
        """The invariant `band_traces` actually relies on, stated directly: no
        two consecutive rows may skip a period of the surface."""
        for surface in (B.COREX_SURFACE, B.LLM_SURFACE):
            order = sorted(table[table["surface"] == surface]["period_order"].unique())
            step = {(a, b) for a, b in zip(order, order[1:])}
            for series in table[table["surface"] == surface]["series"].unique():
                got = list(B.series_band(table, surface, series)["period_order"])
                assert all((a, b) in step for a, b in zip(got, got[1:]))
