"""The estimators in `register.py`: entropy / effective topics, the ratio
measures, and the trend statistics computed over an era series.

This module's output goes into a published research report, so the failure mode
that matters is a silently wrong number, not a crash. Every value asserted here
is hand-computable from the inputs above it — a uniform distribution over k
topics is exactly k effective topics, a Miller-Madow correction is exactly
`exp((K_observed - 1) / 2N)` times the plug-in figure, a Spearman correlation
against a monotone series is exactly 1.

Nothing here reads the corpus or fits a model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import register as R


# =========================================================================== #
class TestEffectiveTopics:
    def test_uniform_attention_over_k_topics_is_exactly_k(self):
        """The definition of the measure: "if attention were spread evenly over N
        topics, what is N?" Under an exactly even spread the plug-in answer must
        be the topic count itself, on the nose."""
        counts = np.array([10.0, 10.0, 10.0, 10.0])

        assert R.effective_topics(counts, miller_madow=False) == pytest.approx(4.0)

    def test_one_topic_attention_is_exactly_one(self):
        """The degenerate end. Entropy is 0 and the Miller-Madow term is
        `(1 - 1) / 2N == 0`, so BOTH estimators must return exactly 1 — a
        correction that fired here would push a single-topic era above 1."""
        counts = np.array([10.0, 0.0, 0.0, 0.0])

        assert R.effective_topics(counts, miller_madow=False) == pytest.approx(1.0)
        assert R.effective_topics(counts, miller_madow=True) == pytest.approx(1.0)

    def test_unused_topic_columns_do_not_dilute_the_estimate(self):
        """Zero-count columns carry no probability mass, so a 50-column taxonomy
        with 4 topics used scores the same as a 4-column one. This is why the
        level-2 and legacy matrices can be compared as trends at all."""
        four = np.array([10.0, 10.0, 10.0, 10.0])
        padded = np.concatenate([four, np.zeros(46)])

        assert R.effective_topics(padded, miller_madow=False) == pytest.approx(
            R.effective_topics(four, miller_madow=False)
        )

    def test_plugin_matches_the_hand_computed_shannon_perplexity(self):
        """A non-uniform case with a closed form: p = (1/2, 1/4, 1/4) gives
        H = 1.5 * ln 2, so perplexity is 2 ** 1.5. Pins the log base — a
        base-2 entropy fed to `exp` would return 2.83 -> 4.76 here."""
        counts = np.array([2.0, 1.0, 1.0])

        assert R.effective_topics(counts, miller_madow=False) == pytest.approx(
            2.0 ** 1.5
        )

    def test_miller_madow_raises_the_estimate_by_exactly_the_correction_factor(self):
        """The correction is `+ (K_observed - 1) / (2N)` on the entropy, which is
        a multiplicative `exp(...)` on the perplexity. Asserting the exact factor
        pins the numerator (K - 1, not K), the denominator (2N, not N) and the
        sign all at once."""
        counts = np.array([3.0, 2.0, 1.0])   # K_observed = 3, N = 6
        plugin = R.effective_topics(counts, miller_madow=False)

        corrected = R.effective_topics(counts, miller_madow=True)

        assert corrected == pytest.approx(plugin * math.exp((3 - 1) / (2 * 6)))

    def test_correction_is_upward_and_larger_where_the_sample_is_thinner(self):
        """Plug-in entropy is biased DOWNWARD at small N, and the thin early eras
        are exactly where a downward bias would manufacture the rising-breadth
        finding this module is testing. So the correction must push up, and push
        up harder on the smaller sample."""
        thin = np.array([3.0, 2.0, 1.0])
        thick = thin * 100

        thin_gap = R.effective_topics(thin) - R.effective_topics(thin, False)
        thick_gap = R.effective_topics(thick) - R.effective_topics(thick, False)

        assert thin_gap > 0
        assert thick_gap > 0
        assert thin_gap > thick_gap

    def test_correction_counts_observed_topics_not_taxonomy_size(self):
        """`K_observed` is the number of topics that actually appeared. Padding
        the same distribution with unused columns must not change the correction
        — using the column count instead would inflate every early era, which is
        the exact artifact the module exists to rule out."""
        counts = np.array([3.0, 2.0, 1.0])
        padded = np.concatenate([counts, np.zeros(47)])

        assert R.effective_topics(padded) == pytest.approx(R.effective_topics(counts))

    def test_no_observations_is_nan_not_zero(self):
        """An era/genre cell with no label assignments has an UNDEFINED entropy.
        Returning 0 would read as "one topic" and drag a group mean down."""
        assert np.isnan(R.effective_topics(np.zeros(5)))

    def test_broadcasts_over_leading_axes_for_bootstrap_replicates(self):
        """The bootstrap feeds a (n_replicates, n_topics) matrix straight in, so
        the last axis must be the topic axis and every leading axis must be
        preserved — with the same answers as row-by-row evaluation."""
        matrix = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 0.0, 0.0, 0.0], np.zeros(4)])

        out = R.effective_topics(matrix, miller_madow=False)

        assert out.shape == (3,)
        assert out[0] == pytest.approx(4.0)
        assert out[1] == pytest.approx(1.0)
        assert np.isnan(out[2])

    def test_scaling_all_counts_leaves_the_plugin_estimate_unchanged(self):
        """The plug-in figure is a function of the SHARES only, so doubling the
        corpus cannot move it. (The Miller-Madow figure legitimately does move —
        that is the sample-size correction.)"""
        counts = np.array([5.0, 3.0, 2.0])

        assert R.effective_topics(counts * 7, miller_madow=False) == pytest.approx(
            R.effective_topics(counts, miller_madow=False)
        )


# =========================================================================== #
class TestRatio:
    def test_divides_and_applies_the_scale(self):
        assert R._ratio(np.array([3.0]), np.array([1_000.0]), 1_000.0)[0] == 3.0

    def test_zero_denominator_is_nan_not_zero_or_inf(self):
        """A measure with an empty denominator is undefined. Zero would be
        published as a real observation; inf would poison every downstream mean."""
        out = R._ratio(np.array([5.0, 5.0]), np.array([0.0, 2.0]))

        assert np.isnan(out[0])
        assert out[1] == 2.5

    def test_zero_numerator_over_positive_denominator_is_zero(self):
        """The opposite case is a genuine observation and must stay 0.0 — an era
        that really did produce no non-policy paragraphs."""
        assert R._ratio(np.array([0.0]), np.array([10.0]))[0] == 0.0


# =========================================================================== #
def _sums(**over):
    """One row of additive per-group sums, with every column `measures_from_sums`
    reads present and hand-chosen so each measure has a round answer."""
    row = {
        "n_paragraphs": 10.0,
        "para_words": 1_000.0,
        "llm_labels": 20.0,
        "legacy_labels": 15.0,
        "llm_non_policy_paras": 2.0,
        "llm_zero_paras": 1.0,
        "legacy_zero_paras": 3.0,
        "pv_proposal": 4.0,
        "pv_values": 2.0,
        "pv_mixed": 1.0,
        "pv_neither": 3.0,
        "n_tokens": 1_200.0,
        "n_sents": 60.0,
        "i_count": 3.0,
        "we_count": 7.0,
        "fk_x_tokens": 10_200.0,
        "n_words": 1_000.0,
        **{marker: 5.0 for marker in R.STYLE_MARKERS},
    }
    row.update(over)
    return pd.DataFrame([row])


def _topic_counts(n_rows=1):
    return {
        "legacy15": np.tile(np.array([10.0, 10.0, 10.0, 10.0]), (n_rows, 1)),
        "llm_level2": np.tile(np.array([8.0, 0.0, 0.0, 0.0]), (n_rows, 1)),
    }


class TestMeasuresFromSums:
    def test_every_ratio_measure_matches_its_hand_computed_value(self):
        """The published numbers. Each is a ratio of two columns of `_sums`, so a
        transposed numerator/denominator or a wrong scale shows up immediately."""
        out = R.measures_from_sums(_sums(), _topic_counts()).iloc[0]

        assert out[("labels_per_paragraph", "legacy15")] == pytest.approx(1.5)
        assert out[("labels_per_paragraph", "llm_level2")] == pytest.approx(2.0)
        assert out[("labels_per_1k_words", "legacy15")] == pytest.approx(15.0)
        assert out[("labels_per_1k_words", "llm_level2")] == pytest.approx(20.0)
        assert out[("non_policy_share", "legacy15")] == pytest.approx(0.3)
        assert out[("non_policy_share", "llm_level2")] == pytest.approx(0.2)
        assert out[("zero_topic_share", "llm_level2")] == pytest.approx(0.1)
        assert out[("mean_paragraph_words", "none")] == pytest.approx(100.0)

    def test_proposal_and_values_measures_use_their_documented_denominators(self):
        """`proposal_vs_values` is restricted to the unambiguous paragraphs
        (proposal + values), while the two share measures count `mixed` on BOTH
        sides and divide by all paragraphs. Conflating the three would make 0.5
        mean something different in each."""
        out = R.measures_from_sums(_sums(), _topic_counts()).iloc[0]

        assert out[("proposal_vs_values", "none")] == pytest.approx(4 / 6)
        assert out[("proposal_share", "none")] == pytest.approx(0.5)   # (4 + 1) / 10
        assert out[("values_share", "none")] == pytest.approx(0.3)     # (2 + 1) / 10

    def test_neither_share_is_the_plain_residual_not_one_minus_the_others(self):
        """`neither_share` is `pv_neither / n_paragraphs`, which is NOT
        `1 - proposal_share - values_share`: those two double-count `mixed`, so
        the naive complement would come out negative here (1 - 0.5 - 0.3 = 0.2
        against the true 0.3). The note's "neutral description collapsed" figure
        is read straight off this column, so the two must not be confused."""
        out = R.measures_from_sums(_sums(), _topic_counts()).iloc[0]

        assert out[("neither_share", "none")] == pytest.approx(0.3)     # 3 / 10
        assert out[("neither_share", "none")] != pytest.approx(
            1 - out[("proposal_share", "none")] - out[("values_share", "none")]
        )

    def test_the_four_stance_shares_reconstruct_the_whole_split(self):
        """The reason `neither_share` is published at all: with it, the mixed
        share — which has no column of its own — is recoverable, because
        `proposal_share + values_share + neither_share == 1 + mixed_share`. A
        reader can therefore check the full four-way stance decomposition against
        the parquet instead of taking the note's prose on trust."""
        out = R.measures_from_sums(_sums(), _topic_counts()).iloc[0]

        recovered = (
            out[("proposal_share", "none")]
            + out[("values_share", "none")]
            + out[("neither_share", "none")]
            - 1.0
        )

        assert recovered == pytest.approx(0.1)      # pv_mixed 1 / n_paragraphs 10

    def test_style_measures_match_their_hand_computed_values(self):
        out = R.measures_from_sums(_sums(), _topic_counts()).iloc[0]

        assert out[("fk_grade", "none")] == pytest.approx(8.5)          # 10200 / 1200
        assert out[("words_per_sentence", "none")] == pytest.approx(20.0)
        assert out[("self_reference", "none")] == pytest.approx(0.3)    # 3 / (3 + 7)
        for marker in R.STYLE_MARKERS:
            # 5 occurrences per 1,000 words == 50 per 10,000 words.
            assert out[(marker, "none")] == pytest.approx(50.0)

    def test_entropy_measures_are_computed_per_taxonomy(self):
        out = R.measures_from_sums(_sums(), _topic_counts()).iloc[0]

        assert out[("effective_topics_plugin", "legacy15")] == pytest.approx(4.0)
        assert out[("effective_topics_plugin", "llm_level2")] == pytest.approx(1.0)
        assert out[("effective_topics", "legacy15")] > out[
            ("effective_topics_plugin", "legacy15")
        ]

    def test_normalized_entropy_divides_by_log_of_the_column_count(self):
        """Normalization is by the TAXONOMY size (the matrix's column count), not
        by the number of topics OBSERVED — that is what puts a 15-category and a
        50-category taxonomy on one 0-1 scale.

        The two candidate denominators are only separable on a matrix where they
        differ, so the fixture spends its 40 assignments evenly over four of SIX
        columns: `K_total` is 6 while `K_observed` is 4. A fixture using all of
        its columns cannot tell them apart, and the difference runs in the
        direction that matters — the observed count is smaller in the thin early
        eras, so normalizing by it would inflate exactly the periods whose low
        values the rising-breadth headline rests on.

        The expected value is written out from the definition rather than read
        back off the frame, so the numerator (Miller-Madow corrected entropy) and
        the denominator (log of the column count) are each pinned on their own.
        """
        counts = {"legacy15": np.array([[10.0, 10.0, 10.0, 10.0, 0.0, 0.0]])}

        out = R.measures_from_sums(_sums(), counts).iloc[0]

        # H_MM = ln 4 + (K_observed - 1) / 2N = ln 4 + 3/80, over ln(K_total = 6).
        assert out[("normalized_entropy", "legacy15")] == pytest.approx(
            (math.log(4) + 3 / 80) / math.log(6)
        )
        # The same numerator over the OBSERVED-topic count is a different number.
        assert out[("normalized_entropy", "legacy15")] != pytest.approx(
            (math.log(4) + 3 / 80) / math.log(4)
        )

    def test_uniform_attention_over_every_column_is_the_top_of_the_scale(self):
        """The scale's upper anchor: attention spread evenly over ALL columns is
        a PLUG-IN normalized entropy of exactly 1.

        The published `normalized_entropy` sits fractionally ABOVE 1 here,
        because its numerator carries the Miller-Madow correction while its
        denominator does not. That is asserted rather than glossed: the measure's
        docstring calls this a "0-1 scale" and it is one only up to the
        correction term (see Discovered Issues). Pinning the real value means a
        future change to either end of the ratio has to be deliberate."""
        counts = {"legacy15": np.full((1, 4), 25.0)}      # N = 100, K observed = 4

        out = R.measures_from_sums(_sums(), counts).iloc[0]

        assert math.log(
            out[("effective_topics_plugin", "legacy15")]
        ) / math.log(4) == pytest.approx(1.0)
        assert out[("normalized_entropy", "legacy15")] == pytest.approx(
            (math.log(4) + 3 / 200) / math.log(4)
        )
        assert out[("normalized_entropy", "legacy15")] > 1.0

    def test_is_shape_agnostic_so_replicates_go_through_the_same_code(self):
        """The whole reason the panel stores sufficient statistics: one input row
        is a point estimate and n input rows are n bootstrap replicates, computed
        by the identical function. A row-count change must not change a value."""
        one = _sums()
        three = pd.concat([_sums(), _sums(n_paragraphs=20.0), _sums()], ignore_index=True)

        single = R.measures_from_sums(one, _topic_counts(1))
        many = R.measures_from_sums(three, _topic_counts(3))

        assert len(many) == 3
        assert many.iloc[0][("labels_per_paragraph", "legacy15")] == pytest.approx(
            single.iloc[0][("labels_per_paragraph", "legacy15")]
        )
        assert many.iloc[1][("labels_per_paragraph", "legacy15")] == pytest.approx(0.75)

    def test_empty_group_gives_nan_measures_not_zeros(self):
        """A cell with no speeches in it (a genre absent from an era, a resample
        that drew nothing) has undefined measures. Zeros here would be published
        as real observations and would drag every contrast toward them."""
        empty = _sums(**{col: 0.0 for col in _sums().columns})

        out = R.measures_from_sums(empty, {"legacy15": np.zeros((1, 4))}).iloc[0]

        assert np.isnan(out[("labels_per_paragraph", "legacy15")])
        assert np.isnan(out[("non_policy_share", "legacy15")])
        assert np.isnan(out[("effective_topics", "legacy15")])
        assert np.isnan(out[("fk_grade", "none")])
        assert np.isnan(out[("neither_share", "none")])

    def test_columns_are_a_measure_taxonomy_multiindex(self):
        """Structural, not numerical: the assembly step iterates these pairs to
        emit one long-format row each, so both level names must be present."""
        out = R.measures_from_sums(_sums(), _topic_counts())

        assert list(out.columns.names) == ["measure", "taxonomy"]
        assert ("effective_topics", "legacy15") in out.columns
        assert ("mean_paragraph_words", R.NO_TAXONOMY) in out.columns


# =========================================================================== #
# The thirteen style-marker series, written out as literals. Every other test in
# this file (and the `register_corpus` / `register_panel` builders in conftest)
# iterates `R.STYLE_MARKERS` itself, so the constant is self-referential
# everywhere else: DELETING a marker from it removes the column from the fixture
# and from the expectation in one step, and the whole suite stays green while a
# published series silently disappears.
_PUBLISHED_STYLE_MARKERS = (
    "mechanism",
    "religiosity",
    "nostalgia",
    "future",
    "us_them",
    "opponents",
    "hype",
    "doom",
    "boosters",
    "hedges",
    "superlatives",
    "nrc_hope",
    "nrc_fear",
)


class TestStyleMarkerSpec:
    def test_the_published_marker_series_are_exactly_these_thirteen(self):
        """Spec anchor, decoupled from the constant the production code reads.
        `mechanism` in particular is the depth measure `indices.py` names "the
        vocabulary of actually governing" — the one style series that speaks to
        the "presidents say less about each subject" half of the headline rather
        than to the values/combat register displacing it."""
        assert R.STYLE_MARKERS == _PUBLISHED_STYLE_MARKERS
        assert len(R.STYLE_MARKERS) == 13
        assert len(set(R.STYLE_MARKERS)) == 13

    def test_every_published_marker_gets_its_own_per_10k_word_column(self):
        """One measure column per marker, at the documented per-10,000-word
        scale. The sums are keyed by literal name, so a marker dropped from
        `STYLE_MARKERS` still reaches this function's input and its absence from
        the OUTPUT is what fails."""
        sums = _sums(**{marker: 5.0 for marker in _PUBLISHED_STYLE_MARKERS})

        out = R.measures_from_sums(sums, _topic_counts())

        for marker in _PUBLISHED_STYLE_MARKERS:
            assert (marker, R.NO_TAXONOMY) in out.columns, marker
            # 5 occurrences per 1,000 words == 50 per 10,000 words.
            assert out.iloc[0][(marker, R.NO_TAXONOMY)] == pytest.approx(50.0), marker


# =========================================================================== #
class TestRank:
    def test_ranks_ascending_from_one(self):
        assert R._rank(np.array([[30.0, 10.0, 20.0]])).tolist() == [[3.0, 1.0, 2.0]]

    def test_ties_share_the_average_rank(self):
        """Several style markers are constant at zero across the early eras. Left
        to argsort's tie-breaking they would get a spurious 1-2-3 ordering and
        manufacture a trend out of nothing."""
        assert R._rank(np.array([[5.0, 5.0, 9.0]])).tolist() == [[1.5, 1.5, 3.0]]

    def test_a_fully_constant_row_gets_one_shared_rank(self):
        assert R._rank(np.array([[7.0, 7.0, 7.0]])).tolist() == [[2.0, 2.0, 2.0]]

    def test_rows_are_ranked_independently(self):
        out = R._rank(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))

        assert out.tolist() == [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]


# =========================================================================== #
class TestSpearmanVsPosition:
    def test_monotone_increasing_series_is_exactly_plus_one(self):
        assert R.spearman_vs_position(np.array([1.0, 2.0, 3.0, 9.0]))[0] == pytest.approx(1.0)

    def test_monotone_decreasing_series_is_exactly_minus_one(self):
        assert R.spearman_vs_position(np.array([9.0, 3.0, 2.0, 1.0]))[0] == pytest.approx(-1.0)

    def test_constant_series_is_nan_because_the_correlation_is_undefined(self):
        """The distinction the report depends on. A flat series has zero rank
        variance, so its correlation does not exist — publishing 0.0 would read as
        "measured, and flat" when the truth is "not measurable"."""
        assert np.isnan(R.spearman_vs_position(np.array([2.0, 2.0, 2.0, 2.0]))[0])

    def test_all_zero_series_is_nan_too(self):
        """The shape this actually takes in the data: a style marker that never
        fires in any era."""
        assert np.isnan(R.spearman_vs_position(np.zeros(9))[0])

    def test_any_missing_value_makes_the_whole_row_nan(self):
        """A correlation over a partly missing series would not mean what the
        column says it means, so it is refused rather than computed on what is
        left."""
        assert np.isnan(R.spearman_vs_position(np.array([1.0, np.nan, 3.0, 4.0]))[0])

    def test_fewer_than_three_eras_is_nan(self):
        """Two points are perfectly correlated by construction, which is not
        evidence of a trend."""
        assert np.isnan(R.spearman_vs_position(np.array([1.0, 5.0]))[0])

    def test_partial_ties_use_averaged_ranks(self):
        """Ranks (1.5, 1.5, 3) against positions (1, 2, 3) give sqrt(3)/2. Pins
        the tie handling to a closed form rather than "some number below 1"."""
        out = R.spearman_vs_position(np.array([1.0, 1.0, 2.0]))[0]

        assert out == pytest.approx(math.sqrt(3) / 2)

    def test_rows_are_scored_independently_and_bad_rows_do_not_infect_good_ones(self):
        """The bootstrap hands in a (B, n_eras) block where individual replicates
        can be undefined. One nan row must not nan the whole column."""
        values = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, np.nan, 3.0],
            [5.0, 5.0, 5.0],
        ])

        out = R.spearman_vs_position(values)

        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(-1.0)
        assert np.isnan(out[2])
        assert np.isnan(out[3])


# =========================================================================== #
_NINE_ERAS = (1770, 1800, 1830, 1860, 1890, 1920, 1950, 1980, 2010)


class TestTrendStatistics:
    def test_reports_exactly_the_four_published_statistics(self):
        values = np.arange(9.0)[None, :]

        out = R.trend_statistics(values, _NINE_ERAS)

        assert set(out) == {
            "spearman_vs_era",
            "delta_last_minus_first",
            "delta_modern_minus_early",
            "delta_modern_minus_postbellum",
        }

    def test_a_statistic_computed_but_not_published_raises(self, monkeypatch):
        """`TREND_STATISTICS` and this function's keys are two sources of truth.

        `build_trends` publishes by iterating the constant, so a statistic added
        here and not there would be computed on all 111 arms and silently never
        reach the parquet — and nothing downstream could notice, because the
        parquet is the only thing anyone reads. This mirrors the guard
        `reference_genre_mix` already raises for `REFERENCE_GENRES`.
        """
        monkeypatch.setattr(
            R, "BLOCK_CONTRASTS",
            {**R.BLOCK_CONTRASTS, "delta_modern_minus_gilded": (R.MODERN_ERAS, (1890,))},
        )

        with pytest.raises(ValueError, match="drifted from the published set"):
            R.trend_statistics(np.arange(9.0)[None, :], _NINE_ERAS)

    def test_a_published_statistic_that_stops_being_computed_raises(self, monkeypatch):
        """The other direction: dropping a block contrast without updating the
        constant would publish an all-NaN statistic rather than fail."""
        monkeypatch.setattr(
            R, "BLOCK_CONTRASTS",
            {"delta_modern_minus_early": R.BLOCK_CONTRASTS["delta_modern_minus_early"]},
        )

        with pytest.raises(ValueError, match="drifted from the published set"):
            R.trend_statistics(np.arange(9.0)[None, :], _NINE_ERAS)

    def test_the_guard_names_both_sides_so_the_fix_is_obvious(self, monkeypatch):
        monkeypatch.setattr(
            R, "BLOCK_CONTRASTS",
            {**R.BLOCK_CONTRASTS, "delta_extra": (R.MODERN_ERAS, R.EARLY_ERAS)},
        )

        with pytest.raises(ValueError) as excinfo:
            R.trend_statistics(np.arange(9.0)[None, :], _NINE_ERAS)

        message = str(excinfo.value)
        assert "TREND_STATISTICS" in message
        assert "delta_extra" in message

    def test_delta_last_minus_first_is_the_endpoint_contrast(self):
        values = np.array([[2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 11.0]])

        out = R.trend_statistics(values, _NINE_ERAS)

        assert out["delta_last_minus_first"][0] == pytest.approx(9.0)

    def test_block_contrasts_average_three_eras_against_three(self):
        """The block contrasts exist because the 1770 era holds 28 speeches and an
        endpoint contrast would rest almost entirely on it. Each is a mean of
        three eras minus a mean of three others, hand-checked here."""
        values = np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]])

        out = R.trend_statistics(values, _NINE_ERAS)

        assert out["delta_modern_minus_early"][0] == pytest.approx(200.0 - 2.0)
        assert out["delta_modern_minus_postbellum"][0] == pytest.approx(200.0 - 20.0)

    def test_block_contrast_is_nan_when_a_block_is_absent_from_the_eras(self):
        """A subset of eras (a treatment where some era produced no cell) must not
        silently redefine the contrast to whatever eras happened to be present."""
        eras = (1770, 1800, 1830)
        values = np.array([[1.0, 2.0, 3.0]])

        out = R.trend_statistics(values, eras)

        assert np.isnan(out["delta_modern_minus_early"][0])
        assert np.isnan(out["delta_modern_minus_postbellum"][0])
        assert out["delta_last_minus_first"][0] == pytest.approx(2.0)

    def test_operates_row_wise_over_bootstrap_replicates(self):
        values = np.array([np.arange(9.0), np.arange(9.0)[::-1]])

        out = R.trend_statistics(values, _NINE_ERAS)

        assert out["spearman_vs_era"].tolist() == pytest.approx([1.0, -1.0])
        assert out["delta_last_minus_first"].tolist() == pytest.approx([8.0, -8.0])

    def test_era_block_constants_match_the_thirty_year_bands_they_name(self):
        """Spec anchor, decoupled from the constants the production code reads.
        Every other test above references `_NINE_ERAS` and the module's own block
        tuples, so a regression that quietly reassigned an era to another block
        would slip past them all — they would simply contrast different columns
        and still pass."""
        assert R.EARLY_ERAS == (1770, 1800, 1830)
        assert R.POSTBELLUM_ERAS == (1860, 1890, 1920)
        assert R.MODERN_ERAS == (1950, 1980, 2010)
        assert R.BLOCK_CONTRASTS == {
            "delta_modern_minus_early": (R.MODERN_ERAS, R.EARLY_ERAS),
            "delta_modern_minus_postbellum": (R.MODERN_ERAS, R.POSTBELLUM_ERAS),
        }
        assert set(R.TREND_STATISTICS) == {
            "spearman_vs_era",
            "delta_modern_minus_early",
            "delta_modern_minus_postbellum",
            "delta_last_minus_first",
        }


# =========================================================================== #
class TestBootstrapPValue:
    def test_a_replicate_cloud_entirely_on_one_side_hits_the_resolution_floor(self):
        """A bootstrap of B draws cannot resolve a p-value below 1/B, so an
        all-positive cloud reports exactly that rather than 0 — reporting 0 would
        claim precision the resampling does not have."""
        assert R.bootstrap_p_value(np.full(200, 3.0)) == pytest.approx(1 / 200)

    def test_the_floor_moves_with_the_number_of_replicates(self):
        assert R.bootstrap_p_value(np.full(50, 3.0)) == pytest.approx(1 / 50)

    def test_a_cloud_straddling_zero_evenly_is_one(self):
        replicates = np.concatenate([np.full(50, 1.0), np.full(50, -1.0)])

        assert R.bootstrap_p_value(replicates) == pytest.approx(1.0)

    def test_is_twice_the_smaller_tail(self):
        """The percentile-interval inversion: 5 of 100 replicates on the far side
        of zero is p = 0.10, not 0.05."""
        replicates = np.concatenate([np.full(95, 2.0), np.full(5, -2.0)])

        assert R.bootstrap_p_value(replicates) == pytest.approx(0.10)

    def test_sign_symmetric(self):
        """Nothing about the statistic privileges a rise over a fall."""
        rising = np.concatenate([np.full(95, 2.0), np.full(5, -2.0)])

        assert R.bootstrap_p_value(-rising) == pytest.approx(
            R.bootstrap_p_value(rising)
        )

    def test_is_capped_at_one(self):
        """Replicates sitting exactly on zero count in BOTH tails, so the raw
        `2 * min(...)` would be 2 without the cap."""
        assert R.bootstrap_p_value(np.zeros(100)) == pytest.approx(1.0)

    def test_nan_replicates_are_excluded_from_the_denominator(self):
        """Undefined replicates are not evidence in either direction. Counting
        them as zeros would drag every p-value toward 1."""
        replicates = np.concatenate([np.full(95, 2.0), np.full(5, -2.0), np.full(400, np.nan)])

        assert R.bootstrap_p_value(replicates) == pytest.approx(0.10)

    def test_the_floor_is_set_by_the_usable_replicates_not_the_attempted_ones(self):
        """The same exclusion, on the resolution FLOOR rather than the tail
        share. A bootstrap that attempted 500 draws but got 100 usable ones
        cannot resolve below 1/100; quoting 1/500 would claim five times the
        precision the resampling actually delivered.

        The test above cannot see this: its two-sided share is 0.10, far above
        either candidate floor, so both denominators give the same answer. Here
        the cloud is entirely one-sided, which makes the floor the whole
        result."""
        replicates = np.concatenate([np.full(100, 3.0), np.full(400, np.nan)])

        assert R.bootstrap_p_value(replicates) == pytest.approx(1 / 100)

    def test_all_nan_is_nan(self):
        assert np.isnan(R.bootstrap_p_value(np.full(10, np.nan)))

    def test_empty_is_nan(self):
        assert np.isnan(R.bootstrap_p_value(np.array([])))


# =========================================================================== #
class TestPercentileCi:
    def test_bounds_are_the_2_5_and_97_5_percentiles(self):
        replicates = np.arange(1_000.0)[:, None]

        low, high, valid = R._percentile_ci(replicates)

        assert low[0] == pytest.approx(np.quantile(np.arange(1_000.0), 0.025))
        assert high[0] == pytest.approx(np.quantile(np.arange(1_000.0), 0.975))
        assert valid[0] == 1_000

    def test_alpha_is_the_module_constant_and_is_two_sided(self):
        """Spec anchor: a one-sided or 90% interval would still pass every
        "bounds bracket the middle" style check."""
        assert R.CI_ALPHA == 0.05
        replicates = np.arange(10_000.0)[:, None]

        low, high, _ = R._percentile_ci(replicates)

        assert low[0] == pytest.approx(249.75, abs=1.0)
        assert high[0] == pytest.approx(9_749.25, abs=1.0)

    def test_nan_replicates_are_dropped_from_the_quantile_and_counted(self):
        """A resample that happened to contain no paragraphs of the relevant kind
        gives an undefined replicate. Treating those as zeros would drag the
        lower bound down; propagating them would nan an otherwise fine CI."""
        replicates = np.concatenate([np.arange(100.0), np.full(900, np.nan)])[:, None]

        low, high, valid = R._percentile_ci(replicates)

        assert valid[0] == 100
        assert low[0] == pytest.approx(np.quantile(np.arange(100.0), 0.025))
        assert np.isfinite(high[0])

    def test_an_all_nan_column_gives_nan_bounds_and_zero_valid(self):
        replicates = np.column_stack([np.arange(100.0), np.full(100, np.nan)])

        low, high, valid = R._percentile_ci(replicates)

        assert np.isfinite(low[0]) and valid[0] == 100
        assert np.isnan(low[1]) and np.isnan(high[1]) and valid[1] == 0

    def test_no_replicates_at_all_gives_nan_bounds(self):
        low, high, valid = R._percentile_ci(np.zeros((0, 3)))

        assert np.isnan(low).all() and np.isnan(high).all()
        assert valid.tolist() == [0, 0, 0]
