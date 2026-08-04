"""The speech-clustered bootstrap, the trends assembly, and the two functions
either side of it (`rarefied_effective_topics`, `era_counts`, `write_trends`,
`main`).

The load-bearing property here is the resampling UNIT. The within-speech ICC of
paragraph labels runs up to 0.119, so a paragraph-level bootstrap would pretend
36,229 independent observations exist where there are 1,057 speeches and would
publish intervals far too narrow. `TestSpeechClusteredResampling` is written to
fail against a paragraph-level bootstrap specifically, not merely to observe that
some interval came back.

Everything runs on synthetic panels with a small `n_bootstrap`; nothing reads the
corpus and nothing writes outside tmp_path.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from presidential_profiles import register as R

SOTU = R.SOTU_TYPE
INAUGURAL = "inaugural_address"
SPECIAL = "special_message_to_congress"
VETO = "veto_or_signing_statement"
REMARKS = "public_remarks_or_address"

# `register` has no console script — like `taxonomy`, `indices` and the other
# analysis modules it is run as `python -m presidential_profiles.register`, the
# command `notes/register-findings-v1.md` gives for reproducing the table. So
# argv[0] is only argparse's `prog`, and naming a `pp-register` entry point here
# would advertise one that does not exist.
ARGV0 = "python -m presidential_profiles.register"


def _panel_rows(eras=(1800, 1830, 1860), per_genre=3, **overrides):
    """Speeches spanning every reference genre in every era — the condition
    `reference_genre_mix` enforces before a standardized estimate is defined."""
    rows = []
    for era in eras:
        for genre in (INAUGURAL, SPECIAL, SOTU, VETO):
            for i in range(per_genre):
                rows.append({
                    "doc_name": f"d{era}-{genre}-{i}",
                    "year": era + 5,
                    "speech_type": genre,
                    **overrides,
                })
    return rows


def _trend_panel(register_panel, eras=(1800, 1830, 1860), extra=()):
    """A panel whose `mean_paragraph_words` rises 100 -> 200 -> 300 across the
    eras, so the end-to-end trend statistics have a known answer.

    Speeches inside an era deliberately differ from one another (words per
    paragraph runs target-50 / target / target+50 within every genre) while
    averaging exactly to the era target. Without that within-era spread every
    resample would return the identical sums and the bootstrap would be
    silently degenerate — a panel that could not tell a working bootstrap from a
    broken one.
    """
    rows = []
    for step, era in enumerate(eras):
        target = 100 * (step + 1)
        for position, row in enumerate(_panel_rows(eras=(era,))):
            words_per_paragraph = target + 50 * ((position % 3) - 1)
            row["n_paragraphs"] = 16.0
            row["para_words"] = 16.0 * words_per_paragraph
            rows.append(row)
    rows.extend(dict(spec) for spec in extra)
    counts = {
        "legacy15": np.ones((len(rows), 3), dtype=float),
        "llm_level2": np.ones((len(rows), 4), dtype=float),
    }
    return register_panel(rows, topic_counts=counts)


# =========================================================================== #
class TestReplicateWeights:
    def _cells(self, sizes_and_alphas):
        cells = []
        offset = 0
        for size, alpha in sizes_and_alphas:
            cells.append((np.arange(offset, offset + size), alpha))
            offset += size
        return cells

    def test_row_zero_is_the_observed_data_not_a_resample(self):
        """The point estimate must be the data itself. If row 0 were a draw, the
        published `value` would move with the seed."""
        cells = self._cells([(4, 1.0)])

        positions, weights = R._replicate_weights(
            cells, 100, np.random.default_rng(0)
        )

        assert positions.tolist() == [0, 1, 2, 3]
        assert weights[0].tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_there_is_one_weight_column_per_speech(self):
        """Structural, and the crux: the resampling design has as many draws as
        the group has SPEECHES. Paragraphs never appear in this matrix — they
        ride along inside each speech's sufficient statistics."""
        cells = self._cells([(7, 1.0)])

        positions, weights = R._replicate_weights(cells, 20, np.random.default_rng(0))

        assert len(positions) == 7
        assert weights.shape == (21, 7)

    def test_replicate_rows_are_integer_speech_counts_summing_to_the_cell_size(self):
        """A replicate is a multinomial count vector over the cell's speeches —
        identical to drawing n speeches with replacement. Non-integer weights (or
        a different total) would mean something other than a speech bootstrap."""
        cells = self._cells([(5, 1.0)])

        _, weights = R._replicate_weights(cells, 200, np.random.default_rng(3))
        replicates = weights[1:]

        assert np.array_equal(replicates, np.round(replicates))
        assert (replicates.sum(axis=1) == 5).all()
        assert replicates.max() > 1        # some speech really was drawn twice

    def test_resampling_is_within_cell_with_each_cell_size_held_fixed(self):
        """The genre-standardized arm keeps its design: a replicate redraws
        inside each genre cell, so the reference weights stay external to the
        resampling. Pooling the cells would let a replicate contain 8 annual
        messages and no inaugurals."""
        cells = self._cells([(3, 2.0), (5, 0.5)])

        _, weights = R._replicate_weights(cells, 100, np.random.default_rng(1))
        replicates = weights[1:]

        assert np.allclose(replicates[:, :3].sum(axis=1), 3 * 2.0)
        assert np.allclose(replicates[:, 3:].sum(axis=1), 5 * 0.5)

    def test_cell_weight_multiplies_every_draw_in_that_cell(self):
        cells = self._cells([(4, 2.5)])

        _, weights = R._replicate_weights(cells, 50, np.random.default_rng(2))

        assert np.allclose(weights[1:] % 2.5, 0.0)

    def test_is_deterministic_for_a_given_seed(self):
        cells = self._cells([(6, 1.0)])

        a = R._replicate_weights(cells, 50, np.random.default_rng(11))[1]
        b = R._replicate_weights(cells, 50, np.random.default_rng(11))[1]

        assert np.array_equal(a, b)

    def test_different_seeds_give_different_replicates(self):
        cells = self._cells([(6, 1.0)])

        a = R._replicate_weights(cells, 50, np.random.default_rng(11))[1]
        b = R._replicate_weights(cells, 50, np.random.default_rng(12))[1]

        assert not np.array_equal(a, b)

    def test_zero_replicates_leaves_only_the_observed_row(self):
        cells = self._cells([(3, 1.0)])

        _, weights = R._replicate_weights(cells, 0, np.random.default_rng(0))

        assert weights.shape == (1, 3)

    def test_no_cells_gives_an_empty_design(self):
        positions, weights = R._replicate_weights([], 10, np.random.default_rng(0))

        assert positions.size == 0
        assert weights.shape == (11, 0)


# =========================================================================== #
class TestSpeechClusteredResampling:
    """Written to fail against a paragraph-level bootstrap.

    The panel has four speeches of 50 paragraphs each. Two are entirely
    zero-issue paragraphs and two are entirely labelled, so the observed
    non-policy share is 0.5 and ALL of the variation lives between speeches —
    the maximal-ICC case. A speech bootstrap can therefore only ever produce
    shares on the k/4 grid and yields a near-[0, 1] interval; a paragraph
    bootstrap concentrates around 0.5 with a standard error of ~0.035.
    """

    def _panel(self, register_panel):
        rows = []
        for i in range(4):
            all_unlabelled = i < 2
            rows.append({
                "doc_name": f"d{i}",
                "year": 1805,
                "speech_type": SOTU,
                "n_paragraphs": 50.0,
                "para_words": 5_000.0,
                "legacy_zero_paras": 50.0 if all_unlabelled else 0.0,
            })
        return register_panel(rows)

    def _replicates(self, panel, n_bootstrap=2_000, seed=0):
        cells = [(np.arange(4), 1.0)]
        point, replicates, columns, counts = R._estimate_group(
            panel, cells, n_bootstrap, np.random.default_rng(seed)
        )
        column = list(columns).index(("non_policy_share", "legacy15"))
        return point, replicates[:, column], counts

    def test_point_estimate_is_the_observed_share(self, register_panel):
        point, _, counts = self._replicates(self._panel(register_panel), n_bootstrap=10)

        assert point[("non_policy_share", "legacy15")] == pytest.approx(0.5)
        assert counts == {"n_speeches": 4, "n_paragraphs": 200, "n_words": 20_000}

    def test_every_replicate_lands_on_the_four_speech_grid(self, register_panel):
        """The sharpest available discriminator. Resampling four speeches can
        only ever yield shares in {0, .25, .5, .75, 1}; resampling 200
        paragraphs would produce values off that grid on nearly every draw."""
        _, replicates, _ = self._replicates(self._panel(register_panel))

        distinct = set(np.round(replicates, 10))
        assert distinct <= {0.0, 0.25, 0.5, 0.75, 1.0}
        assert len(distinct) > 1        # not vacuous: the bootstrap really varied

    def test_interval_is_far_wider_than_a_paragraph_level_bootstrap_would_give(
        self, register_panel
    ):
        """The consequence, stated as the comparison the plan forbids getting
        wrong. A paragraph-level bootstrap on the identical data understates the
        uncertainty by roughly an order of magnitude."""
        _, replicates, _ = self._replicates(self._panel(register_panel))
        low, high, valid = R._percentile_ci(replicates[:, None])

        rng = np.random.default_rng(0)
        paragraph_level = rng.binomial(200, 0.5, size=2_000) / 200
        p_low, p_high, _ = R._percentile_ci(paragraph_level[:, None])

        assert valid[0] == 2_000
        assert high[0] - low[0] > 0.9
        assert (high[0] - low[0]) > 5 * (p_high[0] - p_low[0])

    def test_the_replicate_block_is_the_draws_and_never_the_observed_row(
        self, register_panel
    ):
        """`_replicate_weights` puts the observed data in row 0 and the draws
        beneath it, so `_estimate_group` has to slice the replicate block from
        row 1. Slicing `[:-1]` instead would put the point estimate INTO the
        resample cloud and drop one real draw — pulling every percentile interval
        toward the observed value by construction, and doing it invisibly at
        B = 2,000 where one row in two thousand moves nothing a test would see.

        With a single replicate the two are directly separable: the observed
        share is 0.5, and this seed's one draw redraws speeches [1, 0, 0, 3] for
        a share of 0.25.
        """
        panel = self._panel(register_panel)
        cells = [(np.arange(4), 1.0)]

        point, replicates, columns, _ = R._estimate_group(
            panel, cells, 1, np.random.default_rng(0)
        )
        column = list(columns).index(("non_policy_share", "legacy15"))

        assert point[("non_policy_share", "legacy15")] == pytest.approx(0.5)
        assert replicates.shape == (1, len(columns))
        assert replicates[0, column] == pytest.approx(0.25)

    def test_paragraphs_are_never_split_away_from_their_speech(self, register_panel):
        """The clustering claim in its own terms: a replicate's paragraph count
        is always a whole number of speeches' worth (a multiple of 50 here), so
        no replicate ever contains half of a speech's paragraphs."""
        panel = self._panel(register_panel)
        cells = [(np.arange(4), 1.0)]
        positions, weights = R._replicate_weights(cells, 500, np.random.default_rng(0))
        paragraphs = panel.scalars["n_paragraphs"].to_numpy()[positions]

        drawn = weights[1:] @ paragraphs

        assert (drawn == 200).all()          # 4 speeches x 50 paragraphs, always
        assert np.allclose(weights[1:] % 1.0, 0.0)


# =========================================================================== #
class TestEstimateGroup:
    def test_genre_standardization_moves_the_estimate_to_the_reference_mix(
        self, register_panel
    ):
        """End-to-end arithmetic on the treatment that matters most. Three annual
        messages (non-policy share 1.0) and three inaugurals (share 0.0), equal
        paragraph mass, so the raw estimate is 0.5. Standardized to a mix that is
        75% annual message it must be exactly 0.75 — the reference weight itself."""
        rows = [
            {"doc_name": f"s{i}", "year": 1805, "speech_type": SOTU,
             "n_paragraphs": 10.0, "para_words": 1_000.0, "legacy_zero_paras": 10.0}
            for i in range(3)
        ] + [
            {"doc_name": f"i{i}", "year": 1805, "speech_type": INAUGURAL,
             "n_paragraphs": 10.0, "para_words": 1_000.0, "legacy_zero_paras": 0.0}
            for i in range(3)
        ]
        panel = register_panel(rows)
        mix = pd.Series(
            {INAUGURAL: 0.25, SPECIAL: 0.0, SOTU: 0.75, VETO: 0.0}
        ).reindex(list(R.REFERENCE_GENRES))
        positions = np.arange(6)

        raw = R._estimate_group(
            panel, R._cell_plan(panel, positions, "raw", mix), 0,
            np.random.default_rng(0),
        )[0]
        standardized = R._estimate_group(
            panel, R._cell_plan(panel, positions, "genre_standardized", mix), 0,
            np.random.default_rng(0),
        )[0]

        assert raw[("non_policy_share", "legacy15")] == pytest.approx(0.5)
        assert standardized[("non_policy_share", "legacy15")] == pytest.approx(0.75)

    def test_counts_are_unweighted_so_the_reader_sees_the_real_sample(
        self, register_panel
    ):
        """The reported `n_speeches` / `n_paragraphs` / `n_words` describe the
        actual data behind the cell, not the reweighted mass — otherwise a
        standardized era would advertise a sample size it never had.

        Both paragraphs per speech (20 vs 10) and words per paragraph (100 vs
        300) differ by genre on purpose, so the reweighted `n_speeches` (4.95)
        and `n_words` (10,800) both genuinely differ from the raw 6 and 15,000.
        A fixture with uniform genres cannot tell the two apart. `n_paragraphs`
        is the exception and always will be: direct standardization preserves
        total paragraph mass exactly, so raw and reweighted agree by
        construction."""
        rows = [
            {"doc_name": f"s{i}", "year": 1805, "speech_type": SOTU,
             "n_paragraphs": 20.0, "para_words": 2_000.0}
            for i in range(3)
        ] + [
            {"doc_name": f"i{i}", "year": 1805, "speech_type": INAUGURAL,
             "n_paragraphs": 10.0, "para_words": 3_000.0}
            for i in range(3)
        ]
        panel = register_panel(rows)
        mix = pd.Series(
            {INAUGURAL: 0.1, SPECIAL: 0.0, SOTU: 0.9, VETO: 0.0}
        ).reindex(list(R.REFERENCE_GENRES))

        cells = R._cell_plan(panel, np.arange(6), "genre_standardized", mix)
        point, _, _, counts = R._estimate_group(
            panel, cells, 0, np.random.default_rng(0)
        )

        assert counts == {"n_speeches": 6, "n_paragraphs": 90, "n_words": 15_000}
        # The reweighted estimate itself is NOT the raw one — proving the counts
        # were held raw while the measure was standardized.
        assert point[("mean_paragraph_words", "none")] == pytest.approx(120.0)
        assert point[("mean_paragraph_words", "none")] != pytest.approx(15_000 / 90)

    def test_speech_positions_stay_matched_to_their_weights_when_genres_interleave(
        self, register_panel
    ):
        """`_estimate_group` multiplies one weight vector into two separate
        blocks — the scalar frame and each topic matrix — and both are indexed by
        the SAME `positions` array, which `_cell_plan` builds in reference-genre
        order rather than in panel order. Re-sorting either one silently reweights
        a different set of speeches.

        Every other fixture in the suite lays its genres out contiguously and
        gives every speech an identical topic row, so the position order is
        unobservable. Here the genres alternate (SOTU on the even rows,
        inaugurals on the odd) and the two differ sharply in breadth: annual
        messages spend all twelve assignments on one topic, inaugurals spread
        theirs over four. So the entropy the standardized arm reports depends on
        the mapping being right.
        """
        rows = []
        for i in range(6):
            rows.append({
                "doc_name": f"d{i}", "year": 1805,
                "speech_type": SOTU if i % 2 == 0 else INAUGURAL,
                "n_paragraphs": 12.0, "para_words": 1_200.0,
            })
        counts = np.zeros((6, 4))
        counts[0::2] = [12.0, 0.0, 0.0, 0.0]     # annual messages: one topic
        counts[1::2] = [3.0, 3.0, 3.0, 3.0]      # inaugurals: four topics
        panel = register_panel(rows, topic_counts={"llm_level2": counts})
        mix = pd.Series(
            {INAUGURAL: 0.25, SPECIAL: 0.0, SOTU: 0.75, VETO: 0.0}
        ).reindex(list(R.REFERENCE_GENRES))

        cells = R._cell_plan(panel, np.arange(6), "genre_standardized", mix)
        point = R._estimate_group(panel, cells, 0, np.random.default_rng(0))[0]

        # Each cell holds 36 of the group's 72 paragraphs, so the weights are
        # 0.25 * 72/36 = 0.5 on the inaugural rows (1, 3, 5) and 0.75 * 72/36 =
        # 1.5 on the annual-message rows (0, 2, 4).
        expected = 0.5 * counts[[1, 3, 5]].sum(axis=0) + 1.5 * counts[[0, 2, 4]].sum(axis=0)
        assert expected.tolist() == [58.5, 4.5, 4.5, 4.5]
        assert point[("effective_topics_plugin", "llm_level2")] == pytest.approx(
            float(R.effective_topics(expected, miller_madow=False))
        )

    def test_an_empty_cell_plan_produces_nan_measures_and_zero_counts(
        self, register_panel
    ):
        """A group with no usable cell must not report a number. This is the path
        a `sotu_only` estimate takes in a year with no annual message."""
        panel = register_panel([
            {"doc_name": "d0", "year": 1805, "speech_type": REMARKS}
        ])

        point, replicates, columns, counts = R._estimate_group(
            panel, [], 5, np.random.default_rng(0)
        )

        assert counts == {"n_speeches": 0, "n_paragraphs": 0, "n_words": 0}
        assert np.isnan(point[("non_policy_share", "legacy15")])
        assert replicates.shape == (5, len(columns))
        assert np.isnan(replicates).all()


# =========================================================================== #
class TestBuildTrends:
    def test_emits_exactly_the_declared_columns(self, register_panel):
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        assert list(trends.columns) == R.TREND_COLUMNS
        assert list(trends.columns) == [
            "unit", "period", "measure", "taxonomy", "genre_treatment", "statistic",
            "value", "ci_low", "ci_high", "p_value", "n_speeches", "n_paragraphs",
            "n_words", "n_bootstrap", "n_bootstrap_valid", "bootstrap_seed",
        ]

    def test_a_row_missing_a_declared_key_raises_instead_of_publishing_nulls(
        self, monkeypatch, register_panel
    ):
        """`pd.DataFrame(rows, columns=...)` reindexes rather than validates.

        A row key that is missing or misspelled becomes an all-NaN COLUMN in the
        shipped artifact — a published series of 16,243 nulls — rather than an
        error. That is the one build failure a reader of the parquet cannot
        distinguish from a genuinely undefined measure, so it has to be loud.
        """
        original = R._rows_for_group

        def drop_a_key(*args, **kwargs):
            rows = original(*args, **kwargs)
            for row in rows:
                row.pop("n_words", None)
            return rows

        monkeypatch.setattr(R, "_rows_for_group", drop_a_key)

        with pytest.raises(ValueError, match="does not match TREND_COLUMNS"):
            R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

    def test_a_misspelled_row_key_names_both_the_missing_and_the_unexpected(
        self, monkeypatch, register_panel
    ):
        original = R._rows_for_group

        def typo(*args, **kwargs):
            rows = original(*args, **kwargs)
            for row in rows:
                row["n_wordz"] = row.pop("n_words")
            return rows

        monkeypatch.setattr(R, "_rows_for_group", typo)

        with pytest.raises(ValueError) as excinfo:
            R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        message = str(excinfo.value)
        assert "n_words" in message and "n_wordz" in message
        assert "all-null column" in message

    def test_a_group_that_produces_no_rows_raises_the_intended_error(
        self, monkeypatch, register_panel
    ):
        """An empty group must reach `_require_trend_columns`, not `IndexError`.

        The fail-fast check indexes into `rows`; if a builder ever returned
        nothing, a bare index would raise the wrong exception from the wrong line
        and hide which group was empty.
        """
        monkeypatch.setattr(R, "_rows_for_group", lambda *a, **k: [])

        with pytest.raises(ValueError, match="does not match TREND_COLUMNS"):
            R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

    def test_the_guard_passes_on_the_real_row_builders(self, register_panel):
        """Both row-dict builders are currently correct; this pins that they stay
        so, rather than only proving the guard fires when broken."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        assert not trends.isna().all(axis=0).any(), "no column is entirely null"

    def test_covers_era_year_and_trend_units(self, register_panel):
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        assert set(trends["unit"]) == {"era", "year", "trend"}
        assert set(trends.loc[trends["unit"] == "era", "period"]) == {1800, 1830, 1860}
        assert set(trends.loc[trends["unit"] == "year", "period"]) == {1805, 1835, 1865}

    def test_era_rows_exist_for_all_three_genre_treatments(self, register_panel):
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        era_rows = trends[trends["unit"] == "era"]

        assert set(era_rows["genre_treatment"]) == set(R.GENRE_TREATMENTS)
        assert set(era_rows["statistic"]) == {"level"}

    def test_year_rows_omit_the_standardized_treatment(self, register_panel):
        """A single year almost never contains all four reference genres, so
        reweighting it would be defined by whichever genres happened to occur —
        the opposite of standardizing."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        year_rows = trends[trends["unit"] == "year"]

        assert set(year_rows["genre_treatment"]) == {"raw", "sotu_only"}
        assert "genre_standardized" not in set(year_rows["genre_treatment"])

    def test_thin_years_are_masked_out(self, register_panel):
        """Matching the masking idiom elsewhere in the repo: a year below
        `MIN_YEAR_PARAGRAPHS` paragraphs is not published rather than allowed to
        spike a chart."""
        # A second year inside the 1800 era, so the era still spans every
        # reference genre and only the YEAR-level masking is under test.
        panel = _trend_panel(register_panel, extra=[{
            "doc_name": "thin-year", "year": 1806, "speech_type": SOTU,
            "n_paragraphs": 2.0, "para_words": 200.0,
        }])

        trends = R.build_trends(panel, n_bootstrap=5, seed=1)
        years = set(trends.loc[trends["unit"] == "year", "period"])

        assert 1806 not in years          # 2 paragraphs < MIN_YEAR_PARAGRAPHS
        assert 1805 in years              # 192 paragraphs, published
        # The thin year's speech is still counted at the era level.
        era_row = trends[
            (trends["unit"] == "era") & (trends["period"] == 1800)
            & (trends["genre_treatment"] == "raw")
            & (trends["measure"] == "mean_paragraph_words")
        ].iloc[0]
        assert era_row["n_speeches"] == 13

    def test_a_year_at_exactly_the_minimum_is_published_and_one_below_is_not(
        self, register_panel
    ):
        """The masking rule's boundary. `MIN_YEAR_PARAGRAPHS` is a strict `<`
        cut, so a year holding exactly 30 paragraphs IS published; the
        `test_thin_years_are_masked_out` fixture sits at 2 paragraphs and stays
        on the same side of the line however the comparison moves. Both years go
        inside the existing 1800 era, so only the YEAR-level rule is under test
        and `reference_genre_mix` still sees all four genres in every era.
        """
        panel = _trend_panel(register_panel, extra=[
            {"doc_name": "at-the-minimum", "year": 1806, "speech_type": SOTU,
             "n_paragraphs": 30.0, "para_words": 3_000.0},
            {"doc_name": "one-below", "year": 1807, "speech_type": SOTU,
             "n_paragraphs": 29.0, "para_words": 2_900.0},
        ])

        trends = R.build_trends(panel, n_bootstrap=5, seed=1)
        years = set(trends.loc[trends["unit"] == "year", "period"])

        assert 1806 in years        # exactly MIN_YEAR_PARAGRAPHS — kept
        assert 1807 not in years    # one paragraph short — masked

    def test_the_reference_mix_is_built_from_paragraph_mass_not_word_mass(
        self, register_panel
    ):
        """`reference_genre_mix` is tested on its own elsewhere; this pins the
        ARGUMENT `build_trends` hands it. The reweighting redistributes paragraph
        mass, so the reference has to be measured in paragraphs — a mix built
        from word mass would score each genre by how wordy it is rather than by
        how much of the record it is.

        The two disagree by a factor of nearly four here on purpose: annual
        messages carry four times the paragraphs of the inaugurals but a fifth of
        the words. Because annual messages are also the only genre holding
        non-policy paragraphs, the standardized `non_policy_share` comes out at
        exactly `mix[SOTU]` and reads the two candidate mixes apart.
        """
        rows = []
        for era in (1800, 1830):
            for i in range(3):
                rows += [
                    {"doc_name": f"{era}-sotu-{i}", "year": era + 5,
                     "speech_type": SOTU, "n_paragraphs": 20.0,
                     "para_words": 200.0, "legacy_zero_paras": 20.0},
                    {"doc_name": f"{era}-inaug-{i}", "year": era + 5,
                     "speech_type": INAUGURAL, "n_paragraphs": 5.0,
                     "para_words": 1_000.0, "legacy_zero_paras": 0.0},
                    {"doc_name": f"{era}-special-{i}", "year": era + 5,
                     "speech_type": SPECIAL, "n_paragraphs": 5.0,
                     "para_words": 50.0, "legacy_zero_paras": 0.0},
                    {"doc_name": f"{era}-veto-{i}", "year": era + 5,
                     "speech_type": VETO, "n_paragraphs": 5.0,
                     "para_words": 50.0, "legacy_zero_paras": 0.0},
                ]
        panel = register_panel(rows)

        trends = R.build_trends(panel, n_bootstrap=5, seed=1)
        row = trends[
            (trends["unit"] == "era")
            & (trends["period"] == 1800)
            & (trends["genre_treatment"] == "genre_standardized")
            & (trends["measure"] == "non_policy_share")
            & (trends["taxonomy"] == "legacy15")
        ].iloc[0]

        by_paragraphs = 120 / 210        # 120 annual-message paragraphs of 210
        by_words = 1_200 / 7_800         # ...but only 1,200 words of 7,800
        assert row["value"] == pytest.approx(by_paragraphs)
        assert row["value"] != pytest.approx(by_words)

    def test_a_year_with_no_annual_message_gets_a_raw_row_but_no_sotu_row(
        self, register_panel
    ):
        """A thick modern year of nothing but public remarks. `raw` publishes it;
        `sotu_only` must skip it entirely rather than emit a row of nans that a
        reader would take for a measured zero."""
        panel = _trend_panel(register_panel, extra=[{
            "doc_name": "remarks-year", "year": 1806, "speech_type": REMARKS,
            "n_paragraphs": 40.0, "para_words": 4_000.0,
        }])

        trends = R.build_trends(panel, n_bootstrap=5, seed=1)
        year_1806 = trends[(trends["unit"] == "year") & (trends["period"] == 1806)]

        assert set(year_1806["genre_treatment"]) == {"raw"}
        assert len(year_1806) > 0

    def test_trend_rows_carry_the_four_statistics_and_no_period(self, register_panel):
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        trend_rows = trends[trends["unit"] == "trend"]

        assert set(trend_rows["statistic"]) == set(R.TREND_STATISTICS)
        assert trend_rows["period"].isna().all()
        assert trend_rows["n_speeches"].isna().all()

    def test_only_trend_rows_carry_a_p_value(self, register_panel):
        """The p-value tests a trend against a null of zero, which is meaningless
        for a level. Level rows must therefore be NaN, and every trend row that
        had usable replicates must carry a real number — a silently dropped
        p-value column would leave the note with intervals and no test."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=50, seed=1)
        levels = trends[trends["statistic"] == "level"]
        trend_rows = trends[trends["unit"] == "trend"]
        usable = trend_rows[trend_rows["n_bootstrap_valid"] > 0]

        assert levels["p_value"].isna().all()
        assert len(usable) > 0
        assert usable["p_value"].notna().all()
        assert ((usable["p_value"] > 0) & (usable["p_value"] <= 1.0)).all()

    def test_a_trend_whose_replicates_all_share_a_sign_reports_the_resolution_floor(
        self, register_panel
    ):
        """Mean paragraph length rises 100 -> 200 -> 300 with within-era spread of
        only +/-50, so every replicate of `delta_last_minus_first` is positive.
        With B replicates the smallest resolvable p-value is 1/B, and that exact
        value must reach the output rather than a 0 or a NaN."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=64, seed=1)
        row = trends[
            (trends["unit"] == "trend")
            & (trends["genre_treatment"] == "raw")
            & (trends["measure"] == "mean_paragraph_words")
            & (trends["statistic"] == "delta_last_minus_first")
        ].iloc[0]

        assert row["n_bootstrap_valid"] == 64
        assert row["p_value"] == pytest.approx(1 / 64)
        assert row["ci_low"] > 0        # the whole interval is above zero

    def test_era_level_values_match_the_hand_computed_measure(self, register_panel):
        """The number that actually gets published, checked against arithmetic:
        the 1830 era holds 12 speeches of 4 paragraphs and 800 words each, so its
        raw mean paragraph length is exactly 200."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        row = trends[
            (trends["unit"] == "era")
            & (trends["period"] == 1830)
            & (trends["genre_treatment"] == "raw")
            & (trends["measure"] == "mean_paragraph_words")
        ].iloc[0]

        assert row["value"] == pytest.approx(200.0)
        assert row["n_speeches"] == 12
        assert row["n_paragraphs"] == 192      # 12 speeches x 16 paragraphs
        assert row["n_words"] == 38_400        # 192 paragraphs x 200 words

    def test_sotu_only_rows_count_only_the_annual_messages(self, register_panel):
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        row = trends[
            (trends["unit"] == "era")
            & (trends["period"] == 1830)
            & (trends["genre_treatment"] == "sotu_only")
            & (trends["measure"] == "mean_paragraph_words")
        ].iloc[0]

        assert row["n_speeches"] == 3          # 3 of the era's 12 speeches
        assert row["value"] == pytest.approx(200.0)

    def test_trend_statistic_values_match_the_era_series_they_summarize(
        self, register_panel
    ):
        """End-to-end wiring: mean paragraph length rises 100 -> 200 -> 300
        across the three eras, so the Spearman is exactly 1 and the endpoint
        contrast exactly 200."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        rows = trends[
            (trends["unit"] == "trend")
            & (trends["genre_treatment"] == "raw")
            & (trends["measure"] == "mean_paragraph_words")
        ].set_index("statistic")

        assert rows.loc["spearman_vs_era", "value"] == pytest.approx(1.0)
        assert rows.loc["delta_last_minus_first", "value"] == pytest.approx(200.0)

    def test_era_series_is_ordered_by_era_not_by_the_panel_row_order(
        self, register_panel
    ):
        """The era series is what every trend statistic is computed over, and its
        column order IS the time axis: a Spearman "against era" and an endpoint
        contrast both read position as time. Taking the eras in the order they
        happen to appear in the panel would make both statistics a function of
        how the corpus table was sorted.

        Every other fixture here feeds its speeches era-ascending, so the two are
        indistinguishable. This one is deliberately fed newest-first — asserted
        below, so the test cannot pass on an accidentally sorted panel — while
        mean paragraph length still rises 100 -> 200 -> 300 with era.
        """
        rows = []
        for step, era in enumerate((1800, 1830, 1860)):
            target = 100 * (step + 1)
            for position, row in enumerate(_panel_rows(eras=(era,))):
                row["n_paragraphs"] = 16.0
                row["para_words"] = 16.0 * (target + 50 * ((position % 3) - 1))
                rows.append(row)
        panel = register_panel(
            list(reversed(rows)),
            topic_counts={"legacy15": np.ones((len(rows), 3), dtype=float)},
        )
        eras = list(panel.speeches["era"])
        assert eras != sorted(eras)      # not vacuous: the panel is out of order

        trends = R.build_trends(panel, n_bootstrap=5, seed=1)
        summary = trends[
            (trends["unit"] == "trend")
            & (trends["genre_treatment"] == "raw")
            & (trends["measure"] == "mean_paragraph_words")
        ].set_index("statistic")
        levels = trends[
            (trends["unit"] == "era")
            & (trends["genre_treatment"] == "raw")
            & (trends["measure"] == "mean_paragraph_words")
        ].set_index("period")["value"]

        assert levels.loc[1800] == pytest.approx(100.0)
        assert levels.loc[1860] == pytest.approx(300.0)
        assert summary.loc["spearman_vs_era", "value"] == pytest.approx(1.0)
        assert summary.loc["delta_last_minus_first", "value"] == pytest.approx(200.0)

    def test_n_bootstrap_valid_is_counted_per_measure(self, register_panel):
        """`n_bootstrap_valid` is the provenance column a reader uses to tell a
        real interval from one built on a handful of usable replicates, so it
        must be that measure's own count rather than the frame's first.

        `proposal_vs_values` is undefined in every replicate here — no proposal
        and no values paragraphs anywhere, so its denominator is always zero —
        while `mean_paragraph_words` is defined in all of them. The two must not
        report the same number."""
        rows = []
        for era in (1800, 1830, 1860):
            for row in _panel_rows(eras=(era,)):
                row.update(n_paragraphs=16.0, para_words=1_600.0,
                           pv_proposal=0.0, pv_values=0.0, pv_mixed=0.0)
                rows.append(row)
        panel = register_panel(rows)

        trends = R.build_trends(panel, n_bootstrap=20, seed=1)
        era_rows = trends[
            (trends["unit"] == "era")
            & (trends["period"] == 1800)
            & (trends["genre_treatment"] == "raw")
        ].set_index("measure")["n_bootstrap_valid"]

        assert era_rows.loc["mean_paragraph_words"] == 20
        assert era_rows.loc["proposal_vs_values"] == 0

    def test_the_thin_year_mask_counts_only_the_paragraphs_the_treatment_uses(
        self, register_panel
    ):
        """The threshold applies to the paragraphs the TREATMENT actually
        estimates over, not to everything the year contains. A year holding one
        short annual message inside a flood of public remarks clears the
        threshold on `raw` and must still be masked on `sotu_only` — otherwise a
        `sotu_only` point would be published resting on five paragraphs, which is
        the exact spike the mask exists to prevent."""
        panel = _trend_panel(register_panel, extra=[
            {"doc_name": "tiny-sotu", "year": 1806, "speech_type": SOTU,
             "n_paragraphs": 5.0, "para_words": 500.0},
            {"doc_name": "many-remarks", "year": 1806, "speech_type": REMARKS,
             "n_paragraphs": 100.0, "para_words": 10_000.0},
        ])

        trends = R.build_trends(panel, n_bootstrap=5, seed=1)
        year_1806 = trends[(trends["unit"] == "year") & (trends["period"] == 1806)]

        assert len(year_1806) > 0                              # 105 paragraphs raw
        assert set(year_1806["genre_treatment"]) == {"raw"}    # but only 5 are SOTU

    def test_is_deterministic_under_a_fixed_seed(self, register_panel):
        panel = _trend_panel(register_panel)

        first = R.build_trends(panel, n_bootstrap=25, seed=99)
        second = R.build_trends(panel, n_bootstrap=25, seed=99)

        pd.testing.assert_frame_equal(first, second)

    def test_the_seed_changes_the_intervals_but_never_the_point_estimates(
        self, register_panel
    ):
        """Two things at once. The published `value` is the observed data and
        must be seed-invariant; the CI is a resampling product and must not be,
        or the bootstrap is not running."""
        panel = _trend_panel(register_panel)

        a = R.build_trends(panel, n_bootstrap=25, seed=1)
        b = R.build_trends(panel, n_bootstrap=25, seed=2)

        pd.testing.assert_series_equal(a["value"], b["value"])
        differing = ~np.isclose(a["ci_low"], b["ci_low"], equal_nan=True)
        assert differing.any()

    def test_the_seed_is_recorded_on_every_row(self, register_panel):
        """A published interval that cannot be reproduced is not evidence. The
        seed travels with the number, not with the run log."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=7, seed=4242)

        assert set(trends["bootstrap_seed"]) == {4242}
        assert set(trends["n_bootstrap"]) == {7}

    def test_default_seed_and_replicate_count_are_the_documented_ones(self):
        """Spec anchor: the committed `trends.parquet` records these, so a change
        silently invalidates every interval already published against them."""
        assert R.BOOTSTRAP_SEED == 20260721
        assert R.N_BOOTSTRAP == 20_000

    def test_intervals_bracket_the_point_estimate_on_a_well_sampled_measure(
        self, register_panel
    ):
        """Sanity on the assembled output rather than on `_percentile_ci` alone:
        a level row's CI must contain its value wherever the bootstrap produced
        usable replicates."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=200, seed=5)
        rows = trends[
            (trends["unit"] == "era")
            & (trends["measure"] == "mean_paragraph_words")
            & (trends["n_bootstrap_valid"] > 0)
        ]

        assert len(rows) == 9        # 3 eras x 3 treatments
        assert (rows["ci_low"] <= rows["value"] + 1e-9).all()
        assert (rows["ci_high"] >= rows["value"] - 1e-9).all()

    def test_rows_are_sorted_by_the_declared_key(self, register_panel):
        """The output is a long-format table a reader slices by hand, and the
        committed `trends.parquet` is compared byte-for-byte between runs, so row
        ORDER is part of the contract rather than an accident of emission.

        Rows are GENERATED era-major — every measure for era 1800 under `raw`,
        then every measure for 1830, then the whole thing again per treatment,
        then the year rows, then the trend rows. So the sorted order is a real
        rearrangement, which the first-row check below makes non-vacuous: in
        emission order row 0 is `effective_topics` (the first column
        `measures_from_sums` builds), and only after sorting does the
        alphabetically first measure lead.
        """
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)
        key = ["unit", "measure", "taxonomy", "genre_treatment", "statistic", "period"]

        pd.testing.assert_frame_equal(
            trends[key],
            trends[key].sort_values(key, kind="stable").reset_index(drop=True),
        )
        first = trends.iloc[0]
        assert first["unit"] == "era"                     # "era" < "trend" < "year"
        assert first["measure"] == min(trends["measure"])
        assert first["measure"] != "effective_topics"     # i.e. NOT emission order
        assert first["genre_treatment"] == "genre_standardized"   # < raw < sotu_only

    def test_period_dtype_is_nullable_integer(self, register_panel):
        """`period` and the three sample-size columns must be Int64 so the trend
        rows' missing values survive a parquet round trip as NA rather than
        turning the whole column into floats."""
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        for column in ("period", "n_speeches", "n_paragraphs", "n_words"):
            assert str(trends[column].dtype) == "Int64"


# =========================================================================== #
class TestRarefiedEffectiveTopics:
    def _panel(self, register_panel):
        """A thin era of one topic and a thick era spread over five topics, with
        every speech carrying five paragraphs. The thick era's full-sample
        entropy is five effective topics; rarefied to the thin era's budget it
        can only ever see one speech, so exactly one."""
        rows = [{
            "doc_name": "thin-0", "year": 1805, "speech_type": SOTU,
            "n_paragraphs": 5.0, "para_words": 500.0,
        }]
        rows += [
            {"doc_name": f"thick-{i}", "year": 1835, "speech_type": SOTU,
             "n_paragraphs": 5.0, "para_words": 500.0}
            for i in range(5)
        ]
        counts = np.zeros((6, 5))
        counts[0, 0] = 5.0
        for i in range(5):
            counts[1 + i, i] = 5.0
        return register_panel(rows, topic_counts={"llm_level2": counts})

    def test_budget_defaults_to_the_smallest_eras_paragraph_count(
        self, register_panel
    ):
        """Every era is scored on the terms of the thinnest one — that is what
        makes the comparison a rarefaction rather than a rescaling."""
        out = R.rarefied_effective_topics(
            self._panel(register_panel), n_draws=20, seed=1
        )

        assert set(out["budget_paragraphs"]) == {5}

    def test_rarefaction_removes_the_sample_size_advantage_of_the_thick_era(
        self, register_panel
    ):
        """The rival explanation, tested directly: more paragraphs means more
        chances to touch a rare topic. Cut to the thin era's budget the thick
        era's five effective topics collapse to one, and the thin era is
        unchanged — so the gap in the full-sample column was sample size."""
        out = R.rarefied_effective_topics(
            self._panel(register_panel), n_draws=50, seed=1
        ).set_index("era")

        assert out.loc[1830, "full_sample"] > 5.0     # 5 topics + Miller-Madow
        assert out.loc[1830, "rarefied_mean"] == pytest.approx(1.0)
        assert out.loc[1830, "rarefied_low"] == pytest.approx(1.0)
        assert out.loc[1830, "rarefied_high"] == pytest.approx(1.0)
        assert out.loc[1800, "full_sample"] == pytest.approx(1.0)
        assert out.loc[1800, "rarefied_mean"] == pytest.approx(1.0)

    def test_draws_whole_speeches_so_a_five_paragraph_budget_takes_one_speech(
        self, register_panel
    ):
        """The clustering argument applies to rarefaction exactly as it does to
        the bootstrap. A PARAGRAPH-level rarefaction of the thick era would draw
        five paragraphs from across five speeches and see five topics, not one —
        which is what the previous test's `rarefied_mean == 1.0` rules out."""
        out = R.rarefied_effective_topics(
            self._panel(register_panel), n_draws=50, seed=7
        ).set_index("era")

        assert out.loc[1830, "rarefied_mean"] < out.loc[1830, "full_sample"] / 4

    def test_each_draw_reshuffles_rather_than_taking_the_same_speeches_every_time(
        self, register_panel
    ):
        """Rarefaction's SPREAD is its content — `rarefied_low` / `rarefied_high`
        are subsampling percentiles — so a missing per-draw permutation would
        collapse the interval onto whichever speech happens to sit first in the
        panel, and report it as a point.

        The other rarefaction tests here cannot see that: their thick era gives
        every speech the same one-topic score, so which speech gets drawn does
        not matter. Here the era's five speeches differ in breadth on purpose
        (one single-topic speech against four five-topic ones) and the budget
        admits exactly one speech per draw, so a working shuffle must produce
        BOTH values while an unshuffled one can only ever produce the first
        speech's 1.0.
        """
        rows = [
            {"doc_name": f"s{i}", "year": 1835, "speech_type": SOTU,
             "n_paragraphs": 5.0, "para_words": 500.0}
            for i in range(5)
        ]
        counts = np.zeros((5, 5))
        counts[0, 0] = 5.0          # first speech: five assignments, one topic
        counts[1:, :] = 1.0         # the rest: one assignment on each of five
        panel = register_panel(rows, topic_counts={"llm_level2": counts})

        row = R.rarefied_effective_topics(
            panel, n_draws=200, seed=1, budget=5
        ).set_index("era").loc[1830]

        assert row["rarefied_low"] == pytest.approx(1.0)   # the narrow speech
        assert row["rarefied_high"] > 4.0                  # ...and a broad one
        assert 1.0 < row["rarefied_mean"] < row["rarefied_high"]

    def test_full_sample_column_is_the_unrarefied_estimate(self, register_panel):
        panel = self._panel(register_panel)

        out = R.rarefied_effective_topics(panel, n_draws=10, seed=1).set_index("era")
        expected = R.effective_topics(
            panel.topic_counts["llm_level2"][1:].sum(axis=0)
        )

        assert out.loc[1830, "full_sample"] == pytest.approx(float(expected))

    def test_is_deterministic_under_a_fixed_seed(self, register_panel):
        panel = self._panel(register_panel)

        a = R.rarefied_effective_topics(panel, n_draws=30, seed=3)
        b = R.rarefied_effective_topics(panel, n_draws=30, seed=3)

        pd.testing.assert_frame_equal(a, b)

    def test_an_explicit_budget_overrides_the_default(self, register_panel):
        out = R.rarefied_effective_topics(
            self._panel(register_panel), n_draws=10, seed=1, budget=25
        )

        assert set(out["budget_paragraphs"]) == {25}
        # With the whole thick era inside the budget, rarefaction is a no-op.
        thick = out.set_index("era").loc[1830]
        assert thick["rarefied_mean"] == pytest.approx(thick["full_sample"])

    def test_one_row_per_era_and_taxonomy_with_the_run_parameters_recorded(
        self, register_panel
    ):
        """Structural: this is a diagnostic table, deliberately kept out of
        `trends.parquet`, so it must carry its own provenance."""
        out = R.rarefied_effective_topics(self._panel(register_panel), n_draws=13, seed=8)

        assert len(out) == 2                       # 1 taxonomy x 2 eras
        assert set(out["taxonomy"]) == {"llm_level2"}
        assert set(out["n_draws"]) == {13}
        assert set(out["seed"]) == {8}


# =========================================================================== #
class TestEraCounts:
    def test_counts_speeches_paragraphs_words_and_the_annual_message_share(
        self, register_panel
    ):
        """Published beside every trend because the 1770 era holds 28 speeches:
        no interval, however narrow, makes that a confident estimate."""
        rows = [
            {"doc_name": "a", "year": 1805, "speech_type": SOTU,
             "n_paragraphs": 10.0, "para_words": 1_000.0},
            {"doc_name": "b", "year": 1806, "speech_type": SOTU,
             "n_paragraphs": 5.0, "para_words": 500.0},
            {"doc_name": "c", "year": 1807, "speech_type": INAUGURAL,
             "n_paragraphs": 2.0, "para_words": 200.0},
            {"doc_name": "d", "year": 1835, "speech_type": REMARKS,
             "n_paragraphs": 8.0, "para_words": 800.0},
        ]

        out = R.era_counts(register_panel(rows)).set_index("era")

        assert out.loc[1800, "n_speeches"] == 3
        assert out.loc[1800, "n_sotu_speeches"] == 2
        assert out.loc[1800, "n_paragraphs"] == 17
        assert out.loc[1800, "n_words"] == 1_700
        assert out.loc[1800, "n_sotu_paragraphs"] == 15
        assert out.loc[1830, "n_speeches"] == 1
        assert out.loc[1830, "n_sotu_speeches"] == 0

    def test_counts_presidents_and_the_largest_president_s_paragraph_share(
        self, register_panel
    ):
        """Every claim in the note is a claim about PRESIDENTS while the bootstrap
        resamples speeches, so an era's president count and its concentration are
        published beside its speech count. The 2010 era is 3 presidents with 51%
        of its paragraphs from one of them; without these columns a reader sees
        121 speeches and infers 121 draws' worth of evidence about presidents.

        Two speeches from one president and one from another, with the lopsided
        president holding 30 of the era's 40 paragraphs, so `nunique` and the
        share are both wrong-answerable by the obvious mistakes (counting
        speeches, or taking the max over speeches instead of over presidents).
        """
        rows = [
            {"doc_name": "a", "year": 1805, "president": "X", "speech_type": SOTU,
             "n_paragraphs": 20.0, "para_words": 2_000.0},
            {"doc_name": "b", "year": 1806, "president": "X", "speech_type": SOTU,
             "n_paragraphs": 10.0, "para_words": 1_000.0},
            {"doc_name": "c", "year": 1807, "president": "Y", "speech_type": INAUGURAL,
             "n_paragraphs": 10.0, "para_words": 1_000.0},
        ]

        out = R.era_counts(register_panel(rows)).set_index("era")

        assert out.loc[1800, "n_speeches"] == 3
        assert out.loc[1800, "n_presidents"] == 2
        # X holds 30 of 40 paragraphs across two speeches; the largest SPEECH is
        # only 20 of 40, so a per-speech max would report 0.5.
        assert out.loc[1800, "top_president_paragraph_share"] == pytest.approx(0.75)

    def test_a_single_president_era_reports_a_share_of_one(self, register_panel):
        """The degenerate end of the concentration column, and the one that
        matters: an era whose paragraphs are all one president's must read 1.0,
        not NaN, because that is precisely the era whose interval is least
        trustworthy as evidence about presidents."""
        rows = [
            {"doc_name": "a", "year": 1985, "president": "Z", "speech_type": SOTU,
             "n_paragraphs": 8.0, "para_words": 800.0},
            {"doc_name": "b", "year": 1986, "president": "Z", "speech_type": REMARKS,
             "n_paragraphs": 2.0, "para_words": 200.0},
        ]

        out = R.era_counts(register_panel(rows)).set_index("era")

        assert out.loc[1980, "n_presidents"] == 1
        assert out.loc[1980, "top_president_paragraph_share"] == pytest.approx(1.0)

    def test_an_era_with_no_annual_message_reports_zero_not_missing(
        self, register_panel
    ):
        """The reindex-with-fill path: an era whose speeches are all press
        conferences must show 0 annual-message paragraphs, not NaN, or the column
        turns to floats and the era drops out of any integer comparison."""
        rows = [{
            "doc_name": "d", "year": 1985, "speech_type": REMARKS,
            "n_paragraphs": 8.0, "para_words": 800.0,
        }]

        out = R.era_counts(register_panel(rows)).set_index("era")

        assert out.loc[1980, "n_sotu_paragraphs"] == 0
        assert out.loc[1980, "n_sotu_speeches"] == 0


# =========================================================================== #
class TestWithinSpeechIcc:
    """The number that justifies clustering the bootstrap on speeches.

    It exists as shipped code because the note used to quote an ICC of 0.119
    inherited from a prior session's scratch work, with nothing in the repo able
    to reproduce it. Every assertion below is hand-computable from its fixture,
    so the function pins a definition rather than a remembered value.
    """

    @staticmethod
    def _frame(values_by_speech):
        rows = [
            {"doc_name": doc, "x": float(v)}
            for doc, values in values_by_speech.items()
            for v in values
        ]
        return pd.DataFrame(rows)

    def test_identical_paragraphs_within_each_speech_is_an_icc_of_one(self):
        """The pathological case the bootstrap unit exists for: every paragraph
        in a speech carries the same value, so a speech contributes exactly one
        independent observation and a paragraph-level bootstrap would claim six.
        Within-group variance is 0, so the ICC must be exactly 1."""
        frame = self._frame({"a": [1, 1, 1], "b": [5, 5, 5]})

        out = R.within_speech_icc(frame, ["x"]).iloc[0]

        assert out["icc"] == pytest.approx(1.0)
        assert out["n_speeches"] == 2
        assert out["n_paragraphs"] == 6

    def test_speeches_with_identical_means_give_a_negative_or_zero_icc(self):
        """The opposite pole. With every group mean equal, between-group variance
        is 0, `MSB - MSW` is negative and the ICC lands at or below 0 — "no more
        alike inside a speech than across speeches", which is the only world in
        which a paragraph-level bootstrap would be honest. Asserted rather than
        clipped, so a future clamp to [0, 1] has to be a deliberate change."""
        frame = self._frame({"a": [0, 10], "b": [0, 10], "c": [0, 10]})

        assert R.within_speech_icc(frame, ["x"]).iloc[0]["icc"] <= 0.0

    def test_matches_the_one_way_anova_formula_on_unequal_group_sizes(self):
        """The `k0` correction is the whole reason this is not the textbook
        equal-`n` formula: real speeches run from 1 to several hundred paragraphs.
        The expected value is written out from the ANOVA definition, so numerator
        and denominator are each pinned on their own."""
        frame = self._frame({"a": [1, 3], "b": [5, 7, 9], "c": [2]})
        values = frame["x"].to_numpy()
        sizes = np.array([2.0, 3.0, 1.0])
        means = np.array([2.0, 7.0, 2.0])
        n, k = 6.0, 3
        ss_between = float((sizes * (means - values.mean()) ** 2).sum())
        ss_within = float(((values - np.repeat(means, sizes.astype(int))) ** 2).sum())
        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (n - k)
        k0 = (n - (sizes**2).sum() / n) / (k - 1)

        out = R.within_speech_icc(frame, ["x"]).iloc[0]

        assert out["icc"] == pytest.approx(
            (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)
        )

    def test_scores_every_requested_column_and_defaults_to_the_label_columns(self):
        """Booleans have to go through the same path as counts — `llm_zero` is
        the highest-ICC quantity in the corpus and is a bool."""
        frame = self._frame({"a": [1, 1], "b": [4, 4]})
        frame["flag"] = [True, True, False, False]

        out = R.within_speech_icc(frame, ["x", "flag"])

        assert list(out["column"]) == ["x", "flag"]
        assert out["icc"].tolist() == pytest.approx([1.0, 1.0])
        assert R.ICC_COLUMNS[0] == "n_legacy"

    def test_refuses_a_frame_with_one_paragraph_per_speech(self):
        """With one paragraph per speech there is no within-group variation to
        measure and `MSW` divides by zero. Refusing beats publishing a `nan` or,
        worse, a confident 1.0 that would appear to justify the clustering."""
        frame = self._frame({"a": [1], "b": [2]})

        with pytest.raises(ValueError, match="more paragraphs than speeches"):
            R.within_speech_icc(frame, ["x"])


# =========================================================================== #
class TestWriteTrends:
    def test_creates_the_directory_and_round_trips_the_frame(
        self, tmp_path, monkeypatch, register_panel
    ):
        register_dir = tmp_path / "register"
        monkeypatch.setattr(R, "REGISTER_DIR", register_dir)
        monkeypatch.setattr(R, "TRENDS_PATH", register_dir / "trends.parquet")
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        path = R.write_trends(trends)

        assert path.exists()
        pd.testing.assert_frame_equal(pd.read_parquet(path), trends)
        # The file is written without the pandas index. A round trip cannot see
        # the difference — pandas stores the RangeIndex and restores it as an
        # index, not a column — but the on-disk SCHEMA gains an
        # `__index_level_0__` field, which changes the bytes of the committed
        # artifact and hands every non-pandas reader a phantom column.
        assert list(pq.read_schema(path).names) == R.TREND_COLUMNS

    def test_nullable_integer_columns_survive_the_round_trip_as_na(
        self, tmp_path, monkeypatch, register_panel
    ):
        """The trend rows carry NA periods and sample sizes. If those columns
        came back as floats or as 0, a consumer filtering `period.isna()` to find
        the trend rows would get the wrong set."""
        register_dir = tmp_path / "register"
        monkeypatch.setattr(R, "REGISTER_DIR", register_dir)
        monkeypatch.setattr(R, "TRENDS_PATH", register_dir / "trends.parquet")
        trends = R.build_trends(_trend_panel(register_panel), n_bootstrap=5, seed=1)

        reloaded = pd.read_parquet(R.write_trends(trends))

        assert str(reloaded["period"].dtype) == "Int64"
        assert reloaded.loc[reloaded["unit"] == "trend", "period"].isna().all()
        assert reloaded.loc[reloaded["unit"] == "era", "period"].notna().all()


# =========================================================================== #
def _cli_corpus(register_corpus):
    """A synthetic corpus spanning every reference genre in two eras, thick
    enough that the year rows clear `MIN_YEAR_PARAGRAPHS`."""
    specs = []
    for era in (1800, 1830):
        for genre in (INAUGURAL, SPECIAL, SOTU, VETO):
            for i in range(3):
                specs.append({
                    "doc_name": f"{era}-{genre}-{i}",
                    "year": era + 5,
                    "speech_type": genre,
                    "paras": [
                        {"legacy": ["War & military"], "topics": ["Jobs & Wages"],
                         "pv": "proposal", "words": 100},
                        {"legacy": [], "topics": ["Holidays & Tributes"],
                         "pv": "values", "words": 80},
                        {"legacy": ["Trade & tariffs"], "topics": ["the War on Terror"],
                         "pv": "mixed", "words": 90},
                        {"legacy": [], "topics": [], "pv": "neither", "words": 60},
                    ],
                })
    return register_corpus(specs)


class TestMain:
    def _install(self, monkeypatch, tmp_path, register_corpus, register_taxonomy):
        corpus = _cli_corpus(register_corpus)
        index = R.build_taxonomy_index(register_taxonomy)
        inputs = R.RegisterInputs(
            paragraphs=corpus.paragraphs,
            issues=corpus.issues,
            annotations=corpus.annotations,
            speeches=corpus.speeches,
            speech_annotations=corpus.speech_annotations,
            stats=corpus.stats,
            markers=corpus.markers,
            taxonomy=index,
        )
        monkeypatch.setattr(R, "load_inputs", lambda: inputs)
        register_dir = tmp_path / "register"
        monkeypatch.setattr(R, "REGISTER_DIR", register_dir)
        monkeypatch.setattr(R, "TRENDS_PATH", register_dir / "trends.parquet")
        return register_dir / "trends.parquet"

    def test_runs_the_whole_pipeline_and_writes_the_table(
        self, monkeypatch, tmp_path, register_corpus, register_taxonomy, capsys
    ):
        """The only test that exercises load -> paragraph frame -> panel ->
        trends -> write as one wiring. Everything upstream of `load_inputs` is
        faked; nothing touches the real data directory."""
        path = self._install(monkeypatch, tmp_path, register_corpus, register_taxonomy)
        monkeypatch.setattr(
            "sys.argv", [ARGV0, "--n-bootstrap", "5", "--seed", "77"]
        )

        assert R.main() == 0

        trends = pd.read_parquet(path)
        assert set(trends["bootstrap_seed"]) == {77}
        assert set(trends["n_bootstrap"]) == {5}
        assert set(trends["unit"]) == {"era", "year", "trend"}
        assert f"wrote {len(trends):,} rows" in capsys.readouterr().out

    def test_defaults_come_from_the_module_constants(self, monkeypatch):
        """The CLI's defaults are what produced the committed artifact, so they
        are checked without paying for 2,000 replicates: `build_trends` is
        replaced by a spy that records what it was handed."""
        seen = {}

        def _spy(panel, n_bootstrap, seed):
            seen.update(n_bootstrap=n_bootstrap, seed=seed)
            return pd.DataFrame({"unit": ["era"]})

        monkeypatch.setattr(
            R, "load_inputs",
            lambda: types.SimpleNamespace(
                paragraphs=None, issues=None, annotations=None, speeches=None,
                speech_annotations=None, stats=None, markers=None, taxonomy=None,
            ),
        )
        monkeypatch.setattr(R, "build_paragraph_frame", lambda *a: "para")
        monkeypatch.setattr(R, "build_speech_panel", lambda *a: "panel")
        monkeypatch.setattr(R, "build_trends", _spy)
        monkeypatch.setattr(R, "write_trends", lambda trends: "path")
        monkeypatch.setattr("sys.argv", [ARGV0])

        assert R.main() == 0
        assert seen == {"n_bootstrap": R.N_BOOTSTRAP, "seed": R.BOOTSTRAP_SEED}

    def test_a_broken_input_stops_the_run_rather_than_writing_a_short_table(
        self, monkeypatch, tmp_path, register_corpus, register_taxonomy
    ):
        """The merge guards have to survive the wiring: a corpus whose annotation
        table is missing a paragraph must abort `main`, not write a trends table
        computed over a silently smaller corpus."""
        path = self._install(monkeypatch, tmp_path, register_corpus, register_taxonomy)
        inputs = R.load_inputs()
        broken = R.RegisterInputs(
            **{
                **{f: getattr(inputs, f) for f in inputs.__dataclass_fields__},
                "annotations": inputs.annotations.iloc[1:].reset_index(drop=True),
            }
        )
        monkeypatch.setattr(R, "load_inputs", lambda: broken)
        monkeypatch.setattr("sys.argv", [ARGV0, "--n-bootstrap", "2"])

        with pytest.raises(ValueError, match="row count"):
            R.main()
        assert not path.exists()
