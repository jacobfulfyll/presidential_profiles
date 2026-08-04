"""Read-only anchors for every tabulated figure in `notes/register-findings-v1.md`.

The rest of the suite proves that `register.py` computes what it says it computes,
on synthetic frames with hand-checkable answers. **Nothing in it asserts the
CONTENTS of the real `data/register/trends.parquet`**, so a change that produced
plausible-but-different real numbers — a different reference mix, a dropped
case-variant, a bootstrap reseeded, an era rebanded — would pass every other test
while silently falsifying a published note.

This file closes that hole. It is the promoted form of a scratchpad harness that
checked the note claim by claim; living in `tests/` is the point, because a
scratch script only runs when someone remembers it. Every assertion here is a
*claim made in the note*, quoted to the precision the note prints, so a failure
reads as "the note now says something the data does not" rather than as a
numerical regression.

**Structural limit, and the reason it matters.** Every assertion here is keyed to a
figure the note *prints*, so this file can only ever catch a printed number that is
wrong. It is constitutionally incapable of catching a number the note **omits** — a
`(measure, taxonomy, genre_treatment, statistic)` cell that `build_trends` computed,
that bears on a claim, and that appears nowhere in the prose. Nine such omissions
reached the note by reading, not by this file, every one of them while the suite
was fully green. **A green run here means "nothing printed is wrong", never
"nothing is missing".** `TestCompletenessOfReporting` below closes as much of that
hole as can be mechanised: it pins the arm counts of the contrasts the note claims
to report in full, and
`test_every_significant_unprinted_cell_falls_inside_a_declared_scope_rule` sweeps
all 444 cells against the prose and requires every significant leftover to sit
inside one of the note's four declared scope rules.

Two consequences worth stating:

- **This file is expected to fail loudly if the parquet is legitimately
  regenerated with different inputs.** That is the contract, not a nuisance. The
  fix is to re-check the note against the new numbers and update both together —
  which is exactly the step this file exists to force. It mirrors the
  `TestCommittedArtifacts` pattern in `test_register_taxonomy_index.py`.
- **It reads real data by absolute worktree path**, never through the module
  constants, because conftest's autouse `redirect_annotation_dirs` repoints
  `ANNOTATIONS_DIR` at `tmp_path`. Both fixtures are module-scoped: the whole
  real pipeline (load, paragraph frame, panel, ICC, rarefaction) runs once, in
  about two seconds, for the entire file.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import register as R

_TRENDS = Path(__file__).resolve().parents[1] / "data" / "register" / "trends.parquet"

_T = {"raw": "raw", "sotu": "sotu_only", "std": "genre_standardized"}
_TAXONOMIES = ("legacy15", "llm_level2", "llm_level1")
_ERAS = (1770, 1800, 1830, 1860, 1890, 1920, 1950, 1980, 2010)


@pytest.fixture(scope="module")
def trends() -> pd.DataFrame:
    """The committed `trends.parquet`, read once for the whole file."""
    return pd.read_parquet(_TRENDS)


@pytest.fixture(scope="module")
def derived() -> SimpleNamespace:
    """The note's figures that live in code rather than in the parquet.

    The rarefaction table, the ICC table, the era counts, the label-resolution
    report and the reference genre mix are all cited by the note and none of them
    ships in `trends.parquet` — `rarefied_effective_topics` and `within_speech_icc`
    are deliberately kept out of it (their spread is not sampling uncertainty).
    An uncited number is the defect this file exists to prevent, so they are
    recomputed here rather than trusted.
    """
    inputs = R.load_inputs()
    para = R.build_paragraph_frame(
        inputs.paragraphs, inputs.issues, inputs.annotations, inputs.taxonomy
    )
    panel = R.build_speech_panel(
        para, inputs.speeches, inputs.speech_annotations, inputs.stats,
        inputs.markers, inputs.taxonomy,
    )
    return SimpleNamespace(
        para=para,
        panel=panel,
        counts=R.era_counts(panel).set_index("era"),
        icc=R.within_speech_icc(para).set_index("column")["icc"],
        per_issue_icc=R.within_speech_icc(para, list(R.LEGACY_ISSUES))["icc"],
        rarefaction=R.rarefied_effective_topics(panel),
        report=R.label_resolution_report(inputs.annotations, inputs.taxonomy),
        mix=R.reference_genre_mix(panel.speeches, panel.scalars["n_paragraphs"]),
        cells=R.genre_cells(panel.speeches),
    )


# --------------------------------------------------------------------------
# query helpers — the note's own vocabulary, so a failure names a note claim
# --------------------------------------------------------------------------


def _level(trends: pd.DataFrame, measure: str, taxonomy: str, treatment: str) -> pd.Series:
    """The nine era levels of one measure/taxonomy/treatment arm, indexed by era."""
    rows = trends[
        (trends["unit"] == "era")
        & (trends["measure"] == measure)
        & (trends["taxonomy"] == taxonomy)
        & (trends["genre_treatment"] == treatment)
        & (trends["statistic"] == "level")
    ]
    return rows.set_index("period")["value"].sort_index()


def _stat(
    trends: pd.DataFrame,
    measure: str,
    taxonomy: str,
    treatment: str,
    statistic: str,
    field: str = "value",
) -> float:
    rows = trends[
        (trends["unit"] == "trend")
        & (trends["measure"] == measure)
        & (trends["taxonomy"] == taxonomy)
        & (trends["genre_treatment"] == treatment)
        & (trends["statistic"] == statistic)
    ]
    assert len(rows) == 1, f"expected one {measure}/{taxonomy}/{treatment}/{statistic}"
    return float(rows.iloc[0][field])


def _close(actual: float, claimed: float, dp: int) -> bool:
    """The note prints `claimed` at `dp` decimals; does `actual` round to it?"""
    return abs(round(float(actual), dp) - claimed) < 10 ** (-dp) / 2 + 1e-12


def _assert_printed(actual: float, claimed: float, dp: int, label: str) -> None:
    assert _close(actual, claimed, dp), (
        f"{label}: the note prints {claimed}, the parquet has {actual:.6f} "
        f"(rounds to {round(float(actual), dp)})"
    )


def _renderings(value: float) -> list[str]:
    """Every way the note could print `value`, for the completeness sweep.

    Never 0 decimals, and never fewer than two significant digits: a rendering
    like "1" or "8" matches almost any region of prose, which is how an earlier
    sweep lost cells. Module-level so the rule itself can be tested.
    """
    out, size = [], abs(float(value))
    for dp in (1, 2, 3, 4):                           # never 0 dp
        text = f"{size:.{dp}f}"
        if len(text.replace(".", "").lstrip("0")) >= 2:
            out.append(text)
    if size < 1.5:                                    # shares are quoted as %
        for dp in (1, 2, 3):
            text = f"{size * 100:.{dp}f}"
            if len(text.replace(".", "").lstrip("0")) >= 2:
                out.append(text)
    return out


def test_renderings_never_matches_on_fewer_than_two_significant_digits() -> None:
    """The 0-dp rule, pinned so a future edit cannot silently re-enable it."""
    assert "1" not in _renderings(0.7833) and "1" not in _renderings(1.0)
    assert "0.8" not in _renderings(0.7833), "one significant digit"
    assert all(len(t.replace(".", "").lstrip("0")) >= 2 for t in _renderings(0.7833))
    assert _renderings(0.0) == []
    assert "0.78" in _renderings(0.7833) and "78.3" in _renderings(0.7833)


# =========================================================================== #
class TestShapeAndSchema:
    """The parquet the note advertises, in the shape the note advertises."""

    def test_row_count_seed_and_replicate_count(self, trends):
        assert len(trends) == 16_243
        assert set(trends["bootstrap_seed"]) == {20260721}
        assert set(trends["n_bootstrap"]) == {2_000}

    def test_declared_columns_and_domains(self, trends):
        assert list(trends.columns) == R.TREND_COLUMNS
        assert set(trends["unit"]) == {"year", "era", "trend"}
        assert set(trends["taxonomy"]) == {"legacy15", "llm_level2", "llm_level1", "none"}
        assert set(trends["statistic"]) == {"level", *R.TREND_STATISTICS}

    def test_thirty_seven_measure_taxonomy_columns_including_neither_share(self, trends):
        assert trends.groupby(["measure", "taxonomy"]).ngroups == 37
        assert "neither_share" in set(trends["measure"])


# =========================================================================== #
class TestMultiplicityExposure:
    """The note's "444 tests, uncorrected" disclosure, anchored.

    These four counts are the whole basis of the multiplicity section. If the
    table grows a measure or a statistic they change, and the section's arithmetic
    (0.05/444, "about 22 by chance") has to be rewritten rather than left stale.
    """

    def test_the_note_reports_the_right_number_of_hypothesis_tests(self, trends):
        tests = trends[trends["unit"] == "trend"]
        assert len(tests) == 444
        assert int(np.isfinite(tests["p_value"]).sum()) == 444
        # 37 measure-taxonomy columns x 3 treatments x 4 statistics.
        assert 37 * len(R.GENRE_TREATMENTS) * len(R.TREND_STATISTICS) == 444

    def test_the_p_value_distribution_the_note_describes(self, trends):
        p = trends.loc[trends["unit"] == "trend", "p_value"].to_numpy(dtype=float)
        assert int((p < 0.05).sum()) == 344
        assert int((p == 0.0005).sum()) == 277, "tests at the 1/B resolution floor"
        assert int(((p >= 0.004) & (p < 0.05)).sum()) == 50, "the band a correction bites"
        # The three bands have to account for every significant test: an earlier
        # draft named 277 and 50 and left 17 in (0.0005, 0.004) unaccounted for.
        assert int(((p > 0.0005) & (p < 0.004)).sum()) == 17
        assert 277 + 50 + 17 == int((p < 0.05).sum())

    def test_no_test_can_clear_a_bonferroni_threshold_at_this_replicate_count(
        self, trends
    ):
        """The hard limit the note states: 1/B = 5e-4 is coarser than 0.05/444.

        This is why the note says the correction is unresolvable rather than
        claiming a set of tests survives it.
        """
        cells = trends[trends["unit"] == "trend"]
        p = cells["p_value"].to_numpy(dtype=float)
        threshold = 0.05 / 444
        # `bootstrap_p_value` floors at 1/n_finite_replicates, NOT literally 1/B.
        # The two coincide only while every trend row keeps all B draws. If an
        # estimator change ever dropped replicates, the note's "1/2000" arithmetic
        # and this file's `p == 0.0005` assertions would go wrong together and
        # quietly, so the note's floor is tied to the realised count here.
        artifact_draws = int(cells["n_bootstrap"].dropna().iloc[0])
        assert set(cells["n_bootstrap_valid"]) == {artifact_draws}, (
            "the note's 1/B floor is only the real floor while every trend row "
            "has all B finite replicates"
        )
        assert threshold < 1.0 / artifact_draws
        assert int((p < threshold).sum()) == 0

    _MARGINAL_TABLE = [
        ("effective_topics", "llm_level1", "sotu_only", "delta_modern_minus_postbellum", 0.004),
        ("effective_topics", "llm_level1", "genre_standardized", "delta_modern_minus_postbellum", 0.009),
        ("non_policy_share", "llm_level2", "raw", "delta_modern_minus_early", 0.009),
        ("non_policy_share", "llm_level2", "sotu_only", "delta_modern_minus_early", 0.014),
        ("non_policy_share", "llm_level2", "genre_standardized", "delta_modern_minus_early", 0.009),
        ("non_policy_share", "llm_level2", "sotu_only", "spearman_vs_era", 0.040),
        ("effective_topics", "llm_level2", "sotu_only", "delta_modern_minus_postbellum", 0.042),
        ("religiosity", "none", "sotu_only", "delta_modern_minus_early", 0.002),
        ("nrc_fear", "none", "sotu_only", "delta_modern_minus_early", 0.009),
        ("nrc_hope", "none", "genre_standardized", "delta_modern_minus_early", 0.021),
        ("effective_topics", "llm_level1", "sotu_only", "spearman_vs_era", 0.008),
        ("effective_topics", "llm_level1", "genre_standardized", "spearman_vs_era", 0.037),
        ("us_them", "none", "raw", "delta_modern_minus_early", 0.003),
        ("labels_per_paragraph", "legacy15", "raw", "delta_modern_minus_early", 0.001),
        ("nrc_fear", "none", "raw", "delta_modern_minus_postbellum", 0.001),
    ]

    @pytest.mark.parametrize(
        "measure, taxonomy, treatment, statistic, p", _MARGINAL_TABLE
    )
    def test_every_row_of_the_marginal_table(
        self, trends, measure, taxonomy, treatment, statistic, p
    ):
        _assert_printed(
            _stat(trends, measure, taxonomy, treatment, statistic, "p_value"),
            p, 4, f"{measure}/{taxonomy}/{treatment}/{statistic} p",
        )

    def test_the_marginal_band_is_stated_as_the_table_actually_spans(self, trends):
        """The note said "[0.002, 0.042]" while its own last row was 0.001."""
        p = [
            _stat(trends, m, tax, t, s, "p_value")
            for m, tax, t, s, _ in self._MARGINAL_TABLE
        ]
        assert min(p) == pytest.approx(0.001)
        assert max(p) == pytest.approx(0.042)

    _FLOOR_TABLE_ADDITIONS = [
        ("values_share", "none", "raw"),
        ("values_share", "none", "sotu_only"),
        ("values_share", "none", "genre_standardized"),
        ("non_policy_share", "legacy15", "raw"),
        ("non_policy_share", "llm_level2", "raw"),
        ("non_policy_share", "llm_level2", "sotu_only"),
        ("non_policy_share", "llm_level2", "genre_standardized"),
        ("labels_per_paragraph", "legacy15", "raw"),
        ("labels_per_paragraph", "llm_level2", "sotu_only"),
        ("labels_per_paragraph", "llm_level2", "genre_standardized"),
    ]

    @pytest.mark.parametrize("measure, taxonomy, treatment", _FLOOR_TABLE_ADDITIONS)
    def test_every_post_1860_row_added_to_the_floor_table_is_at_the_floor(
        self, trends, measure, taxonomy, treatment
    ):
        """The rows the note moved into "would not be excluded" — including the two
        that argue AGAINST its verdicts and the one that argues FOR the headline."""
        assert _stat(trends, measure, taxonomy, treatment,
                     "delta_modern_minus_postbellum", "p_value") == 0.0005


# =========================================================================== #
class TestSampleSizes:
    """The era table, including the president columns added after review."""

    @pytest.mark.parametrize(
        "era, speeches, presidents, paragraphs, words, sotu_speeches, sotu_paragraphs",
        [
            (1770, 28, 2, 355, 44_493, 11, 185),
            (1800, 68, 6, 1_381, 175_727, 30, 1_018),
            (1830, 113, 10, 4_492, 610_490, 30, 2_600),
            (1860, 128, 9, 3_992, 497_246, 30, 2_581),
            (1890, 112, 7, 5_064, 635_951, 30, 3_707),
            (1920, 122, 6, 2_828, 351_606, 16, 804),
            (1950, 191, 8, 6_635, 678_053, 26, 1_560),
            (1980, 174, 6, 5_870, 637_581, 28, 1_441),
            (2010, 121, 3, 5_612, 548_119, 17, 1_295),
        ],
    )
    def test_era_counts_row(
        self, derived, era, speeches, presidents, paragraphs, words,
        sotu_speeches, sotu_paragraphs,
    ):
        row = derived.counts.loc[era]
        assert int(row["n_speeches"]) == speeches
        assert int(row["n_presidents"]) == presidents
        assert int(row["n_paragraphs"]) == paragraphs
        assert int(row["n_words"]) == words
        assert int(row["n_sotu_speeches"]) == sotu_speeches
        assert int(row["n_sotu_paragraphs"]) == sotu_paragraphs

    @pytest.mark.parametrize(
        "era, share",
        [(1770, 71.3), (1800, 29.5), (1830, 23.4), (1860, 20.4), (1890, 30.2),
         (1920, 38.3), (1950, 40.9), (1980, 33.6), (2010, 50.8)],
    )
    def test_largest_president_paragraph_share(self, derived, era, share):
        """The endpoint eras are 71% Washington and 51% Trump. The note leads with
        those two figures, so they are pinned rather than described."""
        _assert_printed(
            100 * derived.counts.loc[era, "top_president_paragraph_share"],
            share, 1, f"era {era} top-president share",
        )

    def test_the_era_rows_still_sum_to_the_whole_corpus(self, derived):
        """A dropped era or a silently shrunk merge shows up here first."""
        assert int(derived.counts["n_speeches"].sum()) == 1_057
        assert int(derived.counts["n_paragraphs"].sum()) == 36_229
        assert int(derived.counts["n_words"].sum()) == 4_179_266
        assert list(derived.counts.index) == list(_ERAS)

    def test_the_block_contrasts_pool_presidents_the_note_claims(self, derived):
        """14 modern presidents against 16 early with NO overlap, and against 18
        postbellum sharing only Truman. This is the note's argument for preferring
        the block contrasts over the endpoints."""
        speeches = derived.panel.speeches

        def presidents(eras):
            return set(speeches.loc[speeches["era"].isin(eras), "president"])

        early, postbellum, modern = (
            presidents(R.EARLY_ERAS), presidents(R.POSTBELLUM_ERAS),
            presidents(R.MODERN_ERAS),
        )
        assert (len(early), len(postbellum), len(modern)) == (16, 18, 14)
        assert early & modern == set()
        assert postbellum & modern == {"Harry S. Truman"}


# =========================================================================== #
class TestArmDependence:
    """M1: `sotu_only` and `genre_standardized` are not independent controls."""

    def test_reference_mix_is_three_quarters_annual_messages(self, derived):
        for genre, want in [
            (R.SOTU_TYPE, 0.744),
            ("special_message_to_congress", 0.149),
            ("inaugural_address", 0.055),
            ("veto_or_signing_statement", 0.052),
        ]:
            _assert_printed(derived.mix[genre], want, 3, f"reference mix {genre}")
        _assert_printed(100 * derived.mix[R.SOTU_TYPE], 74.4, 1, "corpus-wide overlap")

    @pytest.mark.parametrize("era", [1770, 2010])
    def test_the_endpoint_eras_collapse_to_a_93_percent_sotu_mix(self, derived, era):
        """Thin cells drop special messages and vetoes at both endpoints, so
        `genre_standardized` there is 93.1% the same paragraphs as `sotu_only`."""
        kept = derived.cells[
            (derived.cells["era"] == era) & derived.cells["included"]
        ]["speech_type"].tolist()

        assert set(kept) == {"inaugural_address", R.SOTU_TYPE}
        weights = derived.mix[kept] / derived.mix[kept].sum()
        _assert_printed(100 * weights[R.SOTU_TYPE], 93.1, 1, f"era {era} SOTU weight")


# =========================================================================== #
class TestSubClaim1Breadth:
    """"Presidents touch more subjects" — survives, but pre-1860."""

    @pytest.mark.parametrize(
        "taxonomy, v1770, v1860, v2010, spearman, ci_low, ci_high, p",
        [
            ("legacy15", 9.50, 11.25, 11.68, 0.817, 0.683, 0.967, 0.0005),
            ("llm_level2", 14.42, 24.92, 26.98, 0.883, 0.733, 0.983, 0.0005),
            ("llm_level1", 9.47, 10.42, 11.89, 0.567, 0.200, 0.750, 0.0010),
        ],
    )
    def test_taxonomy_fit_table_raw(
        self, trends, taxonomy, v1770, v1860, v2010, spearman, ci_low, ci_high, p
    ):
        levels = _level(trends, "effective_topics", taxonomy, "raw")
        for era, want in ((1770, v1770), (1860, v1860), (2010, v2010)):
            _assert_printed(levels.loc[era], want, 2, f"{taxonomy} raw {era}")
        for field, want in (
            ("value", spearman), ("ci_low", ci_low), ("ci_high", ci_high), ("p_value", p)
        ):
            _assert_printed(
                _stat(trends, "effective_topics", taxonomy, "raw", "spearman_vs_era", field),
                want, 4 if field == "p_value" else 3, f"{taxonomy} raw spearman {field}",
            )

    @pytest.mark.parametrize(
        "taxonomy, sotu, std",
        [("legacy15", 0.733, 0.883), ("llm_level2", 0.833, 0.783), ("llm_level1", 0.417, 0.483)],
    )
    def test_spearman_stays_positive_under_both_genre_controls(
        self, trends, taxonomy, sotu, std
    ):
        for treatment, want in (("sotu_only", sotu), ("genre_standardized", std)):
            _assert_printed(
                _stat(trends, "effective_topics", taxonomy, treatment, "spearman_vs_era"),
                want, 3, f"{taxonomy} {treatment} spearman",
            )

    def test_all_controlled_spearman_p_values_are_at_or_below_the_quoted_bound(
        self, trends
    ):
        worst = max(
            _stat(trends, "effective_topics", tax, tr, "spearman_vs_era", "p_value")
            for tax in _TAXONOMIES
            for tr in ("sotu_only", "genre_standardized")
        )
        assert worst <= 0.037

    @pytest.mark.parametrize(
        "taxonomy, arm, mod_early, p_early, mod_postbellum, p_postbellum",
        [
            ("legacy15", "raw", 1.97, 0.0005, 0.22, 0.211),
            ("legacy15", "sotu", 2.44, 0.0005, 0.13, 0.506),
            ("legacy15", "std", 2.35, 0.0005, 0.21, 0.300),
            ("llm_level2", "raw", 8.33, 0.0005, 0.46, 0.309),
            ("llm_level2", "sotu", 8.62, 0.0005, 1.20, 0.042),
            ("llm_level2", "std", 7.75, 0.0005, 0.11, 0.711),
            ("llm_level1", "raw", 0.82, 0.0100, -0.62, 0.061),
            ("llm_level1", "sotu", 1.33, 0.0005, -0.95, 0.004),
            ("llm_level1", "std", 0.77, 0.0330, -0.90, 0.009),
        ],
    )
    def test_block_contrast_table(
        self, trends, taxonomy, arm, mod_early, p_early, mod_postbellum, p_postbellum
    ):
        treatment = _T[arm]
        for statistic, value, p in (
            ("delta_modern_minus_early", mod_early, p_early),
            ("delta_modern_minus_postbellum", mod_postbellum, p_postbellum),
        ):
            _assert_printed(
                _stat(trends, "effective_topics", taxonomy, treatment, statistic),
                value, 2, f"{taxonomy}/{arm} {statistic}",
            )
            _assert_printed(
                _stat(trends, "effective_topics", taxonomy, treatment, statistic, "p_value"),
                p, 4, f"{taxonomy}/{arm} {statistic} p",
            )

    # ---------------------------------------------------------------- B2
    def test_the_2010_era_ranks_first_on_four_arms_and_top_two_on_eight(self, trends):
        """The disclosure the note withheld for a draft: the post-1860 verdict is
        published beside the fact that the modern era is the corpus maximum on 4
        of 9 taxonomy x treatment arms and top-two on 8 of 9. Both are true; they
        answer different questions."""
        wide = pd.DataFrame({
            (tax, tr): _level(trends, "effective_topics", tax, tr)
            for tax in _TAXONOMIES
            for tr in R.GENRE_TREATMENTS
        })
        ranks = wide.rank(ascending=False, method="min").astype(int).loc[2010]

        assert int((ranks == 1).sum()) == 4
        assert int((ranks <= 2).sum()) == 8
        assert len(ranks) == 9

    @pytest.mark.parametrize(
        "taxonomy, treatment, argmax_era",
        [
            ("legacy15", "raw", 2010), ("legacy15", "sotu_only", 1920),
            ("legacy15", "genre_standardized", 2010),
            ("llm_level2", "raw", 2010), ("llm_level2", "sotu_only", 1950),
            ("llm_level2", "genre_standardized", 1950),
            ("llm_level1", "raw", 2010), ("llm_level1", "sotu_only", 1920),
            ("llm_level1", "genre_standardized", 1890),
        ],
    )
    def test_which_era_is_the_maximum_on_each_arm(
        self, trends, taxonomy, treatment, argmax_era
    ):
        assert int(_level(trends, "effective_topics", taxonomy, treatment).idxmax()) == argmax_era

    def test_level_1_raw_peaks_in_2010_above_1860_the_sharpest_case(self, trends):
        """The single figure that forced the reconciliation: on the arm the note
        calls "significantly falling", the 2010 era is the series maximum, and both
        numbers were already printed in the taxonomy-fit table."""
        levels = _level(trends, "effective_topics", "llm_level1", "raw")

        _assert_printed(levels.loc[2010], 11.89, 2, "level-1 raw 2010")
        _assert_printed(levels.loc[1860], 10.42, 2, "level-1 raw 1860")
        assert levels.loc[2010] == levels.max()

    @pytest.mark.parametrize(
        "treatment, d1950, d1980, d2010",
        [("raw", -1.35, -1.21, 0.71),
         ("sotu_only", -1.09, -1.47, -0.30),
         ("genre_standardized", -1.17, -1.66, 0.12)],
    )
    def test_the_level_1_post_1860_decline_is_the_1950_1980_eras(
        self, trends, treatment, d1950, d1980, d2010
    ):
        """Each modern era against the postbellum block mean. On two of three arms
        the 2010 era is ABOVE it and is pulling the contrast toward zero — the
        decline is a mid-century trough, not a modern narrowing."""
        levels = _level(trends, "effective_topics", "llm_level1", treatment)
        base = levels.loc[list(R.POSTBELLUM_ERAS)].mean()

        for era, want in ((1950, d1950), (1980, d1980), (2010, d2010)):
            _assert_printed(levels.loc[era] - base, want, 2, f"level-1 {treatment} {era}")
        # And the three of them average to the published block contrast.
        _assert_printed(
            levels.loc[list(R.MODERN_ERAS)].mean() - base,
            _stat(trends, "effective_topics", "llm_level1", treatment,
                  "delta_modern_minus_postbellum"),
            4, f"level-1 {treatment} block identity",
        )

    def test_entirely_by_1950_and_1980_is_false_on_sotu_only(self, trends):
        """"Produced entirely by the 1950 and 1980 eras" holds on raw and genre-std,
        where the 2010 era sits ABOVE the postbellum block mean. On `sotu_only` the
        2010 era is -0.30 BELOW it and supplies 10.5% of that arm's decline — and
        `sotu_only` is one of the two arms where the decline is significant."""
        contributions = {}
        for treatment in R.GENRE_TREATMENTS:
            levels = _level(trends, "effective_topics", "llm_level1", treatment)
            base = levels.loc[list(R.POSTBELLUM_ERAS)].mean()
            contributions[treatment] = {
                era: levels.loc[era] - base for era in R.MODERN_ERAS
            }

        assert contributions["raw"][2010] > 0
        assert contributions["genre_standardized"][2010] > 0
        assert contributions["sotu_only"][2010] < 0

        sotu = contributions["sotu_only"]
        _assert_printed(sotu[2010], -0.30, 2, "level-1 sotu 2010 deviation")
        _assert_printed(
            100 * sotu[2010] / sum(sotu.values()), 10.5, 1, "2010 share of sotu decline"
        )
        assert _stat(trends, "effective_topics", "llm_level1", "sotu_only",
                     "delta_modern_minus_postbellum", "p_value") < 0.05

    def test_the_top_two_legacy_raw_eras_are_indistinguishable(self, trends):
        """Why the rank table is not nine ordered findings: rank 1 and rank 2 on
        the legacy raw arm differ by 0.013 and their intervals nest."""
        rows = trends[
            (trends["unit"] == "era") & (trends["measure"] == "effective_topics")
            & (trends["taxonomy"] == "legacy15") & (trends["genre_treatment"] == "raw")
            & (trends["statistic"] == "level")
        ].set_index("period")

        _assert_printed(rows.loc[2010, "value"], 11.679, 3, "legacy raw 2010")
        _assert_printed(rows.loc[1980, "value"], 11.666, 3, "legacy raw 1980")
        assert abs(rows.loc[2010, "value"] - rows.loc[1980, "value"]) < 0.02
        assert rows.loc[2010, "ci_low"] < rows.loc[1980, "value"] < rows.loc[2010, "ci_high"]

    @pytest.mark.parametrize(
        "taxonomy, treatment, share",
        [("llm_level2", "raw", 94.5), ("legacy15", "raw", 89.0)],
    )
    def test_the_share_of_the_rise_that_happens_before_1860(
        self, trends, taxonomy, treatment, share
    ):
        levels = _level(trends, "effective_topics", taxonomy, treatment)
        early = levels.loc[list(R.EARLY_ERAS)].mean()
        postbellum = levels.loc[list(R.POSTBELLUM_ERAS)].mean()
        modern = levels.loc[list(R.MODERN_ERAS)].mean()

        _assert_printed(
            100 * (postbellum - early) / (modern - early), share, 1,
            f"{taxonomy}/{treatment} pre-1860 share",
        )

    # ------------------------------------------------------- rarefaction
    @pytest.mark.parametrize(
        "taxonomy, endpoint_full, endpoint_rare, early_full, early_rare, "
        "postbellum_full, postbellum_rare, endpoint_shrink, early_shrink",
        [
            ("legacy15", 2.18, 1.66, 1.97, 1.72, 0.22, 0.07, 23, 13),
            ("llm_level2", 12.56, 8.25, 8.33, 6.19, 0.46, 0.13, 34, 26),
            ("llm_level1", 2.42, 1.40, 0.82, 0.59, -0.62, -0.76, 42, 29),
        ],
    )
    def test_rarefaction_table_reports_the_block_contrast_the_verdict_uses(
        self, derived, taxonomy, endpoint_full, endpoint_rare, early_full, early_rare,
        postbellum_full, postbellum_rare, endpoint_shrink, early_shrink,
    ):
        """An earlier draft rarefied only the ENDPOINT contrast while citing it
        for a BLOCK conclusion. Both are tabulated now; the block columns are the
        ones the verdicts rest on, and they make the post-1860 case stronger, not
        weaker (+0.22 -> +0.07, +0.46 -> +0.13, -0.62 -> -0.76)."""
        frame = derived.rarefaction
        frame = frame[frame["taxonomy"] == taxonomy].set_index("era")
        assert int(frame["budget_paragraphs"].iloc[0]) == 355

        def contrast(series_column, late, base):
            series = frame[series_column]
            if isinstance(late, tuple):
                return series.loc[list(late)].mean() - series.loc[list(base)].mean()
            return series.loc[late] - series.loc[base]

        _assert_printed(contrast("full_sample", 2010, 1770), endpoint_full, 2, "endpoint full")
        _assert_printed(contrast("rarefied_mean", 2010, 1770), endpoint_rare, 2, "endpoint rare")
        _assert_printed(
            contrast("full_sample", R.MODERN_ERAS, R.EARLY_ERAS), early_full, 2, "block early full"
        )
        _assert_printed(
            contrast("rarefied_mean", R.MODERN_ERAS, R.EARLY_ERAS), early_rare, 2, "block early rare"
        )
        _assert_printed(
            contrast("full_sample", R.MODERN_ERAS, R.POSTBELLUM_ERAS),
            postbellum_full, 2, "block postbellum full",
        )
        _assert_printed(
            contrast("rarefied_mean", R.MODERN_ERAS, R.POSTBELLUM_ERAS),
            postbellum_rare, 2, "block postbellum rare",
        )
        for label, full, rare, want in (
            ("endpoint", contrast("full_sample", 2010, 1770),
             contrast("rarefied_mean", 2010, 1770), endpoint_shrink),
            ("block", contrast("full_sample", R.MODERN_ERAS, R.EARLY_ERAS),
             contrast("rarefied_mean", R.MODERN_ERAS, R.EARLY_ERAS), early_shrink),
        ):
            _assert_printed(100 * (1 - rare / full), want, 0, f"{label} sample-size share")

    @pytest.mark.parametrize(
        "taxonomy, share", [("legacy15", 68), ("llm_level2", 71), ("llm_level1", -22)]
    )
    def test_the_bias_share_on_the_post_hoc_contrast_is_a_different_number(
        self, derived, taxonomy, share
    ):
        """The note's 13 / 26 / 29% are `modern - early` shares and only those. On
        the post-1860 contrast the same arithmetic gives 68% / 72% / negative, so
        the percentages must say which contrast they belong to."""
        frame = derived.rarefaction
        frame = frame[frame["taxonomy"] == taxonomy].set_index("era")
        full = (frame["full_sample"].loc[list(R.MODERN_ERAS)].mean()
                - frame["full_sample"].loc[list(R.POSTBELLUM_ERAS)].mean())
        rare = (frame["rarefied_mean"].loc[list(R.MODERN_ERAS)].mean()
                - frame["rarefied_mean"].loc[list(R.POSTBELLUM_ERAS)].mean())
        bias = 100 * (1 - rare / full)
        _assert_printed(bias, share, 0, f"{taxonomy} post-1860 bias share")
        if share < 0:
            assert abs(rare) > abs(full), "rarefaction DEEPENS the level-1 decline"
        else:
            assert bias > 60, "much larger than the modern-early share"

    def test_the_rarefied_full_sample_series_is_the_raw_arm(self, derived, trends):
        """The residual mismatch the note discloses: rarefaction runs on the full
        sample, so it checks `raw` and only speaks by analogy to the two controls.
        Asserted so the disclosure cannot go stale."""
        for taxonomy in _TAXONOMIES:
            frame = derived.rarefaction
            frame = frame[frame["taxonomy"] == taxonomy].set_index("era")["full_sample"]
            published = _level(trends, "effective_topics", taxonomy, "raw")
            assert np.allclose(frame.to_numpy(), published.to_numpy())


# =========================================================================== #
class TestSubClaim2Depth:
    """"Presidents say less about each" — dead, and the sign reverses."""

    @pytest.mark.parametrize(
        "taxonomy, arm, spearman, p_spearman, delta, ci_low, ci_high, p_delta",
        [
            ("legacy15", "raw", -0.283, 0.109, -0.145, -0.246, -0.049, 0.0010),
            ("legacy15", "sotu", 0.250, 0.367, 0.063, -0.068, 0.194, 0.345),
            ("legacy15", "std", 0.483, 0.405, 0.041, -0.077, 0.158, 0.501),
            ("llm_level2", "raw", 0.133, 0.589, 0.026, -0.038, 0.087, 0.413),
            ("llm_level2", "sotu", 0.150, 0.181, 0.050, -0.019, 0.121, 0.168),
            ("llm_level2", "std", 0.250, 0.165, 0.048, -0.016, 0.110, 0.143),
        ],
    )
    def test_labels_per_paragraph_table(
        self, trends, taxonomy, arm, spearman, p_spearman, delta, ci_low, ci_high, p_delta
    ):
        treatment = _T[arm]
        _assert_printed(
            _stat(trends, "labels_per_paragraph", taxonomy, treatment, "spearman_vs_era"),
            spearman, 3, f"lpp {taxonomy}/{arm} spearman",
        )
        _assert_printed(
            _stat(trends, "labels_per_paragraph", taxonomy, treatment,
                  "spearman_vs_era", "p_value"),
            p_spearman, 3, f"lpp {taxonomy}/{arm} spearman p",
        )
        for field, want in (("value", delta), ("ci_low", ci_low), ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, "labels_per_paragraph", taxonomy, treatment,
                      "delta_modern_minus_early", field),
                want, 3, f"lpp {taxonomy}/{arm} modern-early {field}",
            )
        _assert_printed(
            _stat(trends, "labels_per_paragraph", taxonomy, treatment,
                  "delta_modern_minus_early", "p_value"),
            p_delta, 4, f"lpp {taxonomy}/{arm} modern-early p",
        )

    def test_the_original_1_49_to_0_83_is_a_1920_vs_2010_raw_endpoint_pair(self, trends):
        levels = _level(trends, "labels_per_paragraph", "legacy15", "raw")

        _assert_printed(levels.loc[1920], 1.455, 3, "lpp legacy raw 1920")
        _assert_printed(levels.loc[2010], 0.985, 3, "lpp legacy raw 2010")
        assert int(levels.idxmax()) == 1920, "1920 is the series maximum, not a baseline"

    def test_the_raw_decline_restated_as_a_percentage(self, trends):
        levels = _level(trends, "labels_per_paragraph", "legacy15", "raw")
        early = levels.loc[list(R.EARLY_ERAS)].mean()
        delta = _stat(trends, "labels_per_paragraph", "legacy15", "raw",
                      "delta_modern_minus_early")

        _assert_printed(early, 1.233, 3, "lpp early block mean")
        _assert_printed(100 * delta / early, -11.8, 1, "lpp raw decline as %")

    # ------------------------------------------------------------------ #
    # The sixth withheld-evidence defect: the block contrast on this measure
    # was computed, is significant on three of six arms, and was printed
    # nowhere. These six rows are the note's "none omitted" table.
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "taxonomy, arm, value, ci_low, ci_high, p",
        [
            ("legacy15", "raw", -0.260, -0.319, -0.199, 0.0005),
            ("legacy15", "sotu", -0.061, -0.146, 0.030, 0.190),
            ("legacy15", "std", -0.060, -0.137, 0.020, 0.145),
            ("llm_level2", "raw", 0.051, -0.004, 0.102, 0.065),
            ("llm_level2", "sotu", 0.141, 0.076, 0.204, 0.0005),
            ("llm_level2", "std", 0.120, 0.062, 0.173, 0.0005),
        ],
    )
    def test_labels_per_paragraph_post_1860_all_six_arms(
        self, trends, taxonomy, arm, value, ci_low, ci_high, p
    ):
        treatment = _T[arm]
        for field, want in (("value", value), ("ci_low", ci_low), ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, "labels_per_paragraph", taxonomy, treatment,
                      "delta_modern_minus_postbellum", field),
                want, 3, f"lpp {taxonomy}/{arm} post-1860 {field}",
            )
        _assert_printed(
            _stat(trends, "labels_per_paragraph", taxonomy, treatment,
                  "delta_modern_minus_postbellum", "p_value"),
            p, 4, f"lpp {taxonomy}/{arm} post-1860 p",
        )

    def test_the_legacy_sign_does_not_flip_under_control_post_1860(self, trends):
        """The claim the note had to retract: "the sign flips the moment genre is
        controlled" is true of `modern - early` and FALSE of the block contrast."""
        early = [
            _stat(trends, "labels_per_paragraph", "legacy15", t,
                  "delta_modern_minus_early")
            for t in R.GENRE_TREATMENTS
        ]
        block = [
            _stat(trends, "labels_per_paragraph", "legacy15", t,
                  "delta_modern_minus_postbellum")
            for t in R.GENRE_TREATMENTS
        ]
        assert len({np.sign(v) for v in early}) == 2, "modern-early: the sign does flip"
        assert all(v < 0 for v in block), "post-1860: negative on all three arms"

    def test_three_of_the_six_post_1860_cells_are_at_the_bootstrap_floor(self, trends):
        p = [
            _stat(trends, "labels_per_paragraph", tax, t,
                  "delta_modern_minus_postbellum", "p_value")
            for tax in ("legacy15", "llm_level2")
            for t in R.GENRE_TREATMENTS
        ]
        assert sum(v < 0.05 for v in p) == 3, "three significant, not zero and not one"
        assert sum(v == 0.0005 for v in p) == 3, "and all three are at the floor"

    def test_the_larger_of_the_two_declines_is_the_one_now_quoted(self, trends):
        """-0.260 / -19.3% of the postbellum block, against -0.145 / -11.8%.

        The note had been sizing the decline it dismisses with the smaller of two
        available contrasts. Both are pinned so neither can quietly go missing.
        """
        levels = _level(trends, "labels_per_paragraph", "legacy15", "raw")
        postbellum = levels.loc[list(R.POSTBELLUM_ERAS)].mean()
        delta = _stat(trends, "labels_per_paragraph", "legacy15", "raw",
                      "delta_modern_minus_postbellum")

        _assert_printed(postbellum, 1.347, 3, "lpp postbellum block mean")
        _assert_printed(100 * delta / postbellum, -19.3, 1, "lpp raw post-1860 as %")
        assert abs(delta) > abs(
            _stat(trends, "labels_per_paragraph", "legacy15", "raw",
                  "delta_modern_minus_early")
        ), "the post-1860 decline is the larger of the two"

    def test_the_endpoint_contrast_adjudicates_nothing_on_this_measure(self, trends):
        """The note's "p = 0.24 to 0.85, so it adjudicates nothing either way"."""
        p = [
            _stat(trends, "labels_per_paragraph", tax, t,
                  "delta_last_minus_first", "p_value")
            for tax in ("legacy15", "llm_level2")
            for t in R.GENRE_TREATMENTS
        ]
        assert all(v >= 0.05 for v in p), "non-significant on all six arms"
        _assert_printed(min(p), 0.24, 2, "lpp endpoint min p")
        _assert_printed(max(p), 0.85, 2, "lpp endpoint max p")

    @pytest.mark.parametrize(
        "taxonomy, arm, v1770, v2010, spearman, p_spearman, delta, ci_low, ci_high, p_delta",
        [
            ("legacy15", "sotu", 10.72, 14.33, 0.917, 0.0005, 3.54, 2.53, 4.54, 0.0005),
            ("legacy15", "std", 10.24, 14.33, 0.900, 0.0005, 3.27, 2.36, 4.14, 0.0005),
            ("legacy15", "raw", 8.68, 10.08, 0.500, 0.0200, 1.03, 0.23, 1.81, 0.0060),
            ("llm_level2", "sotu", 13.15, 16.50, 0.617, 0.0005, 3.72, 2.86, 4.54, 0.0005),
            ("llm_level2", "raw", 12.14, 14.68, 0.667, 0.0005, 3.18, 2.52, 3.84, 0.0005),
            ("llm_level2", "std", 12.64, 16.75, 0.617, 0.0005, 3.73, 2.96, 4.45, 0.0005),
        ],
    )
    def test_labels_per_1k_words_table(
        self, trends, taxonomy, arm, v1770, v2010, spearman, p_spearman,
        delta, ci_low, ci_high, p_delta,
    ):
        treatment = _T[arm]
        levels = _level(trends, "labels_per_1k_words", taxonomy, treatment)
        _assert_printed(levels.loc[1770], v1770, 2, f"l1k {taxonomy}/{arm} 1770")
        _assert_printed(levels.loc[2010], v2010, 2, f"l1k {taxonomy}/{arm} 2010")
        _assert_printed(
            _stat(trends, "labels_per_1k_words", taxonomy, treatment, "spearman_vs_era"),
            spearman, 3, f"l1k {taxonomy}/{arm} spearman",
        )
        _assert_printed(
            _stat(trends, "labels_per_1k_words", taxonomy, treatment,
                  "spearman_vs_era", "p_value"),
            p_spearman, 4, f"l1k {taxonomy}/{arm} spearman p",
        )
        for field, want in (("value", delta), ("ci_low", ci_low), ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, "labels_per_1k_words", taxonomy, treatment,
                      "delta_modern_minus_early", field),
                want, 2, f"l1k {taxonomy}/{arm} modern-early {field}",
            )
        _assert_printed(
            _stat(trends, "labels_per_1k_words", taxonomy, treatment,
                  "delta_modern_minus_early", "p_value"),
            p_delta, 4, f"l1k {taxonomy}/{arm} modern-early p",
        )

    @pytest.mark.parametrize(
        "taxonomy, arm, delta, ci_low, ci_high, p",
        [
            ("legacy15", "raw", -0.24, -0.76, 0.32, 0.387),
            ("legacy15", "sotu", 2.20, 1.50, 2.91, 0.0005),
            ("legacy15", "std", 2.19, 1.54, 2.84, 0.0005),
            ("llm_level2", "raw", 2.98, 2.45, 3.50, 0.0005),
            ("llm_level2", "sotu", 4.14, 3.49, 4.85, 0.0005),
            ("llm_level2", "std", 4.00, 3.36, 4.60, 0.0005),
        ],
    )
    def test_all_six_post_1860_arms_including_the_one_that_fails(
        self, trends, taxonomy, arm, delta, ci_low, ci_high, p
    ):
        """The note printed only the two surviving arms in an early draft. Every
        arm is tabulated now, so a future edit cannot quietly drop the damaging
        one again."""
        treatment = _T[arm]
        for field, want in (("value", delta), ("ci_low", ci_low), ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, "labels_per_1k_words", taxonomy, treatment,
                      "delta_modern_minus_postbellum", field),
                want, 2, f"l1k {taxonomy}/{arm} modern-postbellum {field}",
            )
        _assert_printed(
            _stat(trends, "labels_per_1k_words", taxonomy, treatment,
                  "delta_modern_minus_postbellum", "p_value"),
            p, 4, f"l1k {taxonomy}/{arm} modern-postbellum p",
        )

    def test_exactly_one_of_the_six_arms_covers_zero_and_it_is_legacy_raw(self, trends):
        failing = [
            (taxonomy, treatment)
            for taxonomy in ("legacy15", "llm_level2")
            for treatment in R.GENRE_TREATMENTS
            if _stat(trends, "labels_per_1k_words", taxonomy, treatment,
                     "delta_modern_minus_postbellum", "p_value") > 0.05
        ]

        assert failing == [("legacy15", "raw")]

    @pytest.mark.parametrize(
        "taxonomy, arm, peak_era, peak, v2010, rank_2010",
        [
            ("legacy15", "raw", 1920, 11.70, 10.08, 6),
            ("legacy15", "sotu", 2010, 14.33, 14.33, 1),
            ("legacy15", "std", 2010, 14.33, 14.33, 1),
            ("llm_level2", "raw", 1950, 15.15, 14.68, 2),
            ("llm_level2", "sotu", 2010, 16.50, 16.50, 1),
            ("llm_level2", "std", 2010, 16.75, 16.75, 1),
        ],
    )
    def test_which_arms_peak_in_2010_the_rank_disclosure_for_the_dying_sub_claim(
        self, trends, taxonomy, arm, peak_era, peak, v2010, rank_2010
    ):
        """The disclosure sub-claim 1 now matches: the 2010 era is the maximum on
        all four genre-controlled arms and 6th / 2nd on the two raw arms."""
        levels = _level(trends, "labels_per_1k_words", taxonomy, _T[arm])

        assert int(levels.idxmax()) == peak_era
        _assert_printed(levels.max(), peak, 2, f"l1k {taxonomy}/{arm} peak")
        _assert_printed(levels.loc[2010], v2010, 2, f"l1k {taxonomy}/{arm} 2010")
        assert int(levels.rank(ascending=False, method="min").loc[2010]) == rank_2010

    def test_mean_paragraph_words_on_all_three_arms(self, trends):
        raw = _level(trends, "mean_paragraph_words", "none", "raw")
        sotu = _level(trends, "mean_paragraph_words", "none", "sotu_only")

        _assert_printed(raw.loc[1830], 135.9, 1, "mpw raw 1830")
        _assert_printed(raw.loc[2010], 97.7, 1, "mpw raw 2010")
        _assert_printed(sotu.loc[1830], 134.0, 1, "mpw sotu 1830")
        _assert_printed(sotu.loc[2010], 92.9, 1, "mpw sotu 2010")
        for treatment, spearman, delta in (
            ("raw", -0.833, -26.7), ("sotu_only", -0.633, -27.1),
            ("genre_standardized", -0.633, -27.9),
        ):
            _assert_printed(
                _stat(trends, "mean_paragraph_words", "none", treatment, "spearman_vs_era"),
                spearman, 3, f"mpw {treatment} spearman",
            )
            _assert_printed(
                _stat(trends, "mean_paragraph_words", "none", treatment,
                      "delta_modern_minus_early"),
                delta, 1, f"mpw {treatment} modern-early",
            )
            assert _stat(trends, "mean_paragraph_words", "none", treatment,
                         "delta_modern_minus_early", "p_value") == 0.0005

    @pytest.mark.parametrize(
        "arm, postbellum, ci_low, ci_high, endpoint",
        [
            ("raw", -22.0, -24.7, -19.4, -27.7),
            ("sotu", -23.0, -27.6, -18.4, -27.1),
            ("std", -24.2, -28.1, -19.9, -31.2),
        ],
    )
    def test_mean_paragraph_words_post_1860_and_endpoint(
        self, trends, arm, postbellum, ci_low, ci_high, endpoint
    ):
        """The mechanism does not depend on which block it is measured from."""
        treatment = _T[arm]
        for field, want in (("value", postbellum), ("ci_low", ci_low),
                            ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, "mean_paragraph_words", "none", treatment,
                      "delta_modern_minus_postbellum", field),
                want, 1, f"mpw {arm} post-1860 {field}",
            )
        _assert_printed(
            _stat(trends, "mean_paragraph_words", "none", treatment,
                  "delta_last_minus_first"),
            endpoint, 1, f"mpw {arm} endpoint",
        )

    def test_all_twelve_mean_paragraph_words_cells_are_at_the_floor(self, trends):
        cells = trends[
            (trends["unit"] == "trend") & (trends["measure"] == "mean_paragraph_words")
        ]
        assert len(cells) == 12, "3 treatments x 4 statistics"
        assert (cells["p_value"] == 0.0005).all(), "twelve of twelve at the floor"


# =========================================================================== #
class TestSubClaim3OffPolicy:
    """"More speech off-policy" — split by labeler, and the weakest survivor."""

    @pytest.mark.parametrize(
        "arm, v1770, v1920, v2010, spearman, p_spearman, delta, ci_low, ci_high, p_delta",
        [
            ("raw", 0.366, 0.195, 0.382, 0.183, 0.232, 0.045, 0.006, 0.084, 0.022),
            ("sotu", 0.286, 0.134, 0.242, -0.233, 0.154, -0.031, -0.073, 0.009, 0.143),
            ("std", 0.300, 0.154, 0.251, -0.383, 0.095, -0.030, -0.067, 0.007, 0.123),
        ],
    )
    def test_legacy_labeler_table(
        self, trends, arm, v1770, v1920, v2010, spearman, p_spearman,
        delta, ci_low, ci_high, p_delta,
    ):
        treatment = _T[arm]
        levels = _level(trends, "non_policy_share", "legacy15", treatment)
        for era, want in ((1770, v1770), (1920, v1920), (2010, v2010)):
            _assert_printed(levels.loc[era], want, 3, f"nps legacy/{arm} {era}")
        _assert_printed(
            _stat(trends, "non_policy_share", "legacy15", treatment, "spearman_vs_era"),
            spearman, 3, f"nps legacy/{arm} spearman",
        )
        _assert_printed(
            _stat(trends, "non_policy_share", "legacy15", treatment,
                  "spearman_vs_era", "p_value"),
            p_spearman, 3, f"nps legacy/{arm} spearman p",
        )
        for field, want in (("value", delta), ("ci_low", ci_low), ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, "non_policy_share", "legacy15", treatment,
                      "delta_modern_minus_early", field),
                want, 3, f"nps legacy/{arm} modern-early {field}",
            )
        _assert_printed(
            _stat(trends, "non_policy_share", "legacy15", treatment,
                  "delta_modern_minus_early", "p_value"),
            p_delta, 3, f"nps legacy/{arm} modern-early p",
        )

    @pytest.mark.parametrize(
        "arm, v1770, v1830, v2010, spearman, p_spearman, delta, p_delta, "
        "postbellum, p_postbellum",
        [
            ("raw", 0.192, 0.053, 0.174, 0.283, 0.0005, 0.050, 0.009, 0.074, 0.0005),
            ("sotu", 0.141, 0.061, 0.117, 0.100, 0.040, 0.031, 0.014, 0.053, 0.0005),
            ("std", 0.167, 0.057, 0.136, 0.183, 0.0005, 0.030, 0.009, 0.063, 0.0005),
        ],
    )
    def test_llm_labeler_table(
        self, trends, arm, v1770, v1830, v2010, spearman, p_spearman,
        delta, p_delta, postbellum, p_postbellum,
    ):
        treatment = _T[arm]
        levels = _level(trends, "non_policy_share", "llm_level2", treatment)
        for era, want in ((1770, v1770), (1830, v1830), (2010, v2010)):
            _assert_printed(levels.loc[era], want, 3, f"nps llm/{arm} {era}")
        _assert_printed(
            _stat(trends, "non_policy_share", "llm_level2", treatment, "spearman_vs_era"),
            spearman, 3, f"nps llm/{arm} spearman",
        )
        _assert_printed(
            _stat(trends, "non_policy_share", "llm_level2", treatment,
                  "spearman_vs_era", "p_value"),
            p_spearman, 4, f"nps llm/{arm} spearman p",
        )
        for statistic, value, p in (
            ("delta_modern_minus_early", delta, p_delta),
            ("delta_modern_minus_postbellum", postbellum, p_postbellum),
        ):
            _assert_printed(
                _stat(trends, "non_policy_share", "llm_level2", treatment, statistic),
                value, 3, f"nps llm/{arm} {statistic}",
            )
            _assert_printed(
                _stat(trends, "non_policy_share", "llm_level2", treatment, statistic, "p_value"),
                p, 4, f"nps llm/{arm} {statistic} p",
            )

    @pytest.mark.parametrize("arm", ["raw", "sotu", "std"])
    def test_the_modern_level_is_below_the_founding_era_on_every_treatment(
        self, trends, arm
    ):
        """The self-critical half of the surviving sub-claim: `delta_last_minus_first`
        is negative and non-significant everywhere."""
        value = _stat(trends, "non_policy_share", "llm_level2", _T[arm],
                      "delta_last_minus_first")
        p = _stat(trends, "non_policy_share", "llm_level2", _T[arm],
                  "delta_last_minus_first", "p_value")

        assert value < 0
        assert p > 0.05

    @pytest.mark.parametrize(
        "arm, trough_era, trough, dp",
        [("raw", 1830, 0.053, 3), ("sotu", 1920, 0.0560, 4), ("std", 1830, 0.057, 3)],
    )
    def test_where_the_u_bottoms_is_not_the_same_under_every_treatment(
        self, trends, arm, trough_era, trough, dp
    ):
        levels = _level(trends, "non_policy_share", "llm_level2", _T[arm])

        assert int(levels.idxmin()) == trough_era
        _assert_printed(levels.min(), trough, dp, f"nps llm/{arm} trough")

    def test_the_sotu_trough_and_the_1830_value_are_barely_distinguishable(self, trends):
        _assert_printed(
            _level(trends, "non_policy_share", "llm_level2", "sotu_only").loc[1830],
            0.0608, 4, "nps llm sotu 1830",
        )

    @pytest.mark.parametrize("arm, from_trough", [("raw", 12.1), ("sotu", 6.1), ("std", 7.9)])
    def test_the_rise_measured_from_the_trough_is_larger_than_from_the_early_block(
        self, trends, arm, from_trough
    ):
        levels = _level(trends, "non_policy_share", "llm_level2", _T[arm])

        _assert_printed(
            100 * (levels.loc[2010] - levels.min()), from_trough, 1,
            f"nps llm/{arm} 2010 minus trough",
        )

    @pytest.mark.parametrize("arm, points", [("sotu", 3.1), ("std", 3.0), ("raw", 5.0)])
    def test_the_headline_effect_size_in_percentage_points(self, trends, arm, points):
        """The note stands behind the genre-controlled +3.1 / +3.0, not the raw
        +5.0; all three are printed so the choice is visible."""
        _assert_printed(
            100 * _stat(trends, "non_policy_share", "llm_level2", _T[arm],
                        "delta_modern_minus_early"),
            points, 1, f"nps llm/{arm} modern-early in points",
        )

    @pytest.mark.parametrize(
        "arm, delta, p", [("raw", 0.105, 0.0005), ("sotu", 0.026, 0.063), ("std", 0.023, 0.087)],
    )
    def test_the_legacy_labeler_post_1860_contrast_runs_the_headline_s_way(
        self, trends, arm, delta, p
    ):
        """The disclosure an earlier draft omitted. `modern − postbellum` is
        printed for the LLM labeler and was not printed for the legacy one, where
        it is POSITIVE on all three arms and significant on raw — i.e. the arm
        that argues against this note's own "dies on the legacy labeler" verdict.
        Same failure of symmetry the note corrects elsewhere, in the direction
        that flattered its skepticism.
        """
        _assert_printed(
            _stat(trends, "non_policy_share", "legacy15", _T[arm],
                  "delta_modern_minus_postbellum"),
            delta, 3, f"nps legacy/{arm} modern-postbellum",
        )
        _assert_printed(
            _stat(trends, "non_policy_share", "legacy15", _T[arm],
                  "delta_modern_minus_postbellum", "p_value"),
            p, 4, f"nps legacy/{arm} modern-postbellum p",
        )
        assert delta > 0, "positive on every arm, which the verdict must survive"

    def test_zero_topic_share_is_a_modern_non_sotu_artifact(self, trends):
        _assert_printed(
            _level(trends, "zero_topic_share", "llm_level2", "raw").loc[2010],
            0.041, 3, "zts raw 2010",
        )
        for treatment in ("sotu_only", "genre_standardized"):
            _assert_printed(
                _level(trends, "zero_topic_share", "llm_level2", treatment).loc[2010],
                0.000, 3, f"zts {treatment} 2010",
            )

    @pytest.mark.parametrize(
        "statistic, value",
        [
            ("spearman_vs_era", 0.733),
            ("delta_modern_minus_early", 0.019),
            ("delta_modern_minus_postbellum", 0.019),
            ("delta_last_minus_first", 0.041),
        ],
    )
    def test_zero_topic_share_is_a_real_trend_on_the_raw_arm(
        self, trends, statistic, value
    ):
        """The note used to call it "not a trend"; on raw all four statistics are
        at the bootstrap floor. The genre-effect reading is what survives."""
        _assert_printed(
            _stat(trends, "zero_topic_share", "llm_level2", "raw", statistic),
            value, 3, f"zts raw {statistic}",
        )
        assert _stat(trends, "zero_topic_share", "llm_level2", "raw",
                     statistic, "p_value") == 0.0005

    def test_zero_topic_share_is_entirely_a_genre_effect(self, trends):
        """Under genre_standardized nothing is significant; under sotu_only only
        the post-1860 contrast is, at a quarter of a percentage point."""
        for statistic in R.TREND_STATISTICS:
            assert _stat(trends, "zero_topic_share", "llm_level2",
                         "genre_standardized", statistic, "p_value") >= 0.05
        _assert_printed(
            _stat(trends, "zero_topic_share", "llm_level2", "genre_standardized",
                  "delta_last_minus_first"),
            0.000, 3, "zts std endpoint",
        )
        sotu = {
            s: _stat(trends, "zero_topic_share", "llm_level2", "sotu_only", s, "p_value")
            for s in R.TREND_STATISTICS
        }
        assert [s for s, p in sotu.items() if p < 0.05] == [
            "delta_modern_minus_postbellum"
        ]
        _assert_printed(
            _stat(trends, "zero_topic_share", "llm_level2", "sotu_only",
                  "delta_modern_minus_postbellum"),
            0.0026, 4, "zts sotu post-1860",
        )
        _assert_printed(sotu["delta_modern_minus_postbellum"], 0.016, 3, "zts sotu p")


# =========================================================================== #
class TestBonusValuesDriven:
    """"More values-driven" — dead as stated; what moved is neutral description."""

    @pytest.mark.parametrize(
        "arm, spearman, p_spearman, delta, p_delta, postbellum, p_postbellum",
        [
            ("raw", -0.250, 0.391, -0.029, 0.334, -0.131, 0.0005),
            ("sotu", 0.117, 0.943, 0.018, 0.515, -0.092, 0.002),
            ("std", -0.067, 0.972, 0.011, 0.642, -0.079, 0.0005),
        ],
    )
    def test_proposal_vs_values_table(
        self, trends, arm, spearman, p_spearman, delta, p_delta, postbellum, p_postbellum
    ):
        treatment = _T[arm]
        _assert_printed(
            _stat(trends, "proposal_vs_values", "none", treatment, "spearman_vs_era"),
            spearman, 3, f"pvv/{arm} spearman",
        )
        _assert_printed(
            _stat(trends, "proposal_vs_values", "none", treatment,
                  "spearman_vs_era", "p_value"),
            p_spearman, 3, f"pvv/{arm} spearman p",
        )
        for statistic, value, p in (
            ("delta_modern_minus_early", delta, p_delta),
            ("delta_modern_minus_postbellum", postbellum, p_postbellum),
        ):
            _assert_printed(
                _stat(trends, "proposal_vs_values", "none", treatment, statistic),
                value, 3, f"pvv/{arm} {statistic}",
            )
            _assert_printed(
                _stat(trends, "proposal_vs_values", "none", treatment, statistic, "p_value"),
                p, 4, f"pvv/{arm} {statistic} p",
            )

    @pytest.mark.parametrize(
        "measure, arm, delta, p",
        [
            ("values_share", "sotu", 0.153, 0.0005),
            ("proposal_share", "sotu", 0.180, 0.0005),
            ("values_share", "raw", 0.098, 0.0005),
            ("proposal_share", "raw", 0.064, 0.0060),
            ("values_share", "std", 0.161, 0.0005),
            ("proposal_share", "std", 0.162, 0.0005),
        ],
    )
    def test_both_stance_shares_rise_on_all_three_arms(
        self, trends, measure, arm, delta, p
    ):
        _assert_printed(
            _stat(trends, measure, "none", _T[arm], "delta_modern_minus_early"),
            delta, 3, f"{measure}/{arm} modern-early",
        )
        _assert_printed(
            _stat(trends, measure, "none", _T[arm], "delta_modern_minus_early", "p_value"),
            p, 4, f"{measure}/{arm} modern-early p",
        )

    def test_neither_share_collapses_inside_annual_messages(self, trends):
        levels = _level(trends, "neither_share", "none", "sotu_only")

        _assert_printed(levels.loc[1800], 0.480, 3, "neither sotu 1800")
        _assert_printed(levels.loc[1980], 0.089, 3, "neither sotu 1980")
        _assert_printed(levels.loc[2010], 0.205, 3, "neither sotu 2010")
        _assert_printed(
            _stat(trends, "neither_share", "none", "sotu_only", "spearman_vs_era"),
            -0.717, 3, "neither sotu spearman",
        )
        for field, want in (("value", -0.262), ("ci_low", -0.309), ("ci_high", -0.216)):
            _assert_printed(
                _stat(trends, "neither_share", "none", "sotu_only",
                      "delta_modern_minus_early", field),
                want, 3, f"neither sotu modern-early {field}",
            )
        for treatment, want in (("raw", -0.093), ("genre_standardized", -0.254)):
            _assert_printed(
                _stat(trends, "neither_share", "none", treatment, "delta_modern_minus_early"),
                want, 3, f"neither {treatment} modern-early",
            )
            assert _stat(trends, "neither_share", "none", treatment,
                         "delta_modern_minus_early", "p_value") == 0.0005

    @pytest.mark.parametrize(
        "arm, delta, p", [("sotu", -0.114, 0.062), ("std", -0.114, 0.047), ("raw", 0.092, 0.077)],
    )
    def test_the_endpoint_contrast_does_not_support_the_collapse(
        self, trends, arm, delta, p
    ):
        """The disclosure the bonus claim owed itself, matching the one sub-claim 3
        makes: measured 1770 -> 2010 the neutral-prose share is flat or (on the raw
        arm) HIGHER. The finding rests on the block contrast and the Spearman."""
        _assert_printed(
            _stat(trends, "neither_share", "none", _T[arm], "delta_last_minus_first"),
            delta, 3, f"neither {arm} last-first",
        )
        _assert_printed(
            _stat(trends, "neither_share", "none", _T[arm],
                  "delta_last_minus_first", "p_value"),
            p, 3, f"neither {arm} last-first p",
        )
        assert p > 0.04, "at or beyond the edge of significance on every arm"

    def test_the_1770_era_is_already_low_so_the_fall_is_from_the_plateau(self, trends):
        for treatment, want in (("sotu_only", 0.319), ("raw", 0.248)):
            _assert_printed(
                _level(trends, "neither_share", "none", treatment).loc[1770],
                want, 3, f"neither {treatment} 1770",
            )
        sotu = _level(trends, "neither_share", "none", "sotu_only")
        assert sotu.loc[1770] < sotu.loc[1800], "1770 sits below the 1800-era peak"

    @pytest.mark.parametrize(
        "arm, rho, p",
        [("raw", -0.483, 0.006), ("sotu", -0.717, 0.0005), ("std", -0.783, 0.0005)],
    )
    def test_the_neither_share_spearman_is_given_on_all_three_arms(
        self, trends, arm, rho, p
    ):
        """The section says the block contrast and the Spearman carry the finding,
        which one arm of one of them cannot establish."""
        _assert_printed(
            _stat(trends, "neither_share", "none", _T[arm], "spearman_vs_era"),
            rho, 3, f"neither {arm} spearman",
        )
        _assert_printed(
            _stat(trends, "neither_share", "none", _T[arm], "spearman_vs_era", "p_value"),
            p, 4, f"neither {arm} spearman p",
        )

    @pytest.mark.parametrize(
        "arm, top_era, top, floor_era, floor",
        [
            ("sotu", 1800, 0.480, 1860, 0.373),
            ("raw", 1800, 0.432, 1890, 0.371),
            ("std", 1800, 0.454, 1890, 0.376),
        ],
    )
    def test_the_19th_century_plateau_is_quoted_per_arm(
        self, trends, arm, top_era, top, floor_era, floor
    ):
        """An earlier draft took the plateau's top from `sotu_only` (48.0%) and its
        floor from `raw` (37.1%) and printed them as one range."""
        levels = _level(trends, "neither_share", "none", _T[arm])
        _assert_printed(levels.loc[top_era], top, 3, f"neither {arm} plateau top")
        _assert_printed(levels.loc[floor_era], floor, 3, f"neither {arm} plateau floor")
        plateau = levels.loc[[1800, 1830, 1860, 1890]]
        assert int(plateau.idxmax()) == top_era and int(plateau.idxmin()) == floor_era

    @pytest.mark.parametrize(
        "arm, value, p",
        [("raw", -0.060, 0.005), ("sotu", -0.176, 0.0005), ("std", -0.192, 0.0005)],
    )
    def test_neither_share_also_falls_post_1860_on_all_three(self, trends, arm, value, p):
        """Unlike the two stance shares, this half of the bonus finding does not
        depend on which block it is measured from."""
        _assert_printed(
            _stat(trends, "neither_share", "none", _T[arm],
                  "delta_modern_minus_postbellum"),
            value, 3, f"neither {arm} post-1860",
        )
        _assert_printed(
            _stat(trends, "neither_share", "none", _T[arm],
                  "delta_modern_minus_postbellum", "p_value"),
            p, 4, f"neither {arm} post-1860 p",
        )

    # ------------------------------------------------------------------ #
    # MAJOR: the rebuttal "both stance shares rise" was evidenced on
    # `modern - early` while rebutting a post-1860 finding. On the contrast
    # actually at issue, raw proposal_share is flat.
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "measure, arm, value, p",
        [
            ("values_share", "raw", 0.122, 0.0005),
            ("values_share", "sotu", 0.179, 0.0005),
            ("values_share", "std", 0.175, 0.0005),
            ("proposal_share", "raw", -0.011, 0.524),
            ("proposal_share", "sotu", 0.083, 0.0005),
            ("proposal_share", "std", 0.085, 0.0005),
        ],
    )
    def test_stance_shares_post_1860_all_six_arms(self, trends, measure, arm, value, p):
        _assert_printed(
            _stat(trends, measure, "none", _T[arm], "delta_modern_minus_postbellum"),
            value, 3, f"{measure} {arm} post-1860",
        )
        _assert_printed(
            _stat(trends, measure, "none", _T[arm],
                  "delta_modern_minus_postbellum", "p_value"),
            p, 4, f"{measure} {arm} post-1860 p",
        )

    def test_both_shares_rise_on_five_of_six_arms_not_six(self, trends):
        """The exact count the note now claims, and the identity of the exception."""
        rising = [
            (measure, treatment)
            for measure in ("values_share", "proposal_share")
            for treatment in R.GENRE_TREATMENTS
            if _stat(trends, measure, "none", treatment,
                     "delta_modern_minus_postbellum") > 0
            and _stat(trends, measure, "none", treatment,
                      "delta_modern_minus_postbellum", "p_value") < 0.05
        ]
        assert len(rising) == 5
        assert ("proposal_share", "raw") not in rising, "the raw arm is the exception"

    def test_values_share_rises_at_the_floor_on_every_arm_of_every_block(self, trends):
        for statistic in ("delta_modern_minus_early", "delta_modern_minus_postbellum"):
            for treatment in R.GENRE_TREATMENTS:
                assert _stat(trends, "values_share", "none", treatment,
                             statistic) > 0
                assert _stat(trends, "values_share", "none", treatment,
                             statistic, "p_value") == 0.0005

    def test_proposal_share_raw_spearman_is_the_only_non_significant_one(self, trends):
        _assert_printed(
            _stat(trends, "proposal_share", "none", "raw", "spearman_vs_era"),
            0.333, 3, "proposal raw spearman",
        )
        _assert_printed(
            _stat(trends, "proposal_share", "none", "raw",
                  "spearman_vs_era", "p_value"),
            0.151, 3, "proposal raw spearman p",
        )
        for treatment in ("sotu_only", "genre_standardized"):
            _assert_printed(
                _stat(trends, "proposal_share", "none", treatment, "spearman_vs_era"),
                0.733, 3, f"proposal {treatment} spearman",
            )
            assert _stat(trends, "proposal_share", "none", treatment,
                         "spearman_vs_era", "p_value") == 0.0005

    def test_values_share_spearman_is_at_the_floor_on_all_three_arms(self, trends):
        """Redundant with the block contrasts but significant, so printed: the
        completeness rule covers unprinted cells only if they are also n.s."""
        for treatment, rho in (
            ("raw", 0.333), ("sotu_only", 0.583), ("genre_standardized", 0.667),
        ):
            _assert_printed(
                _stat(trends, "values_share", "none", treatment, "spearman_vs_era"),
                rho, 3, f"values_share {treatment} spearman",
            )
            assert _stat(trends, "values_share", "none", treatment,
                         "spearman_vs_era", "p_value") == 0.0005

    def test_the_stance_endpoint_contrasts_are_null_on_all_nine_arms(self, trends):
        """Why the note does not tabulate them: p = 0.32 to 0.88, nine of nine."""
        p = [
            _stat(trends, measure, "none", treatment,
                  "delta_last_minus_first", "p_value")
            for measure in ("proposal_share", "values_share", "proposal_vs_values")
            for treatment in R.GENRE_TREATMENTS
        ]
        assert len(p) == 9 and all(v >= 0.05 for v in p)
        _assert_printed(min(p), 0.32, 2, "stance endpoint min p")
        _assert_printed(max(p), 0.88, 2, "stance endpoint max p")

    @pytest.mark.parametrize("era", [1770, 1830, 1920, 2010])
    def test_the_four_way_stance_split_is_recoverable(self, trends, era):
        """`mixed_share = proposal + values + neither - 1`, which is what makes
        `neither_share` a residual of pre-declared measures rather than a new arm."""
        mixed = sum(
            _level(trends, measure, "none", "sotu_only").loc[era]
            for measure in ("proposal_share", "values_share", "neither_share")
        ) - 1.0

        assert -1e-9 <= mixed <= 1.0


# =========================================================================== #
class TestRegisterBlock:
    """The star finding: eleven markers, all three arms, same direction."""

    _MARKERS = [
        ("mechanism", 47.7, 17.7, -0.72, -0.92, -0.80, -27.0, -32.9, -21.1),
        ("hedges", 46.8, 7.5, -1.00, -0.95, -1.00, -35.7, -39.1, -32.2),
        ("fk_grade", 19.0, 8.6, -1.00, -0.95, -0.98, -9.0, -9.5, -8.5),
        ("words_per_sentence", 37.2, 15.5, -0.98, -0.95, -0.98, -19.5, -20.6, -18.5),
        ("opponents", 0.0, 12.5, 1.00, 1.00, 0.98, 6.5, 5.1, 7.9),
        ("boosters", 5.4, 26.5, 0.87, 0.93, 0.97, 10.4, 7.2, 13.9),
        ("hype", 1.8, 17.3, 0.85, 0.92, 0.93, 8.9, 6.9, 11.0),
        ("doom", 1.4, 10.1, 0.73, 0.98, 0.98, 6.0, 4.8, 7.2),
        ("future", 5.9, 16.4, 0.70, 0.67, 0.67, 13.0, 10.6, 15.4),
        ("nostalgia", 7.7, 15.6, 0.83, 0.47, 0.67, 6.5, 4.7, 8.3),
        ("superlatives", 5.9, 15.3, 0.73, 0.77, 0.80, 6.5, 4.6, 8.5),
    ]

    @pytest.mark.parametrize(
        "marker, v1770, v2010, s_raw, s_sotu, s_std, delta, ci_low, ci_high", _MARKERS
    )
    def test_register_table_row(
        self, trends, marker, v1770, v2010, s_raw, s_sotu, s_std, delta, ci_low, ci_high
    ):
        levels = _level(trends, marker, "none", "sotu_only")
        _assert_printed(levels.loc[1770], v1770, 1, f"{marker} sotu 1770")
        _assert_printed(levels.loc[2010], v2010, 1, f"{marker} sotu 2010")
        for treatment, want in (
            ("raw", s_raw), ("sotu_only", s_sotu), ("genre_standardized", s_std)
        ):
            _assert_printed(
                _stat(trends, marker, "none", treatment, "spearman_vs_era"),
                want, 2, f"{marker} {treatment} spearman",
            )
        for field, want in (("value", delta), ("ci_low", ci_low), ("ci_high", ci_high)):
            _assert_printed(
                _stat(trends, marker, "none", "sotu_only",
                      "delta_modern_minus_early", field),
                want, 1, f"{marker} sotu modern-early {field}",
            )
        assert _stat(trends, marker, "none", "sotu_only",
                     "delta_modern_minus_early", "p_value") == 0.0005

    @pytest.mark.parametrize("marker", [row[0] for row in _MARKERS])
    def test_every_marker_is_significant_on_all_three_arms_not_merely_same_signed(
        self, trends, marker
    ):
        """The claim that makes the star block immune to the arm-dependence caveat:
        the RAW arm — the independent one — is significant too, at the floor."""
        deltas = [
            _stat(trends, marker, "none", treatment, "delta_modern_minus_early")
            for treatment in R.GENRE_TREATMENTS
        ]
        p_values = [
            _stat(trends, marker, "none", treatment,
                  "delta_modern_minus_early", "p_value")
            for treatment in R.GENRE_TREATMENTS
        ]

        assert len({np.sign(d) for d in deltas}) == 1
        assert p_values == [0.0005, 0.0005, 0.0005]

    def test_eight_of_eleven_markers_are_near_monotone_and_three_are_not(self, trends):
        """"Near-perfect monotonicity" restated as a count, because three markers
        are not: superlatives +0.77, future +0.67, nostalgia +0.47.

        The threshold is applied to the UNROUNDED correlation. `mechanism` and
        `hype` both print as 0.92 in the note's table and are 0.9167 in the
        parquet, so a 0.92 cutoff would silently drop two markers the note counts
        among the eight.
        """
        spearman = {
            marker: _stat(trends, marker, "none", "sotu_only", "spearman_vs_era")
            for marker, *_ in self._MARKERS
        }
        strong = {m for m, rho in spearman.items() if abs(rho) >= 0.90}

        assert len(strong) == 8
        assert set(spearman) - strong == {"superlatives", "future", "nostalgia"}
        assert min(abs(spearman[m]) for m in strong) == pytest.approx(0.9167, abs=5e-5)

    @pytest.mark.parametrize(
        "marker, raw, p_raw, sotu, p_sotu, std, p_std",
        [
            ("us_them", 12.7, 0.003, 0.13, 0.984, -0.53, 0.871),
            ("nrc_hope", -74.9, 0.0005, -14.3, 0.196, -24.1, 0.021),
            ("nrc_fear", 1.5, 0.838, 30.7, 0.009, 27.9, 0.009),
            ("religiosity", 2.1, 0.186, 3.6, 0.002, 5.3, 0.0005),
        ],
    )
    def test_markers_that_do_not_survive_are_reported_because_declared(
        self, trends, marker, raw, p_raw, sotu, p_sotu, std, p_std
    ):
        for treatment, value, p in (
            ("raw", raw, p_raw), ("sotu_only", sotu, p_sotu), ("genre_standardized", std, p_std)
        ):
            _assert_printed(
                _stat(trends, marker, "none", treatment, "delta_modern_minus_early"),
                value, 2 if abs(value) < 1 else 1, f"{marker} {treatment} modern-early",
            )
            _assert_printed(
                _stat(trends, marker, "none", treatment,
                      "delta_modern_minus_early", "p_value"),
                p, 4, f"{marker} {treatment} modern-early p",
            )

    # ------------------------------------------------------------------ #
    # The omission that made a finding look WEAKER than it is: the post hoc
    # block contrast confirms the register block on all 33 cells, and was
    # printed nowhere.
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "marker, raw, sotu, std",
        [
            ("mechanism", -26.4, -21.2, -24.7),
            ("hedges", -17.2, -22.9, -22.6),
            ("fk_grade", -6.4, -5.7, -5.9),
            ("words_per_sentence", -11.8, -11.1, -11.2),
            ("opponents", 6.0, 5.0, 4.9),
            ("boosters", 9.4, 8.3, 7.5),
            ("hype", 7.3, 7.9, 7.6),
            ("doom", 2.4, 4.1, 4.0),
            ("future", 9.0, 12.8, 12.3),
            ("nostalgia", 3.9, 6.5, 6.3),
            ("superlatives", 5.1, 6.6, 5.8),
        ],
    )
    def test_register_block_post_1860_table(self, trends, marker, raw, sotu, std):
        for treatment, want in (
            ("raw", raw), ("sotu_only", sotu), ("genre_standardized", std)
        ):
            _assert_printed(
                _stat(trends, marker, "none", treatment,
                      "delta_modern_minus_postbellum"),
                want, 1, f"{marker} {treatment} post-1860",
            )
            assert _stat(trends, marker, "none", treatment,
                         "delta_modern_minus_postbellum", "p_value") == 0.0005

    @pytest.mark.parametrize("marker", [row[0] for row in _MARKERS])
    def test_the_post_1860_sign_matches_modern_minus_early_on_every_arm(
        self, trends, marker
    ):
        for treatment in R.GENRE_TREATMENTS:
            early = _stat(trends, marker, "none", treatment, "delta_modern_minus_early")
            block = _stat(trends, marker, "none", treatment,
                          "delta_modern_minus_postbellum")
            assert np.sign(early) == np.sign(block), f"{marker}/{treatment}"

    def test_the_endpoint_contrast_agrees_too(self, trends):
        """33 of 33 significant, 32 at the floor; `mechanism` sotu_only is 0.002."""
        p = {
            (marker, treatment): _stat(trends, marker, "none", treatment,
                                       "delta_last_minus_first", "p_value")
            for marker, *_ in self._MARKERS
            for treatment in R.GENRE_TREATMENTS
        }
        assert len(p) == 33
        assert all(v < 0.05 for v in p.values())
        assert sum(v == 0.0005 for v in p.values()) == 32
        assert [k for k, v in p.items() if v != 0.0005] == [("mechanism", "sotu_only")]
        _assert_printed(p[("mechanism", "sotu_only")], 0.002, 3, "mechanism sotu endpoint p")

    def test_one_hundred_thirty_two_of_one_hundred_thirty_two(self, trends):
        """The count "survives everything" now stands on, made checkable."""
        markers = [row[0] for row in self._MARKERS]
        cells = trends[
            (trends["unit"] == "trend") & (trends["measure"].isin(markers))
        ]
        assert len(cells) == 11 * 3 * 4 == 132
        assert int((cells["p_value"] < 0.05).sum()) == 132
        assert int((cells["p_value"] == 0.0005).sum()) == 131

    # ------------------------------------------------------------------ #
    # Three of the four non-surviving markers reverse their reading on the
    # post hoc contrast. Every verdict in the table above is a verdict about
    # `modern - early` only.
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "marker, raw, p_raw, sotu, p_sotu, std, p_std",
        [
            ("us_them", 26.4, 0.0005, 18.5, 0.0005, 15.0, 0.0005),
            ("nrc_hope", -36.3, 0.0005, -3.2, 0.822, -1.3, 0.937),
            ("nrc_fear", -20.5, 0.001, 10.1, 0.238, 4.9, 0.528),
            ("religiosity", 6.5, 0.0005, 7.6, 0.0005, 8.5, 0.0005),
        ],
    )
    def test_non_surviving_markers_post_1860(
        self, trends, marker, raw, p_raw, sotu, p_sotu, std, p_std
    ):
        for treatment, value, p in (
            ("raw", raw, p_raw), ("sotu_only", sotu, p_sotu),
            ("genre_standardized", std, p_std),
        ):
            _assert_printed(
                _stat(trends, marker, "none", treatment,
                      "delta_modern_minus_postbellum"),
                value, 1, f"{marker} {treatment} post-1860",
            )
            _assert_printed(
                _stat(trends, marker, "none", treatment,
                      "delta_modern_minus_postbellum", "p_value"),
                p, 4, f"{marker} {treatment} post-1860 p",
            )

    def test_nothing_significant_hides_in_the_untabulated_endpoint_group(self, trends):
        """The completeness rule's claim about the four non-surviving markers:
        nrc_hope at the floor on all three, us_them significant on all three, and
        nrc_fear / religiosity non-significant on all three (p = 0.26 to 0.95)."""
        for treatment in R.GENRE_TREATMENTS:
            assert _stat(trends, "nrc_hope", "none", treatment,
                         "delta_last_minus_first", "p_value") == 0.0005
            assert _stat(trends, "us_them", "none", treatment,
                         "delta_last_minus_first", "p_value") < 0.05
        quiet = [
            _stat(trends, marker, "none", treatment,
                  "delta_last_minus_first", "p_value")
            for marker in ("nrc_fear", "religiosity")
            for treatment in R.GENRE_TREATMENTS
        ]
        assert all(p >= 0.05 for p in quiet)
        _assert_printed(min(quiet), 0.26, 2, "quiet endpoint min p")
        _assert_printed(max(quiet), 0.95, 2, "quiet endpoint max p")

    def test_nrc_fear_is_arm_dependent_rather_than_the_least_stable_measure(
        self, trends
    ):
        """B3(a): two of `nrc_fear`'s Spearmans are significant and were printed
        nowhere, while the same table row's `religiosity` had all three printed.

        With them in, the genre-controlled picture is two significantly positive
        statistics of four and none significantly negative — so it is the raw arm
        that is unstable, not the measure, and "the least stable measure in the
        note" overstated it in the direction of the note's own skepticism.
        """
        for treatment, rho, p in (
            ("raw", 0.100, 0.857), ("sotu_only", 0.733, 0.029),
            ("genre_standardized", 0.783, 0.028),
        ):
            _assert_printed(
                _stat(trends, "nrc_fear", "none", treatment, "spearman_vs_era"),
                rho, 3, f"nrc_fear {treatment} spearman",
            )
            _assert_printed(
                _stat(trends, "nrc_fear", "none", treatment,
                      "spearman_vs_era", "p_value"),
                p, 3, f"nrc_fear {treatment} spearman p",
            )
        for treatment in ("sotu_only", "genre_standardized"):
            significant = [
                (_stat(trends, "nrc_fear", "none", treatment, statistic),
                 _stat(trends, "nrc_fear", "none", treatment, statistic, "p_value"))
                for statistic in R.TREND_STATISTICS
            ]
            positive = [v for v, p in significant if p < 0.05 and v > 0]
            negative = [v for v, p in significant if p < 0.05 and v < 0]
            assert len(positive) == 2 and negative == [], treatment
        # Only `raw` carries a significant negative cell.
        assert _stat(trends, "nrc_fear", "none", "raw",
                     "delta_modern_minus_postbellum", "p_value") < 0.05

    def test_us_them_does_not_die_under_control_on_every_contrast(self, trends):
        """`us_them`'s four statistics split THREE ways, not two.

        "Genre artifact; dies under control" is true of `modern - early` only. The
        Spearman is n.s. on all three arms — nothing was significant on `raw` for
        the control to kill, so that statistic says "no trend anywhere" rather
        than "genre artifact". An earlier docstring here said the split was two
        and two, and the body checked only the half that was right.
        """
        for treatment, rho, p in (
            ("raw", 0.150, 0.147), ("sotu_only", 0.083, 0.649),
            ("genre_standardized", 0.033, 0.737),
        ):
            _assert_printed(
                _stat(trends, "us_them", "none", treatment, "spearman_vs_era"),
                rho, 3, f"us_them {treatment} spearman",
            )
            assert _stat(trends, "us_them", "none", treatment,
                         "spearman_vs_era", "p_value") >= 0.05
            _assert_printed(
                _stat(trends, "us_them", "none", treatment,
                      "spearman_vs_era", "p_value"),
                p, 3, f"us_them {treatment} spearman p",
            )
        assert _stat(trends, "us_them", "none", "raw",
                     "delta_modern_minus_early", "p_value") < 0.05
        for treatment in ("sotu_only", "genre_standardized"):
            assert _stat(trends, "us_them", "none", treatment,
                         "delta_modern_minus_early", "p_value") >= 0.05
        for treatment in R.GENRE_TREATMENTS:
            assert _stat(trends, "us_them", "none", treatment,
                         "delta_modern_minus_postbellum", "p_value") == 0.0005
        for treatment, value, p in (
            ("raw", 51.4, 0.0005), ("sotu_only", 23.4, 0.036),
            ("genre_standardized", 22.6, 0.022),
        ):
            _assert_printed(
                _stat(trends, "us_them", "none", treatment, "delta_last_minus_first"),
                value, 1, f"us_them {treatment} endpoint",
            )
            _assert_printed(
                _stat(trends, "us_them", "none", treatment,
                      "delta_last_minus_first", "p_value"),
                p, 3, f"us_them {treatment} endpoint p",
            )

    def test_religiosity_is_not_only_visible_under_genre_control(self, trends):
        """Raw is n.s. on `modern - early` (+2.1) and on the Spearman (+0.367), and
        at the bootstrap floor post-1860 (+6.5) — three times the size."""
        for treatment, rho, p in (
            ("raw", 0.367, 0.105), ("sotu_only", 0.267, 0.030),
            ("genre_standardized", 0.400, 0.001),
        ):
            _assert_printed(
                _stat(trends, "religiosity", "none", treatment, "spearman_vs_era"),
                rho, 3, f"religiosity {treatment} spearman",
            )
            _assert_printed(
                _stat(trends, "religiosity", "none", treatment,
                      "spearman_vs_era", "p_value"),
                p, 4, f"religiosity {treatment} spearman p",
            )
        early = _stat(trends, "religiosity", "none", "raw", "delta_modern_minus_early")
        block = _stat(trends, "religiosity", "none", "raw",
                      "delta_modern_minus_postbellum")
        assert abs(block) > 3 * abs(early) * 0.9, "roughly three times the size"
        assert _stat(trends, "religiosity", "none", "raw",
                     "delta_modern_minus_postbellum", "p_value") == 0.0005

    def test_nrc_hope_is_significant_under_control_on_three_of_four_statistics(
        self, trends
    ):
        """"Nothing under control" was false against a figure the note printed."""
        assert _stat(trends, "nrc_hope", "none", "genre_standardized",
                     "delta_modern_minus_early", "p_value") < 0.05
        for treatment, rho, p in (
            ("raw", -0.667, 0.0005),
            ("genre_standardized", -0.267, 0.0005), ("sotu_only", -0.233, 0.016),
        ):
            _assert_printed(
                _stat(trends, "nrc_hope", "none", treatment, "spearman_vs_era"),
                rho, 3, f"nrc_hope {treatment} spearman",
            )
            _assert_printed(
                _stat(trends, "nrc_hope", "none", treatment,
                      "spearman_vs_era", "p_value"),
                p, 4, f"nrc_hope {treatment} spearman p",
            )
        for treatment in R.GENRE_TREATMENTS:
            assert _stat(trends, "nrc_hope", "none", treatment,
                         "delta_last_minus_first", "p_value") == 0.0005
        # Only the post-1860 contrast supports "nothing under control".
        for treatment in ("sotu_only", "genre_standardized"):
            assert _stat(trends, "nrc_hope", "none", treatment,
                         "delta_modern_minus_postbellum", "p_value") > 0.05


# =========================================================================== #
class TestMethodNotes:
    """Figures cited in the method section, none of which ship in the parquet."""

    @pytest.mark.parametrize(
        "column, icc",
        [("n_legacy", 0.116), ("n_llm", 0.135), ("legacy_zero", 0.103),
         ("llm_all_non_policy", 0.155), ("llm_zero", 0.294)],
    )
    def test_within_speech_icc_table(self, derived, column, icc):
        _assert_printed(derived.icc[column], icc, 3, f"ICC {column}")

    def test_the_per_issue_icc_exceeds_the_inherited_panel_figures(self, derived):
        """The recomputed mean is nearly double the panel's 0.059 and the max is
        above its 0.119 — the disagreement runs conservative, which is the note's
        whole point in publishing it."""
        _assert_printed(derived.per_issue_icc.mean(), 0.110, 3, "per-issue ICC mean")
        _assert_printed(derived.per_issue_icc.max(), 0.171, 3, "per-issue ICC max")
        assert derived.icc.min() > 0.10

    def test_label_case_variant_repair(self, derived):
        variants = derived.report[~derived.report["exact_match"]]

        assert int(derived.report["n_assignments"].sum()) == 52_855
        assert int(variants["n_assignments"].sum()) == 111
        assert int(variants["n_assignments"].max()) == 84
        _assert_printed(100 * 111 / 52_855, 0.21, 2, "case-variant share")

    def test_the_two_labelers_disagree_about_unlabelled_paragraphs(self, derived):
        _assert_printed(derived.para["legacy_zero"].mean(), 0.287, 3, "legacy zero rate")
        _assert_printed(derived.para["llm_zero"].mean(), 0.011, 3, "llm zero rate")
        _assert_printed(derived.para["n_llm"].mean(), 1.459, 3, "topics per paragraph")

    def test_half_the_intervals_are_degenerate_and_the_schema_does_not_flag_it(
        self, trends
    ):
        degenerate = trends[
            trends["ci_low"].notna() & trends["ci_high"].notna()
            & (trends["ci_low"] == trends["ci_high"])
        ]
        year_sotu = trends[
            (trends["unit"] == "year") & (trends["genre_treatment"] == "sotu_only")
        ]
        single_speech_raw = trends[
            (trends["unit"] == "year") & (trends["genre_treatment"] == "raw")
            & (trends["n_speeches"] == 1)
        ]

        assert len(degenerate) == 8_013
        _assert_printed(100 * len(degenerate) / len(trends), 49.3, 1, "degenerate share")
        assert len(year_sotu) == 6_808
        assert (year_sotu["n_speeches"] == 1).all()
        assert year_sotu["period"].nunique() == 184
        assert len(single_speech_raw) == 999
        assert single_speech_raw["period"].nunique() == 27
        assert len(degenerate) - len(year_sotu) - len(single_speech_raw) == 206

    def test_the_206_remaining_degenerate_rows_are_not_all_benign(self, trends):
        """An earlier draft called all 206 "markers at zero in every early-era
        draw". Two of them are neither markers, nor zero, nor early-era: they are
        genuine n=2 single-cluster collapses, and they are the only two of the 206
        with a non-zero value."""
        degenerate = trends[
            trends["ci_low"].notna() & trends["ci_high"].notna()
            & (trends["ci_low"] == trends["ci_high"])
        ]
        rest = degenerate[
            ~((degenerate["unit"] == "year") & (degenerate["n_speeches"] == 1))
        ]
        assert len(rest) == 206

        zero_topic = rest[rest["measure"] == "zero_topic_share"]
        assert len(zero_topic) == 146, "the majority, and all exactly 0.0"
        assert (zero_topic["value"] == 0.0).all()
        assert len(rest) - len(zero_topic) == 60

        non_zero = rest[rest["value"] != 0.0]
        assert len(non_zero) == 2, "the only two that are not constant-at-zero"
        assert (non_zero["n_speeches"] == 2).all(), "both rest on two speeches"
        by_year = non_zero.set_index("period")
        _assert_printed(by_year.loc[1891, "value"], 10.421, 3, "1891 plugin")
        assert by_year.loc[1891, "measure"] == "effective_topics_plugin"
        assert by_year.loc[1891, "taxonomy"] == "legacy15"
        assert int(by_year.loc[1891, "n_bootstrap_valid"]) == 1_527
        _assert_printed(by_year.loc[1904, "value"], 0.546, 3, "1904 pvv")
        assert by_year.loc[1904, "measure"] == "proposal_vs_values"
        assert int(by_year.loc[1904, "n_bootstrap_valid"]) == 1_514

    def test_the_zero_topic_share_degenerate_rows_are_not_an_early_era_story(
        self, trends
    ):
        """37 of the 144 that carry a period sit in the 1920 and 2010 era bands, so
        "markers at zero in every early-era draw" described a minority of a
        minority."""
        degenerate = trends[
            trends["ci_low"].notna() & trends["ci_high"].notna()
            & (trends["ci_low"] == trends["ci_high"])
            & (trends["measure"] == "zero_topic_share")
        ]
        rest = degenerate[
            ~((degenerate["unit"] == "year") & (degenerate["n_speeches"] == 1))
        ]
        assert len(rest) == 146
        assert rest["unit"].value_counts().to_dict() == {"year": 137, "era": 7, "trend": 2}

        dated = rest[rest["period"].notna()]
        assert len(dated) == 144, "the two trend rows carry no period"
        band = dated["period"].apply(
            lambda p: 1770 + 30 * min(8, max(0, (int(p) - 1770) // 30))
        )
        assert set(band) == set(_ERAS), "present in all nine era bands"
        assert int(band.isin([1920, 2010]).sum()) == 37

    def test_exactly_three_rows_sit_outside_their_interval_and_all_are_year_1984(
        self, trends
    ):
        over = trends[
            trends["value"].notna() & trends["ci_high"].notna()
            & (trends["value"] > trends["ci_high"] + 1e-12)
        ]
        under = trends[
            trends["value"].notna() & trends["ci_low"].notna()
            & (trends["value"] < trends["ci_low"] - 1e-12)
        ]

        assert len(over) == 3 and len(under) == 0
        assert set(over["period"]) == {1984}
        assert set(over["taxonomy"]) == {"llm_level2"}
        assert set(over["genre_treatment"]) == {"raw"}
        assert set(over["measure"]) == {
            "effective_topics", "effective_topics_plugin", "normalized_entropy"
        }
        assert int(over["n_speeches"].iloc[0]) == 9
        row = over[over["measure"] == "effective_topics"].iloc[0]
        _assert_printed(row["value"], 21.066, 3, "1984 effective_topics")
        _assert_printed(row["ci_high"], 21.007, 3, "1984 effective_topics ci_high")

    @pytest.mark.parametrize(
        "arm, spearman, delta, p",
        [("raw", -0.300, -0.066, 0.0020), ("sotu", -0.317, -0.073, 0.0005),
         ("std", -0.350, -0.120, 0.0005)],
    )
    def test_self_reference_ships_and_falls_and_is_now_reported(
        self, trends, arm, spearman, delta, p
    ):
        """An undeclared measure that ships in the parquet and moves significantly.
        Left undiscussed it is a significant series sitting inside the artifact of
        a note that claims to publish everything it computed."""
        _assert_printed(
            _stat(trends, "self_reference", "none", _T[arm], "spearman_vs_era"),
            spearman, 3, f"self_reference/{arm} spearman",
        )
        _assert_printed(
            _stat(trends, "self_reference", "none", _T[arm], "delta_modern_minus_early"),
            delta, 3, f"self_reference/{arm} modern-early",
        )
        _assert_printed(
            _stat(trends, "self_reference", "none", _T[arm],
                  "delta_modern_minus_early", "p_value"),
            p, 4, f"self_reference/{arm} modern-early p",
        )

    @pytest.mark.parametrize(
        "arm, block, p_block, endpoint, p_endpoint",
        [
            ("raw", -0.023, 0.153, -0.202, 0.0005),
            ("sotu", -0.074, 0.0005, -0.136, 0.005),
            ("std", -0.103, 0.0005, -0.197, 0.0005),
        ],
    )
    def test_self_reference_other_two_statistics_are_reported_too(
        self, trends, arm, block, p_block, endpoint, p_endpoint
    ):
        """The completeness promise applies to the measure, not to two of its four
        statistics. The block contrast is significant under both controls and is
        the one cell of the twelve that misses on raw."""
        for statistic, value, p in (
            ("delta_modern_minus_postbellum", block, p_block),
            ("delta_last_minus_first", endpoint, p_endpoint),
        ):
            _assert_printed(
                _stat(trends, "self_reference", "none", _T[arm], statistic),
                value, 3, f"self_reference/{arm} {statistic}",
            )
            _assert_printed(
                _stat(trends, "self_reference", "none", _T[arm], statistic, "p_value"),
                p, 4, f"self_reference/{arm} {statistic} p",
            )

    def test_exactly_one_self_reference_cell_of_twelve_is_not_significant(self, trends):
        cells = trends[
            (trends["unit"] == "trend") & (trends["measure"] == "self_reference")
        ]
        assert len(cells) == 12
        missing = cells[cells["p_value"] >= 0.05]
        assert len(missing) == 1
        row = missing.iloc[0]
        assert row["genre_treatment"] == "raw"
        assert row["statistic"] == "delta_modern_minus_postbellum"

    @pytest.mark.parametrize("taxonomy", list(_TAXONOMIES))
    @pytest.mark.parametrize("treatment", list(R.GENRE_TREATMENTS))
    def test_the_miller_madow_correction_does_not_drive_any_post_1860_conclusion(
        self, trends, taxonomy, treatment
    ):
        """Corrected and uncorrected block contrasts agree within 0.04 on all nine
        arms, with the same sign and the same side of 0.05. If they ever diverged,
        the note's breadth verdicts would be an artifact of the bias correction."""
        corrected = _stat(trends, "effective_topics", taxonomy, treatment,
                          "delta_modern_minus_postbellum")
        plugin = _stat(trends, "effective_topics_plugin", taxonomy, treatment,
                       "delta_modern_minus_postbellum")
        p_corrected = _stat(trends, "effective_topics", taxonomy, treatment,
                            "delta_modern_minus_postbellum", "p_value")
        p_plugin = _stat(trends, "effective_topics_plugin", taxonomy, treatment,
                         "delta_modern_minus_postbellum", "p_value")

        assert abs(corrected - plugin) < 0.04
        assert np.sign(corrected) == np.sign(plugin)
        assert (p_corrected < 0.05) == (p_plugin < 0.05)

    @pytest.mark.parametrize(
        "statistic, largest_gap",
        [("delta_modern_minus_early", 0.098), ("delta_last_minus_first", 0.235)],
    )
    def test_the_plugin_twin_agrees_in_sign_on_the_other_two_contrasts_too(
        self, trends, statistic, largest_gap
    ):
        """The note's "within 0.04" is scoped to the block contrast; the other two
        contrasts are looser and the note now says by how much."""
        gaps = []
        for taxonomy in _TAXONOMIES:
            for treatment in R.GENRE_TREATMENTS:
                corrected = _stat(trends, "effective_topics", taxonomy, treatment,
                                  statistic)
                plugin = _stat(trends, "effective_topics_plugin", taxonomy, treatment,
                               statistic)
                assert np.sign(corrected) == np.sign(plugin)
                gaps.append(abs(corrected - plugin))
        assert len(gaps) == 9
        _assert_printed(max(gaps), largest_gap, 3, f"{statistic} largest twin gap")

    @pytest.mark.parametrize("measure", ["effective_topics_plugin", "normalized_entropy"])
    def test_the_monotone_twins_have_identical_spearman_to_effective_topics(
        self, trends, measure
    ):
        """Why their Spearman rows are not tabulated a second and third time: the
        era ranks are equal, so the rank correlations are equal digit for digit.

        `normalized_entropy` really is a monotone transform, so its *p-values*
        match too. `effective_topics_plugin` is **not** — the Miller-Madow term is
        data-dependent — it merely happens to induce the same era ordering, and
        its p-values differ. The note's scope rule 3 says so; an earlier draft
        called both "monotone transforms".
        """
        levels = trends[(trends["unit"] == "era") & (trends["statistic"] == "level")]
        for taxonomy in _TAXONOMIES:
            for treatment in R.GENRE_TREATMENTS:
                assert _stat(trends, measure, taxonomy, treatment,
                             "spearman_vs_era") == _stat(
                    trends, "effective_topics", taxonomy, treatment, "spearman_vs_era"
                )

                def ranks(name):
                    rows = levels[
                        (levels["measure"] == name)
                        & (levels["taxonomy"] == taxonomy)
                        & (levels["genre_treatment"] == treatment)
                    ]
                    return list(rows.sort_values("period")["value"].rank(ascending=False))

                assert ranks(measure) == ranks("effective_topics")

                same_p = _stat(trends, measure, taxonomy, treatment,
                               "spearman_vs_era", "p_value") == _stat(
                    trends, "effective_topics", taxonomy, treatment,
                    "spearman_vs_era", "p_value"
                )
                if measure == "normalized_entropy":
                    assert same_p, "a true monotone transform cannot move the p-value"

    def test_the_plugins_spearman_p_values_differ_from_the_published_ones(self, trends):
        """The three cells that make "monotone transform" false of the plugin."""
        for treatment, plugin_p, published_p in (
            ("genre_standardized", 0.028, 0.037),
            ("sotu_only", 0.005, 0.008),
            ("raw", 0.0005, 0.0010),
        ):
            _assert_printed(
                _stat(trends, "effective_topics_plugin", "llm_level1", treatment,
                      "spearman_vs_era", "p_value"),
                plugin_p, 4, f"plugin {treatment} spearman p",
            )
            _assert_printed(
                _stat(trends, "effective_topics", "llm_level1", treatment,
                      "spearman_vs_era", "p_value"),
                published_p, 4, f"published {treatment} spearman p",
            )

    def test_normalized_entropy_exceeds_one_on_exactly_one_row(self, trends):
        """The Miller-Madow numerator against an uncorrected log(K) denominator.
        Pinned at ONE row so that a change to the estimator — which would perturb
        a provenance-stamped artifact for a cosmetic gain — cannot slip through."""
        rows = trends[trends["measure"] == "normalized_entropy"]
        over_one = rows[rows["value"] > 1.0]

        assert len(over_one) == 1
        assert len(rows) == 1_317
        row = over_one.iloc[0]
        assert (row["unit"], int(row["period"]), row["taxonomy"], row["genre_treatment"]) == (
            "year", 1965, "legacy15", "sotu_only"
        )
        assert int(row["n_speeches"]) == 1
        _assert_printed(row["value"], 1.0137, 4, "normalized_entropy maximum")


# =========================================================================== #
class TestConfirmatoryDesign:
    """H1's cited figure, which is the one this note predicts will fail."""

    def test_h1_effect_and_p_value(self, trends):
        _assert_printed(
            _stat(trends, "effective_topics", "llm_level2", "sotu_only",
                  "delta_modern_minus_postbellum"),
            1.20, 2, "H1 effect",
        )
        _assert_printed(
            _stat(trends, "effective_topics", "llm_level2", "sotu_only",
                  "delta_modern_minus_postbellum", "p_value"),
            0.042, 3, "H1 p",
        )


# =========================================================================== #
class TestCompletenessOfReporting:
    """The part of the omission problem that can be mechanised.

    Two things are pinned here. The arm count of each contrast the note claims to
    report "in full", so that quietly dropping the arm that disagrees fails — nine
    defects of exactly that shape reached review, each one a table that printed
    some arms of a contrast and not the one running the other way. And the three
    self-referential claims the completeness section makes: scope rule 1's
    enumeration of endpoint/block disagreements, scope rule 2's bound on the two
    columns it omits, and the closing "everything else significant is reported".
    """

    #: (measure, taxonomy, statistic) the note tabulates on all three treatments.
    _FULLY_TABULATED = [
        ("labels_per_paragraph", "legacy15", "delta_modern_minus_early"),
        ("labels_per_paragraph", "legacy15", "delta_modern_minus_postbellum"),
        ("labels_per_paragraph", "llm_level2", "delta_modern_minus_early"),
        ("labels_per_paragraph", "llm_level2", "delta_modern_minus_postbellum"),
        ("labels_per_1k_words", "legacy15", "delta_modern_minus_early"),
        ("labels_per_1k_words", "legacy15", "delta_modern_minus_postbellum"),
        ("labels_per_1k_words", "llm_level2", "delta_modern_minus_early"),
        ("labels_per_1k_words", "llm_level2", "delta_modern_minus_postbellum"),
        ("mean_paragraph_words", "none", "delta_modern_minus_early"),
        ("mean_paragraph_words", "none", "delta_modern_minus_postbellum"),
        ("non_policy_share", "legacy15", "delta_modern_minus_early"),
        ("non_policy_share", "legacy15", "delta_modern_minus_postbellum"),
        ("non_policy_share", "llm_level2", "delta_modern_minus_early"),
        ("non_policy_share", "llm_level2", "delta_modern_minus_postbellum"),
        ("proposal_vs_values", "none", "delta_modern_minus_early"),
        ("proposal_vs_values", "none", "delta_modern_minus_postbellum"),
        ("proposal_share", "none", "delta_modern_minus_postbellum"),
        ("values_share", "none", "delta_modern_minus_postbellum"),
        ("neither_share", "none", "delta_modern_minus_early"),
        ("neither_share", "none", "delta_modern_minus_postbellum"),
        ("neither_share", "none", "delta_last_minus_first"),
        ("self_reference", "none", "delta_modern_minus_early"),
        ("self_reference", "none", "delta_modern_minus_postbellum"),
        ("self_reference", "none", "delta_last_minus_first"),
        ("effective_topics", "legacy15", "delta_modern_minus_early"),
        ("effective_topics", "legacy15", "delta_modern_minus_postbellum"),
        ("effective_topics", "llm_level2", "delta_modern_minus_early"),
        ("effective_topics", "llm_level2", "delta_modern_minus_postbellum"),
        ("effective_topics", "llm_level1", "delta_modern_minus_early"),
        ("effective_topics", "llm_level1", "delta_modern_minus_postbellum"),
        ("effective_topics_plugin", "legacy15", "delta_modern_minus_postbellum"),
        ("effective_topics_plugin", "llm_level2", "delta_modern_minus_postbellum"),
        ("effective_topics_plugin", "llm_level1", "delta_modern_minus_postbellum"),
        ("zero_topic_share", "llm_level2", "delta_modern_minus_postbellum"),
    ] + [
        (marker, "none", statistic)
        for marker in (
            "mechanism", "hedges", "fk_grade", "words_per_sentence", "opponents",
            "boosters", "hype", "doom", "future", "nostalgia", "superlatives",
        )
        for statistic in ("delta_modern_minus_postbellum",)
    ] + [
        (marker, "none", statistic)
        for marker in ("us_them", "nrc_hope", "nrc_fear", "religiosity")
        for statistic in ("delta_modern_minus_early", "delta_modern_minus_postbellum")
    ]

    @pytest.mark.parametrize("measure, taxonomy, statistic", _FULLY_TABULATED)
    def test_each_fully_tabulated_contrast_has_all_three_arms_in_the_parquet(
        self, trends, measure, taxonomy, statistic
    ):
        """If an arm ever disappears from the artifact, the note's "none omitted"
        tables become unverifiable rather than merely wrong."""
        rows = trends[
            (trends["unit"] == "trend")
            & (trends["measure"] == measure)
            & (trends["taxonomy"] == taxonomy)
            & (trends["statistic"] == statistic)
        ]
        assert set(rows["genre_treatment"]) == set(R.GENRE_TREATMENTS)
        assert len(rows) == 3
        assert rows["p_value"].notna().all()

    def test_every_measure_taxonomy_column_has_all_four_statistics_on_all_arms(
        self, trends
    ):
        """No cell of the 444 can go missing without this failing, which is the
        precondition for the prose audit being able to enumerate them at all."""
        cells = trends[trends["unit"] == "trend"]
        grouped = cells.groupby(["measure", "taxonomy", "genre_treatment"])
        assert grouped.ngroups == 111
        assert set(grouped.size().unique()) == {4}
        for (_, _, _), group in grouped:
            assert set(group["statistic"]) == set(R.TREND_STATISTICS)

    # ---- the three assertions the completeness section makes about itself ----

    _REGISTER_MARKERS = (
        "mechanism", "hedges", "fk_grade", "words_per_sentence", "opponents",
        "boosters", "hype", "doom", "future", "nostalgia", "superlatives",
    )

    def test_scope_rule_1s_enumeration_of_endpoint_block_disagreements(self, trends):
        """B3(b): `effective_topics` belongs in that enumeration and is its
        strongest case — the only published measure in the 444 whose significant
        contrasts point in opposite directions, on exactly the arms carrying
        sub-claim 1's "significantly falling" verdict. `labels_per_1k_words` is
        the second omission: 5 of 6 endpoint cells significant, none printed.
        """
        for treatment, endpoint, endpoint_p, block, block_p in (
            ("sotu_only", 2.11, 0.026, -0.95, 0.004),
            ("genre_standardized", 1.97, 0.024, -0.90, 0.009),
        ):
            _assert_printed(
                _stat(trends, "effective_topics", "llm_level1", treatment,
                      "delta_last_minus_first"),
                endpoint, 2, f"level-1 {treatment} endpoint",
            )
            _assert_printed(
                _stat(trends, "effective_topics", "llm_level1", treatment,
                      "delta_last_minus_first", "p_value"),
                endpoint_p, 3, f"level-1 {treatment} endpoint p",
            )
            _assert_printed(
                _stat(trends, "effective_topics", "llm_level1", treatment,
                      "delta_modern_minus_postbellum"),
                block, 2, f"level-1 {treatment} post-1860",
            )
            _assert_printed(
                _stat(trends, "effective_topics", "llm_level1", treatment,
                      "delta_modern_minus_postbellum", "p_value"),
                block_p, 3, f"level-1 {treatment} post-1860 p",
            )

        cells = trends[trends["unit"] == "trend"]
        reversing = {
            measure
            for (measure, _, _), group in cells.groupby(
                ["measure", "taxonomy", "genre_treatment"])
            if (lambda s: len(s) >= 2 and s.min() < 0 < s.max())(
                group.loc[group["p_value"] < 0.05, "value"]
            )
        }
        assert reversing == {
            "effective_topics", "effective_topics_plugin", "normalized_entropy"
        }, "the twins inherit the property; nothing else in the table has it"

        endpoints = {
            (taxonomy, treatment): _stat(
                trends, "labels_per_1k_words", taxonomy, treatment,
                "delta_last_minus_first", "p_value")
            for taxonomy in ("legacy15", "llm_level2")
            for treatment in R.GENRE_TREATMENTS
        }
        assert sum(p < 0.05 for p in endpoints.values()) == 5
        assert endpoints[("legacy15", "raw")] >= 0.05
        _assert_printed(endpoints[("legacy15", "raw")], 0.189, 3, "the failing arm p")
        _assert_printed(
            _stat(trends, "labels_per_1k_words", "legacy15", "raw",
                  "delta_last_minus_first"),
            1.41, 2, "the failing arm",
        )

    def test_scope_rule_2s_bound_on_the_two_unprinted_register_columns(self, trends):
        """Scope rule 2 omits `raw` and `genre_standardized` from the register
        block's `modern - early` table and states a bound on what that costs.

        The earlier bound — "the same size to within the width of the printed
        intervals" — was false, and it was the one sentence in that passage whose
        whole purpose is to be checkable. The bound now stated is: same sign,
        every cell at the floor, no gap above 4.02 in the marker's own units,
        exactly 8 of the 22 outside the printed interval, and exactly one gap
        wider than its own interval (`fk_grade`, raw).
        """
        outside, wider_than_width, raw_larger, gaps = 0, [], 0, []
        for marker in self._REGISTER_MARKERS:
            lo = _stat(trends, marker, "none", "sotu_only",
                       "delta_modern_minus_early", "ci_low")
            hi = _stat(trends, marker, "none", "sotu_only",
                       "delta_modern_minus_early", "ci_high")
            printed = _stat(trends, marker, "none", "sotu_only",
                            "delta_modern_minus_early")
            for treatment in ("raw", "genre_standardized"):
                other = _stat(trends, marker, "none", treatment,
                              "delta_modern_minus_early")
                assert np.sign(other) == np.sign(printed), f"{marker}/{treatment} sign"
                assert _stat(trends, marker, "none", treatment,
                             "delta_modern_minus_early", "p_value") == 0.0005
                gap = abs(other - printed)
                gaps.append(gap)
                if not lo <= other <= hi:
                    outside += 1
                if gap > hi - lo:
                    wider_than_width.append((marker, treatment))
                if treatment == "raw" and abs(other) > abs(printed):
                    raw_larger += 1

        assert len(gaps) == 22
        # The note states 4.02 as a bound, not as a value, so the test checks the
        # bound holds AND that it is tight enough not to be vacuous.
        assert max(gaps) <= 4.02, f"largest gap {max(gaps):.3f} exceeds the stated bound"
        assert max(gaps) > 4.0, "the stated bound has gone slack"
        assert outside == 8, "cells outside the printed sotu_only interval"
        assert wider_than_width == [("fk_grade", "raw")]
        assert raw_larger == 4, "markers where raw is LARGER than the printed figure"

    def test_the_only_measures_the_note_calls_declared_are_the_declared_ones(self):
        """B2: `nrc_fear` was called "pre-declared" in four places and is not in
        the Pre-declaration table — neither is `nrc_hope`.

        "Status: EXPLORATORY" is line 3 of the note and pre-registration
        discipline is its central epistemic asset, so a false claim about what was
        declared is the same defect the note already retracted once and itself
        called "worse than no claim". This reads the table as the single source of
        truth and holds the prose to it.

        Only the ⊆ direction is asserted. Equality would demand that the prose
        describe every declared measure as declared, which it has no reason to do
        — several are declared and then simply reported.
        """
        note = (Path(__file__).resolve().parents[1]
                / "notes" / "register-findings-v1.md").read_text()
        measures = set(pd.read_parquet(_TRENDS)["measure"])

        # The Pre-declaration table: rows between its header and the blank line.
        table = re.search(
            r"\| Measure \| Headline predicts.*?\n\n", note, re.S
        ).group(0)
        declared = {m for m in measures if f"`{m}`" in table}
        assert declared == {
            "effective_topics", "labels_per_paragraph", "labels_per_1k_words",
            "non_policy_share", "proposal_vs_values", "mechanism", "fk_grade",
            "religiosity", "hype", "us_them",
        }
        assert "nrc_fear" not in declared and "nrc_hope" not in declared

        # Every prose claim of the form "`x` ... declared", segment by segment so
        # a claim cannot reach across a sentence or bullet boundary into the next
        # measure's name. Negations do NOT skip the segment: B2 lived in exactly
        # the shape "`a` was not declared while `b` was pre-declared", so skipping
        # would blind this scanner to the defect it exists for. The segment is cut
        # *at* each negation instead and the pieces scanned independently — the
        # negated piece loses its declaredness word, the positive piece keeps its.
        # A split phrasing this pattern misses ("neither `a` nor `b` was declared")
        # fails loudly rather than silently, which is the right direction.
        claimed: set[str] = set()
        negation = re.compile(
            r"(?:not|never|neither|no)\s*(?:\*\*)?\s*(?:pre-)?declar\w*|undeclared"
        )
        for segment in re.split(r"(?<=[.:;])\s+|\n\s*\n|\n(?=\s*[-*|#])", note):
            for piece in negation.split(segment):
                if not re.search(r"\b(?:pre-)?declar(?:ed|ation)\b", piece):
                    continue
                claimed |= {m for m in measures if f"`{m}`" in piece}
        # Non-vacuity: the note does make positive declaredness claims, so an
        # empty `claimed` would mean the scanner had stopped working.
        assert "us_them" in claimed and "religiosity" in claimed
        assert claimed <= declared, (
            f"the note calls {sorted(claimed - declared)} declared, but the "
            "Pre-declaration table does not"
        )

    def test_every_significant_unprinted_cell_falls_inside_a_declared_scope_rule(
        self, trends
    ):
        """The assertion the completeness section actually makes, mechanised.

        "Everything else that is significant and would otherwise go undiscussed is
        reported" is a claim about the 444 minus what the prose renders, and the
        module docstring above says this file structurally cannot check omissions.
        This is the part that can be: enumerate the 444, decide which values the
        note renders, and require every significant leftover to sit inside one of
        the four scope rules the note declares.

        The matcher has three rules, each closing a way a rendering can match a
        number printed for something else:

        * **no 0-decimal rendering, and none under two significant digits** — see
          `_renderings` and its own test above. 0.7833 at 0 dp is "1", which occurs
          in almost any region of prose; that artifact hid `nrc_fear`'s two
          significant Spearmans (+0.783, +0.733) from an earlier sweep. No count of
          hidden cells is quoted: earlier drafts named two different ones, each
          measured against a draft that no longer exists.
        * **whole-token match**, so "0.78" cannot match inside "10.783".
        * **measure-scoped**: the match must land in a section that names the
          measure, so `nrc_fear`'s +0.783 cannot be satisfied by
          `effective_topics`' +0.783 four sections away.

        Residual collisions make the matcher call a cell printed that is not,
        which weakens this test but cannot make it fail spuriously.
        """
        note = (Path(__file__).resolve().parents[1]
                / "notes" / "register-findings-v1.md").read_text()
        note = note.replace("−", "-").replace("–", "-").lower()
        sections, current = [], []
        for line in note.splitlines():
            if re.match(r"^#{1,2} ", line):
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        sections.append("\n".join(current))

        aliases = {
            "effective_topics": ("effective_topics", "effective topics", "breadth"),
            "effective_topics_plugin": ("effective_topics_plugin", "plug-in", "plugin"),
            "labels_per_paragraph": ("labels_per_paragraph", "per paragraph"),
            "labels_per_1k_words": ("labels_per_1k_words", "per 1,000 words",
                                    "labels-per-1,000-words"),
            "mean_paragraph_words": ("mean_paragraph_words", "paragraph words",
                                     "paragraphs got shorter"),
            "non_policy_share": ("non_policy_share", "non-policy", "off-policy",
                                 "zero-anchored-issue"),
        }

        scoped: dict[str, list[str]] = {}

        def rendered(row) -> bool:
            measure = row["measure"]
            if measure not in scoped:
                names = aliases.get(measure, (measure,))
                scoped[measure] = [s for s in sections if any(n in s for n in names)]
            for text in _renderings(row["value"]):
                pattern = re.compile(r"(?<![0-9.])" + re.escape(text) + r"(?![0-9])")
                if any(pattern.search(section) for section in scoped[measure]):
                    return True
            return False

        def in_a_scope_rule(row) -> bool:
            # 1 + 4: endpoint contrasts, summarised rather than tabulated. The
            # note's rule 1 is narrower — "wherever the verdict rests on a block
            # contrast" — but the note's "Sample sizes" section puts every verdict
            # on a block contrast, so the two coincide today. The broad coded form
            # is the permissive direction: it can only excuse a cell the note would
            # not, never fail on one the note does cover.
            if row["statistic"] == "delta_last_minus_first":
                return True
            # 2: the register block's `modern - early` on raw and genre-std.
            if (row["measure"] in self._REGISTER_MARKERS
                    and row["statistic"] == "delta_modern_minus_early"
                    and row["genre_treatment"] in ("raw", "genre_standardized")):
                return True
            # 3: the two `effective_topics` twins.
            return row["measure"] in ("effective_topics_plugin", "normalized_entropy")

        cells = trends[trends["unit"] == "trend"]
        assert len(cells) == 444
        leaked = [
            (r["measure"], r["taxonomy"], r["genre_treatment"], r["statistic"],
             round(float(r["value"]), 4), float(r["p_value"]))
            for _, r in cells.iterrows()
            if r["p_value"] < 0.05 and not rendered(r) and not in_a_scope_rule(r)
        ]
        assert leaked == [], (
            "significant cells the note neither renders nor covers with a scope "
            f"rule: {leaked}"
        )

    def test_the_register_block_is_the_only_measure_with_no_failing_cell(self, trends):
        """"Survives everything" is a claim about the whole 132-cell block, and it
        is the only block in the table with that property."""
        markers = [
            "mechanism", "hedges", "fk_grade", "words_per_sentence", "opponents",
            "boosters", "hype", "doom", "future", "nostalgia", "superlatives",
        ]
        cells = trends[trends["unit"] == "trend"]
        per_measure = cells.groupby("measure")["p_value"].apply(lambda s: (s < 0.05).all())
        clean = set(per_measure[per_measure].index)
        assert set(markers) <= clean, "all eleven register markers are clean"
        # `mean_paragraph_words` is the only non-register measure that also has no
        # failing cell; the note says so where it reports the mechanism.
        assert clean - set(markers) == {"mean_paragraph_words"}
