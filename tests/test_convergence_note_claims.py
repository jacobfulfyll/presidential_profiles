"""Pins every number printed in `notes/convergence-findings-v1.md` to the artifact
it was derived from.

A findings note is a deliverable, not documentation: it gets read as the result.
This project's own history is that a note's *tables* stay correct while its
*prose* drifts -- `issue-attention-over-time` shipped 16 prose defects with
flawless tables, and on `topic-chart-upgrades` the one new prose block WITHOUT a
re-derivation test was the only one that shipped a defect. So the rule is: pin
the numbers wherever they live.

Two properties make this file worth its length:

1. It re-derives from `data/convergence/*.parquet`, so it fails when the
   ARTIFACT moves, not when the sentence is edited. A number that changed
   because the analysis changed is exactly the case a reader cannot detect.
2. Each check names the section it guards, so a failure says which claim went
   stale rather than only which float differs.

Skips cleanly when `data/convergence/` is absent (a fresh clone before the
first build), the same way the artifact-anchor test in `test_convergence.py` does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import convergence as C

NOTE_PATH = Path(__file__).resolve().parents[1] / "notes" / "convergence-findings-v1.md"


@pytest.fixture(scope="module")
def note() -> str:
    if not NOTE_PATH.exists():
        pytest.skip("findings note not present")
    return NOTE_PATH.read_text()


@pytest.fixture(scope="module")
def art() -> dict:
    if not C.META_PATH.exists():
        pytest.skip("data/convergence/ not built")
    return {
        "curves": pd.read_parquet(C.DISPERSION_PATH),
        "null": pd.read_parquet(C.PERMUTATION_PATH),
        "jack": pd.read_parquet(C.JACKKNIFE_PATH),
        "paradox": pd.read_parquet(C.PARADOX_PATH),
        "meta": json.loads(C.META_PATH.read_text()),
    }


def _stat(null: pd.DataFrame, arm: str, statistic: str,
          kind: str = "rolling", treatment: str = "all_windows") -> pd.Series:
    rows = null[
        (null.arm == arm) & (null.window_kind == kind)
        & (null.treatment == treatment) & (null.statistic == statistic)
    ]
    assert len(rows) == 1, f"expected one row for {arm}/{kind}/{treatment}/{statistic}"
    return rows.iloc[0]


# --------------------------------------------------------------------------
# section 1 -- the result tables
# --------------------------------------------------------------------------
class TestSection1TheResult:
    @pytest.mark.parametrize(
        "arm,statistic,rho,p_one,p_two",
        [
            ("corex_all", "dispersion", -0.0047, 0.4808, 0.9580),
            ("corex_all", "floor_corrected", -0.2175, 0.2764, 0.5657),
            ("corex_all", "excess_ratio", 0.2300, 0.7351, 0.5192),
            ("corex_all", "pair_regression", -0.1584, 0.1284, 0.2719),
            ("corex_sotu", "dispersion", 0.4254, 0.8891, 0.2034),
            ("corex_sotu", "floor_corrected", 0.2915, 0.7621, 0.4923),
            ("corex_sotu", "excess_ratio", 0.7833, 0.9965, 0.0095),
            ("corex_sotu", "pair_regression", 0.1530, 0.8546, 0.2799),
            ("llm_all", "dispersion", 0.1895, 0.7456, 0.5257),
            ("llm_all", "floor_corrected", -0.0665, 0.5882, 0.8461),
            ("llm_all", "excess_ratio", 0.2950, 0.8411, 0.2879),
            ("llm_all", "pair_regression", -0.0702, 0.2769, 0.5507),
        ],
    )
    def test_every_printed_statistic_matches_the_artifact(
        self, art, arm, statistic, rho, p_one, p_two
    ):
        row = _stat(art["null"], arm, statistic)
        assert row.rho == pytest.approx(rho, abs=5e-5)
        assert row.p_one_sided == pytest.approx(p_one, abs=5e-5)
        assert row.p_two_sided == pytest.approx(p_two, abs=5e-5)

    @pytest.mark.parametrize(
        "arm,crit", [("corex_all", -0.5246), ("corex_sotu", -0.5514),
                     ("llm_all", -0.5283)]
    )
    def test_the_permutation_critical_values(self, art, arm, crit):
        assert _stat(art["null"], arm, "dispersion").null_crit_05 == pytest.approx(
            crit, abs=5e-5
        )

    def test_nothing_is_a_significant_decline(self, art):
        """The headline claim in one assertion. If any cell ever fires, the
        note's entire framing is wrong, not merely a number."""
        assert not art["null"].significant_decline.any()

    def test_the_decision_cell_and_its_literal_headline(self, art, note):
        decision = art["meta"]["decision"]
        assert decision["cell"] == "no_convergence"
        # The note quotes the pre-committed sentence verbatim; a paraphrase here
        # would be the note quietly rewriting its own pre-registration. Compared
        # on whitespace-normalized text with blockquote markers stripped,
        # because the note hard-wraps it inside a `>` quote -- wrapping and
        # quoting are formatting, rewording is not.
        flat = " ".join(re.sub(r"(?m)^>\s?", "", note).split())
        assert " ".join(decision["headline"].split()) in flat

    @pytest.mark.parametrize(
        "arm,rho", [("corex_all", 0.0970), ("corex_sotu", 0.3995), ("llm_all", 0.2767)]
    )
    def test_the_dating_subset_does_not_flip_sign(self, art, arm, rho):
        assert _stat(
            art["null"], arm, "dispersion", treatment="ends_2014"
        ).rho == pytest.approx(rho, abs=5e-5)

    @pytest.mark.parametrize(
        "arm,rho", [("corex_all", -0.2143), ("corex_sotu", 0.6429), ("llm_all", 0.2619)]
    )
    def test_the_nonoverlapping_grid(self, art, arm, rho):
        assert _stat(
            art["null"], arm, "dispersion", kind="nonoverlapping"
        ).rho == pytest.approx(rho, abs=5e-5)


# --------------------------------------------------------------------------
# section 2 -- the gate
# --------------------------------------------------------------------------
class TestSection2TheSelftest:
    @pytest.mark.parametrize(
        "key,value",
        [
            ("a_cluster_null_rho", -0.089), ("a_cluster_null_rho_sd", 0.136),
            ("a_paragraph_null_rho", -0.406), ("a_paragraph_null_rho_sd", 0.087),
            ("b_injected_rho", -0.970), ("b_injected_rho_sd", 0.014),
            ("c_permutation_size", 0.050),
        ],
    )
    def test_every_printed_selftest_leg(self, art, key, value):
        assert art["meta"]["selftest"][key] == pytest.approx(value, abs=5e-4)

    @pytest.mark.parametrize(
        "key,value", [("corpora", 12), ("c_replicates", 40), ("c_permutations_each", 150)]
    )
    def test_the_selftest_sample_sizes(self, art, key, value):
        assert art["meta"]["selftest"][key] == value

    def test_the_note_does_not_quote_the_reviews_real_data_figure(self, note):
        """-0.605 is the ADVERSARIAL REVIEW's figure on real data, not this
        module's output (-0.406 on synthetic corpora). The note says so
        explicitly; it must never present it as a measurement of this code."""
        for match in re.finditer(r"0\.605", note):
            window = note[max(0, match.start() - 400): match.end() + 200]
            assert "Do not quote" in window or "review's figure" in window

    def test_the_improvement_factor_is_rounded_from_the_fraction(self, art, note):
        """The one defect drafting this note produced: 4.6x came from the
        already-rounded 0.406/0.089. From the fraction it is 4.5x. Pinned
        because double-rounding has now bitten three notes in this project."""
        st = art["meta"]["selftest"]
        ratio = st["a_paragraph_null_rho"] / st["a_cluster_null_rho"]
        assert f"**{ratio:.1f}×**" in note
        # "4.6" may appear ONLY inside the paragraph that discloses it as the
        # defect -- never as a live figure. Checking the neighbourhood rather
        # than mere absence keeps the disclosure honest without banning the word.
        for match in re.finditer(r"4\.6×", note):
            window = note[max(0, match.start() - 200): match.end() + 200]
            assert "defect" in window and "Double-rounding" in window


# --------------------------------------------------------------------------
# section 3 -- the mechanism (breadth, not convergence)
# --------------------------------------------------------------------------
class TestSection3Mechanism:
    @pytest.mark.parametrize(
        "arm,first,last",
        [("corex_all", 2.751, 2.954), ("corex_sotu", 2.900, 3.233),
         ("llm_all", 3.257, 3.730)],
    )
    def test_entropy_rises_in_every_arm(self, art, arm, first, last):
        s = art["curves"]
        s = s[(s.arm == arm) & (s.window_kind == "rolling") & s.used_in_trend]
        s = s.sort_values("window_center")
        assert s.mean_within_president_entropy.iloc[0] == pytest.approx(first, abs=6e-4)
        assert s.mean_within_president_entropy.iloc[-1] == pytest.approx(last, abs=6e-4)
        # The direction is the claim, not just the endpoints.
        assert last > first

    def test_excess_ratio_rises_while_the_matched_null_falls(self, art):
        """The mechanism sentence: presidents became LESS alike than their own
        broadening predicts. Both halves must hold in every arm or the
        explanation in section 3 is wrong even though its numbers are right."""
        from scipy.stats import spearmanr

        for arm in ("corex_all", "corex_sotu", "llm_all"):
            s = art["curves"]
            s = s[(s.arm == arm) & (s.window_kind == "rolling") & s.used_in_trend]
            matched = spearmanr(s.window_center, s.entropy_matched).statistic
            excess = spearmanr(s.window_center, s.excess_ratio).statistic
            assert matched < 0, f"{arm}: entropy-matched null did not fall"
            assert excess > 0, f"{arm}: excess_ratio did not rise"

    def test_no_excess_ratio_trend_is_significant(self, art):
        """Section 3 explicitly disclaims divergence. If one of these ever
        becomes significant the disclaimer must change."""
        rows = art["null"][art["null"].statistic == "excess_ratio"]
        assert not rows.significant_decline.any()


# --------------------------------------------------------------------------
# section 4 -- coverage
# --------------------------------------------------------------------------
class TestSection4Coverage:
    @pytest.mark.parametrize(
        "arm,used,ever,bins,speeches",
        [("corex_all", 105, 38, 16, 923), ("corex_sotu", 92, 35, 16, 216),
         ("llm_all", 105, 38, 51, 923)],
    )
    def test_the_coverage_table(self, art, arm, used, ever, bins, speeches):
        cov = art["meta"]["coverage"][arm]
        assert cov["n_windows_rolling"] == 105
        assert cov["n_windows_used"] == used
        assert cov["n_presidents_ever_eligible"] == ever
        assert cov["n_bins"] == bins
        assert cov["n_qualifying_speeches"] == speeches

    def test_the_sotu_arm_really_does_use_fewer_windows_than_the_other_two(self, art):
        """The 92-vs-101 reconciliation only matters because the bases differ on
        this arm and coincide on the others."""
        cov = art["meta"]["coverage"]
        assert cov["corex_sotu"]["n_windows_used"] < cov["corex_all"]["n_windows_used"]
        assert cov["corex_all"]["n_windows_used"] == cov["corex_all"]["n_windows_rolling"]

    @pytest.mark.parametrize("arm,n", [("corex_all", 99), ("corex_sotu", 86), ("llm_all", 99)])
    def test_the_dating_window_counts(self, art, arm, n):
        s = art["curves"]
        s = s[(s.arm == arm) & (s.window_kind == "rolling")]
        assert int(s.in_ends_2014.sum()) == n


# --------------------------------------------------------------------------
# section 5 -- H2a anchor and H2b
# --------------------------------------------------------------------------
class TestSection5Hypotheses:
    def test_h2a_is_an_unscored_anchor_and_still_reproduces(self, art):
        px = art["paradox"]
        assert not px.anchor_is_scored.any()
        trump = px[px.president.str.contains("Trump")].iloc[0]
        assert trump.stylistic_similarity_anchor == pytest.approx(0.7447, abs=5e-5)
        # "the lowest of all 38" -- a rank claim, not just a value
        assert trump.stylistic_similarity_anchor == px.stylistic_similarity_anchor.min()
        assert len(px) == 38

    def test_h2b_numbers_and_its_non_significance(self, art):
        trump = art["paradox"][art["paradox"].president.str.contains("Trump")].iloc[0]
        others = art["paradox"][~art["paradox"].president.str.contains("Trump")]
        assert trump.corrected_distance == pytest.approx(0.0122, abs=5e-5)
        assert others.corrected_distance.mean() == pytest.approx(0.0285, abs=5e-5)
        assert int(trump.rank_corrected) == 6
        assert trump.percentile_corrected == pytest.approx(0.158, abs=5e-4)
        assert trump.z_vs_others == pytest.approx(-0.93, abs=5e-3)
        assert int(trump.n_windows_eligible) == 5
        assert trump.ci_status == "low_cluster_caution"
        # The pre-registered rule is MET while the p-value is NOT significant.
        # Both halves are load-bearing for how section 5 is phrased.
        assert trump.percentile_corrected <= 0.50
        assert trump.corrected_distance < others.corrected_distance.mean()
        assert trump.percentile_corrected > C.ALPHA

    @pytest.mark.parametrize(
        "arm,lo,hi,trump",
        [("corex_all", -0.2323, 0.0636, 0.0225),
         ("corex_sotu", 0.2241, 0.6118, 0.4101),
         ("llm_all", -0.0577, 0.2296, 0.1567)],
    )
    def test_the_jackknife_table(self, art, arm, lo, hi, trump):
        s = art["jack"][art["jack"].arm == arm]
        assert s.rho.min() == pytest.approx(lo, abs=5e-5)
        assert s.rho.max() == pytest.approx(hi, abs=5e-5)
        assert not s.significant_decline.any()
        row = s[s.left_out_president.str.contains("Trump")]
        assert len(row) == 1, "leave-Trump-out is pre-registered and must exist"
        assert row.rho.iloc[0] == pytest.approx(trump, abs=5e-5)

    def test_the_most_influential_deletion_is_named_correctly(self, art):
        s = art["jack"][art["jack"].arm == "corex_all"]
        assert "James Monroe" in s.loc[s.rho.idxmin()].left_out_president


# --------------------------------------------------------------------------
# section 6 -- the limitations, which are the part most likely to rot
# --------------------------------------------------------------------------
class TestSection6Limitations:
    def test_the_null_does_not_centre_on_the_estimators_own_bias(self, art):
        """Section 6.1's whole argument. If these two ever coincided, the
        correction would be wrong and the ORIGINAL (struck) claim right --
        so this test is what makes the correction falsifiable rather than
        merely asserted."""
        bias = art["meta"]["selftest"]["a_cluster_null_rho"]
        null_mean = _stat(art["null"], "corex_all", "dispersion").null_mean
        assert bias == pytest.approx(-0.0893, abs=5e-4)
        assert null_mean == pytest.approx(0.0142, abs=5e-4)
        assert abs(null_mean - bias) > 0.05, (
            "the null now centres near the estimator's bias; section 6.1 would "
            "need rewriting -- and the struck 'absorbs it' claim revisiting"
        )

    def test_the_worst_jackknife_retention(self, art):
        j = art["jack"]
        assert (j.n_windows_used / j.n_windows_full_design).min() == pytest.approx(
            0.913, abs=5e-4
        )

    def test_the_post_hoc_thresholds_are_the_ones_disclosed(self, note):
        assert f"= {C.JACKKNIFE_SUPPRESS_RETENTION}" in note.replace("`", "") or \
            str(C.JACKKNIFE_SUPPRESS_RETENTION) in note
        assert str(C.JACKKNIFE_CAUTION_RETENTION) in note

    @pytest.mark.parametrize(
        "arm,pct", [("corex_all", 59.0), ("corex_sotu", 48.6), ("llm_all", 65.1)]
    )
    def test_the_floor_is_not_time_constant(self, art, arm, pct):
        s = art["curves"]
        s = s[(s.arm == arm) & (s.window_kind == "rolling") & s.used_in_trend]
        drift = (s.floor.max() - s.floor.min()) / s.floor.min() * 100
        assert drift == pytest.approx(pct, abs=0.05)

    def test_every_plottable_row_still_carries_a_trust_gate(self, art):
        for name in ("curves", "null", "jack", "paradox"):
            assert "ci_status" in art[name].columns, name
            assert art[name].ci_status.notna().all(), name


# --------------------------------------------------------------------------
# section 7 -- the pre-registration is quoted, never edited
# --------------------------------------------------------------------------
class TestSection7PrereRegistrationIntegrity:
    def test_the_note_never_claims_the_prereg_was_amended(self, note):
        assert "never been amended" in note

    def test_the_three_disclosed_divergences_are_all_present(self, note):
        """Striking a claim is only honest if the strike is findable. Section 7
        is the single place a reader scoring the study can see all three."""
        for marker in ("drifting speech supply", "101", "216", "never imports"):
            assert marker in note, marker

    def test_the_zero_dollar_claim_is_scoped_to_the_process_not_the_file(self, note):
        """The prereg's file-level sentence must not be restated as a
        process-level one -- `anthropic` IS resident transitively."""
        assert "true of the *file* and false of the" in note
        assert "api_calls: 0" in note or "`api_calls: 0`" in note


class TestTheNoteDeclaresItsOwnBoundary:
    def test_it_says_which_numbers_are_not_re_derivable(self, note):
        """A note that declares its boundary is worth more than one implying
        completeness. `pair_regression` is the one published statistic a reader
        cannot check from the artifacts."""
        assert "pair_regression" in note
        assert "cannot** re-derive" in note or "cannot re-derive" in note

    def test_it_states_its_rounding_convention(self, note):
        assert "one step" in note and "Double-rounding" in note


class TestThePrintedStringsMatchTheArtifact:
    """The hole the first draft of this file had, found by mutation.

    Every test above pins an ARTIFACT value. That is necessary and not
    sufficient: the note could print "rank 5 of 38" while the artifact says 6
    and every one of them would still pass, because none of them reads the
    prose. Prose drift invisible to a reader of the note alone is the exact
    defect class this project keeps shipping -- so these assert the rendered
    SENTENCE, derived from the artifact rather than hardcoded.
    """

    def test_the_h2b_rank_sentence(self, art, note):
        trump = art["paradox"][art["paradox"].president.str.contains("Trump")].iloc[0]
        n = int(trump.n_presidents_ranked)
        assert f"rank {int(trump.rank_corrected)} of {n}" in note

    def test_the_h2a_anchor_sentence(self, art, note):
        px = art["paradox"]
        trump = px[px.president.str.contains("Trump")].iloc[0]
        assert f"**{trump.stylistic_similarity_anchor:.4f}**" in note
        # "the lowest of all 38" -- both the ordinal claim and the count
        assert f"lowest of all {len(px)}" in note
        # and the note counts the OTHERS correctly (38 presidents -> 37 others)
        assert f"other {len(px) - 1} presidents" in note

    def test_the_trump_window_sentence(self, art, note):
        trump = art["paradox"][art["paradox"].president.str.contains("Trump")].iloc[0]
        rolling = art["curves"]
        rolling = rolling[
            (rolling.arm == "corex_all") & (rolling.window_kind == "rolling")
        ]
        assert f"{int(trump.n_windows_eligible)} of {len(rolling)} windows" in note

    def test_the_coverage_sentence_for_the_sotu_arm(self, art, note):
        cov = art["meta"]["coverage"]["corex_sotu"]
        # The 92-vs-101 reconciliation is the note's most easily-garbled passage.
        assert f"**{cov['n_windows_used']}**" in note
        assert f"{cov['n_qualifying_speeches']} of 218" in note

    def test_the_worst_retention_sentence(self, art, note):
        j = art["jack"]
        worst = (j.n_windows_used / j.n_windows_full_design).min()
        assert f"**{worst:.3f}**" in note

    def test_the_headline_rho_and_critical_value_are_printed(self, art, note):
        row = _stat(art["null"], "corex_all", "dispersion")
        assert f"**{row.rho:.4f}**" in note.replace("−", "-") or \
            f"**−{abs(row.rho):.4f}**" in note
        assert f"−{abs(row.null_crit_05):.4f}" in note

    def test_the_pool_drift_spearmans_are_printed(self, art, note):
        """Section 6.1's mechanism rests on +0.85 vs +0.06. Those are measured
        against the design, not stored in an artifact, so they are pinned as
        printed strings and their SOURCE is named in the note."""
        assert "**+0.85**" in note and "**+0.06**" in note
