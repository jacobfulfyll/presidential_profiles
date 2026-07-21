"""The estimators behind every published number in `notes/combativeness-findings-v1.md`.

Four things are load-bearing here and each gets its own class:

1. **The n-floor.** Pre-registered policy: <5 speech clusters suppresses the
   interval entirely, 5-20 emits it with a caution flag. Drifting this changes
   which numbers the report is allowed to publish (the War & New Deal SOTU cell
   is n=3), so the boundaries are tested exactly, not approximately.

2. **Genre standardization.** The whole point of the estimator is that two eras
   with identical within-genre rates but sharply different genre *mixes* must
   come out equal. If that ever stops holding, the "raw vs within-genre"
   divergence section of the report is measuring nothing.

3. **The ratio bootstrap.** Ratios reuse the marginal resamples so a ratio
   interval can never tell a different story than the marginal intervals it is
   built from, and a suppressed cell must poison any ratio touching it.

4. **The inf-safe percentile.** REGRESSION: `np.percentile` interpolating
   between two infinite order statistics returns NaN, which would disguise an
   unbounded upper limit as a missing one. `test_infinite_upper_tail_reports_inf_not_nan`
   fails against that behaviour and demonstrates it inline.

All of it runs on hand-authored frames of a few dozen rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import combat as C

SOTU = C.SOTU_TYPE
RALLY = "public_remarks_or_address"


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hits = frame[mask]
    assert len(hits) == 1, f"{where} matched {len(hits)} rows, expected 1"
    return hits.iloc[0]


def _cell(df: pd.DataFrame, era: str) -> pd.DataFrame:
    return df[df["era"] == era]


# --------------------------------------------------------------------------- #
# 1. the n-floor policy
# --------------------------------------------------------------------------- #
class TestNFloorPolicy:
    def test_floor_constants_match_the_preregistered_policy(self):
        """Anchored as literals, decoupled from the module: a test parametrized
        over the constant itself would happily follow it wherever it moved.
        These two numbers are quoted in notes/combativeness-findings-v1.md §8.1
        as pre-registered policy."""
        assert C.MIN_CLUSTERS_FOR_CI == 5
        assert C.LOW_CLUSTER_CAUTION == 20

    @pytest.mark.parametrize(
        "n_speeches,expected",
        [
            (0, "no_data"),
            (1, "suppressed_n_floor"),
            (4, "suppressed_n_floor"),  # boundary: last suppressed
            (5, "low_cluster_caution"),  # boundary: first interval emitted
            (19, "low_cluster_caution"),  # boundary: last flagged
            (20, "ok"),  # boundary: first unflagged
            (21, "ok"),
        ],
    )
    def test_ci_status_boundaries(self, n_speeches, expected):
        assert C._ci_status(n_speeches) == expected

    def test_four_clusters_produce_no_bootstrap_draws_at_all(
        self, combat_inputs, combat_paras
    ):
        """Suppression happens at the fit, not at the reporting layer — no draws
        exist to be accidentally re-used downstream (by the ratio table, say)."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 2020, "paras": combat_paras(10, 3)}
                    for k in range(4)
                ]
            )
        )
        fit = C._fit_simple(_cell(df, "The present era"), (0, 0))
        assert fit.n_speeches == 4
        assert fit.draws is None
        assert fit.rates["party_attack"] == pytest.approx(0.3)

    def test_five_clusters_produce_draws(self, combat_inputs, combat_paras):
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 2020, "paras": combat_paras(10, 3)}
                    for k in range(5)
                ]
            )
        )
        fit = C._fit_simple(_cell(df, "The present era"), (0, 0))
        assert fit.n_speeches == 5
        assert fit.draws is not None
        assert fit.draws["party_attack"].shape == (C.N_BOOTSTRAP,)

    def test_a_suppressed_cell_still_reports_its_point_estimate(
        self, combat_inputs, combat_paras
    ):
        """End to end: the rate survives (it is a real count), the interval does
        not, and ci_status says which. Publishing the point without the flag is
        exactly the 1930s mistake the policy exists to prevent."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 2020, "paras": combat_paras(10, 3)}
                    for k in range(4)
                ]
            )
        )
        row = _one(
            C.rate_table(df, "era", ["The present era"], None),
            era="The present era",
            treatment="raw",
            flag="party_attack",
        )
        assert row["rate"] == pytest.approx(0.3)
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        assert row["ci_status"] == "suppressed_n_floor"
        assert row["n_speeches"] == 4
        assert row["min_clusters_for_ci"] == 5

    def test_twenty_clusters_clear_the_caution_flag_end_to_end(
        self, combat_inputs, combat_paras
    ):
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 2020, "paras": combat_paras(5, k % 3)}
                    for k in range(20)
                ]
            )
        )
        row = _one(
            C.rate_table(df, "era", ["The present era"], None),
            era="The present era",
            treatment="raw",
            flag="party_attack",
        )
        assert row["n_speeches"] == 20
        assert row["ci_status"] == "ok"
        assert row["ci_lo"] < row["rate"] < row["ci_hi"]


# --------------------------------------------------------------------------- #
# 2. genre standardization
# --------------------------------------------------------------------------- #
def _mix_shift_inputs(combat_inputs, combat_paras, extra_genre=False):
    """Two eras, identical WITHIN-genre rates, opposite genre mixes.

    Both eras: SOTU paragraphs are flagged at 0.10, rally paragraphs at 0.50.
    Expansion is 2:1 SOTU-heavy, the present era is 1:2 rally-heavy. So the raw
    rates must differ and the standardized rates must not.

    With ``extra_genre`` an inaugural stratum exists only in Expansion, which is
    what makes ``ref_weight_covered`` drop below 1.0 for the present era.
    """
    specs = []
    for k in range(3):
        specs += [
            {"doc": f"e1-sotu{k}", "year": 1820, "type": SOTU, "paras": combat_paras(20, 2)},
            {"doc": f"e1-rally{k}", "year": 1820, "type": RALLY, "paras": combat_paras(10, 5)},
            {"doc": f"e2-sotu{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 1)},
            {"doc": f"e2-rally{k}", "year": 2020, "type": RALLY, "paras": combat_paras(20, 10)},
        ]
        if extra_genre:
            specs.append(
                {
                    "doc": f"e1-inaug{k}",
                    "year": 1820,
                    "type": "inaugural_address",
                    "paras": combat_paras(30, 3),
                }
            )
    return combat_inputs(specs)


class TestGenreStandardization:
    ORDER = ["Expansion", "The present era"]

    def test_raw_rates_diverge_when_only_the_genre_mix_differs(
        self, combat_inputs, combat_paras
    ):
        """The premise of the next test. If raw did NOT diverge here the fixture
        would not pose the confound and the standardization test would be
        vacuous."""
        df = C.load_frame(**_mix_shift_inputs(combat_inputs, combat_paras))
        raw = C.rate_table(df, "era", self.ORDER, None)
        e1 = _one(raw, era="Expansion", treatment="raw", flag="party_attack")
        e2 = _one(raw, era="The present era", treatment="raw", flag="party_attack")
        assert e1["rate"] == pytest.approx(21 / 90)  # 0.2333
        assert e2["rate"] == pytest.approx(33 / 90)  # 0.3667
        assert e2["rate"] > e1["rate"] + 0.1

    def test_identical_within_genre_rates_standardize_to_the_same_number(
        self, combat_inputs, combat_paras
    ):
        """THE point of the estimator: hold the mix fixed and a pure composition
        difference disappears. Both eras must land on
        0.5*0.10 + 0.5*0.50 = 0.30 exactly."""
        df = C.load_frame(**_mix_shift_inputs(combat_inputs, combat_paras))
        out = C.rate_table(df, "era", self.ORDER, None)
        e1 = _one(out, era="Expansion", treatment="genre_standardized", flag="party_attack")
        e2 = _one(
            out, era="The present era", treatment="genre_standardized", flag="party_attack"
        )
        assert e1["rate"] == pytest.approx(0.30)
        assert e2["rate"] == pytest.approx(0.30)
        assert e1["rate"] == pytest.approx(e2["rate"])

    def test_full_reference_coverage_is_reported_as_one(self, combat_inputs, combat_paras):
        df = C.load_frame(**_mix_shift_inputs(combat_inputs, combat_paras))
        out = C.rate_table(df, "era", self.ORDER, None)
        for era in self.ORDER:
            row = _one(out, era=era, treatment="genre_standardized", flag="party_attack")
            assert row["ref_weight_covered"] == pytest.approx(1.0)

    def test_a_missing_stratum_is_reported_as_reduced_reference_coverage(
        self, combat_inputs, combat_paras
    ):
        """`ref_weight_covered` is the honesty valve: an era standardized over
        two thirds of the reference mix must say so, because its standardized
        rate is not strictly comparable with a full-coverage era's."""
        df = C.load_frame(**_mix_shift_inputs(combat_inputs, combat_paras, extra_genre=True))
        reference = C.genre_reference(df)
        assert reference.round(6).to_dict() == {
            SOTU: pytest.approx(1 / 3),
            RALLY: pytest.approx(1 / 3),
            "inaugural_address": pytest.approx(1 / 3),
        }

        out = C.rate_table(df, "era", self.ORDER, None)
        covered = _one(out, era="Expansion", treatment="genre_standardized", flag="party_attack")
        partial = _one(
            out, era="The present era", treatment="genre_standardized", flag="party_attack"
        )
        assert covered["ref_weight_covered"] == pytest.approx(1.0)
        assert partial["ref_weight_covered"] == pytest.approx(2 / 3)
        # ...and the surviving strata are renormalized, so the rate is still the
        # 50/50 blend of the two genres it does cover.
        assert partial["rate"] == pytest.approx(0.30)

    def test_a_stratum_below_the_speech_floor_leaves_both_the_rate_and_coverage(
        self, combat_inputs, combat_paras
    ):
        """A genre rate estimated off two speeches is two speeches, not a genre.
        It must drop out of the weighted average AND out of ref_weight_covered —
        counting it in coverage while excluding it from the rate would overstate
        how much of the reference mix the estimate spans."""
        assert C.MIN_CELL_SPEECHES == 3
        specs = [
            {"doc": f"sotu{k}", "year": 1820, "type": SOTU, "paras": combat_paras(10, 1)}
            for k in range(3)
        ] + [
            {"doc": f"rally{k}", "year": 1820, "type": RALLY, "paras": combat_paras(10, 10)}
            for k in range(2)
        ]
        df = C.load_frame(**combat_inputs(specs))
        assert C.genre_reference(df)[RALLY] == pytest.approx(0.4)

        row = _one(
            C.rate_table(df, "era", ["Expansion"], None),
            era="Expansion",
            treatment="genre_standardized",
            flag="party_attack",
        )
        assert row["rate"] == pytest.approx(0.1)  # the rally stratum is gone
        assert row["ref_weight_covered"] == pytest.approx(0.6)
        assert row["n_speeches"] == 3

    def test_no_qualifying_stratum_yields_nan_rates_and_zero_coverage(
        self, combat_inputs, combat_paras
    ):
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 1820, "type": f"genre{k}", "paras": combat_paras(4, 1)}
                    for k in range(3)
                ]
            )
        )
        fit = C._fit_standardized(
            _cell(df, "Expansion"), C.genre_reference(df), (0, 2)
        )
        assert np.isnan(fit.rates["party_attack"])
        assert fit.ref_weight_covered == 0.0
        assert fit.draws is None
        assert fit.n_speeches == 0

    def test_a_genre_absent_from_the_reference_contributes_nothing(
        self, combat_inputs, combat_paras
    ):
        """A stratum with no reference weight cannot be reweighted onto the
        reference distribution, so it is skipped. Exercised directly with a
        trimmed reference because `_fit_all` always derives the reference from
        the same frame — this branch is otherwise unreachable and would silently
        drop a genre if it ever became reachable."""
        specs = [
            {"doc": f"sotu{k}", "year": 1820, "type": SOTU, "paras": combat_paras(10, 1)}
            for k in range(3)
        ] + [
            {"doc": f"rally{k}", "year": 1820, "type": RALLY, "paras": combat_paras(10, 10)}
            for k in range(3)
        ]
        df = C.load_frame(**combat_inputs(specs))
        trimmed = C.genre_reference(df).drop(RALLY)

        fit = C._fit_standardized(_cell(df, "Expansion"), trimmed, (0, 2))

        assert fit.rates["party_attack"] == pytest.approx(0.1)
        assert fit.n_speeches == 3
        assert fit.ref_weight_covered == pytest.approx(0.5)

    def test_the_standardized_interval_uses_the_reference_weights_not_equal_ones(
        self, combat_inputs, combat_paras
    ):
        """MUTATION-CHECK FINDING: nothing pinned the WEIGHTS inside the draws.

        Every existing standardization test asserts the point `rate`; none
        asserts anything about the stratified bootstrap's own weighting. So
        reweighting the draws equally across genres — while leaving the point
        estimate correct — published an interval centred somewhere the point
        estimate is not, and no test noticed.

        The fixture makes the two weightings maximally different: the reference
        mix is 80/20 by paragraph share while the genres are 50/50 by speech
        count, and the within-genre rates are 0.10 and 0.80. Reference-weighted
        gives 0.8*0.10 + 0.2*0.80 = 0.24; equal-weighted would give 0.45. Every
        speech inside a genre is identical, so each resample reproduces its
        genre's rate exactly and the interval collapses onto the point — which
        makes the assertion exact rather than "roughly centred"."""
        # 5 speeches per genre, not 4: each stratum must clear
        # MIN_CLUSTERS_FOR_CI or the thin-stratum rule suppresses the interval
        # this test exists to inspect (see
        # ``TestThinStratumFloor``). The 80/20 reference split is unaffected —
        # 5*20 and 5*5 paragraphs are still 100 and 25.
        specs = [
            {"doc": f"sotu{k}", "year": 1820, "type": SOTU, "paras": combat_paras(20, 2)}
            for k in range(5)
        ] + [
            {"doc": f"rally{k}", "year": 1820, "type": RALLY, "paras": combat_paras(5, 4)}
            for k in range(5)
        ]
        df = C.load_frame(**combat_inputs(specs))
        reference = C.genre_reference(df)
        assert reference[SOTU] == pytest.approx(0.8)
        assert reference[RALLY] == pytest.approx(0.2)

        row = _one(
            C.rate_table(df, "era", ["Expansion"], None),
            era="Expansion",
            treatment="genre_standardized",
            flag="party_attack",
        )
        assert row["rate"] == pytest.approx(0.24)
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((0.24, 0.24))
        assert row["n_speeches"] == 10
        assert row["ci_status"] == "low_cluster_caution"
        assert row["weight_from_thin_strata"] == pytest.approx(0.0)

    def test_unstandardized_treatments_report_no_reference_coverage(
        self, combat_inputs, combat_paras
    ):
        """`ref_weight_covered` is only meaningful for `genre_standardized`.

        MUTATION-CHECK FINDING: no test read this column on the other two
        treatments, so `raw` and `sotu_only` could claim `1.0` — "standardized
        over the full reference mix" — for estimates that were never
        standardized at all. NaN is the honest value and it is what the
        published parquet must carry."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 3)}
                    for k in range(6)
                ]
            )
        )
        out = C.rate_table(df, "era", ["The present era"], None)
        for treatment in ("raw", "sotu_only"):
            row = _one(
                out, era="The present era", treatment=treatment, flag="party_attack"
            )
            assert np.isnan(row["ref_weight_covered"]), treatment
        standardized = _one(
            out, era="The present era", treatment="genre_standardized", flag="party_attack"
        )
        assert standardized["ref_weight_covered"] == pytest.approx(1.0)

    def test_genre_reference_is_the_pooled_paragraph_share(self, combat_inputs, combat_paras):
        """Paragraph share, not speech share, and pooled over the whole corpus
        so no single era is privileged as "the normal mix"."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a", "year": 1820, "type": SOTU, "paras": combat_paras(30)},
                    {"doc": "b", "year": 2020, "type": RALLY, "paras": combat_paras(10)},
                ]
            )
        )
        reference = C.genre_reference(df)
        assert reference[SOTU] == pytest.approx(0.75)
        assert reference[RALLY] == pytest.approx(0.25)
        assert reference.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. the inf-safe percentile (REGRESSION)
# --------------------------------------------------------------------------- #
class TestPercentileCI:
    def test_finite_draws_get_an_interpolated_interval(self):
        draws = np.arange(0.0, 1000.0)
        lo, hi = C._percentile_ci(draws)
        expected = np.percentile(draws, [2.5, 97.5], method="linear")
        assert (lo, hi) == pytest.approx(tuple(expected))

    def test_infinite_upper_tail_reports_inf_not_nan(self):
        """REGRESSION. A ratio resample can zero the denominator, making the
        draw infinite. Linear interpolation between two infinite order
        statistics computes inf - inf = NaN, so an UNBOUNDED upper limit would
        be published as a MISSING one — two very different claims. The fix
        switches to nearest-rank whenever any draw is infinite.

        The first assertion below is the old behaviour, asserted directly so
        this test is self-evidently a regression test and not a tautology."""
        draws = np.array([1.0] * 1400 + [np.inf] * 600)

        with warnings.catch_warnings():  # inf - inf warns; that IS the old bug
            warnings.simplefilter("ignore", RuntimeWarning)
            old = np.percentile(draws, 97.5, method="linear")
        assert np.isnan(old), "fixture no longer reproduces the NaN bug"

        lo, hi = C._percentile_ci(draws)
        assert np.isinf(hi) and hi > 0
        assert not np.isnan(hi)
        assert lo == pytest.approx(1.0)

    def test_a_lower_bound_stays_finite_when_only_the_upper_tail_is_infinite(self):
        draws = np.concatenate([np.linspace(1.0, 5.0, 1900), np.full(100, np.inf)])
        lo, hi = C._percentile_ci(draws)
        assert np.isfinite(lo)
        assert np.isinf(hi)

    def test_an_all_infinite_draw_vector_reports_an_infinite_interval(self):
        lo, hi = C._percentile_ci(np.full(100, np.inf))
        assert np.isinf(lo) and np.isinf(hi)

    def test_ci_level_is_a_two_sided_95_percent_interval(self):
        assert C.CI_LEVEL == 0.95
        draws = np.arange(0.0, 1000.0)
        lo, hi = C._percentile_ci(draws)
        assert lo == pytest.approx(np.percentile(draws, 2.5))
        assert hi == pytest.approx(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------- #
# 3b. widening a ratio interval by the annotator-disagreement bands
#
# MUTATION-CHECK FINDING: `_widen_ratio` and the populated branch of
# `_band_lookup` were the ONLY uncovered logic in the module (17 statements) —
# no test ever called `ratio_table(..., bands=<not None>)`. Every mutation to
# them survived: deleting the widening entirely, inverting its direction,
# emptying the lookup. That matters because `notes/combativeness-findings-v1.md`
# promises the quoted headline "4.8x [2.3, 14.3]" will widen along with the
# marginals once `inter-model-agreement-check` lands, and the seam is the
# deliverable of the deferred acceptance criterion. Unit-tested here on exact,
# hand-computable arithmetic; driven end to end in test_combat_contracts.py.
# --------------------------------------------------------------------------- #
class TestWidenRatio:
    LO, HI = 2.0, 4.0
    NUM, DEN = 0.5, 0.25

    def _widen(self, h_num, h_den):
        return C._widen_ratio(self.LO, self.HI, self.NUM, self.DEN, h_num, h_den)

    def test_no_band_on_either_side_is_the_identity(self):
        assert self._widen(None, None) == (self.LO, self.HI)

    def test_a_zero_half_width_is_exactly_the_identity(self):
        """The docstring's continuity claim, and the reason the widening scales
        the bootstrapped bounds instead of dividing two widened marginal
        endpoints: the interval must not jump the moment a band file appears
        carrying a negligible disagreement."""
        assert self._widen(0.0, 0.0) == pytest.approx((self.LO, self.HI))

    def test_a_numerator_band_moves_both_bounds_outward_by_its_own_factor(self):
        # 0.5 +/- 0.1 -> factors 0.8 and 1.2
        assert self._widen(0.1, None) == pytest.approx((1.6, 4.8))

    def test_a_denominator_band_moves_the_bounds_the_OPPOSITE_way(self):
        """A larger denominator shrinks the ratio, so the denominator's band
        lowers the LOWER bound and raises the upper one — the mirror of the
        numerator's. Getting this backwards would narrow the published interval
        while claiming to have widened it."""
        # 0.25 -> lower factor 0.25/0.30, upper factor 0.25/0.20
        assert self._widen(None, 0.05) == pytest.approx((2 * (0.25 / 0.30), 4 * 1.25))

    def test_bands_on_both_sides_compose_multiplicatively(self):
        assert self._widen(0.1, 0.05) == pytest.approx(
            (2 * 0.8 * (0.25 / 0.30), 4 * 1.2 * 1.25)
        )

    def test_widening_never_narrows_the_interval(self):
        for h_num, h_den in [(0.1, None), (None, 0.05), (0.1, 0.05), (0.02, 0.02)]:
            lo, hi = self._widen(h_num, h_den)
            assert lo <= self.LO, (h_num, h_den)
            assert hi >= self.HI, (h_num, h_den)

    def test_a_band_that_can_zero_the_denominator_reports_an_unbounded_upper_bound(self):
        """Honest rather than clipped: if annotator disagreement could take the
        denominator rate to zero, the ratio is genuinely unbounded."""
        lo, hi = self._widen(None, 0.25)
        assert np.isinf(hi)
        assert lo == pytest.approx(2 * (0.25 / 0.50))

    def test_a_band_wider_than_the_numerator_rate_floors_the_lower_bound_at_zero(self):
        lo, hi = self._widen(0.6, None)
        assert lo == 0.0
        assert hi == pytest.approx(4 * (0.5 + 0.6) / 0.5)

    def test_a_zero_numerator_rate_is_left_alone(self):
        """A ratio of 0/x is 0 however the numerator's band is drawn; scaling
        would divide by zero."""
        assert C._widen_ratio(0.0, 0.0, 0.0, 0.25, 0.1, None) == pytest.approx((0.0, 0.0))

    def test_a_zero_denominator_rate_with_a_band_is_unbounded_above(self):
        lo, hi = C._widen_ratio(1.0, 2.0, 0.5, 0.0, None, 0.1)
        assert lo == pytest.approx(1.0)
        assert np.isinf(hi)

    def test_a_suppressed_interval_is_never_manufactured_by_a_band(self):
        """The n-floor's decision outranks the seam. Widening NaN must stay
        NaN, or a cell the policy refused to interval would reappear in the
        report wearing an annotator band."""
        lo, hi = C._widen_ratio(np.nan, np.nan, 0.5, 0.25, 0.1, 0.1)
        assert np.isnan(lo) and np.isnan(hi)

    def test_an_already_infinite_upper_bound_stays_infinite(self):
        lo, hi = C._widen_ratio(2.0, np.inf, 0.5, 0.25, 0.1, 0.05)
        assert np.isinf(hi)
        assert np.isfinite(lo)


class TestBandLookup:
    def test_no_band_table_yields_an_empty_lookup(self):
        assert C._band_lookup(None) == {}

    def test_each_era_flag_pair_maps_to_its_half_width(self):
        bands = pd.DataFrame(
            [
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": "Expansion", "flag": "zero_sum", "disagreement_half_width": 0.09},
            ]
        )
        assert C._band_lookup(bands) == {
            ("Expansion", "party_attack"): 0.05,
            ("Expansion", "zero_sum"): 0.09,
        }

    def test_a_missing_half_width_is_omitted_rather_than_read_as_zero(self):
        """A NaN half-width means "not measured", which must leave the row
        `sampling_only` — not "measured as exactly zero disagreement"."""
        bands = pd.DataFrame(
            [
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": np.nan},
                {"era": "Expansion", "flag": "zero_sum", "disagreement_half_width": 0.09},
            ]
        )
        assert C._band_lookup(bands) == {("Expansion", "zero_sum"): 0.09}


# --------------------------------------------------------------------------- #
# 4. the ratio bootstrap
# --------------------------------------------------------------------------- #
def _two_era_inputs(combat_inputs, combat_paras, n_founding=5, founding_flagged=1):
    """Present era at 0.30, founding era with the flag concentrated in ONE
    speech so a large share of resamples zero the denominator."""
    specs = [
        {"doc": f"pres{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 3)}
        for k in range(5)
    ]
    specs += [
        {
            "doc": f"found{k}",
            "year": 1800,
            "type": SOTU,
            "paras": combat_paras(10, founding_flagged if k == 0 else 0),
        }
        for k in range(n_founding)
    ]
    return combat_inputs(specs)


class TestRatioBootstrap:
    ORDER = ["The founding", "The present era"]

    def test_reference_era_is_the_present_era(self):
        assert C.REFERENCE_ERA == "The present era"
        assert C.REFERENCE_ERA in C.ERA_ORDER

    def test_ratio_point_equals_the_two_marginal_rates_it_reports(
        self, combat_inputs, combat_paras
    ):
        """The report quotes "4.8x the Civil War rate" next to both marginal
        rates; if the ratio were recomputed on a different frame the three
        numbers could drift apart in print."""
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        rates = C.rate_table(df, "era", self.ORDER, None)
        ratios = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA)

        row = _one(
            ratios, denominator="The founding", treatment="raw", flag="party_attack"
        )
        num = _one(rates, era="The present era", treatment="raw", flag="party_attack")
        den = _one(rates, era="The founding", treatment="raw", flag="party_attack")

        assert row["numerator_rate"] == pytest.approx(num["rate"])
        assert row["denominator_rate"] == pytest.approx(den["rate"])
        assert row["ratio"] == pytest.approx(num["rate"] / den["rate"])
        assert row["n_speeches_numerator"] == num["n_speeches"]
        assert row["n_speeches_denominator"] == den["n_speeches"]

    def test_ratio_interval_is_built_from_the_same_resamples_as_the_marginals(
        self, combat_inputs, combat_paras
    ):
        """The invariant that makes a ratio interval unable to contradict a
        marginal one: both read the SAME `_CellFit.draws` arrays. Reconstructed
        here from the fits, element by element."""
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        fits = C._fit_all(df, "era", self.ORDER)
        rates = C.rate_table(df, "era", self.ORDER, None, fits=fits)
        ratios = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA, fits=fits)

        num_draws = fits[("The present era", "raw")].draws["enemy_naming"]
        den_draws = fits[("The founding", "raw")].draws["enemy_naming"]

        marginal = _one(rates, era="The present era", treatment="raw", flag="enemy_naming")
        assert (marginal["ci_lo"], marginal["ci_hi"]) == pytest.approx(
            C._percentile_ci(num_draws)
        )

        expected = C._percentile_ci(
            np.where(den_draws > 0, num_draws / np.where(den_draws > 0, den_draws, 1), np.inf)
        )
        row = _one(ratios, denominator="The founding", treatment="raw", flag="enemy_naming")
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx(expected, nan_ok=True)

    def test_the_ratio_draws_are_paired_elementwise_with_the_marginal_draws(
        self, combat_inputs, combat_paras
    ):
        """MUTATION-CHECK FINDING: the test above cannot see a reshuffle.

        `_two_era_inputs` never sets `enemy_naming`, so BOTH cells' draw vectors
        for that flag are identically zero and the reconstructed ratio is the
        all-infinite vector — every element equal, so any permutation of the
        resamples reproduces it exactly. A `ratio_table` that drew FRESH
        resamples instead of reusing the marginals' passed that test untouched.

        This one uses two cells whose speeches differ from one another, so the
        draw vectors are non-degenerate, and then proves the elementwise pairing
        is load-bearing: permuting either side's draws — statistically the same
        distribution, since the two cells hold disjoint speeches — moves the
        interval. That is exactly the divergence that would let a published ratio
        interval contradict the marginal intervals printed beside it."""
        specs = [
            {"doc": f"p{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, k)}
            for k in range(1, 7)
        ] + [
            {"doc": f"f{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, k % 3)}
            for k in range(1, 7)
        ]
        df = C.load_frame(**combat_inputs(specs))
        fits = C._fit_all(df, "era", self.ORDER)
        num = fits[("The present era", "raw")].draws["party_attack"]
        den = fits[("The founding", "raw")].draws["party_attack"]

        # anti-vacuity: neither vector may be constant, or a permutation is a no-op
        assert num.std() > 0.0, "numerator draws are degenerate — test would be vacuous"
        assert den.std() > 0.0, "denominator draws are degenerate — test would be vacuous"

        def ratio_of(n, d):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(d > 0, n / np.where(d > 0, d, 1), np.inf)

        expected = C._percentile_ci(ratio_of(num, den))
        row = _one(
            C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA, fits=fits),
            denominator="The founding",
            treatment="raw",
            flag="party_attack",
        )
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx(expected)

        # ...and the pairing matters: break it and the interval moves.
        reshuffled_den = C._percentile_ci(
            ratio_of(num, np.random.default_rng(0).permutation(den))
        )
        reshuffled_num = C._percentile_ci(
            ratio_of(np.random.default_rng(1).permutation(num), den)
        )
        assert reshuffled_den[0] != pytest.approx(expected[0])
        assert reshuffled_num[0] != pytest.approx(expected[0])

    def test_refitting_independently_reproduces_the_shared_fit_results(
        self, combat_inputs, combat_paras
    ):
        """`build_combativeness` fits once and hands the same dict to both
        tables; a caller may instead let each table fit for itself. Per-cell
        seeding means the two routes must agree exactly — otherwise the
        published parquet and an ad-hoc recomputation would disagree."""
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        fits = C._fit_all(df, "era", self.ORDER)

        shared = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA, fits=fits)
        refit = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA)
        pd.testing.assert_frame_equal(shared, refit)

        shared_rates = C.rate_table(df, "era", self.ORDER, None, fits=fits)
        refit_rates = C.rate_table(df, "era", self.ORDER, None)
        pd.testing.assert_frame_equal(shared_rates, refit_rates)

    def test_a_denominator_below_the_n_floor_suppresses_the_ratio(
        self, combat_inputs, combat_paras
    ):
        """The n-floor has to propagate. A suppressed marginal that silently
        acquired a ratio interval would let an uninterval-able cell back into
        the report through the ratio table — exactly the War & New Deal case."""
        df = C.load_frame(
            **_two_era_inputs(combat_inputs, combat_paras, n_founding=4, founding_flagged=3)
        )
        rates = C.rate_table(df, "era", self.ORDER, None)
        ratios = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA)

        marginal = _one(rates, era="The founding", treatment="raw", flag="party_attack")
        assert marginal["ci_status"] == "suppressed_n_floor"

        row = _one(ratios, denominator="The founding", treatment="raw", flag="party_attack")
        assert row["ci_status"] == "suppressed_n_floor"
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        assert np.isnan(row["pct_denominator_zero"])
        # the point ratio survives — it is a ratio of two real counts
        assert row["ratio"] == pytest.approx(0.3 / 0.075)

    def test_a_numerator_below_the_n_floor_suppresses_every_ratio(
        self, combat_inputs, combat_paras
    ):
        specs = [
            {"doc": f"pres{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 3)}
            for k in range(4)
        ] + [
            {"doc": f"found{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, 1)}
            for k in range(8)
        ]
        df = C.load_frame(**combat_inputs(specs))
        ratios = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA)
        assert (ratios["ci_status"] == "suppressed_n_floor").all()
        assert ratios["ci_lo"].isna().all()

    def test_a_zeroed_denominator_reports_an_unbounded_upper_bound(
        self, combat_inputs, combat_paras
    ):
        """REGRESSION, end to end. The founding cell's single flagged paragraph
        lives in one speech, so ~(4/5)^5 = 33% of resamples drop it and the
        ratio is genuinely infinite. That must publish as [lo, inf), never NaN,
        and the share of zeroed replicates must be reported alongside."""
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        row = _one(
            C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA),
            denominator="The founding",
            treatment="raw",
            flag="party_attack",
        )
        assert row["pct_denominator_zero"] == pytest.approx(0.328, abs=0.05)
        assert np.isfinite(row["ci_lo"])
        assert np.isinf(row["ci_hi"])
        assert not np.isnan(row["ci_hi"])
        assert row["ci_status"] == "low_cluster_caution"

    def test_pct_denominator_zero_is_zero_when_every_speech_carries_the_flag(
        self, combat_inputs, combat_paras
    ):
        df = C.load_frame(
            **_two_era_inputs(combat_inputs, combat_paras, n_founding=6, founding_flagged=2)
        )
        specs_all_flagged = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"pres{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 3)}
                    for k in range(5)
                ]
                + [
                    {"doc": f"found{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, 1)}
                    for k in range(5)
                ]
            )
        )
        row = _one(
            C.ratio_table(specs_all_flagged, "era", self.ORDER, C.REFERENCE_ERA),
            denominator="The founding",
            treatment="raw",
            flag="party_attack",
        )
        assert row["pct_denominator_zero"] == 0.0
        assert np.isfinite(row["ci_hi"])
        assert row["ratio"] == pytest.approx(3.0)
        assert row["ci_lo"] <= row["ratio"] <= row["ci_hi"]

    def test_the_thinner_cell_governs_the_ratio_caution_flag(
        self, combat_inputs, combat_paras
    ):
        """A ratio is only as trustworthy as its thinner side. The published
        headline divides the present era's 10-speech SOTU cell by eras holding
        27-34 speeches; taking the caution flag from the *better*-populated cell
        would label that comparison `ok` and hide precisely the fragility the
        flag exists to advertise."""
        specs = [
            {"doc": f"pres{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 3)}
            for k in range(6)
        ] + [
            {"doc": f"found{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, 1)}
            for k in range(25)
        ]
        df = C.load_frame(**combat_inputs(specs))
        rates = C.rate_table(df, "era", self.ORDER, None)
        assert (
            _one(rates, era="The founding", treatment="raw", flag="party_attack")["ci_status"]
            == "ok"
        )

        row = _one(
            C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA),
            denominator="The founding",
            treatment="raw",
            flag="party_attack",
        )
        assert row["n_speeches_numerator"] == 6
        assert row["n_speeches_denominator"] == 25
        assert row["ci_status"] == "low_cluster_caution"

    def test_a_ratio_between_two_well_populated_cells_is_ok(
        self, combat_inputs, combat_paras
    ):
        specs = [
            {"doc": f"pres{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 3)}
            for k in range(22)
        ] + [
            {"doc": f"found{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, 1)}
            for k in range(25)
        ]
        df = C.load_frame(**combat_inputs(specs))
        row = _one(
            C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA),
            denominator="The founding",
            treatment="raw",
            flag="party_attack",
        )
        assert row["ci_status"] == "ok"

    def test_the_default_order_is_the_full_era_axis(self, combat_inputs, combat_paras):
        """Omitting `order` must compare the reference era against all eight
        others, not against whichever happen to appear in the frame."""
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        ratios = C.ratio_table(df)
        assert set(ratios["denominator"]) == set(C.ERA_ORDER) - {C.REFERENCE_ERA}
        assert len(ratios) == 8 * len(C.FLAGS) * len(C.TREATMENTS)

    def test_a_reference_group_outside_the_order_raises(self, combat_inputs, combat_paras):
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        with pytest.raises(ValueError, match="not in"):
            C.ratio_table(df, "era", self.ORDER, "The Gilded Age")

    def test_the_reference_era_is_never_its_own_denominator(
        self, combat_inputs, combat_paras
    ):
        df = C.load_frame(**_two_era_inputs(combat_inputs, combat_paras))
        ratios = C.ratio_table(df, "era", self.ORDER, C.REFERENCE_ERA)
        assert set(ratios["numerator"]) == {C.REFERENCE_ERA}
        assert C.REFERENCE_ERA not in set(ratios["denominator"])
        assert len(ratios) == 1 * len(C.FLAGS) * len(C.TREATMENTS)


# --------------------------------------------------------------------------- #
# 5. determinism
# --------------------------------------------------------------------------- #
def _heterogeneous_present(combat_paras):
    """Six present-era speeches with DIFFERENT flag counts, so the clustered
    bootstrap produces a non-degenerate interval. A homogeneous cell yields
    [rate, rate] for every resample and would make the determinism tests below
    pass no matter how the RNG was wired."""
    return [
        {"doc": f"p{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, k)}
        for k in range(1, 7)
    ]


class TestDeterminism:
    def test_bootstrap_constants_are_pinned(self):
        assert C.N_BOOTSTRAP == 2000
        assert C.BOOTSTRAP_SEED == 20260721

    def test_each_cell_draws_from_its_own_random_stream(self):
        """`_cell_rng` mixes the cell's (group, treatment) coordinates INTO the
        seed. One generator seeded on BOOTSTRAP_SEED alone would hand every cell
        the identical index matrix, so two equally-sized cells would resample in
        lockstep — and the ratio bootstrap explicitly assumes its numerator and
        denominator cells are independent."""
        size = 64
        base = C._cell_rng(0, 0).integers(0, 100, size=size)
        assert not np.array_equal(base, C._cell_rng(1, 0).integers(0, 100, size=size))
        assert not np.array_equal(base, C._cell_rng(0, 1).integers(0, 100, size=size))
        assert np.array_equal(base, C._cell_rng(0, 0).integers(0, 100, size=size))

    def test_two_identically_shaped_cells_resample_independently(
        self, combat_inputs, combat_paras
    ):
        """The same property at the fit level, on two eras whose data is
        identical row for row: their bootstrap draw vectors must still differ,
        because the resample indices come from different streams."""
        specs = []
        for year in (1800, 2020):
            for k in range(1, 7):
                specs.append(
                    {
                        "doc": f"{year}-{k}",
                        "year": year,
                        "type": SOTU,
                        "paras": combat_paras(10, k),
                    }
                )
        df = C.load_frame(**combat_inputs(specs))
        fits = C._fit_all(df, "era", ["The founding", "The present era"])
        a = fits[("The founding", "raw")].draws["party_attack"]
        b = fits[("The present era", "raw")].draws["party_attack"]

        assert a.shape == b.shape
        assert np.mean(a) == pytest.approx(np.mean(b), abs=0.02)  # same population
        assert not np.array_equal(a, b)  # ...but not the same resamples

    def test_repeated_runs_are_identical(self, combat_inputs, combat_paras):
        df = C.load_frame(**combat_inputs(_heterogeneous_present(combat_paras)))
        first = C.rate_table(df, "era", ["The present era"], None)
        second = C.rate_table(df, "era", ["The present era"], None)
        pd.testing.assert_frame_equal(first, second)

    def test_the_interval_under_test_is_not_degenerate(self, combat_inputs, combat_paras):
        """Guard for the two tests below: if the bootstrap interval collapsed to
        a point, an order- or seed-dependence bug could not show up in them."""
        df = C.load_frame(**combat_inputs(_heterogeneous_present(combat_paras)))
        row = _one(
            C.rate_table(df, "era", ["The present era"], None),
            era="The present era",
            treatment="raw",
            flag="party_attack",
        )
        assert row["ci_lo"] < row["rate"] < row["ci_hi"]

    def test_input_row_order_changes_nothing(self, combat_inputs, combat_paras):
        """The seeding is per-cell (`default_rng([SEED, *stream])`) specifically
        so results do not depend on the order rows or cells are visited in. The
        shuffle is proved to have actually reordered the inputs first, so this
        cannot pass vacuously."""
        inputs = combat_inputs(_heterogeneous_present(combat_paras))
        in_order = C.rate_table(
            C.load_frame(**inputs), "era", ["The present era"], None
        )

        shuffled = {
            k: v.sample(frac=1, random_state=17 + i).reset_index(drop=True)
            for i, (k, v) in enumerate(inputs.items())
        }
        changed = [k for k in inputs if not shuffled[k].equals(inputs[k])]
        assert changed, "no input frame was actually reordered — the test would be vacuous"

        reordered = C.rate_table(
            C.load_frame(**shuffled), "era", ["The present era"], None
        )
        pd.testing.assert_frame_equal(in_order, reordered)

    def test_one_cells_interval_does_not_move_when_another_cells_data_changes(
        self, combat_inputs, combat_paras
    ):
        """Per-cell seeding, stated as a property. Under a single shared
        generator, adding speeches to an EARLIER era would consume different
        draws and shift this era's interval — a published number that moves
        because an unrelated era got more data."""
        present = _heterogeneous_present(combat_paras)
        thin = present + [
            {"doc": f"f{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, 1)}
            for k in range(6)
        ]
        thick = present + [
            {"doc": f"f{k}", "year": 1800, "type": SOTU, "paras": combat_paras(10, 2 + k % 4)}
            for k in range(11)
        ]
        order = ["The founding", "The present era"]
        cols = ["flag", "treatment", "rate", "ci_lo", "ci_hi"]

        def present_rows(specs):
            out = C.rate_table(C.load_frame(**combat_inputs(specs)), "era", order, None)
            # genre_standardized legitimately depends on the pooled reference
            # mix, which the added speeches change; raw/sotu_only must not move.
            out = out[(out["era"] == "The present era") & (out["treatment"] != "genre_standardized")]
            return out[cols].reset_index(drop=True)

        pd.testing.assert_frame_equal(present_rows(thin), present_rows(thick))


# --------------------------------------------------------------------------- #
# 6. empty and degenerate cells
# --------------------------------------------------------------------------- #
class TestDegenerateCells:
    def test_an_era_with_no_paragraphs_reports_no_data_rather_than_raising(
        self, combat_inputs, combat_paras
    ):
        df = C.load_frame(
            **combat_inputs([{"doc": "a", "year": 2020, "paras": combat_paras(4, 1)}])
        )
        out = C.rate_table(df, "era", C.ERA_ORDER, None)
        row = _one(out, era="The Gilded Age", treatment="raw", flag="party_attack")
        assert row["ci_status"] == "no_data"
        assert np.isnan(row["rate"])
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        assert row["n_paragraphs"] == 0
        assert row["n_speeches"] == 0
        assert row["n_flagged"] == 0

    def test_every_era_and_treatment_appears_even_when_unobserved(
        self, combat_inputs, combat_paras
    ):
        """A missing row reads as "nothing to report"; a no_data row reads as
        "we looked". The chart axis needs the second."""
        df = C.load_frame(
            **combat_inputs([{"doc": "a", "year": 2020, "paras": combat_paras(4, 1)}])
        )
        out = C.rate_table(df, "era", C.ERA_ORDER, None)
        assert len(out) == 9 * len(C.FLAGS) * len(C.TREATMENTS)
        assert out["era"].nunique() == 9
        assert set(out["treatment"]) == set(C.TREATMENTS)

    def test_an_era_with_no_sotu_speech_reports_no_data_for_that_treatment_only(
        self, combat_inputs, combat_paras
    ):
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"r{k}", "year": 2020, "type": RALLY, "paras": combat_paras(10, 3)}
                    for k in range(6)
                ]
            )
        )
        out = C.rate_table(df, "era", ["The present era"], None)
        sotu = _one(out, era="The present era", treatment="sotu_only", flag="party_attack")
        raw = _one(out, era="The present era", treatment="raw", flag="party_attack")
        assert sotu["ci_status"] == "no_data"
        assert np.isnan(sotu["rate"])
        assert raw["ci_status"] == "low_cluster_caution"
        assert raw["rate"] == pytest.approx(0.3)

    def test_a_flag_that_never_fires_is_zero_with_a_degenerate_interval(
        self, combat_inputs, combat_paras
    ):
        """Zero is a measurement, not missing data — it must carry an interval
        (here [0, 0]) rather than NaN, or the chart would show a gap where the
        answer is "never"."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": f"s{k}", "year": 2020, "paras": combat_paras(10, 3)}
                    for k in range(6)
                ]
            )
        )
        row = _one(
            C.rate_table(df, "era", ["The present era"], None),
            era="The present era",
            treatment="raw",
            flag="zero_sum",
        )
        assert row["rate"] == 0.0
        assert row["n_flagged"] == 0
        assert (row["ci_lo"], row["ci_hi"]) == (0.0, 0.0)
        assert row["ci_status"] == "low_cluster_caution"

    def test_the_published_row_order_is_group_then_flag_then_treatment(
        self, combat_inputs, combat_paras
    ):
        """MUTATION-CHECK FINDING: the sort key was unpinned.

        `test_applying_bands_preserves_the_declared_flag_order` compares the
        banded table against the unbanded one, so a change to the sort key moves
        both and survives. The parquet is read by the chart code and by a human
        scanning it, and the declared order is era axis first, then FLAGS order
        (party_attack — the headline flag — first), then TREATMENTS order (raw,
        sotu_only, genre_standardized)."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a", "year": 1820, "type": SOTU, "paras": combat_paras(4, 1)},
                    {"doc": "b", "year": 2020, "type": SOTU, "paras": combat_paras(4, 2)},
                ]
            )
        )
        out = C.rate_table(df, "era", ["Expansion", "The present era"], None)
        expected = [
            (era, flag, treatment)
            for era in ["Expansion", "The present era"]
            for flag in C.FLAGS
            for treatment in C.TREATMENTS
        ]
        assert list(
            zip(out["era"].astype(str), out["flag"].astype(str), out["treatment"].astype(str))
        ) == expected

    def test_speech_totals_collapse_a_cell_to_per_speech_sufficient_statistics(
        self, combat_inputs, combat_paras
    ):
        """The bootstrap clusters on speeches (within-speech ICC runs to 0.17),
        so the resampling unit has to be the speech, not the paragraph."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a", "year": 2020, "paras": combat_paras(4, 1)},
                    {"doc": "b", "year": 2020, "paras": combat_paras(6, 5)},
                ]
            )
        )
        n_para, flagged = C._speech_totals(_cell(df, "The present era"))
        assert n_para.tolist() == [4, 6]
        assert flagged["party_attack"].tolist() == [1, 5]


# --------------------------------------------------------------------------- #
# where the two floors cross
# --------------------------------------------------------------------------- #
class TestThinStratumFloor:
    """REGRESSION (review BLOCKER): `MIN_CELL_SPEECHES=3` admits a stratum into
    the standardized point estimate, but `MIN_CLUSTERS_FOR_CI=5` governs whether
    an interval may be built — and nothing crossed the two.

    Because the standardized estimator averages WITHIN-stratum resamples, a cell
    could report `n_speeches=52, ci_status="ok"` while drawing 74% of its weight
    from strata of 3-4 speeches. The identical 3 War & New Deal annual messages
    were `suppressed_n_floor` under `sotu_only` and carried a confident
    published interval under `genre_standardized`.
    """

    def _era(self, combat_inputs, combat_paras, n_sotu, n_rally):
        """One era, two genres, with the stratum sizes under test.

        SOTU paragraphs deliberately outnumber rally ones 6:1 per speech, so the
        SOTU stratum carries the majority of the REFERENCE weight even when it
        holds few speeches — which is exactly the shape that hides a 3-cluster
        estimate behind a large `n_speeches` (War & New Deal: 3 annual messages,
        49% of the weight, n_speeches=52).
        """
        specs = [
            {"doc": f"s{k}", "year": 1820, "type": SOTU, "paras": combat_paras(30, 15)}
            for k in range(n_sotu)
        ] + [
            {"doc": f"r{k}", "year": 1820, "type": RALLY, "paras": combat_paras(5, 1)}
            for k in range(n_rally)
        ]
        return C.load_frame(**combat_inputs(specs))

    def _row(self, df):
        return _one(
            C.rate_table(df, "era", ["Expansion"], None),
            era="Expansion",
            treatment="genre_standardized",
            flag="party_attack",
        )

    def test_a_thin_stratum_carrying_heavy_weight_suppresses_the_interval(
        self, combat_inputs, combat_paras
    ):
        """The blocker itself: plenty of total speeches, but half the weight
        rests on 3 clusters, which the module refuses to build an interval from
        anywhere else."""
        row = self._row(self._era(combat_inputs, combat_paras, n_sotu=3, n_rally=12))
        assert row["n_speeches"] == 15          # comfortably over the total floor
        assert row["effective_min_cluster"] == 3
        assert row["weight_from_thin_strata"] >= C.THIN_STRATUM_SUPPRESS_WEIGHT
        assert row["ci_status"] == "suppressed_n_floor"
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        # The POINT estimate survives — that is the whole reason for preferring
        # a status downgrade over raising MIN_CELL_SPEECHES.
        assert not np.isnan(row["rate"])

    def test_no_standardized_cell_reports_ok_while_leaning_on_thin_strata(
        self, combat_inputs, combat_paras
    ):
        """The invariant, stated directly: `ci_status="ok"` must be unreachable
        for any standardized cell whose thin strata carry suppress-level weight.
        `era-atlas` consumes `ci_status` as its trust gate."""
        for n_sotu, n_rally in [(3, 12), (4, 20), (3, 3), (4, 4)]:
            row = self._row(self._era(combat_inputs, combat_paras, n_sotu, n_rally))
            if row["weight_from_thin_strata"] >= C.THIN_STRATUM_SUPPRESS_WEIGHT:
                assert row["ci_status"] != "ok"

    def _weighted(self, combat_inputs, combat_paras, sotu_paras, n_sotu=3, n_rally=18):
        """A cell with >=20 total speeches (so the TOTAL-cluster rule says `ok`)
        whose thin 3-speech stratum carries a tunable share of the weight."""
        specs = [
            {"doc": f"s{k}", "year": 1820, "type": SOTU, "paras": combat_paras(sotu_paras, 2)}
            for k in range(n_sotu)
        ] + [
            {"doc": f"r{k}", "year": 1820, "type": RALLY, "paras": combat_paras(17, 4)}
            for k in range(n_rally)
        ]
        return C.load_frame(**combat_inputs(specs))

    def test_the_caution_tier_fires_between_the_two_weight_thresholds(
        self, combat_inputs, combat_paras
    ):
        """COVERAGE GAP: the caution tier is inert on the real corpus.

        Observed thin-stratum weights are {0, 0.024, 0.029, 0.042, 0.064} then
        {0.314, 0.359, 0.549, 0.744} — an empty gap straddling both thresholds,
        which is why the published partition is insensitive to exactly where
        THIN_STRATUM_SUPPRESS_WEIGHT sits inside it. But it also means no real
        cell lands in [0.10, 0.25), so the caution branch is never exercised by
        the corpus and a mutation flipping or deleting it would survive.

        3 SOTU speeches x 18 paragraphs against 18 rally speeches x 17 puts the
        thin stratum at exactly 54/360 = 0.15 of the weight, with 21 total
        speeches so the total-cluster rule would otherwise say `ok`.
        """
        row = self._row(self._weighted(combat_inputs, combat_paras, sotu_paras=18))

        thin = row["weight_from_thin_strata"]
        assert thin == pytest.approx(0.15)
        assert C.THIN_STRATUM_CAUTION_WEIGHT <= thin < C.THIN_STRATUM_SUPPRESS_WEIGHT
        assert row["n_speeches"] == 21          # total rule alone would say "ok"
        assert row["effective_min_cluster"] == 3
        assert row["ci_status"] == "low_cluster_caution"
        # Caution downgrades trust; it does NOT withhold the interval.
        assert not np.isnan(row["ci_lo"]) and not np.isnan(row["ci_hi"])

    def test_just_below_the_caution_threshold_keeps_the_base_status(
        self, combat_inputs, combat_paras
    ):
        """The other side of the same boundary: an identically-shaped cell whose
        thin stratum carries 0.081 — under THIN_STRATUM_CAUTION_WEIGHT — is left
        alone at `ok`. Paired with the test above this pins the threshold rather
        than merely reaching the branch."""
        row = self._row(self._weighted(combat_inputs, combat_paras, sotu_paras=9))

        thin = row["weight_from_thin_strata"]
        assert thin < C.THIN_STRATUM_CAUTION_WEIGHT
        assert row["effective_min_cluster"] == 3   # a thin stratum is still there
        assert row["ci_status"] == "ok"
        assert not np.isnan(row["ci_lo"])

    def test_thick_strata_are_left_alone(self, combat_inputs, combat_paras):
        """The floor must not fire when every stratum clears it — otherwise it
        would suppress the present-era cell that carries the headline."""
        row = self._row(self._era(combat_inputs, combat_paras, n_sotu=6, n_rally=16))
        assert row["effective_min_cluster"] == 6
        assert row["weight_from_thin_strata"] == pytest.approx(0.0)
        assert row["ci_status"] in {"ok", "low_cluster_caution"}
        assert not np.isnan(row["ci_lo"])

    def test_a_simple_cell_never_reports_thin_stratum_weight(
        self, combat_inputs, combat_paras
    ):
        """`raw` / `sotu_only` are unstratified: one cluster pool, so the
        effective minimum IS the cell and there is no hidden weight."""
        df = self._era(combat_inputs, combat_paras, n_sotu=6, n_rally=16)
        row = _one(
            C.rate_table(df, "era", ["Expansion"], None),
            era="Expansion", treatment="raw", flag="party_attack",
        )
        assert row["weight_from_thin_strata"] == pytest.approx(0.0)
        assert row["effective_min_cluster"] == row["n_speeches"] == 22

    def test_a_suppressed_standardized_cell_poisons_ratios_built_from_it(
        self, combat_inputs, combat_paras
    ):
        """A ratio may not launder an interval the thin-stratum rule refused."""
        specs = [
            {"doc": f"s{k}", "year": 1820, "type": SOTU, "paras": combat_paras(10, 5)}
            for k in range(3)
        ] + [
            {"doc": f"r{k}", "year": 1820, "type": RALLY, "paras": combat_paras(10, 1)}
            for k in range(12)
        ] + [
            {"doc": f"p{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 5)}
            for k in range(8)
        ] + [
            {"doc": f"q{k}", "year": 2020, "type": RALLY, "paras": combat_paras(10, 1)}
            for k in range(8)
        ]
        df = C.load_frame(**combat_inputs(specs))
        ratios = C.ratio_table(df, "era", ["Expansion", "The present era"], "The present era")
        row = _one(
            ratios, denominator="Expansion",
            treatment="genre_standardized", flag="party_attack",
        )
        assert row["ci_status"] == "suppressed_n_floor"
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
