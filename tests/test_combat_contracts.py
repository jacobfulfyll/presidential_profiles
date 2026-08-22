"""Contracts `combat.py` owes the rest of the project.

Four of them:

* **The annotator-disagreement seam.** `inter-model-agreement-check` had not
  produced `data/llm_annotations/agreement_v1.parquet` when this module was
  written, so the acceptance criterion is *deferred with a working seam*. That
  makes the seam the deliverable, and its four behaviours (absent / bad schema /
  duplicate key / valid) are what these tests pin. The valid case is built in
  `tmp_path`; **nothing here ever creates the real agreement parquet** — that
  path belongs to the other task.
* **$0.** Pure local compute. `tests/conftest.py` strips the API key, and the
  build runs here with `anthropic.Anthropic` booby-trapped.
* **Frozen artifacts.** `data/llm_annotations/` is a paid, provenance-stamped
  output; a build must leave it byte-identical.
* **The published anchors.** One integration-style test (clearly marked, and the
  only one in the suite that reads the real 36k-row corpus) confirms the wiring
  end to end still reproduces the numbers in
  `notes/combativeness-findings-v1.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import combat as C
from presidential_profiles import llm_annotations as ann

SOTU = C.SOTU_TYPE
RALLY = "public_remarks_or_address"

# The write-side artifacts build_combativeness produces, by file name.
EXPECTED_OUTPUT_FILES = {
    "combativeness.parquet",
    "peak_decades.parquet",
    "ratios.parquet",
    "exemplars.parquet",
    "entity_consistency.parquet",
    "lexical_baseline.parquet",
    "adversary_mix.parquet",
    "by_president.parquet",
    "genre_decomposition.parquet",
    "combat_meta.json",
}


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hits = frame[mask]
    assert len(hits) == 1, f"{where} matched {len(hits)} rows, expected 1"
    return hits.iloc[0]


@pytest.fixture
def banded_frame(combat_inputs, combat_paras):
    """Two eras: Expansion clears the n-floor, the present era does not.

    Every speech inside an era is identical, so each bootstrap interval collapses
    to the point rate. That is deliberate — it makes the *widening* arithmetic in
    the agreement tests exact and hand-checkable (0.30 +/- 0.05 -> [0.25, 0.35])
    instead of "wider by roughly something".
    """
    specs = [
        {"doc": f"e1-{k}", "year": 1820, "type": SOTU, "paras": combat_paras(10, 3)}
        for k in range(6)
    ] + [
        {"doc": f"e2-{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 5)}
        for k in range(4)
    ]
    return C.load_frame(**combat_inputs(specs))


def _write_bands(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# the published axes, anchored as literals
# --------------------------------------------------------------------------- #
class TestPublishedAxes:
    """MUTATION-CHECK FINDING: every axis constant was only ever asserted
    against itself.

    `meta["flags"] == C.FLAGS`, `meta["treatments"] == C.TREATMENTS` and
    `set(back["treatment"]) == set(C.TREATMENTS)` all follow the constant
    wherever it moves, so reordering FLAGS (which is the declared publication
    order — party_attack, the headline flag, first), reordering TREATMENTS, or
    changing PEAK_DECADES away from the three peaks the question names all
    passed the suite untouched. These are decoupled literal anchors on the task
    specification, not on the module.
    """

    def test_the_three_flags_are_the_task_specifications_three(self):
        assert C.FLAGS == ["party_attack", "enemy_naming", "zero_sum"]

    def test_the_three_genre_treatments_are_declared_raw_first(self):
        assert C.TREATMENTS == ["raw", "sotu_only", "genre_standardized"]

    def test_the_peak_decades_are_the_1860s_1930s_and_2020s(self):
        """Acceptance criterion: "the three-peaks chart: 1860s / 1930s / 2020s on
        one axis". The decade table exists precisely because the era bands
        dilute those decades, so the three decades themselves are the
        contract."""
        assert C.PEAK_DECADES == [1860, 1930, 2020]

    def test_there_is_no_composite_index(self):
        """A Key Decision, not a preference: the three flags tell different
        stories and any weighted sum hides that. Asserted structurally — no
        published table may carry a flag outside the declared three."""
        assert set(C.FLAGS) == {"party_attack", "enemy_naming", "zero_sum"}
        assert not any(
            "composite" in name.lower() or "index" in name.lower()
            for name in dir(C)
            if not name.startswith("__")
        )


# --------------------------------------------------------------------------- #
# the agreement seam: absent
# --------------------------------------------------------------------------- #
class TestAgreementSeamAbsent:
    def test_an_absent_file_returns_none(self, tmp_path):
        """Absence is a normal, expected state — not an error and never a stub.
        A fabricated disagreement magnitude would be indistinguishable from a
        measured one once it was in the parquet."""
        assert C.load_agreement_bands(path=tmp_path / "agreement_v1.parquet") is None

    def test_the_real_agreement_file_is_still_absent_and_is_not_created_here(self):
        """Documents the state this task shipped in, and guards the boundary with
        `inter-model-agreement-check`: nothing in this suite may create that
        path. If the other task lands, this becomes a legitimate failure and the
        report's limitation 2 needs revisiting."""
        assert C.AGREEMENT_PATH.name == "agreement_v1.parquet"
        assert C.AGREEMENT_PATH.parent.name == "llm_annotations"
        assert not C.AGREEMENT_PATH.exists()

    def test_absent_bands_mark_every_row_sampling_only(self, banded_frame):
        out = C.rate_table(banded_frame, "era", ["Expansion", "The present era"], None)
        assert (out["ci_components"] == "sampling_only").all()
        assert not out["disagreement_band_applied"].any()
        assert (out["agreement_source"] == "absent:agreement_v1.parquet").all()
        assert out["disagreement_half_width"].isna().all()

    def test_absent_bands_leave_the_sampling_interval_untouched(self, banded_frame):
        row = _one(
            C.rate_table(banded_frame, "era", ["Expansion"], None),
            era="Expansion",
            treatment="raw",
            flag="party_attack",
        )
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((0.3, 0.3))


# --------------------------------------------------------------------------- #
# the agreement seam: malformed
# --------------------------------------------------------------------------- #
class TestAgreementSeamSchema:
    def test_required_columns_are_the_documented_contract(self):
        assert C.AGREEMENT_REQUIRED_COLUMNS == {"era", "flag", "disagreement_half_width"}

    def test_a_file_with_the_wrong_schema_raises_listing_what_was_expected(self, tmp_path):
        """If the file appears but does not carry what this module needs, the
        loader must refuse rather than guess — silently mis-reading a band is
        worse than having no band."""
        path = _write_bands(
            tmp_path / "agreement_v1.parquet",
            [{"era": "Expansion", "flag": "party_attack", "half_width": 0.05}],
        )
        with pytest.raises(ValueError) as excinfo:
            C.load_agreement_bands(path=path)
        msg = str(excinfo.value)
        assert "disagreement_half_width" in msg
        assert "refusing to guess" in msg
        for col in sorted(C.AGREEMENT_REQUIRED_COLUMNS):
            assert col in msg

    def test_a_duplicate_era_flag_key_raises(self, tmp_path):
        """A band must identify exactly one row; two rows for one (era, flag)
        would fan the rate table out and double-apply a widening."""
        path = _write_bands(
            tmp_path / "agreement_v1.parquet",
            [
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": 0.09},
            ],
        )
        with pytest.raises(ValueError, match=r"share an \(era, flag\) key"):
            C.load_agreement_bands(path=path)

    def test_distinct_era_flag_pairs_are_not_treated_as_duplicates(self, tmp_path):
        path = _write_bands(
            tmp_path / "agreement_v1.parquet",
            [
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": "Expansion", "flag": "zero_sum", "disagreement_half_width": 0.09},
            ],
        )
        bands = C.load_agreement_bands(path=path)
        assert len(bands) == 2

    def test_extra_columns_are_tolerated(self, tmp_path):
        """The other task owns that file's full schema; this module reads three
        columns and must not break when it carries more."""
        path = _write_bands(
            tmp_path / "agreement_v1.parquet",
            [
                {
                    "era": "Expansion",
                    "flag": "party_attack",
                    "disagreement_half_width": 0.05,
                    "kappa": 0.81,
                }
            ],
        )
        assert len(C.load_agreement_bands(path=path)) == 1


# --------------------------------------------------------------------------- #
# the agreement seam: applied
# --------------------------------------------------------------------------- #
class TestAgreementSeamApplied:
    ORDER = ["Expansion", "The present era"]

    def _bands(self, tmp_path, half_width=0.05, era="Expansion"):
        path = _write_bands(
            tmp_path / "agreement_v1.parquet",
            [{"era": era, "flag": "party_attack", "disagreement_half_width": half_width}],
        )
        return C.load_agreement_bands(path=path)

    def test_a_valid_file_round_trips_and_widens_the_interval(self, banded_frame, tmp_path):
        bands = self._bands(tmp_path)
        out = C.rate_table(banded_frame, "era", self.ORDER, bands)
        row = _one(out, era="Expansion", treatment="raw", flag="party_attack")
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((0.25, 0.35))
        assert row["disagreement_band_applied"]
        assert row["disagreement_half_width"] == pytest.approx(0.05)
        assert row["ci_components"] == "sampling+annotator"
        assert row["agreement_source"] == "agreement_v1.parquet"

    def test_the_point_estimate_is_never_moved_by_a_band(self, banded_frame, tmp_path):
        """Disagreement is uncertainty, not bias correction.

        Compared cell by cell rather than row by row on purpose. Row order is
        now guaranteed stable across the band merge
        (``test_applying_bands_preserves_the_declared_flag_order`` pins that),
        but keying on (era, flag, treatment) keeps this test measuring only what
        it claims to: that a band moves an interval and never a point estimate.
        A positional comparison would silently start failing for ordering
        reasons if that guarantee ever regressed."""
        no_band = C.rate_table(banded_frame, "era", self.ORDER, None)
        banded = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path))
        key = ["era", "flag", "treatment"]
        before = no_band.set_index(key)["rate"]
        after = banded.set_index(key)["rate"]
        assert set(before.index) == set(after.index)
        for cell in before.index:
            assert after[cell] == pytest.approx(before[cell], nan_ok=True)

    def test_no_row_is_added_or_dropped_by_the_band_merge(self, banded_frame, tmp_path):
        """`validate="many_to_one"` plus a partial band table: a fan-out would
        double-count a cell, a bad key would drop one."""
        no_band = C.rate_table(banded_frame, "era", self.ORDER, None)
        banded = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path))
        assert len(banded) == len(no_band)
        assert not banded.duplicated(["era", "flag", "treatment"]).any()

    def test_a_cell_suppressed_by_the_n_floor_never_acquires_a_band(
        self, banded_frame, tmp_path
    ):
        """The deferred criterion's sharpest edge. The present-era cell here has
        4 speeches, so its interval was suppressed; widening NaN by a half-width
        must not manufacture one. An interval appearing on a cell the n-floor
        refused would put an unpublishable number back in the report."""
        bands = self._bands(tmp_path, half_width=0.05, era="The present era")
        out = C.rate_table(banded_frame, "era", self.ORDER, bands)
        row = _one(out, era="The present era", treatment="raw", flag="party_attack")
        assert row["ci_status"] == "suppressed_n_floor"
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        assert row["rate"] == pytest.approx(0.5)

    def test_cells_absent_from_the_band_table_keep_sampling_only_intervals(
        self, banded_frame, tmp_path
    ):
        """A partial band table is expected — the other task may not cover every
        (era, flag). Unmatched rows must say `sampling_only`, not silently look
        as though a band had been applied."""
        out = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path))
        unbanded = _one(out, era="Expansion", treatment="raw", flag="zero_sum")
        assert unbanded["ci_components"] == "sampling_only"
        assert not unbanded["disagreement_band_applied"]
        assert np.isnan(unbanded["disagreement_half_width"])
        assert (unbanded["ci_lo"], unbanded["ci_hi"]) == pytest.approx((0.0, 0.0))

    def test_widened_bounds_are_clipped_to_the_unit_interval(self, banded_frame, tmp_path):
        """These are rates. A band wide enough to push a bound past 0 or 1 must
        clip, not publish a negative probability."""
        out = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path, 0.9))
        row = _one(out, era="Expansion", treatment="raw", flag="party_attack")
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((0.0, 1.0))

    def test_bands_apply_across_all_three_genre_treatments(self, banded_frame, tmp_path):
        out = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path))
        banded = out[
            (out["era"] == "Expansion")
            & (out["flag"] == "party_attack")
            & out["disagreement_band_applied"]
        ]
        assert set(banded["treatment"]) == set(C.TREATMENTS)

    def test_ratio_intervals_record_which_uncertainty_components_they_contain(
        self, banded_frame
    ):
        ratios = C.ratio_table(banded_frame, "era", self.ORDER, "The present era")
        assert "ci_components" in ratios.columns

    def test_the_decade_table_never_applies_bands(self, banded_frame, tmp_path):
        """Bands key on `era`; a decade table has no era column, so it carries
        the same provenance fields with nothing applied rather than joining on
        the wrong axis."""
        out = C.rate_table(banded_frame, "decade", C.PEAK_DECADES, self._bands(tmp_path))
        assert (out["ci_components"] == "sampling_only").all()
        assert (out["agreement_source"] == "absent:agreement_v1.parquet").all()
        assert not out["disagreement_band_applied"].any()

    def test_the_same_fields_are_present_with_and_without_bands(
        self, banded_frame, tmp_path
    ):
        """Downstream readers must not have to branch on whether the file
        existed — only on `ci_components`.

        Compared as a SEQUENCE, not a set. The band merge appends
        `disagreement_half_width` during the join while the absent branch
        assigns `disagreement_band_applied` first, so the column *order* used to
        flip the moment an agreement file appeared — a schema change that a
        positional reader would see and a name-based one would not.
        `_apply_agreement_bands` now pins `BAND_PROVENANCE_COLUMNS` last in a
        fixed order, and this asserts that pin."""
        without = C.rate_table(banded_frame, "era", self.ORDER, None)
        with_bands = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path))
        assert list(without.columns) == list(with_bands.columns)
        assert list(without.columns[-8:-4]) == C.BAND_PROVENANCE_COLUMNS

    def test_a_duplicate_keyed_band_table_handed_straight_to_rate_table_raises(
        self, banded_frame
    ):
        """MUTATION-CHECK FINDING: `validate="many_to_one"` on the band merge was
        unattributable.

        `load_agreement_bands` already rejects duplicate (era, flag) keys, so
        with every existing test routing through the loader the merge's own
        `validate=` could be deleted and nothing failed — belt-and-suspenders
        guarding, with the outer belt doing all the work. `rate_table(bands=)` is
        a public seam that a caller can reach without the loader, and a fan-out
        there would double-apply a widening to a published interval. Driven
        directly so the guard is attributable to its own line."""
        dupes = pd.DataFrame(
            [
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": "Expansion", "flag": "party_attack", "disagreement_half_width": 0.09},
            ]
        )
        with pytest.raises(pd.errors.MergeError):
            C.rate_table(banded_frame, "era", self.ORDER, dupes)

    def test_applying_bands_preserves_the_declared_flag_order(self, banded_frame, tmp_path):
        without = C.rate_table(banded_frame, "era", self.ORDER, None)
        with_bands = C.rate_table(banded_frame, "era", self.ORDER, self._bands(tmp_path))
        assert with_bands["flag"].tolist() == without["flag"].tolist()


# --------------------------------------------------------------------------- #
# the agreement seam: applied to the RATIO table
#
# MUTATION-CHECK FINDING: no test called `ratio_table(..., bands=<not None>)`,
# so the entire band-propagation path was uncovered and every mutation to it
# survived — deleting the widening, inverting its direction, emptying the
# lookup, marking every row banded regardless of whether a band matched it.
# `ratio_table`'s own docstring calls this out as the point of the seam ("a seam
# that widened combativeness.parquet while leaving ratios.parquet silently
# unchanged would defeat the point of deferring the criterion"), and
# `notes/combativeness-findings-v1.md` §limitation 2 promises the quoted
# headline widens with the marginals. Neither claim was tested.
# --------------------------------------------------------------------------- #
@pytest.fixture
def ratio_band_frame(combat_inputs, combat_paras):
    """Present era at 0.50, Expansion at 0.30, six identical speeches each.

    Identical speeches make every resample reproduce its cell's rate exactly, so
    the sampling ratio interval collapses to the point ratio 5/3 and the
    *widening* arithmetic below is exact and hand-checkable rather than
    "wider by roughly something". Both cells clear the n-floor.
    """
    specs = [
        {"doc": f"p{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 5)}
        for k in range(6)
    ] + [
        {"doc": f"e{k}", "year": 1820, "type": SOTU, "paras": combat_paras(10, 3)}
        for k in range(6)
    ]
    return C.load_frame(**combat_inputs(specs))


class TestRatioAgreementBands:
    ORDER = ["Expansion", "The present era"]
    NUM_ERA = "The present era"
    DEN_ERA = "Expansion"

    def _bands(self, rows):
        return pd.DataFrame(rows)

    def _row(self, frame, flag="party_attack"):
        return _one(frame, denominator=self.DEN_ERA, treatment="raw", flag=flag)

    def _ratios(self, frame, bands):
        return C.ratio_table(frame, "era", self.ORDER, self.NUM_ERA, bands=bands)

    def test_the_unbanded_ratio_interval_is_the_point_ratio(self, ratio_band_frame):
        """Premise of everything below: without a band the interval is exactly
        5/3, so any movement in the tests that follow is the widening and
        nothing else."""
        row = self._row(self._ratios(ratio_band_frame, None))
        assert row["ratio"] == pytest.approx(5 / 3)
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((5 / 3, 5 / 3))
        assert row["ci_components"] == "sampling_only"
        assert not row["disagreement_band_applied"]

    def test_a_numerator_band_widens_the_quoted_ratio_interval(self, ratio_band_frame):
        bands = self._bands(
            [{"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05}]
        )
        row = self._row(self._ratios(ratio_band_frame, bands))
        # numerator 0.50 +/- 0.05 -> factors 0.9 and 1.1 on 5/3
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((1.5, 5 / 3 * 1.1))
        assert row["disagreement_band_applied"]
        assert row["ci_components"] == "sampling+annotator"
        assert row["agreement_source"] == "agreement_v1.parquet"
        assert row["disagreement_half_width_numerator"] == pytest.approx(0.05)
        assert np.isnan(row["disagreement_half_width_denominator"])

    def test_a_denominator_band_widens_it_in_the_other_direction(self, ratio_band_frame):
        bands = self._bands(
            [{"era": self.DEN_ERA, "flag": "party_attack", "disagreement_half_width": 0.06}]
        )
        row = self._row(self._ratios(ratio_band_frame, bands))
        # denominator 0.30 -> lower factor 0.30/0.36, upper factor 0.30/0.24
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx(
            (5 / 3 * (0.30 / 0.36), 5 / 3 * (0.30 / 0.24))
        )
        assert np.isnan(row["disagreement_half_width_numerator"])
        assert row["disagreement_half_width_denominator"] == pytest.approx(0.06)

    def test_bands_on_both_eras_compose(self, ratio_band_frame):
        bands = self._bands(
            [
                {"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": self.DEN_ERA, "flag": "party_attack", "disagreement_half_width": 0.06},
            ]
        )
        row = self._row(self._ratios(ratio_band_frame, bands))
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((1.25, 2.291666666))

    def test_the_widened_interval_still_contains_the_point_ratio(self, ratio_band_frame):
        """Direction check, stated as a property. Inverting the widening (the
        numerator's shrink factor applied to the upper bound and vice versa)
        produces lo > ratio > hi — an interval that excludes its own point
        estimate and reads as narrower, not wider."""
        bands = self._bands(
            [
                {"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": self.DEN_ERA, "flag": "party_attack", "disagreement_half_width": 0.06},
            ]
        )
        banded = self._row(self._ratios(ratio_band_frame, bands))
        plain = self._row(self._ratios(ratio_band_frame, None))
        assert banded["ci_lo"] <= banded["ratio"] <= banded["ci_hi"]
        assert banded["ci_lo"] < plain["ci_lo"]
        assert banded["ci_hi"] > plain["ci_hi"]

    def test_a_zero_half_width_leaves_the_interval_exactly_where_it_was(
        self, ratio_band_frame
    ):
        """Continuity: the published number must not jump the instant the band
        file appears carrying a negligible disagreement."""
        bands = self._bands(
            [{"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.0}]
        )
        row = self._row(self._ratios(ratio_band_frame, bands))
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((5 / 3, 5 / 3))
        assert row["disagreement_band_applied"]
        assert row["ci_components"] == "sampling+annotator"

    def test_a_suppressed_row_never_claims_an_annotator_component(
        self, combat_inputs, combat_paras
    ):
        """REGRESSION (review M1): `disagreement_band_applied` was computed from
        band PRESENCE, so an n-floor-suppressed row came back
        `disagreement_band_applied=True`, `ci_components="sampling+annotator"`
        with `ci_lo/ci_hi = NaN`. The interval was correctly not manufactured,
        but the provenance label asserted a component applied to nothing — while
        report §8.2 promises these widen when the file lands.

        The denominator era here has 3 speeches, under MIN_CLUSTERS_FOR_CI.
        """
        specs = [
            {"doc": f"d{k}", "year": 1820, "paras": combat_paras(10, 5)}
            for k in range(3)
        ] + [
            {"doc": f"n{k}", "year": 2020, "paras": combat_paras(10, 5)}
            for k in range(8)
        ]
        df = C.load_frame(**combat_inputs(specs))
        bands = self._bands([
            {"era": e, "flag": "party_attack", "disagreement_half_width": 0.05}
            for e in self.ORDER
        ])
        row = self._row(self._ratios(df, bands))
        assert row["ci_status"] == "suppressed_n_floor"
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        assert not row["disagreement_band_applied"]
        assert row["ci_components"] == "sampling_only"

    def test_a_live_row_alongside_a_suppressed_one_still_claims_its_band(
        self, ratio_band_frame
    ):
        """The other half of M1: the fix must not stop labelling rows that DID
        receive a band, or the seam would under-report instead of over-report."""
        bands = self._bands([
            {"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05}
        ])
        row = self._row(self._ratios(ratio_band_frame, bands))
        assert row["disagreement_band_applied"]
        assert row["ci_components"] == "sampling+annotator"

    def test_a_band_wide_enough_to_zero_the_denominator_is_reported_as_unbounded(
        self, ratio_band_frame
    ):
        bands = self._bands(
            [{"era": self.DEN_ERA, "flag": "party_attack", "disagreement_half_width": 0.30}]
        )
        row = self._row(self._ratios(ratio_band_frame, bands))
        assert np.isinf(row["ci_hi"])
        assert not np.isnan(row["ci_hi"])
        assert row["ci_lo"] == pytest.approx(5 / 3 * 0.5)

    def test_only_the_cells_named_in_the_band_table_are_marked_banded(
        self, ratio_band_frame
    ):
        """`disagreement_band_applied` is a per-(era, flag) fact, not "a band
        file existed". Marking every row banded would claim annotator
        uncertainty for flags the other task never measured."""
        bands = self._bands(
            [{"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05}]
        )
        ratios = self._ratios(ratio_band_frame, bands)
        assert self._row(ratios, "party_attack")["disagreement_band_applied"]
        for flag in ("enemy_naming", "zero_sum"):
            other = self._row(ratios, flag)
            assert not other["disagreement_band_applied"], flag
            assert other["ci_components"] == "sampling_only"
            assert np.isnan(other["disagreement_half_width_numerator"])
            assert np.isnan(other["disagreement_half_width_denominator"])

    def test_a_ratio_suppressed_by_the_n_floor_never_acquires_a_band(
        self, combat_inputs, combat_paras
    ):
        """The n-floor outranks the seam here exactly as it does in the marginal
        table: widening NaN must not manufacture an interval for a cell the
        pre-registered policy refused."""
        specs = [
            {"doc": f"p{k}", "year": 2020, "type": SOTU, "paras": combat_paras(10, 5)}
            for k in range(6)
        ] + [
            {"doc": f"e{k}", "year": 1820, "type": SOTU, "paras": combat_paras(10, 3)}
            for k in range(4)
        ]
        df = C.load_frame(**combat_inputs(specs))
        bands = self._bands(
            [
                {"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": self.DEN_ERA, "flag": "party_attack", "disagreement_half_width": 0.06},
            ]
        )
        row = self._row(self._ratios(df, bands))
        assert row["ci_status"] == "suppressed_n_floor"
        assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
        assert row["ratio"] == pytest.approx(5 / 3)

    def test_a_band_never_moves_the_point_ratio(self, ratio_band_frame):
        bands = self._bands(
            [
                {"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05},
                {"era": self.DEN_ERA, "flag": "party_attack", "disagreement_half_width": 0.06},
            ]
        )
        key = ["denominator", "flag", "treatment"]
        before = self._ratios(ratio_band_frame, None).set_index(key)["ratio"]
        after = self._ratios(ratio_band_frame, bands).set_index(key)["ratio"]
        assert set(before.index) == set(after.index)
        for cell in before.index:
            assert after[cell] == pytest.approx(before[cell], nan_ok=True)

    def test_bands_reach_every_genre_treatment_of_the_ratio_table(self, ratio_band_frame):
        bands = self._bands(
            [{"era": self.NUM_ERA, "flag": "party_attack", "disagreement_half_width": 0.05}]
        )
        ratios = self._ratios(ratio_band_frame, bands)
        banded = ratios[
            (ratios["flag"] == "party_attack") & ratios["disagreement_band_applied"]
        ]
        assert set(banded["treatment"]) == set(C.TREATMENTS)

    def test_a_non_era_grain_ratio_table_carries_the_fields_with_nothing_applied(
        self, combat_inputs, combat_paras
    ):
        """Bands key on era labels; a decade table's groups are integers, so the
        lookup is deliberately not consulted at all. Documented rather than
        merely left to the type mismatch."""
        specs = [
            {"doc": f"a{k}", "year": 1861, "type": SOTU, "paras": combat_paras(10, 5)}
            for k in range(6)
        ] + [
            {"doc": f"b{k}", "year": 2021, "type": SOTU, "paras": combat_paras(10, 3)}
            for k in range(6)
        ]
        df = C.load_frame(**combat_inputs(specs))
        bands = self._bands(
            [{"era": "1860", "flag": "party_attack", "disagreement_half_width": 0.05}]
        )
        ratios = C.ratio_table(df, "decade", [1860, 2020], 2020, bands=bands)
        assert not ratios["disagreement_band_applied"].any()
        assert (ratios["ci_components"] == "sampling_only").all()
        # ...and the provenance label agrees with `ci_components`. This used to
        # name the band file whenever `bands is not None`, even on a grain where
        # no band can apply, so a decade row claimed a source it never read
        # while reporting `sampling_only`. `rate_table` already gated this
        # correctly; both now gate on the same value.
        assert (ratios["agreement_source"] == "absent:agreement_v1.parquet").all()


# --------------------------------------------------------------------------- #
# build + CLI
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_markers(all_era_frame, tmp_path, monkeypatch):
    """Point the lexical proxies at a markers table matching the synthetic frame.

    ``lexical_baseline`` / ``lexical_correlations`` read
    ``speech_markers.parquet``, and the row-completeness guard inside
    ``lexical_baseline`` would (correctly) reject the real 1,057-row markers
    table against a synthetic frame. Both now resolve ``SPEECH_MARKERS_PATH`` at
    CALL time, so redirecting the constant is enough — the real functions run on
    real (small) data instead of being replaced by stubs, and the redirect
    reaches ``main()`` too, which cannot be passed a ``markers_path=``.

    Returns the path so a test can also exercise the explicit
    ``build_combativeness(markers_path=)`` seam.
    """
    docs = sorted(all_era_frame["doc_name"].unique())
    markers = pd.DataFrame(
        {
            "doc_name": docs,
            # Deterministic, varying counts — enough for the per-10k rates and
            # the Spearman correlations to be well defined.
            "us_them": [3 * i for i in range(len(docs))],
            "opponents": [i % 7 for i in range(len(docs))],
            "n_words": [1000] * len(docs),
        }
    )
    path = tmp_path / "speech_markers.parquet"
    markers.to_parquet(path, index=False)
    monkeypatch.setattr(C, "SPEECH_MARKERS_PATH", path)
    return path


@pytest.fixture
def synthetic_entities(all_era_frame, tmp_path, monkeypatch):
    """Point the entity-derived tables at an entities table matching the frame.

    Same pattern as ``synthetic_markers``: ``adversary_mix`` / ``by_president``
    / ``mistyped_nation_recount`` resolve ``PARAGRAPH_ENTITIES_PATH`` at CALL
    time via ``_adversarial_entities``, so redirecting the constant is enough
    and the real functions run on real (small) data rather than being stubbed.
    Merging the real 27k-row entity table against a synthetic frame would drop
    every row.

    One adversarial entity per paragraph, cycling the five types so the
    ``adv_share_*`` columns and the per-president counts are all well defined.
    """
    types = ["nation", "person", "group", "institution", "other"]
    keys = all_era_frame[["doc_name", "para_idx"]].drop_duplicates()
    entities = pd.DataFrame(
        {
            "doc_name": keys["doc_name"].to_numpy(),
            "para_idx": keys["para_idx"].to_numpy(),
            "entity": [f"adversary {i}" for i in range(len(keys))],
            "type": [types[i % len(types)] for i in range(len(keys))],
            "stance": "adversarial",
            "run_id": "synthetic",
        }
    )
    path = tmp_path / "paragraph_entities.parquet"
    entities.to_parquet(path, index=False)
    monkeypatch.setattr(C, "PARAGRAPH_ENTITIES_PATH", path)
    return path


@pytest.fixture
def stub_real_data_leaves(synthetic_markers, synthetic_entities, monkeypatch):
    """The one leaf inside `build_combativeness` with no injectable seam.

    ``corpus_fingerprint()`` takes the *speeches* and *paragraphs* frames, which
    `build_combativeness` does not hold — it holds the merged paragraph frame —
    so there is nothing to forward and it would read the real 36k-row corpus.
    It is covered directly in ``test_combat_frame.py``; what is under test here
    is the orchestration around it. The lexical proxies are no longer stubbed at
    all (see ``synthetic_markers``).
    """
    monkeypatch.setattr(
        C,
        "corpus_fingerprint",
        lambda *a, **k: {"n_speeches": 12, "n_paragraphs": 72, "doc_name_sha256": "deadbeef"},
    )


@pytest.fixture
def all_era_frame(combat_inputs, combat_paras):
    """A speech in every era plus the three peak decades, so every table
    `build_combativeness` produces has rows to produce.

    All three flags vary ACROSS speeches. That is load-bearing now that the
    lexical proxies run for real: a flag that is constant corpus-wide has an
    undefined Spearman correlation against every proxy, which would land as a
    bare `NaN` token in `combat_meta.json` — valid for Python's `json` but not
    for strict JSON, and a poor thing for an orchestration test to assert
    around.
    """
    specs = []
    for i, label in enumerate(C.ERA_ORDER):
        lo, _hi = C.ERA_BOUNDS[label]
        for k in range(5):
            specs.append(
                {
                    "doc": f"{i}-sotu{k}",
                    "year": lo,
                    "type": SOTU,
                    "paras": combat_paras(
                        6, k % 3, adversaries=[("a foe", "nation")], enemy_naming=True
                    ),
                }
            )
            specs.append(
                {
                    "doc": f"{i}-rally{k}",
                    "year": lo,
                    "type": RALLY,
                    "paras": combat_paras(6, k % 2, zero_sum=k % 2 == 0),
                }
            )
    for decade in C.PEAK_DECADES:
        for k in range(5):
            specs.append(
                {"doc": f"d{decade}-{k}", "year": decade + 1, "type": SOTU,
                 "paras": combat_paras(6, k % 3)}
            )
    return C.load_frame(**combat_inputs(specs))


class TestBuildOutputs:
    def test_build_writes_every_declared_table_into_the_requested_dir(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        out_dir = tmp_path / "combat"
        tables = C.build_combativeness(all_era_frame, out_dir=out_dir)

        assert set(tables) == {
            "combativeness",
            "peak_decades",
            "ratios",
            "entity_consistency",
            "lexical_baseline",
            "adversary_mix",
            "by_president",
            "genre_decomposition",
            "exemplars",
        }
        assert {p.name for p in out_dir.iterdir()} == EXPECTED_OUTPUT_FILES

    def test_markers_path_is_forwarded_to_both_lexical_consumers(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """`build_combativeness` reads `speech_markers.parquet` twice — once for
        the published `lexical_baseline` table and once for the `lexical_spearman`
        block of the meta. Both have to honour an explicit `markers_path=`, or a
        caller could get a baseline table and a correlation block computed off
        two different files.

        The override carries deliberately different counts from the
        `synthetic_markers` default, so a forward that was silently dropped on
        either path shows up as the default's numbers.
        """
        docs = sorted(all_era_frame["doc_name"].unique())
        override = tmp_path / "other_markers.parquet"
        pd.DataFrame(
            {
                "doc_name": docs,
                "us_them": [100 + i for i in range(len(docs))],
                "opponents": [i % 5 for i in range(len(docs))],
                "n_words": [2000] * len(docs),
            }
        ).to_parquet(override, index=False)

        out_dir = tmp_path / "combat"
        tables = C.build_combativeness(
            all_era_frame, out_dir=out_dir, markers_path=override
        )

        baseline = tables["lexical_baseline"]
        raw = baseline[baseline["treatment"] == "raw"]
        assert raw["n_words"].sum() == 2000 * len(docs)
        assert raw["us_them"].sum() == sum(100 + i for i in range(len(docs)))

        meta = json.loads((out_dir / "combat_meta.json").read_text())
        direct = C.lexical_correlations(all_era_frame, path=override)
        assert meta["lexical_spearman"] == direct
        assert meta["lexical_spearman"] != C.lexical_correlations(
            all_era_frame, path=C.SPEECH_MARKERS_PATH
        )

    def test_build_loads_the_frame_itself_when_none_is_supplied(
        self, monkeypatch, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """The production entry point passes no frame. `load_frame` is stubbed
        here so the real 36k-row corpus is not pulled in; that path has its own
        integration test at the bottom of this file."""
        calls = []

        def fake_load_frame(*a, **k):
            calls.append(1)
            return all_era_frame

        monkeypatch.setattr(C, "load_frame", fake_load_frame)
        C.build_combativeness(out_dir=tmp_path / "combat")
        assert calls == [1]

    def test_written_parquets_round_trip_without_categorical_columns(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """Categoricals are cast to str on the way out so a consumer that never
        imports this module still reads plain strings."""
        out_dir = tmp_path / "combat"
        C.build_combativeness(all_era_frame, out_dir=out_dir)
        back = pd.read_parquet(out_dir / "combativeness.parquet")
        assert not any(isinstance(back[c].dtype, pd.CategoricalDtype) for c in back.columns)
        assert set(back["treatment"]) == set(C.TREATMENTS)
        assert set(back["flag"]) == set(C.FLAGS)
        assert len(back) == 9 * len(C.FLAGS) * len(C.TREATMENTS)

    def test_meta_records_zero_api_calls_and_the_preregistered_policy(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        out_dir = tmp_path / "combat"
        C.build_combativeness(all_era_frame, out_dir=out_dir)
        meta = json.loads((out_dir / "combat_meta.json").read_text())

        assert meta["api_calls"] == 0
        assert meta["era_scheme"] == "trends.ERAS"
        assert meta["flags"] == C.FLAGS
        assert meta["treatments"] == C.TREATMENTS
        assert "none" in meta["composite_index"]
        assert meta["bootstrap"] == {
            "n": 2000,
            "seed": 20260721,
            "ci_level": 0.95,
            "cluster_unit": "speech",
            "min_clusters_for_ci": 5,
            "low_cluster_caution": 20,
        }
        assert meta["genre_standardization"]["min_cell_speeches"] == 3
        assert meta["annotation_run_ids"] == ["run-test"]

    def test_meta_records_the_agreement_component_as_absent(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """The deferred criterion has to be legible from the artifact alone, not
        only from the prose report."""
        out_dir = tmp_path / "combat"
        C.build_combativeness(all_era_frame, out_dir=out_dir)
        meta = json.loads((out_dir / "combat_meta.json").read_text())["annotator_disagreement"]

        assert meta["applied"] is False
        assert meta["source"] == "data/llm_annotations/agreement_v1.parquet"
        assert "ABSENT" in meta["status"]
        assert "inter-model-agreement-check" in meta["status"]

    def test_each_written_parquet_is_the_table_it_is_named_for(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """MUTATION-CHECK FINDING: only the SET of filenames was checked.

        Swapping two entries in the name map — writing `ratios` into
        `peak_decades.parquet` and vice versa — leaves the directory listing
        identical and passed the whole suite. Every downstream consumer and the
        report itself cite these files by name, so the mapping from table to
        filename is part of the published contract. Checked by content, and the
        frames are compared without an index so a stray index column is caught
        too."""
        out_dir = tmp_path / "combat"
        tables = C.build_combativeness(all_era_frame, out_dir=out_dir)

        for name, table in tables.items():
            back = pd.read_parquet(out_dir / f"{name}.parquet")
            expected = table.copy()
            for col in expected.columns:
                if isinstance(expected[col].dtype, pd.CategoricalDtype):
                    expected[col] = expected[col].astype(str)
            assert list(back.columns) == list(expected.columns), name
            assert len(back) == len(expected), name
            assert not [c for c in back.columns if c.startswith("__index_level")], name
            pd.testing.assert_frame_equal(
                back.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
            )

    def test_build_forwards_the_agreement_bands_into_BOTH_published_tables(
        self, all_era_frame, stub_real_data_leaves, tmp_path, monkeypatch
    ):
        """MUTATION-CHECK FINDING: `build_combativeness` could drop `bands` on
        the way into either table and nothing failed, because the real
        `agreement_v1.parquet` is absent so `bands` is None today and the two
        branches are indistinguishable. The seam is the deliverable of the
        deferred criterion, so it is exercised here with the loader stubbed —
        the real path is never created (it belongs to
        `inter-model-agreement-check`)."""
        bands = pd.DataFrame(
            [
                {"era": era, "flag": "party_attack", "disagreement_half_width": 0.02}
                for era in C.ERA_ORDER
            ]
        )
        monkeypatch.setattr(C, "load_agreement_bands", lambda *a, **k: bands)

        tables = C.build_combativeness(all_era_frame, out_dir=tmp_path / "combat")

        rates = tables["combativeness"]
        assert rates[rates["flag"] == "party_attack"]["disagreement_band_applied"].all()
        assert not rates[rates["flag"] == "zero_sum"]["disagreement_band_applied"].any()

        ratios = tables["ratios"]
        assert ratios[ratios["flag"] == "party_attack"]["disagreement_band_applied"].all()
        assert not ratios[ratios["flag"] == "zero_sum"]["disagreement_band_applied"].any()

        meta = json.loads((tmp_path / "combat" / "combat_meta.json").read_text())
        assert meta["annotator_disagreement"]["applied"] is True
        assert meta["annotator_disagreement"]["status"] == "applied"

    def test_meta_records_the_corpus_fingerprint_and_the_reference_weights(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """MUTATION-CHECK FINDING: two provenance fields were unasserted.

        `corpus_fingerprint` is what ties this artifact to the exact corpus it
        was computed over, and `reference_weights` is the genre distribution the
        standardized series is reweighted onto — without it a reader cannot
        reproduce or audit the `genre_standardized` column at all. Both could be
        emptied without a single test failing."""
        out_dir = tmp_path / "combat"
        C.build_combativeness(all_era_frame, out_dir=out_dir)
        meta = json.loads((out_dir / "combat_meta.json").read_text())

        assert meta["corpus_fingerprint"] == {
            "n_speeches": 12,
            "n_paragraphs": 72,
            "doc_name_sha256": "deadbeef",
        }
        weights = meta["genre_standardization"]["reference_weights"]
        assert set(weights) == set(all_era_frame["speech_type"].unique())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
        assert meta["genre_standardization"]["reference"] == (
            "pooled corpus paragraph share by speech_type"
        )
        assert meta["peak_decades"] == C.PEAK_DECADES
        assert meta["exemplars_per_cell"] == 10
        assert meta["ratios"]["reference_era"] == "The present era"

    def test_the_rate_table_and_the_ratio_table_share_one_set_of_fits(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """`build_combativeness` fits every cell once so the two published
        parquets cannot disagree about a rate."""
        tables = C.build_combativeness(all_era_frame, out_dir=tmp_path / "combat")
        rates, ratios = tables["combativeness"], tables["ratios"]
        for _, row in ratios.iterrows():
            den = _one(
                rates, era=row["denominator"], flag=row["flag"], treatment=row["treatment"]
            )
            assert row["denominator_rate"] == pytest.approx(den["rate"], nan_ok=True)


class TestCli:
    def _install(self, monkeypatch, tmp_path, frame):
        monkeypatch.setattr(C, "load_frame", lambda *a, **k: frame)
        monkeypatch.setattr(C, "COMBAT_DIR", tmp_path / "cli_out")

    def test_main_writes_the_tables_and_prints_all_three_treatments(
        self, monkeypatch, capsys, tmp_path, all_era_frame, stub_real_data_leaves
    ):
        self._install(monkeypatch, tmp_path, all_era_frame)
        monkeypatch.setattr(sys, "argv", ["pp-combat"])

        C.main()

        out = capsys.readouterr().out
        for treatment in C.TREATMENTS:
            assert f"=== {treatment} ===" in out
        assert "consistency" in out
        assert C.REFERENCE_ERA in out
        assert {p.name for p in (tmp_path / "cli_out").iterdir()} == EXPECTED_OUTPUT_FILES

    def test_quiet_writes_the_tables_without_printing(
        self, monkeypatch, capsys, tmp_path, all_era_frame, stub_real_data_leaves
    ):
        self._install(monkeypatch, tmp_path, all_era_frame)
        monkeypatch.setattr(sys, "argv", ["pp-combat", "--quiet"])

        C.main()

        assert capsys.readouterr().out == ""
        assert {p.name for p in (tmp_path / "cli_out").iterdir()} == EXPECTED_OUTPUT_FILES


# --------------------------------------------------------------------------- #
# $0 guard and the frozen artifacts
# --------------------------------------------------------------------------- #
class TestZeroCostAndFrozenArtifacts:
    def test_a_full_build_runs_with_client_construction_booby_trapped(
        self, all_era_frame, stub_real_data_leaves, tmp_path, anthropic_construction_bomb
    ):
        """Behavioural twin of the static check in test_combat_frame.py: driving
        every table with `anthropic.Anthropic` rigged to explode proves no code
        path reached for the paid API. conftest also strips the credentials."""
        tables = C.build_combativeness(all_era_frame, out_dir=tmp_path / "combat")
        # Every table but the meta JSON — derived so that adding an output does
        # not silently narrow what this $0 guard actually drove.
        assert len(tables) == len(EXPECTED_OUTPUT_FILES) - 1

    def test_no_write_target_lives_under_the_frozen_annotations_dir(self):
        """Read-side constants may point into `data/llm_annotations/`; write-side
        ones may not. Enumerated from the module so a new output path added
        without thought is caught."""
        write_targets = {
            name: getattr(C, name)
            for name in dir(C)
            if name.endswith("_PATH") and isinstance(getattr(C, name), Path)
        }
        read_only = {"PARAGRAPHS_PATH", "SPEECH_MARKERS_PATH", "PARAGRAPH_ENTITIES_PATH",
                     "AGREEMENT_PATH"}
        assert set(write_targets) - read_only == {
            "COMBATIVENESS_PATH",
            "PEAK_DECADES_PATH",
            "RATIOS_PATH",
            "EXEMPLARS_PATH",
            "ENTITY_CONSISTENCY_PATH",
            "LEXICAL_BASELINE_PATH",
            "ADVERSARY_MIX_PATH",
            "BY_PRESIDENT_PATH",
            "BY_PRESIDENT_TREATMENTS_V2_PATH",
            "BY_PRESIDENT_SPEAKER_V2_PATH",
            "TARGET_MIX_BY_ERA_SPEAKER_V1_PATH",
            "PRESIDENT_CONFLICT_V2_META_PATH",
            "GENRE_DECOMPOSITION_PATH",
            "COMBAT_META_PATH",
        }
        for name, path in write_targets.items():
            if name in read_only:
                continue
            assert path.parent == C.COMBAT_DIR, f"{name} writes outside data/combat/"
            assert C.ANNOTATIONS_DIR not in path.parents

    def test_a_build_leaves_the_frozen_annotation_artifacts_untouched(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """`data/llm_annotations/` is the $38.56 provenance-stamped output of a
        paid run. A research rebuild must not so much as restat it."""
        annotations_dir = C.ANNOTATIONS_DIR
        assert annotations_dir.exists(), "expected the frozen artifacts in this worktree"

        def snapshot():
            return {
                str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(annotations_dir.rglob("*"))
                if p.is_file()
            }

        before = snapshot()
        C.build_combativeness(all_era_frame, out_dir=tmp_path / "combat")
        assert snapshot() == before

    def test_a_build_with_an_out_dir_does_not_touch_the_committed_data_combat(
        self, all_era_frame, stub_real_data_leaves, tmp_path
    ):
        """`out_dir=` is the seam that lets a test build the real artifact shape
        without overwriting the committed outputs the report cites."""
        committed = C.COMBAT_DIR

        def snapshot():
            if not committed.exists():
                return None
            return {
                str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(committed.rglob("*"))
                if p.is_file()
            }

        before = snapshot()
        C.build_combativeness(all_era_frame, out_dir=tmp_path / "elsewhere")
        assert snapshot() == before


# --------------------------------------------------------------------------- #
# INTEGRATION (the only test in this suite that reads the real corpus)
# --------------------------------------------------------------------------- #
def test_integration_real_corpus_reproduces_the_published_anchor_numbers(monkeypatch):
    """SLOW / INTEGRATION — the single real-data test in this suite.

    Every other test runs on hand-authored frames. This one loads the actual
    36,229-paragraph corpus once (~0.5s) to confirm the wiring end to end still
    produces the numbers `notes/combativeness-findings-v1.md` publishes. A drift
    here means a join changed, not that a threshold moved.

    conftest's autouse `redirect_annotation_dirs` repoints ANNOTATIONS_DIR at
    tmp_path, so the real location is restored first — `combat` captured it at
    import time, before the fixture ran.
    """
    monkeypatch.setattr(ann, "ANNOTATIONS_DIR", C.ANNOTATIONS_DIR)
    if not (C.ANNOTATIONS_DIR / "paragraph_annotations.parquet").exists():
        pytest.skip("frozen annotation artifacts are not present in this checkout")

    df = C.load_frame()

    assert len(df) == 36_229
    assert df["doc_name"].nunique() == 1_057
    assert df["party_attack"].mean() == pytest.approx(0.0656, abs=5e-5)
    assert df["enemy_naming"].mean() == pytest.approx(0.2107, abs=5e-5)
    assert df["zero_sum"].mean() == pytest.approx(0.0768, abs=5e-5)

    fits = C._fit_all(df, "era", C.ERA_ORDER)
    sotu = C.rate_table(df, "era", C.ERA_ORDER, None, fits=fits)
    sotu = sotu[(sotu["treatment"] == "sotu_only") & (sotu["flag"] == "party_attack")]
    sotu = sotu.set_index("era")

    assert sotu.loc["The present era", "rate"] == pytest.approx(0.142197, abs=1e-6)
    assert sotu.loc["Civil War & Reconstruction", "rate"] == pytest.approx(0.029586, abs=1e-6)
    assert sotu.loc["War & New Deal", "rate"] == pytest.approx(0.045977, abs=1e-6)

    # ...and the n=3 War & New Deal cell is the reason the n-floor exists.
    assert sotu.loc["War & New Deal", "n_speeches"] == 3
    assert sotu.loc["War & New Deal", "ci_status"] == "suppressed_n_floor"
    assert np.isnan(sotu.loc["War & New Deal", "ci_lo"])
    assert sotu.loc["The present era", "n_speeches"] == 10
    assert sotu.loc["The present era", "ci_status"] == "low_cluster_caution"

    # the entity cross-check quoted in notes/annotation-qa-v1.md
    consistency = C.entity_consistency(df)
    assert int(consistency["n_backed"].sum()) == 7_393
    assert int(consistency["n_enemy_naming"].sum()) == 7_634

    # ------------------------------------------------------------------ #
    # THE HEADLINE ITSELF.
    #
    # MUTATION-CHECK FINDING: the marginal rates were pinned but the RATIO the
    # report actually quotes was not, so a change to the ratio bootstrap could
    # move "4.8x [2.3, 14.3]" without a single failing test. This is the
    # sentence that gets read aloud; it is anchored here to six figures.
    # ------------------------------------------------------------------ #
    ratios = C.ratio_table(df, "era", C.ERA_ORDER, C.REFERENCE_ERA, fits=fits)
    ratios = ratios[(ratios["treatment"] == "sotu_only")].set_index(["flag", "denominator"])

    headline = ratios.loc[("party_attack", "Civil War & Reconstruction")]
    assert headline["ratio"] == pytest.approx(4.806243, abs=1e-6)
    assert headline["ci_lo"] == pytest.approx(2.348699, abs=1e-6)
    assert headline["ci_hi"] == pytest.approx(14.315654, abs=1e-6)
    assert headline["ci_status"] == "low_cluster_caution"
    assert headline["ci_components"] == "sampling_only"

    # The other two flags tell different stories — which is why the report
    # forbids a composite index — and both of their intervals contain 1.
    founding_enemy = ratios.loc[("enemy_naming", "The founding")]
    assert founding_enemy["ratio"] == pytest.approx(1.091053, abs=1e-6)
    assert (founding_enemy["ci_lo"], founding_enemy["ci_hi"]) == pytest.approx(
        (0.786503, 1.612468), abs=1e-6
    )
    founding_zero_sum = ratios.loc[("zero_sum", "The founding")]
    assert founding_zero_sum["ratio"] == pytest.approx(1.761957, abs=1e-6)
    assert (founding_zero_sum["ci_lo"], founding_zero_sum["ci_hi"]) == pytest.approx(
        (0.945389, 4.878660), abs=1e-6
    )

    # ...and the n=3 War & New Deal cell poisons every ratio built on it.
    for flag in C.FLAGS:
        wnd = ratios.loc[(flag, "War & New Deal")]
        assert wnd["ci_status"] == "suppressed_n_floor", flag
        assert np.isnan(wnd["ci_lo"]) and np.isnan(wnd["ci_hi"]), flag
