"""Offline unit + one integration test for :mod:`presidential_profiles.eras`.

The era atlas is deterministic local compute over the frozen parquets, so almost
every test here drives an ``eras`` function on TINY synthetic frames (the repo's
testing convention, CLAUDE.md) and hand-derives the expected number. Three seams
get special attention because they are where the module is easiest to break
silently:

* the paid PORTRAIT path — the manifest must land under ``data/eras/manifests/``
  (NOT the frozen ``llm_annotations`` manifest dir), and no test ever constructs
  an Anthropic client (conftest deletes the key);
* the ``combativeness`` axis validation — it must refuse a non-``ok`` ``ci_status``
  raw cell and a raw-treatment mismatch;
* the optional annotator-disagreement loader — absent → ``None`` (the normal
  state), present-but-malformed → raise, and it must never claim it propagated a
  band it did not.

One slow-ish test (`TestRealCorpusRun`) drives the real ``run()`` over the frozen
corpus (~1.5 s) to anchor the published headline and byte-determinism; it is the
only test that touches the 36k-row corpus and it restores ``ANNOTATIONS_DIR``
(conftest's autouse ``redirect_annotation_dirs`` otherwise points the annotation
loads at an empty tmp dir).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import eras
from presidential_profiles import llm_annotations as ann
from presidential_profiles.annotate import _rates as _annotate_rates
from presidential_profiles.eras import ERA_ORDER

# ==========================================================================
# synthetic builders (local — tests/ is not a package, so no cross-import)
# ==========================================================================

# A 3-topic / 2-domain taxonomy shaped like taxonomy_v1 (name + level1).
SYNTH_TAX = {
    "level2": [
        {"name": "Jobs", "level1": "Economy"},
        {"name": "Trade", "level1": "Economy"},
        {"name": "War", "level1": "Security"},
    ]
}

# center year of each ERAS band, so one speech per era lands in the right era.
_ERA_MID_YEAR = {
    "The founding": 1800, "Expansion": 1830,
    "Civil War & Reconstruction": 1860, "The Gilded Age": 1890,
    "Progressives & Depression": 1915, "War & New Deal": 1940,
    "The Cold War": 1970, "Post-Cold War": 2000, "The present era": 2020,
}


def _master_from_specs(specs: list[dict]) -> pd.DataFrame:
    """Assemble a real master frame via ``load_master_frame`` from a compact spec.

    Each spec: ``{doc, year, president, speech_type, paras: [{topics, pa, en, zs,
    pv, words}]}``. Building through the real assembly (rather than hand-rolling
    the output columns) exercises ``load_master_frame`` and guarantees the
    downstream frames are shaped exactly as production produces them.
    """
    anns, paras, speeches, sanns = [], [], [], []
    for spec in specs:
        doc = spec["doc"]
        for i, p in enumerate(spec["paras"]):
            anns.append({
                "doc_name": doc, "para_idx": i,
                "topics": list(p.get("topics", [])),
                "party_attack": bool(p.get("pa", False)),
                "enemy_naming": bool(p.get("en", False)),
                "zero_sum": bool(p.get("zs", False)),
                "proposal_values": p.get("pv", "neither"),
            })
            paras.append({
                "doc_name": doc, "para_idx": i,
                "word_count": p.get("words", 50),
                "text": p.get("text", f"{doc} para {i} " + "word " * 30),
            })
        speeches.append({"doc_name": doc, "president": spec["president"],
                         "year": spec["year"]})
        sanns.append({"doc_name": doc, "speech_type": spec["speech_type"]})
    return eras.load_master_frame(
        annotations=pd.DataFrame(anns),
        speech_annotations=pd.DataFrame(sanns),
        paragraphs=pd.DataFrame(paras),
        speeches=pd.DataFrame(speeches),
        taxonomy=SYNTH_TAX,
    )


def _markers(docs, **overrides) -> pd.DataFrame:
    """A speech_markers-shaped frame: n_words + every MARKER_COLUMNS count."""
    rows = []
    for d in docs:
        row = {"doc_name": d, "n_words": overrides.get("n_words", 1000.0)}
        for c in eras.MARKER_COLUMNS:
            row[c] = float(overrides.get(c, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(docs, **overrides) -> pd.DataFrame:
    """A speech_stats-shaped frame: tokens/readability/pronouns/modals."""
    rows = []
    for d in docs:
        row = {
            "doc_name": d, "n_tokens": overrides.get("n_tokens", 1000.0),
            "fk_grade": overrides.get("fk_grade", 10.0),
            "words_per_sentence": overrides.get("words_per_sentence", 20.0),
            "i_count": overrides.get("i_count", 5.0),
            "we_count": overrides.get("we_count", 5.0),
        }
        for c in eras.MODAL_COLUMNS:
            row[c] = float(overrides.get(c, 2.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _combat_raw(rates_by_era: dict[str, tuple[float, float, float]],
                *, ci="ok", extra_treatments=False) -> pd.DataFrame:
    """A combativeness-shaped frame. ``rates_by_era`` maps era -> (party_attack,
    enemy_naming, zero_sum) raw rates."""
    order = {"party_attack": 0, "enemy_naming": 1, "zero_sum": 2}
    rows = []
    for era, rates in rates_by_era.items():
        for flag, r in zip(order, rates):
            rows.append({"era": era, "flag": flag, "treatment": "raw",
                         "rate": r, "ci_status": ci, "n_speeches": 20})
            if extra_treatments:
                rows.append({"era": era, "flag": flag, "treatment": "sotu_only",
                             "rate": r, "ci_status": "suppressed_n_floor",
                             "n_speeches": 3})
                rows.append({"era": era, "flag": flag,
                             "treatment": "genre_standardized", "rate": r,
                             "ci_status": "low_cluster_caution", "n_speeches": 4})
    return pd.DataFrame(rows)


def _period_frame(bins, **axis_cols) -> pd.DataFrame:
    """A minimal fine-grain axis frame for ``periodization`` (bin8 + center_year
    + the given prefixed axis columns)."""
    data = {"bin8": list(bins),
            "center_year": [float(b) + 3.5 for b in bins]}
    data.update({k: list(v) for k, v in axis_cols.items()})
    return pd.DataFrame(data)


def _era_fp(measures_by_era: dict[str, list[tuple[str, float]]]) -> pd.DataFrame:
    """A fingerprint-long-shaped frame (unit/measure/value_z/axis_group)."""
    rows = []
    for era, measures in measures_by_era.items():
        for measure, z in measures:
            rows.append({"unit": era, "measure": measure, "value_z": z,
                         "axis_group": eras.axis_group(measure)})
    return pd.DataFrame(rows)


# ==========================================================================
# PRIORITY 7 — keyed merges (_require_full_merge + load_master_frame)
# ==========================================================================


class TestRequireFullMerge:
    def test_passes_when_every_input_kept(self):
        # No raise when the merged count equals every input.
        eras._require_full_merge(5, {"a": 5, "b": 5}, "stage")

    def test_raises_when_an_input_lost_rows(self):
        with pytest.raises(ValueError, match="key sets diverge"):
            eras._require_full_merge(4, {"a": 5, "b": 4}, "annotations x paragraphs")

    def test_message_names_the_stage_and_breakdown(self):
        with pytest.raises(ValueError, match="my_stage.*a=5.*b=4"):
            eras._require_full_merge(4, {"a": 5, "b": 4}, "my_stage")


class TestLoadMasterFrame:
    def test_assembles_expected_columns_and_derived_fields(self):
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "Washington",
             "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"], "pa": True, "pv": "proposal"},
                       {"topics": ["War"], "en": True, "pv": "values"}]},
            {"doc": "d2", "year": 1862, "president": "Lincoln",
             "speech_type": "sotu",
             "paras": [{"topics": ["Trade", "Jobs"], "zs": True, "pv": "neither"}]},
        ])
        assert set(["doc_name", "para_idx", "year", "era", "president", "bin8",
                    "center_year", "speech_type"]).issubset(df.columns)
        # one boolean topic column per level-1 domain
        assert "topic__Economy" in df.columns and "topic__Security" in df.columns
        # 8-year bin + center: 1790 // 8 * 8 = 1784, center 1784 + 3.5
        founding = df[df["doc_name"] == "d1"].iloc[0]
        assert founding["bin8"] == 1784
        assert founding["center_year"] == 1784 + 3.5
        assert str(founding["era"]) == "The founding"
        # topic domain booleans track the normalized topics
        assert df[df["doc_name"] == "d1"].iloc[0]["topic__Economy"]  # Jobs
        assert df[df["doc_name"] == "d1"].iloc[1]["topic__Security"]  # War

    def test_era_is_ordered_categorical_in_eras_order(self):
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "W", "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"]}]},
        ])
        assert isinstance(df["era"].dtype, pd.CategoricalDtype)
        assert list(df["era"].cat.categories) == ERA_ORDER

    def test_raises_when_annotation_para_missing_from_paragraphs(self):
        # (d1,1) is annotated but absent from paragraphs — the inner join drops
        # it, and the row-completeness guard must catch the divergence.
        anns = pd.DataFrame([
            {"doc_name": "d1", "para_idx": 0, "topics": ["Jobs"],
             "party_attack": False, "enemy_naming": False, "zero_sum": False,
             "proposal_values": "neither"},
            {"doc_name": "d1", "para_idx": 1, "topics": ["War"],
             "party_attack": False, "enemy_naming": False, "zero_sum": False,
             "proposal_values": "neither"},
        ])
        paras = pd.DataFrame([{"doc_name": "d1", "para_idx": 0, "word_count": 50,
                               "text": "t"}])
        speeches = pd.DataFrame([{"doc_name": "d1", "president": "W", "year": 1790}])
        sanns = pd.DataFrame([{"doc_name": "d1", "speech_type": "inaugural"}])
        with pytest.raises(ValueError, match="key sets diverge|dropped"):
            eras.load_master_frame(annotations=anns, speech_annotations=sanns,
                                   paragraphs=paras, speeches=speeches,
                                   taxonomy=SYNTH_TAX)

    def test_raises_when_paragraph_doc_missing_from_speeches(self):
        # d2 has paragraphs+annotations but no speech row — the many_to_one merge
        # drops it, and the explicit post-merge length check fires.
        anns = pd.DataFrame([
            {"doc_name": "d1", "para_idx": 0, "topics": ["Jobs"],
             "party_attack": False, "enemy_naming": False, "zero_sum": False,
             "proposal_values": "neither"},
            {"doc_name": "d2", "para_idx": 0, "topics": ["War"],
             "party_attack": False, "enemy_naming": False, "zero_sum": False,
             "proposal_values": "neither"},
        ])
        paras = pd.DataFrame([
            {"doc_name": "d1", "para_idx": 0, "word_count": 50, "text": "t"},
            {"doc_name": "d2", "para_idx": 0, "word_count": 50, "text": "t"},
        ])
        speeches = pd.DataFrame([{"doc_name": "d1", "president": "W", "year": 1790}])
        sanns = pd.DataFrame([{"doc_name": "d1", "speech_type": "inaugural"}])
        with pytest.raises(ValueError, match="not in the speech table|row count"):
            eras.load_master_frame(annotations=anns, speech_annotations=sanns,
                                   paragraphs=paras, speeches=speeches,
                                   taxonomy=SYNTH_TAX)

    def test_raises_on_year_outside_every_era_band(self):
        anns = pd.DataFrame([{"doc_name": "d1", "para_idx": 0, "topics": ["Jobs"],
                              "party_attack": False, "enemy_naming": False,
                              "zero_sum": False, "proposal_values": "neither"}])
        paras = pd.DataFrame([{"doc_name": "d1", "para_idx": 0, "word_count": 50,
                               "text": "t"}])
        speeches = pd.DataFrame([{"doc_name": "d1", "president": "W", "year": 1700}])
        sanns = pd.DataFrame([{"doc_name": "d1", "speech_type": "inaugural"}])
        with pytest.raises(ValueError, match="outside every trends.ERAS band"):
            eras.load_master_frame(annotations=anns, speech_annotations=sanns,
                                   paragraphs=paras, speeches=speeches,
                                   taxonomy=SYNTH_TAX)


# ==========================================================================
# axis assembly (build_axis_frame + _weighted_mean + axis_columns/group)
# ==========================================================================


class TestWeightedMean:
    def test_weights_the_average(self):
        # (1*1 + 2*1 + 3*2) / 4 = 2.25
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 2.0])
        assert eras._weighted_mean(v, w) == pytest.approx(2.25)

    def test_zero_total_weight_is_nan(self):
        out = eras._weighted_mean(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
        assert np.isnan(out)


class TestBuildAxisFrame:
    def _frame_two_eras(self):
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "W", "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"], "pa": True, "pv": "proposal"},
                       {"topics": ["War"], "en": True, "pv": "values"}]},
            {"doc": "d2", "year": 1862, "president": "L", "speech_type": "sotu",
             "paras": [{"topics": ["Trade"], "zs": True, "pv": "neither"}]},
        ])
        markers = _markers(["d1", "d2"], boosters=1.0)
        stats = _stats(["d1", "d2"], i_count=5.0, we_count=5.0)
        return eras.build_axis_frame(df, "era", markers=markers, stats=stats,
                                     taxonomy=SYNTH_TAX)

    def test_combativeness_axes_are_raw_flag_means(self):
        ax = self._frame_two_eras().set_index("era")
        # founding d1: 2 paras, 1 party_attack, 1 enemy_naming, 0 zero_sum
        assert ax.loc["The founding", "combat_party_attack"] == pytest.approx(0.5)
        assert ax.loc["The founding", "combat_enemy_naming"] == pytest.approx(0.5)
        assert ax.loc["The founding", "combat_zero_sum"] == pytest.approx(0.0)
        # civil war d2: 1 para, zero_sum only
        assert ax.loc["Civil War & Reconstruction", "combat_zero_sum"] == pytest.approx(1.0)

    def test_marker_rate_is_per_1k_words(self):
        ax = self._frame_two_eras().set_index("era")
        # d1 boosters=1 over n_words=1000 -> 1/1000*1000 = 1.0 per 1k
        assert ax.loc["The founding", "marker_boosters"] == pytest.approx(1.0)

    def test_register_shares_and_genre_share(self):
        ax = self._frame_two_eras().set_index("era")
        # founding paras: proposal, values -> 0.5 / 0.5 / 0.0
        assert ax.loc["The founding", "register_proposal_share"] == pytest.approx(0.5)
        assert ax.loc["The founding", "register_values_share"] == pytest.approx(0.5)
        assert ax.loc["The founding", "register_neither_share"] == pytest.approx(0.0)
        # both founding paras are inaugural
        assert ax.loc["The founding", "genre__inaugural"] == pytest.approx(1.0)
        assert ax.loc["The founding", "genre__sotu"] == pytest.approx(0.0)

    def test_self_reference_ratio(self):
        # ASYMMETRIC i/we so the numerator is pinned: i/(i+we), NOT we/(i+we).
        # A 5/5 fixture makes the two indistinguishable (both 0.5) and lets a
        # numerator swap survive; 3 vs 9 forces the correct 0.25.
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "W", "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"]}]},
        ])
        stats = _stats(["d1"], i_count=3.0, we_count=9.0)
        ax = eras.build_axis_frame(df, "era", markers=_markers(["d1"]),
                                   stats=stats, taxonomy=SYNTH_TAX).set_index("era")
        # i/(i+we) = 3/12 = 0.25 (we/(i+we) would be 0.75)
        assert ax.loc["The founding", "stat_self_reference"] == pytest.approx(0.25)

    def test_rows_sorted_by_center_year(self):
        ax = self._frame_two_eras()
        assert list(ax["center_year"]) == sorted(ax["center_year"])

    def test_unknown_grain_raises(self):
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "W", "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"]}]},
        ])
        with pytest.raises(ValueError, match="unknown grain"):
            eras.build_axis_frame(df, "decade", markers=_markers(["d1"]),
                                  stats=_stats(["d1"]), taxonomy=SYNTH_TAX)

    def test_raises_when_a_doc_is_missing_from_speech_markers(self):
        # A corpus doc absent from speech_markers would have its marker rates
        # computed off a partial speech set while n_speeches still counts it — the
        # silent-partial class this module guards. The up-front coverage check must
        # RAISE and name the offending table.
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "W", "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"]}]},
            {"doc": "d2", "year": 1862, "president": "L", "speech_type": "sotu",
             "paras": [{"topics": ["War"]}]},
        ])
        with pytest.raises(ValueError, match="absent from speech_markers"):
            eras.build_axis_frame(df, "era", markers=_markers(["d1"]),   # d2 missing
                                  stats=_stats(["d1", "d2"]), taxonomy=SYNTH_TAX)

    def test_raises_when_a_doc_is_missing_from_speech_stats(self):
        df = _master_from_specs([
            {"doc": "d1", "year": 1790, "president": "W", "speech_type": "inaugural",
             "paras": [{"topics": ["Jobs"]}]},
            {"doc": "d2", "year": 1862, "president": "L", "speech_type": "sotu",
             "paras": [{"topics": ["War"]}]},
        ])
        with pytest.raises(ValueError, match="absent from speech_stats"):
            eras.build_axis_frame(df, "era", markers=_markers(["d1", "d2"]),
                                  stats=_stats(["d1"]), taxonomy=SYNTH_TAX)   # d2 missing


class TestAxisColumnsAndGroup:
    def test_axis_columns_selects_only_prefixed_columns_in_order(self):
        frame = pd.DataFrame(columns=[
            "era", "center_year", "n_paragraphs",  # not axes
            "topic__A", "combat_x", "register_y", "marker_z", "stat_q", "genre__g",
        ])
        assert eras.axis_columns(frame) == [
            "topic__A", "combat_x", "register_y", "marker_z", "stat_q", "genre__g"]

    @pytest.mark.parametrize("axis,group", [
        ("topic__Economy", "topic"),
        ("combat_zero_sum", "combat"),
        ("register_proposal_share", "register"),
        ("marker_opponents", "marker"),
        ("stat_fk_grade", "stat"),
        ("genre__sotu", "genre"),
    ])
    def test_axis_group_classifies_each_prefix(self, axis, group):
        assert eras.axis_group(axis) == group

    def test_axis_group_rejects_unknown(self):
        with pytest.raises(ValueError, match="matches no known group"):
            eras.axis_group("mystery_axis")


# ==========================================================================
# PRIORITY 5 — z-scoring (per-axis, ddof=0, zero-variance dropped)
# ==========================================================================


class TestZScore:
    def test_population_std_ddof0_values(self):
        # x=[1,3] -> mean 2, pop std 1 -> z=[-1,1]; y=[0,4] -> mean 2, pop std 2 -> z=[-1,1]
        frame = pd.DataFrame({"topic__x": [1.0, 3.0], "topic__y": [0.0, 4.0]})
        z, kept = eras.zscore(frame, ["topic__x", "topic__y"])
        assert kept == ["topic__x", "topic__y"]
        np.testing.assert_allclose(z, [[-1.0, -1.0], [1.0, 1.0]])

    def test_zero_variance_axis_is_dropped_not_nan(self):
        frame = pd.DataFrame({"topic__x": [1.0, 3.0], "topic__const": [5.0, 5.0]})
        z, kept = eras.zscore(frame, ["topic__x", "topic__const"])
        assert kept == ["topic__x"]           # constant axis dropped
        assert z.shape == (2, 1)
        assert not np.isnan(z).any()          # no 0/0 poison column

    def test_ddof0_not_ddof1(self):
        # three points [0,3,6]: pop std = sqrt(6) (ddof0); sample std = 3 (ddof1).
        frame = pd.DataFrame({"topic__x": [0.0, 3.0, 6.0]})
        z, _ = eras.zscore(frame, ["topic__x"])
        np.testing.assert_allclose(z[:, 0], (np.array([0, 3, 6]) - 3) / np.sqrt(6))


# ==========================================================================
# cosine / normalize
# ==========================================================================


class TestCosineMatrix:
    def test_orthogonal_identical_opposite(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        m = eras.cosine_matrix(vecs, ["a", "b", "c"])
        assert m.loc["a", "a"] == pytest.approx(1.0)     # self
        assert m.loc["a", "b"] == pytest.approx(0.0)     # orthogonal
        assert m.loc["a", "c"] == pytest.approx(-1.0)    # opposite
        assert m.loc["b", "c"] == pytest.approx(0.0)

    def test_labels_index_and_columns(self):
        m = eras.cosine_matrix(np.array([[1.0], [1.0]]), ["x", "y"])
        assert list(m.index) == ["x", "y"] and list(m.columns) == ["x", "y"]

    def test_l2_normalize_guards_zero_vector(self):
        out = eras._l2_normalize(np.array([[0.0, 0.0]]))
        assert np.isfinite(out).all()      # 0 / 1e-12 -> 0, never NaN/inf


# ==========================================================================
# PRIORITY 6 — detrending (era-grain adjacent; president-grain window+fallback)
# ==========================================================================


class TestDetrendAdjacentEras:
    def test_subtracts_adjacent_neighbor_mean(self):
        # [10,20,30,40]: i0 -10, i1 0, i2 0, i3 +10
        v = np.array([[10.0], [20.0], [30.0], [40.0]])
        out = eras.detrend_adjacent_eras(v)
        np.testing.assert_allclose(out[:, 0], [-10.0, 0.0, 0.0, 10.0])

    def test_endpoints_use_single_neighbor(self):
        v = np.array([[0.0], [4.0], [10.0]])
        out = eras.detrend_adjacent_eras(v)
        # i0: 0 - 4 = -4 ; i2: 10 - 4 = 6 ; i1: 4 - mean(0,10) = -1
        np.testing.assert_allclose(out[:, 0], [-4.0, -1.0, 6.0])


class TestDetrendLocalMean:
    def test_window_peers_and_isolated_fallback(self):
        # 1800/1805/1810 are mutual peers within 24y; 1900 has none -> 4-nearest
        years = np.array([1800.0, 1805.0, 1810.0, 1900.0])
        v = np.array([[1.0], [2.0], [3.0], [100.0]])
        out = eras.detrend_local_mean(v, years)
        np.testing.assert_allclose(out[:, 0], [
            1 - (2 + 3) / 2,        # -1.5
            2 - (1 + 3) / 2,        #  0.0
            3 - (1 + 2) / 2,        #  1.5
            100 - (1 + 2 + 3) / 3,  # 98.0 (fallback over the 3 available)
        ])

    def test_fallback_picks_exactly_the_four_nearest(self):
        # unit at 1900 has no window peer; the 4 nearest are 1851..1854 — 1850
        # (the farthest) MUST be excluded. If the code averaged all 5 others the
        # answer would be 10 - 20 = -10 instead of 10 - 0 = 10.
        years = np.array([1850.0, 1851.0, 1852.0, 1853.0, 1854.0, 1900.0])
        v = np.array([[100.0], [0.0], [0.0], [0.0], [0.0], [10.0]])
        out = eras.detrend_local_mean(v, years)
        assert out[5, 0] == pytest.approx(10.0)

    def test_window_is_inclusive_at_exactly_window_years(self):
        # The window uses `<= window` (inclusive). Unit at 1824 has peers at
        # diffs 24 (== window), 14, 4 -> all three are window peers, so it
        # subtracts mean(30,0,0)=10 -> 12-10 = 2. Under a STRICT `< window` the
        # 1800 peer (diff exactly 24) drops out, leaving two window peers
        # (mean 0) -> 12-0 = 12. The 2/12 split pins the boundary as inclusive.
        years = np.array([1800.0, 1810.0, 1820.0, 1824.0])
        v = np.array([[30.0], [0.0], [0.0], [12.0]])
        out = eras.detrend_local_mean(v, years)  # default window = ERA_WINDOW = 24
        assert out[3, 0] == pytest.approx(2.0)


# ==========================================================================
# group-balance weights
# ==========================================================================


class TestGroupBalanceWeights:
    def test_inverse_sqrt_group_size(self):
        kept = ["topic__a", "topic__b", "combat_x"]  # topic=2, combat=1
        w = eras.group_balance_weights(kept)
        np.testing.assert_allclose(w, [1 / np.sqrt(2), 1 / np.sqrt(2), 1.0])


class TestGroupBalancedEraDetrended:
    def test_returns_square_detrended_cosine_labelled_by_era(self):
        # A 4-era synthetic frame (in ERAS order, incl. the present era) exercises
        # the full group-balanced headline pipeline: zscore -> weight -> detrend
        # adjacent -> cosine.
        labels = ["The founding", "Expansion",
                  "Civil War & Reconstruction", "The present era"]
        frame = pd.DataFrame({
            "era": labels,
            "topic__a": [0.0, 1.0, 2.0, 3.0],
            "topic__b": [3.0, 1.0, 0.0, 2.0],
            "combat_x": [1.0, 0.0, 2.0, 0.5],
        })
        out = eras.group_balanced_era_detrended(frame)
        assert list(out.index) == labels and list(out.columns) == labels
        np.testing.assert_allclose(np.diag(out.to_numpy()), np.ones(len(labels)))

    def test_group_balanced_offdiagonal_matches_hand_derivation(self):
        # VALUE assertion (not just the diagonal): pins the whole composition
        # zscore(ddof0) -> *1/sqrt(n_group) -> adjacent detrend -> cosine, so a
        # dropped/altered weight-application (`zb = z * w` -> `zb = z`) that leaves
        # the diagonal at 1.0 is caught. 3 eras, topic group (2 axes) + combat
        # group (1 axis); weights 1/sqrt(2), 1/sqrt(2), 1.
        labels = ["The founding", "Expansion", "Civil War & Reconstruction"]
        frame = pd.DataFrame({
            "era": labels,
            "topic__a": [0.0, 1.0, 2.0],
            "topic__b": [0.0, 2.0, 1.0],
            "combat_x": [2.0, 0.0, 1.0],
        })
        out = eras.group_balanced_era_detrended(frame)
        # Hand-derived: with s=sqrt(3/2), q=sqrt(3)/2, the founding/Civil-War
        # detrended vectors are [-q,-2q,2s] and [q,-q,s]; their cosine is
        # (q^2+2s^2)/sqrt((5q^2+4s^2)(2q^2+s^2)) = 3.75/sqrt(29.25) = 0.693375.
        assert out.loc["The founding", "Civil War & Reconstruction"] == pytest.approx(
            0.693375, abs=1e-6)
        # The UNWEIGHTED (zb = z) cosine for this pair is 0.577350 — asserting the
        # group-balanced 0.693375 proves the 1/sqrt(n_group) weighting was applied.
        assert abs(out.loc["The founding", "Civil War & Reconstruction"] - 0.577350) > 1e-3


# ==========================================================================
# similarity long tables + nearest neighbors
# ==========================================================================


class TestSimilarityLong:
    def test_melts_and_carries_provenance(self):
        m = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
        order = {"a": 0, "b": 1}
        out = eras._similarity_long(m, "era", "raw", order, "sampling_only", "absent:x")
        assert len(out) == 4  # 2x2
        assert set(out["ci_components"]) == {"sampling_only"}
        assert set(out["agreement_source"]) == {"absent:x"}
        ab = out[(out["unit_i"] == "a") & (out["unit_j"] == "b")].iloc[0]
        assert ab["cosine"] == pytest.approx(0.5)
        assert ab["unit_i_order"] == 0 and ab["unit_j_order"] == 1


class TestNearestNeighbors:
    def test_picks_max_offdiagonal_and_flags_temporal(self):
        # a's nearest is c (non-adjacent -> False); b's nearest is a (adjacent
        # -> True); c's nearest is b (adjacent -> True).
        labels = ["a", "b", "c"]
        m = pd.DataFrame(
            [[1.0, 0.1, 0.9], [0.8, 1.0, 0.2], [0.5, 0.6, 1.0]],
            index=labels, columns=labels,
        )
        center = {"a": 1.0, "b": 2.0, "c": 3.0}
        out = eras.nearest_neighbors(m, "era", "raw", center, "sampling_only", "src")
        out = out.set_index("unit")
        assert out.loc["a", "nearest"] == "c"
        assert not out.loc["a", "nearest_is_temporal_neighbor"]
        assert out.loc["b", "nearest"] == "a"
        assert out.loc["b", "nearest_is_temporal_neighbor"]
        assert out.loc["c", "nearest"] == "b"
        assert out.loc["c", "nearest_is_temporal_neighbor"]
        assert out.loc["a", "cosine"] == pytest.approx(0.9)
        assert set(out["ci_components"]) == {"sampling_only"}


# ==========================================================================
# PRIORITY 4 — periodization / _boundaries_at_k (+ LOO) and PRIORITY 9 distinct
# ==========================================================================


class TestChainConnectivity:
    def test_path_graph_is_symmetric_with_only_adjacent_edges(self):
        adj = eras._chain_connectivity(4)
        assert adj[0, 1] == 1 and adj[1, 0] == 1
        assert adj[0, 2] == 0                          # non-adjacent
        assert (adj == adj.T).all()                    # symmetric
        assert np.trace(adj) == 0                       # no self-loops


class TestMatchCanonical:
    def test_nearest_canonical_and_signed_gap(self):
        # canonical boundaries include 1850 and 1878
        near, gap = eras._match_canonical(1848)
        assert near == 1850 and gap == -2
        near, gap = eras._match_canonical(1885)
        assert near == 1878 and gap == 7


class TestBoundariesAtK:
    def _z_two_blocks(self):
        # 6 units: first three near 0, last three near 10 -> one clean split.
        frame = _period_frame(
            [1800, 1808, 1816, 1824, 1832, 1840],
            topic__A=[0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        )
        z, _ = eras.zscore(frame, eras.axis_columns(frame))
        bins = frame["bin8"].to_numpy()
        return z, bins

    def test_k1_and_out_of_range_return_no_boundary(self):
        z, bins = self._z_two_blocks()
        assert eras._boundaries_at_k(z, bins, 1) == []
        assert eras._boundaries_at_k(z, bins, len(z) + 1) == []

    def test_clean_two_block_split(self):
        z, bins = self._z_two_blocks()
        assert eras._boundaries_at_k(z, bins, 2) == [1824]

    def test_contiguity_exactly_k_minus_1_boundaries(self):
        # With a contiguity constraint each of the k clusters is ONE run of time,
        # so the number of label transitions is exactly k-1. An interleaved
        # (non-contiguous) assignment would produce more than k-1 boundaries.
        z, bins = self._z_two_blocks()
        for k in range(2, len(z) + 1):
            assert len(eras._boundaries_at_k(z, bins, k)) == k - 1

    def test_deterministic_same_input_same_boundaries(self):
        z, bins = self._z_two_blocks()
        first = eras._boundaries_at_k(z, bins, 3)
        for _ in range(3):
            assert eras._boundaries_at_k(z, bins, 3) == first

    def test_contiguity_constraint_forces_contiguous_runs(self):
        # ALTERNATING signal [0,10,0,10,0,10]: an UNCONSTRAINED Ward would group
        # all the 0-bins into one cluster and all the 10-bins into another (two
        # temporally-interleaved clusters), producing a label transition at EVERY
        # step -> 5 boundaries at k=2. The path-graph connectivity forbids that:
        # each cluster must be one contiguous run of time, so k=2 yields exactly
        # ONE boundary. This is the test the clean two-block fixture cannot make —
        # there contiguous and unconstrained agree, so dropping `connectivity`
        # survives it. (The two-block fixture never exercises the constraint.)
        frame = _period_frame(
            [1800, 1808, 1816, 1824, 1832, 1840],
            topic__A=[0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
        )
        z, _ = eras.zscore(frame, eras.axis_columns(frame))
        bins = frame["bin8"].to_numpy()
        # exactly k-1 contiguous boundaries at every k (unconstrained -> 5 for all k)
        assert eras._boundaries_at_k(z, bins, 2) == [1840]
        for k in range(2, len(z) + 1):
            assert len(eras._boundaries_at_k(z, bins, k)) == k - 1


class TestPeriodization:
    def test_emits_full_and_drop_opponents_conditions(self):
        frame = _period_frame(
            [1800, 1808, 1816, 1824],
            topic__A=[0.0, 0.0, 10.0, 10.0],
            marker_opponents=[0.0, 1.0, 0.0, 1.0],
        )
        out = eras.periodization({"bin8": frame}, k_range=range(2, 4))
        assert set(out["condition"]) == {"full", "drop_opponents"}
        assert set(out["grain"]) == {"bin8"}
        # every boundary is matched to a nearest canonical + signed gap
        assert (out["matches_canonical"] == (out["gap_years"].abs() <= eras.BIN_WIDTH)).all()

    def test_loo_identity_when_opponents_is_redundant(self):
        # marker_opponents duplicates topic__A exactly -> carries no independent
        # signal, so dropping it MUST NOT move any boundary at any k.
        A = [0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0]
        B = [0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0]
        bins = [1800, 1808, 1816, 1824, 1832, 1840, 1848, 1856]
        frame = _period_frame(bins, topic__A=A, topic__B=B, marker_opponents=list(A))
        out = eras.periodization({"bin8": frame}, k_range=range(2, 7))
        for k in out["k"].unique():
            full = sorted(out[(out.condition == "full") & (out.k == k)]["boundary_year"])
            drop = sorted(out[(out.condition == "drop_opponents") & (out.k == k)]["boundary_year"])
            assert full == drop, f"k={k}: opponents was inert yet boundaries moved"

    def test_loo_counter_case_dropping_dominant_opponents_moves_boundaries(self):
        # marker_opponents carries the ONLY clean split (at 1832); topic__A alone
        # splits elsewhere. Dropping opponents MUST change at least one k — this
        # proves the LOO machinery is live, not inert.
        bins = [1800, 1808, 1816, 1824, 1832, 1840]
        frame = _period_frame(
            bins,
            topic__A=[0.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            marker_opponents=[0.0, 0.0, 0.0, 0.0, 100.0, 100.0],
        )
        out = eras.periodization({"bin8": frame}, k_range=range(2, 5))
        changed = False
        for k in out["k"].unique():
            full = sorted(out[(out.condition == "full") & (out.k == k)]["boundary_year"])
            drop = sorted(out[(out.condition == "drop_opponents") & (out.k == k)]["boundary_year"])
            if full != drop:
                changed = True
        assert changed, "dropping a dominant opponents axis left every boundary unmoved"

    def test_president_grain_uses_rounded_center_year_as_boundary(self):
        # No bin8 column -> boundary years come from round(center_year).
        frame = pd.DataFrame({
            "center_year": [1801.4, 1809.6, 1861.5, 1869.5],
            "topic__A": [0.0, 0.0, 10.0, 10.0],
        })
        out = eras.periodization({"president": frame}, k_range=range(2, 3))
        assert set(out["grain"]) == {"president"}
        # no marker_opponents column -> full and drop_opponents are identical; the
        # split falls between 1809.6 and 1861.5 -> boundary round(1861.5)=1862.
        full = out[out["condition"] == "full"]
        assert full["boundary_year"].tolist() == [1862]

    def test_matches_canonical_tolerance_is_exactly_one_bin_width(self):
        # The match tolerance is BIN_WIDTH (8y), pinned on BOTH sides so a
        # loosen (<= 2*BIN_WIDTH) or tighten (< BIN_WIDTH) is caught — the
        # `matches_canonical == (gap.abs() <= BIN_WIDTH)` self-check in
        # test_emits_* recomputes the SUT formula and cannot catch either.
        #
        # (a) gap 12 (> 8) MUST be a miss. President split at 1862; nearest
        # canonical 1850 -> gap +12. A 2*BIN_WIDTH loosening would wrongly
        # mark it recovered.
        miss = eras.periodization(
            {"president": pd.DataFrame({
                "center_year": [1801.4, 1809.6, 1861.5, 1869.5],
                "topic__A": [0.0, 0.0, 10.0, 10.0]})},
            k_range=range(2, 3),
        )
        miss = miss[miss["condition"] == "full"].iloc[0]
        assert miss["nearest_canonical"] == 1850 and miss["gap_years"] == 12
        assert bool(miss["matches_canonical"]) is False
        # (b) gap exactly 8 (== BIN_WIDTH) MUST be a hit (inclusive boundary).
        # bin8 split at 1824; nearest canonical 1816 -> gap +8.
        hit = eras.periodization(
            {"bin8": _period_frame([1808, 1816, 1824, 1832],
                                   topic__A=[0.0, 0.0, 10.0, 10.0])},
            k_range=range(2, 3),
        )
        hit = hit[hit["condition"] == "full"].iloc[0]
        assert hit["nearest_canonical"] == 1816 and hit["gap_years"] == 8
        assert bool(hit["matches_canonical"]) is True


class TestDistinctCanonicalCounting:
    """PRIORITY 9 — the periodization match accounting counts DISTINCT canonical
    boundaries (two discovered boundaries hitting the same canonical = 1). Pins
    the §5 defect fixed in 27e7387."""

    @staticmethod
    def _distinct_recovered(period, grain, condition="full"):
        sub = period[(period["grain"] == grain)
                     & (period["condition"] == condition)
                     & (period["is_primary_k"])
                     & (period["matches_canonical"])]
        return sub["nearest_canonical"].nunique(), len(sub)

    def test_distinct_less_than_rows_when_two_boundaries_share_a_canonical(self):
        # Synthetic periodization frame: 1881 and 1882 BOTH match canonical 1878.
        period = pd.DataFrame([
            {"grain": "president", "condition": "full", "is_primary_k": True,
             "matches_canonical": True, "nearest_canonical": 1878, "boundary_year": 1881},
            {"grain": "president", "condition": "full", "is_primary_k": True,
             "matches_canonical": True, "nearest_canonical": 1878, "boundary_year": 1882},
            {"grain": "president", "condition": "full", "is_primary_k": True,
             "matches_canonical": True, "nearest_canonical": 1946, "boundary_year": 1940},
        ])
        distinct, rows = self._distinct_recovered(period, "president")
        assert rows == 3            # three matching rows
        assert distinct == 2        # but only two DISTINCT canonicals

    def test_mutating_out_the_dedup_changes_the_count(self):
        # Guard against the §5 regression: counting matching ROWS (the buggy way)
        # over-counts relative to counting DISTINCT canonicals.
        period = pd.DataFrame([
            {"grain": "president", "condition": "full", "is_primary_k": True,
             "matches_canonical": True, "nearest_canonical": 1878, "boundary_year": 1881},
            {"grain": "president", "condition": "full", "is_primary_k": True,
             "matches_canonical": True, "nearest_canonical": 1878, "boundary_year": 1882},
        ])
        distinct, rows = self._distinct_recovered(period, "president")
        assert distinct != rows     # the dedup is what makes them differ

    def test_committed_periodization_president_grain_recovers_four_distinct(self):
        # Anchors the published number: 5 matching president-grain rows, but
        # 1881+1882 both hit 1878, so DISTINCT canonicals recovered = 4.
        period = pd.read_parquet(_REPO_DATA / "eras" / "periodization.parquet")
        distinct, rows = self._distinct_recovered(period, "president")
        assert rows == 5 and distinct == 4

    def test_committed_periodization_bin8_grain_recovers_six_distinct(self):
        period = pd.read_parquet(_REPO_DATA / "eras" / "periodization.parquet")
        distinct, rows = self._distinct_recovered(period, "bin8")
        assert rows == 6 and distinct == 6


# ==========================================================================
# PRIORITY 2 — combativeness ci_status audit + raw-treatment validation
# ==========================================================================


class TestCombativenessCiCheck:
    def test_only_raw_treatment_feeds_an_axis(self):
        comb = _combat_raw({"A": (0.1, 0.2, 0.3)}, extra_treatments=True)
        out = eras.combativeness_ci_check(comb)
        assert out[out["treatment"] == "raw"]["used_as_axis"].all()
        assert not out[out["treatment"] != "raw"]["used_as_axis"].any()

    def test_suppressed_cells_in_other_treatments_do_not_raise(self):
        # sotu_only/genre_standardized carry suppressed_n_floor cells; because
        # they never seed an axis, the audit passes (and records them).
        comb = _combat_raw({"A": (0.1, 0.2, 0.3)}, extra_treatments=True)
        out = eras.combativeness_ci_check(comb)
        assert (out["ci_status"] == "suppressed_n_floor").any()  # present but unused

    def test_raises_when_a_raw_cell_is_not_ok(self):
        comb = _combat_raw({"A": (0.1, 0.2, 0.3)}, ci="suppressed_n_floor")
        with pytest.raises(ValueError, match="non-ok cell"):
            eras.combativeness_ci_check(comb)

    def test_default_reads_the_real_combat_parquet_and_passes(self):
        # No arg -> reads the frozen combativeness.parquet; every raw cell is ok,
        # so the audit passes over real data.
        out = eras.combativeness_ci_check()
        assert (out[out["used_as_axis"]]["ci_status"] == "ok").all()


class TestValidateCombatAxes:
    def _era_axis(self, rates):
        rows = []
        for era, (pa, en, zs) in rates.items():
            rows.append({"era": era, "combat_party_attack": pa,
                         "combat_enemy_naming": en, "combat_zero_sum": zs})
        return pd.DataFrame(rows)

    def test_matching_axes_report_zero_diff(self):
        rates = {"A": (0.1, 0.3, 0.5), "B": (0.2, 0.4, 0.6)}
        era_axis = self._era_axis(rates)
        comb = _combat_raw(rates)
        diffs = eras.validate_combat_axes(era_axis, comb)
        assert diffs == {"party_attack": 0.0, "enemy_naming": 0.0, "zero_sum": 0.0}

    def test_raises_when_a_raw_rate_diverges(self):
        rates = {"A": (0.1, 0.3, 0.5), "B": (0.2, 0.4, 0.6)}
        comb = _combat_raw(rates)
        bad = self._era_axis(rates)
        bad.loc[0, "combat_party_attack"] = 0.999   # diverge from combat.parquet
        with pytest.raises(ValueError, match="disagree with combat.parquet raw"):
            eras.validate_combat_axes(bad, comb)


# ==========================================================================
# PRIORITY 3 — optional annotator-disagreement seam
# ==========================================================================


class TestAgreementSeam:
    def _write_bands(self, path, cols=("era", "flag", "disagreement_half_width")):
        pd.DataFrame({c: (["A"] if c in ("era", "flag") else [0.01]) for c in cols}
                     ).to_parquet(path)

    def test_absent_file_returns_none(self, tmp_path):
        assert eras.load_agreement_bands(path=tmp_path / "nope.parquet") is None

    def test_present_valid_file_returns_frame(self, tmp_path):
        p = tmp_path / "agreement_v1.parquet"
        self._write_bands(p)
        bands = eras.load_agreement_bands(path=p)
        assert bands is not None and len(bands) == 1

    def test_present_but_missing_columns_raises(self, tmp_path):
        p = tmp_path / "agreement_v1.parquet"
        self._write_bands(p, cols=("era",))   # missing flag + half_width
        with pytest.raises(ValueError, match="missing columns"):
            eras.load_agreement_bands(path=p)

    def test_default_path_is_bound_in_body_not_signature(self, tmp_path, monkeypatch):
        # The seam's contract: monkeypatching the module constant must actually
        # redirect a no-arg call. If the default were bound in the signature at
        # import time, this would silently read the real (absent) path -> None.
        p = tmp_path / "agreement_v1.parquet"
        self._write_bands(p)
        monkeypatch.setattr(eras, "AGREEMENT_PATH", p)
        bands = eras.load_agreement_bands()   # no path arg
        assert bands is not None and len(bands) == 1

    def test_real_agreement_artifact_is_still_absent(self):
        # Boundary guard: this task must NEVER create the other task's file.
        assert not eras.AGREEMENT_PATH.exists()

    def test_provenance_absent_is_sampling_only(self):
        assert eras._agreement_provenance(None) == (
            "sampling_only", f"absent:{eras.AGREEMENT_PATH.name}")

    def test_provenance_present_never_claims_annotator_was_propagated(self):
        # Even WITH a band table, the module has no cosine-propagation model, so
        # it must record sampling_only and never fabricate "sampling+annotator".
        bands = pd.DataFrame({"era": ["A"], "flag": ["party_attack"],
                              "disagreement_half_width": [0.01]})
        comp, source = eras._agreement_provenance(bands)
        assert comp == "sampling_only"
        assert "present_not_propagated" in source
        assert "sampling+annotator" not in comp


# ==========================================================================
# PRIORITY 8 — staleness guard
# ==========================================================================


class TestStaleness:
    def test_match_when_fingerprints_agree(self, tmp_path, monkeypatch):
        fp = {"n_speeches": 1, "n_paragraphs": 2, "doc_name_sha256": "abc"}
        meta = tmp_path / "combat_meta.json"
        meta.write_text(json.dumps({"corpus_fingerprint": fp}))
        monkeypatch.setattr(eras, "COMBAT_META_PATH", meta)
        out = eras.check_staleness(fp)
        assert out["combat_meta.json"]["match"] is True

    def test_mismatch_warns_loudly(self, tmp_path, monkeypatch, caplog):
        meta = tmp_path / "combat_meta.json"
        meta.write_text(json.dumps({"corpus_fingerprint": {"doc_name_sha256": "OLD"}}))
        monkeypatch.setattr(eras, "COMBAT_META_PATH", meta)
        with caplog.at_level("WARNING"):
            out = eras.check_staleness({"doc_name_sha256": "NEW"})
        assert out["combat_meta.json"]["match"] is False
        assert any("STALE INPUT" in r.message for r in caplog.records)

    def test_absent_meta_yields_empty_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(eras, "COMBAT_META_PATH", tmp_path / "nope.json")
        assert eras.check_staleness({"x": 1}) == {}

    def test_computes_live_fingerprint_when_omitted(self, tmp_path, monkeypatch):
        # No fingerprint passed -> it computes one; stub corpus_fingerprint so the
        # default branch runs without touching the real corpus.
        monkeypatch.setattr(eras, "corpus_fingerprint", lambda: {"sha": "live"})
        monkeypatch.setattr(eras, "COMBAT_META_PATH", tmp_path / "nope.json")
        assert eras.check_staleness() == {}


# ==========================================================================
# fingerprint long table
# ==========================================================================


class TestFingerprintLong:
    def test_value_z_matches_zscore_and_flags_zero_variance(self):
        frame = pd.DataFrame({
            "era": ["A", "B"],
            "center_year": [1790.0, 1862.0],
            "n_paragraphs": [2, 1],
            "n_speeches": [1, 1],
            "topic__x": [1.0, 3.0],       # varies -> z = [-1, 1]
            "topic__const": [5.0, 5.0],   # zero variance -> value_z 0, flagged
        })
        out = eras._fingerprint_long(frame, "era", "era")
        varying = out[out["measure"] == "topic__x"].set_index("unit")
        assert varying.loc["A", "value_z"] == pytest.approx(-1.0)
        assert varying.loc["B", "value_z"] == pytest.approx(1.0)
        assert not varying["zero_variance"].any()
        const = out[out["measure"] == "topic__const"]
        assert const["zero_variance"].all()
        assert (const["value_z"] == 0.0).all()
        # raw values preserved regardless of z-scoring
        assert varying.loc["A", "value_raw"] == pytest.approx(1.0)


# ==========================================================================
# similarity assembly determinism (light smoke, synthetic)
# ==========================================================================


class TestSimilarityDeterminism:
    def test_same_synthetic_input_twice_gives_identical_frames(self):
        frame = pd.DataFrame({
            "era": ERA_ORDER[:4],
            "topic__a": [0.0, 1.0, 2.0, 3.0],
            "topic__b": [3.0, 1.0, 0.0, 2.0],
        })

        def assemble():
            z, _ = eras.zscore(frame, ["topic__a", "topic__b"])
            labels = frame["era"].tolist()
            raw = eras.cosine_matrix(z, labels)
            det = eras.cosine_matrix(eras.detrend_adjacent_eras(z), labels)
            order = {e: i for i, e in enumerate(labels)}
            return pd.concat([
                eras._similarity_long(raw, "era", "raw", order, "sampling_only", "s"),
                eras._similarity_long(det, "era", "detrended", order, "sampling_only", "s"),
            ], ignore_index=True)

        assert assemble().equals(assemble())


# ==========================================================================
# PRIORITY 1 — portrait manifest routing + placeholder seam + helpers
# ==========================================================================


class TestWritePortraits:
    def _era_fp_all(self):
        return _era_fp({e: [("topic__Economy", 2.0), ("marker_opponents", -1.0),
                            ("combat_zero_sum", 0.5)] for e in ERA_ORDER})

    def test_manifest_lands_under_eras_dir_not_llm_annotations(
            self, tmp_path, monkeypatch):
        eras_dir = tmp_path / "eras"
        man_dir = eras_dir / "manifests"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", man_dir)
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        exemplar_ids = {e: [{"doc_name": "d1", "para_idx": 0, "year": 1790}]
                        for e in ERA_ORDER}
        texts = {e: f"portrait for {e}" for e in ERA_ORDER}
        usage = {"input_tokens": 1000, "output_tokens": 500}

        eras._write_portraits(eras_dir, exemplar_ids, self._era_fp_all(),
                              texts, usage)

        run_id = f"era-portraits-{date.today().isoformat()}"
        assert (man_dir / f"{run_id}.json").exists()          # correct dir
        # the frozen llm_annotations manifest dir (redirected at tmp by conftest)
        # must NOT have received the era manifest — that is the routing regression.
        assert not (ann.MANIFESTS_DIR / f"{run_id}.json").exists()

    def test_manifest_cost_is_actual_from_usage_not_estimate(
            self, tmp_path, monkeypatch):
        eras_dir = tmp_path / "eras"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", eras_dir / "manifests")
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        usage = {"input_tokens": 1000, "output_tokens": 500}
        eras._write_portraits(
            eras_dir, {e: [] for e in ERA_ORDER}, self._era_fp_all(),
            {e: "t" for e in ERA_ORDER}, usage)

        run_id = f"era-portraits-{date.today().isoformat()}"
        manifest = json.loads((eras_dir / "manifests" / f"{run_id}.json").read_text())
        in_rate, out_rate = _annotate_rates()
        expected = (1000 * in_rate + 500 * out_rate) / 1_000_000
        assert manifest["cost_usd"] == pytest.approx(round(expected, 6))
        assert manifest["annotation_files"] == ["era_portraits.parquet"]
        assert manifest["corpus_fingerprint"]["doc_name_sha256"] == "x"

    def test_writes_nine_generated_portraits(self, tmp_path, monkeypatch):
        eras_dir = tmp_path / "eras"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", eras_dir / "manifests")
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        eras._write_portraits(eras_dir, {e: [] for e in ERA_ORDER},
                              self._era_fp_all(), {e: "t" for e in ERA_ORDER},
                              {"input_tokens": 1, "output_tokens": 1})
        out = pd.read_parquet(eras_dir / "era_portraits.parquet")
        assert len(out) == len(ERA_ORDER)
        assert set(out["status"]) == {"generated"}

    def test_empty_completion_is_status_empty_never_generated(self, tmp_path, monkeypatch):
        # A model that returns an empty string produced NO portrait — it must be
        # marked "empty", never "generated" (which would imply real content).
        eras_dir = tmp_path / "eras"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", eras_dir / "manifests")
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        texts = {e: "real portrait" for e in ERA_ORDER}
        texts[ERA_ORDER[0]] = ""   # one era came back empty
        eras._write_portraits(eras_dir, {e: [] for e in ERA_ORDER},
                              self._era_fp_all(), texts,
                              {"input_tokens": 1, "output_tokens": 1})
        out = pd.read_parquet(eras_dir / "era_portraits.parquet").set_index("era")
        assert out.loc[ERA_ORDER[0], "status"] == "empty"
        assert (out.drop(ERA_ORDER[0])["status"] == "generated").all()

    def test_placeholder_branch_never_fabricates_a_portrait(self, tmp_path, monkeypatch):
        # texts is None -> placeholders, no manifest (the seam: never invent a
        # portrait when the model did not produce one).
        eras_dir = tmp_path / "eras"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", eras_dir / "manifests")
        eras._write_portraits(eras_dir, {e: [] for e in ERA_ORDER},
                              self._era_fp_all(), None, None)
        out = pd.read_parquet(eras_dir / "era_portraits.parquet")
        assert set(out["status"]) == {"not_generated"}
        assert out["portrait"].isna().all()
        assert not (eras_dir / "manifests").exists()   # no manifest written


class TestPortraitHelpers:
    def test_prompt_hash_is_stable_and_prefixed(self):
        assert eras._prompt_hash() == eras._prompt_hash()
        assert eras._prompt_hash().startswith("sha256:")

    def test_estimate_cost_assumes_max_output_per_prompt(self):
        prompts = ["a" * 400, "b" * 400, "c" * 400]
        est_in, est_out, cost = eras._estimate_cost(prompts)
        assert est_out == eras.PORTRAIT_MAX_TOKENS * len(prompts)
        in_rate, out_rate = _annotate_rates()
        assert cost == pytest.approx((est_in * in_rate + est_out * out_rate) / 1_000_000)

    @pytest.mark.parametrize("measure,expected", [
        ("topic__Economy", "topic emphasis: Economy"),
        ("combat_zero_sum", "combativeness: zero sum"),
        ("register_proposal_share", "register: proposal share"),
        ("marker_opponents", "style marker: opponents"),
        ("stat_fk_grade", "readability/usage: fk grade"),
        ("genre__sotu", "speech-type share: sotu"),
    ])
    def test_human_axis_labels(self, measure, expected):
        assert eras._human_axis(measure) == expected


class TestSelectExemplars:
    def _df_and_fp(self):
        df = _master_from_specs([
            {"doc": "d1", "year": 1800, "president": "W", "speech_type": "inaugural",
             "paras": [
                 {"topics": ["Jobs"], "words": 50, "text": "economy para long"},
                 {"topics": ["War"], "words": 40, "text": "security para"},
                 {"topics": ["Jobs"], "words": 5, "text": "too short"},   # < 20 words
             ]},
        ])
        era_fp = _era_fp({"The founding": [("topic__Economy", 3.0),
                                           ("topic__Security", -1.0)]})
        paragraphs = df[["doc_name", "para_idx"]].assign(
            text=["economy para long", "security para", "too short"])
        return df, era_fp, paragraphs

    def test_filters_word_window_and_scores_by_top_topics(self):
        df, era_fp, paragraphs = self._df_and_fp()
        out = eras.select_exemplars(df, "The founding", era_fp, paragraphs=paragraphs)
        # the 5-word paragraph is outside the 20-200 window and is excluded
        assert (out["word_count"] >= eras.EXEMPLAR_MIN_WORDS).all()
        assert "too short" not in set(out["text"])
        # top topic is Economy -> the Economy paragraph scores highest
        assert out.iloc[0]["para_idx"] == 0
        assert set(["doc_name", "para_idx", "year", "word_count", "text"]).issubset(out.columns)

    def test_deterministic(self):
        df, era_fp, paragraphs = self._df_and_fp()
        a = eras.select_exemplars(df, "The founding", era_fp, paragraphs=paragraphs)
        b = eras.select_exemplars(df, "The founding", era_fp, paragraphs=paragraphs)
        assert a.equals(b)


class TestBuildPortraitPrompt:
    def test_prompt_contains_era_header_features_and_exemplars(self):
        era_fp = _era_fp({"The founding": [("topic__Economy", 2.5),
                                           ("marker_opponents", -1.2)]})
        exemplars = pd.DataFrame([
            {"doc_name": "d1", "para_idx": 0, "year": 1800, "word_count": 50,
             "text": "we the people"},
        ])
        prompt = eras.build_portrait_prompt("The founding", era_fp, exemplars)
        assert "ERA: The founding (1789-1815)" in prompt
        assert "DISTINCTIVE FEATURES" in prompt
        assert "topic emphasis: Economy" in prompt
        assert "(1800) we the people" in prompt


# ==========================================================================
# generate_portraits — dry-run + money gates (no client ever constructed)
# ==========================================================================


def _install_portrait_stubs(monkeypatch, tmp_path):
    """Point generate_portraits at a synthetic 9-era corpus so it never reads the
    real 36k corpus, and return the run_data it expects."""
    specs = [
        {"doc": f"d{i}", "year": _ERA_MID_YEAR[e], "president": f"P{i}",
         "speech_type": "sotu",
         "paras": [{"topics": ["Jobs"], "words": 50, "text": f"{e} exemplar text here"}]}
        for i, e in enumerate(ERA_ORDER)
    ]
    df = _master_from_specs(specs)
    paragraphs = df[["doc_name", "para_idx"]].assign(
        text=[f"{doc} exemplar" for doc in df["doc_name"]])
    ppath = tmp_path / "paragraphs.parquet"
    paragraphs.to_parquet(ppath)
    monkeypatch.setattr(eras, "load_master_frame", lambda: df)
    monkeypatch.setattr(eras, "PARAGRAPHS_PATH", ppath)
    era_fp = _era_fp({e: [("topic__Economy", 2.0), ("marker_opponents", -1.0)]
                      for e in ERA_ORDER})
    return {"tables": {"era_fingerprints": era_fp}}


class TestGeneratePortraits:
    def test_dry_run_writes_estimate_and_constructs_no_client(
            self, tmp_path, monkeypatch, anthropic_construction_bomb):
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        out = eras.generate_portraits(run_data=run_data, dry_run=True,
                                      out_dir=tmp_path)
        assert out["dry_run"] is True
        est = json.loads((tmp_path / "portraits_estimate.json").read_text())
        assert est["n_requests"] == len(ERA_ORDER)
        assert est["model"] == eras.PORTRAIT_MODEL

    def test_refuses_without_yes_before_building_a_client(
            self, tmp_path, monkeypatch, anthropic_construction_bomb):
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match="without --yes"):
            eras.generate_portraits(run_data=run_data, dry_run=False, yes=False,
                                    out_dir=tmp_path)

    def test_refuses_above_cost_ceiling_before_building_a_client(
            self, tmp_path, monkeypatch, anthropic_construction_bomb):
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        monkeypatch.setattr(eras, "PORTRAIT_COST_CEILING_USD", -1.0)  # force breach
        with pytest.raises(RuntimeError, match="exceeds"):
            eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                    out_dir=tmp_path)

    def test_no_client_available_ships_not_generated_placeholders(
            self, tmp_path, monkeypatch):
        # yes=True, cost under ceiling, but the client fails to build -> the
        # graceful seam writes placeholders rather than fabricating portraits.
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        monkeypatch.setattr(eras, "MANIFESTS_DIR", tmp_path / "eras" / "manifests")
        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
        out = eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                      out_dir=tmp_path)
        assert out["dry_run"] is False
        portraits = pd.read_parquet(tmp_path / "era_portraits.parquet")
        assert set(portraits["status"]) == {"not_generated"}
        assert not (tmp_path / "eras" / "manifests").exists()

    def _fake_client(self, monkeypatch, fail_after=None, empty_eras=(),
                     bad_content_after=None):
        """A fake anthropic client driving the REAL portrait loop. Each
        `messages.create` returns a text block + usage; after `fail_after`
        successes it RAISES (partial-failure path); after `bad_content_after`
        successes it returns a BILLED response whose content parse raises (the
        usage-before-parse invariant); eras named in `empty_eras` come back empty.
        conftest deletes the API key, so nothing real can ever be constructed."""
        import types

        class _BadContent:
            def __iter__(self):
                raise ValueError("content parse boom")

        class _FakeMessages:
            calls = 0

            def create(self, *, messages, **kw):
                _FakeMessages.calls += 1
                if fail_after is not None and _FakeMessages.calls > fail_after:
                    raise RuntimeError("boom 500 from request")
                usage = types.SimpleNamespace(input_tokens=100, output_tokens=50)
                if bad_content_after is not None and _FakeMessages.calls > bad_content_after:
                    # billed (has usage) but the content join will raise
                    return types.SimpleNamespace(content=_BadContent(), usage=usage)
                body = messages[0]["content"]
                era = next((e for e in ERA_ORDER if e in body), "")
                text = "" if era in empty_eras else f"portrait {_FakeMessages.calls}"
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text=text)],
                    usage=usage,
                )

        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic",
                            lambda *a, **k: types.SimpleNamespace(messages=_FakeMessages()))

    def test_midloop_failure_persists_partial_manifest_then_reraises(
            self, tmp_path, monkeypatch):
        # A paid loop that fails at request 4 of 9 has ALREADY spent money on the 3
        # that succeeded: the partial manifest must land with the accumulated actual
        # cost + how many succeeded, and the exception must still propagate.
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        man_dir = tmp_path / "eras" / "manifests"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", man_dir)
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        self._fake_client(monkeypatch, fail_after=3)

        with pytest.raises(RuntimeError, match="boom 500"):
            eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                    out_dir=tmp_path)

        run_id = f"era-portraits-{date.today().isoformat()}"
        manifest = json.loads((man_dir / f"{run_id}.json").read_text())
        assert manifest["n_requests"] == 3                     # only the 3 that landed
        in_rate, out_rate = _annotate_rates()
        expected = (3 * 100 * in_rate + 3 * 50 * out_rate) / 1_000_000
        assert manifest["cost_usd"] == pytest.approx(round(expected, 6))  # ACTUAL, not 0
        assert "PARTIAL" in manifest["notes"]
        # no half-complete frozen artifact
        assert not (tmp_path / "era_portraits.parquet").exists()

    def test_full_success_writes_manifest_and_marks_empty_completion(
            self, tmp_path, monkeypatch):
        # The happy path through the real loop: all 9 land, the manifest records
        # the full run's actual cost, and an era that returned "" is status=empty
        # (never "generated").
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        man_dir = tmp_path / "eras" / "manifests"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", man_dir)
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        self._fake_client(monkeypatch, empty_eras=(ERA_ORDER[0],))

        eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                out_dir=tmp_path)

        portraits = pd.read_parquet(tmp_path / "era_portraits.parquet").set_index("era")
        assert portraits.loc[ERA_ORDER[0], "status"] == "empty"
        assert (portraits.drop(ERA_ORDER[0])["status"] == "generated").all()
        run_id = f"era-portraits-{date.today().isoformat()}"
        manifest = json.loads((man_dir / f"{run_id}.json").read_text())
        assert manifest["n_requests"] == len(ERA_ORDER)
        in_rate, out_rate = _annotate_rates()
        expected = (len(ERA_ORDER) * 100 * in_rate
                    + len(ERA_ORDER) * 50 * out_rate) / 1_000_000
        assert manifest["cost_usd"] == pytest.approx(round(expected, 6))

    def test_billed_request_whose_parse_raises_is_still_counted_in_cost(
            self, tmp_path, monkeypatch):
        # A request that was CHARGED for but whose content-extraction raises must
        # still land in the partial manifest's cost (usage counted before parse).
        # n_succeeded may exclude it (honest); its dollars must not be lost.
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        man_dir = tmp_path / "eras" / "manifests"
        monkeypatch.setattr(eras, "MANIFESTS_DIR", man_dir)
        monkeypatch.setattr(eras, "corpus_fingerprint",
                            lambda: {"n_speeches": 1, "n_paragraphs": 1,
                                     "doc_name_sha256": "x"})
        self._fake_client(monkeypatch, bad_content_after=2)  # 2 parse OK, 3rd billed but parse raises

        with pytest.raises(ValueError, match="content parse boom"):
            eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                    out_dir=tmp_path)

        run_id = f"era-portraits-{date.today().isoformat()}"
        manifest = json.loads((man_dir / f"{run_id}.json").read_text())
        assert manifest["n_requests"] == 2            # only 2 texts landed — honest
        assert manifest["input_tokens"] == 300        # but 3 requests were BILLED
        in_rate, out_rate = _annotate_rates()
        expected = (3 * 100 * in_rate + 3 * 50 * out_rate) / 1_000_000
        assert manifest["cost_usd"] == pytest.approx(round(expected, 6))  # includes req 3
        assert "PARTIAL" in manifest["notes"]

    def test_no_client_refuses_to_clobber_a_paid_artifact(self, tmp_path, monkeypatch):
        # `portraits --run --yes` without the key sourced: gates pass, client build
        # fails, and the placeholder seam must REFUSE to downgrade an existing paid
        # artifact — never silently overwrite $-paid portraits with not_generated.
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        monkeypatch.setattr(eras, "MANIFESTS_DIR", tmp_path / "eras" / "manifests")
        paid = pd.DataFrame([
            {"era": e, "era_order": i, "portrait": "paid text", "status": "generated",
             "top_axes": "[]", "exemplar_ids": "[]", "model": eras.PORTRAIT_MODEL,
             "run_id": "prior"}
            for i, e in enumerate(ERA_ORDER)])
        paid.to_parquet(tmp_path / "era_portraits.parquet", index=False)

        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
        with pytest.raises(RuntimeError, match="REFUSING to overwrite"):
            eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                    out_dir=tmp_path)
        # the paid artifact is byte-untouched (still all generated, original text)
        after = pd.read_parquet(tmp_path / "era_portraits.parquet")
        assert (after["status"] == "generated").all()
        assert set(after["portrait"]) == {"paid text"}

    def test_no_client_writes_placeholders_when_no_prior_artifact(
            self, tmp_path, monkeypatch):
        # complementary to the refusal test: with NO committed artifact, the
        # graceful placeholder seam still fires (a legitimate first build).
        run_data = _install_portrait_stubs(monkeypatch, tmp_path)
        monkeypatch.setattr(eras, "MANIFESTS_DIR", tmp_path / "eras" / "manifests")
        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
        eras.generate_portraits(run_data=run_data, dry_run=False, yes=True,
                                out_dir=tmp_path)
        out = pd.read_parquet(tmp_path / "era_portraits.parquet")
        assert set(out["status"]) == {"not_generated"}


# ==========================================================================
# real-corpus integration: headline anchors + byte-determinism (ONE slow test)
# ==========================================================================

_REPO_DATA = eras.DATA_DIR


@pytest.fixture
def _real_annotations(monkeypatch):
    """Undo conftest's autouse redirect_annotation_dirs for the few reads that
    must hit the real frozen annotation parquets. eras captured the real
    ANNOTATIONS_DIR at import (before the autouse fixture ran)."""
    monkeypatch.setattr(ann, "ANNOTATIONS_DIR", eras.ANNOTATIONS_DIR)


class TestRealCorpusRun:
    def test_headline_and_seam_state_over_the_frozen_corpus(
            self, tmp_path, _real_annotations):
        res = eras.run(out_dir=tmp_path)
        nn = res["tables"]["nearest_neighbors"]
        det_present = nn[(nn["grain"] == "era") & (nn["matrix"] == "detrended")
                         & (nn["unit"] == "The present era")].iloc[0]
        # the published headline: detrended, the present era rhymes with the
        # Civil War & Reconstruction era (non-adjacent) at cosine ~0.315.
        assert det_present["nearest"] == "Civil War & Reconstruction"
        assert det_present["cosine"] == pytest.approx(0.315, abs=0.01)
        assert not det_present["nearest_is_temporal_neighbor"]
        eras._print_summary(res)   # smoke: the CLI summary must not crash
        # the disagreement seam's active production state: sampling-only on every
        # similarity row (agreement_v1.parquet is absent).
        sim = res["tables"]["era_similarity"]
        assert set(sim["ci_components"]) == {"sampling_only"}
        assert set(sim["agreement_source"]) == {f"absent:{eras.AGREEMENT_PATH.name}"}
        assert res["meta"]["similarity"]["annotator_disagreement"]["applied"] is False

    def test_run_is_byte_identical_on_rerun(self, tmp_path, _real_annotations):
        a, b = tmp_path / "a", tmp_path / "b"
        eras.run(out_dir=a)
        eras.run(out_dir=b)
        names = ["era_fingerprints", "era_similarity", "fine_fingerprints",
                 "president_similarity", "nearest_neighbors", "periodization",
                 "combativeness_ci_check"]
        for n in names:
            d1 = hashlib.sha256((a / f"{n}.parquet").read_bytes()).hexdigest()
            d2 = hashlib.sha256((b / f"{n}.parquet").read_bytes()).hexdigest()
            assert d1 == d2, f"{n}.parquet not deterministic on rerun"
        # meta json too (no wall-clock stamp by design)
        assert (a / "eras_meta.json").read_bytes() == (b / "eras_meta.json").read_bytes()

    def test_combativeness_ci_check_confirms_no_suppressed_axis(
            self, tmp_path, _real_annotations):
        res = eras.run(out_dir=tmp_path)
        ci = res["tables"]["combativeness_ci_check"]
        used = ci[ci["used_as_axis"]]
        # every cell that seeds an axis is ci_status ok (the trust gate held)
        assert (used["ci_status"] == "ok").all()
        # and some suppressed cells DID exist in other treatments (proving the
        # audit is not vacuous — it excluded real suppressed cells).
        assert (ci[~ci["used_as_axis"]]["ci_status"] == "suppressed_n_floor").any()
