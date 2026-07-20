"""Tests for the corpus-taxonomy pipeline (`taxonomy.py`).

Every paid step (per-era proposal, merge, crosswalk, coverage labeling, the
anachronism check) is isolated behind a `_client()`-taking function and is NOT
exercised here — the module's design goal is that the sampling, coverage, and
structural-validation logic is pure and testable on synthetic frames without a
single API call. Per the repo test conventions we build tiny hand-made frames
with KNOWN structure rather than touching the real 36k-row corpus.

Coverage map:
  * attach_metadata        .. TestAttachMetadata      (keyed merge, unlabeled/era,
                                                       silent-shrink guard raises)
  * draw_heldout           .. TestDrawHeldout         (100 unlabeled, disjoint-ready)
  * draw_era_samples       .. TestDrawEraSamples      (oversample, backfill, seed)
  * register_samples       .. TestRegisterSamples     (stable ids -> real keys)
  * dominant_cluster_terms .. TestDominantClusters
  * format_paragraphs      .. TestFormatParagraphs    (id/year tag, truncation)
  * compute_coverage       .. TestComputeCoverage     (missing id == uncovered,
                                                       + level-2 name validation)
  * unknown_labels         .. TestComputeCoverage     (non-exact labels reported)
  * validate_structure     .. TestValidateStructure   (orphans, ranges, non-policy,
                                                       REQUIRED_NON_POLICY enforced,
                                                       malformed-input safe, dup names)
  * check_crosswalk        .. TestCheckCrosswalk      (mapped, dangling, empty, extra,
                                                       malformed-input safe)
  * resolve_exemplars      .. TestResolveExemplars    (drop invalid, dedup, <2 flag)
  * bail_reasons           .. TestBailReasons         (each hard bail condition)
  * revise_merge signature .. TestReviseMergeSignature (dropped unused proposals arg)
  * _extract_json          .. TestExtractJson         (refusal/max_tokens/no-text)
  * Spend.add              .. TestSpend               (cost from actual usage)
  * committed artifacts    .. TestCommittedArtifacts  (real taxonomy/crosswalk sane,
                                                       clears the bail gate)
"""

from __future__ import annotations

import inspect
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import taxonomy as T


# --------------------------------------------------------------------------- #
# frame builders
# --------------------------------------------------------------------------- #
def _paras(rows):
    """rows: list of (doc_name, para_idx, text)."""
    return pd.DataFrame(
        [{"doc_name": d, "para_idx": p, "text": t, "word_count": len(t.split())}
         for d, p, t in rows]
    )


def _issues(rows, fired):
    """rows: list of (doc_name, para_idx, president, year). `fired` maps
    (doc_name, para_idx) -> list of legacy issue names that are True."""
    out = []
    for d, p, pres, yr in rows:
        rec = {"doc_name": d, "para_idx": p, "president": pres, "year": yr}
        on = set(fired.get((d, p), []))
        for iss in T.LEGACY_ISSUES:
            rec[iss] = iss in on
        out.append(rec)
    return pd.DataFrame(out)


def _clusters(rows, k40):
    """k40 maps (doc_name, para_idx) -> cluster id."""
    return pd.DataFrame(
        [{"doc_name": d, "para_idx": p, "cluster_k40": k40[(d, p)], "cluster_k15": 0}
         for d, p, *_ in rows]
    )


# =========================================================================== #
class TestAttachMetadata:
    def test_keyed_merge_sources_year_and_marks_unlabeled(self):
        """paragraphs.parquet has no year; it comes from the issues table by the
        (doc_name, para_idx) key. A row with zero fired legacy issues is
        unlabeled; era is the 30-year band floor."""
        base = [("d", 0, "Adams", 1793), ("d", 1, "Adams", 1793)]
        paras = _paras([("d", 0, "tariff duty"), ("d", 1, "god faith")])
        issues = _issues(base, fired={("d", 0): ["Trade & tariffs"]})
        clusters = _clusters(base, {("d", 0): 2, ("d", 1): 6})

        df = T.attach_metadata(paras, issues, clusters)

        assert set(df["year"]) == {1793}
        assert not bool(df.loc[df["para_idx"] == 0, "unlabeled"].item())
        assert bool(df.loc[df["para_idx"] == 1, "unlabeled"].item())
        assert set(df["era"]) == {1770}  # 1793 // 30 * 30

    def test_merge_is_by_key_not_row_order(self):
        """Shuffling the issues table must not mislabel paragraphs — the merge is
        on the real key, so year/president follow the paragraph, not the row slot."""
        base = [("d", 0, "Adams", 1800), ("d", 1, "Jefferson", 1850)]
        paras = _paras([("d", 0, "alpha"), ("d", 1, "beta")])
        issues = _issues(base, fired={}).iloc[::-1].reset_index(drop=True)  # reversed
        clusters = _clusters(base, {("d", 0): 1, ("d", 1): 1})

        df = T.attach_metadata(paras, issues, clusters).sort_values("para_idx")

        assert list(df["year"]) == [1800, 1850]
        assert list(df["president"]) == ["Adams", "Jefferson"]

    def test_duplicate_key_raises(self):
        """validate='one_to_one' turns a duplicated key into a raise, not a
        silent fan-out."""
        base = [("d", 0, "Adams", 1800), ("d", 0, "Adams", 1800)]
        paras = _paras([("d", 0, "alpha"), ("d", 0, "beta")])
        issues = _issues(base, fired={})
        clusters = _clusters(base, {("d", 0): 1})
        with pytest.raises(Exception):
            T.attach_metadata(paras, issues, clusters)

    def test_unmatched_duplicate_key_in_issues_still_raises(self):
        """The paras×issues merge carries its OWN validate='one_to_one'; this pins
        it independently of the downstream clusters merge. issues holds ('d', 1)
        twice but paras has no ('d', 1) row, so the dup never fans out the join —
        yet one_to_one still rejects the non-unique right key. Relaxing THIS merge
        (e.g. to many_to_many) would let the dup slip through silently, because the
        clusters merge only ever sees keys that actually matched."""
        paras = _paras([("d", 0, "alpha")])
        issues = _issues(
            [("d", 0, "Adams", 1800), ("d", 1, "Adams", 1800), ("d", 1, "Adams", 1800)],
            fired={},
        )
        clusters = _clusters([("d", 0)], {("d", 0): 1})
        with pytest.raises(Exception):
            T.attach_metadata(paras, issues, clusters)

    def test_paras_row_missing_from_issues_raises_not_silently_shrinks(self):
        """A paragraph absent from the issues table would be DROPPED by the inner
        join (validate='one_to_one' only catches dup keys, not divergent key sets).
        That silent shrink — the sample becomes an intersection while the manifest
        fingerprints the full corpus — must raise with the count breakdown, per the
        keyed-merge convention (CLAUDE.md)."""
        paras = _paras([("d", 0, "alpha"), ("d", 1, "beta")])
        issues = _issues([("d", 0, "Adams", 1800)], fired={})  # ('d', 1) missing
        clusters = _clusters([("d", 0), ("d", 1)], {("d", 0): 1, ("d", 1): 1})
        with pytest.raises(ValueError, match="row count"):
            T.attach_metadata(paras, issues, clusters)

    def test_cluster_row_missing_raises(self):
        """The same guard on the second merge: a paragraph with no cluster row is
        dropped by the inner join and must raise rather than shrink the frame."""
        base = [("d", 0, "Adams", 1800), ("d", 1, "Adams", 1800)]
        paras = _paras([("d", 0, "alpha"), ("d", 1, "beta")])
        issues = _issues(base, fired={})
        clusters = _clusters([("d", 0)], {("d", 0): 1})  # ('d', 1) missing
        with pytest.raises(ValueError, match="row count"):
            T.attach_metadata(paras, issues, clusters)

    def test_full_key_overlap_does_not_raise(self):
        """The guard is silent on the healthy path: identical key sets across all
        three tables merge to the same row count and pass."""
        base = [("d", 0, "Adams", 1800), ("d", 1, "Adams", 1800)]
        paras = _paras([("d", 0, "alpha"), ("d", 1, "beta")])
        issues = _issues(base, fired={})
        clusters = _clusters(base, {("d", 0): 1, ("d", 1): 2})
        assert len(T.attach_metadata(paras, issues, clusters)) == 2

    def test_era_band_boundary_is_the_span_multiple(self):
        """Eras are 30-year floor bands: a year one below a multiple of ERA_SPAN
        stays in the lower band; the multiple itself opens the next band. This
        pins the boundary so a paragraph dated on an era edge is never mis-banded."""
        base = [("d", 0, "P", 1799), ("d", 1, "P", 1800),
                ("d", 2, "P", 1829), ("d", 3, "P", 1830)]
        paras = _paras([("d", i, "text") for i in range(4)])
        issues = _issues(base, fired={})
        clusters = _clusters(base, {("d", i): 0 for i in range(4)})

        df = T.attach_metadata(paras, issues, clusters).set_index("para_idx")

        assert df.loc[0, "era"] == 1770  # 1799 // 30 * 30
        assert df.loc[1, "era"] == 1800  # 1800 opens the next band
        assert df.loc[2, "era"] == 1800
        assert df.loc[3, "era"] == 1830


# =========================================================================== #
def _mixed_df(n_unlabeled=150, n_labeled=150, seed=0):
    """A frame spanning two eras with a controllable unlabeled/labeled split."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_unlabeled):
        recs.append({"doc_name": f"u{i}", "para_idx": 0, "text": f"unl {i}",
                     "year": 1800 + (i % 2) * 40, "president": "X",
                     "unlabeled": True, "n_legacy": 0, "cluster_k40": i % 5})
    for i in range(n_labeled):
        recs.append({"doc_name": f"l{i}", "para_idx": 0, "text": f"lab {i}",
                     "year": 1800 + (i % 2) * 40, "president": "Y",
                     "unlabeled": False, "n_legacy": 1, "cluster_k40": i % 5})
    df = pd.DataFrame(recs)
    df["era"] = (df["year"] // T.ERA_SPAN) * T.ERA_SPAN
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class TestDrawHeldout:
    def test_draws_hundred_unlabeled_when_available(self):
        df = _mixed_df()
        held = T.draw_heldout(df, seed=1)
        assert len(held) == T.HELDOUT_TOTAL
        assert int(held["unlabeled"].sum()) == T.HELDOUT_UNLABELED

    def test_is_reproducible_under_seed(self):
        df = _mixed_df()
        a = T.draw_heldout(df, seed=7).index.tolist()
        b = T.draw_heldout(df, seed=7).index.tolist()
        assert a == b

    def test_backfills_labeled_when_unlabeled_scarce(self):
        """With fewer than 100 unlabeled rows, the total is still 200 — the
        shortfall is made up from the labeled mass."""
        df = _mixed_df(n_unlabeled=30, n_labeled=400)
        held = T.draw_heldout(df, seed=2)
        assert len(held) == T.HELDOUT_TOTAL
        assert int(held["unlabeled"].sum()) == 30

    def test_can_be_removed_to_leave_a_disjoint_train_set(self):
        """The run drops held-out rows by index to form the training pool; those
        indices must exist in df so the drop is exact and disjoint."""
        df = _mixed_df()
        held = T.draw_heldout(df, seed=3)
        train = df.drop(held.index)
        assert len(train) == len(df) - len(held)
        assert set(held.index).isdisjoint(set(train.index))

    def test_zero_unlabeled_mass_fills_entirely_from_labeled(self):
        """A corpus with NO unlabeled paragraphs still yields a full 200-row
        held-out sample, drawn wholly from the labeled mass — the empty-pool
        branch must not short the sample."""
        df = _mixed_df(n_unlabeled=0, n_labeled=400)
        held = T.draw_heldout(df, seed=4)
        assert len(held) == T.HELDOUT_TOTAL
        assert int(held["unlabeled"].sum()) == 0


# =========================================================================== #
class TestDrawEraSamples:
    def test_oversamples_unlabeled_to_about_half(self):
        """Natural unlabeled rate here is 50%, but the draw targets per_era//2
        unlabeled explicitly — proven by making labeled dominant and still
        getting ~half unlabeled."""
        df = _mixed_df(n_unlabeled=100, n_labeled=400)
        samples = T.draw_era_samples(df, per_era=40, seed=5)
        for era, s in samples.items():
            assert len(s) == 40
            # ~half unlabeled (40//2 == 20), allowing for pool limits
            assert int(s["unlabeled"].sum()) == 20

    def test_backfills_when_an_era_pool_is_small(self):
        """If an era has too few unlabeled rows to hit the half target, the
        sample is filled from labeled rows rather than coming up short."""
        # era 1800 has few unlabeled, era 1840 has many.
        recs = []
        for i in range(5):
            recs.append({"doc_name": f"u{i}", "para_idx": 0, "text": "x", "year": 1800,
                         "president": "X", "unlabeled": True, "n_legacy": 0, "cluster_k40": 0})
        for i in range(100):
            recs.append({"doc_name": f"l{i}", "para_idx": 0, "text": "x", "year": 1800,
                         "president": "Y", "unlabeled": False, "n_legacy": 1, "cluster_k40": 0})
        df = pd.DataFrame(recs)
        df["era"] = (df["year"] // T.ERA_SPAN) * T.ERA_SPAN
        samples = T.draw_era_samples(df, per_era=40, seed=6)
        assert len(samples[1800]) == 40  # 5 unlabeled + 35 labeled

    def test_reproducible_under_seed(self):
        df = _mixed_df()
        a = {e: s["doc_name"].tolist() for e, s in T.draw_era_samples(df, 40, seed=9).items()}
        b = {e: s["doc_name"].tolist() for e, s in T.draw_era_samples(df, 40, seed=9).items()}
        assert a == b

    def test_era_smaller_than_per_era_returns_all_its_rows(self):
        """When an era holds fewer paragraphs than per_era, the draw takes every
        row it has (no replacement, no padding) rather than raising or duplicating."""
        recs = []
        for i in range(4):
            recs.append({"doc_name": f"u{i}", "para_idx": 0, "text": "x", "year": 1800,
                         "president": "X", "unlabeled": True, "n_legacy": 0, "cluster_k40": 0})
        for i in range(6):
            recs.append({"doc_name": f"l{i}", "para_idx": 0, "text": "x", "year": 1800,
                         "president": "Y", "unlabeled": False, "n_legacy": 1, "cluster_k40": 0})
        df = pd.DataFrame(recs)
        df["era"] = (df["year"] // T.ERA_SPAN) * T.ERA_SPAN
        s = T.draw_era_samples(df, per_era=40, seed=11)[1800]
        assert len(s) == 10  # all 10 rows, not 40
        assert s["doc_name"].nunique() == 10  # no row drawn twice

    def test_empty_unlabeled_mass_draws_all_from_labeled(self):
        """An era with zero unlabeled paragraphs still fills to per_era from the
        labeled pool — the oversample-unlabeled target degrades gracefully to 0."""
        recs = [
            {"doc_name": f"l{i}", "para_idx": 0, "text": "x", "year": 1900,
             "president": "Y", "unlabeled": False, "n_legacy": 1, "cluster_k40": 0}
            for i in range(30)
        ]
        df = pd.DataFrame(recs)
        df["era"] = (df["year"] // T.ERA_SPAN) * T.ERA_SPAN
        s = T.draw_era_samples(df, per_era=20, seed=12)[1890]
        assert len(s) == 20
        assert int(s["unlabeled"].sum()) == 0

    def test_backfill_draws_real_leftover_rows_when_labeled_pool_is_short(self):
        """The non-empty backfill draw: when the labeled pool can't reach per_era
        after the unlabeled half-target is taken, the shortfall is filled from the
        era's LEFTOVER unlabeled rows (60 unlabeled + 10 labeled, per_era=80 ->
        40 unlabeled + 10 labeled = 50, then 20 leftover unlabeled backfilled = 70).
        This branch fires for a small early era (abundant unlabeled, sparse labeled)
        and was previously unexercised with real rows to fill — a `fill = []`
        regression truncated the sample to 50 and slipped past every other test."""
        recs = []
        for i in range(60):
            recs.append({"doc_name": f"u{i}", "para_idx": 0, "text": "x", "year": 1770,
                         "president": "X", "unlabeled": True, "n_legacy": 0, "cluster_k40": 0})
        for i in range(10):
            recs.append({"doc_name": f"l{i}", "para_idx": 0, "text": "x", "year": 1770,
                         "president": "Y", "unlabeled": False, "n_legacy": 1, "cluster_k40": 0})
        df = pd.DataFrame(recs)
        df["era"] = (df["year"] // T.ERA_SPAN) * T.ERA_SPAN
        s = T.draw_era_samples(df, per_era=80, seed=13)[1770]
        assert len(s) == 70                     # 40 unl + 10 lab + 20 backfilled
        assert int(s["unlabeled"].sum()) == 60  # backfill drew from leftover unlabeled
        assert s["doc_name"].nunique() == 70    # no row drawn twice


# =========================================================================== #
class TestRegisterSamples:
    def test_assigns_stable_ids_mapping_to_real_keys(self):
        s = pd.DataFrame([
            {"doc_name": "d", "para_idx": 3, "text": "hello", "year": 1900},
        ])
        reg = T.register_samples({1890: s})
        assert list(reg) == ["p0001"]
        assert reg["p0001"]["doc_name"] == "d"
        assert reg["p0001"]["para_idx"] == 3
        assert reg["p0001"]["era"] == 1890

    def test_ids_are_globally_unique_across_eras(self):
        s1 = pd.DataFrame([{"doc_name": "a", "para_idx": 0, "text": "x", "year": 1800}])
        s2 = pd.DataFrame([{"doc_name": "b", "para_idx": 0, "text": "y", "year": 1900}])
        reg = T.register_samples({1800: s1, 1890: s2})
        assert set(reg) == {"p0001", "p0002"}


# =========================================================================== #
class TestDominantClusters:
    def test_ranks_clusters_by_era_frequency_with_terms(self):
        recs = []
        for i in range(10):
            recs.append({"era": 1800, "cluster_k40": 0})
        for i in range(3):
            recs.append({"era": 1800, "cluster_k40": 7})
        recs.append({"era": 1830, "cluster_k40": 0})  # other era ignored
        df = pd.DataFrame(recs)
        cmeta = {"clusters": {"k40": [
            {"cluster": 0, "terms": ["tariff", "duty"]},
            {"cluster": 7, "terms": ["treaty", "senate"]},
        ]}}
        out = T.dominant_cluster_terms(df, 1800, cmeta, top=5)
        assert out[0]["cluster"] == 0 and out[0]["size"] == 10
        assert out[1]["cluster"] == 7 and out[1]["size"] == 3
        assert out[0]["terms"] == ["tariff", "duty"]


# =========================================================================== #
class TestFormatParagraphs:
    def test_tags_each_paragraph_with_id_and_year(self):
        rows = [{"sample_id": "p0007", "year": 1793, "text": "tariff duty"}]
        assert T.format_paragraphs(rows) == "[p0007] (1793) tariff duty"

    def test_truncates_long_paragraphs(self):
        long = "word " * 1000
        rows = [{"sample_id": "p1", "year": 1800, "text": long}]
        out = T.format_paragraphs(rows)
        assert out.endswith("[...]")
        assert len(out) < len(long) + 30


# =========================================================================== #
class TestComputeCoverage:
    def test_fraction_with_at_least_one_label(self):
        labels = {"a": ["T1"], "b": [], "c": ["T2", "T3"]}
        assert T.compute_coverage(labels, ["a", "b", "c"]) == pytest.approx(2 / 3)

    def test_missing_id_counts_as_uncovered(self):
        """A paragraph the labeler dropped entirely is a coverage miss, not
        excluded from the denominator — the gate must not be gameable by
        silently omitting hard paragraphs."""
        labels = {"a": ["T1"]}  # 'b' absent
        assert T.compute_coverage(labels, ["a", "b"]) == pytest.approx(0.5)

    def test_empty_sample_is_zero(self):
        assert T.compute_coverage({}, []) == 0.0

    def test_unknown_label_only_paragraph_is_not_covered_when_names_validated(self):
        """With the frozen level-2 name set supplied, a paragraph whose only label
        is a string the model invented (not an exact topic name) counts as a MISS.
        Without the set, the legacy any-non-empty rule would have counted it — the
        two modes are contrasted so the stricter gate is unambiguous."""
        labels = {"a": ["Real"], "b": ["Ghost"], "c": []}
        ids = ["a", "b", "c"]
        assert T.compute_coverage(labels, ids) == pytest.approx(2 / 3)  # legacy
        assert T.compute_coverage(labels, ids, {"Real"}) == pytest.approx(1 / 3)

    def test_mix_of_known_and_unknown_labels_still_counts_covered(self):
        """One valid level-2 name is enough — a paragraph is not penalized for
        also carrying a near-miss string alongside a real label."""
        labels = {"a": ["Ghost", "Real"]}
        assert T.compute_coverage(labels, ["a"], {"Real"}) == pytest.approx(1.0)

    def test_missing_id_uncovered_under_name_validation(self):
        """The dropped-paragraph rule holds under name validation too: a missing
        id is a miss, never excluded from the denominator."""
        assert T.compute_coverage({"a": ["Real"]}, ["a", "b"], {"Real"}) == pytest.approx(0.5)

    def test_unknown_labels_reports_only_nonexact_names_sorted(self):
        labels = {"a": ["Real"], "b": ["Ghost", "Also Bad"], "c": [], "d": ["Real"]}
        assert T.unknown_labels(labels, {"Real"}) == ["Also Bad", "Ghost"]

    def test_unknown_labels_empty_when_all_exact(self):
        labels = {"a": ["Real"], "b": ["Other"]}
        assert T.unknown_labels(labels, {"Real", "Other"}) == []


# =========================================================================== #
class TestUncoveredIds:
    def test_is_the_strict_complement_of_the_coverage_gate(self):
        """uncovered_ids returns exactly the ids compute_coverage(valid_names)
        scores as misses, in sample order: a near-miss-only id ('Ghost'), an empty
        id, and a missing id are all uncovered; an id with any EXACT level-2 label
        is not. This is the predicate run() must use to build revise_merge's input,
        so the gate and the revision never disagree about which paragraphs failed."""
        labels = {"a": ["Real"], "b": ["Ghost"], "c": [], "d": ["Real", "Ghost"]}
        ids = ["a", "b", "c", "d", "e"]  # 'e' missing entirely
        assert T.uncovered_ids(labels, ids, {"Real"}) == ["b", "c", "e"]
        # lockstep with the gate: 2 of 5 covered => 3 uncovered.
        assert T.compute_coverage(labels, ids, {"Real"}) == pytest.approx(2 / 5)

    def test_legacy_any_nonempty_label_would_have_hidden_a_near_miss(self):
        """Contrast with the bug this closes: the old any-non-empty rule treated a
        near-miss-labeled paragraph as covered, so it never reached revise_merge
        even when the strict gate counted it against coverage. uncovered_ids uses
        the strict predicate, so 'b' (Ghost-only) is correctly surfaced."""
        labels = {"a": ["Real"], "b": ["Ghost"]}
        ids = ["a", "b"]
        assert T.uncovered_ids(labels, ids, {"Real"}) == ["b"]


# =========================================================================== #
def _tax(level1, level2):
    return {"level1": level1, "level2": level2}


# Non-policy level-1 domains named in the merge's OWN wording (not the bare
# REQUIRED_NON_POLICY keywords) — proves the presence check is semantic, not an
# exact-string match. Keys are the REQUIRED_NON_POLICY bucket ids.
_NON_POLICY_DOMAINS = {
    "ceremonial": "Ceremonial & Commemorative Address",
    "personal narrative": "Personal Narrative & Reflection",
    "procedural/administrative": "Procedural & Administrative",
    "values appeal": "Faith & National Values",
}


def _non_policy_l1(include=tuple(_NON_POLICY_DOMAINS)):
    return [
        {"name": _NON_POLICY_DOMAINS[b], "definition": "d", "kind": "non-policy"}
        for b in include
    ]


class TestValidateStructure:
    def test_counts_levels_and_non_policy(self):
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"},
             {"name": "Ceremonial", "definition": "d", "kind": "non-policy"}],
            [{"name": "Jobs", "definition": "d", "level1": "Economy", "era_note": "all"}],
        )
        r = T.validate_structure(tax)
        assert r["n_level1"] == 2 and r["n_level2"] == 1
        assert r["n_non_policy"] == 1 and r["non_policy_names"] == ["Ceremonial"]

    def test_flags_orphan_parent(self):
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"}],
            [{"name": "Jobs", "definition": "d", "level1": "Nonexistent", "era_note": "x"}],
        )
        assert T.validate_structure(tax)["orphan_parents"] == ["Nonexistent"]

    def test_flags_missing_required_level2_fields(self):
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"}],
            [{"name": "Jobs", "definition": "", "level1": "Economy", "era_note": "x"}],
        )
        assert T.validate_structure(tax)["level2_missing_fields"] == ["Jobs"]

    def test_range_and_hard_max_flags(self):
        small = _tax([], [{"name": f"t{i}", "definition": "d", "level1": "x",
                           "era_note": "n"} for i in range(10)])
        assert T.validate_structure(small)["in_target_range"] is False
        big = _tax([], [{"name": f"t{i}", "definition": "d", "level1": "x",
                         "era_note": "n"} for i in range(61)])
        assert T.validate_structure(big)["over_hard_max"] is True

    def test_all_four_required_non_policy_buckets_present_by_semantics(self):
        """The four required non-policy buckets are matched by the merge's own
        wording (e.g. 'Faith & National Values' satisfies the values-appeal
        bucket), not by exact string equality to the REQUIRED_NON_POLICY keys."""
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"}, *_non_policy_l1()],
            [{"name": "Jobs", "definition": "d", "level1": "Economy", "era_note": "n"}],
        )
        r = T.validate_structure(tax)
        assert r["missing_non_policy"] == []
        assert r["non_policy_complete"] is True

    @pytest.mark.parametrize("dropped", list(T.REQUIRED_NON_POLICY))
    def test_dropping_any_required_non_policy_bucket_is_flagged(self, dropped):
        """This is the guard the dead REQUIRED_NON_POLICY constant lacked: a re-run
        whose merge omits any one of the four non-policy buckets must surface that
        exact bucket in `missing_non_policy` and fail non_policy_complete."""
        kept = [b for b in T.REQUIRED_NON_POLICY if b != dropped]
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"},
             *_non_policy_l1(include=kept)],
            [{"name": "Jobs", "definition": "d", "level1": "Economy", "era_note": "n"}],
        )
        r = T.validate_structure(tax)
        assert r["missing_non_policy"] == [dropped]
        assert r["non_policy_complete"] is False

    def test_policy_only_taxonomy_misses_all_four_buckets(self):
        """A taxonomy with no non-policy domains at all reports every required
        bucket missing — not a silent pass on an empty non-policy set."""
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"}],
            [{"name": "Jobs", "definition": "d", "level1": "Economy", "era_note": "n"}],
        )
        assert T.validate_structure(tax)["missing_non_policy"] == list(T.REQUIRED_NON_POLICY)

    def test_never_raises_on_malformed_entries(self):
        """The docstring promises validate_structure never raises. A level-1 domain
        with no `name`, a level-2 topic with no `level1` and no `name` — all read
        through `.get()`, so the report comes back instead of a KeyError. The
        no-level1 topic is not an orphan (its parent is simply absent, not a broken
        reference to a nonexistent domain)."""
        tax = _tax(
            [{"kind": "non-policy"}, {"name": "Economy", "kind": "policy"}],
            [{"definition": "d", "era_note": "n"},           # no name, no level1
             {"name": "Jobs", "definition": "d", "era_note": "n"}],  # no level1
        )
        r = T.validate_structure(tax)  # must not raise
        assert r["orphan_parents"] == []          # absent parent != orphan
        assert "<unnamed>" in r["level2_missing_fields"]

    def test_orphan_detection_survives_a_nameless_level1(self):
        """A nameless level-1 entry must not mask a genuine orphan: a level-2 topic
        pointing at a domain that doesn't exist is still flagged even when another
        level-1 entry lacks a name (which is dropped from the resolvable-name set)."""
        tax = _tax(
            [{"kind": "policy"}, {"name": "Economy", "kind": "policy"}],
            [{"name": "Jobs", "definition": "d", "level1": "Ghost", "era_note": "n"}],
        )
        assert T.validate_structure(tax)["orphan_parents"] == ["Ghost"]

    def test_detects_case_and_whitespace_normalized_duplicate_level2_names(self):
        """Two level-2 topics that normalize to the same key ('Jobs' vs ' jobs ')
        would give the annotation pass an ambiguous label; surfaced (normalized) in
        `duplicate_level2_names`. A reported field only — not a bail condition."""
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"}],
            [{"name": "Jobs", "definition": "d", "level1": "Economy", "era_note": "n"},
             {"name": " jobs ", "definition": "d", "level1": "Economy", "era_note": "n"}],
        )
        assert T.validate_structure(tax)["duplicate_level2_names"] == ["jobs"]

    def test_distinct_level2_names_report_no_duplicates(self):
        tax = _tax(
            [{"name": "Economy", "definition": "d", "kind": "policy"}],
            [{"name": "Jobs", "definition": "d", "level1": "Economy", "era_note": "n"},
             {"name": "Trade", "definition": "d", "level1": "Economy", "era_note": "n"}],
        )
        assert T.validate_structure(tax)["duplicate_level2_names"] == []


def test_required_non_policy_is_exactly_the_four_acceptance_criteria_buckets():
    """Spec anchor, DECOUPLED from the constant the production code reads. The
    acceptance criteria name four first-class non-policy domains — ceremonial,
    personal narrative, procedural/administrative, values appeal. Every other
    non-policy test parametrizes over / references T.REQUIRED_NON_POLICY, so a
    regression that quietly drops or renames a bucket slips past them (they would
    simply check fewer buckets and still pass). Restating the requirement as a
    literal set makes drift in the requirement itself a failure, and pins the
    SYNONYMS map keys in lockstep so a bucket can never lose its matcher."""
    assert set(T.REQUIRED_NON_POLICY) == {
        "ceremonial",
        "personal narrative",
        "procedural/administrative",
        "values appeal",
    }
    assert set(T.REQUIRED_NON_POLICY_SYNONYMS) == set(T.REQUIRED_NON_POLICY)


# =========================================================================== #
class TestCheckCrosswalk:
    def _full_mappings(self, topic="Jobs"):
        return {"mappings": [{"legacy_issue": iss, "level2_topics": [topic]}
                             for iss in T.CROSSWALK_ISSUES]}

    def test_complete_when_every_issue_maps_to_a_real_topic(self):
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        r = T.check_crosswalk(self._full_mappings(), tax)
        assert r["complete"] is True
        assert r["missing_issues"] == []
        assert r["n_pages_reconstructable"] == len(T.CROSSWALK_ISSUES)

    def test_missing_issue_is_caught(self):
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        m = self._full_mappings()
        m["mappings"] = m["mappings"][:-1]  # drop the last issue
        r = T.check_crosswalk(m, tax)
        assert r["complete"] is False
        assert T.SECURITY_PEACE in r["missing_issues"]

    def test_dangling_topic_reference_is_caught(self):
        """A crosswalk pointing at a level-2 name that isn't in the taxonomy is a
        broken reference, not a valid mapping."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        r = T.check_crosswalk(self._full_mappings(topic="Ghost"), tax)
        assert r["dangling_topics"] == ["Ghost"]
        assert r["complete"] is False

    def test_single_dangling_topic_amid_valid_mappings_is_incomplete(self):
        """A single issue mapping to a nonexistent topic makes the crosswalk
        incomplete via `dangling_topics` even though that issue is technically
        'mapped' (non-empty list) — so it does NOT show up as a missing issue.
        The two failure modes are distinct and must not mask each other."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        m = self._full_mappings()  # all issues -> ["Jobs"]
        m["mappings"][0]["level2_topics"] = ["Ghost"]  # first issue points at nothing
        r = T.check_crosswalk(m, tax)
        assert r["dangling_topics"] == ["Ghost"]
        assert r["missing_issues"] == []  # non-empty list => not "missing"
        assert r["complete"] is False

    def test_empty_topic_list_counts_as_a_missing_issue(self):
        """An issue whose level2_topics is an empty list backs no page and is a
        missing issue, not a satisfied mapping."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        m = self._full_mappings()
        m["mappings"][0]["level2_topics"] = []
        r = T.check_crosswalk(m, tax)
        assert m["mappings"][0]["legacy_issue"] in r["missing_issues"]
        assert r["complete"] is False

    def test_extra_unknown_legacy_issue_does_not_break_completeness(self):
        """A mapping for a legacy issue that isn't one of CROSSWALK_ISSUES is
        ignored for completeness (only the required 16 are scored). With every
        required issue mapped to a real topic the crosswalk is still complete —
        the extra is tolerated. It IS surfaced in `unknown_issues` (crosswalk
        model confusion), but that report field does not touch completeness."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        m = self._full_mappings()
        m["mappings"].append({"legacy_issue": "Not A Real Issue", "level2_topics": ["Jobs"]})
        r = T.check_crosswalk(m, tax)
        assert r["missing_issues"] == []
        assert r["dangling_topics"] == []
        assert r["unknown_issues"] == ["Not A Real Issue"]
        assert r["complete"] is True

    def test_no_unknown_issues_when_only_required_are_mapped(self):
        """The additive `unknown_issues` field is empty for a well-formed crosswalk
        that maps only the required 16 — it fires solely on extras."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        r = T.check_crosswalk(self._full_mappings(), tax)
        assert r["unknown_issues"] == []

    def test_mapping_missing_legacy_issue_is_skipped_not_raised(self):
        """A mapping entry with no `legacy_issue` key is skipped (not KeyError-ed,
        not sorted into unknown_issues as a None). With every required issue still
        mapped, the crosswalk is complete."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        m = self._full_mappings()
        m["mappings"].append({"level2_topics": ["Jobs"]})  # no legacy_issue
        r = T.check_crosswalk(m, tax)  # must not raise
        assert r["complete"] is True
        assert r["unknown_issues"] == []

    def test_null_level2_topics_treated_as_empty_not_iterated_as_none(self):
        """A mapping whose `level2_topics` is null is treated as the empty list: the
        issue counts as missing, and the dangling scan never tries to iterate None."""
        tax = _tax([], [{"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        m = self._full_mappings()
        m["mappings"][0]["level2_topics"] = None
        r = T.check_crosswalk(m, tax)  # must not raise
        assert m["mappings"][0]["legacy_issue"] in r["missing_issues"]
        assert r["complete"] is False

    def test_level2_entry_without_name_does_not_break_dangling_scan(self):
        """A taxonomy level-2 entry lacking a `name` is dropped from the resolvable
        set via `.get()`; a crosswalk pointing at a real name still resolves and a
        bogus one is still dangling — no KeyError from the malformed entry."""
        tax = _tax([], [{"definition": "d", "level1": "x", "era_note": "n"},
                        {"name": "Jobs", "definition": "d", "level1": "x", "era_note": "n"}])
        r = T.check_crosswalk(self._full_mappings(topic="Jobs"), tax)
        assert r["dangling_topics"] == []
        assert r["complete"] is True


# =========================================================================== #
class TestResolveExemplars:
    def test_maps_valid_ids_and_drops_invalid(self):
        reg = {"p0001": {"doc_name": "d", "para_idx": 4},
               "p0002": {"doc_name": "e", "para_idx": 7}}
        tax = _tax([], [{"name": "T", "definition": "d", "level1": "x",
                         "era_note": "n", "exemplars": ["p0001", "p9999", "p0002"]}])
        stats = T.resolve_exemplars(tax, reg)
        ex = tax["level2"][0]["exemplar_paragraphs"]
        assert ex == [{"doc_name": "d", "para_idx": 4}, {"doc_name": "e", "para_idx": 7}]
        assert stats["topics_under_two_exemplars"] == []

    def test_dedups_repeated_exemplars(self):
        reg = {"p0001": {"doc_name": "d", "para_idx": 4}}
        tax = _tax([], [{"name": "T", "definition": "d", "level1": "x",
                         "era_note": "n", "exemplars": ["p0001", "p0001"]}])
        T.resolve_exemplars(tax, reg)
        assert tax["level2"][0]["exemplar_paragraphs"] == [{"doc_name": "d", "para_idx": 4}]

    def test_flags_topics_with_fewer_than_two_valid_exemplars(self):
        reg = {"p0001": {"doc_name": "d", "para_idx": 4}}
        tax = _tax([], [{"name": "Thin", "definition": "d", "level1": "x",
                         "era_note": "n", "exemplars": ["p0001", "p_bad"]}])
        stats = T.resolve_exemplars(tax, reg)
        assert stats["topics_under_two_exemplars"] == ["Thin"]


# =========================================================================== #
def _clean_report(**over):
    r = {"n_level2": 50, "over_hard_max": False, "orphan_parents": []}
    r.update(over)
    return r


def _clean_xw(**over):
    x = {"missing_issues": [], "dangling_topics": []}
    x.update(over)
    return x


class TestBailReasons:
    def test_clean_run_has_no_bail_reasons(self):
        """A passing run — coverage at/above the gate, level-2 count under the hard
        max, no orphans, complete crosswalk — yields an empty list, so run() writes
        the artifacts."""
        assert T.bail_reasons(1.0, _clean_report(), _clean_xw()) == []

    def test_coverage_exactly_at_gate_passes(self):
        """The gate is `< COVERAGE_GATE`, so landing exactly on it is a pass."""
        assert T.bail_reasons(T.COVERAGE_GATE, _clean_report(), _clean_xw()) == []

    def test_coverage_below_gate_bails(self):
        reasons = T.bail_reasons(0.5, _clean_report(), _clean_xw())
        assert len(reasons) == 1 and "coverage" in reasons[0]

    def test_over_hard_max_bails(self):
        reasons = T.bail_reasons(1.0, _clean_report(over_hard_max=True, n_level2=61), _clean_xw())
        assert len(reasons) == 1 and "hard max" in reasons[0]

    def test_orphan_parents_bail(self):
        reasons = T.bail_reasons(1.0, _clean_report(orphan_parents=["Ghost"]), _clean_xw())
        assert len(reasons) == 1 and "nonexistent level-1 parents" in reasons[0]

    def test_missing_crosswalk_issue_bails(self):
        reasons = T.bail_reasons(1.0, _clean_report(), _clean_xw(missing_issues=["Education"]))
        assert len(reasons) == 1 and "no crosswalk target" in reasons[0]

    def test_dangling_crosswalk_topic_bails(self):
        """A crosswalk pointing at a nonexistent level-2 topic breaks page
        reconstruction exactly as a missing issue does — same integrity failure."""
        reasons = T.bail_reasons(1.0, _clean_report(), _clean_xw(dangling_topics=["Ghost"]))
        assert len(reasons) == 1 and "nonexistent level-2 topics" in reasons[0]

    def test_multiple_failures_are_all_reported(self):
        """Every failing condition is surfaced in one pass so a bad run's full
        story is in the raised message, not just the first tripped gate."""
        reasons = T.bail_reasons(
            0.5,
            _clean_report(over_hard_max=True, n_level2=61, orphan_parents=["G"]),
            _clean_xw(missing_issues=["Education"], dangling_topics=["Z"]),
        )
        assert len(reasons) == 5


class TestReviseMergeSignature:
    def test_unused_proposals_param_was_dropped(self):
        """Fix: revise_merge no longer declares the `proposals` param it never used.
        The current taxonomy (passed in full, with each topic's exemplar ids) is the
        sole carry-forward source, so the era proposals aren't re-threaded."""
        params = inspect.signature(T.revise_merge).parameters
        assert "proposals" not in params
        assert list(params) == ["client", "taxonomy", "uncovered", "spend"]

    def test_revise_prompt_forbids_holdout_ids_as_exemplars(self):
        """REVISE_SYSTEM was reworded off the misleading 'ids present in the material
        you were given' (which included the uncovered paragraphs' held-out hNNNN
        ids) to explicitly forbid those ids as exemplars."""
        assert "hNNNN" in T.REVISE_SYSTEM
        assert "material you were given" not in T.REVISE_SYSTEM


# =========================================================================== #
class TestSpend:
    def test_accumulates_cost_from_actual_usage(self):
        """cost_usd is derived from real usage numbers at each model's rate, and
        accumulates across calls and models."""
        spend = T.Spend()
        opus = types.SimpleNamespace(
            input_tokens=1_000_000, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0)
        spend.add("claude-opus-4-8", opus)
        assert spend.cost_usd == pytest.approx(5.0)  # $5/M input

        sonnet = types.SimpleNamespace(
            input_tokens=0, output_tokens=1_000_000,
            cache_creation_input_tokens=0, cache_read_input_tokens=0)
        spend.add("claude-sonnet-5", sonnet)
        assert spend.cost_usd == pytest.approx(5.0 + 10.0)  # + $10/M output
        assert spend.n_requests == 2

    def test_counts_cache_tokens_at_their_own_rates(self):
        spend = T.Spend()
        usage = types.SimpleNamespace(
            input_tokens=0, output_tokens=0,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000)
        spend.add("claude-opus-4-8", usage)
        # cache-write $6.25/M + cache-read $0.50/M
        assert spend.cost_usd == pytest.approx(6.25 + 0.50)


# =========================================================================== #
def _resp(stop_reason, blocks, stop_details=None):
    """A forged structured-output response: only the attributes _extract_json
    reads (stop_reason, stop_details, content). No client, no network."""
    return types.SimpleNamespace(
        stop_reason=stop_reason, stop_details=stop_details, content=blocks
    )


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


class TestExtractJson:
    def test_parses_valid_structured_json(self):
        resp = _resp("end_turn", [_text_block('{"topics": ["a", "b"]}')])
        assert T._extract_json(resp) == {"topics": ["a", "b"]}

    def test_reads_the_text_block_past_a_non_text_block(self):
        """Adaptive thinking prepends a non-text block; _extract_json must select
        the text block (and never touch .text on a thinking block, which lacks it)."""
        blocks = [types.SimpleNamespace(type="thinking"), _text_block('{"ok": 1}')]
        assert T._extract_json(_resp("end_turn", blocks)) == {"ok": 1}

    def test_refusal_stop_reason_raises_with_details(self):
        """A refusal must surface loudly (with stop_details) rather than fall
        through to json.loads on a missing/empty body."""
        resp = _resp("refusal", [_text_block("{}")], stop_details="usage_policy")
        with pytest.raises(RuntimeError, match="refused"):
            T._extract_json(resp)

    def test_max_tokens_stop_reason_raises(self):
        """A truncated response is surfaced, not parsed as if complete —
        json.loads on a half-written object would raise an opaque error instead."""
        resp = _resp("max_tokens", [_text_block('{"topics": [')])
        with pytest.raises(RuntimeError, match="max_tokens"):
            T._extract_json(resp)

    def test_no_text_block_raises_descriptive_runtimeerror(self):
        """A response with an unexpected stop_reason and no text block (only a
        thinking block, say) must raise a descriptive RuntimeError naming the
        stop_reason — not a bare StopIteration bubbling out of next()."""
        blocks = [types.SimpleNamespace(type="thinking")]
        resp = _resp("end_turn", blocks)
        with pytest.raises(RuntimeError, match="no text block"):
            T._extract_json(resp)


# =========================================================================== #
class TestAtomicWriteJson:
    def test_lands_pretty_json_with_trailing_newline_and_no_tmp_left(self, tmp_path):
        """The tmp+rename write produces the same bytes as the direct write it
        replaced (indent=2, trailing newline) and leaves no `.tmp` sibling once
        os.replace has run — the artifact set is never observed half-written."""
        path = tmp_path / "taxonomy_v1.json"
        T._atomic_write_json(path, {"level1": [], "level2": [{"name": "T"}]})
        text = path.read_text()
        assert text.endswith("\n")
        assert text == json.dumps({"level1": [], "level2": [{"name": "T"}]}, indent=2) + "\n"
        assert list(tmp_path.iterdir()) == [path]  # no leftover .tmp

    def test_overwrites_an_existing_file(self, tmp_path):
        path = tmp_path / "crosswalk_v1.json"
        path.write_text("stale contents")
        T._atomic_write_json(path, {"mappings": []})
        assert json.loads(path.read_text()) == {"mappings": []}


# =========================================================================== #
# Read-only sanity checks against the COMMITTED artifacts. Fast (two small JSON
# reads), no API. Read by absolute worktree path — under a git worktree the
# module's ANNOTATIONS_DIR can resolve to the main repo, and conftest's autouse
# redirect points it at tmp_path; the committed files live beside this test tree.
_ARTIFACTS = Path(__file__).resolve().parents[1] / "data" / "llm_annotations"


class TestCommittedArtifacts:
    def test_taxonomy_v1_is_structurally_sound(self):
        """The shipped taxonomy passes every structural gate, including the newly
        enforced REQUIRED_NON_POLICY check — this anchors the semantic non-policy
        matcher against the real merge wording, not just synthetic fixtures."""
        tax = json.loads((_ARTIFACTS / "taxonomy_v1.json").read_text())
        r = T.validate_structure(tax)
        assert r["orphan_parents"] == []
        assert r["level2_missing_fields"] == []
        assert r["in_target_range"] is True
        assert r["over_hard_max"] is False
        assert r["missing_non_policy"] == []
        assert r["non_policy_complete"] is True
        assert r["duplicate_level2_names"] == []

    def test_committed_run_clears_the_bail_gate(self):
        """The shipped run would pass the enforced bail gate: its recorded 100%
        coverage plus the recomputed structure and crosswalk checks yield zero bail
        reasons — the frozen artifacts are exactly what run() would have written."""
        tax = json.loads((_ARTIFACTS / "taxonomy_v1.json").read_text())
        xw = json.loads((_ARTIFACTS / "crosswalk_v1.json").read_text())
        coverage = tax["provenance"]["coverage"]
        assert T.bail_reasons(coverage, T.validate_structure(tax),
                              T.check_crosswalk(xw, tax)) == []

    def test_crosswalk_v1_reconstructs_every_issue_page(self):
        """Every legacy issue page stays reconstructable through the shipped
        crosswalk: no missing issues, no dangling topic references."""
        tax = json.loads((_ARTIFACTS / "taxonomy_v1.json").read_text())
        xw = json.loads((_ARTIFACTS / "crosswalk_v1.json").read_text())
        r = T.check_crosswalk(xw, tax)
        assert r["missing_issues"] == []
        assert r["dangling_topics"] == []
        assert r["complete"] is True
        assert r["n_pages_reconstructable"] == r["n_pages_total"] == len(T.CROSSWALK_ISSUES)
