"""Tests for `triangulate.py` — LLM vs CorEx vs embedding-cluster comparison.

The three highest-value guards here, in order:

1. **The algebraic ceiling on the headline** (`TestAgreementDriversCeiling`).
   The report's central caveat is that Jaccard is bounded above by the very
   quantity it is correlated against: `J <= min(a,b)/max(a,b) == exp(-D)` where
   `D = |log(n_llm/n_corex)|`. That is an identity, so it can be asserted with
   exact tolerances on synthetic data — and it will catch any future metric
   change that quietly breaks the disclosure the report rests on. The Jaccards
   fed to it are produced by the module's own `_pair_metrics` over real boolean
   arrays, so the bound is a derived property, not a hand-set number.

   The identity is only half of it. §0's actual CONCLUSION — "the association
   does not survive the control" — is pinned separately by
   `TestControlledAssociationConclusion`, with `TestControlDetectsARealEffect`
   as the paired contrast that stops it degenerating into "the controlled
   statistic is always dead".

2. **Join integrity** (`TestLoadArmsJoinIntegrity`). `validate="one_to_one"`
   catches DUPLICATE keys but NOT diverging key SETS — an inner join silently
   drops the non-matching rows and agreement gets computed on an unannounced
   subset of the corpus. `_require_full_merge` is the guard; these tests break
   the key sets in both directions to prove it fires.

3. **Name normalization** (`TestNormalizeTopicLists`). `paragraph_annotations`
   holds 58 distinct topic strings for 50 canonical level-2 names; the 8 extras
   are title-case variants concentrated in the war/foreign-policy topics. An
   exact-match join would drop them and bias agreement DOWNWARD exactly where
   the headline prediction is tested.

Conventions: the corpus is tiny and synthetic (400 paragraphs / 8 speeches), but
the frozen `taxonomy_v1.json` and `crosswalk_v1.json` are read for real — they
are cheap JSON reads of provenance-stamped, frozen artifacts, and using the real
50 topic names / 16 crosswalk keys is what makes the projection and composition
tests meaningful. No parquet under `data/` is ever read: `load_arms`'s four path
constants are monkeypatched at tmp_path files.

Coverage map (every public/testable surface has at least one test):
  * `normalize_topic_lists`  .. TestNormalizeTopicLists
  * `_assert_canonical_topics` .. TestAssertCanonicalTopics
  * `load_crosswalk`         .. TestLoadCrosswalk
  * `project_to_legacy`      .. TestProjectToLegacy
  * `load_arms`              .. TestLoadArms, TestLoadArmsJoinIntegrity
  * `_pair_metrics`          .. TestPairMetrics
  * `corex_column`           .. TestCorexColumn
  * `agreement_table`        .. TestAgreementTable
  * `rename_vs_death`        .. TestRenameVsDeath
  * `rename_candidates`      .. TestRenameCandidates
  * `build_compositions`     .. TestBuildCompositions
  * `agreement_drivers`      .. TestAgreementDrivers, TestAgreementDriversCeiling,
                                TestControlledAssociationConclusion,
                                TestControlDetectsARealEffect
  * `run`                    .. TestRun (hermetic end-to-end wiring smoke test)
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import trends
from presidential_profiles import triangulate as T
from presidential_profiles.issues import ISSUE_ANCHORS
from presidential_profiles.taxonomy import (
    CROSSWALK_ISSUES,
    SECURITY_PEACE,
    TAXONOMY_PATH,
    _norm_name,
)

LEGACY = list(ISSUE_ANCHORS)
DISCOVERED = [f"Discovered {i}" for i in range(1, 8)]

# The 8 title-case variants observed in `paragraph_annotations.topics` (111 rows
# across 36,229). Every one must collapse onto a canonical level-2 name.
TITLE_CASE_VARIANTS = [
    "Iraq, Gulf Wars, the War On Terror & Interventions",
    "Early Naval Wars: Barbary & the War Of 1812",
    "Presidential Humility & Reflection On Office",
    "Prosperity, Jobs & The Middle Class",
    "Civil Service Reform & The Merit System",
    "Conservation & The Environment",
    "Early Naval Wars: Barbary & The War of 1812",
    "Executive Power, Vetoes & The Courts",
]


@pytest.fixture(scope="module")
def level2_names() -> list[str]:
    """The 50 canonical level-2 topic names from frozen `taxonomy_v1.json`."""
    return [t["name"] for t in json.loads(TAXONOMY_PATH.read_text())["level2"]]


# --------------------------------------------------------------------------- #
# Synthetic corpus: 8 speeches x 50 paragraphs, two eras, keyed like the real
# tables. Small enough to be instant, structured enough that every real code
# path (keyed merges, era binning, 50-topic assertion) runs unmodified.
# --------------------------------------------------------------------------- #
def _base_rows() -> pd.DataFrame:
    rows = []
    for d in range(8):
        for p in range(50):
            rows.append({
                "doc_name": f"doc{d:02d}",
                "para_idx": p,
                "president": f"P{d // 4}",
                # 1800 falls in trends.ERAS' "The founding" (1789-1815); 1990 in
                # "Post-Cold War" (1989-2016). Two eras of 200 paragraphs each,
                # both above agreement_table's n >= 100 bar. Chosen so the pair
                # is also NON-ADJACENT in ERA_ORDER and reversed under
                # lexicographic order ("Post-Cold War" < "The founding"), which
                # is what makes the ordering tests able to fail.
                "year": 1800 if d < 4 else 1990,
            })
    return pd.DataFrame(rows)


def _write_arms(
    tmp_path,
    monkeypatch,
    level2_names,
    *,
    drop_annotations: int = 0,
    ghost_annotations: int = 0,
    duplicate_key: bool = False,
    topics_override=None,
    collapse_corex_issue: str | None = None,
) -> pd.DataFrame:
    """Write the four keyed parquets `load_arms` reads and repoint its constants.

    The path constants are module-level with no injection seam (see the
    Discovered Issues note in the WRITE-TESTS report), so monkeypatching them is
    the only way to exercise the join-integrity logic.
    """
    rng = np.random.default_rng(7)
    base = _base_rows()
    n = len(base)

    paras = base[["doc_name", "para_idx"]].copy()
    paras["text"] = "synthetic paragraph text"
    paras["word_count"] = 3

    labels = base.copy()
    for col in LEGACY + DISCOVERED:
        labels[col] = rng.random(n) < 0.15
    if collapse_corex_issue:
        # Plant the vocabulary-drift signature. Random labels at p=0.15 across
        # two statistically indistinguishable eras never produce one, so a test
        # asserting on `rename_candidates` output would otherwise run on an empty
        # frame and pass vacuously. CorEx fires across the early era and never in
        # the late one; the LLM arm needs no help, sitting above 0.89 of its own
        # peak in both eras for every issue.
        labels[collapse_corex_issue] = labels["year"] == 1800

    clusters = base[["doc_name", "para_idx"]].copy()
    clusters["cluster_k40"] = rng.integers(0, 40, n)
    clusters["cluster_k15"] = rng.integers(0, 15, n)

    ann = base[["doc_name", "para_idx"]].copy()
    if topics_override is not None:
        ann["topics"] = topics_override
    else:
        # Deterministic stride assignment: every one of the 50 canonical names
        # appears, and ~1/3 of paragraphs get an empty list (the real corpus's
        # 1.12% unlabeled case, exaggerated so it is testable).
        ann["topics"] = [
            [level2_names[(i + j * 7) % 50] for j in range(i % 3)] for i in range(n)
        ]
    if drop_annotations:
        ann = ann.iloc[:-drop_annotations]
    if ghost_annotations:
        ghosts = ann.iloc[:ghost_annotations].copy()
        ghosts["doc_name"] = "ghost-doc"
        ann = pd.concat([ann, ghosts], ignore_index=True)
    if duplicate_key:
        ann = pd.concat([ann, ann.iloc[[0]]], ignore_index=True)

    for attr, frame in [
        ("PARAGRAPHS_PATH", paras),
        ("PARA_LABELS_PATH", labels),
        ("CLUSTERS_PATH", clusters),
        ("PARAGRAPH_ANNOTATIONS_PATH", ann),
    ]:
        path = tmp_path / f"{attr.lower()}.parquet"
        frame.to_parquet(path, index=False)
        monkeypatch.setattr(T, attr, path)
    return base


@pytest.fixture
def arms(tmp_path, monkeypatch, level2_names) -> pd.DataFrame:
    _write_arms(tmp_path, monkeypatch, level2_names)
    return T.load_arms()


@pytest.fixture(scope="module")
def crosswalk() -> dict[str, set[str]]:
    return T.load_crosswalk()


# The issue `collapse_corex_issue` plants the rename signature on.
DRIFTING_ISSUE = "Education"


@pytest.fixture
def drifting_arms(tmp_path, monkeypatch, level2_names):
    """`(arms, llm_legacy)` carrying exactly one genuine rename candidate."""
    _write_arms(tmp_path, monkeypatch, level2_names,
                collapse_corex_issue=DRIFTING_ISSUE)
    df = T.load_arms()
    return df, T.project_to_legacy(df["topic_set"], T.load_crosswalk())


@pytest.fixture
def llm_legacy(arms, crosswalk) -> pd.DataFrame:
    return T.project_to_legacy(arms["topic_set"], crosswalk)


# =========================================================================== #
# normalize_topic_lists — the 8 title-case variants
# =========================================================================== #
class TestNormalizeTopicLists:
    def test_produces_frozensets(self):
        out = T.normalize_topic_lists(pd.Series([["Alpha", "Beta"], []]))

        assert out.iloc[0] == frozenset({"alpha", "beta"})
        assert out.iloc[1] == frozenset()

    def test_case_and_whitespace_are_collapsed(self):
        out = T.normalize_topic_lists(pd.Series([["  Trade  &   Tariffs "]]))

        assert out.iloc[0] == frozenset({"trade & tariffs"})

    def test_duplicates_within_one_paragraph_collapse(self):
        """A frozenset, not a list: a paragraph labelled with a name and its
        title-case variant counts once, not twice."""
        out = T.normalize_topic_lists(
            pd.Series([["Conservation & The Environment",
                        "Conservation & the Environment"]])
        )

        assert out.iloc[0] == frozenset({"conservation & the environment"})

    @pytest.mark.parametrize("variant", TITLE_CASE_VARIANTS)
    def test_known_variant_maps_onto_a_canonical_name(self, variant, level2_names):
        """Each of the 8 observed variants must land on a real taxonomy_v1 name.
        These are concentrated in war/foreign-policy topics, so a silent drop
        would bias agreement downward exactly where the prediction is tested."""
        canonical = {_norm_name(n) for n in level2_names}

        normalized = T.normalize_topic_lists(pd.Series([[variant]])).iloc[0]

        assert set(normalized) <= canonical

    def test_variants_collapse_onto_their_canonicals_without_adding_names(
        self, level2_names
    ):
        """The actual bug being guarded: mixing all 8 variants into a corpus that
        already carries every canonical name must not increase the distinct
        count. An exact-match join would report 58."""
        canonical_only = pd.Series([[n] for n in level2_names])
        with_variants = pd.Series(
            [[n] for n in level2_names] + [[v] for v in TITLE_CASE_VARIANTS]
        )

        a = {t for s in T.normalize_topic_lists(canonical_only) for t in s}
        b = {t for s in T.normalize_topic_lists(with_variants) for t in s}

        assert a == b
        assert len(b) == T.N_LEVEL2_TOPICS

    def test_off_taxonomy_name_survives_normalization_and_is_therefore_caught(
        self, level2_names
    ):
        """Normalization is case-folding, not fuzzy matching: a genuinely new
        label stays distinct so the downstream count assertion can catch it."""
        canonical = {_norm_name(n) for n in level2_names}

        out = T.normalize_topic_lists(pd.Series([["Cryptocurrency Regulation"]]))

        assert set(out.iloc[0]) - canonical == {"cryptocurrency regulation"}

    def test_index_is_preserved(self):
        """`load_arms` assigns the result back onto a frame and `agreement_table`
        does `.loc[slc.index]`, so alignment must survive."""
        s = pd.Series([["A"], ["B"]], index=[17, 42])

        assert list(T.normalize_topic_lists(s).index) == [17, 42]


class TestAssertCanonicalTopics:
    def test_exact_count_passes(self):
        T._assert_canonical_topics(
            pd.Series([frozenset({"a", "b"}), frozenset({"c"})]), expected=3
        )

    def test_an_extra_label_raises(self):
        with pytest.raises(ValueError, match="collapsed to 3 distinct"):
            T._assert_canonical_topics(
                pd.Series([frozenset({"a", "b", "leaked"})]), expected=2
            )

    def test_a_missing_label_raises(self):
        """Under-count is the dangerous direction — it means `_norm_name` merged
        two genuinely different topics."""
        with pytest.raises(ValueError, match="collapsed to 1 distinct"):
            T._assert_canonical_topics(pd.Series([frozenset({"a"})]), expected=2)

    def test_error_explains_the_silent_drop_risk(self):
        with pytest.raises(ValueError, match="silently drop rows"):
            T._assert_canonical_topics(pd.Series([frozenset({"a"})]), expected=2)

    def test_real_corpus_default_is_fifty(self, level2_names):
        """The default `expected` is `N_LEVEL2_TOPICS`; assert the constant and
        the frozen taxonomy still agree."""
        assert T.N_LEVEL2_TOPICS == 50

        T._assert_canonical_topics(
            T.normalize_topic_lists(
                pd.Series([[n] for n in level2_names] +
                          [[v] for v in TITLE_CASE_VARIANTS])
            )
        )


# =========================================================================== #
# load_crosswalk
# =========================================================================== #
class TestLoadCrosswalk:
    def test_reads_and_normalizes_a_synthetic_file(self, tmp_path):
        path = tmp_path / "crosswalk.json"
        path.write_text(json.dumps({"mappings": [
            {"legacy_issue": "Economy & jobs",
             "level2_topics": ["Prosperity, Jobs & The Middle Class"]},
            {"legacy_issue": "Immigration", "level2_topics": []},
        ]}))

        out = T.load_crosswalk(path)

        assert out == {
            "Economy & jobs": {"prosperity, jobs & the middle class"},
            "Immigration": set(),
        }

    def test_real_crosswalk_has_the_sixteen_expected_keys(self, crosswalk):
        """15 legacy issues plus `Security & peace` — Discovered 5's display name
        is already a first-class crosswalk key."""
        assert list(crosswalk) == CROSSWALK_ISSUES
        assert SECURITY_PEACE in crosswalk

    def test_real_crosswalk_values_are_normalized(self, crosswalk):
        targets = [t for row in crosswalk.values() for t in row]

        # Anchor the quantifier first: `all()` over an empty crosswalk is True.
        assert min(len(row) for row in crosswalk.values()) >= 1
        assert all(t == _norm_name(t) for t in targets)

    def test_real_crosswalk_is_not_a_partition(self, crosswalk):
        """Level-2 topics are reused across legacy issues, which is why the
        projection must be an explicit union rather than an argmax."""
        counts: dict[str, int] = {}
        for row in crosswalk.values():
            for t in row:
                counts[t] = counts.get(t, 0) + 1

        assert [t for t, c in counts.items() if c > 1]

    def test_real_crosswalk_targets_are_canonical_taxonomy_names(
        self, crosswalk, level2_names
    ):
        canonical = {_norm_name(n) for n in level2_names}

        targets = set().union(*crosswalk.values())

        # `<=` is vacuous on an empty left side, and the crosswalk covers most of
        # the taxonomy (43 of 50; the other 7 are non-policy topics with no
        # legacy counterpart), so require substantial coverage before the subset
        # claim means anything.
        assert len(targets) > len(canonical) / 2
        assert targets <= canonical


# =========================================================================== #
# project_to_legacy — union / any-activation
# =========================================================================== #
class TestProjectToLegacy:
    CW = {
        "Economy & jobs": {"jobs", "trusts"},
        "Trade & tariffs": {"tariffs", "trusts"},
        "Immigration": {"border"},
    }

    def test_any_matching_topic_activates_the_issue(self):
        sets = pd.Series([frozenset({"jobs"})])

        out = T.project_to_legacy(sets, self.CW)

        assert out.iloc[0].to_dict() == {
            "Economy & jobs": True, "Trade & tariffs": False, "Immigration": False,
        }

    def test_a_shared_topic_activates_several_issues_at_once(self):
        """The crosswalk is not a partition, so `trusts` legitimately fires two
        legacy issues. This is the documented union projection, and choosing
        argmax instead would move every agreement number."""
        sets = pd.Series([frozenset({"trusts"})])

        out = T.project_to_legacy(sets, self.CW)

        assert out.iloc[0]["Economy & jobs"] and out.iloc[0]["Trade & tariffs"]

    def test_empty_topic_set_projects_to_no_issue(self):
        out = T.project_to_legacy(pd.Series([frozenset()]), self.CW)

        assert not out.iloc[0].any()

    def test_topic_outside_every_crosswalk_row_projects_to_no_issue(self):
        """7 of the 50 level-2 topics (presidential humility, partisan combat,
        ...) are non-policy with no legacy counterpart. Projecting to zero
        issues is the correct outcome, not a lookup failure."""
        out = T.project_to_legacy(pd.Series([frozenset({"presidential humility"})]),
                                  self.CW)

        assert not out.iloc[0].any()

    def test_columns_follow_the_crosswalk_order(self):
        out = T.project_to_legacy(pd.Series([frozenset()]), self.CW)

        assert list(out.columns) == list(self.CW)

    def test_index_is_preserved(self):
        """`agreement_table` and `rename_vs_death` both index this frame by the
        arms frame's index, so a reset would silently misalign the two labelers."""
        sets = pd.Series([frozenset({"jobs"}), frozenset()], index=[11, 99])

        out = T.project_to_legacy(sets, self.CW)

        assert list(out.index) == [11, 99]

    def test_real_projection_is_boolean_and_row_complete(self, arms, crosswalk):
        out = T.project_to_legacy(arms["topic_set"], crosswalk)

        assert out.shape == (len(arms), 16)
        assert out.dtypes.eq(bool).all()

    def test_unlabeled_paragraphs_project_to_all_false(self, arms, llm_legacy):
        unlabeled = llm_legacy[arms["llm_unlabeled"].to_numpy()]

        assert len(unlabeled) > 0
        assert not unlabeled.to_numpy().any()


# =========================================================================== #
# load_arms
# =========================================================================== #
class TestLoadArms:
    def test_carries_all_three_labelers_on_one_keyed_frame(self, arms):
        assert len(arms) == 400
        assert {"doc_name", "para_idx", "topic_set", "llm_unlabeled", "era",
                "cluster_k15", "cluster_k40"} <= set(arms.columns)
        assert set(LEGACY + DISCOVERED) <= set(arms.columns)

    def test_key_is_unique(self, arms):
        assert not arms.duplicated(["doc_name", "para_idx"]).any()

    def test_era_uses_the_trends_reporting_axis(self, arms):
        """The reporting axis is `trends.ERAS`, not `taxonomy.ERA_SPAN`.

        ERA_SPAN's equal-width bands are a SAMPLING device (the anachronism
        guard for per-era LLM taxonomy proposals); this table is a published
        axis that `era-atlas` and `convergence-analysis` join on. No second
        periodization may be introduced here, so the labels are asserted to be
        `trends.ERAS`'s own, not merely "some 9 strings".
        """
        assert T.ERA_ORDER == [label for label, _, _ in trends.ERAS]
        # ERA_BOUNDS is the year range downstream joiners (era-atlas,
        # convergence-analysis) need to align their own cuts against this table.
        assert T.ERA_BOUNDS == {lab: (lo, hi) for lab, lo, hi in trends.ERAS}
        assert not hasattr(T, "ERA_SPAN")  # the sampling device must not leak back
        assert set(arms["era"]) == {"The founding", "Post-Cold War"}
        assert (arms.loc[arms["year"] == 1800, "era"] == "The founding").all()
        assert (arms.loc[arms["year"] == 1990, "era"] == "Post-Cold War").all()

    def test_every_year_maps_to_exactly_one_era(self, arms):
        """`trends.ERAS` must partition the corpus years: no gaps, no overlaps.

        A gap is the dangerous direction. `era` is a string on this axis and NaN
        already MEANS "whole-corpus row" in `method_agreement.parquet`, so an
        unmapped year would be labelled NaN and read as an overall row.
        """
        assert arms["era"].notna().all()
        bands = [
            (label, arms["year"].between(lo, hi)) for label, lo, hi in trends.ERAS
        ]
        n_bands_claiming = sum(hit.astype(int) for _, hit in bands)
        assert (n_bands_claiming == 1).all()
        for label, hit in bands:
            assert (arms.loc[hit, "era"] == label).all()

    def test_a_year_outside_every_band_raises_instead_of_becoming_nan(self):
        """THE TRAP. A year in no band must not silently become NaN — that is
        indistinguishable from the whole-corpus marker. 1500 predates the corpus
        and 2200 postdates it; both must raise, not fall through."""
        for stray in (1500, 2200):
            with pytest.raises(ValueError, match="outside every trends.ERAS band"):
                T.assign_eras(pd.Series([1800, stray, 1990]))

    def test_overlapping_bands_raise(self, monkeypatch):
        """The other half of the partition. If two bands ever claim one year the
        label would depend on declaration order — a silent, invisible choice."""
        monkeypatch.setattr(
            T, "ERAS", [("A", 1789, 1900), ("B", 1850, 2026)], raising=True
        )
        with pytest.raises(ValueError, match="bands overlap"):
            T.assign_eras(pd.Series([1860]))

    def test_eras_in_order_is_chronological_not_lexicographic(self):
        """Named eras do not sort correctly as strings. Lexicographic order puts
        "Civil War & Reconstruction" before "Expansion" before "The founding" —
        a chronological axis rendered alphabetically. The old numeric grid sorted
        right by accident; this one only sorts right on purpose."""
        scrambled = ["The founding", "Civil War & Reconstruction", "Expansion"]

        assert T.eras_in_order(scrambled) == [
            "The founding", "Expansion", "Civil War & Reconstruction",
        ]
        assert T.eras_in_order(scrambled) != sorted(scrambled)

    def test_eras_in_order_rejects_labels_off_the_reporting_axis(self):
        """A second periodization leaking in would make cross-task joins fail
        silently, so an unknown label is an error rather than a passthrough."""
        with pytest.raises(ValueError, match="absent from trends.ERAS"):
            T.eras_in_order(["The founding", 1980])

    def test_unlabeled_flag_marks_empty_topic_lists_only(self, arms):
        assert (arms["llm_unlabeled"] == (arms["topic_set"].map(len) == 0)).all()
        assert 0 < arms["llm_unlabeled"].sum() < len(arms)

    def test_topic_sets_are_normalized(self, arms):
        names = {t for s in arms["topic_set"] for t in s}

        # The fixture assigns every canonical name, so an `all()` that quantified
        # over nothing (or over a handful) would be caught here first.
        assert len(names) == T.N_LEVEL2_TOPICS
        assert all(t == _norm_name(t) for t in names)


class TestLoadArmsJoinIntegrity:
    """`validate="one_to_one"` catches duplicate keys but NOT diverging key SETS.
    An inner join on diverging sets silently drops rows, which here would mean
    computing agreement on an unannounced subset of the corpus. `_require_full_merge`
    is the second guard; these tests break the key sets in both directions.
    """

    def test_annotations_missing_rows_raises_rather_than_shrinking(
        self, tmp_path, monkeypatch, level2_names
    ):
        _write_arms(tmp_path, monkeypatch, level2_names, drop_annotations=5)

        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            T.load_arms()

    def test_annotations_with_extra_keys_also_raises(
        self, tmp_path, monkeypatch, level2_names
    ):
        """The other direction: the merged length still equals the paragraph
        count, so a length check against the running frame alone would pass.
        `_require_full_merge` asserts against EVERY input's length."""
        _write_arms(tmp_path, monkeypatch, level2_names, ghost_annotations=5)

        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            T.load_arms()

    def test_error_names_the_stage_and_the_counts(
        self, tmp_path, monkeypatch, level2_names
    ):
        _write_arms(tmp_path, monkeypatch, level2_names, drop_annotations=5)

        with pytest.raises(ValueError) as exc:
            T.load_arms()

        assert "annotations=395" in str(exc.value)
        assert "paragraph_annotations" in str(exc.value)

    def test_duplicate_key_is_caught_by_validate_one_to_one(
        self, tmp_path, monkeypatch, level2_names
    ):
        """The failure `validate=` DOES catch, kept alongside its blind spot so
        the two guards are visibly complementary."""
        _write_arms(tmp_path, monkeypatch, level2_names, duplicate_key=True)

        with pytest.raises(pd.errors.MergeError):
            T.load_arms()

    def test_a_leaked_label_trips_the_canonical_topic_assertion(
        self, tmp_path, monkeypatch, level2_names
    ):
        """A 51st distinct name means either a new label leaked into the
        annotations or `_norm_name` stopped collapsing the variants."""
        topics = [[level2_names[(i + j * 7) % 50] for j in range(i % 3)]
                  for i in range(400)]
        topics[0] = ["Cryptocurrency Regulation"]
        _write_arms(tmp_path, monkeypatch, level2_names, topics_override=topics)

        with pytest.raises(ValueError, match="expected 50"):
            T.load_arms()

    def test_title_case_variants_do_NOT_trip_the_assertion(
        self, tmp_path, monkeypatch, level2_names
    ):
        """The complement of the test above: the 8 known variants must pass,
        because normalization collapses them before the count is taken."""
        topics = [[level2_names[(i + j * 7) % 50] for j in range(i % 3)]
                  for i in range(400)]
        for i, variant in enumerate(TITLE_CASE_VARIANTS):
            topics[i] = [variant]
        _write_arms(tmp_path, monkeypatch, level2_names, topics_override=topics)

        assert len(T.load_arms()) == 400


# =========================================================================== #
# _pair_metrics — hand-computed on 2x2 cases
# =========================================================================== #
def _b(*bits) -> np.ndarray:
    return np.array(bits, dtype=bool)


class TestPairMetrics:
    def test_half_overlap_matches_hand_computation(self):
        """llm={0,1}, corex={0,2} over 4 paragraphs.
        Jaccard = |{0}| / |{0,1,2}| = 1/3.
        kappa: p_o = (1 agreed-positive + 1 agreed-negative)/4 = 0.5;
               p_e = 0.5*0.5 + 0.5*0.5 = 0.5;  kappa = (0.5-0.5)/(1-0.5) = 0."""
        m = T._pair_metrics(_b(1, 1, 0, 0), _b(1, 0, 1, 0))

        assert m == {"n": 4, "n_llm": 2, "n_corex": 2, "n_both": 1,
                     "jaccard": pytest.approx(1 / 3), "kappa": pytest.approx(0.0)}

    def test_identical_sets_score_one_on_both_metrics(self):
        m = T._pair_metrics(_b(1, 1, 0, 0), _b(1, 1, 0, 0))

        assert m["jaccard"] == pytest.approx(1.0)
        assert m["kappa"] == pytest.approx(1.0)

    def test_disjoint_sets_score_zero_jaccard_and_minus_one_kappa(self):
        """p_o = 0, p_e = 0.5, so kappa = (0 - 0.5)/(1 - 0.5) = -1: perfect
        disagreement, not merely chance."""
        m = T._pair_metrics(_b(1, 1, 0, 0), _b(0, 0, 1, 1))

        assert m["jaccard"] == 0.0
        assert m["kappa"] == pytest.approx(-1.0)

    def test_nested_sets(self):
        """llm={0,1,2}, corex={0}: Jaccard = 1/3, and the ceiling min/max = 1/3
        is exactly attained."""
        m = T._pair_metrics(_b(1, 1, 1, 0), _b(1, 0, 0, 0))

        assert m["jaccard"] == pytest.approx(1 / 3)
        assert m["jaccard"] == pytest.approx(m["n_corex"] / m["n_llm"])

    def test_both_empty_gives_nan_not_a_perfect_score(self):
        """Jaccard of two empty sets is conventionally 1. Reporting that here
        would say two labelers perfectly agree about an issue neither ever
        assigned — so this must be NaN and drop out of the aggregates."""
        m = T._pair_metrics(_b(0, 0, 0, 0), _b(0, 0, 0, 0))

        assert m["n_both"] == 0
        assert math.isnan(m["jaccard"])
        assert math.isnan(m["kappa"])

    def test_one_labeler_silent_gives_nan_kappa_not_zero(self):
        """Kappa's expected-agreement term collapses when a labeler is constant.
        A fake 0 would read as "chance-level agreement" — a measurement that was
        never made."""
        m = T._pair_metrics(_b(1, 1, 0, 0), _b(0, 0, 0, 0))

        assert m["jaccard"] == 0.0
        assert math.isnan(m["kappa"])

    def test_one_labeler_always_positive_gives_nan_kappa(self):
        m = T._pair_metrics(_b(1, 1, 1, 1), _b(1, 0, 0, 0))

        assert math.isnan(m["kappa"])
        assert m["jaccard"] == pytest.approx(0.25)

    def test_empty_slice_gives_nan_on_both_metrics(self):
        m = T._pair_metrics(_b(), _b())

        assert m["n"] == 0
        assert math.isnan(m["jaccard"]) and math.isnan(m["kappa"])

    def test_counts_are_plain_ints_for_json_and_parquet(self):
        m = T._pair_metrics(_b(1, 0), _b(1, 1))

        assert all(isinstance(m[k], int)
                   for k in ("n", "n_llm", "n_corex", "n_both"))

    @pytest.mark.parametrize("seed", range(12))
    def test_jaccard_never_exceeds_the_min_over_max_ceiling(self, seed):
        """The algebraic identity the report's central caveat rests on, asserted
        on the metric function itself: |A n B| <= min(|A|,|B|) and
        |A u B| >= max(|A|,|B|), so J <= min/max always."""
        rng = np.random.default_rng(seed)
        llm = rng.random(200) < rng.uniform(0.05, 0.6)
        corex = rng.random(200) < rng.uniform(0.05, 0.6)

        m = T._pair_metrics(llm, corex)

        ceiling = min(m["n_llm"], m["n_corex"]) / max(m["n_llm"], m["n_corex"])
        assert m["jaccard"] <= ceiling + 1e-12


# =========================================================================== #
# corex_column
# =========================================================================== #
class TestCorexColumn:
    def test_security_and_peace_maps_to_the_discovered_column(self):
        """The crosswalk's 16th key is a display name; the parquet column it is
        backed by is still `Discovered 5` (columns are deliberately never
        renamed — every downstream consumer keys on them)."""
        assert T.corex_column(SECURITY_PEACE) == "Discovered 5"
        assert T.SECURITY_PEACE_COLUMN == "Discovered 5"

    @pytest.mark.parametrize("issue", LEGACY)
    def test_legacy_issues_pass_through_unchanged(self, issue):
        assert T.corex_column(issue) == issue

    def test_every_crosswalk_key_resolves_to_a_real_parquet_column(self, arms):
        assert all(T.corex_column(i) in arms.columns for i in CROSSWALK_ISSUES)


# =========================================================================== #
# agreement_table
# =========================================================================== #
class TestAgreementTable:
    def test_one_overall_row_per_issue(self, arms, llm_legacy):
        table = T.agreement_table(arms, llm_legacy)
        overall = table[table["era"].isna()]

        assert list(overall["issue"]) == list(llm_legacy.columns)
        assert len(overall) == 16

    def test_era_rows_use_the_trends_reporting_axis(self, arms, llm_legacy):
        table = T.agreement_table(arms, llm_legacy)

        assert set(table["era"].dropna()) == {"The founding", "Post-Cold War"}
        assert set(table["era"].dropna()) <= set(T.ERA_ORDER)

    def test_era_rows_are_emitted_in_chronological_order(self, arms, llm_legacy):
        """`era` is a string now, so nothing sorts it for free. Under
        lexicographic order "Post-Cold War" precedes "The founding" — the exact
        inversion this asserts against, which is why these two eras were chosen
        for the synthetic corpus."""
        table = T.agreement_table(arms, llm_legacy)
        per_issue = table[table["era"].notna()].groupby("issue")["era"]

        for _, seq in per_issue:
            assert list(seq) == ["The founding", "Post-Cold War"]
        assert list(per_issue.first().unique()) != ["Post-Cold War"]

    def test_only_the_overall_rows_carry_a_nan_era(self, arms, llm_legacy):
        """THE TRAP, at the table level: NaN is this table's whole-corpus marker,
        so exactly one NaN row per issue is allowed. A year-derived NaN would be
        read as a corpus-wide statistic."""
        table = T.agreement_table(arms, llm_legacy)

        assert int(table["era"].isna().sum()) == len(llm_legacy.columns)
        assert table.loc[table["era"].isna(), "issue"].is_unique
        # The complement: every non-overall row carries a real era label.
        assert (table["era"].notna().sum()
                == len(table) - len(llm_legacy.columns))

    def test_a_nan_era_on_the_input_is_rejected(self, arms, llm_legacy):
        """THE TRAP in its live form. A NaN era is dropped when the era list is
        built, so those paragraphs stay in the whole-corpus row but disappear
        from every era row — a partial table reported as complete. Only the
        overall marker may be NaN, and it is never sourced from a year."""
        holed = arms.copy()
        holed.loc[holed.index[:40], "era"] = None

        with pytest.raises(ValueError, match="paragraphs carry era = NaN"):
            T.agreement_table(holed, llm_legacy)

    def test_rectangularity_guard_rejects_a_stray_nan_era(self):
        """THE TRAP as the guard sees it: an extra NaN row means NaN no longer
        identifies exactly the whole-corpus rows, so an era statistic would be
        read as a corpus-wide one."""
        out = pd.DataFrame({"era": [None, None, "The founding", "The founding"]})

        T._require_rectangular(out, n_issues=2, n_eras=1)  # the well-formed shape
        with pytest.raises(ValueError, match="not rectangular"):
            T._require_rectangular(
                pd.DataFrame({"era": [None, None, None, "The founding"]}),
                n_issues=2, n_eras=1,
            )

    def test_rectangularity_guard_rejects_a_dropped_issue_era_cell(self):
        """The other direction: a cell silently missing for one issue but not
        another would publish a ragged table that still looks complete."""
        with pytest.raises(ValueError, match="not rectangular"):
            T._require_rectangular(
                pd.DataFrame({"era": [None, None, "The founding"]}),
                n_issues=2, n_eras=1,
            )

    def test_era_row_count_matches_the_non_overall_rows(self, arms, llm_legacy):
        """Era-labelled row count == issues x kept eras, with no row unaccounted
        for. Pairs with the NaN guard: one says nothing is mislabelled, this says
        nothing is missing."""
        table = T.agreement_table(arms, llm_legacy)
        kept = T.eras_in_order(arms.loc[~arms["llm_unlabeled"], "era"])

        assert len(table) == len(llm_legacy.columns) * (1 + len(kept))
        assert int(table["era"].notna().sum()) == len(llm_legacy.columns) * len(kept)

    def test_a_stray_era_label_is_rejected_rather_than_published(self, arms,
                                                                llm_legacy):
        """A second periodization must not reach the published parquet."""
        stray = arms.copy()
        stray.loc[stray.index[:150], "era"] = "Reconstruction-ish"

        with pytest.raises(ValueError, match="absent from trends.ERAS"):
            T.agreement_table(stray, llm_legacy)

    def test_drop_unlabeled_excludes_the_abstentions(self, arms, llm_legacy):
        """An LLM abstention is not "the LLM judged this issue absent"; counting
        it as an all-negative row would inflate kappa's agreed-negative mass for
        all 16 issues at once."""
        n_labeled = int((~arms["llm_unlabeled"]).sum())

        table = T.agreement_table(arms, llm_legacy, drop_unlabeled=True)

        assert set(table.loc[table["era"].isna(), "n"]) == {n_labeled}
        assert n_labeled < len(arms)

    def test_sensitivity_run_keeps_them(self, arms, llm_legacy):
        table = T.agreement_table(arms, llm_legacy, drop_unlabeled=False)

        assert set(table.loc[table["era"].isna(), "n"]) == {len(arms)}

    def test_thin_eras_are_skipped(self, tmp_path, monkeypatch, level2_names):
        """A kappa on a handful of paragraphs is not stable enough to publish, so
        eras below 100 paragraphs are dropped rather than reported noisily."""
        # Retag 20 paragraphs into a third era. 1860 sits in "Civil War &
        # Reconstruction" — an in-band year, because an out-of-band one would
        # now raise in `assign_eras` and test the wrong guard entirely.
        _write_arms(tmp_path, monkeypatch, level2_names)
        labels = pd.read_parquet(T.PARA_LABELS_PATH)
        labels.loc[labels.index[:20], "year"] = 1860
        labels.to_parquet(T.PARA_LABELS_PATH, index=False)
        df = T.load_arms()
        legacy = T.project_to_legacy(df["topic_set"], T.load_crosswalk())

        table = T.agreement_table(df, legacy)

        assert "Civil War & Reconstruction" in set(df["era"])
        assert "Civil War & Reconstruction" not in set(table["era"].dropna())

    def test_security_and_peace_is_scored_against_discovered_five(
        self, arms, llm_legacy
    ):
        table = T.agreement_table(arms, llm_legacy)
        row = table[(table["issue"] == SECURITY_PEACE) & table["era"].isna()].iloc[0]
        labeled = arms[~arms["llm_unlabeled"]]

        assert row["n_corex"] == int(labeled["Discovered 5"].sum())

    def test_metric_columns_are_present(self, arms, llm_legacy):
        table = T.agreement_table(arms, llm_legacy)

        assert set(table.columns) == {"issue", "era", "n", "n_llm", "n_corex",
                                      "n_both", "jaccard", "kappa",
                                      "min_support", "low_support", "empty_arm"}

    def test_llm_and_corex_arms_stay_row_aligned(self, arms, llm_legacy):
        """`agreement_table` slices with `.loc[slc.index]`. Shuffling the arms
        frame must not change any number — if it does, something is aligning
        positionally instead of by key."""
        table = T.agreement_table(arms, llm_legacy)
        shuffled = arms.sample(frac=1, random_state=3)
        assert list(shuffled.index) != list(arms.index)  # the reorder is real

        reshuffled = T.agreement_table(shuffled, llm_legacy.loc[shuffled.index])

        pd.testing.assert_frame_equal(
            table.sort_values(["issue", "era"]).reset_index(drop=True),
            reshuffled.sort_values(["issue", "era"]).reset_index(drop=True),
        )


# =========================================================================== #
# rename_vs_death
# =========================================================================== #
# The two named eras the drift fixtures use. Deliberately reversed under
# lexicographic order ("Post-Cold War" < "The founding") while EARLY precedes
# LATE in trends.ERAS, so any accidental `sorted()` inverts the trajectory.
EARLY, LATE = "The founding", "Post-Cold War"
MIDDLE = "Civil War & Reconstruction"


def _drift_inputs():
    """A hand-built two-era frame with exactly derivable shares.

    EARLY (200 rows): CorEx fires on 20 (0.10), LLM on 80 (0.40)
    LATE  (200 rows): CorEx fires on 80 (0.40), LLM on 20 (0.10)
    -> corex_rel = 0.25 / 1.00 ; llm_rel = 1.00 / 0.25 (each scaled to its OWN
       peak, so the crossing trajectories are what the drift gap measures)
    """
    n = 200
    rows = []
    for era, corex_hits, llm_hits in [(EARLY, 20, 80), (LATE, 80, 20)]:
        for i in range(n):
            rows.append({"era": era, "llm_unlabeled": False,
                         "Economy & jobs": i < corex_hits,
                         "_llm": i < llm_hits})
    df = pd.DataFrame(rows)
    llm = pd.DataFrame({"Economy & jobs": df.pop("_llm")})
    return df, llm


class TestRenameVsDeath:
    def test_shares_and_peak_relative_values_are_hand_derivable(self):
        df, llm = _drift_inputs()

        out = T.rename_vs_death(df, llm).set_index("era")

        assert out.loc[EARLY, "corex_share"] == pytest.approx(0.10)
        assert out.loc[LATE, "corex_share"] == pytest.approx(0.40)
        assert out.loc[EARLY, "llm_share"] == pytest.approx(0.40)
        assert out.loc[LATE, "llm_share"] == pytest.approx(0.10)
        assert out.loc[EARLY, "corex_rel"] == pytest.approx(0.25)
        assert out.loc[LATE, "corex_rel"] == pytest.approx(1.0)
        assert out.loc[EARLY, "llm_rel"] == pytest.approx(1.0)
        assert out.loc[LATE, "llm_rel"] == pytest.approx(0.25)

    def test_drift_gap_is_llm_minus_corex(self):
        df, llm = _drift_inputs()

        out = T.rename_vs_death(df, llm).set_index("era")

        assert out.loc[EARLY, "drift_gap"] == pytest.approx(0.75)
        assert out.loc[LATE, "drift_gap"] == pytest.approx(-0.75)

    def test_each_arm_is_scaled_to_its_own_peak(self):
        """The two methods have different base rates, so only their trajectories
        are comparable. Each arm must reach exactly 1.0 in its own peak era."""
        df, llm = _drift_inputs()

        out = T.rename_vs_death(df, llm)

        assert out["corex_rel"].max() == pytest.approx(1.0)
        assert out["llm_rel"].max() == pytest.approx(1.0)

    def test_unlabeled_paragraphs_are_excluded(self):
        df, llm = _drift_inputs()
        # 100 extra abstentions in 1800 would drag its shares down if counted.
        extra = pd.DataFrame({"era": [EARLY] * 100, "llm_unlabeled": [True] * 100,
                              "Economy & jobs": [False] * 100})
        df2 = pd.concat([df, extra], ignore_index=True)
        llm2 = pd.concat([llm, pd.DataFrame({"Economy & jobs": [False] * 100})],
                         ignore_index=True)

        out = T.rename_vs_death(df2, llm2).set_index("era")

        assert out.loc[EARLY, "corex_share"] == pytest.approx(0.10)

    def test_thin_era_is_dropped_before_the_peak_is_taken(self):
        """The live trap: an era with too few paragraphs must never become the
        denominator every other era is scaled against. Here a 50-row era where
        CorEx fires on every paragraph would otherwise seize the peak and crush
        both real eras' `corex_rel`."""
        df, llm = _drift_inputs()
        thin = pd.DataFrame({"era": [MIDDLE] * 50, "llm_unlabeled": [False] * 50,
                             "Economy & jobs": [True] * 50})
        df2 = pd.concat([df, thin], ignore_index=True)
        llm2 = pd.concat([llm, pd.DataFrame({"Economy & jobs": [True] * 50})],
                         ignore_index=True)

        out = T.rename_vs_death(df2, llm2).set_index("era")

        assert MIDDLE not in out.index
        assert out.loc[LATE, "corex_rel"] == pytest.approx(1.0)

    def test_issue_never_labelled_by_corex_gives_nan_not_a_divide_by_zero(self):
        df, llm = _drift_inputs()
        df["Economy & jobs"] = False

        out = T.rename_vs_death(df, llm)

        assert out["corex_rel"].isna().all()
        assert out["llm_rel"].notna().all()

    def test_covers_every_issue_and_kept_era(self, arms, llm_legacy):
        out = T.rename_vs_death(arms, llm_legacy)

        assert set(out["issue"]) == set(llm_legacy.columns)
        assert len(out) == 16 * 2

    def test_rows_are_emitted_in_chronological_order(self, arms, llm_legacy):
        """This is a TRAJECTORY table — row order is the trajectory. `groupby`
        on a string era sorts lexicographically, which here puts "Post-Cold War"
        before "The founding" and hands any consumer plotting it in file order a
        reversed history. The peak-relative *values* survive a bad order
        (max is order-free), so only the emitted order can catch this."""
        out = T.rename_vs_death(arms, llm_legacy)

        for _, seq in out.groupby("issue", sort=False)["era"]:
            assert list(seq) == [EARLY, LATE]
        assert [EARLY, LATE] != sorted([EARLY, LATE])  # the trap is real

    def test_era_is_a_label_not_a_number(self, arms, llm_legacy):
        """Guards the old `int(era)` cast, which would now raise or coerce."""
        out = T.rename_vs_death(arms, llm_legacy)

        assert out["era"].map(type).eq(str).all()
        assert set(out["era"]) <= set(T.ERA_ORDER)

    def test_security_and_peace_reads_the_discovered_column(self, arms, llm_legacy):
        labeled = arms[~arms["llm_unlabeled"]]
        expected = labeled.groupby("era")["Discovered 5"].mean()

        out = T.rename_vs_death(arms, llm_legacy)
        got = out[out["issue"] == SECURITY_PEACE].set_index("era")["corex_share"]

        assert got.loc[EARLY] == pytest.approx(expected.loc[EARLY])


# =========================================================================== #
# rename_candidates
# =========================================================================== #
def _drift_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["issue", "era", "corex_rel", "llm_rel"])
    df["drift_gap"] = df["llm_rel"] - df["corex_rel"]
    return df


class TestRenameCandidates:
    def test_collapse_with_persistence_is_flagged(self):
        """The vocabulary-drift signature: CorEx's anchor words stopped
        appearing, the underlying concern did not."""
        drift = _drift_frame([("Civil rights & race", LATE, 0.10, 0.90)])

        assert list(T.rename_candidates(drift)["issue"]) == ["Civil rights & race"]

    def test_both_arms_collapsing_is_not_flagged(self):
        """The mirror case — genuine loss of attention, not a rename. This is the
        distinction the whole table exists to make."""
        drift = _drift_frame([("Agriculture", LATE, 0.10, 0.20)])

        assert T.rename_candidates(drift).empty

    def test_corex_still_healthy_is_not_flagged(self):
        drift = _drift_frame([("War & military", 1980, 0.60, 0.90)])

        assert T.rename_candidates(drift).empty

    def test_both_arms_persisting_is_not_flagged(self):
        drift = _drift_frame([("Economy & jobs", 1980, 0.80, 0.90)])

        assert T.rename_candidates(drift).empty

    def test_thresholds_are_strict(self):
        """`corex_rel < 0.25` and `llm_rel > 0.50`: a row sitting exactly on
        either bound is not a candidate."""
        drift = _drift_frame([
            ("A", 1980, T.COREX_COLLAPSE, 0.90),
            ("B", 1980, 0.10, T.LLM_PERSIST),
        ])

        assert T.rename_candidates(drift).empty

    def test_results_are_sorted_by_drift_gap_descending(self):
        drift = _drift_frame([
            ("small gap", 1980, 0.20, 0.60),
            ("big gap", 1980, 0.05, 0.95),
            ("mid gap", 1980, 0.10, 0.80),
        ])

        out = T.rename_candidates(drift)

        assert list(out["issue"]) == ["big gap", "mid gap", "small gap"]
        assert out["drift_gap"].is_monotonic_decreasing

    def test_index_is_reset_so_the_table_reads_as_a_ranking(self):
        drift = _drift_frame([
            ("keep", 1980, 0.05, 0.95),
            ("drop", 1980, 0.90, 0.10),
            ("keep2", 1980, 0.10, 0.80),
        ])

        assert list(T.rename_candidates(drift).index) == [0, 1]

    def test_nan_rows_are_excluded(self):
        """An issue CorEx never labelled has `corex_rel == NaN`; a NaN comparison
        is False, so it must not be reported as a rename."""
        drift = _drift_frame([("never labelled", 1980, np.nan, 0.90)])

        assert T.rename_candidates(drift).empty

    def test_runs_on_the_real_drift_schema(self, drifting_arms):
        """A column-set check alone would pass on a function that returned the
        whole drift table, or nothing at all. Assert the selection instead: the
        result is a genuine subset of the input rows and every surviving row
        satisfies both thresholds.

        The fixture plants one real candidate, because `0 < len(out)` is the
        assertion the rest depends on: on an empty frame the threshold checks and
        the merge-back are all vacuously true.
        """
        df, llm = drifting_arms
        drift = T.rename_vs_death(df, llm)

        out = T.rename_candidates(drift)

        assert set(out.columns) == set(drift.columns)
        assert 0 < len(out) < len(drift)  # kept something AND dropped something
        assert DRIFTING_ISSUE in set(out["issue"])
        assert (out["corex_rel"] < T.COREX_COLLAPSE).all()
        assert (out["llm_rel"] > T.LLM_PERSIST).all()
        merged = out.merge(drift, on=list(drift.columns), how="inner")
        assert len(merged) == len(out)  # every row came from the drift table


# =========================================================================== #
# build_compositions
# =========================================================================== #
class TestBuildCompositions:
    def test_one_long_frame_covering_all_three_methods(self, arms, llm_legacy):
        comps = T.build_compositions(arms, llm_legacy)

        assert list(comps.columns) == ["doc_name", "method", "topic", "share",
                                       "n_paragraphs"]
        assert set(comps["method"]) == {"llm_taxonomy", "corex_legacy", "embed_k15"}

    def test_every_method_uses_the_same_denominator(self, arms, llm_legacy):
        """All paragraphs in the speech, including the LLM's abstentions — so the
        LLM arm is not silently advantaged by a smaller denominator."""
        comps = T.build_compositions(arms, llm_legacy)
        expected = arms.groupby("doc_name").size()

        got = comps.groupby("doc_name")["n_paragraphs"].unique()

        assert all(len(v) == 1 for v in got)
        assert {k: int(v[0]) for k, v in got.items()} == expected.to_dict()

    def test_corex_shares_are_hand_checkable(self, arms, llm_legacy):
        comps = T.build_compositions(arms, llm_legacy)
        doc = "doc00"
        row = comps[(comps["method"] == "corex_legacy")
                    & (comps["doc_name"] == doc)
                    & (comps["topic"] == "Economy & jobs")].iloc[0]
        sub = arms[arms["doc_name"] == doc]

        assert row["share"] == pytest.approx(sub["Economy & jobs"].mean())

    def test_corex_arm_is_keyed_by_crosswalk_name_not_column_name(
        self, arms, llm_legacy
    ):
        """`Discovered 5` is summed but published as `Security & peace`, so the
        CorEx arm lines up with the crosswalk's 16 keys."""
        comps = T.build_compositions(arms, llm_legacy)
        corex = comps[comps["method"] == "corex_legacy"]

        assert set(corex["topic"]) == set(CROSSWALK_ISSUES)
        assert "Discovered 5" not in set(corex["topic"])

    def test_llm_arm_covers_all_fifty_canonical_topics(self, arms, llm_legacy,
                                                       level2_names):
        comps = T.build_compositions(arms, llm_legacy)
        llm = comps[comps["method"] == "llm_taxonomy"]

        assert set(llm["topic"]) == set(level2_names)

    def test_llm_shares_are_hand_checkable(self, arms, llm_legacy, level2_names):
        comps = T.build_compositions(arms, llm_legacy)
        doc, topic = "doc00", level2_names[0]
        sub = arms[arms["doc_name"] == doc]
        expected = sub["topic_set"].map(lambda s: _norm_name(topic) in s).sum() / len(sub)

        row = comps[(comps["method"] == "llm_taxonomy")
                    & (comps["doc_name"] == doc)
                    & (comps["topic"] == topic)].iloc[0]

        assert row["share"] == pytest.approx(expected)

    def test_embedding_arm_is_zero_padded_and_sums_to_one(self, arms, llm_legacy):
        """Cluster assignment is single-label, so a speech's 15 cluster shares
        must sum to exactly 1."""
        comps = T.build_compositions(arms, llm_legacy)
        embed = comps[comps["method"] == "embed_k15"]

        sums = embed.groupby("doc_name")["share"].sum()

        assert np.allclose(sums.to_numpy(), 1.0)
        assert all(t.startswith("cluster_") for t in set(embed["topic"]))

    def test_off_taxonomy_topic_raises_rather_than_dropping_silently(
        self, arms, llm_legacy
    ):
        """`build_compositions` maps normalized names back to canonical ones; an
        unmappable name would otherwise pivot into a NaN column."""
        broken = arms.copy()
        broken.loc[broken.index[0], "topic_set"] = frozenset({"cryptocurrency"})

        with pytest.raises(ValueError, match="absent from frozen taxonomy_v1"):
            T.build_compositions(broken, llm_legacy)

    def test_one_row_per_doc_method_topic(self, arms, llm_legacy):
        comps = T.build_compositions(arms, llm_legacy)

        assert not comps.duplicated(["doc_name", "method", "topic"]).any()
        assert len(comps) == 8 * (50 + 16 + 15)

    def test_downstream_gets_a_one_line_method_switch(self, arms, llm_legacy):
        """The success metric: `comps[comps.method == "llm_taxonomy"]`."""
        comps = T.build_compositions(arms, llm_legacy)

        assert len(comps[comps["method"] == "llm_taxonomy"]) == 8 * 50


# =========================================================================== #
# agreement_drivers
# =========================================================================== #
def _agreement_from_arrays(pairs: dict[str, tuple]) -> pd.DataFrame:
    """Build an overall-only agreement table by running the module's own
    `_pair_metrics` over real boolean arrays, so every Jaccard is a genuine
    set-overlap rather than a hand-set number."""
    rows = [{"issue": issue, "era": np.nan, **T._pair_metrics(llm, corex)}
            for issue, (llm, corex) in pairs.items()]
    return pd.DataFrame(rows)


def _synthetic_pairs(seed: int = 0, n: int = 500) -> dict[str, tuple]:
    """One boolean pair per legacy issue, with deliberately varied base rates so
    the divergence spread is wide enough for a rank correlation."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for k, issue in enumerate(LEGACY):
        p_llm = 0.05 + 0.04 * k                     # 0.05 .. 0.61
        p_corex = 0.40 - 0.02 * k                   # 0.40 .. 0.12
        llm = rng.random(n) < p_llm
        corex = (llm & (rng.random(n) < 0.5)) | (rng.random(n) < p_corex)
        pairs[issue] = (llm, corex)
    return pairs


@pytest.fixture
def drivers() -> dict:
    primary = _agreement_from_arrays(_synthetic_pairs(seed=0))
    sensitivity = _agreement_from_arrays(_synthetic_pairs(seed=1))
    return T.agreement_drivers(primary, sensitivity)


class TestAgreementDriversCeiling:
    """The report's central caveat, as an executable identity.

    `J <= min(a,b)/max(a,b) = exp(-|log(a/b)|) = exp(-D)` holds deterministically
    for every issue before any data is seen — so a negative rank association
    between D and J is partly guaranteed by arithmetic rather than discovered.
    These are exact-tolerance assertions on synthetic data; any future metric
    change that breaks them breaks the disclosure the headline rests on.
    """

    def test_ceiling_equals_min_over_max(self, drivers):
        t = drivers["drivers"]

        expected = (np.minimum(t["n_llm"], t["n_corex"])
                    / np.maximum(t["n_llm"], t["n_corex"]))

        assert np.allclose(t["ceiling"], expected, atol=0, rtol=0)

    def test_ceiling_equals_exp_minus_divergence(self, drivers):
        """The identity that makes base-rate divergence the negative log of the
        outcome metric's own ceiling."""
        t = drivers["drivers"]

        assert np.allclose(t["ceiling"], np.exp(-t["divergence"]), atol=1e-12)

    def test_divergence_is_the_absolute_log_base_rate_ratio(self, drivers):
        t = drivers["drivers"]

        assert np.allclose(t["divergence"],
                           np.abs(np.log(t["n_llm"] / t["n_corex"])), atol=1e-15)

    def test_the_bound_holds_for_every_issue(self, drivers):
        t = drivers["drivers"]

        assert len(t) == 15
        assert t["bound_holds"].all()
        assert (t["jaccard"] <= t["ceiling"] + 1e-12).all()

    def test_fill_is_the_realised_share_of_the_ceiling(self, drivers):
        t = drivers["drivers"]

        assert np.allclose(t["fill"], t["jaccard"] / t["ceiling"])
        assert ((t["fill"] >= 0) & (t["fill"] <= 1 + 1e-12)).all()

    def test_the_controlled_test_is_reported_alongside_the_uncontrolled_one(
        self, drivers
    ):
        """Publishing D-vs-Jaccard without D-vs-fill would present a partly
        tautological association as a discovery."""
        stats = drivers["spearman"]

        assert "divergence_vs_jaccard" in stats
        assert "divergence_vs_fill_CONTROLLED" in stats

    def test_ceiling_is_genuinely_binding_not_a_formality(self, drivers):
        """If every fill were ~1 the control would be vacuous. The fill fraction
        must actually vary AND must sit meaningfully below the ceiling — a
        spread test alone would pass on fills clustered at 0.95-0.99, where the
        ceiling is attained everywhere and `fill` carries no information."""
        t = drivers["drivers"]

        assert t["fill"].max() - t["fill"].min() > 0.05
        assert t["fill"].max() < 0.9

    @pytest.mark.parametrize("seed", range(6))
    def test_identity_survives_arbitrary_base_rates(self, seed):
        primary = _agreement_from_arrays(_synthetic_pairs(seed=seed))
        out = T.agreement_drivers(primary, primary)["drivers"]

        assert out["bound_holds"].all()
        assert np.allclose(out["ceiling"], np.exp(-out["divergence"]), atol=1e-12)


# --------------------------------------------------------------------------- #
# The §0 conclusion itself: "the association does not survive the control".
# --------------------------------------------------------------------------- #
_CEILING_N = 2000       # paragraphs per synthetic issue
_CEILING_N_LLM = 400    # n_llm held fixed, so divergence is driven by n_corex
# n_corex per issue: 400/40 down to 400/385, i.e. D from 2.30 to 0.04.
_CEILING_N_COREX = [40, 55, 70, 90, 110, 135, 160, 190, 220, 250, 280, 310, 340,
                    365, 385]


def _pair_at_fill(n_corex: int, fill: float) -> tuple:
    """Boolean arrays whose Jaccard realises `fill` x its own algebraic ceiling.

    CorEx's positives are mostly NESTED inside the LLM's, which is what puts
    Jaccard near its `min/max` ceiling; the few positives placed OUTSIDE enlarge
    the union and pull the realised fraction down to the requested value.

    With A = n_llm, C = n_corex and B of C nested: J = B/(A + C - B), so
    B = f*C*(A+C)/(A + f*C) inverts the fill exactly (up to integer rounding,
    which the assertions absorb by reading `fill` back off the real output).
    """
    a = _CEILING_N_LLM
    b = min(round(fill * n_corex * (a + n_corex) / (a + fill * n_corex)), n_corex)
    llm = np.zeros(_CEILING_N, dtype=bool)
    llm[:a] = True
    corex = np.zeros(_CEILING_N, dtype=bool)
    corex[:b] = True
    corex[a:a + (n_corex - b)] = True
    return llm, corex


def _drivers_at_fills(fills: list[float]) -> dict:
    """Run the real `agreement_drivers` over 15 issues with prescribed fills."""
    pairs = {issue: _pair_at_fill(_CEILING_N_COREX[k], fills[k])
             for k, issue in enumerate(LEGACY)}
    primary = _agreement_from_arrays(pairs)
    return T.agreement_drivers(primary, primary)


# Fills near their ceilings, ordered so their rank correlation with divergence
# is ~0: the ceiling is the ONLY thing linking divergence to Jaccard here.
CEILING_ONLY_FILLS = [0.90, 0.97, 0.86, 0.95, 0.88, 0.99, 0.87, 0.93, 0.96,
                      0.85, 0.98, 0.89, 0.94, 0.91, 0.92]
# The contrast case: fill rises monotonically with n_corex (i.e. falls with
# divergence), so a real effect exists on top of the ceiling.
REAL_EFFECT_FILLS = [0.55, 0.58, 0.62, 0.65, 0.68, 0.72, 0.75, 0.78, 0.81,
                     0.84, 0.87, 0.90, 0.93, 0.96, 0.99]


@pytest.fixture(scope="module")
def ceiling_only() -> dict:
    return _drivers_at_fills(CEILING_ONLY_FILLS)


@pytest.fixture(scope="module")
def real_effect() -> dict:
    return _drivers_at_fills(REAL_EFFECT_FILLS)


class TestControlledAssociationConclusion:
    """Report §0's central caveat, as an executable claim rather than a shape.

    §0 states: "**The association does not survive the control.** |rho| falls
    from 0.74 to 0.18 and p rises from 0.0016 to 0.52". The structural tests
    above only prove that BOTH statistics are reported; nothing asserted that
    controlling for the ceiling actually attenuates the association, which is
    the finding the report's whole honesty argument rests on and the reason
    IMPLEMENT failed its first gate.

    The fixture is built so the ceiling is the ONLY link between divergence and
    Jaccard: every issue's overlap sits near its algebraic maximum, and the
    realised fraction is jittered in an order deliberately uncorrelated with
    divergence. The uncontrolled rank association is then near-perfect and
    entirely arithmetic, and the controlled one must collapse.

    `TestControlDetectsARealEffect` is the other half — without it, this class
    would pass equally well against an `agreement_drivers` that always reported
    a dead controlled statistic.
    """

    def test_the_fixture_really_does_sit_near_the_ceiling(self, ceiling_only):
        """Anti-vacuity: if the fills were low and varied, a weak controlled rho
        would prove nothing about the ceiling. Assert the premise first."""
        t = ceiling_only["drivers"]

        assert len(t) == 15
        assert t["fill"].min() > 0.8
        assert t["bound_holds"].all()

    def test_divergence_spans_a_wide_range(self, ceiling_only):
        """A rank correlation over a degenerate predictor is meaningless."""
        d = ceiling_only["drivers"]["divergence"]

        assert d.max() - d.min() > 2.0

    def test_the_uncontrolled_association_is_strongly_negative(self, ceiling_only):
        """By construction J ~= const x exp(-D), so D vs J is a near-perfect
        monotone decrease — every bit of it arithmetic."""
        rho = ceiling_only["spearman"]["divergence_vs_jaccard"]

        assert rho["rho"] < -0.9
        assert rho["p"] < ceiling_only["spearman"]["bonferroni_alpha"]

    def test_the_association_does_not_survive_the_control(self, ceiling_only):
        """THE §0 CLAIM. Removing the ceiling must collapse the association."""
        stats = ceiling_only["spearman"]
        uncontrolled = stats["divergence_vs_jaccard"]
        controlled = stats["divergence_vs_fill_CONTROLLED"]

        assert abs(controlled["rho"]) < abs(uncontrolled["rho"])
        assert abs(controlled["rho"]) < 0.3
        assert controlled["p"] > uncontrolled["p"]
        assert controlled["p"] > stats["bonferroni_alpha"]

    def test_the_control_is_what_moves_it_not_the_sample_size(self, ceiling_only):
        """Both statistics are computed at n = 15 over the same 15 issues, so
        the attenuation cannot be attributed to a different sample."""
        assert len(ceiling_only["drivers"]) == 15


class TestControlDetectsARealEffect:
    """The control must be a MEASUREMENT, not a guaranteed null.

    If `fill` genuinely co-varies with divergence — i.e. base-rate divergence
    explains something beyond its own algebra — the controlled statistic has to
    stay large. Without this, `test_the_association_does_not_survive_the_control`
    would also pass against an implementation that computed `fill` wrongly and
    always produced noise.
    """

    def test_a_genuine_effect_survives_the_control(self, real_effect):
        controlled = real_effect["spearman"]["divergence_vs_fill_CONTROLLED"]

        assert controlled["rho"] < -0.9
        assert controlled["p"] < real_effect["spearman"]["bonferroni_alpha"]

    def test_the_two_fixtures_differ_only_in_the_fill_pattern(
        self, real_effect, ceiling_only
    ):
        """Same issues, same base rates, same ceilings — so the opposite verdicts
        of the two classes are attributable to the fill pattern alone."""
        a, b = real_effect["drivers"], ceiling_only["drivers"]

        assert list(a["n_llm"]) == list(b["n_llm"])
        assert list(a["n_corex"]) == list(b["n_corex"])
        np.testing.assert_allclose(a["ceiling"], b["ceiling"])
        assert a["fill"].max() - a["fill"].min() > 0.4  # ...and the fills do not


class TestEveryPredictorGetsTheCeilingControl:
    """Fan-out and NPMI must be controllable too, not just base-rate divergence.

    Fan-out inflates `n_llm`, so it acts through the SAME bounded channel the
    ceiling disclosure is about: an uncontrolled fan-out correlation is subject
    to the identical artifact. The report cites the controlled figure, so it has
    to exist and has to be a real control rather than an alias of the
    uncontrolled one.
    """

    def _drivers_with_predictors(self):
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0))
        coherence = {issue: {"npmi": 0.01 * k}
                     for k, issue in enumerate(LEGACY)}
        # Fan-out rises with k, exactly as n_llm does in _synthetic_pairs, so
        # fan-out is entangled with the base rate the way it is in the corpus.
        crosswalk = {issue: {f"t{i}" for i in range(k + 1)}
                     for k, issue in enumerate(LEGACY)}
        return T.agreement_drivers(primary, primary, coherence, crosswalk)

    def test_controlled_fanout_and_npmi_statistics_are_reported(self):
        stats = self._drivers_with_predictors()["spearman"]

        assert "fanout_vs_fill_CONTROLLED" in stats
        assert "npmi_vs_fill_CONTROLLED" in stats
        assert "fanout_vs_divergence" in stats

    def test_the_controlled_fanout_statistic_is_not_the_uncontrolled_one(self):
        """A control that silently aliased its uncontrolled twin would pass every
        presence check while disclosing nothing."""
        stats = self._drivers_with_predictors()["spearman"]

        assert (stats["fanout_vs_fill_CONTROLLED"]["rho"]
                != pytest.approx(stats["fanout_vs_jaccard"]["rho"]))
        assert (stats["fanout_vs_fill_CONTROLLED"]["rho"]
                != pytest.approx(stats["fanout_vs_kappa"]["rho"]))

    def test_fanout_entangled_with_base_rate_shows_it_in_the_divergence_stat(self):
        """The substantive claim (wide crosswalk rows inflate LLM breadth) is
        evidenced by fan-out vs divergence, which is NOT ceiling-bounded.

        Asserted on MAGNITUDE, not sign: divergence is `|log(a/b)|`, so whether
        a rising fan-out pushes it up or down depends on which side of parity the
        two base rates sit. The entanglement is the point — it is why the
        uncontrolled fan-out correlation cannot be read as evidence.
        """
        stats = self._drivers_with_predictors()["spearman"]

        assert abs(stats["fanout_vs_divergence"]["rho"]) > 0.4

    def test_bonferroni_counts_only_the_uncontrolled_tests_actually_run(self):
        """Hardcoding 6 would misstate the correction whenever a predictor is
        absent — the count must follow what was computed."""
        with_predictors = self._drivers_with_predictors()["spearman"]
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0))
        bare = T.agreement_drivers(primary, primary)["spearman"]

        assert with_predictors["n_tests"] == 6      # 3 predictors x 2 metrics
        assert bare["n_tests"] == 2                 # divergence only
        assert bare["bonferroni_alpha"] == pytest.approx(0.05 / 2)


class TestAgreementDriversRejectsEmptyArms:
    def test_an_empty_arm_raises_instead_of_silently_voiding_every_spearman(self):
        """A zero arm makes the log ratio infinite and the ceiling zero, which
        would propagate NaN through every correlation and void the table without
        any visible failure."""
        pairs = _synthetic_pairs(seed=0)
        n = len(next(iter(pairs.values()))[0])
        pairs["Immigration"] = (np.zeros(n, dtype=bool), np.ones(n, dtype=bool))
        primary = _agreement_from_arrays(pairs)

        with pytest.raises(ValueError, match="empty arm"):
            T.agreement_drivers(primary, primary)


class TestAgreementDrivers:
    def test_only_the_overall_rows_are_used(self):
        """Era rows must not be folded into the per-issue driver frame."""
        primary = _agreement_from_arrays(_synthetic_pairs())
        era_rows = primary.copy()
        era_rows["era"] = 1980.0

        out = T.agreement_drivers(pd.concat([primary, era_rows], ignore_index=True),
                                  primary)

        assert len(out["drivers"]) == 15

    def test_security_and_peace_is_excluded_from_the_driver_frame(self):
        """It has no anchor-coherence entry and is not one of the legacy 15, so
        including it would change the n the report's p-values are computed at."""
        pairs = _synthetic_pairs()
        pairs[SECURITY_PEACE] = pairs[LEGACY[0]]
        primary = _agreement_from_arrays(pairs)

        out = T.agreement_drivers(primary, primary)

        assert SECURITY_PEACE not in out["drivers"].index
        assert len(out["drivers"]) == 15

    def test_optional_predictors_are_absent_when_not_supplied(self, drivers):
        assert "npmi" not in drivers["drivers"].columns
        assert "fanout" not in drivers["drivers"].columns
        assert "npmi_vs_jaccard" not in drivers["spearman"]
        assert "fanout_vs_jaccard" not in drivers["spearman"]

    def test_coherence_adds_the_npmi_predictor(self):
        primary = _agreement_from_arrays(_synthetic_pairs())
        coherence = {i: {"npmi": 0.02 * k} for k, i in enumerate(LEGACY)}

        out = T.agreement_drivers(primary, primary, coherence=coherence)

        assert list(out["drivers"]["npmi"]) == [0.02 * k for k in range(15)]
        assert {"npmi_vs_jaccard", "npmi_vs_kappa"} <= set(out["spearman"])

    def test_crosswalk_adds_the_fanout_predictor(self, crosswalk):
        primary = _agreement_from_arrays(_synthetic_pairs())

        out = T.agreement_drivers(primary, primary, crosswalk=crosswalk)

        assert list(out["drivers"]["fanout"]) == [len(crosswalk[i]) for i in LEGACY]
        assert {"fanout_vs_jaccard", "fanout_vs_kappa"} <= set(out["spearman"])

    def test_multiple_comparison_correction_matches_the_published_table(self):
        """3 predictors x 2 metrics = the 6 pre-planned tests in the report's
        §0 table. Counted from the tests actually run, not hardcoded, so the
        correction stays right when a predictor is absent (see
        TestEveryPredictorGetsTheCeilingControl for the reduced case)."""
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0))
        coherence = {issue: {"npmi": 0.01 * k} for k, issue in enumerate(LEGACY)}
        crosswalk = {issue: {f"t{i}" for i in range(k + 1)}
                     for k, issue in enumerate(LEGACY)}

        stats = T.agreement_drivers(primary, primary, coherence,
                                    crosswalk)["spearman"]

        assert stats["n_tests"] == 6
        assert stats["bonferroni_alpha"] == pytest.approx(0.05 / 6)

    def test_spearman_entries_carry_rho_and_p(self, drivers):
        entry = drivers["spearman"]["divergence_vs_jaccard"]

        assert set(entry) == {"rho", "p"}
        assert -1.0 <= entry["rho"] <= 1.0
        assert 0.0 <= entry["p"] <= 1.0

    def test_sensitivity_summarises_the_unlabeled_paragraph_policy(self):
        """§6: how much the numbers move if the LLM's abstentions are counted as
        all-negative rows instead of dropped."""
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0))
        sensitivity = primary.copy()
        sensitivity["jaccard"] = sensitivity["jaccard"] + 0.01
        sensitivity["kappa"] = sensitivity["kappa"] - 0.02

        out = T.agreement_drivers(primary, sensitivity)["sensitivity"]

        assert out["mean_abs_delta_jaccard"] == pytest.approx(0.01)
        assert out["mean_abs_delta_kappa"] == pytest.approx(0.02)
        assert out["max_abs_delta"] == pytest.approx(0.02)

    def test_max_abs_delta_takes_the_larger_of_the_two_metrics(self):
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0))
        sensitivity = primary.copy()
        sensitivity["jaccard"] = sensitivity["jaccard"] + 0.01
        sensitivity["kappa"] = sensitivity["kappa"] + 0.05

        out = T.agreement_drivers(primary, sensitivity)["sensitivity"]

        assert out["max_abs_delta"] == pytest.approx(0.05)

    def test_max_abs_delta_issue_names_the_metric_the_magnitude_came_from(self):
        """The magnitude and the issue name must describe the SAME metric.

        `max_abs_delta` is the max over BOTH metrics, so the argmax must be taken
        on whichever metric carried that max — not on Jaccard unconditionally.
        Here kappa carries the larger delta, so the reported issue is kappa's,
        not Jaccard's. (Was a known defect fixed at SIMPLIFY; this test used to
        pin the mismatch.)
        """
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0)).set_index("issue")
        sensitivity = primary.copy()
        # Jaccard moves most on issue[0]; kappa moves more, on issue[1].
        sensitivity["jaccard"] = primary["jaccard"]
        sensitivity.loc[LEGACY[0], "jaccard"] += 0.10
        sensitivity["kappa"] = primary["kappa"]
        sensitivity.loc[LEGACY[1], "kappa"] += 0.40

        out = T.agreement_drivers(primary.reset_index(),
                                  sensitivity.reset_index())["sensitivity"]

        assert out["max_abs_delta"] == pytest.approx(0.40)      # from kappa...
        assert out["max_abs_delta_issue"] == LEGACY[1]          # ...so kappa names it
        assert out["max_abs_delta_issue"] != LEGACY[0]

    def test_max_abs_delta_issue_follows_jaccard_when_jaccard_carries_the_max(self):
        """The mirror case, so the fix is not just "always use kappa"."""
        primary = _agreement_from_arrays(_synthetic_pairs(seed=0)).set_index("issue")
        sensitivity = primary.copy()
        sensitivity["jaccard"] = primary["jaccard"]
        sensitivity.loc[LEGACY[0], "jaccard"] += 0.40
        sensitivity["kappa"] = primary["kappa"]
        sensitivity.loc[LEGACY[1], "kappa"] += 0.10

        out = T.agreement_drivers(primary.reset_index(),
                                  sensitivity.reset_index())["sensitivity"]

        assert out["max_abs_delta"] == pytest.approx(0.40)
        assert out["max_abs_delta_issue"] == LEGACY[0]

    def test_returns_the_three_report_sections(self, drivers):
        assert set(drivers) == {"drivers", "spearman", "sensitivity"}


# =========================================================================== #
# run — hermetic end-to-end wiring smoke test
# =========================================================================== #
class TestRun:
    @staticmethod
    def _hermetic_env(tmp_path, monkeypatch, level2_names):
        """Repoint every path `run` reads or writes at tmp_path.

        Returns the two output paths so a test can assert on their existence.
        """
        _write_arms(tmp_path, monkeypatch, level2_names,
                    collapse_corex_issue=DRIFTING_ISSUE)
        agreement = tmp_path / "agreement.parquet"
        comps = tmp_path / "comps.parquet"
        monkeypatch.setattr(T, "AGREEMENT_PATH", agreement)
        monkeypatch.setattr(T, "COMPOSITIONS_PATH", comps)

        meta = tmp_path / "issues_meta.json"
        meta.write_text(json.dumps({
            "issues": LEGACY,
            "coherence": {i: {"npmi": 0.02 * k} for k, i in enumerate(LEGACY)},
        }))
        monkeypatch.setattr(T, "ISSUES_META_PATH", meta)

        cmeta = tmp_path / "clusters_meta.json"
        cmeta.write_text(json.dumps({"clusters": {}}))
        monkeypatch.setattr(T, "CLUSTERS_META_PATH", cmeta)
        monkeypatch.setattr(
            T, "load",
            lambda: pd.DataFrame({"doc_name": [f"doc{d:02d}" for d in range(8)]}),
        )
        return agreement, comps

    @pytest.fixture
    def result(self, tmp_path, monkeypatch, level2_names):
        self._hermetic_env(tmp_path, monkeypatch, level2_names)
        return T.run()

    def test_a_failed_run_leaves_no_artifacts_behind(
        self, tmp_path, monkeypatch, level2_names
    ):
        """Both parquets are persisted only after everything that can raise has
        succeeded. Writing them earlier left a half-finished run on disk with
        nothing to distinguish it from a complete one."""
        agreement, comps = self._hermetic_env(tmp_path, monkeypatch, level2_names)

        def boom(*a, **k):
            raise RuntimeError("late failure")

        monkeypatch.setattr(T, "agreement_drivers", boom)

        with pytest.raises(RuntimeError, match="late failure"):
            T.run()

        assert not agreement.exists()
        assert not comps.exists()

    def test_returns_every_artifact_the_report_writer_needs(self, result):
        assert set(result) == {
            "drivers", "df", "llm_legacy", "agreement", "agreement_sensitivity",
            "drift", "rename_candidates", "compositions", "cluster_meta",
            "speeches",
        }

    def test_persists_the_per_era_agreement_table(self, result):
        """AC2's deliverable is per issue PER ERA, so the era rows are persisted,
        not just the aggregate that made it into the report."""
        saved = pd.read_parquet(T.AGREEMENT_PATH)

        assert saved["era"].notna().any()
        assert set(saved["issue"]) == set(CROSSWALK_ISSUES)

    def test_persists_the_compositions(self, result):
        saved = pd.read_parquet(T.COMPOSITIONS_PATH)

        assert set(saved["method"]) == {"llm_taxonomy", "corex_legacy", "embed_k15"}

    def test_primary_and_sensitivity_use_opposite_unlabeled_policies(self, result):
        primary = result["agreement"]
        sens = result["agreement_sensitivity"]
        n_labeled = int((~result["df"]["llm_unlabeled"]).sum())

        assert set(primary.loc[primary["era"].isna(), "n"]) == {n_labeled}
        assert set(sens.loc[sens["era"].isna(), "n"]) == {len(result["df"])}

    def test_drivers_carry_both_optional_predictors(self, result):
        cols = result["drivers"]["drivers"].columns

        assert {"npmi", "fanout", "ceiling", "fill", "bound_holds"} <= set(cols)
        assert result["drivers"]["drivers"]["bound_holds"].all()

    def test_rename_candidates_is_a_subset_of_the_drift_table(self, result):
        """`len(a) <= len(b)` is nearly free; assert the rows themselves are
        drift rows and that each one meets the published thresholds.

        `0 < len(rc)` first: the fixture plants one candidate precisely because
        every assertion below is vacuously true on an empty frame.
        """
        rc, drift = result["rename_candidates"], result["drift"]

        assert set(rc.columns) == set(drift.columns)
        assert 0 < len(rc) < len(drift)
        assert DRIFTING_ISSUE in set(rc["issue"])
        assert len(rc.merge(drift, on=list(drift.columns), how="inner")) == len(rc)
        assert (rc["corex_rel"] < T.COREX_COLLAPSE).all()
        assert (rc["llm_rel"] > T.LLM_PERSIST).all()
