"""Label resolution for `register.py` — the highest-consequence path in the module.

111 of the 52,855 level-2 label assignments the annotation run returned are
case-variants of real taxonomy names (`the War On Terror` for `the War on
Terror`), and 84 of them land on one modern-era topic. An exact-match join would
drop those 111 silently, pulling modern *policy* attention down and flattering
the "presidents talk about policy less" half of the published headline. So the
resolution is casefolded, and anything that still fails to resolve raises rather
than being dropped.

These tests cover `build_taxonomy_index`, `TaxonomyIndex.resolve_labels`,
`TaxonomyIndex.non_policy_level2` and `label_resolution_report` on synthetic
taxonomies, plus a read-only anchor against the FROZEN committed artifacts that
pins the 111/84 claim to the real data rather than restating it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from presidential_profiles import register as R

# Read by absolute worktree path, never the module constant: under a git
# worktree `ANNOTATIONS_DIR` can resolve to the main repo, and conftest's autouse
# `redirect_annotation_dirs` repoints it at tmp_path.
_ARTIFACTS = Path(__file__).resolve().parents[1] / "data" / "llm_annotations"


# =========================================================================== #
class TestBuildTaxonomyIndex:
    def test_indexes_names_parents_and_kinds(self, register_taxonomy):
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.level1 == ("Economy", "Security", "Ceremonial", "Personal Narrative")
        assert index.level2[0] == "Jobs & Wages"
        assert len(index.level2) == 5
        assert index.parent["the War on Terror"] == "Security"
        assert index.kind["Ceremonial"] == "non-policy"

    def test_non_policy_level2_is_the_topics_under_non_policy_domains(
        self, register_taxonomy
    ):
        """The non-policy share is defined by the topic's PARENT domain kind, not
        by any property of the topic itself — so the set must be exactly the
        level-2 topics whose level-1 domain is marked non-policy."""
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.non_policy_level2 == frozenset(
            {"Holidays & Tributes", "Reflection on Office"}
        )

    def test_flipping_a_domain_to_policy_removes_its_topics_from_the_set(
        self, register_taxonomy
    ):
        """Pins the direction of the lookup: `Ceremonial` going policy must drop
        its topic, and must not disturb the other non-policy domain."""
        register_taxonomy["level1"][2]["kind"] = "policy"
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.non_policy_level2 == frozenset({"Reflection on Office"})

    def test_casefold_collision_raises_rather_than_shadowing_a_topic(
        self, register_taxonomy
    ):
        """Two level-2 names that differ only in case would make case-insensitive
        resolution ambiguous — one topic would silently absorb the other's
        assignments. Refused at index-construction time."""
        register_taxonomy["level2"].append(
            {"name": "JOBS & WAGES", "definition": "d", "level1": "Economy"}
        )

        with pytest.raises(ValueError, match="collide when casefolded"):
            R.build_taxonomy_index(register_taxonomy)

    def test_orphan_parent_raises(self, register_taxonomy):
        """A level-2 topic naming a nonexistent domain would KeyError later, deep
        inside the level-1 rollup; caught up front with the offending name."""
        register_taxonomy["level2"].append(
            {"name": "Ghost Topic", "definition": "d", "level1": "Nonexistent"}
        )

        with pytest.raises(ValueError, match="unknown level-1 domains"):
            R.build_taxonomy_index(register_taxonomy)

    def test_distinct_names_that_share_no_casefold_are_accepted(
        self, register_taxonomy
    ):
        """The collision guard fires on collisions only — a taxonomy whose names
        merely share words still indexes."""
        register_taxonomy["level2"].append(
            {"name": "Jobs & Wages Abroad", "definition": "d", "level1": "Economy"}
        )

        assert len(R.build_taxonomy_index(register_taxonomy).level2) == 6


# =========================================================================== #
class TestResolveLabels:
    def test_exact_names_round_trip_unchanged(self, register_taxonomy):
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.resolve_labels(["Jobs & Wages", "the War on Terror"]) == [
            "Jobs & Wages",
            "the War on Terror",
        ]

    @pytest.mark.parametrize(
        "variant",
        [
            "the War On Terror",  # the exact shape of the real 84-assignment variant
            "The War on Terror",
            "THE WAR ON TERROR",
            "the war on terror",
        ],
    )
    def test_case_variants_resolve_to_the_canonical_name(
        self, register_taxonomy, variant
    ):
        """The repair itself. Each variant must come back as the ONE canonical
        spelling — not as itself, and not dropped."""
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.resolve_labels([variant]) == ["the War on Terror"]

    def test_unknown_label_raises_and_is_not_dropped(self, register_taxonomy):
        """A label that does not resolve even case-insensitively is a genuine
        unknown. Dropping it would bias whichever era produced it; the module
        refuses instead."""
        index = R.build_taxonomy_index(register_taxonomy)

        with pytest.raises(KeyError, match="does not resolve"):
            index.resolve_labels(["Jobs & Wages", "Cryptocurrency Policy"])

    def test_whitespace_variant_is_treated_as_unknown_not_silently_trimmed(
        self, register_taxonomy
    ):
        """Casefolding is the ONLY normalization applied. A padded name is not
        quietly stripped — the repair's scope is deliberately narrow, because a
        broader fuzzy match could merge two genuinely different topics."""
        index = R.build_taxonomy_index(register_taxonomy)

        with pytest.raises(KeyError):
            index.resolve_labels([" Jobs & Wages "])

    def test_empty_label_list_resolves_to_empty(self, register_taxonomy):
        """A paragraph the model gave no topics is legal (1.1% of the corpus) and
        must not raise — it becomes the `zero_topic_share` series."""
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.resolve_labels([]) == []

    def test_repeated_labels_are_preserved_not_deduplicated(self, register_taxonomy):
        """Counts are over label ASSIGNMENTS, so resolution must not collapse
        duplicates — that would silently change the entropy denominator."""
        index = R.build_taxonomy_index(register_taxonomy)

        assert index.resolve_labels(["Jobs & Wages", "Jobs & Wages"]) == [
            "Jobs & Wages",
            "Jobs & Wages",
        ]


# =========================================================================== #
class TestLabelResolutionReport:
    def _annotations(self, topic_lists):
        return pd.DataFrame({"topics": topic_lists})

    def test_counts_assignments_and_flags_the_case_variant_as_inexact(
        self, register_taxonomy
    ):
        """The report is what the published note cites for the size of the
        repair, so both halves must be right: the count of assignments per
        returned string, and whether that string matched taxonomy_v1 exactly."""
        index = R.build_taxonomy_index(register_taxonomy)
        annotations = self._annotations([
            ["the War On Terror"],
            ["the War On Terror", "Jobs & Wages"],
            ["the War on Terror"],
        ])

        report = R.label_resolution_report(annotations, index).set_index(
            "returned_label"
        )

        assert report.loc["the War On Terror", "n_assignments"] == 2
        assert bool(report.loc["the War On Terror", "exact_match"]) is False
        assert report.loc["the War On Terror", "canonical"] == "the War on Terror"
        assert bool(report.loc["the War on Terror", "exact_match"]) is True
        assert report.loc["Jobs & Wages", "n_assignments"] == 1

    def test_rows_are_sorted_by_assignment_count_descending(self, register_taxonomy):
        index = R.build_taxonomy_index(register_taxonomy)
        annotations = self._annotations([
            ["Jobs & Wages"],
            ["the War on Terror"],
            ["the War on Terror"],
            ["the War on Terror"],
        ])

        report = R.label_resolution_report(annotations, index)

        assert report["returned_label"].tolist() == ["the War on Terror", "Jobs & Wages"]
        assert report["n_assignments"].tolist() == [3, 1]

    def test_unresolvable_label_raises_from_the_report_too(self, register_taxonomy):
        """The report walks the same resolution path, so an unknown label is
        fatal here as well — the diagnostic can never disagree with the frame it
        is meant to explain."""
        index = R.build_taxonomy_index(register_taxonomy)

        with pytest.raises(KeyError, match="does not resolve"):
            R.label_resolution_report(self._annotations([["Not A Topic"]]), index)


# =========================================================================== #
# Read-only anchors against the FROZEN committed artifacts (two small files, no
# API). These pin the module docstring's numeric claims — 111 repaired
# assignments across 8 labels, 84 of them on one modern topic, 52,855 total, a
# 50/17 taxonomy — to the real data instead of restating them in prose. None of
# the four can be anchored on a synthetic fixture, which is the whole reason the
# suite reads the real corpus exactly here and nowhere else.
@pytest.fixture(scope="module")
def frozen():
    """Read the frozen artifacts ONCE for the whole class.

    Per-test this was three reads of the 36k-row annotation parquet and three
    walks of all 52,855 label assignments. Building the report here is itself the
    "everything resolves" assertion — `label_resolution_report` raises on an
    unresolvable label — so a corpus that broke that claim surfaces as a loud
    setup error on every test in the class rather than a single failure.

    Module-scoped deliberately: it takes no function-scoped fixture, and it reads
    by absolute worktree path, so conftest's autouse `redirect_annotation_dirs`
    (which repoints the module path constants at tmp_path) cannot reach it.
    """
    index = R.build_taxonomy_index(
        json.loads((_ARTIFACTS / "taxonomy_v1.json").read_text())
    )
    annotations = pd.read_parquet(
        _ARTIFACTS / "paragraph_annotations.parquet", columns=["topics"]
    )
    return SimpleNamespace(
        index=index,
        annotations=annotations,
        report=R.label_resolution_report(annotations, index),
    )


class TestCommittedArtifacts:
    def test_real_taxonomy_indexes_with_no_casefold_collision(self, frozen):
        assert len(frozen.index.level2) == 50
        assert len(frozen.index.level1) == 17
        assert len(frozen.index.by_casefold) == len(frozen.index.level2)
        assert set(frozen.index.parent.values()) <= set(frozen.index.level1)

    def test_every_returned_label_in_the_frozen_corpus_resolves(self, frozen):
        """The load-bearing claim: `build_paragraph_frame` raises on any
        unresolvable label, so if this ever failed the whole module would refuse
        to run. Asserted directly rather than inferred — the report was built
        without raising, and every canonical name it produced is a real level-2
        topic."""
        assert len(frozen.report) > 0
        assert frozen.report["canonical"].isin(frozen.index.level2).all()

    def test_case_variant_repair_is_exactly_111_assignments_across_8_labels(
        self, frozen
    ):
        """The docstring's numbers, anchored. If a future annotation run changes
        the corpus this fails loudly and the note's cited figures get updated
        deliberately — which is the point of a frozen, provenance-stamped
        artifact."""
        inexact = frozen.report[~frozen.report["exact_match"]]

        assert len(inexact) == 8
        assert int(inexact["n_assignments"].sum()) == 111
        assert int(frozen.report["n_assignments"].sum()) == 52_855

    def test_84_of_the_repaired_assignments_land_on_one_modern_topic(self, frozen):
        """Why the repair matters and is not cosmetic: three quarters of it sits
        on a single modern-era policy topic, so dropping the variants would bend
        exactly the era and exactly the direction the headline is about."""
        inexact = frozen.report[~frozen.report["exact_match"]]
        biggest = inexact.sort_values("n_assignments", ascending=False).iloc[0]

        assert biggest["n_assignments"] == 84
        assert biggest["canonical"] == (
            "Iraq, Gulf Wars, the War on Terror & Interventions"
        )
        assert biggest["returned_label"] != biggest["canonical"]
