"""Frame assembly for `register.py`: `build_paragraph_frame`, `build_speech_panel`,
`load_inputs`, and the keyed-merge discipline both frames depend on.

The merge contract is the repo-wide one stated in CLAUDE.md and
`taxonomy.py::_require_full_merge`: `validate="one_to_one"` catches DUPLICATE
keys, but an inner join silently DROPS rows when the key sets merely diverge, so
every merge is followed by a row-count assertion. In this module a silent drop
would shrink the corpus a published trend is measured over while every reported
`n_paragraphs` still claimed the full count — so both halves are pinned here.

All frames are tiny and hand-countable. Nothing reads the 36k-row corpus.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import register as R

SOTU = R.SOTU_TYPE
INAUGURAL = "inaugural_address"


def _index(register_taxonomy):
    return R.build_taxonomy_index(register_taxonomy)


def _two_speech_specs():
    """Two speeches with hand-countable labels.

    d1: 2 paragraphs — one policy-labelled, one with NO legacy issue and only
        non-policy topics. d2: 1 paragraph with no topics at all.
    """
    return [
        {
            "doc_name": "d1",
            "year": 1805,
            "speech_type": SOTU,
            "paras": [
                {
                    "legacy": ["War & military"],
                    "topics": ["the War on Terror", "Jobs & Wages"],
                    "pv": "proposal",
                    "words": 120,
                },
                {
                    "legacy": [],
                    "topics": ["Holidays & Tributes"],
                    "pv": "values",
                    "words": 80,
                },
            ],
        },
        {
            "doc_name": "d2",
            "year": 1835,
            "speech_type": INAUGURAL,
            "paras": [{"legacy": [], "topics": [], "pv": "neither", "words": 50}],
        },
    ]


# Thirteen distinct marker counts, written out as LITERAL names rather than read
# off `R.STYLE_MARKERS`. Both the fixture builder in conftest and every other
# expectation in the suite iterate that constant, so a marker deleted from it
# disappears from the input and the assertion in one step; spelling the names out
# here means the panel has to actually carry each one. The values are pairwise
# distinct, and distinct from the stat columns below, so a crossed wiring fails
# on the specific column rather than averaging out.
_DISTINCT_MARKER_COUNTS = {
    "mechanism": 11.0,
    "religiosity": 12.0,
    "nostalgia": 13.0,
    "future": 14.0,
    "us_them": 15.0,
    "opponents": 16.0,
    "hype": 17.0,
    "doom": 18.0,
    "boosters": 19.0,
    "hedges": 20.0,
    "superlatives": 21.0,
    "nrc_hope": 22.0,
    "nrc_fear": 23.0,
}


def _multi_issue_specs():
    """One speech whose first paragraph fires THREE legacy issues.

    Every other fixture in the suite gives a paragraph at most one issue, which
    leaves "how many issues fired" and "did any issue fire" indistinguishable.
    """
    return [
        {
            "doc_name": "d1",
            "year": 1805,
            "speech_type": SOTU,
            "paras": [
                {
                    "legacy": ["War & military", "Trade & tariffs", "Agriculture"],
                    "topics": ["Jobs & Wages"],
                    "pv": "proposal",
                    "words": 100,
                },
                {"legacy": ["War & military"], "topics": [], "pv": "neither",
                 "words": 100},
                {"legacy": [], "topics": [], "pv": "neither", "words": 100},
            ],
        },
    ]


# =========================================================================== #
class TestBuildParagraphFrame:
    def test_derives_label_counts_and_zero_flags_per_paragraph(
        self, register_corpus, register_taxonomy
    ):
        """Every downstream measure is a sum of these five per-paragraph
        quantities, so each is pinned on a frame whose answer can be counted by
        eye."""
        c = register_corpus(_two_speech_specs())

        para = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        ).set_index(["doc_name", "para_idx"])

        assert para.loc[("d1", 0), "n_legacy"] == 1
        assert para.loc[("d1", 0), "n_llm"] == 2
        assert bool(para.loc[("d1", 0), "legacy_zero"]) is False
        assert bool(para.loc[("d1", 1), "legacy_zero"]) is True
        assert para.loc[("d2", 0), "n_llm"] == 0
        assert bool(para.loc[("d2", 0), "llm_zero"]) is True
        # `llm_zero` is "no topics at all", not "few topics": the single-topic
        # paragraph is NOT part of the 1.1% zero-topic residue.
        assert para.loc[("d1", 1), "n_llm"] == 1
        assert bool(para.loc[("d1", 1), "llm_zero"]) is False

    def test_n_llm_counts_assignments_so_a_repeated_topic_counts_twice(
        self, register_corpus, register_taxonomy
    ):
        """`n_llm` sums into `labels_per_paragraph`, and the entropy matrices
        count the same way, so the unit is the label ASSIGNMENT. Deduplicating a
        paragraph's labels would quietly change the density measure's numerator
        and the entropy denominator together. `resolve_labels` is already pinned
        to preserve duplicates; this is the same contract one layer up, where the
        count is actually taken."""
        specs = _two_speech_specs()
        specs[0]["paras"][0]["topics"] = [
            "Jobs & Wages", "Jobs & Wages", "the War on Terror"
        ]
        c = register_corpus(specs)

        para = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        ).set_index(["doc_name", "para_idx"])

        assert para.loc[("d1", 0), "n_llm"] == 3
        assert para.loc[("d1", 0), "topics_resolved"] == [
            "Jobs & Wages", "Jobs & Wages", "the War on Terror"
        ]

    def test_n_legacy_counts_how_many_issues_fired_not_whether_any_did(
        self, register_corpus, register_taxonomy
    ):
        """`labels_per_paragraph` on the legacy taxonomy — the "presidents say
        less about each subject" half of the published headline (1.49 -> 0.83) —
        is a sum of this column, so it must be a COUNT and not a flag.

        No other fixture in the suite gives a paragraph more than one legacy
        issue, which leaves `sum(axis=1)` and `any(axis=1)` producing identical
        output everywhere else. Collapsing the two would bite hardest on the
        19th-century departmental catalog paragraphs that touch several issues at
        once — i.e. it would manufacture precisely the fall the module exists to
        test.
        """
        c = register_corpus(_multi_issue_specs())

        para = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        ).set_index(["doc_name", "para_idx"])

        assert para.loc[("d1", 0), "n_legacy"] == 3
        assert para.loc[("d1", 1), "n_legacy"] == 1
        assert para.loc[("d1", 2), "n_legacy"] == 0
        assert bool(para.loc[("d1", 0), "legacy_zero"]) is False
        assert bool(para.loc[("d1", 2), "legacy_zero"]) is True

    def test_all_non_policy_flag_requires_topics_and_all_of_them_non_policy(
        self, register_corpus, register_taxonomy
    ):
        """Three distinct states must not be conflated: a paragraph whose topics
        are all non-policy, one that mixes policy with non-policy, and one with
        no topics at all. The last is the `zero_topic_share` residue and is
        explicitly NOT counted as non-policy."""
        c = register_corpus(_two_speech_specs())

        para = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        ).set_index(["doc_name", "para_idx"])

        assert bool(para.loc[("d1", 1), "llm_all_non_policy"]) is True   # all non-policy
        assert bool(para.loc[("d1", 0), "llm_all_non_policy"]) is False  # mixed
        assert bool(para.loc[("d2", 0), "llm_all_non_policy"]) is False  # no topics
        assert bool(para.loc[("d2", 0), "llm_zero"]) is True

    def test_case_variant_topic_is_canonicalized_onto_the_frame(
        self, register_corpus, register_taxonomy
    ):
        """The casefold repair, observed where it matters: the resolved topic on
        the frame is the canonical spelling, so the level-2 and level-1 rollups
        downstream both find it."""
        specs = _two_speech_specs()
        specs[0]["paras"][0]["topics"] = ["the War On Terror"]
        c = register_corpus(specs)

        para = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        )

        assert para.loc[0, "topics_resolved"] == ["the War on Terror"]
        assert para.loc[0, "n_llm"] == 1

    def test_unresolvable_topic_label_is_fatal(
        self, register_corpus, register_taxonomy
    ):
        """A label outside the taxonomy stops the build. Dropping it instead
        would silently reduce that paragraph's label count and its era's
        entropy."""
        specs = _two_speech_specs()
        specs[0]["paras"][0]["topics"] = ["Space Policy"]
        c = register_corpus(specs)

        with pytest.raises(KeyError, match="does not resolve"):
            R.build_paragraph_frame(
                c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
            )

    def test_result_is_keyed_not_positional(self, register_corpus, register_taxonomy):
        """Shuffling the issue and annotation tables must not change a single
        derived value: the join is on `(doc_name, para_idx)`, so a reordered
        input is self-healing rather than silently mislabelling every row."""
        c = register_corpus(_two_speech_specs())
        in_order = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        )

        shuffled_issues = c.issues.iloc[::-1].reset_index(drop=True)
        shuffled_annotations = c.annotations.iloc[::-1].reset_index(drop=True)
        # Guard against a vacuous test: the reversal must actually change the key
        # sequence, or "reordering is harmless" would be proven on unmoved rows.
        keys = lambda f: list(zip(f["doc_name"], f["para_idx"]))  # noqa: E731
        assert keys(shuffled_issues) != keys(c.issues)
        assert keys(shuffled_annotations) != keys(c.annotations)

        reordered = R.build_paragraph_frame(
            c.paragraphs, shuffled_issues, shuffled_annotations,
            _index(register_taxonomy),
        )

        pd.testing.assert_frame_equal(reordered, in_order)

    def test_duplicate_key_in_issues_raises_merge_error(
        self, register_corpus, register_taxonomy
    ):
        c = register_corpus(_two_speech_specs())
        dup = pd.concat([c.issues, c.issues.iloc[[0]]], ignore_index=True)

        with pytest.raises(pd.errors.MergeError):
            R.build_paragraph_frame(
                c.paragraphs, dup, c.annotations, _index(register_taxonomy)
            )

    def test_paragraph_missing_from_issues_raises_instead_of_shrinking(
        self, register_corpus, register_taxonomy
    ):
        """The silent-drop mode `validate="one_to_one"` cannot catch. Without the
        row-count guard this returns a 2-row frame while the caller still reports
        3 paragraphs."""
        c = register_corpus(_two_speech_specs())
        short = c.issues.iloc[1:].reset_index(drop=True)

        with pytest.raises(ValueError, match="row count"):
            R.build_paragraph_frame(
                c.paragraphs, short, c.annotations, _index(register_taxonomy)
            )

    def test_extra_key_in_issues_raises_even_though_no_paragraph_is_lost(
        self, register_corpus, register_taxonomy
    ):
        """The diverging-key-set guard is symmetric: an issue row with no
        paragraph leaves the merged length equal to `len(paragraphs)`, so a guard
        that only compared against the LEFT frame would pass. It must compare
        against every input."""
        c = register_corpus(_two_speech_specs())
        extra = pd.concat(
            [c.issues, c.issues.iloc[[0]].assign(para_idx=99)], ignore_index=True
        )

        with pytest.raises(ValueError, match="row count"):
            R.build_paragraph_frame(
                c.paragraphs, extra, c.annotations, _index(register_taxonomy)
            )

    def test_paragraph_missing_from_annotations_raises(
        self, register_corpus, register_taxonomy
    ):
        """The second merge carries its own guard — a paragraph the annotation
        run never returned must stop the build, not vanish from the denominator."""
        c = register_corpus(_two_speech_specs())
        short = c.annotations.iloc[1:].reset_index(drop=True)

        with pytest.raises(ValueError, match="row count"):
            R.build_paragraph_frame(
                c.paragraphs, c.issues, short, _index(register_taxonomy)
            )

    def test_healthy_key_sets_produce_one_row_per_paragraph(
        self, register_corpus, register_taxonomy
    ):
        """The guard is silent on the happy path."""
        c = register_corpus(_two_speech_specs())

        para = R.build_paragraph_frame(
            c.paragraphs, c.issues, c.annotations, _index(register_taxonomy)
        )

        assert len(para) == len(c.paragraphs) == 3
        assert not para.duplicated(["doc_name", "para_idx"]).any()


# =========================================================================== #
class TestRequireFullMerge:
    def test_passes_when_every_input_kept_all_its_rows(self):
        R._require_full_merge(10, {"a": 10, "b": 10}, "stage")  # must not raise

    def test_names_the_stage_and_the_counts_it_saw(self):
        """The message has to be diagnosable from a batch log, so it carries the
        stage and every input's count."""
        with pytest.raises(ValueError) as excinfo:
            R._require_full_merge(9, {"paragraphs": 10, "issues": 9}, "paras x issues")

        message = str(excinfo.value)
        assert "paras x issues" in message
        assert "merged=9" in message
        assert "paragraphs=10" in message

    def test_raises_when_only_one_input_diverges(self):
        with pytest.raises(ValueError, match="row count"):
            R._require_full_merge(10, {"a": 10, "b": 11}, "stage")


# =========================================================================== #
class TestBuildSpeechPanel:
    def _panel(self, corpus, register_taxonomy):
        index = _index(register_taxonomy)
        para = R.build_paragraph_frame(
            corpus.paragraphs, corpus.issues, corpus.annotations, index
        )
        return R.build_speech_panel(
            para,
            corpus.speeches,
            corpus.speech_annotations,
            corpus.stats,
            corpus.markers,
            index,
        )

    def test_rolls_paragraph_sums_up_to_one_row_per_speech(
        self, register_corpus, register_taxonomy
    ):
        """The panel's scalars are the sufficient statistics every measure and
        every bootstrap replicate is built from, so each is checked against a
        count taken by hand off the fixture."""
        panel = self._panel(register_corpus(_two_speech_specs()), register_taxonomy)
        scalars = panel.scalars.set_index(panel.speeches["doc_name"])

        assert panel.speeches["doc_name"].tolist() == ["d1", "d2"]
        assert scalars.loc["d1", "n_paragraphs"] == 2
        assert scalars.loc["d1", "para_words"] == 200      # 120 + 80
        assert scalars.loc["d1", "llm_labels"] == 3        # 2 topics + 1 topic
        assert scalars.loc["d1", "legacy_labels"] == 1
        assert scalars.loc["d1", "legacy_zero_paras"] == 1
        assert scalars.loc["d1", "llm_non_policy_paras"] == 1
        assert scalars.loc["d2", "llm_zero_paras"] == 1
        assert scalars.loc["d1", "llm_zero_paras"] == 0    # both d1 paragraphs got topics
        assert scalars.loc["d1", "pv_proposal"] == 1
        assert scalars.loc["d1", "pv_values"] == 1

    def test_proposal_values_levels_count_the_paragraphs_at_each_level(
        self, register_corpus, register_taxonomy
    ):
        """`proposal_share`, `values_share` and `proposal_vs_values` are all sums
        of these four flags, so each must count the paragraphs AT its level.

        The two-paragraph fixture used everywhere else splits one proposal
        against one values paragraph, which makes `== level` and `!= level`
        produce the identical count for every level — an inverted flag would
        swap the two published shares outright and no existing assertion would
        move. An asymmetric spread separates them, and the partition check below
        makes the claim total: the four flags must account for every paragraph
        exactly once."""
        levels = ("proposal", "values", "mixed", "neither")
        specs = [{
            "doc_name": "d1", "year": 1805, "speech_type": SOTU,
            "paras": [
                {"legacy": [], "topics": [], "pv": pv, "words": 100}
                for pv in ("proposal", "values", "values", "mixed",
                           "neither", "neither", "neither")
            ],
        }]
        panel = self._panel(register_corpus(specs), register_taxonomy)
        scalars = panel.scalars.iloc[0]

        assert scalars["n_paragraphs"] == 7
        assert scalars["pv_proposal"] == 1
        assert scalars["pv_values"] == 2
        assert scalars["pv_mixed"] == 1
        assert scalars["pv_neither"] == 3
        assert sum(scalars[f"pv_{level}"] for level in levels) == 7

    def test_case_variant_labels_reach_the_matrices_under_their_canonical_name(
        self, register_corpus, register_taxonomy
    ):
        """The repair has to survive all the way to the COUNT MATRICES, not just
        onto the paragraph frame.

        Both rollups must read the resolved column. Counting the raw labels
        instead leaves a variant with no column to land in, `reindex` fills it
        with zero, and the assignment vanishes — on the real corpus that is 111
        assignments, 84 of them on one modern-era policy topic, biasing modern
        POLICY attention downward and so flattering the "more non-policy speech"
        half of the headline. The level-1 rollup fails louder (a KeyError on the
        unknown parent), but loud or silent it is the same bug, and the existing
        case-variant test stops at `build_paragraph_frame`.
        """
        specs = _two_speech_specs()
        specs[0]["paras"][0]["topics"] = ["the War On Terror"]   # variant casing
        specs[0]["paras"][1]["topics"] = ["THE WAR ON TERROR"]   # and another
        panel = self._panel(register_corpus(specs), register_taxonomy)
        level2 = pd.DataFrame(
            panel.topic_counts["llm_level2"], columns=list(panel.topic_names["llm_level2"])
        )
        level1 = pd.DataFrame(
            panel.topic_counts["llm_level1"], columns=list(panel.topic_names["llm_level1"])
        )

        assert level2.loc[0, "the War on Terror"] == 2
        assert level2.loc[0].sum() == 2      # nothing was dropped on the floor
        assert level1.loc[0, "Security"] == 2
        assert level1.loc[0].sum() == 2
        assert panel.scalars.loc[0, "llm_labels"] == 2

    def test_era_is_the_thirty_year_band_floor_of_the_speech_year(
        self, register_corpus, register_taxonomy
    ):
        """Eras are the grouping variable of the whole analysis. A year one below
        a band multiple stays in the lower band; the multiple opens the next."""
        specs = _two_speech_specs()
        specs[0]["year"] = 1799
        specs[1]["year"] = 1800
        for spec in specs:
            for para in spec["paras"]:
                para["year"] = spec["year"]
        panel = self._panel(register_corpus(specs), register_taxonomy)

        assert panel.speeches.set_index("doc_name")["era"].to_dict() == {
            "d1": 1770,
            "d2": 1800,
        }

    def test_fk_grade_is_carried_as_a_token_weighted_product(
        self, register_corpus, register_taxonomy
    ):
        """Averaging per-speech averages would let a 200-word proclamation
        outweigh a 30,000-word annual message. The panel stores fk x tokens so
        aggregation divides it back out by total tokens."""
        specs = _two_speech_specs()
        specs[0].update(fk_grade=12.0, n_tokens=1000.0)
        specs[1].update(fk_grade=4.0, n_tokens=100.0)
        panel = self._panel(register_corpus(specs), register_taxonomy)
        scalars = panel.scalars.set_index(panel.speeches["doc_name"])

        assert scalars.loc["d1", "fk_x_tokens"] == 12_000.0
        assert scalars.loc["d2", "fk_x_tokens"] == 400.0
        pooled = scalars["fk_x_tokens"].sum() / scalars["n_tokens"].sum()
        assert pooled == pytest.approx(12_400 / 1_100)   # NOT (12 + 4) / 2

    def test_every_speech_level_statistic_is_copied_onto_its_own_scalar_column(
        self, register_corpus, register_taxonomy
    ):
        """The eighteen columns copied straight off `speech_stats.parquet` and
        `speech_markers.parquet` — four scalars (`n_tokens`, `n_sents`,
        `i_count`, `we_count`), the marker word count, and the thirteen style
        markers — with pairwise distinct values so a crossed wiring fails on the
        specific column instead of averaging out.

        Every one of them denominates or numerates a published series
        (`self_reference`, `words_per_sentence`, and one per-10k-word rate per
        marker), and nothing else in the suite reads them off the panel: the
        measure tests hand `measures_from_sums` a sums frame directly, so the
        copy itself was untested. Three concrete failures this catches:
        markers arriving as zeros, `n_words` (the marker denominator, from the
        regex table) being fed from spaCy's `n_tokens`, and the `i`/`we` pair
        arriving swapped — which inverts `self_reference` without changing its
        range.
        """
        specs = _two_speech_specs()
        specs[0].update(
            n_tokens=1_000.0, n_sents=40.0, i_count=7.0, we_count=3.0,
            n_words=900.0, **_DISTINCT_MARKER_COUNTS,
        )
        panel = self._panel(register_corpus(specs), register_taxonomy)
        d1 = panel.scalars.set_index(panel.speeches["doc_name"]).loc["d1"]

        assert d1["n_tokens"] == 1_000.0
        assert d1["n_sents"] == 40.0
        assert d1["i_count"] == 7.0     # `self_reference` is i / (i + we) ...
        assert d1["we_count"] == 3.0    # ... so the pair must not be swapped
        assert d1["n_words"] == 900.0   # the marker denominator, NOT n_tokens
        for marker, count in _DISTINCT_MARKER_COUNTS.items():
            assert d1[marker] == count, marker

    def test_legacy_matrix_counts_every_issue_a_paragraph_fires(
        self, register_corpus, register_taxonomy
    ):
        """The legacy15 count matrix is the input to `effective_topics` on the
        anchored taxonomy — the breadth series the headline is actually about. A
        paragraph firing three issues must put a 1 in each of three columns, so
        that a speech touching three subjects scores as broader than one touching
        the same subject three times."""
        panel = self._panel(register_corpus(_multi_issue_specs()), register_taxonomy)
        legacy = pd.DataFrame(
            panel.topic_counts["legacy15"], columns=list(panel.topic_names["legacy15"])
        )

        assert panel.scalars.loc[0, "legacy_labels"] == 4      # 3 + 1 + 0
        assert legacy.loc[0, "War & military"] == 2
        assert legacy.loc[0, "Trade & tariffs"] == 1
        assert legacy.loc[0, "Agriculture"] == 1
        assert legacy.loc[0].sum() == 4

    def test_topic_matrices_count_assignments_on_all_three_taxonomies(
        self, register_corpus, register_taxonomy
    ):
        """Rows are speeches, columns are topics, entries are label ASSIGNMENTS
        (a paragraph with k topics contributes 1 to each). The level-1 matrix is
        the same assignments rolled up through each topic's parent domain, which
        is what makes the two LLM entropies comparable as trends."""
        panel = self._panel(register_corpus(_two_speech_specs()), register_taxonomy)
        level2 = pd.DataFrame(
            panel.topic_counts["llm_level2"], columns=list(panel.topic_names["llm_level2"])
        )
        level1 = pd.DataFrame(
            panel.topic_counts["llm_level1"], columns=list(panel.topic_names["llm_level1"])
        )
        legacy = pd.DataFrame(
            panel.topic_counts["legacy15"], columns=list(panel.topic_names["legacy15"])
        )

        assert level2.loc[0, "the War on Terror"] == 1
        assert level2.loc[0, "Jobs & Wages"] == 1
        assert level2.loc[0, "Holidays & Tributes"] == 1
        assert level2.loc[0].sum() == 3
        assert level1.loc[0, "Security"] == 1
        assert level1.loc[0, "Economy"] == 1
        assert level1.loc[0, "Ceremonial"] == 1
        assert legacy.loc[0, "War & military"] == 1
        assert level2.loc[1].sum() == 0        # d2 has no topics

    def test_level1_rollup_sums_sibling_topics_into_their_domain(
        self, register_corpus, register_taxonomy
    ):
        """Two topics under one domain must ADD in the level-1 matrix — a rollup
        that deduplicated would flatten the level-1 entropy series."""
        specs = _two_speech_specs()
        specs[0]["paras"][0]["topics"] = ["Jobs & Wages", "Trade & Tariffs"]
        specs[0]["paras"][1]["topics"] = ["Jobs & Wages"]
        panel = self._panel(register_corpus(specs), register_taxonomy)
        level1 = pd.DataFrame(
            panel.topic_counts["llm_level1"], columns=list(panel.topic_names["llm_level1"])
        )

        assert level1.loc[0, "Economy"] == 3
        assert level1.loc[0].sum() == 3

    def test_a_corpus_with_no_topics_at_all_yields_zero_matrices_not_a_crash(
        self, register_corpus, register_taxonomy
    ):
        """The 1.1% zero-topic residue taken to its limit. An era or cell where
        the labeler returned nothing must produce an all-zero count matrix of the
        right SHAPE — a ragged or empty one would break the `weights @ matrix`
        product in the bootstrap, and a dropped column would silently renumber
        the taxonomy."""
        specs = _two_speech_specs()
        for spec in specs:
            for para in spec["paras"]:
                para["topics"] = []
        panel = self._panel(register_corpus(specs), register_taxonomy)
        index = _index(register_taxonomy)

        assert panel.topic_counts["llm_level2"].shape == (2, len(index.level2))
        assert panel.topic_counts["llm_level1"].shape == (2, len(index.level1))
        assert not panel.topic_counts["llm_level2"].any()
        assert not panel.topic_counts["llm_level1"].any()
        # The legacy issues are unaffected — the two labelers are independent.
        assert panel.scalars["legacy_labels"].sum() == 1

    def test_topic_matrix_columns_follow_the_taxonomy_order(
        self, register_corpus, register_taxonomy
    ):
        """Structural: the matrices are positional downstream (`weights @ matrix`),
        so their column order must be the index's, and every taxonomy topic must
        get a column even when unused."""
        panel = self._panel(register_corpus(_two_speech_specs()), register_taxonomy)
        index = _index(register_taxonomy)

        assert panel.topic_names["llm_level2"] == index.level2
        assert panel.topic_names["llm_level1"] == index.level1
        assert panel.topic_names["legacy15"] == tuple(R.LEGACY_ISSUES)
        assert panel.topic_counts["llm_level2"].shape == (2, len(index.level2))

    def test_speech_rows_and_matrix_rows_share_one_order(
        self, register_corpus, register_taxonomy
    ):
        """The panel is a positional join between `speeches`, `scalars` and the
        matrices. Feeding the speeches in a different order must not change any
        speech's own numbers — otherwise every measure would be attributed to the
        wrong era."""
        specs = _two_speech_specs()
        c = register_corpus(specs)
        forward = self._panel(c, register_taxonomy)

        reversed_c = register_corpus(list(reversed(specs)))
        assert reversed_c.speeches["doc_name"].tolist() == ["d2", "d1"]
        backward = self._panel(reversed_c, register_taxonomy)

        assert backward.speeches["doc_name"].tolist() == ["d1", "d2"]
        pd.testing.assert_frame_equal(backward.scalars, forward.scalars)
        assert np.array_equal(
            backward.topic_counts["llm_level2"], forward.topic_counts["llm_level2"]
        )

    def test_panel_row_order_is_the_total_doc_name_order(
        self, register_corpus, register_taxonomy
    ):
        """The panel's row order is the positional key that `scalars`, the three
        count matrices and every cell plan are all indexed by, so it has to be a
        TOTAL order fixed by the corpus content. `doc_name` is unique; sorting on
        `year` (or any other non-unique column) leaves ties broken by whatever
        order the input table happened to arrive in, which would make the panel —
        and the seeded bootstrap draws indexed off it — a function of upstream
        row order rather than of the corpus.

        Both speeches share a year here, and their names sort the opposite way
        from their input order, so a year sort and a doc_name sort disagree. The
        `test_speech_rows_and_matrix_rows_share_one_order` fixture cannot see
        this: its two speeches are 30 years apart, so both sorts agree.
        """
        specs = _two_speech_specs()
        specs[0]["doc_name"] = "zulu"
        specs[1]["doc_name"] = "alpha"
        specs[1]["year"] = specs[0]["year"]        # tie the sort key
        panel = self._panel(register_corpus(specs), register_taxonomy)

        assert panel.speeches["doc_name"].tolist() == ["alpha", "zulu"]
        # ...and the scalars followed the same order: alpha is the 1-paragraph
        # speech, zulu the 2-paragraph one.
        assert panel.scalars["n_paragraphs"].tolist() == [1.0, 2.0]

    def test_speech_with_paragraphs_but_no_speech_row_raises(
        self, register_corpus, register_taxonomy
    ):
        """A paragraph whose speech is unknown has no year and therefore no era —
        it would be dropped from every series while still being counted in the
        corpus total."""
        c = register_corpus(_two_speech_specs())
        index = _index(register_taxonomy)
        para = R.build_paragraph_frame(c.paragraphs, c.issues, c.annotations, index)
        speeches = c.speeches[c.speeches["doc_name"] != "d2"]

        with pytest.raises(ValueError, match="no speech-level row"):
            R.build_speech_panel(
                para, speeches,
                c.speech_annotations[c.speech_annotations["doc_name"] != "d2"],
                c.stats[c.stats["doc_name"] != "d2"],
                c.markers[c.markers["doc_name"] != "d2"],
                index,
            )

    def test_speech_with_no_paragraphs_raises(
        self, register_corpus, register_taxonomy
    ):
        """A speech contributing zero paragraphs would make every per-paragraph
        ratio undefined for it and silently zero-weight it in the bootstrap."""
        c = register_corpus(_two_speech_specs())
        index = _index(register_taxonomy)
        para = R.build_paragraph_frame(c.paragraphs, c.issues, c.annotations, index)
        para = para[para["doc_name"] != "d2"]

        with pytest.raises(ValueError, match="no paragraphs"):
            R.build_speech_panel(
                para, c.speeches, c.speech_annotations, c.stats, c.markers, index
            )

    @pytest.mark.parametrize(
        "table, stage",
        [
            ("speech_annotations", "speech_annotations"),
            ("stats", "speech_stats"),
            ("markers", "speech_markers"),
        ],
    )
    def test_each_speech_level_merge_guards_its_own_row_count(
        self, register_corpus, register_taxonomy, table, stage
    ):
        """Three separate merges through ONE guarded loop. The loop is the point:
        a table added to it cannot arrive without its row-count check. But a loop
        is also a single call site, so a guard that only fired for the first
        table would pass a test that exercised the merges transitively — each
        table is therefore shrunk on its own and asserted to name itself in the
        error."""
        c = register_corpus(_two_speech_specs())
        index = _index(register_taxonomy)
        para = R.build_paragraph_frame(c.paragraphs, c.issues, c.annotations, index)
        frames = {
            "speech_annotations": c.speech_annotations,
            "stats": c.stats,
            "markers": c.markers,
        }
        frames[table] = frames[table].iloc[1:].reset_index(drop=True)

        with pytest.raises(ValueError, match=stage):
            R.build_speech_panel(
                para, c.speeches, frames["speech_annotations"], frames["stats"],
                frames["markers"], index,
            )

    def test_duplicate_doc_name_in_speech_annotations_raises_merge_error(
        self, register_corpus, register_taxonomy
    ):
        c = register_corpus(_two_speech_specs())
        index = _index(register_taxonomy)
        para = R.build_paragraph_frame(c.paragraphs, c.issues, c.annotations, index)
        dup = pd.concat(
            [c.speech_annotations, c.speech_annotations.iloc[[0]]], ignore_index=True
        )

        with pytest.raises(pd.errors.MergeError):
            R.build_speech_panel(para, c.speeches, dup, c.stats, c.markers, index)


# =========================================================================== #
class TestParagraphSpeechYearAgreement:
    """The paragraph-year / speech-year cross-check.

    Era banding here comes from `speeches.parquet`, while every other era-keyed
    artifact in the repo bands off the per-paragraph year on
    `paragraph_issues.parquet`. They agree on today's corpus and are supposed to
    — the paragraph year is derived from the speech — but a drift would make this
    module's era series silently incomparable with every other one, which is
    invisible in the output. Hence a real guard rather than a docstring promise.
    """

    def _para_and_index(self, corpus, register_taxonomy):
        index = _index(register_taxonomy)
        return (
            R.build_paragraph_frame(
                corpus.paragraphs, corpus.issues, corpus.annotations, index
            ),
            index,
        )

    def test_paragraph_frame_carries_the_paragraph_level_year(
        self, register_corpus, register_taxonomy
    ):
        """The column has to survive the merges for the check to have anything to
        compare — this is the seam a `year`-less select would silently remove."""
        c = register_corpus(_two_speech_specs())
        para, _ = self._para_and_index(c, register_taxonomy)

        assert "year" in para.columns
        assert para.set_index(["doc_name", "para_idx"]).loc[("d1", 0), "year"] == 1805

    def test_agreeing_years_pass(self, register_corpus, register_taxonomy):
        c = register_corpus(_two_speech_specs())
        para, index = self._para_and_index(c, register_taxonomy)

        panel = R.build_speech_panel(
            para, c.speeches, c.speech_annotations, c.stats, c.markers, index
        )

        assert len(panel.speeches) == 2

    def test_disagreeing_year_raises(self, register_corpus, register_taxonomy):
        """A paragraph-level year that contradicts the speech-level year: the two
        artifacts would band this speech into different eras."""
        specs = _two_speech_specs()
        specs[0]["paras"][0]["year"] = 1795   # speech-level year is 1805
        specs[0]["paras"][1]["year"] = 1795
        c = register_corpus(specs)
        para, index = self._para_and_index(c, register_taxonomy)

        with pytest.raises(ValueError, match="disagree about their year"):
            R.build_speech_panel(
                para, c.speeches, c.speech_annotations, c.stats, c.markers, index
            )

    def test_two_paragraph_years_within_one_speech_raises(
        self, register_corpus, register_taxonomy
    ):
        """The other shape the same corruption takes: one speech whose paragraphs
        carry two different years. Matching the speech year on ONE of them is not
        agreement."""
        specs = _two_speech_specs()
        specs[0]["paras"][1]["year"] = 1806   # sibling paragraph stays 1805
        c = register_corpus(specs)
        para, index = self._para_and_index(c, register_taxonomy)

        with pytest.raises(ValueError, match="disagree about their year"):
            R.build_speech_panel(
                para, c.speeches, c.speech_annotations, c.stats, c.markers, index
            )

    def test_error_names_the_offending_speech(
        self, register_corpus, register_taxonomy
    ):
        specs = _two_speech_specs()
        specs[1]["paras"][0]["year"] = 1999
        c = register_corpus(specs)
        para, index = self._para_and_index(c, register_taxonomy)

        with pytest.raises(ValueError) as excinfo:
            R.build_speech_panel(
                para, c.speeches, c.speech_annotations, c.stats, c.markers, index
            )

        assert "'d2'" in str(excinfo.value)


# =========================================================================== #
class TestLoadInputs:
    """`load_inputs` is the only function in the module that touches disk. It is
    driven here over tmp_path parquets so the test stays hermetic and the real
    frozen artifacts are never read or written."""

    def test_reads_every_table_from_its_own_path_constant(
        self, tmp_path, monkeypatch, register_corpus, register_taxonomy
    ):
        c = register_corpus(_two_speech_specs())
        paths = {
            "PARAGRAPHS_PATH": ("paragraphs", c.paragraphs),
            "PARAGRAPH_ISSUES_PATH": ("issues", c.issues),
            "PARAGRAPH_ANNOTATIONS_PATH": ("annotations", c.annotations),
            "SPEECHES_PATH": ("speeches", c.speeches),
            "SPEECH_ANNOTATIONS_PATH": ("speech_annotations", c.speech_annotations),
            "STATS_PATH": ("stats", c.stats),
            "MARKERS_PATH": ("markers", c.markers),
        }
        for constant, (name, frame) in paths.items():
            path = tmp_path / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            monkeypatch.setattr(R, constant, path)
        taxonomy_path = tmp_path / "taxonomy_v1.json"
        taxonomy_path.write_text(json.dumps(register_taxonomy))
        monkeypatch.setattr(R, "TAXONOMY_PATH", taxonomy_path)

        inputs = R.load_inputs()

        # Each table landed on the attribute fed by its OWN path constant — a
        # crossed wiring (e.g. reading speeches into `stats`) is what this
        # catches, since the frames have distinguishable shapes.
        for _, (name, frame) in paths.items():
            pd.testing.assert_frame_equal(getattr(inputs, name), frame)
        assert inputs.taxonomy.level2 == R.build_taxonomy_index(
            register_taxonomy
        ).level2

    def test_output_feeds_the_pure_pipeline_unchanged(
        self, tmp_path, monkeypatch, register_corpus, register_taxonomy
    ):
        """Contract check: whatever `load_inputs` returns must be exactly what
        `build_paragraph_frame` / `build_speech_panel` expect, including the
        taxonomy index it builds for them."""
        c = register_corpus(_two_speech_specs())
        for constant, (name, frame) in {
            "PARAGRAPHS_PATH": ("paragraphs", c.paragraphs),
            "PARAGRAPH_ISSUES_PATH": ("issues", c.issues),
            "PARAGRAPH_ANNOTATIONS_PATH": ("annotations", c.annotations),
            "SPEECHES_PATH": ("speeches", c.speeches),
            "SPEECH_ANNOTATIONS_PATH": ("speech_annotations", c.speech_annotations),
            "STATS_PATH": ("stats", c.stats),
            "MARKERS_PATH": ("markers", c.markers),
        }.items():
            path = tmp_path / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            monkeypatch.setattr(R, constant, path)
        taxonomy_path = tmp_path / "taxonomy_v1.json"
        taxonomy_path.write_text(json.dumps(register_taxonomy))
        monkeypatch.setattr(R, "TAXONOMY_PATH", taxonomy_path)

        inputs = R.load_inputs()
        para = R.build_paragraph_frame(
            inputs.paragraphs, inputs.issues, inputs.annotations, inputs.taxonomy
        )
        panel = R.build_speech_panel(
            para, inputs.speeches, inputs.speech_annotations, inputs.stats,
            inputs.markers, inputs.taxonomy,
        )

        assert len(para) == 3
        assert panel.speeches["doc_name"].tolist() == ["d1", "d2"]
