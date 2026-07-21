"""``combat.load_frame`` and the tables that hang off it.

This module's output is a *published research claim*, so these tests are aimed
at the failure modes that would corrupt a number without raising: a keyed merge
that silently drops rows, an entity table that fans a paragraph out into six,
an era boundary that moves, an exemplar that quotes the wrong paragraph.

Everything here runs on hand-authored frames of a dozen rows. ``load_frame``
takes all five of its inputs as injected DataFrames precisely so the real
36,229-row corpus never has to be loaded (CLAUDE.md).

Background on why the merge guards get this much attention: this repo already
shipped a positional-alignment bug that could silently mislabel every paragraph
(see ``test_keyed_merge.py``), and ``validate="one_to_one"`` alone does NOT
close it — it catches duplicate keys but an inner join still drops rows when the
two key *sets* merely diverge. ``combat._require_full_merge`` is the second half
of that guard and each of its call sites is exercised below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from presidential_profiles import combat as C
from presidential_profiles.trends import ERAS

SOTU = C.SOTU_TYPE


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    """The single row matching every ``column=value``. Asserts uniqueness, so a
    filter that silently matched two rows (or none) fails here rather than
    quietly asserting against the wrong one."""
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hits = frame[mask]
    assert len(hits) == 1, f"{where} matched {len(hits)} rows, expected 1"
    return hits.iloc[0]


# --------------------------------------------------------------------------- #
# era assignment
# --------------------------------------------------------------------------- #
class TestEraAssignment:
    def test_era_axis_is_trends_eras_unchanged(self):
        """The task forbids a competing periodization — data-driven eras belong
        to `era-atlas`, which depends on this task. Anchored as literals rather
        than derived from ERAS, so a silent edit to trends.ERAS is caught here
        instead of quietly re-cutting every published rate."""
        assert C.ERA_ORDER == [label for label, _, _ in ERAS]
        assert len(C.ERA_ORDER) == 9
        assert C.ERA_ORDER[0] == "The founding"
        assert C.ERA_ORDER[-1] == "The present era"
        assert C.ERA_BOUNDS["The founding"] == (1789, 1815)
        assert C.ERA_BOUNDS["Civil War & Reconstruction"] == (1850, 1877)
        assert C.ERA_BOUNDS["War & New Deal"] == (1933, 1945)
        assert C.ERA_BOUNDS["The present era"] == (2017, 2026)

    def test_every_era_band_is_pinned_as_a_literal(self):
        """All NINE bands, not a sample of four.

        MUTATION-CHECK FINDING: the spot-check above anchors only the founding,
        Civil War, War & New Deal and present-era bounds, so a *consistent* shift
        of an unpinned interior boundary — e.g. Gilded Age 1878-1900/Progressives
        1901-1932 becoming 1878-1901/1902-1932 — leaves no overlap, no gap, and
        no failing test, while silently re-cutting two published eras and moving
        every rate, CI and ratio computed for them. The era axis is the reporting
        contract with `era-atlas`, so every boundary is anchored here as a
        literal, decoupled from `trends.ERAS` itself."""
        assert ERAS == [
            ("The founding", 1789, 1815),
            ("Expansion", 1816, 1849),
            ("Civil War & Reconstruction", 1850, 1877),
            ("The Gilded Age", 1878, 1900),
            ("Progressives & Depression", 1901, 1932),
            ("War & New Deal", 1933, 1945),
            ("The Cold War", 1946, 1988),
            ("Post-Cold War", 1989, 2016),
            ("The present era", 2017, 2026),
        ]

    def test_every_corpus_year_falls_in_exactly_one_era(self):
        """Not "at least one" — exactly one. Overlapping bands would double-count
        a year into two eras and no other assertion in the module would notice."""
        for year in range(1789, 2027):
            hits = [label for label, lo, hi in ERAS if lo <= year <= hi]
            assert len(hits) == 1, f"{year} matched {hits}"
            assert C.era_of(year) == hits[0]

    @pytest.mark.parametrize("label,lo,hi", ERAS, ids=[e[0] for e in ERAS])
    def test_era_of_claims_both_of_its_own_boundary_years(self, label, lo, hi):
        assert C.era_of(lo) == label
        assert C.era_of(hi) == label

    @pytest.mark.parametrize("year", [1700, 1788, 2027, 3000])
    def test_era_of_returns_none_outside_every_band(self, year):
        assert C.era_of(year) is None

    def test_load_frame_names_the_offending_years_when_one_is_out_of_range(
        self, combat_inputs, combat_paras
    ):
        """A year outside every band must stop the build and say which year —
        silently dropping it would shrink a rate's denominator invisibly."""
        inputs = combat_inputs(
            [
                {"doc": "ok", "year": 2020, "paras": combat_paras(2, 1)},
                {"doc": "bad", "year": 1750, "paras": combat_paras(2, 1)},
            ]
        )
        with pytest.raises(ValueError, match=r"outside every trends\.ERAS band.*1750"):
            C.load_frame(**inputs)


# --------------------------------------------------------------------------- #
# the row-completeness guard itself
# --------------------------------------------------------------------------- #
class TestRequireFullMerge:
    def test_accepts_a_merge_that_kept_every_input_row(self):
        assert C._require_full_merge(5, {"left": 5, "right": 5}, "stage") is None

    def test_raises_with_the_stage_and_every_input_count(self):
        with pytest.raises(ValueError) as excinfo:
            C._require_full_merge(4, {"left": 5, "right": 4}, "paragraphs x flags")
        msg = str(excinfo.value)
        assert "paragraphs x flags" in msg
        assert "merged=4" in msg and "left=5" in msg and "right=4" in msg
        assert "key sets diverge" in msg

    def test_raises_when_the_merge_grew_not_only_when_it_shrank(self):
        """A fan-out is as wrong as a drop; the guard is an equality check, not
        a floor."""
        with pytest.raises(ValueError):
            C._require_full_merge(7, {"left": 5, "right": 5}, "stage")


# --------------------------------------------------------------------------- #
# load_frame: the keyed merges
# --------------------------------------------------------------------------- #
class TestLoadFrameKeyedMerges:
    def _inputs(self, combat_inputs):
        return combat_inputs(
            [
                {
                    "doc": "doc-a",
                    "year": 1860,
                    "paras": [
                        {"text": "calm gardens"},
                        {"text": "the enemy", "party_attack": True, "enemy_naming": True},
                    ],
                },
                {"doc": "doc-b", "year": 2020, "paras": [{"text": "calm rivers"}]},
            ]
        )

    def test_each_paragraphs_flags_land_on_its_own_text(self, combat_inputs):
        df = C.load_frame(**self._inputs(combat_inputs))
        by_text = df.set_index("text")["party_attack"]
        assert bool(by_text["the enemy"]) is True
        assert bool(by_text["calm gardens"]) is False
        assert bool(by_text["calm rivers"]) is False

    def test_shuffling_every_input_frame_changes_nothing(self, combat_inputs):
        """The keyed merge must be self-healing under reordering — the exact
        corruption the positional-alignment bug could produce. The shuffle is
        asserted to have actually reordered something first, so the invariance
        claim cannot pass vacuously."""
        inputs = self._inputs(combat_inputs)
        in_order = C.load_frame(**inputs)

        shuffled = {
            k: v.sample(frac=1, random_state=i).reset_index(drop=True)
            for i, (k, v) in enumerate(inputs.items())
        }
        reordered_any = any(
            not shuffled[k].equals(inputs[k]) for k in inputs if len(inputs[k]) > 1
        )
        assert reordered_any, "fixture too small to reorder — the test would be vacuous"

        out = C.load_frame(**shuffled)
        assert out.set_index(["doc_name", "para_idx"])["party_attack"].to_dict() == (
            in_order.set_index(["doc_name", "para_idx"])["party_attack"].to_dict()
        )
        assert len(out) == len(in_order)

    def test_shuffling_the_inputs_keeps_EVERY_cross_frame_column_on_its_own_key(
        self, combat_inputs
    ):
        """MUTATION-CHECK FINDING: the shuffle test above cannot see the bug it
        names.

        It asserts only the `(doc_name, para_idx) -> party_attack` mapping — and
        all three of those columns come from the SAME input frame
        (`annotations`), so their mutual alignment survives any join, correct or
        not. Re-attaching `text` positionally after the keyed merge
        (`df["text"] = paras["text"].to_numpy()`), or `president` likewise,
        passed the entire suite. That is precisely the positional-alignment
        corruption CLAUDE.md says this repo already shipped once, and here it
        would put the wrong quotation beside the wrong president in the
        exemplars table — the report's "receipts".

        So this checks every column that crosses a frame boundary: text and
        word_count from `paragraphs`, president/title/year from `speeches`,
        speech_type from `speech_annotations`, and the adversary counts from
        `paragraph_entities`. The fixture makes each value unique per key, so a
        misalignment cannot coincidentally reproduce the right answer.
        """
        specs = []
        for d, (year, president) in enumerate(
            [(1820, "Monroe"), (1900, "McKinley"), (2020, "Trump")]
        ):
            specs.append(
                {
                    "doc": f"doc{d}",
                    "year": year,
                    "president": president,
                    "title": f"Address of {president}",
                    "type": f"genre{d}",
                    "paras": [
                        {
                            "text": f"doc{d} para{i} unique text",
                            "word_count": 100 * d + i,
                            "party_attack": (i % 2 == 0),
                            "adversaries": [(f"foe{d}{i}{j}", "nation") for j in range(i)],
                        }
                        for i in range(4)
                    ],
                }
            )
        inputs = combat_inputs(specs)
        carried = [
            "text",
            "word_count",
            "president",
            "title",
            "year",
            "speech_type",
            "party_attack",
            "n_adversarial",
        ]

        def keyed(frames):
            out = C.load_frame(**frames)
            return out.set_index(["doc_name", "para_idx"])[carried].sort_index()

        expected = keyed(inputs)
        # sanity: the fixture really does distinguish rows
        assert expected["text"].nunique() == len(expected)
        assert expected["word_count"].nunique() == len(expected)

        # A reversal, not `.sample(frac=1)`: on a frame this small a fixed seed
        # can hand back the original order (random_state=45 does exactly that to
        # the 3-row speeches frame), which would make the invariance claim
        # vacuous for whichever frame it skipped.
        reversed_inputs = {k: v.iloc[::-1].reset_index(drop=True) for k, v in inputs.items()}
        still_ordered = [k for k in inputs if reversed_inputs[k].equals(inputs[k])]
        assert not still_ordered, f"{still_ordered} were not reordered — test would be vacuous"
        pd.testing.assert_frame_equal(keyed(reversed_inputs), expected)

        for seed in (0, 7, 42):
            shuffled = {
                k: v.sample(frac=1, random_state=seed + i).reset_index(drop=True)
                for i, (k, v) in enumerate(inputs.items())
            }
            assert any(not shuffled[k].equals(inputs[k]) for k in inputs), seed
            pd.testing.assert_frame_equal(keyed(shuffled), expected)

    def test_duplicate_paragraph_key_raises_a_merge_error(self, combat_inputs):
        inputs = self._inputs(combat_inputs)
        inputs["annotations"] = pd.concat(
            [inputs["annotations"], inputs["annotations"].iloc[[0]]], ignore_index=True
        )
        with pytest.raises(pd.errors.MergeError):
            C.load_frame(**inputs)

    def test_a_paragraph_missing_its_annotation_raises(self, combat_inputs):
        """Keys stay unique, so validate="one_to_one" is satisfied — only the
        row-completeness assertion catches this. Without it every rate would be
        computed over the intersection while the report claimed full coverage."""
        inputs = self._inputs(combat_inputs)
        inputs["annotations"] = inputs["annotations"].iloc[1:].reset_index(drop=True)
        with pytest.raises(ValueError, match="paragraph_annotations x paragraphs"):
            C.load_frame(**inputs)

    def test_an_annotation_missing_its_paragraph_raises(self, combat_inputs):
        """The other direction of the same divergence."""
        inputs = self._inputs(combat_inputs)
        inputs["paragraphs"] = inputs["paragraphs"].iloc[1:].reset_index(drop=True)
        with pytest.raises(ValueError, match="key sets diverge"):
            C.load_frame(**inputs)

    def test_a_speech_missing_its_speech_annotation_raises(self, combat_inputs):
        inputs = self._inputs(combat_inputs)
        inputs["speech_annotations"] = (
            inputs["speech_annotations"].iloc[1:].reset_index(drop=True)
        )
        with pytest.raises(ValueError, match="speeches x speech_annotations"):
            C.load_frame(**inputs)

    def test_a_paragraph_whose_speech_is_unknown_raises(self, combat_inputs):
        """Both paragraph tables agree, and speeches x speech_annotations is
        complete — but a doc_name exists only on the paragraph side, so the
        many_to_one merge would silently drop it."""
        inputs = self._inputs(combat_inputs)
        ghost_para = pd.DataFrame(
            [{"doc_name": "ghost", "para_idx": 0, "text": "orphan", "word_count": 5}]
        )
        ghost_ann = pd.DataFrame(
            [
                {
                    "doc_name": "ghost",
                    "para_idx": 0,
                    "run_id": "run-test",
                    **{f: False for f in C.FLAGS},
                }
            ]
        )
        inputs["paragraphs"] = pd.concat([inputs["paragraphs"], ghost_para], ignore_index=True)
        inputs["annotations"] = pd.concat([inputs["annotations"], ghost_ann], ignore_index=True)
        with pytest.raises(ValueError, match="not in the speech table"):
            C.load_frame(**inputs)

    def test_derived_columns_are_present_and_typed(self, combat_inputs):
        df = C.load_frame(**self._inputs(combat_inputs))
        assert df["era"].tolist() == [
            "Civil War & Reconstruction",
            "Civil War & Reconstruction",
            "The present era",
        ]
        assert list(df["era"].cat.categories) == C.ERA_ORDER
        assert df["decade"].tolist() == [1860, 1860, 2020]
        assert df["is_sotu"].all()


# --------------------------------------------------------------------------- #
# paragraph_entities: deliberately many-per-key
# --------------------------------------------------------------------------- #
class TestAdversaryCounts:
    def _entity_inputs(self, combat_inputs):
        return combat_inputs(
            [
                {
                    "doc": "x1",
                    "year": 2020,
                    "paras": [
                        {  # three adversaries on ONE key: the fan-out risk
                            "enemy_naming": True,
                            "adversaries": [
                                ("Iran", "nation"),
                                ("China", "nation"),
                                ("the Governor", "person"),
                            ],
                        },
                        {"enemy_naming": True, "adversaries": [("the trusts", "group")]},
                        {"enemy_naming": True},  # flagged but names nobody
                        {"adversaries": [("something", "other")]},
                        {"entities": [("an ally", "nation", "favorable")]},
                    ],
                }
            ]
        )

    def test_many_entities_per_paragraph_collapse_to_one_row_per_key(self, combat_inputs):
        ents = self._entity_inputs(combat_inputs)["entities"]
        assert ents.duplicated(["doc_name", "para_idx"]).any(), "fixture must be many-per-key"

        counts = C._adversary_counts(entities=ents)

        assert not counts.duplicated(["doc_name", "para_idx"]).any()
        row = counts.set_index("para_idx").loc[0]
        assert row["n_adversarial"] == 3
        assert row["n_adv_foreign"] == 2
        assert row["n_adv_domestic"] == 1

    def test_only_adversarial_stance_is_counted(self, combat_inputs):
        counts = C._adversary_counts(entities=self._entity_inputs(combat_inputs)["entities"])
        # para 4 names a favorable nation and must not appear at all.
        assert 4 not in set(counts["para_idx"])

    def test_other_type_counts_in_the_total_but_in_neither_component(self, combat_inputs):
        """The foreign/domestic split is deliberately NOT a partition, so the two
        components can never over-claim."""
        counts = C._adversary_counts(entities=self._entity_inputs(combat_inputs)["entities"])
        row = counts.set_index("para_idx").loc[3]
        assert row["n_adversarial"] == 1
        assert row["n_adv_foreign"] == 0
        assert row["n_adv_domestic"] == 0

    def test_load_frame_does_not_fan_out_on_the_many_per_key_entity_table(
        self, combat_inputs
    ):
        """The regression this guards: joining paragraph_entities one_to_one (or
        without aggregating) would triple-count paragraph 0 and inflate every
        rate that uses it. Row count must equal the annotation table exactly."""
        inputs = self._entity_inputs(combat_inputs)
        df = C.load_frame(**inputs)
        assert len(df) == len(inputs["annotations"])
        assert not df.duplicated(["doc_name", "para_idx"]).any()

    def test_a_paragraph_naming_nobody_survives_with_zero_counts(self, combat_inputs):
        """Naming no entity is legitimate data, not a missing row — the entity
        join is a LEFT merge, and the counts must land as 0, not NaN."""
        df = C.load_frame(**self._entity_inputs(combat_inputs)).set_index("para_idx")
        assert df.loc[2, "n_adversarial"] == 0
        for col in ["n_adversarial", "n_adv_foreign", "n_adv_domestic"]:
            assert df[col].dtype.kind == "i"

    def test_enemy_backed_requires_the_flag_and_an_adversarial_entity(self, combat_inputs):
        df = C.load_frame(**self._entity_inputs(combat_inputs)).set_index("para_idx")
        assert bool(df.loc[0, "enemy_backed"]) is True
        assert bool(df.loc[2, "enemy_backed"]) is False  # flagged, no entity
        assert bool(df.loc[3, "enemy_backed"]) is False  # entity, not flagged

    def test_domestic_only_excludes_paragraphs_that_also_name_a_nation(self, combat_inputs):
        """Paragraph 0 names two nations AND a person; it belongs to the foreign
        component only. This asymmetry is what makes the Civil War adjudication
        conservative."""
        df = C.load_frame(**self._entity_inputs(combat_inputs)).set_index("para_idx")
        assert bool(df.loc[0, "enemy_foreign"]) is True
        assert bool(df.loc[0, "enemy_domestic_only"]) is False
        assert bool(df.loc[1, "enemy_foreign"]) is False
        assert bool(df.loc[1, "enemy_domestic_only"]) is True


# --------------------------------------------------------------------------- #
# entity_consistency / adversary_mix
# --------------------------------------------------------------------------- #
class TestEntityConsistency:
    def _frame(self, combat_inputs):
        return C.load_frame(
            **combat_inputs(
                [
                    {
                        "doc": "cw",
                        "year": 1860,
                        "paras": [
                            {"enemy_naming": True, "adversaries": [("the insurgents", "group")]},
                            {"enemy_naming": True, "adversaries": [("Britain", "nation")]},
                            {"enemy_naming": True, "adversaries": [("Congress", "institution")]},
                            {"enemy_naming": True},  # unbacked
                            {"enemy_naming": False},
                        ],
                    }
                ]
            )
        )

    def test_consistency_rate_is_backed_over_flagged(self, combat_inputs):
        out = self._frame(combat_inputs).pipe(C.entity_consistency).set_index("era")
        row = out.loc["Civil War & Reconstruction"]
        assert row["n_enemy_naming"] == 4  # the unflagged paragraph is excluded
        assert row["n_backed"] == 3
        assert row["consistency_rate"] == pytest.approx(0.75)

    def test_foreign_and_domestic_shares_are_taken_over_flagged_paragraphs_only(
        self, combat_inputs
    ):
        out = self._frame(combat_inputs).pipe(C.entity_consistency).set_index("era")
        row = out.loc["Civil War & Reconstruction"]
        assert row["share_foreign_adversary"] == pytest.approx(1 / 4)
        assert row["share_domestic_only"] == pytest.approx(2 / 4)

    def test_every_era_is_present_even_with_no_flagged_paragraphs(self, combat_inputs):
        """The table is the per-era honesty check; an era silently missing from
        it reads as "fine" rather than "unmeasured"."""
        out = self._frame(combat_inputs).pipe(C.entity_consistency)
        assert out["era"].tolist() == C.ERA_ORDER
        assert out["era_order"].tolist() == list(range(9))
        assert pd.isna(out.set_index("era").loc["The founding", "consistency_rate"])


class TestAdversaryMix:
    def test_carries_both_era_and_decade_grain(self, combat_inputs, combat_paras):
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a", "year": 1861, "paras": combat_paras(4, 1, "enemy_naming")},
                    {"doc": "b", "year": 1871, "paras": combat_paras(4, 2, "enemy_naming")},
                ]
            )
        )
        out = C.adversary_mix(df)
        assert set(out["grain"]) == {"era", "decade"}
        era_rows = out[out["grain"] == "era"]
        assert era_rows["group"].tolist() == ["Civil War & Reconstruction"]
        assert sorted(out[out["grain"] == "decade"]["group"]) == ["1860", "1870"]

    def test_rates_are_group_means_and_counts_are_group_sizes(self, combat_inputs, combat_paras):
        """The era band 1850-1877 fuses the war decade with Reconstruction; the
        decade grain exists so that dilution is visible, so both must be right."""
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a", "year": 1861, "paras": combat_paras(4, 1, "enemy_naming")},
                    {"doc": "b", "year": 1871, "paras": combat_paras(4, 2, "enemy_naming")},
                ]
            )
        )
        out = C.adversary_mix(df).set_index(["grain", "group"])
        assert out.loc[("era", "Civil War & Reconstruction"), "enemy_naming"] == pytest.approx(
            3 / 8
        )
        assert out.loc[("decade", "1860"), "enemy_naming"] == pytest.approx(1 / 4)
        assert out.loc[("decade", "1870"), "enemy_naming"] == pytest.approx(2 / 4)
        assert out.loc[("era", "Civil War & Reconstruction"), "n_paragraphs"] == 8
        assert out.loc[("era", "Civil War & Reconstruction"), "n_speeches"] == 2


# --------------------------------------------------------------------------- #
# lexical baseline (the uncorrelated-error cross-check)
# --------------------------------------------------------------------------- #
class TestLexicalBaseline:
    def _frame_and_markers(self, combat_inputs, combat_paras, tmp_path, markers=None):
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a1", "year": 1820, "type": SOTU, "paras": combat_paras(4, 1)},
                    {"doc": "a2", "year": 1820, "type": SOTU, "paras": combat_paras(4, 2)},
                    {"doc": "a3", "year": 1820, "type": "rally", "paras": combat_paras(4, 4)},
                ]
            )
        )
        if markers is None:
            markers = pd.DataFrame(
                {
                    "doc_name": ["a1", "a2", "a3"],
                    "us_them": [10, 20, 300],
                    "opponents": [1, 2, 30],
                    "n_words": [1000, 1000, 1000],
                }
            )
        path = tmp_path / "speech_markers.parquet"
        markers.to_parquet(path, index=False)
        return df, path

    def test_per_10k_rates_pool_counts_over_pooled_words(
        self, combat_inputs, combat_paras, tmp_path
    ):
        """A rate, not a mean of rates: sum(counts)/sum(words), so a long speech
        weighs more than a short one."""
        df, path = self._frame_and_markers(combat_inputs, combat_paras, tmp_path)
        out = C.lexical_baseline(df, path=path)
        raw = _one(out, treatment="raw", era="Expansion")
        assert raw["us_them"] == 330
        assert raw["n_words"] == 3000
        assert raw["us_them_per_10k"] == pytest.approx(330 / 3000 * 10_000)
        assert raw["opponents_per_10k"] == pytest.approx(33 / 3000 * 10_000)
        assert raw["n_speeches"] == 3

    def test_sotu_only_treatment_drops_the_non_sotu_speech(
        self, combat_inputs, combat_paras, tmp_path
    ):
        df, path = self._frame_and_markers(combat_inputs, combat_paras, tmp_path)
        out = C.lexical_baseline(df, path=path)
        sotu = _one(out, treatment="sotu_only", era="Expansion")
        assert sotu["n_speeches"] == 2
        assert sotu["us_them"] == 30
        assert sotu["us_them_per_10k"] == pytest.approx(30 / 2000 * 10_000)

    def test_a_speech_missing_from_the_markers_table_raises(
        self, combat_inputs, combat_paras, tmp_path
    ):
        markers = pd.DataFrame(
            {
                "doc_name": ["a1", "a2"],
                "us_them": [10, 20],
                "opponents": [1, 2],
                "n_words": [1000, 1000],
            }
        )
        df, path = self._frame_and_markers(combat_inputs, combat_paras, tmp_path, markers)
        with pytest.raises(ValueError, match="speech flags x speech_markers"):
            C.lexical_baseline(df, path=path)

    def test_an_era_with_no_sotu_speech_still_carries_the_treatment_label(
        self, combat_inputs, combat_paras, tmp_path
    ):
        # The present era has a speech, but it is not an annual message — so it
        # is present in the `raw` series and absent from `sotu_only`.
        df = C.load_frame(
            **combat_inputs(
                [
                    {"doc": "a1", "year": 1820, "type": SOTU, "paras": combat_paras(4, 1)},
                    {"doc": "b1", "year": 2020, "type": "rally", "paras": combat_paras(4, 3)},
                ]
            )
        )
        markers = pd.DataFrame(
            {
                "doc_name": ["a1", "b1"],
                "us_them": [10, 30],
                "opponents": [1, 3],
                "n_words": [1000, 1000],
            }
        )
        path = tmp_path / "speech_markers.parquet"
        markers.to_parquet(path, index=False)

        out = C.lexical_baseline(df, path=path)
        present = out[out["era"] == "The present era"]
        assert sorted(present["treatment"].tolist()) == ["raw", "sotu_only"]

    def test_duplicate_marker_key_raises(self, combat_inputs, combat_paras, tmp_path):
        markers = pd.DataFrame(
            {
                "doc_name": ["a1", "a1", "a2", "a3"],
                "us_them": [10, 10, 20, 300],
                "opponents": [1, 1, 2, 30],
                "n_words": [1000, 1000, 1000, 1000],
            }
        )
        df, path = self._frame_and_markers(combat_inputs, combat_paras, tmp_path, markers)
        with pytest.raises(pd.errors.MergeError):
            C.lexical_baseline(df, path=path)


class TestLexicalCorrelations:
    def test_a_perfectly_monotone_relation_scores_one(
        self, combat_inputs, combat_paras, tmp_path
    ):
        """Spearman on speech-level rates. Built monotone so the expected value
        is exactly 1.0 and the test does not depend on a fitted magnitude."""
        df = C.load_frame(
            **combat_inputs(
                [
                    # every flag varies across speeches, so no correlation is
                    # computed against a constant column (which is undefined).
                    {"doc": "s1", "year": 2020, "paras": combat_paras(4, 1, zero_sum=True)},
                    {"doc": "s2", "year": 2020, "paras": combat_paras(4, 2, enemy_naming=True)},
                    {"doc": "s3", "year": 2020, "paras": combat_paras(4, 3)},
                ]
            )
        )
        markers = pd.DataFrame(
            {
                "doc_name": ["s1", "s2", "s3"],
                "us_them": [5, 50, 500],
                "opponents": [500, 50, 5],
                "n_words": [1000, 1000, 1000],
            }
        )
        path = tmp_path / "markers.parquet"
        markers.to_parquet(path, index=False)

        out = C.lexical_correlations(df, path=path)

        assert out["party_attack~us_them_per_10k"] == pytest.approx(1.0)
        assert out["party_attack~opponents_per_10k"] == pytest.approx(-1.0)

    def test_reports_every_flag_by_proxy_pair(self, combat_inputs, combat_paras, tmp_path):
        df = C.load_frame(
            **combat_inputs([{"doc": "s1", "year": 2020, "paras": combat_paras(4, 1)}])
        )
        markers = pd.DataFrame(
            {"doc_name": ["s1"], "us_them": [5], "opponents": [5], "n_words": [1000]}
        )
        path = tmp_path / "markers.parquet"
        markers.to_parquet(path, index=False)

        out = C.lexical_correlations(df, path=path)

        assert set(out) == {
            f"{flag}~{proxy}"
            for flag in C.FLAGS
            for proxy in ["us_them_per_10k", "opponents_per_10k"]
        }


# --------------------------------------------------------------------------- #
# exemplars — the receipts behind every point on the chart
# --------------------------------------------------------------------------- #
class TestExemplars:
    def _frame(self, combat_inputs, n_candidates=12):
        paras = [
            {
                "party_attack": True,
                "adversaries": [(f"foe{j}", "person") for j in range(n_candidates - i)],
                "text": f"candidate {i}",
                "word_count": 50,
            }
            for i in range(n_candidates)
        ]
        return C.load_frame(
            **combat_inputs([{"doc": "d1", "year": 2020, "paras": paras}])
        )

    def test_ranks_by_adversarial_count_then_flag_count_then_key(self, combat_inputs):
        """Ranking uses the entity count — a signal INDEPENDENT of the flag —
        so an exemplar is not merely the flag agreeing with itself."""
        df = self._frame(combat_inputs, n_candidates=4)
        out = C.exemplars(df)
        cell = out[(out["flag"] == "party_attack") & (out["era"] == "The present era")]
        assert cell["rank"].tolist() == [1, 2, 3, 4]
        assert cell["n_adversarial"].tolist() == [4, 3, 2, 1]
        assert cell["text"].tolist() == [
            "candidate 0",
            "candidate 1",
            "candidate 2",
            "candidate 3",
        ]

    def test_caps_at_the_face_validity_protocol_size(self, combat_inputs):
        """The exemplar table IS the spot-read corpus the plan specifies (10 per
        era per flag), so the cap and the protocol must be the same number."""
        assert C.EXEMPLARS_PER_CELL == 10
        out = C.exemplars(self._frame(combat_inputs, n_candidates=12))
        cell = out[(out["flag"] == "party_attack") & (out["era"] == "The present era")]
        assert len(cell) == 10
        assert cell["rank"].tolist() == list(range(1, 11))

    def test_excludes_paragraphs_outside_the_disclosed_word_window(self, combat_inputs):
        """A quotability filter, not a scoring filter — but it still has to be
        applied at its stated bounds (inclusive) or the evidence is biased."""
        assert (C.EXEMPLAR_MIN_WORDS, C.EXEMPLAR_MAX_WORDS) == (20, 200)
        df = C.load_frame(
            **combat_inputs(
                [
                    {
                        "doc": "d1",
                        "year": 2020,
                        "paras": [
                            {"party_attack": True, "word_count": 19, "text": "too short"},
                            {"party_attack": True, "word_count": 20, "text": "lower bound"},
                            {"party_attack": True, "word_count": 200, "text": "upper bound"},
                            {"party_attack": True, "word_count": 201, "text": "too long"},
                        ],
                    }
                ]
            )
        )
        texts = set(C.exemplars(df)["text"])
        assert texts == {"lower bound", "upper bound"}

    def test_every_row_records_the_selection_window_and_its_attribution(self, combat_inputs):
        out = C.exemplars(self._frame(combat_inputs, n_candidates=2))
        row = out.iloc[0]
        assert row["selection_min_words"] == C.EXEMPLAR_MIN_WORDS
        assert row["selection_max_words"] == C.EXEMPLAR_MAX_WORDS
        assert row["doc_name"] == "d1"
        assert row["year"] == 2020
        assert row["president"] == "A President"
        assert row["speech_type"] == SOTU
        assert row["n_flags"] == 1

    def test_ties_are_broken_deterministically_by_key_and_survive_a_reshuffle(
        self, combat_inputs
    ):
        """MUTATION-CHECK FINDING: the tie-break itself was untested.

        `test_ranks_by_adversarial_count_then_flag_count_then_key` gives every
        candidate a DIFFERENT adversary count, so no tie ever occurs and the
        `(doc_name, para_idx)` tie-break could be deleted without a failure.
        On the real corpus ties are the common case — most flagged paragraphs
        name exactly one adversary and fire exactly one flag — and pandas'
        default sort is not stable, so without the key the ten quoted passages
        per era would be an arbitrary ten that could change between runs, row
        orders, or pandas versions. The exemplars table is the report's
        evidence; it has to be reproducible.

        Two docs, six fully-tied candidates each, specified with `d2` FIRST so
        that "whatever order the frame happened to be in" and "sorted by key"
        are different answers.
        """
        tied = [
            {"party_attack": True, "adversaries": [("a foe", "person")], "word_count": 50}
            for _ in range(6)
        ]
        inputs = combat_inputs(
            [
                {"doc": "d2", "year": 2020, "paras": list(tied)},
                {"doc": "d1", "year": 2020, "paras": list(tied)},
            ]
        )
        df = C.load_frame(**inputs)
        assert df["n_adversarial"].nunique() == 1, "fixture must be fully tied"

        expected = [("d1", i) for i in range(6)] + [("d2", i) for i in range(4)]

        out = C.exemplars(df)
        cell = out[(out["flag"] == "party_attack") & (out["era"] == "The present era")]
        assert list(zip(cell["doc_name"], cell["para_idx"])) == expected
        assert cell["rank"].tolist() == list(range(1, 11))

        shuffled = {
            k: v.sample(frac=1, random_state=3 + i).reset_index(drop=True)
            for i, (k, v) in enumerate(inputs.items())
        }
        assert any(not shuffled[k].equals(inputs[k]) for k in inputs), "shuffle was a no-op"
        reshuffled = C.exemplars(C.load_frame(**shuffled))
        rcell = reshuffled[
            (reshuffled["flag"] == "party_attack")
            & (reshuffled["era"] == "The present era")
        ]
        assert list(zip(rcell["doc_name"], rcell["para_idx"])) == expected

    def test_a_flag_that_never_fires_contributes_no_rows(self, combat_inputs):
        out = C.exemplars(self._frame(combat_inputs, n_candidates=2))
        assert set(out["flag"]) == {"party_attack"}


# --------------------------------------------------------------------------- #
# module-level guard: nothing here is allowed to reach the paid API
# --------------------------------------------------------------------------- #
def test_combat_module_never_imports_or_touches_anthropic():
    """$0 guard, checked statically so it holds for code paths no test walks.

    combat.py is pure local compute over frozen parquets; a stray client would
    turn a research rebuild into a billed run. conftest's ``_no_anthropic_creds``
    would make that fail loudly at runtime, but only on a path a test exercises —
    this closes the rest of the file."""
    source = Path(C.__file__).read_text()
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "anthropic" not in imported

    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert not [r for r in referenced if "anthropic" in r.lower()]
    assert "Anthropic(" not in source
