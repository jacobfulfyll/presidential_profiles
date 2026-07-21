"""Tests for the topic-attention pipeline (`attention.py`).

`attention.py` is $0 by construction — every input is a frozen local artifact and
no Anthropic client is ever built (pinned in `TestModuleContract`). Per the repo
test conventions every fixture here is a tiny hand-made frame with KNOWN
structure; the real 36k-row corpus is touched only by the read-only
`TestCommittedArtifacts` anchors, and no CorEx model is ever fitted.

The module's numeric rules are *pre-registered* in its docstring, so most of
these tests are threshold pins: they exist so a later edit cannot silently move
a constant or an inequality that a published finding rests on.

Coverage map (public + private-by-name helpers):
  * era_name / era_series      .. TestEras            (pin: 1789-2026 equivalence,
                                                       the bins/right=True off-by-one)
  * era_span_label             .. TestEras
  * load_taxonomy              .. TestLabelNormalization
  * canonical_label_map        .. TestLabelNormalization (non-injective casefold raises)
  * normalize_topics           .. TestLabelNormalization (dedup + raise on unmapped)
  * load_inputs                .. TestLoadInputs       (keyed-merge discipline:
                                                       _require_full_merge is wired in)
  * level_topics / slice_for   .. TestLevelTopicsAndSlices
  * _year_index / _counts      .. TestCountsAndSmoothing
  * _smooth                    .. TestCountsAndSmoothing (centered rolling SUM)
  * _combine_strata            .. TestCombineStrata    (pin: missing-genre fallback)
  * attention_curves           .. TestAttentionCurves
  * corex_curves               .. TestCorexCurves
  * substantive_mask           .. TestSubstantiveMask  (pin: one curve per leg)
  * topic_lifecycle            .. TestTopicLifecycle   (pin: centered-window clamp,
                                                       classes, under-powered guard)
  * extract_lifecycles         .. TestExtractLifecycles
  * invert_crosswalk           .. TestInvertCrosswalk  (many-to-many)
  * _decline_ratio / _pearson  .. TestDeclineRatioAndPearson
  * find_successor             .. TestFindSuccessor    (four filters; filter 4 is
                                                       SKIPPED with no parent — pinned
                                                       as current behaviour)
  * classify_deaths            .. TestClassifyDeaths   (pin: multi-parent rule,
                                                       successor precedence)
  * _speech_design / _era_shares .. TestSpeechDesignAndEraShares (pin: the era
                                                       grid keeps a sub-minimum
                                                       genre stratum while the
                                                       year curves discard it)
  * bootstrap_era_shares       .. TestBootstrapEraShares (pin: seed determinism,
                                                       point estimate == groupby)
  * exemplar_quotes            .. TestExemplarQuotes   (pin: the guarded merge
                                                       RAISES on a pooled row the
                                                       text parquet lacks)
  * build_lifecycle_table      .. TestBuildLifecycleTable (end-to-end anchor)
  * write_lifecycles           .. TestBuildLifecycleTable (byte-reproducible)
  * run_ground_truth           .. TestGroundTruthAndAnachronism
  * anachronism_report         .. TestGroundTruthAndAnachronism
  * main                       .. TestCli
  * committed artifacts        .. TestCommittedArtifacts

Fixture note: `tests/` is not an importable package, so helpers cannot be shared
across test modules; the builders below are local by necessity. Where shared
scaffolding DOES exist it is reused — the autouse `redirect_annotation_dirs`
fixture from `conftest.py` is what makes the hermetic `load_inputs` tests
possible, and the monkeypatch-the-module-path idiom follows
`test_keyed_merge._install_parquets`.

Coverage note: this suite reports no line-coverage number, and the reason is
NOT that `pytest-cov` is unavailable — it is declared at `pyproject.toml:29`
(`pytest-cov>=5.0`) and resolved in `uv.lock` at 7.1.0. The reason is that
installing it writes into the SHARED `.venv` at the main repo, which every
worktree imports through, so a coverage run here would mutate an environment
this task does not own. Confidence in these tests rests on mutation probing
instead (see the QUALITY-CHECK handoff): line coverage says a line ran, a
surviving mutant says no assertion depended on it, and it is the second question
that matters for a suite whose job is to stop a constant moving silently.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import attention as A
from presidential_profiles.taxonomy import LEGACY_ISSUES, SECURITY_PEACE
from presidential_profiles.trends import ERAS

_REPO = Path(__file__).resolve().parents[1]
_ARTIFACTS = _REPO / "data" / "llm_annotations"
_ATTENTION_ARTIFACTS = _REPO / "data" / "attention"

SOTU = A.SOTU_TYPE
OTHER = "public_remarks_or_address"


# --------------------------------------------------------------------------- #
# frame / artifact builders
# --------------------------------------------------------------------------- #
def _taxonomy(level2: dict[str, str | None]) -> dict:
    """Minimal taxonomy_v1-shaped dict. `level2` maps topic name -> level-1 name.

    A None parent is legal here on purpose: it is how the orphan-topic guard in
    `load_inputs` gets exercised.
    """
    domains = sorted({d for d in level2.values() if d is not None})
    return {
        "level1": [
            {"name": d, "definition": f"{d} definition", "kind": "policy"} for d in domains
        ],
        "level2": [
            {
                "name": name,
                "definition": f"{name} definition",
                "level1": parent,
                "era_note": "",
                "era_of_birth": "1800s",
                "era_of_death": None,
                "exemplar_paragraphs": [],
            }
            for name, parent in level2.items()
        ],
    }


def _crosswalk(mappings: dict[str, list[str]]) -> dict:
    """crosswalk_v1-shaped dict: legacy issue -> level-2 topics."""
    return {
        "mappings": [
            {"legacy_issue": issue, "level2_topics": topics}
            for issue, topics in mappings.items()
        ]
    }


def _corpus(rows, parents: dict[str, str]):
    """Build `(paragraphs, assignments)` in exactly `load_inputs`' output schema.

    rows: iterable of (doc_name, para_idx, year, speech_type, [topic, ...]).
    """
    rows = list(rows)
    paragraphs = pd.DataFrame(
        [
            {"doc_name": d, "para_idx": p, "year": y, "speech_type": st}
            for d, p, y, st, _ in rows
        ]
    )
    paragraphs["era"] = A.era_series(paragraphs["year"])
    assignments = pd.DataFrame(
        [
            {"doc_name": d, "para_idx": p, "topic": t}
            for d, p, _, _, topics in rows
            for t in topics
        ],
        columns=["doc_name", "para_idx", "topic"],
    )
    assignments["level1"] = assignments["topic"].map(parents)
    assignments = assignments.merge(
        paragraphs[["doc_name", "para_idx", "year", "era", "speech_type"]],
        on=["doc_name", "para_idx"],
        validate="many_to_one",
    )
    return (
        paragraphs.sort_values(["doc_name", "para_idx"]).reset_index(drop=True),
        assignments.sort_values(["doc_name", "para_idx", "topic"]).reset_index(drop=True),
    )


def _curve(years, *, n_topic, n_topic_window, n_total_window, share_smooth,
           topic="T", level="level2", treatment="raw"):
    """One topic's rows in `attention_curves` schema, from per-year arrays."""
    years = list(years)
    def _fill(v):
        return list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series)) else [v] * len(years)
    frame = pd.DataFrame(
        {
            "level": level,
            "treatment": treatment,
            "topic": topic,
            "year": years,
            "n_topic": _fill(n_topic),
            "n_total": _fill(n_total_window),
            "share": np.nan,
            "n_topic_window": _fill(n_topic_window),
            "n_total_window": _fill(n_total_window),
            "share_smooth": _fill(share_smooth),
        }
    )
    return frame


def _install_annotation_parquets(monkeypatch, tmp_path, redirect_annotation_dirs,
                                 *, annotations, speeches, issues):
    """Write the three read-side inputs `load_inputs` consumes and point the
    module path constants at them (same idiom as test_keyed_merge)."""
    root = redirect_annotation_dirs
    root.mkdir(parents=True, exist_ok=True)
    annotations.to_parquet(root / "paragraph_annotations.parquet", index=False)
    speeches.to_parquet(root / "speech_annotations.parquet", index=False)
    labels = tmp_path / "paragraph_issues.parquet"
    issues.to_parquet(labels, index=False)
    monkeypatch.setattr(A, "PARA_LABELS_PATH", labels)
    return labels


def _annotation_frame(rows):
    """rows: (doc_name, para_idx, [raw topic labels])."""
    return pd.DataFrame(
        [{"doc_name": d, "para_idx": p, "topics": t, "run_id": "test"} for d, p, t in rows]
    )


def _speech_frame(rows):
    """rows: (doc_name, speech_type)."""
    return pd.DataFrame(
        [{"doc_name": d, "speech_type": st, "run_id": "test"} for d, st in rows]
    )


def _issue_frame(rows, fired=None):
    """rows: (doc_name, para_idx, year). `fired` maps key -> [legacy issue names]."""
    fired = fired or {}
    out = []
    for d, p, y in rows:
        rec = {"doc_name": d, "para_idx": p, "year": y, "president": "Someone"}
        on = set(fired.get((d, p), []))
        for name in A.COREX_COLUMNS.values():
            rec[name] = name in on
        out.append(rec)
    return pd.DataFrame(out)


def _series(years, spans: list[tuple[int, int, float]], default=0.0) -> pd.Series:
    """A year-indexed curve built from [lo, hi] -> value spans."""
    idx = pd.Index(years, name="year")
    values = np.full(len(idx), float(default))
    for lo, hi, value in spans:
        values[(idx >= lo) & (idx <= hi)] = value
    return pd.Series(values, index=idx)


def _curves_long(series_by_topic: dict[str, pd.Series]) -> pd.DataFrame:
    """`attention_curves`-shaped long frame carrying only what classify_deaths
    pivots on (year, topic, share_smooth)."""
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "level": "level2",
                    "treatment": "raw",
                    "topic": topic,
                    "year": series.index,
                    "share_smooth": series.to_numpy(),
                }
            )
            for topic, series in series_by_topic.items()
        ],
        ignore_index=True,
    )


# =========================================================================== #
class TestModuleContract:
    """The module's own promises: $0, fixed thresholds, fixed vocabulary."""

    def test_importing_attention_never_loads_anthropic(self):
        """The whole task is $0. A fresh interpreter that imports only
        `attention` must not pull `anthropic` into sys.modules — that is the
        structural version of "no paid path", stronger than a credentials guard."""
        src_root = Path(A.__file__).resolve().parents[1]
        code = (
            "import sys; import presidential_profiles.attention; "
            "print('anthropic' in sys.modules)"
        )
        env = {**os.environ, "PYTHONPATH": str(src_root)}
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "False"

    def test_module_source_imports_no_client(self):
        """Belt to the subprocess check's braces: a LAZILY imported client inside
        a function body would not show up in sys.modules unless that function
        ran, but it would show up here."""
        source = Path(A.__file__).read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )

        assert "import anthropic" not in code
        assert "Anthropic(" not in code

    def test_pre_registered_thresholds_match_the_module_docstring(self):
        """Decoupled literal spec-anchor. Every other test in this file reads the
        constants from the module, so moving one would silently move the tests
        with it; these literals are the pre-registered values from the docstring
        and must be changed deliberately."""
        assert A.SMOOTH_WINDOW == 5
        assert A.MIN_WINDOW_PARAGRAPHS == 30
        assert A.MIN_TOPIC_PARAGRAPHS == 3
        assert A.ABS_SHARE_FLOOR == 0.002
        assert A.REL_PEAK_FRACTION == 0.10
        assert A.UNDER_POWERED_N == 50
        assert A.MIN_GENRE_STRATUM == 3
        assert A.DEATH_GAP_YEARS == 40
        assert A.BIRTH_GAP_YEARS == 40
        assert A.REVIVAL_GAP_YEARS == 30
        assert A.COREX_COLLAPSE_RATIO == 0.25
        assert A.PEAK_WINDOW_HALF == 10
        assert (A.BOOTSTRAP_DRAWS, A.BOOTSTRAP_SEED) == (500, 20260721)
        assert (A.CI_LOW, A.CI_HIGH) == (2.5, 97.5)

    def test_treatments_and_levels_are_the_three_and_two_the_spec_names(self):
        assert A.TREATMENTS == ("raw", "sotu", "genre_standardized")
        assert A.LEVELS == ("level2", "level1")
        assert A.SOTU_TYPE == "state_of_the_union_or_annual_message"

    def test_corex_columns_are_the_15_legacy_issues_plus_discovered_5(self):
        """`Discovered 5` is the one free CorEx topic with an agreed identity;
        the other six are deliberately unused (backlog: discovered-topics-1-7)."""
        assert set(A.COREX_COLUMNS) == set(LEGACY_ISSUES) | {SECURITY_PEACE}
        assert A.COREX_COLUMNS[SECURITY_PEACE] == "Discovered 5"
        assert all(A.COREX_COLUMNS[i] == i for i in LEGACY_ISSUES)
        assert not any(
            col.startswith("Discovered") and col != "Discovered 5"
            for col in A.COREX_COLUMNS.values()
        )


# =========================================================================== #
class TestEras:
    """PIN 1: `era_series`'s `bins = [ERAS[0][1] - 1] + [hi ...]` with pandas'
    default `right=True` is a non-obvious off-by-one that happens to be exactly
    right. Any future edit to either implementation must keep them identical."""

    def test_era_series_matches_era_name_for_every_year_1789_to_2026(self):
        years = pd.Series(range(1789, 2027))
        vectorized = A.era_series(years)

        assert not vectorized.isna().any(), "every in-range year must land in a band"
        assert list(vectorized.astype(object)) == [A.era_name(y) for y in years]

    @pytest.mark.parametrize("year,expected", [
        (1789, "The founding"), (1815, "The founding"),
        (1816, "Expansion"), (1849, "Expansion"),
        (1850, "Civil War & Reconstruction"), (1877, "Civil War & Reconstruction"),
        (1878, "The Gilded Age"), (1900, "The Gilded Age"),
        (1901, "Progressives & Depression"), (1932, "Progressives & Depression"),
        (1933, "War & New Deal"), (1945, "War & New Deal"),
        (1946, "The Cold War"), (1988, "The Cold War"),
        (1989, "Post-Cold War"), (2016, "Post-Cold War"),
        (2017, "The present era"), (2026, "The present era"),
    ])
    def test_every_era_boundary_year_lands_in_the_expected_band(self, year, expected):
        """Explicit boundary pins: both the closing year of each era and the
        opening year of the next. `right=True` makes each `hi` inclusive and each
        `lo` exclusive-of-the-previous-band, which is what ERAS means."""
        assert A.era_name(year) == expected
        assert A.era_series(pd.Series([year])).astype(object).iloc[0] == expected

    def test_years_outside_every_band_are_none_in_both_implementations(self):
        """1788 is one year before the first era starts — the exact value the
        `ERAS[0][1] - 1` left edge is derived from, and it must NOT be in band 1."""
        outside = pd.Series([1700, 1788, 2027, 3000])

        assert [A.era_name(y) for y in outside] == [None] * 4
        assert A.era_series(outside).isna().all()

    def test_era_name_tolerates_nan_and_none(self):
        assert A.era_name(np.nan) is None
        assert A.era_name(None) is None
        assert A.era_name(1900.0) == "The Gilded Age"

    def test_era_span_label_collapses_a_single_era_and_joins_two(self):
        assert A.era_span_label(1789, 1815) == "The founding"
        assert A.era_span_label(1789, 1900) == "The founding → The Gilded Age"

    def test_era_span_label_is_none_when_either_endpoint_is_out_of_band(self):
        assert A.era_span_label(1700, 1900) is None
        assert A.era_span_label(1900, 3000) is None
        assert A.era_span_label(np.nan, 1900) is None


# =========================================================================== #
class TestLabelNormalization:
    """PIN 7: case-variant dedup, raise-on-unmapped, and the injectivity guard."""

    def test_load_taxonomy_round_trips_a_json_file(self, tmp_path):
        path = tmp_path / "taxonomy.json"
        tax = _taxonomy({"Alpha": "Domain"})
        path.write_text(json.dumps(tax))

        assert A.load_taxonomy(path) == tax

    def test_label_map_folds_every_level2_name_to_its_canonical_spelling(self):
        label_map = A.canonical_label_map(_taxonomy({"War of 1812": "D", "Alpha": "D"}))

        assert label_map["war of 1812"] == "War of 1812"
        assert label_map["alpha"] == "Alpha"
        assert len(label_map) == 2

    def test_non_injective_casefold_raises_rather_than_merging_two_topics(self):
        """If two canonical names differed only by case, normalizing would
        silently merge two distinct topics — worse than the fragmentation the
        normalization exists to fix, so it must refuse."""
        tax = _taxonomy({"War of 1812": "D", "War Of 1812": "D"})

        with pytest.raises(ValueError, match="not 1:1"):
            A.canonical_label_map(tax)

    def test_two_case_variants_on_one_paragraph_count_once(self):
        """The measured defect this normalization exists for: 111 of 52,855 label
        assignments are case variants. A paragraph listing both spellings is ONE
        (paragraph, topic) pair, not two."""
        label_map = A.canonical_label_map(_taxonomy({"War of 1812": "D"}))

        assert A.normalize_topics(["War of 1812", "War Of 1812"], label_map) == ["War of 1812"]

    def test_normalization_is_order_stable_and_dedups_exact_repeats(self):
        label_map = A.canonical_label_map(_taxonomy({"Beta": "D", "Alpha": "D"}))

        assert A.normalize_topics(["Beta", "Alpha", "BETA"], label_map) == ["Beta", "Alpha"]

    def test_an_unmapped_label_raises_instead_of_inventing_a_51st_topic(self):
        """Correction #2 of the task context is explicit: raise, don't warn. An
        unmapped label means the annotations and the taxonomy have diverged."""
        label_map = A.canonical_label_map(_taxonomy({"Alpha": "D"}))

        with pytest.raises(ValueError, match="not in taxonomy_v1"):
            A.normalize_topics(["Alpha", "Not A Real Topic"], label_map)

    def test_none_and_empty_topic_lists_normalize_to_no_assignments(self):
        """The 404 paragraphs with an empty `topics` list are real rows; they
        contribute zero assignments but stay in the denominator."""
        label_map = A.canonical_label_map(_taxonomy({"Alpha": "D"}))

        assert A.normalize_topics(None, label_map) == []
        assert A.normalize_topics([], label_map) == []
        assert A.normalize_topics(np.array([], dtype=object), label_map) == []


# =========================================================================== #
class TestLoadInputs:
    """`load_inputs` is where the repo's keyed-merge discipline lives. These
    tests prove `_require_full_merge` is actually WIRED IN, not just
    `validate="one_to_one"` — the two catch different failures."""

    TAX = staticmethod(lambda: _taxonomy({"Alpha": "Domain A", "Beta": "Domain B"}))

    def _install(self, monkeypatch, tmp_path, redirect, *, annotations=None,
                 speeches=None, issues=None):
        annotations = annotations if annotations is not None else _annotation_frame([
            ("doc-a", 0, ["Alpha"]),
            ("doc-a", 1, ["alpha", "ALPHA"]),   # two case variants of one topic
            ("doc-b", 0, []),                    # real row, no topic
            ("doc-b", 1, ["Alpha", "Beta"]),
        ])
        speeches = speeches if speeches is not None else _speech_frame([
            ("doc-a", SOTU), ("doc-b", OTHER),
        ])
        issues = issues if issues is not None else _issue_frame([
            ("doc-a", 0, 1800), ("doc-a", 1, 1800),
            ("doc-b", 0, 1900), ("doc-b", 1, 1900),
        ])
        return _install_annotation_parquets(
            monkeypatch, tmp_path, redirect,
            annotations=annotations, speeches=speeches, issues=issues,
        )

    def test_paragraph_universe_keeps_topicless_rows_and_gains_year_era_genre(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """The denominator is ALL paragraphs. `doc-b`/0 carries no topic and must
        still be one row of the paragraph frame, or every share is inflated."""
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs)

        paragraphs, _ = A.load_inputs(self.TAX())

        assert len(paragraphs) == 4
        assert set(paragraphs.columns) == {
            "doc_name", "para_idx", "year", "era", "speech_type"
        }
        assert ("doc-b", 0) in set(zip(paragraphs["doc_name"], paragraphs["para_idx"]))
        assert list(paragraphs["era"].astype(object)) == [
            "The founding", "The founding", "The Gilded Age", "The Gilded Age"
        ]

    def test_assignments_are_one_row_per_paragraph_topic_pair_after_dedup(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """`doc-a`/1 lists "alpha" and "ALPHA" — one canonical assignment.
        `doc-b`/1 is genuinely multi-label and yields two."""
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs)

        _, assignments = A.load_inputs(self.TAX())

        pairs = sorted(
            zip(assignments["doc_name"], assignments["para_idx"], assignments["topic"])
        )
        assert pairs == [
            ("doc-a", 0, "Alpha"),
            ("doc-a", 1, "Alpha"),
            ("doc-b", 1, "Alpha"),
            ("doc-b", 1, "Beta"),
        ]
        assert list(assignments["level1"]) == [
            "Domain A", "Domain A", "Domain A", "Domain B"
        ]
        assert list(assignments["speech_type"]) == [SOTU, SOTU, OTHER, OTHER]
        # The context columns must carry VALUES, not just exist: `era` here is
        # what `bootstrap_era_shares` groups `n_topic_paragraphs` by, so an
        # all-null era column would zero every published per-era sample size
        # without changing a single column name.
        assert list(assignments["year"]) == [1800, 1800, 1900, 1900]
        assert list(assignments["era"].astype(object)) == [
            "The founding", "The founding", "The Gilded Age", "The Gilded Age"
        ]

    def test_annotation_row_missing_from_issues_raises_via_require_full_merge(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """`validate="one_to_one"` would NOT catch this — an inner join just
        drops the row. Only the post-merge length assert does."""
        issues = _issue_frame([("doc-a", 0, 1800), ("doc-a", 1, 1800), ("doc-b", 0, 1900)])
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs, issues=issues)

        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            A.load_inputs(self.TAX())

    def test_extra_issues_row_with_no_annotation_also_raises(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """The guard is symmetric: divergence in EITHER direction is a refusal."""
        issues = _issue_frame([
            ("doc-a", 0, 1800), ("doc-a", 1, 1800),
            ("doc-b", 0, 1900), ("doc-b", 1, 1900), ("doc-c", 0, 1950),
        ])
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs, issues=issues)

        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            A.load_inputs(self.TAX())

    def test_duplicate_key_in_issues_raises_a_merge_error(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """The other half of the discipline: a duplicated key must not fan out."""
        issues = _issue_frame([
            ("doc-a", 0, 1800), ("doc-a", 0, 1800), ("doc-a", 1, 1800),
            ("doc-b", 0, 1900), ("doc-b", 1, 1900),
        ])
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs, issues=issues)

        with pytest.raises(pd.errors.MergeError):
            A.load_inputs(self.TAX())

    def test_a_speech_with_no_genre_annotation_raises(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """A missing speech_type would silently drop those paragraphs from every
        genre treatment, so it is a refusal rather than a NaN."""
        self._install(
            monkeypatch, tmp_path, redirect_annotation_dirs,
            speeches=_speech_frame([("doc-a", SOTU)]),
        )

        with pytest.raises(ValueError, match="no speech_type"):
            A.load_inputs(self.TAX())

    def test_a_year_outside_the_era_bands_raises_rather_than_dropping_silently(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        issues = _issue_frame([
            ("doc-a", 0, 1700), ("doc-a", 1, 1800),
            ("doc-b", 0, 1900), ("doc-b", 1, 1900),
        ])
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs, issues=issues)

        with pytest.raises(ValueError, match="fall outside trends.ERAS"):
            A.load_inputs(self.TAX())

    def test_a_level2_topic_with_no_level1_parent_raises(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs)

        with pytest.raises(ValueError, match="no level-1 parent"):
            A.load_inputs(_taxonomy({"Alpha": None, "Beta": "Domain B"}))

    def test_an_unmapped_annotation_label_propagates_the_normalization_raise(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        annotations = _annotation_frame([
            ("doc-a", 0, ["Alpha"]), ("doc-a", 1, ["Gamma"]),
            ("doc-b", 0, []), ("doc-b", 1, ["Beta"]),
        ])
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs,
                      annotations=annotations)

        with pytest.raises(ValueError, match="not in taxonomy_v1"):
            A.load_inputs(self.TAX())

    def test_row_order_of_the_inputs_does_not_change_the_output(
        self, monkeypatch, tmp_path, redirect_annotation_dirs
    ):
        """The merge is keyed, so shuffling the issues table is self-healing.
        The shuffle is asserted to have actually reordered first, otherwise the
        invariance claim would be vacuous."""
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs)
        paragraphs, assignments = A.load_inputs(self.TAX())

        base = _issue_frame([
            ("doc-a", 0, 1800), ("doc-a", 1, 1800),
            ("doc-b", 0, 1900), ("doc-b", 1, 1900),
        ])
        shuffled = base.iloc[::-1].reset_index(drop=True)
        assert list(shuffled["para_idx"]) != list(base["para_idx"])
        self._install(monkeypatch, tmp_path, redirect_annotation_dirs, issues=shuffled)
        paragraphs2, assignments2 = A.load_inputs(self.TAX())

        pd.testing.assert_frame_equal(paragraphs, paragraphs2)
        pd.testing.assert_frame_equal(assignments, assignments2)


# =========================================================================== #
class TestLevelTopicsAndSlices:
    def test_level_topics_come_from_the_taxonomy_not_the_observed_labels(self):
        """A topic with zero paragraphs in a slice must still get a row rather
        than vanishing, so the list is sourced from the taxonomy."""
        tax = _taxonomy({"Zeta": "D2", "Alpha": "D1"})

        assert A.level_topics(tax, "level2") == ["Alpha", "Zeta"]
        assert A.level_topics(tax, "level1") == ["D1", "D2"]

    def test_an_unknown_level_raises(self):
        with pytest.raises(ValueError, match="level must be one of"):
            A.level_topics(_taxonomy({"Alpha": "D"}), "level3")

    def test_raw_and_genre_standardized_see_the_whole_corpus(self):
        """`genre_standardized` REWEIGHTS, it does not filter — that distinction
        is the whole point of the treatment."""
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["T"]), ("d", 1, 1800, OTHER, [])], {"T": "D"}
        )

        for treatment in ("raw", "genre_standardized"):
            paras, assign = A.slice_for(paragraphs, assignments, treatment)
            assert len(paras) == 2 and len(assign) == 1

    def test_sotu_keeps_only_annual_messages_in_both_frames(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["T"]), ("e", 0, 1800, OTHER, ["T"])], {"T": "D"}
        )

        paras, assign = A.slice_for(paragraphs, assignments, "sotu")

        assert list(paras["doc_name"]) == ["d"]
        assert list(assign["doc_name"]) == ["d"]

    def test_an_unknown_treatment_raises(self):
        paragraphs, assignments = _corpus([("d", 0, 1800, SOTU, [])], {})

        with pytest.raises(ValueError, match="treatment must be one of"):
            A.slice_for(paragraphs, assignments, "sotu-only")


# =========================================================================== #
class TestCountsAndSmoothing:
    def test_year_index_is_the_dense_inclusive_span_of_the_corpus(self):
        """Years with zero paragraphs (1946 and 1950 in the real corpus) must
        exist in the index so the smoothing window can cover them."""
        paragraphs, _ = _corpus(
            [("d", 0, 1800, SOTU, []), ("e", 0, 1804, SOTU, [])], {}
        )

        years = A._year_index(paragraphs)

        assert list(years) == [1800, 1801, 1802, 1803, 1804]
        assert years.name == "year"

    def test_counts_fill_absent_years_and_absent_topics_with_zero(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["T"]), ("d", 1, 1800, SOTU, []),
             ("e", 0, 1802, SOTU, ["T"])],
            {"T": "D"},
        )
        years = A._year_index(paragraphs)

        num, den = A._counts(paragraphs, assignments, "topic", ["T", "Unseen"], years)

        assert list(num.index) == [1800, 1801, 1802]
        assert list(num.columns) == ["T", "Unseen"]
        assert num["T"].tolist() == [1.0, 0.0, 1.0]
        assert num["Unseen"].tolist() == [0.0, 0.0, 0.0]
        assert den.tolist() == [2.0, 0.0, 1.0]

    def test_counts_handle_an_empty_assignment_frame(self):
        """The `sotu` slice can legitimately contain no topic assignments."""
        paragraphs, assignments = _corpus([("d", 0, 1800, SOTU, [])], {})
        years = A._year_index(paragraphs)

        num, den = A._counts(paragraphs, assignments, "topic", ["T"], years)

        assert num["T"].tolist() == [0.0]
        assert den.tolist() == [1.0]

    def test_smooth_is_a_centered_rolling_sum_with_partial_edges(self):
        """Centered + min_periods=1: the edges are partial windows, and the
        window LEADS and LAGS the data by two years — the reason the lifecycle
        clamp in `topic_lifecycle` has to exist."""
        series = pd.Series([0.0, 0.0, 5.0, 0.0, 0.0], index=range(1800, 1805))

        smoothed = A._smooth(series, 5)

        assert smoothed.tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]

    def test_smoothing_sums_numerator_and_denominator_separately(self):
        """`sum(num)/sum(den)`, never a mean of ratios: a 1-paragraph year must
        not shout as loudly as a 3-paragraph one."""
        rows = [("d", 0, 1800, SOTU, ["T"])]
        for year in range(1801, 1805):
            rows += [(f"y{year}", i, year, SOTU, []) for i in range(3)]
        paragraphs, assignments = _corpus(rows, {"T": "D"})

        curves = A.attention_curves(paragraphs, assignments, _taxonomy({"T": "D"}))
        at_1802 = curves.loc[curves["year"] == 1802, "share_smooth"].iloc[0]

        assert at_1802 == pytest.approx(1 / 13)
        assert at_1802 != pytest.approx(0.2)  # what a mean-of-ratios would give


# =========================================================================== #
class TestCombineStrata:
    """PIN 2: the missing-genre-year rule. It must NEVER silently emit NaN on a
    year that actually holds paragraphs."""

    YEARS = pd.RangeIndex(1800, 1805, name="year")
    WEIGHTS = {"A": 0.75, "B": 0.25}

    def _strata(self):
        # 1800 fallback, unequal shares  | 1801 fallback, equal shares
        # 1802 one genre eligible        | 1803 both eligible | 1804 empty
        num_a = pd.DataFrame({"T": [2.0, 1.0, 4.0, 4.0, 0.0]}, index=self.YEARS)
        den_a = pd.Series([2.0, 2.0, 10.0, 10.0, 0.0], index=self.YEARS)
        num_b = pd.DataFrame({"T": [0.0, 1.0, 1.0, 2.0, 0.0]}, index=self.YEARS)
        den_b = pd.Series([1.0, 2.0, 1.0, 4.0, 0.0], index=self.YEARS)
        return {"A": num_a, "B": num_b}, {"A": den_a, "B": den_b}

    def test_a_year_where_no_genre_clears_the_minimum_falls_back_to_present_genres(self):
        """Both strata hold fewer than MIN_GENRE_STRATUM paragraphs, so the
        `den >= min_stratum` rule would zero every weight. The `den > 0` fallback
        must keep the year alive with a finite share."""
        num, den = self._strata()

        out = A._combine_strata(num, den, self.WEIGHTS, A.MIN_GENRE_STRATUM)

        # shares are A=1.0, B=0.0 with renormalized weights 0.75 / 0.25
        assert out.loc[1800, "T"] == pytest.approx(0.75)
        assert np.isfinite(out.loc[1800, "T"])

    def test_fallback_weights_sum_to_one(self):
        """Constructed so both strata have the SAME within-genre share (0.5): the
        combined value can only equal 0.5 if the renormalized weights sum to 1."""
        num, den = self._strata()

        out = A._combine_strata(num, den, self.WEIGHTS, A.MIN_GENRE_STRATUM)

        assert out.loc[1801, "T"] == pytest.approx(0.5)

    def test_an_ineligible_stratum_loses_its_weight_to_the_eligible_one(self):
        """A one-paragraph stratum must not carry its genre's full corpus weight;
        1802's B side (den=1, share=1.0) is excluded entirely, so the answer is
        A's share alone rather than 0.75*0.4 + 0.25*1.0 = 0.55."""
        num, den = self._strata()

        out = A._combine_strata(num, den, self.WEIGHTS, A.MIN_GENRE_STRATUM)

        assert out.loc[1802, "T"] == pytest.approx(0.4)

    def test_two_eligible_strata_combine_at_the_fixed_corpus_weights(self):
        num, den = self._strata()

        out = A._combine_strata(num, den, self.WEIGHTS, A.MIN_GENRE_STRATUM)

        assert out.loc[1803, "T"] == pytest.approx(0.75 * 0.4 + 0.25 * 0.5)

    def test_only_a_year_with_zero_paragraphs_anywhere_is_nan(self):
        """The one NaN the rule is allowed to produce."""
        num, den = self._strata()

        out = A._combine_strata(num, den, self.WEIGHTS, A.MIN_GENRE_STRATUM)

        assert np.isnan(out.loc[1804, "T"])
        assert out["T"].isna().sum() == 1

    def test_min_stratum_of_one_makes_every_present_genre_eligible(self):
        """The unsmoothed per-year `share` is combined at min_stratum=1, which is
        how `attention_curves` calls it — pinned so the two call sites cannot be
        conflated."""
        num, den = self._strata()

        out = A._combine_strata(num, den, self.WEIGHTS, min_stratum=1)

        assert out.loc[1802, "T"] == pytest.approx(0.75 * 0.4 + 0.25 * 1.0)

    def test_a_genre_absent_from_the_weight_map_gets_zero_weight(self):
        num, den = self._strata()

        out = A._combine_strata(num, den, {"A": 1.0}, A.MIN_GENRE_STRATUM)

        assert out.loc[1803, "T"] == pytest.approx(0.4)


# =========================================================================== #
class TestAttentionCurves:
    TAX = staticmethod(lambda: _taxonomy({"Alpha": "Domain A", "Beta": "Domain A",
                                          "Gamma": "Domain B"}))

    def test_denominator_is_every_paragraph_including_the_topicless_ones(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha"]), ("d", 1, 1800, SOTU, []),
             ("d", 2, 1800, SOTU, [])],
            {"Alpha": "Domain A"},
        )

        curves = A.attention_curves(paragraphs, assignments, self.TAX())
        row = curves[(curves["topic"] == "Alpha") & (curves["year"] == 1800)].iloc[0]

        assert row["n_topic"] == 1 and row["n_total"] == 3
        assert row["share"] == pytest.approx(1 / 3)

    def test_topics_with_no_paragraphs_still_get_rows(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha"])], {"Alpha": "Domain A"}
        )

        curves = A.attention_curves(paragraphs, assignments, self.TAX())

        assert set(curves["topic"]) == {"Alpha", "Beta", "Gamma"}
        assert curves.loc[curves["topic"] == "Gamma", "n_topic"].sum() == 0

    def test_level1_rollup_counts_a_multi_topic_paragraph_once_per_domain(self):
        """A paragraph tagged with two level-2 topics from the same domain is ONE
        level-1 paragraph — otherwise a domain's share could exceed 1."""
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha", "Beta"]), ("d", 1, 1800, SOTU, ["Gamma"])],
            {"Alpha": "Domain A", "Beta": "Domain A", "Gamma": "Domain B"},
        )

        curves = A.attention_curves(paragraphs, assignments, self.TAX(), level="level1")
        row = curves[(curves["topic"] == "Domain A") & (curves["year"] == 1800)].iloc[0]

        assert row["n_topic"] == 1
        assert row["share"] == pytest.approx(0.5)

    def test_level2_shares_do_not_sum_to_one_because_labels_are_multi_label(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha", "Beta"])],
            {"Alpha": "Domain A", "Beta": "Domain A"},
        )

        curves = A.attention_curves(paragraphs, assignments, self.TAX())
        total = curves.loc[curves["year"] == 1800, "share"].sum()

        assert total == pytest.approx(2.0)

    def test_a_year_with_no_paragraphs_keeps_its_row_with_a_nan_share(self):
        """pandas>=3 `stack()` retains NA rows; a year whose share is undefined
        must stay in the frame rather than silently vanishing."""
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha"]), ("e", 0, 1802, SOTU, ["Alpha"])],
            {"Alpha": "Domain A"},
        )

        curves = A.attention_curves(paragraphs, assignments, self.TAX())
        blank = curves[(curves["topic"] == "Alpha") & (curves["year"] == 1801)]

        assert len(blank) == 1
        assert np.isnan(blank["share"].iloc[0])
        assert blank["n_total"].iloc[0] == 0

    def test_sotu_treatment_restricts_numerator_and_denominator_together(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha"]), ("d", 1, 1800, SOTU, []),
             ("e", 0, 1800, OTHER, ["Alpha"]), ("e", 1, 1800, OTHER, ["Alpha"])],
            {"Alpha": "Domain A"},
        )

        raw = A.attention_curves(paragraphs, assignments, self.TAX(), treatment="raw")
        sotu = A.attention_curves(paragraphs, assignments, self.TAX(), treatment="sotu")

        raw_row = raw[(raw["topic"] == "Alpha") & (raw["year"] == 1800)].iloc[0]
        sotu_row = sotu[(sotu["topic"] == "Alpha") & (sotu["year"] == 1800)].iloc[0]
        assert raw_row["share"] == pytest.approx(3 / 4)
        assert sotu_row["share"] == pytest.approx(1 / 2)
        assert sotu_row["n_total"] == 2

    def _mix_shift_corpus(self):
        """Corpus mix is 50/50 by genre, but 1800 is 5:1 SOTU and 1900 is 1:5.
        The topic saturates whichever genre dominates that year, so a raw reading
        reports steady 5/6 attention while the standardized one reports 0.5."""
        rows = []
        rows += [("s1800", i, 1800, SOTU, ["Alpha"]) for i in range(5)]
        rows += [("o1800", 0, 1800, OTHER, [])]
        rows += [("s1900", 0, 1900, SOTU, [])]
        rows += [("o1900", i, 1900, OTHER, ["Alpha"]) for i in range(5)]
        return _corpus(rows, {"Alpha": "Domain A"})

    def test_genre_standardization_removes_a_pure_genre_mix_effect(self):
        paragraphs, assignments = self._mix_shift_corpus()

        raw = A.attention_curves(paragraphs, assignments, self.TAX(), treatment="raw")
        std = A.attention_curves(
            paragraphs, assignments, self.TAX(), treatment="genre_standardized"
        )

        def share(frame, year):
            hit = frame[(frame["topic"] == "Alpha") & (frame["year"] == year)]
            return hit["share"].iloc[0]

        assert share(raw, 1800) == pytest.approx(5 / 6)
        assert share(raw, 1900) == pytest.approx(5 / 6)
        assert share(std, 1800) == pytest.approx(0.5)
        assert share(std, 1900) == pytest.approx(0.5)

    def _unequal_mix_corpus(self):
        """Corpus mix is 80/20 SOTU:OTHER, and 1800 is an even 4:4 split.

        `_mix_shift_corpus` above is an exact 50/50 corpus, where "combine at the
        FIXED CORPUS-WIDE genre weights" and "combine at equal weights" give the
        same answer — so it cannot pin where the weights come from. This one can:
        the two candidate rules differ by 0.15 in 1800.
        """
        rows = []
        rows += [("s1800", i, 1800, SOTU, ["Alpha"] if i == 0 else []) for i in range(4)]
        rows += [("o1800", i, 1800, OTHER, ["Alpha"] if i < 3 else []) for i in range(4)]
        rows += [("s1810", i, 1810, SOTU, []) for i in range(12)]
        return _corpus(rows, {"Alpha": "Domain A"})

    def test_strata_are_combined_at_the_corpus_mix_not_at_equal_weights(self):
        """CONTEXT's genre treatment #3 is "combine with FIXED CORPUS-WIDE genre
        weights". With an 80/20 corpus the OTHER stratum's 3/4 must be discounted
        to a fifth of the answer, not half of it."""
        paragraphs, assignments = self._unequal_mix_corpus()

        std = A.attention_curves(
            paragraphs, assignments, self.TAX(), treatment="genre_standardized"
        )
        row = std[(std["topic"] == "Alpha") & (std["year"] == 1800)].iloc[0]

        # 0.2 * (3/4) + 0.8 * (1/4); equal weights would give 0.5, as would the
        # pooled 4/8 that no standardization at all produces.
        assert row["share"] == pytest.approx(0.35)

    def test_smoothed_standardized_share_applies_the_min_genre_stratum(self):
        """The smoothed curve uses MIN_GENRE_STRATUM (3) while the per-year share
        uses 1. In 1800 the OTHER stratum holds a single paragraph, so it drops
        out of the smoothed estimate and the answer is the SOTU share alone."""
        paragraphs, assignments = self._mix_shift_corpus()

        std = A.attention_curves(
            paragraphs, assignments, self.TAX(), treatment="genre_standardized"
        )
        row = std[(std["topic"] == "Alpha") & (std["year"] == 1800)].iloc[0]

        assert row["share_smooth"] == pytest.approx(1.0)
        assert row["share"] == pytest.approx(0.5)

    def test_window_counts_stay_unweighted_under_genre_standardization(self):
        """The share is reweighted but the sufficiency test must still be about
        real sample size, so n_topic_window / n_total_window are raw counts."""
        paragraphs, assignments = self._mix_shift_corpus()

        raw = A.attention_curves(paragraphs, assignments, self.TAX(), treatment="raw")
        std = A.attention_curves(
            paragraphs, assignments, self.TAX(), treatment="genre_standardized"
        )

        cols = ["topic", "year", "n_topic", "n_total", "n_topic_window", "n_total_window"]
        pd.testing.assert_frame_equal(
            raw[cols].reset_index(drop=True), std[cols].reset_index(drop=True)
        )

    def test_output_schema_and_sort_order_are_stable(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha"])], {"Alpha": "Domain A"}
        )

        curves = A.attention_curves(paragraphs, assignments, self.TAX())

        assert list(curves.columns) == [
            "level", "treatment", "topic", "year", "n_topic", "n_total", "share",
            "n_topic_window", "n_total_window", "share_smooth",
        ]
        assert set(curves["level"]) == {"level2"}
        assert set(curves["treatment"]) == {"raw"}
        assert curves[["topic", "year"]].equals(
            curves[["topic", "year"]].sort_values(["topic", "year"])
        )


# =========================================================================== #
class TestCorexCurves:
    def _install_issues(self, monkeypatch, tmp_path, frame):
        path = tmp_path / "paragraph_issues.parquet"
        frame.to_parquet(path, index=False)
        monkeypatch.setattr(A, "PARA_LABELS_PATH", path)
        return path

    def _paragraphs(self):
        paragraphs, _ = _corpus(
            [("d", 0, 1800, SOTU, []), ("d", 1, 1800, SOTU, []),
             ("e", 0, 1800, OTHER, []), ("e", 1, 1800, OTHER, [])],
            {},
        )
        return paragraphs

    def test_columns_are_renamed_to_the_crosswalk_issue_names(self, monkeypatch, tmp_path):
        """`Discovered 5` is read from the parquet but reported under the name
        the crosswalk uses, so `classify_deaths` can index it by parent name."""
        paragraphs = self._paragraphs()
        self._install_issues(monkeypatch, tmp_path, _issue_frame(
            [("d", 0, 1800), ("d", 1, 1800), ("e", 0, 1800), ("e", 1, 1800)],
            fired={("d", 0): ["Discovered 5"]},
        ))

        curves = A.corex_curves(paragraphs)

        assert SECURITY_PEACE in curves.columns
        assert "Discovered 5" not in curves.columns
        assert set(curves.columns) == set(A.COREX_COLUMNS)
        assert curves.loc[1800, SECURITY_PEACE] == pytest.approx(0.25)

    def test_sotu_treatment_restricts_the_slice(self, monkeypatch, tmp_path):
        paragraphs = self._paragraphs()
        self._install_issues(monkeypatch, tmp_path, _issue_frame(
            [("d", 0, 1800), ("d", 1, 1800), ("e", 0, 1800), ("e", 1, 1800)],
            fired={("d", 0): ["Immigration"], ("e", 0): ["Immigration"]},
        ))

        raw = A.corex_curves(paragraphs, "raw")
        sotu = A.corex_curves(paragraphs, "sotu")

        assert raw.loc[1800, "Immigration"] == pytest.approx(0.5)
        assert sotu.loc[1800, "Immigration"] == pytest.approx(0.5)  # 1 of 2 SOTU paras
        assert raw.index.equals(sotu.index)

    def test_genre_standardized_is_deliberately_computed_as_raw(self, monkeypatch, tmp_path):
        """Documented decision: the CorEx side is only a collapse/persist
        indicator, so it is never post-stratified. Pinned so a later "fix" that
        reweights it is a conscious change."""
        paragraphs = self._paragraphs()
        self._install_issues(monkeypatch, tmp_path, _issue_frame(
            [("d", 0, 1800), ("d", 1, 1800), ("e", 0, 1800), ("e", 1, 1800)],
            fired={("d", 0): ["Immigration"]},
        ))

        pd.testing.assert_frame_equal(
            A.corex_curves(paragraphs, "raw"),
            A.corex_curves(paragraphs, "genre_standardized"),
        )

    def test_a_paragraph_missing_from_the_issues_table_raises(self, monkeypatch, tmp_path):
        """`_require_full_merge` is wired in here too — a diverging key set is a
        refusal, not a silently shorter denominator."""
        paragraphs = self._paragraphs()
        self._install_issues(monkeypatch, tmp_path, _issue_frame(
            [("d", 0, 1800), ("d", 1, 1800), ("e", 0, 1800)]
        ))

        with pytest.raises(ValueError, match="corex_curves\\[raw\\]"):
            A.corex_curves(paragraphs)


# =========================================================================== #
class TestSubstantiveMask:
    """PIN 3: the three-leg pre-registered threshold. Each curve below is built
    so exactly ONE leg is the binding constraint in the failing year — a mutation
    that drops any single leg is caught by the test for that leg."""

    YEARS = list(range(1800, 1811))

    def test_window_paragraph_leg_binds_when_the_window_is_too_thin(self):
        """Leg 1: >= MIN_WINDOW_PARAGRAPHS. 1800 has 20 window paragraphs but
        passes both other legs (10 topic paragraphs, share 0.10 >= floor 0.01)."""
        curve = _curve(
            self.YEARS, n_topic=10,
            n_topic_window=10,
            n_total_window=[20] + [100] * 10,
            share_smooth=0.10,
        )

        mask = A.substantive_mask(curve)

        assert not mask.iloc[0]
        assert mask.iloc[1:].all()

    def test_window_paragraph_leg_is_inclusive_at_exactly_the_minimum(self):
        curve = _curve(
            self.YEARS, n_topic=10, n_topic_window=10,
            n_total_window=[29, 30] + [100] * 9, share_smooth=0.10,
        )

        mask = A.substantive_mask(curve)

        assert not mask.iloc[0]
        assert mask.iloc[1]

    def test_topic_paragraph_leg_binds_when_fewer_than_three_carry_the_topic(self):
        """Leg 2: >= MIN_TOPIC_PARAGRAPHS. This is the single-stray-anachronism
        killer. 1800's share (0.02) still clears the 0.01 floor, so leg 3 passes
        and only the topic count can be responsible."""
        topic_window = [2] + [10] * 10
        curve = _curve(
            self.YEARS, n_topic=topic_window, n_topic_window=topic_window,
            n_total_window=100, share_smooth=[n / 100 for n in topic_window],
        )

        mask = A.substantive_mask(curve)

        assert not mask.iloc[0]
        assert mask.iloc[1:].all()

    def test_topic_paragraph_leg_is_inclusive_at_exactly_three(self):
        topic_window = [2, 3] + [10] * 9
        curve = _curve(
            self.YEARS, n_topic=topic_window, n_topic_window=topic_window,
            n_total_window=100, share_smooth=[n / 100 for n in topic_window],
        )

        mask = A.substantive_mask(curve)

        assert not mask.iloc[0]
        assert mask.iloc[1]

    def test_relative_floor_leg_binds_at_a_tenth_of_the_topics_own_peak(self):
        """Leg 3, relative half. 1800's share is 0.005 — comfortably above the
        0.002 absolute floor, so ONLY the 0.10 x peak (= 0.010) rule can exclude
        it. This is what keeps the definition scale-free across topics."""
        topic_window = [5] + [100] * 10
        curve = _curve(
            self.YEARS, n_topic=topic_window, n_topic_window=topic_window,
            n_total_window=1000, share_smooth=[n / 1000 for n in topic_window],
        )

        mask = A.substantive_mask(curve)

        assert curve["share_smooth"].iloc[0] > A.ABS_SHARE_FLOOR
        assert not mask.iloc[0]
        assert mask.iloc[1:].all()

    def test_absolute_floor_leg_binds_for_a_topic_whose_own_peak_is_tiny(self):
        """Leg 3, absolute half. Peak share is 0.010, so the relative floor is
        only 0.001; 1800's 0.0015 clears that but not the 0.002 absolute floor.
        Together with the previous test this pins `max(...)` as a MAX."""
        topic_window = [6] + [40] * 10
        curve = _curve(
            self.YEARS, n_topic=topic_window, n_topic_window=topic_window,
            n_total_window=4000, share_smooth=[n / 4000 for n in topic_window],
        )

        mask = A.substantive_mask(curve)

        peak = curve["share_smooth"].max()
        assert curve["share_smooth"].iloc[0] > A.REL_PEAK_FRACTION * peak
        assert not mask.iloc[0]
        assert mask.iloc[1:].all()

    def test_a_topic_with_no_positive_share_anywhere_is_never_substantive(self):
        curve = _curve(self.YEARS, n_topic=0, n_topic_window=0,
                       n_total_window=1000, share_smooth=0.0)

        assert not A.substantive_mask(curve).any()

    def test_an_all_nan_curve_is_never_substantive(self):
        """The `sotu` treatment genuinely produces all-NaN curves for topics that
        never appear in an annual message."""
        curve = _curve(self.YEARS, n_topic=0, n_topic_window=0,
                       n_total_window=0, share_smooth=np.nan)

        mask = A.substantive_mask(curve)

        assert not mask.any()
        assert mask.index.equals(curve.index)


# =========================================================================== #
class TestTopicLifecycle:
    RECORD = list(range(1789, 2027))

    def _record_curve(self, spans, *, n_total_window=1000, topic_paragraphs=100,
                      present_spans=None):
        """A full-record curve substantive exactly across `spans` (inclusive
        year ranges). `present_spans` controls where n_topic > 0 independently,
        which is how the centered-window clamp is exercised."""
        years = np.array(self.RECORD)
        topic_window = np.zeros(len(years))
        for lo, hi in spans:
            topic_window[(years >= lo) & (years <= hi)] = topic_paragraphs
        present = np.zeros(len(years))
        for lo, hi in (present_spans if present_spans is not None else spans):
            present[(years >= lo) & (years <= hi)] = 1
        return _curve(
            years, n_topic=present, n_topic_window=topic_window,
            n_total_window=n_total_window,
            share_smooth=topic_window / n_total_window,
        )

    def test_centered_window_clamp_moves_the_birth_to_the_first_real_paragraph(self):
        """PIN 4. The 5-year centered window LEADS the data by two years, so the
        smoothed mask turns on in 1808 while the topic's first actual paragraph
        is in 1810. A birth two years before its own evidence would be a
        smoothing artifact wearing the costume of a finding."""
        curve = self._record_curve([(1808, 1900)], present_spans=[(1810, 1900)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert bool(A.substantive_mask(curve).loc[curve["year"] == 1808].iloc[0])
        assert out["first_year"] == 1810

    def test_the_clamp_is_symmetric_at_the_end_of_a_topics_life(self):
        """The window LAGS as well as leads: the death cannot postdate the last
        real paragraph either."""
        curve = self._record_curve([(1810, 1902)], present_spans=[(1810, 1900)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["last_year"] == 1900

    def test_the_clamp_falling_outside_every_substantive_year_falls_back(self):
        """`attention.py:794` is LIVE CODE, not a dead defensive branch — SIMPLIFY
        refused to delete it and this is the witness.

        Both substantive years sit at the OUTER edge of the centered window that
        made them substantive: the topic is substantive only in 1800 and 1830 but
        only ever APPEARS in 1802 and 1828. The clamp therefore narrows the span
        to [1802, 1828], which contains neither substantive year, and `live` goes
        empty. Without the fallback the `idxmax` immediately below raises
        `ValueError: attempt to get argmax of an empty sequence`, so this asserts
        the un-clamped span is restored rather than the call blowing up."""
        curve = self._record_curve([(1800, 1800), (1830, 1830)],
                                   present_spans=[(1802, 1802), (1828, 1828)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        # The clamp really does empty the frame — otherwise this pins nothing.
        mask = A.substantive_mask(curve)
        assert curve.loc[mask, "year"].tolist() == [1800, 1830]
        assert curve.loc[curve["n_topic"] > 0, "year"].tolist() == [1802, 1828]
        assert curve[mask]["year"].between(1802, 1828).sum() == 0

        assert out["first_year"] == 1800
        assert out["last_year"] == 1830

    def test_a_clamp_boundary_that_is_not_itself_a_substantive_year_does_not_raise(self):
        """REVIEW MAJOR 1: the PARTIAL clamp miss, which the total-miss fallback
        above does NOT cover.

        The sibling test covers the clamp emptying `live` entirely. This is the
        other half: `live` SURVIVES the clamp, but the clamped boundary is not
        one of the surviving substantive years. Mask is
        [1800, 1803, 1804, 1828, 1829, 1830] and the topic only ever appears in
        1802 and 1828, so the span clamps to [1802, 1828] and `live` becomes
        {1803, 1804, 1828} — non-empty, so the fallback never fires, yet 1802 is
        not in it. The `first_share` lookup then did `.iloc[0]` on an empty
        selection and raised `IndexError: single positional indexer is
        out-of-bounds`. Verified to raise against the pre-fix source and to pass
        after it; the reviewer reached the identical state end-to-end through
        `attention_curves` from a paragraph corpus, so this is reachable through
        the public API and not an artifact of a hand-built curve.

        The fix recomputes BOTH boundaries from the clamped frame, so the reported
        `first_year` is the first substantive year that actually survived."""
        curve = self._record_curve([(1800, 1800), (1803, 1804), (1828, 1830)],
                                   present_spans=[(1802, 1802), (1828, 1828)])

        # The precondition, asserted so this cannot quietly stop pinning anything:
        # the clamp must leave `live` NON-empty (else it is the sibling test) with
        # the lower boundary outside it (else there was never a bug to fix).
        mask = A.substantive_mask(curve)
        assert curve.loc[mask, "year"].tolist() == [1800, 1803, 1804, 1828, 1829, 1830]
        assert curve.loc[curve["n_topic"] > 0, "year"].tolist() == [1802, 1828]
        clamped = curve[mask][curve[mask]["year"].between(1802, 1828)]
        assert clamped["year"].tolist() == [1803, 1804, 1828]
        assert 1802 not in set(clamped["year"])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["first_year"] == 1803
        assert out["last_year"] == 1828
        # The boundaries must be real substantive years, and the shares must be
        # the shares OF those years — the bug was reading a year that isn't there.
        assert out["first_share"] == pytest.approx(
            float(curve.loc[curve["year"] == 1803, "share_smooth"].iloc[0])
        )
        assert out["last_share"] == pytest.approx(
            float(curve.loc[curve["year"] == 1828, "share_smooth"].iloc[0])
        )

    def test_without_a_leading_paragraph_gap_the_first_substantive_year_stands(self):
        """Control for the clamp test: when the mask and the evidence start
        together the clamp is a no-op, so the clamp is not just clipping."""
        curve = self._record_curve([(1810, 1900)])

        assert A.topic_lifecycle(curve, n_paragraphs=500)["first_year"] == 1810

    def test_a_topic_last_substantive_40_years_before_the_record_ends_has_died(self):
        curve = self._record_curve([(1800, 1986)])

        assert A.topic_lifecycle(curve, n_paragraphs=500)["lifecycle_class"] == "died"

    def test_a_topic_substantive_one_year_later_has_not_died(self):
        """DEATH_GAP_YEARS boundary: `last_year <= record_end - 40` is inclusive."""
        curve = self._record_curve([(1800, 1987)])

        assert A.topic_lifecycle(curve, n_paragraphs=500)["lifecycle_class"] != "died"

    def test_a_30_year_internal_gap_makes_a_topic_revived(self):
        curve = self._record_curve([(1850, 1900), (1931, 2026)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["max_gap_years"] == 30
        assert out["lifecycle_class"] == "revived"

    def test_death_takes_precedence_over_revival(self):
        """Documented precedence: died > revived > born > persistent."""
        curve = self._record_curve([(1800, 1810), (1900, 1910)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["max_gap_years"] >= A.REVIVAL_GAP_YEARS
        assert out["lifecycle_class"] == "died"

    def test_revival_takes_precedence_over_birth(self):
        curve = self._record_curve([(1850, 1860), (1990, 2026)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["first_year"] >= 1789 + A.BIRTH_GAP_YEARS
        assert out["lifecycle_class"] == "revived"

    def test_a_topic_first_substantive_40_years_into_the_record_was_born(self):
        curve = self._record_curve([(1829, 2026)])

        assert A.topic_lifecycle(curve, n_paragraphs=500)["lifecycle_class"] == "born"

    def test_a_topic_substantive_from_the_start_to_the_end_is_persistent(self):
        curve = self._record_curve([(1789, 2026)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["lifecycle_class"] == "persistent"
        assert out["max_gap_years"] == 0

    def test_descriptive_fields_come_from_the_smoothed_curve(self):
        years = np.array(self.RECORD)
        topic_window = np.where((years >= 1850) & (years <= 1900), 40.0, 0.0)
        topic_window[years == 1875] = 100.0
        curve = _curve(
            years, n_topic=(topic_window > 0).astype(float),
            n_topic_window=topic_window, n_total_window=1000,
            share_smooth=topic_window / 1000,
        )

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert (out["first_year"], out["peak_year"], out["last_year"]) == (1850, 1875, 1900)
        assert out["peak_share"] == pytest.approx(0.10)
        assert out["first_share"] == pytest.approx(0.04)
        assert out["peak_era"] == "Civil War & Reconstruction"
        assert out["era_of_relevance"] == "Civil War & Reconstruction → The Gilded Age"
        assert out["n_substantive_years"] == 51
        assert out["rise_rate_per_decade"] == pytest.approx((0.10 - 0.04) / 25 * 10)
        assert out["fall_rate_per_decade"] == pytest.approx((0.10 - 0.04) / 25 * 10)

    def test_the_rise_rate_is_nan_when_the_peak_is_the_first_year(self):
        """A flat curve peaks on its first substantive year, so there is no rise
        to measure; the fall is still defined (and flat, hence 0)."""
        curve = self._record_curve([(1800, 1900)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["peak_year"] == out["first_year"] == 1800
        assert np.isnan(out["rise_rate_per_decade"])
        assert out["fall_rate_per_decade"] == pytest.approx(0.0)

    def test_both_rates_are_nan_for_a_single_substantive_year(self):
        curve = self._record_curve([(1900, 1900)])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["first_year"] == out["peak_year"] == out["last_year"] == 1900
        assert np.isnan(out["rise_rate_per_decade"])
        assert np.isnan(out["fall_rate_per_decade"])

    def test_a_never_substantive_topic_is_absent_with_all_fields_nan(self):
        curve = self._record_curve([])

        out = A.topic_lifecycle(curve, n_paragraphs=500)

        assert out["lifecycle_class"] == "absent"
        assert out["n_substantive_years"] == 0
        assert all(np.isnan(out[c]) for c in ("first_year", "peak_year", "last_year"))
        assert out["peak_era"] is None and out["era_of_relevance"] is None

    def test_the_under_powered_guard_replaces_the_class_but_keeps_the_evidence(self):
        """This guard flags NOTHING at level 2 on the real corpus (thinnest topic
        is 61 paragraphs), so this synthetic test is the only thing keeping it
        alive. The descriptive columns must survive — they ARE the evidence that
        the topic is thin."""
        curve = self._record_curve([(1850, 1900)])

        out = A.topic_lifecycle(curve, n_paragraphs=A.UNDER_POWERED_N - 1)

        assert out["under_powered"] is True
        assert out["lifecycle_class"] == "under_powered"
        assert out["first_year"] == 1850 and out["last_year"] == 1900
        assert np.isfinite(out["peak_share"])

    def test_exactly_the_threshold_number_of_paragraphs_is_powered_enough(self):
        curve = self._record_curve([(1850, 1900)])

        out = A.topic_lifecycle(curve, n_paragraphs=A.UNDER_POWERED_N)

        assert out["under_powered"] is False
        assert out["lifecycle_class"] == "died"

    def test_a_thin_and_never_substantive_topic_is_reported_under_powered(self):
        """The early-return path also has to honour the guard, otherwise a thin
        topic could still be charted as merely 'absent'."""
        curve = self._record_curve([])

        out = A.topic_lifecycle(curve, n_paragraphs=1)

        assert out["lifecycle_class"] == "under_powered"
        assert out["n_substantive_years"] == 0

    def test_row_order_of_the_curve_does_not_change_the_result(self):
        curve = self._record_curve([(1850, 1900)])
        shuffled = curve.sample(frac=1, random_state=3).reset_index(drop=True)
        assert list(shuffled["year"]) != list(curve["year"])

        assert A.topic_lifecycle(shuffled, 500) == A.topic_lifecycle(curve, 500)


# =========================================================================== #
class TestExtractLifecycles:
    def test_one_row_per_topic_with_level_and_treatment_carried_through(self):
        paragraphs, assignments = _corpus(
            [("d", 0, 1800, SOTU, ["Alpha"]), ("d", 1, 1800, SOTU, ["Beta"])],
            {"Alpha": "Domain A", "Beta": "Domain A"},
        )
        tax = _taxonomy({"Alpha": "Domain A", "Beta": "Domain A"})
        curves = A.attention_curves(paragraphs, assignments, tax, treatment="sotu")

        life = A.extract_lifecycles(curves, pd.Series({"Alpha": 60, "Beta": 10}))

        assert list(life["topic"]) == ["Alpha", "Beta"]
        assert set(life["level"]) == {"level2"} and set(life["treatment"]) == {"sotu"}
        assert list(life["n_paragraphs"]) == [60, 10]
        assert list(life["under_powered"]) == [False, True]

    def test_a_topic_absent_from_the_totals_index_counts_as_zero_paragraphs(self):
        paragraphs, assignments = _corpus([("d", 0, 1800, SOTU, ["Alpha"])],
                                          {"Alpha": "Domain A"})
        tax = _taxonomy({"Alpha": "Domain A", "Gamma": "Domain B"})
        curves = A.attention_curves(paragraphs, assignments, tax)

        life = A.extract_lifecycles(curves, pd.Series({"Alpha": 60}))
        gamma = life[life["topic"] == "Gamma"].iloc[0]

        assert gamma["n_paragraphs"] == 0
        assert gamma["lifecycle_class"] == "under_powered"


# =========================================================================== #
class TestInvertCrosswalk:
    def test_a_topic_under_two_legacy_parents_keeps_both_sorted(self):
        """The named many-to-many case from the spec: `Chinese Immigration &
        Exclusion` sits under both Immigration and Civil rights & race. Taking
        the first parent would let crosswalk ordering decide a death claim."""
        crosswalk = _crosswalk({
            "Immigration": ["Chinese Immigration & Exclusion", "Alpha"],
            "Civil rights & race": ["Chinese Immigration & Exclusion"],
        })

        inverted = A.invert_crosswalk(crosswalk)

        assert inverted["Chinese Immigration & Exclusion"] == [
            "Civil rights & race", "Immigration"
        ]
        assert inverted["Alpha"] == ["Immigration"]

    def test_a_topic_named_twice_under_one_issue_is_not_duplicated(self):
        inverted = A.invert_crosswalk(_crosswalk({"Immigration": ["Alpha", "Alpha"]}))

        assert inverted["Alpha"] == ["Immigration"]

    def test_a_topic_in_no_mapping_is_simply_absent(self):
        inverted = A.invert_crosswalk(_crosswalk({"Immigration": ["Alpha"]}))

        assert "Beta" not in inverted
        assert inverted.get("Beta", []) == []


# =========================================================================== #
class TestDeclineRatioAndPearson:
    YEARS = pd.RangeIndex(1800, 1901, name="year")

    def test_ratio_is_post_death_mean_over_peak_window_mean(self):
        """The denominator is the MEAN over the peak WINDOW (peak +/- 10), not
        the single peak year — a one-year denominator would be noise. The curve
        here spikes only at 1810, so a peak-year denominator would answer 0.2
        while the windowed one answers 0.2 / (9/21)."""
        series = _series(
            self.YEARS,
            [(1800, 1820, 0.4), (1810, 1810, 1.0), (1821, 1850, 0.9),
             (1851, 1900, 0.2)],
        )

        ratio = A._decline_ratio(series, 1810, 1850)

        assert ratio == pytest.approx(0.2 / (9 / 21))
        assert ratio != pytest.approx(0.2 / 1.0)  # what a peak-year base gives

    def test_the_peak_window_half_width_is_what_selects_the_denominator(self):
        """Values outside peak_year +/- PEAK_WINDOW_HALF must not enter the base;
        here everything outside [1800, 1820] is ~200x larger and would swamp it."""
        series = _series(
            self.YEARS,
            [(1800, 1809, 0.4), (1810, 1820, 0.6), (1851, 1900, 0.5)],
            default=100.0,
        )

        ratio = A._decline_ratio(series, 1810, 1850)

        assert ratio == pytest.approx(0.5 / (10.6 / 21))
        assert ratio > 0.9  # a base polluted by the 100.0 years would be ~0.01

    def test_a_ratio_above_one_signals_a_curve_that_never_tracked_the_topic(self):
        series = _series(self.YEARS, [(1800, 1820, 0.1), (1851, 1900, 0.4)])

        assert A._decline_ratio(series, 1810, 1850) == pytest.approx(4.0)

    def test_an_empty_post_window_is_nan(self):
        series = _series(self.YEARS, [(1800, 1820, 1.0)])

        assert np.isnan(A._decline_ratio(series, 1810, 1900))

    def test_an_empty_peak_window_is_nan(self):
        series = _series(self.YEARS, [(1800, 1900, 1.0)])

        assert np.isnan(A._decline_ratio(series, 1500, 1850))

    def test_a_zero_or_nan_base_is_nan_rather_than_infinity(self):
        zero_base = _series(self.YEARS, [(1851, 1900, 0.5)])
        nan_base = _series(self.YEARS, [(1851, 1900, 0.5)], default=np.nan)

        assert np.isnan(A._decline_ratio(zero_base, 1810, 1850))
        assert np.isnan(A._decline_ratio(nan_base, 1810, 1850))

    def test_pearson_is_computed_over_the_overlapping_defined_years(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0], index=[1, 2, 3, 4])
        b = pd.Series([4.0, 3.0, 2.0, 1.0], index=[1, 2, 3, 4])

        assert A._pearson(a, b) == pytest.approx(-1.0)

    def test_pearson_is_nan_with_fewer_than_three_shared_points(self):
        a = pd.Series([1.0, 2.0, np.nan], index=[1, 2, 3])
        b = pd.Series([2.0, 1.0, 5.0], index=[1, 2, 3])

        assert np.isnan(A._pearson(a, b))

    def test_pearson_is_nan_for_a_flat_curve(self):
        a = pd.Series([1.0, 1.0, 1.0, 1.0], index=[1, 2, 3, 4])
        b = pd.Series([4.0, 3.0, 2.0, 1.0], index=[1, 2, 3, 4])

        assert np.isnan(A._pearson(a, b))
        assert np.isnan(A._pearson(b, a))


# =========================================================================== #
class _SuccessorWorld:
    """A dying topic, a plausible heir, and the knobs each filter turns."""

    YEARS = pd.RangeIndex(1800, 1901, name="year")
    DYING = "Dying Topic"
    HEIR = "Heir Topic"

    def __init__(self, **overrides):
        self.parents = {self.DYING: "Domain A", self.HEIR: "Domain A"}
        self.inverted = {self.DYING: ["Immigration"], self.HEIR: ["Immigration"]}
        heir = dict(topic=self.HEIR, lifecycle_class="born", first_year=1830.0,
                    last_year=1900.0, peak_year=1890.0)
        heir.update(overrides)
        self.heir_row = heir
        falling = _series(self.YEARS, [(1800, 1820, 1.0), (1821, 1850, 0.5)])
        rising = _series(self.YEARS, [(1830, 1900, 1.0)])
        self.curves = {self.DYING: falling, self.HEIR: rising}

    @property
    def dying_row(self):
        return dict(topic=self.DYING, lifecycle_class="died", first_year=1800.0,
                    peak_year=1810.0, last_year=1850.0)

    @property
    def lifecycles(self):
        return pd.DataFrame([self.dying_row, self.heir_row])

    @property
    def smoothed(self):
        return pd.DataFrame(self.curves)

    def find(self, **kw):
        return A.find_successor(
            pd.Series(self.dying_row), self.lifecycles, self.smoothed,
            self.parents, self.inverted, **kw,
        )


class TestFindSuccessor:
    def test_all_four_filters_passing_names_the_heir_with_its_correlation(self):
        world = _SuccessorWorld()
        expected_r = A._pearson(world.smoothed[world.DYING], world.smoothed[world.HEIR])

        successor, r = world.find()

        assert successor == _SuccessorWorld.HEIR
        assert r == pytest.approx(expected_r)
        assert r < 0

    def test_filter_1_rejects_a_candidate_from_another_level1_domain(self):
        world = _SuccessorWorld()
        world.parents[_SuccessorWorld.HEIR] = "Domain B"

        assert world.find() == (None, pytest.approx(np.nan, nan_ok=True))

    def test_filter_1_is_lifted_for_the_cross_domain_diagnostic(self):
        """`same_domain=False` is a DIAGNOSTIC only — it is recorded but never
        used to classify, because spurious anti-correlation is trivial to find
        over a 240-year record."""
        world = _SuccessorWorld()
        world.parents[_SuccessorWorld.HEIR] = "Domain B"

        assert world.find(same_domain=False)[0] == _SuccessorWorld.HEIR

    def test_filter_2_rejects_a_birth_before_the_decline_window_opens(self):
        """The handoff test: a topic already substantive at the dying topic's
        peak did not take over from it."""
        world = _SuccessorWorld(first_year=1805.0)

        assert world.find()[0] is None

    def test_filter_2_rejects_a_birth_after_the_dying_topic_is_gone(self):
        """This is what separates a rename from mere sequence — Vietnam is born
        40 years after World War I ends and must not be its 'successor'."""
        world = _SuccessorWorld(first_year=1870.0)

        assert world.find()[0] is None

    def test_filter_2_is_inclusive_at_both_ends_of_the_decline_window(self):
        assert _SuccessorWorld(first_year=1810.0).find()[0] == _SuccessorWorld.HEIR
        assert _SuccessorWorld(first_year=1850.0).find()[0] == _SuccessorWorld.HEIR

    def test_filter_3_rejects_a_candidate_that_does_not_outlive_the_dying_topic(self):
        assert _SuccessorWorld(last_year=1850.0).find()[0] is None
        assert _SuccessorWorld(last_year=1851.0).find()[0] == _SuccessorWorld.HEIR

    def test_filter_4_rejects_a_candidate_sharing_no_legacy_crosswalk_parent(self):
        """The coarse lexical taxonomy must at least agree the two are the same
        kind of concern."""
        world = _SuccessorWorld()
        world.inverted[_SuccessorWorld.HEIR] = ["Health care"]

        assert world.find()[0] is None

    def test_filter_4_accepts_a_partial_overlap_of_legacy_parents(self):
        world = _SuccessorWorld()
        world.inverted[_SuccessorWorld.HEIR] = ["Health care", "Immigration"]

        assert world.find()[0] == _SuccessorWorld.HEIR

    def test_filter_4_is_SKIPPED_when_the_dying_topic_has_no_crosswalk_parent(self):
        """PINS CURRENT BEHAVIOUR, WHICH CONTRADICTS THE DOCSTRING.

        `find_successor` reads `if legacy and not legacy & ...` — when the DYING
        topic has no crosswalk parent, `legacy` is empty and falsy, so filter 4
        never runs and a candidate sharing nothing is accepted. The docstring
        promises "four independent filters". Carry-forward defect #10.

        SIMPLIFY (2026-07-21) applied the fix and then REVERTED it: contrary to
        the work order's premise it is NOT output-neutral. Regenerating with
        filter 4 unconditional moves `topic_lifecycles.parquet` off sha
        `825fbe66...` — `Constitutional Union & Federalism` flips `rename` ->
        `unresolved_death` under the `sotu` treatment (losing successor `Civil
        Service Reform & the Merit System`, r=-0.278) and six
        `cross_domain_candidate` diagnostics go NaN across all three treatments.
        That is a methodological decision about a published finding, not a
        behaviour-preserving cleanup, so it was escalated rather than absorbed.
        `find_successor`'s own docstring now records the deviation instead of
        promising four unconditional filters."""
        world = _SuccessorWorld()
        world.inverted[_SuccessorWorld.DYING] = []
        world.inverted[_SuccessorWorld.HEIR] = ["Health care"]

        assert world.find()[0] == _SuccessorWorld.HEIR

    def test_filter_4_rejects_even_a_candidate_the_crosswalk_does_not_map(self):
        """The mirror case: an UNMAPPED candidate shares nothing either, so it
        cannot be a successor no matter how well it clears filters 1-3."""
        world = _SuccessorWorld()
        world.inverted[_SuccessorWorld.HEIR] = []

        assert world.find()[0] is None

    def test_a_positively_correlated_candidate_is_rejected(self):
        world = _SuccessorWorld()
        world.curves[_SuccessorWorld.HEIR] = world.curves[_SuccessorWorld.DYING] * 2

        assert world.find()[0] is None

    def test_a_candidate_that_is_absent_or_under_powered_is_skipped(self):
        for cls in ("absent", "under_powered"):
            assert _SuccessorWorld(lifecycle_class=cls).find()[0] is None

    def test_a_candidate_with_a_nan_first_or_last_year_is_skipped(self):
        assert _SuccessorWorld(first_year=np.nan).find()[0] is None
        assert _SuccessorWorld(last_year=np.nan).find()[0] is None

    def test_no_qualifying_candidate_returns_none_and_a_nan_correlation(self):
        world = _SuccessorWorld(lifecycle_class="absent")

        successor, r = world.find()

        assert successor is None
        assert np.isnan(r)

    def test_the_most_anticorrelated_survivor_wins(self):
        """Ranking by size instead would just elect the biggest topic in the
        domain, so the tie-break has to be the correlation."""
        world = _SuccessorWorld()
        rival = "Rival Topic"
        world.parents[rival] = "Domain A"
        world.inverted[rival] = ["Immigration"]
        # rival is only weakly anti-correlated with the dying curve
        world.curves[rival] = _series(
            _SuccessorWorld.YEARS, [(1800, 1820, 0.8), (1830, 1900, 1.0)]
        )
        rows = [world.dying_row, world.heir_row,
                dict(topic=rival, lifecycle_class="born", first_year=1830.0,
                     last_year=1900.0, peak_year=1890.0)]
        lifecycles = pd.DataFrame(rows)
        smoothed = pd.DataFrame(world.curves)

        best, r = A.find_successor(
            pd.Series(world.dying_row), lifecycles, smoothed,
            world.parents, world.inverted,
        )

        r_heir = A._pearson(smoothed[_SuccessorWorld.DYING],
                            smoothed[_SuccessorWorld.HEIR])
        r_rival = A._pearson(smoothed[_SuccessorWorld.DYING], smoothed[rival])
        assert r_heir < r_rival < 0, "fixture must offer a genuinely weaker rival"
        assert (best, r) == (_SuccessorWorld.HEIR, pytest.approx(r_heir))


# =========================================================================== #
class TestClassifyDeaths:
    """PINS 5 and 6: the multi-parent CorEx rule and successor precedence."""

    YEARS = pd.RangeIndex(1800, 1901, name="year")
    DYING = "Dying Topic"
    HEIR = "Heir Topic"

    def _corex(self, ratios: dict[str, float]) -> pd.DataFrame:
        """One column per legacy parent whose decline ratio is EXACTLY `ratio`:
        the peak window [1800, 1820] averages 1.0 and the post window
        [1851, 1900] averages `ratio`."""
        return pd.DataFrame({
            parent: _series(self.YEARS, [(1800, 1820, 1.0), (1821, 1850, 0.5),
                                         (1851, 1900, ratio)])
            for parent, ratio in ratios.items()
        })

    def _world(self, ratios, *, heir_first_year):
        falling = _series(self.YEARS, [(1800, 1820, 1.0), (1821, 1850, 0.5)])
        rising = _series(self.YEARS, [(1830, 1900, 1.0)])
        lifecycles = pd.DataFrame([
            dict(topic=self.DYING, lifecycle_class="died", first_year=1800.0,
                 peak_year=1810.0, last_year=1850.0),
            dict(topic=self.HEIR, lifecycle_class="born",
                 first_year=heir_first_year, last_year=1900.0, peak_year=1890.0),
        ])
        curves = _curves_long({self.DYING: falling, self.HEIR: rising})
        # sorted, exactly as `invert_crosswalk` hands the parents over
        inverted = {self.DYING: sorted(ratios), self.HEIR: sorted(ratios) or ["Immigration"]}
        parents = {self.DYING: "Domain A", self.HEIR: "Domain A"}
        return A.classify_deaths(lifecycles, curves, self._corex(ratios),
                                 inverted, parents)

    def _dying_row(self, out):
        return out[out["topic"] == self.DYING].iloc[0]

    def test_one_persisting_parent_out_of_two_means_corex_persists(self):
        """PIN 5a. Ratios {0.10, 0.90}: a true death needs EVERY parent to
        collapse, so one survivor is enough for `corex_persists=True`. A
        first-parent or majority rule would answer False here."""
        out = self._world({"Immigration": 0.10, "Civil rights & race": 0.90},
                          heir_first_year=1870.0)
        row = self._dying_row(out)

        assert row["corex_persists"] is True
        assert row["successor_topic"] is None
        assert row["rename_class"] == "unresolved_death"

    def test_every_parent_collapsing_means_corex_does_not_persist(self):
        """PIN 5b. Ratios {0.10, 0.20}, both at or under COREX_COLLAPSE_RATIO."""
        out = self._world({"Immigration": 0.10, "Civil rights & race": 0.20},
                          heir_first_year=1870.0)
        row = self._dying_row(out)

        assert row["corex_persists"] is False
        assert row["rename_class"] == "true_death"

    def test_no_crosswalk_parent_is_treated_as_persisting(self):
        """PIN 5c. With no usable parent the cross-check is UNAVAILABLE, and the
        conservative direction is to refuse to claim the two labelers agreed."""
        out = self._world({}, heir_first_year=1870.0)
        row = self._dying_row(out)

        assert row["corex_persists"] is True
        assert row["legacy_parents"] is None
        assert np.isnan(row["corex_decline_ratio"])
        assert row["rename_class"] == "unresolved_death"

    def test_the_collapse_threshold_is_strict_at_exactly_the_ratio(self):
        """`any(v > COREX_COLLAPSE_RATIO)`: a parent sitting exactly on 0.25 has
        collapsed, one hair above has not."""
        assert self._dying_row(self._world(
            {"Immigration": A.COREX_COLLAPSE_RATIO}, heir_first_year=1870.0
        ))["corex_persists"] is False
        assert self._dying_row(self._world(
            {"Immigration": A.COREX_COLLAPSE_RATIO + 0.01}, heir_first_year=1870.0
        ))["corex_persists"] is True

    def test_every_parent_ratio_is_recorded_so_the_rule_can_be_relitigated(self):
        out = self._world({"Immigration": 0.10, "Civil rights & race": 0.90},
                          heir_first_year=1870.0)
        row = self._dying_row(out)

        assert row["legacy_parents"] == "Civil rights & race; Immigration"
        assert row["corex_parent_ratios"] == "Civil rights & race=0.90; Immigration=0.10"
        assert row["corex_decline_ratio"] == pytest.approx(0.90)

    def test_a_named_successor_wins_over_every_parent_having_collapsed(self):
        """PIN 6. The precedence rule: signal 2 (a concrete, checkable heir)
        beats signal 1, because the 15-bucket crosswalk's anchor vocabulary is
        itself period-bound and can appear to collapse for exactly the renaming
        reason under test. Without this precedence, Slavery -> Civil Rights would
        be reported as a `true_death`."""
        out = self._world({"Immigration": 0.10}, heir_first_year=1830.0)
        row = self._dying_row(out)

        assert row["successor_topic"] == self.HEIR
        assert row["corex_persists"] is False, "all parents collapsed"
        assert row["rename_class"] == "rename"

    def test_corex_persists_is_recorded_even_when_the_successor_decides(self):
        """The reader must always be able to see whether the two signals agreed."""
        out = self._world({"Immigration": 0.90}, heir_first_year=1830.0)
        row = self._dying_row(out)

        assert row["rename_class"] == "rename"
        assert row["corex_persists"] is True

    def test_the_llm_decline_ratio_is_recorded_for_the_dying_topic(self):
        out = self._world({"Immigration": 0.10}, heir_first_year=1870.0)
        row = self._dying_row(out)

        assert row["llm_decline_ratio"] == pytest.approx(0.0)

    def test_a_cross_domain_riser_is_only_ever_a_diagnostic(self):
        """Recorded in `cross_domain_candidate`, never allowed to classify."""
        falling = _series(self.YEARS, [(1800, 1820, 1.0), (1821, 1850, 0.5)])
        rising = _series(self.YEARS, [(1830, 1900, 1.0)])
        lifecycles = pd.DataFrame([
            dict(topic=self.DYING, lifecycle_class="died", first_year=1800.0,
                 peak_year=1810.0, last_year=1850.0),
            dict(topic=self.HEIR, lifecycle_class="born", first_year=1830.0,
                 last_year=1900.0, peak_year=1890.0),
        ])
        curves = _curves_long({self.DYING: falling, self.HEIR: rising})
        inverted = {self.DYING: ["Immigration"], self.HEIR: ["Immigration"]}
        parents = {self.DYING: "Domain A", self.HEIR: "Domain B"}

        out = A.classify_deaths(lifecycles, curves,
                                self._corex({"Immigration": 0.10}), inverted, parents)
        row = self._dying_row(out)

        assert row["successor_topic"] is None
        assert row["cross_domain_candidate"] == self.HEIR
        assert row["rename_class"] == "true_death"

    def test_a_living_topic_gets_parents_but_no_classification(self):
        out = self._world({"Immigration": 0.10}, heir_first_year=1830.0)
        heir = out[out["topic"] == self.HEIR].iloc[0]

        assert heir["legacy_parents"] == "Immigration"
        assert heir["rename_class"] is None
        assert heir["corex_persists"] is None
        assert np.isnan(heir["llm_decline_ratio"])

    def test_every_new_column_exists_even_with_no_deaths_at_all(self):
        curves = _curves_long({self.HEIR: _series(self.YEARS, [(1830, 1900, 1.0)])})
        lifecycles = pd.DataFrame([
            dict(topic=self.HEIR, lifecycle_class="persistent", first_year=1800.0,
                 last_year=1900.0, peak_year=1890.0)
        ])

        out = A.classify_deaths(lifecycles, curves, self._corex({}),
                                {}, {self.HEIR: "Domain A"})

        for col in ("rename_class", "legacy_parents", "corex_parent_ratios",
                    "corex_decline_ratio", "corex_persists", "llm_decline_ratio",
                    "successor_topic", "successor_corr", "cross_domain_candidate"):
            assert col in out.columns
        assert out["rename_class"].isna().all()

    def test_a_parent_absent_from_the_corex_frame_is_ignored_not_fatal(self):
        """`Discovered 1-7` are deliberately excluded from the cross-check, so a
        crosswalk parent with no CorEx column must simply not contribute."""
        falling = _series(self.YEARS, [(1800, 1820, 1.0), (1821, 1850, 0.5)])
        lifecycles = pd.DataFrame([
            dict(topic=self.DYING, lifecycle_class="died", first_year=1800.0,
                 peak_year=1810.0, last_year=1850.0)
        ])
        curves = _curves_long({self.DYING: falling})

        out = A.classify_deaths(
            lifecycles, curves, self._corex({"Immigration": 0.10}),
            {self.DYING: ["Immigration", "Not A CorEx Column"]},
            {self.DYING: "Domain A"},
        )
        row = self._dying_row(out)

        assert row["legacy_parents"] == "Immigration; Not A CorEx Column"
        assert row["corex_parent_ratios"] == "Immigration=0.10"
        assert row["corex_persists"] is False


# =========================================================================== #
def _bootstrap_corpus():
    """Six speeches over two eras and two genres, with hand-computable shares."""
    rows = [
        ("a", 0, 1800, SOTU, ["T1"]), ("a", 1, 1800, SOTU, ["T1"]),
        ("a", 2, 1800, SOTU, []),
        ("b", 0, 1810, OTHER, ["T1"]), ("b", 1, 1810, OTHER, ["T2"]),
        ("c", 0, 2000, SOTU, ["T2"]), ("c", 1, 2000, SOTU, ["T2"]),
        ("d", 0, 2010, OTHER, ["T1", "T2"]), ("d", 1, 2010, OTHER, []),
        ("d", 2, 2010, OTHER, []),
        ("e", 0, 1990, SOTU, ["T1"]),
        ("f", 0, 1805, OTHER, []),
    ]
    return _corpus(rows, {"T1": "Domain A", "T2": "Domain A"})


def _wide_bootstrap_corpus(n_speeches: int = 30):
    """Enough speeches that the 2.5/97.5 percentiles are not saturated at the
    extremes — needed to show that changing the seed moves the interval."""
    rows = []
    for s in range(n_speeches):
        year = 1789 + (s % 20)
        genre = SOTU if s % 2 == 0 else OTHER
        for idx in range(4):
            topics = ["T1"] if idx < (s % 5) else []
            if idx == 3 and s % 3 == 0:
                topics = [*topics, "T2"]
            rows.append((f"doc{s:02d}", idx, year, genre, topics))
    return _corpus(rows, {"T1": "Domain A", "T2": "Domain A"})


def _stratum_corpus(n_other_founding: int):
    """A founding era with DELIBERATELY UNEQUAL genre strata.

    `_bootstrap_corpus` cannot be used to test genre standardization: both of its
    eras hold 3 SOTU and 3 OTHER paragraphs and its corpus genre weights are 6/6,
    so the post-stratified share collapses onto the pooled share exactly and
    `_era_shares(..., "genre_standardized")` is elementwise IDENTICAL to `"raw"`.
    Every genre-standardized assertion on it is therefore vacuous.

    Here instead:
      * founding SOTU stratum: 4 paragraphs, ONE carries T1  -> within-share 1/4
      * founding OTHER stratum: `n_other_founding` paragraphs, ALL carry T1 -> 1.0
      * modern filler fixes the corpus-wide genre weights at OTHER 4 / SOTU 12
        (i.e. 0.25 / 0.75) whatever the founding split is.

    So the founding T1 answer separates four different rules:
      post-stratified at corpus weights .. 0.25*1.0 + 0.75*0.25 = 0.4375  (actual)
      pooled (no standardization) ........ (1 + n) / (4 + n)
      post-stratified at EQUAL weights ... 0.5*1.0 + 0.5*0.25   = 0.625
      eligibility `den >= MIN_GENRE_STRATUM` at n=2 (SOTU alone) = 0.25
    """
    rows = []
    for idx in range(4):
        rows.append(("sotu-founding", idx, 1800, SOTU, ["T1"] if idx == 0 else []))
    for idx in range(n_other_founding):
        rows.append(("other-founding", idx, 1805, OTHER, ["T1"]))
    for idx in range(4 - n_other_founding):
        rows.append(("other-modern", idx, 2000, OTHER, []))
    for idx in range(8):
        rows.append(("sotu-modern", idx, 2000, SOTU, []))
    return _corpus(rows, {"T1": "Domain A"})


_BOOTSTRAP_TAX = _taxonomy({"T1": "Domain A", "T2": "Domain A"})


class TestSpeechDesignAndEraShares:
    def test_design_is_one_row_per_speech_not_per_paragraph(self):
        """The bootstrap resamples SPEECHES; within-speech ICC runs to 0.17, so
        paragraph resampling would badly understate the interval."""
        paragraphs, assignments = _bootstrap_corpus()

        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        assert design["counts"].shape == (6, 2)
        assert design["n_para"].tolist() == [3.0, 2.0, 2.0, 3.0, 1.0, 1.0]
        assert design["counts"][0].tolist() == [2.0, 0.0]   # speech "a": two T1
        assert design["counts"][3].tolist() == [1.0, 1.0]   # speech "d"
        assert design["is_sotu"].tolist() == [True, False, True, False, True, False]

    def test_genre_weights_are_corpus_wide_paragraph_counts(self):
        paragraphs, assignments = _bootstrap_corpus()

        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        assert design["genres"] == [OTHER, SOTU]
        assert design["genre_weights"].tolist() == [6.0, 6.0]

    def test_all_nine_eras_are_carried_even_when_empty(self):
        paragraphs, assignments = _bootstrap_corpus()

        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        assert design["eras"] == [label for label, _, _ in ERAS]
        assert len(design["eras"]) == 9

    def test_era_shares_at_unit_weights_reproduce_the_observed_shares(self):
        paragraphs, assignments = _bootstrap_corpus()
        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        observed = A._era_shares(design, np.ones(6), "raw")

        founding = design["eras"].index("The founding")
        # founding: 6 paragraphs, 3 carry T1, 1 carries T2
        assert observed[founding].tolist() == [pytest.approx(0.5), pytest.approx(1 / 6)]

    def test_an_era_with_no_paragraphs_is_nan_not_zero(self):
        paragraphs, assignments = _bootstrap_corpus()
        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        observed = A._era_shares(design, np.ones(6), "raw")
        empty = design["eras"].index("War & New Deal")

        assert np.isnan(observed[empty]).all()

    def test_sotu_zeroes_out_non_annual_messages_on_both_sides(self):
        paragraphs, assignments = _bootstrap_corpus()
        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        observed = A._era_shares(design, np.ones(6), "sotu")
        founding = design["eras"].index("The founding")

        # only speech "a" survives: 3 paragraphs, 2 carry T1, 0 carry T2
        assert observed[founding].tolist() == [pytest.approx(2 / 3), pytest.approx(0.0)]

    def test_genre_standardized_era_shares_average_the_within_genre_strata(self):
        """Post-stratification, at the FIXED CORPUS-WIDE genre weights.

        Both founding strata clear MIN_GENRE_STRATUM here (4 SOTU, 3 OTHER), so
        this test is deliberately independent of the eligibility rule — it pins
        only that the era grid combines WITHIN-genre shares at the corpus mix.
        Its discriminating power is in the three values it separates:
        0.4375 (actual) vs 4/7 (pooled, i.e. no standardization at all) vs
        0.625 (post-stratified but at equal weights).

        This replaces a vacuous predecessor. The old version ran on
        `_bootstrap_corpus`, whose strata are 3/3 with 6/6 corpus weights, where
        `genre_standardized` is elementwise identical to `raw` — see
        `_stratum_corpus`."""
        paragraphs, assignments = _stratum_corpus(3)
        design = A._speech_design(paragraphs, assignments, "topic", ["T1"])
        founding = design["eras"].index("The founding")
        ones = np.ones(len(design["n_para"]))

        observed = A._era_shares(design, ones, "genre_standardized")

        assert design["genre_weights"].tolist() == [4.0, 12.0]   # OTHER, SOTU
        # 0.25 * (3/3) + 0.75 * (1/4)
        assert observed[founding][0] == pytest.approx(0.4375)
        # ... and is NOT the pooled share, which is what makes the test non-vacuous
        pooled = A._era_shares(design, ones, "raw")[founding][0]
        assert pooled == pytest.approx(4 / 7)

    def test_a_SUB_MINIMUM_genre_stratum_still_carries_weight_in_the_era_grid(self):
        """PINS CURRENT BEHAVIOUR AND A KNOWN INCONSISTENCY (carry-forward #9).

        `_era_shares` admits any stratum with `den > 0`, while `_combine_strata`
        (the year-level curves) requires `den >= MIN_GENRE_STRATUM` with a
        `den > 0` fallback — so the published CI grid and the published point
        curves use DIFFERENT eligibility rules, despite the comment at
        attention.py:1085 claiming they are the same.

        The founding era here holds a 2-paragraph OTHER stratum (below
        MIN_GENRE_STRATUM = 3) beside a 4-paragraph SOTU stratum. Fed the same
        configuration, the two code paths return DIFFERENT numbers, and this test
        asserts both: the era grid keeps the tiny stratum (0.4375) and the
        year-level rule discards it (0.25).

        If SIMPLIFY unifies the rule, this test flips loudly — which is the whole
        point. It is the only thing standing between a one-line "consistency"
        edit and a silent numeric change to the published CI grid."""
        paragraphs, assignments = _stratum_corpus(2)
        design = A._speech_design(paragraphs, assignments, "topic", ["T1"])
        founding = design["eras"].index("The founding")
        ones = np.ones(len(design["n_para"]))

        era_grid = A._era_shares(design, ones, "genre_standardized")[founding][0]

        # The identical stratum configuration through the YEAR-level path:
        # OTHER den=2 num=2, SOTU den=4 num=1, weights 0.25 / 0.75.
        years = pd.RangeIndex(1800, 1801, name="year")
        num = {OTHER: pd.DataFrame({"T1": [2.0]}, index=years),
               SOTU: pd.DataFrame({"T1": [1.0]}, index=years)}
        den = {OTHER: pd.Series([2.0], index=years),
               SOTU: pd.Series([4.0], index=years)}
        year_level = A._combine_strata(
            num, den, {OTHER: 0.25, SOTU: 0.75}, A.MIN_GENRE_STRATUM
        ).loc[1800, "T1"]

        assert era_grid == pytest.approx(0.4375)     # den > 0: tiny stratum kept
        assert year_level == pytest.approx(0.25)     # den >= 3: tiny stratum dropped
        assert era_grid != pytest.approx(year_level)

    def test_a_zero_weight_speech_drops_out_of_its_era(self):
        """The mechanism the bootstrap relies on: a speech not drawn in a
        replicate contributes nothing to numerator or denominator."""
        paragraphs, assignments = _bootstrap_corpus()
        design = A._speech_design(paragraphs, assignments, "topic", ["T1", "T2"])

        w = np.ones(6)
        w[0] = 0.0  # drop speech "a"
        observed = A._era_shares(design, w, "raw")
        founding = design["eras"].index("The founding")

        # remaining founding paragraphs: b (2) + f (1) = 3, one carries T1
        assert observed[founding][0] == pytest.approx(1 / 3)


class TestBootstrapEraShares:
    """PIN 8: determinism, and the point estimate's identity with a plain
    groupby share."""

    def test_the_same_seed_produces_identical_intervals_across_two_calls(self):
        paragraphs, assignments = _bootstrap_corpus()

        first = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                       n_draws=50, seed=A.BOOTSTRAP_SEED)
        second = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                        n_draws=50, seed=A.BOOTSTRAP_SEED)

        pd.testing.assert_frame_equal(first, second)

    def test_the_bootstrap_actually_varies_so_determinism_is_not_vacuous(self):
        """Without this guard "identical across calls" could pass on a degenerate
        run where every replicate equals the point estimate."""
        paragraphs, assignments = _bootstrap_corpus()

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     n_draws=50, seed=A.BOOTSTRAP_SEED)
        spread = (out["share_hi"] - out["share_lo"]).dropna()

        assert len(spread) > 0
        assert (spread > 0).any()

    def test_clustering_on_speeches_costs_the_effective_sample_size_it_should(self):
        """Acceptance criterion 5 is "bootstrap CIs CLUSTERED ON SPEECHES", and
        this is what makes that word load-bearing rather than decorative.

        The fixture has maximal within-speech correlation: each of the 40
        speeches is homogeneous — all ten of its paragraphs carry the topic, or
        none do. The effective sample size is therefore 40 speeches, NOT 400
        paragraphs, and the interval must be about sqrt(10) wider than a
        paragraph-level resample of the same data. The paragraph-level bootstrap
        below is computed here from scratch (plain numpy over the 400 indicators)
        so it shares no code with the module and cannot drift with it.

        If `bootstrap_era_shares` ever resampled paragraphs, its interval would
        collapse onto the narrow one and this fails — which is the whole point:
        the real corpus's within-speech ICC runs to 0.17, so paragraph
        resampling would understate every published interval."""
        rows = [
            (f"doc{s:02d}", i, 1800, SOTU, ["T1"] if s % 2 == 0 else [])
            for s in range(40)
            for i in range(10)
        ]
        paragraphs, assignments = _corpus(rows, {"T1": "Domain A"})

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     n_draws=500, seed=A.BOOTSTRAP_SEED)
        row = out[(out["era"] == "The founding") & (out["topic"] == "T1")].iloc[0]
        clustered_width = row["share_hi"] - row["share_lo"]

        indicator = np.concatenate([
            np.full(10, 1.0 if s % 2 == 0 else 0.0) for s in range(40)
        ])
        rng = np.random.default_rng(A.BOOTSTRAP_SEED)
        draws = rng.integers(0, len(indicator), size=(500, len(indicator)))
        replicates = indicator[draws].mean(axis=1)
        naive_width = np.percentile(replicates, A.CI_HIGH) - np.percentile(
            replicates, A.CI_LOW
        )

        assert row["share"] == pytest.approx(0.5)
        assert naive_width > 0, "the naive bootstrap must itself be non-degenerate"
        assert clustered_width > naive_width
        # sqrt(400/40) = 3.16x expected; assert 2x so the claim is unambiguous
        # without pinning a magic number.
        assert clustered_width > 2 * naive_width

    def test_a_different_seed_moves_the_interval_but_not_the_point_estimate(self):
        """Proves the seed argument is actually threaded into the RNG, and that
        the observed share is computed at unit weights rather than from the
        replicates. Needs the wider corpus — with six speeches the percentiles
        saturate at 0 and 1 for every seed."""
        paragraphs, assignments = _wide_bootstrap_corpus()

        default = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                         n_draws=50, seed=A.BOOTSTRAP_SEED)
        other = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                       n_draws=50, seed=A.BOOTSTRAP_SEED + 1)

        assert not default["share_lo"].equals(other["share_lo"])
        pd.testing.assert_series_equal(default["share"], other["share"])

    def test_the_point_estimate_equals_an_independent_groupby_era_share(self):
        """Computed here from the frames directly, with no reference to
        `_era_shares` — if the design matrices are wrong this diverges."""
        paragraphs, assignments = _bootstrap_corpus()

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     n_draws=10, seed=A.BOOTSTRAP_SEED)

        den = paragraphs.groupby("era", observed=True).size()
        num = assignments.groupby(["topic", "era"], observed=True).size()
        checked = 0
        for _, row in out.iterrows():
            if row["era"] not in den.index:
                assert np.isnan(row["share"])
                continue
            expected = num.get((row["topic"], row["era"]), 0) / den[row["era"]]
            assert row["share"] == pytest.approx(expected)
            checked += 1
        assert checked > 0, "fixture must populate at least one era"

    def test_each_replicate_redraws_the_full_number_of_speeches(self):
        """A cluster bootstrap must draw n speeches per replicate, not a subset —
        drawing fewer silently inflates every interval by sqrt(n / n_drawn).

        Forty single-paragraph speeches, half carrying the topic, make the era
        share an ordinary sample proportion over 40 clusters, so the interval
        width has a closed form: 2 * 1.96 * sqrt(p(1-p)/n). Halving the draw size
        would widen it by sqrt(2) = 41%, well outside the 20% band below. The
        seed is fixed, so this is deterministic rather than statistical luck."""
        rows = [
            (f"doc{i:02d}", 0, 1800, SOTU, ["T1"] if i % 2 == 0 else [])
            for i in range(40)
        ]
        paragraphs, assignments = _corpus(rows, {"T1": "Domain A"})

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     n_draws=500, seed=A.BOOTSTRAP_SEED)
        row = out[(out["era"] == "The founding") & (out["topic"] == "T1")].iloc[0]

        expected = 2 * 1.96 * np.sqrt(0.25 / 40)
        width = row["share_hi"] - row["share_lo"]
        assert row["share"] == pytest.approx(0.5)
        assert width == pytest.approx(expected, rel=0.20)
        assert width < expected * np.sqrt(2) * 0.9, "a half-size resample would be wider"

    def test_sample_sizes_are_reported_per_era_and_topic(self):
        paragraphs, assignments = _bootstrap_corpus()

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     n_draws=10, seed=A.BOOTSTRAP_SEED)
        founding = out[(out["era"] == "The founding") & (out["topic"] == "T1")].iloc[0]

        assert founding["n_paragraphs"] == 6
        assert founding["n_topic_paragraphs"] == 3

    def test_every_topic_gets_a_row_in_every_era(self):
        paragraphs, assignments = _bootstrap_corpus()

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     n_draws=10, seed=A.BOOTSTRAP_SEED)

        assert len(out) == 9 * 2
        assert set(out["era"]) == {label for label, _, _ in ERAS}

    def test_level1_rollup_counts_a_multi_topic_paragraph_once(self):
        """Speech "d" paragraph 0 carries T1 and T2, both in Domain A — it is one
        Domain A paragraph, so 4 of the 6 Post-Cold War paragraphs are Domain A
        paragraphs, not 5.

        The `share <= 1` bound below is NOT sufficient on its own: dropping the
        rollup dedup here yields 5/6, which is still under 1, so the mutation
        survives a bound-only assertion. The exact value is what pins it."""
        paragraphs, assignments = _bootstrap_corpus()

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     level="level1", n_draws=10,
                                     seed=A.BOOTSTRAP_SEED)
        present = out[(out["era"] == "The present era")]
        modern = out[out["era"] == "Post-Cold War"].iloc[0]

        assert set(out["topic"]) == {"Domain A"}
        assert (out["share"].dropna() <= 1.0).all()
        assert len(present) == 1
        assert (modern["n_paragraphs"], modern["n_topic_paragraphs"]) == (6, 4)
        assert modern["share"] == pytest.approx(4 / 6)

    def test_the_sotu_treatment_resamples_all_speeches_with_non_sotu_at_zero(self):
        """Documented decision: this correctly propagates the uncertainty in how
        many annual messages an era happens to contain."""
        paragraphs, assignments = _bootstrap_corpus()

        out = A.bootstrap_era_shares(paragraphs, assignments, _BOOTSTRAP_TAX,
                                     treatment="sotu", n_draws=50,
                                     seed=A.BOOTSTRAP_SEED)
        founding = out[(out["era"] == "The founding") & (out["topic"] == "T1")].iloc[0]

        assert founding["share"] == pytest.approx(2 / 3)
        assert founding["n_paragraphs"] == 3


# =========================================================================== #
class TestExemplarQuotes:
    def _text_parquet(self, tmp_path, rows):
        """rows: (doc_name, para_idx, n_words)."""
        frame = pd.DataFrame([
            {"doc_name": d, "para_idx": p, "text": " ".join(["word"] * n),
             "word_count": n}
            for d, p, n in rows
        ])
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "paragraphs.parquet"
        frame.to_parquet(path, index=False)
        return path

    def _assignments(self, rows):
        """rows: (doc_name, para_idx, year, topic)."""
        return pd.DataFrame(
            [
                {"doc_name": d, "para_idx": p, "year": y, "topic": t, "level1": "D",
                 "era": A.era_name(y), "speech_type": SOTU}
                for d, p, y, t in rows
            ],
            columns=["doc_name", "para_idx", "year", "topic", "level1", "era",
                     "speech_type"],
        )

    def test_picks_are_deterministic_and_spread_across_the_window(self, tmp_path):
        rows = [("d", i, 1800 + i, "Alpha") for i in range(5)]
        path = self._text_parquet(tmp_path, [("d", i, 50) for i in range(5)])

        first = A.exemplar_quotes("Alpha", self._assignments(rows), 1800, 1810,
                                  n=3, paragraphs_path=path)
        second = A.exemplar_quotes("Alpha", self._assignments(rows), 1800, 1810,
                                   n=3, paragraphs_path=path)

        assert [q["para_idx"] for q in first] == [0, 2, 4]
        assert first == second

    def test_paragraphs_outside_the_word_count_band_are_skipped(self, tmp_path):
        rows = [("d", 0, 1800, "Alpha"), ("d", 1, 1801, "Alpha"), ("d", 2, 1802, "Alpha")]
        path = self._text_parquet(tmp_path, [("d", 0, 5), ("d", 1, 50), ("d", 2, 500)])

        quotes = A.exemplar_quotes("Alpha", self._assignments(rows), 1789, 2026,
                                   n=3, paragraphs_path=path)

        assert [q["para_idx"] for q in quotes] == [1]
        assert quotes[0]["word_count"] == 50

    def test_the_year_window_is_inclusive_at_both_ends(self, tmp_path):
        rows = [("d", i, 1800 + i, "Alpha") for i in range(4)]
        path = self._text_parquet(tmp_path, [("d", i, 50) for i in range(4)])

        quotes = A.exemplar_quotes("Alpha", self._assignments(rows), 1801, 1802,
                                   n=10, paragraphs_path=path)

        assert [q["year"] for q in quotes] == [1801, 1802]

    def test_another_topics_paragraphs_are_never_returned(self, tmp_path):
        rows = [("d", 0, 1800, "Alpha"), ("d", 1, 1800, "Beta")]
        path = self._text_parquet(tmp_path, [("d", 0, 50), ("d", 1, 50)])

        quotes = A.exemplar_quotes("Alpha", self._assignments(rows), 1789, 2026,
                                   paragraphs_path=path)

        assert [q["para_idx"] for q in quotes] == [0]

    def test_an_empty_pool_returns_an_empty_list(self, tmp_path):
        path = self._text_parquet(tmp_path, [("d", 0, 50)])

        assert A.exemplar_quotes("Nobody", self._assignments([]), 1789, 2026,
                                 paragraphs_path=path) == []

    def test_a_pooled_paragraph_absent_from_the_text_parquet_RAISES(self, tmp_path):
        """Carry-forward defect #12, fixed at SIMPLIFY: the merge now carries
        `_require_full_merge`, matching `load_inputs` and `corex_curves`.

        Per CLAUDE.md, `validate="one_to_one"` catches duplicate keys but NOT
        diverging key SETS: an assignment whose `(doc_name, para_idx)` has no row
        in the text parquet used to be dropped by the inner join with the caller
        told nothing.

        Both halves matter. A partial drop quietly narrowed the quote pool; a
        TOTAL drop returned `[]`, which is indistinguishable from "this topic has
        no quotable paragraphs in the window" — that is exactly how the CLI
        `--quotes` path could silently print nothing."""
        rows = [("d", 0, 1800, "Alpha"), ("d", 1, 1801, "Alpha"),
                ("d", 2, 1802, "Alpha")]
        partial = self._text_parquet(tmp_path / "a", [("d", 0, 50), ("d", 2, 50)])
        none_of_them = self._text_parquet(tmp_path / "b", [("other", 9, 50)])

        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            A.exemplar_quotes("Alpha", self._assignments(rows), 1789, 2026,
                              n=5, paragraphs_path=partial)
        with pytest.raises(ValueError, match="keyed merge changed the row count"):
            A.exemplar_quotes("Alpha", self._assignments(rows), 1789, 2026,
                              n=5, paragraphs_path=none_of_them)

    def test_a_text_parquet_LARGER_than_the_pool_is_not_an_error(self, tmp_path):
        """The guard asserts only that the POOL survived. `paragraphs.parquet` is
        the whole 36k-row corpus while a pool is a handful of rows, so asserting
        against the text side too would make every real call raise."""
        rows = [("d", 0, 1800, "Alpha")]
        path = self._text_parquet(
            tmp_path, [("d", 0, 50), ("d", 1, 50), ("other", 9, 50)]
        )

        quotes = A.exemplar_quotes("Alpha", self._assignments(rows), 1789, 2026,
                                   paragraphs_path=path)

        assert [q["para_idx"] for q in quotes] == [0]

    def test_returned_records_carry_the_citation_key(self, tmp_path):
        rows = [("d", 0, 1800, "Alpha")]
        path = self._text_parquet(tmp_path, [("d", 0, 50)])

        quote = A.exemplar_quotes("Alpha", self._assignments(rows), 1789, 2026,
                                  paragraphs_path=path)[0]

        assert set(quote) == {"doc_name", "para_idx", "year", "word_count", "text"}
        assert (quote["doc_name"], quote["para_idx"], quote["year"]) == ("d", 0, 1800)


# =========================================================================== #
def _e2e_world(tmp_path, monkeypatch):
    """A 45-year-spaced synthetic corpus with a textbook rename.

    Each populated year holds one 20-paragraph annual message and one
    20-paragraph other speech, so every 5-year window holds 40 paragraphs — just
    over MIN_WINDOW_PARAGRAPHS under `raw`, and just under it under `sotu`.
    `Old Topic` runs 1800-1900 and hands off to `New Topic` (1880-2020) in the
    same domain and under the same legacy crosswalk parent.
    """
    years = list(range(1800, 2021, 5))
    rows = []
    for year in years:
        for genre, tag in ((SOTU, "s"), (OTHER, "o")):
            doc = f"{tag}{year}"
            for idx in range(20):
                topics = []
                if year <= 1900 and idx < 4:
                    topics.append("Old Topic")
                if year >= 1880 and idx >= 16:
                    topics.append("New Topic")
                if idx in (8, 9):
                    topics.append("Side Topic")
                if year == 1900 and genre == SOTU and idx == 12:
                    topics.append("Thin Topic")
                rows.append((doc, idx, year, genre, topics))

    parents = {"Old Topic": "Domain A", "New Topic": "Domain A",
               "Side Topic": "Domain B", "Thin Topic": "Domain B"}
    paragraphs, assignments = _corpus(rows, parents)
    taxonomy = _taxonomy(parents)
    crosswalk = _crosswalk({
        "Immigration": ["Old Topic", "New Topic"],
        "Foreign policy": ["Side Topic"],
    })

    issue_rows = [(d, p, y) for d, p, y, _, _ in rows]
    fired = {
        (d, p): ["Immigration"]
        for d, p, y, _, topics in rows
        if "Old Topic" in topics
    }
    path = tmp_path / "paragraph_issues.parquet"
    _issue_frame(issue_rows, fired).to_parquet(path, index=False)
    monkeypatch.setattr(A, "PARA_LABELS_PATH", path)
    return paragraphs, assignments, taxonomy, crosswalk


class TestBuildLifecycleTable:
    """End-to-end anchor: the real assembly path over a synthetic corpus, so the
    unit pins above are known to compose."""

    @pytest.fixture
    def built(self, tmp_path, monkeypatch):
        paragraphs, assignments, taxonomy, crosswalk = _e2e_world(tmp_path, monkeypatch)
        table, curves, eras = A.build_lifecycle_table(
            paragraphs, assignments, taxonomy, crosswalk, n_draws=5,
            seed=A.BOOTSTRAP_SEED,
        )
        return table, curves, eras

    def test_the_table_is_one_row_per_level_treatment_topic(self, built):
        table, _, _ = built

        assert list(table.columns) == A.OUTPUT_COLUMNS
        counts = table.groupby(["level", "treatment"]).size()
        assert counts.loc[("level2", "raw")] == 4
        assert counts.loc[("level1", "raw")] == 2
        assert len(table) == 3 * 4 + 3 * 2

    def test_curves_and_era_grids_are_returned_for_every_key(self, built):
        _, curves, eras = built

        expected = {(lvl, t) for lvl in A.LEVELS for t in A.TREATMENTS}
        assert set(curves) == expected
        assert set(eras) == expected

    def test_the_dying_topic_is_classified_as_a_rename_by_its_named_heir(self, built):
        """The whole point of the module, end to end: the LLM curve collapses AND
        its only legacy CorEx parent collapses with it, but a same-domain heir
        with a shared parent rises through the decline window — so it is a
        rename, not a true death."""
        table, _, _ = built
        row = table[(table["level"] == "level2") & (table["treatment"] == "raw")
                    & (table["topic"] == "Old Topic")].iloc[0]

        assert row["lifecycle_class"] == "died"
        assert row["last_year"] == 1900
        assert row["corex_persists"] is False
        assert row["successor_topic"] == "New Topic"
        assert row["successor_corr"] < 0
        assert row["rename_class"] == "rename"

    def test_the_heir_is_classified_as_born(self, built):
        table, _, _ = built
        row = table[(table["level"] == "level2") & (table["treatment"] == "raw")
                    & (table["topic"] == "New Topic")].iloc[0]

        assert row["lifecycle_class"] == "born"
        assert row["first_year"] == 1880
        assert row["rename_class"] is None

    def test_a_topic_present_throughout_is_persistent(self, built):
        table, _, _ = built
        row = table[(table["level"] == "level2") & (table["treatment"] == "raw")
                    & (table["topic"] == "Side Topic")].iloc[0]

        assert row["lifecycle_class"] == "persistent"

    def test_the_under_powered_guard_fires_for_a_one_paragraph_topic(self, built):
        """The guard flags nothing on the real corpus at level 2, so this
        end-to-end firing is what stops it becoming dead code."""
        table, _, _ = built
        row = table[(table["level"] == "level2") & (table["treatment"] == "raw")
                    & (table["topic"] == "Thin Topic")].iloc[0]

        assert row["n_paragraphs"] < A.UNDER_POWERED_N
        assert bool(row["under_powered"]) is True
        assert row["lifecycle_class"] == "under_powered"

    def test_the_sotu_treatment_cannot_confirm_a_death_on_a_thin_slice(self, built):
        """The genre check has real teeth: with only 20 paragraphs per SOTU
        window, nothing clears MIN_WINDOW_PARAGRAPHS and every topic reads as
        absent — which is the honest answer, not a death."""
        table, _, _ = built
        sotu = table[(table["level"] == "level2") & (table["treatment"] == "sotu")]

        assert len(sotu) == 4
        assert set(sotu["lifecycle_class"]) <= {"absent", "under_powered"}

    def test_level1_rows_carry_no_rename_classification(self, built):
        table, _, _ = built
        level1 = table[table["level"] == "level1"]

        # non-empty guard: `.all()` on an empty selection is vacuously True, so
        # a filter that silently matched nothing would pass this test.
        assert len(level1) == 2 * len(A.TREATMENTS)   # Domain A, Domain B
        assert level1["level1"].isna().all()
        assert level1["rename_class"].isna().all()

    def test_era_ci_columns_are_populated_and_bracket_the_point_estimate(self, built):
        table, _, _ = built
        raw = table[(table["level"] == "level2") & (table["treatment"] == "raw")]

        assert len(raw) == 4        # non-empty guard, see above
        for prefix in ("first_era", "final_era"):
            lo = raw[f"{prefix}_share_lo"].to_numpy(float)
            point = raw[f"{prefix}_share"].to_numpy(float)
            hi = raw[f"{prefix}_share_hi"].to_numpy(float)
            assert np.isfinite(point).all()
            assert (lo <= point + 1e-12).all()
            assert (point <= hi + 1e-12).all()

    def test_dominant_era_is_the_era_with_the_largest_share(self, built):
        table, eras = built[0], built[2]
        raw = table[(table["level"] == "level2") & (table["treatment"] == "raw")]
        grid = eras[("level2", "raw")]

        for _, row in raw.iterrows():
            topic_grid = grid[grid["topic"] == row["topic"]]
            best = topic_grid.loc[topic_grid["share"].idxmax(), "era"]
            assert row["dominant_era"] == best

    def test_the_whole_table_is_reproducible_for_a_fixed_seed(self, tmp_path,
                                                              monkeypatch):
        paragraphs, assignments, taxonomy, crosswalk = _e2e_world(tmp_path, monkeypatch)

        first, _, _ = A.build_lifecycle_table(paragraphs, assignments, taxonomy,
                                              crosswalk, n_draws=5, seed=1234)
        second, _, _ = A.build_lifecycle_table(paragraphs, assignments, taxonomy,
                                               crosswalk, n_draws=5, seed=1234)

        pd.testing.assert_frame_equal(first, second)

    def test_write_lifecycles_is_byte_identical_on_rerun(self, built, tmp_path):
        """The `nondeterministic-plot-output` backlog item is what happens when
        an artifact is not byte-reproducible; this one must be."""
        table, _, _ = built
        first = A.write_lifecycles(table, tmp_path / "nested" / "one.parquet")
        second = A.write_lifecycles(table, tmp_path / "nested" / "two.parquet")

        assert first.exists()
        assert first.read_bytes() == second.read_bytes()

        round_trip = pd.read_parquet(first)
        assert list(round_trip.columns) == A.OUTPUT_COLUMNS
        # `index=False` is part of the artifact contract, not a detail: writing
        # the index changes the file's bytes (so the committed sha256 moves) and
        # round-trips as a plain Index. The column list alone cannot see it.
        assert isinstance(round_trip.index, pd.RangeIndex)
        # parquet normalizes the object columns' None to NaN, so compare on a
        # common representation rather than pretending the dtypes survive.
        keys = ["level", "treatment", "topic", "lifecycle_class", "rename_class"]
        pd.testing.assert_frame_equal(
            round_trip[keys].fillna("").astype(str).reset_index(drop=True),
            table[keys].fillna("").astype(str).reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            round_trip["peak_share"], table["peak_share"].reset_index(drop=True)
        )


# =========================================================================== #
class TestGroundTruthAndAnachronism:
    def _table(self, rows):
        return pd.DataFrame([
            {"level": "level2", "treatment": "raw", "first_year": 1800.0,
             "peak_year": 1850.0, "peak_era": "Civil War & Reconstruction",
             "peak_share": 0.1, "last_year": 1900.0, "lifecycle_class": "died",
             "rename_class": "rename", **row}
            for row in rows
        ])

    def test_the_registered_checks_are_the_task_verification_strategy(self):
        """Decoupled literal spec-anchor for the falsification checks: these are
        the ground-truth targets from the task, and dropping one must be a
        visible change rather than a quietly shorter list."""
        labels = [label for label, _, _ in A.GROUND_TRUTH]

        assert labels == [
            "Indian Affairs peaks 1830s-1870s and dies",
            "Coinage/currency peaks in the 1890s free-silver era",
            "Terrorism is born around 2001",
            "Prohibition is a sharp 1920s spike",
            "Slavery -> civil rights classifies as a RENAME",
        ]
        topics = {label: topic for label, topic, _ in A.GROUND_TRUTH}
        assert topics["Prohibition is a sharp 1920s spike"] is None
        assert topics["Slavery -> civil rights classifies as a RENAME"] == (
            "Slavery, Emancipation & Sectionalism"
        )

    def test_a_check_with_no_topic_is_reported_not_testable(self):
        results = A.run_ground_truth(self._table([{"topic": "Anything"}]))
        prohibition = next(r for r in results if "Prohibition" in r["check"])

        assert prohibition["passed"] == "not_testable"
        assert "no such topic" in prohibition["observed"]

    def test_a_missing_topic_is_reported_not_testable_rather_than_skipped(self):
        results = A.run_ground_truth(self._table([{"topic": "Anything"}]))
        coinage = next(r for r in results if "Coinage" in r["check"])

        assert coinage["passed"] == "not_testable"
        assert "absent" in coinage["observed"]

    def test_a_satisfied_predicate_passes_and_reports_the_observation(self, monkeypatch):
        monkeypatch.setattr(A, "GROUND_TRUTH", [
            ("peak is in the 1850s", "Alpha", lambda r: 1850 <= r["peak_year"] <= 1859)
        ])

        results = A.run_ground_truth(self._table([{"topic": "Alpha"}]))

        assert results[0]["passed"] is True
        assert "peak=1850" in results[0]["observed"]
        assert "class=died" in results[0]["observed"]

    def test_a_violated_predicate_is_reported_as_a_failure_not_an_error(self, monkeypatch):
        """A failed check is a finding about the annotation layer, never an
        exception and never something to tune away."""
        monkeypatch.setattr(A, "GROUND_TRUTH", [
            ("peak is in the 1990s", "Alpha", lambda r: r["peak_year"] >= 1990)
        ])

        results = A.run_ground_truth(self._table([{"topic": "Alpha"}]))

        assert results[0]["passed"] is False

    def test_only_the_raw_level2_rows_are_checked(self, monkeypatch):
        monkeypatch.setattr(A, "GROUND_TRUTH", [
            ("peak is in the 1850s", "Alpha", lambda r: 1850 <= r["peak_year"] <= 1859)
        ])
        table = pd.concat([
            self._table([{"topic": "Alpha", "treatment": "sotu", "peak_year": 1990.0}]),
            self._table([{"topic": "Alpha", "peak_year": 1850.0}]),
        ], ignore_index=True)

        assert A.run_ground_truth(table)[0]["passed"] is True

    def test_anachronism_report_measures_the_gap_to_the_raw_first_appearance(self):
        """A large gap is the signature of stray mislabels — the failure mode the
        substantive threshold exists to absorb."""
        assignments = pd.DataFrame({
            "topic": ["Alpha", "Alpha", "Beta"],
            "year": [1700 + 61, 1900, 1899],
        })
        table = self._table([
            {"topic": "Alpha", "first_year": 1900.0, "n_paragraphs": 10},
            {"topic": "Beta", "first_year": 1900.0, "n_paragraphs": 10},
        ])

        report = A.anachronism_report(assignments, table)

        assert list(report["topic"]) == ["Alpha", "Beta"]  # sorted by gap desc
        assert report.loc[0, "gap_years"] == 139
        assert report.loc[1, "gap_years"] == 1
        assert list(report.columns) == [
            "topic", "first_year", "peak_year", "last_year", "n_paragraphs",
            "first_any_year", "gap_years",
        ]


# =========================================================================== #
class TestCli:
    def _install_world(self, tmp_path, monkeypatch, redirect_annotation_dirs):
        paragraphs, assignments, taxonomy, crosswalk = _e2e_world(tmp_path, monkeypatch)
        annotations = _annotation_frame([
            (d, p, list(assignments[(assignments["doc_name"] == d)
                                    & (assignments["para_idx"] == p)]["topic"]))
            for d, p in zip(paragraphs["doc_name"], paragraphs["para_idx"])
        ])
        speeches = _speech_frame(
            paragraphs.drop_duplicates("doc_name")[["doc_name", "speech_type"]].values
        )
        root = redirect_annotation_dirs
        root.mkdir(parents=True, exist_ok=True)
        annotations.to_parquet(root / "paragraph_annotations.parquet", index=False)
        speeches.to_parquet(root / "speech_annotations.parquet", index=False)

        crosswalk_path = tmp_path / "crosswalk.json"
        crosswalk_path.write_text(json.dumps(crosswalk))
        monkeypatch.setattr(A, "CROSSWALK_PATH", crosswalk_path)
        monkeypatch.setattr(A, "load_taxonomy", lambda *a, **k: taxonomy)

        # Paragraph TEXT for this synthetic world. `main()` calls
        # `exemplar_quotes` without a `paragraphs_path`, and that default is bound
        # at DEF time, so patching `A.PARAGRAPHS_PATH` would not take — the test
        # would read the real 36k-row corpus, whose keys share nothing with these
        # doc names.
        text_path = tmp_path / "paragraphs.parquet"
        pd.DataFrame([
            {"doc_name": d, "para_idx": p,
             "text": f"body of {d} paragraph {p} " + " ".join(["word"] * 45),
             "word_count": 50}
            for d, p in zip(paragraphs["doc_name"], paragraphs["para_idx"])
        ]).to_parquet(text_path, index=False)
        return taxonomy, text_path

    def test_main_writes_the_table_and_prints_the_checks(
        self, tmp_path, monkeypatch, redirect_annotation_dirs, capsys
    ):
        self._install_world(tmp_path, monkeypatch, redirect_annotation_dirs)
        out = tmp_path / "out" / "topic_lifecycles.parquet"
        # `main` refuses a `--out` outside ATTENTION_DIR (a fat-finger guard on
        # the frozen paid artifacts). Redirect the directory rather than weaken
        # the guard; `TestOutPathGuard` pins that it still fires.
        monkeypatch.setattr(A, "ATTENTION_DIR", tmp_path / "out")

        A.main(["--draws", "3", "--seed", "7", "--out", str(out)])

        printed = capsys.readouterr().out
        assert out.exists()
        table = pd.read_parquet(out)
        assert list(table.columns) == A.OUTPUT_COLUMNS
        assert f"wrote {out}" in printed
        assert "GROUND-TRUTH CHECKS" in printed
        assert "NOT_TESTABLE" in printed  # none of the real topics exist here
        assert "LIFECYCLE CLASSES" in printed
        assert "UNDER-POWERED" in printed
        # No other test asserts this block. `print_checks` is the only caller of
        # `anachronism_report`, so without this line deleting the call leaves the
        # whole suite green while a pre-registered check silently stops printing.
        assert "ANACHRONISM CHECK" in printed

    def test_the_quotes_flag_short_circuits_before_building_anything(
        self, tmp_path, monkeypatch, redirect_annotation_dirs, capsys
    ):
        """Hermetic: the paragraph text comes from the synthetic world, not from
        the real 36k-row `data/paragraphs.parquet`.

        The printed-quote assertion is a NON-VACUITY GUARD, not decoration.
        Before this test was made hermetic it read the real 36k-row
        `data/paragraphs.parquet`, whose keys match none of the synthetic
        `(doc_name, para_idx)` pairs, so `main` printed no quotes at all and
        `not out.exists()` passed over a DEAD quotes path. It now passes over a
        live one.

        Since SIMPLIFY added `_require_full_merge` at `attention.py:1289` that
        same key mismatch would now RAISE rather than print nothing (see
        `test_a_pooled_paragraph_absent_from_the_text_parquet_RAISES`), so the
        `paragraphs_path` redirection below is load-bearing in both directions:
        without it this test errors instead of silently passing."""
        _, text_path = self._install_world(tmp_path, monkeypatch, redirect_annotation_dirs)
        monkeypatch.setattr(
            A, "exemplar_quotes",
            functools.partial(A.exemplar_quotes, paragraphs_path=text_path),
        )
        out = tmp_path / "out" / "topic_lifecycles.parquet"

        A.main(["--quotes", "Old Topic", "--quote-years", "1800", "1810",
                "--out", str(out)])

        printed = capsys.readouterr().out
        assert not out.exists()
        assert "[1800] o1800 #0" in printed
        assert "body of o1800 paragraph 0" in printed


# =========================================================================== #
class TestCommittedArtifacts:
    """Fast read-only anchors on the frozen artifacts, by absolute worktree path
    (conftest repoints ANNOTATIONS_DIR at tmp_path, and under a worktree the
    module constant can resolve to the main repo)."""

    def test_the_real_taxonomys_casefold_is_injective_over_50_level2_names(self):
        """Correction #2's load-bearing premise: `{name.casefold(): name}` over
        the 50 canonical names is 1:1, which is what makes normalization safe."""
        taxonomy = json.loads((_ARTIFACTS / "taxonomy_v1.json").read_text())

        label_map = A.canonical_label_map(taxonomy)

        assert len(label_map) == len(taxonomy["level2"]) == 50
        assert set(label_map.values()) == {e["name"] for e in taxonomy["level2"]}

    def test_the_real_crosswalk_inverts_to_a_genuine_many_to_many(self):
        crosswalk = json.loads((_ARTIFACTS / "crosswalk_v1.json").read_text())

        inverted = A.invert_crosswalk(crosswalk)

        assert inverted["Chinese Immigration & Exclusion"] == [
            "Civil rights & race", "Immigration"
        ]
        assert any(len(parents) > 1 for parents in inverted.values())

    def test_every_crosswalk_parent_has_a_corex_curve_column(self):
        """`classify_deaths` looks each legacy parent up in the CorEx frame; a
        parent name with no column silently contributes nothing to the rule."""
        crosswalk = json.loads((_ARTIFACTS / "crosswalk_v1.json").read_text())

        issues = {m["legacy_issue"] for m in crosswalk["mappings"]}

        assert issues <= set(A.COREX_COLUMNS)

    def test_the_committed_lifecycle_table_has_the_declared_schema(self):
        table = pd.read_parquet(_ATTENTION_ARTIFACTS / "topic_lifecycles.parquet")

        assert list(table.columns) == A.OUTPUT_COLUMNS
        grid = set(zip(table["level"], table["treatment"]))
        assert grid == {(lvl, t) for lvl in A.LEVELS for t in A.TREATMENTS}
        assert not table.duplicated(["level", "treatment", "topic"]).any()

    def test_the_committed_table_classifies_slavery_as_a_rename(self):
        """The task's headline falsification target: slavery -> civil rights must
        come out a rename, not a death. If this ever flips, the rule is wrong."""
        table = pd.read_parquet(_ATTENTION_ARTIFACTS / "topic_lifecycles.parquet")
        row = table[(table["level"] == "level2") & (table["treatment"] == "raw")
                    & (table["topic"] == "Slavery, Emancipation & Sectionalism")]

        assert len(row) == 1
        assert row.iloc[0]["lifecycle_class"] == "died"
        assert row.iloc[0]["rename_class"] == "rename"
        assert row.iloc[0]["successor_topic"] is not None

    def test_the_committed_table_passes_its_own_ground_truth_checks(self):
        """Two checks ship as FAIL by design (Indian Affairs' peak year and the
        absent terrorism topic) — that is disclosed in the findings note. This
        pins WHICH ones, so a silent change in either direction is visible."""
        table = pd.read_parquet(_ATTENTION_ARTIFACTS / "topic_lifecycles.parquet")

        results = {r["check"]: r["passed"] for r in A.run_ground_truth(table)}

        assert results["Coinage/currency peaks in the 1890s free-silver era"] is True
        assert results["Slavery -> civil rights classifies as a RENAME"] is True
        assert results["Prohibition is a sharp 1920s spike"] == "not_testable"
        assert results["Indian Affairs peaks 1830s-1870s and dies"] is False
        assert results["Terrorism is born around 2001"] is False

    def test_no_topic_is_under_powered_at_level_2_under_the_raw_treatment(self):
        """Documented expectation: the thinnest level-2 topic holds 61
        paragraphs, so the guard flags nothing here. If a future annotation pass
        thins a topic below 50 this test is the alarm."""
        table = pd.read_parquet(_ATTENTION_ARTIFACTS / "topic_lifecycles.parquet")
        raw = table[(table["level"] == "level2") & (table["treatment"] == "raw")]

        assert len(raw) == 50
        assert not raw["under_powered"].any()
        assert raw["n_paragraphs"].min() >= A.UNDER_POWERED_N

    def test_the_under_powered_guard_fires_exactly_once_across_the_table(self):
        """REVIEW MAJOR 2: the module docstring used to claim the guard "fires
        routinely under `sotu`". It does not — it fires ONCE in the entire table.

        Nothing pinned the `sotu` half, which is why a false empirical claim
        survived five gated stages while its `raw` half (the test above) stayed
        green. This pins the exact firing so the docstring cannot rot again in
        either direction: a guard that stops firing has gone vacuous, and a guard
        that starts firing widely would make the deaths uncheckable inside annual
        messages — a finding, not a detail."""
        table = pd.read_parquet(_ATTENTION_ARTIFACTS / "topic_lifecycles.parquet")

        fired = table[table["under_powered"]]

        assert len(fired) == 1
        row = fired.iloc[0]
        assert (row["level"], row["treatment"]) == ("level2", "sotu")
        assert row["topic"] == "Chinese Immigration & Exclusion"
        assert row["n_paragraphs"] == 45
        assert row["lifecycle_class"] == "under_powered"
        # Per treatment, so a future shift shows up as WHERE it moved, not just
        # that a total changed.
        by_treatment = {
            t: int(table[(table["level"] == "level2") & (table["treatment"] == t)]
                   ["under_powered"].sum())
            for t in A.TREATMENTS
        }
        assert by_treatment == {"raw": 0, "sotu": 1, "genre_standardized": 0}
        assert not table[table["level"] == "level1"]["under_powered"].any()

    def test_most_raw_deaths_survive_the_annual_message_check(self):
        """The robustness result the old docstring inverted. The guard firing once
        under `sotu` is NOT evidence that deaths fail inside annual messages: 15 of
        the 17 `raw` deaths are still `died` under `sotu` and 16 of 17 under
        `genre_standardized`. Pinned because the docstring now asserts these two
        counts, and because a real drop here would undercut the note's central
        claim that the deaths are not a genre artifact."""
        table = pd.read_parquet(_ATTENTION_ARTIFACTS / "topic_lifecycles.parquet")
        level2 = table[table["level"] == "level2"]
        deaths = set(level2[(level2["treatment"] == "raw")
                            & (level2["lifecycle_class"] == "died")]["topic"])

        assert len(deaths) == 17

        survived = {}
        for treatment in ("sotu", "genre_standardized"):
            classes = level2[level2["treatment"] == treatment].set_index("topic")
            survived[treatment] = sum(
                classes.loc[topic, "lifecycle_class"] == "died" for topic in deaths
            )

        assert survived == {"sotu": 15, "genre_standardized": 16}
        # The single under-powered firing is one of the two `sotu` exceptions —
        # the same fact seen twice, exactly as the docstring now says.
        sotu = level2[level2["treatment"] == "sotu"].set_index("topic")
        assert sotu.loc["Chinese Immigration & Exclusion", "lifecycle_class"] == (
            "under_powered"
        )


class TestOutPathGuard:
    """`main --out` must not be able to clobber a frozen paid artifact.

    The operator is not a trust boundary (they could write the file directly),
    so this is a fat-finger guard rather than a security control -- but
    `data/llm_annotations/` is provenance-stamped output of a $38.56 run that
    cannot be regenerated for free, and `--out` is the module's only route to
    writing outside `data/attention/`. Raised as a LOW/MEDIUM integrity item in
    SECURITY-REVIEW; this class is what keeps the guard from rotting away.
    """

    def test_an_out_path_escaping_the_attention_dir_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(A, "ATTENTION_DIR", tmp_path / "attention")
        escape = tmp_path / "attention" / ".." / "llm_annotations" / "paragraph_annotations.parquet"

        with pytest.raises(SystemExit, match="must stay inside"):
            A._checked_out_path(escape)

        # The guard must refuse BEFORE anything is created -- a check that
        # mkdir'd its way to the target first would be worse than no check.
        assert not (tmp_path / "llm_annotations").exists()

    def test_a_plain_sibling_directory_is_refused_too(self, tmp_path, monkeypatch):
        monkeypatch.setattr(A, "ATTENTION_DIR", tmp_path / "attention")
        with pytest.raises(SystemExit, match="frozen paid-run output"):
            A._checked_out_path(tmp_path / "elsewhere" / "out.parquet")

    def test_a_path_inside_the_attention_dir_is_allowed_and_resolved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(A, "ATTENTION_DIR", tmp_path / "attention")
        nested = tmp_path / "attention" / "sub" / "out.parquet"

        checked = A._checked_out_path(nested)

        assert checked == nested.resolve()
        assert checked.is_absolute()

    def test_the_real_default_out_path_passes_its_own_guard(self):
        """The shipped default must not trip the guard -- otherwise a bare
        `python -m presidential_profiles.attention` would refuse to run."""
        assert A._checked_out_path(A.LIFECYCLES_PATH) == A.LIFECYCLES_PATH.resolve()

    def test_the_quotes_path_never_reaches_the_guard(self, tmp_path, monkeypatch,
                                                     redirect_annotation_dirs, capsys):
        """A read-only `--quotes` run must not be rejected for a `--out` it
        ignores. Pins the placement decision, not just the predicate."""
        _, text_path = TestCli()._install_world(tmp_path, monkeypatch, redirect_annotation_dirs)
        monkeypatch.setattr(A, "ATTENTION_DIR", tmp_path / "nonexistent-attention-dir")
        monkeypatch.setattr(
            A, "exemplar_quotes", functools.partial(A.exemplar_quotes, paragraphs_path=text_path)
        )

        A.main(["--quotes", "Old Topic", "--quote-years", "1800", "1810"])

        assert "must stay inside" not in capsys.readouterr().out
