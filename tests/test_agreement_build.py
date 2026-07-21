"""build_agreement / _load_joined (agreement.py) — join semantics against
SYNTHETIC primary + Opus tables, fully offline.

conftest's autouse `redirect_annotation_dirs` already repoints
llm_annotations.ANNOTATIONS_DIR at tmp_path, so the per-model parquets these
tests write land there and never touch the real (paid, in-flight) Opus run under
data/llm_annotations/. The read-only corpus paragraphs path is monkeypatched to a
tiny synthetic parquet so the coverage arithmetic is fully controlled.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from presidential_profiles import agreement as G
from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------


def _prow(doc, idx, pa, en, zs, topics, pv):
    return dict(doc_name=doc, para_idx=idx, party_attack=pa, enemy_naming=en,
                zero_sum=zs, topics=topics, proposal_values=pv, run_id="run-x")


def _speeches() -> pd.DataFrame:
    # docA -> bin 0 (1800), docB -> bin 3 (1900), docC -> bin 7 (2000, NOT sampled)
    return pd.DataFrame({"doc_name": ["docA", "docB", "docC"],
                         "year": [1800, 1900, 2000]})


def _write_para(name: str, df: pd.DataFrame) -> None:
    ann.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ann.annotation_path(name))


def _srow(doc, speech_type, audience, medium):
    return dict(doc_name=doc, speech_type=speech_type, audience=audience,
                medium=medium, run_id="run-x")


def _write_speech(name: str, df: pd.DataFrame) -> None:
    ann.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ann.annotation_path(name))


@pytest.fixture
def synthetic_layer(monkeypatch, tmp_path):
    """Write a controlled primary + Opus paragraph layer and a matching synthetic
    corpus-paragraphs parquet. Returns (sample_path, speeches). The Opus table
    deliberately MISSES docB para 1 so coverage is 3/4 (partial), unless a test
    overrides it."""
    corpus = pd.DataFrame([
        {"doc_name": "docA", "para_idx": 0, "text": "a0"},
        {"doc_name": "docA", "para_idx": 1, "text": "a1"},
        {"doc_name": "docB", "para_idx": 0, "text": "b0"},
        {"doc_name": "docB", "para_idx": 1, "text": "b1"},
        {"doc_name": "docC", "para_idx": 0, "text": "c0"},  # outside the sample
    ])
    corpus_path = tmp_path / "paragraphs.parquet"
    corpus.to_parquet(corpus_path)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", corpus_path)

    primary = pd.DataFrame([
        _prow("docA", 0, True, False, True, ["x"], "proposal"),
        _prow("docA", 1, False, False, True, ["x", "y"], "values"),
        _prow("docB", 0, True, True, False, ["z"], "proposal"),
        _prow("docB", 1, False, False, False, ["z"], "values"),
        _prow("docC", 0, True, True, True, ["q"], "proposal"),  # not in sample
    ])
    opus = pd.DataFrame([
        _prow("docA", 0, True, False, True, ["x"], "proposal"),
        _prow("docA", 1, True, False, True, ["y"], "proposal"),
        _prow("docB", 0, True, True, False, ["z"], "proposal"),
        # docB para 1 intentionally absent -> partial coverage
    ])
    _write_para("paragraph_annotations", primary)
    _write_para(A._table_name("paragraph_annotations", G.OPUS_MODEL), opus)

    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(
        {"doc_names": ["docA", "docB"], "n_sampled": 2, "seed": 1, "draw_date": "2026-07-21"}))
    return sample_path, _speeches()


# ---------------------------------------------------------------------------
# restriction to the sample + join geometry
# ---------------------------------------------------------------------------


def test_build_restricts_to_sample_docs_and_writes_the_parquet(synthetic_layer, tmp_path):
    sample_path, speeches = synthetic_layer
    out_path = tmp_path / "agreement.parquet"
    table = G.build_agreement(sample_path=sample_path, out_path=out_path, speeches=speeches)

    assert out_path.exists()
    # docC (year 2000, bin 7) is in the corpus + primary but NOT in the sample —
    # it must not appear as an era bin
    assert 7 not in set(table["era_bin"])
    # only the sampled docs' bins: docA -> 0, docB -> 3, plus the overall row
    assert set(table["era_bin"]) == {G.OVERALL_BIN, 0, 3}


def test_build_only_scores_paragraphs_present_in_both_models(synthetic_layer, tmp_path):
    """docB para 1 exists for the primary but not for Opus -> it is dropped by the
    inner join. The party_attack overall n therefore counts the 3 shared
    paragraphs (docA 0, docA 1, docB 0), not the primary's 4."""
    sample_path, speeches = synthetic_layer
    table = G.build_agreement(sample_path=sample_path,
                              out_path=tmp_path / "a.parquet", speeches=speeches)
    pa_overall = table[(table["field"] == "party_attack")
                       & (table["metric"] == "cohen_kappa")
                       & (table["era_bin"] == G.OVERALL_BIN)]
    assert len(pa_overall) == 1
    assert int(pa_overall.iloc[0]["n"]) == 3


# ---------------------------------------------------------------------------
# speech-factual join branch (gap A): only fires when the Opus speech table
# exists on disk (_load_joined ~L466-476, `if opus_sp_path.exists()`)
# ---------------------------------------------------------------------------


def test_build_includes_speech_factual_fields_when_opus_speech_table_present(
        synthetic_layer, tmp_path):
    """Drive the optional speech-factual branch end-to-end: write a primary AND an
    Opus per-model speech table, run build_agreement, and assert speech_type /
    audience / medium land in the metrics parquet with the RIGHT values and n —
    not merely present. Values are engineered so each factual field has a distinct,
    hand-computed exact_match (a field mix-up would flip them):

      docA (bin 0): type address/address, audience nation/nation, medium spoken/written
      docB (bin 3): type letter/letter,   audience congress/press, medium written/spoken

    overall n=2:  speech_type exact=1.0 kappa=+1.0 (identical, 2 classes)
                  audience    exact=0.5
                  medium      exact=0.0 kappa=-1.0 (perfect inversion, 2 classes)
    """
    sample_path, speeches = synthetic_layer
    _write_speech("speech_annotations", pd.DataFrame([
        _srow("docA", "address", "nation", "spoken"),
        _srow("docB", "letter", "congress", "written"),
        _srow("docC", "proclamation", "world", "spoken"),  # not sampled -> excluded
    ]))
    _write_speech(A._table_name("speech_annotations", G.OPUS_MODEL), pd.DataFrame([
        _srow("docA", "address", "nation", "written"),
        _srow("docB", "letter", "press", "spoken"),
    ]))

    out_path = tmp_path / "agreement.parquet"
    table = G.build_agreement(sample_path=sample_path, out_path=out_path, speeches=speeches)

    # all three factual fields present (they are absent without the Opus speech table)
    assert set(G.FACTUAL_FIELDS) <= set(table["field"])

    def cell(field, metric, era=G.OVERALL_BIN):
        sub = table[(table["field"] == field) & (table["metric"] == metric)
                    & (table["era_bin"] == era)]
        assert len(sub) == 1, f"expected exactly one {field}/{metric}/era={era} row"
        return sub.iloc[0]

    # correct VALUES (not just presence), and n = the 2 sampled+shared speeches
    st = cell("speech_type", "exact_match")
    assert st["value"] == pytest.approx(1.0) and int(st["n"]) == 2
    assert cell("speech_type", "cohen_kappa")["value"] == pytest.approx(1.0)

    au = cell("audience", "exact_match")
    assert au["value"] == pytest.approx(0.5) and int(au["n"]) == 2

    md = cell("medium", "exact_match")
    assert md["value"] == pytest.approx(0.0) and int(md["n"]) == 2
    assert cell("medium", "cohen_kappa")["value"] == pytest.approx(-1.0)

    # docC (not sampled) never contributes: no factual cell has n > 2
    factual = table[table["field"].isin(G.FACTUAL_FIELDS)]
    assert factual["n"].max() == 2


def test_build_omits_speech_factual_fields_when_opus_speech_table_absent(
        synthetic_layer, tmp_path):
    """The opposite branch: with NO Opus speech table on disk (the synthetic_layer
    default), build_agreement must still succeed over the paragraph layer and the
    metrics parquet must carry NONE of the factual fields — the speech pass is
    genuinely optional, not silently half-wired."""
    sample_path, speeches = synthetic_layer
    # sanity: the fixture writes no speech table, so the branch guard is False
    assert not ann.annotation_path(
        A._table_name("speech_annotations", G.OPUS_MODEL)).exists()

    table = G.build_agreement(sample_path=sample_path,
                              out_path=tmp_path / "a.parquet", speeches=speeches)
    assert not table.empty  # build still succeeds
    assert not (set(G.FACTUAL_FIELDS) & set(table["field"]))
    # the paragraph-level fields ARE still there — the build did real work
    assert {"party_attack", "topics", "proposal_values", "entities"} <= set(table["field"])


# ---------------------------------------------------------------------------
# coverage warning (M2 — advisory, NOT a hard exit)
# ---------------------------------------------------------------------------


def test_partial_opus_coverage_warns_but_still_returns_a_table(synthetic_layer, tmp_path, capsys):
    """M2 (judge minor): partial Opus coverage is SURFACED as a warning, not a
    fatal exit. build_agreement must still write the table over the overlap. If a
    future change makes this a hard bail, update this test deliberately."""
    sample_path, speeches = synthetic_layer
    table = G.build_agreement(sample_path=sample_path,
                              out_path=tmp_path / "a.parquet", speeches=speeches)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "< 100%" in out
    assert not table.empty  # published over the overlap, not suppressed


def test_full_opus_coverage_emits_no_warning(monkeypatch, synthetic_layer, tmp_path, capsys):
    """When Opus covers every sampled paragraph, the coverage line prints 100%
    and no WARNING. Add the missing docB para 1 to the Opus table."""
    sample_path, speeches = synthetic_layer
    opus_name = A._table_name("paragraph_annotations", G.OPUS_MODEL)
    opus = pd.read_parquet(ann.annotation_path(opus_name))
    opus = pd.concat([opus, pd.DataFrame([
        _prow("docB", 1, False, False, False, ["z"], "values")])], ignore_index=True)
    opus.to_parquet(ann.annotation_path(opus_name))

    G.build_agreement(sample_path=sample_path, out_path=tmp_path / "a.parquet", speeches=speeches)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "100.00%" in out


# ---------------------------------------------------------------------------
# key-integrity guards
# ---------------------------------------------------------------------------


def test_duplicate_primary_key_is_rejected(synthetic_layer, tmp_path):
    """The loaders enforce one-row-per-key on BOTH sides (the guard behind the
    one_to_one merge). A duplicated (doc_name, para_idx) in the primary table
    must abort the build rather than silently double-count or mis-join."""
    sample_path, speeches = synthetic_layer
    primary = pd.read_parquet(ann.annotation_path("paragraph_annotations"))
    dup = pd.concat([primary, primary.iloc[[0]]], ignore_index=True)
    dup.to_parquet(ann.annotation_path("paragraph_annotations"))
    with pytest.raises(ValueError, match="duplicate key"):
        G.build_agreement(sample_path=sample_path,
                          out_path=tmp_path / "a.parquet", speeches=speeches)


# ---------------------------------------------------------------------------
# clear failures when artifacts are absent
# ---------------------------------------------------------------------------


def test_build_errors_clearly_when_opus_artifacts_absent(monkeypatch, tmp_path):
    """No Opus table on disk -> a clear, actionable FileNotFoundError naming the
    submit command, not an opaque parquet read error."""
    corpus = pd.DataFrame([{"doc_name": "docA", "para_idx": 0, "text": "a0"}])
    corpus_path = tmp_path / "paragraphs.parquet"
    corpus.to_parquet(corpus_path)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", corpus_path)
    # primary present, Opus absent
    _write_para("paragraph_annotations", pd.DataFrame([
        _prow("docA", 0, True, False, True, ["x"], "proposal")]))
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps({"doc_names": ["docA"]}))

    with pytest.raises(FileNotFoundError, match="Opus second-opinion artifacts not found"):
        G.build_agreement(sample_path=sample_path, out_path=tmp_path / "a.parquet",
                          speeches=pd.DataFrame({"doc_name": ["docA"], "year": [1800]}))


def test_report_errors_clearly_when_agreement_parquet_absent(synthetic_layer, tmp_path):
    """agreement_report before build_agreement -> a clear error pointing at the
    build step, not a bare missing-file traceback."""
    sample_path, speeches = synthetic_layer
    with pytest.raises(FileNotFoundError, match="run build_agreement first"):
        G.agreement_report(sample_path=sample_path,
                           agreement_path=tmp_path / "does-not-exist.parquet",
                           report_path=tmp_path / "report.md", speeches=speeches)


# ---------------------------------------------------------------------------
# agreement_report happy path (an acceptance-criteria artifact)
# ---------------------------------------------------------------------------


def test_report_renders_all_required_sections(synthetic_layer, tmp_path):
    """The report is a named deliverable: overall + by-era per field, an entity
    section, and the highest-disagreement quotes. Render it end-to-end over the
    synthetic layer and assert the sections and the two-model provenance are
    present. Because this layer has partial Opus coverage, the coverage WARNING
    must also propagate into the markdown, not just stdout."""
    sample_path, speeches = synthetic_layer
    agreement_path = tmp_path / "agreement.parquet"
    G.build_agreement(sample_path=sample_path, out_path=agreement_path, speeches=speeches)

    report_path = tmp_path / "report.md"
    result = G.agreement_report(sample_path=sample_path, agreement_path=agreement_path,
                                report_path=report_path, speeches=speeches)
    assert result == report_path
    md = report_path.read_text()

    assert "# Inter-model annotation agreement" in md
    assert "## Agreement by field and era" in md
    assert "## Entities" in md
    assert "highest-disagreement paragraphs" in md
    # both models named, and the "model is the only variable" framing preserved
    assert "claude-sonnet-5" in md and "claude-opus-4-8" in md
    # partial-coverage warning surfaced in the document itself
    assert "WARNING" in md


# ---------------------------------------------------------------------------
# REVIEW blocker regression: null cells in the top-disagreement render path
# ---------------------------------------------------------------------------


def test_report_survives_null_topics_and_flags_in_top_disagreements(
        synthetic_layer, tmp_path):
    """REVIEW blocker regression: a null topics cell on the Opus side is
    selection-BIASED into the top-disagreement list (null vs a real list scores
    maximal topic disagreement), so the qualitative render path must apply the
    same null-coercion as the metric path instead of raising TypeError — and a
    null flag must render as 'null', not bool(nan)'s misleading True."""
    sample_path, speeches = synthetic_layer
    opus = pd.DataFrame([
        _prow("docA", 0, None, False, True, None, "proposal"),  # null flag + null topics
        _prow("docA", 1, True, False, True, ["y"], "proposal"),
        _prow("docB", 0, True, True, False, ["z"], "proposal"),
    ])
    _write_para(A._table_name("paragraph_annotations", G.OPUS_MODEL), opus)

    agreement_path = tmp_path / "agreement.parquet"
    G.build_agreement(sample_path=sample_path, out_path=agreement_path,
                      speeches=speeches)
    report_path = tmp_path / "report.md"
    G.agreement_report(sample_path=sample_path, agreement_path=agreement_path,
                       report_path=report_path, speeches=speeches)
    md = report_path.read_text()
    assert "party=null" in md                       # null flag rendered explicitly
    assert "primary: ['x'] | opus: []" in md        # null topics coerced, not crashed
