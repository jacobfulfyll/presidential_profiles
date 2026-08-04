"""Build 2 + 3 — converged QA mode and amendment recording.

CONVERGED mode gates the corpus-wide, post-convergence acceptance criterion:
100% paragraph coverage AND 100% speech coverage (both real specs), plus the
non-degenerate flag and entity-substring gates evaluated corpus-wide. The report
records the pre-registration amendment explicitly (coverage gate redefined; the
chunk-escalated stragglers listed by doc_name, or "none"). All synthetic/offline.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann

_DOCS = ["docA", "docB"]
_NPARAS = {"docA": 3, "docB": 2}  # 5 corpus paragraphs total


def _setup(monkeypatch, tmp_path, *, para_keys=None, speech_docs=None, chunk_doc=None):
    """Lay down a 2-speech corpus and both annotation tables; knobs punch a hole
    in paragraph coverage (`para_keys`), speech coverage (`speech_docs`), or add a
    chunk-escalated run dir (`chunk_doc`)."""
    rows = []
    for d in _DOCS:
        for i in range(_NPARAS[d]):
            rows.append({"doc_name": d, "para_idx": i,
                         "text": f"Paragraph {i} names Senator Foo{i} of {d}.", "word_count": 8})
    corpus = pd.DataFrame(rows)
    ppath = tmp_path / "paras.parquet"
    corpus.to_parquet(ppath, index=False)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", ppath)
    monkeypatch.setattr(A, "load", lambda: pd.DataFrame(
        {"doc_name": _DOCS, "decade": [1850, 1860], "year": [1855, 1865]}))

    keys = para_keys if para_keys is not None else [(d, i) for d in _DOCS for i in range(_NPARAS[d])]
    n = len(keys)
    pj = pd.DataFrame({
        "doc_name": [d for d, _ in keys],
        "para_idx": [i for _, i in keys],
        "topics": [["Labor, Wages & Working Conditions"] for _ in range(n)],
        "party_attack": [j % 3 == 0 for j in range(n)],  # ~33% — non-degenerate
        "enemy_naming": [j % 5 == 0 for j in range(n)],  # ~20%
        "zero_sum": [j % 5 == 1 for j in range(n)],      # ~20%
        "proposal_values": ["neither"] * n,
        "run_id": ["r1"] * n,
    })
    ann.write_annotations("paragraph_annotations", pj, "paragraph")

    ann.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({  # entity names appear verbatim in their paragraph text
        "doc_name": ["docA", "docA"], "para_idx": [0, 0],
        "entity": ["Senator Foo0", "Senator Foo0"], "type": ["person", "person"],
        "stance": ["adversarial", "neutral"], "run_id": ["r1", "r1"],
    }).to_parquet(ann.annotation_path("paragraph_entities"), index=False)

    sdocs = speech_docs if speech_docs is not None else _DOCS
    ann.write_annotations("speech_annotations", pd.DataFrame({
        "doc_name": sdocs,
        "speech_type": ["inaugural_address"] * len(sdocs),
        "audience": ["congress"] * len(sdocs),
        "medium": ["written_message"] * len(sdocs),
        "run_id": ["r1"] * len(sdocs),
    }), "speech")

    if chunk_doc:  # a final-round chunked request that was INGESTED -> amendment provenance
        rd = ann.RUNS_DIR / "round3"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "requests_index.json").write_text(json.dumps({
            "paragraph_annotations-deadbeef-c0": {
                "doc_name": chunk_doc, "spec": "paragraph_annotations",
                "unit": "paragraph", "para_idxs": [0], "chunk": 0}}))
        # ingest writes a manifest — the amendment list only counts ingested runs.
        ann.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        ann.manifest_path("round3").write_text("{}")


def _cchecks():
    return {name: passed for name, passed, _ in A._compute_qa(None, converged=True)["checks"]}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def test_converged_passes_on_a_fully_covered_corpus(args, monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path)
    A.cmd_qa(args(converged=True, out=str(tmp_path / "qa.md")))  # must NOT raise
    out = capsys.readouterr().out
    assert "CONVERGED QA PASSED" in out
    checks = _cchecks()
    assert checks["paragraph_coverage==100%"] is True
    assert checks["speech_coverage==100%"] is True
    assert all(checks.values())


def test_converged_fails_when_a_paragraph_is_missing(args, monkeypatch, tmp_path):
    # drop docB/1 -> 4 of 5 paragraphs covered
    keys = [("docA", 0), ("docA", 1), ("docA", 2), ("docB", 0)]
    _setup(monkeypatch, tmp_path, para_keys=keys)
    assert _cchecks()["paragraph_coverage==100%"] is False
    with pytest.raises(SystemExit, match="CONVERGED QA FAILED"):
        A.cmd_qa(args(converged=True, out=str(tmp_path / "qa.md")))


def test_converged_fails_when_a_speech_row_is_missing(args, monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, speech_docs=["docA"])  # docB has no speech row
    checks = _cchecks()
    assert checks["speech_coverage==100%"] is False
    assert checks["paragraph_coverage==100%"] is True  # paragraphs are complete
    with pytest.raises(SystemExit, match="CONVERGED QA FAILED"):
        A.cmd_qa(args(converged=True, out=str(tmp_path / "qa.md")))


def test_converged_gate_is_stricter_than_the_pilot_gate(monkeypatch, tmp_path):
    """A corpus at 99.5% (199/200) PASSES the pilot's >=99% gate but FAILS the
    converged ==100% gate — the redefinition actually bites at the boundary a
    coarser threshold would wave through."""
    n = 200
    corpus = pd.DataFrame({
        "doc_name": ["docBig"] * n,
        "para_idx": list(range(n)),
        "text": [f"Paragraph {i} names Senator Foo{i}." for i in range(n)],
        "word_count": [6] * n,
    })
    ppath = tmp_path / "big.parquet"
    corpus.to_parquet(ppath, index=False)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", ppath)
    monkeypatch.setattr(A, "load", lambda: pd.DataFrame(
        {"doc_name": ["docBig"], "decade": [1900], "year": [1905]}))

    covered = n - 1  # 199 of 200 -> 99.5%: one paragraph short of complete
    ann.write_annotations("paragraph_annotations", pd.DataFrame({
        "doc_name": ["docBig"] * covered,
        "para_idx": list(range(covered)),
        "topics": [["Labor, Wages & Working Conditions"] for _ in range(covered)],
        "party_attack": [j % 3 == 0 for j in range(covered)],
        "enemy_naming": [j % 5 == 0 for j in range(covered)],
        "zero_sum": [j % 5 == 1 for j in range(covered)],
        "proposal_values": ["neither"] * covered,
        "run_id": ["r1"] * covered,
    }), "paragraph")

    pilot = {name: passed for name, passed, _ in A._compute_qa(None, converged=False)["checks"]}
    conv = {name: passed for name, passed, _ in A._compute_qa(None, converged=True)["checks"]}
    assert pilot["paragraph_coverage>=99%"] is True    # 99.5% clears the pilot gate
    assert conv["paragraph_coverage==100%"] is False   # ...but not the converged gate


# ---------------------------------------------------------------------------
# amendment recording (Build 3)
# ---------------------------------------------------------------------------


def test_report_records_amendment_and_lists_chunk_escalated_stragglers(args, monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, chunk_doc="docB")
    out = tmp_path / "qa.md"
    A.cmd_qa(args(converged=True, out=str(out)))
    text = out.read_text()
    assert "## Amendment (pre-registration)" in text
    assert "Amendment #1" in text and "post-convergence == 100%" in text
    assert "Amendment #2" in text and "resubmitted to convergence" in text
    assert "docB" in text  # the chunk-escalated straggler is named


def test_report_says_none_when_no_speech_was_chunked(args, monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)  # no chunk_doc -> no chunked run dir
    out = tmp_path / "qa.md"
    A.cmd_qa(args(converged=True, out=str(out)))
    text = out.read_text()
    assert "## Amendment (pre-registration)" in text
    assert "none — no speech required chunking" in text


def test_converged_and_pilot_are_mutually_exclusive(args, tmp_path):
    with pytest.raises(SystemExit, match="not both"):
        A.cmd_qa(args(pilot=True, converged=True, out=str(tmp_path / "qa.md")))


def test_amendment_lists_only_chunk_runs_that_were_ingested(redirect_annotation_dirs):
    """The straggler list must name only speeches whose chunk data LANDED. A run
    with chunk markers but no manifest (a dry-run / un-ingested submit) is
    excluded; a run WITH a manifest (ingest wrote one) is included."""
    # ingested run: chunk markers + a manifest
    rd_in = ann.RUNS_DIR / "ingested"
    rd_in.mkdir(parents=True, exist_ok=True)
    (rd_in / "requests_index.json").write_text(json.dumps({
        "paragraph_annotations-aa-c0": {"doc_name": "docIngested", "spec": "paragraph_annotations",
                                        "unit": "paragraph", "para_idxs": [0], "chunk": 0}}))
    ann.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    ann.manifest_path("ingested").write_text("{}")  # a manifest exists -> this run was ingested

    # dry-run / un-ingested run: chunk markers but NO manifest
    rd_dry = ann.RUNS_DIR / "dryrun"
    rd_dry.mkdir(parents=True, exist_ok=True)
    (rd_dry / "requests_index.json").write_text(json.dumps({
        "paragraph_annotations-bb-c0": {"doc_name": "docDryrun", "spec": "paragraph_annotations",
                                        "unit": "paragraph", "para_idxs": [0], "chunk": 0}}))

    assert A._all_chunk_escalated_docs() == ["docIngested"]  # docDryrun excluded
