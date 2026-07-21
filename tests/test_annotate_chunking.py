"""Build 1 — chunked request path (Option A convergence lever).

A paragraph-unit speech can be split into <= N-paragraph chunk requests, each its
own custom_id. Chunks of one speech all key on the same doc_name, so their rows
merge into full coverage on ingest with NO keying change — `_already_ingested`
marks the speech done once every chunk lands. Speech-unit specs have no array to
chunk, so the combination is rejected. Everything here is offline.
"""

from __future__ import annotations

import json
import re
import types

import pandas as pd
import pytest

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann

_DOC = "/the-presidency/presidential-speeches/synthetic-chunk-1901"
_PRESIDENT = "Zebediah Q. Fauxington"
_TITLE = "Fifth Annual Message of Zebediah Q. Fauxington"


def _speech_and_paras(n: int = 5):
    speeches = pd.DataFrame({
        "doc_name": [_DOC], "title": [_TITLE], "president": [_PRESIDENT],
        "year": [1901], "decade": [1900], "date": [pd.Timestamp("1901-12-01")],
        "transcript": ["x"],
    })
    paras = pd.DataFrame({
        "doc_name": [_DOC] * n,
        "para_idx": list(range(n)),
        "text": [f"Paragraph {i} concerns the tariff and the trusts." for i in range(n)],
        "word_count": [8] * n,
    })
    return speeches, paras


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_chunking_splits_speech_into_size_bounded_requests():
    sp, pa = _speech_and_paras(5)
    reqs, idx = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa, chunk_size=2)
    assert len(reqs) == 3  # 2 + 2 + 1
    chunk_idxs = [idx[r["custom_id"]]["para_idxs"] for r in reqs]
    assert chunk_idxs == [[0, 1], [2, 3], [4]]                    # each chunk's exact keys
    assert sorted(i for c in chunk_idxs for i in c) == [0, 1, 2, 3, 4]  # together cover the speech


def test_chunk_custom_ids_are_unique_stable_legal_and_carry_chunk_index():
    sp, pa = _speech_and_paras(5)
    reqs, idx = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa, chunk_size=2)
    cids = [r["custom_id"] for r in reqs]
    assert len(set(cids)) == len(cids)                                   # unique
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c) for c in cids)    # legal Batches charset
    reqs2, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa, chunk_size=2)
    assert [r["custom_id"] for r in reqs2] == cids                       # stable across rebuilds
    assert [idx[c]["chunk"] for c in cids] == [0, 1, 2]                  # index carries chunk index
    assert all(c.endswith(f"-c{idx[c]['chunk']}") for c in cids)


def test_unchunked_id_and_index_unchanged_by_the_chunk_feature():
    sp, pa = _speech_and_paras(3)
    reqs, idx = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa)  # no chunking
    cid = reqs[0]["custom_id"]
    assert cid == A._custom_id("paragraph_annotations", _DOC)  # byte-identical to before
    assert "chunk" not in idx[cid]                            # no chunk marker when unchunked


def test_per_chunk_count_contract_masking_and_max_tokens():
    sp, pa = _speech_and_paras(5)
    reqs, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa, chunk_size=2)
    first, last = reqs[0], reqs[-1]
    fbody, lbody = (r["params"]["messages"][0]["content"] for r in (first, last))

    # the count contract states THIS chunk's paragraph count, not the speech's
    assert "Return EXACTLY 2 annotation object(s)" in fbody
    assert "Return EXACTLY 1 annotation object(s)" in lbody
    # max_tokens is sized to the chunk (2000 + 130*n)
    assert first["params"]["max_tokens"] == 2000 + 130 * 2
    assert last["params"]["max_tokens"] == 2000 + 130 * 1

    for r in reqs:
        body = r["params"]["messages"][0]["content"]
        assert "Fauxington" not in body and "Annual Message" not in body  # masking preserved
        assert "1900s" in body                                            # decade injected
        assert r["params"]["output_config"]["effort"] == "medium"         # judgment effort


def test_chunk_size_rejected_for_speech_unit_spec():
    sp, pa = _speech_and_paras(5)
    with pytest.raises(ValueError, match="paragraph-unit specs only"):
        A.build_requests(A.FIELD_SPECS["speech_annotations"], sp, pa, chunk_size=2)


def test_chunk_size_below_one_rejected():
    sp, pa = _speech_and_paras(5)
    with pytest.raises(ValueError, match=">= 1"):
        A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa, chunk_size=0)


def test_chunk_requests_pass_the_paid_path_validator():
    sp, pa = _speech_and_paras(5)
    reqs, _ = A.build_requests(A.FIELD_SPECS["paragraph_annotations"], sp, pa, chunk_size=2)
    for req in reqs:
        A._validate_request(req)  # must not raise (thinking disabled, valid SDK shape)


# ---------------------------------------------------------------------------
# ingest convergence — chunks of one speech merge into full coverage
# ---------------------------------------------------------------------------


def test_chunks_of_one_speech_merge_into_full_coverage(forge_run, args):
    """Two chunk requests for one speech (paras 0-1 and 2-3) ingest into a single
    fully-covered speech, and _already_ingested then marks it done."""
    idx = {
        "placeholder-c0": {"doc_name": "docA", "spec": "placeholder",
                           "unit": "paragraph", "para_idxs": [0, 1], "chunk": 0},
        "placeholder-c1": {"doc_name": "docA", "spec": "placeholder",
                           "unit": "paragraph", "para_idxs": [2, 3], "chunk": 1},
    }
    u = {"input_tokens": 50, "output_tokens": 5,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    l0 = forge_run.succeeded_line("placeholder-c0", [(0, "placeholder"), (1, "placeholder")], u)
    l1 = forge_run.succeeded_line("placeholder-c1", [(2, "placeholder"), (3, "placeholder")], u)
    forge_run("chunks", index=idx, result_lines=[l0, l1],
              submitted_custom_ids=["placeholder-c0", "placeholder-c1"])

    A.cmd_ingest(args(run_id="chunks"))

    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1, 2, 3]     # both chunks landed
    assert set(rows["doc_name"]) == {"docA"}            # merged under one doc_name

    paragraphs = pd.DataFrame({"doc_name": ["docA"] * 4, "para_idx": [0, 1, 2, 3]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert done["placeholder"] == {"docA"}              # coverage-based done, no keying change


def test_partial_chunk_coverage_leaves_speech_re_requestable(forge_run, args):
    """Only chunk 0 comes back (paras 0-1); the speech (0..3) is not fully
    covered, so it is not marked done and the next submit re-requests it."""
    idx = {
        "placeholder-c0": {"doc_name": "docA", "spec": "placeholder",
                           "unit": "paragraph", "para_idxs": [0, 1], "chunk": 0},
        "placeholder-c1": {"doc_name": "docA", "spec": "placeholder",
                           "unit": "paragraph", "para_idxs": [2, 3], "chunk": 1},
    }
    u = {"input_tokens": 10, "output_tokens": 2,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    l0 = forge_run.succeeded_line("placeholder-c0", [(0, "placeholder"), (1, "placeholder")], u)
    err = forge_run.errored_line("placeholder-c1", "overloaded_error")  # chunk 1 fails, retryable
    forge_run("halfchunks", index=idx, result_lines=[l0, err],
              submitted_custom_ids=["placeholder-c0", "placeholder-c1"])

    A.cmd_ingest(args(run_id="halfchunks"))

    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1]           # only chunk 0's rows
    paragraphs = pd.DataFrame({"doc_name": ["docA"] * 4, "para_idx": [0, 1, 2, 3]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert "docA" not in done["placeholder"]            # not done -> re-requestable


def test_ingest_manifest_records_chunk_escalated_docs(forge_run, args):
    """A chunk request (index carries `chunk`) is recorded in the manifest notes
    as an amendment-#2 chunk-escalated (partial-context) speech."""
    idx = {"placeholder-c0": {"doc_name": "docChunked", "spec": "placeholder",
                              "unit": "paragraph", "para_idxs": [0], "chunk": 0}}
    u = {"input_tokens": 10, "output_tokens": 1,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    l0 = forge_run.succeeded_line("placeholder-c0", [(0, "placeholder")], u)
    forge_run("chunknote", index=idx, result_lines=[l0],
              submitted_custom_ids=["placeholder-c0"])

    A.cmd_ingest(args(run_id="chunknote"))

    m = ann.read_manifest("chunknote")
    assert "chunk-escalated" in m.notes and "docChunked" in m.notes


def test_chunk_returning_a_subset_of_its_own_idxs_keeps_rows_and_stays_re_requestable(
    forge_run, args
):
    """A chunk sent [0,1,2] but returning only [0,1]: the returned rows are kept
    (correctly keyed), the chunk is recorded INCOMPLETE, and — the speech is not
    fully covered — it is not sealed done and stays re-requestable."""
    idx = {"placeholder-c0": {"doc_name": "docA", "spec": "placeholder",
                              "unit": "paragraph", "para_idxs": [0, 1, 2], "chunk": 0}}
    u = {"input_tokens": 30, "output_tokens": 3,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    line = forge_run.succeeded_line(  # returns 0,1 — a strict subset of the sent 0,1,2
        "placeholder-c0", [(0, "placeholder"), (1, "placeholder")], u)
    forge_run("subsetchunk", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="subsetchunk"))

    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1]      # the subset that came back is kept, keyed
    m = ann.read_manifest("subsetchunk")
    assert "INCOMPLETE" in m.notes                 # the dropped para is flagged
    paragraphs = pd.DataFrame({"doc_name": ["docA"] * 3, "para_idx": [0, 1, 2]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert "docA" not in done["placeholder"]        # missing para 2 -> re-requestable


# --- entity supersession through a chunked re-request (judgment spec) ---------


def _judgment_para(para_idx: int, entities: list[dict]) -> dict:
    return {"para_idx": para_idx, "topics": ["Labor, Wages & Working Conditions"],
            "party_attack": False, "enemy_naming": False, "zero_sum": False,
            "proposal_values": "neither", "entities": entities}


def _judgment_line(cid: str, items: list[dict], usage: dict) -> dict:
    return {"custom_id": cid, "result": {"type": "succeeded", "message": {
        "usage": usage, "content": [{"type": "text", "text": json.dumps({"annotations": items})}]}}}


def test_entity_supersession_through_a_chunked_re_request(forge_run, args):
    """A paragraph re-annotated via a later chunk must SUPERSEDE its prior entity
    rows, never append. Alpha (round 1) -> Beta (round 2, different) -> [] (round
    3, zero) — each round's entities fully replace the paragraph's prior rows."""
    u = {"input_tokens": 40, "output_tokens": 8,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    cid = "paragraph_annotations-e-c0"
    idx = {cid: {"doc_name": "docE", "spec": "paragraph_annotations",
                 "unit": "paragraph", "para_idxs": [0], "chunk": 0}}

    def _ingest(run_id, entities):
        line = _judgment_line(cid, [_judgment_para(0, entities)], u)
        forge_run(run_id, index=idx, result_lines=[line],
                  submitted_custom_ids=[cid], specs=["paragraph_annotations"])
        A.cmd_ingest(args(run_id=run_id))
        return pd.read_parquet(ann.annotation_path("paragraph_entities"))

    # round 1: docE/0 names Alpha
    ent1 = _ingest("r1", [{"name": "Alpha", "type": "person", "stance": "adversarial"}])
    assert list(ent1["entity"]) == ["Alpha"]

    # round 2: same chunk re-requested, docE/0 now names Beta only -> Alpha superseded
    ent2 = _ingest("r2", [{"name": "Beta", "type": "person", "stance": "neutral"}])
    assert list(ent2["entity"]) == ["Beta"]
    assert set(ent2["run_id"]) == {"r2"}  # not appended to Alpha's rows

    # round 3: re-requested with ZERO entities -> the paragraph's rows are cleared
    ent3 = _ingest("r3", [])
    assert len(ent3) == 0


# ---------------------------------------------------------------------------
# submit persists only the SUBMITTED chunk index (BLOCKER regression)
# ---------------------------------------------------------------------------


def test_submit_persists_only_submitted_chunk_index_entries(
    monkeypatch, tmp_path, patch_anthropic, args
):
    """build_requests emits an index entry per speech, but the resume filter trims
    the REQUESTS. cmd_submit must persist only the SUBMITTED cids' index — else a
    --chunk-size round marks every speech chunk-escalated in the amendment
    provenance, not just the stragglers actually sent.

    Prior round covers docA fully (resume skips it); the chunk round submits only
    docB. The persisted requests_index must contain only docB's chunk cids, and
    (once a manifest exists) `_all_chunk_escalated_docs()` must be [docB]."""
    corpus = pd.DataFrame({
        "doc_name": ["docA", "docA", "docB", "docB"],
        "para_idx": [0, 1, 0, 1],
        "text": [f"Paragraph {i} about the tariff." for i in [0, 1, 0, 1]],
        "word_count": [5, 5, 5, 5],
    })
    ppath = tmp_path / "paras.parquet"
    corpus.to_parquet(ppath, index=False)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", ppath)
    monkeypatch.setattr(A, "load", lambda: pd.DataFrame({
        "doc_name": ["docA", "docB"], "title": ["A", "B"], "president": ["Zed", "Yan"],
        "year": [1901, 1902], "decade": [1900, 1900],
        "date": [pd.Timestamp("1901-01-01"), pd.Timestamp("1902-01-01")],
        "transcript": ["x", "y"],
    }))

    # docA is already fully annotated -> the resume filter skips it.
    ann.write_annotations("paragraph_annotations", pd.DataFrame({
        "doc_name": ["docA", "docA"], "para_idx": [0, 1],
        "topics": [[], []], "party_attack": [False, False], "enemy_naming": [False, False],
        "zero_sum": [False, False], "proposal_values": ["neither", "neither"],
        "run_id": ["r0", "r0"],
    }), "paragraph")

    patch_anthropic(create_batch=types.SimpleNamespace(id="batch_chunk",
                                                       processing_status="in_progress"))
    A.cmd_submit(args(run_id="chunkrun", spec=["paragraph_annotations"], chunk_size=2,
                      yes=True, max_cost_usd=100.0))

    idx = json.loads((ann.RUNS_DIR / "chunkrun" / "requests_index.json").read_text())
    assert idx, "persisted index must not be empty"
    assert {m["doc_name"] for m in idx.values()} == {"docB"}      # docA trimmed out
    assert all(m.get("chunk") is not None for m in idx.values())  # all chunk entries

    # once ingest has written a manifest, the amendment names ONLY docB
    ann.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    ann.manifest_path("chunkrun").write_text("{}")
    assert A._all_chunk_escalated_docs() == ["docB"]
