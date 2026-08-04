"""Ingest: provenance integrity + coverage gates.

Covers regressions:
  #1 zero-result cache is refused, not recorded
  #2 atomic results cache (a mid-stream failure leaves no usable cache)
  #3 stale-batch refetch (cached A + state B must not be silently paired)
  #7 partial coverage is caught (incomplete speech not sealed done; extra-index
     hallucination raises) and a resume re-requests exactly the gap
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann


def _index(doc="docA", cid="placeholder-a", para_idxs=(0, 1, 2)):
    return {cid: {"doc_name": doc, "spec": "placeholder", "unit": "paragraph",
                  "para_idxs": list(para_idxs)}}


# ---------------------------------------------------------------------------
# happy path — the paid path we are locking in
# ---------------------------------------------------------------------------


def test_ingest_writes_keyed_rows_and_actual_cost(forge_run, args):
    idx = _index(para_idxs=(0, 1))
    usage = {"input_tokens": 1000, "output_tokens": 200,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    line = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")], usage)
    forge_run("h1", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="h1"))

    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1]
    assert set(rows["doc_name"]) == {"docA"}
    assert list(rows["run_id"].unique()) == ["h1"]

    m = ann.read_manifest("h1")
    assert m.batch_id == "batch_test"
    assert m.n_requests == 1
    assert m.cost_usd == round(A.cost_usd(**usage), 4)  # ACTUAL cost from usage
    assert m.input_tokens == 1000 and m.output_tokens == 200


# ---------------------------------------------------------------------------
# regression #1 — zero-result cache refused, not recorded
# ---------------------------------------------------------------------------


def test_zero_byte_cache_raises_and_writes_no_manifest(forge_run, args):
    forge_run("z1", index=_index(para_idxs=(0,)), results_text="",
              results_batch_id="batch_test")  # matches batch_id -> not treated as stale
    with pytest.raises(SystemExit, match="zero results"):
        A.cmd_ingest(args(run_id="z1"))
    assert not ann.manifest_path("z1").exists()


# ---------------------------------------------------------------------------
# regression #2 — atomic results cache
# ---------------------------------------------------------------------------


def test_fetch_that_raises_midstream_leaves_no_usable_cache(patch_anthropic, redirect_annotation_dirs):
    run_dir = ann.RUNS_DIR / "atomic"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "results.jsonl"

    class _Result:  # minimal streamed result: _fetch_results only calls to_json
        def to_json(self, indent=None):
            return json.dumps({"custom_id": "placeholder-a", "result": {"type": "succeeded"}})

    def stream(batch_id):
        # yield one good line, then die partway — the classic truncation case
        yield _Result()
        raise RuntimeError("connection dropped mid-stream")

    patch_anthropic(results_source=stream)
    with pytest.raises(RuntimeError, match="mid-stream"):
        A._fetch_results("atomic", "batch_test", raw_path)

    assert not raw_path.exists()  # no truncated results.jsonl a later ingest would accept
    assert not raw_path.with_name("results.jsonl.tmp").exists()  # the tmp was cleaned up too


def test_fetch_that_returns_zero_results_refuses_to_cache(patch_anthropic, redirect_annotation_dirs):
    run_dir = ann.RUNS_DIR / "empty"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "results.jsonl"
    patch_anthropic(results_source=lambda bid: iter([]))
    with pytest.raises(SystemExit, match="zero results"):
        A._fetch_results("empty", "batch_test", raw_path)
    assert not raw_path.exists()
    assert not raw_path.with_name("results.jsonl.tmp").exists()


# ---------------------------------------------------------------------------
# regression #3 — stale-batch refetch, never silently paired
# ---------------------------------------------------------------------------


def test_cached_results_from_a_different_batch_trigger_refetch_not_pairing(forge_run, args, patch_anthropic):
    """state.batch_id = B but the cache is tagged A. Ingest must refetch B's
    results — never parse A's cache and stamp B's id onto A's tokens. Here the
    refetch fails (no creds), which is fine: what matters is that NO manifest
    pairing B with A's usage is written."""
    idx = _index(para_idxs=(0,))
    good_line = forge_run.succeeded_line(
        "placeholder-a", [(0, "placeholder")],
        {"input_tokens": 999999, "output_tokens": 0,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    )
    forge_run("stale", index=idx, result_lines=[good_line],
              batch_id="batchB", results_batch_id="batchA")

    def refetch_fails(batch_id):
        raise RuntimeError("refetch attempted for batchB but no credentials")

    patch_anthropic(results_source=refetch_fails)
    with pytest.raises(RuntimeError, match="refetch attempted"):
        A.cmd_ingest(args(run_id="stale"))

    # The cardinal sin would be a manifest pairing batchB's id with batchA's
    # 999,999 input tokens. It must not exist.
    assert not ann.manifest_path("stale").exists()


# ---------------------------------------------------------------------------
# regression #7 — partial coverage caught, extra-index hallucination raises
# ---------------------------------------------------------------------------


def test_incomplete_response_is_recorded_not_sealed_done(forge_run, args):
    """Sent para_idx 0,1,2 but only 0,1 returned: the returned rows are kept
    (correctly keyed, paid for), the missing index is reported, and the speech
    is NOT marked done."""
    idx = _index(para_idxs=(0, 1, 2))
    line = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")],
                                    {"input_tokens": 10, "output_tokens": 2})
    forge_run("part", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="part"))

    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1]  # idx 2 was dropped, not fabricated
    m = ann.read_manifest("part")
    assert "INCOMPLETE" in m.notes

    # and the resume gate agrees: with the corpus expecting 0,1,2 for docA, the
    # speech is not done, so the next submit re-requests exactly the gap.
    paragraphs = pd.DataFrame({"doc_name": ["docA", "docA", "docA"], "para_idx": [0, 1, 2]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert "docA" not in done["placeholder"]


def test_returned_index_never_sent_is_quarantined_not_aborted(forge_run, args):
    """A hallucinated para_idx that was never in the request must not become a
    key — but it must QUARANTINE the speech (recorded retryable, no rows written)
    rather than abort ingest of the whole paid batch. The manifest is still
    written, and the speech stays re-requestable."""
    idx = _index(para_idxs=(0, 1))
    line = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (5, "placeholder")],
                                    {"input_tokens": 10, "output_tokens": 2})
    forge_run("extra", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="extra"))  # must NOT raise

    # No rows written for the quarantined speech (the bad idx never became a key).
    assert not ann.annotation_path("placeholder").exists()
    m = ann.read_manifest("extra")  # manifest IS written
    assert "retryable failures: 1" in m.notes
    # not sealed done -> the next submit re-requests it
    paragraphs = pd.DataFrame({"doc_name": ["docA", "docA"], "para_idx": [0, 1]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert "docA" not in done["placeholder"]


def test_returned_index_twice_is_quarantined_not_aborted(forge_run, args):
    """A duplicate echoed para_idx quarantines the speech (recorded retryable,
    no rows) instead of raising and stranding the batch."""
    idx = _index(para_idxs=(0, 1))
    line = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (0, "placeholder")],
                                    {"input_tokens": 10, "output_tokens": 2})
    forge_run("twice", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="twice"))  # must NOT raise

    assert not ann.annotation_path("placeholder").exists()
    m = ann.read_manifest("twice")
    assert "retryable failures: 1" in m.notes


def test_one_bad_speech_quarantined_others_still_ingested(forge_run, args):
    """The point of quarantine: a single degenerate/duplicate response does not
    block ingesting the rest of a paid batch. The good speech's rows land; the
    bad one is dropped whole and left re-requestable."""
    idx = {
        "placeholder-a": {"doc_name": "docA", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
        "placeholder-b": {"doc_name": "docB", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
    }
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")],
                                    {"input_tokens": 100, "output_tokens": 10})
    # docB returns a duplicated para_idx -> quarantined
    bad = forge_run.succeeded_line("placeholder-b", [(0, "placeholder"), (0, "placeholder")],
                                   {"input_tokens": 50, "output_tokens": 5})
    forge_run("mixq", index=idx, result_lines=[good, bad],
              submitted_custom_ids=["placeholder-a", "placeholder-b"])

    A.cmd_ingest(args(run_id="mixq"))  # must not raise

    rows = ann.load_paragraph_annotations("placeholder")
    assert set(rows["doc_name"]) == {"docA"}  # only the good speech contributed rows
    assert sorted(rows["para_idx"]) == [0, 1]

    m = ann.read_manifest("mixq")
    assert m.n_requests == 2  # both results counted
    assert "retryable failures: 1" in m.notes
    # only docB is left re-requestable
    paragraphs = pd.DataFrame({"doc_name": ["docA", "docA", "docB", "docB"],
                               "para_idx": [0, 1, 0, 1]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert done["placeholder"] == {"docA"}


def test_truncated_json_is_quarantined_not_aborted(forge_run, args):
    """A response truncated against max_tokens arrives as syntactically incomplete
    JSON (cut mid-object). `output_config.format` only guarantees valid JSON for a
    COMPLETE response, so json.loads would raise — and the raise sits BEFORE the
    para_idx quarantine, so without a guard it aborts the whole paid batch. It
    must instead quarantine that speech (retryable, no rows) and keep ingesting."""
    idx = {
        "placeholder-a": {"doc_name": "docA", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
        "placeholder-b": {"doc_name": "docB", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
    }
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")],
                                    {"input_tokens": 100, "output_tokens": 10})
    # docB's text is a truncated array — valid-JSON-so-far but cut mid-object.
    truncated = {
        "custom_id": "placeholder-b",
        "result": {"type": "succeeded", "message": {
            "usage": {"input_tokens": 80, "output_tokens": 8,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "content": [{"type": "text",
                         "text": '{"annotations": [{"para_idx": 0, "label": "placeholde'}],
        }},
    }
    forge_run("trunc_json", index=idx, result_lines=[good, truncated],
              submitted_custom_ids=["placeholder-a", "placeholder-b"])

    A.cmd_ingest(args(run_id="trunc_json"))  # must NOT raise (was JSONDecodeError)

    # good speech landed; the truncated one wrote nothing
    rows = ann.load_paragraph_annotations("placeholder")
    assert set(rows["doc_name"]) == {"docA"}
    assert sorted(rows["para_idx"]) == [0, 1]

    m = ann.read_manifest("trunc_json")  # manifest still written
    assert m.n_requests == 2
    assert "retryable failures: 1" in m.notes
    # docB not sealed -> re-requestable
    paragraphs = pd.DataFrame({"doc_name": ["docA", "docA", "docB", "docB"],
                               "para_idx": [0, 1, 0, 1]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert done["placeholder"] == {"docA"}


def test_malformed_json_speech_unit_is_quarantined_not_aborted(forge_run, args):
    """The guard is BEFORE the unit split, so a malformed SPEECH-unit payload
    quarantines too (does not abort). Uses the real speech_annotations spec so no
    registry patching is needed."""
    idx = {"speech_annotations-a": {"doc_name": "docA", "spec": "speech_annotations",
                                     "unit": "speech", "para_idxs": []}}
    bad = {
        "custom_id": "speech_annotations-a",
        "result": {"type": "succeeded", "message": {
            "usage": {"input_tokens": 20, "output_tokens": 2,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "content": [{"type": "text", "text": '{"speech_type": "inaug'}],  # truncated
        }},
    }
    forge_run("speech_trunc", index=idx, result_lines=[bad], specs=["speech_annotations"])

    A.cmd_ingest(args(run_id="speech_trunc"))  # must NOT raise (was JSONDecodeError)

    assert not ann.annotation_path("speech_annotations").exists()  # nothing written
    m = ann.read_manifest("speech_trunc")
    assert "retryable failures: 1" in m.notes


def test_null_annotations_payload_is_quarantined_not_aborted(forge_run, args):
    """A JSON-VALID but shape-unexpected response ({"annotations": null}) would
    raise TypeError on `for item in None` and abort the whole paid batch. It must
    instead quarantine that speech ('malformed_payload') while the good speech in
    the same batch still ingests."""
    idx = {
        "placeholder-a": {"doc_name": "docA", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
        "placeholder-b": {"doc_name": "docB", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0]},
    }
    u = {"input_tokens": 50, "output_tokens": 5,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")], u)
    bad = {"custom_id": "placeholder-b", "result": {"type": "succeeded", "message": {
        "usage": u, "content": [{"type": "text", "text": json.dumps({"annotations": None})}]}}}
    forge_run("nullann", index=idx, result_lines=[good, bad],
              submitted_custom_ids=["placeholder-a", "placeholder-b"])

    A.cmd_ingest(args(run_id="nullann"))  # must NOT raise (was TypeError)

    rows = ann.load_paragraph_annotations("placeholder")
    assert set(rows["doc_name"]) == {"docA"}     # good speech landed; docB quarantined
    assert sorted(rows["para_idx"]) == [0, 1]
    m = ann.read_manifest("nullann")
    assert m.n_requests == 2                      # both results counted
    assert "retryable failures: 1" in m.notes


def test_missing_annotations_key_payload_is_quarantined_not_aborted(forge_run, args):
    """A succeeded response missing the `annotations` key (KeyError) quarantines
    as 'malformed_payload' rather than aborting; nothing is written for it."""
    idx = {"placeholder-a": {"doc_name": "docA", "spec": "placeholder",
                             "unit": "paragraph", "para_idxs": [0]}}
    bad = {"custom_id": "placeholder-a", "result": {"type": "succeeded", "message": {
        "usage": {"input_tokens": 10, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        "content": [{"type": "text", "text": json.dumps({"wrong_key": []})}]}}}
    forge_run("misskey", index=idx, result_lines=[bad])

    A.cmd_ingest(args(run_id="misskey"))  # must NOT raise (was KeyError)

    assert not ann.annotation_path("placeholder").exists()  # nothing written
    m = ann.read_manifest("misskey")
    assert "retryable failures: 1" in m.notes


# ---------------------------------------------------------------------------
# shape hardening: bad entities element quarantines; smuggled keys can't clobber
# ---------------------------------------------------------------------------


def _judgment_result(cid: str, items: list[dict], usage: dict) -> dict:
    return {"custom_id": cid, "result": {"type": "succeeded", "message": {
        "usage": usage, "content": [{"type": "text", "text": json.dumps({"annotations": items})}]}}}


def _jpara(para_idx: int, entities: list) -> dict:
    return {"para_idx": para_idx, "topics": [], "party_attack": False,
            "enemy_naming": False, "zero_sum": False, "proposal_values": "neither",
            "entities": entities}


def test_malformed_entities_element_is_quarantined_not_aborted(forge_run, args):
    """A schema-violating entities element (a bare string / null, not an object)
    on a succeeded judgment result would raise AttributeError in the entity
    explosion (`e.get(...)`) and abort the whole batch. It must instead quarantine
    that speech as 'malformed_payload' while a co-batch speech ingests cleanly."""
    cid_ok, cid_bad = "paragraph_annotations-ok", "paragraph_annotations-bad"
    idx = {
        cid_ok: {"doc_name": "docOK", "spec": "paragraph_annotations",
                 "unit": "paragraph", "para_idxs": [0]},
        cid_bad: {"doc_name": "docBad", "spec": "paragraph_annotations",
                  "unit": "paragraph", "para_idxs": [0]},
    }
    u = {"input_tokens": 30, "output_tokens": 4,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    good = _judgment_result(cid_ok, [_jpara(0, [{"name": "China", "type": "nation",
                                                 "stance": "neutral"}])], u)
    bad = _judgment_result(cid_bad, [_jpara(0, ["China"])], u)  # entities: ["China"] -> not objects
    forge_run("badent", index=idx, result_lines=[good, bad],
              submitted_custom_ids=[cid_ok, cid_bad], specs=["paragraph_annotations"])

    A.cmd_ingest(args(run_id="badent"))  # must NOT raise (was AttributeError in explosion)

    rows = ann.load_paragraph_annotations("paragraph_annotations")
    assert set(rows["doc_name"]) == {"docOK"}   # good speech landed; docBad quarantined
    ent = pd.read_parquet(ann.annotation_path("paragraph_entities"))
    assert set(ent["doc_name"]) == {"docOK"}    # only the good speech's entities exploded
    m = ann.read_manifest("badent")
    assert m.n_requests == 2
    assert "retryable failures: 1" in m.notes


def test_smuggled_doc_name_and_run_id_cannot_clobber_the_index_anchor(forge_run, args):
    """A model that smuggles `doc_name`/`run_id` fields into an item must not
    override the index-anchored key or the provenance pointer — trusted keys are
    spread LAST, so the anchored values always win."""
    idx = {"placeholder-a": {"doc_name": "docReal", "spec": "placeholder",
                             "unit": "paragraph", "para_idxs": [0]}}
    u = {"input_tokens": 10, "output_tokens": 1,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    payload = {"annotations": [{"para_idx": 0, "label": "placeholder",
                                "doc_name": "EVIL", "run_id": "EVIL"}]}
    line = {"custom_id": "placeholder-a", "result": {"type": "succeeded", "message": {
        "usage": u, "content": [{"type": "text", "text": json.dumps(payload)}]}}}
    forge_run("smuggle", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="smuggle"))

    rows = ann.load_paragraph_annotations("placeholder")
    assert list(rows["doc_name"]) == ["docReal"]  # index-anchored key, not "EVIL"
    assert list(rows["run_id"]) == ["smuggle"]    # run_id provenance, not "EVIL"
    assert list(rows["label"]) == ["placeholder"]  # the real payload field survives


def test_truncated_cache_missing_a_submitted_request_raises(forge_run, args):
    """Every submitted custom_id must come back. A cache missing one is
    truncated (or from another batch) and must be refused, not silently shorted."""
    idx = _index(para_idxs=(0,))
    idx["placeholder-b"] = {"doc_name": "docB", "spec": "placeholder",
                            "unit": "paragraph", "para_idxs": [0]}
    line = forge_run.succeeded_line("placeholder-a", [(0, "placeholder")],
                                    {"input_tokens": 10, "output_tokens": 2})
    # submitted both a and b, but only a came back
    forge_run("trunc", index=idx, result_lines=[line],
              submitted_custom_ids=["placeholder-a", "placeholder-b"])
    with pytest.raises(SystemExit, match="no result line"):
        A.cmd_ingest(args(run_id="trunc"))
    assert not ann.manifest_path("trunc").exists()


# ---------------------------------------------------------------------------
# resume coverage (the _already_ingested gate that drives regression #7 resume)
# ---------------------------------------------------------------------------


def test_already_ingested_marks_only_fully_covered_paragraph_speeches(redirect_annotation_dirs):
    paragraphs = pd.DataFrame({
        "doc_name": ["docA", "docA", "docA", "docB"],
        "para_idx": [0, 1, 2, 0],
    })
    # docA has only 0,1 on disk (gap at 2); docB is complete.
    ann.write_annotations("placeholder", pd.DataFrame({
        "doc_name": ["docA", "docA", "docB"],
        "para_idx": [0, 1, 0],
        "label": ["x", "x", "x"],
        "run_id": ["r", "r", "r"],
    }), unit="paragraph")
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert done["placeholder"] == {"docB"}


# ---------------------------------------------------------------------------
# errored / mixed-result ingest — one bad result must NOT abort ingest of all N
# (the IMPLEMENT-review risk, previously unexercised: _errored_line was unused)
# ---------------------------------------------------------------------------


def test_errored_result_does_not_abort_ingest_and_succeeded_rows_still_land(forge_run, args):
    """A batch whose results are a MIX of succeeded + errored must ingest the
    succeeded rows (correctly keyed) and NOT abort on the errored one."""
    idx = {
        "placeholder-a": {"doc_name": "docA", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
        "placeholder-b": {"doc_name": "docB", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0]},
    }
    usage = {"input_tokens": 500, "output_tokens": 40,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")], usage)
    bad = forge_run.errored_line("placeholder-b", "invalid_request")
    forge_run("mix", index=idx, result_lines=[good, bad],
              submitted_custom_ids=["placeholder-a", "placeholder-b"])

    A.cmd_ingest(args(run_id="mix"))  # must not raise on the errored line

    # (a) the succeeded speech's rows are written and keyed by (doc_name, para_idx)
    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1]
    assert set(rows["doc_name"]) == {"docA"}  # the errored docB wrote nothing

    m = ann.read_manifest("mix")
    # (b) ingest did not abort — a manifest exists and (d) counts BOTH results
    assert m.n_requests == 2
    # (c) the errored result is classified permanent (invalid_request)
    assert "errored=1" in m.notes
    assert "permanent (invalid_request) failures: 1" in m.notes
    assert "retryable failures: 0" in m.notes
    # actual cost reflects only the succeeded result's usage
    assert m.input_tokens == 500 and m.output_tokens == 40


def test_succeeded_but_textless_result_does_not_abort_ingest(forge_run, args):
    """A "succeeded" result with no text block (a refusal or an all-thinking
    response) once crashed the whole batch on next(): StopIteration on the paid,
    cached path, naming no custom_id. It must instead be recorded like a
    retryable failure — good rows still land, the text-less speech writes nothing,
    stays re-requestable, and the manifest reflects it."""
    idx = {
        "placeholder-a": {"doc_name": "docA", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0, 1]},
        "placeholder-b": {"doc_name": "docB", "spec": "placeholder",
                          "unit": "paragraph", "para_idxs": [0]},
    }
    usage = {"input_tokens": 300, "output_tokens": 20,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")], usage)
    # "succeeded" per the API, but content holds only a thinking block: no text
    # to parse. This is the line that used to raise StopIteration.
    textless = {
        "custom_id": "placeholder-b",
        "result": {"type": "succeeded", "message": {
            "usage": {"input_tokens": 50, "output_tokens": 5,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "content": [{"type": "thinking", "thinking": "...", "signature": "x"}],
        }},
    }
    forge_run("textless", index=idx, result_lines=[good, textless],
              submitted_custom_ids=["placeholder-a", "placeholder-b"])

    A.cmd_ingest(args(run_id="textless"))  # must NOT raise (was StopIteration)

    # the good speech's rows landed; the text-less one wrote nothing
    rows = ann.load_paragraph_annotations("placeholder")
    assert sorted(rows["para_idx"]) == [0, 1]
    assert set(rows["doc_name"]) == {"docA"}  # docB contributed no row

    m = ann.read_manifest("textless")
    assert m.n_requests == 2  # both results counted, none dropped
    # recorded as retryable, not silently swallowed
    assert "retryable failures: 1" in m.notes

    # and it is NOT sealed done — the next submit re-requests docB
    paragraphs = pd.DataFrame({"doc_name": ["docA", "docA", "docB"], "para_idx": [0, 1, 0]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert "docB" not in done["placeholder"]
    assert "docA" in done["placeholder"]


def test_succeeded_with_empty_content_list_does_not_abort_ingest(forge_run, args):
    """The other text-less shape: content == [] (no blocks at all). Same guard —
    next() with a default returns None rather than raising."""
    idx = _index(doc="docA", cid="placeholder-a", para_idxs=(0,))
    empty = {
        "custom_id": "placeholder-a",
        "result": {"type": "succeeded", "message": {"usage": {}, "content": []}},
    }
    forge_run("empty_content", index=idx, result_lines=[empty])

    A.cmd_ingest(args(run_id="empty_content"))  # must not raise

    m = ann.read_manifest("empty_content")
    assert "retryable failures: 1" in m.notes
    # nothing was written for the sole (text-less) speech
    assert not ann.annotation_path("placeholder").exists()


def test_ingest_classifies_all_result_types_and_counts_failures_into_n_requests(forge_run, args):
    """succeeded + invalid_request (permanent) + a retryable error + expired:
    every type is counted, n_requests includes the failures, and the
    permanent/retryable split is recorded."""
    idx = {cid: {"doc_name": doc, "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]}
           for cid, doc in [("placeholder-a", "docA"), ("placeholder-perm", "docPerm"),
                            ("placeholder-retry", "docRetry"), ("placeholder-exp", "docExp")]}
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder")],
                                    {"input_tokens": 100, "output_tokens": 10})
    perm = forge_run.errored_line("placeholder-perm", "invalid_request")
    retry = forge_run.errored_line("placeholder-retry", "overloaded_error")
    # expired/canceled arrive with a result.type and no error object at all
    expired = {"custom_id": "placeholder-exp", "result": {"type": "expired"}}
    forge_run("classify", index=idx, result_lines=[good, perm, retry, expired],
              submitted_custom_ids=sorted(idx))

    A.cmd_ingest(args(run_id="classify"))

    m = ann.read_manifest("classify")
    assert m.n_requests == 4  # (d) all four counted, failures included
    assert "succeeded=1 errored=2 canceled=0 expired=1" in m.notes
    # invalid_request is the only permanent class; the retryable error and the
    # expired result (no error object) both count as retryable.
    assert "permanent (invalid_request) failures: 1" in m.notes
    assert "retryable failures: 2" in m.notes
    # only the succeeded speech contributed rows
    assert set(ann.load_paragraph_annotations("placeholder")["doc_name"]) == {"docA"}


def test_failed_speeches_stay_re_requestable_permanent_and_retryable_alike(forge_run, args):
    """A failed result writes NO rows, so its speech stays out of
    _already_ingested and will be re-requested. The code records the
    permanent/retryable split in the manifest for a human but does NOT
    terminally seal a permanent failure — both remain re-requestable."""
    idx = {cid: {"doc_name": doc, "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]}
           for cid, doc in [("placeholder-a", "docA"), ("placeholder-perm", "docPerm"),
                            ("placeholder-retry", "docRetry")]}
    good = forge_run.succeeded_line("placeholder-a", [(0, "placeholder")],
                                    {"input_tokens": 100, "output_tokens": 10})
    perm = forge_run.errored_line("placeholder-perm", "invalid_request")
    retry = forge_run.errored_line("placeholder-retry", "overloaded_error")
    forge_run("reref", index=idx, result_lines=[good, perm, retry],
              submitted_custom_ids=sorted(idx))

    A.cmd_ingest(args(run_id="reref"))

    paragraphs = pd.DataFrame({"doc_name": ["docA", "docPerm", "docRetry"],
                               "para_idx": [0, 0, 0]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert done["placeholder"] == {"docA"}  # both failures are still re-requestable


# ---------------------------------------------------------------------------
# cross-run merge (annotate.py ~800-806): prior rows preserved for keys this
# run does not touch; this run wins on a key collision; key stays unique
# ---------------------------------------------------------------------------


def test_ingest_merges_with_prior_run_preserving_nonoverlapping_keys(forge_run, args):
    # A PRIOR annotation parquet already on disk: docA/0 + docA/1 will collide
    # with this run; docZ/0 is prior-only and must survive the merge.
    ann.write_annotations("placeholder", pd.DataFrame({
        "doc_name": ["docA", "docA", "docZ"],
        "para_idx": [0, 1, 0],
        "label": ["old", "old", "old"],
        "run_id": ["prior", "prior", "prior"],
    }), unit="paragraph")

    idx = _index(doc="docA", cid="placeholder-a", para_idxs=(0, 1))
    line = forge_run.succeeded_line("placeholder-a", [(0, "placeholder"), (1, "placeholder")],
                                    {"input_tokens": 10, "output_tokens": 2})
    forge_run("merge", index=idx, result_lines=[line])

    A.cmd_ingest(args(run_id="merge"))

    # key-asserting loader: also proves no duplicate-key explosion after merge
    rows = ann.load_paragraph_annotations("placeholder").sort_values(["doc_name", "para_idx"])
    assert len(rows) == 3  # docZ/0 (prior) + docA/0 + docA/1 (this run)

    # non-overlapping prior row is preserved verbatim
    z = rows[(rows["doc_name"] == "docZ") & (rows["para_idx"] == 0)].iloc[0]
    assert z["run_id"] == "prior" and z["label"] == "old"

    # overlapping keys take THIS run's values, not the prior's
    a = rows[rows["doc_name"] == "docA"]
    assert list(a["run_id"]) == ["merge", "merge"]
    assert list(a["label"]) == ["placeholder", "placeholder"]


def test_already_ingested_treats_any_present_speech_as_done_for_speech_unit(redirect_annotation_dirs):
    speech_spec = A.FieldSpec(
        name="tone", unit="speech", prompt_version="v", rubric="r", instruction="i",
        json_schema={"type": "object", "properties": {"label": {"type": "string"}}},
    )
    ann.write_annotations("tone", pd.DataFrame({
        "doc_name": ["docA"], "label": ["C"], "run_id": ["r"],
    }), unit="speech")
    paragraphs = pd.DataFrame({"doc_name": ["docA", "docB"], "para_idx": [0, 0]})
    done = A._already_ingested([speech_spec], paragraphs)
    assert done["tone"] == {"docA"}
