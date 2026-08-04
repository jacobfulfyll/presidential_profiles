"""Target #1 — the first real speech-unit spec, end to end.

Every prior ingest test drives a paragraph-unit spec (placeholder). ``speech_
annotations`` (the unmasked factual pass) is the first ``unit="speech"`` spec in
the registry, so this locks the whole speech path: build_requests -> forged
batch results -> cmd_ingest -> a parquet keyed by doc_name (no para_idx, no
annotations wrapper) + a provenance manifest with the actual cost.
"""

from __future__ import annotations

import json

import pandas as pd

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann


def _speech_line(cid: str, payload: dict, usage: dict) -> dict:
    """A succeeded speech-unit result line: one object for the whole speech, no
    ``annotations`` array (that shape is paragraph-unit only)."""
    return {
        "custom_id": cid,
        "result": {
            "type": "succeeded",
            "message": {"usage": usage, "content": [{"type": "text", "text": json.dumps(payload)}]},
        },
    }


def _synthetic_corpus():
    speeches = pd.DataFrame(
        {
            "doc_name": ["docA", "docB"],
            "title": ["First Inaugural Address", "Fourth Annual Message"],
            "president": ["Alpha", "Beta"],
            "year": [1853, 1888],
            "decade": [1850, 1880],
            "date": [pd.Timestamp("1853-03-04"), pd.Timestamp("1888-12-03")],
            "transcript": ["inaugural body", "annual message body"],
        }
    )
    paras = pd.DataFrame(
        {
            "doc_name": ["docA", "docB"],
            "para_idx": [0, 0],
            "text": ["Fellow citizens, I am called to the office.", "To the Congress of the United States:"],
            "word_count": [8, 6],
        }
    )
    return speeches, paras


def test_speech_spec_ingests_to_a_doc_keyed_parquet_and_manifest(forge_run, args):
    speeches, paras = _synthetic_corpus()

    # Build the REAL speech spec's requests, then forge each speech's answer.
    requests, index = A.build_requests(A.FIELD_SPECS["speech_annotations"], speeches, paras)
    assert len(requests) == 2
    # every request is speech-unit (records no para_idxs)
    assert all(meta["unit"] == "speech" and meta["para_idxs"] == [] for meta in index.values())

    cid_by_doc = {meta["doc_name"]: cid for cid, meta in index.items()}
    usage = {"input_tokens": 400, "output_tokens": 30,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    lines = [
        _speech_line(cid_by_doc["docA"],
                     {"speech_type": "inaugural_address", "audience": "general_public",
                      "medium": "spoken_address"}, usage),
        _speech_line(cid_by_doc["docB"],
                     {"speech_type": "state_of_the_union_or_annual_message", "audience": "congress",
                      "medium": "written_message"}, usage),
    ]
    forge_run("sp1", index=index, result_lines=lines, specs=["speech_annotations"],
              submitted_custom_ids=sorted(index))

    A.cmd_ingest(args(run_id="sp1"))

    # (a) parquet is keyed by doc_name, flat, with the three factual fields and
    #     NO annotations/para_idx column
    rows = ann.load_speech_annotations("speech_annotations").set_index("doc_name")
    assert set(rows.index) == {"docA", "docB"}
    assert "para_idx" not in rows.columns and "annotations" not in rows.columns
    assert rows.loc["docA", "speech_type"] == "inaugural_address"
    assert rows.loc["docA", "audience"] == "general_public"
    assert rows.loc["docB", "medium"] == "written_message"
    assert set(rows["run_id"]) == {"sp1"}

    # (b) manifest records both requests and the ACTUAL cost from cumulative usage
    m = ann.read_manifest("sp1")
    assert m.n_requests == 2
    assert m.input_tokens == 800 and m.output_tokens == 60  # summed across 2 speeches
    assert m.cost_usd == round(A.cost_usd(input_tokens=800, output_tokens=60), 4)
    assert "speech_annotations=speech-factual/v1" in m.prompt_version


def test_speech_spec_marks_present_speeches_done_for_resume(forge_run, args):
    """A speech-unit spec is 'done' for any speech that has a row — there is no
    per-paragraph coverage to complete — so a resume submit will not re-pay for
    an already-ingested speech."""
    speeches, paras = _synthetic_corpus()
    requests, index = A.build_requests(A.FIELD_SPECS["speech_annotations"], speeches, paras)
    cid_by_doc = {meta["doc_name"]: cid for cid, meta in index.items()}
    usage = {"input_tokens": 100, "output_tokens": 10,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    # only docA comes back this round
    forge_run("sp2", index=index,
              result_lines=[_speech_line(cid_by_doc["docA"],
                            {"speech_type": "inaugural_address", "audience": "general_public",
                             "medium": "spoken_address"}, usage)],
              specs=["speech_annotations"], submitted_custom_ids=[cid_by_doc["docA"]])

    A.cmd_ingest(args(run_id="sp2"))

    done = A._already_ingested([A.FIELD_SPECS["speech_annotations"]], paras)
    assert done["speech_annotations"] == {"docA"}  # docB never returned -> re-requestable
