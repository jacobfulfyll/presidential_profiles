"""Target #6 — nested entities explode to a long-format table, keyed and
supersession-aware, while the paragraph table stays flat.

The judgment pass returns entities nested per paragraph. Ingest pops that array
into ``paragraph_entities.parquet`` (``doc_name, para_idx, entity, type, stance,
run_id`` — deliberately MANY rows per paragraph key), and the paragraph table
keeps no ``entities`` column. Re-annotating a paragraph SUPERSEDES its prior
entities, including the case where the re-annotation now finds zero — the old
rows must be cleared, not orphaned.
"""

from __future__ import annotations

import json

import pandas as pd

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann

_USAGE = {"input_tokens": 100, "output_tokens": 10,
          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _judgment_item(para_idx: int, entities: list[dict], topics=None) -> dict:
    return {
        "para_idx": para_idx,
        "topics": topics or [],
        "party_attack": False,
        "enemy_naming": bool(entities),
        "zero_sum": False,
        "proposal_values": "neither",
        "entities": entities,
    }


def _judgment_line(cid: str, items: list[dict]) -> dict:
    text = json.dumps({"annotations": items})
    return {"custom_id": cid, "result": {"type": "succeeded",
            "message": {"usage": _USAGE, "content": [{"type": "text", "text": text}]}}}


def _index(cid="pa", doc="docA", para_idxs=(0, 1)):
    return {cid: {"doc_name": doc, "spec": "paragraph_annotations",
                  "unit": "paragraph", "para_idxs": list(para_idxs)}}


def test_entities_explode_to_long_table_and_paragraph_table_stays_flat(forge_run, args):
    items = [
        _judgment_item(0, [
            {"name": "Senator Foo", "type": "person", "stance": "adversarial"},
            {"name": "the trusts", "type": "group", "stance": "adversarial"},
        ], topics=["Trusts, Corporations & Antitrust"]),
        _judgment_item(1, [{"name": "Congress", "type": "institution", "stance": "neutral"}]),
    ]
    forge_run("ent1", index=_index(para_idxs=(0, 1)),
              result_lines=[_judgment_line("pa", items)], specs=["paragraph_annotations"])

    A.cmd_ingest(args(run_id="ent1"))

    # (a) the entity table is long-format, keyed by (doc_name, para_idx), many
    #     rows per paragraph, with exactly the acceptance-criterion columns.
    ent = pd.read_parquet(ann.annotation_path("paragraph_entities"))
    assert list(ent.columns) == ["doc_name", "para_idx", "entity", "type", "stance", "run_id"]
    p0 = ent[ent["para_idx"] == 0]
    assert sorted(p0["entity"]) == ["Senator Foo", "the trusts"]  # two entities, one paragraph
    assert set(p0["stance"]) == {"adversarial"}
    assert set(ent["run_id"]) == {"ent1"}
    p1 = ent[ent["para_idx"] == 1]
    assert list(p1["entity"]) == ["Congress"] and list(p1["type"]) == ["institution"]

    # (b) the paragraph table is FLAT — the nested field was popped, not stored.
    pj = ann.load_paragraph_annotations("paragraph_annotations")
    assert "entities" not in pj.columns
    assert {"party_attack", "enemy_naming", "zero_sum", "proposal_values", "topics"} <= set(pj.columns)
    assert sorted(pj["para_idx"]) == [0, 1]


def test_reannotation_clears_prior_entities_including_now_zero_entity_paragraph(forge_run, args):
    # run 1: docA para0 (2 entities) + para1 (1 entity)
    run1_items = [
        _judgment_item(0, [
            {"name": "Senator Foo", "type": "person", "stance": "adversarial"},
            {"name": "the trusts", "type": "group", "stance": "adversarial"},
        ]),
        _judgment_item(1, [{"name": "Congress", "type": "institution", "stance": "neutral"}]),
    ]
    forge_run("ent_run1", index=_index(cid="pa1", para_idxs=(0, 1)),
              result_lines=[_judgment_line("pa1", run1_items)], specs=["paragraph_annotations"])
    A.cmd_ingest(args(run_id="ent_run1"))

    # run 2: re-annotate ONLY docA para0, now with ZERO entities.
    forge_run("ent_run2", index=_index(cid="pa2", para_idxs=(0,)),
              result_lines=[_judgment_line("pa2", [_judgment_item(0, [])])],
              specs=["paragraph_annotations"], batch_id="batch2", results_batch_id="batch2")
    A.cmd_ingest(args(run_id="ent_run2"))

    ent = pd.read_parquet(ann.annotation_path("paragraph_entities"))
    # para0's prior rows are cleared even though the re-annotation added none...
    assert ent[ent["para_idx"] == 0].empty
    # ...and para1 (untouched by run 2) keeps its original entity/run_id.
    p1 = ent[ent["para_idx"] == 1]
    assert list(p1["entity"]) == ["Congress"]
    assert list(p1["run_id"]) == ["ent_run1"]
