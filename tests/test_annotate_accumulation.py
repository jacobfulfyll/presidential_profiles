"""Target #3 — cross-batch manifest accumulation is cumulative and idempotent.

A ``submit --force`` under an existing run_id creates a NEW batch. Ingesting each
batch must ADD to the run's recorded spend, never overwrite it with only the last
batch's numbers — the layer's invariant is that a manifest can never claim a run
was cheaper than it was. And re-ingesting the SAME batch must be idempotent: its
per-batch entry is replaced, not added, so spend never double-counts.

Flow: ingest batch A -> (simulate submit --force -> batch B) -> ingest B ->
re-ingest B. Expected: cumulative cost/tokens/n_requests = A + B, counted once.
"""

from __future__ import annotations

import json

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann

_USAGE_A = {"input_tokens": 1000, "output_tokens": 200,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
_USAGE_B = {"input_tokens": 500, "output_tokens": 50,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _para_line(cid: str, para_idx: int, usage: dict) -> dict:
    text = json.dumps({"annotations": [{"para_idx": para_idx, "label": "placeholder"}]})
    return {"custom_id": cid, "result": {"type": "succeeded",
            "message": {"usage": usage, "content": [{"type": "text", "text": text}]}}}


def test_two_batch_ingest_accumulates_spend_and_reingest_is_idempotent(forge_run, args):
    # The run's index knows both speeches from the start; each batch returns one.
    index = {
        "ph-a": {"doc_name": "docA", "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]},
        "ph-b": {"doc_name": "docB", "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]},
    }

    # --- batch A ---------------------------------------------------------
    forge_run("acc", index=index, result_lines=[_para_line("ph-a", 0, _USAGE_A)],
              batch_id="batchA", results_batch_id="batchA", submitted_custom_ids=["ph-a"])
    A.cmd_ingest(args(run_id="acc"))

    m_a = ann.read_manifest("acc")
    assert m_a.batch_id == "batchA"
    assert m_a.n_requests == 1
    assert m_a.input_tokens == 1000
    assert m_a.cost_usd == round(A.cost_usd(**_USAGE_A), 4)

    # --- simulate `submit --force` creating batch B ----------------------
    # submit merges new batch fields onto state (preserving batch_usage) and
    # resets results_batch_id; here we then place batch B's completed results.
    A._write_state("acc", batch_id="batchB", results_batch_id="batchB",
                   submitted_custom_ids=["ph-b"], n_requests=1)
    (ann.RUNS_DIR / "acc" / "results.jsonl").write_text(
        json.dumps(_para_line("ph-b", 0, _USAGE_B)) + "\n"
    )

    # --- ingest batch B --------------------------------------------------
    A.cmd_ingest(args(run_id="acc"))

    m_ab = ann.read_manifest("acc")
    # cumulative across BOTH batches, counted once each
    assert m_ab.n_requests == 2
    assert m_ab.input_tokens == 1500 and m_ab.output_tokens == 250
    assert m_ab.cost_usd == round(A.cost_usd(input_tokens=1500, output_tokens=250), 4)
    # the manifest names both batches, not just the last
    assert m_ab.batch_id == "batchA,batchB"
    # spend can only have gone UP relative to batch A alone
    assert m_ab.cost_usd > m_a.cost_usd

    state = json.loads((ann.RUNS_DIR / "acc" / "state.json").read_text())
    assert sorted(state["batch_usage"]) == ["batchA", "batchB"]

    # --- re-ingest batch B (idempotent) ----------------------------------
    A.cmd_ingest(args(run_id="acc"))

    m_re = ann.read_manifest("acc")
    assert m_re.n_requests == 2                       # not 3
    assert m_re.input_tokens == 1500                  # not 2000
    assert m_re.cost_usd == m_ab.cost_usd             # unchanged: no double count
    state_re = json.loads((ann.RUNS_DIR / "acc" / "state.json").read_text())
    assert sorted(state_re["batch_usage"]) == ["batchA", "batchB"]  # keys stable
