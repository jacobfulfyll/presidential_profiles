"""Target #7 — permanent (invalid_request) failures are terminally sealed.

A permanent 400 fails identically on every retry, so re-requesting it burns money
to fail again. Ingest records such doc_names in ``state.sealed_permanent``; the
next ``submit`` drops them from the resume set unless ``--resubmit-sealed`` is
passed after a human fix. Retryable failures (overloaded, expired) are NOT sealed
— they write no rows and resume normally.
"""

from __future__ import annotations

import json
import types

import pandas as pd

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann


def _install_synthetic_corpus(monkeypatch, tmp_path, doc_names=("docA", "docB")):
    """Point load() and PARAGRAPHS_PATH at a tiny synthetic corpus so cmd_submit
    runs fully offline over a controlled set of speeches."""
    speeches = pd.DataFrame(
        {"doc_name": list(doc_names),
         "date": [pd.Timestamp("1850-01-01") + pd.Timedelta(days=i) for i in range(len(doc_names))]}
    )
    monkeypatch.setattr(A, "load", lambda: speeches)
    paras = pd.DataFrame(
        {"doc_name": list(doc_names), "para_idx": [0] * len(doc_names),
         "text": [f"body of {d}" for d in doc_names], "word_count": [3] * len(doc_names)}
    )
    ppath = tmp_path / "paragraphs.parquet"
    paras.to_parquet(ppath, index=False)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", ppath)
    return speeches


# ---------------------------------------------------------------------------
# ingest side: invalid_request seals, retryable does not
# ---------------------------------------------------------------------------


def test_ingest_seals_only_invalid_request_docs_into_state(forge_run, args):
    idx = {cid: {"doc_name": doc, "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]}
           for cid, doc in [("ph-good", "docGood"), ("ph-perm", "docPerm"), ("ph-retry", "docRetry")]}
    good = forge_run.succeeded_line("ph-good", [(0, "placeholder")], {"input_tokens": 10, "output_tokens": 2})
    perm = forge_run.errored_line("ph-perm", "invalid_request")   # permanent -> sealed
    retry = forge_run.errored_line("ph-retry", "overloaded_error")  # retryable -> NOT sealed
    forge_run("seal", index=idx, result_lines=[good, perm, retry], submitted_custom_ids=sorted(idx))

    A.cmd_ingest(args(run_id="seal"))

    state = json.loads((ann.RUNS_DIR / "seal" / "state.json").read_text())
    assert state["sealed_permanent"] == ["docPerm"]  # only the invalid_request doc
    assert "docRetry" not in state["sealed_permanent"]
    assert "docGood" not in state["sealed_permanent"]


def _errored_with_message(cid: str, error_type: str, message: str) -> dict:
    """An errored result carrying a message (forge_run.errored_line omits it)."""
    return {"custom_id": cid, "result": {"type": "errored",
                                         "error": {"type": error_type, "message": message}}}


_CREDIT_MSG = ("Your credit balance is too low to access the Anthropic API. "
               "Please go to Plans & Billing to upgrade or purchase credits.")


def test_billing_block_is_retryable_not_sealed(forge_run, args):
    """The credit/billing block arrives as `invalid_request_error` — the SAME
    type as a real validation failure — but is transient: it must stay retryable,
    never sealed, so it re-requests once the account is funded. (124 such results
    in the full run.)"""
    idx = {"ph-bill": {"doc_name": "docBill", "spec": "placeholder",
                       "unit": "paragraph", "para_idxs": [0]}}
    billing = _errored_with_message("ph-bill", "invalid_request_error", _CREDIT_MSG)
    forge_run("billing", index=idx, result_lines=[billing], submitted_custom_ids=["ph-bill"])

    A.cmd_ingest(args(run_id="billing"))

    state = json.loads((ann.RUNS_DIR / "billing" / "state.json").read_text())
    assert state["sealed_permanent"] == []  # NOT sealed — transient billing block
    m = ann.read_manifest("billing")
    assert "permanent (invalid_request) failures: 0" in m.notes   # not counted permanent
    assert "billing-blocked: 1" in m.notes                        # surfaced as its own subclass
    # no rows written -> not done -> re-requestable
    paragraphs = pd.DataFrame({"doc_name": ["docBill"], "para_idx": [0]})
    done = A._already_ingested([A.FIELD_SPECS["placeholder"]], paragraphs)
    assert "docBill" not in done["placeholder"]


def test_invalid_request_error_validation_message_seals_skips_and_reincludes(
    args, patch_anthropic, monkeypatch, tmp_path, forge_run
):
    """A GENUINE malformed request also arrives as `invalid_request_error`, but
    with a validation message (not billing). It must SEAL, so resume skips it and
    only --resubmit-sealed re-includes it."""
    # ingest: docA fails with a NON-billing invalid_request_error -> sealed
    idx = {"ph-a": {"doc_name": "docA", "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]}}
    bad = _errored_with_message("ph-a", "invalid_request_error", "max_tokens: field required")
    forge_run("valfail", index=idx, result_lines=[bad], submitted_custom_ids=["ph-a"])
    A.cmd_ingest(args(run_id="valfail"))
    sealed = json.loads((ann.RUNS_DIR / "valfail" / "state.json").read_text())["sealed_permanent"]
    assert sealed == ["docA"]  # non-billing invalid_request_error -> sealed permanent

    # resume submit carrying that seal skips docA (sends only docB)
    _install_synthetic_corpus(monkeypatch, tmp_path)  # docA, docB
    run2 = ann.RUNS_DIR / "valfail_submit"
    run2.mkdir(parents=True, exist_ok=True)
    (run2 / "state.json").write_text(json.dumps({"sealed_permanent": sealed}))
    patch_anthropic(create_batch=types.SimpleNamespace(id="b_new", processing_status="in_progress"))
    A.cmd_submit(args(run_id="valfail_submit", spec=["placeholder"], yes=True))
    submitted = json.loads((run2 / "state.json").read_text())["submitted_custom_ids"]
    assert A._custom_id("placeholder", "docA") not in submitted  # sealed -> skipped
    assert A._custom_id("placeholder", "docB") in submitted

    # --resubmit-sealed re-includes docA
    run3 = ann.RUNS_DIR / "valfail_reseal"
    run3.mkdir(parents=True, exist_ok=True)
    (run3 / "state.json").write_text(json.dumps({"sealed_permanent": sealed}))
    patch_anthropic(create_batch=types.SimpleNamespace(id="b2", processing_status="in_progress"))
    A.cmd_submit(args(run_id="valfail_reseal", spec=["placeholder"], yes=True, resubmit_sealed=True))
    submitted3 = json.loads((run3 / "state.json").read_text())["submitted_custom_ids"]
    assert A._custom_id("placeholder", "docA") in submitted3  # re-included after the flag


# ---------------------------------------------------------------------------
# submit side: sealed docs are skipped; --resubmit-sealed re-includes them
# ---------------------------------------------------------------------------


def test_submit_skips_sealed_docs_by_default(args, patch_anthropic, monkeypatch, tmp_path, capsys):
    _install_synthetic_corpus(monkeypatch, tmp_path)
    run_dir = ann.RUNS_DIR / "sub_seal"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps({"sealed_permanent": ["docA"]}))
    patch_anthropic(create_batch=types.SimpleNamespace(id="batch_new", processing_status="in_progress"))

    A.cmd_submit(args(run_id="sub_seal", spec=["placeholder"], yes=True))

    assert "sealed: skipping" in capsys.readouterr().out
    submitted = json.loads((run_dir / "state.json").read_text())["submitted_custom_ids"]
    assert A._custom_id("placeholder", "docA") not in submitted   # sealed docA dropped
    assert A._custom_id("placeholder", "docB") in submitted       # docB still sent


def test_resubmit_sealed_flag_reincludes_the_sealed_doc(args, patch_anthropic, monkeypatch, tmp_path):
    _install_synthetic_corpus(monkeypatch, tmp_path)
    run_dir = ann.RUNS_DIR / "sub_reseal"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps({"sealed_permanent": ["docA"]}))
    patch_anthropic(create_batch=types.SimpleNamespace(id="batch_new", processing_status="in_progress"))

    A.cmd_submit(args(run_id="sub_reseal", spec=["placeholder"], yes=True, resubmit_sealed=True))

    submitted = json.loads((run_dir / "state.json").read_text())["submitted_custom_ids"]
    assert A._custom_id("placeholder", "docA") in submitted  # re-included after the flag
    assert A._custom_id("placeholder", "docB") in submitted


def test_retryable_failure_is_not_sealed_and_resubmits_normally(args, patch_anthropic, monkeypatch,
                                                                 tmp_path, forge_run):
    """A retryable failure writes no rows and is not sealed, so a later submit
    re-requests it with no special flag."""
    # ingest a run where docA fails retryably (no seal), docB succeeds
    idx = {cid: {"doc_name": doc, "spec": "placeholder", "unit": "paragraph", "para_idxs": [0]}
           for cid, doc in [("ph-a", "docA"), ("ph-b", "docB")]}
    retry = forge_run.errored_line("ph-a", "overloaded_error")
    good = forge_run.succeeded_line("ph-b", [(0, "placeholder")], {"input_tokens": 10, "output_tokens": 2})
    forge_run("retry_resume", index=idx, result_lines=[retry, good], submitted_custom_ids=sorted(idx))
    A.cmd_ingest(args(run_id="retry_resume"))

    state = json.loads((ann.RUNS_DIR / "retry_resume" / "state.json").read_text())
    assert state["sealed_permanent"] == []  # nothing sealed

    # a fresh submit over the same corpus: docA (retryable, no rows) is re-sent;
    # docB (already ingested) is skipped by the coverage resume, not by sealing.
    _install_synthetic_corpus(monkeypatch, tmp_path)
    run_dir = ann.RUNS_DIR / "retry_submit"
    run_dir.mkdir(parents=True, exist_ok=True)
    patch_anthropic(create_batch=types.SimpleNamespace(id="batch_new", processing_status="in_progress"))
    A.cmd_submit(args(run_id="retry_submit", spec=["placeholder"], yes=True))

    submitted = json.loads((run_dir / "state.json").read_text())["submitted_custom_ids"]
    assert A._custom_id("placeholder", "docA") in submitted  # retryable -> re-requested
    assert A._custom_id("placeholder", "docB") not in submitted  # already fully annotated
