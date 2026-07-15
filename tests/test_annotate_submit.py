"""Money gates and status (regression #8).

Every gate must fire BEFORE a client is constructed — proven by installing an
``anthropic_construction_bomb`` that raises if ``anthropic.Anthropic()`` is ever
called. With creds unset, a real construction would also fail; the bomb makes the
"no client was built" assertion explicit and independent of SDK behaviour.
"""

from __future__ import annotations

import json
import types

import pytest

from presidential_profiles import annotate as A


def test_submit_without_yes_refuses_and_builds_no_client(args, anthropic_construction_bomb):
    with pytest.raises(SystemExit, match="without --yes"):
        A.cmd_submit(args(run_id="s1", limit=2, spec=["placeholder"], yes=False))


def test_submit_refuses_when_estimate_exceeds_ceiling(args, anthropic_construction_bomb):
    with pytest.raises(SystemExit, match="exceeds --max-cost-usd"):
        A.cmd_submit(args(run_id="s2", limit=2, spec=["placeholder"], yes=True, max_cost_usd=0.0))


def test_submit_ceiling_reacts_to_a_realistically_magnituded_estimate(args, anthropic_construction_bomb):
    """The max_cost_usd=0.0 ceiling test above trips on ANY positive estimate,
    so a silent 10x under-count in _estimate_tokens would slip a real ceiling.
    Pin the estimate MAGNITUDE for a small known batch and set the ceiling just
    under it: 3 real speeches estimate ~$0.006, bracketed ~3x either side (wide
    enough to survive heuristic tweaks, tight enough to catch a ~10x regression).
    """
    reqs, index, _, _ = A._prepare([A.FIELD_SPECS["placeholder"]], 3)
    est = A.cost_usd(*A._estimate_tokens(reqs, index, [A.FIELD_SPECS["placeholder"]]))
    assert 0.002 < est < 0.02  # order-of-magnitude band around the real ~$0.006

    # A ceiling inside that band but under the estimate must refuse — and refuse
    # BEFORE constructing a client (the bomb would fire otherwise). A 10x
    # under-count would drop the estimate below 0.004 and silently let it pass.
    with pytest.raises(SystemExit, match="exceeds --max-cost-usd"):
        A.cmd_submit(args(run_id="mag", limit=3, spec=["placeholder"], yes=True, max_cost_usd=0.004))


def test_submit_enforces_request_cap_before_client(args, anthropic_construction_bomb, monkeypatch):
    """The size caps are enforced on the SUBMIT path, not only in dry-run — a
    check that runs only in dry-run protects nothing, since dry-run is free."""
    monkeypatch.setattr(A, "BATCH_MAX_REQUESTS", 0)
    with pytest.raises(ValueError, match="requests exceeds"):
        A.cmd_submit(args(run_id="s3", limit=1, spec=["placeholder"], yes=True))


def test_submit_enforces_byte_cap_before_client(args, anthropic_construction_bomb, monkeypatch):
    monkeypatch.setattr(A, "BATCH_MAX_BYTES", 1)
    with pytest.raises(ValueError, match="bytes exceeds"):
        A.cmd_submit(args(run_id="s4", limit=1, spec=["placeholder"], yes=True))


def test_submit_refuses_resubmit_over_existing_batch_without_force(args, anthropic_construction_bomb):
    """A run that already owns a batch_id must not be resubmitted (and repaid)
    without an explicit --force."""
    run_dir = A.ann.RUNS_DIR / "s5"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps({"batch_id": "batch_existing"}))
    with pytest.raises(SystemExit, match="already has batch_id"):
        A.cmd_submit(args(run_id="s5", limit=2, spec=["placeholder"], yes=True))


def test_submit_creates_batch_and_records_provenance_state(args, patch_anthropic):
    """Happy path with a forged client: the batch id, submitted custom_ids and
    corpus fingerprint are all recorded so a later ingest can be validated."""
    fake_batch = types.SimpleNamespace(id="batch_new", processing_status="in_progress")
    patch_anthropic(create_batch=fake_batch)
    A.cmd_submit(args(run_id="s6", limit=2, spec=["placeholder"], yes=True))
    state = json.loads((A.ann.RUNS_DIR / "s6" / "state.json").read_text())
    assert state["batch_id"] == "batch_new"
    assert state["results_batch_id"] is None  # a fresh batch has no cached results yet
    assert state["submitted_custom_ids"]
    assert state["corpus_fingerprint"]["n_speeches"] == 1057


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_without_batch_id_refuses(args):
    run_dir = A.ann.RUNS_DIR / "st0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit, match="no batch_id"):
        A.cmd_status(args(run_id="st0"))


def test_status_reports_counts_from_forged_client(args, patch_anthropic, capsys):
    run_dir = A.ann.RUNS_DIR / "st1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps({"batch_id": "batch_poll"}))
    counts = types.SimpleNamespace(processing=1, succeeded=2, errored=0, canceled=0, expired=0)
    batch = types.SimpleNamespace(processing_status="ended", request_counts=counts)
    patch_anthropic(retrieve_batch=batch)
    A.cmd_status(args(run_id="st1"))
    out = capsys.readouterr().out
    assert "batch_poll" in out
    assert "ended" in out
    assert "ready to ingest" in out
