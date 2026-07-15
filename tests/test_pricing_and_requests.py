"""Pure-function seams and request construction:

* cost_usd (batch -50%, intro vs standard pricing)
* _validate_spec (rejects an unsupported unit — the F5 fix — and schema shapes)
* _check_batch_limits (size caps)
* _custom_id collision-freedom / charset / determinism + index round-trip (#6)
* the sonnet-5 thinking cost trap on the dry-run JSONL (#9)
"""

from __future__ import annotations

import datetime as _dt
import json
import re

import pandas as pd
import pytest

from presidential_profiles import annotate as A
from presidential_profiles.corpus import load


# ---------------------------------------------------------------------------
# cost_usd
# ---------------------------------------------------------------------------


def _force_intro(monkeypatch, on: bool):
    """Pin date.today() either inside or after the intro-pricing window."""
    day = _dt.date(2026, 7, 1) if on else _dt.date(2026, 9, 1)

    class _FakeDate(_dt.date):
        @classmethod
        def today(cls):
            return day

    monkeypatch.setattr(A, "date", _FakeDate)


def test_cost_applies_intro_rate_and_batch_discount(monkeypatch):
    _force_intro(monkeypatch, on=True)
    # 1M input tokens at $2.00, batch -50% -> $1.00; no output.
    assert A.cost_usd(1_000_000, 0) == pytest.approx(1.00)
    # 1M output at $10.00, batch -50% -> $5.00.
    assert A.cost_usd(0, 1_000_000) == pytest.approx(5.00)


def test_cost_uses_list_pricing_after_intro_window(monkeypatch):
    _force_intro(monkeypatch, on=False)
    # After 2026-08-31: $3.00 input, batch -50% -> $1.50.
    assert A.cost_usd(1_000_000, 0) == pytest.approx(1.50)
    assert A.cost_usd(0, 1_000_000) == pytest.approx(7.50)


def test_cost_bills_cache_reads_cheaply_and_writes_at_a_premium(monkeypatch):
    _force_intro(monkeypatch, on=True)
    # cache read at 0.10x input: 1M read -> 100k billable -> $0.20 * 0.5 = $0.10
    assert A.cost_usd(0, 0, cache_read_input_tokens=1_000_000) == pytest.approx(0.10)
    # cache write at 1.25x input: 1M write -> 1.25M billable -> $2.50 * 0.5 = $1.25
    assert A.cost_usd(0, 0, cache_creation_input_tokens=1_000_000) == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# _validate_spec
# ---------------------------------------------------------------------------


def _spec(**over):
    base = dict(
        name="t", unit="paragraph", prompt_version="v", rubric="r", instruction="i",
        json_schema={
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "items": {"type": "object",
                              "properties": {"para_idx": {"type": "integer"},
                                             "label": {"type": "string"}}},
                }
            },
        },
    )
    base.update(over)
    return A.FieldSpec(**base)


def test_validate_spec_rejects_unsupported_unit():
    """The F5 fix: a unit that is neither speech nor paragraph is refused."""
    with pytest.raises(ValueError, match="unit"):
        A._validate_spec(_spec(unit="sentence"))


def test_validate_spec_rejects_paragraph_schema_without_para_idx():
    bad = _spec(json_schema={"type": "object", "properties": {
        "annotations": {"type": "array", "items": {"type": "object", "properties": {}}}}})
    with pytest.raises(ValueError, match="para_idx"):
        A._validate_spec(bad)


def test_validate_spec_rejects_reserved_column_clash():
    bad = _spec(json_schema={"type": "object", "properties": {
        "annotations": {"type": "array", "items": {"type": "object", "properties": {
            "para_idx": {"type": "integer"}, "run_id": {"type": "string"}}}}}})
    with pytest.raises(ValueError, match="collide"):
        A._validate_spec(bad)


def test_validate_spec_rejects_speech_unit_wrapped_in_annotations():
    bad = _spec(unit="speech", json_schema={"type": "object", "properties": {
        "annotations": {"type": "array"}}})
    with pytest.raises(ValueError, match="annotations"):
        A._validate_spec(bad)


def test_shipped_placeholder_spec_validates():
    A._validate_spec(A.FIELD_SPECS["placeholder"])  # must not raise


# ---------------------------------------------------------------------------
# _estimate_tokens — per-spec rubric attribution (SHOULD-FIX)
# ---------------------------------------------------------------------------


def test_estimate_attributes_rubric_per_request_by_its_spec_not_cross_product():
    """Each request carries only ITS spec's rubric. The estimate must add each
    request's own rubric length, NOT the sum of every spec's rubric times the
    request count — the latter inflates ~Nx for N specs. Built with two specs of
    deliberately different rubric lengths so the two formulas disagree."""
    specs = [_spec(name="ra", rubric="a" * 40), _spec(name="rb", rubric="b" * 80)]
    speeches = load().head(3)
    paragraphs = pd.read_parquet(A.ann.PARAGRAPHS_PATH)
    paragraphs = paragraphs[paragraphs["doc_name"].isin(set(speeches["doc_name"]))]

    requests, index = [], {}
    for spec in specs:
        reqs, idx = A.build_requests(spec, speeches, paragraphs)
        requests += reqs
        index |= idx
    assert len(requests) == 2 * len(speeches)  # one request per (spec, speech)

    est_in, _ = A._estimate_tokens(requests, index, specs)

    by_name = {s.name: s for s in specs}
    messages_part = sum(len(json.dumps(r["params"]["messages"])) for r in requests) // 4
    attributed = sum(len(by_name[index[r["custom_id"]]["spec"]].rubric) // 4 for r in requests)
    cross_product = sum(len(s.rubric) for s in specs) // 4 * len(requests)  # the OLD bug

    assert est_in == messages_part + attributed
    assert est_in != messages_part + cross_product  # fails against the old formula
    assert attributed < cross_product  # the fix strictly de-inflates the estimate


# ---------------------------------------------------------------------------
# _check_batch_limits (unit)
# ---------------------------------------------------------------------------


def test_check_batch_limits_flags_too_many_requests():
    with pytest.raises(ValueError, match="requests exceeds"):
        A._check_batch_limits(A.BATCH_MAX_REQUESTS + 1, 10)


def test_check_batch_limits_flags_oversize_payload():
    with pytest.raises(ValueError, match="bytes exceeds"):
        A._check_batch_limits(1, A.BATCH_MAX_BYTES + 1)


# ---------------------------------------------------------------------------
# _custom_id (regression #6)
# ---------------------------------------------------------------------------


def test_custom_id_is_deterministic():
    doc = "/the-presidency/presidential-speeches/april-30-1789-first-inaugural-address"
    assert A._custom_id("placeholder", doc) == A._custom_id("placeholder", doc)


def test_custom_id_is_collision_free_and_legal_across_all_real_doc_names():
    """All 1,057 real doc_names must map to distinct custom_ids within the
    Batches API charset/length limits — a slash-laden 100-char URL can't be a
    custom_id literally, so it is hashed."""
    doc_names = list(load()["doc_name"])
    assert len(doc_names) == 1057
    cids = [A._custom_id("placeholder", d) for d in doc_names]
    assert len(set(cids)) == len(cids)  # no collisions
    legal = re.compile(r"[A-Za-z0-9_-]{1,64}")
    assert all(legal.fullmatch(c) for c in cids)


def test_requests_index_round_trips_custom_id_to_doc_name():
    speeches = load().head(4)
    paragraphs = pd.read_parquet(A.ann.PARAGRAPHS_PATH)
    paragraphs = paragraphs[paragraphs["doc_name"].isin(set(speeches["doc_name"]))]
    _, index = A.build_requests(A.FIELD_SPECS["placeholder"], speeches, paragraphs)
    for cid, meta in index.items():
        assert A._custom_id("placeholder", meta["doc_name"]) == cid


# ---------------------------------------------------------------------------
# sonnet-5 thinking cost trap (regression #9) — asserted on the dry-run JSONL
# ---------------------------------------------------------------------------


def test_dry_run_jsonl_disables_thinking_and_omits_sampling_params(args, redirect_annotation_dirs):
    """Every generated request must carry thinking={"type":"disabled"} and set
    NONE of temperature/top_p/top_k/budget_tokens/fallbacks — omitting `thinking`
    on claude-sonnet-5 silently runs adaptive thinking and bills for it."""
    A.cmd_dry_run(args(run_id="trap", limit=5, spec=["placeholder"]))
    jsonl = A.ann.RUNS_DIR / "trap" / "requests.jsonl"
    lines = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    assert lines  # dry-run actually produced requests
    for req in lines:
        params = req["params"]
        assert params["thinking"] == {"type": "disabled"}
        assert "budget_tokens" not in params.get("thinking", {})
        for forbidden in ("temperature", "top_p", "top_k", "fallbacks"):
            assert forbidden not in params


def test_validate_request_rejects_missing_thinking_on_sonnet5():
    """The validation seam behind the dry-run assertion: a sonnet-5 request that
    forgets to disable thinking is refused before it can be paid for."""
    speeches = load().head(1)
    paragraphs = pd.read_parquet(A.ann.PARAGRAPHS_PATH)
    paragraphs = paragraphs[paragraphs["doc_name"].isin(set(speeches["doc_name"]))]
    reqs, _ = A.build_requests(A.FIELD_SPECS["placeholder"], speeches, paragraphs)
    req = reqs[0]
    del req["params"]["thinking"]
    with pytest.raises(ValueError, match="thinking must be EXPLICITLY disabled"):
        A._validate_request(req)


def test_validate_request_rejects_forbidden_sampling_param():
    speeches = load().head(1)
    paragraphs = pd.read_parquet(A.ann.PARAGRAPHS_PATH)
    paragraphs = paragraphs[paragraphs["doc_name"].isin(set(speeches["doc_name"]))]
    reqs, _ = A.build_requests(A.FIELD_SPECS["placeholder"], speeches, paragraphs)
    req = reqs[0]
    req["params"]["temperature"] = 0.7
    with pytest.raises(ValueError, match="temperature"):
        A._validate_request(req)


def test_dry_run_makes_no_network_call(args, redirect_annotation_dirs, anthropic_construction_bomb):
    """dry-run must construct no client at all — it works with creds unset."""
    A.cmd_dry_run(args(run_id="offline", limit=3, spec=["placeholder"]))
    # anthropic_construction_bomb would have raised if a client were built.
