"""Target #9 — cmd_qa pilot gating fires pass/fail correctly.

The pilot auto-check is the gate that decides whether the full paid run proceeds.
Its thresholds (schema failure < 1%, paragraph coverage >= 99%, no degenerate
flag distribution > 60% or ~0%, entity names actually appear in their paragraph)
must each be able to FAIL a genuinely bad pilot and PASS a healthy one — and full
(non-pilot) mode must only REPORT, never raise.

Everything is synthetic and offline: load() and PARAGRAPHS_PATH are repointed at a
tiny controlled corpus, and the annotation tables are hand-written so each metric
is exactly known.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from presidential_profiles import annotate as A
from presidential_profiles import llm_annotations as ann

_DOC = "docQ"
_DECADE = 1850
_N_CORPUS_PARAS = 10


def _setup_qa(
    monkeypatch,
    tmp_path,
    *,
    cov_paras: int = _N_CORPUS_PARAS,
    party_count: int = 3,
    entity_good: bool = True,
    results: list[str] | None = None,
    with_annotations: bool = True,
):
    """Lay down a controlled corpus + annotation tables and return the run_id.

    Knobs let each test flip exactly one metric out of range while the rest stay
    healthy, so a FAIL is attributable to the gate under test.
    """
    # corpus paragraphs (read via PARAGRAPHS_PATH); each names a distinct entity
    corpus = pd.DataFrame({
        "doc_name": [_DOC] * _N_CORPUS_PARAS,
        "para_idx": list(range(_N_CORPUS_PARAS)),
        "text": [f"Paragraph {i} names Senator Foo{i} before the Congress." for i in range(_N_CORPUS_PARAS)],
        "word_count": [8] * _N_CORPUS_PARAS,
    })
    ppath = tmp_path / "corpus_paras.parquet"
    corpus.to_parquet(ppath, index=False)
    monkeypatch.setattr(ann, "PARAGRAPHS_PATH", ppath)
    monkeypatch.setattr(A, "load", lambda: pd.DataFrame({"doc_name": [_DOC], "decade": [_DECADE], "year": [1855]}))

    if with_annotations:
        rng = list(range(cov_paras))
        pj = pd.DataFrame({
            "doc_name": [_DOC] * cov_paras,
            "para_idx": rng,
            "topics": [["Labor, Wages & Working Conditions"] for _ in rng],
            "party_attack": [i < party_count for i in rng],
            "enemy_naming": [i < 2 for i in rng],
            "zero_sum": [i < 1 for i in rng],
            "proposal_values": ["neither"] * cov_paras,
            "run_id": ["r"] * cov_paras,
        })
        ann.write_annotations("paragraph_annotations", pj, "paragraph")

        ent_name = (lambda i: f"Senator Foo{i}") if entity_good else (lambda i: "Ghost Nobody")
        ent = pd.DataFrame({
            "doc_name": [_DOC, _DOC],
            "para_idx": [0, 1],
            "entity": [ent_name(0), ent_name(1)],
            "type": ["person", "person"],
            "stance": ["adversarial", "adversarial"],
            "run_id": ["r", "r"],
        })
        ann.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        ent.to_parquet(ann.annotation_path("paragraph_entities"), index=False)

    run_dir = ann.RUNS_DIR / "qa"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "requests_index.json").write_text(json.dumps({"c": {"doc_name": _DOC}}))
    if results is not None:
        def _line(kind: str) -> dict:
            if kind == "ok":
                return {"custom_id": "c", "result": {"type": "succeeded",
                        "message": {"content": [{"type": "text", "text": "{}"}]}}}
            return {"custom_id": "c", "result": {"type": "errored", "error": {"type": "invalid_request"}}}
        (run_dir / "results.jsonl").write_text("".join(json.dumps(_line(k)) + "\n" for k in results))
    return "qa"


def _checks(run_id="qa"):
    return {name: passed for name, passed, _ in A._compute_qa(run_id)["checks"]}


# ---------------------------------------------------------------------------
# healthy pilot passes every gate
# ---------------------------------------------------------------------------


def test_pilot_passes_all_gates_on_healthy_data(args, monkeypatch, tmp_path, capsys):
    _setup_qa(monkeypatch, tmp_path, results=["ok"] * 10)  # 0% schema failure
    A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))  # must NOT raise

    out = capsys.readouterr().out
    assert "PILOT QA PASSED" in out
    checks = _checks()
    # every gate present and green
    assert checks["paragraph_coverage>=99%"] is True
    assert checks["schema_failure_rate<1%"] is True
    assert checks["party_attack_non_degenerate"] is True
    assert checks["entity_name_in_paragraph>=80%"] is True
    assert all(checks.values())


# ---------------------------------------------------------------------------
# each gate can FAIL a bad pilot (SystemExit)
# ---------------------------------------------------------------------------


def test_pilot_fails_on_low_paragraph_coverage(args, monkeypatch, tmp_path):
    _setup_qa(monkeypatch, tmp_path, cov_paras=8)  # 8/10 = 80% < 99%
    assert _checks()["paragraph_coverage>=99%"] is False
    with pytest.raises(SystemExit, match="PILOT QA FAILED"):
        A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))


def test_pilot_fails_on_degenerate_high_flag_distribution(args, monkeypatch, tmp_path):
    _setup_qa(monkeypatch, tmp_path, party_count=10)  # party_attack at 100% > 60%
    assert _checks()["party_attack_non_degenerate"] is False
    with pytest.raises(SystemExit, match="PILOT QA FAILED"):
        A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))


def test_pilot_fails_on_degenerate_low_flag_distribution(args, monkeypatch, tmp_path):
    # party_attack at 0% -> below the ~0% degeneracy floor (enemy_naming 20% /
    # zero_sum 10% stay healthy, so the FAIL is attributable to the low flag)
    _setup_qa(monkeypatch, tmp_path, party_count=0)
    checks = _checks()
    assert checks["party_attack_non_degenerate"] is False  # 0% is degenerate
    with pytest.raises(SystemExit, match="PILOT QA FAILED"):
        A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))


def test_pilot_fails_when_entity_names_do_not_appear_in_their_paragraphs(args, monkeypatch, tmp_path):
    _setup_qa(monkeypatch, tmp_path, entity_good=False)  # 0% substring hits < 80%
    assert _checks()["entity_name_in_paragraph>=80%"] is False
    with pytest.raises(SystemExit, match="PILOT QA FAILED"):
        A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))


def test_pilot_fails_on_schema_failure_rate_above_one_percent(args, monkeypatch, tmp_path):
    # 2 errored of 10 = 20% >> 1%
    _setup_qa(monkeypatch, tmp_path, results=["ok"] * 8 + ["err"] * 2)
    assert _checks()["schema_failure_rate<1%"] is False
    with pytest.raises(SystemExit, match="PILOT QA FAILED"):
        A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))


# ---------------------------------------------------------------------------
# edge: nothing to gate, and full mode never gates
# ---------------------------------------------------------------------------


def test_pilot_refuses_when_there_are_no_ingested_annotations(args, monkeypatch, tmp_path):
    _setup_qa(monkeypatch, tmp_path, with_annotations=False)  # no parquet, no results
    with pytest.raises(SystemExit, match="nothing to gate"):
        A.cmd_qa(args(run_id="qa", pilot=True, out=str(tmp_path / "qa.md")))


def test_full_mode_reports_a_failing_metric_without_raising(args, monkeypatch, tmp_path, capsys):
    _setup_qa(monkeypatch, tmp_path, cov_paras=8)  # coverage would FAIL a pilot
    A.cmd_qa(args(run_id="qa", pilot=False, out=str(tmp_path / "qa.md")))  # must NOT raise

    out = capsys.readouterr().out
    assert "full: reported" in out
    assert "PILOT QA" not in out
    # the report still records the failing check for a human to read
    report = (tmp_path / "qa.md").read_text()
    assert "[FAIL] paragraph_coverage>=99%" in report


# ---------------------------------------------------------------------------
# the gate CONSTANTS themselves are pinned to task.md's Verification Strategy
# ---------------------------------------------------------------------------


def test_qa_gate_thresholds_match_the_verification_strategy_spec():
    """Decoupled literal spec-anchor on the gate constants.

    The pass/fail tests above use clearly-healthy (100% coverage, 30% flag) and
    clearly-bad (80% coverage, 100%/0% flag) fixtures, so they prove the gate
    *fires* correctly but would still pass if a threshold silently drifted a long
    way (e.g. coverage 0.99 -> 0.85, or degeneracy-high 0.60 -> 0.95 both leave
    every fixture on the same side of the line).

    Only three of these constants are literal task.md Verification-Strategy
    values: coverage >= 99%, schema failure < 1%, and degeneracy-high > 60%. The
    other two are the implementer's operationalization of task.md's *qualitative*
    language — DEGENERATE_LOW=0.005 makes concrete its "~0% corpus-wide" floor,
    and ENTITY_SUBSTRING=0.80 makes concrete its "entity names actually appear in
    their paragraphs (substring check)". Pinning all five here means a drift in
    either a named threshold OR a chosen operationalization trips a test rather
    than quietly loosening the gate that guards the paid run."""
    assert A.QA_MIN_COVERAGE == 0.99            # task.md: paragraph coverage >= 99%
    assert A.QA_MAX_SCHEMA_FAILURE == 0.01      # task.md: schema failure < 1%
    assert A.QA_FLAG_DEGENERATE_HIGH == 0.60    # task.md: a flag firing on > 60% is degenerate
    assert A.QA_FLAG_DEGENERATE_LOW == 0.005    # operationalizes task.md's "~0%" floor
    assert A.QA_MIN_ENTITY_SUBSTRING == 0.80    # operationalizes task.md's "(substring check)"
