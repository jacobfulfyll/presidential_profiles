from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from presidential_profiles import annotation_ledger as ledger
from presidential_profiles import annotation_refresh as refresh
from presidential_profiles import annotation_refresh_provisional as provisional
from presidential_profiles import (
    annotation_refresh_provisional_execution as execution,
)
from presidential_profiles import annotation_workflow as workflow
from presidential_profiles import annotation_refresh_codex_worker as codex_worker
from presidential_profiles import (
    annotation_refresh_codex_coordinator as codex_coordinator,
)


def _runtime_receipt(session: str, **overrides: object) -> dict:
    value = {
        "receipt_version": "pytest-runtime-v1",
        "model_id": "gpt-5.6-sol",
        "specificity": "exact_runtime_identifier",
        "reasoning_effort": "high",
        "session_id": session,
        "source": "pytest",
        "fresh_restricted_labeling_session": True,
        "sibling_responses_inspected": False,
    }
    value.update(overrides)
    return value


def _mini_campaign(tmp_path: Path, n_assignments: int = 4) -> tuple[Path, str]:
    root = tmp_path / "ledger"
    run_id = execution._run_id()
    bundle = refresh.resolve_bundle(
        provisional.COMBINED_BUNDLE_ID,
        provisional.COMBINED_BUNDLE_VERSION,
        registry_path=refresh.BUNDLE_REGISTRY_PATH,
        label_registry_path=ledger.REGISTRY_V2_PATH,
    )
    subjects_path = tmp_path / "subjects.json"
    subjects = []
    for index in range(n_assignments):
        subjects.append(
            {
                "doc_name": f"/speech-{index}",
                "para_idx": 0,
                "text": f"Paragraph {index}.",
                "planned_batch_id": f"batch-{index}",
                "planned_batch_order": 1,
                "context_before_json": "[]",
                "context_after_json": "[]",
            }
        )
    ledger.atomic_write_json(subjects_path, subjects)
    refresh.initialize_bundle_run(
        bundle_id=bundle["bundle_id"],
        bundle_version=bundle["bundle_version"],
        run_id=run_id,
        subjects_path=subjects_path,
        model_receipt=_runtime_receipt("initializer"),
        reasoning_effort="high",
        annotator_id="pytest",
        root=root,
        bundle_registry_path=refresh.BUNDLE_REGISTRY_PATH,
        label_registry_path=ledger.REGISTRY_V2_PATH,
        authorization_review_item_id="ARCV1-G007",
    )
    run_dir = ledger.work_dir(run_id, root)
    (run_dir / "accepted-responses").mkdir()
    (run_dir / "execution-session-receipts").mkdir()
    (run_dir / "attempt-receipts").mkdir()
    run = workflow._work_run(run_id, root)
    specs = workflow._specs_for_run(run, ledger.REGISTRY_V2_PATH)
    stored = ledger.read_jsonl(run_dir / "subjects.jsonl")
    rows = []
    for index, subject in enumerate(stored):
        assignment = workflow.build_assignment(
            run=run,
            specs=specs,
            bundle=bundle,
            subjects=stored,
            selected_subjects=[subject],
            context_window=1,
            assigned_at="2026-07-25T00:00:00.000000Z",
        )
        ledger.atomic_write_json(
            run_dir / "assignments" / f"{assignment['assignment_id']}.json",
            assignment,
        )
        rendered = workflow.render_assignment(assignment)
        rows.append(
            {
                "assignment_index": index,
                "assignment_id": assignment["assignment_id"],
                "planned_batch_id": subject["planned_batch_id"],
                "rendered_sha256": ledger.sha256_text(
                    json.dumps(
                        rendered,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                ),
                "input_proxy_tokens": execution._assignment_proxy_tokens(
                    rendered
                ),
            }
        )
    semantic = {
        "run_id": run_id,
        "plan_sha256": "sha256:" + "1" * 64,
        "status": "initialized",
    }
    campaign = execution._artifact(
        semantic,
        prefix="pcamp",
        id_field="campaign_id",
        sha_field="campaign_sha256",
    )
    campaign_path = (
        root
        / execution.CAMPAIGNS_DIR
        / campaign["campaign_id"]
        / "campaign.json"
    )
    ledger.atomic_write_json(
        campaign_path,
        {**campaign, "initialized_at": "2026-07-25T00:00:00.000000Z"},
    )
    execution._initialize_control(
        run_dir, campaign_id=campaign["campaign_id"], assignments=rows
    )
    ledger.write_jsonl(run_dir / "assignment-index.jsonl", rows)
    ledger.atomic_write_json(
        run_dir / execution.BASELINE_FILENAME,
        execution._protected_state(root),
    )
    return root, campaign["campaign_id"]


def test_retry_policy_is_content_addressed_and_mechanical() -> None:
    policy = execution._retry_policy()
    semantic = dict(policy)
    policy_id = semantic.pop("retry_policy_id")
    policy_sha = semantic.pop("retry_policy_sha256")
    assert policy_sha == ledger.sha256_text(ledger.canonical_json(semantic))
    assert policy_id == "retry_" + policy_sha.removeprefix("sha256:")
    assert policy["maximum_model_invocations_per_assignment"] == 3
    assert policy["repair_prompt"] == "prohibited"
    assert policy["semantic_disagreement_retry"] == "prohibited"
    assert policy["input_token_ceiling"] == 89_830_303


def test_attempt_limit_exception_is_assignment_scoped(
    tmp_path: Path,
) -> None:
    root, _ = _mini_campaign(tmp_path, n_assignments=2)
    run_dir = ledger.work_dir(execution._run_id(), root)
    connection = execution._connect(run_dir)
    try:
        rows = list(
            connection.execute(
                "SELECT assignment_id FROM assignments ORDER BY assignment_index"
            )
        )
        connection.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?)",
            (
                "attempt_limit_exceptions",
                ledger.canonical_json({rows[0]["assignment_id"]: 4}),
            ),
        )
        assert execution._attempt_limit(
            connection, rows[0]["assignment_id"]
        ) == 4
        assert execution._attempt_limit(
            connection, rows[1]["assignment_id"]
        ) == 3
        connection.execute(
            "UPDATE meta SET value_json = ? "
            "WHERE key = 'attempt_limit_exceptions'",
            (
                ledger.canonical_json({rows[0]["assignment_id"]: 5}),
            ),
        )
        assert execution._attempt_limit(
            connection, rows[0]["assignment_id"]
        ) == 5
        connection.execute(
            "UPDATE meta SET value_json = ? "
            "WHERE key = 'attempt_limit_exceptions'",
            (
                ledger.canonical_json({rows[0]["assignment_id"]: 6}),
            ),
        )
        assert execution._attempt_limit(
            connection, rows[0]["assignment_id"]
        ) == 6
    finally:
        connection.close()


def test_declared_attempt_limit_exceptions_cover_all_authorized_amendments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [
        (
            execution.RECOVERY_AMENDMENT_RESOLUTION_ID,
            4,
            12,
            "assignment-12",
        ),
        (
            execution.VALIDATOR_REPAIR_RESOLUTION_ID,
            5,
            12,
            "assignment-12",
        ),
        (
            execution.VALIDATOR_RETRY_POLICY_RESOLUTION_ID,
            4,
            41,
            "assignment-41",
        ),
        (
            execution.ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID,
            4,
            8912,
            "assignment-8912",
        ),
    ]
    monkeypatch.setattr(
        execution,
        "_validated_execution_amendments",
        lambda _run_dir: (
            [
                {
                    "resolution_id": resolution_id,
                    "retry_exception": {
                        "amended_maximum_attempts": maximum,
                        "assignment_index": assignment_index,
                        "assignment_id": assignment_id,
                    },
                }
                for resolution_id, maximum, assignment_index, assignment_id
                in expected
            ]
            + [
                {
                    "resolution_id":
                    execution.TARGETED_LITERAL_RETRY_RESOLUTION_ID,
                },
                {
                    "resolution_id":
                    execution.COMPLETION_AMENDMENT_RESOLUTION_ID,
                    "retry_exceptions": [
                        {
                            "assignment_index": assignment_index,
                            "assignment_id":
                            f"assignment-{assignment_index}",
                            "base_maximum_attempts": 3,
                            "amended_maximum_attempts": 6,
                            "additional_attempts": 3,
                        }
                        for assignment_index in (
                            execution.COMPLETION_RETRY_ASSIGNMENT_INDICES
                        )
                    ],
                }
            ]
        ),
    )
    targeted_feedback = execution.targeted_literal_retry_feedback(
        [
            {
                "local_id": 2,
                "claim_index": 1,
                "group_text": "the people",
                "exact_candidate":
                "It has stood for progress; it has stood for the people.",
                "exact_candidate_verified": True,
            }
        ]
    )
    monkeypatch.setattr(
        execution,
        "_validated_literal_retry_grants",
        lambda _run_dir: [
            {
                "resolution_id":
                execution.TARGETED_LITERAL_RETRY_RESOLUTION_ID,
                "assignment_id": "assignment-targeted",
                "base_maximum_attempts": 3,
                "amended_maximum_attempts": 4,
                "additional_attempts": 1,
                "verified_literal_diagnostics": [{"verified": True}],
                "validator_feedback": {
                    "text": targeted_feedback,
                    "sha256": ledger.sha256_text(targeted_feedback),
                },
            }
        ],
    )
    assert execution._declared_attempt_limit_exceptions(Path(".")) == {
        "assignment-12": 5,
        "assignment-41": 4,
        "assignment-8912": 4,
        "assignment-targeted": 4,
        **{
            f"assignment-{assignment_index}": 6
            for assignment_index in (
                execution.COMPLETION_RETRY_ASSIGNMENT_INDICES
            )
        },
    }


def test_effective_input_token_ceiling_requires_a_valid_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution,
        "_validated_execution_amendments",
        lambda _run_dir: [
            {
                "input_token_ceiling_amendment": {
                    "previous_value": execution.INPUT_TOKEN_CEILING,
                    "replacement_value":
                    execution.COMPLETION_INPUT_TOKEN_CEILING,
                }
            }
        ],
    )
    assert execution._effective_input_token_ceiling(Path(".")) == 93_000_000

    monkeypatch.setattr(
        execution,
        "_validated_execution_amendments",
        lambda _run_dir: [
            {
                "input_token_ceiling_amendment": {
                    "previous_value": execution.INPUT_TOKEN_CEILING - 1,
                    "replacement_value":
                    execution.COMPLETION_INPUT_TOKEN_CEILING,
                }
            }
        ],
    )
    with pytest.raises(execution.IntegrityDriftError):
        execution._effective_input_token_ceiling(Path("."))


def test_literal_retry_feedback_preserves_damaged_transcript_wording() -> None:
    target = "t has stood for concern for the people's welfare."
    submitted = "It has stood for concern for the people's welfare."
    diagnostics = [
        {
            "local_id": 1,
            "claim_index": 0,
            "group_text": "the people",
            "exact_candidate": execution._closest_exact_evidence_span(
                target,
                "the people",
                submitted,
            ),
            "exact_candidate_verified": True,
        }
    ]
    assert diagnostics[0]["exact_candidate"] == target
    feedback = execution.targeted_literal_retry_feedback(diagnostics)
    assert json.dumps(target) in feedback
    assert json.dumps(submitted) not in feedback
    assert "shorten it to an exact target substring" in feedback
    assert "never repair the target" in feedback


def test_codex_worker_receipt_and_prompt_are_exact_and_assignment_only() -> None:
    receipt = codex_worker._runtime_receipt(
        "launcher",
        codex_version="codex-test",
        range_start=0,
        range_end=1,
    )
    assert receipt["model_id"] == "gpt-5.6-sol"
    assert receipt["reasoning_effort"] == "high"
    assert receipt["fresh_restricted_labeling_session"] is True
    assert receipt["sibling_responses_inspected"] is False
    rendered = {
        "prompt": "locked prompt",
        "targets": [{"local_id": 1, "text": "Only this."}],
        "response_schema": {"type": "object"},
    }
    prompt = codex_worker._prompt(rendered)
    assert "Only this." in prompt
    assert "Do not use tools" in prompt
    assert "sibling" not in prompt.lower()
    repair_prompt = codex_worker._prompt(
        rendered,
        repair_feedback=execution.VALIDATOR_REPAIR_FEEDBACK,
    )
    assert repair_prompt.startswith(prompt)
    assert execution.VALIDATOR_REPAIR_FEEDBACK in repair_prompt
    assert "prior response" in repair_prompt
    locked_schema = {
        "type": "array",
        "items": {"type": "string"},
        "uniqueItems": True,
    }
    projected = codex_worker._transport_schema(locked_schema)
    assert locked_schema["uniqueItems"] is True
    assert projected == {
        "type": "array",
        "items": {"type": "string"},
    }
    events, thread_id, usage = codex_worker._parse_events(
        "\n".join(
            [
                json.dumps(
                    {"type": "thread.started", "thread_id": "thread-1"}
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 2,
                        },
                    }
                ),
            ]
        )
    )
    assert len(events) == 2
    assert thread_id == "thread-1"
    assert usage["input_tokens"] == 10


def test_codex_coordinator_partitions_exactly_without_overlap() -> None:
    assert codex_coordinator.partition_ranges(1, 10, 4) == [
        (1, 4),
        (4, 6),
        (6, 8),
        (8, 10),
    ]
    assert codex_coordinator.partition_ranges(4, 6, 8) == [(4, 5), (5, 6)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": "other-model"},
        {"specificity": "family_name"},
        {"reasoning_effort": "xhigh"},
        {"fresh_restricted_labeling_session": False},
        {"sibling_responses_inspected": True},
    ],
)
def test_claim_rejects_runtime_or_blindness_drift_before_state(
    tmp_path: Path, overrides: dict
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    run_dir = ledger.work_dir(execution._run_id(), root)
    before = (run_dir / execution.CONTROL_FILENAME).read_bytes()
    with pytest.raises((ValueError, execution.IntegrityDriftError)):
        execution.claim_provisional_assignment(
            campaign_id,
            runtime_receipt=_runtime_receipt("worker", **overrides),
            range_end=1,
            root=root,
        )
    assert (run_dir / execution.CONTROL_FILENAME).read_bytes() == before


def test_concurrent_claims_are_atomic_unique_and_range_bounded(
    tmp_path: Path,
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=8)

    def claim(index: int) -> dict:
        return execution.claim_provisional_assignment(
            campaign_id,
            runtime_receipt=_runtime_receipt(f"worker-{index}"),
            range_end=8,
            root=root,
            now=f"2026-07-25T00:00:{index:02d}.000000Z",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert {row["status"] for row in results} == {"claimed"}
    assert len({row["assignment_id"] for row in results}) == 8
    assert len({row["claim_id"] for row in results}) == 8
    assert {
        row["assignment_index"] for row in results
    } == set(range(8))
    assert all(
        "rendered_assignment" in row and "response" not in row
        for row in results
    )


def test_expired_claim_is_reclaimed_without_duplicate_acceptance(
    tmp_path: Path,
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    first = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=_runtime_receipt("worker-1"),
        range_end=1,
        lease_seconds=60,
        root=root,
        now="2026-07-25T00:00:00.000000Z",
    )
    second = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=_runtime_receipt("worker-2"),
        range_end=1,
        root=root,
        now="2026-07-25T00:02:00.000000Z",
    )
    assert second["assignment_id"] == first["assignment_id"]
    assert second["claim_id"] != first["claim_id"]
    run_dir = ledger.work_dir(execution._run_id(), root)
    connection = execution._connect(run_dir)
    try:
        events = [
            row["event_type"]
            for row in connection.execute(
                "SELECT event_type FROM claim_events ORDER BY event_seq"
            )
        ]
    finally:
        connection.close()
    assert events == ["claimed", "claim_expired", "claimed"]


def test_valid_attempt_is_accepted_once_and_prepared_without_shared_writes(
    tmp_path: Path,
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    receipt = _runtime_receipt("worker")
    claim = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=receipt,
        range_end=1,
        root=root,
    )
    started = execution.start_provisional_attempt(
        campaign_id,
        claim["assignment_id"],
        claim["claim_id"],
        runtime_receipt=receipt,
        root=root,
    )
    response = claim["rendered_assignment"]["response_template"]
    run_dir = ledger.work_dir(execution._run_id(), root)
    before_events = (run_dir / "label_events.jsonl").read_bytes()
    result = execution.complete_provisional_attempt(
        campaign_id,
        started["attempt_id"],
        raw_output=response,
        runtime_receipt=receipt,
        execution_receipt={
            "session_id": "worker",
            "model_id": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "source": "pytest",
            "status": "completed",
        },
        root=root,
    )
    assert result["status"] == "accepted"
    assert (run_dir / "label_events.jsonl").read_bytes() == before_events
    package = execution._load_json(
        run_dir
        / "accepted-responses"
        / f"{claim['assignment_id']}.json"
    )
    assert len(package["field_values"]) == 7
    assert len(
        {
            (row["canonical_subject_id"], row["label_type"])
            for row in package["field_values"]
        }
    ) == 7
    again = execution.complete_provisional_attempt(
        campaign_id,
        started["attempt_id"],
        raw_output={"different": True},
        runtime_receipt=receipt,
        execution_receipt={
            "session_id": "worker",
            "model_id": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "source": "pytest",
            "status": "completed",
        },
        root=root,
    )
    assert again["status"] == "accepted"
    assert len(list((run_dir / "accepted-responses").glob("*.json"))) == 1


def test_invalid_attempts_are_logged_and_bounded_without_response_writes(
    tmp_path: Path,
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    run_dir = ledger.work_dir(execution._run_id(), root)
    for attempt_no in range(1, 4):
        receipt = _runtime_receipt(f"worker-{attempt_no}")
        claim = execution.claim_provisional_assignment(
            campaign_id,
            runtime_receipt=receipt,
            range_end=1,
            root=root,
            now=f"2026-07-25T00:0{attempt_no}:00.000000Z",
        )
        started = execution.start_provisional_attempt(
            campaign_id,
            claim["assignment_id"],
            claim["claim_id"],
            runtime_receipt=receipt,
            root=root,
            now=f"2026-07-25T00:0{attempt_no}:01.000000Z",
        )
        result = execution.complete_provisional_attempt(
            campaign_id,
            started["attempt_id"],
            raw_output="{not-json",
            runtime_receipt=receipt,
            execution_receipt={
                "session_id": f"worker-{attempt_no}",
                "model_id": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "source": "pytest",
                "status": "completed",
            },
            root=root,
            now=f"2026-07-25T00:0{attempt_no}:02.000000Z",
        )
        assert result["validation_code"] == "json_parse_failure"
        assert result["assignment_status"] == (
            "terminal_failed" if attempt_no == 3 else "available"
        )
        assert execution.provisional_assignment_status(
            campaign_id,
            claim["assignment_id"],
            root=root,
        ) == result["assignment_status"]
    status = execution.provisional_execution_status(
        campaign_id, root=root
    )
    assert status["status"] == "retry_exhausted_blocked"
    assert status["assignments"] == {"terminal_failed": 1}
    assert status["attempts"] == {"schema_invalid": 3}
    assert len(list((run_dir / "attempt-receipts").glob("*.json"))) == 3
    assert not list((run_dir / "responses").glob("*.json"))
    assert not list((run_dir / "accepted-responses").glob("*.json"))
    complete = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=_runtime_receipt("worker-4"),
        range_end=1,
        root=root,
    )
    assert complete["status"] == "range_complete"


def test_validator_feedback_is_conditional_on_recorded_constituency_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    receipt = _runtime_receipt("worker-1")
    claim = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=receipt,
        range_end=1,
        root=root,
    )
    started = execution.start_provisional_attempt(
        campaign_id,
        claim["assignment_id"],
        claim["claim_id"],
        runtime_receipt=receipt,
        root=root,
    )
    execution.complete_provisional_attempt(
        campaign_id,
        started["attempt_id"],
        raw_output="{not-json",
        runtime_receipt=receipt,
        execution_receipt={
            "session_id": "worker-1",
            "model_id": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "source": "pytest",
            "status": "completed",
        },
        root=root,
    )
    run_dir = ledger.work_dir(execution._run_id(), root)
    connection = execution._connect(run_dir)
    try:
        connection.execute(
            """
            UPDATE attempts
            SET validation_code = ?,
                validation_error = ?
            WHERE attempt_id = ?
            """,
            (
                "registered_special_validator_failure",
                "constituency claim 0: explicit claim has a resolved referent",
                started["attempt_id"],
            ),
        )
    finally:
        connection.close()
    monkeypatch.setattr(
        execution,
        "_validated_execution_amendments",
        lambda _run_dir: [
            {
                "resolution_id":
                execution.VALIDATOR_RETRY_POLICY_RESOLUTION_ID
            }
        ],
    )
    assert (
        execution.validator_feedback_for_attempt(
            campaign_id,
            claim["assignment_id"],
            1,
            root=root,
        )
        is None
    )
    assert execution.validator_feedback_for_attempt(
        campaign_id,
        claim["assignment_id"],
        2,
        root=root,
    ) == execution.VALIDATOR_REPAIR_FEEDBACK
    monkeypatch.setattr(
        execution,
        "_validated_execution_amendments",
        lambda _run_dir: [
            {
                "resolution_id":
                execution.VALIDATOR_RETRY_POLICY_RESOLUTION_ID
            },
            {
                "resolution_id":
                execution.ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID
            },
        ],
    )
    assert execution.validator_feedback_for_attempt(
        campaign_id,
        claim["assignment_id"],
        2,
        root=root,
    ) == execution.ENHANCED_VALIDATOR_REPAIR_FEEDBACK
    assert "without correcting grammar" in (
        execution.ENHANCED_VALIDATOR_REPAIR_FEEDBACK
    )
    monkeypatch.setattr(
        execution,
        "_validated_execution_amendments",
        lambda _run_dir: [
            {
                "resolution_id":
                execution.VALIDATOR_RETRY_POLICY_RESOLUTION_ID
            },
            {
                "resolution_id":
                execution.ENHANCED_VALIDATOR_RETRY_RESOLUTION_ID
            },
            {
                "resolution_id":
                execution.TARGETED_LITERAL_RETRY_RESOLUTION_ID
            },
        ],
    )
    exact_candidate = (
        "It has stood for progress; it has stood for the people's welfare."
    )
    monkeypatch.setattr(
        execution,
        "diagnose_constituency_literal_mismatches",
        lambda *_args, **_kwargs: [
            {
                "attempt_no": 1,
                "local_id": 1,
                "claim_index": 0,
                "group_text": "the people",
                "exact_candidate": exact_candidate,
                "exact_candidate_verified": True,
            }
        ],
    )
    connection = execution._connect(run_dir)
    try:
        connection.execute(
            """
            UPDATE attempts
            SET validation_error = ?
            WHERE attempt_id = ?
            """,
            (
                "constituency claim 0: evidence_span is not an exact "
                "target substring",
                started["attempt_id"],
            ),
        )
    finally:
        connection.close()
    targeted = execution.validator_feedback_for_attempt(
        campaign_id,
        claim["assignment_id"],
        2,
        root=root,
    )
    assert targeted is not None
    assert json.dumps(exact_candidate) in targeted
    assert "shorten it to an exact target substring" in targeted


def test_token_ceiling_refuses_attempt_before_model_invocation(
    tmp_path: Path,
) -> None:
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    receipt = _runtime_receipt("worker")
    claim = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=receipt,
        range_end=1,
        root=root,
    )
    run_dir = ledger.work_dir(execution._run_id(), root)
    connection = execution._connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE assignments SET input_proxy_tokens = ?
            WHERE assignment_id = ?
            """,
            (execution.INPUT_TOKEN_CEILING + 1, claim["assignment_id"]),
        )
        connection.execute("COMMIT")
    finally:
        connection.close()
    with pytest.raises(execution.TokenCeilingError):
        execution.start_provisional_attempt(
            campaign_id,
            claim["assignment_id"],
            claim["claim_id"],
            runtime_receipt=receipt,
            root=root,
        )
    status = execution.provisional_execution_status(
        campaign_id, root=root
    )
    assert status["status"] == "token_ceiling_blocked"
    assert status["attempts"] == {}


def test_complete_audit_seal_and_composite_are_idempotent_and_nonpromoting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution, "EXPECTED_ASSIGNMENTS", 1)
    monkeypatch.setattr(execution, "EXPECTED_FRESH_SUBJECTS", 1)
    monkeypatch.setattr(execution, "EXPECTED_FRESH_FIELDS", 7)
    monkeypatch.setattr(execution, "EXPECTED_REUSED_SUBJECTS", 1)
    monkeypatch.setattr(execution, "EXPECTED_REUSED_FIELDS", 1)
    monkeypatch.setattr(execution, "EXPECTED_TOTAL_SUBJECTS", 2)
    monkeypatch.setattr(execution, "EXPECTED_TOTAL_FIELDS", 8)
    root, campaign_id = _mini_campaign(tmp_path, n_assignments=1)
    protected_before = execution._protected_state(root)
    receipt = _runtime_receipt("worker")
    claim = execution.claim_provisional_assignment(
        campaign_id,
        runtime_receipt=receipt,
        range_end=1,
        root=root,
    )
    started = execution.start_provisional_attempt(
        campaign_id,
        claim["assignment_id"],
        claim["claim_id"],
        runtime_receipt=receipt,
        root=root,
    )
    accepted = execution.complete_provisional_attempt(
        campaign_id,
        started["attempt_id"],
        raw_output=claim["rendered_assignment"]["response_template"],
        runtime_receipt=receipt,
        execution_receipt={
            "session_id": "worker",
            "model_id": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "source": "pytest",
            "status": "completed",
        },
        root=root,
    )
    assert accepted["status"] == "accepted"
    compacted = execution.compact_provisional_work_storage(
        campaign_id, root=root
    )
    assert compacted["converted_files"] == 1
    audit = execution.audit_provisional_execution(
        campaign_id, root=root, consolidate=True
    )
    assert audit["status"] == "ok", audit
    assert audit["assignments"] == 1
    assert audit["subjects"] == 1
    assert audit["subject_label_fields"] == 7
    assert execution._protected_state(root) == protected_before

    sealed = execution.seal_provisional_execution(
        campaign_id, root=root
    )
    assert sealed["status"] == "sealed"
    again = execution.seal_provisional_execution(
        campaign_id, root=root
    )
    assert again["status"] == "already_sealed"
    assert again["artifact_set_sha256"] == sealed["artifact_set_sha256"]

    selected_value: list[str] = []
    selected_json = ledger.canonical_json(selected_value)
    reuse = execution._artifact(
        {
            "reuse_manifest_version": "pytest-reuse-v1",
            "scope": "provisional-corpus",
            "status": "evaluated_reuse",
            "counts": {
                "subjects": 1,
                "subject_label_entries": 1,
            },
            "entries": [
                {
                    "canonical_subject_id": "sub_" + "f" * 64,
                    "label_type": "topics",
                    "selected_value": selected_value,
                    "selected_value_sha256": ledger.sha256_text(selected_json),
                    "confidence_provenance": "three_way_same_model_unanimous",
                }
            ],
        },
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    monkeypatch.setattr(execution, "REUSE_ID", reuse["reuse_manifest_id"])
    ledger.atomic_write_json(
        root
        / provisional.REUSE_MANIFESTS_DIR
        / f"{reuse['reuse_manifest_id']}.json",
        reuse,
    )
    composite = execution.publish_provisional_composite(
        campaign_id, root=root
    )
    assert composite["status"] == "published"
    assert composite["counts"] == {
        "fresh_assignments": 1,
        "fresh_subjects": 1,
        "fresh_subject_label_fields": 7,
        "reused_subjects": 1,
        "reused_subject_label_fields": 1,
        "total_subjects": 2,
        "total_subject_label_fields": 8,
        "missing_or_duplicate_keys": 0,
    }
    repeated = execution.publish_provisional_composite(
        campaign_id, root=root
    )
    assert repeated["status"] == "already_published"
    assert repeated["composite_id"] == composite["composite_id"]
    assert execution._protected_state(root) == protected_before
    assert not (root / "materialized" / "current").exists()
    assert not (root / "decisions" / "promotions").exists()
