from __future__ import annotations

import builtins
import json
import socket
from pathlib import Path

import pytest

from presidential_profiles import annotation_backfill as backfill
from presidential_profiles import annotation_ledger as ledger
from presidential_profiles import annotation_refresh as refresh
from presidential_profiles import annotation_workflow as workflow
from presidential_profiles.prompts import annotation_v1 as av1


def _judgment_run(
    tmp_path: Path, *, mixed_speeches: bool = False
) -> tuple[Path, Path, dict]:
    root = tmp_path / "ledger"
    backfill.write_initial_registry(root)
    registry = root / "specs" / "registry-v1.json"
    subjects = tmp_path / "subjects.json"
    ledger.atomic_write_json(
        subjects,
        [
            {"doc_name": "/speech", "para_idx": 0, "text": "First paragraph."},
            {
                "doc_name": "/other-speech" if mixed_speeches else "/speech",
                "para_idx": 1,
                "text": "Second paragraph.",
            },
        ],
    )
    workflow.initialize_run(
        run_id="judgment-test",
        label_types=sorted(workflow.JUDGMENT_LABEL_TYPES),
        subjects_path=subjects,
        model_id="test-runtime",
        model_identity_source="unknown",
        annotator_id="pytest",
        root=root,
        registry_path=registry,
    )
    assignment = workflow.next_assignment(
        "judgment-test",
        batch_size=2,
        root=root,
        registry_path=registry,
    )
    return root, registry, assignment


def _response(assignment: dict) -> dict:
    topic = av1.JUDGMENT_SCHEMA["properties"]["annotations"]["items"][
        "properties"
    ]["topics"]["items"]["enum"][0]
    return {
        "annotations": [
            {
                "para_idx": target["subject_key"]["para_idx"],
                "topics": [topic],
                "party_attack": False,
                "enemy_naming": False,
                "zero_sum": False,
                "proposal_values": "neither",
                "entities": [],
            }
            for target in assignment["targets"]
        ]
    }


def _planned_bundle_run(
    tmp_path: Path,
) -> tuple[Path, dict, list[dict], list[dict], dict]:
    root = tmp_path / "planned-ledger"
    subjects_path = tmp_path / "planned-subjects.json"
    ledger.atomic_write_json(
        subjects_path,
        [
            {
                "doc_name": "/planned-speech",
                "para_idx": index,
                "text": f"Planned paragraph {index}.",
                "decade": "1900s",
                "planned_batch_id": "batch_test_locked",
                "planned_batch_order": index + 1,
            }
            for index in range(2)
        ],
    )
    bundle = refresh.resolve_bundle(
        "paragraph_judgment_v2_candidate", "candidate-1"
    )
    workflow.initialize_run(
        run_id="planned-test",
        label_types=[
            row["label_type"] for row in bundle["emitted_label_specs"]
        ],
        spec_versions={
            row["label_type"]: row["spec_version"]
            for row in bundle["emitted_label_specs"]
        },
        subjects_path=subjects_path,
        model_id="test-runtime-exact",
        model_identity_source="runtime_metadata",
        model_identity_receipt={
            "model_id": "test-runtime-exact",
            "specificity": "exact_runtime_identifier",
        },
        annotator_id="pytest",
        execution_bundle=bundle,
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    run = workflow._work_run("planned-test", root)
    specs = workflow._specs_for_run(run, ledger.REGISTRY_V2_PATH)
    persisted_subjects = ledger.read_jsonl(
        ledger.work_dir("planned-test", root) / "subjects.jsonl"
    )
    return root, run, specs, persisted_subjects, bundle


def test_frozen_judgment_assignments_never_mix_speeches(tmp_path: Path) -> None:
    _, _, assignment = _judgment_run(tmp_path, mixed_speeches=True)
    assert len(assignment["targets"]) == 1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["annotations"].pop(),
        lambda value: value["annotations"].append(
            json.loads(json.dumps(value["annotations"][0]))
        ),
        lambda value: value["annotations"][0].__setitem__("para_idx", 999),
        lambda value: value["annotations"][0].__setitem__(
            "proposal_values", "drifted-category"
        ),
        lambda value: value["annotations"][0].__setitem__("unknown", True),
    ],
)
def test_invalid_incomplete_duplicate_unassigned_and_drifted_responses_fail_before_writes(
    tmp_path: Path, mutator
) -> None:
    root, registry, assignment = _judgment_run(tmp_path)
    response = _response(assignment)
    mutator(response)
    work = ledger.work_dir("judgment-test", root)
    before = {
        name: (work / name).read_bytes()
        for name in [
            "label_events.jsonl",
            "canonical_label_projection.jsonl",
        ]
    }
    with pytest.raises(workflow.ResponseValidationError):
        workflow.ingest_response(
            "judgment-test",
            assignment["assignment_id"],
            response,
            root=root,
            registry_path=registry,
        )
    assert not (work / "responses" / f"{assignment['assignment_id']}.json").exists()
    assert {
        name: (work / name).read_bytes() for name in before
    } == before


def test_explicit_planned_assignment_builder_preserves_ids_and_render_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assigned_at = "2026-07-25T12:34:56.000000Z"
    monkeypatch.setattr(ledger, "utc_now", lambda: assigned_at)
    root, _, _, _, _ = _planned_bundle_run(tmp_path)
    built = workflow.build_planned_assignment(
        "planned-test",
        "batch_test_locked",
        persist=False,
        check_unassigned=False,
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    issued = workflow.next_assignment(
        "planned-test",
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    rendered_bytes = json.dumps(
        workflow.render_assignment(built),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    assert issued == built
    assert (
        built["assignment_id"],
        built["assignment_input_sha256"],
        ledger.sha256_bytes(rendered_bytes),
    ) == (
        "asg_b03a37f5edf8cb85a5215576044adeeda263395381853835d3b8d5794953d3b1",
        "sha256:b03a37f5edf8cb85a5215576044adeeda263395381853835d3b8d5794953d3b1",
        "sha256:cfae618ae509eaace9cfaf0ea5d0e848c7c0369b74e2866d9f09c8437b131259",
    )


def test_prepare_response_artifacts_is_pure_and_matches_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labeled_at = "2026-07-25T12:34:56.000000Z"
    monkeypatch.setattr(ledger, "utc_now", lambda: labeled_at)
    root, registry, assignment = _judgment_run(tmp_path)
    work = ledger.work_dir("judgment-test", root)
    response = _response(assignment)
    before = {
        path.relative_to(work): path.read_bytes()
        for path in work.rglob("*")
        if path.is_file()
    }
    kwargs = {
        "run_id": "judgment-test",
        "assignment_id": assignment["assignment_id"],
        "response": response,
        "root": root,
        "registry_path": registry,
        "labeled_at": labeled_at,
    }

    prepared = workflow.prepare_response_artifacts(**kwargs)

    assert workflow.prepare_response_artifacts(**kwargs) == prepared
    assert {
        path.relative_to(work): path.read_bytes()
        for path in work.rglob("*")
        if path.is_file()
    } == before
    assert prepared["result"]["n_events"] == 12
    assert workflow.ingest_response(
        "judgment-test",
        assignment["assignment_id"],
        response,
        root=root,
        registry_path=registry,
    ) == prepared["result"]
    assert ledger.read_jsonl(work / "label_events.jsonl") == prepared["all_events"]
    assert ledger.read_jsonl(
        work / "canonical_label_projection.jsonl"
    ) == prepared["all_projections"]
    assert json.loads(
        (work / "responses" / f"{assignment['assignment_id']}.json").read_text(
            encoding="utf-8"
        )
    ) == prepared["saved_response"]


def test_prepare_invalid_response_raises_response_validation_without_writes(
    tmp_path: Path,
) -> None:
    root, registry, assignment = _judgment_run(tmp_path)
    work = ledger.work_dir("judgment-test", root)
    invalid = _response(assignment)
    invalid["annotations"].pop()
    before = {
        path.relative_to(work): path.read_bytes()
        for path in work.rglob("*")
        if path.is_file()
    }

    with pytest.raises(
        workflow.ResponseValidationError,
        match="incomplete, duplicate, or unassigned",
    ):
        workflow.prepare_response_artifacts(
            "judgment-test",
            assignment["assignment_id"],
            response=invalid,
            root=root,
            registry_path=registry,
            labeled_at="2026-07-25T12:34:56.000000Z",
        )

    assert {
        path.relative_to(work): path.read_bytes()
        for path in work.rglob("*")
        if path.is_file()
    } == before


def test_prepare_integrity_drift_is_not_response_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ledger, "utc_now", lambda: "2026-07-25T12:34:56.000000Z"
    )
    _, run, specs, subjects, bundle = _planned_bundle_run(tmp_path)
    assignment = workflow.build_assignment(
        run=run,
        specs=specs,
        bundle=bundle,
        subjects=subjects,
        selected_subjects=subjects,
        context_window=1,
        assigned_at=ledger.utc_now(),
    )
    drifted = {**assignment, "bundle_sha256": "sha256:" + "0" * 64}

    with pytest.raises(ValueError) as raised:
        workflow._prepare_response_artifacts(
            run=run,
            assignment=drifted,
            response={},
            specs=specs,
            bundle=bundle,
            existing_events=[],
            existing_projections=[],
            labeled_at=ledger.utc_now(),
        )
    assert not isinstance(raised.value, workflow.ResponseValidationError)
    assert str(raised.value) == "assignment/bundle identity drift"


def test_empty_entity_outputs_are_explicit_only_for_new_terminal_responses(
    tmp_path: Path,
) -> None:
    root, registry, assignment = _judgment_run(tmp_path)
    workflow.ingest_response(
        "judgment-test",
        assignment["assignment_id"],
        _response(assignment),
        root=root,
        registry_path=registry,
    )
    events = ledger.read_jsonl(
        ledger.work_dir("judgment-test", root) / "label_events.jsonl"
    )
    entity_events = [row for row in events if row["label_type"] == "entities"]
    assert len(entity_events) == 2
    assert {row["event_role"] for row in entity_events} == {"empty_group_marker"}
    assert {row["raw_value_json"] for row in entity_events} == {"[]"}


def test_runtime_model_identity_requires_an_exact_receipt(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    backfill.write_initial_registry(root)
    registry = root / "specs" / "registry-v1.json"
    subjects = tmp_path / "subjects.json"
    ledger.atomic_write_json(
        subjects,
        [{"doc_name": "/speech", "para_idx": 0, "text": "Text."}],
    )
    with pytest.raises(ValueError, match="requires a receipt"):
        workflow.initialize_run(
            run_id="runtime-no-receipt",
            label_types=["topics"],
            subjects_path=subjects,
            model_id="claimed-model",
            model_identity_source="runtime_metadata",
            annotator_id="pytest",
            root=root,
            registry_path=registry,
        )
    with pytest.raises(ValueError, match="not exact enough"):
        workflow.initialize_run(
            run_id="runtime-family-only",
            label_types=["topics"],
            subjects_path=subjects,
            model_id="gpt-family",
            model_identity_source="runtime_metadata",
            annotator_id="pytest",
            model_identity_receipt={
                "model_id": "gpt-family",
                "specificity": "family_name",
            },
            root=root,
            registry_path=registry,
        )


def test_core_workflow_neither_imports_model_clients_nor_opens_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name.split(".", 1)[0] in {
            "anthropic",
            "openai",
            "httpx",
            "requests",
            "aiohttp",
        }:
            raise AssertionError(f"model/network client import attempted: {name}")
        return real_import(name, *args, **kwargs)

    def blocked_connection(*args: object, **kwargs: object):
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", blocked_connection)
    root, registry, assignment = _judgment_run(tmp_path)
    workflow.ingest_response(
        "judgment-test",
        assignment["assignment_id"],
        _response(assignment),
        root=root,
        registry_path=registry,
    )
    assert workflow.audit_run(
        "judgment-test", root=root, registry_path=registry
    )["status"] == "ok"
