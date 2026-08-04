from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from presidential_profiles import annotation_backfill as backfill
from presidential_profiles import annotation_ledger as ledger
from presidential_profiles import annotation_workflow as workflow


def _registry_with_imaginary_spec(root: Path) -> Path:
    backfill.write_initial_registry(root)
    registry_path = root / "specs" / "registry-v1.json"
    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number"},
            "signals": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["confidence", "signals"],
        "additionalProperties": False,
    }
    spec = backfill._make_spec(
        label_type="imaginary_signal",
        spec_version="test-v1",
        stability="test",
        subject_type="paragraph",
        historical_quantity="Imaginary extensibility-test signal.",
        prompt_version="imaginary/test-v1",
        prompt_text="Return one imaginary structured signal.",
        response_schema=schema,
        kind="object",
        extractor={"mode": "single"},
        context_policy={"neighbors_each_side": 0},
        special_validator=None,
        source_spec_refs=[],
        limitations=["Test-only specification."],
    )
    spec_path = root / "specs" / "imaginary_signal" / "test-v1.json"
    ledger.atomic_write_json(spec_path, spec)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["entries"].append(
        {
            "label_type": "imaginary_signal",
            "spec_version": "test-v1",
            "spec_sha256": spec["spec_sha256"],
            "stability": "test",
            "path": "imaginary_signal/test-v1.json",
        }
    )
    registry["entries"] = sorted(
        registry["entries"],
        key=lambda row: (row["label_type"], row["spec_version"]),
    )
    registry["registry_sha256"] = ledger.sha256_text(
        ledger.canonical_json(
            {key: value for key, value in registry.items() if key != "registry_sha256"}
        )
    )
    ledger.atomic_write_json(registry_path, registry)
    ledger.read_registry(registry_path)
    return registry_path


def _complete_imaginary_run(root: Path) -> tuple[Path, dict, dict]:
    registry_path = _registry_with_imaginary_spec(root)
    subjects_path = root / "selection.json"
    ledger.atomic_write_json(
        subjects_path,
        [
            {
                "doc_name": "/imaginary-speech",
                "para_idx": 4,
                "text": "An imaginary signal appears here.",
            }
        ],
    )
    workflow.initialize_run(
        run_id="imaginary-run",
        label_types=["imaginary_signal"],
        subjects_path=subjects_path,
        model_id="local-test-annotator",
        model_identity_source="unknown",
        annotator_id="pytest",
        root=root,
        registry_path=registry_path,
    )
    assignment = workflow.next_assignment(
        "imaginary-run", root=root, registry_path=registry_path
    )
    response = {
        "labels": [
            {
                "subject_key": assignment["targets"][0]["subject_key"],
                "values": {
                    "imaginary_signal": {
                        "confidence": 0.75,
                        "signals": ["novel", "structured"],
                    }
                },
            }
        ]
    }
    workflow.ingest_response(
        "imaginary-run",
        assignment["assignment_id"],
        response,
        root=root,
        registry_path=registry_path,
    )
    return registry_path, assignment, response


def test_extensible_label_passes_full_core_lifecycle_without_core_edits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    registry_path, _, _ = _complete_imaginary_run(root)
    assert workflow.audit_run(
        "imaginary-run", root=root, registry_path=registry_path
    )["status"] == "ok"
    sealed = ledger.seal_run(
        "imaginary-run",
        root=root,
        registry_path=registry_path,
        sealed_at="2026-07-24T12:00:00Z",
    )
    first = ledger.materialize(root)
    run_dir = ledger.sealed_artifact_dir("imaginary-run", root)
    events = ledger.read_jsonl(run_dir / "label_events.jsonl")
    assert len(events) == 1
    assert json.loads(events[0]["raw_value_json"]) == {
        "confidence": 0.75,
        "signals": ["novel", "structured"],
    }
    comparison = ledger.compare_runs(
        "imaginary-run", "imaginary-run", root=root
    )
    assert comparison["overlap"] == 1
    assert comparison["n_disagreements"] == 0
    projection = ledger.read_jsonl(
        run_dir / "canonical_label_projection.jsonl"
    )[0]
    adjudication = ledger.publish_adjudication(
        {
            "canonical_subject_id": projection["canonical_subject_id"],
            "label_type": "imaginary_signal",
            "spec_sha256": events[0]["spec_sha256"],
            "candidate_label_group_ids": [events[0]["label_group_id"]],
            "resolution_kind": "select_existing",
            "selected_label_group_id": events[0]["label_group_id"],
            "resolved_label_group_id": events[0]["label_group_id"],
            "adjudicator_id": "pytest",
            "rationale": "Acceptance-test selection.",
            "review_resolution_id": None,
            "decided_at": "2026-07-24T12:01:00Z",
        },
        root=root,
    )
    promoted = ledger.publish_promotions(
        [
            {
                "label_group_id": events[0]["label_group_id"],
                "promotion_channel": "primary",
                "replaces_label_group_ids": [],
                "expected_current_state_sha256": ledger.current_state_hash([]),
                "adjudication_id": adjudication["adjudication_id"],
            }
        ],
        operator_id="pytest",
        promotion_rule_id="imaginary-acceptance-v1",
        review_resolution_id=None,
        root=root,
        promoted_at="2026-07-24T12:02:00Z",
    )
    assert promoted["n_transitions"] == 1
    second = ledger.materialize(root)
    assert second["generation_sha256"] != first["generation_sha256"]
    current = pq.read_table(
        Path(second["path"]) / "current_labels.parquet"
    ).to_pylist()
    assert [(row["label_type"], row["label_group_id"]) for row in current] == [
        ("imaginary_signal", events[0]["label_group_id"])
    ]
    assert sealed["artifact_set_sha256"].startswith("sha256:")


def test_sealed_runs_are_immutable_and_materialization_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    registry_path, assignment, response = _complete_imaginary_run(root)
    ledger.seal_run(
        "imaginary-run",
        root=root,
        registry_path=registry_path,
        sealed_at="2026-07-24T12:00:00Z",
    )
    with pytest.raises(ValueError, match="sealed"):
        workflow.ingest_response(
            "imaginary-run",
            assignment["assignment_id"],
            response,
            root=root,
            registry_path=registry_path,
        )
    first = ledger.materialize(root)
    first_hashes = {
        path.name: ledger.sha256_file(path)
        for path in Path(first["path"]).iterdir()
        if path.is_file()
    }
    second = ledger.materialize(root)
    second_hashes = {
        path.name: ledger.sha256_file(path)
        for path in Path(second["path"]).iterdir()
        if path.is_file()
    }
    assert first["generation_sha256"] == second["generation_sha256"]
    assert first_hashes == second_hashes


def test_failed_materialization_does_not_replace_active_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ledger"
    registry_path, _, _ = _complete_imaginary_run(root)
    ledger.seal_run(
        "imaginary-run",
        root=root,
        registry_path=registry_path,
        sealed_at="2026-07-24T12:00:00Z",
    )
    ledger.materialize(root)
    pointer = root / "materialized" / "current"
    before = pointer.read_bytes()

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated publication failure")

    monkeypatch.setattr(ledger, "write_parquet_stream", fail_write)
    with pytest.raises(RuntimeError, match="simulated"):
        ledger.materialize(root)
    assert pointer.read_bytes() == before


def test_review_records_are_append_only_and_never_silently_resolved(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    item = {
        "item_id": "TEST-R001",
        "category": "test",
        "question": "Resolve this explicitly?",
        "opened_by": "pytest",
        "source_authority": "test",
        "required_before": ["test-completion"],
        "affected_artifacts": ["none"],
        "status": "pending",
    }
    ledger.append_review_item(item, root=root)
    assert "Derived status: `pending`" in ledger.render_review_report(root)
    resolution = {
        "resolution_id": "TEST-RES001",
        "item_id": "TEST-R001",
        "authority": "test",
        "decided_by": "pytest",
        "decision": "approved",
        "resolution_date": "2026-07-24",
        "resolution_date_precision": "date",
        "rationale": "Explicit test resolution.",
    }
    ledger.append_review_resolution(resolution, root=root)
    assert "Derived status: `approved`" in ledger.render_review_report(root)
    with pytest.raises(ValueError, match="already exists"):
        ledger.append_review_resolution(resolution, root=root)
