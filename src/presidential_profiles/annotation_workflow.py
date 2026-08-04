"""Offline provider-neutral terminal workflow for registered label specs.

The execution layer is intentionally absent.  This module creates stable
assignments, renders prompt/schema material, validates returned JSON, appends
immutable response artifacts, and delegates sealing/materialization/decisions
to :mod:`presidential_profiles.annotation_ledger`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import annotation_ledger as ledger
from .prompts import annotation_v1 as av1


PROTOCOL_VERSION = "annotation-terminal-protocol-v1"
JUDGMENT_LABEL_TYPES = {
    "topics",
    "party_attack",
    "enemy_naming",
    "zero_sum",
    "proposal_values",
    "entities",
}


class ResponseValidationError(ValueError):
    """A submitted response is invalid for its locked assignment/schema."""


def _read_subjects(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix in {".json", ".jsonl"}:
        if path.suffix == ".jsonl":
            frame = pd.DataFrame(ledger.read_jsonl(path))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get("rows", value.get("keys"))
            if not isinstance(value, list):
                raise ValueError("subject JSON must be a list or contain rows/keys")
            frame = pd.DataFrame(value)
    else:
        raise ValueError(f"unsupported subject table: {path.suffix}")
    return frame


def _subject_records(
    frame: pd.DataFrame, subject_type: str
) -> list[dict[str, Any]]:
    if subject_type == "paragraph":
        required = {"doc_name", "para_idx", "text"}
        key = ["doc_name", "para_idx"]
    elif subject_type == "speech":
        required = {"doc_name"}
        key = ["doc_name"]
    elif subject_type == "invocation_span":
        required = {"doc_name", "char_start", "char_end"}
        key = ["doc_name", "char_start", "char_end"]
    else:
        raise ValueError(f"unsupported subject_type: {subject_type}")
    if missing := required - set(frame.columns):
        raise ValueError(f"subject table missing columns: {sorted(missing)}")
    if frame.duplicated(key).any():
        raise ValueError(f"subject table contains duplicate keys on {key}")
    rows: list[dict[str, Any]] = []
    for raw in frame.sort_values(key).to_dict("records"):
        record: dict[str, Any] = {}
        for field, value in raw.items():
            is_container = isinstance(value, (dict, list, tuple))
            if value is None or (not is_container and pd.isna(value)):
                record[field] = None
            elif hasattr(value, "item"):
                record[field] = value.item()
            else:
                record[field] = value
        subject = ledger.subject_key(
            subject_type,
            doc_name=record.get("doc_name"),
            para_idx=record.get("para_idx"),
            char_start=record.get("char_start"),
            char_end=record.get("char_end"),
        )
        record["subject_type"] = subject_type
        record["subject_key"] = subject
        record["subject_id"] = ledger.subject_identity(subject_type, subject)
        text = record.get("text")
        if text is None and record.get("transcript") is not None:
            text = record["transcript"]
        if text is None and record.get("opening_paragraphs") is not None:
            text = ledger.canonical_json(
                {
                    "opening_paragraphs": record["opening_paragraphs"],
                    "title": record.get("title"),
                    "year": record.get("year"),
                }
            )
        record["source_text_sha256"] = (
            ledger.sha256_text(str(text)) if text is not None else None
        )
        rows.append(record)
    return rows


def _run_spec_map(specs: Sequence[Mapping[str, Any]]) -> str:
    return ledger.canonical_json(
        {
            spec["label_type"]: {
                "spec_version": spec["spec_version"],
                "spec_sha256": spec["spec_sha256"],
            }
            for spec in sorted(specs, key=lambda value: value["label_type"])
        }
    )


def _bundle_hash(bundle: Mapping[str, Any]) -> str:
    payload = dict(bundle)
    payload.pop("bundle_sha256", None)
    return ledger.sha256_text(ledger.canonical_json(payload))


def _validate_execution_bundle(
    bundle: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    required = {
        "bundle_schema_version",
        "bundle_id",
        "bundle_version",
        "stability",
        "eligibility",
        "subject_type",
        "emitted_label_specs",
        "prompt_version",
        "prompt_text",
        "prompt_sha256",
        "response_schema",
        "response_schema_sha256",
        "response_collection_pointer",
        "response_template_item",
        "context_policy",
        "batching_policy",
        "local_identifier_policy",
        "bundle_validators",
        "typed_projections",
        "evaluation_requirements",
        "publication_gates",
        "limitations",
        "bundle_sha256",
    }
    if set(bundle) != required:
        raise ValueError("bundle field mismatch")
    if bundle["bundle_schema_version"] != "annotation-label-bundle-v1":
        raise ValueError("unsupported bundle schema")
    ledger.validate_spec_part(bundle["bundle_id"], "bundle_id")
    ledger.validate_spec_part(bundle["bundle_version"], "bundle_version")
    if _bundle_hash(bundle) != bundle["bundle_sha256"]:
        raise ValueError("bundle hash drift")
    if ledger.sha256_text(str(bundle["prompt_text"])) != bundle["prompt_sha256"]:
        raise ValueError("bundle prompt hash drift")
    if ledger.sha256_text(ledger.canonical_json(bundle["response_schema"])) != bundle[
        "response_schema_sha256"
    ]:
        raise ValueError("bundle response schema hash drift")
    if bundle["response_collection_pointer"] != "/annotations":
        raise ValueError("unsupported response collection pointer")
    if bundle["local_identifier_policy"] != {
        "field": "local_id",
        "type": "integer",
        "scope": "assignment",
        "trusted_mapping": "local_program",
    }:
        raise ValueError("unsupported bundle local-identifier policy")
    validators = bundle["bundle_validators"]
    if not isinstance(validators, list) or len(validators) != len(set(validators)):
        raise ValueError("bundle validators must be a unique list")
    unknown_validators = set(validators) - {
        "complete_local_id_set",
        "constituency_evidence",
        "registered_invocation_span",
    }
    if unknown_validators:
        raise ValueError(f"unknown bundle validators: {sorted(unknown_validators)}")
    emitted = bundle["emitted_label_specs"]
    if not isinstance(emitted, list) or not emitted:
        raise ValueError("bundle emitted label specs must be non-empty")
    if any(
        not isinstance(row, dict)
        or set(row) != {"label_type", "spec_version", "response_pointer"}
        for row in emitted
    ):
        raise ValueError("bundle emitted label spec field mismatch")
    expected = {(spec["label_type"], spec["spec_version"]) for spec in specs}
    actual = {
        (
            row.get("label_type"),
            row.get("spec_version"),
        )
        for row in emitted
        if isinstance(row, dict)
    }
    if actual != expected or len(actual) != len(emitted):
        raise ValueError("bundle/spec identity drift")
    subject_types = {spec["subject_type"] for spec in specs}
    if subject_types != {bundle["subject_type"]}:
        raise ValueError("bundle/spec subject-type drift")
    for spec in specs:
        if spec.get("spec_schema_version") != "annotation-label-spec-v2":
            continue
        matching = [
            ref
            for ref in spec["execution_binding"]["bundle_refs"]
            if (
                ref["bundle_id"],
                ref["bundle_version"],
                ref["bundle_sha256"],
            )
            == (
                bundle["bundle_id"],
                bundle["bundle_version"],
                bundle["bundle_sha256"],
            )
        ]
        emitted_ref = next(
            row for row in emitted if row["label_type"] == spec["label_type"]
        )
        if len(matching) != 1 or (
            matching[0]["response_pointer"] != emitted_ref["response_pointer"]
        ):
            raise ValueError(f"{spec['label_type']}: bundle binding drift")
    return dict(bundle)


def _validate_bundle_subjects(
    bundle: Mapping[str, Any], subjects: Sequence[Mapping[str, Any]]
) -> None:
    subject_type = bundle["subject_type"]
    for subject in subjects:
        if subject_type == "paragraph":
            if not isinstance(subject.get("text"), str):
                raise ValueError("paragraph bundle subject requires text")
        elif subject_type == "speech":
            opening = subject.get("opening_paragraphs")
            if (
                not isinstance(subject.get("title"), str)
                or not isinstance(subject.get("year"), int)
                or not isinstance(opening, list)
                or not 1 <= len(opening) <= int(
                    bundle["context_policy"]["opening_paragraphs"]
                )
                or any(not isinstance(value, str) for value in opening)
            ):
                raise ValueError(
                    "speech bundle subject requires title, year, and bounded "
                    "opening_paragraphs"
                )
        elif subject_type == "invocation_span":
            if not isinstance(subject.get("text"), str) or not isinstance(
                subject.get("mention"), str
            ):
                raise ValueError(
                    "invocation bundle subject requires mention and exact window text"
                )
            if subject["mention"] not in subject["text"]:
                raise ValueError("registered invocation mention is absent from its window")


def initialize_run(
    *,
    run_id: str,
    label_types: Sequence[str],
    subjects_path: Path,
    model_id: str,
    model_identity_source: str,
    annotator_id: str,
    provider_id: str | None = None,
    agent_id: str | None = None,
    spec_versions: Mapping[str, str] | None = None,
    canonical_corpus_fingerprint: str | None = None,
    selection_artifact_id: str | None = None,
    authorization_review_item_id: str | None = None,
    human_validation_status: str = "not_reviewed",
    independent_check_status: str = "not_checked",
    reasoning_effort: str = "unknown",
    reasoning_effort_provenance: str = "unknown",
    model_identity_receipt: Mapping[str, Any] | None = None,
    execution_bundle: Mapping[str, Any] | None = None,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_PATH,
) -> dict[str, Any]:
    run_id = ledger.validate_run_id(run_id)
    if not label_types or len(label_types) != len(set(label_types)):
        raise ValueError("label_types must be a non-empty unique list")
    ledger.ensure_unsealed(run_id, root)
    directory = ledger.work_dir(run_id, root)
    if directory.exists():
        raise FileExistsError(directory)
    versions = dict(spec_versions or {})
    specs = [
        ledger.resolve_spec(
            label_type, versions.get(label_type), registry_path=registry_path
        )
        for label_type in label_types
    ]
    subject_types = {spec["subject_type"] for spec in specs}
    if len(subject_types) != 1:
        raise ValueError("one terminal run cannot mix subject types")
    subject_type = next(iter(subject_types))
    bundle = (
        _validate_execution_bundle(execution_bundle, specs)
        if execution_bundle is not None
        else None
    )
    if bundle is not None and model_identity_source != "runtime_metadata":
        raise ValueError("bundle runs require an exact runtime model identity")
    subjects = _subject_records(_read_subjects(subjects_path), subject_type)
    if not subjects:
        raise ValueError("run selection is empty")
    if bundle is not None:
        _validate_bundle_subjects(bundle, subjects)
    model_id = str(model_id).strip()
    annotator_id = str(annotator_id).strip()
    if not model_id or not annotator_id:
        raise ValueError("model_id and annotator_id are required")
    if model_identity_source == "runtime_metadata":
        if not model_identity_receipt:
            raise ValueError("runtime_metadata model identity requires a receipt")
        exposed = str(model_identity_receipt.get("model_id", "")).strip()
        if exposed != model_id:
            raise ValueError("runtime model identity receipt does not match model_id")
        if model_identity_receipt.get("specificity") != "exact_runtime_identifier":
            raise ValueError("runtime model identity is not exact enough")
    elif model_identity_source not in {"legacy_manifest", "unknown"}:
        raise ValueError("invalid model_identity_source")

    directory.mkdir(parents=True)
    (directory / "assignments").mkdir()
    (directory / "responses").mkdir()
    ledger.write_jsonl(directory / "subjects.jsonl", subjects)
    if bundle is not None:
        ledger.atomic_write_json(directory / "bundle.json", bundle)
    if model_identity_receipt is not None:
        ledger.atomic_write_json(
            directory / "model-identity-receipt.json", model_identity_receipt
        )
    ingested_at = ledger.utc_now()
    artifacts = [
        {
            "artifact_id": ledger.content_id(
                "art",
                {
                    "path": str(Path(subjects_path)),
                    "subjects_sha256": ledger.sha256_text(
                        "".join(
                            ledger.canonical_json(row) + "\n" for row in subjects
                        )
                    ),
                },
            ),
            "path": str(Path(subjects_path)),
            "role": "selection_subjects",
        }
    ]
    if bundle is not None:
        artifacts.append(
            {
                "artifact_id": ledger.content_id(
                    "art",
                    {
                        "bundle_id": bundle["bundle_id"],
                        "bundle_sha256": bundle["bundle_sha256"],
                        "bundle_version": bundle["bundle_version"],
                    },
                ),
                "path": "bundle.json",
                "role": "label_bundle_snapshot",
            }
        )
    run = {
        "run_id": run_id,
        "run_kind": "terminal_annotation",
        "scope": "selection",
        "label_spec_map_json": _run_spec_map(specs),
        "provider_id": provider_id,
        "model_id": model_id,
        "model_identity_source": model_identity_source,
        "annotator_id": annotator_id,
        "agent_id": agent_id,
        "reasoning_effort": reasoning_effort,
        "reasoning_effort_provenance": reasoning_effort_provenance,
        "reasoning_effort_by_spec_json": None,
        "manifest_date": None,
        "assigned_at": None,
        "labeled_at": None,
        "ingested_at": ingested_at,
        "timestamp_precision": "microsecond",
        "source_corpus_fingerprint": canonical_corpus_fingerprint,
        "canonical_corpus_fingerprint": canonical_corpus_fingerprint,
        "selection_artifact_id": selection_artifact_id,
        "authorization_review_item_id": authorization_review_item_id,
        "human_validation_status": human_validation_status,
        "independent_check_status": independent_check_status,
        "source_manifest_path": None,
        "source_manifest_sha256": None,
        "artifacts_json": ledger.canonical_json(artifacts),
        "limitations_json": ledger.canonical_json(
            [
                "offline terminal annotation; model execution occurred outside the ledger",
                (
                    "user-authorized trusted single pass; not human-validated or "
                    "independently checked"
                    if authorization_review_item_id
                    else "not human-validated unless an adjudication records otherwise"
                ),
            ]
        ),
        "completion_policy": "complete_selection",
        "expected_event_count": None,
        "subject_type": subject_type,
        "n_subjects": len(subjects),
        "registry_path": str(Path(registry_path)),
        "bundle_id": bundle["bundle_id"] if bundle is not None else None,
        "bundle_version": bundle["bundle_version"] if bundle is not None else None,
        "bundle_sha256": bundle["bundle_sha256"] if bundle is not None else None,
    }
    ledger.atomic_write_json(directory / "run.json", run)
    ledger.snapshot_specs(directory, run["label_spec_map_json"], registry_path=registry_path)
    ledger.write_jsonl(directory / "label_events.jsonl", [])
    ledger.write_jsonl(directory / "canonical_label_projection.jsonl", [])
    ledger.write_jsonl(directory / "evaluation_metrics.jsonl", [])
    return run


def _work_run(run_id: str, root: Path) -> dict[str, Any]:
    ledger.ensure_unsealed(run_id, root)
    path = ledger.work_dir(run_id, root) / "run.json"
    if not path.exists():
        raise FileNotFoundError(path)
    run = json.loads(path.read_text(encoding="utf-8"))
    if run["run_id"] != run_id:
        raise ValueError("run identity drift")
    return run


def _specs_for_run(
    run: Mapping[str, Any], registry_path: Path
) -> list[dict[str, Any]]:
    mapping = json.loads(run["label_spec_map_json"])
    specs = []
    for label_type, identity in mapping.items():
        spec = ledger.resolve_spec(
            label_type,
            identity["spec_version"],
            registry_path=registry_path,
        )
        if spec["spec_sha256"] != identity["spec_sha256"]:
            raise ValueError(f"{label_type}: run spec hash drift")
        specs.append(spec)
    return sorted(specs, key=lambda value: value["label_type"])


def _bundle_for_run(
    run: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], root: Path
) -> dict[str, Any] | None:
    if run.get("bundle_id") is None:
        return None
    path = ledger.work_dir(str(run["run_id"]), root) / "bundle.json"
    if not path.exists():
        raise ValueError("bundle run is missing its bundle snapshot")
    bundle = _validate_execution_bundle(
        json.loads(path.read_text(encoding="utf-8")), specs
    )
    identity = (
        bundle["bundle_id"],
        bundle["bundle_version"],
        bundle["bundle_sha256"],
    )
    if identity != (
        run.get("bundle_id"),
        run.get("bundle_version"),
        run.get("bundle_sha256"),
    ):
        raise ValueError("run/bundle identity drift")
    return bundle


def _assignment_paths(run_id: str, root: Path) -> list[Path]:
    return sorted(
        (ledger.work_dir(run_id, root) / "assignments").glob("*.json")
    )


def _completed_assignment_ids(run_id: str, root: Path) -> set[str]:
    return {
        path.stem
        for path in (ledger.work_dir(run_id, root) / "responses").glob("*.json")
    }


def _assigned_subject_ids(run_id: str, root: Path) -> set[str]:
    ids: set[str] = set()
    for path in _assignment_paths(run_id, root):
        value = ledger.read_json(path)
        ids.update(target["subject_id"] for target in value["targets"])
    return ids


def _open_assignment(run_id: str, root: Path) -> dict[str, Any] | None:
    completed = _completed_assignment_ids(run_id, root)
    open_paths = [
        path for path in _assignment_paths(run_id, root) if path.stem not in completed
    ]
    if len(open_paths) > 1:
        raise ValueError("run has more than one open assignment")
    if not open_paths:
        return None
    return ledger.read_json(open_paths[0])


def _paragraph_context(
    all_subjects: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    context_window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        target.get("context_before_json") is not None
        and target.get("context_after_json") is not None
    ):
        before = json.loads(str(target["context_before_json"]))
        after = json.loads(str(target["context_after_json"]))
        for side, rows in [("before", before), ("after", after)]:
            if not isinstance(rows, list) or len(rows) > 2:
                raise ValueError(f"stored context_{side} is not bounded")
            if any(
                row.get("doc_name") != target["subject_key"]["doc_name"]
                or set(row) != {"doc_name", "para_idx", "text"}
                for row in rows
            ):
                raise ValueError(f"stored context_{side} has invalid keys")
        return before[-context_window:] if context_window else [], (
            after[:context_window] if context_window else []
        )
    speech = sorted(
        (
            row
            for row in all_subjects
            if row["subject_key"]["doc_name"] == target["subject_key"]["doc_name"]
        ),
        key=lambda row: int(row["subject_key"]["para_idx"]),
    )
    positions = {row["subject_id"]: index for index, row in enumerate(speech)}
    position = positions[target["subject_id"]]

    def compact(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "doc_name": row["subject_key"]["doc_name"],
                "para_idx": int(row["subject_key"]["para_idx"]),
                "text": str(row["text"]),
            }
            for row in rows
        ]

    return (
        compact(speech[max(0, position - context_window) : position]),
        compact(speech[position + 1 : position + 1 + context_window]),
    )


def build_assignment(
    *,
    run: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    subjects: Sequence[Mapping[str, Any]],
    selected_subjects: Sequence[Mapping[str, Any]],
    context_window: int,
    assigned_at: str,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Purely build one assignment for an explicit, already ordered subject batch.

    Selection, open-assignment checks, and persistence remain the caller's
    responsibility.  Keeping those concerns outside this helper lets a planner
    prebuild every locked batch without changing the assignment ID or rendered
    model input used by :func:`next_assignment`.
    """
    selected = list(selected_subjects)
    if not selected:
        raise ValueError("assignment subject batch must be non-empty")
    targets: list[dict[str, Any]] = []
    for local_id, subject in enumerate(selected, start=1):
        target = dict(subject)
        if bundle is not None:
            target["local_id"] = local_id
        if run["subject_type"] == "paragraph":
            before, after = _paragraph_context(subjects, subject, context_window)
            target["context_before"] = before
            target["context_after"] = after
        targets.append(target)
    label_types = {spec["label_type"] for spec in specs}
    judgment_bundle = bundle is None and label_types == JUDGMENT_LABEL_TYPES
    assignment_body = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run["run_id"],
        "label_spec_map": json.loads(run["label_spec_map_json"]),
        "context_window": context_window,
        "subject_type": run["subject_type"],
        "targets": targets,
        "response_mode": (
            "label_bundle_v1"
            if bundle is not None
            else (
                "paragraph_judgment_bundle"
                if judgment_bundle
                else "per_subject_values"
            )
        ),
    }
    if bundle is not None:
        assignment_body["bundle_id"] = bundle["bundle_id"]
        assignment_body["bundle_version"] = bundle["bundle_version"]
        assignment_body["bundle_sha256"] = bundle["bundle_sha256"]
        assignment_body["prompt_version"] = bundle["prompt_version"]
        assignment_body["prompt"] = bundle["prompt_text"]
        assignment_body["response_schema"] = bundle["response_schema"]
        template_item = dict(bundle["response_template_item"])
        assignment_body["response_template"] = {
            "annotations": [
                {**template_item, "local_id": target["local_id"]}
                for target in targets
            ]
        }
    elif judgment_bundle:
        assignment_body["rubric"] = av1.JUDGMENT_RUBRIC
        assignment_body["instruction"] = av1.JUDGMENT_INSTRUCTION
        assignment_body["response_schema"] = av1.JUDGMENT_SCHEMA
        assignment_body["response_template"] = {
            "annotations": [
                {
                    "para_idx": target["subject_key"]["para_idx"],
                    "topics": [],
                    "party_attack": False,
                    "enemy_naming": False,
                    "zero_sum": False,
                    "proposal_values": "neither",
                    "entities": [],
                }
                for target in targets
            ]
        }
    else:
        assignment_body["specs"] = list(specs)
        assignment_body["response_template"] = {
            "labels": [
                {
                    "subject_key": target["subject_key"],
                    "values": {
                        spec["label_type"]: None
                        for spec in specs
                    },
                }
                for target in targets
            ]
        }
    assignment_id = ledger.content_id("asg", assignment_body)
    return {
        **assignment_body,
        "assignment_id": assignment_id,
        "assignment_input_sha256": ledger.sha256_text(
            ledger.canonical_json(assignment_body)
        ),
        "assigned_at": assigned_at,
    }


def build_planned_assignment(
    run_id: str,
    planned_batch_id: str,
    *,
    persist: bool = True,
    check_unassigned: bool = True,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_PATH,
) -> dict[str, Any]:
    """Build one locked planned batch, optionally persisting it idempotently."""
    run = _work_run(run_id, root)
    specs = _specs_for_run(run, registry_path)
    bundle = _bundle_for_run(run, specs, root)
    if bundle is None:
        raise ValueError("planned assignments require a locked bundle")
    subjects = ledger.read_jsonl(ledger.work_dir(run_id, root) / "subjects.jsonl")
    if not subjects or any(
        row.get("planned_batch_id") is None for row in subjects
    ):
        raise ValueError(
            "planned batches require every subject to belong to a bundle batch"
        )
    selected = [
        row
        for row in subjects
        if row.get("planned_batch_id") == planned_batch_id
    ]
    batch_size = int(bundle["batching_policy"]["targets_per_assignment"])
    if (
        not isinstance(planned_batch_id, str)
        or not planned_batch_id
        or not selected
        or len(selected) > batch_size
    ):
        raise ValueError("invalid planned assignment batch")
    orders = [row.get("planned_batch_order") for row in selected]
    if (
        any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in orders
        )
        or len(set(orders)) != len(orders)
        or sorted(orders) != list(range(1, len(orders) + 1))
    ):
        raise ValueError("invalid planned assignment batch order")
    selected = sorted(
        selected, key=lambda row: int(row["planned_batch_order"])
    )
    assignment = build_assignment(
        run=run,
        specs=specs,
        bundle=bundle,
        subjects=subjects,
        selected_subjects=selected,
        context_window=int(
            bundle["context_policy"].get("neighbors_each_side", 0)
        ),
        assigned_at=ledger.utc_now(),
    )
    if not persist:
        return assignment
    path = (
        ledger.work_dir(run_id, root)
        / "assignments"
        / f"{assignment['assignment_id']}.json"
    )
    if path.exists():
        existing = ledger.read_json(path)
        if {
            key: value for key, value in existing.items() if key != "assigned_at"
        } != {
            key: value for key, value in assignment.items() if key != "assigned_at"
        }:
            raise ValueError("planned assignment identity drift")
        return existing
    if check_unassigned:
        assigned = _assigned_subject_ids(run_id, root)
        if assigned & {row["subject_id"] for row in selected}:
            raise ValueError("planned assignment batch is partially assigned")
    ledger.atomic_write_json(path, assignment)
    return assignment


def next_assignment(
    run_id: str,
    *,
    batch_size: int | None = None,
    context_window: int | None = None,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_PATH,
) -> dict[str, Any]:
    run = _work_run(run_id, root)
    specs = _specs_for_run(run, registry_path)
    bundle = _bundle_for_run(run, specs, root)
    if bundle is not None:
        bundle_batch_size = int(bundle["batching_policy"]["targets_per_assignment"])
        bundle_context_window = int(
            bundle["context_policy"].get("neighbors_each_side", 0)
        )
        if batch_size is not None and batch_size != bundle_batch_size:
            raise ValueError("batch_size conflicts with locked bundle")
        if context_window is not None and context_window != bundle_context_window:
            raise ValueError("context_window conflicts with locked bundle")
        batch_size = bundle_batch_size
        context_window = bundle_context_window
    else:
        batch_size = 5 if batch_size is None else batch_size
        context_window = 1 if context_window is None else context_window
    if not 1 <= batch_size <= 20:
        raise ValueError("batch_size must be between 1 and 20")
    if not 0 <= context_window <= 2:
        raise ValueError("context_window must be between 0 and 2")
    if existing := _open_assignment(run_id, root):
        return existing
    subjects = ledger.read_jsonl(ledger.work_dir(run_id, root) / "subjects.jsonl")
    assigned = _assigned_subject_ids(run_id, root)
    unassigned = [row for row in subjects if row["subject_id"] not in assigned]
    selected = unassigned[:batch_size]
    planned = [
        row for row in subjects if row.get("planned_batch_id") is not None
    ]
    has_planned_batch = bool(planned)
    if planned:
        if bundle is None or len(planned) != len(subjects):
            raise ValueError(
                "planned batches require every subject to belong to a bundle batch"
            )
        first = unassigned[0] if unassigned else None
        if first is not None:
            planned_batch_id = first["planned_batch_id"]
            batch_rows = [
                row
                for row in subjects
                if row.get("planned_batch_id") == planned_batch_id
            ]
            if any(row["subject_id"] in assigned for row in batch_rows):
                raise ValueError("planned assignment batch is partially assigned")
            if (
                not isinstance(planned_batch_id, str)
                or not planned_batch_id
                or len(batch_rows) > batch_size
                or not batch_rows
            ):
                raise ValueError("invalid planned assignment batch")
            orders = [row.get("planned_batch_order") for row in batch_rows]
            if (
                any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1
                    for value in orders
                )
                or len(set(orders)) != len(orders)
                or sorted(orders) != list(range(1, len(orders) + 1))
            ):
                raise ValueError("invalid planned assignment batch order")
            selected = sorted(
                batch_rows, key=lambda row: int(row["planned_batch_order"])
            )
    same_speech = bool(
        bundle is not None
        and bundle["batching_policy"].get("speech_mixing") == "prohibited"
    )
    if selected and not has_planned_batch and (
        (
            {spec["label_type"] for spec in specs} == JUDGMENT_LABEL_TYPES
            and run["subject_type"] == "paragraph"
        )
        or same_speech
    ):
        # The frozen v1 judgment response echoes para_idx but not doc_name
        # because original paid requests were speech-local.  Preserve that
        # schema by never mixing speeches in one assignment.
        first_doc = selected[0]["subject_key"]["doc_name"]
        selected = [
            row
            for row in unassigned
            if row["subject_key"]["doc_name"] == first_doc
        ][:batch_size]
    if not selected:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "status": "complete",
            "remaining": 0,
        }
    assignment = build_assignment(
        run=run,
        specs=specs,
        bundle=bundle,
        subjects=subjects,
        selected_subjects=selected,
        context_window=context_window,
        assigned_at=ledger.utc_now(),
    )
    path = (
        ledger.work_dir(run_id, root)
        / "assignments"
        / f"{assignment['assignment_id']}.json"
    )
    if path.exists():
        raise FileExistsError(path)
    ledger.atomic_write_json(path, assignment)
    return assignment


def render_assignment(assignment: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provider-neutral model-facing view of one assignment.

    Trusted corpus keys remain in the persisted assignment and never enter this
    rendering. The response needs only assignment-local integer identifiers.
    """
    if assignment.get("response_mode") != "label_bundle_v1":
        return dict(assignment)

    def compact_context(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{"text": str(row["text"])} for row in rows]

    targets = []
    for target in assignment["targets"]:
        rendered: dict[str, Any] = {
            "local_id": target["local_id"],
        }
        if assignment["subject_type"] == "paragraph":
            rendered["text"] = str(target["text"])
            if target.get("decade") is not None:
                rendered["decade"] = target["decade"]
        elif assignment["subject_type"] == "speech":
            rendered["title"] = target["title"]
            rendered["year"] = target["year"]
            rendered["opening_paragraphs"] = list(target["opening_paragraphs"])
        elif assignment["subject_type"] == "invocation_span":
            rendered["mention"] = target["mention"]
            rendered["text"] = target["text"]
        targets.append(rendered)
    result = {
        "protocol_version": assignment["protocol_version"],
        "bundle_id": assignment["bundle_id"],
        "bundle_version": assignment["bundle_version"],
        "prompt_version": assignment["prompt_version"],
        "prompt": assignment["prompt"],
        "targets": targets,
        "response_schema": assignment["response_schema"],
        "response_template": assignment["response_template"],
    }
    if assignment["subject_type"] == "paragraph":
        persisted_targets = list(assignment["targets"])
        docs = {
            target["subject_key"]["doc_name"] for target in persisted_targets
        }
        if len(docs) != 1:
            raise ValueError("paragraph bundle rendering cannot mix speeches")
        if int(assignment.get("context_window", 0)) > 0:
            for previous, current in zip(
                persisted_targets, persisted_targets[1:]
            ):
                previous_after = previous.get("context_after", [])
                current_before = current.get("context_before", [])
                if (
                    not previous_after
                    or not current_before
                    or previous_after[0]["doc_name"]
                    != current["subject_key"]["doc_name"]
                    or int(previous_after[0]["para_idx"])
                    != int(current["subject_key"]["para_idx"])
                    or current_before[-1]["doc_name"]
                    != previous["subject_key"]["doc_name"]
                    or int(current_before[-1]["para_idx"])
                    != int(previous["subject_key"]["para_idx"])
                ):
                    raise ValueError(
                        "paragraph bundle targets are not canonically contiguous"
                    )
        result["shared_context"] = {
            "before": compact_context(
                persisted_targets[0].get("context_before", [])
            ),
            "after": compact_context(
                persisted_targets[-1].get("context_after", [])
            ),
        }
    return result


def _validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value does not match const")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is outside enum")
    expected = schema.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }
    if expected is not None and not valid.get(expected, False):
        raise ValueError(f"{path}: expected {expected}, got {type(value).__name__}")
    if expected == "string":
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"{path}: string is shorter than minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValueError(f"{path}: string is longer than maxLength")
    if expected == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if missing := required - set(value):
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(
                f"{path}: extra fields {sorted(set(value) - set(properties))}"
            )
        for key, item in value.items():
            if key in properties:
                _validate_json_schema(item, properties[key], f"{path}.{key}")
    elif expected == "array":
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path}: array is longer than maxItems")
        if schema.get("uniqueItems") and len(
            {ledger.canonical_json(item) for item in value}
        ) != len(value):
            raise ValueError(f"{path}: array items are not unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, f"{path}[{index}]")


def _validate_response_json_schema(
    value: Any, schema: Mapping[str, Any], path: str = "$"
) -> None:
    try:
        _validate_json_schema(value, schema, path)
    except ValueError as exc:
        raise ResponseValidationError(str(exc)) from exc


def _special_validate(
    spec: Mapping[str, Any], value: Any, target: Mapping[str, Any]
) -> None:
    validator = spec.get("special_validator")
    if validator is None:
        return
    if validator != "constituency_evidence":
        raise ValueError(f"unknown special validator: {validator}")
    text = str(target.get("text", ""))
    if not isinstance(value, dict):
        raise ResponseValidationError("constituency value must be an object")
    claims = value.get("claims", [])
    outcome = value.get("outcome")
    if spec.get("spec_schema_version") == "annotation-label-spec-v2":
        if outcome == "claim" and not claims:
            raise ResponseValidationError(
                "constituency claim outcome requires at least one claim"
            )
        if outcome in {"none", "unclear"} and claims:
            raise ResponseValidationError(
                "non-claim constituency outcome requires an empty claim list"
            )
        if outcome == "none" and value.get("unclear_reason"):
            raise ResponseValidationError(
                "none outcome cannot carry an unclear reason"
            )
        if outcome == "unclear" and not value.get("unclear_reason"):
            raise ResponseValidationError(
                "unclear outcome requires an unclear reason"
            )
    seen: set[str] = set()
    for index, claim in enumerate(claims):
        for field in ("group_text", "evidence_span"):
            field_value = str(claim.get(field, ""))
            if not field_value or field_value not in text:
                raise ResponseValidationError(
                    f"constituency claim {index}: {field} is not an exact target substring"
                )
        if str(claim["group_text"]) not in str(claim["evidence_span"]):
            raise ResponseValidationError(
                f"constituency claim {index}: evidence_span does not contain group_text"
            )
        identity = ledger.canonical_json(claim)
        if identity in seen:
            raise ResponseValidationError(
                f"constituency claim {index}: exact duplicate claim"
            )
        seen.add(identity)
        if spec.get("spec_schema_version") == "annotation-label-spec-v2":
            certainty = claim.get("certainty")
            referent = str(claim.get("resolved_referent", ""))
            if certainty == "explicit" and referent:
                raise ResponseValidationError(
                    f"constituency claim {index}: explicit claim has a resolved referent"
                )
            if certainty == "context_resolved" and not referent:
                raise ResponseValidationError(
                    f"constituency claim {index}: context-resolved claim lacks a referent"
                )


def _group_events(
    *,
    run: Mapping[str, Any],
    spec: Mapping[str, Any],
    target: Mapping[str, Any],
    value: Any,
    assignment: Mapping[str, Any],
    source_response_id: str,
    labeled_at: str,
) -> list[dict[str, Any]]:
    _validate_response_json_schema(value, spec["response_schema"])
    _special_validate(spec, value, target)
    extractor = spec["event_extractor"]
    mode = extractor["mode"]
    items = value if mode == "items" else [value]
    if mode == "items" and not isinstance(value, list):
        raise ResponseValidationError(
            f"{spec['label_type']}: items extractor requires a list"
        )
    group_id = ledger.content_id(
        "lgrp",
        {
            "label_type": spec["label_type"],
            "run_id": run["run_id"],
            "source_response_id": source_response_id,
            "spec_sha256": spec["spec_sha256"],
            "subject_id": target["subject_id"],
        },
    )
    if not items:
        event_values = [("empty_group_marker", None, [], 0)]
    else:
        event_values = [
            ("value", index, item, len(items)) for index, item in enumerate(items)
        ]
    events = []
    for role, index, item, group_size in event_values:
        raw = ledger.canonical_json(item)
        raw_hash = ledger.sha256_text(raw)
        declared_kind = (
            "list" if role == "empty_group_marker" else spec["value_kind"]
        )
        event = {
            "label_group_id": group_id,
            "run_id": run["run_id"],
            "label_type": spec["label_type"],
            "spec_version": spec["spec_version"],
            "spec_sha256": spec["spec_sha256"],
            "subject_type": target["subject_type"],
            "subject_id": target["subject_id"],
            "subject_key_json": ledger.canonical_json(target["subject_key"]),
            "doc_name": target["subject_key"].get("doc_name"),
            "para_idx": target["subject_key"].get("para_idx"),
            "char_start": target["subject_key"].get("char_start"),
            "char_end": target["subject_key"].get("char_end"),
            "source_text_sha256": target.get("source_text_sha256"),
            "assignment_input_sha256": assignment["assignment_input_sha256"],
            "event_role": role,
            "item_index": index,
            "group_size": group_size,
            "value_kind": declared_kind,
            "raw_value_json": raw,
            "raw_value_sha256": raw_hash,
            "assignment_id": assignment["assignment_id"],
            "batch_id": assignment["assignment_id"],
            "source_response_id": source_response_id,
            "source_artifact_id": ledger.content_id(
                "art",
                {
                    "assignment_id": assignment["assignment_id"],
                    "source_response_id": source_response_id,
                },
            ),
            "source_table": None,
            "source_key_json": ledger.canonical_json(target["subject_key"]),
            "source_duplicate_ordinal": 0,
            "assigned_at": assignment["assigned_at"],
            "labeled_at": labeled_at,
            "ingested_at": labeled_at,
            "timestamp_precision": "microsecond",
            "provenance_status_json": ledger.canonical_json(
                {
                    "keys": "observed_local",
                    "model_id": run["model_identity_source"],
                    "timestamps": "observed_local",
                    "value": "returned_response",
                }
            ),
        }
        event["label_id"] = ledger.content_id(
            "lbl",
            {
                "event_role": role,
                "item_index": index,
                "label_group_id": group_id,
                "raw_value_sha256": raw_hash,
                "source_duplicate_ordinal": 0,
            },
        )
        events.append(event)
    return events


def _direct_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "label_id": event["label_id"],
        "canonical_subject_id": event["subject_id"],
        "mapping_rule_id": "direct-canonical-annotation-v1",
    }
    return {
        "projection_id": ledger.content_id("proj", identity),
        "label_id": event["label_id"],
        "label_group_id": event["label_group_id"],
        "source_subject_id": event["subject_id"],
        "canonical_subject_id": event["subject_id"],
        "canonical_subject_key_json": event["subject_key_json"],
        "canonical_doc_name": event["doc_name"],
        "canonical_para_idx": event["para_idx"],
        "mapping_status": "direct_canonical_annotation",
        "mapping_rule_id": "direct-canonical-annotation-v1",
        "source_text_sha256": event["source_text_sha256"],
        "canonical_text_sha256": event["source_text_sha256"],
        "source_input_sha256": event["assignment_input_sha256"],
        "canonical_input_sha256": event["assignment_input_sha256"],
        "eligible_for_promotion": True,
        "review_item_id": None,
        "reason": "Terminal annotation was assigned directly against the canonical subject.",
    }


def _response_pointer_value(
    value: Mapping[str, Any],
    pointer: str,
    *,
    response_derived: bool = False,
) -> Any:
    if not pointer.startswith("/") or pointer.count("/") != 1:
        raise ValueError(f"unsupported response pointer: {pointer!r}")
    field = pointer[1:].replace("~1", "/").replace("~0", "~")
    if field not in value:
        error = ResponseValidationError if response_derived else ValueError
        raise error(f"response pointer is missing: {pointer}")
    return value[field]


def _prepare_response_artifacts(
    *,
    run: Mapping[str, Any],
    assignment: Mapping[str, Any],
    response: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any] | None,
    existing_events: Sequence[Mapping[str, Any]],
    existing_projections: Sequence[Mapping[str, Any]],
    labeled_at: str,
) -> dict[str, Any]:
    """Purely validate a response and construct every artifact ingest will write.

    Integrity failures in the locked run, assignment, specs, or bundle remain
    ordinary :class:`ValueError` instances.  Only invalid submitted response
    content raises :class:`ResponseValidationError`.
    """
    run_id = str(run["run_id"])
    assignment_id = str(assignment["assignment_id"])
    if assignment.get("run_id") != run_id:
        raise ValueError("run/assignment identity drift")
    run_spec_map = json.loads(str(run["label_spec_map_json"]))
    if assignment.get("label_spec_map") != run_spec_map:
        raise ValueError("run/assignment spec identity drift")
    supplied_spec_map = {
        spec["label_type"]: {
            "spec_version": spec["spec_version"],
            "spec_sha256": spec["spec_sha256"],
        }
        for spec in specs
    }
    if supplied_spec_map != run_spec_map:
        raise ValueError("run/spec identity drift")
    targets = {target["subject_id"]: target for target in assignment["targets"]}
    specs_by_type = {spec["label_type"]: spec for spec in specs}
    extracted: list[tuple[dict[str, Any], dict[str, Any]]] = []

    if assignment["response_mode"] == "label_bundle_v1":
        if bundle is None:
            raise ValueError("bundle response mode has no locked bundle")
        if (
            assignment.get("bundle_id"),
            assignment.get("bundle_version"),
            assignment.get("bundle_sha256"),
        ) != (
            bundle["bundle_id"],
            bundle["bundle_version"],
            bundle["bundle_sha256"],
        ):
            raise ValueError("assignment/bundle identity drift")
        emitted_by_type = {
            row["label_type"]: row for row in bundle["emitted_label_specs"]
        }
        if set(emitted_by_type) != set(specs_by_type):
            raise ValueError("bundle emitted label-type set drift")
        _validate_response_json_schema(response, bundle["response_schema"])
        annotations = _response_pointer_value(
            response,
            bundle["response_collection_pointer"],
            response_derived=True,
        )
        by_local_id = {
            int(target["local_id"]): target for target in assignment["targets"]
        }
        local_ids = [int(row["local_id"]) for row in annotations]
        if len(local_ids) != len(set(local_ids)) or set(local_ids) != set(
            by_local_id
        ):
            raise ResponseValidationError(
                "incomplete, duplicate, or unassigned bundle response"
            )
        for annotation in annotations:
            target = by_local_id[int(annotation["local_id"])]
            values = {
                label_type: _response_pointer_value(
                    annotation,
                    emitted_by_type[label_type]["response_pointer"],
                    response_derived=True,
                )
                for label_type in sorted(specs_by_type)
            }
            extracted.append((target, values))
    elif assignment["response_mode"] == "paragraph_judgment_bundle":
        _validate_response_json_schema(response, av1.JUDGMENT_SCHEMA)
        by_para = {
            int(target["subject_key"]["para_idx"]): target
            for target in assignment["targets"]
        }
        annotations = response["annotations"]
        indices = [int(row["para_idx"]) for row in annotations]
        if len(indices) != len(set(indices)) or set(indices) != set(by_para):
            raise ResponseValidationError(
                "incomplete, duplicate, or unassigned paragraph response"
            )
        for annotation in annotations:
            target = by_para[int(annotation["para_idx"])]
            values = {
                key: annotation[key]
                for key in JUDGMENT_LABEL_TYPES
            }
            extracted.append((target, values))
    elif assignment["response_mode"] == "per_subject_values":
        if (
            not isinstance(response, Mapping)
            or set(response) != {"labels"}
            or not isinstance(response["labels"], list)
        ):
            raise ResponseValidationError(
                "generic response must contain only a labels list"
            )
        seen: set[str] = set()
        for raw in response["labels"]:
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {"subject_key", "values"}
                or not isinstance(raw["subject_key"], Mapping)
                or not isinstance(raw["values"], Mapping)
            ):
                raise ResponseValidationError("generic label field mismatch")
            subject_type = run["subject_type"]
            try:
                key = dict(raw["subject_key"])
                subject_id = ledger.subject_identity(subject_type, key)
            except (TypeError, ValueError) as exc:
                raise ResponseValidationError(str(exc)) from exc
            if subject_id in seen:
                raise ResponseValidationError("duplicate response subject")
            seen.add(subject_id)
            if subject_id not in targets:
                raise ResponseValidationError("response subject was not assigned")
            if set(raw["values"]) != set(specs_by_type):
                raise ResponseValidationError(
                    "response label-type set is incomplete or unknown"
                )
            extracted.append((targets[subject_id], dict(raw["values"])))
        if seen != set(targets):
            raise ResponseValidationError(
                "response subject key set is incomplete"
            )
    else:
        raise ValueError("unknown assignment response mode")

    new_events: list[dict[str, Any]] = []
    for target, values in extracted:
        source_response_id = ledger.content_id(
            "resp",
            {
                "assignment_id": assignment_id,
                "run_id": run_id,
                "subject_id": target["subject_id"],
            },
        )
        for label_type, value in sorted(values.items()):
            new_events.extend(
                _group_events(
                    run=run,
                    spec=specs_by_type[label_type],
                    target=target,
                    value=value,
                    assignment=assignment,
                    source_response_id=source_response_id,
                    labeled_at=labeled_at,
                )
            )
    events_before = [dict(event) for event in existing_events]
    ids = {event["label_id"] for event in events_before}
    if ids & {event["label_id"] for event in new_events}:
        raise ValueError("response would duplicate an existing label event")
    projections = [_direct_projection(event) for event in new_events]
    saved_response = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "assignment_id": assignment_id,
        "assignment_input_sha256": assignment["assignment_input_sha256"],
        "labeled_at": labeled_at,
        "response": response,
        "label_ids": [event["label_id"] for event in new_events],
    }
    result = {
        "status": "ingested",
        "run_id": run_id,
        "assignment_id": assignment_id,
        "n_events": len(new_events),
    }
    return {
        "new_events": new_events,
        "new_projections": projections,
        "events": new_events,
        "projections": projections,
        "all_events": events_before + new_events,
        "all_projections": [
            *(dict(projection) for projection in existing_projections),
            *projections,
        ],
        "saved_response": saved_response,
        "result": result,
    }


def prepare_response_artifacts(
    run_id: str,
    assignment_id: str,
    response: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_PATH,
    labeled_at: str | None = None,
) -> dict[str, Any]:
    """Read locked inputs and purely prepare, but do not persist, a response."""
    run = _work_run(run_id, root)
    assignment_path = (
        ledger.work_dir(run_id, root)
        / "assignments"
        / f"{assignment_id}.json"
    )
    if not assignment_path.exists():
        raise FileNotFoundError(f"unassigned response: {assignment_id}")
    response_path = (
        ledger.work_dir(run_id, root) / "responses" / f"{assignment_id}.json"
    )
    if response_path.exists():
        raise ValueError("assignment response already exists")
    assignment = ledger.read_json(assignment_path)
    if assignment["assignment_id"] != assignment_id:
        raise ValueError("assignment identity drift")
    specs = _specs_for_run(run, registry_path)
    bundle = _bundle_for_run(run, specs, root)
    work = ledger.work_dir(run_id, root)
    return _prepare_response_artifacts(
        run=run,
        assignment=assignment,
        response=response,
        specs=specs,
        bundle=bundle,
        existing_events=ledger.read_jsonl(work / "label_events.jsonl"),
        existing_projections=ledger.read_jsonl(
            work / "canonical_label_projection.jsonl"
        ),
        labeled_at=labeled_at or ledger.utc_now(),
    )


def ingest_response(
    run_id: str,
    assignment_id: str,
    response: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_PATH,
) -> dict[str, Any]:
    run = _work_run(run_id, root)
    assignment_path = (
        ledger.work_dir(run_id, root)
        / "assignments"
        / f"{assignment_id}.json"
    )
    if not assignment_path.exists():
        raise FileNotFoundError(f"unassigned response: {assignment_id}")
    response_path = (
        ledger.work_dir(run_id, root) / "responses" / f"{assignment_id}.json"
    )
    if response_path.exists():
        raise ValueError("assignment response already exists")
    assignment = ledger.read_json(assignment_path)
    if assignment["assignment_id"] != assignment_id:
        raise ValueError("assignment identity drift")
    specs = _specs_for_run(run, registry_path)
    bundle = _bundle_for_run(run, specs, root)
    work = ledger.work_dir(run_id, root)
    prepared = _prepare_response_artifacts(
        run=run,
        assignment=assignment,
        response=response,
        specs=specs,
        bundle=bundle,
        existing_events=ledger.read_jsonl(work / "label_events.jsonl"),
        existing_projections=ledger.read_jsonl(
            work / "canonical_label_projection.jsonl"
        ),
        labeled_at=ledger.utc_now(),
    )
    # Validate all response-derived state before the first write.
    ledger.write_jsonl(
        work / "label_events.jsonl", prepared["all_events"]
    )
    ledger.write_jsonl(
        work / "canonical_label_projection.jsonl",
        prepared["all_projections"],
    )
    ledger.atomic_write_json(response_path, prepared["saved_response"])
    return prepared["result"]


def status_run(
    run_id: str, *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    run = _work_run(run_id, root)
    subjects = ledger.read_jsonl(ledger.work_dir(run_id, root) / "subjects.jsonl")
    assigned = _assigned_subject_ids(run_id, root)
    completed_assignment_ids = _completed_assignment_ids(run_id, root)
    completed_subjects: set[str] = set()
    for path in _assignment_paths(run_id, root):
        if path.stem in completed_assignment_ids:
            assignment = ledger.read_json(path)
            completed_subjects.update(target["subject_id"] for target in assignment["targets"])
    return {
        "run_id": run_id,
        "n_subjects": len(subjects),
        "assigned_subjects": len(assigned),
        "completed_subjects": len(completed_subjects),
        "remaining": len(subjects) - len(assigned),
        "open_assignments": len(_assignment_paths(run_id, root))
        - len(completed_assignment_ids),
        "n_events": len(
            ledger.read_jsonl(ledger.work_dir(run_id, root) / "label_events.jsonl")
        ),
    }


def audit_run(
    run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_PATH,
) -> dict[str, Any]:
    result = ledger.audit_work_run(
        run_id, root=root, registry_path=registry_path
    )
    if result["status"] != "ok":
        return result
    status = status_run(run_id, root=root)
    if status["completed_subjects"] != status["n_subjects"]:
        return {
            **result,
            "status": "failed",
            "issues": [
                f"incomplete selection: {status['completed_subjects']}/{status['n_subjects']}"
            ],
        }
    return {**result, **status}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-run")
    init.add_argument("--run-id", required=True)
    init.add_argument("--label-type", action="append", required=True)
    init.add_argument("--subjects", type=Path, required=True)
    init.add_argument("--model-id", required=True)
    init.add_argument("--model-identity-source", required=True)
    init.add_argument("--annotator-id", required=True)
    init.add_argument("--provider-id")
    init.add_argument("--agent-id")
    init.add_argument(
        "--model-identity-receipt",
        type=Path,
        help="JSON runtime-metadata receipt; required for runtime_metadata",
    )
    init.add_argument("--canonical-corpus-fingerprint")
    init.add_argument("--selection-artifact-id")
    init.add_argument("--authorization-review-item-id")
    nxt = sub.add_parser("next")
    nxt.add_argument("--run-id", required=True)
    nxt.add_argument("--batch-size", type=int, default=5)
    nxt.add_argument("--context-window", type=int, default=1)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--run-id", required=True)
    ingest.add_argument("--assignment-id", required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-id", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--run-id", required=True)
    seal = sub.add_parser("seal-run")
    seal.add_argument("--run-id", required=True)
    sub.add_parser("materialize")
    compare = sub.add_parser("compare")
    compare.add_argument("--left-run-id", required=True)
    compare.add_argument("--right-run-id", required=True)
    sub.add_parser(
        "adjudicate",
        help="read one adjudication decision as JSON from stdin",
    )
    promote = sub.add_parser(
        "promote",
        help="read a promotion batch as JSON from stdin",
    )
    promote.add_argument("--operator-id", required=True)
    promote.add_argument("--promotion-rule-id", required=True)
    promote.add_argument("--review-resolution-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-run":
        identity_receipt = (
            json.loads(args.model_identity_receipt.read_text(encoding="utf-8"))
            if args.model_identity_receipt
            else None
        )
        result = initialize_run(
            run_id=args.run_id,
            label_types=args.label_type,
            subjects_path=args.subjects,
            model_id=args.model_id,
            model_identity_source=args.model_identity_source,
            annotator_id=args.annotator_id,
            provider_id=args.provider_id,
            agent_id=args.agent_id,
            canonical_corpus_fingerprint=args.canonical_corpus_fingerprint,
            selection_artifact_id=args.selection_artifact_id,
            authorization_review_item_id=args.authorization_review_item_id,
            model_identity_receipt=identity_receipt,
        )
    elif args.command == "next":
        result = next_assignment(
            args.run_id,
            batch_size=args.batch_size,
            context_window=args.context_window,
        )
    elif args.command == "ingest":
        result = ingest_response(
            args.run_id, args.assignment_id, json.load(sys.stdin)
        )
    elif args.command == "status":
        result = status_run(args.run_id)
    elif args.command == "audit":
        result = audit_run(args.run_id)
    elif args.command == "seal-run":
        result = ledger.seal_run(args.run_id)
    elif args.command == "materialize":
        result = ledger.materialize()
    elif args.command == "compare":
        result = ledger.compare_runs(args.left_run_id, args.right_run_id)
    elif args.command == "adjudicate":
        result = ledger.publish_adjudication(json.load(sys.stdin))
    elif args.command == "promote":
        transitions = json.load(sys.stdin)
        if not isinstance(transitions, list):
            raise ValueError("promotion input must be a JSON list of transitions")
        result = ledger.publish_promotions(
            transitions,
            operator_id=args.operator_id,
            promotion_rule_id=args.promotion_rule_id,
            review_resolution_id=args.review_resolution_id,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
