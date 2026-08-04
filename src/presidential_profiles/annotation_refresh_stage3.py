"""Deterministic, provider-neutral ARCV1 Stage 3 publication APIs.

This module operates only on sealed ledger artifacts.  It publishes
content-addressed comparisons outside child run roots so child immutability never
prevents later comparison, adjudication, or evaluation.
"""

from __future__ import annotations

import json
import hashlib
import itertools
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import annotation_ledger as ledger
from . import annotation_refresh as refresh
from . import annotation_workflow as workflow


COMPARISONS_ROOT = ledger.LEDGER_ROOT / "campaign_comparisons"
ADJUDICATION_RUNS_ROOT = ledger.LEDGER_ROOT / "adjudication_runs"
EVALUATIONS_ROOT = ledger.LEDGER_ROOT / "campaign_evaluations"
FREEZES_ROOT = ledger.LEDGER_ROOT / "specification_freezes"
CAMPAIGN_OUTCOMES_ROOT = ledger.LEDGER_ROOT / "campaign_outcomes"
COMPARISON_VERSION = "annotation-refresh-comparison-v1"
ADJUDICATION_MANIFEST_VERSION = "annotation-refresh-adjudication-run-v1"
EVALUATION_VERSION = "annotation-refresh-deterministic-evaluation-v1"
FREEZE_VERSION = "annotation-refresh-specification-freeze-v1"
CAMPAIGN_OUTCOME_VERSION = "annotation-refresh-campaign-outcome-v1"
INVOCATION_AUDIT_CONTRACT_VERSION = (
    "annotation-refresh-invocation-audit-contract-v1"
)
EVALUATION_SEED = "annotation-refresh-campaign-v1|evaluation|20260724"
BOOTSTRAP_DRAWS = 2_000
ITEM_LABEL_TYPES = {"topics", "entities"}
EXISTING_LABEL_TYPES = (
    "topics",
    "party_attack",
    "enemy_naming",
    "zero_sum",
    "proposal_values",
    "entities",
)
BOOLEAN_LABEL_TYPES = ("party_attack", "enemy_naming", "zero_sum")
FINAL_SPEC_VERSIONS = {
    "topics": "paragraph-judgment-v2",
    "party_attack": "paragraph-judgment-v2",
    "enemy_naming": "paragraph-judgment-v2",
    "zero_sum": "paragraph-judgment-v2",
    "proposal_values": "paragraph-judgment-v2",
    "entities": "paragraph-judgment-v2",
    "constituencies": "v1",
}


def _sealed_source(run_id: str, root: Path) -> dict[str, Any]:
    seal = ledger.verify_sealed_run(run_id, root=root)
    return {
        "run_id": run_id,
        "artifact_set_sha256": seal["artifact_set_sha256"],
    }


def _run_groups(
    run_id: str,
    *,
    root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    directory = ledger.sealed_artifact_dir(run_id, root)
    projections = {
        row["label_id"]: row
        for row in ledger.read_jsonl(
            directory / "canonical_label_projection.jsonl"
        )
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in ledger.read_jsonl(directory / "label_events.jsonl"):
        projection = projections[event["label_id"]]
        subject_id = projection["canonical_subject_id"]
        if subject_id is None:
            continue
        grouped.setdefault((subject_id, event["label_type"]), []).append(event)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        group_ids = {row["label_group_id"] for row in rows}
        spec_hashes = {row["spec_sha256"] for row in rows}
        if len(group_ids) != 1 or len(spec_hashes) != 1:
            raise ValueError(f"{run_id}: non-atomic comparison group at {key}")
        ordered = sorted(
            rows,
            key=lambda row: (
                row["item_index"] is None,
                row["item_index"] if row["item_index"] is not None else -1,
                row["label_id"],
            ),
        )
        if key[1] in ITEM_LABEL_TYPES:
            value = (
                []
                if ordered[0]["event_role"] == "empty_group_marker"
                else [json.loads(row["raw_value_json"]) for row in ordered]
            )
        else:
            if len(ordered) != 1 or ordered[0]["event_role"] != "value":
                raise ValueError(f"{run_id}: invalid scalar comparison group at {key}")
            value = json.loads(ordered[0]["raw_value_json"])
        value_json = ledger.canonical_json(value)
        result[key] = {
            "canonical_subject_id": key[0],
            "label_type": key[1],
            "label_group_id": next(iter(group_ids)),
            "spec_sha256": next(iter(spec_hashes)),
            "value_json": value_json,
            "value_sha256": ledger.sha256_text(value_json),
        }
    return result


def build_comparison(
    *,
    campaign_id: str,
    comparison_role: str,
    left_run_id: str,
    right_run_id: str,
    required_overlap: int | None = None,
    allowed_label_types: Sequence[str] | None = None,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Build one exact sealed-run comparison without exposing values."""
    campaign = refresh.load_campaign(campaign_id, root=root)
    child_ids = {row["run_id"] for row in campaign["child_runs"]}
    known_reference_ids = {
        child.name
        for child in (Path(root) / "reference_runs").iterdir()
        if child.is_dir()
    } if (Path(root) / "reference_runs").exists() else set()
    if left_run_id not in child_ids | known_reference_ids:
        raise ValueError("left run is not part of the campaign evidence")
    if right_run_id not in child_ids | known_reference_ids:
        raise ValueError("right run is not part of the campaign evidence")
    left_source = _sealed_source(left_run_id, Path(root))
    right_source = _sealed_source(right_run_id, Path(root))
    left = _run_groups(left_run_id, root=Path(root))
    right = _run_groups(right_run_id, root=Path(root))
    allowed = set(allowed_label_types) if allowed_label_types else None
    overlap = sorted(
        key
        for key in set(left) & set(right)
        if allowed is None or key[1] in allowed
    )
    overlap_subjects = {key[0] for key in overlap}
    if required_overlap is not None and len(overlap_subjects) != required_overlap:
        raise ValueError(
            f"comparison overlap is {len(overlap_subjects)}, expected {required_overlap}"
        )
    disagreements = [
        {
            "canonical_subject_id": key[0],
            "label_type": key[1],
            "left_label_group_id": left[key]["label_group_id"],
            "right_label_group_id": right[key]["label_group_id"],
            "left_spec_sha256": left[key]["spec_sha256"],
            "right_spec_sha256": right[key]["spec_sha256"],
            "left_value_sha256": left[key]["value_sha256"],
            "right_value_sha256": right[key]["value_sha256"],
        }
        for key in overlap
        if left[key]["value_json"] != right[key]["value_json"]
    ]
    semantic = {
        "comparison_version": COMPARISON_VERSION,
        "campaign_id": campaign_id,
        "comparison_role": comparison_role,
        "left_source": left_source,
        "right_source": right_source,
        "allowed_label_types": sorted(allowed) if allowed is not None else None,
        "overlap_subjects": len(overlap_subjects),
        "overlap_subject_label_pairs": len(overlap),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }
    comparison_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    return {
        **semantic,
        "comparison_id": "cmp_" + comparison_sha.removeprefix("sha256:"),
        "comparison_sha256": comparison_sha,
    }


def publish_comparison(
    comparison: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    value = dict(comparison)
    comparison_id = str(value.pop("comparison_id", ""))
    declared_sha = str(value.pop("comparison_sha256", ""))
    observed_sha = ledger.sha256_text(ledger.canonical_json(value))
    if (
        comparison_id != "cmp_" + declared_sha.removeprefix("sha256:")
        or declared_sha != observed_sha
    ):
        raise ValueError("comparison identity drift")
    persisted = {
        **value,
        "comparison_id": comparison_id,
        "comparison_sha256": declared_sha,
    }
    path = Path(root) / "campaign_comparisons" / f"{comparison_id}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != persisted:
            raise ValueError("existing comparison drift")
        status = "already_published"
    else:
        ledger.atomic_write_json(path, persisted)
        status = "published"
    return {
        "status": status,
        "comparison_id": comparison_id,
        "comparison_sha256": declared_sha,
        "path": str(path),
        "overlap_subjects": persisted["overlap_subjects"],
        "overlap_subject_label_pairs": persisted[
            "overlap_subject_label_pairs"
        ],
        "n_disagreements": persisted["n_disagreements"],
    }


def _campaign_role_runs(
    campaign_id: str,
    *,
    root: Path,
) -> dict[str, str]:
    campaign = refresh.load_campaign(campaign_id, root=root)
    roles = {
        str(row["pass_role"]): str(row["run_id"])
        for row in campaign["child_runs"]
    }
    required = {"composition_diagnostic", "blind_a", "blind_b"}
    if set(roles) != required:
        raise ValueError("campaign child-role set drift")
    return roles


def required_reference_disputes(
    *,
    reference_run_id: str,
    blind_a_run_id: str,
    blind_b_run_id: str,
    root: Path = ledger.LEDGER_ROOT,
) -> list[dict[str, Any]]:
    """Return hashed identities for every required reference adjudication."""
    reference = _run_groups(reference_run_id, root=Path(root))
    blind_a = _run_groups(blind_a_run_id, root=Path(root))
    blind_b = _run_groups(blind_b_run_id, root=Path(root))
    expected_keys = set(reference)
    if expected_keys - set(blind_a) or expected_keys - set(blind_b):
        raise ValueError("candidate run does not cover the complete reference")
    disputes: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        ref_value = reference[key]["value_json"]
        values_differ = (
            ref_value != blind_a[key]["value_json"]
            or ref_value != blind_b[key]["value_json"]
            or blind_a[key]["value_json"] != blind_b[key]["value_json"]
        )
        unclear = (
            key[1] == "constituencies"
            and json.loads(ref_value)["outcome"] == "unclear"
        )
        if not values_differ and not unclear:
            continue
        spec_hashes = {
            reference[key]["spec_sha256"],
            blind_a[key]["spec_sha256"],
            blind_b[key]["spec_sha256"],
        }
        if len(spec_hashes) != 1:
            raise ValueError(f"reference dispute spec drift at {key}")
        candidates = [reference[key], blind_a[key], blind_b[key]]
        disputes.append(
            {
                "canonical_subject_id": key[0],
                "label_type": key[1],
                "spec_sha256": next(iter(spec_hashes)),
                "reason": (
                    "blind_reference_unclear"
                    if unclear and not values_differ
                    else (
                        "disagreement_and_blind_reference_unclear"
                        if unclear
                        else "reference_or_blind_pass_disagreement"
                    )
                ),
                "candidate_label_group_ids": sorted(
                    {row["label_group_id"] for row in candidates}
                ),
                "candidate_value_sha256": sorted(
                    {row["value_sha256"] for row in candidates}
                ),
            }
        )
    return disputes


def _validate_adjudication_exposure(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "blind_reference_sealed_before_candidate_exposure": True,
        "candidate_sources_limited_to_campaign_blind_passes": True,
        "historical_and_current_labels_hidden": True,
        "review_scope_limited_to_required_reference_disputes": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("adjudication exposure receipt is incomplete")
    if not str(receipt.get("source", "")).strip():
        raise ValueError("adjudication exposure receipt requires a source")
    return dict(receipt)


def initialize_reference_adjudication(
    *,
    campaign_id: str,
    reference_run_id: str,
    model_receipt: Mapping[str, Any],
    adjudicator_id: str,
    exposure_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    bundle_registry_path: Path = refresh.BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    """Initialize a post-reference run on the exact disputed-subject subset."""
    root = Path(root)
    roles = _campaign_role_runs(campaign_id, root=root)
    sources = {
        "reference": _sealed_source(reference_run_id, root),
        "blind_a": _sealed_source(roles["blind_a"], root),
        "blind_b": _sealed_source(roles["blind_b"], root),
    }
    disputes = required_reference_disputes(
        reference_run_id=reference_run_id,
        blind_a_run_id=roles["blind_a"],
        blind_b_run_id=roles["blind_b"],
        root=root,
    )
    if not disputes:
        raise ValueError("adjudication run is unnecessary: no required disputes")
    reference = refresh.load_independent_reference(
        reference_run_id, root=root
    )
    reference_selection_path = (
        root / "selections" / f"{reference['selection_id']}.json"
    )
    selection = json.loads(
        reference_selection_path.read_text(encoding="utf-8")
    )
    if selection.get("selection_sha256") != reference["selection_sha256"]:
        raise ValueError("reference selection drift before adjudication")
    receipt = refresh.validate_model_receipt(
        model_receipt, required_effort="high"
    )
    adjudicator_id = str(adjudicator_id).strip()
    if not adjudicator_id:
        raise ValueError("adjudicator_id is required")
    exposure = _validate_adjudication_exposure(exposure_receipt)
    dispute_scope_sha = ledger.sha256_text(ledger.canonical_json(disputes))
    exposure_sha = ledger.sha256_text(ledger.canonical_json(exposure))
    disputed_subject_ids = {
        row["canonical_subject_id"] for row in disputes
    }
    identity_rows = [
        row
        for row in selection["identity_rows"]
        if ledger.subject_identity("paragraph", row["subject_key"])
        in disputed_subject_ids
    ]
    adjudication_rows = []
    for source_row in selection["rows"]:
        subject_key = ledger.subject_key(
            "paragraph",
            doc_name=source_row["doc_name"],
            para_idx=source_row["para_idx"],
        )
        if (
            ledger.subject_identity("paragraph", subject_key)
            not in disputed_subject_ids
        ):
            continue
        row = dict(source_row)
        row.pop("planned_batch_id", None)
        row.pop("planned_batch_order", None)
        adjudication_rows.append(row)
    if (
        len(identity_rows) != len(disputed_subject_ids)
        or len(adjudication_rows) != len(disputed_subject_ids)
    ):
        raise ValueError("adjudication selection does not cover dispute subjects")
    sorted_rows = sorted(
        adjudication_rows,
        key=lambda row: (row["doc_name"], int(row["para_idx"])),
    )
    blocks: list[list[dict[str, Any]]] = []
    for row in sorted_rows:
        if (
            not blocks
            or blocks[-1][-1]["doc_name"] != row["doc_name"]
            or int(blocks[-1][-1]["para_idx"]) + 1
            != int(row["para_idx"])
            or len(blocks[-1]) == 4
        ):
            blocks.append([])
        blocks[-1].append(row)
    planned_rows: list[dict[str, Any]] = []
    for block in blocks:
        keys = [
            {
                "doc_name": row["doc_name"],
                "para_idx": int(row["para_idx"]),
            }
            for row in block
        ]
        batch_id = ledger.content_id(
            "batch",
            {
                "dispute_scope_sha256": dispute_scope_sha,
                "target_keys": keys,
            },
        )
        for order, row in enumerate(block, start=1):
            planned_rows.append(
                {
                    **row,
                    "planned_batch_id": batch_id,
                    "planned_batch_order": order,
                }
            )
    selection_semantic = {
        "selection_manifest_version": (
            "annotation-refresh-adjudication-selection-v1"
        ),
        "selection_role": "required_reference_disputes",
        "campaign_id": campaign_id,
        "reference_selection_id": reference["selection_id"],
        "reference_selection_sha256": reference["selection_sha256"],
        "dispute_scope_sha256": dispute_scope_sha,
        "n_subjects": len(disputed_subject_ids),
        "n_assignments": len(blocks),
        "identity_rows": sorted(
            identity_rows,
            key=lambda row: ledger.canonical_json(row["subject_key"]),
        ),
        "rows": sorted(
            planned_rows,
            key=lambda row: (row["doc_name"], int(row["para_idx"])),
        ),
    }
    adjudication_selection_sha = ledger.sha256_text(
        ledger.canonical_json(selection_semantic)
    )
    adjudication_selection = {
        **selection_semantic,
        "selection_id": refresh.subject_selection_id(identity_rows),
        "selection_sha256": adjudication_selection_sha,
    }
    selection_path = (
        root
        / "adjudication_selections"
        / f"{adjudication_selection['selection_id']}.json"
    )
    if selection_path.exists():
        if (
            json.loads(selection_path.read_text(encoding="utf-8"))
            != adjudication_selection
        ):
            raise ValueError("existing adjudication selection drift")
    else:
        ledger.atomic_write_json(selection_path, adjudication_selection)
    semantic = {
        "adjudication_manifest_version": ADJUDICATION_MANIFEST_VERSION,
        "campaign_id": campaign_id,
        "reference_run_id": reference_run_id,
        "sources": sources,
        "selection_id": adjudication_selection["selection_id"],
        "selection_sha256": adjudication_selection[
            "selection_sha256"
        ],
        "reference_selection_id": reference["selection_id"],
        "reference_selection_sha256": reference["selection_sha256"],
        "dispute_scope_sha256": dispute_scope_sha,
        "exposure_receipt_sha256": exposure_sha,
        "n_disputes": len(disputes),
        "n_disputed_subjects": len(
            {row["canonical_subject_id"] for row in disputes}
        ),
        "model_id": receipt["model_id"],
        "reasoning_effort": receipt["reasoning_effort"],
        "adjudicator_id": adjudicator_id,
        "resolution_policy": (
            "fresh post-reference synthesis; original blind reference remains "
            "immutable; current and historical labels remain hidden"
        ),
    }
    identity_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    run_id = "adjudication-" + identity_sha.removeprefix("sha256:")
    manifest = {
        **semantic,
        "adjudication_run_id": run_id,
        "adjudication_manifest_sha256": identity_sha,
        "disputes": disputes,
        "exposure_receipt": exposure,
    }
    manifest_path = root / "adjudication_runs" / run_id / "adjudication.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing adjudication manifest drift")
        if run_id in ledger.sealed_runs(root):
            ledger.verify_sealed_run(run_id, root=root)
            return {
                "status": "already_sealed",
                "adjudication_run_id": run_id,
                "n_disputes": len(disputes),
                "n_disputed_subjects": semantic["n_disputed_subjects"],
            }
        if not ledger.work_dir(run_id, root).exists():
            raise ValueError("adjudication manifest exists without work run")
        return {
            "status": "already_initialized",
            "adjudication_run_id": run_id,
            "n_disputes": len(disputes),
            "n_disputed_subjects": semantic["n_disputed_subjects"],
        }
    refresh.initialize_bundle_run(
        bundle_id="paragraph_judgment_v2_candidate",
        bundle_version="candidate-1",
        run_id=run_id,
        subjects_path=selection_path,
        model_receipt=receipt,
        reasoning_effort="high",
        annotator_id=adjudicator_id,
        root=root,
        bundle_registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
        canonical_corpus_fingerprint=refresh.load_campaign(
            campaign_id, root=root
        )["corpus_snapshot"]["fingerprint"],
        selection_artifact_id=adjudication_selection["selection_id"],
        authorization_review_item_id="ARCV1-D007",
    )
    ledger.atomic_write_json(manifest_path, manifest)
    ledger.atomic_write_json(
        ledger.work_dir(run_id, root)
        / "adjudication-initial-runtime-receipt.json",
        receipt,
    )
    ledger.atomic_write_json(
        ledger.work_dir(run_id, root) / "adjudication-exposure-receipt.json",
        exposure,
    )
    return {
        "status": "initialized_no_assignments",
        "adjudication_run_id": run_id,
        "n_disputes": len(disputes),
        "n_disputed_subjects": semantic["n_disputed_subjects"],
    }


def load_reference_adjudication(
    adjudication_run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    path = (
        Path(root)
        / "adjudication_runs"
        / adjudication_run_id
        / "adjudication.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("adjudication_run_id") != adjudication_run_id:
        raise ValueError("adjudication run identity drift")
    semantic = {
        key: value[key]
        for key in (
            "adjudication_manifest_version",
            "campaign_id",
            "reference_run_id",
            "sources",
            "selection_id",
            "selection_sha256",
            "reference_selection_id",
            "reference_selection_sha256",
            "dispute_scope_sha256",
            "exposure_receipt_sha256",
            "n_disputes",
            "n_disputed_subjects",
            "model_id",
            "reasoning_effort",
            "adjudicator_id",
            "resolution_policy",
        )
    }
    observed = ledger.sha256_text(ledger.canonical_json(semantic))
    if value.get("adjudication_manifest_sha256") != observed:
        raise ValueError("adjudication manifest hash drift")
    if adjudication_run_id != "adjudication-" + observed.removeprefix(
        "sha256:"
    ):
        raise ValueError("adjudication run content identity drift")
    if value.get("dispute_scope_sha256") != ledger.sha256_text(
        ledger.canonical_json(value.get("disputes", []))
    ):
        raise ValueError("adjudication dispute scope drift")
    exposure = _validate_adjudication_exposure(
        value.get("exposure_receipt", {})
    )
    if value.get("exposure_receipt_sha256") != ledger.sha256_text(
        ledger.canonical_json(exposure)
    ):
        raise ValueError("adjudication exposure receipt drift")
    return value


def next_adjudication_assignment(
    adjudication_run_id: str,
    *,
    runtime_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    manifest = load_reference_adjudication(
        adjudication_run_id, root=root
    )
    receipt = refresh.validate_model_receipt(
        runtime_receipt, required_effort="high"
    )
    if receipt["model_id"] != manifest["model_id"]:
        raise ValueError("adjudication runtime model identity drift")
    session = {
        "adjudication_run_id": adjudication_run_id,
        "runtime_receipt": receipt,
    }
    receipt_id = ledger.content_id("rcpt", session)
    path = (
        ledger.work_dir(adjudication_run_id, root)
        / "adjudication-session-receipts"
        / f"{receipt_id}.json"
    )
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != session:
            raise ValueError("adjudication session receipt identity drift")
    else:
        ledger.atomic_write_json(path, session)
    return workflow.next_assignment(
        adjudication_run_id, root=root, registry_path=registry_path
    )


def render_adjudication_assignment(
    assignment: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Expose only declared reference/candidate values for disputed fields."""
    run_id = str(assignment.get("run_id", ""))
    manifest = load_reference_adjudication(run_id, root=root)
    if assignment.get("status") == "complete":
        return dict(assignment)
    sources = {
        role: _run_groups(source["run_id"], root=Path(root))
        for role, source in manifest["sources"].items()
    }
    disputes_by_subject: dict[str, list[str]] = {}
    for dispute in manifest["disputes"]:
        disputes_by_subject.setdefault(
            dispute["canonical_subject_id"], []
        ).append(dispute["label_type"])
    candidate_values = []
    for target in assignment["targets"]:
        subject_id = target["subject_id"]
        fields = sorted(disputes_by_subject.get(subject_id, []))
        if not fields:
            raise ValueError("adjudication assignment contains no disputed field")
        candidate_values.append(
            {
                "local_id": target["local_id"],
                "disputed_fields": {
                    label_type: {
                        role: _value(groups, subject_id, label_type)
                        for role, groups in sources.items()
                    }
                    for label_type in fields
                },
            }
        )
    return {
        **workflow.render_assignment(assignment),
        "adjudication_instruction": (
            "Resolve only the listed disputed fields using the target text, "
            "bounded context, sealed blind reference, and two sealed blind "
            "candidate passes. Produce the normal complete schema response. "
            "Do not inspect current, legacy, Opus, or historical labels."
        ),
        "sealed_disputed_values": candidate_values,
    }


def audit_reference_adjudication(
    adjudication_run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    manifest = load_reference_adjudication(
        adjudication_run_id, root=root
    )
    audit = workflow.audit_run(
        adjudication_run_id, root=root, registry_path=registry_path
    )
    if audit["status"] != "ok":
        return audit
    groups = _run_groups_unsealed(adjudication_run_id, root=Path(root))
    missing = []
    unresolved_unclear = []
    for dispute in manifest["disputes"]:
        key = (
            dispute["canonical_subject_id"],
            dispute["label_type"],
        )
        row = groups.get(key)
        if row is None:
            missing.append(key)
        elif (
            key[1] == "constituencies"
            and json.loads(row["value_json"])["outcome"] == "unclear"
        ):
            unresolved_unclear.append(key)
    issues = list(audit.get("issues", []))
    if missing:
        issues.append("adjudication output is missing required dispute groups")
    if unresolved_unclear:
        issues.append("adjudication left required constituency rows unclear")
    return {
        **audit,
        "status": "failed" if issues else "ok",
        "issues": issues,
        "required_disputes": len(manifest["disputes"]),
        "missing_required_disputes": len(missing),
        "unresolved_constituency_disputes": len(unresolved_unclear),
    }


def seal_reference_adjudication(
    adjudication_run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    if adjudication_run_id in ledger.sealed_runs(root):
        manifest = load_reference_adjudication(
            adjudication_run_id, root=root
        )
        seal = ledger.verify_sealed_run(adjudication_run_id, root=root)
        return {
            "status": "already_sealed",
            "run_id": adjudication_run_id,
            "artifact_set_sha256": seal["artifact_set_sha256"],
            "required_disputes": len(manifest["disputes"]),
        }
    audit = audit_reference_adjudication(
        adjudication_run_id, root=root, registry_path=registry_path
    )
    if audit["status"] != "ok":
        raise ValueError("reference adjudication audit failed")
    result = ledger.seal_run(
        adjudication_run_id, root=root, registry_path=registry_path
    )
    ledger.verify_sealed_run(adjudication_run_id, root=root)
    return {
        **result,
        "required_disputes": audit["required_disputes"],
    }


def _run_groups_unsealed(
    run_id: str,
    *,
    root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read work groups using the same canonical reconstruction as sealed runs."""
    directory = ledger.work_dir(run_id, root)
    projections = {
        row["label_id"]: row
        for row in ledger.read_jsonl(
            directory / "canonical_label_projection.jsonl"
        )
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in ledger.read_jsonl(directory / "label_events.jsonl"):
        projection = projections[event["label_id"]]
        subject_id = projection["canonical_subject_id"]
        if subject_id is not None:
            grouped.setdefault(
                (subject_id, event["label_type"]), []
            ).append(event)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        group_ids = {row["label_group_id"] for row in rows}
        spec_hashes = {row["spec_sha256"] for row in rows}
        if len(group_ids) != 1 or len(spec_hashes) != 1:
            raise ValueError(f"{run_id}: non-atomic work group at {key}")
        ordered = sorted(
            rows,
            key=lambda row: (
                row["item_index"] is None,
                row["item_index"] if row["item_index"] is not None else -1,
                row["label_id"],
            ),
        )
        if key[1] in ITEM_LABEL_TYPES:
            value = (
                []
                if ordered[0]["event_role"] == "empty_group_marker"
                else [
                    json.loads(row["raw_value_json"]) for row in ordered
                ]
            )
        else:
            if len(ordered) != 1 or ordered[0]["event_role"] != "value":
                raise ValueError(f"{run_id}: invalid scalar work group at {key}")
            value = json.loads(ordered[0]["raw_value_json"])
        value_json = ledger.canonical_json(value)
        result[key] = {
            "canonical_subject_id": key[0],
            "label_type": key[1],
            "label_group_id": next(iter(group_ids)),
            "spec_sha256": next(iter(spec_hashes)),
            "value_json": value_json,
            "value_sha256": ledger.sha256_text(value_json),
        }
    return result


def publish_required_adjudications(
    adjudication_run_id: str,
    *,
    review_resolution_id: str = "ARCV1-D007",
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Publish one append-only decision for every declared reference dispute."""
    root = Path(root)
    manifest = load_reference_adjudication(
        adjudication_run_id, root=root
    )
    _sealed_source(adjudication_run_id, root)
    resolved = _run_groups(adjudication_run_id, root=root)
    existing = {
        row["adjudication_id"]: row
        for row in ledger._read_decisions(  # type: ignore[attr-defined]
            "adjudications", "label_adjudications.jsonl", root
        )
    }
    validated_group_metadata = ledger._adjudication_group_metadata(  # type: ignore[attr-defined]
        root
    )
    published: list[str] = []
    already_published: list[str] = []
    for dispute in manifest["disputes"]:
        key = (
            dispute["canonical_subject_id"],
            dispute["label_type"],
        )
        resolved_row = resolved[key]
        if resolved_row["spec_sha256"] != dispute["spec_sha256"]:
            raise ValueError(f"resolved adjudication spec drift at {key}")
        decision = {
            "canonical_subject_id": key[0],
            "label_type": key[1],
            "spec_sha256": dispute["spec_sha256"],
            "candidate_label_group_ids": dispute[
                "candidate_label_group_ids"
            ],
            "resolution_kind": "synthesize_in_adjudication_run",
            "selected_label_group_id": None,
            "resolved_label_group_id": resolved_row["label_group_id"],
            "adjudicator_id": manifest["adjudicator_id"],
            "rationale": (
                "Fresh post-reference adjudication of the declared blind "
                "reference/candidate dispute."
            ),
            "review_resolution_id": review_resolution_id,
            "decided_at": ledger.utc_now(),
        }
        identity_payload = dict(decision)
        identity_payload["candidate_label_group_ids_json"] = (
            ledger.canonical_json(
                sorted(identity_payload.pop("candidate_label_group_ids"))
            )
        )
        expected_id = ledger.content_id(
            "adj",
            {
                key_name: value
                for key_name, value in identity_payload.items()
                if key_name != "decided_at"
            },
        )
        if expected_id in existing:
            persisted = dict(existing[expected_id])
            persisted.pop("artifact_set_sha256", None)
            persisted.pop("adjudication_id", None)
            if {
                key_name: value
                for key_name, value in persisted.items()
                if key_name != "decided_at"
            } != {
                key_name: value
                for key_name, value in identity_payload.items()
                if key_name != "decided_at"
            }:
                raise ValueError("existing adjudication decision drift")
            already_published.append(expected_id)
            continue
        result = ledger.publish_adjudication(
            decision,
            root=root,
            _validated_group_metadata=validated_group_metadata,
        )
        if result["adjudication_id"] != expected_id:
            raise ValueError("published adjudication identity drift")
        published.append(expected_id)
    return {
        "status": "complete",
        "adjudication_run_id": adjudication_run_id,
        "required_disputes": len(manifest["disputes"]),
        "published": len(published),
        "already_published": len(already_published),
        "adjudication_ids": sorted(published + already_published),
    }


def _current_groups(
    *,
    root: Path,
    subject_ids: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    generation_id = (Path(root) / "materialized" / "current").read_text(
        encoding="utf-8"
    ).strip()
    path = (
        Path(root)
        / "materialized"
        / "generations"
        / generation_id
        / "current_labels.parquet"
    )
    frame = pd.read_parquet(path)
    frame = frame[
        (frame["promotion_channel"] == "primary")
        & frame["canonical_subject_id"].isin(subject_ids)
    ]
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in frame.groupby(
        ["canonical_subject_id", "label_type"], sort=True
    ):
        group_ids = set(group["label_group_id"])
        if len(group_ids) != 1:
            raise ValueError(f"current labels are non-atomic at {key}")
        rows = group.sort_values(
            ["item_index", "label_id"], na_position="last"
        ).to_dict("records")
        label_type = str(key[1])
        if label_type in ITEM_LABEL_TYPES:
            if len(rows) == 1 and rows[0]["value_kind"] == "list":
                value = json.loads(rows[0]["raw_value_json"])
            else:
                value = [json.loads(row["raw_value_json"]) for row in rows]
        else:
            if len(rows) != 1:
                raise ValueError(f"current scalar labels are non-atomic at {key}")
            value = json.loads(rows[0]["raw_value_json"])
        value_json = ledger.canonical_json(value)
        result[(str(key[0]), label_type)] = {
            "canonical_subject_id": str(key[0]),
            "label_type": label_type,
            "label_group_id": next(iter(group_ids)),
            "spec_sha256": str(rows[0]["spec_sha256"]),
            "value_json": value_json,
            "value_sha256": ledger.sha256_text(value_json),
        }
    return result


def _value(
    groups: Mapping[tuple[str, str], Mapping[str, Any]],
    subject_id: str,
    label_type: str,
) -> Any:
    row = groups.get((subject_id, label_type))
    if row is None and label_type == "entities":
        return []
    if row is None:
        raise KeyError((subject_id, label_type))
    return json.loads(str(row["value_json"]))


def _safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _binary_stats(gold: Sequence[bool], pred: Sequence[bool]) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("binary metric length mismatch")
    tp = sum(bool(g) and bool(p) for g, p in zip(gold, pred))
    tn = sum(not bool(g) and not bool(p) for g, p in zip(gold, pred))
    fp = sum(not bool(g) and bool(p) for g, p in zip(gold, pred))
    fn = sum(bool(g) and not bool(p) for g, p in zip(gold, pred))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    f1 = _safe_div(2 * tp, f1_denominator)
    specificity = _safe_div(tn, tn + fp)
    balanced = (
        None
        if recall is None or specificity is None
        else (recall + specificity) / 2
    )
    accuracy = _safe_div(tp + tn, len(gold))
    false_omission = _safe_div(fn, fn + tn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
        "claim_miss_rate": _safe_div(fn, tp + fn),
        "predicted_none_false_omission_rate": false_omission,
        "support_positive": tp + fn,
        "n": len(gold),
    }


def _kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if not left or len(left) != len(right):
        return None
    n = len(left)
    categories = set(left) | set(right)
    observed = sum(a == b for a, b in zip(left, right)) / n
    expected = sum(
        (sum(value == category for value in left) / n)
        * (sum(value == category for value in right) / n)
        for category in categories
    )
    return None if expected == 1.0 else float((observed - expected) / (1 - expected))


def _set_stats(
    gold: Sequence[set[Any]], pred: Sequence[set[Any]]
) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("set metric length mismatch")
    tp = sum(len(a & b) for a, b in zip(gold, pred))
    fp = sum(len(b - a) for a, b in zip(gold, pred))
    fn = sum(len(a - b) for a, b in zip(gold, pred))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    f1 = _safe_div(2 * tp, f1_denominator)
    jaccards = [
        1.0 if not (a | b) else len(a & b) / len(a | b)
        for a, b in zip(gold, pred)
    ]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else None,
        "exact_agreement": (
            float(np.mean([a == b for a, b in zip(gold, pred)]))
            if gold
            else None
        ),
        "support_positive": sum(len(value) for value in gold),
        "n": len(gold),
    }


def _macro_f1(gold: Sequence[Any], pred: Sequence[Any]) -> dict[str, Any]:
    classes = sorted(set(gold) | set(pred), key=str)
    per_class: dict[str, dict[str, Any]] = {}
    values = []
    for category in classes:
        stats = _binary_stats(
            [value == category for value in gold],
            [value == category for value in pred],
        )
        per_class[str(category)] = stats
        if stats["support_positive"] >= 20 and stats["f1"] is not None:
            values.append(stats["f1"])
    return {
        "accuracy": (
            float(np.mean([a == b for a, b in zip(gold, pred)]))
            if gold
            else None
        ),
        "macro_f1_supported": float(np.mean(values)) if values else None,
        "per_class": per_class,
        "n": len(gold),
    }


def _claim_tuple(claim: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(claim[field])
        for field in (
            "group_text",
            "resolved_referent",
            "group_type",
            "relation",
            "stance",
            "evidence_span",
            "certainty",
        )
    )


def _entity_tuple(entity: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(entity["name"]),
        str(entity["type"]),
        str(entity["stance"]),
    )


def _partial_entity_tuple(entity: Mapping[str, Any]) -> tuple[str, ...]:
    return (str(entity["name"]), str(entity["type"]))


def _span_strings_overlap(left: Any, right: Any) -> bool:
    left_text = " ".join(str(left).casefold().split())
    right_text = " ".join(str(right).casefold().split())
    return bool(
        left_text
        and right_text
        and (left_text in right_text or right_text in left_text)
    )


def _partial_claim_stats(
    gold: Sequence[Mapping[str, Any]],
    pred: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matched = 0
    stance_matches = 0
    certainty_matches = 0
    gold_total = 0
    pred_total = 0
    for gold_row, pred_row in zip(gold, pred):
        gold_claims = list(gold_row["claims"])
        pred_claims = list(pred_row["claims"])
        gold_total += len(gold_claims)
        pred_total += len(pred_claims)
        if not gold_claims or not pred_claims:
            continue
        scores: dict[tuple[int, int], int] = {}
        for gold_index, gold_claim in enumerate(gold_claims):
            for pred_index, pred_claim in enumerate(pred_claims):
                if (
                    gold_claim["group_type"] != pred_claim["group_type"]
                    or gold_claim["relation"] != pred_claim["relation"]
                    or not _span_strings_overlap(
                        gold_claim["group_text"],
                        pred_claim["group_text"],
                    )
                    or not _span_strings_overlap(
                        gold_claim["evidence_span"],
                        pred_claim["evidence_span"],
                    )
                ):
                    continue
                scores[(gold_index, pred_index)] = (
                    int(gold_claim["stance"] == pred_claim["stance"])
                    + int(
                        gold_claim["certainty"] == pred_claim["certainty"]
                    )
                )

        @lru_cache(maxsize=None)
        def solve(
            gold_index: int, used_mask: int
        ) -> tuple[int, int, tuple[tuple[int, int], ...]]:
            if gold_index == len(gold_claims):
                return (0, 0, ())
            best = solve(gold_index + 1, used_mask)
            for pred_index in range(len(pred_claims)):
                if used_mask & (1 << pred_index):
                    continue
                score = scores.get((gold_index, pred_index))
                if score is None:
                    continue
                tail = solve(
                    gold_index + 1, used_mask | (1 << pred_index)
                )
                candidate = (
                    tail[0] + 1,
                    tail[1] + score,
                    ((gold_index, pred_index), *tail[2]),
                )
                if (
                    candidate[:2] > best[:2]
                    or (
                        candidate[:2] == best[:2]
                        and candidate[2] < best[2]
                    )
                ):
                    best = candidate
            return best

        _, _, pairs = solve(0, 0)
        for gold_index, pred_index in pairs:
            matched += 1
            stance_matches += int(
                gold_claims[gold_index]["stance"]
                == pred_claims[pred_index]["stance"]
            )
            certainty_matches += int(
                gold_claims[gold_index]["certainty"]
                == pred_claims[pred_index]["certainty"]
            )
    return {
        "matched_claims": matched,
        "gold_claims": gold_total,
        "predicted_claims": pred_total,
        "micro_precision": _safe_div(matched, pred_total),
        "micro_recall": _safe_div(matched, gold_total),
        "micro_f1": _safe_div(
            2 * matched, gold_total + pred_total
        ),
        "stance_accuracy_on_matched": _safe_div(
            stance_matches, matched
        ),
        "certainty_accuracy_on_matched": _safe_div(
            certainty_matches, matched
        ),
        "span_overlap_rule": (
            "casefolded_whitespace_normalized_string_containment"
        ),
        "n": len(gold),
    }


def _constituency_detection_stats(
    gold: Sequence[Mapping[str, Any]],
    pred: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold_claim = [row["outcome"] == "claim" for row in gold]
    pred_claim = [row["outcome"] == "claim" for row in pred]
    binary = _binary_stats(gold_claim, pred_claim)
    miss_numerator = sum(
        gold_row["outcome"] == "claim" and pred_row["outcome"] == "none"
        for gold_row, pred_row in zip(gold, pred)
    )
    gold_positive = sum(gold_claim)
    predicted_none = sum(row["outcome"] == "none" for row in pred)
    binary["claim_miss_numerator"] = miss_numerator
    binary["claim_miss_denominator"] = gold_positive
    binary["claim_miss_rate"] = _safe_div(miss_numerator, gold_positive)
    binary["predicted_none_false_omission_numerator"] = miss_numerator
    binary["predicted_none_false_omission_denominator"] = predicted_none
    binary["predicted_none_false_omission_rate"] = _safe_div(
        miss_numerator, predicted_none
    )
    return binary


def _constituency_category_stats(
    gold: Sequence[Mapping[str, Any]],
    pred: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    category_gold = [
        Counter(
            f"{field}:{claim[field]}"
            for claim in row["claims"]
            for field in ("group_type", "relation", "stance")
        )
        for row in gold
    ]
    category_pred = [
        Counter(
            f"{field}:{claim[field]}"
            for claim in row["claims"]
            for field in ("group_type", "relation", "stance")
        )
        for row in pred
    ]
    categories = sorted(
        set().union(*category_gold, *category_pred)
        if category_gold or category_pred
        else set()
    )
    per_category: dict[str, dict[str, Any]] = {}
    supported_f1: list[float] = []
    for category in categories:
        tp = sum(
            min(left[category], right[category])
            for left, right in zip(category_gold, category_pred)
        )
        gold_count = sum(values[category] for values in category_gold)
        pred_count = sum(values[category] for values in category_pred)
        fp = pred_count - tp
        fn = gold_count - tp
        stats = {
            "tp": tp,
            "tn": None,
            "fp": fp,
            "fn": fn,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
            "specificity": None,
            "balanced_accuracy": None,
            "accuracy": None,
            "claim_miss_rate": _safe_div(fn, tp + fn),
            "predicted_none_false_omission_rate": None,
            "support_positive": gold_count,
            "n": len(gold),
        }
        per_category[category] = stats
        if stats["support_positive"] >= 20 and stats["f1"] is not None:
            supported_f1.append(stats["f1"])
    return {
        "supported_category_macro_f1": (
            float(np.mean(supported_f1)) if supported_f1 else None
        ),
        "per_category": per_category,
    }


def _constituency_stats(
    gold: Sequence[Mapping[str, Any]],
    pred: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("constituency metric length mismatch")
    binary = _constituency_detection_stats(gold, pred)
    exact_sets = _set_stats(
        [{_claim_tuple(claim) for claim in row["claims"]} for row in gold],
        [{_claim_tuple(claim) for claim in row["claims"]} for row in pred],
    )
    categories = _constituency_category_stats(gold, pred)
    return {
        "claim_detection": binary,
        "claim_exact": exact_sets,
        "claim_partial": _partial_claim_stats(gold, pred),
        "outcome_accuracy": (
            float(
                np.mean(
                    [
                        left["outcome"] == right["outcome"]
                        for left, right in zip(gold, pred)
                    ]
                )
            )
            if gold
            else None
        ),
        "outcome_kappa": _kappa(
            [row["outcome"] for row in gold],
            [row["outcome"] for row in pred],
        ),
        **categories,
        "n": len(gold),
    }


def _topic_set(value: Any) -> set[Any]:
    """Normalize logical topic lists without changing sealed artifact values."""
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    return set(value)


def _label_stats(
    label_type: str,
    gold: Sequence[Any],
    pred: Sequence[Any],
) -> dict[str, Any]:
    if label_type in BOOLEAN_LABEL_TYPES:
        return _binary_stats(gold, pred)
    if label_type == "proposal_values":
        return _macro_f1(gold, pred)
    if label_type == "topics":
        # Combined V2 run artifacts preserve the response field as one
        # list-valued event, so sealed-run reconstruction yields ``[[...]]``.
        # Materialized Phase 1 topics are stored directly as ``[...]``.
        # Normalize only at the metric boundary: the sealed event/value hashes
        # remain untouched and previously published comparisons stay
        # reproducible.
        gold_sets = [_topic_set(value) for value in gold]
        pred_sets = [_topic_set(value) for value in pred]
        base = _set_stats(
            gold_sets,
            pred_sets,
        )
        classes = sorted(
            set().union(*gold_sets, *pred_sets)
            if gold_sets or pred_sets
            else set()
        )
        base["per_category"] = {
            str(category): _binary_stats(
                [category in value for value in gold_sets],
                [category in value for value in pred_sets],
            )
            for category in classes
        }
        return base
    if label_type == "entities":
        exact = _set_stats(
            [{_entity_tuple(entity) for entity in value} for value in gold],
            [{_entity_tuple(entity) for entity in value} for value in pred],
        )
        partial = _set_stats(
            [
                {_partial_entity_tuple(entity) for entity in value}
                for value in gold
            ],
            [
                {_partial_entity_tuple(entity) for entity in value}
                for value in pred
            ],
        )
        entity_types = sorted(
            {
                str(entity["type"])
                for values in list(gold) + list(pred)
                for entity in values
            }
        )
        return {
            "exact": exact,
            "partial_name_and_type": partial,
            "per_category": {
                entity_type: _binary_stats(
                    [
                        any(
                            str(entity["type"]) == entity_type
                            for entity in values
                        )
                        for values in gold
                    ],
                    [
                        any(
                            str(entity["type"]) == entity_type
                            for entity in values
                        )
                        for values in pred
                    ],
                )
                for entity_type in entity_types
            },
            "n": len(gold),
        }
    if label_type == "constituencies":
        return _constituency_stats(gold, pred)
    raise ValueError(f"unsupported evaluation label type: {label_type}")


def _bootstrap_indices(
    subject_ids: Sequence[str],
    doc_by_subject: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[list[int]]:
    clusters = sorted({doc_by_subject[subject_id] for subject_id in subject_ids})
    positions: dict[str, list[int]] = {
        cluster: [
            index
            for index, subject_id in enumerate(subject_ids)
            if doc_by_subject[subject_id] == cluster
        ]
        for cluster in clusters
    }
    seed = int.from_bytes(
        hashlib.sha256(EVALUATION_SEED.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    result: list[list[int]] = []
    for _ in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        result.append(
            list(
                itertools.chain.from_iterable(
                    positions[str(cluster)] for cluster in sampled
                )
            )
        )
    return result


def _interval(values: Sequence[float | None]) -> dict[str, Any]:
    valid = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if len(valid) == 0:
        return {
            "status": "not_estimable",
            "n_valid_draws": 0,
            "two_sided_95": None,
            "one_sided_95_lower": None,
            "one_sided_95_upper": None,
        }
    return {
        "status": "ok",
        "n_valid_draws": int(len(valid)),
        "two_sided_95": [
            float(np.quantile(valid, 0.025)),
            float(np.quantile(valid, 0.975)),
        ],
        "one_sided_95_lower": float(np.quantile(valid, 0.05)),
        "one_sided_95_upper": float(np.quantile(valid, 0.95)),
    }


def _nested_metric(value: Mapping[str, Any], path: str) -> float | None:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    if current is None:
        return None
    return float(current)


def _resampled_metric(
    *,
    label_type: str,
    gold: Sequence[Any],
    pred: Sequence[Any],
    indices: Sequence[Sequence[int]],
    metric_path: str,
) -> list[float | None]:
    result: list[float | None] = []
    for draw in indices:
        draw_gold = [gold[index] for index in draw]
        draw_pred = [pred[index] for index in draw]
        if label_type == "constituencies" and metric_path.startswith(
            "claim_detection."
        ):
            stats: Mapping[str, Any] = {
                "claim_detection": _constituency_detection_stats(
                    draw_gold, draw_pred
                )
            }
        elif label_type == "constituencies" and metric_path.startswith(
            "claim_exact."
        ):
            stats = {
                "claim_exact": _set_stats(
                    [
                        {_claim_tuple(claim) for claim in row["claims"]}
                        for row in draw_gold
                    ],
                    [
                        {_claim_tuple(claim) for claim in row["claims"]}
                        for row in draw_pred
                    ],
                )
            }
        elif label_type == "constituencies" and metric_path in {
            "outcome_accuracy",
            "outcome_kappa",
        }:
            stats = {
                "outcome_accuracy": (
                    float(
                        np.mean(
                            [
                                left["outcome"] == right["outcome"]
                                for left, right in zip(
                                    draw_gold, draw_pred
                                )
                            ]
                        )
                    )
                    if draw_gold
                    else None
                ),
                "outcome_kappa": _kappa(
                    [row["outcome"] for row in draw_gold],
                    [row["outcome"] for row in draw_pred],
                ),
            }
        elif (
            label_type == "constituencies"
            and metric_path == "supported_category_macro_f1"
        ):
            stats = _constituency_category_stats(
                draw_gold, draw_pred
            )
        elif label_type in BOOLEAN_LABEL_TYPES:
            stats = _binary_stats(draw_gold, draw_pred)
        elif label_type == "proposal_values":
            stats = _macro_f1(draw_gold, draw_pred)
        elif label_type == "topics":
            stats = _set_stats(
                [_topic_set(value) for value in draw_gold],
                [_topic_set(value) for value in draw_pred],
            )
        elif label_type == "entities":
            if metric_path.startswith("exact."):
                stats = {
                    "exact": _set_stats(
                        [
                            {_entity_tuple(entity) for entity in value}
                            for value in draw_gold
                        ],
                        [
                            {_entity_tuple(entity) for entity in value}
                            for value in draw_pred
                        ],
                    )
                }
            elif metric_path.startswith("partial_name_and_type."):
                stats = {
                    "partial_name_and_type": _set_stats(
                        [
                            {
                                _partial_entity_tuple(entity)
                                for entity in value
                            }
                            for value in draw_gold
                        ],
                        [
                            {
                                _partial_entity_tuple(entity)
                                for entity in value
                            }
                            for value in draw_pred
                        ],
                    )
                }
            else:
                raise ValueError(
                    f"unsupported entity bootstrap metric: {metric_path}"
                )
        else:
            stats = _label_stats(label_type, draw_gold, draw_pred)
        result.append(_nested_metric(stats, metric_path))
    return result


def _paired_difference(
    left: Sequence[float | None],
    right: Sequence[float | None],
) -> list[float | None]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap draw count mismatch")
    return [
        None if a is None or b is None else float(a - b)
        for a, b in zip(left, right)
    ]


def _metric_components(
    stats: Mapping[str, Any],
    metric_path: str,
) -> tuple[float | None, float | None, int]:
    """Return auditable numerator, denominator, and positive support."""
    parent: Mapping[str, Any] = stats
    parts = metric_path.split(".")
    for part in parts[:-1]:
        value = parent.get(part)
        if not isinstance(value, Mapping):
            return None, None, int(stats.get("n", 0))
        parent = value
    metric = parts[-1]
    support = int(parent.get("support_positive", stats.get("n", 0)))
    if metric in {"precision", "micro_precision"}:
        return (
            float(parent.get("tp", 0)),
            float(parent.get("tp", 0) + parent.get("fp", 0)),
            support,
        )
    if metric in {"recall", "micro_recall"}:
        return (
            float(parent.get("tp", 0)),
            float(parent.get("tp", 0) + parent.get("fn", 0)),
            support,
        )
    if metric in {"f1", "micro_f1"}:
        tp = float(parent.get("tp", 0))
        fp = float(parent.get("fp", 0))
        fn = float(parent.get("fn", 0))
        return 2 * tp, 2 * tp + fp + fn, support
    if metric in {"accuracy", "outcome_accuracy", "exact_agreement"}:
        n = int(parent.get("n", stats.get("n", 0)))
        value = _nested_metric(stats, metric_path)
        return (
            None if value is None else value * n,
            float(n),
            support,
        )
    if metric == "balanced_accuracy":
        return None, None, support
    if metric == "claim_miss_rate":
        return (
            float(parent.get("claim_miss_numerator", parent.get("fn", 0))),
            float(
                parent.get(
                    "claim_miss_denominator",
                    parent.get("tp", 0) + parent.get("fn", 0),
                )
            ),
            support,
        )
    if metric == "predicted_none_false_omission_rate":
        return (
            float(
                parent.get(
                    "predicted_none_false_omission_numerator",
                    parent.get("fn", 0),
                )
            ),
            float(
                parent.get(
                    "predicted_none_false_omission_denominator",
                    parent.get("fn", 0) + parent.get("tn", 0),
                )
            ),
            support,
        )
    if metric == "mean_jaccard":
        n = int(parent.get("n", stats.get("n", 0)))
        value = _nested_metric(stats, metric_path)
        return (
            None if value is None else value * n,
            float(n),
            support,
        )
    return None, None, support


def _metric_record(
    *,
    metric_id: str,
    stats: Mapping[str, Any],
    metric_path: str,
    interval: Mapping[str, Any] | None,
    sample_frame: str,
    n_eligible: int,
    n_observed: int,
    source_artifact_hashes: Sequence[str],
    weighting_rule: str = "unweighted_design_balanced_reference_frame",
) -> dict[str, Any]:
    value = _nested_metric(stats, metric_path)
    numerator, denominator, support = _metric_components(stats, metric_path)
    return {
        "metric_id": metric_id,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "excluded_cases": n_eligible - n_observed,
        "sample_frame": sample_frame,
        "weighting_rule": weighting_rule,
        "support": support,
        "n_eligible": n_eligible,
        "n_observed": n_observed,
        "seed": EVALUATION_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "source_artifact_hashes": sorted(set(source_artifact_hashes)),
        "interval": dict(interval) if interval is not None else None,
    }


def _difference_record(
    *,
    metric_id: str,
    point_difference: float | None,
    interval: Mapping[str, Any],
    margin: float,
    bound: str,
    sample_frame: str,
    n_eligible: int,
    source_artifact_hashes: Sequence[str],
) -> dict[str, Any]:
    if bound == "lower":
        observed_bound = interval.get("one_sided_95_lower")
        passed = (
            interval.get("status") == "ok"
            and observed_bound is not None
            and float(observed_bound) > margin
        )
    elif bound == "upper":
        observed_bound = interval.get("one_sided_95_upper")
        passed = (
            interval.get("status") == "ok"
            and observed_bound is not None
            and float(observed_bound) < margin
        )
    else:
        raise ValueError("non-inferiority bound must be lower or upper")
    return {
        "metric_id": metric_id,
        "value": point_difference,
        "numerator": None,
        "denominator": None,
        "excluded_cases": 0,
        "sample_frame": sample_frame,
        "weighting_rule": "paired_unweighted_design_balanced_reference_frame",
        "support": n_eligible,
        "n_eligible": n_eligible,
        "n_observed": n_eligible,
        "seed": EVALUATION_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "source_artifact_hashes": sorted(set(source_artifact_hashes)),
        "interval": dict(interval),
        "noninferiority_margin": margin,
        "required_bound": bound,
        "observed_bound": observed_bound,
        "passed": bool(passed),
    }


def _load_comparison(
    comparison_id: str,
    *,
    root: Path,
) -> dict[str, Any]:
    path = root / "campaign_comparisons" / f"{comparison_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    semantic = dict(value)
    declared_id = semantic.pop("comparison_id")
    declared_sha = semantic.pop("comparison_sha256")
    observed_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    if (
        declared_id != comparison_id
        or declared_sha != observed_sha
        or comparison_id != "cmp_" + observed_sha.removeprefix("sha256:")
    ):
        raise ValueError("comparison artifact identity drift")
    return value


def _gate_record(
    record: Mapping[str, Any],
    *,
    threshold: float,
    direction: str,
) -> dict[str, Any]:
    value = record.get("value")
    if direction == "at_least":
        passed = value is not None and float(value) >= threshold
    elif direction == "at_most":
        passed = value is not None and float(value) <= threshold
    else:
        raise ValueError("gate direction must be at_least or at_most")
    return {
        **dict(record),
        "threshold": threshold,
        "direction": direction,
        "passed": bool(passed),
    }


def _structural_record(
    *,
    metric_id: str,
    numerator: int,
    denominator: int,
    source_artifact_hashes: Sequence[str],
) -> dict[str, Any]:
    value = _safe_div(numerator, denominator)
    return {
        "metric_id": metric_id,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "excluded_cases": denominator - numerator,
        "sample_frame": "exact_persisted_pilot_or_reference_selection",
        "weighting_rule": "structural_count",
        "support": denominator,
        "n_eligible": denominator,
        "n_observed": numerator,
        "seed": None,
        "bootstrap_draws": 0,
        "source_artifact_hashes": sorted(set(source_artifact_hashes)),
        "interval": None,
        "threshold": 1.0,
        "direction": "at_least",
        "passed": numerator == denominator,
    }


def _sealed_blindness_audit(
    *,
    roles: Mapping[str, str],
    reference_run_id: str,
    root: Path,
) -> dict[str, Any]:
    campaign_receipts = 0
    compromised = 0
    for run_id in roles.values():
        directory = ledger.sealed_artifact_dir(run_id, root)
        for path in sorted(
            (directory / "execution-session-receipts").glob("*.json")
        ):
            campaign_receipts += 1
            value = json.loads(path.read_text(encoding="utf-8"))
            blind = value.get("blindness_receipt", {})
            if (
                blind.get("fresh_restricted_labeling_session") is not True
                or blind.get("sibling_responses_inspected") is not False
            ):
                compromised += 1
    reference = refresh.load_independent_reference(
        reference_run_id, root=root
    )
    directory = ledger.sealed_artifact_dir(reference_run_id, root)
    initial = json.loads(
        (directory / "reference-initial-blindness-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        refresh._validate_reference_blindness(initial)  # type: ignore[attr-defined]
    except ValueError:
        compromised += 1
    reference_receipts = 0
    for path in sorted(
        (directory / "reference-session-receipts").glob("*.json")
    ):
        reference_receipts += 1
        value = json.loads(path.read_text(encoding="utf-8"))
        try:
            refresh._validate_reference_blindness(  # type: ignore[attr-defined]
                value.get("blindness_receipt", {})
            )
        except ValueError:
            compromised += 1
    if reference.get("candidate_exposure_before_seal") != "prohibited":
        compromised += 1
    return {
        "campaign_session_receipts": campaign_receipts,
        "reference_session_receipts": reference_receipts,
        "compromised_blindness_receipts": compromised,
    }


def build_evaluation(
    *,
    campaign_id: str,
    reference_run_id: str,
    adjudication_run_id: str,
    policy_amendment_path: Path,
    comparison_ids: Sequence[str],
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Build the complete deterministic Stage 3 gate evaluation."""
    root = Path(root)
    campaign = refresh.load_campaign(campaign_id, root=root)
    roles = _campaign_role_runs(campaign_id, root=root)
    source_runs = {
        role: _sealed_source(run_id, root)
        for role, run_id in roles.items()
    }
    source_runs["reference"] = _sealed_source(reference_run_id, root)
    source_runs["adjudication"] = _sealed_source(
        adjudication_run_id, root
    )
    amendment = json.loads(
        Path(policy_amendment_path).read_text(encoding="utf-8")
    )
    amendment_result = refresh.publish_evaluation_policy_amendment(
        amendment, root=root
    )
    if amendment_result["campaign_id"] != campaign_id:
        raise ValueError("evaluation policy amendment campaign drift")
    adjudication = load_reference_adjudication(
        adjudication_run_id, root=root
    )
    reference_manifest = refresh.load_independent_reference(
        reference_run_id, root=root
    )
    if (
        adjudication["campaign_id"] != campaign_id
        or adjudication["reference_run_id"] != reference_run_id
    ):
        raise ValueError("adjudication evidence belongs to another campaign")
    comparisons = [
        _load_comparison(comparison_id, root=root)
        for comparison_id in comparison_ids
    ]
    if any(row["campaign_id"] != campaign_id for row in comparisons):
        raise ValueError("comparison evidence belongs to another campaign")
    active_generation_id = (root / "materialized" / "current").read_text(
        encoding="utf-8"
    ).strip()
    expected_generation_id = campaign["corpus_snapshot"][
        "active_generation_id"
    ]
    if active_generation_id != expected_generation_id:
        raise ValueError("active primary-label generation drift")
    active_generation_path = (
        root / "materialized" / "generations" / active_generation_id
    )
    active_generation_manifest_sha = ledger.sha256_file(
        active_generation_path / "generation.json"
    )
    if (
        active_generation_manifest_sha
        != campaign["corpus_snapshot"][
            "active_generation_manifest_sha256"
        ]
    ):
        raise ValueError("active generation manifest hash drift")
    required_comparison_roles = {
        "blind_a_vs_blind_b",
        "blind_a_vs_diagnostic",
        "blind_b_vs_diagnostic",
        "reference_vs_blind_a",
        "reference_vs_blind_b",
    }
    comparison_roles = {row["comparison_role"] for row in comparisons}
    if comparison_roles != required_comparison_roles:
        raise ValueError("evaluation comparison role set is incomplete")

    groups = {
        "blind_a": _run_groups(roles["blind_a"], root=root),
        "blind_b": _run_groups(roles["blind_b"], root=root),
        "diagnostic": _run_groups(
            roles["composition_diagnostic"], root=root
        ),
        "reference": _run_groups(reference_run_id, root=root),
        "adjudication": _run_groups(adjudication_run_id, root=root),
    }
    disputes = required_reference_disputes(
        reference_run_id=reference_run_id,
        blind_a_run_id=roles["blind_a"],
        blind_b_run_id=roles["blind_b"],
        root=root,
    )
    if disputes != adjudication["disputes"]:
        raise ValueError("required dispute set changed after adjudication")
    decisions = ledger._read_decisions(  # type: ignore[attr-defined]
        "adjudications", "label_adjudications.jsonl", root
    )
    decisions_by_key = {
        (row["canonical_subject_id"], row["label_type"]): row
        for row in decisions
        if row["resolved_label_group_id"]
        in {
            groups["adjudication"][key]["label_group_id"]
            for key in groups["adjudication"]
        }
    }
    for dispute in disputes:
        key = (
            dispute["canonical_subject_id"],
            dispute["label_type"],
        )
        decision = decisions_by_key.get(key)
        if (
            decision is None
            or decision["resolution_kind"]
            != "synthesize_in_adjudication_run"
            or decision["resolved_label_group_id"]
            != groups["adjudication"][key]["label_group_id"]
            or set(
                json.loads(decision["candidate_label_group_ids_json"])
            )
            != set(dispute["candidate_label_group_ids"])
        ):
            raise ValueError(f"required adjudication is incomplete at {key}")

    reference_subjects = sorted(
        {
            subject_id
            for subject_id, label_type in groups["reference"]
            if label_type == "constituencies"
        }
    )
    if len(reference_subjects) != 240:
        raise ValueError("evaluation reference is not exactly 240 subjects")
    gold = dict(groups["reference"])
    for dispute in disputes:
        key = (
            dispute["canonical_subject_id"],
            dispute["label_type"],
        )
        gold[key] = groups["adjudication"][key]
    if any(
        _value(gold, subject_id, "constituencies")["outcome"] == "unclear"
        for subject_id in reference_subjects
    ):
        raise ValueError("adjudicated reference contains unresolved unclear rows")
    sealed_subject_rows = ledger.read_jsonl(
        ledger.sealed_artifact_dir(reference_run_id, root)
        / "subjects.jsonl"
    )
    doc_by_subject = {
        str(row["subject_id"]): str(row["subject_key"]["doc_name"])
        for row in sealed_subject_rows
    }
    if set(reference_subjects) - set(doc_by_subject):
        raise ValueError("reference speech-cluster map is incomplete")
    reference_draws = _bootstrap_indices(
        reference_subjects, doc_by_subject
    )
    source_hashes = [
        row["artifact_set_sha256"] for row in source_runs.values()
    ] + [
        active_generation_manifest_sha,
        reference_manifest["reference_manifest_sha256"],
        adjudication["adjudication_manifest_sha256"],
        amendment_result["policy_amendment_sha256"],
        *[row["comparison_sha256"] for row in comparisons],
    ]

    structural_gates: list[dict[str, Any]] = []
    expected_coverage = {
        "blind_a": 722,
        "blind_b": 722,
        "diagnostic": 144,
        "reference": 240,
        "adjudication": adjudication["n_disputed_subjects"],
    }
    for role, expected in expected_coverage.items():
        label_type = (
            "constituencies"
            if role in {"diagnostic", "reference", "adjudication"}
            else None
        )
        observed = len(
            {
                key[0]
                for key in groups[role]
                if label_type is None or key[1] == label_type
            }
        )
        structural_gates.append(
            _structural_record(
                metric_id=f"structural.{role}.subject_coverage",
                numerator=observed,
                denominator=expected,
                source_artifact_hashes=source_hashes,
            )
        )
        expected_labels = 1 if role == "diagnostic" else 7
        structural_gates.append(
            _structural_record(
                metric_id=f"structural.{role}.subject_label_group_coverage",
                numerator=len(groups[role]),
                denominator=expected * expected_labels,
                source_artifact_hashes=source_hashes,
            )
        )
    total_accepted_events = sum(
        len(
            ledger.read_jsonl(
                ledger.sealed_artifact_dir(source["run_id"], root)
                / "label_events.jsonl"
            )
        )
        for source in source_runs.values()
    )
    structural_gates.extend(
        [
            _structural_record(
                metric_id="structural.accepted_event_validity",
                numerator=total_accepted_events,
                denominator=total_accepted_events,
                source_artifact_hashes=source_hashes,
            ),
            _structural_record(
                metric_id="structural.sealed_artifact_verification",
                numerator=len(source_runs),
                denominator=len(source_runs),
                source_artifact_hashes=source_hashes,
            ),
        ]
    )
    blindness = _sealed_blindness_audit(
        roles=roles,
        reference_run_id=reference_run_id,
        root=root,
    )
    structural_gates.append(
        {
            "metric_id": "structural.compromised_blindness_receipts",
            "value": float(blindness["compromised_blindness_receipts"]),
            "numerator": blindness["compromised_blindness_receipts"],
            "denominator": (
                blindness["campaign_session_receipts"]
                + blindness["reference_session_receipts"]
                + 1
            ),
            "excluded_cases": 0,
            "sample_frame": "all_persisted_execution_and_reference_sessions",
            "weighting_rule": "structural_count",
            "support": (
                blindness["campaign_session_receipts"]
                + blindness["reference_session_receipts"]
                + 1
            ),
            "n_eligible": (
                blindness["campaign_session_receipts"]
                + blindness["reference_session_receipts"]
                + 1
            ),
            "n_observed": (
                blindness["campaign_session_receipts"]
                + blindness["reference_session_receipts"]
                + 1
            ),
            "seed": None,
            "bootstrap_draws": 0,
            "source_artifact_hashes": sorted(set(source_hashes)),
            "interval": None,
            "threshold": 0.0,
            "direction": "at_most",
            "passed": blindness["compromised_blindness_receipts"] == 0,
        }
    )

    constituency_values = {
        role: [
            _value(groups[role], subject_id, "constituencies")
            for subject_id in reference_subjects
        ]
        for role in ("blind_a", "blind_b")
    }
    gold_constituencies = [
        _value(gold, subject_id, "constituencies")
        for subject_id in reference_subjects
    ]
    positive_support = sum(
        row["outcome"] == "claim" for row in gold_constituencies
    )
    constituency_gates: list[dict[str, Any]] = [
        {
            **_structural_record(
                metric_id="constituency.reference_positive_support",
                numerator=positive_support,
                denominator=60,
                source_artifact_hashes=source_hashes,
            ),
            "value": float(positive_support),
            "numerator": positive_support,
            "denominator": None,
            "excluded_cases": 0,
            "support": positive_support,
            "n_eligible": 240,
            "n_observed": 240,
            "threshold": 60.0,
            "direction": "at_least",
            "passed": positive_support >= 60,
        }
    ]
    constituency_descriptive: dict[str, Any] = {}
    constituency_thresholds = {
        "claim_detection.precision": (0.85, "at_least"),
        "claim_detection.recall": (0.85, "at_least"),
        "claim_detection.f1": (0.85, "at_least"),
        "claim_detection.balanced_accuracy": (0.85, "at_least"),
        "claim_detection.claim_miss_rate": (0.15, "at_most"),
        "claim_detection.predicted_none_false_omission_rate": (
            0.10,
            "at_most",
        ),
        "claim_exact.micro_f1": (0.80, "at_least"),
        "supported_category_macro_f1": (0.75, "at_least"),
        "claim_exact.mean_jaccard": (0.70, "at_least"),
    }
    for role, pred in constituency_values.items():
        stats = _label_stats(
            "constituencies", gold_constituencies, pred
        )
        constituency_descriptive[role] = stats
        for metric_path, (threshold, direction) in (
            constituency_thresholds.items()
        ):
            draws = _resampled_metric(
                label_type="constituencies",
                gold=gold_constituencies,
                pred=pred,
                indices=reference_draws,
                metric_path=metric_path,
            )
            record = _metric_record(
                metric_id=f"constituency.{role}.{metric_path}",
                stats=stats,
                metric_path=metric_path,
                interval=_interval(draws),
                sample_frame="adjudicated_240_paragraph_reference",
                n_eligible=240,
                n_observed=240,
                source_artifact_hashes=source_hashes,
            )
            constituency_gates.append(
                _gate_record(
                    record, threshold=threshold, direction=direction
                )
            )
        for category, category_stats in stats["per_category"].items():
            if category_stats["support_positive"] < 20:
                continue
            for metric_name in ("precision", "recall"):
                record = _metric_record(
                    metric_id=(
                        f"constituency.{role}.category.{category}."
                        f"{metric_name}"
                    ),
                    stats=category_stats,
                    metric_path=metric_name,
                    interval=None,
                    sample_frame="adjudicated_240_paragraph_reference",
                    n_eligible=240,
                    n_observed=240,
                    source_artifact_hashes=source_hashes,
                )
                constituency_gates.append(
                    _gate_record(
                        record,
                        threshold=0.70,
                        direction="at_least",
                    )
                )
    interblind = _label_stats(
        "constituencies",
        constituency_values["blind_a"],
        constituency_values["blind_b"],
    )
    for metric_path, threshold in (
        ("outcome_accuracy", 0.85),
        ("outcome_kappa", 0.70),
    ):
        draws = _resampled_metric(
            label_type="constituencies",
            gold=constituency_values["blind_a"],
            pred=constituency_values["blind_b"],
            indices=reference_draws,
            metric_path=metric_path,
        )
        record = _metric_record(
            metric_id=f"constituency.interblind.{metric_path}",
            stats=interblind,
            metric_path=metric_path,
            interval=_interval(draws),
            sample_frame="fixed_240_paragraph_reference_overlap",
            n_eligible=240,
            n_observed=240,
            source_artifact_hashes=source_hashes,
        )
        constituency_gates.append(
            _gate_record(
                record, threshold=threshold, direction="at_least"
            )
        )

    diagnostic_subjects = sorted(
        {
            key[0]
            for key in groups["diagnostic"]
            if key[1] == "constituencies"
        }
    )
    if len(diagnostic_subjects) != 144 or not set(
        diagnostic_subjects
    ) <= set(reference_subjects):
        raise ValueError("composition diagnostic overlap is not exact")
    diagnostic_draws = _bootstrap_indices(
        diagnostic_subjects, doc_by_subject
    )
    diagnostic_gold = [
        _value(gold, subject_id, "constituencies")
        for subject_id in diagnostic_subjects
    ]
    diagnostic_pred = [
        _value(groups["diagnostic"], subject_id, "constituencies")
        for subject_id in diagnostic_subjects
    ]
    diagnostic_stats = _label_stats(
        "constituencies", diagnostic_gold, diagnostic_pred
    )
    composition_gates: list[dict[str, Any]] = []
    composition_descriptive: dict[str, Any] = {
        "diagnostic": diagnostic_stats
    }
    composition_metrics = {
        "claim_detection.f1": (-0.05, "lower"),
        "claim_detection.recall": (-0.05, "lower"),
        "claim_detection.predicted_none_false_omission_rate": (
            0.05,
            "upper",
        ),
    }
    diagnostic_draw_values = {
        metric_path: _resampled_metric(
            label_type="constituencies",
            gold=diagnostic_gold,
            pred=diagnostic_pred,
            indices=diagnostic_draws,
            metric_path=metric_path,
        )
        for metric_path in composition_metrics
    }
    for role in ("blind_a", "blind_b"):
        combined_pred = [
            _value(groups[role], subject_id, "constituencies")
            for subject_id in diagnostic_subjects
        ]
        combined_stats = _label_stats(
            "constituencies", diagnostic_gold, combined_pred
        )
        composition_descriptive[role] = combined_stats
        for metric_path, (margin, bound) in composition_metrics.items():
            combined_draws = _resampled_metric(
                label_type="constituencies",
                gold=diagnostic_gold,
                pred=combined_pred,
                indices=diagnostic_draws,
                metric_path=metric_path,
            )
            differences = _paired_difference(
                combined_draws,
                diagnostic_draw_values[metric_path],
            )
            left = _nested_metric(combined_stats, metric_path)
            right = _nested_metric(diagnostic_stats, metric_path)
            point = (
                None if left is None or right is None else left - right
            )
            composition_gates.append(
                _difference_record(
                    metric_id=(
                        f"composition.{role}_minus_diagnostic."
                        f"{metric_path}"
                    ),
                    point_difference=point,
                    interval=_interval(differences),
                    margin=margin,
                    bound=bound,
                    sample_frame="fixed_144_paragraph_composition_diagnostic",
                    n_eligible=144,
                    source_artifact_hashes=source_hashes,
                )
            )

    current = _current_groups(
        root=root, subject_ids=set(reference_subjects)
    )
    existing_metric_paths = {
        "party_attack": ("accuracy", "f1"),
        "enemy_naming": ("accuracy", "f1"),
        "zero_sum": ("accuracy", "f1"),
        "proposal_values": ("accuracy", "macro_f1_supported"),
        "topics": ("micro_f1", "mean_jaccard"),
        "entities": (
            "exact.micro_f1",
            "partial_name_and_type.micro_f1",
        ),
    }
    existing_margins = {
        "party_attack": -0.03,
        "enemy_naming": -0.03,
        "zero_sum": -0.03,
        "proposal_values": -0.03,
        "topics": -0.05,
        "entities": -0.05,
    }
    existing_gates: list[dict[str, Any]] = []
    existing_descriptive: dict[str, Any] = {}
    for label_type in EXISTING_LABEL_TYPES:
        gold_values = [
            _value(gold, subject_id, label_type)
            for subject_id in reference_subjects
        ]
        current_values = [
            _value(current, subject_id, label_type)
            for subject_id in reference_subjects
        ]
        current_stats = _label_stats(
            label_type, gold_values, current_values
        )
        existing_descriptive[label_type] = {
            "current_primary": current_stats
        }
        current_draws = {
            metric_path: _resampled_metric(
                label_type=label_type,
                gold=gold_values,
                pred=current_values,
                indices=reference_draws,
                metric_path=metric_path,
            )
            for metric_path in existing_metric_paths[label_type]
        }
        for role in ("blind_a", "blind_b"):
            candidate_values = [
                _value(groups[role], subject_id, label_type)
                for subject_id in reference_subjects
            ]
            candidate_stats = _label_stats(
                label_type, gold_values, candidate_values
            )
            existing_descriptive[label_type][role] = candidate_stats
            for metric_path in existing_metric_paths[label_type]:
                candidate_draws = _resampled_metric(
                    label_type=label_type,
                    gold=gold_values,
                    pred=candidate_values,
                    indices=reference_draws,
                    metric_path=metric_path,
                )
                differences = _paired_difference(
                    candidate_draws, current_draws[metric_path]
                )
                left = _nested_metric(candidate_stats, metric_path)
                right = _nested_metric(current_stats, metric_path)
                point = (
                    None if left is None or right is None else left - right
                )
                existing_gates.append(
                    _difference_record(
                        metric_id=(
                            f"existing.{role}_minus_current."
                            f"{label_type}.{metric_path}"
                        ),
                        point_difference=point,
                        interval=_interval(differences),
                        margin=existing_margins[label_type],
                        bound="lower",
                        sample_frame="adjudicated_240_paragraph_reference",
                        n_eligible=240,
                        source_artifact_hashes=source_hashes,
                    )
                )
            if label_type in BOOLEAN_LABEL_TYPES:
                supported = (
                    [("positive", candidate_stats)]
                    if candidate_stats["support_positive"] >= 20
                    else []
                )
            elif label_type == "proposal_values":
                supported = [
                    (category, stats)
                    for category, stats in candidate_stats[
                        "per_class"
                    ].items()
                    if stats["support_positive"] >= 20
                ]
            else:
                supported = [
                    (category, stats)
                    for category, stats in candidate_stats[
                        "per_category"
                    ].items()
                    if stats["support_positive"] >= 20
                ]
            for category, category_stats in supported:
                record = _metric_record(
                    metric_id=(
                        f"existing.{role}.{label_type}.category."
                        f"{category}.absolute_f1"
                    ),
                    stats=category_stats,
                    metric_path="f1",
                    interval=None,
                    sample_frame="adjudicated_240_paragraph_reference",
                    n_eligible=240,
                    n_observed=240,
                    source_artifact_hashes=source_hashes,
                )
                existing_gates.append(
                    _gate_record(
                        record,
                        threshold=0.80,
                        direction="at_least",
                    )
                )

    all_gates = (
        structural_gates
        + constituency_gates
        + composition_gates
        + existing_gates
    )
    failed = [
        row["metric_id"] for row in all_gates if not row["passed"]
    ]
    semantic = {
        "evaluation_version": EVALUATION_VERSION,
        "campaign_id": campaign_id,
        "effective_evaluation_policy_sha256": amendment_result[
            "effective_evaluation_policy_sha256"
        ],
        "policy_amendment_id": amendment_result["amendment_id"],
        "removed_requirements": amendment_result[
            "removed_requirements"
        ],
        "source_runs": source_runs,
        "reference_manifest_sha256": reference_manifest[
            "reference_manifest_sha256"
        ],
        "adjudication_manifest_sha256": adjudication[
            "adjudication_manifest_sha256"
        ],
        "adjudication_selection_sha256": adjudication[
            "selection_sha256"
        ],
        "adjudication_dispute_scope_sha256": adjudication[
            "dispute_scope_sha256"
        ],
        "current_primary_source": {
            "active_generation_id": active_generation_id,
            "generation_manifest_sha256": active_generation_manifest_sha,
        },
        "comparison_sources": [
            {
                "comparison_id": row["comparison_id"],
                "comparison_sha256": row["comparison_sha256"],
                "comparison_role": row["comparison_role"],
            }
            for row in sorted(
                comparisons, key=lambda value: value["comparison_role"]
            )
        ],
        "adjudication_run_id": adjudication_run_id,
        "adjudication_ids": sorted(
            row["adjudication_id"]
            for row in decisions_by_key.values()
            if (
                row["canonical_subject_id"],
                row["label_type"],
            )
            in {
                (
                    dispute["canonical_subject_id"],
                    dispute["label_type"],
                )
                for dispute in disputes
            }
        ),
        "evaluation_seed": EVALUATION_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_unit": "speech",
        "sample_frames": {
            "reference": 240,
            "composition_diagnostic": 144,
        },
        "blindness_audit": blindness,
        "structural_gates": structural_gates,
        "constituency_gates": constituency_gates,
        "composition_gates": composition_gates,
        "existing_label_gates": existing_gates,
        "descriptive_metrics": {
            "constituency": constituency_descriptive,
            "interblind": interblind,
            "composition": composition_descriptive,
            "existing_labels": existing_descriptive,
        },
        "gate_summary": {
            "status": "pass" if not failed else "fail",
            "total": len(all_gates),
            "passed": len(all_gates) - len(failed),
            "failed": len(failed),
            "failed_metric_ids": failed,
        },
    }
    evaluation_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    return {
        **semantic,
        "evaluation_id": "eval_" + evaluation_sha.removeprefix("sha256:"),
        "evaluation_sha256": evaluation_sha,
    }


def publish_evaluation(
    evaluation: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Atomically publish one content-addressed Stage 3 evaluation."""
    value = dict(evaluation)
    evaluation_id = str(value.pop("evaluation_id", ""))
    declared_sha = str(value.pop("evaluation_sha256", ""))
    observed_sha = ledger.sha256_text(ledger.canonical_json(value))
    if (
        declared_sha != observed_sha
        or evaluation_id != "eval_" + declared_sha.removeprefix("sha256:")
    ):
        raise ValueError("evaluation identity drift")
    persisted = {
        **value,
        "evaluation_id": evaluation_id,
        "evaluation_sha256": declared_sha,
    }
    path = Path(root) / "campaign_evaluations" / f"{evaluation_id}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != persisted:
            raise ValueError("existing evaluation drift")
        status = "already_published"
    else:
        ledger.atomic_write_json(path, persisted)
        status = "published"
    summary = persisted["gate_summary"]
    return {
        "status": status,
        "evaluation_id": evaluation_id,
        "evaluation_sha256": declared_sha,
        "path": str(path),
        "gate_status": summary["status"],
        "gates_total": summary["total"],
        "gates_passed": summary["passed"],
        "gates_failed": summary["failed"],
    }


def _load_evaluation(
    evaluation_id: str,
    *,
    root: Path,
) -> dict[str, Any]:
    path = root / "campaign_evaluations" / f"{evaluation_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    semantic = dict(value)
    declared_id = semantic.pop("evaluation_id")
    declared_sha = semantic.pop("evaluation_sha256")
    observed = ledger.sha256_text(ledger.canonical_json(semantic))
    if (
        declared_id != evaluation_id
        or declared_sha != observed
        or evaluation_id != "eval_" + observed.removeprefix("sha256:")
    ):
        raise ValueError("evaluation artifact identity drift")
    return value


def _registry_with_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str = "registry_sha256",
) -> dict[str, Any]:
    result = dict(value)
    result.pop(hash_field, None)
    result[hash_field] = ledger.sha256_text(ledger.canonical_json(result))
    return result


def build_freeze_artifacts(
    *,
    evaluation_id: str,
    root: Path = ledger.LEDGER_ROOT,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
    bundle_registry_path: Path = refresh.BUNDLE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Build additive final specs/bundle from a fully passing evaluation."""
    root = Path(root)
    label_registry_path = Path(label_registry_path)
    bundle_registry_path = Path(bundle_registry_path)
    evaluation = _load_evaluation(evaluation_id, root=root)
    if evaluation["gate_summary"]["status"] != "pass":
        raise ValueError("a failed or inconclusive evaluation cannot freeze")
    label_registry = ledger.read_registry(label_registry_path)
    bundle_registry = refresh.read_bundle_registry(
        bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    candidate_bundle_entry = next(
        row
        for row in bundle_registry["entries"]
        if (
            row["bundle_id"],
            row["bundle_version"],
        )
        == ("paragraph_judgment_v2_candidate", "candidate-1")
    )
    candidate_bundle = json.loads(
        (
            bundle_registry_path.parent
            / candidate_bundle_entry["path"]
        ).read_text(encoding="utf-8")
    )
    final_bundle = json.loads(json.dumps(candidate_bundle))
    final_bundle.update(
        {
            "bundle_id": "paragraph_judgment_v2",
            "bundle_version": "v2",
            "stability": "frozen",
            "eligibility": "production_eligible",
            "prompt_version": "paragraph-judgment-v2/v2",
            "emitted_label_specs": [
                {
                    **row,
                    "spec_version": FINAL_SPEC_VERSIONS[row["label_type"]],
                }
                for row in candidate_bundle["emitted_label_specs"]
            ],
            "limitations": [
                "Frozen from the byte-identical passing candidate prompt and "
                f"response schema under evaluation {evaluation_id}.",
                "Publication eligibility does not itself promote or materialize labels.",
            ],
        }
    )
    final_bundle["bundle_sha256"] = refresh.bundle_hash(final_bundle)
    if (
        final_bundle["prompt_text"] != candidate_bundle["prompt_text"]
        or final_bundle["prompt_sha256"]
        != candidate_bundle["prompt_sha256"]
        or final_bundle["response_schema"]
        != candidate_bundle["response_schema"]
        or final_bundle["response_schema_sha256"]
        != candidate_bundle["response_schema_sha256"]
    ):
        raise ValueError("final bundle prompt/schema differs from passing candidate")

    registry_entries = {
        (row["label_type"], row["spec_version"]): row
        for row in label_registry["entries"]
    }
    candidate_versions = {
        **{
            label_type: "paragraph-judgment-v2-candidate-1"
            for label_type in EXISTING_LABEL_TYPES
        },
        "constituencies": "constituency-v1-candidate-1",
    }
    final_specs: dict[str, dict[str, Any]] = {}
    final_spec_paths: dict[str, str] = {}
    for label_type, final_version in FINAL_SPEC_VERSIONS.items():
        entry = registry_entries[(label_type, candidate_versions[label_type])]
        candidate = json.loads(
            (label_registry_path.parent / entry["path"]).read_text(
                encoding="utf-8"
            )
        )
        final = json.loads(json.dumps(candidate))
        pointer = next(
            row["response_pointer"]
            for row in final_bundle["emitted_label_specs"]
            if row["label_type"] == label_type
        )
        final.update(
            {
                "spec_version": final_version,
                "stability": "frozen",
                "prompt_version": (
                    "constituencies/v1"
                    if label_type == "constituencies"
                    else f"{label_type}/paragraph-judgment-v2"
                ),
                "limitations": [
                    "Frozen from the byte-identical passing candidate field "
                    f"contract under evaluation {evaluation_id}.",
                    "Registration does not itself promote or materialize labels.",
                ],
                "execution_binding": {
                    "mode": "bundle",
                    "bundle_refs": [
                        {
                            "bundle_id": final_bundle["bundle_id"],
                            "bundle_version": final_bundle["bundle_version"],
                            "bundle_sha256": final_bundle[
                                "bundle_sha256"
                            ],
                            "response_pointer": pointer,
                        }
                    ],
                    "decoder": "json_pointer",
                    "atomic_group_rule": "one_label_type_per_subject_response",
                },
                "source_spec_refs": [
                    *candidate["source_spec_refs"],
                    {
                        "inherited_spec_version": candidate["spec_version"],
                        "inherited_spec_sha256": candidate["spec_sha256"],
                    },
                ],
            }
        )
        final["spec_sha256"] = ledger.spec_hash(final)
        ledger.validate_spec(final)
        if (
            final["prompt_text"] != candidate["prompt_text"]
            or final["prompt_sha256"] != candidate["prompt_sha256"]
            or final["response_schema"] != candidate["response_schema"]
            or final["response_schema_sha256"]
            != candidate["response_schema_sha256"]
        ):
            raise ValueError(
                f"{label_type}: final prompt/schema differs from passing candidate"
            )
        final_specs[label_type] = final
        final_spec_paths[label_type] = f"{label_type}/{final_version}.json"

    updated_label = json.loads(json.dumps(label_registry))
    entries_by_identity = {
        (row["label_type"], row["spec_version"]): row
        for row in updated_label["entries"]
    }
    for label_type, final in final_specs.items():
        entry = {
            "label_type": label_type,
            "spec_version": final["spec_version"],
            "spec_sha256": final["spec_sha256"],
            "stability": "frozen",
            "path": final_spec_paths[label_type],
        }
        identity = (label_type, final["spec_version"])
        existing = entries_by_identity.get(identity)
        if existing is not None and existing != entry:
            raise ValueError(f"existing final spec identity drift: {identity}")
        entries_by_identity[identity] = entry
    updated_label["entries"] = sorted(
        entries_by_identity.values(),
        key=lambda row: (row["label_type"], row["spec_version"]),
    )
    updated_label = _registry_with_hash(updated_label)

    final_bundle_path = "paragraph_judgment_v2/v2.json"
    final_bundle_entry = {
        "bundle_id": final_bundle["bundle_id"],
        "bundle_version": final_bundle["bundle_version"],
        "bundle_sha256": final_bundle["bundle_sha256"],
        "eligibility": final_bundle["eligibility"],
        "path": final_bundle_path,
    }
    updated_bundle = json.loads(json.dumps(bundle_registry))
    bundle_entries = {
        (row["bundle_id"], row["bundle_version"]): row
        for row in updated_bundle["entries"]
    }
    identity = (
        final_bundle["bundle_id"],
        final_bundle["bundle_version"],
    )
    existing = bundle_entries.get(identity)
    if existing is not None and existing != final_bundle_entry:
        raise ValueError("existing final bundle identity drift")
    bundle_entries[identity] = final_bundle_entry
    updated_bundle["entries"] = sorted(
        bundle_entries.values(),
        key=lambda row: (row["bundle_id"], row["bundle_version"]),
    )
    updated_bundle["label_registry_sha256"] = updated_label[
        "registry_sha256"
    ]
    updated_bundle = _registry_with_hash(updated_bundle)

    semantic = {
        "freeze_version": FREEZE_VERSION,
        "campaign_id": evaluation["campaign_id"],
        "evaluation_id": evaluation_id,
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "effective_evaluation_policy_sha256": evaluation[
            "effective_evaluation_policy_sha256"
        ],
        "source_candidate_bundle": {
            "bundle_id": candidate_bundle["bundle_id"],
            "bundle_version": candidate_bundle["bundle_version"],
            "bundle_sha256": candidate_bundle["bundle_sha256"],
        },
        "final_bundle": final_bundle_entry,
        "final_specs": [
            {
                "label_type": label_type,
                "spec_version": final_specs[label_type]["spec_version"],
                "spec_sha256": final_specs[label_type]["spec_sha256"],
                "path": final_spec_paths[label_type],
            }
            for label_type in sorted(final_specs)
        ],
        "label_registry_sha256_before": label_registry["registry_sha256"],
        "label_registry_sha256_after": updated_label["registry_sha256"],
        "bundle_registry_sha256_before": bundle_registry[
            "registry_sha256"
        ],
        "bundle_registry_sha256_after": updated_bundle["registry_sha256"],
        "prompt_byte_identity": {
            "bundle_prompt_sha256": final_bundle["prompt_sha256"],
            "bundle_response_schema_sha256": final_bundle[
                "response_schema_sha256"
            ],
            "all_candidate_equal": True,
        },
        "production_eligible": True,
        "promotion_or_materialization_performed": False,
    }
    for path in sorted(
        (root / "specification_freezes").glob("freeze_*.json")
    ):
        existing_freeze = json.loads(path.read_text(encoding="utf-8"))
        if existing_freeze.get("evaluation_id") != evaluation_id:
            continue
        expected_specs = {
            (row["label_type"], row["spec_version"], row["spec_sha256"])
            for row in semantic["final_specs"]
        }
        observed_specs = {
            (row["label_type"], row["spec_version"], row["spec_sha256"])
            for row in existing_freeze.get("final_specs", [])
        }
        if (
            existing_freeze.get("final_bundle") != semantic["final_bundle"]
            or observed_specs != expected_specs
        ):
            raise ValueError("existing freeze for evaluation has artifact drift")
        semantic["label_registry_sha256_before"] = existing_freeze[
            "label_registry_sha256_before"
        ]
        semantic["bundle_registry_sha256_before"] = existing_freeze[
            "bundle_registry_sha256_before"
        ]
        break
    freeze_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    manifest = {
        **semantic,
        "freeze_id": "freeze_" + freeze_sha.removeprefix("sha256:"),
        "freeze_sha256": freeze_sha,
    }
    return {
        "manifest": manifest,
        "final_specs": final_specs,
        "final_spec_paths": final_spec_paths,
        "final_bundle": final_bundle,
        "final_bundle_path": final_bundle_path,
        "updated_label_registry": updated_label,
        "updated_bundle_registry": updated_bundle,
    }


def publish_freeze_artifacts(
    freeze: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
    bundle_registry_path: Path = refresh.BUNDLE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Publish additive specs first and registry commit markers last."""
    root = Path(root)
    label_registry_path = Path(label_registry_path)
    bundle_registry_path = Path(bundle_registry_path)
    manifest = dict(freeze["manifest"])
    semantic = dict(manifest)
    freeze_id = semantic.pop("freeze_id")
    declared_sha = semantic.pop("freeze_sha256")
    observed = ledger.sha256_text(ledger.canonical_json(semantic))
    if (
        declared_sha != observed
        or freeze_id != "freeze_" + observed.removeprefix("sha256:")
    ):
        raise ValueError("freeze manifest identity drift")
    evaluation = _load_evaluation(manifest["evaluation_id"], root=root)
    if (
        evaluation["evaluation_sha256"] != manifest["evaluation_sha256"]
        or evaluation["gate_summary"]["status"] != "pass"
    ):
        raise ValueError("freeze evaluation evidence is not passing")
    final_specs = dict(freeze["final_specs"])
    paths = dict(freeze["final_spec_paths"])
    for label_type, spec in final_specs.items():
        ledger.validate_spec(spec)
        path = label_registry_path.parent / paths[label_type]
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != spec:
                raise ValueError(f"existing frozen spec drift: {label_type}")
        else:
            ledger.atomic_write_json(path, spec)
    final_bundle = dict(freeze["final_bundle"])
    if refresh.bundle_hash(final_bundle) != final_bundle["bundle_sha256"]:
        raise ValueError("final bundle hash drift")
    bundle_path = (
        bundle_registry_path.parent / str(freeze["final_bundle_path"])
    )
    if bundle_path.exists():
        if json.loads(bundle_path.read_text(encoding="utf-8")) != final_bundle:
            raise ValueError("existing frozen bundle drift")
    else:
        ledger.atomic_write_json(bundle_path, final_bundle)

    updated_label = dict(freeze["updated_label_registry"])
    current_label = json.loads(label_registry_path.read_text(encoding="utf-8"))
    current_label_sha = current_label["registry_sha256"]
    if current_label_sha == manifest["label_registry_sha256_before"]:
        ledger.atomic_write_json(label_registry_path, updated_label)
    elif current_label != updated_label:
        raise ValueError("label registry changed during freeze publication")

    updated_bundle = dict(freeze["updated_bundle_registry"])
    current_bundle = json.loads(
        bundle_registry_path.read_text(encoding="utf-8")
    )
    current_bundle_sha = current_bundle["registry_sha256"]
    if current_bundle_sha == manifest["bundle_registry_sha256_before"]:
        ledger.atomic_write_json(bundle_registry_path, updated_bundle)
    elif current_bundle != updated_bundle:
        raise ValueError("bundle registry changed during freeze publication")
    refresh.read_bundle_registry(
        bundle_registry_path,
        label_registry_path=label_registry_path,
    )

    path = root / "specification_freezes" / f"{freeze_id}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing freeze manifest drift")
        status = "already_published"
    else:
        ledger.atomic_write_json(path, manifest)
        status = "published"
    return {
        "status": status,
        "freeze_id": freeze_id,
        "freeze_sha256": declared_sha,
        "path": str(path),
        "final_bundle_id": final_bundle["bundle_id"],
        "final_bundle_version": final_bundle["bundle_version"],
        "final_bundle_sha256": final_bundle["bundle_sha256"],
        "n_final_specs": len(final_specs),
        "production_eligible": True,
        "promotion_or_materialization_performed": False,
    }


def build_campaign_outcome(
    *,
    campaign_id: str,
    evaluation_id: str,
    freeze_id: str,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Build the terminal pilot outcome without promoting any labels."""
    root = Path(root)
    campaign = refresh.load_campaign(campaign_id, root=root)
    evaluation = _load_evaluation(evaluation_id, root=root)
    if (
        evaluation["campaign_id"] != campaign_id
        or evaluation["gate_summary"]["status"] != "pass"
    ):
        raise ValueError("campaign outcome requires its passing evaluation")
    freeze_path = root / "specification_freezes" / f"{freeze_id}.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_semantic = dict(freeze)
    declared_freeze_id = freeze_semantic.pop("freeze_id")
    declared_freeze_sha = freeze_semantic.pop("freeze_sha256")
    if (
        declared_freeze_id != freeze_id
        or declared_freeze_sha
        != ledger.sha256_text(ledger.canonical_json(freeze_semantic))
        or freeze["evaluation_id"] != evaluation_id
    ):
        raise ValueError("campaign freeze evidence drift")
    semantic = {
        "campaign_outcome_version": CAMPAIGN_OUTCOME_VERSION,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign[
            "campaign_manifest_sha256"
        ],
        "child_seals": [
            _sealed_source(row["run_id"], root)
            for row in campaign["child_runs"]
        ],
        "reference_source": evaluation["source_runs"]["reference"],
        "adjudication_source": evaluation["source_runs"]["adjudication"],
        "comparison_sources": evaluation["comparison_sources"],
        "adjudication_ids": evaluation["adjudication_ids"],
        "evaluation_id": evaluation_id,
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "freeze_id": freeze_id,
        "freeze_sha256": freeze["freeze_sha256"],
        "effective_evaluation_policy_sha256": evaluation[
            "effective_evaluation_policy_sha256"
        ],
        "outcome": "pilot_passed_and_specs_frozen",
        "full_shadow_authorized": False,
        "promotion_or_materialization_performed": False,
    }
    outcome_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    return {
        **semantic,
        "campaign_outcome_id": (
            "outcome_" + outcome_sha.removeprefix("sha256:")
        ),
        "campaign_outcome_sha256": outcome_sha,
    }


def publish_campaign_outcome(
    outcome: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    value = dict(outcome)
    outcome_id = value.pop("campaign_outcome_id", "")
    declared_sha = value.pop("campaign_outcome_sha256", "")
    observed = ledger.sha256_text(ledger.canonical_json(value))
    if (
        declared_sha != observed
        or outcome_id != "outcome_" + observed.removeprefix("sha256:")
    ):
        raise ValueError("campaign outcome identity drift")
    persisted = {
        **value,
        "campaign_outcome_id": outcome_id,
        "campaign_outcome_sha256": declared_sha,
    }
    path = Path(root) / "campaign_outcomes" / f"{outcome_id}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != persisted:
            raise ValueError("existing campaign outcome drift")
        status = "already_published"
    else:
        ledger.atomic_write_json(path, persisted)
        status = "published"
    return {
        "status": status,
        "campaign_outcome_id": outcome_id,
        "campaign_outcome_sha256": declared_sha,
        "path": str(path),
        "outcome": persisted["outcome"],
        "promotion_or_materialization_performed": False,
    }


def build_proposed_invocation_audit_contract(
    *,
    root: Path = ledger.LEDGER_ROOT,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
    bundle_registry_path: Path = refresh.BUNDLE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Build the non-executable proposal required before invocation labeling."""
    root = Path(root)
    spec = ledger.resolve_spec(
        "invocation_tone",
        "invocation-tone-v1-candidate-1",
        registry_path=label_registry_path,
    )
    bundle = refresh.resolve_bundle(
        "invocation_tone",
        "candidate-1",
        registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    generation_id = (root / "materialized" / "current").read_text(
        encoding="utf-8"
    ).strip()
    generation_path = (
        root / "materialized" / "generations" / generation_id
    )
    generation = json.loads(
        (generation_path / "generation.json").read_text(encoding="utf-8")
    )
    current = pd.read_parquet(
        generation_path / "current_labels.parquet",
        columns=[
            "canonical_subject_id",
            "label_type",
            "promotion_channel",
        ],
    )
    invocation = current[
        (current["promotion_channel"] == "primary")
        & (current["label_type"] == "invocation_tone")
    ]
    n_subjects = int(invocation["canonical_subject_id"].nunique())
    if n_subjects != 101:
        raise ValueError(
            f"invocation audit universe drift: {n_subjects}/101 subjects"
        )
    semantic = {
        "contract_version": INVOCATION_AUDIT_CONTRACT_VERSION,
        "contract_status": "proposed_requires_explicit_owner_approval",
        "review_item_id": "ARCV1-D018",
        "labeling_authorized": False,
        "freeze_authorized": False,
        "scope": {
            "subject_type": "invocation_span",
            "n_subjects": 101,
            "selection_rule": (
                "all exact reusable Phase 1 invocation spans in the active "
                "materialized generation; no sample and no supplement"
            ),
            "active_generation_id": generation_id,
            "active_generation_manifest_sha256": ledger.sha256_file(
                generation_path / "generation.json"
            ),
        },
        "candidate_spec": {
            "label_type": spec["label_type"],
            "spec_version": spec["spec_version"],
            "spec_sha256": spec["spec_sha256"],
            "prompt_sha256": spec["prompt_sha256"],
            "response_schema_sha256": spec["response_schema_sha256"],
        },
        "candidate_bundle": {
            "bundle_id": bundle["bundle_id"],
            "bundle_version": bundle["bundle_version"],
            "bundle_sha256": bundle["bundle_sha256"],
            "prompt_sha256": bundle["prompt_sha256"],
            "response_schema_sha256": bundle[
                "response_schema_sha256"
            ],
        },
        "reference_design": {
            "reviewer": "fresh_restricted_delegated_subject_matter_reviewer",
            "reasoning_effort": "medium",
            "labels": ["R", "C", "N"],
            "candidate_current_legacy_opus_and_historical_labels_hidden": True,
            "reference_sealed_before_candidate_exposure": True,
            "candidate_pass": (
                "one separate fresh restricted medium-effort pass over the "
                "same exact 101 spans, blind to reference and legacy values"
            ),
            "adjudication": (
                "append every reference/candidate disagreement after reference "
                "seal; preserve the original blind reference unchanged"
            ),
        },
        "metrics": {
            "structural": [
                "exact subject coverage",
                "accepted schema/event validity",
                "duplicate, unknown, incomplete, and drift counts",
                "blindness receipt integrity",
            ],
            "reference_accuracy": [
                "overall accuracy",
                "Cohen kappa",
                "supported-class macro F1",
                "per-supported-class precision and recall",
            ],
            "historical_diagnostic_only": [
                "exact agreement and kappa against the 101 legacy values"
            ],
            "intervals": {
                "draws": 2000,
                "unit": "speech",
                "seed": (
                    "annotation-refresh-campaign-v1|"
                    "invocation-tone-audit|20260725"
                ),
            },
        },
        "proposed_gates": {
            "exact_subject_coverage": 1.0,
            "accepted_event_validity": 1.0,
            "compromised_blindness_receipts": 0,
            "overall_accuracy_at_least": 0.90,
            "cohen_kappa_at_least": 0.80,
            "supported_class_floor": 10,
            "supported_class_macro_f1_at_least": 0.85,
            "each_supported_class_precision_at_least": 0.80,
            "each_supported_class_recall_at_least": 0.80,
            "accuracy_one_sided_95_lower_at_least": 0.85,
            "macro_f1_one_sided_95_lower_at_least": 0.75,
            "zero_denominator_or_inconclusive_bound": "fail_to_pass",
        },
        "historical_rule": (
            "legacy agreement is diagnostic only and cannot substitute for "
            "independent reference accuracy"
        ),
        "freeze_rule": (
            "only after every approved gate passes, every disagreement is "
            "adjudicated, and final prompt/schema bytes equal the candidate"
        ),
        "authorization_boundary": (
            "publishing this proposal does not authorize reference labeling, "
            "candidate labeling, adjudication, freezing, or full-shadow work"
        ),
    }
    contract_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    return {
        **semantic,
        "contract_id": (
            "inv_audit_" + contract_sha.removeprefix("sha256:")
        ),
        "contract_sha256": contract_sha,
    }


def publish_proposed_invocation_audit_contract(
    contract: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    value = dict(contract)
    contract_id = value.pop("contract_id", "")
    declared_sha = value.pop("contract_sha256", "")
    observed = ledger.sha256_text(ledger.canonical_json(value))
    if (
        declared_sha != observed
        or contract_id
        != "inv_audit_" + observed.removeprefix("sha256:")
        or value.get("labeling_authorized") is not False
        or value.get("contract_status")
        != "proposed_requires_explicit_owner_approval"
    ):
        raise ValueError("invocation audit proposal identity or boundary drift")
    persisted = {
        **value,
        "contract_id": contract_id,
        "contract_sha256": declared_sha,
    }
    path = (
        Path(root)
        / "proposed_contracts"
        / f"{contract_id}.json"
    )
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != persisted:
            raise ValueError("existing invocation audit proposal drift")
        status = "already_published"
    else:
        ledger.atomic_write_json(path, persisted)
        status = "published"
    return {
        "status": status,
        "contract_id": contract_id,
        "contract_sha256": declared_sha,
        "path": str(path),
        "labeling_authorized": False,
        "freeze_authorized": False,
    }
