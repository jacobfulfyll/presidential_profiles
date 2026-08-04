"""Non-executable planning for a provisional corpus-wide Sol evidence layer.

This module validates sealed ARCV1 evaluation evidence, reuses the final values
for the evaluated 240-paragraph reference, derives the remaining current
canonical paragraph universe by key, and publishes only three content-addressed
planning artifacts.  It cannot initialize a campaign or run, issue an
assignment, ingest a response, promote labels, or materialize a generation.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import annotation_ledger as ledger
from . import annotation_refresh as refresh
from . import annotation_refresh_planning as pilot_planning
from . import annotation_refresh_stage3 as stage3


PROVISIONAL_REUSE_VERSION = "annotation-refresh-provisional-reuse-v1"
PROVISIONAL_SELECTION_VERSION = "annotation-refresh-provisional-selection-v1"
PROVISIONAL_PLAN_VERSION = "annotation-refresh-provisional-corpus-plan-v1"
PROVISIONAL_BATCH_VERSION = "annotation-refresh-provisional-batches-v1"

PLANNING_APPROVAL_ID = "ARCV1-RES012"
CAMPAIGN_ID = (
    "camp_7cb35b8b739bcc2fb36ebe9609e81d629fee97c2d604cad5bd64d7cc25b5b316"
)
EVALUATION_ID = (
    "eval_bbb2212dea6936070b23bff11100615c602396b72d2fe6058cc707864f060b61"
)
EFFECTIVE_POLICY_SHA256 = (
    "sha256:43c4b5c1ff7bdb6d43c7586b1413e5db8ef6df8b08419a815b52d689b17098ae"
)
REFERENCE_RUN_ID = (
    "reference-ac9a418e8e50a2de826d48628394f455ea3dff90c050c415c85e2a1b4c1e65d1"
)
ADJUDICATION_RUN_ID = (
    "adjudication-72502897c022dbde22eab45e9cb90edd40b6c69cf61a22625cab2420b32f9f55"
)
COMBINED_BUNDLE_ID = "paragraph_judgment_v2_candidate"
COMBINED_BUNDLE_VERSION = "candidate-1"
REQUIRED_MODEL_ID = "gpt-5.6-sol"
REQUIRED_REASONING_EFFORT = "high"
EXPECTED_LABEL_TYPES = (
    "constituencies",
    "enemy_naming",
    "entities",
    "party_attack",
    "proposal_values",
    "topics",
    "zero_sum",
)
EXPECTED_REFERENCE_SUBJECTS = 240
EXPECTED_LABELS_PER_SUBJECT = 7
EXPECTED_REUSE_ENTRIES = 1_680
EXPECTED_UNANIMOUS = 972
EXPECTED_ADJUDICATED = 708
EXPECTED_PILOT_SUBJECTS = 722
EXPECTED_OTHER_PILOT_SUBJECTS = 482

REUSE_MANIFESTS_DIR = "provisional_corpus_reuse_manifests"
FRESH_SELECTIONS_DIR = "provisional_corpus_selections"
PROVISIONAL_PLANS_DIR = "provisional_corpus_plans"


def _artifact(
    semantic: Mapping[str, Any],
    *,
    prefix: str,
    id_field: str,
    sha_field: str,
) -> dict[str, Any]:
    sha256 = ledger.sha256_text(ledger.canonical_json(semantic))
    return {
        **dict(semantic),
        id_field: f"{prefix}_" + sha256.removeprefix("sha256:"),
        sha_field: sha256,
    }


def _validate_artifact(
    value: Mapping[str, Any],
    *,
    prefix: str,
    id_field: str,
    sha_field: str,
) -> dict[str, Any]:
    semantic = dict(value)
    artifact_id = str(semantic.pop(id_field, ""))
    declared_sha = str(semantic.pop(sha_field, ""))
    observed_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    if (
        declared_sha != observed_sha
        or artifact_id != f"{prefix}_" + observed_sha.removeprefix("sha256:")
    ):
        raise ValueError(f"{id_field}: content-addressed artifact drift")
    return dict(value)


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"content-addressed artifact drift: {path}")
        return "already_published"
    ledger.atomic_write_json(path, value)
    return "published"


def _source_record(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": source["run_id"],
        "run_artifact_set_sha256": source["artifact_set_sha256"],
        "label_group_id": row["label_group_id"],
        "value_sha256": row["value_sha256"],
        "spec_sha256": row["spec_sha256"],
    }


def _decision_rows(root: Path) -> list[dict[str, Any]]:
    return ledger._read_decisions(  # type: ignore[attr-defined]
        "adjudications", "label_adjudications.jsonl", root
    )


def _validate_final_value(label_type: str, value_json: str) -> Any:
    value = json.loads(value_json)
    if value is None:
        raise ValueError(f"{label_type}: final value is unresolved")
    if (
        label_type == "constituencies"
        and isinstance(value, dict)
        and value.get("outcome") == "unclear"
    ):
        raise ValueError("constituency final value remains unclear")
    return value


def _resolution_relation(
    source_values: Mapping[str, str],
    resolved_value: str,
) -> tuple[str, list[str], list[str]]:
    counts = Counter(source_values.values())
    majority_value = next(
        (value for value, count in counts.items() if count >= 2), None
    )
    majority_roles = sorted(
        role
        for role, value in source_values.items()
        if majority_value is not None and value == majority_value
    )
    matched_roles = sorted(
        role for role, value in source_values.items() if value == resolved_value
    )
    if majority_value is not None and resolved_value == majority_value:
        classification = "adjudicated_matches_majority"
    elif "reference" in matched_roles:
        classification = "adjudicated_matches_reference"
    elif matched_roles:
        classification = "adjudicated_matches_single_source"
    else:
        classification = "adjudicated_novel_synthesis"
    return classification, matched_roles, majority_roles


def _load_evidence(root: Path) -> dict[str, Any]:
    root = Path(root)
    campaign = refresh.load_campaign(CAMPAIGN_ID, root=root)
    if campaign["corpus_snapshot"]["active_generation_id"] != (
        root / "materialized" / "current"
    ).read_text(encoding="utf-8").strip():
        raise ValueError("active Phase 1 generation drift")
    roles = {
        str(row["pass_role"]): str(row["run_id"])
        for row in campaign["child_runs"]
    }
    if set(roles) != {"composition_diagnostic", "blind_a", "blind_b"}:
        raise ValueError("campaign child-role set drift")

    evaluation = stage3._load_evaluation(EVALUATION_ID, root=root)
    if (
        evaluation["campaign_id"] != CAMPAIGN_ID
        or evaluation["effective_evaluation_policy_sha256"]
        != EFFECTIVE_POLICY_SHA256
        or evaluation["reference_manifest_sha256"]
        != refresh.load_independent_reference(
            REFERENCE_RUN_ID, root=root
        )["reference_manifest_sha256"]
        or evaluation["adjudication_run_id"] != ADJUDICATION_RUN_ID
    ):
        raise ValueError("evaluation evidence identity drift")

    comparisons = []
    for receipt in evaluation["comparison_sources"]:
        path = root / "campaign_comparisons" / f"{receipt['comparison_id']}.json"
        comparison = _validate_artifact(
            json.loads(path.read_text(encoding="utf-8")),
            prefix="cmp",
            id_field="comparison_id",
            sha_field="comparison_sha256",
        )
        if any(comparison[field] != receipt[field] for field in receipt):
            raise ValueError("evaluation comparison receipt drift")
        comparisons.append(dict(receipt))

    adjudication = stage3.load_reference_adjudication(
        ADJUDICATION_RUN_ID, root=root
    )
    if (
        adjudication["campaign_id"] != CAMPAIGN_ID
        or adjudication["reference_run_id"] != REFERENCE_RUN_ID
        or adjudication["n_disputes"] != EXPECTED_ADJUDICATED
        or evaluation["adjudication_manifest_sha256"]
        != adjudication["adjudication_manifest_sha256"]
    ):
        raise ValueError("adjudication evidence identity drift")

    source_ids = {
        "reference": REFERENCE_RUN_ID,
        "blind_a": roles["blind_a"],
        "blind_b": roles["blind_b"],
        "adjudication": ADJUDICATION_RUN_ID,
    }
    sources = {
        role: stage3._sealed_source(run_id, root)
        for role, run_id in source_ids.items()
    }
    groups = {
        role: stage3._run_groups(run_id, root=root)
        for role, run_id in source_ids.items()
    }
    return {
        "campaign": campaign,
        "evaluation": evaluation,
        "comparisons": comparisons,
        "adjudication": adjudication,
        "sources": sources,
        "groups": groups,
        "roles": roles,
    }


def build_reuse_manifest(
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Build and validate the exact 1,680-field evaluated reuse manifest."""
    root = Path(root)
    evidence = _load_evidence(root)
    groups = evidence["groups"]
    reference = groups["reference"]
    blind_a = groups["blind_a"]
    blind_b = groups["blind_b"]
    resolved = groups["adjudication"]

    reference_keys = set(reference)
    reference_subjects = {key[0] for key in reference_keys}
    by_subject: dict[str, set[str]] = defaultdict(set)
    for subject_id, label_type in reference_keys:
        by_subject[subject_id].add(label_type)
    if (
        len(reference_subjects) != EXPECTED_REFERENCE_SUBJECTS
        or len(reference_keys) != EXPECTED_REUSE_ENTRIES
        or any(
            labels != set(EXPECTED_LABEL_TYPES)
            for labels in by_subject.values()
        )
    ):
        raise ValueError("reference is not exactly 240 subjects by seven labels")
    if reference_keys - set(blind_a) or reference_keys - set(blind_b):
        raise ValueError("blind passes do not cover the complete reference")

    declared_disputes = {
        (row["canonical_subject_id"], row["label_type"]): row
        for row in evidence["adjudication"]["disputes"]
    }
    if len(declared_disputes) != EXPECTED_ADJUDICATED:
        raise ValueError("adjudication dispute key set is duplicate or incomplete")

    decisions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    resolved_group_ids = {
        row["label_group_id"] for row in resolved.values()
    }
    for decision in _decision_rows(root):
        if decision.get("resolved_label_group_id") not in resolved_group_ids:
            continue
        key = (
            str(decision["canonical_subject_id"]),
            str(decision["label_type"]),
        )
        if key in decisions_by_key:
            raise ValueError("duplicate published adjudication for reuse key")
        decisions_by_key[key] = decision

    entries = []
    counts: Counter[str] = Counter()
    adjudication_ids = set()
    for key in sorted(reference_keys):
        source_rows = {
            "reference": reference[key],
            "blind_a": blind_a[key],
            "blind_b": blind_b[key],
        }
        if len({row["spec_sha256"] for row in source_rows.values()}) != 1:
            raise ValueError(f"source spec drift at {key}")
        source_values = {
            role: row["value_json"] for role, row in source_rows.items()
        }
        source_groups = {
            role: _source_record(row, evidence["sources"][role])
            for role, row in source_rows.items()
        }
        common = {
            "canonical_subject_id": key[0],
            "label_type": key[1],
            "spec_sha256": source_rows["reference"]["spec_sha256"],
            "source_groups": source_groups,
            "human_ground_truth_claim": False,
            "numeric_confidence_probability": None,
        }
        if len(set(source_values.values())) == 1:
            if key in declared_disputes:
                raise ValueError("unanimous key unexpectedly declared disputed")
            selected = source_rows["reference"]
            final_value = _validate_final_value(key[1], selected["value_json"])
            entry = {
                **common,
                "resolution_kind": "three_way_unanimous",
                "selected_source_role": "reference",
                "selected_label_group_id": selected["label_group_id"],
                "selected_value": final_value,
                "selected_value_sha256": selected["value_sha256"],
                "confidence_provenance": "three_way_same_model_unanimous",
            }
            counts["three_way_unanimous"] += 1
        else:
            if key not in declared_disputes:
                raise ValueError("disputed key is absent from adjudication scope")
            if key not in resolved or key not in decisions_by_key:
                raise ValueError("missing resolved or published adjudication")
            selected = resolved[key]
            decision = decisions_by_key[key]
            declared = declared_disputes[key]
            expected_candidate_groups = sorted(
                {row["label_group_id"] for row in source_rows.values()}
            )
            if (
                selected["spec_sha256"] != common["spec_sha256"]
                or decision["resolved_label_group_id"]
                != selected["label_group_id"]
                or json.loads(decision["candidate_label_group_ids_json"])
                != expected_candidate_groups
                or declared["candidate_label_group_ids"]
                != expected_candidate_groups
            ):
                raise ValueError(f"published adjudication provenance drift at {key}")
            final_value = _validate_final_value(key[1], selected["value_json"])
            classification, matched_roles, majority_roles = _resolution_relation(
                source_values, selected["value_json"]
            )
            entry = {
                **common,
                "resolution_kind": "adjudicated_final",
                "selected_source_role": "adjudication",
                "selected_label_group_id": selected["label_group_id"],
                "selected_value": final_value,
                "selected_value_sha256": selected["value_sha256"],
                "resolved_adjudication_group_id": selected["label_group_id"],
                "published_adjudication_id": decision["adjudication_id"],
                "published_adjudication_artifact_set_sha256": decision[
                    "artifact_set_sha256"
                ],
                "adjudication_run_id": ADJUDICATION_RUN_ID,
                "adjudication_run_artifact_set_sha256": evidence["sources"][
                    "adjudication"
                ]["artifact_set_sha256"],
                "confidence_provenance": classification,
                "matched_source_roles": matched_roles,
                "majority_source_roles": majority_roles,
            }
            counts["adjudicated_final"] += 1
            adjudication_ids.add(decision["adjudication_id"])
        entries.append(entry)

    if (
        len(entries) != EXPECTED_REUSE_ENTRIES
        or counts["three_way_unanimous"] != EXPECTED_UNANIMOUS
        or counts["adjudicated_final"] != EXPECTED_ADJUDICATED
        or len(adjudication_ids) != EXPECTED_ADJUDICATED
        or adjudication_ids
        != set(evidence["evaluation"]["adjudication_ids"])
        or set(declared_disputes)
        != {
            (row["canonical_subject_id"], row["label_type"])
            for row in entries
            if row["resolution_kind"] == "adjudicated_final"
        }
    ):
        raise ValueError("derived reuse classification count drift")

    semantic = {
        "reuse_manifest_version": PROVISIONAL_REUSE_VERSION,
        "scope": "provisional-corpus",
        "status": "immutable_evaluated_reuse_evidence",
        "campaign_id": CAMPAIGN_ID,
        "evaluation_id": EVALUATION_ID,
        "effective_evaluation_policy_sha256": EFFECTIVE_POLICY_SHA256,
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_manifest_sha256": evidence["evaluation"][
            "reference_manifest_sha256"
        ],
        "adjudication_run_id": ADJUDICATION_RUN_ID,
        "adjudication_manifest_sha256": evidence["adjudication"][
            "adjudication_manifest_sha256"
        ],
        "source_run_seals": evidence["sources"],
        "counts": {
            "subjects": len(reference_subjects),
            "label_types_per_subject": EXPECTED_LABELS_PER_SUBJECT,
            "subject_label_entries": len(entries),
            "three_way_unanimous": counts["three_way_unanimous"],
            "adjudicated_final": counts["adjudicated_final"],
            "missing_duplicate_unclear_or_unresolved": 0,
        },
        "confidence_policy": {
            "categorical_evidence_only": True,
            "numerical_probability_without_calibration": "prohibited",
            "human_ground_truth_claim": False,
        },
        "entries": entries,
    }
    return _artifact(
        semantic,
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )


def _bundle_receipt(
    bundle: Mapping[str, Any],
    *,
    bundle_registry_path: Path,
    label_registry_path: Path,
) -> dict[str, Any]:
    bundle_registry = refresh.read_bundle_registry(
        bundle_registry_path, label_registry_path=label_registry_path
    )
    label_registry = ledger.read_registry(label_registry_path)
    return {
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "bundle_sha256": bundle["bundle_sha256"],
        "eligibility": bundle["eligibility"],
        "prompt_version": bundle["prompt_version"],
        "prompt_sha256": ledger.sha256_text(str(bundle["prompt_text"])),
        "response_schema_sha256": ledger.sha256_text(
            ledger.canonical_json(bundle["response_schema"])
        ),
        "context_policy_sha256": ledger.sha256_text(
            ledger.canonical_json(bundle["context_policy"])
        ),
        "render_policy_sha256": ledger.sha256_text(
            ledger.canonical_json(
                pilot_planning.SHARED_CONTEXT_RENDER_POLICY
            )
        ),
        "bundle_registry_sha256": bundle_registry["registry_sha256"],
        "label_registry_sha256": label_registry["registry_sha256"],
    }


def _validate_bundle_evidence(
    bundle: Mapping[str, Any],
    receipt: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if (
        bundle["bundle_id"] != COMBINED_BUNDLE_ID
        or bundle["bundle_version"] != COMBINED_BUNDLE_VERSION
        or bundle["eligibility"] != "pilot_only"
    ):
        raise ValueError("provisional planner requires the unchanged pilot bundle")
    campaign_bundle = next(
        (
            row
            for row in evidence["campaign"]["bundles"]
            if row["bundle_id"] == COMBINED_BUNDLE_ID
        ),
        None,
    )
    if campaign_bundle is None or (
        campaign_bundle["bundle_version"],
        campaign_bundle["bundle_sha256"],
        campaign_bundle["model_receipt"]["model_id"],
        campaign_bundle["reasoning_effort"],
    ) != (
        COMBINED_BUNDLE_VERSION,
        receipt["bundle_sha256"],
        REQUIRED_MODEL_ID,
        REQUIRED_REASONING_EFFORT,
    ):
        raise ValueError("campaign bundle/model/effort drift")
    source_plan = refresh._validate_content_addressed_artifact(
        json.loads(
            (
                root
                / "campaign_plans"
                / f"{evidence['campaign']['source_plan_id']}.json"
            ).read_text(encoding="utf-8")
        ),
        id_field="plan_id",
        sha_field="plan_sha256",
        id_prefix="plan_",
    )
    prior_receipt = next(
        row
        for row in source_plan["bundle_receipts"]
        if row["bundle_id"] == COMBINED_BUNDLE_ID
    )
    for field in (
        "bundle_id",
        "bundle_version",
        "bundle_sha256",
        "prompt_sha256",
        "response_schema_sha256",
        "context_policy_sha256",
    ):
        if receipt[field] != prior_receipt[field]:
            raise ValueError(f"registered bundle {field} drift")
    if (
        receipt["render_policy_sha256"]
        != evidence["campaign"]["render_policy_sha256"]
        or source_plan["render_policy_sha256"]
        != receipt["render_policy_sha256"]
    ):
        raise ValueError("render policy drift")
    return source_plan


def _universe_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame, snapshot = pilot_planning._corpus_snapshot()
    complete, tails = pilot_planning._blocks(frame)
    rows = []
    for block in [*complete, *tails]:
        for subject in block["subjects"]:
            subject_key = pilot_planning._key(subject)
            subject_id = ledger.subject_identity("paragraph", subject_key)
            rows.append(
                {
                    "canonical_subject_id": subject_id,
                    "subject_key": subject_key,
                    "doc_name": str(subject["doc_name"]),
                    "para_idx": int(subject["para_idx"]),
                    "text": str(subject["text"]),
                    "source_text_sha256": ledger.sha256_text(
                        str(subject["text"])
                    ),
                    "bundle_input_sha256": subject["bundle_input_sha256"],
                    "context_before_json": subject["context_before_json"],
                    "context_after_json": subject["context_after_json"],
                    "decade": str(subject["decade"]),
                    "era": str(subject["era"]),
                    "genre": str(subject["genre"]),
                }
            )
    rows.sort(key=lambda row: (row["doc_name"], row["para_idx"]))
    identities = {row["canonical_subject_id"] for row in rows}
    if len(rows) != len(identities):
        raise ValueError("eligible universe contains duplicate canonical subjects")
    return rows, snapshot


def _partition_fresh_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundle_receipt: Mapping[str, Any],
    corpus_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_speech: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speech[str(row["doc_name"])].append(dict(row))
    batches = []
    planned_rows = []
    for doc_name in sorted(by_speech):
        speech = sorted(by_speech[doc_name], key=lambda row: row["para_idx"])
        segments: list[list[dict[str, Any]]] = []
        for row in speech:
            if (
                not segments
                or row["para_idx"] != segments[-1][-1]["para_idx"] + 1
            ):
                segments.append([])
            segments[-1].append(row)
        for segment in segments:
            for start in range(0, len(segment), 4):
                targets = segment[start : start + 4]
                target_keys = [row["subject_key"] for row in targets]
                batch_id = ledger.content_id(
                    "batch",
                    {
                        "batch_version": PROVISIONAL_BATCH_VERSION,
                        "bundle_sha256": bundle_receipt["bundle_sha256"],
                        "corpus_fingerprint": corpus_fingerprint,
                        "target_keys": target_keys,
                    },
                )
                batch_rows = []
                for order, row in enumerate(targets, start=1):
                    planned = {
                        **row,
                        "planned_batch_id": batch_id,
                        "planned_batch_order": order,
                        "confidence_provenance": "fresh_single_pass",
                    }
                    planned_rows.append(planned)
                    batch_rows.append(planned)
                batches.append(
                    {
                        "planned_batch_id": batch_id,
                        "doc_name": doc_name,
                        "target_keys": target_keys,
                        "target_source_text_sha256": [
                            row["source_text_sha256"] for row in batch_rows
                        ],
                        "target_bundle_input_sha256": [
                            row["bundle_input_sha256"] for row in batch_rows
                        ],
                    }
                )
    return planned_rows, batches


def _token_receipt(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_subject = {
        row["canonical_subject_id"]: dict(row) for row in rows
    }
    total = 0
    for batch in batches:
        subjects = [
            rows_by_subject[
                ledger.subject_identity("paragraph", subject_key)
            ]
            for subject_key in batch["target_keys"]
        ]
        rendered = pilot_planning._rendered_assignment(
            bundle, {"subjects": subjects}
        )
        total += pilot_planning._proxy_tokens(
            json.dumps(
                rendered,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
    return {
        "estimator": "deterministic_ceil_utf8_bytes_divided_by_4",
        "runtime_tokenizer_available": False,
        "reasoning_and_internal_tokens_included": False,
        "render_serialization": "indent_2_sorted_keys_ensure_ascii_false",
        "fresh_subjects": len(rows),
        "fresh_assignments": len(batches),
        "fresh_input_tokens": total,
        "fresh_output_tokens": len(rows) * 160,
        "fresh_input_token_ceiling_10_percent": math.ceil(total * 1.10),
    }


def build_fresh_selection(
    reuse_manifest: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
    bundle_registry_path: Path = refresh.BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the exact fresh-key complement and its deterministic batches."""
    root = Path(root)
    reuse_manifest = _validate_artifact(
        reuse_manifest,
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    evidence = _load_evidence(root)
    bundle = refresh.resolve_bundle(
        COMBINED_BUNDLE_ID,
        COMBINED_BUNDLE_VERSION,
        registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    bundle_receipt = _bundle_receipt(
        bundle,
        bundle_registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    source_plan = _validate_bundle_evidence(
        bundle, bundle_receipt, evidence, root=root
    )

    universe_rows, corpus_snapshot = _universe_rows()
    if corpus_snapshot != evidence["campaign"]["corpus_snapshot"]:
        raise ValueError("current corpus snapshot differs from locked campaign")
    universe_by_id = {
        row["canonical_subject_id"]: row for row in universe_rows
    }
    reuse_ids = {
        row["canonical_subject_id"] for row in reuse_manifest["entries"]
    }
    if len(reuse_ids) != EXPECTED_REFERENCE_SUBJECTS:
        raise ValueError("reuse manifest does not contain exactly 240 subjects")
    if reuse_ids - set(universe_by_id):
        raise ValueError("reuse manifest contains stale or out-of-universe keys")

    fresh_ids = set(universe_by_id) - reuse_ids
    if reuse_ids & fresh_ids or reuse_ids | fresh_ids != set(universe_by_id):
        raise ValueError("reuse/fresh union and disjointness validation failed")
    fresh_rows = [
        universe_by_id[subject_id] for subject_id in fresh_ids
    ]
    fresh_rows.sort(key=lambda row: (row["doc_name"], row["para_idx"]))

    combined_receipt = source_plan["selection_receipts"]["combined"]
    combined_path = (
        root
        / "selections"
        / f"{combined_receipt['selection_id']}.json"
    )
    combined = refresh._validate_selection_artifact(
        json.loads(combined_path.read_text(encoding="utf-8")),
        combined_receipt,
    )
    pilot_ids = {
        ledger.subject_identity("paragraph", row["subject_key"])
        for row in combined["identity_rows"]
    }
    other_pilot_ids = pilot_ids - reuse_ids
    if (
        len(pilot_ids) != EXPECTED_PILOT_SUBJECTS
        or len(other_pilot_ids) != EXPECTED_OTHER_PILOT_SUBJECTS
        or not other_pilot_ids <= fresh_ids
    ):
        raise ValueError("the other 482 pilot paragraphs are not all fresh")

    planned_rows, batches = _partition_fresh_rows(
        fresh_rows,
        bundle_receipt=bundle_receipt,
        corpus_fingerprint=corpus_snapshot["fingerprint"],
    )
    if (
        len(planned_rows) != len(fresh_ids)
        or len({row["canonical_subject_id"] for row in planned_rows})
        != len(planned_rows)
        or any(
            row["source_text_sha256"]
            != ledger.sha256_text(row["text"])
            for row in planned_rows
        )
    ):
        raise ValueError("fresh selection fingerprint or key validation failed")
    token_receipt = _token_receipt(bundle, planned_rows, batches)
    semantic = {
        "selection_manifest_version": PROVISIONAL_SELECTION_VERSION,
        "scope": "provisional-corpus",
        "status": "non_executable_fresh_target_selection",
        "reuse_manifest_id": reuse_manifest["reuse_manifest_id"],
        "reuse_manifest_sha256": reuse_manifest["reuse_manifest_sha256"],
        "bundle_receipt": bundle_receipt,
        "corpus_snapshot": corpus_snapshot,
        "n_eligible_subjects": len(universe_rows),
        "n_reused_subjects": len(reuse_ids),
        "n_subjects": len(planned_rows),
        "n_assignments": len(batches),
        "other_pilot_subjects_included": len(other_pilot_ids),
        "partition_version": PROVISIONAL_BATCH_VERSION,
        "identity_rows": [
            {
                "canonical_subject_id": row["canonical_subject_id"],
                "subject_key": row["subject_key"],
                "source_text_sha256": row["source_text_sha256"],
                "bundle_input_sha256": row["bundle_input_sha256"],
            }
            for row in planned_rows
        ],
        "rows": planned_rows,
        "batches": batches,
        "validation": {
            "reuse_and_fresh_disjoint": True,
            "reuse_and_fresh_union_complete": True,
            "duplicate_stale_or_out_of_universe_keys": 0,
            "excluded_by_canonical_identity_not_position": True,
            "other_482_pilot_paragraphs_included": True,
            "current_corpus_matches_locked_snapshot": True,
            "active_phase1_generation_matches_locked_snapshot": True,
        },
    }
    selection = _artifact(
        semantic,
        prefix="fresh",
        id_field="fresh_selection_id",
        sha_field="fresh_selection_sha256",
    )
    return selection, token_receipt, bundle_receipt


def build_provisional_plan(
    reuse_manifest: Mapping[str, Any],
    fresh_selection: Mapping[str, Any],
    token_receipt: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    *,
    approval_id: str,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Build the locked, explicitly non-executable composite-layer plan."""
    if approval_id != PLANNING_APPROVAL_ID:
        raise ValueError(
            f"provisional planning requires {PLANNING_APPROVAL_ID}"
        )
    reuse_manifest = _validate_artifact(
        reuse_manifest,
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    fresh_selection = _validate_artifact(
        fresh_selection,
        prefix="fresh",
        id_field="fresh_selection_id",
        sha_field="fresh_selection_sha256",
    )
    evidence = _load_evidence(Path(root))
    semantic = {
        "provisional_plan_version": PROVISIONAL_PLAN_VERSION,
        "scope": "provisional-corpus",
        "status": "non_executable_planning_only",
        "layer_status": "provisional",
        "production_eligible": False,
        "approval_id": approval_id,
        "reuse_manifest": {
            "reuse_manifest_id": reuse_manifest["reuse_manifest_id"],
            "reuse_manifest_sha256": reuse_manifest[
                "reuse_manifest_sha256"
            ],
            "subjects": reuse_manifest["counts"]["subjects"],
            "subject_label_entries": reuse_manifest["counts"][
                "subject_label_entries"
            ],
        },
        "fresh_selection": {
            "fresh_selection_id": fresh_selection["fresh_selection_id"],
            "fresh_selection_sha256": fresh_selection[
                "fresh_selection_sha256"
            ],
            "n_subjects": fresh_selection["n_subjects"],
            "n_assignments": fresh_selection["n_assignments"],
            "partition_version": fresh_selection["partition_version"],
        },
        "bundle_receipt": dict(bundle_receipt),
        "required_execution_profile": {
            "model_id": REQUIRED_MODEL_ID,
            "model_identity_specificity": "exact_runtime_identifier",
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
            "fresh_single_pass": True,
            "new_runtime_receipt_required": True,
        },
        "corpus_snapshot": fresh_selection["corpus_snapshot"],
        "evidence_identities": {
            "campaign_id": CAMPAIGN_ID,
            "evaluation_id": EVALUATION_ID,
            "evaluation_sha256": evidence["evaluation"][
                "evaluation_sha256"
            ],
            "effective_evaluation_policy_sha256": EFFECTIVE_POLICY_SHA256,
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_manifest_sha256": evidence["evaluation"][
                "reference_manifest_sha256"
            ],
            "adjudication_run_id": ADJUDICATION_RUN_ID,
            "adjudication_manifest_sha256": evidence["adjudication"][
                "adjudication_manifest_sha256"
            ],
            "adjudication_dispute_scope_sha256": evidence["adjudication"][
                "dispute_scope_sha256"
            ],
            "comparisons": evidence["comparisons"],
            "source_run_seals": evidence["sources"],
        },
        "token_receipt": dict(token_receipt),
        "later_assembly_rule": {
            "evaluated_240_source": "reuse_manifest_final_values",
            "all_other_paragraphs_source": "future_fresh_sol_run",
            "join_keys": [
                "canonical_subject_id",
                "label_type",
            ],
            "raw_and_source_layers_immutable": True,
            "reuse_and_fresh_provenance_remain_distinct": True,
            "current_primary_labels_unchanged": True,
            "promotion_or_active_materialization_pointer_change": "prohibited",
            "layer_eligibility": "provisional_or_experimental_only",
            "cross_layer_analysis": [
                "existing_sonnet",
                "existing_opus_subset",
                "sol_evaluated_subset",
                "sol_adjudication",
                "future_sol_corpus_layer",
            ],
            "collapse_distinct_evidence_layers": "prohibited",
        },
        "confidence_metadata": {
            "categorical_only": True,
            "allowed_initial_classes": [
                "three_way_unanimous",
                "adjudicated_matches_majority",
                "adjudicated_matches_reference",
                "adjudicated_matches_single_source",
                "adjudicated_novel_synthesis",
                "fresh_single_pass",
            ],
            "later_cross_model_agreement_classes_allowed": True,
            "uncalibrated_numeric_probability": "prohibited",
        },
        "authorization_boundary": {
            "separate_owner_authorization_required_before_initialization": True,
            "separate_owner_authorization_required_before_execution": True,
            "this_plan_authorizes_initialization": False,
            "this_plan_authorizes_execution": False,
        },
        "prohibited_effects": [
            "campaign_initialization",
            "run_initialization",
            "assignment_creation",
            "provider_or_model_invocation",
            "response_write",
            "new_run_seal",
            "adjudication_write",
            "promotion",
            "materialized_generation",
            "active_materialization_pointer_change",
            "site_generation",
            "deployment",
        ],
    }
    return _artifact(
        semantic,
        prefix="pcplan",
        id_field="plan_id",
        sha_field="plan_sha256",
    )


def create_provisional_corpus_plan(
    *,
    approval_id: str,
    root: Path = ledger.LEDGER_ROOT,
    publish_root: Path | None = None,
    bundle_registry_path: Path = refresh.BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    """Build, validate, and publish only the three planning artifacts."""
    if approval_id != PLANNING_APPROVAL_ID:
        raise ValueError(
            f"provisional planning requires {PLANNING_APPROVAL_ID}"
        )
    root = Path(root)
    destination = Path(publish_root) if publish_root is not None else root
    reuse_manifest = build_reuse_manifest(root=root)
    fresh_selection, token_receipt, bundle_receipt = build_fresh_selection(
        reuse_manifest,
        root=root,
        bundle_registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    plan = build_provisional_plan(
        reuse_manifest,
        fresh_selection,
        token_receipt,
        bundle_receipt,
        approval_id=approval_id,
        root=root,
    )

    paths = {
        "reuse_manifest": (
            destination
            / REUSE_MANIFESTS_DIR
            / f"{reuse_manifest['reuse_manifest_id']}.json"
        ),
        "fresh_selection": (
            destination
            / FRESH_SELECTIONS_DIR
            / f"{fresh_selection['fresh_selection_id']}.json"
        ),
        "plan": (
            destination
            / PROVISIONAL_PLANS_DIR
            / f"{plan['plan_id']}.json"
        ),
    }
    statuses = {
        "reuse_manifest": _publish(paths["reuse_manifest"], reuse_manifest),
        "fresh_selection": _publish(paths["fresh_selection"], fresh_selection),
        "plan": _publish(paths["plan"], plan),
    }
    return {
        "status": (
            "already_planned"
            if set(statuses.values()) == {"already_published"}
            else "planned_not_initialized"
        ),
        "publication_status": statuses,
        "reuse_manifest_id": reuse_manifest["reuse_manifest_id"],
        "reuse_manifest_sha256": reuse_manifest["reuse_manifest_sha256"],
        "fresh_selection_id": fresh_selection["fresh_selection_id"],
        "fresh_selection_sha256": fresh_selection[
            "fresh_selection_sha256"
        ],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "counts": {
            **reuse_manifest["counts"],
            "fresh_subjects": fresh_selection["n_subjects"],
            "fresh_assignments": fresh_selection["n_assignments"],
            "other_pilot_subjects_included": fresh_selection[
                "other_pilot_subjects_included"
            ],
        },
        "token_receipt": token_receipt,
        "paths": {name: str(path) for name, path in paths.items()},
    }
