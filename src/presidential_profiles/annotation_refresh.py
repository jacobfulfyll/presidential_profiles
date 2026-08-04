"""Provider-neutral bundle and refresh-campaign coordination.

This module validates declarative label bundles, selects eligible bundle sets,
constructs deterministic in-memory campaign manifests, and delegates individual
bundle runs to :mod:`presidential_profiles.annotation_workflow`.

Campaign initialization remains provider-neutral: it validates the approved plan,
review resolutions, corpus snapshot, selections, bundles, and exact runtime
receipt before atomically publishing a campaign commit marker over three
assignment-free child work runs. Model execution remains outside this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import annotation_ledger as ledger
from . import annotation_workflow as workflow


BUNDLES_ROOT = ledger.LEDGER_ROOT / "bundles"
BUNDLE_REGISTRY_PATH = BUNDLES_ROOT / "registry-v1.json"
CAMPAIGNS_ROOT = ledger.LEDGER_ROOT / "campaigns"
POLICY_AMENDMENTS_ROOT = ledger.LEDGER_ROOT / "campaign_policy_amendments"
REFERENCE_RUNS_ROOT = ledger.LEDGER_ROOT / "reference_runs"

BUNDLE_REGISTRY_VERSION = "annotation-label-bundle-registry-v1"
CAMPAIGN_MANIFEST_VERSION = "annotation-refresh-campaign-manifest-v1"
SELECTION_MANIFEST_VERSION = "annotation-refresh-selection-v1"
POLICY_AMENDMENT_VERSION = "annotation-refresh-evaluation-policy-amendment-v1"

FIRST_RESPONSE_REQUIREMENTS = [
    "13.1.first_response_schema_validity_metric",
    "14.1.first_response_assignment_schema_validity_at_least_0.99",
    "14.3.schema_validity_noninferiority_margin_minus_0.01",
    "14.4.schema_validity_noninferiority_margin_minus_0.01",
]
REFERENCE_BLINDNESS_REQUIREMENTS = {
    "fresh_restricted_reference_session": True,
    "candidate_responses_inspected": False,
    "current_labels_inspected": False,
    "legacy_labels_inspected": False,
    "opus_labels_inspected": False,
    "historical_response_artifacts_inspected": False,
    "sibling_reviewer_work_inspected": False,
}

SCOPE_MODES = {"pilot", "full-shadow", "delta", "audit", "explicit"}
BUNDLE_ELIGIBILITIES = {
    "production_eligible",
    "pilot_only",
    "pilot_diagnostic",
    "draft",
    "failed",
    "superseded",
}


def bundle_hash(bundle: Mapping[str, Any]) -> str:
    payload = dict(bundle)
    payload.pop("bundle_sha256", None)
    return ledger.sha256_text(ledger.canonical_json(payload))


def _relative_artifact_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a contained relative path")
    return path


def validate_bundle(
    bundle: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one bundle and its exact emitted atomic specifications."""
    validated = workflow._validate_execution_bundle(bundle, specs)
    if validated["eligibility"] not in BUNDLE_ELIGIBILITIES:
        raise ValueError("invalid bundle eligibility")
    batching = validated["batching_policy"]
    if not isinstance(batching, dict) or set(batching) != {
        "targets_per_assignment",
        "speech_mixing",
        "packing",
    }:
        raise ValueError("invalid bundle batching policy")
    target_count = batching["targets_per_assignment"]
    if (
        not isinstance(target_count, int)
        or isinstance(target_count, bool)
        or not 1 <= target_count <= 20
    ):
        raise ValueError("bundle target count must be between 1 and 20")
    if batching["speech_mixing"] not in {"prohibited", "not_applicable"}:
        raise ValueError("invalid speech-mixing policy")
    if batching["packing"] not in {
        "canonical_contiguous_same_speech",
        "one_subject",
        "canonical_same_speech",
    }:
        raise ValueError("invalid bundle packing policy")
    return validated


def read_bundle_registry(
    registry_path: Path = BUNDLE_REGISTRY_PATH,
    *,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    if set(registry) != {
        "bundle_registry_version",
        "label_registry_sha256",
        "entries",
        "registry_sha256",
    }:
        raise ValueError("bundle registry field mismatch")
    if registry["bundle_registry_version"] != BUNDLE_REGISTRY_VERSION:
        raise ValueError("unsupported bundle registry version")
    label_registry = ledger.read_registry(label_registry_path)
    if registry["label_registry_sha256"] != label_registry["registry_sha256"]:
        raise ValueError("bundle/label registry identity drift")
    specs = ledger.registered_specs(label_registry_path)
    identities: set[tuple[str, str]] = set()
    for entry in registry["entries"]:
        if not isinstance(entry, dict) or set(entry) != {
            "bundle_id",
            "bundle_version",
            "bundle_sha256",
            "eligibility",
            "path",
        }:
            raise ValueError("bundle registry entry field mismatch")
        identity = (entry["bundle_id"], entry["bundle_version"])
        if identity in identities:
            raise ValueError(f"duplicate bundle identity: {identity}")
        identities.add(identity)
        bundle_path = path.parent / _relative_artifact_path(
            entry["path"], "bundle path"
        )
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
        emitted_specs = []
        for emitted in raw.get("emitted_label_specs", []):
            spec_identity = (emitted["label_type"], emitted["spec_version"])
            if spec_identity not in specs:
                raise ValueError(
                    f"{identity}: unknown emitted label spec {spec_identity}"
                )
            spec = specs[spec_identity]
            emitted_specs.append(spec)
        bundle = validate_bundle(raw, emitted_specs)
        if (
            bundle["bundle_id"],
            bundle["bundle_version"],
            bundle["bundle_sha256"],
            bundle["eligibility"],
        ) != (
            entry["bundle_id"],
            entry["bundle_version"],
            entry["bundle_sha256"],
            entry["eligibility"],
        ):
            raise ValueError(f"bundle registry identity drift: {identity}")
    expected = ledger.sha256_text(
        ledger.canonical_json(
            {key: value for key, value in registry.items() if key != "registry_sha256"}
        )
    )
    if registry["registry_sha256"] != expected:
        raise ValueError("bundle registry hash drift")
    return registry


def registered_bundles(
    registry_path: Path = BUNDLE_REGISTRY_PATH,
    *,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[tuple[str, str], dict[str, Any]]:
    registry = read_bundle_registry(
        registry_path, label_registry_path=label_registry_path
    )
    root = Path(registry_path).parent
    return {
        (entry["bundle_id"], entry["bundle_version"]): json.loads(
            (
                root / _relative_artifact_path(entry["path"], "bundle path")
            ).read_text(encoding="utf-8")
        )
        for entry in registry["entries"]
    }


def resolve_bundle(
    bundle_id: str,
    bundle_version: str | None = None,
    *,
    registry_path: Path = BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    bundles = registered_bundles(
        registry_path, label_registry_path=label_registry_path
    )
    matches = [
        bundle
        for (candidate_id, candidate_version), bundle in bundles.items()
        if candidate_id == bundle_id
        and (bundle_version is None or candidate_version == bundle_version)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"bundle resolution for {bundle_id!r}/{bundle_version!r} "
            f"returned {len(matches)} candidates"
        )
    return matches[0]


def list_bundles(
    registry_path: Path = BUNDLE_REGISTRY_PATH,
    *,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "bundle_id": bundle["bundle_id"],
                "bundle_version": bundle["bundle_version"],
                "bundle_sha256": bundle["bundle_sha256"],
                "eligibility": bundle["eligibility"],
                "subject_type": bundle["subject_type"],
                "label_types": [
                    row["label_type"] for row in bundle["emitted_label_specs"]
                ],
            }
            for bundle in registered_bundles(
                registry_path, label_registry_path=label_registry_path
            ).values()
        ),
        key=lambda row: (row["bundle_id"], row["bundle_version"]),
    )


def select_bundles(
    selection: str | Sequence[tuple[str, str]],
    *,
    registry_path: Path = BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> list[dict[str, Any]]:
    bundles = registered_bundles(
        registry_path, label_registry_path=label_registry_path
    )
    if selection == "all":
        selected = [
            bundle
            for bundle in bundles.values()
            if bundle["eligibility"] == "production_eligible"
        ]
    else:
        if isinstance(selection, str):
            raise ValueError("explicit bundle selection requires (id, version) pairs")
        identities = list(selection)
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("explicit bundle selection must be non-empty and unique")
        unknown = set(identities) - set(bundles)
        if unknown:
            raise ValueError(f"unknown bundle selection: {sorted(unknown)}")
        selected = [bundles[identity] for identity in identities]
    return sorted(
        selected, key=lambda value: (value["bundle_id"], value["bundle_version"])
    )


def subject_selection_id(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    identities: set[str] = set()
    for row in rows:
        required = {
            "subject_key",
            "bundle_input_sha256",
            "stratum",
            "inclusion_probability",
        }
        if set(row) != required:
            raise ValueError("selection row field mismatch")
        identity = ledger.canonical_json(row["subject_key"])
        if identity in identities:
            raise ValueError("selection contains duplicate subject keys")
        identities.add(identity)
        if not ledger.HASH_RE.fullmatch(str(row["bundle_input_sha256"])):
            raise ValueError("selection contains invalid input fingerprint")
        probability = row["inclusion_probability"]
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0 < probability <= 1
        ):
            raise ValueError("selection probability must be in (0, 1]")
        normalized.append(dict(row))
    normalized.sort(key=lambda row: ledger.canonical_json(row["subject_key"]))
    return "sel_" + ledger.sha256_text(
        ledger.canonical_json(normalized)
    ).removeprefix("sha256:")


def select_scope(
    scope: str,
    eligible: Sequence[Mapping[str, Any]],
    *,
    persisted_selection: Sequence[Mapping[str, Any]] | None = None,
    prior_fingerprints: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return an exact deterministic key set without writing campaign state."""
    if scope not in SCOPE_MODES:
        raise ValueError(f"unknown campaign scope: {scope}")
    eligible_by_key = {
        ledger.canonical_json(row["subject_key"]): dict(row) for row in eligible
    }
    if len(eligible_by_key) != len(eligible):
        raise ValueError("eligible universe contains duplicate subject keys")
    if scope == "full-shadow":
        selected = list(eligible_by_key.values())
    elif scope == "delta":
        if prior_fingerprints is None:
            raise ValueError("delta scope requires prior fingerprints")
        selected = [
            row
            for key, row in eligible_by_key.items()
            if prior_fingerprints.get(key) != row["bundle_input_sha256"]
        ]
    else:
        if persisted_selection is None:
            raise ValueError(f"{scope} scope requires a persisted selection")
        requested = {
            ledger.canonical_json(row["subject_key"]) for row in persisted_selection
        }
        if len(requested) != len(persisted_selection):
            raise ValueError("persisted selection contains duplicate keys")
        unknown = requested - set(eligible_by_key)
        if unknown:
            raise ValueError("persisted selection contains out-of-universe keys")
        selected = [eligible_by_key[key] for key in requested]
    return sorted(selected, key=lambda row: ledger.canonical_json(row["subject_key"]))


def validate_model_receipt(
    receipt: Mapping[str, Any], *, required_effort: str
) -> dict[str, Any]:
    if str(receipt.get("model_id", "")).strip() == "":
        raise ValueError("model receipt has no exact model identifier")
    if receipt.get("specificity") != "exact_runtime_identifier":
        raise ValueError("model identity is not exact enough")
    if receipt.get("reasoning_effort") != required_effort:
        raise ValueError("model receipt reasoning effort mismatch")
    return dict(receipt)


def expand_campaign_children(
    campaign_hash: str,
    bundles: Sequence[Mapping[str, Any]],
    pass_roles: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    prefix = campaign_hash.removeprefix("sha256:")[:16]
    children = []
    for bundle in sorted(
        bundles, key=lambda value: (value["bundle_id"], value["bundle_version"])
    ):
        roles = list(pass_roles.get(bundle["bundle_id"], ["primary"]))
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("campaign pass roles must be non-empty and unique")
        for role in sorted(roles):
            ledger.validate_spec_part(role, "pass_role")
            children.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "bundle_version": bundle["bundle_version"],
                    "bundle_sha256": bundle["bundle_sha256"],
                    "pass_role": role,
                    "run_id": (
                        f"refresh-{prefix}-{bundle['bundle_id']}-{role}"
                    ),
                    "subject_type": bundle["subject_type"],
                }
            )
    return children


def build_campaign_manifest(
    *,
    corpus_snapshot: Mapping[str, Any],
    bundles: Sequence[Mapping[str, Any]],
    scope: str,
    selection_receipts: Mapping[str, Mapping[str, Any]],
    model_receipts: Mapping[str, Mapping[str, Any]],
    reasoning_effort: Mapping[str, str],
    pass_roles: Mapping[str, Sequence[str]] | None = None,
    evaluation_policy_sha256: str,
    promotion_policy_sha256: str,
    approval_ids: Sequence[str],
    source_plan_id: str | None = None,
    source_plan_sha256: str | None = None,
    render_policy_sha256: str | None = None,
    blindness: Mapping[str, Any] | None = None,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    """Build a deterministic executable manifest in memory.

    Persistence remains a separate, explicitly authorized operation.
    """
    if scope not in SCOPE_MODES:
        raise ValueError("invalid campaign scope")
    if not bundles:
        raise ValueError("campaign must contain at least one bundle")
    bundle_identities = [
        (bundle["bundle_id"], bundle["bundle_version"]) for bundle in bundles
    ]
    if len(bundle_identities) != len(set(bundle_identities)):
        raise ValueError("campaign bundles must be unique")
    if not approval_ids:
        raise ValueError("campaign requires explicit approval references")
    if (source_plan_id is None) != (source_plan_sha256 is None):
        raise ValueError("campaign source plan identity must be complete")
    if source_plan_id is not None:
        if (
            not source_plan_id.startswith("plan_")
            or not ledger.HASH_RE.fullmatch(str(source_plan_sha256))
            or source_plan_id != "plan_" + str(source_plan_sha256).removeprefix("sha256:")
        ):
            raise ValueError("campaign source plan identity is invalid")
    if render_policy_sha256 is not None and not ledger.HASH_RE.fullmatch(
        render_policy_sha256
    ):
        raise ValueError("campaign render policy hash is invalid")
    if not ledger.HASH_RE.fullmatch(str(corpus_snapshot.get("fingerprint", ""))):
        raise ValueError("campaign corpus fingerprint is invalid")
    for name, value in {
        "evaluation policy": evaluation_policy_sha256,
        "promotion policy": promotion_policy_sha256,
    }.items():
        if not ledger.HASH_RE.fullmatch(str(value)):
            raise ValueError(f"campaign {name} hash is invalid")
    registered_specs = ledger.registered_specs(label_registry_path)
    locked_bundles = []
    for bundle in sorted(
        bundles, key=lambda value: (value["bundle_id"], value["bundle_version"])
    ):
        bundle_id = bundle["bundle_id"]
        if scope == "full-shadow" and bundle["eligibility"] != "production_eligible":
            raise ValueError(
                f"{bundle_id}: non-production bundle cannot enter full-shadow"
            )
        if bundle_id not in selection_receipts:
            raise ValueError(f"{bundle_id}: missing selection receipt")
        effort = reasoning_effort.get(bundle_id)
        if not effort:
            raise ValueError(f"{bundle_id}: missing reasoning effort")
        receipt = validate_model_receipt(
            model_receipts.get(bundle_id, {}), required_effort=effort
        )
        selection = dict(selection_receipts[bundle_id])
        if not str(selection.get("selection_id", "")).startswith("sel_"):
            raise ValueError(f"{bundle_id}: invalid selection receipt")
        if not ledger.HASH_RE.fullmatch(str(selection.get("selection_sha256", ""))):
            raise ValueError(f"{bundle_id}: invalid selection hash")
        if (
            not isinstance(selection.get("n_subjects"), int)
            or isinstance(selection["n_subjects"], bool)
            or selection["n_subjects"] <= 0
        ):
            raise ValueError(f"{bundle_id}: invalid selection count")
        if bundle_id == "constituency_only_diagnostic" and (
            scope != "pilot" or selection["n_subjects"] != 144
        ):
            raise ValueError(
                "constituency diagnostic requires the 144-subject pilot selection"
            )
        label_specs = []
        for emitted in bundle["emitted_label_specs"]:
            identity = (emitted["label_type"], emitted["spec_version"])
            if identity not in registered_specs:
                raise ValueError(f"{bundle_id}: unknown atomic spec {identity}")
            atomic = registered_specs[identity]
            label_specs.append(
                {
                    "label_type": atomic["label_type"],
                    "spec_version": atomic["spec_version"],
                    "spec_sha256": atomic["spec_sha256"],
                }
            )
        locked_bundles.append(
            {
                "bundle_id": bundle_id,
                "bundle_version": bundle["bundle_version"],
                "bundle_sha256": bundle["bundle_sha256"],
                "subject_type": bundle["subject_type"],
                "label_specs": label_specs,
                "selection": selection,
                "model_receipt": receipt,
                "reasoning_effort": effort,
            }
        )
    locked_by_id = {row["bundle_id"]: row for row in locked_bundles}
    if {
        "paragraph_judgment_v2_candidate",
        "constituency_only_diagnostic",
    } <= set(locked_by_id):
        paragraph = locked_by_id["paragraph_judgment_v2_candidate"]
        diagnostic = locked_by_id["constituency_only_diagnostic"]
        if (
            paragraph["model_receipt"]["model_id"]
            != diagnostic["model_receipt"]["model_id"]
            or paragraph["reasoning_effort"] != diagnostic["reasoning_effort"]
        ):
            raise ValueError(
                "combined and diagnostic bundles require the same model and effort"
            )
        if diagnostic["selection"].get("parent_selection_id") != paragraph[
            "selection"
        ]["selection_id"]:
            raise ValueError(
                "diagnostic selection must identify the combined pilot selection"
            )
    semantic = {
        "campaign_manifest_version": CAMPAIGN_MANIFEST_VERSION,
        "corpus_snapshot": dict(corpus_snapshot),
        "scope": scope,
        "bundles": locked_bundles,
        "evaluation_policy_sha256": evaluation_policy_sha256,
        "promotion_policy_sha256": promotion_policy_sha256,
        "approval_ids": sorted(set(approval_ids)),
        "pass_roles": {
            bundle["bundle_id"]: sorted(
                (pass_roles or {}).get(bundle["bundle_id"], ["primary"])
            )
            for bundle in bundles
        },
    }
    if source_plan_id is not None:
        semantic["source_plan_id"] = source_plan_id
        semantic["source_plan_sha256"] = source_plan_sha256
    if render_policy_sha256 is not None:
        semantic["render_policy_sha256"] = render_policy_sha256
    if blindness is not None:
        semantic["blindness"] = dict(blindness)
    campaign_hash = ledger.sha256_text(ledger.canonical_json(semantic))
    children = expand_campaign_children(
        campaign_hash, bundles, semantic["pass_roles"]
    )
    return {
        **semantic,
        "campaign_id": "camp_" + campaign_hash.removeprefix("sha256:"),
        "campaign_identity_sha256": campaign_hash,
        "child_runs": children,
    }


def initialize_bundle_run(
    *,
    bundle_id: str,
    bundle_version: str,
    run_id: str,
    subjects_path: Path,
    model_receipt: Mapping[str, Any],
    reasoning_effort: str,
    annotator_id: str,
    root: Path = ledger.LEDGER_ROOT,
    bundle_registry_path: Path = BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
    **metadata: Any,
) -> dict[str, Any]:
    """Initialize one child run after its caller supplies later-stage authority."""
    bundle = resolve_bundle(
        bundle_id,
        bundle_version,
        registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    receipt = validate_model_receipt(
        model_receipt, required_effort=reasoning_effort
    )
    if not metadata.get("authorization_review_item_id"):
        raise ValueError("bundle run requires an explicit authorization review item")
    spec_versions = {
        row["label_type"]: row["spec_version"]
        for row in bundle["emitted_label_specs"]
    }
    return workflow.initialize_run(
        run_id=run_id,
        label_types=list(spec_versions),
        subjects_path=subjects_path,
        model_id=receipt["model_id"],
        model_identity_source="runtime_metadata",
        model_identity_receipt=receipt,
        annotator_id=annotator_id,
        reasoning_effort=reasoning_effort,
        reasoning_effort_provenance="runtime_metadata",
        spec_versions=spec_versions,
        execution_bundle=bundle,
        root=root,
        registry_path=label_registry_path,
        **metadata,
    )


def _validate_content_addressed_artifact(
    value: Mapping[str, Any],
    *,
    id_field: str,
    sha_field: str,
    id_prefix: str,
) -> dict[str, Any]:
    artifact = dict(value)
    artifact_id = str(artifact.pop(id_field, ""))
    declared_sha = str(artifact.pop(sha_field, ""))
    observed_sha = ledger.sha256_text(ledger.canonical_json(artifact))
    if (
        declared_sha != observed_sha
        or artifact_id != id_prefix + declared_sha.removeprefix("sha256:")
    ):
        raise ValueError(f"{id_field}: content-addressed artifact drift")
    return dict(value)


def _validate_selection_artifact(
    selection: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validated = dict(selection)
    semantic = dict(validated)
    semantic.pop("selection_id", None)
    declared_sha = str(semantic.pop("selection_sha256", ""))
    if declared_sha != ledger.sha256_text(ledger.canonical_json(semantic)):
        raise ValueError("selection_sha256: content-addressed artifact drift")
    if validated["selection_id"] != subject_selection_id(validated["identity_rows"]):
        raise ValueError("selection subject identity drift")
    for field in ["selection_id", "selection_sha256", "n_subjects"]:
        if validated.get(field) != receipt.get(field):
            raise ValueError(f"selection receipt mismatch: {field}")
    if len(validated.get("rows", [])) != validated["n_subjects"]:
        raise ValueError("selection row count mismatch")
    if receipt.get("n_assignments") is not None and (
        validated.get("n_assignments") != receipt["n_assignments"]
    ):
        raise ValueError("selection assignment count mismatch")
    return validated


def _validate_campaign_manifest_identity(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(manifest)
    child_runs = value.pop("child_runs")
    campaign_id = str(value.pop("campaign_id"))
    declared_sha = str(value.pop("campaign_identity_sha256"))
    observed_sha = ledger.sha256_text(ledger.canonical_json(value))
    if (
        declared_sha != observed_sha
        or campaign_id != "camp_" + declared_sha.removeprefix("sha256:")
    ):
        raise ValueError("campaign identity drift")
    expected_children = expand_campaign_children(
        declared_sha, value["bundles"], value["pass_roles"]
    )
    if child_runs != expected_children:
        raise ValueError("campaign child expansion drift")
    return dict(manifest)


def _campaign_path(campaign_id: str, root: Path) -> Path:
    if not ledger.ID_PREFIX_RE.fullmatch(campaign_id) or not campaign_id.startswith(
        "camp_"
    ):
        raise ValueError("invalid campaign_id")
    return Path(root) / "campaigns" / campaign_id


def load_campaign(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    path = _campaign_path(campaign_id, root) / "campaign.json"
    if not path.exists():
        raise FileNotFoundError(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    manifest = {
        key: value
        for key, value in persisted.items()
        if key
        not in {
            "campaign_manifest_sha256",
            "initialized_at",
            "initialized_by",
            "status",
        }
    }
    _validate_campaign_manifest_identity(manifest)
    expected_manifest_sha = persisted["campaign_manifest_sha256"]
    sha_payload = dict(persisted)
    sha_payload.pop("campaign_manifest_sha256")
    if expected_manifest_sha != ledger.sha256_text(
        ledger.canonical_json(sha_payload)
    ):
        raise ValueError("persisted campaign manifest hash drift")
    if persisted.get("status") != "initialized":
        raise ValueError("campaign is not in initialized state")
    return persisted


def publish_campaign_initialization(
    manifest: Mapping[str, Any],
    *,
    selection_paths: Mapping[str, Path],
    initialized_by: str,
    root: Path = ledger.LEDGER_ROOT,
    bundle_registry_path: Path = BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    """Publish one complete assignment-free campaign initialization.

    Child work directories are prepared under a hidden candidate. They become
    authoritative only when the campaign directory is atomically renamed as the
    final commit marker. Interrupted moves leave no campaign manifest; such work
    directories are orphan candidates and must be inspected before retrying.
    """
    manifest = _validate_campaign_manifest_identity(manifest)
    initialized_by = str(initialized_by).strip()
    if not initialized_by:
        raise ValueError("initialized_by is required")
    campaign_id = manifest["campaign_id"]
    final_campaign = _campaign_path(campaign_id, root)
    if final_campaign.exists():
        existing = load_campaign(campaign_id, root=root)
        if existing["child_runs"] != manifest["child_runs"]:
            raise ValueError("existing campaign child set drift")
        missing_children = [
            child["run_id"]
            for child in existing["child_runs"]
            if not ledger.work_dir(child["run_id"], root).is_dir()
        ]
        if missing_children:
            raise ValueError(
                f"initialized campaign is missing child runs: {missing_children}"
            )
        return {
            "status": "already_initialized",
            "campaign_id": campaign_id,
            "campaign_path": str(final_campaign),
            "child_runs": existing["child_runs"],
        }
    locked_bundles = {row["bundle_id"]: row for row in manifest["bundles"]}
    if set(selection_paths) != set(locked_bundles):
        raise ValueError("campaign selection path set mismatch")
    validated_selections: dict[str, dict[str, Any]] = {}
    for bundle_id, path in selection_paths.items():
        selection = json.loads(Path(path).read_text(encoding="utf-8"))
        validated_selections[bundle_id] = _validate_selection_artifact(
            selection, locked_bundles[bundle_id]["selection"]
        )
    candidate = (
        Path(root)
        / "campaigns"
        / f".{campaign_id}.{os.getpid()}.tmp"
    )
    if candidate.exists():
        raise FileExistsError(candidate)
    candidate_root = candidate / "candidate-ledger"
    try:
        for child in manifest["child_runs"]:
            locked = locked_bundles[child["bundle_id"]]
            run = initialize_bundle_run(
                bundle_id=child["bundle_id"],
                bundle_version=child["bundle_version"],
                run_id=child["run_id"],
                subjects_path=Path(selection_paths[child["bundle_id"]]),
                model_receipt=locked["model_receipt"],
                reasoning_effort=locked["reasoning_effort"],
                annotator_id=f"arcv1-{child['pass_role']}",
                root=candidate_root,
                bundle_registry_path=bundle_registry_path,
                label_registry_path=label_registry_path,
                canonical_corpus_fingerprint=manifest["corpus_snapshot"][
                    "fingerprint"
                ],
                selection_artifact_id=locked["selection"]["selection_id"],
                authorization_review_item_id="ARCV1-G006",
            )
            selection = validated_selections[child["bundle_id"]]
            if (
                run["n_subjects"] != selection["n_subjects"]
                or run["assigned_at"] is not None
                or list(
                    (
                        ledger.work_dir(child["run_id"], candidate_root)
                        / "assignments"
                    ).glob("*.json")
                )
            ):
                raise ValueError("child run was not initialized assignment-free")
        persisted = {
            **manifest,
            "initialized_at": ledger.utc_now(),
            "initialized_by": initialized_by,
            "status": "initialized",
        }
        persisted["campaign_manifest_sha256"] = ledger.sha256_text(
            ledger.canonical_json(persisted)
        )
        ledger.atomic_write_json(candidate / "campaign.json", persisted)
        with ledger.ledger_lock(root):
            if final_campaign.exists():
                raise FileExistsError(final_campaign)
            work_root = Path(root) / "work"
            work_root.mkdir(parents=True, exist_ok=True)
            for child in manifest["child_runs"]:
                final_run = ledger.work_dir(child["run_id"], root)
                if final_run.exists():
                    raise FileExistsError(
                        f"orphan or existing campaign child requires inspection: "
                        f"{final_run}"
                    )
            for child in manifest["child_runs"]:
                os.replace(
                    ledger.work_dir(child["run_id"], candidate_root),
                    ledger.work_dir(child["run_id"], root),
                )
            ledger.fsync_directory(work_root)
            (candidate_root / "work").rmdir()
            candidate_root.rmdir()
            os.replace(candidate, final_campaign)
            ledger.fsync_directory(final_campaign.parent)
    except Exception:
        # Preserve any candidate or orphan child directories for operator
        # inspection; automatic deletion would hide partial publication state.
        raise
    return {
        "status": "initialized_no_assignments",
        "campaign_id": campaign_id,
        "campaign_path": str(final_campaign),
        "child_runs": manifest["child_runs"],
    }


def initialize_approved_pilot(
    *,
    plan_path: Path,
    review_queue_path: Path,
    initialized_by: str,
    root: Path = ledger.LEDGER_ROOT,
    bundle_registry_path: Path = BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    plan = _validate_content_addressed_artifact(
        json.loads(Path(plan_path).read_text(encoding="utf-8")),
        id_field="plan_id",
        sha_field="plan_sha256",
        id_prefix="plan_",
    )
    queue = json.loads(Path(review_queue_path).read_text(encoding="utf-8"))
    resolutions = {
        row["resolution_id"]: row for row in queue.get("resolutions", [])
    }
    required_resolutions = {
        "ARCV1-RES006": "ARCV1-G002",
        "ARCV1-RES007": "ARCV1-G001",
        "ARCV1-RES008": "ARCV1-G006",
    }
    for resolution_id, item_id in required_resolutions.items():
        resolution = resolutions.get(resolution_id)
        if resolution is None or item_id not in resolution.get(
            "applies_to_items", []
        ):
            raise ValueError(f"missing required review resolution: {resolution_id}")
    receipt = validate_model_receipt(
        resolutions["ARCV1-RES007"].get("runtime_receipt", {}),
        required_effort="high",
    )
    if (
        resolutions["ARCV1-RES008"].get("decision")
        != "approved_exact_pilot_execution"
    ):
        raise ValueError("ARCV1-G006 pilot execution is not approved")
    from . import annotation_refresh_planning as planning

    _, live_snapshot = planning._corpus_snapshot()
    if live_snapshot != plan["corpus_snapshot"]:
        raise ValueError("approved plan corpus snapshot drift")
    bundles = [
        resolve_bundle(
            receipt_row["bundle_id"],
            receipt_row["bundle_version"],
            registry_path=bundle_registry_path,
            label_registry_path=label_registry_path,
        )
        for receipt_row in plan["bundle_receipts"]
    ]
    bundle_by_id = {bundle["bundle_id"]: bundle for bundle in bundles}
    for declared in plan["bundle_receipts"]:
        bundle = bundle_by_id[declared["bundle_id"]]
        observed = {
            "bundle_id": bundle["bundle_id"],
            "bundle_version": bundle["bundle_version"],
            "bundle_sha256": bundle["bundle_sha256"],
            "prompt_sha256": bundle["prompt_sha256"],
            "response_schema_sha256": bundle["response_schema_sha256"],
            "context_policy_sha256": ledger.sha256_text(
                ledger.canonical_json(bundle["context_policy"])
            ),
        }
        if observed != declared:
            raise ValueError(f"{bundle['bundle_id']}: approved bundle receipt drift")
    combined_id = "paragraph_judgment_v2_candidate"
    diagnostic_id = "constituency_only_diagnostic"
    if set(bundle_by_id) != {combined_id, diagnostic_id}:
        raise ValueError("approved pilot bundle set drift")
    selection_paths = {
        combined_id: Path(root)
        / "selections"
        / f"{plan['selection_receipts']['combined']['selection_id']}.json",
        diagnostic_id: Path(root)
        / "selections"
        / f"{plan['selection_receipts']['diagnostic']['selection_id']}.json",
    }
    for bundle_id, role in [
        (combined_id, "combined"),
        (diagnostic_id, "diagnostic"),
    ]:
        selection = json.loads(selection_paths[bundle_id].read_text(encoding="utf-8"))
        _validate_selection_artifact(selection, plan["selection_receipts"][role])
    approval_ids = sorted(
        {
            *plan["approval_ids"],
            "ARCV1-RES006",
            "ARCV1-RES007",
            "ARCV1-RES008",
        }
    )
    manifest = build_campaign_manifest(
        corpus_snapshot=live_snapshot,
        bundles=bundles,
        scope="pilot",
        selection_receipts={
            combined_id: plan["selection_receipts"]["combined"],
            diagnostic_id: plan["selection_receipts"]["diagnostic"],
        },
        model_receipts={combined_id: receipt, diagnostic_id: receipt},
        reasoning_effort={combined_id: "high", diagnostic_id: "high"},
        pass_roles={
            combined_id: ["blind_a", "blind_b"],
            diagnostic_id: ["composition_diagnostic"],
        },
        evaluation_policy_sha256=plan["evaluation_policy_sha256"],
        promotion_policy_sha256=plan["promotion_policy_sha256"],
        approval_ids=approval_ids,
        source_plan_id=plan["plan_id"],
        source_plan_sha256=plan["plan_sha256"],
        render_policy_sha256=plan["render_policy_sha256"],
        blindness=plan["blindness"],
        label_registry_path=label_registry_path,
    )
    expected_passes = {
        (row["bundle_id"], row["pass_role"]) for row in plan["passes"]
    }
    observed_passes = {
        (row["bundle_id"], row["pass_role"]) for row in manifest["child_runs"]
    }
    if observed_passes != expected_passes:
        raise ValueError("approved pilot pass expansion drift")
    return publish_campaign_initialization(
        manifest,
        selection_paths=selection_paths,
        initialized_by=initialized_by,
        root=root,
        bundle_registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )


def campaign_status(
    campaign_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    campaign = load_campaign(campaign_id, root=root)
    return {
        "campaign_id": campaign_id,
        "status": campaign["status"],
        "child_runs": [
            workflow.status_run(child["run_id"], root=root)
            for child in campaign["child_runs"]
        ],
    }


def build_evaluation_policy_amendment(
    *,
    campaign_id: str,
    base_evaluation_policy_sha256: str,
    resolution: Mapping[str, Any],
    owner_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact pre-results removal of first-response validity gates.

    The base campaign remains immutable.  This additive artifact is valid only
    when the repository owner explicitly removes the requirements before any
    reference labeling, comparison, evaluation, or gate decision.
    """
    if not ledger.ID_PREFIX_RE.fullmatch(campaign_id) or not campaign_id.startswith(
        "camp_"
    ):
        raise ValueError("invalid campaign_id")
    if not ledger.HASH_RE.fullmatch(base_evaluation_policy_sha256):
        raise ValueError("invalid base evaluation policy hash")
    required_resolution = {
        "resolution_id": "ARCV1-RES010",
        "decision": "approved_pre_results_first_response_gate_removal",
        "decided_by": "repository_user",
    }
    if any(resolution.get(key) != value for key, value in required_resolution.items()):
        raise ValueError("first-response gate removal lacks exact owner resolution")
    if not {
        "ARCV1-B001",
        "ARCV1-D017",
    } <= set(resolution.get("applies_to_items", [])):
        raise ValueError("owner resolution does not cover the policy amendment")
    required_attestation = {
        "responses_machine_produced": True,
        "repository_user_edited_response_payloads": False,
        "label_content_inspected_before_amendment": False,
        "reference_labeling_started": False,
        "comparison_started": False,
        "evaluation_started": False,
    }
    if any(
        owner_attestation.get(key) != value
        for key, value in required_attestation.items()
    ):
        raise ValueError("pre-results owner attestation is incomplete")
    semantic = {
        "amendment_version": POLICY_AMENDMENT_VERSION,
        "campaign_id": campaign_id,
        "base_evaluation_policy_sha256": base_evaluation_policy_sha256,
        "resolution_id": resolution["resolution_id"],
        "decision": resolution["decision"],
        "removed_requirements": FIRST_RESPONSE_REQUIREMENTS,
        "retained_schema_rule": (
            "accepted responses and emitted events must remain completely "
            "schema-valid; invalid responses never write events"
        ),
        "rationale": (
            "The completed pilot responses were machine-produced without "
            "repository-user payload editing. First-attempt formatting is not "
            "decision-relevant to label validity or the remaining pilot gates."
        ),
        "timing": "pre_reference_pre_comparison_pre_evaluation",
        "owner_attestation": dict(owner_attestation),
    }
    amendment_sha = ledger.sha256_text(ledger.canonical_json(semantic))
    effective_policy_sha = ledger.sha256_text(
        ledger.canonical_json(
            {
                "base_evaluation_policy_sha256": base_evaluation_policy_sha256,
                "policy_amendment_sha256": amendment_sha,
            }
        )
    )
    return {
        **semantic,
        "amendment_id": "amend_" + amendment_sha.removeprefix("sha256:"),
        "policy_amendment_sha256": amendment_sha,
        "effective_evaluation_policy_sha256": effective_policy_sha,
    }


def publish_evaluation_policy_amendment(
    amendment: Mapping[str, Any],
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Atomically publish one content-addressed policy amendment."""
    value = dict(amendment)
    amendment_id = str(value.pop("amendment_id", ""))
    declared_sha = str(value.pop("policy_amendment_sha256", ""))
    effective_sha = str(value.pop("effective_evaluation_policy_sha256", ""))
    observed_sha = ledger.sha256_text(ledger.canonical_json(value))
    if (
        not amendment_id.startswith("amend_")
        or amendment_id != "amend_" + declared_sha.removeprefix("sha256:")
        or declared_sha != observed_sha
    ):
        raise ValueError("evaluation policy amendment identity drift")
    observed_effective = ledger.sha256_text(
        ledger.canonical_json(
            {
                "base_evaluation_policy_sha256": value[
                    "base_evaluation_policy_sha256"
                ],
                "policy_amendment_sha256": declared_sha,
            }
        )
    )
    if effective_sha != observed_effective:
        raise ValueError("effective evaluation policy identity drift")
    persisted = {
        **value,
        "amendment_id": amendment_id,
        "policy_amendment_sha256": declared_sha,
        "effective_evaluation_policy_sha256": effective_sha,
    }
    path = Path(root) / "campaign_policy_amendments" / f"{amendment_id}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != persisted:
            raise ValueError("existing evaluation policy amendment drift")
        return {
            "status": "already_published",
            "path": str(path),
            **persisted,
        }
    ledger.atomic_write_json(path, persisted)
    return {"status": "published", "path": str(path), **persisted}


def _validate_reference_blindness(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if any(
        receipt.get(key) != value
        for key, value in REFERENCE_BLINDNESS_REQUIREMENTS.items()
    ) or not str(receipt.get("source", "")).strip():
        raise ValueError("independent-reference blindness receipt is incomplete")
    return dict(receipt)


def initialize_independent_reference(
    *,
    campaign_id: str,
    selection_path: Path,
    policy_amendment_path: Path,
    model_receipt: Mapping[str, Any],
    reviewer_id: str,
    blindness_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    bundle_registry_path: Path = BUNDLE_REGISTRY_PATH,
    label_registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    """Initialize the exact 240-subject reference without candidate exposure."""
    campaign = load_campaign(campaign_id, root=root)
    plan_path = (
        Path(root)
        / "campaign_plans"
        / f"{campaign['source_plan_id']}.json"
    )
    plan = _validate_content_addressed_artifact(
        json.loads(plan_path.read_text(encoding="utf-8")),
        id_field="plan_id",
        sha_field="plan_sha256",
        id_prefix="plan_",
    )
    expected_selection = plan["selection_receipts"]["reference"]
    selection = _validate_selection_artifact(
        json.loads(Path(selection_path).read_text(encoding="utf-8")),
        expected_selection,
    )
    if selection["n_subjects"] != 240:
        raise ValueError("independent reference must contain exactly 240 subjects")
    amendment = json.loads(Path(policy_amendment_path).read_text(encoding="utf-8"))
    amendment_result = publish_evaluation_policy_amendment(amendment, root=root)
    if amendment_result["campaign_id"] != campaign_id:
        raise ValueError("policy amendment belongs to another campaign")
    if (
        amendment_result["base_evaluation_policy_sha256"]
        != campaign["evaluation_policy_sha256"]
    ):
        raise ValueError("policy amendment base identity drift")
    receipt = validate_model_receipt(model_receipt, required_effort="high")
    reviewer_id = str(reviewer_id).strip()
    if not reviewer_id:
        raise ValueError("reviewer_id is required")
    blindness = _validate_reference_blindness(blindness_receipt)
    bundle = resolve_bundle(
        "paragraph_judgment_v2_candidate",
        "candidate-1",
        registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    semantic = {
        "campaign_id": campaign_id,
        "selection_id": selection["selection_id"],
        "selection_sha256": selection["selection_sha256"],
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "bundle_sha256": bundle["bundle_sha256"],
        "model_id": receipt["model_id"],
        "reasoning_effort": receipt["reasoning_effort"],
        "effective_evaluation_policy_sha256": amendment_result[
            "effective_evaluation_policy_sha256"
        ],
        "reference_role": "independent_blind_reference",
    }
    run_id = "reference-" + ledger.sha256_text(
        ledger.canonical_json(semantic)
    ).removeprefix("sha256:")
    reference_manifest = {
        **semantic,
        "reference_run_id": run_id,
        "reviewer_id": reviewer_id,
        "blindness_contract": REFERENCE_BLINDNESS_REQUIREMENTS,
        "candidate_exposure_before_seal": "prohibited",
    }
    reference_manifest["reference_manifest_sha256"] = ledger.sha256_text(
        ledger.canonical_json(reference_manifest)
    )
    manifest_path = Path(root) / "reference_runs" / run_id / "reference.json"
    if manifest_path.exists():
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        if persisted != reference_manifest:
            raise ValueError("existing independent reference identity drift")
        if run_id in ledger.sealed_runs(root):
            ledger.verify_sealed_run(run_id, root=root)
            return {
                "status": "already_sealed",
                "reference_run_id": run_id,
                "reference_manifest_sha256": reference_manifest[
                    "reference_manifest_sha256"
                ],
            }
        if not ledger.work_dir(run_id, root).exists():
            raise ValueError("reference manifest exists without its work run")
        return {
            "status": "already_initialized",
            "reference_run_id": run_id,
            "reference_manifest_sha256": reference_manifest[
                "reference_manifest_sha256"
            ],
        }
    initialize_bundle_run(
        bundle_id=bundle["bundle_id"],
        bundle_version=bundle["bundle_version"],
        run_id=run_id,
        subjects_path=Path(selection_path),
        model_receipt=receipt,
        reasoning_effort="high",
        annotator_id=reviewer_id,
        root=root,
        bundle_registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
        canonical_corpus_fingerprint=campaign["corpus_snapshot"]["fingerprint"],
        selection_artifact_id=selection["selection_id"],
        authorization_review_item_id="ARCV1-D007",
    )
    ledger.atomic_write_json(manifest_path, reference_manifest)
    ledger.atomic_write_json(
        ledger.work_dir(run_id, root) / "reference-initial-runtime-receipt.json",
        receipt,
    )
    ledger.atomic_write_json(
        ledger.work_dir(run_id, root) / "reference-initial-blindness-receipt.json",
        blindness,
    )
    return {
        "status": "initialized_no_assignments",
        "reference_run_id": run_id,
        "reference_manifest_sha256": reference_manifest[
            "reference_manifest_sha256"
        ],
    }


def load_independent_reference(
    reference_run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    path = Path(root) / "reference_runs" / reference_run_id / "reference.json"
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = value.get("reference_manifest_sha256")
    semantic = dict(value)
    semantic.pop("reference_manifest_sha256", None)
    if declared != ledger.sha256_text(ledger.canonical_json(semantic)):
        raise ValueError("independent reference manifest hash drift")
    if value.get("reference_run_id") != reference_run_id:
        raise ValueError("independent reference run identity drift")
    return value


def next_reference_assignment(
    reference_run_id: str,
    *,
    runtime_receipt: Mapping[str, Any],
    blindness_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    reference = load_independent_reference(reference_run_id, root=root)
    receipt = validate_model_receipt(runtime_receipt, required_effort="high")
    if receipt["model_id"] != reference["model_id"]:
        raise ValueError("reference runtime model identity drift")
    blindness = _validate_reference_blindness(blindness_receipt)
    session = {
        "reference_run_id": reference_run_id,
        "runtime_receipt": receipt,
        "blindness_receipt": blindness,
    }
    receipt_id = ledger.content_id("rcpt", session)
    path = (
        ledger.work_dir(reference_run_id, root)
        / "reference-session-receipts"
        / f"{receipt_id}.json"
    )
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != session:
            raise ValueError("reference session receipt identity drift")
    else:
        ledger.atomic_write_json(path, session)
    return workflow.next_assignment(
        reference_run_id, root=root, registry_path=registry_path
    )


def audit_independent_reference(
    reference_run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    """Audit reference structure and report only support counts."""
    load_independent_reference(reference_run_id, root=root)
    audit = workflow.audit_run(
        reference_run_id, root=root, registry_path=registry_path
    )
    if audit["status"] != "ok":
        return audit
    events = ledger.read_jsonl(
        ledger.work_dir(reference_run_id, root) / "label_events.jsonl"
    )
    constituency = [
        json.loads(row["raw_value_json"])
        for row in events
        if row["label_type"] == "constituencies"
        and row["event_role"] == "value"
    ]
    if len(constituency) != 240:
        return {
            **audit,
            "status": "failed",
            "issues": ["reference constituency event count is not 240"],
        }
    n_positive = sum(row["outcome"] == "claim" for row in constituency)
    n_unclear = sum(row["outcome"] == "unclear" for row in constituency)
    category_support: dict[str, int] = {}
    for value in constituency:
        for claim in value["claims"]:
            for field in ("group_type", "relation", "stance"):
                key = f"{field}:{claim[field]}"
                category_support[key] = category_support.get(key, 0) + 1
    support_status = "ok" if n_positive >= 60 else "insufficient"
    return {
        **audit,
        "reference_subjects": len(constituency),
        "constituency_positive_subjects": n_positive,
        "constituency_unclear_subjects": n_unclear,
        "category_support": dict(sorted(category_support.items())),
        "support_floor": 60,
        "support_status": support_status,
    }


def seal_independent_reference(
    reference_run_id: str,
    *,
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    if reference_run_id in ledger.sealed_runs(root):
        seal = ledger.verify_sealed_run(reference_run_id, root=root)
        events = ledger.read_jsonl(
            ledger.sealed_artifact_dir(reference_run_id, root)
            / "label_events.jsonl"
        )
        values = [
            json.loads(row["raw_value_json"])
            for row in events
            if row["label_type"] == "constituencies"
            and row["event_role"] == "value"
        ]
        return {
            "status": "already_sealed",
            "run_id": reference_run_id,
            "artifact_set_sha256": seal["artifact_set_sha256"],
            "constituency_positive_subjects": sum(
                row["outcome"] == "claim" for row in values
            ),
            "constituency_unclear_subjects": sum(
                row["outcome"] == "unclear" for row in values
            ),
        }
    audit = audit_independent_reference(
        reference_run_id, root=root, registry_path=registry_path
    )
    if audit.get("status") != "ok" or audit.get("support_status") != "ok":
        raise ValueError("independent reference audit or support floor failed")
    result = ledger.seal_run(
        reference_run_id, root=root, registry_path=registry_path
    )
    ledger.verify_sealed_run(reference_run_id, root=root)
    return {
        **result,
        "constituency_positive_subjects": audit[
            "constituency_positive_subjects"
        ],
        "constituency_unclear_subjects": audit[
            "constituency_unclear_subjects"
        ],
    }


def next_campaign_assignment(
    campaign_id: str,
    run_id: str,
    *,
    runtime_receipt: Mapping[str, Any],
    blindness_receipt: Mapping[str, Any],
    root: Path = ledger.LEDGER_ROOT,
    registry_path: Path = ledger.REGISTRY_V2_PATH,
) -> dict[str, Any]:
    campaign = load_campaign(campaign_id, root=root)
    children = {
        child["run_id"]: child for child in campaign["child_runs"]
    }
    if run_id not in children:
        raise ValueError("run is not a child of the declared campaign")
    child = children[run_id]
    locked = next(
        bundle
        for bundle in campaign["bundles"]
        if bundle["bundle_id"] == child["bundle_id"]
    )
    receipt = validate_model_receipt(
        runtime_receipt, required_effort=locked["reasoning_effort"]
    )
    if receipt["model_id"] != locked["model_receipt"]["model_id"]:
        raise ValueError("current runtime model differs from locked campaign model")
    required_blindness = {
        "fresh_restricted_labeling_session": True,
        "sibling_responses_inspected": False,
    }
    if any(
        blindness_receipt.get(key) != value
        for key, value in required_blindness.items()
    ) or not str(blindness_receipt.get("source", "")).strip():
        raise ValueError("fresh-session blindness receipt is incomplete")
    session_receipt = {
        "campaign_id": campaign_id,
        "run_id": run_id,
        "runtime_receipt": receipt,
        "blindness_receipt": dict(blindness_receipt),
    }
    receipt_id = ledger.content_id("rcpt", session_receipt)
    receipt_path = (
        ledger.work_dir(run_id, root)
        / "execution-session-receipts"
        / f"{receipt_id}.json"
    )
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != session_receipt:
            raise ValueError("execution session receipt identity drift")
    else:
        ledger.atomic_write_json(receipt_path, session_receipt)
    return workflow.next_assignment(
        run_id, root=root, registry_path=registry_path
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-registry", type=Path, default=BUNDLE_REGISTRY_PATH
    )
    parser.add_argument(
        "--label-registry", type=Path, default=ledger.REGISTRY_V2_PATH
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-bundles")
    subparsers.add_parser("validate-registry")
    pilot = subparsers.add_parser(
        "plan-pilot",
        help="publish the approved non-executable 722-subject pilot plan",
    )
    pilot.add_argument("--approval-id", required=True)
    provisional = subparsers.add_parser(
        "plan-provisional-corpus",
        help=(
            "publish the non-executable provisional Sol corpus reuse, "
            "fresh-selection, and plan artifacts"
        ),
    )
    provisional.add_argument("--approval-id", required=True)
    initialize = subparsers.add_parser(
        "init-pilot",
        help="initialize the exact approved pilot without issuing assignments",
    )
    initialize.add_argument("--plan", type=Path, required=True)
    initialize.add_argument("--review-queue", type=Path, required=True)
    initialize.add_argument("--initialized-by", required=True)
    status = subparsers.add_parser("campaign-status")
    status.add_argument("--campaign-id", required=True)
    nxt = subparsers.add_parser(
        "next",
        help="issue the next assignment for one initialized campaign child",
    )
    nxt.add_argument("--campaign-id", required=True)
    nxt.add_argument("--run-id", required=True)
    nxt.add_argument("--runtime-receipt", type=Path, required=True)
    nxt.add_argument("--blindness-receipt", type=Path, required=True)
    amend = subparsers.add_parser(
        "amend-evaluation-policy",
        help="publish an owner-approved pre-results evaluation-policy amendment",
    )
    amend.add_argument("--campaign-id", required=True)
    amend.add_argument("--review-queue", type=Path, required=True)
    amend.add_argument("--owner-attestation", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list-bundles":
        result: Any = list_bundles(
            args.bundle_registry, label_registry_path=args.label_registry
        )
    elif args.command == "validate-registry":
        registry = read_bundle_registry(
            args.bundle_registry, label_registry_path=args.label_registry
        )
        result = {
            "status": "ok",
            "registry_sha256": registry["registry_sha256"],
            "n_bundles": len(registry["entries"]),
        }
    elif args.command == "plan-pilot":
        if args.approval_id != "ARCV1-RES004":
            raise ValueError(
                "pilot planning requires the approved ARCV1-RES004 amendment"
            )
        from . import annotation_refresh_planning as planning

        result = planning.create_pilot_plan(
            bundles=registered_bundles(
                args.bundle_registry, label_registry_path=args.label_registry
            )
        )
    elif args.command == "plan-provisional-corpus":
        from . import annotation_refresh_provisional as provisional

        result = provisional.create_provisional_corpus_plan(
            approval_id=args.approval_id,
            bundle_registry_path=args.bundle_registry,
            label_registry_path=args.label_registry,
        )
    elif args.command == "init-pilot":
        result = initialize_approved_pilot(
            plan_path=args.plan,
            review_queue_path=args.review_queue,
            initialized_by=args.initialized_by,
            bundle_registry_path=args.bundle_registry,
            label_registry_path=args.label_registry,
        )
    elif args.command == "campaign-status":
        result = campaign_status(args.campaign_id)
    elif args.command == "next":
        result = next_campaign_assignment(
            args.campaign_id,
            args.run_id,
            runtime_receipt=json.loads(
                args.runtime_receipt.read_text(encoding="utf-8")
            ),
            blindness_receipt=json.loads(
                args.blindness_receipt.read_text(encoding="utf-8")
            ),
            registry_path=args.label_registry,
        )
    elif args.command == "amend-evaluation-policy":
        campaign = load_campaign(args.campaign_id)
        queue = json.loads(args.review_queue.read_text(encoding="utf-8"))
        resolution = next(
            (
                row
                for row in queue.get("resolutions", [])
                if row.get("resolution_id") == "ARCV1-RES010"
            ),
            None,
        )
        if resolution is None:
            raise ValueError("missing ARCV1-RES010 owner resolution")
        amendment = build_evaluation_policy_amendment(
            campaign_id=args.campaign_id,
            base_evaluation_policy_sha256=campaign[
                "evaluation_policy_sha256"
            ],
            resolution=resolution,
            owner_attestation=json.loads(
                args.owner_attestation.read_text(encoding="utf-8")
            ),
        )
        result = publish_evaluation_policy_amendment(amendment)
    else:  # pragma: no cover - argparse owns the command vocabulary
        raise AssertionError(args.command)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
