from __future__ import annotations

import builtins
import copy
import json
import socket
from pathlib import Path

import pytest

from presidential_profiles import annotation_ledger as ledger
from presidential_profiles import annotation_refresh as refresh
from presidential_profiles import annotation_refresh_provisional as provisional


@pytest.fixture(scope="session")
def provisional_publication(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict, dict[str, bytes]]:
    publish_root = tmp_path_factory.mktemp("provisional-corpus-plan")
    result = provisional.create_provisional_corpus_plan(
        approval_id=provisional.PLANNING_APPROVAL_ID,
        publish_root=publish_root,
    )
    first_bytes = {
        role: Path(path).read_bytes()
        for role, path in result["paths"].items()
    }
    again = provisional.create_provisional_corpus_plan(
        approval_id=provisional.PLANNING_APPROVAL_ID,
        publish_root=publish_root,
    )
    assert again["status"] == "already_planned"
    assert {
        role: Path(path).read_bytes()
        for role, path in again["paths"].items()
    } == first_bytes
    return result, first_bytes


@pytest.fixture(scope="session")
def real_evidence() -> dict:
    return provisional._load_evidence(ledger.LEDGER_ROOT)


def _load_result_artifacts(result: dict) -> tuple[dict, dict, dict]:
    return tuple(
        json.loads(Path(result["paths"][role]).read_text(encoding="utf-8"))
        for role in ("reuse_manifest", "fresh_selection", "plan")
    )


def test_provisional_reuse_derives_exact_972_708_and_final_adjudications(
    provisional_publication: tuple[dict, dict[str, bytes]],
) -> None:
    result, _ = provisional_publication
    reuse, _, _ = _load_result_artifacts(result)
    assert reuse["counts"] == {
        "subjects": 240,
        "label_types_per_subject": 7,
        "subject_label_entries": 1_680,
        "three_way_unanimous": 972,
        "adjudicated_final": 708,
        "missing_duplicate_unclear_or_unresolved": 0,
    }
    keys = {
        (row["canonical_subject_id"], row["label_type"])
        for row in reuse["entries"]
    }
    assert len(keys) == 1_680
    by_subject: dict[str, set[str]] = {}
    for subject_id, label_type in keys:
        by_subject.setdefault(subject_id, set()).add(label_type)
    assert len(by_subject) == 240
    assert all(
        labels == set(provisional.EXPECTED_LABEL_TYPES)
        for labels in by_subject.values()
    )

    unanimous = [
        row
        for row in reuse["entries"]
        if row["resolution_kind"] == "three_way_unanimous"
    ]
    adjudicated = [
        row
        for row in reuse["entries"]
        if row["resolution_kind"] == "adjudicated_final"
    ]
    assert len(unanimous) == 972
    assert len(adjudicated) == 708
    assert all(
        row["selected_source_role"] == "reference"
        and row["confidence_provenance"]
        == "three_way_same_model_unanimous"
        and len(
            {
                source["value_sha256"]
                for source in row["source_groups"].values()
            }
        )
        == 1
        for row in unanimous
    )
    assert len(
        {row["published_adjudication_id"] for row in adjudicated}
    ) == 708
    assert {
        row["confidence_provenance"] for row in adjudicated
    } == {
        "adjudicated_matches_majority",
        "adjudicated_matches_reference",
        "adjudicated_matches_single_source",
        "adjudicated_novel_synthesis",
    }
    assert all(
        row["human_ground_truth_claim"] is False
        and row["numeric_confidence_probability"] is None
        for row in reuse["entries"]
    )


def test_provisional_reuse_rejects_missing_and_unclear_adjudication(
    monkeypatch: pytest.MonkeyPatch,
    real_evidence: dict,
) -> None:
    monkeypatch.setattr(
        provisional, "_load_evidence", lambda root: real_evidence
    )
    decisions = provisional._decision_rows(ledger.LEDGER_ROOT)
    resolved_ids = {
        row["label_group_id"]
        for row in real_evidence["groups"]["adjudication"].values()
    }
    selected = [
        row
        for row in decisions
        if row.get("resolved_label_group_id") in resolved_ids
    ]
    missing_id = selected[0]["adjudication_id"]
    monkeypatch.setattr(
        provisional,
        "_decision_rows",
        lambda root: [
            row
            for row in decisions
            if row.get("adjudication_id") != missing_id
        ],
    )
    with pytest.raises(ValueError, match="missing resolved or published"):
        provisional.build_reuse_manifest()

    monkeypatch.setattr(
        provisional, "_decision_rows", lambda root: decisions
    )
    unclear_evidence = {
        **real_evidence,
        "groups": {
            **real_evidence["groups"],
            "adjudication": {
                key: dict(value)
                for key, value in real_evidence["groups"][
                    "adjudication"
                ].items()
            },
        },
    }
    unclear_key = next(
        key
        for key in real_evidence["adjudication"]["disputes"]
        if key["label_type"] == "constituencies"
    )
    key = (
        unclear_key["canonical_subject_id"],
        unclear_key["label_type"],
    )
    unclear_json = ledger.canonical_json(
        {"claims": [], "outcome": "unclear", "unclear_reason": "test"}
    )
    unclear_evidence["groups"]["adjudication"][key].update(
        {
            "value_json": unclear_json,
            "value_sha256": ledger.sha256_text(unclear_json),
        }
    )
    monkeypatch.setattr(
        provisional, "_load_evidence", lambda root: unclear_evidence
    )
    with pytest.raises(ValueError, match="remains unclear"):
        provisional.build_reuse_manifest()


def test_provisional_reuse_is_key_joined_not_row_order_joined(
    monkeypatch: pytest.MonkeyPatch,
    real_evidence: dict,
) -> None:
    baseline = provisional.build_reuse_manifest()
    reordered = {
        **real_evidence,
        "groups": {
            role: dict(reversed(list(groups.items())))
            for role, groups in real_evidence["groups"].items()
        },
    }
    monkeypatch.setattr(
        provisional, "_load_evidence", lambda root: reordered
    )
    assert provisional.build_reuse_manifest() == baseline


def test_fresh_selection_is_exact_240_key_complement_and_keeps_other_482(
    provisional_publication: tuple[dict, dict[str, bytes]],
) -> None:
    result, _ = provisional_publication
    reuse, fresh, _ = _load_result_artifacts(result)
    reuse_ids = {
        row["canonical_subject_id"] for row in reuse["entries"]
    }
    fresh_ids = {
        row["canonical_subject_id"] for row in fresh["identity_rows"]
    }
    assert len(reuse_ids) == 240
    assert len(fresh_ids) == 35_154
    assert reuse_ids.isdisjoint(fresh_ids)
    assert len(reuse_ids | fresh_ids) == 35_394
    assert fresh["other_pilot_subjects_included"] == 482
    assert fresh["validation"] == {
        "reuse_and_fresh_disjoint": True,
        "reuse_and_fresh_union_complete": True,
        "duplicate_stale_or_out_of_universe_keys": 0,
        "excluded_by_canonical_identity_not_position": True,
        "other_482_pilot_paragraphs_included": True,
        "current_corpus_matches_locked_snapshot": True,
        "active_phase1_generation_matches_locked_snapshot": True,
    }
    assert all(
        row["source_text_sha256"]
        == ledger.sha256_text(row["text"])
        and row["bundle_input_sha256"].startswith("sha256:")
        for row in fresh["rows"]
    )


def test_provisional_batches_and_token_receipt_are_deterministic(
    provisional_publication: tuple[dict, dict[str, bytes]],
) -> None:
    result, _ = provisional_publication
    _, fresh, plan = _load_result_artifacts(result)
    assert fresh["n_assignments"] == 9_186
    assert len(fresh["batches"]) == 9_186
    assert all(1 <= len(batch["target_keys"]) <= 4 for batch in fresh["batches"])
    assert len(
        {batch["planned_batch_id"] for batch in fresh["batches"]}
    ) == 9_186
    assert plan["token_receipt"] == {
        "estimator": "deterministic_ceil_utf8_bytes_divided_by_4",
        "runtime_tokenizer_available": False,
        "reasoning_and_internal_tokens_included": False,
        "render_serialization": "indent_2_sorted_keys_ensure_ascii_false",
        "fresh_subjects": 35_154,
        "fresh_assignments": 9_186,
        "fresh_input_tokens": 81_663_911,
        "fresh_output_tokens": 5_624_640,
        "fresh_input_token_ceiling_10_percent": 89_830_303,
    }
    assert plan["required_execution_profile"] == {
        "model_id": "gpt-5.6-sol",
        "model_identity_specificity": "exact_runtime_identifier",
        "reasoning_effort": "high",
        "fresh_single_pass": True,
        "new_runtime_receipt_required": True,
    }


def test_provisional_publication_is_content_addressed_and_byte_idempotent(
    provisional_publication: tuple[dict, dict[str, bytes]],
) -> None:
    result, artifact_bytes = provisional_publication
    reuse, fresh, plan = _load_result_artifacts(result)
    provisional._validate_artifact(
        reuse,
        prefix="reuse",
        id_field="reuse_manifest_id",
        sha_field="reuse_manifest_sha256",
    )
    provisional._validate_artifact(
        fresh,
        prefix="fresh",
        id_field="fresh_selection_id",
        sha_field="fresh_selection_sha256",
    )
    provisional._validate_artifact(
        plan,
        prefix="pcplan",
        id_field="plan_id",
        sha_field="plan_sha256",
    )
    assert all(artifact_bytes.values())
    drifted = copy.deepcopy(plan)
    drifted["required_execution_profile"]["model_id"] = "drift"
    with pytest.raises(ValueError, match="content-addressed artifact drift"):
        provisional._validate_artifact(
            drifted,
            prefix="pcplan",
            id_field="plan_id",
            sha_field="plan_sha256",
        )


def test_provisional_planner_rejects_bundle_prompt_schema_model_and_corpus_drift(
    monkeypatch: pytest.MonkeyPatch,
    real_evidence: dict,
) -> None:
    bundle = refresh.resolve_bundle(
        provisional.COMBINED_BUNDLE_ID,
        provisional.COMBINED_BUNDLE_VERSION,
    )
    receipt = provisional._bundle_receipt(
        bundle,
        bundle_registry_path=refresh.BUNDLE_REGISTRY_PATH,
        label_registry_path=ledger.REGISTRY_V2_PATH,
    )
    for field in (
        "bundle_sha256",
        "prompt_sha256",
        "response_schema_sha256",
        "context_policy_sha256",
    ):
        drifted = {**receipt, field: "sha256:" + "0" * 64}
        with pytest.raises(ValueError, match="drift"):
            provisional._validate_bundle_evidence(
                bundle,
                drifted,
                real_evidence,
                root=ledger.LEDGER_ROOT,
            )

    model_drift = copy.deepcopy(real_evidence)
    campaign_bundle = next(
        row
        for row in model_drift["campaign"]["bundles"]
        if row["bundle_id"] == provisional.COMBINED_BUNDLE_ID
    )
    campaign_bundle["model_receipt"]["model_id"] = "other-model"
    with pytest.raises(ValueError, match="model/effort drift"):
        provisional._validate_bundle_evidence(
            bundle,
            receipt,
            model_drift,
            root=ledger.LEDGER_ROOT,
        )

    reuse = provisional.build_reuse_manifest()
    monkeypatch.setattr(
        provisional, "_load_evidence", lambda root: real_evidence
    )
    monkeypatch.setattr(
        provisional,
        "_universe_rows",
        lambda: (
            [],
            {
                **real_evidence["campaign"]["corpus_snapshot"],
                "fingerprint": "sha256:" + "0" * 64,
            },
        ),
    )
    with pytest.raises(ValueError, match="locked campaign"):
        provisional.build_fresh_selection(reuse)


def test_planning_path_has_no_campaign_run_or_materialization_side_effects(
    provisional_publication: tuple[dict, dict[str, bytes]],
) -> None:
    result, _ = provisional_publication
    assert set(result["publication_status"].values()) <= {
        "published",
        "already_published",
    }
    for role, path in result["paths"].items():
        assert role in {
            "reuse_manifest",
            "fresh_selection",
            "plan",
        }
        assert Path(path).is_file()
    assert all(
        effect in result_effects
        for effect in (
            "campaign_initialization",
            "run_initialization",
            "assignment_creation",
            "provider_or_model_invocation",
            "response_write",
            "promotion",
            "materialized_generation",
        )
        for result_effects in [
            json.loads(
                Path(result["paths"]["plan"]).read_text(encoding="utf-8")
            )["prohibited_effects"]
        ]
    )


def test_provisional_planner_is_provider_neutral_and_offline(
    monkeypatch: pytest.MonkeyPatch,
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
            raise AssertionError(f"provider/network client import attempted: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network connection attempted")
        ),
    )
    evidence = provisional._load_evidence(ledger.LEDGER_ROOT)
    assert evidence["evaluation"]["evaluation_id"] == provisional.EVALUATION_ID
    source = Path(provisional.__file__).read_text(encoding="utf-8")
    assert all(
        marker not in source
        for marker in (
            "import openai",
            "import anthropic",
            "import requests",
            "import httpx",
            "socket.create_connection",
        )
    )


def test_full_shadow_guard_remains_production_eligible_only() -> None:
    bundle = refresh.resolve_bundle(
        provisional.COMBINED_BUNDLE_ID,
        provisional.COMBINED_BUNDLE_VERSION,
    )
    with pytest.raises(ValueError, match="non-production"):
        refresh.build_campaign_manifest(
            corpus_snapshot={"fingerprint": "sha256:" + "1" * 64},
            bundles=[bundle],
            scope="full-shadow",
            selection_receipts={
                bundle["bundle_id"]: {
                    "selection_id": "sel_" + "2" * 64,
                    "selection_sha256": "sha256:" + "3" * 64,
                    "n_subjects": 35_394,
                }
            },
            model_receipts={
                bundle["bundle_id"]: {
                    "model_id": "gpt-5.6-sol",
                    "specificity": "exact_runtime_identifier",
                    "reasoning_effort": "high",
                }
            },
            reasoning_effort={bundle["bundle_id"]: "high"},
            evaluation_policy_sha256="sha256:" + "4" * 64,
            promotion_policy_sha256="sha256:" + "5" * 64,
            approval_ids=["not-authorized"],
        )
