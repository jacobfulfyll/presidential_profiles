from __future__ import annotations

import builtins
import json
import shutil
import socket
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from presidential_profiles import annotation_ledger as ledger
from presidential_profiles import annotation_refresh as refresh
from presidential_profiles import annotation_refresh_planning as planning
from presidential_profiles import annotation_refresh_stage3 as stage3
from presidential_profiles import annotation_workflow as workflow


def _exact_receipt(
    model_id: str = "test-runtime-exact", effort: str = "high"
) -> dict:
    return {
        "model_id": model_id,
        "specificity": "exact_runtime_identifier",
        "reasoning_effort": effort,
        "source": "pytest-runtime-metadata",
    }


def _paragraph_subjects(path: Path, count: int = 1, *, split_speeches: bool = False):
    rows = []
    for index in range(count):
        rows.append(
            {
                "doc_name": f"/speech-{index}" if split_speeches else "/speech",
                "para_idx": index if not split_speeches else 0,
                "text": (
                    "The people authorize us to protect workers from the hostile "
                    "Combine."
                    if index == 0
                    else f"Paragraph {index} contains no declared group relationship."
                ),
                "decade": "1900s",
            }
        )
    ledger.atomic_write_json(path, rows)


def _selection_artifact(path: Path, rows: list[dict]) -> dict:
    identity_rows = [
        {
            "subject_key": {
                "doc_name": row["doc_name"],
                "para_idx": row["para_idx"],
            },
            "bundle_input_sha256": ledger.sha256_text(
                ledger.canonical_json(
                    {
                        "text": row["text"],
                        "decade": row["decade"],
                    }
                )
            ),
            "stratum": "test",
            "inclusion_probability": 1.0,
        }
        for row in rows
    ]
    semantic = {
        "identity_rows": identity_rows,
        "rows": rows,
        "n_subjects": len(rows),
        "n_assignments": 1,
    }
    selection_sha256 = ledger.sha256_text(ledger.canonical_json(semantic))
    selection = {
        **semantic,
        "selection_id": refresh.subject_selection_id(identity_rows),
        "selection_sha256": selection_sha256,
    }
    ledger.atomic_write_json(path, selection)
    return selection


def _combined_bundle() -> dict:
    return refresh.resolve_bundle(
        "paragraph_judgment_v2_candidate", "candidate-1"
    )


def _combined_response(assignment: dict) -> dict:
    bundle = _combined_bundle()
    topics = next(
        row
        for row in bundle["emitted_label_specs"]
        if row["label_type"] == "topics"
    )
    topic_spec = ledger.resolve_spec(
        "topics",
        topics["spec_version"],
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    topic = topic_spec["response_schema"]["items"]["enum"][0]
    annotations = []
    for target in assignment["targets"]:
        local_id = target["local_id"]
        if local_id == 1:
            constituency = {
                "outcome": "claim",
                "claims": [
                    {
                        "group_text": "workers",
                        "resolved_referent": "",
                        "group_type": "occupation_or_industry",
                        "relation": "protected_group",
                        "stance": "favorable",
                        "evidence_span": "protect workers",
                        "certainty": "explicit",
                    }
                ],
                "unclear_reason": "",
            }
            enemy_naming = True
            entities = [
                {"name": "workers", "type": "group", "stance": "favorable"},
                {"name": "Combine", "type": "group", "stance": "adversarial"},
            ]
        elif local_id == 2:
            constituency = {
                "outcome": "none",
                "claims": [],
                "unclear_reason": "",
            }
            enemy_naming = True
            entities = [
                {"name": "opposition", "type": "group", "stance": "adversarial"}
            ]
        elif local_id == 3:
            constituency = {
                "outcome": "claim",
                "claims": [
                    {
                        "group_text": "Paragraph",
                        "resolved_referent": "",
                        "group_type": "other",
                        "relation": "represented_constituency",
                        "stance": "neutral",
                        "evidence_span": "Paragraph",
                        "certainty": "explicit",
                    }
                ],
                "unclear_reason": "",
            }
            enemy_naming = False
            entities = []
        else:
            constituency = {
                "outcome": "none",
                "claims": [],
                "unclear_reason": "",
            }
            enemy_naming = False
            entities = []
        annotations.append(
            {
                "local_id": local_id,
                "topics": [topic],
                "party_attack": False,
                "enemy_naming": enemy_naming,
                "zero_sum": False,
                "proposal_values": "mixed",
                "entities": entities,
                "constituencies": constituency,
            }
        )
    return {"annotations": annotations}


def _initialize_combined(
    tmp_path: Path, *, count: int = 1, split_speeches: bool = False
) -> tuple[Path, dict]:
    root = tmp_path / "ledger"
    subjects = tmp_path / "subjects.json"
    _paragraph_subjects(subjects, count=count, split_speeches=split_speeches)
    refresh.initialize_bundle_run(
        bundle_id="paragraph_judgment_v2_candidate",
        bundle_version="candidate-1",
        run_id="combined-test",
        subjects_path=subjects,
        model_receipt=_exact_receipt(),
        reasoning_effort="high",
        annotator_id="pytest",
        root=root,
        canonical_corpus_fingerprint="sha256:" + "1" * 64,
        authorization_review_item_id="ARCV1-C001",
    )
    assignment = workflow.next_assignment(
        "combined-test",
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    return root, assignment


def test_stage1_bundle_registry_and_all_eligibility() -> None:
    rows = refresh.list_bundles()
    assert {
        (row["bundle_id"], row["bundle_version"], row["eligibility"])
        for row in rows
    } == {
        ("paragraph_judgment_v2_candidate", "candidate-1", "pilot_only"),
        ("speech_factual", "v1", "production_eligible"),
        ("invocation_tone", "candidate-1", "pilot_only"),
        ("constituency_only_diagnostic", "candidate-1", "pilot_diagnostic"),
    }
    assert [
        bundle["bundle_id"] for bundle in refresh.select_bundles("all")
    ] == ["speech_factual"]


def test_combined_bundle_keeps_atomic_types_and_roles_independent(
    tmp_path: Path,
) -> None:
    root, assignment = _initialize_combined(tmp_path, count=4)
    rendered = workflow.render_assignment(assignment)
    rendered_json = ledger.canonical_json(rendered)
    assert "doc_name" not in rendered_json
    assert "para_idx" not in rendered_json
    assert [row["local_id"] for row in rendered["targets"]] == [1, 2, 3, 4]
    assert rendered["shared_context"] == {"before": [], "after": []}
    assert all(
        "context_before" not in row and "context_after" not in row
        for row in rendered["targets"]
    )
    for target in assignment["targets"]:
        assert rendered_json.count(target["text"]) == 1

    workflow.ingest_response(
        "combined-test",
        assignment["assignment_id"],
        _combined_response(assignment),
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    events = ledger.read_jsonl(
        ledger.work_dir("combined-test", root) / "label_events.jsonl"
    )
    first_subject = assignment["targets"][0]["subject_id"]
    first = [row for row in events if row["subject_id"] == first_subject]
    assert {row["label_type"] for row in first} == {
        "topics",
        "party_attack",
        "enemy_naming",
        "zero_sum",
        "proposal_values",
        "entities",
        "constituencies",
    }
    assert len(
        {row["label_group_id"] for row in first if row["label_type"] != "entities"}
    ) == 6
    entity_rows = [row for row in first if row["label_type"] == "entities"]
    assert len(entity_rows) == 2
    assert len({row["label_group_id"] for row in entity_rows}) == 1

    by_subject = {}
    for target in assignment["targets"]:
        subject_events = [
            row for row in events if row["subject_id"] == target["subject_id"]
        ]
        values = {
            row["label_type"]: json.loads(row["raw_value_json"])
            for row in subject_events
            if row["label_type"] in {"enemy_naming", "constituencies"}
        }
        by_subject[target["local_id"]] = (
            values["enemy_naming"],
            values["constituencies"]["outcome"],
        )
    assert by_subject == {
        1: (True, "claim"),
        2: (True, "none"),
        3: (False, "claim"),
        4: (False, "none"),
    }


def test_same_speech_local_ids_prevent_cross_speech_para_alias(
    tmp_path: Path,
) -> None:
    _, assignment = _initialize_combined(
        tmp_path, count=2, split_speeches=True
    )
    assert len(assignment["targets"]) == 1
    assert assignment["targets"][0]["subject_key"]["para_idx"] == 0


def test_persisted_planned_batch_partition_is_honored(tmp_path: Path) -> None:
    subjects = tmp_path / "planned-subjects.json"
    rows = []
    for index in range(6):
        rows.append(
            {
                "doc_name": "/speech",
                "para_idx": index,
                "text": f"Paragraph {index}.",
                "decade": "1900s",
                "planned_batch_id": "batch_first" if index < 2 else "batch_second",
                "planned_batch_order": index + 1 if index < 2 else index - 1,
            }
        )
    ledger.atomic_write_json(subjects, rows)
    root = tmp_path / "planned-ledger"
    refresh.initialize_bundle_run(
        bundle_id="paragraph_judgment_v2_candidate",
        bundle_version="candidate-1",
        run_id="planned-test",
        subjects_path=subjects,
        model_receipt=_exact_receipt(),
        reasoning_effort="high",
        annotator_id="pytest",
        root=root,
        authorization_review_item_id="TEST-ONLY",
    )
    assignment = workflow.next_assignment(
        "planned-test",
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    assert [
        target["subject_key"]["para_idx"] for target in assignment["targets"]
    ] == [0, 1]
    rendered = workflow.render_assignment(assignment)
    assert rendered["shared_context"]["before"] == []
    assert rendered["shared_context"]["after"] == [{"text": "Paragraph 2."}]


def test_stage2_pilot_plan_is_content_addressed_and_non_executable() -> None:
    plan_id = (
        "plan_3af71613624c605209c169717c7c73024235e01ba0750d046940a01d2512a17f"
    )
    plan_path = planning.CAMPAIGN_PLANS_ROOT / f"{plan_id}.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_semantic = dict(plan)
    assert plan_semantic.pop("plan_id") == plan_id
    plan_sha256 = plan_semantic.pop("plan_sha256")
    assert plan_sha256 == ledger.sha256_text(ledger.canonical_json(plan_semantic))
    assert plan["status"].startswith("non_executable_")
    assert plan["required_execution_profile"]["model_id"].startswith(
        "pending_exact_runtime_receipt"
    )
    assert plan["required_execution_profile"]["reasoning_effort"] == "high"
    assert plan["gates"] == {
        "ARCV1-G001": "pending_exact_future_label_runtime_receipt",
        "ARCV1-G002": "pending_human_plan_approval",
    }

    selections = {}
    for role, receipt in plan["selection_receipts"].items():
        path = planning.SELECTIONS_ROOT / f"{receipt['selection_id']}.json"
        selection = json.loads(path.read_text(encoding="utf-8"))
        semantic = dict(selection)
        semantic.pop("selection_id")
        selection_sha256 = semantic.pop("selection_sha256")
        assert selection_sha256 == ledger.sha256_text(
            ledger.canonical_json(semantic)
        )
        assert selection["selection_id"] == refresh.subject_selection_id(
            selection["identity_rows"]
        )
        assert len(selection["rows"]) == selection["n_subjects"]
        selections[role] = selection

    assert {
        role: selection["n_subjects"] for role, selection in selections.items()
    } == {"combined": 722, "diagnostic": 144, "reference": 240}
    assert selections["combined"]["n_assignments"] == 182
    assert selections["diagnostic"]["n_assignments"] == 36
    combined_keys = {
        ledger.canonical_json(row["subject_key"])
        for row in selections["combined"]["identity_rows"]
    }
    diagnostic_keys = {
        ledger.canonical_json(row["subject_key"])
        for row in selections["diagnostic"]["identity_rows"]
    }
    reference_keys = {
        ledger.canonical_json(row["subject_key"])
        for row in selections["reference"]["identity_rows"]
    }
    assert diagnostic_keys < combined_keys
    assert reference_keys < combined_keys
    assert len(reference_keys - diagnostic_keys) == 96
    frames = {}
    for row in selections["combined"]["rows"]:
        frames[row["selection_frame"]] = frames.get(row["selection_frame"], 0) + 1
    assert frames == {
        "complete_block": 476,
        "sparse_census": 6,
        "rare_enrichment": 96,
        "composition_diagnostic": 144,
    }
    assert plan["token_receipt"]["pilot"] == {
        "combined_assignments_per_pass": 182,
        "combined_input_tokens_per_pass": 1_631_631,
        "combined_output_tokens_per_pass": 115_520,
        "complete_input_tokens": 3_359_122,
        "complete_output_tokens": 237_520,
        "diagnostic_assignments": 36,
        "diagnostic_input_tokens": 95_860,
        "diagnostic_output_tokens": 6_480,
        "human_rereview_if_absolute_change_exceeds": 0.1,
        "input_token_ceiling_10_percent": 3_695_035,
        "preplan_change_fraction": pytest.approx(0.24874423791821562),
        "preplan_input_estimate": 2_690_000,
    }


def test_speech_and_invocation_bundles_render_only_bounded_inputs(
    tmp_path: Path,
) -> None:
    speech_subjects = tmp_path / "speech-subjects.json"
    ledger.atomic_write_json(
        speech_subjects,
        [
            {
                "doc_name": "/speech",
                "title": "Annual Message",
                "year": 1900,
                "opening_paragraphs": ["Opening one.", "Opening two."],
            }
        ],
    )
    speech_root = tmp_path / "speech-ledger"
    refresh.initialize_bundle_run(
        bundle_id="speech_factual",
        bundle_version="v1",
        run_id="speech-test",
        subjects_path=speech_subjects,
        model_receipt=_exact_receipt(effort="low"),
        reasoning_effort="low",
        annotator_id="pytest",
        root=speech_root,
        authorization_review_item_id="TEST-ONLY",
    )
    speech_assignment = workflow.next_assignment(
        "speech-test",
        root=speech_root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    speech_rendered = workflow.render_assignment(speech_assignment)
    assert speech_rendered["targets"] == [
        {
            "local_id": 1,
            "title": "Annual Message",
            "year": 1900,
            "opening_paragraphs": ["Opening one.", "Opening two."],
        }
    ]
    workflow.ingest_response(
        "speech-test",
        speech_assignment["assignment_id"],
        speech_assignment["response_template"],
        root=speech_root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )

    invocation_subjects = tmp_path / "invocation-subjects.json"
    ledger.atomic_write_json(
        invocation_subjects,
        [
            {
                "doc_name": "/speech",
                "char_start": 10,
                "char_end": 18,
                "mention": "Providence",
                "text": "We appeal to Providence for guidance.",
            },
            {
                "doc_name": "/speech",
                "char_start": 40,
                "char_end": 47,
                "mention": "the Lord",
                "text": "May the Lord protect the nation.",
            },
        ],
    )
    invocation_root = tmp_path / "invocation-ledger"
    refresh.initialize_bundle_run(
        bundle_id="invocation_tone",
        bundle_version="candidate-1",
        run_id="invocation-test",
        subjects_path=invocation_subjects,
        model_receipt=_exact_receipt(effort="medium"),
        reasoning_effort="medium",
        annotator_id="pytest",
        root=invocation_root,
        authorization_review_item_id="TEST-ONLY",
    )
    invocation_assignment = workflow.next_assignment(
        "invocation-test",
        root=invocation_root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    invocation_rendered = workflow.render_assignment(invocation_assignment)
    assert {tuple(row) for row in invocation_rendered["targets"]} == {
        ("local_id", "mention", "text")
    }
    assert "doc_name" not in ledger.canonical_json(invocation_rendered)
    workflow.ingest_response(
        "invocation-test",
        invocation_assignment["assignment_id"],
        invocation_assignment["response_template"],
        root=invocation_root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["annotations"].pop(),
        lambda value: value["annotations"].append(
            json.loads(json.dumps(value["annotations"][0]))
        ),
        lambda value: value["annotations"][0].__setitem__("local_id", 999),
        lambda value: value["annotations"][0].__setitem__(
            "proposal_values", "drifted"
        ),
        lambda value: value["annotations"][0].__setitem__("unknown", True),
    ],
)
def test_bundle_response_failures_happen_before_writes(
    tmp_path: Path, mutator
) -> None:
    root, assignment = _initialize_combined(tmp_path, count=2)
    response = _combined_response(assignment)
    mutator(response)
    work = ledger.work_dir("combined-test", root)
    before = {
        name: (work / name).read_bytes()
        for name in ["label_events.jsonl", "canonical_label_projection.jsonl"]
    }
    with pytest.raises((ValueError, KeyError)):
        workflow.ingest_response(
            "combined-test",
            assignment["assignment_id"],
            response,
            root=root,
            registry_path=ledger.REGISTRY_V2_PATH,
        )
    assert {
        name: (work / name).read_bytes() for name in before
    } == before
    assert not (
        work / "responses" / f"{assignment['assignment_id']}.json"
    ).exists()


def test_resume_returns_open_assignment_and_never_reissues_completed(
    tmp_path: Path,
) -> None:
    root, assignment = _initialize_combined(tmp_path, count=4)
    again = workflow.next_assignment(
        "combined-test", root=root, registry_path=ledger.REGISTRY_V2_PATH
    )
    assert again == assignment
    workflow.ingest_response(
        "combined-test",
        assignment["assignment_id"],
        _combined_response(assignment),
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    complete = workflow.next_assignment(
        "combined-test", root=root, registry_path=ledger.REGISTRY_V2_PATH
    )
    assert complete["status"] == "complete"
    assert workflow.status_run("combined-test", root=root)["completed_subjects"] == 4


def test_exact_model_and_effort_receipts_are_required(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects.json"
    _paragraph_subjects(subjects)
    with pytest.raises(ValueError, match="not exact enough"):
        refresh.initialize_bundle_run(
            bundle_id="paragraph_judgment_v2_candidate",
            bundle_version="candidate-1",
            run_id="bad-model",
            subjects_path=subjects,
            model_receipt={
                "model_id": "family",
                "specificity": "family_name",
                "reasoning_effort": "high",
            },
            reasoning_effort="high",
            annotator_id="pytest",
            root=tmp_path / "ledger",
        )
    with pytest.raises(ValueError, match="effort mismatch"):
        refresh.validate_model_receipt(_exact_receipt(effort="medium"), required_effort="high")
    assert not (tmp_path / "ledger" / "work").exists()


def test_campaign_expansion_and_manifest_locking_are_deterministic() -> None:
    bundle = refresh.resolve_bundle("speech_factual", "v1")
    selection = {
        "selection_id": "sel_" + "2" * 64,
        "selection_sha256": "sha256:" + "3" * 64,
        "n_subjects": 1053,
    }
    kwargs = {
        "corpus_snapshot": {
            "snapshot_id": "canonical-v1",
            "fingerprint": "sha256:" + "4" * 64,
            "subject_counts": {"speech": 1053},
        },
        "bundles": [bundle],
        "scope": "full-shadow",
        "selection_receipts": {"speech_factual": selection},
        "model_receipts": {
            "speech_factual": _exact_receipt(
                model_id="exact-model-build", effort="low"
            )
        },
        "reasoning_effort": {"speech_factual": "low"},
        "pass_roles": {"speech_factual": ["primary"]},
        "evaluation_policy_sha256": "sha256:" + "5" * 64,
        "promotion_policy_sha256": "sha256:" + "6" * 64,
        "approval_ids": ["FUTURE-APPROVAL"],
    }
    first = refresh.build_campaign_manifest(**kwargs)
    second = refresh.build_campaign_manifest(**kwargs)
    assert first == second
    assert len(first["child_runs"]) == 1
    assert first["child_runs"][0]["bundle_id"] == "speech_factual"
    changed = refresh.build_campaign_manifest(
        **{
            **kwargs,
            "model_receipts": {
                "speech_factual": _exact_receipt(
                    model_id="another-exact-build", effort="low"
                )
            },
        }
    )
    assert changed["campaign_id"] != first["campaign_id"]
    blind = refresh.build_campaign_manifest(
        **{
            **kwargs,
            "scope": "pilot",
            "pass_roles": {"speech_factual": ["blind-a", "blind-b"]},
        }
    )
    assert {
        (row["bundle_id"], row["pass_role"]) for row in blind["child_runs"]
    } == {
        ("speech_factual", "blind-a"),
        ("speech_factual", "blind-b"),
    }
    assert blind["bundles"][0]["selection"] == selection
    assert blind["campaign_id"] != first["campaign_id"]


def test_campaign_manifest_locks_approved_plan_and_blindness() -> None:
    bundle = _combined_bundle()
    manifest = refresh.build_campaign_manifest(
        corpus_snapshot={
            "fingerprint": "sha256:" + "1" * 64,
        },
        bundles=[bundle],
        scope="pilot",
        selection_receipts={
            bundle["bundle_id"]: {
                "selection_id": "sel_" + "2" * 64,
                "selection_sha256": "sha256:" + "3" * 64,
                "n_subjects": 4,
                "n_assignments": 1,
            }
        },
        model_receipts={bundle["bundle_id"]: _exact_receipt()},
        reasoning_effort={bundle["bundle_id"]: "high"},
        pass_roles={bundle["bundle_id"]: ["blind_a", "blind_b"]},
        evaluation_policy_sha256="sha256:" + "4" * 64,
        promotion_policy_sha256="sha256:" + "5" * 64,
        approval_ids=["ARCV1-RES008"],
        source_plan_id="plan_" + "6" * 64,
        source_plan_sha256="sha256:" + "6" * 64,
        render_policy_sha256="sha256:" + "7" * 64,
        blindness={
            "fresh_restricted_labeling_sessions_required": True,
        },
    )
    assert manifest["source_plan_id"] == "plan_" + "6" * 64
    assert manifest["source_plan_sha256"] == "sha256:" + "6" * 64
    assert manifest["render_policy_sha256"] == "sha256:" + "7" * 64
    assert manifest["blindness"] == {
        "fresh_restricted_labeling_sessions_required": True,
    }


def test_campaign_publication_is_assignment_free_and_idempotent(
    tmp_path: Path,
) -> None:
    bundle = _combined_bundle()
    rows = [
        {
            "doc_name": "/speech",
            "para_idx": index,
            "text": f"Paragraph {index}.",
            "decade": "1900s",
            "planned_batch_id": "batch-test",
            "planned_batch_order": index + 1,
        }
        for index in range(4)
    ]
    selection_path = tmp_path / "selection.json"
    selection = _selection_artifact(selection_path, rows)
    manifest = refresh.build_campaign_manifest(
        corpus_snapshot={"fingerprint": "sha256:" + "1" * 64},
        bundles=[bundle],
        scope="pilot",
        selection_receipts={
            bundle["bundle_id"]: {
                "selection_id": selection["selection_id"],
                "selection_sha256": selection["selection_sha256"],
                "n_subjects": 4,
                "n_assignments": 1,
            }
        },
        model_receipts={bundle["bundle_id"]: _exact_receipt()},
        reasoning_effort={bundle["bundle_id"]: "high"},
        pass_roles={bundle["bundle_id"]: ["blind_a", "blind_b"]},
        evaluation_policy_sha256="sha256:" + "4" * 64,
        promotion_policy_sha256="sha256:" + "5" * 64,
        approval_ids=["ARCV1-RES008"],
        source_plan_id="plan_" + "6" * 64,
        source_plan_sha256="sha256:" + "6" * 64,
        render_policy_sha256="sha256:" + "7" * 64,
        blindness={
            "fresh_restricted_labeling_sessions_required": True,
        },
    )
    root = tmp_path / "ledger"
    result = refresh.publish_campaign_initialization(
        manifest,
        selection_paths={bundle["bundle_id"]: selection_path},
        initialized_by="pytest",
        root=root,
    )
    assert result["status"] == "initialized_no_assignments"
    persisted = refresh.load_campaign(manifest["campaign_id"], root=root)
    assert persisted["status"] == "initialized"
    status = refresh.campaign_status(manifest["campaign_id"], root=root)
    assert {
        (
            child["n_subjects"],
            child["assigned_subjects"],
            child["completed_subjects"],
            child["open_assignments"],
            child["n_events"],
        )
        for child in status["child_runs"]
    } == {(4, 0, 0, 0, 0)}
    annotators = set()
    for child in manifest["child_runs"]:
        run_dir = ledger.work_dir(child["run_id"], root)
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        annotators.add(run["annotator_id"])
        assert run["reasoning_effort"] == "high"
        assert run["selection_artifact_id"] == selection["selection_id"]
        assert run["authorization_review_item_id"] == "ARCV1-G006"
        assert not list((run_dir / "assignments").glob("*.json"))
        assert not list((run_dir / "responses").glob("*.json"))
    assert annotators == {"arcv1-blind_a", "arcv1-blind_b"}
    again = refresh.publish_campaign_initialization(
        manifest,
        selection_paths={bundle["bundle_id"]: selection_path},
        initialized_by="pytest",
        root=root,
    )
    assert again["status"] == "already_initialized"
    target_run = next(
        child["run_id"]
        for child in manifest["child_runs"]
        if child["pass_role"] == "blind_a"
    )
    with pytest.raises(ValueError, match="effort mismatch"):
        refresh.next_campaign_assignment(
            manifest["campaign_id"],
            target_run,
            runtime_receipt=_exact_receipt(effort="xhigh"),
            blindness_receipt={
                "fresh_restricted_labeling_session": True,
                "sibling_responses_inspected": False,
                "source": "pytest",
            },
            root=root,
        )
    assignment = refresh.next_campaign_assignment(
        manifest["campaign_id"],
        target_run,
        runtime_receipt=_exact_receipt(),
        blindness_receipt={
            "fresh_restricted_labeling_session": True,
            "sibling_responses_inspected": False,
            "source": "pytest",
        },
        root=root,
    )
    assert len(assignment["targets"]) == 4
    receipt_paths = list(
        (
            ledger.work_dir(target_run, root)
            / "execution-session-receipts"
        ).glob("*.json")
    )
    assert len(receipt_paths) == 1
    for child in manifest["child_runs"]:
        if child["run_id"] == target_run:
            continue
        assert not list(
            (
                ledger.work_dir(child["run_id"], root)
                / "assignments"
            ).glob("*.json")
        )


def test_pre_results_policy_amendment_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    resolution = {
        "resolution_id": "ARCV1-RES010",
        "decision": "approved_pre_results_first_response_gate_removal",
        "decided_by": "repository_user",
        "applies_to_items": ["ARCV1-B001", "ARCV1-D017"],
    }
    attestation = {
        "responses_machine_produced": True,
        "repository_user_edited_response_payloads": False,
        "label_content_inspected_before_amendment": False,
        "reference_labeling_started": False,
        "comparison_started": False,
        "evaluation_started": False,
    }
    amendment = refresh.build_evaluation_policy_amendment(
        campaign_id="camp_" + "1" * 64,
        base_evaluation_policy_sha256="sha256:" + "2" * 64,
        resolution=resolution,
        owner_attestation=attestation,
    )
    assert amendment["removed_requirements"] == refresh.FIRST_RESPONSE_REQUIREMENTS
    assert amendment["amendment_id"] == (
        "amend_" + amendment["policy_amendment_sha256"].removeprefix("sha256:")
    )
    result = refresh.publish_evaluation_policy_amendment(
        amendment, root=tmp_path
    )
    assert result["status"] == "published"
    again = refresh.publish_evaluation_policy_amendment(
        amendment, root=tmp_path
    )
    assert again["status"] == "already_published"
    path = Path(result["path"])
    assert json.loads(path.read_text(encoding="utf-8")) == amendment


def test_policy_amendment_requires_exact_owner_resolution_and_pre_results_state() -> None:
    resolution = {
        "resolution_id": "ARCV1-RES010",
        "decision": "approved_pre_results_first_response_gate_removal",
        "decided_by": "repository_user",
        "applies_to_items": ["ARCV1-B001", "ARCV1-D017"],
    }
    attestation = {
        "responses_machine_produced": True,
        "repository_user_edited_response_payloads": False,
        "label_content_inspected_before_amendment": False,
        "reference_labeling_started": False,
        "comparison_started": False,
        "evaluation_started": False,
    }
    with pytest.raises(ValueError, match="owner resolution"):
        refresh.build_evaluation_policy_amendment(
            campaign_id="camp_" + "1" * 64,
            base_evaluation_policy_sha256="sha256:" + "2" * 64,
            resolution={**resolution, "decided_by": "codex"},
            owner_attestation=attestation,
        )
    with pytest.raises(ValueError, match="attestation"):
        refresh.build_evaluation_policy_amendment(
            campaign_id="camp_" + "1" * 64,
            base_evaluation_policy_sha256="sha256:" + "2" * 64,
            resolution=resolution,
            owner_attestation={**attestation, "evaluation_started": True},
        )


def test_stage3_reconstructs_atomic_sealed_groups_without_value_loss(
    tmp_path: Path,
) -> None:
    root, assignment = _initialize_combined(tmp_path, count=4)
    workflow.ingest_response(
        "combined-test",
        assignment["assignment_id"],
        _combined_response(assignment),
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    ledger.seal_run(
        "combined-test", root=root, registry_path=ledger.REGISTRY_V2_PATH
    )
    groups = stage3._run_groups("combined-test", root=root)
    assert len(groups) == 4 * 7
    topic_values = [
        json.loads(row["value_json"])
        for key, row in groups.items()
        if key[1] == "topics"
    ]
    assert len(topic_values) == 4
    assert all(isinstance(value, list) and len(value) == 1 for value in topic_values)
    constituency_values = [
        json.loads(row["value_json"])
        for key, row in groups.items()
        if key[1] == "constituencies"
    ]
    assert {value["outcome"] for value in constituency_values} == {"claim", "none"}


def test_stage3_comparison_publication_is_content_addressed(
    tmp_path: Path,
) -> None:
    semantic = {
        "comparison_version": stage3.COMPARISON_VERSION,
        "campaign_id": "camp_" + "1" * 64,
        "comparison_role": "pytest",
        "left_source": {
            "run_id": "left",
            "artifact_set_sha256": "sha256:" + "2" * 64,
        },
        "right_source": {
            "run_id": "right",
            "artifact_set_sha256": "sha256:" + "3" * 64,
        },
        "allowed_label_types": None,
        "overlap_subjects": 4,
        "overlap_subject_label_pairs": 28,
        "n_disagreements": 0,
        "disagreements": [],
    }
    sha = ledger.sha256_text(ledger.canonical_json(semantic))
    comparison = {
        **semantic,
        "comparison_id": "cmp_" + sha.removeprefix("sha256:"),
        "comparison_sha256": sha,
    }
    first = stage3.publish_comparison(comparison, root=tmp_path)
    second = stage3.publish_comparison(comparison, root=tmp_path)
    assert first["status"] == "published"
    assert second["status"] == "already_published"
    assert first["comparison_id"] == second["comparison_id"]


def test_stage3_constituency_false_negative_denominators_are_explicit() -> None:
    claim = {
        "outcome": "claim",
        "claims": [
            {
                "group_text": "workers",
                "resolved_referent": "",
                "group_type": "occupation_or_industry",
                "relation": "protected_group",
                "stance": "favorable",
                "evidence_span": "protect workers",
                "certainty": "explicit",
            }
        ],
        "unclear_reason": "",
    }
    none = {"outcome": "none", "claims": [], "unclear_reason": ""}
    unclear = {
        "outcome": "unclear",
        "claims": [],
        "unclear_reason": "bounded ambiguity",
    }
    stats = stage3._constituency_stats(
        [claim, claim, none, none],
        [none, unclear, none, claim],
    )
    detection = stats["claim_detection"]
    assert detection["claim_miss_numerator"] == 1
    assert detection["claim_miss_denominator"] == 2
    assert detection["claim_miss_rate"] == 0.5
    assert detection["predicted_none_false_omission_numerator"] == 1
    assert detection["predicted_none_false_omission_denominator"] == 2
    assert detection["predicted_none_false_omission_rate"] == 0.5


def test_stage3_entity_exact_and_partial_metrics_remain_distinct() -> None:
    gold = [
        [{"name": "Congress", "type": "institution", "stance": "neutral"}]
    ]
    pred = [
        [
            {
                "name": "Congress",
                "type": "institution",
                "stance": "adversarial",
            }
        ]
    ]
    stats = stage3._label_stats("entities", gold, pred)
    assert stats["exact"]["micro_f1"] == 0.0
    assert stats["partial_name_and_type"]["micro_f1"] == 1.0


def test_stage3_topic_metrics_normalize_sealed_list_valued_events() -> None:
    stats = stage3._label_stats(
        "topics",
        [[["economy", "labor"]], [["foreign_policy"]]],
        [["economy", "labor"], ["foreign_policy"]],
    )
    assert stats["micro_f1"] == 1.0
    assert stats["mean_jaccard"] == 1.0
    assert all(
        row["accuracy"] == 1.0
        for row in stats["per_category"].values()
    )
    draws = stage3._resampled_metric(
        label_type="topics",
        gold=[[["economy", "labor"]], [["foreign_policy"]]],
        pred=[["economy", "labor"], ["foreign_policy"]],
        indices=[[0, 1], [1, 0]],
        metric_path="micro_f1",
    )
    assert draws == [1.0, 1.0]


def test_stage3_noninferiority_uses_strict_one_sided_bound() -> None:
    interval = {
        "status": "ok",
        "n_valid_draws": stage3.BOOTSTRAP_DRAWS,
        "two_sided_95": [-0.05, 0.02],
        "one_sided_95_lower": -0.05,
        "one_sided_95_upper": 0.01,
    }
    record = stage3._difference_record(
        metric_id="pytest",
        point_difference=0.0,
        interval=interval,
        margin=-0.05,
        bound="lower",
        sample_frame="pytest",
        n_eligible=10,
        source_artifact_hashes=["sha256:" + "a" * 64],
    )
    assert record["passed"] is False


def test_stage3_freeze_is_additive_byte_identical_and_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    specs_root = tmp_path / "specs"
    bundles_root = tmp_path / "bundles"
    shutil.copytree(ledger.SPECS_ROOT, specs_root)
    shutil.copytree(refresh.BUNDLES_ROOT, bundles_root)
    evaluation_semantic = {
        "evaluation_version": stage3.EVALUATION_VERSION,
        "campaign_id": "camp_" + "1" * 64,
        "effective_evaluation_policy_sha256": "sha256:" + "2" * 64,
        "gate_summary": {
            "status": "pass",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "failed_metric_ids": [],
        },
    }
    evaluation_sha = ledger.sha256_text(
        ledger.canonical_json(evaluation_semantic)
    )
    evaluation_id = "eval_" + evaluation_sha.removeprefix("sha256:")
    ledger.atomic_write_json(
        root / "campaign_evaluations" / f"{evaluation_id}.json",
        {
            **evaluation_semantic,
            "evaluation_id": evaluation_id,
            "evaluation_sha256": evaluation_sha,
        },
    )
    candidate = json.loads(
        (
            bundles_root
            / "paragraph_judgment_v2_candidate"
            / "candidate-1.json"
        ).read_text(encoding="utf-8")
    )
    built = stage3.build_freeze_artifacts(
        evaluation_id=evaluation_id,
        root=root,
        label_registry_path=specs_root / "registry-v2.json",
        bundle_registry_path=bundles_root / "registry-v1.json",
    )
    final_bundle = built["final_bundle"]
    assert final_bundle["prompt_text"] == candidate["prompt_text"]
    assert final_bundle["response_schema"] == candidate["response_schema"]
    assert final_bundle["eligibility"] == "production_eligible"
    first = stage3.publish_freeze_artifacts(
        built,
        root=root,
        label_registry_path=specs_root / "registry-v2.json",
        bundle_registry_path=bundles_root / "registry-v1.json",
    )
    second = stage3.publish_freeze_artifacts(
        built,
        root=root,
        label_registry_path=specs_root / "registry-v2.json",
        bundle_registry_path=bundles_root / "registry-v1.json",
    )
    rebuilt = stage3.build_freeze_artifacts(
        evaluation_id=evaluation_id,
        root=root,
        label_registry_path=specs_root / "registry-v2.json",
        bundle_registry_path=bundles_root / "registry-v1.json",
    )
    assert first["status"] == "published"
    assert second["status"] == "already_published"
    assert rebuilt["manifest"]["freeze_id"] == first["freeze_id"]
    assert refresh.resolve_bundle(
        "paragraph_judgment_v2",
        "v2",
        registry_path=bundles_root / "registry-v1.json",
        label_registry_path=specs_root / "registry-v2.json",
    )["bundle_sha256"] == first["final_bundle_sha256"]


def test_invocation_audit_contract_is_proposed_and_non_executable(
    tmp_path: Path,
) -> None:
    contract = stage3.build_proposed_invocation_audit_contract()
    assert contract["scope"]["n_subjects"] == 101
    assert contract["contract_status"] == (
        "proposed_requires_explicit_owner_approval"
    )
    assert contract["labeling_authorized"] is False
    assert contract["freeze_authorized"] is False
    assert contract["historical_rule"].startswith(
        "legacy agreement is diagnostic only"
    )
    first = stage3.publish_proposed_invocation_audit_contract(
        contract, root=tmp_path
    )
    second = stage3.publish_proposed_invocation_audit_contract(
        contract, root=tmp_path
    )
    assert first["status"] == "published"
    assert second["status"] == "already_published"
    assert first["contract_id"] == second["contract_id"]


def test_draft_bundle_cannot_enter_full_shadow_campaign() -> None:
    bundle = refresh.resolve_bundle(
        "paragraph_judgment_v2_candidate", "candidate-1"
    )
    with pytest.raises(ValueError, match="non-production"):
        refresh.build_campaign_manifest(
            corpus_snapshot={
                "snapshot_id": "canonical-v1",
                "fingerprint": "sha256:" + "1" * 64,
                "subject_counts": {"paragraph": 35394},
            },
            bundles=[bundle],
            scope="full-shadow",
            selection_receipts={
                bundle["bundle_id"]: {
                    "selection_id": "sel_" + "2" * 64,
                    "selection_sha256": "sha256:" + "3" * 64,
                    "n_subjects": 35394,
                }
            },
            model_receipts={
                bundle["bundle_id"]: _exact_receipt(effort="high")
            },
            reasoning_effort={bundle["bundle_id"]: "high"},
            evaluation_policy_sha256="sha256:" + "4" * 64,
            promotion_policy_sha256="sha256:" + "5" * 64,
            approval_ids=["FUTURE-APPROVAL"],
        )


def test_delta_and_full_shadow_select_exact_key_sets() -> None:
    eligible = [
        {
            "subject_key": {"doc_name": "/a", "para_idx": 0},
            "bundle_input_sha256": "sha256:" + "a" * 64,
        },
        {
            "subject_key": {"doc_name": "/a", "para_idx": 1},
            "bundle_input_sha256": "sha256:" + "b" * 64,
        },
        {
            "subject_key": {"doc_name": "/b", "para_idx": 0},
            "bundle_input_sha256": "sha256:" + "c" * 64,
        },
    ]
    full = refresh.select_scope("full-shadow", eligible)
    assert full == sorted(
        eligible, key=lambda row: ledger.canonical_json(row["subject_key"])
    )
    prior = {
        ledger.canonical_json(eligible[0]["subject_key"]): "sha256:" + "a" * 64,
        ledger.canonical_json(eligible[1]["subject_key"]): "sha256:" + "0" * 64,
    }
    delta = refresh.select_scope(
        "delta", eligible, prior_fingerprints=prior
    )
    assert [row["subject_key"] for row in delta] == [
        {"doc_name": "/a", "para_idx": 1},
        {"doc_name": "/b", "para_idx": 0},
    ]


def test_promoted_constituency_projection_is_derived_and_grouped(
    tmp_path: Path,
) -> None:
    root, assignment = _initialize_combined(tmp_path)
    response = _combined_response(assignment)
    workflow.ingest_response(
        "combined-test",
        assignment["assignment_id"],
        response,
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
    )
    assert workflow.audit_run(
        "combined-test", root=root, registry_path=ledger.REGISTRY_V2_PATH
    )["status"] == "ok"
    workflow.next_assignment(
        "combined-test", root=root, registry_path=ledger.REGISTRY_V2_PATH
    )
    ledger.seal_run(
        "combined-test",
        root=root,
        registry_path=ledger.REGISTRY_V2_PATH,
        sealed_at="2026-07-24T20:00:00Z",
    )
    before = ledger.materialize(root)
    assert json.loads(
        (Path(before["path"]) / "generation.json").read_text(encoding="utf-8")
    )["generation_version"] == "annotation-ledger-materialization-v2"
    assert pq.read_table(
        Path(before["path"]) / "constituency_claims.parquet"
    ).num_rows == 0

    run_dir = ledger.sealed_artifact_dir("combined-test", root)
    events = ledger.read_jsonl(run_dir / "label_events.jsonl")
    event = next(row for row in events if row["label_type"] == "constituencies")
    raw_before = event["raw_value_json"]
    ledger.publish_promotions(
        [
            {
                "label_group_id": event["label_group_id"],
                "promotion_channel": "primary",
                "replaces_label_group_ids": [],
                "expected_current_state_sha256": ledger.current_state_hash([]),
                "adjudication_id": None,
            }
        ],
        operator_id="pytest",
        promotion_rule_id="constituency-projection-test",
        review_resolution_id="TEST-ONLY",
        root=root,
        promoted_at="2026-07-24T20:01:00Z",
    )
    after = ledger.materialize(root)
    claims = pq.read_table(
        Path(after["path"]) / "constituency_claims.parquet"
    ).to_pylist()
    assert len(claims) == 1
    assert claims[0]["group_text"] == "workers"
    assert claims[0]["stance"] == "favorable"
    assert claims[0]["normalized_group"] is None
    assert claims[0]["normalization_status"] == "not_normalized"
    sealed_event = next(
        row
        for row in ledger.read_jsonl(run_dir / "label_events.jsonl")
        if row["label_type"] == "constituencies"
    )
    assert sealed_event["raw_value_json"] == raw_before
    assert json.loads(raw_before) == response["annotations"][0]["constituencies"]


def test_registry_only_extension_adds_a_normal_bundle_to_all(
    tmp_path: Path,
) -> None:
    specs_root = tmp_path / "specs"
    bundles_root = tmp_path / "bundles"
    shutil.copytree(ledger.SPECS_ROOT, specs_root)
    shutil.copytree(refresh.BUNDLES_ROOT, bundles_root)
    label_registry_path = specs_root / "registry-v2.json"
    bundle_registry_path = bundles_root / "registry-v1.json"

    response_value_schema = {"type": "boolean"}
    response_schema = {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "local_id": {"type": "integer"},
                        "imaginary": response_value_schema,
                    },
                    "required": ["local_id", "imaginary"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["annotations"],
        "additionalProperties": False,
    }
    imaginary_bundle = {
        "bundle_schema_version": "annotation-label-bundle-v1",
        "bundle_id": "imaginary_bundle",
        "bundle_version": "test-v1",
        "stability": "test",
        "eligibility": "production_eligible",
        "subject_type": "paragraph",
        "emitted_label_specs": [
            {
                "label_type": "imaginary",
                "spec_version": "test-v1",
                "response_pointer": "/imaginary",
            }
        ],
        "prompt_version": "imaginary/test-v1",
        "prompt_text": "Return one imaginary boolean.",
        "prompt_sha256": ledger.sha256_text("Return one imaginary boolean."),
        "response_schema": response_schema,
        "response_schema_sha256": ledger.sha256_text(
            ledger.canonical_json(response_schema)
        ),
        "response_collection_pointer": "/annotations",
        "response_template_item": {"local_id": 0, "imaginary": False},
        "context_policy": {
            "neighbors_each_side": 0,
            "trusted_keys_model_visible": False,
        },
        "batching_policy": {
            "targets_per_assignment": 1,
            "speech_mixing": "prohibited",
            "packing": "canonical_contiguous_same_speech",
        },
        "local_identifier_policy": {
            "field": "local_id",
            "type": "integer",
            "scope": "assignment",
            "trusted_mapping": "local_program",
        },
        "bundle_validators": ["complete_local_id_set"],
        "typed_projections": [],
        "evaluation_requirements": ["test"],
        "publication_gates": ["test"],
        "limitations": ["Test-only extension."],
    }
    imaginary_bundle["bundle_sha256"] = refresh.bundle_hash(imaginary_bundle)
    imaginary_spec = {
        "spec_schema_version": "annotation-label-spec-v2",
        "label_type": "imaginary",
        "spec_version": "test-v1",
        "stability": "test",
        "subject_type": "paragraph",
        "historical_quantity": "Imaginary extensibility signal.",
        "prompt_version": "imaginary/test-v1",
        "prompt_text": "Imaginary bundle field.",
        "prompt_sha256": ledger.sha256_text("Imaginary bundle field."),
        "response_schema": response_value_schema,
        "response_schema_sha256": ledger.sha256_text(
            ledger.canonical_json(response_value_schema)
        ),
        "value_kind": "boolean",
        "event_extractor": {"mode": "single"},
        "context_policy": {"neighbors_each_side": 0},
        "special_validator": None,
        "source_spec_refs": [],
        "limitations": ["Test-only extension."],
        "execution_binding": {
            "mode": "bundle",
            "bundle_refs": [
                {
                    "bundle_id": "imaginary_bundle",
                    "bundle_version": "test-v1",
                    "bundle_sha256": imaginary_bundle["bundle_sha256"],
                    "response_pointer": "/imaginary",
                }
            ],
            "decoder": "json_pointer",
            "atomic_group_rule": "one_label_type_per_subject_response",
        },
    }
    imaginary_spec["spec_sha256"] = ledger.spec_hash(imaginary_spec)
    ledger.atomic_write_json(
        specs_root / "imaginary" / "test-v1.json", imaginary_spec
    )
    label_registry = json.loads(label_registry_path.read_text(encoding="utf-8"))
    label_registry["entries"].append(
        {
            "label_type": "imaginary",
            "spec_version": "test-v1",
            "spec_sha256": imaginary_spec["spec_sha256"],
            "stability": "test",
            "path": "imaginary/test-v1.json",
        }
    )
    label_registry["entries"] = sorted(
        label_registry["entries"],
        key=lambda row: (row["label_type"], row["spec_version"]),
    )
    label_registry["registry_sha256"] = ledger.sha256_text(
        ledger.canonical_json(
            {
                key: value
                for key, value in label_registry.items()
                if key != "registry_sha256"
            }
        )
    )
    ledger.atomic_write_json(label_registry_path, label_registry)
    ledger.atomic_write_json(
        bundles_root / "imaginary_bundle" / "test-v1.json", imaginary_bundle
    )
    bundle_registry = json.loads(bundle_registry_path.read_text(encoding="utf-8"))
    bundle_registry["label_registry_sha256"] = label_registry["registry_sha256"]
    bundle_registry["entries"].append(
        {
            "bundle_id": "imaginary_bundle",
            "bundle_version": "test-v1",
            "bundle_sha256": imaginary_bundle["bundle_sha256"],
            "eligibility": "production_eligible",
            "path": "imaginary_bundle/test-v1.json",
        }
    )
    bundle_registry["entries"] = sorted(
        bundle_registry["entries"],
        key=lambda row: (row["bundle_id"], row["bundle_version"]),
    )
    bundle_registry["registry_sha256"] = ledger.sha256_text(
        ledger.canonical_json(
            {
                key: value
                for key, value in bundle_registry.items()
                if key != "registry_sha256"
            }
        )
    )
    ledger.atomic_write_json(bundle_registry_path, bundle_registry)

    selected = refresh.select_bundles(
        "all",
        registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
    )
    assert [bundle["bundle_id"] for bundle in selected] == [
        "imaginary_bundle",
        "speech_factual",
    ]
    subjects = tmp_path / "imaginary-subjects.json"
    ledger.atomic_write_json(
        subjects,
        [{"doc_name": "/imaginary", "para_idx": 0, "text": "Imaginary text."}],
    )
    root = tmp_path / "imaginary-ledger"
    refresh.initialize_bundle_run(
        bundle_id="imaginary_bundle",
        bundle_version="test-v1",
        run_id="imaginary-bundle-run",
        subjects_path=subjects,
        model_receipt=_exact_receipt(),
        reasoning_effort="high",
        annotator_id="pytest",
        root=root,
        bundle_registry_path=bundle_registry_path,
        label_registry_path=label_registry_path,
        authorization_review_item_id="TEST-ONLY",
    )
    assignment = workflow.next_assignment(
        "imaginary-bundle-run",
        root=root,
        registry_path=label_registry_path,
    )
    workflow.ingest_response(
        "imaginary-bundle-run",
        assignment["assignment_id"],
        {"annotations": [{"local_id": 1, "imaginary": True}]},
        root=root,
        registry_path=label_registry_path,
    )
    events = ledger.read_jsonl(
        ledger.work_dir("imaginary-bundle-run", root) / "label_events.jsonl"
    )
    assert [(row["label_type"], json.loads(row["raw_value_json"])) for row in events] == [
        ("imaginary", True)
    ]


def test_refresh_core_is_provider_neutral_and_does_not_open_network(
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

    def blocked_connection(*args: object, **kwargs: object):
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", blocked_connection)
    assert refresh.read_bundle_registry()["registry_sha256"].startswith("sha256:")
    assert refresh.select_bundles("all")[0]["bundle_id"] == "speech_factual"
    stage3_source = Path(stage3.__file__).read_text(encoding="utf-8")
    assert all(
        marker not in stage3_source
        for marker in (
            "import openai",
            "import anthropic",
            "import requests",
            "import httpx",
            "socket.create_connection",
        )
    )
