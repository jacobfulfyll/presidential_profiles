from __future__ import annotations

import json
import ast
from pathlib import Path

import pytest
import pandas as pd

from presidential_profiles import annotation_ledger as ledger
from presidential_profiles import combat
from presidential_profiles import speaker_attribution as speaker


def test_registry_v3_is_additive_and_old_registry_bytes_are_unchanged():
    before_v1 = ledger.sha256_file(ledger.REGISTRY_PATH)
    before_v2 = ledger.sha256_file(ledger.REGISTRY_V2_PATH)
    registry, specs = speaker.build_registry_v3()
    assert registry["registry_version"] == "annotation-label-registry-v3"
    assert registry["entries"][: len(json.loads(ledger.REGISTRY_V2_PATH.read_text())["entries"])] != []
    assert set(specs) == {"document_speaker_class", "paragraph_speaker"}
    assert ledger.sha256_file(ledger.REGISTRY_PATH) == before_v1
    assert ledger.sha256_file(ledger.REGISTRY_V2_PATH) == before_v2
    assert ledger.read_registry(speaker.REGISTRY_V3)["registry_sha256"] == registry["registry_sha256"]


def test_runtime_receipt_uses_exact_rollout_metadata(tmp_path, monkeypatch):
    rollout = tmp_path / "rollout-thread-test.jsonl"
    rows = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": "thread-test",
                "cli_version": "test-cli",
                "originator": "test",
            },
        },
        {
            "type": "turn_context",
            "payload": {"model": "gpt-5.6-sol", "effort": "low"},
        },
    ]
    rollout.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(speaker, "_runtime_session_path", lambda _thread_id: rollout)

    receipt = speaker.capture_runtime_receipt(thread_id="thread-test")
    assert receipt["thread_id"] == "thread-test"
    assert receipt["session_id"] == receipt["thread_id"]
    assert receipt["model_id"] == "gpt-5.6-sol"
    assert receipt["reasoning_effort"] == "low"
    assert receipt["specificity"] == "exact_runtime_identifier"


def _document_packet():
    return {
        "assignment_id": "spk_document_abc",
        "paragraphs": [
            {"para_idx": 0, "text": "MODERATOR: Question."},
            {"para_idx": 1, "text": "CARTER: Answer."},
        ],
    }


def test_document_response_rejects_wrong_assignment_before_write():
    response = {
        "assignment_id": "wrong",
        "result": {
            "document_class": "multi_speaker",
            "paragraph_review_required": True,
            "evidence_para_indices": [0],
            "reason": "Explicit moderator cue.",
        },
    }
    with pytest.raises(ValueError, match="open assignment"):
        speaker.validate_document_response(_document_packet(), response)


@pytest.mark.parametrize("outcome", sorted(speaker.PARAGRAPH_CLASSES - {"canonical_president"}))
def test_non_president_outcomes_forbid_president(outcome):
    packet = {
        "assignment_id": "spk_paragraph_abc",
        "context": [{"para_idx": 0, "text": "Q."}],
    }
    response = {
        "assignment_id": packet["assignment_id"],
        "result": {
            "outcome": outcome,
            "president": "Jimmy Carter",
            "evidence_para_indices": [0],
            "reason_code": "explicit_speaker_cue",
            "reason": "Cue resolves the speaker.",
        },
    }
    with pytest.raises(ValueError, match="forbid"):
        speaker.validate_paragraph_response(packet, response)


def test_canonical_president_requires_controlled_exact_name():
    packet = {
        "assignment_id": "spk_paragraph_abc",
        "context": [{"para_idx": 0, "text": "FORD: Answer."}],
    }
    response = {
        "assignment_id": packet["assignment_id"],
        "result": {
            "outcome": "canonical_president",
            "president": "Ford",
            "evidence_para_indices": [0],
            "reason_code": "explicit_speaker_cue",
            "reason": "Explicit cue.",
        },
    }
    with pytest.raises(ValueError, match="exact controlled"):
        speaker.validate_paragraph_response(packet, response)


def test_paragraph_batch_requires_complete_locked_order_and_validates_each_label():
    packet = {
        "assignment_id": "spk_production_paragraph_batch_abc",
        "paragraphs": [
            {"para_idx": 4, "text": "Q. A question.", "source_text_sha256": "sha256:a"},
            {"para_idx": 5, "text": "THE PRESIDENT. An answer.", "source_text_sha256": "sha256:b"},
        ],
    }
    response = {
        "assignment_id": packet["assignment_id"],
        "results": [
            {"para_idx": 4, "outcome": "non_president", "president": None, "evidence_para_indices": [4], "reason_code": "explicit_speaker_cue", "reason": "Reporter question."},
            {"para_idx": 5, "outcome": "canonical_president", "president": "Jimmy Carter", "evidence_para_indices": [5], "reason_code": "explicit_speaker_cue", "reason": "Explicit presidential cue."},
        ],
    }
    accepted = speaker.validate_paragraph_batch_response(packet, response)
    assert [row["para_idx"] for row in accepted["results"]] == [4, 5]
    response["results"].reverse()
    with pytest.raises(ValueError, match="locked order"):
        speaker.validate_paragraph_batch_response(packet, response)


def test_source_census_is_key_complete():
    documents, paragraphs, meta = speaker.build_source_census()
    assert len(documents) == documents.doc_name.nunique() == 1057
    assert documents.canonical_retained.sum() == 1053
    assert len(paragraphs) == 35394
    assert not paragraphs.duplicated(["doc_name", "para_idx"]).any()
    assert meta["canonical_documents"] == 1053
    assert len(meta["controlled_presidents"]) == 45


def test_module_has_no_provider_or_network_imports():
    source = Path(speaker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"anthropic", "openai", "socket", "requests", "httpx", "urllib"}


def test_review_batch_is_blind_and_requires_locked_target_order(tmp_path, monkeypatch):
    packet = {
        "assignment_id": "review_paragraph_batch_abc",
        "target_para_indices": [1],
        "paragraphs": [
            {"para_idx": 0, "text": "Q: Question.", "source_text_sha256": "sha256:a"},
            {"para_idx": 1, "text": "THE PRESIDENT: Answer.", "source_text_sha256": "sha256:b"},
        ],
    }
    assert "primary" not in json.dumps(packet)
    response = {
        "assignment_id": packet["assignment_id"],
        "results": [{
            "para_idx": 0,
            "outcome": "non_president",
            "president": None,
            "evidence_para_indices": [0],
            "reason_code": "explicit_speaker_cue",
            "reason": "Reporter question.",
        }],
    }
    observed = [row["para_idx"] for row in response["results"]]
    assert observed != packet["target_para_indices"]


def test_review_era_buckets_are_total_at_boundaries():
    assert speaker._era_bucket(1860) == "founding_antebellum"
    assert speaker._era_bucket(1861) == "civil_war_industrial"
    assert speaker._era_bucket(1933) == "new_deal_cold_war"
    assert speaker._era_bucket(1981) == "modern"


def test_derived_attribution_is_key_complete_and_exclusions_fail_closed():
    frame = pd.read_parquet(speaker.ROOT / "paragraph_attribution_v1.parquet")
    assert len(frame) == 35394
    assert not frame.duplicated(["doc_name", "para_idx"]).any()
    excluded = frame.loc[~frame["speaker_audited_all"]]
    assert excluded["attributed_speaker"].isna().all()
    assert not excluded[["debate_excluded", "single_president_documents", "annual_message_strict"]].any().any()
    assert set(excluded["speaker_outcome"]) <= {
        "non_president", "multiple_speakers", "joint_or_shared", "scaffolding", "uncertain",
    }


@pytest.mark.parametrize(
    ("doc_fragment", "document_owner", "credited_president"),
    [
        ("october-6-1976-debate-president-gerald-ford", "Jimmy Carter", "Gerald Ford"),
        ("september-26-1960-debate", "John F. Kennedy", "Richard M. Nixon"),
        ("october-28-1980-debate-ronald-reagan", "Jimmy Carter", "Ronald Reagan"),
    ],
)
def test_cross_owner_debate_turns_credit_actual_president(
    doc_fragment, document_owner, credited_president
):
    frame = pd.read_parquet(speaker.ROOT / "paragraph_attribution_v1.parquet")
    rows = frame.loc[frame["doc_name"].str.contains(doc_fragment, regex=False)]
    assert not rows.empty
    assert set(rows["document_owner"]) == {document_owner}
    credited = rows.loc[rows["attributed_speaker"].eq(credited_president)]
    assert not credited.empty
    assert credited["speaker_audited_all"].all()


def test_president_conflict_v2_has_all_treatments_and_integral_numerators():
    treatments = pd.read_parquet(
        speaker.DATA_DIR / "combat" / "by_president_treatments_v2.parquet"
    )
    selected = pd.read_parquet(
        speaker.DATA_DIR / "combat" / "by_president_speaker_audited_v2.parquet"
    )
    target_mix = pd.read_parquet(
        speaker.DATA_DIR
        / "combat"
        / "target_mix_by_era_speaker_audited_v1.parquet"
    )
    assert len(treatments) == 225
    assert treatments.groupby("treatment")["president"].nunique().eq(45).all()
    assert len(selected) == selected["president"].nunique() == 45
    assert set(selected["schema_version"]) == {"president-conflict-v2"}
    assert set(selected["treatment"]) == {"speaker_audited_all"}
    assert len(target_mix) == 9
    assert set(target_mix["schema_version"]) == {"conflict-target-mix-v1"}
    assert set(target_mix["treatment"]) == {"speaker_audited_all"}
    assert set(target_mix["era_scheme"]) == {"trends.ERAS"}
    assert int(target_mix["n_adversarial_entities"].sum()) == 8_393
    assert int(target_mix["n_paragraphs"].sum()) == 32_531
    assert tuple(target_mix["era"]) == tuple(combat.ERA_ORDER)
    for flag in ("party_attack", "enemy_naming", "zero_sum"):
        assert (selected[f"n_{flag}"] >= 0).all()
        assert (selected[f"n_{flag}"] <= selected["n_paragraphs"]).all()
        assert selected.loc[selected["n_paragraphs"].gt(0), flag].between(0, 1).all()


def test_target_mix_pools_entity_counts_instead_of_averaging_year_shares():
    paragraphs = pd.DataFrame(
        [
            {
                "doc_name": "one",
                "para_idx": 0,
                "year": 1789,
                "speaker_audited_all": True,
            },
            {
                "doc_name": "nine",
                "para_idx": 0,
                "year": 1790,
                "speaker_audited_all": True,
            },
            {
                "doc_name": "excluded",
                "para_idx": 0,
                "year": 1790,
                "speaker_audited_all": False,
            },
        ]
    )
    entities = pd.DataFrame(
        [
            {"doc_name": "one", "para_idx": 0, "type": "nation"},
            *[
                {"doc_name": "nine", "para_idx": 0, "type": "group"}
                for _ in range(9)
            ],
            {"doc_name": "excluded", "para_idx": 0, "type": "person"},
        ]
    )
    out = combat.speaker_audited_target_mix_by_era(paragraphs, entities)
    founding = out.loc[out["era"].eq("The founding")].iloc[0]
    assert founding["adv_n_nation"] == 1
    assert founding["adv_n_group"] == 9
    assert founding["adv_n_person"] == 0
    assert founding["n_adversarial_entities"] == 10
    assert founding["n_adversarial_speeches"] == 2
