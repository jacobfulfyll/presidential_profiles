from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from presidential_profiles import annotation_backfill as backfill
from presidential_profiles import annotation_ledger as ledger
from presidential_profiles.annotate import FIELD_SPECS


def test_frozen_source_reconciliation_targets_are_rederived() -> None:
    evidence = backfill.validate_sources()
    assert set(evidence["manifests"]) == backfill.EXPECTED_MANIFESTS
    assert evidence["mapping_counts"]["primary_judgments"] == {
        "unchanged_reusable": 35_381,
        "excluded_duplicate": 751,
        "excluded_duplicate_document": 84,
        "changed_requires_reannotation": 13,
    }
    assert evidence["mapping_counts"]["opus_judgments"] == {
        "unchanged_reusable": 8_435,
        "excluded_duplicate": 106,
        "excluded_duplicate_document": 26,
        "changed_requires_reannotation": 3,
    }
    assert evidence["mapping_counts"]["primary_entities"] == {
        "unchanged_reusable": 26_570,
        "excluded_duplicate": 542,
        "excluded_duplicate_document": 94,
        "changed_requires_reannotation": 8,
    }
    assert evidence["mapping_counts"]["opus_entities"] == {
        "unchanged_reusable": 5_848,
        "excluded_duplicate": 87,
        "excluded_duplicate_document": 16,
        "changed_requires_reannotation": 3,
    }
    assert len(evidence["primary_speeches"]) == 1_057
    assert len(evidence["opus_speeches"]) == 266
    assert len(evidence["invocation"]) == 101
    assert len(evidence["agreement"]) == 135


def test_reasoning_effort_is_inferred_only_from_exact_surviving_spec_hashes() -> None:
    assert (
        FIELD_SPECS["paragraph_annotations"].prompt_hash()
        == backfill.JUDGMENT_HASH
    )
    assert (
        FIELD_SPECS["speech_annotations"].prompt_hash()
        == backfill.FACTUAL_HASH
    )
    manifests = backfill._read_manifests()
    assert backfill._manifest_effort(manifests["judgment-full-20260720"]) == (
        "medium",
        "inferred_from_versioned_spec",
        None,
    )
    assert backfill._manifest_effort(manifests["factual-full-20260720"]) == (
        "low",
        "inferred_from_versioned_spec",
        None,
    )
    assert backfill._manifest_effort(manifests["taxonomy-v1-20260719"]) == (
        "unknown",
        "unknown",
        None,
    )
    altered = dict(manifests["judgment-full-20260720"])
    altered["prompt_hash"] = altered["prompt_hash"] + "-drift"
    assert backfill._manifest_effort(altered) == ("unknown", "unknown", None)


def test_frozen_inventory_is_stable_across_read_only_validation() -> None:
    before = backfill.frozen_inventory()
    backfill.validate_sources()
    backfill.verify_frozen_inventory(before)


def _legacy_event_counter() -> tuple[pd.DataFrame, Counter]:
    generation = ledger.load_current_generation()
    events = pq.read_table(
        generation / "label_events.parquet",
        columns=[
            "run_id",
            "label_type",
            "source_table",
            "source_key_json",
            "raw_value_json",
            "event_role",
            "assigned_at",
            "labeled_at",
            "timestamp_precision",
        ],
    ).to_pandas()
    legacy = events[events["source_table"].notna()].copy()
    counter = Counter(
        zip(
            legacy["run_id"],
            legacy["label_type"],
            legacy["source_table"],
            legacy["source_key_json"],
            legacy["raw_value_json"],
            strict=True,
        )
    )
    return legacy, counter


def test_every_frozen_legacy_value_and_run_id_reconciles_exactly() -> None:
    legacy, actual = _legacy_event_counter()
    expected: Counter = Counter()
    for source_path, source_table in [
        (backfill.PARAGRAPH_PATH, "paragraph_annotations"),
        (backfill.OPUS_PARAGRAPH_PATH, "paragraph_annotations__opus4-8"),
    ]:
        for row in pd.read_parquet(source_path).itertuples(index=False):
            key = ledger.canonical_json(
                {"doc_name": str(row.doc_name), "para_idx": int(row.para_idx)}
            )
            values = {
                "topics": list(row.topics),
                "party_attack": bool(row.party_attack),
                "enemy_naming": bool(row.enemy_naming),
                "zero_sum": bool(row.zero_sum),
                "proposal_values": str(row.proposal_values),
            }
            for label_type, value in values.items():
                expected[
                    (
                        str(row.run_id),
                        label_type,
                        source_table,
                        key,
                        ledger.canonical_json(value),
                    )
                ] += 1
    for source_path, source_table in [
        (backfill.ENTITY_PATH, "paragraph_entities"),
        (backfill.OPUS_ENTITY_PATH, "paragraph_entities__opus4-8"),
    ]:
        for row in pd.read_parquet(source_path).itertuples(index=False):
            source_key = ledger.canonical_json(
                {
                    "doc_name": str(row.doc_name),
                    "para_idx": int(row.para_idx),
                    "entity": str(row.entity),
                    "type": str(row.type),
                    "stance": str(row.stance),
                }
            )
            value = ledger.canonical_json(
                {
                    "name": str(row.entity),
                    "type": str(row.type),
                    "stance": str(row.stance),
                }
            )
            expected[
                (str(row.run_id), "entities", source_table, source_key, value)
            ] += 1
    for source_path, source_table in [
        (backfill.SPEECH_PATH, "speech_annotations"),
        (backfill.OPUS_SPEECH_PATH, "speech_annotations__opus4-8"),
    ]:
        for row in pd.read_parquet(source_path).itertuples(index=False):
            key = ledger.canonical_json({"doc_name": str(row.doc_name)})
            for label_type in ["speech_type", "audience", "medium"]:
                expected[
                    (
                        str(row.run_id),
                        label_type,
                        source_table,
                        key,
                        ledger.canonical_json(str(getattr(row, label_type))),
                    )
                ] += 1
    for row in pd.read_parquet(backfill.INVOCATION_PATH).itertuples(index=False):
        key = ledger.canonical_json(
            {
                "char_end": int(row.char_end),
                "char_start": int(row.char_start),
                "doc_name": str(row.doc_name),
            }
        )
        expected[
            (
                str(row.run_id),
                "invocation_tone",
                "invocation_tone",
                key,
                ledger.canonical_json(str(row.label)),
            )
        ] += 1
    assert actual == expected
    assert len(legacy) == 261_233
    assert not legacy["assigned_at"].notna().any()
    assert not legacy["labeled_at"].notna().any()
    assert set(legacy["timestamp_precision"]) == {"date"}
    legacy_entities = legacy[legacy["label_type"].eq("entities")]
    assert len(legacy_entities) == 27_214 + 5_954
    assert set(legacy_entities["event_role"]) == {"value"}


def test_canonical_projection_and_promotions_match_phase1_contract() -> None:
    generation = ledger.load_current_generation()
    events = pq.read_table(
        generation / "label_events.parquet",
        columns=[
            "label_id",
            "label_group_id",
            "run_id",
            "label_type",
            "doc_name",
            "para_idx",
            "source_text_sha256",
        ],
    ).to_pandas()
    projections = pq.read_table(
        generation / "canonical_label_projection.parquet"
    ).to_pandas()
    current = pq.read_table(generation / "current_labels.parquet").to_pandas()
    assert set(events["label_id"]) == set(projections["label_id"])
    assert len(events) == len(projections) == 261_335

    excluded = projections[
        projections["mapping_status"].isin(
            ["excluded_duplicate", "excluded_duplicate_document"]
        )
    ]
    assert len(excluded) == 4_914 + 678
    assert not excluded["eligible_for_promotion"].any()
    assert set(excluded["label_id"]).isdisjoint(set(current["label_id"]))

    changed = projections[
        projections["mapping_status"].eq("changed_requires_reannotation")
    ]
    assert len(changed) == 91
    assert not changed["eligible_for_promotion"].any()
    assert set(changed["label_id"]).isdisjoint(set(current["label_id"]))
    changed_events = events[events["label_id"].isin(changed["label_id"])]
    primary_changed = changed_events[
        ~changed_events["run_id"].str.startswith("opus-")
    ]
    assert len(set(zip(primary_changed["doc_name"], primary_changed["para_idx"]))) == 13

    overlay = events[
        events["run_id"].eq("correction-overlay-13-codex-20260724-r2")
    ]
    assert len(overlay) == 102
    assert len(set(zip(overlay["doc_name"], overlay["para_idx"]))) == 13
    overlay_projection = projections[
        projections["label_id"].isin(overlay["label_id"])
    ]
    assert set(overlay_projection["mapping_status"]) == {
        "direct_canonical_annotation"
    }
    assert (
        overlay_projection["source_text_sha256"]
        == overlay_projection["canonical_text_sha256"]
    ).all()
    assert set(overlay["label_id"]) <= set(current["label_id"])

    judgment_types = {
        "topics",
        "party_attack",
        "enemy_naming",
        "zero_sum",
        "proposal_values",
    }
    judgment = current[current["label_type"].isin(judgment_types)]
    coverage = judgment.groupby(
        ["canonical_doc_name", "canonical_para_idx"]
    )["label_type"].nunique()
    assert len(coverage) == 35_394
    assert set(coverage) == {5}
    speech = current[
        current["label_type"].isin(["speech_type", "audience", "medium"])
    ]
    speech_coverage = speech.groupby("canonical_doc_name")["label_type"].nunique()
    assert len(speech_coverage) == 1_053
    assert set(speech_coverage) == {3}
    assert not current["run_id"].str.startswith("opus-").any()


def test_april3_is_the_only_promotable_legacy_input_mismatch() -> None:
    generation = ledger.load_current_generation()
    projections = pq.read_table(
        generation / "canonical_label_projection.parquet",
        columns=[
            "label_id",
            "canonical_doc_name",
            "mapping_status",
            "mapping_rule_id",
            "source_input_sha256",
            "canonical_input_sha256",
            "eligible_for_promotion",
            "review_item_id",
        ],
    ).to_pandas()
    mismatch = projections[
        projections["source_input_sha256"].notna()
        & projections["canonical_input_sha256"].notna()
        & projections["source_input_sha256"].ne(
            projections["canonical_input_sha256"]
        )
        & projections["eligible_for_promotion"]
    ]
    assert len(mismatch) == 6
    assert set(mismatch["canonical_doc_name"]) == {backfill.APRIL3}
    assert set(mismatch["mapping_rule_id"]) == {
        "speech-input-mismatch-human-approved-april-3-1968-v1"
    }
    assert set(mismatch["review_item_id"]) == {"ALRV1-D007"}
    assert set(mismatch["source_input_sha256"]) == {backfill.APRIL3_OLD_INPUT}
    assert set(mismatch["canonical_input_sha256"]) == {backfill.APRIL3_NEW_INPUT}


def test_seals_frozen_bytes_provenance_and_typed_projection_verify() -> None:
    generation = ledger.load_current_generation()
    runs = pq.read_table(generation / "annotation_runs.parquet").to_pandas()
    assert len(runs) == 16
    legacy = runs[runs["run_kind"].eq("legacy_backfill")]
    assert len(legacy) == 15
    assert not legacy["assigned_at"].notna().any()
    assert not legacy["labeled_at"].notna().any()
    unknown = legacy[legacy["reasoning_effort"].eq("unknown")]
    assert set(unknown["run_id"]) == {
        "2026-07-12-invocation-tone-fable5",
        "taxonomy-v1-20260719",
    }
    assert set(unknown["reasoning_effort_provenance"]) == {"unknown"}
    overlay = runs[
        runs["run_id"].eq("correction-overlay-13-codex-20260724-r2")
    ].iloc[0]
    assert overlay["model_id"] == "gpt-5.6-sol"
    assert overlay["model_identity_source"] == "runtime_metadata"
    assert overlay["reasoning_effort"] == "xhigh"
    assert overlay["human_validation_status"] == "not_human_validated"
    assert overlay["independent_check_status"] == "not_independently_checked"

    for run_id in ledger.sealed_runs():
        ledger.verify_sealed_run(run_id)
    inventory_root = ledger.LEDGER_ROOT / "input_inventories"
    inventory_hash = (inventory_root / "current").read_text(
        encoding="ascii"
    ).strip()
    inventory = json.loads(
        (
            inventory_root / inventory_hash / "source-inventory.json"
        ).read_text(encoding="utf-8")
    )
    assert len(inventory) == 37
    backfill.verify_frozen_inventory(inventory)
    assert pq.read_metadata(
        generation / "constituency_claims.parquet"
    ).num_rows == 0
