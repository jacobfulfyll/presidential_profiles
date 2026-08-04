"""Legacy backfill and canonical projection for the annotation ledger.

The module reads frozen paid artifacts but never writes below
``data/llm_annotations``.  It validates every reconciliation target before
creating the registry or a work run, then emits one immutable sealed set per
legacy manifest.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import annotation_ledger as ledger
from . import constituency_labeling as constituency
from .corpus import DATA_DIR, REPO_ROOT
from .prompts import annotation_v1 as av1


LLM_ROOT = DATA_DIR / "llm_annotations"
MANIFEST_ROOT = LLM_ROOT / "manifests"
CORRECTION_ROOT = DATA_DIR / "corpus_corrections"

PARAGRAPH_PATH = LLM_ROOT / "paragraph_annotations.parquet"
OPUS_PARAGRAPH_PATH = LLM_ROOT / "paragraph_annotations__opus4-8.parquet"
ENTITY_PATH = LLM_ROOT / "paragraph_entities.parquet"
OPUS_ENTITY_PATH = LLM_ROOT / "paragraph_entities__opus4-8.parquet"
SPEECH_PATH = LLM_ROOT / "speech_annotations.parquet"
OPUS_SPEECH_PATH = LLM_ROOT / "speech_annotations__opus4-8.parquet"
INVOCATION_PATH = LLM_ROOT / "invocation_tone.parquet"
AGREEMENT_PATH = LLM_ROOT / "agreement_v1.parquet"
AGREEMENT_SAMPLE_PATH = LLM_ROOT / "agreement_sample_v1.json"

MAPPING_PATH = CORRECTION_ROOT / "paragraph_key_mapping_v1.parquet"
CANONICAL_PARAGRAPHS_PATH = CORRECTION_ROOT / "canonical_paragraphs_v1.parquet"
CANONICAL_SPEECHES_PATH = CORRECTION_ROOT / "canonical_speeches_v1.parquet"
CORRECTION_META_PATH = CORRECTION_ROOT / "meta_v1.json"
REANNOTATION_PATH = CORRECTION_ROOT / "reannotation_required_v1.parquet"

OLD_PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
OLD_SPEECHES_PATH = DATA_DIR / "speeches.parquet"

EXPECTED_MANIFESTS = {
    "2026-07-12-invocation-tone-fable5",
    "factual-full-20260720",
    "factual-r2-20260720",
    "judgment-chunk1-20260720",
    "judgment-chunk10-20260720",
    "judgment-chunk2-20260720",
    "judgment-chunk25-20260720",
    "judgment-chunk5-20260720",
    "judgment-full-20260720",
    "judgment-r2-20260720",
    "opus-factual-20260721",
    "opus-judgment-20260721",
    "opus-judgment-r2-20260722",
    "pilot-v2-20260720",
    "taxonomy-v1-20260719",
}

PRIMARY_RUNS = {
    "2026-07-12-invocation-tone-fable5",
    "factual-full-20260720",
    "factual-r2-20260720",
    "judgment-chunk1-20260720",
    "judgment-chunk10-20260720",
    "judgment-chunk25-20260720",
    "judgment-chunk5-20260720",
    "judgment-full-20260720",
    "judgment-r2-20260720",
    "pilot-v2-20260720",
}

JUDGMENT_HASH = (
    "sha256:6fa617391761c6b7b0e934e3a4782aa28d259d15678e4e4e4665c8b2d93ecdfe"
)
FACTUAL_HASH = (
    "sha256:450c667db92e324c1a1455914dd17528a999c88d3e32989e0d32fb3a0a7e95f1"
)
APRIL3 = "/the-presidency/presidential-speeches/april-3-1968-press-conference"
APRIL3_OLD_INPUT = (
    "sha256:3754024e1ebd4990f5d8f4a1a7699654ff06ee3f0995803ec468f136b0ab2044"
)
APRIL3_NEW_INPUT = (
    "sha256:8cc60b9c0c3aaadab673853cc3e13f0973593a724004eb3a5030784cb2dbcd0f"
)


def _value_schema(label_type: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    item = av1.JUDGMENT_SCHEMA["properties"]["annotations"]["items"]["properties"]
    if label_type == "topics":
        return item["topics"], "list", {"mode": "single"}
    if label_type in {"party_attack", "enemy_naming", "zero_sum"}:
        return item[label_type], "boolean", {"mode": "single"}
    if label_type == "proposal_values":
        return item[label_type], "category", {"mode": "single"}
    if label_type == "entities":
        return item["entities"], "object", {"mode": "items"}
    if label_type in {"speech_type", "audience", "medium"}:
        return av1.FACTUAL_SCHEMA["properties"][label_type], "category", {
            "mode": "single"
        }
    if label_type == "invocation_tone":
        return {"type": "string", "enum": ["C", "N", "R"]}, "category", {
            "mode": "single"
        }
    raise KeyError(label_type)


def _constituency_schema() -> dict[str, Any]:
    claim = {
        "type": "object",
        "properties": {
            "group_text": {"type": "string"},
            "normalized_group": {"type": "string"},
            "group_type": {
                "type": "string",
                "enum": list(constituency.GROUP_TYPES),
            },
            "relation": {"type": "string", "enum": list(constituency.RELATIONS)},
            "evidence_span": {"type": "string"},
            "certainty": {
                "type": "string",
                "enum": list(constituency.CERTAINTIES),
            },
            "notes": {"type": "string"},
        },
        "required": [
            "group_text",
            "normalized_group",
            "group_type",
            "relation",
            "evidence_span",
            "certainty",
            "notes",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": list(constituency.OUTCOMES),
            },
            "claims": {"type": "array", "items": claim},
            "unclear_reason": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["outcome", "claims", "unclear_reason", "rationale"],
        "additionalProperties": False,
    }


def _make_spec(
    *,
    label_type: str,
    spec_version: str,
    stability: str,
    subject_type: str,
    historical_quantity: str,
    prompt_version: str,
    prompt_text: str,
    response_schema: dict[str, Any],
    kind: str,
    extractor: dict[str, Any],
    context_policy: dict[str, Any],
    special_validator: str | None,
    source_spec_refs: list[dict[str, Any]],
    limitations: list[str],
) -> dict[str, Any]:
    spec = {
        "label_type": label_type,
        "spec_version": spec_version,
        "stability": stability,
        "subject_type": subject_type,
        "historical_quantity": historical_quantity,
        "prompt_version": prompt_version,
        "prompt_text": prompt_text,
        "prompt_sha256": ledger.sha256_text(prompt_text),
        "response_schema": response_schema,
        "response_schema_sha256": ledger.sha256_text(
            ledger.canonical_json(response_schema)
        ),
        "value_kind": kind,
        "event_extractor": extractor,
        "context_policy": context_policy,
        "special_validator": special_validator,
        "source_spec_refs": source_spec_refs,
        "limitations": limitations,
    }
    spec["spec_sha256"] = ledger.spec_hash(spec)
    return ledger.validate_spec(spec)


def initial_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    judgment_prompt = av1.JUDGMENT_RUBRIC + "\n\n" + av1.JUDGMENT_INSTRUCTION
    for label_type in [
        "topics",
        "party_attack",
        "enemy_naming",
        "zero_sum",
        "proposal_values",
        "entities",
    ]:
        schema, kind, extractor = _value_schema(label_type)
        specs.append(
            _make_spec(
                label_type=label_type,
                spec_version="paragraph-judgment-v1",
                stability="frozen",
                subject_type="paragraph",
                historical_quantity=f"Frozen paragraph judgment field: {label_type}.",
                prompt_version="paragraph-judgment/v1",
                prompt_text=judgment_prompt,
                response_schema=schema,
                kind=kind,
                extractor=extractor,
                context_policy={
                    "mask": ["president", "party", "title"],
                    "era": "decade",
                    "overlay_neighbors_each_side": 1,
                },
                special_validator=None,
                source_spec_refs=[
                    {
                        "module": "presidential_profiles.prompts.annotation_v1",
                        "source_prompt_hash": JUDGMENT_HASH,
                    }
                ],
                limitations=[],
            )
        )
    factual_prompt = av1.FACTUAL_RUBRIC + "\n\n" + av1.FACTUAL_INSTRUCTION
    for label_type in ["speech_type", "audience", "medium"]:
        schema, kind, extractor = _value_schema(label_type)
        specs.append(
            _make_spec(
                label_type=label_type,
                spec_version="speech-factual-v1",
                stability="frozen",
                subject_type="speech",
                historical_quantity=f"Frozen speech factual field: {label_type}.",
                prompt_version="speech-factual/v1",
                prompt_text=factual_prompt,
                response_schema=schema,
                kind=kind,
                extractor=extractor,
                context_policy={"title": True, "year": True, "opening_paragraphs": 5},
                special_validator=None,
                source_spec_refs=[
                    {
                        "module": "presidential_profiles.prompts.annotation_v1",
                        "source_prompt_hash": FACTUAL_HASH,
                    }
                ],
                limitations=[],
            )
        )
    invocation_schema, kind, extractor = _value_schema("invocation_tone")
    specs.append(
        _make_spec(
            label_type="invocation_tone",
            spec_version="legacy-2026-07-12",
            stability="legacy_partial",
            subject_type="invocation_span",
            historical_quantity="Tone of a named presidential invocation.",
            prompt_version="invocation_tone/in-session-2026-07-12",
            prompt_text=(
                "Original prompt did not survive. Surviving method: classify each "
                "mention window as R reverent, C critical, or N neutral/procedural."
            ),
            response_schema=invocation_schema,
            kind=kind,
            extractor=extractor,
            context_policy={"historical_claimed_window": 110, "actual_window": 130},
            special_validator=None,
            source_spec_refs=[
                {
                    "surviving_method_hash": (
                        "sha256:5952db301a5c0e4aa1f7d567a467df2cf66b37f7744b7f35"
                        "cb4595bf391d7845"
                    )
                }
            ],
            limitations=[
                "Original prompt and cost are unknown.",
                "The surviving source hash identifies the method string, not a prompt.",
            ],
        )
    )
    constituency_schema = _constituency_schema()
    specs.append(
        _make_spec(
            label_type="constituencies",
            spec_version="draft-2026-07-24",
            stability="draft_pilot",
            subject_type="paragraph",
            historical_quantity=(
                "Whose authority, welfare, rights, or interests does the presidential "
                "voice explicitly invoke as a constituency for presidential action?"
            ),
            prompt_version="constituency-rubric-draft-2026-07-24",
            prompt_text=constituency.PROMPT,
            response_schema=constituency_schema,
            kind="object",
            extractor={"mode": "single"},
            context_policy={
                "neighbors_each_side": 1,
                "evidence_must_be_target_substring": True,
            },
            special_validator="constituency_evidence",
            source_spec_refs=[
                {
                    "draft_prompt_sha256": "sha256:" + constituency.PROMPT_SHA256,
                    "draft_schema_sha256": "sha256:" + constituency.SCHEMA_SHA256,
                }
            ],
            limitations=[
                "Draft pilot candidate only; not publication version 1.",
                "No production constituency run is authorized in Phase 1.",
            ],
        )
    )
    return specs


def write_initial_registry(root: Path = ledger.LEDGER_ROOT) -> dict[str, Any]:
    specs = initial_specs()
    specs_root = Path(root) / "specs"
    entries = []
    for spec in specs:
        path = (
            specs_root
            / spec["label_type"]
            / f"{spec['spec_version']}.json"
        )
        ledger.atomic_write_json(path, spec)
        entries.append(
            {
                "label_type": spec["label_type"],
                "spec_version": spec["spec_version"],
                "spec_sha256": spec["spec_sha256"],
                "stability": spec["stability"],
                "path": str(path.relative_to(specs_root)),
            }
        )
    registry = {
        "registry_version": "annotation-label-registry-v1",
        "entries": sorted(
            entries, key=lambda row: (row["label_type"], row["spec_version"])
        ),
    }
    registry["registry_sha256"] = ledger.sha256_text(
        ledger.canonical_json(registry)
    )
    path = specs_root / "registry-v1.json"
    ledger.atomic_write_json(path, registry)
    ledger.read_registry(path)
    return registry


def _read_manifests() -> dict[str, dict[str, Any]]:
    paths = sorted(MANIFEST_ROOT.glob("*.json"))
    manifests = {
        json.loads(path.read_text(encoding="utf-8"))["run_id"]: {
            **json.loads(path.read_text(encoding="utf-8")),
            "_path": path,
        }
        for path in paths
    }
    if set(manifests) != EXPECTED_MANIFESTS:
        raise ValueError(
            "manifest set mismatch: "
            f"missing={sorted(EXPECTED_MANIFESTS - set(manifests))}, "
            f"extra={sorted(set(manifests) - EXPECTED_MANIFESTS)}"
        )
    return manifests


def _mapped_counts(
    frame: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    joined = frame.merge(
        mapping[
            [
                "old_doc_name",
                "old_para_idx",
                "canonical_doc_name",
                "canonical_para_idx",
                "status",
                "old_text_sha256",
                "canonical_text_sha256",
            ]
        ],
        left_on=["doc_name", "para_idx"],
        right_on=["old_doc_name", "old_para_idx"],
        how="left",
        validate="many_to_one",
    )
    if len(joined) != len(frame) or joined["status"].isna().any():
        raise ValueError("annotation/mapping key-set mismatch")
    return joined, {
        str(key): int(value)
        for key, value in joined["status"].value_counts().to_dict().items()
    }


def validate_sources() -> dict[str, Any]:
    manifests = _read_manifests()
    mapping = pd.read_parquet(MAPPING_PATH)
    if mapping.duplicated(["old_doc_name", "old_para_idx"]).any():
        raise ValueError("correction mapping has duplicate old keys")
    frames = {
        "primary_judgments": pd.read_parquet(PARAGRAPH_PATH),
        "opus_judgments": pd.read_parquet(OPUS_PARAGRAPH_PATH),
        "primary_entities": pd.read_parquet(ENTITY_PATH),
        "opus_entities": pd.read_parquet(OPUS_ENTITY_PATH),
    }
    expected_total = {
        "primary_judgments": 36_229,
        "opus_judgments": 8_570,
        "primary_entities": 27_214,
        "opus_entities": 5_954,
    }
    expected_reusable = {
        "primary_judgments": 35_381,
        "opus_judgments": 8_435,
        "primary_entities": 26_570,
        "opus_entities": 5_848,
    }
    mapped: dict[str, pd.DataFrame] = {}
    mapping_counts: dict[str, dict[str, int]] = {}
    for name, frame in frames.items():
        if len(frame) != expected_total[name]:
            raise ValueError(f"{name}: {len(frame)} != {expected_total[name]}")
        unknown_runs = set(frame["run_id"]) - set(manifests)
        if unknown_runs:
            raise ValueError(f"{name}: unknown run IDs {sorted(unknown_runs)}")
        joined, counts = _mapped_counts(frame, mapping)
        if counts.get("unchanged_reusable", 0) != expected_reusable[name]:
            raise ValueError(f"{name}: reusable reconciliation mismatch {counts}")
        mapped[name] = joined
        mapping_counts[name] = counts

    primary_speeches = pd.read_parquet(SPEECH_PATH)
    opus_speeches = pd.read_parquet(OPUS_SPEECH_PATH)
    invocation = pd.read_parquet(INVOCATION_PATH)
    agreement = pd.read_parquet(AGREEMENT_PATH)
    canonical_speeches = pd.read_parquet(CANONICAL_SPEECHES_PATH)
    canonical_docs = set(canonical_speeches["doc_name"].astype(str))
    required = {
        "primary speech rows": (len(primary_speeches), 1_057),
        "Opus speech rows": (len(opus_speeches), 266),
        "invocation rows": (len(invocation), 101),
        "agreement rows": (len(agreement), 135),
        "canonical speeches": (len(canonical_speeches), 1_053),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError(f"{name}: {actual} != {expected}")
    if int(primary_speeches["doc_name"].isin(canonical_docs).sum()) != 1_053:
        raise ValueError("primary retained speech reconciliation mismatch")
    if int(opus_speeches["doc_name"].isin(canonical_docs).sum()) != 264:
        raise ValueError("Opus retained speech reconciliation mismatch")
    for frame in [primary_speeches, opus_speeches, invocation]:
        unknown_runs = set(frame["run_id"]) - set(manifests)
        if unknown_runs:
            raise ValueError(f"unknown source run IDs: {sorted(unknown_runs)}")
    transcript = canonical_speeches.set_index("doc_name")["transcript"]
    exact_invocations = 0
    for row in invocation.itertuples(index=False):
        if row.doc_name in transcript.index and str(transcript.loc[row.doc_name])[
            int(row.char_start) : int(row.char_end)
        ] == row.mention:
            exact_invocations += 1
    if exact_invocations != 101:
        raise ValueError(f"invocation exact-span mismatch: {exact_invocations}/101")
    correction_meta = json.loads(CORRECTION_META_PATH.read_text(encoding="utf-8"))
    if correction_meta["canonical_counts"] != {
        "paragraphs": 35_394,
        "speeches": 1_053,
    }:
        raise ValueError("canonical correction counts drift")
    if len(pd.read_parquet(REANNOTATION_PATH)) != 13:
        raise ValueError("correction overlay selection is no longer 13")
    correction_manifest = json.loads(
        (CORRECTION_ROOT / "manifest_v1.json").read_text(encoding="utf-8")
    )
    if ledger.sha256_text(ledger.canonical_json(correction_manifest)) != (
        "sha256:" + correction_meta["manifest_sha256"]
    ):
        raise ValueError("correction manifest semantic fingerprint drift")
    for filename, digest in correction_meta["output_fingerprints"].items():
        if ledger.sha256_file(CORRECTION_ROOT / filename) != "sha256:" + digest:
            raise ValueError(f"correction output byte fingerprint drift: {filename}")
    expected_source_hashes = {
        OLD_PARAGRAPHS_PATH: correction_meta["source_fingerprints"][
            "paragraphs_parquet_sha256"
        ],
        OLD_SPEECHES_PATH: correction_meta["source_fingerprints"][
            "speeches_parquet_sha256"
        ],
        DATA_DIR / "raw" / "miller_center_speeches.tgz": correction_meta[
            "source_fingerprints"
        ]["tarball_sha256"],
    }
    for path, digest in expected_source_hashes.items():
        if ledger.sha256_file(path) != "sha256:" + digest:
            raise ValueError(f"correction source byte fingerprint drift: {path.name}")
    return {
        "manifests": manifests,
        "mapping": mapping,
        "mapped": mapped,
        "mapping_counts": mapping_counts,
        "primary_speeches": primary_speeches,
        "opus_speeches": opus_speeches,
        "invocation": invocation,
        "agreement": agreement,
        "canonical_speeches": canonical_speeches,
        "canonical_paragraphs": pd.read_parquet(CANONICAL_PARAGRAPHS_PATH),
        "correction_meta": correction_meta,
    }


def frozen_inventory() -> list[dict[str, Any]]:
    paths = sorted(p for p in LLM_ROOT.rglob("*") if p.is_file())
    paths.extend(p for p in CORRECTION_ROOT.rglob("*") if p.is_file())
    paths.extend(
        [
            DATA_DIR / "raw" / "miller_center_speeches.tgz",
            OLD_SPEECHES_PATH,
            OLD_PARAGRAPHS_PATH,
        ]
    )
    unique = sorted({path.resolve() for path in paths})
    return [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": ledger.sha256_file(path),
        }
        for path in unique
    ]


def publish_complete_input_inventory(
    *, root: Path = ledger.LEDGER_ROOT
) -> dict[str, Any]:
    """Publish a complete content-addressed frozen-input byte inventory."""
    inventory = frozen_inventory()
    inventory_sha256 = ledger.sha256_text(ledger.canonical_json(inventory))
    parent = Path(root) / "input_inventories"
    final = parent / inventory_sha256.removeprefix("sha256:")
    if final.exists():
        existing = json.loads(
            (final / "source-inventory.json").read_text(encoding="utf-8")
        )
        if existing != inventory:
            raise ValueError("input inventory hash collision or corruption")
    else:
        temporary = parent / f".{inventory_sha256.removeprefix('sha256:')}.tmp"
        temporary.mkdir(parents=True)
        ledger.atomic_write_json(temporary / "source-inventory.json", inventory)
        members = [
            {
                "path": "source-inventory.json",
                "bytes": (temporary / "source-inventory.json").stat().st_size,
                "sha256": ledger.sha256_file(
                    temporary / "source-inventory.json"
                ),
            }
        ]
        ledger.atomic_write_json(
            temporary / "seal.json",
            {
                "artifact_set_sha256": ledger.artifact_root(members),
                "inventory_sha256": inventory_sha256,
                "members": members,
            },
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final)
        ledger.fsync_directory(final.parent)
    ledger.atomic_write_text(
        parent / "current",
        inventory_sha256.removeprefix("sha256:") + "\n",
    )
    return {
        "status": "published",
        "inventory_sha256": inventory_sha256,
        "n_files": len(inventory),
        "path": str(final),
    }


def _manifest_effort(manifest: Mapping[str, Any]) -> tuple[str, str, str | None]:
    prompt_hash = str(manifest.get("prompt_hash", ""))
    if manifest["run_id"] == "pilot-v2-20260720":
        expected = (
            "paragraph_annotations=" + JUDGMENT_HASH
            + ",speech_annotations=" + FACTUAL_HASH
        )
        if prompt_hash == expected:
            return (
                "mixed",
                "inferred_from_versioned_spec",
                ledger.canonical_json(
                    {
                        "paragraph-judgment-v1": "medium",
                        "speech-factual-v1": "low",
                    }
                ),
            )
        return "unknown", "unknown", None
    if prompt_hash.endswith(JUDGMENT_HASH):
        return "medium", "inferred_from_versioned_spec", None
    if prompt_hash.endswith(FACTUAL_HASH):
        return "low", "inferred_from_versioned_spec", None
    return "unknown", "unknown", None


@lru_cache(maxsize=None)
def _source_artifact(path: Path) -> dict[str, Any]:
    digest = ledger.sha256_file(path)
    return {
        "artifact_id": "art_" + digest.removeprefix("sha256:"),
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _legacy_event(
    *,
    run_id: str,
    spec: Mapping[str, Any],
    subject_type: str,
    key: Mapping[str, Any],
    value: Any,
    source_response_id: str,
    source_path: Path,
    source_table: str,
    source_key: Mapping[str, Any],
    duplicate_ordinal: int,
    item_index: int | None,
    group_size: int | None,
    source_text_sha256: str | None,
    ingested_at: str,
) -> dict[str, Any]:
    subject_id = ledger.subject_identity(subject_type, key)
    group_id = ledger.content_id(
        "lgrp",
        {
            "label_type": spec["label_type"],
            "run_id": run_id,
            "source_response_id": source_response_id,
            "spec_sha256": spec["spec_sha256"],
            "subject_id": subject_id,
        },
    )
    raw = ledger.canonical_json(value)
    raw_hash = ledger.sha256_text(raw)
    event = {
        "label_group_id": group_id,
        "run_id": run_id,
        "label_type": spec["label_type"],
        "spec_version": spec["spec_version"],
        "spec_sha256": spec["spec_sha256"],
        "subject_type": subject_type,
        "subject_id": subject_id,
        "subject_key_json": ledger.canonical_json(dict(key)),
        "doc_name": key.get("doc_name"),
        "para_idx": key.get("para_idx"),
        "char_start": key.get("char_start"),
        "char_end": key.get("char_end"),
        "source_text_sha256": source_text_sha256,
        "assignment_input_sha256": None,
        "event_role": "value",
        "item_index": item_index,
        "group_size": group_size,
        "value_kind": spec["value_kind"],
        "raw_value_json": raw,
        "raw_value_sha256": raw_hash,
        "assignment_id": None,
        "batch_id": None,
        "source_response_id": source_response_id,
        "source_artifact_id": _source_artifact(source_path)["artifact_id"],
        "source_table": source_table,
        "source_key_json": ledger.canonical_json(dict(source_key)),
        "source_duplicate_ordinal": duplicate_ordinal,
        "assigned_at": None,
        "labeled_at": None,
        "ingested_at": ingested_at,
        "timestamp_precision": "date",
        "provenance_status_json": ledger.canonical_json(
            {
                "keys": "surviving_frozen_row",
                "timestamps": "unknown_except_manifest_date",
                "value": "surviving_final_label",
            }
        ),
    }
    event["label_id"] = ledger.content_id(
        "lbl",
        {
            "event_role": "value",
            "item_index": item_index,
            "label_group_id": group_id,
            "raw_value_sha256": raw_hash,
            "source_duplicate_ordinal": duplicate_ordinal,
        },
    )
    return event


def _paragraph_projection(
    event: Mapping[str, Any], mapping_row: Mapping[str, Any]
) -> dict[str, Any]:
    status = str(mapping_row["status"])
    canonical_key = None
    canonical_id = None
    eligible = status == "unchanged_reusable"
    if status in {"unchanged_reusable", "changed_requires_reannotation"}:
        canonical_key = {
            "doc_name": str(mapping_row["canonical_doc_name"]),
            "para_idx": int(mapping_row["canonical_para_idx"]),
        }
        canonical_id = ledger.subject_identity("paragraph", canonical_key)
    rule = {
        "unchanged_reusable": "paragraph-exact-text-continuity-v1",
        "changed_requires_reannotation": "paragraph-changed-requires-overlay-v1",
        "excluded_duplicate": "paragraph-excluded-internal-repeat-v1",
        "excluded_duplicate_document": "paragraph-excluded-duplicate-document-v1",
    }[status]
    return {
        "projection_id": ledger.content_id(
            "proj",
            {
                "label_id": event["label_id"],
                "canonical_subject_id": canonical_id,
                "mapping_rule_id": rule,
            },
        ),
        "label_id": event["label_id"],
        "label_group_id": event["label_group_id"],
        "source_subject_id": event["subject_id"],
        "canonical_subject_id": canonical_id,
        "canonical_subject_key_json": (
            ledger.canonical_json(canonical_key) if canonical_key else None
        ),
        "canonical_doc_name": canonical_key["doc_name"] if canonical_key else None,
        "canonical_para_idx": canonical_key["para_idx"] if canonical_key else None,
        "mapping_status": (
            "reusable_exact_text" if status == "unchanged_reusable" else status
        ),
        "mapping_rule_id": rule,
        "source_text_sha256": "sha256:" + str(mapping_row["old_text_sha256"]),
        "canonical_text_sha256": (
            "sha256:" + str(mapping_row["canonical_text_sha256"])
            if canonical_key
            else None
        ),
        "source_input_sha256": None,
        "canonical_input_sha256": None,
        "eligible_for_promotion": eligible,
        "review_item_id": "ALRV1-D004" if eligible else None,
        "reason": (
            "Source and canonical paragraph text fingerprints are identical."
            if eligible
            else "Historical event is retained but is not continuity-promotion eligible."
        ),
    }


def _factual_input(
    speeches: pd.DataFrame, paragraphs: pd.DataFrame, doc_name: str
) -> str:
    speech = speeches[speeches["doc_name"].eq(doc_name)]
    if len(speech) != 1:
        raise ValueError(f"speech lookup failed: {doc_name}")
    paras = paragraphs[paragraphs["doc_name"].eq(doc_name)].sort_values("para_idx")
    return av1.factual_context(speech.iloc[0], paras)


def _speech_projection(
    event: Mapping[str, Any],
    *,
    old_speeches: pd.DataFrame,
    old_paragraphs: pd.DataFrame,
    canonical_speeches: pd.DataFrame,
    canonical_paragraphs: pd.DataFrame,
) -> dict[str, Any]:
    doc_name = str(event["doc_name"])
    retained = bool(canonical_speeches["doc_name"].eq(doc_name).any())
    source_input = ledger.sha256_text(
        _factual_input(old_speeches, old_paragraphs, doc_name)
    )
    canonical_input = (
        ledger.sha256_text(
            _factual_input(canonical_speeches, canonical_paragraphs, doc_name)
        )
        if retained
        else None
    )
    if retained and source_input == canonical_input:
        status = "speech_retained_exact_input"
        rule = "speech-exact-input-continuity-v1"
        review_item = "ALRV1-D004"
        eligible = True
    elif retained and doc_name == APRIL3:
        if source_input != APRIL3_OLD_INPUT or canonical_input != APRIL3_NEW_INPUT:
            raise ValueError("April 3 factual input hashes drifted")
        status = "speech_retained_input_mismatch_approved"
        rule = "speech-input-mismatch-human-approved-april-3-1968-v1"
        review_item = "ALRV1-D007"
        eligible = True
    elif retained:
        raise ValueError(f"unapproved retained speech input mismatch: {doc_name}")
    else:
        status = "excluded_duplicate_document"
        rule = "speech-excluded-duplicate-document-v1"
        review_item = None
        eligible = False
    canonical_key = {"doc_name": doc_name} if retained else None
    canonical_id = (
        ledger.subject_identity("speech", canonical_key) if canonical_key else None
    )
    canonical_text_sha256 = None
    if retained:
        canonical_row = canonical_speeches[
            canonical_speeches["doc_name"].eq(doc_name)
        ]
        if len(canonical_row) != 1:
            raise ValueError(f"canonical speech lookup failed: {doc_name}")
        canonical_text_sha256 = ledger.sha256_text(
            str(canonical_row.iloc[0]["transcript"])
        )
    return {
        "projection_id": ledger.content_id(
            "proj",
            {
                "label_id": event["label_id"],
                "canonical_subject_id": canonical_id,
                "mapping_rule_id": rule,
            },
        ),
        "label_id": event["label_id"],
        "label_group_id": event["label_group_id"],
        "source_subject_id": event["subject_id"],
        "canonical_subject_id": canonical_id,
        "canonical_subject_key_json": (
            ledger.canonical_json(canonical_key) if canonical_key else None
        ),
        "canonical_doc_name": doc_name if retained else None,
        "canonical_para_idx": None,
        "mapping_status": status,
        "mapping_rule_id": rule,
        "source_text_sha256": event["source_text_sha256"],
        "canonical_text_sha256": canonical_text_sha256,
        "source_input_sha256": source_input,
        "canonical_input_sha256": canonical_input,
        "eligible_for_promotion": eligible,
        "review_item_id": review_item,
        "reason": (
            "Human-approved April 3, 1968 provisional carry-forward."
            if status == "speech_retained_input_mismatch_approved"
            else (
                "Factual input is fingerprint-identical."
                if eligible
                else "Removed duplicate speech remains historical."
            )
        ),
    }


def _invocation_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "projection_id": ledger.content_id(
            "proj",
            {
                "label_id": event["label_id"],
                "canonical_subject_id": event["subject_id"],
                "mapping_rule_id": "invocation-exact-span-continuity-v1",
            },
        ),
        "label_id": event["label_id"],
        "label_group_id": event["label_group_id"],
        "source_subject_id": event["subject_id"],
        "canonical_subject_id": event["subject_id"],
        "canonical_subject_key_json": event["subject_key_json"],
        "canonical_doc_name": event["doc_name"],
        "canonical_para_idx": None,
        "mapping_status": "invocation_span_reusable_exact",
        "mapping_rule_id": "invocation-exact-span-continuity-v1",
        "source_text_sha256": event["source_text_sha256"],
        "canonical_text_sha256": event["source_text_sha256"],
        "source_input_sha256": None,
        "canonical_input_sha256": None,
        "eligible_for_promotion": True,
        "review_item_id": "ALRV1-D004",
        "reason": "Mention span matches the corrected canonical transcript exactly.",
    }


def _metric_rows(agreement: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    source = _source_artifact(AGREEMENT_PATH)
    for raw in agreement.to_dict("records"):
        value = {
            key: (
                {"$number": "NaN"}
                if isinstance(item, float) and math.isnan(item)
                else (item.item() if hasattr(item, "item") else item)
            )
            for key, item in raw.items()
        }
        label_type = str(value["field"])
        row = {
            "record_kind": "metric",
            "evaluation_id": "agreement-v1",
            "label_type": label_type,
            "left_run_id": None,
            "right_run_id": None,
            "stratum_json": ledger.canonical_json(
                {
                    "era_bin": int(value["era_bin"]),
                    "era_label": str(value["era_label"]),
                }
            ),
            "metric_name": str(value["metric"]),
            "metric_value": (
                None
                if value["value"] == {"$number": "NaN"}
                else float(value["value"])
            ),
            "n": int(value["n"]),
            "source_value_json": ledger.canonical_json(value),
            "artifact_role": None,
            "artifact_path": str(AGREEMENT_PATH.relative_to(REPO_ROOT)),
            "artifact_sha256": source["sha256"],
            "source_artifact_id": source["artifact_id"],
            "provenance_json": ledger.canonical_json(
                {
                    "classification": "evaluation_metric_not_label_event",
                    "nonfinite_encoding": (
                        "tagged_nan"
                        if value["value"] == {"$number": "NaN"}
                        else None
                    ),
                }
            ),
        }
        row["metric_id"] = ledger.content_id("met", row)
        rows.append(row)
    sample = _source_artifact(AGREEMENT_SAMPLE_PATH)
    artifact_row = {
        "record_kind": "evaluation_artifact",
        "evaluation_id": "agreement-v1",
        "label_type": None,
        "left_run_id": None,
        "right_run_id": None,
        "stratum_json": ledger.canonical_json({}),
        "metric_name": None,
        "metric_value": None,
        "n": None,
        "source_value_json": ledger.canonical_json(
            {"persisted_sample": AGREEMENT_SAMPLE_PATH.name}
        ),
        "artifact_role": "persisted_agreement_sample",
        "artifact_path": str(AGREEMENT_SAMPLE_PATH.relative_to(REPO_ROOT)),
        "artifact_sha256": sample["sha256"],
        "source_artifact_id": sample["artifact_id"],
        "provenance_json": ledger.canonical_json(
            {"classification": "evaluation_artifact_not_label_event"}
        ),
    }
    artifact_row["metric_id"] = ledger.content_id("met", artifact_row)
    rows.append(artifact_row)
    return rows


def _initialize_review(root: Path) -> None:
    queue_path = REPO_ROOT / "notes" / "annotation-ledger-review-queue-v1.json"
    report_path = REPO_ROOT / "notes" / "annotation-ledger-review-report-v1.md"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    items = []
    resolutions = []
    for item in queue["items"]:
        item_copy = {k: v for k, v in item.items() if k != "resolution"}
        resolution = item.get("resolution")
        items.append(item_copy)
        if resolution is not None:
            resolutions.append(
                {**resolution, "item_id": item["item_id"]}
            )
    review_root = Path(root) / "review"
    ledger.write_jsonl(review_root / "review_items.jsonl", items)
    ledger.write_jsonl(review_root / "review_resolutions.jsonl", resolutions)
    # Readability is derived from the append-only machine records.  Keep the
    # checkpoint report registered as an input artifact rather than copying it
    # as an independently editable ledger view.
    ledger.publish_review_report(root)
    ledger.atomic_write_json(
        review_root / "initial-review-report-source.json",
        _source_artifact(report_path),
    )


def backfill_legacy(
    *,
    root: Path = ledger.LEDGER_ROOT,
    ingested_at: str | None = None,
    sealed_at: str | None = None,
) -> dict[str, Any]:
    """Validate all sources, create 15 legacy runs, and seal each one."""
    evidence = validate_sources()  # all refusal checks occur before writes
    inventory = frozen_inventory()
    if any(ledger.sealed_runs(root)) or (Path(root) / "work").exists():
        raise FileExistsError("annotation ledger already contains run state")
    registry = write_initial_registry(root)
    published_inventory = publish_complete_input_inventory(root=root)
    _initialize_review(root)
    specs = ledger.registered_specs(Path(root) / "specs" / "registry-v1.json")
    ingested_at = ingested_at or ledger.utc_now()
    sealed_at = sealed_at or ledger.utc_now()
    manifests = evidence["manifests"]
    mapping = evidence["mapping"].set_index(["old_doc_name", "old_para_idx"])
    old_paragraphs = pd.read_parquet(OLD_PARAGRAPHS_PATH)
    old_speeches = pd.read_parquet(OLD_SPEECHES_PATH)
    old_text = old_paragraphs.set_index(["doc_name", "para_idx"])["text"]
    old_transcript = old_speeches.set_index("doc_name")["transcript"]
    canonical_transcript = evidence["canonical_speeches"].set_index("doc_name")[
        "transcript"
    ]

    event_counts: Counter[str] = Counter()
    projection_counts: Counter[str] = Counter()
    spec_maps: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    event_handles = {}
    projection_handles = {}
    for run_id in sorted(EXPECTED_MANIFESTS):
        directory = ledger.work_dir(run_id, root)
        directory.mkdir(parents=True)
        (directory / "assignments").mkdir()
        (directory / "responses").mkdir()
        (directory / "registered_artifacts").mkdir()
        event_handles[run_id] = ledger.open_deterministic_gzip_text(
            directory / "label_events.jsonl"
        )
        projection_handles[run_id] = ledger.open_deterministic_gzip_text(
            directory / "canonical_label_projection.jsonl"
        )

    def record(
        event: Mapping[str, Any], projection: Mapping[str, Any]
    ) -> None:
        run_id = str(event["run_id"])
        event_handles[run_id].write(ledger.canonical_json(dict(event)) + "\n")
        projection_handles[run_id].write(
            ledger.canonical_json(dict(projection)) + "\n"
        )
        event_counts[run_id] += 1
        projection_counts[run_id] += 1
        spec_maps[run_id][str(event["label_type"])] = {
            "spec_version": str(event["spec_version"]),
            "spec_sha256": str(event["spec_sha256"]),
        }

    judgment_specs = {
        label_type: specs[(label_type, "paragraph-judgment-v1")]
        for label_type in [
            "topics",
            "party_attack",
            "enemy_naming",
            "zero_sum",
            "proposal_values",
            "entities",
        ]
    }
    for source_path, source_table in [
        (PARAGRAPH_PATH, "paragraph_annotations"),
        (OPUS_PARAGRAPH_PATH, "paragraph_annotations__opus4-8"),
    ]:
        frame = pd.read_parquet(source_path)
        for row in frame.itertuples(index=False):
            key = {"doc_name": str(row.doc_name), "para_idx": int(row.para_idx)}
            source_response_id = ledger.content_id(
                "resp",
                {
                    "run_id": row.run_id,
                    "source_table": source_table,
                    "subject_key": key,
                    "status": "surviving_final_labels_only",
                },
            )
            source_hash = ledger.sha256_text(str(old_text.loc[(row.doc_name, row.para_idx)]))
            values = {
                "topics": list(row.topics),
                "party_attack": bool(row.party_attack),
                "enemy_naming": bool(row.enemy_naming),
                "zero_sum": bool(row.zero_sum),
                "proposal_values": str(row.proposal_values),
            }
            mapping_row = mapping.loc[(row.doc_name, int(row.para_idx))]
            for label_type, value in values.items():
                event = _legacy_event(
                    run_id=str(row.run_id),
                    spec=judgment_specs[label_type],
                    subject_type="paragraph",
                    key=key,
                    value=value,
                    source_response_id=source_response_id,
                    source_path=source_path,
                    source_table=source_table,
                    source_key=key,
                    duplicate_ordinal=0,
                    item_index=0,
                    group_size=1,
                    source_text_sha256=source_hash,
                    ingested_at=ingested_at,
                )
                record(event, _paragraph_projection(event, mapping_row))

    for source_path, source_table in [
        (ENTITY_PATH, "paragraph_entities"),
        (OPUS_ENTITY_PATH, "paragraph_entities__opus4-8"),
    ]:
        frame = pd.read_parquet(source_path).copy()
        frame["_value_json"] = frame.apply(
            lambda row: ledger.canonical_json(
                {
                    "name": str(row["entity"]),
                    "type": str(row["type"]),
                    "stance": str(row["stance"]),
                }
            ),
            axis=1,
        )
        groups = frame.groupby(["run_id", "doc_name", "para_idx"], sort=True)
        for (run_id, doc_name, para_idx), group in groups:
            key = {"doc_name": str(doc_name), "para_idx": int(para_idx)}
            source_response_id = ledger.content_id(
                "resp",
                {
                    "run_id": str(run_id),
                    "source_table": source_table,
                    "subject_key": key,
                    "status": "surviving_final_labels_only",
                },
            )
            ordered = group.sort_values(
                ["_value_json"], kind="mergesort"
            ).reset_index(drop=True)
            duplicate_counts: Counter[str] = Counter()
            mapping_row = mapping.loc[(doc_name, int(para_idx))]
            for item_index, row in ordered.iterrows():
                raw_value = str(row["_value_json"])
                duplicate_ordinal = duplicate_counts[raw_value]
                duplicate_counts[raw_value] += 1
                event = _legacy_event(
                    run_id=str(run_id),
                    spec=judgment_specs["entities"],
                    subject_type="paragraph",
                    key=key,
                    value=json.loads(raw_value),
                    source_response_id=source_response_id,
                    source_path=source_path,
                    source_table=source_table,
                    source_key={
                        **key,
                        "entity": str(row["entity"]),
                        "type": str(row["type"]),
                        "stance": str(row["stance"]),
                    },
                    duplicate_ordinal=duplicate_ordinal,
                    item_index=int(item_index),
                    group_size=len(ordered),
                    source_text_sha256=ledger.sha256_text(
                        str(old_text.loc[(doc_name, int(para_idx))])
                    ),
                    ingested_at=ingested_at,
                )
                record(event, _paragraph_projection(event, mapping_row))

    speech_specs = {
        label_type: specs[(label_type, "speech-factual-v1")]
        for label_type in ["speech_type", "audience", "medium"]
    }
    for source_path, source_table in [
        (SPEECH_PATH, "speech_annotations"),
        (OPUS_SPEECH_PATH, "speech_annotations__opus4-8"),
    ]:
        frame = pd.read_parquet(source_path)
        for row in frame.itertuples(index=False):
            key = {"doc_name": str(row.doc_name)}
            source_response_id = ledger.content_id(
                "resp",
                {
                    "run_id": row.run_id,
                    "source_table": source_table,
                    "subject_key": key,
                    "status": "surviving_final_labels_only",
                },
            )
            for label_type in ["speech_type", "audience", "medium"]:
                event = _legacy_event(
                    run_id=str(row.run_id),
                    spec=speech_specs[label_type],
                    subject_type="speech",
                    key=key,
                    value=str(getattr(row, label_type)),
                    source_response_id=source_response_id,
                    source_path=source_path,
                    source_table=source_table,
                    source_key=key,
                    duplicate_ordinal=0,
                    item_index=0,
                    group_size=1,
                    source_text_sha256=ledger.sha256_text(
                        str(old_transcript.loc[row.doc_name])
                    ),
                    ingested_at=ingested_at,
                )
                record(
                    event,
                    _speech_projection(
                        event,
                        old_speeches=old_speeches,
                        old_paragraphs=old_paragraphs,
                        canonical_speeches=evidence["canonical_speeches"],
                        canonical_paragraphs=evidence["canonical_paragraphs"],
                    ),
                )

    invocation_spec = specs[("invocation_tone", "legacy-2026-07-12")]
    for row in evidence["invocation"].itertuples(index=False):
        key = {
            "char_end": int(row.char_end),
            "char_start": int(row.char_start),
            "doc_name": str(row.doc_name),
        }
        source_response_id = ledger.content_id(
            "resp",
            {
                "run_id": row.run_id,
                "source_table": "invocation_tone",
                "subject_key": key,
                "status": "surviving_final_labels_only",
            },
        )
        event = _legacy_event(
            run_id=str(row.run_id),
            spec=invocation_spec,
            subject_type="invocation_span",
            key=key,
            value=str(row.label),
            source_response_id=source_response_id,
            source_path=INVOCATION_PATH,
            source_table="invocation_tone",
            source_key=key,
            duplicate_ordinal=0,
            item_index=0,
            group_size=1,
            source_text_sha256=ledger.sha256_text(
                str(canonical_transcript.loc[row.doc_name])
            ),
            ingested_at=ingested_at,
        )
        record(event, _invocation_projection(event))

    for handle in [*event_handles.values(), *projection_handles.values()]:
        handle.flush()
        handle.close()

    metrics_by_run = {
        "opus-judgment-r2-20260722": _metric_rows(evidence["agreement"])
    }
    source_files_by_run: dict[str, set[Path]] = defaultdict(set)
    for path, run_column in [
        (PARAGRAPH_PATH, "run_id"),
        (OPUS_PARAGRAPH_PATH, "run_id"),
        (ENTITY_PATH, "run_id"),
        (OPUS_ENTITY_PATH, "run_id"),
        (SPEECH_PATH, "run_id"),
        (OPUS_SPEECH_PATH, "run_id"),
        (INVOCATION_PATH, "run_id"),
    ]:
        for run_id in pd.read_parquet(path, columns=[run_column])[run_column].unique():
            source_files_by_run[str(run_id)].add(path)
    source_files_by_run["taxonomy-v1-20260719"].update(
        {LLM_ROOT / "taxonomy_v1.json", LLM_ROOT / "crosswalk_v1.json"}
    )
    source_files_by_run["opus-judgment-r2-20260722"].update(
        {AGREEMENT_PATH, AGREEMENT_SAMPLE_PATH}
    )

    sealed = []
    for run_id in sorted(EXPECTED_MANIFESTS):
        manifest = manifests[run_id]
        directory = ledger.work_dir(run_id, root)
        effort, effort_provenance, effort_map = _manifest_effort(manifest)
        spec_map = spec_maps[run_id]
        artifacts = [_source_artifact(manifest["_path"])]
        for source_path in sorted(source_files_by_run[run_id]):
            artifact = _source_artifact(source_path)
            role = (
                "run_artifact"
                if source_path.name in {"taxonomy_v1.json", "crosswalk_v1.json"}
                else (
                    "evaluation_artifact"
                    if source_path == AGREEMENT_SAMPLE_PATH
                    else (
                        "evaluation_metrics"
                        if source_path == AGREEMENT_PATH
                        else "source_annotation_table"
                    )
                )
            )
            artifacts.append({**artifact, "role": role})
            if source_path.suffix == ".json" and source_path.name in {
                "taxonomy_v1.json",
                "crosswalk_v1.json",
                "agreement_sample_v1.json",
            }:
                shutil.copyfile(
                    source_path,
                    directory / "registered_artifacts" / source_path.name,
                )
        shutil.copyfile(
            manifest["_path"],
            directory / "registered_artifacts" / manifest["_path"].name,
        )
        limitations = [
            (
                "Backfill preserves surviving final labels but cannot recreate "
                "discarded intermediate model responses."
            )
        ]
        if run_id == "2026-07-12-invocation-tone-fable5":
            limitations.extend(
                [
                    "Original prompt and cost are unknown.",
                    "Stored hash describes the surviving method string, not a prompt.",
                ]
            )
        corpus_fingerprint = manifest.get("corpus_fingerprint")
        run = {
            "run_id": run_id,
            "run_kind": "legacy_backfill",
            "scope": "legacy_surviving",
            "label_spec_map_json": ledger.canonical_json(spec_map),
            "provider_id": (
                "anthropic"
                if str(manifest.get("model", "")).startswith("claude")
                else None
            ),
            "model_id": manifest.get("model"),
            "model_identity_source": "legacy_manifest",
            "annotator_id": None,
            "agent_id": None,
            "reasoning_effort": effort,
            "reasoning_effort_provenance": effort_provenance,
            "reasoning_effort_by_spec_json": effort_map,
            "manifest_date": manifest.get("date"),
            "assigned_at": None,
            "labeled_at": None,
            "ingested_at": ingested_at,
            "timestamp_precision": "date",
            "source_corpus_fingerprint": (
                ledger.sha256_text(ledger.canonical_json(corpus_fingerprint))
                if corpus_fingerprint is not None
                else None
            ),
            "canonical_corpus_fingerprint": evidence["correction_meta"][
                "canonical_corpus_fingerprint"
            ],
            "selection_artifact_id": None,
            "authorization_review_item_id": "ALRV1-D002",
            "human_validation_status": "not_applicable",
            "independent_check_status": "not_applicable",
            "source_manifest_path": str(manifest["_path"].relative_to(REPO_ROOT)),
            "source_manifest_sha256": ledger.sha256_file(manifest["_path"]),
            "artifacts_json": ledger.canonical_json(artifacts),
            "limitations_json": ledger.canonical_json(limitations),
            "completion_policy": "legacy_surviving",
            "expected_event_count": event_counts[run_id],
            "jsonl_storage_encoding": "deterministic-gzip-mtime-zero",
        }
        ledger.atomic_write_json(directory / "run.json", run)
        ledger.snapshot_specs(
            directory,
            run["label_spec_map_json"],
            registry_path=Path(root) / "specs" / "registry-v1.json",
        )
        ledger.write_jsonl(
            directory / "evaluation_metrics.jsonl",
            sorted(metrics_by_run.get(run_id, []), key=lambda row: row["metric_id"]),
        )
        ledger.atomic_write_json(directory / "source-inventory.json", inventory)
        ledger.atomic_write_json(directory / "artifacts.json", artifacts)
        sealed.append(
            ledger.seal_run(
                run_id,
                root=root,
                registry_path=Path(root) / "specs" / "registry-v1.json",
                sealed_at=sealed_at,
            )
        )
    return {
        "status": "backfilled",
        "registry_sha256": registry["registry_sha256"],
        "n_runs": len(sealed),
        "n_events": sum(event_counts.values()),
        "n_projections": sum(projection_counts.values()),
        "n_evaluation_records": sum(len(rows) for rows in metrics_by_run.values()),
        "sealed_runs": sealed,
        "source_inventory": inventory,
        "published_input_inventory": published_inventory,
    }


def continuity_transitions(
    *, root: Path = ledger.LEDGER_ROOT
) -> list[dict[str, Any]]:
    transitions = []
    for run_id in sorted(PRIMARY_RUNS):
        directory = ledger.sealed_artifact_dir(run_id, root)
        group_eligibility: dict[str, bool] = {}
        events = ledger.iter_jsonl(directory / "label_events.jsonl")
        projections = ledger.iter_jsonl(
            directory / "canonical_label_projection.jsonl"
        )
        for event, projection in zip(events, projections, strict=True):
            if event["label_id"] != projection["label_id"]:
                raise ValueError(f"{run_id}: event/projection storage order drift")
            group_id = event["label_group_id"]
            group_eligibility[group_id] = (
                group_eligibility.get(group_id, True)
                and bool(projection["eligible_for_promotion"])
            )
        for group_id, eligible in group_eligibility.items():
            if eligible:
                transitions.append(
                    {
                        "label_group_id": group_id,
                        "promotion_channel": "primary",
                        "replaces_label_group_ids": [],
                        "expected_current_state_sha256": ledger.current_state_hash(
                            []
                        ),
                    }
                )
    return sorted(
        transitions,
        key=lambda row: (
            row["promotion_channel"],
            row["label_group_id"],
        ),
    )


def prepare_correction_overlay_selection(
    *,
    root: Path = ledger.LEDGER_ROOT,
) -> dict[str, Any]:
    """Persist the exact 13 corrected targets with one bounded neighbor per side."""
    evidence = validate_sources()
    required = pd.read_parquet(REANNOTATION_PATH).sort_values(
        ["doc_name", "para_idx"]
    )
    canonical = evidence["canonical_paragraphs"].sort_values(
        ["doc_name", "para_idx"]
    )
    canonical_keys = set(
        zip(canonical["doc_name"], canonical["para_idx"], strict=True)
    )
    required_keys = set(
        zip(required["doc_name"], required["para_idx"], strict=True)
    )
    if len(required_keys) != 13 or not required_keys <= canonical_keys:
        raise ValueError("correction overlay/canonical key-set mismatch")
    required_by_key = required.set_index(["doc_name", "para_idx"])
    rows: list[dict[str, Any]] = []
    for doc_name, speech in canonical.groupby("doc_name", sort=True):
        speech = speech.sort_values("para_idx").reset_index(drop=True)
        for position, paragraph in speech.iterrows():
            key = (str(doc_name), int(paragraph["para_idx"]))
            if key not in required_keys:
                continue
            evidence_row = required_by_key.loc[key]

            def compact(index: int) -> dict[str, Any]:
                row = speech.iloc[index]
                return {
                    "doc_name": str(row["doc_name"]),
                    "para_idx": int(row["para_idx"]),
                    "text": str(row["text"]),
                }

            before = [compact(position - 1)] if position > 0 else []
            after = [compact(position + 1)] if position + 1 < len(speech) else []
            text = str(paragraph["text"])
            canonical_hash = ledger.sha256_text(text)
            expected_hash = "sha256:" + str(evidence_row["canonical_text_sha256"])
            if canonical_hash != expected_hash:
                raise ValueError(f"{key}: canonical overlay text hash drift")
            rows.append(
                {
                    "doc_name": key[0],
                    "para_idx": key[1],
                    "text": text,
                    "word_count": int(paragraph["word_count"]),
                    "correction_id": str(evidence_row["correction_id"]),
                    "old_text_sha256": "sha256:"
                    + str(evidence_row["old_text_sha256"]),
                    "canonical_text_sha256": canonical_hash,
                    "context_before_json": ledger.canonical_json(before),
                    "context_after_json": ledger.canonical_json(after),
                }
            )
    if {
        (row["doc_name"], row["para_idx"]) for row in rows
    } != required_keys:
        raise ValueError("correction overlay selection key-set mismatch")
    selection_id = ledger.content_id(
        "sel",
        {
            "purpose": "trusted-13-paragraph-correction-overlay-v1",
            "rows": rows,
        },
    )
    path = Path(root) / "selections" / f"{selection_id}.json"
    ledger.atomic_write_json(path, rows)
    return {
        "selection_artifact_id": selection_id,
        "path": str(path),
        "sha256": ledger.sha256_file(path),
        "n_subjects": len(rows),
    }


def verify_frozen_inventory(
    expected: Sequence[Mapping[str, Any]],
) -> None:
    actual = frozen_inventory()
    if list(expected) != actual:
        expected_map = {row["path"]: row for row in expected}
        actual_map = {row["path"]: row for row in actual}
        changed = sorted(
            path
            for path in set(expected_map) | set(actual_map)
            if expected_map.get(path) != actual_map.get(path)
        )
        raise ValueError(f"frozen input inventory changed: {changed[:10]}")
