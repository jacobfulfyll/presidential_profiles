"""Governed actual-speaker president/topic network contract.

This module owns the one reusable join between the accepted Story speaker
foundation and the active promoted topic projection.  It publishes descriptive
paragraph-presence measures only; it deliberately contains no layout, chart,
similarity, influence, importance, confidence, or causal fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from . import annotation_ledger, attention, corpus, era_profiles, story_foundation


REPO_ROOT = corpus.DATA_DIR.parent
NETWORK_DIR = corpus.DATA_DIR / "speaker_topic_network"
PUBLIC_DIR_NAME = "speaker-topic-network"
PUBLIC_DIR = REPO_ROOT / "docs" / "data" / PUBLIC_DIR_NAME
TAXONOMY_PATH = attention.TAXONOMY_PATH

CONTRACT_VERSION = "actual-speaker-topic-network-v1"
PARAGRAPH_SCHEMA = "actual-speaker-topic-paragraph-v1"
MEMBERSHIP_SCHEMA = "actual-speaker-topic-membership-v1"
PRESIDENT_NODE_SCHEMA = "actual-speaker-topic-president-node-v1"
TOPIC_NODE_SCHEMA = "actual-speaker-topic-topic-node-v1"
PRESIDENT_SUPPORT_SCHEMA = "actual-speaker-topic-president-support-v1"
TOPIC_SUPPORT_SCHEMA = "actual-speaker-topic-topic-support-v1"
EDGE_SCHEMA = "actual-speaker-topic-edge-v1"
EVIDENCE_SCHEMA = "actual-speaker-topic-evidence-v1"
META_SCHEMA = "actual-speaker-topic-network-meta-v1"
ACCEPTANCE_SCHEMA = "actual-speaker-topic-network-acceptance-v1"
PUBLIC_NETWORK_SCHEMA = "actual-speaker-topic-network-public-v1"
PUBLIC_MANIFEST_SCHEMA = "actual-speaker-topic-network-public-manifest-v1"

PARAGRAPH_KEY = ("doc_name", "para_idx")
APPEARANCE_KEY = ("doc_name", "attributed_speaker_profile_id")
EXPECTED_ACTIVE_GENERATION = (
    "5b3f0e252c63a3b6cf3cba5d5e84e23def38eaa8edf29b89af827c92e5bce307"
)

PRESIDENT_APPEARANCE_FLOOR = 5
EDGE_PARAGRAPH_FLOOR = 5
EDGE_APPEARANCE_FLOOR = 2
DEFAULT_PARAGRAPH_FLOOR = 20
DEFAULT_APPEARANCE_FLOOR = 5
MAX_EVIDENCE_RECEIPTS = 3
MAX_EXCERPT_CHARS = 360

TOPIC_FREE_POLICY = (
    "All eligible actual-president paragraphs, including topic-free paragraphs, "
    "remain in president and scope denominators; topic-free paragraphs create no "
    "membership and no edge."
)
MULTI_LABEL_POLICY = (
    "Count each distinct normalized Level 2 topic once per paragraph and each "
    "distinct derived Level 1 parent once per paragraph. Preserve one full "
    "paragraph-presence membership for every topic; do not divide a paragraph "
    "fractionally. Topic shares are non-additive and may sum above 100%."
)
INTERPRETATION_POLICY = (
    "Descriptive AI-assigned topic presence in speaker-audited Miller Center "
    "paragraphs. It is not historical validation, model confidence, presidential "
    "intent, importance, influence, causality, policy success, or representation "
    "of the public or presidency as a whole."
)
EVIDENCE_POLICY = (
    "Deterministic audit examples, not representative samples or historical proof: "
    "select the first, lower-middle, and last qualifying actual-speaker appearance "
    "chronologically, deduplicate appearances, and take the lowest qualifying "
    "paragraph index within each selected appearance."
)

GOVERNED_FILES = {
    "paragraph_topics": "paragraph_topics_v1.parquet",
    "memberships": "memberships_v1.parquet",
    "president_nodes": "president_nodes_v1.parquet",
    "topic_nodes": "topic_nodes_v1.parquet",
    "president_support": "president_support_v1.parquet",
    "topic_support": "topic_support_v1.parquet",
    "edges": "edges_v1.parquet",
    "edge_evidence": "edge_evidence_v1.parquet",
    "acceptance": "acceptance_report_v1.json",
    "meta": "meta_v1.json",
}
PUBLIC_FILES = {
    "network": "network_v1.json",
    "edges": "edges_v1.csv",
    "edge_evidence": "edge_evidence_v1.csv",
    "memberships": "memberships_v1.parquet",
    "manifest": "manifest_v1.json",
}

EXPECTED_ACCEPTANCE = {
    "retained_paragraphs": 35_394,
    "eligible_paragraphs": 32_531,
    "excluded_paragraphs": 2_863,
    "cross_owner_presidential_paragraphs": 296,
    "actual_speaker_appearances": 1_054,
    "presidents": 45,
    "story_eras": 9,
    "topic_bearing_eligible_paragraphs": 32_394,
    "topic_free_eligible_paragraphs": 137,
    "raw_level2_assignments": 48_211,
    "normalized_level2_memberships": 47_549,
    "removed_duplicate_or_case_variant_assignments": 662,
    "level1_memberships": 43_570,
    "memberships": 91_119,
    "topic_nodes": 67,
    "network_nodes": 112,
    "president_support_rows": 450,
    "topic_support_rows": 670,
    "edges": 4_408,
    "edge_evidence_receipts": 11_576,
    "default_visible_edges": 2_104,
}
EXPECTED_DISTRIBUTIONS = {
    "president_support_status": {"supported": 85, "thin": 10, "no_record": 355},
    "topic_support_status": {"observed": 589, "no_record": 81},
    "edge_support_status": {
        "supported": 3_340,
        "thin_edge": 864,
        "thin_president": 204,
    },
    "edge_scope_and_level": {
        "corpus|level1": 700,
        "corpus|level2": 1_464,
        "story_era|level1": 728,
        "story_era|level2": 1_516,
    },
}

PROTECTED_OUTPUT_ROOTS = (
    corpus.DATA_DIR / "llm_annotations",
    corpus.DATA_DIR / "speaker_attribution",
    corpus.DATA_DIR / "annotation_ledger",
    corpus.DATA_DIR / "speaker_views",
    corpus.DATA_DIR / "reference_entities",
)


class SpeakerTopicNetworkError(RuntimeError):
    """The governed actual-speaker topic-network contract failed closed."""


@dataclass(frozen=True)
class ActiveTopicProjection:
    records: pd.DataFrame
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SpeakerTopicNetworkBundle:
    paragraph_topics: pd.DataFrame
    memberships: pd.DataFrame
    president_nodes: pd.DataFrame
    topic_nodes: pd.DataFrame
    president_support: pd.DataFrame
    topic_support: pd.DataFrame
    edges: pd.DataFrame
    edge_evidence: pd.DataFrame
    meta: dict[str, Any]
    acceptance: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha256_bytes(
        _canonical_json({key: item for key, item in value.items() if key != field}).encode()
    )


def _content_id(prefix: str, identity: Mapping[str, Any]) -> str:
    return prefix + "_" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _source_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpeakerTopicNetworkError(f"could not read governed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SpeakerTopicNetworkError(f"expected a JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SpeakerTopicNetworkError(f"{label} is missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    _require_columns(frame, keys, label)
    duplicated = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicated.empty:
        raise SpeakerTopicNetworkError(
            f"{label} has duplicate keys on {list(keys)}: "
            f"{duplicated.head(5).to_dict('records')}"
        )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _normal_text(value: Any) -> str:
    return " ".join(str(value).split())


def _excerpt(value: Any, limit: int = MAX_EXCERPT_CHARS) -> str:
    text = _normal_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _iso_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _topic_id(level: str, label: str) -> str:
    return _content_id("topic", {"topic_level": level, "canonical_label": label})


def _appearance_id(doc_name: str, president_profile_id: str) -> str:
    return _content_id(
        "appearance",
        {"doc_name": doc_name, "attributed_speaker_profile_id": president_profile_id},
    )


def _scope_specs() -> list[dict[str, str | None]]:
    return [
        {"scope_type": "corpus", "scope_id": "all-corpus", "scope_label": "All eligible corpus"}
    ] + [
        {"scope_type": "story_era", "scope_id": spec.key, "scope_label": spec.label}
        for spec in era_profiles.ERA_PROFILE_SPECS
    ]


def _in_scope(frame: pd.DataFrame, scope: Mapping[str, Any]) -> pd.Series:
    if scope["scope_type"] == "corpus":
        return pd.Series(True, index=frame.index)
    return frame["story_era_id"].eq(scope["scope_id"])


def _assert_safe_output_path(path: Path) -> None:
    resolved = Path(path).resolve()
    for protected in PROTECTED_OUTPUT_ROOTS:
        protected_resolved = protected.resolve()
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise SpeakerTopicNetworkError(
                f"refusing output path inside protected governed inputs: {resolved}"
            )


def resolve_active_topics(
    *, materialized_root: Path = annotation_ledger.MATERIALIZED_ROOT
) -> ActiveTopicProjection:
    """Resolve and validate the active primary promoted topic projection."""
    root = Path(materialized_root)
    pointer = root / "current"
    try:
        pointer_bytes = pointer.read_bytes()
        generation_id = pointer_bytes.decode("ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SpeakerTopicNetworkError("could not read active promotion pointer") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", generation_id):
        raise SpeakerTopicNetworkError("active promotion pointer is malformed")
    generation_dir = root / "generations" / generation_id
    if not generation_dir.is_dir():
        raise SpeakerTopicNetworkError("active promotion pointer targets a missing generation")
    try:
        observed_root = annotation_ledger.artifact_root(
            annotation_ledger._member_inventory(generation_dir)
        ).removeprefix("sha256:")
    except (OSError, ValueError) as exc:
        raise SpeakerTopicNetworkError("could not verify active generation identity") from exc
    if observed_root != generation_id:
        raise SpeakerTopicNetworkError(
            f"active generation content hash drift: pointer={generation_id}, observed={observed_root}"
        )

    manifest_path = generation_dir / "generation.json"
    labels_path = generation_dir / "current_labels.parquet"
    manifest = _read_json(manifest_path)
    try:
        current = pd.read_parquet(labels_path)
    except Exception as exc:
        raise SpeakerTopicNetworkError("could not read active promoted labels") from exc
    if manifest.get("tables", {}).get("current_labels") != len(current):
        raise SpeakerTopicNetworkError("active generation manifest/current-label count drift")
    required = {
        "promotion_channel", "canonical_subject_id", "label_type", "promotion_id",
        "promotion_batch_id", "label_group_id", "label_id", "run_id", "spec_version",
        "spec_sha256", "canonical_subject_key_json", "canonical_doc_name",
        "canonical_para_idx", "event_role", "value_kind", "raw_value_json",
        "promotion_rule_id", "adjudication_id", "review_resolution_id",
    }
    _require_columns(current, required, "active promoted labels")
    selected = current.loc[
        current["promotion_channel"].eq("primary")
        & current["label_type"].eq("topics")
        & current["event_role"].eq("value")
        & current["value_kind"].eq("list")
    ].copy()
    if len(selected) != 35_394:
        raise SpeakerTopicNetworkError(
            f"active primary topic projection has {len(selected):,} rows, expected 35,394"
        )
    selected["doc_name"] = selected["canonical_doc_name"].astype(str)
    if selected["canonical_para_idx"].isna().any():
        raise SpeakerTopicNetworkError("active primary topic projection has null paragraph keys")
    selected["para_idx"] = selected["canonical_para_idx"].astype(int)
    _require_unique(selected, PARAGRAPH_KEY, "active primary topic projection")

    def parse_topics(value: Any) -> list[str]:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise SpeakerTopicNetworkError("promoted topic value is not valid JSON") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise SpeakerTopicNetworkError("promoted topic value is not a list of strings")
        return parsed

    selected["raw_topics"] = selected["raw_value_json"].map(parse_topics)
    selected = selected.sort_values(list(PARAGRAPH_KEY), kind="mergesort").reset_index(drop=True)
    hash_fields = sorted(required)
    promoted_hash = _sha256_bytes(
        "".join(
            _canonical_json(
                {
                    field: None if pd.isna(value) else value
                    for field, value in zip(hash_fields, row, strict=True)
                }
            )
            + "\n"
            for row in selected[hash_fields].itertuples(index=False, name=None)
        ).encode()
    )
    provenance = {
        "active_generation_id": generation_id,
        "pointer_path": _source_path(pointer),
        "pointer_sha256": _sha256_bytes(pointer_bytes),
        "generation_manifest_path": _source_path(manifest_path),
        "generation_manifest_sha256": _sha256_file(manifest_path),
        "generation_artifact_root": "sha256:" + observed_root,
        "current_labels_path": _source_path(labels_path),
        "current_labels_sha256": _sha256_file(labels_path),
        "promoted_topic_records_sha256": promoted_hash,
        "promotion_contract": {
            "promotion_channel": "primary",
            "label_type": "topics",
            "event_role": "value",
            "value_kind": "list",
        },
    }
    return ActiveTopicProjection(selected, provenance)


def exact_topic_join(
    foundation_paragraphs: pd.DataFrame, topic_records: pd.DataFrame
) -> pd.DataFrame:
    """Perform a key-complete one-to-one join independent of source row order."""
    _require_unique(foundation_paragraphs, PARAGRAPH_KEY, "foundation paragraphs")
    _require_unique(topic_records, PARAGRAPH_KEY, "promoted topics")
    left_keys = set(map(tuple, foundation_paragraphs[list(PARAGRAPH_KEY)].to_numpy()))
    right_keys = set(map(tuple, topic_records[list(PARAGRAPH_KEY)].to_numpy()))
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)[:5]
        extra = sorted(right_keys - left_keys)[:5]
        raise SpeakerTopicNetworkError(
            f"foundation/topic key-set mismatch: missing={missing}, extra={extra}"
        )
    projection = topic_records.rename(
        columns={
            "run_id": "topic_annotation_run_id",
            "spec_version": "topic_annotation_spec_version",
        }
    )
    joined = foundation_paragraphs.merge(
        projection,
        on=list(PARAGRAPH_KEY),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(foundation_paragraphs):
        raise SpeakerTopicNetworkError("foundation/topic one-to-one join changed row count")
    return joined.sort_values(list(PARAGRAPH_KEY), kind="mergesort").reset_index(drop=True)


def _validate_story_eras(frame: pd.DataFrame) -> None:
    expected = {spec.key: spec.label for spec in era_profiles.ERA_PROFILE_SPECS}
    observed = frame[["story_era_key", "story_era"]].drop_duplicates()
    mapping = dict(observed.itertuples(index=False, name=None))
    if mapping != expected:
        raise SpeakerTopicNetworkError(
            f"foundation Story-era identity drift: expected={expected}, observed={mapping}"
        )


def _president_nodes(paragraphs: pd.DataFrame) -> pd.DataFrame:
    identities = paragraphs[
        ["president_profile_id", "actual_speaker"]
    ].drop_duplicates()
    if identities["president_profile_id"].duplicated().any() or identities["actual_speaker"].duplicated().any():
        raise SpeakerTopicNetworkError("actual-president identity is not one-to-one")
    by_name = identities.set_index("actual_speaker")["president_profile_id"].to_dict()
    if set(by_name) != set(corpus.PARTY):
        raise SpeakerTopicNetworkError("actual-president identity does not cover the canonical 45 presidents")
    rows = []
    for order, (name, party) in enumerate(corpus.PARTY.items(), start=1):
        rows.append(
            {
                "schema_version": PRESIDENT_NODE_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "president_profile_id": str(by_name[name]),
                "president_name": name,
                "party": party,
                "display_order": order,
                "node_type": "president",
            }
        )
    return pd.DataFrame(rows)


def _topic_nodes(taxonomy: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    level1_ids = {entry["name"]: _topic_id("level1", entry["name"]) for entry in taxonomy["level1"]}
    for order, entry in enumerate(taxonomy["level1"], start=1):
        rows.append(
            {
                "schema_version": TOPIC_NODE_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "topic_id": level1_ids[entry["name"]],
                "topic_level": "level1",
                "topic_label": entry["name"],
                "topic_definition": entry["definition"],
                "topic_kind": entry.get("kind"),
                "parent_topic_id": None,
                "parent_topic_label": None,
                "display_order": order,
                "node_type": "topic",
            }
        )
    for order, entry in enumerate(taxonomy["level2"], start=1):
        rows.append(
            {
                "schema_version": TOPIC_NODE_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "topic_id": _topic_id("level2", entry["name"]),
                "topic_level": "level2",
                "topic_label": entry["name"],
                "topic_definition": entry["definition"],
                "topic_kind": None,
                "parent_topic_id": level1_ids[entry["level1"]],
                "parent_topic_label": entry["level1"],
                "display_order": order,
                "node_type": "topic",
            }
        )
    return pd.DataFrame(rows)


def _normalize_joined(
    joined: pd.DataFrame, taxonomy: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    required = {
        "doc_name", "para_idx", "text", "analysis_eligible", "attributed_speaker",
        "attributed_speaker_profile_id", "document_owner", "document_owner_profile_id",
        "cross_owner_paragraph", "story_era_key", "story_era", "date", "title",
        "source_url", "raw_topics", "promotion_id", "label_id",
        "topic_annotation_run_id", "topic_annotation_spec_version", "spec_sha256",
        "promotion_rule_id",
    }
    _require_columns(joined, required, "joined foundation/topics")
    _validate_story_eras(joined)
    label_map = attention.canonical_label_map(dict(taxonomy))
    parents = attention.level1_parents(dict(taxonomy))
    try:
        joined = joined.copy()
        joined["level2_topics"] = joined["raw_topics"].map(
            lambda raw: attention.normalize_topics(raw, label_map)
        )
    except ValueError as exc:
        raise SpeakerTopicNetworkError(str(exc)) from exc
    eligible = joined.loc[joined["analysis_eligible"].astype(bool)].copy()
    missing_identity = eligible[
        ["attributed_speaker", "attributed_speaker_profile_id", "story_era_key"]
    ].isna().any(axis=1)
    if missing_identity.any():
        raise SpeakerTopicNetworkError("eligible paragraph lacks actual-speaker, appearance, or era identity")
    eligible["level1_topics"] = eligible["level2_topics"].map(
        lambda topics: list(dict.fromkeys(parents[topic] for topic in topics))
    )
    eligible["raw_topic_count"] = eligible["raw_topics"].map(len)
    eligible["level2_topic_count"] = eligible["level2_topics"].map(len)
    eligible["level1_topic_count"] = eligible["level1_topics"].map(len)
    eligible["topic_free"] = eligible["level2_topic_count"].eq(0)
    eligible["appearance_id"] = [
        _appearance_id(str(doc), str(profile))
        for doc, profile in eligible[
            ["doc_name", "attributed_speaker_profile_id"]
        ].itertuples(index=False)
    ]

    paragraph_rows = []
    membership_rows = []
    ordered_eligible = eligible.sort_values(list(PARAGRAPH_KEY), kind="mergesort")
    for row in ordered_eligible.itertuples(index=False):
        owner_profile_id = (
            None
            if pd.isna(row.document_owner_profile_id)
            else str(row.document_owner_profile_id)
        )
        paragraph_rows.append(
            {
                "schema_version": PARAGRAPH_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "doc_name": str(row.doc_name),
                "para_idx": int(row.para_idx),
                "president_profile_id": str(row.attributed_speaker_profile_id),
                "actual_speaker": str(row.attributed_speaker),
                "source_document_owner_profile_id": owner_profile_id,
                "source_document_owner": str(row.document_owner),
                "cross_owner": bool(row.cross_owner_paragraph),
                "appearance_id": str(row.appearance_id),
                "story_era_id": str(row.story_era_key),
                "story_era_label": str(row.story_era),
                "speech_date": _iso_date(row.date),
                "speech_title": str(row.title),
                "source_url": str(row.source_url),
                "topic_free": bool(row.topic_free),
                "raw_topic_count": int(row.raw_topic_count),
                "normalized_level2_topic_count": int(row.level2_topic_count),
                "derived_level1_topic_count": int(row.level1_topic_count),
                "normalized_level2_topics_json": _canonical_json(row.level2_topics),
                "derived_level1_topics_json": _canonical_json(row.level1_topics),
                "paragraph_text_sha256": _sha256_bytes(_normal_text(row.text).encode()),
                "annotation_promotion_id": str(row.promotion_id),
                "annotation_label_id": str(row.label_id),
                "annotation_run_id": str(row.topic_annotation_run_id),
                "annotation_spec_version": str(row.topic_annotation_spec_version),
                "annotation_spec_sha256": str(row.spec_sha256),
                "annotation_promotion_rule_id": str(row.promotion_rule_id),
                "topic_free_denominator_policy": TOPIC_FREE_POLICY,
                "multi_label_counting_policy": MULTI_LABEL_POLICY,
            }
        )
        for level, topics in (("level1", row.level1_topics), ("level2", row.level2_topics)):
            for topic in topics:
                membership_rows.append(
                    {
                        "schema_version": MEMBERSHIP_SCHEMA,
                        "contract_version": CONTRACT_VERSION,
                        "doc_name": str(row.doc_name),
                        "para_idx": int(row.para_idx),
                        "president_profile_id": str(row.attributed_speaker_profile_id),
                        "actual_speaker": str(row.attributed_speaker),
                        "source_document_owner_profile_id": owner_profile_id,
                        "source_document_owner": str(row.document_owner),
                        "cross_owner": bool(row.cross_owner_paragraph),
                        "appearance_id": str(row.appearance_id),
                        "story_era_id": str(row.story_era_key),
                        "story_era_label": str(row.story_era),
                        "speech_date": _iso_date(row.date),
                        "speech_title": str(row.title),
                        "source_url": str(row.source_url),
                        "topic_id": _topic_id(level, topic),
                        "topic_level": level,
                        "topic_label": topic,
                        "evidence_excerpt": _excerpt(row.text),
                        "annotation_promotion_id": str(row.promotion_id),
                    }
                )
    paragraph_topics = (
        pd.DataFrame(paragraph_rows)
        .sort_values(list(PARAGRAPH_KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    memberships = pd.DataFrame(membership_rows).sort_values(
        ["topic_level", "topic_id", "president_profile_id", "speech_date", "doc_name", "para_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    _require_unique(memberships, ["doc_name", "para_idx", "topic_id"], "topic memberships")
    counts = {
        "raw_level2_assignments": int(eligible["raw_topic_count"].sum()),
        "normalized_level2_memberships": int(eligible["level2_topic_count"].sum()),
        "level1_memberships": int(eligible["level1_topic_count"].sum()),
        "topic_bearing_eligible_paragraphs": int((~eligible["topic_free"]).sum()),
        "topic_free_eligible_paragraphs": int(eligible["topic_free"].sum()),
    }
    counts["removed_duplicate_or_case_variant_assignments"] = (
        counts["raw_level2_assignments"] - counts["normalized_level2_memberships"]
    )
    return paragraph_topics, memberships, counts


def president_support_status(appearance_count: int) -> str:
    if appearance_count == 0:
        return "no_record"
    if appearance_count < PRESIDENT_APPEARANCE_FLOOR:
        return "thin"
    return "supported"


def edge_support_status(
    president_status: str, topic_paragraph_count: int, topic_appearance_count: int
) -> str:
    if president_status == "thin":
        return "thin_president"
    if president_status != "supported":
        raise SpeakerTopicNetworkError("no-record presidents cannot have observed edges")
    if (
        topic_paragraph_count >= EDGE_PARAGRAPH_FLOOR
        and topic_appearance_count >= EDGE_APPEARANCE_FLOOR
    ):
        return "supported"
    return "thin_edge"


def is_default_visible(
    president_status: str, topic_paragraph_count: int, topic_appearance_count: int
) -> bool:
    return bool(
        president_status == "supported"
        and topic_paragraph_count >= DEFAULT_PARAGRAPH_FLOOR
        and topic_appearance_count >= DEFAULT_APPEARANCE_FLOOR
    )


def _support_tables(
    paragraphs: pd.DataFrame,
    memberships: pd.DataFrame,
    president_nodes: pd.DataFrame,
    topic_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    president_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    president_names = president_nodes.set_index("president_profile_id")["president_name"].to_dict()
    topic_meta = topic_nodes.set_index("topic_id")[["topic_level", "topic_label"]].to_dict("index")

    for scope in _scope_specs():
        scoped_paragraphs = paragraphs.loc[_in_scope(paragraphs, scope)].copy()
        scoped_memberships = memberships.loc[_in_scope(memberships, scope)].copy()
        scope_eligible_n = len(scoped_paragraphs)
        president_denominators: dict[str, tuple[int, int, str]] = {}
        for node in president_nodes.sort_values("display_order").itertuples(index=False):
            rows = scoped_paragraphs[
                scoped_paragraphs["president_profile_id"].eq(node.president_profile_id)
            ]
            paragraph_n = len(rows)
            appearance_n = int(rows["appearance_id"].nunique())
            status = president_support_status(appearance_n)
            president_denominators[str(node.president_profile_id)] = (
                paragraph_n,
                appearance_n,
                status,
            )
            president_rows.append(
                {
                    "schema_version": PRESIDENT_SUPPORT_SCHEMA,
                    "contract_version": CONTRACT_VERSION,
                    "scope_type": scope["scope_type"],
                    "scope_id": scope["scope_id"],
                    "scope_label": scope["scope_label"],
                    "story_era_id": (
                        scope["scope_id"] if scope["scope_type"] == "story_era" else None
                    ),
                    "president_profile_id": str(node.president_profile_id),
                    "president_name": str(node.president_name),
                    "eligible_president_paragraph_count": paragraph_n,
                    "eligible_president_appearance_count": appearance_n,
                    "president_support_status": status,
                    "topic_free_denominator_policy": TOPIC_FREE_POLICY,
                }
            )

        topic_denominators: dict[str, int] = {}
        for node in topic_nodes.sort_values(["topic_level", "display_order"]).itertuples(index=False):
            rows = scoped_memberships[scoped_memberships["topic_id"].eq(node.topic_id)]
            topic_paragraph_n = len(rows)
            topic_denominators[str(node.topic_id)] = topic_paragraph_n
            topic_rows.append(
                {
                    "schema_version": TOPIC_SUPPORT_SCHEMA,
                    "contract_version": CONTRACT_VERSION,
                    "scope_type": scope["scope_type"],
                    "scope_id": scope["scope_id"],
                    "scope_label": scope["scope_label"],
                    "story_era_id": (
                        scope["scope_id"] if scope["scope_type"] == "story_era" else None
                    ),
                    "topic_id": str(node.topic_id),
                    "topic_level": str(node.topic_level),
                    "topic_label": str(node.topic_label),
                    "scope_topic_paragraph_count": topic_paragraph_n,
                    "scope_eligible_paragraph_count": scope_eligible_n,
                    "topic_scope_share": _safe_ratio(topic_paragraph_n, scope_eligible_n),
                    "topic_support_status": "observed" if topic_paragraph_n else "no_record",
                    "topic_free_denominator_policy": TOPIC_FREE_POLICY,
                    "multi_label_counting_policy": MULTI_LABEL_POLICY,
                }
            )

        if scoped_memberships.empty:
            continue
        grouped = scoped_memberships.groupby(
            ["president_profile_id", "topic_id", "topic_level", "topic_label"],
            sort=True,
            observed=True,
        )
        for (president_id, topic_id, topic_level, topic_label), rows in grouped:
            topic_paragraph_n = len(rows)
            topic_appearance_n = int(rows["appearance_id"].nunique())
            president_paragraph_n, president_appearance_n, president_status = (
                president_denominators[str(president_id)]
            )
            scope_topic_n = topic_denominators[str(topic_id)]
            support = edge_support_status(
                president_status, topic_paragraph_n, topic_appearance_n
            )
            edge_id = _content_id(
                "edge",
                {
                    "contract_version": CONTRACT_VERSION,
                    "scope_type": scope["scope_type"],
                    "scope_id": scope["scope_id"],
                    "president_profile_id": str(president_id),
                    "topic_id": str(topic_id),
                },
            )
            edge_rows.append(
                {
                    "schema_version": EDGE_SCHEMA,
                    "contract_version": CONTRACT_VERSION,
                    "edge_id": edge_id,
                    "edge_type": "president_topic",
                    "scope_type": scope["scope_type"],
                    "scope_id": scope["scope_id"],
                    "scope_label": scope["scope_label"],
                    "story_era_id": (
                        scope["scope_id"] if scope["scope_type"] == "story_era" else None
                    ),
                    "president_profile_id": str(president_id),
                    "president_name": president_names[str(president_id)],
                    "topic_id": str(topic_id),
                    "topic_level": str(topic_level),
                    "topic_label": str(topic_label),
                    "topic_paragraph_count": topic_paragraph_n,
                    "topic_appearance_count": topic_appearance_n,
                    "eligible_president_paragraph_count": president_paragraph_n,
                    "eligible_president_appearance_count": president_appearance_n,
                    "scope_topic_paragraph_count": scope_topic_n,
                    "scope_eligible_paragraph_count": scope_eligible_n,
                    "speaker_paragraph_share": _safe_ratio(
                        topic_paragraph_n, president_paragraph_n
                    ),
                    "topic_contribution_share": _safe_ratio(topic_paragraph_n, scope_topic_n),
                    "topic_scope_share": _safe_ratio(scope_topic_n, scope_eligible_n),
                    "president_support_status": president_status,
                    "edge_support_status": support,
                    "default_visible": is_default_visible(
                        president_status, topic_paragraph_n, topic_appearance_n
                    ),
                    "topic_free_denominator_policy": TOPIC_FREE_POLICY,
                    "multi_label_counting_policy": MULTI_LABEL_POLICY,
                    "evidence_receipt_count": 0,
                    "interpretation_policy": INTERPRETATION_POLICY,
                }
            )

    president_support = pd.DataFrame(president_rows).sort_values(
        ["scope_type", "scope_id", "president_profile_id"], kind="mergesort"
    ).reset_index(drop=True)
    topic_support = pd.DataFrame(topic_rows).sort_values(
        ["scope_type", "scope_id", "topic_level", "topic_id"], kind="mergesort"
    ).reset_index(drop=True)
    edges = pd.DataFrame(edge_rows).sort_values(
        ["scope_type", "scope_id", "topic_level", "topic_id", "president_profile_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    _require_unique(
        president_support, ["scope_type", "scope_id", "president_profile_id"], "president support"
    )
    _require_unique(topic_support, ["scope_type", "scope_id", "topic_id"], "topic support")
    _require_unique(edges, ["scope_type", "scope_id", "president_profile_id", "topic_id"], "edges")
    if set(edges["edge_type"]) != {"president_topic"}:
        raise SpeakerTopicNetworkError("network contains a non-bipartite edge type")
    if any(column == "weight" or column.endswith("_confidence") for column in edges.columns):
        raise SpeakerTopicNetworkError("network contains a prohibited ambiguous measure")
    return president_support, topic_support, edges


def evidence_selection_positions(appearance_count: int) -> list[tuple[str, int]]:
    """Return deterministic role/index pairs, deduplicating thin appearance sets."""
    if appearance_count <= 0:
        return []
    candidates = [
        ("first", 0),
        ("middle", (appearance_count - 1) // 2),
        ("last", appearance_count - 1),
    ]
    seen: set[int] = set()
    output = []
    for role, position in candidates:
        if position not in seen:
            output.append((role, position))
            seen.add(position)
    return output


def _evidence_receipts(
    edges: pd.DataFrame, memberships: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = {
        key: rows.copy()
        for key, rows in memberships.groupby(
            ["president_profile_id", "topic_id"], sort=False, observed=True
        )
    }
    receipt_rows: list[dict[str, Any]] = []
    receipt_counts: dict[str, int] = {}
    for edge in edges.itertuples(index=False):
        rows = grouped[(edge.president_profile_id, edge.topic_id)]
        if edge.scope_type == "story_era":
            rows = rows[rows["story_era_id"].eq(edge.scope_id)]
        appearances = (
            rows.sort_values(
                ["speech_date", "doc_name", "appearance_id", "para_idx"], kind="mergesort"
            )
            .drop_duplicates("appearance_id", keep="first")
            .reset_index(drop=True)
        )
        chosen = evidence_selection_positions(len(appearances))
        receipt_counts[edge.edge_id] = len(chosen)
        for role, position in chosen:
            selected = appearances.iloc[position]
            receipt_id = _content_id(
                "receipt",
                {"edge_id": edge.edge_id, "selection_role": role, "appearance_id": selected.appearance_id},
            )
            receipt_rows.append(
                {
                    "schema_version": EVIDENCE_SCHEMA,
                    "contract_version": CONTRACT_VERSION,
                    "receipt_id": receipt_id,
                    "edge_id": edge.edge_id,
                    "scope_type": edge.scope_type,
                    "scope_id": edge.scope_id,
                    "scope_label": edge.scope_label,
                    "president_profile_id": edge.president_profile_id,
                    "president_name": edge.president_name,
                    "topic_id": edge.topic_id,
                    "topic_level": edge.topic_level,
                    "topic_label": edge.topic_label,
                    "doc_name": selected.doc_name,
                    "para_idx": int(selected.para_idx),
                    "appearance_id": selected.appearance_id,
                    "actual_speaker_profile_id": selected.president_profile_id,
                    "actual_speaker": selected.actual_speaker,
                    "source_document_owner_profile_id": selected.source_document_owner_profile_id,
                    "source_document_owner": selected.source_document_owner,
                    "cross_owner": bool(selected.cross_owner),
                    "story_era_id": selected.story_era_id,
                    "story_era_label": selected.story_era_label,
                    "speech_date": selected.speech_date,
                    "speech_title": selected.speech_title,
                    "source_url": selected.source_url,
                    "evidence_excerpt": selected.evidence_excerpt,
                    "selection_role": role,
                    "selection_position": position,
                    "evidence_policy": EVIDENCE_POLICY,
                }
            )
    evidence = pd.DataFrame(receipt_rows).sort_values(
        ["scope_type", "scope_id", "topic_level", "topic_id", "president_profile_id", "selection_position"],
        kind="mergesort",
    ).reset_index(drop=True)
    _require_unique(evidence, ["edge_id", "appearance_id"], "edge evidence")
    edges = edges.copy()
    edges["evidence_receipt_count"] = edges["edge_id"].map(receipt_counts).astype(int)
    expected = evidence.groupby("edge_id").size().to_dict()
    if expected != receipt_counts:
        raise SpeakerTopicNetworkError("edge evidence receipt counts do not reconcile")
    return edges, evidence


def _metric_definitions() -> dict[str, Any]:
    common = {
        "population": "Eligible actual-president paragraphs in the accepted speaker-audited foundation.",
        "unit": (
            "Distinct paragraph presence; appearances are distinct "
            "(doc_name, actual-speaker profile ID) receipts."
        ),
        "topic_free_treatment": TOPIC_FREE_POLICY,
        "multi_label_treatment": MULTI_LABEL_POLICY,
        "scope_behavior": (
            "Computed for the complete eligible corpus and separately within "
            "each accepted Story era carried by the foundation."
        ),
        "interpretation": INTERPRETATION_POLICY,
        "thin_record_rules": {
            "president_supported_minimum_appearances": PRESIDENT_APPEARANCE_FLOOR,
            "edge_supported_minimum_paragraphs": EDGE_PARAGRAPH_FLOOR,
            "edge_supported_minimum_topic_appearances": EDGE_APPEARANCE_FLOOR,
        },
    }
    return {
        "actual_speaker_topic_presence": {
            **common,
            "measure": "speaker_paragraph_share",
            "definition": (
                "Topic-bearing eligible paragraphs attributed to the president "
                "divided by all eligible paragraphs attributed to that president "
                "in the scope."
            ),
            "formula": "topic_paragraph_count / eligible_president_paragraph_count",
        },
        "actual_speaker_topic_contribution": {
            **common,
            "measure": "topic_contribution_share",
            "definition": (
                "Topic-bearing eligible paragraphs attributed to the president "
                "divided by all eligible paragraphs carrying that topic across "
                "presidents in the scope."
            ),
            "formula": "topic_paragraph_count / scope_topic_paragraph_count",
        },
        "topic_scope_share": {
            **common,
            "measure": "topic_scope_share",
            "definition": (
                "All eligible paragraphs carrying the topic divided by all "
                "eligible actual-president paragraphs in the scope."
            ),
            "formula": "scope_topic_paragraph_count / scope_eligible_paragraph_count",
        },
    }


def _acceptance_payload(
    *,
    foundation: story_foundation.StoryFoundationBundle,
    paragraph_topics: pd.DataFrame,
    memberships: pd.DataFrame,
    president_nodes: pd.DataFrame,
    topic_nodes: pd.DataFrame,
    president_support: pd.DataFrame,
    topic_support: pd.DataFrame,
    edges: pd.DataFrame,
    evidence: pd.DataFrame,
    normalization_counts: Mapping[str, int],
) -> dict[str, Any]:
    counts = {
        "retained_paragraphs": len(foundation.paragraph_view),
        "eligible_paragraphs": len(paragraph_topics),
        "excluded_paragraphs": int(
            (~foundation.paragraph_view["analysis_eligible"].astype(bool)).sum()
        ),
        "cross_owner_presidential_paragraphs": int(
            foundation.paragraph_view["cross_owner_paragraph"].sum()
        ),
        "actual_speaker_appearances": len(foundation.appearances),
        "presidents": len(president_nodes),
        "story_eras": len(era_profiles.ERA_PROFILE_SPECS),
        **dict(normalization_counts),
        "memberships": len(memberships),
        "topic_nodes": len(topic_nodes),
        "network_nodes": len(president_nodes) + len(topic_nodes),
        "president_support_rows": len(president_support),
        "topic_support_rows": len(topic_support),
        "edges": len(edges),
        "edge_evidence_receipts": len(evidence),
        "default_visible_edges": int(edges["default_visible"].sum()),
    }
    distributions = {
        "president_support_status": {
            str(key): int(value)
            for key, value in president_support["president_support_status"].value_counts().items()
        },
        "topic_support_status": {
            str(key): int(value)
            for key, value in topic_support["topic_support_status"].value_counts().items()
        },
        "edge_support_status": {
            str(key): int(value)
            for key, value in edges["edge_support_status"].value_counts().items()
        },
        "edge_scope_and_level": {
            f"{scope}|{level}": int(len(rows))
            for (scope, level), rows in edges.groupby(["scope_type", "topic_level"], sort=True)
        },
    }
    checks = {
        "exact_key_set_before_eligibility": True,
        "one_to_one_topic_join": True,
        "actual_speaker_identity_reused": True,
        "appearance_identity_reused": True,
        "story_era_identity_reused": True,
        "topic_taxonomy_canonical": True,
        "topic_free_paragraphs_retained_in_denominators": True,
        "multi_label_memberships_non_additive": True,
        "edges_are_undirected_bipartite_president_topic_only": True,
        "evidence_receipts_reconcile": True,
    }
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "counts": counts,
        "distributions": distributions,
        "checks": checks,
        "expected_counts": EXPECTED_ACCEPTANCE,
        "expected_distributions": EXPECTED_DISTRIBUTIONS,
        "all_pinned_acceptance_values_match": (
            counts == EXPECTED_ACCEPTANCE
            and distributions == EXPECTED_DISTRIBUTIONS
        ),
    }


def _validate_pinned_acceptance(acceptance: Mapping[str, Any]) -> None:
    if acceptance.get("counts") != EXPECTED_ACCEPTANCE:
        raise SpeakerTopicNetworkError(
            f"pinned network acceptance count drift: {acceptance.get('counts')}"
        )
    if acceptance.get("distributions") != EXPECTED_DISTRIBUTIONS:
        raise SpeakerTopicNetworkError(
            f"pinned network acceptance distribution drift: {acceptance.get('distributions')}"
        )
    if not acceptance.get("all_pinned_acceptance_values_match"):
        raise SpeakerTopicNetworkError("pinned network acceptance did not pass")


def _foundation_provenance(
    foundation: story_foundation.StoryFoundationBundle,
) -> dict[str, Any]:
    return {
        "contract_identity": story_foundation.PUBLIC_SCHEMA,
        "governed_schema_version": foundation.provenance["schema_version"],
        "speaker_view_schema": foundation.provenance["speaker_view_schema"],
        "speaker_metadata_sha256": foundation.provenance["speaker_metadata_sha256"],
        "reference_metadata_sha256": foundation.provenance["reference_metadata_sha256"],
        "canonical_corpus_fingerprint": foundation.provenance["canonical_corpus_fingerprint"],
        "attribution_run_id": foundation.provenance["attribution_run_id"],
        "annotation_generation": foundation.provenance["annotation_generation"],
        "source_paths": dict(foundation.provenance["source_paths"]),
        "source_hashes": dict(foundation.provenance["source_hashes"]),
        "speaker_meta_file_sha256": _sha256_file(story_foundation.SPEAKER_META_PATH),
        "reference_meta_file_sha256": _sha256_file(story_foundation.REFERENCE_META_PATH),
    }


def build_network_bundle(
    *,
    foundation: story_foundation.StoryFoundationBundle | None = None,
    materialized_root: Path = annotation_ledger.MATERIALIZED_ROOT,
    taxonomy_path: Path = TAXONOMY_PATH,
    enforce_pinned_acceptance: bool = True,
) -> SpeakerTopicNetworkBundle:
    """Build the complete contract in memory without writing accepted outputs."""
    foundation = foundation or story_foundation.load_story_foundation()
    story_foundation.check_story_contract(foundation)
    projection = resolve_active_topics(materialized_root=materialized_root)
    if enforce_pinned_acceptance and (
        projection.provenance["active_generation_id"] != EXPECTED_ACTIVE_GENERATION
    ):
        raise SpeakerTopicNetworkError(
            "active topic generation differs from the audited Plan 3 baseline: "
            f"{projection.provenance['active_generation_id']}"
        )
    foundation_generation = str(foundation.provenance["annotation_generation"])
    if foundation_generation != projection.provenance["active_generation_id"]:
        raise SpeakerTopicNetworkError(
            "accepted foundation and active topic promotion pointer disagree: "
            f"foundation={foundation_generation}, active={projection.provenance['active_generation_id']}"
        )
    joined = exact_topic_join(foundation.paragraph_view, projection.records)
    try:
        taxonomy_bytes = Path(taxonomy_path).read_bytes()
        taxonomy = json.loads(taxonomy_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise SpeakerTopicNetworkError("could not read canonical topic taxonomy") from exc
    if not isinstance(taxonomy, dict):
        raise SpeakerTopicNetworkError("canonical topic taxonomy is not an object")
    # Calling all three repository helpers is intentional: they are this
    # contract's only normalization/taxonomy implementation.
    try:
        attention.canonical_label_map(taxonomy)
        parents = attention.level1_parents(taxonomy)
    except (KeyError, TypeError, ValueError) as exc:
        raise SpeakerTopicNetworkError("canonical topic taxonomy failed validation") from exc
    if (
        len(taxonomy.get("level1", [])) != 17
        or len(taxonomy.get("level2", [])) != 50
        or len(parents) != 50
        or set(parents.values()) != {row["name"] for row in taxonomy["level1"]}
    ):
        raise SpeakerTopicNetworkError("canonical taxonomy is not the governed 17/50 hierarchy")

    paragraph_topics, memberships, normalization_counts = _normalize_joined(joined, taxonomy)
    president_nodes = _president_nodes(paragraph_topics)
    topic_nodes = _topic_nodes(taxonomy)
    president_support, topic_support, edges = _support_tables(
        paragraph_topics, memberships, president_nodes, topic_nodes
    )
    edges, evidence = _evidence_receipts(edges, memberships)
    acceptance = _acceptance_payload(
        foundation=foundation,
        paragraph_topics=paragraph_topics,
        memberships=memberships,
        president_nodes=president_nodes,
        topic_nodes=topic_nodes,
        president_support=president_support,
        topic_support=topic_support,
        edges=edges,
        evidence=evidence,
        normalization_counts=normalization_counts,
    )
    if enforce_pinned_acceptance:
        _validate_pinned_acceptance(acceptance)

    foundation_provenance = _foundation_provenance(foundation)
    configuration = {
        "story_eras": [
            {
                "scope_id": spec.key,
                "scope_label": spec.label,
                "declared_start_year": spec.start_year,
                "declared_end_year": spec.end_year,
            }
            for spec in era_profiles.ERA_PROFILE_SPECS
        ],
        "support_thresholds": {
            "president_supported_minimum_appearances": PRESIDENT_APPEARANCE_FLOOR,
            "edge_supported_minimum_paragraphs": EDGE_PARAGRAPH_FLOOR,
            "edge_supported_minimum_topic_appearances": EDGE_APPEARANCE_FLOOR,
            "default_visible_minimum_paragraphs": DEFAULT_PARAGRAPH_FLOOR,
            "default_visible_minimum_topic_appearances": DEFAULT_APPEARANCE_FLOOR,
        },
        "topic_free_denominator_policy": TOPIC_FREE_POLICY,
        "multi_label_counting_policy": MULTI_LABEL_POLICY,
        "evidence_selection_policy": EVIDENCE_POLICY,
        "interpretation_policy": INTERPRETATION_POLICY,
    }
    meta = {
        "schema_version": META_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "schema_versions": {
            "paragraph_topics": PARAGRAPH_SCHEMA,
            "memberships": MEMBERSHIP_SCHEMA,
            "president_nodes": PRESIDENT_NODE_SCHEMA,
            "topic_nodes": TOPIC_NODE_SCHEMA,
            "president_support": PRESIDENT_SUPPORT_SCHEMA,
            "topic_support": TOPIC_SUPPORT_SCHEMA,
            "edges": EDGE_SCHEMA,
            "edge_evidence": EVIDENCE_SCHEMA,
            "acceptance": ACCEPTANCE_SCHEMA,
        },
        "input_provenance": {
            "story_foundation": foundation_provenance,
            "active_topic_promotion": projection.provenance,
            "topic_taxonomy": {
                "path": _source_path(Path(taxonomy_path)),
                "sha256": _sha256_bytes(taxonomy_bytes),
                "normalization_helpers": [
                    "attention.canonical_label_map",
                    "attention.normalize_topics",
                    "attention.level1_parents",
                ],
            },
        },
        "source_configuration": configuration,
        "source_configuration_sha256": _sha256_bytes(_canonical_json(configuration).encode()),
        "metrics": _metric_definitions(),
        "counts": dict(acceptance["counts"]),
    }
    return SpeakerTopicNetworkBundle(
        paragraph_topics,
        memberships,
        president_nodes,
        topic_nodes,
        president_support,
        topic_support,
        edges,
        evidence,
        meta,
        acceptance,
    )


def _frame_map(bundle: SpeakerTopicNetworkBundle) -> dict[str, pd.DataFrame]:
    return {
        "paragraph_topics": bundle.paragraph_topics,
        "memberships": bundle.memberships,
        "president_nodes": bundle.president_nodes,
        "topic_nodes": bundle.topic_nodes,
        "president_support": bundle.president_support,
        "topic_support": bundle.topic_support,
        "edges": bundle.edges,
        "edge_evidence": bundle.edge_evidence,
    }


def validate_bundle(
    bundle: SpeakerTopicNetworkBundle, *, enforce_pinned_acceptance: bool = True
) -> None:
    """Validate schema, keys, measures, receipts, and baseline acceptance."""
    schemas = {
        "paragraph_topics": PARAGRAPH_SCHEMA,
        "memberships": MEMBERSHIP_SCHEMA,
        "president_nodes": PRESIDENT_NODE_SCHEMA,
        "topic_nodes": TOPIC_NODE_SCHEMA,
        "president_support": PRESIDENT_SUPPORT_SCHEMA,
        "topic_support": TOPIC_SUPPORT_SCHEMA,
        "edges": EDGE_SCHEMA,
        "edge_evidence": EVIDENCE_SCHEMA,
    }
    for name, frame in _frame_map(bundle).items():
        if frame.empty:
            raise SpeakerTopicNetworkError(f"{name} is unexpectedly empty")
        if set(frame["schema_version"].astype(str)) != {schemas[name]}:
            raise SpeakerTopicNetworkError(f"{name} schema-version drift")
        if set(frame["contract_version"].astype(str)) != {CONTRACT_VERSION}:
            raise SpeakerTopicNetworkError(f"{name} contract-version drift")
    _require_unique(bundle.paragraph_topics, PARAGRAPH_KEY, "paragraph topics")
    _require_unique(bundle.memberships, ["doc_name", "para_idx", "topic_id"], "memberships")
    _require_unique(bundle.president_nodes, ["president_profile_id"], "president nodes")
    _require_unique(bundle.topic_nodes, ["topic_id"], "topic nodes")
    _require_unique(
        bundle.president_support,
        ["scope_type", "scope_id", "president_profile_id"],
        "president support",
    )
    _require_unique(bundle.topic_support, ["scope_type", "scope_id", "topic_id"], "topic support")
    _require_unique(
        bundle.edges,
        ["scope_type", "scope_id", "president_profile_id", "topic_id"],
        "edges",
    )
    _require_unique(bundle.edge_evidence, ["edge_id", "appearance_id"], "edge evidence")
    if set(bundle.topic_nodes["topic_level"].value_counts().to_dict().items()) != {
        ("level1", 17), ("level2", 50)
    }:
        raise SpeakerTopicNetworkError("topic-node level composition drift")
    if set(bundle.edges["edge_type"]) != {"president_topic"}:
        raise SpeakerTopicNetworkError("edges are not exclusively president/topic")
    edge_ids = set(bundle.edges["edge_id"])
    if set(bundle.edge_evidence["edge_id"]) - edge_ids:
        raise SpeakerTopicNetworkError("evidence references an unknown edge")
    receipt_counts = bundle.edge_evidence.groupby("edge_id").size().to_dict()
    if any(
        int(row.evidence_receipt_count) != receipt_counts.get(row.edge_id, 0)
        for row in bundle.edges.itertuples(index=False)
    ):
        raise SpeakerTopicNetworkError("edge/evidence receipt count drift")
    for row in bundle.edges.itertuples(index=False):
        expected_ratios = (
            _safe_ratio(row.topic_paragraph_count, row.eligible_president_paragraph_count),
            _safe_ratio(row.topic_paragraph_count, row.scope_topic_paragraph_count),
            _safe_ratio(row.scope_topic_paragraph_count, row.scope_eligible_paragraph_count),
        )
        observed_ratios = (
            row.speaker_paragraph_share,
            row.topic_contribution_share,
            row.topic_scope_share,
        )
        for observed, expected in zip(observed_ratios, expected_ratios, strict=True):
            if expected is None:
                if observed is not None and not pd.isna(observed):
                    raise SpeakerTopicNetworkError("zero-denominator ratio is not null")
            elif not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-15):
                raise SpeakerTopicNetworkError("edge share does not reconcile to explicit counts")
        if row.edge_support_status != edge_support_status(
            row.president_support_status,
            row.topic_paragraph_count,
            row.topic_appearance_count,
        ):
            raise SpeakerTopicNetworkError("edge support status drift")
        if bool(row.default_visible) != is_default_visible(
            row.president_support_status,
            row.topic_paragraph_count,
            row.topic_appearance_count,
        ):
            raise SpeakerTopicNetworkError("edge default-visible status drift")
    if bundle.meta.get("schema_version") != META_SCHEMA:
        raise SpeakerTopicNetworkError("network metadata schema drift")
    if bundle.acceptance.get("schema_version") != ACCEPTANCE_SCHEMA:
        raise SpeakerTopicNetworkError("network acceptance schema drift")
    if enforce_pinned_acceptance:
        _validate_pinned_acceptance(bundle.acceptance)


def _write_candidate_governed(
    bundle: SpeakerTopicNetworkBundle, output_dir: Path
) -> dict[str, Any]:
    """Write a complete candidate directory; metadata is always written last."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_bundle(bundle)
    for name, frame in _frame_map(bundle).items():
        frame.to_parquet(output / GOVERNED_FILES[name], index=False)
    (output / GOVERNED_FILES["acceptance"]).write_bytes(_json_bytes(bundle.acceptance))
    artifacts = {}
    for name in [*list(_frame_map(bundle)), "acceptance"]:
        path = output / GOVERNED_FILES[name]
        artifacts[path.name] = {
            "schema_version": (
                bundle.acceptance["schema_version"]
                if name == "acceptance"
                else str(_frame_map(bundle)[name]["schema_version"].iloc[0])
            ),
            "rows": 1 if name == "acceptance" else len(_frame_map(bundle)[name]),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    meta = {**bundle.meta, "artifacts": artifacts}
    meta["metadata_sha256"] = _self_hash(meta, "metadata_sha256")
    (output / GOVERNED_FILES["meta"]).write_bytes(_json_bytes(meta))
    return meta


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, destination)


def _publish_candidate(candidate: Path, output: Path, order: Sequence[str]) -> None:
    """Replace a validated file set with rollback; the receipt file is last."""
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="speaker-topic-network-rollback-") as raw:
        backup = Path(raw)
        existing = {filename for filename in order if (output / filename).is_file()}
        for filename in existing:
            shutil.copyfile(output / filename, backup / filename)
        try:
            for filename in order:
                _atomic_replace(candidate / filename, output / filename)
        except Exception:
            for filename in order:
                destination = output / filename
                if filename in existing:
                    _atomic_replace(backup / filename, destination)
                elif destination.is_file():
                    destination.unlink()
            raise


def write_governed_bundle(
    bundle: SpeakerTopicNetworkBundle, output_dir: Path = NETWORK_DIR
) -> list[Path]:
    """Validate in isolation, then replace per-file with metadata last."""
    output = Path(output_dir)
    _assert_safe_output_path(output)
    with tempfile.TemporaryDirectory(prefix="speaker-topic-network-governed-") as raw:
        candidate = Path(raw)
        _write_candidate_governed(bundle, candidate)
        load_network_bundle(
            network_dir=candidate,
            foundation=None,
            validate_current_inputs=False,
        )
        order = [
            GOVERNED_FILES[name]
            for name in [*list(_frame_map(bundle)), "acceptance", "meta"]
        ]
        _publish_candidate(candidate, output, order)
    return [output / filename for filename in order]


def load_network_bundle(
    *,
    network_dir: Path = NETWORK_DIR,
    foundation: story_foundation.StoryFoundationBundle | None = None,
    validate_current_inputs: bool = True,
) -> SpeakerTopicNetworkBundle:
    """Load one accepted bundle and fail closed on artifacts or stale inputs."""
    root = Path(network_dir)
    meta = _read_json(root / GOVERNED_FILES["meta"])
    if meta.get("schema_version") != META_SCHEMA or meta.get("contract_version") != CONTRACT_VERSION:
        raise SpeakerTopicNetworkError("unsupported network metadata contract")
    if meta.get("metadata_sha256") != _self_hash(meta, "metadata_sha256"):
        raise SpeakerTopicNetworkError("network metadata self-hash drift")
    artifacts = meta.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SpeakerTopicNetworkError("network metadata lacks an artifact inventory")
    expected_names = {GOVERNED_FILES[name] for name in [*list(GOVERNED_FILES)[:-1]]}
    # The metadata file self-authenticates rather than recursively inventorying itself.
    expected_names = set(GOVERNED_FILES.values()) - {GOVERNED_FILES["meta"]}
    if set(artifacts) != expected_names:
        raise SpeakerTopicNetworkError("network artifact inventory membership drift")
    for filename, receipt in artifacts.items():
        path = root / filename
        if (
            not path.is_file()
            or path.stat().st_size != receipt.get("bytes")
            or _sha256_file(path) != receipt.get("sha256")
        ):
            raise SpeakerTopicNetworkError(f"network artifact hash drift: {filename}")

    frames = {
        name: pd.read_parquet(root / filename)
        for name, filename in GOVERNED_FILES.items()
        if name not in {"meta", "acceptance"}
    }
    acceptance = _read_json(root / GOVERNED_FILES["acceptance"])
    for name, frame in frames.items():
        receipt = artifacts[GOVERNED_FILES[name]]
        if len(frame) != receipt.get("rows"):
            raise SpeakerTopicNetworkError(f"network row-count drift: {name}")
    bundle = SpeakerTopicNetworkBundle(
        frames["paragraph_topics"],
        frames["memberships"],
        frames["president_nodes"],
        frames["topic_nodes"],
        frames["president_support"],
        frames["topic_support"],
        frames["edges"],
        frames["edge_evidence"],
        meta,
        acceptance,
    )
    validate_bundle(bundle)

    if validate_current_inputs:
        foundation = foundation or story_foundation.load_story_foundation()
        story_foundation.check_story_contract(foundation)
        projection = resolve_active_topics()
        current_foundation = _foundation_provenance(foundation)
        recorded_inputs = meta.get("input_provenance", {})
        if recorded_inputs.get("story_foundation") != current_foundation:
            raise SpeakerTopicNetworkError("network is stale against the accepted Story foundation")
        if recorded_inputs.get("active_topic_promotion") != projection.provenance:
            raise SpeakerTopicNetworkError("network is stale against the active topic promotion")
        taxonomy_receipt = recorded_inputs.get("topic_taxonomy", {})
        if taxonomy_receipt.get("sha256") != _sha256_file(TAXONOMY_PATH):
            raise SpeakerTopicNetworkError("network is stale against the canonical topic taxonomy")
        if projection.provenance["active_generation_id"] != EXPECTED_ACTIVE_GENERATION:
            raise SpeakerTopicNetworkError("active topic generation is outside the audited baseline")
    return bundle


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def _public_network_payload(bundle: SpeakerTopicNetworkBundle) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_NETWORK_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "question": (
            "Within the speaker-audited Miller Center corpus, which canonical "
            "AI-assigned topics appear in paragraphs actually attributed to each "
            "president, and what share of that president's eligible paragraph record "
            "do those paragraphs represent in the corpus or an existing Story era?"
        ),
        "network_type": "undirected_bipartite_president_topic",
        "scopes": _scope_specs(),
        "president_nodes": _records(bundle.president_nodes),
        "topic_nodes": _records(bundle.topic_nodes),
        "president_support": _records(bundle.president_support),
        "topic_support": _records(bundle.topic_support),
        "edges": _records(bundle.edges),
        "metrics": bundle.meta["metrics"],
        "policies": bundle.meta["source_configuration"],
        "input_provenance": bundle.meta["input_provenance"],
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_candidate_public(
    bundle: SpeakerTopicNetworkBundle, output_dir: Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_bundle(bundle)
    (output / PUBLIC_FILES["network"]).write_bytes(_json_bytes(_public_network_payload(bundle)))
    (output / PUBLIC_FILES["edges"]).write_bytes(_csv_bytes(bundle.edges))
    (output / PUBLIC_FILES["edge_evidence"]).write_bytes(_csv_bytes(bundle.edge_evidence))
    bundle.memberships.to_parquet(output / PUBLIC_FILES["memberships"], index=False)
    files = {}
    public_schemas = {
        PUBLIC_FILES["network"]: PUBLIC_NETWORK_SCHEMA,
        PUBLIC_FILES["edges"]: EDGE_SCHEMA,
        PUBLIC_FILES["edge_evidence"]: EVIDENCE_SCHEMA,
        PUBLIC_FILES["memberships"]: MEMBERSHIP_SCHEMA,
    }
    public_counts = {
        PUBLIC_FILES["network"]: 1,
        PUBLIC_FILES["edges"]: len(bundle.edges),
        PUBLIC_FILES["edge_evidence"]: len(bundle.edge_evidence),
        PUBLIC_FILES["memberships"]: len(bundle.memberships),
    }
    for filename in (
        PUBLIC_FILES["network"],
        PUBLIC_FILES["edges"],
        PUBLIC_FILES["edge_evidence"],
        PUBLIC_FILES["memberships"],
    ):
        path = output / filename
        files[filename] = {
            "schema_version": public_schemas[filename],
            "rows": public_counts[filename],
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": PUBLIC_MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "schema_versions": {
            **bundle.meta["schema_versions"],
            "public_network": PUBLIC_NETWORK_SCHEMA,
            "public_manifest": PUBLIC_MANIFEST_SCHEMA,
        },
        "counts": bundle.acceptance["counts"],
        "files": files,
        "input_provenance": bundle.meta["input_provenance"],
        "metrics": bundle.meta["metrics"],
        "topic_free_denominator_policy": TOPIC_FREE_POLICY,
        "multi_label_counting_policy": MULTI_LABEL_POLICY,
        "support_thresholds": bundle.meta["source_configuration"]["support_thresholds"],
        "evidence_selection_policy": EVIDENCE_POLICY,
        "interpretation_policy": INTERPRETATION_POLICY,
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    (output / PUBLIC_FILES["manifest"]).write_bytes(_json_bytes(manifest))
    return manifest


def validate_publication(
    bundle: SpeakerTopicNetworkBundle, site_dir: Path = REPO_ROOT / "docs"
) -> dict[str, Any]:
    output = Path(site_dir) / "data" / PUBLIC_DIR_NAME
    manifest = _read_json(output / PUBLIC_FILES["manifest"])
    if manifest.get("schema_version") != PUBLIC_MANIFEST_SCHEMA:
        raise SpeakerTopicNetworkError("public network manifest schema drift")
    if manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256"):
        raise SpeakerTopicNetworkError("public network manifest self-hash drift")
    expected_files = set(PUBLIC_FILES.values()) - {PUBLIC_FILES["manifest"]}
    if set(manifest.get("files", {})) != expected_files:
        raise SpeakerTopicNetworkError("public network file inventory drift")
    for filename, receipt in manifest["files"].items():
        path = output / filename
        if (
            not path.is_file()
            or path.stat().st_size != receipt.get("bytes")
            or _sha256_file(path) != receipt.get("sha256")
        ):
            raise SpeakerTopicNetworkError(f"public network artifact drift: {filename}")
    network = _read_json(output / PUBLIC_FILES["network"])
    if network.get("schema_version") != PUBLIC_NETWORK_SCHEMA:
        raise SpeakerTopicNetworkError("public network JSON schema drift")
    if len(network.get("edges", [])) != len(bundle.edges):
        raise SpeakerTopicNetworkError("public network JSON edge-count drift")
    edges = pd.read_csv(output / PUBLIC_FILES["edges"])
    evidence = pd.read_csv(output / PUBLIC_FILES["edge_evidence"])
    memberships = pd.read_parquet(output / PUBLIC_FILES["memberships"])
    if (len(edges), len(evidence), len(memberships)) != (
        len(bundle.edges), len(bundle.edge_evidence), len(bundle.memberships)
    ):
        raise SpeakerTopicNetworkError("public network download row-count drift")
    if set(edges["edge_id"].astype(str)) != set(bundle.edges["edge_id"].astype(str)):
        raise SpeakerTopicNetworkError("public edge identifiers drift")
    if set(evidence["receipt_id"].astype(str)) != set(bundle.edge_evidence["receipt_id"].astype(str)):
        raise SpeakerTopicNetworkError("public evidence identifiers drift")
    _require_unique(memberships, ["doc_name", "para_idx", "topic_id"], "public memberships")
    if manifest.get("input_provenance") != bundle.meta.get("input_provenance"):
        raise SpeakerTopicNetworkError("public network input provenance drift")
    return manifest


def write_public_downloads(
    bundle: SpeakerTopicNetworkBundle, site_dir: Path = REPO_ROOT / "docs"
) -> list[Path]:
    """Publish the four downloads atomically and their manifest last."""
    output = Path(site_dir) / "data" / PUBLIC_DIR_NAME
    _assert_safe_output_path(output)
    with tempfile.TemporaryDirectory(prefix="speaker-topic-network-public-") as raw:
        candidate = Path(raw)
        _write_candidate_public(bundle, candidate)
        with tempfile.TemporaryDirectory(prefix="speaker-topic-network-public-check-") as check_raw:
            candidate_site = Path(check_raw)
            candidate_output = candidate_site / "data" / PUBLIC_DIR_NAME
            shutil.copytree(candidate, candidate_output)
            validate_publication(bundle, candidate_site)
        order = [
            PUBLIC_FILES["network"],
            PUBLIC_FILES["edges"],
            PUBLIC_FILES["edge_evidence"],
            PUBLIC_FILES["memberships"],
            PUBLIC_FILES["manifest"],
        ]
        _publish_candidate(candidate, output, order)
    return [output / filename for filename in order]


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def verify_reproducible() -> dict[str, Any]:
    """Build twice independently and require byte equality for every output."""
    with tempfile.TemporaryDirectory(prefix="speaker-topic-network-repro-a-") as a_raw, tempfile.TemporaryDirectory(
        prefix="speaker-topic-network-repro-b-"
    ) as b_raw:
        roots = [Path(a_raw), Path(b_raw)]
        for root in roots:
            bundle = build_network_bundle()
            _write_candidate_governed(bundle, root / "governed")
            _write_candidate_public(bundle, root / "public")
        left = _directory_bytes(roots[0])
        right = _directory_bytes(roots[1])
        if left.keys() != right.keys():
            raise SpeakerTopicNetworkError("reproducibility output membership drift")
        drift = [name for name in left if left[name] != right[name]]
        if drift:
            raise SpeakerTopicNetworkError(f"reproducibility byte drift: {drift}")
        return {
            "status": "reproducible",
            "files": len(left),
            "hashes": {name: _sha256_bytes(payload) for name, payload in left.items()},
        }


def rebuild_network(output_dir: Path = NETWORK_DIR) -> SpeakerTopicNetworkBundle:
    """Approved explicit rebuild; all validation completes before accepted writes."""
    _assert_safe_output_path(output_dir)
    bundle = build_network_bundle()
    validate_bundle(bundle)
    write_governed_bundle(bundle, output_dir)
    return load_network_bundle(network_dir=output_dir)


def check_network(
    *,
    network_dir: Path = NETWORK_DIR,
    site_dir: Path | None = None,
) -> SpeakerTopicNetworkBundle:
    bundle = load_network_bundle(network_dir=network_dir)
    if site_dir is not None:
        validate_publication(bundle, site_dir)
    return bundle


def _summary(bundle: SpeakerTopicNetworkBundle) -> str:
    counts = bundle.acceptance["counts"]
    return (
        f"{CONTRACT_VERSION}: {counts['eligible_paragraphs']:,} paragraphs, "
        f"{counts['memberships']:,} memberships, {counts['network_nodes']:,} nodes, "
        f"{counts['edges']:,} edges, {counts['edge_evidence_receipts']:,} receipts"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the governed actual-speaker topic network"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--rebuild", action="store_true", help="explicitly rebuild governed artifacts")
    action.add_argument("--check", action="store_true", help="validate accepted artifacts and current inputs")
    action.add_argument(
        "--verify-reproducible",
        action="store_true",
        help="compare two independent temporary governed and public builds byte-for-byte",
    )
    parser.add_argument("--output-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=None,
        help="with --check, also validate published downloads under this site directory",
    )
    args = parser.parse_args(argv)
    if args.verify_reproducible:
        result = verify_reproducible()
        print(f"{result['status']}: {result['files']} files are byte-identical")
        return 0
    if args.rebuild:
        bundle = rebuild_network(args.output_dir)
    else:
        bundle = check_network(network_dir=args.output_dir, site_dir=args.site_dir)
    print(_summary(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
