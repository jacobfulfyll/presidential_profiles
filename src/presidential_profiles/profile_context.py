"""Deterministic public context projection for president profile pages.

This contract is deliberately separate from ``president-profile-v3``.  It
projects the accepted actual-speaker topic and invocation bundles and adds an
auditable, parallel representation of the legacy document-owner issue cards.
It never mutates a profile view model or changes a v3 serializer value.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import gzip
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import (
    actual_speaker_invocation_network,
    corpus,
    issues,
    profiles,
    speaker_topic_network,
    story_foundation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR_NAME = "profile-context"
PUBLIC_DIR = REPO_ROOT / "docs" / "data" / PUBLIC_DIR_NAME

CONTRACT_VERSION = "president-profile-context-v1"
INDEX_SCHEMA = "president-profile-context-index-v1"
PRESIDENT_SCHEMA = "president-profile-context-president-v1"
MANIFEST_SCHEMA = "president-profile-context-manifest-v1"
LEGACY_EVIDENCE_SCHEMA = "president-profile-legacy-issue-evidence-v1"
INDEX_FILE = "index_v1.json"
MANIFEST_FILE = "manifest_v1.json"
INVOCATION_EDGE_CSV = "actual_speaker_invocation_edges_v1.csv"
INVOCATION_EVIDENCE_CSV = "actual_speaker_invocation_evidence_v1.csv"

SCOPE_TYPE = "corpus"
SCOPE_ID = "all-corpus"

INDEX_RAW_MAX = 750_000
INDEX_GZIP_MAX = 90_000
SHARD_RAW_MAX = 500_000
SHARD_GZIP_MAX = 75_000
ALL_SHARDS_RAW_MAX = 8_000_000
ALL_SHARDS_GZIP_MAX = 1_500_000
INVOCATION_EDGE_CSV_RAW_MAX = 200_000
INVOCATION_EDGE_CSV_GZIP_MAX = 30_000
INVOCATION_EVIDENCE_CSV_RAW_MAX = 1_250_000
INVOCATION_EVIDENCE_CSV_GZIP_MAX = 225_000

EXPECTED_COUNTS = {
    "presidents": 45,
    "level1_topics": 17,
    "supported_presidents": 42,
    "thin_presidents": 3,
    "observed_topic_edges": 700,
    "supported_topic_edges": 595,
    "default_visible_topic_edges": 427,
    "topic_receipts": 1_945,
    "invocation_edges": 303,
    "invocation_rows": 1_243,
    "invocation_shard_edge_references": 606,
    "invocation_shard_evidence_references": 2_486,
    "legacy_cards": 163,
    "legacy_primary_receipts": 161,
    "legacy_rate_only_cards": 2,
    "public_files": 49,
}

EXPECTED_LEGACY_SOURCE_HASHES = {
    "paragraphs": "sha256:7f647adbde21bedb8ffa9ef65b2e5a14e87ebddf8918febe511992700264e4d3",
    "paragraph_issues": "sha256:de110914fecca39d648a4967a539095827bf4923c16afac7d578353cbe5ed297",
    "speeches": "sha256:065c24997b03483a1fc7bc689144699a505b627ab95ab1eb79d02c54f86348b5",
}

PRESIDENT_FIELDS = (
    "president_profile_id", "president_name", "president_display_name",
    "slug", "party", "display_order", "node_type",
)
TOPIC_FIELDS = (
    "topic_id", "topic_level", "topic_label", "topic_definition",
    "topic_kind", "parent_topic_id", "parent_topic_label", "display_order",
    "node_type",
)
TOPIC_EDGE_FIELDS = (
    "edge_id", "edge_type", "scope_type", "scope_id", "topic_level",
    "president_profile_id", "topic_id", "topic_paragraph_count",
    "topic_appearance_count", "eligible_president_paragraph_count",
    "eligible_president_appearance_count", "scope_topic_paragraph_count",
    "scope_eligible_paragraph_count", "speaker_paragraph_share",
    "topic_contribution_share", "topic_scope_share",
    "president_support_status", "edge_support_status", "default_visible",
    "evidence_receipt_count",
)
TOPIC_RECEIPT_FIELDS = (
    "receipt_id", "edge_id", "doc_name", "para_idx", "appearance_id",
    "actual_speaker_profile_id", "actual_speaker",
    "source_document_owner_profile_id", "source_document_owner",
    "cross_owner", "story_era_id", "story_era_label", "speech_date",
    "speech_title", "source_url", "evidence_excerpt", "selection_role",
    "selection_position",
)
PRESIDENT_SUPPORT_FIELDS = (
    "president_profile_id", "eligible_president_paragraph_count",
    "eligible_president_appearance_count", "president_support_status",
)
TOPIC_SUPPORT_FIELDS = (
    "topic_id", "scope_topic_paragraph_count", "scope_eligible_paragraph_count",
    "topic_scope_share", "topic_support_status",
)

INDEX_KEYS = (
    "schema_version", "contract_version", "source_identity", "scope", "metrics",
    "policies", "presidents", "level1_topics", "president_support",
    "topic_support", "topic_edges", "invocation_edges", "president_shards",
    "downloads", "counts", "index_sha256",
)
SHARD_KEYS = (
    "schema_version", "contract_version", "source_identity", "president",
    "president_support", "topic_edges", "topic_receipts", "topic_selections",
    "invocations", "legacy_issue_evidence", "counts", "shard_sha256",
)
MANIFEST_KEYS = (
    "schema_version", "contract_version", "source_identity", "files", "counts",
    "budgets", "manifest_sha256",
)
SHARD_RECEIPT_FIELDS = ("president_profile_id", "filename", "sha256")
FILE_RECEIPT_FIELDS = ("schema_version", "rows", "bytes", "gzip_bytes", "sha256")
INVOCATION_SHARD_KEYS = (
    "default_direction", "outgoing_edge_ids", "incoming_edge_ids",
    "receipts_by_edge", "selected_receipt_positions_by_edge",
)
TOPIC_SELECTIONS_KEYS = ("default", "expanded", "observed_thin")
TOPIC_SELECTION_KEYS = (
    "node_ids", "edge_ids", "mobile_topic_peer_node_ids",
)
THIN_TOPIC_SELECTION_KEYS = (
    "comparison_allowed", "edge_style", *TOPIC_SELECTION_KEYS,
)
SHARD_COUNT_KEYS = (
    "focal_topic_edges", "focal_topic_receipts", "outgoing_invocation_edges",
    "incoming_invocation_edges", "invocation_receipt_references",
    "legacy_issue_cards",
)
LEGACY_EVIDENCE_KEYS = (
    "schema_version", "population", "cards", "voice", "n_speeches",
    "n_paragraphs", "thin_record", "thin_record_warning",
)
LEGACY_CARD_KEYS = (
    "issue", "legacy_v3", "claim", "exact_evidence", "why_shown", "receipts",
    "no_qualifying_excerpt_selected", "no_qualifying_excerpt_copy", "method",
    "limitation",
)
LEGACY_RECEIPT_FIELDS = (
    "doc_name", "para_idx", "title", "speech_date", "year", "source_url",
    "source_document_owner", "source_document_owner_profile_id", "actual_speaker",
    "actual_speaker_profile_id", "cross_owner", "speaker_eligibility_state",
    "speaker_exclusion_reason", "excerpt", "selection_role",
)
LEGACY_CLAIM_KEYS = ("type", "text")
LEGACY_EXACT_EVIDENCE_KEYS = (
    "issue_paragraph_count", "total_document_owned_paragraph_count",
    "percentage", "source_document_count", "corpus_baseline",
    "corpus_baseline_percentage", "corpus_baseline_multiple",
    "era_difference_percentage_points",
)
LEGACY_WHY_SHOWN_KEYS = (
    "thresholds_passed", "text", "minimum_issue_paragraphs",
    "absolute_threshold_multiple", "era_threshold_percentage_points",
)
LEGACY_METHOD_KEYS = (
    "distinctive_vocabulary", "distinctive_vocabulary_method", "stance",
    "stance_method",
)
MANIFEST_BUDGET_KEYS = (
    "index_raw_max", "index_gzip_max", "president_shard_raw_max",
    "president_shard_gzip_max", "all_president_shards_raw_max",
    "all_president_shards_gzip_max", "invocation_edge_csv_raw_max",
    "invocation_edge_csv_gzip_max", "invocation_evidence_csv_raw_max",
    "invocation_evidence_csv_gzip_max",
)

DISTINCTIVE_METHOD = (
    "Current legacy method: attach a term after at least two issue uses when "
    "its per-word issue rate is at least 1.5× its rate in the president's text."
)
STANCE_METHOD = (
    "Current legacy method: count fixed era-portable phrases in issue paragraphs; "
    "require eight matches and use a 60/40 split for a directional label."
)
LEGACY_LIMITATION = (
    "This rate is based on source documents assigned to the president. Speaker "
    "attribution is reported separately. Excerpts are deterministic audit "
    "examples, not proof of representativeness, intent, influence, policy "
    "success, or historical importance."
)


class ProfileContextError(RuntimeError):
    """The profile-context contract failed closed."""


@dataclass(frozen=True)
class ProfileContextProjection:
    index: dict[str, Any]
    shards: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    files: dict[str, bytes]


def invocation_receipt_examples(
    shard: Mapping[str, Any], edge_id: str,
) -> list[dict[str, Any]]:
    """Return the preselected invocation audit examples for server rendering."""
    invocations = shard.get("invocations", {})
    rows = invocations.get("receipts_by_edge", {}).get(edge_id, [])
    positions = invocations.get("selected_receipt_positions_by_edge", {}).get(
        edge_id, []
    )
    try:
        return [rows[int(position)] for position in positions]
    except (IndexError, TypeError, ValueError) as exc:
        raise ProfileContextError("invocation receipt position is invalid") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    without = {key: item for key, item in value.items() if key != field}
    return _sha256_bytes(_canonical_json(without).encode("utf-8"))


def _gzip_size(value: bytes) -> int:
    return len(gzip.compress(value, compresslevel=9, mtime=0))


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise ProfileContextError("profile context contains a non-finite value")
    return value


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return _scalar(value)


def _records(frame: pd.DataFrame, fields: Sequence[str]) -> list[dict[str, Any]]:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ProfileContextError(f"source rows lack fields: {sorted(missing)}")
    return [
        {field: _scalar(value) for field, value in zip(fields, row, strict=True)}
        for row in frame.loc[:, list(fields)].itertuples(index=False, name=None)
    ]


def _require_exact_keys(
    value: Any, expected: Sequence[str], label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileContextError(f"{label} must be an object")
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise ProfileContextError(
            f"{label} key drift: missing={sorted(wanted - observed)}, "
            f"extra={sorted(observed - wanted)}"
        )
    return value


def _reject_literal_nan_identifiers(
    value: Any, *, field: str | None = None, path: str = "root",
) -> None:
    """Reject stringified missing identifiers while allowing ordinary prose."""
    identifier_field = bool(
        field
        and (
            field == "slug"
            or field.endswith("_id")
            or field.endswith("_ids")
            or field in {"filename", "doc_name"}
        )
    )
    if identifier_field and isinstance(value, str) and value.strip().lower() == "nan":
        raise ProfileContextError(f"literal nan identifier at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                field in {"receipts_by_edge", "selected_receipt_positions_by_edge"}
                and str(key).strip().lower() == "nan"
            ):
                raise ProfileContextError(f"literal nan identifier at {path}.{key}")
            _reject_literal_nan_identifiers(
                item, field=str(key), path=f"{path}.{key}"
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_literal_nan_identifiers(
                item, field=field, path=f"{path}[{index}]"
            )


def _file_receipt(value: bytes, schema: str, rows: int) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "rows": int(rows),
        "bytes": len(value),
        "gzip_bytes": _gzip_size(value),
        "sha256": _sha256_bytes(value),
    }


def _corpus_level1(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["scope_type"].eq(SCOPE_TYPE)
        & frame["scope_id"].eq(SCOPE_ID)
        & frame["topic_level"].eq("level1")
    ].copy()


def _safe_topic_receipts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    normalized_urls: list[str] = []
    for row in rows.itertuples(index=False):
        try:
            normalized = actual_speaker_invocation_network.normalized_miller_url(
                str(row.doc_name), str(row.source_url)
            )
        except actual_speaker_invocation_network.ActualSpeakerInvocationNetworkError as exc:
            raise ProfileContextError("unsafe topic-receipt source URL") from exc
        normalized_urls.append(normalized)
    rows["source_url"] = normalized_urls
    return rows


def _resolve_governed_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_current_hash(value: str | Path, expected: Any, label: str) -> None:
    path = _resolve_governed_path(value)
    if not path.is_file():
        raise ProfileContextError(f"governed source is missing: {label}")
    if not isinstance(expected, str) or _sha256_file(path) != expected:
        raise ProfileContextError(f"governed source hash is stale: {label}")


def _rehash_governed_sources(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
) -> None:
    """Re-hash every resolvable accepted source immediately before projection."""
    meta_path = speaker_topic_network.NETWORK_DIR / "meta_v1.json"
    if not meta_path.is_file():
        raise ProfileContextError("governed speaker-topic metadata is missing")
    try:
        disk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileContextError("governed speaker-topic metadata is invalid") from exc
    if disk_meta != topic_bundle.meta:
        raise ProfileContextError("governed speaker-topic metadata is stale")
    artifacts = topic_bundle.meta.get("artifacts", {})
    for filename, receipt in artifacts.items():
        _require_current_hash(
            speaker_topic_network.NETWORK_DIR / filename,
            receipt.get("sha256"),
            f"speaker topic artifact {filename}",
        )
    try:
        disk_topic_bundle = speaker_topic_network.load_network_bundle(
            network_dir=speaker_topic_network.NETWORK_DIR,
            foundation=None,
            validate_current_inputs=False,
        )
    except speaker_topic_network.SpeakerTopicNetworkError as exc:
        raise ProfileContextError(
            "governed speaker-topic bundle cannot be replayed from disk"
        ) from exc
    for field in (
        "paragraph_topics", "memberships", "president_nodes", "topic_nodes",
        "president_support", "topic_support", "edges", "edge_evidence",
    ):
        if not getattr(topic_bundle, field).equals(getattr(disk_topic_bundle, field)):
            raise ProfileContextError(
                f"loaded speaker-topic bundle differs from disk: {field}"
            )
    if topic_bundle.acceptance != disk_topic_bundle.acceptance:
        raise ProfileContextError(
            "loaded speaker-topic acceptance differs from disk"
        )
    foundation = topic_bundle.meta.get("input_provenance", {}).get(
        "story_foundation", {}
    )
    source_paths = foundation.get("source_paths", {})
    source_hashes = foundation.get("source_hashes", {})
    if set(source_paths) != set(source_hashes):
        raise ProfileContextError("speaker-foundation source receipt drift")
    for key, value in source_paths.items():
        _require_current_hash(value, source_hashes[key], f"speaker foundation {key}")
    promotion = topic_bundle.meta.get("input_provenance", {}).get(
        "active_topic_promotion", {}
    )
    for path_key, hash_key in (
        ("current_labels_path", "current_labels_sha256"),
        ("generation_manifest_path", "generation_manifest_sha256"),
        ("pointer_path", "pointer_sha256"),
    ):
        _require_current_hash(
            promotion.get(path_key, ""), promotion.get(hash_key),
            f"active topic promotion {path_key}",
        )
    taxonomy = topic_bundle.meta.get("input_provenance", {}).get(
        "topic_taxonomy", {}
    )
    _require_current_hash(
        taxonomy.get("path", ""), taxonomy.get("sha256"), "topic taxonomy"
    )
    invocation_sources = invocation_bundle.source_identity.get("accepted_sources", {})
    if not invocation_sources:
        raise ProfileContextError("invocation accepted-source receipts are missing")
    for key, receipt in invocation_sources.items():
        _require_current_hash(
            receipt.get("path", ""), receipt.get("sha256"),
            f"invocation accepted source {key}",
        )


def _source_identity(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
    profile_views: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _rehash_governed_sources(topic_bundle, invocation_bundle)
    artifacts = topic_bundle.meta.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ProfileContextError("speaker-topic bundle lacks governed artifacts")
    public_v3 = {
        president: {
            key: _json_native(value)
            for key, value in view.items()
            if key != "_view"
        }
        for president, view in profile_views.items()
    }
    legacy_paths = {
        "paragraphs": profiles.PARAGRAPHS_PATH,
        "paragraph_issues": issues.PARA_LABELS_PATH,
        "speeches": corpus.DATA_DIR / "speeches.parquet",
    }
    legacy_receipts: dict[str, Any] = {}
    for key, path in legacy_paths.items():
        if not Path(path).is_file():
            raise ProfileContextError(f"legacy issue source is missing: {path}")
        digest = _sha256_file(Path(path))
        if digest != EXPECTED_LEGACY_SOURCE_HASHES[key]:
            raise ProfileContextError(f"legacy issue source hash drift: {key}")
        legacy_receipts[key] = {
            "path": str(Path(path).resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": digest,
            "bytes": Path(path).stat().st_size,
        }
    invocation_identity = invocation_bundle.source_identity
    invocation_sources = invocation_identity.get("accepted_sources", {})
    foundation_identity = topic_bundle.meta.get("input_provenance", {}).get(
        "story_foundation", {}
    )
    foundation_receipt = {
        "schema_version": foundation_identity.get("governed_schema_version"),
        "canonical_corpus_fingerprint": foundation_identity.get(
            "canonical_corpus_fingerprint"
        ),
        "speaker_metadata_sha256": foundation_identity.get("speaker_metadata_sha256"),
        "reference_metadata_sha256": foundation_identity.get(
            "reference_metadata_sha256"
        ),
    }
    invocation_source_hashes = {
        key: value.get("sha256")
        for key, value in sorted(invocation_sources.items())
    }
    legacy_source_hashes = {
        key: value["sha256"] for key, value in sorted(legacy_receipts.items())
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "topic_network": {
            "contract_version": speaker_topic_network.CONTRACT_VERSION,
            "metadata_sha256": topic_bundle.meta.get("metadata_sha256"),
            "artifact_set_sha256": _sha256_bytes(
                _canonical_json(
                    {
                        filename: receipt.get("sha256")
                        for filename, receipt in sorted(artifacts.items())
                    }
                ).encode("utf-8")
            ),
            "artifact_count": len(artifacts),
            "active_topic_generation": topic_bundle.meta.get("input_provenance", {})
            .get("active_topic_promotion", {})
            .get("active_generation_id"),
            "story_foundation_identity_sha256": _sha256_bytes(
                _canonical_json(foundation_receipt).encode("utf-8")
            ),
        },
        "invocation_network": {
            "contract_version": invocation_identity.get("contract_version"),
            "candidate_fingerprint": invocation_identity.get("candidate_fingerprint"),
            "rubric_version": invocation_identity.get("rubric_version"),
            "classifier_prompt_sha256": invocation_identity.get(
                "classifier_prompt_sha256"
            ),
            "accepted_source_set_sha256": _sha256_bytes(
                _canonical_json(invocation_source_hashes).encode("utf-8")
            ),
            "accepted_source_count": len(invocation_source_hashes),
            "president_catalog_metadata_sha256": invocation_identity.get(
                "president_catalog_metadata_sha256"
            ),
        },
        "legacy_issue_evidence": {
            "schema_version": LEGACY_EVIDENCE_SCHEMA,
            "source_set_sha256": _sha256_bytes(
                _canonical_json(legacy_source_hashes).encode("utf-8")
            ),
            "source_count": len(legacy_source_hashes),
            "profile_v3_sha256": _sha256_bytes(
                _canonical_json(public_v3).encode("utf-8")
            ),
        },
    }


def _president_catalog(bundle: speaker_topic_network.SpeakerTopicNetworkBundle) -> pd.DataFrame:
    rows = bundle.president_nodes.sort_values("display_order", kind="stable").copy()
    rows["president_display_name"] = rows["president_name"].map(profiles.public_display_name)
    rows["slug"] = rows["president_name"].map(profiles.slug)
    if len(rows) != EXPECTED_COUNTS["presidents"] or rows["slug"].duplicated().any():
        raise ProfileContextError("president catalog population drift")
    if not rows["slug"].eq(rows["president_profile_id"]).all():
        raise ProfileContextError("president profile IDs and canonical slugs diverge")
    return rows


def _ordered_sources(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    presidents = _president_catalog(topic_bundle)
    topics = topic_bundle.topic_nodes.loc[
        topic_bundle.topic_nodes["topic_level"].eq("level1")
    ].sort_values(["display_order", "topic_id"], kind="stable")
    president_order = presidents.set_index("president_profile_id")["display_order"]
    topic_order = topics.set_index("topic_id")["display_order"]
    president_support = topic_bundle.president_support.loc[
        topic_bundle.president_support["scope_type"].eq(SCOPE_TYPE)
        & topic_bundle.president_support["scope_id"].eq(SCOPE_ID)
    ].copy()
    president_support["_order"] = president_support["president_profile_id"].map(president_order)
    president_support = president_support.sort_values(
        ["_order"], kind="stable"
    ).drop(columns="_order")
    topic_support = topic_bundle.topic_support.loc[
        topic_bundle.topic_support["scope_type"].eq(SCOPE_TYPE)
        & topic_bundle.topic_support["scope_id"].eq(SCOPE_ID)
        & topic_bundle.topic_support["topic_level"].eq("level1")
    ].copy()
    topic_support["_order"] = topic_support["topic_id"].map(topic_order)
    topic_support = topic_support.sort_values(["_order"], kind="stable").drop(columns="_order")
    edges = _corpus_level1(topic_bundle.edges)
    edges["_topic_order"] = edges["topic_id"].map(topic_order)
    edges["_president_order"] = edges["president_profile_id"].map(president_order)
    edges = edges.sort_values(
        ["_president_order", "_topic_order", "edge_id"], kind="stable"
    ).drop(columns=["_topic_order", "_president_order"])
    if (
        len(topics) != EXPECTED_COUNTS["level1_topics"]
        or len(edges) != EXPECTED_COUNTS["observed_topic_edges"]
        or int(edges["edge_support_status"].eq("supported").sum())
        != EXPECTED_COUNTS["supported_topic_edges"]
        or int(edges["default_visible"].sum())
        != EXPECTED_COUNTS["default_visible_topic_edges"]
        or president_support["president_support_status"].value_counts().to_dict()
        != {"supported": 42, "thin": 3}
    ):
        raise ProfileContextError("all-corpus Level 1 population/support drift")
    return presidents, topics, president_support, topic_support, edges


def _rank_edges(
    rows: pd.DataFrame,
    topic_order: Mapping[str, Any],
    president_order: Mapping[str, Any],
) -> pd.DataFrame:
    ranked = rows.copy()
    ranked["_topic_order"] = ranked["topic_id"].map(topic_order)
    ranked["_president_order"] = ranked["president_profile_id"].map(president_order)
    if ranked[["_topic_order", "_president_order"]].isna().any().any():
        raise ProfileContextError("topic selection contains an unknown node")
    return ranked.sort_values(
        [
            "speaker_paragraph_share", "topic_paragraph_count",
            "topic_appearance_count", "_topic_order", "_president_order",
            "edge_id",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).drop(columns=["_topic_order", "_president_order"])


def _topic_selection(
    focal_id: str,
    all_edges: pd.DataFrame,
    *,
    supported_president: bool,
    expanded: bool,
    topic_order: Mapping[str, Any],
    president_order: Mapping[str, Any],
) -> dict[str, Any]:
    """Precompute one field-preserving radial selection; the browser never ranks."""
    if not supported_president:
        return {
            "node_ids": [focal_id],
            "edge_ids": [],
            "mobile_topic_peer_node_ids": [],
        }
    focal = all_edges.loc[
        all_edges["president_profile_id"].eq(focal_id)
        & all_edges["edge_support_status"].eq("supported")
    ].copy()
    if not expanded:
        focal = focal.loc[focal["default_visible"]].copy()
    focal = _rank_edges(focal, topic_order, president_order).head(10 if expanded else 5)
    topic_ids = focal["topic_id"].astype(str).tolist()
    peers_per_topic = 4 if expanded else 2
    unique_peer_cap = 20 if expanded else 10
    candidate_by_topic: dict[str, list[dict[str, str]]] = {}
    for topic_id in topic_ids:
        candidates = all_edges.loc[
            all_edges["topic_id"].eq(topic_id)
            & all_edges["edge_support_status"].eq("supported")
            & all_edges["president_profile_id"].ne(focal_id)
        ].copy()
        if not expanded:
            candidates = candidates.loc[candidates["default_visible"]].copy()
        ranked = _rank_edges(candidates, topic_order, president_order).head(peers_per_topic)
        candidate_by_topic[topic_id] = [
            {
                "president_profile_id": str(row.president_profile_id),
                "edge_id": str(row.edge_id),
            }
            for row in ranked.itertuples(index=False)
        ]

    peer_ids: list[str] = []
    peer_edge_ids: list[str] = []
    mobile_node_order: list[str] = []
    seen_peers: set[str] = set()
    seen_edges: set[str] = set()
    # Round-robin by focal-topic rank first, then peer rank.
    for peer_rank in range(peers_per_topic):
        for topic_id in topic_ids:
            candidates = candidate_by_topic[topic_id]
            if peer_rank >= len(candidates):
                continue
            candidate = candidates[peer_rank]
            peer_id = candidate["president_profile_id"]
            if peer_id not in seen_peers:
                if len(peer_ids) >= unique_peer_cap:
                    continue
                seen_peers.add(peer_id)
                peer_ids.append(peer_id)
            edge_id = candidate["edge_id"]
            if edge_id not in seen_edges:
                seen_edges.add(edge_id)
                peer_edge_ids.append(edge_id)
                for node_id in (topic_id, peer_id):
                    if node_id not in mobile_node_order:
                        mobile_node_order.append(node_id)

    focal_edge_ids = focal["edge_id"].astype(str).tolist()
    return {
        "node_ids": [focal_id, *topic_ids, *peer_ids],
        "edge_ids": [*focal_edge_ids, *peer_edge_ids],
        "mobile_topic_peer_node_ids": mobile_node_order[:6],
    }


def _thin_topic_selection(
    focal_id: str,
    all_edges: pd.DataFrame,
    topic_order: Mapping[str, Any],
    president_order: Mapping[str, Any],
) -> dict[str, Any]:
    focal = _rank_edges(
        all_edges.loc[all_edges["president_profile_id"].eq(focal_id)],
        topic_order,
        president_order,
    ).head(5)
    edge_ids = focal["edge_id"].astype(str).tolist()
    topic_ids = focal["topic_id"].astype(str).tolist()
    return {
        "comparison_allowed": False,
        "edge_style": "dashed",
        "node_ids": [focal_id, *topic_ids],
        "edge_ids": edge_ids,
        "mobile_topic_peer_node_ids": topic_ids[:6],
    }


def _invocation_receipt_selection(rows: list[dict[str, Any]]) -> list[str]:
    """Select first/lower-middle/last distinct-speech audit examples."""
    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            row["speech_date"], row["doc_name"], int(row["para_idx"]),
            row["candidate_id"],
        ),
    )
    by_speech: dict[str, dict[str, Any]] = {}
    for row in ordered:
        by_speech.setdefault(row["doc_name"], row)
    speeches = list(by_speech.values())
    positions = [0, (len(speeches) - 1) // 2, len(speeches) - 1]
    selected: list[str] = []
    for position in positions:
        candidate_id = str(speeches[position]["candidate_id"])
        if candidate_id not in selected:
            selected.append(candidate_id)
    return selected


def _invocation_receipt_selection_positions(
    rows: list[dict[str, Any]],
) -> list[int]:
    selected = _invocation_receipt_selection(rows)
    index_by_id = {
        str(row["candidate_id"]): index for index, row in enumerate(rows)
    }
    return [index_by_id[candidate_id] for candidate_id in selected]


def _sentence_score(sentence: str, anchors: Sequence[str], words: Sequence[str]) -> int:
    low = sentence.lower()
    return 2 * sum(
        bool(re.search(rf"\b{re.escape(term)}\w*", low)) for term in anchors
    ) + sum(
        bool(re.search(rf"\b{re.escape(term)}\b", low)) for term in words
    )


def _legacy_sources(
    foundation: story_foundation.StoryFoundationBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paragraphs = pd.read_parquet(profiles.PARAGRAPHS_PATH)
    labels = pd.read_parquet(issues.PARA_LABELS_PATH)
    keys = ["doc_name", "para_idx"]
    for label, frame in (("paragraphs", paragraphs), ("paragraph issues", labels)):
        if frame.duplicated(keys).any():
            raise ProfileContextError(f"{label} contains duplicate paragraph keys")
    if set(map(tuple, paragraphs[keys].itertuples(index=False, name=None))) != set(
        map(tuple, labels[keys].itertuples(index=False, name=None))
    ):
        raise ProfileContextError("paragraph and legacy issue key sets diverge")
    merged = paragraphs.merge(labels, on=keys, how="inner", validate="one_to_one")
    speeches = corpus.load().loc[:, ["doc_name", "president", "date", "title", "year"]]
    if speeches["doc_name"].duplicated().any():
        raise ProfileContextError("speech metadata contains duplicate document keys")
    merged = merged.merge(
        speeches, on="doc_name", how="left", validate="many_to_one",
        suffixes=("", "_speech"),
    )
    if merged[["president_speech", "date", "title", "year_speech"]].isna().any().any():
        raise ProfileContextError("legacy issue paragraph lacks speech metadata")
    if not merged["president"].eq(merged["president_speech"]).all():
        raise ProfileContextError("legacy issue document owner drift")
    overlay = foundation.paragraph_view.copy()
    if overlay.duplicated(keys).any():
        raise ProfileContextError("speaker overlay contains duplicate paragraph keys")
    return merged.reset_index(drop=True), overlay


def _receipt(
    row: pd.Series,
    sentence: str,
    selection_role: str,
    overlay_by_key: pd.DataFrame,
    owner_ids: Mapping[str, str],
) -> dict[str, Any]:
    key = (str(row["doc_name"]), int(row["para_idx"]))
    speaker: dict[str, Any]
    if key in overlay_by_key.index:
        source = overlay_by_key.loc[key]
        if isinstance(source, pd.DataFrame):
            raise ProfileContextError("speaker overlay paragraph key is not one-to-one")
        eligible = bool(source["analysis_eligible"])
        speaker = {
            "actual_speaker": _scalar(source["attributed_speaker"]),
            "actual_speaker_profile_id": _scalar(source["attributed_speaker_profile_id"]),
            "cross_owner": bool(source["cross_owner_paragraph"]),
            "speaker_eligibility_state": "eligible" if eligible else "excluded",
            "speaker_exclusion_reason": None if eligible else _scalar(source["exclusion_reason"]),
        }
        source_url = str(source["source_url"])
    else:
        speaker = {
            "actual_speaker": None,
            "actual_speaker_profile_id": None,
            "cross_owner": None,
            "speaker_eligibility_state": "not_in_accepted_speaker_overlay",
            "speaker_exclusion_reason": (
                "Paragraph is outside the accepted speaker-view population."
            ),
        }
        source_url = profiles.miller_speech_url(key[0])
    try:
        safe_url = actual_speaker_invocation_network.normalized_miller_url(key[0], source_url)
    except actual_speaker_invocation_network.ActualSpeakerInvocationNetworkError as exc:
        raise ProfileContextError("unsafe legacy receipt source URL") from exc
    owner = str(row["president"])
    return {
        "doc_name": key[0],
        "para_idx": key[1],
        "title": str(row["title"]),
        "speech_date": pd.Timestamp(row["date"]).date().isoformat(),
        "year": int(row["year_speech"]),
        "source_url": safe_url,
        "source_document_owner": owner,
        "source_document_owner_profile_id": owner_ids[owner],
        **speaker,
        "excerpt": sentence,
        "selection_role": selection_role,
    }


def _legacy_receipts(
    president: str,
    card: Mapping[str, Any],
    merged: pd.DataFrame,
    overlay_by_key: pd.DataFrame,
    owner_ids: Mapping[str, str],
) -> list[dict[str, Any]]:
    issue = str(card["issue"])
    if issue not in merged.columns:
        raise ProfileContextError(f"unknown legacy issue in profile card: {issue}")
    anchors = {**issues.ISSUE_ANCHORS, **profiles._EXTRA_ANCHORS}
    if issue not in anchors:
        raise ProfileContextError(f"legacy issue lacks receipt anchors: {issue}")
    rows = merged.loc[merged["president"].eq(president) & merged[issue].astype(bool)].copy()
    if rows.empty:
        raise ProfileContextError("legacy evidence card has no keyed issue paragraph")
    words = [str(item) for item in card.get("words", [])]
    hits = rows["text"].str.lower().str.count(
        "|".join(rf"\b{re.escape(term)}\w*" for term in anchors[issue])
    )
    primary_sentence: str | None = None
    primary_index: Any = None
    top_indices = hits.nlargest(3).index
    if card.get("quote") is not None:
        primary_sentence = profiles._pick_sentence(
            [rows.loc[index, "text"] for index in top_indices], anchors[issue], words
        )
        if not primary_sentence or html.escape(primary_sentence) != card.get("quote"):
            raise ProfileContextError("legacy primary excerpt drifted from v3 selection")
        primary_index = next(
            index for index in top_indices if primary_sentence in rows.loc[index, "text"]
        )
        source = rows.loc[primary_index]
        expected_cite = (
            f"{str(source['title']).split(':', 1)[-1].strip()}, "
            f"{int(source['year_speech'])}"
        )
        if expected_cite != card.get("cite"):
            raise ProfileContextError("legacy primary citation drifted from v3 selection")
    elif card.get("cite") is not None:
        raise ProfileContextError("legacy rate-only card unexpectedly carries a citation")

    receipts: list[dict[str, Any]] = []
    excluded_docs: set[str] = set()
    if primary_sentence is not None:
        primary_row = rows.loc[primary_index]
        receipts.append(
            _receipt(primary_row, primary_sentence, "primary", overlay_by_key, owner_ids)
        )
        excluded_docs.add(str(primary_row["doc_name"]))
    else:
        # The existing rate-only state is part of v3 and must remain honest.
        return receipts

    candidates: list[tuple[tuple[Any, ...], pd.Series, str]] = []
    for index, row in rows.iterrows():
        doc_name = str(row["doc_name"])
        if doc_name in excluded_docs:
            continue
        sentence = profiles._pick_sentence([str(row["text"])], anchors[issue], words)
        if not sentence:
            continue
        anchor_hits = int(hits.loc[index])
        score = _sentence_score(sentence, anchors[issue], words)
        order = (
            -anchor_hits, -score, pd.Timestamp(row["date"]).isoformat(),
            doc_name, int(row["para_idx"]), sentence,
        )
        candidates.append((order, row, sentence))
    for number, (_, row, sentence) in enumerate(sorted(candidates, key=lambda item: item[0])):
        doc_name = str(row["doc_name"])
        if doc_name in excluded_docs:
            continue
        receipts.append(
            _receipt(
                row, sentence, f"additional_{len(receipts)}", overlay_by_key, owner_ids
            )
        )
        excluded_docs.add(doc_name)
        if len(receipts) == 3:
            break
    return receipts


def _legacy_evidence_by_president(
    profile_views: Mapping[str, Mapping[str, Any]],
    foundation: story_foundation.StoryFoundationBundle,
    presidents: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    if set(profile_views) != set(presidents["president_name"].astype(str)):
        raise ProfileContextError("profile views do not match the 45-president catalog")
    merged, overlay = _legacy_sources(foundation)
    overlay_by_key = overlay.set_index(["doc_name", "para_idx"], drop=False)
    owner_ids = presidents.set_index("president_name")["president_profile_id"].to_dict()
    first_year = merged.groupby("president", sort=False)["year_speech"].min()
    result: dict[str, dict[str, Any]] = {}
    for node in presidents.itertuples(index=False):
        president = str(node.president_name)
        view = profile_views[president]
        if (
            view.get("schema_version") != "president-profile-v3"
            or view.get("president") != president
        ):
            raise ProfileContextError("profile view identity/schema drift")
        source = _json_native(copy.deepcopy(view.get("issue_evidence") or {}))
        source_cards = list(source.get("cards", []))
        owner_rows = merged.loc[merged["president"].eq(president)]
        total = len(owner_rows)
        if total != int(source.get("n_paragraphs", total)):
            raise ProfileContextError("legacy issue denominator drift")
        cards: list[dict[str, Any]] = []
        for card in source_cards:
            issue = str(card["issue"])
            issue_rows = owner_rows.loc[owner_rows[issue].astype(bool)]
            issue_count = len(issue_rows)
            share = float(card["share"])
            base = float(card["base"])
            era_difference = float(card["rel"])
            if not math.isclose(share, issue_count / total, rel_tol=0, abs_tol=1e-12):
                raise ProfileContextError("legacy issue numerator does not reproduce v3 share")
            expected_base = float(merged[issue].astype(bool).mean())
            if not math.isclose(base, expected_base, rel_tol=0, abs_tol=1e-12):
                raise ProfileContextError("legacy paragraph-weighted corpus baseline drift")
            president_shares = merged.groupby("president", sort=False)[issue].mean()
            peer_mask = (first_year - first_year[president]).abs() <= 24
            peer_mask.loc[president] = False
            peer_shares = (
                president_shares.loc[peer_mask]
                if int(peer_mask.sum()) >= 2
                else president_shares.drop(president)
            )
            expected_era_difference = float(
                (president_shares.loc[president] - peer_shares.mean()) * 100
            )
            if not math.isclose(
                era_difference, expected_era_difference, rel_tol=0, abs_tol=1e-12
            ):
                raise ProfileContextError("legacy era-peer difference/fallback drift")
            baseline_multiple = share / base if base > 0 else None
            passed_absolute = bool(
                baseline_multiple is not None
                and baseline_multiple >= profiles.RAW_ELEVATED_MULT
            )
            passed_era = era_difference >= profiles.REL_DISTINCT_PP
            if issue_count < profiles.MIN_ISSUE_PARAS or not (passed_absolute or passed_era):
                raise ProfileContextError(
                    "published legacy issue card fails its declared threshold"
                )
            if passed_absolute and passed_era:
                claim_type = "absolute_and_era_relative_emphasis"
                claim = (
                    "This issue was elevated both above its corpus baseline and above "
                    "the president's era peers in the source-document population."
                )
            elif passed_era:
                claim_type = "era_relative_emphasis"
                claim = (
                    "This issue stood above the president's era peers in the "
                    "source-document population."
                )
            elif bool(card.get("topic_of_day")):
                claim_type = "absolute_but_era_typical"
                claim = (
                    "This issue was elevated above its corpus baseline but remained "
                    "within the declared era-typical band."
                )
            else:
                claim_type = "absolute_emphasis"
                claim = (
                    "This issue was elevated above its paragraph-weighted corpus "
                    "baseline in the source-document population."
                )
            threshold_ids = []
            threshold_copy = []
            if passed_absolute:
                threshold_ids.append("corpus_baseline_multiple")
                threshold_copy.append(
                    f"its share is {baseline_multiple:.2f}× the corpus baseline "
                    f"(threshold: {profiles.RAW_ELEVATED_MULT:.1f}×)"
                )
            if passed_era:
                threshold_ids.append("era_relative_difference")
                threshold_copy.append(
                    f"its era difference is +{era_difference:.2f} percentage points "
                    f"(threshold: +{profiles.REL_DISTINCT_PP:.2f})"
                )
            why = "Shown because " + " and ".join(threshold_copy) + "."
            receipts = _legacy_receipts(
                president, card, merged, overlay_by_key, owner_ids
            )
            cards.append(
                {
                    "issue": issue,
                    "legacy_v3": card,
                    "claim": {"type": claim_type, "text": claim},
                    "exact_evidence": {
                        "issue_paragraph_count": issue_count,
                        "total_document_owned_paragraph_count": total,
                        "percentage": share * 100,
                        "source_document_count": int(issue_rows["doc_name"].nunique()),
                        "corpus_baseline": base,
                        "corpus_baseline_percentage": base * 100,
                        "corpus_baseline_multiple": baseline_multiple,
                        "era_difference_percentage_points": era_difference,
                    },
                    "why_shown": {
                        "thresholds_passed": threshold_ids,
                        "text": why,
                        "minimum_issue_paragraphs": profiles.MIN_ISSUE_PARAS,
                        "absolute_threshold_multiple": profiles.RAW_ELEVATED_MULT,
                        "era_threshold_percentage_points": profiles.REL_DISTINCT_PP,
                    },
                    "receipts": receipts,
                    "no_qualifying_excerpt_selected": not receipts,
                    "no_qualifying_excerpt_copy": (
                        "No qualifying excerpt was selected by the current deterministic method."
                        if not receipts else None
                    ),
                    "method": {
                        "distinctive_vocabulary": list(card.get("words", [])),
                        "distinctive_vocabulary_method": "legacy_distinctive_vocabulary_v3",
                        "stance": card.get("stance"),
                        "stance_method": "legacy_stance_v3",
                    },
                    "limitation": LEGACY_LIMITATION,
                }
            )
        thin = bool(view.get("sample", {}).get("thin_record"))
        result[president] = {
            "schema_version": LEGACY_EVIDENCE_SCHEMA,
            "population": "Source documents assigned to the president.",
            "cards": cards,
            "voice": list(source.get("voice", [])),
            "n_speeches": int(source.get("n_speeches", view["sample"]["n_speeches"])),
            "n_paragraphs": total,
            "thin_record": thin,
            "thin_record_warning": (
                "Fewer than five source speeches are present; rates are descriptive "
                "but the record is too thin for precise ranking."
                if thin else None
            ),
        }
    return result


def _csv_bytes(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for record in records:
        row = {}
        for field in fields:
            value = record[field]
            if isinstance(value, (dict, list)):
                value = _canonical_json(value)
            elif value is None:
                value = ""
            elif isinstance(value, bool):
                value = "true" if value else "false"
            row[field] = value
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _invocation_sort(
    rows: pd.DataFrame,
    focal_id: str,
    direction: str,
    president_order: Mapping[str, Any],
) -> pd.DataFrame:
    counterpart = (
        "target_president_profile_id" if direction == "outgoing"
        else "source_president_profile_id"
    )
    filtered = rows.loc[
        rows[
            "source_president_profile_id" if direction == "outgoing"
            else "target_president_profile_id"
        ].eq(focal_id)
    ].copy()
    filtered["_counterpart_order"] = filtered[counterpart].map(president_order)
    if filtered["_counterpart_order"].isna().any():
        raise ProfileContextError("invocation edge contains an unknown counterpart")
    return filtered.sort_values(
        [
            "reference_paragraph_count", "raw_mentions", "distinct_speeches",
            "_counterpart_order", "edge_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).drop(columns="_counterpart_order")


def _metrics(topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle) -> dict[str, Any]:
    topic_metrics = topic_bundle.meta.get("metrics")
    if not isinstance(topic_metrics, Mapping):
        raise ProfileContextError("speaker-topic metrics are missing")
    projected_topic_metrics = {
        key: {
            field: _json_native(spec[field])
            for field in ("measure", "definition", "formula", "unit")
        }
        for key, spec in topic_metrics.items()
    }
    return {
        **projected_topic_metrics,
        "actual_speaker_invocation_paragraph_count": {
            "measure": "reference_paragraph_count",
            "definition": (
                "Distinct keyed paragraphs in which the accepted invocation evidence "
                "records the actual speaker naming the former-president target."
            ),
            "formula": "count(distinct (doc_name, para_idx)) per directed edge",
            "population": (
                "Resolved, analysis-eligible, former-president, non-self rows in "
                "the accepted actual-speaker invocation overlay."
            ),
            "unit": "reference paragraphs",
            "interpretation": (
                "Descriptive corpus evidence only; not a normalized rate, historical "
                "completeness claim, influence measure, confidence score, or causal claim."
            ),
        },
    }


def _policies(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> dict[str, Any]:
    configuration = topic_bundle.meta["source_configuration"]
    return {
        "topic_free_denominator_policy": configuration[
            "topic_free_denominator_policy"
        ],
        "multi_label_counting_policy": configuration["multi_label_counting_policy"],
        "topic_support_thresholds": configuration["support_thresholds"],
        "topic_evidence_selection_policy": configuration[
            "evidence_selection_policy"
        ],
        "topic_interpretation_policy": configuration["interpretation_policy"],
        "topic_selection_policy": (
            "Python precomputes focal topics and peers in the declared measure/order; "
            "peer selection is round-robin and never browser-derived."
        ),
        "invocation_population_policy": (
            "Resolved, analysis-eligible, former-president, non-self accepted "
            "invocation rows attributed through the speaker overlay."
        ),
        "invocation_target_status_limitation": (
            "target_status is determined from corpus speech dates rather than "
            "an independent legal-term calendar."
        ),
        "legacy_issue_thresholds": {
            "minimum_issue_paragraphs": profiles.MIN_ISSUE_PARAS,
            "corpus_baseline_multiple": profiles.RAW_ELEVATED_MULT,
            "era_difference_percentage_points": profiles.REL_DISTINCT_PP,
            "era_peer_window_years": 24,
            "era_peer_fallback": (
                "All other presidents when fewer than two window peers exist."
            ),
        },
        "legacy_issue_methods": {
            "legacy_distinctive_vocabulary_v3": DISTINCTIVE_METHOD,
            "legacy_stance_v3": STANCE_METHOD,
        },
        "population_labels": {
            "core_profile": "Source documents assigned to the president.",
            "connections": "Eligible paragraphs attributed to the actual speaker.",
        },
    }


def _manifest_budgets() -> dict[str, int]:
    return {
        "index_raw_max": INDEX_RAW_MAX,
        "index_gzip_max": INDEX_GZIP_MAX,
        "president_shard_raw_max": SHARD_RAW_MAX,
        "president_shard_gzip_max": SHARD_GZIP_MAX,
        "all_president_shards_raw_max": ALL_SHARDS_RAW_MAX,
        "all_president_shards_gzip_max": ALL_SHARDS_GZIP_MAX,
        "invocation_edge_csv_raw_max": INVOCATION_EDGE_CSV_RAW_MAX,
        "invocation_edge_csv_gzip_max": INVOCATION_EDGE_CSV_GZIP_MAX,
        "invocation_evidence_csv_raw_max": INVOCATION_EVIDENCE_CSV_RAW_MAX,
        "invocation_evidence_csv_gzip_max": INVOCATION_EVIDENCE_CSV_GZIP_MAX,
    }


def _supported_index_edges(
    all_edges: pd.DataFrame,
    topic_order: Mapping[str, Any],
    president_order: Mapping[str, Any],
) -> pd.DataFrame:
    rows = all_edges.loc[all_edges["edge_support_status"].eq("supported")].copy()
    rows["_topic_order"] = rows["topic_id"].map(topic_order)
    rows["_president_order"] = rows["president_profile_id"].map(president_order)
    return rows.sort_values(
        ["_topic_order", "_president_order", "edge_id"], kind="stable"
    ).drop(columns=["_topic_order", "_president_order"])


def _validate_byte_budgets(projection: ProfileContextProjection) -> None:
    try:
        index_value = projection.files[INDEX_FILE]
        edge_csv = projection.files[INVOCATION_EDGE_CSV]
        evidence_csv = projection.files[INVOCATION_EVIDENCE_CSV]
        shard_values = [projection.files[path] for path in projection.shards]
    except KeyError as exc:
        raise ProfileContextError("profile-context budget input is missing") from exc
    checks = (
        (len(index_value), INDEX_RAW_MAX, "index raw"),
        (_gzip_size(index_value), INDEX_GZIP_MAX, "index gzip"),
        (max(map(len, shard_values), default=0), SHARD_RAW_MAX, "president shard raw"),
        (
            max(map(_gzip_size, shard_values), default=0), SHARD_GZIP_MAX,
            "president shard gzip",
        ),
        (sum(map(len, shard_values)), ALL_SHARDS_RAW_MAX, "all shards raw"),
        (
            sum(map(_gzip_size, shard_values)), ALL_SHARDS_GZIP_MAX,
            "all shards gzip",
        ),
        (len(edge_csv), INVOCATION_EDGE_CSV_RAW_MAX, "invocation edge CSV raw"),
        (
            _gzip_size(edge_csv), INVOCATION_EDGE_CSV_GZIP_MAX,
            "invocation edge CSV gzip",
        ),
        (
            len(evidence_csv), INVOCATION_EVIDENCE_CSV_RAW_MAX,
            "invocation evidence CSV raw",
        ),
        (
            _gzip_size(evidence_csv), INVOCATION_EVIDENCE_CSV_GZIP_MAX,
            "invocation evidence CSV gzip",
        ),
    )
    for observed, limit, label in checks:
        if observed > limit:
            raise ProfileContextError(
                f"profile-context {label} budget exceeded: {observed} > {limit}"
            )


def build_projection(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
    foundation: story_foundation.StoryFoundationBundle,
    profile_views: Mapping[str, Mapping[str, Any]],
) -> ProfileContextProjection:
    """Build and validate all 49 public files in memory without writing ``docs``."""
    story_foundation.check_story_contract(foundation)
    speaker_topic_network.validate_bundle(topic_bundle)
    actual_speaker_invocation_network.validate_bundle(invocation_bundle)
    presidents, topics, president_support, topic_support, all_edges = _ordered_sources(
        topic_bundle
    )
    source_identity = _source_identity(topic_bundle, invocation_bundle, profile_views)
    legacy = _legacy_evidence_by_president(profile_views, foundation, presidents)
    president_order = presidents.set_index("president_profile_id")["display_order"].to_dict()
    topic_order = topics.set_index("topic_id")["display_order"].to_dict()

    topic_receipts = topic_bundle.edge_evidence.loc[
        topic_bundle.edge_evidence["scope_type"].eq(SCOPE_TYPE)
        & topic_bundle.edge_evidence["scope_id"].eq(SCOPE_ID)
        & topic_bundle.edge_evidence["topic_level"].eq("level1")
    ].copy()
    topic_receipts = _safe_topic_receipts(topic_receipts).sort_values(
        ["edge_id", "selection_position", "receipt_id"], kind="stable"
    )
    if len(topic_receipts) != EXPECTED_COUNTS["topic_receipts"]:
        raise ProfileContextError("all-corpus Level 1 receipt population drift")

    invocation_edges = actual_speaker_invocation_network.edge_records(invocation_bundle)
    invocation_evidence = actual_speaker_invocation_network.evidence_records(invocation_bundle)
    if (
        len(invocation_edges) != EXPECTED_COUNTS["invocation_edges"]
        or len(invocation_evidence) != EXPECTED_COUNTS["invocation_rows"]
    ):
        raise ProfileContextError("invocation population drift")
    invocation_edge_frame = invocation_bundle.edges.copy()
    evidence_by_edge: dict[str, list[dict[str, Any]]] = {}
    for row in invocation_evidence:
        evidence_by_edge.setdefault(str(row["edge_id"]), []).append(row)
    for edge_id, rows in evidence_by_edge.items():
        evidence_by_edge[edge_id] = sorted(
            rows,
            key=lambda row: (
                row["speech_date"], row["doc_name"], int(row["para_idx"]),
                row["candidate_id"],
            ),
        )

    shards: dict[str, dict[str, Any]] = {}
    shard_bytes: dict[str, bytes] = {}
    shard_receipts: list[dict[str, Any]] = []
    support_by_id = president_support.set_index("president_profile_id", drop=False)
    for node in presidents.itertuples(index=False):
        focal_id = str(node.president_profile_id)
        president = str(node.president_name)
        support_row = support_by_id.loc[focal_id]
        supported = str(support_row["president_support_status"]) == "supported"
        focal_edges = all_edges.loc[
            all_edges["president_profile_id"].eq(focal_id)
        ].sort_values(
            "topic_id",
            key=lambda values: values.map(topic_order),
            kind="stable",
        )
        focal_edge_ids = set(focal_edges["edge_id"].astype(str))
        focal_receipts = topic_receipts.loc[
            topic_receipts["edge_id"].astype(str).isin(focal_edge_ids)
        ].sort_values(["edge_id", "selection_position", "receipt_id"], kind="stable")

        outgoing = _invocation_sort(
            invocation_edge_frame, focal_id, "outgoing", president_order
        )
        incoming = _invocation_sort(
            invocation_edge_frame, focal_id, "incoming", president_order
        )
        outgoing_ids = outgoing["edge_id"].astype(str).tolist()
        incoming_ids = incoming["edge_id"].astype(str).tolist()
        invocation_ids = [*outgoing_ids, *incoming_ids]
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ProfileContextError("self/duplicate invocation edge reference in shard")
        receipt_map = {
            edge_id: copy.deepcopy(evidence_by_edge[edge_id])
            for edge_id in invocation_ids
        }
        selected_receipts = {
            edge_id: _invocation_receipt_selection_positions(receipt_map[edge_id])
            for edge_id in invocation_ids
        }
        default_selection = _topic_selection(
            focal_id, all_edges, supported_president=supported, expanded=False,
            topic_order=topic_order, president_order=president_order,
        )
        expanded_selection = _topic_selection(
            focal_id, all_edges, supported_president=supported, expanded=True,
            topic_order=topic_order, president_order=president_order,
        )
        thin_selection = _thin_topic_selection(
            focal_id, all_edges, topic_order, president_order
        ) if not supported else {
            "comparison_allowed": False,
            "edge_style": "dashed",
            "node_ids": [focal_id],
            "edge_ids": [],
            "mobile_topic_peer_node_ids": [],
        }
        shard = {
            "schema_version": PRESIDENT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "source_identity": source_identity,
            "president": _records(
                presidents.loc[presidents["president_profile_id"].eq(focal_id)],
                PRESIDENT_FIELDS,
            )[0],
            "president_support": _records(
                president_support.loc[
                    president_support["president_profile_id"].eq(focal_id)
                ],
                PRESIDENT_SUPPORT_FIELDS,
            )[0],
            "topic_edges": _records(focal_edges, TOPIC_EDGE_FIELDS),
            "topic_receipts": _records(focal_receipts, TOPIC_RECEIPT_FIELDS),
            "topic_selections": {
                "default": default_selection,
                "expanded": expanded_selection,
                "observed_thin": thin_selection,
            },
            "invocations": {
                "default_direction": (
                    "outgoing" if outgoing_ids else "incoming" if incoming_ids else None
                ),
                "outgoing_edge_ids": outgoing_ids,
                "incoming_edge_ids": incoming_ids,
                "receipts_by_edge": receipt_map,
                "selected_receipt_positions_by_edge": selected_receipts,
            },
            "legacy_issue_evidence": legacy[president],
            "counts": {
                "focal_topic_edges": len(focal_edges),
                "focal_topic_receipts": len(focal_receipts),
                "outgoing_invocation_edges": len(outgoing_ids),
                "incoming_invocation_edges": len(incoming_ids),
                "invocation_receipt_references": sum(
                    len(rows) for rows in receipt_map.values()
                ),
                "legacy_issue_cards": len(legacy[president]["cards"]),
            },
        }
        shard["shard_sha256"] = _self_hash(shard, "shard_sha256")
        relative = f"presidents/{node.slug}_v1.json"
        value = _json_bytes(shard)
        if len(value) > SHARD_RAW_MAX or _gzip_size(value) > SHARD_GZIP_MAX:
            raise ProfileContextError(f"president context shard exceeds budget: {relative}")
        shards[relative] = shard
        shard_bytes[relative] = value
        shard_file_receipt = _file_receipt(
            value,
            PRESIDENT_SCHEMA,
            len(focal_edges) + len(focal_receipts)
            + len(invocation_ids)
            + sum(len(rows) for rows in receipt_map.values())
            + len(legacy[president]["cards"]),
        )
        shard_receipts.append(
            {
                "president_profile_id": focal_id,
                "filename": relative,
                "sha256": shard_file_receipt["sha256"],
            }
        )

    all_shard_raw = sum(len(value) for value in shard_bytes.values())
    all_shard_gzip = sum(_gzip_size(value) for value in shard_bytes.values())
    if all_shard_raw > ALL_SHARDS_RAW_MAX or all_shard_gzip > ALL_SHARDS_GZIP_MAX:
        raise ProfileContextError("combined president shards exceed their byte budget")

    supported_edges = _supported_index_edges(all_edges, topic_order, president_order)
    index = {
        "schema_version": INDEX_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "source_identity": source_identity,
        "scope": {
            "scope_type": SCOPE_TYPE,
            "scope_id": SCOPE_ID,
        },
        "metrics": _metrics(topic_bundle),
        "policies": _policies(topic_bundle),
        "presidents": _records(presidents, PRESIDENT_FIELDS),
        "level1_topics": _records(topics, TOPIC_FIELDS),
        "president_support": _records(president_support, PRESIDENT_SUPPORT_FIELDS),
        "topic_support": _records(topic_support, TOPIC_SUPPORT_FIELDS),
        "topic_edges": _records(supported_edges, TOPIC_EDGE_FIELDS),
        "invocation_edges": invocation_edges,
        "president_shards": shard_receipts,
        "downloads": {
            "invocation_edges": INVOCATION_EDGE_CSV,
            "invocation_evidence": INVOCATION_EVIDENCE_CSV,
        },
        "counts": {
            "eligible_actual_speaker_paragraphs": 32_531,
            "presidents": len(presidents),
            "supported_presidents": int(
                president_support["president_support_status"].eq("supported").sum()
            ),
            "thin_presidents": int(
                president_support["president_support_status"].eq("thin").sum()
            ),
            "level1_topics": len(topics),
            "observed_topic_edges_across_shards": len(all_edges),
            "supported_comparison_topic_edges": len(supported_edges),
            "default_visible_topic_edges": int(all_edges["default_visible"].sum()),
            "topic_receipts_across_shards": len(topic_receipts),
            "invocation_edges": len(invocation_edges),
            "invocation_rows": len(invocation_evidence),
            "invocation_edge_references_across_shards": sum(
                shard["counts"]["outgoing_invocation_edges"]
                + shard["counts"]["incoming_invocation_edges"]
                for shard in shards.values()
            ),
            "invocation_evidence_references_across_shards": sum(
                shard["counts"]["invocation_receipt_references"]
                for shard in shards.values()
            ),
            "legacy_issue_cards_across_shards": sum(
                shard["counts"]["legacy_issue_cards"] for shard in shards.values()
            ),
        },
    }
    index["index_sha256"] = _self_hash(index, "index_sha256")
    index_bytes = _json_bytes(index)
    if len(index_bytes) > INDEX_RAW_MAX or _gzip_size(index_bytes) > INDEX_GZIP_MAX:
        raise ProfileContextError("profile-context index exceeds its byte budget")

    edge_csv = _csv_bytes(
        invocation_edges, actual_speaker_invocation_network.EDGE_FIELDS
    )
    evidence_csv = _csv_bytes(
        invocation_evidence, actual_speaker_invocation_network.EVIDENCE_FIELDS
    )
    if (
        len(edge_csv) > INVOCATION_EDGE_CSV_RAW_MAX
        or _gzip_size(edge_csv) > INVOCATION_EDGE_CSV_GZIP_MAX
    ):
        raise ProfileContextError("invocation edge CSV exceeds its byte budget")
    if (
        len(evidence_csv) > INVOCATION_EVIDENCE_CSV_RAW_MAX
        or _gzip_size(evidence_csv) > INVOCATION_EVIDENCE_CSV_GZIP_MAX
    ):
        raise ProfileContextError("invocation evidence CSV exceeds its byte budget")

    files = {
        INDEX_FILE: index_bytes,
        INVOCATION_EDGE_CSV: edge_csv,
        INVOCATION_EVIDENCE_CSV: evidence_csv,
        **shard_bytes,
    }
    file_receipts = {
        INDEX_FILE: _file_receipt(index_bytes, INDEX_SCHEMA, len(supported_edges)),
        INVOCATION_EDGE_CSV: _file_receipt(
            edge_csv, actual_speaker_invocation_network.EDGE_SCHEMA,
            len(invocation_edges),
        ),
        INVOCATION_EVIDENCE_CSV: _file_receipt(
            evidence_csv, actual_speaker_invocation_network.EVIDENCE_SCHEMA,
            len(invocation_evidence),
        ),
        **{
            relative: _file_receipt(
                value,
                PRESIDENT_SCHEMA,
                shards[relative]["counts"]["focal_topic_edges"]
                + shards[relative]["counts"]["focal_topic_receipts"]
                + shards[relative]["counts"]["outgoing_invocation_edges"]
                + shards[relative]["counts"]["incoming_invocation_edges"]
                + shards[relative]["counts"]["invocation_receipt_references"]
                + shards[relative]["counts"]["legacy_issue_cards"],
            )
            for relative, value in sorted(shard_bytes.items())
        },
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "source_identity": source_identity,
        "files": file_receipts,
        "counts": {
            "files_excluding_manifest": len(files),
            "president_shards": len(shards),
            **index["counts"],
        },
        "budgets": _manifest_budgets(),
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    files[MANIFEST_FILE] = _json_bytes(manifest)
    projection = ProfileContextProjection(index, shards, manifest, files)
    validate_projection(
        projection, topic_bundle, invocation_bundle, foundation, profile_views
    )
    return projection


def _assert_records_equal(
    projected: Iterable[Mapping[str, Any]],
    source: pd.DataFrame,
    fields: Sequence[str],
    keys: Sequence[str],
    label: str,
) -> None:
    expected = _records(source, fields)
    actual: list[dict[str, Any]] = []
    for index, row in enumerate(projected):
        exact = _require_exact_keys(row, fields, f"{label} record {index}")
        actual.append(dict(exact))
    expected_keys = [tuple(row[key] for key in keys) for row in expected]
    actual_keys = [tuple(row[key] for key in keys) for row in actual]
    if len(set(expected_keys)) != len(expected_keys):
        raise ProfileContextError(f"governed {label} source keys are not unique")
    if len(set(actual_keys)) != len(actual_keys):
        raise ProfileContextError(f"projected {label} keys are not unique")
    if actual != expected:
        raise ProfileContextError(f"projected {label} differs from governed source rows")


def _validate_selection(
    selection: Mapping[str, Any],
    focal_id: str,
    supported_edges: set[str],
    focal_edges: set[str],
    *,
    thin: bool,
) -> None:
    edge_ids = list(selection.get("edge_ids", []))
    if len(edge_ids) != len(set(edge_ids)):
        raise ProfileContextError("topic selection edge inventory/order drift")
    if not thin and not set(edge_ids).issubset(supported_edges):
        raise ProfileContextError("comparative topic selection includes an unsupported edge")
    if thin and not set(edge_ids).issubset(focal_edges):
        raise ProfileContextError("thin topic selection contains comparative peer edges")
    node_ids = list(selection.get("node_ids", []))
    if not node_ids or node_ids[0] != focal_id or len(node_ids) != len(set(node_ids)):
        raise ProfileContextError("topic selection node inventory drift")
    if len(selection.get("mobile_topic_peer_node_ids", [])) > 6:
        raise ProfileContextError("mobile topic selection exceeds six topic/peer nodes")


def validate_projection(
    projection: ProfileContextProjection,
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
    foundation: story_foundation.StoryFoundationBundle,
    profile_views: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reconcile every public row, reference, hash, and byte budget."""
    story_foundation.check_story_contract(foundation)
    speaker_topic_network.validate_bundle(topic_bundle)
    actual_speaker_invocation_network.validate_bundle(invocation_bundle)
    presidents, topics, president_support, topic_support, all_edges = _ordered_sources(
        topic_bundle
    )
    expected_identity = _source_identity(topic_bundle, invocation_bundle, profile_views)
    _reject_literal_nan_identifiers(projection.index, path="index")
    _reject_literal_nan_identifiers(projection.shards, path="shards")
    _reject_literal_nan_identifiers(projection.manifest, path="manifest")
    index = _require_exact_keys(projection.index, INDEX_KEYS, "profile-context index")
    if index.get("schema_version") != INDEX_SCHEMA:
        raise ProfileContextError("profile-context index schema drift")
    if index.get("contract_version") != CONTRACT_VERSION:
        raise ProfileContextError("profile-context index contract drift")
    if index.get("index_sha256") != _self_hash(index, "index_sha256"):
        raise ProfileContextError("profile-context index self-hash drift")
    if index.get("source_identity") != expected_identity:
        raise ProfileContextError("profile-context source identity is stale")
    if index.get("scope") != {
        "scope_type": SCOPE_TYPE,
        "scope_id": SCOPE_ID,
    }:
        raise ProfileContextError("profile-context scope drift")
    if index.get("metrics") != _metrics(topic_bundle):
        raise ProfileContextError("profile-context metric definitions drift")
    if index.get("policies") != _policies(topic_bundle):
        raise ProfileContextError("profile-context policy definitions drift")
    if index.get("downloads") != {
        "invocation_edges": INVOCATION_EDGE_CSV,
        "invocation_evidence": INVOCATION_EVIDENCE_CSV,
    }:
        raise ProfileContextError("profile-context download map drift")
    _assert_records_equal(
        index.get("presidents", []), presidents, PRESIDENT_FIELDS,
        ("president_profile_id",), "president nodes",
    )
    _assert_records_equal(
        index.get("level1_topics", []), topics, TOPIC_FIELDS,
        ("topic_id",), "Level 1 topic nodes",
    )
    _assert_records_equal(
        index.get("president_support", []), president_support,
        PRESIDENT_SUPPORT_FIELDS, ("president_profile_id",), "president support",
    )
    _assert_records_equal(
        index.get("topic_support", []), topic_support,
        TOPIC_SUPPORT_FIELDS, ("topic_id",), "topic support",
    )
    president_order = presidents.set_index("president_profile_id")["display_order"].to_dict()
    topic_order = topics.set_index("topic_id")["display_order"].to_dict()
    supported_source = _supported_index_edges(all_edges, topic_order, president_order)
    _assert_records_equal(
        index.get("topic_edges", []), supported_source, TOPIC_EDGE_FIELDS,
        ("edge_id",), "supported comparison topic edges",
    )
    expected_invocation_edges = actual_speaker_invocation_network.edge_records(
        invocation_bundle
    )
    if index.get("invocation_edges") != expected_invocation_edges:
        raise ProfileContextError("index invocation edges differ from governed aggregation")
    expected_count_values = {
        "eligible_actual_speaker_paragraphs": 32_531,
        "presidents": EXPECTED_COUNTS["presidents"],
        "supported_presidents": EXPECTED_COUNTS["supported_presidents"],
        "thin_presidents": EXPECTED_COUNTS["thin_presidents"],
        "level1_topics": EXPECTED_COUNTS["level1_topics"],
        "observed_topic_edges_across_shards": EXPECTED_COUNTS["observed_topic_edges"],
        "supported_comparison_topic_edges": EXPECTED_COUNTS["supported_topic_edges"],
        "default_visible_topic_edges": EXPECTED_COUNTS["default_visible_topic_edges"],
        "topic_receipts_across_shards": EXPECTED_COUNTS["topic_receipts"],
        "invocation_edges": EXPECTED_COUNTS["invocation_edges"],
        "invocation_rows": EXPECTED_COUNTS["invocation_rows"],
        "invocation_edge_references_across_shards": EXPECTED_COUNTS[
            "invocation_shard_edge_references"
        ],
        "invocation_evidence_references_across_shards": EXPECTED_COUNTS[
            "invocation_shard_evidence_references"
        ],
        "legacy_issue_cards_across_shards": EXPECTED_COUNTS["legacy_cards"],
    }
    if index.get("counts") != expected_count_values:
        raise ProfileContextError("profile-context pinned count/key drift")

    shard_receipt_rows: list[dict[str, Any]] = []
    for position, row in enumerate(index.get("president_shards", [])):
        exact = _require_exact_keys(
            row, SHARD_RECEIPT_FIELDS, f"president shard receipt {position}"
        )
        shard_receipt_rows.append(dict(exact))
    shard_receipts = {
        row["president_profile_id"]: row for row in shard_receipt_rows
    }
    if (
        len(shard_receipts) != 45
        or [row["president_profile_id"] for row in shard_receipt_rows]
        != presidents["president_profile_id"].astype(str).tolist()
    ):
        raise ProfileContextError("president shard index drift")
    source_receipts = _safe_topic_receipts(topic_bundle.edge_evidence.loc[
        topic_bundle.edge_evidence["scope_type"].eq(SCOPE_TYPE)
        & topic_bundle.edge_evidence["scope_id"].eq(SCOPE_ID)
        & topic_bundle.edge_evidence["topic_level"].eq("level1")
    ]).sort_values(["edge_id", "selection_position", "receipt_id"], kind="stable")
    supported_edge_ids = set(supported_source["edge_id"].astype(str))
    expected_legacy = _legacy_evidence_by_president(
        profile_views, foundation, presidents
    )
    covered_topic_edges: list[str] = []
    covered_topic_receipts: list[str] = []
    invocation_edge_refs: list[str] = []
    invocation_evidence_refs: list[str] = []
    legacy_card_count = 0
    primary_receipt_count = 0
    rate_only_count = 0
    president_name_by_id = presidents.set_index("president_profile_id")["president_name"].to_dict()
    for node in presidents.itertuples(index=False):
        focal_id = str(node.president_profile_id)
        relative = f"presidents/{node.slug}_v1.json"
        raw_shard = projection.shards.get(relative)
        if raw_shard is None:
            raise ProfileContextError(f"president shard schema/inventory drift: {relative}")
        shard = _require_exact_keys(raw_shard, SHARD_KEYS, f"president shard {relative}")
        if shard.get("schema_version") != PRESIDENT_SCHEMA:
            raise ProfileContextError(f"president shard schema/inventory drift: {relative}")
        if shard.get("contract_version") != CONTRACT_VERSION:
            raise ProfileContextError(f"president shard contract drift: {relative}")
        if shard.get("shard_sha256") != _self_hash(shard, "shard_sha256"):
            raise ProfileContextError(f"president shard self-hash drift: {relative}")
        if shard.get("source_identity") != expected_identity:
            raise ProfileContextError(f"president shard source identity drift: {relative}")
        receipt = shard_receipts.get(focal_id)
        if receipt is None or receipt.get("filename") != relative:
            raise ProfileContextError(f"president shard receipt drift: {relative}")
        shard_value = projection.files.get(relative)
        if shard_value is None or shard_value != _json_bytes(shard):
            raise ProfileContextError(f"president shard object/byte drift: {relative}")
        if receipt.get("sha256") != _sha256_bytes(shard_value):
            raise ProfileContextError(f"president shard index hash drift: {relative}")
        _assert_records_equal(
            [shard.get("president", {})],
            presidents.loc[presidents["president_profile_id"].eq(focal_id)],
            PRESIDENT_FIELDS, ("president_profile_id",), f"{focal_id} president",
        )
        _assert_records_equal(
            [shard.get("president_support", {})],
            president_support.loc[
                president_support["president_profile_id"].eq(focal_id)
            ],
            PRESIDENT_SUPPORT_FIELDS, ("president_profile_id",),
            f"{focal_id} president support",
        )
        focal_source = all_edges.loc[all_edges["president_profile_id"].eq(focal_id)]
        _assert_records_equal(
            shard.get("topic_edges", []), focal_source, TOPIC_EDGE_FIELDS,
            ("edge_id",), f"{focal_id} focal topic edges",
        )
        focal_edge_ids = set(focal_source["edge_id"].astype(str))
        receipt_source = source_receipts.loc[
            source_receipts["edge_id"].astype(str).isin(focal_edge_ids)
        ]
        _assert_records_equal(
            shard.get("topic_receipts", []), receipt_source, TOPIC_RECEIPT_FIELDS,
            ("receipt_id",), f"{focal_id} topic receipts",
        )
        covered_topic_edges.extend(row["edge_id"] for row in shard["topic_edges"])
        covered_topic_receipts.extend(row["receipt_id"] for row in shard["topic_receipts"])
        support = shard.get("president_support", {}).get("president_support_status")
        selections = _require_exact_keys(
            shard.get("topic_selections", {}), TOPIC_SELECTIONS_KEYS,
            f"{focal_id} topic selections",
        )
        _require_exact_keys(
            selections.get("default", {}), TOPIC_SELECTION_KEYS,
            f"{focal_id} default topic selection",
        )
        _require_exact_keys(
            selections.get("expanded", {}), TOPIC_SELECTION_KEYS,
            f"{focal_id} expanded topic selection",
        )
        _require_exact_keys(
            selections.get("observed_thin", {}), THIN_TOPIC_SELECTION_KEYS,
            f"{focal_id} thin topic selection",
        )
        expected_default = _topic_selection(
            focal_id, all_edges, supported_president=support == "supported",
            expanded=False, topic_order=topic_order, president_order=president_order,
        )
        expected_expanded = _topic_selection(
            focal_id, all_edges, supported_president=support == "supported",
            expanded=True, topic_order=topic_order, president_order=president_order,
        )
        expected_thin = _thin_topic_selection(
            focal_id, all_edges, topic_order, president_order
        ) if support == "thin" else {
            "comparison_allowed": False,
            "edge_style": "dashed",
            "node_ids": [focal_id],
            "edge_ids": [],
            "mobile_topic_peer_node_ids": [],
        }
        if selections != {
            "default": expected_default,
            "expanded": expected_expanded,
            "observed_thin": expected_thin,
        }:
            raise ProfileContextError("precomputed topic selection ordering drift")
        _validate_selection(
            selections.get("default", {}), focal_id, supported_edge_ids,
            focal_edge_ids, thin=False,
        )
        _validate_selection(
            selections.get("expanded", {}), focal_id, supported_edge_ids,
            focal_edge_ids, thin=False,
        )
        _validate_selection(
            selections.get("observed_thin", {}), focal_id, supported_edge_ids,
            focal_edge_ids, thin=True,
        )
        if support == "thin" and (
            selections["default"]["edge_ids"] or selections["expanded"]["edge_ids"]
        ):
            raise ProfileContextError("thin president received comparative topic selection")
        if support == "supported" and selections["observed_thin"]["edge_ids"]:
            raise ProfileContextError("supported president received thin-only topic selection")

        invocations = _require_exact_keys(
            shard.get("invocations", {}), INVOCATION_SHARD_KEYS,
            f"{focal_id} shard invocations",
        )
        outgoing = list(invocations.get("outgoing_edge_ids", []))
        incoming = list(invocations.get("incoming_edge_ids", []))
        expected_outgoing = _invocation_sort(
            invocation_bundle.edges, focal_id, "outgoing", president_order
        )["edge_id"].astype(str).tolist()
        expected_incoming = _invocation_sort(
            invocation_bundle.edges, focal_id, "incoming", president_order
        )["edge_id"].astype(str).tolist()
        if outgoing != expected_outgoing or incoming != expected_incoming:
            raise ProfileContextError("invocation shard direction/order drift")
        expected_direction = (
            "outgoing" if expected_outgoing else "incoming" if expected_incoming else None
        )
        if invocations.get("default_direction") != expected_direction:
            raise ProfileContextError("invocation shard default direction drift")
        edge_ids = [*outgoing, *incoming]
        edge_by_id = invocation_bundle.edges.set_index("edge_id", drop=False)
        if any(
            str(edge_by_id.loc[edge_id, "source_president_profile_id"]) != focal_id
            for edge_id in outgoing
        ) or any(
            str(edge_by_id.loc[edge_id, "target_president_profile_id"]) != focal_id
            for edge_id in incoming
        ):
            raise ProfileContextError("invocation shard focal endpoint drift")
        invocation_edge_refs.extend(edge_ids)
        receipts_by_edge = invocations.get("receipts_by_edge", {})
        selected_by_edge = invocations.get("selected_receipt_positions_by_edge", {})
        if not isinstance(receipts_by_edge, Mapping) or not isinstance(
            selected_by_edge, Mapping
        ):
            raise ProfileContextError("invocation shard receipt maps must be objects")
        if set(receipts_by_edge) != set(edge_ids) or set(selected_by_edge) != set(edge_ids):
            raise ProfileContextError("invocation shard edge/receipt map mismatch")
        for edge_id in edge_ids:
            rows = receipts_by_edge[edge_id]
            source_rows = invocation_bundle.evidence.loc[
                invocation_bundle.evidence["edge_id"].eq(edge_id)
            ].sort_values(
                ["speech_date", "doc_name", "para_idx", "candidate_id"],
                kind="stable",
            )
            _assert_records_equal(
                rows, source_rows, actual_speaker_invocation_network.EVIDENCE_FIELDS,
                ("candidate_id",), f"{focal_id} invocation receipts",
            )
            expected_selected = _invocation_receipt_selection_positions(rows)
            if selected_by_edge[edge_id] != expected_selected:
                raise ProfileContextError("invocation audit-receipt selection drift")
            invocation_evidence_refs.extend(row["candidate_id"] for row in rows)
        legacy = _require_exact_keys(
            shard.get("legacy_issue_evidence", {}), LEGACY_EVIDENCE_KEYS,
            f"{focal_id} legacy issue evidence",
        )
        if legacy.get("schema_version") != LEGACY_EVIDENCE_SCHEMA:
            raise ProfileContextError("legacy issue evidence schema drift")
        cards = legacy.get("cards", [])
        if not isinstance(cards, list):
            raise ProfileContextError("legacy issue evidence cards must be an array")
        legacy_card_count += len(cards)
        for card_index, card in enumerate(cards):
            _require_exact_keys(
                card, LEGACY_CARD_KEYS,
                f"{focal_id} legacy issue card {card_index}",
            )
            _require_exact_keys(
                card.get("claim", {}), LEGACY_CLAIM_KEYS,
                f"{focal_id} legacy claim {card_index}",
            )
            _require_exact_keys(
                card.get("exact_evidence", {}), LEGACY_EXACT_EVIDENCE_KEYS,
                f"{focal_id} legacy exact evidence {card_index}",
            )
            _require_exact_keys(
                card.get("why_shown", {}), LEGACY_WHY_SHOWN_KEYS,
                f"{focal_id} legacy why-shown {card_index}",
            )
            _require_exact_keys(
                card.get("method", {}), LEGACY_METHOD_KEYS,
                f"{focal_id} legacy method {card_index}",
            )
            receipts = card.get("receipts", [])
            if not isinstance(receipts, list):
                raise ProfileContextError("legacy issue evidence receipts must be an array")
            if receipts:
                primary_receipt_count += int(receipts[0].get("selection_role") == "primary")
            else:
                rate_only_count += 1
            if card.get("limitation") != LEGACY_LIMITATION:
                raise ProfileContextError("legacy evidence card limitation drift")
            for receipt_index, evidence in enumerate(receipts):
                _require_exact_keys(
                    evidence, LEGACY_RECEIPT_FIELDS,
                    f"{focal_id} legacy receipt {card_index}.{receipt_index}",
                )
                try:
                    actual_speaker_invocation_network.normalized_miller_url(
                        evidence["doc_name"], evidence["source_url"]
                    )
                except actual_speaker_invocation_network.ActualSpeakerInvocationNetworkError as exc:
                    raise ProfileContextError("unsafe legacy evidence URL") from exc
        if legacy != expected_legacy[president_name_by_id[focal_id]]:
            raise ProfileContextError("enriched legacy issue evidence drift")
        expected_shard_counts = {
            "focal_topic_edges": len(focal_source),
            "focal_topic_receipts": len(receipt_source),
            "outgoing_invocation_edges": len(expected_outgoing),
            "incoming_invocation_edges": len(expected_incoming),
            "invocation_receipt_references": sum(
                len(receipts_by_edge[edge_id]) for edge_id in edge_ids
            ),
            "legacy_issue_cards": len(cards),
        }
        _require_exact_keys(
            shard.get("counts", {}), SHARD_COUNT_KEYS, f"{focal_id} shard counts"
        )
        if shard.get("counts") != expected_shard_counts:
            raise ProfileContextError("president shard count drift")
    if len(covered_topic_edges) != 700 or len(set(covered_topic_edges)) != 700:
        raise ProfileContextError("focal topic edges are not covered exactly once")
    if set(covered_topic_edges) != set(all_edges["edge_id"].astype(str)):
        raise ProfileContextError("focal topic edge set differs from governed source")
    if len(covered_topic_receipts) != 1_945 or len(set(covered_topic_receipts)) != 1_945:
        raise ProfileContextError("focal topic receipts are not covered exactly once")
    if set(covered_topic_receipts) != set(source_receipts["receipt_id"].astype(str)):
        raise ProfileContextError("focal topic receipt set differs from governed source")
    expected_edge_ids = set(invocation_bundle.edges["edge_id"].astype(str))
    if len(invocation_edge_refs) != 606 or set(invocation_edge_refs) != expected_edge_ids:
        raise ProfileContextError("invocation edges are not referenced twice across shards")
    if any(invocation_edge_refs.count(edge_id) != 2 for edge_id in expected_edge_ids):
        raise ProfileContextError("invocation edge shard multiplicity drift")
    expected_candidate_ids = set(invocation_bundle.evidence["candidate_id"].astype(str))
    if (
        len(invocation_evidence_refs) != 2_486
        or set(invocation_evidence_refs) != expected_candidate_ids
    ):
        raise ProfileContextError("invocation evidence is not referenced twice across shards")
    if any(invocation_evidence_refs.count(candidate) != 2 for candidate in expected_candidate_ids):
        raise ProfileContextError("invocation evidence shard multiplicity drift")
    if (
        legacy_card_count != EXPECTED_COUNTS["legacy_cards"]
        or primary_receipt_count != EXPECTED_COUNTS["legacy_primary_receipts"]
        or rate_only_count != EXPECTED_COUNTS["legacy_rate_only_cards"]
    ):
        raise ProfileContextError("legacy issue card/primary/rate-only parity drift")

    manifest = _require_exact_keys(
        projection.manifest, MANIFEST_KEYS, "profile-context manifest"
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ProfileContextError("profile-context manifest schema drift")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ProfileContextError("profile-context manifest contract drift")
    if manifest.get("source_identity") != expected_identity:
        raise ProfileContextError("profile-context manifest source identity drift")
    expected_manifest_counts = {
        "files_excluding_manifest": 48,
        "president_shards": 45,
        **expected_count_values,
    }
    if manifest.get("counts") != expected_manifest_counts:
        raise ProfileContextError("profile-context manifest count/key drift")
    if manifest.get("budgets") != _manifest_budgets():
        raise ProfileContextError("profile-context manifest budget/key drift")
    _require_exact_keys(
        manifest.get("budgets", {}), MANIFEST_BUDGET_KEYS,
        "profile-context manifest budgets",
    )
    if manifest.get("manifest_sha256") != _self_hash(
        manifest, "manifest_sha256"
    ):
        raise ProfileContextError("profile-context manifest self-hash drift")
    expected_paths = {
        INDEX_FILE, INVOCATION_EDGE_CSV, INVOCATION_EVIDENCE_CSV, *projection.shards
    }
    if len(expected_paths) != 48:
        raise ProfileContextError("profile-context expected pre-manifest inventory drift")
    if set(manifest.get("files", {})) != expected_paths:
        raise ProfileContextError("profile-context manifest inventory drift")
    if set(projection.files) != expected_paths | {MANIFEST_FILE}:
        raise ProfileContextError("profile-context in-memory inventory drift")
    if len(projection.files) != EXPECTED_COUNTS["public_files"]:
        raise ProfileContextError("profile-context file-count drift")
    if projection.files[INDEX_FILE] != _json_bytes(projection.index):
        raise ProfileContextError("profile-context index bytes drift")
    if projection.files[MANIFEST_FILE] != _json_bytes(manifest):
        raise ProfileContextError("profile-context manifest bytes drift")
    _validate_byte_budgets(projection)
    expected_receipts: dict[str, dict[str, Any]] = {
        INDEX_FILE: _file_receipt(
            projection.files[INDEX_FILE], INDEX_SCHEMA,
            EXPECTED_COUNTS["supported_topic_edges"],
        ),
        INVOCATION_EDGE_CSV: _file_receipt(
            projection.files[INVOCATION_EDGE_CSV],
            actual_speaker_invocation_network.EDGE_SCHEMA,
            EXPECTED_COUNTS["invocation_edges"],
        ),
        INVOCATION_EVIDENCE_CSV: _file_receipt(
            projection.files[INVOCATION_EVIDENCE_CSV],
            actual_speaker_invocation_network.EVIDENCE_SCHEMA,
            EXPECTED_COUNTS["invocation_rows"],
        ),
    }
    for relative, shard in projection.shards.items():
        counts = shard["counts"]
        rows = sum(int(counts[key]) for key in SHARD_COUNT_KEYS)
        expected_receipts[relative] = _file_receipt(
            projection.files[relative], PRESIDENT_SCHEMA, rows
        )
    for relative, receipt in manifest["files"].items():
        _require_exact_keys(
            receipt, FILE_RECEIPT_FIELDS,
            f"profile-context manifest receipt {relative}",
        )
        if receipt != expected_receipts[relative]:
            raise ProfileContextError(
                f"profile-context manifest/file receipt drift: {relative}"
            )
    if projection.files[INVOCATION_EDGE_CSV] != _csv_bytes(
        expected_invocation_edges, actual_speaker_invocation_network.EDGE_FIELDS
    ):
        raise ProfileContextError("invocation edge CSV differs from governed rows")
    expected_invocation_evidence = actual_speaker_invocation_network.evidence_records(
        invocation_bundle
    )
    if projection.files[INVOCATION_EVIDENCE_CSV] != _csv_bytes(
        expected_invocation_evidence,
        actual_speaker_invocation_network.EVIDENCE_FIELDS,
    ):
        raise ProfileContextError("invocation evidence CSV differs from governed rows")


def verify_reproducible(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
    foundation: story_foundation.StoryFoundationBundle,
    profile_views: Mapping[str, Mapping[str, Any]],
) -> ProfileContextProjection:
    """Build twice in memory and require byte-identical complete inventories."""
    first = build_projection(topic_bundle, invocation_bundle, foundation, profile_views)
    second = build_projection(topic_bundle, invocation_bundle, foundation, profile_views)
    if first.files != second.files:
        raise ProfileContextError("two profile-context builds differ byte-for-byte")
    return first


def _write_candidate(projection: ProfileContextProjection, output: Path) -> None:
    for relative in sorted(projection.files, key=lambda item: item == MANIFEST_FILE):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(projection.files[relative])


def _load_projection_from_site(site_dir: Path) -> ProfileContextProjection:
    output = Path(site_dir) / "data" / PUBLIC_DIR_NAME
    manifest_path = output / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ProfileContextError("profile-context manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ProfileContextError("profile-context manifest is invalid JSON") from exc
    _require_exact_keys(manifest, MANIFEST_KEYS, "profile-context manifest")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise ProfileContextError("profile-context manifest files must be an object")
    expected: set[str] = set()
    output_resolved = output.resolve()
    for relative in manifest_files:
        if not isinstance(relative, str) or not (
            relative in {
                INDEX_FILE, INVOCATION_EDGE_CSV, INVOCATION_EVIDENCE_CSV,
            }
            or re.fullmatch(r"presidents/[a-z0-9-]+_v1\.json", relative)
        ):
            raise ProfileContextError("profile-context manifest contains an unsafe path")
        candidate = output / relative
        try:
            candidate.resolve().relative_to(output_resolved)
        except ValueError as exc:
            raise ProfileContextError(
                "profile-context manifest path escapes the owned directory"
            ) from exc
        expected.add(relative)
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
    }
    if actual != expected | {MANIFEST_FILE}:
        raise ProfileContextError("profile-context disk inventory drift")
    files = {relative: (output / relative).read_bytes() for relative in expected}
    files[MANIFEST_FILE] = manifest_bytes
    try:
        index = json.loads(files[INDEX_FILE])
        shards = {
            relative: json.loads(files[relative])
            for relative in expected if relative.startswith("presidents/")
        }
    except (KeyError, json.JSONDecodeError) as exc:
        raise ProfileContextError("profile-context public JSON is invalid") from exc
    return ProfileContextProjection(index, shards, manifest, files)


def validate_publication(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
    foundation: story_foundation.StoryFoundationBundle,
    profile_views: Mapping[str, Mapping[str, Any]],
    site_dir: Path = REPO_ROOT / "docs",
) -> dict[str, Any]:
    projection = _load_projection_from_site(Path(site_dir))
    validate_projection(
        projection, topic_bundle, invocation_bundle, foundation, profile_views
    )
    return {
        "schema_version": INDEX_SCHEMA,
        "files": len(projection.files),
        **projection.index["counts"],
    }


def write_public_projection(
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    invocation_bundle: actual_speaker_invocation_network.ActualSpeakerInvocationNetworkBundle,
    foundation: story_foundation.StoryFoundationBundle,
    profile_views: Mapping[str, Mapping[str, Any]],
    site_dir: Path = REPO_ROOT / "docs",
    *,
    projection: ProfileContextProjection | None = None,
) -> list[Path]:
    """Validate in isolation, then atomically swap only the owned directory.

    A second complete in-memory build is mandatory even when the caller passes
    a projection, so no public write can occur without byte reproducibility.
    The existing complete directory is restored if either the swap or the
    post-swap validation fails.
    """
    first = projection or build_projection(
        topic_bundle, invocation_bundle, foundation, profile_views
    )
    validate_projection(first, topic_bundle, invocation_bundle, foundation, profile_views)
    repeat = build_projection(topic_bundle, invocation_bundle, foundation, profile_views)
    if first.files != repeat.files:
        raise ProfileContextError("two profile-context builds differ before publication")

    site_root = Path(site_dir)
    parent = site_root / "data"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / PUBLIC_DIR_NAME
    transaction = Path(
        tempfile.mkdtemp(prefix=f".{PUBLIC_DIR_NAME}-transaction-", dir=parent)
    )
    candidate = transaction / "candidate"
    candidate.mkdir()
    backup = transaction / "backup"
    failed = transaction / "failed-candidate"
    candidate_site = transaction / "validation-site"
    existed = output.exists()
    preserve_transaction = False
    try:
        _write_candidate(first, candidate)
        try:
            (candidate_site / "data").mkdir(parents=True, exist_ok=True)
            validation_link = candidate_site / "data" / PUBLIC_DIR_NAME
            os.symlink(candidate, validation_link, target_is_directory=True)
            validate_publication(
                topic_bundle, invocation_bundle, foundation, profile_views,
                candidate_site,
            )
        finally:
            if candidate_site.exists() or candidate_site.is_symlink():
                shutil.rmtree(candidate_site, ignore_errors=True)
        if existed:
            os.replace(output, backup)
        try:
            os.replace(candidate, output)
            validate_publication(
                topic_bundle, invocation_bundle, foundation, profile_views, site_root
            )
        except BaseException as publish_error:
            try:
                if output.exists() or output.is_symlink():
                    os.replace(output, failed)
                if existed:
                    if not backup.exists():
                        raise ProfileContextError(
                            "profile-context rollback backup is missing"
                        )
                    os.replace(backup, output)
            except BaseException as restore_error:
                # The unique transaction directory is intentionally retained.
                # In particular, never delete the sole complete backup when an
                # external filesystem failure prevents restoration.
                preserve_transaction = True
                raise ProfileContextError(
                    "profile-context rollback failed; preserved recovery data at "
                    f"{transaction}"
                ) from restore_error
            raise
    finally:
        if not preserve_transaction and transaction.exists():
            shutil.rmtree(transaction)
    return [output / relative for relative in sorted(first.files)]
