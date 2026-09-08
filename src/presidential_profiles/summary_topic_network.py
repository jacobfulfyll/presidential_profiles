"""Field-preserving Summary projection of the governed speaker/topic network.

This module never derives speakers, topics, denominators, support, measures, or
evidence.  It selects the all-corpus rows and public fields needed by the
page-agnostic relationship browser, validates them back to the accepted Plan 3
bundle, and writes a deterministic lazy-loading projection with an exact
recurring-relationship CSV download.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import speaker_topic_network
from .summary_topic_network_assets import SUMMARY_TOPIC_NETWORK_CSS, SUMMARY_TOPIC_NETWORK_JS


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR_NAME = "summary-topic-network"
PUBLIC_DIR = REPO_ROOT / "docs" / "data" / PUBLIC_DIR_NAME
ASSET_JS_NAME = "topic-relationship-browser-v1.js"
ASSET_CSS_NAME = "topic-relationship-browser-v1.css"

INDEX_SCHEMA = "summary-actual-speaker-topic-network-index-v1"
TOPIC_SCHEMA = "summary-actual-speaker-topic-network-topic-v1"
MANIFEST_SCHEMA = "summary-actual-speaker-topic-network-manifest-v1"
INDEX_FILE = "index_v1.json"
MANIFEST_FILE = "manifest_v1.json"
RECURRING_TABLE_FILE = "recurring_topic_table_v1.csv"
SCOPE_TYPE = "corpus"
SCOPE_ID = "all-corpus"

INDEX_RAW_MAX = 550_000
INDEX_GZIP_MAX = 80_000
SHARD_RAW_MAX = 1_200_000
SHARD_GZIP_MAX = 275_000
RENDERER_JS_RAW_MAX = 50_000

PRESIDENT_FIELDS = (
    "president_profile_id",
    "president_name",
    "party",
    "display_order",
    "node_type",
)
TOPIC_FIELDS = (
    "topic_id",
    "topic_level",
    "topic_label",
    "topic_definition",
    "topic_kind",
    "parent_topic_id",
    "parent_topic_label",
    "display_order",
    "node_type",
)
PRESIDENT_SUPPORT_FIELDS = (
    "president_profile_id",
    "president_name",
    "eligible_president_paragraph_count",
    "eligible_president_appearance_count",
    "president_support_status",
)
TOPIC_SUPPORT_FIELDS = (
    "topic_id",
    "topic_level",
    "topic_label",
    "scope_topic_paragraph_count",
    "scope_eligible_paragraph_count",
    "topic_scope_share",
    "topic_support_status",
)
EDGE_FIELDS = (
    "edge_id",
    "president_profile_id",
    "topic_id",
    "topic_paragraph_count",
    "topic_appearance_count",
    "speaker_paragraph_share",
    "topic_contribution_share",
    "president_support_status",
    "edge_support_status",
    "default_visible",
)
EVIDENCE_FIELDS = (
    "receipt_id",
    "edge_id",
    "para_idx",
    "actual_speaker",
    "source_document_owner",
    "cross_owner",
    "story_era_label",
    "speech_date",
    "speech_title",
    "source_url",
    "evidence_excerpt",
    "selection_role",
    "selection_position",
)


class SummaryTopicNetworkError(RuntimeError):
    """The Summary projection failed closed."""


@dataclass(frozen=True)
class SummaryTopicNetworkProjection:
    index: dict[str, Any]
    shards: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    files: dict[str, bytes]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha256_bytes(
        _canonical_json({key: item for key, item in value.items() if key != field}).encode()
    )


def _gzip_size(value: bytes) -> int:
    return len(gzip.compress(value, compresslevel=9, mtime=0))


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise SummaryTopicNetworkError("projection contains a non-finite value")
    return value


def _records(frame: pd.DataFrame, fields: Sequence[str]) -> list[dict[str, Any]]:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise SummaryTopicNetworkError(f"Plan 3 rows lack projected fields: {sorted(missing)}")
    return [
        {field: _scalar(value) for field, value in zip(fields, row, strict=True)}
        for row in frame.loc[:, list(fields)].itertuples(index=False, name=None)
    ]


def _source_identity(bundle: speaker_topic_network.SpeakerTopicNetworkBundle) -> dict[str, Any]:
    artifacts = bundle.meta.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SummaryTopicNetworkError("Plan 3 metadata lacks governed artifact hashes")
    return {
        "contract_version": speaker_topic_network.CONTRACT_VERSION,
        "metadata_sha256": bundle.meta.get("metadata_sha256"),
        "artifacts": {
            filename: receipt.get("sha256")
            for filename, receipt in sorted(artifacts.items())
        },
        "input_provenance": bundle.meta.get("input_provenance"),
    }


def _corpus_rows(frame: pd.DataFrame, *, topic_level: str | None = None) -> pd.DataFrame:
    rows = frame.loc[
        frame["scope_type"].eq(SCOPE_TYPE) & frame["scope_id"].eq(SCOPE_ID)
    ].copy()
    if topic_level is not None:
        rows = rows.loc[rows["topic_level"].eq(topic_level)].copy()
    return rows


def _file_receipt(value: bytes, schema_version: str, rows: int) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "rows": int(rows),
        "bytes": len(value),
        "gzip_bytes": _gzip_size(value),
        "sha256": _sha256_bytes(value),
    }


def _recurring_table_frame(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> pd.DataFrame:
    """Select the supported recurring Level 1 rows without changing their fields."""
    president_order = bundle.president_nodes.set_index("president_profile_id")["display_order"]
    topic_order = bundle.topic_nodes.loc[
        bundle.topic_nodes["topic_level"].eq("level1")
    ].set_index("topic_id")["display_order"]
    rows = _corpus_rows(bundle.edges, topic_level="level1").loc[
        lambda frame: frame["default_visible"]
        & frame["president_support_status"].eq("supported")
    ].copy()
    rows["_president_order"] = rows["president_profile_id"].map(president_order)
    rows["_topic_order"] = rows["topic_id"].map(topic_order)
    rows = rows.sort_values(
        ["_president_order", "_topic_order", "edge_id"], kind="stable"
    )
    return rows.loc[:, list(bundle.edges.columns)].reset_index(drop=True)


def _recurring_table_bytes(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> bytes:
    return _recurring_table_frame(bundle).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")


def build_projection(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> SummaryTopicNetworkProjection:
    """Select and validate the complete all-corpus Summary projection in memory."""
    speaker_topic_network.validate_bundle(bundle)
    source_identity = _source_identity(bundle)

    presidents = bundle.president_nodes.sort_values("display_order", kind="stable")
    level1_topics = bundle.topic_nodes.loc[
        bundle.topic_nodes["topic_level"].eq("level1")
    ].sort_values("display_order", kind="stable")
    level2_topics = bundle.topic_nodes.loc[
        bundle.topic_nodes["topic_level"].eq("level2")
    ].sort_values(["parent_topic_id", "display_order", "topic_id"], kind="stable")
    president_support = _corpus_rows(bundle.president_support).sort_values(
        "president_profile_id", key=lambda col: col.map(
            presidents.set_index("president_profile_id")["display_order"]
        ), kind="stable"
    )
    topic_support = _corpus_rows(bundle.topic_support, topic_level="level1").sort_values(
        "topic_id", key=lambda col: col.map(
            level1_topics.set_index("topic_id")["display_order"]
        ), kind="stable"
    )
    level1_edges = _corpus_rows(bundle.edges, topic_level="level1").merge(
        presidents[["president_profile_id", "display_order"]],
        on="president_profile_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_president"),
    ).merge(
        level1_topics[["topic_id", "display_order"]],
        on="topic_id",
        how="left",
        validate="many_to_one",
        suffixes=("_president", "_topic"),
    ).sort_values(["display_order_topic", "display_order_president"], kind="stable")

    if len(presidents) != 45 or len(level1_topics) != 17 or len(level1_edges) != 700:
        raise SummaryTopicNetworkError("Summary projection population drift")
    support_counts = president_support["president_support_status"].value_counts().to_dict()
    if support_counts != {"supported": 42, "thin": 3}:
        raise SummaryTopicNetworkError("Summary president-support population drift")
    if int(level1_edges["default_visible"].sum()) != 427:
        raise SummaryTopicNetworkError("Summary default-visible edge population drift")

    shard_payloads: dict[str, dict[str, Any]] = {}
    shard_bytes: dict[str, bytes] = {}
    shard_receipts: list[dict[str, Any]] = []
    covered_children: set[str] = set()
    corpus_level2_edges = _corpus_rows(bundle.edges, topic_level="level2")

    for parent in level1_topics.itertuples(index=False):
        children = level2_topics.loc[level2_topics["parent_topic_id"].eq(parent.topic_id)].copy()
        child_ids = set(children["topic_id"].astype(str))
        if not child_ids or covered_children & child_ids:
            raise SummaryTopicNetworkError("Level 2 shard ancestry is missing or duplicated")
        covered_children.update(child_ids)
        edges = corpus_level2_edges.loc[corpus_level2_edges["topic_id"].isin(child_ids)].copy()
        edges = edges.merge(
            presidents[["president_profile_id", "display_order"]],
            on="president_profile_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_president"),
        ).merge(
            children[["topic_id", "display_order"]],
            on="topic_id",
            how="left",
            validate="many_to_one",
            suffixes=("_president", "_topic"),
        ).sort_values(["display_order_topic", "display_order_president"], kind="stable")
        parent_edge_ids = set(
            level1_edges.loc[level1_edges["topic_id"].eq(parent.topic_id), "edge_id"].astype(str)
        )
        edge_ids = parent_edge_ids | set(edges["edge_id"].astype(str))
        evidence = bundle.edge_evidence.loc[
            bundle.edge_evidence["edge_id"].astype(str).isin(edge_ids)
        ].sort_values(["edge_id", "selection_position", "receipt_id"], kind="stable")
        child_support = _corpus_rows(bundle.topic_support, topic_level="level2").loc[
            lambda rows: rows["topic_id"].isin(child_ids)
        ].sort_values(
            "topic_id", key=lambda col: col.map(
                children.set_index("topic_id")["display_order"]
            ), kind="stable"
        )
        parent_record = _records(
            level1_topics.loc[level1_topics["topic_id"].eq(parent.topic_id)], TOPIC_FIELDS
        )[0]
        shard = {
            "schema_version": TOPIC_SCHEMA,
            "source_identity": source_identity,
            "scope": {"scope_type": SCOPE_TYPE, "scope_id": SCOPE_ID},
            "parent_topic": parent_record,
            "child_topics": _records(children, TOPIC_FIELDS),
            "topic_support": _records(child_support, TOPIC_SUPPORT_FIELDS),
            "edges": _records(edges, EDGE_FIELDS),
            "evidence": _records(evidence, EVIDENCE_FIELDS),
            "counts": {
                "child_topics": len(children),
                "level2_edges": len(edges),
                "evidence_receipts": len(evidence),
                "parent_level1_evidence_receipts": int(
                    evidence["edge_id"].astype(str).isin(parent_edge_ids).sum()
                ),
            },
        }
        shard["shard_sha256"] = _self_hash(shard, "shard_sha256")
        relative = f"topics/{parent.topic_id}_v1.json"
        value = _json_bytes(shard)
        if len(value) > SHARD_RAW_MAX or _gzip_size(value) > SHARD_GZIP_MAX:
            raise SummaryTopicNetworkError(f"topic shard exceeds its byte budget: {relative}")
        shard_payloads[relative] = shard
        shard_bytes[relative] = value
        receipt = _file_receipt(value, TOPIC_SCHEMA, len(children) + len(edges) + len(evidence))
        shard_receipts.append(
            {
                "parent_topic_id": parent.topic_id,
                "parent_topic_label": parent.topic_label,
                "filename": relative,
                "child_topic_count": len(children),
                "edge_count": len(edges),
                "evidence_receipt_count": len(evidence),
                **receipt,
            }
        )

    if covered_children != set(level2_topics["topic_id"].astype(str)) or len(covered_children) != 50:
        raise SummaryTopicNetworkError("seventeen topic shards do not cover all 50 Level 2 nodes")

    index = {
        "schema_version": INDEX_SCHEMA,
        "source_identity": source_identity,
        "question": "Which broad topics recur in the paragraphs actually attributed to each president?",
        "scope": {
            "scope_type": SCOPE_TYPE,
            "scope_id": SCOPE_ID,
            "scope_label": "All eligible corpus",
            "topic_level": "level1",
        },
        "edge_width_measure": "speaker_paragraph_share",
        "metrics": bundle.meta["metrics"],
        "policies": {
            "topic_free_denominator_policy": bundle.meta["source_configuration"][
                "topic_free_denominator_policy"
            ],
            "multi_label_counting_policy": bundle.meta["source_configuration"][
                "multi_label_counting_policy"
            ],
            "support_thresholds": bundle.meta["source_configuration"]["support_thresholds"],
            "evidence_selection_policy": bundle.meta["source_configuration"][
                "evidence_selection_policy"
            ],
            "interpretation_policy": bundle.meta["source_configuration"][
                "interpretation_policy"
            ],
        },
        "presidents": _records(presidents, PRESIDENT_FIELDS),
        "level1_topics": _records(level1_topics, TOPIC_FIELDS),
        "president_support": _records(president_support, PRESIDENT_SUPPORT_FIELDS),
        "topic_support": _records(topic_support, TOPIC_SUPPORT_FIELDS),
        "edges": _records(level1_edges, EDGE_FIELDS),
        "topic_shards": shard_receipts,
        "counts": {
            "eligible_actual_president_paragraphs": 32_531,
            "presidents": len(presidents),
            "supported_presidents": support_counts["supported"],
            "thin_presidents": support_counts["thin"],
            "level1_topics": len(level1_topics),
            "observed_level1_edges": len(level1_edges),
            "supported_level1_edges": int(
                level1_edges["edge_support_status"].eq("supported").sum()
            ),
            "default_visible_level1_edges": int(level1_edges["default_visible"].sum()),
        },
    }
    if index["counts"]["supported_level1_edges"] != 595:
        raise SummaryTopicNetworkError("Summary supported-edge population drift")
    index["index_sha256"] = _self_hash(index, "index_sha256")
    index_bytes = _json_bytes(index)
    if len(index_bytes) > INDEX_RAW_MAX or _gzip_size(index_bytes) > INDEX_GZIP_MAX:
        raise SummaryTopicNetworkError("Summary index exceeds its byte budget")

    recurring_table = _recurring_table_bytes(bundle)
    if len(_recurring_table_frame(bundle)) != 427:
        raise SummaryTopicNetworkError("recurring-topic download population drift")
    files = {
        INDEX_FILE: index_bytes,
        RECURRING_TABLE_FILE: recurring_table,
        **shard_bytes,
    }
    file_receipts = {
        INDEX_FILE: _file_receipt(index_bytes, INDEX_SCHEMA, len(level1_edges)),
        RECURRING_TABLE_FILE: _file_receipt(
            recurring_table, speaker_topic_network.EDGE_SCHEMA, 427
        ),
        **{
            path: _file_receipt(
                value,
                TOPIC_SCHEMA,
                shard_payloads[path]["counts"]["child_topics"]
                + shard_payloads[path]["counts"]["level2_edges"]
                + shard_payloads[path]["counts"]["evidence_receipts"],
            )
            for path, value in sorted(shard_bytes.items())
        },
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_identity": source_identity,
        "files": file_receipts,
        "counts": {
            "files_excluding_manifest": len(files),
            "topic_shards": len(shard_payloads),
            "level1_edges": len(level1_edges),
            "level2_edges": len(corpus_level2_edges),
            "level2_topics": len(level2_topics),
        },
        "budgets": {
            "index_raw_max": INDEX_RAW_MAX,
            "index_gzip_max": INDEX_GZIP_MAX,
            "topic_shard_raw_max": SHARD_RAW_MAX,
            "topic_shard_gzip_max": SHARD_GZIP_MAX,
            "renderer_js_raw_max": RENDERER_JS_RAW_MAX,
        },
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    manifest_bytes = _json_bytes(manifest)
    files[MANIFEST_FILE] = manifest_bytes
    projection = SummaryTopicNetworkProjection(index, shard_payloads, manifest, files)
    validate_projection(projection, bundle)
    return projection


def _assert_records_equal(
    projected: Iterable[Mapping[str, Any]],
    source: pd.DataFrame,
    fields: Sequence[str],
    key_fields: Sequence[str],
    label: str,
) -> None:
    expected = _records(source, fields)
    expected_map = {tuple(row[key] for key in key_fields): row for row in expected}
    actual = list(projected)
    actual_map = {tuple(row.get(key) for key in key_fields): dict(row) for row in actual}
    if len(actual_map) != len(actual) or actual_map != expected_map:
        raise SummaryTopicNetworkError(f"projected {label} rows or values differ from Plan 3")


def validate_projection(
    projection: SummaryTopicNetworkProjection,
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> None:
    """Compare every projected row/value and file receipt to Plan 3."""
    speaker_topic_network.validate_bundle(bundle)
    index = projection.index
    if index.get("schema_version") != INDEX_SCHEMA:
        raise SummaryTopicNetworkError("Summary index schema drift")
    if index.get("index_sha256") != _self_hash(index, "index_sha256"):
        raise SummaryTopicNetworkError("Summary index self-hash drift")
    if index.get("source_identity") != _source_identity(bundle):
        raise SummaryTopicNetworkError("Summary projection is stale against Plan 3 hashes")
    if index.get("scope") != {
        "scope_type": SCOPE_TYPE,
        "scope_id": SCOPE_ID,
        "scope_label": "All eligible corpus",
        "topic_level": "level1",
    }:
        raise SummaryTopicNetworkError("Summary projection scope drift")
    if index.get("edge_width_measure") != "speaker_paragraph_share":
        raise SummaryTopicNetworkError("Summary edge-width metric drift")

    presidents = bundle.president_nodes.sort_values("display_order", kind="stable")
    level1_topics = bundle.topic_nodes.loc[
        bundle.topic_nodes["topic_level"].eq("level1")
    ].sort_values("display_order", kind="stable")
    _assert_records_equal(
        index.get("presidents", []), presidents, PRESIDENT_FIELDS,
        ("president_profile_id",), "president",
    )
    _assert_records_equal(
        index.get("level1_topics", []), level1_topics, TOPIC_FIELDS,
        ("topic_id",), "Level 1 topic",
    )
    _assert_records_equal(
        index.get("president_support", []), _corpus_rows(bundle.president_support),
        PRESIDENT_SUPPORT_FIELDS, ("president_profile_id",), "president support",
    )
    _assert_records_equal(
        index.get("topic_support", []), _corpus_rows(bundle.topic_support, topic_level="level1"),
        TOPIC_SUPPORT_FIELDS, ("topic_id",), "Level 1 topic support",
    )
    _assert_records_equal(
        index.get("edges", []), _corpus_rows(bundle.edges, topic_level="level1"),
        EDGE_FIELDS, ("edge_id",), "Level 1 edge",
    )

    shard_index = {
        row["parent_topic_id"]: row for row in index.get("topic_shards", [])
    }
    if len(shard_index) != 17 or set(projection.shards) != {
        row["filename"] for row in shard_index.values()
    }:
        raise SummaryTopicNetworkError("Summary topic-shard index mismatch")
    covered_children: set[str] = set()
    for relative, shard in projection.shards.items():
        if shard.get("schema_version") != TOPIC_SCHEMA:
            raise SummaryTopicNetworkError(f"topic shard schema drift: {relative}")
        if shard.get("shard_sha256") != _self_hash(shard, "shard_sha256"):
            raise SummaryTopicNetworkError(f"topic shard self-hash drift: {relative}")
        if shard.get("source_identity") != index["source_identity"]:
            raise SummaryTopicNetworkError(f"topic shard source drift: {relative}")
        parent_id = shard.get("parent_topic", {}).get("topic_id")
        receipt = shard_index.get(parent_id)
        if receipt is None or receipt.get("filename") != relative:
            raise SummaryTopicNetworkError(f"topic shard ancestry drift: {relative}")
        source_children = bundle.topic_nodes.loc[
            bundle.topic_nodes["parent_topic_id"].eq(parent_id)
        ]
        child_ids = set(source_children["topic_id"].astype(str))
        if covered_children & child_ids:
            raise SummaryTopicNetworkError("Level 2 topic appears in more than one shard")
        covered_children.update(child_ids)
        _assert_records_equal(
            shard.get("child_topics", []), source_children, TOPIC_FIELDS,
            ("topic_id",), f"{parent_id} child topic",
        )
        source_topic_support = _corpus_rows(bundle.topic_support, topic_level="level2").loc[
            lambda rows: rows["topic_id"].isin(child_ids)
        ]
        _assert_records_equal(
            shard.get("topic_support", []), source_topic_support, TOPIC_SUPPORT_FIELDS,
            ("topic_id",), f"{parent_id} Level 2 topic support",
        )
        source_edges = _corpus_rows(bundle.edges, topic_level="level2").loc[
            lambda rows: rows["topic_id"].isin(child_ids)
        ]
        _assert_records_equal(
            shard.get("edges", []), source_edges, EDGE_FIELDS,
            ("edge_id",), f"{parent_id} Level 2 edge",
        )
        parent_edge_ids = set(
            _corpus_rows(bundle.edges, topic_level="level1").loc[
                lambda rows: rows["topic_id"].eq(parent_id), "edge_id"
            ].astype(str)
        )
        evidence_edge_ids = parent_edge_ids | set(source_edges["edge_id"].astype(str))
        source_evidence = bundle.edge_evidence.loc[
            bundle.edge_evidence["edge_id"].astype(str).isin(evidence_edge_ids)
        ]
        _assert_records_equal(
            shard.get("evidence", []), source_evidence, EVIDENCE_FIELDS,
            ("receipt_id",), f"{parent_id} evidence",
        )

    all_level2 = set(
        bundle.topic_nodes.loc[bundle.topic_nodes["topic_level"].eq("level2"), "topic_id"].astype(str)
    )
    if covered_children != all_level2:
        raise SummaryTopicNetworkError("Level 2 topic-shard coverage drift")

    if projection.manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SummaryTopicNetworkError("Summary manifest schema drift")
    if projection.manifest.get("manifest_sha256") != _self_hash(
        projection.manifest, "manifest_sha256"
    ):
        raise SummaryTopicNetworkError("Summary manifest self-hash drift")
    expected_paths = {INDEX_FILE, RECURRING_TABLE_FILE, *projection.shards}
    if set(projection.manifest.get("files", {})) != expected_paths:
        raise SummaryTopicNetworkError("Summary manifest inventory drift")
    if set(projection.files) != expected_paths | {MANIFEST_FILE}:
        raise SummaryTopicNetworkError("Summary in-memory file inventory drift")
    for relative, receipt in projection.manifest["files"].items():
        value = projection.files.get(relative)
        if value is None or len(value) != receipt.get("bytes") or _gzip_size(value) != receipt.get(
            "gzip_bytes"
        ) or _sha256_bytes(value) != receipt.get("sha256"):
            raise SummaryTopicNetworkError(f"Summary manifest/file mismatch: {relative}")
    if projection.files[MANIFEST_FILE] != _json_bytes(projection.manifest):
        raise SummaryTopicNetworkError("Summary manifest bytes differ from its payload")
    if projection.files[RECURRING_TABLE_FILE] != _recurring_table_bytes(bundle):
        raise SummaryTopicNetworkError(
            "recurring-topic download differs from the supported Plan 3 rows"
        )


def _write_candidate(projection: SummaryTopicNetworkProjection, output: Path) -> None:
    for relative, value in sorted(projection.files.items()):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, destination)


def write_public_projection(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    site_dir: Path = REPO_ROOT / "docs",
    *,
    projection: SummaryTopicNetworkProjection | None = None,
) -> list[Path]:
    """Validate in isolation, then publish the owned projection and CSV download."""
    projection = projection or build_projection(bundle)
    validate_projection(projection, bundle)
    site_root = Path(site_dir)
    output = site_root / "data" / PUBLIC_DIR_NAME
    with tempfile.TemporaryDirectory(prefix="summary-topic-network-") as raw:
        candidate_site = Path(raw) / "site"
        candidate_output = candidate_site / "data" / PUBLIC_DIR_NAME
        _write_candidate(projection, candidate_output)
        validate_publication(bundle, candidate_site)
        with tempfile.TemporaryDirectory(prefix="summary-topic-network-rollback-") as backup_raw:
            backup = Path(backup_raw) / PUBLIC_DIR_NAME
            existed = output.exists()
            if existed:
                shutil.copytree(output, backup)
            try:
                expected = set(projection.files)
                output.mkdir(parents=True, exist_ok=True)
                for stale in output.rglob("*"):
                    if stale.is_file() and stale.relative_to(output).as_posix() not in expected:
                        stale.unlink()
                for relative in sorted(expected, key=lambda item: item == MANIFEST_FILE):
                    _atomic_replace(candidate_output / relative, output / relative)
            except Exception:
                if output.exists():
                    shutil.rmtree(output)
                if existed:
                    shutil.copytree(backup, output)
                raise
    return [output / relative for relative in sorted(projection.files)]


def write_renderer_assets(site_dir: Path = REPO_ROOT / "docs") -> list[Path]:
    """Write the reusable page-agnostic renderer and scoped stylesheet."""
    js = (SUMMARY_TOPIC_NETWORK_JS.strip() + "\n").encode("utf-8")
    css = (SUMMARY_TOPIC_NETWORK_CSS.strip() + "\n").encode("utf-8")
    if len(js) > RENDERER_JS_RAW_MAX:
        raise SummaryTopicNetworkError("topic relationship renderer exceeds 50 KB raw")
    output = Path(site_dir) / "assets"
    output.mkdir(parents=True, exist_ok=True)
    values = {ASSET_JS_NAME: js, ASSET_CSS_NAME: css}
    with tempfile.TemporaryDirectory(prefix="summary-topic-assets-") as raw:
        candidate = Path(raw)
        for filename, value in values.items():
            (candidate / filename).write_bytes(value)
            _atomic_replace(candidate / filename, output / filename)
    return [output / filename for filename in values]


def _load_projection_from_site(site_dir: Path) -> SummaryTopicNetworkProjection:
    output = Path(site_dir) / "data" / PUBLIC_DIR_NAME
    manifest_path = output / MANIFEST_FILE
    if not manifest_path.is_file():
        raise SummaryTopicNetworkError("Summary topic-network manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SummaryTopicNetworkError("Summary topic-network manifest schema drift")
    expected = set(manifest.get("files", {}))
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual != expected | {MANIFEST_FILE}:
        raise SummaryTopicNetworkError("Summary topic-network disk inventory drift")
    files = {relative: (output / relative).read_bytes() for relative in expected}
    files[MANIFEST_FILE] = manifest_bytes
    index = json.loads(files[INDEX_FILE])
    shards = {
        relative: json.loads(files[relative])
        for relative in expected
        if relative.startswith("topics/")
    }
    return SummaryTopicNetworkProjection(index, shards, manifest, files)


def validate_publication(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    site_dir: Path = REPO_ROOT / "docs",
) -> dict[str, Any]:
    projection = _load_projection_from_site(Path(site_dir))
    validate_projection(projection, bundle)
    asset_dir = Path(site_dir) / "assets"
    expected_assets = {
        ASSET_JS_NAME: (SUMMARY_TOPIC_NETWORK_JS.strip() + "\n").encode("utf-8"),
        ASSET_CSS_NAME: (SUMMARY_TOPIC_NETWORK_CSS.strip() + "\n").encode("utf-8"),
    }
    # Asset validation is optional for isolated projection candidates and mandatory
    # for a real generated site once either companion asset is present.
    if any((asset_dir / name).exists() for name in expected_assets):
        for name, value in expected_assets.items():
            path = asset_dir / name
            if not path.is_file() or path.read_bytes() != value:
                raise SummaryTopicNetworkError(f"Summary renderer asset drift: {name}")
    return {
        "schema_version": INDEX_SCHEMA,
        "files": len(projection.files),
        "presidents": projection.index["counts"]["presidents"],
        "level1_topics": projection.index["counts"]["level1_topics"],
        "level1_edges": projection.index["counts"]["observed_level1_edges"],
        "topic_shards": projection.manifest["counts"]["topic_shards"],
    }


def verify_reproducible(bundle: speaker_topic_network.SpeakerTopicNetworkBundle) -> None:
    first = build_projection(bundle)
    second = build_projection(bundle)
    if first.files != second.files:
        raise SummaryTopicNetworkError("two Summary projection builds differ byte-for-byte")


def render_summary_section(projection: SummaryTopicNetworkProjection) -> str:
    """Render the server-side browser shell and exact CSV download."""
    index = projection.index
    topic_controls = "".join(
        '<button type="button" aria-pressed="false" data-summary-topic-choice="{id}" '
        'title="{definition}">{label}</button>'.format(
            id=html.escape(row["topic_id"], quote=True),
            label=html.escape(row["topic_label"]),
            definition=html.escape(row["topic_definition"], quote=True),
        )
        for row in index["level1_topics"]
    )
    president_support = {
        row["president_profile_id"]: row for row in index["president_support"]
    }
    president_options = "".join(
        '<option value="{id}">{name}</option>'.format(
            id=html.escape(row["president_profile_id"], quote=True),
            name=html.escape(row["president_name"]),
        )
        for row in index["presidents"]
        if president_support[row["president_profile_id"]][
            "president_support_status"
        ]
        == "supported"
    )
    rendered = f"""<section class="summary-chapter summary-topic-section" id="summary-topics"
  aria-labelledby="summary-topics-title" data-summary-topic-browser
  data-index-url="data/{PUBLIC_DIR_NAME}/{INDEX_FILE}"
  data-chart-key="summary_topic_relationships"
  data-edge-width-measure="speaker_paragraph_share"
  data-node-size-measure="selected_topic_paragraph_memberships">
  <p class="summary-chapter-no">05 · Topics</p>
  <h2 id="summary-topics-title">Presidents and their recurring topics</h2>
  <p class="summary-lede">Choose up to eight broad topics to see how presidents gather between
  them. Each president is pulled toward topics that recur more often in paragraphs actually
  attributed to that president. The view describes this
  speaker-audited corpus; it does not measure importance, intent, influence, or policy success.</p>
  <p class="summary-topic-population"><strong>32,531</strong> eligible actual-president paragraphs
  <span>·</span> <strong>45</strong> presidents <span>·</span> <strong>17</strong> broad topics</p>
  <div class="summary-topic-browser">
    <section class="summary-topic-controls" aria-labelledby="summary-topic-choose-title">
      <div class="summary-topic-control-heading">
        <h3 id="summary-topic-choose-title">Choose Topics</h3>
        <div class="summary-topic-selection-tools">
          <strong data-summary-topic-selection-count>0 of 8 selected</strong>
          <button type="button" data-summary-topic-clear disabled>Clear</button>
        </div>
      </div>
      <div class="summary-topic-picker" data-summary-topic-picker>{topic_controls}</div>
      <div class="summary-topic-president-filter">
        <div class="summary-topic-president-line">
          <label for="summary-topic-president-filter">Compare presidents</label>
          <select id="summary-topic-president-filter" data-summary-topic-president-filter>
            <option value="">Add a president…</option>{president_options}
          </select>
        </div>
        <div class="summary-topic-president-row" data-summary-topic-president-row hidden>
          <button type="button" data-summary-topic-president-clear>Show all presidents</button>
          <div class="summary-topic-president-chips" data-summary-topic-president-chips></div>
        </div>
      </div>
      <p class="summary-topic-status" role="status" aria-live="polite"
         data-summary-topic-status></p>
    </section>
    <figure class="summary-topic-stage">
      <div class="summary-topic-stage-tools">
        <p data-summary-topic-stage-readout>Hover, focus, or select a president to inspect it here.</p>
        <p data-summary-topic-size-key><strong>Portrait area</strong> = selected-topic paragraph
        memberships. One paragraph can count under more than one topic.</p>
      </div>
      <svg viewBox="0 0 900 610" role="img" aria-labelledby="summary-topic-svg-title summary-topic-svg-desc"
           data-summary-topic-svg>
        <title id="summary-topic-svg-title">President topic-gravity field</title>
        <desc id="summary-topic-svg-desc">No topics are selected. Choose up to eight broad topics.</desc>
      </svg>
      <figcaption>Topics stay fixed around the edge. A president begins at its share-weighted
      center; deterministic display stretch and collision spacing separate nearby portraits.
      Absolute distance has no separate meaning. Bubble area shows selected-topic paragraph
      memberships, which overlap because topics are multi-label.</figcaption>
      <noscript><p>The topic-gravity field requires JavaScript. Download the exact recurring-topic
      table below for the supported relationship values.</p></noscript>
    </figure>
  </div>
  <p class="summary-topic-download"><a href="data/{PUBLIC_DIR_NAME}/{RECURRING_TABLE_FILE}"
  download>Download the exact recurring-topic table</a></p>
</section>"""
    return re.sub(r">\s+<", "><", rendered).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Summary topic-network projection")
    parser.add_argument("--check", action="store_true", help="validate generated projection parity")
    parser.add_argument(
        "--verify-reproducible", action="store_true",
        help="build twice in isolation and compare every byte",
    )
    parser.add_argument("--site-dir", type=Path, default=REPO_ROOT / "docs")
    args = parser.parse_args()
    bundle = speaker_topic_network.load_network_bundle()
    if args.verify_reproducible:
        verify_reproducible(bundle)
        print("Summary topic-network projection is byte-reproducible across two builds")
        return
    if args.check:
        receipt = validate_publication(bundle, args.site_dir)
    else:
        projection = build_projection(bundle)
        write_public_projection(bundle, args.site_dir, projection=projection)
        write_renderer_assets(args.site_dir)
        receipt = validate_publication(bundle, args.site_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
