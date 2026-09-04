"""Deterministic public projection for the Compare V3 agenda and evidence views.

The projection copies accepted actual-speaker topic values from the governed
Plan 3 bundle, projects the existing source-document CorEx issue shares, and
packages existing evidence artifacts.  It does not derive speakers, topics,
denominators, support thresholds, taxonomies, cross-method scores, or rankings.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import (
    ai_labels,
    corpus,
    issues_site,
    profiles,
    similarity,
    speaker_topic_network,
    topic_quality,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR_NAME = "compare"
PUBLIC_DIR = REPO_ROOT / "docs" / "data" / PUBLIC_DIR_NAME

AGENDA_INDEX_SCHEMA = "compare-agenda-index-v1"
AGENDA_TOPIC_SCHEMA = "compare-agenda-topic-v1"
EVIDENCE_INDEX_SCHEMA = "compare-evidence-index-v1"
MANIFEST_SCHEMA = "compare-publication-manifest-v1"

AGENDA_INDEX_FILE = "agenda_index_v1.json"
EVIDENCE_INDEX_FILE = "evidence_index_v1.json"
AI_VALUES_FILE = "ai_topic_values_v1.csv"
LEGACY_VALUES_FILE = "legacy_issue_values_v1.csv"
MANIFEST_FILE = "manifest_v1.json"

SCOPE_TYPE = "corpus"
SCOPE_ID = "all-corpus"
EXPECTED_PRESIDENTS = 45
EXPECTED_LEVEL1 = 17
EXPECTED_LEVEL2 = 50
EXPECTED_LEGACY = 16
EXPECTED_AI_ROWS = EXPECTED_PRESIDENTS * (EXPECTED_LEVEL1 + EXPECTED_LEVEL2)
EXPECTED_LEGACY_ROWS = EXPECTED_PRESIDENTS * EXPECTED_LEGACY

AGENDA_INDEX_RAW_MAX = 500_000
AGENDA_INDEX_GZIP_MAX = 100_000
TOPIC_SHARD_RAW_MAX = 1_200_000
TOPIC_SHARD_GZIP_MAX = 275_000
EVIDENCE_INDEX_RAW_MAX = 750_000
EVIDENCE_INDEX_GZIP_MAX = 175_000
AI_CSV_RAW_MAX = 1_000_000
AI_CSV_GZIP_MAX = 200_000
LEGACY_CSV_RAW_MAX = 200_000
LEGACY_CSV_GZIP_MAX = 50_000

KNOWN_UNMAPPED_LEVEL2 = (
    "Constitutional Union & Federalism",
    "Civil Service Reform & the Merit System",
    "Territorial Organization, Statehood & Insular Governance",
    "Presidential Humility & Reflection on Office",
    "Personal Narrative, Storytelling & Boasting",
    "Executive Departments, Postal & Administrative Housekeeping",
    "Partisan Combat, Press Conferences & Media Attacks",
)


class CompareProjectionError(RuntimeError):
    """The Compare projection failed closed."""


@dataclass(frozen=True)
class CompareProjectionBundle:
    agenda_index: dict[str, Any]
    topic_shards: dict[str, dict[str, Any]]
    evidence_index: dict[str, Any]
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


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise CompareProjectionError("projection contains a non-finite value")
    return value


def _file_receipt(value: bytes, schema: str, rows: int) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "rows": int(rows),
        "bytes": len(value),
        "gzip_bytes": _gzip_size(value),
        "sha256": _sha256_bytes(value),
    }


def _records(frame: pd.DataFrame, fields: Sequence[str]) -> list[dict[str, Any]]:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise CompareProjectionError(f"source rows lack fields: {sorted(missing)}")
    return [
        {field: _scalar(item) for field, item in zip(fields, row, strict=True)}
        for row in frame.loc[:, list(fields)].itertuples(index=False, name=None)
    ]


def _corpus_rows(frame: pd.DataFrame, *, level: str | None = None) -> pd.DataFrame:
    rows = frame.loc[
        frame["scope_type"].eq(SCOPE_TYPE) & frame["scope_id"].eq(SCOPE_ID)
    ].copy()
    if level is not None:
        rows = rows.loc[rows["topic_level"].eq(level)].copy()
    return rows


def _topic_filename(topic: Mapping[str, Any]) -> str:
    return f"agenda_{profiles.slug(str(topic['topic_label']))}_v1.json"


def _source_identity(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> dict[str, Any]:
    artifacts = bundle.meta.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CompareProjectionError("Plan 3 metadata lacks artifact hashes")
    inputs = {
        "crosswalk_v1.json": _sha256_file(ai_labels.CROSSWALK_PATH),
        "issues_president.parquet": _sha256_file(corpus.DATA_DIR / "issues_president.parquet"),
        "paragraph_entities.parquet": _sha256_file(ai_labels.ENTITY_PATH),
        "invocation_candidates.parquet": _sha256_file(
            corpus.DATA_DIR / "invocations_v2" / "candidates.parquet"
        ),
        "invocation_classifications.parquet": _sha256_file(
            corpus.DATA_DIR / "invocations_v2" / "classifications.parquet"
        ),
        "president_distinctive.parquet": _sha256_file(
            corpus.DATA_DIR / "president_distinctive.parquet"
        ),
        "speech_embeddings.parquet": _sha256_file(similarity.SPEECH_EMB_PATH),
    }
    if any(value is None for value in inputs.values()):
        missing = sorted(key for key, value in inputs.items() if value is None)
        raise CompareProjectionError(f"Compare evidence inputs are missing: {missing}")
    return {
        "speaker_topic_contract": speaker_topic_network.CONTRACT_VERSION,
        "speaker_topic_metadata_sha256": bundle.meta.get("metadata_sha256"),
        "speaker_topic_artifacts": {
            name: receipt.get("sha256")
            for name, receipt in sorted(artifacts.items())
        },
        "source_files": inputs,
    }


def _catalogs(bundle: speaker_topic_network.SpeakerTopicNetworkBundle) -> tuple[list, list, list]:
    presidents = bundle.president_nodes.sort_values("display_order", kind="stable")
    topics = bundle.topic_nodes.sort_values("display_order", kind="stable")
    level1 = topics.loc[topics["topic_level"].eq("level1")]
    level2 = topics.loc[topics["topic_level"].eq("level2")]
    if (len(presidents), len(level1), len(level2)) != (
        EXPECTED_PRESIDENTS,
        EXPECTED_LEVEL1,
        EXPECTED_LEVEL2,
    ):
        raise CompareProjectionError("governed president/topic catalogs changed")
    president_records = _records(
        presidents,
        ("president_profile_id", "president_name", "party", "display_order"),
    )
    topic_fields = (
        "topic_id",
        "topic_level",
        "topic_label",
        "topic_definition",
        "topic_kind",
        "parent_topic_id",
        "parent_topic_label",
        "display_order",
    )
    return president_records, _records(level1, topic_fields), _records(level2, topic_fields)


def _president_support(bundle: speaker_topic_network.SpeakerTopicNetworkBundle) -> list[dict]:
    rows = _corpus_rows(bundle.president_support).copy()
    order = bundle.president_nodes.set_index("president_profile_id")["display_order"]
    rows["_order"] = rows["president_profile_id"].map(order)
    rows = rows.sort_values("_order", kind="stable")
    if len(rows) != EXPECTED_PRESIDENTS or rows["president_profile_id"].duplicated().any():
        raise CompareProjectionError("all-corpus president support must be complete and unique")
    return _records(
        rows,
        (
            "president_profile_id",
            "president_name",
            "eligible_president_paragraph_count",
            "eligible_president_appearance_count",
            "president_support_status",
            "topic_free_denominator_policy",
        ),
    )


def _edge_state(edge: Mapping[str, Any] | None, support: Mapping[str, Any]) -> str:
    president_thin = support["president_support_status"] != "supported"
    if edge is None:
        return "thin_zero" if president_thin else "supported_zero"
    if president_thin:
        return "thin_president"
    if edge["edge_support_status"] != "supported":
        return "thin_edge"
    return "supported"


def _dense_ai_cells(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    topics: Sequence[Mapping[str, Any]],
    supports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    all_edges = _corpus_rows(bundle.edges)
    edge_map = {
        (row.president_profile_id, row.topic_id): row._asdict()
        for row in all_edges.itertuples(index=False)
    }
    if len(edge_map) != len(all_edges):
        raise CompareProjectionError("Plan 3 all-corpus edges are not uniquely keyed")
    cells: list[dict[str, Any]] = []
    for topic in topics:
        for support in supports:
            key = (support["president_profile_id"], topic["topic_id"])
            edge = edge_map.get(key)
            state = _edge_state(edge, support)
            if edge is None:
                cell = {
                    "president_profile_id": key[0],
                    "topic_id": key[1],
                    "topic_level": topic["topic_level"],
                    "state": state,
                    "topic_paragraph_count": 0,
                    "topic_appearance_count": 0,
                    "eligible_president_paragraph_count": int(
                        support["eligible_president_paragraph_count"]
                    ),
                    "eligible_president_appearance_count": int(
                        support["eligible_president_appearance_count"]
                    ),
                    "speaker_paragraph_share": 0.0,
                    "president_support_status": support["president_support_status"],
                    "edge_support_status": "thin",
                    "evidence_receipt_count": 0,
                }
            else:
                cell = {
                    field: _scalar(edge[field])
                    for field in (
                        "president_profile_id",
                        "topic_id",
                        "topic_level",
                        "topic_paragraph_count",
                        "topic_appearance_count",
                        "eligible_president_paragraph_count",
                        "eligible_president_appearance_count",
                        "speaker_paragraph_share",
                        "president_support_status",
                        "edge_support_status",
                        "evidence_receipt_count",
                    )
                }
                cell["state"] = state
            cells.append(cell)
    return cells


def _public_ai_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Compact browser cell; president-level support facts live in one keyed catalog."""
    return {
        key: cell[key]
        for key in (
            "president_profile_id",
            "topic_id",
            "state",
            "topic_paragraph_count",
            "topic_appearance_count",
            "eligible_president_paragraph_count",
            "speaker_paragraph_share",
        )
    }


def _crosswalk(
    level2: Sequence[Mapping[str, Any]], legacy_issues: Sequence[str]
) -> tuple[list[dict[str, str]], list[str]]:
    raw = json.loads(ai_labels.CROSSWALK_PATH.read_text(encoding="utf-8"))
    topic_by_label = {str(row["topic_label"]): str(row["topic_id"]) for row in level2}
    issue_set = set(legacy_issues)
    mappings: list[dict[str, str]] = []
    mapped: set[str] = set()
    for row in raw.get("mappings", []):
        issue = str(row.get("legacy_issue", ""))
        if issue not in issue_set:
            raise CompareProjectionError(f"crosswalk has unknown legacy issue: {issue}")
        for label in row.get("level2_topics", []):
            if label not in topic_by_label:
                raise CompareProjectionError(f"crosswalk has unknown Level 2 topic: {label}")
            mappings.append({"topic_id": topic_by_label[label], "legacy_issue": issue})
            mapped.add(label)
    order = {name: index for index, name in enumerate(legacy_issues)}
    topic_order = {str(row["topic_id"]): index for index, row in enumerate(level2)}
    mappings.sort(key=lambda item: (topic_order[item["topic_id"]], order[item["legacy_issue"]]))
    unmapped = [str(row["topic_label"]) for row in level2 if row["topic_label"] not in mapped]
    if tuple(unmapped) != KNOWN_UNMAPPED_LEVEL2:
        raise CompareProjectionError(f"unexpected unmapped Level 2 topics: {unmapped}")
    return mappings, unmapped


def _legacy_catalog_and_cells(
    profile_data: Mapping[str, Any],
    profile_views: Mapping[str, Mapping[str, Any]],
    display_issues: Sequence[str],
    presidents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(display_issues) != EXPECTED_LEGACY or len(set(display_issues)) != len(display_issues):
        raise CompareProjectionError("legacy issue catalog must contain 16 unique issues")
    issues = profile_data["issues"]
    catalog = [
        {
            "issue_id": issues_site.issue_slug(topic_quality.display_name(issue)),
            "issue_label": topic_quality.display_name(issue),
            "source_issue_key": issue,
            "display_order": index,
            "issue_url": f"issues/{issues_site.issue_slug(topic_quality.display_name(issue))}.html",
        }
        for index, issue in enumerate(display_issues)
    ]
    cells: list[dict[str, Any]] = []
    for issue_row in catalog:
        issue = issue_row["source_issue_key"]
        share_column = f"share_{issue}"
        if share_column not in issues.columns:
            raise CompareProjectionError(f"legacy source lacks {share_column}")
        for president in presidents:
            name = str(president["president_name"])
            if name not in issues.index or name not in profile_views:
                cells.append({
                    "president_profile_id": president["president_profile_id"],
                    "issue_id": issue_row["issue_id"],
                    "state": "unavailable",
                    "positive_paragraph_count": None,
                    "source_document_paragraph_count": None,
                    "source_document_speech_count": None,
                    "paragraph_share": None,
                })
                continue
            row = issues.loc[name]
            value = _scalar(row[share_column])
            denominator = int(row["n_paragraphs"])
            speech_count = int(profile_views[name]["sample"]["n_speeches"])
            if value is None or denominator <= 0:
                state = "unavailable"
                numerator = None
                share = None
            else:
                share = float(value)
                numerator = int(round(share * denominator))
                if abs(numerator / denominator - share) > 1e-12:
                    raise CompareProjectionError("legacy share does not reconcile to its paragraph count")
                thin = speech_count < 5
                state = "thin_zero" if thin and numerator == 0 else (
                    "thin_president" if thin else ("supported_zero" if numerator == 0 else "supported")
                )
            cells.append({
                "president_profile_id": president["president_profile_id"],
                "issue_id": issue_row["issue_id"],
                "state": state,
                "positive_paragraph_count": numerator,
                "source_document_paragraph_count": denominator,
                "source_document_speech_count": speech_count,
                "paragraph_share": share,
            })
    return catalog, cells


def _receipt_records(
    bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    topic_ids: set[str],
) -> list[dict[str, Any]]:
    rows = _corpus_rows(bundle.edge_evidence)
    rows = rows.loc[rows["topic_id"].isin(topic_ids)].copy()
    fields = (
        "receipt_id",
        "edge_id",
        "president_profile_id",
        "topic_id",
        "doc_name",
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
    return _records(rows.sort_values(["topic_id", "president_profile_id", "receipt_id"]), fields)


def _sample_positions(size: int) -> list[int]:
    if size <= 0:
        return []
    if size <= 3:
        return list(range(size))
    return [0, (size - 1) // 2, size - 1]


def _evidence_excerpt(value: str, limit: int = 420) -> str:
    """Bound receipt payload without changing which supporting paragraph is cited."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    head = text[: limit + 1].rsplit(" ", 1)[0]
    return head + "…"


def _evidence_receipts(rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = rows.sort_values(["date", "doc_name", "para_idx"], kind="stable").reset_index(drop=True)
    receipts = []
    for index in _sample_positions(len(rows)):
        row = rows.iloc[index]
        receipts.append({
            "receipt_id": str(row["receipt_id"]),
            "doc_name": str(row["doc_name"]),
            "para_idx": int(row["para_idx"]),
            "speech_date": str(row["date"]),
            "speech_title": str(row["title"]),
            "source_url": profiles.miller_speech_url(str(row["doc_name"])),
            "excerpt": _evidence_excerpt(str(row["text"])),
        })
    return receipts


def _adversary_evidence(
    profile_data: Mapping[str, Any], owner_rows: pd.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    entities = pd.read_parquet(ai_labels.ENTITY_PATH)
    paragraphs = pd.read_parquet(corpus.DATA_DIR / "paragraphs.parquet")
    if paragraphs.duplicated(["doc_name", "para_idx"]).any():
        raise CompareProjectionError("paragraph receipt keys are not unique")
    rows = entities.loc[entities["stance"].eq("adversarial")].merge(
        owner_rows, on="doc_name", how="inner", validate="many_to_one"
    ).merge(paragraphs, on=["doc_name", "para_idx"], how="left", validate="many_to_one")
    if rows["text"].isna().any():
        raise CompareProjectionError("adversary receipt failed paragraph join")
    rows["receipt_id"] = rows.apply(
        lambda row: "adversary:" + hashlib.sha256(
            f"{row.doc_name}|{row.para_idx}|{row.entity}".encode()
        ).hexdigest()[:20], axis=1
    )
    rows["label"] = rows["entity"]
    by_president = profile_data["ai"]["by_president"]
    result: dict[str, list[dict[str, Any]]] = {}
    for president, info in by_president.items():
        groups = []
        for item in info.get("adversaries", [])[:3]:
            label = str(item["name"])
            matches = rows.loc[rows["president"].eq(president) & rows["entity"].eq(label)]
            if len(matches) != int(item["n"]):
                raise CompareProjectionError(f"adversary count drift for {president}: {label}")
            groups.append({
                "label": label,
                "mention_count": int(item["n"]),
                "receipts": _evidence_receipts(matches),
            })
        result[profiles.slug(president)] = groups
    return result


def _invocation_evidence(
    owner_rows: pd.DataFrame,
    profile_views: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidates = pd.read_parquet(corpus.DATA_DIR / "invocations_v2" / "candidates.parquet")
    labels = pd.read_parquet(corpus.DATA_DIR / "invocations_v2" / "classifications.parquet")
    rows = candidates.merge(labels, on="candidate_id", validate="one_to_one")
    rows = rows.loc[
        rows["excluded_reason"].eq("")
        & rows["speaker"].ne(rows["target"])
        & rows["target_status"].eq("former_president")
    ].copy()
    rows = rows.merge(
        owner_rows[["doc_name", "date", "title"]], on="doc_name", how="left", validate="many_to_one"
    )
    if rows[["date", "title"]].isna().any().any():
        raise CompareProjectionError("invocation receipt failed source-document join")
    rows["text"] = rows["context"]
    rows["receipt_id"] = "invocation:" + rows["candidate_id"].astype(str)
    rows["label"] = rows["target"]
    result: dict[str, list[dict[str, Any]]] = {}
    for president, view in profile_views.items():
        group = rows.loc[rows["speaker"].eq(president)]
        values = []
        for item in view.get("classified_invocations", [])[:6]:
            matches = group.loc[
                group["target"].eq(item["target"])
                & group["function"].eq(item["function"])
                & group["stance"].eq(item["stance"])
            ]
            if len(matches) != int(item["mentions"]):
                raise CompareProjectionError(
                    f"invocation-v2 count drift for {president}: {item['target']}"
                )
            values.append({
                "target": str(item["target"]),
                "function": str(item["function"]),
                "stance": str(item["stance"]),
                "mention_count": int(item["mentions"]),
                "evidence_status": "AI-classified; not human-validated",
                "receipts": _evidence_receipts(matches),
            })
        result[profiles.slug(president)] = values
    return result


def _signature_evidence(profile_data: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    speech_embeddings = pd.read_parquet(similarity.SPEECH_EMB_PATH)
    vector_columns = [column for column in speech_embeddings if column.startswith("e")]
    embedding_matrix = speech_embeddings[vector_columns].to_numpy()
    adjusted = profile_data["adj"]
    adjusted_vectors = {
        row["president"]: adjusted.loc[index, vector_columns].to_numpy(dtype=float)
        for index, row in adjusted.iterrows()
    }
    frame = profile_data["df"]
    merged = frame[["doc_name", "president", "title", "year"]].merge(
        speech_embeddings[["doc_name"]].assign(_row=range(len(speech_embeddings))),
        on="doc_name",
        validate="one_to_one",
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for president, group in merged.groupby("president", sort=False):
        scores = embedding_matrix[group["_row"].to_numpy()] @ adjusted_vectors[president]
        scored = group.assign(
            cosine_score=scores,
            source_url=lambda value: value["doc_name"].map(profiles.miller_speech_url),
        )
        score_by_url = dict(zip(scored["source_url"], scored["cosine_score"], strict=True))
        established = profile_data["signatures"].get(president, [])[:3]
        projected = []
        for rank, item in enumerate(established, start=1):
            source_url = profiles.miller_speech_url(str(item["url"]))
            if source_url not in score_by_url:
                raise CompareProjectionError(f"signature source missing for {president}: {source_url}")
            projected.append({
                "rank": rank,
                "title": str(item["title"]),
                "year": int(item["year"]),
                "cosine_score": float(score_by_url[source_url]),
                "source_url": source_url,
            })
        result[profiles.slug(str(president))] = projected
    return result


def _build_evidence_index(
    profile_data: Mapping[str, Any],
    profile_views: Mapping[str, Mapping[str, Any]],
    presidents: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    owner_rows = profile_data["df"][["doc_name", "president", "date", "title"]].copy()
    if owner_rows["doc_name"].duplicated().any():
        raise CompareProjectionError("source-document owner map is not unique")
    owner_rows["date"] = owner_rows["date"].astype(str)
    adversaries = _adversary_evidence(profile_data, owner_rows)
    invocations = _invocation_evidence(owner_rows, profile_views)
    signatures = _signature_evidence(profile_data)
    vocabulary: dict[str, list[dict[str, Any]]] = {}
    for president, group in profile_data["distinctive"].groupby("president", sort=False):
        ranked = group.sort_values(["rank", "term"], kind="stable").head(6)
        vocabulary[profiles.slug(str(president))] = _records(ranked, ("term", "z", "rank"))
    records = []
    support_by_id = {row["president_profile_id"]: row for row in presidents}
    for president_id in support_by_id:
        name = str(support_by_id[president_id]["president_name"])
        view = profile_views[name]
        signature_values = signatures.get(president_id, [])
        expected_signatures = [
            (item["title"], item["year"], item["url"])
            for item in view.get("signature_speeches", [])[:3]
        ]
        actual_signatures = [
            (item["title"], item["year"], item["source_url"])
            for item in signature_values
        ]
        if actual_signatures != expected_signatures:
            raise CompareProjectionError(f"signature ranking drift for {name}")
        records.append({
            "president_profile_id": president_id,
            "footprint": {
                "source_document_speech_count": int(view["sample"]["n_speeches"]),
                "source_document_paragraph_count": int(view["_view"]["n_paragraphs"]),
                "source_document_word_count": int(view["sample"]["n_words"]),
                "support_state": "thin" if view["sample"]["thin_record"] else "supported",
            },
            "adversarial_entities": adversaries.get(president_id, []),
            "presidential_invocations": invocations.get(president_id, []),
            "distinctive_vocabulary": vocabulary.get(president_id, []),
            "signature_speeches": signature_values,
        })
    value = {
        "schema_version": EVIDENCE_INDEX_SCHEMA,
        "population_contracts": {
            "footprint": "source-document corpus counts",
            "adversarial_entities": "source-document paragraph-level exploratory AI extraction; raw mentions",
            "presidential_invocations": "source-document invocation-v2 candidates and AI classifications; raw mentions",
            "distinctive_vocabulary": "source-document lexical aggregate versus all other president records",
            "signature_speeches": "source-document speech-level era-adjusted embedding alignment",
        },
        "source_identity": source_identity,
        "presidents": records,
    }
    value["projection_sha256"] = _self_hash(value, "projection_sha256")
    return value


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _validate_budget(filename: str, value: bytes) -> None:
    limits = {
        AGENDA_INDEX_FILE: (AGENDA_INDEX_RAW_MAX, AGENDA_INDEX_GZIP_MAX),
        EVIDENCE_INDEX_FILE: (EVIDENCE_INDEX_RAW_MAX, EVIDENCE_INDEX_GZIP_MAX),
        AI_VALUES_FILE: (AI_CSV_RAW_MAX, AI_CSV_GZIP_MAX),
        LEGACY_VALUES_FILE: (LEGACY_CSV_RAW_MAX, LEGACY_CSV_GZIP_MAX),
    }
    raw_max, gzip_max = limits.get(filename, (TOPIC_SHARD_RAW_MAX, TOPIC_SHARD_GZIP_MAX))
    if len(value) > raw_max or _gzip_size(value) > gzip_max:
        raise CompareProjectionError(
            f"{filename} exceeds budget: {len(value)} raw / {_gzip_size(value)} gzip"
        )


def build_projection(
    network_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    profile_data: Mapping[str, Any],
    profile_views: Mapping[str, Mapping[str, Any]],
    display_issues: Sequence[str],
) -> CompareProjectionBundle:
    """Build and validate the complete 22-file Compare projection in memory."""
    speaker_topic_network.validate_bundle(network_bundle)
    presidents, level1, level2 = _catalogs(network_bundle)
    support = _president_support(network_bundle)
    source_identity = _source_identity(network_bundle)
    legacy_catalog, legacy_cells = _legacy_catalog_and_cells(
        profile_data, profile_views, display_issues, presidents
    )
    mappings, unmapped = _crosswalk(
        level2, [row["issue_label"] for row in legacy_catalog]
    )
    broad_cells_full = _dense_ai_cells(network_bundle, level1, support)
    all_fine_cells_full = _dense_ai_cells(network_bundle, level2, support)
    max_percent = max(
        float(cell["speaker_paragraph_share"] or 0) * 100
        for cell in all_fine_cells_full
    )
    fine_ceiling = min(100, max(10, int(math.ceil(max_percent / 10.0) * 10)))

    topic_files = {topic["topic_id"]: _topic_filename(topic) for topic in level1}
    agenda_index = {
        "schema_version": AGENDA_INDEX_SCHEMA,
        "source_identity": source_identity,
        "population_contracts": {
            "ai_topics": "eligible actual-speaker paragraphs; topic-free paragraphs remain in the denominator; multi-label shares are non-additive",
            "legacy_issues": "source-document-owner paragraphs; CorEx issue shares are methodologically separate from AI topics",
            "method_bridge": "declared hierarchy and crosswalk navigation only; no combined measure",
        },
        "presidents": presidents,
        "president_support": support,
        "level1_topics": [dict(topic, shard_url=f"data/compare/{topic_files[topic['topic_id']]}") for topic in level1],
        "legacy_issues": legacy_catalog,
        "broad_cells": [_public_ai_cell(cell) for cell in broad_cells_full],
        "legacy_cells": legacy_cells,
        "crosswalk_relationships": mappings,
        "unmapped_level2_labels": unmapped,
        "display_scales": {"broad_percent_max": 100, "fine_percent_max": fine_ceiling, "legacy_percent_max": 100},
        "downloads": {
            "ai_topic_values": f"data/compare/{AI_VALUES_FILE}",
            "legacy_issue_values": f"data/compare/{LEGACY_VALUES_FILE}",
        },
    }
    agenda_index["projection_sha256"] = _self_hash(agenda_index, "projection_sha256")

    topic_shards: dict[str, dict[str, Any]] = {}
    for parent in level1:
        children = [topic for topic in level2 if topic["parent_topic_id"] == parent["topic_id"]]
        child_ids = {str(topic["topic_id"]) for topic in children}
        ids = child_ids | {str(parent["topic_id"])}
        cells = [
            _public_ai_cell(cell)
            for cell in all_fine_cells_full
            if cell["topic_id"] in child_ids
        ]
        shard = {
            "schema_version": AGENDA_TOPIC_SCHEMA,
            "parent_topic": parent,
            "topics": children,
            "cells": cells,
            "evidence_receipts": _receipt_records(network_bundle, ids),
            "display_percent_max": fine_ceiling,
            "population_contract": agenda_index["population_contracts"]["ai_topics"],
        }
        shard["projection_sha256"] = _self_hash(shard, "projection_sha256")
        topic_shards[topic_files[parent["topic_id"]]] = shard

    evidence_index = _build_evidence_index(
        profile_data, profile_views, presidents, source_identity
    )
    ai_rows = broad_cells_full + all_fine_cells_full
    ai_fields = (
        "president_profile_id", "topic_id", "topic_level", "state",
        "topic_paragraph_count", "topic_appearance_count",
        "eligible_president_paragraph_count", "eligible_president_appearance_count",
        "speaker_paragraph_share", "president_support_status", "edge_support_status",
        "evidence_receipt_count",
    )
    legacy_fields = (
        "president_profile_id", "issue_id", "state", "positive_paragraph_count",
        "source_document_paragraph_count", "source_document_speech_count", "paragraph_share",
    )

    files: dict[str, bytes] = {
        AGENDA_INDEX_FILE: _json_bytes(agenda_index),
        EVIDENCE_INDEX_FILE: _json_bytes(evidence_index),
        AI_VALUES_FILE: _csv_bytes(ai_rows, ai_fields),
        LEGACY_VALUES_FILE: _csv_bytes(legacy_cells, legacy_fields),
    }
    files.update({filename: _json_bytes(shard) for filename, shard in topic_shards.items()})
    for filename, value in files.items():
        _validate_budget(filename, value)
    schemas = {
        AGENDA_INDEX_FILE: AGENDA_INDEX_SCHEMA,
        EVIDENCE_INDEX_FILE: EVIDENCE_INDEX_SCHEMA,
        AI_VALUES_FILE: "compare-ai-topic-values-v1",
        LEGACY_VALUES_FILE: "compare-legacy-issue-values-v1",
        **{filename: AGENDA_TOPIC_SCHEMA for filename in topic_shards},
    }
    row_counts = {
        AGENDA_INDEX_FILE: len(broad_cells_full) + len(legacy_cells),
        EVIDENCE_INDEX_FILE: len(evidence_index["presidents"]),
        AI_VALUES_FILE: len(ai_rows),
        LEGACY_VALUES_FILE: len(legacy_cells),
        **{filename: len(shard["cells"]) for filename, shard in topic_shards.items()},
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_identity": source_identity,
        "inventory": {
            filename: _file_receipt(files[filename], schemas[filename], row_counts[filename])
            for filename in sorted(files)
        },
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    files[MANIFEST_FILE] = _json_bytes(manifest)
    bundle = CompareProjectionBundle(agenda_index, topic_shards, evidence_index, manifest, files)
    validate_projection(bundle, network_bundle=network_bundle)
    return bundle


def validate_projection(
    projection: CompareProjectionBundle,
    *,
    network_bundle: speaker_topic_network.SpeakerTopicNetworkBundle | None = None,
) -> None:
    """Fail closed on schema, cardinality, determinism, provenance, and parity drift."""
    expected_files = {
        AGENDA_INDEX_FILE, EVIDENCE_INDEX_FILE, AI_VALUES_FILE,
        LEGACY_VALUES_FILE, MANIFEST_FILE, *projection.topic_shards.keys(),
    }
    if len(projection.topic_shards) != EXPECTED_LEVEL1 or len(expected_files) != 22:
        raise CompareProjectionError("Compare publication must contain exactly 22 files")
    if set(projection.files) != expected_files:
        raise CompareProjectionError("Compare publication file inventory drifted")
    if projection.agenda_index.get("schema_version") != AGENDA_INDEX_SCHEMA:
        raise CompareProjectionError("invalid agenda index schema")
    if projection.evidence_index.get("schema_version") != EVIDENCE_INDEX_SCHEMA:
        raise CompareProjectionError("invalid evidence index schema")
    if projection.manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise CompareProjectionError("invalid manifest schema")
    if len(projection.agenda_index["broad_cells"]) != EXPECTED_PRESIDENTS * EXPECTED_LEVEL1:
        raise CompareProjectionError("broad cell cardinality drifted")
    if len(projection.agenda_index["legacy_cells"]) != EXPECTED_LEGACY_ROWS:
        raise CompareProjectionError("legacy cell cardinality drifted")
    fine_count = sum(len(shard["cells"]) for shard in projection.topic_shards.values())
    if fine_count != EXPECTED_PRESIDENTS * EXPECTED_LEVEL2:
        raise CompareProjectionError("fine cell cardinality drifted")
    if tuple(projection.agenda_index["unmapped_level2_labels"]) != KNOWN_UNMAPPED_LEVEL2:
        raise CompareProjectionError("unmapped topic contract drifted")
    if len(projection.manifest["inventory"]) != 21:
        raise CompareProjectionError("manifest must inventory the other 21 files")
    for filename, receipt in projection.manifest["inventory"].items():
        value = projection.files.get(filename)
        if value is None or receipt["sha256"] != _sha256_bytes(value) or receipt["bytes"] != len(value):
            raise CompareProjectionError(f"manifest receipt drift for {filename}")
    if network_bundle is not None:
        source = _source_identity(network_bundle)
        if projection.agenda_index["source_identity"] != source:
            raise CompareProjectionError("source-set seal changed during projection build")
        edge_map = {
            (row.president_profile_id, row.topic_id): row
            for row in _corpus_rows(network_bundle.edges).itertuples(index=False)
        }
        cells = list(projection.agenda_index["broad_cells"])
        cells.extend(cell for shard in projection.topic_shards.values() for cell in shard["cells"])
        for cell in cells:
            source_row = edge_map.get((cell["president_profile_id"], cell["topic_id"]))
            if source_row is None:
                if cell["topic_paragraph_count"] != 0 or cell["speaker_paragraph_share"] != 0:
                    raise CompareProjectionError("absent membership was not materialized as an exact zero")
                continue
            for field in (
                "topic_paragraph_count", "topic_appearance_count",
                "eligible_president_paragraph_count", "speaker_paragraph_share",
            ):
                if cell[field] != _scalar(getattr(source_row, field)):
                    raise CompareProjectionError(f"Plan 3 parity drift for {field}")


def write_public_projection(
    projection: CompareProjectionBundle,
    site_dir: Path,
    *,
    network_bundle: speaker_topic_network.SpeakerTopicNetworkBundle | None = None,
) -> Path:
    """Atomically replace ``docs/data/compare`` with a prevalidated bundle."""
    validate_projection(projection, network_bundle=network_bundle)
    if network_bundle is not None:
        if projection.agenda_index["source_identity"] != _source_identity(network_bundle):
            raise CompareProjectionError("governed sources changed before publication")
    target = site_dir / "data" / PUBLIC_DIR_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="compare-projection-", dir=target.parent))
    temporary = temporary_root / PUBLIC_DIR_NAME
    temporary.mkdir()
    try:
        for filename, value in projection.files.items():
            (temporary / filename).write_bytes(value)
        for filename, expected in projection.files.items():
            if (temporary / filename).read_bytes() != expected:
                raise CompareProjectionError(f"temporary projection drift for {filename}")
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    validate_publication(projection, site_dir, network_bundle=network_bundle)
    return target


def validate_publication(
    projection: CompareProjectionBundle,
    site_dir: Path,
    *,
    network_bundle: speaker_topic_network.SpeakerTopicNetworkBundle | None = None,
) -> None:
    validate_projection(projection, network_bundle=network_bundle)
    target = site_dir / "data" / PUBLIC_DIR_NAME
    actual = {path.name for path in target.iterdir() if path.is_file()} if target.exists() else set()
    if actual != set(projection.files):
        raise CompareProjectionError("published Compare file inventory drifted")
    for filename, expected in projection.files.items():
        if (target / filename).read_bytes() != expected:
            raise CompareProjectionError(f"published Compare file drift for {filename}")
