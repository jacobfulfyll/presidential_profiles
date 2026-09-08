"""Governed actual-speaker invocation network for president profiles.

This module validates the accepted invocation-v2 evidence against its source
candidates, classifications, manifest, deterministic classifier replay, and
the accepted speaker foundation.  It then aggregates only resolved,
analysis-eligible, former-president, non-self rows by actual speaker.  It does
not read the legacy invocation edge table or calculate a normalized rate,
majority label, confidence score, or browser-side analytical field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import corpus, invocations, speaker_topic_network, story_foundation, trends


CONTRACT_VERSION = "actual-speaker-invocation-network-v1"
EDGE_SCHEMA = "actual-speaker-invocation-edge-v1"
EVIDENCE_SCHEMA = "actual-speaker-invocation-evidence-v1"

FUNCTIONS = (
    "legacy/inheritance",
    "institutional precedent",
    "policy inheritance",
    "historical comparison",
    "contemporary rivalry",
    "ceremonial/biographical",
    "other/unclear",
)
STANCES = ("positive", "negative", "mixed", "neutral", "unclear")

EXPECTED_SOURCE_HASHES = {
    "candidates": "sha256:20ba360a2c85f97e53e54b061291ecf7ece398db858625c49aa977a397127ded",
    "classifications": "sha256:7b6c2239fa9f1ab102654ff38d80fe4364ce661fe95bcc4de041f786362d6747",
    "manifest": "sha256:d39c9a787a63409526ca9b2cfb6c4a5bd1ad45375649d992c6999dbf84347b67",
    "accepted_evidence": "sha256:9379f4428a5de057a2c6fc92ee73d296f596735bb341a0784918cd76ae96d10f",
    "candidate_schema": "sha256:3702000334502f86437c50461da4a0efbf55ed8c7d6a58d4ec1c7aa71f8b48c0",
    "classification_schema": "sha256:7da90330ae0db92665d80431d199b2c382ea18778b41c267194bae5bfa720ec2",
}
EXPECTED_ACCEPTANCE = {
    "candidate_rows": 5_610,
    "classification_rows": 2_032,
    "accepted_evidence_rows": 2_032,
    "former_president_source_rows": 1_447,
    "unresolved_keys": 48,
    "ineligible_rows": 156,
    "reassigned_speakers": 27,
    "retained_rows": 1_243,
    "reference_paragraphs": 915,
    "source_speeches": 405,
    "directed_edges": 303,
    "source_presidents": 38,
    "target_presidents": 44,
    "actual_speaker_self_rows": 0,
}
EXPECTED_EDGE_TYPE = "actual_speaker_invokes_former_president"
EXPECTED_EVIDENCE_STATUS = "AI-classified; not human-validated"
EXPECTED_TARGET_STATUS = "former_president"

CANDIDATE_COLUMNS = (
    "candidate_id", "doc_name", "para_idx", "char_start", "char_end",
    "target", "speaker", "speech_date", "raw_mention", "context",
    "quotation_status", "source_role", "target_status", "excluded_reason",
)
CLASSIFICATION_COLUMNS = (
    "candidate_id", "function", "stance", "evidence_span", "rationale",
    "rubric_version", "evidence_status",
)
ACCEPTED_EVIDENCE_COLUMNS = (
    "candidate_id", "speaker", "target", "doc_name", "para_idx",
    "speech_date", "era", "raw_mention", "function", "stance",
    "evidence_span", "rationale", "quotation_status", "target_status",
    "evidence_status",
)
EDGE_FIELDS = (
    "edge_id", "edge_type", "source_president_profile_id",
    "source_president_name", "target_president_profile_id",
    "target_president_name", "raw_mentions", "reference_paragraph_count",
    "distinct_speeches", "function_counts", "stance_counts",
    "first_speech_date", "last_speech_date", "evidence_row_count",
    "evidence_status",
)
EVIDENCE_FIELDS = (
    "candidate_id", "edge_id", "source_president_profile_id",
    "source_president_name", "target_president_profile_id",
    "target_president_name", "doc_name", "para_idx", "speech_date",
    "story_era_id", "story_era_label", "speech_title", "source_url",
    "raw_mention", "function", "stance", "evidence_span", "rationale",
    "quotation_status", "target_status", "rubric_version", "evidence_status",
    "annotation_speaker", "source_document_owner_profile_id",
    "source_document_owner", "cross_owner",
)

MILLER_PREFIX = "https://millercenter.org/the-presidency/presidential-speeches/"
MILLER_ROUTE = "/the-presidency/presidential-speeches/"
_SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class ActualSpeakerInvocationNetworkError(RuntimeError):
    """The actual-speaker invocation contract failed closed."""


def normalized_miller_url(doc_name: str, source_url: str) -> str:
    """Return the canonical corpus URL after validating its keyed source.

    Eighteen accepted documents retain a known legacy concatenation bug in the
    source metadata (``https://millercenter.org`` followed directly by the
    slug).  The governed document key is authoritative for normalization, but
    arbitrary or mismatched URLs still fail closed.
    """
    route = str(doc_name or "").strip().replace("\\", "/")
    if route.startswith(MILLER_ROUTE):
        slug = route[len(MILLER_ROUTE):]
    elif route.startswith(MILLER_ROUTE.lstrip("/")):
        slug = route[len(MILLER_ROUTE.lstrip("/")):]
    else:
        slug = route
    if not _SLUG_RE.fullmatch(slug):
        raise ActualSpeakerInvocationNetworkError(
            f"invalid Miller Center document key: {doc_name!r}"
        )
    canonical = MILLER_PREFIX + slug
    supplied = str(source_url or "").strip()
    known_legacy = "https://millercenter.org" + slug
    if supplied not in {canonical, known_legacy}:
        raise ActualSpeakerInvocationNetworkError(
            f"source URL does not match governed document key: {doc_name!r}"
        )
    return canonical


@dataclass(frozen=True)
class ActualSpeakerInvocationNetworkBundle:
    edges: pd.DataFrame
    evidence: pd.DataFrame
    source_identity: dict[str, Any]
    acceptance: dict[str, int]
    # These immutable receipts authenticate the exact in-memory projection that
    # was derived from the governed sources.  They are validation state, not
    # public contract fields.
    _edge_projection_sha256: str
    _evidence_projection_sha256: str
    _source_identity_sha256: str
    _president_catalog: tuple[tuple[str, str, int], ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _content_id(prefix: str, identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _require_exact_columns(
    frame: pd.DataFrame, columns: Sequence[str], label: str,
) -> None:
    expected_order = tuple(columns)
    observed_order = tuple(frame.columns)
    expected = set(expected_order)
    observed = set(observed_order)
    if observed != expected or observed_order != expected_order:
        raise ActualSpeakerInvocationNetworkError(
            f"{label} schema drift: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}, "
            f"expected_order={list(expected_order)}, observed_order={list(observed_order)}"
        )


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    duplicated = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicated.empty:
        raise ActualSpeakerInvocationNetworkError(
            f"{label} has duplicate keys: {duplicated.head(5).to_dict('records')}"
        )


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
        raise ActualSpeakerInvocationNetworkError("invocation bundle contains a non-finite value")
    return value


def _records(frame: pd.DataFrame, fields: Sequence[str]) -> list[dict[str, Any]]:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ActualSpeakerInvocationNetworkError(
            f"invocation records are missing fields: {sorted(missing)}"
        )
    return [
        {field: _scalar(value) for field, value in zip(fields, row, strict=True)}
        for row in frame.loc[:, list(fields)].itertuples(index=False, name=None)
    ]


def _projection_sha256(frame: pd.DataFrame, fields: Sequence[str]) -> str:
    payload = _canonical_json(_records(frame, fields)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _value_sha256(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _strict_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _required_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActualSpeakerInvocationNetworkError(
            f"could not read invocation manifest: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ActualSpeakerInvocationNetworkError("invocation manifest is not an object")
    return value


def _era_label(speech_date: Any) -> str:
    year = pd.Timestamp(speech_date).year
    for label, start, end in trends.ERAS:
        if start <= year <= end:
            return label
    return "outside corpus"


def _normalized(value: Any) -> str:
    return " ".join(str(value).split())


def _validate_source_hashes(paths: Mapping[str, Path]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for name, path in paths.items():
        if not Path(path).is_file():
            raise ActualSpeakerInvocationNetworkError(
                f"required invocation source is missing: {path}"
            )
        digest = _sha256_file(path)
        if digest != EXPECTED_SOURCE_HASHES[name]:
            raise ActualSpeakerInvocationNetworkError(
                f"accepted invocation source hash drift: {name}"
            )
        receipts[name] = {
            "path": str(Path(path).resolve().relative_to(corpus.DATA_DIR.parent.resolve())),
            "sha256": digest,
            "bytes": Path(path).stat().st_size,
        }
    return receipts


def _validate_runtime_enums() -> None:
    if set(FUNCTIONS) != set(invocations.FUNCTIONS):
        raise ActualSpeakerInvocationNetworkError(
            "runtime invocation function enum drift"
        )
    if set(STANCES) != set(invocations.STANCES):
        raise ActualSpeakerInvocationNetworkError(
            "runtime invocation stance enum drift"
        )


def _governed_source_path(value: Any, label: str) -> Path:
    if not _required_text(value):
        raise ActualSpeakerInvocationNetworkError(
            f"{label} source path is missing"
        )
    raw = Path(str(value))
    root = corpus.DATA_DIR.parent.resolve()
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ActualSpeakerInvocationNetworkError(
            f"{label} source path leaves the repository"
        ) from exc
    return path


def _validate_foundation_source_identity(identity: Mapping[str, Any]) -> None:
    required = {
        "contract_identity", "governed_schema_version", "speaker_view_schema",
        "speaker_metadata_sha256", "reference_metadata_sha256",
        "canonical_corpus_fingerprint", "attribution_run_id",
        "annotation_generation", "source_paths", "source_hashes",
        "speaker_meta_file_sha256", "reference_meta_file_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != required:
        raise ActualSpeakerInvocationNetworkError(
            "speaker-foundation source identity schema drift"
        )
    if (
        identity["contract_identity"] != story_foundation.PUBLIC_SCHEMA
        or identity["governed_schema_version"]
        != story_foundation.foundation_audit.SCHEMA_VERSION
        or identity["speaker_view_schema"]
        != story_foundation.foundation_audit.SPEAKER_VIEW_VERSION
    ):
        raise ActualSpeakerInvocationNetworkError(
            "speaker-foundation contract identity drift"
        )
    source_paths = identity["source_paths"]
    source_hashes = identity["source_hashes"]
    expected_artifacts = {
        "paragraph_view", "appearances", "entity_mentions", "era_distinctive"
    }
    if (
        not isinstance(source_paths, Mapping)
        or not isinstance(source_hashes, Mapping)
        or set(source_paths) != expected_artifacts
        or set(source_hashes) != expected_artifacts
    ):
        raise ActualSpeakerInvocationNetworkError(
            "speaker-foundation artifact identity drift"
        )
    for name in sorted(expected_artifacts):
        path = _governed_source_path(
            source_paths[name], f"speaker foundation {name}"
        )
        if not path.is_file() or _sha256_file(path) != source_hashes[name]:
            raise ActualSpeakerInvocationNetworkError(
                f"speaker-foundation source hash drift: {name}"
            )
    meta_receipts = {
        "speaker_meta_file_sha256": story_foundation.SPEAKER_META_PATH,
        "reference_meta_file_sha256": story_foundation.REFERENCE_META_PATH,
    }
    for field, path in meta_receipts.items():
        if not path.is_file() or _sha256_file(path) != identity[field]:
            raise ActualSpeakerInvocationNetworkError(
                f"speaker-foundation metadata hash drift: {field}"
            )


def _authenticate_foundation(
    foundation: story_foundation.StoryFoundationBundle,
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
) -> tuple[tuple[str, str, int], ...]:
    """Authenticate the exact paragraph overlay and president catalog in memory."""
    recorded = topic_bundle.meta.get("input_provenance", {}).get(
        "story_foundation"
    )
    try:
        observed = speaker_topic_network._foundation_provenance(foundation)
    except (KeyError, TypeError, OSError) as exc:
        raise ActualSpeakerInvocationNetworkError(
            "speaker-foundation provenance is incomplete"
        ) from exc
    if observed != recorded:
        raise ActualSpeakerInvocationNetworkError(
            "speaker foundation does not match the accepted topic bundle"
        )
    _validate_foundation_source_identity(observed)

    paragraph_path = _governed_source_path(
        observed["source_paths"]["paragraph_view"],
        "speaker foundation paragraph view",
    )
    canonical_paragraphs = pd.read_parquet(paragraph_path)
    try:
        pd.testing.assert_frame_equal(
            foundation.paragraph_view.reset_index(drop=True),
            canonical_paragraphs.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ActualSpeakerInvocationNetworkError(
            "in-memory speaker foundation differs from its governed artifact"
        ) from exc

    eligible_paragraphs = canonical_paragraphs.loc[
        canonical_paragraphs["analysis_eligible"].astype(bool)
    ]
    actual_identities = eligible_paragraphs[
        ["attributed_speaker", "attributed_speaker_profile_id"]
    ].drop_duplicates()
    owner_identities = canonical_paragraphs[
        ["document_owner", "document_owner_profile_id"]
    ].drop_duplicates()
    if (
        actual_identities["attributed_speaker"].duplicated().any()
        or actual_identities["attributed_speaker_profile_id"].duplicated().any()
        or owner_identities["document_owner"].duplicated().any()
        or owner_identities["document_owner_profile_id"].duplicated().any()
    ):
        raise ActualSpeakerInvocationNetworkError(
            "speaker-foundation president identity is not one-to-one"
        )
    actual_map = dict(actual_identities.itertuples(index=False, name=None))
    owner_map = dict(owner_identities.itertuples(index=False, name=None))
    if actual_map != owner_map or set(actual_map) != set(corpus.PARTY):
        raise ActualSpeakerInvocationNetworkError(
            "speaker-foundation president identity catalog drift"
        )

    node_rows = topic_bundle.president_nodes.sort_values(
        "display_order", kind="stable"
    )
    catalog: list[tuple[str, str, int]] = []
    for display_order, name in enumerate(corpus.PARTY, start=1):
        rows = node_rows.loc[node_rows["president_name"].eq(name)]
        if len(rows) != 1:
            raise ActualSpeakerInvocationNetworkError(
                "topic president catalog name drift"
            )
        row = rows.iloc[0]
        profile_id = str(row["president_profile_id"])
        if (
            profile_id != str(actual_map[name])
            or not _strict_int(row["display_order"])
            or int(row["display_order"]) != display_order
        ):
            raise ActualSpeakerInvocationNetworkError(
                "topic president catalog identity or order drift"
            )
        catalog.append((name, profile_id, display_order))
    if len(node_rows) != len(catalog):
        raise ActualSpeakerInvocationNetworkError(
            "topic president catalog cardinality drift"
        )
    return tuple(catalog)


def _compare_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    left_rows = left.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right_rows = right.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if len(left_rows) != len(right_rows):
        raise ActualSpeakerInvocationNetworkError(f"{label} row-count drift")
    for column in columns:
        left_values = left_rows[column].map(_scalar).tolist()
        right_values = right_rows[column].map(_scalar).tolist()
        if left_values != right_values:
            raise ActualSpeakerInvocationNetworkError(f"{label} value drift: {column}")


def _validate_accepted_evidence(
    candidates: pd.DataFrame,
    classifications: pd.DataFrame,
    accepted: pd.DataFrame,
    manifest: Mapping[str, Any],
    candidate_schema: Mapping[str, Any],
    classification_schema: Mapping[str, Any],
) -> None:
    if candidate_schema != {
        "schema_version": "invocation-candidate-v2",
        "primary_key": ["candidate_id"],
        "required": list(CANDIDATE_COLUMNS[1:]),
    }:
        raise ActualSpeakerInvocationNetworkError("invocation candidate schema drift")
    if (
        classification_schema.get("schema_version")
        != "invocation-classification-v2"
        or classification_schema.get("primary_key") != ["candidate_id"]
        or classification_schema.get("required") != list(CLASSIFICATION_COLUMNS[1:])
        or classification_schema.get("function_enum") != list(FUNCTIONS)
        or classification_schema.get("stance_enum") != list(STANCES)
    ):
        raise ActualSpeakerInvocationNetworkError(
            "invocation classification schema or enum order drift"
        )
    _require_exact_columns(candidates, CANDIDATE_COLUMNS, "invocation candidates")
    _require_exact_columns(
        classifications, CLASSIFICATION_COLUMNS, "invocation classifications"
    )
    _require_exact_columns(
        accepted, ACCEPTED_EVIDENCE_COLUMNS, "accepted invocation evidence"
    )
    for label, frame in (
        ("invocation candidates", candidates),
        ("invocation classifications", classifications),
        ("accepted invocation evidence", accepted),
    ):
        _require_unique(frame, ("candidate_id",), label)

    fingerprint = hashlib.sha256(
        "\n".join(candidates["candidate_id"].astype(str)).encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("version") != "invocation-v2"
        or manifest.get("corpus_fingerprint") != fingerprint
        or manifest.get("rubric_version") != invocations.RUBRIC_VERSION
        or manifest.get("classification_rows") != len(classifications)
    ):
        raise ActualSpeakerInvocationNetworkError("invocation manifest identity drift")
    expected_checksums = [
        hashlib.sha256(
            "\n".join(
                classifications["candidate_id"].iloc[index:index + 50].astype(str)
            ).encode("utf-8")
        ).hexdigest()
        for index in range(0, len(classifications), 50)
    ]
    if manifest.get("batch_checksums") != expected_checksums:
        raise ActualSpeakerInvocationNetworkError("invocation manifest batch checksum drift")

    eligible = candidates.loc[candidates["excluded_reason"].eq("")].copy()
    if (
        len(candidates) != EXPECTED_ACCEPTANCE["candidate_rows"]
        or len(eligible) != EXPECTED_ACCEPTANCE["classification_rows"]
        or len(classifications) != EXPECTED_ACCEPTANCE["classification_rows"]
        or len(accepted) != EXPECTED_ACCEPTANCE["accepted_evidence_rows"]
        or set(eligible["candidate_id"]) != set(classifications["candidate_id"])
        or set(eligible["candidate_id"]) != set(accepted["candidate_id"])
    ):
        raise ActualSpeakerInvocationNetworkError("invocation candidate/classification parity drift")
    speeches = corpus.load()
    replay_statuses = [
        invocations._status_on(str(row.target), str(row.speech_date), speeches)
        for row in candidates.itertuples(index=False)
    ]
    if candidates["target_status"].astype(str).tolist() != replay_statuses:
        raise ActualSpeakerInvocationNetworkError(
            "invocation candidate target-status replay drift"
        )
    if set(classifications["function"]) - set(FUNCTIONS):
        raise ActualSpeakerInvocationNetworkError("invocation function enum drift")
    if set(classifications["stance"]) - set(STANCES):
        raise ActualSpeakerInvocationNetworkError("invocation stance enum drift")
    if not classifications["rubric_version"].eq(invocations.RUBRIC_VERSION).all():
        raise ActualSpeakerInvocationNetworkError("invocation rubric version drift")

    labels_by_id = classifications.set_index("candidate_id")
    replay_fields = CLASSIFICATION_COLUMNS[1:]
    for row in eligible.itertuples(index=False):
        replay = invocations.classify_candidate(pd.Series(row._asdict()))
        recorded = labels_by_id.loc[row.candidate_id]
        if any(replay[field] != recorded[field] for field in replay_fields):
            raise ActualSpeakerInvocationNetworkError(
                f"invocation classifier replay drift: {row.candidate_id}"
            )
        if (
            not _normalized(recorded["evidence_span"])
            or _normalized(recorded["evidence_span"]) not in _normalized(row.context)
        ):
            raise ActualSpeakerInvocationNetworkError(
                f"invocation evidence span is outside candidate context: {row.candidate_id}"
            )

    candidate_fields = (
        "candidate_id", "speaker", "target", "doc_name", "para_idx",
        "speech_date", "raw_mention", "quotation_status", "target_status",
    )
    _compare_columns(eligible, accepted, candidate_fields, "accepted candidate evidence")
    _compare_columns(
        classifications,
        accepted,
        ("candidate_id", "function", "stance", "evidence_span", "rationale", "evidence_status"),
        "accepted classification evidence",
    )
    expected_eras = accepted["speech_date"].map(_era_label).tolist()
    if accepted["era"].astype(str).tolist() != expected_eras:
        raise ActualSpeakerInvocationNetworkError("accepted invocation era drift")


def _count_map(values: pd.Series, order: Sequence[str]) -> dict[str, int]:
    counts = values.value_counts().to_dict()
    unknown = set(counts) - set(order)
    if unknown:
        raise ActualSpeakerInvocationNetworkError(
            f"invocation enum drift during aggregation: {sorted(unknown)}"
        )
    return {key: int(counts.get(key, 0)) for key in order}


def _date_only(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _build_records(
    retained: pd.DataFrame,
    president_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = president_nodes.sort_values("display_order", kind="stable")
    name_to_id = nodes.set_index("president_name")["president_profile_id"].to_dict()
    name_to_order = nodes.set_index("president_name")["display_order"].to_dict()
    if set(name_to_id) != set(corpus.PARTY):
        raise ActualSpeakerInvocationNetworkError("invocation president catalog drift")
    if not set(retained["speaker"]).issubset(name_to_id):
        raise ActualSpeakerInvocationNetworkError("unknown actual-speaker president identity")
    if not set(retained["target"]).issubset(name_to_id):
        raise ActualSpeakerInvocationNetworkError("unknown invocation target president identity")
    if retained["speaker"].eq(retained["target"]).any():
        raise ActualSpeakerInvocationNetworkError("actual-speaker self invocation survived")

    edge_ids: dict[tuple[str, str], str] = {}
    edge_rows: list[dict[str, Any]] = []
    grouped = retained.groupby(["speaker", "target"], sort=False)
    for (source, target), rows in grouped:
        source_id = str(name_to_id[source])
        target_id = str(name_to_id[target])
        edge_id = _content_id(
            "invocation_edge",
            {
                "contract_version": CONTRACT_VERSION,
                "source_president_profile_id": source_id,
                "target_president_profile_id": target_id,
            },
        )
        edge_ids[(source, target)] = edge_id
        dates = rows["speech_date"].map(pd.Timestamp)
        statuses = rows["evidence_status"].drop_duplicates().tolist()
        if len(statuses) != 1:
            raise ActualSpeakerInvocationNetworkError("invocation evidence-status drift")
        edge_rows.append(
            {
                "edge_id": edge_id,
                "edge_type": "actual_speaker_invokes_former_president",
                "source_president_profile_id": source_id,
                "source_president_name": source,
                "target_president_profile_id": target_id,
                "target_president_name": target,
                "raw_mentions": len(rows),
                "reference_paragraph_count": len(
                    rows[["doc_name", "para_idx"]].drop_duplicates()
                ),
                "distinct_speeches": int(rows["doc_name"].nunique()),
                "function_counts": _count_map(rows["function"], FUNCTIONS),
                "stance_counts": _count_map(rows["stance"], STANCES),
                "first_speech_date": dates.min().date().isoformat(),
                "last_speech_date": dates.max().date().isoformat(),
                "evidence_row_count": len(rows),
                "evidence_status": statuses[0],
                "_target_order": int(name_to_order[target]),
            }
        )

    edges = pd.DataFrame(edge_rows).sort_values(
        [
            "reference_paragraph_count", "raw_mentions", "distinct_speeches",
            "_target_order", "edge_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).drop(columns=["_target_order"]).reset_index(drop=True)

    evidence_rows: list[dict[str, Any]] = []
    ordered = retained.sort_values(
        ["speech_date", "doc_name", "para_idx", "candidate_id"], kind="stable"
    )
    for row in ordered.itertuples(index=False):
        source = str(row.speaker)
        target = str(row.target)
        source_url = normalized_miller_url(str(row.doc_name), str(row.source_url))
        owner_id = _scalar(row.document_owner_profile_id)
        evidence_rows.append(
            {
                "candidate_id": str(row.candidate_id),
                "edge_id": edge_ids[(source, target)],
                "source_president_profile_id": str(name_to_id[source]),
                "source_president_name": source,
                "target_president_profile_id": str(name_to_id[target]),
                "target_president_name": target,
                "doc_name": str(row.doc_name),
                "para_idx": int(row.para_idx),
                "speech_date": _date_only(row.speech_date),
                "story_era_id": str(row.source_key),
                "story_era_label": str(row.era),
                "speech_title": str(row.title),
                "source_url": source_url,
                "raw_mention": str(row.raw_mention),
                "function": str(row.function),
                "stance": str(row.stance),
                "evidence_span": str(row.evidence_span),
                "rationale": str(row.rationale),
                "quotation_status": str(row.quotation_status),
                "target_status": str(row.target_status),
                "rubric_version": str(row.rubric_version),
                "evidence_status": str(row.evidence_status),
                "annotation_speaker": str(row.annotation_speaker),
                "source_document_owner_profile_id": owner_id,
                "source_document_owner": str(row.document_owner),
                "cross_owner": bool(row.cross_owner),
            }
        )
    evidence = pd.DataFrame(evidence_rows).sort_values(
        ["edge_id", "speech_date", "doc_name", "para_idx", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    return edges, evidence


def _source_identity(
    receipts: Mapping[str, Any],
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "accepted_sources": dict(receipts),
        "candidate_fingerprint": manifest["corpus_fingerprint"],
        "rubric_version": manifest["rubric_version"],
        "classifier_prompt_sha256": manifest.get("prompt_sha256"),
        "speaker_foundation": topic_bundle.meta["input_provenance"]["story_foundation"],
        "president_catalog_metadata_sha256": topic_bundle.meta["metadata_sha256"],
    }


def _validate_source_identity(
    bundle: ActualSpeakerInvocationNetworkBundle,
) -> None:
    identity = bundle.source_identity
    expected_fields = {
        "contract_version", "accepted_sources", "candidate_fingerprint",
        "rubric_version", "classifier_prompt_sha256", "speaker_foundation",
        "president_catalog_metadata_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_fields:
        raise ActualSpeakerInvocationNetworkError(
            "invocation source identity schema drift"
        )
    if identity["contract_version"] != CONTRACT_VERSION:
        raise ActualSpeakerInvocationNetworkError(
            "invocation source identity contract drift"
        )
    sources = identity["accepted_sources"]
    if not isinstance(sources, Mapping) or set(sources) != set(EXPECTED_SOURCE_HASHES):
        raise ActualSpeakerInvocationNetworkError(
            "invocation accepted-source inventory drift"
        )
    resolved_paths: dict[str, Path] = {}
    for name in EXPECTED_SOURCE_HASHES:
        receipt = sources[name]
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "path", "sha256", "bytes"
        }:
            raise ActualSpeakerInvocationNetworkError(
                f"invocation accepted-source receipt drift: {name}"
            )
        if Path(str(receipt["path"])).is_absolute():
            raise ActualSpeakerInvocationNetworkError(
                f"invocation accepted-source path is not portable: {name}"
            )
        path = _governed_source_path(
            receipt["path"], f"invocation accepted source {name}"
        )
        resolved_paths[name] = path
        if (
            receipt["sha256"] != EXPECTED_SOURCE_HASHES[name]
            or not _strict_int(receipt["bytes"])
            or int(receipt["bytes"]) <= 0
            or not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256_file(path) != receipt["sha256"]
        ):
            raise ActualSpeakerInvocationNetworkError(
                f"invocation accepted-source hash or size drift: {name}"
            )
    manifest = _read_json(resolved_paths["manifest"])
    if (
        identity["candidate_fingerprint"] != manifest.get("corpus_fingerprint")
        or identity["rubric_version"] != invocations.RUBRIC_VERSION
        or identity["rubric_version"] != manifest.get("rubric_version")
        or identity["classifier_prompt_sha256"] != manifest.get("prompt_sha256")
    ):
        raise ActualSpeakerInvocationNetworkError(
            "invocation source identity manifest drift"
        )
    _validate_foundation_source_identity(identity["speaker_foundation"])
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(identity["president_catalog_metadata_sha256"])
    ):
        raise ActualSpeakerInvocationNetworkError(
            "invocation president-catalog metadata identity drift"
        )
    if _value_sha256(identity) != bundle._source_identity_sha256:
        raise ActualSpeakerInvocationNetworkError(
            "invocation source identity projection drift"
        )


def _acceptance(retained: pd.DataFrame, overlay: Mapping[str, int]) -> dict[str, int]:
    pairs = retained[["speaker", "target"]].drop_duplicates()
    values = {
        "candidate_rows": EXPECTED_ACCEPTANCE["candidate_rows"],
        "classification_rows": EXPECTED_ACCEPTANCE["classification_rows"],
        "accepted_evidence_rows": EXPECTED_ACCEPTANCE["accepted_evidence_rows"],
        "former_president_source_rows": int(overlay["source_rows"]),
        "unresolved_keys": int(overlay["unresolved_keys"]),
        "ineligible_rows": int(overlay["ineligible_rows"]),
        "reassigned_speakers": int(overlay["reassigned_speakers"]),
        "retained_rows": len(retained),
        "reference_paragraphs": len(retained[["doc_name", "para_idx"]].drop_duplicates()),
        "source_speeches": int(retained["doc_name"].nunique()),
        "directed_edges": len(pairs),
        "source_presidents": int(pairs["speaker"].nunique()),
        "target_presidents": int(pairs["target"].nunique()),
        "actual_speaker_self_rows": int(retained["speaker"].eq(retained["target"]).sum()),
    }
    if values != EXPECTED_ACCEPTANCE:
        raise ActualSpeakerInvocationNetworkError(
            f"actual-speaker invocation acceptance drift: {values}"
        )
    return values


def build_network_bundle(
    foundation: story_foundation.StoryFoundationBundle,
    topic_bundle: speaker_topic_network.SpeakerTopicNetworkBundle,
    *,
    candidates_path: Path = invocations.CANDIDATES,
    classifications_path: Path = invocations.CLASSIFICATIONS,
    manifest_path: Path = invocations.ROOT / "manifest.json",
    evidence_path: Path = story_foundation.INVOCATION_EVIDENCE_PATH,
    candidate_schema_path: Path = corpus.DATA_DIR / "schemas" / "invocation-candidate-v2.json",
    classification_schema_path: Path = corpus.DATA_DIR / "schemas" / "invocation-classification-v2.json",
) -> ActualSpeakerInvocationNetworkBundle:
    """Validate accepted inputs and build the governed in-memory bundle."""
    _validate_runtime_enums()
    story_foundation.check_story_contract(foundation)
    speaker_topic_network.validate_bundle(topic_bundle)
    president_catalog = _authenticate_foundation(foundation, topic_bundle)
    paths = {
        "candidates": Path(candidates_path),
        "classifications": Path(classifications_path),
        "manifest": Path(manifest_path),
        "accepted_evidence": Path(evidence_path),
        "candidate_schema": Path(candidate_schema_path),
        "classification_schema": Path(classification_schema_path),
    }
    receipts = _validate_source_hashes(paths)
    candidates = pd.read_parquet(paths["candidates"])
    classifications = pd.read_parquet(paths["classifications"])
    accepted = pd.read_parquet(paths["accepted_evidence"])
    manifest = _read_json(paths["manifest"])
    candidate_schema = _read_json(paths["candidate_schema"])
    classification_schema = _read_json(paths["classification_schema"])
    _validate_accepted_evidence(
        candidates,
        classifications,
        accepted,
        manifest,
        candidate_schema,
        classification_schema,
    )

    accepted_with_rubric = accepted.merge(
        classifications.loc[:, ["candidate_id", "rubric_version"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if accepted_with_rubric["rubric_version"].isna().any():
        raise ActualSpeakerInvocationNetworkError(
            "accepted invocation evidence is missing rubric metadata"
        )
    retained, overlay = story_foundation.overlay_invocation_evidence(
        foundation, accepted_with_rubric
    )
    retained_text = retained["text"].map(_normalized)
    retained_spans = retained["evidence_span"].map(_normalized)
    if any(not span or span not in text for span, text in zip(retained_spans, retained_text)):
        raise ActualSpeakerInvocationNetworkError(
            "invocation evidence span is outside its retained paragraph"
        )
    acceptance = _acceptance(retained, overlay)
    edges, evidence = _build_records(retained, topic_bundle.president_nodes)
    source_identity = _source_identity(receipts, topic_bundle, manifest)
    bundle = ActualSpeakerInvocationNetworkBundle(
        edges=edges,
        evidence=evidence,
        source_identity=source_identity,
        acceptance=acceptance,
        _edge_projection_sha256=_projection_sha256(edges, EDGE_FIELDS),
        _evidence_projection_sha256=_projection_sha256(evidence, EVIDENCE_FIELDS),
        _source_identity_sha256=_value_sha256(source_identity),
        _president_catalog=president_catalog,
    )
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: ActualSpeakerInvocationNetworkBundle) -> None:
    """Reconcile every aggregate with its retained evidence rows."""
    _validate_runtime_enums()
    _require_exact_columns(bundle.edges, EDGE_FIELDS, "invocation edges")
    _require_exact_columns(bundle.evidence, EVIDENCE_FIELDS, "invocation evidence")
    _require_unique(bundle.edges, ("edge_id",), "invocation edges")
    _require_unique(bundle.evidence, ("candidate_id",), "invocation evidence")

    catalog = bundle._president_catalog
    if (
        not isinstance(catalog, tuple)
        or len(catalog) != len(corpus.PARTY)
        or [row[0] for row in catalog] != list(corpus.PARTY)
        or any(
            not isinstance(row, tuple)
            or len(row) != 3
            or not _required_text(row[0])
            or not _required_text(row[1])
            or not _strict_int(row[2])
            or int(row[2]) != index
            for index, row in enumerate(catalog, start=1)
        )
        or len({row[1] for row in catalog}) != len(catalog)
    ):
        raise ActualSpeakerInvocationNetworkError(
            "invocation president catalog validation drift"
        )
    name_to_id = {name: profile_id for name, profile_id, _ in catalog}
    id_to_name = {profile_id: name for name, profile_id, _ in catalog}
    display_order = {profile_id: int(order) for name, profile_id, order in catalog}

    if bundle.acceptance != EXPECTED_ACCEPTANCE:
        raise ActualSpeakerInvocationNetworkError("invocation acceptance receipt drift")
    if len(bundle.edges) != EXPECTED_ACCEPTANCE["directed_edges"]:
        raise ActualSpeakerInvocationNetworkError("invocation edge-count drift")
    if len(bundle.evidence) != EXPECTED_ACCEPTANCE["retained_rows"]:
        raise ActualSpeakerInvocationNetworkError("invocation evidence-count drift")
    if set(bundle.evidence["edge_id"]) != set(bundle.edges["edge_id"]):
        raise ActualSpeakerInvocationNetworkError("invocation edge/evidence membership drift")

    expected_evidence_order = sorted(
        bundle.evidence.itertuples(index=False, name=None),
        key=lambda row: (
            row[EVIDENCE_FIELDS.index("edge_id")],
            row[EVIDENCE_FIELDS.index("speech_date")],
            row[EVIDENCE_FIELDS.index("doc_name")],
            row[EVIDENCE_FIELDS.index("para_idx")],
            row[EVIDENCE_FIELDS.index("candidate_id")],
        ),
    )
    if list(bundle.evidence.itertuples(index=False, name=None)) != expected_evidence_order:
        raise ActualSpeakerInvocationNetworkError(
            "invocation evidence deterministic ordering drift"
        )

    for row in bundle.evidence.itertuples(index=False):
        required_identity = {
            "candidate_id": row.candidate_id,
            "edge_id": row.edge_id,
            "source_president_profile_id": row.source_president_profile_id,
            "source_president_name": row.source_president_name,
            "target_president_profile_id": row.target_president_profile_id,
            "target_president_name": row.target_president_name,
            "doc_name": row.doc_name,
            "story_era_id": row.story_era_id,
            "story_era_label": row.story_era_label,
            "speech_title": row.speech_title,
            "source_document_owner_profile_id": row.source_document_owner_profile_id,
            "source_document_owner": row.source_document_owner,
        }
        if any(not _required_text(value) for value in required_identity.values()):
            raise ActualSpeakerInvocationNetworkError(
                "invocation evidence has a null or empty required identity"
            )
        if (
            row.source_president_profile_id == row.target_president_profile_id
            or row.source_president_name == row.target_president_name
        ):
            raise ActualSpeakerInvocationNetworkError("invocation self edge survived")
        if (
            row.source_president_name not in name_to_id
            or row.target_president_name not in name_to_id
            or row.source_document_owner not in name_to_id
            or row.annotation_speaker not in name_to_id
            or row.source_president_profile_id not in id_to_name
            or row.target_president_profile_id not in id_to_name
            or row.source_document_owner_profile_id not in id_to_name
            or name_to_id[row.source_president_name]
            != row.source_president_profile_id
            or name_to_id[row.target_president_name]
            != row.target_president_profile_id
            or name_to_id[row.source_document_owner]
            != row.source_document_owner_profile_id
        ):
            raise ActualSpeakerInvocationNetworkError(
                "unknown or inconsistent invocation president identity"
            )
        if row.target_status != EXPECTED_TARGET_STATUS:
            raise ActualSpeakerInvocationNetworkError(
                "non-former invocation target survived"
            )
        if row.rubric_version != invocations.RUBRIC_VERSION:
            raise ActualSpeakerInvocationNetworkError(
                "invocation evidence rubric-version drift"
            )
        if row.evidence_status != EXPECTED_EVIDENCE_STATUS:
            raise ActualSpeakerInvocationNetworkError(
                "invocation evidence-status drift"
            )
        if row.function not in FUNCTIONS:
            raise ActualSpeakerInvocationNetworkError(
                "invocation evidence function enum drift"
            )
        if row.stance not in STANCES:
            raise ActualSpeakerInvocationNetworkError(
                "invocation evidence stance enum drift"
            )
        if not _strict_int(row.para_idx) or int(row.para_idx) < 0:
            raise ActualSpeakerInvocationNetworkError(
                "invocation paragraph index type drift"
            )
        if not _strict_bool(row.cross_owner):
            raise ActualSpeakerInvocationNetworkError(
                "invocation cross-owner type drift"
            )
        if (
            row.annotation_speaker != row.source_document_owner
            or bool(row.cross_owner)
            != (
                row.source_president_profile_id
                != row.source_document_owner_profile_id
            )
        ):
            raise ActualSpeakerInvocationNetworkError(
                "invocation speaker/owner attribution drift"
            )
        try:
            canonical_url = normalized_miller_url(row.doc_name, row.source_url)
        except ActualSpeakerInvocationNetworkError as exc:
            raise ActualSpeakerInvocationNetworkError(
                "unsafe invocation source URL"
            ) from exc
        if row.source_url != canonical_url:
            raise ActualSpeakerInvocationNetworkError(
                "non-canonical invocation source URL"
            )
        if (
            not isinstance(row.speech_date, str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.speech_date)
            or _date_only(row.speech_date) != row.speech_date
        ):
            raise ActualSpeakerInvocationNetworkError(
                "invocation speech-date schema drift"
            )

    computed_population = {
        "retained_rows": len(bundle.evidence),
        "reference_paragraphs": len(
            bundle.evidence[["doc_name", "para_idx"]].drop_duplicates()
        ),
        "source_speeches": int(bundle.evidence["doc_name"].nunique()),
        "directed_edges": len(
            bundle.evidence[
                ["source_president_profile_id", "target_president_profile_id"]
            ].drop_duplicates()
        ),
        "source_presidents": int(
            bundle.evidence["source_president_profile_id"].nunique()
        ),
        "target_presidents": int(
            bundle.evidence["target_president_profile_id"].nunique()
        ),
        "actual_speaker_self_rows": int(
            bundle.evidence["source_president_profile_id"].eq(
                bundle.evidence["target_president_profile_id"]
            ).sum()
        ),
        "reassigned_speakers": int(
            bundle.evidence["annotation_speaker"].ne(
                bundle.evidence["source_president_name"]
            ).sum()
        ),
    }
    if any(
        computed_population[key] != EXPECTED_ACCEPTANCE[key]
        for key in computed_population
    ):
        raise ActualSpeakerInvocationNetworkError(
            f"invocation evidence population drift: {computed_population}"
        )

    edge_lookup = bundle.edges.set_index("edge_id", drop=False)
    for edge_id, rows in bundle.evidence.groupby("edge_id", sort=False):
        edge = edge_lookup.loc[edge_id]
        for field in (
            "edge_id", "source_president_profile_id", "source_president_name",
            "target_president_profile_id", "target_president_name",
            "first_speech_date", "last_speech_date", "evidence_status",
        ):
            if not _required_text(edge[field]):
                raise ActualSpeakerInvocationNetworkError(
                    f"invocation edge has a null or empty field: {field}"
                )
        if edge["edge_type"] != EXPECTED_EDGE_TYPE:
            raise ActualSpeakerInvocationNetworkError("invocation edge-type drift")
        if (
            edge["source_president_name"] not in name_to_id
            or edge["target_president_name"] not in name_to_id
            or name_to_id[edge["source_president_name"]]
            != edge["source_president_profile_id"]
            or name_to_id[edge["target_president_name"]]
            != edge["target_president_profile_id"]
        ):
            raise ActualSpeakerInvocationNetworkError(
                "unknown or inconsistent invocation edge president identity"
            )
        expected_edge_id = _content_id(
            "invocation_edge",
            {
                "contract_version": CONTRACT_VERSION,
                "source_president_profile_id": edge["source_president_profile_id"],
                "target_president_profile_id": edge["target_president_profile_id"],
            },
        )
        if edge_id != expected_edge_id:
            raise ActualSpeakerInvocationNetworkError(
                "invocation edge content identity drift"
            )
        identity_fields = (
            "source_president_profile_id", "source_president_name",
            "target_president_profile_id", "target_president_name",
        )
        if any(
            not rows[field].eq(edge[field]).all() for field in identity_fields
        ):
            raise ActualSpeakerInvocationNetworkError(
                "invocation edge/evidence president identity drift"
            )
        if edge["evidence_status"] != EXPECTED_EVIDENCE_STATUS or not rows[
            "evidence_status"
        ].eq(edge["evidence_status"]).all():
            raise ActualSpeakerInvocationNetworkError(
                "invocation edge/evidence status drift"
            )
        for field in (
            "raw_mentions", "reference_paragraph_count", "distinct_speeches",
            "evidence_row_count",
        ):
            if not _strict_int(edge[field]) or int(edge[field]) < 0:
                raise ActualSpeakerInvocationNetworkError(
                    f"invocation edge integer-count type drift: {field}"
                )
        if not isinstance(edge["function_counts"], dict) or list(
            edge["function_counts"]
        ) != list(FUNCTIONS):
            raise ActualSpeakerInvocationNetworkError("function-count order drift")
        if not isinstance(edge["stance_counts"], dict) or list(
            edge["stance_counts"]
        ) != list(STANCES):
            raise ActualSpeakerInvocationNetworkError("stance-count order drift")
        for label, counts in (
            ("function", edge["function_counts"]),
            ("stance", edge["stance_counts"]),
        ):
            if any(
                not _strict_int(value) or int(value) < 0
                for value in counts.values()
            ):
                raise ActualSpeakerInvocationNetworkError(
                    f"{label}-count integer type drift"
                )
        expected_functions = _count_map(rows["function"], FUNCTIONS)
        expected_stances = _count_map(rows["stance"], STANCES)
        checks = (
            edge["raw_mentions"] == len(rows),
            edge["reference_paragraph_count"]
            == len(rows[["doc_name", "para_idx"]].drop_duplicates()),
            edge["distinct_speeches"] == int(rows["doc_name"].nunique()),
            edge["evidence_row_count"] == len(rows),
            edge["function_counts"] == expected_functions,
            edge["stance_counts"] == expected_stances,
            sum(edge["function_counts"].values()) == len(rows),
            sum(edge["stance_counts"].values()) == len(rows),
            edge["first_speech_date"] == rows["speech_date"].min(),
            edge["last_speech_date"] == rows["speech_date"].max(),
        )
        if not all(checks):
            raise ActualSpeakerInvocationNetworkError(
                f"invocation edge does not reconcile: {edge_id}"
            )

    expected_edge_ids = [
        row.edge_id
        for row in sorted(
            bundle.edges.itertuples(index=False),
            key=lambda row: (
                -int(row.reference_paragraph_count),
                -int(row.raw_mentions),
                -int(row.distinct_speeches),
                display_order[row.target_president_profile_id],
                row.edge_id,
            ),
        )
    ]
    if bundle.edges["edge_id"].tolist() != expected_edge_ids:
        raise ActualSpeakerInvocationNetworkError(
            "invocation edge deterministic ordering drift"
        )

    _validate_source_identity(bundle)
    if _projection_sha256(bundle.edges, EDGE_FIELDS) != bundle._edge_projection_sha256:
        raise ActualSpeakerInvocationNetworkError(
            "invocation edge projection identity drift"
        )
    if (
        _projection_sha256(bundle.evidence, EVIDENCE_FIELDS)
        != bundle._evidence_projection_sha256
    ):
        raise ActualSpeakerInvocationNetworkError(
            "invocation evidence projection identity drift"
        )


def edge_records(bundle: ActualSpeakerInvocationNetworkBundle) -> list[dict[str, Any]]:
    validate_bundle(bundle)
    return _records(bundle.edges, EDGE_FIELDS)


def evidence_records(bundle: ActualSpeakerInvocationNetworkBundle) -> list[dict[str, Any]]:
    validate_bundle(bundle)
    return _records(bundle.evidence, EVIDENCE_FIELDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the governed actual-speaker invocation network"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required; this accepted bundle is read-only")
    foundation = story_foundation.load_story_foundation()
    topic_bundle = speaker_topic_network.load_network_bundle(foundation=foundation)
    bundle = build_network_bundle(foundation, topic_bundle)
    print(json.dumps(bundle.acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
