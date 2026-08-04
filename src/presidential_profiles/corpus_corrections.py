"""Build a canonical corpus without rewriting raw or frozen paid artifacts.

The Miller Center release contains two confirmed duplication classes:

* a speech transcript repeated inside one source document; and
* two source records representing the same historical speech.

This module applies only the reviewed entries in
``data/corpus_corrections/manifest_v1.json``.  Internal repeats are removed
from the source HTML *before* analytical paragraph chunking, so a chunk that
straddled the copy boundary is rebuilt rather than preserved with duplicate
text.  Existing paragraph annotations are reusable only when both their key
and exact text fingerprint survive.

The downloaded tarball and ``data/llm_annotations/`` are read-only inputs.
All writes are deterministic, local, and confined to ``data/corpus_corrections``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .fetch import chunk_paragraphs, clean_html_paragraphs

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SOURCE_TARBALL_PATH = DATA_DIR / "raw" / "miller_center_speeches.tgz"
SOURCE_SPEECHES_PATH = DATA_DIR / "speeches.parquet"
SOURCE_PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"

CORRECTIONS_DIR = DATA_DIR / "corpus_corrections"
MANIFEST_PATH = CORRECTIONS_DIR / "manifest_v1.json"
CANONICAL_SPEECHES_PATH = CORRECTIONS_DIR / "canonical_speeches_v1.parquet"
CANONICAL_PARAGRAPHS_PATH = CORRECTIONS_DIR / "canonical_paragraphs_v1.parquet"
CANONICAL_KEYS_PATH = CORRECTIONS_DIR / "canonical_paragraph_keys_v1.parquet"
KEY_MAPPING_PATH = CORRECTIONS_DIR / "paragraph_key_mapping_v1.parquet"
EXCLUSIONS_PATH = CORRECTIONS_DIR / "exclusions_v1.parquet"
REANNOTATION_PATH = CORRECTIONS_DIR / "reannotation_required_v1.parquet"
META_PATH = CORRECTIONS_DIR / "meta_v1.json"

SCHEMA_VERSION = "corpus-corrections-v1"
KEYS = ["doc_name", "para_idx"]
KNOWN_OPERATIONS = {"repeat_within_document", "duplicate_document"}
REUSABLE_STATUS = "unchanged_reusable"
REANNOTATION_STATUSES = {
    "changed_requires_reannotation",
    "new_requires_reannotation",
}
EXCLUDED_STATUSES = {
    "excluded_duplicate",
    "excluded_duplicate_document",
}


class CorrectionError(RuntimeError):
    """A correction manifest or source artifact failed a refusal guard."""


class AnnotationProjectionBlocked(CorrectionError):
    """A frozen annotation table cannot fully cover the canonical corpus."""


@dataclass(frozen=True)
class CanonicalProjection:
    """In-memory canonical corpus and its provenance/audit tables."""

    speeches: pd.DataFrame
    paragraphs: pd.DataFrame
    keys: pd.DataFrame
    key_mapping: pd.DataFrame
    exclusions: pd.DataFrame
    reannotation_required: pd.DataFrame
    meta: dict[str, Any]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_tokens(text: str) -> list[str]:
    token = []
    tokens = []
    for char in text.casefold():
        if char.isascii() and char.isalnum():
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def _normalized_sha256(text: str) -> str:
    return _sha256_text(" ".join(_normalized_tokens(text)))


def _paragraph_fingerprint(paragraphs: Iterable[str]) -> str:
    payload = json.dumps(
        list(enumerate(paragraphs)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _corpus_fingerprint(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    for row in speeches.sort_values("doc_name").itertuples(index=False):
        digest.update(str(row.doc_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.transcript).encode("utf-8"))
        digest.update(b"\n")
    for row in paragraphs.sort_values(KEYS).itertuples(index=False):
        digest.update(str(row.doc_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row.para_idx)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row.text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(canonical)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and structurally validate the versioned correction manifest."""
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CorrectionError(
            f"Expected manifest schema {SCHEMA_VERSION!r}; "
            f"got {manifest.get('schema_version')!r}"
        )
    corrections = manifest.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise CorrectionError("Manifest corrections must be a non-empty list")

    ids: set[str] = set()
    operated_docs: set[str] = set()
    for entry in corrections:
        if not isinstance(entry, dict):
            raise CorrectionError("Every correction entry must be an object")
        correction_id = entry.get("id")
        if not isinstance(correction_id, str) or not correction_id:
            raise CorrectionError("Every correction requires a non-empty id")
        if correction_id in ids:
            raise CorrectionError(f"Duplicate correction id: {correction_id}")
        ids.add(correction_id)

        operation = entry.get("operation")
        if operation not in KNOWN_OPERATIONS:
            raise CorrectionError(
                f"Unknown correction operation {operation!r} in {correction_id}"
            )
        if entry.get("review_status") != "confirmed":
            raise CorrectionError(
                f"Correction {correction_id} is not confirmed for application"
            )

        if operation == "repeat_within_document":
            docs = [entry.get("doc_name")]
        else:
            docs = [
                entry.get("canonical_doc_name"),
                entry.get("excluded_doc_name"),
            ]
            if docs[0] == docs[1]:
                raise CorrectionError(
                    f"Duplicate-document correction {correction_id} names "
                    "the same canonical and excluded record"
                )
        if any(not isinstance(doc, str) or not doc for doc in docs):
            raise CorrectionError(
                f"Correction {correction_id} has an invalid document name"
            )
        overlap = operated_docs.intersection(docs)
        if overlap:
            raise CorrectionError(
                f"Documents occur in overlapping corrections: {sorted(overlap)}"
            )
        operated_docs.update(docs)
    return manifest


def load_raw_documents(
    tarball_path: Path,
    doc_names: set[str],
) -> dict[str, dict[str, Any]]:
    """Read only the named JSON documents from the preserved source tarball."""
    documents: dict[str, dict[str, Any]] = {}
    with tarfile.open(tarball_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".json"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            document = json.load(handle)
            doc_name = document.get("doc_name")
            if doc_name not in doc_names:
                continue
            if doc_name in documents:
                raise CorrectionError(
                    f"Source tarball contains duplicate doc_name {doc_name}"
                )
            documents[doc_name] = document
    missing = doc_names - documents.keys()
    if missing:
        raise CorrectionError(
            f"Manifest documents missing from source tarball: {sorted(missing)}"
        )
    return documents


def _require_unique_source_keys(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
) -> None:
    if "doc_name" not in speeches or speeches["doc_name"].duplicated().any():
        raise CorrectionError("Source speeches must have unique doc_name keys")
    missing = {"doc_name", "para_idx", "text"} - set(paragraphs.columns)
    if missing:
        raise CorrectionError(
            f"Source paragraphs missing required columns: {sorted(missing)}"
        )
    if paragraphs.duplicated(KEYS).any():
        raise CorrectionError(
            "Source paragraphs contain duplicate (doc_name, para_idx) keys"
        )
    for doc_name, group in paragraphs.groupby("doc_name", sort=False):
        indices = group["para_idx"].sort_values().astype(int).tolist()
        if indices != list(range(len(indices))):
            raise CorrectionError(
                f"Paragraph indices are not contiguous from zero for {doc_name}"
            )


def _bounds(entry: dict[str, Any], field: str) -> tuple[int, int]:
    value = entry.get(field)
    if not isinstance(value, dict):
        raise CorrectionError(f"{entry['id']} is missing {field}")
    start = value.get("start")
    end = value.get("end_inclusive")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise CorrectionError(f"{entry['id']} has invalid {field}")
    return start, end


def _validate_repeat(
    entry: dict[str, Any],
    speech: pd.Series,
    old_paragraphs: pd.DataFrame,
    raw_document: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    correction_id = entry["id"]
    doc_name = entry["doc_name"]
    source_paragraphs = clean_html_paragraphs(
        raw_document.get("transcript_html") or ""
    )
    if len(source_paragraphs) != entry.get("source_paragraph_count"):
        raise CorrectionError(
            f"{correction_id}: source paragraph count changed "
            f"({len(source_paragraphs)} != {entry.get('source_paragraph_count')})"
        )

    retained_start, retained_end = _bounds(
        entry, "retained_source_para_idx"
    )
    excluded_start, excluded_end = _bounds(
        entry, "excluded_source_para_idx"
    )
    if retained_start != 0 or retained_end + 1 != excluded_start:
        raise CorrectionError(
            f"{correction_id}: source retained/excluded ranges overlap or gap"
        )
    if excluded_end != len(source_paragraphs) - 1:
        raise CorrectionError(
            f"{correction_id}: excluded source range is out of bounds"
        )

    retained_source = source_paragraphs[retained_start : retained_end + 1]
    excluded_source = source_paragraphs[excluded_start : excluded_end + 1]
    if (
        _paragraph_fingerprint(retained_source)
        != entry.get("source_retained_fingerprint")
    ):
        raise CorrectionError(
            f"{correction_id}: retained source paragraph fingerprint changed"
        )
    if (
        _paragraph_fingerprint(excluded_source)
        != entry.get("source_excluded_fingerprint")
    ):
        raise CorrectionError(
            f"{correction_id}: excluded source paragraph fingerprint changed"
        )

    old = old_paragraphs.sort_values("para_idx")
    if len(old) != entry.get("source_analytical_paragraph_count"):
        raise CorrectionError(
            f"{correction_id}: analytical paragraph count changed "
            f"({len(old)} != {entry.get('source_analytical_paragraph_count')})"
        )
    canonical_texts = chunk_paragraphs(retained_source)
    if len(canonical_texts) != entry.get("canonical_paragraph_count"):
        raise CorrectionError(
            f"{correction_id}: canonical paragraph count changed "
            f"({len(canonical_texts)} != {entry.get('canonical_paragraph_count')})"
        )
    if (
        _paragraph_fingerprint(canonical_texts)
        != entry.get("canonical_paragraph_fingerprint")
    ):
        raise CorrectionError(
            f"{correction_id}: canonical paragraph fingerprint changed"
        )
    changed = [
        index
        for index, text in enumerate(canonical_texts)
        if index >= len(old) or text != old.iloc[index]["text"]
    ]
    if changed != entry.get("changed_canonical_para_idx"):
        raise CorrectionError(
            f"{correction_id}: changed canonical paragraph keys drifted "
            f"({changed} != {entry.get('changed_canonical_para_idx')})"
        )

    transcript = str(speech["transcript"])
    if _sha256_text(transcript) != entry.get("source_transcript_sha256"):
        raise CorrectionError(
            f"{correction_id}: source transcript fingerprint changed"
        )
    split = entry.get("transcript_split_char")
    if not isinstance(split, int) or not 0 < split < len(transcript):
        raise CorrectionError(f"{correction_id}: transcript split is invalid")
    retained_transcript = transcript[:split].strip()
    excluded_transcript = transcript[split:].strip()
    if (
        _sha256_text(retained_transcript)
        != entry.get("retained_transcript_sha256")
    ):
        raise CorrectionError(
            f"{correction_id}: retained transcript fingerprint changed"
        )
    if (
        _sha256_text(excluded_transcript)
        != entry.get("excluded_transcript_sha256")
    ):
        raise CorrectionError(
            f"{correction_id}: excluded transcript fingerprint changed"
        )

    canonical_speech = speech.copy()
    canonical_speech["transcript"] = retained_transcript
    canonical_speech["word_count"] = len(retained_transcript.split())
    canonical_paragraphs = pd.DataFrame(
        {
            "doc_name": doc_name,
            "para_idx": range(len(canonical_texts)),
            "text": canonical_texts,
            "word_count": [len(text.split()) for text in canonical_texts],
        }
    )
    return canonical_speech, canonical_paragraphs


def _validate_duplicate_document(
    entry: dict[str, Any],
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
) -> None:
    correction_id = entry["id"]
    canonical_name = entry["canonical_doc_name"]
    excluded_name = entry["excluded_doc_name"]
    rows: dict[str, pd.Series] = {}
    for role, doc_name in (
        ("canonical", canonical_name),
        ("excluded", excluded_name),
    ):
        match = speeches[speeches["doc_name"] == doc_name]
        if len(match) != 1:
            raise CorrectionError(
                f"{correction_id}: expected one {role} speech row for {doc_name}"
            )
        row = match.iloc[0]
        rows[role] = row
        evidence = entry.get(role)
        if not isinstance(evidence, dict):
            raise CorrectionError(
                f"{correction_id}: missing {role} evidence block"
            )
        actual_date = pd.Timestamp(row["date"]).isoformat()
        checks = {
            "title": str(row["title"]),
            "date": actual_date,
            "paragraph_count": int(
                (paragraphs["doc_name"] == doc_name).sum()
            ),
            "transcript_sha256": _sha256_text(str(row["transcript"])),
            "normalized_transcript_sha256": _normalized_sha256(
                str(row["transcript"])
            ),
            "introduction_sha256": _sha256_text(str(row["introduction"])),
        }
        for field, actual in checks.items():
            if evidence.get(field) != actual:
                raise CorrectionError(
                    f"{correction_id}: {role} {field} changed "
                    f"({actual!r} != {evidence.get(field)!r})"
                )

    if rows["canonical"]["president"] != rows["excluded"]["president"]:
        raise CorrectionError(
            f"{correction_id}: duplicate records have different presidents"
        )


def _mapping_and_audits(
    old_paragraphs: pd.DataFrame,
    canonical_paragraphs: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_by_doc = {
        entry["doc_name"]: entry
        for entry in manifest["corrections"]
        if entry["operation"] == "repeat_within_document"
    }
    excluded_document_by_doc = {
        entry["excluded_doc_name"]: entry
        for entry in manifest["corrections"]
        if entry["operation"] == "duplicate_document"
    }

    old_lookup = {
        (str(row.doc_name), int(row.para_idx)): str(row.text)
        for row in old_paragraphs.itertuples(index=False)
    }
    canonical_lookup = {
        (str(row.doc_name), int(row.para_idx)): str(row.text)
        for row in canonical_paragraphs.itertuples(index=False)
    }

    mapping_rows: list[dict[str, Any]] = []
    for key, old_text in old_lookup.items():
        doc_name, para_idx = key
        canonical_text = canonical_lookup.get(key)
        repeat = repeat_by_doc.get(doc_name)
        duplicate = excluded_document_by_doc.get(doc_name)
        if canonical_text is not None:
            status = (
                REUSABLE_STATUS
                if canonical_text == old_text
                else "changed_requires_reannotation"
            )
            correction_id = repeat["id"] if repeat else None
            canonical_doc_name: str | None = doc_name
            canonical_para_idx: int | None = para_idx
        elif duplicate is not None:
            status = "excluded_duplicate_document"
            correction_id = duplicate["id"]
            canonical_doc_name = duplicate["canonical_doc_name"]
            canonical_para_idx = None
        elif repeat is not None:
            status = "excluded_duplicate"
            correction_id = repeat["id"]
            canonical_doc_name = doc_name
            canonical_para_idx = None
        else:
            raise CorrectionError(
                "Canonical projection removed an undeclared paragraph key: "
                f"{key}"
            )
        mapping_rows.append(
            {
                "old_doc_name": doc_name,
                "old_para_idx": para_idx,
                "canonical_doc_name": canonical_doc_name,
                "canonical_para_idx": canonical_para_idx,
                "status": status,
                "correction_id": correction_id,
                "old_text_sha256": _sha256_text(old_text),
                "canonical_text_sha256": (
                    _sha256_text(canonical_text)
                    if canonical_text is not None
                    else None
                ),
            }
        )

    for key, canonical_text in canonical_lookup.items():
        if key in old_lookup:
            continue
        doc_name, para_idx = key
        repeat = repeat_by_doc.get(doc_name)
        if repeat is None:
            raise CorrectionError(
                f"Canonical projection created an undeclared new key: {key}"
            )
        mapping_rows.append(
            {
                "old_doc_name": None,
                "old_para_idx": None,
                "canonical_doc_name": doc_name,
                "canonical_para_idx": para_idx,
                "status": "new_requires_reannotation",
                "correction_id": repeat["id"],
                "old_text_sha256": None,
                "canonical_text_sha256": _sha256_text(canonical_text),
            }
        )

    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["canonical_doc_name", "canonical_para_idx", "old_doc_name", "old_para_idx"],
        na_position="last",
    )
    mapping["old_para_idx"] = mapping["old_para_idx"].astype("Int64")
    mapping["canonical_para_idx"] = mapping["canonical_para_idx"].astype(
        "Int64"
    )

    reannotation = mapping[
        mapping["status"].isin(REANNOTATION_STATUSES)
    ].copy()
    reannotation = reannotation[
        [
            "canonical_doc_name",
            "canonical_para_idx",
            "status",
            "correction_id",
            "old_text_sha256",
            "canonical_text_sha256",
        ]
    ].rename(
        columns={
            "canonical_doc_name": "doc_name",
            "canonical_para_idx": "para_idx",
        }
    )

    exclusion_rows: list[dict[str, Any]] = []
    excluded_mapping = mapping[mapping["status"].isin(EXCLUDED_STATUSES)]
    for row in excluded_mapping.itertuples(index=False):
        operation = (
            "duplicate_document"
            if row.status == "excluded_duplicate_document"
            else "repeat_within_document"
        )
        exclusion_rows.append(
            {
                "scope": "paragraph",
                "operation": operation,
                "correction_id": row.correction_id,
                "doc_name": row.old_doc_name,
                "para_idx": row.old_para_idx,
                "canonical_doc_name": row.canonical_doc_name,
                "reason": row.status,
                "text_sha256": row.old_text_sha256,
            }
        )
    for entry in manifest["corrections"]:
        if entry["operation"] != "duplicate_document":
            continue
        exclusion_rows.append(
            {
                "scope": "speech",
                "operation": "duplicate_document",
                "correction_id": entry["id"],
                "doc_name": entry["excluded_doc_name"],
                "para_idx": None,
                "canonical_doc_name": entry["canonical_doc_name"],
                "reason": "excluded_duplicate_document",
                "text_sha256": entry["excluded"]["transcript_sha256"],
            }
        )
    exclusions = pd.DataFrame(exclusion_rows).sort_values(
        ["scope", "doc_name", "para_idx"], na_position="last"
    )
    exclusions["para_idx"] = exclusions["para_idx"].astype("Int64")
    return mapping.reset_index(drop=True), exclusions.reset_index(
        drop=True
    ), reannotation.reset_index(drop=True)


def build_projection(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
    raw_documents: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> CanonicalProjection:
    """Apply a validated manifest to in-memory source artifacts."""
    _require_unique_source_keys(speeches, paragraphs)
    speeches = speeches.copy()
    paragraphs = paragraphs.copy()

    repeat_entries = [
        entry
        for entry in manifest["corrections"]
        if entry["operation"] == "repeat_within_document"
    ]
    duplicate_entries = [
        entry
        for entry in manifest["corrections"]
        if entry["operation"] == "duplicate_document"
    ]

    replacement_speeches: dict[str, pd.Series] = {}
    replacement_paragraphs: dict[str, pd.DataFrame] = {}
    for entry in repeat_entries:
        doc_name = entry["doc_name"]
        speech_match = speeches[speeches["doc_name"] == doc_name]
        if len(speech_match) != 1:
            raise CorrectionError(
                f"{entry['id']}: expected one speech row for {doc_name}"
            )
        old_paragraphs = paragraphs[paragraphs["doc_name"] == doc_name]
        canonical_speech, canonical_paragraphs = _validate_repeat(
            entry,
            speech_match.iloc[0],
            old_paragraphs,
            raw_documents[doc_name],
        )
        replacement_speeches[doc_name] = canonical_speech
        replacement_paragraphs[doc_name] = canonical_paragraphs

    excluded_docs: set[str] = set()
    for entry in duplicate_entries:
        _validate_duplicate_document(entry, speeches, paragraphs)
        excluded_docs.add(entry["excluded_doc_name"])

    retained_speeches = speeches[
        ~speeches["doc_name"].isin(excluded_docs | replacement_speeches.keys())
    ]
    speech_frames = [retained_speeches]
    if replacement_speeches:
        speech_frames.append(
            pd.DataFrame(list(replacement_speeches.values()))
        )
    canonical_speeches = pd.concat(speech_frames, ignore_index=True)
    canonical_speeches = canonical_speeches.sort_values("date").reset_index(
        drop=True
    )

    replaced_or_excluded = (
        excluded_docs | replacement_paragraphs.keys()
    )
    paragraph_frames = [
        paragraphs[~paragraphs["doc_name"].isin(replaced_or_excluded)]
    ]
    paragraph_frames.extend(replacement_paragraphs.values())
    canonical_paragraphs = pd.concat(paragraph_frames, ignore_index=True)
    canonical_paragraphs = canonical_paragraphs.sort_values(KEYS).reset_index(
        drop=True
    )
    _require_unique_source_keys(canonical_speeches, canonical_paragraphs)
    canonical_doc_names = set(canonical_speeches["doc_name"])
    paragraph_doc_names = set(canonical_paragraphs["doc_name"])
    if not paragraph_doc_names <= canonical_doc_names:
        raise CorrectionError(
            "Canonical paragraphs reference missing canonical speech rows"
        )

    mapping, exclusions, reannotation = _mapping_and_audits(
        paragraphs,
        canonical_paragraphs,
        manifest,
    )
    keys = canonical_paragraphs[KEYS].copy()
    repeat_excluded = int(
        (
            mapping["status"]
            == "excluded_duplicate"
        ).sum()
    )
    duplicate_document_excluded = int(
        (
            mapping["status"]
            == "excluded_duplicate_document"
        ).sum()
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _manifest_sha256(manifest),
        "source_counts": {
            "speeches": int(len(speeches)),
            "paragraphs": int(len(paragraphs)),
        },
        "canonical_counts": {
            "speeches": int(len(canonical_speeches)),
            "paragraphs": int(len(canonical_paragraphs)),
        },
        "correction_counts": {
            "repeat_within_document": len(repeat_entries),
            "duplicate_document": len(duplicate_entries),
            "paragraphs_excluded_internal_repeat": repeat_excluded,
            "paragraphs_excluded_duplicate_document": (
                duplicate_document_excluded
            ),
            "paragraphs_requiring_reannotation": int(len(reannotation)),
        },
        "annotation_projection_status": (
            "blocked_requires_reannotation"
            if len(reannotation)
            else "complete_reuse_available"
        ),
        "canonical_corpus_fingerprint": _corpus_fingerprint(
            canonical_speeches,
            canonical_paragraphs,
        ),
    }
    return CanonicalProjection(
        speeches=canonical_speeches,
        paragraphs=canonical_paragraphs,
        keys=keys,
        key_mapping=mapping,
        exclusions=exclusions,
        reannotation_required=reannotation,
        meta=meta,
    )


def build_from_paths(
    *,
    manifest_path: Path = MANIFEST_PATH,
    tarball_path: Path = SOURCE_TARBALL_PATH,
    speeches_path: Path = SOURCE_SPEECHES_PATH,
    paragraphs_path: Path = SOURCE_PARAGRAPHS_PATH,
) -> CanonicalProjection:
    """Load source artifacts, validate every correction, and project them."""
    manifest = load_manifest(manifest_path)
    doc_names: set[str] = set()
    for entry in manifest["corrections"]:
        if entry["operation"] == "repeat_within_document":
            doc_names.add(entry["doc_name"])
        else:
            doc_names.add(entry["canonical_doc_name"])
            doc_names.add(entry["excluded_doc_name"])
    raw_documents = load_raw_documents(tarball_path, doc_names)
    speeches = pd.read_parquet(speeches_path)
    paragraphs = pd.read_parquet(paragraphs_path)
    projection = build_projection(
        speeches,
        paragraphs,
        raw_documents,
        manifest,
    )
    meta = dict(projection.meta)
    meta["source_fingerprints"] = {
        "tarball_sha256": _sha256_file(tarball_path),
        "speeches_parquet_sha256": _sha256_file(speeches_path),
        "paragraphs_parquet_sha256": _sha256_file(paragraphs_path),
    }
    return CanonicalProjection(
        speeches=projection.speeches,
        paragraphs=projection.paragraphs,
        keys=projection.keys,
        key_mapping=projection.key_mapping,
        exclusions=projection.exclusions,
        reannotation_required=projection.reannotation_required,
        meta=meta,
    )


def project_frozen_paragraph_table(
    table: pd.DataFrame,
    projection: CanonicalProjection,
    *,
    require_complete: bool,
    allow_multiple_per_key: bool = False,
) -> pd.DataFrame:
    """Project frozen paragraph rows only where key and text are reusable.

    Known excluded duplicate rows are discarded.  A key absent from the
    old-to-canonical mapping is an unknown orphan and always fails.  Changed
    boundary chunks never inherit an old label; when ``require_complete`` is
    true, their absence raises :class:`AnnotationProjectionBlocked`.
    """
    missing_columns = set(KEYS) - set(table.columns)
    if missing_columns:
        raise CorrectionError(
            f"Frozen table missing key columns: {sorted(missing_columns)}"
        )
    if not allow_multiple_per_key and table.duplicated(KEYS).any():
        raise CorrectionError("Frozen table has duplicate paragraph keys")

    old_mapping = projection.key_mapping[
        projection.key_mapping["old_doc_name"].notna()
    ]
    known_old_keys = set(
        zip(
            old_mapping["old_doc_name"],
            old_mapping["old_para_idx"].astype(int),
            strict=True,
        )
    )
    table_keys = set(
        zip(
            table["doc_name"].astype(str),
            table["para_idx"].astype(int),
            strict=True,
        )
    )
    unknown = table_keys - known_old_keys
    if unknown:
        sample = sorted(unknown)[:5]
        raise CorrectionError(
            f"Frozen table has {len(unknown)} unknown annotation orphans; "
            f"sample={sample}"
        )

    reusable = old_mapping[old_mapping["status"] == REUSABLE_STATUS][
        ["old_doc_name", "old_para_idx"]
    ].rename(
        columns={
            "old_doc_name": "doc_name",
            "old_para_idx": "para_idx",
        }
    )
    filtered = table.merge(reusable, on=KEYS, how="inner", validate=(
        "many_to_one" if allow_multiple_per_key else "one_to_one"
    ))
    if require_complete:
        projected_keys = set(
            zip(
                filtered["doc_name"].astype(str),
                filtered["para_idx"].astype(int),
                strict=True,
            )
        )
        canonical_keys = set(
            zip(
                projection.keys["doc_name"].astype(str),
                projection.keys["para_idx"].astype(int),
                strict=True,
            )
        )
        missing = canonical_keys - projected_keys
        if missing:
            sample = sorted(missing)[:5]
            raise AnnotationProjectionBlocked(
                f"Frozen annotations cannot fully cover the canonical corpus: "
                f"{len(missing)} canonical keys require labels; sample={sample}"
            )
    return filtered


def write_projection(
    projection: CanonicalProjection,
    *,
    out_dir: Path = CORRECTIONS_DIR,
) -> None:
    """Write a fully validated projection to fixed basenames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "canonical_speeches_v1.parquet": projection.speeches,
        "canonical_paragraphs_v1.parquet": projection.paragraphs,
        "canonical_paragraph_keys_v1.parquet": projection.keys,
        "paragraph_key_mapping_v1.parquet": projection.key_mapping,
        "exclusions_v1.parquet": projection.exclusions,
        "reannotation_required_v1.parquet": projection.reannotation_required,
    }
    # All refusal guards have already completed in build_projection.  Stage
    # every parquet successfully before replacing any published artifact.
    staged: list[tuple[Path, Path]] = []
    try:
        for basename, frame in paths.items():
            destination = out_dir / basename
            temporary = out_dir / f".{basename}.{uuid.uuid4().hex}.tmp"
            frame.to_parquet(temporary, index=False)
            staged.append((temporary, destination))
        meta_payload = dict(projection.meta)
        meta_payload["output_fingerprints"] = {
            destination.name: _sha256_file(temporary)
            for temporary, destination in staged
        }
        meta_destination = out_dir / "meta_v1.json"
        meta_temporary = out_dir / f".meta_v1.json.{uuid.uuid4().hex}.tmp"
        meta_temporary.write_text(
            json.dumps(meta_payload, indent=2) + "\n"
        )
        staged.append((meta_temporary, meta_destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def _summary(projection: CanonicalProjection) -> dict[str, Any]:
    return {
        **projection.meta["source_counts"],
        **{
            f"canonical_{key}": value
            for key, value in projection.meta["canonical_counts"].items()
        },
        **projection.meta["correction_counts"],
        "annotation_projection_status": projection.meta[
            "annotation_projection_status"
        ],
        "canonical_corpus_fingerprint": projection.meta[
            "canonical_corpus_fingerprint"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and build the canonical corrected corpus"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "audit",
        help="validate the manifest and print the in-memory correction summary",
    )
    subparsers.add_parser(
        "build",
        help="validate then write the canonical corpus and audit artifacts",
    )
    args = parser.parse_args()

    projection = build_from_paths()
    if args.command == "build":
        write_projection(projection)
    print(json.dumps(_summary(projection), indent=2))


if __name__ == "__main__":
    main()
