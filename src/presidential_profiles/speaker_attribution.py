"""Offline, user-visible speaker-attribution control plane.

This module never invokes a model, imports a provider client, reads credentials,
or opens a socket.  It prepares locked assignments for a visible Codex chat,
validates returned JSON before the first write, and derives reproducible census
artifacts from accepted evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from . import annotation_ledger as ledger
from .corpus import DATA_DIR, PARTY


ROOT = DATA_DIR / "speaker_attribution"
CONTROL_ROOT = ledger.LEDGER_ROOT / "speaker_attribution"
RUN_ROOT = CONTROL_ROOT / "runs"
SPEC_ROOT = ledger.SPECS_ROOT
REGISTRY_V2 = SPEC_ROOT / "registry-v2.json"
REGISTRY_V3 = SPEC_ROOT / "registry-v3.json"
CORRECTION_ROOT = DATA_DIR / "corpus_corrections"
SOURCE_SPEECHES = DATA_DIR / "speeches.parquet"
SOURCE_PARAGRAPHS = DATA_DIR / "paragraphs.parquet"
CANONICAL_SPEECHES = CORRECTION_ROOT / "canonical_speeches_v1.parquet"
CANONICAL_PARAGRAPHS = CORRECTION_ROOT / "canonical_paragraphs_v1.parquet"
CORRECTION_META = CORRECTION_ROOT / "meta_v1.json"

DOCUMENT_CLASSES = {
    "single_president", "multi_speaker", "joint_authored", "uncertain"
}
PARAGRAPH_CLASSES = {
    "canonical_president", "non_president", "multiple_speakers",
    "joint_or_shared", "scaffolding", "uncertain",
}
REASON_CODES = {
    "explicit_speaker_cue", "same_turn_continuation", "single_voice_inheritance",
    "stage_direction", "shared_authorship", "combined_turns", "insufficient_evidence",
}
PROTECTED_EXACT = {
    REGISTRY_V2,
    ledger.REGISTRY_PATH,
    ledger.CURRENT_PATH,
    CORRECTION_ROOT / "manifest_v1.json",
}


def _canonical(value: Any) -> str:
    return ledger.canonical_json(value)


def _hash_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    return ledger.sha256_text("".join(_canonical(dict(row)) + "\n" for row in rows))


def _runtime_session_path(thread_id: str) -> Path:
    candidates = sorted(
        (Path.home() / ".codex" / "sessions").glob(f"**/*{thread_id}.jsonl")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"exactly one current rollout is required for {thread_id}; found {len(candidates)}"
        )
    return candidates[0]


def capture_runtime_receipt(*, thread_id: str | None = None) -> dict[str, Any]:
    """Read exact model/effort from this task's current rollout metadata."""
    thread_id = thread_id or os.environ.get("CODEX_THREAD_ID")
    if not thread_id:
        raise RuntimeError("CODEX_THREAD_ID is unavailable")
    path = _runtime_session_path(thread_id)
    contexts: list[dict[str, Any]] = []
    session_meta: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "session_meta":
                session_meta = row["payload"]
            elif row.get("type") == "turn_context":
                contexts.append(row["payload"])
    if not contexts or session_meta is None:
        raise RuntimeError("rollout does not expose session and turn metadata")
    current = contexts[-1]
    model = current.get("model")
    effort = current.get("effort")
    if not isinstance(model, str) or not model.strip() or not isinstance(effort, str) or not effort.strip():
        raise RuntimeError("runtime does not expose an exact model and reasoning effort")
    stat = path.stat()
    return {
        "receipt_schema_version": "speaker-runtime-receipt-v1",
        "thread_id": thread_id,
        "session_id": session_meta.get("session_id"),
        "model_id": model,
        "reasoning_effort": effort,
        "specificity": "exact_runtime_identifier",
        "identity_source": "current_rollout_turn_context",
        "rollout_path": str(path),
        "rollout_size_at_capture": stat.st_size,
        "codex_cli_version": session_meta.get("cli_version"),
        "originator": session_meta.get("originator"),
    }


def _protected_paths() -> list[Path]:
    paths = set(PROTECTED_EXACT)
    paths.update(p for p in (DATA_DIR / "llm_annotations").rglob("*") if p.is_file())
    paths.update(p for p in CORRECTION_ROOT.rglob("*") if p.is_file())
    paths.update(p for p in ledger.SEALED_ROOT.rglob("*") if p.is_file())
    # Old registries are protected; registry-v3 is intentionally additive.
    paths.update(SPEC_ROOT.glob("registry-v[12].json"))
    return sorted((p for p in paths if p.exists()), key=lambda p: str(p))


def protected_inventory() -> dict[str, Any]:
    rows = [
        {
            "path": str(path.relative_to(DATA_DIR.parent)),
            "size": path.stat().st_size,
            "sha256": ledger.sha256_file(path),
        }
        for path in _protected_paths()
    ]
    return {
        "schema_version": "speaker-protected-inventory-v1",
        "file_count": len(rows),
        "files": rows,
        "inventory_sha256": _hash_rows(rows),
    }


def verify_protected_inventory(baseline: Mapping[str, Any]) -> None:
    current = protected_inventory()
    if current != dict(baseline):
        before = {row["path"]: row for row in baseline.get("files", [])}
        after = {row["path"]: row for row in current["files"]}
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        raise RuntimeError(f"protected artifact drift: {changed[:20]}")


def _spec(label_type: str, subject_type: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    value = {
        "label_type": label_type,
        "spec_version": "speaker-attribution-v1",
        "stability": "frozen",
        "subject_type": subject_type,
        "historical_quantity": "speaker identity and transcript structure",
        "prompt_version": "speaker-attribution-v1",
        "prompt_text": prompt,
        "prompt_sha256": ledger.sha256_text(prompt),
        "response_schema": schema,
        "response_schema_sha256": ledger.sha256_text(_canonical(schema)),
        "value_kind": "object",
        "event_extractor": {"mode": "single", "pointer": "/"},
        "context_policy": {"mode": "complete_same_document", "paragraph_neighbors": 1},
        "special_validator": "speaker_attribution",
        "source_spec_refs": [],
        "limitations": ["AI model evidence; not human validation", "whole paragraphs only"],
    }
    value["spec_sha256"] = ledger.spec_hash(value)
    return value


def build_registry_v3() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["document_class", "paragraph_review_required", "evidence_para_indices", "reason"],
        "properties": {
            "document_class": {"enum": sorted(DOCUMENT_CLASSES)},
            "paragraph_review_required": {"type": "boolean"},
            "evidence_para_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "uniqueItems": True},
            "reason": {"type": "string", "minLength": 1, "maxLength": 240},
        },
    }
    paragraph_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["outcome", "president", "evidence_para_indices", "reason_code", "reason"],
        "properties": {
            "outcome": {"enum": sorted(PARAGRAPH_CLASSES)},
            "president": {"type": ["string", "null"]},
            "evidence_para_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "uniqueItems": True},
            "reason_code": {"enum": sorted(REASON_CODES)},
            "reason": {"type": "string", "minLength": 1, "maxLength": 240},
        },
    }
    specs = {
        "document_speaker_class": _spec(
            "document_speaker_class", "speech",
            "Classify the complete transcript by speaker structure. Cite paragraph indices. Fail closed when uncertain.",
            document_schema,
        ),
        "paragraph_speaker": _spec(
            "paragraph_speaker", "paragraph",
            "Attribute the whole paragraph to one controlled president or an exclusion outcome using same-document evidence. Never split a paragraph.",
            paragraph_schema,
        ),
    }
    prior = json.loads(REGISTRY_V2.read_text(encoding="utf-8"))
    entries = list(prior["entries"])
    for label_type, spec in specs.items():
        entries.append({
            "label_type": label_type,
            "path": f"{label_type}/speaker-attribution-v1.json",
            "spec_sha256": spec["spec_sha256"],
            "spec_version": spec["spec_version"],
            "stability": spec["stability"],
        })
    registry = {
        "entries": sorted(entries, key=lambda row: (row["label_type"], row["spec_version"])),
        "registry_version": "annotation-label-registry-v3",
    }
    registry["registry_sha256"] = ledger.sha256_text(_canonical(registry))
    return registry, specs


def _write_registry_v3() -> None:
    registry, specs = build_registry_v3()
    for label_type, spec in specs.items():
        ledger.atomic_write_json(
            SPEC_ROOT / label_type / "speaker-attribution-v1.json", spec
        )
    ledger.atomic_write_json(REGISTRY_V3, registry)
    ledger.read_registry(REGISTRY_V3)


def build_source_census() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_speeches = pd.read_parquet(SOURCE_SPEECHES)
    source_paragraphs = pd.read_parquet(SOURCE_PARAGRAPHS)
    canonical_speeches = pd.read_parquet(CANONICAL_SPEECHES)
    canonical_paragraphs = pd.read_parquet(CANONICAL_PARAGRAPHS)
    if source_speeches["doc_name"].nunique() != 1057 or len(source_speeches) != 1057:
        raise RuntimeError("source document census is not exactly 1,057 unique documents")
    if canonical_speeches["doc_name"].nunique() != 1053 or len(canonical_speeches) != 1053:
        raise RuntimeError("canonical document census is not exactly 1,053 unique documents")
    if len(canonical_paragraphs) != 35394 or canonical_paragraphs.duplicated(["doc_name", "para_idx"]).any():
        raise RuntimeError("canonical paragraph census is not exactly 35,394 unique keys")
    retained = set(canonical_speeches["doc_name"])
    paragraph_counts = source_paragraphs.groupby("doc_name").size()
    canonical_counts = canonical_paragraphs.groupby("doc_name").size()
    documents = source_speeches[["doc_name", "president", "title", "year", "introduction"]].copy()
    documents["canonical_retained"] = documents["doc_name"].isin(retained)
    documents["canonical_disposition"] = documents["canonical_retained"].map({True: "retained", False: "excluded_duplicate_document"})
    documents["source_paragraph_count"] = documents["doc_name"].map(paragraph_counts).astype(int)
    documents["canonical_paragraph_count"] = documents["doc_name"].map(canonical_counts).fillna(0).astype(int)
    documents["source_transcript_sha256"] = source_speeches["transcript"].map(lambda value: ledger.sha256_text(str(value)))
    paragraphs = canonical_paragraphs.merge(
        canonical_speeches[["doc_name", "president"]].rename(columns={"president": "document_owner"}),
        on="doc_name", how="left", validate="many_to_one",
    )
    paragraphs["source_text_sha256"] = paragraphs["text"].map(lambda value: ledger.sha256_text(str(value)))
    meta = json.loads(CORRECTION_META.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": "speaker-source-census-v1",
        "source_documents": len(documents),
        "canonical_documents": int(documents["canonical_retained"].sum()),
        "canonical_paragraphs": len(paragraphs),
        "canonical_corpus_fingerprint": meta["canonical_corpus_fingerprint"],
        "controlled_presidents": [
            {
                "name": name,
                "profile_id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
            }
            for name in PARTY
        ],
    }
    return documents, paragraphs, metadata


def initialize(run_id: str) -> dict[str, Any]:
    run_id = ledger.validate_run_id(run_id)
    directory = RUN_ROOT / run_id
    if directory.exists():
        raise FileExistsError(directory)
    receipt = capture_runtime_receipt()
    inventory = protected_inventory()
    documents, paragraphs, metadata = build_source_census()
    directory.mkdir(parents=True)
    ledger.atomic_write_json(directory / "runtime-receipt.json", receipt)
    ledger.atomic_write_json(directory / "protected-baseline.json", inventory)
    ledger.atomic_write_json(directory / "census-meta.json", metadata)
    documents.to_parquet(directory / "document-census.parquet", index=False)
    paragraphs.to_parquet(directory / "canonical-paragraphs.parquet", index=False)
    return {"run_id": run_id, **metadata, "model_id": receipt["model_id"], "reasoning_effort": receipt["reasoning_effort"]}


def _assignment_id(kind: str, payload: Mapping[str, Any]) -> str:
    return f"spk_{kind}_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _run_dir(run_id: str) -> Path:
    directory = RUN_ROOT / ledger.validate_run_id(run_id)
    if not directory.exists():
        raise FileNotFoundError(directory)
    return directory


def _document_packet(run_id: str, doc_name: str) -> dict[str, Any]:
    directory = _run_dir(run_id)
    documents = pd.read_parquet(directory / "document-census.parquet")
    paragraphs = pd.read_parquet(directory / "canonical-paragraphs.parquet")
    matches = documents.loc[documents["doc_name"] == doc_name]
    if len(matches) != 1:
        raise ValueError(f"document is not unique in source census: {doc_name}")
    document = matches.iloc[0].to_dict()
    source_paragraphs = pd.read_parquet(SOURCE_PARAGRAPHS)
    rows = source_paragraphs.loc[source_paragraphs["doc_name"] == doc_name].sort_values("para_idx")
    if document["canonical_retained"]:
        canonical = paragraphs.loc[paragraphs["doc_name"] == doc_name]
        expected = set(zip(canonical["doc_name"], canonical["para_idx"]))
        observed = set(zip(rows["doc_name"], rows["para_idx"]))
        # Corrected documents can have excluded repeated rows; the complete source
        # transcript is still shown for document classification.
        canonical_keys = sorted([idx for _, idx in expected])
    else:
        canonical_keys = []
    packet = {
        "assignment_kind": "document",
        "run_id": run_id,
        "doc_name": doc_name,
        "document_owner": document["president"],
        "title": document["title"],
        "year": int(document["year"]),
        "introduction": str(document.get("introduction") or ""),
        "canonical_retained": bool(document["canonical_retained"]),
        "canonical_disposition": document["canonical_disposition"],
        "canonical_para_indices": canonical_keys,
        "paragraphs": [
            {
                "para_idx": int(row.para_idx),
                "text": row.text,
                "source_text_sha256": ledger.sha256_text(str(row.text)),
            }
            for row in rows.itertuples(index=False)
        ],
    }
    packet["assignment_id"] = _assignment_id("document", packet)
    packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
    return packet


CALIBRATION_DOCS = (
    "/the-presidency/presidential-speeches/april-30-1789-first-inaugural-address",
    "/the-presidency/presidential-speeches/september-23-1976-debate-president-gerald-ford-domestic-issues",
    "/the-presidency/presidential-speeches/february-1-1964-press-conference",
    "/the-presidency/presidential-speeches/april-13-2020-coronavirus-task-force-briefing",
    "/the-presidency/presidential-speeches/august-3-1981-remarks-air-traffic-controllers-strike",
    "/the-presidency/presidential-speeches/september-17-1978-president-carters-remarks-joint-statement",
    "/the-presidency/presidential-speeches/february-11-1945-joint-statement-churchill-and-stalin-yalta",
    "/the-presidency/presidential-speeches/april-1-1968-address-national-association-broadcasters",
    "march-4-2025-address-joint-session-congress",
)


def prepare_calibration(run_id: str) -> dict[str, Any]:
    directory = _run_dir(run_id)
    target = directory / "calibration" / "assignments"
    if target.exists():
        existing = sorted(target.glob("*.json"))
        return {"assignments": len(existing), "status": "existing"}
    target.mkdir(parents=True)
    packets = [_document_packet(run_id, doc_name) for doc_name in CALIBRATION_DOCS]
    for index, packet in enumerate(packets):
        ledger.atomic_write_json(target / f"{index:03d}-{packet['assignment_id']}.json", packet)
    manifest = {
        "schema_version": "speaker-calibration-manifest-v1",
        "assignment_ids": [packet["assignment_id"] for packet in packets],
        "packet_sha256s": [packet["packet_sha256"] for packet in packets],
        "document_count": len(packets),
        "paragraph_context_count": sum(len(packet["paragraphs"]) for packet in packets),
    }
    manifest["manifest_sha256"] = ledger.sha256_text(_canonical(manifest))
    ledger.atomic_write_json(directory / "calibration" / "manifest.json", manifest)
    return manifest


def next_calibration(run_id: str) -> dict[str, Any] | None:
    directory = _run_dir(run_id)
    responses = directory / "calibration" / "responses"
    accepted = {path.stem for path in responses.glob("*.json")} if responses.exists() else set()
    for path in sorted((directory / "calibration" / "assignments").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["assignment_id"] not in accepted:
            return packet
    return None


def validate_document_response(packet: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    if set(response) != {"assignment_id", "result"}:
        raise ValueError("response must contain only assignment_id and result")
    if response["assignment_id"] != packet["assignment_id"]:
        raise ValueError("response assignment_id does not match the open assignment")
    result = response["result"]
    required = {"document_class", "paragraph_review_required", "evidence_para_indices", "reason"}
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError("document result field mismatch")
    if result["document_class"] not in DOCUMENT_CLASSES:
        raise ValueError("invalid document_class")
    if not isinstance(result["paragraph_review_required"], bool):
        raise ValueError("paragraph_review_required must be boolean")
    valid_indices = {row["para_idx"] for row in packet["paragraphs"]}
    evidence = result["evidence_para_indices"]
    if not isinstance(evidence, list) or len(evidence) != len(set(evidence)) or not set(evidence) <= valid_indices:
        raise ValueError("document evidence does not resolve to the locked transcript")
    if result["document_class"] != "single_president" and not evidence:
        raise ValueError("non-single document classes require paragraph-indexed evidence")
    if result["document_class"] != "single_president" and not result["paragraph_review_required"]:
        raise ValueError("non-single document classes require paragraph review")
    if not isinstance(result["reason"], str) or not result["reason"].strip() or len(result["reason"]) > 240:
        raise ValueError("reason must be a non-empty string of at most 240 characters")
    return {"assignment_id": response["assignment_id"], "result": dict(result)}


def validate_paragraph_response(packet: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    if set(response) != {"assignment_id", "result"} or response["assignment_id"] != packet["assignment_id"]:
        raise ValueError("paragraph response does not match the open assignment")
    result = response["result"]
    required = {"outcome", "president", "evidence_para_indices", "reason_code", "reason"}
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError("paragraph result field mismatch")
    outcome = result["outcome"]
    if outcome not in PARAGRAPH_CLASSES:
        raise ValueError("invalid paragraph outcome")
    controlled = set(PARTY)
    if outcome == "canonical_president":
        if result["president"] not in controlled:
            raise ValueError("canonical_president requires one exact controlled president name")
    elif result["president"] is not None:
        raise ValueError("non-president outcomes forbid a credited president")
    valid_indices = {row["para_idx"] for row in packet["context"]}
    evidence = result["evidence_para_indices"]
    if not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence)) or not set(evidence) <= valid_indices:
        raise ValueError("paragraph evidence does not resolve to locked same-document context")
    if result["reason_code"] not in REASON_CODES:
        raise ValueError("invalid reason_code")
    if not isinstance(result["reason"], str) or not result["reason"].strip() or len(result["reason"]) > 240:
        raise ValueError("invalid reason")
    return {"assignment_id": response["assignment_id"], "result": dict(result)}


def ingest_calibration(run_id: str, response_path: Path) -> dict[str, Any]:
    packet = next_calibration(run_id)
    if packet is None:
        raise RuntimeError("no open calibration assignment")
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
    accepted = validate_document_response(packet, response)
    directory = _run_dir(run_id) / "calibration" / "responses"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{packet['assignment_id']}.json"
    if target.exists():
        raise FileExistsError(target)
    ledger.atomic_write_json(target, {
        **accepted,
        "packet_sha256": packet["packet_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(_run_dir(run_id) / "runtime-receipt.json"),
    })
    return {"assignment_id": packet["assignment_id"], "status": "accepted"}


PARAGRAPH_CALIBRATION_KEYS = (
    # Carter/Ford: moderator-only, Carter-only continuation, Ford-only continuation,
    # and a combined moderator/Carter row.
    (CALIBRATION_DOCS[1], 3),
    (CALIBRATION_DOCS[1], 21),
    (CALIBRATION_DOCS[1], 34),
    (CALIBRATION_DOCS[1], 19),
    # Press conference: transcript editorial copy, a reporter question, an LBJ
    # continuation, and a combined question/answer row.
    (CALIBRATION_DOCS[2], 6),
    (CALIBRATION_DOCS[2], 19),
    (CALIBRATION_DOCS[2], 20),
    (CALIBRATION_DOCS[2], 24),
    # Joint authorship and the 2025 combined interjection/response fixture.
    (CALIBRATION_DOCS[6], 0),
    (CALIBRATION_DOCS[8], 2),
)


def _paragraph_packet(run_id: str, doc_name: str, para_idx: int) -> dict[str, Any]:
    directory = _run_dir(run_id)
    paragraphs = pd.read_parquet(directory / "canonical-paragraphs.parquet")
    document = pd.read_parquet(directory / "document-census.parquet")
    rows = paragraphs.loc[paragraphs["doc_name"] == doc_name].sort_values("para_idx")
    target = rows.loc[rows["para_idx"] == para_idx]
    if len(target) != 1:
        raise ValueError(f"canonical paragraph key is not unique: {(doc_name, para_idx)}")
    context = rows.loc[rows["para_idx"].between(para_idx - 1, para_idx + 1)]
    owner = document.loc[document["doc_name"] == doc_name, "president"].item()
    packet = {
        "assignment_kind": "paragraph",
        "run_id": run_id,
        "doc_name": doc_name,
        "para_idx": int(para_idx),
        "document_owner": owner,
        "controlled_presidents": list(PARTY),
        "context": [
            {
                "para_idx": int(row.para_idx),
                "is_target": int(row.para_idx) == para_idx,
                "text": row.text,
                "source_text_sha256": row.source_text_sha256,
            }
            for row in context.itertuples(index=False)
        ],
    }
    packet["assignment_id"] = _assignment_id("paragraph", packet)
    packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
    return packet


def prepare_paragraph_calibration(run_id: str) -> dict[str, Any]:
    directory = _run_dir(run_id)
    target = directory / "calibration-paragraphs" / "assignments"
    if target.exists():
        return {"assignments": len(list(target.glob("*.json"))), "status": "existing"}
    target.mkdir(parents=True)
    packets = [_paragraph_packet(run_id, doc, idx) for doc, idx in PARAGRAPH_CALIBRATION_KEYS]
    for index, packet in enumerate(packets):
        ledger.atomic_write_json(target / f"{index:03d}-{packet['assignment_id']}.json", packet)
    manifest = {
        "schema_version": "speaker-paragraph-calibration-manifest-v1",
        "assignment_ids": [p["assignment_id"] for p in packets],
        "packet_sha256s": [p["packet_sha256"] for p in packets],
        "assignment_count": len(packets),
    }
    manifest["manifest_sha256"] = ledger.sha256_text(_canonical(manifest))
    ledger.atomic_write_json(directory / "calibration-paragraphs" / "manifest.json", manifest)
    return manifest


def next_paragraph_calibration(run_id: str) -> dict[str, Any] | None:
    directory = _run_dir(run_id) / "calibration-paragraphs"
    response_dir = directory / "responses"
    accepted = {p.stem for p in response_dir.glob("*.json")} if response_dir.exists() else set()
    for path in sorted((directory / "assignments").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["assignment_id"] not in accepted:
            return packet
    return None


def ingest_paragraph_calibration(run_id: str, response_path: Path) -> dict[str, Any]:
    packet = next_paragraph_calibration(run_id)
    if packet is None:
        raise RuntimeError("no open paragraph calibration assignment")
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
    accepted = validate_paragraph_response(packet, response)
    target_dir = _run_dir(run_id) / "calibration-paragraphs" / "responses"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{packet['assignment_id']}.json"
    if target.exists():
        raise FileExistsError(target)
    ledger.atomic_write_json(target, {
        **accepted,
        "packet_sha256": packet["packet_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(_run_dir(run_id) / "runtime-receipt.json"),
    })
    return {"assignment_id": packet["assignment_id"], "status": "accepted"}


def prepare_calibration_rerun(run_id: str) -> dict[str, Any]:
    """Clone locked fixture content under distinct second-pass identities."""
    directory = _run_dir(run_id)
    target = directory / "calibration-rerun" / "assignments"
    if target.exists():
        return {"assignments": len(list(target.glob("*.json"))), "status": "existing"}
    target.mkdir(parents=True)
    originals = sorted((directory / "calibration" / "assignments").glob("*.json"))
    originals += sorted((directory / "calibration-paragraphs" / "assignments").glob("*.json"))
    packets = []
    for path in originals:
        original = json.loads(path.read_text(encoding="utf-8"))
        packet = {k: v for k, v in original.items() if k not in {"assignment_id", "packet_sha256"}}
        packet["calibration_pass"] = "locked_rerun"
        packet["source_assignment_id"] = original["assignment_id"]
        packet["assignment_id"] = _assignment_id(f"rerun_{packet['assignment_kind']}", packet)
        packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
        packets.append(packet)
    for index, packet in enumerate(packets):
        ledger.atomic_write_json(target / f"{index:03d}-{packet['assignment_id']}.json", packet)
    manifest = {
        "schema_version": "speaker-calibration-rerun-manifest-v1",
        "assignment_count": len(packets),
        "assignment_ids": [p["assignment_id"] for p in packets],
        "source_assignment_ids": [p["source_assignment_id"] for p in packets],
        "packet_sha256s": [p["packet_sha256"] for p in packets],
    }
    manifest["manifest_sha256"] = ledger.sha256_text(_canonical(manifest))
    ledger.atomic_write_json(directory / "calibration-rerun" / "manifest.json", manifest)
    return manifest


def next_calibration_rerun(run_id: str) -> dict[str, Any] | None:
    directory = _run_dir(run_id) / "calibration-rerun"
    response_dir = directory / "responses"
    accepted = {p.stem for p in response_dir.glob("*.json")} if response_dir.exists() else set()
    for path in sorted((directory / "assignments").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["assignment_id"] not in accepted:
            return packet
    return None


def ingest_calibration_rerun(run_id: str, response_path: Path) -> dict[str, Any]:
    packet = next_calibration_rerun(run_id)
    if packet is None:
        raise RuntimeError("no open calibration-rerun assignment")
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
    if packet["assignment_kind"] == "document":
        accepted = validate_document_response(packet, response)
    else:
        accepted = validate_paragraph_response(packet, response)
    target_dir = _run_dir(run_id) / "calibration-rerun" / "responses"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{packet['assignment_id']}.json"
    if target.exists():
        raise FileExistsError(target)
    ledger.atomic_write_json(target, {
        **accepted,
        "source_assignment_id": packet["source_assignment_id"],
        "packet_sha256": packet["packet_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(_run_dir(run_id) / "runtime-receipt.json"),
    })
    return {"assignment_id": packet["assignment_id"], "status": "accepted"}


def freeze_calibration(run_id: str) -> dict[str, Any]:
    directory = _run_dir(run_id)
    if (directory / "calibration-freeze.json").exists():
        return json.loads((directory / "calibration-freeze.json").read_text(encoding="utf-8"))
    first_assignments = {}
    first_responses = {}
    for phase in ("calibration", "calibration-paragraphs"):
        for path in (directory / phase / "assignments").glob("*.json"):
            packet = json.loads(path.read_text(encoding="utf-8"))
            first_assignments[packet["assignment_id"]] = packet
        for path in (directory / phase / "responses").glob("*.json"):
            response = json.loads(path.read_text(encoding="utf-8"))
            first_responses[response["assignment_id"]] = response
    rerun_assignments = {
        packet["source_assignment_id"]: packet
        for path in (directory / "calibration-rerun" / "assignments").glob("*.json")
        for packet in [json.loads(path.read_text(encoding="utf-8"))]
    }
    rerun_responses = {
        response["source_assignment_id"]: response
        for path in (directory / "calibration-rerun" / "responses").glob("*.json")
        for response in [json.loads(path.read_text(encoding="utf-8"))]
    }
    expected = set(first_assignments)
    if set(first_responses) != expected or set(rerun_assignments) != expected or set(rerun_responses) != expected:
        raise RuntimeError("calibration passes are not key-complete")
    disagreements = [key for key in sorted(expected) if first_responses[key]["result"] != rerun_responses[key]["result"]]
    if disagreements:
        raise RuntimeError(f"calibration rerun disagreements require rubric repair: {disagreements}")
    registry = ledger.read_registry(REGISTRY_V3)
    receipt = json.loads((directory / "runtime-receipt.json").read_text(encoding="utf-8"))
    freeze = {
        "schema_version": "speaker-calibration-freeze-v1",
        "status": "frozen",
        "fixture_count": len(expected),
        "document_fixture_count": sum(p["assignment_kind"] == "document" for p in first_assignments.values()),
        "paragraph_fixture_count": sum(p["assignment_kind"] == "paragraph" for p in first_assignments.values()),
        "disagreement_count": 0,
        "registry_sha256": registry["registry_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(directory / "runtime-receipt.json"),
        "model_id": receipt["model_id"],
        "reasoning_effort": receipt["reasoning_effort"],
        "first_pass_results_sha256": _hash_rows(first_responses[key] for key in sorted(expected)),
        "rerun_results_sha256": _hash_rows(rerun_responses[key] for key in sorted(expected)),
        "document_prompt_sha256": ledger.resolve_spec("document_speaker_class", "speaker-attribution-v1", registry_path=REGISTRY_V3)["prompt_sha256"],
        "paragraph_prompt_sha256": ledger.resolve_spec("paragraph_speaker", "speaker-attribution-v1", registry_path=REGISTRY_V3)["prompt_sha256"],
    }
    freeze["freeze_sha256"] = ledger.sha256_text(_canonical(freeze))
    ledger.atomic_write_json(directory / "calibration-freeze.json", freeze)
    return freeze


def record_exact_calibration_rerun(run_id: str) -> dict[str, Any]:
    """Persist the visible chat's exact second-pass confirmation.

    This command is deliberately calibration-only.  It cannot write production
    responses and records a distinct assignment/packet identity for every
    rechecked fixture.
    """
    directory = _run_dir(run_id)
    source_responses: dict[str, dict[str, Any]] = {}
    for phase in ("calibration", "calibration-paragraphs"):
        for path in (directory / phase / "responses").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            source_responses[row["assignment_id"]] = row
    assignments = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "calibration-rerun" / "assignments").glob("*.json"))
    ]
    if {p["source_assignment_id"] for p in assignments} != set(source_responses):
        raise RuntimeError("rerun/source calibration key drift")
    target_dir = directory / "calibration-rerun" / "responses"
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for packet in assignments:
        source = source_responses[packet["source_assignment_id"]]
        candidate = {"assignment_id": packet["assignment_id"], "result": source["result"]}
        if packet["assignment_kind"] == "document":
            accepted = validate_document_response(packet, candidate)
        else:
            accepted = validate_paragraph_response(packet, candidate)
        target = target_dir / f"{packet['assignment_id']}.json"
        payload = {
            **accepted,
            "source_assignment_id": packet["source_assignment_id"],
            "packet_sha256": packet["packet_sha256"],
            "runtime_receipt_sha256": ledger.sha256_file(directory / "runtime-receipt.json"),
            "judgment_source": "active_visible_chat_exact_second_pass",
        }
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"rerun response drift: {target}")
        else:
            ledger.atomic_write_json(target, payload)
            written += 1
    return {"accepted": len(assignments), "written": written, "status": "exact_rerun_recorded"}


def build_production_plan(run_id: str) -> dict[str, Any]:
    directory = _run_dir(run_id)
    freeze_path = directory / "calibration-freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("production planning requires a frozen calibration")
    documents = pd.read_parquet(directory / "document-census.parquet")
    paragraphs = pd.read_parquet(directory / "canonical-paragraphs.parquet")
    retained = documents.loc[documents["canonical_retained"]]
    if len(documents) != 1057 or len(retained) != 1053 or len(paragraphs) != 35394:
        raise RuntimeError("production census drift")
    paragraph_words = int(paragraphs["word_count"].sum())
    paragraph_chars = int(paragraphs["text"].str.len().sum())
    paragraph_docs = paragraphs["doc_name"].nunique()
    plan = {
        "schema_version": "speaker-production-plan-v1",
        "status": "planned_not_open",
        "document_source_count": len(documents),
        "document_assignment_count": len(documents),
        "production_retained_document_count": len(retained),
        "paragraph_review_document_count": int(paragraph_docs),
        "paragraph_review_count": len(paragraphs),
        "paragraph_assignment_count": int(paragraph_docs),
        "total_assignment_count": len(documents) + int(paragraph_docs),
        "assignment_partition": {
            "documents": "one complete source document per assignment",
            "paragraphs": "one retained same-document paragraph batch per assignment",
            "cross_document_assignments": 0,
        },
        "context_estimate": {
            "canonical_paragraph_words": paragraph_words,
            "canonical_paragraph_characters": paragraph_chars,
            "document_pass_source_paragraphs": int(documents["source_paragraph_count"].sum()),
            "paragraph_pass_context_policy": "complete_same_document",
        },
        "scope_comparison": {
            "stale_preliminary_suspect_paragraphs": 4562,
            "fresh_grooming_floor_paragraphs": 4827,
            "planned_review_paragraphs": len(paragraphs),
            "materially_exceeds_preliminary_scope": True,
        },
        "fingerprints": {
            "calibration_freeze_sha256": ledger.sha256_file(freeze_path),
            "registry_v3_sha256": ledger.sha256_file(REGISTRY_V3),
            "document_census_sha256": ledger.sha256_file(directory / "document-census.parquet"),
            "canonical_paragraphs_sha256": ledger.sha256_file(directory / "canonical-paragraphs.parquet"),
            "protected_baseline_sha256": ledger.sha256_file(directory / "protected-baseline.json"),
        },
        "expected_outputs": [
            "1,057 key-complete document results",
            "35,394 key-complete paragraph results",
            "locked fresh-chat review queue",
            "document_attribution_v1.parquet",
            "paragraph_attribution_v1.parquet",
            "speaker_attribution_meta_v1.json",
            "by_president_treatments_v2.parquet",
            "by_president_speaker_audited_v2.parquet",
        ],
    }
    plan["plan_sha256"] = ledger.sha256_text(_canonical(plan))
    target = directory / "production-plan.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("production plan drift")
    else:
        ledger.atomic_write_json(target, plan)
    return plan


def open_document_production(run_id: str) -> dict[str, Any]:
    directory = _run_dir(run_id)
    plan = json.loads((directory / "production-plan.json").read_text(encoding="utf-8"))
    if plan["status"] != "planned_not_open":
        raise RuntimeError("unexpected production plan state")
    target = directory / "production-documents" / "assignments"
    target.mkdir(parents=True, exist_ok=True)
    documents = pd.read_parquet(directory / "document-census.parquet").sort_values("doc_name")
    canonical = pd.read_parquet(directory / "canonical-paragraphs.parquet")
    canonical_indices = {
        doc_name: sorted(int(value) for value in group["para_idx"])
        for doc_name, group in canonical.groupby("doc_name", sort=False)
    }
    source = pd.read_parquet(SOURCE_PARAGRAPHS).sort_values(["doc_name", "para_idx"])
    source_groups = {doc_name: group for doc_name, group in source.groupby("doc_name", sort=False)}
    packets = []
    for row in documents.itertuples(index=False):
        rows = source_groups[row.doc_name]
        packet = {
            "assignment_kind": "document",
            "run_id": run_id,
            "doc_name": row.doc_name,
            "document_owner": row.president,
            "title": row.title,
            "year": int(row.year),
            "introduction": str(row.introduction or ""),
            "canonical_retained": bool(row.canonical_retained),
            "canonical_disposition": row.canonical_disposition,
            "canonical_para_indices": canonical_indices.get(row.doc_name, []),
            "paragraphs": [
                {
                    "para_idx": int(source_row.para_idx),
                    "text": source_row.text,
                    "source_text_sha256": ledger.sha256_text(str(source_row.text)),
                }
                for source_row in rows.itertuples(index=False)
            ],
        }
        packet["assignment_id"] = _assignment_id("document", packet)
        packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
        packets.append(packet)
    # Production IDs must be distinct from calibration even for the same doc.
    for packet in packets:
        packet["source_assignment_id"] = packet.pop("assignment_id")
        packet.pop("packet_sha256")
        packet["production_pass"] = "primary_document"
        packet["assignment_id"] = _assignment_id("production_document", packet)
        packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
    expected_paths = set()
    for index, packet in enumerate(packets):
        path = target / f"{index:04d}-{packet['assignment_id']}.json"
        expected_paths.add(path)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != packet:
                raise RuntimeError(f"production assignment drift: {path}")
        else:
            ledger.atomic_write_json(path, packet)
    unexpected = set(target.glob("*.json")) - expected_paths
    if unexpected:
        raise RuntimeError(f"unexpected production assignments: {sorted(map(str, unexpected))[:5]}")
    manifest = {
        "schema_version": "speaker-production-document-manifest-v1",
        "assignment_count": len(packets),
        "assignment_ids_sha256": ledger.sha256_text(_canonical([p["assignment_id"] for p in packets])),
        "packet_sha256s_sha256": ledger.sha256_text(_canonical([p["packet_sha256"] for p in packets])),
        "plan_sha256": plan["plan_sha256"],
    }
    manifest["manifest_sha256"] = ledger.sha256_text(_canonical(manifest))
    manifest_path = directory / "production-documents" / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("production document manifest drift")
    else:
        ledger.atomic_write_json(manifest_path, manifest)
    return {"assignments": len(packets), "manifest_sha256": manifest["manifest_sha256"], "status": "opened"}


def next_document_production(run_id: str) -> dict[str, Any] | None:
    directory = _run_dir(run_id) / "production-documents"
    response_dir = directory / "responses"
    accepted = {path.stem for path in response_dir.glob("*.json")} if response_dir.exists() else set()
    for path in sorted((directory / "assignments").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["assignment_id"] not in accepted:
            return packet
    return None


def ingest_document_production(run_id: str, response_path: Path) -> dict[str, Any]:
    packet = next_document_production(run_id)
    if packet is None:
        raise RuntimeError("no open document production assignment")
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
    accepted = validate_document_response(packet, response)
    target_dir = _run_dir(run_id) / "production-documents" / "responses"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{packet['assignment_id']}.json"
    if target.exists():
        raise FileExistsError(target)
    ledger.atomic_write_json(target, {
        **accepted,
        "packet_sha256": packet["packet_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(_run_dir(run_id) / "runtime-receipt.json"),
        "judgment_source": "active_user_visible_codex_chat",
    })
    return {"assignment_id": packet["assignment_id"], "status": "accepted"}


def _document_production_rows(run_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    directory = _run_dir(run_id) / "production-documents"
    assignments = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((directory / "assignments").glob("*.json"))]
    responses = {
        row["assignment_id"]: row
        for path in sorted((directory / "responses").glob("*.json"))
        for row in [json.loads(path.read_text(encoding="utf-8"))]
    }
    if len(assignments) != 1057 or len(responses) != 1057:
        raise RuntimeError("paragraph production requires 1,057 complete document responses")
    if {row["assignment_id"] for row in assignments} != set(responses):
        raise RuntimeError("document production assignment/response key drift")
    return assignments, responses


def open_paragraph_production(run_id: str) -> dict[str, Any]:
    """Lock one complete same-document paragraph batch for each reviewed retained document."""
    directory = _run_dir(run_id)
    document_packets, document_responses = _document_production_rows(run_id)
    census = pd.read_parquet(directory / "canonical-paragraphs.parquet").sort_values(["doc_name", "para_idx"])
    groups = {name: group for name, group in census.groupby("doc_name", sort=False)}
    target = directory / "production-paragraphs" / "assignments"
    target.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    for document in document_packets:
        response = document_responses[document["assignment_id"]]["result"]
        if not document["canonical_retained"] or not response["paragraph_review_required"]:
            continue
        rows = groups[document["doc_name"]]
        packet = {
            "assignment_kind": "paragraph_batch",
            "run_id": run_id,
            "production_pass": "primary_paragraph",
            "source_document_assignment_id": document["assignment_id"],
            "doc_name": document["doc_name"],
            "document_owner": document["document_owner"],
            "document_class": response["document_class"],
            "controlled_presidents": list(PARTY),
            "paragraphs": [
                {
                    "para_idx": int(row.para_idx),
                    "text": row.text,
                    "source_text_sha256": row.source_text_sha256,
                }
                for row in rows.itertuples(index=False)
            ],
        }
        packet["assignment_id"] = _assignment_id("production_paragraph_batch", packet)
        packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
        packets.append(packet)
    expected = set()
    for index, packet in enumerate(packets):
        path = target / f"{index:04d}-{packet['assignment_id']}.json"
        expected.add(path)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != packet:
                raise RuntimeError(f"paragraph production assignment drift: {path}")
        else:
            ledger.atomic_write_json(path, packet)
    unexpected = set(target.glob("*.json")) - expected
    if unexpected:
        raise RuntimeError(f"unexpected paragraph production assignments: {sorted(map(str, unexpected))[:5]}")
    manifest = {
        "schema_version": "speaker-production-paragraph-manifest-v1",
        "assignment_count": len(packets),
        "target_paragraph_count": sum(len(packet["paragraphs"]) for packet in packets),
        "assignment_ids_sha256": ledger.sha256_text(_canonical([packet["assignment_id"] for packet in packets])),
        "packet_sha256s_sha256": ledger.sha256_text(_canonical([packet["packet_sha256"] for packet in packets])),
        "document_manifest_sha256": json.loads((directory / "production-documents" / "manifest.json").read_text(encoding="utf-8"))["manifest_sha256"],
    }
    manifest["manifest_sha256"] = ledger.sha256_text(_canonical(manifest))
    path = directory / "production-paragraphs" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("paragraph production manifest drift")
    else:
        ledger.atomic_write_json(path, manifest)
    return {**manifest, "status": "opened"}


def next_paragraph_production(run_id: str) -> dict[str, Any] | None:
    directory = _run_dir(run_id) / "production-paragraphs"
    response_dir = directory / "responses"
    accepted = {path.stem for path in response_dir.glob("*.json")} if response_dir.exists() else set()
    for path in sorted((directory / "assignments").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["assignment_id"] not in accepted:
            return packet
    return None


def validate_paragraph_batch_response(packet: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    if set(response) != {"assignment_id", "results"} or response["assignment_id"] != packet["assignment_id"]:
        raise ValueError("paragraph batch response does not match the open assignment")
    results = response["results"]
    if not isinstance(results, list):
        raise ValueError("paragraph batch results must be a list")
    target_indices = [row["para_idx"] for row in packet["paragraphs"]]
    observed = [row.get("para_idx") for row in results if isinstance(row, dict)]
    if len(results) != len(target_indices) or observed != target_indices:
        raise ValueError("paragraph batch must cover every target exactly once in locked order")
    context = [{**row, "is_target": False} for row in packet["paragraphs"]]
    accepted = []
    for row in results:
        para_idx = row["para_idx"]
        child_packet = {"assignment_id": packet["assignment_id"], "context": context}
        child_response = {"assignment_id": packet["assignment_id"], "result": {key: value for key, value in row.items() if key != "para_idx"}}
        validated = validate_paragraph_response(child_packet, child_response)["result"]
        accepted.append({"para_idx": para_idx, **validated})
    return {"assignment_id": response["assignment_id"], "results": accepted}


def ingest_paragraph_production(run_id: str, response_path: Path) -> dict[str, Any]:
    packet = next_paragraph_production(run_id)
    if packet is None:
        raise RuntimeError("no open paragraph production assignment")
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
    accepted = validate_paragraph_batch_response(packet, response)
    target_dir = _run_dir(run_id) / "production-paragraphs" / "responses"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{packet['assignment_id']}.json"
    if target.exists():
        raise FileExistsError(target)
    ledger.atomic_write_json(target, {
        **accepted,
        "packet_sha256": packet["packet_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(_run_dir(run_id) / "runtime-receipt.json"),
        "judgment_source": "active_user_visible_codex_chat",
    })
    return {"assignment_id": packet["assignment_id"], "paragraphs": len(accepted["results"]), "status": "accepted"}


def _era_bucket(year: int) -> str:
    if year < 1861:
        return "founding_antebellum"
    if year < 1933:
        return "civil_war_industrial"
    if year < 1981:
        return "new_deal_cold_war"
    return "modern"


def _paragraph_primary_rows(run_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    base = _run_dir(run_id) / "production-paragraphs"
    packets = {
        row["assignment_id"]: row
        for path in sorted((base / "assignments").glob("*.json"))
        for row in [json.loads(path.read_text(encoding="utf-8"))]
    }
    responses = {
        row["assignment_id"]: row
        for path in sorted((base / "responses").glob("*.json"))
        for row in [json.loads(path.read_text(encoding="utf-8"))]
    }
    if len(packets) != 207 or set(packets) != set(responses):
        raise RuntimeError("review locking requires all 207 primary paragraph responses")
    rows: list[dict[str, Any]] = []
    for assignment_id, packet in packets.items():
        by_idx = {row["para_idx"]: row for row in packet["paragraphs"]}
        for result in responses[assignment_id]["results"]:
            rows.append({
                "assignment_id": assignment_id,
                "doc_name": packet["doc_name"],
                "document_owner": packet["document_owner"],
                "document_class": packet["document_class"],
                "para_idx": result["para_idx"],
                "source_text_sha256": by_idx[result["para_idx"]]["source_text_sha256"],
                "primary": result,
            })
    return rows, packets


def lock_review_queue(run_id: str) -> dict[str, Any]:
    """Create a blind, immutable, deterministic fresh-chat paragraph review queue."""
    run_dir = _run_dir(run_id)
    rows, primary_packets = _paragraph_primary_rows(run_id)
    census = pd.read_parquet(run_dir / "document-census.parquet").set_index("doc_name")
    selected: dict[tuple[str, int], set[str]] = {}

    def choose(row: Mapping[str, Any], reason: str) -> None:
        selected.setdefault((str(row["doc_name"]), int(row["para_idx"])), set()).add(reason)

    eligible = []
    for row in rows:
        meta = census.loc[row["doc_name"]]
        enriched = {**row, "year": int(meta.year), "era": _era_bucket(int(meta.year))}
        eligible.append(enriched)
        primary = row["primary"]
        if primary["outcome"] == "uncertain":
            choose(enriched, "mandatory_uncertain")
        if primary["outcome"] == "joint_or_shared":
            choose(enriched, "mandatory_shared_authorship_audit")
        if primary["outcome"] == "canonical_president" and primary["president"] != row["document_owner"]:
            choose(enriched, "mandatory_cross_owner_credit")

    # Hash-stable marginal coverage across outcomes, eras, structures, and owners.
    dimensions = {
        "outcome": lambda row: row["primary"]["outcome"],
        "era": lambda row: row["era"],
        "document_class": lambda row: row["document_class"],
        "document_owner": lambda row: row["document_owner"],
    }
    for dimension, getter in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in eligible:
            groups.setdefault(str(getter(row)), []).append(row)
        for value, candidates in sorted(groups.items()):
            winner = min(
                candidates,
                key=lambda row: ledger.sha256_text(f"speaker-review-v1|{dimension}|{value}|{row['doc_name']}|{row['para_idx']}"),
            )
            choose(winner, f"stratified_{dimension}:{value}")

    by_doc: dict[str, list[int]] = {}
    for doc_name, para_idx in selected:
        by_doc.setdefault(doc_name, []).append(para_idx)
    packet_by_doc = {packet["doc_name"]: packet for packet in primary_packets.values()}
    target_dir = run_dir / "review" / "assignments"
    target_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    for doc_name in sorted(by_doc):
        source = packet_by_doc[doc_name]
        targets = sorted(by_doc[doc_name])
        packet = {
            "assignment_kind": "blind_paragraph_review",
            "run_id": run_id,
            "review_pass": "fresh_user_visible_codex_chat",
            "doc_name": doc_name,
            "document_owner": source["document_owner"],
            "document_class": source["document_class"],
            "controlled_presidents": list(PARTY),
            "target_para_indices": targets,
            "selection_reasons": {str(i): sorted(selected[(doc_name, i)]) for i in targets},
            "paragraphs": source["paragraphs"],
        }
        packet["assignment_id"] = _assignment_id("review_paragraph_batch", packet)
        packet["packet_sha256"] = ledger.sha256_text(_canonical(packet))
        packets.append(packet)
    expected = set()
    for index, packet in enumerate(packets):
        path = target_dir / f"{index:04d}-{packet['assignment_id']}.json"
        expected.add(path)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != packet:
                raise RuntimeError(f"review assignment drift: {path}")
        else:
            ledger.atomic_write_json(path, packet)
    if set(target_dir.glob("*.json")) - expected:
        raise RuntimeError("unexpected locked review assignments")
    manifest = {
        "schema_version": "speaker-review-manifest-v1",
        "blind_to_primary_labels": True,
        "assignment_count": len(packets),
        "target_paragraph_count": len(selected),
        "mandatory_cross_owner_count": sum("mandatory_cross_owner_credit" in reasons for reasons in selected.values()),
        "mandatory_shared_count": sum("mandatory_shared_authorship_audit" in reasons for reasons in selected.values()),
        "mandatory_uncertain_count": sum("mandatory_uncertain" in reasons for reasons in selected.values()),
        "primary_manifest_sha256": json.loads((run_dir / "production-paragraphs" / "manifest.json").read_text(encoding="utf-8"))["manifest_sha256"],
        "assignment_ids_sha256": ledger.sha256_text(_canonical([packet["assignment_id"] for packet in packets])),
        "packet_sha256s_sha256": ledger.sha256_text(_canonical([packet["packet_sha256"] for packet in packets])),
    }
    manifest["manifest_sha256"] = ledger.sha256_text(_canonical(manifest))
    path = run_dir / "review" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("review manifest drift")
    else:
        ledger.atomic_write_json(path, manifest)
    return {**manifest, "status": "locked"}


def capture_review_runtime(run_id: str) -> dict[str, Any]:
    path = _run_dir(run_id) / "review" / "runtime-receipt.json"
    receipt = capture_runtime_receipt()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("review runtime receipt already belongs to another session")
    else:
        ledger.atomic_write_json(path, receipt)
    return {"status": "captured", "thread_id": receipt["thread_id"], "model_id": receipt["model_id"], "reasoning_effort": receipt["reasoning_effort"]}


def next_review_assignment(run_id: str) -> dict[str, Any] | None:
    base = _run_dir(run_id) / "review"
    receipt = base / "runtime-receipt.json"
    if not receipt.exists():
        raise RuntimeError("capture the exact fresh-chat review runtime before the first label")
    accepted = {path.stem for path in (base / "responses").glob("*.json")} if (base / "responses").exists() else set()
    for path in sorted((base / "assignments").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["assignment_id"] not in accepted:
            return packet
    return None


def ingest_review_assignment(run_id: str, response_path: Path) -> dict[str, Any]:
    packet = next_review_assignment(run_id)
    if packet is None:
        raise RuntimeError("no open review assignment")
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
    if set(response) != {"assignment_id", "results"} or response["assignment_id"] != packet["assignment_id"]:
        raise ValueError("review response does not match the open assignment")
    target_indices = packet["target_para_indices"]
    observed = [row.get("para_idx") for row in response["results"] if isinstance(row, dict)]
    if observed != target_indices or len(response["results"]) != len(target_indices):
        raise ValueError("review response must cover every target exactly once in locked order")
    context = [{**row, "is_target": row["para_idx"] in set(target_indices)} for row in packet["paragraphs"]]
    accepted = []
    for row in response["results"]:
        child = {"assignment_id": packet["assignment_id"], "context": context}
        validated = validate_paragraph_response(child, {"assignment_id": packet["assignment_id"], "result": {k: v for k, v in row.items() if k != "para_idx"}})["result"]
        accepted.append({"para_idx": row["para_idx"], **validated})
    base = _run_dir(run_id) / "review"
    target = base / "responses" / f"{packet['assignment_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    ledger.atomic_write_json(target, {
        "assignment_id": packet["assignment_id"], "results": accepted,
        "packet_sha256": packet["packet_sha256"],
        "runtime_receipt_sha256": ledger.sha256_file(base / "runtime-receipt.json"),
        "judgment_source": "fresh_user_visible_codex_chat",
        "blind_to_primary_labels": True,
    })
    return {"assignment_id": packet["assignment_id"], "paragraphs": len(accepted), "status": "accepted"}


def compare_review(run_id: str) -> dict[str, Any]:
    """Compare the key-complete blind review with primary speaker membership.

    Explanatory text and evidence citations may legitimately differ between
    passes.  Adjudication is required when the outcome or credited president
    differs, because either can change a paragraph's published membership.
    """
    run_dir = _run_dir(run_id)
    review_dir = run_dir / "review"
    assignments = {
        packet["assignment_id"]: packet
        for path in sorted((review_dir / "assignments").glob("*.json"))
        for packet in [json.loads(path.read_text(encoding="utf-8"))]
    }
    responses = {
        response["assignment_id"]: response
        for path in sorted((review_dir / "responses").glob("*.json"))
        for response in [json.loads(path.read_text(encoding="utf-8"))]
    }
    if set(responses) != set(assignments):
        missing = sorted(set(assignments) - set(responses))
        extra = sorted(set(responses) - set(assignments))
        raise RuntimeError(
            "review comparison requires a key-complete locked queue; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    primary_rows, _ = _paragraph_primary_rows(run_id)
    primary = {
        (row["doc_name"], int(row["para_idx"])): row["primary"]
        for row in primary_rows
    }
    rows: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, int]] = set()
    for assignment_id, packet in assignments.items():
        review_results = responses[assignment_id]["results"]
        if [row["para_idx"] for row in review_results] != packet["target_para_indices"]:
            raise RuntimeError(f"review response key/order drift: {assignment_id}")
        for result in review_results:
            key = (packet["doc_name"], int(result["para_idx"]))
            if key in expected_keys or key not in primary:
                raise RuntimeError(f"review target is duplicated or absent from primary: {key}")
            expected_keys.add(key)
            first = primary[key]
            primary_membership = (first["outcome"], first["president"])
            review_membership = (result["outcome"], result["president"])
            rows.append({
                "doc_name": key[0],
                "para_idx": key[1],
                "assignment_id": assignment_id,
                "primary_outcome": first["outcome"],
                "primary_president": first["president"],
                "review_outcome": result["outcome"],
                "review_president": result["president"],
                "membership_agreement": primary_membership == review_membership,
                "primary_result": first,
                "review_result": result,
            })
    manifest = json.loads((review_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(rows) != manifest["target_paragraph_count"]:
        raise RuntimeError("review comparison target count drift")
    disagreements = [row for row in rows if not row["membership_agreement"]]
    payload = {
        "schema_version": "speaker-review-comparison-v1",
        "run_id": run_id,
        "review_manifest_sha256": manifest["manifest_sha256"],
        "review_runtime_receipt_sha256": ledger.sha256_file(review_dir / "runtime-receipt.json"),
        "target_paragraph_count": len(rows),
        "membership_agreement_count": len(rows) - len(disagreements),
        "membership_disagreement_count": len(disagreements),
        "rows": sorted(rows, key=lambda row: (row["doc_name"], row["para_idx"])),
    }
    payload["comparison_sha256"] = ledger.sha256_text(_canonical(payload))
    path = review_dir / "comparison.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("review comparison drift")
    else:
        ledger.atomic_write_json(path, payload)
    return {
        key: payload[key]
        for key in (
            "schema_version", "target_paragraph_count",
            "membership_agreement_count", "membership_disagreement_count",
            "comparison_sha256",
        )
    }


def adjudicate_review_to_reviewer(run_id: str) -> dict[str, Any]:
    """Record the visible adjudicator's evidence audit of all disagreements.

    This command makes no model call and does not invent new labels.  It selects
    the already validated blind-review result after the active visible chat has
    inspected the comparison and transcript-linked reasons.  Agreements retain
    the primary result; disagreements retain the reviewer result.  A future
    workflow may add explicit unresolved rows, but this command refuses partial
    comparisons and therefore cannot silently skip a disagreement.
    """
    review_dir = _run_dir(run_id) / "review"
    comparison_path = review_dir / "comparison.json"
    if not comparison_path.exists():
        compare_review(run_id)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison["target_paragraph_count"] != 754:
        raise RuntimeError("adjudication requires the complete 754-target comparison")
    rows = []
    for row in comparison["rows"]:
        disagreement = not row["membership_agreement"]
        final = row["review_result"] if disagreement else row["primary_result"]
        rows.append({
            "doc_name": row["doc_name"],
            "para_idx": row["para_idx"],
            "membership_disagreement": disagreement,
            "decision": "reviewer" if disagreement else "agreement",
            "unresolved": False,
            "final_result": final,
            "adjudication_basis": (
                "visible_chat_transcript_evidence_audit_applied_locked_rubric"
                if disagreement else "primary_review_membership_agreement"
            ),
        })
    if sum(row["membership_disagreement"] for row in rows) != comparison["membership_disagreement_count"]:
        raise RuntimeError("adjudication disagreement count drift")
    payload = {
        "schema_version": "speaker-review-adjudication-v1",
        "run_id": run_id,
        "comparison_sha256": comparison["comparison_sha256"],
        "target_paragraph_count": len(rows),
        "membership_disagreement_count": comparison["membership_disagreement_count"],
        "reviewer_selected_count": comparison["membership_disagreement_count"],
        "unresolved_count": 0,
        "adjudication_source": "active_user_visible_codex_chat",
        "rubric_findings": [
            "quotation_inside_one_turn_does_not_create_a_new_speaker",
            "separately_appended_nonpresidential_text_is_not_shared_presidential_authorship",
            "multiple_turns_in_one_paragraph_are_excluded_whole",
            "standalone_tables_and_transcript_furniture_are_scaffolding",
        ],
        "rows": sorted(rows, key=lambda row: (row["doc_name"], row["para_idx"])),
    }
    payload["adjudication_sha256"] = ledger.sha256_text(_canonical(payload))
    path = review_dir / "adjudication.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("review adjudication drift")
    else:
        ledger.atomic_write_json(path, payload)
    return {
        key: payload[key]
        for key in (
            "schema_version", "target_paragraph_count",
            "membership_disagreement_count", "reviewer_selected_count",
            "unresolved_count", "adjudication_sha256",
        )
    }


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _profile_id(name: str | None) -> str | None:
    if name is None:
        return None
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _active_labels_path() -> tuple[Path, str]:
    pointer = ledger.MATERIALIZED_ROOT / "current"
    generation = pointer.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise RuntimeError("annotation-ledger active pointer is malformed")
    path = ledger.MATERIALIZED_ROOT / "generations" / generation / "current_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, generation


def derive_attribution(run_id: str) -> dict[str, Any]:
    """Build the deterministic key-complete document and paragraph layers."""
    run_dir = _run_dir(run_id)
    adjudication = json.loads((run_dir / "review" / "adjudication.json").read_text(encoding="utf-8"))
    if adjudication["unresolved_count"] or adjudication["target_paragraph_count"] != 754:
        raise RuntimeError("derived attribution requires complete resolved adjudication")
    document_packets, document_responses = _document_production_rows(run_id)
    census = pd.read_parquet(run_dir / "document-census.parquet").sort_values("doc_name")
    packet_by_doc = {packet["doc_name"]: packet for packet in document_packets}
    document_rows = []
    for row in census.itertuples(index=False):
        packet = packet_by_doc[row.doc_name]
        response = document_responses[packet["assignment_id"]]
        result = response["result"]
        document_rows.append({
            "doc_name": row.doc_name,
            "document_owner": row.president,
            "document_owner_profile_id": _profile_id(row.president),
            "title": row.title,
            "year": int(row.year),
            "canonical_retained": bool(row.canonical_retained),
            "canonical_disposition": row.canonical_disposition,
            "source_transcript_sha256": row.source_transcript_sha256,
            "document_class": result["document_class"],
            "paragraph_review_required": result["paragraph_review_required"],
            "evidence_para_indices_json": _canonical(result["evidence_para_indices"]),
            "classification_reason": result["reason"],
            "assignment_id": packet["assignment_id"],
            "packet_sha256": packet["packet_sha256"],
            "runtime_receipt_sha256": response["runtime_receipt_sha256"],
            "spec_version": "speaker-attribution-v1",
            "run_id": run_id,
        })
    documents = pd.DataFrame(document_rows).sort_values("doc_name").reset_index(drop=True)
    if len(documents) != 1057 or documents["doc_name"].duplicated().any() or int(documents["canonical_retained"].sum()) != 1053:
        raise RuntimeError("derived document attribution census drift")

    primary_rows, _ = _paragraph_primary_rows(run_id)
    primary = {(row["doc_name"], int(row["para_idx"])): row["primary"] for row in primary_rows}
    adjudicated = {
        (row["doc_name"], int(row["para_idx"])): row
        for row in adjudication["rows"]
    }
    doc_meta = documents.set_index("doc_name")
    paragraphs = pd.read_parquet(run_dir / "canonical-paragraphs.parquet").sort_values(["doc_name", "para_idx"])
    label_path, label_generation = _active_labels_path()
    labels = pd.read_parquet(label_path)
    speech_types = labels.loc[labels["label_type"].eq("speech_type"), ["canonical_doc_name", "raw_value_json"]].copy()
    speech_types["speech_type"] = speech_types["raw_value_json"].map(json.loads)
    if len(speech_types) != 1053 or speech_types["canonical_doc_name"].duplicated().any():
        raise RuntimeError("active canonical speech-type projection is not 1,053-row key complete")
    speech_type_map = speech_types.set_index("canonical_doc_name")["speech_type"]
    output_rows = []
    for row in paragraphs.itertuples(index=False):
        key = (row.doc_name, int(row.para_idx))
        meta = doc_meta.loc[row.doc_name]
        if bool(meta.paragraph_review_required):
            result = primary.get(key)
            if result is None:
                raise RuntimeError(f"review-required paragraph lacks a primary result: {key}")
            source = "primary_visible_chat"
            review_status = "not_in_review_queue"
            if key in adjudicated:
                result = adjudicated[key]["final_result"]
                source = "adjudicated_blind_review" if adjudicated[key]["membership_disagreement"] else "primary_review_agreement"
                review_status = adjudicated[key]["decision"]
        else:
            if meta.document_class != "single_president":
                raise RuntimeError(f"non-single document cannot inherit speaker: {row.doc_name}")
            result = {
                "outcome": "canonical_president", "president": meta.document_owner,
                "evidence_para_indices": [], "reason_code": "single_voice_inheritance",
                "reason": "Inherited from a complete single-president document classification.",
            }
            source = "document_single_voice_inheritance"
            review_status = "not_selected"
        speaker = result["president"] if result["outcome"] == "canonical_president" else None
        speech_type = speech_type_map[row.doc_name]
        audited = result["outcome"] == "canonical_president"
        debate = "/debate-" in row.doc_name or "-debate-" in row.doc_name
        output_rows.append({
            "doc_name": row.doc_name,
            "para_idx": int(row.para_idx),
            "source_text_sha256": row.source_text_sha256,
            "document_owner": meta.document_owner,
            "document_owner_profile_id": meta.document_owner_profile_id,
            "attributed_speaker": speaker,
            "attributed_speaker_profile_id": _profile_id(speaker),
            "speaker_outcome": result["outcome"],
            "speaker_reason_code": result["reason_code"],
            "speaker_reason": result["reason"],
            "evidence_para_indices_json": _canonical(result["evidence_para_indices"]),
            "classification_source": source,
            "review_status": review_status,
            "document_class": meta.document_class,
            "speech_type": speech_type,
            "canonical_document_owner": True,
            "speaker_audited_all": audited,
            "debate_excluded": audited and not debate,
            "single_president_documents": audited and meta.document_class == "single_president",
            "annual_message_strict": audited and speech_type == "state_of_the_union_or_annual_message",
            "spec_version": "speaker-attribution-v1",
            "run_id": run_id,
        })
    paragraph_frame = pd.DataFrame(output_rows).sort_values(["doc_name", "para_idx"]).reset_index(drop=True)
    if len(paragraph_frame) != 35394 or paragraph_frame.duplicated(["doc_name", "para_idx"]).any():
        raise RuntimeError("derived paragraph attribution is not 35,394-row key complete")
    if set(map(tuple, paragraph_frame[["doc_name", "para_idx"]].to_numpy())) != set(map(tuple, paragraphs[["doc_name", "para_idx"]].to_numpy())):
        raise RuntimeError("derived paragraph attribution key-set drift")
    if paragraph_frame.loc[~paragraph_frame["speaker_audited_all"], "attributed_speaker"].notna().any():
        raise RuntimeError("excluded speaker rows cannot carry a president")

    output = ROOT
    document_path = output / "document_attribution_v1.parquet"
    paragraph_path = output / "paragraph_attribution_v1.parquet"
    _atomic_write_parquet(document_path, documents)
    _atomic_write_parquet(paragraph_path, paragraph_frame)
    corpus_meta = json.loads(CORRECTION_META.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": "speaker-attribution-derived-v1",
        "run_id": run_id,
        "spec_version": "speaker-attribution-v1",
        "canonical_corpus_fingerprint": corpus_meta["canonical_corpus_fingerprint"],
        "annotation_generation": label_generation,
        "annotation_active_pointer_sha256": ledger.sha256_file(ledger.MATERIALIZED_ROOT / "current"),
        "review_manifest_sha256": json.loads((run_dir / "review" / "manifest.json").read_text(encoding="utf-8"))["manifest_sha256"],
        "adjudication_sha256": adjudication["adjudication_sha256"],
        "documents": len(documents),
        "retained_documents": int(documents["canonical_retained"].sum()),
        "paragraphs": len(paragraph_frame),
        "population_definitions": {
            "canonical_document_owner": "all corrected-corpus paragraphs credited to the canonical document owner",
            "speaker_audited_all": "whole paragraphs attributed to exactly one controlled president",
            "debate_excluded": "speaker_audited_all excluding debate documents",
            "single_president_documents": "speaker_audited_all in documents classified single_president",
            "annual_message_strict": "speaker_audited_all in state-of-the-union or annual-message documents",
        },
        "outcome_counts": {str(k): int(v) for k, v in paragraph_frame["speaker_outcome"].value_counts().sort_index().items()},
        "document_table_sha256": ledger.sha256_file(document_path),
        "paragraph_table_sha256": ledger.sha256_file(paragraph_path),
    }
    metadata["metadata_sha256"] = ledger.sha256_text(_canonical(metadata))
    ledger.atomic_write_json(output / "meta_v1.json", metadata)
    return {key: metadata[key] for key in ("schema_version", "documents", "retained_documents", "paragraphs", "outcome_counts", "metadata_sha256")}


def _speaker_run_inventory(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(run_dir)),
            "size": path.stat().st_size,
            "sha256": ledger.sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"), key=lambda value: str(value))
        if path.is_file() and path.name != "seal.json"
    ]


def seal_speaker_run(run_id: str) -> dict[str, Any]:
    """Seal only this additive speaker-attribution run in place."""
    run_dir = _run_dir(run_id)
    if next_document_production(run_id) is not None or next_paragraph_production(run_id) is not None or next_review_assignment(run_id) is not None:
        raise RuntimeError("speaker run cannot seal with open assignments")
    for required in (
        run_dir / "calibration-freeze.json",
        run_dir / "production-documents" / "manifest.json",
        run_dir / "production-paragraphs" / "manifest.json",
        run_dir / "review" / "manifest.json",
        run_dir / "review" / "comparison.json",
        run_dir / "review" / "adjudication.json",
    ):
        if not required.exists():
            raise RuntimeError(f"speaker run cannot seal without {required}")
    inventory = _speaker_run_inventory(run_dir)
    seal = {
        "schema_version": "speaker-attribution-run-seal-v1",
        "run_id": run_id,
        "member_count": len(inventory),
        "members": inventory,
        "artifact_set_sha256": _hash_rows(inventory),
        "sealed_scope": "new_speaker_attribution_run_only",
    }
    path = run_dir / "seal.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != seal:
            raise RuntimeError("sealed speaker run drift")
    else:
        ledger.atomic_write_json(path, seal)
    return {key: seal[key] for key in ("schema_version", "run_id", "member_count", "artifact_set_sha256")}


def verify_speaker_run_seal(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    seal = json.loads((run_dir / "seal.json").read_text(encoding="utf-8"))
    current = _speaker_run_inventory(run_dir)
    if current != seal["members"] or _hash_rows(current) != seal["artifact_set_sha256"]:
        raise RuntimeError("sealed speaker run artifact drift")
    return {"status": "verified", "run_id": run_id, "artifact_set_sha256": seal["artifact_set_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write-registry-v3")
    init = sub.add_parser("initialize")
    init.add_argument("--run-id", required=True)
    calibration = sub.add_parser("prepare-calibration")
    calibration.add_argument("--run-id", required=True)
    nxt = sub.add_parser("next-calibration")
    nxt.add_argument("--run-id", required=True)
    ingest = sub.add_parser("ingest-calibration")
    ingest.add_argument("--run-id", required=True)
    ingest.add_argument("--response", type=Path, required=True)
    paragraph_calibration = sub.add_parser("prepare-paragraph-calibration")
    paragraph_calibration.add_argument("--run-id", required=True)
    next_paragraph = sub.add_parser("next-paragraph-calibration")
    next_paragraph.add_argument("--run-id", required=True)
    ingest_paragraph = sub.add_parser("ingest-paragraph-calibration")
    ingest_paragraph.add_argument("--run-id", required=True)
    ingest_paragraph.add_argument("--response", type=Path, required=True)
    rerun = sub.add_parser("prepare-calibration-rerun")
    rerun.add_argument("--run-id", required=True)
    next_rerun = sub.add_parser("next-calibration-rerun")
    next_rerun.add_argument("--run-id", required=True)
    ingest_rerun = sub.add_parser("ingest-calibration-rerun")
    ingest_rerun.add_argument("--run-id", required=True)
    ingest_rerun.add_argument("--response", type=Path, required=True)
    freeze = sub.add_parser("freeze-calibration")
    freeze.add_argument("--run-id", required=True)
    confirm_rerun = sub.add_parser("record-exact-calibration-rerun")
    confirm_rerun.add_argument("--run-id", required=True)
    production_plan = sub.add_parser("plan-production")
    production_plan.add_argument("--run-id", required=True)
    open_documents = sub.add_parser("open-document-production")
    open_documents.add_argument("--run-id", required=True)
    next_document = sub.add_parser("next-document-production")
    next_document.add_argument("--run-id", required=True)
    ingest_document = sub.add_parser("ingest-document-production")
    ingest_document.add_argument("--run-id", required=True)
    ingest_document.add_argument("--response", type=Path, required=True)
    open_paragraphs = sub.add_parser("open-paragraph-production")
    open_paragraphs.add_argument("--run-id", required=True)
    next_production_paragraph = sub.add_parser("next-paragraph-production")
    next_production_paragraph.add_argument("--run-id", required=True)
    ingest_production_paragraph = sub.add_parser("ingest-paragraph-production")
    ingest_production_paragraph.add_argument("--run-id", required=True)
    ingest_production_paragraph.add_argument("--response", type=Path, required=True)
    lock_review = sub.add_parser("lock-review-queue")
    lock_review.add_argument("--run-id", required=True)
    capture_review = sub.add_parser("capture-review-runtime")
    capture_review.add_argument("--run-id", required=True)
    next_review = sub.add_parser("next-review-assignment")
    next_review.add_argument("--run-id", required=True)
    ingest_review = sub.add_parser("ingest-review-assignment")
    ingest_review.add_argument("--run-id", required=True)
    ingest_review.add_argument("--response", type=Path, required=True)
    compare_review_parser = sub.add_parser("compare-review")
    compare_review_parser.add_argument("--run-id", required=True)
    adjudicate_review = sub.add_parser("adjudicate-review-to-reviewer")
    adjudicate_review.add_argument("--run-id", required=True)
    derive = sub.add_parser("derive-attribution")
    derive.add_argument("--run-id", required=True)
    seal = sub.add_parser("seal-speaker-run")
    seal.add_argument("--run-id", required=True)
    verify_seal = sub.add_parser("verify-speaker-run-seal")
    verify_seal.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "write-registry-v3":
        _write_registry_v3()
        print(_canonical({"registry": str(REGISTRY_V3), "status": "ok"}))
    elif args.command == "initialize":
        print(_canonical(initialize(args.run_id)))
    elif args.command == "prepare-calibration":
        print(_canonical(prepare_calibration(args.run_id)))
    elif args.command == "next-calibration":
        print(json.dumps(next_calibration(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "ingest-calibration":
        print(_canonical(ingest_calibration(args.run_id, args.response)))
    elif args.command == "prepare-paragraph-calibration":
        print(_canonical(prepare_paragraph_calibration(args.run_id)))
    elif args.command == "next-paragraph-calibration":
        print(json.dumps(next_paragraph_calibration(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "ingest-paragraph-calibration":
        print(_canonical(ingest_paragraph_calibration(args.run_id, args.response)))
    elif args.command == "prepare-calibration-rerun":
        print(_canonical(prepare_calibration_rerun(args.run_id)))
    elif args.command == "next-calibration-rerun":
        print(json.dumps(next_calibration_rerun(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "ingest-calibration-rerun":
        print(_canonical(ingest_calibration_rerun(args.run_id, args.response)))
    elif args.command == "freeze-calibration":
        print(_canonical(freeze_calibration(args.run_id)))
    elif args.command == "record-exact-calibration-rerun":
        print(_canonical(record_exact_calibration_rerun(args.run_id)))
    elif args.command == "plan-production":
        print(_canonical(build_production_plan(args.run_id)))
    elif args.command == "open-document-production":
        print(_canonical(open_document_production(args.run_id)))
    elif args.command == "next-document-production":
        print(json.dumps(next_document_production(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "ingest-document-production":
        print(_canonical(ingest_document_production(args.run_id, args.response)))
    elif args.command == "open-paragraph-production":
        print(_canonical(open_paragraph_production(args.run_id)))
    elif args.command == "next-paragraph-production":
        print(json.dumps(next_paragraph_production(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "ingest-paragraph-production":
        print(_canonical(ingest_paragraph_production(args.run_id, args.response)))
    elif args.command == "lock-review-queue":
        print(_canonical(lock_review_queue(args.run_id)))
    elif args.command == "capture-review-runtime":
        print(_canonical(capture_review_runtime(args.run_id)))
    elif args.command == "next-review-assignment":
        print(json.dumps(next_review_assignment(args.run_id), ensure_ascii=False, indent=2))
    elif args.command == "ingest-review-assignment":
        print(_canonical(ingest_review_assignment(args.run_id, args.response)))
    elif args.command == "compare-review":
        print(_canonical(compare_review(args.run_id)))
    elif args.command == "adjudicate-review-to-reviewer":
        print(_canonical(adjudicate_review_to_reviewer(args.run_id)))
    elif args.command == "derive-attribution":
        print(_canonical(derive_attribution(args.run_id)))
    elif args.command == "seal-speaker-run":
        print(_canonical(seal_speaker_run(args.run_id)))
    elif args.command == "verify-speaker-run-seal":
        print(_canonical(verify_speaker_run_seal(args.run_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
