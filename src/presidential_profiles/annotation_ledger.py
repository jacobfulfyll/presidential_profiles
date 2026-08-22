"""Provider-neutral append-only storage for annotation runs and label events.

The historical source of truth is one immutable, checksummed artifact set per
sealed run.  Parquet tables are complete deterministic materializations over
those sets and explicit adjudication/promotion artifacts; they are never edited
in place.

This module is deliberately provider-free.  It does not import a model client,
read credentials, choose a model, or open a network connection.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import re
import shutil
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .corpus import DATA_DIR


LEDGER_ROOT = DATA_DIR / "annotation_ledger"
SPECS_ROOT = LEDGER_ROOT / "specs"
WORK_ROOT = LEDGER_ROOT / "work"
SEALED_ROOT = LEDGER_ROOT / "sealed_runs"
DECISIONS_ROOT = LEDGER_ROOT / "decisions"
MATERIALIZED_ROOT = LEDGER_ROOT / "materialized"
REVIEW_ROOT = LEDGER_ROOT / "review"

REGISTRY_PATH = SPECS_ROOT / "registry-v1.json"
REGISTRY_V2_PATH = SPECS_ROOT / "registry-v2.json"
REGISTRY_V3_PATH = SPECS_ROOT / "registry-v3.json"
CURRENT_PATH = MATERIALIZED_ROOT / "current"
LOCK_PATH = LEDGER_ROOT / ".annotation-ledger.lock"

HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9](?!.*\.\.)[A-Za-z0-9._-]{0,159}")
SPEC_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
ID_PREFIX_RE = re.compile(r"[a-z]+_[0-9a-f]{64}")

VALUE_KINDS = {
    "boolean",
    "integer",
    "number",
    "string",
    "category",
    "list",
    "object",
    "null",
}


def canonical_json(value: Any) -> str:
    """Return the ledger's canonical UTF-8 JSON representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_canonical_array(values: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(canonical_json(dict(value)).encode("utf-8"))
        first = False
    digest.update(b"]")
    return "sha256:" + digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def content_id(prefix: str, identity: Mapping[str, Any]) -> str:
    return f"{prefix}_" + hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_run_id(run_id: str) -> str:
    value = str(run_id)
    if not RUN_ID_RE.fullmatch(value) or "/" in value or "\\" in value:
        raise ValueError(f"invalid run_id: {value!r}")
    return value


def validate_spec_part(value: str, field: str) -> str:
    value = str(value)
    if not SPEC_PART_RE.fullmatch(value):
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def subject_key(
    subject_type: str,
    *,
    doc_name: str | None = None,
    para_idx: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
) -> dict[str, Any]:
    if subject_type == "paragraph":
        if doc_name is None or para_idx is None:
            raise ValueError("paragraph subject requires doc_name and para_idx")
        return {"doc_name": str(doc_name), "para_idx": int(para_idx)}
    if subject_type == "speech":
        if doc_name is None:
            raise ValueError("speech subject requires doc_name")
        return {"doc_name": str(doc_name)}
    if subject_type == "invocation_span":
        if doc_name is None or char_start is None or char_end is None:
            raise ValueError(
                "invocation_span requires doc_name, char_start, and char_end"
            )
        return {
            "char_end": int(char_end),
            "char_start": int(char_start),
            "doc_name": str(doc_name),
        }
    raise ValueError(f"unregistered subject_type: {subject_type!r}")


def subject_identity(subject_type: str, key: Mapping[str, Any]) -> str:
    return content_id(
        "sub",
        {"subject_key": canonical_json(dict(key)), "subject_type": subject_type},
    )


def value_kind(value: Any, *, category: bool = False) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "category" if category else "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def spec_hash(spec: Mapping[str, Any]) -> str:
    payload = dict(spec)
    payload.pop("spec_sha256", None)
    return sha256_text(canonical_json(payload))


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    v1_required = {
        "label_type",
        "spec_version",
        "stability",
        "subject_type",
        "historical_quantity",
        "prompt_version",
        "prompt_text",
        "prompt_sha256",
        "response_schema",
        "response_schema_sha256",
        "value_kind",
        "event_extractor",
        "context_policy",
        "special_validator",
        "source_spec_refs",
        "limitations",
        "spec_sha256",
    }
    schema_version = spec.get("spec_schema_version")
    if schema_version is None:
        required = v1_required
    elif schema_version == "annotation-label-spec-v2":
        required = v1_required | {"spec_schema_version", "execution_binding"}
    else:
        raise ValueError(f"unsupported label specification schema: {schema_version!r}")
    actual = set(spec)
    if actual != required:
        raise ValueError(
            f"spec field mismatch: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )
    label_type = validate_spec_part(str(spec["label_type"]), "label_type")
    version = validate_spec_part(str(spec["spec_version"]), "spec_version")
    if spec["stability"] not in {"frozen", "legacy_partial", "draft_pilot", "test"}:
        raise ValueError(f"{label_type}: invalid stability")
    if spec["value_kind"] not in VALUE_KINDS:
        raise ValueError(f"{label_type}: invalid value_kind")
    if sha256_text(str(spec["prompt_text"])) != spec["prompt_sha256"]:
        raise ValueError(f"{label_type}: prompt hash drift")
    if sha256_text(canonical_json(spec["response_schema"])) != spec[
        "response_schema_sha256"
    ]:
        raise ValueError(f"{label_type}: response schema hash drift")
    if spec_hash(spec) != spec["spec_sha256"]:
        raise ValueError(f"{label_type}@{version}: specification hash drift")
    extractor = spec["event_extractor"]
    if not isinstance(extractor, dict) or extractor.get("mode") not in {
        "single",
        "items",
    }:
        raise ValueError(f"{label_type}: invalid event extractor")
    if schema_version == "annotation-label-spec-v2":
        binding = spec["execution_binding"]
        if not isinstance(binding, dict) or set(binding) != {
            "mode",
            "bundle_refs",
            "decoder",
            "atomic_group_rule",
        }:
            raise ValueError(f"{label_type}: invalid execution binding")
        if binding["mode"] != "bundle" or binding["decoder"] != "json_pointer":
            raise ValueError(f"{label_type}: unsupported execution binding")
        if binding["atomic_group_rule"] != "one_label_type_per_subject_response":
            raise ValueError(f"{label_type}: unsupported atomic group rule")
        refs = binding["bundle_refs"]
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{label_type}: bundle_refs must be non-empty")
        identities: set[tuple[str, str]] = set()
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {
                "bundle_id",
                "bundle_version",
                "bundle_sha256",
                "response_pointer",
            }:
                raise ValueError(f"{label_type}: invalid bundle reference")
            identity = (
                validate_spec_part(ref["bundle_id"], "bundle_id"),
                validate_spec_part(ref["bundle_version"], "bundle_version"),
            )
            if identity in identities:
                raise ValueError(f"{label_type}: duplicate bundle reference")
            identities.add(identity)
            if not HASH_RE.fullmatch(str(ref["bundle_sha256"])):
                raise ValueError(f"{label_type}: invalid bundle hash")
            pointer = str(ref["response_pointer"])
            if not pointer.startswith("/") or pointer.count("/") != 1:
                raise ValueError(f"{label_type}: response pointer must name one field")
    return dict(spec)


def read_registry(registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("registry_version") not in {
        "annotation-label-registry-v1",
        "annotation-label-registry-v2",
        "annotation-label-registry-v3",
    }:
        raise ValueError("unsupported label registry version")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("registry entries must be a list")
    identities: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("registry entry must be an object")
        identity = (str(entry.get("label_type")), str(entry.get("spec_version")))
        if identity in identities:
            raise ValueError(f"duplicate registry identity: {identity}")
        identities.add(identity)
        relative_path = Path(str(entry["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"registry spec path escapes its root: {relative_path}")
        spec_path = path.parent / relative_path
        spec = validate_spec(json.loads(spec_path.read_text(encoding="utf-8")))
        if (
            spec["label_type"] != identity[0]
            or spec["spec_version"] != identity[1]
            or spec["spec_sha256"] != entry.get("spec_sha256")
        ):
            raise ValueError(f"registry/spec identity drift: {identity}")
    expected = sha256_text(
        canonical_json({k: v for k, v in registry.items() if k != "registry_sha256"})
    )
    if registry.get("registry_sha256") != expected:
        raise ValueError("registry hash drift")
    return registry


def registered_specs(
    registry_path: Path = REGISTRY_PATH,
) -> dict[tuple[str, str], dict[str, Any]]:
    registry = read_registry(registry_path)
    root = Path(registry_path).parent
    return {
        (entry["label_type"], entry["spec_version"]): validate_spec(
            json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        )
        for entry in registry["entries"]
    }


def resolve_spec(
    label_type: str,
    spec_version: str | None = None,
    *,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    specs = registered_specs(registry_path)
    candidates = [
        spec
        for (candidate_type, candidate_version), spec in specs.items()
        if candidate_type == label_type
        and (spec_version is None or candidate_version == spec_version)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"label spec resolution for {label_type!r}/{spec_version!r} "
            f"returned {len(candidates)} candidates"
        )
    return candidates[0]


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not Path(path).exists():
        return
    with Path(path).open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else Path(path).open
    with opener(path, "rt", encoding="utf-8") if compressed else opener(
        "r", encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def read_json(path: Path) -> Any:
    """Read one JSON value, detecting deterministic gzip by file signature."""
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else path.open
    with opener(path, "rt", encoding="utf-8") if compressed else opener(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def open_deterministic_gzip_text(path: Path) -> io.TextIOWrapper:
    """Open a new deterministic gzip-backed UTF-8 text artifact.

    Legacy partitions are large and repeat run/spec provenance on every row.
    Gzip preserves the canonical JSONL byte stream while keeping the immutable
    artifact set practical on constrained analyst workstations.  The filename
    and modification time are omitted from the gzip header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("xb")
    compressed = gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=raw,
        mtime=0,
    )
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open_deterministic_gzip_text(temporary) as handle:
            for row in rows:
                handle.write(canonical_json(dict(row)) + "\n")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def ledger_lock(root: Path = LEDGER_ROOT) -> Iterator[None]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_PATH.name
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"annotation ledger is locked: {lock}; inspect the owner before removal"
        ) from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if lock.exists():
            lock.unlink()


def work_dir(run_id: str, root: Path = LEDGER_ROOT) -> Path:
    return Path(root) / "work" / validate_run_id(run_id)


def sealed_run_dir(run_id: str, root: Path = LEDGER_ROOT) -> Path:
    return Path(root) / "sealed_runs" / validate_run_id(run_id)


def sealed_artifact_dir(run_id: str, root: Path = LEDGER_ROOT) -> Path:
    parent = sealed_run_dir(run_id, root)
    if not parent.exists():
        raise FileNotFoundError(f"run is not sealed: {run_id}")
    candidates = [path for path in parent.iterdir() if path.is_dir()]
    if len(candidates) != 1 or not HASH_RE.fullmatch("sha256:" + candidates[0].name):
        raise ValueError(f"{run_id}: sealed run must contain exactly one hash directory")
    return candidates[0]


def ensure_unsealed(run_id: str, root: Path = LEDGER_ROOT) -> None:
    if sealed_run_dir(run_id, root).exists():
        raise ValueError(f"run is sealed and immutable: {run_id}")


def snapshot_specs(
    run_directory: Path,
    spec_map_json: str,
    *,
    registry_path: Path = REGISTRY_PATH,
) -> None:
    """Persist the exact registered specifications used by one run."""
    mapping = json.loads(spec_map_json)
    snapshot_root = Path(run_directory) / "specs"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for label_type, identity in sorted(mapping.items()):
        spec = resolve_spec(
            label_type,
            identity["spec_version"],
            registry_path=registry_path,
        )
        if spec["spec_sha256"] != identity["spec_sha256"]:
            raise ValueError(f"{label_type}: run spec hash drift")
        atomic_write_json(
            snapshot_root / f"{label_type}--{spec['spec_version']}.json",
            spec,
        )


def _member_inventory(directory: Path, *, exclude: set[str] | None = None) -> list[dict]:
    exclude = exclude or set()
    rows = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(directory).as_posix()
        if rel in exclude:
            continue
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def artifact_root(inventory: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(list(inventory)))


def _validate_event(event: Mapping[str, Any], specs: Mapping[tuple[str, str], dict]) -> None:
    required = {
        "label_id",
        "label_group_id",
        "run_id",
        "label_type",
        "spec_version",
        "spec_sha256",
        "subject_type",
        "subject_id",
        "subject_key_json",
        "doc_name",
        "para_idx",
        "char_start",
        "char_end",
        "source_text_sha256",
        "assignment_input_sha256",
        "event_role",
        "item_index",
        "group_size",
        "value_kind",
        "raw_value_json",
        "raw_value_sha256",
        "assignment_id",
        "batch_id",
        "source_response_id",
        "source_artifact_id",
        "source_table",
        "source_key_json",
        "source_duplicate_ordinal",
        "assigned_at",
        "labeled_at",
        "ingested_at",
        "timestamp_precision",
        "provenance_status_json",
    }
    if set(event) != required:
        raise ValueError(
            f"label event field mismatch: missing={sorted(required - set(event))}, "
            f"extra={sorted(set(event) - required)}"
        )
    identity = (event["label_type"], event["spec_version"])
    if identity not in specs:
        raise ValueError(f"unregistered label spec: {identity}")
    spec = specs[identity]
    if event["spec_sha256"] != spec["spec_sha256"]:
        raise ValueError(f"{identity}: event spec hash drift")
    key = json.loads(event["subject_key_json"])
    if event["subject_id"] != subject_identity(event["subject_type"], key):
        raise ValueError(f"{event['label_id']}: subject identity drift")
    if event["doc_name"] != key.get("doc_name"):
        raise ValueError(f"{event['label_id']}: doc_name/key drift")
    if event["para_idx"] != key.get("para_idx"):
        raise ValueError(f"{event['label_id']}: para_idx/key drift")
    if event["char_start"] != key.get("char_start"):
        raise ValueError(f"{event['label_id']}: char_start/key drift")
    if event["char_end"] != key.get("char_end"):
        raise ValueError(f"{event['label_id']}: char_end/key drift")
    if event["event_role"] not in {"value", "empty_group_marker"}:
        raise ValueError(f"{event['label_id']}: invalid event role")
    raw = json.loads(event["raw_value_json"])
    if canonical_json(raw) != event["raw_value_json"]:
        raise ValueError(f"{event['label_id']}: non-canonical raw value")
    if event["raw_value_sha256"] != sha256_text(event["raw_value_json"]):
        raise ValueError(f"{event['label_id']}: raw value hash drift")
    declared = event["value_kind"]
    actual_kind = value_kind(raw, category=declared == "category")
    if declared != actual_kind:
        raise ValueError(
            f"{event['label_id']}: value kind {declared!r} != {actual_kind!r}"
        )
    expected_label = content_id(
        "lbl",
        {
            "event_role": event["event_role"],
            "item_index": event["item_index"],
            "label_group_id": event["label_group_id"],
            "raw_value_sha256": event["raw_value_sha256"],
            "source_duplicate_ordinal": event["source_duplicate_ordinal"],
        },
    )
    if event["label_id"] != expected_label:
        raise ValueError(f"{event['label_id']}: label identity drift")


def audit_work_run(
    run_id: str,
    *,
    root: Path = LEDGER_ROOT,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    directory = work_dir(run_id, root)
    issues: list[str] = []
    try:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run.get("run_id") != run_id:
            raise ValueError("run identity drift")
        specs = registered_specs(registry_path)
        spec_map = json.loads(run["label_spec_map_json"])
        expected_snapshots = {
            f"{label_type}--{identity['spec_version']}.json"
            for label_type, identity in spec_map.items()
        }
        snapshot_root = directory / "specs"
        actual_snapshots = (
            {path.name for path in snapshot_root.glob("*.json")}
            if snapshot_root.exists()
            else set()
        )
        if actual_snapshots != expected_snapshots:
            raise ValueError(
                "run spec snapshot set mismatch: "
                f"missing={sorted(expected_snapshots - actual_snapshots)}, "
                f"extra={sorted(actual_snapshots - expected_snapshots)}"
            )
        for label_type, identity in spec_map.items():
            snapshot = validate_spec(
                json.loads(
                    (
                        snapshot_root
                        / f"{label_type}--{identity['spec_version']}.json"
                    ).read_text(encoding="utf-8")
                )
            )
            registered = specs[(label_type, identity["spec_version"])]
            if (
                snapshot["spec_sha256"] != identity["spec_sha256"]
                or snapshot != registered
            ):
                raise ValueError(f"{label_type}: run spec snapshot drift")
        ids: set[str] = set()
        group_projection: dict[str, tuple[Any, ...]] = {}
        groups: dict[str, dict[str, Any]] = {}
        event_count = 0
        event_ids_by_assignment: dict[str, set[str]] = {}
        for event in iter_jsonl(directory / "label_events.jsonl"):
            event_count += 1
            _validate_event(event, specs)
            if event["run_id"] != run_id:
                raise ValueError(f"{event['label_id']}: wrong run_id")
            if event["label_id"] in ids:
                raise ValueError(f"duplicate label_id: {event['label_id']}")
            ids.add(event["label_id"])
            expected_group = content_id(
                "lgrp",
                {
                    "label_type": event["label_type"],
                    "run_id": event["run_id"],
                    "source_response_id": event["source_response_id"],
                    "spec_sha256": event["spec_sha256"],
                    "subject_id": event["subject_id"],
                },
            )
            if event["label_group_id"] != expected_group:
                raise ValueError(f"{event['label_id']}: label group identity drift")
            state = groups.setdefault(
                event["label_group_id"],
                {
                    "count": 0,
                    "event_roles": set(),
                    "group_sizes": set(),
                    "item_indexes": [],
                },
            )
            state["count"] += 1
            state["event_roles"].add(event["event_role"])
            state["group_sizes"].add(event["group_size"])
            state["item_indexes"].append(event["item_index"])
            if event["assignment_id"] is not None:
                event_ids_by_assignment.setdefault(event["assignment_id"], set()).add(
                    event["label_id"]
                )
        for group_id, state in groups.items():
            if len(state["group_sizes"]) != 1 or len(state["event_roles"]) != 1:
                raise ValueError(f"{group_id}: inconsistent label group")
            role = next(iter(state["event_roles"]))
            group_size = next(iter(state["group_sizes"]))
            if role == "empty_group_marker":
                if state["count"] != 1 or state["item_indexes"] != [None]:
                    raise ValueError(f"{group_id}: invalid empty group marker")
            else:
                indices = sorted(state["item_indexes"])
                if group_size != state["count"]:
                    raise ValueError(f"{group_id}: incomplete label group")
                if group_size > 1 and indices != list(range(group_size)):
                    raise ValueError(f"{group_id}: non-contiguous item indexes")
        projection_ids: set[str] = set()
        unprojected_ids = set(ids)
        projection_count = 0
        for row in iter_jsonl(directory / "canonical_label_projection.jsonl"):
            projection_count += 1
            if row["label_id"] not in ids:
                raise ValueError(f"projection for unknown event: {row['label_id']}")
            if row["projection_id"] in projection_ids:
                raise ValueError(f"duplicate projection_id: {row['projection_id']}")
            projection_ids.add(row["projection_id"])
            group_state = (
                row["canonical_subject_id"],
                row["canonical_subject_key_json"],
                row["eligible_for_promotion"],
                row["mapping_status"],
            )
            previous = group_projection.setdefault(row["label_group_id"], group_state)
            if previous != group_state:
                raise ValueError(
                    f"group has inconsistent projection: {row['label_group_id']}"
                )
            if row["label_id"] not in unprojected_ids:
                raise ValueError(f"duplicate projected label_id: {row['label_id']}")
            unprojected_ids.remove(row["label_id"])
        if unprojected_ids or projection_count != event_count:
            raise ValueError("projection/event key-set mismatch")
        assignment_paths = sorted((directory / "assignments").glob("*.json"))
        response_paths = sorted((directory / "responses").glob("*.json"))
        assignments = {path.name for path in assignment_paths if path.is_file()}
        responses = {path.name for path in response_paths if path.is_file()}
        if responses - assignments:
            raise ValueError(f"orphan responses: {sorted(responses - assignments)[:5]}")
        assigned_once: set[str] = set()
        for assignment_path in assignment_paths:
            assignment = read_json(assignment_path)
            assignment_body = {
                key: value
                for key, value in assignment.items()
                if key
                not in {"assignment_id", "assignment_input_sha256", "assigned_at"}
            }
            expected_assignment_id = content_id("asg", assignment_body)
            if (
                assignment["assignment_id"] != expected_assignment_id
                or assignment_path.stem != expected_assignment_id
                or assignment["assignment_input_sha256"]
                != sha256_text(canonical_json(assignment_body))
            ):
                raise ValueError(f"{assignment_path}: assignment identity drift")
            target_ids = [target["subject_id"] for target in assignment["targets"]]
            if len(target_ids) != len(set(target_ids)):
                raise ValueError(f"{assignment_path}: duplicate assignment target")
            if assigned_once & set(target_ids):
                raise ValueError(f"{assignment_path}: target assigned more than once")
            assigned_once.update(target_ids)
            response_path = directory / "responses" / assignment_path.name
            if response_path.exists():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if (
                    response.get("assignment_id") != expected_assignment_id
                    or response.get("run_id") != run_id
                    or response.get("assignment_input_sha256")
                    != assignment["assignment_input_sha256"]
                    or set(response.get("label_ids", []))
                    != event_ids_by_assignment.get(expected_assignment_id, set())
                ):
                    raise ValueError(f"{response_path}: response/event drift")
        if run.get("completion_policy") == "complete_selection" and assignments != responses:
            raise ValueError(
                f"open assignments: {sorted(assignments - responses)[:5]}"
            )
        if run.get("completion_policy") == "complete_selection":
            subjects = read_jsonl(directory / "subjects.jsonl")
            subject_ids = {row["subject_id"] for row in subjects}
            assigned_subject_ids: set[str] = set()
            for assignment_path in assignment_paths:
                assignment = read_json(assignment_path)
                assigned_subject_ids.update(
                    target["subject_id"] for target in assignment["targets"]
                )
            if assigned_subject_ids != subject_ids:
                raise ValueError(
                    "assignment/selection key-set mismatch: "
                    f"missing={len(subject_ids - assigned_subject_ids)}, "
                    f"extra={len(assigned_subject_ids - subject_ids)}"
                )
        if run.get("expected_event_count") is not None and event_count != int(
            run["expected_event_count"]
        ):
            raise ValueError(
                f"event count mismatch: {event_count} != {run['expected_event_count']}"
            )
    except Exception as exc:
        issues.append(str(exc))
    return {
        "status": "ok" if not issues else "failed",
        "run_id": run_id,
        "issues": issues,
    }


def seal_run(
    run_id: str,
    *,
    root: Path = LEDGER_ROOT,
    registry_path: Path = REGISTRY_PATH,
    sealed_at: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one immutable run artifact set."""
    run_id = validate_run_id(run_id)
    ensure_unsealed(run_id, root)
    work = work_dir(run_id, root)
    if not work.exists():
        raise FileNotFoundError(work)
    audit = audit_work_run(run_id, root=root, registry_path=registry_path)
    if audit["status"] != "ok":
        raise ValueError(f"run audit failed: {audit['issues']}")
    atomic_write_json(work / "audit.json", audit)
    sealed_at = sealed_at or utc_now()
    atomic_write_json(
        work / "seal-metadata.json",
        {"run_id": run_id, "sealed_at": sealed_at, "timestamp_precision": "microsecond"},
    )

    with ledger_lock(root):
        ensure_unsealed(run_id, root)
        temp_parent = Path(root) / "sealed_runs" / f".{run_id}.{os.getpid()}.tmp"
        if temp_parent.exists():
            raise FileExistsError(temp_parent)
        payload = temp_parent / "payload"
        payload.parent.mkdir(parents=True, exist_ok=False)
        shutil.copytree(work, payload)
        candidate_marker = payload / "seal-candidate"
        if candidate_marker.exists():
            shutil.rmtree(candidate_marker)
        inventory = _member_inventory(payload, exclude={"seal.json"})
        root_hash = artifact_root(inventory)
        hash_dir = temp_parent / root_hash.removeprefix("sha256:")
        os.replace(payload, hash_dir)
        seal = {
            "run_id": run_id,
            "artifact_set_sha256": root_hash,
            "members": inventory,
        }
        atomic_write_json(hash_dir / "seal.json", seal)
        final_parent = Path(root) / "sealed_runs" / run_id
        os.replace(temp_parent, final_parent)
        fsync_directory(final_parent.parent)
        # The verified sealed copy is now authoritative.  Removing the
        # redundant mutable work tree prevents two copies of large legacy
        # partitions from lingering after the atomic publication succeeds.
        shutil.rmtree(work)
    return {
        "status": "sealed",
        "run_id": run_id,
        "artifact_set_sha256": root_hash,
        "path": str(final_parent / root_hash.removeprefix("sha256:")),
    }


def verify_sealed_run(run_id: str, root: Path = LEDGER_ROOT) -> dict[str, Any]:
    directory = sealed_artifact_dir(run_id, root)
    seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    inventory = _member_inventory(directory, exclude={"seal.json"})
    root_hash = artifact_root(inventory)
    if inventory != seal.get("members") or root_hash != seal.get(
        "artifact_set_sha256"
    ):
        raise ValueError(f"{run_id}: sealed artifact set hash mismatch")
    return seal


def sealed_runs(root: Path = LEDGER_ROOT) -> list[str]:
    path = Path(root) / "sealed_runs"
    if not path.exists():
        return []
    return sorted(
        child.name
        for child in path.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


def _string(name: str, nullable: bool = False) -> pa.Field:
    return pa.field(name, pa.string(), nullable=nullable)


def _i64(name: str, nullable: bool = False) -> pa.Field:
    return pa.field(name, pa.int64(), nullable=nullable)


def _bool(name: str, nullable: bool = False) -> pa.Field:
    return pa.field(name, pa.bool_(), nullable=nullable)


UTC_TS = pa.timestamp("us", tz="UTC")

RUN_SCHEMA = pa.schema(
    [
        _string("run_id"),
        _string("run_kind"),
        _string("run_status"),
        _string("scope"),
        _string("label_spec_map_json"),
        _string("provider_id", True),
        _string("model_id", True),
        _string("model_identity_source"),
        _string("annotator_id", True),
        _string("agent_id", True),
        _string("reasoning_effort"),
        _string("reasoning_effort_provenance"),
        _string("reasoning_effort_by_spec_json", True),
        pa.field("manifest_date", pa.date32(), nullable=True),
        pa.field("assigned_at", UTC_TS, nullable=True),
        pa.field("labeled_at", UTC_TS, nullable=True),
        pa.field("ingested_at", UTC_TS),
        pa.field("sealed_at", UTC_TS),
        _string("timestamp_precision"),
        _string("source_corpus_fingerprint", True),
        _string("canonical_corpus_fingerprint", True),
        _string("selection_artifact_id", True),
        _string("authorization_review_item_id", True),
        _string("human_validation_status"),
        _string("independent_check_status"),
        _string("source_manifest_path", True),
        _string("source_manifest_sha256", True),
        _string("artifacts_json"),
        _string("limitations_json"),
        _string("run_manifest_sha256"),
        _string("artifact_set_sha256"),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        _string("label_id"),
        _string("label_group_id"),
        _string("run_id"),
        _string("label_type"),
        _string("spec_version"),
        _string("spec_sha256"),
        _string("subject_type"),
        _string("subject_id"),
        _string("subject_key_json"),
        _string("doc_name", True),
        _i64("para_idx", True),
        _i64("char_start", True),
        _i64("char_end", True),
        _string("source_text_sha256", True),
        _string("assignment_input_sha256", True),
        _string("event_role"),
        _i64("item_index", True),
        _i64("group_size", True),
        _string("value_kind"),
        _string("raw_value_json"),
        _string("raw_value_sha256"),
        _string("assignment_id", True),
        _string("batch_id", True),
        _string("source_response_id"),
        _string("source_artifact_id"),
        _string("source_table", True),
        _string("source_key_json"),
        _i64("source_duplicate_ordinal"),
        pa.field("assigned_at", UTC_TS, nullable=True),
        pa.field("labeled_at", UTC_TS, nullable=True),
        pa.field("ingested_at", UTC_TS),
        _string("timestamp_precision"),
        _string("provenance_status_json"),
    ]
)

PROJECTION_SCHEMA = pa.schema(
    [
        _string("projection_id"),
        _string("label_id"),
        _string("label_group_id"),
        _string("source_subject_id"),
        _string("canonical_subject_id", True),
        _string("canonical_subject_key_json", True),
        _string("canonical_doc_name", True),
        _i64("canonical_para_idx", True),
        _string("mapping_status"),
        _string("mapping_rule_id"),
        _string("source_text_sha256", True),
        _string("canonical_text_sha256", True),
        _string("source_input_sha256", True),
        _string("canonical_input_sha256", True),
        _bool("eligible_for_promotion"),
        _string("review_item_id", True),
        _string("reason"),
    ]
)

ADJUDICATION_SCHEMA = pa.schema(
    [
        _string("adjudication_id"),
        _string("canonical_subject_id"),
        _string("label_type"),
        _string("spec_sha256"),
        _string("candidate_label_group_ids_json"),
        _string("resolution_kind"),
        _string("selected_label_group_id", True),
        _string("resolved_label_group_id", True),
        _string("adjudicator_id"),
        _string("rationale"),
        _string("review_resolution_id", True),
        pa.field("decided_at", UTC_TS),
        _string("artifact_set_sha256"),
    ]
)

PROMOTION_SCHEMA = pa.schema(
    [
        _string("promotion_id"),
        _string("promotion_batch_id"),
        _string("promotion_channel"),
        _string("canonical_subject_id"),
        _string("label_type"),
        _string("action"),
        _string("label_group_id", True),
        _string("replaces_label_group_ids_json"),
        _string("expected_current_state_sha256"),
        _string("promotion_rule_id"),
        _string("adjudication_id", True),
        _string("review_resolution_id", True),
        _string("operator_id"),
        pa.field("promoted_at", UTC_TS),
        _string("source_run_artifact_set_sha256"),
        _string("artifact_set_sha256"),
    ]
)

CURRENT_SCHEMA = pa.schema(
    [
        _string("promotion_channel"),
        _string("canonical_subject_id"),
        _string("label_type"),
        _string("promotion_id"),
        _string("promotion_batch_id"),
        _string("label_group_id"),
        _string("label_id"),
        _string("run_id"),
        _string("spec_version"),
        _string("spec_sha256"),
        _string("canonical_subject_key_json"),
        _string("canonical_doc_name", True),
        _i64("canonical_para_idx", True),
        _string("event_role"),
        _i64("item_index", True),
        _string("value_kind"),
        _string("raw_value_json"),
        _string("promotion_rule_id"),
        _string("adjudication_id", True),
        _string("review_resolution_id", True),
    ]
)

METRIC_SCHEMA = pa.schema(
    [
        _string("metric_id"),
        _string("record_kind"),
        _string("evaluation_id"),
        _string("label_type", True),
        _string("left_run_id", True),
        _string("right_run_id", True),
        _string("stratum_json"),
        _string("metric_name", True),
        pa.field("metric_value", pa.float64(), nullable=True),
        _i64("n", True),
        _string("source_value_json"),
        _string("artifact_role", True),
        _string("artifact_path", True),
        _string("artifact_sha256", True),
        _string("source_artifact_id"),
        _string("provenance_json"),
    ]
)

CONSTITUENCY_SCHEMA = pa.schema(
    [
        _string("promotion_channel"),
        _string("label_id"),
        _string("label_group_id"),
        _string("run_id"),
        _string("doc_name"),
        _i64("para_idx"),
        _string("outcome"),
        _i64("claim_index", True),
        _string("group_text", True),
        _string("resolved_referent", True),
        _string("normalized_group", True),
        _string("normalization_status", True),
        _string("normalization_spec_sha256", True),
        _string("group_type", True),
        _string("relation", True),
        _string("stance", True),
        _string("evidence_span", True),
        _string("certainty", True),
        _string("notes", True),
        _string("spec_version"),
        _string("spec_sha256"),
        _string("model_id", True),
        _string("annotator_id", True),
        _string("canonical_corpus_fingerprint", True),
        _string("promotion_id"),
    ]
)


def _coerce_row(row: Mapping[str, Any], schema: pa.Schema) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in schema:
        value = row.get(field.name)
        if pa.types.is_timestamp(field.type) and isinstance(value, str):
            value = parse_utc(value)
        elif pa.types.is_date32(field.type) and isinstance(value, str):
            value = date.fromisoformat(value)
        result[field.name] = value
    return result


def write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    sort_by: Sequence[str],
) -> None:
    ordered = sorted(
        (_coerce_row(row, schema) for row in rows),
        key=lambda row: tuple(
            (row.get(column) is None, row.get(column)) for column in sort_by
        ),
    )
    table = pa.Table.from_pylist(ordered, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        version="2.6",
        data_page_version="2.0",
        write_statistics=True,
        use_dictionary=False,
    )


def write_parquet_stream(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    schema: pa.Schema,
    *,
    batch_size: int = 5_000,
) -> int:
    """Write a deterministic row stream with fixed schemas and row groups."""
    count = 0
    batch: list[dict[str, Any]] = []
    writer = pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=9,
        version="2.6",
        data_page_version="2.0",
        write_statistics=True,
        use_dictionary=False,
    )
    try:
        for row in rows:
            batch.append(_coerce_row(row, schema))
            count += 1
            if len(batch) == batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        elif count == 0:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
    finally:
        writer.close()
    return count


def _run_row(run_id: str, root: Path) -> dict[str, Any]:
    directory = sealed_artifact_dir(run_id, root)
    seal = verify_sealed_run(run_id, root)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (directory / "seal-metadata.json").read_text(encoding="utf-8")
    )
    return {
        **run,
        "run_status": "sealed",
        "sealed_at": metadata["sealed_at"],
        "run_manifest_sha256": sha256_file(directory / "run.json"),
        "artifact_set_sha256": seal["artifact_set_sha256"],
    }


def _decision_artifact_dirs(kind: str, root: Path) -> list[Path]:
    base = Path(root) / "decisions" / kind
    if not base.exists():
        return []
    result = []
    for parent in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
        children = [p for p in parent.iterdir() if p.is_dir()]
        if len(children) != 1:
            raise ValueError(f"{parent}: decision artifact must have one hash directory")
        result.append(children[0])
    return result


def _read_decisions(kind: str, filename: str, root: Path) -> list[dict[str, Any]]:
    return list(_iter_decisions(kind, filename, root))


def _iter_decisions(
    kind: str, filename: str, root: Path
) -> Iterator[dict[str, Any]]:
    for directory in _decision_artifact_dirs(kind, root):
        seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
        inventory = _member_inventory(directory, exclude={"seal.json"})
        root_hash = artifact_root(inventory)
        if inventory != seal.get("members") or root_hash != seal.get(
            "artifact_set_sha256"
        ):
            raise ValueError(f"{directory}: decision artifact hash mismatch")
        for row in read_jsonl(directory / filename):
            row["artifact_set_sha256"] = root_hash
            yield row


def current_state_hash(group_ids: Sequence[str]) -> str:
    return sha256_text(canonical_json(sorted(group_ids)))


def _replay_promotion_state(
    promotions: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], tuple[str, Mapping[str, Any]]]:
    state: dict[tuple[str, str, str], tuple[str, Mapping[str, Any]]] = {}
    for row in sorted(
        promotions,
        key=lambda value: (
            parse_utc(str(value["promoted_at"])),
            value["promotion_batch_id"],
            value["promotion_channel"],
            value["canonical_subject_id"],
            value["label_type"],
        ),
    ):
        key = (
            str(row["promotion_channel"]),
            str(row["canonical_subject_id"]),
            str(row["label_type"]),
        )
        current = [state[key][0]] if key in state else []
        replaces = json.loads(str(row["replaces_label_group_ids_json"]))
        if sorted(current) != sorted(replaces):
            raise ValueError(f"promotion replay state mismatch for {key}")
        if current_state_hash(current) != row["expected_current_state_sha256"]:
            raise ValueError(f"promotion replay hash mismatch for {key}")
        if row["action"] == "deactivate":
            state.pop(key, None)
        elif row["action"] == "activate":
            state[key] = (str(row["label_group_id"]), row)
        else:
            raise ValueError(f"unknown promotion action: {row['action']}")
    return state


def publish_promotions(
    transitions: Sequence[Mapping[str, Any]],
    *,
    operator_id: str,
    promotion_rule_id: str,
    review_resolution_id: str | None,
    root: Path = LEDGER_ROOT,
    promoted_at: str | None = None,
) -> dict[str, Any]:
    """Publish one all-or-nothing promotion batch.

    The function validates against the currently materialized state when one is
    active.  An empty ledger has the canonical empty-state hash.
    """
    if not transitions:
        raise ValueError("promotion batch must contain at least one transition")
    promoted_at = promoted_at or utc_now()
    existing_promotions = _read_decisions(
        "promotions", "label_promotions.jsonl", Path(root)
    )
    existing_state = _replay_promotion_state(existing_promotions)
    current = {
        key: [value[0]]
        for key, value in existing_state.items()
    }
    existing_times = [
        parse_utc(str(row["promoted_at"])) for row in existing_promotions
    ]
    new_time = parse_utc(promoted_at)
    if new_time is None:
        raise ValueError("promoted_at must be a UTC timestamp")
    if existing_times and new_time <= max(
        value for value in existing_times if value is not None
    ):
        raise ValueError(
            "promoted_at must be later than every published promotion batch"
        )

    event_groups: dict[str, tuple[str, str, str]] = {}
    source_seals: dict[str, str] = {}
    for run_id in sealed_runs(root):
        directory = sealed_artifact_dir(run_id, root)
        seal = verify_sealed_run(run_id, root)
        source_seals[run_id] = seal["artifact_set_sha256"]
        events = iter_jsonl(directory / "label_events.jsonl")
        projections = iter_jsonl(
            directory / "canonical_label_projection.jsonl"
        )
        for event, projection in zip(events, projections, strict=True):
            if event["label_id"] != projection["label_id"]:
                raise ValueError(f"{run_id}: event/projection storage order drift")
            if projection["eligible_for_promotion"]:
                event_groups[event["label_group_id"]] = (
                    projection["canonical_subject_id"],
                    event["label_type"],
                    event["run_id"],
                )

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for transition in transitions:
        group_id = str(transition["label_group_id"])
        if group_id not in event_groups:
            raise ValueError(f"unknown or ineligible label group: {group_id}")
        subject_id, label_type, run_id = event_groups[group_id]
        channel = str(transition.get("promotion_channel", "primary"))
        key = (channel, subject_id, label_type)
        if key in seen_keys:
            raise ValueError(f"promotion batch repeats current-state key: {key}")
        seen_keys.add(key)
        replaces = sorted(transition.get("replaces_label_group_ids", []))
        actual = sorted(current.get(key, []))
        expected_hash = str(
            transition.get("expected_current_state_sha256", current_state_hash(replaces))
        )
        if actual != replaces or current_state_hash(actual) != expected_hash:
            raise ValueError(
                f"stale promotion state for {key}: actual={actual}, replaces={replaces}"
            )
        normalized.append(
            {
                "promotion_channel": channel,
                "canonical_subject_id": subject_id,
                "label_type": label_type,
                "action": "activate",
                "label_group_id": group_id,
                "replaces_label_group_ids_json": canonical_json(replaces),
                "expected_current_state_sha256": expected_hash,
                "promotion_rule_id": promotion_rule_id,
                "adjudication_id": transition.get("adjudication_id"),
                "review_resolution_id": review_resolution_id,
                "operator_id": operator_id,
                "promoted_at": promoted_at,
                "source_run_artifact_set_sha256": source_seals[run_id],
            }
        )
    normalized.sort(
        key=lambda row: (
            row["promotion_channel"],
            row["canonical_subject_id"],
            row["label_type"],
        )
    )
    batch_identity_hash = sha256_canonical_array(
        {k: v for k, v in row.items() if k != "promoted_at"}
        for row in normalized
    )
    batch_id = "pbat_" + batch_identity_hash.removeprefix("sha256:")
    for row in normalized:
        row["promotion_batch_id"] = batch_id
        row["promotion_id"] = content_id(
            "prom",
            {k: v for k, v in row.items() if k not in {"promoted_at"}},
        )

    with ledger_lock(root):
        parent = Path(root) / "decisions" / "promotions" / batch_id
        if parent.exists():
            raise FileExistsError(f"promotion batch already exists: {batch_id}")
        temp_parent = parent.with_name(f".{batch_id}.{os.getpid()}.tmp")
        payload = temp_parent / "payload"
        payload.mkdir(parents=True)
        write_gzip_jsonl(payload / "label_promotions.jsonl", normalized)
        atomic_write_json(
            payload / "promotion-batch.json",
            {
                "promotion_batch_id": batch_id,
                "promotion_rule_id": promotion_rule_id,
                "review_resolution_id": review_resolution_id,
                "n_transitions": len(normalized),
            },
        )
        inventory = _member_inventory(payload)
        final_root = artifact_root(inventory)
        atomic_write_json(
            payload / "seal.json",
            {
                "artifact_set_sha256": final_root,
                "members": inventory,
                "promotion_batch_id": batch_id,
            },
        )
        hash_dir = temp_parent / final_root.removeprefix("sha256:")
        os.replace(payload, hash_dir)
        os.replace(temp_parent, parent)
        fsync_directory(parent.parent)
    return {
        "status": "published",
        "promotion_batch_id": batch_id,
        "artifact_set_sha256": final_root,
        "n_transitions": len(normalized),
    }


def _adjudication_group_metadata(
    root: Path,
) -> dict[str, tuple[str, str, str]]:
    """Build the validated sealed-group index once for a decision batch."""
    group_metadata: dict[str, tuple[str, str, str]] = {}
    for run_id in sealed_runs(root):
        directory = sealed_artifact_dir(run_id, root)
        verify_sealed_run(run_id, root)
        projections = {
            row["label_id"]: row
            for row in read_jsonl(
                directory / "canonical_label_projection.jsonl"
            )
        }
        for event in read_jsonl(directory / "label_events.jsonl"):
            projection = projections[event["label_id"]]
            canonical_subject_id = projection["canonical_subject_id"]
            if canonical_subject_id is not None:
                group_metadata[event["label_group_id"]] = (
                    canonical_subject_id,
                    event["label_type"],
                    event["spec_sha256"],
                )
    return group_metadata


def publish_adjudication(
    decision: Mapping[str, Any],
    *,
    root: Path = LEDGER_ROOT,
    _validated_group_metadata: Mapping[
        str, tuple[str, str, str]
    ]
    | None = None,
) -> dict[str, Any]:
    required = {
        "canonical_subject_id",
        "label_type",
        "spec_sha256",
        "candidate_label_group_ids",
        "resolution_kind",
        "selected_label_group_id",
        "resolved_label_group_id",
        "adjudicator_id",
        "rationale",
        "review_resolution_id",
        "decided_at",
    }
    if set(decision) != required:
        raise ValueError("adjudication field mismatch")
    if decision["resolution_kind"] not in {
        "select_existing",
        "synthesize_in_adjudication_run",
        "reject_all",
        "defer",
    }:
        raise ValueError("invalid adjudication resolution kind")
    candidate_ids = sorted(decision["candidate_label_group_ids"])
    if not candidate_ids:
        raise ValueError("adjudication requires at least one candidate label group")
    group_metadata = (
        dict(_validated_group_metadata)
        if _validated_group_metadata is not None
        else _adjudication_group_metadata(Path(root))
    )
    unknown = set(candidate_ids) - set(group_metadata)
    if unknown:
        raise ValueError(f"unknown adjudication candidates: {sorted(unknown)}")
    expected = (
        decision["canonical_subject_id"],
        decision["label_type"],
        decision["spec_sha256"],
    )
    if any(group_metadata[group_id] != expected for group_id in candidate_ids):
        raise ValueError("adjudication candidates do not share the declared identity")
    if (
        decision["resolution_kind"] == "select_existing"
        and decision["selected_label_group_id"] not in candidate_ids
    ):
        raise ValueError("selected adjudication group is not a candidate")
    if decision["resolution_kind"] != "select_existing" and decision[
        "selected_label_group_id"
    ] is not None:
        raise ValueError("only select_existing may set selected_label_group_id")
    payload = dict(decision)
    payload["candidate_label_group_ids_json"] = canonical_json(
        sorted(payload.pop("candidate_label_group_ids"))
    )
    adjudication_id = content_id(
        "adj", {k: v for k, v in payload.items() if k != "decided_at"}
    )
    payload["adjudication_id"] = adjudication_id
    parent = Path(root) / "decisions" / "adjudications" / adjudication_id
    with ledger_lock(root):
        if parent.exists():
            raise FileExistsError(parent)
        temp = parent.with_name(f".{adjudication_id}.{os.getpid()}.tmp")
        content = temp / "payload"
        content.mkdir(parents=True)
        write_jsonl(content / "label_adjudications.jsonl", [payload])
        inventory = _member_inventory(content)
        final_root = artifact_root(inventory)
        atomic_write_json(
            content / "seal.json",
            {
                "adjudication_id": adjudication_id,
                "artifact_set_sha256": final_root,
                "members": inventory,
            },
        )
        hash_dir = temp / final_root.removeprefix("sha256:")
        os.replace(content, hash_dir)
        os.replace(temp, parent)
        fsync_directory(parent.parent)
    return {
        "status": "published",
        "adjudication_id": adjudication_id,
        "artifact_set_sha256": final_root,
    }


def _apply_promotions(
    promotions: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    projections: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    projection_by_label = {row["label_id"]: row for row in projections}
    group_events: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        group_events.setdefault(event["label_group_id"], []).append(event)
    state = _replay_promotion_state(promotions)
    result: list[dict[str, Any]] = []
    for key, (group_id, promotion) in state.items():
        for event in group_events.get(group_id, []):
            projection = projection_by_label[event["label_id"]]
            result.append(
                {
                    "promotion_channel": key[0],
                    "canonical_subject_id": key[1],
                    "label_type": key[2],
                    "promotion_id": promotion["promotion_id"],
                    "promotion_batch_id": promotion["promotion_batch_id"],
                    "label_group_id": group_id,
                    "label_id": event["label_id"],
                    "run_id": event["run_id"],
                    "spec_version": event["spec_version"],
                    "spec_sha256": event["spec_sha256"],
                    "canonical_subject_key_json": projection[
                        "canonical_subject_key_json"
                    ],
                    "canonical_doc_name": projection["canonical_doc_name"],
                    "canonical_para_idx": projection["canonical_para_idx"],
                    "event_role": event["event_role"],
                    "item_index": event["item_index"],
                    "value_kind": event["value_kind"],
                    "raw_value_json": event["raw_value_json"],
                    "promotion_rule_id": promotion["promotion_rule_id"],
                    "adjudication_id": promotion["adjudication_id"],
                    "review_resolution_id": promotion["review_resolution_id"],
                }
            )
    return result


def load_current_generation(
    root: Path = LEDGER_ROOT, *, required: bool = True
) -> Path | None:
    pointer = Path(root) / "materialized" / "current"
    if not pointer.exists():
        if required:
            raise FileNotFoundError("no active annotation-ledger generation")
        return None
    generation_hash = pointer.read_text(encoding="ascii").strip()
    if not HASH_RE.fullmatch("sha256:" + generation_hash):
        raise ValueError("invalid materialized current pointer")
    path = Path(root) / "materialized" / "generations" / generation_hash
    if not path.exists():
        raise ValueError("current pointer targets a missing generation")
    return path


def materialize(root: Path = LEDGER_ROOT) -> dict[str, Any]:
    """Build and atomically activate one complete deterministic generation."""
    run_ids = sealed_runs(root)
    runs: list[dict[str, Any]] = []
    seals: list[dict[str, str]] = []
    for run_id in run_ids:
        seal = verify_sealed_run(run_id, root)
        seals.append({"run_id": run_id, "artifact_set_sha256": seal["artifact_set_sha256"]})
        runs.append(_run_row(run_id, root))
    adjudications = _read_decisions(
        "adjudications", "label_adjudications.jsonl", Path(root)
    )
    promotions = _read_decisions(
        "promotions", "label_promotions.jsonl", Path(root)
    )
    promotion_state = _replay_promotion_state(promotions)
    active_by_group = {
        group_id: (key, promotion)
        for key, (group_id, promotion) in promotion_state.items()
    }

    def run_rows(filename: str) -> Iterator[dict[str, Any]]:
        for run_id in run_ids:
            yield from iter_jsonl(sealed_artifact_dir(run_id, root) / filename)

    def current_rows() -> Iterator[dict[str, Any]]:
        for run_id in run_ids:
            directory = sealed_artifact_dir(run_id, root)
            events = iter_jsonl(directory / "label_events.jsonl")
            projections = iter_jsonl(
                directory / "canonical_label_projection.jsonl"
            )
            for event, projection in zip(events, projections, strict=True):
                if event["label_id"] != projection["label_id"]:
                    raise ValueError(
                        f"{run_id}: event/projection storage order drift"
                    )
                active = active_by_group.get(event["label_group_id"])
                if active is None:
                    continue
                key, promotion = active
                yield {
                    "promotion_channel": key[0],
                    "canonical_subject_id": key[1],
                    "label_type": key[2],
                    "promotion_id": promotion["promotion_id"],
                    "promotion_batch_id": promotion["promotion_batch_id"],
                    "label_group_id": event["label_group_id"],
                    "label_id": event["label_id"],
                    "run_id": event["run_id"],
                    "spec_version": event["spec_version"],
                    "spec_sha256": event["spec_sha256"],
                    "canonical_subject_key_json": projection[
                        "canonical_subject_key_json"
                    ],
                    "canonical_doc_name": projection["canonical_doc_name"],
                    "canonical_para_idx": projection["canonical_para_idx"],
                    "event_role": event["event_role"],
                    "item_index": event["item_index"],
                    "value_kind": event["value_kind"],
                    "raw_value_json": event["raw_value_json"],
                    "promotion_rule_id": promotion["promotion_rule_id"],
                    "adjudication_id": promotion["adjudication_id"],
                    "review_resolution_id": promotion["review_resolution_id"],
                }

    runs_by_id = {row["run_id"]: row for row in runs}

    def constituency_rows() -> Iterator[dict[str, Any]]:
        for current in current_rows():
            if current["label_type"] != "constituencies":
                continue
            value = json.loads(current["raw_value_json"])
            if not isinstance(value, dict):
                raise ValueError("promoted constituency value must be an object")
            outcome = value.get("outcome")
            claims = value.get("claims")
            if outcome not in {"claim", "none", "unclear"} or not isinstance(
                claims, list
            ):
                raise ValueError("promoted constituency value has invalid shape")
            if outcome == "claim" and not claims:
                raise ValueError("promoted constituency claim outcome is empty")
            if outcome != "claim" and claims:
                raise ValueError("non-claim constituency outcome has claims")
            rows = list(enumerate(claims)) if claims else [(None, {})]
            run = runs_by_id[current["run_id"]]
            for claim_index, claim in rows:
                if not isinstance(claim, dict):
                    raise ValueError("constituency claim must be an object")
                normalized_group = claim.get("normalized_group")
                yield {
                    "promotion_channel": current["promotion_channel"],
                    "label_id": current["label_id"],
                    "label_group_id": current["label_group_id"],
                    "run_id": current["run_id"],
                    "doc_name": current["canonical_doc_name"],
                    "para_idx": current["canonical_para_idx"],
                    "outcome": outcome,
                    "claim_index": claim_index,
                    "group_text": claim.get("group_text"),
                    "resolved_referent": claim.get("resolved_referent"),
                    "normalized_group": normalized_group,
                    "normalization_status": (
                        "legacy_model_supplied"
                        if normalized_group not in (None, "")
                        else "not_normalized"
                    ),
                    "normalization_spec_sha256": None,
                    "group_type": claim.get("group_type"),
                    "relation": claim.get("relation"),
                    "stance": claim.get("stance"),
                    "evidence_span": claim.get("evidence_span"),
                    "certainty": claim.get("certainty"),
                    "notes": claim.get("notes") or value.get("unclear_reason"),
                    "spec_version": current["spec_version"],
                    "spec_sha256": current["spec_sha256"],
                    "model_id": run.get("model_id"),
                    "annotator_id": run.get("annotator_id"),
                    "canonical_corpus_fingerprint": run.get(
                        "canonical_corpus_fingerprint"
                    ),
                    "promotion_id": current["promotion_id"],
                }

    temp = (
        Path(root)
        / "materialized"
        / "generations"
        / f".generation.{os.getpid()}.tmp"
    )
    if temp.exists():
        raise FileExistsError(temp)
    temp.mkdir(parents=True)
    try:
        n_runs = write_parquet_stream(
            temp / "annotation_runs.parquet", runs, RUN_SCHEMA
        )
        n_events = write_parquet_stream(
            temp / "label_events.parquet",
            run_rows("label_events.jsonl"),
            EVENT_SCHEMA,
        )
        n_projections = write_parquet_stream(
            temp / "canonical_label_projection.parquet",
            run_rows("canonical_label_projection.jsonl"),
            PROJECTION_SCHEMA,
        )
        n_adjudications = write_parquet_stream(
            temp / "label_adjudications.parquet",
            adjudications,
            ADJUDICATION_SCHEMA,
        )
        n_promotions = write_parquet_stream(
            temp / "label_promotions.parquet",
            promotions,
            PROMOTION_SCHEMA,
        )
        n_current = write_parquet_stream(
            temp / "current_labels.parquet",
            current_rows(),
            CURRENT_SCHEMA,
        )
        n_metrics = write_parquet_stream(
            temp / "evaluation_metrics.parquet",
            run_rows("evaluation_metrics.jsonl"),
            METRIC_SCHEMA,
        )
        n_constituencies = write_parquet_stream(
            temp / "constituency_claims.parquet",
            constituency_rows(),
            CONSTITUENCY_SCHEMA,
        )
        manifest = {
            "generation_version": "annotation-ledger-materialization-v2",
            "sealed_runs": sorted(seals, key=lambda row: row["run_id"]),
            "promotion_batches": sorted(
                {row["promotion_batch_id"] for row in promotions}
            ),
            "adjudications": sorted(
                {row["adjudication_id"] for row in adjudications}
            ),
            "tables": {
                "annotation_runs": n_runs,
                "label_events": n_events,
                "canonical_label_projection": n_projections,
                "label_adjudications": n_adjudications,
                "label_promotions": n_promotions,
                "current_labels": n_current,
                "evaluation_metrics": n_metrics,
                "constituency_claims": n_constituencies,
            },
        }
        atomic_write_json(temp / "generation.json", manifest)
        inventory = _member_inventory(temp)
        generation_hash = artifact_root(inventory)
        final = (
            Path(root)
            / "materialized"
            / "generations"
            / generation_hash.removeprefix("sha256:")
        )
        with ledger_lock(root):
            if final.exists():
                existing = _member_inventory(final)
                if existing != inventory:
                    raise ValueError("generation hash collision or existing corruption")
                shutil.rmtree(temp)
            else:
                os.replace(temp, final)
                fsync_directory(final.parent)
            atomic_write_text(
                Path(root) / "materialized" / "current",
                generation_hash.removeprefix("sha256:") + "\n",
            )
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {
        "status": "materialized",
        "generation_sha256": generation_hash,
        "path": str(final),
        "tables": manifest["tables"],
    }


def compare_runs(
    left_run_id: str, right_run_id: str, *, root: Path = LEDGER_ROOT
) -> dict[str, Any]:
    left_dir = sealed_artifact_dir(left_run_id, root)
    right_dir = sealed_artifact_dir(right_run_id, root)
    left = read_jsonl(left_dir / "label_events.jsonl")
    right = read_jsonl(right_dir / "label_events.jsonl")
    def grouped(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], str]:
        by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault((row["subject_id"], row["label_type"]), []).append(row)
        return {
            key: canonical_json(
                [
                    json.loads(row["raw_value_json"])
                    for row in sorted(
                        values,
                        key=lambda item: (
                            item["item_index"] is None,
                            item["item_index"],
                            item["label_id"],
                        ),
                    )
                ]
            )
            for key, values in by_group.items()
        }
    a, b = grouped(left), grouped(right)
    overlap = sorted(set(a) & set(b))
    disagreements = [
        {"subject_id": key[0], "label_type": key[1], "left": a[key], "right": b[key]}
        for key in overlap
        if a[key] != b[key]
    ]
    return {
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "overlap": len(overlap),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def append_review_item(
    item: Mapping[str, Any], *, root: Path = LEDGER_ROOT
) -> dict[str, Any]:
    """Append a new review question without resolving it implicitly."""
    required = {
        "item_id",
        "category",
        "question",
        "opened_by",
        "source_authority",
        "required_before",
        "affected_artifacts",
        "status",
    }
    if set(item) != required or item["status"] != "pending":
        raise ValueError("review item must be an exact pending-item record")
    path = Path(root) / "review" / "review_items.jsonl"
    with ledger_lock(root):
        rows = read_jsonl(path)
        if item["item_id"] in {row["item_id"] for row in rows}:
            raise ValueError(f"review item already exists: {item['item_id']}")
        write_jsonl(path, [*rows, dict(item)])
    return {"status": "appended", "item_id": item["item_id"]}


def append_review_resolution(
    resolution: Mapping[str, Any], *, root: Path = LEDGER_ROOT
) -> dict[str, Any]:
    """Append one explicit resolution tied to an existing review item."""
    required = {
        "resolution_id",
        "item_id",
        "authority",
        "decided_by",
        "decision",
        "resolution_date",
        "resolution_date_precision",
        "rationale",
    }
    if set(resolution) != required:
        raise ValueError("review resolution field mismatch")
    item_path = Path(root) / "review" / "review_items.jsonl"
    resolution_path = Path(root) / "review" / "review_resolutions.jsonl"
    with ledger_lock(root):
        items = read_jsonl(item_path)
        if resolution["item_id"] not in {row["item_id"] for row in items}:
            raise ValueError("review resolution references an unknown item")
        rows = read_jsonl(resolution_path)
        if resolution["resolution_id"] in {
            row["resolution_id"] for row in rows
        }:
            raise ValueError(
                f"review resolution already exists: {resolution['resolution_id']}"
            )
        write_jsonl(resolution_path, [*rows, dict(resolution)])
    return {
        "status": "appended",
        "item_id": resolution["item_id"],
        "resolution_id": resolution["resolution_id"],
    }


def render_review_report(root: Path = LEDGER_ROOT) -> str:
    """Render the append-only machine records as deterministic Markdown."""
    items = read_jsonl(Path(root) / "review" / "review_items.jsonl")
    resolutions = read_jsonl(Path(root) / "review" / "review_resolutions.jsonl")
    by_item: dict[str, list[dict[str, Any]]] = {}
    for resolution in resolutions:
        by_item.setdefault(resolution["item_id"], []).append(resolution)
    lines = [
        "# Annotation ledger review report",
        "",
        "Generated from append-only `review_items.jsonl` and "
        "`review_resolutions.jsonl`.",
        "",
    ]
    for item in sorted(items, key=lambda row: row["item_id"]):
        item_resolutions = sorted(
            by_item.get(item["item_id"], []),
            key=lambda row: row["resolution_id"],
        )
        status = (
            item_resolutions[-1]["decision"] if item_resolutions else "pending"
        )
        lines.extend(
            [
                f"## {item['item_id']} — {item['category']}",
                "",
                str(item["question"]),
                "",
                f"Derived status: `{status}`.",
                "",
            ]
        )
        for resolution in item_resolutions:
            lines.extend(
                [
                    (
                        f"- `{resolution['resolution_id']}`: "
                        f"{resolution['decision']} by "
                        f"{resolution['decided_by']} "
                        f"({resolution['resolution_date']}; "
                        f"{resolution['resolution_date_precision']}). "
                        f"{resolution['rationale']}"
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def publish_review_report(root: Path = LEDGER_ROOT) -> dict[str, Any]:
    report = render_review_report(root)
    path = Path(root) / "review" / "review-report.md"
    atomic_write_text(path, report)
    return {"status": "rendered", "path": str(path), "sha256": sha256_text(report)}
