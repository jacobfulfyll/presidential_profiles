"""Draft constituency pilot helpers over canonical paragraph keys.

The generic Phase 1 execution path lives in :mod:`annotation_workflow`.  This
module retains the pre-ledger pilot helpers for compatibility and supplies the
draft rubric/schema used by the registry.  It is not a publication-v1 surface,
and Phase 1 creates no production constituency dataset.

The quantity is narrow: whose authority, welfare, rights, or interests does the
presidential voice explicitly invoke as a constituency for presidential
action?  Audience, praise, subject matter, and an unresolved ``we`` are not
automatically constituency claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "constituencies"
DEFAULT_DATASET_ID = "constituency-draft-pilot"
DEFAULT_PARAGRAPHS = (
    PROJECT_ROOT / "data" / "corpus_corrections" / "canonical_paragraphs_v1.parquet"
)
DEFAULT_SPEECHES = (
    PROJECT_ROOT / "data" / "corpus_corrections" / "canonical_speeches_v1.parquet"
)
DEFAULT_EXCLUSIONS = (
    PROJECT_ROOT / "data" / "corpus_corrections" / "exclusions_v1.parquet"
)
DEFAULT_CORPUS_META = (
    PROJECT_ROOT / "data" / "corpus_corrections" / "meta_v1.json"
)
DEFAULT_SPEECH_ANNOTATIONS = (
    PROJECT_ROOT / "data" / "llm_annotations" / "speech_annotations.parquet"
)

PROTOCOL_VERSION = "constituency-terminal-protocol-draft"
ANNOTATION_VERSION = "constituency-draft-2026-07-24"
SCHEMA_VERSION = "constituency-claim-schema-draft-2026-07-24"
PROMPT_VERSION = "constituency-rubric-draft-2026-07-24"

OUTCOMES = ("claim", "none", "unclear")
GROUP_TYPES = (
    "national_public",
    "regional_sectional",
    "racial_ethnic_legal_status",
    "economic_occupational",
    "military_veteran",
    "party_movement",
    "religious",
    "age_gender_family",
    "foreign_people",
    "other",
)
RELATIONS = (
    "source_of_authority",
    "represented_constituency",
    "protected_group",
    "intended_beneficiary",
    "included_national_member",
)
CERTAINTIES = ("explicit", "context_resolved")
PASS_KINDS = ("primary", "agreement")
SCOPES = ("full", "pilot")

# Registry-facing identity.  The legacy helper still accepts ``full`` in
# isolated compatibility tests, but Phase 1 only authorizes the generic pilot
# path and never invokes a full run.
LEDGER_LABEL_TYPE = "constituencies"
LEDGER_SPEC_VERSION = "draft-2026-07-24"


def initialize_ledger_pilot(**kwargs: Any) -> dict[str, Any]:
    """Initialize a draft pilot through the shared provider-neutral workflow."""
    from .annotation_workflow import initialize_run

    label_types = kwargs.pop("label_types", [LEDGER_LABEL_TYPE])
    if label_types != [LEDGER_LABEL_TYPE]:
        raise ValueError("constituency pilot wrapper accepts only constituencies")
    return initialize_run(
        label_types=label_types,
        spec_versions={LEDGER_LABEL_TYPE: LEDGER_SPEC_VERSION},
        **kwargs,
    )

SCHEMA_DEFINITION = {
    "unit": ["doc_name", "para_idx"],
    "outcomes": list(OUTCOMES),
    "claim_fields": {
        "group_text": "Exact literal wording from the target paragraph.",
        "normalized_group": "Stable human-readable group name.",
        "group_type": list(GROUP_TYPES),
        "relation": list(RELATIONS),
        "evidence_span": "Exact case-sensitive substring of the target paragraph.",
        "certainty": list(CERTAINTIES),
        "notes": "Optional short disambiguation; no historical interpretation.",
    },
    "response_label_fields": [
        "doc_name",
        "para_idx",
        "outcome",
        "claims",
        "unclear_reason",
        "rationale",
    ],
    "rules": {
        "claim": "One or more claim entries; unclear_reason must be empty.",
        "none": "No claim entries and no unclear_reason.",
        "unclear": "No claim entries and a non-empty unclear_reason.",
    },
}

PROMPT = """\
Label only the TARGET paragraph for this historical quantity:

Whose authority, welfare, rights, or interests does the presidential voice
explicitly invoke as a constituency for presidential action?

A qualifying claim explicitly does at least one of the following:
- treats a named group as the source of governmental or presidential authority;
- says the presidential or national government speaks, stands, or acts for it;
- claims a duty to protect its rights, security, liberty, or welfare;
- identifies it as an intended beneficiary of action or policy; or
- explicitly includes it within the nation, citizenry, or governing people.

Do not infer a claim from audience, praise, a favorable mention, subject matter,
a named adversary, or a generic/ambiguous first-person plural. Neighboring
paragraphs may resolve the referent of wording in the TARGET paragraph, but
every saved group_text and evidence_span must be an exact case-sensitive
substring of the TARGET paragraph. Use certainty=context_resolved only when the
neighboring context is necessary to identify the group. A quotation counts only
when the presidential voice adopts it; merely reporting another speaker does
not. Multiple qualifying groups or relations may be recorded. Use none when no
qualifying claim is present. Use unclear only for a genuine ambiguity and give
a short reason. Keep notes descriptive rather than interpretive.

Return only the response JSON object described by response_template. Do not
include Markdown or commentary.
"""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


SCHEMA_SHA256 = hashlib.sha256(
    _canonical_json(SCHEMA_DEFINITION).encode("utf-8")
).hexdigest()
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def _validate_identifier(value: str, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field} is required")
    if len(value) > 160 or any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} is invalid")
    return value


def _validate_dataset_id(value: str) -> str:
    value = _validate_identifier(value, "dataset_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError(
            "dataset_id must contain only letters, numbers, dot, underscore, or hyphen"
        )
    return value


def _dataset_dir(dataset_id: str, root: Path = DATA_ROOT) -> Path:
    return Path(root) / _validate_dataset_id(dataset_id)


def _manifest_path(dataset_id: str, root: Path = DATA_ROOT) -> Path:
    return _dataset_dir(dataset_id, root) / "dataset.json"


def _run_dir(dataset_dir: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{24}", run_id):
        raise ValueError("annotation_run_id is invalid")
    return dataset_dir / "runs" / run_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_source_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


@contextmanager
def _dataset_lock(dataset_dir: Path) -> Iterator[None]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    lock = dataset_dir / ".constituency-labeling.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"dataset is locked: {lock}. Inspect the active process before removing it."
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


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("keys", payload.get("exclusions", payload.get("rows")))
        if not isinstance(payload, list):
            raise ValueError(f"{path}: JSON table must be a list or contain keys/rows")
        return pd.DataFrame(payload)
    raise ValueError(f"unsupported table format: {path.suffix}")


def _require_unique(frame: pd.DataFrame, key: list[str], name: str) -> None:
    missing = [column for column in key if column not in frame.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    duplicates = frame[frame.duplicated(key, keep=False)]
    if not duplicates.empty:
        sample = duplicates[key].head(5).to_dict("records")
        raise ValueError(f"{name}: duplicate keys on {key}: {sample}")


def _load_exclusions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["scope", "doc_name", "para_idx", "reason"])
    frame = _read_table(path).copy()
    if "doc_name" not in frame.columns:
        raise ValueError("exclusions: missing doc_name")
    if "scope" not in frame.columns:
        frame["scope"] = "paragraph"
    if "para_idx" not in frame.columns:
        frame["para_idx"] = pd.NA
    if "reason" not in frame.columns:
        frame["reason"] = ""
    unknown = set(frame["scope"].dropna().astype(str)) - {"paragraph", "speech"}
    if unknown:
        raise ValueError(f"exclusions: unknown scope values {sorted(unknown)}")
    paragraph_rows = frame[frame["scope"].eq("paragraph")]
    if paragraph_rows["para_idx"].isna().any():
        raise ValueError("exclusions: paragraph rows require para_idx")
    if not paragraph_rows.empty:
        paragraph_rows = paragraph_rows.assign(
            para_idx=paragraph_rows["para_idx"].astype(int)
        )
        _require_unique(paragraph_rows, ["doc_name", "para_idx"], "exclusions")
    speech_rows = frame[frame["scope"].eq("speech")]
    if speech_rows["doc_name"].duplicated().any():
        raise ValueError("exclusions: duplicate speech doc_name")
    return frame


def _validate_excluded_keys(
    canonical: pd.DataFrame, exclusions: pd.DataFrame
) -> None:
    canonical_keys = set(
        zip(canonical["doc_name"].astype(str), canonical["para_idx"].astype(int))
    )
    paragraph_rows = exclusions[exclusions["scope"].eq("paragraph")]
    excluded_keys = set(
        zip(
            paragraph_rows["doc_name"].astype(str),
            paragraph_rows["para_idx"].astype(int),
        )
    )
    overlap = canonical_keys & excluded_keys
    if overlap:
        raise ValueError(
            f"canonical paragraphs still contain declared exclusions: {sorted(overlap)[:5]}"
        )
    excluded_docs = set(
        exclusions.loc[exclusions["scope"].eq("speech"), "doc_name"].astype(str)
    )
    overlap_docs = excluded_docs & set(canonical["doc_name"].astype(str))
    if overlap_docs:
        raise ValueError(
            f"canonical paragraphs still contain excluded speeches: {sorted(overlap_docs)[:5]}"
        )


def _load_labeling_frame(
    paragraphs_path: Path,
    speeches_path: Path,
    speech_annotations_path: Path,
    exclusions_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paragraphs = _read_table(paragraphs_path).copy()
    speeches = _read_table(speeches_path).copy()
    speech_annotations = _read_table(speech_annotations_path).copy()

    _require_unique(paragraphs, ["doc_name", "para_idx"], "paragraphs")
    _require_unique(speeches, ["doc_name"], "speeches")
    _require_unique(speech_annotations, ["doc_name"], "speech annotations")

    paragraph_columns = {"doc_name", "para_idx", "text", "word_count"}
    speech_columns = {"doc_name", "president", "year", "title"}
    annotation_columns = {"doc_name", "speech_type"}
    if missing := paragraph_columns - set(paragraphs.columns):
        raise ValueError(f"paragraphs: missing columns {sorted(missing)}")
    if missing := speech_columns - set(speeches.columns):
        raise ValueError(f"speeches: missing columns {sorted(missing)}")
    if missing := annotation_columns - set(speech_annotations.columns):
        raise ValueError(f"speech annotations: missing columns {sorted(missing)}")

    if paragraphs[["doc_name", "text"]].isna().any().any():
        raise ValueError("paragraphs: doc_name and text must be non-null")
    if speeches[list(speech_columns)].isna().any().any():
        raise ValueError("speeches: labeling metadata must be non-null")
    if speech_annotations[list(annotation_columns)].isna().any().any():
        raise ValueError("speech annotations: doc_name and speech_type must be non-null")

    merged = paragraphs.merge(
        speeches[["doc_name", "president", "year", "title"]],
        on="doc_name",
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(paragraphs) or merged[
        ["president", "year", "title"]
    ].isna().any().any():
        raise ValueError("paragraphs and speeches have divergent key sets")
    merged = merged.merge(
        speech_annotations[["doc_name", "speech_type"]],
        on="doc_name",
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(paragraphs) or merged["speech_type"].isna().any():
        raise ValueError("canonical speeches lack speech_type annotations")

    merged["doc_name"] = merged["doc_name"].astype(str)
    merged["para_idx"] = merged["para_idx"].astype(int)
    merged["text"] = merged["text"].astype(str)
    merged["word_count"] = merged["word_count"].astype(int)
    merged["president"] = merged["president"].astype(str)
    merged["year"] = merged["year"].astype(int)
    merged["title"] = merged["title"].astype(str)
    merged["speech_type"] = merged["speech_type"].astype(str)
    merged = merged.sort_values(["doc_name", "para_idx"]).reset_index(drop=True)

    exclusions = _load_exclusions(exclusions_path)
    _validate_excluded_keys(merged, exclusions)
    return merged, exclusions


def _fingerprints(frame: pd.DataFrame) -> dict[str, Any]:
    key_digest = hashlib.sha256()
    corpus_digest = hashlib.sha256()
    columns = [
        "doc_name",
        "para_idx",
        "text",
        "word_count",
        "president",
        "year",
        "title",
        "speech_type",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        record = dict(zip(columns, row))
        key_digest.update(
            f"{record['doc_name']}\x1f{record['para_idx']}\n".encode("utf-8")
        )
        corpus_digest.update((_canonical_json(record) + "\n").encode("utf-8"))
    return {
        "n_paragraphs": int(len(frame)),
        "n_speeches": int(frame["doc_name"].nunique()),
        "key_sha256": key_digest.hexdigest(),
        "canonical_corpus_sha256": corpus_digest.hexdigest(),
    }


def _select_queue(
    frame: pd.DataFrame, selection_keys_path: Path | None
) -> pd.DataFrame:
    if selection_keys_path is None:
        return frame.copy()
    keys = _read_table(selection_keys_path).copy()
    _require_unique(keys, ["doc_name", "para_idx"], "selection keys")
    selected_keys = set(
        zip(keys["doc_name"].astype(str), keys["para_idx"].astype(int))
    )
    corpus_keys = set(zip(frame["doc_name"], frame["para_idx"]))
    unknown = selected_keys - corpus_keys
    if unknown:
        raise ValueError(
            f"selection keys are outside the canonical corpus: {sorted(unknown)[:5]}"
        )
    selected = frame[
        frame.apply(
            lambda row: (str(row["doc_name"]), int(row["para_idx"]))
            in selected_keys,
            axis=1,
        )
    ].copy()
    if len(selected) != len(selected_keys):
        raise ValueError("selection key-set mismatch")
    return selected.sort_values(["doc_name", "para_idx"]).reset_index(drop=True)


def _exclusions_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    columns = sorted(frame.columns)
    rows: list[dict[str, Any]] = []
    for values in frame[columns].itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for column, value in zip(columns, values):
            if pd.isna(value):
                record[column] = None
            elif hasattr(value, "item"):
                record[column] = value.item()
            else:
                record[column] = value
        rows.append(record)
    for record in sorted(rows, key=_canonical_json):
        digest.update((_canonical_json(record) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _validate_upstream_artifacts(
    upstream_meta: dict[str, Any],
    *,
    paragraphs_path: Path,
    speeches_path: Path,
    fingerprints: dict[str, Any],
) -> None:
    """Bind Track A's declared identity to the exact files being labeled."""
    output_fingerprints = upstream_meta.get("output_fingerprints")
    if output_fingerprints is not None:
        if not isinstance(output_fingerprints, dict):
            raise ValueError(
                "upstream corpus metadata has invalid output_fingerprints"
            )
        for path in (paragraphs_path, speeches_path):
            expected = output_fingerprints.get(path.name)
            if not isinstance(expected, str) or not expected:
                raise ValueError(
                    "upstream corpus metadata lacks an output fingerprint for "
                    f"{path.name}"
                )
            if _sha256_file(path) != expected:
                raise ValueError(
                    f"upstream output fingerprint mismatch for {path.name}"
                )

    canonical_counts = upstream_meta.get("canonical_counts")
    if canonical_counts is not None:
        if not isinstance(canonical_counts, dict):
            raise ValueError(
                "upstream corpus metadata has invalid canonical_counts"
            )
        expected_counts = {
            "paragraphs": fingerprints["n_paragraphs"],
            "speeches": fingerprints["n_speeches"],
        }
        for field, actual in expected_counts.items():
            if canonical_counts.get(field) != actual:
                raise ValueError(
                    f"upstream canonical {field} count mismatch: "
                    f"{canonical_counts.get(field)!r} != {actual}"
                )


def initialize(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    paragraphs_path: Path = DEFAULT_PARAGRAPHS,
    speeches_path: Path = DEFAULT_SPEECHES,
    speech_annotations_path: Path = DEFAULT_SPEECH_ANNOTATIONS,
    exclusions_path: Path | None = DEFAULT_EXCLUSIONS,
    corpus_meta_path: Path | None = DEFAULT_CORPUS_META,
    selection_keys_path: Path | None = None,
    annotation_version: str = ANNOTATION_VERSION,
    scope: str = "full",
    root: Path = DATA_ROOT,
) -> dict[str, Any]:
    """Fingerprint the canonical universe and create an empty dataset."""
    dataset_id = _validate_dataset_id(dataset_id)
    annotation_version = _validate_identifier(
        annotation_version, "annotation_version"
    )
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if scope == "pilot" and selection_keys_path is None:
        raise ValueError("pilot scope requires a persisted --keys table")
    if scope == "full" and selection_keys_path is not None:
        raise ValueError("full scope cannot use a partial --keys table")
    dataset_dir = _dataset_dir(dataset_id, root)
    if dataset_dir.exists():
        raise FileExistsError(
            f"{dataset_dir} already exists; annotation datasets are immutable"
        )

    frame, exclusions = _load_labeling_frame(
        Path(paragraphs_path),
        Path(speeches_path),
        Path(speech_annotations_path),
        Path(exclusions_path) if exclusions_path is not None else None,
    )
    fingerprints = _fingerprints(frame)
    queue = _select_queue(
        frame, Path(selection_keys_path) if selection_keys_path is not None else None
    )
    queue_fingerprints = _fingerprints(queue)
    upstream_meta: dict[str, Any] | None = None
    upstream_meta_sha256: str | None = None
    if corpus_meta_path is not None:
        corpus_meta_path = Path(corpus_meta_path)
        if not corpus_meta_path.exists():
            raise FileNotFoundError(corpus_meta_path)
        upstream_meta = json.loads(corpus_meta_path.read_text(encoding="utf-8"))
        upstream_meta_sha256 = _sha256_file(corpus_meta_path)
        if "canonical_corpus_fingerprint" not in upstream_meta:
            raise ValueError(
                "upstream corpus metadata lacks canonical_corpus_fingerprint"
            )
        _validate_upstream_artifacts(
            upstream_meta,
            paragraphs_path=Path(paragraphs_path),
            speeches_path=Path(speeches_path),
            fingerprints=fingerprints,
        )
    canonical_identity = (
        upstream_meta["canonical_corpus_fingerprint"]
        if upstream_meta is not None
        else fingerprints["canonical_corpus_sha256"]
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "annotation_version": annotation_version,
        "scope": scope,
        "historical_quantity": (
            "Whose authority, welfare, rights, or interests does the presidential "
            "voice explicitly invoke as a constituency for presidential action?"
        ),
        "source_paths": {
            "paragraphs": _portable_source_path(Path(paragraphs_path)),
            "speeches": _portable_source_path(Path(speeches_path)),
            "speech_annotations": _portable_source_path(
                Path(speech_annotations_path)
            ),
            "exclusions": (
                _portable_source_path(Path(exclusions_path))
                if exclusions_path is not None
                else None
            ),
            "corpus_meta": (
                _portable_source_path(Path(corpus_meta_path))
                if corpus_meta_path is not None
                else None
            ),
            "selection_keys": (
                _portable_source_path(Path(selection_keys_path))
                if selection_keys_path is not None
                else None
            ),
        },
        "corpus": {
            **fingerprints,
            "canonical_corpus_fingerprint": canonical_identity,
            "labeling_input_sha256": fingerprints["canonical_corpus_sha256"],
            "exclusions_sha256": _exclusions_sha256(exclusions),
            "upstream_meta_sha256": upstream_meta_sha256,
        },
        "queue": {
            "n_paragraphs": queue_fingerprints["n_paragraphs"],
            "n_speeches": queue_fingerprints["n_speeches"],
            "key_sha256": queue_fingerprints["key_sha256"],
            "queue_content_sha256": queue_fingerprints[
                "canonical_corpus_sha256"
            ],
        },
        "schema": {
            "version": SCHEMA_VERSION,
            "sha256": SCHEMA_SHA256,
            "definition": SCHEMA_DEFINITION,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "sha256": PROMPT_SHA256,
            "text": PROMPT,
        },
        "storage": {
            "annotation_standard": (
                "offline AI-assisted labels with row-level model provenance; "
                "not human-validated unless separately adjudicated"
            ),
            "network_calls": False,
            "frozen_paid_artifacts_modified": False,
        },
    }
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "runs").mkdir()
    (dataset_dir / "exports").mkdir()
    _atomic_write_json(dataset_dir / "dataset.json", manifest)
    return manifest


def _read_manifest(
    dataset_id: str, root: Path = DATA_ROOT, *, verify_code: bool = True
) -> dict[str, Any]:
    path = _manifest_path(dataset_id, root)
    if not path.exists():
        raise FileNotFoundError(f"dataset is not initialized: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError("dataset manifest identity drift")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("terminal protocol drift")
    if verify_code:
        if manifest.get("schema", {}).get("version") != SCHEMA_VERSION:
            raise ValueError("schema-version drift")
        if manifest.get("schema", {}).get("sha256") != SCHEMA_SHA256:
            raise ValueError("schema-hash drift")
        if manifest.get("prompt", {}).get("version") != PROMPT_VERSION:
            raise ValueError("prompt-version drift")
        if manifest.get("prompt", {}).get("sha256") != PROMPT_SHA256:
            raise ValueError("prompt-hash drift")
    return manifest


def _live_frame(
    manifest: dict[str, Any], *, verify_fingerprint: bool = True
) -> pd.DataFrame:
    paths = manifest["source_paths"]
    frame, exclusions = _load_labeling_frame(
        _resolve_source_path(paths["paragraphs"]),
        _resolve_source_path(paths["speeches"]),
        _resolve_source_path(paths["speech_annotations"]),
        _resolve_source_path(paths["exclusions"]) if paths.get("exclusions") else None,
    )
    if verify_fingerprint:
        expected = manifest["corpus"]
        actual = _fingerprints(frame)
        for field in (
            "n_paragraphs",
            "n_speeches",
            "key_sha256",
            "canonical_corpus_sha256",
        ):
            if actual[field] != expected.get(field):
                raise ValueError(
                    f"canonical corpus fingerprint drift in {field}: "
                    f"{expected.get(field)!r} -> {actual[field]!r}"
                )
        if _exclusions_sha256(exclusions) != expected.get("exclusions_sha256"):
            raise ValueError("exclusions fingerprint drift")
        if paths.get("corpus_meta"):
            meta_path = _resolve_source_path(paths["corpus_meta"])
            if _sha256_file(meta_path) != expected.get("upstream_meta_sha256"):
                raise ValueError("upstream corpus metadata drift")
            upstream_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if upstream_meta.get("canonical_corpus_fingerprint") != expected.get(
                "canonical_corpus_fingerprint"
            ):
                raise ValueError("upstream canonical corpus fingerprint drift")
    queue = _select_queue(
        frame,
        _resolve_source_path(paths["selection_keys"])
        if paths.get("selection_keys")
        else None,
    )
    queue_fingerprints = _fingerprints(queue)
    expected_queue = manifest.get("queue", {})
    for field, actual in (
        ("n_paragraphs", queue_fingerprints["n_paragraphs"]),
        ("n_speeches", queue_fingerprints["n_speeches"]),
        ("key_sha256", queue_fingerprints["key_sha256"]),
        (
            "queue_content_sha256",
            queue_fingerprints["canonical_corpus_sha256"],
        ),
    ):
        if actual != expected_queue.get(field):
            raise ValueError(
                f"labeling queue fingerprint drift in {field}: "
                f"{expected_queue.get(field)!r} -> {actual!r}"
            )
    return queue


def _key_tuple(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["doc_name"]), int(record["para_idx"])


def _stable_shard(doc_name: str, para_idx: int, shard_count: int) -> int:
    payload = f"{doc_name}\x1f{para_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % shard_count


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")


def _run_identity(
    *,
    manifest: dict[str, Any],
    annotator_id: str,
    model_id: str,
    agent_id: str | None,
    pass_kind: str,
    reference_run_id: str | None,
    shard_index: int,
    shard_count: int,
) -> str:
    identity = {
        "dataset_id": manifest["dataset_id"],
        "annotation_version": manifest["annotation_version"],
        "annotator_id": annotator_id,
        "model_id": model_id,
        "agent_id": agent_id or "",
        "pass_kind": pass_kind,
        "reference_run_id": reference_run_id,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "schema_sha256": SCHEMA_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "canonical_corpus_fingerprint": manifest["corpus"][
            "canonical_corpus_fingerprint"
        ],
        "labeling_input_sha256": manifest["corpus"]["labeling_input_sha256"],
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:24]


def _read_run(dataset_dir: Path, run_id: str) -> dict[str, Any]:
    path = _run_dir(dataset_dir, run_id) / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown annotation_run_id: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_run(
    *,
    dataset_dir: Path,
    manifest: dict[str, Any],
    annotator_id: str,
    model_id: str,
    agent_id: str | None,
    pass_kind: str,
    reference_run_id: str | None,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    annotator_id = _validate_identifier(annotator_id, "annotator_id")
    model_id = _validate_identifier(model_id, "model_id")
    agent_id = (
        _validate_identifier(agent_id, "agent_id") if agent_id is not None else None
    )
    if pass_kind not in PASS_KINDS:
        raise ValueError(f"pass_kind must be one of {PASS_KINDS}")
    _validate_shard(shard_index, shard_count)
    if pass_kind == "agreement" and not reference_run_id:
        raise ValueError("agreement pass requires reference_run_id")
    if pass_kind == "primary" and reference_run_id:
        raise ValueError("primary pass cannot have reference_run_id")
    if reference_run_id:
        reference = _read_run(dataset_dir, reference_run_id)
        if reference["canonical_corpus_fingerprint"] != manifest["corpus"][
            "canonical_corpus_fingerprint"
        ]:
            raise ValueError("reference run corpus fingerprint drift")

    run_id = _run_identity(
        manifest=manifest,
        annotator_id=annotator_id,
        model_id=model_id,
        agent_id=agent_id,
        pass_kind=pass_kind,
        reference_run_id=reference_run_id,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    run = {
        "protocol_version": PROTOCOL_VERSION,
        "annotation_run_id": run_id,
        "dataset_id": manifest["dataset_id"],
        "annotation_version": manifest["annotation_version"],
        "annotator_id": annotator_id,
        "model_id": model_id,
        "agent_id": agent_id,
        "pass_kind": pass_kind,
        "reference_run_id": reference_run_id,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_id": f"{shard_index}-of-{shard_count}",
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": SCHEMA_SHA256,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "canonical_corpus_fingerprint": manifest["corpus"][
            "canonical_corpus_fingerprint"
        ],
        "labeling_input_sha256": manifest["corpus"]["labeling_input_sha256"],
    }
    run_dir = _run_dir(dataset_dir, run_id)
    path = run_dir / "run.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != run:
            raise ValueError("annotation run identity drift")
    else:
        (run_dir / "assignments").mkdir(parents=True)
        (run_dir / "responses").mkdir()
        _atomic_write_json(path, run)
    return run


def _run_paths(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "runs").glob("*/run.json"))


def _assignment_paths(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "runs").glob("*/assignments/*.json"))


def _response_paths(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "runs").glob("*/responses/*.json"))


def _labels_from_response(path: Path) -> list[dict[str, Any]]:
    response = json.loads(path.read_text(encoding="utf-8"))
    labels = response.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"{path}: response labels must be a list")
    return labels


def _labels_for_run(dataset_dir: Path, run_id: str) -> list[dict[str, Any]]:
    response_dir = _run_dir(dataset_dir, run_id) / "responses"
    labels: list[dict[str, Any]] = []
    for path in sorted(response_dir.glob("*.json")):
        labels.extend(_labels_from_response(path))
    return labels


def _assigned_keys_for_run(dataset_dir: Path, run_id: str) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for path in sorted((_run_dir(dataset_dir, run_id) / "assignments").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        keys.update(_key_tuple(item["target"]) for item in batch["paragraphs"])
    return keys


def _completed_keys_for_run(dataset_dir: Path, run_id: str) -> set[tuple[str, int]]:
    return {_key_tuple(label) for label in _labels_for_run(dataset_dir, run_id)}


def _primary_assigned_keys(dataset_dir: Path) -> set[tuple[str, int]]:
    primary_runs = {
        json.loads(path.read_text(encoding="utf-8"))["annotation_run_id"]
        for path in _run_paths(dataset_dir)
        if json.loads(path.read_text(encoding="utf-8")).get("pass_kind") == "primary"
    }
    keys: set[tuple[str, int]] = set()
    for path in _assignment_paths(dataset_dir):
        run_id = path.parents[1].name
        if run_id not in primary_runs:
            continue
        batch = json.loads(path.read_text(encoding="utf-8"))
        keys.update(_key_tuple(item["target"]) for item in batch["paragraphs"])
    return keys


def _open_assignment(run_dir: Path) -> dict[str, Any] | None:
    open_batches = []
    for path in sorted((run_dir / "assignments").glob("*.json")):
        response = run_dir / "responses" / path.name
        if not response.exists():
            open_batches.append(path)
    if len(open_batches) > 1:
        raise ValueError(
            f"run has multiple open batches: {[path.name for path in open_batches]}"
        )
    if not open_batches:
        return None
    return json.loads(open_batches[0].read_text(encoding="utf-8"))


def _context_records(
    frame: pd.DataFrame, key: tuple[str, int], context_window: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    speech = frame[frame["doc_name"].eq(key[0])].sort_values("para_idx")
    positions = {
        (str(row.doc_name), int(row.para_idx)): position
        for position, row in enumerate(speech.itertuples(index=False))
    }
    position = positions[key]

    def records(block: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "doc_name": str(row.doc_name),
                "para_idx": int(row.para_idx),
                "text": str(row.text),
            }
            for row in block.itertuples(index=False)
        ]

    before = speech.iloc[max(0, position - context_window) : position]
    after = speech.iloc[position + 1 : position + 1 + context_window]
    return records(before), records(after)


def _response_template(
    run: dict[str, Any], keys: list[tuple[str, int]], batch_id: str
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "annotation_run_id": run["annotation_run_id"],
        "batch_id": batch_id,
        "labels": [
            {
                "doc_name": doc_name,
                "para_idx": para_idx,
                "outcome": None,
                "claims": [
                    {
                        "group_text": "",
                        "normalized_group": "",
                        "group_type": None,
                        "relation": None,
                        "evidence_span": "",
                        "certainty": None,
                        "notes": "",
                    }
                ],
                "unclear_reason": "",
                "rationale": "",
            }
            for doc_name, para_idx in keys
        ],
    }


def _batch_hash(batch: dict[str, Any]) -> str:
    payload = dict(batch)
    payload.pop("batch_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def next_batch(
    *,
    dataset_id: str,
    annotator_id: str,
    model_id: str,
    agent_id: str | None = None,
    pass_kind: str = "primary",
    reference_run_id: str | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    batch_size: int = 5,
    context_window: int = 1,
    root: Path = DATA_ROOT,
) -> dict[str, Any]:
    """Claim or resume one stable batch and return its terminal JSON object."""
    if batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20")
    if context_window < 0 or context_window > 2:
        raise ValueError("context_window must be between 0 and 2")
    manifest = _read_manifest(dataset_id, root)
    frame = _live_frame(manifest)
    dataset_dir = _dataset_dir(dataset_id, root)

    with _dataset_lock(dataset_dir):
        run = _ensure_run(
            dataset_dir=dataset_dir,
            manifest=manifest,
            annotator_id=annotator_id,
            model_id=model_id,
            agent_id=agent_id,
            pass_kind=pass_kind,
            reference_run_id=reference_run_id,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        run_dir = _run_dir(dataset_dir, run["annotation_run_id"])
        if existing := _open_assignment(run_dir):
            return existing

        keys = [
            (str(row.doc_name), int(row.para_idx))
            for row in frame.itertuples(index=False)
            if _stable_shard(str(row.doc_name), int(row.para_idx), shard_count)
            == shard_index
        ]
        if pass_kind == "primary":
            unavailable = _primary_assigned_keys(dataset_dir)
            keys = [key for key in keys if key not in unavailable]
        else:
            assert reference_run_id is not None
            reference_keys = _completed_keys_for_run(dataset_dir, reference_run_id)
            assigned = _assigned_keys_for_run(dataset_dir, run["annotation_run_id"])
            keys = [
                key
                for key in keys
                if key in reference_keys and key not in assigned
            ]
        keys = sorted(keys)[:batch_size]
        if not keys:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "annotation_run_id": run["annotation_run_id"],
                "status": "complete",
                "remaining": 0,
            }

        lookup = frame.set_index(["doc_name", "para_idx"])
        if not lookup.index.is_unique:
            raise ValueError("canonical paragraph keys are no longer unique")
        paragraphs: list[dict[str, Any]] = []
        for key in keys:
            row = lookup.loc[key]
            before, after = _context_records(frame, key, context_window)
            paragraphs.append(
                {
                    "metadata": {
                        "year": int(row["year"]),
                        "president": str(row["president"]),
                        "title": str(row["title"]),
                        "speech_type": str(row["speech_type"]),
                    },
                    "context_before": before,
                    "target": {
                        "doc_name": key[0],
                        "para_idx": key[1],
                        "text": str(row["text"]),
                    },
                    "context_after": after,
                }
            )
        batch_identity = {
            "annotation_run_id": run["annotation_run_id"],
            "keys": keys,
            "context_window": context_window,
            "canonical_corpus_fingerprint": manifest["corpus"][
                "canonical_corpus_fingerprint"
            ],
        }
        batch_id = hashlib.sha256(
            _canonical_json(batch_identity).encode("utf-8")
        ).hexdigest()[:24]
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "batch_id": batch_id,
            "context_window": context_window,
            "annotation_run": run,
            "rubric": {
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "prompt": PROMPT,
                "schema_version": SCHEMA_VERSION,
                "schema_sha256": SCHEMA_SHA256,
                "schema": SCHEMA_DEFINITION,
            },
            "paragraphs": paragraphs,
            "response_template": _response_template(run, keys, batch_id),
        }
        request["batch_sha256"] = _batch_hash(request)
        _atomic_write_json(run_dir / "assignments" / f"{batch_id}.json", request)
        return request


def _require_exact_fields(
    value: dict[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name}: field mismatch; missing={missing}, extra={extra}")


def _validate_response_label(
    raw: dict[str, Any], target_text: str
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each label must be an object")
    _require_exact_fields(
        raw,
        {
            "doc_name",
            "para_idx",
            "outcome",
            "claims",
            "unclear_reason",
            "rationale",
        },
        "label",
    )
    outcome = raw["outcome"]
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome!r}")
    if not isinstance(raw["claims"], list):
        raise ValueError("claims must be a list")
    unclear_reason = str(raw["unclear_reason"]).strip()
    rationale = str(raw["rationale"]).strip()
    if outcome == "claim":
        if not raw["claims"]:
            raise ValueError("claim outcome requires at least one claim")
        if unclear_reason:
            raise ValueError("claim outcome cannot carry unclear_reason")
    elif outcome == "none":
        if raw["claims"] or unclear_reason:
            raise ValueError("none outcome requires empty claims and unclear_reason")
    else:
        if raw["claims"]:
            raise ValueError("unclear outcome cannot carry claims")
        if not unclear_reason:
            raise ValueError("unclear outcome requires unclear_reason")

    claims: list[dict[str, Any]] = []
    signatures: set[str] = set()
    claim_fields = {
        "group_text",
        "normalized_group",
        "group_type",
        "relation",
        "evidence_span",
        "certainty",
        "notes",
    }
    for index, claim in enumerate(raw["claims"]):
        if not isinstance(claim, dict):
            raise ValueError(f"claim {index}: must be an object")
        _require_exact_fields(claim, claim_fields, f"claim {index}")
        cleaned = {
            "group_text": str(claim["group_text"]).strip(),
            "normalized_group": str(claim["normalized_group"]).strip(),
            "group_type": claim["group_type"],
            "relation": claim["relation"],
            "evidence_span": str(claim["evidence_span"]).strip(),
            "certainty": claim["certainty"],
            "notes": str(claim["notes"]).strip(),
        }
        for field in ("group_text", "normalized_group", "evidence_span"):
            if not cleaned[field]:
                raise ValueError(f"claim {index}: {field} is required")
        if cleaned["group_type"] not in GROUP_TYPES:
            raise ValueError(
                f"claim {index}: unknown group_type {cleaned['group_type']!r}"
            )
        if cleaned["relation"] not in RELATIONS:
            raise ValueError(
                f"claim {index}: unknown relation {cleaned['relation']!r}"
            )
        if cleaned["certainty"] not in CERTAINTIES:
            raise ValueError(
                f"claim {index}: unknown certainty {cleaned['certainty']!r}"
            )
        if cleaned["group_text"] not in target_text:
            raise ValueError(
                f"claim {index}: group_text is not an exact target substring"
            )
        if cleaned["evidence_span"] not in target_text:
            raise ValueError(
                f"claim {index}: evidence_span is not an exact target substring"
            )
        if cleaned["group_text"] not in cleaned["evidence_span"]:
            raise ValueError(
                f"claim {index}: evidence_span must contain group_text"
            )
        signature = _canonical_json(cleaned)
        if signature in signatures:
            raise ValueError(f"claim {index}: duplicate claim")
        signatures.add(signature)
        claims.append(cleaned)
    return {
        "doc_name": str(raw["doc_name"]),
        "para_idx": int(raw["para_idx"]),
        "outcome": outcome,
        "claims": claims,
        "unclear_reason": unclear_reason,
        "rationale": rationale,
    }


def _validate_assigned_batch(
    batch: dict[str, Any],
    *,
    run: dict[str, Any],
    manifest: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    if batch.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("batch protocol-version drift")
    if batch.get("annotation_run") != run:
        raise ValueError("batch annotation-run identity drift")
    rubric = batch.get("rubric", {})
    if rubric.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("batch prompt-version drift")
    if rubric.get("prompt_sha256") != PROMPT_SHA256:
        raise ValueError("batch prompt-hash drift")
    if rubric.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("batch schema-version drift")
    if rubric.get("schema_sha256") != SCHEMA_SHA256:
        raise ValueError("batch schema-hash drift")
    context_window = batch.get("context_window")
    if not isinstance(context_window, int) or not 0 <= context_window <= 2:
        raise ValueError("batch context-window drift")
    paragraphs = batch.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("batch paragraphs must be a non-empty list")
    keys = [_key_tuple(item.get("target", {})) for item in paragraphs]
    if len(keys) != len(set(keys)):
        raise ValueError("batch contains duplicate paragraph keys")
    expected_batch_id = hashlib.sha256(
        _canonical_json(
            {
                "annotation_run_id": run["annotation_run_id"],
                "keys": keys,
                "context_window": context_window,
                "canonical_corpus_fingerprint": manifest["corpus"][
                    "canonical_corpus_fingerprint"
                ],
            }
        ).encode("utf-8")
    ).hexdigest()[:24]
    if batch.get("batch_id") != expected_batch_id:
        raise ValueError("batch-id drift")
    if batch.get("batch_sha256") != _batch_hash(batch):
        raise ValueError("batch checksum drift")

    lookup = frame.set_index(["doc_name", "para_idx"])
    if not lookup.index.is_unique:
        raise ValueError("canonical paragraph keys are no longer unique")
    for item, key in zip(paragraphs, keys):
        if key not in lookup.index:
            raise ValueError(f"batch key is outside the canonical corpus: {key}")
        row = lookup.loc[key]
        before, after = _context_records(frame, key, context_window)
        expected = {
            "metadata": {
                "year": int(row["year"]),
                "president": str(row["president"]),
                "title": str(row["title"]),
                "speech_type": str(row["speech_type"]),
            },
            "context_before": before,
            "target": {
                "doc_name": key[0],
                "para_idx": key[1],
                "text": str(row["text"]),
            },
            "context_after": after,
        }
        if item != expected:
            raise ValueError(f"batch paragraph/context drift at {key}")
    if batch.get("response_template") != _response_template(
        run, keys, expected_batch_id
    ):
        raise ValueError("batch response-template drift")


def _find_batch(
    dataset_dir: Path,
    run_id: str,
    batch_id: str,
) -> tuple[Path, dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{24}", batch_id):
        raise ValueError("batch_id is invalid")
    path = _run_dir(dataset_dir, run_id) / "assignments" / f"{batch_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"batch was not assigned: {batch_id}")
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch.get("batch_id") != batch_id:
        raise ValueError("batch identity or checksum drift")
    return path, batch


def ingest(
    response: dict[str, Any],
    *,
    dataset_id: str,
    root: Path = DATA_ROOT,
) -> dict[str, Any]:
    """Validate one complete assigned response and save it atomically."""
    if not isinstance(response, dict):
        raise ValueError("response must be one JSON object")
    _require_exact_fields(
        response,
        {"protocol_version", "annotation_run_id", "batch_id", "labels"},
        "response",
    )
    if response["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("response protocol-version drift")
    run_id = str(response["annotation_run_id"])
    batch_id = str(response["batch_id"])
    if not isinstance(response["labels"], list):
        raise ValueError("response labels must be a list")

    manifest = _read_manifest(dataset_id, root)
    frame = _live_frame(manifest)
    dataset_dir = _dataset_dir(dataset_id, root)
    with _dataset_lock(dataset_dir):
        run = _read_run(dataset_dir, run_id)
        if run["dataset_id"] != dataset_id:
            raise ValueError("annotation run belongs to another dataset")
        _, batch = _find_batch(dataset_dir, run_id, batch_id)
        _validate_assigned_batch(
            batch, run=run, manifest=manifest, frame=frame
        )
        response_path = (
            _run_dir(dataset_dir, run_id) / "responses" / f"{batch_id}.json"
        )
        if response_path.exists():
            raise ValueError("batch was already ingested")

        targets = {
            _key_tuple(item["target"]): item for item in batch["paragraphs"]
        }
        raw_keys = [_key_tuple(label) for label in response["labels"]]
        if len(raw_keys) != len(set(raw_keys)):
            raise ValueError("response contains duplicate paragraph keys")
        if set(raw_keys) != set(targets):
            missing = sorted(set(targets) - set(raw_keys))
            extra = sorted(set(raw_keys) - set(targets))
            raise ValueError(
                f"incomplete or unassigned response; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )

        existing_run_keys = _completed_keys_for_run(dataset_dir, run_id)
        overlap = set(raw_keys) & existing_run_keys
        if overlap:
            raise ValueError(
                f"paragraph already labeled in this annotator pass: {sorted(overlap)[:5]}"
            )
        if run["pass_kind"] == "primary":
            other_primary: dict[tuple[str, int], str] = {}
            for run_path in _run_paths(dataset_dir):
                other = json.loads(run_path.read_text(encoding="utf-8"))
                if (
                    other["pass_kind"] != "primary"
                    or other["annotation_run_id"] == run_id
                ):
                    continue
                for label in _labels_for_run(
                    dataset_dir, other["annotation_run_id"]
                ):
                    other_primary[_key_tuple(label)] = other["annotation_run_id"]
            duplicate_primary = set(raw_keys) & set(other_primary)
            if duplicate_primary:
                raise ValueError(
                    "primary paragraph already completed by another annotator: "
                    f"{sorted(duplicate_primary)[:5]}"
                )

        validated: list[dict[str, Any]] = []
        for raw in response["labels"]:
            key = _key_tuple(raw)
            target = targets[key]
            label = _validate_response_label(raw, str(target["target"]["text"]))
            metadata = target["metadata"]
            label.update(
                {
                    "label_id": hashlib.sha256(
                        f"{run_id}\x1f{key[0]}\x1f{key[1]}".encode("utf-8")
                    ).hexdigest()[:24],
                    "annotation_version": run["annotation_version"],
                    "annotation_run_id": run_id,
                    "annotator_id": run["annotator_id"],
                    "model_id": run["model_id"],
                    "agent_id": run["agent_id"],
                    "pass_kind": run["pass_kind"],
                    "batch_id": batch_id,
                    "shard_id": run["shard_id"],
                    "shard_index": run["shard_index"],
                    "shard_count": run["shard_count"],
                    "schema_version": run["schema_version"],
                    "schema_sha256": run["schema_sha256"],
                    "prompt_version": run["prompt_version"],
                    "prompt_sha256": run["prompt_sha256"],
                    "canonical_corpus_fingerprint": run[
                        "canonical_corpus_fingerprint"
                    ],
                    "labeling_input_sha256": run["labeling_input_sha256"],
                    "year": int(metadata["year"]),
                    "president": str(metadata["president"]),
                    "title": str(metadata["title"]),
                    "speech_type": str(metadata["speech_type"]),
                }
            )
            validated.append(label)
        validated.sort(key=lambda label: (label["doc_name"], label["para_idx"]))
        saved = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_id": dataset_id,
            "annotation_run_id": run_id,
            "batch_id": batch_id,
            "batch_sha256": batch["batch_sha256"],
            "labels": validated,
        }
        _atomic_write_json(response_path, saved)
    return {
        "status": "ingested",
        "dataset_id": dataset_id,
        "annotation_run_id": run_id,
        "batch_id": batch_id,
        "n_labels": len(validated),
    }


def _era(year: int) -> str:
    # Lazy so ``next``/``ingest`` stay a small pandas-only terminal path.
    from .trends import ERAS

    for label, start, end in ERAS:
        if start <= int(year) <= end:
            return label
    return "outside_declared_eras"


def _scope_frame_for_run(
    frame: pd.DataFrame, dataset_dir: Path, run: dict[str, Any]
) -> pd.DataFrame:
    mask = frame.apply(
        lambda row: _stable_shard(
            str(row["doc_name"]), int(row["para_idx"]), int(run["shard_count"])
        )
        == int(run["shard_index"]),
        axis=1,
    )
    scoped = frame[mask].copy()
    if run["pass_kind"] == "agreement":
        reference_keys = _completed_keys_for_run(
            dataset_dir, run["reference_run_id"]
        )
        scoped = scoped[
            scoped.apply(
                lambda row: (str(row["doc_name"]), int(row["para_idx"]))
                in reference_keys,
                axis=1,
            )
        ]
    return scoped


def _breakdown(
    scope: pd.DataFrame, completed: set[tuple[str, int]], field: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in scope.groupby(field, sort=True):
        keys = set(zip(group["doc_name"].astype(str), group["para_idx"].astype(int)))
        done = len(keys & completed)
        rows.append(
            {
                field: str(value),
                "eligible": len(keys),
                "completed": done,
                "coverage": done / len(keys) if keys else 0.0,
            }
        )
    return rows


def status(
    *, dataset_id: str, root: Path = DATA_ROOT
) -> dict[str, Any]:
    manifest = _read_manifest(dataset_id, root)
    frame = _live_frame(manifest)
    frame = frame.assign(era=frame["year"].map(_era))
    dataset_dir = _dataset_dir(dataset_id, root)
    runs: list[dict[str, Any]] = []
    primary_keys: list[tuple[str, int]] = []
    for path in _run_paths(dataset_dir):
        run = json.loads(path.read_text(encoding="utf-8"))
        labels = _labels_for_run(dataset_dir, run["annotation_run_id"])
        completed = {_key_tuple(label) for label in labels}
        scope = _scope_frame_for_run(frame, dataset_dir, run)
        open_batches = sum(
            not (
                _run_dir(dataset_dir, run["annotation_run_id"])
                / "responses"
                / batch_path.name
            ).exists()
            for batch_path in (
                _run_dir(dataset_dir, run["annotation_run_id"]) / "assignments"
            ).glob("*.json")
        )
        counts = pd.Series(
            [label["outcome"] for label in labels], dtype="string"
        ).value_counts()
        runs.append(
            {
                **run,
                "eligible_in_shard": int(len(scope)),
                "completed": len(completed),
                "coverage": (
                    len(completed) / len(scope) if len(scope) else 1.0
                ),
                "claims": int(counts.get("claim", 0)),
                "none": int(counts.get("none", 0)),
                "unclear": int(counts.get("unclear", 0)),
                "open_batches": int(open_batches),
                "by_era": _breakdown(scope, completed, "era"),
                "by_genre": _breakdown(scope, completed, "speech_type"),
                "by_shard": [
                    {
                        "shard_id": run["shard_id"],
                        "eligible": int(len(scope)),
                        "completed": len(completed),
                    }
                ],
            }
        )
        if run["pass_kind"] == "primary":
            primary_keys.extend(completed)
    duplicate_primary = len(primary_keys) - len(set(primary_keys))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "annotation_version": manifest["annotation_version"],
        "scope": manifest["scope"],
        "canonical_paragraphs": int(len(frame)),
        "primary_completed": len(set(primary_keys)),
        "primary_missing": int(len(frame) - len(set(primary_keys))),
        "duplicate_primary_keys": duplicate_primary,
        "runs": runs,
    }


def _saved_label_semantics(label: dict[str, Any], target_text: str) -> None:
    raw = {
        field: label[field]
        for field in SCHEMA_DEFINITION["response_label_fields"]
    }
    _validate_response_label(raw, target_text)


def audit(
    *, dataset_id: str, root: Path = DATA_ROOT
) -> dict[str, Any]:
    issues: list[str] = []
    try:
        manifest = _read_manifest(dataset_id, root)
        frame = _live_frame(manifest)
    except Exception as exc:
        return {
            "status": "failed",
            "dataset_id": dataset_id,
            "issues": [str(exc)],
        }
    dataset_dir = _dataset_dir(dataset_id, root)
    lookup = frame.set_index(["doc_name", "para_idx"])
    if not lookup.index.is_unique:
        raise ValueError("canonical paragraph keys are no longer unique")
    corpus_keys = set(zip(frame["doc_name"], frame["para_idx"]))
    primary_owners: dict[tuple[str, int], str] = {}
    primary_assignment_owners: dict[tuple[str, int], str] = {}
    unclear = 0
    open_batches = 0

    for run_path in _run_paths(dataset_dir):
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run.get("schema_sha256") != SCHEMA_SHA256:
                raise ValueError("run schema-hash drift")
            if run.get("prompt_sha256") != PROMPT_SHA256:
                raise ValueError("run prompt-hash drift")
            if run.get("canonical_corpus_fingerprint") != manifest["corpus"][
                "canonical_corpus_fingerprint"
            ]:
                raise ValueError("run corpus-fingerprint drift")
            if run.get("labeling_input_sha256") != manifest["corpus"][
                "labeling_input_sha256"
            ]:
                raise ValueError("run labeling-input fingerprint drift")
            expected_run_id = _run_identity(
                manifest=manifest,
                annotator_id=run["annotator_id"],
                model_id=run["model_id"],
                agent_id=run.get("agent_id"),
                pass_kind=run["pass_kind"],
                reference_run_id=run.get("reference_run_id"),
                shard_index=int(run["shard_index"]),
                shard_count=int(run["shard_count"]),
            )
            if expected_run_id != run["annotation_run_id"]:
                raise ValueError("run identity hash drift")
            seen: set[tuple[str, int]] = set()
            seen_assignments: set[tuple[str, int]] = set()
            run_dir = _run_dir(dataset_dir, run["annotation_run_id"])
            assignment_paths = sorted((run_dir / "assignments").glob("*.json"))
            response_paths = sorted((run_dir / "responses").glob("*.json"))
            orphan_responses = {path.name for path in response_paths} - {
                path.name for path in assignment_paths
            }
            if orphan_responses:
                raise ValueError(
                    f"responses without assignments: {sorted(orphan_responses)[:5]}"
                )
            for assignment_path in assignment_paths:
                batch = json.loads(assignment_path.read_text(encoding="utf-8"))
                _validate_assigned_batch(
                    batch, run=run, manifest=manifest, frame=frame
                )
                assigned_keys = {
                    _key_tuple(item["target"]) for item in batch["paragraphs"]
                }
                duplicate_assignment = assigned_keys & seen_assignments
                if duplicate_assignment:
                    raise ValueError(
                        "paragraph assigned more than once within run: "
                        f"{sorted(duplicate_assignment)[:5]}"
                    )
                seen_assignments.update(assigned_keys)
                if run["pass_kind"] == "primary":
                    for key in assigned_keys:
                        owner = primary_assignment_owners.get(key)
                        if owner and owner != run["annotation_run_id"]:
                            raise ValueError(
                                f"primary key assigned to multiple runs: {key}"
                            )
                        primary_assignment_owners[key] = run[
                            "annotation_run_id"
                        ]
                response_path = run_dir / "responses" / assignment_path.name
                if not response_path.exists():
                    open_batches += 1
                    continue
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("batch_sha256") != batch["batch_sha256"]:
                    raise ValueError(
                        f"{response_path.name}: response-to-batch checksum drift"
                    )
                assigned = {
                    _key_tuple(item["target"]): item for item in batch["paragraphs"]
                }
                labels = response.get("labels")
                if not isinstance(labels, list):
                    raise ValueError(f"{response_path.name}: labels must be a list")
                response_keys = [_key_tuple(label) for label in labels]
                if len(response_keys) != len(set(response_keys)):
                    raise ValueError(
                        f"{response_path.name}: duplicate response keys"
                    )
                if set(response_keys) != set(assigned):
                    raise ValueError(
                        f"{response_path.name}: incomplete or unassigned keys"
                    )
                for label in labels:
                    key = _key_tuple(label)
                    if key in seen:
                        raise ValueError(f"duplicate key within run: {key}")
                    seen.add(key)
                    if key not in corpus_keys:
                        raise ValueError(f"label outside canonical corpus: {key}")
                    for field, expected in (
                        ("annotation_run_id", run["annotation_run_id"]),
                        ("annotator_id", run["annotator_id"]),
                        ("model_id", run["model_id"]),
                        ("agent_id", run["agent_id"]),
                        ("pass_kind", run["pass_kind"]),
                        ("annotation_version", run["annotation_version"]),
                        ("batch_id", batch["batch_id"]),
                        ("shard_id", run["shard_id"]),
                        ("shard_index", run["shard_index"]),
                        ("shard_count", run["shard_count"]),
                        ("schema_version", run["schema_version"]),
                        ("schema_sha256", SCHEMA_SHA256),
                        ("prompt_version", run["prompt_version"]),
                        ("prompt_sha256", PROMPT_SHA256),
                        (
                            "canonical_corpus_fingerprint",
                            manifest["corpus"]["canonical_corpus_fingerprint"],
                        ),
                        (
                            "labeling_input_sha256",
                            manifest["corpus"]["labeling_input_sha256"],
                        ),
                    ):
                        if label.get(field) != expected:
                            raise ValueError(f"{key}: saved {field} drift")
                    _saved_label_semantics(label, str(lookup.loc[key]["text"]))
                    unclear += int(label["outcome"] == "unclear")
                    if run["pass_kind"] == "primary":
                        if key in primary_owners:
                            raise ValueError(
                                f"primary key labeled by multiple runs: {key}"
                            )
                        primary_owners[key] = run["annotation_run_id"]
        except Exception as exc:
            issues.append(f"{run_path.parent.name}: {exc}")

    missing = corpus_keys - set(primary_owners)
    if missing:
        issues.append(f"missing primary labels: {len(missing)}")
    if unclear:
        issues.append(f"unadjudicated unclear primary/agreement labels: {unclear}")
    if open_batches:
        issues.append(f"open batches: {open_batches}")
    return {
        "status": "ok" if not issues else "failed",
        "dataset_id": dataset_id,
        "canonical_paragraphs": len(corpus_keys),
        "primary_labels": len(primary_owners),
        "missing_primary_labels": len(missing),
        "unclear_labels": unclear,
        "open_batches": open_batches,
        "issues": issues,
    }


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export(
    *, dataset_id: str, root: Path = DATA_ROOT
) -> dict[str, Any]:
    manifest = _read_manifest(dataset_id, root)
    if manifest["scope"] != "full":
        raise ValueError("publication export requires a full-scope dataset")
    result = audit(dataset_id=dataset_id, root=root)
    if result["status"] != "ok":
        raise ValueError(f"export gates failed: {result['issues']}")
    dataset_dir = _dataset_dir(dataset_id, root)
    primary: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for run_path in _run_paths(dataset_dir):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["pass_kind"] != "primary":
            continue
        run_ids.add(run["annotation_run_id"])
        primary.extend(_labels_for_run(dataset_dir, run["annotation_run_id"]))
    primary.sort(key=lambda label: (label["doc_name"], label["para_idx"]))

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    provenance_fields = [
        "label_id",
        "doc_name",
        "para_idx",
        "outcome",
        "unclear_reason",
        "rationale",
        "annotation_version",
        "annotation_run_id",
        "annotator_id",
        "model_id",
        "agent_id",
        "batch_id",
        "shard_id",
        "schema_version",
        "schema_sha256",
        "prompt_version",
        "prompt_sha256",
        "canonical_corpus_fingerprint",
        "labeling_input_sha256",
        "year",
        "president",
        "title",
        "speech_type",
    ]
    for label in primary:
        base = {field: label.get(field) for field in provenance_fields}
        claims = label["claims"]
        if claims:
            for claim_index, claim in enumerate(claims):
                long_rows.append({**base, "claim_index": claim_index, **claim})
        else:
            long_rows.append(
                {
                    **base,
                    "claim_index": pd.NA,
                    "group_text": None,
                    "normalized_group": None,
                    "group_type": None,
                    "relation": None,
                    "evidence_span": None,
                    "certainty": None,
                    "notes": None,
                }
            )
        summary_rows.append(
            {
                **base,
                "n_claims": len(claims),
                "normalized_groups_json": _canonical_json(
                    sorted({claim["normalized_group"] for claim in claims})
                ),
                "group_types_json": _canonical_json(
                    sorted({claim["group_type"] for claim in claims})
                ),
                "relations_json": _canonical_json(
                    sorted({claim["relation"] for claim in claims})
                ),
            }
        )
    long_frame = pd.DataFrame(long_rows)
    summary_frame = pd.DataFrame(summary_rows)
    export_dir = dataset_dir / "exports"
    long_path = export_dir / "constituency_labels.parquet"
    summary_path = export_dir / "paragraph_summary.parquet"
    meta_path = export_dir / "provenance.json"
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "annotation_version": manifest["annotation_version"],
        "canonical_corpus_fingerprint": manifest["corpus"][
            "canonical_corpus_fingerprint"
        ],
        "labeling_input_sha256": manifest["corpus"]["labeling_input_sha256"],
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": SCHEMA_SHA256,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "primary_run_ids": sorted(run_ids),
        "n_paragraph_labels": len(primary),
        "n_long_rows": len(long_frame),
        "outcomes": (
            summary_frame["outcome"].value_counts().sort_index().to_dict()
        ),
        "annotation_standard": manifest["storage"]["annotation_standard"],
        "network_calls": False,
    }
    _atomic_write_parquet(long_frame, long_path)
    _atomic_write_parquet(summary_frame, summary_path)
    _atomic_write_json(meta_path, meta)
    return {
        "status": "exported",
        "labels": str(long_path),
        "summary": str(summary_path),
        "provenance": str(meta_path),
        "n_paragraph_labels": len(primary),
        "n_long_rows": len(long_frame),
    }


def _claim_semantics(label: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(claim["normalized_group"]).casefold(),
            str(claim["group_type"]),
            str(claim["relation"]),
        )
        for claim in label["claims"]
    }


def _claim_evidence(label: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(claim["normalized_group"]).casefold(),
            str(claim["evidence_span"]),
            str(claim["certainty"]),
        )
        for claim in label["claims"]
    }


def disagreements(
    *,
    dataset_id: str,
    left_run_id: str,
    right_run_id: str,
    root: Path = DATA_ROOT,
) -> dict[str, Any]:
    manifest = _read_manifest(dataset_id, root)
    frame = _live_frame(manifest)
    dataset_dir = _dataset_dir(dataset_id, root)
    left_run = _read_run(dataset_dir, left_run_id)
    right_run = _read_run(dataset_dir, right_run_id)
    if left_run_id == right_run_id:
        raise ValueError("disagreement runs must be distinct")
    for field in (
        "annotation_version",
        "canonical_corpus_fingerprint",
        "labeling_input_sha256",
    ):
        if left_run[field] != right_run[field]:
            raise ValueError(f"runs differ on {field}")
    left = {
        _key_tuple(label): label
        for label in _labels_for_run(dataset_dir, left_run_id)
    }
    right = {
        _key_tuple(label): label
        for label in _labels_for_run(dataset_dir, right_run_id)
    }
    lookup = frame.set_index(["doc_name", "para_idx"])
    if not lookup.index.is_unique:
        raise ValueError("canonical paragraph keys are no longer unique")
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        reasons = []
        if a["outcome"] != b["outcome"]:
            reasons.append("outcome")
        if _claim_semantics(a) != _claim_semantics(b):
            reasons.append("claim_semantics")
        if _claim_evidence(a) != _claim_evidence(b):
            reasons.append("evidence_or_certainty")
        if a["outcome"] == "unclear" or b["outcome"] == "unclear":
            reasons.append("unclear")
        if not reasons:
            continue
        source = lookup.loc[key]
        rows.append(
            {
                "doc_name": key[0],
                "para_idx": key[1],
                "year": int(source["year"]),
                "president": str(source["president"]),
                "title": str(source["title"]),
                "speech_type": str(source["speech_type"]),
                "text": str(source["text"]),
                "reasons": reasons,
                "left": a,
                "right": b,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "overlap": len(set(left) & set(right)),
        "disagreements": rows,
        "n_disagreements": len(rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline terminal workflow for evidence-bound constituency labels. "
            "No command calls a model API or opens a network connection."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="fingerprint the canonical corpus and initialize a dataset"
    )
    init_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    init_parser.add_argument("--annotation-version", default=ANNOTATION_VERSION)
    init_parser.add_argument("--scope", choices=SCOPES, default="pilot")
    init_parser.add_argument("--paragraphs", type=Path, default=DEFAULT_PARAGRAPHS)
    init_parser.add_argument("--speeches", type=Path, default=DEFAULT_SPEECHES)
    init_parser.add_argument(
        "--speech-annotations", type=Path, default=DEFAULT_SPEECH_ANNOTATIONS
    )
    init_parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    init_parser.add_argument("--corpus-meta", type=Path, default=DEFAULT_CORPUS_META)
    init_parser.add_argument(
        "--keys",
        type=Path,
        help="persisted canonical (doc_name, para_idx) queue; required for pilot scope",
    )

    next_parser = subparsers.add_parser(
        "next", help="print or resume one stable terminal batch"
    )
    next_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    next_parser.add_argument("--annotator-id", required=True)
    next_parser.add_argument(
        "--model-id",
        required=True,
        help="exact model identity reported by the runner",
    )
    next_parser.add_argument("--agent-id")
    next_parser.add_argument(
        "--pass-kind", choices=PASS_KINDS, default="primary"
    )
    next_parser.add_argument("--reference-run-id")
    next_parser.add_argument("--shard-index", type=int, default=0)
    next_parser.add_argument("--shard-count", type=int, default=1)
    next_parser.add_argument("--batch-size", type=int, default=5)
    next_parser.add_argument("--context-window", type=int, default=1)

    ingest_parser = subparsers.add_parser(
        "ingest", help="validate completed JSON from stdin and save atomically"
    )
    ingest_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ingest_parser.add_argument(
        "--input",
        default="-",
        help="'-' for stdin (default), or a local JSON response file",
    )

    for command in ("status", "audit", "export"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)

    disagreement_parser = subparsers.add_parser(
        "disagreements",
        help="emit independently labeled paragraphs whose results differ",
    )
    disagreement_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    disagreement_parser.add_argument("--left-run-id", required=True)
    disagreement_parser.add_argument("--right-run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(
                dataset_id=args.dataset_id,
                paragraphs_path=args.paragraphs,
                speeches_path=args.speeches,
                speech_annotations_path=args.speech_annotations,
                exclusions_path=args.exclusions,
                corpus_meta_path=args.corpus_meta,
                selection_keys_path=args.keys,
                annotation_version=args.annotation_version,
                scope=args.scope,
            )
        elif args.command == "next":
            result = next_batch(
                dataset_id=args.dataset_id,
                annotator_id=args.annotator_id,
                model_id=args.model_id,
                agent_id=args.agent_id,
                pass_kind=args.pass_kind,
                reference_run_id=args.reference_run_id,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                batch_size=args.batch_size,
                context_window=args.context_window,
            )
        elif args.command == "ingest":
            if args.input == "-":
                response = json.load(sys.stdin)
            else:
                response = json.loads(
                    Path(args.input).read_text(encoding="utf-8")
                )
            result = ingest(response, dataset_id=args.dataset_id)
        elif args.command == "status":
            result = status(dataset_id=args.dataset_id)
        elif args.command == "audit":
            result = audit(dataset_id=args.dataset_id)
        elif args.command == "export":
            result = export(dataset_id=args.dataset_id)
        else:
            result = disagreements(
                dataset_id=args.dataset_id,
                left_run_id=args.left_run_id,
                right_run_id=args.right_run_id,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if args.command == "audit" and result.get("status") != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
