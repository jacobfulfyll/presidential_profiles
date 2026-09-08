"""Coarse part-of-speech artifacts for descriptive language-change analysis.

This deliberately publishes broad POS shares, not a claim about syntactic
complexity. The spaCy tagger is useful for nouns, verbs, pronouns, and function
words; historical spelling, OCR, quotations, and genre shifts remain important
limitations.

The artifact is local and reproducible, but it is still a model output. Cached
files are therefore accepted only when their source corpus, runtime model,
declared schema, and content hashes match the current environment. A stale or
partial cache fails closed instead of silently appearing on the public site.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import spacy

from . import corpus, era_profiles
from .llm_annotations import corpus_fingerprint

GRAMMAR_DIR = corpus.DATA_DIR / "grammar"
SPEECH_POS_PATH = GRAMMAR_DIR / "speech_pos.parquet"
ERA_POS_PATH = GRAMMAR_DIR / "era_pos.csv"
MANIFEST_PATH = GRAMMAR_DIR / "manifest.json"

SCHEMA_VERSION = "grammar-pos-v2"
MODEL_PACKAGE = "en_core_web_sm"
DISABLED_PIPELINES = ("parser", "ner")

POS_LABELS = (
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET",
    "ADP", "CCONJ", "SCONJ", "NUM",
)


class GrammarArtifactError(ValueError):
    """A cached grammar artifact is incomplete, corrupt, or stale."""


def _era(year: int) -> str:
    for spec in era_profiles.ERA_PROFILE_SPECS:
        if spec.start_year <= year <= spec.end_year:
            return spec.label
    return "Outside declared eras"


def _era_record(year: int) -> tuple[str, str, int]:
    for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS):
        if spec.start_year <= year <= spec.end_year:
            return spec.key, spec.label, order
    return "outside", "Outside declared eras", len(era_profiles.ERA_PROFILE_SPECS)


def _era_contract() -> list[dict[str, Any]]:
    return [
        {
            "era_key": spec.key,
            "era_label": spec.label,
            "era_order": order,
            "start_year": int(spec.start_year),
            "end_year": int(spec.end_year),
        }
        for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:  # pragma: no cover - environment failure
        raise GrammarArtifactError(
            f"required POS model package {name!r} is not installed"
        ) from exc


def _runtime_identity() -> dict[str, Any]:
    return {
        "library": "spacy",
        "library_version": spacy.__version__,
        "package": MODEL_PACKAGE,
        "package_version": _package_version(MODEL_PACKAGE),
        "disabled_pipelines": list(DISABLED_PIPELINES),
        "sentence_boundaries": "senter",
    }


def _source_identity(speeches: pd.DataFrame | None = None) -> dict[str, Any]:
    speeches = corpus.load() if speeches is None else speeches
    paragraphs = pd.read_parquet(corpus.PARAGRAPHS_PATH)
    return {
        "speech_source": {
            "path": str(corpus.PARQUET_PATH.relative_to(corpus.REPO_ROOT)),
            "sha256": _sha256(corpus.PARQUET_PATH),
            "size_bytes": corpus.PARQUET_PATH.stat().st_size,
        },
        "paragraph_source": {
            "path": str(corpus.PARAGRAPHS_PATH.relative_to(corpus.REPO_ROOT)),
            "sha256": _sha256(corpus.PARAGRAPHS_PATH),
            "size_bytes": corpus.PARAGRAPHS_PATH.stat().st_size,
        },
        "corpus_fingerprint": corpus_fingerprint(
            speeches=speeches, paragraphs=paragraphs
        ),
    }


def _required_columns() -> list[str]:
    return [
        "doc_name", "president", "year", "title", "n_tagged_tokens",
        *(f"pos_{label.lower()}" for label in POS_LABELS),
        "era_key", "era", "era_order",
    ]


def _read_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GrammarArtifactError(
            f"grammar manifest is unreadable: {MANIFEST_PATH}"
        ) from exc
    if not isinstance(value, dict):
        raise GrammarArtifactError("grammar manifest must be a JSON object")
    return value


def validate_cached_artifacts(
    *, speeches: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the cached speech table and manifest only when both are current."""
    expected_paths = (SPEECH_POS_PATH, ERA_POS_PATH, MANIFEST_PATH)
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise GrammarArtifactError(
            "grammar artifact bundle is incomplete; missing " + ", ".join(missing)
        )

    manifest = _read_manifest()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GrammarArtifactError(
            f"grammar artifact schema is stale: expected {SCHEMA_VERSION!r}; "
            f"observed {manifest.get('schema_version')!r}"
        )
    if manifest.get("labels") != list(POS_LABELS):
        raise GrammarArtifactError("grammar POS label contract drift")
    if manifest.get("era_contract") != _era_contract():
        raise GrammarArtifactError("grammar Story-era contract drift")
    if manifest.get("model") != _runtime_identity():
        raise GrammarArtifactError(
            "grammar runtime model identity drift; rerun build(force=True)"
        )

    current_source = _source_identity(speeches)
    if manifest.get("source_identity") != current_source:
        raise GrammarArtifactError(
            "grammar source corpus identity drift; rerun build(force=True)"
        )
    expected_hashes = {
        "speech_pos.parquet": _sha256(SPEECH_POS_PATH),
        "era_pos.csv": _sha256(ERA_POS_PATH),
    }
    if manifest.get("artifacts") != expected_hashes:
        raise GrammarArtifactError("grammar artifact content hash drift")

    out = pd.read_parquet(SPEECH_POS_PATH)
    missing_columns = [column for column in _required_columns() if column not in out]
    if missing_columns:
        raise GrammarArtifactError(
            f"grammar speech table is missing columns {missing_columns}"
        )
    if out["doc_name"].duplicated().any():
        raise GrammarArtifactError("grammar speech table has duplicate doc_name keys")
    if len(out) != int(manifest.get("n_speeches", -1)):
        raise GrammarArtifactError("grammar speech count does not match its manifest")
    if (out["n_tagged_tokens"] <= 0).any():
        raise GrammarArtifactError("grammar speech table contains an empty tagged speech")

    era = pd.read_csv(ERA_POS_PATH)
    expected_era_columns = {
        "era_key", "era", "era_order", "part_of_speech", "share_percent",
        "n_tagged_tokens",
    }
    if set(era) != expected_era_columns:
        raise GrammarArtifactError("grammar era table schema drift")
    if era.duplicated(["era_key", "part_of_speech"]).any():
        raise GrammarArtifactError("grammar era table has duplicate semantic keys")
    if not era["share_percent"].between(0, 100).all():
        raise GrammarArtifactError("grammar era shares fall outside 0-100")
    return out, manifest


def manifest_summary() -> dict[str, Any] | None:
    """Read a validated public provenance summary, or return None when absent."""
    if not MANIFEST_PATH.exists():
        return None
    _, manifest = validate_cached_artifacts()
    return manifest


def _era_table(out: pd.DataFrame) -> pd.DataFrame:
    count_cols = [f"pos_{label.lower()}" for label in POS_LABELS]
    era = (
        out.groupby(["era_order", "era_key", "era"], sort=False)[
            ["n_tagged_tokens", *count_cols]
        ]
        .sum()
        .sort_index(level="era_order")
    )
    rows = []
    for (era_order, era_key, era_name), row in era.iterrows():
        for label, column in zip(POS_LABELS, count_cols):
            rows.append({
                "era_key": era_key,
                "era": era_name,
                "era_order": int(era_order),
                "part_of_speech": label,
                "share_percent": float(
                    row[column] / max(row.n_tagged_tokens, 1) * 100
                ),
                "n_tagged_tokens": int(row.n_tagged_tokens),
            })
    return pd.DataFrame(rows)


def build(force: bool = False) -> pd.DataFrame:
    """Build or load the POS bundle, refusing stale cached artifacts."""
    if not force and any(
        path.exists() for path in (SPEECH_POS_PATH, ERA_POS_PATH, MANIFEST_PATH)
    ):
        out, _ = validate_cached_artifacts()
        return out

    df = corpus.load()
    # Keep the attribute ruler: the tagger predicts Penn tags and the ruler
    # maps them onto the coarse Universal POS values published here.
    nlp = spacy.load(MODEL_PACKAGE, disable=list(DISABLED_PIPELINES))
    nlp.enable_pipe("senter")
    nlp.max_length = 2_000_000
    rows = []
    for i, doc in enumerate(nlp.pipe(df.transcript.tolist(), batch_size=16)):
        counts = Counter(
            token.pos_ for token in doc
            if token.is_alpha and token.pos_ in POS_LABELS
        )
        tagged = sum(counts.values())
        rows.append({
            "doc_name": df.iloc[i].doc_name,
            "n_tagged_tokens": tagged,
            **{f"pos_{label.lower()}": counts[label] for label in POS_LABELS},
        })
        if (i + 1) % 100 == 0:
            print(f"  POS-tagged {i + 1}/{len(df)} speeches")
    out = df[["doc_name", "president", "year", "title"]].merge(
        pd.DataFrame(rows), on="doc_name", validate="one_to_one"
    )
    era_records = out["year"].map(_era_record)
    out["era_key"] = era_records.map(lambda value: value[0])
    out["era"] = era_records.map(lambda value: value[1])
    out["era_order"] = era_records.map(lambda value: value[2])
    if out["doc_name"].duplicated().any() or (out["n_tagged_tokens"] <= 0).any():
        raise GrammarArtifactError("new grammar speech table failed key/support checks")

    GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SPEECH_POS_PATH, index=False)
    era = _era_table(out)
    era.to_csv(ERA_POS_PATH, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": _runtime_identity(),
        "unit": "speech",
        "semantic_key": ["doc_name"],
        "era_semantic_key": ["era_key", "part_of_speech"],
        "era_contract": _era_contract(),
        "labels": list(POS_LABELS),
        "n_speeches": int(len(out)),
        "n_tagged_tokens": int(out["n_tagged_tokens"].sum()),
        "source_identity": _source_identity(df),
        "artifacts": {
            "speech_pos.parquet": _sha256(SPEECH_POS_PATH),
            "era_pos.csv": _sha256(ERA_POS_PATH),
        },
        "generation": {
            "kind": "deterministic_local_derived",
            "timestamps": "omitted_by_design",
            "paid_api_calls": False,
        },
        "limitations": [
            "Coarse POS shares are not a full syntax or grammar model.",
            "Historical spelling, OCR, quotations, and genre can affect tags.",
            "The artifact describes the formal speech corpus, not all presidential communication.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return out


if __name__ == "__main__":
    built = build(force=True)
    print(f"wrote {SPEECH_POS_PATH.relative_to(corpus.REPO_ROOT)} ({len(built):,} speeches)")
