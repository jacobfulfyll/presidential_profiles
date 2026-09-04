"""Validated Story reader for the governed speaker/reference foundation.

This module is the only Story-facing reader for the four Plan 1 parquet
artifacts.  It validates their provenance and keyed relationships before a
site build can write generated output, then supplies compact public
projections and downloads without changing the governed source bytes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd

from . import era_profiles, foundation_audit
from .corpus import DATA_DIR


REPO_ROOT = DATA_DIR.parent
SPEAKER_DIR = DATA_DIR / "speaker_views"
REFERENCE_DIR = DATA_DIR / "reference_entities"
INVOCATION_EVIDENCE_PATH = DATA_DIR / "networks" / "invocation_evidence.parquet"

PARAGRAPH_PATH = SPEAKER_DIR / "paragraph_view_v1.parquet"
APPEARANCES_PATH = SPEAKER_DIR / "appearances_v1.parquet"
ENTITY_MENTIONS_PATH = REFERENCE_DIR / "entity_mentions_v1.parquet"
ERA_DISTINCTIVE_PATH = REFERENCE_DIR / "era_distinctive_v1.parquet"
SPEAKER_META_PATH = SPEAKER_DIR / "meta_v1.json"
REFERENCE_META_PATH = REFERENCE_DIR / "meta_v1.json"

PUBLIC_DIR_NAME = "story"
PUBLIC_ERA_NAME = "era_distinctive_v1.csv"
PUBLIC_APPEARANCES_NAME = "speaker_appearances_v1.csv"
PUBLIC_ENTITIES_NAME = "entity_mentions_v1.parquet"
PUBLIC_MANIFEST_NAME = "manifest_v1.json"
PUBLIC_SCHEMA = "story-foundation-public-v1"

PARAGRAPH_KEY = ["doc_name", "para_idx"]
APPEARANCE_KEY = ["doc_name", "attributed_speaker_profile_id"]
ENTITY_KEY = ["doc_name", "para_idx", "normalized_entity"]
CANDIDATE_RANK_KEY = ["story_era_key", "rank"]
CANDIDATE_ENTITY_KEY = ["story_era_key", "normalized_entity"]

EXPECTED_ERA_DENOMINATORS = (672, 3_847, 2_610, 6_721, 2_134, 1_409, 5_583, 6_144, 3_411)
EXPECTED_POPULATION = {
    "retained_attribution_rows": 35_394,
    "eligible_presidential_paragraphs": 32_531,
    "excluded_paragraphs": 2_863,
    "cross_owner_presidential_paragraphs": 296,
    "appearances": 1_054,
}
EXPECTED_INVOCATION_OVERLAY = {
    "source_rows": 1_447,
    "unresolved_keys": 48,
    "ineligible_rows": 156,
    "reassigned_speakers": 27,
    "retained_rows": 1_243,
}


class StoryFoundationError(RuntimeError):
    """The governed Story foundation or a public projection failed closed."""


@dataclass(frozen=True)
class StoryFoundationBundle:
    paragraph_view: pd.DataFrame
    appearances: pd.DataFrame
    entity_mentions: pd.DataFrame
    era_distinctive: pd.DataFrame
    provenance: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _metadata_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "metadata_sha256"}
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoryFoundationError(f"could not read governed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StoryFoundationError(f"expected a JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise StoryFoundationError(f"{label} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    _require_columns(frame, keys, label)
    duplicates = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicates.empty:
        raise StoryFoundationError(
            f"{label} has duplicate keys on {list(keys)}: "
            f"{duplicates.head(5).to_dict('records')}"
        )


def _require_schema(frame: pd.DataFrame, expected: str, label: str) -> None:
    _require_columns(frame, ["schema_version"], label)
    observed = set(frame["schema_version"].dropna().astype(str))
    if observed != {expected} or frame["schema_version"].isna().any():
        raise StoryFoundationError(
            f"{label} schema drift: expected {expected!r}, observed {sorted(observed)}"
        )


def _validate_metadata(
    speaker_meta: Mapping[str, Any],
    reference_meta: Mapping[str, Any],
    audit: Mapping[str, Any],
    speaker_dir: Path = SPEAKER_DIR,
    reference_dir: Path = REFERENCE_DIR,
) -> None:
    if speaker_meta.get("metadata_sha256") != _metadata_hash(speaker_meta):
        raise StoryFoundationError("speaker metadata identity drift")
    if reference_meta.get("metadata_sha256") != _metadata_hash(reference_meta):
        raise StoryFoundationError("reference metadata identity drift")
    expected_versions = {
        "speaker": foundation_audit.SCHEMA_VERSION,
        "view": foundation_audit.SPEAKER_VIEW_VERSION,
        "entity": foundation_audit.REFERENCE_ENTITY_VERSION,
        "era": foundation_audit.ERA_DISTINCTIVE_VERSION,
    }
    observed_versions = {
        "speaker": speaker_meta.get("schema_version"),
        "view": speaker_meta.get("view_schema_version"),
        "entity": reference_meta.get("entity_schema_version"),
        "era": reference_meta.get("era_schema_version"),
    }
    if observed_versions != expected_versions:
        raise StoryFoundationError(
            f"foundation metadata schema drift: {observed_versions}"
        )
    identities = {
        "corpus_fingerprint": {
            speaker_meta.get("canonical_corpus_fingerprint"),
            reference_meta.get("canonical_corpus_fingerprint"),
            audit.get("corpus_fingerprint"),
        },
        "attribution_run": {
            speaker_meta.get("attribution_run_id"),
            reference_meta.get("attribution_run_id"),
        },
        "annotation_generation": {
            speaker_meta.get("annotation_generation"),
            reference_meta.get("annotation_generation"),
        },
    }
    for label, values in identities.items():
        if None in values or len(values) != 1:
            raise StoryFoundationError(f"foundation {label} identity drift")
    expected_thresholds = {
        "min_eligible_paragraphs": foundation_audit.MIN_ENTITY_PARAGRAPHS,
        "min_source_documents": foundation_audit.MIN_ENTITY_DOCUMENTS,
        "top_per_era": foundation_audit.TOP_ENTITIES_PER_ERA,
        "prior_success": foundation_audit.LOG_ODDS_PRIOR_SUCCESS,
        "prior_failure": foundation_audit.LOG_ODDS_PRIOR_FAILURE,
    }
    if reference_meta.get("thresholds") != expected_thresholds:
        raise StoryFoundationError("foundation threshold metadata drift")
    expected_ner = {
        "spacy_version": foundation_audit.EXPECTED_SPACY_VERSION,
        "spacy_model": foundation_audit.NER_PACKAGE,
        "spacy_model_version": foundation_audit.NER_MODEL_VERSION,
        "ner_pipeline_version": foundation_audit.NER_PIPELINE_VERSION,
    }
    if any(reference_meta.get(key) != value for key, value in expected_ner.items()):
        raise StoryFoundationError("foundation pinned NER provenance drift")
    for meta, directory in (
        (speaker_meta, speaker_dir),
        (reference_meta, reference_dir),
    ):
        for name, expected_hash in meta.get("artifacts", {}).items():
            path = directory / name
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise StoryFoundationError(f"foundation artifact hash drift: {path}")
    paid_rows = speaker_meta.get("frozen_paid_inventory")
    if not isinstance(paid_rows, list) or not paid_rows:
        raise StoryFoundationError("foundation frozen paid inventory is absent")
    for row in paid_rows:
        path = REPO_ROOT / str(row.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise StoryFoundationError(f"frozen paid inventory drift: {path}")
    input_rows = speaker_meta.get("input_inventory")
    if not isinstance(input_rows, list) or not input_rows:
        raise StoryFoundationError("foundation input inventory is absent")
    for row in input_rows:
        path = REPO_ROOT / str(row.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise StoryFoundationError(f"foundation input inventory drift: {path}")


def _normalize_optional(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the governed, fully deterministic headline ordering."""
    return frame.sort_values(
        ["log_odds", "era_paragraphs", "era_documents", "normalized_entity"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )


def _validate_candidate_evidence(
    candidates: pd.DataFrame,
    entities: pd.DataFrame,
) -> None:
    entity_index = entities.set_index(ENTITY_KEY)
    for row in candidates.itertuples(index=False):
        key = (row.evidence_doc_name, int(row.evidence_para_idx), row.normalized_entity)
        if key not in entity_index.index:
            raise StoryFoundationError(f"candidate evidence does not resolve: {key}")
        evidence = entity_index.loc[key]
        comparisons = {
            "evidence_excerpt": (row.evidence_excerpt, evidence["text"]),
            "evidence_speaker": (row.evidence_speaker, evidence["attributed_speaker"]),
            "evidence_document_owner": (
                row.evidence_document_owner,
                evidence["document_owner"],
            ),
            "story_era_key": (row.story_era_key, evidence["story_era_key"]),
            "evidence_source_url": (row.evidence_source_url, evidence["source_url"]),
            "source_badge": (row.source_badge, evidence["source_badge"]),
            "evidence_ai_mention": (row.evidence_ai_mention, evidence["ai_entity"]),
            "evidence_ner_mention": (
                _normalize_optional(row.evidence_ner_mention),
                _normalize_optional(evidence["ner_text"]),
            ),
            "evidence_ai_stance": (
                _normalize_optional(row.evidence_ai_stance),
                _normalize_optional(evidence["ai_stance"]),
            ),
            "evidence_document_title": (
                row.evidence_document_title,
                evidence["title"],
            ),
            "evidence_cross_owner": (
                bool(row.evidence_cross_owner),
                bool(evidence["cross_owner_paragraph"]),
            ),
        }
        drift = [label for label, (left, right) in comparisons.items() if left != right]
        if drift:
            raise StoryFoundationError(
                f"candidate evidence disagrees with entity row {key}: {drift}"
            )
        if pd.isna(evidence["ai_entity"]) or evidence["source_badge"] == "NER only":
            raise StoryFoundationError(f"headline candidate is not primary-AI-backed: {key}")


def _validate_frames(
    paragraphs: pd.DataFrame,
    appearances: pd.DataFrame,
    entities: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    _require_schema(paragraphs, foundation_audit.SPEAKER_VIEW_VERSION, "paragraph view")
    _require_schema(appearances, foundation_audit.SPEAKER_VIEW_VERSION, "appearances")
    _require_schema(entities, foundation_audit.REFERENCE_ENTITY_VERSION, "entity mentions")
    _require_schema(candidates, foundation_audit.ERA_DISTINCTIVE_VERSION, "era candidates")
    _require_unique(paragraphs, PARAGRAPH_KEY, "paragraph view")
    _require_unique(appearances, APPEARANCE_KEY, "appearances")
    _require_unique(entities, ENTITY_KEY, "entity mentions")
    _require_unique(candidates, CANDIDATE_RANK_KEY, "era candidate ranks")
    _require_unique(candidates, CANDIDATE_ENTITY_KEY, "era candidate entities")

    population = {
        "retained_attribution_rows": len(paragraphs),
        "eligible_presidential_paragraphs": int(paragraphs["analysis_eligible"].sum()),
        "excluded_paragraphs": int((~paragraphs["analysis_eligible"]).sum()),
        "cross_owner_presidential_paragraphs": int(paragraphs["cross_owner_paragraph"].sum()),
        "appearances": len(appearances),
    }
    if population != EXPECTED_POPULATION:
        raise StoryFoundationError(f"foundation population drift: {population}")
    if int(appearances["n_paragraphs"].sum()) != population["eligible_presidential_paragraphs"]:
        raise StoryFoundationError("appearance paragraph totals do not reconcile")

    specs = list(era_profiles.ERA_PROFILE_SPECS)
    expected_keys = [spec.key for spec in specs]
    observed_keys = list(candidates["story_era_key"].drop_duplicates())
    if observed_keys != expected_keys:
        raise StoryFoundationError(
            f"candidate era order drift: expected {expected_keys}, observed {observed_keys}"
        )
    denominators = []
    for spec in specs:
        rows = candidates[candidates["story_era_key"].eq(spec.key)].copy()
        if len(rows) != 5 or rows["rank"].tolist() != [1, 2, 3, 4, 5]:
            raise StoryFoundationError(f"{spec.key} does not publish exact ranks 1..5")
        expected_meta = {
            "story_era": spec.label,
            "era_start_year": spec.start_year,
            "era_end_year": spec.end_year,
        }
        for column, expected in expected_meta.items():
            if set(rows[column]) != {expected}:
                raise StoryFoundationError(f"candidate {column} drift for {spec.key}")
        ordered = _sort_candidates(rows)
        if ordered.index.tolist() != rows.index.tolist():
            raise StoryFoundationError(f"candidate ordering drift for {spec.key}")
        eligible_n = int(
            paragraphs["analysis_eligible"]
            .where(paragraphs["story_era_key"].eq(spec.key), False)
            .sum()
        )
        era_denominators = set(rows["era_denominator_paragraphs"].astype(int))
        other_denominators = set(rows["other_denominator_paragraphs"].astype(int))
        if era_denominators != {eligible_n} or other_denominators != {32_531 - eligible_n}:
            raise StoryFoundationError(f"candidate denominator drift for {spec.key}")
        denominators.append(eligible_n)
    if tuple(denominators) != EXPECTED_ERA_DENOMINATORS or sum(denominators) != 32_531:
        raise StoryFoundationError(f"canonical Story denominators drift: {denominators}")
    if not candidates["era_paragraphs"].ge(foundation_audit.MIN_ENTITY_PARAGRAPHS).all():
        raise StoryFoundationError("candidate paragraph support floor drift")
    if not candidates["era_documents"].ge(foundation_audit.MIN_ENTITY_DOCUMENTS).all():
        raise StoryFoundationError("candidate document support floor drift")
    if not candidates["log_odds"].gt(0).all():
        raise StoryFoundationError("candidate list contains a non-distinctive row")
    if not candidates["source_badge"].isin(["AI + NER", "AI only"]).all():
        raise StoryFoundationError("headline candidate contains prohibited NER-only evidence")
    if int(candidates["ai_ner_paragraphs"].sum()) <= 0:
        raise StoryFoundationError("candidate source-agreement counts are absent")
    _validate_candidate_evidence(candidates, entities)


def load_story_foundation(
    *,
    speaker_dir: Path = SPEAKER_DIR,
    reference_dir: Path = REFERENCE_DIR,
) -> StoryFoundationBundle:
    """Load and fully validate the four governed Story parquet artifacts."""
    audit = foundation_audit.check_foundation(
        speaker_output_dir=speaker_dir,
        reference_output_dir=reference_dir,
    )
    speaker_meta = _read_json(speaker_dir / "meta_v1.json")
    reference_meta = _read_json(reference_dir / "meta_v1.json")
    _validate_metadata(
        speaker_meta,
        reference_meta,
        audit,
        speaker_dir,
        reference_dir,
    )

    # These are intentionally the only four governed parquet reads.
    paragraphs = pd.read_parquet(speaker_dir / PARAGRAPH_PATH.name)
    appearances = pd.read_parquet(speaker_dir / APPEARANCES_PATH.name)
    entities = pd.read_parquet(reference_dir / ENTITY_MENTIONS_PATH.name)
    candidates = pd.read_parquet(reference_dir / ERA_DISTINCTIVE_PATH.name)
    _validate_frames(paragraphs, appearances, entities, candidates)
    def source_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    provenance = {
        "schema_version": foundation_audit.SCHEMA_VERSION,
        "speaker_view_schema": foundation_audit.SPEAKER_VIEW_VERSION,
        "reference_entity_schema": foundation_audit.REFERENCE_ENTITY_VERSION,
        "era_distinctive_schema": foundation_audit.ERA_DISTINCTIVE_VERSION,
        "speaker_metadata_sha256": speaker_meta["metadata_sha256"],
        "reference_metadata_sha256": reference_meta["metadata_sha256"],
        "canonical_corpus_fingerprint": speaker_meta["canonical_corpus_fingerprint"],
        "attribution_run_id": speaker_meta["attribution_run_id"],
        "annotation_generation": speaker_meta["annotation_generation"],
        "thresholds": dict(reference_meta["thresholds"]),
        "source_paths": {
            "paragraph_view": source_path(speaker_dir / PARAGRAPH_PATH.name),
            "appearances": source_path(speaker_dir / APPEARANCES_PATH.name),
            "entity_mentions": source_path(reference_dir / ENTITY_MENTIONS_PATH.name),
            "era_distinctive": source_path(reference_dir / ERA_DISTINCTIVE_PATH.name),
        },
        "source_hashes": {
            "paragraph_view": speaker_meta["artifacts"][PARAGRAPH_PATH.name],
            "appearances": speaker_meta["artifacts"][APPEARANCES_PATH.name],
            "entity_mentions": reference_meta["artifacts"][ENTITY_MENTIONS_PATH.name],
            "era_distinctive": reference_meta["artifacts"][ERA_DISTINCTIVE_PATH.name],
        },
    }
    return StoryFoundationBundle(paragraphs, appearances, entities, candidates, provenance)


def project_distinctive_references(bundle: StoryFoundationBundle) -> dict[str, dict]:
    """Project the 45 governed candidates into the era-profile reader contract."""
    output: dict[str, dict] = {}
    for spec in era_profiles.ERA_PROFILE_SPECS:
        rows = bundle.era_distinctive[
            bundle.era_distinctive["story_era_key"].eq(spec.key)
        ].sort_values("rank")
        denominator = int(rows["era_denominator_paragraphs"].iloc[0])
        output[spec.key] = {
            "definition": (
                "Positive Jeffreys-smoothed paragraph log odds against the other "
                "eight Story eras; distinctive does not mean important, prevalent, "
                "representative, or historically significant."
            ),
            "ranking": (
                "Descending log odds, then era paragraphs, source documents, and "
                "normalized entity name."
            ),
            "denominator": {
                "unit": "speaker_audited_paragraphs",
                "paragraphs": denominator,
            },
            "thresholds": {
                "minimum_paragraphs": foundation_audit.MIN_ENTITY_PARAGRAPHS,
                "minimum_source_documents": foundation_audit.MIN_ENTITY_DOCUMENTS,
                "top_per_era": foundation_audit.TOP_ENTITIES_PER_ERA,
                "prior_success": foundation_audit.LOG_ODDS_PRIOR_SUCCESS,
                "prior_failure": foundation_audit.LOG_ODDS_PRIOR_FAILURE,
            },
            "rows": [
                {
                    "rank": int(row.rank),
                    "normalized_entity": str(row.normalized_entity),
                    "label": str(row.display_entity),
                    "entity_type": str(row.display_type),
                    "support": {
                        "paragraphs": int(row.era_paragraphs),
                        "source_documents": int(row.era_documents),
                        "paragraph_share": float(row.paragraph_share),
                    },
                    "comparison": {
                        "paragraphs": int(row.other_paragraphs),
                        "denominator_paragraphs": int(row.other_denominator_paragraphs),
                        "paragraph_share": float(row.other_paragraph_share),
                    },
                    "source_agreement": {
                        "badge": str(row.source_badge),
                        "ai_ner_paragraphs": int(row.ai_ner_paragraphs),
                        "ai_only_paragraphs": int(row.ai_only_paragraphs),
                    },
                    "evidence": {
                        "doc_name": str(row.evidence_doc_name),
                        "para_idx": int(row.evidence_para_idx),
                        "title": str(row.evidence_document_title),
                        "source_url": str(row.evidence_source_url),
                        "actual_speaker": str(row.evidence_speaker),
                        "document_owner": str(row.evidence_document_owner),
                        "cross_owner": bool(row.evidence_cross_owner),
                        "excerpt": str(row.evidence_excerpt),
                        "ai_mention": str(row.evidence_ai_mention),
                        "ner_mention": _normalize_optional(row.evidence_ner_mention),
                        "ai_stance": _normalize_optional(row.evidence_ai_stance),
                    },
                }
                for row in rows.itertuples(index=False)
            ],
        }
    return output


def overlay_invocation_evidence(
    bundle: StoryFoundationBundle,
    invocation_evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Overlay invocation rows on the key-complete speaker view with receipts."""
    required = {
        "candidate_id", "speaker", "target", "doc_name", "para_idx", "era",
        "target_status", "evidence_span",
    }
    missing = sorted(required - set(invocation_evidence.columns))
    if missing:
        raise StoryFoundationError(f"invocation evidence is missing columns: {missing}")
    source = invocation_evidence[
        invocation_evidence["target_status"].eq("former_president")
        & invocation_evidence["speaker"].ne(invocation_evidence["target"])
    ].copy()
    source["annotation_speaker"] = source["speaker"]
    source["annotation_era"] = source["era"]
    metadata = bundle.paragraph_view[
        PARAGRAPH_KEY
        + [
            "text", "analysis_eligible", "attributed_speaker",
            "attributed_speaker_profile_id", "document_owner",
            "document_owner_profile_id", "cross_owner_paragraph",
            "story_era_key", "story_era", "source_url", "title",
        ]
    ]
    joined = source.merge(
        metadata,
        on=PARAGRAPH_KEY,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unresolved = joined["_merge"].ne("both")
    resolved = joined.loc[~unresolved].copy()
    ineligible = ~resolved["analysis_eligible"].astype(bool)
    retained = resolved.loc[~ineligible].copy()
    retained["speaker"] = retained["attributed_speaker"]
    retained["era"] = retained["story_era"]
    retained["source_key"] = retained["story_era_key"]
    retained["cross_owner"] = retained["cross_owner_paragraph"].astype(bool)
    retained = retained.drop(columns="_merge")
    normalized_span = retained["evidence_span"].map(lambda value: " ".join(str(value).split()))
    normalized_text = retained["text"].map(lambda value: " ".join(str(value).split()))
    if not all(span in text for span, text in zip(normalized_span, normalized_text, strict=True)):
        raise StoryFoundationError("retained invocation evidence does not resolve inside its paragraph")
    receipt = {
        "source_rows": len(source),
        "unresolved_keys": int(unresolved.sum()),
        "ineligible_rows": int(ineligible.sum()),
        "reassigned_speakers": int(
            retained["annotation_speaker"].ne(retained["speaker"]).sum()
        ),
        "retained_rows": len(retained),
    }
    return retained.reset_index(drop=True), receipt


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as source_handle, temp.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
    os.replace(temp, destination)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _appearance_download_frame(bundle: StoryFoundationBundle) -> pd.DataFrame:
    return bundle.appearances.drop(columns=["appearance_text"])


def write_story_downloads(bundle: StoryFoundationBundle, site_dir: Path) -> list[Path]:
    """Atomically publish the compact Story foundation download set."""
    output_dir = site_dir / "data" / PUBLIC_DIR_NAME
    era_path = output_dir / PUBLIC_ERA_NAME
    appearances_path = output_dir / PUBLIC_APPEARANCES_NAME
    entities_path = output_dir / PUBLIC_ENTITIES_NAME
    manifest_path = output_dir / PUBLIC_MANIFEST_NAME
    _atomic_write_bytes(era_path, _csv_bytes(bundle.era_distinctive))
    _atomic_write_bytes(appearances_path, _csv_bytes(_appearance_download_frame(bundle)))
    entity_source = Path(bundle.provenance["source_paths"]["entity_mentions"])
    if not entity_source.is_absolute():
        entity_source = REPO_ROOT / entity_source
    _atomic_copy(entity_source, entities_path)
    if _sha256_file(entities_path) != bundle.provenance["source_hashes"]["entity_mentions"]:
        raise StoryFoundationError("public entity parquet is not byte-identical to its source")
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in (era_path, appearances_path, entities_path)
    }
    manifest = {
        "schema_version": PUBLIC_SCHEMA,
        "counts": {
            "retained_paragraphs": len(bundle.paragraph_view),
            "eligible_presidential_paragraphs": int(bundle.paragraph_view["analysis_eligible"].sum()),
            "speaker_appearances": len(bundle.appearances),
            "entity_mentions": len(bundle.entity_mentions),
            "era_candidates": len(bundle.era_distinctive),
        },
        "schema_versions": {
            "speaker_view": foundation_audit.SPEAKER_VIEW_VERSION,
            "reference_entity": foundation_audit.REFERENCE_ENTITY_VERSION,
            "era_distinctive": foundation_audit.ERA_DISTINCTIVE_VERSION,
        },
        "provenance": {
            key: bundle.provenance[key]
            for key in (
                "speaker_metadata_sha256", "reference_metadata_sha256",
                "canonical_corpus_fingerprint", "attribution_run_id",
                "annotation_generation",
            )
        },
        "files": files,
    }
    _atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    return [era_path, appearances_path, entities_path, manifest_path]


def _check_public_downloads(bundle: StoryFoundationBundle, site_dir: Path) -> dict[str, Any]:
    output_dir = site_dir / "data" / PUBLIC_DIR_NAME
    paths = {
        "era": output_dir / PUBLIC_ERA_NAME,
        "appearances": output_dir / PUBLIC_APPEARANCES_NAME,
        "entities": output_dir / PUBLIC_ENTITIES_NAME,
        "manifest": output_dir / PUBLIC_MANIFEST_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise StoryFoundationError(f"public Story downloads are missing: {missing}")
    if paths["era"].read_bytes() != _csv_bytes(bundle.era_distinctive):
        raise StoryFoundationError("public era-distinctive CSV drift")
    if paths["appearances"].read_bytes() != _csv_bytes(_appearance_download_frame(bundle)):
        raise StoryFoundationError("public speaker-appearance CSV drift")
    if _sha256_file(paths["entities"]) != bundle.provenance["source_hashes"]["entity_mentions"]:
        raise StoryFoundationError("public entity parquet hash drift")
    manifest = _read_json(paths["manifest"])
    if manifest.get("schema_version") != PUBLIC_SCHEMA:
        raise StoryFoundationError("public Story manifest schema drift")
    for name, row in manifest.get("files", {}).items():
        path = output_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise StoryFoundationError(f"public Story manifest file drift: {name}")
    return {
        "era_rows": len(bundle.era_distinctive),
        "appearance_rows": len(bundle.appearances),
        "entity_sha256": _sha256_file(paths["entities"]),
    }


def _check_public_json(bundle: StoryFoundationBundle, site_dir: Path) -> dict[str, Any]:
    paths = {
        "profiles": site_dir / "data" / "era_profiles.json",
        "visualizations": site_dir / "data" / "era_visualizations.json",
        "contextualizations": site_dir / "data" / "era_contextualizations.json",
    }
    payloads = {key: _read_json(path) for key, path in paths.items()}
    expected_schemas = {
        "profiles": "era-profile-v6",
        "visualizations": "era-visualizations-v9",
        "contextualizations": "era-contextualizations-v11",
    }
    observed = {key: payload.get("schema_version") for key, payload in payloads.items()}
    if observed != expected_schemas:
        raise StoryFoundationError(f"public Story JSON schema drift: {observed}")
    projected = project_distinctive_references(bundle)
    for spec in era_profiles.ERA_PROFILE_SPECS:
        profile = payloads["profiles"]["profiles"][spec.key]
        if profile.get("distinctive_references") != projected[spec.key]:
            raise StoryFoundationError(f"public distinctive-reference projection drift: {spec.key}")
        if any(key in profile for key in ("constituents", "constituency_status", "constituency_note")):
            raise StoryFoundationError(f"legacy constituency fields remain public: {spec.key}")
    overlay = payloads["contextualizations"]["contextualizations"]["founding"]["support"].get("invocation_overlay")
    if overlay != EXPECTED_INVOCATION_OVERLAY:
        raise StoryFoundationError(f"public invocation overlay receipt drift: {overlay}")
    return {"schemas": observed, "profiles": len(projected)}


def check_story_contract(
    bundle: StoryFoundationBundle,
    site_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate the in-memory Story contract and optional generated outputs."""
    _validate_frames(
        bundle.paragraph_view,
        bundle.appearances,
        bundle.entity_mentions,
        bundle.era_distinctive,
    )
    badge_counts = bundle.era_distinctive["source_badge"].value_counts().to_dict()
    if badge_counts != {"AI + NER": 29, "AI only": 16}:
        raise StoryFoundationError(f"headline source-agreement receipt drift: {badge_counts}")
    invocation = pd.read_parquet(INVOCATION_EVIDENCE_PATH)
    _, overlay = overlay_invocation_evidence(bundle, invocation)
    if overlay != EXPECTED_INVOCATION_OVERLAY:
        raise StoryFoundationError(f"invocation overlay receipt drift: {overlay}")
    result: dict[str, Any] = {
        "status": "accepted",
        "population": dict(EXPECTED_POPULATION),
        "schemas": {
            "paragraph_view": foundation_audit.SPEAKER_VIEW_VERSION,
            "appearances": foundation_audit.SPEAKER_VIEW_VERSION,
            "entity_mentions": foundation_audit.REFERENCE_ENTITY_VERSION,
            "era_distinctive": foundation_audit.ERA_DISTINCTIVE_VERSION,
        },
        "era_denominators": list(EXPECTED_ERA_DENOMINATORS),
        "era_candidates": len(bundle.era_distinctive),
        "source_agreement": {
            "AI + NER": badge_counts["AI + NER"],
            "AI only": badge_counts["AI only"],
            "NER only": 0,
        },
        "evidence_rows_resolved": len(bundle.era_distinctive),
        "invocation_overlay": overlay,
        "public_json_parity": "not_requested",
        "download_parity": "not_requested",
    }
    if site_dir is not None:
        result["public_json_parity"] = _check_public_json(bundle, site_dir)
        result["download_parity"] = _check_public_downloads(bundle, site_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the governed Story foundation.")
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args(argv)
    bundle = load_story_foundation()
    result = check_story_contract(bundle, args.site_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
