"""Deterministic speaker-view and reference-entity foundation.

This module is deliberately separate from site generation.  It projects the
governed speaker attribution onto the corrected corpus, runs the pinned local
spaCy NER model, conservatively aligns those spans with the promoted primary
AI entity labels, and publishes governed era-distinctive candidates.  It never
calls a provider API and never writes to frozen annotations, sealed speaker
inputs, or ``docs/``.

Fast validation (does not rerun NER)::

    arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --check

The slower local rebuild and two-build byte reproducibility check are separate::

    arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --rebuild
    arch -x86_64 .venv/bin/python -m presidential_profiles.foundation_audit --verify-reproducible
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from . import annotation_ledger, era_profiles
from .corpus import DATA_DIR


REPO_ROOT = DATA_DIR.parent
SPEAKER_INPUT_DIR = DATA_DIR / "speaker_attribution"
SPEAKER_OUTPUT_DIR = DATA_DIR / "speaker_views"
REFERENCE_OUTPUT_DIR = DATA_DIR / "reference_entities"
CORRECTION_DIR = DATA_DIR / "corpus_corrections"
ANNOTATIONS_DIR = DATA_DIR / "llm_annotations"

CANONICAL_SPEECHES_PATH = CORRECTION_DIR / "canonical_speeches_v1.parquet"
CANONICAL_PARAGRAPHS_PATH = CORRECTION_DIR / "canonical_paragraphs_v1.parquet"
CORRECTION_META_PATH = CORRECTION_DIR / "meta_v1.json"
ATTRIBUTION_PATH = SPEAKER_INPUT_DIR / "paragraph_attribution_v1.parquet"
ATTRIBUTION_META_PATH = SPEAKER_INPUT_DIR / "meta_v1.json"
SECONDARY_ENTITIES_PATH = ANNOTATIONS_DIR / "paragraph_entities__opus4-8.parquet"
SECONDARY_PARAGRAPH_ANNOTATIONS_PATH = (
    ANNOTATIONS_DIR / "paragraph_annotations__opus4-8.parquet"
)
AGREEMENT_SAMPLE_PATH = ANNOTATIONS_DIR / "agreement_sample_v1.json"

PARAGRAPH_VIEW_NAME = "paragraph_view_v1.parquet"
APPEARANCES_NAME = "appearances_v1.parquet"
COVERAGE_NAME = "coverage_v1.parquet"
CONSUMER_INVENTORY_NAME = "consumer_inventory_v1.json"
SPEAKER_META_NAME = "meta_v1.json"

NER_MENTIONS_NAME = "ner_mentions_v1.parquet"
HYBRID_ENTITIES_NAME = "entity_mentions_v1.parquet"
ERA_DISTINCTIVE_NAME = "era_distinctive_v1.parquet"
ALIAS_MAP_NAME = "alias_map_v1.json"
SOURCE_QUALITY_NAME = "source_quality_v1.json"
REFERENCE_META_NAME = "meta_v1.json"
ACCEPTANCE_REPORT_NAME = "acceptance_report_v1.json"

SCHEMA_VERSION = "speaker-reference-foundation-v1"
SPEAKER_VIEW_VERSION = "speaker-view-v1"
REFERENCE_ENTITY_VERSION = "reference-entity-v1"
ERA_DISTINCTIVE_VERSION = "era-distinctive-reference-v1"
ALIAS_VERSION = "reference-entity-alias-v1"
NER_PACKAGE = "en_core_web_sm"
NER_MODEL_VERSION = "3.8.0"
NER_PIPELINE_VERSION = "spacy-ner-v1"
EXPECTED_SPACY_VERSION = "3.8.14"
MIN_ENTITY_PARAGRAPHS = 5
MIN_ENTITY_DOCUMENTS = 2
TOP_ENTITIES_PER_ERA = 5
LOG_ODDS_PRIOR_SUCCESS = 0.5
LOG_ODDS_PRIOR_FAILURE = 0.5
MILLER_CENTER_ORIGIN = "https://millercenter.org"

KEY = ["doc_name", "para_idx"]


class FoundationError(RuntimeError):
    """The foundation or one of its governed inputs failed a refusal guard."""


SAFE_ALIASES = {
    "the united states": "united states",
    "united states of america": "united states",
    "the united states of america": "united states",
    "u.s.": "united states",
    "u.s": "united states",
    "u.s.a.": "united states",
    "u.s.a": "united states",
    "usa": "united states",
    "the congress": "congress",
    "u.s. congress": "congress",
    "united states congress": "congress",
    "the republican party": "republican party",
    "the democratic party": "democratic party",
}

# These examples are pinned as deliberately distinct so a future alias edit
# cannot quietly turn historical succession into lexical equivalence.
DELIBERATELY_UNMERGED = (
    ("great britain", "united kingdom"),
    ("prussia", "germany"),
    ("austria-hungary", "austria"),
    ("soviet union", "russia"),
    ("yugoslavia", "serbia"),
    ("ottoman empire", "turkey"),
)

NER_DISPLAY_TYPES = {
    "PERSON": "person",
    "NORP": "group",
    "ORG": "institution",
    "GPE": "place_or_political_entity",
    "LOC": "place",
    "FAC": "place",
    "EVENT": "event",
    "LAW": "law",
    "LANGUAGE": "language",
    "PRODUCT": "work_or_product",
    "WORK_OF_ART": "work_or_product",
    "DATE": "date_or_time",
    "TIME": "date_or_time",
    "MONEY": "quantity",
    "PERCENT": "quantity",
    "QUANTITY": "quantity",
    "ORDINAL": "quantity",
    "CARDINAL": "quantity",
}


# This is an explicit migration ledger, not an instruction to mutate the
# current site.  Render-only pass-throughs are folded into their analytical
# producer so a single metric is not misleadingly counted several times.
CONSUMER_INVENTORY: tuple[dict[str, str], ...] = (
    {
        "consumer": "combat.president_conflict_v2",
        "surface": "Summary / Conflict president portrait",
        "current_unit": "speaker-attributed paragraph",
        "replacement_unit": "speaker-view-v1 paragraph",
        "denominator": "speaker_audited_all paragraphs credited to the actual president",
        "migration_status": "already_speaker_audited; adapter_to_foundation_pending",
        "follow_on": "Summary",
    },
    {
        "consumer": "indices.president_scores",
        "surface": "Profiles, Compare, networks, Summary president measures",
        "current_unit": "document-owned speech and speech-marker row",
        "replacement_unit": "speaker appearance",
        "denominator": "speaker appearances and words attributed to the actual president",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Profiles/Compare/Summary",
    },
    {
        "consumer": "ai_labels.build_ai_labels",
        "surface": "Profile and Compare AI topics, flags, adversaries",
        "current_unit": "paragraph joined to document owner",
        "replacement_unit": "speaker-view-v1 paragraph and reference entity",
        "denominator": "all speaker-audited paragraphs for the actual president",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Profiles/Compare",
    },
    {
        "consumer": "profiles.distinctive_words",
        "surface": "Profile vocabulary and Compare vocabulary chips",
        "current_unit": "document-owned speech transcript",
        "replacement_unit": "speaker appearance text",
        "denominator": "all words in the president's speaker appearances",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Profiles/Compare",
    },
    {
        "consumer": "profiles.signature_speeches",
        "surface": "Profile and Compare signature-speech ranking",
        "current_unit": "document-owned speech",
        "replacement_unit": "speaker appearance within a source document",
        "denominator": "eligible speaker appearances for the actual president",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Profiles/Compare",
    },
    {
        "consumer": "similarity.build_similarity",
        "surface": "Profile and Compare nearest-neighbor lenses",
        "current_unit": "document-owned president vector",
        "replacement_unit": "speaker-appearance president vector",
        "denominator": "all eligible appearances or their attributed paragraphs by lens",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Compare/Profiles",
    },
    {
        "consumer": "networks.build_network_atlas",
        "surface": "Explore network atlas",
        "current_unit": "document-owned speech/paragraph",
        "replacement_unit": "speaker appearance and speaker-view-v1 paragraph",
        "denominator": "actual-speaker appearances and attributed paragraphs",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Explore",
    },
    {
        "consumer": "issues.president_emphasis",
        "surface": "Issue detail president emphasis",
        "current_unit": "paragraph joined to document owner",
        "replacement_unit": "speaker-view-v1 paragraph",
        "denominator": "eligible attributed paragraphs for the president",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Issues",
    },
    {
        "consumer": "era_profiles.president_composition",
        "surface": "Story era profiles",
        "current_unit": "document-owned speech and paragraph",
        "replacement_unit": "speaker appearance and speaker-view-v1 paragraph",
        "denominator": "actual-speaker appearances/paragraphs assigned to the governed Story era",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Story",
    },
    {
        "consumer": "era_visualizations.president_measures",
        "surface": "Story era president charts",
        "current_unit": "document-owned paragraph/word totals",
        "replacement_unit": "speaker-view-v1 paragraph/word totals",
        "denominator": "actual-speaker paragraphs or words within each Story era",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Story",
    },
    {
        "consumer": "site.summary_temporal_president_contract",
        "surface": "Summary / Time all-president portrait",
        "current_unit": "document-owned speech",
        "replacement_unit": "speaker appearance",
        "denominator": "all available actual-speaker appearances and their marker words",
        "migration_status": "explicit_document_owner_current; migration_deferred",
        "follow_on": "Summary",
    },
    {
        "consumer": "site.founding_president_contracts",
        "surface": "Founding Story president comparisons",
        "current_unit": "document-owned speech/paragraph",
        "replacement_unit": "speaker appearance and speaker-view-v1 paragraph",
        "denominator": "actual-speaker evidence within the Founding Story era",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Story",
    },
    {
        "consumer": "invocations.build",
        "surface": "Profiles and rhetoric invocation measures",
        "current_unit": "document-owned concatenated transcript",
        "replacement_unit": "speaker appearance text",
        "denominator": "all words in the president's speaker appearances",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Profiles",
    },
    {
        "consumer": "rhetoric.build",
        "surface": "President rhetoric summaries",
        "current_unit": "document-owned speech",
        "replacement_unit": "speaker appearance",
        "denominator": "actual-speaker appearances and words",
        "migration_status": "foundation_ready_not_migrated",
        "follow_on": "Profiles/Compare",
    },
    {
        "consumer": "expansion_site.data_quality",
        "surface": "Data Quality second-model entity audit",
        "current_unit": "sampled document paragraph",
        "replacement_unit": "unchanged data-quality-only paragraph sample",
        "denominator": "canonical paragraphs in retained sampled documents",
        "migration_status": "data_quality_only; prohibited_from_headline_ranking",
        "follow_on": "Data/Methods",
    },
    {
        "consumer": "convergence.president_window_compositions",
        "surface": "Registered convergence analysis and findings",
        "current_unit": "paragraph joined to document-owned presidential career",
        "replacement_unit": "speaker-view-v1 paragraph joined to actual-speaker career",
        "denominator": "eligible attributed paragraphs within each president/window pool",
        "migration_status": "foundation_ready_not_migrated; registered_result_requires_new_preregistration",
        "follow_on": "Data/Methods",
    },
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _metadata_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "metadata_sha256"}
    return _sha256_text(_canonical_json(payload))


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((Path(path) for path in paths), key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _inventory_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_text("".join(_canonical_json(dict(row)) + "\n" for row in rows))


def _paid_inventory() -> list[dict[str, Any]]:
    return _inventory(path for path in ANNOTATIONS_DIR.rglob("*") if path.is_file())


def _active_labels_path() -> tuple[Path, str]:
    pointer = annotation_ledger.MATERIALIZED_ROOT / "current"
    generation = pointer.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise FoundationError("annotation-ledger active generation is malformed")
    path = (
        annotation_ledger.MATERIALIZED_ROOT
        / "generations"
        / generation
        / "current_labels.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, generation


def normalize_entity(value: str) -> str:
    """Casefold and collapse whitespace without changing punctuation."""
    if not isinstance(value, str) or not value.strip():
        raise FoundationError("entity names must be non-empty strings")
    return " ".join(value.casefold().split())


def canonical_entity(value: str, aliases: Mapping[str, str] | None = None) -> str:
    normalized = normalize_entity(value)
    mapping = SAFE_ALIASES if aliases is None else aliases
    return mapping.get(normalized, normalized)


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    missing = [column for column in keys if column not in frame]
    if missing:
        raise FoundationError(f"{label} is missing key columns: {missing}")
    duplicated = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicated.empty:
        raise FoundationError(
            f"{label} has duplicate keys on {list(keys)}: "
            f"{duplicated.head(5).to_dict('records')}"
        )


def _require_same_keys(
    left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str], label: str
) -> None:
    left_keys = set(map(tuple, left[list(keys)].to_numpy()))
    right_keys = set(map(tuple, right[list(keys)].to_numpy()))
    if left_keys != right_keys:
        raise FoundationError(
            f"{label} key-set mismatch: left_only={len(left_keys - right_keys)}, "
            f"right_only={len(right_keys - left_keys)}"
        )


def _story_era(year: int, attributed_speaker: Any, document_owner: str) -> tuple[str, str, str]:
    speaker = (
        str(attributed_speaker)
        if attributed_speaker is not None and not pd.isna(attributed_speaker)
        else document_owner
    )
    basis = "attributed_speaker" if speaker != document_owner or pd.notna(attributed_speaker) else "document_owner_fallback"
    spec = era_profiles.story_era_for_speech(int(year), speaker)
    return spec.key, spec.label, basis


def build_speaker_views(
    paragraphs: pd.DataFrame,
    speeches: pd.DataFrame,
    attribution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the key-complete paragraph view, appearances, and coverage table."""
    _require_unique(paragraphs, KEY, "canonical paragraphs")
    _require_unique(speeches, ["doc_name"], "canonical speeches")
    _require_unique(attribution, KEY, "speaker attribution")
    _require_same_keys(paragraphs, attribution, KEY, "paragraph/attribution")
    if set(paragraphs["doc_name"]) != set(speeches["doc_name"]):
        raise FoundationError("canonical paragraph and speech document key sets differ")

    speech_columns = [
        "doc_name",
        "date",
        "title",
        "year",
        "president",
    ]
    missing = [column for column in speech_columns if column not in speeches]
    if missing:
        raise FoundationError(f"canonical speeches missing metadata columns: {missing}")
    view = paragraphs.merge(
        speeches[speech_columns].rename(columns={"president": "corpus_document_owner"}),
        on="doc_name",
        how="left",
        validate="many_to_one",
    ).merge(attribution, on=KEY, how="left", validate="one_to_one")
    if len(view) != len(paragraphs):
        raise FoundationError("speaker paragraph view lost rows")
    if not view["document_owner"].eq(view["corpus_document_owner"]).all():
        raise FoundationError("document owner drift between corpus and attribution")
    expected_hashes = view["text"].map(_sha256_text)
    if not expected_hashes.eq(view["source_text_sha256"]).all():
        raise FoundationError("speaker attribution text fingerprints do not match the corpus")

    era_rows = [
        _story_era(row.year, row.attributed_speaker, row.document_owner)
        for row in view.itertuples(index=False)
    ]
    view["story_era_key"] = [row[0] for row in era_rows]
    view["story_era"] = [row[1] for row in era_rows]
    view["story_era_assignment_basis"] = [row[2] for row in era_rows]
    view["source_url"] = view["doc_name"].map(lambda value: MILLER_CENTER_ORIGIN + str(value))
    view["analysis_eligible"] = view["speaker_audited_all"].astype(bool)
    view["cross_owner_paragraph"] = (
        view["analysis_eligible"]
        & view["attributed_speaker_profile_id"].ne(view["document_owner_profile_id"])
    )
    view["exclusion_reason"] = view["speaker_outcome"].where(~view["analysis_eligible"], None)
    view["schema_version"] = SPEAKER_VIEW_VERSION
    view = view.drop(columns=["corpus_document_owner"]).sort_values(KEY).reset_index(drop=True)

    appearances = build_appearances(view)
    coverage = build_coverage(view)
    return view, appearances, coverage


def build_appearances(view: pd.DataFrame) -> pd.DataFrame:
    eligible = view.loc[view["analysis_eligible"]].copy()
    if eligible["attributed_speaker_profile_id"].isna().any():
        raise FoundationError("eligible paragraphs cannot lack a speaker profile")
    rows: list[dict[str, Any]] = []
    group_keys = ["doc_name", "attributed_speaker_profile_id"]
    for (doc_name, profile_id), group in eligible.groupby(group_keys, sort=True):
        group = group.sort_values("para_idx")
        stable = [
            "attributed_speaker",
            "document_owner",
            "document_owner_profile_id",
            "title",
            "source_url",
            "date",
            "year",
            "speech_type",
            "story_era_key",
            "story_era",
            "spec_version",
            "run_id",
        ]
        values: dict[str, Any] = {}
        for column in stable:
            unique = group[column].drop_duplicates()
            if len(unique) != 1:
                raise FoundationError(
                    f"appearance metadata drift for {(doc_name, profile_id)}: {column}"
                )
            values[column] = unique.iloc[0]
        para_indices = [int(value) for value in group["para_idx"]]
        text = "\n\n".join(str(value) for value in group["text"])
        rows.append(
            {
                "doc_name": doc_name,
                "attributed_speaker_profile_id": profile_id,
                **values,
                "cross_owner_appearance": profile_id != values["document_owner_profile_id"],
                "n_paragraphs": len(group),
                "n_words": int(group["word_count"].sum()),
                "para_indices_json": _canonical_json(para_indices),
                "appearance_text": text,
                "appearance_text_sha256": _sha256_text(text),
                "schema_version": SPEAKER_VIEW_VERSION,
            }
        )
    appearances = pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)
    _require_unique(appearances, group_keys, "speaker appearances")
    if int(appearances["n_paragraphs"].sum()) != len(eligible):
        raise FoundationError("appearance paragraph counts do not reconcile")
    return appearances


def build_coverage(view: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(view)
    for outcome, group in view.groupby("speaker_outcome", sort=True, dropna=False):
        eligible = bool(group["analysis_eligible"].all())
        if group["analysis_eligible"].any() != eligible:
            raise FoundationError(f"speaker outcome {outcome!r} mixes eligibility")
        rows.append(
            {
                "speaker_outcome": str(outcome),
                "analysis_eligible": eligible,
                "exclusion_reason": None if eligible else str(outcome),
                "n_paragraphs": len(group),
                "n_documents": int(group["doc_name"].nunique()),
                "share_of_retained_paragraphs": len(group) / total,
                "schema_version": SPEAKER_VIEW_VERSION,
            }
        )
    coverage = pd.DataFrame(rows).sort_values("speaker_outcome").reset_index(drop=True)
    if int(coverage["n_paragraphs"].sum()) != total:
        raise FoundationError("coverage rows do not reconcile to the paragraph view")
    return coverage


def _load_primary_ai_entities(labels: pd.DataFrame) -> pd.DataFrame:
    rows = labels.loc[
        labels["label_type"].eq("entities") & labels["event_role"].eq("value")
    ].copy()
    required = [
        "canonical_doc_name",
        "canonical_para_idx",
        "raw_value_json",
        "run_id",
        "spec_version",
        "item_index",
        "promotion_channel",
        "label_id",
    ]
    missing = [column for column in required if column not in rows]
    if missing:
        raise FoundationError(f"canonical AI entity projection is missing: {missing}")
    parsed = rows["raw_value_json"].map(json.loads)
    if not parsed.map(lambda value: isinstance(value, dict) and {"name", "type", "stance"} <= set(value)).all():
        raise FoundationError("canonical AI entity values do not match the entity object contract")
    output = pd.DataFrame(
        {
            "doc_name": rows["canonical_doc_name"],
            "para_idx": rows["canonical_para_idx"].astype(int),
            "ai_entity": parsed.map(lambda value: value["name"]),
            "ai_type": parsed.map(lambda value: value["type"]),
            "ai_stance": parsed.map(lambda value: value["stance"]),
            "ai_run_id": rows["run_id"],
            "ai_spec_version": rows["spec_version"],
            "ai_item_index": rows["item_index"].astype(int),
            "ai_promotion_channel": rows["promotion_channel"],
            "ai_label_id": rows["label_id"],
        }
    )
    output["normalized_entity_raw"] = output["ai_entity"].map(normalize_entity)
    output["normalized_entity"] = output["ai_entity"].map(canonical_entity)
    return output.sort_values(KEY + ["ai_item_index", "ai_entity"]).reset_index(drop=True)


def run_ner(paragraph_view: pd.DataFrame, nlp: Any | None = None) -> pd.DataFrame:
    """Run pinned ``en_core_web_sm`` NER over every retained paragraph."""
    if nlp is None:
        spacy_version = importlib.metadata.version("spacy")
        model_version = importlib.metadata.version("en-core-web-sm")
        if spacy_version != EXPECTED_SPACY_VERSION or model_version != NER_MODEL_VERSION:
            raise FoundationError(
                "pinned spaCy environment drift: "
                f"expected spacy={EXPECTED_SPACY_VERSION}, en_core_web_sm={NER_MODEL_VERSION}; "
                f"observed spacy={spacy_version}, en_core_web_sm={model_version}"
            )
        import spacy

        nlp = spacy.load(NER_PACKAGE, disable=["tagger", "parser", "lemmatizer"])
    texts = paragraph_view["text"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    metadata_columns = [
        "doc_name",
        "para_idx",
        "speaker_outcome",
        "analysis_eligible",
        "attributed_speaker",
        "attributed_speaker_profile_id",
        "document_owner",
        "document_owner_profile_id",
        "cross_owner_paragraph",
        "year",
        "story_era_key",
        "story_era",
        "title",
        "source_url",
        "speech_type",
    ]
    records = paragraph_view[metadata_columns].to_dict("records")
    docs = nlp.pipe(texts, batch_size=128, n_process=1)
    for metadata, doc in zip(records, docs, strict=True):
        for entity in doc.ents:
            raw = str(entity.text)
            rows.append(
                {
                    **metadata,
                    "ner_text": raw,
                    "start_char": int(entity.start_char),
                    "end_char": int(entity.end_char),
                    "ner_label": str(entity.label_),
                    "display_type": NER_DISPLAY_TYPES.get(str(entity.label_), "other"),
                    "normalized_entity_raw": normalize_entity(raw),
                    "normalized_entity": canonical_entity(raw),
                    "ner_pipeline_version": NER_PIPELINE_VERSION,
                    "spacy_version": importlib.metadata.version("spacy"),
                    "spacy_model": NER_PACKAGE,
                    "spacy_model_version": importlib.metadata.version("en-core-web-sm"),
                    "schema_version": REFERENCE_ENTITY_VERSION,
                }
            )
    columns = metadata_columns + [
        "ner_text",
        "start_char",
        "end_char",
        "ner_label",
        "display_type",
        "normalized_entity_raw",
        "normalized_entity",
        "ner_pipeline_version",
        "spacy_version",
        "spacy_model",
        "spacy_model_version",
        "schema_version",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(KEY + ["start_char", "end_char", "ner_label", "ner_text"]).reset_index(drop=True)


def _json_values(series: pd.Series) -> str:
    values = sorted({str(value) for value in series if pd.notna(value)})
    return _canonical_json(values)


def _single_or_none(series: pd.Series) -> str | None:
    values = sorted({str(value) for value in series if pd.notna(value)})
    return values[0] if len(values) == 1 else None


def build_hybrid_entities(
    paragraph_view: pd.DataFrame,
    ai_entities: pd.DataFrame,
    ner_mentions: pd.DataFrame,
) -> pd.DataFrame:
    """Conservatively align primary AI names and local NER within paragraphs."""
    paragraph_keys = set(map(tuple, paragraph_view[KEY].to_numpy()))
    for label, frame in (("AI entity", ai_entities), ("NER mention", ner_mentions)):
        extra = set(map(tuple, frame[KEY].to_numpy())) - paragraph_keys
        if extra:
            raise FoundationError(f"{label} rows contain non-canonical paragraph keys")

    ai_rows: list[dict[str, Any]] = []
    group_key = KEY + ["normalized_entity"]
    for key, group in ai_entities.groupby(group_key, sort=True):
        group = group.sort_values(["ai_item_index", "ai_entity", "ai_label_id"])
        first = group.iloc[0]
        ai_rows.append(
            {
                "doc_name": key[0],
                "para_idx": int(key[1]),
                "normalized_entity": key[2],
                "ai_entity": first["ai_entity"],
                "ai_type": _single_or_none(group["ai_type"]),
                "ai_stance": _single_or_none(group["ai_stance"]),
                "ai_entities_json": _json_values(group["ai_entity"]),
                "ai_types_json": _json_values(group["ai_type"]),
                "ai_stances_json": _json_values(group["ai_stance"]),
                "ai_run_ids_json": _json_values(group["ai_run_id"]),
                "ai_label_ids_json": _json_values(group["ai_label_id"]),
                "ai_mention_count": len(group),
            }
        )
    ai_grouped = pd.DataFrame(ai_rows)

    ner_rows: list[dict[str, Any]] = []
    for key, group in ner_mentions.groupby(group_key, sort=True):
        group = group.sort_values(["start_char", "end_char", "ner_label", "ner_text"])
        first = group.iloc[0]
        span_rows = [
            {
                "text": row.ner_text,
                "start_char": int(row.start_char),
                "end_char": int(row.end_char),
                "label": row.ner_label,
            }
            for row in group.itertuples(index=False)
        ]
        ner_rows.append(
            {
                "doc_name": key[0],
                "para_idx": int(key[1]),
                "normalized_entity": key[2],
                "ner_text": first["ner_text"],
                "ner_label": _single_or_none(group["ner_label"]),
                "ner_display_type": _single_or_none(group["display_type"]),
                "ner_labels_json": _json_values(group["ner_label"]),
                "ner_spans_json": _canonical_json(span_rows),
                "ner_mention_count": len(group),
            }
        )
    ner_grouped = pd.DataFrame(ner_rows)

    if ai_grouped.empty:
        ai_grouped = pd.DataFrame(columns=group_key)
    if ner_grouped.empty:
        ner_grouped = pd.DataFrame(columns=group_key)
    hybrid = ai_grouped.merge(ner_grouped, on=group_key, how="outer", validate="one_to_one")
    metadata_columns = [
        "doc_name",
        "para_idx",
        "text",
        "speaker_outcome",
        "analysis_eligible",
        "attributed_speaker",
        "attributed_speaker_profile_id",
        "document_owner",
        "document_owner_profile_id",
        "cross_owner_paragraph",
        "year",
        "story_era_key",
        "story_era",
        "title",
        "source_url",
        "speech_type",
    ]
    hybrid = hybrid.merge(
        paragraph_view[metadata_columns], on=KEY, how="left", validate="many_to_one"
    )
    has_ai = hybrid["ai_entity"].notna()
    has_ner = hybrid["ner_text"].notna()
    hybrid["source_kind"] = "ner_only"
    hybrid.loc[has_ai & ~has_ner, "source_kind"] = "ai_only"
    hybrid.loc[has_ai & has_ner, "source_kind"] = "ai_and_ner"
    hybrid["source_badge"] = hybrid["source_kind"].map(
        {"ai_and_ner": "AI + NER", "ai_only": "AI only", "ner_only": "NER only"}
    )
    hybrid["source_agreement"] = has_ai & has_ner
    hybrid["display_entity"] = hybrid["ai_entity"].where(has_ai, hybrid["ner_text"])
    hybrid["display_type"] = hybrid["ai_type"].where(has_ai, hybrid["ner_display_type"])
    hybrid["schema_version"] = REFERENCE_ENTITY_VERSION
    _require_unique(hybrid, group_key, "hybrid entity layer")
    return hybrid.sort_values(group_key).reset_index(drop=True)


def _mode_string(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    counts = values.value_counts()
    if counts.empty:
        return ""
    maximum = int(counts.max())
    return sorted(counts.loc[counts.eq(maximum)].index)[0]


def rank_era_entities(
    paragraph_view: pd.DataFrame,
    hybrid_entities: pd.DataFrame,
    *,
    min_paragraphs: int = MIN_ENTITY_PARAGRAPHS,
    min_documents: int = MIN_ENTITY_DOCUMENTS,
    top_n: int = TOP_ENTITIES_PER_ERA,
) -> pd.DataFrame:
    """Rank primary-AI candidates with Jeffreys-smoothed paragraph log odds."""
    eligible_paragraphs = paragraph_view.loc[paragraph_view["analysis_eligible"]]
    denominators = eligible_paragraphs.groupby("story_era_key").size().to_dict()
    if set(denominators) != {spec.key for spec in era_profiles.ERA_PROFILE_SPECS}:
        raise FoundationError("every canonical Story era must have eligible paragraphs")
    total_paragraphs = len(eligible_paragraphs)

    candidates = hybrid_entities.loc[
        hybrid_entities["analysis_eligible"] & hybrid_entities["ai_entity"].notna()
    ].copy()
    # The hybrid key is already one normalized entity per paragraph.  Assert
    # the de-duplication contract rather than relying on that implementation.
    _require_unique(candidates, KEY + ["normalized_entity"], "era candidate paragraphs")
    global_counts = candidates.groupby("normalized_entity").size().to_dict()
    rows: list[dict[str, Any]] = []
    for spec in era_profiles.ERA_PROFILE_SPECS:
        era = candidates.loc[candidates["story_era_key"].eq(spec.key)]
        era_n = int(denominators[spec.key])
        other_n = total_paragraphs - era_n
        for normalized, group in era.groupby("normalized_entity", sort=True):
            paragraph_count = len(group)
            document_count = int(group["doc_name"].nunique())
            if paragraph_count < min_paragraphs or document_count < min_documents:
                continue
            other_count = int(global_counts[normalized] - paragraph_count)
            era_failures = era_n - paragraph_count
            other_failures = other_n - other_count
            if min(era_failures, other_failures, other_count) < 0:
                raise FoundationError("entity incidence exceeds its paragraph denominator")
            log_odds = math.log(
                (paragraph_count + LOG_ODDS_PRIOR_SUCCESS)
                / (era_failures + LOG_ODDS_PRIOR_FAILURE)
            ) - math.log(
                (other_count + LOG_ODDS_PRIOR_SUCCESS)
                / (other_failures + LOG_ODDS_PRIOR_FAILURE)
            )
            variance = (
                1 / (paragraph_count + LOG_ODDS_PRIOR_SUCCESS)
                + 1 / (era_failures + LOG_ODDS_PRIOR_FAILURE)
                + 1 / (other_count + LOG_ODDS_PRIOR_SUCCESS)
                + 1 / (other_failures + LOG_ODDS_PRIOR_FAILURE)
            )
            evidence_pool = group.sort_values(
                ["source_agreement", "doc_name", "para_idx", "ai_entity"],
                ascending=[False, True, True, True],
            )
            evidence = evidence_pool.iloc[0]
            agreement_paragraphs = int(group["source_agreement"].sum())
            rows.append(
                {
                    "story_era_key": spec.key,
                    "story_era": spec.label,
                    "era_start_year": spec.start_year,
                    "era_end_year": spec.end_year,
                    "normalized_entity": normalized,
                    "display_entity": _mode_string(group["ai_entity"]),
                    "display_type": _mode_string(group["ai_type"]),
                    "era_paragraphs": paragraph_count,
                    "era_documents": document_count,
                    "era_denominator_paragraphs": era_n,
                    "other_paragraphs": other_count,
                    "other_denominator_paragraphs": other_n,
                    "paragraph_share": paragraph_count / era_n,
                    "other_paragraph_share": other_count / other_n,
                    "log_odds": log_odds,
                    "log_odds_standard_error": math.sqrt(variance),
                    "log_odds_z": log_odds / math.sqrt(variance),
                    "ai_ner_paragraphs": agreement_paragraphs,
                    "ai_only_paragraphs": paragraph_count - agreement_paragraphs,
                    "source_badge": evidence["source_badge"],
                    "evidence_ai_mention": evidence["ai_entity"],
                    "evidence_ner_mention": evidence["ner_text"],
                    "evidence_ai_stance": evidence["ai_stance"],
                    "evidence_doc_name": evidence["doc_name"],
                    "evidence_para_idx": int(evidence["para_idx"]),
                    "evidence_document_title": evidence["title"],
                    "evidence_source_url": evidence["source_url"],
                    "evidence_speaker": evidence["attributed_speaker"],
                    "evidence_document_owner": evidence["document_owner"],
                    "evidence_cross_owner": bool(evidence["cross_owner_paragraph"]),
                    "evidence_excerpt": evidence["text"],
                    "min_eligible_paragraphs": min_paragraphs,
                    "min_eligible_documents": min_documents,
                    "log_odds_prior_success": LOG_ODDS_PRIOR_SUCCESS,
                    "log_odds_prior_failure": LOG_ODDS_PRIOR_FAILURE,
                    "schema_version": ERA_DISTINCTIVE_VERSION,
                }
            )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        raise FoundationError("no era-distinctive entity candidates passed support floors")
    ranked = ranked.loc[ranked["log_odds"].gt(0)].copy()
    ranked = ranked.sort_values(
        ["era_start_year", "log_odds", "era_paragraphs", "era_documents", "normalized_entity"],
        ascending=[True, False, False, False, True],
    )
    ranked["rank"] = ranked.groupby("story_era_key").cumcount() + 1
    ranked = ranked.loc[ranked["rank"].le(top_n)].copy()
    counts = ranked.groupby("story_era_key").size()
    if not counts.eq(top_n).all() or len(counts) != len(era_profiles.ERA_PROFILE_SPECS):
        raise FoundationError("each Story era must publish exactly five eligible candidates")
    leading = ["story_era_key", "story_era", "era_start_year", "era_end_year", "rank"]
    return ranked[leading + [column for column in ranked if column not in leading]].reset_index(drop=True)


def _quality_audit(
    ai_entities: pd.DataFrame,
    canonical_paragraphs: pd.DataFrame,
) -> dict[str, Any]:
    sample = json.loads(AGREEMENT_SAMPLE_PATH.read_text(encoding="utf-8"))
    sample_docs = set(sample["doc_names"])
    canonical_docs = set(canonical_paragraphs["doc_name"])
    secondary_paragraphs = pd.read_parquet(SECONDARY_PARAGRAPH_ANNOTATIONS_PATH)
    _require_unique(secondary_paragraphs, KEY, "secondary-model paragraph audit")
    canonical_keys = set(map(tuple, canonical_paragraphs[KEY].to_numpy()))
    secondary_keys = set(map(tuple, secondary_paragraphs[KEY].to_numpy()))
    keys = sorted(canonical_keys & secondary_keys)
    retained_docs = {key[0] for key in keys}
    if not retained_docs <= sample_docs or not retained_docs <= canonical_docs:
        raise FoundationError("secondary-model QA keys escape the governed sample/corpus")
    primary = (
        ai_entities.groupby(KEY)["normalized_entity"]
        .apply(set)
        .to_dict()
    )
    secondary_raw = pd.read_parquet(SECONDARY_ENTITIES_PATH)
    secondary_raw = secondary_raw.loc[
        pd.MultiIndex.from_frame(secondary_raw[KEY]).isin(keys)
    ].copy()
    secondary_raw["normalized_entity"] = secondary_raw["entity"].map(canonical_entity)
    secondary = (
        secondary_raw.groupby(KEY)["normalized_entity"]
        .apply(set)
        .to_dict()
    )
    jaccards = []
    exact = 0
    matched = primary_only = secondary_only = 0
    for key in keys:
        left = primary.get(key, set())
        right = secondary.get(key, set())
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
        exact += left == right
        matched += len(left & right)
        primary_only += len(left - right)
        secondary_only += len(right - left)
    return {
        "schema_version": "reference-entity-source-quality-v1",
        "role": "data_quality_only",
        "prohibited_uses": ["headline_candidates", "era_ranking", "stance_inference"],
        "secondary_artifact": str(SECONDARY_ENTITIES_PATH.relative_to(REPO_ROOT)),
        "secondary_artifact_sha256": _sha256_file(SECONDARY_ENTITIES_PATH),
        "secondary_paragraph_artifact": str(
            SECONDARY_PARAGRAPH_ANNOTATIONS_PATH.relative_to(REPO_ROOT)
        ),
        "secondary_paragraph_artifact_sha256": _sha256_file(
            SECONDARY_PARAGRAPH_ANNOTATIONS_PATH
        ),
        "agreement_sample": str(AGREEMENT_SAMPLE_PATH.relative_to(REPO_ROOT)),
        "agreement_sample_sha256": _sha256_file(AGREEMENT_SAMPLE_PATH),
        "sample_documents_declared": len(sample_docs),
        "secondary_labeled_documents_declared": int(
            secondary_paragraphs["doc_name"].nunique()
        ),
        "secondary_labeled_paragraphs_declared": len(secondary_paragraphs),
        "canonical_exact_key_documents_compared": len(retained_docs),
        "canonical_exact_key_paragraphs_compared": len(keys),
        "exact_name_set_agreement_paragraphs": exact,
        "exact_name_set_agreement_share": exact / len(keys),
        "mean_name_set_jaccard": sum(jaccards) / len(jaccards),
        "matched_normalized_names": matched,
        "primary_only_normalized_names": primary_only,
        "secondary_only_normalized_names": secondary_only,
    }


def _alias_payload() -> dict[str, Any]:
    return {
        "schema_version": ALIAS_VERSION,
        "normalization": ["Unicode-preserving casefold", "whitespace collapse"],
        "aliases": dict(sorted(SAFE_ALIASES.items())),
        "deliberately_unmerged": [list(pair) for pair in DELIBERATELY_UNMERGED],
        "policy": "safe lexical equivalents only; historically distinct entities remain distinct",
    }


def _consumer_payload() -> dict[str, Any]:
    consumers = [dict(row) for row in CONSUMER_INVENTORY]
    names = [row["consumer"] for row in consumers]
    if len(names) != len(set(names)):
        raise FoundationError("consumer inventory contains duplicate consumer ids")
    return {
        "schema_version": "president-attributed-consumer-inventory-v1",
        "scope": "analytical producers; render-only pass-throughs are folded into their producer",
        "default_rule": "actual-speaker attribution for president-attributed results",
        "document_rule": "explicit document-level questions may retain ownership only when labeled",
        "consumers": consumers,
    }


def _source_inputs() -> tuple[list[dict[str, Any]], Path, str]:
    labels_path, generation = _active_labels_path()
    paths = [
        CANONICAL_SPEECHES_PATH,
        CANONICAL_PARAGRAPHS_PATH,
        CORRECTION_META_PATH,
        ATTRIBUTION_PATH,
        ATTRIBUTION_META_PATH,
        labels_path,
        annotation_ledger.MATERIALIZED_ROOT / "current",
        SECONDARY_ENTITIES_PATH,
        SECONDARY_PARAGRAPH_ANNOTATIONS_PATH,
        AGREEMENT_SAMPLE_PATH,
    ]
    return _inventory(paths), labels_path, generation


def _artifact_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: _sha256_file(directory / name) for name in names}


def _validate_population(view: pd.DataFrame, appearances: pd.DataFrame) -> None:
    if len(view) != 35_394:
        raise FoundationError(f"expected 35,394 retained paragraphs; observed {len(view):,}")
    eligible = view.loc[view["analysis_eligible"]]
    if len(eligible) != 32_531:
        raise FoundationError(f"expected 32,531 eligible paragraphs; observed {len(eligible):,}")
    if len(view) - len(eligible) != 2_863:
        raise FoundationError("expected exactly 2,863 excluded paragraphs")
    cross_owner = eligible.loc[eligible["cross_owner_paragraph"]]
    if len(cross_owner) != 296:
        raise FoundationError(f"expected 296 cross-owner paragraphs; observed {len(cross_owner)}")
    if int(appearances["n_paragraphs"].sum()) != 32_531:
        raise FoundationError("appearance population does not reconcile to 32,531")
    excluded = view.loc[~view["analysis_eligible"]]
    if excluded["attributed_speaker_profile_id"].notna().any():
        raise FoundationError("excluded paragraphs cannot carry a president profile")


def _carter_reagan_regression(
    view: pd.DataFrame, appearances: pd.DataFrame
) -> dict[str, Any]:
    fragment = "october-28-1980-debate-ronald-reagan"
    doc_rows = view.loc[view["doc_name"].str.contains(fragment, regex=False)]
    if len(doc_rows) != 151 or doc_rows["doc_name"].nunique() != 1:
        raise FoundationError("Carter–Reagan regression document must contain 151 paragraphs")
    doc_name = str(doc_rows["doc_name"].iloc[0])
    counts = doc_rows.groupby("speaker_outcome").size().to_dict()
    carter = doc_rows.loc[doc_rows["attributed_speaker"].eq("Jimmy Carter")]
    reagan = doc_rows.loc[doc_rows["attributed_speaker"].eq("Ronald Reagan")]
    if len(carter) != 44 or len(reagan) != 42:
        raise FoundationError("Carter–Reagan attributed paragraph counts drifted")
    if counts.get("non_president") != 22 or counts.get("multiple_speakers") != 43:
        raise FoundationError("Carter–Reagan exclusion counts drifted")
    app = appearances.loc[appearances["doc_name"].eq(doc_name)]
    if set(app["attributed_speaker"]) != {"Jimmy Carter", "Ronald Reagan"}:
        raise FoundationError("Carter–Reagan appearances are incomplete")
    app_by = app.set_index("attributed_speaker")
    carter_text = str(app_by.loc["Jimmy Carter", "appearance_text"])
    reagan_text = str(app_by.loc["Ronald Reagan", "appearance_text"])
    if any(str(text) in carter_text for text in reagan["text"]):
        raise FoundationError("Carter appearance contains a Reagan paragraph")
    if any(str(text) in reagan_text for text in carter["text"]):
        raise FoundationError("Reagan appearance contains a Carter paragraph")
    shared = ["title", "source_url", "date", "year", "document_owner", "doc_name"]
    for column in shared:
        if app[column].nunique(dropna=False) != 1:
            raise FoundationError(f"Carter–Reagan source metadata drift: {column}")
    return {
        "doc_name": doc_name,
        "total_paragraphs": 151,
        "carter_paragraphs": 44,
        "reagan_paragraphs": 42,
        "non_president_paragraphs": 22,
        "multiple_speaker_paragraphs": 43,
        "shared_title": app["title"].iloc[0],
        "shared_source_url": app["source_url"].iloc[0],
        "shared_date": str(app["date"].iloc[0]),
        "document_owner": app["document_owner"].iloc[0],
    }


def _compact_excerpt(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _acceptance_report(
    view: pd.DataFrame,
    appearances: pd.DataFrame,
    coverage: pd.DataFrame,
    era_entities: pd.DataFrame,
    source_quality: Mapping[str, Any],
    speaker_meta: Mapping[str, Any],
    reference_meta: Mapping[str, Any],
) -> dict[str, Any]:
    cross_examples = (
        view.loc[view["cross_owner_paragraph"]]
        .sort_values(KEY)
        .groupby(["document_owner", "attributed_speaker"], sort=True)
        .first()
        .reset_index()
        .head(10)
    )
    examples = [
        {
            "doc_name": row.doc_name,
            "para_idx": int(row.para_idx),
            "document_owner": row.document_owner,
            "attributed_speaker": row.attributed_speaker,
            "title": row.title,
            "source_url": row.source_url,
            "excerpt": _compact_excerpt(row.text),
        }
        for row in cross_examples.itertuples(index=False)
    ]
    era_rows = []
    for spec in era_profiles.ERA_PROFILE_SPECS:
        rows = era_entities.loc[era_entities["story_era_key"].eq(spec.key)]
        era_rows.append(
            {
                "story_era_key": spec.key,
                "story_era": spec.label,
                "denominator_paragraphs": int(rows["era_denominator_paragraphs"].iloc[0]),
                "candidates": [
                    {
                        "rank": int(row.rank),
                        "entity": row.display_entity,
                        "normalized_entity": row.normalized_entity,
                        "paragraphs": int(row.era_paragraphs),
                        "documents": int(row.era_documents),
                        "log_odds": float(row.log_odds),
                        "source_badge": row.source_badge,
                        "ai_ner_paragraphs": int(row.ai_ner_paragraphs),
                        "ai_only_paragraphs": int(row.ai_only_paragraphs),
                        "evidence": {
                            "exact_ai_mention": row.evidence_ai_mention,
                            "exact_ner_mention": None if pd.isna(row.evidence_ner_mention) else row.evidence_ner_mention,
                            "document": row.evidence_document_title,
                            "doc_name": row.evidence_doc_name,
                            "para_idx": int(row.evidence_para_idx),
                            "speaker": row.evidence_speaker,
                            "source_url": row.evidence_source_url,
                            "excerpt": _compact_excerpt(row.evidence_excerpt),
                        },
                    }
                    for row in rows.sort_values("rank").itertuples(index=False)
                ],
            }
        )
    report = {
        "schema_version": "speaker-reference-foundation-acceptance-v1",
        "status": "accepted",
        "population": {
            "retained_attribution_rows": len(view),
            "eligible_presidential_paragraphs": int(view["analysis_eligible"].sum()),
            "excluded_paragraphs": int((~view["analysis_eligible"]).sum()),
            "cross_owner_presidential_paragraphs": int(view["cross_owner_paragraph"].sum()),
            "appearances": len(appearances),
            "coverage": [
                {
                    key: (None if pd.isna(value) else value)
                    for key, value in row.items()
                }
                for row in coverage.to_dict("records")
            ],
        },
        "carter_reagan_regression": _carter_reagan_regression(view, appearances),
        "cross_owner_examples": examples,
        "era_candidates": era_rows,
        "source_quality": dict(source_quality),
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "speaker_metadata_sha256": speaker_meta["metadata_sha256"],
            "reference_metadata_sha256": reference_meta["metadata_sha256"],
            "canonical_corpus_fingerprint": speaker_meta["canonical_corpus_fingerprint"],
            "attribution_run_id": speaker_meta["attribution_run_id"],
        },
    }
    report["report_sha256"] = _sha256_text(_canonical_json(report))
    return report


def build_foundation(
    *,
    speaker_output_dir: Path = SPEAKER_OUTPUT_DIR,
    reference_output_dir: Path = REFERENCE_OUTPUT_DIR,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """Build every Plan 1 artifact from immutable/governed local inputs."""
    paid_before = _paid_inventory()
    source_inventory, labels_path, generation = _source_inputs()
    correction_meta = json.loads(CORRECTION_META_PATH.read_text(encoding="utf-8"))
    attribution_meta = json.loads(ATTRIBUTION_META_PATH.read_text(encoding="utf-8"))
    if correction_meta["canonical_corpus_fingerprint"] != attribution_meta["canonical_corpus_fingerprint"]:
        raise FoundationError("correction and attribution corpus fingerprints differ")

    paragraphs = pd.read_parquet(CANONICAL_PARAGRAPHS_PATH)
    speeches = pd.read_parquet(CANONICAL_SPEECHES_PATH)
    attribution = pd.read_parquet(ATTRIBUTION_PATH)
    paragraph_view, appearances, coverage = build_speaker_views(
        paragraphs, speeches, attribution
    )
    _validate_population(paragraph_view, appearances)
    _carter_reagan_regression(paragraph_view, appearances)

    speaker_output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(speaker_output_dir / PARAGRAPH_VIEW_NAME, paragraph_view)
    _atomic_write_parquet(speaker_output_dir / APPEARANCES_NAME, appearances)
    _atomic_write_parquet(speaker_output_dir / COVERAGE_NAME, coverage)
    consumer_payload = _consumer_payload()
    _atomic_write_json(speaker_output_dir / CONSUMER_INVENTORY_NAME, consumer_payload)

    labels = pd.read_parquet(labels_path)
    ai_entities = _load_primary_ai_entities(labels)
    ner_mentions = run_ner(paragraph_view, nlp=nlp)
    hybrid = build_hybrid_entities(paragraph_view, ai_entities, ner_mentions)
    era_entities = rank_era_entities(paragraph_view, hybrid)
    source_quality = _quality_audit(ai_entities, paragraphs)
    alias_payload = _alias_payload()

    reference_output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(reference_output_dir / NER_MENTIONS_NAME, ner_mentions)
    _atomic_write_parquet(reference_output_dir / HYBRID_ENTITIES_NAME, hybrid)
    _atomic_write_parquet(reference_output_dir / ERA_DISTINCTIVE_NAME, era_entities)
    _atomic_write_json(reference_output_dir / ALIAS_MAP_NAME, alias_payload)
    _atomic_write_json(reference_output_dir / SOURCE_QUALITY_NAME, source_quality)

    paid_after = _paid_inventory()
    if paid_before != paid_after:
        raise FoundationError("frozen paid annotation bytes changed during the build")

    speaker_artifacts = _artifact_hashes(
        speaker_output_dir,
        [PARAGRAPH_VIEW_NAME, APPEARANCES_NAME, COVERAGE_NAME, CONSUMER_INVENTORY_NAME],
    )
    speaker_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "view_schema_version": SPEAKER_VIEW_VERSION,
        "canonical_corpus_fingerprint": correction_meta["canonical_corpus_fingerprint"],
        "attribution_run_id": attribution_meta["run_id"],
        "attribution_spec_version": attribution_meta["spec_version"],
        "annotation_generation": generation,
        "input_inventory": source_inventory,
        "input_inventory_sha256": _inventory_hash(source_inventory),
        "frozen_paid_inventory": paid_after,
        "frozen_paid_inventory_sha256": _inventory_hash(paid_after),
        "paragraphs": len(paragraph_view),
        "eligible_presidential_paragraphs": int(paragraph_view["analysis_eligible"].sum()),
        "excluded_paragraphs": int((~paragraph_view["analysis_eligible"]).sum()),
        "cross_owner_presidential_paragraphs": int(paragraph_view["cross_owner_paragraph"].sum()),
        "appearances": len(appearances),
        "artifacts": speaker_artifacts,
        "timestamps": "omitted_by_design",
    }
    speaker_meta["metadata_sha256"] = _metadata_hash(speaker_meta)
    _atomic_write_json(speaker_output_dir / SPEAKER_META_NAME, speaker_meta)

    reference_artifacts = _artifact_hashes(
        reference_output_dir,
        [NER_MENTIONS_NAME, HYBRID_ENTITIES_NAME, ERA_DISTINCTIVE_NAME, ALIAS_MAP_NAME, SOURCE_QUALITY_NAME],
    )
    reference_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entity_schema_version": REFERENCE_ENTITY_VERSION,
        "era_schema_version": ERA_DISTINCTIVE_VERSION,
        "canonical_corpus_fingerprint": correction_meta["canonical_corpus_fingerprint"],
        "attribution_run_id": attribution_meta["run_id"],
        "annotation_generation": generation,
        "primary_ai_source": str(labels_path.relative_to(REPO_ROOT)),
        "primary_ai_source_sha256": _sha256_file(labels_path),
        "secondary_ai_role": "data_quality_only",
        "spacy_version": importlib.metadata.version("spacy"),
        "spacy_model": NER_PACKAGE,
        "spacy_model_version": importlib.metadata.version("en-core-web-sm"),
        "ner_pipeline_version": NER_PIPELINE_VERSION,
        "alias_version": ALIAS_VERSION,
        "denominator": "every speaker-audited paragraph in the governed Story era, including paragraphs without entities",
        "paragraph_entity_deduplication": "one normalized entity at most once per paragraph",
        "ranking": "Jeffreys-smoothed log odds against the other eight canonical Story eras",
        "thresholds": {
            "min_eligible_paragraphs": MIN_ENTITY_PARAGRAPHS,
            "min_source_documents": MIN_ENTITY_DOCUMENTS,
            "top_per_era": TOP_ENTITIES_PER_ERA,
            "prior_success": LOG_ODDS_PRIOR_SUCCESS,
            "prior_failure": LOG_ODDS_PRIOR_FAILURE,
        },
        "ner_mentions": len(ner_mentions),
        "hybrid_entities": len(hybrid),
        "era_candidates": len(era_entities),
        "artifacts": reference_artifacts,
        "timestamps": "omitted_by_design",
    }
    reference_meta["metadata_sha256"] = _metadata_hash(reference_meta)
    _atomic_write_json(reference_output_dir / REFERENCE_META_NAME, reference_meta)

    report = _acceptance_report(
        paragraph_view,
        appearances,
        coverage,
        era_entities,
        source_quality,
        speaker_meta,
        reference_meta,
    )
    _atomic_write_json(reference_output_dir / ACCEPTANCE_REPORT_NAME, report)
    return report


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FoundationError(f"expected a JSON object: {path}")
    return value


def _check_inventory(meta: Mapping[str, Any]) -> None:
    paths = [REPO_ROOT / row["path"] for row in meta["input_inventory"]]
    current = _inventory(paths)
    if current != meta["input_inventory"] or _inventory_hash(current) != meta["input_inventory_sha256"]:
        raise FoundationError("foundation source input inventory is stale")
    paid = _paid_inventory()
    if paid != meta["frozen_paid_inventory"] or _inventory_hash(paid) != meta["frozen_paid_inventory_sha256"]:
        raise FoundationError("frozen paid annotation inventory drifted")


def _check_artifacts(directory: Path, meta: Mapping[str, Any]) -> None:
    for name, expected in meta["artifacts"].items():
        observed = _sha256_file(directory / name)
        if observed != expected:
            raise FoundationError(f"artifact hash mismatch: {directory / name}")


def check_foundation(
    *,
    speaker_output_dir: Path = SPEAKER_OUTPUT_DIR,
    reference_output_dir: Path = REFERENCE_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the fast acceptance check without invoking spaCy."""
    speaker_meta = _load_json(speaker_output_dir / SPEAKER_META_NAME)
    reference_meta = _load_json(reference_output_dir / REFERENCE_META_NAME)
    report = _load_json(reference_output_dir / ACCEPTANCE_REPORT_NAME)
    if speaker_meta.get("metadata_sha256") != _metadata_hash(speaker_meta):
        raise FoundationError("speaker metadata identity drift")
    if reference_meta.get("metadata_sha256") != _metadata_hash(reference_meta):
        raise FoundationError("reference metadata identity drift")
    if report.get("report_sha256") != _sha256_text(
        _canonical_json({key: value for key, value in report.items() if key != "report_sha256"})
    ):
        raise FoundationError("acceptance report identity drift")
    _check_inventory(speaker_meta)
    _check_artifacts(speaker_output_dir, speaker_meta)
    _check_artifacts(reference_output_dir, reference_meta)
    if _load_json(reference_output_dir / ALIAS_MAP_NAME) != _alias_payload():
        raise FoundationError("stored alias policy is stale relative to the reader")
    if _load_json(speaker_output_dir / CONSUMER_INVENTORY_NAME) != _consumer_payload():
        raise FoundationError("stored consumer inventory is stale relative to the reader")
    observed_versions = {
        "spacy_version": importlib.metadata.version("spacy"),
        "spacy_model": NER_PACKAGE,
        "spacy_model_version": importlib.metadata.version("en-core-web-sm"),
        "ner_pipeline_version": NER_PIPELINE_VERSION,
    }
    if any(reference_meta.get(key) != value for key, value in observed_versions.items()):
        raise FoundationError("stored NER provenance is stale relative to the pinned reader")
    expected_thresholds = {
        "min_eligible_paragraphs": MIN_ENTITY_PARAGRAPHS,
        "min_source_documents": MIN_ENTITY_DOCUMENTS,
        "top_per_era": TOP_ENTITIES_PER_ERA,
        "prior_success": LOG_ODDS_PRIOR_SUCCESS,
        "prior_failure": LOG_ODDS_PRIOR_FAILURE,
    }
    if reference_meta.get("thresholds") != expected_thresholds:
        raise FoundationError("stored era thresholds are stale relative to the reader")

    correction_meta = _load_json(CORRECTION_META_PATH)
    attribution_meta = _load_json(ATTRIBUTION_META_PATH)
    fingerprints = {
        correction_meta["canonical_corpus_fingerprint"],
        attribution_meta["canonical_corpus_fingerprint"],
        speaker_meta["canonical_corpus_fingerprint"],
        reference_meta["canonical_corpus_fingerprint"],
        report["provenance"]["canonical_corpus_fingerprint"],
    }
    if len(fingerprints) != 1:
        raise FoundationError("corpus fingerprints do not match across inputs and artifacts")
    if not (
        speaker_meta["attribution_run_id"]
        == reference_meta["attribution_run_id"]
        == attribution_meta["run_id"]
    ):
        raise FoundationError("speaker attribution run identity drift")

    view = pd.read_parquet(speaker_output_dir / PARAGRAPH_VIEW_NAME)
    appearances = pd.read_parquet(speaker_output_dir / APPEARANCES_NAME)
    coverage = pd.read_parquet(speaker_output_dir / COVERAGE_NAME)
    era_entities = pd.read_parquet(reference_output_dir / ERA_DISTINCTIVE_NAME)
    _require_unique(view, KEY, "stored paragraph view")
    _require_unique(appearances, ["doc_name", "attributed_speaker_profile_id"], "stored appearances")
    _validate_population(view, appearances)
    regression = _carter_reagan_regression(view, appearances)
    if int(coverage["n_paragraphs"].sum()) != 35_394:
        raise FoundationError("coverage receipt does not reconcile")
    if len(era_entities) != 45 or not era_entities.groupby("story_era_key").size().eq(5).all():
        raise FoundationError("era summaries are not complete five-candidate sets")
    if set(view["schema_version"]) != {SPEAKER_VIEW_VERSION}:
        raise FoundationError("stored speaker-view schema is stale")
    if set(era_entities["schema_version"]) != {ERA_DISTINCTIVE_VERSION}:
        raise FoundationError("stored era-summary schema is stale")
    evidence_keys = set(map(tuple, view[KEY].to_numpy()))
    if any(
        (row.evidence_doc_name, int(row.evidence_para_idx)) not in evidence_keys
        for row in era_entities.itertuples(index=False)
    ):
        raise FoundationError("era summary evidence does not resolve to the speaker view")
    if not era_entities["source_badge"].isin(["AI + NER", "AI only"]).all():
        raise FoundationError("headline candidates contain an invalid source badge")
    if not era_entities["era_paragraphs"].ge(MIN_ENTITY_PARAGRAPHS).all():
        raise FoundationError("era summary support floor drift")
    if not era_entities["era_documents"].ge(MIN_ENTITY_DOCUMENTS).all():
        raise FoundationError("era summary document floor drift")
    if report["population"]["eligible_presidential_paragraphs"] != 32_531:
        raise FoundationError("acceptance report population drift")
    return {
        "status": "accepted",
        "retained_attribution_rows": len(view),
        "eligible_presidential_paragraphs": int(view["analysis_eligible"].sum()),
        "excluded_paragraphs": int((~view["analysis_eligible"]).sum()),
        "cross_owner_presidential_paragraphs": int(view["cross_owner_paragraph"].sum()),
        "appearances": len(appearances),
        "era_candidates": len(era_entities),
        "corpus_fingerprint": fingerprints.pop(),
        "frozen_paid_annotations": "unchanged",
        "carter_reagan_regression": regression,
    }


def verify_reproducible() -> dict[str, Any]:
    """Run two isolated slow rebuilds and compare every emitted byte."""
    with tempfile.TemporaryDirectory(prefix="presidential-foundation-a-") as first_root, tempfile.TemporaryDirectory(
        prefix="presidential-foundation-b-"
    ) as second_root:
        first = Path(first_root)
        second = Path(second_root)
        build_foundation(
            speaker_output_dir=first / "speaker_views",
            reference_output_dir=first / "reference_entities",
        )
        build_foundation(
            speaker_output_dir=second / "speaker_views",
            reference_output_dir=second / "reference_entities",
        )
        first_files = {
            str(path.relative_to(first)): _sha256_file(path)
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            str(path.relative_to(second)): _sha256_file(path)
            for path in second.rglob("*")
            if path.is_file()
        }
        if first_files != second_files:
            changed = sorted(set(first_files) | set(second_files))
            changed = [name for name in changed if first_files.get(name) != second_files.get(name)]
            raise FoundationError(f"isolated rebuilds differ: {changed}")
        return {"status": "byte_identical", "files": len(first_files)}


def _print_summary(result: Mapping[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _compact_build_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "population": report["population"],
        "carter_reagan_regression": report["carter_reagan_regression"],
        "era_candidates": {
            row["story_era_key"]: [candidate["entity"] for candidate in row["candidates"]]
            for row in report["era_candidates"]
        },
        "acceptance_report": str(
            (REFERENCE_OUTPUT_DIR / ACCEPTANCE_REPORT_NAME).relative_to(REPO_ROOT)
        ),
        "report_sha256": report["report_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the deterministic speaker/reference foundation."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="fast artifact and acceptance validation")
    action.add_argument("--rebuild", action="store_true", help="run the slower pinned local NER rebuild")
    action.add_argument(
        "--verify-reproducible",
        action="store_true",
        help="run two slower isolated rebuilds and compare emitted bytes",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        help="optional root containing speaker_views/ and reference_entities/",
    )
    args = parser.parse_args(argv)
    speaker_dir = args.out_root / "speaker_views" if args.out_root else SPEAKER_OUTPUT_DIR
    reference_dir = args.out_root / "reference_entities" if args.out_root else REFERENCE_OUTPUT_DIR
    if args.check:
        result = check_foundation(
            speaker_output_dir=speaker_dir, reference_output_dir=reference_dir
        )
    elif args.rebuild:
        result = build_foundation(
            speaker_output_dir=speaker_dir, reference_output_dir=reference_dir
        )
    else:
        if args.out_root:
            raise FoundationError("--out-root is not supported with --verify-reproducible")
        result = verify_reproducible()
    if args.rebuild:
        result = _compact_build_summary(result)
        if args.out_root:
            result["acceptance_report"] = str(
                args.out_root / "reference_entities" / ACCEPTANCE_REPORT_NAME
            )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
