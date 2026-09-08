"""Deterministic, execution-blocked plans for future Data Trust validation.

This module selects the preregistered human-validity and decade-hidden
measurement-invariance samples.  It performs local reads and deterministic
serialization only: there is no provider SDK, network client, model choice,
human label, or write into ``data/llm_annotations`` anywhere in this module.

The generated bundle is deliberately useful before either study runs.  It
freezes semantic keys, quotas, blinding views, prompt bytes, request ordering,
source identities, estimands, gates, and the cost formula while leaving every
value that could authorize spending unset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from . import attention, era_profiles, foundation_audit
from .corpus import DATA_DIR, REPO_ROOT
from .prompts import annotation_v1 as prompt_v1


PROTOCOL_VERSION = "data-trust-validation-protocols-v1"
MANIFEST_SCHEMA = "data-trust-validation-manifest-v1"
PARAGRAPH_SAMPLE_SCHEMA = "human-validity-paragraph-sample-v1"
SPEECH_SAMPLE_SCHEMA = "human-validity-speech-sample-v1"
BLINDED_PARAGRAPH_SCHEMA = "human-validity-blinded-paragraph-task-v1"
BLINDED_SPEECH_SCHEMA = "human-validity-blinded-speech-task-v1"
REQUEST_PLAN_SCHEMA = "measurement-invariance-request-plan-v1"
SELECTION_CELL_SCHEMA = "validation-selection-cells-v1"

SEED_NAMESPACE = "presidential-profiles/data-trust/v1/20260908"
PARAGRAPH_KEY = ["doc_name", "para_idx"]
PARAGRAPHS_PER_ERA = 40
HIGH_DISAGREEMENT_PER_ERA = 10
RARE_POSITIVE_PER_ERA = 10
BASE_PER_ERA = 20
SPEECHES_PER_ERA = 10
RARE_LABEL_MIN_SUPPORT = 20
EXPECTED_FACTUAL_SOURCE_SPEECHES = 1_057

OUTPUT_DIR = DATA_DIR / "validation_protocols" / "v1"
MANIFEST_NAME = "manifest_v1.json"

PARAGRAPH_SAMPLE_NAME = "restricted/selection/paragraph_sample_v1.csv"
PARAGRAPH_CELLS_NAME = "paragraph_selection_cells_v1.csv"
PARAGRAPH_TASKS_NAME = "restricted/assignments/paragraph_coder_tasks_v1.jsonl"
SPEECH_SAMPLE_NAME = "restricted/selection/speech_sample_v1.csv"
SPEECH_CELLS_NAME = "speech_selection_cells_v1.csv"
SPEECH_TASKS_NAME = "restricted/assignments/speech_coder_tasks_v1.jsonl"
REQUEST_PLAN_NAME = "restricted/invariance/invariance_request_plan_v1.csv"
CURRENT_PROMPT_NAME = "current_context_prompt_v1.json"
HIDDEN_PROMPT_NAME = "decade_hidden_prompt_v1.json"

PUBLIC_FILES = (
    PARAGRAPH_CELLS_NAME,
    SPEECH_CELLS_NAME,
    CURRENT_PROMPT_NAME,
    HIDDEN_PROMPT_NAME,
    MANIFEST_NAME,
)
RESTRICTED_FILES = (
    PARAGRAPH_SAMPLE_NAME,
    PARAGRAPH_TASKS_NAME,
    SPEECH_SAMPLE_NAME,
    SPEECH_TASKS_NAME,
    REQUEST_PLAN_NAME,
)
GOVERNED_FILES = (*PUBLIC_FILES, *RESTRICTED_FILES)

FLAG_FIELDS = ("party_attack", "enemy_naming", "zero_sum")
ARM_ORDER = {
    "high-disagreement": 0,
    "rare-positive": 1,
    "genre-balanced-base": 2,
}
FAMILY_ORDER = (
    "annual/governance",
    "ceremonial",
    "interactive/political",
    "public/other",
)
SPEECH_TYPE_TO_FAMILY = {
    "state_of_the_union_or_annual_message": "annual/governance",
    "special_message_to_congress": "annual/governance",
    "veto_or_signing_statement": "annual/governance",
    "inaugural_address": "ceremonial",
    "eulogy_or_commemoration": "ceremonial",
    "proclamation": "ceremonial",
    "campaign_or_debate": "interactive/political",
    "press_conference_or_interview": "interactive/political",
    "public_remarks_or_address": "public/other",
    "other": "public/other",
}
PARAGRAPH_FAMILY_TARGETS = {family: 5 for family in FAMILY_ORDER}
SPEECH_FAMILY_TARGETS = {
    "annual/governance": 3,
    "ceremonial": 2,
    "interactive/political": 2,
    "public/other": 3,
}

_DATA_SOURCE_PATHS = (
    "data/speaker_views/paragraph_view_v1.parquet",
    "data/speaker_views/meta_v1.json",
    "data/llm_annotations/paragraph_annotations.parquet",
    "data/llm_annotations/paragraph_annotations__opus4-8.parquet",
    "data/llm_annotations/paragraph_entities.parquet",
    "data/llm_annotations/paragraph_entities__opus4-8.parquet",
    "data/llm_annotations/speech_annotations.parquet",
    "data/llm_annotations/agreement_sample_v1.json",
    "data/llm_annotations/taxonomy_v1.json",
    "data/speeches.parquet",
    "data/paragraphs.parquet",
    "src/presidential_profiles/era_profiles.py",
    "src/presidential_profiles/prompts/annotation_v1.py",
    "src/presidential_profiles/validation_protocols.py",
)

_BLINDED_PARAGRAPH_FIELDS = {
    "schema_version", "item_id", "paragraph_text", "decade"
}
_BLINDED_SPEECH_FIELDS = {
    "schema_version", "item_id", "title", "year", "opening_paragraphs"
}
_PROHIBITED_BLINDED_FIELDS = {
    "doc_name", "para_idx", "president", "party", "source_url",
    "selection_arm", "story_era_key", "topics", "party_attack",
    "enemy_naming", "zero_sum", "proposal_values", "entities", "run_id",
}


class ValidationProtocolError(RuntimeError):
    """A source, population, blinding, prompt, or publication guard failed."""


@dataclass(frozen=True)
class ProtocolInputs:
    paragraph_view: pd.DataFrame
    source_paragraphs: pd.DataFrame
    source_speeches: pd.DataFrame
    primary: pd.DataFrame
    secondary: pd.DataFrame
    primary_entities: pd.DataFrame
    secondary_entities: pd.DataFrame
    speech_annotations: pd.DataFrame
    taxonomy: Mapping[str, Any]
    source_inventory: tuple[Mapping[str, Any], ...]
    frozen_annotation_runs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ValidationProtocolBundle:
    paragraph_sample: pd.DataFrame
    paragraph_cells: pd.DataFrame
    paragraph_tasks: tuple[Mapping[str, Any], ...]
    speech_sample: pd.DataFrame
    speech_cells: pd.DataFrame
    speech_tasks: tuple[Mapping[str, Any], ...]
    request_plan: pd.DataFrame
    current_prompt: Mapping[str, Any]
    hidden_prompt: Mapping[str, Any]
    manifest: Mapping[str, Any]
    files: Mapping[str, bytes]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
    else:
        text = _canonical_json(value)
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("".join(_canonical_json(dict(row)) + "\n" for row in rows)).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        # Seventeen significant digits round-trip every IEEE-754 binary64
        # probability and its reciprocal without weakening the validator.
        float_format="%.17g",
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _metadata_hash(manifest: Mapping[str, Any]) -> str:
    semantic = {key: value for key, value in manifest.items() if key != "metadata_sha256"}
    return _sha256_bytes(_canonical_json(semantic).encode("utf-8"))


def _rank_hash(arm: str, era_key: str, doc_name: str, para_idx: int) -> str:
    material = f"{SEED_NAMESPACE}|{arm}|{era_key}|{doc_name}|{int(para_idx)}"
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _opaque_id(kind: str, doc_name: str, para_idx: int) -> str:
    material = f"{SEED_NAMESPACE}|{kind}|{doc_name}|{int(para_idx)}"
    prefix = "hvp" if kind == "paragraph-item" else "hvs"
    return prefix + "_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValidationProtocolError(f"{label} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    _require_columns(frame, keys, label)
    duplicates = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicates.empty:
        raise ValidationProtocolError(
            f"{label} has duplicate semantic keys on {list(keys)}: "
            f"{duplicates.head(5).to_dict('records')}"
        )


def _key_set(frame: pd.DataFrame, keys: Sequence[str] = PARAGRAPH_KEY) -> set[tuple[Any, ...]]:
    return set(map(tuple, frame.loc[:, list(keys)].to_numpy()))


def _require_subset_keys(
    subset: pd.DataFrame,
    population: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> None:
    extra = _key_set(subset, keys) - _key_set(population, keys)
    if extra:
        raise ValidationProtocolError(
            f"{label} contains {len(extra):,} keys outside its governed population"
        )


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _source_status(relative: str) -> str:
    if relative.startswith("data/llm_annotations/"):
        return "frozen_paid_read_only"
    if relative.startswith("data/speaker_views/"):
        return "governed_derived_read_only"
    if relative.startswith("data/"):
        return "source_corpus_read_only"
    return "governed_source_code"


def _source_receipt(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = _safe_relative(path, repo_root)
    return {
        "path": relative,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "status": _source_status(relative),
    }


def _read_manifest_runs(
    repo_root: Path, run_ids: Sequence[str]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Path, ...]]:
    rows: list[Mapping[str, Any]] = []
    paths: list[Path] = []
    for run_id in sorted(set(run_ids)):
        path = repo_root / "data" / "llm_annotations" / "manifests" / f"{run_id}.json"
        if not path.is_file():
            raise ValidationProtocolError(f"frozen run {run_id!r} has no manifest: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("run_id") != run_id:
            raise ValidationProtocolError(f"annotation manifest/run_id mismatch: {path}")
        rows.append({
            "run_id": run_id,
            "model": raw.get("model"),
            "prompt_version": raw.get("prompt_version"),
            "prompt_hash": raw.get("prompt_hash"),
            "manifest_path": _safe_relative(path, repo_root),
            "manifest_sha256": _sha256_file(path),
        })
        paths.append(path)
    return tuple(rows), tuple(paths)


def load_inputs(
    *, data_dir: Path = DATA_DIR, repo_root: Path = REPO_ROOT
) -> ProtocolInputs:
    """Load and fail-closed validate the governed population and frozen labels."""
    data_dir = Path(data_dir)
    repo_root = Path(repo_root)
    view_path = data_dir / "speaker_views" / "paragraph_view_v1.parquet"
    primary_path = data_dir / "llm_annotations" / "paragraph_annotations.parquet"
    secondary_path = (
        data_dir / "llm_annotations" / "paragraph_annotations__opus4-8.parquet"
    )
    primary_entities_path = data_dir / "llm_annotations" / "paragraph_entities.parquet"
    secondary_entities_path = (
        data_dir / "llm_annotations" / "paragraph_entities__opus4-8.parquet"
    )
    speech_annotations_path = data_dir / "llm_annotations" / "speech_annotations.parquet"
    taxonomy_path = data_dir / "llm_annotations" / "taxonomy_v1.json"
    sample_path = data_dir / "llm_annotations" / "agreement_sample_v1.json"
    source_speeches_path = data_dir / "speeches.parquet"
    source_paragraphs_path = data_dir / "paragraphs.parquet"

    view = pd.read_parquet(view_path)
    primary = pd.read_parquet(primary_path)
    secondary = pd.read_parquet(secondary_path)
    primary_entities = pd.read_parquet(primary_entities_path)
    secondary_entities = pd.read_parquet(secondary_entities_path)
    speech_annotations = pd.read_parquet(speech_annotations_path)
    source_speeches = pd.read_parquet(source_speeches_path)
    source_paragraphs = pd.read_parquet(source_paragraphs_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    agreement_sample = json.loads(sample_path.read_text(encoding="utf-8"))

    _require_columns(
        view,
        [
            *PARAGRAPH_KEY, "text", "source_text_sha256", "year", "speech_type",
            "story_era_key", "story_era", "analysis_eligible",
        ],
        "speaker paragraph view",
    )
    _require_unique(view, PARAGRAPH_KEY, "speaker paragraph view")
    for label, frame in (("primary annotations", primary), ("secondary annotations", secondary)):
        _require_columns(
            frame,
            [*PARAGRAPH_KEY, "topics", *FLAG_FIELDS, "proposal_values", "run_id"],
            label,
        )
        _require_unique(frame, PARAGRAPH_KEY, label)
    for label, frame in (
        ("primary entities", primary_entities),
        ("secondary entities", secondary_entities),
    ):
        _require_columns(frame, [*PARAGRAPH_KEY, "entity", "stance", "run_id"], label)
    _require_columns(speech_annotations, ["doc_name", "speech_type", "run_id"], "speech annotations")
    _require_unique(speech_annotations, ["doc_name"], "speech annotations")
    _require_columns(
        source_speeches,
        ["doc_name", "president", "title", "year"],
        "source speeches",
    )
    _require_unique(source_speeches, ["doc_name"], "source speeches")
    _require_columns(source_paragraphs, [*PARAGRAPH_KEY, "text"], "source paragraphs")
    _require_unique(source_paragraphs, PARAGRAPH_KEY, "source paragraphs")

    source_speech_keys = set(source_speeches["doc_name"].astype(str))
    factual_annotation_keys = set(speech_annotations["doc_name"].astype(str))
    if (
        len(source_speeches) != EXPECTED_FACTUAL_SOURCE_SPEECHES
        or len(speech_annotations) != EXPECTED_FACTUAL_SOURCE_SPEECHES
        or source_speech_keys != factual_annotation_keys
    ):
        raise ValidationProtocolError(
            "factual validation requires one speech annotation for each of the "
            f"{EXPECTED_FACTUAL_SOURCE_SPEECHES:,} source speeches"
        )
    unknown_factual_types = sorted(
        set(speech_annotations["speech_type"].astype(str))
        - set(SPEECH_TYPE_TO_FAMILY)
    )
    if unknown_factual_types:
        raise ValidationProtocolError(
            "factual speech types have no preregistered family: "
            f"{unknown_factual_types}"
        )

    eligible = view.loc[view["analysis_eligible"].astype(bool)]
    _require_subset_keys(eligible, primary, PARAGRAPH_KEY, "eligible paragraph population")
    _require_subset_keys(secondary, primary, PARAGRAPH_KEY, "secondary annotations")
    _require_subset_keys(primary_entities, primary, PARAGRAPH_KEY, "primary entities")
    _require_subset_keys(secondary_entities, secondary, PARAGRAPH_KEY, "secondary entities")

    expected_eras = {spec.key for spec in era_profiles.ERA_PROFILE_SPECS}
    observed_eras = set(eligible["story_era_key"].astype(str))
    if observed_eras != expected_eras:
        raise ValidationProtocolError(
            f"eligible population Story eras drifted: expected={sorted(expected_eras)}, "
            f"observed={sorted(observed_eras)}"
        )
    unknown_types = sorted(set(eligible["speech_type"].astype(str)) - set(SPEECH_TYPE_TO_FAMILY))
    if unknown_types:
        raise ValidationProtocolError(f"speech types have no preregistered family: {unknown_types}")

    sampled_docs = set(agreement_sample.get("doc_names", []))
    if not sampled_docs:
        raise ValidationProtocolError("agreement_sample_v1.json contains no doc_names")
    outside_sample = set(secondary["doc_name"].astype(str)) - sampled_docs
    if outside_sample:
        raise ValidationProtocolError(
            f"secondary annotations contain {len(outside_sample)} documents outside agreement sample"
        )

    factual_docs = eligible[["doc_name", "speech_type"]].drop_duplicates()
    factual_join = factual_docs.merge(
        speech_annotations[["doc_name", "speech_type"]],
        on="doc_name",
        how="left",
        suffixes=("_view", "_annotation"),
        validate="one_to_one",
    )
    if factual_join["speech_type_annotation"].isna().any():
        raise ValidationProtocolError("an eligible source document lacks a factual annotation")
    mismatch = factual_join["speech_type_view"].ne(factual_join["speech_type_annotation"])
    if mismatch.any():
        raise ValidationProtocolError("speaker-view and factual-annotation speech types diverge")

    # Use the project's governed topic normalization.  Calling this now makes a
    # taxonomy collision or malformed label a build-stopping input failure.
    label_map = attention.canonical_label_map(taxonomy)
    for values in primary["topics"]:
        attention.normalize_topics(values, label_map)
    for values in secondary["topics"]:
        attention.normalize_topics(values, label_map)

    run_ids = [
        *primary["run_id"].astype(str).unique(),
        *secondary["run_id"].astype(str).unique(),
        *primary_entities["run_id"].astype(str).unique(),
        *secondary_entities["run_id"].astype(str).unique(),
        *speech_annotations["run_id"].astype(str).unique(),
        "taxonomy-v1-20260719",
    ]
    frozen_runs, manifest_paths = _read_manifest_runs(repo_root, run_ids)

    source_paths = [repo_root / relative for relative in _DATA_SOURCE_PATHS]
    source_paths.extend(manifest_paths)
    inventory = tuple(
        _source_receipt(path, repo_root)
        for path in sorted(set(source_paths), key=lambda value: str(value))
    )
    return ProtocolInputs(
        paragraph_view=view,
        source_paragraphs=source_paragraphs,
        source_speeches=source_speeches,
        primary=primary,
        secondary=secondary,
        primary_entities=primary_entities,
        secondary_entities=secondary_entities,
        speech_annotations=speech_annotations,
        taxonomy=taxonomy,
        source_inventory=inventory,
        frozen_annotation_runs=frozen_runs,
    )


def _jaccard(left: Sequence[str] | set[str], right: Sequence[str] | set[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _entity_sets(
    entities: pd.DataFrame, allowed_keys: pd.DataFrame
) -> dict[tuple[str, int], frozenset[str]]:
    filtered = entities.merge(
        allowed_keys.loc[:, PARAGRAPH_KEY],
        on=PARAGRAPH_KEY,
        how="inner",
        validate="many_to_one",
    )
    values: dict[tuple[str, int], set[str]] = {}
    for row in filtered.itertuples(index=False):
        key = (str(row.doc_name), int(row.para_idx))
        # This is the governed foundation normalizer: Unicode-preserving
        # casefold plus collapsed whitespace, with no fuzzy alias inference.
        values.setdefault(key, set()).add(foundation_audit.normalize_entity(row.entity))
    return {key: frozenset(names) for key, names in values.items()}


def _prepare_population(inputs: ProtocolInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = inputs.paragraph_view.loc[
        inputs.paragraph_view["analysis_eligible"].astype(bool),
        [
            *PARAGRAPH_KEY, "text", "source_text_sha256", "year", "speech_type",
            "story_era_key", "story_era",
        ],
    ].copy()
    era_order = {spec.key: order for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)}
    eligible["story_era_order"] = eligible["story_era_key"].map(era_order)
    if eligible["story_era_order"].isna().any():
        raise ValidationProtocolError("eligible population has an ungoverned Story era")
    eligible["story_era_order"] = eligible["story_era_order"].astype(int)
    eligible["speech_type_family"] = eligible["speech_type"].map(SPEECH_TYPE_TO_FAMILY)

    judgment_columns = ["topics", *FLAG_FIELDS, "proposal_values"]
    primary = inputs.primary[[*PARAGRAPH_KEY, *judgment_columns]].rename(
        columns={column: f"{column}_primary" for column in judgment_columns}
    )
    master = eligible.merge(primary, on=PARAGRAPH_KEY, how="left", validate="one_to_one")
    if master[[f"{column}_primary" for column in judgment_columns]].isna().any().any():
        raise ValidationProtocolError("eligible population is not fully covered by primary labels")

    label_map = attention.canonical_label_map(dict(inputs.taxonomy))
    master["topic_set_primary"] = master["topics_primary"].map(
        lambda values: frozenset(attention.normalize_topics(values, label_map))
    )

    secondary = inputs.secondary[[*PARAGRAPH_KEY, *judgment_columns]].rename(
        columns={column: f"{column}_secondary" for column in judgment_columns}
    )
    paired = master.merge(secondary, on=PARAGRAPH_KEY, how="inner", validate="one_to_one")
    paired["topic_set_secondary"] = paired["topics_secondary"].map(
        lambda values: frozenset(attention.normalize_topics(values, label_map))
    )
    primary_names = _entity_sets(inputs.primary_entities, paired)
    secondary_names = _entity_sets(inputs.secondary_entities, paired)

    disagreements: list[float] = []
    for row in paired.itertuples(index=False):
        flag_score = sum(
            int(bool(getattr(row, f"{field}_primary")) != bool(getattr(row, f"{field}_secondary")))
            for field in FLAG_FIELDS
        )
        topic_score = 1.0 - _jaccard(row.topic_set_primary, row.topic_set_secondary)
        proposal_score = int(row.proposal_values_primary != row.proposal_values_secondary)
        key = (str(row.doc_name), int(row.para_idx))
        entity_score = 1.0 - _jaccard(
            primary_names.get(key, frozenset()), secondary_names.get(key, frozenset())
        )
        disagreements.append(float(flag_score + topic_score + proposal_score + entity_score))
    paired["selection_score"] = disagreements
    return master, paired


def _positive_label_counts(master: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {field: int(master[f"{field}_primary"].astype(bool).sum()) for field in FLAG_FIELDS}
    for topics in master["topic_set_primary"]:
        for topic in topics:
            counts[topic] = counts.get(topic, 0) + 1
    return {label: count for label, count in counts.items() if count >= RARE_LABEL_MIN_SUPPORT}


def _rare_score(topics: frozenset[str], row: pd.Series, counts: Mapping[str, int], n: int) -> float:
    labels = set(topics)
    labels.update(field for field in FLAG_FIELDS if bool(row[f"{field}_primary"]))
    eligible_counts = [counts[label] for label in labels if label in counts]
    if not eligible_counts:
        return 0.0
    return float(n / min(eligible_counts))


def _select_genre_balanced_paragraphs(
    pool: pd.DataFrame, era_key: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = pool.copy()
    work["paragraph_tie_break_sha256"] = [
        _rank_hash(
            "genre-balanced-base-paragraph", era_key, row.doc_name, int(row.para_idx)
        )
        for row in work.itertuples(index=False)
    ]
    selected_parts: list[pd.DataFrame] = []
    selected_keys: set[tuple[str, int]] = set()
    cells: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_rows = work.loc[work["speech_type_family"].eq(family)].copy()
        family_rows["document_tie_break_sha256"] = family_rows["doc_name"].map(
            lambda doc_name: _rank_hash(
                "genre-balanced-base-document", era_key, str(doc_name), -1
            )
        )
        document_order = (
            family_rows[["doc_name", "document_tie_break_sha256"]]
            .drop_duplicates("doc_name")
            .sort_values(["document_tie_break_sha256", "doc_name"], kind="stable")
        )
        target = PARAGRAPH_FAMILY_TARGETS[family]
        n_documents = len(document_order)
        selected_documents = document_order.head(min(target, n_documents))["doc_name"]
        document_probability = min(target, n_documents) / n_documents if n_documents else None
        picks_parts: list[pd.DataFrame] = []
        for doc_name in selected_documents:
            doc_rows = family_rows.loc[family_rows["doc_name"].eq(doc_name)].sort_values(
                ["paragraph_tie_break_sha256", "para_idx"], kind="stable"
            )
            pick = doc_rows.head(1).copy()
            paragraph_probability = 1.0 / len(doc_rows)
            probability = float(document_probability * paragraph_probability)
            pick["tie_break_sha256"] = pick["paragraph_tie_break_sha256"]
            pick["selection_probability"] = probability
            pick["sampling_weight"] = 1.0 / probability
            pick["conditional_stage_probability"] = probability
            pick["conditional_stage_weight"] = 1.0 / probability
            pick["probability_basis"] = (
                f"hash-random document {min(target, n_documents)}/{n_documents} × "
                f"hash-random paragraph 1/{len(doc_rows)}"
            )
            pick["weight_status"] = "design_weight_available"
            picks_parts.append(pick)
        picks = (
            pd.concat(picks_parts, ignore_index=True)
            if picks_parts else family_rows.head(0).copy()
        )
        picks["selection_stage"] = "unique-document-first-pass"
        picks["allocation_family"] = family
        selected_parts.append(picks)
        selected_keys.update(
            (str(row.doc_name), int(row.para_idx)) for row in picks.itertuples(index=False)
        )
        cells.append({
            "schema_version": SELECTION_CELL_SCHEMA,
            "study": "paragraph-human-validity",
            "story_era_key": era_key,
            "selection_arm": "genre-balanced-base",
            "allocation_cell": family,
            "candidate_units": int(len(family_rows)),
            "candidate_documents": int(n_documents),
            "target_units": int(target),
            "selected_first_pass": int(len(picks)),
            "selected_fallback": 0,
            "selected_total": int(len(picks)),
        })

    first_pass_n = sum(len(frame) for frame in selected_parts)
    shortfall = BASE_PER_ERA - first_pass_n
    remaining = work.loc[
        ~work.apply(lambda row: (str(row.doc_name), int(row.para_idx)) in selected_keys, axis=1)
    ].copy()
    remaining["tie_break_sha256"] = [
        _rank_hash(
            "genre-balanced-base-fallback", era_key, row.doc_name, int(row.para_idx)
        )
        for row in remaining.itertuples(index=False)
    ]
    remaining = remaining.sort_values(
        ["tie_break_sha256", "doc_name", "para_idx"], kind="stable"
    )
    fallback = remaining.head(shortfall).copy()
    fallback["selection_stage"] = "era-remainder-fallback"
    fallback["allocation_family"] = "fallback"
    fallback["selection_probability"] = float("nan")
    fallback["sampling_weight"] = float("nan")
    conditional_probability = shortfall / len(remaining) if shortfall else None
    fallback["conditional_stage_probability"] = conditional_probability
    fallback["conditional_stage_weight"] = (
        1.0 / conditional_probability if conditional_probability else float("nan")
    )
    fallback["probability_basis"] = (
        "conditional hash-random draw from the realized post-first-pass remainder; "
        "marginal inclusion probability not derived"
    )
    fallback["weight_status"] = "not_available_sequential_fallback"
    selected_parts.append(fallback)
    cells.append({
        "schema_version": SELECTION_CELL_SCHEMA,
        "study": "paragraph-human-validity",
        "story_era_key": era_key,
        "selection_arm": "genre-balanced-base",
        "allocation_cell": "fallback",
        "candidate_units": int(len(remaining)),
        "candidate_documents": int(remaining["doc_name"].nunique()),
        "target_units": int(shortfall),
        "selected_first_pass": 0,
        "selected_fallback": int(len(fallback)),
        "selected_total": int(len(fallback)),
    })
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != BASE_PER_ERA:
        raise ValidationProtocolError(
            f"{era_key}: genre-balanced base produced {len(selected)} rows, expected {BASE_PER_ERA}"
        )
    _require_unique(selected, PARAGRAPH_KEY, f"{era_key} base selection")
    return selected, cells


def _paragraph_selection(
    master: pd.DataFrame, paired: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, int]]:
    selected_parts: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    high_keys: set[tuple[str, int]] = set()
    rare_keys: set[tuple[str, int]] = set()
    candidate_counts: dict[tuple[str, str], int] = {}
    label_counts = _positive_label_counts(master)

    for spec in era_profiles.ERA_PROFILE_SPECS:
        candidates = paired.loc[paired["story_era_key"].eq(spec.key)].copy()
        candidates["tie_break_sha256"] = [
            _rank_hash("high-disagreement", spec.key, row.doc_name, int(row.para_idx))
            for row in candidates.itertuples(index=False)
        ]
        candidates = candidates.sort_values(
            ["selection_score", "tie_break_sha256", "doc_name", "para_idx"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        if len(candidates) < HIGH_DISAGREEMENT_PER_ERA:
            raise ValidationProtocolError(f"{spec.key}: fewer than 10 paired candidates")
        picks = candidates.head(HIGH_DISAGREEMENT_PER_ERA).copy()
        picks["selection_arm"] = "high-disagreement"
        picks["selection_stage"] = "ranked-enrichment"
        picks["allocation_family"] = "all"
        picks["selection_probability"] = float("nan")
        picks["sampling_weight"] = float("nan")
        picks["conditional_stage_probability"] = float("nan")
        picks["conditional_stage_weight"] = float("nan")
        picks["probability_basis"] = "purposive top-k by disagreement score"
        picks["weight_status"] = "not_applicable_purposive_enrichment"
        selected_parts.append(picks)
        candidate_counts[(spec.key, "high-disagreement")] = int(len(candidates))
        high_keys.update((str(row.doc_name), int(row.para_idx)) for row in picks.itertuples())
        cell_rows.append({
            "schema_version": SELECTION_CELL_SCHEMA,
            "study": "paragraph-human-validity",
            "story_era_key": spec.key,
            "selection_arm": "high-disagreement",
            "allocation_cell": "all-paired-eligible",
            "candidate_units": int(len(candidates)),
            "candidate_documents": int(candidates["doc_name"].nunique()),
            "target_units": HIGH_DISAGREEMENT_PER_ERA,
            "selected_first_pass": HIGH_DISAGREEMENT_PER_ERA,
            "selected_fallback": 0,
            "selected_total": HIGH_DISAGREEMENT_PER_ERA,
        })

    for spec in era_profiles.ERA_PROFILE_SPECS:
        candidates = master.loc[master["story_era_key"].eq(spec.key)].copy()
        candidates = candidates.loc[
            ~candidates.apply(
                lambda row: (str(row.doc_name), int(row.para_idx)) in high_keys, axis=1
            )
        ]
        candidates["selection_score"] = candidates.apply(
            lambda row: _rare_score(
                row["topic_set_primary"], row, label_counts, len(master)
            ),
            axis=1,
        )
        candidates["tie_break_sha256"] = [
            _rank_hash("rare-positive", spec.key, row.doc_name, int(row.para_idx))
            for row in candidates.itertuples(index=False)
        ]
        candidates = candidates.sort_values(
            ["selection_score", "tie_break_sha256", "doc_name", "para_idx"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        if len(candidates) < RARE_POSITIVE_PER_ERA:
            raise ValidationProtocolError(f"{spec.key}: fewer than 10 rare-positive candidates")
        picks = candidates.head(RARE_POSITIVE_PER_ERA).copy()
        picks["selection_arm"] = "rare-positive"
        picks["selection_stage"] = "ranked-enrichment"
        picks["allocation_family"] = "all"
        picks["selection_probability"] = float("nan")
        picks["sampling_weight"] = float("nan")
        picks["conditional_stage_probability"] = float("nan")
        picks["conditional_stage_weight"] = float("nan")
        picks["probability_basis"] = "purposive top-k by inverse prevalence score"
        picks["weight_status"] = "not_applicable_purposive_enrichment"
        selected_parts.append(picks)
        candidate_counts[(spec.key, "rare-positive")] = int(len(candidates))
        rare_keys.update((str(row.doc_name), int(row.para_idx)) for row in picks.itertuples())
        cell_rows.append({
            "schema_version": SELECTION_CELL_SCHEMA,
            "study": "paragraph-human-validity",
            "story_era_key": spec.key,
            "selection_arm": "rare-positive",
            "allocation_cell": "all-remaining-eligible",
            "candidate_units": int(len(candidates)),
            "candidate_documents": int(candidates["doc_name"].nunique()),
            "target_units": RARE_POSITIVE_PER_ERA,
            "selected_first_pass": RARE_POSITIVE_PER_ERA,
            "selected_fallback": 0,
            "selected_total": RARE_POSITIVE_PER_ERA,
        })

    excluded = high_keys | rare_keys
    for spec in era_profiles.ERA_PROFILE_SPECS:
        candidates = master.loc[master["story_era_key"].eq(spec.key)].copy()
        candidates = candidates.loc[
            ~candidates.apply(
                lambda row: (str(row.doc_name), int(row.para_idx)) in excluded, axis=1
            )
        ]
        candidate_counts[(spec.key, "genre-balanced-base")] = int(len(candidates))
        picks, cells = _select_genre_balanced_paragraphs(candidates, spec.key)
        picks["selection_arm"] = "genre-balanced-base"
        picks["selection_score"] = 0.0
        selected_parts.append(picks)
        cell_rows.extend(cells)

    selected = pd.concat(selected_parts, ignore_index=True)
    _require_unique(selected, PARAGRAPH_KEY, "paragraph validation selection")
    if len(selected) != PARAGRAPHS_PER_ERA * len(era_profiles.ERA_PROFILE_SPECS):
        raise ValidationProtocolError("paragraph selection does not contain exactly 360 keys")

    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        arm = str(row.selection_arm)
        candidate_count = candidate_counts[(str(row.story_era_key), arm)]
        selected_count = {
            "high-disagreement": HIGH_DISAGREEMENT_PER_ERA,
            "rare-positive": RARE_POSITIVE_PER_ERA,
            "genre-balanced-base": BASE_PER_ERA,
        }[arm]
        probability = getattr(row, "selection_probability")
        weight = getattr(row, "sampling_weight")
        conditional_probability = getattr(row, "conditional_stage_probability")
        conditional_weight = getattr(row, "conditional_stage_weight")
        rows.append({
            "schema_version": PARAGRAPH_SAMPLE_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "study_item_id": _opaque_id("paragraph-item", row.doc_name, int(row.para_idx)),
            "doc_name": str(row.doc_name),
            "para_idx": int(row.para_idx),
            "source_text_sha256": str(row.source_text_sha256),
            "year": int(row.year),
            "story_era_key": str(row.story_era_key),
            "story_era_label": str(row.story_era),
            "story_era_order": int(row.story_era_order),
            "selection_arm": arm,
            "selection_stage": str(row.selection_stage),
            "speech_type_family": str(row.speech_type_family),
            "allocation_family": str(row.allocation_family),
            "candidate_count": int(candidate_count),
            "selected_count": int(selected_count),
            "selection_probability": (
                None if pd.isna(probability) else float(probability)
            ),
            "sampling_weight": None if pd.isna(weight) else float(weight),
            "conditional_stage_probability": (
                None if pd.isna(conditional_probability) else float(conditional_probability)
            ),
            "conditional_stage_weight": (
                None if pd.isna(conditional_weight) else float(conditional_weight)
            ),
            "probability_basis": str(row.probability_basis),
            "weight_status": str(row.weight_status),
            "selection_score": float(row.selection_score),
            "tie_break_sha256": str(row.tie_break_sha256),
            "presentation_sha256": _rank_hash(
                "presentation", row.story_era_key, row.doc_name, int(row.para_idx)
            ),
        })
    output = pd.DataFrame(rows)
    output["arm_order"] = output["selection_arm"].map(ARM_ORDER)
    output = output.sort_values(
        ["story_era_order", "arm_order", "tie_break_sha256"], kind="stable"
    ).drop(columns="arm_order").reset_index(drop=True)
    output["selection_order"] = range(1, len(output) + 1)
    presentation_order = {
        item_id: order
        for order, item_id in enumerate(
            output.sort_values(["presentation_sha256", "study_item_id"], kind="stable")[
                "study_item_id"
            ],
            start=1,
        )
    }
    output["presentation_order"] = output["study_item_id"].map(presentation_order).astype(int)
    output = output[[
        "schema_version", "protocol_version", "selection_order", "presentation_order",
        "study_item_id", "doc_name", "para_idx", "source_text_sha256", "year",
        "story_era_key", "story_era_label", "story_era_order", "selection_arm",
        "selection_stage", "speech_type_family", "allocation_family",
        "candidate_count", "selected_count", "selection_probability",
        "sampling_weight", "conditional_stage_probability",
        "conditional_stage_weight", "probability_basis", "weight_status",
        "selection_score", "tie_break_sha256",
        "presentation_sha256",
    ]]
    cell_frame = pd.DataFrame(cell_rows)
    era_order = {spec.key: order for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)}
    cell_frame["story_era_order"] = cell_frame["story_era_key"].map(era_order).astype(int)
    cell_frame["arm_order"] = cell_frame["selection_arm"].map(ARM_ORDER).astype(int)
    cell_frame = cell_frame.sort_values(
        ["story_era_order", "arm_order", "allocation_cell"], kind="stable"
    ).drop(columns="arm_order").reset_index(drop=True)
    cell_frame = cell_frame[[
        "schema_version", "study", "story_era_key", "story_era_order",
        "selection_arm", "allocation_cell", "candidate_units",
        "candidate_documents", "target_units", "selected_first_pass",
        "selected_fallback", "selected_total",
    ]]
    return output, cell_frame, label_counts


def _select_speeches(
    _paragraph_master: pd.DataFrame, inputs: ProtocolInputs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select factual tasks from the complete source-speech annotation population."""
    source = inputs.source_speeches[
        ["doc_name", "president", "title", "year"]
    ].copy()
    factual = inputs.speech_annotations[["doc_name", "speech_type"]].copy()
    docs = source.merge(factual, on="doc_name", how="inner", validate="one_to_one")
    if len(docs) != EXPECTED_FACTUAL_SOURCE_SPEECHES:
        raise ValidationProtocolError(
            "factual candidate population must contain all "
            f"{EXPECTED_FACTUAL_SOURCE_SPEECHES:,} source speeches"
        )
    docs["story_era_key"] = era_profiles.story_era_key_series(
        docs["year"], docs["president"]
    )
    docs["story_era"] = era_profiles.story_era_series(
        docs["year"], docs["president"]
    )
    era_order = {
        spec.key: order for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)
    }
    docs["story_era_order"] = docs["story_era_key"].map(era_order)
    if docs["story_era_order"].isna().any():
        raise ValidationProtocolError("a factual source speech has no governed Story era")
    docs["story_era_order"] = docs["story_era_order"].astype(int)
    docs["speech_type_family"] = docs["speech_type"].map(SPEECH_TYPE_TO_FAMILY)
    if docs["speech_type_family"].isna().any():
        raise ValidationProtocolError("a factual source speech has no genre family")
    _require_unique(docs, ["doc_name"], "complete factual speech population")

    selected_parts: list[pd.DataFrame] = []
    cells: list[dict[str, Any]] = []
    for spec in era_profiles.ERA_PROFILE_SPECS:
        era = docs.loc[docs["story_era_key"].eq(spec.key)].copy()
        era["tie_break_sha256"] = [
            _rank_hash("factual-speech", spec.key, row.doc_name, -1)
            for row in era.itertuples(index=False)
        ]
        chosen_docs: set[str] = set()
        parts: list[pd.DataFrame] = []
        for family in FAMILY_ORDER:
            candidates = era.loc[era["speech_type_family"].eq(family)].sort_values(
                ["tie_break_sha256", "doc_name"], kind="stable"
            )
            target = SPEECH_FAMILY_TARGETS[family]
            picks = candidates.head(min(target, len(candidates))).copy()
            picks["selection_stage"] = "genre-allocation-first-pass"
            picks["allocation_family"] = family
            probability = len(picks) / len(candidates) if len(candidates) else None
            picks["selection_probability"] = probability
            picks["sampling_weight"] = 1.0 / probability if probability else float("nan")
            picks["conditional_stage_probability"] = probability
            picks["conditional_stage_weight"] = (
                1.0 / probability if probability else float("nan")
            )
            picks["probability_basis"] = (
                f"hash-random document {len(picks)}/{len(candidates)} within genre family"
            )
            picks["weight_status"] = "design_weight_available"
            picks["stage_candidate_count"] = len(candidates)
            picks["stage_selected_count"] = len(picks)
            parts.append(picks)
            chosen_docs.update(picks["doc_name"].astype(str))
            cells.append({
                "schema_version": SELECTION_CELL_SCHEMA,
                "study": "speech-factual-human-validity",
                "story_era_key": spec.key,
                "selection_arm": "factual-genre-balanced",
                "allocation_cell": family,
                "candidate_units": int(len(candidates)),
                "candidate_documents": int(len(candidates)),
                "target_units": int(target),
                "selected_first_pass": int(len(picks)),
                "selected_fallback": 0,
                "selected_total": int(len(picks)),
            })
        first_n = sum(len(part) for part in parts)
        shortfall = SPEECHES_PER_ERA - first_n
        remaining = era.loc[~era["doc_name"].isin(chosen_docs)].sort_values(
            ["tie_break_sha256", "doc_name"], kind="stable"
        )
        fallback = remaining.head(shortfall).copy()
        fallback["selection_stage"] = "era-remainder-fallback"
        fallback["allocation_family"] = "fallback"
        fallback["selection_probability"] = float("nan")
        fallback["sampling_weight"] = float("nan")
        conditional_probability = shortfall / len(remaining) if shortfall else None
        fallback["conditional_stage_probability"] = conditional_probability
        fallback["conditional_stage_weight"] = (
            1.0 / conditional_probability if conditional_probability else float("nan")
        )
        fallback["probability_basis"] = (
            "conditional hash-random draw from the realized post-allocation remainder; "
            "marginal inclusion probability not derived"
        )
        fallback["weight_status"] = "not_available_sequential_fallback"
        fallback["stage_candidate_count"] = len(remaining)
        fallback["stage_selected_count"] = len(fallback)
        parts.append(fallback)
        cells.append({
            "schema_version": SELECTION_CELL_SCHEMA,
            "study": "speech-factual-human-validity",
            "story_era_key": spec.key,
            "selection_arm": "factual-genre-balanced",
            "allocation_cell": "fallback",
            "candidate_units": int(len(remaining)),
            "candidate_documents": int(len(remaining)),
            "target_units": int(shortfall),
            "selected_first_pass": 0,
            "selected_fallback": int(len(fallback)),
            "selected_total": int(len(fallback)),
        })
        selected = pd.concat(parts, ignore_index=True)
        if len(selected) != SPEECHES_PER_ERA:
            raise ValidationProtocolError(
                f"{spec.key}: factual sample produced {len(selected)} speeches, expected 10"
            )
        selected_parts.append(selected)

    selected = pd.concat(selected_parts, ignore_index=True)
    _require_unique(selected, ["doc_name"], "factual validation sample")
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        candidate_count = int(row.stage_candidate_count)
        selected_count = int(row.stage_selected_count)
        probability = row.selection_probability
        weight = row.sampling_weight
        conditional_probability = row.conditional_stage_probability
        conditional_weight = row.conditional_stage_weight
        rows.append({
            "schema_version": SPEECH_SAMPLE_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "study_item_id": _opaque_id("speech-item", row.doc_name, -1),
            "doc_name": str(row.doc_name),
            "year": int(row.year),
            "story_era_key": str(row.story_era_key),
            "story_era_label": str(row.story_era),
            "story_era_order": int(row.story_era_order),
            "selection_arm": "factual-genre-balanced",
            "selection_stage": str(row.selection_stage),
            "speech_type_family": str(row.speech_type_family),
            "allocation_family": str(row.allocation_family),
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "selection_probability": (
                None if pd.isna(probability) else float(probability)
            ),
            "sampling_weight": None if pd.isna(weight) else float(weight),
            "conditional_stage_probability": (
                None if pd.isna(conditional_probability) else float(conditional_probability)
            ),
            "conditional_stage_weight": (
                None if pd.isna(conditional_weight) else float(conditional_weight)
            ),
            "probability_basis": str(row.probability_basis),
            "weight_status": str(row.weight_status),
            "tie_break_sha256": str(row.tie_break_sha256),
        })
    output = pd.DataFrame(rows).sort_values(
        ["story_era_order", "selection_stage", "tie_break_sha256"], kind="stable"
    ).reset_index(drop=True)
    output["selection_order"] = range(1, len(output) + 1)
    output = output[[
        "schema_version", "protocol_version", "selection_order", "study_item_id",
        "doc_name", "year", "story_era_key", "story_era_label", "story_era_order",
        "selection_arm", "selection_stage", "speech_type_family", "allocation_family",
        "candidate_count", "selected_count", "selection_probability", "sampling_weight",
        "conditional_stage_probability", "conditional_stage_weight", "probability_basis",
        "weight_status", "tie_break_sha256",
    ]]
    cell_frame = pd.DataFrame(cells)
    era_order = {spec.key: order for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)}
    cell_frame["story_era_order"] = cell_frame["story_era_key"].map(era_order).astype(int)
    cell_frame = cell_frame.sort_values(
        ["story_era_order", "allocation_cell"], kind="stable"
    ).reset_index(drop=True)
    cell_frame = cell_frame[[
        "schema_version", "study", "story_era_key", "story_era_order",
        "selection_arm", "allocation_cell", "candidate_units",
        "candidate_documents", "target_units", "selected_first_pass",
        "selected_fallback", "selected_total",
    ]]
    return output, cell_frame


_CURRENT_CONTEXT_TEMPLATE = (
    "SPEECH ERA: the <DECADE>s. The speaker's name, party, and the speech "
    "title are withheld. Judge every paragraph only from the words below, "
    "against the norms of the <DECADE>s — not today's.\n\n"
    "COUNT CONTRACT: you are given 1 paragraph(s) below, with para_idx "
    "values [<PARA_IDX>]. Return EXACTLY 1 annotation object(s) — exactly "
    "one per para_idx listed, no more, no fewer, and no duplicates.\n\n"
    "PARAGRAPHS:\n\n[para_idx=<PARA_IDX>]\n<PARAGRAPH_TEXT>\n\n"
    "END OF PARAGRAPHS. Remember: the `annotations` array must contain "
    "exactly 1 object(s), one for each of para_idx [<PARA_IDX>]."
)

_CURRENT_SYSTEM_OPENING = (
    "You are a careful annotator of United States presidential rhetoric. You are given\n"
    "several consecutive paragraphs from a single speech, and the DECADE in which the\n"
    "speech was delivered. The speaker's name, party, and the speech title are\n"
    "withheld on purpose: judge each paragraph on its own words, not on any guess\n"
    "about who wrote it.\n\n"
    "ERA-ANCHORING (important): judge combativeness against the norms of the speech's\n"
    "OWN era, not against 2026 norms. Formal 19th-century invective and plain-spoken\n"
    "21st-century attacks can both be \"combative\" for their time.\n\n"
)
_HIDDEN_SYSTEM_OPENING = (
    "You are a careful annotator of United States presidential rhetoric. You are given\n"
    "several consecutive paragraphs from a single speech. The speaker's name, party,\n"
    "and the speech title are withheld on purpose: judge each paragraph on its own\n"
    "words, not on any guess about who wrote it.\n\n"
)
_CURRENT_INSTRUCTION_SUFFIX = (
    ", and judge combativeness against the paragraph's own era."
)
_HIDDEN_CONTEXT_OPENING = (
    "The speaker's name, party, and the speech title are withheld. Judge every "
    "paragraph only from the words below.\n\n"
)


def _hide_decade_prompt(current: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the complete allowlisted temporal-context removal and nothing else."""
    system = str(current["system"])
    if system.count(_CURRENT_SYSTEM_OPENING) != 1:
        raise ValidationProtocolError("production system prompt temporal seam drifted")
    instruction = str(current["user_instruction"])
    if instruction.count(_CURRENT_INSTRUCTION_SUFFIX) != 1:
        raise ValidationProtocolError("production user instruction temporal seam drifted")
    context = str(current["context_template"])
    current_opening = context.split("COUNT CONTRACT:", 1)[0]
    if current_opening != _CURRENT_CONTEXT_TEMPLATE.split("COUNT CONTRACT:", 1)[0]:
        raise ValidationProtocolError("production context-template temporal seam drifted")
    hidden = {
        "system": system.replace(_CURRENT_SYSTEM_OPENING, _HIDDEN_SYSTEM_OPENING, 1),
        "user_instruction": instruction.replace(_CURRENT_INSTRUCTION_SUFFIX, ".", 1),
        "context_template": _HIDDEN_CONTEXT_OPENING + "COUNT CONTRACT:" + context.split(
            "COUNT CONTRACT:", 1
        )[1],
        "output_schema": current["output_schema"],
    }
    return hidden


def build_prompt_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze production prompt bytes and its allowlisted decade-hidden mate."""
    current = {
        "system": prompt_v1.JUDGMENT_RUBRIC,
        "user_instruction": prompt_v1.JUDGMENT_INSTRUCTION,
        "context_template": _CURRENT_CONTEXT_TEMPLATE,
        "output_schema": prompt_v1.JUDGMENT_SCHEMA,
    }
    hidden = _hide_decade_prompt(current)
    _validate_prompt_pair(current, hidden)
    return current, hidden


def render_current_user_prompt(*, paragraph_text: str, decade: int, para_idx: int) -> str:
    """Render the frozen one-item template; used to prove production parity."""
    context = (
        _CURRENT_CONTEXT_TEMPLATE.replace("<DECADE>", str(int(decade)))
        .replace("<PARA_IDX>", str(int(para_idx)))
        .replace("<PARAGRAPH_TEXT>", str(paragraph_text))
    )
    return prompt_v1.JUDGMENT_INSTRUCTION + "\n\n" + context


def _validate_prompt_pair(current: Mapping[str, Any], hidden: Mapping[str, Any]) -> None:
    if dict(hidden) != _hide_decade_prompt(current):
        raise ValidationProtocolError("prompt arms differ outside the allowlisted decade removal")
    if current["output_schema"] != hidden["output_schema"]:
        raise ValidationProtocolError("prompt arms do not share one output schema")
    current_context = str(current["context_template"])
    hidden_context = str(hidden["context_template"])
    if current_context.count("<PARAGRAPH_TEXT>") != 1 or hidden_context.count("<PARAGRAPH_TEXT>") != 1:
        raise ValidationProtocolError("prompt templates must carry one identical paragraph slot")
    if "<DECADE>" not in current_context or "<DECADE>" in hidden_context:
        raise ValidationProtocolError("decade field was not removed exactly from the hidden arm")
    if "DECADE" in str(hidden["system"]) or "OWN era" in str(hidden["system"]):
        raise ValidationProtocolError("decade-hidden system prompt retains explicit era anchoring")


def _paragraph_tasks(
    paragraph_sample: pd.DataFrame, master: pd.DataFrame
) -> tuple[Mapping[str, Any], ...]:
    evidence = master[[*PARAGRAPH_KEY, "text"]]
    joined = paragraph_sample.merge(evidence, on=PARAGRAPH_KEY, how="left", validate="one_to_one")
    if joined[["text", "year"]].isna().any().any():
        raise ValidationProtocolError("selected paragraph is missing blinded evidence")
    joined = joined.sort_values("presentation_order", kind="stable")
    return tuple({
        "schema_version": BLINDED_PARAGRAPH_SCHEMA,
        "item_id": str(row.study_item_id),
        "paragraph_text": str(row.text),
        "decade": f"{int(row.year) // 10 * 10}s",
    } for row in joined.itertuples(index=False))


def _speech_tasks(
    speech_sample: pd.DataFrame, inputs: ProtocolInputs
) -> tuple[Mapping[str, Any], ...]:
    speech_info = inputs.source_speeches.set_index("doc_name")
    paragraph_groups = {
        doc_name: rows.sort_values("para_idx", kind="stable")["text"].head(
            prompt_v1.FACTUAL_INTRO_PARAS
        ).astype(str).tolist()
        for doc_name, rows in inputs.source_paragraphs.groupby("doc_name", sort=False)
    }
    tasks: list[Mapping[str, Any]] = []
    for row in speech_sample.sort_values("selection_order", kind="stable").itertuples(index=False):
        if row.doc_name not in speech_info.index or row.doc_name not in paragraph_groups:
            raise ValidationProtocolError("selected speech lacks factual evidence")
        speech = speech_info.loc[row.doc_name]
        tasks.append({
            "schema_version": BLINDED_SPEECH_SCHEMA,
            "item_id": str(row.study_item_id),
            "title": str(speech["title"]),
            "year": int(speech["year"]),
            "opening_paragraphs": paragraph_groups[row.doc_name],
        })
    return tuple(tasks)


def _request_plan(
    paragraph_sample: pd.DataFrame,
    current_prompt_sha256: str,
    hidden_prompt_sha256: str,
) -> pd.DataFrame:
    rows = []
    ordered = paragraph_sample.sort_values("presentation_order", kind="stable")
    for row in ordered.itertuples(index=False):
        request_id = "miv_" + hashlib.sha256(
            f"{SEED_NAMESPACE}|invariance-request|{row.doc_name}|{int(row.para_idx)}".encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        rows.append({
            "schema_version": REQUEST_PLAN_SCHEMA,
            "presentation_order": int(row.presentation_order),
            "request_id": request_id,
            "study_item_id": str(row.study_item_id),
            "items_per_request": 1,
            "planned_arms": 2,
            "planned_evaluations": 2,
            "current_context_prompt_sha256": current_prompt_sha256,
            "decade_hidden_prompt_sha256": hidden_prompt_sha256,
            "runtime_model": "UNSET",
            "execution_status": "blocked_not_run",
        })
    return pd.DataFrame(rows)


def estimate_cost(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
    cache_write_multiplier: float | None,
    cache_read_multiplier: float | None,
    batch_discount_multiplier: float | None,
) -> dict[str, Any]:
    """Return a cost receipt, blocked until every model-specific input is set."""
    inputs = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "input_usd_per_million": input_usd_per_million,
        "output_usd_per_million": output_usd_per_million,
        "cache_write_multiplier": cache_write_multiplier,
        "cache_read_multiplier": cache_read_multiplier,
        "batch_discount_multiplier": batch_discount_multiplier,
    }
    formula = (
        "batch_discount_multiplier * ((input_tokens + "
        "cache_creation_input_tokens * cache_write_multiplier + "
        "cache_read_input_tokens * cache_read_multiplier) * input_usd_per_million + "
        "output_tokens * output_usd_per_million) / 1_000_000"
    )
    if any(value is None for value in inputs.values()):
        return {
            **inputs,
            "formula": formula,
            "estimated_cost_usd": None,
            "status": "blocked_model_price_and_token_inputs_unset",
        }
    numeric = {key: float(value) for key, value in inputs.items()}
    if any(not math.isfinite(value) or value < 0 for value in numeric.values()):
        raise ValidationProtocolError("cost-estimation inputs must be finite and non-negative")
    billable_input = (
        numeric["input_tokens"]
        + numeric["cache_creation_input_tokens"] * numeric["cache_write_multiplier"]
        + numeric["cache_read_input_tokens"] * numeric["cache_read_multiplier"]
    )
    estimate = numeric["batch_discount_multiplier"] * (
        billable_input * numeric["input_usd_per_million"]
        + numeric["output_tokens"] * numeric["output_usd_per_million"]
    ) / 1_000_000
    return {
        **inputs,
        "formula": formula,
        "estimated_cost_usd": float(estimate),
        "status": "estimate_ready_requires_separate_approval",
    }


def _semantic_sample_hash(frame: pd.DataFrame) -> str:
    rows = frame.sort_values(PARAGRAPH_KEY, kind="stable")[[
        *PARAGRAPH_KEY, "selection_arm"
    ]].to_dict("records")
    return _sha256_bytes(
        "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")
    )


def _paragraph_key_hash(frame: pd.DataFrame) -> str:
    rows = frame.sort_values(PARAGRAPH_KEY, kind="stable")[PARAGRAPH_KEY].to_dict("records")
    return _sha256_bytes(
        "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")
    )


def _speech_key_hash(frame: pd.DataFrame) -> str:
    rows = frame.sort_values("doc_name", kind="stable")[["doc_name"]].to_dict("records")
    return _sha256_bytes(
        "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")
    )


def _artifact_receipt(filename: str, payload: bytes, rows: int | None) -> dict[str, Any]:
    return {
        "filename": filename,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "rows": rows,
    }


def _manifest(
    *,
    inputs: ProtocolInputs,
    paragraph_sample: pd.DataFrame,
    paragraph_cells: pd.DataFrame,
    paragraph_tasks: Sequence[Mapping[str, Any]],
    speech_sample: pd.DataFrame,
    speech_cells: pd.DataFrame,
    speech_tasks: Sequence[Mapping[str, Any]],
    request_plan: pd.DataFrame,
    current_prompt_bytes: bytes,
    hidden_prompt_bytes: bytes,
    artifact_files: Mapping[str, bytes],
) -> dict[str, Any]:
    era_records = [
        {
            "key": spec.key,
            "label": spec.label,
            "start_year": int(spec.start_year),
            "end_year": int(spec.end_year),
            "order": order,
        }
        for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)
    ]
    row_counts = {
        PARAGRAPH_SAMPLE_NAME: len(paragraph_sample),
        PARAGRAPH_CELLS_NAME: len(paragraph_cells),
        PARAGRAPH_TASKS_NAME: len(paragraph_tasks),
        SPEECH_SAMPLE_NAME: len(speech_sample),
        SPEECH_CELLS_NAME: len(speech_cells),
        SPEECH_TASKS_NAME: len(speech_tasks),
        REQUEST_PLAN_NAME: len(request_plan),
        CURRENT_PROMPT_NAME: None,
        HIDDEN_PROMPT_NAME: None,
    }
    artifacts = {
        filename: _artifact_receipt(filename, payload, row_counts[filename])
        for filename, payload in sorted(artifact_files.items())
    }
    eligible = inputs.paragraph_view["analysis_eligible"].astype(bool)
    paired_keys = _key_set(inputs.secondary) & _key_set(inputs.paragraph_view.loc[eligible])
    cost = estimate_cost(
        input_tokens=None,
        output_tokens=None,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        input_usd_per_million=None,
        output_usd_per_million=None,
        cache_write_multiplier=None,
        cache_read_multiplier=None,
        batch_discount_multiplier=None,
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "generation_status": "deterministic_local_no_api_calls",
        "publication_status": "planned_validation_only_no_results",
        "execution_authorized": False,
        "selection_seed_namespace": SEED_NAMESPACE,
        "semantic_key": PARAGRAPH_KEY,
        "story_eras": era_records,
        "populations": {
            "eligible_paragraphs": int(eligible.sum()),
            "eligible_source_documents": int(
                inputs.paragraph_view.loc[eligible, "doc_name"].nunique()
            ),
            "paired_evaluation_paragraphs_within_eligible": int(len(paired_keys)),
            "factual_candidate_source_documents": int(len(inputs.source_speeches)),
            "selected_human_validity_paragraphs": len(paragraph_sample),
            "selected_factual_source_documents": len(speech_sample),
        },
        "selection": {
            "paragraph_quota": {
                "per_story_era": PARAGRAPHS_PER_ERA,
                "genre_balanced_base": BASE_PER_ERA,
                "rare_positive": RARE_POSITIVE_PER_ERA,
                "high_disagreement": HIGH_DISAGREEMENT_PER_ERA,
            },
            "factual_speech_quota": {
                "per_story_era": SPEECHES_PER_ERA,
                "family_targets": SPEECH_FAMILY_TARGETS,
                "speech_level_hash_para_idx_sentinel": -1,
            },
            "arm_priority": [
                "high-disagreement", "rare-positive", "genre-balanced-base"
            ],
            "tie_break_material": (
                "<seed namespace>|<arm>|<story_era_key>|<doc_name>|<para_idx>"
            ),
            "probability_sampling_hash_arms": {
                "base_document": "genre-balanced-base-document with para_idx=-1",
                "base_paragraph_within_document": "genre-balanced-base-paragraph",
                "base_fallback": "genre-balanced-base-fallback",
                "factual_document": "factual-speech with para_idx=-1",
            },
            "rare_label_min_global_eligible_paragraphs": RARE_LABEL_MIN_SUPPORT,
            "selection_probability_status": (
                "design probabilities and inverse weights exist only for independent "
                "hash-random first-pass base/factual strata; purposive enrichment and "
                "sequential fallback rows have null population weights"
            ),
            "population_inference_gate": (
                "never use high-disagreement, rare-positive, or fallback rows for "
                "population-weighted inference; report them descriptively"
            ),
            "paragraph_sample_sha256": _semantic_sample_hash(paragraph_sample),
            "paragraph_key_set_sha256": _paragraph_key_hash(paragraph_sample),
            "factual_speech_key_set_sha256": _speech_key_hash(speech_sample),
            "presentation_order_sha256": _sha256_bytes(
                "\n".join(
                    paragraph_sample.sort_values("presentation_order")["study_item_id"]
                ).encode("utf-8")
            ),
        },
        "blinding": {
            "paragraph_task_fields": sorted(_BLINDED_PARAGRAPH_FIELDS),
            "speech_task_fields": sorted(_BLINDED_SPEECH_FIELDS),
            "prohibited_fields": sorted(_PROHIBITED_BLINDED_FIELDS),
            "paragraph_evidence": "paragraph text plus decade only",
            "speech_evidence": "title, year, and up to the first five source paragraphs",
            "validation_status": "passed",
        },
        "human_validity_study": {
            "status": "planned_not_run",
            "coders": {"independent_blinded_coders_required": 2, "named_coders": []},
            "adjudication": {
                "third_reviewer_required": True,
                "all_field_disagreements": True,
                "deterministic_agreement_audit_fraction": 0.10,
            },
            "reporting": {
                "unweighted_enriched_results": True,
                "population_weighted_results": (
                    "allowed only for hash-random first-pass genre strata with "
                    "design_weight_available; enriched and fallback rows excluded"
                ),
                "enrichment_population_weighting_prohibited": True,
                "bootstrap_unit": "doc_name",
                "bootstrap_draws": 2_000,
                "confidence_interval": "95% percentile",
                "minimum_evaluated_units": 20,
                "minimum_positive_cases": 5,
                "minimum_source_documents": 5,
                "suppression_status": "suppressed_low_support",
            },
            "metrics": [
                "human-human Cohen kappa for binary and factual fields",
                "human-human exact proposal/values agreement",
                "topic mean Jaccard and micro/macro precision recall F1",
                "entity strict normalized-name detection plus type and stance agreement",
                "model-human precision recall F1 against both coders and adjudication",
            ],
            "execution_gates": {
                "named_coders": "blocked_unset",
                "conflict_and_privacy_review": "blocked_unset",
                "codebook_frozen": "blocked_unset",
                "written_approval": "blocked_unset",
                "key_and_blinding_checks": "passed",
            },
        },
        "measurement_invariance_study": {
            "status": "dry_run_only_blocked_not_run",
            "same_360_key_sample": True,
            "request_partition": {
                "one_paragraph_per_request": True,
                "requests_per_arm": len(request_plan),
                "planned_evaluations_total": len(request_plan) * 2,
                "presentation_order_persisted": True,
            },
            "prompt_pair": {
                "current_context_file": CURRENT_PROMPT_NAME,
                "current_context_sha256": _sha256_bytes(current_prompt_bytes),
                "decade_hidden_file": HIDDEN_PROMPT_NAME,
                "decade_hidden_sha256": _sha256_bytes(hidden_prompt_bytes),
                "difference_policy": (
                    "same rubric, taxonomy, examples, output schema, and paragraph slot; "
                    "only the decade field and explicit era-anchoring instructions are removed"
                ),
                "difference_validation_status": "passed",
            },
            "runtime_model": None,
            "decoding_configuration": None,
            "retry_policy": None,
            "planned_comparisons": [
                "paired field-specific agreement between prompt arms",
                "paired marginal prevalence difference in percentage points for flags and topics",
                "proposal/values exact-switch matrix",
                "entity-stance exact-switch matrix",
                "overall and Story-era estimates with low-support suppression",
            ],
            "diagnostic_thresholds": {
                "overall_absolute_prevalence_shift_pp": 2.0,
                "story_era_absolute_prevalence_shift_pp": 5.0,
                "paired_kappa_below": 0.80,
                "interpretation": "diagnostic only; not evidence that either arm is correct",
            },
            "cost_estimation": cost,
            "hard_max_cost_usd": None,
            "approver": None,
            "execution_gates": {
                "source_hashes": "passed",
                "sample_hash": "passed",
                "prompt_hashes": "passed",
                "prompt_only_difference": "passed",
                "runtime_model_assigned": "blocked_unset",
                "current_prices_recorded": "blocked_unset",
                "token_projection_recorded": "blocked_unset",
                "retry_policy_frozen": "blocked_unset",
                "truncation_recovery_dry_run": "blocked_unset",
                "written_cost_approval": "blocked_unset",
            },
        },
        "frozen_annotation_runs": list(inputs.frozen_annotation_runs),
        "source_inventory": list(inputs.source_inventory),
        "artifacts": artifacts,
        "public_artifacts": list(PUBLIC_FILES),
        "restricted_artifacts": list(RESTRICTED_FILES),
        "restriction_contract": {
            "public_projection_contains_identifying_crosswalk": False,
            "selection_masters": "restricted_internal",
            "coder_assignments": "restricted_coder_only",
            "invariance_request_plan": "restricted_pre_execution",
            "rule": (
                "never publish a selected semantic key or study-item crosswalk beside "
                "coder-facing evidence"
            ),
        },
        "validation_results": {
            "unique_semantic_keys": "passed",
            "era_and_arm_quotas": "passed",
            "genre_fallback": "passed",
            "probability_and_weight_statuses": "passed_no_weights_for_purposive_or_fallback_rows",
            "coder_blinding": "passed",
            "prompt_only_difference": "passed",
            "source_hashes": "passed",
            "no_network_or_execution": "passed",
        },
    }
    manifest["metadata_sha256"] = _metadata_hash(manifest)
    return manifest


def build_protocol_bundle(
    *, inputs: ProtocolInputs | None = None, data_dir: Path = DATA_DIR, repo_root: Path = REPO_ROOT
) -> ValidationProtocolBundle:
    """Build the complete preregistered bundle in memory without external work."""
    inputs = inputs or load_inputs(data_dir=data_dir, repo_root=repo_root)
    master, paired = _prepare_population(inputs)
    paragraph_sample, paragraph_cells, _ = _paragraph_selection(master, paired)
    speech_sample, speech_cells = _select_speeches(master, inputs)
    paragraph_tasks = _paragraph_tasks(paragraph_sample, master)
    speech_tasks = _speech_tasks(speech_sample, inputs)
    current_prompt, hidden_prompt = build_prompt_pair()
    current_prompt_bytes = _json_bytes(current_prompt, pretty=True)
    hidden_prompt_bytes = _json_bytes(hidden_prompt, pretty=True)
    request_plan = _request_plan(
        paragraph_sample,
        _sha256_bytes(current_prompt_bytes),
        _sha256_bytes(hidden_prompt_bytes),
    )
    artifact_files: dict[str, bytes] = {
        PARAGRAPH_SAMPLE_NAME: _csv_bytes(paragraph_sample),
        PARAGRAPH_CELLS_NAME: _csv_bytes(paragraph_cells),
        PARAGRAPH_TASKS_NAME: _jsonl_bytes(paragraph_tasks),
        SPEECH_SAMPLE_NAME: _csv_bytes(speech_sample),
        SPEECH_CELLS_NAME: _csv_bytes(speech_cells),
        SPEECH_TASKS_NAME: _jsonl_bytes(speech_tasks),
        REQUEST_PLAN_NAME: _csv_bytes(request_plan),
        CURRENT_PROMPT_NAME: current_prompt_bytes,
        HIDDEN_PROMPT_NAME: hidden_prompt_bytes,
    }
    manifest = _manifest(
        inputs=inputs,
        paragraph_sample=paragraph_sample,
        paragraph_cells=paragraph_cells,
        paragraph_tasks=paragraph_tasks,
        speech_sample=speech_sample,
        speech_cells=speech_cells,
        speech_tasks=speech_tasks,
        request_plan=request_plan,
        current_prompt_bytes=current_prompt_bytes,
        hidden_prompt_bytes=hidden_prompt_bytes,
        artifact_files=artifact_files,
    )
    files = {**artifact_files, MANIFEST_NAME: _json_bytes(manifest, pretty=True)}
    bundle = ValidationProtocolBundle(
        paragraph_sample=paragraph_sample,
        paragraph_cells=paragraph_cells,
        paragraph_tasks=paragraph_tasks,
        speech_sample=speech_sample,
        speech_cells=speech_cells,
        speech_tasks=speech_tasks,
        request_plan=request_plan,
        current_prompt=current_prompt,
        hidden_prompt=hidden_prompt,
        manifest=manifest,
        files=files,
    )
    validate_bundle(bundle)
    return bundle


def _validate_probabilities(frame: pd.DataFrame, label: str) -> None:
    _require_columns(
        frame,
        [
            "selection_probability", "sampling_weight",
            "conditional_stage_probability", "conditional_stage_weight",
            "probability_basis", "weight_status", "selection_score",
        ],
        label,
    )
    scores = pd.to_numeric(frame["selection_score"], errors="coerce")
    if scores.isna().any() or not scores.map(math.isfinite).all():
        raise ValidationProtocolError(f"{label} has non-finite selection_score")
    allowed = {
        "design_weight_available",
        "not_applicable_purposive_enrichment",
        "not_available_sequential_fallback",
    }
    if not set(frame["weight_status"]).issubset(allowed):
        raise ValidationProtocolError(f"{label} has an unknown weight status")
    weighted = frame["weight_status"].eq("design_weight_available")
    unweighted = ~weighted
    if frame.loc[unweighted, ["selection_probability", "sampling_weight"]].notna().any().any():
        raise ValidationProtocolError(
            f"{label} assigns a population weight to purposive/fallback rows"
        )
    probability = pd.to_numeric(
        frame.loc[weighted, "selection_probability"], errors="coerce"
    )
    weight = pd.to_numeric(frame.loc[weighted, "sampling_weight"], errors="coerce")
    if (
        probability.isna().any()
        or weight.isna().any()
        or not probability.map(math.isfinite).all()
        or not weight.map(math.isfinite).all()
        or not probability.between(0, 1, inclusive="both").all()
        or (probability <= 0).any()
        or (weight <= 0).any()
    ):
        raise ValidationProtocolError(f"{label} has an invalid design probability or weight")
    if not all(
        math.isclose(float(left), 1.0 / float(right), rel_tol=1e-12, abs_tol=1e-10)
        for left, right in zip(weight, probability, strict=True)
    ):
        raise ValidationProtocolError(f"{label} weights are not inverse probabilities")

    fallback = frame["weight_status"].eq("not_available_sequential_fallback")
    conditional_probability = pd.to_numeric(
        frame.loc[fallback, "conditional_stage_probability"], errors="coerce"
    )
    conditional_weight = pd.to_numeric(
        frame.loc[fallback, "conditional_stage_weight"], errors="coerce"
    )
    if (
        conditional_probability.isna().any()
        or conditional_weight.isna().any()
        or not conditional_probability.between(0, 1, inclusive="both").all()
        or (conditional_probability <= 0).any()
        or (conditional_weight <= 0).any()
        or not all(
            math.isclose(float(left), 1.0 / float(right), rel_tol=1e-12, abs_tol=1e-10)
            for left, right in zip(
                conditional_weight, conditional_probability, strict=True
            )
        )
    ):
        raise ValidationProtocolError(f"{label} has invalid conditional fallback receipts")
    purposive = frame["weight_status"].eq("not_applicable_purposive_enrichment")
    if frame.loc[
        purposive, ["conditional_stage_probability", "conditional_stage_weight"]
    ].notna().any().any():
        raise ValidationProtocolError(f"{label} gives purposive rows a conditional weight")


def _validate_blinded_tasks(
    paragraph_tasks: Sequence[Mapping[str, Any]],
    speech_tasks: Sequence[Mapping[str, Any]],
    paragraph_sample: pd.DataFrame,
    speech_sample: pd.DataFrame,
) -> None:
    if len(paragraph_tasks) != 360 or len(speech_tasks) != 90:
        raise ValidationProtocolError("blinded task counts do not match selected populations")
    paragraph_ids = set(paragraph_sample["study_item_id"])
    speech_ids = set(speech_sample["study_item_id"])
    observed_paragraph_ids: list[str] = []
    for row in paragraph_tasks:
        if set(row) != _BLINDED_PARAGRAPH_FIELDS:
            raise ValidationProtocolError("paragraph coder task exposes an unapproved field")
        if set(row) & _PROHIBITED_BLINDED_FIELDS:
            raise ValidationProtocolError("paragraph coder task violates blinding")
        if not str(row["paragraph_text"]).strip() or not str(row["decade"]).endswith("s"):
            raise ValidationProtocolError("paragraph coder task has malformed evidence")
        observed_paragraph_ids.append(str(row["item_id"]))
    for row in speech_tasks:
        if set(row) != _BLINDED_SPEECH_FIELDS:
            raise ValidationProtocolError("speech coder task exposes an unapproved field")
        if set(row) & _PROHIBITED_BLINDED_FIELDS:
            raise ValidationProtocolError("speech coder task violates blinding")
        paragraphs = row["opening_paragraphs"]
        if not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 5:
            raise ValidationProtocolError("speech coder task has malformed opening paragraphs")
    if set(observed_paragraph_ids) != paragraph_ids or len(observed_paragraph_ids) != len(set(observed_paragraph_ids)):
        raise ValidationProtocolError("paragraph blinded-item mapping is not one-to-one")
    observed_speech_ids = [str(row["item_id"]) for row in speech_tasks]
    if set(observed_speech_ids) != speech_ids or len(observed_speech_ids) != len(set(observed_speech_ids)):
        raise ValidationProtocolError("speech blinded-item mapping is not one-to-one")


def _validate_sample_frames(
    paragraphs: pd.DataFrame,
    speeches: pd.DataFrame,
    paragraph_cells: pd.DataFrame,
    speech_cells: pd.DataFrame,
) -> None:
    """Validate quotas and fallback receipts independently of build-time state."""
    _require_unique(paragraphs, PARAGRAPH_KEY, "paragraph sample")
    _require_unique(paragraphs, ["study_item_id"], "paragraph sample item ids")
    _require_unique(speeches, ["doc_name"], "speech sample")
    _require_unique(speeches, ["study_item_id"], "speech sample item ids")
    if len(paragraphs) != 360 or len(speeches) != 90:
        raise ValidationProtocolError("validation sample counts drifted")

    expected_eras = [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
    if list(paragraphs.sort_values("story_era_order")["story_era_key"].drop_duplicates()) != expected_eras:
        raise ValidationProtocolError("paragraph sample does not use governed era order")
    if list(speeches.sort_values("story_era_order")["story_era_key"].drop_duplicates()) != expected_eras:
        raise ValidationProtocolError("speech sample does not use governed era order")
    if not paragraphs.groupby("story_era_key").size().eq(PARAGRAPHS_PER_ERA).all():
        raise ValidationProtocolError("paragraph sample is not exactly 40 per Story era")
    arm_counts = paragraphs.groupby(["story_era_key", "selection_arm"]).size().unstack(fill_value=0)
    expected_arms = {
        "genre-balanced-base": BASE_PER_ERA,
        "rare-positive": RARE_POSITIVE_PER_ERA,
        "high-disagreement": HIGH_DISAGREEMENT_PER_ERA,
    }
    if set(arm_counts.columns) != set(expected_arms):
        raise ValidationProtocolError("paragraph selection arms drifted")
    for arm, count in expected_arms.items():
        if not arm_counts[arm].eq(count).all():
            raise ValidationProtocolError(f"paragraph arm quota drifted: {arm}")
    if not speeches.groupby("story_era_key").size().eq(SPEECHES_PER_ERA).all():
        raise ValidationProtocolError("factual sample is not exactly 10 speeches per Story era")

    cell_columns = [
        "schema_version", "study", "story_era_key", "story_era_order",
        "selection_arm", "allocation_cell", "candidate_units",
        "candidate_documents", "target_units", "selected_first_pass",
        "selected_fallback", "selected_total",
    ]
    _require_columns(paragraph_cells, cell_columns, "paragraph selection cells")
    _require_columns(speech_cells, cell_columns, "speech selection cells")
    for label, cells in (("paragraph", paragraph_cells), ("speech", speech_cells)):
        if set(cells["schema_version"]) != {SELECTION_CELL_SCHEMA}:
            raise ValidationProtocolError(f"{label} selection-cell schema drift")
        numeric = cells[[
            "candidate_units", "candidate_documents", "target_units",
            "selected_first_pass", "selected_fallback", "selected_total",
        ]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or (numeric < 0).any().any():
            raise ValidationProtocolError(f"{label} selection cells contain invalid counts")
        if not (
            numeric["selected_first_pass"] + numeric["selected_fallback"]
        ).eq(numeric["selected_total"]).all():
            raise ValidationProtocolError(f"{label} selection-cell totals do not reconcile")
    for era_key in expected_eras:
        paragraph_era = paragraph_cells.loc[paragraph_cells.story_era_key.eq(era_key)]
        high = paragraph_era.loc[paragraph_era.selection_arm.eq("high-disagreement")]
        rare = paragraph_era.loc[paragraph_era.selection_arm.eq("rare-positive")]
        base = paragraph_era.loc[paragraph_era.selection_arm.eq("genre-balanced-base")]
        if len(high) != 1 or int(high.selected_total.iloc[0]) != 10:
            raise ValidationProtocolError(f"{era_key}: high-disagreement cell drift")
        if len(rare) != 1 or int(rare.selected_total.iloc[0]) != 10:
            raise ValidationProtocolError(f"{era_key}: rare-positive cell drift")
        if int(base.selected_total.sum()) != 20:
            raise ValidationProtocolError(f"{era_key}: base allocation cells do not sum to 20")
        speech_era = speech_cells.loc[speech_cells.story_era_key.eq(era_key)]
        if int(speech_era.selected_total.sum()) != 10:
            raise ValidationProtocolError(f"{era_key}: factual allocation cells do not sum to 10")


def validate_bundle(bundle: ValidationProtocolBundle) -> None:
    """Validate every in-memory population, byte, prompt, and blocked gate."""
    paragraphs = bundle.paragraph_sample
    speeches = bundle.speech_sample
    _validate_sample_frames(
        paragraphs, speeches, bundle.paragraph_cells, bundle.speech_cells
    )
    _validate_probabilities(paragraphs, "paragraph sample")
    _validate_probabilities(speeches.assign(selection_score=0.0), "speech sample")
    _validate_blinded_tasks(
        bundle.paragraph_tasks, bundle.speech_tasks, paragraphs, speeches
    )
    _validate_prompt_pair(bundle.current_prompt, bundle.hidden_prompt)

    if len(bundle.request_plan) != 360 or not bundle.request_plan["presentation_order"].is_unique:
        raise ValidationProtocolError("invariance request partition is not one-to-one")
    if set(bundle.request_plan["study_item_id"]) != set(paragraphs["study_item_id"]):
        raise ValidationProtocolError("invariance request plan does not reuse the 360-key sample")
    if set(bundle.request_plan["runtime_model"]) != {"UNSET"}:
        raise ValidationProtocolError("dry-run request plan assigned a runtime model")

    manifest = bundle.manifest
    if manifest.get("metadata_sha256") != _metadata_hash(manifest):
        raise ValidationProtocolError("manifest metadata hash mismatch")
    if manifest.get("execution_authorized") is not False:
        raise ValidationProtocolError("validation protocol must remain execution-blocked")
    invariance = manifest.get("measurement_invariance_study", {})
    if invariance.get("runtime_model") is not None or invariance.get("approver") is not None:
        raise ValidationProtocolError("measurement-invariance runtime authority is not unset")
    cost = invariance.get("cost_estimation", {})
    if cost.get("estimated_cost_usd") is not None or not str(cost.get("status", "")).startswith("blocked_"):
        raise ValidationProtocolError("cost receipt is not blocked on unset inputs")
    if set(bundle.files) != set(GOVERNED_FILES):
        raise ValidationProtocolError("protocol artifact inventory drifted")
    for filename, receipt in manifest["artifacts"].items():
        payload = bundle.files.get(filename)
        if payload is None or receipt["sha256"] != _sha256_bytes(payload) or receipt["bytes"] != len(payload):
            raise ValidationProtocolError(f"artifact receipt mismatch: {filename}")
    if bundle.files[MANIFEST_NAME] != _json_bytes(manifest, pretty=True):
        raise ValidationProtocolError("manifest bytes are not canonical")
    _validate_cross_artifact_blinding(bundle.files, paragraphs, speeches)


def _resolve_source_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def validate_source_hashes(manifest: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    for receipt in manifest.get("source_inventory", []):
        path = _resolve_source_path(str(receipt["path"]), Path(repo_root))
        if not path.is_file():
            raise ValidationProtocolError(f"protocol source is missing: {path}")
        if path.stat().st_size != int(receipt["bytes"]) or _sha256_file(path) != receipt["sha256"]:
            raise ValidationProtocolError(f"protocol source hash drift: {receipt['path']}")


def _read_jsonl(payload: bytes) -> tuple[Mapping[str, Any], ...]:
    return tuple(json.loads(line) for line in payload.decode("utf-8").splitlines() if line)


def _read_csv(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


def validate_publication(
    output_dir: Path = OUTPUT_DIR, *, repo_root: Path = REPO_ROOT,
    validate_current_sources: bool = True,
) -> Mapping[str, Any]:
    """Validate a persisted bundle and, by default, refuse stale source hashes."""
    output = Path(output_dir)
    missing = [name for name in GOVERNED_FILES if not (output / name).is_file()]
    if missing:
        raise ValidationProtocolError(f"protocol publication is missing files: {missing}")
    actual_files = {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    }
    if actual_files != set(GOVERNED_FILES):
        raise ValidationProtocolError(
            f"protocol publication inventory drift: expected={sorted(GOVERNED_FILES)}, "
            f"observed={sorted(actual_files)}"
        )
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValidationProtocolError("protocol manifest schema drift")
    if manifest.get("metadata_sha256") != _metadata_hash(manifest):
        raise ValidationProtocolError("persisted protocol metadata hash mismatch")
    if set(manifest.get("artifacts", {})) != set(GOVERNED_FILES) - {MANIFEST_NAME}:
        raise ValidationProtocolError("persisted protocol artifact inventory drift")
    if set(manifest.get("public_artifacts", [])) != set(PUBLIC_FILES):
        raise ValidationProtocolError("persisted public protocol inventory drift")
    if set(manifest.get("restricted_artifacts", [])) != set(RESTRICTED_FILES):
        raise ValidationProtocolError("persisted restricted protocol inventory drift")
    for filename, receipt in manifest.get("artifacts", {}).items():
        path = output / filename
        if not path.is_file() or path.stat().st_size != receipt["bytes"] or _sha256_file(path) != receipt["sha256"]:
            raise ValidationProtocolError(f"persisted artifact hash mismatch: {filename}")
    if validate_current_sources:
        validate_source_hashes(manifest, repo_root=repo_root)

    paragraph_payload = (output / PARAGRAPH_SAMPLE_NAME).read_bytes()
    speech_payload = (output / SPEECH_SAMPLE_NAME).read_bytes()
    paragraph_sample = _read_csv(paragraph_payload)
    speech_sample = _read_csv(speech_payload)
    paragraph_cells = _read_csv((output / PARAGRAPH_CELLS_NAME).read_bytes())
    speech_cells = _read_csv((output / SPEECH_CELLS_NAME).read_bytes())
    request_plan = _read_csv((output / REQUEST_PLAN_NAME).read_bytes())
    current = json.loads((output / CURRENT_PROMPT_NAME).read_text(encoding="utf-8"))
    hidden = json.loads((output / HIDDEN_PROMPT_NAME).read_text(encoding="utf-8"))
    _validate_sample_frames(
        paragraph_sample, speech_sample, paragraph_cells, speech_cells
    )
    _validate_prompt_pair(current, hidden)
    _validate_probabilities(paragraph_sample, "persisted paragraph sample")
    _validate_probabilities(speech_sample.assign(selection_score=0.0), "persisted speech sample")
    _validate_blinded_tasks(
        _read_jsonl((output / PARAGRAPH_TASKS_NAME).read_bytes()),
        _read_jsonl((output / SPEECH_TASKS_NAME).read_bytes()),
        paragraph_sample,
        speech_sample,
    )
    if _semantic_sample_hash(paragraph_sample) != manifest["selection"]["paragraph_sample_sha256"]:
        raise ValidationProtocolError("persisted paragraph sample semantic hash mismatch")
    if _paragraph_key_hash(paragraph_sample) != manifest["selection"]["paragraph_key_set_sha256"]:
        raise ValidationProtocolError("persisted paragraph key-set hash mismatch")
    if _speech_key_hash(speech_sample) != manifest["selection"]["factual_speech_key_set_sha256"]:
        raise ValidationProtocolError("persisted factual key-set hash mismatch")
    prompt_pair = manifest["measurement_invariance_study"]["prompt_pair"]
    if (
        prompt_pair["current_context_sha256"] != _sha256_file(output / CURRENT_PROMPT_NAME)
        or prompt_pair["decade_hidden_sha256"] != _sha256_file(output / HIDDEN_PROMPT_NAME)
    ):
        raise ValidationProtocolError("persisted prompt-pair hash mismatch")
    if (
        len(request_plan) != 360
        or not request_plan["presentation_order"].is_unique
        or set(request_plan["study_item_id"]) != set(paragraph_sample["study_item_id"])
        or set(request_plan["runtime_model"]) != {"UNSET"}
    ):
        raise ValidationProtocolError("persisted invariance request plan drift")
    invariance = manifest.get("measurement_invariance_study", {})
    if manifest.get("execution_authorized") is not False or invariance.get("runtime_model") is not None:
        raise ValidationProtocolError("persisted protocol no longer blocks execution")
    return manifest


def _validate_cross_artifact_blinding(
    files: Mapping[str, bytes], paragraph_sample: pd.DataFrame, speech_sample: pd.DataFrame
) -> None:
    """Prove that the public projection contains no selected key or item-id crosswalk."""
    identifying_values = {
        *paragraph_sample["doc_name"].astype(str),
        *paragraph_sample["study_item_id"].astype(str),
        *speech_sample["doc_name"].astype(str),
        *speech_sample["study_item_id"].astype(str),
    }
    for filename in PUBLIC_FILES:
        payload = files[filename].decode("utf-8")
        leaked = next((value for value in identifying_values if value in payload), None)
        if leaked is not None:
            raise ValidationProtocolError(
                f"public protocol artifact {filename} leaks a selected identifier"
            )


def validate_public_projection(output_dir: Path) -> Mapping[str, Any]:
    """Validate the aggregate/hash-only files safe for a public site."""
    output = Path(output_dir)
    actual = {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    }
    if actual != set(PUBLIC_FILES):
        raise ValidationProtocolError(
            f"public protocol projection inventory drift: {sorted(actual)}"
        )
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("metadata_sha256") != _metadata_hash(manifest):
        raise ValidationProtocolError("public protocol manifest metadata hash mismatch")
    if set(manifest.get("public_artifacts", [])) != set(PUBLIC_FILES):
        raise ValidationProtocolError("public projection is not declared by its manifest")
    for filename in set(PUBLIC_FILES) - {MANIFEST_NAME}:
        receipt = manifest["artifacts"][filename]
        path = output / filename
        if path.stat().st_size != receipt["bytes"] or _sha256_file(path) != receipt["sha256"]:
            raise ValidationProtocolError(f"public protocol artifact hash mismatch: {filename}")
    # The public subset has no selected key table by construction; reject the
    # identifying column names that would create an accidental crosswalk.
    for filename in (PARAGRAPH_CELLS_NAME, SPEECH_CELLS_NAME):
        columns = set(pd.read_csv(output / filename, nrows=0).columns)
        if columns & {"doc_name", "para_idx", "study_item_id", "paragraph_text"}:
            raise ValidationProtocolError(f"public aggregate table exposes identifying columns: {filename}")
    return manifest


def _assert_safe_output_path(output: Path) -> None:
    resolved = output.resolve()
    prohibited = {Path("/").resolve(), Path.home().resolve(), REPO_ROOT.resolve(), DATA_DIR.resolve()}
    if resolved in prohibited:
        raise ValidationProtocolError(f"refusing broad protocol output path: {resolved}")


def _publish_candidate(
    candidate: Path,
    destination: Path,
    *,
    post_validate,
) -> None:
    """Atomically swap one complete directory and restore it on failed validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(prefix=".validation-protocol-publish-", dir=destination.parent)
    )
    held = transaction / "previous"
    failed = transaction / "failed"
    existed = destination.exists()
    preserve = False
    try:
        if existed:
            os.replace(destination, held)
        try:
            os.replace(candidate, destination)
            post_validate(destination)
        except BaseException:
            try:
                if destination.exists():
                    os.replace(destination, failed)
                if existed:
                    if not held.exists():
                        raise ValidationProtocolError("protocol rollback backup is missing")
                    os.replace(held, destination)
            except BaseException as restore_error:
                preserve = True
                raise ValidationProtocolError(
                    f"protocol rollback failed; recovery data retained at {transaction}"
                ) from restore_error
            raise
    finally:
        if not preserve and transaction.exists():
            shutil.rmtree(transaction)


def write_publication(
    bundle: ValidationProtocolBundle,
    output_dir: Path = OUTPUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Write the complete governed bundle, then atomically publish it.

    This output includes restricted selection masters and coder assignments.  It
    is therefore an internal governed directory, not the public-site projection.
    Use :func:`write_public_projection` for files safe to copy into a public
    build.
    """
    validate_bundle(bundle)
    output = Path(output_dir)
    _assert_safe_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(prefix=".validation-protocol-candidate-", dir=output.parent)
    )
    candidate = transaction / "candidate"
    candidate.mkdir()
    try:
        for filename in GOVERNED_FILES:
            destination = candidate / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.files[filename])
        validate_publication(candidate, repo_root=repo_root)
        _publish_candidate(
            candidate,
            output,
            post_validate=lambda path: validate_publication(path, repo_root=repo_root),
        )
    finally:
        if transaction.exists():
            shutil.rmtree(transaction)
    return [output / filename for filename in GOVERNED_FILES]


def write_public_projection(
    output_dir: Path,
    *,
    governed_dir: Path = OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Atomically publish only the aggregate/hash-only, non-identifying subset.

    The source governed bundle is fully validated first, including its current
    source hashes.  Selected semantic keys, study-item crosswalks, coder-facing
    evidence, and the invariance request plan are never copied.
    """
    governed = Path(governed_dir)
    validate_publication(governed, repo_root=repo_root)
    paragraph_sample = _read_csv((governed / PARAGRAPH_SAMPLE_NAME).read_bytes())
    speech_sample = _read_csv((governed / SPEECH_SAMPLE_NAME).read_bytes())
    files = {name: (governed / name).read_bytes() for name in PUBLIC_FILES}
    _validate_cross_artifact_blinding(files, paragraph_sample, speech_sample)

    output = Path(output_dir)
    _assert_safe_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(prefix=".validation-protocol-public-", dir=output.parent)
    )
    candidate = transaction / "candidate"
    candidate.mkdir()
    try:
        for filename in PUBLIC_FILES:
            destination = candidate / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(files[filename])
        validate_public_projection(candidate)
        _publish_candidate(
            candidate,
            output,
            post_validate=validate_public_projection,
        )
    finally:
        if transaction.exists():
            shutil.rmtree(transaction)
    return [output / filename for filename in PUBLIC_FILES]


def rebuild(
    output_dir: Path = OUTPUT_DIR,
    *,
    data_dir: Path = DATA_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Require two byte-identical builds before publishing the planned studies."""
    first = build_protocol_bundle(data_dir=data_dir, repo_root=repo_root)
    repeat = build_protocol_bundle(data_dir=data_dir, repo_root=repo_root)
    if first.files != repeat.files:
        raise ValidationProtocolError("two validation-protocol builds are not byte-identical")
    return write_publication(first, output_dir, repo_root=repo_root)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rebuild", action="store_true", help="build and atomically publish")
    mode.add_argument("--check", action="store_true", help="validate the published bundle")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.rebuild:
        paths = rebuild(args.out_dir)
        print(f"published {len(paths)} execution-blocked validation-protocol files")
    else:
        manifest = validate_publication(args.out_dir)
        print(
            f"validated {manifest['protocol_version']}: "
            f"{manifest['publication_status']}"
        )


if __name__ == "__main__":
    main()
