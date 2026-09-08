"""Governed, deterministic inputs for the public Data Quality page.

This module is deliberately separate from HTML rendering.  It reads frozen paid
annotation artifacts and deterministic governed layers, validates their semantic
keys and populations, and returns an in-memory bundle.  The only write path is
``write_publication``; it emits public projections atomically and performs no API
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import agreement, attention, era_profiles, invocations
from .corpus import DATA_DIR


CONTRACT_VERSION = "data-quality-v2"
MANIFEST_SCHEMA = "data-quality-public-manifest-v2"
MANIFEST_NAME = "manifest_v2.json"
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260908
CI_LOW = 2.5
CI_HIGH = 97.5
PARAGRAPH_KEY = ["doc_name", "para_idx"]

PUBLIC_TABLES = {
    "population_ledger": "population_ledger.csv",
    "second_model_by_speech": "second_model_by_speech.csv",
    "second_model_by_era": "second_model_by_era.csv",
    "zero_sum_confusion": "zero_sum_confusion.csv",
    "zero_sum_by_era": "zero_sum_by_era.csv",
    "zero_sum_disagreement_excerpts": "zero_sum_disagreement_excerpts.csv",
    "entity_name_agreement": "entity_name_agreement.csv",
    "entity_stance_matches": "entity_stance_matches.csv",
    "entity_case_examples": "entity_case_examples.csv",
    "president_alias_examples": "president_alias_examples.csv",
    "uncertainty_decomposition": "uncertainty_decomposition.csv",
    "president_coverage": "president_coverage.csv",
    "era_coverage": "era_coverage.csv",
    "taxonomy_label_behavior": "taxonomy_label_behavior.csv",
    "agreement_by_field_and_era": "agreement_by_field_and_era.csv",
    "public_chart_treatments": "public_chart_treatments.csv",
}
PUBLIC_KEYS = {
    "population_ledger": ["population_id"],
    "second_model_by_speech": ["doc_name"],
    "second_model_by_era": ["era_key"],
    "zero_sum_confusion": ["primary", "second"],
    "zero_sum_by_era": ["era_key"],
    "zero_sum_disagreement_excerpts": PARAGRAPH_KEY,
    "entity_name_agreement": PARAGRAPH_KEY,
    "entity_stance_matches": [*PARAGRAPH_KEY, "entity_normalized"],
    "entity_case_examples": ["case_id"],
    "president_alias_examples": ["canonical_president", "alias"],
    "uncertainty_decomposition": [
        "surface", "period_kind", "ci_components", "disagreement_status"
    ],
    "president_coverage": ["president"],
    "era_coverage": ["era_key"],
    "taxonomy_label_behavior": ["era_key"],
    "agreement_by_field_and_era": ["field", "metric", "era_key"],
    "public_chart_treatments": ["chart_id"],
}
PUBLIC_COLUMNS = {
    "population_ledger": ["population_id", "label", "unit", "count", "definition"],
    "second_model_by_speech": [
        "doc_name", "president", "year", "era_key", "era_label", "era_order", "title",
        "expected_paragraphs", "completed_paragraphs", "missing_paragraphs",
        "completion_rate",
    ],
    "second_model_by_era": [
        "era_key", "era_label", "era_order", "expected_paragraphs",
        "completed_paragraphs", "missing_paragraphs", "completion_rate",
    ],
    "zero_sum_confusion": ["primary", "second", "n", "row_percent"],
    "zero_sum_by_era": [
        "era_key", "era_label", "era_order", "n", "n_speeches",
        "primary_positive_rate", "second_positive_rate", "kappa",
    ],
    "zero_sum_disagreement_excerpts": [
        "doc_name", "para_idx", "president", "year", "era_key", "era_label",
        "zero_sum_primary", "zero_sum_second", "text",
    ],
    "entity_name_agreement": [
        "doc_name", "para_idx", "jaccard", "entity_bearing", "primary_only",
        "second_only", "matched", "era_key", "era_label", "era_order",
    ],
    "entity_stance_matches": [
        "doc_name", "para_idx", "entity_normalized", "primary_present",
        "second_present", "stance_primary", "stance_second", "stance_agrees",
    ],
    "entity_case_examples": [
        "case_id", "case_kind", "doc_name", "para_idx", "president", "year",
        "era_key", "era_label", "entity_primary", "entity_second",
        "entity_normalized", "stance_primary", "stance_second", "text",
        "interpretation_status",
    ],
    "president_alias_examples": ["canonical_president", "alias"],
    "uncertainty_decomposition": [
        "surface", "period_kind", "ci_components", "disagreement_status", "n_cells",
        "n_resolved_sampling", "n_resolved_combined", "mean_sampling_width_pp",
        "mean_combined_width_pp", "mean_disagreement_half_width_pp",
    ],
    "president_coverage": ["president", "n_speeches", "thin_record"],
    "era_coverage": [
        "era_key", "era_label", "era_order", "n_speeches", "n_presidents", "n_words",
    ],
    "taxonomy_label_behavior": [
        "era_key", "era_label", "era_order", "n_paragraphs", "unlabeled_paragraphs",
        "multi_label_paragraphs", "normalized_assignments", "mean_labels",
    ],
    "agreement_by_field_and_era": [
        "field", "metric", "era_key", "era_label", "era_order", "value", "ci_low",
        "ci_high", "n_units", "support_unit", "n_paragraphs", "n_speeches",
        "prevalence_primary", "prevalence_second", "interval_status",
    ],
    "public_chart_treatments": [
        "chart_id", "claim", "denominator", "treatment", "statistical_status",
        "limitation",
    ],
}
EXPECTED_HEADLINE_POPULATIONS = {
    "sampled": 9_048,
    "paired": 8_570,
    "missing": 478,
    "affected_speeches": 28,
}


class QualityAuditError(RuntimeError):
    """A source, population, unit, or publication contract failed closed."""


@dataclass(frozen=True)
class QualityAuditBundle:
    tables: Mapping[str, pd.DataFrame]
    summary: Mapping[str, Any]
    manifest: Mapping[str, Any]


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _metadata_hash(value: Mapping[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != "metadata_sha256"}
    return _sha256_bytes(_canonical_json(semantic).encode("utf-8"))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise QualityAuditError(f"{label} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    _require_columns(frame, keys, label)
    duplicates = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
    if not duplicates.empty:
        raise QualityAuditError(
            f"{label} has duplicate keys on {list(keys)}: "
            f"{duplicates.head(5).to_dict('records')}"
        )


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, PARAGRAPH_KEY])


def _require_same_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    left_keys = _key_index(left)
    right_keys = _key_index(right)
    if len(left_keys) != len(right_keys) or set(left_keys) != set(right_keys):
        raise QualityAuditError(
            f"{label} key sets diverge: left={len(left_keys):,}, right={len(right_keys):,}"
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityAuditError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise QualityAuditError(f"{label} must be a JSON object: {path}")
    return value


def _era_records() -> list[dict[str, Any]]:
    return [
        {
            "era_key": spec.key,
            "era_label": spec.label,
            "era_start": int(spec.start_year),
            "era_end": int(spec.end_year),
            "era_order": order,
        }
        for order, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)
    ]


def _assign_eras(frame: pd.DataFrame, *, year_column: str = "year") -> pd.DataFrame:
    out = frame.copy()
    out["era_key"] = None
    out["era_label"] = None
    out["era_order"] = pd.NA
    for record in _era_records():
        mask = out[year_column].between(record["era_start"], record["era_end"])
        out.loc[mask, "era_key"] = record["era_key"]
        out.loc[mask, "era_label"] = record["era_label"]
        out.loc[mask, "era_order"] = record["era_order"]
    if out["era_key"].isna().any():
        years = sorted(out.loc[out["era_key"].isna(), year_column].dropna().unique())
        raise QualityAuditError(f"years fall outside the governed Story eras: {years[:10]}")
    out["era_order"] = out["era_order"].astype(int)
    return out


def _ordered_group(frame: pd.DataFrame, columns: Sequence[str]) -> Any:
    return frame.groupby(list(columns), sort=False, observed=True, dropna=False)


def _contingency_stat(counts: np.ndarray, metric: str) -> float:
    return agreement.compute_contingency_metric(counts, metric)


def _set_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    return agreement.jaccard(left, right)


def _bootstrap_weights(doc_names: Sequence[str], token: str, draws: int) -> np.ndarray:
    n_docs = len(doc_names)
    if n_docs < 2 or draws <= 0:
        return np.empty((0, n_docs), dtype=np.int16)
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{token}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.multinomial(n_docs, np.repeat(1.0 / n_docs, n_docs), size=draws)


def _interval(values: Sequence[float], draws: int) -> tuple[float | None, float | None, str]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if draws <= 0:
        return None, None, "not_requested"
    if len(finite) < max(100, math.ceil(draws * 0.9)):
        return None, None, "unavailable_degenerate_resamples"
    low, high = np.percentile(finite, [CI_LOW, CI_HIGH])
    return float(low), float(high), "speech_cluster_bootstrap"


def _categorical_metric(
    frame: pd.DataFrame,
    primary_col: str,
    second_col: str,
    metric: str,
    token: str,
    draws: int,
) -> tuple[float, float | None, float | None, str]:
    clean = frame[["doc_name", primary_col, second_col]].dropna().copy()
    categories = sorted(
        set(clean[primary_col].astype(str)) | set(clean[second_col].astype(str))
    )
    category_index = {value: index for index, value in enumerate(categories)}
    docs = sorted(clean["doc_name"].unique())
    doc_index = {name: index for index, name in enumerate(docs)}
    contributions = np.zeros((len(docs), len(categories), len(categories)), dtype=np.int64)
    for row in clean.itertuples(index=False):
        contributions[
            doc_index[row.doc_name],
            category_index[str(getattr(row, primary_col))],
            category_index[str(getattr(row, second_col))],
        ] += 1
    point = _contingency_stat(contributions.sum(axis=0), metric)
    weights = _bootstrap_weights(docs, token, draws)
    estimates = [
        _contingency_stat(np.tensordot(weight, contributions, axes=(0, 0)), metric)
        for weight in weights
    ]
    low, high, status = _interval(estimates, draws)
    return point, low, high, status


def _mean_metric(
    frame: pd.DataFrame,
    value_col: str,
    token: str,
    draws: int,
) -> tuple[float, float | None, float | None, str]:
    clean = frame[["doc_name", value_col]].dropna()
    by_doc = clean.groupby("doc_name", sort=True)[value_col].agg(["sum", "count"])
    point = float(by_doc["sum"].sum() / by_doc["count"].sum())
    weights = _bootstrap_weights(list(by_doc.index), token, draws)
    numerators = weights @ by_doc["sum"].to_numpy(float)
    denominators = weights @ by_doc["count"].to_numpy(float)
    estimates = np.divide(
        numerators,
        denominators,
        out=np.full(len(weights), np.nan),
        where=denominators > 0,
    )
    low, high, status = _interval(estimates, draws)
    return point, low, high, status


def _ratio_metric(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    token: str,
    draws: int,
) -> tuple[float, float | None, float | None, str]:
    by_doc = frame.groupby("doc_name", sort=True)[[numerator, denominator]].sum()
    total_denominator = float(by_doc[denominator].sum())
    point = (
        float(by_doc[numerator].sum() / total_denominator)
        if total_denominator > 0
        else math.nan
    )
    weights = _bootstrap_weights(list(by_doc.index), token, draws)
    numerators = weights @ by_doc[numerator].to_numpy(float)
    denominators = weights @ by_doc[denominator].to_numpy(float)
    estimates = np.divide(
        numerators,
        denominators,
        out=np.full(len(weights), np.nan),
        where=denominators > 0,
    )
    low, high, status = _interval(estimates, draws)
    return point, low, high, status


def _entity_frames(
    primary_entities: pd.DataFrame,
    second_entities: pd.DataFrame,
    paired_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return paragraph agreement, matched stances, and doc-level ratio counts."""
    for frame, label in (
        (primary_entities, "primary entities"),
        (second_entities, "second entities"),
    ):
        _require_columns(frame, [*PARAGRAPH_KEY, "entity", "stance"], label)
    _require_unique(paired_keys, PARAGRAPH_KEY, "paired paragraph keys")

    def restrict(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        out = frame.merge(
            paired_keys,
            on=PARAGRAPH_KEY,
            how="inner",
            validate="many_to_one",
        ).copy()
        normalized = out["entity"].map(agreement.normalize_entity_name)
        if normalized.eq("").any():
            raise QualityAuditError(f"{label} contains an empty normalized entity name")
        return out

    left = restrict(primary_entities, "primary entities")
    right = restrict(second_entities, "second entities")
    detail = agreement.entity_agreement_details(left, right)
    names_left: dict[tuple[str, int], set[str]] = {}
    names_right: dict[tuple[str, int], set[str]] = {}
    stance_by_paragraph: dict[tuple[str, int], int] = {}
    for row in detail.itertuples(index=False):
        key = (str(row.doc_name), int(row.para_idx))
        if row.primary_present:
            names_left.setdefault(key, set()).add(str(row.entity_normalized))
        if row.second_present:
            names_right.setdefault(key, set()).add(str(row.entity_normalized))
        if row.primary_present and row.second_present and row.stance_agrees:
            stance_by_paragraph[key] = stance_by_paragraph.get(key, 0) + 1

    paragraph_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    for key in paired_keys.sort_values(PARAGRAPH_KEY).itertuples(index=False, name=None):
        doc_name, para_idx = str(key[0]), int(key[1])
        left_names = names_left.get((doc_name, para_idx), set())
        right_names = names_right.get((doc_name, para_idx), set())
        matched = left_names & right_names
        union = left_names | right_names
        stance_matches = stance_by_paragraph.get((doc_name, para_idx), 0)
        paragraph_rows.append(
            {
                "doc_name": doc_name,
                "para_idx": para_idx,
                "jaccard": agreement.jaccard(left_names, right_names),
                "entity_bearing": bool(union),
                "primary_only": len(left_names - right_names),
                "second_only": len(right_names - left_names),
                "matched": len(matched),
            }
        )
        doc_rows.append(
            {
                "doc_name": doc_name,
                "n_union": len(union),
                "n_matched": len(matched),
                "n_stance_agree": stance_matches,
            }
        )

    matched = detail.loc[
        detail["primary_present"] & detail["second_present"]
    ].reset_index(drop=True)
    doc_counts = pd.DataFrame(doc_rows).groupby("doc_name", sort=True).sum().reset_index()
    canonical = agreement.compute_entity_metrics(left, right)
    if (
        canonical["n_union"] != int(doc_counts["n_union"].sum())
        or canonical["n_matched"] != int(doc_counts["n_matched"].sum())
        or not math.isclose(
            canonical["stance_agreement"],
            float(matched["stance_agrees"].mean()),
            abs_tol=1e-12,
        )
    ):
        raise QualityAuditError("entity detail diverges from the canonical agreement helper")
    return pd.DataFrame(paragraph_rows), matched, doc_counts


def _entity_case_examples(
    primary_entities: pd.DataFrame,
    second_entities: pd.DataFrame,
    paired_keys: pd.DataFrame,
    paragraph_context: pd.DataFrame,
) -> pd.DataFrame:
    """Select stable, public examples of matched and unmatched entity extraction.

    The examples are descriptive instrument comparisons, not adjudicated labels.  A
    matched example prefers a pair whose source strings differ but whose conservative
    case/whitespace normalization produces the same semantic key.  The two unmatched
    rows expose each direction symmetrically.
    """
    _require_unique(paired_keys, PARAGRAPH_KEY, "paired paragraph keys")
    _require_columns(
        paragraph_context,
        [*PARAGRAPH_KEY, "president", "year", "era_key", "era_label", "text"],
        "entity case paragraph context",
    )
    _require_unique(paragraph_context, PARAGRAPH_KEY, "entity case paragraph context")

    def collapse(frame: pd.DataFrame, side: str) -> pd.DataFrame:
        _require_columns(frame, [*PARAGRAPH_KEY, "entity", "stance"], f"{side} entities")
        scoped = frame.merge(
            paired_keys,
            on=PARAGRAPH_KEY,
            how="inner",
            validate="many_to_one",
        ).copy()
        scoped["entity_normalized"] = scoped["entity"].map(agreement.normalize_entity_name)
        if scoped["entity_normalized"].eq("").any():
            raise QualityAuditError(
                f"{side} entities contain an empty normalized entity name"
            )

        def joined(values: pd.Series) -> str:
            return " | ".join(sorted({str(value).strip() for value in values}))

        return (
            scoped.groupby([*PARAGRAPH_KEY, "entity_normalized"], sort=True, as_index=False)
            .agg(
                **{
                    f"entity_{side}": ("entity", joined),
                    f"stance_{side}": ("stance", joined),
                }
            )
        )

    primary = collapse(primary_entities, "primary")
    second = collapse(second_entities, "second")
    candidates = primary.merge(
        second,
        on=[*PARAGRAPH_KEY, "entity_normalized"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    candidates["case_kind"] = candidates["_merge"].map(
        {"both": "matched", "left_only": "primary_only", "right_only": "second_only"}
    )
    candidates = candidates.drop(columns="_merge").merge(
        paragraph_context,
        on=PARAGRAPH_KEY,
        validate="many_to_one",
    )
    for column in ("entity_primary", "entity_second", "stance_primary", "stance_second"):
        candidates[column] = candidates[column].fillna("").astype(str)

    matched = candidates.loc[candidates["case_kind"].eq("matched")]
    normalized_variant = matched.loc[
        matched["entity_primary"].ne(matched["entity_second"])
    ]
    if not normalized_variant.empty:
        matched = normalized_variant

    selections: list[pd.Series] = []
    for case_id, case_kind, pool in (
        ("matched_after_normalization", "matched", matched),
        (
            "primary_only_detection",
            "primary_only",
            candidates.loc[candidates["case_kind"].eq("primary_only")],
        ),
        (
            "second_only_detection",
            "second_only",
            candidates.loc[candidates["case_kind"].eq("second_only")],
        ),
    ):
        if pool.empty:
            raise QualityAuditError(f"no candidate exists for entity case {case_id}")
        selected = pool.sort_values(["year", "doc_name", "para_idx", "entity_normalized"]).iloc[0].copy()
        selected["case_id"] = case_id
        selected["case_kind"] = case_kind
        selections.append(selected)

    result = pd.DataFrame(selections)
    result["interpretation_status"] = "illustrative_cross_model_comparison_not_adjudicated"
    columns = [
        "case_id", "case_kind", *PARAGRAPH_KEY, "president", "year", "era_key",
        "era_label", "entity_primary", "entity_second", "entity_normalized",
        "stance_primary", "stance_second", "text", "interpretation_status",
    ]
    return result.loc[:, columns].reset_index(drop=True)


def _agreement_rows(
    paired: pd.DataFrame,
    speech_paired: pd.DataFrame,
    entity_doc_counts: pd.DataFrame,
    entity_paragraphs: pd.DataFrame,
    *,
    draws: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    scopes: list[tuple[str, str, int, pd.DataFrame]] = [
        ("overall", "Overall paired sample", -1, paired)
    ]
    scopes.extend(
        (
            record["era_key"],
            record["era_label"],
            record["era_order"],
            paired.loc[paired["era_key"].eq(record["era_key"])],
        )
        for record in _era_records()
    )

    paragraph_specs = [
        ("party_attack", "cohen_kappa", "party_attack_primary", "party_attack_second"),
        ("enemy_naming", "cohen_kappa", "enemy_naming_primary", "enemy_naming_second"),
        ("zero_sum", "cohen_kappa", "zero_sum_primary", "zero_sum_second"),
        ("proposal_values", "exact_match", "proposal_values_primary", "proposal_values_second"),
    ]
    for era_key, era_label, era_order, scope in scopes:
        n_paragraphs = int(len(scope))
        n_speeches = int(scope["doc_name"].nunique())
        for field, metric, primary_col, second_col in paragraph_specs:
            value, low, high, status = _categorical_metric(
                scope,
                primary_col,
                second_col,
                metric,
                f"{field}:{metric}:{era_key}",
                draws,
            )
            is_flag = field in {"party_attack", "enemy_naming", "zero_sum"}
            rows.append(
                {
                    "field": field,
                    "metric": metric,
                    "era_key": era_key,
                    "era_label": era_label,
                    "era_order": era_order,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                    "n_units": n_paragraphs,
                    "support_unit": "paired_paragraphs",
                    "n_paragraphs": n_paragraphs,
                    "n_speeches": n_speeches,
                    "prevalence_primary": float(scope[primary_col].mean()) if is_flag else None,
                    "prevalence_second": float(scope[second_col].mean()) if is_flag else None,
                    "interval_status": status,
                }
            )
        value, low, high, status = _mean_metric(
            scope, "topic_jaccard", f"topics:jaccard:{era_key}", draws
        )
        rows.append(
            {
                "field": "topics",
                "metric": "jaccard_normalized",
                "era_key": era_key,
                "era_label": era_label,
                "era_order": era_order,
                "value": value,
                "ci_low": low,
                "ci_high": high,
                "n_units": n_paragraphs,
                "support_unit": "paired_paragraphs",
                "n_paragraphs": n_paragraphs,
                "n_speeches": n_speeches,
                "prevalence_primary": None,
                "prevalence_second": None,
                "interval_status": status,
            }
        )

        scope_keys = scope.loc[:, PARAGRAPH_KEY]
        paragraph_entity_scope = entity_paragraphs.merge(
            scope_keys, on=PARAGRAPH_KEY, how="inner", validate="one_to_one"
        )
        entity_docs = entity_doc_counts.loc[
            entity_doc_counts["doc_name"].isin(scope["doc_name"].unique())
        ]
        for metric, numerator, denominator, support_unit in (
            ("name_match_rate", "n_matched", "n_union", "normalized_entity_union"),
            ("stance_agreement", "n_stance_agree", "n_matched", "matched_entities"),
        ):
            value, low, high, status = _ratio_metric(
                entity_docs,
                numerator,
                denominator,
                f"entities:{metric}:{era_key}",
                draws,
            )
            rows.append(
                {
                    "field": "entities",
                    "metric": metric,
                    "era_key": era_key,
                    "era_label": era_label,
                    "era_order": era_order,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                    "n_units": int(entity_docs[denominator].sum()),
                    "support_unit": support_unit,
                    "n_paragraphs": int(len(paragraph_entity_scope)),
                    "n_speeches": n_speeches,
                    "prevalence_primary": None,
                    "prevalence_second": None,
                    "interval_status": status,
                }
            )

    speech_scopes: list[tuple[str, str, int, pd.DataFrame]] = [
        ("overall", "Overall paired sample", -1, speech_paired)
    ]
    speech_scopes.extend(
        (
            record["era_key"],
            record["era_label"],
            record["era_order"],
            speech_paired.loc[speech_paired["era_key"].eq(record["era_key"])],
        )
        for record in _era_records()
    )
    for era_key, era_label, era_order, scope in speech_scopes:
        for field in ("speech_type", "audience", "medium"):
            for metric in ("cohen_kappa", "exact_match"):
                value, low, high, status = _categorical_metric(
                    scope,
                    f"{field}_primary",
                    f"{field}_second",
                    metric,
                    f"{field}:{metric}:{era_key}",
                    draws,
                )
                rows.append(
                    {
                        "field": field,
                        "metric": metric,
                        "era_key": era_key,
                        "era_label": era_label,
                        "era_order": era_order,
                        "value": value,
                        "ci_low": low,
                        "ci_high": high,
                        "n_units": int(len(scope)),
                        "support_unit": "paired_speeches",
                        "n_paragraphs": None,
                        "n_speeches": int(len(scope)),
                        "prevalence_primary": None,
                        "prevalence_second": None,
                        "interval_status": status,
                    }
                )
    result = pd.DataFrame(rows)
    nonfinite = ~np.isfinite(result["value"].astype(float))
    result.loc[nonfinite, ["ci_low", "ci_high"]] = None
    result.loc[nonfinite, "interval_status"] = "unavailable_degenerate_point"
    return result


def _validate_canonical_agreement(
    canonical: pd.DataFrame,
    agreement_table: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, Any]:
    _require_columns(canonical, ["field", "era_label", "metric", "value", "n"], "agreement_v1")
    overall = agreement_table.loc[agreement_table["era_key"].eq("overall")]
    checked: list[str] = []
    for row in canonical.loc[canonical["era_label"].eq("overall")].itertuples(index=False):
        if row.field == "topics":
            raw_values = [
                _set_jaccard(left, right)
                for left, right in zip(
                    paired["topics_primary_raw"], paired["topics_second_raw"], strict=True
                )
            ]
            raw_value, raw_n = float(np.mean(raw_values)), len(raw_values)
            if int(row.n) != raw_n or not math.isclose(float(row.value), raw_value, abs_tol=1e-12):
                raise QualityAuditError("agreement_v1 raw topic Jaccard does not reproduce")
            checked.append("topics:raw_jaccard")
            continue
        if row.metric in {"unmatched_primary_rate", "unmatched_opus_rate"}:
            continue
        metric = "jaccard_normalized" if row.field == "topics" else row.metric
        current = overall.loc[
            overall["field"].eq(row.field) & overall["metric"].eq(metric)
        ]
        if len(current) != 1:
            raise QualityAuditError(
                f"agreement_v1 check could not resolve {row.field}/{row.metric}"
            )
        if not math.isclose(float(current.iloc[0]["value"]), float(row.value), abs_tol=1e-12):
            raise QualityAuditError(
                f"agreement_v1 drift for {row.field}/{row.metric}: "
                f"{current.iloc[0]['value']} != {row.value}"
            )
        checked.append(f"{row.field}:{row.metric}")
    normalized_topic = float(
        overall.loc[
            overall["field"].eq("topics")
            & overall["metric"].eq("jaccard_normalized"),
            "value",
        ].iloc[0]
    )
    canonical_topic = float(
        canonical.loc[
            canonical["era_label"].eq("overall")
            & canonical["field"].eq("topics")
            & canonical["metric"].eq("jaccard"),
            "value",
        ].iloc[0]
    )
    return {
        "status": "passed",
        "checked_metrics": sorted(checked),
        "canonical_topic_jaccard_raw_labels": canonical_topic,
        "published_topic_jaccard_normalized_labels": normalized_topic,
        "topic_difference_reason": "case normalization and within-paragraph deduplication",
    }


def _source_paths(data_dir: Path) -> dict[str, Path]:
    annotation_dir = data_dir / "llm_annotations"
    speaker_dir = data_dir / "speaker_views"
    return {
        "speeches": data_dir / "speeches.parquet",
        "paragraphs": data_dir / "paragraphs.parquet",
        "primary_paragraph_annotations": annotation_dir / "paragraph_annotations.parquet",
        "second_paragraph_annotations": annotation_dir / "paragraph_annotations__opus4-8.parquet",
        "primary_entities": annotation_dir / "paragraph_entities.parquet",
        "second_entities": annotation_dir / "paragraph_entities__opus4-8.parquet",
        "primary_speech_annotations": annotation_dir / "speech_annotations.parquet",
        "second_speech_annotations": annotation_dir / "speech_annotations__opus4-8.parquet",
        "agreement_sample": annotation_dir / "agreement_sample_v1.json",
        "agreement_metrics": annotation_dir / "agreement_v1.parquet",
        "taxonomy": annotation_dir / "taxonomy_v1.json",
        "bands": data_dir / "bands.parquet",
        "bands_metadata": data_dir / "bands_meta.json",
        "speaker_paragraph_view": speaker_dir / "paragraph_view_v1.parquet",
        "speaker_appearances": speaker_dir / "appearances_v1.parquet",
        "speaker_metadata": speaker_dir / "meta_v1.json",
    }


def _build_uncertainty_decomposition(bands: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        bands,
        [
            "surface", "series", "period_kind", "lo_sampling", "hi_sampling", "lo", "hi",
            "ci_components", "disagreement_status", "disagreement_half_width",
        ],
        "bands",
    )
    for low, high in (("lo_sampling", "hi_sampling"), ("lo", "hi")):
        resolved = bands[[low, high]].dropna()
        if not np.isfinite(resolved.to_numpy(dtype=float)).all():
            raise QualityAuditError(f"bands contains non-finite resolved {low}/{high} values")
        invalid = (
            resolved[low].lt(0)
            | resolved[high].gt(1)
            | resolved[low].gt(resolved[high])
        )
        if invalid.any():
            raise QualityAuditError(
                f"bands {low}/{high} must be ordered proportions in [0, 1]; "
                "a percentage-point input would be a unit mismatch"
            )
    disagreement = bands["disagreement_half_width"].dropna()
    if (
        not np.isfinite(disagreement.to_numpy(dtype=float)).all()
        or disagreement.lt(0).any()
        or disagreement.gt(1).any()
    ):
        raise QualityAuditError("disagreement half-widths must be finite proportions in [0, 1]")

    band_widths = bands.assign(
        sampling_width_pp=(bands["hi_sampling"] - bands["lo_sampling"]) * 100.0,
        combined_width_pp=(bands["hi"] - bands["lo"]) * 100.0,
        disagreement_half_width_pp=bands["disagreement_half_width"] * 100.0,
    )
    decomposition = (
        _ordered_group(
            band_widths,
            ["surface", "period_kind", "ci_components", "disagreement_status"],
        )
        .agg(
            n_cells=("series", "size"),
            n_resolved_sampling=("sampling_width_pp", "count"),
            n_resolved_combined=("combined_width_pp", "count"),
            mean_sampling_width_pp=("sampling_width_pp", "mean"),
            mean_combined_width_pp=("combined_width_pp", "mean"),
            mean_disagreement_half_width_pp=("disagreement_half_width_pp", "mean"),
        )
        .reset_index()
    )
    if (
        decomposition[["mean_sampling_width_pp", "mean_combined_width_pp"]]
        .dropna(how="all")
        .lt(0)
        .any()
        .any()
    ):
        raise QualityAuditError("uncertainty widths cannot be negative")
    return decomposition


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records: pandas missing/non-finite scalars become ``None``."""
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for column, value in zip(frame.columns, row, strict=True):
            if isinstance(value, np.generic):
                value = value.item()
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            if missing or (isinstance(value, float) and not math.isfinite(value)):
                value = None
            record[str(column)] = value
        records.append(record)
    return records


def _annotation_run_receipts(data_dir: Path, frames: Sequence[pd.DataFrame]) -> list[dict[str, Any]]:
    run_ids = sorted(
        {
            str(run_id)
            for frame in frames
            for run_id in frame["run_id"].dropna().unique()
        }
    )
    receipts: list[dict[str, Any]] = []
    for run_id in run_ids:
        path = data_dir / "llm_annotations" / "manifests" / f"{run_id}.json"
        manifest = _read_json(path, f"annotation manifest {run_id}")
        receipts.append(
            {
                "run_id": run_id,
                "model": manifest.get("model"),
                "prompt_version": manifest.get("prompt_version"),
                "prompt_hash": manifest.get("prompt_hash"),
                "manifest_sha256": _sha256_file(path),
            }
        )
    return receipts


def build_quality_audit(
    data_dir: Path = DATA_DIR,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> QualityAuditBundle:
    """Build and validate the complete Data Quality projection without writing."""
    data_dir = Path(data_dir)
    paths = _source_paths(data_dir)
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        raise QualityAuditError(f"required quality sources are missing: {missing_paths}")

    speeches = pd.read_parquet(paths["speeches"])
    paragraphs = pd.read_parquet(paths["paragraphs"])
    primary = pd.read_parquet(paths["primary_paragraph_annotations"])
    second = pd.read_parquet(paths["second_paragraph_annotations"])
    primary_entities = pd.read_parquet(paths["primary_entities"])
    second_entities = pd.read_parquet(paths["second_entities"])
    primary_speech = pd.read_parquet(paths["primary_speech_annotations"])
    second_speech = pd.read_parquet(paths["second_speech_annotations"])
    canonical_agreement = pd.read_parquet(paths["agreement_metrics"])
    bands = pd.read_parquet(paths["bands"])
    speaker_view = pd.read_parquet(paths["speaker_paragraph_view"])
    appearances = pd.read_parquet(paths["speaker_appearances"])
    sample = _read_json(paths["agreement_sample"], "agreement sample")
    taxonomy = _read_json(paths["taxonomy"], "taxonomy")
    bands_meta = _read_json(paths["bands_metadata"], "bands metadata")
    speaker_meta = _read_json(paths["speaker_metadata"], "speaker metadata")

    _require_unique(speeches, ["doc_name"], "speeches")
    _require_unique(paragraphs, PARAGRAPH_KEY, "paragraphs")
    _require_unique(primary, PARAGRAPH_KEY, "primary paragraph annotations")
    _require_unique(second, PARAGRAPH_KEY, "second paragraph annotations")
    _require_unique(primary_speech, ["doc_name"], "primary speech annotations")
    _require_unique(second_speech, ["doc_name"], "second speech annotations")
    _require_unique(speaker_view, PARAGRAPH_KEY, "speaker paragraph view")
    _require_unique(appearances, ["doc_name", "attributed_speaker_profile_id"], "speaker appearances")
    _require_same_keys(paragraphs, primary, "corpus/primary annotation")

    if _sha256_file(paths["speaker_paragraph_view"]) != speaker_meta["artifacts"][paths["speaker_paragraph_view"].name]:
        raise QualityAuditError("speaker paragraph view hash differs from governed metadata")
    if _sha256_file(paths["speaker_appearances"]) != speaker_meta["artifacts"][paths["speaker_appearances"].name]:
        raise QualityAuditError("speaker appearances hash differs from governed metadata")

    sampled_docs = set(map(str, sample.get("doc_names", [])))
    if len(sampled_docs) != int(sample.get("n_sampled", -1)):
        raise QualityAuditError("agreement sample document count is inconsistent")
    expected = paragraphs.loc[paragraphs["doc_name"].isin(sampled_docs), PARAGRAPH_KEY].copy()
    second_keys = second.loc[:, PARAGRAPH_KEY]
    unexpected_second = set(_key_index(second_keys)) - set(_key_index(expected))
    if unexpected_second:
        raise QualityAuditError(
            f"second annotations contain {len(unexpected_second)} keys outside the sampled population"
        )
    completion = expected.merge(
        second_keys.assign(completed=True),
        on=PARAGRAPH_KEY,
        how="left",
        validate="one_to_one",
    )
    completion["completed"] = completion["completed"].fillna(False).astype(bool)
    completion = completion.merge(
        speeches[["doc_name", "president", "year", "title"]],
        on="doc_name",
        validate="many_to_one",
    )
    completion = _assign_eras(completion)

    missing_by_speech = (
        _ordered_group(
            completion,
            ["doc_name", "president", "year", "era_key", "era_label", "era_order", "title"],
        )["completed"]
        .agg(expected_paragraphs="size", completed_paragraphs="sum")
        .reset_index()
        .sort_values(["year", "doc_name"])
        .reset_index(drop=True)
    )
    missing_by_speech["completed_paragraphs"] = missing_by_speech["completed_paragraphs"].astype(int)
    missing_by_speech["missing_paragraphs"] = (
        missing_by_speech["expected_paragraphs"] - missing_by_speech["completed_paragraphs"]
    )
    missing_by_speech["completion_rate"] = (
        missing_by_speech["completed_paragraphs"] / missing_by_speech["expected_paragraphs"]
    )
    missing_by_era = (
        _ordered_group(completion, ["era_key", "era_label", "era_order"])["completed"]
        .agg(expected_paragraphs="size", completed_paragraphs="sum")
        .reset_index()
        .sort_values("era_order")
        .reset_index(drop=True)
    )
    missing_by_era["completed_paragraphs"] = missing_by_era["completed_paragraphs"].astype(int)
    missing_by_era["missing_paragraphs"] = (
        missing_by_era["expected_paragraphs"] - missing_by_era["completed_paragraphs"]
    )
    missing_by_era["completion_rate"] = (
        missing_by_era["completed_paragraphs"] / missing_by_era["expected_paragraphs"]
    )

    paired = primary.merge(
        second,
        on=PARAGRAPH_KEY,
        how="inner",
        suffixes=("_primary", "_second"),
        validate="one_to_one",
    ).merge(
        speeches[["doc_name", "year", "president"]],
        on="doc_name",
        validate="many_to_one",
    )
    paired = _assign_eras(paired)
    label_map = attention.canonical_label_map(taxonomy)
    paired["topics_primary_raw"] = paired["topics_primary"]
    paired["topics_second_raw"] = paired["topics_second"]
    paired["topics_primary"] = paired["topics_primary"].map(
        lambda value: attention.normalize_topics(value, label_map)
    )
    paired["topics_second"] = paired["topics_second"].map(
        lambda value: attention.normalize_topics(value, label_map)
    )
    paired["topic_jaccard"] = [
        _set_jaccard(left, right)
        for left, right in zip(paired["topics_primary"], paired["topics_second"], strict=True)
    ]

    confusion = pd.crosstab(
        paired["zero_sum_primary"],
        paired["zero_sum_second"],
        rownames=["primary"],
        colnames=["second"],
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)
    confusion_long = confusion.stack(future_stack=True).rename("n").reset_index()
    row_totals = confusion_long.groupby("primary")["n"].transform("sum")
    confusion_long["row_percent"] = confusion_long["n"] / row_totals * 100.0

    zero_sum_rows: list[dict[str, Any]] = []
    for era_key, era_label, era_order, group in [
        ("overall", "Overall paired sample", -1, paired),
        *[
            (
                record["era_key"],
                record["era_label"],
                record["era_order"],
                paired.loc[paired["era_key"].eq(record["era_key"])],
            )
            for record in _era_records()
        ],
    ]:
        scoped_confusion = pd.crosstab(
            group["zero_sum_primary"], group["zero_sum_second"]
        ).reindex(index=[False, True], columns=[False, True], fill_value=0)
        value, n = _contingency_stat(scoped_confusion.to_numpy(), "cohen_kappa"), len(group)
        zero_sum_rows.append(
            {
                "era_key": era_key,
                "era_label": era_label,
                "era_order": era_order,
                "n": n,
                "n_speeches": int(group["doc_name"].nunique()),
                "primary_positive_rate": float(group["zero_sum_primary"].mean()),
                "second_positive_rate": float(group["zero_sum_second"].mean()),
                "kappa": value,
            }
        )
    zero_sum_by_era = pd.DataFrame(zero_sum_rows)
    paragraph_text = paragraphs.loc[:, [*PARAGRAPH_KEY, "text"]]
    disagreements = paired.loc[
        paired["zero_sum_primary"].ne(paired["zero_sum_second"]),
        [*PARAGRAPH_KEY, "president", "year", "era_key", "era_label", "zero_sum_primary", "zero_sum_second"],
    ].merge(paragraph_text, on=PARAGRAPH_KEY, validate="one_to_one")
    disagreements = disagreements.sort_values(
        ["zero_sum_primary", "zero_sum_second", "year", "doc_name", "para_idx"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    zero_sum_cases = pd.concat(
        [
            disagreements.loc[
                disagreements["zero_sum_primary"] & ~disagreements["zero_sum_second"]
            ].head(1),
            disagreements.loc[
                ~disagreements["zero_sum_primary"] & disagreements["zero_sum_second"]
            ].head(1),
        ],
        ignore_index=True,
    )
    if len(zero_sum_cases) != 2:
        raise QualityAuditError(
            "zero-sum case study requires both primary-only and second-only examples"
        )

    paired_keys = paired.loc[:, PARAGRAPH_KEY].copy()
    entity_audit, matched_entities, entity_doc_counts = _entity_frames(
        primary_entities, second_entities, paired_keys
    )
    entity_audit = entity_audit.merge(
        paired[[*PARAGRAPH_KEY, "era_key", "era_label", "era_order"]],
        on=PARAGRAPH_KEY,
        validate="one_to_one",
    )
    entity_cases = _entity_case_examples(
        primary_entities,
        second_entities,
        paired_keys,
        paired[[*PARAGRAPH_KEY, "president", "year", "era_key", "era_label"]].merge(
            paragraph_text,
            on=PARAGRAPH_KEY,
            validate="one_to_one",
        ),
    )

    speech_paired = primary_speech.loc[primary_speech["doc_name"].isin(sampled_docs)].merge(
        second_speech.loc[second_speech["doc_name"].isin(sampled_docs)],
        on="doc_name",
        how="inner",
        suffixes=("_primary", "_second"),
        validate="one_to_one",
    ).merge(speeches[["doc_name", "year"]], on="doc_name", validate="one_to_one")
    speech_paired = _assign_eras(speech_paired)
    agreement_by_field = _agreement_rows(
        paired,
        speech_paired,
        entity_doc_counts,
        entity_audit,
        draws=bootstrap_draws,
    )
    canonical_checks = _validate_canonical_agreement(
        canonical_agreement, agreement_by_field, paired
    )

    decomposition = _build_uncertainty_decomposition(bands)

    thin = speeches.groupby("president", sort=True).size().rename("n_speeches").reset_index()
    thin["thin_record"] = thin["n_speeches"].lt(5)
    speech_era = _assign_eras(speeches)
    era_coverage = (
        _ordered_group(speech_era, ["era_key", "era_label", "era_order"])
        .agg(
            n_speeches=("doc_name", "nunique"),
            n_presidents=("president", "nunique"),
            n_words=("word_count", "sum"),
        )
        .reset_index()
        .sort_values("era_order")
        .reset_index(drop=True)
    )

    topic_behavior = primary[[*PARAGRAPH_KEY, "topics"]].merge(
        speeches[["doc_name", "year"]], on="doc_name", validate="many_to_one"
    )
    topic_behavior = _assign_eras(topic_behavior)
    topic_behavior["normalized_topics"] = topic_behavior["topics"].map(
        lambda value: attention.normalize_topics(value, label_map)
    )
    topic_behavior["n_topics"] = topic_behavior["normalized_topics"].map(len)
    taxonomy_behavior = (
        _ordered_group(topic_behavior, ["era_key", "era_label", "era_order"])["n_topics"]
        .agg(
            n_paragraphs="size",
            unlabeled_paragraphs=lambda values: int(values.eq(0).sum()),
            multi_label_paragraphs=lambda values: int(values.gt(1).sum()),
            normalized_assignments="sum",
            mean_labels="mean",
        )
        .reset_index()
        .sort_values("era_order")
        .reset_index(drop=True)
    )

    population_ledger = pd.DataFrame(
        [
            ("source_documents", "Source corpus documents", "documents", len(speeches), "Miller Center corpus"),
            ("source_paragraphs", "Source corpus paragraphs", "paragraphs", len(paragraphs), "Complete primary annotation population"),
            (
                "speaker_retained_documents",
                "Documents retained after speaker audit",
                "documents",
                speaker_view["doc_name"].nunique(),
                f"{len(speeches) - speaker_view['doc_name'].nunique()} source documents excluded by the governed attribution layer",
            ),
            ("speaker_retained_paragraphs", "Paragraphs retained after speaker audit", "paragraphs", len(speaker_view), "Actual-speaker attribution population"),
            ("speaker_eligible_paragraphs", "Analysis-eligible attributed paragraphs", "paragraphs", int(speaker_view["analysis_eligible"].sum()), "Eligible for actual-speaker results"),
            ("speaker_excluded_paragraphs", "Excluded attributed paragraphs", "paragraphs", int((~speaker_view["analysis_eligible"]).sum()), "Retained with explicit exclusion reasons"),
            ("cross_owner_paragraphs", "Eligible paragraphs credited across document owners", "paragraphs", int((speaker_view["analysis_eligible"] & speaker_view["cross_owner_paragraph"]).sum()), "Actual speaker differs from source-document owner"),
            ("speaker_appearances", "Actual-speaker appearances", "appearances", len(appearances), "A document can contain more than one presidential speaker"),
            ("second_model_expected_documents", "Second-opinion sample", "documents", len(sampled_docs), "Persisted stratified speech draw"),
            ("second_model_expected_paragraphs", "Expected second-opinion paragraphs", "paragraphs", len(expected), "All paragraphs in sampled documents"),
            ("second_model_paired_documents", "Documents with paired paragraph judgments", "documents", paired["doc_name"].nunique(), "At least one completed second-model paragraph"),
            ("second_model_paired_paragraphs", "Paired paragraph judgments", "paragraphs", len(paired), "Shared semantic keys only"),
        ],
        columns=["population_id", "label", "unit", "count", "definition"],
    )

    alias_rows = [
        {"canonical_president": president, "alias": alias}
        for president, aliases in invocations.alias_registry().items()
        for alias in aliases
    ]
    alias_examples = pd.DataFrame(alias_rows).sort_values(["canonical_president", "alias"])

    chart_treatments = pd.DataFrame(
        [
            ("population-ledger", "population counts", "documents, paragraphs, or appearances as labeled", "governed corpus and speaker-attribution artifacts", "descriptive census", "Population stages use different units and must not be added together."),
            ("completion-by-era", "second-model completion", "all expected paragraph keys in sampled documents", "persisted document-level sample", "descriptive completeness", "Completion is not correctness."),
            (
                "zero-sum-flow",
                "cross-model zero-sum judgments",
                f"{len(paired):,} paired paragraph keys",
                "primary and second LLM passes",
                "counts and primary-row percentages",
                "Neither model is human ground truth.",
            ),
            ("zero-sum-by-era", "zero-sum reproducibility", "paired paragraph keys within Story era", "primary and second LLM passes", "Cohen's kappa plus prevalence", "Rare positives make raw agreement optimistic."),
            ("entity-funnel", "entity detection then stance", "normalized entity union, then matched names", "paired paragraph keys only", "name match and stance agreement", "Case/whitespace normalization is not alias resolution."),
            ("corex-uncertainty", "CorEx sampling intervals", "resolved issue-by-five-year cells", "speech-cluster bootstrap", "mean interval width in percentage points", "No LLM annotator exists on this surface."),
            ("llm-uncertainty", "LLM topic intervals", "resolved topic-by-Story-era cells", "speech-cluster bootstrap plus paired-model half-gap", "mean interval width in percentage points", "The paired-sample disagreement component assumes transfer to the full corpus."),
            ("taxonomy-behavior", "topic assignment behavior", "all primary-annotated paragraphs", "canonical case normalization and paragraph-topic deduplication", "paragraph counts and shares", "Multi-label shares are non-additive."),
            ("era-coverage", "corpus coverage", "source documents assigned to Story eras", "document-owner corpus", "speech, president, and word counts", "This corpus is not a census of all presidential communication."),
        ],
        columns=["chart_id", "claim", "denominator", "treatment", "statistical_status", "limitation"],
    )

    both_no = int(confusion.loc[False, False])
    second_only = int(confusion.loc[False, True])
    primary_only = int(confusion.loc[True, False])
    both_yes = int(confusion.loc[True, True])
    primary_positives = primary_only + both_yes
    second_positives = second_only + both_yes
    missing_rows = int(len(expected) - len(paired))
    affected_speeches = int(missing_by_speech["missing_paragraphs"].gt(0).sum())
    matched_total = int(entity_audit["matched"].sum())
    entity_union = int(
        entity_audit[["primary_only", "second_only", "matched"]].sum().sum()
    )
    stance_agrees = int(matched_entities["stance_agrees"].sum())
    entity_bearing = entity_audit.loc[entity_audit["entity_bearing"]]
    entity_bins = (
        pd.cut(
            entity_audit["jaccard"],
            bins=[-0.001, 0.25, 0.5, 0.75, 0.999, 1.001],
            labels=["0–.25", ".25–.50", ".50–.75", ".75–<1", "1 exact"],
        )
        .value_counts(sort=False)
        .rename_axis("agreement_bin")
        .rename("paragraphs")
        .reset_index()
    )
    summary: dict[str, Any] = {
        "source_documents": int(len(speeches)),
        "source_paragraphs": int(len(paragraphs)),
        "paired": int(len(paired)),
        "paired_speeches": int(paired["doc_name"].nunique()),
        "sampled": int(len(expected)),
        "sampled_speeches": int(len(sampled_docs)),
        "missing": missing_rows,
        "affected_speeches": affected_speeches,
        "zero_sum_kappa": float(zero_sum_by_era.iloc[0]["kappa"]),
        "zero_sum_primary": float(zero_sum_by_era.iloc[0]["primary_positive_rate"]),
        "zero_sum_second": float(zero_sum_by_era.iloc[0]["second_positive_rate"]),
        "zero_sum_disagreements": int(len(disagreements)),
        "zero_sum_both_no": both_no,
        "zero_sum_second_only": second_only,
        "zero_sum_primary_only": primary_only,
        "zero_sum_both_yes": both_yes,
        "zero_sum_raw_agreement": (both_no + both_yes) / len(paired),
        "zero_sum_primary_confirmation": both_yes / primary_positives,
        "zero_sum_second_confirmation": both_yes / second_positives,
        "entity_jaccard_all": float(entity_audit["jaccard"].mean()),
        "entity_jaccard_bearing": float(entity_bearing["jaccard"].mean()),
        "entity_bearing_paragraphs": int(len(entity_bearing)),
        "entity_match_rate": matched_total / entity_union,
        "entity_union": entity_union,
        "entity_matched": matched_total,
        "entity_stance_agreement": stance_agrees / matched_total,
        "entity_stance_matches": stance_agrees,
        "thin_presidents": int(thin["thin_record"].sum()),
        "unlabeled_paragraphs": int(topic_behavior["n_topics"].eq(0).sum()),
        "multi_label_paragraphs": int(topic_behavior["n_topics"].gt(1).sum()),
        "normalized_topic_assignments": int(topic_behavior["n_topics"].sum()),
        "retained_documents": int(speaker_view["doc_name"].nunique()),
        "retained_paragraphs": int(len(speaker_view)),
        "eligible_paragraphs": int(speaker_view["analysis_eligible"].sum()),
        "excluded_paragraphs": int((~speaker_view["analysis_eligible"]).sum()),
        "cross_owner_paragraphs": int((speaker_view["analysis_eligible"] & speaker_view["cross_owner_paragraph"]).sum()),
        "speaker_appearances": int(len(appearances)),
        "_charts": {
            "completion": _json_records(missing_by_era),
            "confusion": _json_records(confusion_long),
            "zero_sum_cases": _json_records(zero_sum_cases),
            "zero_sum_era": _json_records(zero_sum_by_era),
            "agreement": _json_records(agreement_by_field),
            "entities": _json_records(entity_bins),
            "entity_cases": _json_records(entity_cases),
            "uncertainty": _json_records(decomposition),
            "taxonomy": _json_records(taxonomy_behavior),
            "era_coverage": _json_records(era_coverage),
        },
    }

    expected_headlines = {
        "paired": 8_570,
        "sampled": 9_048,
        "missing": 478,
        "affected_speeches": 28,
        "entity_union": 8_704,
        "entity_matched": 4_147,
        "entity_stance_matches": 3_653,
        "normalized_topic_assignments": 52_133,
        "multi_label_paragraphs": 14_507,
    }
    observed_headlines = {key: summary[key] for key in expected_headlines}
    if observed_headlines != expected_headlines:
        raise QualityAuditError(
            f"quality headline regression: expected {expected_headlines}, observed {observed_headlines}"
        )
    expected_speaker = {
        "retained_paragraphs": 35_394,
        "eligible_paragraphs": 32_531,
        "excluded_paragraphs": 2_863,
        "cross_owner_paragraphs": 296,
        "speaker_appearances": 1_054,
    }
    observed_speaker = {key: summary[key] for key in expected_speaker}
    if observed_speaker != expected_speaker:
        raise QualityAuditError(
            f"speaker population regression: expected {expected_speaker}, observed {observed_speaker}"
        )
    if not math.isclose(summary["entity_jaccard_bearing"], 0.532, abs_tol=0.0005):
        raise QualityAuditError("entity-bearing Jaccard moved outside its governed tolerance")
    if not math.isclose(summary["entity_jaccard_all"], 0.756, abs_tol=0.0005):
        raise QualityAuditError("all-paired-paragraph entity Jaccard moved outside its governed tolerance")

    tables: dict[str, pd.DataFrame] = {
        "population_ledger": population_ledger,
        "second_model_by_speech": missing_by_speech,
        "second_model_by_era": missing_by_era,
        "zero_sum_confusion": confusion_long,
        "zero_sum_by_era": zero_sum_by_era,
        "zero_sum_disagreement_excerpts": disagreements,
        "entity_name_agreement": entity_audit,
        "entity_stance_matches": matched_entities,
        "entity_case_examples": entity_cases,
        "president_alias_examples": alias_examples,
        "uncertainty_decomposition": decomposition,
        "president_coverage": thin,
        "era_coverage": era_coverage,
        "taxonomy_label_behavior": taxonomy_behavior,
        "agreement_by_field_and_era": agreement_by_field,
        "public_chart_treatments": chart_treatments,
    }
    if (
        set(tables) != set(PUBLIC_TABLES)
        or set(PUBLIC_KEYS) != set(PUBLIC_TABLES)
        or set(PUBLIC_COLUMNS) != set(PUBLIC_TABLES)
    ):
        raise QualityAuditError("public quality table registry is incomplete")
    for name, frame in tables.items():
        if list(frame.columns) != PUBLIC_COLUMNS[name]:
            raise QualityAuditError(
                f"public quality table {name} schema drift: "
                f"{list(frame.columns)} != {PUBLIC_COLUMNS[name]}"
            )
        _require_unique(frame, PUBLIC_KEYS[name], f"public quality table {name}")

    source_receipts = {
        name: {
            "path": str(path.relative_to(data_dir.parent)),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "generation_status": "deterministic_local_no_api_calls",
        "human_validity_status": "not_measured",
        "source_artifacts": source_receipts,
        "annotation_runs": _annotation_run_receipts(
            data_dir,
            [primary, second, primary_entities, second_entities, primary_speech, second_speech],
        ),
        "populations": {
            key: summary[key]
            for key in [
                "source_documents", "source_paragraphs", "sampled", "paired", "missing",
                "affected_speeches", "retained_documents",
                "retained_paragraphs", "eligible_paragraphs", "excluded_paragraphs",
                "cross_owner_paragraphs", "speaker_appearances",
            ]
        },
        "units": {
            "agreement": "proportion_0_to_1",
            "prevalence": "proportion_0_to_1",
            "completion_rate": "proportion_0_to_1",
            "uncertainty_width": "percentage_points",
        },
        "table_schemas": {
            PUBLIC_TABLES[name]: {
                "columns": list(frame.columns),
                "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
                "semantic_key": PUBLIC_KEYS[name],
            }
            for name, frame in sorted(tables.items())
        },
        "agreement_bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": BOOTSTRAP_SEED,
            "confidence_percentiles": [CI_LOW, CI_HIGH],
            "cluster_unit": "doc_name (whole speeches)",
            "algorithm": "multinomial speech-cluster resampling; all paragraphs and entities from a selected speech repeat together",
            "era_axis": "ERA_PROFILE_SPECS (approved Story eras)",
        },
        "agreement_v1_validation": canonical_checks,
        "bands_contract": {
            "metadata_sha256": _sha256_file(paths["bands_metadata"]),
            "surfaces": bands_meta.get("surfaces"),
            "cross_surface_comparison": "prohibited_different_temporal_grains",
        },
        "speaker_contract": {
            "schema_version": speaker_meta.get("schema_version"),
            "metadata_sha256": speaker_meta.get("metadata_sha256"),
        },
        "validations": {
            "headline_regression": "passed",
            "paragraph_key_set": "passed",
            "paired_entity_population": "passed",
            "canonical_agreement": "passed",
            "speaker_artifact_hashes": "passed",
            "chronological_story_eras": "passed",
            "finite_nonnegative_uncertainty_widths": "passed",
        },
    }
    manifest["metadata_sha256"] = _metadata_hash(manifest)
    return QualityAuditBundle(tables=tables, summary=summary, manifest=manifest)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_completion_projection(frame: pd.DataFrame, label: str) -> None:
    columns = [
        "expected_paragraphs", "completed_paragraphs", "missing_paragraphs",
        "completion_rate",
    ]
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise QualityAuditError(f"{label} contains non-finite completion values")
    if (numeric[["expected_paragraphs", "completed_paragraphs", "missing_paragraphs"]] < 0).any().any():
        raise QualityAuditError(f"{label} contains negative missingness or support")
    if (numeric["expected_paragraphs"] <= 0).any():
        raise QualityAuditError(f"{label} contains an empty expected population")
    if not (
        numeric["expected_paragraphs"] - numeric["completed_paragraphs"]
    ).equals(numeric["missing_paragraphs"]):
        raise QualityAuditError(f"{label} missingness identity failed")
    expected_rate = numeric["completed_paragraphs"] / numeric["expected_paragraphs"]
    if not np.allclose(expected_rate, numeric["completion_rate"], rtol=0, atol=1e-12):
        raise QualityAuditError(f"{label} completion-rate identity failed")


def _validate_public_invariants(
    tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]
) -> None:
    populations = manifest.get("populations")
    if not isinstance(populations, Mapping):
        raise QualityAuditError("quality manifest populations are missing")
    try:
        population_values = {key: int(value) for key, value in populations.items()}
    except (TypeError, ValueError) as exc:
        raise QualityAuditError("quality manifest populations must be integer counts") from exc
    if any(value < 0 for value in population_values.values()):
        raise QualityAuditError("quality manifest populations cannot be negative")
    if {
        key: population_values.get(key) for key in EXPECTED_HEADLINE_POPULATIONS
    } != EXPECTED_HEADLINE_POPULATIONS:
        raise QualityAuditError("quality manifest headline population regression failed")
    if population_values["sampled"] - population_values["paired"] != population_values["missing"]:
        raise QualityAuditError("quality manifest missingness identity failed")

    ledger = tables["population_ledger"].set_index("population_id")
    ledger_mapping = {
        "source_documents": "source_documents",
        "source_paragraphs": "source_paragraphs",
        "retained_documents": "speaker_retained_documents",
        "retained_paragraphs": "speaker_retained_paragraphs",
        "eligible_paragraphs": "speaker_eligible_paragraphs",
        "excluded_paragraphs": "speaker_excluded_paragraphs",
        "cross_owner_paragraphs": "cross_owner_paragraphs",
        "speaker_appearances": "speaker_appearances",
        "sampled": "second_model_expected_paragraphs",
        "paired": "second_model_paired_paragraphs",
    }
    for population_key, ledger_key in ledger_mapping.items():
        if ledger_key not in ledger.index:
            raise QualityAuditError(f"population ledger is missing {ledger_key}")
        if int(ledger.loc[ledger_key, "count"]) != population_values[population_key]:
            raise QualityAuditError(
                f"population ledger disagrees with manifest for {population_key}"
            )
    if population_values["eligible_paragraphs"] + population_values["excluded_paragraphs"] != population_values["retained_paragraphs"]:
        raise QualityAuditError("speaker eligibility population identity failed")

    by_speech = tables["second_model_by_speech"]
    by_era = tables["second_model_by_era"]
    _validate_completion_projection(by_speech, "second-model speech projection")
    _validate_completion_projection(by_era, "second-model era projection")
    for frame, label in ((by_speech, "speech"), (by_era, "era")):
        totals = frame[[
            "expected_paragraphs", "completed_paragraphs", "missing_paragraphs"
        ]].sum()
        observed = tuple(map(int, totals))
        expected = (
            population_values["sampled"], population_values["paired"],
            population_values["missing"],
        )
        if observed != expected:
            raise QualityAuditError(
                f"second-model {label} totals {observed} do not match populations {expected}"
            )
    affected = int(pd.to_numeric(by_speech["missing_paragraphs"]).gt(0).sum())
    if affected != population_values["affected_speeches"]:
        raise QualityAuditError("affected-speech population identity failed")

    expected_eras = [spec.key for spec in era_profiles.ERA_PROFILE_SPECS]
    for name in ("second_model_by_era", "era_coverage", "taxonomy_label_behavior"):
        frame = tables[name]
        if frame["era_key"].astype(str).tolist() != expected_eras:
            raise QualityAuditError(f"{name} does not follow governed Story chronology")
        if pd.to_numeric(frame["era_order"], errors="raise").astype(int).tolist() != list(range(9)):
            raise QualityAuditError(f"{name} has invalid governed era order")
    zero_era = tables["zero_sum_by_era"]
    if zero_era["era_key"].astype(str).tolist() != ["overall", *expected_eras]:
        raise QualityAuditError("zero-sum projection does not follow governed Story chronology")

    confusion = tables["zero_sum_confusion"]
    if int(pd.to_numeric(confusion["n"], errors="raise").sum()) != population_values["paired"]:
        raise QualityAuditError("zero-sum confusion population differs from paired paragraphs")
    if (pd.to_numeric(confusion["n"], errors="raise") < 0).any():
        raise QualityAuditError("zero-sum confusion contains negative counts")
    row_percent = pd.to_numeric(confusion["row_percent"], errors="raise")
    if not row_percent.between(0, 100).all():
        raise QualityAuditError("zero-sum row percentages must use percentage-point units")
    if not np.allclose(
        confusion.assign(_pct=row_percent).groupby("primary", dropna=False)["_pct"].sum(),
        100.0,
        rtol=0,
        atol=1e-9,
    ):
        raise QualityAuditError("zero-sum row percentages do not sum to 100")

    entity = tables["entity_name_agreement"]
    if len(entity) != population_values["paired"]:
        raise QualityAuditError("entity paragraph population differs from paired paragraphs")
    entity_counts = entity[["primary_only", "second_only", "matched"]].apply(
        pd.to_numeric, errors="raise"
    )
    if (entity_counts < 0).any().any():
        raise QualityAuditError("entity agreement contains negative counts")
    if int(entity_counts.to_numpy().sum()) != 8_704 or int(entity_counts["matched"].sum()) != 4_147:
        raise QualityAuditError("entity detection headline regression failed")
    jaccard = pd.to_numeric(entity["jaccard"], errors="raise")
    if not np.isfinite(jaccard.to_numpy(float)).all() or not jaccard.between(0, 1).all():
        raise QualityAuditError("entity Jaccard values must be finite proportions")
    bearing = entity["entity_bearing"]
    if not pd.api.types.is_bool_dtype(bearing):
        raise QualityAuditError("entity-bearing status must be boolean")
    if not math.isclose(float(jaccard.mean()), 0.7559823691153913, abs_tol=1e-12):
        raise QualityAuditError("all-paragraph entity Jaccard regression failed")
    if not math.isclose(
        float(jaccard.loc[bearing].mean()), 0.5323722950176439, abs_tol=1e-12
    ):
        raise QualityAuditError("entity-bearing Jaccard regression failed")
    stance = tables["entity_stance_matches"]
    if len(stance) != 4_147 or not pd.api.types.is_bool_dtype(stance["stance_agrees"]):
        raise QualityAuditError("matched-entity stance population regression failed")
    if int(stance["stance_agrees"].sum()) != 3_653:
        raise QualityAuditError("entity stance-agreement headline regression failed")

    taxonomy = tables["taxonomy_label_behavior"]
    topic_totals = {
        "paragraphs": int(pd.to_numeric(taxonomy["n_paragraphs"]).sum()),
        "multi": int(pd.to_numeric(taxonomy["multi_label_paragraphs"]).sum()),
        "assignments": int(pd.to_numeric(taxonomy["normalized_assignments"]).sum()),
    }
    if topic_totals != {
        "paragraphs": population_values["source_paragraphs"],
        "multi": 14_507,
        "assignments": 52_133,
    }:
        raise QualityAuditError("normalized topic headline regression failed")

    uncertainty = tables["uncertainty_decomposition"]
    width_columns = [column for column in uncertainty if column.endswith("_width_pp")]
    widths = uncertainty[width_columns].apply(pd.to_numeric, errors="coerce")
    resolved_widths = widths.to_numpy(float)
    if np.isinf(resolved_widths).any() or (widths < 0).any().any():
        raise QualityAuditError("uncertainty widths must be finite nonnegative percentage points")


def validate_publication(
    public_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Re-open and fail closed on a complete ``data-quality-v2`` projection."""
    public_dir = Path(public_dir)
    root = Path(repo_root) if repo_root is not None else DATA_DIR.parent
    root = root.resolve()
    manifest_path = public_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise QualityAuditError("quality publication manifest is missing or invalid") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("contract_version") != CONTRACT_VERSION:
        raise QualityAuditError("quality publication schema or contract version is invalid")
    if manifest.get("metadata_sha256") != _metadata_hash(manifest):
        raise QualityAuditError("quality publication manifest self-hash mismatch")
    expected_units = {
        "agreement": "proportion_0_to_1",
        "prevalence": "proportion_0_to_1",
        "completion_rate": "proportion_0_to_1",
        "uncertainty_width": "percentage_points",
    }
    if manifest.get("units") != expected_units:
        raise QualityAuditError("quality publication unit contract mismatch")

    sources = manifest.get("source_artifacts")
    if not isinstance(sources, Mapping) or not sources:
        raise QualityAuditError("quality publication source receipts are missing")
    for name, receipt in sources.items():
        if not isinstance(receipt, Mapping):
            raise QualityAuditError(f"quality source receipt {name} is invalid")
        source = (root / str(receipt.get("path", ""))).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise QualityAuditError(f"quality source receipt {name} escapes the repository") from exc
        if not source.is_file():
            raise QualityAuditError(f"quality source receipt {name} is missing")
        if int(receipt.get("bytes", -1)) != source.stat().st_size or receipt.get("sha256") != _sha256_file(source):
            raise QualityAuditError(f"quality source receipt {name} is stale")

    artifacts = manifest.get("artifacts")
    schemas = manifest.get("table_schemas")
    expected_files = set(PUBLIC_TABLES.values())
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_files:
        raise QualityAuditError("quality publication artifact inventory is incomplete")
    if not isinstance(schemas, Mapping) or set(schemas) != expected_files:
        raise QualityAuditError("quality publication table-schema inventory is incomplete")

    tables: dict[str, pd.DataFrame] = {}
    for name, filename in PUBLIC_TABLES.items():
        path = public_dir / filename
        if not path.is_file():
            raise QualityAuditError(f"quality publication is missing {filename}")
        receipt = artifacts[filename]
        if (
            receipt.get("schema_version") != CONTRACT_VERSION
            or int(receipt.get("bytes", -1)) != path.stat().st_size
            or receipt.get("sha256") != _sha256_file(path)
        ):
            raise QualityAuditError(f"quality artifact receipt mismatch for {filename}")
        schema = schemas[filename]
        if schema.get("columns") != PUBLIC_COLUMNS[name]:
            raise QualityAuditError(f"quality schema columns mismatch for {filename}")
        if schema.get("semantic_key") != PUBLIC_KEYS[name]:
            raise QualityAuditError(f"quality semantic key mismatch for {filename}")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise QualityAuditError(f"quality artifact {filename} is not a readable CSV") from exc
        if list(frame.columns) != PUBLIC_COLUMNS[name]:
            raise QualityAuditError(f"quality CSV columns mismatch for {filename}")
        if int(receipt.get("rows", -1)) != len(frame):
            raise QualityAuditError(f"quality row-count receipt mismatch for {filename}")
        _require_unique(frame, PUBLIC_KEYS[name], f"published quality table {filename}")
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.size and np.isinf(numeric.to_numpy(float)).any():
            raise QualityAuditError(f"quality artifact {filename} contains infinite values")
        tables[name] = frame
    _validate_public_invariants(tables, manifest)
    return dict(manifest)


def write_publication(bundle: QualityAuditBundle, public_dir: Path) -> dict[str, Any]:
    """Atomically publish CSV projections, then their self-hashed manifest."""
    public_dir = Path(public_dir)
    missing = sorted(set(PUBLIC_TABLES) - set(bundle.tables))
    if missing:
        raise QualityAuditError(f"quality bundle is missing public tables: {missing}")
    payloads = {
        PUBLIC_TABLES[name]: _csv_bytes(bundle.tables[name])
        for name in sorted(PUBLIC_TABLES)
    }
    manifest = dict(bundle.manifest)
    manifest["artifacts"] = {
        filename: {
            "schema_version": CONTRACT_VERSION,
            "rows": int(len(bundle.tables[name])),
            "bytes": len(payloads[filename]),
            "sha256": _sha256_bytes(payloads[filename]),
        }
        for name, filename in sorted(PUBLIC_TABLES.items())
    }
    manifest["metadata_sha256"] = _metadata_hash(manifest)
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    for filename, payload in sorted(payloads.items()):
        _atomic_write(public_dir / filename, payload)
    _atomic_write(public_dir / MANIFEST_NAME, manifest_payload)
    # The POS analysis moved out of Data Quality.  Refuse to leave a stale export
    # behind during an incremental build.
    (public_dir / "era_part_of_speech.csv").unlink(missing_ok=True)
    return validate_publication(public_dir)


def build_and_write(
    data_dir: Path,
    public_dir: Path,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    bundle = build_quality_audit(data_dir, bootstrap_draws=bootstrap_draws)
    manifest = write_publication(bundle, public_dir)
    summary = dict(bundle.summary)
    summary["manifest"] = manifest
    return summary


def assert_no_network_dependencies() -> None:
    """Static defense against coupling this derived builder to paid/network code."""
    source = Path(__file__).read_text(encoding="utf-8")
    tokens = ("anth" + "ropic", "requests" + ".", "httpx" + ".")
    prohibited = [token for token in tokens if token in source]
    if prohibited:
        raise QualityAuditError(f"quality audit contains prohibited network dependencies: {prohibited}")


if __name__ == "__main__":  # pragma: no cover - operator convenience
    assert_no_network_dependencies()
    result = build_quality_audit()
    print(_canonical_json({key: value for key, value in result.summary.items() if key != "_charts"}))
