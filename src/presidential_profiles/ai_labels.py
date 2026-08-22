"""Read-only adapters from the frozen LLM annotations to website payloads.

The paid annotation files are provenance artifacts.  This module never writes
or regenerates them; it only performs deterministic, keyed aggregation for the
static site.  Topic normalization deliberately delegates to ``attention`` so
the website cannot drift from the research pipeline's case-folding and
de-duplication rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import attention, indices
from .corpus import load
from .llm_annotations import (
    ANNOTATIONS_DIR,
    MANIFESTS_DIR,
    load_paragraph_annotations,
    load_speech_annotations,
)
from .similarity import ERA_WINDOW

ENTITY_PATH = ANNOTATIONS_DIR / "paragraph_entities.parquet"
AGREEMENT_PATH = ANNOTATIONS_DIR / "agreement_v1.parquet"
CROSSWALK_PATH = ANNOTATIONS_DIR / "crosswalk_v1.json"

FLAG_FIELDS = ("party_attack", "enemy_naming", "zero_sum")
PROPOSAL_FIELDS = ("proposal", "values", "mixed", "neither")

DISPLAY_LABELS = {
    "party_attack": "Partisan attack",
    "enemy_naming": "Named adversary",
    "zero_sum": "Zero-sum framing",
    "proposal": "Concrete proposal",
    "values": "Values appeal",
    "mixed": "Proposal + values",
    "neither": "Neither",
}


def humanize(value: str) -> str:
    """Human-readable closed-enum label for site prose."""
    special = {
        "state_of_the_union_or_annual_message": "State of the Union / annual message",
        "veto_or_signing_statement": "Veto / signing statement",
        "press_conference_or_interview": "Press conference / interview",
        "public_remarks_or_address": "Public remarks / address",
        "campaign_or_debate": "Campaign / debate",
        "foreign_or_diplomatic": "Foreign / diplomatic",
        "specific_organization_or_group": "Specific organization / group",
        "broadcast_radio_or_tv": "Radio / TV broadcast",
    }
    return special.get(value, value.replace("_", " ").capitalize())


def _require_full_merge(n_merged: int, inputs: dict[str, int], stage: str) -> None:
    if any(n_merged != n for n in inputs.values()):
        detail = ", ".join(f"{k}={v}" for k, v in inputs.items())
        raise ValueError(
            f"{stage}: keyed merge changed row count (merged={n_merged}; {detail})"
        )


def _topic_assignments(annotations: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    """One normalized row per distinct (paragraph, level-2 topic)."""
    label_map = attention.canonical_label_map(taxonomy)
    parents = attention.level1_parents(taxonomy)
    rows: list[tuple[str, int, str, str]] = []
    for doc, idx, raw in annotations[["doc_name", "para_idx", "topics"]].itertuples(
        index=False, name=None
    ):
        for topic in attention.normalize_topics(raw, label_map):
            rows.append((doc, int(idx), topic, parents[topic]))
    return pd.DataFrame(rows, columns=["doc_name", "para_idx", "topic", "level1"])


def _shares(
    assignments: pd.DataFrame,
    paragraphs: pd.DataFrame,
    column: str,
    values: list[str],
) -> pd.DataFrame:
    """President x label paragraph shares with the full paragraph denominator."""
    counts = (
        assignments.drop_duplicates(["doc_name", "para_idx", column])
        .groupby(["president", column], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=values, fill_value=0)
    )
    denom = paragraphs.groupby("president").size()
    return counts.reindex(denom.index, fill_value=0).div(denom, axis=0)


def _era_relative(shares: pd.DataFrame, first_year: pd.Series) -> pd.DataFrame:
    """Percentage-point difference from presidents whose careers start nearby."""
    out = pd.DataFrame(index=shares.index, columns=shares.columns, dtype=float)
    for president in shares.index:
        mask = (first_year - first_year[president]).abs() <= ERA_WINDOW
        mask.loc[president] = False
        peers = shares.loc[mask] if int(mask.sum()) >= 2 else shares.drop(president)
        out.loc[president] = (shares.loc[president] - peers.mean()) * 100
    return out


def _overall_agreement(path: Path = AGREEMENT_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    overall = df[df["era_label"] == "overall"]
    return {
        f"{row.field}:{row.metric}": {"value": float(row.value), "n": int(row.n)}
        for row in overall.itertuples()
    }


def _manifest_summary(manifest_dir: Path = MANIFESTS_DIR) -> dict:
    manifests = [json.loads(path.read_text()) for path in sorted(manifest_dir.glob("*.json"))]
    taxonomy = next((m for m in manifests if m.get("run_id") == "taxonomy-v1-20260719"), {})
    primary_names = {
        "paragraph_annotations.parquet",
        "paragraph_entities.parquet",
        "speech_annotations.parquet",
    }
    primary = [
        m for m in manifests
        if primary_names.intersection(m.get("annotation_files", []))
        and not any("__opus4-8" in f for f in m.get("annotation_files", []))
    ]
    secondary = [
        m for m in manifests
        if any("__opus4-8" in f for f in m.get("annotation_files", []))
    ]

    def summarize(rows: list[dict]) -> dict:
        return {
            "cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in rows), 4),
            "requests": sum(int(r.get("n_requests") or 0) for r in rows),
            "models": sorted({str(r.get("model")) for r in rows if r.get("model")}),
        }

    return {
        "taxonomy": {
            "cost_usd": float(taxonomy.get("cost_usd") or 0),
            "requests": int(taxonomy.get("n_requests") or 0),
            "models": [taxonomy.get("model")] if taxonomy else [],
        },
        "primary": summarize(primary),
        "secondary": summarize(secondary),
    }


def build_ai_data(
    *,
    speeches: pd.DataFrame | None = None,
    annotations: pd.DataFrame | None = None,
    speech_annotations: pd.DataFrame | None = None,
    entities: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
) -> dict:
    """Build all AI-derived website payloads without mutating source artifacts."""
    speeches = (load() if speeches is None else speeches).copy()
    speeches = speeches.drop_duplicates("doc_name")
    annotations = (
        load_paragraph_annotations("paragraph_annotations")
        if annotations is None else annotations.copy()
    )
    speech_annotations = (
        load_speech_annotations("speech_annotations")
        if speech_annotations is None else speech_annotations.copy()
    )
    entities = (
        pd.read_parquet(ENTITY_PATH) if entities is None else entities.copy()
    )
    taxonomy = attention.load_taxonomy() if taxonomy is None else taxonomy

    speech_meta = speeches[["doc_name", "president", "year"]]
    paragraphs = annotations.merge(
        speech_meta, on="doc_name", how="inner", validate="many_to_one"
    )
    _require_full_merge(
        len(paragraphs),
        {"annotations": len(annotations)},
        "paragraph annotations x speeches",
    )

    assignments = _topic_assignments(annotations, taxonomy).merge(
        speech_meta, on="doc_name", how="inner", validate="many_to_one"
    )
    topic_names = [entry["name"] for entry in taxonomy["level2"]]
    domain_names = [entry["name"] for entry in taxonomy["level1"]]
    definitions = {entry["name"]: entry["definition"] for entry in taxonomy["level2"]}
    parent = {entry["name"]: entry["level1"] for entry in taxonomy["level2"]}
    topic_shares = _shares(assignments, paragraphs, "topic", topic_names)
    domain_shares = _shares(assignments, paragraphs, "level1", domain_names)
    first_year = speeches.groupby("president")["year"].min().reindex(topic_shares.index)
    topic_rel = _era_relative(topic_shares, first_year)
    domain_rel = _era_relative(domain_shares, first_year)
    topic_counts = (
        assignments.groupby(["president", "topic"], observed=True).size()
        .unstack(fill_value=0).reindex(index=topic_shares.index, columns=topic_names, fill_value=0)
    )

    flag_rates = paragraphs.groupby("president")[list(FLAG_FIELDS)].mean() * 100
    proposal_rates = (
        pd.crosstab(paragraphs["president"], paragraphs["proposal_values"], normalize="index")
        .reindex(index=topic_shares.index, columns=PROPOSAL_FIELDS, fill_value=0) * 100
    )
    speech_labeled = speech_annotations.merge(
        speech_meta, on="doc_name", how="inner", validate="one_to_one"
    )
    _require_full_merge(
        len(speech_labeled),
        {"speech_annotations": len(speech_annotations), "speeches": len(speech_meta)},
        "speech annotations x speeches",
    )
    speech_type_rates = (
        pd.crosstab(speech_labeled["president"], speech_labeled["speech_type"], normalize="index")
        * 100
    )

    adversarial = entities[entities["stance"] == "adversarial"].merge(
        speech_meta[["doc_name", "president"]], on="doc_name", validate="many_to_one"
    )
    adversary_counts = adversarial.groupby(["president", "entity"]).size()

    n_paragraphs = paragraphs.groupby("president").size()
    n_speeches = speeches.groupby("president").size()
    by_president: dict[str, dict] = {}
    for president in topic_shares.index:
        raw_order = topic_shares.loc[president].sort_values(ascending=False)
        distinctive_pool = [
            topic for topic in topic_names if int(topic_counts.loc[president, topic]) >= 4
        ]
        rel_order = topic_rel.loc[president, distinctive_pool].sort_values(ascending=False)
        domain_order = domain_shares.loc[president].sort_values(ascending=False)
        top_types = speech_type_rates.loc[president].sort_values(ascending=False)
        adv = (
            adversary_counts.loc[president].sort_values(ascending=False).head(5)
            if president in adversary_counts.index.get_level_values(0) else pd.Series(dtype=int)
        )
        by_president[president] = {
            "n_paragraphs": int(n_paragraphs[president]),
            "n_speeches": int(n_speeches[president]),
            "low_confidence": bool(
                n_speeches[president]
                < indices.PRESIDENT_PERCENTILE_MIN_SPEECHES
            ),
            "top_topics": [
                {
                    "name": topic,
                    "domain": parent[topic],
                    "definition": definitions[topic],
                    "share": round(float(topic_shares.loc[president, topic] * 100), 2),
                    "rel": round(float(topic_rel.loc[president, topic]), 2),
                    "n": int(topic_counts.loc[president, topic]),
                }
                for topic in raw_order.head(6).index
            ],
            "distinctive_topics": [
                {
                    "name": topic,
                    "domain": parent[topic],
                    "definition": definitions[topic],
                    "share": round(float(topic_shares.loc[president, topic] * 100), 2),
                    "rel": round(float(topic_rel.loc[president, topic]), 2),
                    "n": int(topic_counts.loc[president, topic]),
                }
                for topic in rel_order.head(5).index
            ],
            "domains": [
                {
                    "name": domain,
                    "share": round(float(domain_shares.loc[president, domain] * 100), 2),
                    "rel": round(float(domain_rel.loc[president, domain]), 2),
                }
                for domain in domain_order.head(5).index
            ],
            "topic_attention": [
                {"name": topic, "domain": parent[topic],
                 "share": round(float(topic_shares.loc[president, topic] * 100), 3),
                 "rel": round(float(topic_rel.loc[president, topic]), 3),
                 "n": int(topic_counts.loc[president, topic])}
                for topic in topic_names
            ],
            "domain_attention": [
                {"name": domain,
                 "share": round(float(domain_shares.loc[president, domain] * 100), 3),
                 "rel": round(float(domain_rel.loc[president, domain]), 3)}
                for domain in domain_names
            ],
            "flags": {
                field: round(float(flag_rates.loc[president, field]), 2)
                for field in FLAG_FIELDS
            },
            "proposal_values": {
                field: round(float(proposal_rates.loc[president, field]), 2)
                for field in PROPOSAL_FIELDS
            },
            "speech_types": [
                {"name": humanize(name), "value": round(float(value), 2)}
                for name, value in top_types.head(4).items()
            ],
            "adversaries": [
                {"name": str(name), "n": int(value)} for name, value in adv.items()
            ],
        }

    # A separate AI radar uses comparable president percentiles while retaining
    # absolute rates beside them. Thin records stay visible but are not assigned
    # a precise outlier rank.
    breadth = topic_shares.apply(
        lambda row: float(np.exp(-(row[row > 0] / row.sum()
                                   * np.log(row[row > 0] / row.sum())).sum())),
        axis=1,
    )
    radar_raw = pd.DataFrame({
        "party_attack": flag_rates["party_attack"],
        "enemy_naming": flag_rates["enemy_naming"],
        "zero_sum": flag_rates["zero_sum"],
        "proposal": proposal_rates["proposal"],
        "values": proposal_rates["values"],
        "topic_breadth": breadth,
    })
    precise = (
        n_speeches.reindex(radar_raw.index)
        >= indices.PRESIDENT_PERCENTILE_MIN_SPEECHES
    )
    percentiles = radar_raw.loc[precise].rank(pct=True, method="average") * 100
    for president, info in by_president.items():
        info["ai_radar"] = {
            key: {
                "absolute": round(float(radar_raw.loc[president, key]), 3),
                "percentile": (round(float(percentiles.loc[president, key]), 1)
                               if president in percentiles.index else None),
            }
            for key in radar_raw.columns
        }

    topic_overall = (
        assignments.groupby("topic").size().reindex(topic_names, fill_value=0) / len(paragraphs)
    ).sort_values(ascending=False)
    empty_topics = int(annotations["topics"].map(lambda x: len(list(x)) == 0).sum())
    agreement = _overall_agreement()
    summary = {
        "n_speeches": int(len(speech_annotations)),
        "n_paragraphs": int(len(annotations)),
        "n_assignments": int(len(assignments)),
        "mean_topics": round(float(len(assignments) / len(annotations)), 3),
        "empty_topics": empty_topics,
        "empty_share": round(float(empty_topics / len(annotations) * 100), 2),
        "n_entities": int(len(entities)),
        "n_domains": len(domain_names),
        "n_topics": len(topic_names),
        "flag_rates": {
            field: round(float(annotations[field].mean() * 100), 2) for field in FLAG_FIELDS
        },
        "proposal_values": {
            field: round(float((annotations["proposal_values"] == field).mean() * 100), 2)
            for field in PROPOSAL_FIELDS
        },
        "top_topics": [
            {"name": name, "share": round(float(value * 100), 2), "domain": parent[name]}
            for name, value in topic_overall.head(10).items()
        ],
    }

    return {
        "taxonomy": taxonomy,
        "by_president": by_president,
        "summary": summary,
        "agreement": agreement,
        "manifests": _manifest_summary(),
        "topic_assignments": assignments,
        "paragraphs": paragraphs,
    }


def explorer_topic_series(ai_data: dict, min_year_paragraphs: int = 10) -> dict[str, dict]:
    """Per-year AI topic shares for the word explorer's topic menu."""
    assignments = ai_data["topic_assignments"]
    paragraphs = ai_data["paragraphs"]
    den = paragraphs.groupby("year").size()
    num = assignments.groupby(["year", "topic"], observed=True).size().unstack(fill_value=0)
    names = [entry["name"] for entry in ai_data["taxonomy"]["level2"]]
    out = {}
    for name in names:
        share = (num.get(name, pd.Series(0, index=den.index)).reindex(den.index, fill_value=0)
                 / den * 100)
        share = share[den >= min_year_paragraphs].round(2)
        label = f"AI topic · {name}"
        out[label] = {"y": [int(y) for y in share.index], "v": list(share.values)}
    return out


def topic_bucket_series(
    ai_data: dict,
    bucket_years: int,
    min_paragraphs: int = 20,
) -> dict[str, dict]:
    """Weighted AI-topic shares in complete 10- or 20-year buckets."""
    if bucket_years not in {10, 20}:
        raise ValueError("topic buckets must be 10 or 20 years")
    assignments = ai_data["topic_assignments"].copy()
    paragraphs = ai_data["paragraphs"].copy()
    assignments["period"] = (assignments.year // bucket_years) * bucket_years
    paragraphs["period"] = (paragraphs.year // bucket_years) * bucket_years
    denominator = paragraphs.groupby("period").size()
    numerator = (
        assignments.drop_duplicates(["doc_name", "para_idx", "topic"])
        .groupby(["period", "topic"], observed=True).size().unstack(fill_value=0)
    )
    names = [entry["name"] for entry in ai_data["taxonomy"]["level2"]]
    output = {}
    for name in names:
        counts = numerator.get(name, pd.Series(0, index=denominator.index))
        share = (
            counts.reindex(denominator.index, fill_value=0)
            .div(denominator).mul(100)
            .where(denominator >= min_paragraphs)
            .dropna().round(3)
        )
        output[name] = {
            "x": [int(period) for period in share.index],
            "v": [float(value) for value in share.values],
            "n": [
                int(denominator.loc[period])
                for period in share.index
            ],
        }
    return output


def issue_crosswalk(ai_data: dict, path: Path = CROSSWALK_PATH) -> dict[str, list[dict]]:
    """Legacy issue name -> finer AI topics with definitions and domains."""
    raw = json.loads(path.read_text())
    topic_meta = {entry["name"]: entry for entry in ai_data["taxonomy"]["level2"]}
    out: dict[str, list[dict]] = {}
    for row in raw["mappings"]:
        out[row["legacy_issue"]] = [topic_meta[name] for name in row["level2_topics"]]
    return out
