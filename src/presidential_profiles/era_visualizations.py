"""Reusable data contract for the shared Visualize-the-era evidence.

The public entry points mirror :mod:`presidential_profiles.era_profiles`:

* ``load_all_era_visualizations()`` derives every canonical era from current
  local artifacts.
* ``load_era_visualization("expansion")`` is the one-argument convenience
  path used by future era-specific templates.
* ``build_era_visualizations(...)`` accepts already-loaded tables so the site
  build performs no duplicate I/O.

Every displayed chart is derived from exact keyed joins.  The retained
communication summary remains available to synthesis consumers even though it
is no longer a standalone era graph.  The generated JSON is a portable,
era-indexed input contract; the HTML renderer never needs to know which era it
is displaying.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from . import attention, corpus, era_profiles


PRESIDENT_COLORS = (
    "#315f78",
    "#9a7133",
    "#a24f52",
    "#4f7557",
    "#6d5a88",
    "#477f7b",
    "#a7653f",
    "#667342",
    "#7f5b70",
    "#4c6c91",
)
AGENDA_MIN_SPEECHES = 5

AGENDA_DOMAIN_SHORT_LABELS = {
    "Foreign Relations & Diplomacy": "Diplomacy",
    "War & Military Affairs": "War & military",
    "Money, Banking & Currency": "Money & banking",
    "Trade, Commerce & Tariffs": "Trade & tariffs",
    "Public Finance & Taxation": "Public finance",
    "Economy, Labor & Social Welfare": "Economy & welfare",
    "Agriculture, Land & Environment": "Land & agriculture",
    "Infrastructure & Internal Improvements": "Infrastructure",
    "Civil Rights & Immigration": "Civil rights",
    "Indian & Tribal Affairs": "Native affairs",
    "Law, Crime & Justice": "Law & justice",
    "Constitutional Order & Governance": "Constitutional order",
}

AGENDA_DOMAIN_COLORS = {
    "Foreign Relations & Diplomacy": "#315f78",
    "War & Military Affairs": "#a24f52",
    "Money, Banking & Currency": "#9a7133",
    "Trade, Commerce & Tariffs": "#b5793d",
    "Public Finance & Taxation": "#8a6a45",
    "Economy, Labor & Social Welfare": "#667342",
    "Agriculture, Land & Environment": "#78905e",
    "Infrastructure & Internal Improvements": "#477f7b",
    "Civil Rights & Immigration": "#6d5a88",
    "Indian & Tribal Affairs": "#a7653f",
    "Law, Crime & Justice": "#7f5b70",
    "Constitutional Order & Governance": "#4c6c91",
}

TOPIC_SHORT_LABELS = {
    "Constitutional Union & Federalism": "Constitutional union & federalism",
    "Crime, Insurrection & Federal Law Enforcement": "Federal law enforcement",
    "Executive Power, Vetoes & the Courts": "Executive power & courts",
    "Public Debt, Revenue & Treasury Finance": "Debt, revenue & Treasury",
    "Taxes, Budget Deficits & Federal Spending": "Taxes, budgets & spending",
    "Coinage, Currency & Specie": "Currency & specie",
    "National Bank & Banking Crises": "National bank & banking",
    "Tariffs, Reciprocity & Navigation Laws": "Tariffs & navigation laws",
    "Treaties, Diplomacy & International Arbitration": "Treaties & diplomacy",
    "Neutral Rights & Maritime Depredations": "Maritime rights",
    "Military Preparedness, Armed Forces & Veterans": "Military preparedness",
    "Early Naval Wars: Barbary & the War of 1812": "Naval wars",
    "Indian Affairs, Removal & Allotment": "Native affairs",
    "Providence, Faith & American Ideals": "Faith & national ideals",
    "Faith, Providence & National Ideals": "Faith & national ideals",
    "Relations with Spain, Mexico & Territorial Claims": "Territorial claims",
}


def topic_icon(topic: str) -> str:
    """Return one stable semantic icon for a level-2 topic label."""
    lowered = topic.casefold()
    if any(term in lowered for term in ("maritime", "indian", "native")):
        return "🌎"
    if any(
        term in lowered
        for term in (
            "war",
            "military",
            "armed forces",
            "veterans",
            "treat",
            "diplom",
            "foreign",
            "defense",
            "peace",
        )
    ):
        return "🛡️"
    if any(
        term in lowered
        for term in ("faith", "providence", "religion", "ideal")
    ):
        return "✨"
    if any(
        term in lowered
        for term in (
            "constitution",
            "federal",
            "court",
            "crime",
            "law enforcement",
            "civil rights",
            "race",
        )
    ):
        return "🏛️"
    if any(
        term in lowered
        for term in (
            "finance",
            "debt",
            "revenue",
            "treasury",
            "tax",
            "budget",
            "currency",
            "bank",
            "econom",
            "trade",
            "tariff",
            "jobs",
        )
    ):
        return "💰"
    if any(term in lowered for term in ("agriculture", "farm", "land")):
        return "🌾"
    if any(term in lowered for term in ("health", "education", "welfare")):
        return "👥"
    return "•"


def _strict_one_to_one_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    stage: str,
) -> pd.DataFrame:
    joined = left.merge(
        right,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    mismatch = joined["_merge"].ne("both")
    if mismatch.any():
        counts = joined.loc[mismatch, "_merge"].value_counts().to_dict()
        raise ValueError(
            f"{stage}: key sets differ on {keys}; unmatched rows={counts}"
        )
    return joined.drop(columns="_merge")


def _communication_rows(
    frame: pd.DataFrame,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    column: str,
) -> list[dict]:
    return [
        {
            "label": label,
            "raw_values": list(raw_values),
            "count": int(frame[column].isin(raw_values).sum()),
            "share": float(frame[column].isin(raw_values).mean() * 100),
        }
        for label, raw_values in groups
    ]


def _president_rows(era_speeches: pd.DataFrame) -> list[dict]:
    rows = (
        era_speeches.groupby("president", observed=True)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            speeches=("doc_name", "nunique"),
        )
        .reset_index()
        .sort_values(["first_year", "president"])
    )
    return [
        {
            "name": str(row.president),
            "years": (
                str(int(row.first_year))
                if row.first_year == row.last_year
                else f"{int(row.first_year)}–{int(row.last_year)}"
            ),
            "speeches": int(row.speeches),
            "color": PRESIDENT_COLORS[index % len(PRESIDENT_COLORS)],
        }
        for index, row in enumerate(rows.itertuples(index=False))
    ]


def _agenda(
    era_paragraphs: pd.DataFrame,
    presidents: list[dict],
    taxonomy: dict,
) -> dict:
    """Build a non-exclusive paragraph-presence agenda for every president.

    A paragraph contributes one full observation to every distinct assigned
    level-1 domain.  Shares therefore answer "in what percentage of this
    president's paragraphs did the domain appear?" and are intentionally
    non-additive.  Raw-word presence is retained only as a sensitivity receipt.
    """
    president_order = [row["name"] for row in presidents]
    speech_support = {
        row["name"]: int(row["speeches"]) for row in presidents
    }
    supported_presidents = [
        president
        for president in president_order
        if speech_support[president] >= AGENDA_MIN_SPEECHES
    ]
    thin_presidents = [
        president
        for president in president_order
        if president not in supported_presidents
    ]
    if not supported_presidents:
        supported_presidents = list(president_order)
        thin_presidents = []

    level1_rows = taxonomy["level1"]
    level2_rows = taxonomy["level2"]
    domain_order = {
        str(row["name"]): index for index, row in enumerate(level1_rows)
    }
    domain_kind = {
        str(row["name"]): str(row["kind"]) for row in level1_rows
    }
    topic_domain = {
        str(row["name"]): str(row["level1"]) for row in level2_rows
    }
    known_topics = set(topic_domain)
    policy_rows = [
        row for row in level1_rows if row["kind"] == "policy"
    ]
    policy_domains = [str(row["name"]) for row in policy_rows]
    if set(AGENDA_DOMAIN_SHORT_LABELS) != set(policy_domains):
        raise ValueError(
            "agenda display mapping does not match taxonomy policy domains"
        )
    if set(topic_domain.values()) - set(domain_kind):
        raise ValueError("agenda topics reference unknown level-1 domains")

    category_specs = [
        {
            "key": str(row["name"]),
            "domain": str(row["name"]),
            "label": AGENDA_DOMAIN_SHORT_LABELS[str(row["name"])],
            "full_label": str(row["name"]),
            "kind": "policy",
            "color": AGENDA_DOMAIN_COLORS[str(row["name"])],
            "icon": topic_icon(str(row["name"])),
            "definition": str(row["definition"]),
        }
        for row in policy_rows
    ]
    category_by_key = {
        category["key"]: category for category in category_specs
    }
    compositions = []
    president_support = []
    for president in president_order:
        president_frame = era_paragraphs[
            era_paragraphs["president"].eq(president)
        ]
        paragraph_denominator = len(president_frame)
        word_denominator = float(president_frame["word_count"].sum())
        domain_counts = {domain: 0 for domain in domain_kind}
        domain_word_presence = {domain: 0.0 for domain in domain_kind}
        domain_sets = []
        domain_word_rows = []
        assigned_paragraphs = 0
        assigned_words = 0.0
        unassigned_words = 0.0
        for row in president_frame[["word_count", "topic_set"]].itertuples(
            index=False
        ):
            words = float(row.word_count)
            if words < 0:
                raise ValueError("agenda word counts must be nonnegative")
            topics = row.topic_set
            if not topics:
                unassigned_words += words
                domain_sets.append(frozenset())
                domain_word_rows.append((frozenset(), words))
                continue
            unknown = set(topics) - known_topics
            if unknown:
                raise ValueError(
                    f"agenda contains topics absent from taxonomy: "
                    f"{sorted(unknown)}"
                )
            assigned_paragraphs += 1
            assigned_words += words
            domains = {topic_domain[topic] for topic in topics}
            domain_sets.append(frozenset(domains))
            domain_word_rows.append((frozenset(domains), words))
            for domain in domains:
                domain_counts[domain] += 1
                domain_word_presence[domain] += words

        def paragraph_share(count: int) -> float:
            return (
                0.0
                if paragraph_denominator == 0
                else count / paragraph_denominator * 100
            )

        def word_share(words: float) -> float:
            return (
                0.0
                if word_denominator == 0
                else words / word_denominator * 100
            )

        top_domains = [
            domain
            for domain in sorted(
                policy_domains,
                key=lambda domain: (
                    -domain_counts[domain],
                    domain_order[domain],
                ),
            )
            if domain_counts[domain] > 0
        ][:5]
        word_top_domains = [
            domain
            for domain in sorted(
                policy_domains,
                key=lambda domain: (
                    -domain_word_presence[domain],
                    domain_order[domain],
                ),
            )
            if domain_word_presence[domain] > 0
        ][:5]
        all_policy_count = sum(
            bool(domains & set(policy_domains)) for domains in domain_sets
        )
        all_policy_word_presence = sum(
            words
            for domains, words in domain_word_rows
            if domains & set(policy_domains)
        )
        unassigned_paragraphs = paragraph_denominator - assigned_paragraphs

        domain_presence = [
            {
                **category,
                "count": int(domain_counts[category["key"]]),
                "denominator_paragraphs": paragraph_denominator,
                "share": float(
                    paragraph_share(domain_counts[category["key"]])
                ),
                "word_presence_words": float(
                    domain_word_presence[category["key"]]
                ),
                "word_denominator": float(word_denominator),
                "word_presence_share": float(
                    word_share(domain_word_presence[category["key"]])
                ),
            }
            for category in category_specs
        ]
        compositions.append({
            "president": president,
            "domain_presence": domain_presence,
            "top_policy_domains": list(top_domains),
            "word_weighted_top_policy_domains": list(word_top_domains),
            "leading_domain": (
                None
                if not top_domains
                else category_by_key[top_domains[0]]["label"]
            ),
            "all_policy": {
                "count": int(all_policy_count),
                "denominator_paragraphs": paragraph_denominator,
                "share": float(paragraph_share(all_policy_count)),
                "word_presence_words": float(all_policy_word_presence),
                "word_denominator": float(word_denominator),
                "word_presence_share": float(
                    word_share(all_policy_word_presence)
                ),
            },
            "unassigned": {
                "count": int(unassigned_paragraphs),
                "denominator_paragraphs": paragraph_denominator,
                "share": float(paragraph_share(unassigned_paragraphs)),
                "word_presence_words": float(unassigned_words),
                "word_denominator": float(word_denominator),
                "word_presence_share": float(word_share(unassigned_words)),
            },
            "weighting_sensitivity": {
                "leading_domain_same": (
                    top_domains[:1] == word_top_domains[:1]
                ),
                "top_five_set_same": (
                    set(top_domains) == set(word_top_domains)
                ),
            },
        })
        president_support.append({
            "president": president,
            "speeches": speech_support[president],
            "paragraphs": paragraph_denominator,
            "words": int(word_denominator),
            "assigned_paragraphs": int(assigned_paragraphs),
            "unassigned_paragraphs": int(unassigned_paragraphs),
            "assigned_words": int(assigned_words),
            "unassigned_words": int(unassigned_words),
            "status": (
                "supported"
                if president in supported_presidents
                else "thin"
            ),
        })

    for composition in compositions:
        domain_presence_by_key = {
            row["key"]: row for row in composition["domain_presence"]
        }
        segments = [
            {
                **domain_presence_by_key[domain],
                "president_rank": rank,
            }
            for rank, domain in enumerate(
                composition["top_policy_domains"], start=1
            )
        ]
        composition["segments"] = segments
        composition["displayed_share_sum"] = float(
            sum(segment["share"] for segment in segments)
        )

    supported_compositions = [
        row
        for row in compositions
        if row["president"] in supported_presidents
    ]
    return {
        "categories": category_specs,
        "compositions": compositions,
        "supported_presidents": supported_presidents,
        "thin_presidents": thin_presidents,
        "minimum_supported_speeches": AGENDA_MIN_SPEECHES,
        "president_support": president_support,
        "weighting_sensitivity": {
            "supported_records": len(supported_compositions),
            "same_leading_domain": sum(
                row["weighting_sensitivity"]["leading_domain_same"]
                for row in supported_compositions
            ),
            "same_top_five_set": sum(
                row["weighting_sensitivity"]["top_five_set_same"]
                for row in supported_compositions
            ),
            "comparison": "paragraph presence versus raw-word presence",
        },
        "measure": (
            "Share of all presidential paragraphs in which a policy domain "
            "appears. A multi-label paragraph counts once in every applicable "
            "domain, so the five displayed shares are independent, may "
            "overlap, and may sum above 100%."
        ),
        "selection": (
            "Each card shows the five policy domains appearing in the largest "
            "shares of that president's paragraphs. Bars are scaled within "
            "that president's card to emphasize the shape of the agenda; "
            "printed percentages preserve the measured values. Thin records "
            "remain visible as supporting evidence."
        ),
    }


def _governance(
    era_speeches: pd.DataFrame,
    presidents: list[dict],
) -> dict:
    audience_labels = {
        raw_value: label
        for label, raw_values in era_profiles.AUDIENCE_GROUPS
        for raw_value in raw_values
    }
    medium_labels = {
        raw_value: label
        for label, raw_values in era_profiles.MEDIUM_GROUPS
        for raw_value in raw_values
    }
    rows = []
    for president_row in presidents:
        president = president_row["name"]
        group = era_speeches[
            era_speeches["president"].eq(president)
        ].copy()
        group["audience_group"] = group["audience"].map(audience_labels)
        group["medium_group"] = group["medium"].map(medium_labels)
        if group[["audience_group", "medium_group"]].isna().any().any():
            missing = group.loc[
                group[["audience_group", "medium_group"]].isna().any(axis=1),
                ["audience", "medium"],
            ].drop_duplicates().to_dict("records")
            raise ValueError(
                f"era visualizations have unmapped communication routes for "
                f"{president}: {missing}"
            )
        route_counts = (
            group.groupby(
                ["audience_group", "medium_group"],
                observed=True,
            )
            .size()
            .rename("count")
            .reset_index()
            .sort_values(
                ["count", "audience_group", "medium_group"],
                ascending=[False, True, True],
            )
        )
        rows.append({
            "president": president,
            "speeches": len(group),
            "color": president_row["color"],
            "written_count": int(
                group["medium"].eq("written_message").sum()
            ),
            "written_share": float(
                group["medium"].eq("written_message").mean() * 100
            ),
            "congress_count": int(group["audience"].eq("congress").sum()),
            "congress_share": float(
                group["audience"].eq("congress").mean() * 100
            ),
            "audience": _communication_rows(
                group, era_profiles.AUDIENCE_GROUPS, "audience"
            ),
            "medium": _communication_rows(
                group, era_profiles.MEDIUM_GROUPS, "medium"
            ),
            "routes": [
                {
                    "audience": str(row.audience_group),
                    "medium": str(row.medium_group),
                    "count": int(row.count),
                    "share": float(row.count / len(group) * 100),
                }
                for row in route_counts.itertuples(index=False)
            ],
        })
    return {
        "audience_options": _communication_rows(
            era_speeches, era_profiles.AUDIENCE_GROUPS, "audience"
        ),
        "medium_options": _communication_rows(
            era_speeches, era_profiles.MEDIUM_GROUPS, "medium"
        ),
        "presidents": rows,
    }


def _adversaries(
    era_paragraphs: pd.DataFrame,
    era_entities: pd.DataFrame,
    presidents: list[dict],
) -> dict:
    president_order = [row["name"] for row in presidents]
    adversarial = era_entities[
        era_entities["stance"].eq("adversarial")
        & ~era_entities["entity"].isin(era_profiles.GENERIC_ADVERSARIES)
    ].copy()
    adversarial["display_entity"] = adversarial["entity"].map(
        era_profiles.ADVERSARY_ALIASES
    ).fillna(adversarial["entity"])
    normalized = adversarial.drop_duplicates(
        ["president", "display_entity", "doc_name", "para_idx"]
    )
    named_paragraphs = normalized.drop_duplicates(
        ["president", "doc_name", "para_idx"]
    )

    summaries = []
    edges = []
    for president in president_order:
        president_rows = normalized[
            normalized["president"].eq(president)
        ]
        president_named_paragraphs = named_paragraphs[
            named_paragraphs["president"].eq(president)
        ]
        total_words = int(
            era_paragraphs.loc[
                era_paragraphs["president"].eq(president), "word_count"
            ].sum()
        )
        adversary_words = int(
            president_named_paragraphs["word_count"].sum()
        )
        summaries.append({
            "president": president,
            "total_words": total_words,
            "adversary_words": adversary_words,
            "word_share": (
                0.0
                if total_words == 0
                else float(adversary_words / total_words * 100)
            ),
            "named_adversaries": int(
                president_rows["display_entity"].nunique()
            ),
            "named_paragraphs": int(len(president_named_paragraphs)),
        })
        counts = (
            president_rows.groupby("display_entity", observed=True)
            .size()
            .rename("paragraphs")
            .reset_index()
            .sort_values(
                ["paragraphs", "display_entity"],
                ascending=[False, True],
            )
        )
        selected_names = set(
            counts.head(3)["display_entity"]
        ) | set(
            counts.loc[counts["paragraphs"].ge(4), "display_entity"]
        )
        selected = counts[
            counts["display_entity"].isin(selected_names)
        ].head(5)
        for row in selected.itertuples(index=False):
            raw_entities = sorted(
                set(
                    president_rows.loc[
                        president_rows["display_entity"].eq(
                            row.display_entity
                        ),
                        "entity",
                    ].astype(str)
                )
            )
            edges.append({
                "president": president,
                "adversary": str(row.display_entity),
                "paragraphs": int(row.paragraphs),
                "raw_entities": raw_entities,
            })
    return {
        "edges": edges,
        "president_summary": summaries,
        "selection": (
            "Three leading nodes per president plus every node with at least "
            "four paragraphs, capped at five."
        ),
    }


def build_era_visualizations(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
    paragraph_annotations: pd.DataFrame,
    speech_annotations: pd.DataFrame,
    paragraph_entities: pd.DataFrame,
    taxonomy: dict,
) -> dict[str, dict]:
    """Derive all three reusable graphs for every canonical era."""
    speech_required = {"doc_name", "president", "year"}
    missing = sorted(speech_required - set(speeches.columns))
    if missing:
        raise ValueError(
            f"speeches is missing era-visualization columns: {missing}"
        )
    speech_frame = _strict_one_to_one_merge(
        speeches[sorted(speech_required)],
        speech_annotations[["doc_name", "audience", "medium"]],
        ["doc_name"],
        "era visualizations speeches x speech annotations",
    )
    speech_frame["era_key"] = era_profiles.story_era_key_series(
        speech_frame["year"], speech_frame["president"]
    )

    paragraph_frame = _strict_one_to_one_merge(
        paragraphs[["doc_name", "para_idx", "word_count"]],
        paragraph_annotations[["doc_name", "para_idx", "topics"]],
        ["doc_name", "para_idx"],
        "era visualizations paragraphs x paragraph annotations",
    )
    paragraph_frame = paragraph_frame.merge(
        speech_frame[["doc_name", "president", "year", "era_key"]],
        on="doc_name",
        how="left",
        validate="many_to_one",
    )
    if paragraph_frame[["president", "year"]].isna().any().any():
        raise ValueError(
            "era visualizations contain paragraphs with unknown speeches"
        )
    label_map = attention.canonical_label_map(taxonomy)
    paragraph_frame["topic_set"] = [
        frozenset(attention.normalize_topics(raw, label_map))
        for raw in paragraph_frame["topics"]
    ]

    entity_required = {"doc_name", "para_idx", "entity", "stance"}
    missing = sorted(entity_required - set(paragraph_entities.columns))
    if missing:
        raise ValueError(
            f"paragraph_entities is missing era-visualization columns: {missing}"
        )
    entity_frame = paragraph_entities[sorted(entity_required)].merge(
        paragraph_frame[
            [
                "doc_name", "para_idx", "president", "year", "era_key",
                "word_count",
            ]
        ],
        on=["doc_name", "para_idx"],
        how="left",
        validate="many_to_one",
    )
    if entity_frame["year"].isna().any():
        raise ValueError(
            "era visualizations contain entities with unknown paragraphs"
        )

    visualizations = {}
    for spec in era_profiles.ERA_PROFILE_SPECS:
        era_speeches = speech_frame[
            speech_frame["era_key"].eq(spec.key)
        ].copy()
        era_paragraphs = paragraph_frame[
            paragraph_frame["era_key"].eq(spec.key)
        ].copy()
        era_entities = entity_frame[
            entity_frame["era_key"].eq(spec.key)
        ].copy()
        if era_speeches.empty or era_paragraphs.empty:
            raise ValueError(
                f"era visualization {spec.key!r} has no corpus rows"
            )
        presidents = _president_rows(era_speeches)
        visualizations[spec.key] = {
            **asdict(spec),
            "years": f"{spec.start_year}–{spec.end_year}",
            "presidents": presidents,
            "agenda": _agenda(era_paragraphs, presidents, taxonomy),
            "governance": _governance(era_speeches, presidents),
            "adversaries": _adversaries(
                era_paragraphs, era_entities, presidents
            ),
            "support": {
                "speeches": len(era_speeches),
                "paragraphs": len(era_paragraphs),
            },
        }
    return visualizations


def load_all_era_visualizations() -> dict[str, dict]:
    """Load current artifacts and derive all reusable era graphs."""
    data_dir = corpus.DATA_DIR
    return build_era_visualizations(
        corpus.load(),
        pd.read_parquet(data_dir / "paragraphs.parquet"),
        pd.read_parquet(
            data_dir / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        pd.read_parquet(
            data_dir / "llm_annotations" / "speech_annotations.parquet"
        ),
        pd.read_parquet(
            data_dir / "llm_annotations" / "paragraph_entities.parquet"
        ),
        attention.load_taxonomy(),
    )


def load_era_visualization(era: str) -> dict:
    """Load one visualization record by stable era key or label."""
    spec = era_profiles.resolve_era(era)
    return load_all_era_visualizations()[spec.key]


def write_era_visualizations(
    visualizations: dict[str, dict],
    site_dir: Path,
) -> Path:
    """Publish deterministic graph inputs for era-template consumers."""
    path = site_dir / "data" / "era_visualizations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "era-visualizations-v8",
        "era_order": [
            spec.key for spec in era_profiles.ERA_PROFILE_SPECS
        ],
        "visualizations": visualizations,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    return path
