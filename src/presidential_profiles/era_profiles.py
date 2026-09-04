"""Reusable era-profile data contract for every chronological story band.

The public entry points deliberately separate loading from derivation:

* ``load_all_era_profiles()`` reads the current local artifacts and returns all
  nine records.
* ``load_era_profile("expansion")`` is the one-argument convenience path.
* ``build_era_profiles(...)`` accepts already-loaded tables for the site build
  and tests, avoiding duplicate I/O.

No profile value is read from generated HTML. Every row is re-derived from the
keyed corpus and frozen annotation artifacts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from . import attention, corpus, trends, word_families

if TYPE_CHECKING:
    from .story_foundation import StoryFoundationBundle


DISTINCTIVE_MIN_CORPUS_USES = 50
DISTINCTIVE_MIN_ERA_SPEECHES = 10
DISTINCTIVE_MIN_ERA_PRESIDENTS = 2
DISTINCTIVE_EVIDENCE_SCALE_USES = 100.0
DISTINCTIVE_EFFECT_CAP_RATIO = 25.0

REFERENCE_LANDSCAPE_TYPES = (
    ("person", "People"),
    ("institution", "Institutions"),
    ("group", "Groups or communities"),
    ("nation", "Nations or places"),
)
REFERENCE_LANDSCAPE_MIN_PARAGRAPHS = 5
REFERENCE_LANDSCAPE_LIMITED_MIN_PARAGRAPHS = 4
REFERENCE_LANDSCAPE_MIN_DOCUMENTS = 2
REFERENCE_LANDSCAPE_MIN_NON_ADVERSARIAL_SHARE = 0.80
REFERENCE_LANDSCAPE_PRIOR = 0.5

ADVERSARY_TYPE_STYLES = {
    "nation": ("Nation or state", "#315f78"),
    "person": ("Person", "#a24f52"),
    "group": ("Group", "#4f7557"),
    "institution": ("Institution", "#9a7133"),
    "other": ("Other", "#74695f"),
}

ADVERSARY_ALIASES = {
    "Britain": "Great Britain",
    "British": "Great Britain",
    "British Government": "Great Britain",
    "British cabinet": "Great Britain",
    "British commander": "Great Britain",
    "British commanders": "Great Britain",
    "British naval force": "Great Britain",
    "British squadron": "Great Britain",
    "British vessel": "Great Britain",
    "French Republic": "France",
    "Executive Directory": "France",
    "French Directory": "France",
    "the Directory": "France",
    "Barbary Powers": "Tripoli / Barbary states",
    "Barbary states": "Tripoli / Barbary states",
    "Tripoli": "Tripoli / Barbary states",
    "Tripoline cruisers": "Tripoli / Barbary states",
    "western Pennsylvania insurgents": "Pennsylvania insurgents",
}

GENERIC_ADVERSARIES = frozenset({
    "the enemy",
    "belligerent powers",
    "faction",
    "insurgents",
    "foreign and domestic factions",
    "foreign influence",
    "spirit of party",
    "the savages",
})

AUDIENCE_GROUPS = (
    ("Congress", ("congress",)),
    ("General public", ("general_public",)),
    ("Specific organizations or groups", ("specific_organization_or_group",)),
    ("Other audiences", ("press", "foreign_or_diplomatic", "military", "other")),
)

MEDIUM_GROUPS = (
    ("Written message", ("written_message",)),
    ("Spoken address", ("spoken_address",)),
    ("Radio/TV broadcast", ("broadcast_radio_or_tv",)),
    (
        "Press conference/debate or other performed form",
        ("press_conference", "debate"),
    ),
)

FOUNDING_FINANCE_TOPICS = (
    "Public Debt, Revenue & Treasury Finance",
    "Taxes, Budget Deficits & Federal Spending",
)

FOUNDING_MAJOR_TOPIC_GROUPS = (
    (
        "Executive Power, Vetoes & the Courts",
        "Executive power & courts",
        ("Executive Power, Vetoes & the Courts",),
    ),
    ("Public finance", "Public finance", FOUNDING_FINANCE_TOPICS),
    (
        "Military Preparedness, Armed Forces & Veterans",
        "Military preparedness",
        ("Military Preparedness, Armed Forces & Veterans",),
    ),
    (
        "Crime, Insurrection & Federal Law Enforcement",
        "Federal law enforcement",
        ("Crime, Insurrection & Federal Law Enforcement",),
    ),
    (
        "Indian Affairs, Removal & Allotment",
        "Indian & Native affairs",
        ("Indian Affairs, Removal & Allotment",),
    ),
    (
        "Early Naval Wars: Barbary & the War of 1812",
        "Barbary & War of 1812",
        ("Early Naval Wars: Barbary & the War of 1812",),
    ),
)

@dataclass(frozen=True)
class EraProfileSpec:
    key: str
    label: str
    start_year: int
    end_year: int
    section_key: str
    title: str
    subtitle: str


ERA_PROFILE_SPECS = (
    EraProfileSpec(
        "founding",
        "Establishing the republic",
        1789,
        1808,
        "written_republic",
        "Establishing the Republic",
        "Sovereignty, federal capacity, and expansion in a new constitutional order.",
    ),
    EraProfileSpec(
        "expansion",
        "The continental republic",
        1809,
        1849,
        "expansion_conflict",
        "The Continental Republic",
        "Diplomacy, state-building, and continental conquest.",
    ),
    EraProfileSpec(
        "civil-war-reconstruction",
        "Sectional crisis & constitutional rupture",
        1850,
        1868,
        "union_crisis",
        "Sectional Crisis & Constitutional Rupture",
        "From the sectional break through war, emancipation, and the Fourteenth Amendment.",
    ),
    EraProfileSpec(
        "gilded-age",
        "Administrative-industrial order",
        1869,
        1912,
        "procedural_presidency",
        "The Administrative-Industrial Order",
        "Grant's accession opens a long regime of reconstruction, administration, and industrial power.",
    ),
    EraProfileSpec(
        "progressives-depression",
        "Reform, world war & collapse",
        1913,
        1932,
        "progressive_transform",
        "Reform, World War & Collapse",
        "A reform presidency gives way to world war, interwar government, and economic collapse.",
    ),
    EraProfileSpec(
        "war-new-deal",
        "New Deal, world war & settlement",
        1933,
        1952,
        "recovery_mobilization",
        "New Deal, World War & Settlement",
        "Recovery, mobilization, and the first postwar settlement share one durable governing regime.",
    ),
    EraProfileSpec(
        "cold-war",
        "Mature Cold War",
        1953,
        1980,
        "broadcast_presidency",
        "The Mature Cold War Presidency",
        "Cold War institutions and broadcast performance become ordinary presidential conditions.",
    ),
    EraProfileSpec(
        "post-cold-war",
        "Conservative turn & always-on presidency",
        1981,
        2016,
        "always_on",
        "The Conservative Turn & Always-On Presidency",
        "The Reagan transition precedes cable, permanent campaigning, and digital communication.",
    ),
    EraProfileSpec(
        "present",
        "Platform-era intensification",
        2017,
        2026,
        "platform_dominance",
        "Platform-Era Intensification",
        "People, scale, adversaries, and topics in this corpus.",
    ),
)

STORY_ERAS = tuple(
    (spec.label, spec.start_year, spec.end_year)
    for spec in ERA_PROFILE_SPECS
)

if STORY_ERAS[0][1] != 1789 or STORY_ERAS[-1][2] != 2026:
    raise ValueError("story eras must span the complete 1789–2026 corpus")
for previous, current in zip(ERA_PROFILE_SPECS, ERA_PROFILE_SPECS[1:]):
    if previous.end_year + 1 != current.start_year:
        raise ValueError("story eras must form one contiguous calendar partition")

_SPECS_BY_KEY = {spec.key: spec for spec in ERA_PROFILE_SPECS}
_SPECS_BY_LABEL = {spec.label: spec for spec in ERA_PROFILE_SPECS}
_LEGACY_LABEL_TO_KEY = {
    "The founding": "founding",
    "Expansion": "expansion",
    "War, institutions & expansion": "expansion",
    "Civil War & Reconstruction": "civil-war-reconstruction",
    "The Gilded Age": "gilded-age",
    "Progressives & Depression": "progressives-depression",
    "War & New Deal": "war-new-deal",
    "The Cold War": "cold-war",
    "Post-Cold War": "post-cold-war",
    "The present era": "present",
}

# Calendar years remain the public chronological axis.  At accession
# boundaries, however, a speech by the outgoing president belongs to the
# regime that presidency is closing.  This is a story-layer ownership rule,
# not a rewrite of a speech's date and not a change to the boundary analysis.
STORY_OUTGOING_REGIME_OVERRIDES = {
    (1809, "Thomas Jefferson"): "founding",
    (1869, "Andrew Johnson"): "civil-war-reconstruction",
    (1913, "William Taft"): "gilded-age",
    (1933, "Herbert Hoover"): "progressives-depression",
    (1953, "Harry S. Truman"): "war-new-deal",
    (1981, "Jimmy Carter"): "cold-war",
    (2017, "Barack Obama"): "post-cold-war",
}


def resolve_era(era: str) -> EraProfileSpec:
    """Resolve the stable key or canonical label for one profile."""
    if era in _SPECS_BY_KEY:
        return _SPECS_BY_KEY[era]
    if era in _SPECS_BY_LABEL:
        return _SPECS_BY_LABEL[era]
    if era in _LEGACY_LABEL_TO_KEY:
        return _SPECS_BY_KEY[_LEGACY_LABEL_TO_KEY[era]]
    choices = ", ".join(spec.key for spec in ERA_PROFILE_SPECS)
    raise KeyError(f"unknown era {era!r}; choose one of: {choices}")


def story_era_for_year(year: int) -> EraProfileSpec:
    """Return the reviewed chronological-story era containing ``year``."""
    for spec in ERA_PROFILE_SPECS:
        if spec.start_year <= int(year) <= spec.end_year:
            return spec
    raise ValueError(f"year {year!r} falls outside the story era axis")


def story_era_for_speech(year: int, president: str) -> EraProfileSpec:
    """Return one speech's governed story era without changing its date."""
    override = STORY_OUTGOING_REGIME_OVERRIDES.get(
        (int(year), str(president).strip())
    )
    if override is not None:
        return resolve_era(override)
    return story_era_for_year(year)


def story_era_key_series(
    years: pd.Series,
    presidents: pd.Series | None = None,
) -> pd.Series:
    """Map speech years, optionally with speakers, to stable story-era keys."""
    if presidents is None:
        return years.map(lambda year: story_era_for_year(int(year)).key)
    if not years.index.equals(presidents.index):
        raise ValueError("story era years and presidents must share an index")
    return pd.Series(
        [
            story_era_for_speech(year, president).key
            for year, president in zip(years, presidents)
        ],
        index=years.index,
        dtype="object",
    )


def story_era_series(
    years: pd.Series,
    presidents: pd.Series | None = None,
) -> pd.Series:
    """Map speeches to reviewed story-era labels."""
    keys = story_era_key_series(years, presidents)
    return keys.map(lambda key: _SPECS_BY_KEY[key].label)


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


def _major_topics(
    spec: EraProfileSpec,
    frame: pd.DataFrame,
) -> list[dict]:
    if spec.key == "founding":
        rows = []
        for topic, label, grouped_topics in FOUNDING_MAJOR_TOPIC_GROUPS:
            wanted = frozenset(grouped_topics)
            count = int(
                frame["topic_set"].map(
                    lambda assigned: not assigned.isdisjoint(wanted)
                ).sum()
            )
            rows.append({
                "topic": topic,
                "label": label,
                "topics": list(grouped_topics),
                "count": count,
                "share": float(count / len(frame) * 100),
            })
    else:
        counts: Counter[str] = Counter(
            topic
            for assigned in frame["topic_set"]
            for topic in assigned
        )
        rows = [
            {
                "topic": topic,
                "label": topic,
                "topics": [topic],
                "count": int(count),
                "share": float(count / len(frame) * 100),
            }
            for topic, count in counts.items()
        ]
    return sorted(rows, key=lambda row: (-row["count"], row["topic"]))[:6]


def _president_name_terms(presidents) -> frozenset[str]:
    """Grouped word-family labels used in presidential display names.

    Distinctive vocabulary describes the era's language, not which presidents
    happen to be named in its speeches. Map every substantive name token
    through the same Explore family layer before excluding it from rank.
    """
    return frozenset(
        word_families.form_to_node(token)
        for president in presidents
        for token in word_families.tokenize(str(president))
        if len(token) > 2 and "'" not in token
    )


def _reference_landscape_inputs(
    foundation: "StoryFoundationBundle",
) -> dict[str, dict]:
    """Build audited type-presence rows and supported highlight candidates.

    Type presence is paragraph presence, so repeated mentions of one type in a
    paragraph count once. Candidate selection stays on primary-AI-backed rows;
    local NER is retained as source agreement and never supplies stance.
    """
    paragraph_key = ["doc_name", "para_idx"]
    allowed_types = {key for key, _ in REFERENCE_LANDSCAPE_TYPES}
    required = {
        *paragraph_key,
        "normalized_entity",
        "display_entity",
        "display_type",
        "ai_entity",
        "ai_stance",
        "ner_text",
        "analysis_eligible",
        "story_era_key",
        "source_badge",
        "source_agreement",
        "text",
        "attributed_speaker",
        "document_owner",
        "cross_owner_paragraph",
        "title",
        "source_url",
        "year",
    }
    missing = sorted(required - set(foundation.entity_mentions.columns))
    if missing:
        raise ValueError(
            "foundation entity mentions are missing reference-landscape "
            f"columns: {missing}"
        )
    eligible = foundation.paragraph_view[
        foundation.paragraph_view["analysis_eligible"]
    ][paragraph_key + ["story_era_key"]].copy()
    if eligible.duplicated(paragraph_key).any():
        raise ValueError("reference-landscape eligible paragraphs are not unique")
    denominators = eligible.groupby("story_era_key", observed=True).size()
    expected_eras = [spec.key for spec in ERA_PROFILE_SPECS]
    if set(denominators.index) != set(expected_eras):
        raise ValueError("reference-landscape era denominators are incomplete")

    entities = foundation.entity_mentions[sorted(required)].copy()
    entities = entities[
        entities["analysis_eligible"]
        & entities["ai_entity"].notna()
        & entities["display_type"].isin(allowed_types)
    ].copy()
    entities = entities.drop_duplicates(
        ["story_era_key", "normalized_entity", *paragraph_key]
    )
    if entities.empty:
        raise ValueError("reference landscape has no primary-AI-backed entities")

    type_presence = entities.drop_duplicates(
        ["story_era_key", "display_type", *paragraph_key]
    )
    era_type_counts = type_presence.groupby(
        ["story_era_key", "display_type"], observed=True
    ).size()
    corpus_type_counts = type_presence.groupby("display_type", observed=True).size()
    corpus_denominator = len(eligible)
    if corpus_denominator != int(denominators.sum()):
        raise ValueError("reference-landscape denominators do not reconcile")

    entity_totals = entities.groupby("normalized_entity", observed=True).size()
    candidates: dict[str, list[dict]] = {key: [] for key in expected_eras}
    for (era_key, normalized_entity), rows in entities.groupby(
        ["story_era_key", "normalized_entity"], observed=True, sort=False
    ):
        era_paragraphs = len(rows)
        source_documents = int(rows["doc_name"].nunique())
        other_paragraphs = int(entity_totals[normalized_entity]) - era_paragraphs
        era_denominator = int(denominators[era_key])
        other_denominator = corpus_denominator - era_denominator
        log_odds = math.log(
            (era_paragraphs + REFERENCE_LANDSCAPE_PRIOR)
            / (era_denominator - era_paragraphs + REFERENCE_LANDSCAPE_PRIOR)
        ) - math.log(
            (other_paragraphs + REFERENCE_LANDSCAPE_PRIOR)
            / (other_denominator - other_paragraphs + REFERENCE_LANDSCAPE_PRIOR)
        )
        stance_counts = rows["ai_stance"].value_counts()
        favorable = int(stance_counts.get("favorable", 0))
        neutral = int(stance_counts.get("neutral", 0))
        adversarial = int(stance_counts.get("adversarial", 0))
        non_adversarial_share = (favorable + neutral) / era_paragraphs
        if (
            era_paragraphs < REFERENCE_LANDSCAPE_LIMITED_MIN_PARAGRAPHS
            or source_documents < REFERENCE_LANDSCAPE_MIN_DOCUMENTS
            or log_odds <= 0
            or non_adversarial_share
            < REFERENCE_LANDSCAPE_MIN_NON_ADVERSARIAL_SHARE
        ):
            continue
        type_counts = rows["display_type"].value_counts()
        entity_type = sorted(
            type_counts[type_counts.eq(type_counts.max())].index.astype(str)
        )[0]
        display_counts = rows["display_entity"].value_counts()
        label = sorted(
            display_counts[display_counts.eq(display_counts.max())].index.astype(str)
        )[0]
        evidence_rows = rows[rows["ai_stance"].isin(["favorable", "neutral"])].copy()
        evidence_rows = evidence_rows.sort_values(
            ["source_agreement", "year", "doc_name", "para_idx"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        evidence = evidence_rows.iloc[0]
        candidates[str(era_key)].append({
            "normalized_entity": str(normalized_entity),
            "label": label,
            "entity_type": entity_type,
            "support_status": (
                "supported"
                if era_paragraphs >= REFERENCE_LANDSCAPE_MIN_PARAGRAPHS
                else "limited_record"
            ),
            "support": {
                "paragraphs": era_paragraphs,
                "source_documents": source_documents,
            },
            "stance_mix": {
                "favorable": favorable,
                "neutral": neutral,
                "adversarial": adversarial,
            },
            "source_agreement": {
                "badge": str(evidence["source_badge"]),
                "ai_ner": bool(evidence["source_agreement"]),
            },
            "evidence": {
                "doc_name": str(evidence["doc_name"]),
                "para_idx": int(evidence["para_idx"]),
                "title": str(evidence["title"]),
                "source_url": str(evidence["source_url"]),
                "actual_speaker": str(evidence["attributed_speaker"]),
                "document_owner": str(evidence["document_owner"]),
                "cross_owner": bool(evidence["cross_owner_paragraph"]),
                "excerpt": str(evidence["text"]),
                "ai_mention": str(evidence["ai_entity"]),
                "ner_mention": (
                    None
                    if pd.isna(evidence["ner_text"])
                    else str(evidence["ner_text"])
                ),
                "ai_stance": str(evidence["ai_stance"]),
            },
            "_selection": {
                "log_odds": float(log_odds),
                "non_adversarial_share": float(non_adversarial_share),
            },
        })

    output: dict[str, dict] = {}
    for era_key in expected_eras:
        rows = []
        for entity_type, label in REFERENCE_LANDSCAPE_TYPES:
            era_count = int(era_type_counts.get((era_key, entity_type), 0))
            corpus_count = int(corpus_type_counts.get(entity_type, 0))
            rows.append({
                "entity_type": entity_type,
                "label": label,
                "paragraphs": era_count,
                "paragraph_share": float(era_count / denominators[era_key]),
                "corpus_paragraphs": corpus_count,
                "corpus_paragraph_share": float(
                    corpus_count / corpus_denominator
                ),
            })
        output[era_key] = {
            "denominator": {
                "unit": "speaker_audited_paragraphs",
                "paragraphs": int(denominators[era_key]),
                "corpus_paragraphs": corpus_denominator,
            },
            "types": rows,
            "candidates": candidates[era_key],
        }
    return output


def _project_reference_landscape(
    base: dict,
    named_adversaries: set[str],
) -> dict:
    """Select at most one non-adversarial highlight for each reference type."""
    candidates = [
        row for row in base["candidates"]
        if row["normalized_entity"] not in named_adversaries
    ]
    output_rows = []
    for type_row in base["types"]:
        type_candidates = [
            row for row in candidates
            if row["entity_type"] == type_row["entity_type"]
        ]
        type_candidates.sort(
            key=lambda row: (
                -row["_selection"]["log_odds"],
                -row["support"]["paragraphs"],
                -row["support"]["source_documents"],
                row["normalized_entity"],
            )
        )
        highlight = None
        supported = [
            row for row in type_candidates
            if row["support_status"] == "supported"
        ]
        limited = [
            row for row in type_candidates
            if row["support_status"] == "limited_record"
        ]
        selected = supported[0] if supported else (limited[0] if limited else None)
        if selected is not None:
            highlight = {
                key: value
                for key, value in selected.items()
                if key != "_selection"
            }
        output_rows.append({**type_row, "highlight": highlight})
    return {
        "definition": (
            "Paragraph presence for primary-AI-backed people, institutions, "
            "groups, and nations or places in actual-president paragraphs."
        ),
        "denominator": dict(base["denominator"]),
        "selection": {
            "minimum_paragraphs": REFERENCE_LANDSCAPE_MIN_PARAGRAPHS,
            "limited_record_minimum_paragraphs": (
                REFERENCE_LANDSCAPE_LIMITED_MIN_PARAGRAPHS
            ),
            "limited_record_only_when_no_supported_highlight": True,
            "minimum_source_documents": REFERENCE_LANDSCAPE_MIN_DOCUMENTS,
            "minimum_favorable_or_neutral_share": (
                REFERENCE_LANDSCAPE_MIN_NON_ADVERSARIAL_SHARE
            ),
            "excludes_displayed_named_adversaries": True,
            "one_highlight_per_type": True,
            "ner_supplies_stance": False,
        },
        "types": output_rows,
    }


def _distinctive_words(
    spec: EraProfileSpec,
    era_speeches: pd.DataFrame,
    era_counts: Counter,
    other_counts: Counter,
    excluded_terms: frozenset[str],
) -> tuple[list[dict], dict]:
    n_presidents = int(era_speeches["president"].nunique())
    min_presidents = min(DISTINCTIVE_MIN_ERA_PRESIDENTS, n_presidents)
    scores = trends.log_odds_scores(
        era_counts,
        other_counts,
        min_count=DISTINCTIVE_MIN_CORPUS_USES,
    )
    scores = scores[
        ~scores["term"].isin(trends.REGISTER_WORDS | excluded_terms)
    ].copy()
    candidates = frozenset(scores["term"])
    speech_counts: Counter[str] = Counter()
    president_sets = {term: set() for term in candidates}
    for row in era_speeches[["president", "transcript"]].itertuples(index=False):
        present = trends.grouped_word_set(row.transcript) & candidates
        speech_counts.update(present)
        for term in present:
            president_sets[term].add(str(row.president))
    scores["era_speech_count"] = scores["term"].map(speech_counts)
    scores["era_president_count"] = scores["term"].map(
        lambda term: len(president_sets[term])
    )
    scores = scores[
        scores["era_speech_count"].ge(DISTINCTIVE_MIN_ERA_SPEECHES)
        & scores["era_president_count"].ge(min_presidents)
    ]
    if len(scores) < 3:
        raise ValueError(
            f"era profile {spec.key!r} has fewer than three eligible "
            "distinctive words"
        )
    scores["era_count"] = scores["term"].map(era_counts)
    scores["evidence_weight"] = scores["era_count"].map(
        lambda count: 1.0 - math.exp(
            -float(count) / DISTINCTIVE_EVIDENCE_SCALE_USES
        )
    )
    scores["capped_effect"] = scores["delta"].clip(
        lower=0.0,
        upper=math.log(DISTINCTIVE_EFFECT_CAP_RATIO),
    )
    scores["distinctiveness_score"] = (
        scores["capped_effect"] * scores["evidence_weight"]
    )
    scores = scores.sort_values(
        ["distinctiveness_score", "delta", "z", "term"],
        ascending=[False, False, False, True],
    )
    era_total = sum(era_counts.values())
    other_total = sum(other_counts.values())
    words = []
    for rank, row in enumerate(
        scores.head(3).itertuples(index=False), start=1
    ):
        term = str(row.term)
        era_rate = era_counts[term] / era_total * 10_000
        other_rate = other_counts[term] / other_total * 10_000
        words.append({
            "rank": rank,
            "term": term,
            "era_count": int(era_counts[term]),
            "other_count": int(other_counts[term]),
            "era_speech_count": int(row.era_speech_count),
            "era_president_count": int(row.era_president_count),
            "era_rate_per_10000": float(era_rate),
            "other_rate_per_10000": float(other_rate),
            "rate_ratio": (
                None if other_rate == 0 else float(era_rate / other_rate)
            ),
            "log_odds_delta": float(row.delta),
            "evidence_weight": float(row.evidence_weight),
            "distinctiveness_score": float(row.distinctiveness_score),
            "z": float(row.z),
        })
    eligibility = {
        "min_corpus_uses": DISTINCTIVE_MIN_CORPUS_USES,
        "min_era_speeches": DISTINCTIVE_MIN_ERA_SPEECHES,
        "min_era_presidents": min_presidents,
        "evidence_scale_uses": int(DISTINCTIVE_EVIDENCE_SCALE_USES),
        "effect_cap_ratio": int(DISTINCTIVE_EFFECT_CAP_RATIO),
        "excludes_president_name_terms": True,
    }
    return words, eligibility


def build_era_profiles(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
    paragraph_annotations: pd.DataFrame,
    speech_annotations: pd.DataFrame,
    taxonomy: dict,
    foundation: "StoryFoundationBundle",
) -> dict[str, dict]:
    """Derive all nine era profiles from keyed source artifacts."""
    speech_required = {"doc_name", "president", "year", "word_count", "transcript"}
    missing = sorted(speech_required - set(speeches.columns))
    if missing:
        raise ValueError(f"speeches is missing era-profile columns: {missing}")
    speech_frame = _strict_one_to_one_merge(
        speeches[sorted(speech_required)],
        speech_annotations[["doc_name", "audience", "medium"]],
        ["doc_name"],
        "era profiles speeches x speech annotations",
    )
    speech_frame["era_key"] = story_era_key_series(
        speech_frame["year"], speech_frame["president"]
    )

    paragraph_frame = _strict_one_to_one_merge(
        paragraphs[["doc_name", "para_idx", "word_count"]],
        paragraph_annotations[["doc_name", "para_idx", "topics"]],
        ["doc_name", "para_idx"],
        "era profiles paragraphs x paragraph annotations",
    )
    paragraph_frame = paragraph_frame.merge(
        speech_frame[["doc_name", "president", "year", "era_key"]],
        on="doc_name",
        how="left",
        validate="many_to_one",
    )
    if paragraph_frame[["president", "year"]].isna().any().any():
        raise ValueError("era profiles contain paragraphs with unknown speeches")
    label_map = attention.canonical_label_map(taxonomy)
    paragraph_frame["topic_set"] = [
        frozenset(attention.normalize_topics(raw, label_map))
        for raw in paragraph_frame["topics"]
    ]

    entity_required = {
        "doc_name", "para_idx", "normalized_entity", "display_entity",
        "display_type", "ai_entity", "ai_stance", "analysis_eligible",
        "story_era_key",
    }
    missing = sorted(entity_required - set(foundation.entity_mentions.columns))
    if missing:
        raise ValueError(
            f"foundation entity mentions are missing era-profile columns: {missing}"
        )
    generic = {value.casefold() for value in GENERIC_ADVERSARIES}
    entity_frame = foundation.entity_mentions[list(entity_required)].copy()
    adversarial = entity_frame[
        entity_frame["analysis_eligible"]
        & entity_frame["ai_entity"].notna()
        & entity_frame["ai_stance"].eq("adversarial")
        & ~entity_frame["normalized_entity"].isin(generic)
    ].copy()
    corpus_type_counts = adversarial["display_type"].value_counts()
    from .story_foundation import project_distinctive_references

    distinctive_references = project_distinctive_references(foundation)
    reference_landscape_inputs = _reference_landscape_inputs(foundation)

    counts_by_key = {}
    president_name_terms = _president_name_terms(
        speech_frame["president"].drop_duplicates()
    )
    for spec in ERA_PROFILE_SPECS:
        texts = speech_frame.loc[
            speech_frame["era_key"].eq(spec.key),
            "transcript",
        ]
        counts_by_key[spec.key] = trends.grouped_word_counts(texts)
    total_counts = sum(counts_by_key.values(), start=Counter())

    corpus_speeches = len(speech_frame)
    corpus_paragraphs = len(paragraph_frame)
    corpus_words = int(speech_frame["word_count"].sum())
    profiles = {}
    for spec in ERA_PROFILE_SPECS:
        era_speeches = speech_frame[
            speech_frame["era_key"].eq(spec.key)
        ].copy()
        era_paragraphs = paragraph_frame[
            paragraph_frame["era_key"].eq(spec.key)
        ].copy()
        if era_speeches.empty or era_paragraphs.empty:
            raise ValueError(f"era profile {spec.key!r} has no corpus rows")

        era_appearances = foundation.appearances[
            foundation.appearances["story_era_key"].eq(spec.key)
        ].copy()
        era_speaker_paragraphs = foundation.paragraph_view[
            foundation.paragraph_view["analysis_eligible"]
            & foundation.paragraph_view["story_era_key"].eq(spec.key)
        ].copy()
        president_rows = (
            era_appearances.groupby(
                ["attributed_speaker_profile_id", "attributed_speaker"],
                observed=True,
            )
            .agg(
                first_source_year=("year", "min"),
                last_source_year=("year", "max"),
                appearances=("doc_name", "size"),
                paragraphs=("n_paragraphs", "sum"),
                words=("n_words", "sum"),
                cross_owner_appearances=("cross_owner_appearance", "sum"),
            )
            .reset_index()
            .sort_values(["first_source_year", "attributed_speaker"])
        )
        presidents = [
            {
                "profile_id": str(row.attributed_speaker_profile_id),
                "name": str(row.attributed_speaker),
                "first_source_year": int(row.first_source_year),
                "last_source_year": int(row.last_source_year),
                "source_years": sorted(
                    int(value)
                    for value in era_appearances.loc[
                        era_appearances["attributed_speaker_profile_id"].eq(
                            row.attributed_speaker_profile_id
                        ),
                        "year",
                    ].unique()
                ),
                "appearances": int(row.appearances),
                "paragraphs": int(row.paragraphs),
                "words": int(row.words),
                "cross_owner_appearances": int(row.cross_owner_appearances),
                "cross_owner_paragraphs": int(
                    era_speaker_paragraphs[
                        era_speaker_paragraphs["attributed_speaker_profile_id"].eq(
                            row.attributed_speaker_profile_id
                        )
                        & era_speaker_paragraphs["cross_owner_paragraph"]
                    ].shape[0]
                ),
                "cross_owner_only": bool(
                    int(row.cross_owner_appearances) == int(row.appearances)
                ),
            }
            for row in president_rows.itertuples(index=False)
        ]

        era_adversarial = adversarial[
            adversarial["story_era_key"].eq(spec.key)
        ].copy()
        type_rows = (
            era_adversarial.groupby(
                ["normalized_entity", "display_entity", "display_type"],
                observed=True,
            )
            .size()
            .rename("type_mentions")
            .reset_index()
            .sort_values(
                ["normalized_entity", "type_mentions", "display_entity", "display_type"],
                ascending=[True, False, True, True],
            )
            .drop_duplicates("normalized_entity")
        )
        entity_counts = (
            era_adversarial.drop_duplicates(
                ["normalized_entity", "doc_name", "para_idx"]
            )
            .groupby("normalized_entity", observed=True)
            .size()
            .rename("paragraphs")
            .reset_index()
            .sort_values(
                ["paragraphs", "normalized_entity"], ascending=[False, True]
            )
            .head(5)
            .merge(
                type_rows[
                    ["normalized_entity", "display_entity", "display_type"]
                ],
                on="normalized_entity",
                how="left",
                validate="one_to_one",
            )
        )
        adversaries = [
            {
                "name": str(row.display_entity),
                "normalized_entity": str(row.normalized_entity),
                "type": str(row.display_type),
                "paragraphs": int(row.paragraphs),
            }
            for row in entity_counts.itertuples(index=False)
        ]
        reference_landscape = _project_reference_landscape(
            reference_landscape_inputs[spec.key],
            {row["normalized_entity"] for row in adversaries},
        )
        present_types = {row["type"] for row in adversaries}
        ordered_types = [
            entity_type
            for entity_type in ADVERSARY_TYPE_STYLES
            if entity_type in corpus_type_counts.index
        ]
        ordered_types.extend(sorted(
            str(entity_type)
            for entity_type in corpus_type_counts.index
            if entity_type not in ADVERSARY_TYPE_STYLES
        ))
        adversary_types = [
            {
                "type": entity_type,
                "label": ADVERSARY_TYPE_STYLES.get(
                    entity_type,
                    (entity_type.replace("_", " ").title(), "#74695f"),
                )[0],
                "color": ADVERSARY_TYPE_STYLES.get(
                    entity_type, ("Other", "#74695f")
                )[1],
                "corpus_mentions": int(corpus_type_counts[entity_type]),
                "present": entity_type in present_types,
            }
            for entity_type in ordered_types
        ]

        audience = _communication_rows(
            era_speeches, AUDIENCE_GROUPS, "audience"
        )
        medium = _communication_rows(era_speeches, MEDIUM_GROUPS, "medium")
        distinctive_words, eligibility = _distinctive_words(
            spec,
            era_speeches,
            counts_by_key[spec.key],
            total_counts - counts_by_key[spec.key],
            president_name_terms,
        )
        era_words = int(era_speeches["word_count"].sum())
        profiles[spec.key] = {
            **asdict(spec),
            "years": f"{spec.start_year}–{spec.end_year}",
            "presidents": presidents,
            "distinctive_references": distinctive_references[spec.key],
            "reference_landscape": reference_landscape,
            "adversaries": adversaries,
            "adversary_types": adversary_types,
            "footprint": {
                "unit": "source_document_corpus",
                "speeches": len(era_speeches),
                "corpus_speeches": corpus_speeches,
                "speech_share": float(
                    len(era_speeches) / corpus_speeches * 100
                ),
                "paragraphs": len(era_paragraphs),
                "corpus_paragraphs": corpus_paragraphs,
                "paragraph_share": float(
                    len(era_paragraphs) / corpus_paragraphs * 100
                ),
                "words": era_words,
                "corpus_words": corpus_words,
                "word_share": float(era_words / corpus_words * 100),
            },
            "major_topics": _major_topics(spec, era_paragraphs),
            "major_topics_receipt": {"unit": "source_document_corpus"},
            "topic_note": (
                "Ranked by Founding-era paragraph share · labels overlap"
                if spec.key == "founding"
                else "Ranked by era paragraph share · labels overlap"
            ),
            "distinctive_words": distinctive_words,
            "distinctive_words_receipt": {"unit": "source_document_corpus"},
            "distinctive_eligibility": eligibility,
            "style": {
                "unit": "source_document_corpus",
                "audience": max(audience, key=lambda row: row["count"]),
                "medium": max(medium, key=lambda row: row["count"]),
                "audience_options": audience,
                "medium_options": medium,
            },
        }
    return profiles


def load_all_era_profiles() -> dict[str, dict]:
    """Load current artifacts and derive every era profile."""
    data_dir = corpus.DATA_DIR
    from .story_foundation import load_story_foundation

    foundation = load_story_foundation()
    return build_era_profiles(
        corpus.load(),
        pd.read_parquet(data_dir / "paragraphs.parquet"),
        pd.read_parquet(
            data_dir / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        pd.read_parquet(
            data_dir / "llm_annotations" / "speech_annotations.parquet"
        ),
        attention.load_taxonomy(),
        foundation,
    )


def load_era_profile(era: str) -> dict:
    """One-argument convenience loader for a stable era key or label."""
    spec = resolve_era(era)
    return load_all_era_profiles()[spec.key]


def write_era_profiles(
    profiles: dict[str, dict],
    site_dir: Path,
) -> Path:
    """Publish a deterministic JSON payload for template switching."""
    path = site_dir / "data" / "era_profiles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "era-profile-v6",
        "profile_order": [spec.key for spec in ERA_PROFILE_SPECS],
        "profiles": profiles,
    }
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path
