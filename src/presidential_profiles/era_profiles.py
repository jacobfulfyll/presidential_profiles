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
from pathlib import Path

import pandas as pd

from . import attention, corpus, trends, word_families


DISTINCTIVE_MIN_CORPUS_USES = 50
DISTINCTIVE_MIN_ERA_SPEECHES = 10
DISTINCTIVE_MIN_ERA_PRESIDENTS = 2
DISTINCTIVE_EVIDENCE_SCALE_USES = 100.0
DISTINCTIVE_EFFECT_CAP_RATIO = 25.0

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

FOUNDING_CONSTITUENCY_FALLBACK = (
    "The national public",
    "Commercial and creditor interests",
    "Western settlers and land claimants",
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
    paragraph_entities: pd.DataFrame,
    taxonomy: dict,
    constituency_claims: pd.DataFrame | None = None,
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

    entity_required = {"doc_name", "para_idx", "entity", "type", "stance"}
    missing = sorted(entity_required - set(paragraph_entities.columns))
    if missing:
        raise ValueError(
            f"paragraph_entities is missing era-profile columns: {missing}"
        )
    entity_frame = paragraph_entities[sorted(entity_required)].merge(
        paragraph_frame[
            ["doc_name", "para_idx", "president", "year", "era_key"]
        ],
        on=["doc_name", "para_idx"],
        how="left",
        validate="many_to_one",
    )
    if entity_frame["year"].isna().any():
        raise ValueError("era profiles contain entities with unknown paragraphs")
    entity_frame["display_entity"] = entity_frame["entity"].map(
        ADVERSARY_ALIASES
    ).fillna(entity_frame["entity"])
    adversarial = entity_frame[
        entity_frame["stance"].eq("adversarial")
        & ~entity_frame["entity"].isin(GENERIC_ADVERSARIES)
    ].copy()
    corpus_type_counts = paragraph_entities["type"].value_counts()

    claim_frame = None
    if constituency_claims is not None and not constituency_claims.empty:
        claim_required = {
            "doc_name", "para_idx", "outcome", "normalized_group"
        }
        missing = sorted(claim_required - set(constituency_claims.columns))
        if missing:
            raise ValueError(
                f"constituency_claims is missing era-profile columns: {missing}"
            )
        claim_frame = constituency_claims[sorted(claim_required)].merge(
            paragraph_frame[["doc_name", "para_idx", "year", "era_key"]],
            on=["doc_name", "para_idx"],
            how="left",
            validate="many_to_one",
        )
        if claim_frame["year"].isna().any():
            raise ValueError(
                "era profiles contain constituency claims with unknown paragraphs"
            )

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

        president_rows = (
            era_speeches.groupby("president", observed=True)
            .agg(
                first_year=("year", "min"),
                last_year=("year", "max"),
                speeches=("doc_name", "nunique"),
            )
            .reset_index()
            .sort_values(["first_year", "president"])
        )
        presidents = [
            {
                "name": str(row.president),
                "years": (
                    str(int(row.first_year))
                    if row.first_year == row.last_year
                    else f"{int(row.first_year)}–{int(row.last_year)}"
                ),
                "speeches": int(row.speeches),
            }
            for row in president_rows.itertuples(index=False)
        ]

        era_adversarial = adversarial[
            adversarial["era_key"].eq(spec.key)
        ].copy()
        type_rows = (
            era_adversarial.groupby(["display_entity", "type"], observed=True)
            .size()
            .rename("type_mentions")
            .reset_index()
            .sort_values(
                ["display_entity", "type_mentions", "type"],
                ascending=[True, False, True],
            )
            .drop_duplicates("display_entity")
        )
        entity_counts = (
            era_adversarial.drop_duplicates(
                ["display_entity", "doc_name", "para_idx"]
            )
            .groupby("display_entity", observed=True)
            .size()
            .rename("paragraphs")
            .reset_index()
            .sort_values(
                ["paragraphs", "display_entity"], ascending=[False, True]
            )
            .head(5)
            .merge(
                type_rows[["display_entity", "type"]],
                on="display_entity",
                how="left",
                validate="one_to_one",
            )
        )
        adversaries = [
            {
                "name": str(row.display_entity),
                "type": str(row.type),
                "paragraphs": int(row.paragraphs),
            }
            for row in entity_counts.itertuples(index=False)
        ]
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

        constituents = []
        constituency_status = "pending"
        constituency_note = "No promoted constituency claims yet"
        if claim_frame is not None:
            claims = claim_frame[
                claim_frame["era_key"].eq(spec.key)
                & claim_frame["outcome"].eq("claim")
                & claim_frame["normalized_group"].notna()
            ].copy()
            claims["normalized_group"] = (
                claims["normalized_group"].astype(str).str.strip()
            )
            claims = claims[claims["normalized_group"].ne("")]
            if not claims.empty:
                claim_counts = (
                    claims.drop_duplicates(
                        ["normalized_group", "doc_name", "para_idx"]
                    )
                    .groupby("normalized_group", observed=True)
                    .size()
                    .rename("paragraphs")
                    .reset_index()
                    .sort_values(
                        ["paragraphs", "normalized_group"],
                        ascending=[False, True],
                    )
                    .head(5)
                )
                constituents = [
                    {
                        "name": str(row.normalized_group),
                        "paragraphs": int(row.paragraphs),
                    }
                    for row in claim_counts.itertuples(index=False)
                ]
                constituency_status = "artifact"
                constituency_note = "Promoted constituency claims"
        if not constituents and spec.key == "founding":
            constituents = [
                {"name": name, "paragraphs": None}
                for name in FOUNDING_CONSTITUENCY_FALLBACK
            ]
            constituency_status = "fallback"
            constituency_note = "Illustrative only · promoted claims pending"

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
            "constituents": constituents,
            "constituency_status": constituency_status,
            "constituency_note": constituency_note,
            "adversaries": adversaries,
            "adversary_types": adversary_types,
            "footprint": {
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
            "topic_note": (
                "Ranked by Founding-era paragraph share · labels overlap"
                if spec.key == "founding"
                else "Ranked by era paragraph share · labels overlap"
            ),
            "distinctive_words": distinctive_words,
            "distinctive_eligibility": eligibility,
            "style": {
                "audience": max(audience, key=lambda row: row["count"]),
                "medium": max(medium, key=lambda row: row["count"]),
                "audience_options": audience,
                "medium_options": medium,
            },
        }
    return profiles


def _load_current_constituency_claims() -> pd.DataFrame | None:
    materialized = corpus.DATA_DIR / "annotation_ledger" / "materialized"
    pointer = materialized / "current"
    if not pointer.exists():
        return None
    generation = pointer.read_text().strip()
    if len(generation) != 64 or any(
        character not in "0123456789abcdef" for character in generation
    ):
        raise ValueError("annotation-ledger current generation is not a SHA-256 id")
    path = materialized / "generations" / generation / "constituency_claims.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_all_era_profiles() -> dict[str, dict]:
    """Load current artifacts and derive every era profile."""
    data_dir = corpus.DATA_DIR
    return build_era_profiles(
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
        _load_current_constituency_claims(),
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
        "schema_version": "era-profile-v4",
        "profile_order": [spec.key for spec in ERA_PROFILE_SPECS],
        "profiles": profiles,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return path
