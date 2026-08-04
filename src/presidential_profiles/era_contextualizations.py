"""Reusable data contract for the era-defined and era-echoes views.

The public story gives every chronological era the same graph structure:

* ``era_defined`` traces four focal-era topic families across the shared
  nine-era axis.  Founding and Continental Republic keep their historically
  declared families; later eras use the four leading era-profile topics.
* ``era_echoes`` follows the exact three-part path from the president making
  a reference, through the paragraph's assigned topic, to the president named.
  Separate directional networks power the ``Invoked by`` and ``Invoking``
  switch, so each view's geometry reflects only the selected direction.
* ``topic_life`` remains in the published data contract for research and
  compatibility, but it is not part of the four-view story workspace.

Every paragraph join is keyed on ``(doc_name, para_idx)``.  The public loader,
single-era selector, and JSON writer mirror :mod:`era_profiles` and
:mod:`era_visualizations`.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from . import attention, corpus, era_profiles, era_visualizations


TOPIC_LIFE_ANNOTATIONS_PATH = (
    corpus.DATA_DIR
    / "contextualization"
    / "topic_life_annotations_v1.json"
)
TOPIC_LIFE_ANNOTATION_SCHEMA = "topic-life-annotations-v1"
CONTEXTUALIZATION_SCHEMA = "era-contextualizations-v10"
REFRAME_MAX_SPEARMAN = -0.50
REFRAME_MIN_SOURCE_DECLINE_PP = 3.0
REFRAME_MIN_SUCCESSOR_GROWTH_PP = 1.0

FOUNDING_DEFINED_SPECS = (
    (
        "abroad",
        "⚓",
        "Exercise independence abroad",
        (
            "Treaties, Diplomacy & International Arbitration",
            "Neutral Rights & Maritime Depredations",
            "Early Naval Wars: Barbary & the War of 1812",
        ),
        "Defend treaty rights, neutral commerce, and national independence.",
        "#4f7557",
    ),
    (
        "native",
        "🪶",
        "Navigate Native nations and continental power",
        ("Indian Affairs, Removal & Allotment",),
        "Confront Native sovereignty, land, and continental power.",
        "#a24f52",
    ),
    (
        "union",
        "🏛️",
        "Hold the union together",
        (
            "Constitutional Union & Federalism",
            "Crime, Insurrection & Federal Law Enforcement",
            "Executive Power, Vetoes & the Courts",
        ),
        "Make constitutional authority hold at home.",
        "#315f78",
    ),
    (
        "finance",
        "💰",
        "Finance the government",
        (
            "Public Debt, Revenue & Treasury Finance",
            "Taxes, Budget Deficits & Federal Spending",
            "Coinage, Currency & Specie",
            "National Bank & Banking Crises",
            "Tariffs, Reciprocity & Navigation Laws",
        ),
        "Establish credit, revenue, money, and banking.",
        "#9a7133",
    ),
)

EXPANSION_DEFINED_PHASES = (
    (
        "war-survival",
        "War tests sovereignty",
        1809,
        1816,
        "Renewed war asks whether the republic can defend itself and endure.",
    ),
    (
        "postwar-order",
        "Building the postwar order",
        1817,
        1828,
        "Diplomacy, finance, and territorial administration become routine.",
    ),
    (
        "institutional-conflict",
        "Institutions become conflict",
        1829,
        1840,
        "Banking and federal authority become sustained domestic arguments.",
    ),
    (
        "continental-conquest",
        "Continental conquest",
        1841,
        1849,
        "Acquisition, war, removal, and settlement redraw continental power.",
    ),
)
EXPANSION_DEFINED_SERIES = (
    (
        "war",
        "War & military survival",
        "⚔",
        (
            "Military Preparedness, Armed Forces & Veterans",
            "Early Naval Wars: Barbary & the War of 1812",
            "Mexican War",
        ),
        "#a24f52",
    ),
    (
        "diplomacy",
        "Treaties & diplomacy",
        "⚓",
        ("Treaties, Diplomacy & International Arbitration",),
        "#4f7557",
    ),
    (
        "institutions",
        "Federal institutions",
        "🏛",
        (
            "National Bank & Banking Crises",
            "Constitutional Union & Federalism",
        ),
        "#315f78",
    ),
    (
        "territory",
        "Territory & continental power",
        "↦",
        (
            "Relations with Spain, Mexico & Territorial Claims",
            "Territorial Organization, Statehood & Insular Governance",
            "Mexican War",
            "Indian Affairs, Removal & Allotment",
            "Public Lands & Homestead Settlement",
        ),
        "#9a7133",
    ),
)
EXPANSION_EXTERNAL_TOPICS = (
    "Relations with Spain, Mexico & Territorial Claims",
    "Territorial Organization, Statehood & Insular Governance",
    "Mexican War",
)
EXPANSION_NATIVE_TOPICS = (
    "Indian Affairs, Removal & Allotment",
    "Public Lands & Homestead Settlement",
)
EXPANSION_DIPLOMATIC_MARKERS = (
    {
        "year": 1814,
        "label": "Treaty of Ghent",
        "detail": "Ends the War of 1812; historical context, not a causal marker.",
    },
    {
        "year": 1819,
        "label": "Adams–Onís Treaty",
        "detail": "Settlement with Spain; historical context, not a causal marker.",
    },
    {
        "year": 1846,
        "label": "Oregon Treaty",
        "detail": "Boundary settlement with Britain; historical context, not a causal marker.",
    },
    {
        "year": 1848,
        "label": "Guadalupe Hidalgo",
        "detail": "Ends the Mexican War; historical context, not a causal marker.",
    },
)

ERA_SHORT_LABELS = (
    "Establishing",
    "Continental republic",
    "Sectional crisis",
    "Admin-industrial",
    "Reform & collapse",
    "New Deal / WWII",
    "Mature Cold War",
    "Conservative / always-on",
    "Platform era",
)

ECHO_TOPIC_COLORS = {
    "🌎": "#4f7557",
    "🛡️": "#a24f52",
    "✨": "#9a7133",
    "🏛️": "#315f78",
    "💰": "#9a7133",
    "🌾": "#667342",
    "👥": "#6d5a88",
    "•": "#74695f",
}

DEFINED_TRAJECTORY_COLORS = (
    "#a24f52",
    "#4f7557",
    "#315f78",
    "#9a7133",
)
ECHO_TOPIC_SHORT_LABELS = {
    "Commemoration, Eulogy & National Mourning": "Commemoration & mourning",
    "Slavery, Emancipation & Sectionalism": "Slavery & emancipation",
    "Presidential Humility & Reflection on Office": "Reflection on office",
    "Monroe Doctrine & Latin American Policy": "Monroe Doctrine",
    "Personal Narrative, Storytelling & Boasting": "Personal narrative",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_topic_life_annotations(
    taxonomy: dict,
    profiles_by_era: dict[str, dict],
    path: Path = TOPIC_LIFE_ANNOTATIONS_PATH,
    *,
    verify_fingerprints: bool = True,
) -> dict[tuple[str, str], dict]:
    """Load timestamped Topic Life annotation history and select current rows.

    The source is append-only by ``(era_key, profile_topic)``: every editorial
    change is a new event, and the latest UTC timestamp is canonical.  This is
    not a paid annotation artifact or a substitute for the measured path.
    """
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != TOPIC_LIFE_ANNOTATION_SCHEMA:
        raise ValueError(
            "topic-life annotation schema must be "
            f"{TOPIC_LIFE_ANNOTATION_SCHEMA!r}"
        )
    if payload.get("layer_type") != "editorial_annotation_history":
        raise ValueError(
            "topic-life annotations must declare "
            "editorial_annotation_history"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("topic-life interpretations require provenance")
    fingerprints = provenance.get("inputs")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError(
            "topic-life annotations require source fingerprints"
        )
    if verify_fingerprints:
        repo_root = Path(__file__).resolve().parents[2]
        for relative, expected in fingerprints.items():
            source = repo_root / relative
            if not source.is_file():
                raise ValueError(
                    f"topic-life interpretation input is missing: {relative}"
                )
            actual = _sha256(source)
            if actual != expected:
                raise ValueError(
                    "topic-life annotation input changed; append a reviewed "
                    f"annotation before reuse: {relative}"
                )

    era_keys = {spec.key for spec in era_profiles.ERA_PROFILE_SPECS}
    if set(profiles_by_era) != era_keys:
        raise ValueError(
            "topic-life annotations require all canonical era profiles"
        )
    profile_topics = {
        (era_key, row["topic"])
        for era_key, profile in profiles_by_era.items()
        for row in profile["major_topics"]
    }
    level2 = {entry["name"] for entry in taxonomy["level2"]}
    level1 = {entry["name"] for entry in taxonomy["level1"]}
    required = {
        "annotation_id",
        "era_key",
        "profile_topic",
        "recorded_at",
        "source_topic",
        "interpretive_label",
        "rationale",
        "validation_status",
        "annotator",
    }
    allowed_labels = {
        "Fades",
        "Persists",
        "Returns",
        "Grows",
        "Peaks here",
        "Reframes",
        "Changes",
    }
    histories: dict[tuple[str, str], list[dict]] = {}
    annotation_ids: set[str] = set()
    key_timestamps: set[tuple[str, str, str]] = set()
    for raw in payload.get("records", []):
        if not isinstance(raw, dict):
            raise ValueError("topic-life annotation records must be objects")
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                f"topic-life annotation record is missing: {missing}"
            )
        annotation_id = raw["annotation_id"]
        if not isinstance(annotation_id, str) or not annotation_id.strip():
            raise ValueError("topic-life annotation_id must be non-empty")
        if annotation_id in annotation_ids:
            raise ValueError(
                f"duplicate topic-life annotation_id: {annotation_id!r}"
            )
        annotation_ids.add(annotation_id)
        if raw["era_key"] not in era_keys:
            raise ValueError(
                f"unknown topic-life era key: {raw['era_key']!r}"
            )
        key = (raw["era_key"], raw["profile_topic"])
        if key not in profile_topics:
            raise ValueError(
                f"unknown topic-life profile topic: {key!r}"
            )
        recorded_at = raw["recorded_at"]
        if not isinstance(recorded_at, str) or not recorded_at.endswith("Z"):
            raise ValueError(
                "topic-life recorded_at must be an ISO 8601 UTC timestamp "
                "ending in Z"
            )
        try:
            parsed_timestamp = datetime.fromisoformat(
                recorded_at.removesuffix("Z") + "+00:00"
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid topic-life recorded_at: {recorded_at!r}"
            ) from exc
        timestamp_key = (*key, recorded_at)
        if timestamp_key in key_timestamps:
            raise ValueError(
                "topic-life annotations cannot share one timestamp for the "
                f"same era/topic: {key!r} at {recorded_at}"
            )
        key_timestamps.add(timestamp_key)
        if raw["source_topic"] not in level2:
            raise ValueError(
                f"unknown topic-life source topic: {raw['source_topic']!r}"
            )
        if raw["interpretive_label"] not in allowed_labels:
            raise ValueError(
                "unknown topic-life interpretive_label: "
                f"{raw['interpretive_label']!r}"
            )
        if raw["validation_status"] not in {
            "unvalidated",
            "validated",
            "rejected",
        }:
            raise ValueError(
                "topic-life validation_status must be unvalidated, "
                "validated, or rejected"
            )
        annotator = raw["annotator"]
        if not isinstance(annotator, dict):
            raise ValueError("topic-life annotator must be an object")
        if annotator.get("type") == "ai":
            if not all(
                isinstance(annotator.get(field), str)
                and annotator[field].strip()
                for field in ("model", "reasoning_effort")
            ):
                raise ValueError(
                    "AI topic-life annotations require model and "
                    "reasoning_effort"
                )
        elif annotator.get("type") == "human":
            if not isinstance(annotator.get("name"), str) or not annotator[
                "name"
            ].strip():
                raise ValueError(
                    "human topic-life annotations require a name"
                )
        else:
            raise ValueError(
                "topic-life annotator type must be human or ai"
            )
        successor_topics: list[str] = []
        if raw["interpretive_label"] == "Reframes":
            successor_required = {
                "successor_kind",
                "successor_topic",
                "successor_display",
            }
            missing_successor = sorted(successor_required - set(raw))
            if missing_successor:
                raise ValueError(
                    "Reframes annotation is missing successor fields: "
                    f"{missing_successor}"
                )
            if raw["successor_kind"] == "level1":
                successor_vocabulary = level1
                successor_topics = [
                    entry["name"]
                    for entry in taxonomy["level2"]
                    if entry["level1"] == raw["successor_topic"]
                ]
            elif raw["successor_kind"] == "level2":
                successor_vocabulary = level2
                successor_topics = [raw["successor_topic"]]
            else:
                raise ValueError(
                    "topic-life successor_kind must be level1 or level2"
                )
            if raw["successor_topic"] not in successor_vocabulary:
                raise ValueError(
                    f"unknown topic-life successor: "
                    f"{raw['successor_topic']!r}"
                )
            if raw.get("successor_excludes_source"):
                successor_topics = [
                    topic
                    for topic in successor_topics
                    if topic != raw["source_topic"]
                ]
            if not successor_topics:
                raise ValueError(
                    "topic-life successor must resolve to at least one "
                    "level-2 topic after exclusions"
                )
        normalized = {
            **raw,
            "successor_topics": successor_topics,
            "_recorded_at": parsed_timestamp,
        }
        histories.setdefault(key, []).append(normalized)
    canonical: dict[tuple[str, str], dict] = {}
    for key, history in histories.items():
        history.sort(key=lambda record: record["_recorded_at"])
        public_history = [
            {
                field: value
                for field, value in record.items()
                if field != "_recorded_at"
            }
            for record in history
        ]
        current = dict(public_history[-1])
        current["annotation_history"] = public_history
        canonical[key] = current
    return canonical


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


def _era_key_for_year(year: int) -> str:
    return era_profiles.story_era_for_year(year).key


def _topic_mask(frame: pd.DataFrame, topics: tuple[str, ...]) -> pd.Series:
    wanted = frozenset(topics)
    return frame["topic_set"].map(
        lambda assigned: not assigned.isdisjoint(wanted)
    )


def _topic_pattern(shares: list[float], focal_index: int) -> tuple[str, str]:
    """Describe an observed attention path without inferring a successor."""
    focal = shares[focal_index]
    later = shares[focal_index + 1:]
    earlier = shares[:focal_index]
    epsilon = 0.25
    if later:
        later_max = max(later)
        later_mean = sum(later) / len(later)
        later_peak_index = later.index(later_max)
        trough_before_peak = min(later[:later_peak_index + 1])
        if (
            later_peak_index > 0
            and trough_before_peak <= max(epsilon, focal * 0.45)
            and later_max >= max(1.0, focal * 0.75)
        ):
            if min(shares) > 0:
                return (
                    "Persists + returns",
                    "Present throughout the era axis, with a trough before "
                    "attention rises again.",
                )
            return "Returns", "Falls away, then regains attention."
        if focal >= 1.0 and later_mean <= focal * 0.30:
            return "Fades", "Later eras devote far less attention to it."
        if (
            min(shares) > 0
            and max(shares) >= epsilon
            and shares[-1] >= max(epsilon, focal * 0.35)
        ):
            return "Persists", "It remains present across all nine eras."
        if later_max >= focal * 1.25 and later_max - focal >= 1.0:
            return "Peaks later", "A later era gives it more attention."
        if later_mean >= focal * 0.65:
            return "Persists", "It remains a recurring presidential concern."
        return "Declines", "Later eras devote less attention to it."
    if earlier:
        earlier_max = max(earlier)
        if focal >= earlier_max * 1.25 and focal - earlier_max >= 1.0:
            return "Peaks here", "The present era gives it its highest share."
        if min(earlier) <= max(epsilon, focal * 0.45) and focal >= 1.0:
            return "Returns", "It regains attention in the present era."
        if sum(earlier) / len(earlier) >= focal * 0.65:
            return "Persists", "It remains a recurring presidential concern."
    return "Changes", "Its measured share changes across the era axis."


def _topic_values(
    topics: tuple[str, ...],
    frames: dict[str, pd.DataFrame],
) -> tuple[list[dict], list[float]]:
    values = []
    shares = []
    for era_spec in era_profiles.ERA_PROFILE_SPECS:
        frame = frames[era_spec.key]
        count = int(_topic_mask(frame, topics).sum())
        share = count / len(frame) * 100
        shares.append(float(share))
        values.append({
            "era_key": era_spec.key,
            "era_label": era_spec.label,
            "count": count,
            "denominator": len(frame),
            "share": float(share),
        })
    return values, shares


def _reframe_gate(
    source_shares: list[float],
    successor_shares: list[float],
    focal_index: int,
) -> dict:
    """Require a declining source and a negatively related growing successor."""
    source_rank = pd.Series(source_shares).rank(method="average")
    successor_rank = pd.Series(successor_shares).rank(method="average")
    spearman = source_rank.corr(successor_rank)
    correlation = None if pd.isna(spearman) else float(spearman)
    source_decline = float(source_shares[focal_index] - source_shares[-1])
    successor_growth = float(
        successor_shares[-1] - successor_shares[focal_index]
    )
    passes = bool(
        correlation is not None
        and correlation <= REFRAME_MAX_SPEARMAN
        and source_decline >= REFRAME_MIN_SOURCE_DECLINE_PP
        and successor_growth >= REFRAME_MIN_SUCCESSOR_GROWTH_PP
    )
    return {
        "passes": passes,
        "spearman_correlation": correlation,
        "max_spearman": REFRAME_MAX_SPEARMAN,
        "source_decline_pp": source_decline,
        "minimum_source_decline_pp": REFRAME_MIN_SOURCE_DECLINE_PP,
        "successor_growth_pp": successor_growth,
        "minimum_successor_growth_pp": REFRAME_MIN_SUCCESSOR_GROWTH_PP,
    }


def _founding_defined(
    frames: dict[str, pd.DataFrame],
) -> dict:
    founding = frames["founding"]
    rows = []
    for key, icon, label, topics, explanation, color in FOUNDING_DEFINED_SPECS:
        founding_count = int(_topic_mask(founding, topics).sum())
        founding_share = founding_count / len(founding) * 100
        historical_values = []
        for era_spec in era_profiles.ERA_PROFILE_SPECS[1:]:
            frame = frames[era_spec.key]
            count = int(_topic_mask(frame, topics).sum())
            historical_values.append({
                "era_key": era_spec.key,
                "era_label": era_spec.label,
                "count": count,
                "denominator": len(frame),
                "share": float(count / len(frame) * 100),
            })
        historical_average = sum(
            value["share"] for value in historical_values
        ) / len(historical_values)
        era_values = [{
            "era_key": "founding",
            "era_label": era_profiles.ERA_PROFILE_SPECS[0].label,
            "count": founding_count,
            "denominator": len(founding),
            "share": float(founding_share),
        }, *historical_values]
        president_values = []
        for president, frame in founding.groupby("president", observed=True):
            count = int(_topic_mask(frame, topics).sum())
            president_values.append({
                "president": str(president),
                "count": count,
                "denominator": len(frame),
                "share": float(count / len(frame) * 100),
            })
        president_values.sort(key=lambda value: value["president"])
        president_average = sum(
            value["share"] for value in president_values
        ) / len(president_values)
        rows.append({
            "key": key,
            "icon": icon,
            "label": label,
            "topics": list(topics),
            "explanation": explanation,
            "color": color,
            "aggregate_count": founding_count,
            "aggregate_denominator": len(founding),
            "aggregate_share": float(founding_share),
            "historical_average_share": float(historical_average),
            "historical_era_values": historical_values,
            "era_values": era_values,
            "president_average_share": float(president_average),
            "president_values": president_values,
            "difference_pp": float(founding_share - historical_average),
            "rate_ratio": (
                None
                if historical_average == 0
                else float(founding_share / historical_average)
            ),
        })
    rows.sort(
        key=lambda row: (-row["difference_pp"], row["label"])
    )
    observed_max_share = max(
        value["share"]
        for row in rows
        for value in row["era_values"]
    )
    scale_step = 5
    scale_max = int(
        (math.floor(observed_max_share / scale_step) + 1) * scale_step
    )
    defined_topics = tuple(dict.fromkeys(
        topic
        for _, _, _, topics, _, _ in FOUNDING_DEFINED_SPECS
        for topic in topics
    ))
    founding_coverage_count = int(
        _topic_mask(founding, defined_topics).sum()
    )
    historical_coverage_values = []
    for era_spec in era_profiles.ERA_PROFILE_SPECS[1:]:
        frame = frames[era_spec.key]
        count = int(_topic_mask(frame, defined_topics).sum())
        historical_coverage_values.append({
            "era_key": era_spec.key,
            "era_label": era_spec.label,
            "count": count,
            "denominator": len(frame),
            "share": float(count / len(frame) * 100),
        })
    historical_coverage_average = sum(
        value["share"] for value in historical_coverage_values
    ) / len(historical_coverage_values)
    return {
        "status": "authored",
        "visual_kind": "combined_trajectory",
        "title": "Era Defined",
        "headline": "Getting the Republic Up and Running",
        "dek": (
            "The startup agenda combined unusually intense sovereignty "
            "problems with the recurring work of financing government."
        ),
        "rows": rows,
        "coverage": {
            "founding_count": founding_coverage_count,
            "founding_denominator": len(founding),
            "founding_share": float(
                founding_coverage_count / len(founding) * 100
            ),
            "founding_summed_share": float(
                sum(row["aggregate_share"] for row in rows)
            ),
            "historical_average_share": float(
                historical_coverage_average
            ),
            "historical_summed_share": float(
                sum(row["historical_average_share"] for row in rows)
            ),
            "historical_era_values": historical_coverage_values,
        },
        "scale_max": scale_max,
        "scale_step": scale_step,
        "observed_max_share": float(observed_max_share),
        "benchmark_label": "Historical average",
        "measure_note": (
            "The benchmark gives each of the other eight eras equal weight; "
            "one combined trajectory graph shows every era separately on a "
            "shared zero baseline. Its ceiling is the next five-point mark "
            "above the largest observed family share, rather than 100%. "
            "Each family is a union of assigned topic labels. Multi-label "
            "paragraphs can appear in more than one family, so the four shares "
            "are independent rather than parts of a 100% whole."
        ),
    }


def _shared_trajectory_defined(
    spec: era_profiles.EraProfileSpec,
    profile: dict,
    frames: dict[str, pd.DataFrame],
) -> dict:
    """Apply the Founding combined-trajectory grammar to every story era."""
    if spec.key == "founding":
        chart = _founding_defined(frames)
        chart.update({
            "focal_index": 0,
            "focal_key": spec.key,
            "focal_label": spec.label,
            "focal_short": ERA_SHORT_LABELS[0],
            "selection_basis": "historically declared topic families",
        })
        chart["coverage"].update({
            "focal_count": chart["coverage"]["founding_count"],
            "focal_denominator": chart["coverage"]["founding_denominator"],
            "focal_share": chart["coverage"]["founding_share"],
            "focal_summed_share": chart["coverage"]["founding_summed_share"],
        })
        return chart

    focal_index = next(
        index
        for index, candidate in enumerate(era_profiles.ERA_PROFILE_SPECS)
        if candidate.key == spec.key
    )
    if spec.key == "expansion":
        raw_specs = [
            {
                "key": key,
                "icon": icon,
                "label": label,
                "short_label": label,
                "topics": tuple(topics),
                "explanation": "",
                "color": color,
            }
            for key, label, icon, topics, color in EXPANSION_DEFINED_SERIES
        ]
        headline = "Diplomacy, institutions, war, and continental power"
        dek = (
            "The four established themes of the continental republic are "
            "traced across the same nine-era axis."
        )
        selection_basis = "historically declared topic families"
    else:
        raw_specs = []
        for rank, topic in enumerate(profile["major_topics"][:4]):
            canonical = str(topic["topic"])
            label = str(topic["label"])
            raw_specs.append({
                "key": f"priority-{rank + 1}",
                "icon": era_visualizations.topic_icon(canonical),
                "label": label,
                "short_label": era_visualizations.TOPIC_SHORT_LABELS.get(
                    canonical,
                    label,
                ),
                "topics": tuple(str(value) for value in topic["topics"]),
                "explanation": "",
                "color": DEFINED_TRAJECTORY_COLORS[rank],
            })
        headline = f"Four subjects at the center of {profile['title']}"
        dek = (
            "The era’s four most prevalent profile topics are traced across "
            "the same nine-era axis."
        )
        selection_basis = "four leading era-profile topics by paragraph share"

    focal = frames[spec.key]
    rows = []
    for raw in raw_specs:
        topics = raw["topics"]
        era_values, _ = _topic_values(topics, frames)
        focal_value = era_values[focal_index]
        comparison_values = [
            value
            for index, value in enumerate(era_values)
            if index != focal_index
        ]
        comparison_average = sum(
            value["share"] for value in comparison_values
        ) / len(comparison_values)
        president_values = []
        for president, president_frame in focal.groupby(
            "president",
            observed=True,
        ):
            count = int(_topic_mask(president_frame, topics).sum())
            president_values.append({
                "president": str(president),
                "count": count,
                "denominator": len(president_frame),
                "share": float(count / len(president_frame) * 100),
            })
        president_values.sort(key=lambda value: value["president"])
        president_average = sum(
            value["share"] for value in president_values
        ) / len(president_values)
        focal_share = float(focal_value["share"])
        rows.append({
            **raw,
            "topics": list(topics),
            "aggregate_count": int(focal_value["count"]),
            "aggregate_denominator": int(focal_value["denominator"]),
            "aggregate_share": focal_share,
            "historical_average_share": float(comparison_average),
            "historical_era_values": comparison_values,
            "era_values": era_values,
            "president_average_share": float(president_average),
            "president_values": president_values,
            "difference_pp": float(focal_share - comparison_average),
            "rate_ratio": (
                None
                if comparison_average == 0
                else float(focal_share / comparison_average)
            ),
        })
    rows.sort(key=lambda row: (-row["difference_pp"], row["label"]))

    observed_max_share = max(
        value["share"]
        for row in rows
        for value in row["era_values"]
    )
    scale_step = 5
    scale_max = max(
        scale_step,
        int((math.floor(observed_max_share / scale_step) + 1) * scale_step),
    )
    defined_topics = tuple(dict.fromkeys(
        topic
        for raw in raw_specs
        for topic in raw["topics"]
    ))
    focal_coverage_count = int(
        _topic_mask(focal, defined_topics).sum()
    )
    comparison_coverage_values = []
    for index, era_spec in enumerate(era_profiles.ERA_PROFILE_SPECS):
        if index == focal_index:
            continue
        frame = frames[era_spec.key]
        count = int(_topic_mask(frame, defined_topics).sum())
        comparison_coverage_values.append({
            "era_key": era_spec.key,
            "era_label": era_spec.label,
            "count": count,
            "denominator": len(frame),
            "share": float(count / len(frame) * 100),
        })
    comparison_coverage_average = sum(
        value["share"] for value in comparison_coverage_values
    ) / len(comparison_coverage_values)
    return {
        "status": "authored",
        "visual_kind": "combined_trajectory",
        "title": "Era Defined",
        "headline": headline,
        "dek": dek,
        "rows": rows,
        "coverage": {
            "focal_count": focal_coverage_count,
            "focal_denominator": len(focal),
            "focal_share": float(
                focal_coverage_count / len(focal) * 100
            ),
            "focal_summed_share": float(
                sum(row["aggregate_share"] for row in rows)
            ),
            "historical_average_share": float(
                comparison_coverage_average
            ),
            "historical_summed_share": float(
                sum(row["historical_average_share"] for row in rows)
            ),
            "historical_era_values": comparison_coverage_values,
        },
        "scale_max": scale_max,
        "scale_step": scale_step,
        "observed_max_share": float(observed_max_share),
        "benchmark_label": "Equal-era average of the other eight eras",
        "focal_index": focal_index,
        "focal_key": spec.key,
        "focal_label": spec.label,
        "focal_short": ERA_SHORT_LABELS[focal_index],
        "selection_basis": selection_basis,
        "measure_note": (
            f"The four lines use {selection_basis}. Each point is the share "
            "of all corpus paragraphs in that era carrying at least one "
            "assigned topic in the named family, including topic-free "
            "paragraphs in the denominator. The graph uses a shared zero "
            "baseline, and its ceiling is the next five-point mark above the "
            "largest observed value rather than 100%. Multi-label paragraphs "
            "can appear in more than one family, so the shares are "
            "independent rather than parts of a 100% whole."
        ),
    }


def _expansion_defined(frames: dict[str, pd.DataFrame]) -> dict:
    """Define the era as four historical phases joined by diplomacy."""
    expansion = frames["expansion"]
    phases = []
    for key, label, start_year, end_year, explanation in (
        EXPANSION_DEFINED_PHASES
    ):
        phase = expansion[
            expansion["year"].between(start_year, end_year)
        ].copy()
        if phase.empty:
            raise ValueError(f"expansion defining phase {key!r} is empty")
        phases.append({
            "key": key,
            "label": label,
            "years": f"{start_year}–{str(end_year)[-2:]}",
            "start_year": start_year,
            "end_year": end_year,
            "explanation": explanation,
            "denominator": len(phase),
            "presidents": list(dict.fromkeys(
                phase.sort_values(["year", "president"])["president"].astype(str)
            )),
        })

    series = []
    for key, label, icon, topics, color in EXPANSION_DEFINED_SERIES:
        values = []
        for phase in phases:
            phase_frame = expansion[
                expansion["year"].between(
                    phase["start_year"], phase["end_year"]
                )
            ]
            count = int(_topic_mask(phase_frame, topics).sum())
            value = {
                "phase_key": phase["key"],
                "count": count,
                "denominator": len(phase_frame),
                "share": float(count / len(phase_frame) * 100),
            }
            if key == "territory":
                external_count = int(
                    _topic_mask(phase_frame, EXPANSION_EXTERNAL_TOPICS).sum()
                )
                native_count = int(
                    _topic_mask(phase_frame, EXPANSION_NATIVE_TOPICS).sum()
                )
                value["subseries"] = [
                    {
                        "key": "external",
                        "label": "External acquisition",
                        "count": external_count,
                        "share": float(
                            external_count / len(phase_frame) * 100
                        ),
                    },
                    {
                        "key": "native",
                        "label": "Native removal & settlement",
                        "count": native_count,
                        "share": float(
                            native_count / len(phase_frame) * 100
                        ),
                    },
                ]
            values.append(value)
        series.append({
            "key": key,
            "label": label,
            "icon": icon,
            "topics": list(topics),
            "color": color,
            "values": values,
        })

    return {
        "status": "authored",
        "visual_kind": "diplomatic_tree",
        "title": "Era Defined",
        "headline": "Diplomacy holds the continental republic together",
        "dek": (
            "War recedes, institutional conflict swells, and territorial "
            "conquest takes its place. Treaties and diplomacy persist across "
            "all four historical phases."
        ),
        "phases": phases,
        "series": series,
        "scale_max": 50,
        "event_markers": list(EXPANSION_DIPLOMATIC_MARKERS),
        "measure_note": (
            "Each step is an independent share of all corpus paragraphs in "
            "that historical phase, including topic-free paragraphs. All four "
            "threads use one 0–50% axis. Assigned topic labels overlap, so the "
            "lines are not stacked and do not sum to 100%. Treaty markers "
            "provide noncausal historical context."
        ),
    }


def _pending_defined(spec: era_profiles.EraProfileSpec) -> dict:
    return {
        "status": "pending",
        "title": "Era Defined",
        "headline": spec.title,
        "dek": (
            "This slot is reserved for the era-specific graph that proves the "
            "era’s defining takeaway."
        ),
        "rows": [],
        "scale_max": None,
        "benchmark_label": None,
        "measure_note": (
            "No generalized substitute is shown: each era’s defining claim "
            "needs its own declared evidence and acceptance test."
        ),
    }


def _topic_life(
    spec: era_profiles.EraProfileSpec,
    profile: dict,
    frames: dict[str, pd.DataFrame],
    annotations: dict[tuple[str, str], dict],
) -> dict:
    focal_index = [
        candidate.key for candidate in era_profiles.ERA_PROFILE_SPECS
    ].index(spec.key)
    rows = []
    for topic in profile["major_topics"]:
        interpretation_candidate = annotations.get(
            (spec.key, topic["topic"])
        )
        source_topic = (
            interpretation_candidate["source_topic"]
            if interpretation_candidate
            else topic["topics"][0]
        )
        source_topics = (source_topic,)
        values, shares = _topic_values(source_topics, frames)
        pattern, pattern_detail = _topic_pattern(shares, focal_index)
        peak_index = max(range(len(shares)), key=shares.__getitem__)
        peak_spec = era_profiles.ERA_PROFILE_SPECS[peak_index]
        successor_values = None
        successor_topics: tuple[str, ...] = ()
        successor_shares: list[float] = []
        reframe_gate = None
        interpretation = None
        reframe_switch_index = None
        if (
            interpretation_candidate
            and interpretation_candidate["interpretive_label"] == "Reframes"
        ):
            candidate_topics = tuple(
                interpretation_candidate["successor_topics"]
            )
            candidate_values, candidate_shares = _topic_values(
                candidate_topics,
                frames,
            )
            reframe_gate = _reframe_gate(
                shares,
                candidate_shares,
                focal_index,
            )
            reframe_gate["candidate_status"] = interpretation_candidate[
                "validation_status"
            ]
            if (
                interpretation_candidate["validation_status"] != "rejected"
                and reframe_gate["passes"]
            ):
                interpretation = interpretation_candidate
                successor_topics = candidate_topics
                successor_values = candidate_values
                successor_shares = candidate_shares
                reframe_switch_index = next(
                    (
                        index
                        for index in range(focal_index + 1, len(shares))
                        if candidate_shares[index] >= shares[index]
                    ),
                    None,
                )
        row_scale_max = max(
            1.0,
            max(shares + successor_shares),
        )
        rows.append({
            "topic": topic["topic"],
            "label": topic["label"],
            "source_topic": source_topic,
            "source_label": era_visualizations.TOPIC_SHORT_LABELS.get(
                source_topic,
                topic["label"],
            ),
            "topics": list(source_topics),
            "icon": era_visualizations.topic_icon(source_topic),
            "values": values,
            "focal_share": shares[focal_index],
            "attention_pattern": pattern,
            "pattern_detail": pattern_detail,
            "peak_era_key": peak_spec.key,
            "peak_era_label": peak_spec.label,
            "peak_share": shares[peak_index],
            "first_share": shares[0],
            "present_share": shares[-1],
            "successor_label": (
                interpretation["successor_display"]
                if interpretation
                else None
            ),
            "successor_topics": list(successor_topics),
            "successor_values": successor_values,
            "reframe_switch_index": reframe_switch_index,
            "reframe_switch_era_key": (
                era_profiles.ERA_PROFILE_SPECS[
                    reframe_switch_index
                ].key
                if reframe_switch_index is not None
                else None
            ),
            "scale_max": float(row_scale_max),
            "reframe_gate": reframe_gate,
            "canonical_annotation_id": (
                interpretation_candidate["annotation_id"]
                if interpretation_candidate
                else None
            ),
            "annotation_history": (
                interpretation_candidate["annotation_history"]
                if interpretation_candidate
                else []
            ),
            "annotation": (
                {
                    "annotation_id": interpretation_candidate[
                        "annotation_id"
                    ],
                    "recorded_at": interpretation_candidate["recorded_at"],
                    "label": interpretation_candidate[
                        "interpretive_label"
                    ],
                    "rationale": interpretation_candidate["rationale"],
                    "validation_status": interpretation_candidate[
                        "validation_status"
                    ],
                    "annotator": interpretation_candidate["annotator"],
                }
                if interpretation_candidate
                else None
            ),
            "interpretation": (
                {
                    "annotation_id": interpretation["annotation_id"],
                    "recorded_at": interpretation["recorded_at"],
                    "annotator": interpretation["annotator"],
                    "label": interpretation["interpretive_label"],
                    "successor_kind": interpretation["successor_kind"],
                    "successor_topic": interpretation["successor_topic"],
                    "successor_display": interpretation[
                        "successor_display"
                    ],
                    "successor_topics": list(successor_topics),
                    "rationale": interpretation["rationale"],
                    "validation_status": interpretation[
                        "validation_status"
                    ],
                }
                if interpretation
                else None
            ),
        })
    scale_max = max(
        5,
        int(math.ceil(
            max(
                value["share"]
                for row in rows
                for values in (
                    row["values"],
                    row["successor_values"] or [],
                )
                for value in values
            ) / 5
        ) * 5),
    )
    return {
        "title": "Topic Life",
        "headline": (
            "What happened to the Founding’s major topics?"
            if spec.key == "founding"
            else "What happened to this era’s major topics?"
        ),
        "dek": (
            "Observed topic paths, with separately marked and unvalidated "
            "reframing hypotheses."
        ),
        "rows": rows,
        "scale_max": scale_max,
        "scale_mode": "shared_percent",
        "reframe_rule": {
            "measure": "spearman_correlation_across_nine_era_shares",
            "max_spearman": REFRAME_MAX_SPEARMAN,
            "minimum_source_decline_pp": REFRAME_MIN_SOURCE_DECLINE_PP,
            "minimum_successor_growth_pp": (
                REFRAME_MIN_SUCCESSOR_GROWTH_PP
            ),
        },
        "focal_index": focal_index,
    }


def _reference_evidence(
    speeches: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    invocation_evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required = {
        "candidate_id",
        "speaker",
        "target",
        "doc_name",
        "para_idx",
        "era",
        "target_status",
    }
    missing = sorted(required - set(invocation_evidence.columns))
    if missing:
        raise ValueError(
            f"invocation_evidence is missing era-echo columns: {missing}"
        )
    first_year = speeches.groupby("president", observed=True)["year"].min()
    target_era = {
        str(president): _era_key_for_year(int(year))
        for president, year in first_year.items()
    }
    evidence = invocation_evidence[
        invocation_evidence["target_status"].eq("former_president")
        & invocation_evidence["speaker"].ne(invocation_evidence["target"])
    ].copy()
    speech_metadata = (
        speeches[["doc_name", "year", "president"]]
        .drop_duplicates()
        .set_index("doc_name")
    )
    if speech_metadata.index.duplicated().any():
        raise ValueError(
            "era echoes require one calendar year per speech document"
        )
    source_by_doc = {
        str(doc_name): era_profiles.story_era_for_speech(
            int(row.year), str(row.president)
        ).key
        for doc_name, row in speech_metadata.iterrows()
    }
    evidence["source_key"] = evidence["doc_name"].map(source_by_doc)
    evidence["target_key"] = evidence["target"].map(target_era)
    if evidence[["source_key", "target_key"]].isna().any().any():
        missing_sources = sorted(
            evidence.loc[evidence["source_key"].isna(), "doc_name"]
            .astype(str)
            .unique()
        )
        missing_targets = sorted(
            evidence.loc[evidence["target_key"].isna(), "target"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "era echoes could not map invocation evidence: "
            f"source documents={missing_sources}, targets={missing_targets}"
        )
    order = {
        spec.key: index
        for index, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)
    }
    future = evidence[
        evidence.apply(
            lambda row: order[row["target_key"]] > order[row["source_key"]],
            axis=1,
        )
    ]
    if not future.empty:
        raise ValueError(
            "era echoes found a reference to a president from a future era"
        )
    paragraph_meta = pd.concat(
        [
            frame[
                ["doc_name", "para_idx", "topic_set", "era_key", "president"]
            ]
            for frame in frames.values()
        ],
        ignore_index=True,
    )
    if paragraph_meta.duplicated(["doc_name", "para_idx"]).any():
        raise ValueError("era-echo paragraph metadata has duplicate keys")
    evidence = evidence.merge(
        paragraph_meta,
        on=["doc_name", "para_idx"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_paragraph"),
    )
    if evidence["topic_set"].isna().any():
        raise ValueError(
            "era echoes contain invocation rows without paragraph annotations"
        )
    if not evidence["source_key"].eq(evidence["era_key"]).all():
        raise ValueError(
            "era echoes disagree on source era after exact paragraph join"
        )
    return evidence, int(len(evidence))


def _direction_counts(evidence: pd.DataFrame) -> dict:
    return {
        "reference_paragraphs": int(
            len(evidence[["doc_name", "para_idx"]].drop_duplicates())
        ),
        "raw_mentions": int(len(evidence)),
        "speeches": int(evidence["doc_name"].nunique()),
    }


def _topic_nodes(evidence: pd.DataFrame) -> list[dict]:
    if evidence.empty:
        return []
    paragraphs = evidence[
        ["doc_name", "para_idx", "topic_set"]
    ].drop_duplicates(["doc_name", "para_idx"])
    denominator = len(paragraphs)
    assignments = (
        paragraphs.explode("topic_set")
        .rename(columns={"topic_set": "topic"})
        .dropna(subset=["topic"])
    )
    rows = []
    for topic, group in assignments.groupby(
        "topic",
        observed=True,
        sort=False,
    ):
        paragraph_keys = group[["doc_name", "para_idx"]]
        mention_rows = paragraph_keys.merge(
            evidence,
            on=["doc_name", "para_idx"],
            how="left",
            validate="one_to_many",
        )
        icon = era_visualizations.topic_icon(str(topic))
        rows.append({
            "topic": str(topic),
            "label": ECHO_TOPIC_SHORT_LABELS.get(
                str(topic),
                era_visualizations.TOPIC_SHORT_LABELS.get(
                    str(topic),
                    str(topic),
                ),
            ),
            "icon": icon,
            "color": ECHO_TOPIC_COLORS[icon],
            "paragraphs": int(len(group)),
            "reference_paragraphs": denominator,
            "share": float(len(group) / denominator * 100),
            "raw_mentions": int(len(mention_rows)),
            "speeches": int(mention_rows["doc_name"].nunique()),
            "source_eras": sorted(
                str(value) for value in mention_rows["source_key"].unique()
            ),
            "source_presidents": sorted(
                str(value) for value in mention_rows["speaker"].unique()
            ),
            "named_presidents": sorted(
                str(value) for value in mention_rows["target"].unique()
            ),
        })
    rows.sort(
        key=lambda row: (-row["paragraphs"], row["topic"])
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _actor_items(evidence: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for name, group in evidence.groupby(column, observed=True, sort=False):
        rows.append({
            "name": str(name),
            "raw_mentions": int(len(group)),
            "paragraphs": int(
                len(group[["doc_name", "para_idx"]].drop_duplicates())
            ),
            "speeches": int(group["doc_name"].nunique()),
        })
    rows.sort(
        key=lambda row: (
            -row["raw_mentions"],
            -row["paragraphs"],
            row["name"],
        )
    )
    return rows


def _gravity_network(
    spec: era_profiles.EraProfileSpec,
    touching: pd.DataFrame,
    focal_presidents: list[str],
) -> dict:
    """Place every exact echo around the presidents in the focal era.

    The renderer treats ``anchor_nodes`` as fixed planets.  External
    presidents and paragraph topics are satellites whose semantic position is
    the connection-weighted centroid of those anchors.  Collision handling is
    deliberately left to the renderer so the published contract retains the
    exact, reusable weights rather than one viewport-specific layout.
    """
    anchor_order = {
        president: index for index, president in enumerate(focal_presidents)
    }
    if len(anchor_order) != len(focal_presidents):
        raise ValueError(
            f"era echoes contain duplicate focal presidents for {spec.key}"
        )
    gravity_rows = touching.copy()
    if gravity_rows.empty:
        return {
            "layout": "connection_weighted_gravity",
            "anchor_nodes": [
                {
                    "key": president,
                    "label": president,
                    "chronology_index": index,
                    "is_connected": False,
                    "directions": [],
                    "reference_paragraphs": 0,
                    "raw_mentions": 0,
                    "speeches": 0,
                }
                for index, president in enumerate(focal_presidents)
            ],
            "president_satellites": [],
            "topic_satellites": [],
            "reference_paragraphs": 0,
            "raw_mentions": 0,
            "speeches": 0,
        }
    gravity_rows["focal_anchor"] = gravity_rows["target"].where(
        gravity_rows["direction"].eq("incoming"),
        gravity_rows["speaker"],
    )
    gravity_rows["external_president"] = gravity_rows["speaker"].where(
        gravity_rows["direction"].eq("incoming"),
        gravity_rows["target"],
    )
    unknown_anchors = sorted(
        set(gravity_rows["focal_anchor"].astype(str)) - set(anchor_order)
    )
    if unknown_anchors:
        raise ValueError(
            f"era echoes mapped non-focal anchors for {spec.key}: "
            f"{unknown_anchors}"
        )
    expanded = (
        gravity_rows.explode("topic_set")
        .rename(columns={"topic_set": "topic"})
        .dropna(subset=["topic"])
    )

    def connection_rows(group: pd.DataFrame) -> list[dict]:
        rows = []
        for anchor, anchor_group in group.groupby(
            "focal_anchor",
            observed=True,
            sort=False,
        ):
            rows.append({
                "anchor": str(anchor),
                **_direction_counts(anchor_group),
                "directions": sorted(
                    str(value)
                    for value in anchor_group["direction"].unique()
                ),
            })
        rows.sort(
            key=lambda row: (
                anchor_order[row["anchor"]],
                row["anchor"],
            )
        )
        incidence_total = sum(
            row["reference_paragraphs"] for row in rows
        )
        for row in rows:
            row["weight_share"] = float(
                row["reference_paragraphs"] / incidence_total * 100
            )
        return rows

    def ranked_related(
        group: pd.DataFrame,
        column: str,
        *,
        limit: int | None = 4,
    ) -> list[dict]:
        rows = []
        for label, label_group in group.groupby(
            column,
            observed=True,
            sort=False,
        ):
            rows.append({
                "label": str(label),
                **_direction_counts(label_group),
            })
        rows.sort(
            key=lambda row: (
                -row["reference_paragraphs"],
                -row["raw_mentions"],
                row["label"],
            )
        )
        return rows if limit is None else rows[:limit]

    anchors = []
    for president in focal_presidents:
        group = gravity_rows[
            gravity_rows["focal_anchor"].eq(president)
        ]
        anchors.append({
            "key": president,
            "label": president,
            "chronology_index": anchor_order[president],
            "is_connected": not group.empty,
            "directions": (
                sorted(
                    str(value) for value in group["direction"].unique()
                )
                if not group.empty
                else []
            ),
            **_direction_counts(group),
            "top_presidents": (
                ranked_related(group, "external_president")
                if not group.empty
                else []
            ),
            "top_topics": (
                ranked_related(
                    expanded[expanded["focal_anchor"].eq(president)],
                    "topic",
                )
                if not group.empty
                else []
            ),
        })

    president_satellites = []
    for president, group in gravity_rows.groupby(
        "external_president",
        observed=True,
        sort=False,
    ):
        topic_group = expanded[
            expanded["external_president"].eq(president)
        ]
        president_satellites.append({
            "key": str(president),
            "label": str(president),
            "directions": sorted(
                str(value) for value in group["direction"].unique()
            ),
            "connections": connection_rows(group),
            "top_topics": ranked_related(topic_group, "topic"),
            **_direction_counts(group),
        })
    president_satellites.sort(
        key=lambda row: (
            -row["reference_paragraphs"],
            -row["raw_mentions"],
            row["label"],
        )
    )

    topic_satellites = []
    for topic, group in expanded.groupby(
        "topic",
        observed=True,
        sort=False,
    ):
        icon = era_visualizations.topic_icon(str(topic))
        president_connections = ranked_related(
            group,
            "external_president",
            limit=None,
        )
        topic_satellites.append({
            "key": str(topic),
            "label": ECHO_TOPIC_SHORT_LABELS.get(
                str(topic),
                era_visualizations.TOPIC_SHORT_LABELS.get(
                    str(topic),
                    str(topic),
                ),
            ),
            "icon": icon,
            "color": ECHO_TOPIC_COLORS[icon],
            "directions": sorted(
                str(value) for value in group["direction"].unique()
            ),
            "connections": connection_rows(group),
            "president_connections": president_connections,
            "top_presidents": president_connections[:4],
            **_direction_counts(group),
        })
    topic_satellites.sort(
        key=lambda row: (
            -row["reference_paragraphs"],
            -row["raw_mentions"],
            row["label"],
        )
    )
    return {
        "layout": "connection_weighted_gravity",
        "anchor_nodes": anchors,
        "president_satellites": president_satellites,
        "topic_satellites": topic_satellites,
        **_direction_counts(gravity_rows),
    }


def _tripartite_network(
    spec: era_profiles.EraProfileSpec,
    incoming: pd.DataFrame,
    outgoing: pd.DataFrame,
    focal_presidents: list[str],
) -> dict:
    """Build exact president → topic → president paths touching one era."""
    touching = pd.concat(
        [
            incoming.assign(direction="incoming"),
            outgoing.assign(direction="outgoing"),
        ],
        ignore_index=True,
    )
    gravity = _gravity_network(spec, touching, focal_presidents)
    if touching.empty:
        return {
            "source_nodes": [],
            "topic_nodes": [],
            "target_nodes": [],
            "source_target_links": [],
            "source_topic_links": [],
            "topic_target_links": [],
            "paths": [],
            "gravity": gravity,
            "reference_paragraphs": 0,
            "raw_mentions": 0,
            "speeches": 0,
        }
    expanded = (
        touching.explode("topic_set")
        .rename(columns={"topic_set": "topic"})
        .dropna(subset=["topic"])
    )
    source_rank = [
        row["name"] for row in _actor_items(touching, "speaker")
    ]
    target_rank = [
        row["name"] for row in _actor_items(touching, "target")
    ]
    topic_rank = [row["topic"] for row in _topic_nodes(touching)]
    source_keep = set(source_rank[:7])
    target_keep = set(target_rank[:7])
    topic_keep = set(topic_rank[:6])
    source_other = "Other invoking presidents"
    target_other = "Other presidents invoked"
    topic_other = "Other topics"
    touching["source_node"] = touching["speaker"].where(
        touching["speaker"].isin(source_keep),
        source_other,
    )
    touching["target_node"] = touching["target"].where(
        touching["target"].isin(target_keep),
        target_other,
    )
    expanded["source_node"] = expanded["speaker"].where(
        expanded["speaker"].isin(source_keep),
        source_other,
    )
    expanded["target_node"] = expanded["target"].where(
        expanded["target"].isin(target_keep),
        target_other,
    )
    expanded["topic_node"] = expanded["topic"].where(
        expanded["topic"].isin(topic_keep),
        topic_other,
    )

    def actor_nodes(
        display_column: str,
        raw_column: str,
        era_column: str,
        rank: list[str],
        other_label: str,
    ) -> list[dict]:
        rows = []
        order = {
            name: index for index, name in enumerate(rank[:7])
        }
        order[other_label] = len(order)
        for label, group in touching.groupby(
            display_column,
            observed=True,
            sort=False,
        ):
            rows.append({
                "key": str(label),
                "label": str(label),
                "members": sorted(
                    str(value) for value in group[raw_column].unique()
                ),
                "is_other": str(label) == other_label,
                "is_focal_era": bool(group[era_column].eq(spec.key).any()),
                "directions": sorted(
                    str(value) for value in group["direction"].unique()
                ),
                **_direction_counts(group),
            })
        rows.sort(
            key=lambda row: (
                order.get(row["key"], len(order) + 1),
                row["label"],
            )
        )
        return rows

    topic_nodes = []
    topic_order = {
        name: index for index, name in enumerate(topic_rank[:6])
    }
    topic_order[topic_other] = len(topic_order)
    denominator = len(
        touching[["doc_name", "para_idx"]].drop_duplicates()
    )
    for label, group in expanded.groupby(
        "topic_node",
        observed=True,
        sort=False,
    ):
        paragraphs = group[
            ["doc_name", "para_idx"]
        ].drop_duplicates()
        members = sorted(str(value) for value in group["topic"].unique())
        if str(label) == topic_other:
            icon = "•"
            color = ECHO_TOPIC_COLORS[icon]
            display_label = topic_other
        else:
            icon = era_visualizations.topic_icon(str(label))
            color = ECHO_TOPIC_COLORS[icon]
            display_label = ECHO_TOPIC_SHORT_LABELS.get(
                str(label),
                era_visualizations.TOPIC_SHORT_LABELS.get(
                    str(label),
                    str(label),
                ),
            )
        topic_nodes.append({
            "key": str(label),
            "label": display_label,
            "members": members,
            "is_other": str(label) == topic_other,
            "icon": icon,
            "color": color,
            "paragraphs": int(len(paragraphs)),
            "reference_paragraphs": denominator,
            "share": float(len(paragraphs) / denominator * 100),
            "raw_mentions": int(group["candidate_id"].nunique()),
            "speeches": int(group["doc_name"].nunique()),
            "directions": sorted(
                str(value) for value in group["direction"].unique()
            ),
        })
    topic_nodes.sort(
        key=lambda row: (
            topic_order.get(row["key"], len(topic_order) + 1),
            row["label"],
        )
    )

    def link_rows(left: str, right: str) -> list[dict]:
        rows = []
        for (left_key, right_key), group in expanded.groupby(
            [left, right],
            observed=True,
            sort=False,
        ):
            paragraphs = group[
                ["doc_name", "para_idx"]
            ].drop_duplicates()
            rows.append({
                "source": str(left_key),
                "target": str(right_key),
                "paragraphs": int(len(paragraphs)),
                "raw_mentions": int(group["candidate_id"].nunique()),
                "speeches": int(group["doc_name"].nunique()),
                "directions": sorted(
                    str(value) for value in group["direction"].unique()
                ),
            })
        rows.sort(
            key=lambda row: (
                -row["paragraphs"],
                row["source"],
                row["target"],
            )
        )
        return rows

    source_target_links = []
    for (source_key, target_key), group in touching.groupby(
        ["source_node", "target_node"],
        observed=True,
        sort=False,
    ):
        source_target_links.append({
            "source": str(source_key),
            "target": str(target_key),
            "paragraphs": int(
                len(group[["doc_name", "para_idx"]].drop_duplicates())
            ),
            "raw_mentions": int(group["candidate_id"].nunique()),
            "speeches": int(group["doc_name"].nunique()),
            "directions": sorted(
                str(value) for value in group["direction"].unique()
            ),
        })
    source_target_links.sort(
        key=lambda row: (
            -row["paragraphs"],
            row["source"],
            row["target"],
        )
    )

    paths = []
    for (speaker, topic, target, direction), group in expanded.groupby(
        ["speaker", "topic", "target", "direction"],
        observed=True,
        sort=False,
    ):
        paths.append({
            "speaker": str(speaker),
            "topic": str(topic),
            "target": str(target),
            "direction": str(direction),
            "paragraphs": int(
                len(group[["doc_name", "para_idx"]].drop_duplicates())
            ),
            "raw_mentions": int(group["candidate_id"].nunique()),
            "speeches": int(group["doc_name"].nunique()),
        })
    paths.sort(
        key=lambda row: (
            -row["paragraphs"],
            row["speaker"],
            row["topic"],
            row["target"],
        )
    )
    return {
        "source_nodes": actor_nodes(
            "source_node",
            "speaker",
            "source_key",
            source_rank,
            source_other,
        ),
        "topic_nodes": topic_nodes,
        "target_nodes": actor_nodes(
            "target_node",
            "target",
            "target_key",
            target_rank,
            target_other,
        ),
        "source_target_links": source_target_links,
        "source_topic_links": link_rows("source_node", "topic_node"),
        "topic_target_links": link_rows("topic_node", "target_node"),
        "paths": paths,
        "gravity": gravity,
        **_direction_counts(touching),
    }


def _era_echoes(
    spec: era_profiles.EraProfileSpec,
    evidence: pd.DataFrame,
    profile: dict,
) -> dict:
    order = {
        candidate.key: index
        for index, candidate in enumerate(era_profiles.ERA_PROFILE_SPECS)
    }
    incoming = evidence[
        evidence["target_key"].eq(spec.key)
        & evidence["source_key"].map(order).gt(order[spec.key])
    ].copy()
    outgoing = evidence[
        evidence["source_key"].eq(spec.key)
        & evidence["target_key"].map(order).lt(order[spec.key])
    ].copy()
    incoming_topics = _topic_nodes(incoming)
    outgoing_topics = _topic_nodes(outgoing)
    focal_presidents = [row["name"] for row in profile["presidents"]]
    network = _tripartite_network(
        spec,
        incoming,
        outgoing,
        focal_presidents,
    )
    empty_incoming = incoming.iloc[0:0].copy()
    empty_outgoing = outgoing.iloc[0:0].copy()
    directional_views = {
        "incoming": {
            "key": "incoming",
            "label": "Invoked by",
            "description": (
                "Later presidents invoking presidents from this era"
            ),
            "available": not incoming.empty,
            "network": _tripartite_network(
                spec,
                incoming,
                empty_outgoing,
                focal_presidents,
            ),
        },
        "outgoing": {
            "key": "outgoing",
            "label": "Invoking",
            "description": (
                "Presidents from this era invoking earlier presidents"
            ),
            "available": not outgoing.empty,
            "network": _tripartite_network(
                spec,
                empty_incoming,
                outgoing,
                focal_presidents,
            ),
        },
    }
    if spec.key == "founding":
        headline = "Who invoked the Founding—and around what?"
        receipts = [
            {
                "title": "Founding figures invoked",
                "items": _actor_items(incoming, "target"),
            },
            {
                "title": "Presidents looking back",
                "items": _actor_items(incoming, "speaker"),
            },
        ]
    elif spec.key == "present":
        headline = (
            "What were present-era presidents discussing when they invoked "
            "earlier presidents?"
        )
        receipts = [
            {
                "title": "Earlier figures invoked",
                "items": _actor_items(outgoing, "target"),
            },
            {
                "title": "Presidents making the references",
                "items": _actor_items(outgoing, "speaker"),
            },
        ]
    else:
        headline = (
            "What surrounded this era’s references to the past—and later "
            "references to this era?"
        )
        receipts = [
            {
                "title": "Earlier figures this era invoked",
                "items": _actor_items(outgoing, "target"),
            },
            {
                "title": "Later presidents looking back",
                "items": _actor_items(incoming, "speaker"),
            },
        ]
    receipts = [receipt for receipt in receipts if receipt["items"]]
    if incoming_topics and outgoing_topics:
        direction_mode = "both"
    elif incoming_topics:
        direction_mode = "incoming"
    elif outgoing_topics:
        direction_mode = "outgoing"
    else:
        direction_mode = "empty"
    return {
        "title": "Era Echoes",
        "headline": headline,
        "dek": (
            "See who spoke, which president they named, and the assigned "
            "topic of that exact paragraph."
        ),
        "direction_mode": direction_mode,
        "incoming": {
            **_direction_counts(incoming),
            "topics": incoming_topics,
        },
        "outgoing": {
            **_direction_counts(outgoing),
            "topics": outgoing_topics,
        },
        "network": network,
        "directional_views": directional_views,
        "default_direction": (
            "incoming" if not incoming.empty else "outgoing"
        ),
        "receipts": receipts,
        "scope": (
            "Named presidential invocations only. This does not yet capture "
            "broader references such as “the Founders,” the Constitution, "
            "Reconstruction, the New Deal, or Pearl Harbor."
        ),
        "measure_note": (
            "Planet area and satellite area use distinct reference "
            "paragraphs. A satellite’s position is the weighted centroid of "
            "its connections to the focal era’s presidents; deterministic "
            "collision spacing prevents overlap without changing those "
            "weights. Every uncollapsed path remains in the published data. "
            "Topic labels overlap, so paths are non-additive."
        ),
    }


def build_era_contextualizations(
    speeches: pd.DataFrame,
    paragraphs: pd.DataFrame,
    paragraph_annotations: pd.DataFrame,
    taxonomy: dict,
    profiles_by_era: dict[str, dict],
    invocation_evidence: pd.DataFrame,
    topic_life_annotations: (
        dict[tuple[str, str], dict] | None
    ) = None,
) -> dict[str, dict]:
    """Derive all nine recurring Contextualize records."""
    expected_keys = [
        spec.key for spec in era_profiles.ERA_PROFILE_SPECS
    ]
    if list(profiles_by_era) != expected_keys:
        raise ValueError(
            "era contextualizations require profiles in canonical era order"
        )
    speech_required = {"doc_name", "president", "year"}
    missing = sorted(speech_required - set(speeches.columns))
    if missing:
        raise ValueError(
            f"speeches is missing era-contextualization columns: {missing}"
        )
    frame = _strict_one_to_one_merge(
        paragraphs[["doc_name", "para_idx"]],
        paragraph_annotations[["doc_name", "para_idx", "topics"]],
        ["doc_name", "para_idx"],
        "era contextualizations paragraphs x annotations",
    )
    frame = frame.merge(
        speeches[["doc_name", "president", "year"]],
        on="doc_name",
        how="left",
        validate="many_to_one",
    )
    if frame[["president", "year"]].isna().any().any():
        raise ValueError(
            "era contextualizations contain paragraphs with unknown speeches"
        )
    label_map = attention.canonical_label_map(taxonomy)
    frame["topic_set"] = [
        frozenset(attention.normalize_topics(raw, label_map))
        for raw in frame["topics"]
    ]
    frame["era_key"] = era_profiles.story_era_key_series(
        frame["year"], frame["president"]
    )
    frames = {
        spec.key: frame[frame["era_key"].eq(spec.key)].copy()
        for spec in era_profiles.ERA_PROFILE_SPECS
    }
    if any(era_frame.empty for era_frame in frames.values()):
        empty = [key for key, era_frame in frames.items() if era_frame.empty]
        raise ValueError(f"era contextualizations have empty eras: {empty}")
    if topic_life_annotations is None:
        topic_life_annotations = load_topic_life_annotations(
            taxonomy,
            profiles_by_era,
        )

    axis = [
        {
            **asdict(spec),
            "short": ERA_SHORT_LABELS[index],
            "years": f"{spec.start_year}–{spec.end_year}",
            "index": index,
        }
        for index, spec in enumerate(era_profiles.ERA_PROFILE_SPECS)
    ]
    reference_evidence, reference_row_count = _reference_evidence(
        speeches,
        frames,
        invocation_evidence,
    )
    records = {}
    for spec in era_profiles.ERA_PROFILE_SPECS:
        records[spec.key] = {
            **asdict(spec),
            "years": f"{spec.start_year}–{spec.end_year}",
            "era_axis": axis,
            "era_defined": _shared_trajectory_defined(
                spec,
                profiles_by_era[spec.key],
                frames,
            ),
            "topic_life": _topic_life(
                spec,
                profiles_by_era[spec.key],
                frames,
                topic_life_annotations,
            ),
            "era_echoes": _era_echoes(
                spec,
                reference_evidence,
                profiles_by_era[spec.key],
            ),
            "support": {
                "speeches": int(
                    era_profiles.story_era_key_series(
                        speeches["year"], speeches["president"]
                    ).eq(spec.key).sum()
                ),
                "paragraphs": len(frames[spec.key]),
                "invocation_rows": reference_row_count,
            },
        }
    return records


def load_all_era_contextualizations() -> dict[str, dict]:
    """Load current artifacts and derive all reusable contextualizations."""
    data_dir = corpus.DATA_DIR
    taxonomy = attention.load_taxonomy()
    profiles_by_era = era_profiles.load_all_era_profiles()
    return build_era_contextualizations(
        corpus.load(),
        pd.read_parquet(data_dir / "paragraphs.parquet"),
        pd.read_parquet(
            data_dir / "llm_annotations" / "paragraph_annotations.parquet"
        ),
        taxonomy,
        profiles_by_era,
        pd.read_parquet(
            data_dir / "networks" / "invocation_evidence.parquet"
        ),
        load_topic_life_annotations(
            taxonomy,
            profiles_by_era,
        ),
    )


def load_era_contextualization(era: str) -> dict:
    """Load one contextualization record by stable era key or label."""
    spec = era_profiles.resolve_era(era)
    return load_all_era_contextualizations()[spec.key]


def write_era_contextualizations(
    contextualizations: dict[str, dict],
    site_dir: Path,
) -> Path:
    """Publish deterministic Contextualize inputs for era templates."""
    path = site_dir / "data" / "era_contextualizations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONTEXTUALIZATION_SCHEMA,
        "era_order": [
            spec.key for spec in era_profiles.ERA_PROFILE_SPECS
        ],
        "contextualizations": contextualizations,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    return path
