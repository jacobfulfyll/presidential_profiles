"""Deterministic evidence layer for the 1816–1849 expansion story chapter.

The layer measures eight exact level-2 topic rows and one ``enemy_naming`` row
over seven historical five-year periods plus a fixed 1850–1854 next-era
preview.  Every denominator is the full set of corpus paragraphs in the
period, including paragraphs with no assigned topic.  Composite rows are
paragraph-level unions and therefore de-duplicate both repeated labels and
paragraphs carrying more than one constituent label.

Uncertainty follows the site's existing five-year policy in ``bands.py``:
500 percentile-bootstrap draws resample whole speeches, period support sets
``ci_status``, and a separate cell gate withdraws an interval the bootstrap
could not resolve.  The primary/secondary annotator half-range is measured on
the paired paragraph sample and may widen, never narrow, a sampling interval.

This module is pure local computation over frozen inputs.  It never imports an
API client and never writes below ``data/llm_annotations/``.

Run as:
    arch -x86_64 .venv/bin/python -m presidential_profiles.expansion_story
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from . import attention, bands
from .corpus import DATA_DIR


OUTPUT_DIR = DATA_DIR / "expansion_story"
METRICS_PATH = OUTPUT_DIR / "period_metrics.parquet"
META_PATH = OUTPUT_DIR / "meta.json"

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
PARAGRAPH_ISSUES_PATH = DATA_DIR / "paragraph_issues.parquet"
SPEECHES_PATH = DATA_DIR / "speeches.parquet"
TAXONOMY_PATH = DATA_DIR / "llm_annotations" / "taxonomy_v1.json"
PRIMARY_ANNOTATIONS_PATH = (
    DATA_DIR / "llm_annotations" / "paragraph_annotations.parquet"
)
SECONDARY_ANNOTATIONS_PATH = (
    DATA_DIR
    / "llm_annotations"
    / "paragraph_annotations__opus4-8.parquet"
)
PARAGRAPH_ENTITIES_PATH = (
    DATA_DIR / "llm_annotations" / "paragraph_entities.parquet"
)

KEYS = ["doc_name", "para_idx"]
BOOTSTRAP_DRAWS = bands.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = bands.BOOTSTRAP_SEED
CI_LOW = bands.CI_LOW
CI_HIGH = bands.CI_HIGH
MIN_PAIRED_PARAGRAPHS = bands.MIN_PAIRED_PARAGRAPHS

PERIODS = (
    ("1816–20", 1816, 1820, False),
    ("1821–25", 1821, 1825, False),
    ("1826–30", 1826, 1830, False),
    ("1831–35", 1831, 1835, False),
    ("1836–40", 1836, 1840, False),
    ("1841–45", 1841, 1845, False),
    ("1846–49", 1846, 1849, False),
    ("1850–54", 1850, 1854, True),
)
PERIOD_BY_LABEL = {label: (lo, hi, preview) for label, lo, hi, preview in PERIODS}

DIPLOMACY = "Treaties, Diplomacy & International Arbitration"
FEDERAL_FINANCE = "Public Debt, Revenue & Treasury Finance"
TARIFFS = "Tariffs, Reciprocity & Navigation Laws"
BANKING = "National Bank & Banking Crises"
RELATIONS = "Relations with Spain, Mexico & Territorial Claims"
TERRITORIES = "Territorial Organization, Statehood & Insular Governance"
MEXICAN_WAR = "Mexican War"
INDIAN_AFFAIRS = "Indian Affairs, Removal & Allotment"
PUBLIC_LANDS = "Public Lands & Homestead Settlement"
SLAVERY = "Slavery, Emancipation & Sectionalism"

EXTERNAL_ACQUISITION = (RELATIONS, TERRITORIES, MEXICAN_WAR)
NATIVE_REMOVAL_SETTLEMENT = (INDIAN_AFFAIRS, PUBLIC_LANDS)
TERRITORIAL_EXPANSION = EXTERNAL_ACQUISITION + NATIVE_REMOVAL_SETTLEMENT

TOPIC_ROWS: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("Diplomacy and treaties", (DIPLOMACY,)),
        ("Federal finance", (FEDERAL_FINANCE,)),
        ("Tariffs", (TARIFFS,)),
        ("Banking", (BANKING,)),
        ("Territorial expansion", TERRITORIAL_EXPANSION),
        ("↳ External acquisition", EXTERNAL_ACQUISITION),
        ("↳ Native removal and settlement", NATIVE_REMOVAL_SETTLEMENT),
        ("Slavery and sectionalism", (SLAVERY,)),
    ]
)
ADVERSARY_ROW = "Adversary naming"

MONROE_1817 = (
    "/the-presidency/presidential-speeches/"
    "december-2-1817-first-annual-message"
)
JACKSON_BANK_VETO = (
    "/the-presidency/presidential-speeches/july-10-1832-bank-veto"
)
JACKSON_NULLIFICATION = (
    "/the-presidency/presidential-speeches/"
    "december-10-1832-nullification-proclamation"
)
JACKSON_REMOVAL = (
    "/the-presidency/presidential-speeches/"
    "february-15-1832-message-regarding-indian-removal"
)
POLK_WAR = (
    "/the-presidency/presidential-speeches/"
    "may-13-1846-announcement-war-mexico"
)
POLK_SLAVERY = (
    "/the-presidency/presidential-speeches/"
    "august-14-1848-message-regarding-slavery-territories"
)

# Editorial selection is fixed; every key and excerpt is revalidated against
# the frozen paragraph table before the artifact is written.
RECEIPT_SPECS = (
    {
        "group": "monroe",
        "doc_name": MONROE_1817,
        "para_idx": 1,
        "excerpt": (
            "an arrangement which had been commenced by my predecessor with "
            "the British government for the reduction of the naval force by "
            "Great Britain and the United States on the lakes has been concluded"
        ),
        "reason": "postwar diplomacy",
    },
    {
        "group": "monroe",
        "doc_name": MONROE_1817,
        "para_idx": 15,
        "excerpt": (
            "the receipts from those lands will annually add to the public "
            "revenue the sum of $1.5 million"
        ),
        "reason": "public credit and land revenue",
    },
    {
        "group": "monroe",
        "doc_name": MONROE_1817,
        "para_idx": 18,
        "excerpt": (
            "By these purchases the Indian title, with moderate reservations, "
            "has been extinguished"
        ),
        "reason": "Native land acquisition",
    },
    {
        "group": "monroe",
        "doc_name": MONROE_1817,
        "para_idx": 22,
        "excerpt": (
            "Several new states have been admitted into our Union to the west "
            "and south, and territorial governments, happily organized"
        ),
        "reason": "settlement and territorial administration",
    },
    {
        "group": "jackson",
        "doc_name": JACKSON_BANK_VETO,
        "para_idx": 0,
        "excerpt": (
            "some of the powers and privileges possessed by the existing bank "
            "are unauthorized by the Constitution, subversive of the rights "
            "of the States, and dangerous to the liberties of the people"
        ),
        "reason": "Bank Veto",
    },
    {
        "group": "jackson",
        "doc_name": JACKSON_NULLIFICATION,
        "para_idx": 53,
        "excerpt": "Disunion by armed force is treason.",
        "reason": "Nullification Proclamation",
    },
    {
        "group": "jackson",
        "doc_name": JACKSON_REMOVAL,
        "para_idx": 0,
        "excerpt": (
            "all the arrangements necessary to the complete execution of the "
            "plan of removal"
        ),
        "reason": "administrative removal message",
    },
    {
        "group": "polk",
        "doc_name": POLK_WAR,
        "para_idx": 0,
        "excerpt": (
            "by the act of the Republic of Mexico a state of war exists "
            "between that Government and the United States"
        ),
        "reason": "War Message",
    },
    {
        "group": "polk",
        "doc_name": POLK_SLAVERY,
        "para_idx": 2,
        "excerpt": "This question is slavery.",
        "reason": "sectional hinge in the acquired territories",
    },
)

INPUT_PATHS = OrderedDict(
    [
        ("paragraphs", PARAGRAPHS_PATH),
        ("paragraph_issues", PARAGRAPH_ISSUES_PATH),
        ("speeches", SPEECHES_PATH),
        ("taxonomy", TAXONOMY_PATH),
        ("primary_annotations", PRIMARY_ANNOTATIONS_PATH),
        ("secondary_annotations", SECONDARY_ANNOTATIONS_PATH),
        ("paragraph_entities", PARAGRAPH_ENTITIES_PATH),
    ]
)

METRIC_COLUMNS = [
    "period",
    "period_order",
    "period_start",
    "period_end",
    "is_preview",
    "row_kind",
    "row_order",
    "display_row",
    "constituent_labels",
    "point",
    "n_labeled_paragraphs",
    "lo_sampling",
    "hi_sampling",
    "lo",
    "hi",
    "n_paragraphs",
    "n_speeches",
    "n_paired_paragraphs",
    "paired_primary_share",
    "paired_secondary_share",
    "ci_status",
    "interval_unresolvable",
    "disagreement_band_applied",
    "disagreement_half_width",
    "ci_components",
    "disagreement_status",
    "agreement_source",
]

SCHEMA_DESCRIPTIONS = {
    "period": "Fixed displayed period label.",
    "period_order": "Zero-based left-to-right column order.",
    "period_start": "Inclusive first year.",
    "period_end": "Inclusive last year.",
    "is_preview": "True only for the 1850–1854 next-era coda.",
    "row_kind": "topic_union or adversary_rate.",
    "row_order": "Zero-based vertical order; adversary follows the eight topic rows.",
    "display_row": "Reader-facing row label.",
    "constituent_labels": "JSON array of exact topic labels, or enemy_naming=True.",
    "point": "Observed share as a fraction of all period paragraphs.",
    "n_labeled_paragraphs": "Unique numerator paragraph keys.",
    "lo_sampling": "Speech-clustered sampling lower bound, fraction.",
    "hi_sampling": "Speech-clustered sampling upper bound, fraction.",
    "lo": "Published lower bound after disagreement widening, fraction.",
    "hi": "Published upper bound after disagreement widening, fraction.",
    "n_paragraphs": "All period paragraphs, including topic-free paragraphs.",
    "n_speeches": "Distinct speech clusters in the period.",
    "n_paired_paragraphs": "Paragraphs labeled by both models in the period.",
    "paired_primary_share": "Primary-model share on the paired subset.",
    "paired_secondary_share": "Secondary-model share on the paired subset.",
    "ci_status": "Existing period-level cluster/paragraph support gate.",
    "interval_unresolvable": "Existing per-cell unresolved-bootstrap gate.",
    "disagreement_band_applied": "Whether measured model disagreement widened the interval.",
    "disagreement_half_width": "Half-range between paired-model shares, fraction.",
    "ci_components": "sampling_only or sampling+annotator_disagreement.",
    "disagreement_status": "Reason disagreement was applied or unavailable.",
    "agreement_source": "Named primary and secondary annotation artifacts.",
}


def _key_tuples(frame: pd.DataFrame) -> set[tuple[object, object]]:
    return set(frame[KEYS].itertuples(index=False, name=None))


def _assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in KEYS if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing key columns {missing}")
    duplicate = frame.duplicated(KEYS, keep=False)
    if duplicate.any():
        examples = list(
            frame.loc[duplicate, KEYS]
            .drop_duplicates()
            .head(5)
            .itertuples(index=False, name=None)
        )
        raise ValueError(
            f"{name} has {int(duplicate.sum())} rows on duplicate paragraph "
            f"keys; e.g. {examples}"
        )


def require_same_key_set(
    left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str
) -> None:
    """Require unique, identical paragraph key sets before a one-to-one join."""
    _assert_unique_keys(left, left_name)
    _assert_unique_keys(right, right_name)
    left_keys, right_keys = _key_tuples(left), _key_tuples(right)
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)[:5]
        extra = sorted(right_keys - left_keys)[:5]
        raise ValueError(
            f"{left_name} and {right_name} paragraph key sets diverge "
            f"(missing_from_{right_name}={len(left_keys - right_keys)}, "
            f"extra_in_{right_name}={len(right_keys - left_keys)}; "
            f"examples missing={missing}, extra={extra})"
        )


def _require_subset_keys(
    subset: pd.DataFrame, universe: pd.DataFrame, subset_name: str
) -> None:
    _assert_unique_keys(subset, subset_name)
    unknown = _key_tuples(subset) - _key_tuples(universe)
    if unknown:
        raise ValueError(
            f"{subset_name} contains {len(unknown)} paragraph keys absent from "
            f"the corpus; e.g. {sorted(unknown)[:5]}"
        )


def _period_label(year: int) -> str | None:
    for label, lo, hi, _ in PERIODS:
        if lo <= int(year) <= hi:
            return label
    return None


def _normalized_topic_sets(
    raw: Iterable[object], label_map: dict[str, str]
) -> list[frozenset[str]]:
    return [
        frozenset(attention.normalize_topics(value, label_map)) for value in raw
    ]


def prepare_inputs(
    paragraphs: pd.DataFrame,
    paragraph_issues: pd.DataFrame,
    speeches: pd.DataFrame,
    primary_annotations: pd.DataFrame,
    secondary_annotations: pd.DataFrame,
    entities: pd.DataFrame,
    taxonomy: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build keyed primary, paired-model, and adversarial-entity frames."""
    required_para = {"doc_name", "para_idx", "text", "word_count"}
    required_issues = {"doc_name", "para_idx", "year"}
    required_ann = {"doc_name", "para_idx", "topics", "enemy_naming"}
    for name, frame, required in (
        ("paragraphs", paragraphs, required_para),
        ("paragraph_issues", paragraph_issues, required_issues),
        ("primary_annotations", primary_annotations, required_ann),
        ("secondary_annotations", secondary_annotations, required_ann),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} is missing required columns {sorted(missing)}")

    require_same_key_set(
        paragraphs, paragraph_issues, "paragraphs", "paragraph_issues"
    )
    require_same_key_set(
        paragraphs, primary_annotations, "paragraphs", "primary_annotations"
    )
    _require_subset_keys(
        secondary_annotations, paragraphs, "secondary_annotations"
    )

    master = (
        paragraphs[["doc_name", "para_idx", "text", "word_count"]]
        .merge(
            paragraph_issues[["doc_name", "para_idx", "year"]],
            on=KEYS,
            validate="one_to_one",
        )
        .merge(
            primary_annotations[["doc_name", "para_idx", "topics", "enemy_naming"]],
            on=KEYS,
            validate="one_to_one",
        )
        .rename(
            columns={
                "topics": "topics_primary_raw",
                "enemy_naming": "enemy_naming_primary",
            }
        )
    )
    if len(master) != len(paragraphs):
        raise ValueError(
            "paragraph keyed join changed row count despite preflight key checks"
        )

    if speeches["doc_name"].duplicated().any():
        raise ValueError("speeches has duplicate doc_name keys")
    missing_docs = set(master["doc_name"]) - set(speeches["doc_name"])
    if missing_docs:
        raise ValueError(
            f"{len(missing_docs)} paragraph doc_name values are absent from "
            f"speeches; e.g. {sorted(missing_docs)[:5]}"
        )

    label_map = attention.canonical_label_map(taxonomy)
    master["topics_primary"] = _normalized_topic_sets(
        master["topics_primary_raw"], label_map
    )
    master["period"] = master["year"].map(_period_label)
    master = master[master["period"].notna()].copy()
    order = {label: i for i, (label, _, _, _) in enumerate(PERIODS)}
    master["period_order"] = master["period"].map(order).astype(int)

    paired = (
        secondary_annotations[
            ["doc_name", "para_idx", "topics", "enemy_naming"]
        ]
        .rename(
            columns={
                "topics": "topics_secondary_raw",
                "enemy_naming": "enemy_naming_secondary",
            }
        )
        .merge(
            master[
                [
                    "doc_name",
                    "para_idx",
                    "period",
                    "period_order",
                    "topics_primary",
                    "enemy_naming_primary",
                ]
            ],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    )
    expected_in_window = secondary_annotations.merge(
        paragraph_issues[["doc_name", "para_idx", "year"]],
        on=KEYS,
        validate="one_to_one",
    )
    expected_in_window = expected_in_window[
        expected_in_window["year"].map(_period_label).notna()
    ]
    if len(paired) != len(expected_in_window):
        raise ValueError(
            "paired-model join did not retain every secondary paragraph in "
            "the displayed periods"
        )
    paired["topics_secondary"] = _normalized_topic_sets(
        paired["topics_secondary_raw"], label_map
    )

    entity_required = {"doc_name", "para_idx", "entity", "type", "stance"}
    missing_entity = entity_required - set(entities)
    if missing_entity:
        raise ValueError(
            f"paragraph_entities is missing required columns "
            f"{sorted(missing_entity)}"
        )
    entity_keys = entities[KEYS].drop_duplicates()
    unknown_entity_keys = _key_tuples(entity_keys) - _key_tuples(paragraphs)
    if unknown_entity_keys:
        raise ValueError(
            f"paragraph_entities contains {len(unknown_entity_keys)} unknown "
            f"paragraph keys; e.g. {sorted(unknown_entity_keys)[:5]}"
        )
    adversarial = entities[entities["stance"].eq("adversarial")].merge(
        master[["doc_name", "para_idx", "period"]],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    return (
        master.sort_values(KEYS).reset_index(drop=True),
        paired.sort_values(KEYS).reset_index(drop=True),
        adversarial.sort_values([*KEYS, "entity", "type"]).reset_index(drop=True),
    )


def _indicator_frame(
    frame: pd.DataFrame, topic_column: str, enemy_column: str
) -> pd.DataFrame:
    out = frame[["doc_name", "period"]].copy()
    for display_row, labels in TOPIC_ROWS.items():
        label_set = frozenset(labels)
        out[display_row] = frame[topic_column].map(
            lambda assigned, wanted=label_set: not assigned.isdisjoint(wanted)
        )
    out[ADVERSARY_ROW] = frame[enemy_column].astype(bool)
    return out


def bootstrap_period(
    indicators: pd.DataFrame,
    columns: list[str],
    *,
    period_start: int,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Speech-clustered point estimates and intervals for one period."""
    per_doc = indicators.groupby("doc_name", sort=True)
    counts = per_doc[columns].sum().to_numpy(dtype=float)
    sizes = per_doc.size().to_numpy(dtype=float)
    n_speeches = len(sizes)
    n_paragraphs = int(sizes.sum())
    if n_speeches == 0 or n_paragraphs == 0:
        raise ValueError("cannot estimate an empty expansion-story period")
    observed = counts.sum(axis=0) / n_paragraphs
    status = bands._ci_status(n_speeches, n_paragraphs)

    if status == "suppressed_n_floor":
        lo = hi = np.full(len(columns), np.nan)
    else:
        rng = np.random.default_rng([seed, int(period_start)])
        draws = rng.integers(0, n_speeches, size=(n_draws, n_speeches))
        weights = np.zeros((n_draws, n_speeches), dtype=float)
        np.add.at(
            weights,
            (
                np.repeat(np.arange(n_draws), n_speeches),
                draws.ravel(),
            ),
            1.0,
        )
        replicates = (weights @ counts) / (weights @ sizes)[:, None]
        lo = np.percentile(replicates, CI_LOW, axis=0)
        hi = np.percentile(replicates, CI_HIGH, axis=0)

    unresolvable = bands._unresolvable_interval(lo, hi, n_speeches)
    lo = np.where(unresolvable, np.nan, lo)
    hi = np.where(unresolvable, np.nan, hi)
    return {
        "observed": observed,
        "n_labeled": counts.sum(axis=0).astype(int),
        "lo": lo,
        "hi": hi,
        "n_paragraphs": n_paragraphs,
        "n_speeches": n_speeches,
        "ci_status": status,
        "interval_unresolvable": unresolvable,
    }


def _paired_disagreement(
    paired_indicators: pd.DataFrame, columns: list[str]
) -> dict[str, object]:
    n = len(paired_indicators)
    if n:
        primary = paired_indicators[[f"{name}__primary" for name in columns]]
        secondary = paired_indicators[
            [f"{name}__secondary" for name in columns]
        ]
        primary_share = primary.mean().to_numpy(dtype=float)
        secondary_share = secondary.mean().to_numpy(dtype=float)
    else:
        primary_share = secondary_share = np.full(len(columns), np.nan)
    available = n >= MIN_PAIRED_PARAGRAPHS
    half_width = (
        np.abs(primary_share - secondary_share) / 2
        if available
        else np.full(len(columns), np.nan)
    )
    return {
        "n": n,
        "primary_share": primary_share,
        "secondary_share": secondary_share,
        "half_width": half_width,
        "available": available,
    }


def build_period_metrics(
    paragraphs: pd.DataFrame | None = None,
    paragraph_issues: pd.DataFrame | None = None,
    speeches: pd.DataFrame | None = None,
    primary_annotations: pd.DataFrame | None = None,
    secondary_annotations: pd.DataFrame | None = None,
    entities: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Build the fixed 8×8 topic matrix plus the aligned adversary row."""
    paragraphs = (
        pd.read_parquet(PARAGRAPHS_PATH)
        if paragraphs is None
        else paragraphs
    )
    paragraph_issues = (
        pd.read_parquet(PARAGRAPH_ISSUES_PATH)
        if paragraph_issues is None
        else paragraph_issues
    )
    speeches = (
        pd.read_parquet(SPEECHES_PATH) if speeches is None else speeches
    )
    primary_annotations = (
        pd.read_parquet(PRIMARY_ANNOTATIONS_PATH)
        if primary_annotations is None
        else primary_annotations
    )
    secondary_annotations = (
        pd.read_parquet(SECONDARY_ANNOTATIONS_PATH)
        if secondary_annotations is None
        else secondary_annotations
    )
    entities = (
        pd.read_parquet(PARAGRAPH_ENTITIES_PATH)
        if entities is None
        else entities
    )
    taxonomy = taxonomy or attention.load_taxonomy(TAXONOMY_PATH)

    master, paired, _ = prepare_inputs(
        paragraphs,
        paragraph_issues,
        speeches,
        primary_annotations,
        secondary_annotations,
        entities,
        taxonomy,
    )
    primary_indicators = _indicator_frame(
        master, "topics_primary", "enemy_naming_primary"
    )
    paired_primary = _indicator_frame(
        paired, "topics_primary", "enemy_naming_primary"
    )
    paired_secondary = _indicator_frame(
        paired, "topics_secondary", "enemy_naming_secondary"
    )

    columns = [*TOPIC_ROWS, ADVERSARY_ROW]
    rows: list[dict[str, object]] = []
    agreement_source = (
        f"{PRIMARY_ANNOTATIONS_PATH.name} vs "
        f"{SECONDARY_ANNOTATIONS_PATH.name}"
    )
    for period_order, (period, lo_year, hi_year, preview) in enumerate(PERIODS):
        primary_chunk = primary_indicators[
            primary_indicators["period"].eq(period)
        ]
        sampling = bootstrap_period(
            primary_chunk,
            columns,
            period_start=lo_year,
            n_draws=n_draws,
            seed=seed,
        )

        paired_mask = paired["period"].eq(period)
        combined = pd.DataFrame(index=paired.loc[paired_mask].index)
        for name in columns:
            combined[f"{name}__primary"] = paired_primary.loc[
                paired_mask, name
            ].to_numpy()
            combined[f"{name}__secondary"] = paired_secondary.loc[
                paired_mask, name
            ].to_numpy()
        disagreement = _paired_disagreement(combined, columns)

        for row_order, name in enumerate(columns):
            sampling_lo = float(sampling["lo"][row_order])
            sampling_hi = float(sampling["hi"][row_order])
            interval_available = np.isfinite(sampling_lo) and np.isfinite(
                sampling_hi
            )
            half_width = float(disagreement["half_width"][row_order])
            disagreement_available = bool(disagreement["available"])
            applied = disagreement_available and interval_available
            if applied:
                published_lo = max(0.0, sampling_lo - half_width)
                published_hi = min(1.0, sampling_hi + half_width)
                disagreement_status = bands.MEASURED
                ci_components = "sampling+annotator_disagreement"
            else:
                published_lo, published_hi = sampling_lo, sampling_hi
                disagreement_status = (
                    bands.NO_INTERVAL
                    if disagreement_available
                    else bands.THIN_PAIRED
                )
                ci_components = "sampling_only"
                half_width = np.nan

            labels = (
                list(TOPIC_ROWS[name])
                if name in TOPIC_ROWS
                else ["enemy_naming=True"]
            )
            rows.append(
                {
                    "period": period,
                    "period_order": period_order,
                    "period_start": lo_year,
                    "period_end": hi_year,
                    "is_preview": preview,
                    "row_kind": (
                        "topic_union"
                        if name in TOPIC_ROWS
                        else "adversary_rate"
                    ),
                    "row_order": row_order,
                    "display_row": name,
                    "constituent_labels": json.dumps(
                        labels, ensure_ascii=False, separators=(",", ":")
                    ),
                    "point": float(sampling["observed"][row_order]),
                    "n_labeled_paragraphs": int(
                        sampling["n_labeled"][row_order]
                    ),
                    "lo_sampling": sampling_lo,
                    "hi_sampling": sampling_hi,
                    "lo": published_lo,
                    "hi": published_hi,
                    "n_paragraphs": int(sampling["n_paragraphs"]),
                    "n_speeches": int(sampling["n_speeches"]),
                    "n_paired_paragraphs": int(disagreement["n"]),
                    "paired_primary_share": float(
                        disagreement["primary_share"][row_order]
                    ),
                    "paired_secondary_share": float(
                        disagreement["secondary_share"][row_order]
                    ),
                    "ci_status": str(sampling["ci_status"]),
                    "interval_unresolvable": bool(
                        sampling["interval_unresolvable"][row_order]
                    ),
                    "disagreement_band_applied": applied,
                    "disagreement_half_width": half_width,
                    "ci_components": ci_components,
                    "disagreement_status": disagreement_status,
                    "agreement_source": agreement_source,
                }
            )

    out = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    out["is_preview"] = out["is_preview"].astype(bool)
    out["interval_unresolvable"] = out["interval_unresolvable"].astype(bool)
    out["disagreement_band_applied"] = out[
        "disagreement_band_applied"
    ].astype(bool)
    return out.sort_values(
        ["row_order", "period_order"], kind="mergesort"
    ).reset_index(drop=True)


def validate_receipts(
    paragraphs: pd.DataFrame, speeches: pd.DataFrame
) -> list[dict[str, object]]:
    """Validate and enrich every fixed receipt key and excerpt."""
    _assert_unique_keys(paragraphs, "paragraphs")
    if speeches["doc_name"].duplicated().any():
        raise ValueError("speeches has duplicate doc_name keys")
    speech_lookup = speeches.set_index("doc_name")
    paragraph_lookup = paragraphs.set_index(KEYS)
    validated: list[dict[str, object]] = []
    for spec in RECEIPT_SPECS:
        key = (spec["doc_name"], spec["para_idx"])
        if key not in paragraph_lookup.index:
            raise ValueError(f"receipt paragraph key is absent: {key}")
        if spec["doc_name"] not in speech_lookup.index:
            raise ValueError(
                f"receipt doc_name is absent from speeches: {spec['doc_name']}"
            )
        paragraph = paragraph_lookup.loc[key]
        excerpt = str(spec["excerpt"])
        if excerpt not in str(paragraph["text"]):
            raise ValueError(
                f"receipt excerpt is not an exact substring of {key}"
            )
        speech = speech_lookup.loc[spec["doc_name"]]
        validated.append(
            {
                **spec,
                "title": str(speech["title"]),
                "president": str(speech["president"]),
                "year": int(speech["year"]),
            }
        )
    return validated


def _speech_enemy_share(master: pd.DataFrame, doc_name: str) -> dict[str, object]:
    chunk = master[master["doc_name"].eq(doc_name)]
    if chunk.empty:
        raise ValueError(f"character receipt speech has no paragraphs: {doc_name}")
    return {
        "doc_name": doc_name,
        "n_enemy_naming": int(chunk["enemy_naming_primary"].sum()),
        "n_paragraphs": int(len(chunk)),
        "share": float(chunk["enemy_naming_primary"].mean()),
    }


def _anchored_entity_names(
    adversarial: pd.DataFrame, anchors: tuple[str, ...]
) -> list[dict[str, object]]:
    rows = []
    for anchor in anchors:
        candidates = adversarial[
            adversarial["entity"].str.contains(
                anchor, case=False, regex=False, na=False
            )
        ]
        if candidates.empty:
            raise ValueError(
                f"no adversarial entity in the requested period contains "
                f"{anchor!r}"
            )
        counts = candidates["entity"].value_counts()
        rows.append({"entity": str(counts.index[0]), "mentions": int(counts.iloc[0])})
    return rows


def _entity_annotation(
    adversarial: pd.DataFrame,
    period: str,
    anchors: tuple[str, ...],
) -> dict[str, object]:
    chunk = adversarial[adversarial["period"].eq(period)]
    if chunk.empty:
        raise ValueError(f"{period} has no adversarial entity mentions")
    counts = chunk["type"].value_counts()
    dominant = str(counts.index[0])
    return {
        "period": period,
        "dominant_type": dominant,
        "dominant_type_mentions": int(counts.iloc[0]),
        "n_adversarial_mentions": int(len(chunk)),
        "dominant_type_share": float(counts.iloc[0] / len(chunk)),
        "recurring_names": _anchored_entity_names(chunk, anchors),
    }


def derive_story_details(
    paragraphs: pd.DataFrame,
    paragraph_issues: pd.DataFrame,
    speeches: pd.DataFrame,
    primary_annotations: pd.DataFrame,
    secondary_annotations: pd.DataFrame,
    entities: pd.DataFrame,
    taxonomy: dict,
) -> dict[str, object]:
    master, _, adversarial = prepare_inputs(
        paragraphs,
        paragraph_issues,
        speeches,
        primary_annotations,
        secondary_annotations,
        entities,
        taxonomy,
    )
    return {
        "receipts": validate_receipts(paragraphs, speeches),
        "jackson_character": {
            "bank_veto": _speech_enemy_share(master, JACKSON_BANK_VETO),
            "nullification_proclamation": _speech_enemy_share(
                master, JACKSON_NULLIFICATION
            ),
            "removal_message": _speech_enemy_share(master, JACKSON_REMOVAL),
        },
        "entity_annotations": {
            "1831–35": _entity_annotation(
                adversarial, "1831–35", ("bank", "senate")
            ),
            "1846–49": _entity_annotation(
                adversarial, "1846–49", ("mexico",)
            ),
        },
    }


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_schema(table: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "name": column,
            "dtype": str(table[column].dtype),
            "description": SCHEMA_DESCRIPTIONS[column],
        }
        for column in METRIC_COLUMNS
    ]


def build_meta(
    table: pd.DataFrame,
    story_details: dict[str, object],
    *,
    n_draws: int,
    seed: int,
) -> dict[str, object]:
    paired_total = pd.read_parquet(
        SECONDARY_ANNOTATIONS_PATH, columns=KEYS
    )
    corpus_total = pd.read_parquet(PARAGRAPHS_PATH, columns=KEYS)
    return {
        "artifact": "data/expansion_story/period_metrics.parquet",
        "input_fingerprints": {
            name: {
                "path": str(path.relative_to(DATA_DIR.parent)),
                "sha256": _sha256(path),
            }
            for name, path in INPUT_PATHS.items()
        },
        "periods": [
            {
                "label": label,
                "start": lo,
                "end": hi,
                "is_preview": preview,
            }
            for label, lo, hi, preview in PERIODS
        ],
        "topic_unions": {
            display: list(labels) for display, labels in TOPIC_ROWS.items()
        },
        "denominator": (
            "Every unique (doc_name, para_idx) paragraph key in the period, "
            "including paragraphs assigned no topic."
        ),
        "normalization": {
            "rule": (
                "taxonomy_v1-backed casefold normalization; unknown labels "
                "raise; duplicate canonical labels within a paragraph are removed"
            ),
            "composite_rule": (
                "A paragraph counts once when it carries one or more constituent "
                "labels; marginal topic intervals are never combined."
            ),
            "year_source": "paragraph_issues.parquet keyed by (doc_name, para_idx)",
        },
        "bootstrap": {
            "unit": "speech (doc_name)",
            "draws": n_draws,
            "seed": seed,
            "seed_scope": "[seed, period_start]",
            "percentiles": [CI_LOW, CI_HIGH],
        },
        "confidence_gates": {
            "min_speeches_for_ci": bands.MIN_CLUSTERS_FOR_CI,
            "low_cluster_caution": bands.LOW_CLUSTER_CAUTION,
            "min_period_paragraphs": bands.MIN_PERIOD_PARAGRAPHS,
            "min_clusters_for_resolvable_ci": (
                bands.MIN_CLUSTERS_FOR_RESOLVABLE_CI
            ),
            "interval_unresolvable_is_separate": True,
        },
        "paired_model": {
            "primary": PRIMARY_ANNOTATIONS_PATH.name,
            "secondary": SECONDARY_ANNOTATIONS_PATH.name,
            "n_paired_paragraphs": int(len(paired_total)),
            "n_corpus_paragraphs": int(len(corpus_total)),
            "coverage_fraction": float(len(paired_total) / len(corpus_total)),
            "min_paired_paragraphs_per_period": MIN_PAIRED_PARAGRAPHS,
            "composition_rule": (
                "Published bounds widen sampling bounds by the paired-model "
                "half-range when paired support and a sampling interval exist; "
                "bounds never narrow."
            ),
        },
        "story_details": story_details,
        "artifact_schema": _artifact_schema(table),
    }


def write_artifact(
    out_dir: Path = OUTPUT_DIR,
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build and atomically replace the deterministic table and metadata."""
    out_dir = Path(out_dir)
    if out_dir.resolve() == OUTPUT_DIR.resolve() and (
        n_draws != BOOTSTRAP_DRAWS or seed != BOOTSTRAP_SEED
    ):
        raise ValueError(
            "refusing to overwrite the published expansion-story artifact "
            "with non-default draws or seed; pass --out-dir for experiments"
        )

    paragraphs = pd.read_parquet(PARAGRAPHS_PATH)
    paragraph_issues = pd.read_parquet(PARAGRAPH_ISSUES_PATH)
    speeches = pd.read_parquet(SPEECHES_PATH)
    primary = pd.read_parquet(PRIMARY_ANNOTATIONS_PATH)
    secondary = pd.read_parquet(SECONDARY_ANNOTATIONS_PATH)
    entities = pd.read_parquet(PARAGRAPH_ENTITIES_PATH)
    taxonomy = attention.load_taxonomy(TAXONOMY_PATH)

    table = build_period_metrics(
        paragraphs,
        paragraph_issues,
        speeches,
        primary,
        secondary,
        entities,
        taxonomy,
        n_draws=n_draws,
        seed=seed,
    )
    story_details = derive_story_details(
        paragraphs,
        paragraph_issues,
        speeches,
        primary,
        secondary,
        entities,
        taxonomy,
    )
    meta = build_meta(
        table, story_details, n_draws=n_draws, seed=seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / METRICS_PATH.name
    meta_path = out_dir / META_PATH.name
    with tempfile.TemporaryDirectory(dir=out_dir) as temp_name:
        temp_dir = Path(temp_name)
        temp_metrics = temp_dir / metrics_path.name
        temp_meta = temp_dir / meta_path.name
        table.to_parquet(temp_metrics, index=False)
        temp_meta.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        )
        temp_metrics.replace(metrics_path)
        temp_meta.replace(meta_path)
    return table, meta


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    table = pd.read_parquet(path)
    missing = set(METRIC_COLUMNS) - set(table)
    if missing:
        raise ValueError(
            f"expansion story artifact is missing columns {sorted(missing)}; "
            "rebuild it with python -m presidential_profiles.expansion_story"
        )
    return table[METRIC_COLUMNS]


def load_meta(path: Path = META_PATH) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _format_interval(row: pd.Series) -> str:
    if bool(row["interval_unresolvable"]):
        return "interval unresolvable"
    if pd.isna(row["lo"]) or pd.isna(row["hi"]):
        return "interval suppressed"
    return f"{row['lo'] * 100:.1f}–{row['hi'] * 100:.1f}%"


def accessible_table_html(
    table: pd.DataFrame, *, include_preview: bool = True
) -> str:
    """Native no-JavaScript fallback for the full matrix and adversary row."""
    topic = table[table["row_kind"].eq("topic_union")]
    adversary = table[table["row_kind"].eq("adversary_rate")]
    period_labels = [
        label
        for label, _, _, is_preview in PERIODS
        if include_preview or not is_preview
    ]

    header = "".join(
        f'<th scope="col">{escape(label)}'
        f'{" · next-era preview" if label == "1850–54" else ""}</th>'
        for label in period_labels
    )
    body_rows = []
    for display_row in TOPIC_ROWS:
        cells = []
        for period in period_labels:
            row = topic[
                topic["display_row"].eq(display_row)
                & topic["period"].eq(period)
            ].iloc[0]
            cells.append(
                "<td>"
                f"<strong>{row['point'] * 100:.1f}%</strong><br>"
                f"95% interval {_format_interval(row)}<br>"
                f"{int(row['n_paragraphs']):,} paragraphs · "
                f"{int(row['n_speeches'])} speeches<br>"
                f"confidence: {escape(str(row['ci_status']))}; "
                f"model disagreement: "
                f"{escape(str(row['disagreement_status']))}"
                "</td>"
            )
        body_rows.append(
            f'<tr><th scope="row">{escape(display_row)}</th>'
            + "".join(cells)
            + "</tr>"
        )

    adversary_cells = []
    for period in period_labels:
        row = adversary[adversary["period"].eq(period)].iloc[0]
        adversary_cells.append(
            "<td>"
            f"<strong>{row['point'] * 100:.1f}%</strong><br>"
            f"95% interval {_format_interval(row)}<br>"
            f"{int(row['n_paragraphs']):,} paragraphs · "
            f"{int(row['n_speeches'])} speeches<br>"
            f"confidence: {escape(str(row['ci_status']))}; "
            f"model disagreement: "
            f"{escape(str(row['disagreement_status']))}"
            "</td>"
        )
    body_rows.append(
        '<tr class="adversary-row"><th scope="row">'
        "Adversary naming · enemy_naming=True</th>"
        + "".join(adversary_cells)
        + "</tr>"
    )

    return (
        '<details class="expansion-story-table">'
        "<summary>Text alternative · topic matrix, intervals, and support</summary>"
        '<div class="table-scroll"><table><thead><tr>'
        '<th scope="col">Measure</th>'
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "</div></details>"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic 1816–1854 expansion story layer."
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    table, meta = write_artifact(
        args.out_dir, n_draws=args.draws, seed=args.seed
    )
    if not args.quiet:
        print(
            f"Wrote {len(table)} rows to "
            f"{args.out_dir / METRICS_PATH.name}"
        )
        print(
            "Paired-model coverage: "
            f"{meta['paired_model']['n_paired_paragraphs']:,} / "
            f"{meta['paired_model']['n_corpus_paragraphs']:,} paragraphs"
        )


if __name__ == "__main__":
    main()
