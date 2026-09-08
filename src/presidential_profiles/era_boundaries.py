"""Deterministic evidence for choosing editorial era boundaries.

The first Era Atlas periodization used globally aligned eight-year bins.  This
module retains that audit trail, but makes the primary question match the
editorial task more closely:

* compare every possible year with both four- and eight-year symmetric windows;
* combine their percentile ranks so one arbitrary window length cannot decide;
* divide presidential-cycle four-year units into contiguous eras by minimizing
  within-era squared distance;
* audit the reviewed nine-era specification across cluster counts, duration
  constraints, grid anchors, axis-family weighting, corpus endpoints, axis
  omissions, and a common-genre subset.

The complete segmentation selects persistent rhetorical regimes.  The sliding
curves remain an abruptness diagnostic: several high neighboring years may
describe one turbulent transition mountain rather than several usable cuts.
Exact editorial dates can snap a stable data-derived zone to a historically
legible accession or institutional event, with the fit cost kept visible.

Run:
    arch -x86_64 .venv/bin/python -m presidential_profiles.era_boundaries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import era_profiles, eras
from .attention import load_taxonomy
from .corpus import DATA_DIR

ANALYSIS_DIR = DATA_DIR / "era_boundaries"
SLIDING_SCORES_PATH = ANALYSIS_DIR / "sliding_scores.parquet"
CLUSTER_STABILITY_PATH = ANALYSIS_DIR / "cluster_stability.parquet"
CONSENSUS_SCORES_PATH = ANALYSIS_DIR / "consensus_scores.parquet"
CYCLE_FINGERPRINTS_PATH = ANALYSIS_DIR / "cycle4_fingerprints.parquet"
SEGMENTATION_PATH = ANALYSIS_DIR / "segmentation.parquet"
SENSITIVITY_PATH = ANALYSIS_DIR / "sensitivity.parquet"
BOUNDARY_EVIDENCE_PATH = ANALYSIS_DIR / "boundary_evidence.parquet"
BOUNDARY_DRIVERS_PATH = ANALYSIS_DIR / "boundary_drivers.parquet"
META_PATH = ANALYSIS_DIR / "meta.json"

SCHEMA_VERSION = "era-boundaries-v3"
WINDOW_YEARS = 8
WINDOW_OPTIONS = (4, 8)
PRIMARY_K = 9
K_RANGE = range(2, 13)
AXIS_GROUPS = ("topic", "combat", "register", "marker", "stat", "genre")
MIN_SEGMENT_UNITS = 3
# The Compromise-era editorial boundary falls inside the 1849–52 cycle; all
# other reviewed starts already coincide with the 1789-anchored grid.
STORY_TO_GRID_START = {1850: 1849}
REVIEWED_GRID_STARTS = tuple(
    STORY_TO_GRID_START.get(spec.start_year, spec.start_year)
    for spec in era_profiles.ERA_PROFILE_SPECS
)
REVIEWED_MODEL_ADJUSTMENTS = {
    "expansion": 1805,
    "war-new-deal": 1937,
}
GRID_OFFSETS = (0, 1, 2, 3)
MIN_SEGMENT_OPTIONS = (3, 4)
WEIGHTING_OPTIONS = ("per_axis", "per_axis_family")
SOTU_SPEECH_TYPE = "state_of_the_union_or_annual_message"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bin_fingerprint_frame(
    fingerprints: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Reconstruct the bin-grain raw matrix from the governed long artifact."""
    required = {
        "grain", "unit", "measure", "axis_group", "value_raw",
        "n_paragraphs", "n_speeches",
    }
    missing = required - set(fingerprints.columns)
    if missing:
        raise ValueError(
            "fine_fingerprints.parquet is missing required columns "
            f"{sorted(missing)}"
        )
    fine = fingerprints[fingerprints["grain"] == "bin8"].copy()
    if fine.empty:
        raise ValueError("fine_fingerprints.parquet has no bin8 rows")
    if fine.duplicated(["unit", "measure"]).any():
        raise ValueError("bin8 fingerprints are not one-to-one on (unit, measure)")

    order = (
        fine[["measure", "axis_group"]]
        .drop_duplicates()
        .sort_values(["axis_group", "measure"])
    )
    if order["measure"].duplicated().any():
        raise ValueError("a fingerprint measure is assigned to multiple axis groups")
    axes = order["measure"].tolist()
    groups = dict(zip(order["measure"], order["axis_group"]))

    pivot = fine.pivot(index="unit", columns="measure", values="value_raw")
    pivot.index = pivot.index.astype(float).astype(int)
    pivot = pivot.sort_index().reindex(columns=axes)
    if pivot.isna().any().any():
        raise ValueError("bin8 fingerprint matrix is incomplete")
    return pivot, axes, groups


def build_cluster_stability(
    fingerprints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Boundary locations across k=2..12 and leave-one-group-out conditions."""
    if fingerprints is None:
        fingerprints = pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
    pivot, axes, groups = _bin_fingerprint_frame(fingerprints)
    bins = pivot.index.to_numpy(dtype=int)

    conditions: list[tuple[str, list[str]]] = [("full", axes)]
    conditions.extend(
        (f"drop_{group}", [axis for axis in axes if groups[axis] != group])
        for group in AXIS_GROUPS
    )
    conditions.append(
        ("drop_opponents", [axis for axis in axes if axis != eras.OPPONENTS_AXIS])
    )

    rows: list[dict] = []
    for condition, kept_axes in conditions:
        raw = pivot[kept_axes].to_numpy(dtype=float)
        sigma = raw.std(axis=0, ddof=0)
        keep = sigma > 0
        z = (raw[:, keep] - raw[:, keep].mean(axis=0)) / sigma[keep]
        for k in K_RANGE:
            for boundary in eras._boundaries_at_k(z, bins, k):
                rows.append(
                    {
                        "condition": condition,
                        "k": int(k),
                        "boundary_year": int(boundary),
                        "is_primary_k": bool(k == PRIMARY_K),
                        "n_axes": int(keep.sum()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["condition", "k", "boundary_year"]
    ).reset_index(drop=True)


def _window_axis(
    master: pd.DataFrame,
    start: int,
    end: int,
    axes: list[str],
    markers: pd.DataFrame,
    stats: pd.DataFrame,
    taxonomy: dict,
) -> pd.Series:
    """One raw fingerprint for an inclusive calendar-year window."""
    subset = master[master["year"].between(start, end)].copy()
    if subset.empty:
        raise ValueError(f"window {start}-{end} contains no paragraphs")
    # build_axis_frame already implements every denominator and doc-set guard.
    # Assign one temporary unit so windows may cross the existing named eras.
    subset["era"] = "window"
    frame = eras.build_axis_frame(
        subset,
        "era",
        markers=markers,
        stats=stats,
        taxonomy=taxonomy,
    )
    if len(frame) != 1:
        raise ValueError(f"window {start}-{end} did not produce exactly one row")
    row = frame.iloc[0].copy()
    # A speech type absent from a short window is a genuine zero share.  The
    # shared builder only emits genres present in its input, so restore every
    # governed axis before comparing two windows.
    for axis in axes:
        if axis not in row.index:
            row[axis] = 0.0
    return row


def sliding_transition_scores(
    master: pd.DataFrame | None = None,
    fingerprints: pd.DataFrame | None = None,
    markers: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
    window_years: int = WINDOW_YEARS,
) -> pd.DataFrame:
    """Score every calendar-year boundary without snapping it to a bin grid.

    At boundary ``b`` the left fingerprint covers ``b-window_years`` through
    ``b-1`` and the right covers ``b`` through ``b+window_years-1``.  Each
    right-minus-left axis difference is divided by that axis's population
    standard deviation across the governed eight-year-bin fingerprints.  The
    reported score is the Euclidean norm of the standardized differences.
    """
    if window_years < 2:
        raise ValueError("window_years must be at least 2")
    master = eras.load_master_frame() if master is None else master
    fingerprints = (
        pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
        if fingerprints is None else fingerprints
    )
    markers = (
        pd.read_parquet(eras.SPEECH_MARKERS_PATH) if markers is None else markers
    )
    stats = pd.read_parquet(eras.SPEECH_STATS_PATH) if stats is None else stats
    taxonomy = load_taxonomy() if taxonomy is None else taxonomy

    pivot, axes, groups = _bin_fingerprint_frame(fingerprints)
    sigma = pivot[axes].to_numpy(dtype=float).std(axis=0, ddof=0)
    keep = sigma > 0
    kept_axes = [axis for axis, kept in zip(axes, keep) if kept]
    scale = pd.Series(sigma[keep], index=kept_axes)

    first = int(master["year"].min()) + window_years
    last = int(master["year"].max()) - window_years + 1
    rows: list[dict] = []
    for boundary in range(first, last + 1):
        left = _window_axis(
            master,
            boundary - window_years,
            boundary - 1,
            kept_axes,
            markers,
            stats,
            taxonomy,
        )
        right = _window_axis(
            master,
            boundary,
            boundary + window_years - 1,
            kept_axes,
            markers,
            stats,
            taxonomy,
        )
        delta = (right[kept_axes].astype(float) - left[kept_axes].astype(float)) / scale
        row = {
            "boundary_year": int(boundary),
            "window_years": int(window_years),
            "left_start": int(boundary - window_years),
            "left_end": int(boundary - 1),
            "right_start": int(boundary),
            "right_end": int(boundary + window_years - 1),
            "score": float(np.linalg.norm(delta.to_numpy(dtype=float))),
            "n_axes": int(len(kept_axes)),
            "n_left_paragraphs": int(left["n_paragraphs"]),
            "n_right_paragraphs": int(right["n_paragraphs"]),
            "n_left_speeches": int(left["n_speeches"]),
            "n_right_speeches": int(right["n_speeches"]),
        }
        for group in AXIS_GROUPS:
            group_axes = [axis for axis in kept_axes if groups[axis] == group]
            row[f"{group}_score"] = float(
                np.linalg.norm(delta[group_axes].to_numpy(dtype=float))
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    out["rank"] = out["score"].rank(method="min", ascending=False).astype(int)
    return out.sort_values("boundary_year").reset_index(drop=True)


def consensus_transition_scores(sliding: pd.DataFrame) -> pd.DataFrame:
    """Combine four- and eight-year abruptness ranks on their common years.

    Scores have window-specific scales, so combining their raw magnitudes would
    silently privilege one window.  Each rank is converted to a 0..1 percentile
    within its complete candidate-year series and the two percentiles are
    averaged.  High remains "more abrupt", not "more historically desirable".
    """
    required = {"boundary_year", "window_years", "score", "rank"}
    missing = required - set(sliding.columns)
    if missing:
        raise ValueError(f"sliding scores are missing {sorted(missing)}")
    frames: dict[int, pd.DataFrame] = {}
    for window in WINDOW_OPTIONS:
        frame = sliding[sliding["window_years"] == window].copy()
        if frame.empty:
            raise ValueError(f"sliding scores contain no {window}-year windows")
        if frame["boundary_year"].duplicated().any():
            raise ValueError(f"{window}-year scores duplicate a boundary year")
        n = len(frame)
        frame[f"score_{window}"] = frame["score"].astype(float)
        frame[f"rank_{window}"] = frame["rank"].astype(int)
        frame[f"percentile_{window}"] = (
            1.0 - (frame["rank"].astype(float) - 1.0) / (n - 1.0)
        )
        frames[window] = frame[
            [
                "boundary_year",
                f"score_{window}",
                f"rank_{window}",
                f"percentile_{window}",
            ]
        ]
    out = frames[4].merge(frames[8], on="boundary_year", validate="one_to_one")
    out["consensus"] = (
        out["percentile_4"] + out["percentile_8"]
    ) / 2.0
    out["consensus_rank"] = (
        out["consensus"].rank(method="min", ascending=False).astype(int)
    )
    return out.sort_values("boundary_year").reset_index(drop=True)


def build_cycle_fingerprints(
    master: pd.DataFrame | None = None,
    fingerprints: pd.DataFrame | None = None,
    markers: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
    grid_offset: int = 0,
    anchor_year: int | None = None,
) -> pd.DataFrame:
    """Build four-year fingerprints on one declared calendar-grid offset.

    Offset zero is the primary 1789 presidential-cycle anchor. Offsets one
    through three are sensitivity conditions; they intentionally omit the
    leading years before their first complete grid origin instead of silently
    assigning those years to a differently-sized initial unit.
    """
    if grid_offset not in GRID_OFFSETS:
        raise ValueError(f"grid_offset must be one of {GRID_OFFSETS}")
    master = eras.load_master_frame() if master is None else master
    fingerprints = (
        pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
        if fingerprints is None else fingerprints
    )
    markers = (
        pd.read_parquet(eras.SPEECH_MARKERS_PATH) if markers is None else markers
    )
    stats = pd.read_parquet(eras.SPEECH_STATS_PATH) if stats is None else stats
    taxonomy = load_taxonomy() if taxonomy is None else taxonomy
    _, axes, _ = _bin_fingerprint_frame(fingerprints)

    if master.empty:
        raise ValueError("cannot build cycle fingerprints from an empty corpus")
    first = (
        int(master["year"].min()) if anchor_year is None else int(anchor_year)
    ) + grid_offset
    last = int(master["year"].max())
    if first > last:
        raise ValueError("cycle-grid anchor falls after the corpus endpoint")
    rows: list[dict] = []
    for start in range(first, last + 1, 4):
        end = min(start + 3, last)
        axis = _window_axis(
            master, start, end, axes, markers, stats, taxonomy
        )
        row = {
            "cycle_start": int(start),
            "cycle_end": int(end),
            "grid_offset": int(grid_offset),
            "is_complete_cycle": bool(end - start == 3),
            "n_paragraphs": int(axis["n_paragraphs"]),
            "n_speeches": int(axis["n_speeches"]),
        }
        row.update({measure: float(axis[measure]) for measure in axes})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cycle_start").reset_index(drop=True)


def _segmentation_costs(
    cycle: pd.DataFrame,
    axes: list[str],
    weighting: str = "per_axis",
) -> np.ndarray:
    if weighting not in WEIGHTING_OPTIONS:
        raise ValueError(f"weighting must be one of {WEIGHTING_OPTIONS}")
    if not axes:
        raise ValueError("segmentation requires at least one axis")
    missing = set(axes) - set(cycle.columns)
    if missing:
        raise ValueError(f"cycle fingerprints are missing axes {sorted(missing)}")
    raw = cycle[axes].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("cycle fingerprint axes must all be finite")
    sigma = raw.std(axis=0, ddof=0)
    keep = sigma > 0
    if not keep.any():
        raise ValueError("segmentation has no nonconstant axes")
    z = (raw[:, keep] - raw[:, keep].mean(axis=0)) / sigma[keep]
    if weighting == "per_axis_family":
        kept_axes = [axis for axis, kept in zip(axes, keep) if kept]
        z = z * eras.group_balance_weights(kept_axes)
    n = len(z)
    costs = np.full((n, n), np.inf)
    cumulative = np.vstack([np.zeros(z.shape[1]), np.cumsum(z, axis=0)])
    squared = np.r_[0.0, np.cumsum((z * z).sum(axis=1))]
    for start in range(n):
        for end in range(start, n):
            count = end - start + 1
            total = cumulative[end + 1] - cumulative[start]
            costs[start, end] = (
                squared[end + 1]
                - squared[start]
                - float(total @ total) / count
            )
    return costs


def _optimal_partition(
    costs: np.ndarray,
    k: int = PRIMARY_K,
    min_units: int = MIN_SEGMENT_UNITS,
) -> tuple[list[int], float]:
    """Dynamic-programming optimum as zero-based segment-start indices."""
    n = costs.shape[0]
    if costs.shape != (n, n):
        raise ValueError("segmentation costs must be square")
    if k < 2 or min_units < 1 or k * min_units > n:
        raise ValueError("infeasible segmentation request")
    infinity = float("inf")
    objective = np.full((k + 1, n + 1), infinity)
    previous = np.full((k + 1, n + 1), -1, dtype=int)
    objective[0, 0] = 0.0
    for segment in range(1, k + 1):
        for end in range(segment * min_units, n + 1):
            candidates = np.arange(
                (segment - 1) * min_units,
                end - min_units + 1,
            )
            values = objective[segment - 1, candidates] + np.array(
                [costs[start, end - 1] for start in candidates]
            )
            best = int(np.argmin(values))
            objective[segment, end] = float(values[best])
            previous[segment, end] = int(candidates[best])

    end = n
    starts: list[int] = []
    for segment in range(k, 0, -1):
        start = int(previous[segment, end])
        if start < 0:
            raise ValueError("segmentation backtrace failed")
        starts.append(start)
        end = start
    return starts[::-1], float(objective[k, n])


def build_segmentation(
    cycle: pd.DataFrame,
    fingerprints: pd.DataFrame | None = None,
    k: int = PRIMARY_K,
    min_units: int = MIN_SEGMENT_UNITS,
    weighting: str = "per_axis",
) -> pd.DataFrame:
    """Optimal cuts under the full and declared axis-removal conditions."""
    fingerprints = (
        pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
        if fingerprints is None else fingerprints
    )
    _, axes, groups = _bin_fingerprint_frame(fingerprints)
    conditions: list[tuple[str, list[str]]] = [("full", axes)]
    conditions.extend(
        (f"drop_{group}", [axis for axis in axes if groups[axis] != group])
        for group in AXIS_GROUPS
    )
    conditions.append(
        ("drop_opponents", [axis for axis in axes if axis != eras.OPPONENTS_AXIS])
    )
    years = cycle["cycle_start"].astype(int).tolist()
    rows: list[dict] = []
    for condition, kept_axes in conditions:
        costs = _segmentation_costs(cycle, kept_axes, weighting=weighting)
        starts, objective = _optimal_partition(
            costs, k=k, min_units=min_units
        )
        for segment_index, start_index in enumerate(starts):
            rows.append(
                {
                    "condition": condition,
                    "segment_index": int(segment_index),
                    "segment_start": int(years[start_index]),
                    "is_boundary": bool(segment_index > 0),
                    "k": int(k),
                    "min_units": int(min_units),
                    "weighting": weighting,
                    "objective": float(objective),
                    "n_axes": int(len(kept_axes)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["condition", "segment_index"]
    ).reset_index(drop=True)


def partition_objective(
    cycle: pd.DataFrame,
    starts: list[int] | tuple[int, ...],
    axes: list[str],
    weighting: str = "per_axis",
) -> float:
    """Within-segment objective for a declared complete partition."""
    years = cycle["cycle_start"].astype(int).tolist()
    if not starts or starts[0] != years[0]:
        raise ValueError("partition must begin at the first cycle")
    if any(start not in years for start in starts):
        raise ValueError("every partition start must be an observed cycle start")
    indices = [years.index(start) for start in starts] + [len(years)]
    if indices != sorted(indices):
        raise ValueError("partition starts must be chronological")
    costs = _segmentation_costs(cycle, axes, weighting=weighting)
    return float(
        sum(
            costs[left, right - 1]
            for left, right in zip(indices[:-1], indices[1:])
        )
    )


def sensitivity_specifications() -> list[dict]:
    """Decision-complete registry for every deterministic robustness rerun."""
    primary = {
        "condition": "primary",
        "label": "Primary: nine eras",
        "analysis_family": (
            "primary;k_sweep;minimum_duration;grid_anchor;weighting;terminal"
        ),
        "k": PRIMARY_K,
        "min_units": MIN_SEGMENT_UNITS,
        "grid_offset": 0,
        "weighting": "per_axis",
        "terminal_policy": "include_incomplete",
        "genre_filter": "all",
        "omitted_axis_group": "none",
    }
    specs = [primary]
    for k in K_RANGE:
        if k != PRIMARY_K:
            specs.append({
                **primary,
                "condition": f"k_{k:02d}",
                "label": f"{k} segments",
                "analysis_family": "k_sweep",
                "k": int(k),
            })
    specs.append({
        **primary,
        "condition": "minimum_4_cycles",
        "label": "Minimum four cycle units",
        "analysis_family": "minimum_duration",
        "min_units": 4,
    })
    for offset in GRID_OFFSETS[1:]:
        specs.append({
            **primary,
            "condition": f"grid_offset_{offset}",
            "label": f"Cycle grid +{offset} year{'s' if offset != 1 else ''}",
            "analysis_family": "grid_anchor",
            "grid_offset": int(offset),
        })
    specs.append({
        **primary,
        "condition": "equal_axis_family",
        "label": "Equal axis-family weight",
        "analysis_family": "weighting",
        "weighting": "per_axis_family",
    })
    specs.append({
        **primary,
        "condition": "complete_cycles_only",
        "label": "Exclude incomplete 2025–26 cycle",
        "analysis_family": "terminal",
        "terminal_policy": "complete_only",
    })
    for group in AXIS_GROUPS:
        specs.append({
            **primary,
            "condition": f"drop_{group}",
            "label": f"Without {group} axes",
            "analysis_family": "axis_omission",
            "omitted_axis_group": group,
        })
    specs.append({
        **primary,
        "condition": "drop_opponents",
        "label": "Without opponents marker",
        "analysis_family": "axis_omission",
        "omitted_axis_group": "opponents_marker",
    })
    specs.append({
        **primary,
        "condition": "sotu_annual_only",
        "label": "Annual messages / State of the Union only",
        "analysis_family": "common_genre",
        "genre_filter": SOTU_SPEECH_TYPE,
    })
    conditions = [spec["condition"] for spec in specs]
    if len(conditions) != len(set(conditions)):
        raise ValueError("sensitivity condition names must be unique")
    return specs


def _partition_labels(
    starts: list[int], first_year: int, last_year: int
) -> np.ndarray:
    """Convert chronological starts to annual segment labels for comparison."""
    if first_year > last_year:
        raise ValueError("partition comparison has no overlapping years")
    ordered = np.asarray(starts, dtype=int)
    if len(ordered) == 0 or not np.all(ordered[:-1] < ordered[1:]):
        raise ValueError("partition starts must be strictly chronological")
    years = np.arange(first_year, last_year + 1, dtype=int)
    return np.searchsorted(ordered, years, side="right") - 1


def _partition_similarity(
    reference_starts: list[int],
    candidate_starts: list[int],
    reference_end: int,
    candidate_end: int,
) -> float:
    """Adjusted Rand similarity over the partitions' shared calendar years."""
    first = max(reference_starts[0], candidate_starts[0])
    last = min(reference_end, candidate_end)
    reference = _partition_labels(reference_starts, first, last)
    candidate = _partition_labels(candidate_starts, first, last)
    return float(adjusted_rand_score(reference, candidate))


def build_sensitivity_analysis(
    master: pd.DataFrame | None = None,
    fingerprints: pd.DataFrame | None = None,
    markers: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
) -> pd.DataFrame:
    """Run every declared segmentation sensitivity and retain segment support.

    The output is one row per segment, not merely one row per boundary. That
    makes the population behind every proposed era auditable and allows the
    final incomplete cycle to carry an explicit right-censor flag.
    """
    master = eras.load_master_frame() if master is None else master
    fingerprints = (
        pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
        if fingerprints is None else fingerprints
    )
    markers = (
        pd.read_parquet(eras.SPEECH_MARKERS_PATH) if markers is None else markers
    )
    stats = pd.read_parquet(eras.SPEECH_STATS_PATH) if stats is None else stats
    taxonomy = load_taxonomy() if taxonomy is None else taxonomy
    _, axes, groups = _bin_fingerprint_frame(fingerprints)
    corpus_start = int(master["year"].min())

    cycle_cache: dict[tuple[str, int], pd.DataFrame] = {}

    def cycles(genre_filter: str, offset: int) -> pd.DataFrame:
        key = (genre_filter, offset)
        if key not in cycle_cache:
            subset = master
            if genre_filter != "all":
                subset = master[master["speech_type"] == genre_filter].copy()
                if subset.empty:
                    raise ValueError(
                        f"sensitivity genre {genre_filter!r} has no paragraphs"
                    )
            cycle_cache[key] = build_cycle_fingerprints(
                subset,
                fingerprints=fingerprints,
                markers=markers,
                stats=stats,
                taxonomy=taxonomy,
                grid_offset=offset,
                anchor_year=corpus_start,
            )
        return cycle_cache[key].copy()

    primary_cycle = cycles("all", 0)
    primary_costs = _segmentation_costs(primary_cycle, axes)
    primary_indices, _ = _optimal_partition(
        primary_costs, PRIMARY_K, MIN_SEGMENT_UNITS
    )
    primary_starts = [
        int(primary_cycle.iloc[index]["cycle_start"])
        for index in primary_indices
    ]
    primary_end = int(primary_cycle.iloc[-1]["cycle_end"])

    output: list[dict] = []
    for spec in sensitivity_specifications():
        try:
            cycle = cycles(spec["genre_filter"], spec["grid_offset"])
        except ValueError as exc:
            if spec["genre_filter"] == "all":
                raise
            output.append({
                **spec,
                "status": "not_estimable",
                "status_reason": str(exc),
                "segment_index": -1,
                "segment_start": pd.NA,
                "segment_end": pd.NA,
                "is_boundary": False,
                "objective": float("nan"),
                "objective_scale": "not_estimable",
                "partition_similarity_to_primary": float("nan"),
                "partition_starts_json": "[]",
                "n_axes": 0,
                "n_cycles": 0,
                "n_paragraphs": 0,
                "n_speeches": 0,
                "terminal_cycle_complete": False,
                "right_censored": False,
            })
            continue
        if spec["terminal_policy"] == "complete_only":
            cycle = cycle[cycle["is_complete_cycle"]].reset_index(drop=True)
        if cycle.empty:
            raise ValueError(f"{spec['condition']}: no cycles remain")

        kept_axes = list(axes)
        omitted = spec["omitted_axis_group"]
        if omitted in AXIS_GROUPS:
            kept_axes = [axis for axis in axes if groups[axis] != omitted]
        elif omitted == "opponents_marker":
            kept_axes = [axis for axis in axes if axis != eras.OPPONENTS_AXIS]

        costs = _segmentation_costs(
            cycle, kept_axes, weighting=spec["weighting"]
        )
        start_indices, objective = _optimal_partition(
            costs, k=spec["k"], min_units=spec["min_units"]
        )
        starts = [
            int(cycle.iloc[index]["cycle_start"])
            for index in start_indices
        ]
        similarity = _partition_similarity(
            primary_starts,
            starts,
            primary_end,
            int(cycle.iloc[-1]["cycle_end"]),
        )
        raw = cycle[kept_axes].to_numpy(dtype=float)
        n_nonconstant_axes = int((raw.std(axis=0, ddof=0) > 0).sum())
        starts_json = json.dumps(starts, separators=(",", ":"))
        terminal_complete = bool(cycle.iloc[-1]["is_complete_cycle"])
        for segment_index, start_index in enumerate(start_indices):
            stop_index = (
                start_indices[segment_index + 1]
                if segment_index + 1 < len(start_indices)
                else len(cycle)
            )
            segment = cycle.iloc[start_index:stop_index]
            output.append({
                **spec,
                "status": "computed",
                "status_reason": "",
                "segment_index": int(segment_index),
                "segment_start": int(segment.iloc[0]["cycle_start"]),
                "segment_end": int(segment.iloc[-1]["cycle_end"]),
                "is_boundary": bool(segment_index > 0),
                "objective": float(objective),
                "objective_scale": (
                    "z_sse_per_axis_family"
                    if spec["weighting"] == "per_axis_family"
                    else "z_sse_per_axis"
                ),
                "partition_similarity_to_primary": similarity,
                "partition_starts_json": starts_json,
                "n_axes": n_nonconstant_axes,
                "n_cycles": int(len(segment)),
                "n_paragraphs": int(segment["n_paragraphs"].sum()),
                "n_speeches": int(segment["n_speeches"].sum()),
                "terminal_cycle_complete": terminal_complete,
                "right_censored": bool(
                    segment_index == len(start_indices) - 1
                    and not terminal_complete
                ),
            })
    result = pd.DataFrame(output)
    if result.duplicated(["condition", "segment_index"]).any():
        raise ValueError("sensitivity output duplicates a condition segment")
    computed = result["status"] == "computed"
    if not np.isfinite(result.loc[computed, "objective"]).all():
        raise ValueError("sensitivity output contains a nonfinite objective")
    return result.reset_index(drop=True)


def build_boundary_drivers(
    cycle: pd.DataFrame,
    fingerprints: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
    markers: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top signed raw-axis changes around each reviewed internal boundary.

    Equal-cycle estimates match the segmentation estimand. Paragraph-weighted
    estimates are retained alongside exact pooled Story-era estimates so the
    public page never silently substitutes one denominator for the other.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    fingerprints = (
        pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
        if fingerprints is None else fingerprints
    )
    master = eras.load_master_frame() if master is None else master
    markers = (
        pd.read_parquet(eras.SPEECH_MARKERS_PATH) if markers is None else markers
    )
    stats = pd.read_parquet(eras.SPEECH_STATS_PATH) if stats is None else stats
    taxonomy = load_taxonomy() if taxonomy is None else taxonomy
    _, axes, groups = _bin_fingerprint_frame(fingerprints)
    raw = cycle[axes].to_numpy(dtype=float)
    sigma = raw.std(axis=0, ddof=0)
    scale = pd.Series(sigma, index=axes)
    keyed_master = master.copy()
    keyed_master["_story_era_key"] = era_profiles.story_era_key_series(
        keyed_master["year"], keyed_master["president"]
    )

    def pooled_story_axis(spec: era_profiles.EraProfileSpec) -> pd.Series:
        subset = keyed_master[keyed_master["_story_era_key"] == spec.key].copy()
        if subset.empty:
            raise ValueError(f"Story era {spec.key!r} has no paragraphs")
        subset["era"] = "story"
        frame = eras.build_axis_frame(
            subset,
            "era",
            markers=markers,
            stats=stats,
            taxonomy=taxonomy,
        )
        if len(frame) != 1:
            raise ValueError(f"Story era {spec.key!r} did not produce one axis row")
        row = frame.iloc[0].copy()
        for axis in axes:
            if axis not in row.index:
                row[axis] = 0.0
        return row

    story_axes = {
        spec.key: pooled_story_axis(spec)
        for spec in era_profiles.ERA_PROFILE_SPECS
    }
    rows: list[dict] = []
    for boundary_index, grid_start in enumerate(REVIEWED_GRID_STARTS[1:], start=1):
        left_start = REVIEWED_GRID_STARTS[boundary_index - 1]
        right_end = (
            REVIEWED_GRID_STARTS[boundary_index + 1] - 1
            if boundary_index + 1 < len(REVIEWED_GRID_STARTS)
            else int(cycle.iloc[-1]["cycle_end"])
        )
        left = cycle[
            (cycle["cycle_start"] >= left_start)
            & (cycle["cycle_start"] < grid_start)
        ]
        right = cycle[
            (cycle["cycle_start"] >= grid_start)
            & (cycle["cycle_start"] <= right_end)
        ]
        if left.empty or right.empty:
            raise ValueError(f"boundary {grid_start} has an empty adjacent regime")
        left_equal = left[axes].mean(axis=0)
        right_equal = right[axes].mean(axis=0)
        standardized = (right_equal - left_equal).divide(scale.replace(0, np.nan))
        ranked = standardized.abs().sort_values(ascending=False).head(top_n).index
        editorial_start = next(
            spec.start_year for spec in era_profiles.ERA_PROFILE_SPECS
            if STORY_TO_GRID_START.get(spec.start_year, spec.start_year) == grid_start
        )
        spec = next(
            item for item in era_profiles.ERA_PROFILE_SPECS
            if item.start_year == editorial_start
        )
        previous_spec = era_profiles.ERA_PROFILE_SPECS[
            list(era_profiles.ERA_PROFILE_SPECS).index(spec) - 1
        ]
        left_pooled = story_axes[previous_spec.key]
        right_pooled = story_axes[spec.key]
        for rank, axis in enumerate(ranked, start=1):
            rows.append({
                "era_key": spec.key,
                "story_section_key": spec.section_key,
                "editorial_start": int(editorial_start),
                "grid_start": int(grid_start),
                "driver_rank": int(rank),
                "axis": axis,
                "axis_group": groups[axis],
                "z_delta_equal_cycle": float(standardized[axis]),
                "direction": (
                    "increase" if standardized[axis] >= 0 else "decrease"
                ),
                "left_equal_cycle_value": float(left_equal[axis]),
                "right_equal_cycle_value": float(right_equal[axis]),
                "left_story_pooled_value": float(left_pooled[axis]),
                "right_story_pooled_value": float(right_pooled[axis]),
                "n_left_paragraphs": int(left_pooled["n_paragraphs"]),
                "n_right_paragraphs": int(right_pooled["n_paragraphs"]),
                "n_left_speeches": int(left_pooled["n_speeches"]),
                "n_right_speeches": int(right_pooled["n_speeches"]),
            })
    return pd.DataFrame(rows)


def build_boundary_evidence(
    consensus: pd.DataFrame,
    sensitivity: pd.DataFrame,
    drivers: pd.DataFrame,
    reviewed_adjustments: list[dict],
) -> pd.DataFrame:
    """One governed, artifact-derived evidence record per editorial boundary."""
    primary = sensitivity[
        (sensitivity["condition"] == "primary") & sensitivity["is_boundary"]
    ]
    statuses = sensitivity[["condition", "status"]].drop_duplicates()
    if statuses["condition"].duplicated().any():
        raise ValueError("each sensitivity condition must have exactly one status")
    n_conditions = int(len(statuses))
    n_estimable_conditions = int(statuses["status"].eq("computed").sum())
    n_unavailable_conditions = n_conditions - n_estimable_conditions
    adjustments = {
        int(row["reviewed_start"]): row for row in reviewed_adjustments
    }
    rows: list[dict] = []
    for spec in era_profiles.ERA_PROFILE_SPECS[1:]:
        start = int(spec.start_year)
        grid_start = STORY_TO_GRID_START.get(start, start)
        exact = consensus[consensus["boundary_year"] == start]
        if len(exact) != 1:
            raise ValueError(f"consensus artifact has no unique score for {start}")
        exact_row = exact.iloc[0]
        local = consensus[consensus["boundary_year"].between(start - 4, start + 4)]
        peak = local.loc[local["consensus"].idxmax()]
        primary_start = int(
            primary.loc[
                (primary["segment_start"] - grid_start).abs().idxmin(),
                "segment_start",
            ]
        )
        nearest_by_condition: list[int] = []
        for _, condition_rows in sensitivity[
            sensitivity["is_boundary"]
        ].groupby("condition", sort=False):
            nearest = condition_rows.loc[
                (condition_rows["segment_start"] - grid_start).abs().idxmin(),
                "segment_start",
            ]
            if abs(int(nearest) - grid_start) <= 4:
                nearest_by_condition.append(int(nearest))
        support = drivers[drivers["editorial_start"] == start].iloc[0]
        adjustment = adjustments.get(start)
        rows.append({
            "era_key": spec.key,
            "story_section_key": spec.section_key,
            "era_title": spec.title,
            "editorial_start": start,
            "editorial_end": int(spec.end_year),
            "grid_start": int(grid_start),
            "data_optimal_start": primary_start,
            "n_annual_candidates": int(len(consensus)),
            "annual_abruptness_rank": int(exact_row["consensus_rank"]),
            "annual_abruptness_percentile": float(exact_row["consensus"]),
            "local_peak_year": int(peak["boundary_year"]),
            "local_peak_percentile": float(peak["consensus"]),
            "n_sensitivity_conditions": n_conditions,
            "n_estimable_conditions": n_estimable_conditions,
            "n_unavailable_conditions": n_unavailable_conditions,
            "n_conditions_within_four_years": int(len(nearest_by_condition)),
            "sensitivity_start_min": (
                min(nearest_by_condition) if nearest_by_condition else None
            ),
            "sensitivity_start_max": (
                max(nearest_by_condition) if nearest_by_condition else None
            ),
            "objective_penalty": (
                float(adjustment["objective_penalty"]) if adjustment else 0.0
            ),
            "penalty_pct_of_optimum": (
                float(adjustment["penalty_pct_of_optimum"])
                if adjustment else 0.0
            ),
            "n_left_paragraphs": int(support["n_left_paragraphs"]),
            "n_right_paragraphs": int(support["n_right_paragraphs"]),
            "n_left_speeches": int(support["n_left_speeches"]),
            "n_right_speeches": int(support["n_right_speeches"]),
            "right_censored": bool(spec is era_profiles.ERA_PROFILE_SPECS[-1]),
        })
    return pd.DataFrame(rows)


def _source_fingerprints() -> dict[str, str]:
    return {
        "era_boundaries.py": _sha256(Path(__file__)),
        "era_profiles.py": _sha256(Path(era_profiles.__file__)),
        eras.FINE_FINGERPRINTS_PATH.name: _sha256(eras.FINE_FINGERPRINTS_PATH),
        eras.PERIODIZATION_PATH.name: _sha256(eras.PERIODIZATION_PATH),
        eras.ERAS_META_PATH.name: _sha256(eras.ERAS_META_PATH),
        eras.SPEECH_MARKERS_PATH.name: _sha256(eras.SPEECH_MARKERS_PATH),
        eras.SPEECH_STATS_PATH.name: _sha256(eras.SPEECH_STATS_PATH),
    }


def _publish_bundle(candidate: Path, destination: Path) -> None:
    """Replace one governed directory as a bundle, restoring it on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="era-boundaries-rollback-", dir=destination.parent
    ) as backup_name:
        held = Path(backup_name) / "previous"
        existed = destination.exists()
        if existed:
            os.replace(destination, held)
        try:
            os.replace(candidate, destination)
        except Exception:
            if destination.exists():
                shutil.rmtree(destination)
            if held.exists():
                os.replace(held, destination)
            raise


def run(out_dir: Path | None = None) -> dict:
    """Build the deterministic public analysis layer."""
    out_dir = ANALYSIS_DIR if out_dir is None else out_dir
    fingerprints = pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
    master = eras.load_master_frame()
    markers = pd.read_parquet(eras.SPEECH_MARKERS_PATH)
    stats = pd.read_parquet(eras.SPEECH_STATS_PATH)
    taxonomy = load_taxonomy()
    cluster = build_cluster_stability(fingerprints)
    sliding = pd.concat(
        [
            sliding_transition_scores(
                master=master,
                fingerprints=fingerprints,
                markers=markers,
                stats=stats,
                taxonomy=taxonomy,
                window_years=window,
            )
            for window in WINDOW_OPTIONS
        ],
        ignore_index=True,
    ).sort_values(["window_years", "boundary_year"]).reset_index(drop=True)
    consensus = consensus_transition_scores(sliding)
    cycle = build_cycle_fingerprints(
        master=master,
        fingerprints=fingerprints,
        markers=markers,
        stats=stats,
        taxonomy=taxonomy,
        anchor_year=int(master["year"].min()),
    )
    segmentation = build_segmentation(cycle, fingerprints)
    sensitivity = build_sensitivity_analysis(
        master=master,
        fingerprints=fingerprints,
        markers=markers,
        stats=stats,
        taxonomy=taxonomy,
    )
    drivers = build_boundary_drivers(
        cycle,
        fingerprints,
        master=master,
        markers=markers,
        stats=stats,
        taxonomy=taxonomy,
    )
    _, axes, _ = _bin_fingerprint_frame(fingerprints)
    full_starts = (
        segmentation[
            segmentation["condition"] == "full"
        ]["segment_start"].astype(int).tolist()
    )
    full_objective = float(
        segmentation.loc[
            segmentation["condition"] == "full", "objective"
        ].iloc[0]
    )
    single_indices, single_objective = _optimal_partition(
        _segmentation_costs(cycle, axes),
        k=2,
        min_units=MIN_SEGMENT_UNITS,
    )
    single_split_start = int(
        cycle.iloc[single_indices[1]]["cycle_start"]
    )
    reviewed_adjustments = []
    for era_key, data_start in REVIEWED_MODEL_ADJUSTMENTS.items():
        reviewed_start = era_profiles.resolve_era(era_key).start_year
        adjusted = list(full_starts)
        adjusted[adjusted.index(data_start)] = reviewed_start
        adjusted.sort()
        adjusted_objective = partition_objective(cycle, adjusted, axes)
        penalty = adjusted_objective - full_objective
        reviewed_adjustments.append(
            {
                "data_start": data_start,
                "reviewed_start": reviewed_start,
                "objective_penalty": penalty,
                "penalty_pct_of_optimum": penalty / full_objective * 100.0,
            }
        )
    reviewed_objective = partition_objective(
        cycle, REVIEWED_GRID_STARTS, axes
    )
    evidence = build_boundary_evidence(
        consensus, sensitivity, drivers, reviewed_adjustments
    )
    atlas_meta = json.loads(eras.ERAS_META_PATH.read_text())
    sensitivity_statuses = sensitivity[["condition", "status"]].drop_duplicates()
    if sensitivity_statuses["condition"].duplicated().any():
        raise ValueError("each sensitivity condition must have exactly one status")
    n_estimable_conditions = int(
        sensitivity_statuses["status"].eq("computed").sum()
    )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "current Story scheme; descriptive corpus periodization with reviewed "
            "exact dates and explicit right-censoring"
        ),
        "window_years_each_side": WINDOW_YEARS,
        "window_options_each_side": list(WINDOW_OPTIONS),
        "primary_cluster_count": PRIMARY_K,
        "cluster_count_range": [min(K_RANGE), max(K_RANGE)],
        "minimum_segment_units": MIN_SEGMENT_UNITS,
        "minimum_segment_options": list(MIN_SEGMENT_OPTIONS),
        "grid_offsets": list(GRID_OFFSETS),
        "weighting_options": list(WEIGHTING_OPTIONS),
        "cycle_anchor_year": int(cycle["cycle_start"].min()),
        "data_optimal_grid_starts": full_starts,
        "single_split_start": single_split_start,
        "single_split_objective": single_objective,
        "reviewed_grid_starts": list(REVIEWED_GRID_STARTS),
        "reviewed_adjustments": reviewed_adjustments,
        "optimal_partition_objective": full_objective,
        "reviewed_partition_objective": reviewed_objective,
        "reviewed_excess_objective_pct": (
            (reviewed_objective / full_objective - 1.0) * 100.0
        ),
        "axis_groups": list(AXIS_GROUPS),
        "sensitivity_conditions": sensitivity_specifications(),
        "n_sensitivity_conditions": int(sensitivity["condition"].nunique()),
        "n_estimable_sensitivity_conditions": n_estimable_conditions,
        "n_unavailable_sensitivity_conditions": int(
            len(sensitivity_statuses) - n_estimable_conditions
        ),
        "n_axes": int(sliding["n_axes"].max()),
        "sliding_method": (
            "For every calendar year b, compare symmetric four-year and "
            "eight-year windows on the same raw fingerprint; standardize axis "
            "differences by population SD across the governed eight-year bins; "
            "each score is the Euclidean norm."
        ),
        "consensus_method": (
            "Convert the complete four- and eight-year score rankings to "
            "percentiles on their common candidate years and average them."
        ),
        "segmentation_method": (
            "Build non-overlapping four-year presidential-cycle fingerprints "
            "anchored at 1789; z-score each axis across cycles; use dynamic "
            "programming to minimize total within-segment squared distance for "
            "exactly nine contiguous segments of at least three cycle units; "
            "repeat while omitting each axis group and the opponents marker."
        ),
        "sensitivity_method": (
            "Repeat the dynamic-programming partition over k=2..12, minimum "
            "durations of three and four four-year-grid units, all four grid "
            "offsets, equal "
            "per-axis and equal per-axis-family weighting, inclusion and exclusion "
            "of the incomplete terminal cycle, each axis-family omission, the "
            "opponents-marker omission, and an annual-message/State-of-the-Union "
            "common-genre subset. Partition similarity is adjusted Rand index over "
            "shared calendar years."
        ),
        "driver_method": (
            "Rank raw-axis right-minus-left changes after standardizing by the "
            "population SD across primary four-year cycles. Preserve both the "
            "algorithm's equal-cycle mean and an exact pooled mean over governed "
            "Story-era membership, including outgoing-president overrides."
        ),
        "cluster_method": (
            "Contiguity-constrained Ward clustering on raw z-scored bin "
            "fingerprints, repeated at k=2..12 in full and leave-one-group-out forms."
        ),
        "corpus_fingerprint": atlas_meta["corpus_fingerprint"],
        "source_sha256": _source_fingerprints(),
        "story_eras": [
            {
                "key": spec.key,
                "label": spec.label,
                "start_year": spec.start_year,
                "end_year": spec.end_year,
                "section_key": spec.section_key,
            }
            for spec in era_profiles.ERA_PROFILE_SPECS
        ],
        "right_censoring": {
            "final_era_key": era_profiles.ERA_PROFILE_SPECS[-1].key,
            "terminal_cycle": (
                f"{int(cycle.iloc[-1]['cycle_start'])}–"
                f"{int(cycle.iloc[-1]['cycle_end'])}"
            ),
            "terminal_cycle_complete": bool(
                cycle.iloc[-1]["is_complete_cycle"]
            ),
            "interpretation": (
                "The final era and its 2017 boundary are provisional because the "
                "last four-year unit is incomplete at the corpus endpoint."
            ),
        },
        "api_calls": 0,
    }

    result = {
        "sliding_scores": sliding,
        "cluster_stability": cluster,
        "consensus_scores": consensus,
        "cycle_fingerprints": cycle,
        "segmentation": segmentation,
        "sensitivity": sensitivity,
        "boundary_evidence": evidence,
        "boundary_drivers": drivers,
        "meta": meta,
    }
    artifact_names = {
        "sliding_scores": SLIDING_SCORES_PATH.name,
        "cluster_stability": CLUSTER_STABILITY_PATH.name,
        "consensus_scores": CONSENSUS_SCORES_PATH.name,
        "cycle_fingerprints": CYCLE_FINGERPRINTS_PATH.name,
        "segmentation": SEGMENTATION_PATH.name,
        "sensitivity": SENSITIVITY_PATH.name,
        "boundary_evidence": BOUNDARY_EVIDENCE_PATH.name,
        "boundary_drivers": BOUNDARY_DRIVERS_PATH.name,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="era-boundaries-v3-", dir=out_dir.parent
    ) as temporary_name:
        staged = Path(temporary_name) / "bundle"
        staged.mkdir()
        for key, filename in artifact_names.items():
            result[key].to_parquet(staged / filename, index=False)
        meta["artifact_rows"] = {
            filename: int(len(result[key]))
            for key, filename in artifact_names.items()
        }
        meta["artifact_sha256"] = {
            filename: _sha256(staged / filename)
            for filename in artifact_names.values()
        }
        (staged / META_PATH.name).write_text(json.dumps(meta, indent=2) + "\n")
        # Publish all tables and their receipt together. If the second rename
        # fails, _publish_bundle restores the complete prior directory.
        _publish_bundle(staged, out_dir)
    return result


def load() -> dict:
    """Load the derived layer and refuse stale source fingerprints."""
    artifacts = {
        "sliding_scores": SLIDING_SCORES_PATH,
        "cluster_stability": CLUSTER_STABILITY_PATH,
        "consensus_scores": CONSENSUS_SCORES_PATH,
        "cycle_fingerprints": CYCLE_FINGERPRINTS_PATH,
        "segmentation": SEGMENTATION_PATH,
        "sensitivity": SENSITIVITY_PATH,
        "boundary_evidence": BOUNDARY_EVIDENCE_PATH,
        "boundary_drivers": BOUNDARY_DRIVERS_PATH,
    }
    if not all(path.exists() for path in (*artifacts.values(), META_PATH)):
        raise FileNotFoundError(
            "era-boundary artifacts are missing; run "
            "`python -m presidential_profiles.era_boundaries`"
        )
    meta = json.loads(META_PATH.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected {SCHEMA_VERSION}, found {meta.get('schema_version')!r}; "
            "regenerate the era-boundary artifacts"
        )
    expected = meta.get("source_sha256", {})
    live = _source_fingerprints()
    if expected != live:
        raise ValueError(
            "era-boundary artifacts are stale against their governed Era Atlas "
            "inputs; regenerate them before building the site"
        )
    expected_artifacts = meta.get("artifact_sha256", {})
    live_artifacts = {
        path.name: _sha256(path) for path in artifacts.values()
    }
    if expected_artifacts != live_artifacts:
        raise ValueError(
            "era-boundary artifact hashes do not match the v3 commit receipt; "
            "regenerate before building the site"
        )
    result = {key: pd.read_parquet(path) for key, path in artifacts.items()}
    for key, frame in result.items():
        expected_rows = meta.get("artifact_rows", {}).get(artifacts[key].name)
        if expected_rows != len(frame):
            raise ValueError(f"{artifacts[key].name} row count contradicts meta.json")
    result["meta"] = meta
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="alternate output directory (tests and review only)",
    )
    args = parser.parse_args()
    result = run(args.out_dir)
    print(
        "wrote era-boundary analysis: "
        f"{len(result['sliding_scores'])} windowed yearly scores, "
        f"{len(result['consensus_scores'])} consensus years, "
        f"{len(result['segmentation'])} segmentation rows, "
        f"{len(result['sensitivity'])} sensitivity rows"
    )


if __name__ == "__main__":
    main()
