"""Deterministic evidence for choosing editorial era boundaries.

The first Era Atlas periodization used globally aligned eight-year bins.  This
module retains that audit trail, but makes the primary question match the
editorial task more closely:

* compare every possible year with both four- and eight-year symmetric windows;
* combine their percentile ranks so one arbitrary window length cannot decide;
* divide presidential-cycle four-year units into exactly nine contiguous eras,
  subject to a three-unit minimum, by minimizing within-era squared distance;
* repeat that complete segmentation while omitting every axis group in turn.

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
from pathlib import Path

import numpy as np
import pandas as pd

from . import eras
from .attention import load_taxonomy
from .corpus import DATA_DIR

ANALYSIS_DIR = DATA_DIR / "era_boundaries"
SLIDING_SCORES_PATH = ANALYSIS_DIR / "sliding_scores.parquet"
CLUSTER_STABILITY_PATH = ANALYSIS_DIR / "cluster_stability.parquet"
CONSENSUS_SCORES_PATH = ANALYSIS_DIR / "consensus_scores.parquet"
CYCLE_FINGERPRINTS_PATH = ANALYSIS_DIR / "cycle4_fingerprints.parquet"
SEGMENTATION_PATH = ANALYSIS_DIR / "segmentation.parquet"
META_PATH = ANALYSIS_DIR / "meta.json"

SCHEMA_VERSION = "era-boundaries-v2"
WINDOW_YEARS = 8
WINDOW_OPTIONS = (4, 8)
PRIMARY_K = 9
K_RANGE = range(2, 13)
AXIS_GROUPS = ("topic", "combat", "register", "marker", "stat", "genre")
MIN_SEGMENT_UNITS = 3
REVIEWED_GRID_STARTS = (1789, 1809, 1849, 1869, 1913, 1933, 1953, 1981, 2017)


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
) -> pd.DataFrame:
    """Build non-overlapping presidential-cycle fingerprints anchored at 1789."""
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

    first = int(master["year"].min())
    last = int(master["year"].max())
    rows: list[dict] = []
    for start in range(first, last + 1, 4):
        end = min(start + 3, last)
        axis = _window_axis(
            master, start, end, axes, markers, stats, taxonomy
        )
        row = {
            "cycle_start": int(start),
            "cycle_end": int(end),
            "n_paragraphs": int(axis["n_paragraphs"]),
            "n_speeches": int(axis["n_speeches"]),
        }
        row.update({measure: float(axis[measure]) for measure in axes})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cycle_start").reset_index(drop=True)


def _segmentation_costs(
    cycle: pd.DataFrame,
    axes: list[str],
) -> np.ndarray:
    raw = cycle[axes].to_numpy(dtype=float)
    sigma = raw.std(axis=0, ddof=0)
    keep = sigma > 0
    z = (raw[:, keep] - raw[:, keep].mean(axis=0)) / sigma[keep]
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
) -> pd.DataFrame:
    """Exactly-nine-era optimal cuts under full and axis-removal conditions."""
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
        costs = _segmentation_costs(cycle, kept_axes)
        starts, objective = _optimal_partition(costs)
        for segment_index, start_index in enumerate(starts):
            rows.append(
                {
                    "condition": condition,
                    "segment_index": int(segment_index),
                    "segment_start": int(years[start_index]),
                    "is_boundary": bool(segment_index > 0),
                    "k": PRIMARY_K,
                    "min_units": MIN_SEGMENT_UNITS,
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
    costs = _segmentation_costs(cycle, axes)
    return float(
        sum(
            costs[left, right - 1]
            for left, right in zip(indices[:-1], indices[1:])
        )
    )


def _source_fingerprints() -> dict[str, str]:
    return {
        eras.FINE_FINGERPRINTS_PATH.name: _sha256(eras.FINE_FINGERPRINTS_PATH),
        eras.PERIODIZATION_PATH.name: _sha256(eras.PERIODIZATION_PATH),
        eras.ERAS_META_PATH.name: _sha256(eras.ERAS_META_PATH),
    }


def run(out_dir: Path | None = None) -> dict:
    """Build the deterministic public analysis layer."""
    out_dir = ANALYSIS_DIR if out_dir is None else out_dir
    fingerprints = pd.read_parquet(eras.FINE_FINGERPRINTS_PATH)
    cluster = build_cluster_stability(fingerprints)
    sliding = pd.concat(
        [
            sliding_transition_scores(
                fingerprints=fingerprints,
                window_years=window,
            )
            for window in WINDOW_OPTIONS
        ],
        ignore_index=True,
    ).sort_values(["window_years", "boundary_year"]).reset_index(drop=True)
    consensus = consensus_transition_scores(sliding)
    cycle = build_cycle_fingerprints(fingerprints=fingerprints)
    segmentation = build_segmentation(cycle, fingerprints)
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
    for data_start, reviewed_start in ((1805, 1809), (1937, 1933)):
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
    atlas_meta = json.loads(eras.ERAS_META_PATH.read_text())

    meta = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "descriptive corpus periodization; reviewed exact dates may snap a "
            "stable transition zone to a presidential or institutional boundary"
        ),
        "window_years_each_side": WINDOW_YEARS,
        "window_options_each_side": list(WINDOW_OPTIONS),
        "primary_cluster_count": PRIMARY_K,
        "cluster_count_range": [min(K_RANGE), max(K_RANGE)],
        "minimum_segment_units": MIN_SEGMENT_UNITS,
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
        "cluster_method": (
            "Contiguity-constrained Ward clustering on raw z-scored bin "
            "fingerprints, repeated at k=2..12 in full and leave-one-group-out forms."
        ),
        "corpus_fingerprint": atlas_meta["corpus_fingerprint"],
        "source_sha256": _source_fingerprints(),
        "api_calls": 0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    sliding.to_parquet(out_dir / SLIDING_SCORES_PATH.name, index=False)
    cluster.to_parquet(out_dir / CLUSTER_STABILITY_PATH.name, index=False)
    consensus.to_parquet(out_dir / CONSENSUS_SCORES_PATH.name, index=False)
    cycle.to_parquet(out_dir / CYCLE_FINGERPRINTS_PATH.name, index=False)
    segmentation.to_parquet(out_dir / SEGMENTATION_PATH.name, index=False)
    (out_dir / META_PATH.name).write_text(json.dumps(meta, indent=2) + "\n")
    return {
        "sliding_scores": sliding,
        "cluster_stability": cluster,
        "consensus_scores": consensus,
        "cycle_fingerprints": cycle,
        "segmentation": segmentation,
        "meta": meta,
    }


def load() -> dict:
    """Load the derived layer and refuse stale source fingerprints."""
    if not all(path.exists() for path in (
        SLIDING_SCORES_PATH,
        CLUSTER_STABILITY_PATH,
        CONSENSUS_SCORES_PATH,
        CYCLE_FINGERPRINTS_PATH,
        SEGMENTATION_PATH,
        META_PATH,
    )):
        raise FileNotFoundError(
            "era-boundary artifacts are missing; run "
            "`python -m presidential_profiles.era_boundaries`"
        )
    meta = json.loads(META_PATH.read_text())
    expected = meta.get("source_sha256", {})
    live = _source_fingerprints()
    if expected != live:
        raise ValueError(
            "era-boundary artifacts are stale against their governed Era Atlas "
            "inputs; regenerate them before building the site"
        )
    return {
        "sliding_scores": pd.read_parquet(SLIDING_SCORES_PATH),
        "cluster_stability": pd.read_parquet(CLUSTER_STABILITY_PATH),
        "consensus_scores": pd.read_parquet(CONSENSUS_SCORES_PATH),
        "cycle_fingerprints": pd.read_parquet(CYCLE_FINGERPRINTS_PATH),
        "segmentation": pd.read_parquet(SEGMENTATION_PATH),
        "meta": meta,
    }


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
        f"{len(result['segmentation'])} segmentation rows"
    )


if __name__ == "__main__":
    main()
