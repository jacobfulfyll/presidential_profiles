"""Era atlas: fingerprint every era, compare them, and find which past era the
present most resembles.

Every era (and every finer unit — presidency, 8-year bin) gets a **fingerprint**:
a standardized vector across everything the pipeline measures — level-1 topic mix,
combativeness rates, proposal-vs-values register, style markers (certainty,
us-vs-them, hype/doom, religiosity, ...), readability/pronoun/modal usage, and
speech-type mix. Three things fall out of it.

1. **A similarity matrix** — which eras resemble which, and the headline query:
   *which past era does the present most resemble?* Reported as BOTH a raw and a
   drift-detrended matrix, because PC1 of presidential language is time (README):
   a naive matrix trivially answers "the 2020s resemble the 2010s". The detrended
   matrix subtracts the local-era mean (the `similarity.build_adjusted` trick,
   `ERA_WINDOW = 24`), so genuine non-adjacent rhyme becomes visible.

2. **Data-driven periodization** — contiguity-constrained hierarchical clustering
   of the fine-grained fingerprints draws boundaries from the data, held against
   the historians' canonical `trends.ERAS`. Agreements validate; disagreements are
   the finding. A mandatory leave-one-out check drops the availability-bounded
   `opponents` marker (it names `democrats`/`republicans`/`fake news`, words that
   did not exist for most of the record) and re-derives the boundaries: any
   boundary that survives is about rhetoric, any that vanishes was about when the
   words became available.

3. **LLM-written era portraits** — short descriptions generated from each era's
   fingerprint plus exemplar quotes, provenance-stamped with a manifest.

--------------------------------------------------------------------------
What is free and what is paid
--------------------------------------------------------------------------
Everything except the portraits is FREE, deterministic local compute over frozen
parquets: `eras.run()` writes byte-identical outputs on rerun (no wall-clock
stamp — provenance identity is the `corpus_fingerprint`, following the
`combat_meta.json` precedent). The portraits are a PAID Anthropic run behind a
`--dry-run` estimate and a hard cost ceiling; the portrait artifact is the one
output allowed to carry a date, like the annotation layer.

Run as:
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.eras
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.eras portraits --dry-run
    set -a; source .env.local; set +a  # then:
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.eras portraits --run --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from .annotate import _rates as _annotate_rates
from .attention import (
    canonical_label_map,
    era_series,
    level1_parents,
    load_taxonomy,
    normalize_topics,
)
from .corpus import DATA_DIR, load
from .llm_annotations import (
    ANNOTATIONS_DIR,
    Manifest,
    corpus_fingerprint,
    load_paragraph_annotations,
    load_speech_annotations,
)
from .similarity import ERA_WINDOW
from .trends import ERAS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

ERAS_DIR = DATA_DIR / "eras"
MANIFESTS_DIR = ERAS_DIR / "manifests"

ERA_FINGERPRINTS_PATH = ERAS_DIR / "era_fingerprints.parquet"
ERA_SIMILARITY_PATH = ERAS_DIR / "era_similarity.parquet"
FINE_FINGERPRINTS_PATH = ERAS_DIR / "fine_fingerprints.parquet"
PRESIDENT_SIMILARITY_PATH = ERAS_DIR / "president_similarity.parquet"
NEAREST_NEIGHBORS_PATH = ERAS_DIR / "nearest_neighbors.parquet"
PERIODIZATION_PATH = ERAS_DIR / "periodization.parquet"
COMBAT_CI_CHECK_PATH = ERAS_DIR / "combativeness_ci_check.parquet"
PORTRAITS_PATH = ERAS_DIR / "era_portraits.parquet"
ERAS_META_PATH = ERAS_DIR / "eras_meta.json"

# Consumed-layer metas whose corpus_fingerprint must match the live corpus.
COMBAT_META_PATH = DATA_DIR / "combat" / "combat_meta.json"
COMBATIVENESS_PATH = DATA_DIR / "combat" / "combativeness.parquet"

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
SPEECH_MARKERS_PATH = DATA_DIR / "speech_markers.parquet"
SPEECH_STATS_PATH = DATA_DIR / "speech_stats.parquet"

# The optional annotator-disagreement input — produced by the separate
# `inter-model-agreement-check` task and ABSENT when this module was written, so
# every consumer of it goes through the optional loader below.
AGREEMENT_PATH = ANNOTATIONS_DIR / "agreement_v1.parquet"
AGREEMENT_REQUIRED_COLUMNS = {"era", "flag", "disagreement_half_width"}

# --------------------------------------------------------------------------
# axis registry
# --------------------------------------------------------------------------

ERA_ORDER = [label for label, _, _ in ERAS]
ERA_BOUNDS = {label: (lo, hi) for label, lo, hi in ERAS}
# Canonical INTERNAL boundaries: the first year of each era after the founding.
# These are the 8 boundaries the data-driven periodization is scored against.
CANONICAL_BOUNDARIES = [lo for _, lo, _ in ERAS[1:]]

BIN_WIDTH = 8  # "8-year bins" — do NOT impose decades (Key Decision).

# The three paragraph-flag combativeness axes. Computed here as raw per-unit flag
# means over the FULL paragraph set. Well powered at the era and 8-year-bin grains
# (the smallest bin holds 67 paragraphs); the presidency grain has a much thinner
# floor — Garfield is 29 paragraphs / 1 speech — so per-president fingerprints are
# the noisier grain (their marker/stat rates especially), which is why the era grain
# is the reliable similarity axis and the president grain is reported as robustness.
# The combat pipeline's genre_standardized / sotu_only estimates carry `ci_status`
# because their genre sub-cells can be thin (the n=3 War & New Deal SOTU cell);
# those suppressed cells never enter a fingerprint axis — see `combativeness_ci_check`.
COMBAT_FLAGS = ["party_attack", "enemy_naming", "zero_sum"]

# proposal_values register categories. `mixed` is dropped as the simplex residual
# (the four shares sum to 1) to avoid a perfectly collinear axis; it is recoverable
# as 1 - proposal - values - neither.
PROPOSAL_SHARES = {"proposal": "proposal_share", "values": "values_share",
                   "neither": "neither_share"}

# Style markers: raw COUNTS in speech_markers.parquet, converted to per-1k-word
# rates per unit (sum(count) / sum(n_words) * 1000). us_vs_them / hype / doom /
# opponents / religiosity are the acceptance-criteria style axes; the rest are the
# module's full marker set.
MARKER_COLUMNS = [
    "boosters", "hedges", "concessives", "assertive_modals", "need_to",
    "us_them", "superlatives", "opponents", "hype", "doom", "mechanism",
    "nrc_hope", "nrc_fear", "religiosity", "god_bless", "nostalgia", "future",
    "united_states", "america",
]
# The availability-bounded axis. Kept in the fingerprint (it is real signal) but
# subjected to a mandatory leave-one-out re-derivation of the periodization; see
# `periodization` and the note. NEVER "fix" indices.py:MARKERS from this task.
OPPONENTS_AXIS = "marker_opponents"

# speech_stats modal columns — counts, converted to per-1k-token rates.
MODAL_COLUMNS = [
    "modal_shall", "modal_will", "modal_must", "modal_should",
    "modal_can", "modal_may", "modal_would", "modal_could",
]

# --------------------------------------------------------------------------
# staleness guard (the cheapest fix lives at the consumer — here)
# --------------------------------------------------------------------------


def check_staleness(fingerprint: dict | None = None) -> dict:
    """Compare the live corpus fingerprint against each consumed layer's meta.

    A derived table goes stale silently when the corpus is refreshed under it
    (`derived-tables-go-stale-against-corpus-refresh`). The cheapest place to
    catch it is the consumer, so this warns LOUDLY on any mismatch rather than
    letting a stale number ship.

    Args:
        fingerprint: The live corpus fingerprint; computed when omitted.

    Returns:
        A dict of ``{layer: {"match": bool, "expected": ..., "live": ...}}``.
    """
    if fingerprint is None:
        fingerprint = corpus_fingerprint()
    results: dict[str, dict] = {}
    if COMBAT_META_PATH.exists():
        expected = json.loads(COMBAT_META_PATH.read_text()).get("corpus_fingerprint")
        match = expected == fingerprint
        results["combat_meta.json"] = {
            "match": match, "expected": expected, "live": fingerprint,
        }
        if not match:
            logger.warning(
                "STALE INPUT: %s corpus_fingerprint %s does not match the live "
                "corpus %s. The combativeness axes were computed against a "
                "different corpus; refuse to trust the atlas until combat is "
                "regenerated.", COMBAT_META_PATH.name, expected, fingerprint,
            )
    return results


# --------------------------------------------------------------------------
# optional annotator-disagreement seam (mirrors combat.load_agreement_bands)
# --------------------------------------------------------------------------


def load_agreement_bands(path: Path | None = None) -> pd.DataFrame | None:
    """Annotator-disagreement half-widths per (era, flag), or None if absent.

    The bands come from `inter-model-agreement-check`, which had not produced
    `agreement_v1.parquet` when this module was written. Absence is a normal,
    expected state and returns None — callers then ship sampling-only similarity
    and record that fact on every output row. Absence is NEVER filled with a
    stub: a fabricated disagreement magnitude would be indistinguishable from a
    measured one, which is exactly the failure this seam prevents. If the file
    DOES appear but lacks the expected columns, this raises rather than guessing.

    The default is bound in the body, not the signature, so monkeypatching the
    module constant in a test is not silently inert (the combat precedent).
    """
    if path is None:
        path = AGREEMENT_PATH
    if not path.exists():
        return None
    bands = pd.read_parquet(path)
    missing = AGREEMENT_REQUIRED_COLUMNS - set(bands.columns)
    if missing:
        raise ValueError(
            f"{path.name} exists but is missing columns {sorted(missing)}; "
            f"refusing to guess at the disagreement-band schema. Expected at "
            f"least {sorted(AGREEMENT_REQUIRED_COLUMNS)}."
        )
    return bands


def _agreement_provenance(bands: pd.DataFrame | None) -> tuple[str, str]:
    """(ci_components, agreement_source) for the similarity rows.

    Similarity is a cosine over the whole fingerprint vector, not a per-(era,
    flag) rate, so even when a band table exists it cannot be applied to a
    cosine without a model of how per-flag disagreement propagates through the
    z-scoring and the dot product — which does not exist yet. The seam therefore
    records the component as absent whenever no such propagation is available,
    and never claims `sampling+annotator` on a matrix it did not widen.
    """
    if bands is None:
        return "sampling_only", f"absent:{AGREEMENT_PATH.name}"
    # A band table exists but this module has no cosine-propagation model for it.
    return "sampling_only", f"present_not_propagated:{AGREEMENT_PATH.name}"


# --------------------------------------------------------------------------
# master paragraph frame
# --------------------------------------------------------------------------


def _require_full_merge(n_merged: int, inputs: dict[str, int], stage: str) -> None:
    """Raise unless a keyed merge kept every row of every input.

    `validate="one_to_one"` catches DUPLICATE keys, but an inner join silently
    DROPS rows when the two key sets merely diverge (CLAUDE.md). Mirrors
    `taxonomy._require_full_merge` / `combat._require_full_merge` rather than
    importing either (this module's contract is zero API calls; it should not
    depend on the paid module's import graph).
    """
    if any(n_merged != n for n in inputs.values()):
        breakdown = ", ".join(f"{k}={v}" for k, v in inputs.items())
        raise ValueError(
            f"{stage}: keyed merge changed the row count (merged={n_merged}; "
            f"{breakdown}). The key sets diverge and the inner join dropped "
            f"unmatched rows. Refusing to proceed."
        )


def load_master_frame(
    annotations: pd.DataFrame | None = None,
    speech_annotations: pd.DataFrame | None = None,
    paragraphs: pd.DataFrame | None = None,
    speeches: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
) -> pd.DataFrame:
    """One paragraph-level frame carrying every fingerprint input.

    Columns: doc_name, para_idx, year, era, president, bin8, center_year,
    speech_type, the three combativeness flags, proposal_values, and one boolean
    column per level-1 domain (`topic__<domain>`) recording whether the
    paragraph's normalized topics touch that domain.

    Every input is injectable so the assembly can be exercised on tiny synthetic
    frames without the 36k-row corpus (the repo's testing convention). Passing
    nothing reads the real frozen parquets.
    """
    taxonomy = taxonomy or load_taxonomy()
    label_map = canonical_label_map(taxonomy)
    parent = level1_parents(taxonomy)
    domains = sorted(set(parent.values()))

    anns = (
        load_paragraph_annotations("paragraph_annotations")
        if annotations is None else annotations
    )
    speech_anns = (
        load_speech_annotations("speech_annotations")
        if speech_annotations is None else speech_annotations
    )
    paras = pd.read_parquet(PARAGRAPHS_PATH) if paragraphs is None else paragraphs
    speeches = load() if speeches is None else speeches

    df = anns[["doc_name", "para_idx", "topics", *COMBAT_FLAGS,
               "proposal_values"]].merge(
        paras[["doc_name", "para_idx", "word_count"]],
        on=["doc_name", "para_idx"], validate="one_to_one",
    )
    _require_full_merge(
        len(df), {"paragraph_annotations": len(anns), "paragraphs": len(paras)},
        "paragraph_annotations x paragraphs",
    )

    meta = speeches[["doc_name", "president", "year"]].merge(
        speech_anns[["doc_name", "speech_type"]], on="doc_name",
        validate="one_to_one",
    )
    _require_full_merge(
        len(meta), {"speeches": len(speeches), "speech_annotations": len(speech_anns)},
        "speeches x speech_annotations",
    )

    n_before = len(df)
    df = df.merge(meta, on="doc_name", validate="many_to_one")
    if len(df) != n_before:
        raise ValueError(
            f"paragraphs x speech metadata changed the row count "
            f"({n_before} -> {len(df)}); some paragraph's doc_name is not in the "
            f"speech table."
        )

    df["era"] = era_series(df["year"])
    if df["era"].isna().any():
        bad = sorted(df.loc[df["era"].isna(), "year"].unique())
        raise ValueError(f"years outside every trends.ERAS band: {bad}")
    df["era"] = pd.Categorical(df["era"], ERA_ORDER, ordered=True)
    df["bin8"] = (df["year"] // BIN_WIDTH) * BIN_WIDTH
    df["center_year"] = df["bin8"] + (BIN_WIDTH - 1) / 2.0

    # One boolean per level-1 domain. normalize_topics de-dups case variants and
    # RAISES on any label not in taxonomy_v1's 50 names (reused, not re-rolled).
    topic_domains = df["topics"].map(
        lambda raw: {parent[t] for t in normalize_topics(raw, label_map)}
    )
    for domain in domains:
        df[f"topic__{domain}"] = topic_domains.map(lambda ds, d=domain: d in ds)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# per-unit axis assembly
# --------------------------------------------------------------------------

GRAIN_KEY = {"era": "era", "bin8": "bin8", "president": "president"}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total else float("nan")


def build_axis_frame(
    df: pd.DataFrame,
    grain: str,
    markers: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
) -> pd.DataFrame:
    """Per-unit RAW axis values at one grain (``era`` | ``bin8`` | ``president``).

    Returns one row per unit with: the grouping key, ``center_year`` (paragraph-
    mean year — the temporal position used for detrending and clustering ordering),
    ``n_paragraphs`` / ``n_speeches``, and every fingerprint axis as a raw value.
    Axes are grouped by prefix: ``topic__``, ``combat_``, ``register_``,
    ``marker_``, ``stat_``, ``genre_``.
    """
    if grain not in GRAIN_KEY:
        raise ValueError(f"unknown grain {grain!r}; expected one of {list(GRAIN_KEY)}")
    key = GRAIN_KEY[grain]
    markers = pd.read_parquet(SPEECH_MARKERS_PATH) if markers is None else markers
    stats = pd.read_parquet(SPEECH_STATS_PATH) if stats is None else stats
    taxonomy = taxonomy or load_taxonomy()
    domains = sorted(set(level1_parents(taxonomy).values()))

    # Doc-set completeness guard (the same silent-partial class the paragraph
    # joins guard). The per-unit marker/stat aggregation filters with
    # `.isin(docs)`; a doc missing from speech_markers/speech_stats would be
    # silently dropped from the numerator while `n_speeches` still counts it,
    # quietly biasing that unit's marker/stat rates. An inner join would hide it,
    # so check coverage up front over the whole corpus and RAISE naming the gap.
    corpus_docs = set(df["doc_name"].unique())
    for name, table in (("speech_markers", markers), ("speech_stats", stats)):
        missing = sorted(corpus_docs - set(table["doc_name"].unique()))
        if missing:
            raise ValueError(
                f"build_axis_frame: {len(missing)} corpus doc(s) are absent from "
                f"{name}, so their marker/stat rates would be computed off a partial "
                f"speech set while n_speeches still counts them. Missing (first 5): "
                f"{missing[:5]}. Refusing to build a silently-biased fingerprint."
            )

    speech_types = sorted(df["speech_type"].unique())

    rows: list[dict] = []
    for unit, para_group in df.groupby(key, observed=True):
        n_para = len(para_group)
        docs = para_group["doc_name"].unique()
        row: dict = {
            key: unit,
            "center_year": float(para_group["year"].mean()),
            "n_paragraphs": int(n_para),
            "n_speeches": int(len(docs)),
        }

        # --- topic mix (level-1 shares) ---
        for domain in domains:
            row[f"topic__{domain}"] = float(para_group[f"topic__{domain}"].mean())

        # --- combativeness (raw flag means over the full paragraph set) ---
        for flag in COMBAT_FLAGS:
            row[f"combat_{flag}"] = float(para_group[flag].mean())

        # --- register (proposal-vs-values shares) ---
        for cat, name in PROPOSAL_SHARES.items():
            row[f"register_{name}"] = float((para_group["proposal_values"] == cat).mean())

        # --- style markers (per-1k-word rates over the unit's speeches) ---
        m = markers[markers["doc_name"].isin(docs)]
        total_words = float(m["n_words"].sum())
        for col in MARKER_COLUMNS:
            row[f"marker_{col}"] = (
                float(m[col].sum()) / total_words * 1000.0 if total_words else float("nan")
            )

        # --- readability / pronoun / modal (from speech_stats) ---
        s = stats[stats["doc_name"].isin(docs)]
        tok = s["n_tokens"].to_numpy(dtype=float)
        row["stat_fk_grade"] = _weighted_mean(s["fk_grade"].to_numpy(dtype=float), tok)
        row["stat_words_per_sentence"] = _weighted_mean(
            s["words_per_sentence"].to_numpy(dtype=float), tok
        )
        i_tot, we_tot = float(s["i_count"].sum()), float(s["we_count"].sum())
        row["stat_self_reference"] = (
            i_tot / (i_tot + we_tot) if (i_tot + we_tot) else float("nan")
        )
        total_tok = float(tok.sum())
        for col in MODAL_COLUMNS:
            row[f"stat_{col}_per_1k"] = (
                float(s[col].sum()) / total_tok * 1000.0 if total_tok else float("nan")
            )

        # --- speech-type mix (paragraph share by genre) ---
        genre_share = para_group["speech_type"].value_counts(normalize=True)
        for st in speech_types:
            row[f"genre__{st}"] = float(genre_share.get(st, 0.0))

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("center_year").reset_index(drop=True)
    return out


def axis_columns(frame: pd.DataFrame) -> list[str]:
    """The fingerprint-axis columns of an axis frame, in stable declared order."""
    prefixes = ("topic__", "combat_", "register_", "marker_", "stat_", "genre__")
    return [c for c in frame.columns if c.startswith(prefixes)]


def axis_group(axis: str) -> str:
    """The coarse group an axis belongs to (for group-balanced robustness)."""
    for grp in ("topic", "combat", "register", "marker", "stat", "genre"):
        if axis.startswith(grp + "__") or axis.startswith(grp + "_"):
            return grp
    raise ValueError(f"axis {axis!r} matches no known group")


# --------------------------------------------------------------------------
# z-scoring
# --------------------------------------------------------------------------


def zscore(frame: pd.DataFrame, axes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Z-score each axis across the frame's units (population std, ddof=0).

    Returns ``(Z, kept_axes)``. A zero-variance axis carries no discriminative
    signal across these units and is DROPPED (recorded by its absence from
    `kept_axes`) rather than yielding a 0/0 NaN column that would poison the
    cosine. Determinism: no randomness; ddof=0 is exact.
    """
    raw = frame[axes].to_numpy(dtype=float)
    mu = raw.mean(axis=0)
    sigma = raw.std(axis=0, ddof=0)
    keep = sigma > 0
    kept_axes = [a for a, k in zip(axes, keep) if k]
    z = (raw[:, keep] - mu[keep]) / sigma[keep]
    return z, kept_axes


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def cosine_matrix(vectors: np.ndarray, labels: list) -> pd.DataFrame:
    """Cosine similarity between rows of `vectors`, labelled by `labels`."""
    unit = _l2_normalize(vectors)
    sim = unit @ unit.T
    return pd.DataFrame(sim, index=labels, columns=labels)


def detrend_local_mean(
    vectors: np.ndarray, years: np.ndarray, window: float = ERA_WINDOW
) -> np.ndarray:
    """Subtract each unit's local-era mean, reusing `similarity.build_adjusted`.

    For each unit, subtract the mean of the units whose center year is within
    `window` years (excluding self). When fewer than two contemporaries fall in
    the window, fall back to the four nearest in time (build_adjusted's own
    fallback). The residual is what distinguishes a unit from its contemporaries,
    which is what makes cross-era comparison meaningful once the monotonic drift
    (PC1 = time) is removed.
    """
    adjusted = np.zeros_like(vectors)
    n = len(vectors)
    for i in range(n):
        mask = (np.abs(years - years[i]) <= window) & (np.arange(n) != i)
        if mask.sum() < 2:
            nearest = np.argsort(np.abs(years - years[i]))[1:5]
            mask = np.zeros(n, dtype=bool)
            mask[nearest] = True
        adjusted[i] = vectors[i] - vectors[mask].mean(axis=0)
    return adjusted


def group_balance_weights(kept_axes: list[str]) -> np.ndarray:
    """Per-axis weights that equalize the six axis GROUPS in the cosine.

    Every z-axis already has unit variance, so a group with more axes contributes
    more to the dot product simply by count (topic has 17, marker 19, genre 10).
    Weighting each z-axis in group *g* by ``1/sqrt(n_g)`` makes group *g*
    contribute the MEAN of its per-axis products to the cosine
    (``sum_{i in g} (1/sqrt(n_g))^2 a_i b_i = mean_{i in g} a_i b_i``), so all six
    groups enter with equal weight regardless of how many axes each holds.
    """
    from collections import Counter  # noqa: PLC0415 — local, single use

    sizes = Counter(axis_group(a) for a in kept_axes)
    return np.array([1.0 / np.sqrt(sizes[axis_group(a)]) for a in kept_axes])


def group_balanced_era_detrended(era_axis: pd.DataFrame) -> pd.DataFrame:
    """Detrended era cosine under group balancing — the headline robustness check.

    Same pipeline as the primary detrended era matrix (z-score, adjacent-era
    detrend, cosine) but with the group-balance weights applied to the z-vectors
    first, so the nearest-neighbor result cannot be an artifact of one axis group
    out-numbering the others. Deterministic; emitted so the reported number is
    diffable on rerun."""
    axes = axis_columns(era_axis)
    z, kept = zscore(era_axis, axes)
    zb = z * group_balance_weights(kept)
    det = detrend_adjacent_eras(zb)
    labels = era_axis["era"].astype(str).tolist()
    return cosine_matrix(det, labels)


def detrend_adjacent_eras(vectors: np.ndarray) -> np.ndarray:
    """Subtract each era's adjacent-era mean (the build_adjusted pattern adapted
    to the 9 ordered eras, where a fixed 24-year window would capture only the
    era itself).

    `vectors` MUST be in chronological era order. For era *i* subtract the mean
    of eras {i-1, i+1}; the endpoints (founding, present) subtract their single
    neighbor. This removes the local sequential drift so that non-adjacent rhyme
    is what survives.
    """
    n = len(vectors)
    adjusted = np.zeros_like(vectors)
    for i in range(n):
        neighbors = [j for j in (i - 1, i + 1) if 0 <= j < n]
        adjusted[i] = vectors[i] - vectors[neighbors].mean(axis=0)
    return adjusted


# --------------------------------------------------------------------------
# similarity long tables + nearest neighbors
# --------------------------------------------------------------------------


def _similarity_long(
    matrix: pd.DataFrame, grain: str, kind: str,
    order: dict, ci_components: str, agreement_source: str,
) -> pd.DataFrame:
    """Melt a square similarity matrix to (unit_i, unit_j, cosine) long form."""
    rows = []
    labels = list(matrix.index)
    for a in labels:
        for b in labels:
            rows.append({
                "grain": grain, "matrix": kind,
                "unit_i": str(a), "unit_j": str(b),
                "unit_i_order": order[a], "unit_j_order": order[b],
                "cosine": float(matrix.loc[a, b]),
                "ci_components": ci_components,
                "agreement_source": agreement_source,
            })
    return pd.DataFrame(rows)


def nearest_neighbors(
    matrix: pd.DataFrame, grain: str, kind: str, center_year: dict,
    ci_components: str, agreement_source: str,
) -> pd.DataFrame:
    """Each unit's single most-similar OTHER unit, and whether that neighbor is
    its immediate temporal neighbor (the detrending sanity check).

    Carries the same `ci_components` / `agreement_source` provenance as the
    similarity long tables, so every similarity-bearing artifact — not just the
    matrices — records which uncertainty components its cosine contains."""
    labels = list(matrix.index)
    order = sorted(labels, key=lambda u: center_year[u])
    pos = {u: i for i, u in enumerate(order)}
    rows = []
    for u in labels:
        others = matrix.loc[u].drop(u)
        nn = others.idxmax()
        temporal = {order[pos[u] - 1] if pos[u] > 0 else None,
                    order[pos[u] + 1] if pos[u] < len(order) - 1 else None}
        rows.append({
            "grain": grain, "matrix": kind, "unit": str(u),
            "nearest": str(nn), "cosine": float(others.max()),
            "nearest_is_temporal_neighbor": nn in temporal,
            "ci_components": ci_components,
            "agreement_source": agreement_source,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# data-driven periodization
# --------------------------------------------------------------------------


def _chain_connectivity(n: int) -> np.ndarray:
    """A path graph over `n` temporally-ordered units. Feeding this as the
    `connectivity` constraint to agglomerative clustering forces every cluster to
    be a contiguous run of time — so a cluster boundary IS a period boundary."""
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    return adj


def _boundaries_at_k(z: np.ndarray, bins: np.ndarray, k: int) -> list[int]:
    """Contiguity-constrained Ward clustering into `k` periods; return the START
    years of every period after the first (the discovered boundaries).

    Deterministic: Ward linkage with a fixed connectivity graph has no random
    state. The boundary year is ``bins[i]`` — the LATER unit's bin-start, i.e. the
    NEW period's start year (a maintainer must not "correct" this to ``bins[i-1]``;
    that would shift every reported boundary one bin earlier)."""
    n = len(z)
    if k <= 1 or k > n:
        return []
    model = AgglomerativeClustering(
        n_clusters=k, linkage="ward", connectivity=_chain_connectivity(n),
    )
    labels = model.fit_predict(z)
    boundaries = []
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            boundaries.append(int(bins[i]))
    return sorted(boundaries)


def _match_canonical(boundary: int) -> tuple[int, int]:
    """Nearest canonical boundary to `boundary` and the signed ``boundary -
    nearest`` gap. This helper only MEASURES the gap; whether it counts as a
    match is the caller's single-sourced decision (`abs(gap) <= BIN_WIDTH` in
    `periodization`), so the tolerance lives at that one site, not here."""
    nearest = min(CANONICAL_BOUNDARIES, key=lambda c: abs(c - boundary))
    return nearest, boundary - nearest


def periodization(
    fine_frames: dict[str, pd.DataFrame], k_primary: int = len(ERAS),
    k_range: range = range(2, 13),
) -> pd.DataFrame:
    """Data-driven boundaries vs the historians' canonical eras, at each grain and
    under the mandatory leave-one-out on `opponents`.

    For each (grain, condition, k) the contiguity-constrained clustering yields a
    set of boundary years; each is matched to its nearest canonical boundary. The
    `drop_opponents` condition rebuilds the fingerprint with the availability-
    bounded `opponents` marker removed: a boundary that survives is about
    rhetoric, one that vanishes was about when the party-name words became
    available (2026-07-21 amendment).
    """
    rows = []
    for grain, frame in fine_frames.items():
        axes_all = axis_columns(frame)
        # president grain has no bin; use the rounded center year as the boundary.
        boundary_years = (
            frame["bin8"].to_numpy() if "bin8" in frame
            else np.round(frame["center_year"]).astype(int).to_numpy()
        )
        for condition, axes in (
            ("full", axes_all),
            ("drop_opponents", [a for a in axes_all if a != OPPONENTS_AXIS]),
        ):
            z, _ = zscore(frame, axes)
            for k in k_range:
                for b in _boundaries_at_k(z, boundary_years, k):
                    nearest, gap = _match_canonical(b)
                    rows.append({
                        "grain": grain, "condition": condition, "k": k,
                        "boundary_year": int(b),
                        "nearest_canonical": int(nearest),
                        "gap_years": int(gap),
                        "matches_canonical": abs(gap) <= BIN_WIDTH,
                        "is_primary_k": k == k_primary,
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# combativeness ci_status audit
# --------------------------------------------------------------------------


def combativeness_ci_check(combativeness: pd.DataFrame | None = None) -> pd.DataFrame:
    """Audit every combat.parquet cell for whether it could seed a fingerprint
    axis, honoring `ci_status` (CLAUDE.md's trust gate).

    The fingerprint's combativeness axes are per-unit RAW flag means over the full
    paragraph set, which corresponds to the combat table's `raw` treatment (all
    `ci_status == ok`). The `genre_standardized` / `sotu_only` treatments carry
    `suppressed_n_floor` cells (thin genre sub-cells); this table records that
    NONE of them was used, and how many were excluded. Never read a combat
    `rate` without its `ci_status` — this is that read, made explicit.
    """
    if combativeness is None:
        combativeness = pd.read_parquet(COMBATIVENESS_PATH)
    out = combativeness[["era", "flag", "treatment", "rate", "ci_status",
                         "n_speeches"]].copy()
    # Only the `raw` treatment feeds a fingerprint axis (as a per-unit flag mean).
    out["used_as_axis"] = (out["treatment"] == "raw")
    out["reason"] = np.where(
        out["used_as_axis"],
        "raw all-genre flag mean -> combativeness fingerprint axis",
        "not used: genre_standardized/sotu_only estimate, not a fingerprint axis",
    )
    if (out.loc[out["used_as_axis"], "ci_status"] != "ok").any():
        bad = out[(out["used_as_axis"]) & (out["ci_status"] != "ok")]
        raise ValueError(
            "a combativeness axis would be built from a non-ok cell:\n"
            f"{bad.to_string(index=False)}"
        )
    return out.reset_index(drop=True)


def validate_combat_axes(era_axis: pd.DataFrame, combativeness: pd.DataFrame) -> dict:
    """Confirm the per-era raw flag means equal the combat pipeline's `raw`
    treatment rate (same estimand, independently computed). A mismatch means the
    two derivations of combativeness have diverged."""
    raw = combativeness[combativeness["treatment"] == "raw"]
    diffs = {}
    for flag in COMBAT_FLAGS:
        pipeline = raw[raw["flag"] == flag].set_index("era")["rate"]
        mine = era_axis.set_index("era")[f"combat_{flag}"]
        merged = pd.concat([pipeline.rename("pipe"), mine.rename("mine")], axis=1).dropna()
        diffs[flag] = float((merged["pipe"] - merged["mine"]).abs().max())
    worst = max(diffs.values())
    if worst > 1e-9:
        raise ValueError(
            f"per-era raw combativeness axes disagree with combat.parquet raw "
            f"treatment (max abs diff {worst:.2e}): {diffs}"
        )
    return diffs


# --------------------------------------------------------------------------
# fingerprint long tables
# --------------------------------------------------------------------------


def _fingerprint_long(frame: pd.DataFrame, grain: str, key: str) -> pd.DataFrame:
    """Melt an axis frame to (unit, measure, value_raw, value_z) long form.

    Z-scored across the frame's own units per axis, so `era_fingerprints` is
    z-scored across the 9 eras and `fine_fingerprints` across the fine units."""
    axes = axis_columns(frame)
    z, kept = zscore(frame, axes)
    zmap = {a: z[:, i] for i, a in enumerate(kept)}
    rows = []
    for pos in range(len(frame)):
        r = frame.iloc[pos]
        unit = r[key]
        for a in axes:
            rows.append({
                "grain": grain, "unit": str(unit),
                "center_year": float(r["center_year"]),
                "measure": a, "axis_group": axis_group(a),
                "value_raw": float(r[a]),
                "value_z": float(zmap[a][pos]) if a in zmap else 0.0,
                "zero_variance": a not in zmap,
                "n_paragraphs": int(r["n_paragraphs"]),
                "n_speeches": int(r["n_speeches"]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def run(df: pd.DataFrame | None = None, out_dir: Path | None = None) -> dict:
    """Build every DETERMINISTIC atlas artifact (portraits are a separate paid
    step). Writes byte-identical outputs on rerun."""
    out_dir = ERAS_DIR if out_dir is None else out_dir
    df = load_master_frame() if df is None else df

    fingerprint = corpus_fingerprint()
    staleness = check_staleness(fingerprint)

    combativeness = pd.read_parquet(COMBATIVENESS_PATH)
    ci_check = combativeness_ci_check(combativeness)

    era_axis = build_axis_frame(df, "era")
    # trends.ERAS order, not center-year order (identical here, but explicit).
    era_axis["_order"] = era_axis["era"].map({e: i for i, e in enumerate(ERA_ORDER)})
    era_axis = era_axis.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    combat_diffs = validate_combat_axes(era_axis, combativeness)

    bin_axis = build_axis_frame(df, "bin8")
    pres_axis = build_axis_frame(df, "president")

    # --- fingerprints ---
    era_fp = _fingerprint_long(era_axis, "era", "era")
    fine_fp = pd.concat([
        _fingerprint_long(bin_axis, "bin8", "bin8"),
        _fingerprint_long(pres_axis, "president", "president"),
    ], ignore_index=True)

    # --- similarity (era grain, both matrices) ---
    ci_components, agreement_source = _agreement_provenance(load_agreement_bands())
    era_order_map = {e: i for i, e in enumerate(ERA_ORDER)}
    era_center = dict(zip(era_axis["era"].astype(str), era_axis["center_year"]))
    z_era, _ = zscore(era_axis, axis_columns(era_axis))
    era_labels = era_axis["era"].astype(str).tolist()
    raw_era = cosine_matrix(z_era, era_labels)
    det_era = cosine_matrix(detrend_adjacent_eras(z_era), era_labels)
    era_sim = pd.concat([
        _similarity_long(raw_era, "era", "raw", era_order_map, ci_components, agreement_source),
        _similarity_long(det_era, "era", "detrended", era_order_map, ci_components, agreement_source),
    ], ignore_index=True)

    # --- similarity (president grain — the faithful build_adjusted reuse) ---
    pres_order = {p: i for i, p in enumerate(pres_axis["president"])}
    pres_center = dict(zip(pres_axis["president"], pres_axis["center_year"]))
    z_pres, _ = zscore(pres_axis, axis_columns(pres_axis))
    pres_labels = pres_axis["president"].tolist()
    pres_years = pres_axis["center_year"].to_numpy()
    raw_pres = cosine_matrix(z_pres, pres_labels)
    det_pres = cosine_matrix(detrend_local_mean(z_pres, pres_years), pres_labels)
    pres_sim = pd.concat([
        _similarity_long(raw_pres, "president", "raw", pres_order, ci_components, agreement_source),
        _similarity_long(det_pres, "president", "detrended", pres_order, ci_components, agreement_source),
    ], ignore_index=True)

    # --- nearest neighbors (headline + sanity check) ---
    nn = pd.concat([
        nearest_neighbors(raw_era, "era", "raw", era_center, ci_components, agreement_source),
        nearest_neighbors(det_era, "era", "detrended", era_center, ci_components, agreement_source),
        nearest_neighbors(raw_pres, "president", "raw", pres_center, ci_components, agreement_source),
        nearest_neighbors(det_pres, "president", "detrended", pres_center, ci_components, agreement_source),
    ], ignore_index=True)

    # --- periodization (+ leave-one-out) ---
    period = periodization({"bin8": bin_axis, "president": pres_axis})

    # --- group-balanced robustness on the headline (deterministic, emitted) ---
    gb_era = group_balanced_era_detrended(era_axis)
    ref = "The present era"
    gb_present = gb_era.loc[ref].drop(ref).sort_values(ascending=False)
    robustness = {
        "weighting": "each z-axis multiplied by 1/sqrt(n_group) so all six axis "
                     "groups contribute their MEAN per-axis product to the cosine "
                     "(equalizing groups regardless of axis count)",
        "matrix": "detrended, era grain",
        "present_era_nearest": str(gb_present.index[0]),
        "present_era_nearest_cosine": round(float(gb_present.iloc[0]), 6),
        "present_era_ranking": {str(k): round(float(v), 6) for k, v in gb_present.items()},
        "agrees_with_primary": str(gb_present.index[0]) == "Civil War & Reconstruction",
    }

    tables = {
        "era_fingerprints": era_fp,
        "era_similarity": era_sim,
        "fine_fingerprints": fine_fp,
        "president_similarity": pres_sim,
        "nearest_neighbors": nn,
        "periodization": period,
        "combativeness_ci_check": ci_check,
    }
    paths = {
        "era_fingerprints": ERA_FINGERPRINTS_PATH,
        "era_similarity": ERA_SIMILARITY_PATH,
        "fine_fingerprints": FINE_FINGERPRINTS_PATH,
        "president_similarity": PRESIDENT_SIMILARITY_PATH,
        "nearest_neighbors": NEAREST_NEIGHBORS_PATH,
        "periodization": PERIODIZATION_PATH,
        "combativeness_ci_check": COMBAT_CI_CHECK_PATH,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        t = table.copy()
        for col in t.columns:
            if isinstance(t[col].dtype, pd.CategoricalDtype):
                t[col] = t[col].astype(str)
        t.to_parquet(out_dir / paths[name].name, index=False)

    unused = ci_check[~ci_check["used_as_axis"]]
    n_suppressed = int((unused["ci_status"] == "suppressed_n_floor").sum())
    n_caution = int((unused["ci_status"] == "low_cluster_caution").sum())
    detrended_signal = float(det_era.to_numpy()[~np.eye(len(det_era), dtype=bool)].std())
    meta = {
        # No wall-clock stamp on purpose (the combat_meta.json precedent): every
        # output here is a pure function of the frozen inputs, so a timestamp
        # would dirty `git status data/eras/` on rerun for no information.
        # Provenance identity is `corpus_fingerprint`; WHEN it ran is git's job.
        "era_scheme": "trends.ERAS",
        "grains": ["era", "bin8 (8-year bins)", "president"],
        "bin_width_years": BIN_WIDTH,
        "detrend": {
            "era_grain": "adjacent-era mean subtraction (build_adjusted pattern, "
                         "adapted to the 9 ordered eras)",
            "president_grain": "local-era mean subtraction within ERA_WINDOW",
            "era_window_years": ERA_WINDOW,
        },
        "axes": {
            "n_total": len(axis_columns(era_axis)),
            "groups": {g: sum(axis_group(a) == g for a in axis_columns(era_axis))
                       for g in ("topic", "combat", "register", "marker", "stat", "genre")},
            "availability_bounded": OPPONENTS_AXIS,
        },
        "combativeness": {
            "axis_source": "raw per-unit flag means (== combat.parquet raw "
                           "treatment, all ci_status=ok)",
            "combat_raw_max_abs_diff": {k: round(v, 12) for k, v in combat_diffs.items()},
            "n_cells_excluded_suppressed_n_floor": n_suppressed,
            "n_cells_excluded_low_cluster_caution": n_caution,
            "ci_status_note": "genre_standardized/sotu_only suppressed cells never "
                              "seeded a fingerprint axis (CLAUDE.md ci_status rule)",
        },
        "periodization": {
            "method": "contiguity-constrained Ward clustering (path-graph "
                      "connectivity) on the RAW z-fingerprints",
            "primary_k": len(ERAS),
            "canonical_boundaries": CANONICAL_BOUNDARIES,
            "match_tolerance_years": BIN_WIDTH,
            "leave_one_out": f"rebuilt with {OPPONENTS_AXIS} dropped",
        },
        "similarity": {
            "detrended_offdiag_std": round(detrended_signal, 6),
            "group_balanced_robustness": robustness,
            "annotator_disagreement": {
                "applied": False,
                "ci_components": ci_components,
                "agreement_source": agreement_source,
                "status": "ABSENT — similarity contains sampling structure only; "
                          "agreement_v1.parquet pending inter-model-agreement-check",
            },
        },
        "staleness_check": staleness,
        "corpus_fingerprint": fingerprint,
        "portraits": {
            "artifact": PORTRAITS_PATH.name,
            "status": "generated separately via `eras portraits` (paid path)",
        },
        "api_calls": 0,
    }
    (out_dir / ERAS_META_PATH.name).write_text(json.dumps(meta, indent=2) + "\n")
    return {"tables": tables, "meta": meta,
            "matrices": {"raw_era": raw_era, "det_era": det_era,
                         "raw_pres": raw_pres, "det_pres": det_pres},
            "axis_frames": {"era": era_axis, "bin8": bin_axis, "president": pres_axis}}


# --------------------------------------------------------------------------
# LLM era portraits (PAID path — dry-run first, hard cost ceiling)
# --------------------------------------------------------------------------

PORTRAIT_MODEL = "claude-sonnet-5"
PORTRAIT_PROMPT_VERSION = "era-portrait/v1-2026-07-21"
PORTRAIT_MAX_TOKENS = 400
PORTRAIT_COST_CEILING_USD = 5.00
EXEMPLARS_PER_ERA = 4
EXEMPLAR_MIN_WORDS = 20
EXEMPLAR_MAX_WORDS = 200
TOP_AXES_IN_PROMPT = 10

PORTRAIT_RUBRIC = (
    "You are writing a compact, evidence-bound portrait of one era of United "
    "States presidential rhetoric. You are given (a) the era's most DISTINCTIVE "
    "measured features, each as a z-score relative to the 240-year average "
    "(positive = far above average, negative = far below), and (b) a few "
    "verbatim exemplar paragraphs from the era.\n\n"
    "Write 90-140 words. Ground EVERY claim in the supplied features or quotes — "
    "do not add outside historical facts, dates, president names, or events that "
    "are not present in the material. Describe what the LANGUAGE emphasizes and "
    "de-emphasizes. Prefer the strongest z-scores. If the features are mixed, "
    "say so rather than inventing a clean story."
)


def _human_axis(measure: str) -> str:
    """A readable label for an axis in the portrait prompt."""
    grp = axis_group(measure)
    body = measure.split("__", 1)[-1] if "__" in measure else measure.split("_", 1)[-1]
    body = body.replace("_", " ")
    return {"topic": f"topic emphasis: {body}",
            "combat": f"combativeness: {body}",
            "register": f"register: {body}",
            "marker": f"style marker: {body}",
            "stat": f"readability/usage: {body}",
            "genre": f"speech-type share: {body}"}[grp]


def select_exemplars(df: pd.DataFrame, era: str, era_fp: pd.DataFrame,
                     paragraphs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deterministically pick exemplar paragraphs grounding an era's portrait.

    Candidates are the era's paragraphs within the quotability word window
    (`EXEMPLAR_MIN_WORDS`..`EXEMPLAR_MAX_WORDS`, 20-200 — chosen to match combat's
    exemplar bounds, but a VALUE COPY here, not an import, so the two can drift).
    They are scored by how many of the era's
    top-3 positive-z level-1 topics they carry, then sorted by (score desc,
    word_count desc, year, doc_name, para_idx) — a total order, so the selection
    is reproducible. Provenance (doc_name, para_idx, year) travels with the text.
    """
    paragraphs = pd.read_parquet(PARAGRAPHS_PATH) if paragraphs is None else paragraphs
    top_topics = (
        era_fp[(era_fp["unit"] == era) & (era_fp["axis_group"] == "topic")]
        .sort_values("value_z", ascending=False)["measure"].head(3).tolist()
    )
    top_domains = [m.split("topic__", 1)[1] for m in top_topics]

    cell = df[df["era"].astype(str) == era].copy()
    cell = cell.merge(paragraphs[["doc_name", "para_idx", "text"]],
                      on=["doc_name", "para_idx"], validate="one_to_one")
    cell = cell[(cell["word_count"] >= EXEMPLAR_MIN_WORDS)
                & (cell["word_count"] <= EXEMPLAR_MAX_WORDS)]
    cell["topic_score"] = sum(
        cell[f"topic__{d}"].astype(int) for d in top_domains
    ) if top_domains else 0
    cell = cell.sort_values(
        ["topic_score", "word_count", "year", "doc_name", "para_idx"],
        ascending=[False, False, True, True, True],
    )
    return cell.head(EXEMPLARS_PER_ERA)[
        ["doc_name", "para_idx", "year", "word_count", "text"]
    ].reset_index(drop=True)


def build_portrait_prompt(era: str, era_fp: pd.DataFrame, exemplars: pd.DataFrame) -> str:
    """The per-era user prompt: top |z| features + exemplar quotes."""
    lo, hi = ERA_BOUNDS[era]
    feats = (
        era_fp[era_fp["unit"] == era].assign(absz=lambda d: d["value_z"].abs())
        .sort_values("absz", ascending=False).head(TOP_AXES_IN_PROMPT)
    )
    lines = [f"ERA: {era} ({lo}-{hi})", "", "DISTINCTIVE FEATURES (z-score vs the 240-year average):"]
    for _, r in feats.iterrows():
        lines.append(f"  - {_human_axis(r['measure'])}: z={r['value_z']:+.2f}")
    lines += ["", "EXEMPLAR PARAGRAPHS FROM THE ERA:"]
    for i, r in exemplars.iterrows():
        lines.append(f"  [{i + 1}] ({int(r['year'])}) {r['text'].strip()}")
    return "\n".join(lines)


def _prompt_hash() -> str:
    payload = json.dumps({"rubric": PORTRAIT_RUBRIC, "version": PORTRAIT_PROMPT_VERSION,
                          "max_tokens": PORTRAIT_MAX_TOKENS}, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _estimate_cost(prompts: list[str]) -> tuple[int, int, float]:
    """Offline cost estimate (~4 chars/token; NOT batch-discounted — portraits use
    the synchronous Messages API). Reuses annotate's intro-pricing rate helper so
    the date-gated price constant lives in exactly one place."""
    in_rate, out_rate = _annotate_rates()
    est_in = sum(len(PORTRAIT_RUBRIC) + len(p) for p in prompts) // 4
    est_out = PORTRAIT_MAX_TOKENS * len(prompts)  # conservative: assume max
    cost = (est_in * in_rate + est_out * out_rate) / 1_000_000
    return est_in, est_out, cost


def generate_portraits(
    run_data: dict | None = None, dry_run: bool = True, yes: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Generate LLM era portraits (PAID). Dry-run costs $0 and prints an estimate;
    the real run refuses without `yes`, refuses above the cost ceiling, and reads
    the ACTUAL cost back from usage into the manifest (never the estimate).

    If no API key is available the real run is skipped and every portrait is
    written marked `not_generated` — the seam philosophy: never fabricate a
    portrait (mirrors the missing-agreement seam).
    """
    out_dir = ERAS_DIR if out_dir is None else out_dir
    if run_data is None:
        run_data = run(out_dir=out_dir)
    df = load_master_frame()
    era_fp = run_data["tables"]["era_fingerprints"]

    prompts, exemplar_ids = {}, {}
    for era in ERA_ORDER:
        ex = select_exemplars(df, era, era_fp)
        prompts[era] = build_portrait_prompt(era, era_fp, ex)
        exemplar_ids[era] = [
            {"doc_name": r.doc_name, "para_idx": int(r.para_idx), "year": int(r.year)}
            for r in ex.itertuples()
        ]

    est_in, est_out, est_cost = _estimate_cost(list(prompts.values()))
    print(f"PORTRAIT ESTIMATE: {len(prompts)} requests, ~{est_in} in / ~{est_out} out "
          f"tokens, est ${est_cost:.4f} (ceiling ${PORTRAIT_COST_CEILING_USD:.2f})")

    if dry_run:
        (out_dir / "portraits_estimate.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "portraits_estimate.json").write_text(json.dumps({
            "n_requests": len(prompts), "model": PORTRAIT_MODEL,
            "estimated_input_tokens": est_in, "estimated_output_tokens": est_out,
            "estimated_cost_usd": round(est_cost, 4),
        }, indent=2) + "\n")
        print("DRY RUN — no network calls, no cost.")
        return {"dry_run": True, "estimated_cost_usd": est_cost, "prompts": prompts}

    if est_cost > PORTRAIT_COST_CEILING_USD:
        raise RuntimeError(
            f"REFUSING TO RUN: estimated ${est_cost:.4f} exceeds the ${PORTRAIT_COST_CEILING_USD:.2f} "
            f"ceiling. Investigate before spending."
        )
    if not yes:
        raise RuntimeError(
            f"REFUSING TO RUN without --yes: this SPENDS MONEY (est ${est_cost:.4f})."
        )

    try:
        import anthropic  # noqa: PLC0415 — only imported on the paid network path
        client = anthropic.Anthropic()  # zero-arg: SDK resolves credentials
    except Exception as exc:  # noqa: BLE001
        logger.warning("no Anthropic client (%s); shipping portraits as not_generated", exc)
        return _write_portraits(out_dir, exemplar_ids, era_fp,
                                texts=None, usage=None)

    texts, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    try:
        for era in ERA_ORDER:
            msg = client.messages.create(
                model=PORTRAIT_MODEL, max_tokens=PORTRAIT_MAX_TOKENS,
                system=PORTRAIT_RUBRIC,
                messages=[{"role": "user", "content": prompts[era]}],
            )
            # Count the billed usage BEFORE parsing the content: a request that was
            # charged for but whose content-extraction raises must still land in the
            # partial manifest's cost ("a manifest can never claim cheaper than
            # actual"). n_succeeded (len(texts)) may exclude such a request — that is
            # honest — but its dollars must not be lost.
            usage["input_tokens"] += int(msg.usage.input_tokens)
            usage["output_tokens"] += int(msg.usage.output_tokens)
            texts[era] = "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as exc:  # noqa: BLE001 — persist provenance for what was paid
        # A mid-loop failure (e.g. request 5 of 9 returns a 500) has ALREADY spent
        # money on the requests that succeeded. Persist a partial manifest carrying
        # the accumulated ACTUAL cost + how many succeeded BEFORE re-raising, so the
        # spend is never silently discarded — the money-path invariant "a manifest
        # can never claim a run was cheaper than it was". No portraits parquet is
        # written (a half-complete frozen artifact would be worse than none).
        _write_partial_manifest(out_dir, usage, n_succeeded=len(texts), error=exc)
        raise
    return _write_portraits(out_dir, exemplar_ids, era_fp, texts, usage)


def _actual_cost(usage: dict) -> float:
    """Actual (non-batch) USD cost from usage token counts. Reuses annotate's
    date-gated rate helper so the price constant lives in exactly one place."""
    in_rate, out_rate = _annotate_rates()
    return (usage["input_tokens"] * in_rate + usage["output_tokens"] * out_rate) / 1_000_000


def _write_manifest(manifest: Manifest) -> None:
    """Serialize a Manifest under `MANIFESTS_DIR` (data/eras/manifests/), NOT via
    `llm_annotations.write_manifest` — that helper hardcodes the frozen
    paid-annotation manifests dir. `MANIFESTS_DIR` is a module constant so tests
    can redirect it to a tmp dir."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / f"{manifest.run_id}.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )


def _write_partial_manifest(
    out_dir: Path, usage: dict, n_succeeded: int, error: BaseException,
) -> None:
    """Persist provenance for a paid run that failed mid-loop.

    The successful requests already cost money; this records their accumulated
    ACTUAL cost and how many landed, so a mid-loop 500 never silently discards the
    spend. No portraits parquet is written — the run did not complete. Uses the
    same date-based `run_id` as a full run, so a successful re-run on the same day
    overwrites this partial record. Conversely, a partial run on the same day as a
    prior success would overwrite that success manifest — recoverable via git, and
    the era_portraits.parquet it described is left untouched, so the artifact and
    its provenance can still be reconciled."""
    from datetime import date  # noqa: PLC0415 — the ONE date-carrying artifact

    run_id = f"era-portraits-{date.today().isoformat()}"
    cost = _actual_cost(usage)
    _write_manifest(Manifest(
        run_id=run_id, model=PORTRAIT_MODEL,
        prompt_version=PORTRAIT_PROMPT_VERSION, prompt_hash=_prompt_hash(),
        date=date.today().isoformat(), batch_id=None, n_requests=n_succeeded,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        cost_usd=round(cost, 6), corpus_fingerprint=corpus_fingerprint(),
        fields=["portrait"], annotation_files=[],
        notes=(
            f"PARTIAL / FAILED RUN — the paid loop raised after {n_succeeded} of "
            f"{len(ERA_ORDER)} requests succeeded ({type(error).__name__}: {error}). "
            f"No era_portraits.parquet was written. cost_usd is the ACTUAL "
            f"accumulated cost of the {n_succeeded} successful request(s), read back "
            f"from usage and recorded so the spend is never silently discarded. "
            f"Re-run to complete; a same-day success overwrites this record."
        ),
    ))
    logger.warning(
        "PARTIAL PORTRAIT RUN: %d/%d requests succeeded before failure; actual "
        "cost $%.6f persisted to the manifest (run_id=%s).",
        n_succeeded, len(ERA_ORDER), cost, run_id,
    )


def _write_portraits(out_dir, exemplar_ids, era_fp, texts, usage) -> dict:
    """Write the portraits parquet + manifest. `texts is None` writes placeholders
    (never a fabricated portrait).

    The placeholder path REFUSES to run if a generated (paid) artifact already
    exists: `portraits --run --yes` without the key sourced would otherwise pass the
    money gates, fail to build a client, and clobber the $-paid parquet with
    `not_generated` rows — while the existing manifest still claimed `generated`.
    The seam is correct only for a FIRST build with no committed artifact."""
    from datetime import date  # noqa: PLC0415 — the ONE date-carrying artifact

    portraits_path = out_dir / PORTRAITS_PATH.name
    if texts is None and portraits_path.exists():
        existing = pd.read_parquet(portraits_path)
        n_generated = int((existing["status"] == "generated").sum())
        if n_generated:
            raise RuntimeError(
                f"REFUSING to overwrite {portraits_path} with placeholders: it "
                f"already holds {n_generated} generated (paid) portrait(s). The "
                f"placeholder seam is only for a first build with no committed "
                f"artifact — a paid artifact must never be silently downgraded to "
                f"not_generated. Source the API key and re-run, or restore the "
                f"artifact from git."
            )

    run_id = f"era-portraits-{date.today().isoformat()}"
    rows = []
    for era in ERA_ORDER:
        top = (era_fp[era_fp["unit"] == era].assign(a=lambda d: d["value_z"].abs())
               .sort_values("a", ascending=False).head(TOP_AXES_IN_PROMPT))
        if texts is None:
            portrait, status = None, "not_generated"
        else:
            portrait = texts[era]
            # An empty completion is NOT a portrait — never label it "generated".
            status = "generated" if portrait else "empty"
        rows.append({
            "era": era, "era_order": ERA_ORDER.index(era),
            "portrait": portrait,
            "status": status,
            "top_axes": json.dumps([
                {"measure": m, "z": round(float(z), 3)}
                for m, z in zip(top["measure"], top["value_z"])
            ]),
            "exemplar_ids": json.dumps(exemplar_ids[era]),
            "model": PORTRAIT_MODEL if texts else None,
            "run_id": run_id if texts else None,
        })
    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / PORTRAITS_PATH.name, index=False)

    if texts is not None:
        cost = _actual_cost(usage)
        _write_manifest(Manifest(
            run_id=run_id, model=PORTRAIT_MODEL,
            prompt_version=PORTRAIT_PROMPT_VERSION, prompt_hash=_prompt_hash(),
            date=date.today().isoformat(), batch_id=None, n_requests=len(ERA_ORDER),
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
            cost_usd=round(cost, 6), corpus_fingerprint=corpus_fingerprint(),
            fields=["portrait"], annotation_files=[PORTRAITS_PATH.name],
            notes=(
                "Synchronous Messages API (not batched), one request per "
                "trends.ERAS era. Portraits are grounded in each era's z-scored "
                "fingerprint plus deterministically-selected exemplar quotes; "
                "the exemplar (doc_name, para_idx, year) triples are stored in "
                "the portraits parquet. cost_usd is ACTUAL, read back from usage."
            ),
        ))
        print(f"PORTRAITS: wrote {len(out)} portraits, ACTUAL cost ${cost:.6f} (run_id={run_id})")
        # Post-hoc tripwire: the money was already spent, but a manifest whose
        # ACTUAL cost blew past the ceiling must fail loudly rather than pass
        # silently (provenance is persisted above first, so nothing is lost).
        if cost > PORTRAIT_COST_CEILING_USD:
            raise RuntimeError(
                f"ACTUAL portrait cost ${cost:.6f} exceeded the "
                f"${PORTRAIT_COST_CEILING_USD:.2f} ceiling — provenance was written "
                f"(run_id={run_id}); investigate before any further run."
            )
    else:
        print(f"PORTRAITS: no client available — wrote {len(out)} placeholders (not_generated)")
    return {"dry_run": False, "portraits": out}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_summary(result: dict) -> None:
    pd.set_option("display.width", 200)
    nn = result["tables"]["nearest_neighbors"]
    print("\n=== nearest neighbor (detrended, era grain) — the headline ===")
    print(nn[(nn["grain"] == "era") & (nn["matrix"] == "detrended")][
        ["unit", "nearest", "cosine", "nearest_is_temporal_neighbor"]
    ].to_string(index=False))
    print("\n=== raw-matrix adjacency sanity (should be mostly True) ===")
    raw = nn[(nn["grain"] == "era") & (nn["matrix"] == "raw")]
    print(f"  raw temporal-neighbor rate: {raw['nearest_is_temporal_neighbor'].mean():.0%}")
    det = nn[(nn["grain"] == "era") & (nn["matrix"] == "detrended")]
    print(f"  detrended temporal-neighbor rate: {det['nearest_is_temporal_neighbor'].mean():.0%}")
    print(f"\nWrote {len(result['tables'])} tables to {ERAS_DIR}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Era atlas (deterministic build + paid portraits).")
    sub = ap.add_subparsers(dest="command")
    ap.add_argument("--quiet", action="store_true", help="write outputs without printing")

    p = sub.add_parser("portraits", help="generate LLM era portraits (PAID)")
    # Mutually exclusive so `--dry-run --run` can never be passed together — the
    # old `dry_run = not args.run` ignored --dry-run entirely, so `--dry-run --run
    # --yes` would have SPENT. Neither flag → dry-run (the safe default).
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="estimate cost, ZERO network calls (default)")
    mode.add_argument("--run", action="store_true", help="make the PAID call")
    p.add_argument("--yes", action="store_true", help="required to spend money")

    args = ap.parse_args()
    if args.command == "portraits":
        generate_portraits(dry_run=not args.run, yes=args.yes)
        return

    result = run()
    if not args.quiet:
        _print_summary(result)


if __name__ == "__main__":
    main()
