"""Combativeness over time: is right now uniquely combative, or does it just
feel that way from inside?

Three separately-scored paragraph flags from the frozen annotation pass —
`party_attack`, `enemy_naming`, `zero_sum` — charted across 240 years.

Two design commitments carry the whole module:

1. **No composite index.** The three flags tell different stories (the 1930s
   are a party_attack spike with almost no foreign enemy naming; the 1940s are
   the mirror image), and any weighted sum of them would hide exactly that.
   They are reported separately, always.

2. **SOTU-only is CO-PRIMARY, not a control.** The corpus admits more informal
   genres over time — annual messages are 55-76% of 19th-century paragraphs but
   21% of the present era's — so a raw all-genre rise is confounded with the
   corpus changing rather than presidents changing. The annual message is the
   one genre present in every era, and it is the only apples-to-apples
   comparison this corpus can offer. Both series ship side by side; neither is
   allowed to be the headline alone.

A third treatment, `genre_standardized`, reweights each era's genre mix to a
fixed reference (the pooled corpus paragraph share by speech type) so that mix
shift cannot drive an era difference. It is a middle path between the two:
broader than SOTU-only, mix-controlled unlike raw. Because no era observes every
genre, the reweight runs over the genres that era actually covers and the output
carries `ref_weight_covered` so a reader can see how much of the reference
distribution each era's estimate actually spans.

Uncertainty is a **speech-clustered** percentile bootstrap: within-speech ICC on
these flags runs as high as 0.17, so resampling paragraphs would understate the
interval badly. Clusters are speeches; the estimator is the ratio
sum(flagged)/sum(paragraphs), recomputed on each resample.

The CI is SAMPLING uncertainty only. Annotator-disagreement bands are read
through an optional seam (`load_agreement_bands`) that returns None when
`data/llm_annotations/agreement_v1.parquet` is absent; every output row records
which components its interval contains. Nothing here fabricates a disagreement
magnitude.

$0 GUARD: this module is pure local compute over frozen parquets. It never
constructs an Anthropic client and never imports `anthropic`. Exemplar quotes
are paragraph *selection*, not generation.

Run as:
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.combat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .corpus import DATA_DIR, PARTY, load
from .llm_annotations import (
    ANNOTATIONS_DIR,
    corpus_fingerprint,
    load_paragraph_annotations,
    load_speech_annotations,
)
from .trends import ERAS

COMBAT_DIR = DATA_DIR / "combat"
COMBATIVENESS_PATH = COMBAT_DIR / "combativeness.parquet"
PEAK_DECADES_PATH = COMBAT_DIR / "peak_decades.parquet"
RATIOS_PATH = COMBAT_DIR / "ratios.parquet"
EXEMPLARS_PATH = COMBAT_DIR / "exemplars.parquet"
ENTITY_CONSISTENCY_PATH = COMBAT_DIR / "entity_consistency.parquet"
LEXICAL_BASELINE_PATH = COMBAT_DIR / "lexical_baseline.parquet"
ADVERSARY_MIX_PATH = COMBAT_DIR / "adversary_mix.parquet"
BY_PRESIDENT_PATH = COMBAT_DIR / "by_president.parquet"
BY_PRESIDENT_TREATMENTS_V2_PATH = COMBAT_DIR / "by_president_treatments_v2.parquet"
BY_PRESIDENT_SPEAKER_V2_PATH = COMBAT_DIR / "by_president_speaker_audited_v2.parquet"
TARGET_MIX_BY_ERA_SPEAKER_V1_PATH = (
    COMBAT_DIR / "target_mix_by_era_speaker_audited_v1.parquet"
)
PRESIDENT_CONFLICT_V2_META_PATH = COMBAT_DIR / "president_conflict_v2_meta.json"
GENRE_DECOMPOSITION_PATH = COMBAT_DIR / "genre_decomposition.parquet"
COMBAT_META_PATH = COMBAT_DIR / "combat_meta.json"

# table name -> published file. Kept beside the constants it maps rather than
# inside `build_combativeness`, so the table set and the file set are declared
# once and cannot drift apart mid-function. Every downstream consumer and the
# report itself cite these files by name, so this mapping is part of the
# published contract.
OUTPUT_PATHS = {
    "combativeness": COMBATIVENESS_PATH,
    "peak_decades": PEAK_DECADES_PATH,
    "ratios": RATIOS_PATH,
    "entity_consistency": ENTITY_CONSISTENCY_PATH,
    "lexical_baseline": LEXICAL_BASELINE_PATH,
    "adversary_mix": ADVERSARY_MIX_PATH,
    "by_president": BY_PRESIDENT_PATH,
    "genre_decomposition": GENRE_DECOMPOSITION_PATH,
    "exemplars": EXEMPLARS_PATH,
}

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
SPEECH_MARKERS_PATH = DATA_DIR / "speech_markers.parquet"
PARAGRAPH_ENTITIES_PATH = ANNOTATIONS_DIR / "paragraph_entities.parquet"

# The optional annotator-disagreement input. Produced by the separate
# `inter-model-agreement-check` task; ABSENT at the time this module was
# written, which is why every consumer of it goes through the optional loader.
AGREEMENT_PATH = ANNOTATIONS_DIR / "agreement_v1.parquet"

FLAGS = ["party_attack", "enemy_naming", "zero_sum"]
SOTU_TYPE = "state_of_the_union_or_annual_message"

TREATMENTS = ["raw", "sotu_only", "genre_standardized"]

# The three peaks the original question named. They are DECADES, while the
# reporting axis is trends.ERAS — and the two disagree in ways that matter:
# the 1850-1877 era band dilutes the 1860s with low-combat Reconstruction, and
# the 1933-1945 band fuses a domestic-attack decade (1930s) with a
# foreign-enemy decade (1940s). So the peaks get their own decade-level table
# rather than being read off the era series.
PEAK_DECADES = [1860, 1930, 2020]

# --------------------------------------------------------------------------
# bootstrap + n-floor policy (pre-registered constants, not tuned to results)
# --------------------------------------------------------------------------

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260721
CI_LEVEL = 0.95

# Speech-clustered bootstrap n-floor. A percentile interval over 3 clusters is
# not an interval, it is three numbers wearing a hat: the War & New Deal SOTU
# cell has exactly 3 annual messages in this corpus. Below MIN_CLUSTERS_FOR_CI
# no interval is emitted at all and the point is marked indicative; between
# that and LOW_CLUSTER_CAUTION an interval is emitted but flagged, because the
# present era's SOTU cell (10 speeches) is the headline and suppressing it
# would be over-correction.
MIN_CLUSTERS_FOR_CI = 5
LOW_CLUSTER_CAUTION = 20

# Genre-standardization cell floor: an (era, genre) cell contributes to the
# reweighted rate only if it holds at least this many speeches. A genre-rate
# estimated off one speech is a speech, not a genre.
#
# This floor governs the POINT ESTIMATE only, and it is deliberately lower than
# MIN_CLUSTERS_FOR_CI. Raising it to 5 would drop whole genres out of an era's
# reweight and collapse `ref_weight_covered` (War & New Deal 0.86 -> 0.22, Civil
# War 0.90 -> 0.58) — trading an interval problem for a far worse
# representativeness problem, since a 22%-coverage "standardized" rate is a
# different and much narrower estimand. The interval is disciplined separately,
# by the two thresholds below.
MIN_CELL_SPEECHES = 3

# The standardized estimator is a weighted average of WITHIN-stratum resamples,
# so its interval is only as trustworthy as its thinnest heavily-weighted
# stratum — not as its total speech count. War & New Deal standardized draws
# 74% of its weight from strata under 5 speeches (49% of it from the very 3
# annual messages that `sotu_only` suppresses), yet reports n_speeches=52.
# Neither `n_speeches` nor `ref_weight_covered` exposes that, so the two floors
# are crossed explicitly here.
#
# Thresholds mirror the existing two-tier policy rather than inventing a new
# scale: refuse the interval when a quarter or more of the estimate rests on
# strata the module already refuses to build an interval from, and caution at
# 0.10 — roughly one whole genre's reference weight (campaign_or_debate is
# 0.104), i.e. the smallest share that can still swing a published number.
THIN_STRATUM_SUPPRESS_WEIGHT = 0.25
THIN_STRATUM_CAUTION_WEIGHT = 0.10

# Exemplar quotability window. A scoring filter would bias the evidence, so
# this is applied as an explicit, disclosed *quotability* filter and recorded
# in the output: sub-20-word paragraphs are fragments and 200+ word 19th-century
# paragraphs cannot be quoted in a report.
EXEMPLAR_MIN_WORDS = 20
EXEMPLAR_MAX_WORDS = 200
# 10 per era per flag, matching the plan's face-validity protocol ("spot-read 10
# per era per flag"). The exemplar table IS the spot-read corpus, so it has to
# be at least as large as the protocol it serves.
EXEMPLARS_PER_CELL = 10

ERA_ORDER = [label for label, _, _ in ERAS]
ERA_BOUNDS = {label: (lo, hi) for label, lo, hi in ERAS}

# The era every published comparison is stated against. Ratios are computed with
# this as the numerator so the headline number carries its own interval.
REFERENCE_ERA = "The present era"


# --------------------------------------------------------------------------
# keyed-merge guard
# --------------------------------------------------------------------------


def _require_full_merge(n_merged: int, inputs: dict[str, int], stage: str) -> None:
    """Raise unless a keyed merge kept every row of every input.

    `validate="one_to_one"` catches DUPLICATE keys, but an inner join silently
    DROPS rows when the two key sets merely diverge (CLAUDE.md). Here that would
    quietly restrict every rate to the intersection of the annotation and corpus
    key sets while the report claimed full-corpus coverage.

    Mirrors `taxonomy._require_full_merge` rather than importing it: `taxonomy`
    is the paid money-path module, and a module whose contract is "zero API
    calls" should not have its import graph depend on that module's client
    construction staying lazy forever.
    """
    if any(n_merged != n for n in inputs.values()):
        breakdown = ", ".join(f"{k}={v}" for k, v in inputs.items())
        raise ValueError(
            f"{stage}: keyed merge changed the row count (merged={n_merged}; "
            f"{breakdown}). The key sets diverge and the inner join dropped "
            f"unmatched rows. Refusing to proceed."
        )


def era_of(year: int) -> str | None:
    """The trends.ERAS band containing `year`. Reused deliberately: a competing
    periodization here would make this task incomparable with everything else in
    the repo, and data-driven periodization belongs to `era-atlas`."""
    for label, lo, hi in ERAS:
        if lo <= year <= hi:
            return label
    return None


# --------------------------------------------------------------------------
# the master frame
# --------------------------------------------------------------------------


def load_frame(
    annotations: pd.DataFrame | None = None,
    speech_annotations: pd.DataFrame | None = None,
    paragraphs: pd.DataFrame | None = None,
    speeches: pd.DataFrame | None = None,
    entities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One paragraph-level frame: the three flags + speech metadata + genre +
    era + per-paragraph adversarial-entity counts.

    Every join is keyed and row-complete-checked. `paragraph_entities` is
    deliberately many-per-key (a paragraph names several entities), so it is
    aggregated to one row per paragraph BEFORE it is joined, and joined as a
    left merge because a paragraph naming no entity is legitimate data, not a
    missing row.

    Every input is injectable as a DataFrame so the whole assembly — including
    the merge guards and the era mapping — can be exercised on small synthetic
    frames without touching the 36k-row corpus (the repo's testing convention;
    see CLAUDE.md). Passing nothing reads the real frozen parquets.
    """
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

    # speech-level metadata x speech-level annotations, one row per speech
    meta = speeches[["doc_name", "president", "party", "date", "year", "title"]].merge(
        speech_anns[["doc_name", "speech_type", "audience", "medium"]],
        on="doc_name",
        validate="one_to_one",
    )
    _require_full_merge(
        len(meta),
        {"speeches": len(speeches), "speech_annotations": len(speech_anns)},
        "speeches x speech_annotations",
    )

    # paragraph text x paragraph flags, on the real (doc_name, para_idx) key
    df = anns[["doc_name", "para_idx", *FLAGS, "run_id"]].merge(
        paras[["doc_name", "para_idx", "text", "word_count"]],
        on=["doc_name", "para_idx"],
        validate="one_to_one",
    )
    _require_full_merge(
        len(df),
        {"paragraph_annotations": len(anns), "paragraphs": len(paras)},
        "paragraph_annotations x paragraphs",
    )

    n_before = len(df)
    df = df.merge(meta, on="doc_name", validate="many_to_one")
    if len(df) != n_before:
        raise ValueError(
            f"paragraphs x speech metadata changed the row count "
            f"({n_before} -> {len(df)}); some paragraph's doc_name is not in the "
            f"speech table."
        )

    df["era"] = df["year"].map(era_of)
    if df["era"].isna().any():
        bad = sorted(df.loc[df["era"].isna(), "year"].unique())
        raise ValueError(f"years outside every trends.ERAS band: {bad}")
    df["era"] = pd.Categorical(df["era"], ERA_ORDER, ordered=True)
    df["decade"] = (df["year"] // 10) * 10
    df["is_sotu"] = df["speech_type"] == SOTU_TYPE

    df = df.merge(
        _adversary_counts(entities=entities), on=["doc_name", "para_idx"], how="left"
    )
    for col in ["n_adversarial", "n_adv_foreign", "n_adv_domestic"]:
        df[col] = df[col].fillna(0).astype(int)

    # Independent cross-check: a paragraph flagged enemy_naming with no
    # adversarial entity is suspect (the corpus-wide consistency rate is
    # reported per era by `entity_consistency`).
    df["enemy_backed"] = df["enemy_naming"] & (df["n_adversarial"] > 0)
    # enemy_naming decomposed by WHO is named. This is the axis the Civil War
    # adjudication turns on: a civil war suppresses the foreign-nation
    # component that dominates the 1810s and 1940s peaks.
    df["enemy_foreign"] = df["enemy_naming"] & (df["n_adv_foreign"] > 0)
    df["enemy_domestic_only"] = (
        df["enemy_naming"] & (df["n_adv_foreign"] == 0) & (df["n_adv_domestic"] > 0)
    )
    return df.reset_index(drop=True)


def _adversarial_entities(
    path: Path | None = None, entities: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The adversarial-stance rows of `paragraph_entities`, one per mention.

    Late-bound default (as `load_agreement_bands`) so patching the module
    constant is not silently inert."""
    if entities is None:
        entities = pd.read_parquet(PARAGRAPH_ENTITIES_PATH if path is None else path)
    return entities[entities["stance"] == "adversarial"]


def _adversary_counts(
    path: Path | None = None, entities: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-paragraph adversarial-entity counts, split foreign vs domestic.

    `type == "nation"` is the foreign adversary; person / group / institution
    are domestic-shaped adversaries (a named opponent, a faction, a branch of
    government). `other` is counted in the total but assigned to neither, so the
    two components never over-claim — the split is deliberately NOT a partition.
    """
    adv = _adversarial_entities(path=path, entities=entities)
    out = (
        adv.assign(
            _foreign=adv["type"].eq("nation"),
            _domestic=adv["type"].isin(["person", "group", "institution"]),
        )
        .groupby(["doc_name", "para_idx"], as_index=False)
        .agg(
            n_adversarial=("entity", "size"),
            n_adv_foreign=("_foreign", "sum"),
            n_adv_domestic=("_domestic", "sum"),
        )
    )
    return out


# --------------------------------------------------------------------------
# optional annotator-disagreement seam
# --------------------------------------------------------------------------

# The columns this module needs from agreement_v1.parquet when it exists. The
# file is produced by another task and does not exist yet, so this contract is
# validated loudly rather than guessed at: silently mis-reading a band would be
# worse than not having one.
AGREEMENT_REQUIRED_COLUMNS = {"era", "flag", "disagreement_half_width"}

# The provenance columns every rate row carries, in their PUBLISHED order.
# Pinned explicitly because the two branches of `_apply_agreement_bands` build
# them in different orders — the band merge appends `disagreement_half_width`
# during the join, while the absent branch assigns `disagreement_band_applied`
# first — so without this the parquet's column order would flip the moment an
# agreement file appeared. The column SET was already stable; this makes the
# sequence stable too, so a positional reader cannot be broken by the seam
# filling in.
BAND_PROVENANCE_COLUMNS = [
    "disagreement_band_applied",
    "disagreement_half_width",
    "ci_components",
    "agreement_source",
]


def _agreement_source(bands: pd.DataFrame | None) -> str:
    """Provenance label for the disagreement seam.

    Takes the bands the caller could actually APPLY, not the bands it was
    handed: bands key on `era`, so a decade-grain table passes None here and
    must report the component as absent rather than naming a file it never
    consulted. Naming the file while `ci_components` says `sampling_only` would
    tell a reader two different stories about the same row.
    """
    return AGREEMENT_PATH.name if bands is not None else f"absent:{AGREEMENT_PATH.name}"


def load_agreement_bands(path: Path | None = None) -> pd.DataFrame | None:
    """Annotator-disagreement half-widths per (era, flag), or None if absent.

    The bands come from `inter-model-agreement-check`, which had not produced
    `agreement_v1.parquet` when this module was written. Absence is a normal,
    expected state and returns None — callers then ship sampling-only intervals
    and record that fact in the output. Absence is NEVER filled with a stub: a
    fabricated disagreement magnitude would be indistinguishable from a measured
    one in the parquet, which is precisely the failure this seam prevents.

    If the file DOES appear but lacks the expected columns, this raises rather
    than guessing at the schema.

    The default is bound HERE, not in the signature: an `AGREEMENT_PATH` default
    in the `def` line is captured at import time, so monkeypatching the module
    constant would be silently inert. `build_combativeness` deliberately does not
    thread a path through (the value feeds three provenance labels, and SIMPLIFY
    declined to spread it), which makes patching the constant the only way to
    redirect this in a test — so it has to actually work.
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
    dupes = bands[bands.duplicated(["era", "flag"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"{path.name}: {len(dupes)} rows share an (era, flag) key; a band "
            f"must identify exactly one row."
        )
    return bands


def _apply_agreement_bands(rows: pd.DataFrame, bands: pd.DataFrame | None) -> pd.DataFrame:
    """Widen sampling intervals by the annotator-disagreement half-width, or
    record that no such component was available."""
    if bands is None:
        out = rows.copy()
        out["disagreement_band_applied"] = False
        out["disagreement_half_width"] = np.nan
        out["ci_components"] = "sampling_only"
    else:
        out = rows.merge(
            bands[["era", "flag", "disagreement_half_width"]],
            on=["era", "flag"],
            how="left",
            validate="many_to_one",
        )
        # Merging an ordered Categorical key against a plain-object column on
        # the other side downgrades it to object. `rate_table` then sorts on
        # `flag` ALPHABETICALLY (enemy_naming first) instead of the declared
        # FLAGS order (party_attack first — the headline flag), silently
        # reordering the published table the moment a band file appears.
        # Restore the declared dtypes rather than casting the band table, so an
        # unrecognised flag in the band table still fails to match loudly
        # instead of becoming a silent NaN.
        for col in ("flag", "treatment"):
            if col in rows.columns:
                out[col] = out[col].astype(rows[col].dtype)

        have = out["disagreement_half_width"].notna()
        old_lo, old_hi = out["ci_lo"].copy(), out["ci_hi"].copy()
        out["ci_lo"] = pd.Series(
            np.where(have, old_lo - out["disagreement_half_width"].fillna(0), old_lo),
            index=out.index,
        ).clip(0, 1)
        out["ci_hi"] = pd.Series(
            np.where(have, old_hi + out["disagreement_half_width"].fillna(0), old_hi),
            index=out.index,
        ).clip(0, 1)
        # "Applied" requires a band AND an interval for it to apply TO. A row
        # suppressed by the n-floor has NaN bounds: the interval is correctly
        # not manufactured, but labelling it "sampling+annotator" would assert a
        # component applied to nothing, and §8.2 promises these widen when the
        # file lands. A band of exactly 0.0 still counts as applied — a measured
        # zero disagreement is a real annotator component, not an absent one.
        applied = have & old_lo.notna() & old_hi.notna()
        out["disagreement_band_applied"] = applied
        out["ci_components"] = np.where(applied, "sampling+annotator", "sampling_only")

    out["agreement_source"] = _agreement_source(bands)
    lead = [c for c in out.columns if c not in BAND_PROVENANCE_COLUMNS]
    return out.reindex(columns=lead + BAND_PROVENANCE_COLUMNS)


def _band_lookup(bands: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    """`(era, flag) -> half_width`, empty when no band table is available.

    A dict rather than a second merge: `ratio_table` needs TWO lookups per row
    (numerator era and denominator era), and chaining two merges would both
    clutter the frame with suffixed columns and re-introduce the Categorical
    downgrade that `_apply_agreement_bands` has to defend against.
    """
    if bands is None:
        return {}
    return {
        (str(row.era), str(row.flag)): float(row.disagreement_half_width)
        for row in bands.itertuples()
        if pd.notna(row.disagreement_half_width)
    }


def _widen_ratio(
    lo: float, hi: float, num_rate: float, den_rate: float,
    h_num: float | None, h_den: float | None,
) -> tuple[float, float]:
    """Widen a bootstrapped ratio interval by the annotator-disagreement bands
    on its numerator and denominator.

    Annotator disagreement is a SYSTEMATIC term on each rate, not a resampling
    term, so it composes with the sampling interval rather than being drawn
    through the bootstrap: each bound is scaled by the factor its band could
    move the point ratio (numerator up / denominator down for the upper bound,
    and the reverse for the lower).

    Multiplicative scaling rather than dividing the two widened marginal
    endpoints: endpoint division would replace the joint bootstrap with a far
    more conservative quotient, so the interval would jump discontinuously the
    moment a band file appeared even with a negligible half-width. Scaling is
    continuous — a zero half-width is exactly the identity.

    A band wide enough to reach zero in the denominator makes the upper bound
    genuinely unbounded, and it is reported as `inf` rather than clipped. A
    suppressed (NaN) interval is returned untouched: widening must never
    manufacture an interval the n-floor refused.
    """
    if h_num is None and h_den is None:
        return lo, hi
    if np.isnan(lo) or np.isnan(hi):
        return lo, hi

    hn, hd = h_num or 0.0, h_den or 0.0

    # A zero rate has no factor to scale — the band cannot move a point estimate
    # that is already at the floor — so both sides fall back to the identity.
    if num_rate > 0:
        f_lo_num = max(num_rate - hn, 0.0) / num_rate
        f_hi_num = (num_rate + hn) / num_rate
    else:
        f_lo_num = f_hi_num = 1.0

    if den_rate > 0:
        f_lo_den = den_rate / (den_rate + hd)
        den_floor = den_rate - hd
        # A band wide enough to reach zero in the denominator makes the upper
        # bound genuinely unbounded.
        f_hi_den = np.inf if den_floor <= 0 else den_rate / den_floor
    else:
        # A denominator already at zero: any non-zero band straddles it, so the
        # ratio's upper bound is unbounded.
        f_lo_den = 1.0
        f_hi_den = np.inf if hd > 0 else 1.0

    new_lo = max(lo * f_lo_num * f_lo_den, 0.0)
    new_hi = np.inf if np.isinf(f_hi_den) else hi * f_hi_num * f_hi_den
    return float(new_lo), float(new_hi)


# --------------------------------------------------------------------------
# speech-clustered bootstrap
# --------------------------------------------------------------------------


def _speech_totals(cell: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Collapse a paragraph frame to per-speech (n_paragraphs, n_flagged) —
    the sufficient statistics for a speech-clustered ratio bootstrap."""
    grouped = cell.groupby("doc_name", observed=True)
    n_para = grouped.size().to_numpy(dtype=np.int64)
    flagged = {f: grouped[f].sum().to_numpy(dtype=np.int64) for f in FLAGS}
    return n_para, flagged


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(N_BOOTSTRAP, n))


def _percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    """Percentile interval, inf-safe.

    Ratio draws can be infinite (a resample can zero out a rare flag in the
    denominator cell). Linear interpolation between two infinite order
    statistics yields `inf - inf = nan`, which would silently disguise an
    unbounded upper limit as a missing one. When any draw is infinite we switch
    to nearest-rank, which always returns an actual order statistic, so an
    unbounded tail reports as `inf` — the honest answer — instead of NaN.
    """
    alpha = (1.0 - CI_LEVEL) / 2.0
    qs = [alpha * 100.0, (1.0 - alpha) * 100.0]
    method = "linear" if np.isfinite(draws).all() else "nearest"
    lo, hi = np.percentile(draws, qs, method=method)
    return float(lo), float(hi)


def _ci_status(n_speeches: int) -> str:
    if n_speeches == 0:
        return "no_data"
    if n_speeches < MIN_CLUSTERS_FOR_CI:
        return "suppressed_n_floor"
    if n_speeches < LOW_CLUSTER_CAUTION:
        return "low_cluster_caution"
    return "ok"


# Worst-first: a ratio is only as trustworthy as its weaker cell.
_STATUS_SEVERITY = ["no_data", "suppressed_n_floor", "low_cluster_caution", "ok"]


def _worse_status(*statuses: str) -> str:
    return min(statuses, key=_STATUS_SEVERITY.index)


def _status_for_fit(fit: _CellFit) -> str:
    """The cell's status, taking the WORSE of the total-cluster rule and the
    thin-stratum rule.

    A standardized cell can clear the total-cluster floor comfortably and still
    rest most of its weight on strata that would individually be suppressed.
    `ci_status` is the machine-readable trust gate that `era-atlas` consumes, so
    it has to reflect whichever constraint actually binds.
    """
    status = _ci_status(fit.n_speeches)
    if status == "no_data":
        return status
    if fit.weight_from_thin_strata >= THIN_STRATUM_SUPPRESS_WEIGHT:
        return "suppressed_n_floor"
    if fit.weight_from_thin_strata >= THIN_STRATUM_CAUTION_WEIGHT:
        return "low_cluster_caution" if status == "ok" else status
    return status


def _cell_rng(*stream: int) -> np.random.Generator:
    """A generator seeded from (BOOTSTRAP_SEED, *stream), so every cell is
    reproducible independently of the order cells happen to be visited in."""
    return np.random.default_rng([BOOTSTRAP_SEED, *stream])


class _CellFit(NamedTuple):
    """Everything one (group, treatment) cell contributes: point rates, the raw
    bootstrap draw vectors, and sample sizes.

    Exposing `draws` rather than just the collapsed interval is what lets the
    RATIO intervals reuse the exact same resamples as the marginal intervals, so
    the two can never tell inconsistent stories. `draws is None` encodes the
    n-floor: a cell below the floor has no interval, and any ratio built from it
    inherits that suppression instead of silently getting one.
    """

    rates: dict[str, float]
    counts: dict[str, int]
    draws: dict[str, np.ndarray] | None
    n_paragraphs: int
    n_speeches: int
    ref_weight_covered: float
    # Effective cluster structure. For a simple cell these degenerate to the
    # cell itself; for a standardized cell they are what `n_speeches` hides.
    effective_min_cluster: int
    weight_from_thin_strata: float

    @classmethod
    def empty(cls, ref_weight_covered: float) -> _CellFit:
        """A cell with nothing in it. `ref_weight_covered` is the one field the
        two empty cases disagree about, so it stays an explicit argument: an
        unobserved `raw`/`sotu_only` cell was never standardized at all (NaN),
        while a `genre_standardized` cell with no qualifying stratum genuinely
        covers 0.0 of the reference mix."""
        return cls(
            rates={f: np.nan for f in FLAGS},
            counts={f: 0 for f in FLAGS},
            draws=None,
            n_paragraphs=0,
            n_speeches=0,
            ref_weight_covered=ref_weight_covered,
            effective_min_cluster=0,
            weight_from_thin_strata=0.0,
        )


def _fit_simple(cell: pd.DataFrame, stream: tuple[int, ...]) -> _CellFit:
    """Unweighted ratio estimator + speech-clustered draws (`raw`, `sotu_only`)."""
    n_speeches = cell["doc_name"].nunique()
    if n_speeches == 0:
        return _CellFit.empty(ref_weight_covered=np.nan)

    n_para, flagged = _speech_totals(cell)
    total_para = int(n_para.sum())
    counts = {f: int(flagged[f].sum()) for f in FLAGS}
    rates = {f: counts[f] / total_para for f in FLAGS}

    draws = None
    if n_speeches >= MIN_CLUSTERS_FOR_CI:
        # One index draw shared across the three flags: the flags are measured
        # on the same speeches, so resampling them jointly is the correct design.
        idx = _bootstrap_indices(len(n_para), _cell_rng(*stream))
        denom = n_para[idx].sum(axis=1)
        draws = {f: flagged[f][idx].sum(axis=1) / denom for f in FLAGS}

    # An unstratified cell IS one cluster pool, so there is no thin-stratum
    # weight for `n_speeches` to hide.
    return _CellFit(
        rates=rates,
        counts=counts,
        draws=draws,
        n_paragraphs=total_para,
        n_speeches=n_speeches,
        ref_weight_covered=np.nan,
        effective_min_cluster=n_speeches,
        weight_from_thin_strata=0.0,
    )


class _GenreStratum(NamedTuple):
    """One (group, genre) sub-cell that qualified for the standardized estimate:
    its reference weight and its per-speech sufficient statistics."""

    genre: str
    weight: float
    n_para: np.ndarray
    flagged: dict[str, np.ndarray]

    def rate(self, flag: str) -> float:
        return float(self.flagged[flag].sum()) / float(self.n_para.sum())


def _fit_standardized(
    cell: pd.DataFrame, reference: pd.Series, stream: tuple[int, ...]
) -> _CellFit:
    """Genre-reweighted rates + stratified draws.

    Only (group, genre) sub-cells with at least MIN_CELL_SPEECHES speeches
    contribute; the reference weights are renormalized over those, and
    `ref_weight_covered` records what share of the reference distribution the
    surviving genres carry. That number is the honesty valve — an era at 0.58
    coverage is standardized over a little more than half the reference mix and
    the reader can see it.

    The bootstrap resamples speeches WITHIN each genre (a stratified cluster
    bootstrap), because the genre mix is fixed by the reference here rather than
    estimated: resampling across genres would put design variance back into an
    estimator whose whole point is to remove it.
    """
    strata: list[_GenreStratum] = []
    for genre, part in cell.groupby("speech_type", observed=True):
        if genre not in reference.index:
            continue
        if part["doc_name"].nunique() < MIN_CELL_SPEECHES:
            continue
        n_para, flagged = _speech_totals(part)
        strata.append(
            _GenreStratum(
                genre=genre, weight=float(reference[genre]), n_para=n_para, flagged=flagged
            )
        )

    if not strata:
        return _CellFit.empty(ref_weight_covered=0.0)

    ref_weights = np.array([s.weight for s in strata], dtype=float)
    ref_covered = float(ref_weights.sum())
    weights = ref_weights / ref_covered

    n_speeches = int(sum(len(s.n_para) for s in strata))
    total_para = int(sum(int(s.n_para.sum()) for s in strata))

    # The two floors, crossed. `n_speeches` is a total over strata, but the
    # estimator averages WITHIN-stratum resamples, so the interval's real
    # cluster support is the thinnest heavily-weighted stratum.
    effective_min_cluster = int(min(len(s.n_para) for s in strata))
    thin_weight = float(
        sum(w for w, s in zip(weights, strata) if len(s.n_para) < MIN_CLUSTERS_FOR_CI)
    )

    rates, counts = {}, {}
    for f in FLAGS:
        genre_rates = np.array([s.rate(f) for s in strata])
        rates[f] = float((weights * genre_rates).sum())
        counts[f] = int(sum(int(s.flagged[f].sum()) for s in strata))

    draws = None
    if (
        n_speeches >= MIN_CLUSTERS_FOR_CI
        and thin_weight < THIN_STRATUM_SUPPRESS_WEIGHT
    ):
        rng = _cell_rng(*stream)
        idx_per_genre = [_bootstrap_indices(len(s.n_para), rng) for s in strata]
        draws = {}
        for f in FLAGS:
            acc = np.zeros(N_BOOTSTRAP)
            for w, s, idx in zip(weights, strata, idx_per_genre):
                acc += w * (s.flagged[f][idx].sum(axis=1) / s.n_para[idx].sum(axis=1))
            draws[f] = acc

    return _CellFit(
        rates=rates,
        counts=counts,
        draws=draws,
        n_paragraphs=total_para,
        n_speeches=n_speeches,
        ref_weight_covered=ref_covered,
        effective_min_cluster=effective_min_cluster,
        weight_from_thin_strata=thin_weight,
    )


def _rows_from_fit(fit: _CellFit, group: str, treatment: str) -> list[dict]:
    status = _status_for_fit(fit)
    rows = []
    for f in FLAGS:
        if fit.draws is None:
            lo = hi = np.nan
        else:
            lo, hi = _percentile_ci(fit.draws[f])
        rows.append({
            "group": group,
            "flag": f,
            "treatment": treatment,
            "rate": fit.rates[f],
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_status": status,
            "n_paragraphs": fit.n_paragraphs,
            "n_flagged": fit.counts[f],
            "n_speeches": fit.n_speeches,
            "ref_weight_covered": fit.ref_weight_covered,
            "effective_min_cluster": fit.effective_min_cluster,
            "weight_from_thin_strata": fit.weight_from_thin_strata,
        })
    return rows


def _fit_all(df: pd.DataFrame, by: str, order: list) -> dict[tuple, _CellFit]:
    """Fit every (group, treatment) cell once. Keyed so both the rate table and
    the ratio table read the same fits — and therefore the same resamples."""
    reference = genre_reference(df)
    fits: dict[tuple, _CellFit] = {}
    for gi, group in enumerate(order):
        cell = df[df[by] == group]
        fits[(group, "raw")] = _fit_simple(cell, (gi, 0))
        fits[(group, "sotu_only")] = _fit_simple(cell[cell["is_sotu"]], (gi, 1))
        fits[(group, "genre_standardized")] = _fit_standardized(cell, reference, (gi, 2))
    return fits


def genre_reference(df: pd.DataFrame) -> pd.Series:
    """The fixed reference genre distribution: pooled corpus paragraph share by
    speech type. Chosen because it is the corpus's own centre of gravity — no
    era is privileged as "the normal mix"."""
    return df["speech_type"].value_counts(normalize=True)


def rate_table(
    df: pd.DataFrame,
    by: str,
    order: list,
    bands: pd.DataFrame | None,
    fits: dict[tuple, _CellFit] | None = None,
) -> pd.DataFrame:
    """era (or decade) x flag x treatment x rate x CI x n, for all three
    genre treatments."""
    if fits is None:
        fits = _fit_all(df, by, order)
    rows: list[dict] = []
    for group in order:
        for treatment in TREATMENTS:
            rows += _rows_from_fit(fits[(group, treatment)], group, treatment)

    out = pd.DataFrame(rows).rename(columns={"group": by})
    if by == "era":
        out["era_start"] = out["era"].map(lambda e: ERA_BOUNDS[e][0])
        out["era_end"] = out["era"].map(lambda e: ERA_BOUNDS[e][1])
    out[f"{by}_order"] = out[by].map({g: i for i, g in enumerate(order)})
    out["flag"] = pd.Categorical(out["flag"], FLAGS, ordered=True)
    out["treatment"] = pd.Categorical(out["treatment"], TREATMENTS, ordered=True)

    # The bands seam keys on `era`; a decade table has no era column, so it
    # carries the same provenance fields with nothing applied.
    out = _apply_agreement_bands(out, bands if by == "era" else None)

    out["n_bootstrap"] = N_BOOTSTRAP
    out["bootstrap_seed"] = BOOTSTRAP_SEED
    out["ci_level"] = CI_LEVEL
    out["min_clusters_for_ci"] = MIN_CLUSTERS_FOR_CI
    return out.sort_values([f"{by}_order", "flag", "treatment"]).reset_index(drop=True)


def ratio_table(
    df: pd.DataFrame,
    by: str = "era",
    order: list | None = None,
    reference_group: str = REFERENCE_ERA,
    fits: dict[tuple, _CellFit] | None = None,
    bands: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Interval for the *ratio* between the reference era and every other era.

    Why this exists: "the present era attacks the opposing party at 4.8x the
    Civil War rate" is the sentence that gets quoted, and a bare point ratio
    implies a precision the data does not have. Non-overlapping marginal
    intervals establish "substantially higher"; they do NOT establish 4.8 to two
    significant figures. Dividing the CI endpoints is worse — it is far too
    conservative, because it pairs the numerator's worst case with the
    denominator's best case as if they were jointly attainable.

    So the ratio is bootstrapped directly, reusing the SAME speech resamples
    that produced the marginal intervals (the two cells hold disjoint speeches,
    so their draws are independent by construction). If either cell is below the
    CI n-floor the ratio interval is suppressed, exactly as the marginal is.

    A resample can make the denominator zero for a rare flag in a small cell, in
    which case the ratio is infinite and the upper bound is genuinely unbounded.
    That is reported (`ci_hi = inf`, with `pct_denominator_zero`) rather than
    hidden by dropping those replicates, which would silently truncate the tail.

    Annotator-disagreement bands are propagated here too, on the SAME terms as
    the marginal table: this is the number the report actually quotes, so a seam
    that widened `combativeness.parquet` while leaving `ratios.parquet` silently
    unchanged would defeat the point of deferring the criterion. Every row
    records which uncertainty components its interval contains.
    """
    if order is None:
        order = ERA_ORDER
    if fits is None:
        fits = _fit_all(df, by, order)
    if reference_group not in order:
        raise ValueError(f"reference_group {reference_group!r} not in {order}")
    # Bands key on `era`; any other grain carries the provenance fields with
    # nothing applied rather than joining on the wrong axis (as `rate_table`).
    # Gated ONCE and reused for both the lookup and the provenance label, so a
    # decade table cannot name the band file while reporting `sampling_only`.
    applicable_bands = bands if by == "era" else None
    lookup = _band_lookup(applicable_bands)
    source = _agreement_source(applicable_bands)

    rows = []
    for treatment in TREATMENTS:
        num = fits[(reference_group, treatment)]
        for group in order:
            if group == reference_group:
                continue
            den = fits[(group, treatment)]
            for f in FLAGS:
                ratio = (
                    np.nan if not den.rates[f] else num.rates[f] / den.rates[f]
                )
                suppressed = num.draws is None or den.draws is None
                if suppressed:
                    lo = hi = np.nan
                    pct_zero = np.nan
                else:
                    d = den.draws[f]
                    pct_zero = float((d == 0).mean())
                    with np.errstate(divide="ignore", invalid="ignore"):
                        draws = np.where(d > 0, num.draws[f] / np.where(d > 0, d, 1), np.inf)
                    lo, hi = _percentile_ci(draws)

                h_num = lookup.get((reference_group, f))
                h_den = lookup.get((group, f))
                pre_lo, pre_hi = lo, hi
                lo, hi = _widen_ratio(lo, hi, num.rates[f], den.rates[f], h_num, h_den)
                # As in `_apply_agreement_bands`: a band needs an interval to
                # apply TO. A suppressed row keeps its NaN bounds and must not
                # claim an annotator component it never received.
                banded = bool(
                    (h_num is not None or h_den is not None)
                    and not np.isnan(pre_lo)
                    and not np.isnan(pre_hi)
                )

                rows.append({
                    "numerator": reference_group,
                    "denominator": group,
                    "flag": f,
                    "treatment": treatment,
                    "ratio": ratio,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    # The worse of the two cells' statuses. `no_data` is
                    # preserved rather than collapsed into `suppressed_n_floor`:
                    # "we have no speeches here" and "we have too few to build
                    # an interval" are different facts about the corpus.
                    "ci_status": _worse_status(
                        _status_for_fit(num), _status_for_fit(den)
                    ),
                    "pct_denominator_zero": pct_zero,
                    "numerator_rate": num.rates[f],
                    "denominator_rate": den.rates[f],
                    "n_speeches_numerator": num.n_speeches,
                    "n_speeches_denominator": den.n_speeches,
                    "disagreement_band_applied": banded,
                    "disagreement_half_width_numerator": (
                        np.nan if h_num is None else h_num
                    ),
                    "disagreement_half_width_denominator": (
                        np.nan if h_den is None else h_den
                    ),
                    "ci_components": (
                        "sampling+annotator" if banded else "sampling_only"
                    ),
                    "agreement_source": source,
                })

    out = pd.DataFrame(rows)
    out["flag"] = pd.Categorical(out["flag"], FLAGS, ordered=True)
    out["treatment"] = pd.Categorical(out["treatment"], TREATMENTS, ordered=True)
    out["denominator_order"] = out["denominator"].map({g: i for i, g in enumerate(order)})
    out["n_bootstrap"] = N_BOOTSTRAP
    out["bootstrap_seed"] = BOOTSTRAP_SEED
    out["ci_level"] = CI_LEVEL
    return out.sort_values(
        ["treatment", "flag", "denominator_order"]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# cross-checks
# --------------------------------------------------------------------------


def entity_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Per-era enemy-naming <-> adversarial-entity consistency.

    Corpus-wide this is 96.84% (notes/annotation-qa-v1.md); the per-era cut is
    what matters here, because an era where the two independent signals come
    apart is an era whose combativeness numbers cannot be trusted. Also carries
    the foreign/domestic split of who gets named, which is the evidence the
    Civil War adjudication rests on.
    """
    flagged = df[df["enemy_naming"]]
    out = (
        flagged.groupby("era", observed=True)
        .agg(
            n_enemy_naming=("enemy_naming", "size"),
            n_backed=("enemy_backed", "sum"),
            share_foreign_adversary=("enemy_foreign", "mean"),
            share_domestic_only=("enemy_domestic_only", "mean"),
        )
        .reindex(ERA_ORDER)
        .reset_index()
    )
    out["consistency_rate"] = out["n_backed"] / out["n_enemy_naming"]
    out["era_order"] = out["era"].map({e: i for i, e in enumerate(ERA_ORDER)})
    return out


def adversary_mix(df: pd.DataFrame, entities: pd.DataFrame | None = None,
                  path: Path | None = None) -> pd.DataFrame:
    """Per-era decomposition of enemy_naming into foreign-adversary and
    domestic-only components, plus the adversarial-entity TYPE shares, at both
    era and decade grain.

    This is the table that adjudicates the Civil War bail condition: the era's
    headline enemy_naming rate is not elevated, but its DOMESTIC component is
    the highest of the 19th century while its foreign component collapses —
    exactly what a civil war fought against people the President refused to
    recognize as a foreign sovereign should look like.

    The `adv_share_*` columns carry the entity-type mix the report's §5.2 turns
    on. They ship as code rather than as a hand query because §5.2 claims the
    mix "confirms it mechanically" — a claim a future maintainer must be able to
    re-derive from the artifact rather than re-run by hand.

    NOTE the two denominators, which are easy to confuse and are NOT the same
    quantity as `entity_consistency.share_foreign_adversary`:
      * `enemy_foreign` / `enemy_domestic_only` — share of ALL paragraphs.
      * `adv_share_<type>` — share of adversarial ENTITY MENTIONS.
      * `entity_consistency.share_foreign_adversary` — share of the
        enemy_naming-flagged PARAGRAPHS.
    """
    adv = _adversarial_entities(entities=entities, path=path)
    # NOTE: deliberately NOT guarded by `_require_full_merge`, unlike the joins
    # in `load_frame`. There the two sides must describe the same paragraphs and
    # a dropped row is a silent bug; here an inner join onto `df` IS the
    # semantics — the caller may pass a subset (a single era, a test frame) and
    # entities outside it should fall away. On the full corpus nothing drops
    # (9,447 adversarial mentions in, 9,447 out). Same reasoning applies to
    # `by_president` and `mistyped_nation_recount`.
    adv = adv.merge(
        df.drop_duplicates("doc_name")[["doc_name", "era", "decade"]],
        on="doc_name",
        validate="many_to_one",
    )

    def type_shares(key: str) -> pd.DataFrame:
        share = (
            pd.crosstab(adv[key], adv["type"], normalize="index")
            .add_prefix("adv_share_")
            .reset_index()
            .rename(columns={key: "group"})
        )
        share["n_adversarial_entities"] = (
            adv.groupby(key, observed=True).size().reindex(share["group"]).to_numpy()
        )
        share["group"] = share["group"].astype(str)
        return share

    def at_grain(key: str) -> pd.DataFrame:
        out = (
            df.groupby(key, observed=True)
            .agg(
                enemy_naming=("enemy_naming", "mean"),
                enemy_foreign=("enemy_foreign", "mean"),
                enemy_domestic_only=("enemy_domestic_only", "mean"),
                party_attack=("party_attack", "mean"),
                zero_sum=("zero_sum", "mean"),
                n_paragraphs=("enemy_naming", "size"),
                n_speeches=("doc_name", "nunique"),
            )
            .reset_index()
            .rename(columns={key: "group"})
        )
        # One `group` column has to hold both era labels and decade integers, so
        # both grains are stringified before they are stacked.
        out["group"] = out["group"].astype(str)
        out = out.merge(type_shares(key), on="group", how="left", validate="one_to_one")
        # A group with paragraphs but NO adversarial entity (decade 1780: 14
        # paragraphs, 2 speeches, zero adversaries) is absent from the entity
        # side, so the left join leaves NaN. The `adv_share_*` NaNs are correct —
        # a share of nothing is undefined — but the COUNT is a true 0, and a NaN
        # there poisons any downstream sum. Same bug class as the "reindex
        # FIRST, label SECOND" note in `lexical_baseline`.
        out["n_adversarial_entities"] = (
            out["n_adversarial_entities"].fillna(0).astype(int)
        )
        out["grain"] = key
        return out

    return pd.concat([at_grain("era"), at_grain("decade")], ignore_index=True)


# US states and the Confederacy that `paragraph_entities` types as `nation`.
# Defensible in context (they claimed sovereignty) but it means the
# foreign/domestic split counts them as foreign. Listed here so §5.7's
# robustness recount is reproducible instead of a hand tally; the direction is
# conservative, since reclassifying them lowers the Civil War foreign share.
#
# SCOPE: hand-curated by reading the Civil War era's adversarial `nation`
# entities. `mistyped_nation_recount(era=...)` accepts any era, but this list
# does not generalise — a later era's domestic-entity mistypes (a state, a
# territory, a city) are simply absent from it, so a recount on another era
# UNDER-counts silently. Re-derive the list before trusting it elsewhere.
DOMESTIC_ENTITIES_TYPED_AS_NATION = frozenset({
    "South Carolina", "Virginia", "Texas", "Georgia", "Arkansas",
    "North Carolina", "Tennessee", "Wisconsin", "Confederate States",
})


def mistyped_nation_recount(
    df: pd.DataFrame, era: str = "Civil War & Reconstruction",
    entities: pd.DataFrame | None = None, path: Path | None = None,
) -> dict:
    """How much of an era's adversarial `nation` mass is really domestic.

    §5.7 leans on this as its robustness note, so it is computed rather than
    counted by hand."""
    adv = _adversarial_entities(entities=entities, path=path).merge(
        df.drop_duplicates("doc_name")[["doc_name", "era"]],
        on="doc_name", validate="many_to_one",
    )
    adv = adv[adv["era"] == era]
    nations = adv[adv["type"] == "nation"]
    mistyped = nations["entity"].isin(DOMESTIC_ENTITIES_TYPED_AS_NATION)
    return {
        "era": era,
        "n_adversarial": int(len(adv)),
        "n_typed_nation": int(len(nations)),
        "n_domestic_mistyped": int(mistyped.sum()),
        "share_of_nation_mistyped": float(mistyped.mean()) if len(nations) else float("nan"),
        "foreign_share_as_typed": float(len(nations) / len(adv)) if len(adv) else float("nan"),
        "foreign_share_reclassified": (
            float((len(nations) - mistyped.sum()) / len(adv)) if len(adv) else float("nan")
        ),
    }


def by_president(df: pd.DataFrame, entities: pd.DataFrame | None = None,
                 path: Path | None = None) -> pd.DataFrame:
    """Per-president flag rates (all-genre and SOTU-only) plus adversarial
    entity-type counts.

    Ships the report's §4 ("is it just one president?") and §5.3 (Lincoln's
    adversarial entities are 115 group / 89 person / 12 nation) as derivable
    artifact rather than hand analysis. §4 is load-bearing: it is the answer to
    the first objection any reader raises about a 2-president era.
    """
    sotu = df[df["is_sotu"]]
    out = (
        df.groupby("president", observed=True)
        .agg(
            era_first=("era", "first"),
            year_first=("year", "min"),
            year_last=("year", "max"),
            n_speeches=("doc_name", "nunique"),
            n_paragraphs=("para_idx", "size"),
            **{f: (f, "mean") for f in FLAGS},
        )
        .reset_index()
    )
    s = (
        sotu.groupby("president", observed=True)
        .agg(
            n_sotu_speeches=("doc_name", "nunique"),
            n_sotu_paragraphs=("para_idx", "size"),
            **{f"sotu_{f}": (f, "mean") for f in FLAGS},
        )
        .reset_index()
    )
    out = out.merge(s, on="president", how="left", validate="one_to_one")
    out[["n_sotu_speeches", "n_sotu_paragraphs"]] = (
        out[["n_sotu_speeches", "n_sotu_paragraphs"]].fillna(0).astype(int)
    )

    adv = _adversarial_entities(entities=entities, path=path).merge(
        df.drop_duplicates("doc_name")[["doc_name", "president"]],
        on="doc_name",
        validate="many_to_one",
    )
    counts = (
        pd.crosstab(adv["president"], adv["type"])
        .add_prefix("adv_n_")
        .reset_index()
    )
    out = out.merge(counts, on="president", how="left", validate="one_to_one")
    for col in [c for c in out.columns if c.startswith("adv_n_")]:
        out[col] = out[col].fillna(0).astype(int)
    return out.sort_values("year_first").reset_index(drop=True)


def speaker_audited_target_mix_by_era(
    paragraph_frame: pd.DataFrame,
    adversarial_entities: pd.DataFrame,
    *,
    metadata: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pool audited adversarial-entity mentions within ``trends.ERAS``.

    The denominator is the number of entity mentions, not paragraphs, and the
    period estimator is a ratio of pooled counts.  It therefore never averages
    annual or president percentages.  Calendar year comes from the canonical
    source speech joined onto ``paragraph_frame`` before this function runs;
    attributed-speaker tenure and document ownership do not define the era.
    """
    paragraph_required = {
        "doc_name", "para_idx", "year", "speaker_audited_all",
    }
    entity_required = {"doc_name", "para_idx", "type"}
    missing_paragraph = paragraph_required - set(paragraph_frame.columns)
    missing_entity = entity_required - set(adversarial_entities.columns)
    if missing_paragraph:
        raise ValueError(
            "target-mix paragraphs are missing columns: "
            + ", ".join(sorted(missing_paragraph))
        )
    if missing_entity:
        raise ValueError(
            "target-mix entities are missing columns: "
            + ", ".join(sorted(missing_entity))
        )
    keys = ["doc_name", "para_idx"]
    if paragraph_frame.duplicated(keys).any():
        raise ValueError("target-mix paragraphs need unique paragraph keys")
    if paragraph_frame[list(paragraph_required)].isna().any().any():
        raise ValueError("target-mix paragraphs contain null required values")
    if adversarial_entities[list(entity_required)].isna().any().any():
        raise ValueError("target-mix entities contain null required values")

    allowed_types = {"nation", "group", "person", "institution", "other"}
    unknown_types = set(adversarial_entities["type"].astype(str)) - allowed_types
    if unknown_types:
        raise ValueError(
            "target-mix entities contain unknown types: "
            + ", ".join(sorted(unknown_types))
        )
    entity_key_check = adversarial_entities[keys].drop_duplicates().merge(
        paragraph_frame[keys], on=keys, how="left", validate="one_to_one",
        indicator=True,
    )
    if entity_key_check["_merge"].ne("both").any():
        raise ValueError("target-mix entities contain keys outside the paragraph frame")

    eligible = paragraph_frame.loc[paragraph_frame["speaker_audited_all"]].copy()
    eligible["era"] = eligible["year"].map(era_of)
    if eligible["era"].isna().any():
        bad_years = sorted(eligible.loc[eligible["era"].isna(), "year"].unique())
        raise ValueError(
            "target-mix paragraphs contain years outside trends.ERAS: "
            + ", ".join(map(str, bad_years))
        )
    entity_credit = adversarial_entities.merge(
        eligible[keys + ["year", "era"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    rows = pd.DataFrame(
        [
            {
                "era": label,
                "era_order": order,
                "era_start": start,
                "era_end": end,
                "n_calendar_years": end - start + 1,
            }
            for order, (label, start, end) in enumerate(ERAS)
        ]
    )
    paragraph_support = (
        eligible.groupby("era", observed=True)
        .agg(
            n_speeches=("doc_name", "nunique"),
            n_paragraphs=("para_idx", "size"),
            n_eligible_years=("year", "nunique"),
        )
        .reindex(ERA_ORDER, fill_value=0)
    )
    entity_support = (
        entity_credit.groupby("era", observed=True)
        .agg(
            n_adversarial_speeches=("doc_name", "nunique"),
            n_active_years=("year", "nunique"),
        )
        .reindex(ERA_ORDER, fill_value=0)
    )
    counts = (
        pd.crosstab(entity_credit["era"], entity_credit["type"])
        .reindex(index=ERA_ORDER, columns=sorted(allowed_types), fill_value=0)
        .add_prefix("adv_n_")
    )
    for support in (paragraph_support, entity_support, counts):
        support.index.name = "era"
        rows = rows.merge(
            support.reset_index(), on="era", how="left", validate="one_to_one"
        )
    count_columns = [f"adv_n_{entity_type}" for entity_type in sorted(allowed_types)]
    integral_columns = [
        "era_order", "era_start", "era_end", "n_calendar_years",
        "n_speeches", "n_paragraphs", "n_eligible_years",
        "n_adversarial_speeches", "n_active_years", *count_columns,
    ]
    rows[integral_columns] = rows[integral_columns].fillna(0).astype(int)
    rows["n_adversarial_entities"] = rows[count_columns].sum(axis=1)
    if int(rows["n_adversarial_entities"].sum()) != len(entity_credit):
        raise ValueError("target-mix category counts do not reconcile to entity rows")
    if (rows["n_adversarial_speeches"] > rows["n_speeches"]).any():
        raise ValueError("target-mix adversarial speech support exceeds eligible support")
    if (rows["n_active_years"] > rows["n_eligible_years"]).any():
        raise ValueError("target-mix active-year support exceeds eligible years")
    if (rows["n_eligible_years"] > rows["n_calendar_years"]).any():
        raise ValueError("target-mix eligible years exceed era bounds")

    defaults = {
        "schema_version": "conflict-target-mix-v1",
        "treatment": "speaker_audited_all",
        "treatment_label": "Speaker-audited all eligible paragraphs",
        "speaker_scope_status": "speaker_audit_complete",
        "source_label": "combat/target_mix_by_era_speaker_audited_v1.parquet",
        "era_scheme": "trends.ERAS",
        "year_basis": "canonical_source_speech_year",
    }
    supplied = metadata or {}
    unknown_metadata = set(supplied) - set(defaults)
    if unknown_metadata:
        raise ValueError(
            "target-mix metadata contains unknown fields: "
            + ", ".join(sorted(unknown_metadata))
        )
    for column, value in {**defaults, **supplied}.items():
        rows[column] = value
    return rows


def build_president_conflict_v2(
    attribution_path: Path | None = None,
    attribution_meta_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the governed five-treatment president-conflict-v2 contract."""
    from . import annotation_ledger as annotation_ledger_module
    from .speaker_attribution import ROOT as SPEAKER_ROOT

    attribution_path = attribution_path or SPEAKER_ROOT / "paragraph_attribution_v1.parquet"
    attribution_meta_path = attribution_meta_path or SPEAKER_ROOT / "meta_v1.json"
    out_dir = out_dir or COMBAT_DIR
    attr = pd.read_parquet(attribution_path)
    if len(attr) != 35394 or attr.duplicated(["doc_name", "para_idx"]).any():
        raise ValueError("speaker attribution must contain 35,394 unique paragraph keys")
    pointer = annotation_ledger_module.MATERIALIZED_ROOT / "current"
    generation = pointer.read_text(encoding="utf-8").strip()
    labels_path = annotation_ledger_module.MATERIALIZED_ROOT / "generations" / generation / "current_labels.parquet"
    labels = pd.read_parquet(labels_path)
    flag_parts = []
    for flag in FLAGS:
        part = labels.loc[labels["label_type"].eq(flag), ["canonical_doc_name", "canonical_para_idx", "raw_value_json"]].copy()
        part = part.rename(columns={"canonical_doc_name": "doc_name", "canonical_para_idx": "para_idx", "raw_value_json": flag})
        part["para_idx"] = part["para_idx"].astype(int)
        part[flag] = part[flag].map(json.loads).astype(bool)
        if len(part) != 35394 or part.duplicated(["doc_name", "para_idx"]).any():
            raise ValueError(f"canonical {flag} projection is not key complete")
        flag_parts.append(part)
    flags = flag_parts[0]
    for part in flag_parts[1:]:
        flags = flags.merge(part, on=["doc_name", "para_idx"], validate="one_to_one")
    if set(map(tuple, flags[["doc_name", "para_idx"]].to_numpy())) != set(map(tuple, attr[["doc_name", "para_idx"]].to_numpy())):
        raise ValueError("speaker and combat canonical key sets differ")
    speeches = pd.read_parquet(
        DATA_DIR / "corpus_corrections" / "canonical_speeches_v1.parquet"
    )
    base = attr.merge(flags, on=["doc_name", "para_idx"], validate="one_to_one")
    base = base.merge(speeches[["doc_name", "year"]], on="doc_name", validate="many_to_one")

    entity_rows = labels.loc[
        labels["label_type"].eq("entities") & labels["event_role"].eq("value"),
        ["canonical_doc_name", "canonical_para_idx", "raw_value_json"],
    ].copy()
    entity_rows = entity_rows.rename(columns={"canonical_doc_name": "doc_name", "canonical_para_idx": "para_idx"})
    parsed = entity_rows["raw_value_json"].map(json.loads)
    entity_rows["stance"] = parsed.map(lambda value: value["stance"])
    entity_rows["type"] = parsed.map(lambda value: value["type"])
    entity_rows = entity_rows.loc[entity_rows["stance"].eq("adversarial")]

    president_order = list(PARTY)
    president_order.remove("Donald Trump")
    president_order.append("Donald Trump")
    treatments = [
        "canonical_document_owner", "speaker_audited_all", "debate_excluded",
        "single_president_documents", "annual_message_strict",
    ]
    attr_meta = json.loads(Path(attribution_meta_path).read_text(encoding="utf-8"))
    rows = []
    for treatment in treatments:
        included = base.loc[base[treatment]].copy()
        included["credited_president"] = (
            included["document_owner"] if treatment == "canonical_document_owner"
            else included["attributed_speaker"]
        )
        if included["credited_president"].isna().any():
            raise ValueError(f"{treatment} includes rows without a credited president")
        entity_credit = entity_rows.merge(
            included[["doc_name", "para_idx", "credited_president"]],
            on=["doc_name", "para_idx"], validate="many_to_one",
        )
        entity_counts = entity_credit.groupby(["credited_president", "type"]).size().unstack(fill_value=0)
        for display_order, president in enumerate(president_order):
            cell = included.loc[included["credited_president"].eq(president)]
            n_paragraphs = int(len(cell))
            n_speeches = int(cell["doc_name"].nunique())
            years = speeches.loc[speeches["president"].eq(president), "year"]
            if treatment == "canonical_document_owner":
                eligible_pool = base.loc[base["document_owner"].eq(president)]
            else:
                eligible_pool = base.loc[base["attributed_speaker"].eq(president)]
            year_first = int(cell["year"].min()) if n_paragraphs else int(years.min())
            year_last = int(cell["year"].max()) if n_paragraphs else int(years.max())
            out = {
                "president": president,
                "display_order": display_order,
                "era_first": era_of(year_first),
                "year_first": year_first,
                "year_last": year_last,
                "n_speeches": n_speeches,
                "n_paragraphs": n_paragraphs,
                "n_documents_included": n_speeches,
                "n_documents_excluded": int(eligible_pool["doc_name"].nunique() - n_speeches),
                "n_paragraphs_excluded": int(len(eligible_pool) - n_paragraphs),
                "schema_version": "president-conflict-v2",
                "treatment": treatment,
                "treatment_label": {
                    "canonical_document_owner": "Canonical document owner comparison",
                    "speaker_audited_all": "Speaker-audited all eligible paragraphs",
                    "debate_excluded": "Speaker-audited, debates excluded",
                    "single_president_documents": "Single-president documents only",
                    "annual_message_strict": "Annual messages, speaker-audited",
                }[treatment],
                "speaker_scope_status": "speaker_audit_complete",
                "source_label": "combat/by_president_treatments_v2.parquet",
                "speaker_run_id": attr_meta["run_id"],
                "speaker_spec_version": attr_meta["spec_version"],
                "canonical_corpus_fingerprint": attr_meta["canonical_corpus_fingerprint"],
                "review_manifest_sha256": attr_meta["review_manifest_sha256"],
                "adjudication_sha256": attr_meta["adjudication_sha256"],
            }
            for flag in FLAGS:
                count = int(cell[flag].sum())
                out[f"n_{flag}"] = count
                out[flag] = count / n_paragraphs if n_paragraphs else np.nan
            for entity_type in ("group", "institution", "nation", "other", "person"):
                out[f"adv_n_{entity_type}"] = int(entity_counts.loc[president, entity_type]) if president in entity_counts.index and entity_type in entity_counts.columns else 0
            out["support_status"] = "not_available" if not n_paragraphs else ("thin_record" if n_speeches < 5 or n_paragraphs < 100 else "observed")
            rows.append(out)
    long = pd.DataFrame(rows).sort_values(["treatment", "display_order"]).reset_index(drop=True)
    if len(long) != 225 or long.duplicated(["treatment", "president"]).any():
        raise ValueError("president-conflict-v2 must have 45 rows for each of five treatments")
    selected = long.loc[long["treatment"].eq("speaker_audited_all")].copy().reset_index(drop=True)
    selected["source_label"] = "combat/by_president_speaker_audited_v2.parquet"
    corpus_end_date = pd.Timestamp(speeches["date"].max()).date().isoformat()
    target_mix = speaker_audited_target_mix_by_era(
        base,
        entity_rows,
        metadata={
            "source_label": "combat/target_mix_by_era_speaker_audited_v1.parquet",
        },
    )
    target_mix["corpus_end_date"] = corpus_end_date
    target_mix["annotation_generation"] = generation
    provenance_columns = {
        "speaker_run_id": "run_id",
        "speaker_spec_version": "spec_version",
        "canonical_corpus_fingerprint": "canonical_corpus_fingerprint",
        "review_manifest_sha256": "review_manifest_sha256",
        "adjudication_sha256": "adjudication_sha256",
    }
    for column, source_column in provenance_columns.items():
        target_mix[column] = attr_meta[source_column]
    category_columns = [
        "adv_n_nation", "adv_n_group", "adv_n_person",
        "adv_n_institution", "adv_n_other",
    ]
    if not np.array_equal(
        target_mix[category_columns].sum().to_numpy(dtype=int),
        selected[category_columns].sum().to_numpy(dtype=int),
    ):
        raise ValueError(
            "era and president target-mix artifacts do not partition the same mentions"
        )
    if int(target_mix["n_paragraphs"].sum()) != int(selected["n_paragraphs"].sum()):
        raise ValueError(
            "era and president target-mix artifacts do not partition the same paragraphs"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_dir / BY_PRESIDENT_TREATMENTS_V2_PATH.name, index=False)
    selected.to_parquet(out_dir / BY_PRESIDENT_SPEAKER_V2_PATH.name, index=False)
    target_mix.to_parquet(
        out_dir / TARGET_MIX_BY_ERA_SPEAKER_V1_PATH.name, index=False
    )
    meta = {
        "schema_version": "president-conflict-v2",
        "selected_treatment": "speaker_audited_all",
        "treatments": treatments,
        "presidents_per_treatment": 45,
        "rows": len(long),
        "target_mix_schema": "conflict-target-mix-v1",
        "target_mix_eras": len(target_mix),
        "target_mix_mentions": int(target_mix["n_adversarial_entities"].sum()),
        "target_mix_path": TARGET_MIX_BY_ERA_SPEAKER_V1_PATH.name,
        "speaker_attribution_metadata_sha256": attr_meta["metadata_sha256"],
        "annotation_generation": generation,
        "annotation_active_pointer_sha256": annotation_ledger_module.sha256_file(pointer),
        "api_calls": 0,
    }
    meta["metadata_sha256"] = annotation_ledger_module.sha256_text(annotation_ledger_module.canonical_json(meta))
    (out_dir / PRESIDENT_CONFLICT_V2_META_PATH.name).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "treatments": long,
        "speaker_audited_all": selected,
        "target_mix_by_era": target_mix,
    }


def genre_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Where each era's / decade's flagged paragraphs actually LIVE, by genre.

    §3's claim that the 1930s `party_attack` peak is 85% campaign oratory (140
    of 164 flagged paragraphs from 6 speeches) is the report's main genre
    argument, so it ships as a table. `share_of_flagged` is the column that
    argument rests on: a genre can carry a high rate and still be irrelevant to
    an era's total if it holds few paragraphs.
    """
    rows = []
    for key in ("era", "decade"):
        for group, cell in df.groupby(key, observed=True):
            for f in FLAGS:
                total = int(cell[f].sum())
                for genre, part in cell.groupby("speech_type", observed=True):
                    flagged = int(part[f].sum())
                    rows.append({
                        "grain": key,
                        "group": str(group),
                        "flag": f,
                        "speech_type": genre,
                        "n_speeches": part["doc_name"].nunique(),
                        "n_paragraphs": len(part),
                        "paragraph_share": len(part) / len(cell),
                        "rate": part[f].mean(),
                        "n_flagged": flagged,
                        "share_of_flagged": (flagged / total) if total else np.nan,
                    })
    return pd.DataFrame(rows)


def lexical_baseline(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """The dumb regex proxies (`us_them`, `opponents`) per era, raw and
    SOTU-only, as an uncorrelated-error cross-check.

    Deliberately NOT ground truth: this repo already retracted an NRC-lexicon
    tone finding (commit c214a40) because lexicon counting cannot see sarcasm.
    The proxies are here so that agreement is corroboration from a method with
    different failure modes, and disagreement is a flag to investigate — never
    to overrule the LLM flags.

    `path` resolves to `SPEECH_MARKERS_PATH` at CALL time, not import time, so
    the module constant is a working redirect (the same idiom as
    `build_combativeness`'s `out_dir`). That matters because the row-complete
    guard below rejects any markers table that does not cover exactly the
    speeches in `df` — so a caller driving a synthetic frame has to be able to
    point this at a matching synthetic markers table, including through `main()`
    where no argument can be threaded.
    """
    markers = pd.read_parquet(SPEECH_MARKERS_PATH if path is None else path)
    speech_flags = (
        df.groupby("doc_name", observed=True)
        .agg(
            era=("era", "first"),
            is_sotu=("is_sotu", "first"),
            **{f: (f, "mean") for f in FLAGS},
        )
        .reset_index()
    )
    joined = speech_flags.merge(
        markers[["doc_name", "us_them", "opponents", "n_words"]],
        on="doc_name",
        validate="one_to_one",
    )
    _require_full_merge(
        len(joined),
        {"speech_flags": len(speech_flags), "speech_markers": len(markers)},
        "speech flags x speech_markers",
    )

    rows = []
    for treatment, frame in [("raw", joined), ("sotu_only", joined[joined["is_sotu"]])]:
        grouped = frame.groupby("era", observed=True)
        agg = grouped[["us_them", "opponents", "n_words"]].sum()
        agg["us_them_per_10k"] = agg["us_them"] / agg["n_words"] * 10_000
        agg["opponents_per_10k"] = agg["opponents"] / agg["n_words"] * 10_000
        agg["n_speeches"] = grouped.size()
        # Reindex FIRST, label SECOND. The other order back-fills an era with no
        # speech in this treatment with `treatment = NaN`, so a consumer
        # filtering `treatment == "sotu_only"` silently drops that era instead of
        # seeing it as unmeasured — the silent-row-loss failure this repo's
        # keyed-merge convention exists to prevent. Rates stay NaN (genuinely
        # unmeasured); the speech count is a true 0.
        agg = agg.reindex(ERA_ORDER)
        agg["n_speeches"] = agg["n_speeches"].fillna(0).astype(int)
        agg["treatment"] = treatment
        rows.append(agg.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["era_order"] = out["era"].map({e: i for i, e in enumerate(ERA_ORDER)})
    return out


def lexical_correlations(df: pd.DataFrame, path: Path | None = None) -> dict:
    """Speech-level Spearman correlation between each LLM flag rate and each
    lexical proxy rate. Reported as a magnitude check, not a validation gate.

    `path` late-binds to `SPEECH_MARKERS_PATH` for the same reason as
    `lexical_baseline`."""
    markers = pd.read_parquet(SPEECH_MARKERS_PATH if path is None else path)
    speech_flags = (
        df.groupby("doc_name", observed=True)
        .agg(**{f: (f, "mean") for f in FLAGS})
        .reset_index()
    )
    joined = speech_flags.merge(
        markers[["doc_name", "us_them", "opponents", "n_words"]],
        on="doc_name",
        validate="one_to_one",
    )
    joined["us_them_per_10k"] = joined["us_them"] / joined["n_words"] * 10_000
    joined["opponents_per_10k"] = joined["opponents"] / joined["n_words"] * 10_000
    return {
        f"{flag}~{proxy}": round(
            float(joined[flag].corr(joined[proxy], method="spearman")), 4
        )
        for flag in FLAGS
        for proxy in ["us_them_per_10k", "opponents_per_10k"]
    }


def exemplars(df: pd.DataFrame) -> pd.DataFrame:
    """Top-scoring passages per era per flag — receipts for every point on the
    chart.

    Ranking is by adversarial-entity count (an INDEPENDENT signal from the flag
    itself, so the exemplar is not merely the flag agreeing with itself), then
    by how many of the three flags fired, then deterministically by key. Length
    bounds are a disclosed quotability filter, recorded on every row.
    """
    df = df.assign(n_flags=df[FLAGS].sum(axis=1))
    pool = df[df["word_count"].between(EXEMPLAR_MIN_WORDS, EXEMPLAR_MAX_WORDS)]
    rows = []
    for flag in FLAGS:
        hits = pool[pool[flag]]
        for era in ERA_ORDER:
            cell = hits[hits["era"] == era].sort_values(
                ["n_adversarial", "n_flags", "doc_name", "para_idx"],
                ascending=[False, False, True, True],
            ).head(EXEMPLARS_PER_CELL)
            for rank, (_, r) in enumerate(cell.iterrows(), start=1):
                rows.append({
                    "era": era,
                    "era_order": ERA_ORDER.index(era),
                    "flag": flag,
                    "rank": rank,
                    "president": r["president"],
                    "year": int(r["year"]),
                    "title": r["title"],
                    "speech_type": r["speech_type"],
                    "doc_name": r["doc_name"],
                    "para_idx": int(r["para_idx"]),
                    "text": r["text"],
                    "word_count": int(r["word_count"]),
                    "n_adversarial": int(r["n_adversarial"]),
                    "n_adv_foreign": int(r["n_adv_foreign"]),
                    "n_adv_domestic": int(r["n_adv_domestic"]),
                    "n_flags": int(r["n_flags"]),
                    "party_attack": bool(r["party_attack"]),
                    "enemy_naming": bool(r["enemy_naming"]),
                    "zero_sum": bool(r["zero_sum"]),
                    "selection_min_words": EXEMPLAR_MIN_WORDS,
                    "selection_max_words": EXEMPLAR_MAX_WORDS,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build_combativeness(
    df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    markers_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute every table and write them under `out_dir` (default data/combat/).

    `out_dir` is injectable so a test can build the real artifact shape into a
    tmp_path instead of overwriting the committed outputs. `markers_path` is
    forwarded to both lexical consumers of `speech_markers.parquet`, so a caller
    driving a synthetic frame can point them at a matching synthetic markers
    table instead of having to replace the functions themselves. Both default to
    None and resolve to the module constants at call time.
    """
    if df is None:
        df = load_frame()
    if out_dir is None:
        out_dir = COMBAT_DIR
    bands = load_agreement_bands()

    # Fit each cell ONCE; the rate table and the ratio table then share the same
    # resamples, so a marginal interval and a ratio interval can never disagree.
    era_fits = _fit_all(df, "era", ERA_ORDER)

    tables = {
        "combativeness": rate_table(df, "era", ERA_ORDER, bands, fits=era_fits),
        "peak_decades": rate_table(df, "decade", PEAK_DECADES, bands),
        "ratios": ratio_table(
            df, "era", ERA_ORDER, REFERENCE_ERA, fits=era_fits, bands=bands
        ),
        "entity_consistency": entity_consistency(df),
        "lexical_baseline": lexical_baseline(df, path=markers_path),
        "adversary_mix": adversary_mix(df),
        "by_president": by_president(df),
        "genre_decomposition": genre_decomposition(df),
        "exemplars": exemplars(df),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        out = table.copy()
        for col in out.columns:
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].astype(str)
        out.to_parquet(out_dir / OUTPUT_PATHS[name].name, index=False)

    reference = genre_reference(df)
    meta = {
        # No wall-clock timestamp on purpose. Every other output here is a pure
        # function of the frozen inputs, so stamping the meta file with
        # date.today() would make `git status data/combat/` dirty on any rerun
        # after the generation date — for a reason that carries no information.
        # Provenance identity comes from `corpus_fingerprint` (what the numbers
        # were computed against) and git (when they landed).
        "era_scheme": "trends.ERAS",
        "flags": FLAGS,
        "treatments": TREATMENTS,
        "composite_index": "none — the three flags are reported separately by design",
        "peak_decades": PEAK_DECADES,
        "bootstrap": {
            "n": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "cluster_unit": "speech",
            "min_clusters_for_ci": MIN_CLUSTERS_FOR_CI,
            "low_cluster_caution": LOW_CLUSTER_CAUTION,
        },
        "genre_standardization": {
            "reference": "pooled corpus paragraph share by speech_type",
            "reference_weights": {k: round(float(v), 6) for k, v in reference.items()},
            "min_cell_speeches": MIN_CELL_SPEECHES,
        },
        "annotator_disagreement": {
            "applied": bands is not None,
            # Keep the public artifact identifier stable when tests (or callers)
            # redirect the physical input path to an isolated workspace.
            "source": "data/llm_annotations/agreement_v1.parquet",
            "status": (
                "ABSENT — intervals contain sampling error only. Pending "
                "inter-model-agreement-check."
                if bands is None
                else "applied"
            ),
        },
        "ratios": {
            "reference_era": REFERENCE_ERA,
            "method": "speech-clustered bootstrap of the ratio, reusing the "
                      "marginal resamples; suppressed if either cell is below "
                      "the n-floor",
        },
        "exemplars_per_cell": EXEMPLARS_PER_CELL,
        "annotation_run_ids": sorted(df["run_id"].unique().tolist()),
        "corpus_fingerprint": corpus_fingerprint(),
        "lexical_spearman": lexical_correlations(df, path=markers_path),
        "mistyped_nation_recount": mistyped_nation_recount(df),
        "thin_stratum_policy": {
            "min_cell_speeches": MIN_CELL_SPEECHES,
            "suppress_weight": THIN_STRATUM_SUPPRESS_WEIGHT,
            "caution_weight": THIN_STRATUM_CAUTION_WEIGHT,
            "rationale": (
                "the standardized estimator averages within-stratum resamples, "
                "so its interval is disciplined by the weight sitting in strata "
                "below min_clusters_for_ci, not by the total speech count"
            ),
        },
        "api_calls": 0,
    }
    (out_dir / COMBAT_META_PATH.name).write_text(json.dumps(meta, indent=2) + "\n")
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combativeness over time (pure local compute; $0, no API calls)."
    )
    ap.add_argument("--quiet", action="store_true", help="write outputs without printing")
    ap.add_argument("--president-v2", action="store_true", help="build only president-conflict-v2")
    args = ap.parse_args()

    if args.president_v2:
        tables = build_president_conflict_v2()
        if not args.quiet:
            print(tables["speaker_audited_all"].to_string(index=False))
        return

    df = load_frame()
    tables = build_combativeness(df)
    if args.quiet:
        return

    pd.set_option("display.width", 200)
    combat = tables["combativeness"]
    for treatment in TREATMENTS:
        print(f"\n=== {treatment} ===")
        pivot = combat[combat["treatment"] == treatment].pivot_table(
            index=["era_order", "era"], columns="flag", values="rate", observed=True
        )
        print(pivot.round(4).to_string())
    print("\n=== enemy-naming <-> adversarial-entity consistency ===")
    print(tables["entity_consistency"].round(4).to_string(index=False))
    print(f"\n=== {REFERENCE_ERA} vs each era, SOTU-only (ratio [95% CI]) ===")
    r = tables["ratios"]
    r = r[(r["treatment"] == "sotu_only")][
        ["flag", "denominator", "ratio", "ci_lo", "ci_hi", "ci_status"]
    ]
    print(r.round(2).to_string(index=False))
    print(f"\nWrote {len(tables)} tables to {COMBAT_DIR}")


if __name__ == "__main__":
    main()
