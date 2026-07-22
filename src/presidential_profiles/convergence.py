"""Have presidential agendas converged? A pre-registered, confound-controlled,
permutation-calibrated test.

Read `notes/convergence-prereg-v1.md` FIRST. It was committed before this file
existed, and it — not this docstring — is the specification. The git ordering of
the two commits IS the pre-registration.

Four design commitments carry the whole module, and every one of them exists
because the obvious version of this analysis **manufactures the finding**:

1. **Cluster rarefaction, not paragraph rarefaction.** Issue labels cluster
   within speeches (within-speech ICC 0.05-0.17), and the number of distinct
   speeches behind a president's 50 in-window paragraphs rises from ~10
   (1800-1850) to ~38 (2000+). Equalizing paragraph count therefore does NOT
   equalize independent observations: on a clustered null with zero real
   convergence the paragraph-rarefied estimator returns rho = -0.605 (the
   2026-07-13 review's measurement; `_selftest` leg (a) reproduces the same sign
   and shape on its own synthetic corpora). This
   module samples S speeches and B paragraphs from each, **both with
   replacement** — with replacement precisely because a without-replacement
   finite-population correction `(1 - S/n)` would reinstall an era-varying noise
   floor keyed to the same drifting pool size `n`.

2. **`scipy.stats.spearmanr`'s p-value is BANNED from this analysis** — from H1,
   H2, the jackknife, the rival test, every control curve and every sensitivity.
   Measured type-I error on these 93%-overlapping windows is 58%. `spearman_rho`
   below discards the p-value at the call site and returns a bare float; it is
   the only way rho enters this module. Every p reported anywhere here comes
   from the president-permutation null.

3. **Composition is an EQUATION with an explicit no-topic bin.** Each paragraph
   contributes total mass exactly 1, split 1/k across its k labels; a paragraph
   with no label puts its whole mass in the no-topic bin:

       c[j] = (1/m) * SUM_i [ 1{k_i>=1} 1{j in L_i} / k_i + 1{k_i=0} 1{j=no-topic} ]

   so `sum_j c[j] = 1` identically and label-density drift (1.49 -> 0.83
   labels/paragraph across the corpus) cannot move the measure by itself.

4. **Two things are subtracted before the word "convergence" may be used.** The
   per-window speech-block permutation floor (published as its own column, never
   a scalar — it drifts 29-53%), and an entropy-matched null that separates
   "agendas got more ALIKE" from "each agenda got BROADER".

Arms: `corex_all` (PRIMARY), `corex_sotu` (CO-PRIMARY), `llm_all` (sensitivity).
**No vote rule** — a ">=3 of 4 methods decline" rule fires 12-46% under a true
null. **Dispersion LEVELS are never compared across arms**, only trends within
an arm: the arms have different bin counts and different sampling budgets, and
`llm_all` additionally inherits `union-projection-inflates-wide-crosswalk-issues`
(sidestepped here by using native level-2 labels, but the annotations are
unchanged).

$0 GUARD: pure local compute over frozen parquets. This module never imports
`anthropic` and never constructs a client. Zero API calls.

Run as:
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.convergence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from .attention import canonical_label_map, load_taxonomy, normalize_topics
from .corpus import DATA_DIR
from .eras import check_staleness
from .llm_annotations import (
    corpus_fingerprint,
    load_paragraph_annotations,
    load_speech_annotations,
)

CONVERGENCE_DIR = DATA_DIR / "convergence"
DISPERSION_PATH = CONVERGENCE_DIR / "dispersion_curves.parquet"
PERMUTATION_PATH = CONVERGENCE_DIR / "permutation_null.parquet"
JACKKNIFE_PATH = CONVERGENCE_DIR / "jackknife.parquet"
PARADOX_PATH = CONVERGENCE_DIR / "paradox.parquet"
META_PATH = CONVERGENCE_DIR / "convergence_meta.json"

OUTPUT_PATHS = {
    "dispersion_curves": DISPERSION_PATH,
    "permutation_null": PERMUTATION_PATH,
    "jackknife": JACKKNIFE_PATH,
    "paradox": PARADOX_PATH,
}

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
PARA_ISSUES_PATH = DATA_DIR / "paragraph_issues.parquet"
# Read-only ANCHOR input for H2a. Never recomputed here, never scored.
PRESIDENT_EMBEDDINGS_PATH = DATA_DIR / "president_embeddings.parquet"

SOTU_TYPE = "state_of_the_union_or_annual_message"
NO_TOPIC = "(no topic)"

# The 15 anchored legacy issues of `paragraph_issues.parquet`. The `Discovered
# 1..7` columns are deliberately excluded: `Discovered 3` is the corpus's one
# real noise topic (NPMI -0.047) and the discovered set is not a taxonomy, so
# admitting it would put an uninterpretable bin into a composition equation.
ISSUES = [
    "Economy & jobs", "Taxes & budget", "War & military", "Foreign policy",
    "Immigration", "Civil rights & race", "Health care", "Education",
    "Crime & justice", "Energy & environment", "Trade & tariffs", "Agriculture",
    "Religion & values", "Money & banking", "Infrastructure",
]

# --------------------------------------------------------------------------
# pre-registered constants (notes/convergence-prereg-v1.md, section 11)
# --------------------------------------------------------------------------

WINDOW_LEN = 30
WINDOW_STEP = 2
DRAWS = 20                 # D: rarefaction draws per window
FLOOR_DRAWS = 50           # F: speech-block re-deals per window
N_PERMUTATIONS = 2000      # R
ALPHA = 0.05               # one-sided
MIN_PRESIDENTS = 3         # a window below this is not used in any trend
MASTER_SEED = 20260721

SELFTEST_REPLICATES = 40
SELFTEST_PERMUTATIONS = 150

# `_selftest` pass criteria, fixed in the pre-registration before any run.
SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO = 0.25
SELFTEST_PARAGRAPH_BIAS_MAX_RHO = -0.30
SELFTEST_INJECTED_MAX_RHO = -0.50
SELFTEST_MAX_SIZE = 0.152  # 3-sigma upper bound at 40 replicates under a true 5%

WINDOW_KINDS = ("rolling", "nonoverlapping")
TREATMENTS = ("all_windows", "ends_2014")
DATING_END_YEAR = 2014     # the H1 Trump-clause dating procedure (prereg 7.4)

# The ONE trust-gate vocabulary for this whole layer, shared with `combat.py`
# and `eras.py`. Every table under `data/convergence/` carries a `ci_status`
# drawn from it on every row, because every one of those tables has rows a
# consumer could plot: window rows (`window_status`), leave-one-out rows
# (`jackknife_status`), permutation rows (`null_status`) and president rows
# (`paradox_table`). A reader must never be able to plot an estimate blind, and
# must never have to learn a second vocabulary to do it.
CI_STATUS_VALUES = ("no_data", "suppressed_n_floor", "low_cluster_caution", "ok")

# Jackknife gate. A leave-one-out rho is scored against the FULL-design
# permutation null, so the comparison is only as good as the agreement between
# the leave-one-out window set and the full one.
JACKKNIFE_MIN_WINDOWS = 3          # below this `spearman_rho` is NaN by construction
JACKKNIFE_SUPPRESS_RETENTION = 0.75
JACKKNIFE_CAUTION_RETENTION = 0.90

# permutation_null gate: a rho scored against a null whose draws mostly came
# back NaN is not an inference.
NULL_SUPPRESS_VALID_FRACTION = 0.75
NULL_CAUTION_VALID_FRACTION = 0.99

# paradox_table gate, on the number of windows a president's corrected distance
# was averaged over. POST-HOC, in the same sense as the JACKKNIFE_* thresholds
# below: prereg 7.2 fixes the estimand but declares no window floor, and these
# two numbers were named after the run. They may only downgrade a row, and no
# pre-registered inference reads them.
PARADOX_OK_WINDOWS = 10
PARADOX_CAUTION_WINDOWS = 3

STATISTICS = ("dispersion", "floor_corrected", "excess_ratio", "pair_regression")


class ArmSpec(NamedTuple):
    """One pre-registered arm: label source, genre filter, cluster budget."""

    name: str
    label_source: str            # "corex" | "llm"
    speech_types: tuple[str, ...] | None   # None = every genre
    n_speeches: int              # S
    n_paragraphs: int            # B
    role: str


ARMS = (
    ArmSpec("corex_all", "corex", None, 8, 6, "primary"),
    ArmSpec("corex_sotu", "corex", (SOTU_TYPE,), 4, 12, "co_primary"),
    ArmSpec("llm_all", "llm", None, 8, 6, "sensitivity"),
)


# --------------------------------------------------------------------------
# the banned p-value
# --------------------------------------------------------------------------


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho, and ONLY rho.

    `scipy.stats.spearmanr` also returns a p-value. That p-value is **banned
    from this analysis**: on 30-year windows stepped 2 years (93% shared
    content, ~8 independent blocks in 240 years) its measured type-I error at
    nominal 5% is 58%, and at nominal 1% it is 47%. The ban covers H1, H2, the
    jackknife, the rival test, every control curve and every sensitivity.

    The discard happens HERE, at the call site, so no caller can reach the
    p-value even by accident: this function returns a bare float. Every p-value
    published by this module comes from `permutation_null`.

    Returns NaN for fewer than 3 points or a constant input, rather than letting
    scipy's NaN-with-a-warning path decide.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    from scipy.stats import spearmanr  # local import: scipy is only needed here

    result = spearmanr(a, b)
    # `.statistic` only. The `.pvalue` attribute is deliberately never read.
    return float(result.statistic)


# --------------------------------------------------------------------------
# keyed-merge guard (mirrors combat._require_full_merge / taxonomy's original)
# --------------------------------------------------------------------------


def _require_full_merge(n_merged: int, inputs: dict[str, int], stage: str) -> None:
    """Raise unless a keyed merge kept every row of every input.

    `validate="one_to_one"` catches DUPLICATE keys but an inner join silently
    DROPS rows when the key sets merely diverge. Here that would quietly
    restrict the composition universe to an intersection while the note claimed
    full-corpus coverage.
    """
    if any(n_merged != n for n in inputs.values()):
        breakdown = ", ".join(f"{k}={v}" for k, v in inputs.items())
        raise ValueError(
            f"{stage}: keyed merge changed the row count (merged={n_merged}; "
            f"{breakdown}). The key sets diverge and the inner join dropped "
            f"unmatched rows. Refusing to proceed."
        )


# --------------------------------------------------------------------------
# compositions
# --------------------------------------------------------------------------


class Composition(NamedTuple):
    """One paragraph-level mass matrix plus the metadata every design needs.

    Rows are sorted by `(doc_name, para_idx)`, so each speech's paragraphs are
    CONTIGUOUS — the whole sampler is built on that, and `build_design` asserts
    it rather than assuming it.
    """

    label_source: str
    bins: list[str]
    mass: np.ndarray          # (n_paragraphs, K) float32, rows sum to 1
    doc_name: np.ndarray
    para_idx: np.ndarray
    president: np.ndarray
    year: np.ndarray
    speech_type: np.ndarray
    n_label_assignments: int  # post-dedup, for the meta record


def _mass_from_label_lists(labels: list[list[int]], n_bins: int) -> np.ndarray:
    """The composition equation of prereg section 3, one row per paragraph.

    `labels[i]` is paragraph i's list of substantive bin indices (possibly
    empty). Bin `n_bins - 1` is the no-topic bin. Each paragraph contributes
    total mass exactly 1, so label-density drift cannot move any downstream
    average by itself.
    """
    mass = np.zeros((len(labels), n_bins), dtype=np.float32)
    for i, ls in enumerate(labels):
        if ls:
            share = np.float32(1.0 / len(ls))
            for j in ls:
                mass[i, j] += share
        else:
            mass[i, n_bins - 1] = 1.0
    return mass


def build_compositions(
    label_source: str = "corex",
    issues: pd.DataFrame | None = None,
    annotations: pd.DataFrame | None = None,
    speech_annotations: pd.DataFrame | None = None,
    taxonomy: dict | None = None,
) -> Composition:
    """Build the paragraph-level composition mass matrix for one label source.

    `corex` uses the 15 anchored issues of `paragraph_issues.parquet` plus a
    no-topic bin (16 bins). `llm` uses `taxonomy_v1`'s 50 level-2 topic names
    plus a no-topic bin (51 bins) — the NATIVE label set, deliberately not the
    legacy-15 projection, because that projection goes through `triangulate.py`'s
    **union** crosswalk which inflates apparent breadth for wide-fan-out issues
    and is never sensitivity-tested (`union-projection-inflates-wide-crosswalk-issues`),
    and because the crosswalk is one-directional (7 of 50 level-2 topics have no
    legacy parent). The price is that `llm_all`'s dispersion LEVEL is not
    comparable to a CorEx arm's — which the pre-registration forbids anyway.

    Every input is injectable so the assembly, its merge guards and the
    composition equation can be exercised on tiny synthetic frames rather than
    the 36k-row corpus (CLAUDE.md's testing convention).

    Raises:
        ValueError: if a keyed merge drops rows, if a paragraph has no
            `speech_type`, or (llm) if a topic label is not in `taxonomy_v1`.
    """
    if label_source not in {"corex", "llm"}:
        raise ValueError(f"unknown label_source {label_source!r}")

    iss = pd.read_parquet(PARA_ISSUES_PATH) if issues is None else issues
    missing = [c for c in ISSUES if c not in iss.columns]
    if missing:
        raise ValueError(f"paragraph_issues is missing anchored issue columns {missing}")
    speech = (
        load_speech_annotations("speech_annotations")
        if speech_annotations is None else speech_annotations
    )[["doc_name", "speech_type"]]

    base = iss[["doc_name", "para_idx", "president", "year", *ISSUES]].copy()

    if label_source == "corex":
        bins = [*ISSUES, NO_TOPIC]
        flags = base[ISSUES].to_numpy(dtype=bool)
        label_lists = [list(np.flatnonzero(row)) for row in flags]
    else:
        taxonomy = taxonomy or load_taxonomy()
        label_map = canonical_label_map(taxonomy)
        names = sorted({entry["name"] for entry in taxonomy["level2"]})
        bins = [*names, NO_TOPIC]
        index = {name: i for i, name in enumerate(names)}
        ann = (
            load_paragraph_annotations("paragraph_annotations")
            if annotations is None else annotations
        )[["doc_name", "para_idx", "topics"]]
        n_before = len(base)
        base = base.merge(ann, on=["doc_name", "para_idx"], validate="one_to_one")
        _require_full_merge(
            len(base), {"paragraph_issues": n_before, "paragraph_annotations": len(ann)},
            "paragraph_issues x paragraph_annotations",
        )
        # Both mandatory normalizations, in order, through the ONE normalizer
        # this repo keeps (`attention`): casefold-to-canonical with a raise on
        # any unmapped label, then de-duplication of (paragraph, topic) pairs —
        # `normalize_topics` does both and 722 duplicate pairs die here.
        label_lists = [
            [index[t] for t in normalize_topics(raw, label_map)]
            for raw in base["topics"]
        ]

    # No row-count guard here, unlike the keyed merge above: a LEFT merge whose
    # right side is validated `many_to_one` cannot change the row count — a
    # duplicated `doc_name` raises `MergeError` first, and an unmatched one
    # produces a NaN `speech_type`, which the next check refuses.
    base = base.merge(speech, on="doc_name", how="left", validate="many_to_one")
    if base["speech_type"].isna().any():
        n_missing = int(base["speech_type"].isna().sum())
        raise ValueError(
            f"{n_missing} paragraphs have no speech_type; the genre arm would "
            "silently drop them. speech_annotations must cover every speech."
        )

    order = np.lexsort((base["para_idx"].to_numpy(), base["doc_name"].to_numpy()))
    base = base.iloc[order].reset_index(drop=True)
    label_lists = [label_lists[i] for i in order]

    mass = _mass_from_label_lists(label_lists, len(bins))
    return Composition(
        label_source=label_source,
        bins=bins,
        mass=mass,
        doc_name=base["doc_name"].to_numpy(),
        para_idx=base["para_idx"].to_numpy(),
        president=base["president"].to_numpy(),
        year=base["year"].to_numpy(dtype=np.int64),
        speech_type=base["speech_type"].to_numpy(),
        n_label_assignments=int(sum(len(ls) for ls in label_lists)),
    )


# --------------------------------------------------------------------------
# window grid + design
# --------------------------------------------------------------------------


class Window(NamedTuple):
    start: int
    end: int
    center: float
    presidents: np.ndarray          # president indices eligible in this window
    pools: tuple[np.ndarray, ...]   # per eligible president, in-window speech ids


class Design(NamedTuple):
    """Everything the sampler needs, with the DESIGN separated from the DATA.

    `windows` fixes who is eligible where; nothing in a permutation is allowed
    to move it. Only the CONTENT that fills a slot is permuted, which is what
    makes the null absorb window overlap, drifting speech supply and drifting
    eligible-president counts instead of assuming them away.
    """

    arm: str
    label_source: str
    bins: list[str]
    mass: np.ndarray
    starts: np.ndarray              # (n_speeches,) first paragraph row of each speech
    lens: np.ndarray                # (n_speeches,) paragraphs per speech
    speech_president: np.ndarray
    speech_year: np.ndarray
    qualifying: np.ndarray          # (n_qualifying,) speech ids passing genre + B
    presidents: list[str]
    president_first_year: np.ndarray
    global_pool: tuple[np.ndarray, ...]   # per president, ALL qualifying speech ids
    n_speeches: int                 # S
    n_paragraphs: int               # B
    windows: tuple[Window, ...]
    window_kind: str

    @property
    def n_bins(self) -> int:
        return self.mass.shape[1]


def rolling_windows(first_year: int, last_year: int) -> list[tuple[int, int]]:
    """30-year windows stepped 2, anchored at the corpus's first year.

    Anchoring at `first_year` (rather than one year later, as the 2026-07-13
    review memo's grid did) covers the corpus's first AND last year; the
    deviation and its consequence are declared in prereg section 2.1.
    """
    return [
        (s, s + WINDOW_LEN - 1)
        for s in range(first_year, last_year - WINDOW_LEN + 2, WINDOW_STEP)
    ]


def nonoverlapping_windows(first_year: int, last_year: int) -> list[tuple[int, int]]:
    """The low-power sensitivity grid: 8 disjoint 30-year windows, last truncated."""
    out = []
    s = first_year
    while s <= last_year:
        out.append((s, min(s + WINDOW_LEN - 1, last_year)))
        s += WINDOW_LEN
    return out


def window_status(n_eligible: int) -> str:
    """The machine-readable trust gate, published on every window row.

    A downstream reader must never be able to plot a dispersion blind:
    `no_data` = fewer than two presidents (no pair exists at all);
    `suppressed_n_floor` = exactly two (a single pair is not a dispersion);
    `low_cluster_caution` = three or four; `ok` = five or more. Only `ok` and
    `low_cluster_caution` windows enter any trend statistic, and because the
    rule reads the DESIGN (not the values) the retained set is identical inside
    every permutation.
    """
    if n_eligible < 2:
        return "no_data"
    if n_eligible < MIN_PRESIDENTS:
        return "suppressed_n_floor"
    if n_eligible < 5:
        return "low_cluster_caution"
    return "ok"


def n_windows_in_trend(design: Design, drop: int | None = None) -> int:
    """How many of `design`'s windows enter a trend, optionally dropping one
    president. Reads the DESIGN only — no sampling, no values — which is what
    makes it usable as a trust gate rather than as a result.
    """
    total = 0
    for w in design.windows:
        n = int((w.presidents != drop).sum()) if drop is not None else len(w.presidents)
        if window_status(n) not in {"no_data", "suppressed_n_floor"}:
            total += 1
    return total


def jackknife_status(n_windows_used: int, n_windows_full: int) -> str:
    """The trust gate for one leave-one-president-out row.

    `n_windows_used` on its own is only a proxy. What actually decides whether
    a jackknife rho may be read is how far the leave-one-out design has drifted
    from the FULL design, because the row's p-value comes from the full-design
    permutation null (the declared approximation of prereg 7.1). Deleting a
    president who was the third eligible one in many windows pushes those
    windows below the `MIN_PRESIDENTS` floor; the surviving rho is then a
    different estimand from the null it is being scored against.

    `no_data` = fewer than three windows, where `spearman_rho` is NaN by
    construction; `suppressed_n_floor` = under 75% of the full design's trend
    windows survive; `low_cluster_caution` = under 90%; `ok` otherwise.

    **The 0.75 and 0.90 thresholds are POST-HOC, and this is a disclosure, not
    an apology.** Prereg 7.1 states the approximation ("deleting one president
    barely perturbs the design") as prose and fixes NO operational threshold;
    `JACKKNIFE_SUPPRESS_RETENTION` / `JACKKNIFE_CAUTION_RETENTION` were named
    later, when the retention distribution was already computable — the worst
    observed retention is **0.913** (Nixon and LBJ on `corex_sotu`, 84/92;
    Madison on the two 105-window arms, 96/105), so on the real run every row
    lands `ok`, 1.3 points above the caution line. Two things keep that
    acceptable and both must stay true: the gate can only ever DOWNGRADE a row,
    and no pre-registered inference reads it. The pre-registration is NOT to be
    amended to contain these numbers — the git ordering of its commit is the
    deliverable, and back-filling a threshold into it would destroy exactly the
    property it exists to have.
    """
    if n_windows_full <= 0 or n_windows_used < JACKKNIFE_MIN_WINDOWS:
        return "no_data"
    retained = n_windows_used / n_windows_full
    if retained < JACKKNIFE_SUPPRESS_RETENTION:
        return "suppressed_n_floor"
    if retained < JACKKNIFE_CAUTION_RETENTION:
        return "low_cluster_caution"
    return "ok"


def null_status(rho: float, n_valid: int, n_permutations: int) -> str:
    """The trust gate for one `permutation_null` row.

    A permuted curve can fail to yield a rho (too few usable windows, a
    constant series), and every such draw is dropped from the null before the
    p-value is formed. A p computed against a null that mostly evaporated is
    not an inference, so the surviving fraction is published as a status rather
    than left for a reader to reconstruct from `n_permutations_valid`.
    """
    if not np.isfinite(rho) or n_valid == 0 or n_permutations <= 0:
        return "no_data"
    fraction = n_valid / n_permutations
    if fraction < NULL_SUPPRESS_VALID_FRACTION:
        return "suppressed_n_floor"
    if fraction < NULL_CAUTION_VALID_FRACTION:
        return "low_cluster_caution"
    return "ok"


def paradox_status(n_windows_eligible: int) -> str:
    """The trust gate for one `paradox_table` president row.

    A president's corrected distance is a mean over the windows they were
    eligible in, so a president who appears in one window is a single noisy draw
    dressed as a ranked statistic. `ok` = at least `PARADOX_OK_WINDOWS`;
    `low_cluster_caution` = at least `PARADOX_CAUTION_WINDOWS`;
    `suppressed_n_floor` below that.

    The thresholds are **post-hoc** (see `PARADOX_OK_WINDOWS`). This lives here,
    beside the other three gates, rather than as an inline `np.where` inside
    `paradox_table` — as an inline expression it was the one gate not reachable
    by name, which is exactly why it drifted out of the shared vocabulary test.
    """
    if n_windows_eligible >= PARADOX_OK_WINDOWS:
        return "ok"
    if n_windows_eligible >= PARADOX_CAUTION_WINDOWS:
        return "low_cluster_caution"
    return "suppressed_n_floor"


def build_design(
    comp: Composition, arm: ArmSpec, window_kind: str = "rolling",
    eligibility: str = "speeches",
) -> Design:
    """Assemble the fixed design for one arm on one window grid.

    `eligibility="speeches"` is the pre-registered rule: a president needs at
    least S qualifying in-window speeches. `eligibility="paragraphs"` is the
    ORIGINAL, superseded design's rule — at least m = S*B in-window paragraphs,
    with no speech floor at all — and exists only so `_selftest` can reconstruct
    that design end to end (rule *and* sampler) rather than swapping the sampler
    inside the new rule, which would understate the bias it manufactures.
    """
    if window_kind not in WINDOW_KINDS:
        raise ValueError(f"unknown window_kind {window_kind!r}")
    if eligibility not in {"speeches", "paragraphs"}:
        raise ValueError(f"unknown eligibility {eligibility!r}")

    codes, uniques = pd.factorize(comp.doc_name)
    if np.any(np.diff(codes) < 0):
        raise ValueError(
            "Composition rows are not grouped by doc_name; the sampler indexes "
            "each speech as a contiguous row block and would silently mix speeches."
        )
    n_docs = len(uniques)
    starts = np.searchsorted(codes, np.arange(n_docs))
    lens = np.bincount(codes, minlength=n_docs)
    speech_president = comp.president[starts]
    speech_year = comp.year[starts]
    speech_type = comp.speech_type[starts]

    genre_ok = (
        np.ones(n_docs, dtype=bool)
        if arm.speech_types is None
        else np.isin(speech_type, list(arm.speech_types))
    )
    qualifying = np.flatnonzero(genre_ok & (lens >= arm.n_paragraphs))

    # Presidents ordered by first speech year, then name — a stable, readable
    # order that does not depend on how the parquet happens to be sorted.
    first_year = pd.Series(comp.year).groupby(comp.president).min()
    presidents = sorted(first_year.index, key=lambda p: (int(first_year[p]), str(p)))
    p_index = {p: i for i, p in enumerate(presidents)}
    speech_pres_idx = np.array([p_index[p] for p in speech_president])

    global_pool = tuple(
        qualifying[speech_pres_idx[qualifying] == i] for i in range(len(presidents))
    )

    lo, hi = int(comp.year.min()), int(comp.year.max())
    spans = (
        rolling_windows(lo, hi) if window_kind == "rolling"
        else nonoverlapping_windows(lo, hi)
    )

    windows = []
    q_year = speech_year[qualifying]
    q_pres = speech_pres_idx[qualifying]
    for start, end in spans:
        inside = (q_year >= start) & (q_year <= end)
        pres, pools = [], []
        for i in range(len(presidents)):
            ids = qualifying[inside & (q_pres == i)]
            enough = (
                len(ids) >= arm.n_speeches
                if eligibility == "speeches"
                else int(lens[ids].sum()) >= arm.n_speeches * arm.n_paragraphs
            )
            if enough:
                pres.append(i)
                pools.append(ids)
        windows.append(
            Window(
                start=start, end=end, center=start + (WINDOW_LEN - 1) / 2.0,
                presidents=np.array(pres, dtype=np.int64), pools=tuple(pools),
            )
        )

    return Design(
        arm=arm.name, label_source=comp.label_source, bins=comp.bins, mass=comp.mass,
        starts=starts.astype(np.int64), lens=lens.astype(np.int64),
        speech_president=speech_pres_idx, speech_year=speech_year,
        qualifying=qualifying, presidents=[str(p) for p in presidents],
        president_first_year=np.array([int(first_year[p]) for p in presidents]),
        global_pool=global_pool,
        n_speeches=arm.n_speeches, n_paragraphs=arm.n_paragraphs,
        windows=tuple(windows), window_kind=window_kind,
    )


def ever_eligible(design: Design) -> np.ndarray:
    """Presidents eligible in at least one window — the permutation's donor set.

    Restricting donors to this set is what lets the permutation sample S
    speeches WITH replacement from a pool that is guaranteed to hold at least S
    of them, so no slot is ever filled from a one-speech presidency.
    """
    seen: set[int] = set()
    for w in design.windows:
        seen.update(w.presidents.tolist())
    return np.array(sorted(seen), dtype=np.int64)


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------


def _pad_pools(pools: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.array([len(p) for p in pools], dtype=np.int64)
    padded = np.zeros((len(pools), int(sizes.max())), dtype=np.int64)
    for e, p in enumerate(pools):
        padded[e, : len(p)] = p
    return padded, sizes


def draw_compositions(
    design: Design, pools: Sequence[np.ndarray], rng: np.random.Generator,
    n_draws: int,
) -> np.ndarray:
    """Cluster-rarefied compositions: (n_draws, n_slots, K).

    S speeches with replacement, then B paragraphs with replacement from each.
    Both stages are with-replacement on purpose: a without-replacement finite
    population correction `(1 - S/n)` depends on the pool size `n`, and `n` is
    exactly the quantity that drifts ~10 -> ~38 across the corpus. With
    replacement the sampling variance is `sigma^2_between / S` regardless of pool
    size, so the estimator's noise is design-invariant by construction — which is
    the entire point of rarefying speeches instead of paragraphs.
    """
    padded, sizes = _pad_pools(pools)
    n_slots, S, B = len(pools), design.n_speeches, design.n_paragraphs
    pos = (rng.random((n_draws, n_slots, S)) * sizes[None, :, None]).astype(np.int64)
    speech_ids = np.take_along_axis(
        np.broadcast_to(padded, (n_draws, n_slots, padded.shape[1])), pos, axis=2
    )
    para_pos = (
        rng.random((n_draws, n_slots, S, B)) * design.lens[speech_ids][..., None]
    ).astype(np.int64)
    rows = design.starts[speech_ids][..., None] + para_pos
    return design.mass[rows.reshape(n_draws, n_slots, S * B)].mean(axis=2)


def draw_compositions_paragraph(
    design: Design, pools: Sequence[np.ndarray], rng: np.random.Generator,
    n_draws: int,
) -> np.ndarray:
    """The ORIGINAL, broken estimator: m paragraphs sampled ignoring speeches.

    Present only so `_selftest` can demonstrate the contrast — that the same
    clustered-null data on which the cluster-rarefied estimator is flat drives
    this one to a strong spurious decline (validated: rho = -0.605). It is never
    used to produce a published number.
    """
    m = design.n_speeches * design.n_paragraphs
    paragraph_pools = [
        np.concatenate([
            np.arange(design.starts[s], design.starts[s] + design.lens[s]) for s in p
        ])
        for p in pools
    ]
    padded, sizes = _pad_pools(paragraph_pools)
    n_slots = len(pools)
    pos = (rng.random((n_draws, n_slots, m)) * sizes[None, :, None]).astype(np.int64)
    rows = np.take_along_axis(
        np.broadcast_to(padded, (n_draws, n_slots, padded.shape[1])), pos, axis=2
    )
    return design.mass[rows].mean(axis=2)


def _entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0).sum(-1)


def pairwise_jsd(c: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in BITS between every pair: (..., E, E) in [0,1]."""
    mixture = 0.5 * (c[..., :, None, :] + c[..., None, :, :])
    h = _entropy(c)
    out = _entropy(mixture) - 0.5 * (h[..., :, None] + h[..., None, :])
    return np.clip(out, 0.0, 1.0)


def _mean_offdiag(j: np.ndarray) -> np.ndarray:
    """Mean over unordered pairs, per leading index."""
    e = j.shape[-1]
    iu, ju = np.triu_indices(e, k=1)
    return j[..., iu, ju].mean(axis=-1)


def speech_block_floor(
    design: Design, pools: Sequence[np.ndarray], rng: np.random.Generator,
    n_draws: int = FLOOR_DRAWS,
) -> tuple[float, np.ndarray]:
    """The per-window noise floor: dispersion if these presidents had no agendas.

    Pools the eligible presidents' qualifying speeches and re-deals them to the
    slots **as whole speech blocks**, preserving each slot's speech count, then
    runs the identical rarefaction. Whole blocks, because the confound being
    priced is precisely the within-speech clustering of labels — dealing
    paragraphs would price a floor that does not exist.

    Published per window, never as a scalar: on real multi-label data the floor
    drifts 29-53% across the corpus, so a time-constant floor is empirically
    false.

    Returns:
        `(mean pairwise JSD under re-dealt blocks, per-slot mean JSD to others)`.
    """
    sizes = np.array([len(p) for p in pools], dtype=np.int64)
    pooled = np.concatenate(list(pools))
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    n_slots = len(pools)

    order = np.argsort(rng.random((n_draws, len(pooled))), axis=1, kind="stable")
    dealt = pooled[order]

    S = design.n_speeches
    speech_ids = np.empty((n_draws, n_slots, S), dtype=np.int64)
    for e in range(n_slots):
        block = dealt[:, bounds[e]: bounds[e + 1]]
        pick = (rng.random((n_draws, S)) * sizes[e]).astype(np.int64)
        speech_ids[:, e, :] = np.take_along_axis(block, pick, axis=1)

    B = design.n_paragraphs
    para_pos = (
        rng.random((n_draws, n_slots, S, B)) * design.lens[speech_ids][..., None]
    ).astype(np.int64)
    rows = design.starts[speech_ids][..., None] + para_pos
    c = design.mass[rows.reshape(n_draws, n_slots, S * B)].mean(axis=2)
    j = pairwise_jsd(c)
    per_slot = (j.sum(axis=-1) / max(n_slots - 1, 1)).mean(axis=0)
    return float(_mean_offdiag(j).mean()), per_slot


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------


class Curve(NamedTuple):
    """One arm's dispersion curve on one window grid, plus everything derived."""

    start: np.ndarray
    end: np.ndarray
    center: np.ndarray
    n_eligible: np.ndarray
    status: list[str]
    used: np.ndarray
    dispersion: np.ndarray
    entropy_matched: np.ndarray
    floor: np.ndarray
    mean_entropy: np.ndarray
    # president-pair accumulators (prereg 7.6) and per-president accumulators
    # (prereg 7.2); both are sums over used windows and draws.
    pair_jsd: np.ndarray
    pair_year: np.ndarray
    pair_n: np.ndarray
    pres_jsd: np.ndarray
    pres_floor: np.ndarray
    pres_n: np.ndarray

    @property
    def floor_corrected(self) -> np.ndarray:
        return self.dispersion - self.floor

    @property
    def excess_ratio(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.entropy_matched > 0, self.dispersion / self.entropy_matched, np.nan
            )


def dispersion_curve(
    design: Design,
    seed: Sequence[int],
    donor: np.ndarray | None = None,
    drop: int | None = None,
    n_draws: int = DRAWS,
    floor_draws: int = FLOOR_DRAWS,
    with_floor: bool = True,
    with_entropy_match: bool = True,
    rarefaction: str = "cluster",
) -> Curve:
    """Compute one dispersion curve over the design's windows.

    Args:
        design: the fixed design (who is eligible where).
        seed: explicit RNG seed path; every draw in this module is seeded.
        donor: `donor[p]` = the president whose content fills president `p`'s
            slot. `None` = identity (the observed curve). Under a permutation a
            slot draws from the donor's GLOBAL qualifying pool, so the design is
            untouched and only the content moves.
        drop: a president index to delete from every window (the jackknife).
        rarefaction: `"cluster"` (the pre-registered estimator) or
            `"paragraph"` (the broken original, `_selftest` contrast only).
    """
    rng = np.random.default_rng(list(seed))
    n_pres = len(design.presidents)
    n_win = len(design.windows)

    out = {
        k: np.full(n_win, np.nan)
        for k in ("dispersion", "entropy_matched", "floor", "mean_entropy")
    }
    n_eligible = np.zeros(n_win, dtype=np.int64)
    status: list[str] = []
    used = np.zeros(n_win, dtype=bool)
    pair_jsd = np.zeros((n_pres, n_pres))
    pair_year = np.zeros((n_pres, n_pres))
    pair_n = np.zeros((n_pres, n_pres))
    pres_jsd = np.zeros(n_pres)
    pres_floor = np.zeros(n_pres)
    pres_n = np.zeros(n_pres)

    sampler = (
        draw_compositions if rarefaction == "cluster" else draw_compositions_paragraph
    )

    for wi, window in enumerate(design.windows):
        keep = [i for i, p in enumerate(window.presidents) if p != drop]
        pres = window.presidents[keep]
        n_eligible[wi] = len(pres)
        st = window_status(len(pres))
        status.append(st)
        if st in {"no_data", "suppressed_n_floor"}:
            continue
        used[wi] = True

        if donor is None:
            pools = [window.pools[i] for i in keep]
        else:
            pools = [design.global_pool[donor[p]] for p in pres]

        c = sampler(design, pools, rng, n_draws)
        j = pairwise_jsd(c)
        out["dispersion"][wi] = float(_mean_offdiag(j).mean())
        out["mean_entropy"][wi] = float(_entropy(c).mean())

        if with_entropy_match:
            # Bin-order shuffle: preserves each president's entropy EXACTLY (it
            # is a permutation of the same probability values) while destroying
            # any agreement between presidents. The result is the dispersion
            # those breadths alone predict, absent a shared agenda.
            perm = np.argsort(rng.random(c.shape), axis=-1, kind="stable")
            out["entropy_matched"][wi] = float(
                _mean_offdiag(pairwise_jsd(np.take_along_axis(c, perm, axis=-1))).mean()
            )

        floor_per_slot = None
        if with_floor:
            floor_value, floor_per_slot = speech_block_floor(
                design, pools, rng, floor_draws
            )
            out["floor"][wi] = floor_value

        per_slot = j.sum(axis=-1).mean(axis=0) / max(len(pres) - 1, 1)
        pres_jsd[pres] += per_slot
        pres_n[pres] += 1
        if floor_per_slot is not None:
            pres_floor[pres] += floor_per_slot

        mean_j = j.mean(axis=0)
        grid = np.ix_(pres, pres)
        pair_jsd[grid] += mean_j
        pair_year[grid] += window.center
        pair_n[grid] += 1

    return Curve(
        start=np.array([w.start for w in design.windows]),
        end=np.array([w.end for w in design.windows]),
        center=np.array([w.center for w in design.windows]),
        n_eligible=n_eligible, status=status, used=used,
        dispersion=out["dispersion"], entropy_matched=out["entropy_matched"],
        floor=out["floor"], mean_entropy=out["mean_entropy"],
        pair_jsd=pair_jsd, pair_year=pair_year, pair_n=pair_n,
        pres_jsd=pres_jsd, pres_floor=pres_floor, pres_n=pres_n,
    )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _series_for(curve: Curve, statistic: str) -> np.ndarray:
    if statistic == "dispersion":
        return curve.dispersion
    if statistic == "floor_corrected":
        return curve.floor_corrected
    if statistic == "excess_ratio":
        return curve.excess_ratio
    raise ValueError(f"unknown statistic {statistic!r}")


def ends_2014_mask(curve: Curve, window_kind: str) -> np.ndarray:
    """The H1 dating treatment (prereg 7.4), defined ONCE for both consumers.

    A trend-eligible window whose END year is at or before `DATING_END_YEAR`.
    The rule reads the window END, not its centre or start, so no window
    containing a Trump year can enter the dating trend. It exists on the rolling
    grid only — `ends_2014` is a subset of that grid — so any other window kind
    gets an all-False mask rather than a silently different treatment.

    This is the rule both `curve_statistics` (which forms the rho) and
    `_curve_rows` (which publishes `in_ends_2014` on every window row) apply.
    They had independent copies, so each needed its own mutant and its own
    guard, and the two could drift apart without any test noticing.
    """
    if window_kind != "rolling":
        return np.zeros(len(curve.center), dtype=bool)
    return curve.used & (curve.end <= DATING_END_YEAR)


def _pair_regression_rho(curve: Curve) -> float:
    """Prereg 7.6: mean pairwise JSD against mean shared-window centre year,
    one point per president pair. Power ~53%, stated in advance."""
    iu, ju = np.triu_indices(curve.pair_n.shape[0], k=1)
    n = curve.pair_n[iu, ju]
    ok = n > 0
    if ok.sum() < 3:
        return float("nan")
    return spearman_rho(
        curve.pair_year[iu, ju][ok] / n[ok], curve.pair_jsd[iu, ju][ok] / n[ok]
    )


def curve_statistics(curve: Curve, window_kind: str) -> dict[tuple[str, str], float]:
    """Every `(treatment, statistic)` rho this curve contributes.

    `ends_2014` exists only on the rolling grid (it is a subset of it); the
    pair regression is computed on the rolling grid only. Both scope rules are
    pre-registered (sections 7.4, 7.6, 7.7).
    """
    stats: dict[tuple[str, str], float] = {}
    treatments = ["all_windows"]
    masks = {"all_windows": curve.used}
    if window_kind == "rolling":
        treatments.append("ends_2014")
        masks["ends_2014"] = ends_2014_mask(curve, window_kind)
    for treatment in treatments:
        mask = masks[treatment]
        for statistic in ("dispersion", "floor_corrected", "excess_ratio"):
            series = _series_for(curve, statistic)
            stats[(treatment, statistic)] = spearman_rho(
                curve.center[mask], series[mask]
            )
    if window_kind == "rolling":
        stats[("all_windows", "pair_regression")] = _pair_regression_rho(curve)
    return stats


# --------------------------------------------------------------------------
# permutation null
# --------------------------------------------------------------------------


def _donor_map(design: Design, rng: np.random.Generator) -> np.ndarray:
    """A uniform random bijection over the ever-eligible presidents.

    A bijection, not an i.i.d. relabelling: a president must move as a UNIT so
    that the persistence of the same presidents across adjacent windows — the
    dominant source of autocorrelation in the curve — survives into the null.
    """
    pool = ever_eligible(design)
    donor = np.arange(len(design.presidents))
    donor[pool] = rng.permutation(pool)
    return donor


def permutation_null(
    designs: dict[str, Design],
    arm_index: int,
    n_permutations: int = N_PERMUTATIONS,
    n_draws: int = DRAWS,
    floor_draws: int = FLOOR_DRAWS,
    statistics: Sequence[str] = STATISTICS,
    progress: bool = False,
) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str, str], np.ndarray]]:
    """Full-pipeline president permutation.

    The DESIGN is held exactly fixed — who is eligible in which window, and the
    `ci_status` exclusions — and only the content filling each slot is permuted.
    That is what lets this null absorb window overlap (93% shared content, ~8
    independent blocks in 240 years), the drifting eligible-president count and
    the drifting speech supply, none of which any closed-form p-value sees.

    Returns:
        `(observed rho per (window_kind, treatment, statistic), null rho draws)`.
    """
    want_floor = "floor_corrected" in statistics
    want_match = "excess_ratio" in statistics

    observed: dict[tuple[str, str, str], float] = {}
    for kind, design in designs.items():
        curve = dispersion_curve(
            design, [MASTER_SEED, arm_index, WINDOW_KINDS.index(kind), 0],
            n_draws=n_draws, floor_draws=floor_draws,
            with_floor=want_floor, with_entropy_match=want_match,
        )
        for (treatment, statistic), rho in curve_statistics(curve, kind).items():
            if statistic in statistics:
                observed[(kind, treatment, statistic)] = rho

    # NaN-filled, not `np.empty`: every slot is written on a complete run, but
    # an unwritten slot in an `np.empty` array carries RECYCLED MEMORY straight
    # into `_one_sided_p` / `null_mean` / `null_sd` / `null_crit_05` (observed
    # under mutation as 1829.5 — a window-centre year from an earlier array).
    # With NaN an unwritten draw is structurally visible: it drops out of the
    # null and downgrades `ci_status` through `n_permutations_valid`.
    null = {k: np.full(n_permutations, np.nan) for k in observed}
    perm_rng = np.random.default_rng([MASTER_SEED, 7, arm_index])
    for r in range(n_permutations):
        if progress and r % 200 == 0:
            print(f"    permutation {r}/{n_permutations}", flush=True)
        for kind, design in designs.items():
            donor = _donor_map(design, perm_rng)
            curve = dispersion_curve(
                design, [MASTER_SEED, arm_index, WINDOW_KINDS.index(kind), r + 1],
                donor=donor, n_draws=n_draws, floor_draws=floor_draws,
                with_floor=want_floor, with_entropy_match=want_match,
            )
            for (treatment, statistic), rho in curve_statistics(curve, kind).items():
                if (kind, treatment, statistic) in null:
                    null[(kind, treatment, statistic)][r] = rho
    return observed, null


def _one_sided_p(observed: float, null: np.ndarray) -> float:
    """P(rho <= observed) under the null, with the standard +1 correction."""
    valid = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(valid) == 0:
        return float("nan")
    return float((1 + int((valid <= observed).sum())) / (len(valid) + 1))


def _two_sided_p(observed: float, null: np.ndarray) -> float:
    valid = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(valid) == 0:
        return float("nan")
    centre = float(np.median(valid))
    return float(
        (1 + int((np.abs(valid - centre) >= abs(observed - centre)).sum()))
        / (len(valid) + 1)
    )


def null_rows(
    arm: ArmSpec, observed: dict, null: dict, n_permutations: int
) -> list[dict]:
    rows = []
    for key, rho in observed.items():
        kind, treatment, statistic = key
        draws = null[key]
        valid = draws[np.isfinite(draws)]
        rows.append({
            "arm": arm.name,
            "role": arm.role,
            "window_kind": kind,
            "treatment": treatment,
            "statistic": statistic,
            "rho": rho,
            "p_one_sided": _one_sided_p(rho, draws),
            "p_two_sided": _two_sided_p(rho, draws),
            "significant_decline": bool(
                np.isfinite(rho) and rho < 0 and _one_sided_p(rho, draws) <= ALPHA
            ),
            "null_mean": float(valid.mean()) if len(valid) else np.nan,
            "null_sd": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            "null_crit_05": (
                float(np.percentile(valid, 100 * ALPHA)) if len(valid) else np.nan
            ),
            "n_permutations": n_permutations,
            "n_permutations_valid": int(len(valid)),
            "ci_status": null_status(rho, len(valid), n_permutations),
            "alpha_one_sided": ALPHA,
            "seed": MASTER_SEED,
            "p_source": "president_permutation",
            "scipy_pvalue_used": False,
        })
    return rows


# --------------------------------------------------------------------------
# jackknife, paradox
# --------------------------------------------------------------------------


def jackknife(
    design: Design, arm_index: int, null_draws: np.ndarray, n_draws: int = DRAWS,
    floor_draws: int = FLOOR_DRAWS,
) -> pd.DataFrame:
    """Leave-one-president-out, INCLUDING leave-Trump-out.

    Each jackknife rho is scored against the FULL-design permutation null for
    the arm. That is a declared approximation (prereg 7.1): 45 x 2,000
    re-permutations is not affordable, and the null's spread is a property of
    the design, which deleting one president barely perturbs. It is recorded on
    every row as `null_source`.

    "Barely perturbs" is exactly the assumption a reader must be able to check,
    so every row also carries `ci_status` (`jackknife_status`) beside
    `n_windows_used` / `n_windows_full_design`: a president whose deletion
    collapses the trend window set is flagged rather than silently plotted
    against a null that no longer describes its design.
    """
    rows = []
    n_windows_full = n_windows_in_trend(design)
    for p in ever_eligible(design):
        curve = dispersion_curve(
            design, [MASTER_SEED, arm_index, 0, 900_000 + int(p)], drop=int(p),
            n_draws=n_draws, floor_draws=floor_draws, with_floor=False,
            with_entropy_match=False,
        )
        rho = spearman_rho(curve.center[curve.used], curve.dispersion[curve.used])
        n_used = int(curve.used.sum())
        rows.append({
            "arm": design.arm,
            "left_out_president": design.presidents[p],
            "left_out_first_year": int(design.president_first_year[p]),
            "rho": rho,
            "p_one_sided": _one_sided_p(rho, null_draws),
            "significant_decline": bool(
                np.isfinite(rho) and rho < 0 and _one_sided_p(rho, null_draws) <= ALPHA
            ),
            "n_windows_used": n_used,
            "n_windows_full_design": n_windows_full,
            "ci_status": jackknife_status(n_used, n_windows_full),
            "null_source": "full_design_permutation_null",
            "p_source": "president_permutation",
            "scipy_pvalue_used": False,
        })
    return pd.DataFrame(rows)


def _stylistic_anchor(path: Path | None = None) -> pd.Series:
    """H2a ANCHOR: mean cosine similarity to the other 44 presidents.

    Already published in the README. Read, never recomputed, never scored — it
    carries zero evidential weight. Returns an empty Series (and the paradox
    table records the absence) if the embeddings artifact is missing, rather
    than fabricating a magnitude.
    """
    path = PRESIDENT_EMBEDDINGS_PATH if path is None else path
    if not path.exists():
        return pd.Series(dtype=float)
    emb = pd.read_parquet(path)
    vec_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    v = emb[vec_cols].to_numpy(dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    np.fill_diagonal(sim, np.nan)
    return pd.Series(np.nanmean(sim, axis=1), index=emb["president"].to_numpy())


def paradox_table(design: Design, curve: Curve, anchor: pd.Series | None = None) -> pd.DataFrame:
    """H2b: is each president agenda-TYPICAL, once the bias floor is removed?

    `d_hat = mean JSD to contemporaries - mean speech-block floor` over the
    president's eligible windows. The floor subtraction is what makes presidents
    from different eras comparable: an uncorrected distance ranks them partly by
    which era's topic entropy they happened to sit in.

    Under the president-permutation null a president's percentile of `d_hat` is
    uniform, so the one-sided p for H2b IS that percentile (prereg 7.2).
    """
    anchor = _stylistic_anchor() if anchor is None else anchor
    rows = []
    for p in range(len(design.presidents)):
        if curve.pres_n[p] == 0:
            continue
        n = curve.pres_n[p]
        mean_jsd = curve.pres_jsd[p] / n
        mean_floor = curve.pres_floor[p] / n
        rows.append({
            "arm": design.arm,
            "president": design.presidents[p],
            "first_year": int(design.president_first_year[p]),
            "n_windows_eligible": int(n),
            "mean_jsd_to_contemporaries": float(mean_jsd),
            "mean_speech_block_floor": float(mean_floor),
            "corrected_distance": float(mean_jsd - mean_floor),
        })
    out = pd.DataFrame(rows)
    d = out["corrected_distance"].to_numpy()
    out["rank_corrected"] = out["corrected_distance"].rank(method="min").astype(int)
    out["percentile_corrected"] = out["corrected_distance"].rank(pct=True)
    # Leave-one-out z: each president is compared against the OTHERS, so a
    # president cannot deflate the very mean and sd they are being scored on.
    n_all = len(out)
    others_mean = (d.sum() - d) / (n_all - 1)
    others_sd = np.array([
        d[np.arange(n_all) != i].std(ddof=1) for i in range(n_all)
    ])
    out["z_vs_others"] = (d - others_mean) / others_sd
    out["n_presidents_ranked"] = n_all
    out["ci_status"] = [paradox_status(n) for n in out["n_windows_eligible"]]
    out["stylistic_similarity_anchor"] = out["president"].map(anchor)
    out["anchor_is_scored"] = False
    out["anchor_source"] = (
        PRESIDENT_EMBEDDINGS_PATH.name if len(anchor) else f"absent:{PRESIDENT_EMBEDDINGS_PATH.name}"
    )
    out["p_source"] = "president_permutation"
    out["scipy_pvalue_used"] = False
    return out.sort_values("corrected_distance").reset_index(drop=True)


# --------------------------------------------------------------------------
# decision table (prereg section 8) — evaluated in order, first rule wins
# --------------------------------------------------------------------------

HEADLINES = {
    "artifact_arms_disagree": (
        "The two pre-registered arms disagree: one shows a significant decline in "
        "between-president agenda dispersion and the other does not, so this corpus "
        "cannot tell us whether presidential agendas converged."
    ),
    "no_convergence": (
        "Across 240 years we find no significant decline in between-president agenda "
        "dispersion; at 68% power this is weak evidence against convergence, not a "
        "refutation of it."
    ),
    "artifact_floor": (
        "Between-president agenda dispersion does decline, but the decline does not "
        "survive subtraction of the per-window noise floor, so it is a property of the "
        "measurement rather than of the presidencies."
    ),
    "convergence": (
        "Presidential agendas have converged: between-president dispersion of issue "
        "composition declines over 1789-2026 in both pre-registered arms, survives the "
        "per-window noise floor, and survives an entropy-matched null - presidents are "
        "more alike than their own broadening agendas can explain."
    ),
    "broadening": (
        "Between-president agenda dispersion declines, but an entropy-matched null "
        "accounts for it: presidents did not converge on a shared agenda so much as each "
        "of them started talking about everything."
    ),
}


def score_decision_table(null_table: pd.DataFrame) -> dict:
    """Apply prereg section 8, in order. The first rule that fires decides."""

    def sig(arm: str, statistic: str) -> bool:
        row = null_table[
            (null_table["arm"] == arm)
            & (null_table["window_kind"] == "rolling")
            & (null_table["treatment"] == "all_windows")
            & (null_table["statistic"] == statistic)
        ]
        if row.empty:
            raise ValueError(f"decision table needs ({arm}, {statistic}) and it is absent")
        return bool(row["significant_decline"].iloc[0])

    a_disp, c_disp = sig("corex_all", "dispersion"), sig("corex_sotu", "dispersion")
    if a_disp != c_disp:
        cell = "artifact_arms_disagree"
    elif not a_disp:
        cell = "no_convergence"
    elif not (sig("corex_all", "floor_corrected") and sig("corex_sotu", "floor_corrected")):
        cell = "artifact_floor"
    elif sig("corex_all", "excess_ratio") and sig("corex_sotu", "excess_ratio"):
        cell = "convergence"
    else:
        cell = "broadening"
    return {
        "cell": cell,
        "headline": HEADLINES[cell],
        "inputs": {
            "corex_all.dispersion": a_disp,
            "corex_sotu.dispersion": c_disp,
            "corex_all.floor_corrected": sig("corex_all", "floor_corrected"),
            "corex_sotu.floor_corrected": sig("corex_sotu", "floor_corrected"),
            "corex_all.excess_ratio": sig("corex_all", "excess_ratio"),
            "corex_sotu.excess_ratio": sig("corex_sotu", "excess_ratio"),
        },
    }


# --------------------------------------------------------------------------
# _selftest — the gate. Nothing real runs until all three legs pass.
# --------------------------------------------------------------------------


def _centered_tilts(
    rng: np.random.Generator, mu_p: np.ndarray, n: int, icc_alpha: float
) -> np.ndarray:
    """`n` speech tilts around `mu_p` whose MEAN is `mu_p`, for every `n`.

    This centring is a large part of why the leg-(a) corpus is a true null, and
    it is easy to skip. Draw `n` i.i.d. Dirichlet tilts and their mean sits
    `Var(theta)/n` from `mu_p`, so a president who gave 10 speeches has a
    genuinely more idiosyncratic *speech mix* than one who gave 38 — a real,
    time-varying difference that a generator would then be injecting under the
    name "null". Without this the cluster-rarefied estimator reads roughly
    -0.3 to -0.6 on a corpus advertised as having zero convergence, and leg (a)
    would be measuring the generator rather than the estimator.

    Three rounds of multiplicative (IPF-style) rescaling pull the realized tilt
    mean onto `mu_p` while keeping every tilt non-negative and preserving the
    within-speech clustering leg (a) exists to stress. What survives is the
    paragraph-level multinomial noise, which is exactly the noise floor the
    estimator is supposed to hold constant across eras.
    """
    theta = rng.dirichlet(icc_alpha * mu_p, size=n)
    for _ in range(3):
        theta = theta * (mu_p / np.maximum(theta.mean(axis=0), 1e-12))
        theta /= theta.sum(axis=1, keepdims=True)
    return theta


def _synthetic_composition(
    rng: np.random.Generator,
    n_presidents: int = 48,
    term_years: int = 5,
    first_year: int = 1789,
    n_bins: int = 8,
    speeches_first: int = 10,
    speeches_last: int = 38,
    paragraphs_per_speech: int = 34,
    icc_alpha: float = 9.0,
    convergence_strength: float = 0.0,
) -> Composition:
    """A synthetic corpus with the REAL design's confound built in.

    Every default is pinned to a MEASURED property of this corpus rather than
    tuned until the gate goes green: 48 presidents on 5-year terms (45
    presidents over 238 years), 34 paragraphs per speech (36,229 paragraphs over
    1,057 speeches = 34.3), a per-president in-window speech supply of 10 -> 38
    (the review's measured drift in distinct speeches behind 50 in-window
    paragraphs), and `icc_alpha = 9` for an intraclass correlation of 0.10,
    inside the 0.05-0.17 measured on the real issue labels.

    Four properties matter and all four are deliberate:
    * **within-speech clustering** — every speech carries a Dirichlet tilt and
      its paragraphs draw from that tilt, giving an intraclass correlation of
      about `1/(1+icc_alpha)`. The default 9.0 puts ICC at 0.10, inside the
      0.05-0.17 measured on this corpus's real issue labels;
    * **a growing speech supply** — a president's speech count rises linearly
      from `speeches_first` to `speeches_last`, matching the measured ~10 -> ~38
      drift in the number of distinct speeches behind 50 in-window paragraphs,
      which is the confound that breaks paragraph rarefaction;
    * **`convergence_strength = 0` means ZERO real convergence** — every
      president shares one time-invariant `mu`, AND (see `_centered_tilts`)
      every president's realized pool composition equals it regardless of how
      many speeches they gave. Any trend an estimator reports here is
      manufactured;
    * `convergence_strength > 0` gives each president an idiosyncratic tilt
      whose magnitude decays linearly to zero at the end of the timeline: real,
      injected convergence.
    """
    mu = np.full(n_bins, 1.0 / n_bins)
    doc_names, para_idx, presidents, years, labels = [], [], [], [], []
    counts = np.linspace(speeches_first, speeches_last, n_presidents).round().astype(int)
    for p in range(n_presidents):
        start = first_year + p * term_years
        name = f"P{p:03d}"
        if convergence_strength > 0:
            decay = 1.0 - p / max(n_presidents - 1, 1)
            tilt = rng.dirichlet(np.full(n_bins, 0.6))
            mu_p = (1 - convergence_strength * decay) * mu + convergence_strength * decay * tilt
        else:
            mu_p = mu
        thetas = _centered_tilts(rng, mu_p, int(counts[p]), icc_alpha)
        for s in range(int(counts[p])):
            year = start + s % term_years
            doc = f"{name}-s{s:04d}"
            picks = rng.choice(n_bins, size=paragraphs_per_speech, p=thetas[s])
            for i, b in enumerate(picks):
                doc_names.append(doc)
                para_idx.append(i)
                presidents.append(name)
                years.append(year)
                labels.append([int(b)])
    bins = [f"bin{j}" for j in range(n_bins)] + [NO_TOPIC]
    return Composition(
        label_source="synthetic",
        bins=bins,
        mass=_mass_from_label_lists(labels, len(bins)),
        doc_name=np.array(doc_names),
        para_idx=np.array(para_idx),
        president=np.array(presidents),
        year=np.array(years, dtype=np.int64),
        speech_type=np.array(["synthetic"] * len(doc_names)),
        n_label_assignments=len(labels),
    )


_SELFTEST_ARM = ArmSpec("selftest", "corex", None, 8, 6, "selftest")

# Legs (a) and (b) are averaged over this many independent synthetic corpora.
# A single corpus is not enough: the per-corpus spread of rho is about 0.11, so
# a one-corpus gate would pass or fail on the luck of one seed rather than on
# the estimator's behaviour. Averaging estimates the BIAS, which is what the
# legs are about.
SELFTEST_CORPORA = 12


def _selftest_design(comp: Composition, eligibility: str = "speeches") -> Design:
    return build_design(comp, _SELFTEST_ARM, "rolling", eligibility=eligibility)


def _selftest(verbose: bool = True) -> dict:
    """Three legs, all on synthetic data, all pre-registered (section 9).

    (a) flat on the CLUSTERED null — AND the paragraph-rarefied contrast must
        still be strongly negative on the same corpora, or leg (a) proves only
        the absence of a signal rather than the presence of the fix;
    (b) detects injected convergence;
    (c) permutation-null size ~ 5%.

    The leg-(a) contrast reconstructs the superseded design END TO END — its
    paragraph eligibility rule (m in-window paragraphs, no speech floor) as well
    as its sampler. Swapping only the sampler inside the new speech-floor rule
    would understate the bias, because the new rule already excludes the
    two-and-three-speech presidents where the design effect is worst.

    Raises:
        AssertionError: on any failing leg. The caller must then STOP: running
            an estimator that cannot pass its own null is exactly how the
            original design got here.
    """
    results: dict[str, object] = {}

    # ---- legs (a) and (b) -------------------------------------------------
    cluster_rhos, paragraph_rhos, injected_rhos = [], [], []
    cluster_windows: list[int] = []
    for k in range(SELFTEST_CORPORA):
        null_comp = _synthetic_composition(np.random.default_rng([MASTER_SEED, 101, k]))
        cluster = dispersion_curve(
            _selftest_design(null_comp), [MASTER_SEED, 101, k, 1],
            with_floor=False, with_entropy_match=False,
        )
        cluster_rhos.append(
            spearman_rho(cluster.center[cluster.used], cluster.dispersion[cluster.used])
        )
        cluster_windows.append(int(cluster.used.sum()))
        paragraph = dispersion_curve(
            _selftest_design(null_comp, eligibility="paragraphs"),
            [MASTER_SEED, 101, k, 2], with_floor=False, with_entropy_match=False,
            rarefaction="paragraph",
        )
        paragraph_rhos.append(
            spearman_rho(paragraph.center[paragraph.used], paragraph.dispersion[paragraph.used])
        )

        injected_comp = _synthetic_composition(
            np.random.default_rng([MASTER_SEED, 202, k]), convergence_strength=0.9
        )
        injected = dispersion_curve(
            _selftest_design(injected_comp), [MASTER_SEED, 202, k, 1],
            with_floor=False, with_entropy_match=False,
        )
        injected_rhos.append(
            spearman_rho(injected.center[injected.used], injected.dispersion[injected.used])
        )

    rho_cluster = float(np.mean(cluster_rhos))
    rho_paragraph = float(np.mean(paragraph_rhos))
    rho_injected = float(np.mean(injected_rhos))
    results["a_cluster_null_rho"] = rho_cluster
    results["a_cluster_null_rho_sd"] = float(np.std(cluster_rhos, ddof=1))
    results["a_paragraph_null_rho"] = rho_paragraph
    results["a_paragraph_null_rho_sd"] = float(np.std(paragraph_rhos, ddof=1))
    results["b_injected_rho"] = rho_injected
    results["b_injected_rho_sd"] = float(np.std(injected_rhos, ddof=1))
    results["corpora"] = SELFTEST_CORPORA
    # Named for what it actually is, and computed without leaking the loop
    # variable. This was previously published as `n_windows` read off the LAST
    # leg-(a) corpus's curve after the loop had ended: it read as a property of
    # the selftest as a whole, and is not one — the 12 corpora differ from each
    # other and leg (c) uses a shorter timeline entirely.
    results["a_corpus_windows_in_trend_min"] = int(min(cluster_windows))
    results["a_corpus_windows_in_trend_max"] = int(max(cluster_windows))

    # ---- leg (c): permutation-null size ----------------------------------
    hits, sizes_seen = 0, 0
    for r in range(SELFTEST_REPLICATES):
        # Half the timeline of legs (a)/(b) — 24 presidents, ~46 windows. Size
        # is a Monte-Carlo estimate over 40 replicate corpora x 150
        # permutations each, so the corpus is shortened rather than the
        # replicate count, which is what the precision of the estimate depends
        # on. Every other property (supply drift, ICC, speech length) is the
        # same generator as legs (a)/(b).
        comp = _synthetic_composition(
            np.random.default_rng([MASTER_SEED, 303, r]), n_presidents=24,
        )
        design = _selftest_design(comp)
        obs_curve = dispersion_curve(
            design, [MASTER_SEED, 303, r, 0], with_floor=False, with_entropy_match=False
        )
        rho_obs = spearman_rho(
            obs_curve.center[obs_curve.used], obs_curve.dispersion[obs_curve.used]
        )
        perm_rng = np.random.default_rng([MASTER_SEED, 404, r])
        # NaN-filled, never `np.empty` — same reason as `permutation_null`.
        draws = np.full(SELFTEST_PERMUTATIONS, np.nan)
        for k in range(SELFTEST_PERMUTATIONS):
            donor = _donor_map(design, perm_rng)
            c = dispersion_curve(
                design, [MASTER_SEED, 303, r, k + 1], donor=donor,
                with_floor=False, with_entropy_match=False,
            )
            draws[k] = spearman_rho(c.center[c.used], c.dispersion[c.used])
        sizes_seen += 1
        if _one_sided_p(rho_obs, draws) <= ALPHA:
            hits += 1
    size = hits / max(sizes_seen, 1)
    results["c_permutation_size"] = size
    results["c_replicates"] = sizes_seen
    results["c_permutations_each"] = SELFTEST_PERMUTATIONS

    checks = {
        "a_cluster_flat": abs(rho_cluster) <= SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO,
        "a_paragraph_contrast": rho_paragraph <= SELFTEST_PARAGRAPH_BIAS_MAX_RHO,
        "b_detects_injected": rho_injected <= SELFTEST_INJECTED_MAX_RHO,
        "c_size_at_nominal_5pct": size <= SELFTEST_MAX_SIZE,
    }
    results["checks"] = checks
    results["passed"] = all(checks.values())

    if verbose:
        print(f"=== _selftest (legs a/b averaged over {SELFTEST_CORPORA} corpora) ===")
        print(f"  (a) clustered null, CLUSTER-rarefied  rho = {rho_cluster:+.3f} "
              f"(pass |rho| <= {SELFTEST_CLUSTER_FLAT_MAX_ABS_RHO}) "
              f"-> {'PASS' if checks['a_cluster_flat'] else 'FAIL'}")
        print(f"  (a) same corpora, PARAGRAPH-rarefied  rho = {rho_paragraph:+.3f} "
              f"(pass rho <= {SELFTEST_PARAGRAPH_BIAS_MAX_RHO}) "
              f"-> {'PASS' if checks['a_paragraph_contrast'] else 'FAIL'}")
        print(f"  (b) injected convergence              rho = {rho_injected:+.3f} "
              f"(pass rho <= {SELFTEST_INJECTED_MAX_RHO}) "
              f"-> {'PASS' if checks['b_detects_injected'] else 'FAIL'}")
        print(f"  (c) permutation-null size             = {size:.3f} "
              f"({hits}/{sizes_seen} at alpha={ALPHA}; pass <= {SELFTEST_MAX_SIZE}) "
              f"-> {'PASS' if checks['c_size_at_nominal_5pct'] else 'FAIL'}")
        print(f"  overall: {'PASS' if results['passed'] else 'FAIL'}")

    if not results["passed"]:
        failed = [k for k, v in checks.items() if not v]
        raise AssertionError(
            f"_selftest FAILED on {failed}. STOP: running an estimator that "
            f"cannot pass its own null is exactly how the original design got "
            f"here. Do not proceed to the real run. Details: {results}"
        )
    return results


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _curve_rows(arm: ArmSpec, design: Design, curve: Curve) -> list[dict]:
    rows = []
    in_ends_2014 = ends_2014_mask(curve, design.window_kind)
    for i in range(len(curve.center)):
        rows.append({
            "arm": arm.name,
            "role": arm.role,
            "window_kind": design.window_kind,
            "window_start": int(curve.start[i]),
            "window_end": int(curve.end[i]),
            "window_center": float(curve.center[i]),
            "n_eligible_presidents": int(curve.n_eligible[i]),
            "ci_status": curve.status[i],
            "used_in_trend": bool(curve.used[i]),
            "in_ends_2014": bool(in_ends_2014[i]),
            "dispersion": float(curve.dispersion[i]),
            "entropy_matched": float(curve.entropy_matched[i]),
            "excess_ratio": float(curve.excess_ratio[i]),
            "floor": float(curve.floor[i]),
            "floor_corrected": float(curve.floor_corrected[i]),
            "mean_within_president_entropy": float(curve.mean_entropy[i]),
            "n_bins": design.n_bins,
            "s_speeches": design.n_speeches,
            "b_paragraphs": design.n_paragraphs,
            "draws": DRAWS,
            "floor_draws": FLOOR_DRAWS,
            "seed": MASTER_SEED,
            "ci_components": "permutation_trend_only",
        })
    return rows


SELFTEST_BYPASS_TOKEN = "yes-i-know-this-run-is-not-publishable"


def build_convergence(
    out_dir: Path | None = None,
    n_permutations: int = N_PERMUTATIONS,
    progress: bool = True,
    selftest_bypass: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the whole pre-registered analysis and write `data/convergence/`.

    `_selftest` gates everything: if any leg fails this raises before a single
    real number is computed and before `out_dir` is even created, so no partial
    artifact can be left behind. That is the pre-registered bail condition —
    running an estimator that cannot pass its own null is exactly how the
    original design got here.

    `selftest_bypass` exists only for tests and development. It is deliberately
    awkward: it must equal `SELFTEST_BYPASS_TOKEN` verbatim (a bare `True` will
    not do), and a bypassed run is **forbidden from writing the published
    layer** — `out_dir` must be given and must not be `CONVERGENCE_DIR`. The
    meta of a bypassed run records `publishable: False`. There is no way to put
    an ungated number into `data/convergence/`.

    Raises:
        AssertionError: from `_selftest`, on any failing leg.
        ValueError: on a malformed or mis-targeted `selftest_bypass`.
    """
    if selftest_bypass is None:
        selftest = _selftest(verbose=progress)
    else:
        if selftest_bypass != SELFTEST_BYPASS_TOKEN:
            raise ValueError(
                "selftest_bypass must be SELFTEST_BYPASS_TOKEN verbatim; "
                f"got {selftest_bypass!r}. The gate is not a boolean flag on "
                "purpose — skipping it must be an obviously deliberate act."
            )
        if out_dir is None or Path(out_dir).resolve() == CONVERGENCE_DIR.resolve():
            raise ValueError(
                "a selftest-bypassed run may never write the published layer "
                f"({CONVERGENCE_DIR}); pass an explicit scratch out_dir."
            )
        selftest = {
            "ran": False,
            "bypassed": True,
            "publishable": False,
            "note": (
                "the three-leg gate was bypassed; no number in this directory "
                "is publishable"
            ),
        }
    out_dir = CONVERGENCE_DIR if out_dir is None else Path(out_dir)

    compositions = {
        source: build_compositions(source)
        for source in sorted({a.label_source for a in ARMS})
    }

    curve_rows: list[dict] = []
    null_rows_all: list[dict] = []
    jackknife_frames: list[pd.DataFrame] = []
    paradox_frame: pd.DataFrame | None = None
    coverage: dict[str, dict] = {}

    for arm_index, arm in enumerate(ARMS):
        comp = compositions[arm.label_source]
        designs = {kind: build_design(comp, arm, kind) for kind in WINDOW_KINDS}
        if progress:
            print(f"\n=== arm {arm.name} ({arm.role}) ===", flush=True)

        observed, null = permutation_null(
            designs, arm_index, n_permutations=n_permutations, progress=progress
        )
        null_rows_all += null_rows(arm, observed, null, n_permutations)

        curves: dict[str, Curve] = {}
        for kind, design in designs.items():
            curve = dispersion_curve(
                design, [MASTER_SEED, arm_index, WINDOW_KINDS.index(kind), 0]
            )
            curve_rows += _curve_rows(arm, design, curve)
            curves[kind] = curve
        # Bound explicitly, not left leaking out of the loop above: the previous
        # spelling worked only because `WINDOW_KINDS[0] == "rolling"` and would
        # have raised `NameError` at a distance if that tuple were reordered.
        rolling_curve, rolling_design = curves["rolling"], designs["rolling"]

        used = np.array([r["used_in_trend"] for r in curve_rows if
                         r["arm"] == arm.name and r["window_kind"] == "rolling"])
        n_elig = np.array([r["n_eligible_presidents"] for r in curve_rows if
                           r["arm"] == arm.name and r["window_kind"] == "rolling"])
        coverage[arm.name] = {
            "n_windows_rolling": int(len(used)),
            "n_windows_used": int(used.sum()),
            "n_eligible_min": int(n_elig.min()),
            "n_eligible_median": float(np.median(n_elig)),
            "n_eligible_max": int(n_elig.max()),
            "n_presidents_ever_eligible": int(len(ever_eligible(designs["rolling"]))),
            "n_bins": designs["rolling"].n_bins,
            "n_qualifying_speeches": int(len(designs["rolling"].qualifying)),
        }

        jackknife_frames.append(
            jackknife(
                rolling_design, arm_index,
                null[("rolling", "all_windows", "dispersion")],
            )
        )
        if arm.role == "primary":
            paradox_frame = paradox_table(rolling_design, rolling_curve)

    tables = {
        "dispersion_curves": pd.DataFrame(curve_rows),
        "permutation_null": pd.DataFrame(null_rows_all),
        "jackknife": pd.concat(jackknife_frames, ignore_index=True),
        "paradox": paradox_frame,
    }

    decision = score_decision_table(tables["permutation_null"])

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out_dir / OUTPUT_PATHS[name].name, index=False)

    meta = {
        # No wall-clock stamp, on purpose (CLAUDE.md): every output here is a
        # pure function of the frozen inputs and the recorded seeds, so a dirty
        # `git status data/convergence/` must mean the numbers moved. Provenance
        # identity is `corpus_fingerprint`; when it ran is git's job.
        "preregistration": "notes/convergence-prereg-v1.md",
        "preregistration_commit_precedes_results": True,
        "arms": [
            {"name": a.name, "role": a.role, "label_source": a.label_source,
             "speech_types": list(a.speech_types) if a.speech_types else "all",
             "s_speeches": a.n_speeches, "b_paragraphs": a.n_paragraphs}
            for a in ARMS
        ],
        "estimator": {
            "rarefaction": "cluster (S speeches x B paragraphs, both with replacement)",
            "m_paragraphs": ARMS[0].n_speeches * ARMS[0].n_paragraphs,
            "draws_per_window": DRAWS,
            "floor_draws_per_window": FLOOR_DRAWS,
            "metric": "mean pairwise Jensen-Shannon divergence, base 2",
            "composition": (
                "c[j] = (1/m) sum_i [ 1{k_i>=1} 1{j in L_i}/k_i + "
                "1{k_i=0} 1{j = no-topic} ]; every paragraph contributes mass 1"
            ),
            "no_topic_bin": NO_TOPIC,
        },
        "windows": {
            "length": WINDOW_LEN, "step": WINDOW_STEP,
            "min_presidents_for_trend": MIN_PRESIDENTS,
            "kinds": list(WINDOW_KINDS),
            "dating_end_year": DATING_END_YEAR,
        },
        "inference": {
            "null": "full-pipeline president permutation (bijection over donors)",
            "n_permutations": n_permutations,
            "alpha_one_sided": ALPHA,
            "scipy_spearmanr_pvalue": (
                "BANNED — measured type-I error 58% on these overlapping windows; "
                "spearman_rho() discards it at the call site"
            ),
            "power_statement": (
                "At the permutation-calibrated critical value this design has 68% "
                "power against the effect size it is hunting. A null result is "
                "weakly informative, not a refutation."
            ),
        },
        "seeds": {
            "master": MASTER_SEED,
            "scheme": (
                "np.random.default_rng([MASTER_SEED, arm_index, window_kind_index, "
                "permutation_index]); permutation donor maps from "
                "[MASTER_SEED, 7, arm_index]"
            ),
        },
        "coverage": coverage,
        "decision": decision,
        "selftest": selftest,
        "label_normalization": {
            "normalizer": "attention.canonical_label_map + attention.normalize_topics",
            "raises_on_unmapped_label": True,
            "llm_assignments_post_dedup": int(
                compositions["llm"].n_label_assignments
            ) if "llm" in compositions else None,
        },
        "genre_other_bucket": (
            "the 8 `other`-typed speeches enter corex_all and llm_all and are "
            "excluded from corex_sotu, which is defined by one positive speech type"
        ),
        "limitations": [
            "llm_all uses native taxonomy_v1 level-2 labels, avoiding triangulate's "
            "union projection; its dispersion LEVEL is still not comparable to a "
            "CorEx arm's (different bin count and budget)",
            "taxonomy_v1 is frozen: no standalone Education topic, no modern monetary "
            "policy, no Prohibition, no standalone Terrorism; 7 of 50 level-2 topics "
            "have no legacy parent",
            "no annotator-disagreement component is propagated into these curves",
            "indices.py MARKERS are not used anywhere here (era-biased regexes)",
        ],
        "corpus_fingerprint": corpus_fingerprint(),
        "staleness": check_staleness(),
        "api_calls": 0,
    }
    (out_dir / META_PATH.name).write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return tables


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Pre-registered agenda-convergence test (pure local compute; "
                    "$0, no API calls)."
    )
    ap.add_argument("--selftest-only", action="store_true",
                    help="run the three-leg gate and stop")
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS,
                    help="override R (the published artifact uses the "
                         "pre-registered 2000; lower values are for development "
                         "only and are recorded in the meta)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest_only:
        _selftest(verbose=not args.quiet)
        return

    tables = build_convergence(
        n_permutations=args.permutations, progress=not args.quiet
    )
    if args.quiet:
        return

    pd.set_option("display.width", 220)
    print("\n=== permutation null (rolling, all windows) ===")
    n = tables["permutation_null"]
    print(
        n[(n["window_kind"] == "rolling") & (n["treatment"] == "all_windows")][
            ["arm", "statistic", "rho", "p_one_sided", "null_crit_05",
             "significant_decline"]
        ].round(4).to_string(index=False)
    )
    decision = score_decision_table(n)
    print(f"\n=== decision cell: {decision['cell']} ===")
    print(decision["headline"])
    print(f"\nWrote {len(tables)} tables to {CONVERGENCE_DIR}")


if __name__ == "__main__":
    main()
