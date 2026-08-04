"""Confidence bands for the issue/topic trend charts.

Every issue trend line on the site was a bare line: a 1785 point resting on two
speeches looked exactly like a 1965 point resting on ninety. This module builds
`data/bands.parquet` — one row per (surface, series, period) carrying a point
estimate, a sampling-only interval, a composed interval, and the provenance a
consumer needs to know which of the two it is looking at.

Everything here is FREE local compute over frozen artifacts. **No Anthropic API
calls, ever.** Rerunning is $0 and byte-identical: there is no wall-clock stamp
in any output, so a dirty `git status data/bands.parquet` means the numbers
moved, not that the clock did (the `data/combat/` rule).

--------------------------------------------------------------------------
Two surfaces, two different uncertainty budgets — and why they differ
--------------------------------------------------------------------------

**Surface A — `corex_issues`** (`fig_issue_timeline`, `fig_issues_decade`).
These charts plot `issues.PARA_LABELS_PATH`, whose labels come from a seeded
CorEx anchored topic model (`issues.py`, `ct.Corex(..., seed=42)`). That is a
deterministic classifier run over the full corpus. **There is no LLM annotator
anywhere in its pipeline**, so annotator disagreement is not a component that is
*missing* from these rows — it is **not applicable** to them. Every Surface A
row therefore carries `ci_components="sampling_only"` and
`disagreement_status="not_applicable_no_annotator_in_pipeline"`, permanently.

`data/llm_annotations/agreement_v1.parquet` (Sonnet-5 vs Opus-4.8) measures a
*different labeling system* over the same paragraphs. Widening a CorEx share by
an LLM annotator's jaccard would attach a measured magnitude to a quantity it
does not describe — which is the fabrication this repo's seam convention exists
to prevent. It is deliberately not read here.

**Surface B — `llm_topics`.** Here the labels ARE the LLM annotations
(`paragraph_annotations.topics`, level-2 of `taxonomy_v1`), so a second
annotator's disagreement is genuine uncertainty about the plotted quantity.
These rows carry `ci_components="sampling+annotator_disagreement"`.

--------------------------------------------------------------------------
How each component is computed
--------------------------------------------------------------------------

**Sampling noise — speech-clustered, both surfaces.** Speeches, not paragraphs,
are resampled with replacement: within-speech ICC on paragraph-level labels runs
to 0.17, so a paragraph bootstrap would report an interval several times too
narrow. Surface B does not re-derive this at all — it reuses
`attention.bootstrap_era_shares`, which is already the seeded, byte-reproducible,
speech-clustered bootstrap for exactly this quantity. Surface A needs its own
(different labels, different grain: 5-year periods) and mirrors that function's
clustering and seeding discipline rather than inventing its own.

Surface A's RNG is seeded per period (`default_rng([BOOTSTRAP_SEED, period])`),
so the draws for 1850 do not depend on how many periods were processed before
it — adding or dropping a period cannot silently move another period's interval.

**Annotator disagreement — measured, never converted.** The half-width is
computed HERE, from the two paired annotation tables:

    disagreement_half_width(era, topic)
        = |share_primary(era, topic) - share_secondary(era, topic)| / 2

over the 8,570 paragraphs both `paragraph_annotations` (Sonnet 5) and
`paragraph_annotations__opus4-8` (Opus 4.8) labeled. This is the half-range of
the two annotators' estimates of the *same plotted quantity*, in the *same
units* (fraction of paragraphs), on the *same axis* (`trends.ERAS`).

That choice is a deliberate deviation from the original plan, which specified
reading the per-era half-width off `agreement_v1.parquet`. Two things make that
table unusable as a band width, and neither is fixable by a better join:
  * its topic metric is a **jaccard** (0.678 corpus-wide), a set-overlap
    coefficient with no conversion to percentage points of paragraph share —
    any conversion factor would be invented, i.e. a fabricated magnitude;
  * it is keyed on `taxonomy.ERA_SPAN` 30-year sampling bins, not the
    `trends.ERAS` reporting axis, and the two do not nest.
Measuring the difference directly removes both problems: there is no cross-axis
mapping in this module, because the disagreement is computed on the reporting
axis to begin with. `agreement_v1.parquet` is not read.

**What that half-width honestly is — and it errs in BOTH directions.** Three
caveats, stated together because disclosing only the inflating one would leave a
reader with a one-sided picture of the error:

  1. *Inflating.* The two annotators labeled an 8,570-paragraph subsample, so
     the observed difference between their era shares carries the subsample's
     own sampling noise on top of the genuine annotator effect. That makes the
     half-width upper-leaning — the safe direction for a band. Not corrected for.
  2. *Deflating.* The band is centred on the PRIMARY annotator's point estimate
     (`point` is the primary's share; the primary table is the one the whole
     site plots), and it moves by `±half_width` = half the gap between the two
     annotators. So the composed interval reaches exactly HALFWAY toward the
     secondary annotator's estimate and **never covers it**. A band that had to
     contain both readings would use the full `|difference|`, not half of it.
     This is a deliberate convention — the interval is "how uncertain is the
     primary estimate", not "what is the envelope of all annotators" — but it is
     a real term pulling the other way from (1), and the two do not cancel in
     any principled amount.
  3. *Transfer assumption.* The paired set is a DOCUMENT-level sample: the
     second annotator ran over `agreement_sample_v1.json`'s sampled speeches
     (`agreement.py::_load_joined` restricts to `sample_docs`), not over a
     paragraph-level random sample of the corpus. That file lists 266
     sampled speeches; 262 of them carry secondary annotations and so reach
     this module (four were lost in the batch recovery documented in CLAUDE.md's
     Batches-API section). Those 262 speeches hold 8,570 of the corpus's 36,229
     paragraphs — 23.7% — and their measured disagreement is assumed to
     transfer to the full-corpus era intervals it widens. Within-speech
     clustering is exactly why the sampling bootstrap resamples speeches; the
     same clustering applies here and is NOT modelled in the half-width.
     `paired_coverage()` recomputes all of these from the artifacts.

The paired sample covers every one of the 9 eras (thinnest: The founding, 219
paragraphs), so no era falls back to a sampling-only interval today; the
`MIN_PAIRED_PARAGRAPHS` floor below is a live guard, not a formality, and an era
that fell under it would be published as `sampling_only` rather than given a
half-width estimated off a handful of rows.

**Composition.** `lo = lo_sampling - half_width`, `hi = hi_sampling + half_width`,
clipped to [0, 1] — the same widen-never-narrow rule as
`combat._apply_agreement_bands`. On Surface A the composed interval is by
construction identical to the sampling interval, and `lo_sampling`/`hi_sampling`
are published alongside `lo`/`hi` on BOTH surfaces so a consumer can strip the
disagreement component and see exactly what it contributed.

`disagreement_half_width` records the half-width that was APPLIED, so it is null
wherever `disagreement_band_applied` is False — including the `no_interval_to_widen`
case, where a half-width WAS measured but had no sampling interval to widen. The
column is the band's provenance, not an archive of the measurement; the measured
value stays recoverable from `disagreement_half_widths()`, which is a pure
function of the frozen annotation tables.

--------------------------------------------------------------------------
`interval_unresolvable` — the per-cell gate at the bottom of the cluster floor
--------------------------------------------------------------------------

A bootstrap over `n` clusters has only `C(2n-1, n)` distinct resamples: 3 at
n=2, 10 at n=3, 35 at n=4. At the corpus's thinnest period (1785, two speeches)
a series with zero paragraphs in BOTH speeches therefore produces the same
replicate every draw, and its percentile interval collapses to `lo = hi = 0` —
which renders as a confident zero when what it means is "two speeches told us
nothing". That is NOT the same object as the legitimate `lo = hi = 0` cells
where 13-14 speeches genuinely never touched an issue. **Before this gate runs**
the Surface A bootstrap produces 19 zero-width cells; 7 of them (1790, 1795,
1800, 1810 — all `ci_status == "ok"`) are exactly that legitimate finding and are
published untouched, and the gate below withdraws the other 12. So the SHIPPED
table contains 7 zero-width Surface A cells, not 19 — 19 is the pre-gate census,
and quoting it as a property of the artifact is a tense error, not an arithmetic
one.

**The gate is per-cell, and combines "no variation" with "too few clusters".**
`interval_unresolvable` is True where the bootstrap returned a zero-width
interval AND the period has fewer than `MIN_CLUSTERS_FOR_RESOLVABLE_CI`
clusters. Those cells publish `lo = hi = NaN` (and `lo_sampling`/`hi_sampling`
NaN with them) — the interval is withdrawn, not narrowed. `point` is kept: the
observed share is a real measurement; only the interval was unresolvable.

`MIN_CLUSTERS_FOR_RESOLVABLE_CI` is DERIVED, not chosen (see
`_min_clusters_for_resolvable_ci`), and it is a cluster count rather than
`ci_status`. Two reasons for both choices:

  * `ci_status` fires on `n_speeches < 8` **or** `n_paragraphs < 40`, so the
    otherwise-tempting predicate `ci_status == "low_cluster_caution" and
    lo == hi` would also suppress a paragraph-thin but *speech-rich* period,
    where the resample space is enormous and an all-zero result is a genuine
    finding. No such period exists in today's corpus, so the two predicates
    select the same 12 cells — but only one of them says what it means.
  * The floor itself falls out of the percentile the interval quotes. The most
    concentrated resample (all `n` draws landing on one cluster) carries
    probability `n**-n`: 0.25 at n=2, 0.037 at n=3, 0.0039 at n=4. Until that
    is below `CI_LOW/100 = 0.025`, the 2.5th percentile cannot exclude even the
    single most extreme resample, so the interval reports the full range of the
    resample space rather than a 95% region. n=4 is the first n where it can —
    hence a floor of 4, computed from `CI_LOW` so a retuned percentile moves it
    instead of silently invalidating it.

Note what is deliberately NOT done: `MIN_CLUSTERS_FOR_CI` stays at 2. Raising it
would delete the 1785 period entirely, and CLAUDE.md's paragraph-rate lesson is
explicit that raising a floor trades an interval problem for a worse
representativeness one. Discipline the interval; the point estimate was never
the defect.

--------------------------------------------------------------------------
`ci_status` — the trust gate
--------------------------------------------------------------------------

Never read `point` without it (the `data/combat/` rule):

  * `suppressed_n_floor` — fewer than `MIN_CLUSTERS_FOR_CI` speech clusters in
    the period. A percentile bootstrap over one cluster resamples the same
    speech every draw and returns a **zero-width** interval, which would be a
    lie in exactly the wrong direction. `lo`/`hi` are NaN and no band is drawn.
  * `low_cluster_caution` — the period exists but is thin: fewer than
    `LOW_CLUSTER_CAUTION` speeches or fewer than `MIN_PERIOD_PARAGRAPHS`
    paragraphs. The interval is real; it will be very wide.
  * `ok` — otherwise.

`MIN_PERIOD_PARAGRAPHS = 40` is the charts' former hard mask. It dropped exactly
ONE of the 49 five-year periods corpus-wide (1785, 14 paragraphs, 2 speeches).
It is now a de-emphasis flag rather than a drop, so absence of data is drawn as
absence of data instead of as absence of interest — but that recovers one
period, and should not be described as more.

**De-emphasis is wider than recovery, and both directions must be stated.**
`LOW_CLUSTER_CAUTION = 8` is a NEW gate that the old mask did not have, and it
demotes a period the old charts drew as an ordinary solid segment: 1820
(277 paragraphs — well clear of the 40-paragraph mask — but only 7 speeches).
So the change recovers one period (1785) and newly marks one previously-confident
period as provisional (1820); two of 49 periods are drawn dotted. Reporting only
the recovery would state the flattering half of a two-sided change, which is
exactly CLAUDE.md's "completeness cuts both ways".

No weighted average is formed here, so the `combat.py` effective-vs-total floor
trap does not arise: Surface A's estimator is an unweighted pooled share within
a period, and its floor binds on the count that actually drives the resample
(`n_speeches`, the cluster count) rather than on `n_paragraphs`. Both are
published.
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from . import attention, combat, issues
from .llm_annotations import corpus_fingerprint, load_paragraph_annotations
from .taxonomy import _require_full_merge
from .trends import ERAS

DATA_DIR = issues.DATA_DIR
BANDS_PATH = DATA_DIR / "bands.parquet"
BANDS_META_PATH = DATA_DIR / "bands_meta.json"

SECONDARY_ANNOTATIONS = "paragraph_annotations__opus4-8"
PRIMARY_ANNOTATIONS = "paragraph_annotations"
SAMPLE_PATH = DATA_DIR / "llm_annotations" / "agreement_sample_v1.json"

COREX_SURFACE = "corex_issues"
LLM_SURFACE = "llm_topics"

# ---- pre-registered constants (do not tune to taste) ----------------------
PERIOD_YEARS = 5
BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 20260721
CI_LOW, CI_HIGH = 2.5, 97.5
# NOT the same numbers as `combat.MIN_CLUSTERS_FOR_CI` (5) and
# `combat.LOW_CLUSTER_CAUTION` (20), and deliberately so — a consumer reading
# `ci_status` across `data/combat/` and `data/bands.parquet` must not assume one
# gate. combat's floors guard an ERA-grain rate (an era spans decades; 20 speech
# clusters is a low bar there, and its thinnest cell is a genuine n=3 warning).
# These guard a FIVE-YEAR period, where the corpus median is 20 speeches per
# period — so combat's floor of 20 lands ON the median and would mark 24 of the
# 49 periods, roughly half the record, provisional. A caution flag that fires on
# half the chart carries no information. Same column name, same direction, different
# thresholds; `bands_meta.json` records both sets so the divergence is legible
# from the artifact and not only from this comment.
MIN_CLUSTERS_FOR_CI = 2
LOW_CLUSTER_CAUTION = 8
MIN_PERIOD_PARAGRAPHS = 40
MIN_PAIRED_PARAGRAPHS = 50


def distinct_resamples(n_clusters: int) -> int:
    """`C(2n-1, n)` — how many distinct multisets an n-cluster bootstrap has.

    3 at n=2, 10 at n=3, 35 at n=4, 92,378 at n=10. Published in
    `bands_meta.json` so the degeneracy gate's premise is legible from the
    artifact and not only from this module.
    """
    return int(comb(2 * n_clusters - 1, n_clusters))


# Loop bound for `_min_clusters_for_resolvable_ci`. Unreachable for any
# positive `ci_low`: `n**-n` first evaluates to exactly 0.0 at n=149, and 0.0 is
# not greater than any positive threshold, so the search always terminates well
# below this. Present as a second line of defence, not as a working limit.
_MAX_CLUSTER_SEARCH = 1000


def _min_clusters_for_resolvable_ci(ci_low: float = CI_LOW) -> int:
    """Fewest clusters at which a `CI_LOW` percentile can exclude anything.

    The most concentrated resample — all `n` draws landing on one cluster —
    carries probability `n**-n` (0.25 at n=2, 0.037 at n=3, 0.0039 at n=4).
    While that exceeds `ci_low`, the lower percentile cannot fall inside the
    resample space's own range: the "95% interval" is just the full span of the
    resamples, and when every resample agrees it is a zero-width span reported
    as certainty.

    Derived from `CI_LOW` rather than typed as a literal so that retuning the
    percentile moves the floor with it instead of quietly invalidating it. On
    today's constants this returns 4.

    Args:
        ci_low: Lower percentile, in percent. Must be positive.

    Raises:
        ValueError: If `ci_low` is not positive. Written as `not (ci_low > 0)`
            so it also catches NaN, which is the more dangerous input of the
            two: every comparison against NaN is False, so the loop would exit
            on its first test and silently return `MIN_CLUSTERS_FOR_CI` — a
            plausible-looking floor derived from nothing. A non-positive
            `ci_low` fails louder (the loop never exits) but no less wrongly.
    """
    if not (ci_low > 0):
        raise ValueError(
            f"ci_low must be a positive percentile, got {ci_low!r}; there is no "
            "n at which a non-positive percentile can exclude a resample, so "
            "the floor this derives is undefined rather than large."
        )
    n = MIN_CLUSTERS_FOR_CI
    while n ** (-n) > ci_low / 100.0:
        n += 1
        if n > _MAX_CLUSTER_SEARCH:
            raise ValueError(
                f"no cluster count <= {_MAX_CLUSTER_SEARCH} makes n**-n fall "
                f"below ci_low/100 = {ci_low / 100.0!r}. This should be "
                "unreachable for a positive ci_low; treat it as a bug in the "
                "derivation rather than as a floor to raise."
            )
    return n


# A per-CELL floor, and a different quantity from MIN_CLUSTERS_FOR_CI: that one
# decides whether a bootstrap runs at all, this one decides whether a zero-width
# result from a bootstrap that DID run can be believed. Never used alone — see
# `_unresolvable_interval`, which requires zero width as well.
MIN_CLUSTERS_FOR_RESOLVABLE_CI = _min_clusters_for_resolvable_ci()

LLM_TREATMENT = "raw"
LLM_LEVEL = "level2"

NOT_APPLICABLE = "not_applicable_no_annotator_in_pipeline"
MEASURED = "measured_paired_annotators"
THIN_PAIRED = "unavailable_thin_paired_sample"
# A half-width exists but there is no sampling interval to widen (the cell's
# bootstrap was suppressed). Distinct from THIN_PAIRED: the annotator component
# was measurable, the sampling one was not — reporting the paired sample as thin
# there would name the wrong cause.
NO_INTERVAL = "no_interval_to_widen"

COREX_AGREEMENT_SOURCE = "n/a: CorEx labels have no annotator"
LLM_AGREEMENT_SOURCE = (
    f"{PRIMARY_ANNOTATIONS}.parquet vs {SECONDARY_ANNOTATIONS}.parquet"
)

# Fixed column order, for the same reason `combat.BAND_PROVENANCE_COLUMNS`
# exists: the two surfaces are assembled by different code paths, and a
# positional reader must not be broken by whichever one runs first.
BANDS_COLUMNS = [
    "surface",
    "series",
    "period_kind",
    "period",
    "period_order",
    "period_start",
    "period_end",
    "x",
    "point",
    "lo_sampling",
    "hi_sampling",
    "lo",
    "hi",
    "n_paragraphs",
    "n_speeches",
    "n_paired_paragraphs",
    "ci_status",
    "interval_unresolvable",
    "disagreement_band_applied",
    "disagreement_half_width",
    "ci_components",
    "disagreement_status",
    "agreement_source",
]


# --------------------------------------------------------------------------
# Surface A: CorEx issue labels, 5-year periods
# --------------------------------------------------------------------------


def corex_issue_columns(labels: pd.DataFrame) -> list[str]:
    """The boolean issue columns of `paragraph_issues`, in table order.

    Read off the parquet rather than from `issues.build_issues()`, which fits a
    full CorEx model — far too heavy for a presentation-layer builder, and it
    would make this module's output depend on a model refit rather than on the
    frozen labels the charts actually plot.
    """
    return [c for c in labels.columns if labels[c].dtype == bool]


def _ci_status(n_speeches: int, n_paragraphs: int) -> str:
    """The trust gate, shared by both surfaces — and its thresholds were
    calibrated on Surface A's 5-year periods.

    Surface B reuses it unchanged on the far coarser `trends.ERAS` grain, where
    every era clears both floors comfortably. The two floors bind on different
    eras and the sentence has to say so: the thinnest era by speeches is War &
    New Deal at 55, the thinnest by paragraphs is The founding at 927, against
    floors of 8 and 40 — so all 450 LLM rows publish `ok`. On that grain it is
    therefore a live-but-slack guard, and it is deliberately NOT retuned per
    grain: two different definitions of `ci_status` inside one table would make
    the column unreadable to the consumer that keys on it.
    """
    if n_speeches < MIN_CLUSTERS_FOR_CI:
        return "suppressed_n_floor"
    if n_speeches < LOW_CLUSTER_CAUTION or n_paragraphs < MIN_PERIOD_PARAGRAPHS:
        return "low_cluster_caution"
    return "ok"


def _unresolvable_interval(
    lo: np.ndarray, hi: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Per-cell mask: the bootstrap ran, returned zero width, and could not have
    returned anything else.

    BOTH conditions are required, and the conjunction is the whole point. Zero
    width alone would suppress the legitimate confident zeros — of the 19
    zero-width Surface A cells this gate is shown (the PRE-gate census; the
    shipped table keeps 7), 7 sit in `ci_status == "ok"` periods where 13-14
    speeches genuinely never touched the issue, and those are findings, not
    artifacts. That 13-14 is the whole range, not a floor: 14 is the largest
    `n_speeches` on ANY zero-width Surface A cell, flagged or kept. The corpus's
    per-period maximum is 67, but that period has no zero-width cell — quoting
    67, or "dozens", would describe evidence this conjunction has never actually
    had to weigh. Too-few-clusters alone would suppress 1785's
    Religion & values (0.0–66.7%), whose two speeches disagree and whose
    enormous band is the honest signal this module exists to draw.

    A NaN bound (the `suppressed_n_floor` case) compares False and is therefore
    never flagged: there is no interval to withdraw, and `ci_status` already
    names that cause. Two columns must not claim the same suppression.
    """
    if n_clusters >= MIN_CLUSTERS_FOR_RESOLVABLE_CI:
        return np.zeros(len(lo), dtype=bool)
    return np.asarray(lo == hi)


def bootstrap_corex_periods(
    labels: pd.DataFrame,
    issue_names: list[str],
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Speech-clustered bootstrap of each issue's share, per 5-year period.

    Args:
        labels: `paragraph_issues.parquet` — one row per paragraph, with a
            boolean column per issue and the `doc_name` that clusters them.
        issue_names: Boolean columns to estimate.
        n_draws: Bootstrap replicates per period.
        seed: Base seed; each period draws from `default_rng([seed, period])`
            so a period's interval does not depend on its neighbours.

    Returns:
        One row per (issue, period) with `point`, `lo_sampling`, `hi_sampling`
        as FRACTIONS (not percentages), plus `n_paragraphs`, `n_speeches`,
        `ci_status` and the per-cell `interval_unresolvable`. A period with
        fewer than `MIN_CLUSTERS_FOR_CI` speeches gets NaN bounds for every
        series; an individual cell whose bootstrap returned zero width below
        `MIN_CLUSTERS_FOR_RESOLVABLE_CI` clusters gets NaN bounds too, and is
        the one flagged — see the module docstring on why a degenerate
        zero-width interval is worse than none.
    """
    d = labels.assign(period=(labels["year"] // PERIOD_YEARS) * PERIOD_YEARS)
    rows = []
    for period, chunk in d.groupby("period", sort=True):
        per_doc = chunk.groupby("doc_name", sort=True)
        counts = per_doc[issue_names].sum().to_numpy(dtype=float)
        sizes = per_doc.size().to_numpy(dtype=float)
        n_docs = len(sizes)
        n_paragraphs = int(sizes.sum())
        status = _ci_status(n_docs, n_paragraphs)
        observed = counts.sum(axis=0) / n_paragraphs

        if status == "suppressed_n_floor":
            lo = hi = np.full(len(issue_names), np.nan)
        else:
            rng = np.random.default_rng([seed, int(period)])
            draws = rng.integers(0, n_docs, size=(n_draws, n_docs))
            weights = np.zeros((n_draws, n_docs))
            np.add.at(
                weights,
                (np.repeat(np.arange(n_draws), n_docs), draws.ravel()),
                1.0,
            )
            num = weights @ counts
            den = weights @ sizes
            replicates = num / den[:, None]
            lo = np.percentile(replicates, CI_LOW, axis=0)
            hi = np.percentile(replicates, CI_HIGH, axis=0)

        # Withdraw the interval where the bootstrap could not resolve one, and
        # do it HERE rather than at render time: a consumer reading the parquet
        # without the site's code must not find a fabricated 0.0-0.0 bound.
        unresolvable = _unresolvable_interval(lo, hi, n_docs)
        lo = np.where(unresolvable, np.nan, lo)
        hi = np.where(unresolvable, np.nan, hi)

        for i, name in enumerate(issue_names):
            rows.append({
                "series": name,
                "period_kind": "year5",
                "period": str(int(period)),
                "period_order": int(period),
                "period_start": int(period),
                "period_end": int(period) + PERIOD_YEARS - 1,
                "x": int(period),
                "point": float(observed[i]),
                "lo_sampling": float(lo[i]),
                "hi_sampling": float(hi[i]),
                "n_paragraphs": n_paragraphs,
                "n_speeches": n_docs,
                "ci_status": status,
                "interval_unresolvable": bool(unresolvable[i]),
            })
    return pd.DataFrame(rows)


def corex_bands(labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Surface A rows: sampling-only intervals over CorEx issue labels."""
    if labels is None:
        labels = pd.read_parquet(issues.PARA_LABELS_PATH)
    issue_names = corex_issue_columns(labels)
    out = bootstrap_corex_periods(labels, issue_names)
    out["surface"] = COREX_SURFACE
    # Not "absent", not "pending": there is no annotator in the CorEx pipeline,
    # so no measurement of annotator disagreement could ever apply to these rows.
    out["disagreement_band_applied"] = False
    out["disagreement_half_width"] = np.nan
    out["n_paired_paragraphs"] = pd.NA
    out["ci_components"] = "sampling_only"
    out["disagreement_status"] = NOT_APPLICABLE
    out["agreement_source"] = COREX_AGREEMENT_SOURCE
    out["lo"] = out["lo_sampling"]
    out["hi"] = out["hi_sampling"]
    return out


# --------------------------------------------------------------------------
# Surface B: LLM topic annotations, era grain
# --------------------------------------------------------------------------


def paired_annotations(taxonomy: dict) -> pd.DataFrame:
    """The paragraphs BOTH annotators labeled, with normalized topic sets.

    Returns:
        One row per paired paragraph: `doc_name`, `para_idx`, `year`, `era`,
        `topics_primary` and `topics_secondary` (canonical level-2 name sets).

    Raises:
        ValueError: If the secondary table does not survive either keyed merge
            intact, or if a paired paragraph's year falls outside `trends.ERAS`.
    """
    label_map = attention.canonical_label_map(taxonomy)
    primary = load_paragraph_annotations(PRIMARY_ANNOTATIONS)[
        ["doc_name", "para_idx", "topics"]
    ]
    secondary = load_paragraph_annotations(SECONDARY_ANNOTATIONS)[
        ["doc_name", "para_idx", "topics"]
    ].rename(columns={"topics": "topics_secondary"})

    paired = secondary.merge(
        primary.rename(columns={"topics": "topics_primary"}),
        on=["doc_name", "para_idx"],
        validate="one_to_one",
    )
    # Only the SECONDARY side must survive whole: `primary` is the full corpus
    # and is expected to be much larger, so it is deliberately not asserted
    # against (same rule as `attention.exemplar_quotes`).
    _require_full_merge(
        len(paired), {"secondary": len(secondary)}, "paired annotator tables"
    )

    years = pd.read_parquet(
        issues.PARA_LABELS_PATH, columns=["doc_name", "para_idx", "year"]
    )
    n_paired = len(paired)
    paired = paired.merge(
        years, on=["doc_name", "para_idx"], validate="one_to_one"
    )
    _require_full_merge(len(paired), {"paired": n_paired}, "paired x paragraph years")

    paired["era"] = attention.era_series(paired["year"])
    if paired["era"].isna().any():
        stray = sorted(paired.loc[paired["era"].isna(), "year"].unique())
        raise ValueError(
            f"paired years {stray} fall outside trends.ERAS; refusing to let "
            "them fall through into a per-era aggregate."
        )
    for col in ("topics_primary", "topics_secondary"):
        paired[col] = [
            frozenset(attention.normalize_topics(raw, label_map)) for raw in paired[col]
        ]
    return paired.sort_values(["doc_name", "para_idx"]).reset_index(drop=True)


def paired_coverage() -> dict:
    """How much of the corpus the SECOND annotator actually read.

    The single number governing how much weight the annotator component
    deserves, so it is computed from the artifacts rather than typed anywhere:
    the rendered dashboard prose, `bands_meta.json` and this module's docstring
    all trace back here. `n_sampled_speeches` is what
    `agreement_sample_v1.json` DREW; `n_paired_speeches` is what carries
    secondary annotations and therefore reaches this module — the gap is the
    batch-recovery loss, and conflating the two would credit the second reader
    with speeches it never saw.
    """
    secondary = load_paragraph_annotations(SECONDARY_ANNOTATIONS)[
        ["doc_name", "para_idx"]
    ]
    corpus = pd.read_parquet(
        issues.PARA_LABELS_PATH, columns=["doc_name", "para_idx"]
    )
    # The INTERSECTION, not `len(secondary)`. The two are equal today, but that
    # equality is enforced by `paired_annotations`'s `_require_full_merge` — and
    # this function is the one feeding the RENDERED sentence, on a path that no
    # longer runs that merge. Counting one side and calling it "paragraphs both
    # annotators read" would be true only by coincidence.
    primary = load_paragraph_annotations(PRIMARY_ANNOTATIONS)[
        ["doc_name", "para_idx"]
    ]
    paired = secondary.merge(
        primary, on=["doc_name", "para_idx"], validate="one_to_one"
    )
    _require_full_merge(
        len(paired), {"secondary": len(secondary)}, "paired_coverage"
    )
    sample = json.loads(SAMPLE_PATH.read_text()) if SAMPLE_PATH.exists() else {}
    return {
        "n_paired_paragraphs": int(len(paired)),
        "n_corpus_paragraphs": int(len(corpus)),
        "paragraph_fraction": float(len(secondary) / len(corpus)),
        "n_paired_speeches": int(paired["doc_name"].nunique()),
        "n_corpus_speeches": int(corpus["doc_name"].nunique()),
        "n_sampled_speeches": sample.get("n_sampled"),
        "sample_file": SAMPLE_PATH.name,
    }


def disagreement_half_widths(
    taxonomy: dict, paired: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per (era, topic) annotator half-width, in fraction-of-paragraphs units.

    `|share_primary - share_secondary| / 2` over the paragraphs both annotators
    saw — the half-range of the two estimates of the plotted quantity. Measured
    on `trends.ERAS` directly, so nothing is mapped across era axes.

    Returns:
        One row per (era, topic) with `share_primary`, `share_secondary`,
        `disagreement_half_width`, `n_paired_paragraphs`, and `available`
        (False where the era's paired sample is below `MIN_PAIRED_PARAGRAPHS`,
        in which case the half-width is NaN rather than an estimate off a
        handful of rows).
    """
    if paired is None:
        paired = paired_annotations(taxonomy)
    topics = attention.level_topics(taxonomy, LLM_LEVEL)
    rows = []
    for era, _, _ in ERAS:
        sub = paired[paired["era"] == era]
        n = len(sub)
        available = n >= MIN_PAIRED_PARAGRAPHS
        primary_counts = pd.Series(
            [t for s in sub["topics_primary"] for t in s], dtype=object
        ).value_counts()
        secondary_counts = pd.Series(
            [t for s in sub["topics_secondary"] for t in s], dtype=object
        ).value_counts()
        for topic in topics:
            sp = float(primary_counts.get(topic, 0)) / n if n else np.nan
            ss = float(secondary_counts.get(topic, 0)) / n if n else np.nan
            rows.append({
                "era": era,
                "topic": topic,
                "share_primary": sp,
                "share_secondary": ss,
                "disagreement_half_width": (
                    abs(sp - ss) / 2 if available else np.nan
                ),
                "n_paired_paragraphs": n,
                "available": available,
            })
    return pd.DataFrame(rows)


def llm_bands(taxonomy: dict | None = None) -> pd.DataFrame:
    """Surface B rows: sampling noise (reused) composed with disagreement.

    The sampling component is `attention.bootstrap_era_shares` verbatim — the
    speech-clustered, seeded bootstrap that already exists for this exact
    quantity. This function adds the annotator component and the provenance; it
    does not re-derive sampling noise.
    """
    taxonomy = taxonomy or attention.load_taxonomy()
    paragraphs, assignments = attention.load_inputs(taxonomy)
    sampling = attention.bootstrap_era_shares(
        paragraphs, assignments, taxonomy, level=LLM_LEVEL, treatment=LLM_TREATMENT
    )
    hw = disagreement_half_widths(taxonomy)

    n_sampling = len(sampling)
    out = sampling.merge(
        hw, on=["topic", "era"], how="left", validate="one_to_one",
    )
    _require_full_merge(
        len(out), {"sampling": n_sampling, "half_widths": len(hw)},
        "era shares x annotator half-widths",
    )
    # The length assertion above is necessary but NOT sufficient here: a LEFT
    # merge preserves length even when nothing matches, so a diverging `era`
    # dtype (`bootstrap_era_shares` can hand back a Categorical while `hw`
    # builds plain objects) would sail through it and surface only as every row
    # silently losing its disagreement component. Assert the keys actually
    # matched — this is the "validate catches duplicate keys but not diverging
    # key SETS" rule from CLAUDE.md, in its left-join form.
    unmatched = out["available"].isna()
    if unmatched.any():
        missing = sorted(set(zip(
            out.loc[unmatched, "topic"], out.loc[unmatched, "era"]
        )))[:5]
        raise ValueError(
            f"{int(unmatched.sum())} of {len(out)} (topic, era) cells found no "
            f"annotator half-width row; e.g. {missing}. The two frames' era "
            "keys have diverged — refusing to publish sampling-only intervals "
            "mislabelled as a thin paired sample."
        )

    era_bounds = {label: (lo, hi) for label, lo, hi in ERAS}
    era_order = {label: i for i, (label, _, _) in enumerate(ERAS)}
    out["surface"] = LLM_SURFACE
    out["series"] = out["topic"]
    out["period_kind"] = "era"
    out["period"] = out["era"].astype(str)
    out["period_order"] = out["period"].map(era_order)
    out["period_start"] = out["period"].map(lambda e: era_bounds[e][0])
    out["period_end"] = out["period"].map(lambda e: era_bounds[e][1])
    # Era midpoint, so Surface B shares the dashboard's year x-axis.
    out["x"] = (out["period_start"] + out["period_end"]) / 2
    out["point"] = out["share"]
    out["lo_sampling"] = out["share_lo"]
    out["hi_sampling"] = out["share_hi"]

    n_speeches = (
        paragraphs.groupby("era", observed=False)["doc_name"].nunique().to_dict()
    )
    out["n_speeches"] = out["period"].map(n_speeches).astype(int)
    out["ci_status"] = [
        _ci_status(int(s), int(p))
        for s, p in zip(out["n_speeches"], out["n_paragraphs"])
    ]

    # Same per-cell gate as Surface A, computed rather than asserted absent.
    # Surface B's thinnest era carries 55 speech clusters, so nothing here is
    # flagged today — but hardcoding False would make that a claim about the
    # code instead of a fact about the data, and would go stale silently if the
    # era axis were ever recut. Note it is tested on the SAMPLING interval: the
    # bootstrap is what did or did not resolve, and a nonzero annotator
    # half-width added on top would otherwise mask a degenerate one as a real
    # width.
    #
    # Called per row rather than vectorized, deliberately. `_unresolvable_interval`
    # takes ONE cluster count for a whole array because Surface A calls it once
    # per period, where that is exactly right; Surface B's cluster count varies
    # by row. Vectorizing here would mean either changing that signature — a
    # freshly test-pinned function shared by both surfaces — or inlining the
    # gate's two conditions a second time, which is how the two surfaces come to
    # disagree about what "unresolvable" means. 450 scalar calls in a $0 offline
    # builder is not a cost worth that.
    unresolvable = np.array([
        _unresolvable_interval(np.array([lo_s]), np.array([hi_s]), int(n))[0]
        for lo_s, hi_s, n in zip(
            out["lo_sampling"], out["hi_sampling"], out["n_speeches"]
        )
    ], dtype=bool)
    out["interval_unresolvable"] = unresolvable
    out["lo_sampling"] = out["lo_sampling"].where(~unresolvable)
    out["hi_sampling"] = out["hi_sampling"].where(~unresolvable)

    # Two independent preconditions, and they fail for different reasons — so
    # they get different statuses. Collapsing them would report "thin paired
    # sample" for a cell whose paired sample was ample and whose *sampling*
    # interval was the thing missing. `combat._apply_agreement_bands` keeps the
    # same distinction; dormant on today's data (all 450 rows apply) is not a
    # reason to state a false cause when it stops being dormant.
    have_half_width = out["available"].astype(bool)
    have_interval = out["lo_sampling"].notna() & out["hi_sampling"].notna()
    applied = have_half_width & have_interval
    out["disagreement_band_applied"] = applied
    out["disagreement_half_width"] = out["disagreement_half_width"].where(applied)
    out["ci_components"] = np.where(
        applied, "sampling+annotator_disagreement", "sampling_only"
    )
    # When BOTH preconditions fail the paired-sample reason is reported: it is
    # the more fundamental one (no half-width exists to apply, whatever the
    # sampling interval does).
    out["disagreement_status"] = np.where(
        applied, MEASURED, np.where(have_half_width, NO_INTERVAL, THIN_PAIRED)
    )
    out["agreement_source"] = LLM_AGREEMENT_SOURCE
    # Widen, never narrow — the `combat._apply_agreement_bands` rule.
    hw_col = out["disagreement_half_width"].fillna(0.0)
    out["lo"] = (out["lo_sampling"] - hw_col).clip(0, 1)
    out["hi"] = (out["hi_sampling"] + hw_col).clip(0, 1)
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build_bands() -> pd.DataFrame:
    """Both surfaces, one table, fixed column order and deterministic row order."""
    frames = [corex_bands(), llm_bands()]
    out = pd.concat([f[BANDS_COLUMNS] for f in frames], ignore_index=True)
    out["n_paired_paragraphs"] = out["n_paired_paragraphs"].astype("Int64")
    out["n_paragraphs"] = out["n_paragraphs"].astype(int)
    out["n_speeches"] = out["n_speeches"].astype(int)
    out["disagreement_band_applied"] = out["disagreement_band_applied"].astype(bool)
    out["interval_unresolvable"] = out["interval_unresolvable"].astype(bool)
    return out.sort_values(
        ["surface", "series", "period_order"], kind="mergesort"
    ).reset_index(drop=True)


def meta_path_for(path: Path) -> Path:
    """The meta sidecar for a band parquet: `bands.parquet` -> `bands_meta.json`.

    Derived from the parquet path rather than hardcoded, so redirecting
    `write_bands` redirects BOTH files. `llm_annotations.write_manifest`
    hardcoding its own directory is the documented version of this bug.
    """
    return path.with_name(f"{path.stem}_meta.json")


def write_bands(table: pd.DataFrame, path: Path = BANDS_PATH) -> Path:
    """Write the band table plus its meta sidecar.

    Deliberately a pure table-to-parquet write: the meta's coverage figures are
    stated as text rather than recomputed here, because calling
    `paired_coverage()` at write time would make this depend on the annotation
    parquets as well as its own argument. Those figures are pinned against a
    re-derivation in `tests/test_band_charts.py`
    (`TestMetaCoverageFiguresAreNotDrifting`) instead, which is where a drifting
    literal actually gets caught.

    Raises:
        ValueError: If `CI_LOW` no longer derives the floor the gate applies.
            Checked FIRST, before any file is touched — see below.
    """
    # Checked before the mkdir and before the parquet write, deliberately.
    # `derived_floor` follows LIVE `CI_LOW`; the `predicate` string and
    # `min_clusters_for_resolvable_ci` in the meta report the IMPORT-BOUND
    # `MIN_CLUSTERS_FOR_RESOLVABLE_CI`, which is what the gate in
    # `_unresolvable_interval` actually applies. Unreachable in production —
    # there is no `--ci-low` flag, so the two are the same object — but a test
    # that patches `CI_LOW` alone already makes them diverge, and the failure
    # mode is the self-defeating rationale: a `floor_derivation` block
    # concluding "n=3 is the first n where it can" printed beside a published
    # floor of 4. The gate stays truthful either way; the JUSTIFICATION would
    # be arguing for a different number than the one in force.
    #
    # Ordering is the point. This guard used to sit AFTER `to_parquet`, so a
    # divergent call replaced `data/bands.parquet` and then raised — leaving a
    # new table on disk beside a sidecar describing the old one. In a repo whose
    # rule is "a dirty `git status data/bands.parquet` means the numbers moved",
    # a half-written pair is worse than either writing both or writing neither:
    # the numbers moved and the provenance did not follow. It depends on nothing
    # but module constants, so there is no reason for it to run late.
    derived_floor = _min_clusters_for_resolvable_ci(CI_LOW)
    if derived_floor != MIN_CLUSTERS_FOR_RESOLVABLE_CI:
        raise ValueError(
            f"CI_LOW={CI_LOW} derives a resolvable-CI floor of {derived_floor}, "
            f"but the gate in force is MIN_CLUSTERS_FOR_RESOLVABLE_CI="
            f"{MIN_CLUSTERS_FOR_RESOLVABLE_CI}. Writing this meta would publish "
            "a derivation that argues for a floor the artifact does not use. "
            "Retune CI_LOW at module scope (so the constant follows it) and "
            "rebuild, rather than patching one of the two. Nothing has been "
            "written; neither the parquet nor its sidecar was touched."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)

    n_llm = int((table["surface"] == LLM_SURFACE).sum())
    applied = int(table["disagreement_band_applied"].sum())

    # Derived from the table being written wherever it can be; every number
    # below that IS typed is pinned by a re-derivation test. `bands_meta.json` is
    # the one new prose block on `topic-chart-upgrades` that shipped with a
    # wrong number, and the two blocks with a re-derivation stayed correct
    # (CLAUDE.md, "a provenance artifact's explanatory strings are prose").
    flagged = {
        s: int(table.loc[table["surface"] == s, "interval_unresolvable"].sum())
        for s in (COREX_SURFACE, LLM_SURFACE)
    }
    rows_total = {
        s: int((table["surface"] == s).sum())
        for s in (COREX_SURFACE, LLM_SURFACE)
    }
    flagged_periods = {
        s: sorted(table.loc[
            (table["surface"] == s) & table["interval_unresolvable"], "period"
        ].unique().tolist())
        for s in (COREX_SURFACE, LLM_SURFACE)
    }
    corex = table[table["surface"] == COREX_SURFACE]
    survivors = corex[corex["lo"].notna() & (corex["lo"] == corex["hi"])]
    kept_zero_width = int(len(survivors))
    kept_zero_width_periods = sorted(survivors["period"].unique().tolist())

    # `floor_derivation` below is a pure function of CI_LOW — the ladder, the
    # threshold AND the conclusion, not just the threshold. Interpolating one
    # number of an argument and typing the rest is how a rationale ends up
    # contradicting itself: at CI_LOW = 5.0 a typed ladder reported "0.0039 at
    # n=4 ... n=4 is the first n where it can" beside a threshold of 0.05, which
    # its own n=3 rung (0.037) already clears. The block was correct as
    # published; this keeps it correct after a retune, which is what its closing
    # sentence promises. `derived_floor` is computed at the top of this function
    # so its consistency check can run before anything is written.
    concentration_ladder = ", ".join(
        f"{n ** (-n):.2g} at n={n}"
        for n in range(MIN_CLUSTERS_FOR_CI, derived_floor + 1)
    )
    meta = {
        # No wall-clock stamp, on purpose: every value here is a pure function
        # of the frozen inputs, so a timestamp would make `git status` dirty on
        # every rerun for a reason that carries no information. Provenance
        # identity is `corpus_fingerprint`; *when* it ran is git's job.
        "surfaces": {
            COREX_SURFACE: {
                "labels": "CorEx anchored topic model (issues.py, seed=42)",
                "grain": f"{PERIOD_YEARS}-year periods",
                "ci_components": "sampling_only",
                "annotator_disagreement": (
                    "NOT APPLICABLE — the CorEx pipeline contains no LLM "
                    "annotator, so there is no annotator whose disagreement "
                    "could widen these intervals. This is not a missing "
                    "component awaiting a dependency."
                ),
            },
            LLM_SURFACE: {
                "labels": f"paragraph_annotations.topics ({LLM_LEVEL} of taxonomy_v1)",
                "grain": "trends.ERAS (9 named eras)",
                "treatment": LLM_TREATMENT,
                "sampling_source": "attention.bootstrap_era_shares (reused, not re-derived)",
                "annotator_disagreement": {
                    "definition": (
                        "|share_primary - share_secondary| / 2 over the "
                        "paragraphs both annotators labeled"
                    ),
                    "source": LLM_AGREEMENT_SOURCE,
                    "era_axis": (
                        "measured directly on trends.ERAS — no cross-axis "
                        "mapping from taxonomy.ERA_SPAN is performed"
                    ),
                    "agreement_v1_parquet": (
                        "deliberately NOT read: its topic metric is a jaccard "
                        "with no conversion to percentage points of share, and "
                        "it is keyed on the ERA_SPAN sampling axis"
                    ),
                    "caveats": {
                        "inflating": (
                            "the paired subsample's own sampling noise is "
                            "included in the observed difference, so the "
                            "half-width leans high — the safe direction for a "
                            "band, and not corrected for"
                        ),
                        "deflating": (
                            "the band is centred on the PRIMARY annotator's "
                            "point estimate and moves by half the gap between "
                            "the two, so it reaches halfway toward the "
                            "secondary annotator's estimate and never covers "
                            "it; an envelope containing both readings would "
                            "use the full |difference|"
                        ),
                        "transfer_assumption": (
                            "the paired set is a DOCUMENT-level sample: 262 "
                            "speeches (of the 266 agreement_sample_v1.json "
                            "drew; agreement.py::_load_joined restricts to "
                            "sample_docs), holding 8570 of the corpus's 36229 "
                            "paragraphs — 23.7%. Its disagreement is assumed "
                            "to transfer to the full-corpus era intervals it "
                            "widens; within-speech clustering is not modelled "
                            "in the half-width, only in the sampling "
                            "bootstrap. `bands.paired_coverage()` recomputes "
                            "every figure in this sentence."
                        ),
                    },
                    "min_paired_paragraphs": MIN_PAIRED_PARAGRAPHS,
                    "rows_with_component_applied": applied,
                    "rows_total": n_llm,
                },
            },
        },
        "composition": "lo = lo_sampling - hw; hi = hi_sampling + hw; clipped to [0, 1]",
        "units": "fraction of paragraphs (NOT percent) — charts multiply by 100",
        "bootstrap": {
            "n_draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "ci_percentiles": [CI_LOW, CI_HIGH],
            "cluster_unit": "speech",
            "rationale": "within-speech ICC up to 0.17; paragraph resampling "
                         "would understate the interval",
            "surface_a_seeding": "default_rng([seed, period]) — per period, so a "
                                 "period's draws do not depend on its neighbours",
        },
        "ci_status": {
            "min_clusters_for_ci": MIN_CLUSTERS_FOR_CI,
            "low_cluster_caution": LOW_CLUSTER_CAUTION,
            "min_period_paragraphs": MIN_PERIOD_PARAGRAPHS,
            "differs_from_combat": {
                "combat_min_clusters_for_ci": combat.MIN_CLUSTERS_FOR_CI,
                "combat_low_cluster_caution": combat.LOW_CLUSTER_CAUTION,
                "reason": "combat's floors guard an ERA-grain rate; these guard "
                          "a 5-year period, where the corpus median is 20 "
                          "speeches per period, so combat's floor of 20 would "
                          "mark 24 of the 49 periods — roughly half the record "
                          "— provisional, and a caution flag that fires on half "
                          "the chart carries no information. Same column name "
                          "and direction, different grain — do not read one "
                          "gate across both artifacts.",
            },
            "de_emphasis_is_two_sided": (
                "MIN_PERIOD_PARAGRAPHS was the charts' former hard mask and "
                "dropped exactly one of 49 periods (1785); LOW_CLUSTER_CAUTION "
                "is a new gate that additionally demotes 1820 (277 paragraphs, "
                "7 speeches) from solid to provisional. One period recovered, "
                "one newly marked provisional."
            ),
            "note": "MIN_PERIOD_PARAGRAPHS was the charts' former hard mask; it "
                    "dropped exactly one of 49 five-year periods (1785, 14 "
                    "paragraphs, 2 speeches) and is now a de-emphasis flag",
            "degenerate_interval_predicate": (
                "SUPERSEDED by the per-cell `interval_unresolvable` column "
                "below. Cells matching it no longer publish lo/hi at all: the "
                "zero-width interval is withdrawn (lo, hi, lo_sampling and "
                "hi_sampling are all null) and the point estimate is kept."
            ),
        },
        "interval_unresolvable": {
            "column": "interval_unresolvable",
            "grain": "per CELL (surface, series, period) — unlike ci_status, "
                     "which is per PERIOD by construction and keeps its own "
                     "meaning unchanged",
            "predicate": (
                "the bootstrap ran and returned a zero-width interval "
                "(lo_sampling == hi_sampling) in a period with fewer than "
                f"{MIN_CLUSTERS_FOR_RESOLVABLE_CI} speech clusters"
            ),
            "effect": (
                "lo, hi, lo_sampling and hi_sampling are null; `point` is still "
                "published, because the observed share is a real measurement "
                "and only the interval was unresolvable. The charts draw a "
                "hollow marker at these points so a reader who never hovers can "
                "still tell them apart from a banded estimate."
            ),
            "why_both_conditions": (
                "zero width ALONE would suppress legitimate confident zeros — "
                f"{kept_zero_width} of the "
                f"{kept_zero_width + flagged[COREX_SURFACE]} zero-width "
                f"corex_issues cells survive this gate, in periods "
                f"{kept_zero_width_periods} where enough speeches genuinely "
                "never touched the issue, which is a finding. Too-few-clusters "
                "ALONE would suppress 1785's Religion & values, whose two "
                "speeches disagree and whose 0.0-66.7% band is the honest "
                "signal. Both conditions are required."
            ),
            "why_not_ci_status": (
                "ci_status fires on n_speeches < 8 OR n_paragraphs < 40, so "
                "'ci_status == low_cluster_caution and lo == hi' would also "
                "suppress a paragraph-thin but SPEECH-RICH period, where the "
                "resample space is large and an all-zero result is real. No such "
                "period exists in this corpus, so the two predicates select the "
                f"same {flagged[COREX_SURFACE]} cells today — but only the "
                "cluster-count one says what it means."
            ),
            "min_clusters_for_resolvable_ci": MIN_CLUSTERS_FOR_RESOLVABLE_CI,
            "floor_derivation": (
                "not chosen: the most concentrated resample (all n draws on one "
                "cluster) carries probability n**-n = "
                f"{concentration_ladder}. Until that drops below CI_LOW/100 = "
                f"{CI_LOW / 100}, the lower percentile cannot exclude even the "
                "single most extreme resample, so the 'interval' is the full "
                f"span of the resample space. n={derived_floor} is the first n "
                "where it can. Computed from CI_LOW by "
                "bands._min_clusters_for_resolvable_ci, so retuning the "
                "percentile moves the floor with it."
            ),
            "distinct_resamples": {
                str(n): distinct_resamples(n) for n in (2, 3, 4, 5, 10)
            },
            "min_clusters_for_ci_unchanged": (
                f"MIN_CLUSTERS_FOR_CI stays at {MIN_CLUSTERS_FOR_CI}. Raising it "
                "to 3 would delete the 1785 period outright; the defect was the "
                "interval, never the point estimate."
            ),
            "rows_flagged": flagged,
            "rows_total": rows_total,
            "flagged_periods": flagged_periods,
        },
        "corpus_fingerprint": corpus_fingerprint(),
        "api_calls": 0,
    }
    meta_path_for(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_bands(path: Path | None = None) -> pd.DataFrame | None:
    """The band table, or None when it has not been built yet.

    A seam, not a stub: the chart functions render an unbanded line when this
    returns None rather than inventing an interval. `data/bands.parquet` is
    derived and $0 to regenerate (`python -m presidential_profiles.bands`), so
    absence means "not built here yet", never "unknowable".
    """
    if path is None:
        path = BANDS_PATH
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    missing = set(BANDS_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(
            f"{path.name} exists but is missing columns {sorted(missing)}; "
            "refusing to guess at the band schema. Rebuild with "
            "`python -m presidential_profiles.bands`."
        )
    return table


def series_band(
    table: pd.DataFrame | None, surface: str, series: str
) -> pd.DataFrame | None:
    """One series' rows, x-ordered, or None if the table has nothing for it.

    Returns None rather than an empty frame so a caller cannot accidentally
    draw a zero-length band and believe it drew something.
    """
    if table is None:
        return None
    sub = table[(table["surface"] == surface) & (table["series"] == series)]
    if sub.empty:
        return None
    return sub.sort_values("period_order").reset_index(drop=True)


def print_checks(table: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    for surface in (COREX_SURFACE, LLM_SURFACE):
        sub = table[table["surface"] == surface]
        width = (sub["hi"] - sub["lo"]) * 100
        print(f"\n=== {surface} ===")
        print(f"  rows: {len(sub)}  series: {sub['series'].nunique()}  "
              f"periods: {sub['period'].nunique()}")
        print(f"  ci_status: {sub['ci_status'].value_counts().to_dict()}")
        print(f"  ci_components: {sub['ci_components'].value_counts().to_dict()}")
        flagged = sub[sub["interval_unresolvable"]]
        print(f"  interval_unresolvable: {len(flagged)} "
              f"(periods {sorted(flagged['period'].unique().tolist())})")
        kept = sub[sub["lo"].notna() & (sub["lo"] == sub["hi"])]
        print(f"  zero-width intervals KEPT (legitimate): {len(kept)} "
              f"(periods {sorted(kept['period'].unique().tolist())})")
        print(f"  band width (pp): median {np.nanmedian(width):.2f}  "
              f"max {np.nanmax(width):.2f}")
    corex = table[table["surface"] == COREX_SURFACE]
    thin = corex.loc[corex.groupby("series")["n_speeches"].idxmin()]
    print("\n=== thinnest CorEx period per issue (band width, pp) ===")
    print(thin.assign(width=(thin["hi"] - thin["lo"]) * 100)[
        ["series", "period", "n_speeches", "n_paragraphs", "ci_status", "width"]
    ].head(6).to_string(index=False))
    llm = table[table["surface"] == LLM_SURFACE]
    contrib = (llm["disagreement_half_width"] * 200).describe()
    print("\n=== disagreement contribution to LLM band width (pp) ===")
    print(contrib.round(3).to_string())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Confidence bands for the issue/topic charts "
                    "(pure local compute; $0, no API calls)."
    )
    ap.add_argument("--quiet", action="store_true",
                    help="write outputs without printing checks")
    args = ap.parse_args(argv)

    table = build_bands()
    path = write_bands(table)
    print(f"wrote {path} ({len(table):,} rows)")
    if not args.quiet:
        print_checks(table)


if __name__ == "__main__":
    main()
