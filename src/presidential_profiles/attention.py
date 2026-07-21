"""Topic attention over 240 years: birth, death, revival — and rename vs death.

The legacy 15-issue CorEx taxonomy in `issues.py` is blind to 28.7% of
paragraphs, and blind in a *time-varying* way, so the very topics whose deaths
would answer "are we caring about the same things?" (Indian affairs,
Reconstruction, coinage, civil-service reform) were literally unmeasurable. The
corpus-native taxonomy (`taxonomy_v1.json`, 17 level-1 domains / 50 level-2
topics) plus the full-corpus LLM annotation pass (`paragraph_annotations`,
36,229 paragraphs) make them measurable. This module turns those frozen
artifacts into per-topic attention curves, lifecycles, and a rename-vs-death
classification.

Everything here is FREE local compute over frozen artifacts. **No Anthropic API
calls, ever** — do not add one. Exemplar quotes come from `paragraphs.parquet`.

--------------------------------------------------------------------------
Method, and every judgement call it rests on
--------------------------------------------------------------------------

**Label normalization (mandatory, not defensive).** `paragraph_annotations`
holds 58 distinct topic strings against the taxonomy's 50 canonical level-2
names. All 8 extras are pure case variants ("… & The Environment" vs "… & the
Environment", "War Of 1812" vs "War of 1812"). Counting raw strings fragments
real topics — `Early Naval Wars` alone loses 12 paragraphs. Every label is
mapped through `{name.casefold(): name}` and an unmapped label RAISES: it means
the taxonomy and the annotations have diverged, which is a data-integrity fact,
not a rounding error.

**Denominator.** Share of ALL paragraphs in the slice, including the 404 that
carry an empty `topics` list. Those are real rows (the annotator saw them and
assigned nothing), not missing data; excluding them would inflate every share.

**Multi-label.** Mean 1.439 topics per paragraph as this module counts them
(52,133 `assignments` rows / 36,229 paragraphs), so per-topic shares do NOT sum
to 1 and must never be normalized as if they should. A (paragraph, topic) pair
is counted once even if the annotator listed the topic twice — the 1.46 quoted
upstream of this module is the PRE-dedup figure, and the gap between the two is
precisely what that dedup rule removes.

**Smoothing.** Curves are reported both raw per year and smoothed with a
centered 5-year rolling window over the numerator and denominator *separately*
(`sum(num)/sum(den)`, not a mean of ratios), which is the repo's established
idiom for sparse years (`site.py:73`, `indices.py:156`). Two corpus years
(1946, 1950) contain zero paragraphs; the window covers them.

**"Substantive" year — the pre-registered threshold.** A stray single paragraph
must not set a topic's birth year, and it demonstrably would: seven topics have
a raw first appearance that is historically impossible (World War II Military
Operations first appears in an 1861 Lincoln message). A (topic, year) is
substantive iff ALL of:

  1. the 5-year window holds >= 30 paragraphs in the slice (enough data to say
     anything at all — the thinnest founding-era window holds 67), AND
  2. the window holds >= 3 paragraphs carrying the topic (kills single-stray
     anachronisms), AND
  3. the smoothed share >= max(0.002, 0.10 x the topic's own peak smoothed
     share) — an absolute floor of 0.2% of paragraphs so trace mentions never
     count, plus a relative floor so a topic is not called "alive" at a
     hundredth of its own peak. The relative leg makes the definition
     scale-free across topics whose overall prevalence differs 60-fold.

These numbers are fixed constants at the top of this module. They were chosen
on the reasoning above BEFORE the ground-truth checks were run and were not
tuned afterwards; a failed check is reported as a finding, not smoothed away.

A centered window leads and lags its data by two years, so the substantive span
is intersected with the years the topic actually has a paragraph. Without that
clamp a topic could be "born" two years before its first real paragraph, which
would be a smoothing artifact wearing the costume of a finding.

**Genre treatments (all three).** Annual messages are 55-76% of pre-1933
paragraphs but only 21-31% of modern ones, so a raw-only reading would report
the annual message's death as every 19th-century issue's death.

  * `raw` — share of every paragraph in the year.
  * `sotu` — restricted to `speech_type == state_of_the_union_or_annual_message`
    (218 speeches, 15,191 paragraphs). This is the genre check: does a claimed
    death survive inside annual messages alone?
  * `genre_standardized` — post-stratification. Compute the within-genre share
    per year, then recombine with FIXED corpus-wide genre weights
    `w_g = paragraphs of genre g / all paragraphs`, so a changing genre mix
    cannot masquerade as an attention shift.

    MISSING-GENRE-YEAR RULE (stated here because it must never silently produce
    NaN): in year *y* a genre is *eligible* if its 5-year window holds at least
    `MIN_GENRE_STRATUM` (=3) paragraphs — a stratum of one paragraph must not
    receive a genre's full corpus weight. Weights are renormalized over the
    eligible genres only, so they always sum to 1. If NO genre clears the
    threshold, the rule falls back to every genre actually present (>=1
    paragraph) in that window; only a window containing zero paragraphs
    anywhere yields NaN, and no such window exists in this corpus (verified:
    zero NaN smoothed shares under `raw` and `genre_standardized`). `sotu` is
    the one treatment that does produce NaN years — eight windows contain no
    annual message at all — and that comes from the slice filter, not from this
    rule; a year with no annual messages genuinely has no SOTU estimate.

    The threshold is deliberately NOT applied to the era-level bootstrap grid
    (`_era_shares`), which uses plain `den > 0`: `MIN_GENRE_STRATUM` guards a
    5-year window, whereas an era spans decades and has far more data per
    stratum. The era grid therefore applies the same eligibility RULE as this
    module's unsmoothed yearly `share`, and a different one from `share_smooth`.
    That is rule equality only — an era share and a yearly share aggregate over
    different grains and are not expected to be equal as VALUES. Which of the two
    year-level calls the era grid ought to match is an open methodological
    question logged to the backlog, not a bug.

**Lifecycle class.** Over the observed record [1789, 2026]:
`died` if the last substantive year is >= 40 years before the end of the record;
else `revived` if there is a >= 30-year internal gap between substantive years;
else `born` if the first substantive year is >= 40 years after the start of the
record; else `persistent`. Precedence is died > revived > born > persistent.

**Under-powered guard.** A topic with < 50 paragraphs corpus-wide in the slice
is flagged `under_powered=True` and its `lifecycle_class` is set to
`under_powered` so a consumer filtering on the class cannot chart noise; the
descriptive columns are still populated (they are the evidence that the topic is
thin). Measured on the shipped `topic_lifecycles.parquet`, the guard flags
NOTHING at level 2 under `raw` (the thinnest topic is Polygamy in the Territories
at 61 paragraphs) and nothing under `genre_standardized`; it fires EXACTLY ONCE
in the whole table — level 2, `sotu`, `Chinese Immigration & Exclusion`, which
holds 45 paragraphs inside annual messages. Nothing fires at level 1 under any
treatment.

That single firing is the guard's actual value: it is implemented and it is
demonstrably non-vacuous on a real slice, rather than dead code nobody could tell
was dead. (The plan predicted it would flag nothing at level 2, and for `raw` and
`genre_standardized` that prediction was right.) It is emphatically NOT evidence
that claimed deaths generally fail the annual-message check — the module's
headline robustness result runs the other way: 15 of the 17 `raw` deaths still
classify `died` under `sotu`, and 16 of 17 under `genre_standardized`. The one
topic this guard catches is itself one of those two `sotu` exceptions, so the
guard and the robustness result are the same fact seen twice, not two findings.
Pinned by `test_the_under_powered_guard_fires_exactly_once_across_the_table`.

**Rename vs death — two independent signals, both computed here.** (The
`topic-method-comparison` task owns `triangulate.py`/`topic_quality.py`; those
do not exist on this branch and are deliberately not imported.)

  1. *Crosswalk divergence, the LLM<->CorEx test.* `crosswalk_v1.json` is
     inverted to level-2 topic -> legacy issue(s). For a dying topic, its LLM
     curve and its legacy CorEx issue curve are each reduced by the SAME
     statistic — a decline ratio of (mean smoothed share after the topic's last
     substantive year) / (mean smoothed share over the topic's peak window,
     peak year +/- 10). Either curve has "collapsed" if that ratio is <= 0.25.
     A ratio above 1 means the legacy issue is HIGHER after the death than
     during the topic's peak, i.e. that crosswalk parent never tracked this
     topic and its verdict carries no information.

     HOW A NO-INFORMATION PARENT VOTES (it does vote, deliberately). Declaring
     a parent uninformative is NOT the same as discarding it. Such a parent is
     still counted as "persists", because the only alternative — dropping it —
     would let a topic reach `true_death` on the strength of a cross-check that
     was never actually informative. That is the same conservative reflex as the
     no-parent case below: absence of a usable signal must never be read as
     agreement between the labelers. Four dying topics are decided this way at
     level 2 / `raw`: Public Lands & Homestead Settlement (Energy & environment
     6.32), Indian Affairs, Removal & Allotment (Civil rights & race 3.85 — its
     ONLY parent), Polygamy in the Territories (Religion & values 2.23) and
     Mexican War (War & military 1.04). Excluding `> 1` parents was measured and
     changes NO row's `rename_class`, so this is a statement about how to read
     the numbers, not a numbers bug.

     `corex_decline_ratio` IS A MAX, NOT A MEAN. The single reported ratio is
     `max` over the topic's finite parent ratios — the MOST-PERSISTING parent —
     which is the scalar the "persists if ANY parent persists" rule below
     actually turns on, so the column and the rule cannot disagree. Read it that
     way when quoting divergence multiples: the report's 560x for Polygamy is
     `2.23 / 0.00398`, whose numerator is a parent this module itself declares
     non-tracking. Every parent's own ratio is preserved verbatim in
     `corex_parent_ratios` so any other reduction can be recomputed without
     rerunning anything.

     THIS ARM IS ONE-SIDED BY CONSTRUCTION, and that is why `true_death` ships
     empty. Fifty level-2 topics are being cross-checked against 15 legacy CorEx
     buckets, so a parent is almost always broader than the topic hanging off it
     and can easily persist on vocabulary that has nothing to do with the topic
     that died. The crosswalk arm can therefore veto a death far more readily
     than it can confirm one. The two signals are NOT symmetric evidence and must
     not be presented as such: signal 2 is the affirmative one.

     MULTI-PARENT RULE (explicit, per the spec's warning that the inversion is
     many-to-many — `Chinese Immigration & Exclusion` sits under both
     `Immigration` and `Civil rights & race`): a topic is credited as a true
     death only if EVERY one of its legacy parents collapsed. Equivalently,
     CorEx "persists" if ANY parent persists. This is the conservative
     direction and it follows directly from the key decision that a death
     requires BOTH labelers to show collapse — taking the first parent, or a
     majority vote, would let an arbitrary crosswalk ordering decide a
     substantive claim. Every parent's ratio is recorded in
     `corex_parent_ratios` so the rule can be re-litigated without rerunning.

  2. *Successor detection, within-taxonomy.* A candidate successor for a dying
     topic is another topic in the SAME level-1 domain that (a) first becomes
     substantive INSIDE the dying topic's decline window [peak year, last
     substantive year] — the handoff test, (b) outlives it, (c) shares at least
     one legacy crosswalk parent with it, and (d) has a negatively correlated
     year curve; the most anti-correlated survivor wins. All four filters are
     load-bearing: over a 240-year record any two topics occupying disjoint
     periods are anti-correlated, so correlation alone nominates World War II
     as the Mexican War's "successor". A named successor is what makes a rename
     claim concrete rather than an inference from an absence.

  Final classes for a dying topic, in precedence order: `rename` (a successor is
  named — signal 2 is the concrete, checkable evidence and it wins, because
  signal 1 runs through a 15-bucket crosswalk whose anchor vocabulary is itself
  period-bound: `Civil rights & race` is anchored on "slavery"/"emancipation"/
  "negro" and therefore *appears* to collapse for exactly the renaming reason
  under test); `true_death` (no successor AND every CorEx parent collapsed too —
  both labelers agree and nothing took over); `unresolved_death` (no successor
  but the lexical family lives on — the LLM says dead, CorEx says the vocabulary
  survives, and the taxonomy names no heir; flag for review). `corex_persists`
  is recorded on every dying row regardless, so a reader can always see whether
  the two signals agreed.

  A cross-domain best-anticorrelated riser is recorded as a DIAGNOSTIC only
  (`cross_domain_candidate`) and never used to classify: over a 240-year record
  spurious anti-correlation is trivially easy to find.

**Bootstrap CIs.** Resampled on SPEECHES with replacement, never paragraphs —
within-speech ICC runs to 0.17, so paragraph-level resampling would badly
understate the interval. One shared draw matrix (seeded, `BOOTSTRAP_SEED`) is
reused across every level and treatment so replicates are comparable and the
output is byte-reproducible. Era shares are the bootstrapped quantity because
era is the grain at which born/died claims are actually made; thin eras get wide
intervals, which is the correct answer and not a bug to smooth away.

--------------------------------------------------------------------------
Outputs
--------------------------------------------------------------------------

`data/attention/topic_lifecycles.parquet` — one row per (level, topic,
treatment). The full 9-era share + CI + sample-size grid is NOT written to disk
(this task's file allowlist permits one data file); regenerate it deterministically
with the public `bootstrap_era_shares()`, and see `notes/attention-findings-v1.md`
for the tables that back the report.

Run as:
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.attention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import DATA_DIR

# "Do not modify `issues.py`" guards against a merge conflict with a parallel
# session; an import cannot cause one, and it is transitively unavoidable anyway
# since `taxonomy.py` (which this task reuses) already imports `issues`.
# Re-declaring the path here would fork its single source of truth.
from .issues import PARA_LABELS_PATH
from .llm_annotations import (
    ANNOTATIONS_DIR,
    PARAGRAPHS_PATH,
    load_paragraph_annotations,
    load_speech_annotations,
)

# `_require_full_merge` is private by name but public by convention: CLAUDE.md
# names it as THE keyed-merge guard for this repo ("also assert merged length ==
# each input's length"). Re-implementing it here would be duplicate code and
# would let the two copies drift.
from .taxonomy import LEGACY_ISSUES, SECURITY_PEACE, _require_full_merge
from .trends import ERAS

ATTENTION_DIR = DATA_DIR / "attention"
LIFECYCLES_PATH = ATTENTION_DIR / "topic_lifecycles.parquet"
TAXONOMY_PATH = ANNOTATIONS_DIR / "taxonomy_v1.json"
CROSSWALK_PATH = ANNOTATIONS_DIR / "crosswalk_v1.json"

# CorEx's Discovered 5 is the one free topic with an agreed identity, and the
# crosswalk refers to it by that name (taxonomy.SECURITY_PEACE). The other six
# Discovered columns are unnamed (`discovered-topics-1-7-unnamed` in the backlog)
# and are deliberately not used as cross-check curves.
COREX_COLUMNS = {issue: issue for issue in LEGACY_ISSUES}
COREX_COLUMNS[SECURITY_PEACE] = "Discovered 5"

SOTU_TYPE = "state_of_the_union_or_annual_message"
TREATMENTS = ("raw", "sotu", "genre_standardized")
LEVELS = ("level2", "level1")

# ---- pre-registered thresholds (see module docstring; do not tune to taste) --
SMOOTH_WINDOW = 5
MIN_WINDOW_PARAGRAPHS = 30
MIN_TOPIC_PARAGRAPHS = 3
ABS_SHARE_FLOOR = 0.002
REL_PEAK_FRACTION = 0.10
UNDER_POWERED_N = 50
MIN_GENRE_STRATUM = 3
DEATH_GAP_YEARS = 40
BIRTH_GAP_YEARS = 40
REVIVAL_GAP_YEARS = 30
COREX_COLLAPSE_RATIO = 0.25
PEAK_WINDOW_HALF = 10

BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 20260721
CI_LOW, CI_HIGH = 2.5, 97.5


# --------------------------------------------------------------------------
# eras
# --------------------------------------------------------------------------

# ONE definition of the era bands, shared by the scalar and the vectorized lookup
# so they cannot drift. `trends.ERAS` states INCLUSIVE `[lo, hi]` bands and they
# are contiguous, which is exactly `pd.cut`'s default `right=True` half-open
# `(previous hi, hi]` over these edges — hence the left edge one year BEFORE the
# first era starts. Both functions below use the same edges and the same half-open
# convention, so their equivalence is by construction (TestEras pins it yearly).
_ERA_LABELS = [label for label, _, _ in ERAS]
_ERA_EDGES = [ERAS[0][1] - 1] + [hi for _, _, hi in ERAS]


def era_name(year: int | float) -> str | None:
    """The named era (from `trends.ERAS`) containing `year`.

    Args:
        year: A calendar year; NaN is tolerated and returns None.

    Returns:
        The era label, or None if the year falls outside every era band.
    """
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return None
    year = int(year)
    for label, lo, hi in zip(_ERA_LABELS, _ERA_EDGES, _ERA_EDGES[1:]):
        if lo < year <= hi:
            return label
    return None


def era_series(years: pd.Series) -> pd.Series:
    """Vectorized `era_name` over a year column (kept identical to ERAS bands)."""
    return pd.cut(years, bins=_ERA_EDGES, labels=_ERA_LABELS, ordered=False)


def era_span_label(first_year: float, last_year: float) -> str | None:
    """Human "era of relevance": the era span a topic was substantive across."""
    first, last = era_name(first_year), era_name(last_year)
    if first is None or last is None:
        return None
    return first if first == last else f"{first} → {last}"


# --------------------------------------------------------------------------
# taxonomy + label normalization
# --------------------------------------------------------------------------


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    """Load the frozen `taxonomy_v1.json` (read-only artifact of a paid run)."""
    return json.loads(Path(path).read_text())


def canonical_label_map(taxonomy: dict) -> dict[str, str]:
    """Build `{casefolded name: canonical name}` over the level-2 topics.

    Args:
        taxonomy: The parsed `taxonomy_v1.json`.

    Returns:
        A casefold-to-canonical lookup.

    Raises:
        ValueError: If casefolding is not injective over the canonical names —
            then normalization would silently merge two distinct topics, which
            is worse than the fragmentation it is meant to fix.
    """
    names = [entry["name"] for entry in taxonomy["level2"]]
    folded = {name.casefold(): name for name in names}
    if len(folded) != len(names):
        collisions = sorted({n for n in names if sum(m.casefold() == n.casefold()
                                                     for m in names) > 1})
        raise ValueError(
            "casefold() is not 1:1 over the taxonomy's level-2 names, so "
            f"normalizing would merge distinct topics: {collisions}"
        )
    return folded


def level1_parents(taxonomy: dict) -> dict[str, str]:
    """`{level-2 topic name: its level-1 domain}` — the domain lookup, once.

    Both the assignment build (`load_inputs`) and the successor search (which
    only ever looks inside the dying topic's own domain) key on this, so the two
    must not drift.
    """
    return {entry["name"]: entry["level1"] for entry in taxonomy["level2"]}


def normalize_topics(raw: object, label_map: dict[str, str]) -> list[str]:
    """Map one paragraph's raw topic labels onto canonical taxonomy names.

    Args:
        raw: The annotation's `topics` cell — a list/array of strings, or None.
        label_map: Output of `canonical_label_map`.

    Returns:
        The canonical names, de-duplicated and order-stable (a paragraph that
        lists a topic twice, once per case variant, counts once).

    Raises:
        ValueError: If any label has no canonical match. An unmapped label means
            the annotations and the taxonomy have diverged; counting it as its
            own topic would invent a 51st topic out of a typo.
    """
    if raw is None:
        return []
    out: list[str] = []
    for label in list(raw):
        canonical = label_map.get(str(label).casefold())
        if canonical is None:
            raise ValueError(
                f"topic label {label!r} is not in taxonomy_v1's 50 level-2 names "
                "(not even case-insensitively). The annotations and the taxonomy "
                "have diverged — refusing to count an unknown topic."
            )
        if canonical not in out:
            out.append(canonical)
    return out


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_inputs(taxonomy: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two frames every curve is computed from.

    Args:
        taxonomy: Parsed `taxonomy_v1.json`; loaded from disk when omitted.

    Returns:
        `(paragraphs, assignments)`.
        `paragraphs` is one row per corpus paragraph — the DENOMINATOR universe,
        including the 404 paragraphs annotated with no topic at all — with
        columns `doc_name, para_idx, year, era, speech_type`.
        `assignments` is one row per (paragraph, level-2 topic) with the same
        context columns plus `topic` and `level1`.

    Raises:
        ValueError: If any of the three keyed merges drops rows (annotations x
            issues, paragraphs x speech types, assignments x paragraphs — see
            `_require_full_merge`), if a paragraph has no `speech_type`, if a
            year falls outside `trends.ERAS`, or if a topic label cannot be
            normalized or has no level-1 parent.
    """
    taxonomy = taxonomy or load_taxonomy()
    label_map = canonical_label_map(taxonomy)
    parent = level1_parents(taxonomy)

    ann = load_paragraph_annotations("paragraph_annotations")[
        ["doc_name", "para_idx", "topics"]
    ]
    issues = pd.read_parquet(PARA_LABELS_PATH, columns=["doc_name", "para_idx", "year"])
    speech = load_speech_annotations("speech_annotations")[["doc_name", "speech_type"]]

    paragraphs = ann.merge(issues, on=["doc_name", "para_idx"], validate="one_to_one")
    _require_full_merge(
        len(paragraphs), {"annotations": len(ann), "issues": len(issues)},
        "paragraph_annotations x paragraph_issues",
    )
    paragraphs = paragraphs.merge(speech, on="doc_name", how="left", validate="many_to_one")
    missing = int(paragraphs["speech_type"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} paragraphs have no speech_type; the genre treatments "
            "would silently drop them. speech_annotations must cover every speech."
        )
    paragraphs["era"] = era_series(paragraphs["year"])
    if paragraphs["era"].isna().any():
        stray = sorted(paragraphs.loc[paragraphs["era"].isna(), "year"].unique())
        raise ValueError(
            f"years {stray} fall outside trends.ERAS ({ERAS[0][1]}-{ERAS[-1][2]}). "
            "Extend ERAS rather than letting era-level aggregates silently drop them."
        )

    rows = []
    for doc_name, para_idx, raw in zip(
        paragraphs["doc_name"], paragraphs["para_idx"], paragraphs["topics"]
    ):
        for topic in normalize_topics(raw, label_map):
            rows.append((doc_name, para_idx, topic))
    assignments = pd.DataFrame(rows, columns=["doc_name", "para_idx", "topic"])
    assignments["level1"] = assignments["topic"].map(parent)
    if assignments["level1"].isna().any():
        orphans = sorted(assignments.loc[assignments["level1"].isna(), "topic"].unique())
        raise ValueError(f"level-2 topics with no level-1 parent in taxonomy_v1: {orphans}")
    n_assign = len(assignments)
    assignments = assignments.merge(
        paragraphs[["doc_name", "para_idx", "year", "era", "speech_type"]],
        on=["doc_name", "para_idx"],
        validate="many_to_one",
    )
    # Structurally unreachable today — `assignments` is built by iterating
    # `paragraphs`, so its key set is a subset by construction. Guarded anyway:
    # "safe today" is exactly what the positional-alignment bug this repo's merge
    # convention exists to prevent looked like, and the same gap was closed at
    # `exemplar_quotes`. Only the left length can move, so only it is asserted.
    _require_full_merge(
        len(assignments), {"assignments": n_assign}, "assignments x paragraphs"
    )

    paragraphs = paragraphs.drop(columns=["topics"])
    return (
        paragraphs.sort_values(["doc_name", "para_idx"]).reset_index(drop=True),
        assignments.sort_values(["doc_name", "para_idx", "topic"]).reset_index(drop=True),
    )


def level_topics(taxonomy: dict, level: str) -> list[str]:
    """Canonical topic names at `level` ("level2" or "level1"), sorted.

    Sourced from the taxonomy rather than from the observed labels so a topic
    with zero paragraphs in a slice still gets a row instead of vanishing.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return sorted({entry["name"] for entry in taxonomy[level]})


def level_assignments(assignments: pd.DataFrame, level: str) -> tuple[str, pd.DataFrame]:
    """The column a level is counted on, plus `assignments` deduped for it.

    Level-2 rows are already unique per (paragraph, topic) — `normalize_topics`
    guarantees it — so nothing is dropped there. At level 1 a paragraph carrying
    two topics from the SAME domain must count ONCE for that domain, or a rollup
    share can exceed the share of paragraphs it is built from.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if level == "level2":
        return "topic", assignments
    return "level1", assignments.drop_duplicates(["doc_name", "para_idx", "level1"])


def slice_for(
    paragraphs: pd.DataFrame, assignments: pd.DataFrame, treatment: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to the paragraphs a treatment is computed over.

    `raw` and `genre_standardized` see the whole corpus (the latter reweights
    rather than filters); `sotu` keeps annual messages only.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}, got {treatment!r}")
    if treatment != "sotu":
        return paragraphs, assignments
    return (
        paragraphs[paragraphs["speech_type"] == SOTU_TYPE],
        assignments[assignments["speech_type"] == SOTU_TYPE],
    )


# --------------------------------------------------------------------------
# attention curves
# --------------------------------------------------------------------------


def _year_index(paragraphs: pd.DataFrame) -> pd.RangeIndex:
    return pd.RangeIndex(
        int(paragraphs["year"].min()), int(paragraphs["year"].max()) + 1, name="year"
    )


def _counts(
    paragraphs: pd.DataFrame,
    assignments: pd.DataFrame,
    topic_col: str,
    topics: list[str],
    years: pd.RangeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-year topic counts (year x topic) and per-year paragraph totals."""
    num = (
        assignments.groupby(["year", topic_col], observed=True).size().unstack(fill_value=0)
        if len(assignments)
        else pd.DataFrame(index=pd.Index([], name="year"))
    )
    num = num.reindex(index=years, fill_value=0).reindex(columns=topics, fill_value=0)
    den = paragraphs.groupby("year").size().reindex(years, fill_value=0)
    return num.astype(float), den.astype(float)


def _smooth(frame: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Centered rolling SUM (numerator and denominator smoothed separately).

    Matching `site.py`/`indices.py`: summing both sides then dividing weights
    sparse years by their evidence instead of averaging ratios, which would let
    a 4-paragraph year shout as loudly as a 400-paragraph one.
    """
    return frame.rolling(window, center=True, min_periods=1).sum()


def _combine_strata(
    num_by_genre: dict[str, pd.DataFrame],
    den_by_genre: dict[str, pd.Series],
    weights: dict[str, float],
    min_stratum: int,
) -> pd.DataFrame:
    """Post-stratified share: within-genre shares recombined at fixed weights.

    Args:
        num_by_genre: genre -> (year x topic) topic counts.
        den_by_genre: genre -> year -> paragraph totals.
        weights: genre -> fixed corpus-wide paragraph share (need not sum to 1;
            it is renormalized over eligible genres each year).
        min_stratum: A genre needs at least this many paragraphs in a year to be
            eligible; see the missing-genre-year rule in the module docstring.

    Returns:
        A (year x topic) frame of weighted shares. NaN only where a year holds
        no paragraphs of any genre at all.
    """
    genres = list(num_by_genre)
    any_frame = num_by_genre[genres[0]]
    years, topics = any_frame.index, any_frame.columns

    den = pd.DataFrame({g: den_by_genre[g] for g in genres}, index=years)
    eligible = den >= min_stratum
    # Fall back to "any genre actually present" for years where no stratum
    # clears the threshold, so the rule can never emit NaN off a live year.
    fallback = den > 0
    eligible = eligible.where(eligible.any(axis=1), fallback)

    w = pd.DataFrame(
        {g: np.full(len(years), float(weights.get(g, 0.0))) for g in genres}, index=years
    )
    w = w.where(eligible, 0.0)
    total = w.sum(axis=1)
    w = w.div(total.replace(0.0, np.nan), axis=0)

    out = pd.DataFrame(0.0, index=years, columns=topics)
    for g in genres:
        share_g = num_by_genre[g].div(den[g].replace(0.0, np.nan), axis=0)
        out = out.add(share_g.mul(w[g], axis=0).fillna(0.0), fill_value=0.0)
    return out.where(total.gt(0.0), np.nan)


def attention_curves(
    paragraphs: pd.DataFrame,
    assignments: pd.DataFrame,
    taxonomy: dict,
    level: str = "level2",
    treatment: str = "raw",
    window: int = SMOOTH_WINDOW,
) -> pd.DataFrame:
    """Per-topic, per-year attention curves under one genre treatment.

    Args:
        paragraphs: Full-corpus paragraph frame from `load_inputs`.
        assignments: Full-corpus (paragraph, topic) frame from `load_inputs`.
        taxonomy: Parsed `taxonomy_v1.json`.
        level: "level2" (50 topics) or "level1" (17 domains, rolled up so a
            paragraph carrying two topics in one domain still counts once).
        treatment: One of `TREATMENTS`.
        window: Centered smoothing window in years.

    Returns:
        Long frame with columns `level, treatment, topic, year, n_topic,
        n_total, share, n_topic_window, n_total_window, share_smooth`.
        `n_topic_window`/`n_total_window` are always UNWEIGHTED paragraph counts
        — under `genre_standardized` the share is reweighted but the sufficiency
        test must still be about real sample size.
    """
    topics = level_topics(taxonomy, level)
    years = _year_index(paragraphs)
    paras, assign = slice_for(paragraphs, assignments, treatment)
    topic_col, assign = level_assignments(assign, level)

    num, den = _counts(paras, assign, topic_col, topics, years)
    num_w, den_w = _smooth(num, window), _smooth(den, window)

    if treatment == "genre_standardized":
        genres = sorted(paragraphs["speech_type"].unique())
        weights = (
            paragraphs["speech_type"].value_counts(normalize=True).reindex(genres).fillna(0.0)
        ).to_dict()
        num_g, den_g, numw_g, denw_g = {}, {}, {}, {}
        for genre in genres:
            gp = paras[paras["speech_type"] == genre]
            ga = assign[assign["speech_type"] == genre]
            n_g, d_g = _counts(gp, ga, topic_col, topics, years)
            num_g[genre], den_g[genre] = n_g, d_g
            numw_g[genre], denw_g[genre] = _smooth(n_g, window), _smooth(d_g, window)
        share = _combine_strata(num_g, den_g, weights, min_stratum=1)
        share_smooth = _combine_strata(numw_g, denw_g, weights, MIN_GENRE_STRATUM)
    else:
        share = num.div(den.replace(0.0, np.nan), axis=0)
        share_smooth = num_w.div(den_w.replace(0.0, np.nan), axis=0)

    # pandas >= 3 `stack()` keeps NA rows (the old `dropna=False`), so a year
    # whose share is undefined stays in the frame instead of silently vanishing.
    out = pd.concat(
        [
            num.stack().rename("n_topic"),
            share.stack().rename("share"),
            num_w.stack().rename("n_topic_window"),
            share_smooth.stack().rename("share_smooth"),
        ],
        axis=1,
    ).reset_index(names=["year", "topic"])
    out["n_total"] = out["year"].map(den)
    out["n_total_window"] = out["year"].map(den_w)
    out.insert(0, "treatment", treatment)
    out.insert(0, "level", level)
    return out[
        [
            "level", "treatment", "topic", "year", "n_topic", "n_total", "share",
            "n_topic_window", "n_total_window", "share_smooth",
        ]
    ].sort_values(["topic", "year"]).reset_index(drop=True)


def corex_curves(
    paragraphs: pd.DataFrame, treatment: str = "raw", window: int = SMOOTH_WINDOW
) -> pd.DataFrame:
    """Legacy CorEx issue curves on the SAME paragraph slice, for the cross-check.

    Args:
        paragraphs: Full-corpus paragraph frame from `load_inputs`.
        treatment: `raw`/`sotu` filter the slice; `genre_standardized` is treated
            as `raw` here on purpose — the CorEx side is used only as a
            collapse/persist indicator, and reweighting it would mix a
            post-stratified curve with an unstratified one in the same ratio.
        window: Centered smoothing window in years.

    Returns:
        A (year x legacy issue) frame of smoothed shares, columns named by the
        crosswalk's issue names (including "Security & peace" = Discovered 5).

    Raises:
        ValueError: If the keyed merge onto `paragraph_issues` drops rows.
    """
    issues = pd.read_parquet(
        PARA_LABELS_PATH, columns=["doc_name", "para_idx", *COREX_COLUMNS.values()]
    )
    slice_paras = (
        paragraphs[paragraphs["speech_type"] == SOTU_TYPE] if treatment == "sotu"
        else paragraphs
    )
    merged = slice_paras[["doc_name", "para_idx", "year"]].merge(
        issues, on=["doc_name", "para_idx"], validate="one_to_one"
    )
    _require_full_merge(
        len(merged), {"slice": len(slice_paras)}, f"corex_curves[{treatment}]"
    )
    years = _year_index(paragraphs)
    den = merged.groupby("year").size().reindex(years, fill_value=0).astype(float)
    num = (
        merged.groupby("year")[list(COREX_COLUMNS.values())].sum()
        .reindex(years, fill_value=0).astype(float)
    )
    num.columns = list(COREX_COLUMNS)
    smoothed = _smooth(num, window).div(_smooth(den, window).replace(0.0, np.nan), axis=0)
    return smoothed


# --------------------------------------------------------------------------
# lifecycles
# --------------------------------------------------------------------------


def substantive_mask(curve: pd.DataFrame) -> pd.Series:
    """Boolean mask of the years a topic is substantively present.

    Args:
        curve: One topic's rows from `attention_curves`, year-sorted.

    Returns:
        A boolean Series aligned to `curve` implementing the three-part
        pre-registered threshold documented at the top of this module.
    """
    peak = curve["share_smooth"].max()
    if not np.isfinite(peak) or peak <= 0:
        return pd.Series(False, index=curve.index)
    floor = max(ABS_SHARE_FLOOR, REL_PEAK_FRACTION * float(peak))
    return (
        (curve["n_total_window"] >= MIN_WINDOW_PARAGRAPHS)
        & (curve["n_topic_window"] >= MIN_TOPIC_PARAGRAPHS)
        & (curve["share_smooth"] >= floor)
    )


def lifecycle_class(
    first_year: int, last_year: int, max_gap: int, record_start: int, record_end: int
) -> str:
    """The pre-registered born/died/revived/persistent rule, stated in one place.

    Over the observed record `[record_start, record_end]`, first match wins (so a
    topic that both died and had a gap is reported `died`):

      * `died` — last substantive year >= DEATH_GAP_YEARS (40) before record end;
      * `revived` — a >= REVIVAL_GAP_YEARS (30) gap between substantive years;
      * `born` — first substantive year >= BIRTH_GAP_YEARS (40) after record start;
      * `persistent` — otherwise.

    `max_gap` is the largest run of non-substantive years between substantive
    ones; `first_year` is already clamped to the topic's own evidence.
    """
    if last_year <= record_end - DEATH_GAP_YEARS:
        return "died"
    if max_gap >= REVIVAL_GAP_YEARS:
        return "revived"
    if first_year >= record_start + BIRTH_GAP_YEARS:
        return "born"
    return "persistent"


def topic_lifecycle(curve: pd.DataFrame, n_paragraphs: int) -> dict:
    """Extract one topic's life history from its attention curve.

    Args:
        curve: One topic's rows from `attention_curves` (any treatment).
        n_paragraphs: The topic's corpus-wide paragraph count in that slice,
            used only for the under-powered guard.

    Returns:
        A dict of lifecycle fields: first/peak/last substantive year, peak era
        and share, rise and fall rates in share-points per decade, era of
        relevance, the largest internal gap, and `lifecycle_class`.
    """
    curve = curve.sort_values("year")
    mask = substantive_mask(curve)
    out: dict = {
        "n_paragraphs": int(n_paragraphs),
        "under_powered": bool(n_paragraphs < UNDER_POWERED_N),
        "n_substantive_years": int(mask.sum()),
        "first_year": np.nan, "peak_year": np.nan, "last_year": np.nan,
        "peak_share": np.nan, "first_share": np.nan, "last_share": np.nan,
        "peak_era": None, "era_of_relevance": None,
        "rise_rate_per_decade": np.nan, "fall_rate_per_decade": np.nan,
        "max_gap_years": np.nan, "lifecycle_class": "absent",
    }
    if not mask.any():
        if out["under_powered"]:
            out["lifecycle_class"] = "under_powered"
        return out

    # A CENTERED window leads and lags the data by window//2 years, so the
    # smoothed curve can be substantive up to two years before the topic's first
    # real paragraph. Intersect the substantive span with the years the topic
    # actually occurs, so a birth can never predate the topic's own evidence.
    live = curve[mask]
    present = curve.loc[curve["n_topic"] > 0, "year"]
    first_year, last_year = int(live["year"].min()), int(live["year"].max())
    if len(present):
        first_year = min(max(first_year, int(present.min())), last_year)
        last_year = max(min(last_year, int(present.max())), first_year)
        live = live[live["year"].between(first_year, last_year)]
        if live.empty:
            # TOTAL clamp miss. The clamp excludes every substantive year when a
            # topic's only substantive years sit at the OUTER edge of the window
            # that made them substantive. Verified example: 30 paragraphs in 1798,
            # 10 in 1802 (3 on-topic), 10 in 1828 (3 on-topic), 30 in 1832 is
            # substantive only in 1800/1830 while the topic only appears in
            # 1802/1828, so the clamp yields [1802, 1828] — neither live year.
            # Without this fallback the `idxmax` below raises on an empty frame.
            # Pinned by
            # `test_the_clamp_falling_outside_every_substantive_year_falls_back`.
            live = curve[mask]
        # PARTIAL clamp miss. Even when `live` survives, the clamped boundary
        # need not be one of the surviving substantive years: with mask
        # [1800, 1803, 1804, 1828, 1829, 1830] and evidence in [1802, 1828] the
        # span clamps to [1802, 1828] and `live` is {1803, 1804, 1828} — non-empty,
        # but 1802 is not in it, so the `first_share` lookup below indexed an empty
        # selection and raised `IndexError`. Recompute BOTH boundaries from the
        # frame actually in hand instead of assuming the pre-clamp boundary
        # survived; this is what the total-miss branch above already does. When
        # the boundary did survive (every real-corpus row today) min/max return it
        # unchanged, so this is a no-op there. Pinned by
        # `test_a_clamp_boundary_that_is_not_itself_a_substantive_year_does_not_raise`.
        first_year, last_year = int(live["year"].min()), int(live["year"].max())

    years = live["year"].to_numpy()
    peak_row = live.loc[live["share_smooth"].idxmax()]
    peak_year = int(peak_row["year"])
    peak_share = float(peak_row["share_smooth"])
    first_share = float(live.loc[live["year"] == first_year, "share_smooth"].iloc[0])
    last_share = float(live.loc[live["year"] == last_year, "share_smooth"].iloc[0])
    gaps = np.diff(years) - 1
    max_gap = int(gaps.max()) if len(gaps) else 0

    cls = lifecycle_class(
        first_year, last_year, max_gap,
        record_start=int(curve["year"].min()), record_end=int(curve["year"].max()),
    )

    out.update({
        "first_year": first_year,
        "peak_year": peak_year,
        "last_year": last_year,
        "peak_share": peak_share,
        "first_share": first_share,
        "last_share": last_share,
        "peak_era": era_name(peak_year),
        "era_of_relevance": era_span_label(first_year, last_year),
        "rise_rate_per_decade": (
            (peak_share - first_share) / (peak_year - first_year) * 10
            if peak_year > first_year else np.nan
        ),
        "fall_rate_per_decade": (
            (peak_share - last_share) / (last_year - peak_year) * 10
            if last_year > peak_year else np.nan
        ),
        "max_gap_years": max_gap,
        # The guard must not silently vanish: an under-powered topic keeps its
        # descriptive columns (they are the evidence it is thin) but loses its
        # lifecycle class, so a consumer filtering on the class cannot chart it.
        "lifecycle_class": "under_powered" if out["under_powered"] else cls,
    })
    return out


def extract_lifecycles(curves: pd.DataFrame, topic_totals: pd.Series) -> pd.DataFrame:
    """Run `topic_lifecycle` over every topic in a curve frame.

    Args:
        curves: Output of `attention_curves` for one (level, treatment).
        topic_totals: topic -> corpus-wide paragraph count in that slice.

    Returns:
        One row per topic, with the `level`/`treatment` carried through.
    """
    level = curves["level"].iloc[0]
    treatment = curves["treatment"].iloc[0]
    rows = []
    for topic, sub in curves.groupby("topic", sort=True):
        row = {"level": level, "treatment": treatment, "topic": topic}
        row.update(topic_lifecycle(sub, int(topic_totals.get(topic, 0))))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# rename vs death
# --------------------------------------------------------------------------


def invert_crosswalk(crosswalk: dict) -> dict[str, list[str]]:
    """Invert `crosswalk_v1.json` to level-2 topic -> legacy issue parents.

    The mapping is many-to-many by construction (`Chinese Immigration &
    Exclusion` sits under both `Immigration` and `Civil rights & race`), so the
    value is a sorted list and the multi-parent rule is applied downstream, never
    by silently taking the first entry.
    """
    inverted: dict[str, set[str]] = {}
    for mapping in crosswalk["mappings"]:
        for topic in mapping["level2_topics"]:
            inverted.setdefault(topic, set()).add(mapping["legacy_issue"])
    return {topic: sorted(parents) for topic, parents in inverted.items()}


def _decline_ratio(
    series: pd.Series, peak_year: int, last_year: int, half_width: int = PEAK_WINDOW_HALF
) -> float:
    """How much of its peak-era level a smoothed curve retains after a death.

    The denominator is the mean over the topic's PEAK WINDOW
    (`peak_year` +/- `half_width`) rather than the single peak year: a one-year
    denominator is noisy, and indexing a legacy CorEx curve at the LLM topic's
    exact peak year would be arbitrary. The numerator is the mean over every
    year after `last_year`.

    Args:
        series: A smoothed year-indexed share curve (LLM topic or CorEx issue).
        peak_year: The dying topic's peak year.
        last_year: The dying topic's last substantive year.
        half_width: Half-width of the peak window, in years.

    Returns:
        0.0 for total collapse, ~1.0 for "unchanged", > 1 when the curve is
        HIGHER after the death than during it (which means the curve is not
        tracking this topic at all). NaN if either window is empty.
    """
    peak_window = series[series.index.to_series().between(
        peak_year - half_width, peak_year + half_width
    )]
    post = series[series.index > last_year]
    if peak_window.empty or post.empty:
        return np.nan
    base, after = peak_window.mean(), post.mean()
    if not np.isfinite(base) or base <= 0 or not np.isfinite(after):
        return np.nan
    return float(after / base)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    """Pearson r over the years where both curves are defined (NaN if degenerate)."""
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].std() == 0 or joined.iloc[:, 1].std() == 0:
        return np.nan
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def find_successor(
    dying: pd.Series,
    lifecycles: pd.DataFrame,
    smoothed: pd.DataFrame,
    parents: dict[str, str],
    inverted_crosswalk: dict[str, list[str]],
    same_domain: bool = True,
) -> tuple[str | None, float]:
    """Name the topic that plausibly took over from a dying topic.

    A candidate must clear four independent filters, because over a 240-year
    record ANY two topics occupying disjoint periods are anti-correlated —
    negative correlation alone would happily nominate the Mexican War's
    "successor" as World War II.

      1. Same level-1 domain (unless `same_domain` is False — the diagnostic).
      2. Its FIRST substantive year falls inside the dying topic's DECLINE
         WINDOW `[peak_year, last_year]`. This is the handoff test, and it is
         what separates a rename from mere sequence: Civil Rights first becomes
         substantive in 1865 while Slavery is still dying (last year 1868),
         whereas Vietnam is born 40 years after World War I ends.
      3. It outlives the dying topic (`last_year > dying.last_year`).
      4. It shares at least one legacy crosswalk parent with the dying topic —
         the coarse lexical taxonomy must at least agree the two are the same
         kind of concern. KNOWN DEVIATION (defect #10, deliberately NOT fixed at
         SIMPLIFY): when the DYING topic has no crosswalk parent at all, filter 4
         is skipped instead of rejecting everyone, so `Constitutional Union &
         Federalism` and `Territorial Organization...` are judged on filters 1-3.
         Making it unconditional is a BEHAVIOUR change, not a cleanup — it flips
         the former to `unresolved_death` under `sotu` and blanks six
         `cross_domain_candidate` diagnostics, so it needs a method decision.

    Among survivors the winner is the MOST anti-correlated, per the spec's
    "strongly anti-correlated"; ranking by size instead would just elect the
    biggest topic in the domain.

    Args:
        dying: The dying topic's lifecycle row.
        lifecycles: All level-2 lifecycle rows for the same treatment.
        smoothed: (year x topic) smoothed shares for the same treatment.
        parents: level-2 topic -> level-1 domain.
        inverted_crosswalk: level-2 topic -> legacy issue parents.
        same_domain: Restrict candidates to the dying topic's own level-1 domain.
            False produces the cross-domain DIAGNOSTIC only — never used to
            classify.

    Returns:
        `(successor topic or None, Pearson r with the dying topic's curve)`.
    """
    domain = parents.get(dying["topic"])
    legacy = set(inverted_crosswalk.get(dying["topic"], []))
    best: tuple[str, float] | None = None
    for _, cand in lifecycles.iterrows():
        if cand["topic"] == dying["topic"]:
            continue
        if cand["lifecycle_class"] in ("absent", "under_powered"):
            continue
        if same_domain and parents.get(cand["topic"]) != domain:
            continue
        if not np.isfinite(cand["first_year"]) or not np.isfinite(cand["last_year"]):
            continue
        if not dying["peak_year"] <= cand["first_year"] <= dying["last_year"]:
            continue
        if cand["last_year"] <= dying["last_year"]:
            continue
        if legacy and not legacy & set(inverted_crosswalk.get(cand["topic"], [])):
            continue
        r = _pearson(smoothed[dying["topic"]], smoothed[cand["topic"]])
        if not np.isfinite(r) or r >= 0:
            continue
        if best is None or r < best[1]:
            best = (cand["topic"], r)
    return (None, np.nan) if best is None else best


def classify_deaths(
    lifecycles: pd.DataFrame,
    curves: pd.DataFrame,
    corex: pd.DataFrame,
    inverted_crosswalk: dict[str, list[str]],
    parents: dict[str, str],
) -> pd.DataFrame:
    """Rename-vs-death classification for every apparently dead level-2 topic.

    Both signals from the spec are computed: the LLM<->CorEx crosswalk
    divergence (with the ALL-parents-must-collapse multi-parent rule) and
    within-domain successor detection. See the module docstring for the decision
    table and the rationale for each rule.

    Args:
        lifecycles: Level-2 lifecycle rows for ONE treatment.
        curves: `attention_curves` output for the same (level2, treatment).
        corex: `corex_curves` output for the same treatment.
        inverted_crosswalk: Output of `invert_crosswalk`.
        parents: level-2 topic -> level-1 domain.

    Returns:
        `lifecycles` with `rename_class`, `legacy_parents`, `corex_parent_ratios`,
        `corex_decline_ratio`, `corex_persists`, `llm_decline_ratio`,
        `successor_topic`, `successor_corr`, and the `cross_domain_candidate`
        diagnostic appended.

        `corex_decline_ratio` is the **max** over the topic's finite parent
        ratios — the most-persisting parent — not a mean and not the first
        parent. That is deliberately the same scalar the ANY-parent-persists rule
        turns on, so the reported number and the verdict can never disagree; it
        is NaN when no parent has a finite ratio. A parent whose ratio exceeds 1
        carries no information (the legacy issue is higher after the death than
        during the peak) and is still counted as persisting, for the reason given
        in the module docstring. `corex_parent_ratios` keeps every parent's own
        ratio so a different reduction can be recomputed without a rerun.
    """
    smoothed = curves.pivot(index="year", columns="topic", values="share_smooth")
    out = lifecycles.copy()
    for col, default in (
        ("rename_class", None), ("legacy_parents", None), ("corex_parent_ratios", None),
        ("corex_decline_ratio", np.nan), ("corex_persists", None),
        ("llm_decline_ratio", np.nan), ("successor_topic", None),
        ("successor_corr", np.nan), ("cross_domain_candidate", None),
    ):
        out[col] = default

    for i, row in out.iterrows():
        parents_of = inverted_crosswalk.get(row["topic"], [])
        out.at[i, "legacy_parents"] = "; ".join(parents_of) or None
        if row["lifecycle_class"] != "died":
            continue
        peak_year, last_year = int(row["peak_year"]), int(row["last_year"])
        out.at[i, "llm_decline_ratio"] = _decline_ratio(
            smoothed[row["topic"]], peak_year, last_year
        )
        ratios = {
            parent: _decline_ratio(corex[parent], peak_year, last_year)
            for parent in parents_of if parent in corex.columns
        }
        out.at[i, "corex_parent_ratios"] = "; ".join(
            f"{k}={v:.2f}" for k, v in sorted(ratios.items())
        ) or None
        finite = [v for v in ratios.values() if np.isfinite(v)]
        # MULTI-PARENT RULE: a true death needs EVERY legacy parent to collapse,
        # so CorEx persists if ANY parent persists. With no usable parent the
        # cross-check is unavailable and we must not claim agreement. A parent
        # whose ratio is > 1 is uninformative but still votes "persists" — same
        # reflex, see the module docstring.
        persists = not finite or any(v > COREX_COLLAPSE_RATIO for v in finite)
        # MAX, not mean: report the MOST-PERSISTING parent, which is exactly the
        # one the ANY rule above turns on, so the published scalar always
        # explains the verdict sitting beside it.
        out.at[i, "corex_decline_ratio"] = max(finite) if finite else np.nan
        out.at[i, "corex_persists"] = bool(persists)

        successor, corr = find_successor(
            row, lifecycles, smoothed, parents, inverted_crosswalk
        )
        out.at[i, "successor_topic"] = successor
        out.at[i, "successor_corr"] = corr
        cross, _ = find_successor(
            row, lifecycles, smoothed, parents, inverted_crosswalk, same_domain=False
        )
        out.at[i, "cross_domain_candidate"] = cross

        # PRECEDENCE: a named successor wins. Signal 2 is concrete and checkable
        # (this specific topic, in this domain, rose as that one fell); signal 1
        # runs through a 15-bucket crosswalk whose anchor vocabulary is itself
        # period-bound — "Civil rights & race" is anchored on slavery-era words,
        # so it can appear to "collapse" for exactly the renaming reason we are
        # testing for. `corex_persists` is retained on every row so the reader
        # can see whether the two signals agreed.
        if successor is not None:
            out.at[i, "rename_class"] = "rename"
        elif not persists:
            out.at[i, "rename_class"] = "true_death"
        else:
            out.at[i, "rename_class"] = "unresolved_death"
    return out


# --------------------------------------------------------------------------
# speech-clustered bootstrap
# --------------------------------------------------------------------------


def _speech_design(
    paragraphs: pd.DataFrame, assignments: pd.DataFrame, topic_col: str, topics: list[str]
) -> dict:
    """Per-speech matrices the bootstrap resamples over (speeches, never paragraphs)."""
    speeches = (
        paragraphs.drop_duplicates("doc_name")[["doc_name", "era", "speech_type"]]
        .sort_values("doc_name").reset_index(drop=True)
    )
    n_para = (
        paragraphs.groupby("doc_name").size().reindex(speeches["doc_name"], fill_value=0)
        .to_numpy(float)
    )
    counts = (
        assignments.pivot_table(
            index="doc_name", columns=topic_col, aggfunc="size", fill_value=0, observed=True
        )
        if len(assignments) else pd.DataFrame(index=pd.Index([], name="doc_name"))
    )
    counts = counts.reindex(index=speeches["doc_name"], fill_value=0).reindex(
        columns=topics, fill_value=0
    )
    era_labels = [label for label, _, _ in ERAS]
    genres = sorted(paragraphs["speech_type"].unique())
    era_idx = speeches["era"].astype(str).map({e: i for i, e in enumerate(era_labels)})
    genre_idx = speeches["speech_type"].map({g: i for i, g in enumerate(genres)})
    return {
        "counts": counts.to_numpy(float),
        "n_para": n_para,
        "era_idx": era_idx.to_numpy(int),
        "genre_idx": genre_idx.to_numpy(int),
        "is_sotu": (speeches["speech_type"] == SOTU_TYPE).to_numpy(),
        "eras": era_labels,
        "genres": genres,
        "topics": topics,
        "genre_weights": np.array(
            [float((paragraphs["speech_type"] == g).sum()) for g in genres]
        ),
    }


def _era_shares(design: dict, w: np.ndarray, treatment: str) -> np.ndarray:
    """Era x topic shares for one set of speech weights (all ones = observed)."""
    n_eras, n_genres = len(design["eras"]), len(design["genres"])
    counts, n_para = design["counts"], design["n_para"]
    if treatment == "sotu":
        counts = counts * design["is_sotu"][:, None]
        n_para = n_para * design["is_sotu"]
    weighted = counts * w[:, None]
    para_w = n_para * w

    if treatment != "genre_standardized":
        num = np.zeros((n_eras, counts.shape[1]))
        den = np.zeros(n_eras)
        np.add.at(num, design["era_idx"], weighted)
        np.add.at(den, design["era_idx"], para_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den[:, None] > 0, num / den[:, None], np.nan)

    stratum = design["era_idx"] * n_genres + design["genre_idx"]
    num = np.zeros((n_eras * n_genres, counts.shape[1]))
    den = np.zeros(n_eras * n_genres)
    np.add.at(num, stratum, weighted)
    np.add.at(den, stratum, para_w)
    num = num.reshape(n_eras, n_genres, -1)
    den = den.reshape(n_eras, n_genres)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(den[:, :, None] > 0, num / np.maximum(den, 1)[:, :, None], 0.0)
    # Eligibility here is `den > 0`, deliberately NOT `_combine_strata`'s
    # `MIN_GENRE_STRATUM` — see the missing-genre-year rule in the module
    # docstring for why an era-wide stratum does not need that guard. As there,
    # the fixed corpus weights are renormalized over the eligible strata.
    weights = design["genre_weights"][None, :] * (den > 0)
    total = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(total > 0, weights / np.maximum(total, 1e-12), np.nan)
    return np.einsum("eg,egt->et", np.nan_to_num(weights), share) * np.where(
        total.ravel()[:, None] > 0, 1.0, np.nan
    )


def bootstrap_era_shares(
    paragraphs: pd.DataFrame,
    assignments: pd.DataFrame,
    taxonomy: dict,
    level: str = "level2",
    treatment: str = "raw",
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Era-level attention shares with speech-clustered bootstrap CIs.

    Speeches — not paragraphs — are resampled with replacement, because
    within-speech ICC runs to 0.17 and paragraph resampling would understate the
    interval badly. The draw matrix is generated from `seed` in a fixed order, so
    the result is byte-reproducible.

    Under `sotu` the resample is still over ALL speeches with non-SOTU speeches
    contributing zero, which correctly propagates the uncertainty in how many
    annual messages an era happens to contain.

    Args:
        paragraphs: Full-corpus paragraph frame from `load_inputs`.
        assignments: Full-corpus (paragraph, topic) frame from `load_inputs`.
        taxonomy: Parsed `taxonomy_v1.json`.
        level: "level2" or "level1".
        treatment: One of `TREATMENTS`.
        n_draws: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        One row per (topic, era) with `share` (the observed point estimate),
        `share_lo`/`share_hi` (percentile CI), `n_paragraphs` (era sample size in
        the slice) and `n_topic_paragraphs`.
    """
    topics = level_topics(taxonomy, level)
    topic_col, assign = level_assignments(assignments, level)
    design = _speech_design(paragraphs, assign, topic_col, topics)

    n_speeches = len(design["n_para"])
    ones = np.ones(n_speeches)
    observed = _era_shares(design, ones, treatment)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_speeches, size=(n_draws, n_speeches))
    replicates = np.empty((n_draws, *observed.shape))
    for b in range(n_draws):
        w = np.bincount(draws[b], minlength=n_speeches).astype(float)
        replicates[b] = _era_shares(design, w, treatment)
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(replicates, CI_LOW, axis=0)
        hi = np.nanpercentile(replicates, CI_HIGH, axis=0)

    slice_paras, slice_assign = slice_for(paragraphs, assign, treatment)
    era_n = slice_paras.groupby("era", observed=False).size()
    topic_n = slice_assign.groupby([topic_col, "era"], observed=False).size()

    rows = []
    for e, era in enumerate(design["eras"]):
        for t, topic in enumerate(topics):
            rows.append({
                "level": level, "treatment": treatment, "topic": topic, "era": era,
                "share": observed[e, t], "share_lo": lo[e, t], "share_hi": hi[e, t],
                "n_paragraphs": int(era_n.get(era, 0)),
                "n_topic_paragraphs": int(topic_n.get((topic, era), 0)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# exemplar quotes (from paragraph text, never from a model)
# --------------------------------------------------------------------------


def exemplar_quotes(
    topic: str,
    assignments: pd.DataFrame,
    year_lo: int,
    year_hi: int,
    n: int = 3,
    min_words: int = 35,
    max_words: int = 160,
    paragraphs_path: Path = PARAGRAPHS_PATH,
) -> list[dict]:
    """Deterministic exemplar paragraphs for a topic inside a year window.

    Args:
        topic: Canonical level-2 topic name.
        assignments: The (paragraph, topic) frame from `load_inputs`.
        year_lo: Inclusive start year.
        year_hi: Inclusive end year.
        n: How many exemplars to return.
        min_words: Skip paragraphs shorter than this (fragments read badly).
        max_words: Skip paragraphs longer than this (unquotable walls of text).
        paragraphs_path: Source of paragraph text.

    Returns:
        Up to `n` dicts with `doc_name`, `para_idx`, `year`, `word_count`, `text`,
        drawn evenly across the window and key-sorted, so the same call always
        returns the same quotes.

    Raises:
        ValueError: If a pooled `(doc_name, para_idx)` has no row in the text
            parquet. `validate="one_to_one"` catches duplicate keys but NOT
            diverging key SETS, so without this the inner join would quietly
            narrow the pool — and a TOTAL miss would return `[]`, which is
            indistinguishable from "this topic has no quotable paragraphs here".
    """
    text = pd.read_parquet(paragraphs_path)
    pool = assignments[
        (assignments["topic"] == topic)
        & assignments["year"].between(year_lo, year_hi)
    ][["doc_name", "para_idx", "year"]]
    merged = pool.merge(text, on=["doc_name", "para_idx"], validate="one_to_one")
    # Only the POOL must survive: `text` is the whole corpus and is expected to
    # be much larger, so it is deliberately not asserted against.
    _require_full_merge(len(merged), {"pool": len(pool)}, f"exemplar_quotes[{topic}]")
    pool = merged[merged["word_count"].between(min_words, max_words)]
    pool = pool.sort_values(["year", "doc_name", "para_idx"]).reset_index(drop=True)
    if pool.empty:
        return []
    picks = np.linspace(0, len(pool) - 1, num=min(n, len(pool))).round().astype(int)
    return pool.loc[sorted(set(picks))].to_dict("records")


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "level", "treatment", "topic", "level1", "n_paragraphs", "under_powered",
    "first_year", "peak_year", "peak_era", "peak_share", "last_year",
    "first_share", "last_share", "rise_rate_per_decade", "fall_rate_per_decade",
    "era_of_relevance", "dominant_era", "n_substantive_years", "max_gap_years",
    "lifecycle_class", "rename_class", "legacy_parents", "corex_parent_ratios",
    "corex_decline_ratio", "corex_persists", "llm_decline_ratio",
    "successor_topic", "successor_corr", "cross_domain_candidate",
    "first_era_share", "first_era_share_lo", "first_era_share_hi",
    "peak_era_share", "peak_era_share_lo", "peak_era_share_hi",
    "final_era_share", "final_era_share_lo", "final_era_share_hi",
]


def _attach_era_shares(
    life: pd.DataFrame, era_grid: pd.DataFrame, prefix: str, eras: list
) -> None:
    """Copy one era's share and CI bounds onto every lifecycle row, in place.

    Args:
        life: Lifecycle rows, mutated to gain `{prefix}_share{,_lo,_hi}`.
        era_grid: `bootstrap_era_shares` output indexed by `(topic, era)`.
        prefix: Output column prefix, e.g. `peak_era`.
        eras: One era label per row of `life` — a constant era for the
            first/final columns, each topic's own peak era for `peak_era_*`. A
            non-string entry (a topic with no peak era) yields NaN.
    """
    for suffix, col in (("", "share"), ("_lo", "share_lo"), ("_hi", "share_hi")):
        life[f"{prefix}_share{suffix}"] = [
            era_grid.loc[(topic, era), col] if isinstance(era, str) else np.nan
            for topic, era in zip(life["topic"], eras)
        ]


def build_lifecycle_table(
    paragraphs: pd.DataFrame,
    assignments: pd.DataFrame,
    taxonomy: dict,
    crosswalk: dict,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    """Assemble the full lifecycle table across both levels and all treatments.

    Args:
        paragraphs: Full-corpus paragraph frame from `load_inputs`.
        assignments: Full-corpus (paragraph, topic) frame from `load_inputs`.
        taxonomy: Parsed `taxonomy_v1.json`.
        crosswalk: Parsed `crosswalk_v1.json`.
        n_draws: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        `(lifecycles, curves_by_key, era_by_key)` where the two dicts are keyed
        `(level, treatment)` so the caller (report writing, tests) can inspect the
        curves and era CI grids without recomputing them.
    """
    parents = level1_parents(taxonomy)
    inverted = invert_crosswalk(crosswalk)
    first_era, final_era = ERAS[0][0], ERAS[-1][0]

    curves_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    era_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    tables = []
    for level in LEVELS:
        for treatment in TREATMENTS:
            curves = attention_curves(paragraphs, assignments, taxonomy, level, treatment)
            curves_by_key[(level, treatment)] = curves
            _, slice_assign = slice_for(paragraphs, assignments, treatment)
            topic_col, slice_assign = level_assignments(slice_assign, level)
            life = extract_lifecycles(curves, slice_assign[topic_col].value_counts())

            if level == "level2":
                corex = corex_curves(paragraphs, treatment)
                life = classify_deaths(life, curves, corex, inverted, parents)
                life["level1"] = life["topic"].map(parents)
            else:
                life["level1"] = None

            era = bootstrap_era_shares(
                paragraphs, assignments, taxonomy, level, treatment, n_draws, seed
            )
            era_by_key[(level, treatment)] = era
            era_grid = era.set_index(["topic", "era"])
            life["dominant_era"] = life["topic"].map(
                era.loc[era.groupby("topic")["share"].idxmax()].set_index("topic")["era"]
            )
            _attach_era_shares(life, era_grid, "first_era", [first_era] * len(life))
            _attach_era_shares(life, era_grid, "final_era", [final_era] * len(life))
            _attach_era_shares(life, era_grid, "peak_era", list(life["peak_era"]))
            tables.append(life)

    out = pd.concat(tables, ignore_index=True)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[OUTPUT_COLUMNS]
    return (
        out.sort_values(["level", "treatment", "topic"]).reset_index(drop=True),
        curves_by_key,
        era_by_key,
    )


def write_lifecycles(table: pd.DataFrame, path: Path = LIFECYCLES_PATH) -> Path:
    """Write the lifecycle table, deterministically (byte-identical on rerun)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------
# ground-truth falsification checks
# --------------------------------------------------------------------------

def headline_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The (level2, raw) slice — the rows the ground-truth checks and the
    findings note quote. Every other slice is a robustness check against it."""
    return table[(table["level"] == "level2") & (table["treatment"] == "raw")]


# Each check is (label, topic, assertion) evaluated against the raw level-2
# lifecycle row. These are FALSIFICATION tests: a failure is reported, never
# tuned away, because a wrong lifecycle is a finding about the annotation layer.
GROUND_TRUTH = [
    ("Indian Affairs peaks 1830s-1870s and dies",
     "Indian Affairs, Removal & Allotment",
     lambda r: 1830 <= r["peak_year"] <= 1879 and r["lifecycle_class"] == "died"),
    ("Coinage/currency peaks in the 1890s free-silver era",
     "Coinage, Currency & Specie",
     lambda r: 1890 <= r["peak_year"] <= 1899),
    ("Terrorism is born around 2001",
     "Iraq, Gulf Wars, the War on Terror & Interventions",
     lambda r: 1998 <= r["first_year"] <= 2004),
    ("Prohibition is a sharp 1920s spike", None, None),
    ("Slavery -> civil rights classifies as a RENAME",
     "Slavery, Emancipation & Sectionalism",
     lambda r: r["rename_class"] == "rename"),
]


def run_ground_truth(table: pd.DataFrame) -> list[dict]:
    """Evaluate the pre-registered ground-truth checks on the raw level-2 rows.

    Args:
        table: The assembled lifecycle table.

    Returns:
        One dict per check with `check`, `passed`, and an `observed` string. A
        check whose topic does not exist in the taxonomy is reported as
        `not_testable` rather than quietly skipped.
    """
    raw = headline_rows(table)
    results = []
    for label, topic, predicate in GROUND_TRUTH:
        if topic is None:
            results.append({"check": label, "passed": "not_testable",
                            "observed": "no such topic in taxonomy_v1"})
            continue
        rows = raw[raw["topic"] == topic]
        if rows.empty:
            results.append({"check": label, "passed": "not_testable",
                            "observed": f"topic {topic!r} absent"})
            continue
        row = rows.iloc[0]
        results.append({
            "check": label,
            "passed": bool(predicate(row)),
            "observed": (
                f"first={row['first_year']:.0f} peak={row['peak_year']:.0f} "
                f"({row['peak_era']}, share={row['peak_share']:.3f}) "
                f"last={row['last_year']:.0f} class={row['lifecycle_class']} "
                f"rename={row['rename_class']}"
            ),
        })
    return results


def anachronism_report(
    assignments: pd.DataFrame, table: pd.DataFrame
) -> pd.DataFrame:
    """Compare each topic's raw first appearance to its first SUBSTANTIVE year.

    A large gap is the signature of stray mislabels — the failure mode the
    substantive threshold exists to absorb. Returned sorted by gap, descending.
    """
    raw = headline_rows(table)
    first_any = assignments.groupby("topic")["year"].min()
    out = raw[["topic", "first_year", "peak_year", "last_year", "n_paragraphs"]].copy()
    out["first_any_year"] = out["topic"].map(first_any)
    out["gap_years"] = out["first_year"] - out["first_any_year"]
    return out.sort_values("gap_years", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def print_checks(table: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Print every falsification check a rerun is supposed to surface.

    Deliberately includes the anachronism report: the task's verification
    strategy lists "any topic whose birth precedes its real-world existence" as
    a check, and the findings note cites it, so a rerun must print it rather than
    leaving it to be recomputed by hand.
    """
    anachronism_preview_rows = 10

    print("\nGROUND-TRUTH CHECKS")
    for result in run_ground_truth(table):
        print(f"  [{str(result['passed']).upper():>12}] {result['check']}\n"
              f"                 {result['observed']}")

    raw = headline_rows(table)
    print("\nLIFECYCLE CLASSES (level2, raw):")
    print(raw["lifecycle_class"].value_counts().to_string())
    print("\nUNDER-POWERED (level2) by treatment:")
    print(
        table[table["level"] == "level2"]
        .groupby("treatment")["under_powered"].sum().to_string()
    )

    gaps = anachronism_report(assignments, table)
    print(
        f"\nANACHRONISM CHECK (top {anachronism_preview_rows} of {len(gaps)} topics "
        "by the gap between the raw first appearance and the first SUBSTANTIVE "
        "year; a large gap is the signature of stray mislabels):"
    )
    print(gaps.head(anachronism_preview_rows).to_string(index=False))


def _checked_out_path(out: Path) -> Path:
    """Refuse a `--out` that would write outside `data/attention/`.

    The operator is not a trust boundary here — they could write the file
    directly — so this is a fat-finger guard, not a security control. It exists
    because `data/llm_annotations/` holds frozen, provenance-stamped output of a
    paid run that cannot be regenerated for free, and `--out` is this module's
    only route to writing anywhere outside `data/attention/`.

    Checked on the write path rather than at argument-parse time so that a
    read-only `--quotes` invocation is never rejected for a `--out` it ignores,
    and resolved against the module-level `ATTENTION_DIR` at call time so tests
    can redirect it.

    Args:
        out: The path supplied to `--out`.

    Returns:
        The resolved path, when it is inside `ATTENTION_DIR`.

    Raises:
        SystemExit: If the resolved path escapes `ATTENTION_DIR`.
    """
    resolved = Path(out).resolve()
    if not resolved.is_relative_to(ATTENTION_DIR.resolve()):
        raise SystemExit(
            f"--out must stay inside {ATTENTION_DIR}; refusing to write "
            f"{resolved}. data/llm_annotations/ is frozen paid-run output and "
            "is read-only."
        )
    return resolved


def main(argv: list[str] | None = None) -> None:
    """Build `data/attention/topic_lifecycles.parquet` and print the checks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--out", type=Path, default=LIFECYCLES_PATH)
    parser.add_argument("--quotes", type=str, default=None,
                        help="print exemplar paragraphs for one topic instead of building")
    parser.add_argument("--quote-years", type=int, nargs=2, default=(1789, 2026))
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy()
    paragraphs, assignments = load_inputs(taxonomy)

    if args.quotes:
        for quote in exemplar_quotes(args.quotes, assignments, *args.quote_years, n=4):
            print(f"[{quote['year']}] {quote['doc_name']} #{quote['para_idx']}")
            print(quote["text"], "\n")
        return

    crosswalk = json.loads(CROSSWALK_PATH.read_text())
    table, _, _ = build_lifecycle_table(
        paragraphs, assignments, taxonomy, crosswalk, args.draws, args.seed
    )
    path = write_lifecycles(table, _checked_out_path(args.out))
    print(f"wrote {path} ({len(table)} rows)")
    print_checks(table, assignments)


if __name__ == "__main__":
    main()
