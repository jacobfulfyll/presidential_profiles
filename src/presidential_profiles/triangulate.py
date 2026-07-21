"""Method triangulation: LLM vs CorEx vs embedding clusters over one corpus.

Three labelers now cover the same 36,229 paragraphs:

  * **LLM** on the discovered 50-topic taxonomy — semantic, robust to 240 years
    of vocabulary drift, multi-label.
  * **CorEx** on the frozen legacy 15 — lexical, keyword-anchored, deliberately
    dumb, and *frozen on purpose* so it stays a bias-independent control.
  * **Embedding clusters** at k=15 — unsupervised, no supplied vocabulary.

The point is not to crown a winner. It is that their errors are **uncorrelated**:
CorEx goes blind when the words change ("the peculiar institution" is invisible
to a model anchored on "slavery"); the LLM goes wrong when its judgment is
biased. Agreement between a lexical and a semantic labeler is therefore much
stronger evidence than either alone, and *disagreement is itself a measurement*.

The single most valuable disagreement this module can detect:
**a topic CorEx says died but the LLM says is alive has been RENAMED, not
abandoned.** Distinguishing vocabulary drift from genuine loss of attention is
the whole reason the worse model is kept around.

PROJECTION DIRECTION — the choice that moves the numbers most
    `crosswalk_v1.json` runs legacy -> level-2, one-to-many, and is NOT a
    partition: level-2 topics are reused across legacy issues (`Immigration &
    Border Security` sits under both `Immigration` and `Security & peace`).
    There is no reverse index, and inverting it is many-to-many. We use an
    explicit **union / any-activation** projection: a paragraph counts for legacy
    issue L if ANY of its LLM topics is in L's crosswalk row. One paragraph can
    therefore activate several legacy issues at once, exactly as CorEx's
    multi-label output can. This is stated in the report because a different
    choice (e.g. argmax to a single issue) would produce different agreement.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .corpus import DATA_DIR, load
from .embed_topics import CLUSTERS_META_PATH, CLUSTERS_PATH
from .fetch import PARAGRAPHS_PATH
from .issues import ISSUES_META_PATH, PARA_LABELS_PATH
from .llm_annotations import ANNOTATIONS_DIR
from .taxonomy import (
    CROSSWALK_PATH,
    SECURITY_PEACE,
    TAXONOMY_PATH,
    _norm_name,
    _require_full_merge,
)
from .trends import ERAS

PARAGRAPH_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "paragraph_annotations.parquet"
COMPOSITIONS_PATH = DATA_DIR / "method_compositions.parquet"
AGREEMENT_PATH = DATA_DIR / "method_agreement.parquet"

# THE REPORTING AXIS — `trends.ERAS`, not `taxonomy.ERA_SPAN`.
#
# The two era grids exist for different jobs and the split is functional:
#   * `taxonomy.ERA_SPAN = 30` sits under `# Sampling / gate parameters.` Its
#     job is SAMPLING STRATIFICATION — equal-width and deliberately
#     atheoretical, because per-era LLM taxonomy proposals must not inherit a
#     periodization the taxonomy was supposed to discover independently. That
#     regularity is the anachronism guard, not a claim about history.
#   * `trends.ERAS` is the REPORTING axis, with boundaries on real events.
#
# A published per-issue x per-era agreement table that `era-atlas` and
# `convergence-analysis` join on is a reporting axis, so it takes the reporting
# grid. There is also a substantive cost to the equal-width bands here: they cut
# at 1860-1889 (fusing the Civil War with the Gilded Age) and 1920-1949 (fusing
# the Depression with WWII). `combat.py` hit exactly that and documented it.
#
# Derived from `trends` directly, in the same shape `combat.py` uses. NOT
# imported from `combat` — that would point the dependency the wrong way — and
# NOT redefined locally, because two competing era axes would make every
# cross-task number unjoinable.
ERA_ORDER = [label for label, _, _ in ERAS]
ERA_BOUNDS = {label: (lo, hi) for label, lo, hi in ERAS}

# CorEx's free topic 5 is the crosswalk's "Security & peace" (see
# taxonomy.SECURITY_PEACE). Renaming it here is what lets the CorEx arm be
# compared against all 16 crosswalk keys rather than only the 15 legacy ones.
SECURITY_PEACE_COLUMN = "Discovered 5"

# The canonical level-2 topic count in frozen taxonomy_v1. Asserted after
# normalization: paragraph_annotations.topics holds 58 distinct raw strings, 8 of
# which are title-case variants (111 rows). An exact-match join would silently
# drop them — and they are concentrated in the war/foreign-policy topics the
# analysis cares about most, so the drop would bias agreement DOWNWARD exactly
# where the headline prediction is being tested.
N_LEVEL2_TOPICS = 50

# Two DIFFERENT populations, deliberately named apart — they were once the same
# bare `100` literal in two places, which read as one rule and is not.
# `MIN_ERA_PARAGRAPHS` gates an era's total paragraph count (is this era worth
# scoring at all?). `MIN_ISSUE_SUPPORT` gates the positives one ISSUE has within
# an era (is this cell's agreement number meaningful?). A 6,578-paragraph era
# easily clears the first and can still fail the second for a given issue.
MIN_ERA_PARAGRAPHS = 100
MIN_ISSUE_SUPPORT = 25


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------

def normalize_topic_lists(topics: pd.Series) -> pd.Series:
    """Per-paragraph topic lists -> frozensets of `_norm_name`-normalized names.

    Reuses `taxonomy._norm_name` rather than defining a second normalizer, so
    this collapses names the same way `validate_structure` does.
    """
    return topics.map(lambda lst: frozenset(_norm_name(t) for t in lst))


def _assert_canonical_topics(topic_sets: pd.Series, expected=N_LEVEL2_TOPICS) -> None:
    observed = {t for s in topic_sets for t in s}
    if len(observed) != expected:
        raise ValueError(
            f"normalized topic names collapsed to {len(observed)} distinct "
            f"values, expected {expected}. Either a new label leaked into the "
            f"annotations or _norm_name stopped collapsing the known title-case "
            f"variants; a join on these would silently drop rows."
        )


def load_crosswalk(path=CROSSWALK_PATH) -> dict[str, set[str]]:
    """legacy issue -> set of normalized level-2 topic names.

    Keys are the 16 crosswalk issues (15 legacy + "Security & peace").
    """
    cw = json.loads(path.read_text())
    return {
        m["legacy_issue"]: {_norm_name(t) for t in m["level2_topics"]}
        for m in cw["mappings"]
    }


def project_to_legacy(
    topic_sets: pd.Series, crosswalk: dict[str, set[str]]
) -> pd.DataFrame:
    """Union projection: LLM level-2 topics -> boolean columns per legacy issue.

    A paragraph is positive for issue L iff at least one of its LLM topics falls
    in L's crosswalk row. Because the crosswalk is not a partition, a paragraph
    can be positive for several issues simultaneously — which is correct, and
    matches CorEx's own multi-label behaviour.

    Note 7 of the 50 level-2 topics (personal narrative, presidential humility,
    partisan combat, procedural housekeeping, ...) appear in NO crosswalk row.
    They are non-policy topics with no legacy counterpart, so a paragraph labeled
    only with those projects to zero legacy issues — a real and expected outcome,
    not a lookup failure.
    """
    return pd.DataFrame(
        {issue: topic_sets.map(lambda s, t=topics: bool(s & t))
         for issue, topics in crosswalk.items()},
        index=topic_sets.index,
    )


def assign_eras(years: pd.Series) -> pd.Series:
    """Year -> named era label from `trends.ERAS`. Every year, exactly one band.

    THE NaN TRAP — why this asserts instead of using `pd.cut` and moving on.
        On this axis `era` is a **string**, and NaN already carries meaning
        downstream: the whole-corpus rows of `method_agreement.parquet` are
        identified by `era` being NaN. A year falling outside every band would
        ALSO come out NaN and be silently conflated with the overall row — a
        corpus-wide statistic read as an era statistic, with nothing on the
        surface to distinguish them.

        The bands are contiguous 1789->2026 with no gaps, so this cannot fire on
        today's corpus. It is asserted rather than assumed precisely because the
        failure mode is silent and the consequence is a wrong published number.
    """
    labels = pd.Series(None, index=years.index, dtype=object)
    for label, lo, hi in ERAS:
        hit = years.between(lo, hi)
        clash = hit & labels.notna()
        if clash.any():
            raise ValueError(
                f"trends.ERAS bands overlap: {label!r} also claims year(s) "
                f"{sorted({int(y) for y in years[clash]})}, which are already "
                f"assigned. Era assignment must be a partition of the years."
            )
        labels[hit] = label
    if labels.isna().any():
        orphans = sorted({int(y) for y in years[labels.isna()]})
        raise ValueError(
            f"years outside every trends.ERAS band: {orphans}. They would be "
            f"labelled NaN — which this pipeline already uses to mark the "
            f"whole-corpus rows — so the two would be indistinguishable. "
            f"Extend trends.ERAS rather than letting years fall through."
        )
    return labels


def eras_in_order(values) -> list[str]:
    """The eras present in `values`, in `trends.ERAS` DECLARATION order.

    Never `sorted()`. These labels are strings, and lexicographic order reads
    "Civil War & Reconstruction" < "Expansion" < "The founding" — a
    chronological axis rendered alphabetically, which would silently scramble
    every era trend in the report. The old numeric grid sorted correctly by
    accident; named eras do not, so ordering is explicit everywhere.
    """
    present = set(pd.Series(list(values), dtype=object).dropna())
    unknown = present - set(ERA_ORDER)
    if unknown:
        raise ValueError(
            f"era labels absent from trends.ERAS: {sorted(unknown)}. The "
            f"reporting axis is trends.ERAS; a second periodization would make "
            f"every cross-task number unjoinable."
        )
    return [e for e in ERA_ORDER if e in present]


def load_arms() -> pd.DataFrame:
    """One frame keyed (doc_name, para_idx) carrying all three labelers.

    Every merge is keyed and checked twice: `validate="one_to_one"` for duplicate
    keys, and `_require_full_merge` for diverging key SETS — an inner join
    silently drops non-matching rows, which here would mean computing agreement
    on an unannounced subset of the corpus.
    """
    paras = pd.read_parquet(PARAGRAPHS_PATH)[["doc_name", "para_idx"]]
    issues = pd.read_parquet(PARA_LABELS_PATH)
    clusters = pd.read_parquet(CLUSTERS_PATH)
    ann = pd.read_parquet(PARAGRAPH_ANNOTATIONS_PATH)[
        ["doc_name", "para_idx", "topics"]
    ]

    df = paras.merge(issues, on=["doc_name", "para_idx"], validate="one_to_one")
    _require_full_merge(len(df), {"paras": len(paras), "issues": len(issues)},
                        "paragraphs x paragraph_issues")
    df = df.merge(clusters, on=["doc_name", "para_idx"], validate="one_to_one")
    _require_full_merge(len(df), {"running": len(paras), "clusters": len(clusters)},
                        "(paragraphs+issues) x paragraph_clusters")
    df = df.merge(ann, on=["doc_name", "para_idx"], validate="one_to_one")
    _require_full_merge(len(df), {"running": len(paras), "annotations": len(ann)},
                        "(paragraphs+issues+clusters) x paragraph_annotations")

    df["topic_set"] = normalize_topic_lists(df["topics"])
    _assert_canonical_topics(df["topic_set"])
    df["llm_unlabeled"] = df["topic_set"].map(len) == 0
    df["era"] = assign_eras(df["year"])
    return df


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def _pair_metrics(llm: np.ndarray, corex: np.ndarray) -> dict:
    """Jaccard + Cohen's kappa for one issue over one slice of paragraphs.

    Jaccard is over the two PARAGRAPH SETS the labelers assign to this issue
    (|both| / |either|) — it ignores the vast agreed-negative mass, which for a
    rare issue would otherwise make any two labelers look near-identical. Kappa
    keeps the negatives but corrects for chance agreement. They answer different
    questions, so both are reported.
    """
    n_llm, n_corex = int(llm.sum()), int(corex.sum())
    n_both = int((llm & corex).sum())
    n_either = int((llm | corex).sum())
    jaccard = n_both / n_either if n_either else np.nan

    # Kappa is undefined when one labeler is constant over the slice (its
    # expected-agreement term collapses); report NaN rather than a fake 0.
    if len(llm) == 0 or llm.all() or corex.all() or not llm.any() or not corex.any():
        kappa = np.nan
    else:
        kappa = float(cohen_kappa_score(llm, corex))

    return {"n": len(llm), "n_llm": n_llm, "n_corex": n_corex,
            "n_both": n_both, "jaccard": jaccard, "kappa": kappa}


def corex_column(issue: str) -> str:
    """The `paragraph_issues.parquet` column backing a crosswalk issue."""
    return SECURITY_PEACE_COLUMN if issue == SECURITY_PEACE else issue


def _require_rectangular(out: pd.DataFrame, n_issues: int, n_eras: int) -> None:
    """The agreement table must be rectangular, and NaN must mean one thing only.

    `era is NaN` is this table's marker for the whole-corpus rows, so the NaN
    count must equal the issue count exactly — a year-derived NaN would be read
    as a corpus-wide statistic with nothing on the surface to distinguish it.
    The era-labelled rows must then account for every remaining cell,
    issue x kept era, with none quietly dropped.
    """
    n_overall = int(out["era"].isna().sum())
    n_era_rows = int(out["era"].notna().sum())
    if n_overall != n_issues or n_era_rows != n_issues * n_eras:
        raise ValueError(
            f"agreement table is not rectangular: {n_overall} whole-corpus rows "
            f"(expected {n_issues}) and {n_era_rows} era rows (expected "
            f"{n_issues} x {n_eras} = {n_issues * n_eras}). `era is NaN` must "
            f"identify exactly the whole-corpus rows, and no issue-era cell may "
            f"be silently dropped."
        )


def agreement_table(
    df: pd.DataFrame,
    llm_legacy: pd.DataFrame,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """LLM<->CorEx agreement per issue, overall and per era.

    `drop_unlabeled=True` (the primary analysis) removes the 404 paragraphs the
    LLM returned no topics for. An LLM abstention is NOT the same event as "the
    LLM judged this issue absent": counting it as an all-negative row would let
    it inflate kappa's agreed-negative mass for all 16 issues at once. The
    sensitivity run flips this flag; both are published.
    """
    mask = ~df["llm_unlabeled"] if drop_unlabeled else pd.Series(True, index=df.index)
    sub, llm_sub = df[mask], llm_legacy[mask]

    # THE NaN TRAP, on the INPUT. A NaN era would be dropped by `eras_in_order`,
    # so those paragraphs would vanish from every era row while still counting in
    # the whole-corpus row — an unannounced subset, reported as if complete.
    # `assign_eras` already forbids this upstream; checked again here because
    # `agreement_table` is callable directly and the failure is silent.
    if sub["era"].isna().any():
        raise ValueError(
            f"{int(sub['era'].isna().sum())} paragraphs carry era = NaN. They "
            f"would be counted in the whole-corpus rows but silently absent "
            f"from every era row. Era labels come from `assign_eras`, which "
            f"rejects unmappable years rather than emitting NaN."
        )

    # Chronological, from the ERAS declaration — `sorted()` here would emit the
    # era series in alphabetical order (see `eras_in_order`). The size filter
    # depends only on the era slice, never on the issue, so it is applied once
    # here rather than re-derived on every pass of the 16-issue loop.
    eras = [e for e in eras_in_order(sub["era"])
            if int((sub["era"] == e).sum()) >= MIN_ERA_PARAGRAPHS]

    rows = []
    for issue in llm_legacy.columns:
        col = corex_column(issue)
        for era in [None, *eras]:
            slc = sub if era is None else sub[sub["era"] == era]
            m = _pair_metrics(
                llm_sub.loc[slc.index, issue].to_numpy(dtype=bool),
                slc[col].to_numpy(dtype=bool),
            )
            rows.append({"issue": issue, "era": era, **m})
    out = pd.DataFrame(rows)
    _require_rectangular(out, len(llm_legacy.columns), len(eras))

    # Slice size is NOT support. An era can hold thousands of paragraphs while an
    # issue has near-zero positives in it — in "The founding" the LLM has zero
    # positives for several modern issues, because those topics did not exist
    # yet. Those cells produce a real-looking Jaccard of 0.000 that
    # is an artifact of anachronism, not of labeler disagreement, and averaging
    # them into an era mean silently drags early eras down. Flagged rather than
    # dropped so callers choose; `min_support` makes the choice one comparison.
    out["min_support"] = out[["n_llm", "n_corex"]].min(axis=1)
    out["low_support"] = out["min_support"] < MIN_ISSUE_SUPPORT
    out["empty_arm"] = (out["n_llm"] == 0) | (out["n_corex"] == 0)
    return out


# ---------------------------------------------------------------------------
# Rename vs death
# ---------------------------------------------------------------------------

def rename_vs_death(df: pd.DataFrame, llm_legacy: pd.DataFrame) -> pd.DataFrame:
    """Per issue x era: each labeler's attention, scaled to its own peak.

    Both arms are normalized to their OWN peak era, because the two methods have
    different base rates and only their *trajectories* are comparable. The
    vocabulary-drift signature is then simply: `corex_rel` collapses while
    `llm_rel` persists — CorEx's anchor words stopped appearing, the underlying
    concern did not.
    """
    sub = df[~df["llm_unlabeled"]]
    llm_sub = llm_legacy.loc[sub.index]

    # Grouped once, outside the 16-issue loop: these are whole-frame aggregations
    # and recomputing them per issue was the dominant cost here.
    by_era = sub.groupby("era")
    n = by_era.size()
    corex_all = by_era[[corex_column(i) for i in llm_legacy.columns]].mean()
    llm_all = llm_sub.groupby(sub["era"]).mean()

    # Eras too thin for a stable share are dropped BEFORE the peak is taken, so a
    # small-n era can never become the denominator every other era is scaled
    # against. Inert today (the thinnest of the 9 named eras clears the bar) but a
    # live trap the moment the corpus or the era grid changes.
    #
    # `eras_in_order` is applied AFTER the size filter, never instead of it —
    # it only fixes the order. Collapsing the two into one expression is how the
    # filter gets lost.
    keep = eras_in_order(n[n >= MIN_ERA_PARAGRAPHS].index)

    rows = []
    for issue in llm_legacy.columns:
        col = corex_column(issue)
        corex = corex_all.loc[keep, col]
        llm = llm_all.loc[keep, issue]
        c_peak, l_peak = corex.max(), llm.max()
        for era in corex.index:
            rows.append({
                "issue": issue, "era": str(era), "n_paragraphs": int(n[era]),
                "corex_share": float(corex[era]), "llm_share": float(llm[era]),
                "corex_rel": float(corex[era] / c_peak) if c_peak else np.nan,
                "llm_rel": float(llm[era] / l_peak) if l_peak else np.nan,
            })
    out = pd.DataFrame(rows)
    out["drift_gap"] = out["llm_rel"] - out["corex_rel"]
    return out


# Thresholds for FLAGGING rename candidates in the summary table. These select
# what to look at; they do not create the effect, and the full unfiltered
# trajectory table is published alongside so a reader can pick their own.
COREX_COLLAPSE = 0.25   # CorEx down to a quarter of its own peak era
LLM_PERSIST = 0.50      # LLM still at half of its own peak era


def rename_candidates(drift: pd.DataFrame) -> pd.DataFrame:
    """Issue-eras where CorEx attention collapsed but LLM attention held."""
    hit = drift[(drift["corex_rel"] < COREX_COLLAPSE)
                & (drift["llm_rel"] > LLM_PERSIST)]
    return hit.sort_values("drift_gap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-speech compositions
# ---------------------------------------------------------------------------

def build_compositions(df: pd.DataFrame, llm_legacy: pd.DataFrame) -> pd.DataFrame:
    """Per-speech topic-share vectors under each surviving method.

    Long/tidy format (`doc_name, method, topic, share, n_paragraphs`) rather than
    one wide frame: the three methods have different and non-comparable
    vocabularies (50 level-2 topics, 16 crosswalk issues, 15 clusters), so a wide
    frame would be 81 mostly-meaningless columns. Long format also gives
    downstream analyses the promised one-line method switch —
    `comps[comps.method == "llm_taxonomy"]`.

    Every method uses the SAME denominator: all paragraphs in the speech,
    including the 404 the LLM left unlabeled. Shares are therefore directly
    comparable across methods, and the LLM arm is not silently advantaged by a
    smaller denominator.
    """
    n_paras = df.groupby("doc_name").size().rename("n_paragraphs")
    frames = []

    # --- LLM on the discovered taxonomy (50 level-2 topics) ---
    canonical = {_norm_name(t["name"]): t["name"]
                 for t in json.loads(TAXONOMY_PATH.read_text())["level2"]}
    exploded = (
        df[["doc_name", "topic_set"]]
        .explode("topic_set").dropna(subset=["topic_set"])
    )
    llm_counts = (
        exploded.groupby(["doc_name", "topic_set"]).size().rename("hits").reset_index()
    )
    llm_counts["topic"] = llm_counts["topic_set"].map(canonical)
    if llm_counts["topic"].isna().any():
        missing = sorted(set(llm_counts.loc[llm_counts["topic"].isna(), "topic_set"]))
        raise ValueError(f"topics absent from frozen taxonomy_v1: {missing}")
    # Reindexed on BOTH axes: columns so all 50 topics are present even when a
    # topic never fires, and rows so a speech whose every paragraph went
    # unlabeled still appears with all-zero shares. The row reindex is what makes
    # the "same denominator across methods" guarantee structural rather than a
    # property of this corpus — without it such a speech would be missing from
    # the LLM arm while appearing in the other two.
    llm_wide = (
        llm_counts.pivot(index="doc_name", columns="topic", values="hits")
        .reindex(index=n_paras.index, columns=sorted(canonical.values()))
        .fillna(0.0)
    )
    frames.append(("llm_taxonomy", llm_wide))

    # --- CorEx on the legacy 15 + Security & peace, via the crosswalk keys ---
    corex_cols = [corex_column(i) for i in llm_legacy.columns]
    corex_wide = df.groupby("doc_name")[corex_cols].sum()
    corex_wide.columns = list(llm_legacy.columns)
    frames.append(("corex_legacy", corex_wide))

    # --- Embedding clusters at k=15 (dimension-matched to the legacy 15) ---
    cluster_names = df["cluster_k15"].map(lambda c: f"cluster_{c:02d}")
    embed_wide = pd.get_dummies(cluster_names).groupby(df["doc_name"]).sum()
    frames.append(("embed_k15", embed_wide))

    tidy_frames = []
    for method, wide in frames:
        shares = wide.div(n_paras.reindex(wide.index), axis=0)
        tidy = (
            shares.stack().rename("share").reset_index()
            .rename(columns={"level_1": "topic"})
        )
        tidy["method"] = method
        tidy_frames.append(tidy)

    comps = pd.concat(tidy_frames, ignore_index=True)
    comps = comps.merge(n_paras.reset_index(), on="doc_name", validate="many_to_one")
    return comps[["doc_name", "method", "topic", "share", "n_paragraphs"]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def agreement_drivers(
    agreement: pd.DataFrame,
    sensitivity: pd.DataFrame,
    coherence: dict[str, dict] | None = None,
    crosswalk: dict[str, set[str]] | None = None,
) -> dict:
    """The report's headline statistics, as code rather than prose.

    Returns the per-issue driver frame, the Spearman table behind §0, and the
    §6 unlabeled-paragraph sensitivity summary.

    THE CEILING CONTROL — why `fill` exists
        Jaccard is bounded above by the very quantity we correlate it against.
        With a = n_llm, b = n_corex: |A n B| <= min(a,b) and |A u B| >= max(a,b),
        so

            J <= min(a,b)/max(a,b) = exp(-|log(a/b)|) = exp(-D)

        deterministically, for every issue, before any data is seen. A negative
        rank association between D and J is therefore partly GUARANTEED BY
        ARITHMETIC rather than discovered. `fill` = J / ceiling is the share of
        the attainable maximum actually realised; correlating `fill` against D
        removes the tautological component and is the honest test of whether
        base-rate divergence explains anything beyond its own algebra.

        Kappa is NOT an independent corroboration of the same claim: it is
        separately depressed by marginal imbalance (the well-known "kappa
        paradox" / prevalence-and-bias effect), which is itself a function of D.
    """
    overall = agreement[agreement["era"].isna()].set_index("issue")
    legacy = [i for i in overall.index if i != SECURITY_PEACE]
    drivers = overall.loc[legacy, ["n_llm", "n_corex", "jaccard", "kappa"]].copy()

    # A zero on either arm makes the log ratio infinite and the ceiling zero,
    # which would propagate NaN through every Spearman below and silently void
    # the whole table. No legacy issue has an empty arm corpus-wide today, so
    # this raises rather than imputing: a zero here means the caller passed era
    # rows or a filtered frame, and the fix belongs upstream, not in a
    # fabricated value.
    if (drivers[["n_llm", "n_corex"]] == 0).any().any():
        empty = drivers.index[
            (drivers[["n_llm", "n_corex"]] == 0).any(axis=1)].tolist()
        raise ValueError(
            f"issues with an empty arm cannot be scored on the log base-rate "
            f"ratio: {empty}. agreement_drivers expects the corpus-wide rows "
            f"(era is NaN), where every legacy issue has positives on both arms."
        )

    drivers["divergence"] = np.abs(np.log(drivers["n_llm"] / drivers["n_corex"]))
    drivers["ceiling"] = (np.minimum(drivers["n_llm"], drivers["n_corex"])
                          / np.maximum(drivers["n_llm"], drivers["n_corex"]))
    drivers["fill"] = drivers["jaccard"] / drivers["ceiling"]
    drivers["bound_holds"] = drivers["jaccard"] <= drivers["ceiling"] + 1e-12
    if coherence is not None:
        drivers["npmi"] = [coherence[i]["npmi"] for i in drivers.index]
    if crosswalk is not None:
        drivers["fanout"] = [len(crosswalk[i]) for i in drivers.index]

    def rho(x, y):
        r = spearmanr(drivers[x], drivers[y])
        return {"rho": float(r.statistic), "p": float(r.pvalue)}

    stats = {
        "divergence_vs_jaccard": rho("divergence", "jaccard"),
        "divergence_vs_kappa": rho("divergence", "kappa"),
        # The controlled test. If base-rate divergence carries information beyond
        # the algebraic ceiling, this stays significant. It does not.
        "divergence_vs_fill_CONTROLLED": rho("divergence", "fill"),
    }
    # Every predictor gets the ceiling control, not just base-rate divergence.
    # Fan-out inflates n_llm, so it acts THROUGH the same bounded channel and an
    # uncontrolled fan-out correlation is subject to the identical artifact.
    if coherence is not None:
        stats["npmi_vs_jaccard"] = rho("npmi", "jaccard")
        stats["npmi_vs_kappa"] = rho("npmi", "kappa")
        stats["npmi_vs_fill_CONTROLLED"] = rho("npmi", "fill")
    if crosswalk is not None:
        stats["fanout_vs_jaccard"] = rho("fanout", "jaccard")
        stats["fanout_vs_kappa"] = rho("fanout", "kappa")
        stats["fanout_vs_fill_CONTROLLED"] = rho("fanout", "fill")
        # Direct evidence for the SUBSTANTIVE fan-out claim (wide crosswalk rows
        # inflate LLM breadth), which survives even though the agreement
        # correlation does not.
        stats["fanout_vs_divergence"] = rho("fanout", "divergence")

    # Counted from what was actually run rather than hardcoded, so the correction
    # stays correct when a predictor is absent.
    uncontrolled = [k for k in stats
                    if k.endswith(("_vs_jaccard", "_vs_kappa"))]
    stats["n_tests"] = len(uncontrolled)
    stats["bonferroni_alpha"] = 0.05 / len(uncontrolled)

    sens_overall = sensitivity[sensitivity["era"].isna()].set_index("issue")
    d_jac = (sens_overall["jaccard"] - overall["jaccard"]).abs()
    d_kap = (sens_overall["kappa"] - overall["kappa"]).abs()
    # The magnitude and the issue name must come from the SAME metric. Taking
    # the max over both metrics but the argmax over Jaccard alone would report
    # a kappa-sized delta under a Jaccard-chosen issue name.
    worst = d_jac if d_jac.max() >= d_kap.max() else d_kap
    sens = {
        "mean_abs_delta_jaccard": float(d_jac.mean()),
        "mean_abs_delta_kappa": float(d_kap.mean()),
        "max_abs_delta": float(worst.max()),
        "max_abs_delta_issue": str(worst.idxmax()),
    }
    return {"drivers": drivers, "spearman": stats, "sensitivity": sens}


def run() -> dict:
    """Compute every artifact; return them for the report writer."""
    df = load_arms()
    crosswalk = load_crosswalk()
    llm_legacy = project_to_legacy(df["topic_set"], crosswalk)

    primary = agreement_table(df, llm_legacy, drop_unlabeled=True)
    sensitivity = agreement_table(df, llm_legacy, drop_unlabeled=False)
    drift = rename_vs_death(df, llm_legacy)
    comps = build_compositions(df, llm_legacy)
    coherence = json.loads(ISSUES_META_PATH.read_text()).get("coherence")
    drivers = agreement_drivers(primary, sensitivity, coherence, crosswalk)
    cluster_meta = json.loads(CLUSTERS_META_PATH.read_text())
    speeches = load()
    candidates = rename_candidates(drift)

    # Persisted LAST, after everything that can raise has already succeeded. A
    # run that died midway used to leave both parquets on disk with nothing to
    # distinguish them from a complete run's — a silent half-artifact.
    # AC2's deliverable is per issue PER ERA, so the era rows are persisted, not
    # just the aggregate trend that made it into the report.
    primary.to_parquet(AGREEMENT_PATH, index=False)
    comps.to_parquet(COMPOSITIONS_PATH, index=False)

    return {
        "drivers": drivers,
        "df": df,
        "llm_legacy": llm_legacy,
        "agreement": primary,
        "agreement_sensitivity": sensitivity,
        "drift": drift,
        "rename_candidates": candidates,
        "compositions": comps,
        "cluster_meta": cluster_meta,
        "speeches": speeches,
    }
