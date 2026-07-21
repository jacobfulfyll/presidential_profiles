"""Breadth, depth and register of the formal presidential record.

Three quantities measured by the adversarial review of the convergence design
(2026-07-13) all pointed one way — issue entropy rising 9.44 -> 11.23 effective
issues, labels per paragraph falling 1.49 -> 0.83, and the share of paragraphs
touching no policy issue nearly doubling. Read together they say: *presidents now
touch more subjects, say less about each, and spend a growing share of their most
formal speech not on policy at all.*

This module exists to give that headline its fairest chance to die. Every trend
is computed under controls that could plausibly fake it:

**Taxonomy fit.** Entropy is computed on the legacy 15 anchored CorEx issues AND
on the corpus-native `taxonomy_v1` (50 level-2 topics, 17 level-1 domains). A
taxonomy built from modern anchor vocabulary fits modern speech better, which on
its own produces a rising-breadth artifact. If the rise lives only on the legacy
15, "broader agendas" was never about presidents.

**Genre mix.** The 19th-century annual message was a departmental catalog —
Treasury, War, Navy, Post Office, Interior in sequence — which forced every
president to touch every issue. Annual messages fall from 39-44% of early
speeches to 13-16% of modern ones while `public_remarks_or_address` rises from
0-7% to 51-61%. So every trend is computed three ways: `raw`, `sotu_only` (the
one genre spanning all 240 years), and `genre_standardized` (each era reweighted
to a common mix of the four genres present in every era). A trend that survives
inside the annual message alone is about presidents; one that exists only across
the genre mix means the *form* of presidential speech changed — still a finding,
a different one, and the report must say which.

**Paragraph length.** Not in the original plan, and a first-order threat to the
"shallower" half of the headline: mean paragraph length falls from 135.9 words
(1830s era) to 97.7 (2010s), and from 134.0 to 92.9 inside annual messages
alone. Labels *per paragraph* must fall by roughly that much for purely
typographic reasons. So each density measure is published twice — per paragraph
(comparable to the original observation) and per 1,000 words (length-invariant).

**Estimator bias.** Plug-in Shannon entropy is biased downward at small sample
sizes, and the thin early eras are exactly where the headline needs low values.
`effective_topics` therefore carries the Miller-Madow correction by default; the
uncorrected plug-in figure ships alongside it under `effective_topics_plugin`.

**Uncertainty.** The within-speech ICC of paragraph labels runs from 0.103 to
0.294 on the shipped paragraph frame (`within_speech_icc`), so every CI is a
speech-clustered bootstrap: speeches are the resampling unit, paragraphs ride
along with their speech. A paragraph-level bootstrap would understate
uncertainty by pretending 36,229 independent observations exist where there are
1,057.

**Multiplicity.** `build_trends` emits 444 hypothesis tests — 37 measure x
taxonomy columns x 3 genre treatments x 4 trend statistics, so the same quantity
under 3 arms counts as 3 of them. Nothing here corrects for that, and
nothing here should: the p-values are per-arm interval inversions and the note
that consumes them is exploratory. The correction is the *reader's* to apply,
which is why the note publishes the exposure. Note the resolution limit while
doing it: a bootstrap of B replicates cannot report a p below 1/B, so at
B = 2,000 no test can be shown to clear a Bonferroni threshold of 0.05/444
(= 1.13e-4) at all — that would need B >= ~9,000.

Two operationalizations of "non-policy share" are reported, never merged into one
series, because the two labelers disagree about what an unlabeled paragraph
means. CorEx leaves 28.7% of paragraphs with zero anchored issues; the LLM leaves
only 1.1% with zero topics because it almost always finds *something*, often a
non-policy topic. So on the legacy taxonomy the measure is the zero-anchored-issue
rate, and on `taxonomy_v1` it is the share of paragraphs whose assigned topics are
*all* non-policy-parented, with the 1.1% zero-topic residue published separately
as `zero_topic_share`.

This module makes no API calls: every input is already on disk, and
`taxonomy_v1` / the annotation parquets are frozen provenance-stamped artifacts
that must never be regenerated here.

Load and compute are kept apart — `load_inputs` is the only function that touches
the filesystem, everything downstream is DataFrames in, DataFrames out.

Run as:
    PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.register
(the repo venv is x86_64 under Rosetta).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import DATA_DIR
from .indices import MARKERS_PATH
from .llm_annotations import ANNOTATIONS_DIR
from .taxonomy import ERA_SPAN, LEGACY_ISSUES, TAXONOMY_PATH

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
PARAGRAPH_ISSUES_PATH = DATA_DIR / "paragraph_issues.parquet"
SPEECHES_PATH = DATA_DIR / "speeches.parquet"
STATS_PATH = DATA_DIR / "speech_stats.parquet"
PARAGRAPH_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "paragraph_annotations.parquet"
SPEECH_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "speech_annotations.parquet"

REGISTER_DIR = DATA_DIR / "register"
TRENDS_PATH = REGISTER_DIR / "trends.parquet"

# --------------------------------------------------------------------------
# analysis parameters
# --------------------------------------------------------------------------

SOTU_TYPE = "state_of_the_union_or_annual_message"

# The genres present in ALL nine 30-year eras, and therefore the only ones a
# common reference mix can be built from. Pinned rather than derived so that a
# corpus change surfaces as a raised exception in `reference_genre_mix` instead
# of silently redefining what "genre-standardized" means between runs.
REFERENCE_GENRES = (
    "inaugural_address",
    "special_message_to_congress",
    SOTU_TYPE,
    "veto_or_signing_statement",
)

# An (era, genre) cell thinner than this is dropped from the standardized
# estimate and the remaining reference weights are renormalized. Three speeches
# is already a fragile within-cell mean; below it the reweighting amplifies pure
# noise. Which cells this drops is reported by `genre_cells`, because in the
# 1770 era it drops two of the four reference genres.
MIN_CELL_SPEECHES = 3

# Years thinner than this are not published at all, matching the masking idiom
# in `indices.yearly_raw_rates` / `site._stats_yearly` (both mask sparse years
# rather than let them spike a chart).
MIN_YEAR_PARAGRAPHS = 30

N_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = 20260721
CI_ALPHA = 0.05

GENRE_TREATMENTS = ("raw", "sotu_only", "genre_standardized")

# Era-level series exist for all three treatments; the year-level series omits
# `genre_standardized` because a single year almost never contains all four
# reference genres, so the reweighting would be defined by which genres happened
# to occur — the opposite of standardizing.
YEAR_TREATMENTS = ("raw", "sotu_only")

NO_TAXONOMY = "none"

# Style markers carried over from the existing per-speech tables, as per-10k-word
# rates. `mechanism` is the depth measure indices.py already names ("the
# vocabulary of actually governing"); the rest cover the values / combat /
# emotion register the headline claims is displacing policy.
STYLE_MARKERS = (
    "mechanism",
    "religiosity",
    "nostalgia",
    "future",
    "us_them",
    "opponents",
    "hype",
    "doom",
    "boosters",
    "hedges",
    "superlatives",
    "nrc_hope",
    "nrc_fear",
)

TREND_STATISTICS = (
    "spearman_vs_era",
    "delta_modern_minus_early",
    "delta_modern_minus_postbellum",
    "delta_last_minus_first",
)

# `delta_modern_minus_early` contrasts three-era blocks rather than the endpoints,
# because the 1770 era holds only 28 speeches / 355 paragraphs and an endpoint
# contrast would rest almost entirely on it.
EARLY_ERAS = (1770, 1800, 1830)
POSTBELLUM_ERAS = (1860, 1890, 1920)
MODERN_ERAS = (1950, 1980, 2010)

# `delta_modern_minus_early` is pre-declared; `delta_modern_minus_postbellum` --
# the second key below, and only that one -- is POST HOC, and is labelled as such
# (with a marginal bar) wherever the note reports it. It was added after the era
# series were computed, because every breadth measure turned out to rise steeply
# to the 1860 era and then sit flat for 150 years. A reader told only "breadth
# rose across the corpus" would infer a modern change that the series does not
# contain. It is kept because it *weakens* the headline: it asks whether anything
# moved after the Civil War era, which is the period the headline is actually
# about. (It does not weaken it everywhere -- on the legacy labeler's
# `non_policy_share` and on the whole register block it runs the other way, and
# the note reports that too.)
BLOCK_CONTRASTS = {
    "delta_modern_minus_early": (MODERN_ERAS, EARLY_ERAS),
    "delta_modern_minus_postbellum": (MODERN_ERAS, POSTBELLUM_ERAS),
}


# ==========================================================================
# taxonomy index
# ==========================================================================


@dataclass(frozen=True)
class TaxonomyIndex:
    """Lookup structures over `taxonomy_v1.json`, with case-insensitive label
    resolution.

    111 of the 52,855 label assignments the annotation run returned (0.21%) are
    case-variants of real level-2 names — `the War On Terror` for `the War on
    Terror`, and seven others differing only in the casing of a small word. All
    resolve case-insensitively and none is unresolvable, but an exact-match join
    would silently drop them, and 84 of the 111 land on one modern-era topic
    (`Iraq, Gulf Wars, the War on Terror & Interventions`). Dropping them would
    bias modern policy attention *downward*, which flatters the "more non-policy
    speech" half of the headline. So resolution is casefolded and
    `resolve_labels` raises on anything that still fails to resolve."""

    level2: tuple[str, ...]
    level1: tuple[str, ...]
    parent: Mapping[str, str]
    kind: Mapping[str, str]
    by_casefold: Mapping[str, str]

    @property
    def non_policy_level2(self) -> frozenset[str]:
        """Level-2 topics whose level-1 domain is a non-policy domain."""
        return frozenset(
            t for t in self.level2 if self.kind[self.parent[t]] == "non-policy"
        )

    def resolve_labels(self, labels: Iterable[str]) -> list[str]:
        """Canonical level-2 names for one paragraph's labels.

        Raises:
            KeyError: If a label does not resolve even after casefolding. This is
                deliberately fatal: a silently dropped label is a silently biased
                attention series.
        """
        out = []
        for label in labels:
            canonical = self.by_casefold.get(str(label).casefold())
            if canonical is None:
                raise KeyError(
                    f"label {label!r} does not resolve to a taxonomy_v1 level-2 "
                    "topic even case-insensitively; refusing to drop it silently"
                )
            out.append(canonical)
        return out


def build_taxonomy_index(taxonomy: dict) -> TaxonomyIndex:
    """Index a parsed `taxonomy_v1.json`.

    Args:
        taxonomy: Parsed taxonomy with `level1` (name/definition/kind) and
            `level2` (name/definition/level1/...) entries.

    Returns:
        A `TaxonomyIndex` over its level-1 domains and level-2 topics.

    Raises:
        ValueError: If two level-2 names collide when casefolded (which would
            make case-insensitive resolution ambiguous), or if a level-2 topic
            names a parent domain that does not exist.
    """
    level1 = tuple(d["name"] for d in taxonomy["level1"])
    kind = {d["name"]: d["kind"] for d in taxonomy["level1"]}
    level2 = tuple(t["name"] for t in taxonomy["level2"])
    parent = {t["name"]: t["level1"] for t in taxonomy["level2"]}

    orphans = sorted({p for p in parent.values() if p not in kind})
    if orphans:
        raise ValueError(f"level-2 topics name unknown level-1 domains: {orphans}")

    by_casefold: dict[str, str] = {}
    for name in level2:
        folded = name.casefold()
        if folded in by_casefold:
            raise ValueError(
                f"level-2 names {by_casefold[folded]!r} and {name!r} collide when "
                "casefolded; case-insensitive label resolution would be ambiguous"
            )
        by_casefold[folded] = name

    return TaxonomyIndex(
        level2=level2, level1=level1, parent=parent, kind=kind, by_casefold=by_casefold
    )


# ==========================================================================
# input loading  (the only functions that touch the filesystem)
# ==========================================================================


@dataclass(frozen=True)
class RegisterInputs:
    """The six on-disk tables plus the frozen taxonomy this module reads."""

    paragraphs: pd.DataFrame
    issues: pd.DataFrame
    annotations: pd.DataFrame
    speeches: pd.DataFrame
    speech_annotations: pd.DataFrame
    stats: pd.DataFrame
    markers: pd.DataFrame
    taxonomy: TaxonomyIndex


def load_inputs() -> RegisterInputs:
    """Read every input table. Nothing here is regenerated or written."""
    return RegisterInputs(
        paragraphs=pd.read_parquet(PARAGRAPHS_PATH),
        issues=pd.read_parquet(PARAGRAPH_ISSUES_PATH),
        annotations=pd.read_parquet(PARAGRAPH_ANNOTATIONS_PATH),
        speeches=pd.read_parquet(SPEECHES_PATH),
        speech_annotations=pd.read_parquet(SPEECH_ANNOTATIONS_PATH),
        stats=pd.read_parquet(STATS_PATH),
        markers=pd.read_parquet(MARKERS_PATH),
        taxonomy=build_taxonomy_index(json.loads(TAXONOMY_PATH.read_text())),
    )


# ==========================================================================
# keyed-merge discipline
# ==========================================================================


def _require_full_merge(n_merged: int, inputs: Mapping[str, int], stage: str) -> None:
    """Raise unless a keyed merge kept every row of every input.

    `validate="one_to_one"` catches DUPLICATE keys but an inner join silently
    DROPS rows when the key sets merely diverge. Here that would shrink the
    corpus a trend is measured over while every published count still claimed
    the full 36,229. This is the repo-wide convention stated canonically in
    `taxonomy.py::_require_full_merge` and CLAUDE.md; it is restated locally so
    this module does not import a private name across module boundaries.
    """
    if any(n_merged != n for n in inputs.values()):
        breakdown = ", ".join(f"{k}={v}" for k, v in inputs.items())
        raise ValueError(
            f"{stage}: keyed merge changed the row count (merged={n_merged}; "
            f"{breakdown}). The key sets diverge and the inner join dropped "
            f"unmatched rows. Refusing to proceed."
        )


# ==========================================================================
# paragraph frame  (PURE)
# ==========================================================================


def label_resolution_report(
    annotations: pd.DataFrame, index: TaxonomyIndex
) -> pd.DataFrame:
    """Every distinct returned label with its assignment count and whether it
    matched `taxonomy_v1` exactly.

    Published so the note can cite the size of the case-variant repair rather
    than assert it. Raises through `resolve_labels` if anything is unresolvable.
    """
    counts: dict[str, int] = {}
    for labels in annotations["topics"]:
        for label in labels:
            counts[str(label)] = counts.get(str(label), 0) + 1
    rows = [
        {
            "returned_label": label,
            "canonical": index.resolve_labels([label])[0],
            "n_assignments": n,
            "exact_match": label in index.by_casefold.values(),
        }
        for label, n in counts.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("n_assignments", ascending=False)
        .reset_index(drop=True)
    )


def build_paragraph_frame(
    paragraphs: pd.DataFrame,
    issues: pd.DataFrame,
    annotations: pd.DataFrame,
    index: TaxonomyIndex,
) -> pd.DataFrame:
    """Join text length, legacy issue labels and LLM topic labels onto one frame
    keyed `(doc_name, para_idx)`.

    Args:
        paragraphs: `paragraphs.parquet` — `doc_name`, `para_idx`, `word_count`.
            It carries no `year`; the paragraph-level year lives on the issues
            table.
        issues: `paragraph_issues.parquet` — the key plus `year` plus the 15
            anchored issue booleans (the seven `Discovered N` free topics are
            deliberately excluded; the legacy taxonomy under test is the curated
            15). `year` is carried purely so `build_speech_panel` can check it
            against `speeches.parquet`; era banding itself uses the speech-level
            year.
        annotations: `paragraph_annotations.parquet` — the key plus `topics` and
            `proposal_values`.
        index: Taxonomy index used to canonicalize and classify topic labels.

    Returns:
        One row per paragraph with `word_count`, `year`, `n_legacy`,
        `legacy_zero`, `topics_resolved`, `n_llm`, `llm_zero`,
        `llm_all_non_policy` and `proposal_values`, plus the 15 legacy issue
        booleans.

    Raises:
        ValueError: If either keyed merge drops rows.
        KeyError: If a returned topic label does not resolve.
    """
    df = paragraphs[["doc_name", "para_idx", "word_count"]].merge(
        issues[["doc_name", "para_idx", "year", *LEGACY_ISSUES]],
        on=["doc_name", "para_idx"],
        validate="one_to_one",
    )
    _require_full_merge(
        len(df), {"paragraphs": len(paragraphs), "issues": len(issues)},
        "paragraphs x paragraph_issues",
    )
    df = df.merge(
        annotations[["doc_name", "para_idx", "topics", "proposal_values"]],
        on=["doc_name", "para_idx"],
        validate="one_to_one",
    )
    _require_full_merge(
        len(df), {"paragraphs": len(paragraphs), "annotations": len(annotations)},
        "(paragraphs+issues) x paragraph_annotations",
    )

    df["n_legacy"] = df[list(LEGACY_ISSUES)].sum(axis=1).astype(int)
    df["legacy_zero"] = df["n_legacy"] == 0

    non_policy = index.non_policy_level2
    resolved = [index.resolve_labels(labels) for labels in df["topics"]]
    df["topics_resolved"] = resolved
    df["n_llm"] = [len(t) for t in resolved]
    df["llm_zero"] = df["n_llm"] == 0
    # "All topics non-policy" is only meaningful for a paragraph that got topics
    # at all; the empty-label residue is its own series (`zero_topic_share`), not
    # silently folded into the non-policy rate.
    df["llm_all_non_policy"] = [
        bool(t) and all(name in non_policy for name in t) for t in resolved
    ]
    return df.reset_index(drop=True)


# ==========================================================================
# speech panel  (PURE)
# ==========================================================================

_PROPOSAL_VALUES_LEVELS = ("proposal", "values", "mixed", "neither")


@dataclass(frozen=True)
class SpeechPanel:
    """Per-speech sufficient statistics for every measure.

    Every published measure is a ratio of sums that are additive over speeches,
    so a speech-clustered bootstrap replicate is a weighted sum of these rows —
    no paragraph is ever revisited. `scalars` holds the additive numerators and
    denominators; `topic_counts[taxonomy]` is a (n_speeches x n_topics) matrix of
    label-assignment counts, which is likewise additive.

    Attributes:
        speeches: `doc_name`, `president`, `year`, `era`, `speech_type`, in the
            row order the matrices use.
        scalars: Additive per-speech sums, one column per sufficient statistic.
        topic_counts: Taxonomy name -> (n_speeches, n_topics) float matrix.
        topic_names: Taxonomy name -> the ordered topic names of its columns.
    """

    speeches: pd.DataFrame
    scalars: pd.DataFrame
    topic_counts: Mapping[str, np.ndarray]
    topic_names: Mapping[str, tuple[str, ...]]


def _topic_matrix(
    para: pd.DataFrame, doc_order: pd.Index, column: str, names: Sequence[str]
) -> np.ndarray:
    """Speech x topic assignment counts from a list-valued paragraph column.

    A paragraph with k topics contributes 1 to each of them (assignment-weighted,
    not paragraph-weighted), so the resulting distribution is over label
    assignments — the same convention on both taxonomies, which is what makes
    their entropies comparable as trends.
    """
    exploded = para[["doc_name", column]].explode(column).dropna(subset=[column])
    if exploded.empty:
        return np.zeros((len(doc_order), len(names)), dtype=float)
    counts = (
        exploded.groupby(["doc_name", column], observed=True).size().unstack(fill_value=0)
    )
    return (
        counts.reindex(index=doc_order, columns=list(names), fill_value=0)
        .to_numpy(dtype=float)
    )


def build_speech_panel(
    para: pd.DataFrame,
    speeches: pd.DataFrame,
    speech_annotations: pd.DataFrame,
    stats: pd.DataFrame,
    markers: pd.DataFrame,
    index: TaxonomyIndex,
) -> SpeechPanel:
    """Roll the paragraph frame up to one row per speech and attach style tables.

    Args:
        para: Output of `build_paragraph_frame` (its `year` column is the
            paragraph-level year, checked against `speeches` below).
        speeches: `speeches.parquet` — `doc_name`, `president`, `year`. This is
            the year era banding uses.
        speech_annotations: `speech_annotations.parquet` — `doc_name`,
            `speech_type`.
        stats: `speech_stats.parquet` — spaCy-derived per-speech counts.
        markers: `speech_markers.parquet` — regex/NRC per-speech counts.
        index: Taxonomy index (supplies the level-1 and level-2 column orders).

    Returns:
        A `SpeechPanel` whose rows are speeches in `doc_name` order.

    Raises:
        ValueError: If any keyed merge drops rows, if a speech carries
            paragraphs but has no speech-level row, if a speech has no
            paragraphs, or if the paragraph and speech tables disagree about a
            speech's year.
    """
    # Each speech-level table is attached under the same discipline — one-to-one
    # on `doc_name`, and every input keeps every row — so they are merged in a
    # loop rather than three times over: a table added here cannot arrive without
    # its row-count guard.
    meta = speeches[["doc_name", "president", "year"]]
    for table_name, table, columns in (
        ("speech_annotations", speech_annotations, ["speech_type"]),
        (
            "speech_stats",
            stats,
            ["n_tokens", "n_sents", "i_count", "we_count", "fk_grade"],
        ),
        ("speech_markers", markers, ["n_words", *STYLE_MARKERS]),
    ):
        meta = meta.merge(
            table[["doc_name", *columns]], on="doc_name", validate="one_to_one"
        )
        _require_full_merge(
            len(meta),
            {"speeches": len(speeches), table_name: len(table)},
            f"speeches x {table_name}",
        )

    meta = meta.sort_values("doc_name").reset_index(drop=True)
    meta["era"] = (meta["year"] // ERA_SPAN) * ERA_SPAN
    doc_order = pd.Index(meta["doc_name"], name="doc_name")

    missing = set(para["doc_name"]) - set(doc_order)
    if missing:
        raise ValueError(
            f"{len(missing)} speeches carry paragraphs but no speech-level row; "
            f"examples: {sorted(missing)[:3]}"
        )

    # Era banding — the grouping variable the entire analysis is reported over —
    # is taken from the SPEECH-level year above, while every other era-keyed
    # artifact in the repo (taxonomy.py's `attach_metadata`, the issues site)
    # bands off the PARAGRAPH-level year on `paragraph_issues.parquet`. The two
    # agree on today's corpus (0 of 1,057 speeches disagree, 0 carry more than
    # one paragraph-year), and they are supposed to: the paragraph year is
    # derived from the speech. If they ever drifted, this module's era series
    # would be silently incomparable with every other era series in the project
    # — an error invisible in the output and fatal to the published note. So it
    # is checked rather than assumed.
    speech_year = dict(zip(meta["doc_name"], meta["year"].astype(int)))
    disagreeing = sorted(
        doc
        for doc, years in para.groupby("doc_name")["year"].unique().items()
        if len(years) != 1 or int(years[0]) != speech_year[doc]
    )
    if disagreeing:
        raise ValueError(
            f"{len(disagreeing)} speeches disagree about their year between the "
            f"paragraph-level and speech-level tables (or carry more than one "
            f"paragraph-year); examples: {disagreeing[:3]}. Era banding would "
            f"differ from every other era-keyed artifact in the corpus. "
            f"Refusing to proceed."
        )

    work = para.copy()
    for level in _PROPOSAL_VALUES_LEVELS:
        work[f"pv_{level}"] = work["proposal_values"] == level
    work["level1_topics"] = [
        [index.parent[name] for name in topics] for topics in work["topics_resolved"]
    ]

    agg = work.groupby("doc_name").agg(
        n_paragraphs=("para_idx", "size"),
        para_words=("word_count", "sum"),
        llm_labels=("n_llm", "sum"),
        legacy_labels=("n_legacy", "sum"),
        llm_non_policy_paras=("llm_all_non_policy", "sum"),
        llm_zero_paras=("llm_zero", "sum"),
        legacy_zero_paras=("legacy_zero", "sum"),
        **{f"pv_{lvl}": (f"pv_{lvl}", "sum") for lvl in _PROPOSAL_VALUES_LEVELS},
    )
    agg = agg.reindex(doc_order, fill_value=0)

    # `agg` and `meta` are both in `doc_order`, so the per-speech style columns
    # are carried across positionally.
    scalars = agg.astype(float).reset_index(drop=True)
    for col in ("n_tokens", "n_sents", "i_count", "we_count"):
        scalars[col] = meta[col].to_numpy(dtype=float)
    # fk_grade is a per-speech average, so it is carried as a token-weighted
    # product and divided back out at aggregation time — averaging averages over
    # speeches of wildly different lengths would let a 200-word proclamation
    # outweigh a 30,000-word annual message.
    scalars["fk_x_tokens"] = (meta["fk_grade"] * meta["n_tokens"]).to_numpy(dtype=float)
    for col in ("n_words", *STYLE_MARKERS):
        scalars[col] = meta[col].to_numpy(dtype=float)

    if (scalars["n_paragraphs"] == 0).any():
        empty = meta.loc[scalars["n_paragraphs"] == 0, "doc_name"].tolist()
        raise ValueError(f"{len(empty)} speeches have no paragraphs: {empty[:3]}")

    topic_names = {
        "legacy15": tuple(LEGACY_ISSUES),
        "llm_level2": index.level2,
        "llm_level1": index.level1,
    }
    legacy = (
        work.groupby("doc_name")[list(LEGACY_ISSUES)]
        .sum()
        .reindex(doc_order, fill_value=0)
        .to_numpy(dtype=float)
    )
    topic_counts = {
        "legacy15": legacy,
        "llm_level2": _topic_matrix(work, doc_order, "topics_resolved", index.level2),
        "llm_level1": _topic_matrix(work, doc_order, "level1_topics", index.level1),
    }
    return SpeechPanel(
        speeches=meta, scalars=scalars, topic_counts=topic_counts, topic_names=topic_names
    )


# ==========================================================================
# measures  (PURE)
# ==========================================================================


def effective_topics(counts: np.ndarray, miller_madow: bool = True) -> np.ndarray:
    """Perplexity of a topic-attention distribution: `exp(Shannon entropy)`.

    "If attention were spread evenly across N topics, what is N?" Rising means
    talking about more things.

    Args:
        counts: Non-negative assignment counts with topics on the last axis.
            Leading axes are broadcast over, so a (n_replicates, n_topics)
            bootstrap matrix works unchanged.
        miller_madow: Apply the Miller-Madow bias correction
            `+ (K_observed - 1) / (2N)`. Plug-in entropy is biased *downward* at
            small N, and the thin early eras are precisely where a downward bias
            manufactures the rise this module is testing, so the correction is on
            by default and the uncorrected value ships separately.

    Returns:
        Effective topic counts, `nan` wherever no assignments were observed.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1)
    denom = total[..., None]
    p = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
    log_p = np.log(p, out=np.zeros_like(p), where=p > 0)
    entropy = -(p * log_p).sum(axis=-1)
    if miller_madow:
        observed = (counts > 0).sum(axis=-1)
        entropy = entropy + np.divide(
            observed - 1.0, 2.0 * total, out=np.zeros_like(entropy), where=total > 0
        )
    return np.where(total > 0, np.exp(entropy), np.nan)


def _ratio(
    numerator: np.ndarray, denominator: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """Elementwise ratio, `nan` where the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator * scale,
        denominator,
        out=np.full(np.broadcast(numerator, denominator).shape, np.nan),
        where=denominator > 0,
    )


def measures_from_sums(
    sums: pd.DataFrame, topic_counts: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Every published measure, from additive sums.

    Pure and shape-agnostic: one input row is a point estimate, `n` input rows
    are `n` bootstrap replicates. That is the whole reason the panel stores
    sufficient statistics rather than paragraphs.

    Args:
        sums: One row per estimate, columns matching `SpeechPanel.scalars`
            (already summed, and possibly genre-reweighted, over speeches).
        topic_counts: Taxonomy name -> matching (n_rows, n_topics) count matrix.

    Returns:
        A frame with the same row count and a `(measure, taxonomy)` column
        MultiIndex.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    n_paras = sums["n_paragraphs"].to_numpy(dtype=float)
    para_words = sums["para_words"].to_numpy(dtype=float)

    for taxonomy, counts in topic_counts.items():
        n_topics = counts.shape[-1]
        out[("effective_topics", taxonomy)] = effective_topics(counts)
        out[("effective_topics_plugin", taxonomy)] = effective_topics(
            counts, miller_madow=False
        )
        # Entropy normalized by log(K) puts a 15-category and a 50-category
        # taxonomy on one scale. Levels still are not comparable across
        # taxonomies in any deep sense — only the trends are — but this at least
        # removes the mechanical "more bins means more entropy" gap.
        #
        # NOT bounded at 1.0. The numerator carries the Miller-Madow correction
        # and the denominator log(K) does not, so the ratio can exceed 1 by
        # roughly (K_observed - 1) / (2N log K) — i.e. only where the sample is
        # small enough that the correction is a material share of the entropy.
        # In the shipped table it happens on exactly one row of 16,243 (year
        # 1965, legacy15, sotu_only: 1.0137, a single 34-paragraph speech). The
        # numerator is deliberately NOT switched to plug-in entropy to buy a
        # clean [0, 1] range: that would make this column disagree with
        # `effective_topics`, which is the published series, for a one-row
        # cosmetic gain. Read it as "0-1 up to the bias-correction term".
        out[("normalized_entropy", taxonomy)] = np.log(
            out[("effective_topics", taxonomy)]
        ) / np.log(n_topics)

    for taxonomy, labels in (("legacy15", "legacy_labels"), ("llm_level2", "llm_labels")):
        counted = sums[labels].to_numpy(dtype=float)
        out[("labels_per_paragraph", taxonomy)] = _ratio(counted, n_paras)
        # The length-invariant twin. Mean paragraph length falls ~28% across the
        # corpus, so a per-paragraph density can drop by that much while density
        # per unit of speech is flat.
        out[("labels_per_1k_words", taxonomy)] = _ratio(counted, para_words, 1_000.0)

    out[("non_policy_share", "legacy15")] = _ratio(
        sums["legacy_zero_paras"].to_numpy(dtype=float), n_paras
    )
    out[("non_policy_share", "llm_level2")] = _ratio(
        sums["llm_non_policy_paras"].to_numpy(dtype=float), n_paras
    )
    out[("zero_topic_share", "llm_level2")] = _ratio(
        sums["llm_zero_paras"].to_numpy(dtype=float), n_paras
    )

    proposal = sums["pv_proposal"].to_numpy(dtype=float)
    values = sums["pv_values"].to_numpy(dtype=float)
    mixed = sums["pv_mixed"].to_numpy(dtype=float)
    # Restricted to the unambiguous paragraphs: a share of 0.5 means proposals
    # and values appear equally often, below 0.5 means values dominate.
    out[("proposal_vs_values", NO_TAXONOMY)] = _ratio(proposal, proposal + values)
    out[("proposal_share", NO_TAXONOMY)] = _ratio(proposal + mixed, n_paras)
    out[("values_share", NO_TAXONOMY)] = _ratio(values + mixed, n_paras)
    # The residual stance. Published rather than left to prose because it is what
    # actually moves: `proposal_share` and `values_share` BOTH rise, so the story
    # is neutral description collapsing, and a reader cannot recover that from the
    # other two (they double-count `mixed`, so their sum is 1 + mixed_share, not
    # 1 - neither_share). With this column the whole four-way stance split is
    # recoverable: mixed_share = proposal_share + values_share + neither_share - 1.
    out[("neither_share", NO_TAXONOMY)] = _ratio(
        sums["pv_neither"].to_numpy(dtype=float), n_paras
    )
    # Published as a measure in its own right, not a footnote: it is the
    # denominator drift that could produce the "shallower" finding by itself.
    out[("mean_paragraph_words", NO_TAXONOMY)] = _ratio(para_words, n_paras)

    n_tokens = sums["n_tokens"].to_numpy(dtype=float)
    out[("fk_grade", NO_TAXONOMY)] = _ratio(
        sums["fk_x_tokens"].to_numpy(dtype=float), n_tokens
    )
    out[("words_per_sentence", NO_TAXONOMY)] = _ratio(
        n_tokens, sums["n_sents"].to_numpy(dtype=float)
    )
    i_count = sums["i_count"].to_numpy(dtype=float)
    out[("self_reference", NO_TAXONOMY)] = _ratio(
        i_count, i_count + sums["we_count"].to_numpy(dtype=float)
    )
    n_words = sums["n_words"].to_numpy(dtype=float)
    for marker in STYLE_MARKERS:
        out[(marker, NO_TAXONOMY)] = _ratio(
            sums[marker].to_numpy(dtype=float), n_words, 10_000.0
        )

    frame = pd.DataFrame(out, index=sums.index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["measure", "taxonomy"])
    return frame


# ==========================================================================
# genre treatments  (PURE)
# ==========================================================================


def reference_genre_mix(speeches: pd.DataFrame, paragraph_counts: pd.Series) -> pd.Series:
    """The common speech-type mix each era is standardized to.

    Chosen as the **corpus-pooled share of paragraphs among the four genres that
    occur in every one of the nine eras** — inaugural address, special message to
    Congress, annual message / State of the Union, and veto or signing statement.

    Why this mix and not another. Standardizing to the mix of *all ten* genres is
    impossible without inventing counterfactual cells: six genres (campaign
    debates, press conferences, public remarks, ...) simply do not exist before
    the 20th century, so their weight would have to be redistributed era by era,
    which is the confound rather than a control for it. Standardizing to a
    *uniform* mix over the four durable genres would score every era as if a
    signing statement mattered as much as an annual message, which no reading of
    the corpus supports. Pooled paragraph share keeps the reference anchored to
    how much of the record each durable genre actually is. Paragraph share rather
    than speech share because the reweighting redistributes paragraph mass.

    Args:
        speeches: One row per speech with a `speech_type` column.
        paragraph_counts: Paragraphs per speech, indexed like `speeches`.

    Returns:
        Weights over `REFERENCE_GENRES` summing to 1.

    Raises:
        ValueError: If `REFERENCE_GENRES` is no longer exactly the set of genres
            present in every era — a corpus change must not silently redefine
            what "genre-standardized" means.
    """
    present = pd.crosstab(speeches["era"], speeches["speech_type"])
    spanning = {col for col in present.columns if (present[col] > 0).all()}
    if spanning != set(REFERENCE_GENRES):
        raise ValueError(
            "the set of genres present in every era has changed: expected "
            f"{sorted(REFERENCE_GENRES)}, found {sorted(spanning)}. Re-derive the "
            "reference mix deliberately rather than letting it drift."
        )
    mass = (
        pd.Series(np.asarray(paragraph_counts, dtype=float))
        .groupby(speeches["speech_type"].to_numpy())
        .sum()
        .reindex(list(REFERENCE_GENRES), fill_value=0.0)
    )
    return mass / mass.sum()


def genre_cells(speeches: pd.DataFrame, group_col: str = "era") -> pd.DataFrame:
    """The (group x reference genre) cells, and whether each is thick enough.

    Published because the answer is not uniform: in the 1770 era two of the four
    reference genres carry fewer than `MIN_CELL_SPEECHES` speeches, so that era's
    standardized estimate rests on a renormalized two-genre mix. A reader has to
    be able to see that rather than take a CI on faith.
    """
    subset = speeches[speeches["speech_type"].isin(REFERENCE_GENRES)]
    cells = (
        subset.groupby([group_col, "speech_type"], observed=True)
        .size()
        .rename("n_speeches")
        .reset_index()
    )
    cells["included"] = cells["n_speeches"] >= MIN_CELL_SPEECHES
    return cells.sort_values([group_col, "speech_type"]).reset_index(drop=True)


def _cell_plan(
    panel: SpeechPanel, positions: np.ndarray, treatment: str, mix: pd.Series
) -> list[tuple[np.ndarray, float]]:
    """Decompose one group into `(speech positions, weight)` cells.

    `raw` and `sotu_only` are single unweighted cells. `genre_standardized`
    returns one cell per sufficiently thick reference genre, with weight
    `w_t * P_included / P_t` — i.e. the group's paragraph mass held constant but
    redistributed across genres in the reference proportions. Because every
    measure is a ratio whose denominator scales with that mass, this is exact
    direct standardization for the paragraph-denominated measures; for the
    word-denominated style rates it standardizes paragraph mass rather than word
    mass, which differs only to the extent that words-per-paragraph varies by
    genre.
    """
    types = panel.speeches["speech_type"].to_numpy()
    if treatment == "raw":
        return [(positions, 1.0)] if len(positions) else []
    if treatment == "sotu_only":
        sotu = positions[types[positions] == SOTU_TYPE]
        return [(sotu, 1.0)] if len(sotu) else []
    if treatment != "genre_standardized":
        raise ValueError(f"unknown genre treatment {treatment!r}")

    paras = panel.scalars["n_paragraphs"].to_numpy()
    # (speech positions, reference weight, paragraphs in the cell) per thick genre.
    eligible: list[tuple[np.ndarray, float, float]] = []
    for genre in REFERENCE_GENRES:
        members = positions[types[positions] == genre]
        if len(members) < MIN_CELL_SPEECHES:
            continue
        cell_paras = paras[members].sum()
        if cell_paras <= 0:
            continue
        eligible.append((members, float(mix[genre]), cell_paras))
    total_weight = sum(weight for _, weight, _ in eligible)
    if not eligible or total_weight <= 0:
        return []
    total_paras = sum(cell_paras for _, _, cell_paras in eligible)
    return [
        (members, (weight / total_weight) * total_paras / cell_paras)
        for members, weight, cell_paras in eligible
    ]


# ==========================================================================
# speech-clustered bootstrap  (PURE given a seeded Generator)
# ==========================================================================


def _replicate_weights(
    cells: Sequence[tuple[np.ndarray, float]], n_bootstrap: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Speech weights for the point estimate and every bootstrap replicate.

    Resampling is *within cell*, with each cell's speech count held fixed, so the
    genre-standardized arm keeps its design and the reference weights stay
    external to the resampling. Row 0 is the observed data.

    A replicate is expressed as a multinomial count vector over the cell's
    speeches rather than a list of drawn indices — mathematically identical to
    drawing `n` speeches with replacement, but it turns the whole bootstrap into
    two matrix products.

    Returns:
        `(positions, weights)` where `weights` has shape
        `(1 + n_bootstrap, len(positions))`.
    """
    if not cells:
        return np.zeros(0, dtype=int), np.zeros((1 + n_bootstrap, 0), dtype=float)
    positions = np.concatenate([members for members, _ in cells])
    weights = np.zeros((1 + n_bootstrap, len(positions)), dtype=float)
    offset = 0
    for members, alpha in cells:
        n = len(members)
        block = slice(offset, offset + n)
        weights[0, block] = alpha
        if n_bootstrap:
            draws = rng.multinomial(n, np.full(n, 1.0 / n), size=n_bootstrap)
            weights[1:, block] = draws * alpha
        offset += n
    return positions, weights


def _estimate_group(
    panel: SpeechPanel,
    cells: Sequence[tuple[np.ndarray, float]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, np.ndarray, pd.MultiIndex, dict[str, int]]:
    """Point estimate, bootstrap replicates and sample counts for one cell plan.

    Returns:
        `(point, replicates, columns, counts)` — `point` is one value per
        measure, `replicates` is `(n_bootstrap, n_measures)`, and `counts` gives
        the unweighted `n_speeches` / `n_paragraphs` / `n_words` behind the cell.
    """
    positions, weights = _replicate_weights(cells, n_bootstrap, rng)
    scalar_block = panel.scalars.to_numpy(dtype=float)[positions]
    sums = pd.DataFrame(weights @ scalar_block, columns=panel.scalars.columns)
    topic_sums = {
        name: weights @ matrix[positions] for name, matrix in panel.topic_counts.items()
    }
    measures = measures_from_sums(sums, topic_sums)
    counts = {
        "n_speeches": int(len(positions)),
        "n_paragraphs": int(panel.scalars["n_paragraphs"].to_numpy()[positions].sum()),
        "n_words": int(panel.scalars["para_words"].to_numpy()[positions].sum()),
    }
    return (
        measures.iloc[0],
        measures.to_numpy()[1:],
        measures.columns,
        counts,
    )


def _percentile_ci(replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile CI bounds and the count of usable replicates per column.

    Replicates that came out `nan` — a resample that happened to contain no
    paragraphs of the relevant kind, so the ratio was undefined — are excluded
    from the quantile and counted, rather than propagating a `nan` CI over an
    otherwise fine estimate or being silently treated as zeros.
    """
    width = replicates.shape[1] if replicates.ndim == 2 else 0
    low = np.full(width, np.nan)
    high = np.full(width, np.nan)
    if replicates.size == 0:
        return low, high, np.zeros(width, dtype=int)
    valid = np.isfinite(replicates)
    n_valid = valid.sum(axis=0)
    # Quantile only the columns that have something to quantile: nanquantile over
    # an all-nan column is a warning plus a nan, and an all-nan column is a real
    # state here (a measure whose denominator is empty in this cell).
    usable = np.flatnonzero(n_valid > 0)
    if usable.size:
        block = np.where(valid[:, usable], replicates[:, usable], np.nan)
        low[usable] = np.nanquantile(block, CI_ALPHA / 2, axis=0)
        high[usable] = np.nanquantile(block, 1 - CI_ALPHA / 2, axis=0)
    return low, high, n_valid


# ==========================================================================
# trend statistics  (PURE)
# ==========================================================================


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis (ties shared), `nan` preserved."""
    ranks = 1.0 + np.argsort(np.argsort(values, axis=-1, kind="stable"), axis=-1)
    # Average tied ranks, so a series that is flat (or constant at zero, which
    # several style markers are in the early eras) does not get a spurious
    # ordering out of argsort's tie-breaking. Ties are rare in bootstrap output,
    # so the rows that need it are found vectorized and only those are fixed up.
    tied = (np.diff(np.sort(values, axis=-1), axis=-1) == 0).any(axis=-1)
    for idx in zip(*np.nonzero(tied)):
        unique, inverse, counts = np.unique(
            values[idx], return_inverse=True, return_counts=True
        )
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks[idx])
        ranks[idx] = (sums / counts)[inverse]
    return ranks


def spearman_vs_position(values: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row against its column order (0, 1, 2, ...).

    The eras are equally spaced 30-year bands, so "against era" and "against
    position" are the same ranking.

    Two cases return `nan` rather than a number, both deliberately. A row
    containing any `nan` returns `nan`, because a trend across a partly missing
    series would not mean what the column says. A row that is *constant* also
    returns `nan`: its rank variance is zero, so the correlation is undefined,
    and reporting 0.0 there would read as "measured, and flat" when the truth is
    "not measurable". Ties short of a fully constant row are handled by rank
    averaging in `_rank`.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    n = values.shape[-1]
    out = np.full(values.shape[0], np.nan)
    usable = np.isfinite(values).all(axis=-1)
    if not usable.any() or n < 3:
        return out
    ranks = _rank(values[usable])
    position = np.arange(1.0, n + 1.0)
    ranks_centered = ranks - ranks.mean(axis=-1, keepdims=True)
    position_centered = position - position.mean()
    numerator = (ranks_centered * position_centered).sum(axis=-1)
    denominator = np.sqrt(
        (ranks_centered**2).sum(axis=-1) * (position_centered**2).sum()
    )
    out[usable] = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )
    return out


def trend_statistics(values: np.ndarray, eras: Sequence[int]) -> dict[str, np.ndarray]:
    """The three summary trend statistics, computed row-wise.

    Args:
        values: `(n_rows, n_eras)` measure values — one row for the observed
            series, one per bootstrap replicate.
        eras: The era labels of the columns, ascending.

    Returns:
        Statistic name -> one value per row.

    Raises:
        ValueError: If the statistics computed here are no longer exactly
            `TREND_STATISTICS`. The two are separate sources of truth and
            `build_trends` publishes by iterating the constant, so a statistic
            added here and not there would be computed on every arm and silently
            never reach the parquet — the one failure mode a note that promises
            to report every arm it computed cannot detect by reading the parquet.
            Same guard, and same reason, as `reference_genre_mix`.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    eras = list(eras)
    out = {
        "spearman_vs_era": spearman_vs_position(values),
        "delta_last_minus_first": values[:, -1] - values[:, 0],
    }
    for name, (late_eras, base_eras) in BLOCK_CONTRASTS.items():
        late = [eras.index(e) for e in late_eras if e in eras]
        base = [eras.index(e) for e in base_eras if e in eras]
        with np.errstate(invalid="ignore"):
            out[name] = (
                values[:, late].mean(axis=1) - values[:, base].mean(axis=1)
                if late and base
                else np.full(values.shape[0], np.nan)
            )
    if set(out) != set(TREND_STATISTICS):
        raise ValueError(
            "the computed trend statistics have drifted from the published set: "
            f"TREND_STATISTICS is {sorted(TREND_STATISTICS)}, this function "
            f"returns {sorted(out)}. Anything computed but not listed is never "
            "written to the parquet; anything listed but not computed raises in "
            "build_trends. Update both together."
        )
    return out


def bootstrap_p_value(replicates: np.ndarray) -> float:
    """Two-sided bootstrap-percentile p-value against a null of zero.

    `2 * min(share of replicates <= 0, share >= 0)`, floored at `1 / B` because a
    bootstrap of B draws cannot resolve a p-value below that. This is the
    percentile-interval inversion, not an exact test, and the note says so.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return float("nan")
    below = float((finite <= 0).mean())
    above = float((finite >= 0).mean())
    return float(min(1.0, max(2 * min(below, above), 1.0 / finite.size)))


# ==========================================================================
# assembly
# ==========================================================================

TREND_COLUMNS = [
    "unit",
    "period",
    "measure",
    "taxonomy",
    "genre_treatment",
    "statistic",
    "value",
    "ci_low",
    "ci_high",
    "p_value",
    "n_speeches",
    "n_paragraphs",
    "n_words",
    "n_bootstrap",
    "n_bootstrap_valid",
    "bootstrap_seed",
]


def _require_trend_columns(row: dict) -> None:
    """Raise unless `row` carries exactly `TREND_COLUMNS`.

    Args:
        row: One row dict destined for `pd.DataFrame(rows, columns=TREND_COLUMNS)`.

    Raises:
        ValueError: If a declared column is missing or an undeclared key present.
    """
    expected = set(TREND_COLUMNS)
    if set(row) != expected:
        missing = sorted(expected - set(row))
        unknown = sorted(set(row) - expected)
        raise ValueError(
            "a trend row does not match TREND_COLUMNS "
            f"(missing {missing}, unexpected {unknown}); DataFrame would have "
            "silently published an all-null column. Row: "
            f"{ {k: row[k] for k in sorted(row)} }"
        )


def _rows_for_group(
    unit: str,
    period: int,
    treatment: str,
    point: pd.Series,
    replicates: np.ndarray,
    columns: pd.MultiIndex,
    counts: Mapping[str, int],
    n_bootstrap: int,
    seed: int,
) -> list[dict]:
    low, high, valid = _percentile_ci(replicates)
    return [
        {
            "unit": unit,
            "period": period,
            "measure": measure,
            "taxonomy": taxonomy,
            "genre_treatment": treatment,
            "statistic": "level",
            "value": float(point.iloc[i]),
            "ci_low": float(low[i]),
            "ci_high": float(high[i]),
            "p_value": np.nan,
            **counts,
            "n_bootstrap": n_bootstrap,
            "n_bootstrap_valid": int(valid[i]),
            "bootstrap_seed": seed,
        }
        for i, (measure, taxonomy) in enumerate(columns)
    ]


def build_trends(
    panel: SpeechPanel,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Every measure, by era and by year, under all three genre treatments.

    Args:
        panel: Output of `build_speech_panel`.
        n_bootstrap: Speech-clustered bootstrap replicates per estimate.
        seed: Deterministic seed, recorded in every output row.

    Returns:
        Long-format trends with `unit` in `{"era", "year", "trend"}`. Era and
        year rows carry `statistic == "level"`; trend rows summarize the era
        series with a Spearman correlation and two contrasts, each with its own
        bootstrap CI and p-value.
    """
    rng = np.random.default_rng(seed)
    speeches = panel.speeches
    mix = reference_genre_mix(speeches, panel.scalars["n_paragraphs"])
    eras = sorted(speeches["era"].unique())
    rows: list[dict] = []
    # era_replicates[treatment][:, era_position, measure_position]
    era_replicates: dict[str, np.ndarray] = {}
    era_points: dict[str, np.ndarray] = {}
    columns: pd.MultiIndex | None = None

    for treatment in GENRE_TREATMENTS:
        per_era_point: list[np.ndarray] = []
        per_era_reps: list[np.ndarray] = []
        for era in eras:
            positions = np.flatnonzero(speeches["era"].to_numpy() == era)
            cells = _cell_plan(panel, positions, treatment, mix)
            point, replicates, cols, counts = _estimate_group(
                panel, cells, n_bootstrap, rng
            )
            columns = cols
            rows.extend(
                _rows_for_group(
                    "era", int(era), treatment, point, replicates, cols, counts,
                    n_bootstrap, seed,
                )
            )
            # `pd.DataFrame(rows, columns=...)` reindexes rather than validates: a
            # row key that is missing or misspelled becomes an all-NaN column in
            # the published artifact instead of an error. The exhaustive pass below
            # is what guarantees correctness; this checks the row just appended so
            # a key-set defect surfaces in milliseconds rather than after the
            # ~6-second bootstrap. `or [{}]` keeps an empty group raising the
            # intended ValueError rather than an IndexError.
            _require_trend_columns((rows or [{}])[-1])
            per_era_point.append(point.to_numpy(dtype=float))
            per_era_reps.append(replicates)
        era_points[treatment] = np.stack(per_era_point, axis=0)  # (n_eras, n_measures)
        era_replicates[treatment] = np.stack(per_era_reps, axis=1)  # (B, n_eras, n_meas)

    for treatment in YEAR_TREATMENTS:
        for year in sorted(speeches["year"].unique()):
            positions = np.flatnonzero(speeches["year"].to_numpy() == year)
            cells = _cell_plan(panel, positions, treatment, mix)
            if not cells:
                continue
            covered = np.concatenate([members for members, _ in cells])
            covered_paragraphs = panel.scalars["n_paragraphs"].to_numpy()[covered]
            if covered_paragraphs.sum() < MIN_YEAR_PARAGRAPHS:
                continue
            point, replicates, cols, counts = _estimate_group(
                panel, cells, n_bootstrap, rng
            )
            rows.extend(
                _rows_for_group(
                    "year", int(year), treatment, point, replicates, cols, counts,
                    n_bootstrap, seed,
                )
            )

    assert columns is not None
    for treatment in GENRE_TREATMENTS:
        points = era_points[treatment]
        reps = era_replicates[treatment]
        for i, (measure, taxonomy) in enumerate(columns):
            observed = trend_statistics(points[:, i][None, :], eras)
            replicated = trend_statistics(reps[:, :, i], eras)
            for statistic in TREND_STATISTICS:
                draws = replicated[statistic]
                low, high, valid = _percentile_ci(draws[:, None])
                rows.append({
                    "unit": "trend",
                    "period": pd.NA,
                    "measure": measure,
                    "taxonomy": taxonomy,
                    "genre_treatment": treatment,
                    "statistic": statistic,
                    "value": float(observed[statistic][0]),
                    "ci_low": float(low[0]),
                    "ci_high": float(high[0]),
                    "p_value": bootstrap_p_value(draws),
                    "n_speeches": pd.NA,
                    "n_paragraphs": pd.NA,
                    "n_words": pd.NA,
                    "n_bootstrap": n_bootstrap,
                    "n_bootstrap_valid": int(valid[0]),
                    "bootstrap_seed": seed,
                })

    # The per-iteration check above is the fast one; this is the exhaustive one,
    # since the trend rows come from a second builder the era loop never exercises.
    for row in rows:
        _require_trend_columns(row)

    trends = pd.DataFrame(rows, columns=TREND_COLUMNS)
    for col in ("period", "n_speeches", "n_paragraphs", "n_words"):
        trends[col] = trends[col].astype("Int64")
    return trends.sort_values(
        ["unit", "measure", "taxonomy", "genre_treatment", "statistic", "period"]
    ).reset_index(drop=True)


def rarefied_effective_topics(
    panel: SpeechPanel,
    n_draws: int = 400,
    seed: int = BOOTSTRAP_SEED,
    budget: int | None = None,
) -> pd.DataFrame:
    """Effective topics per era after subsampling every era to a common size.

    The sharpest rival to a rising-breadth finding is that observed entropy grows
    with sample size: more paragraphs means more chances to touch a rare topic,
    and the corpus's early eras are its smallest (355 paragraphs in the 1770 era
    against 5,612 in the 2010s, a 16-fold gap). The Miller-Madow term inside
    `effective_topics` is only a first-order correction and does not settle it.

    Rarefaction settles it directly: draw whole speeches, without replacement,
    until a common paragraph budget is met, and recompute. Speeches rather than
    paragraphs because the clustering argument applies here exactly as it does to
    the bootstrap. The budget defaults to the smallest era's paragraph count, so
    every era is scored on the terms of the thinnest one.

    This is a diagnostic rather than a published series, and it is deliberately
    kept out of `trends.parquet`: its spread is subsampling variation, not
    sampling uncertainty, and putting it in a column named `ci_low` beside
    bootstrap intervals would invite exactly the wrong reading.

    Args:
        panel: Output of `build_speech_panel`.
        n_draws: Subsamples per era.
        seed: Deterministic seed.
        budget: Paragraph budget per draw; defaults to the smallest era's total.

    Returns:
        One row per (era, taxonomy) with the full-sample value, the mean rarefied
        value and its 2.5/97.5 subsampling percentiles, and the budget used.
    """
    rng = np.random.default_rng(seed)
    eras = sorted(panel.speeches["era"].unique())
    era_of = panel.speeches["era"].to_numpy()
    paragraphs = panel.scalars["n_paragraphs"].to_numpy()
    if budget is None:
        budget = int(min(paragraphs[era_of == era].sum() for era in eras))

    rows = []
    for taxonomy, matrix in panel.topic_counts.items():
        for era in eras:
            positions = np.flatnonzero(era_of == era)
            draws = []
            for _ in range(n_draws):
                order = rng.permutation(positions)
                total = 0
                taken = []
                for position in order:
                    taken.append(position)
                    total += paragraphs[position]
                    if total >= budget:
                        break
                draws.append(float(effective_topics(matrix[taken].sum(axis=0))))
            rows.append({
                "era": int(era),
                "taxonomy": taxonomy,
                "full_sample": float(effective_topics(matrix[positions].sum(axis=0))),
                "rarefied_mean": float(np.mean(draws)),
                "rarefied_low": float(np.quantile(draws, CI_ALPHA / 2)),
                "rarefied_high": float(np.quantile(draws, 1 - CI_ALPHA / 2)),
                "budget_paragraphs": int(budget),
                "n_draws": int(n_draws),
                "seed": int(seed),
            })
    return pd.DataFrame(rows)


ICC_COLUMNS = ("n_legacy", "n_llm", "llm_zero", "legacy_zero", "llm_all_non_policy")


def within_speech_icc(
    paragraphs: pd.DataFrame, columns: Sequence[str] = ICC_COLUMNS
) -> pd.DataFrame:
    """One-way random-effects ICC(1) of paragraph quantities, grouped by speech.

    The bootstrap resamples speeches rather than paragraphs, and that choice
    widens every interval in `trends.parquet`. It therefore has to be *sourced*
    rather than asserted: this is the number that justifies it, computed from the
    shipped paragraph frame instead of inherited from a prior session's scratch
    work.

    `ICC = (MSB - MSW) / (MSB + (k0 - 1) MSW)` with speeches as groups, where
    `k0 = (N - sum(n_i^2)/N) / (k - 1)` is the standard unequal-group-size
    correction (speech lengths in this corpus range from 1 to several hundred
    paragraphs, so the equal-`n` formula would not apply). An ICC of 0 means
    paragraphs within a speech are no more alike than paragraphs across speeches
    — the only case in which a paragraph-level bootstrap would be honest.

    Diagnostic only: nothing here reaches `trends.parquet`, for the same reason
    `rarefied_effective_topics` does not — it is not a sampling interval and must
    not be read beside one.

    Args:
        paragraphs: Output of `build_paragraph_frame`, keyed by `doc_name`.
        columns: Paragraph-level numeric or boolean columns to score.

    Returns:
        One row per column with `icc`, the group count and the observation count.
    """
    groups = paragraphs["doc_name"].to_numpy()
    codes, sizes = np.unique(groups, return_counts=True)
    index = pd.Index(codes)
    k = len(codes)
    n_total = float(sizes.sum())
    if k < 2 or n_total <= k:
        raise ValueError(
            f"ICC needs at least 2 speeches and more paragraphs than speeches; "
            f"got {k} speeches and {int(n_total)} paragraphs."
        )
    k0 = (n_total - (sizes.astype(float) ** 2).sum() / n_total) / (k - 1)

    rows: list[dict] = []
    for column in columns:
        series = paragraphs[column].astype(float)
        group_means = series.groupby(groups).mean().reindex(index)
        grand = series.mean()
        ss_between = float((sizes * (group_means.to_numpy() - grand) ** 2).sum())
        ss_within = float(
            ((series.to_numpy() - group_means.reindex(groups).to_numpy()) ** 2).sum()
        )
        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (n_total - k)
        denominator = ms_between + (k0 - 1) * ms_within
        rows.append({
            "column": column,
            "icc": float((ms_between - ms_within) / denominator)
            if denominator > 0
            else float("nan"),
            "n_speeches": int(k),
            "n_paragraphs": int(n_total),
        })
    return pd.DataFrame(rows)


def era_counts(panel: SpeechPanel) -> pd.DataFrame:
    """Speeches, presidents, paragraphs, annual messages and words per era.

    Published beside every trend: the 1770 era holds 28 speeches and 355
    paragraphs (11 of those speeches are annual messages), which is not enough to
    support a confident interval however narrow the bootstrap happens to come out.

    `n_presidents` and `top_president_paragraph_share` are here because every
    claim in the note is a claim *about presidents*, while the bootstrap
    resamples speeches. The endpoint eras rest on two and three presidents
    respectively, so an era-level contrast at the endpoints is close to a
    contrast between a handful of individuals, and the interval — which treats
    speeches as exchangeable — does not price that in. The share column says how
    lopsided the era is, and the endpoints are very lopsided: 71.3% of the 1770
    era's paragraphs are Washington's and 50.8% of the 2010 era's are Trump's, so
    each of those eras is one president's prose as much as it is an era.
    """
    frame = panel.speeches[["era", "president", "speech_type"]].copy()
    frame["n_paragraphs"] = panel.scalars["n_paragraphs"].to_numpy()
    frame["para_words"] = panel.scalars["para_words"].to_numpy()
    frame["is_sotu"] = frame["speech_type"] == SOTU_TYPE
    grouped = frame.groupby("era").agg(
        n_speeches=("speech_type", "size"),
        n_presidents=("president", "nunique"),
        n_sotu_speeches=("is_sotu", "sum"),
        n_paragraphs=("n_paragraphs", "sum"),
        n_words=("para_words", "sum"),
    )
    by_president = frame.groupby(["era", "president"])["n_paragraphs"].sum()
    grouped["top_president_paragraph_share"] = (
        by_president.groupby("era").max() / grouped["n_paragraphs"]
    )
    sotu = frame[frame["is_sotu"]].groupby("era")["n_paragraphs"].sum()
    grouped["n_sotu_paragraphs"] = sotu.reindex(grouped.index, fill_value=0)
    return grouped.reset_index()


def write_trends(trends: pd.DataFrame) -> Path:
    """Write `data/register/trends.parquet`, creating the directory if needed."""
    REGISTER_DIR.mkdir(parents=True, exist_ok=True)
    trends.to_parquet(TRENDS_PATH, index=False)
    return TRENDS_PATH


def main() -> int:
    """Rebuild the register trends table from the frozen inputs."""
    parser = argparse.ArgumentParser(
        description="Breadth / depth / register trends of the presidential record",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    inputs = load_inputs()
    para = build_paragraph_frame(
        inputs.paragraphs, inputs.issues, inputs.annotations, inputs.taxonomy
    )
    panel = build_speech_panel(
        para,
        inputs.speeches,
        inputs.speech_annotations,
        inputs.stats,
        inputs.markers,
        inputs.taxonomy,
    )
    trends = build_trends(panel, n_bootstrap=args.n_bootstrap, seed=args.seed)
    path = write_trends(trends)
    print(f"wrote {len(trends):,} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
