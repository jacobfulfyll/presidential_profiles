"""Corpus-derived two-level topic taxonomy.

The anchored CorEx taxonomy in `issues.py` can only find what its 15 anchor
lists already name, and it is blind to 28.7% of paragraphs (10,381 of 36,229
carry zero anchored issues) in a *time-varying* way — blind precisely to the
19th century's defining concerns (Indian tribal relations, Reconstruction,
coinage, internal improvements, civil-service reform, polygamy in the
Territories). `embed_topics.py` is the bias-free counterweight: it lets the
corpus partition itself with no supplied vocabulary. This module turns that
partition — plus the raw paragraph text — into a *named* two-level taxonomy.

The one hard rule: the taxonomy is derived FROM the corpus by a real LLM pass
over sampled paragraphs. Topic names, definitions, and crosswalk mappings are
never authored from a model's prior — fabricating a "corpus-derived" taxonomy is
the exact researcher degree of freedom this module exists to eliminate. So the
pipeline is:

  1. Stratify a sample across nine 30-year eras, OVERSAMPLING the unlabeled
     mass, and attach each era's dominant k=40 clusters + their c-TF-IDF terms.
  2. PER-ERA PROPOSAL — one LLM call per era, seeing that era's paragraphs
     ALONE. No cross-era context: that isolation is the anachronism guard (the
     1800s model never sees the word "healthcare").
  3. MERGE — one LLM call unifies the era proposals into level-1 domains (broad,
     mapping onto the legacy 15, plus first-class non-policy domains) and level-2
     topics (specific, period-bound entries preserved as distinct).
  4. CROSSWALK — one LLM call maps every legacy issue to >= 1 level-2 topic so
     the existing issue pages stay reconstructable.
  5. VALIDATE — label a held-out 200 (100 from the unlabeled mass) against the
     frozen taxonomy; coverage must reach >= 90%. Iterate the merge once if not.

Everything that spends money is isolated behind small `_client()`-taking
functions; the sampling, coverage, and structural-validation logic is pure and
tested on synthetic frames (see tests/test_taxonomy.py). Provenance — model IDs,
prompts, sample IDs, cost from ACTUAL usage — is written to a Manifest via the
`llm_annotations` layer so the derivation is fully reproducible.

Run as:
    PYTHONPATH=src set -a; source .env.local; set +a
    arch -x86_64 .venv/bin/python -m presidential_profiles.taxonomy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import DATA_DIR
from .embed_topics import CLUSTERS_META_PATH, CLUSTERS_PATH
from .issues import PARA_LABELS_PATH
from .llm_annotations import (
    ANNOTATIONS_DIR,
    Manifest,
    corpus_fingerprint,
    write_manifest,
)

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"
TAXONOMY_PATH = ANNOTATIONS_DIR / "taxonomy_v1.json"
CROSSWALK_PATH = ANNOTATIONS_DIR / "crosswalk_v1.json"

# --------------------------------------------------------------------------
# taxonomy shape constants (targets, not authored content)
# --------------------------------------------------------------------------

# The 15 anchored issues from issues.py, in order. A paragraph with zero of
# these across all 15 is the "unlabeled" mass this taxonomy targets.
LEGACY_ISSUES = [
    "Economy & jobs", "Taxes & budget", "War & military", "Foreign policy",
    "Immigration", "Civil rights & race", "Health care", "Education",
    "Crime & justice", "Energy & environment", "Trade & tariffs",
    "Agriculture", "Religion & values", "Money & banking", "Infrastructure",
]

# Crosswalk compatibility target: the 15 legacy issues + "Security & peace"
# (the name docs/issues/security-and-peace.html gives CorEx's Discovered 5).
# Every one of these must map to >= 1 level-2 topic or a live site page breaks.
# There is one docs/issues/<slug>.html page per crosswalk issue, so "every issue
# maps to >= 1 topic" (check_crosswalk) is exactly "every page reconstructable".
SECURITY_PEACE = "Security & peace"
CROSSWALK_ISSUES = LEGACY_ISSUES + [SECURITY_PEACE]

# Non-policy domains the vision names as first-class level-1 entries. Checked for
# presence, not authored into the taxonomy — the merge must surface its own
# wording; these are the semantic buckets that must exist.
REQUIRED_NON_POLICY = [
    "ceremonial", "personal narrative", "procedural/administrative",
    "values appeal",
]

# Because the merge surfaces its OWN wording (e.g. "Faith & National Values" for
# the values-appeal bucket, "Procedural & Administrative" for the procedural one),
# each required bucket is matched by ANY of these distinctive keywords appearing
# — case-insensitively — in a non-policy domain's name, not by exact string equality.
# validate_structure uses this to enforce REQUIRED_NON_POLICY; a re-run that drops
# one of the four buckets surfaces it in the report's `missing_non_policy`.
REQUIRED_NON_POLICY_SYNONYMS = {
    "ceremonial": ("ceremonial", "commemorat"),
    "personal narrative": ("personal narrative", "personal", "autobiograph", "biograph"),
    "procedural/administrative": ("procedural", "administrative"),
    "values appeal": ("values", "faith", "moral"),
}

# Sampling / gate parameters.
ERA_SPAN = 30              # 30-year eras: 1770, 1800, ... 2010 (nine bands)
PER_ERA_SAMPLE = 80        # ~720 total across 9 eras (plan target: 600-900)
HELDOUT_TOTAL = 200        # coverage gate sample (100 unlabeled + 100 labeled)
HELDOUT_UNLABELED = 100
DOMINANT_CLUSTERS = 10     # k=40 clusters shown per era as a vocabulary signal
COVERAGE_BATCH = 20        # paragraphs per held-out labeling call
MAX_PARA_CHARS = 1500      # truncate the ~1% of paragraphs longer than this
RANDOM_STATE = 42          # matches every other model in the repo

# Gates (from the plan's acceptance criteria / bail conditions).
COVERAGE_GATE = 0.90
LEVEL2_MIN, LEVEL2_MAX = 35, 50
LEVEL2_HARD_MAX = 60       # bail: a merge over this many level-2 topics

# --------------------------------------------------------------------------
# models & pricing (authoritative rates for cost-from-actuals)
# --------------------------------------------------------------------------

PROPOSAL_MODEL = "claude-opus-4-8"   # era proposals, merge, crosswalk, checks
COVERAGE_MODEL = "claude-sonnet-5"   # bulk held-out labeling (cheaper)

# $ per token. Opus 4.8: $5/$25 in/out. Sonnet 5 intro rates: $2/$10.
# Cache-write is 1.25x the input rate; cache-read is 0.1x.
_RATES = {
    "claude-opus-4-8": dict(inp=5.0, out=25.0, cw=6.25, cr=0.50),
    "claude-sonnet-5": dict(inp=2.0, out=10.0, cw=2.50, cr=0.20),
}


# ==========================================================================
# cost accounting — cost_usd is real usage-derived, never an estimate
# ==========================================================================


@dataclass
class Spend:
    """Running tally of ACTUAL API usage across a taxonomy run.

    `add` is called with every `resp.usage`; `cost_usd` sums the four token
    classes at each model's rate. The manifest's `cost_usd` is this number —
    the repo convention is that a recorded cost is what the run really cost, so
    it can never quietly claim a run was cheaper than it was."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    n_requests: int = 0

    def add(self, model: str, usage) -> None:
        r = _RATES[model]
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_creation_input_tokens += cw
        self.cache_read_input_tokens += cr
        self.n_requests += 1
        self.cost_usd += (
            inp * r["inp"] + out * r["out"] + cw * r["cw"] + cr * r["cr"]
        ) / 1_000_000


# ==========================================================================
# data loading & era-stratified sampling  (PURE — no API, unit tested)
# ==========================================================================


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Read the four inputs: paragraphs (text), paragraph_issues (labels + year),
    paragraph_clusters (k40/k15), and the cluster meta (c-TF-IDF terms)."""
    paras = pd.read_parquet(PARAGRAPHS_PATH)
    issues = pd.read_parquet(PARA_LABELS_PATH)
    clusters = pd.read_parquet(CLUSTERS_PATH)
    cmeta = json.loads(CLUSTERS_META_PATH.read_text())
    return paras, issues, clusters, cmeta


def _require_full_merge(n_merged: int, inputs: dict[str, int], stage: str) -> None:
    """Raise unless a keyed merge kept every row of every input.

    `validate="one_to_one"` catches DUPLICATE keys, but an inner join silently
    DROPS rows when the two key sets merely diverge — the exact silent-shrink the
    repo's keyed-merge convention exists to prevent (CLAUDE.md). Here it would be
    worse than a mislabel: the sample would be drawn from the intersection while
    the manifest stamps a full-corpus fingerprint over it. So we assert the merged
    length equals EACH input's length and fail loudly with the count breakdown."""
    if any(n_merged != n for n in inputs.values()):
        breakdown = ", ".join(f"{k}={v}" for k, v in inputs.items())
        raise ValueError(
            f"{stage}: keyed merge changed the row count (merged={n_merged}; "
            f"{breakdown}). The key sets diverge and the inner join dropped "
            f"unmatched rows — the sample would be an intersection while the "
            f"manifest fingerprints the full corpus. Refusing to proceed."
        )


def attach_metadata(
    paras: pd.DataFrame, issues: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Join text + year/president + cluster onto one frame keyed (doc_name,
    para_idx).

    paragraphs.parquet has NO year/president columns — they are sourced from
    paragraph_issues via a keyed merge (validate="one_to_one"), never positional
    alignment (a repo-wide convention; see CLAUDE.md). Adds `era` (30-year band),
    `n_legacy` (count of the 15 anchored issues that fired), and `unlabeled`
    (n_legacy == 0). Raises (via `_require_full_merge`) if either keyed merge
    silently drops rows because the key sets diverge."""
    keep_issue_cols = ["doc_name", "para_idx", "president", "year", *LEGACY_ISSUES]
    df = paras.merge(
        issues[keep_issue_cols], on=["doc_name", "para_idx"], validate="one_to_one"
    )
    _require_full_merge(len(df), {"paras": len(paras), "issues": len(issues)},
                        "paragraphs x paragraph_issues")
    df = df.merge(
        clusters[["doc_name", "para_idx", "cluster_k40"]],
        on=["doc_name", "para_idx"],
        validate="one_to_one",
    )
    _require_full_merge(len(df), {"paras": len(paras), "clusters": len(clusters)},
                        "(paragraphs+issues) x paragraph_clusters")
    df["n_legacy"] = df[LEGACY_ISSUES].sum(axis=1)
    df["unlabeled"] = df["n_legacy"] == 0
    df["era"] = (df["year"] // ERA_SPAN) * ERA_SPAN
    return df.reset_index(drop=True)


def _sample_ids(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> list[int]:
    """Index labels of n rows drawn without replacement (all of them if n >= len)."""
    if n <= 0 or len(pool) == 0:
        return []
    if len(pool) <= n:
        return list(pool.index)
    return list(rng.choice(pool.index.to_numpy(), size=n, replace=False))


def draw_heldout(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Held-out coverage sample: HELDOUT_UNLABELED from the unlabeled mass, the
    rest from the labeled mass. Drawn BEFORE the training sample so the two are
    disjoint (a coverage gate on paragraphs the taxonomy was built from would be
    circular)."""
    rng = np.random.default_rng(seed)
    unl = _sample_ids(df[df["unlabeled"]], HELDOUT_UNLABELED, rng)
    labeled_n = HELDOUT_TOTAL - len(unl)
    lab = _sample_ids(df[~df["unlabeled"]], labeled_n, rng)
    return df.loc[unl + lab].copy()


def draw_era_samples(
    df: pd.DataFrame, per_era: int = PER_ERA_SAMPLE, seed: int = RANDOM_STATE
) -> dict[int, pd.DataFrame]:
    """Per-era training sample, oversampling the unlabeled mass to ~half each era.

    Natural unlabeled rates run 19-38%; drawing half the sample from the
    unlabeled pool forces the LLM to confront the blind mass rather than
    re-describe what the anchored issues already cover. Shortfalls in either pool
    are backfilled from the other so each era still yields ~per_era paragraphs.
    Distinct per-era seeds keep the draw reproducible and era-independent."""
    out: dict[int, pd.DataFrame] = {}
    for i, era in enumerate(sorted(df["era"].unique())):
        rng = np.random.default_rng(seed + 1000 + i)
        era_df = df[df["era"] == era]
        half = per_era // 2
        unl = _sample_ids(era_df[era_df["unlabeled"]], half, rng)
        lab = _sample_ids(era_df[~era_df["unlabeled"]], per_era - len(unl), rng)
        # Backfill an unlabeled shortfall from remaining unlabeled rows.
        if len(unl) + len(lab) < per_era:
            chosen = set(unl) | set(lab)
            rest = era_df.loc[~era_df.index.isin(chosen)]
            fill = _sample_ids(rest, per_era - len(unl) - len(lab), rng)
            lab = lab + fill
        out[era] = era_df.loc[unl + lab].copy()
    return out


def era_label(era_df: pd.DataFrame) -> str:
    """Human year-range for an era, e.g. "1789-1799" — the temporal anchor the
    per-era prompt uses so the model reasons within the period, not against a
    nominal band boundary."""
    return f"{int(era_df['year'].min())}-{int(era_df['year'].max())}"


def dominant_cluster_terms(
    df: pd.DataFrame, era: int, cmeta: dict, top: int = DOMINANT_CLUSTERS
) -> list[dict]:
    """The era's most-represented k=40 clusters with their c-TF-IDF terms.

    Computed over the era's FULL paragraph population (not the sample) so the
    vocabulary signal is stable. This is what grounds the proposal in the
    corpus's own partition rather than the model's prior — the model is told
    "these word-clusters dominate this era" and proposes topics that account for
    them."""
    terms_by_cluster = {c["cluster"]: c["terms"] for c in cmeta["clusters"]["k40"]}
    counts = df[df["era"] == era]["cluster_k40"].value_counts().head(top)
    return [
        {"cluster": int(c), "size": int(n), "terms": terms_by_cluster[int(c)]}
        for c, n in counts.items()
    ]


def register_samples(era_samples: dict[int, pd.DataFrame]) -> dict[str, dict]:
    """Assign each sampled paragraph a short stable id (p0001, ...) mapped to its
    real (doc_name, para_idx, year, text).

    The LLM cites exemplars by these short ids; the merge carries them through;
    then they are resolved back to real corpus paragraph keys. This keeps every
    exemplar in the final taxonomy grounded in an actual paragraph rather than a
    plausible-looking id the model invented."""
    registry: dict[str, dict] = {}
    i = 0
    for era in sorted(era_samples):
        for _, row in era_samples[era].iterrows():
            i += 1
            registry[f"p{i:04d}"] = {
                "sample_id": f"p{i:04d}",
                "doc_name": row["doc_name"],
                "para_idx": int(row["para_idx"]),
                "year": int(row["year"]),
                "era": int(era),
                "text": row["text"],
            }
    return registry


def _truncate(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_PARA_CHARS else text[:MAX_PARA_CHARS] + " [...]"


def format_paragraphs(rows: list[dict]) -> str:
    """Render sample paragraphs for a prompt: `[p0042] (1793) <text>` per line."""
    return "\n\n".join(
        f"[{r['sample_id']}] ({r['year']}) {_truncate(r['text'])}" for r in rows
    )


# ==========================================================================
# coverage & structural validation  (PURE — no API, unit tested)
# ==========================================================================


def _is_covered(
    labels: dict[str, list[str]], sid: str, valid_names: set[str] | None
) -> bool:
    """Whether one held-out id counts as covered. With `valid_names` (the frozen
    level-2 name set) only an EXACT level-2 label counts — a near-miss string the
    model invented does not. With `valid_names=None` any non-empty label list
    counts (legacy behavior). A missing id is uncovered either way. Shared by the
    coverage gate and `uncovered_ids` so the two can never disagree."""
    got = labels.get(sid) or []
    if valid_names is None:
        return bool(got)
    return any(lbl in valid_names for lbl in got)


def compute_coverage(
    labels: dict[str, list[str]],
    sample_ids: list[str],
    valid_names: set[str] | None = None,
) -> float:
    """Fraction of the held-out sample that received >= 1 level-2 label. Missing
    ids count as uncovered (a dropped paragraph is a coverage failure, not a
    silent omission).

    When `valid_names` (the frozen level-2 name set) is given, a paragraph counts
    as covered only if it carries at least one label that is an EXACT level-2
    name. A paragraph whose only labels are near-miss strings the model invented
    (in the real run 2/292 assignments were non-exact) is a miss, not a hit — a
    coverage gate must not be satisfied by labels that don't exist in the taxonomy.
    With `valid_names=None` any non-empty label list counts (legacy behavior)."""
    if not sample_ids:
        return 0.0
    covered = sum(1 for sid in sample_ids if _is_covered(labels, sid, valid_names))
    return covered / len(sample_ids)


def uncovered_ids(
    labels: dict[str, list[str]],
    sample_ids: list[str],
    valid_names: set[str],
) -> list[str]:
    """The held-out ids NOT covered under the strict name-validated predicate, in
    `sample_ids` order — the exact complement of the coverage gate.

    run() feeds these paragraphs to revise_merge. Sharing `_is_covered` with the
    gate closes a mismatch where a strict shortfall caused by near-miss label
    names would trigger a revision with those very paragraphs invisible in its
    input (the legacy any-non-empty rule counted a near-miss-labeled paragraph as
    covered here while the gate counted it as a miss)."""
    return [sid for sid in sample_ids if not _is_covered(labels, sid, valid_names)]


def unknown_labels(
    labels: dict[str, list[str]], valid_names: set[str]
) -> list[str]:
    """Sorted, de-duplicated label names the coverage pass emitted that are NOT
    exact level-2 names. Surfaced into provenance so a run records exactly which
    near-miss strings the labeler produced (the real run had 2/292)."""
    seen = set()
    for topics in labels.values():
        for lbl in topics or []:
            if lbl not in valid_names:
                seen.add(lbl)
    return sorted(seen)


def _norm_name(name: str) -> str:
    """Case- and whitespace-normalized topic name, for duplicate detection."""
    return " ".join(str(name).split()).lower()


def validate_structure(taxonomy: dict) -> dict:
    """Structural checks on a merged taxonomy. Returns a report dict and NEVER
    raises (contract: the caller decides which failures are fatal) — so every
    field access below goes through `.get()`, tolerating a malformed level-1/level-2
    entry (missing name/level1/etc.) rather than KeyError-ing on it."""
    level1 = taxonomy.get("level1", [])
    level2 = taxonomy.get("level2", [])
    l1_names = {n for d in level1 if (n := d.get("name"))}
    non_policy = [d for d in level1 if d.get("kind") == "non-policy"]
    non_policy_lc = [d.get("name", "").lower() for d in non_policy]

    # A truthy parent name that resolves to no level-1 domain is an orphan; a
    # topic with a missing/empty parent is caught by level2_missing-style malformity
    # rather than sorted() into a list of strings alongside a None.
    orphan_parents = sorted(
        {p for t in level2 if (p := t.get("level1")) and p not in l1_names}
    )
    missing_fields = [
        t.get("name", "<unnamed>")
        for t in level2
        if not (t.get("name") and t.get("definition") and t.get("era_note"))
    ]
    # Two level-2 topics that normalize to the same name would give the downstream
    # annotation pass an ambiguous label key; surfaced as a report field (the frozen
    # artifact is clean) — deliberately NOT a bail condition.
    norm_counts: dict[str, int] = {}
    for t in level2:
        if name := t.get("name"):
            norm_counts[_norm_name(name)] = norm_counts.get(_norm_name(name), 0) + 1
    duplicate_level2_names = sorted(n for n, c in norm_counts.items() if c > 1)
    # Enforce the four required non-policy buckets by semantics: each is present
    # iff some non-policy domain name contains one of its distinctive keywords.
    missing_non_policy = [
        bucket
        for bucket in REQUIRED_NON_POLICY
        if not any(
            kw in name
            for name in non_policy_lc
            for kw in REQUIRED_NON_POLICY_SYNONYMS[bucket]
        )
    ]
    return {
        "n_level1": len(level1),
        "n_level2": len(level2),
        "n_non_policy": len(non_policy),
        "non_policy_names": [d.get("name", "") for d in non_policy],
        "missing_non_policy": missing_non_policy,
        "non_policy_complete": not missing_non_policy,
        "orphan_parents": orphan_parents,
        "level2_missing_fields": missing_fields,
        "duplicate_level2_names": duplicate_level2_names,
        "in_target_range": LEVEL2_MIN <= len(level2) <= LEVEL2_MAX,
        "over_hard_max": len(level2) > LEVEL2_HARD_MAX,
    }


def check_crosswalk(crosswalk: dict, taxonomy: dict) -> dict:
    """Every crosswalk issue must map to >= 1 level-2 topic, and every mapped
    topic name must exist in the taxonomy. Returns missing issues and dangling
    topic references. Never raises: a mapping missing `legacy_issue` is skipped,
    and a null/absent `level2_topics` is treated as the empty list rather than
    KeyError-ing or being iterated as None."""
    l2_names = {n for t in taxonomy.get("level2", []) if (n := t.get("name"))}
    mapped: dict[str, list[str]] = {}
    for m in crosswalk.get("mappings", []):
        iss = m.get("legacy_issue")
        if iss is not None:
            mapped[iss] = m.get("level2_topics") or []
    required = set(CROSSWALK_ISSUES)
    missing_issues = [iss for iss in CROSSWALK_ISSUES if not mapped.get(iss)]
    dangling = sorted(
        {t for topics in mapped.values() for t in topics if t not in l2_names}
    )
    # Issues the mapping names that aren't among the required 16 — surfaced as a
    # crosswalk-model-confusion signal but deliberately NOT scored: completeness
    # stays pinned to the required issues (see the completeness test).
    unknown_issues = sorted(iss for iss in mapped if iss not in required)
    return {
        "missing_issues": missing_issues,
        "unknown_issues": unknown_issues,
        "dangling_topics": dangling,
        "n_pages_reconstructable": len(CROSSWALK_ISSUES) - len(missing_issues),
        "n_pages_total": len(CROSSWALK_ISSUES),
        "complete": not missing_issues and not dangling,
    }


def resolve_exemplars(taxonomy: dict, registry: dict[str, dict]) -> dict:
    """Replace each level-2 topic's exemplar sample-ids with real
    (doc_name, para_idx) pairs, dropping any id not in the sample registry.

    Mutates `taxonomy` in place (adds `exemplar_paragraphs` per level-2 topic)
    and returns a stats dict: how many topics ended with < 2 valid exemplars —
    the merge occasionally cites an id it half-remembered, and only real
    paragraph keys belong in a provenance artifact."""
    under = []
    for t in taxonomy.get("level2", []):
        resolved = []
        for sid in t.get("exemplars", []):
            r = registry.get(sid)
            if r is not None:
                resolved.append({"doc_name": r["doc_name"], "para_idx": r["para_idx"]})
        # De-dup while preserving order.
        seen = set()
        deduped = []
        for e in resolved:
            key = (e["doc_name"], e["para_idx"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        t["exemplar_paragraphs"] = deduped
        if len(deduped) < 2:
            under.append(t["name"])
    return {"topics_under_two_exemplars": under}


def bail_reasons(coverage: float, report: dict, xw: dict) -> list[str]:
    """The plan's hard bail conditions, as human-readable reasons — empty list ==
    all clear. `run()` calls this after the (optional) single merge revision and,
    when the list is non-empty, raises and writes NOTHING: a taxonomy that fails a
    gate must never overwrite the committed-good artifacts (nor let a failed re-run
    stamp a fresh manifest over them). Pure and API-free so it is unit-testable.

    Covers the plan's four named bail conditions — coverage below the gate after a
    revision, > LEVEL2_HARD_MAX level-2 topics, any legacy issue with no crosswalk
    target, orphaned level-2 parents — plus dangling crosswalk topics (a mapping
    pointing at a nonexistent level-2 topic breaks page reconstruction exactly as a
    missing issue does; it is the same crosswalk-integrity failure)."""
    reasons: list[str] = []
    if coverage < COVERAGE_GATE:
        reasons.append(
            f"held-out coverage {coverage * 100:.1f}% is below the "
            f"{COVERAGE_GATE * 100:.0f}% gate"
        )
    if report.get("over_hard_max"):
        reasons.append(
            f"{report.get('n_level2')} level-2 topics exceed the hard max "
            f"of {LEVEL2_HARD_MAX}"
        )
    if report.get("orphan_parents"):
        reasons.append(
            f"level-2 topics reference nonexistent level-1 parents: "
            f"{report['orphan_parents']}"
        )
    if xw.get("missing_issues"):
        reasons.append(
            f"legacy issues with no crosswalk target: {xw['missing_issues']}"
        )
    if xw.get("dangling_topics"):
        reasons.append(
            f"crosswalk references nonexistent level-2 topics: {xw['dangling_topics']}"
        )
    return reasons


# ==========================================================================
# JSON schemas for structured outputs
# ==========================================================================

_ERA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topics"],
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "definition", "exemplars"],
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "exemplars": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

_MERGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["level1", "level2"],
    "properties": {
        "level1": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "definition", "kind"],
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "kind": {"type": "string", "enum": ["policy", "non-policy"]},
                },
            },
        },
        "level2": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name", "definition", "level1", "era_note",
                    "era_of_birth", "era_of_death", "exemplars",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "level1": {"type": "string"},
                    "era_note": {"type": "string"},
                    "era_of_birth": {"type": "string"},
                    "era_of_death": {"type": "string"},
                    "exemplars": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

_CROSSWALK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mappings"],
    "properties": {
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["legacy_issue", "level2_topics"],
                "properties": {
                    "legacy_issue": {"type": "string"},
                    "level2_topics": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

_COVERAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "topics"],
                "properties": {
                    "id": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

_ANACHRONISM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["flagged"],
    "properties": {
        "flagged": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "reason"],
                "properties": {
                    "topic": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

# --------------------------------------------------------------------------
# prompts (versioned; hashed into the manifest)
# --------------------------------------------------------------------------

PROMPT_VERSION = "taxonomy/v1"

ERA_SYSTEM = (
    "You are a historian building a topic taxonomy of U.S. presidential rhetoric, "
    "working ONE ERA AT A TIME. You will see paragraphs drawn only from {label}, "
    "plus the word-clusters that statistically dominate this era's paragraphs. "
    "Propose 8-15 topics that, together, account for what these paragraphs are "
    "about.\n\n"
    "HARD RULES:\n"
    "- Derive topics from THESE paragraphs and clusters, not from a general theory "
    "of what presidents discuss. If a concern is specific to this era (e.g. "
    "tribal-treaty relations, coinage of silver, internal improvements, "
    "Reconstruction, polygamy in the Territories), name it specifically rather "
    "than folding it into a broad modern bucket.\n"
    "- Use ONLY concepts available in {label}. Never propose an anachronism "
    "(e.g. no 'healthcare policy', 'climate change', or 'cybersecurity' for the "
    "1800s). If the paragraphs are about a period-bound concern, say so plainly.\n"
    "- Include non-policy topics when the text warrants: ceremonial/commemorative "
    "speech, personal narrative, procedural or administrative housekeeping, and "
    "appeals to shared values.\n"
    "- For each topic give a short name, a one-to-two sentence definition, and 2-3 "
    "exemplar paragraph ids (the [pNNNN] tags) drawn ONLY from the paragraphs "
    "shown. Every exemplar id must be one you were shown."
)

MERGE_SYSTEM = (
    "You are unifying per-era topic proposals into a single two-level taxonomy of "
    "U.S. presidential rhetoric spanning 1789-2026. You are given, for each of "
    "nine 30-year eras, that era's proposed topics with definitions and exemplar "
    "paragraph ids.\n\n"
    "Produce:\n"
    "- LEVEL 1: ~12-18 broad domains. Most map cleanly onto conventional policy "
    "areas (economy, war & military, foreign policy, civil rights, money & "
    "banking, trade, agriculture, infrastructure, etc.). CRUCIALLY, include "
    "first-class NON-POLICY domains for speech that is not about policy at all: "
    "ceremonial/commemorative, personal narrative, procedural/administrative, and "
    "values appeal. Mark each domain kind='policy' or kind='non-policy'.\n"
    "- LEVEL 2: 35-50 specific topics, each assigned to exactly one level-1 parent "
    "(by its exact name). This is where period-bound concerns live as DISTINCT "
    "entries — Indian/tribal affairs, Reconstruction, coinage/free-silver, "
    "internal improvements, civil-service reform, polygamy in the Territories, "
    "prohibition, international arbitration, terrorism, etc. Collapse synonyms "
    "across eras into one topic; do NOT collapse genuinely distinct period-bound "
    "concerns into a broad bucket.\n\n"
    "For each level-2 topic: a name, a definition, its level-1 parent name, an "
    "era-of-relevance note, and — for CLEARLY period-bound topics only — an "
    "era_of_birth and era_of_death (e.g. birth '1860s', death '1890s'); leave both "
    "empty strings for topics that persist across the whole range. Carry forward "
    "2-3 exemplar paragraph ids from the era proposals; use ONLY ids that appear "
    "in the proposals you were given.\n\n"
    "Target 35-50 level-2 topics. Do not exceed 50."
)

CROSSWALK_SYSTEM = (
    "You are building a crosswalk from an existing 16-issue taxonomy to a new "
    "corpus-derived two-level taxonomy, so the existing per-issue website pages "
    "stay reconstructable. For EACH of the legacy issues listed, name every "
    "level-2 topic from the new taxonomy whose paragraphs belong on that issue's "
    "page. Every legacy issue must map to at least one level-2 topic. Use ONLY "
    "exact level-2 topic names from the taxonomy provided. A new topic may map to "
    "more than one legacy issue where the content genuinely overlaps."
)

COVERAGE_SYSTEM = (
    "You are labeling U.S. presidential-speech paragraphs against a FROZEN "
    "two-level topic taxonomy. For each paragraph, return the level-2 topic names "
    "(from the taxonomy below) that it is about — usually one or two, occasionally "
    "more. Use ONLY exact level-2 topic names from the taxonomy. Assign at least "
    "one topic to every paragraph that has any discernible subject; the non-policy "
    "domains (ceremonial, personal narrative, procedural/administrative, values "
    "appeal) exist precisely so that non-policy paragraphs still receive a label. "
    "Return an empty topic list only for a paragraph that is genuinely "
    "contentless.\n\nTAXONOMY (level-2 topics, each 'Name [Level-1] — definition'):\n"
    "{taxonomy}"
)

ANACHRONISM_SYSTEM = (
    "You are auditing a historical topic taxonomy for anachronism. For each "
    "level-2 topic you are given its name, definition, and era-of-relevance note. "
    "Flag ONLY topics whose definition relies on a concept that did not exist in "
    "the era the topic is assigned to (e.g. a topic scoped to the 1800s whose "
    "definition invokes healthcare policy, climate science, or nuclear weapons). "
    "Do not flag a topic merely for being broad or persistent. Return an empty "
    "list if nothing is anachronistic."
)


def _all_prompts() -> dict[str, str]:
    return {
        "era_system": ERA_SYSTEM,
        "merge_system": MERGE_SYSTEM,
        "crosswalk_system": CROSSWALK_SYSTEM,
        "coverage_system": COVERAGE_SYSTEM,
        "anachronism_system": ANACHRONISM_SYSTEM,
    }


def _prompt_hash() -> str:
    blob = json.dumps(_all_prompts(), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ==========================================================================
# API calls  (ISOLATED — every paid call lives behind one of these)
# ==========================================================================


def _client():
    import anthropic

    return anthropic.Anthropic()


def _extract_json(resp) -> dict:
    """Pull the guaranteed-valid JSON out of a structured-output response,
    surfacing a refusal or truncation rather than parsing garbage."""
    if resp.stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        raise RuntimeError(f"model refused: {details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("response hit max_tokens; raise the limit or stream")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        # No text block at all (e.g. an unexpected stop_reason not handled above,
        # or a thinking-only response). Bare next() would raise StopIteration deep
        # in the SDK path; name what actually happened instead.
        raise RuntimeError(
            f"response carried no text block to parse (stop_reason={resp.stop_reason})"
        )
    return json.loads(text)


def propose_era_topics(
    client, label: str, rows: list[dict], clusters: list[dict], spend: Spend
) -> dict:
    """One era's proposal. Sees ONLY this era's paragraphs + dominant clusters —
    no cross-era context (the anachronism guard)."""
    cluster_lines = "\n".join(
        f"- cluster {c['cluster']} (n={c['size']}): {', '.join(c['terms'])}"
        for c in clusters
    )
    user = (
        f"ERA: {label}\n\n"
        f"Word-clusters that dominate this era's paragraphs:\n{cluster_lines}\n\n"
        f"PARAGRAPHS:\n{format_paragraphs(rows)}"
    )
    resp = client.messages.create(
        model=PROPOSAL_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=ERA_SYSTEM.format(label=label),
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _ERA_SCHEMA}},
    )
    spend.add(PROPOSAL_MODEL, resp.usage)
    return _extract_json(resp)


def merge_proposals(client, proposals: dict[str, dict], spend: Spend) -> dict:
    """Unify the nine era proposals into the two-level taxonomy. Streamed —
    a 35-50 topic taxonomy with definitions can be a large output."""
    blocks = []
    for label, prop in proposals.items():
        topic_lines = "\n".join(
            f"  - {t['name']}: {t['definition']} [exemplars: {', '.join(t['exemplars'])}]"
            for t in prop["topics"]
        )
        blocks.append(f"ERA {label}:\n{topic_lines}")
    user = "PER-ERA PROPOSALS:\n\n" + "\n\n".join(blocks)
    with client.messages.stream(
        model=PROPOSAL_MODEL,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        system=MERGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _MERGE_SCHEMA}},
    ) as stream:
        resp = stream.get_final_message()
    spend.add(PROPOSAL_MODEL, resp.usage)
    return _extract_json(resp)


def build_crosswalk(client, taxonomy: dict, spend: Spend) -> dict:
    """Map every legacy issue to >= 1 level-2 topic."""
    l2 = "\n".join(
        f"- {t['name']} [{t['level1']}] — {t['definition']}"
        for t in taxonomy["level2"]
    )
    user = (
        "LEGACY ISSUES TO MAP (each must map to >= 1 new topic):\n"
        + "\n".join(f"- {iss}" for iss in CROSSWALK_ISSUES)
        + f"\n\nNEW TAXONOMY (level-2 topics):\n{l2}"
    )
    resp = client.messages.create(
        model=PROPOSAL_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=CROSSWALK_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _CROSSWALK_SCHEMA}},
    )
    spend.add(PROPOSAL_MODEL, resp.usage)
    return _extract_json(resp)


def label_coverage(
    client, rows: list[dict], taxonomy: dict, spend: Spend
) -> dict[str, list[str]]:
    """Label held-out paragraphs against the frozen taxonomy, batched.

    Uses Sonnet 5 with thinking EXPLICITLY disabled — Sonnet 5 runs adaptive
    thinking by default, silently billing thinking tokens and risking truncation
    on bulk classification. The taxonomy prefix is cached across batches."""
    taxonomy_text = "\n".join(
        f"- {t['name']} [{t['level1']}] — {t['definition']}"
        for t in taxonomy["level2"]
    )
    system = [{
        "type": "text",
        "text": COVERAGE_SYSTEM.format(taxonomy=taxonomy_text),
        "cache_control": {"type": "ephemeral"},
    }]
    labels: dict[str, list[str]] = {}
    for start in range(0, len(rows), COVERAGE_BATCH):
        batch = rows[start:start + COVERAGE_BATCH]
        user = "PARAGRAPHS TO LABEL:\n\n" + format_paragraphs(batch)
        resp = client.messages.create(
            model=COVERAGE_MODEL,
            max_tokens=4000,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _COVERAGE_SCHEMA}},
        )
        spend.add(COVERAGE_MODEL, resp.usage)
        for item in _extract_json(resp)["labels"]:
            labels[item["id"]] = item["topics"]
    return labels


def anachronism_check(client, taxonomy: dict, spend: Spend) -> list[dict]:
    """Second-model spot-check for anachronistic level-2 definitions."""
    listing = "\n".join(
        f"- {t['name']} (era: {t['era_note']}): {t['definition']}"
        for t in taxonomy["level2"]
    )
    resp = client.messages.create(
        model=PROPOSAL_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=ANACHRONISM_SYSTEM,
        messages=[{"role": "user", "content": "LEVEL-2 TOPICS:\n" + listing}],
        output_config={"format": {"type": "json_schema", "schema": _ANACHRONISM_SCHEMA}},
    )
    spend.add(PROPOSAL_MODEL, resp.usage)
    return _extract_json(resp)["flagged"]


# ==========================================================================
# orchestration
# ==========================================================================


def _era_sample_ids(era_samples: dict[int, pd.DataFrame]) -> dict[str, list[dict]]:
    """Per-era doc_name/para_idx lists for the provenance record."""
    return {
        str(int(era)): [
            {"doc_name": r["doc_name"], "para_idx": int(r["para_idx"])}
            for _, r in df.iterrows()
        ]
        for era, df in era_samples.items()
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write pretty JSON to `path` via a sibling tmp file + os.replace, so a crash
    mid-write can never leave a truncated file (os.replace is atomic within a
    filesystem). Output matches the direct write it replaced: indent=2, trailing
    newline."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def run(
    run_id: str | None = None,
    per_era: int = PER_ERA_SAMPLE,
    seed: int = RANDOM_STATE,
    cost_ceiling: float = 12.0,
    dry_run: bool = False,
) -> dict:
    """Derive the taxonomy end-to-end and write the three artifacts.

    Returns a summary dict. Two guards: a COST guard — before each stage, if the
    cumulative ACTUAL spend already exceeds `cost_ceiling`, `_guard` raises rather
    than starting the stage (it checks realized spend, not a projection) — and a
    BAIL gate: after the optional single merge revision, `bail_reasons` is checked
    and, if any condition fails, `run` raises and writes NOTHING (see
    `bail_reasons`), so a failed run can never overwrite the committed artifacts."""
    run_id = run_id or f"taxonomy-v1-{date.today():%Y%m%d}"
    paras, issues, clusters, cmeta = load_frames()
    df = attach_metadata(paras, issues, clusters)

    heldout = draw_heldout(df, seed)
    train = df.drop(heldout.index)
    era_samples = draw_era_samples(train, per_era, seed)
    registry = register_samples(era_samples)

    n_train = sum(len(s) for s in era_samples.values())
    print(f"[sample] {n_train} training paragraphs across {len(era_samples)} eras; "
          f"{len(heldout)} held-out ({int(heldout['unlabeled'].sum())} unlabeled)")

    if dry_run:
        return {
            "run_id": run_id, "dry_run": True, "n_train": n_train,
            "n_heldout": len(heldout),
            "eras": {str(int(e)): len(s) for e, s in era_samples.items()},
        }

    client = _client()
    spend = Spend()

    def _guard(stage: str) -> None:
        if spend.cost_usd > cost_ceiling:
            raise RuntimeError(
                f"COST GUARD: ${spend.cost_usd:.2f} exceeds ${cost_ceiling:.2f} "
                f"before {stage}; stopping."
            )

    # ---- per-era proposals (no cross-era context) ----
    proposals: dict[str, dict] = {}
    for era in sorted(era_samples):
        _guard(f"era {era} proposal")
        edf = era_samples[era]
        label = era_label(edf)
        rows = [r for r in registry.values() if r["era"] == era]
        clusters_for_era = dominant_cluster_terms(df, era, cmeta)
        proposals[label] = propose_era_topics(client, label, rows, clusters_for_era, spend)
        print(f"[propose] {label}: {len(proposals[label]['topics'])} topics "
              f"(${spend.cost_usd:.2f} cumulative)")

    # ---- merge ----
    _guard("merge")
    taxonomy = merge_proposals(client, proposals, spend)
    report = validate_structure(taxonomy)
    print(f"[merge] level1={report['n_level1']} level2={report['n_level2']} "
          f"non_policy={report['n_non_policy']} (${spend.cost_usd:.2f})")

    # ---- crosswalk ----
    _guard("crosswalk")
    crosswalk = build_crosswalk(client, taxonomy, spend)
    xw = check_crosswalk(crosswalk, taxonomy)
    print(f"[crosswalk] {xw['n_pages_reconstructable']}/{xw['n_pages_total']} pages; "
          f"missing={xw['missing_issues']} dangling={xw['dangling_topics']}")

    # ---- coverage gate (held-out 200), with one merge revision if it fails ----
    _guard("coverage")
    heldout_rows = _heldout_rows(heldout)
    heldout_ids = [r["sample_id"] for r in heldout_rows]
    labels = label_coverage(client, heldout_rows, taxonomy, spend)
    l2_names = {t["name"] for t in taxonomy["level2"]}
    coverage = compute_coverage(labels, heldout_ids, l2_names)
    print(f"[coverage] {coverage*100:.1f}% (${spend.cost_usd:.2f})")

    revised = False
    if coverage < COVERAGE_GATE and spend.cost_usd < cost_ceiling:
        print(f"[coverage] below {COVERAGE_GATE*100:.0f}%; revising merge once")
        # Feed revise_merge exactly the paragraphs the gate scored as uncovered,
        # via the SAME strict predicate — so a near-miss-name shortfall can't
        # trigger a revision whose failing paragraphs are absent from its input.
        missing = set(uncovered_ids(labels, heldout_ids, l2_names))
        uncovered = [r for r in heldout_rows if r["sample_id"] in missing]
        taxonomy = revise_merge(client, taxonomy, uncovered, spend)
        report = validate_structure(taxonomy)
        crosswalk = build_crosswalk(client, taxonomy, spend)
        xw = check_crosswalk(crosswalk, taxonomy)
        labels = label_coverage(client, heldout_rows, taxonomy, spend)
        l2_names = {t["name"] for t in taxonomy["level2"]}
        coverage = compute_coverage(labels, heldout_ids, l2_names)
        revised = True
        print(f"[coverage] after revision: {coverage*100:.1f}% (${spend.cost_usd:.2f})")

    # ---- ENFORCE bail conditions before spending more or writing anything ----
    # Placed before the anachronism call and the artifact writes so a doomed run
    # neither pays for the extra check nor overwrites the committed-good files.
    reasons = bail_reasons(coverage, report, xw)
    if reasons:
        raise RuntimeError(
            "taxonomy failed bail conditions after "
            + ("one merge revision" if revised else "the initial merge")
            + "; writing nothing:\n  - "
            + "\n  - ".join(reasons)
        )

    # ---- resolve exemplars & verification checks ----
    exemplar_stats = resolve_exemplars(taxonomy, registry)
    anachronisms = anachronism_check(client, taxonomy, spend) if spend.cost_usd < cost_ceiling else []
    print(f"[verify] anachronisms flagged: {len(anachronisms)}; "
          f"topics<2 exemplars: {len(exemplar_stats['topics_under_two_exemplars'])}")

    # ---- assemble provenance & write artifacts ----
    fingerprint = corpus_fingerprint()
    provenance = {
        "run_id": run_id,
        "date": f"{date.today():%Y-%m-%d}",
        "proposal_model": PROPOSAL_MODEL,
        "coverage_model": COVERAGE_MODEL,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _prompt_hash(),
        "prompts": _all_prompts(),
        "seed": seed,
        "per_era": per_era,
        "corpus_fingerprint": fingerprint,
        "era_sample_ids": _era_sample_ids(era_samples),
        "heldout_sample_ids": [
            {"doc_name": r["doc_name"], "para_idx": r["para_idx"],
             "sample_id": r["sample_id"], "unlabeled": r["unlabeled"]}
            for r in heldout_rows
        ],
        "dominant_clusters_by_era": {
            era_label(era_samples[e]): dominant_cluster_terms(df, e, cmeta)
            for e in sorted(era_samples)
        },
        "coverage": coverage,
        "unknown_coverage_labels": unknown_labels(
            labels, {t["name"] for t in taxonomy["level2"]}
        ),
        "merge_revised": revised,
        "structure": report,
        "crosswalk_check": xw,
        "anachronisms_flagged": anachronisms,
        "exemplar_stats": exemplar_stats,
        "cost_usd": round(spend.cost_usd, 4),
    }
    taxonomy_out = dict(taxonomy)
    taxonomy_out["provenance"] = provenance
    crosswalk_out = dict(crosswalk)
    crosswalk_out["provenance"] = {
        "run_id": run_id, "model": PROPOSAL_MODEL,
        "check": xw, "date": provenance["date"],
    }

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    # tmp+rename gives PER-FILE atomicity: neither file is ever observed
    # half-written (os.replace is atomic within a filesystem). The two writes are
    # sequential, NOT one transaction — a crash between them can leave a new
    # taxonomy beside an old crosswalk; that mismatch is acceptable because the run
    # is idempotent (a re-run regenerates both from the same seed/prompts).
    _atomic_write_json(TAXONOMY_PATH, taxonomy_out)
    _atomic_write_json(CROSSWALK_PATH, crosswalk_out)

    # write_manifest lives in the shared llm_annotations module (it does its own
    # file write); left as-is rather than reaching across the module boundary.
    write_manifest(Manifest(
        run_id=run_id,
        model=f"{PROPOSAL_MODEL} (+ {COVERAGE_MODEL} for coverage)",
        prompt_version=PROMPT_VERSION,
        prompt_hash=_prompt_hash(),
        date=provenance["date"],
        batch_id=None,
        n_requests=spend.n_requests,
        input_tokens=spend.input_tokens,
        output_tokens=spend.output_tokens,
        cache_creation_input_tokens=spend.cache_creation_input_tokens,
        cache_read_input_tokens=spend.cache_read_input_tokens,
        cost_usd=round(spend.cost_usd, 4),
        corpus_fingerprint=fingerprint,
        fields=["level1", "level2", "crosswalk"],
        annotation_files=[TAXONOMY_PATH.name, CROSSWALK_PATH.name],
        notes=(
            f"Corpus-derived two-level taxonomy. {report['n_level2']} level-2 "
            f"topics across {report['n_level1']} level-1 domains "
            f"({report['n_non_policy']} non-policy). Held-out coverage "
            f"{coverage*100:.1f}% on {len(heldout_rows)} paragraphs "
            f"({int(heldout['unlabeled'].sum())} from the unlabeled mass). "
            f"Merge revised once: {revised}. Per-era proposals ran with NO "
            f"cross-era context (anachronism guard); merge/crosswalk/checks on "
            f"{PROPOSAL_MODEL}, held-out labeling on {COVERAGE_MODEL}."
        ),
    ))

    return {
        "run_id": run_id,
        "coverage": coverage,
        "n_level1": report["n_level1"],
        "n_level2": report["n_level2"],
        "n_non_policy": report["n_non_policy"],
        "crosswalk_complete": xw["complete"],
        "missing_issues": xw["missing_issues"],
        "anachronisms": anachronisms,
        "cost_usd": round(spend.cost_usd, 4),
        "merge_revised": revised,
        "structure": report,
    }


def _heldout_rows(heldout: pd.DataFrame) -> list[dict]:
    """Give held-out paragraphs stable ids (h0001, ...) for the coverage call."""
    rows = []
    for i, (_, r) in enumerate(heldout.iterrows(), start=1):
        rows.append({
            "sample_id": f"h{i:04d}",
            "doc_name": r["doc_name"],
            "para_idx": int(r["para_idx"]),
            "year": int(r["year"]),
            "unlabeled": bool(r["unlabeled"]),
            "text": r["text"],
        })
    return rows


REVISE_SYSTEM = (
    "You are revising a two-level topic taxonomy that failed a coverage gate: some "
    "held-out paragraphs received NO level-2 label. You are given the current "
    "taxonomy (each topic already carries its exemplar ids) and the paragraphs it "
    "failed to cover. Revise the taxonomy so those paragraphs would receive a "
    "label — usually by adding one or two missing level-2 topics, broadening a "
    "definition, or adding a non-policy domain — WITHOUT removing existing topics "
    "or exceeding 50 level-2 topics. Keep the same output shape (level1 + level2 "
    "with all fields). Preserve each existing topic's exemplar ids unchanged, and "
    "for any topic you add, reuse exemplar ids that already appear on the existing "
    "topics. Do NOT invent new ids, and do NOT use the [hNNNN] tags of the "
    "uncovered paragraphs shown below — those are held-out ids, not valid "
    "exemplars, and are dropped when exemplars are resolved to real paragraphs."
)


def revise_merge(
    client, taxonomy: dict, uncovered: list[dict], spend: Spend
) -> dict:
    """One merge revision after a coverage miss (the plan's single iteration).

    The current taxonomy — passed in full, including each level-2 topic's existing
    exemplar ids — is the only source of carry-forward exemplars; the era proposals
    are not re-shown, because the exemplar ids already on the taxonomy are real
    (registry-resolvable) ids and re-feeding the proposals would add nothing the
    revision can ground a NEW topic in more faithfully. The uncovered paragraphs
    carry held-out [hNNNN] ids, which REVISE_SYSTEM forbids as exemplars (they are
    not in the sample registry and resolve_exemplars would drop them)."""
    current = json.dumps(
        {"level1": taxonomy["level1"], "level2": taxonomy["level2"]}, indent=1
    )
    user = (
        f"CURRENT TAXONOMY:\n{current}\n\n"
        f"PARAGRAPHS THAT RECEIVED NO LABEL:\n{format_paragraphs(uncovered)}"
    )
    with client.messages.stream(
        model=PROPOSAL_MODEL,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        system=REVISE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _MERGE_SCHEMA}},
    ) as stream:
        resp = stream.get_final_message()
    spend.add(PROPOSAL_MODEL, resp.usage)
    return _extract_json(resp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive the corpus taxonomy (spends money).")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--per-era", type=int, default=PER_ERA_SAMPLE)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--cost-ceiling", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="sample + report only; no API calls, no spend")
    args = ap.parse_args()
    summary = run(
        run_id=args.run_id, per_era=args.per_era, seed=args.seed,
        cost_ceiling=args.cost_ceiling, dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
