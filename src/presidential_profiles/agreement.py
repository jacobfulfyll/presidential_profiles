"""Inter-model annotation agreement: draw a persisted era-stratified sample,
and measure where the primary Sonnet 5 annotations and a second-opinion Opus 4.8
pass agree — field by field and era by era.

The falsification value lives here. The two models re-annotate the SAME sample
with BYTE-IDENTICAL prompts (the model is the only variable), so a per-field,
per-era agreement table exposes exactly the failure modes the annotation design
worried about: if the models agree on 1990s speeches but diverge on 1830s ones,
that is the anachronism risk made visible rather than assumed away. Nothing gates
on these numbers — disagreement is published, not hidden.

Three entry points:

* ``draw_sample()``  — proportional 25% per 30-year bin, min 3 per bin, fixed
  seed; writes ``data/llm_annotations/agreement_sample_v1.json`` ONCE. From then
  on the committed FILE (not the seed) is the source of truth: re-running
  selection cannot silently shift membership, and a future hand-graded truth run
  annotates exactly this list.
* ``build_agreement()`` — joins the primary annotation tables (ALL rows — the
  primary side spans the convergence ladder's several run_ids) against the Opus
  second-opinion tables, restricted to the sample, and writes the field x era-bin
  x metric x value x n table ``agreement_v1.parquet``.
* ``agreement_report()`` — renders ``notes/agreement-report-v1.md``: overall +
  per-era tables per field (n for every bin, nothing suppressed), kappa < 0.4
  flagged low-confidence (flagged, NEVER gated), an entity section, and the 10
  highest-disagreement paragraphs quoted from the corpus.

The Opus tables are the per-model-suffixed parquets that ``pp-annotate submit
--model claude-opus-4-8`` writes (see ``annotate._table_name``); ``build_*`` and
``agreement_report`` fail with a clear error until the paid Opus pass has run.

The metric computations are pure functions over DataFrames/sequences
(``compute_kappa``, ``compute_jaccard``, ``compute_exact_match``,
``compute_entity_metrics``, ``build_metrics_table``) so they are testable on tiny
synthetic frames; the file-loading wrappers are thin.

Run as:  PYTHONPATH=src arch -x86_64 .venv/bin/python -m presidential_profiles.agreement <cmd>
(the repo venv is x86_64 under Rosetta).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import annotate, llm_annotations as ann
from .corpus import REPO_ROOT, load

# --------------------------------------------------------------------------
# era binning — 30-year bins anchored at 1789 (the same binning draw_sample and
# build_agreement share; 8 bins over 1789-2026)
# --------------------------------------------------------------------------

ERA_ANCHOR = 1789
ERA_BIN_WIDTH = 30
OVERALL_BIN = -1  # sentinel era_bin for the corpus-wide (non-stratified) rows


def era_bin(year: int) -> int:
    """The 0-based 30-year era bin a year falls in (bin 0 = 1789-1818)."""
    return (int(year) - ERA_ANCHOR) // ERA_BIN_WIDTH


def era_label(b: int) -> str:
    """Human-readable label for an era bin; ``OVERALL_BIN`` -> 'overall'."""
    if int(b) == OVERALL_BIN:
        return "overall"
    start = ERA_ANCHOR + ERA_BIN_WIDTH * int(b)
    return f"{start}-{start + ERA_BIN_WIDTH - 1}"


# --------------------------------------------------------------------------
# sample draw
# --------------------------------------------------------------------------

SAMPLE_PATH = ann.ANNOTATIONS_DIR / "agreement_sample_v1.json"
SAMPLE_SEED = 20260721
SAMPLE_FRACTION = 0.25
MIN_PER_BIN = 3

_SELECTION_RULE = (
    "Speech-level draw from corpus.load()'s unique doc_names. 30-year era bins "
    "anchored at 1789 (bin 0 = 1789-1818). Proportional {fraction:.0%} per bin, "
    "rounded half up (int(fraction*n + 0.5)) and clamped to [{min_per_bin}, "
    "bin_size]. Within each bin the doc_names are SORTED before a seeded "
    "numpy.random.default_rng({seed}) draw, so the selection reproduces "
    "bit-for-bit. The committed JSON — not the seed — is the source of truth for "
    "every downstream run (including a future hand-graded truth pass)."
)


def draw_sample(
    *,
    path: Path = SAMPLE_PATH,
    seed: int = SAMPLE_SEED,
    fraction: float = SAMPLE_FRACTION,
    min_per_bin: int = MIN_PER_BIN,
    draw_date: str | None = None,
    force: bool = False,
    speeches: pd.DataFrame | None = None,
) -> dict:
    """Draw the persisted era-stratified sample and write it to ``path``.

    Refuses to overwrite an existing file unless ``force=True`` — the committed
    file is the source of truth, so a redraw must be a deliberate act (it
    invalidates any Opus run keyed to the old membership). ``draw_date`` is
    recorded in the JSON; if None, today's date is used."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists and is the persisted source of truth for the "
            "agreement sample. Pass force=True to deliberately redraw (this "
            "invalidates any Opus pass keyed to the current membership)."
        )
    if speeches is None:
        speeches = load()
    if draw_date is None:
        draw_date = date.today().isoformat()

    df = speeches[["doc_name", "year"]].drop_duplicates("doc_name").copy()
    df["era_bin"] = df["year"].map(era_bin)
    year_of = dict(zip(df["doc_name"], df["year"]))

    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    bin_summ: list[dict] = []
    for b in sorted(df["era_bin"].unique()):
        docs = sorted(df[df["era_bin"] == b]["doc_name"])  # sort -> reproducible
        n_in = len(docs)
        k = min(n_in, max(min_per_bin, int(fraction * n_in + 0.5)))  # round half up
        picks = np.sort(rng.choice(n_in, size=k, replace=False))
        for i in picks:
            d = docs[int(i)]
            chosen.append({"doc_name": d, "era_bin": int(b),
                           "era_label": era_label(b), "year": int(year_of[d])})
        bin_summ.append({"era_bin": int(b), "era_label": era_label(b),
                         "n_in_bin": n_in, "n_sampled": int(k)})

    doc_names = sorted(r["doc_name"] for r in chosen)
    out = {
        "version": "v1",
        "seed": seed,
        "fraction": fraction,
        "min_per_bin": min_per_bin,
        "era_anchor": ERA_ANCHOR,
        "era_bin_width": ERA_BIN_WIDTH,
        "selection_rule": _SELECTION_RULE.format(
            fraction=fraction, min_per_bin=min_per_bin, seed=seed),
        "draw_date": draw_date,
        "n_corpus_speeches": int(df["doc_name"].nunique()),
        "n_sampled": len(doc_names),
        "corpus_fingerprint": ann.corpus_fingerprint(speeches=speeches),
        "bins": bin_summ,
        "doc_names": doc_names,
        "speeches": sorted(chosen, key=lambda r: (r["era_bin"], r["doc_name"])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def load_sample(path: Path = SAMPLE_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"agreement sample not found: {path}. Run `draw-sample` first "
            "(it writes the persisted, committed sample)."
        )
    return json.loads(p.read_text())


def sample_doc_names(path: Path = SAMPLE_PATH) -> list[str]:
    data = load_sample(path)
    docs = data.get("doc_names") or [s["doc_name"] for s in data.get("speeches", [])]
    if not docs:
        raise ValueError(f"{path} lists no doc_names")
    return sorted(docs)


# --------------------------------------------------------------------------
# pure metric helpers (testable on tiny synthetic sequences / frames)
# --------------------------------------------------------------------------

KAPPA_LOW_THRESHOLD = 0.4  # kappa below this is FLAGGED low-confidence (never gated)

FLAG_FIELDS = ["party_attack", "enemy_naming", "zero_sum"]  # -> cohen_kappa
FACTUAL_FIELDS = ["speech_type", "audience", "medium"]      # -> cohen_kappa + exact_match


def _is_null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def compute_kappa(a: Sequence, b: Sequence) -> tuple[float, int]:
    """Cohen's kappa between two aligned label sequences, plus n.

    The degenerate single-class case — every observation carries the same label
    across BOTH raters, so kappa is 0/0 undefined — returns NaN (with n), rather
    than crashing or letting sklearn emit a warning-laden value.

    A pair with a null (None/NaN) on either side is a MISSING observation:
    excluded pairwise, with n counting only the compared pairs. sklearn raises
    on mixed None/bool labels, and one bad row in an ingested run must never
    crash the whole agreement build."""
    pairs = [(x, y) for x, y in zip(a, b) if not (_is_null(x) or _is_null(y))]
    n = len(pairs)
    if n == 0:
        return math.nan, 0
    a2 = [p[0] for p in pairs]
    b2 = [p[1] for p in pairs]
    categories = list(dict.fromkeys([*a2, *b2]))
    lookup = {value: index for index, value in enumerate(categories)}
    contingency = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for left, right in zip(a2, b2, strict=True):
        contingency[lookup[left], lookup[right]] += 1
    return compute_contingency_metric(contingency, "cohen_kappa"), n


def compute_contingency_metric(counts: np.ndarray, metric: str) -> float:
    """Compute an agreement statistic from a square count matrix.

    This is the shared count-level core used by ``compute_kappa`` and by
    speech-cluster bootstrap projections. Keeping it here prevents public-page
    audit code from maintaining a second agreement implementation.
    """
    matrix = np.asarray(counts, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("agreement contingency matrix must be square")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("agreement contingency counts must be finite and nonnegative")
    total = float(matrix.sum())
    if total <= 0:
        return math.nan
    observed = float(np.trace(matrix) / total)
    if metric == "exact_match":
        return observed
    if metric != "cohen_kappa":
        raise ValueError(f"unsupported contingency agreement metric {metric!r}")
    expected = float((matrix.sum(axis=1) @ matrix.sum(axis=0)) / (total * total))
    if math.isclose(expected, 1.0):
        return math.nan
    return (observed - expected) / (1.0 - expected)


def jaccard(a: Iterable, b: Iterable) -> float:
    """Jaccard overlap of two label sets. BOTH EMPTY == 1.0 by convention: two
    models that both assign no topic to a paragraph agree perfectly. A null
    cell (None / float NaN, as a parquet null loads) counts as an empty set —
    an ingested run must never crash the whole agreement build over one row."""
    sa, sb = _as_label_set(a), _as_label_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _as_label_set(labels: Iterable) -> set:
    return set() if _is_null(labels) else set(labels)


def _fmt_flag(v) -> str:
    """Render a boolean flag cell for the report; a null renders as 'null'
    rather than the misleading True that bool(nan) would produce."""
    return "null" if _is_null(v) else str(bool(v))


def compute_jaccard(a_iter: Sequence[Iterable], b_iter: Sequence[Iterable]) -> tuple[float, int]:
    """Mean per-item Jaccard over aligned sequences of label sets, plus n."""
    vals = [jaccard(a, b) for a, b in zip(a_iter, b_iter)]
    n = len(vals)
    return (float(np.mean(vals)) if n else math.nan, n)


def compute_exact_match(a: Sequence, b: Sequence) -> tuple[float, int]:
    """Exact-match rate between two aligned label sequences, plus n."""
    pairs = list(zip(list(a), list(b)))
    n = len(pairs)
    if n == 0:
        return math.nan, 0
    return sum(1 for x, y in pairs if x == y) / n, n


def normalize_entity_name(name) -> str:
    """Case- and whitespace-normalized entity name used for cross-model matching:
    lowercased, surrounding whitespace stripped, internal whitespace runs collapsed
    to a single space. Deliberately conservative — no stemming, no article
    stripping — so 'The  Governor' matches 'the Governor' but 'Governor' does not
    match 'the Governor'. Granularity/wording differences therefore surface as
    unmatched entities (reported), not as silent stance disagreements."""
    return " ".join(str(name).split()).lower()


def _entity_stance_map(ent: pd.DataFrame) -> dict[tuple, str]:
    """(doc_name, para_idx, normalized_name) -> stance for one model's entity rows.
    On a repeated normalized key within a paragraph, the first row after a
    deterministic sort wins (documented tie-break; rare in practice)."""
    if ent is None or len(ent) == 0:
        return {}
    df = ent.copy()
    df["norm"] = df["entity"].map(normalize_entity_name)
    df = df.sort_values(["doc_name", "para_idx", "norm", "stance"])
    out: dict[tuple, str] = {}
    for r in df.itertuples(index=False):
        key = (r.doc_name, r.para_idx, r.norm)
        if key not in out:
            out[key] = r.stance
    return out


def entity_agreement_details(
    primary_ent: pd.DataFrame, opus_ent: pd.DataFrame
) -> pd.DataFrame:
    """One canonical row per normalized entity key in either model pass."""
    primary = _entity_stance_map(primary_ent)
    second = _entity_stance_map(opus_ent)
    rows = []
    for doc_name, para_idx, normalized_name in sorted(set(primary) | set(second)):
        key = (doc_name, para_idx, normalized_name)
        primary_present = key in primary
        second_present = key in second
        rows.append(
            {
                "doc_name": doc_name,
                "para_idx": int(para_idx),
                "entity_normalized": normalized_name,
                "primary_present": primary_present,
                "second_present": second_present,
                "stance_primary": primary.get(key),
                "stance_second": second.get(key),
                "stance_agrees": bool(
                    primary_present
                    and second_present
                    and primary[key] == second[key]
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "doc_name", "para_idx", "entity_normalized", "primary_present",
            "second_present", "stance_primary", "stance_second", "stance_agrees",
        ],
    )


def compute_entity_metrics(primary_ent: pd.DataFrame, opus_ent: pd.DataFrame) -> dict:
    """Entity agreement between the two models' entity rows.

    Reports NAME-MATCH separately from STANCE agreement, plus per-side
    unmatched-entity rates — because the two models may differ on entity
    granularity (one splits 'the Republicans in Congress' into two), and that is a
    different phenomenon from disagreeing on an entity's stance. Entities are keyed
    on (doc_name, para_idx, normalized_name):

      * name_match_rate  = |matched| / |union|   (both-sides-empty -> 1.0)
      * stance_agreement = share of MATCHED entities with the same stance
      * unmatched_primary_rate / unmatched_opus_rate = each side's entities the
        other side did not name, over that side's own entity count
    """
    detail = entity_agreement_details(primary_ent, opus_ent)
    primary_present = detail["primary_present"] if len(detail) else pd.Series(dtype=bool)
    second_present = detail["second_present"] if len(detail) else pd.Series(dtype=bool)
    matched = primary_present & second_present
    n_union = int(len(detail))
    n_matched = int(matched.sum())
    n_primary = int(primary_present.sum())
    n_opus = int(second_present.sum())
    return {
        "name_match_rate": (n_matched / n_union) if n_union else 1.0,
        "n_union": n_union,
        "stance_agreement": (
            float(detail.loc[matched, "stance_agrees"].mean())
            if n_matched else math.nan
        ),
        "n_matched": n_matched,
        "unmatched_primary_rate": (
            int((primary_present & ~second_present).sum()) / n_primary
            if n_primary else math.nan
        ),
        "n_primary": n_primary,
        "unmatched_opus_rate": (
            int((second_present & ~primary_present).sum()) / n_opus
            if n_opus else math.nan
        ),
        "n_opus": n_opus,
    }


def _metric_rows(era_val: int, para: pd.DataFrame, primary_ent: pd.DataFrame,
                 opus_ent: pd.DataFrame, speech: pd.DataFrame | None) -> list[dict]:
    """All (field, metric, value, n) rows for one scope (an era bin, or overall).
    Pure: everything is read off the passed frames' ``*_primary`` / ``*_opus``
    columns."""
    label = era_label(era_val)
    rows: list[dict] = []

    def add(field: str, metric: str, value: float, n: int) -> None:
        rows.append({"field": field, "era_bin": int(era_val), "era_label": label,
                     "metric": metric, "value": float(value), "n": int(n)})

    for f in FLAG_FIELDS:
        add(f, "cohen_kappa", *compute_kappa(para[f + "_primary"], para[f + "_opus"]))
    add("topics", "jaccard", *compute_jaccard(para["topics_primary"], para["topics_opus"]))
    add("proposal_values", "exact_match",
        *compute_exact_match(para["proposal_values_primary"], para["proposal_values_opus"]))

    if speech is not None and len(speech):
        for f in FACTUAL_FIELDS:
            add(f, "cohen_kappa", *compute_kappa(speech[f + "_primary"], speech[f + "_opus"]))
            add(f, "exact_match", *compute_exact_match(speech[f + "_primary"], speech[f + "_opus"]))

    em = compute_entity_metrics(primary_ent, opus_ent)
    add("entities", "name_match_rate", em["name_match_rate"], em["n_union"])
    add("entities", "stance_agreement", em["stance_agreement"], em["n_matched"])
    add("entities", "unmatched_primary_rate", em["unmatched_primary_rate"], em["n_primary"])
    add("entities", "unmatched_opus_rate", em["unmatched_opus_rate"], em["n_opus"])
    return rows


def build_metrics_table(merged_para: pd.DataFrame, primary_ent: pd.DataFrame,
                        opus_ent: pd.DataFrame,
                        merged_speech: pd.DataFrame | None = None) -> pd.DataFrame:
    """The field x era-bin x metric x value x n agreement table.

    Pure over the merged frames: ``merged_para`` and ``merged_speech`` carry an
    ``era_bin`` column plus each field as ``<field>_primary`` / ``<field>_opus``;
    the entity frames carry an ``era_bin`` column. Every metric is computed once
    overall (era_bin = OVERALL_BIN) and once per era bin present in
    ``merged_para``."""
    rows = _metric_rows(OVERALL_BIN, merged_para, primary_ent, opus_ent, merged_speech)
    for b in sorted(merged_para["era_bin"].unique()):
        para_b = merged_para[merged_para["era_bin"] == b]
        pe_b = primary_ent[primary_ent["era_bin"] == b] if len(primary_ent) else primary_ent
        oe_b = opus_ent[opus_ent["era_bin"] == b] if len(opus_ent) else opus_ent
        sp_b = (merged_speech[merged_speech["era_bin"] == b]
                if merged_speech is not None else None)
        rows += _metric_rows(int(b), para_b, pe_b, oe_b, sp_b)
    return pd.DataFrame(rows, columns=["field", "era_bin", "era_label", "metric", "value", "n"])


# --------------------------------------------------------------------------
# disagreement scoring (for the qualitative section)
# --------------------------------------------------------------------------


def _para_entity_names(ent: pd.DataFrame) -> dict[tuple, set]:
    out: dict[tuple, set] = {}
    if ent is not None and len(ent):
        for r in ent.itertuples(index=False):
            out.setdefault((r.doc_name, r.para_idx), set()).add(normalize_entity_name(r.entity))
    return out


def paragraph_disagreement(merged_para: pd.DataFrame, primary_ent: pd.DataFrame,
                           opus_ent: pd.DataFrame) -> pd.DataFrame:
    """Per-paragraph disagreement score (0 = identical, higher = more divergent).

    A deliberately simple sum across the judgment fields, each contributing at most
    1 per flag / 1 total for the rest, so no single field dominates:
      * flags: +1 for EACH of party_attack / enemy_naming / zero_sum that differ (0-3)
      * topics: +(1 - Jaccard) of the topic sets (0-1)
      * proposal_values: +1 if the classes differ (0-1)
      * entities: +(1 - Jaccard) of the paragraph's normalized entity-name sets (0-1)
    Max ~6. Used only to surface the most contestable paragraphs for a human read;
    nothing gates on it."""
    df = merged_para.copy()
    flag_dis = sum((df[f + "_primary"] != df[f + "_opus"]).astype(int) for f in FLAG_FIELDS)
    topic_dis = df.apply(
        lambda r: 1.0 - jaccard(r["topics_primary"], r["topics_opus"]), axis=1)
    pv_dis = (df["proposal_values_primary"] != df["proposal_values_opus"]).astype(int)

    pa = _para_entity_names(primary_ent)
    ob = _para_entity_names(opus_ent)
    ent_dis = df.apply(
        lambda r: 1.0 - jaccard(pa.get((r["doc_name"], r["para_idx"]), set()),
                                ob.get((r["doc_name"], r["para_idx"]), set())),
        axis=1)
    df["disagreement"] = flag_dis + topic_dis + pv_dis + ent_dis
    return df


# --------------------------------------------------------------------------
# I/O layer — load primary + Opus tables, restrict to the sample, join on keys
# --------------------------------------------------------------------------

OPUS_MODEL = "claude-opus-4-8"
AGREEMENT_PATH = ann.ANNOTATIONS_DIR / "agreement_v1.parquet"
REPORT_PATH = REPO_ROOT / "notes" / "agreement-report-v1.md"

_ENTITY_COLS = ["doc_name", "para_idx", "entity", "type", "stance", "run_id"]


def _opus_table(spec_name: str) -> str:
    """The per-model output table name the Opus pass writes (single source of
    truth: annotate._table_name)."""
    return annotate._table_name(spec_name, OPUS_MODEL)


def _require_opus_artifacts() -> None:
    p = ann.annotation_path(_opus_table("paragraph_annotations"))
    if not p.exists():
        raise FileNotFoundError(
            f"Opus second-opinion artifacts not found (expected {p}). The paid "
            "Opus pass has not run yet. Run it first:\n"
            "  pp-annotate submit --run-id <opus-run> --model claude-opus-4-8 "
            "--sample data/llm_annotations/agreement_sample_v1.json --yes\n"
            "  pp-annotate ingest --run-id <opus-run>"
        )


def _read_entities(name: str, docs: set[str], keys: pd.DataFrame) -> pd.DataFrame:
    """Entity rows for a table, restricted to the sample docs AND to the shared
    (both-models-annotated) paragraph keys. Missing table -> empty frame."""
    path = ann.annotation_path(name)
    if not path.exists():
        return pd.DataFrame(columns=_ENTITY_COLS)
    ent = pd.read_parquet(path)
    ent = ent[ent["doc_name"].isin(docs)]
    ent = ent.merge(keys, on=["doc_name", "para_idx"], how="inner")
    return ent


def _load_joined(sample_docs: set[str], speeches: pd.DataFrame) -> dict:
    """Join the primary tables (ALL run_ids) against the Opus tables, restricted to
    the sample. Raises clearly if the Opus artifacts don't exist yet."""
    _require_opus_artifacts()
    era_of = {d: era_bin(y) for d, y in zip(speeches["doc_name"], speeches["year"])}

    # Paragraph judgment. The loaders enforce one-row-per-key on BOTH sides, so the
    # one_to_one merge is safe; the primary side spans the convergence ladder's
    # several run_ids (we take the whole merged table, never a single run_id).
    primary = ann.load_paragraph_annotations("paragraph_annotations")
    opus = ann.load_paragraph_annotations(_opus_table("paragraph_annotations"))
    primary = primary[primary["doc_name"].isin(sample_docs)]
    opus = opus[opus["doc_name"].isin(sample_docs)]
    merged_para = primary.merge(
        opus, on=ann.PARAGRAPH_KEY, how="inner",
        suffixes=("_primary", "_opus"), validate="one_to_one")
    merged_para["era_bin"] = merged_para["doc_name"].map(era_of)

    shared_keys = merged_para[ann.PARAGRAPH_KEY].drop_duplicates()
    primary_ent = _read_entities("paragraph_entities", sample_docs, shared_keys)
    opus_ent = _read_entities(_opus_table("paragraph_entities"), sample_docs, shared_keys)
    for e in (primary_ent, opus_ent):
        if len(e):
            e["era_bin"] = e["doc_name"].map(era_of)

    # Speech-factual (optional: only if the Opus factual pass has been ingested).
    merged_speech = None
    opus_sp_path = ann.annotation_path(_opus_table("speech_annotations"))
    if opus_sp_path.exists():
        primary_sp_path = ann.annotation_path("speech_annotations")
        if not primary_sp_path.exists():
            raise FileNotFoundError(
                f"the Opus speech-factual table exists ({opus_sp_path}) but the primary "
                f"speech table is missing ({primary_sp_path}) — cannot join the "
                "speech-factual pass. Run the primary speech_annotations pass first."
            )
        p_sp = ann.load_speech_annotations("speech_annotations")
        o_sp = ann.load_speech_annotations(_opus_table("speech_annotations"))
        p_sp = p_sp[p_sp["doc_name"].isin(sample_docs)]
        o_sp = o_sp[o_sp["doc_name"].isin(sample_docs)]
        merged_speech = p_sp.merge(
            o_sp, on=ann.SPEECH_KEY, how="inner",
            suffixes=("_primary", "_opus"), validate="one_to_one")
        merged_speech["era_bin"] = merged_speech["doc_name"].map(era_of)

    # Coverage against the corpus: how much of the sample the Opus pass actually
    # covers (100% is the bar; partial coverage is surfaced, not silently dropped).
    corpus_paras = pd.read_parquet(ann.PARAGRAPHS_PATH)
    expected = corpus_paras[corpus_paras["doc_name"].isin(sample_docs)]
    n_expected = len(expected[ann.PARAGRAPH_KEY].drop_duplicates())
    matched_docs = set(merged_para["doc_name"])
    coverage = {
        "n_sample_docs": len(sample_docs),
        "n_matched_docs": len(matched_docs),
        "n_expected_paras": int(n_expected),
        "n_matched_paras": int(len(merged_para)),
        "para_coverage": (len(merged_para) / n_expected) if n_expected else 0.0,
        "unmatched_docs": sorted(sample_docs - matched_docs),
        "has_speech": merged_speech is not None,
    }
    return {
        "merged_para": merged_para,
        "primary_ent": primary_ent,
        "opus_ent": opus_ent,
        "merged_speech": merged_speech,
        "coverage": coverage,
    }


def build_agreement(sample_path: Path = SAMPLE_PATH, out_path: Path = AGREEMENT_PATH,
                    speeches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the agreement table and write it to ``out_path``. Fails clearly if
    the Opus artifacts don't exist yet."""
    if speeches is None:
        speeches = load()
    sample_docs = set(sample_doc_names(sample_path))
    joined = _load_joined(sample_docs, speeches)
    table = build_metrics_table(
        joined["merged_para"], joined["primary_ent"],
        joined["opus_ent"], joined["merged_speech"])
    ann.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(Path(out_path), index=False)
    cov = joined["coverage"]
    print(f"wrote {out_path}  ({len(table)} metric rows)")
    print(f"  sample coverage: {cov['n_matched_paras']:,}/{cov['n_expected_paras']:,} "
          f"paragraphs ({cov['para_coverage']:.2%}), "
          f"{cov['n_matched_docs']}/{cov['n_sample_docs']} speeches")
    if cov["para_coverage"] < 1.0:
        print(f"  WARNING: Opus coverage < 100% — {len(cov['unmatched_docs'])} sample "
              "speech(es) missing from the join; metrics cover the overlap only.")
    return table


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _fmt(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


# Field/metric render order (entities rendered in their own section).
_REPORT_FIELDS: list[tuple[str, list[str]]] = [
    ("topics", ["jaccard"]),
    ("party_attack", ["cohen_kappa"]),
    ("enemy_naming", ["cohen_kappa"]),
    ("zero_sum", ["cohen_kappa"]),
    ("proposal_values", ["exact_match"]),
    ("speech_type", ["cohen_kappa", "exact_match"]),
    ("audience", ["cohen_kappa", "exact_match"]),
    ("medium", ["cohen_kappa", "exact_match"]),
]


def _metric_block(table: pd.DataFrame, field: str, metric: str) -> list[str]:
    sub = table[(table["field"] == field) & (table["metric"] == metric)]
    if sub.empty:
        return []
    is_kappa = metric == "cohen_kappa"
    lines = [f"**{field} — {metric}**", "",
             "| era | value | n |" + (" flag |" if is_kappa else ""),
             "|---|---|---|" + ("---|" if is_kappa else "")]
    # overall first, then era bins ascending
    ordered = sub.sort_values("era_bin", key=lambda s: s.map(lambda b: -1e9 if b == OVERALL_BIN else b))
    for r in ordered.itertuples(index=False):
        flag = ""
        if is_kappa and not (isinstance(r.value, float) and math.isnan(r.value)) \
                and r.value < KAPPA_LOW_THRESHOLD:
            flag = " LOW (kappa<0.4)"
        cells = f"| {r.era_label} | {_fmt(r.value)} | {r.n:,} |"
        if is_kappa:
            cells += f"{flag} |"
        lines.append(cells)
    lines.append("")
    return lines


def _render_report(table: pd.DataFrame, joined: dict, sample_path: Path,
                   speeches: pd.DataFrame) -> str:
    cov = joined["coverage"]
    sample = load_sample(sample_path)
    low = table[(table["metric"] == "cohen_kappa") & (table["era_bin"] != OVERALL_BIN)
                & (table["value"] < KAPPA_LOW_THRESHOLD)]

    lines = ["# Inter-model annotation agreement — v1", ""]
    lines += [
        f"- primary model: `claude-sonnet-5` (the committed annotation tables, all run_ids)",
        f"- second-opinion model: `{OPUS_MODEL}` (byte-identical prompts; the model is the "
        "only variable)",
        f"- sample: `{sample_path}` — {sample.get('n_sampled')} speeches, seed "
        f"{sample.get('seed')}, drawn {sample.get('draw_date')}",
        f"- join coverage: {cov['n_matched_paras']:,}/{cov['n_expected_paras']:,} sample "
        f"paragraphs ({cov['para_coverage']:.2%}); "
        f"{cov['n_matched_docs']}/{cov['n_sample_docs']} speeches",
        "- speech-factual pass included: "
        + ("yes" if cov["has_speech"] else "NO (Opus factual table absent)"),
        "- Cohen's kappa < 0.4 is FLAGGED low-confidence below. It is flagged, NEVER gated: "
        "disagreement is published, not hidden.",
        "",
    ]
    if cov["para_coverage"] < 1.0:
        lines += [
            f"> WARNING: Opus coverage is below 100% — {len(cov['unmatched_docs'])} sample "
            "speech(es) are missing from the join. Metrics below cover the overlap only; "
            "re-run the Opus pass to convergence before treating these as final.",
            "",
        ]

    lines += ["## Agreement by field and era", "",
              "Every era bin reports its n; nothing is suppressed. `overall` is the "
              "corpus-wide (non-stratified) value.", ""]
    for field, metrics in _REPORT_FIELDS:
        for metric in metrics:
            lines += _metric_block(table, field, metric)

    # Entities
    lines += ["## Entities", "",
              "Name-match (Jaccard over normalized `(doc, para, name)` keys) is reported "
              "SEPARATELY from stance agreement among matched entities, plus each side's "
              "unmatched-entity rate — so a granularity difference (one model splitting an "
              "entity in two) shows up as unmatched entities, not as a stance disagreement. "
              "Names are normalized by lowercasing and collapsing whitespace.", ""]
    for metric in ["name_match_rate", "stance_agreement",
                   "unmatched_primary_rate", "unmatched_opus_rate"]:
        lines += _metric_block(table, "entities", metric)

    if not low.empty:
        flagged = ", ".join(f"{r.field}/{r.era_label}" for r in low.itertuples(index=False))
        lines += ["## Low-confidence flags (kappa < 0.4)", "",
                  f"{len(low)} field x era cell(s) flagged: {flagged}.",
                  "These are the cells where the two models most disagree after correcting "
                  "for chance — candidates for the anachronism / reputation-contamination "
                  "risks the design set out to make visible. Flagged, not gated.", ""]

    lines += _render_top_disagreements(joined, speeches)
    lines.append("")
    return "\n".join(lines)


def _render_top_disagreements(joined: dict, speeches: pd.DataFrame, k: int = 10) -> list[str]:
    scored = paragraph_disagreement(
        joined["merged_para"], joined["primary_ent"], joined["opus_ent"])
    top = scored.nlargest(k, "disagreement")
    paras = pd.read_parquet(ann.PARAGRAPHS_PATH)[["doc_name", "para_idx", "text"]]
    text_of = {(r.doc_name, r.para_idx): r.text for r in paras.itertuples(index=False)}
    p_ent = _para_entity_display(joined["primary_ent"])
    o_ent = _para_entity_display(joined["opus_ent"])

    lines = [f"## {k} highest-disagreement paragraphs (qualitative)", "",
             "Disagreement score = flag mismatches (0-3) + (1 - topic Jaccard) + "
             "proposal_values mismatch + (1 - entity-name Jaccard). Spot-read these: genuine "
             "ambiguity is expected and fine; a systematic schema misread by one model would "
             "be a prompt bug to fix before trusting the table.", ""]
    for r in top.itertuples(index=False):
        key = (r.doc_name, r.para_idx)
        text = str(text_of.get(key, "")).strip()
        if len(text) > 700:
            text = text[:700].rstrip() + " […]"
        lines += [
            f"### {era_label(r.era_bin)} · score {r.disagreement:.2f} · "
            f"`{r.doc_name}` para {r.para_idx}",
            "",
            # _as_label_set, not set(): a null topics cell is selection-BIASED
            # into this top-10 (null vs real list = maximal topic disagreement),
            # so the render path needs the same guard as the metric path.
            f"- topics — primary: {sorted(_as_label_set(r.topics_primary))} | "
            f"opus: {sorted(_as_label_set(r.topics_opus))}",
            f"- flags — primary: "
            f"party={_fmt_flag(r.party_attack_primary)} enemy={_fmt_flag(r.enemy_naming_primary)} "
            f"zero_sum={_fmt_flag(r.zero_sum_primary)} | opus: "
            f"party={_fmt_flag(r.party_attack_opus)} enemy={_fmt_flag(r.enemy_naming_opus)} "
            f"zero_sum={_fmt_flag(r.zero_sum_opus)}",
            f"- proposal_values — primary: {r.proposal_values_primary} | "
            f"opus: {r.proposal_values_opus}",
            f"- entities — primary: {p_ent.get(key, [])} | opus: {o_ent.get(key, [])}",
            "",
            "> " + text.replace("\n", "\n> "),
            "",
        ]
    return lines


def _para_entity_display(ent: pd.DataFrame) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    if ent is not None and len(ent):
        for r in ent.itertuples(index=False):
            out.setdefault((r.doc_name, r.para_idx), []).append(f"{r.entity} ({r.stance})")
    return out


def agreement_report(sample_path: Path = SAMPLE_PATH, agreement_path: Path = AGREEMENT_PATH,
                     report_path: Path = REPORT_PATH,
                     speeches: pd.DataFrame | None = None) -> Path:
    """Render the markdown agreement report. Fails clearly if ``agreement_v1.parquet``
    (run build_agreement first) or the Opus artifacts don't exist yet."""
    agreement_path = Path(agreement_path)
    if not agreement_path.exists():
        raise FileNotFoundError(
            f"{agreement_path} not found — run build_agreement first "
            "(python -m presidential_profiles.agreement build)."
        )
    table = pd.read_parquet(agreement_path)
    if speeches is None:
        speeches = load()
    sample_docs = set(sample_doc_names(sample_path))
    joined = _load_joined(sample_docs, speeches)  # raises clearly if Opus missing
    md = _render_report(table, joined, sample_path, speeches)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md)
    print(f"wrote {report_path}")
    return report_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_draw_sample(args) -> None:
    draw_date = args.draw_date or os.environ.get("PP_DRAW_DATE") or date.today().isoformat()
    out = draw_sample(
        path=Path(args.out) if args.out else SAMPLE_PATH,
        seed=args.seed, draw_date=draw_date, force=args.force)
    print(f"wrote {args.out or SAMPLE_PATH}")
    print(f"  seed={out['seed']}  draw_date={out['draw_date']}  "
          f"total sampled={out['n_sampled']} of {out['n_corpus_speeches']}")
    for b in out["bins"]:
        print(f"  bin {b['era_bin']} ({b['era_label']}): "
              f"{b['n_sampled']:>3} of {b['n_in_bin']:>3}")


def _cmd_build(args) -> None:
    build_agreement(sample_path=Path(args.sample) if args.sample else SAMPLE_PATH)


def _cmd_report(args) -> None:
    agreement_report(sample_path=Path(args.sample) if args.sample else SAMPLE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pp-agreement",
        description="Inter-model annotation agreement: draw the sample, compute the "
                    "per-field/per-era agreement table, render the report. This module "
                    "makes NO API calls — the paid Opus pass runs via pp-annotate.")
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("draw-sample", help="draw + persist the era-stratified sample JSON")
    ds.add_argument("--out", help=f"output path (default {SAMPLE_PATH})")
    ds.add_argument("--seed", type=int, default=SAMPLE_SEED, help=f"RNG seed (default {SAMPLE_SEED})")
    ds.add_argument("--draw-date", help="draw date recorded in the JSON "
                                        "(default: $PP_DRAW_DATE or today)")
    ds.add_argument("--force", action="store_true",
                    help="overwrite an existing sample file (invalidates the current membership)")
    ds.set_defaults(func=_cmd_draw_sample)

    for name, fn, help_text in [
        ("build", _cmd_build, "compute agreement_v1.parquet from the primary + Opus tables"),
        ("report", _cmd_report, "render notes/agreement-report-v1.md"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--sample", help=f"sample JSON (default {SAMPLE_PATH})")
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
