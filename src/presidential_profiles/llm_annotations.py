"""Storage layer for LLM-derived annotations.

Everything else in the pipeline is free, local, deterministic compute: delete a
parquet, rerun, get the same bytes back. LLM annotations are none of those
things — they cost money, they cannot be reproduced exactly, and the model that
produced them will be deprecated. So they need what the rest of the pipeline
gets for free: a way to tell, later, exactly what produced a given number.

Two rules, both learned from `invocation_tone.json` (the file this module
replaces), whose 101 labels were keyed by nothing but their position in a JSON
array:

1. Every annotation is keyed by `doc_name` (speech-level) or
   `(doc_name, para_idx)` (paragraph-level). Never by row order. The loaders
   raise on a duplicate key rather than warn — a silently duplicated key is the
   exact failure this layer exists to prevent, and a warning in a research
   notebook is a warning nobody reads.
2. Every row carries a `run_id` pointing at a manifest recording the model,
   prompt version and hash, batch ID, token counts, actual cost, and a
   fingerprint of the corpus the run was computed against. If the corpus is
   rebuilt and its fingerprint moves, the annotations are stale and you can
   prove it.

Run as:  PYTHONPATH=src arch -x86_64 .venv/bin/python -m ...
(the repo venv is x86_64 under Rosetta).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import pandas as pd

from .corpus import DATA_DIR, load

ANNOTATIONS_DIR = DATA_DIR / "llm_annotations"
MANIFESTS_DIR = ANNOTATIONS_DIR / "manifests"
RUNS_DIR = ANNOTATIONS_DIR / "runs"

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.parquet"

SPEECH_KEY = ["doc_name"]
PARAGRAPH_KEY = ["doc_name", "para_idx"]


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------


@dataclass
class Manifest:
    """Provenance for one annotation run. `cost_usd` is the ACTUAL cost read
    back from the API's usage numbers at ingest — never the pre-flight
    estimate, so a manifest can never quietly claim a run was cheaper than it
    was."""

    run_id: str
    model: str
    prompt_version: str
    prompt_hash: str
    date: str
    batch_id: str | None = None
    n_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float | None = None
    corpus_fingerprint: dict | None = None
    fields: list[str] = field(default_factory=list)
    annotation_files: list[str] = field(default_factory=list)
    notes: str = ""


def manifest_path(run_id: str) -> Path:
    return MANIFESTS_DIR / f"{run_id}.json"


def write_manifest(manifest: Manifest) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = manifest_path(manifest.run_id)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(run_id: str) -> Manifest:
    # Tolerate forward schema drift: a manifest written by a LATER version may
    # carry fields this dataclass doesn't know yet. A provenance record whose
    # thesis is "read this later to learn what produced a number" must stay
    # readable across that drift, so drop unknown keys rather than raise.
    raw = json.loads(manifest_path(run_id).read_text())
    known = {f.name for f in fields(Manifest)}
    return Manifest(**{k: v for k, v in raw.items() if k in known})


# --------------------------------------------------------------------------
# corpus fingerprint
# --------------------------------------------------------------------------


def corpus_fingerprint(
    speeches: pd.DataFrame | None = None, paragraphs: pd.DataFrame | None = None
) -> dict:
    """Row counts plus a stable hash of the sorted doc_name list. Sorted, so
    the hash tracks corpus *content* and not the order rows happen to sit in;
    if this moves, every annotation keyed against it is suspect."""
    if speeches is None:
        speeches = load()
    if paragraphs is None:
        paragraphs = pd.read_parquet(PARAGRAPHS_PATH)
    names = "\n".join(sorted(speeches["doc_name"]))
    return {
        "n_speeches": int(len(speeches)),
        "n_paragraphs": int(len(paragraphs)),
        "doc_name_sha256": hashlib.sha256(names.encode()).hexdigest(),
    }


# --------------------------------------------------------------------------
# annotation tables
# --------------------------------------------------------------------------


def annotation_path(name: str) -> Path:
    return ANNOTATIONS_DIR / f"{name}.parquet"


def _require_key(df: pd.DataFrame, key: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in key + ["run_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: annotation table is missing columns {missing}")
    dupes = df[df.duplicated(key, keep=False)]
    if not dupes.empty:
        sample = dupes[key].head(5).to_dict("records")
        raise ValueError(
            f"{name}: {len(dupes)} rows share a duplicate key on {key}; "
            f"an annotation key must identify exactly one row. Examples: {sample}"
        )
    return df


def load_speech_annotations(name: str) -> pd.DataFrame:
    """Speech-level annotations, keyed by doc_name. Raises on a duplicate key."""
    return _require_key(pd.read_parquet(annotation_path(name)), SPEECH_KEY, name)


def load_paragraph_annotations(name: str) -> pd.DataFrame:
    """Paragraph-level annotations, keyed by (doc_name, para_idx). Raises on a
    duplicate key."""
    return _require_key(pd.read_parquet(annotation_path(name)), PARAGRAPH_KEY, name)


def write_annotations(name: str, df: pd.DataFrame, unit: str) -> Path:
    """Write an annotation table, enforcing the key before it hits disk (a bad
    table should never exist, not merely fail to load)."""
    key = SPEECH_KEY if unit == "speech" else PARAGRAPH_KEY
    _require_key(df, key, name)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = annotation_path(name)
    df.sort_values(key).reset_index(drop=True).to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------
# invocation_tone migration
# --------------------------------------------------------------------------

INVOCATION_TONE_JSON = ANNOTATIONS_DIR / "invocation_tone.json"
INVOCATION_TONE_PATH = ANNOTATIONS_DIR / "invocation_tone.parquet"
INVOCATION_TONE_RUN_ID = "2026-07-12-invocation-tone-fable5"

# The (speaker, target) pair behind each key in the original JSON, and the
# regex profiles.INVOCATION_PATTERNS used for that target. Deliberately NOT
# imported from profiles.py: these 101 labels were generated against *these*
# patterns and *this* concatenation, and pinning them here keeps the migration
# correct even if profiles.py is edited later. The 69/32 assertion is what
# catches drift.
INVOCATION_PAIRS = {
    "biden_to_trump": ("Joe Biden", "Donald Trump", r"\bTrump\b", 69),
    "trump_to_obama": ("Donald Trump", "Barack Obama", r"\bObama\b", 32),
}

# profiles.invocations() reads text[m.start()-130 : m.end()+130]. The stored
# `method` string claims +/-110. Both are recorded: the method string verbatim
# (it is the historical claim), the discrepancy in the manifest `notes`, and
# the code's real behaviour here (it is what actually produced the labels).
TONE_WINDOW = 130


def _speaker_text(df: pd.DataFrame, speaker: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Replicate profiles.invocations()'s per-speaker concatenation, tracking
    where each speech lands in the joined string.

    profiles.py does `text = " ".join(group["transcript"])` over
    `df.groupby("president")`, i.e. the speaker's speeches in corpus order
    (corpus.load() sorts by date), single-space separated. Char offsets are
    only meaningful against exactly that string."""
    group = df[df["president"] == speaker]
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for doc_name, transcript in zip(group["doc_name"], group["transcript"]):
        spans.append((cursor, cursor + len(transcript), doc_name))
        cursor += len(transcript) + 1  # +1 for the " " that join() inserts
    return " ".join(group["transcript"]), spans


def _locate(spans: list[tuple[int, int, str]], start: int, end: int) -> tuple[str, int, int]:
    """Map a global (start, end) in the joined string back to one speech."""
    for doc_start, doc_end, doc_name in spans:
        if doc_start <= start < doc_end:
            if end > doc_end:
                raise ValueError(
                    f"match at {start}:{end} spans the join gap after {doc_name}; "
                    "the reconstruction cannot anchor it to a single speech"
                )
            return doc_name, start - doc_start, end - doc_start
    raise ValueError(f"match at {start} landed in a join gap, not in any speech")


def migrate_invocation_tone(df: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Re-anchor the 101 hand-labelled invocation-tone labels from array
    position to (doc_name, char_start, char_end).

    The original file stored labels as bare JSON arrays whose only key was
    their index. Recovering the anchor means replaying the exact regex scan
    over the exact concatenated text that produced them, then zipping the
    labels back on in match order — the one and only time position is trusted,
    and it is asserted to death: exactly 69 and 32 matches, and label
    histograms identical before and after."""
    if INVOCATION_TONE_PATH.exists() and not force:
        return pd.read_parquet(INVOCATION_TONE_PATH)

    # This commit deletes the source JSON once the parquet supersedes it, so a
    # forced re-derivation against real data would otherwise hit a bare
    # FileNotFoundError deep in read_text(). Signpost where the byte went instead.
    if not INVOCATION_TONE_JSON.exists():
        relpath = INVOCATION_TONE_JSON.relative_to(DATA_DIR.parent)
        raise FileNotFoundError(
            f"invocation_tone source JSON was removed after migration; recover it from "
            f"git history (git show <rev>:{relpath}) to re-derive, or load the committed "
            "parquet without force"
        )
    original = json.loads(INVOCATION_TONE_JSON.read_text())
    if df is None:
        df = load()
    first_year = df.groupby("president")["year"].min()

    rows = []
    for key, (speaker, target, pattern, expected) in INVOCATION_PAIRS.items():
        labels = original[key]
        # profiles.invocations() only counts a target the speaker *followed*.
        if not first_year[target] < first_year[speaker]:
            raise ValueError(f"{key}: {target} does not precede {speaker}")

        text, spans = _speaker_text(df, speaker)
        matches = list(re.compile(pattern).finditer(text))

        # BAIL CONDITION: if the scan no longer reproduces the original match
        # count, the labels cannot be trusted to line up with the matches, and
        # hand-tuning the alignment would be exactly the fragility this whole
        # layer exists to eliminate.
        if len(matches) != expected:
            raise AssertionError(
                f"{key}: reconstruction found {len(matches)} matches, expected "
                f"{expected}. The corpus or the pattern has changed; the stored "
                f"labels can no longer be anchored. STOP — do not hand-align."
            )
        if len(labels) != expected:
            raise AssertionError(f"{key}: {len(labels)} stored labels, expected {expected}")

        for m, label in zip(matches, labels):
            doc_name, local_start, local_end = _locate(spans, m.start(), m.end())
            rows.append({
                "doc_name": doc_name,
                "char_start": local_start,
                "char_end": local_end,
                "speaker": speaker,
                "target": target,
                "mention": m.group(0),
                # The window as profiles.py actually computed it: global, so it
                # can run past a speech boundary. Stored as produced, not as
                # tidied, because it is what the labeller was shown.
                "window": text[max(0, m.start() - TONE_WINDOW):m.end() + TONE_WINDOW],
                "label": label,
                "run_id": INVOCATION_TONE_RUN_ID,
            })

    out = pd.DataFrame(rows)
    before = {k: pd.Series(v).value_counts().to_dict() for k, v in original.items()
              if k in INVOCATION_PAIRS}
    after = {k: out[out["speaker"] == INVOCATION_PAIRS[k][0]]["label"].value_counts().to_dict()
             for k in INVOCATION_PAIRS}
    if before != after:
        raise AssertionError(f"label histogram changed in migration: {before} -> {after}")

    dupes = out[out.duplicated(["doc_name", "char_start"], keep=False)]
    if not dupes.empty:
        raise ValueError(f"{len(dupes)} mentions share a (doc_name, char_start) key")

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.sort_values(["doc_name", "char_start"]).reset_index(drop=True).to_parquet(
        INVOCATION_TONE_PATH, index=False
    )

    write_manifest(Manifest(
        run_id=INVOCATION_TONE_RUN_ID,
        model="claude-fable-5",
        prompt_version="invocation_tone/in-session-2026-07-12",
        # No prompt was recorded — this run predates this layer. The hash is of
        # the only artifact that survives (the method string); calling it a
        # prompt hash would be a lie.
        prompt_hash="sha256:" + hashlib.sha256(original["method"].encode()).hexdigest(),
        date="2026-07-12",
        batch_id=None,
        n_requests=0,
        cost_usd=None,
        corpus_fingerprint=corpus_fingerprint(speeches=df),
        fields=["label"],
        annotation_files=[INVOCATION_TONE_PATH.name],
        notes=(
            "MIGRATED from data/llm_annotations/invocation_tone.json, whose 101 labels "
            "were keyed only by position in two JSON arrays (biden_to_trump=69, "
            "trump_to_obama=32). Re-anchored to (doc_name, char_start, char_end) by "
            "replaying profiles.invocations()'s per-president transcript concatenation "
            "with cumulative offset tracking; match counts and label histograms verified "
            "identical (C=52/N=16/R=1 and C=20/N=10/R=2).\n\n"
            "ORIGINAL `method` FIELD, VERBATIM (the historical record of what was "
            f"claimed):\n{original['method']}\n\n"
            "DISCREPANCY, NOT CORRECTED: the method string above claims a +/-110 char "
            "tone window, but profiles.py:137 — the code that actually produced these "
            "matches — uses +/-130. The reconstruction replicates the CODE (+/-130), "
            "since that is what the labeller was shown. Window size affects only the "
            "window text, not the match count, so the 69/32 counts are unaffected. The "
            "method string is preserved unedited rather than silently fixed.\n\n"
            "This run was interactive, not batched: no batch_id, no request count, and "
            "no usage numbers were recorded, so cost_usd is null (unknown, not zero). "
            "Not reproducible from this manifest — that is the point of the layer that "
            "replaces it."
        ),
    ))
    return out


def load_invocation_tone() -> pd.DataFrame:
    """Invocation-tone labels, keyed by (doc_name, char_start) — a mention-level
    table, so neither the speech nor the paragraph loader applies."""
    df = pd.read_parquet(INVOCATION_TONE_PATH)
    dupes = df[df.duplicated(["doc_name", "char_start"], keep=False)]
    if not dupes.empty:
        raise ValueError(f"invocation_tone: {len(dupes)} rows share a (doc_name, char_start) key")
    return df
