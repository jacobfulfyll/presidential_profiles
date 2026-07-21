"""Drive LLM annotation passes over the corpus via the Anthropic Batches API.

This is the one command in the pipeline that spends money. `pp-analyze` and
`pp-site` are free local compute and can be rerun on a whim; an annotation pass
over 36,229 paragraphs cannot. So it lives behind its own CLI, and nothing in
the data-rebuild path calls into it — a `--force` rebuild can never re-trigger
a paid run by accident.

    pp-annotate dry-run --limit 5           # writes JSONL, estimates cost, ZERO network calls
    pp-annotate submit  --run-id ... --yes  # creates the batch (SPENDS MONEY; --yes is required)
    pp-annotate status  --run-id ...        # polls it
    pp-annotate ingest  --run-id ...        # parses results -> parquet + manifest

Requests are grouped one-per-speech (all paragraph fields for a speech in a
single request), which is 3-4x cheaper than one request per paragraph and keeps
whole-speech context in front of the model. The shared rubric goes in a cached
system block.

Prompts and rubrics are NOT this module's job — FIELD_SPECS ships with one
placeholder spec, enough to exercise dry-run. The real rubrics land with the
annotation pass itself.

Run as:  PYTHONPATH=src arch -x86_64 .venv/bin/python -m ...
(the repo venv is x86_64 under Rosetta).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import TypeAdapter

from . import llm_annotations as ann
from .corpus import REPO_ROOT, load
from .prompts import annotation_v1 as av1

# --------------------------------------------------------------------------
# model + pricing
# --------------------------------------------------------------------------

MODEL = "claude-sonnet-5"

# USD per million tokens. Sonnet 5 lists at $3.00 / $15.00; introductory
# pricing of $2.00 / $10.00 is in effect THROUGH 2026-08-31. After that date
# these two constants must be raised to 3.00 / 15.00 or every cost estimate
# here understates by a third. Batch requests take 50% off both.
INTRO_PRICING_ENDS = "2026-08-31"
INPUT_USD_PER_MTOK = 2.00
OUTPUT_USD_PER_MTOK = 10.00
LIST_INPUT_USD_PER_MTOK = 3.00
LIST_OUTPUT_USD_PER_MTOK = 15.00
BATCH_DISCOUNT = 0.50
# Cache reads bill at ~0.1x input, cache writes at ~1.25x.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 256 * 1024 * 1024

# Ceiling `submit` refuses to cross without an explicit, deliberate override.
# The full corpus (1,057 speeches, 36,229 paragraphs) estimates at ~$8.53 for a
# single paragraph-level spec, so this leaves room for a speech-level spec
# alongside it while staying well inside the annotation budget.
MAX_COST_USD = 25.00

# A speech-level answer is one small object, not one object per paragraph.
_SPEECH_EST_OUTPUT_TOKENS = 64

# --------------------------------------------------------------------------
# pilot + QA thresholds
# --------------------------------------------------------------------------

# The pilot is a deterministic, era-stratified sample: chronologically ordered
# speeches cut into PILOT_N equal-frequency strata (Washington's first speeches
# in stratum 0, the most recent in the last), one speech drawn per stratum with
# a fixed seed. Stable across runs by construction.
PILOT_N = 20
PILOT_RANDOM_STATE = 42

# Pilot auto-check gates. Full-mode QA reports these numbers without gating. The
# first three are literal task.md Verification-Strategy values; the last two
# operationalize task.md's qualitative "~0%" / "(substring check)" language.
QA_MIN_COVERAGE = 0.99            # >= 99% of in-scope paragraphs annotated
QA_MAX_SCHEMA_FAILURE = 0.01      # < 1% of requests failed / returned no text
QA_FLAG_DEGENERATE_HIGH = 0.60    # a flag firing on > 60% corpus-wide is degenerate
QA_FLAG_DEGENERATE_LOW = 0.005    # ...as is one firing on ~0% (operationalizes "~0%")
QA_MIN_ENTITY_SUBSTRING = 0.80    # >= 80% of entity names appear in their paragraph
QA_REPORT_PATH = REPO_ROOT / "notes" / "annotation-qa-v1.md"

_JUDGMENT_FLAGS = ("party_attack", "enemy_naming", "zero_sum")


def _rates() -> tuple[float, float]:
    intro = date.today().isoformat() <= INTRO_PRICING_ENDS
    if intro:
        return INPUT_USD_PER_MTOK, OUTPUT_USD_PER_MTOK
    return LIST_INPUT_USD_PER_MTOK, LIST_OUTPUT_USD_PER_MTOK


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Batch-discounted cost in USD from raw token counts."""
    in_rate, out_rate = _rates()
    billable_in = (
        input_tokens
        + cache_creation_input_tokens * CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens * CACHE_READ_MULTIPLIER
    )
    dollars = (billable_in * in_rate + output_tokens * out_rate) / 1_000_000
    return dollars * BATCH_DISCOUNT


# --------------------------------------------------------------------------
# field spec registry
# --------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """One annotatable field: what to ask, what shape the answer takes, and
    whether it is asked once per speech or once per paragraph.

    `json_schema` is JSON Schema from the start (additionalProperties: false,
    explicit required) so the next task can hand it straight to
    output_config.format with no translation step."""

    name: str
    unit: str  # "speech" | "paragraph"
    prompt_version: str
    rubric: str  # shared system block -> the natural prompt-cache breakpoint
    instruction: str  # per-request user text
    json_schema: dict
    max_tokens: int = 8192

    # Optional extension points, all defaulting to the pre-existing behaviour so
    # the placeholder spec and every existing test are unaffected:
    #   context_builder(speech, paras) -> str  builds the per-request user body
    #     (everything after `instruction`). This is where masking lives: the
    #     judgment pass injects the decade and withholds president/title; the
    #     factual pass sends title + year + the first paragraphs. If None, the
    #     default body is used (para_idx-tagged paragraphs, or the transcript).
    #   max_tokens_fn(n_paragraphs) -> int  overrides the static max_tokens per
    #     request (judgment: 2000 + 80*n). If None, `max_tokens` is used.
    #   est_output_tokens_per_para  scales the offline OUTPUT estimate for a
    #     paragraph-unit spec (richer schemas emit more per paragraph).
    #   entity_field  names a nested array field in each paragraph item that
    #     ingest explodes into a second long-format table (paragraph_entities).
    #   effort  -> output_config.effort per spec. The judgment pass emits a long
    #     structured array and under-produced at "low" in the pilot (some speeches
    #     returned 1-2 items then end_turn), so it runs at "medium"; the tiny
    #     one-object factual pass stays "low". Not a forbidden param — it lives
    #     inside output_config, so _validate_request's key checks are unaffected.
    context_builder: Callable | None = None
    max_tokens_fn: Callable[[int], int] | None = None
    est_output_tokens_per_para: int = 12
    entity_field: str | None = None
    effort: str = "low"

    def prompt_hash(self) -> str:
        payload = json.dumps(
            {"rubric": self.rubric, "instruction": self.instruction,
             "schema": self.json_schema, "version": self.prompt_version},
            sort_keys=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


# PLACEHOLDER. Exists to exercise dry-run end to end; it is not a rubric and is
# not what the paid pass will ask. Real specs land in `run-llm-annotation-pass`.
_PLACEHOLDER_SCHEMA = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "para_idx": {"type": "integer"},
                    "label": {"type": "string", "enum": ["placeholder"]},
                },
                "required": ["para_idx", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["annotations"],
    "additionalProperties": False,
}

FIELD_SPECS: dict[str, FieldSpec] = {
    "placeholder": FieldSpec(
        name="placeholder",
        unit="paragraph",
        prompt_version="placeholder/v0",
        rubric=(
            "PLACEHOLDER RUBRIC — not a real annotation task. It exists only so "
            "`pp-annotate dry-run` can be exercised end to end without a prompt. "
            "Return the label 'placeholder' for every paragraph you are given."
        ),
        instruction=(
            "Return one annotation object per paragraph below, echoing its "
            "para_idx exactly as given."
        ),
        json_schema=_PLACEHOLDER_SCHEMA,
    ),
    # The MASKED judgment pass. Spec name == output parquet name
    # (paragraph_annotations.parquet), which is what downstream analysis loads;
    # the pass identity ("paragraph-judgment") rides in prompt_version. Entities
    # come back nested per paragraph and are exploded to paragraph_entities.
    "paragraph_annotations": FieldSpec(
        name="paragraph_annotations",
        unit="paragraph",
        prompt_version=f"paragraph-judgment/{av1.PROMPT_VERSION}",
        rubric=av1.JUDGMENT_RUBRIC,
        instruction=av1.JUDGMENT_INSTRUCTION,
        json_schema=av1.JUDGMENT_SCHEMA,
        context_builder=av1.judgment_context,
        max_tokens_fn=av1.judgment_max_tokens,
        est_output_tokens_per_para=av1.JUDGMENT_EST_OUTPUT_TOKENS_PER_PARA,
        entity_field=av1.JUDGMENT_ENTITY_FIELD,
        # "low" under-produced on long structured arrays in the pilot; "medium"
        # buys the model room to emit all N items (paired with the count contract
        # in judgment_context) without paying full "high"/thinking cost.
        effort="medium",
    ),
    # The UNMASKED factual pass (speech-typing). One small object per speech.
    "speech_annotations": FieldSpec(
        name="speech_annotations",
        unit="speech",
        prompt_version=f"speech-factual/{av1.PROMPT_VERSION}",
        rubric=av1.FACTUAL_RUBRIC,
        instruction=av1.FACTUAL_INSTRUCTION,
        json_schema=av1.FACTUAL_SCHEMA,
        context_builder=av1.factual_context,
        max_tokens=1024,
    ),
}

# Columns ingest writes itself. A schema field of the same name would silently
# overwrite the key or the provenance pointer.
_RESERVED_COLUMNS = {"doc_name", "para_idx", "run_id"}


def _validate_spec(spec: FieldSpec) -> None:
    """Reject a spec whose unit or schema shape ingest cannot handle.

    Runs at IMPORT, over the whole registry, so a bad spec fails before a batch
    is ever built — never with a KeyError on the ingest side of a batch that has
    already been paid for."""
    if spec.unit not in ("speech", "paragraph"):
        raise ValueError(
            f"{spec.name}: unit={spec.unit!r} is not supported — use 'speech' or 'paragraph'"
        )
    if spec.json_schema.get("type") != "object":
        raise ValueError(f"{spec.name}: json_schema must be an object schema")

    props = spec.json_schema.get("properties") or {}
    if spec.unit == "paragraph":
        annotations = props.get("annotations") or {}
        item_props = (annotations.get("items") or {}).get("properties") or {}
        if annotations.get("type") != "array" or "para_idx" not in item_props:
            raise ValueError(
                f"{spec.name}: a paragraph-unit schema must be an `annotations` array whose "
                "items carry `para_idx` — ingest keys every row on the echoed para_idx"
            )
        payload_fields = set(item_props) - {"para_idx"}
    else:
        if "annotations" in props:
            raise ValueError(
                f"{spec.name}: a speech-unit spec answers once for the whole speech — its "
                "schema must not wrap the fields in an `annotations` array"
            )
        payload_fields = set(props)

    clash = payload_fields & _RESERVED_COLUMNS
    if clash:
        raise ValueError(
            f"{spec.name}: schema fields {sorted(clash)} collide with the key/provenance "
            "columns ingest writes"
        )


for _spec in FIELD_SPECS.values():
    _validate_spec(_spec)


# --------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------


def _custom_id(spec_name: str, doc_name: str, chunk: int | None = None) -> str:
    """custom_id encodes the KEY, never the row position — batch results come
    back in arbitrary order, and keying by position is precisely the bug this
    whole module exists to make impossible.

    doc_name is a 100-char URL path with slashes, so it cannot go into a
    custom_id literally; it is hashed, and requests_index.json (written
    alongside the JSONL) maps every custom_id back to its doc_name and the
    exact paragraph indices that were sent. Ingest re-anchors through that map
    and validates every returned para_idx against it.

    A chunked re-request splits one speech into several requests; `chunk` is the
    0-based chunk index, appended as `-c{i}`. It is deterministic (chunks are the
    sorted paragraphs sliced by size), so the same chunk gets the same id every
    run. Unchunked ids are unchanged (`chunk=None`), so resume and every existing
    custom_id stay byte-identical."""
    digest = hashlib.sha256(doc_name.encode()).hexdigest()[:24]
    base = f"{spec_name}-{digest}"
    return base if chunk is None else f"{base}-c{chunk}"


def build_requests(
    spec: FieldSpec, speeches: pd.DataFrame, paragraphs: pd.DataFrame,
    chunk_size: int | None = None,
) -> tuple[list[Request], dict]:
    """One request per speech — or, when `chunk_size` is set, one request per
    chunk of <= `chunk_size` paragraphs (a convergence lever for speeches whose
    whole-speech judgment request keeps collapsing to a partial array).

    A paragraph-unit spec carries every paragraph of the chunk, each tagged with
    the para_idx the model must echo back (ingest validates every one against the
    chunk's `para_idxs`, in both directions). Chunks of the same speech all key on
    the same doc_name, so their rows merge into full coverage and
    `_already_ingested` sees the speech done once every chunk lands. A speech-unit
    spec answers once for the whole speech — it has no array to chunk, so
    combining it with `chunk_size` is rejected."""
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError(f"--chunk-size must be >= 1, got {chunk_size}")
        if spec.unit != "paragraph":
            raise ValueError(
                f"--chunk-size applies to paragraph-unit specs only; {spec.name!r} is "
                f"unit={spec.unit!r} (a speech-unit request has no array to chunk). "
                "Restrict --spec to the paragraph pass."
            )

    requests: list[Request] = []
    index: dict[str, dict] = {}
    by_doc = {d: g for d, g in paragraphs.groupby("doc_name")}

    for speech in speeches.itertuples():
        doc_name = speech.doc_name
        paras = by_doc.get(doc_name)
        if paras is None or paras.empty:
            continue
        paras = paras.sort_values("para_idx")

        # One "unit of work" per request: the whole speech, or a paragraph chunk.
        if spec.unit == "paragraph" and chunk_size:
            chunks = [paras.iloc[i:i + chunk_size] for i in range(0, len(paras), chunk_size)]
        else:
            chunks = [paras]

        for ci, chunk in enumerate(chunks):
            chunk_no = ci if (spec.unit == "paragraph" and chunk_size) else None
            para_idxs = [int(i) for i in chunk["para_idx"]] if spec.unit == "paragraph" else []

            # The body is built by the spec's per-speech context builder over the
            # CHUNK (the masking seam — decade-only for judgment, with the count
            # contract stating THIS chunk's paragraph count + exact para_idxs), or,
            # for a spec without one, the pre-existing default: para_idx-tagged
            # paragraphs, or the whole transcript for a speech-unit spec.
            if spec.context_builder is not None:
                body = spec.context_builder(speech, chunk)
            elif spec.unit == "paragraph":
                body = "\n\n".join(
                    f"[para_idx={int(r.para_idx)}]\n{r.text}" for r in chunk.itertuples()
                )
            else:
                body = speech.transcript

            # Per-request token budget: static by default, or sized to the chunk
            # (judgment: 2000 + 130*n) so a long chunk is not truncated and a
            # small one does not reserve tokens it will never use.
            max_tokens = (
                spec.max_tokens_fn(len(para_idxs)) if spec.max_tokens_fn else spec.max_tokens
            )
            cid = _custom_id(spec.name, doc_name, chunk=chunk_no)

            params = MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=max_tokens,
                # COST TRAP — DO NOT DELETE THIS LINE.
                # On claude-sonnet-5, OMITTING `thinking` runs ADAPTIVE THINKING BY
                # DEFAULT (unlike older Sonnets, which defaulted to no thinking).
                # Left off, this would silently bill thinking tokens on every one of
                # 1,057 requests and can truncate the JSON answer against max_tokens.
                # A bulk extraction pass wants none of it.
                # Also: `budget_tokens` is REMOVED on Sonnet 5 (400), and non-default
                # temperature/top_p/top_k are REMOVED (400) — depth is controlled by
                # output_config.effort instead. See _validate_request().
                thinking={"type": "disabled"},
                output_config={
                    "effort": spec.effort,
                    "format": {"type": "json_schema", "schema": spec.json_schema},
                },
                system=[{
                    "type": "text",
                    "text": spec.rubric,
                    # Shared across every request in the batch — the natural cache
                    # breakpoint. Prompt caching works inside batches.
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": f"{spec.instruction}\n\n{body}"}],
            )
            requests.append(Request(custom_id=cid, params=params))
            entry = {
                "doc_name": doc_name,
                "spec": spec.name,
                "unit": spec.unit,
                "para_idxs": para_idxs,
            }
            if chunk_no is not None:
                entry["chunk"] = chunk_no  # marks a chunk-escalated request (Build 3)
            index[cid] = entry
    return requests, index


_REQUEST_ADAPTER = TypeAdapter(Request)
_PARAM_KEYS = set(MessageCreateParamsNonStreaming.__annotations__)

# Rejected outright by the Batches API, or by Sonnet 5 (400). TypedDicts ignore
# unknown keys, so pydantic alone would let these through to a paid call.
_FORBIDDEN_PARAMS = {
    "fallbacks": "the `fallbacks` parameter is rejected on the Batches API",
    "temperature": "non-default sampling params return 400 on claude-sonnet-5",
    "top_p": "non-default sampling params return 400 on claude-sonnet-5",
    "top_k": "non-default sampling params return 400 on claude-sonnet-5",
}


def _validate_request(req: Request) -> None:
    """Validate against the INSTALLED SDK types, not against our memory of them.

    Three layers, because no single one is sufficient:
      1. pydantic over the SDK's own Request TypedDict — real structural
         validation (catches a missing max_tokens, a malformed message).
      2. an unknown-key check — TypedDicts silently ignore extra keys, so
         pydantic would happily pass `fallbacks` straight through to a paid call.
      3. Sonnet-5-specific guards — `thinking` must be explicitly disabled, and
         budget_tokens / sampling params must be absent.
    """
    _REQUEST_ADAPTER.validate_python(req)

    params = dict(req["params"])
    for key, why in _FORBIDDEN_PARAMS.items():
        if key in params:
            raise ValueError(f"{req['custom_id']}: `{key}` must not be set — {why}")
    unknown = set(params) - _PARAM_KEYS
    if unknown:
        raise ValueError(f"{req['custom_id']}: unknown request params {sorted(unknown)}")

    for required in ("model", "max_tokens", "messages"):
        if required not in params:
            raise ValueError(f"{req['custom_id']}: missing required param `{required}`")

    thinking = params.get("thinking")
    if params["model"] == MODEL and (thinking or {}).get("type") != "disabled":
        raise ValueError(
            f"{req['custom_id']}: on {MODEL}, thinking must be EXPLICITLY disabled — "
            "omitting it runs adaptive thinking and bills thinking tokens on every request"
        )
    if "budget_tokens" in (thinking or {}):
        raise ValueError(f"{req['custom_id']}: `budget_tokens` is removed on {MODEL} (400)")


# --------------------------------------------------------------------------
# run directory
# --------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    return ann.RUNS_DIR / run_id


def _write_run(run_id: str, requests: list[Request], index: dict) -> Path:
    d = _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    jsonl = d / "requests.jsonl"
    with jsonl.open("w") as fh:
        for req in requests:
            fh.write(json.dumps(req) + "\n")
    (d / "requests_index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return jsonl


def _read_state(run_id: str) -> dict:
    path = _run_dir(run_id) / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _write_state(run_id: str, **updates) -> dict:
    state = _read_state(run_id) | updates
    _run_dir(run_id).mkdir(parents=True, exist_ok=True)
    (_run_dir(run_id) / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True))
    return state


def _load_index(run_id: str) -> dict:
    return json.loads((_run_dir(run_id) / "requests_index.json").read_text())


def _specs(names: list[str] | None) -> list[FieldSpec]:
    specs = [FIELD_SPECS[n] for n in (names or list(FIELD_SPECS))]
    for spec in specs:  # re-checked here, so no path to a paid call skips it
        _validate_spec(spec)
    return specs


def _prepare(
    specs: list[FieldSpec],
    limit: int | None,
    speeches: pd.DataFrame | None = None,
    paragraphs: pd.DataFrame | None = None,
    chunk_size: int | None = None,
) -> tuple[list[Request], dict, pd.DataFrame, pd.DataFrame]:
    """Build every spec's requests, optionally capped at the first `limit`
    speeches, optionally splitting each paragraph-unit speech into `chunk_size`
    chunks. A caller that already holds the corpus (submit, which fingerprints
    the full frame first) can pass it in rather than reload it."""
    if speeches is None:
        speeches = load()
    if paragraphs is None:
        paragraphs = pd.read_parquet(ann.PARAGRAPHS_PATH)
    if limit:
        speeches = speeches.head(limit)
        paragraphs = paragraphs[paragraphs["doc_name"].isin(set(speeches["doc_name"]))]

    requests: list[Request] = []
    index: dict = {}
    for spec in specs:
        reqs, idx = build_requests(spec, speeches, paragraphs, chunk_size=chunk_size)
        requests += reqs
        index |= idx
    return requests, index, speeches, paragraphs


def _pilot_speeches(
    speeches: pd.DataFrame, n: int = PILOT_N, random_state: int = PILOT_RANDOM_STATE
) -> pd.DataFrame:
    """The deterministic era-stratified pilot sample: exactly `n` speeches, one
    per equal-frequency chronological stratum, drawn with a fixed seed.

    Sorting by (date, doc_name) makes the stratum assignment and the within-
    stratum draw stable regardless of the frame's incoming row order — the same
    20 speeches come out every run (Washington's era at one end, the most recent
    at the other)."""
    df = speeches.sort_values(["date", "doc_name"]).reset_index(drop=True)
    total = len(df)
    if total <= n:
        return df
    # Position-based equal-frequency strata: row i (0-indexed, chronological)
    # falls in stratum (i*n)//total, which yields exactly n strata of ~equal size.
    df = df.assign(_era_bin=(pd.Series(range(total)) * n) // total)
    picked = df.groupby("_era_bin").sample(n=1, random_state=random_state)
    return (
        picked.drop(columns="_era_bin")
        .sort_values(["date", "doc_name"])
        .reset_index(drop=True)
    )


def _check_batch_limits(n_requests: int, jsonl_bytes: int) -> None:
    """Hard API limits. Enforced on BOTH paths — a check that only runs in
    dry-run protects nothing, since dry-run is not what spends the money."""
    if n_requests > BATCH_MAX_REQUESTS:
        raise ValueError(f"{n_requests} requests exceeds the {BATCH_MAX_REQUESTS} batch limit")
    if jsonl_bytes > BATCH_MAX_BYTES:
        raise ValueError(f"{jsonl_bytes} bytes exceeds the {BATCH_MAX_BYTES} batch limit")


def _estimate_tokens(
    requests: list[Request], index: dict, specs: list[FieldSpec]
) -> tuple[int, int]:
    """Offline estimate: ~4 chars/token. Deliberately NOT tiktoken (that is
    OpenAI's tokenizer and is simply wrong for Claude). Shared by dry-run and by
    submit's cost ceiling, so the number that gates the paid call is the same
    number dry-run printed."""
    est_in = sum(len(json.dumps(r["params"]["messages"])) for r in requests) // 4
    # Each request carries exactly ONE spec's rubric (its cached system block).
    # Attribute that rubric to the request by its own spec — summing every spec's
    # rubric across every request would inflate ~Nx for N specs. Over-estimate is
    # fine (keeps --max-cost-usd conservative); this ignores the ephemeral-cache
    # discount on the shared rubric, which only makes the estimate safer.
    spec_by_name = {s.name: s for s in specs}
    est_in += sum(
        len(spec_by_name[index[r["custom_id"]]["spec"]].rubric) // 4 for r in requests
    )
    est_out = 0
    for req in requests:
        meta = index[req["custom_id"]]
        spec = spec_by_name[meta["spec"]]
        est_out += (
            len(meta["para_idxs"]) * spec.est_output_tokens_per_para
            if meta["unit"] == "paragraph"
            else _SPEECH_EST_OUTPUT_TOKENS
        )
    return int(est_in), int(est_out)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_dry_run(args) -> None:
    """Build and validate the batch WITHOUT touching the network. No client is
    constructed, so this works with no credentials at all."""
    specs = _specs(args.spec)
    pilot = getattr(args, "pilot", False)
    run_id = args.run_id or f"{date.today().isoformat()}-dryrun{'-pilot' if pilot else ''}"
    chunk_size = getattr(args, "chunk_size", None)
    if pilot:
        # Same deterministic 20 speeches a pilot submit would send.
        sample = _pilot_speeches(load())
        paragraphs_all = pd.read_parquet(ann.PARAGRAPHS_PATH)
        paragraphs_sel = paragraphs_all[paragraphs_all["doc_name"].isin(set(sample["doc_name"]))]
        requests, index, speeches, paragraphs = _prepare(
            specs, None, sample, paragraphs_sel, chunk_size=chunk_size)
    else:
        requests, index, speeches, paragraphs = _prepare(specs, args.limit, chunk_size=chunk_size)

    for req in requests:
        _validate_request(req)

    jsonl = _write_run(run_id, requests, index)
    size = jsonl.stat().st_size
    _check_batch_limits(len(requests), size)

    # For an exact count, --count-tokens calls the API's own count_tokens
    # endpoint — a network call, but a free one, so it stays opt-in and dry-run
    # stays $0 by default.
    est_in, est_out = _estimate_tokens(requests, index, specs)
    method = "offline heuristic (~4 chars/token)"

    if args.count_tokens:
        import anthropic  # noqa: PLC0415 — only imported on the network path

        client = anthropic.Anthropic()
        est_in = 0
        for req in requests:
            p = req["params"]
            est_in += client.messages.count_tokens(
                model=p["model"], system=p["system"], messages=p["messages"]
            ).input_tokens
        method = "messages.count_tokens (exact input)"

    estimate = {
        "run_id": run_id,
        "model": MODEL,
        "n_requests": len(requests),
        "n_speeches": int(len(speeches)),
        "n_paragraphs": int(len(paragraphs)),
        "jsonl_bytes": size,
        "estimated_input_tokens": int(est_in),
        "estimated_output_tokens": int(est_out),
        "estimate_method": method,
        "estimated_cost_usd": round(cost_usd(int(est_in), int(est_out)), 2),
        "intro_pricing_applies": date.today().isoformat() <= INTRO_PRICING_ENDS,
        "specs": {s.name: {"prompt_version": s.prompt_version,
                           "prompt_hash": s.prompt_hash()} for s in specs},
    }
    (_run_dir(run_id) / "estimate.json").write_text(json.dumps(estimate, indent=2))

    print(f"DRY RUN — no network calls, no cost. run_id={run_id}"
          + ("  [PILOT: 20 era-stratified speeches]" if pilot else "")
          + (f"  [CHUNKED: <= {chunk_size} paras/request]" if chunk_size else ""))
    print(f"  specs:      {', '.join(s.name for s in specs)}")
    n_chunked = sum(1 for m in index.values() if m.get("chunk") is not None)
    per = ("1 per speech, all paragraph fields grouped" if not chunk_size
           else f"chunked: {n_chunked} chunk requests across {len({m['doc_name'] for m in index.values()})} speeches")
    print(f"  requests:   {len(requests)} ({per})")
    print(f"  paragraphs: {len(paragraphs):,}")
    print(f"  jsonl:      {jsonl} ({size:,} bytes)")
    print(f"  validated:  {len(requests)}/{len(requests)} against installed SDK types")
    print(f"  est tokens: {est_in:,} in / {est_out:,} out  [{method}]")
    print(f"  EST COST:   ${estimate['estimated_cost_usd']:,.2f} "
          f"(batch -50%, intro pricing {'ON' if estimate['intro_pricing_applies'] else 'OFF'})")


def cmd_submit(args) -> None:
    import anthropic  # noqa: PLC0415 — only imported on the network path

    run_id = args.run_id
    state = _read_state(run_id)
    if state.get("batch_id") and not args.force:
        raise SystemExit(
            f"run {run_id} already has batch_id={state['batch_id']}. "
            "Use `status`/`ingest`, or pass --force to submit a new batch."
        )

    specs = _specs(args.spec)
    # Load the corpus ONCE. The fingerprint must cover the FULL corpus even under
    # --limit, so it is taken here, before `_prepare` caps the frames; the same
    # loaded frames are threaded into `_prepare` rather than reloaded.
    speeches = load()
    paragraphs = pd.read_parquet(ann.PARAGRAPHS_PATH)
    # The fingerprint always covers the FULL corpus, even under --limit/--pilot,
    # so it is taken before the frame is narrowed.
    fingerprint = ann.corpus_fingerprint(speeches=speeches, paragraphs=paragraphs)
    chunk_size = getattr(args, "chunk_size", None)
    if getattr(args, "pilot", False):
        speeches_sel = _pilot_speeches(speeches)
        paragraphs_sel = paragraphs[paragraphs["doc_name"].isin(set(speeches_sel["doc_name"]))]
        requests, index, _, paragraphs = _prepare(
            specs, None, speeches_sel, paragraphs_sel, chunk_size=chunk_size)
    else:
        requests, index, _, paragraphs = _prepare(
            specs, args.limit, speeches, paragraphs, chunk_size=chunk_size)

    # Resume: skip speeches this spec has already FULLY annotated, so a retry
    # after a partial failure does not pay for them twice. Per spec, not pooled:
    # a speech annotated for spec A is not annotated for spec B.
    done = _already_ingested(specs, paragraphs)
    kept = [r for r in requests
            if index[r["custom_id"]]["doc_name"] not in done[index[r["custom_id"]]["spec"]]]
    if len(kept) < len(requests):
        for spec in specs:
            if done[spec.name]:
                print(f"resume: {spec.name}: skipping {len(done[spec.name])} "
                      "fully-annotated speeches")
        requests = kept

    # Terminally sealed permanent failures (invalid_request) must not be
    # re-requested and RE-PAID for on every resume. A deterministic 400 will
    # fail again identically, so ingest records its doc_name in state and submit
    # drops it here unless --resubmit-sealed is passed (after a human fix).
    sealed = set(state.get("sealed_permanent") or [])
    if sealed and not getattr(args, "resubmit_sealed", False):
        before = len(requests)
        requests = [r for r in requests if index[r["custom_id"]]["doc_name"] not in sealed]
        if len(requests) < before:
            print(f"sealed: skipping {before - len(requests)} request(s) for "
                  f"{len(sealed)} permanently-failed speech(es) (invalid_request); "
                  "pass --resubmit-sealed to force after fixing the request")

    if not requests:
        print("nothing to submit — every requested speech is already fully annotated "
              "(or permanently sealed)")
        return

    for req in requests:
        _validate_request(req)

    # Persist ONLY the submitted requests' index entries. build_requests emits an
    # entry per speech, but the resume + sealed filters trimmed `requests`; passing
    # the FULL index would (in a --chunk-size round, where every entry carries a
    # `chunk` marker) make cmd_ingest and _all_chunk_escalated_docs() name ~every
    # speech chunk-escalated in the amendment provenance instead of only the
    # stragglers actually submitted. Trim it to the requests that survived.
    index = {r["custom_id"]: index[r["custom_id"]] for r in requests}
    jsonl = _write_run(run_id, requests, index)
    size = jsonl.stat().st_size
    _check_batch_limits(len(requests), size)

    est_in, est_out = _estimate_tokens(requests, index, specs)
    estimate = cost_usd(est_in, est_out)
    print(f"about to submit {len(requests)} requests ({size:,} bytes)")
    print(f"  est tokens: {est_in:,} in / {est_out:,} out  [offline heuristic]")
    print(f"  EST COST:   ${estimate:,.2f}  (ceiling ${args.max_cost_usd:,.2f})")

    # Two independent brakes on the one command that spends money. Neither exists
    # in dry-run, which is exactly why they have to exist here: dry-run is not
    # what gets typed by mistake.
    if estimate > args.max_cost_usd:
        raise SystemExit(
            f"REFUSING TO SUBMIT: estimated ${estimate:,.2f} exceeds --max-cost-usd "
            f"${args.max_cost_usd:,.2f}. Check --spec/--limit, or raise the ceiling deliberately."
        )
    if not args.yes:
        raise SystemExit(
            f"REFUSING TO SUBMIT without --yes: this call SPENDS MONEY (est ${estimate:,.2f}). "
            f"Re-run with --yes once the dry-run for {run_id} looks right."
        )

    client = anthropic.Anthropic()  # zero-arg: let the SDK resolve credentials
    batch = client.messages.batches.create(requests=requests)

    # A new batch invalidates any results cached from the previous one. Deleting
    # the file here means a `--force` resubmit can never leave batch A's results
    # sitting next to batch B's id; the results_batch_id check in ingest is the
    # second line of defence.
    (_run_dir(run_id) / "results.jsonl").unlink(missing_ok=True)

    _write_state(
        run_id,
        batch_id=batch.id,
        results_batch_id=None,
        model=MODEL,
        submitted=date.today().isoformat(),
        n_requests=len(requests),
        # Every custom_id actually sent. Ingest asserts each one comes back — a
        # truncated results cache would otherwise parse cleanly and short a run.
        submitted_custom_ids=sorted(r["custom_id"] for r in requests),
        specs=[s.name for s in specs],
        corpus_fingerprint=fingerprint,
    )
    print(f"submitted batch {batch.id} ({len(requests)} requests), status={batch.processing_status}")


def cmd_status(args) -> None:
    import anthropic  # noqa: PLC0415

    state = _read_state(args.run_id)
    batch_id = state.get("batch_id")
    if not batch_id:
        raise SystemExit(f"run {args.run_id} has no batch_id — has it been submitted?")

    batch = anthropic.Anthropic().messages.batches.retrieve(batch_id)
    c = batch.request_counts
    print(f"batch {batch_id}: {batch.processing_status}")
    print(f"  processing={c.processing} succeeded={c.succeeded} errored={c.errored} "
          f"canceled={c.canceled} expired={c.expired}")
    if batch.processing_status == "ended":
        print(f"  ready to ingest: pp-annotate ingest --run-id {args.run_id}")


def _already_ingested(
    specs: list[FieldSpec], paragraphs: pd.DataFrame
) -> dict[str, set[str]]:
    """Per spec, the speeches that are FULLY annotated on disk.

    Coverage, not presence. A model that drops items from a long array leaves a
    half-annotated speech behind; keying `done` on the doc_name alone would mark
    that speech finished forever and seal the gap in permanently. A
    paragraph-unit speech is done only when every para_idx the corpus has for it
    has a row. Keyed per spec, since a speech annotated for spec A has not been
    annotated for spec B."""
    expected = {d: set(map(int, idxs))
                for d, idxs in paragraphs.groupby("doc_name")["para_idx"]}
    done: dict[str, set[str]] = {}
    for spec in specs:
        path = ann.annotation_path(spec.name)
        if not path.exists():
            done[spec.name] = set()
            continue
        df = pd.read_parquet(path)
        if spec.unit == "speech":
            done[spec.name] = set(df["doc_name"])
            continue
        have = {d: set(map(int, idxs)) for d, idxs in df.groupby("doc_name")["para_idx"]}
        done[spec.name] = {d for d, idxs in have.items() if expected.get(d, set()) <= idxs}
    return done


def _fetch_results(run_id: str, batch_id: str, raw_path: Path) -> None:
    """Stream results to a sibling .tmp and rename it into place only once the
    stream has completed.

    An interrupted fetch must leave NO usable cache behind. `batches.results()`
    raises if the batch has not ended — a likely first-contact mistake — and
    truncating results.jsonl in place would leave a 0-byte file that the next
    ingest happily accepts, parses to zero rows, and records as a $0.00 run."""
    import anthropic  # noqa: PLC0415 — only imported on the network path

    tmp = raw_path.with_name(raw_path.name + ".tmp")
    n = 0
    try:
        with tmp.open("w") as fh:
            for result in anthropic.Anthropic().messages.batches.results(batch_id):
                fh.write(result.to_json(indent=None) + "\n")
                n += 1
        if n == 0:
            raise SystemExit(
                f"batch {batch_id} returned zero results — refusing to cache an empty file"
            )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, raw_path)  # atomic: the cache appears whole or not at all
    # Written only after the rename. A crash in between leaves results_batch_id
    # unset, so the next ingest treats the cache as stale and refetches — the
    # safe direction.
    _write_state(run_id, results_batch_id=batch_id)
    print(f"fetched {n} raw results -> {raw_path}")


# The billing-block message the Batches API returns for a temporarily unfunded
# account, verbatim: "Your credit balance is too low to access the Anthropic
# API. Please go to Plans & Billing...". It arrives under the SAME error type as
# a genuine validation failure (the API's `invalid_request_error`, and the older
# `invalid_request`), so the two MUST be split by message, not type. The marker
# is kept narrow and case-insensitive: "credit balance".
_BILLING_BLOCK_MARKER = "credit balance"


def _classify_failure(err_type: str | None, err_message: str | None, rtype: str) -> tuple[str, str]:
    """Classify a non-succeeded batch result as ('permanent', reason) or
    ('retryable', reason).

    `invalid_request` / `invalid_request_error` is the one type that covers two
    OPPOSITE cases (both seen verbatim in the full run):
      * a TRANSIENT billing block (credit balance too low) — must stay retryable,
        never sealed, so it re-requests once the account is funded;
      * a DETERMINISTIC malformed request — must seal, so we do not re-pay for a
        request that will 400 identically forever.
    They are distinguished by the narrow, documented "credit balance" marker in
    the message. Every other error type (overloaded, expired, ...) is retryable.
    """
    if err_type in ("invalid_request", "invalid_request_error"):
        if _BILLING_BLOCK_MARKER in (err_message or "").lower():
            return "retryable", "billing_blocked"
        return "permanent", err_type
    return "retryable", err_type or rtype


def cmd_ingest(args) -> None:
    run_id = args.run_id
    state = _read_state(run_id)
    index = _load_index(run_id)
    raw_path = _run_dir(run_id) / "results.jsonl"

    batch_id = state.get("batch_id")
    if not batch_id:
        raise SystemExit(f"run {run_id} has no batch_id — has it been submitted?")

    # Raw results are cached on disk so ingest is re-runnable without re-hitting
    # the API. The cache is valid ONLY for the batch that produced it: `submit
    # --force` creates a new batch, and pairing its id with the previous batch's
    # cached tokens and cost would write a manifest that lies about both.
    cached_batch_id = state.get("results_batch_id")
    stale = raw_path.exists() and cached_batch_id != batch_id
    if stale:
        print(f"cached results are from batch {cached_batch_id or '(unrecorded)'}, "
              f"not {batch_id} — refetching")
    if args.refetch or stale or not raw_path.exists():
        _fetch_results(run_id, batch_id, raw_path)

    lines = [ln for ln in raw_path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(
            f"{raw_path} holds zero results for submitted batch {batch_id}. A manifest "
            "recording n_requests=0 and cost_usd=0.00 for a batch that was paid for is a "
            "false provenance record — refusing to write one. Re-run with --refetch."
        )

    rows: dict[str, list[dict]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    permanent_failures, retryable_failures, incomplete = [], [], []
    seen: set[str] = set()
    n_sent = n_returned = 0

    for line in lines:
        result = json.loads(line)
        # KEY BY custom_id, NEVER BY POSITION. Batch results arrive in arbitrary
        # order; zipping them against the request list would silently mislabel
        # every speech. This lookup is the only anchor.
        cid = result["custom_id"]
        meta = index.get(cid)
        if meta is None:
            raise ValueError(f"result {cid} is not in this run's request index")
        if cid in seen:
            raise ValueError(f"result {cid} appears twice in {raw_path}")
        seen.add(cid)

        rtype = result["result"]["type"]
        counts[rtype] = counts.get(rtype, 0) + 1
        if rtype != "succeeded":
            err = result["result"].get("error") or {}
            # invalid_request* covers BOTH a transient billing block (retryable,
            # never sealed) and a genuinely malformed request (permanent, sealed);
            # they share a type and are split by message. Every other error type
            # is retryable. See _classify_failure.
            kind, reason = _classify_failure(err.get("type"), err.get("message"), rtype)
            (permanent_failures if kind == "permanent"
             else retryable_failures).append((cid, meta["doc_name"], reason))
            continue

        message = result["result"]["message"]
        for k in usage:
            usage[k] += message.get("usage", {}).get(k) or 0

        # Everything from here to row assembly reads MODEL-SUPPLIED shape
        # (content blocks, the parsed payload, per-item fields). A JSON-valid but
        # shape-unexpected response — {"annotations": null}, a missing field, a
        # non-int para_idx, content that isn't a list of text blocks — would raise
        # KeyError/TypeError/ValueError and abort the whole paid batch, violating
        # the "one anomalous response never aborts the batch" invariant. So the
        # block is guarded: any such shape fault quarantines just this speech as a
        # retryable "malformed_payload". The specific reasons below (no_text_block,
        # malformed_json, unsent/duplicate para_idx) route to their own labels
        # first — they `continue` before the guard can generalize them.
        try:
            # A "succeeded" result can still carry no text block — a refusal, or an
            # all-thinking response. Its tokens were billed (already folded into
            # usage above), but there is nothing to parse. Record it like a
            # retryable failure — no rows, so the speech is not sealed done and the
            # next `submit` re-requests it.
            text = next((b["text"] for b in message["content"] if b["type"] == "text"), None)
            if text is None:
                retryable_failures.append((cid, meta["doc_name"], "no_text_block"))
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # output_config.format guarantees valid JSON only for a COMPLETE
                # response. A response truncated against max_tokens (long structured
                # array) arrives as syntactically incomplete JSON — cut mid-object.
                # Quarantine like any other anomaly; applies to both units.
                retryable_failures.append((cid, meta["doc_name"], "malformed_json"))
                continue

            if meta["unit"] == "speech":
                # One object for the whole speech: no para_idx, keyed by doc_name.
                # Trusted keys go LAST so a model-returned doc_name/run_id in the
                # payload can never clobber the index-anchored ones.
                spec_rows = [{**payload, "doc_name": meta["doc_name"], "run_id": run_id}]
                missing: list[int] = []
            else:
                valid_idxs = set(meta["para_idxs"])
                returned: set[int] = set()
                spec_rows = []
                quarantine: str | None = None  # set -> discard this speech's rows
                ent_field = FIELD_SPECS[meta["spec"]].entity_field
                for item in payload["annotations"]:
                    para_idx = int(item["para_idx"])
                    # The model echoes para_idx; trust it only after checking it
                    # against what we actually sent. A hallucinated or duplicated
                    # index must never become a key — but ONE bad response must not
                    # abort ingest of the whole (paid) batch. So we QUARANTINE this
                    # speech: record it like a retryable failure, write NO rows for
                    # it, and keep ingesting every other speech. No rows means it is
                    # not sealed done, so the next `submit` re-requests it.
                    if para_idx not in valid_idxs:
                        quarantine = "unsent_para_idx"
                        break
                    if para_idx in returned:
                        quarantine = "duplicate_para_idx"
                        break
                    returned.add(para_idx)
                    # The entity field is exploded downstream via `e.get(...)`, which
                    # would raise AttributeError on a schema-violating element
                    # ([null], ["China"]) — OUTSIDE this guard. Validate it here so a
                    # bad element becomes a "malformed_payload" quarantine like every
                    # other shape fault, instead of aborting the whole batch ingest.
                    if ent_field is not None:
                        ents = item.get(ent_field)
                        if ents is not None and (
                            not isinstance(ents, list)
                            or any(not isinstance(e, dict) for e in ents)
                        ):
                            raise ValueError(f"{ent_field} must be a list of objects")
                    # Trusted keys LAST so a model-returned doc_name/run_id in the
                    # item can never clobber the index-anchored ones (para_idx is
                    # already excluded from `row`).
                    row = {k: v for k, v in item.items() if k != "para_idx"}
                    spec_rows.append(
                        {**row, "doc_name": meta["doc_name"], "para_idx": para_idx,
                         "run_id": run_id}
                    )
                if quarantine is not None:
                    retryable_failures.append((cid, meta["doc_name"], quarantine))
                    continue  # next result — the integrity fault becomes no key at all
                # `returned ⊆ sent` is enforced above; this is the other direction.
                # Dropping items from a long array is the standard failure mode of
                # this prompt shape, and it arrives as silence — no error, just
                # paragraphs that never appear. The rows that DID come back are kept
                # (correctly keyed, and paid for), but the speech is recorded as
                # incomplete and `_already_ingested` will not call it done.
                missing = sorted(valid_idxs - returned)
                n_sent += len(valid_idxs)
                n_returned += len(returned)
        except (KeyError, TypeError, ValueError):
            retryable_failures.append((cid, meta["doc_name"], "malformed_payload"))
            continue

        if missing:
            incomplete.append((cid, meta["doc_name"], missing))
        rows.setdefault(meta["spec"], []).extend(spec_rows)

    # Every request that was sent must come back — otherwise a truncated cache
    # parses cleanly and silently shorts the run.
    submitted = set(state.get("submitted_custom_ids") or [])
    if submitted and (lost := submitted - seen):
        raise SystemExit(
            f"{len(lost)} of {len(submitted)} submitted requests have no result line in "
            f"{raw_path} (e.g. {sorted(lost)[:3]}). The cache is truncated or belongs to "
            "another batch — re-run with --refetch."
        )
    if not submitted and state.get("n_requests", 0) > len(seen):
        raise SystemExit(
            f"{raw_path} holds {len(seen)} results but state records "
            f"{state['n_requests']} submitted requests — the cache is truncated. "
            "Re-run with --refetch."
        )

    # --- accumulate spend across batches (resume-manifest-accumulation) -------
    # A `submit --force` under an existing run_id creates a NEW batch; ingesting
    # each batch must ADD to the run's recorded spend, never overwrite it with
    # only the last batch's numbers — the layer's invariant is that a manifest
    # can never claim a run was cheaper than it was. We scope per batch_id in
    # state and re-derive the totals from every batch ever ingested, so a
    # re-ingest of the SAME batch is idempotent (its entry is replaced, not added).
    batch_usage = dict(state.get("batch_usage") or {})
    batch_usage[batch_id] = {**usage, "n_requests": sum(counts.values())}

    # Seal permanent (invalid_request) failures so the next `submit` does not
    # re-request and re-pay for a deterministically failing request. Kept as
    # doc_names in state; the submit resume filter drops them unless overridden.
    perm_docs = sorted({doc for _cid, doc, _et in permanent_failures})
    sealed = sorted(set(state.get("sealed_permanent") or []) | set(perm_docs))
    _write_state(run_id, batch_usage=batch_usage, sealed_permanent=sealed)

    total_usage = {k: 0 for k in usage}
    total_requests = 0
    for _bid, b in batch_usage.items():
        for k in total_usage:
            total_usage[k] += int(b.get(k, 0))
        total_requests += int(b.get("n_requests", 0))
    actual_cost = cost_usd(**total_usage)
    all_batch_ids = ",".join(sorted(batch_usage))

    written = []
    entity_rows: list[dict] = []
    entity_touched: list[tuple] = []  # every (doc_name, para_idx) an entity spec annotated
    entity_specs_present = False
    for spec_name, spec_rows in rows.items():
        spec = FIELD_SPECS[spec_name]
        if spec.entity_field:
            # Explode the nested per-paragraph entity array into its own long-
            # format table, keyed by (doc_name, para_idx). Popping the field
            # keeps the paragraph table flat; the entity table is many-per-key,
            # so it cannot use the unique-key writer (see _write_entity_table).
            # Track EVERY annotated paragraph (not just those with entities) so a
            # re-annotation that now finds no entities clears its prior rows.
            entity_specs_present = True
            for r in spec_rows:
                entity_touched.append((r["doc_name"], r["para_idx"]))
                for e in (r.pop(spec.entity_field, None) or []):
                    entity_rows.append({
                        "doc_name": r["doc_name"], "para_idx": r["para_idx"],
                        "entity": e.get("name"), "type": e.get("type"),
                        "stance": e.get("stance"), "run_id": run_id,
                    })
        df = pd.DataFrame(spec_rows)
        path = ann.annotation_path(spec_name)
        if path.exists():
            # Merge with prior runs; rows from THIS run win on a key collision.
            key = ann.SPEECH_KEY if spec.unit == "speech" else ann.PARAGRAPH_KEY
            prior = pd.read_parquet(path)
            prior = prior.merge(df[key], on=key, how="left", indicator=True)
            prior = prior[prior["_merge"] == "left_only"].drop(columns="_merge")
            df = pd.concat([prior, df], ignore_index=True)
        written.append(ann.write_annotations(spec_name, df, spec.unit).name)

    if entity_specs_present:
        written.append(_write_entity_table(entity_rows, entity_touched, run_id))

    coverage = (
        f"paragraph coverage: {n_returned:,}/{n_sent:,} sent para_idxs returned"
        if n_sent
        else "paragraph coverage: n/a (no paragraph-unit spec in this run)"
    )
    if incomplete:
        coverage += (
            f"; {len(incomplete)} responses were INCOMPLETE (the model dropped "
            f"{n_sent - n_returned:,} paragraphs). Those speeches are NOT marked done and "
            "will be re-requested by the next `submit`."
        )

    # Amendment #2 provenance: which speeches this round was chunk-escalated
    # (judged with partial rather than whole-speech context). Derived from the
    # run's requests_index — a chunk request carries a `chunk` key.
    chunked_docs = sorted({m["doc_name"] for m in index.values() if m.get("chunk") is not None})
    if chunked_docs:
        coverage += (
            f" AMENDMENT: {len(chunked_docs)} chunk-escalated speech(es) judged with "
            f"partial (not whole-speech) context: {chunked_docs}."
        )

    # The billing-block subclass of retryable (invalid_request* with a credit-
    # balance message) — surfaced when present, so a resume round is legible as
    # "waiting on funding", not silent. Zero -> no added text (notes byte-stable).
    billing_blocked = sum(1 for _c, _d, reason in retryable_failures if reason == "billing_blocked")

    specs = _specs(state.get("specs"))
    ann.write_manifest(ann.Manifest(
        run_id=run_id,
        model=state.get("model", MODEL),
        prompt_version=",".join(f"{s.name}={s.prompt_version}" for s in specs),
        prompt_hash=",".join(f"{s.name}={s.prompt_hash()}" for s in specs),
        date=date.today().isoformat(),
        # Every batch whose spend is folded into the totals below (one id unless
        # a --force resume added more). Comma-joined so the manifest names them
        # all rather than only the last.
        batch_id=all_batch_ids,
        n_requests=total_requests,           # cumulative across ingested batches
        cost_usd=round(actual_cost, 4),      # ACTUAL, from cumulative usage — not the estimate
        corpus_fingerprint=state.get("corpus_fingerprint") or ann.corpus_fingerprint(),
        fields=[s.name for s in specs],
        annotation_files=written,
        notes=(
            f"succeeded={counts['succeeded']} errored={counts['errored']} "
            f"canceled={counts['canceled']} expired={counts['expired']}. "
            f"permanent (invalid_request) failures: {len(permanent_failures)}; "
            f"retryable failures: {len(retryable_failures)}"
            + (f" (billing-blocked: {billing_blocked}, transient)" if billing_blocked else "")
            + f". {coverage}"
            + (f" [cumulative across {len(batch_usage)} batches: "
               f"{sorted(batch_usage)}]" if len(batch_usage) > 1 else "")
        ),
        **total_usage,
    ))

    print(f"ingested run {run_id}" + (f" (batch {batch_id})" if len(batch_usage) > 1 else ""))
    for k, v in counts.items():
        print(f"  {k}: {v}  (this batch)")
    # Cumulative across every batch ingested for this run — matches the manifest.
    print(f"  tokens: {total_usage['input_tokens']:,} in / {total_usage['output_tokens']:,} out "
          f"(cache: {total_usage['cache_read_input_tokens']:,} read, "
          f"{total_usage['cache_creation_input_tokens']:,} write)"
          + ("  [cumulative]" if len(batch_usage) > 1 else ""))
    print(f"  ACTUAL COST: ${actual_cost:,.4f}"
          + ("  [cumulative across all batches]" if len(batch_usage) > 1 else ""))
    print(f"  wrote: {', '.join(written) or '(nothing)'}")
    if n_sent:
        print(f"  coverage: {n_returned:,}/{n_sent:,} sent paragraphs returned")
    if incomplete:
        print(f"  {len(incomplete)} INCOMPLETE responses — the model dropped "
              f"{n_sent - n_returned:,} paragraphs. These speeches are NOT done; "
              "rerun `submit` to re-request them:")
        for cid, doc, missing in incomplete[:5]:
            preview = missing[:8]
            more = "..." if len(missing) > len(preview) else ""
            print(f"    {cid} {doc} missing para_idx {preview}{more} ({len(missing)} of "
                  f"{len(index[cid]['para_idxs'])})")
    if permanent_failures:
        print(f"  {len(permanent_failures)} PERMANENT failures (do not retry, fix the request):")
        for cid, doc, etype in permanent_failures[:5]:
            print(f"    {cid} {doc} {etype}")
    if retryable_failures:
        print(f"  {len(retryable_failures)} retryable failures — rerun `submit` to fill them in")


# --------------------------------------------------------------------------
# entity table (long-format: many rows per paragraph key)
# --------------------------------------------------------------------------


def _write_entity_table(
    entity_rows: list[dict], touched_keys: list[tuple], run_id: str
) -> str:
    """Write/refresh paragraph_entities.parquet from exploded entity rows.

    The (doc_name, para_idx) key is deliberately NON-unique here — a paragraph
    can name several entities — so the unique-key writer does not apply. This
    run's entities SUPERSEDE any prior run's for the same paragraph: prior rows
    for every paragraph this run re-annotated are dropped (even paragraphs that
    now have zero entities, hence `touched_keys` rather than the entity rows'
    own keys), then this run's rows are concatenated. Keyed joins only — never
    row order."""
    cols = ["doc_name", "para_idx", "entity", "type", "stance", "run_id"]
    df = pd.DataFrame(entity_rows, columns=cols)
    path = ann.annotation_path("paragraph_entities")
    if path.exists() and touched_keys:
        prior = pd.read_parquet(path)
        touched = pd.DataFrame(touched_keys, columns=["doc_name", "para_idx"]).drop_duplicates()
        prior = prior.merge(touched, on=["doc_name", "para_idx"], how="left", indicator=True)
        prior = prior[prior["_merge"] == "left_only"].drop(columns="_merge")
        df = pd.concat([prior, df], ignore_index=True)
    ann.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values(["doc_name", "para_idx", "entity"]).reset_index(drop=True).to_parquet(
        path, index=False
    )
    return path.name


# --------------------------------------------------------------------------
# QA report
# --------------------------------------------------------------------------


def _run_failure_rate(run_id: str) -> float | None:
    """Schema/request failure rate from a run's cached raw results: the share of
    submitted requests that errored, were canceled/expired, or came back with no
    text block to parse. None if there is no cached results file yet."""
    raw = _run_dir(run_id) / "results.jsonl"
    if not raw.exists():
        return None
    lines = [ln for ln in raw.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    failed = 0
    for ln in lines:
        result = json.loads(ln)["result"]
        if result.get("type") != "succeeded":
            failed += 1
            continue
        content = result.get("message", {}).get("content", [])
        if not any(b.get("type") == "text" for b in content):
            failed += 1
    return failed / len(lines)


def _all_chunk_escalated_docs() -> list[str]:
    """Every speech chunk-escalated across the whole convergence, derived by
    scanning INGESTED runs' requests_index for cids that carry a `chunk` key.
    This is the amendment-#2 provenance source: it survives across resubmission
    rounds without threading any extra state.

    Scoped to runs whose run_id has a manifest (`ann.manifest_path` is written
    only by ingest, never by dry-run or a bare submit), so routine
    `dry-run --chunk-size` artifacts do not pollute the straggler list — the
    amendment names only speeches whose chunk data actually LANDED."""
    docs: set[str] = set()
    if not ann.RUNS_DIR.exists():
        return []
    for idx_file in ann.RUNS_DIR.glob("*/requests_index.json"):
        run_id = idx_file.parent.name
        if not ann.manifest_path(run_id).exists():
            continue  # dry-run / un-ingested run — its chunks never landed
        try:
            idx = json.loads(idx_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        docs.update(m["doc_name"] for m in idx.values() if m.get("chunk") is not None)
    return sorted(docs)


def _compute_qa(run_id: str | None, converged: bool = False) -> dict:
    """Compute the QA metrics over the ingested annotation tables. Pilot/full mode
    scopes to a run's speeches (when a run_id is given) and gates coverage at
    >=99%. CONVERGED mode is corpus-wide: it gates BOTH real specs at 100%
    coverage and turns the per-run schema-failure rate informational (a converged
    corpus is assembled across many rounds/caches). Returns metrics + gated checks
    (name, passed, detail); cmd_qa decides whether to enforce."""
    speeches = load()[["doc_name", "decade"]]
    corpus_paras = pd.read_parquet(ann.PARAGRAPHS_PATH)[["doc_name", "para_idx", "text"]]

    # Converged QA is corpus-wide by definition; a run_id only feeds the
    # informational schema-failure rate. Otherwise scope to the run's speeches.
    scope_docs = None
    if run_id and not converged and (idx_path := _run_dir(run_id) / "requests_index.json").exists():
        scope_docs = {m["doc_name"] for m in json.loads(idx_path.read_text()).values()}

    def _scope(df: pd.DataFrame) -> pd.DataFrame:
        return df if scope_docs is None else df[df["doc_name"].isin(scope_docs)]

    scope_paras = _scope(corpus_paras)
    metrics: dict = {"run_id": run_id, "converged": converged, "scope_doc_count": (
        len(scope_docs) if scope_docs is not None else int(corpus_paras["doc_name"].nunique()))}
    checks: list[tuple[str, bool, str]] = []

    # schema / request failure rate (from cached raw results, if present).
    # Gated in pilot/full; informational only in converged (multi-round caches).
    fail_rate = _run_failure_rate(run_id) if run_id else None
    metrics["schema_failure_rate"] = fail_rate
    if fail_rate is not None and not converged:
        checks.append(("schema_failure_rate<1%", fail_rate < QA_MAX_SCHEMA_FAILURE,
                       f"{fail_rate:.3%}"))

    if converged:
        metrics["chunk_escalated_docs"] = _all_chunk_escalated_docs()
        # speech_annotations coverage: every corpus speech must have a row.
        need_sp = set(speeches["doc_name"])
        sp_path = ann.annotation_path("speech_annotations")
        if sp_path.exists():
            have_sp = set(pd.read_parquet(sp_path)["doc_name"]) & need_sp
            sp_cov = (len(have_sp) / len(need_sp)) if need_sp else 0.0
            metrics["speech_coverage"] = {"expected": len(need_sp), "have": len(have_sp),
                                          "rate": round(sp_cov, 4)}
            checks.append(("speech_coverage==100%", bool(need_sp) and len(have_sp) == len(need_sp),
                           f"{sp_cov:.2%} ({len(have_sp):,}/{len(need_sp):,})"))
        else:
            checks.append(("speech_coverage==100%", False, "speech_annotations absent"))

    pj_path = ann.annotation_path("paragraph_annotations")
    if not pj_path.exists():
        if converged:
            checks.append(("paragraph_coverage==100%", False, "paragraph_annotations absent"))
        metrics["paragraph_annotations"] = "absent"
        return {"metrics": metrics, "checks": checks, "by_decade": {}}

    pj = _scope(pd.read_parquet(pj_path))
    if converged and "run_id" in pj.columns:
        metrics["n_rounds"] = int(pj["run_id"].nunique())  # rounds that contributed rows

    # paragraph coverage against the corpus (in scope)
    corpus_keys = scope_paras[["doc_name", "para_idx"]].drop_duplicates()
    have_keys = pj[["doc_name", "para_idx"]].drop_duplicates()
    n_expected, n_have = len(corpus_keys), len(have_keys)
    coverage = (n_have / n_expected) if n_expected else 0.0
    metrics["paragraph_coverage"] = {"expected": n_expected, "have": n_have,
                                     "rate": round(coverage, 4)}
    if converged:
        checks.append(("paragraph_coverage==100%", n_have == n_expected and n_expected > 0,
                       f"{coverage:.2%} ({n_have:,}/{n_expected:,})"))
    else:
        checks.append(("paragraph_coverage>=99%", coverage >= QA_MIN_COVERAGE,
                       f"{coverage:.2%} ({n_have:,}/{n_expected:,})"))

    # per-flag corpus-wide rates + degeneracy gate
    flag_rates = {}
    for flag in _JUDGMENT_FLAGS:
        if flag not in pj.columns:
            continue
        rate = float(pj[flag].mean()) if len(pj) else 0.0
        flag_rates[flag] = round(rate, 4)
        degenerate = rate > QA_FLAG_DEGENERATE_HIGH or rate < QA_FLAG_DEGENERATE_LOW
        checks.append((f"{flag}_non_degenerate", not degenerate,
                       f"{rate:.2%}" + (" DEGENERATE" if degenerate else "")))
    metrics["flag_rates"] = flag_rates

    # flag co-occurrence (how often flags fire together)
    if all(f in pj.columns for f in _JUDGMENT_FLAGS) and len(pj):
        combo = pj[list(_JUDGMENT_FLAGS)].astype(bool)
        metrics["flag_cooccurrence"] = {
            "any_flag": round(float(combo.any(axis=1).mean()), 4),
            "party+enemy": round(float((combo["party_attack"] & combo["enemy_naming"]).mean()), 4),
            "enemy+zero_sum": round(float((combo["enemy_naming"] & combo["zero_sum"]).mean()), 4),
            "all_three": round(float(combo.all(axis=1).mean()), 4),
        }

    # unlabeled-paragraph share (empty topics == "none")
    if "topics" in pj.columns and len(pj):
        unlabeled = float((pj["topics"].apply(lambda t: len(t) == 0)).mean())
        metrics["unlabeled_share"] = round(unlabeled, 4)

    # label rates by decade (top topics per decade, for the report)
    by_decade: dict = {}
    if "topics" in pj.columns and len(pj):
        joined = pj.merge(speeches, on="doc_name", how="left", validate="many_to_one")
        exploded = joined.explode("topics").dropna(subset=["topics"])
        for decade, grp in exploded.groupby("decade"):
            top = grp["topics"].value_counts().head(5)
            by_decade[int(decade)] = {name: int(c) for name, c in top.items()}

    # entities: substring check + enemy-naming <-> adversarial-entity consistency
    ent_path = ann.annotation_path("paragraph_entities")
    if ent_path.exists():
        ent = _scope(pd.read_parquet(ent_path))
        if len(ent):
            merged = ent.merge(scope_paras, on=["doc_name", "para_idx"],
                               how="left", validate="many_to_one")
            has_text = merged.dropna(subset=["text"])
            hits = has_text.apply(
                lambda r: bool(r["entity"]) and str(r["entity"]).lower() in str(r["text"]).lower(),
                axis=1,
            )
            substr_rate = float(hits.mean()) if len(has_text) else 0.0
            metrics["entity_substring_rate"] = round(substr_rate, 4)
            metrics["entity_count"] = int(len(ent))
            checks.append(("entity_name_in_paragraph>=80%", substr_rate >= QA_MIN_ENTITY_SUBSTRING,
                           f"{substr_rate:.2%}"))

            # consistency: share of enemy_naming paragraphs with >=1 adversarial entity
            if "enemy_naming" in pj.columns:
                adversarial = set(map(tuple, ent[ent["stance"] == "adversarial"]
                                      [["doc_name", "para_idx"]].drop_duplicates().to_numpy()))
                enemy = pj[pj["enemy_naming"]]
                if len(enemy):
                    with_adv = sum(1 for r in enemy[["doc_name", "para_idx"]].itertuples(index=False)
                                   if (r.doc_name, r.para_idx) in adversarial)
                    metrics["enemy_naming_adversarial_consistency"] = {
                        "enemy_naming_paras": int(len(enemy)),
                        "with_adversarial_entity": int(with_adv),
                        "rate": round(with_adv / len(enemy), 4),  # reported, not gated
                    }

    return {"metrics": metrics, "checks": checks, "by_decade": by_decade}


def _qa_markdown(result: dict, pilot: bool, converged: bool = False) -> str:
    m, checks, by_decade = result["metrics"], result["checks"], result["by_decade"]
    mode = ("CONVERGED (post-convergence gate)" if converged
            else "PILOT (gated)" if pilot else "full (reported)")
    lines = ["# Annotation QA — v1", ""]
    lines.append(f"- mode: {mode}")
    lines.append(f"- run_id: {m.get('run_id')}")
    lines.append(f"- scope: {m.get('scope_doc_count')} speeches"
                 + (" (corpus-wide)" if converged else ""))

    if converged:
        # Amendment #1 + #2 recorded explicitly in the converged report.
        n_rounds = m.get("n_rounds")
        chunked = m.get("chunk_escalated_docs") or []
        straggler_line = (f"{len(chunked)} chunk-escalated straggler(s): {chunked}"
                          if chunked else "none — no speech required chunking")
        lines += [
            "",
            "## Amendment (pre-registration)",
            "",
            "- Amendment #1: the coverage gate was redefined from single-pass >= 99% to "
            "post-convergence == 100% (corpus-wide, for BOTH real specs).",
            f"- Amendment #2: \"resubmitted once\" -> \"resubmitted to convergence"
            + (f" across {n_rounds} round(s)\"" if n_rounds else "\"")
            + f". Chunk-escalated stragglers (judged with partial, not whole-speech, "
            f"context): {straggler_line}.",
        ]

    lines.append("")
    cov = m.get("paragraph_coverage")
    if cov:
        lines.append(f"- paragraph coverage: {cov['rate']:.2%} ({cov['have']:,}/{cov['expected']:,})")
    sc = m.get("speech_coverage")
    if sc:
        lines.append(f"- speech coverage: {sc['rate']:.2%} ({sc['have']:,}/{sc['expected']:,})")
    if m.get("schema_failure_rate") is not None:
        label = "schema/request failure rate" + (" (informational)" if converged else "")
        lines.append(f"- {label}: {m['schema_failure_rate']:.3%}")
    if "unlabeled_share" in m:
        lines.append(f"- unlabeled-paragraph share (topics==[]): {m['unlabeled_share']:.2%}")
    lines += ["", "## Gated checks", ""]
    for name, passed, detail in checks:
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    if not checks:
        lines.append("- (none — no ingested results to gate yet)")
    if m.get("flag_rates"):
        lines += ["", "## Per-flag corpus-wide rates", ""]
        for flag, rate in m["flag_rates"].items():
            lines.append(f"- {flag}: {rate:.2%}")
    if m.get("flag_cooccurrence"):
        lines += ["", "## Flag co-occurrence", ""]
        for k, v in m["flag_cooccurrence"].items():
            lines.append(f"- {k}: {v:.2%}")
    if m.get("enemy_naming_adversarial_consistency"):
        c = m["enemy_naming_adversarial_consistency"]
        lines += ["", "## Enemy-naming <-> adversarial-entity consistency (reported)", "",
                  f"- {c['with_adversarial_entity']}/{c['enemy_naming_paras']} "
                  f"enemy-naming paragraphs carry >=1 adversarial entity ({c['rate']:.2%})"]
    if m.get("entity_substring_rate") is not None:
        lines += ["", "## Entities", "",
                  f"- rows: {m.get('entity_count')}",
                  f"- entity-name-appears-in-paragraph rate: {m['entity_substring_rate']:.2%}"]
    if by_decade:
        lines += ["", "## Top topics by decade", ""]
        for decade in sorted(by_decade):
            top = ", ".join(f"{n} ({c})" for n, c in by_decade[decade].items())
            lines.append(f"- {decade}s: {top}")
    lines.append("")
    return "\n".join(lines)


def cmd_qa(args) -> None:
    """Compute distribution QA over the ingested annotations and write a markdown
    report. `--pilot` enforces the Verification-Strategy gates (scoped to a run);
    `--converged` enforces the post-convergence gates (corpus-wide 100% coverage
    for both specs); full mode (neither) reports the same numbers without gating."""
    run_id = getattr(args, "run_id", None)
    pilot = getattr(args, "pilot", False)
    converged = getattr(args, "converged", False)
    if pilot and converged:
        raise SystemExit("choose --pilot OR --converged, not both")
    result = _compute_qa(run_id, converged=converged)
    md = _qa_markdown(result, pilot, converged)

    out = Path(getattr(args, "out", None) or QA_REPORT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)

    tag = ("  [CONVERGED: gated]" if converged
           else "  [PILOT: gated]" if pilot else "  [full: reported]")
    print(f"QA report -> {out}{tag}")
    for name, passed, detail in result["checks"]:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    if not result["checks"]:
        print("  (no ingested results to gate yet)")

    if pilot or converged:
        label = "CONVERGED" if converged else "PILOT"
        if not result["checks"]:
            raise SystemExit(f"{label} QA: no ingested annotations to check — nothing to gate")
        failed = [n for n, ok, _ in result["checks"] if not ok]
        if failed:
            raise SystemExit(
                f"{label} QA FAILED: {failed}. See {out}"
                + ("." if converged else " — do not proceed to the full run."))
        print(f"{label} QA PASSED — all gates green.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pp-annotate",
        description="LLM annotation passes over the corpus (Anthropic Batches API). "
                    "This is the only command in the pipeline that costs money.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("dry-run", "build + validate the batch and estimate cost; zero network calls"),
        ("submit", "create the batch (SPENDS MONEY)"),
        ("status", "poll batch processing status"),
        ("ingest", "parse results into parquet + manifest"),
        ("qa", "distribution QA over ingested annotations -> notes/annotation-qa-v1.md"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-id", help="run identifier (directory under data/llm_annotations/runs)")
        if name != "qa":
            p.add_argument("--spec", action="append", choices=list(FIELD_SPECS),
                           help="field spec to run (repeatable; default: all)")
        if name in ("dry-run", "submit"):
            p.add_argument("--limit", type=int, help="only the first N speeches")
            p.add_argument("--pilot", action="store_true",
                           help=f"select exactly the deterministic {PILOT_N} era-stratified "
                                "pilot speeches (ignores --limit)")
            p.add_argument("--chunk-size", type=int,
                           help="split each paragraph-unit speech into <= N-paragraph "
                                "chunk requests (convergence lever for persistent stragglers; "
                                "paragraph-unit specs only)")
        if name == "dry-run":
            p.add_argument("--count-tokens", action="store_true",
                           help="use the API's count_tokens for an exact input count "
                                "(makes a free network call; needs credentials)")
        if name == "submit":
            p.add_argument("--force", action="store_true",
                           help="submit even if this run already has a batch_id")
            p.add_argument("--yes", action="store_true",
                           help="confirm the paid call — submit REFUSES without it")
            p.add_argument("--max-cost-usd", type=float, default=MAX_COST_USD,
                           help="refuse to submit if the estimate exceeds this ceiling "
                                f"(default ${MAX_COST_USD:,.2f})")
            p.add_argument("--resubmit-sealed", action="store_true",
                           help="also re-request speeches sealed after a permanent "
                                "(invalid_request) failure — only after fixing the request")
        if name == "ingest":
            p.add_argument("--refetch", action="store_true",
                           help="re-download raw results instead of using the cached copy")
        if name == "qa":
            p.add_argument("--pilot", action="store_true",
                           help="enforce the pilot gates (exit non-zero on any failure)")
            p.add_argument("--converged", action="store_true",
                           help="post-convergence gate: corpus-wide 100%% coverage for both "
                                "real specs + non-degenerate flag/entity gates (exit non-zero "
                                "on failure). Records the pre-registration amendment.")
            p.add_argument("--out", help=f"report path (default {QA_REPORT_PATH})")

    args = parser.parse_args()
    # Fat-finger guard on run_id — it becomes a directory / manifest filename.
    if getattr(args, "run_id", None) and any(bad in args.run_id for bad in ("/", "\\", "..")):
        parser.error(f"--run-id must not contain '/', '\\', or '..' — got {args.run_id!r}")
    if args.command in ("submit", "status", "ingest") and not args.run_id:
        parser.error(f"{args.command} requires --run-id")

    {"dry-run": cmd_dry_run, "submit": cmd_submit, "status": cmd_status,
     "ingest": cmd_ingest, "qa": cmd_qa}[args.command](args)


if __name__ == "__main__":
    main()
