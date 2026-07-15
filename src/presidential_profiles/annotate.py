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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import TypeAdapter

from . import llm_annotations as ann
from .corpus import load

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


def _custom_id(spec_name: str, doc_name: str) -> str:
    """custom_id encodes the KEY, never the row position — batch results come
    back in arbitrary order, and keying by position is precisely the bug this
    whole module exists to make impossible.

    doc_name is a 100-char URL path with slashes, so it cannot go into a
    custom_id literally; it is hashed, and requests_index.json (written
    alongside the JSONL) maps every custom_id back to its doc_name and the
    exact paragraph indices that were sent. Ingest re-anchors through that map
    and validates every returned para_idx against it."""
    digest = hashlib.sha256(doc_name.encode()).hexdigest()[:24]
    return f"{spec_name}-{digest}"


def build_requests(
    spec: FieldSpec, speeches: pd.DataFrame, paragraphs: pd.DataFrame
) -> tuple[list[Request], dict]:
    """One request per speech.

    A paragraph-unit spec carries every paragraph of that speech, each tagged
    with the para_idx the model must echo back (ingest validates every one of
    them against `para_idxs` below, in both directions). A speech-unit spec
    carries the transcript and answers once, for the speech as a whole — so it
    sends no para_idx and records none."""
    requests: list[Request] = []
    index: dict[str, dict] = {}
    by_doc = {d: g for d, g in paragraphs.groupby("doc_name")}

    for speech in speeches.itertuples():
        doc_name = speech.doc_name
        paras = by_doc.get(doc_name)
        if paras is None or paras.empty:
            continue
        paras = paras.sort_values("para_idx")

        if spec.unit == "paragraph":
            para_idxs = [int(i) for i in paras["para_idx"]]
            body = "\n\n".join(
                f"[para_idx={int(r.para_idx)}]\n{r.text}" for r in paras.itertuples()
            )
        else:
            para_idxs = []
            body = speech.transcript
        cid = _custom_id(spec.name, doc_name)

        params = MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=spec.max_tokens,
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
                "effort": "low",
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
        index[cid] = {
            "doc_name": doc_name,
            "spec": spec.name,
            "unit": spec.unit,
            "para_idxs": para_idxs,
        }
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
) -> tuple[list[Request], dict, pd.DataFrame, pd.DataFrame]:
    """Build every spec's requests, optionally capped at the first `limit`
    speeches. A caller that already holds the corpus (submit, which fingerprints
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
        reqs, idx = build_requests(spec, speeches, paragraphs)
        requests += reqs
        index |= idx
    return requests, index, speeches, paragraphs


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
        est_out += (
            len(meta["para_idxs"]) * 12
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
    run_id = args.run_id or f"{date.today().isoformat()}-dryrun"
    requests, index, speeches, paragraphs = _prepare(specs, args.limit)

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

    print(f"DRY RUN — no network calls, no cost. run_id={run_id}")
    print(f"  specs:      {', '.join(s.name for s in specs)}")
    print(f"  requests:   {len(requests)} (1 per speech, all paragraph fields grouped)")
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
    fingerprint = ann.corpus_fingerprint(speeches=speeches, paragraphs=paragraphs)
    requests, index, _, paragraphs = _prepare(specs, args.limit, speeches, paragraphs)

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
    if not requests:
        print("nothing to submit — every requested speech is already fully annotated")
        return

    for req in requests:
        _validate_request(req)
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
            # invalid_request is a permanent validation failure — retrying it
            # burns money to fail again. Everything else is safe to retry.
            (permanent_failures if err.get("type") == "invalid_request"
             else retryable_failures).append((cid, meta["doc_name"], err.get("type", rtype)))
            continue

        message = result["result"]["message"]
        for k in usage:
            usage[k] += message.get("usage", {}).get(k) or 0

        # A "succeeded" result can still carry no text block — a refusal, or an
        # all-thinking response. Its tokens were billed (already folded into
        # usage above), but there is nothing to parse. Record it like a
        # retryable failure — one anomalous, cached response must not abort
        # ingest of the whole batch (the same class QUALITY-CHECK closed for
        # errored results). No rows are written, so the speech is not sealed done
        # and the next `submit` re-requests it.
        text = next((b["text"] for b in message["content"] if b["type"] == "text"), None)
        if text is None:
            retryable_failures.append((cid, meta["doc_name"], "no_text_block"))
            continue
        payload = json.loads(text)  # output_config.format guarantees valid JSON

        if meta["unit"] == "speech":
            # One object for the whole speech: no para_idx, keyed by doc_name.
            spec_rows = [{"doc_name": meta["doc_name"], **payload, "run_id": run_id}]
            missing: list[int] = []
        else:
            valid_idxs = set(meta["para_idxs"])
            returned: set[int] = set()
            spec_rows = []
            for item in payload["annotations"]:
                para_idx = int(item["para_idx"])
                # The model echoes para_idx; trust it only after checking it
                # against what we actually sent. A hallucinated index must not
                # become a key.
                if para_idx not in valid_idxs:
                    raise ValueError(
                        f"{cid} ({meta['doc_name']}): returned para_idx={para_idx}, "
                        f"which was not in the request"
                    )
                if para_idx in returned:
                    raise ValueError(
                        f"{cid} ({meta['doc_name']}): returned para_idx={para_idx} twice"
                    )
                returned.add(para_idx)
                row = {k: v for k, v in item.items() if k != "para_idx"}
                spec_rows.append(
                    {"doc_name": meta["doc_name"], "para_idx": para_idx, **row,
                     "run_id": run_id}
                )
            # `returned ⊆ sent` is checked above; this is the other direction.
            # Dropping items from a long array is the standard failure mode of
            # this prompt shape, and it arrives as silence — no error, just
            # paragraphs that never appear. The rows that DID come back are kept
            # (correctly keyed, and paid for), but the speech is recorded as
            # incomplete and `_already_ingested` will not call it done.
            missing = sorted(valid_idxs - returned)
            n_sent += len(valid_idxs)
            n_returned += len(returned)

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

    actual_cost = cost_usd(**usage)
    written = []
    for spec_name, spec_rows in rows.items():
        spec = FIELD_SPECS[spec_name]
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

    specs = _specs(state.get("specs"))
    ann.write_manifest(ann.Manifest(
        run_id=run_id,
        model=state.get("model", MODEL),
        prompt_version=",".join(f"{s.name}={s.prompt_version}" for s in specs),
        prompt_hash=",".join(f"{s.name}={s.prompt_hash()}" for s in specs),
        date=date.today().isoformat(),
        batch_id=batch_id,
        n_requests=sum(counts.values()),
        cost_usd=round(actual_cost, 4),  # ACTUAL, from usage — not the estimate
        corpus_fingerprint=state.get("corpus_fingerprint") or ann.corpus_fingerprint(),
        fields=[s.name for s in specs],
        annotation_files=written,
        notes=(
            f"succeeded={counts['succeeded']} errored={counts['errored']} "
            f"canceled={counts['canceled']} expired={counts['expired']}. "
            f"permanent (invalid_request) failures: {len(permanent_failures)}; "
            f"retryable failures: {len(retryable_failures)}. {coverage}"
        ),
        **usage,
    ))

    print(f"ingested run {run_id}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
          f"(cache: {usage['cache_read_input_tokens']:,} read, "
          f"{usage['cache_creation_input_tokens']:,} write)")
    print(f"  ACTUAL COST: ${actual_cost:,.4f}")
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
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-id", help="run identifier (directory under data/llm_annotations/runs)")
        p.add_argument("--spec", action="append", choices=list(FIELD_SPECS),
                       help="field spec to run (repeatable; default: all)")
        if name in ("dry-run", "submit"):
            p.add_argument("--limit", type=int, help="only the first N speeches")
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
        if name == "ingest":
            p.add_argument("--refetch", action="store_true",
                           help="re-download raw results instead of using the cached copy")

    args = parser.parse_args()
    if args.command != "dry-run" and not args.run_id:
        parser.error(f"{args.command} requires --run-id")

    {"dry-run": cmd_dry_run, "submit": cmd_submit,
     "status": cmd_status, "ingest": cmd_ingest}[args.command](args)


if __name__ == "__main__":
    main()
