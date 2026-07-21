---
name: verification-annotation-pass
description: How to verify the two-batch LLM annotation pass (pp-annotate) end-to-end incl. the paid pilot, and the count_tokens calibration gotcha
metadata:
  type: reference
---

Verifying the `run-llm-annotation-pass` machinery (`presidential_profiles.annotate`, subcommands
`dry-run/submit/status/ingest/qa`). Two specs: `paragraph_annotations` (masked judgment: topics on
frozen taxonomy_v1, party_attack/enemy_naming/zero_sum flags, proposal-vs-values, entities+stance)
and `speech_annotations` (unmasked factual: speech_type/audience/medium). Complements
[[verification-workflow]] (the `pp-annotate` $0 dry-run discipline).

**Always pass explicit `--spec paragraph_annotations --spec speech_annotations`** — a bare submit
includes the `placeholder` spec (still in the default set). Credential lives in `.env.local`
(gitignored, NOT exported): `set -a; source .env.local; set +a` in the SAME Bash call as any
network subcommand. `dry-run` without `--count-tokens` needs no credential.

**COUNT_TOKENS CALIBRATION GOTCHA (the load-bearing finding):** the offline estimator uses a
~4-chars/token heuristic that *under*-counts. Exact `messages.count_tokens` on the pilot returned
**300,952 input tokens vs the heuristic's 228,096 — ratio 1.319, i.e. exact is 32% HIGHER**.
CONTEXT.md's note that "offline $18.59 > measured $12-17" is about the *cache discount* (offline
assumes zero batch caching), NOT token-counting — those are opposite effects. So uncached real cost
is ABOVE the offline estimate; batch prompt caching (rubric sent as a cached system block on every
request) is what pulls actual billed cost back down. Extrapolation (scale full-corpus offline input
by 1.319, keep output heuristic): judgment full ~$22.2 (< $25 ceiling, ~$2.8 margin — would need
ratio 1.57 to breach), total ~$24.6 (< $50). Both are conservative upper bounds (ignore caching).

**Pilot mechanics:** `submit --pilot ...` creates ONE batch = 40 requests (20 era-stratified
speeches × 2 specs), est $0.35, ceiling should be `--max-cost-usd 2`. Deterministic selection
(random_state=42, Washington 1796 -> Biden 2021). Full-corpus dry-run: 2114 req / 36,229 paras /
$20.47 (= $18.59 judgment + $1.88 factual). Offline outputs land in the WORKTREE's
`data/llm_annotations/runs/<run_id>/` when `PYTHONPATH=src` points at the worktree — verified.

**Batch timing:** a 40-request pilot batch was still `in_progress` after ~6 min of polling
(processing=40, succeeded=0). Small batches are NOT instant — budget 10min+ and expect the
orchestrator to resume you for `ingest`/`qa`/spot-render once `status` shows `ended`.

**DEGENERATE-RESPONSE FAILURE MODE (pilot v1, 2026-07-20 — the reason the paid run must not
auto-proceed):** the masked judgment pass (long structured-output `annotations` array, `effort:low`)
is unreliable on some speeches — the model emits only 1-2 items then stops with `stop_reason:
end_turn` (NOT max_tokens — genuine under-production, not truncation), dropping most paragraphs.
In the real pilot, **4 of 20 judgment requests degenerated (20%)**: three returned 1-2 of 21/46/64
paragraphs; one returned all 48 but with a duplicate para_idx. Aggregate coverage would be ~79% (gate
needs >=99%) → hard FAIL. TWO compounding problems to always check: (1) don't trust aggregate output
token counts — scan PER-REQUEST returned-vs-sent para_idx counts; (2) **`cmd_ingest` HARD-ABORTS the
entire batch** (`raise ValueError "returned para_idx=N twice"`, ~annotate.py:928) on the FIRST
duplicate-para_idx response — it writes NO parquets/manifest for the whole run, unlike its graceful
handling of merely-dropped paragraphs (incomplete → re-request). So a single degenerate response in
the paid 1,057-speech full run blocks ingesting everything you paid for until code changes. This is a
retryability/prompt-design failure, not something a resubmit fixes.

**Pilot v2 (post-fix, 2026-07-20): quarantine works, degeneracy only PARTLY fixed → still FAILs
coverage.** The fix (ingest quarantine of malformed/duplicate responses + judgment effort=medium +
explicit "annotate every para_idx" count contract + output est 115/para) ELIMINATED the hard-abort:
ingest completed, wrote all 3 parquets, keyed correctly, 20/20 speeches. But the model STILL
degenerated on **3 of 20 judgment requests** (collapsed to 1 annotation, dropped 107/598 paras) →
**coverage 82.11% < 99% gate → PILOT FAIL**. The OTHER 5 gates all passed cleanly and content quality
was excellent (party/enemy/zero_sum non-degenerate at 3.3/18.1/3.9%, entity substring 89.5%, enemy↔
adversarial-entity consistency 97.75%, unlabeled 0.00%, by-decade topics = data not noise). Takeaway:
the collapse-to-1-annotation failure on LONG speeches (Obama SOTU 2014, 64 paras, broke in BOTH v1 and
v2) is partly stochastic, partly inherent to long structured-output arrays — effort=medium + a count
contract REDUCES but does not eliminate it. A prompt tweak alone may not reach 99% first-pass; the
design's "re-request incompletes once" round (blocked at VERIFY-APP by the no-second-batch rule) is
likely load-bearing for coverage.

**Cost/output recalibration (validated in v2):** measured judgment output = **114.0 tok/para** (the
recalibrated 115 constant holds almost exactly); factual 38.5 tok/speech. v2 actual **$0.5952 vs $0.58
offline est — within 3%** (offline heuristic is now an accurate central estimate: input under-count
~1.32x and batch caching ~roughly cancel). BUT the 115/para output (up from the original ~40/para)
**TRIPLED the full-judgment offline estimate from $18.59 to $32.35** — which EXCEEDS the $25 default
MAX_COST_USD ceiling, so the full judgment submit needs `--max-cost-usd ~35`. Full-run offline (v2
code): judgment $32.35 / factual $1.88 / both $34.23; no-cache exact-adjusted upper bound ~$37.9;
realistic actual ~$30-36 depending on caching (v1 cached 53%, v2 only 22% — best-effort, varies).
Grand total incl. $1.06 already spent stays < $50, but the judgment single-submit ceiling is tight.

**FULL-RUN ACTUALS (2026-07-20):** judgment batch (1040 req after resume-skip of 17) came in at
**$26.28 actual vs $31.90 est — UNDER by 18%**, because caching at scale hit **~49%** (8.1M cache-read
of 16.4M input) vs the pilot's 22%. Collapse rate at scale IMPROVED vs pilot: ~11.7% of succeeded
responses dropped paragraphs (pilot was 15%), but still substantial — per-speech corpus result was
**902 fully-covered / 80 partial / 75 absent** of 1057. The collapse hits the LONGEST speeches
hardest: the 5 worst were all 1898-1912 annual messages (150-202 paragraphs each) that collapsed to a
SINGLE returned annotation — so retry rounds likely WON'T fix the longest speeches without chunking
(they'll re-collapse). Paragraph coverage plateaued at **80.6% (29,212/36,229)**. Also confirms the
hardened ingest quarantine works: responses with duplicate/unsent para_idx are routed to retryable
(rows discarded) separate from clean "incomplete" — don't count raw dropped-paragraph responses as
"incomplete"; the ingest splits them (60 quarantined vs 80 incomplete here).

**CONVERGENCE / CHUNKING (retry round 2, 2026-07-20):** two retry rounds took paragraph coverage
80.6% → **95.67%** (speech_annotations reached 1057/1057 complete). The residual **32 speeches all
collapsed in BOTH rounds (2x)** — consistent hard-collapsers, not billing flukes. KEY INSIGHT for the
chunk round: the collapse is NOT purely a length problem — 14 of the 32 are SHORT (<=25 paras, some
just 4-9 paras collapsing to 1), and **`--chunk-size 25` only splits the 18 LONG speeches (>25 paras,
1,392 of the 1,569 missing paras)**; the 14 short ones (177 paras) aren't split by size 25 and will
likely re-collapse. So one chunk round realistically reaches ~99.5% (36,052/36,229), NOT 100% — the
last ~0.5% needs a smaller chunk size or accepting the short-speech collapse as stochastic residue.
Cumulative actual spend through r2 = **$35.52** of $50; chunk round projects ~$1.5-3 more.
Chunk-25 round (81 requests, $1.54) took coverage 95.67% → **99.19%** (32 → 10 remaining speeches),
with PERFECT chunk reassembly (custom_id `-c0/-c1/...` suffixes, ingest re-anchors by para_idx: 0
orphans, 0 dup keys, 0 mis-keyed). BUT collapse persists even at CHUNK level — a 15-para chunk of the
1954 SOTU still collapsed to 1. So the Option-A ladder converges but needs multiple steps: each
smaller chunk size clears most-but-not-all; expect 3b (chunk~10) → ~99.7%, possibly 3c (chunk~5) for
the residue. Cumulative through chunk-25 = **$37.06** of $50.
Chunk-10 (74 req, $1.20) → **99.93%** (10 → 3 remaining); chunk-5 next. Collapse observed even at
5-8-para chunks, so the last 3 speeches (Jun 13 1870 Cuba, Sep 23 2010 UN, Mar 4 1901 2nd Inaugural;
27 paras, collapsed 4x) are STUBBORN — the ladder converges but the tail is content-specific, not
just length. Reassembly integrity stayed perfect (0/0/0) at every chunk size. Cumulative $38.26.
Chunk-5 (16 req, $0.09) → **99.99%** (1 speech, 4 paras left: 1870 Cuba message idx 6-9). Root-caused
the tail by reading the cached response: the model returned a COMPLETE annotation for para 5 then
STOPPED (`end_turn`, 187 out tokens — NOT truncation, NOT refusal). The dropped paras are dry
neutrality-doctrine/filibustering history — no safety/content trigger. So the stubborn tail is
array-position EARLY-STOP laziness on dense prose, not a rubric or content problem. Fix = chunk-2/1
(a 1-2-para array is nearly impossible to under-produce), ~$0.02-0.05, NOT a prompt change. Ladder:
32→10→3→1 speeches across chunk 25→10→5. Cumulative through chunk-5 = **$38.35** of $50.
Chunk-2 (9 req, $0.03) → **99.997%** — and STILL 1 short: para 9 of the 1870 Cuba message dropped
even in a 2-para array `[8,9]` (returned para 8, `end_turn`). This PROVES the early-stop is "annotate
item 1, stop" regardless of array size — a 2-item chunk is NOT enough; only **chunk-1** (a 1-item
array, nothing to stop before) structurally guarantees it. `qa --converged` gates paragraph_coverage
at EXACT ==100% (n_have==n_expected), so 36,228/36,229 FAILs that one gate (displays "100.00%" but
gate is exact). All 5 OTHER converged gates PASS: speech 1057/1057, party 6.56% / enemy 21.07% /
zero_sum 7.67% (non-degenerate), entity substring 85.81%, enemy↔adversarial 96.84%, unlabeled 1.12%
(collapsed from legacy 28.7%), by-decade topics textbook (1850s Slavery, 1940s WWII, 1960s Vietnam,
2000s Iraq). Amendment auto-records 7 rounds + the 32 chunk-escalated speeches (manifest-gated).
LESSON: budget the LADDER to chunk-1, not chunk-2 — a single hyper-stubborn paragraph can survive a
2-item array. Cumulative through chunk-2 = **$38.38** of $50.
**CONVERGED (chunk-1, 18 req, $0.18): 36,229/36,229 = 100%, all 6 gates PASS, `qa --converged` exit 0.**
chunk-1 (1-item array) landed para 9 — a 1-item request literally cannot early-stop. Full ladder was
10 paid batches; TOTAL SPEND **$38.56** of $50. Integrity stayed 0/0/0 (dup/orphan/mis-keyed) the whole
way. PROVENANCE SUBTLETY: the amendment's `n_rounds` = distinct SURVIVING run_ids in the parquet (line
~1393, latest-run-wins on key collision), NOT operational batch count — chunk-2 shows 0 surviving rows
(chunk-1 re-did the same Cuba speech and overwrote all of it), so n_rounds=7 not 8. Correct-but-subtle:
it counts rounds whose annotations are in the FINAL data. Chunk-escalated speech list (32) is separate
(manifest-gated `_all_chunk_escalated_docs`) and correct.

**Pilot v1 cost calibration:** actual $0.461 vs $0.35 offline est vs $0.43 exact-token projection.
Caching 53.3% (170,059 read / 319,111 input). Output ran ~2x the (old) heuristic — the signal that
drove the 115/para recalibration.

**CREDIT-EXHAUSTION FAILURE (full run, 2026-07-20 — verify billing before big batches):** the
factual full batch came back 927 succeeded / **110 errored, ALL identical**: batch result
`type:"errored"`, error `type:"invalid_request_error"`, message *"Your credit balance is too low to
access the Anthropic API..."*. This is a BILLING failure that masquerades as a request-construction
error — the account ran out of credits mid-run (two big batches, ~$34 total, submitted in parallel).
Ingest correctly classified all 110 as **retryable, sealed_permanent=0** (right outcome — top up
credits then resubmit fills them; don't seal). NOTE the seal trigger is `err.get("type")=="invalid_
request"` but the API returns `"invalid_request_error"` — so a genuinely malformed request would ALSO
land in retryable, not sealed (the permanent-seal path may never fire for real API errors; flag for
code review, risk = re-pay loop on a deterministically-bad request). LESSON: before a large batch run,
confirm the Anthropic account credit balance covers the FULL estimate — the pilots ($1.06) plus the
$33.75 submit estimate needed >$34 available and the account had less. Errored requests bill $0, so
the failure is free but leaves coverage gaps to backfill.

**QA gates (pilot mode, `qa --run-id <id> --pilot`, scopes to the run's requests_index docs):**
schema_failure < 1% (QA_MAX_SCHEMA_FAILURE=0.01), paragraph_coverage >= 99% (QA_MIN_COVERAGE=0.99),
each of the 3 flags non-degenerate (not > 60% and not < 0.5% — QA_FLAG_DEGENERATE_HIGH=0.60 /
_LOW=0.005), entity substring >= 80% (QA_MIN_ENTITY_SUBSTRING=0.80). `--pilot` exits non-zero on any
fail. Report writes to `notes/annotation-qa-v1.md` by default. Ingest writes 3 parquets:
`paragraph_annotations` (flat, PARAGRAPH_KEY), `speech_annotations` (SPEECH_KEY),
`paragraph_entities` (exploded long-format: doc_name,para_idx,entity,type,stance,run_id).
