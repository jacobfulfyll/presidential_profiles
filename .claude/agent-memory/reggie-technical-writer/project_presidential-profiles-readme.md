---
name: project_presidential-profiles-readme
description: presidential_profiles README.md structure, voice, and doc-sync conventions for new modules
metadata:
  type: project
---

The presidential_profiles README (`/Users/jacobpress/Desktop/Projects/presidential_profiles/README.md`,
same relative path in worktrees) has a fixed structure worth preserving on future SYNC-DOCS
passes: Findings (narrative + figures) → Data → Running it → **"How it works" table** (Stage |
Module | Method) → Then and now.

**"How it works" table voice**: one row per pipeline module, terse arrow-chained descriptions
("X → Y → Z"), no trailing period, method column packs in the "why" briefly rather than just the
"what" (e.g. issues.py's row says *why* CorEx is anchored, not just that it runs CorEx). Place new
rows near the modules they relate to, not at the table's end — row order roughly follows pipeline
stage order.

**Data section convention**: every new paragraph-keyed parquet artifact gets 1-2 sentences stating
its key — `(doc_name, para_idx)`, never row order — echoing the repo's [[keyed-merge]] rule from
CLAUDE.md ("Paragraph-level tables share a real key... join on it, never assume row order"). This
rule is asserted three separate times in the current README (main Data para, LLM-annotations
para, and now the cluster para) — it is a recurring correctness hazard the maintainer cares about,
always restate it for new keyed tables rather than assuming it's implied.

**Watch for absolute claims going stale when a new paid/standalone script lands**: the README
said `pp-annotate` is "the only command in the pipeline that can spend money" — true when
written, false the moment `taxonomy.py` landed as a second paid entry point. Caught and fixed
(narrowed to "the only `pp-*` command") during the `derive-corpus-taxonomy` SYNC-DOCS pass
(2026-07-19). When adding a new module row/paragraph, grep the README for words like "only",
"never", "always" near the topic to catch this class of staleness, not just the new module's own
section.

**Scope discipline confirmed by a SYNC-DOCS task (2026-07-19)**: this repo distinguishes sharply
between *library building blocks* (no `pp-*` CLI entry point, e.g. `embed_topics.py`) and
*pipeline stages* (wired into `pp-analyze`/`pyproject.toml` `[project.scripts]`). Before
documenting a new module as part of the CLI flow, grep `pyproject.toml` `pp-` scripts and grep the
module name across `src/` for callers — if it has zero callers and no script entry, document it
only in the "How it works" table as a standalone stage, and do not imply it runs via `pp-analyze`.
Do not invent CLI commands for modules that don't have one.

Do NOT create a CHANGELOG.md in this repo — completed tasks are recorded in `HISTORY.md` by the
pipeline itself, not by the technical writer.

**`taxonomy.py` landed 2026-07-19** (task `derive-corpus-taxonomy`, commit `18e1020`) — the
corpus-derived two-level topic taxonomy (17 level-1 domains / 50 level-2 topics, 5 non-policy:
ceremonial, personal narrative, procedural/administrative, faith & values, partisan/media
combat), frozen as `data/llm_annotations/taxonomy_v1.json` + `crosswalk_v1.json`, provenance at
`data/llm_annotations/manifests/taxonomy-v1-20260719.json` (models: `claude-opus-4-8` for
proposals/merge/crosswalk, `claude-sonnet-5` for held-out coverage labeling; v1 run cost $2.44,
100% held-out coverage). Documented in README's Data section (frozen/pre-registration rationale),
Running It (paid vs `--dry-run`, no `pp-*` entry point — same "grep for callers before implying
CLI wiring" rule as `embed_topics.py`), and the How it works table. This superseded an earlier
memory entry that had these as deferred/out-of-scope — a reminder that scope notes from one
SYNC-DOCS pass can go stale by the next; re-verify with a file check rather than trusting the
memory's snapshot.

**`pp-annotate` corpus run completed 2026-07-20/21** (task `run-llm-annotation-pass`) — the two
real `FieldSpec`s: `paragraph_annotations` (masked judgment pass: topics against the frozen
`taxonomy_v1.json` 50-name enum, `party_attack`/`enemy_naming`/`zero_sum` binary flags,
`proposal_values` class, entities with type+stance — model never sees president/title, only
paragraph text + decade) and `speech_annotations` (unmasked factual pass: `speech_type`/
`audience`/`medium` from title+year+opening paras, unmasked because the title is the best signal
for a factual field). New CLI surfaces documented: `--pilot` (deterministic 20-speech
era-stratified sample), `--chunk-size N` (splits a speech into ≤N-paragraph chunk requests),
`qa` subcommand (`--pilot`/`--converged` gate, plain mode reports), `--resubmit-sealed`. Full
corpus (36,229 paragraphs / 1,057 speeches) annotated for $38.56 (budget $50), documented in
both the Data section (new parquets: `paragraph_annotations.parquet`, `paragraph_entities.parquet`,
`speech_annotations.parquet`, linked to `notes/annotation-qa-v1.md`) and Running It (CLI flags +
one paragraph on the "collapse" convergence ladder: Sonnet 5 structured-output arrays sometimes
emit one item and stop at `end_turn`; fixed via resubmission rounds + `--chunk-size` escalating
25→10→5→2→1 for 32 stragglers; recorded as pre-registration amendments in the QA report). **Did
NOT add a How-it-works table row for `annotate.py`** despite `taxonomy.py` (a similar standalone
paid script) having one — flagged as a discovered issue rather than fixed, since the task's file
boundary was scoped to the Data/Running-It prose specifically. If a future SYNC-DOCS task's
boundary allows it, adding an `annotate.py` row would make the table consistent with the
`taxonomy.py` precedent. *(Resolved: an `annotate.py` row exists in the table as of the
`combativeness-over-time` pass — someone added it in between.)*

**`combat.py` landed 2026-07-21** (task `combativeness-over-time`) — first *analysis* layer built
on the frozen annotations, standalone `python -m presidential_profiles.combat`, no `pp-*` entry
point (same "grep pyproject scripts + grep src for callers" check as `taxonomy.py`/`embed_topics.py`;
grepping `combat` in `src/` false-positives on the taxonomy's "partisan/media combat" topic name —
check the match, not the count). Documented in README Data (2 paragraphs), Running It (1
paragraph), and a How-it-works row placed after `annotate.py`. **SYNC-DOCS scope for a task whose
deliverable is a `notes/*-findings-v1.md` report: docs describe the module + artifacts and link the
report — do NOT add a README "Findings" narrative section**, because every existing Findings
subsection is figure-led and this module ships no figures (its own §8.13 records "criterion met in
substance, not the specified medium").

**Verify reproducibility claims by running them, not by trusting the commit message.** `combat.py`'s
"all 10 outputs byte-identical on any date" is testable without touching committed artifacts:
`build_combativeness(out_dir=<scratchpad>)` then `cmp`. Took ~40s and confirmed all 10. Note
`corpus.DATA_DIR` is **package-relative** (`Path(__file__).parents[2]/"data"`), not cwd-relative —
so a `PYTHONPATH=<worktree>/src` run writes into the *worktree's* `data/`, and README wording should
not claim commands must run "from the repo root".
