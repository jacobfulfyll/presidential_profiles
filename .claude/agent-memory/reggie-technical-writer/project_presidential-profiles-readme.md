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

**Scope discipline confirmed by a SYNC-DOCS task (2026-07-19)**: this repo distinguishes sharply
between *library building blocks* (no `pp-*` CLI entry point, e.g. `embed_topics.py`) and
*pipeline stages* (wired into `pp-analyze`/`pyproject.toml` `[project.scripts]`). Before
documenting a new module as part of the CLI flow, grep `pyproject.toml` `pp-` scripts and grep the
module name across `src/` for callers — if it has zero callers and no script entry, document it
only in the "How it works" table as a standalone stage, and do not imply it runs via `pp-analyze`.
Do not invent CLI commands for modules that don't have one.

Do NOT create a CHANGELOG.md in this repo — completed tasks are recorded in `HISTORY.md` by the
pipeline itself, not by the technical writer. `taxonomy.py`/`taxonomy_v1.json`/`crosswalk_v1.json`
were explicitly deferred/out-of-scope as of 2026-07-19 (do not document until they actually exist
in the tree — verify with a file check, not memory, since this is a fast-moving repo).
