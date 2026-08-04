# Presidential Profiles agent contract

This repository builds a deterministic static site from the Miller Center presidential-speech
corpus. Preserve the existing native Claude/Reggie layout; there are no Codex pipeline adapters
and none should be invented.

## Start here

1. Read `HANDOFF.md`, then `README.md`, `TASKS.md`, and the task-specific route in
   `.codex/knowledge/README.md`.
2. Inspect `git status` before editing. The current worktree contains extensive user-owned
   implementation, generated-site, data, and `.claude` changes. Do not clean, reset, discard,
   reformat, or overwrite unrelated work.
3. Use `CLAUDE.md` and the routed `.claude/agent-memory/` notes for project-specific depth.

## Source of truth

- Python under `src/presidential_profiles/` generates the site.
- `docs/` is generated output, not an editing surface. Change generators and rebuild.
- Paragraph tables join on `(doc_name, para_idx)` with key-set and one-to-one validation; never
  rely on row order.
- `data/llm_annotations/` contains frozen paid artifacts. Do not regenerate or hand-edit them.
- Derived local layers such as `data/coverage_pressure/`, `data/register/`, and `data/networks/`
  remain governed by the provenance rules in `CLAUDE.md`.

## Environment and verification

- The uv environment is x86_64 under Rosetta. Use
  `arch -x86_64 .venv/bin/python`, including for `-m pytest` and `-m
  presidential_profiles.site`.
- Build the site with
  `arch -x86_64 .venv/bin/python -m presidential_profiles.site`.
- Run targeted tests first, then the full relevant pytest suite.
- The site build validates navigation, internal links, anchors, JSON shards, and metric
  registration. Run separate static JavaScript syntax validation after generation.
- Preview locally with `python3 -m http.server 8010 --directory docs`; do not deploy unless a
  user explicitly requests it.

## Story Page V2

The current implementation contract is `notes/story-redesign-v2.md`. Historical claims must
remain tied to named artifacts, declared statistical status, readable non-causal event markers,
accessible staged controls, and explicit acceptance tests. Keep the chronology distinct from the
synthesis.

## Reggie

Use documentation mode and the portable doctor:

```bash
~/.codex/scripts/reggie/reggie_doctor.sh --repo /Users/jacobpress/Desktop/Projects/presidential_profiles
```

There is no project finish hook at present. Do not install a generic adapter to create one.
