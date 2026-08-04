---
name: derived-analysis-module-recipe
description: Reusable audit recipe for this repo's recurring "derived analysis module" pattern (agreement.py, combat.py) — how to actually prove the $0 guard instead of trusting the docstring
metadata:
  type: project
---

A recurring shape in `presidential_profiles`: a NEW `src/presidential_profiles/<x>.py` that reads
the frozen paid parquets under `data/llm_annotations/`, computes stats, and writes derived tables
under its own `data/<x>/`. Seen in `agreement.py` (2026-07-21), `combat.py`
(`combativeness-over-time`, 2026-07-21) and `register.py` (`breadth-depth-register`, 2026-07-21).
All PASSed. Expect more (era-atlas, charts).

**`register.py` is the cleanest instance so far and is the one to diff future modules against:**
no path CLI arg (see [[cli-path-arg-containment]]), and its committed `data/register/trends.parquet`
holds **zero free-text corpus strings** — the only string columns are 5 controlled vocabularies
(`unit`/`measure`/`taxonomy`/`genre_treatment`/`statistic`, all module constants), everything else is
numeric. No `doc_name`, no `president`, no `text`. That makes the standing forward-looking
HTML-escape note below **inapplicable to this artifact**, unlike `agreement.py`'s. Check the
committed schema before repeating the note.

**Why:** every one of these claims "$0, no API calls" in its docstring, and a docstring is not
evidence. The top risk the pipeline names is a future refactor quietly coupling one of these to
the money path. The check has to be executable, not textual.

**How to apply — the 4 checks that actually pay off, in order:**

1. **$0 guard, empirically.** Static grep for `anthropic` in the module AND in every transitive
   `presidential_profiles` import is necessary but weak. Run the real build under a harness that
   (a) wraps `builtins.__import__` to raise on any `anthropic*` import and (b) raises on
   `socket.socket.connect` / `connect_ex` / `create_connection`. **Do NOT replace `socket.socket`
   itself** — `sklearn -> joblib -> asyncio -> ssl` does `class SSLSocket(socket)` and you get a
   confusing `TypeError: function() argument 'code' must be code, not str` instead of a result.
   Patch the *methods*. Then assert `"anthropic" not in sys.modules` after a full build.
2. **Frozen-artifact integrity, empirically.** `md5 -q data/llm_annotations/*` before and after a
   full real-data build, plus `git status --porcelain` after the pytest run. Both must be
   unchanged/empty.
3. **Path sinks.** Grep every `Path` join in the module. The pass condition is that each one is a
   string literal or a module constant, and that no data-derived string (`run_id`, `doc_name`)
   reaches a path — `run_id` living only as a JSON *value* is fine. Writes that use
   `out_dir / SOME_CONST.name` are safe by construction because only the basename survives.
4. **Committed-artifact scan.** `pd.api.types.is_string_dtype(...)` — **not** `dtype == object`.
   This repo's pandas gives parquet string columns dtype `str`, so an `== object` filter silently
   scans zero columns and reports a false clean. The expected true result is 0 hits; the only
   long-hex value should be the `corpus_fingerprint.doc_name_sha256`.

**Standing forward-looking note (not a finding on these tasks):** these modules serialize raw
corpus `text` / `title` / `president` and model-derived strings unescaped into parquet and
`notes/*.md`. That is fine while `notes/` is not published and nothing under `docs/` reads
`data/<x>/`. The moment a downstream task renders these into the GitHub-Pages HTML, they need
`html.escape()` exactly like `profiles.py` already applies to quoted corpus text — see
[[project-attack-surface]] for the established trusted/escaped split, and
[[project-inter-model-agreement]] where I logged the same LOW for `agreement.py`.

**That note FIRED on 2026-07-21 in `topic-method-comparison`** — `data/topic_display_names.json`
(a file whose documented purpose is hand-editing) now reaches `docs/*.html` and
`docs/explorer/meta.json` unescaped. Logged MEDIUM, not HIGH: the source is still a committed
repo artifact, and the identical sink already existed for the 15 anchored issue names. The
proven per-sink table lives in [[html-render-sink-inventory]] — consult it instead of re-tracing.
Calibration that held up: severity tracks *who controls the string*, not how bad the payload
looks. Repo-write-access-only means MEDIUM here, because that access already implies control of
the generator itself.
