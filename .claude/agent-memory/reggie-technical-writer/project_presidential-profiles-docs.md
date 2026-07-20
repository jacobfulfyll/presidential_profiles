---
name: project-presidential-profiles-docs
description: Presidential Profiles doc conventions — README-only (no CHANGELOG.md), module table in "How it works", terminology to match UI copy exactly
metadata:
  type: project
---

Presidential Profiles keeps a single `README.md` as its documentation surface — no
`CHANGELOG.md` exists and the project convention is README-only for user-facing change
summaries. `CLAUDE.md`, `TASKS.md`, and `HISTORY.md` are maintained by other pipeline stages,
not the technical-writer/SYNC-DOCS stage.

**Why:** Confirmed explicitly in a SYNC-DOCS task brief (profile-issue-views, 2026-07-19):
"Do NOT create a CHANGELOG.md — this repo doesn't have one and the convention is README-only."

**How to apply:**
- When docs need to reflect a code change, edit `README.md` only, unless told otherwise.
- README has a "## How it works" table (`Stage | Module | Method`) mapping each pipeline
  module (`fetch.py`, `rhetoric.py`, `indices.py`, `issues.py`, `similarity.py`, `trends.py`,
  `profiles.py`, `site.py`/`figures.py`) to a one-line description of what it computes. When a
  module's behavior changes, update its row in this table rather than adding a new section.
- The dashboard blurb near the top (lines ~11-18) is a dense, single-paragraph feature list for
  the profile pages and dashboard — mirror its terse, comma-chained style when adding a new
  feature description rather than adding new paragraphs/bullets.
- Match UI copy verbatim where possible: badge/label text rendered in `profiles_site.py`
  (e.g. `"topic of the day"`, `"Thin record"`, `"N% of their paragraphs"`, `"N pp vs their
  era"`) should be quoted the same way in README so a reader can find the phrase on the live
  page.
- `profiles.py` computes profile-page data (issue cards, fingerprint, vocabulary); the actual
  HTML rendering module is `profiles_site.py` — NOT `site.py`/`figures.py` (those render the
  main dashboard/index, per the existing "Site & figures" table row). The README's module table
  does not currently have its own row for `profiles_site.py` — it's folded into the `profiles.py`
  description. Flagged but left alone (out of scope for a surgical edit) — could be raised as a
  discovered-issue if it causes future confusion.
