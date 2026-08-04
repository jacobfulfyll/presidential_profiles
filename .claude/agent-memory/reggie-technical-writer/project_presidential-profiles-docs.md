---
name: project-presidential-profiles-docs
description: Presidential Profiles doc conventions — README-only (no CHANGELOG.md), module table in "How it works", terminology to match UI copy exactly
metadata:
  type: project
---

Presidential Profiles keeps a single `README.md` as its user-facing documentation surface — no
`CHANGELOG.md` exists and the project convention is README-only for change summaries. `TASKS.md`
and `HISTORY.md` are maintained by other pipeline stages, never by SYNC-DOCS.

**`CLAUDE.md` IS in the writer's scope** (corrected 2026-07-21, `topic-chart-upgrades`) — a
SYNC-DOCS brief can and does direct edits to it, most often a new **Data conventions** bullet
classifying a new artifact as frozen/paid vs derived-deterministic-$0, plus generalizable lessons
appended to an existing lessons section. Its register is *hard-won rules with concrete numbers*,
not a feature list: every claim carries the measured figure, the failure it prevents, and often
the symbol that enforces it. Extend an existing section rather than adding one; the file is long
and a lesson that reads as a war story rather than a rule is not worth its lines. Recompute every
number from the artifact before writing it — the repo's own lessons say prose numbers are the
weakest class, and that now explicitly includes prose inside provenance JSON.

**The implementer often edits `CLAUDE.md` first** (confirmed `suppress-degenerate-band-intervals`,
2026-07-22 — the feature commits already carried a `Data conventions` bullet). Diff
`git diff master..HEAD -- CLAUDE.md` before writing: the job is then to close the *gaps* in that
block (consumer-facing consequences, counting traps, the "why not X" that only the reviewer knows)
and append the generalizable lessons — not to rewrite prose that already scored well. Surgical
`Edit`s into the existing bullet read better than a competing second bullet on the same subject.

**Why (README-only for change summaries):** Confirmed explicitly in a SYNC-DOCS task brief
(profile-issue-views, 2026-07-19): "Do NOT create a CHANGELOG.md — this repo doesn't have one and
the convention is README-only."

**How to apply:**
- When docs need to reflect a code change, edit `README.md` (and `CLAUDE.md` when the brief says
  so) — never invent a CHANGELOG.
- Do not write a `notes/*-findings-*.md` for presentation/infrastructure work; those are read as
  research findings. Say why one is or isn't warranted and stop rather than writing one.
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
