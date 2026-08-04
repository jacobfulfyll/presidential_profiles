# Decisions

## D001 — Publish the generated site from `master:/docs`

Status: accepted on 2026-08-03.

The repository already has a working GitHub Pages project site sourced from the
`docs/` directory on `master`. The site generator is deterministic, performs its
own navigation/data validation, and writes the static files that Pages serves.

Keep that publishing source for the current release. Commit the generator and
its generated output together, add a generated `.nojekyll` marker, and require a
review branch plus local release gates before merging to `master`. A custom
GitHub Actions build may be considered later, but it is not required to publish
the already-generated static site.

## D002 — Separate the site release from the annotation control plane

Status: accepted on 2026-08-03.

Do not add `data/annotation_ledger/`, `.codex/operators/`, or tool telemetry to
the GitHub Pages repository release. The ledger contains a large local control
plane, sealed evidence, and provisional artifacts whose publication and storage
requirements differ from those of the website.

The public repository may contain the provider-neutral source, tests, contracts,
small governed layers, and frozen artifacts that were already intended for Git.
The heavyweight ledger remains local until an explicit archival decision defines
its package, destination, provenance manifest, and access boundary.
