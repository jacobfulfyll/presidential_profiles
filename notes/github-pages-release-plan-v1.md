# GitHub Pages release plan v1

Status: owner-authorized for execution on 2026-08-03.

## Decision

Publish the deterministic static site through the repository's existing GitHub
Pages source, `master:/docs`. Keep the Python generator and the generated site
in the same reviewed release, while keeping the local annotation control plane
outside normal Git history.

This release does not promote, materialize, reinterpret, or publish the
provisional annotation-ledger composite. It does not regenerate or hand-edit
the frozen paid artifacts under `data/llm_annotations/`.

## Release boundary

The release snapshot may contain:

- portable and native project documentation: root Markdown files,
  `.codex/knowledge/`, and `.claude/agent-memory/`;
- Python source, tests, release-audit scripts, project configuration, and lock
  files;
- research contracts under `notes/`;
- tracked and small governed artifacts under `data/`, except for the excluded
  annotation control plane;
- the complete generated `docs/` tree, including `docs/.nojekyll`.

The release snapshot must not contain:

- `data/annotation_ledger/`, including sealed runs, materialized generations,
  SQLite state, provisional selections, responses, or attempt receipts;
- `.codex/operators/`, which contains local annotation-execution helpers rather
  than portable project source;
- `.claude/stats.json` or other tool telemetry;
- ignored credentials, caches, raw batch scratch space, worktrees, or pipeline
  scratch directories;
- a regular Git blob larger than 100 MiB or a symbolic link under `docs/`.

The annotation ledger remains local and user-owned. A later archival decision
may publish a content-addressed data package through a release or research-data
repository, but that is not part of the site deployment.

## Required gates

Run these from the repository root with the x86_64 project environment:

```bash
arch -x86_64 .venv/bin/python -m pytest -q
arch -x86_64 .venv/bin/python -m presidential_profiles.site
node scripts/validate_inline_js.mjs docs
arch -x86_64 .venv/bin/python -m compileall -q src
git diff --check
```

After staging the explicit release allowlist, run:

```bash
python3 scripts/audit_github_release.py origin/master
```

The public-release review must also inspect the staged diff, scan the staged
tree for credential-like material without printing secret values, and verify
that frozen paid artifacts are unchanged.

## Git and deployment sequence

1. Work on `codex/github-pages-release` without cleaning or resetting the dirty
   user-owned worktree.
2. Rebuild `docs/` only through the Python generator.
3. Stage the allowlisted paths explicitly; never use `git add -A` or `git add .`.
4. Run the staged-tree audit and review the exact staged diff.
5. Commit and push the release branch.
6. Open a pull request against `master`, confirm the changed-file and size
   boundary, and merge without rewriting the accumulated local history.
7. Wait for the `pages build and deployment` workflow to complete successfully.
8. Verify the public Story, Summary, Compare, Explore, Profiles, Data pages,
   local assets, and representative JSON/CSV downloads over HTTPS.

GitHub Pages deployment is a publication step only. The Python generator
continues to own `docs/`; the generated tree is not a hand-editing surface.

## Failure and rollback

- If a local gate fails, stop before pushing and fix the source rather than the
  generated output.
- If the pull request audit fails, add a corrective commit on the release branch.
- If Pages fails after merge, inspect the Pages workflow before changing source.
- If the deployed content is wrong, revert the release commit on `master` and
  let Pages deploy the previously reviewed `docs/` tree. Do not rewrite public
  history or reset the user-owned worktree.

## Acceptance

The release is complete only when:

- every required local gate passes;
- the release audit accepts the staged tree;
- the pull request is merged to `master`;
- GitHub Pages reports a successful deployment for the merged commit;
- representative public routes and assets return successful HTTPS responses;
- `HANDOFF.md`, `HISTORY.md`, and `TASKS.md` record the deployed commit and any
  remaining local-only artifacts.
