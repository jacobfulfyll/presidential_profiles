---
name: project-attack-surface
description: presidential_profiles has no runtime attack surface — offline batch static-site generator; where security review effort actually pays off
metadata:
  type: project
---

`presidential_profiles` is an offline batch pipeline: it reads local parquet files (public-domain presidential speech corpus) and writes static HTML into `docs/` for GitHub Pages. There is no auth layer, no network-facing service, no runtime user input, no database, no secrets in the repo.

**Why:** The corpus is public-domain text; output is static HTML served by GitHub Pages. The only "input" is trusted local data files the pipeline itself generates.

**How to apply:** For security reviews here, most OWASP web categories (injection, authn/authz, session, CORS, CSRF, SSRF) are legitimately N/A — say so with the reason, don't manufacture findings. Effort pays off in exactly two places: (1) dependency hygiene / known CVEs (see [[dependency-cve-calibration]]), and (2) data-integrity guards that prevent silent corruption (keyed merges, `validate=`, post-merge length asserts). Note that HTML output does use `html.escape()` on quoted corpus text — fine, but it's defense-in-depth over trusted data, not an XSS boundary.

**Recurring calibration points (seen across topic/embedding modules):**
- `model2vec.StaticModel.from_pretrained(MODEL_NAME)` where `MODEL_NAME` is a hardcoded public HF id (e.g. `minishlab/potion-base-8M`, defined in `similarity.py`) is the repo's standard offline-embed pattern. It loads from the local HF cache; a cold cache falls back to a one-time HF Hub download of a fixed public model. Treat as INFO, not a network-egress finding — the model id is a module constant, never attacker-controllable, and the pattern predates any given task.
- Corpus/meta artifacts are read back with `pd.read_parquet` (pyarrow, no code-exec) and `json.loads` (safe) — never pickle/yaml. `doc_name` values are public URL slugs like `/the-presidency/presidential-speeches/...`, not local FS paths; committed `data/*.parquet` + `*_meta.json` carry only cluster ids/sizes/term lists/floats. No secrets/PII beyond the public corpus.
