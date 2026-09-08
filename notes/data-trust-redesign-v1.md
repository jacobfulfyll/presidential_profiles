# Data Trust redesign v1

Status: **implemented and verified; no deployment performed**

Implementation date: 2026-09-08

Public contracts: `data-quality-v2`, `era-boundaries-v3`, `grammar-pos-v2`, and
`data-trust-validation-protocols-v1`

This release reorganizes the site's analytical evidence around three reading depths: a quick
verdict, a visual explanation, and expandable technical evidence. It corrects and governs the
Data Quality calculations, centralizes Story-era and population contracts, strengthens the
Methods appendices, and publishes planned validation studies without implying that either study
has run.

## Public routes

The redesign preserves every existing route:

- `docs/data-quality.html` — corpus populations, missingness, cross-model reproducibility,
  zero-sum and entity case studies, uncertainty, and corrections;
- `docs/methodology.html` — auditable paragraph judgments, a worked source-to-chart example,
  population passports, agreement by metric family, and feature-engineering provenance;
- `docs/era-boundaries.html` — the current Story scheme, condition-by-boundary sensitivity,
  right-censoring, and boundary receipts;
- `docs/label-models.html` — model comparison, including the failed preregistered prediction;
- `docs/metrics.html` — the grouped, searchable metric dictionary with stable deep links.

Methods, Model Comparison, and the Metric Dictionary share local navigation for Overview,
Labeling, Evaluation, Populations, Model Comparison, Metric Dictionary, and Downloads. Data
Quality and Era Choices use compact page-specific navigation because their sections are different.
Substantive figures identify their metric and exact evidence source, and server-rendered prose or
tables remain available without JavaScript.

## Governed contracts and artifacts

| Contract | Governed source | Public projection | Contract boundary |
|---|---|---|---|
| `data-quality-v2` | Existing corpus, speaker, annotation, agreement, uncertainty, and normalization artifacts | `docs/data/quality/manifest_v2.json` plus its declared CSVs | Deterministic local audit; keys, populations, units, source hashes, models/prompts, table schemas, and validation results fail closed |
| `era-boundaries-v3` | `data/era_boundaries/meta.json` plus its hashed parquet bundle | `docs/data/era-boundaries-v3/manifest.json` plus versioned CSVs; compatible legacy URLs are retained | `ERA_PROFILE_SPECS` owns public era identity and Story anchors; sensitivity conditions and provisional terminal-era right-censoring are explicit |
| `grammar-pos-v2` | `data/grammar/manifest.json`, `speech_pos.parquet`, and `era_pos.csv` | `docs/data/grammar/manifest.json` and `era_pos.csv`; the compatible Data Quality CSV remains | Coarse POS is a Methods feature-engineering appendix; exact spaCy/model and corpus identities are pinned and stale caches are refused |
| `data-trust-validation-protocols-v1` | `data/validation_protocols/v1/manifest_v1.json` and restricted selection/assignment plans | `docs/data/validation-protocols-v1/` contains only the manifest, sampling-cell summaries, and prompt definitions declared public | Planned only; execution is blocked, no runtime model or current prices are assigned, and selected keys/coder tasks/request plans remain restricted |

The manifest and CSV projections are the source of truth for analytical values. Documentation
should name contracts and interpretation boundaries rather than duplicate figures that could
drift on a future governed rebuild.

## Protected boundaries

- The nine approved Story eras and Story content boundaries are unchanged. Era Choices explains
  and stress-tests them; it does not rewrite them.
- `data/llm_annotations/` remains frozen. No paid artifact was regenerated or hand-edited.
- The validation protocols authorize neither human-label collection nor a model call. Their public
  status is exactly: a blinded human-validity study and a decade-hidden sensitivity experiment are
  preregistered but have not run.
- Human-study selection masters, coder assignments, and the invariance request plan are restricted
  artifacts and must never be copied into the public site.
- No deployment, staging, commit, push, branch switch, or worktree cleanup is part of this release.

## Verification

Final acceptance completed on 2026-09-08 in the required x86_64 environment:

```bash
arch -x86_64 .venv/bin/python -m pytest -q
arch -x86_64 .venv/bin/python -m presidential_profiles.site
node scripts/validate_inline_js.mjs docs
node scripts/audit_data_trust_pages.mjs --docs docs
arch -x86_64 .venv/bin/python -m presidential_profiles.validation_protocols --check
git diff --check
~/.codex/scripts/reggie/reggie_doctor.sh \
  --repo /Users/jacobpress/Desktop/Projects/presidential_profiles
```

The final receipt is 3,017 passing repository tests with 64 existing warnings; 106 focused
trust/era/method tests; a canonical build of 73 HTML pages and 429 JSON shards; 129 parsing inline
scripts; clean generated-JavaScript and Python syntax; valid `data-quality-v2`,
`era-boundaries-v3`, `grammar-pos-v2`, `data-trust-validation-protocols-v1`, and whole-site
projections; and a clean whitespace check. The frozen-annotation fingerprint remains
`9a8ac49ed29d63c94f4d7bdeb9f753cf0f38b45b8202ed7e7afa4de1c4297b54`.

The executable and independent rendered reviews passed all five routes at 390, 768, and 1280
pixels, plus JavaScript-disabled evidence, keyboard focus and menu behavior, contrast, sticky
anchors, overflow, metric target sizing, fallbacks, and console diagnostics. Reggie Doctor reports
0 errors, 0 warnings, and 4 informational findings. No deployment was performed.
