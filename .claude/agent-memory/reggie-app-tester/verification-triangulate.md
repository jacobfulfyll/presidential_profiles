---
name: verification-triangulate
description: How to verify triangulate.run() and the topic display-name centralization end-to-end (fast, deterministic, offline, $0)
metadata:
  type: reference
---

Verifying the method-triangulation arm (`triangulate.py` + `topic_quality.py`). Complements
[[verification-workflow]] (site build) and [[verification-embed-clusters]] (clusters).

- **`triangulate.run()` is ~1s and fully deterministic.** It reads committed parquets and writes
  `data/method_agreement.parquet` (160 rows = 16 issues overall + 16x9 = 144 per-era) and
  `data/method_compositions.parquet` (85,617 = 1,057 speeches x 81 topics, where 81 = 50 LLM
  level-2 + 16 crosswalk + 15 embed clusters). Re-running is **byte-identical** — sha the two
  parquets before/after and expect no change. Cheap enough to always run live rather than trust.
- **No API cost, and you can prove it**: inject a poisoned `sys.modules["anthropic"]` whose
  `__getattr__` raises before importing anything. The whole analysis path completes untouched.
- **Degradation matrix for the names file — two genuinely different behaviours**, both verified
  against real site builds:
  | Perturbation | Result |
  |---|---|
  | topic in `surfaced` but `display: null` | degrades to raw column name; 16 pages still built |
  | names file entirely absent | `surfaced == []` → topic **dropped**; only 15 pages built |
  The second is safe (no crash, no wrong label) but **lossy** — it silently removes a live page.
  Distinguish them in reports; "it didn't crash" hides the difference.
- **Latent slug collision**: `profiles.slug` is `re.sub(r"[^a-z]+", "-", ...)`, which strips
  digits, so `Discovered 5/6/3` all slugify to `discovered.html`. Inert today (one surfaced
  discovered topic, and it has a display name) but it bites the moment a second is surfaced
  unnamed. Check `issue_slug()` output whenever display names change.
- **Cross-check published trajectory claims against `method_agreement.parquet`, not the drift
  table.** The drift table is in-memory only; the agreement parquet carries raw `n`/`n_llm`/
  `n_corex` per era, from which `corex_rel`/`llm_rel` recompute independently. That reproduced the
  report's Infrastructure claim (CorEx 7.7% of own peak vs LLM 79.6% at era 1950) via a different
  code path — a much stronger check than reading `rename_vs_death`'s own output back.
