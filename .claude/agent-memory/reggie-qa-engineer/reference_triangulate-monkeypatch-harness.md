---
name: triangulate-monkeypatch-harness
description: triangulate.load_arms/run read module-level parquet path constants with no injection seam — monkeypatch the constants; default args are NOT monkeypatchable
metadata:
  type: reference
---

`triangulate.load_arms()` and `run()` read four real parquets and three JSONs off
**module-level constants** with no `path=` parameter. To test them hermetically,
`monkeypatch.setattr` the module attributes:

- `T.PARAGRAPHS_PATH`, `T.PARA_LABELS_PATH`, `T.CLUSTERS_PATH`,
  `T.PARAGRAPH_ANNOTATIONS_PATH` (for `load_arms`)
- plus `T.AGREEMENT_PATH`, `T.COMPOSITIONS_PATH`, `T.ISSUES_META_PATH`,
  `T.CLUSTERS_META_PATH`, and `T.load` (the `corpus.load` re-export) for `run()`

**Gotcha that costs time:** `_assert_canonical_topics(topic_sets, expected=N_LEVEL2_TOPICS)`
binds its default at def time, so monkeypatching `T.N_LEVEL2_TOPICS` does nothing.
The synthetic annotations must contain exactly 50 distinct normalized topics. Same
trap in `topic_quality.load_names(path=NAMES_PATH)` / `write_names` / `build_names` —
patch `topic_quality.load_names` itself, not `NAMES_PATH`, for the `names=None` paths.

Working harness: `tests/test_triangulate.py::_write_arms` (8 docs x 50 paragraphs,
two 30-year eras both >= 100 rows so `agreement_table`'s thin-era filter doesn't
swallow them). Read the frozen `taxonomy_v1.json` / `crosswalk_v1.json` for real —
cheap JSON, and the real 50 names / 16 keys are what make the projection tests mean
anything.

**How to apply:** reuse `_write_arms` for any future `triangulate`/`taxonomy` join
work rather than reinventing it. See [[test-toolchain]].
