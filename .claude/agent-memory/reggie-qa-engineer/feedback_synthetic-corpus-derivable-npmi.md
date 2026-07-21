---
name: synthetic-corpus-derivable-npmi
description: How to unit-test NPMI/coherence code in presidential_profiles without the 36k corpus — build a 100-doc synthetic corpus whose NPMI values are hand-derivable
metadata:
  type: feedback
---

To test anything that goes through `embed_topics._fit_vectorizer` + `_npmi`
(`topic_quality.compute_coherence`, `issues.attach_coherence`), build a ~100-doc
synthetic `pd.Series` instead of mocking. The real vectorizer and real `_npmi`
then run at unit-test speed and every expected value is derivable:

- **NPMI exactly +1.0** — put N terms in the SAME doc set (`p_single == p_joint`).
- **NPMI exactly -1.0** — disjoint doc blocks (`p_joint == 0` limit branch).
- **NPMI exactly 0.0** — `p_a = p_b = 0.5`, `p_joint = 0.25` (= NPMI's null, which
  is what `COHERENCE_FLOOR` is pinned to).
- **Coverage-clause case** — terms drawn from `topics.EXTRA_STOP` (`ve/ll/don/didn/mr`)
  are stripped by the shared stop list, so the topic scores high on its 2 survivors
  and passes the floor. That is the only way to prove `MIN_SCORED_FRACTION` does
  independent work rather than duplicating the floor.

**Why:** `_fit_vectorizer` uses `min_df=20` / `max_df=0.5`, so a naive 5-doc fixture
produces an empty vocabulary and every NPMI is NaN. 100 docs with term frequencies
in [20, 50] is the smallest size that clears both filters.

**The stop-list seam.** `_fit_vectorizer` resolves `EXTRA_STOP` out of
`embed_topics`'s module globals *at call time*, so
`monkeypatch.setattr(embed_topics, "EXTRA_STOP", set(topics.EXTRA_STOP) - {...})`
swaps the scoring vocabulary without touching `topics.py`. That is how report §3's
stop-list reconciliation (Discovered 3 at −0.031 under the pre-fragment vocabulary
vs −0.047 now) is testable at all. Two useful invariants fall out: adding terms to
the vocabulary cannot change any other term's document frequency, so a
fragment-free topic's NPMI is bit-for-bit invariant (`==`, not `approx`), while a
fragment-heavy topic's score RISES when the fragments become scoreable.

**How to apply:** reuse the fixture in `tests/test_topic_quality.py::texts`.
`compute_coherence(topic_words, texts)` is deliberately pure w.r.t. the corpus —
never let a test fall back to the real `paragraphs.parquet` default; prove it by
scoring ONE topic against two corpora, not two topics against two corpora. See
[[test-toolchain]], [[triangulate-monkeypatch-harness]] and
[[report-numbers-need-artifact-anchors]].
