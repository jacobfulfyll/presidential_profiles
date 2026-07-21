---
name: same-frame-invariance-tests
description: An order-invariance or alignment test must assert on columns that CROSS a frame boundary — asserting key->value where key and value come from the same input frame is vacuous
metadata:
  type: feedback
---

When a function joins N input frames and you write "shuffle the inputs, output
must not change", **assert on columns that came from a DIFFERENT frame than the
key you index by.** Otherwise the test is vacuous.

**Why:** in `combat.load_frame` QUALITY-CHECK (2026-07-21) the shuffle test
asserted `set_index(["doc_name","para_idx"])["party_attack"]` — but `doc_name`,
`para_idx` AND `party_attack` all come from the `annotations` frame, so their
mutual alignment survives *any* join, correct or not. Two mutations that kept the
keyed merge and then re-attached data positionally afterwards
(`df["text"] = paras["text"].to_numpy()`, same for `president`) SURVIVED the
entire 464-test suite. That is exactly the positional-alignment corruption
CLAUDE.md says this repo already shipped once, and it would have put the wrong
quotation beside the wrong president in a published report.

**How to apply:** build a fixture where every carried value is unique per key
(assert `nunique() == len(frame)` as a sanity line), then compare the full
`key -> [every cross-frame column]` mapping. Cover text/word_count from the
paragraph table, president/title/year from speeches, speech_type from speech
annotations, and any aggregated entity counts.

**Companion trap — `.sample(frac=1, random_state=N)` is not a reordering.** On a
3-row frame `random_state=45` returns the original order, so a per-frame
"was it reordered?" guard fails intermittently by seed. Use `.iloc[::-1]` for the
guaranteed case and keep random shuffles as extra coverage with a weaker guard.
See [[mutation-check-recurring-holes]] and [[combat-test-seams]].

**Same shape, sort-order variant:** a "ranks by X then Y then key" test whose
fixture gives every candidate a DIFFERENT X never exercises the tie-break, so
deleting the deterministic `(doc_name, para_idx)` tail survives. pandas'
default sort is not stable, so on the real corpus (where ties are the common
case) the selected rows become arbitrary. Add a fully-tied fixture, specified
with the docs in NON-key order so "frame order" and "key order" differ.
