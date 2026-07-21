"""Tests for `topic_quality.py` — CorEx coherence scoring + the display-name registry.

Two things this module must get right, and both are easy to regress:

1. **The coherence asymmetry.** NPMI is computed for all 22 CorEx topics but the
   `"noise"` gate is applied to the 7 *discovered* ones only. The 15 anchored
   issues are exempt BY DESIGN — their anchor sets deliberately span vocabulary
   that never co-occurs ("railroad" and "broadband" are one issue on purpose), so
   a low NPMI there is the taxonomy working, not a defect. `TestCoherenceAsymmetry`
   exists specifically to fail loudly if a future reader "fixes" this by
   thresholding anchored issues too.
2. **Names-file degradation.** A missing *name* and a missing *file* degrade
   DIFFERENTLY: the first renders the raw column (`Discovered 5`), the second
   drops the topic from the display list entirely — silently removing the live
   Security & peace page. `TestNamesFileDegradation` pins the distinction.
3. **The published numbers.** Report §3's caveat table and §4's gate table quote
   `data/issues_meta.json` and `data/topic_display_names.json` directly, and §3
   explains the one prior figure that did not reproduce by the contraction
   fragments entering `topics.EXTRA_STOP`. `TestCheckedInIssuesMeta` pins the
   artifacts and `TestStopListReconciliation` pins the mechanism, so neither
   claim can go stale while the suite stays green.

Everything runs on tiny synthetic corpora. Per CLAUDE.md we never call
`issues.build_issues()` (it fits a full CorEx model over 36k x 25k), and
`compute_coherence` is deliberately pure w.r.t. the corpus — it takes a `texts`
Series — so a 100-paragraph synthetic corpus scores against the real vectorizer
with the real stop lists and the real `_npmi`, at unit-test speed.

Synthetic-corpus arithmetic (all NPMI values below are derived, not magic):
  * `COHERENT_TERMS` occupy the SAME 30 of 100 docs. p_single == p_joint == 0.3,
    so pmi == -log(0.3) == denom and NPMI is exactly +1.0.
  * `SCRAMBLED_TERMS` occupy four DISJOINT 22-doc blocks. p_joint == 0 for every
    pair, so `_npmi`'s documented limit gives exactly -1.0.
  * `INDEPENDENT_TERMS`: p_a == p_b == 0.5 and p_joint == 0.25 == 0.5 * 0.5,
    so pmi == log(1) == 0 and NPMI is exactly 0.0 — NPMI's own null, which is
    what `COHERENCE_FLOOR` is set to.
  * `SLUDGE_TERMS` are mostly `topics.EXTRA_STOP` register artifacts, which the
    shared vectorizer strips. The two survivors co-occur perfectly, so the topic
    scores +1.0 and PASSES the floor — it is caught only by the coverage clause.
    That case is why `MIN_SCORED_FRACTION` exists and is tested end to end.

Coverage map (every public surface has at least one test):
  * `is_discovered`         .. TestIsDiscovered
  * `compute_coherence`     .. TestComputeCoherence
  * `classify_discovered`   .. TestClassifyDiscovered, TestCoherenceAsymmetry
  * `load_names`            .. TestLoadNames, TestNamesFileDegradation
  * `display_name`          .. TestDisplayName, TestNamesFileDegradation
  * `surfaced_discovered`   .. TestSurfacedDiscovered
  * `discovered_labels`     .. TestDiscoveredLabels
  * `display_issues`        .. TestDisplayIssues, TestNamesFileDegradation
  * `validate_surfaced` /
    `UnanchoredTopicError`  .. TestValidateSurfaced
  * `write_names`           .. TestWriteNames
  * `score_corex_topics`    .. TestScoreCorexTopics
  * `build_names`           .. TestBuildNames
  * `default_provenance`    .. TestDefaultProvenance
  * `issues.attach_coherence` (the consumer that must preserve the asymmetry)
                            .. TestAttachCoherenceAsymmetry

Artifact + report guards (no new public surface, but the report rests on them):
  * report §3 stop-list reconciliation .. TestStopListReconciliation
  * committed `data/issues_meta.json`  .. TestCheckedInIssuesMeta
  * committed names file               .. TestCheckedInNamesFile
"""

import json
import pathlib
import re
from datetime import date

import pandas as pd
import pytest

from presidential_profiles import embed_topics, issues, topic_quality, topics
from presidential_profiles.topic_quality import (
    AUTHORED_NAMES,
    COHERENCE_FLOOR,
    MIN_SCORED_FRACTION,
    UnanchoredTopicError,
    build_names,
    classify_discovered,
    compute_coherence,
    default_provenance,
    discovered_labels,
    display_issues,
    display_name,
    is_discovered,
    load_names,
    score_corex_topics,
    surfaced_discovered,
    validate_surfaced,
    write_names,
)

# --------------------------------------------------------------------------- #
# Synthetic corpus with KNOWN co-occurrence structure (see module docstring).
# --------------------------------------------------------------------------- #
COHERENT_TERMS = ["peace", "nuclear", "treaty", "alliance"]
SCRAMBLED_TERMS = ["railroad", "broadband", "canal", "telegraph"]
INDEPENDENT_TERMS = ["aardvark", "beetlebrow"]
SLUDGE_TERMS = ["ve", "ll", "don", "didn", "mr", "schoolhouse", "harvest"]
_FILLER = "harbor lighthouse ledger quarry"
N_DOCS = 100


@pytest.fixture(scope="module")
def texts() -> pd.Series:
    """100 synthetic paragraphs engineered so every NPMI below is hand-derivable.

    Term document-frequencies are all >= 20 and <= 50 so they survive the shared
    vectorizer's `min_df=20` / `max_df=0.5` filters — except the register
    artifacts, which are removed by the stop list on purpose.
    """
    docs = []
    for i in range(N_DOCS):
        parts = [_FILLER]
        if i < 30:                                  # all four, same 30 docs
            parts.append(" ".join(COHERENT_TERMS))
        block = i // 22                             # four disjoint 22-doc blocks
        if block < 4:
            parts.append(SCRAMBLED_TERMS[block])
        if i < 25:                                  # register artifacts + 2 content words
            parts.append(" ".join(SLUDGE_TERMS))
        if i < 50:
            parts.append(INDEPENDENT_TERMS[0])      # p = 0.50
        if i < 25 or 50 <= i < 75:
            parts.append(INDEPENDENT_TERMS[1])      # p = 0.50, p_joint = 0.25
        docs.append(" ".join(parts))
    return pd.Series(docs)


@pytest.fixture(scope="module")
def disjoint_texts() -> pd.Series:
    """A SECOND 100-paragraph corpus in which `COHERENT_TERMS` never co-occur.

    Same terms, and every one of them clears the shared vectorizer's `min_df=20`
    / `max_df=0.5` filters here (df=25 each) exactly as it does in `texts`
    (df=30 each) — so both corpora put the identical four-term vocabulary in
    front of `_npmi` and only the co-occurrence structure differs. That is what
    makes `compute_coherence`'s corpus-purity claim directly demonstrable: the
    same topic scored against both must move.
    """
    docs = []
    for i in range(N_DOCS):
        docs.append(f"{_FILLER} {COHERENT_TERMS[i // 25]}")
    return pd.Series(docs)


@pytest.fixture(scope="module")
def scored(texts) -> dict:
    """`compute_coherence` over the synthetic corpus, one topic per known case."""
    return compute_coherence(
        {
            "Discovered 1": COHERENT_TERMS,
            "Discovered 2": SCRAMBLED_TERMS,
            "Discovered 3": SLUDGE_TERMS,
            "Discovered 4": INDEPENDENT_TERMS,
        },
        texts,
        n_terms=10,
    )


def _names_file(topics: dict, surfaced: list[str]) -> dict:
    """A minimal in-memory names payload, as `load_names` would return it."""
    return {"version": 1, "provenance": {}, "topics": topics, "surfaced": surfaced}


# =========================================================================== #
# is_discovered
# =========================================================================== #
class TestIsDiscovered:
    @pytest.mark.parametrize("name", ["Discovered 1", "Discovered 5", "Discovered 12"])
    def test_free_topic_names_are_discovered(self, name):
        assert is_discovered(name) is True

    @pytest.mark.parametrize(
        "name", ["Economy & jobs", "Infrastructure", "Security & peace"]
    )
    def test_anchored_issue_names_are_not_discovered(self, name):
        assert is_discovered(name) is False

    def test_match_is_case_sensitive(self):
        """`issues.py` generates exactly `Discovered {n}`. A lowercase variant is
        a different (unknown) name, not a free topic — matching it loosely would
        let a typo'd names-file key slip through the discovered-only gate."""
        assert is_discovered("discovered 5") is False


# =========================================================================== #
# compute_coherence
# =========================================================================== #
class TestComputeCoherence:
    def test_cooccurring_theme_scores_the_upper_limit(self, scored):
        """Four terms sharing one 30-doc set: p_single == p_joint, so NPMI is
        exactly +1.0. This is the "known-good coherent theme" sanity check."""
        assert scored["Discovered 1"]["npmi"] == pytest.approx(1.0)

    def test_scrambled_pseudo_topic_scores_negative(self, scored):
        """Terms in four disjoint doc blocks never co-occur: `_npmi`'s
        `p_joint == 0` limit is -1.0. The paired half of the coherence sanity
        check — and it must land far below the coherent theme, not merely below
        the floor."""
        assert scored["Discovered 2"]["npmi"] == pytest.approx(-1.0)
        assert scored["Discovered 2"]["npmi"] < scored["Discovered 1"]["npmi"] - 1.0

    def test_independent_terms_score_exactly_the_null(self, scored):
        """p_joint == p_a * p_b makes pmi == 0, so NPMI == 0 — the value
        `COHERENCE_FLOOR` is pinned to. Word soup scores here by construction."""
        assert scored["Discovered 4"]["npmi"] == pytest.approx(0.0, abs=1e-12)

    def test_register_artifacts_are_dropped_from_the_scored_vocabulary(self, scored):
        """`topics.EXTRA_STOP` strips ve/ll/don/didn/mr before scoring, so only
        the 2 content words of a 7-term sludge topic are scoreable. Reporting
        that as `n_scored` rather than absorbing it silently is the whole point
        of the coverage clause."""
        rec = scored["Discovered 3"]
        assert (rec["n_scored"], rec["n_terms"]) == (2, 7)
        assert rec["scored_fraction"] == pytest.approx(2 / 7)

    def test_sludge_topic_still_passes_the_npmi_floor(self, scored):
        """The two survivors of the sludge topic co-occur perfectly, so it scores
        +1.0 and the floor alone would call it coherent. The coverage clause is
        therefore doing independent work, not duplicating the floor."""
        assert scored["Discovered 3"]["npmi"] == pytest.approx(1.0)
        assert scored["Discovered 3"]["npmi"] > COHERENCE_FLOOR

    def test_terms_are_truncated_to_n_terms(self, texts):
        """CorEx stores 12 top words; embedding clusters are scored on 10. Same
        vocabulary AND same term count is what makes the two sets of numbers in
        `issues_meta.json` and `paragraph_clusters_meta.json` comparable."""
        padded = COHERENT_TERMS + ["harbor", "lighthouse", "ledger", "quarry"]

        out = compute_coherence({"T": padded}, texts, n_terms=4)

        assert out["T"]["terms"] == COHERENT_TERMS
        assert out["T"]["n_terms"] == 4

    def test_topic_with_fewer_than_two_in_vocab_terms_reports_none(self, texts):
        """`_npmi` returns NaN when a topic has < 2 scoreable terms; the JSON
        writer needs `None`, not a NaN that json.dumps emits as bare `NaN`."""
        out = compute_coherence({"T": ["zzzqx", "yyywv"]}, texts, n_terms=10)

        assert out["T"]["npmi"] is None
        assert out["T"]["n_scored"] == 0
        assert out["T"]["scored_fraction"] == 0.0

    def test_scoring_is_pure_with_respect_to_the_corpus(self, texts, disjoint_texts):
        """ONE topic, two corpora, two opposite scores.

        The score must be a function of the `texts` argument alone — if the
        implementation ever fell back to the real 36k-paragraph parquet, both
        calls would return the same number. The two corpora carry the SAME four
        terms at the same document frequencies and differ only in whether those
        terms share documents, so nothing but co-occurrence structure can be
        responsible for the difference.
        """
        same_block = compute_coherence({"T": COHERENT_TERMS}, texts, n_terms=10)
        disjoint = compute_coherence({"T": COHERENT_TERMS}, disjoint_texts, n_terms=10)

        assert same_block["T"]["npmi"] == pytest.approx(1.0)
        assert disjoint["T"]["npmi"] == pytest.approx(-1.0)

    def test_row_order_of_the_corpus_does_not_change_the_score(self, texts):
        """NPMI is a bag-of-documents statistic. Reversing the Series is the same
        document multiset, so the score must be identical — this is the property
        the previous version of the purity test actually demonstrated, kept
        separately rather than conflated with corpus-purity."""
        forward = compute_coherence({"T": COHERENT_TERMS}, texts, n_terms=10)
        reversed_corpus = compute_coherence(
            {"T": COHERENT_TERMS}, pd.Series(list(texts)[::-1]), n_terms=10
        )

        assert reversed_corpus["T"]["npmi"] == forward["T"]["npmi"]


# =========================================================================== #
# classify_discovered — the pre-registered gate
# =========================================================================== #
def _rec(npmi, n_scored=10, n_terms=10):
    return {"npmi": npmi, "n_scored": n_scored, "n_terms": n_terms,
            "scored_fraction": n_scored / n_terms, "terms": ["w"] * n_terms}


class TestClassifyDiscovered:
    def test_topic_above_floor_with_full_coverage_is_coherent(self):
        out = classify_discovered({"Discovered 1": _rec(0.2)})

        assert out["Discovered 1"]["status"] == "coherent"
        assert out["Discovered 1"]["noise_reasons"] == []

    def test_topic_below_floor_is_noise(self):
        out = classify_discovered({"Discovered 3": _rec(-0.05)})

        assert out["Discovered 3"]["status"] == "noise"
        assert "floor" in out["Discovered 3"]["noise_reasons"][0]

    def test_topic_exactly_at_the_floor_is_noise(self):
        """The gate is `<=`: NPMI == 0 means the top terms co-occur exactly at
        chance, which is word soup by construction, not a borderline pass."""
        out = classify_discovered({"Discovered 3": _rec(COHERENCE_FLOOR)})

        assert out["Discovered 3"]["status"] == "noise"

    def test_unscoreable_topic_is_noise_not_a_crash(self):
        """`npmi is None` (fewer than 2 in-vocab terms) must fail the gate rather
        than raise on the `<=` comparison."""
        out = classify_discovered({"Discovered 3": _rec(None, n_scored=1)})

        assert out["Discovered 3"]["status"] == "noise"

    def test_coverage_failure_alone_marks_noise(self):
        """High NPMI on a handful of survivors does not rescue a topic whose
        defining terms are register artifacts."""
        out = classify_discovered({"Discovered 3": _rec(0.9, n_scored=2, n_terms=10)})

        assert out["Discovered 3"]["status"] == "noise"
        assert out["Discovered 3"]["noise_reasons"] == [
            "only 2/10 top terms are content words (< 50%); topic is defined by "
            "register artifacts"
        ]

    def test_coverage_exactly_at_the_minimum_passes(self):
        """`fails_coverage` is a strict `<`, so a topic sitting exactly on
        `MIN_SCORED_FRACTION` is kept."""
        n = int(MIN_SCORED_FRACTION * 10)
        out = classify_discovered({"Discovered 1": _rec(0.2, n_scored=n, n_terms=10)})

        assert out["Discovered 1"]["status"] == "coherent"

    def test_both_failures_are_reported_together(self):
        out = classify_discovered({"Discovered 3": _rec(-0.1, n_scored=1, n_terms=10)})

        assert len(out["Discovered 3"]["noise_reasons"]) == 2

    def test_original_score_fields_are_preserved(self):
        """Classification enriches the record; it must not drop the numbers the
        caveat table and `issues_meta.json` are built from."""
        out = classify_discovered({"Discovered 1": _rec(0.2)})

        assert out["Discovered 1"]["npmi"] == 0.2
        assert out["Discovered 1"]["n_scored"] == 10
        assert out["Discovered 1"]["terms"] == ["w"] * 10

    def test_end_to_end_over_the_synthetic_corpus(self, scored):
        """The gate applied to real `compute_coherence` output, not hand-built
        records: one coherent theme, one never-co-occurring topic, one register
        topic, one exactly-independent topic."""
        out = classify_discovered(scored)

        assert {k: v["status"] for k, v in out.items()} == {
            "Discovered 1": "coherent",
            "Discovered 2": "noise",
            "Discovered 3": "noise",
            "Discovered 4": "noise",
        }


# =========================================================================== #
# THE COHERENCE ASYMMETRY (scope decision D3) — the highest-value guard here
# =========================================================================== #
class TestCoherenceAsymmetry:
    """NPMI is computed for all 22 CorEx topics; the noise gate applies to the 7
    discovered ones ONLY.

    The 15 anchored issues are exempt by design: their anchor sets deliberately
    span vocabulary that never co-occurs in one paragraph, so a low NPMI is the
    taxonomy working as intended (see the COHERENCE CAVEAT in `embed_topics.py`).
    Four real anchored issues genuinely score below the sludge line today —
    Immigration -0.028, Foreign policy 0.079, Infrastructure 0.089, Civil rights
    & race 0.091. If a future reader "fixes" the apparent inconsistency by
    thresholding anchored issues, these tests fail.
    """

    ANCHORED_BELOW_FLOOR = {
        # The real values from the four-method benchmark. Immigration is BELOW
        # the floor and must still not be marked noise.
        "Immigration": _rec(-0.028),
        "Foreign policy": _rec(0.079),
        "Infrastructure": _rec(0.089),
        "Civil rights & race": _rec(0.091),
    }

    def test_anchored_issue_below_the_floor_is_not_marked_noise(self):
        """The single most inviting place to introduce a bug. Immigration scores
        -0.028 — below `COHERENCE_FLOOR` — and must NOT be classified at all."""
        out = classify_discovered({**self.ANCHORED_BELOW_FLOOR,
                                   "Discovered 5": _rec(0.195)})

        assert "Immigration" not in out
        assert all(rec["status"] != "noise" for name, rec in out.items()
                   if not is_discovered(name))

    def test_no_anchored_issue_ever_appears_in_the_classification(self):
        out = classify_discovered({**self.ANCHORED_BELOW_FLOOR,
                                   "Economy & jobs": _rec(0.31),
                                   "Discovered 1": _rec(0.13),
                                   "Discovered 3": _rec(-0.05)})

        assert set(out) == {"Discovered 1", "Discovered 3"}
        assert all(is_discovered(name) for name in out)

    def test_a_discovered_topic_at_the_same_score_is_marked_noise(self):
        """The asymmetry is about WHICH topics are gated, not about the number.
        An identical -0.028 on a discovered topic is noise; on an anchored issue
        it is exempt. Pinning both halves is what makes this a test of the
        asymmetry rather than of the floor."""
        out = classify_discovered({"Immigration": _rec(-0.028),
                                   "Discovered 3": _rec(-0.028)})

        assert out["Discovered 3"]["status"] == "noise"
        assert "Immigration" not in out

    def test_scoring_still_covers_the_anchored_issues(self, texts):
        """Exempt from the GATE, not from the SCORE — AC6's anchored-coherence
        caveat table needs those numbers, so `compute_coherence` must keep
        returning them."""
        out = compute_coherence(
            {"Infrastructure": SCRAMBLED_TERMS, "Discovered 5": COHERENT_TERMS},
            texts, n_terms=10,
        )

        assert set(out) == {"Infrastructure", "Discovered 5"}
        assert out["Infrastructure"]["npmi"] == pytest.approx(-1.0)


# =========================================================================== #
# THE STOP-LIST RECONCILIATION (report §3) — the mechanism, as a permanent guard
# =========================================================================== #
FRAGMENT_TERMS = ["ve", "ll", "don"]
FRAGMENT_TOPIC_CONTENT = ["harvest", "quarry", "lighthouse", "beacon"]
FRAGMENT_FREE_TOPIC = ["treaty", "alliance", "armistice"]


@pytest.fixture(scope="module")
def fragment_texts() -> pd.Series:
    """100 paragraphs where the 3 fragments share one 30-doc block and the 4
    content terms occupy 4 mutually disjoint 25-doc blocks.

    With fragments stopped, only the 4 disjoint content terms are scoreable and
    every pair scores NPMI's `p_joint == 0` limit. With fragments scoreable, the
    3 mutually-perfect fragment pairs plus their partial overlap with the first
    content block lift the mean — the same shape as the real Discovered 3, at a
    size where the arithmetic is checkable by hand.
    """
    docs = []
    for i in range(N_DOCS):
        parts = ["ledger", FRAGMENT_TOPIC_CONTENT[i // 25]]
        if i < 30:
            parts.append(" ".join(FRAGMENT_TERMS))
        if 60 <= i < 90:
            parts.append(" ".join(FRAGMENT_FREE_TOPIC))
        docs.append(" ".join(parts))
    return pd.Series(docs)


class TestStopListReconciliation:
    """Report §3 explains the ONE prior figure that did not reproduce.

    `task.md` published Discovered 3 at −0.031; the current yardstick gets
    −0.047. The stated cause is that contraction fragments (`ve`/`ll`/`don`/…)
    were added to `topics.EXTRA_STOP` after the prior figures were computed, so
    the older vocabulary scored those terms and today's does not. Discovered 5
    reproduced exactly (0.195) because it contains no fragments.

    Until now that reconciliation existed only as prose plus a one-off
    re-scoring. These tests pin the MECHANISM and its DIRECTION permanently:

      * fragment-heavy topic  -> score RISES when the fragments are removed from
        the stop list (i.e. under the reconstructed prior vocabulary), and its
        scoreable-term count rises with it;
      * fragment-free topic   -> score is EXACTLY invariant;
      * the `noise` verdict is unchanged either way (both values sit below the
        floor), which is why the reconciliation is a footnote and not a finding.

    Exact literals are deliberately not asserted — the report's own numbers are
    corpus-specific. What must not silently change is the sign of the effect and
    the fact that it is confined to topics containing fragments.
    """

    @pytest.fixture
    def prior_vocabulary(self, monkeypatch):
        """Reconstruct the pre-fragment stop list, exactly as §3 describes.

        `_fit_vectorizer` reads `EXTRA_STOP` out of `embed_topics`'s globals at
        call time, so patching the module attribute is what swaps the vocabulary
        without touching `topics.py`.
        """
        monkeypatch.setattr(
            embed_topics, "EXTRA_STOP", set(topics.EXTRA_STOP) - set(FRAGMENT_TERMS)
        )

    @staticmethod
    def _score(texts):
        return compute_coherence(
            {"Discovered 3": FRAGMENT_TERMS + FRAGMENT_TOPIC_CONTENT,
             "Discovered 5": FRAGMENT_FREE_TOPIC},
            texts,
            n_terms=10,
        )

    def test_the_fragments_are_in_the_shared_stop_list(self):
        """The premise of the whole reconciliation, asserted at source: if these
        ever leave `topics.EXTRA_STOP` the §3 explanation stops being true and
        the two tests below stop measuring anything."""
        assert set(FRAGMENT_TERMS) <= topics.EXTRA_STOP

    def test_fragments_are_dropped_from_the_scored_vocabulary_today(
        self, fragment_texts
    ):
        """Today's yardstick can only score the content terms — 4 of 7. The real
        Discovered 3 shows the same shape at 7 of 10."""
        rec = self._score(fragment_texts)["Discovered 3"]

        assert (rec["n_scored"], rec["n_terms"]) == (4, 7)

    def test_removing_the_fragments_from_the_stop_list_makes_them_scoreable(
        self, fragment_texts, prior_vocabulary
    ):
        rec = self._score(fragment_texts)["Discovered 3"]

        assert (rec["n_scored"], rec["n_terms"]) == (7, 7)

    def test_the_fragment_topic_scores_HIGHER_under_the_prior_vocabulary(
        self, fragment_texts, monkeypatch
    ):
        """THE DIRECTION. §3 claims the prior stop list gives the *less negative*
        figure (−0.0307 vs −0.0468) because the fragments themselves co-occur.
        A future change that inverted this would make the report's reconciliation
        wrong while every structural test still passed."""
        current = self._score(fragment_texts)["Discovered 3"]["npmi"]

        monkeypatch.setattr(
            embed_topics, "EXTRA_STOP", set(topics.EXTRA_STOP) - set(FRAGMENT_TERMS)
        )
        prior = self._score(fragment_texts)["Discovered 3"]["npmi"]

        assert prior > current

    def test_a_fragment_free_topic_is_exactly_invariant(
        self, fragment_texts, monkeypatch
    ):
        """The complement, and the reason ONE prior figure diverged rather than
        all of them: adding 3 terms to the vocabulary cannot change any other
        term's document frequency, so a topic with no fragments scores bit for
        bit the same. Asserted with `==`, not `approx`."""
        current = self._score(fragment_texts)["Discovered 5"]["npmi"]

        monkeypatch.setattr(
            embed_topics, "EXTRA_STOP", set(topics.EXTRA_STOP) - set(FRAGMENT_TERMS)
        )
        prior = self._score(fragment_texts)["Discovered 5"]["npmi"]

        assert prior == current
        assert current > COHERENCE_FLOOR  # ...and it is the COHERENT one

    def test_the_noise_verdict_is_unchanged_under_either_vocabulary(
        self, fragment_texts, monkeypatch
    ):
        """Why §3 is a footnote and not a finding: both values sit below the
        floor, so the pre-registered gate reaches the same verdict either way."""
        current = classify_discovered(self._score(fragment_texts))

        monkeypatch.setattr(
            embed_topics, "EXTRA_STOP", set(topics.EXTRA_STOP) - set(FRAGMENT_TERMS)
        )
        prior = classify_discovered(self._score(fragment_texts))

        assert {k: v["status"] for k, v in current.items()} == {
            "Discovered 3": "noise", "Discovered 5": "coherent",
        }
        assert ({k: v["status"] for k, v in prior.items()}
                == {k: v["status"] for k, v in current.items()})


class TestAttachCoherenceAsymmetry:
    """`issues.attach_coherence` is the consumer that writes the asymmetry into
    `issues_meta.json`; it must not collapse the two fields into one."""

    def test_coherence_covers_all_topics_but_status_covers_only_discovered(
        self, texts
    ):
        meta = {
            "issues": ["Infrastructure", "Immigration"],
            "topic_words": {
                "Infrastructure": SCRAMBLED_TERMS,
                "Immigration": SCRAMBLED_TERMS,
                "Discovered 1": COHERENT_TERMS,
                "Discovered 3": SLUDGE_TERMS,
            },
        }

        out = issues.attach_coherence(meta, texts)

        assert set(out["coherence"]) == set(meta["topic_words"])
        assert set(out["discovered_status"]) == {"Discovered 1", "Discovered 3"}
        # Both anchored issues score -1.0 here; neither may be labelled noise.
        assert out["coherence"]["Infrastructure"]["npmi"] == pytest.approx(-1.0)
        assert "Infrastructure" not in out["discovered_status"]

    def test_names_file_pointer_is_recorded(self, texts):
        meta = {"issues": [], "topic_words": {"Discovered 1": COHERENT_TERMS}}

        out = issues.attach_coherence(meta, texts)

        assert out["names_file"] == topic_quality.NAMES_PATH.name


class TestRefreshMetaCoherence:
    """`issues.refresh_meta_coherence` enriches the persisted `issues_meta.json`
    IN PLACE. Its whole reason to exist is that `build_issues()` fits a full CorEx
    model over 36k x 25k and its labels are already frozen into
    `paragraph_issues.parquet` — enriching the metadata must not reproduce that fit.
    """

    @pytest.fixture
    def refreshed(self, tmp_path, monkeypatch, texts):
        meta_path = tmp_path / "issues_meta.json"
        meta_path.write_text(json.dumps({
            "issues": ["Infrastructure"],
            "topic_words": {
                "Infrastructure": SCRAMBLED_TERMS,
                "Discovered 1": COHERENT_TERMS,
                "Discovered 3": SLUDGE_TERMS,
                "Discovered 7": ["zzzqx", "yyywv"],
            },
            "n_paragraphs": 36229,
        }))
        paras = tmp_path / "paragraphs.parquet"
        pd.DataFrame({"text": list(texts)}).to_parquet(paras, index=False)
        monkeypatch.setattr(issues, "ISSUES_META_PATH", meta_path)
        monkeypatch.setattr(issues, "PARAGRAPHS_PATH", paras)
        # A CorEx fit here would be the exact regression this function prevents.
        monkeypatch.setattr(issues.ct, "Corex", lambda *a, **k: pytest.fail(
            "refresh_meta_coherence refitted CorEx"))
        return issues.refresh_meta_coherence(), meta_path

    def test_existing_meta_keys_survive(self, refreshed):
        meta, _ = refreshed

        assert meta["issues"] == ["Infrastructure"]
        assert meta["n_paragraphs"] == 36229
        assert set(meta["topic_words"]) == {"Infrastructure", "Discovered 1",
                                            "Discovered 3", "Discovered 7"}

    def test_coherence_and_status_keep_the_asymmetry_on_disk(self, refreshed):
        _, path = refreshed
        written = json.loads(path.read_text())

        assert set(written["coherence"]) == set(written["topic_words"])
        assert set(written["discovered_status"]) == {"Discovered 1", "Discovered 3",
                                                     "Discovered 7"}
        assert "Infrastructure" not in written["discovered_status"]

    def test_unscoreable_topic_serializes_as_null_not_bare_nan(self, refreshed):
        """`compute_coherence` converts NaN to `None` precisely so this file stays
        valid JSON — `json.dumps` would otherwise emit a bare `NaN` literal that
        strict parsers reject."""
        _, path = refreshed
        raw = path.read_text()

        assert "NaN" not in raw
        assert json.loads(raw)["coherence"]["Discovered 7"]["npmi"] is None

    def test_returned_meta_matches_the_file(self, refreshed):
        meta, path = refreshed

        assert json.loads(path.read_text()) == meta


# =========================================================================== #
# Names file: load / read helpers
# =========================================================================== #
class TestLoadNames:
    def test_missing_file_returns_the_empty_shape_without_raising(self, tmp_path):
        out = load_names(tmp_path / "does-not-exist.json")

        assert out == {"topics": {}, "surfaced": []}

    def test_existing_file_is_returned_verbatim(self, tmp_path):
        payload = _names_file({"Discovered 5": {"display": "Security & peace"}},
                              ["Discovered 5"])
        path = tmp_path / "names.json"
        path.write_text(json.dumps(payload))

        assert load_names(path) == payload


class TestDisplayName:
    def test_named_topic_renders_its_display_label(self):
        names = _names_file({"Discovered 5": {"display": "Security & peace"}}, [])

        assert display_name("Discovered 5", names) == "Security & peace"

    def test_topic_missing_from_the_file_falls_back_to_the_column(self):
        assert display_name("Discovered 6", _names_file({}, [])) == "Discovered 6"

    def test_topic_present_with_a_null_display_falls_back_to_the_column(self):
        """A noise topic is recorded but deliberately unnamed. It must render as
        its column rather than as `None`, which would put the string "None" on a
        page."""
        names = _names_file({"Discovered 3": {"display": None, "status": "noise"}}, [])

        assert display_name("Discovered 3", names) == "Discovered 3"

    def test_anchored_issue_passes_through_unchanged(self):
        assert display_name("Economy & jobs", _names_file({}, [])) == "Economy & jobs"

    def test_default_reads_the_names_file(self, monkeypatch):
        """With no `names` argument the helper reads the file itself — the path
        `profiles_site`/`explorer` use at import time."""
        monkeypatch.setattr(
            topic_quality, "load_names",
            lambda *a, **k: _names_file({"Discovered 5": {"display": "Peace"}}, []),
        )

        assert display_name("Discovered 5") == "Peace"


class TestSurfacedDiscovered:
    def test_returns_column_keys_not_display_labels(self):
        """Callers index DataFrames with these, so they must stay column names
        even though the file also carries a human label."""
        names = _names_file({"Discovered 5": {"display": "Security & peace"}},
                            ["Discovered 5"])

        assert surfaced_discovered(names) == ["Discovered 5"]

    def test_empty_when_nothing_is_surfaced(self):
        assert surfaced_discovered(_names_file({"Discovered 3": {}}, [])) == []

    def test_result_is_a_copy_not_the_stored_list(self):
        """Callers append to the returned list (`display_issues` does); mutating
        the loaded payload would corrupt every later read in the same process."""
        names = _names_file({}, ["Discovered 5"])

        surfaced_discovered(names).append("Discovered 6")

        assert names["surfaced"] == ["Discovered 5"]


class TestDiscoveredLabels:
    def test_named_topics_map_to_their_labels(self):
        names = _names_file(
            {"Discovered 5": {"display": "Security & peace"},
             "Discovered 4": {"display": "Debate & Interview Exchanges"}},
            ["Discovered 5"],
        )

        assert discovered_labels(names) == {
            "Discovered 5": "Security & peace",
            "Discovered 4": "Debate & Interview Exchanges",
        }

    def test_named_but_unsurfaced_topics_are_included(self):
        """A persisted issue card can still hold a column name for a topic that
        is no longer surfaced; `.get(name, name)` must find a real label."""
        names = _names_file({"Discovered 4": {"display": "Debate"}}, [])

        assert discovered_labels(names) == {"Discovered 4": "Debate"}

    def test_noise_topics_are_absent_so_callers_fall_back_to_the_column(self):
        names = _names_file({"Discovered 3": {"display": None, "status": "noise"}}, [])

        labels = discovered_labels(names)

        assert "Discovered 3" not in labels
        assert labels.get("Discovered 3", "Discovered 3") == "Discovered 3"


class TestDisplayIssues:
    def test_appends_surfaced_topics_to_the_anchored_issues(self):
        """Replaces the `issue_names + ["Discovered 5"]` literal that was copied
        into six call sites across five modules."""
        names = _names_file({"Discovered 5": {"display": "Security & peace"}},
                            ["Discovered 5"])

        assert display_issues(["Economy & jobs", "Taxes & budget"], names) == [
            "Economy & jobs", "Taxes & budget", "Discovered 5",
        ]

    def test_returns_column_names_so_it_stays_a_dataframe_selector(self):
        names = _names_file({"Discovered 5": {"display": "Security & peace"}},
                            ["Discovered 5"])
        frame = pd.DataFrame({"Economy & jobs": [1], "Discovered 5": [2]})

        selected = display_issues(["Economy & jobs"], names)

        assert list(frame[selected].columns) == selected

    def test_input_list_is_not_mutated(self):
        issue_names = ["Economy & jobs"]

        display_issues(issue_names, _names_file({}, ["Discovered 5"]))

        assert issue_names == ["Economy & jobs"]


# =========================================================================== #
# Names-file degradation — the two paths behave DIFFERENTLY, on purpose
# =========================================================================== #
class TestNamesFileDegradation:
    """The plan requires a deliberately missing name to degrade gracefully. It
    does — but a missing FILE degrades differently, and the difference is easy to
    regress into "they both just work".

    * missing NAME  -> topic stays in the display list, renders as its column.
    * missing FILE  -> topic is dropped from the display list entirely, which
      silently removes the live Security & peace page. Safe (no crash, no wrong
      label) but LOSSY, and not the same failure.
    """

    ISSUES = ["Economy & jobs", "Taxes & budget"]

    def test_missing_name_keeps_the_topic_and_shows_the_column(self):
        names = _names_file({"Discovered 5": {"display": None}}, ["Discovered 5"])

        display = display_issues(self.ISSUES, names)

        assert display == self.ISSUES + ["Discovered 5"]
        assert display_name("Discovered 5", names) == "Discovered 5"
        assert discovered_labels(names) == {}

    def test_topic_absent_from_the_file_entirely_still_renders_if_surfaced(self):
        """A hand-edited file can surface a topic it never described. The label
        falls back to the column rather than KeyError-ing a page build."""
        names = _names_file({}, ["Discovered 6"])

        assert display_issues(self.ISSUES, names)[-1] == "Discovered 6"
        assert display_name("Discovered 6", names) == "Discovered 6"

    def test_missing_file_drops_surfaced_topics_instead_of_renaming_them(
        self, tmp_path
    ):
        """THE DISTINCTION. No file means `surfaced == []`, so the discovered
        topic disappears from the display list. No crash — but the Security &
        peace page is gone, not renamed. Pinning this stops a future reader from
        assuming the two degraded paths are interchangeable."""
        names = load_names(tmp_path / "absent.json")

        display = display_issues(self.ISSUES, names)

        assert display == self.ISSUES
        assert "Discovered 5" not in display
        # ...while the label helper still degrades gracefully for anything a
        # caller already holds.
        assert display_name("Discovered 5", names) == "Discovered 5"
        assert discovered_labels(names) == {}

    def test_missing_file_leaves_the_anchored_issues_intact(self, tmp_path):
        """The loss is bounded: only the discovered tail is dropped. The 15
        curated issues never depend on the names file."""
        names = load_names(tmp_path / "absent.json")

        assert display_issues(self.ISSUES, names) == self.ISSUES


# =========================================================================== #
# validate_surfaced / UnanchoredTopicError
# =========================================================================== #
class TestValidateSurfaced:
    def test_topic_with_anchors_passes(self):
        """`Discovered 5` is the one discovered topic with an entry in
        `profiles._EXTRA_ANCHORS`, which is why it is the one that is surfaced."""
        validate_surfaced(["Discovered 5"])  # must not raise

    def test_anchored_issue_passes(self):
        validate_surfaced(["Economy & jobs", "Infrastructure"])

    def test_empty_list_passes(self):
        validate_surfaced([])

    def test_topic_without_anchors_raises_the_named_error(self):
        """`issues_site` indexes `anchors_all[name]`, so surfacing an unanchored
        topic would otherwise raise a bare KeyError from deep inside a page
        build."""
        with pytest.raises(UnanchoredTopicError) as exc:
            validate_surfaced(["Discovered 6"])

        assert "Discovered 6" in str(exc.value)
        assert "_EXTRA_ANCHORS" in str(exc.value)

    def test_error_names_every_offender(self):
        with pytest.raises(UnanchoredTopicError) as exc:
            validate_surfaced(["Discovered 5", "Discovered 6", "Discovered 7"])

        assert "Discovered 6" in str(exc.value)
        assert "Discovered 7" in str(exc.value)

    def test_it_is_a_valueerror_so_existing_handlers_still_catch_it(self):
        with pytest.raises(ValueError):
            validate_surfaced(["Discovered 6"])

    def test_read_path_does_NOT_validate(self):
        """Documented-by-test asymmetry: enforcement is on the WRITE path only.
        A hand-edited file surfacing an unanchored topic must still render the
        rest of the site — `display_issues`'s job is to degrade, not to crash.
        The KeyError it defers to `issues_site` is the accepted trade."""
        names = _names_file({"Discovered 6": {"display": "Territory"}},
                            ["Discovered 6"])

        assert display_issues(["Economy & jobs"], names) == [
            "Economy & jobs", "Discovered 6",
        ]
        assert surfaced_discovered(names) == ["Discovered 6"]
        assert display_name("Discovered 6", names) == "Territory"


# =========================================================================== #
# write_names
# =========================================================================== #
class TestWriteNames:
    def test_payload_shape_and_round_trip(self, tmp_path):
        path = tmp_path / "names.json"
        topics = {"Discovered 5": {"display": "Security & peace", "status": "coherent"}}

        payload = write_names(topics, ["Discovered 5"], {"authored_by": "x"}, path)

        assert payload == {
            "version": 1,
            "provenance": {"authored_by": "x"},
            "topics": topics,
            "surfaced": ["Discovered 5"],
        }
        assert json.loads(path.read_text()) == payload
        assert load_names(path) == payload

    def test_written_file_is_consumable_by_the_read_helpers(self, tmp_path):
        path = tmp_path / "names.json"
        write_names({"Discovered 5": {"display": "Security & peace"}},
                    ["Discovered 5"], {}, path)

        names = load_names(path)

        assert display_issues(["Economy & jobs"], names) == [
            "Economy & jobs", "Discovered 5",
        ]
        assert display_name("Discovered 5", names) == "Security & peace"


# =========================================================================== #
# score_corex_topics
# =========================================================================== #
def _meta_path(tmp_path, monkeypatch, *, issues_list, topic_words) -> None:
    path = tmp_path / "issues_meta.json"
    path.write_text(json.dumps({"issues": issues_list, "topic_words": topic_words}))
    monkeypatch.setattr(topic_quality, "ISSUES_META_PATH", path)
    return path


def _synthetic_meta(n_anchored: int = 2) -> tuple[list[str], dict]:
    """`issues_meta.json`-shaped inputs: `n_anchored` issues + exactly
    `N_FREE_TOPICS` discovered topics, which is what `score_corex_topics`
    validates against."""
    anchored = [f"Anchored {i}" for i in range(n_anchored)]
    words = {name: SCRAMBLED_TERMS for name in anchored}
    for i in range(1, topic_quality.N_FREE_TOPICS + 1):
        words[f"Discovered {i}"] = (
            SLUDGE_TERMS if i == 3 else COHERENT_TERMS
        )
    return anchored, words


class TestScoreCorexTopics:
    def test_scores_every_topic_in_the_persisted_meta(self, tmp_path, monkeypatch, texts):
        anchored, words = _synthetic_meta()
        _meta_path(tmp_path, monkeypatch, issues_list=anchored, topic_words=words)

        out = score_corex_topics(texts)

        assert set(out) == set(words)
        assert len(out) == len(anchored) + topic_quality.N_FREE_TOPICS

    def test_topic_count_mismatch_raises(self, tmp_path, monkeypatch, texts):
        """A meta whose `topic_words` and `issues` disagree means the CorEx fit
        and the persisted metadata have drifted apart — refuse rather than score
        a partial taxonomy."""
        anchored, words = _synthetic_meta()
        words.pop("Discovered 7")
        _meta_path(tmp_path, monkeypatch, issues_list=anchored, topic_words=words)

        with pytest.raises(ValueError, match="scored 8 topics but"):
            score_corex_topics(texts)

    def test_reads_the_corpus_parquet_when_texts_is_omitted(
        self, tmp_path, monkeypatch, texts
    ):
        """The default path reads `paragraphs.parquet`; it must never refit
        CorEx to recover the top terms."""
        anchored, words = _synthetic_meta()
        _meta_path(tmp_path, monkeypatch, issues_list=anchored, topic_words=words)
        paras = tmp_path / "paragraphs.parquet"
        pd.DataFrame({"text": list(texts)}).to_parquet(paras, index=False)
        monkeypatch.setattr(topic_quality, "PARAGRAPHS_PATH", paras)
        monkeypatch.setattr(
            issues, "build_issues",
            lambda *a, **k: pytest.fail("build_issues() must never be called"),
        )

        out = score_corex_topics()

        assert out["Discovered 1"]["npmi"] == pytest.approx(1.0)


# =========================================================================== #
# build_names — the full write pipeline
# =========================================================================== #
class TestBuildNames:
    @pytest.fixture
    def built(self, tmp_path, monkeypatch, texts):
        anchored, words = _synthetic_meta()
        _meta_path(tmp_path, monkeypatch, issues_list=anchored, topic_words=words)
        path = tmp_path / "topic_display_names.json"
        return build_names(texts, path), path

    def test_persists_a_readable_names_file(self, built):
        payload, path = built

        assert json.loads(path.read_text()) == payload
        assert payload["version"] == 1

    def test_only_discovered_topics_land_in_topics(self, built):
        payload, _ = built

        assert all(is_discovered(name) for name in payload["topics"])
        assert len(payload["topics"]) == topic_quality.N_FREE_TOPICS

    def test_anchored_issues_are_recorded_as_a_caveat_table_only(self, built):
        """Scored for AC6's caveat table, never classified, never gated. Kept in
        a separate key so no consumer can mistake them for surfacing candidates."""
        payload, path = built
        caveat = payload["anchored_coherence_caveat"]

        assert set(caveat) == {"Anchored 0", "Anchored 1"}
        assert all("status" not in rec and "surface" not in rec
                   for rec in caveat.values())
        assert json.loads(path.read_text())["anchored_coherence_caveat"] == caveat

    def test_surfaces_the_authored_topic(self, built):
        payload, _ = built

        assert payload["surfaced"] == ["Discovered 5"]
        assert payload["topics"]["Discovered 5"]["display"] == "Security & peace"

    def test_surfaced_list_is_sorted(self, built):
        payload, _ = built

        assert payload["surfaced"] == sorted(payload["surfaced"])

    def test_noise_topic_is_never_named(self, built):
        """`Discovered 3` is the sludge topic in the synthetic meta; the
        pre-registered gate must strip its authored name."""
        payload, _ = built
        rec = payload["topics"]["Discovered 3"]

        assert rec["status"] == "noise"
        assert rec["display"] is None
        assert rec["surface"] is False
        assert rec["noise_reasons"]

    def test_editorial_judgment_cannot_override_the_gate(
        self, tmp_path, monkeypatch, texts
    ):
        """The gate is quantitative and pre-registered; `surface` is editorial.
        Editorial may only ever REMOVE, never restore — otherwise the gate could
        be laundered through a hand-edited constant."""
        anchored, words = _synthetic_meta()
        _meta_path(tmp_path, monkeypatch, issues_list=anchored, topic_words=words)
        monkeypatch.setattr(topic_quality, "AUTHORED_NAMES", {
            **AUTHORED_NAMES,
            "Discovered 3": {"display": "Smuggled In", "surface": True,
                             "rationale": "editorial override attempt"},
        })

        payload = build_names(texts, tmp_path / "names.json")

        assert payload["topics"]["Discovered 3"]["display"] is None
        assert payload["topics"]["Discovered 3"]["surface"] is False
        assert "Discovered 3" not in payload["surfaced"]

    def test_surfacing_an_unanchored_topic_raises_at_write_time(
        self, tmp_path, monkeypatch, texts
    ):
        """`validate_surfaced` is enforced on the WRITE path. `Discovered 6` is
        coherent in the synthetic meta but has no anchors, so build must refuse
        before persisting a file that would KeyError a site build."""
        anchored, words = _synthetic_meta()
        _meta_path(tmp_path, monkeypatch, issues_list=anchored, topic_words=words)
        monkeypatch.setattr(topic_quality, "AUTHORED_NAMES", {
            **AUTHORED_NAMES,
            "Discovered 6": {"display": "Territory", "surface": True,
                             "rationale": "no anchors exist for this"},
        })
        path = tmp_path / "names.json"

        with pytest.raises(UnanchoredTopicError, match="Discovered 6"):
            build_names(texts, path)

        assert not path.exists()

    def test_provenance_records_that_no_api_call_was_made(self, built):
        payload, _ = built
        prov = payload["provenance"]

        assert "NO Anthropic API call" in prov["method"]
        assert prov["gate"]["pre_registered"] is True
        assert prov["gate"]["coherence_floor"] == COHERENCE_FLOOR

    def test_records_the_scores_the_gate_was_applied_to(self, built):
        """Every topic carries the numbers its status was derived from, so the
        decision is auditable from the file alone."""
        payload, _ = built
        rec = payload["topics"]["Discovered 1"]

        assert rec["npmi"] == pytest.approx(1.0)
        assert rec["n_scored"] == rec["n_terms"] == len(COHERENT_TERMS)
        assert rec["terms"] == COHERENT_TERMS


class TestDefaultProvenance:
    def test_stamps_model_date_and_the_gate_constants(self, monkeypatch):
        """The date is read off a frozen clock rather than compared against a
        second `date.today()` call: the two calls straddle midnight for a ~1 ms
        window every night, and a provenance stamp is exactly the kind of thing
        nobody wants to see fail once a year for no reason."""
        class _Frozen(date):
            @classmethod
            def today(cls):
                return date(2026, 7, 21)

        monkeypatch.setattr(topic_quality, "date", _Frozen)

        prov = default_provenance("top terms from issues_meta.json")

        assert prov["authored_by"] == "claude-opus-4-8"
        assert prov["date"] == "2026-07-21"
        assert prov["evidence"] == "top terms from issues_meta.json"
        assert prov["gate"] == {
            "coherence_floor": COHERENCE_FLOOR,
            "min_scored_fraction": MIN_SCORED_FRACTION,
            "pre_registered": True,
            "applies_to": "discovered topics only — anchored issues are exempt "
                          "by design (see embed_topics.py COHERENCE CAVEAT)",
        }

    def test_the_asymmetry_is_recorded_in_the_artifact_itself(self):
        """A reader of the names file must be able to see that the gate is
        discovered-only without reading the source."""
        assert "discovered topics only" in default_provenance("e")["gate"]["applies_to"]


class TestAuthoredNames:
    def test_covers_every_free_topic(self):
        assert set(AUTHORED_NAMES) == {
            f"Discovered {i}" for i in range(1, topic_quality.N_FREE_TOPICS + 1)
        }

    def test_security_and_peace_is_the_only_surfaced_topic(self):
        surfaced = [k for k, v in AUTHORED_NAMES.items() if v.get("surface")]

        assert surfaced == ["Discovered 5"]

    def test_security_and_peace_label_matches_the_crosswalk_key(self):
        """Load-bearing string: it is `taxonomy.SECURITY_PEACE`, a key of
        `crosswalk_v1.json`, and it slugifies to the live
        docs/issues/security-and-peace.html page."""
        from presidential_profiles.taxonomy import SECURITY_PEACE

        assert AUTHORED_NAMES["Discovered 5"]["display"] == SECURITY_PEACE

    def test_every_surfaced_topic_has_anchors(self):
        """The same invariant `build_names` enforces, asserted against the
        checked-in constant so a hand edit fails here rather than at site build."""
        validate_surfaced([k for k, v in AUTHORED_NAMES.items() if v.get("surface")])

    def test_every_topic_carries_a_rationale(self):
        assert len(AUTHORED_NAMES) == topic_quality.N_FREE_TOPICS  # not vacuous
        assert all(v["rationale"] for v in AUTHORED_NAMES.values())


# =========================================================================== #
# AC5 — the six hardcode sites now read from the names file
# =========================================================================== #
class TestDisplayNameCentralization:
    """`issue_names + ["Discovered 5"]` was copied into six call sites across five
    modules, and five further `"Discovered 5"` / `"Security & peace"` literals sat
    alongside them. Those literals diverge silently the moment the names file
    changes: the display list and the label would disagree with no error anywhere.

    These are source-text assertions rather than behavioural ones on purpose —
    the failure mode is a NEW copy of the literal appearing in a sixth place,
    which no behavioural test on the existing five can see.
    """

    DISPLAY_MODULES = ["site.py", "explorer.py", "issues_site.py", "profiles.py",
                       "profiles_site.py"]

    @pytest.fixture
    def sources(self) -> dict[str, str]:
        import presidential_profiles

        root = pathlib.Path(presidential_profiles.__file__).parent
        return {name: (root / name).read_text() for name in self.DISPLAY_MODULES}

    def test_no_module_appends_a_hardcoded_discovered_topic(self, sources):
        """The AC5 pattern itself. Dict KEYS like `_EXTRA_ANCHORS["Discovered 5"]`
        are legitimate — those key on the stable column name, which is exactly
        what the names file is designed to preserve."""
        pattern = re.compile(r'\+\s*\[\s*["\']Discovered \d+["\']\s*\]')

        offenders = {n for n, src in sources.items() if pattern.search(src)}

        assert offenders == set()

    def test_no_display_module_hardcodes_the_security_and_peace_label(self, sources):
        offenders = {n for n, src in sources.items() if "Security & peace" in src}

        assert offenders == set()

    @pytest.mark.parametrize(
        "module", ["site.py", "explorer.py", "issues_site.py", "profiles.py",
                   "profiles_site.py"]
    )
    def test_every_display_module_routes_through_the_registry(self, sources, module):
        """Requires a real import AND a real call, not just the substring — a
        comment mentioning the registry must not satisfy this."""
        src = sources[module]

        assert re.search(r"^from \. import .*\btopic_quality\b", src, re.M)
        assert re.search(r"\btopic_quality\.\w+\(", src)

    def test_profiles_site_labels_come_from_the_names_file(self):
        """`profiles_site.DISCOVERED_LABELS` is import-time state read from the
        names file — `site.py`, `issues_site.py` and `compare_site.py` all render
        through it, so it must stay the registry's output and not drift back to a
        literal dict."""
        from presidential_profiles import profiles_site

        assert profiles_site.DISCOVERED_LABELS == discovered_labels()

    def test_display_list_matches_the_registry_for_the_real_names_file(self):
        """End of the wiring: the 15 anchored issues plus whatever the checked-in
        names file surfaces."""
        names = load_names()

        assert display_issues(list(issues.ISSUE_ANCHORS)) == (
            list(issues.ISSUE_ANCHORS) + names["surfaced"]
        )


# =========================================================================== #
# The checked-in names file (cheap structural read of the real artifact)
# =========================================================================== #
class TestCheckedInNamesFile:
    @pytest.fixture
    def names(self):
        return load_names()

    def test_has_the_expected_shape(self, names):
        assert set(names) >= {"version", "provenance", "topics", "surfaced"}

    def test_describes_every_free_topic(self, names):
        assert set(names["topics"]) == {
            f"Discovered {i}" for i in range(1, topic_quality.N_FREE_TOPICS + 1)
        }

    def test_surfaced_topics_are_all_described_coherent_and_anchored(self, names):
        assert names["surfaced"], "nothing surfaced — the loop below would be vacuous"
        for col in names["surfaced"]:
            assert names["topics"][col]["status"] == "coherent"
            assert names["topics"][col]["display"]
        validate_surfaced(names["surfaced"])

    def test_noise_topics_are_unnamed_and_unsurfaced(self, names):
        noise = [c for c, r in names["topics"].items() if r["status"] == "noise"]

        assert noise, "no noise topic on disk — the loop below would be vacuous"
        for col in noise:
            assert names["topics"][col]["display"] is None
            assert col not in names["surfaced"]


# =========================================================================== #
# The checked-in issues_meta.json coherence block (report §3 / §4 numbers)
# =========================================================================== #
class TestCheckedInIssuesMeta:
    """Numeric regression guard on the committed `data/issues_meta.json`.

    Report §3 claims the identical yardstick "reproduce[s] the previously
    published anchored figures to three decimals (Immigration −0.028, Foreign
    policy 0.079, Infrastructure 0.089, Civil rights 0.091)". Those four numbers
    otherwise appear in this suite only as hand-written `_rec()` inputs to
    `TestCoherenceAsymmetry` — which pins the GATE's behaviour at those values
    but says nothing about whether the artifact still holds them. Without the
    tests below, `issues_meta.json` could be regenerated on a drifted vocabulary
    and every existing test would stay green while the report's central §3 claim
    quietly became false.

    Tolerance is `5e-4`: the published figures are 3-decimal, so anything that
    still rounds to the published value passes and anything that does not fails.
    """

    PUBLISHED_ANCHORED = {
        "Immigration": -0.028,
        "Foreign policy": 0.079,
        "Infrastructure": 0.089,
        "Civil rights & race": 0.091,
    }
    # §3's reconciliation and §4's gate table. Discovered 5 is the figure that
    # reproduced across the stop-list change; Discovered 3 is the one that did
    # not (see TestStopListReconciliation for the mechanism).
    PUBLISHED_DISCOVERED = {"Discovered 3": -0.047, "Discovered 5": 0.195}
    THREE_DECIMALS = 5e-4

    @pytest.fixture
    def meta(self):
        return json.loads(issues.ISSUES_META_PATH.read_text())

    def test_every_corex_topic_is_scored(self, meta):
        assert set(meta["coherence"]) == set(meta["topic_words"])
        assert len(meta["coherence"]) == (
            len(meta["issues"]) + topic_quality.N_FREE_TOPICS
        )

    def test_coherence_was_scored_on_the_cluster_yardstick_corpus(self, meta):
        """The recorded scored-count must match the corpus embed_topics fitted on.

        `coherence_n_paragraphs_scored` is otherwise a number nothing compares
        against. The report's §3 claim is that these NPMI values share a yardstick
        with `paragraph_clusters_meta.json` — which only holds if both were scored
        over the same paragraph set. Scoring the speeches-merged frame instead of
        raw `paragraphs.parquet` would shift the vocabulary and silently break
        that comparability while every other assertion here stayed green.
        """
        clusters_meta = json.loads(embed_topics.CLUSTERS_META_PATH.read_text())
        assert (
            meta["coherence_n_paragraphs_scored"] == clusters_meta["n_paragraphs"]
        )

    @pytest.mark.parametrize("issue", sorted(PUBLISHED_ANCHORED))
    def test_published_anchored_caveat_figure_still_reproduces(self, meta, issue):
        assert meta["coherence"][issue]["npmi"] == pytest.approx(
            self.PUBLISHED_ANCHORED[issue], abs=self.THREE_DECIMALS
        )

    @pytest.mark.parametrize("topic", sorted(PUBLISHED_DISCOVERED))
    def test_published_discovered_figure_still_reproduces(self, meta, topic):
        assert meta["coherence"][topic]["npmi"] == pytest.approx(
            self.PUBLISHED_DISCOVERED[topic], abs=self.THREE_DECIMALS
        )

    def test_the_caveat_four_are_exactly_the_lowest_scoring_anchored_issues(
        self, meta
    ):
        """§3's table is ORDERED, and its whole argument is that these four sit
        at the bottom "because the taxonomy is doing its job". Pinning only the
        four values would still pass if a fifth issue dropped below them and the
        caveat table silently became incomplete."""
        anchored = {i: meta["coherence"][i]["npmi"] for i in meta["issues"]}

        bottom_four = sorted(anchored, key=anchored.get)[:4]

        assert set(bottom_four) == set(self.PUBLISHED_ANCHORED)
        assert bottom_four[0] == "Immigration"  # the single least coherent

    def test_the_least_coherent_anchored_issue_is_below_the_gate_floor(self, meta):
        """The asymmetry is load-bearing on the real artifact, not just on
        synthetic records: Immigration would FAIL the gate, and is exempt."""
        assert meta["coherence"]["Immigration"]["npmi"] < COHERENCE_FLOOR

    def test_no_anchored_issue_is_classified_on_disk(self, meta):
        assert set(meta["discovered_status"]) == set(
            f"Discovered {i}" for i in range(1, topic_quality.N_FREE_TOPICS + 1)
        )
        assert not set(meta["discovered_status"]) & set(meta["issues"])

    def test_the_only_noise_topic_on_disk_is_discovered_three(self, meta):
        """§4's table: 6 of 7 discovered clusters are coherent. A silent shift
        here would change what the site is allowed to show."""
        noise = {c for c, r in meta["discovered_status"].items()
                 if r["status"] == "noise"}

        assert noise == {"Discovered 3"}

    def test_meta_points_at_the_names_file(self, meta):
        assert meta["names_file"] == topic_quality.NAMES_PATH.name

    def test_names_file_and_issues_meta_carry_identical_scores(self, meta):
        """The two artifacts are written from one scoring run, and the report
        quotes §3 from one and §4 from the other. If only one is ever
        regenerated the report becomes internally inconsistent with no error
        anywhere — so they are compared with `==`, not a tolerance."""
        names = load_names()
        scored = {**names["topics"], **names["anchored_coherence_caveat"]}

        assert set(scored) == set(meta["coherence"])
        assert {k: v["npmi"] for k, v in scored.items()} == {
            k: v["npmi"] for k, v in meta["coherence"].items()
        }
