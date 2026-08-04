"""Tests for the paragraph embedding-cluster module (`embed_topics.py`).

The module clusters every paragraph at k=40 and k=15 (model2vec + MiniBatchKMeans,
seed 42) and then *describes* each cluster with c-TF-IDF terms and an NPMI
coherence score. The clustering itself (StaticModel + MiniBatchKMeans over the
36k-row corpus) is heavy and stochastic-only-by-seed; per CLAUDE.md we do NOT
load the real model or embed the real corpus in a unit test. Instead we test the
pure, deterministic description helpers on hand-built co-occurrence matrices with
KNOWN structure, and we exercise the real `build_clusters` assembly with a FAKE
StaticModel (fixed encode vectors) so the keying/schema/write contract runs on
real code without any model download or corpus read.

Coverage map (every public/testable surface has at least one test):
  * `_fit_vectorizer` .. TestFitVectorizer
  * `_c_tf_idf`       .. TestCTfIdf
  * `_npmi`           .. TestNpmi
  * `build_clusters`  .. TestBuildClustersCaching   (cache-read + force/no-cache
                        branches, model-load bomb) and
                        TestBuildClustersAssembly    (real assembly via a fake
                        StaticModel: `cluster_k{k}` naming, `(doc_name, para_idx)`
                        key, `meta["clusters"]` shape, parquet+json write)
  * `label`           .. TestLabel

Deliberate gap: MiniBatchKMeans byte-for-byte reproducibility with the *real*
potion-base-8M embeddings at seed 42 is not asserted — it would require loading
the model and embedding the real corpus, which the test conventions forbid (heavy,
and StaticModel could hit the network). Every other part of the pipeline, including
the assembly/write path, runs on real code with a fake embedder.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from presidential_profiles import embed_topics
from presidential_profiles.embed_topics import (
    N_TERMS,
    _c_tf_idf,
    _fit_vectorizer,
    _npmi,
    build_clusters,
    label,
)


# --------------------------------------------------------------------------- #
# Fixture builders: hand-built count matrices with KNOWN co-occurrence.
# --------------------------------------------------------------------------- #
def _presence_counts(n_docs, term_docs):
    """Build a sparse (n_docs x n_terms) presence matrix from a term->doc-set map.

    `term_docs` is an ordered list of (term, set_of_doc_indices); each listed doc
    gets a count of 1 for that term. `_npmi` only cares about presence (`> 0`), so
    1s are enough to encode any co-occurrence structure exactly.
    Returns (csr counts, terms ndarray) — csr to mirror CountVectorizer's output.
    """
    terms = [t for t, _ in term_docs]
    dense = np.zeros((n_docs, len(terms)), dtype=np.float64)
    for j, (_, docs) in enumerate(term_docs):
        for d in docs:
            dense[d, j] = 1.0
    return csr_matrix(dense), np.array(terms)


def _csr(dense_rows):
    """Sparse count matrix from a dense list-of-rows (mirrors CountVectorizer)."""
    return csr_matrix(np.asarray(dense_rows, dtype=np.float64))


# =========================================================================== #
# _npmi — the highest-value target: pairwise NPMI on known co-occurrence.
# =========================================================================== #
class TestNpmi:
    def test_always_cooccurring_terms_score_plus_one(self):
        """Two terms occupying the exact same document set: NPMI's upper limit,
        +1. (p_joint == p_a == p_b makes pmi == -log(p_joint), so the ratio is 1.)"""
        counts, terms = _presence_counts(
            10, [("alpha", set(range(5))), ("beta", set(range(5)))]
        )

        scores = _npmi(counts, terms, [["alpha", "beta"]])

        assert scores[0] == pytest.approx(1.0)

    def test_pair_present_in_every_paragraph_scores_plus_one(self):
        """Two terms in EVERY paragraph give p_joint == 1, so -log(p_joint) == 0
        and pmi/denom is 0/0. This hits the `denom == 0` guard, which returns the
        perfectly-coherent limit +1 instead of NaN. Unreachable in production
        (MAX_DF < 1 filters such ubiquitous terms out of the vocab) but keeps
        `_npmi` correct in isolation."""
        counts, terms = _presence_counts(
            6, [("ever", set(range(6))), ("present", set(range(6)))]
        )

        scores = _npmi(counts, terms, [["ever", "present"]])

        assert scores[0] == 1.0

    def test_never_cooccurring_terms_score_minus_one(self):
        """Two terms that never share a paragraph collapse to the documented
        limit -1.0 (the `p_joint == 0` branch), not a NaN/undefined."""
        counts, terms = _presence_counts(
            10, [("gamma", set(range(5))), ("delta", set(range(5, 10)))]
        )

        scores = _npmi(counts, terms, [["gamma", "delta"]])

        assert scores[0] == -1.0

    def test_independent_terms_score_near_zero(self):
        """Terms whose joint rate equals the product of their marginals are
        independent: pmi == log(1) == 0, so NPMI is exactly 0. x in {0,1,2,3},
        y in {0,1,4,5} over 8 docs -> p_x=p_y=0.5, p_joint=0.25=0.5*0.5."""
        counts, terms = _presence_counts(
            8, [("x", {0, 1, 2, 3}), ("y", {0, 1, 4, 5})]
        )

        scores = _npmi(counts, terms, [["x", "y"]])

        assert scores[0] == pytest.approx(0.0, abs=1e-12)

    def test_cluster_with_fewer_than_two_in_vocab_terms_is_nan(self):
        """A cluster whose top-terms contribute <2 tokens that are actually in the
        vocabulary cannot form a pair, so NPMI is NaN. Here only one word is in
        vocab (the other is OOV), so cols has length 1."""
        counts, terms = _presence_counts(4, [("a_", {0, 1}), ("b_", {2, 3})])

        scores = _npmi(counts, terms, [["a_", "not_in_vocab"]])

        assert math.isnan(scores[0])

    def test_single_term_cluster_is_nan(self):
        """A degenerate one-term cluster likewise yields NaN (no pair to score)."""
        counts, terms = _presence_counts(4, [("a_", {0, 1}), ("b_", {2, 3})])

        scores = _npmi(counts, terms, [["a_"]])

        assert math.isnan(scores[0])

    def test_mean_is_taken_over_all_pairs(self):
        """NPMI per cluster is the MEAN over all term pairs. A,B share a doc set
        (pair->+1); C is disjoint from both (two pairs->-1 each). Mean = -1/3."""
        counts, terms = _presence_counts(
            6, [("A", {0, 1, 2, 3}), ("B", {0, 1, 2, 3}), ("C", {4, 5})]
        )

        scores = _npmi(counts, terms, [["A", "B", "C"]])

        assert scores[0] == pytest.approx(-1.0 / 3.0)

    def test_returns_one_score_per_cluster_mixing_value_and_nan(self):
        """Output length tracks the number of clusters, and heterogeneous
        clusters (coherent / mutually-exclusive / undersized) each get their own
        score in order."""
        counts, terms = _presence_counts(
            10,
            [
                ("A", set(range(5))),
                ("B", set(range(5))),
                ("C", set(range(5, 10))),
                ("D", set(range(5))),
            ],
        )

        scores = _npmi(
            counts,
            terms,
            [["A", "B"], ["C", "D"], ["A", "only_one_in_vocab"]],
        )

        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)   # identical doc sets
        assert scores[1] == -1.0                 # disjoint doc sets
        assert math.isnan(scores[2])             # <2 in-vocab terms


# =========================================================================== #
# _c_tf_idf — the terms that DISTINGUISH each cluster (class-based TF-IDF).
# =========================================================================== #
class TestCTfIdf:
    def test_distinctive_term_outranks_a_globally_ubiquitous_one(self):
        """The whole point of c-TF-IDF over plain within-cluster frequency: a term
        that is *frequent everywhere* (ubiq) is discounted by idf, so a term that
        is *rarer overall but concentrated in this cluster* (rare) wins even with
        lower raw tf.

        cols = [ubiq, rare]; docs 0,1 -> cluster 0, docs 2,3 -> cluster 1.
          per-class ubiq=[8,8], rare=[6,0]; freq ubiq=16, rare=6.
          cluster-0 score(rare) ~ .446 > score(ubiq) ~ .299 despite ubiq's higher tf.
        """
        counts = _csr([[4, 3], [4, 3], [4, 0], [4, 0]])  # rows: (ubiq, rare)
        terms = np.array(["ubiq", "rare"])
        labels = np.array([0, 0, 1, 1])

        top_terms = _c_tf_idf(counts, labels, terms, k=2)

        # Cluster 0: the distinctive term wins despite lower raw frequency.
        assert top_terms[0][0] == "rare"
        # Cluster 1 lacks "rare" entirely, so its top term is ubiq...
        assert top_terms[1][0] == "ubiq"
        # ...and the distinctive term is NOT the label of the cluster missing it.
        assert "rare" != top_terms[1][0]

    def test_output_is_one_list_per_cluster_ranking_the_full_vocab(self):
        """One term list per cluster (k=2 -> 2 lists). When the vocabulary is
        smaller than N_TERMS, each list ranks the ENTIRE vocab (here both terms),
        not a padded/truncated subset — an exact count, not a `<= N_TERMS` bound
        that a 2-term vocab satisfies vacuously."""
        counts = _csr([[4, 3], [4, 3], [4, 0], [4, 0]])
        terms = np.array(["ubiq", "rare"])
        labels = np.array([0, 0, 1, 1])

        top_terms = _c_tf_idf(counts, labels, terms, k=2)

        assert len(top_terms) == 2
        assert all(len(t) == 2 for t in top_terms)                 # full 2-term vocab
        assert all(set(t) == {"ubiq", "rare"} for t in top_terms)  # both, no dupes

    def test_term_list_is_truncated_to_n_terms(self):
        """With more distinct terms than N_TERMS, each cluster's list is sliced to
        exactly N_TERMS unique terms drawn from the vocabulary."""
        n_terms_vocab = N_TERMS + 2
        # One document (one cluster) with distinct per-term frequencies.
        counts = _csr([list(range(1, n_terms_vocab + 1))])
        terms = np.array([f"t{i}" for i in range(n_terms_vocab)])
        labels = np.array([0])

        top_terms = _c_tf_idf(counts, labels, terms, k=1)

        assert len(top_terms[0]) == N_TERMS
        assert len(set(top_terms[0])) == N_TERMS               # no duplicates
        assert set(top_terms[0]).issubset(set(terms.tolist()))  # all real terms


# =========================================================================== #
# _fit_vectorizer — the shared doc-term counts feeding both scorers.
# =========================================================================== #
class TestFitVectorizer:
    # min_df=20 / max_df=0.5 are tuned for the 36k corpus and would drop every
    # term in a tiny synthetic frame; relax them so the tokenizer/stopword logic
    # (the actual thing under test) is what we observe.
    @pytest.fixture(autouse=True)
    def _relax_df_thresholds(self, monkeypatch):
        monkeypatch.setattr(embed_topics, "MIN_DF", 1)
        monkeypatch.setattr(embed_topics, "MAX_DF", 1.0)

    def test_returns_one_row_per_doc_and_the_exact_surviving_vocab(self):
        """The count matrix has one row per input paragraph and one column per
        surviving term, aligned with the returned `terms`. Rather than only
        asserting the (tautological) shape equality, pin the concrete vocab that
        survives tokenizing + stopword/token-pattern filtering: "the"/"government"
        (stopwords), "a" (1 char) and "42" (digits) drop; tariff/revenue/ox/
        schooner/navy remain."""
        texts = pd.Series(
            [
                "tariff revenue ox the government a 42",
                "tariff schooner ox government",
                "revenue navy schooner",
            ]
        )

        counts, terms = _fit_vectorizer(texts)

        assert counts.shape == (len(texts), len(terms))  # one row per doc
        assert set(terms) == {"tariff", "revenue", "ox", "schooner", "navy"}

    def test_stopwords_are_dropped(self):
        """Both the standard English stop list and the project's EXTRA_STOP are
        excluded ("the" and "government" respectively)."""
        texts = pd.Series(
            [
                "tariff revenue ox the government",
                "tariff schooner ox government",
                "revenue navy schooner the",
            ]
        )

        _, terms = _fit_vectorizer(texts)

        assert "the" not in terms          # ENGLISH_STOP_WORDS
        assert "government" not in terms    # EXTRA_STOP

    def test_token_pattern_excludes_single_chars_and_digits(self):
        """`[a-zA-Z][a-zA-Z]+` keeps only >=2-letter alphabetic tokens: the
        single char "a" and the number "42" are excluded, the 2-letter "ox" kept."""
        texts = pd.Series(
            ["tariff ox a 42", "tariff ox schooner", "revenue ox a"]
        )

        _, terms = _fit_vectorizer(texts)

        assert "ox" in terms   # exactly-two-letter token survives
        assert "a" not in terms
        assert "42" not in terms

    def test_distinctive_token_lands_in_its_own_column(self):
        """A term's column carries its per-document counts, keyed by row."""
        texts = pd.Series(
            [
                "tariff tariff revenue",
                "tariff schooner",
                "revenue navy",
            ]
        )

        counts, terms = _fit_vectorizer(texts)

        col = {t: i for i, t in enumerate(terms)}["tariff"]
        column = np.asarray(counts[:, col].todense()).ravel()
        assert column.tolist() == [2, 1, 0]  # doc0 twice, doc1 once, doc2 none

    def test_preprocess_is_applied_collapsing_viet_nam(self):
        """_fit_vectorizer maps `_preprocess` over texts, so "Viet Nam" is unified
        to a single "vietnam" token rather than split into viet/nam."""
        texts = pd.Series(
            ["Viet Nam policy", "Viet Nam era", "vietnam veterans policy"]
        )

        _, terms = _fit_vectorizer(texts)

        assert "vietnam" in terms
        assert "viet" not in terms
        assert "nam" not in terms


# =========================================================================== #
# build_clusters — caching contract (no model load, no real corpus).
# =========================================================================== #
class _ModelLoaded(Exception):
    """Raised by the StaticModel bomb to prove the embedding path was reached."""


def _bomb_static_model(monkeypatch):
    """Make any StaticModel.from_pretrained call explode. If the cache branch is
    taken correctly, this never fires; if the code tries to embed, it does."""

    class _Bomb:
        @staticmethod
        def from_pretrained(*a, **k):
            raise _ModelLoaded("StaticModel.from_pretrained was called")

    monkeypatch.setattr(embed_topics, "StaticModel", _Bomb)


def _write_fake_cache(tmp_path, monkeypatch):
    """Write a minimal assignments parquet + meta json mirroring the real schema
    and point the module's cache-path constants at them. Returns (df, meta)."""
    clusters_path = tmp_path / "paragraph_clusters.parquet"
    meta_path = tmp_path / "paragraph_clusters_meta.json"
    monkeypatch.setattr(embed_topics, "CLUSTERS_PATH", clusters_path)
    monkeypatch.setattr(embed_topics, "CLUSTERS_META_PATH", meta_path)

    df = pd.DataFrame(
        {
            "doc_name": ["doc-a", "doc-a", "doc-b"],
            "para_idx": [0, 1, 0],
            "cluster_k40": [3, 7, 3],
            "cluster_k15": [1, 2, 1],
        }
    )
    meta = {
        "model": "fake-model",
        "n_paragraphs": 3,
        "random_state": 42,
        "clusters": {
            "k40": [{"cluster": 3, "size": 2, "terms": ["tariff"], "npmi": 0.1}],
            "k15": [{"cluster": 1, "size": 2, "terms": ["treaty"], "npmi": 0.2}],
        },
    }
    df.to_parquet(clusters_path, index=False)
    import json

    meta_path.write_text(json.dumps(meta))
    return df, meta


class TestBuildClustersCaching:
    def test_returns_cache_without_loading_model(self, tmp_path, monkeypatch):
        """When both cache files exist and force is False, build_clusters returns
        them verbatim and never touches the model (the bomb would raise if it did)."""
        df, meta = _write_fake_cache(tmp_path, monkeypatch)
        _bomb_static_model(monkeypatch)

        got_df, got_meta = build_clusters(force=False)

        pd.testing.assert_frame_equal(got_df, df)
        assert got_meta == meta

    def test_cached_assignments_key_is_unique_and_non_null(self, tmp_path, monkeypatch):
        """The (doc_name, para_idx) key the callers merge on round-trips through
        the cache intact: exact column set, no nulls, no duplicate keys."""
        _write_fake_cache(tmp_path, monkeypatch)
        _bomb_static_model(monkeypatch)

        got_df, _ = build_clusters(force=False)

        assert list(got_df.columns) == [
            "doc_name",
            "para_idx",
            "cluster_k40",
            "cluster_k15",
        ]
        assert not got_df[["doc_name", "para_idx"]].isnull().any().any()
        assert not got_df.duplicated(["doc_name", "para_idx"]).any()

    def test_force_bypasses_cache_and_reaches_the_model(self, tmp_path, monkeypatch):
        """force=True ignores a present cache and proceeds to embedding; the model
        bomb firing proves the cache short-circuit was skipped. PARAGRAPHS_PATH is
        redirected to a tiny frame so no real 36k corpus is read."""
        _write_fake_cache(tmp_path, monkeypatch)
        tiny = pd.DataFrame(
            {"doc_name": ["d"], "para_idx": [0], "text": ["tariff revenue"]}
        )
        paras_path = tmp_path / "paragraphs.parquet"
        tiny.to_parquet(paras_path, index=False)
        monkeypatch.setattr(embed_topics, "PARAGRAPHS_PATH", paras_path)
        _bomb_static_model(monkeypatch)

        with pytest.raises(_ModelLoaded):
            build_clusters(force=True)

    def test_missing_one_cache_file_falls_through_to_the_model(
        self, tmp_path, monkeypatch
    ):
        """The cache branch requires BOTH files. With only the meta present, the
        conjunction is False and the code falls through to embedding (bomb fires)."""
        _, _ = _write_fake_cache(tmp_path, monkeypatch)
        embed_topics.CLUSTERS_PATH.unlink()  # remove the parquet, keep the json
        tiny = pd.DataFrame(
            {"doc_name": ["d"], "para_idx": [0], "text": ["tariff revenue"]}
        )
        paras_path = tmp_path / "paragraphs.parquet"
        tiny.to_parquet(paras_path, index=False)
        monkeypatch.setattr(embed_topics, "PARAGRAPHS_PATH", paras_path)
        _bomb_static_model(monkeypatch)

        with pytest.raises(_ModelLoaded):
            build_clusters(force=False)


# =========================================================================== #
# build_clusters — the REAL assembly + write path, driven fully offline.
#
# TestBuildClustersCaching pins the *cache-read* branch; the schema/keying
# contract of the *written* artifact was only proxied by a hand-authored cache.
# This class exercises the actual assembly code (lines 169-199): the model call,
# the per-k MiniBatchKMeans loop, the `cluster_k{k}` column naming, the meta
# dict, and the parquet/json write — with a fake StaticModel and a tiny synthetic
# corpus, so no potion-base-8M download and no 36k-row read. K_VALUES is patched
# to (2, 3): two values prove the loop runs per-k and that column/meta names are
# built from `k` rather than hardcoded.
# =========================================================================== #
class _FakeStaticModel:
    """Drop-in for model2vec.StaticModel: no download, deterministic encode.

    `.encode` returns three well-separated base directions cycled over the input
    (plus a tiny per-row jitter so no two rows are identical), giving k-means a
    clean 3-group structure to partition at both k=2 and k=3.
    """

    DIM = 8

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def encode(self, texts):
        # build_clusters passes paras["text"].tolist() — assert the real contract.
        assert isinstance(texts, list)
        n = len(texts)
        base = np.eye(3, self.DIM)  # 3 orthogonal unit directions
        rows = np.stack([base[i % 3] for i in range(n)]).astype(np.float64)
        rows += np.arange(n)[:, None] * 1e-3
        return rows


class TestBuildClustersAssembly:
    K = (2, 3)

    @pytest.fixture
    def built(self, tmp_path, monkeypatch):
        """Run the real build_clusters(force=True) against a fake model + tiny
        corpus, writing to tmp. Returns (assignments, meta, paras, paths)."""
        # Cluster at small ks so both the per-k loop and cluster_k{k} naming show.
        monkeypatch.setattr(embed_topics, "K_VALUES", self.K)
        # The real thresholds (min_df=20/max_df=0.5) would drop every term in a
        # tiny frame; relax so the vectorizer yields a real >=2-term vocabulary.
        monkeypatch.setattr(embed_topics, "MIN_DF", 1)
        monkeypatch.setattr(embed_topics, "MAX_DF", 1.0)
        # No download, no network: the model is a deterministic stand-in.
        monkeypatch.setattr(embed_topics, "StaticModel", _FakeStaticModel)

        # 6 paragraphs, distinct (doc_name, para_idx) keys, domain vocabulary with
        # no stopwords/EXTRA_STOP words so the vectorizer keeps >=2 terms.
        paras = pd.DataFrame(
            {
                "doc_name": ["doc-a", "doc-a", "doc-b", "doc-b", "doc-c", "doc-c"],
                "para_idx": [0, 1, 0, 1, 0, 1],
                "text": [
                    "tariff revenue trade duty",
                    "tariff schooner harbor duty",
                    "navy fleet warship harbor",
                    "navy fleet warship squadron",
                    "treaty senate ratify envoy",
                    "treaty senate ratify diplomacy",
                ],
            }
        )
        paras_path = tmp_path / "paragraphs.parquet"
        paras.to_parquet(paras_path, index=False)
        monkeypatch.setattr(embed_topics, "PARAGRAPHS_PATH", paras_path)

        clusters_path = tmp_path / "paragraph_clusters.parquet"
        meta_path = tmp_path / "paragraph_clusters_meta.json"
        monkeypatch.setattr(embed_topics, "CLUSTERS_PATH", clusters_path)
        monkeypatch.setattr(embed_topics, "CLUSTERS_META_PATH", meta_path)

        assignments, meta = build_clusters(force=True)
        return assignments, meta, paras, (clusters_path, meta_path)

    # ---- assignments schema + keying (the sole in-scope acceptance criterion) ----
    def test_assignments_columns_are_key_plus_one_cluster_col_per_k(self, built):
        """Exactly the (doc_name, para_idx) key followed by one correctly-named
        cluster_k{k} column per k in K_VALUES — this is what catches a column
        named cluster_40 instead of cluster_k40, or a dropped/extra k."""
        assignments, _, _, _ = built

        assert list(assignments.columns) == [
            "doc_name",
            "para_idx",
            "cluster_k2",
            "cluster_k3",
        ]

    def test_assignments_key_matches_input_paragraphs_exactly(self, built):
        """The persisted key is the paragraphs' own (doc_name, para_idx) — same
        rows, no reorder-driven fabrication — so callers can merge one-to-one."""
        assignments, _, paras, _ = built

        got = assignments[["doc_name", "para_idx"]].sort_values(
            ["doc_name", "para_idx"]
        ).reset_index(drop=True)
        want = paras[["doc_name", "para_idx"]].sort_values(
            ["doc_name", "para_idx"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want)

    def test_assignments_key_is_non_null_and_unique(self, built):
        """The merge key has no nulls and no duplicate (doc_name, para_idx) pairs
        — the validate="one_to_one" contract stated in build_clusters' docstring."""
        assignments, _, _, _ = built

        key = assignments[["doc_name", "para_idx"]]
        assert not key.isnull().any().any()
        assert not assignments.duplicated(["doc_name", "para_idx"]).any()

    def test_each_cluster_column_holds_labels_in_range_k(self, built):
        """Every cluster label is a valid cluster id for its k: cluster_k2 in
        {0,1}, cluster_k3 in {0,1,2}. Guards against off-by-one / leaked ids."""
        assignments, _, _, _ = built

        for k in self.K:
            labels = assignments[f"cluster_k{k}"]
            assert set(labels.unique()).issubset(set(range(k)))

    # ---- meta structure ----
    def test_meta_carries_model_paragraph_count_and_seed(self, built):
        """Top-level provenance: the model name, the paragraph count (matching the
        input frame), and the fixed random_state that makes the run reproducible."""
        _, meta, paras, _ = built

        assert meta["model"] == embed_topics.MODEL_NAME
        assert meta["n_paragraphs"] == len(paras)
        assert meta["random_state"] == embed_topics.RANDOM_STATE

    def test_meta_has_one_cluster_block_of_length_k_per_k(self, built):
        """meta["clusters"] is keyed k{k} with a list of exactly k cluster records
        — the other half of the per-k loop producing k-derived names."""
        _, meta, _, _ = built

        assert set(meta["clusters"]) == {"k2", "k3"}
        for k in self.K:
            assert len(meta["clusters"][f"k{k}"]) == k

    def test_meta_cluster_records_are_fully_populated(self, built):
        """Each cluster record carries cluster id (in range k), size, term list,
        and an npmi entry; sizes sum to the paragraph count (a partition)."""
        _, meta, paras, _ = built

        for k in self.K:
            blocks = meta["clusters"][f"k{k}"]
            total = 0
            for c, block in enumerate(blocks):
                assert set(block) == {"cluster", "size", "terms", "npmi"}
                assert block["cluster"] == c
                assert block["cluster"] in range(k)
                assert isinstance(block["terms"], list)
                total += block["size"]
            assert total == len(paras)  # every paragraph assigned exactly once

    # ---- the write actually happened and round-trips ----
    def test_written_parquet_and_json_reload_equal_to_returned(self, built):
        """build_clusters WROTE the artifacts to the configured paths, and reading
        them back yields exactly what it returned — the persistence half of the
        acceptance criterion, not just the in-memory assembly."""
        import json

        assignments, meta, _, (clusters_path, meta_path) = built

        assert clusters_path.exists() and meta_path.exists()
        reloaded = pd.read_parquet(clusters_path)
        pd.testing.assert_frame_equal(reloaded, assignments)
        assert json.loads(meta_path.read_text()) == meta


# =========================================================================== #
# label — the short human-readable cluster label.
# =========================================================================== #
class TestLabel:
    META = {
        "clusters": {
            "k40": [
                {"cluster": 0, "terms": ["tariff", "revenue", "trade", "duty"]},
                {"cluster": 1, "terms": ["navy", "fleet", "ship"]},
            ]
        }
    }

    def test_joins_top_three_terms_by_default(self):
        assert label(self.META, 40, 0) == "tariff / revenue / trade"

    def test_respects_n(self):
        assert label(self.META, 40, 0, n=2) == "tariff / revenue"
        assert label(self.META, 40, 0, n=1) == "tariff"

    def test_selects_the_requested_cluster(self):
        assert label(self.META, 40, 1) == "navy / fleet / ship"

    def test_n_larger_than_available_returns_all_terms(self):
        """Slicing past the end is a no-op, not an error: every term is joined."""
        assert label(self.META, 40, 1, n=10) == "navy / fleet / ship"


# =========================================================================== #
# Determinism of the description helpers (the reproducibility the module claims,
# at the level we can test without loading the model).
# =========================================================================== #
class TestDeterminism:
    def test_helpers_are_pure_and_reproducible(self):
        """Given identical inputs, _c_tf_idf and _npmi return identical outputs on
        repeated calls — the deterministic half of the module's reproducibility
        guarantee (the stochastic half, seeded MiniBatchKMeans, needs the model)."""
        counts = _csr([[4, 3, 0], [4, 3, 0], [0, 0, 5], [0, 0, 5]])
        terms = np.array(["tariff", "revenue", "navy"])
        labels = np.array([0, 0, 1, 1])

        first_terms = _c_tf_idf(counts, labels, terms, k=2)
        second_terms = _c_tf_idf(counts, labels, terms, k=2)
        assert first_terms == second_terms

        first_npmi = _npmi(counts, terms, first_terms)
        second_npmi = _npmi(counts, terms, second_terms)
        # NaN != NaN, so compare with the NaN-aware helper.
        np.testing.assert_array_equal(first_npmi, second_npmi)
