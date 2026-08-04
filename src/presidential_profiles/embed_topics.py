"""Embedding clusters over paragraphs: model2vec + MiniBatchKMeans.

The anchored CorEx taxonomy in `issues.py` is a *policy* taxonomy: it can only
find what its anchor lists already name, and 28.7% of paragraphs (10,381 of
36,229) match none of the 15 anchored issues. This module is the bias-free
counterweight — it lets the corpus partition itself, with no supplied vocabulary
at all, so that categories the curated taxonomy never anticipated (Indian
affairs, coinage, naval shipbuilding, terrorism) can surface on their own.

Its output is an input to taxonomy discovery, not a taxonomy: cluster
assignments keyed `(doc_name, para_idx)` plus, per cluster, the c-TF-IDF terms
that distinguish it and an NPMI coherence score.

Two values of k, both persisted:
  * k=40 — discovery. Fine enough to separate period-bound concerns from the
    broad domains that would otherwise swallow them.
  * k=15 — dimension-matched to the 15 anchored issues, so any later dispersion
    or concentration measure compares like with like. At k=40 a paragraph's
    composition vector is mostly smoothing prior, which distorts those measures.

COHERENCE CAVEAT — READ BEFORE REUSING `npmi`:
    NPMI is a valid quality signal for DISCOVERED topics only (these clusters,
    CorEx's free "Discovered N" topics). It is NOT valid for the anchored
    issues. Immigration scores -0.028 and Foreign policy 0.079 not because they
    are bad issues but because their anchor sets deliberately span vocabulary
    that never co-occurs in one paragraph ("railroad" and "broadband" are one
    issue — Infrastructure — by design, yet share no context). Never apply a
    coherence threshold to an anchored issue — you would delete the taxonomy's
    whole point.
"""

import json

import numpy as np
import pandas as pd
from model2vec import StaticModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .corpus import DATA_DIR
from .fetch import PARAGRAPHS_PATH
from .similarity import MODEL_NAME, _normalize
from .topics import EXTRA_STOP, _preprocess

CLUSTERS_PATH = DATA_DIR / "paragraph_clusters.parquet"
CLUSTERS_META_PATH = DATA_DIR / "paragraph_clusters_meta.json"

# Discovery k and dimension-matched k. Both are persisted; see module docstring.
K_VALUES = (40, 15)

# Seed 42 matches every other model in the repo (topics.py, issues.py,
# similarity.py). MiniBatchKMeans is fully determined by it, so re-running
# reproduces assignments byte for byte.
RANDOM_STATE = 42

# Terms per cluster: enough to read the cluster, few enough that NPMI is not
# dominated by the tail.
N_TERMS = 10

# Vocabulary for c-TF-IDF and for the NPMI reference counts. min_df=20 keeps
# NPMI off hapax pairs whose probabilities are pure noise; max_df drops terms so
# ubiquitous they distinguish nothing. min_df and max_features follow issues.py;
# the tokenizer and EXTRA_STOP come from topics.py.
MIN_DF = 20
MAX_DF = 0.5
MAX_FEATURES = 25_000


def _fit_vectorizer(texts: pd.Series):
    """Doc-term counts over paragraphs, shared by c-TF-IDF and NPMI.

    Both need the same vocabulary — a term c-TF-IDF surfaces must be scoreable
    by NPMI — so they are counted once here rather than twice with drifting
    settings.
    """
    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP),
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        token_pattern=r"[a-zA-Z][a-zA-Z]+",
    )
    counts = vectorizer.fit_transform(texts.map(_preprocess))
    return counts, np.asarray(vectorizer.get_feature_names_out())


def _c_tf_idf(counts, labels: np.ndarray, terms: np.ndarray, k: int) -> list[list[str]]:
    """Class-based TF-IDF: the terms that distinguish each cluster.

    Every cluster is treated as one long document. tf is the term's share of its
    cluster's words; idf is log(1 + avg_words_per_cluster / total_frequency), so
    a term common to all clusters is discounted even if it is frequent inside
    one. Plain per-paragraph TF-IDF would instead rank each cluster by whatever
    is merely frequent in it, which is the same list for every cluster.
    """
    # (k, vocab_size) — one row of term counts per cluster.
    per_class = np.vstack(
        [np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in range(k)]
    )
    words_per_class = per_class.sum(axis=1, keepdims=True)
    tf = per_class / np.maximum(words_per_class, 1)
    freq = per_class.sum(axis=0)
    idf = np.log(1.0 + words_per_class.mean() / np.maximum(freq, 1))
    scores = tf * idf
    return [terms[np.argsort(scores[c])[::-1][:N_TERMS]].tolist() for c in range(k)]


def _npmi(counts, terms: np.ndarray, top_terms: list[list[str]]) -> list[float]:
    """Mean pairwise NPMI of each cluster's top terms.

    Reference corpus is the paragraph set itself and the co-occurrence window is
    the paragraph — the same unit we cluster, so coherence measures the thing we
    actually built. NPMI = PMI / -log P(a,b), bounded to [-1, 1]: +1 = the terms
    only ever appear together, 0 = independent, -1 = mutually exclusive.

    Valid for these discovered clusters ONLY — see the module docstring's
    coherence caveat before applying this to the anchored issues.
    """
    binary = (counts > 0).astype(np.float64)
    n_docs = binary.shape[0]
    index = {t: i for i, t in enumerate(terms)}
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()

    out = []
    for words in top_terms:
        cols = [index[w] for w in words if w in index]
        if len(cols) < 2:
            out.append(float("nan"))
            continue
        sub = binary[:, cols]
        # (m, m) co-occurrence counts among this cluster's top terms.
        co = np.asarray((sub.T @ sub).todense())
        p_single = doc_freq[cols] / n_docs

        scores = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                p_joint = co[i, j] / n_docs
                if p_joint == 0:
                    # Never co-occur: NPMI's limit is -1, not undefined.
                    scores.append(-1.0)
                    continue
                denom = -np.log(p_joint)
                if denom == 0:
                    # p_joint == 1: the pair appears in every paragraph, the
                    # perfectly-coherent extreme, so NPMI's limit is +1 — not
                    # the 0/0 that pmi/denom would compute. Unreachable while
                    # MAX_DF < 1 filters such ubiquitous terms out of the vocab,
                    # but keep _npmi correct without leaning on that constant.
                    scores.append(1.0)
                    continue
                pmi = np.log(p_joint / (p_single[i] * p_single[j]))
                scores.append(float(pmi / denom))
        out.append(float(np.mean(scores)))
    return out


def build_clusters(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Cluster every paragraph at each k in K_VALUES.

    Returns (assignments, meta). `assignments` is keyed `(doc_name, para_idx)`
    with one `cluster_k{k}` column per k — the real key, so callers can merge it
    back against paragraphs.parquet with validate="one_to_one" rather than
    trusting row order (see CLAUDE.md). `meta` carries, per k and cluster, its
    size, top c-TF-IDF terms, and NPMI.
    """
    if CLUSTERS_PATH.exists() and CLUSTERS_META_PATH.exists() and not force:
        return (
            pd.read_parquet(CLUSTERS_PATH),
            json.loads(CLUSTERS_META_PATH.read_text()),
        )

    paras = pd.read_parquet(PARAGRAPHS_PATH)

    # Static embeddings: no torch, no network (the model is cached), ~3s for the
    # full corpus. Normalized so k-means' Euclidean objective ranks neighbours
    # the same way cosine does, which is how this model is meant to be compared.
    model = StaticModel.from_pretrained(MODEL_NAME)
    vectors = _normalize(np.asarray(model.encode(paras["text"].tolist())))

    counts, terms = _fit_vectorizer(paras["text"])

    assignments = paras[["doc_name", "para_idx"]].copy()
    meta: dict = {"model": MODEL_NAME, "n_paragraphs": len(paras),
                  "random_state": RANDOM_STATE, "clusters": {}}

    for k in K_VALUES:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=1024
        )
        labels = km.fit_predict(vectors)
        assignments[f"cluster_k{k}"] = labels

        top_terms = _c_tf_idf(counts, labels, terms, k)
        coherence = _npmi(counts, terms, top_terms)
        meta["clusters"][f"k{k}"] = [
            {
                "cluster": c,
                "size": int((labels == c).sum()),
                "terms": top_terms[c],
                "npmi": coherence[c],
            }
            for c in range(k)
        ]

    assignments.to_parquet(CLUSTERS_PATH, index=False)
    CLUSTERS_META_PATH.write_text(json.dumps(meta, indent=2))
    return assignments, meta


def label(meta: dict, k: int, cluster: int, n: int = 3) -> str:
    """Short human-readable label: the cluster's top-n c-TF-IDF terms."""
    return " / ".join(meta["clusters"][f"k{k}"][cluster]["terms"][:n])
