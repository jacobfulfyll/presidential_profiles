"""Topic modeling over the full corpus: TF-IDF + NMF.

The 2019 capstone ran NMF/LDA in notebooks; this is the same family of model
with reproducible seeding, run over the refreshed 1789-2026 corpus.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from .corpus import DATA_DIR, load

DOC_TOPICS_PATH = DATA_DIR / "doc_topics.parquet"
TOPIC_TERMS_PATH = DATA_DIR / "topic_terms.json"

N_TOPICS = 12

# Transcript noise and empty rhetoric markers, on top of standard stop words.
EXTRA_STOP = {
    "applause", "laughter", "inaudible", "crosstalk", "audience",
    "ladies", "gentlemen", "thank", "thanks", "tonight", "today",
    "mr", "mrs", "dont", "didnt", "im", "ive", "weve", "youre",
    "going", "want", "know", "say", "said", "just", "let", "make",
    "america", "american", "americans", "united", "states", "people",
    "country", "nation", "president", "congress", "government", "year", "years",
}


def build_topics(df: pd.DataFrame | None = None, force: bool = False):
    """Fit NMF topics; return (doc_topics DataFrame, topic term lists)."""
    if DOC_TOPICS_PATH.exists() and TOPIC_TERMS_PATH.exists() and not force:
        return (
            pd.read_parquet(DOC_TOPICS_PATH),
            json.loads(TOPIC_TERMS_PATH.read_text()),
        )

    if df is None:
        df = load()

    stop_words = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_df=0.6,
        min_df=5,
        max_features=30_000,
        token_pattern=r"[a-zA-Z][a-zA-Z]+",
    )
    tfidf = vectorizer.fit_transform(df["transcript"].str.lower())
    terms = np.array(vectorizer.get_feature_names_out())

    nmf = NMF(n_components=N_TOPICS, init="nndsvda", max_iter=400, random_state=42)
    weights = nmf.fit_transform(tfidf)

    topic_terms = {
        f"topic_{k}": terms[np.argsort(nmf.components_[k])[::-1][:12]].tolist()
        for k in range(N_TOPICS)
    }

    # Row-normalize so each speech's topic weights sum to 1.
    shares = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    doc_topics = pd.DataFrame(
        shares, columns=[f"topic_{k}" for k in range(N_TOPICS)]
    )
    doc_topics.insert(0, "uuid", df["uuid"].values)
    doc_topics = df[["uuid", "president", "party", "year", "decade"]].merge(
        doc_topics, on="uuid"
    )

    doc_topics.to_parquet(DOC_TOPICS_PATH, index=False)
    TOPIC_TERMS_PATH.write_text(json.dumps(topic_terms, indent=2))
    return doc_topics, topic_terms


def label(topic_terms: dict, key: str, n: int = 3) -> str:
    """Short human-readable label: the topic's top-n terms."""
    return " / ".join(topic_terms[key][:n])
