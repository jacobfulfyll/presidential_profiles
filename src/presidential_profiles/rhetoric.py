"""Per-speech linguistic statistics via spaCy: modal verbs, pronouns, readability.

Replaces the 2019 NLTK word-by-word DataFrame build (which needed 181 batch
files on AWS) with a single streamed spaCy pass over the corpus.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
import textstat

from .corpus import DATA_DIR, load

STATS_PATH = DATA_DIR / "speech_stats.parquet"

MODALS = ["shall", "will", "must", "should", "can", "may", "would", "could"]
FIRST_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves"}


def _nlp():
    # Tagger for fine-grained POS (MD = modal), senter for cheap sentence splits.
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "ner", "lemmatizer", "attribute_ruler"],
    )
    nlp.enable_pipe("senter")
    nlp.max_length = 2_000_000
    return nlp


def build_stats(df: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Compute (or load cached) per-speech linguistic stats."""
    if STATS_PATH.exists() and not force:
        return pd.read_parquet(STATS_PATH)

    if df is None:
        df = load()
    nlp = _nlp()

    rows = []
    texts = df["transcript"].tolist()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=16)):
        modal_counts: Counter[str] = Counter()
        n_tokens = 0
        i_count = 0
        we_count = 0
        for tok in doc:
            if not tok.is_alpha:
                continue
            n_tokens += 1
            low = tok.lower_
            if tok.tag_ == "MD":
                modal_counts[low] += 1
            if low in FIRST_SINGULAR:
                i_count += 1
            elif low in FIRST_PLURAL:
                we_count += 1
        n_sents = sum(1 for _ in doc.sents)
        row = {
            "uuid": df.iloc[i]["uuid"],
            "n_tokens": n_tokens,
            "n_sents": max(n_sents, 1),
            "i_count": i_count,
            "we_count": we_count,
            "fk_grade": textstat.flesch_kincaid_grade(texts[i]),
        }
        for m in MODALS:
            row[f"modal_{m}"] = modal_counts.get(m, 0)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  tagged {i + 1}/{len(texts)} speeches")

    stats = pd.DataFrame(rows)
    stats = df[["uuid", "president", "party", "date", "year", "decade", "title"]].merge(
        stats, on="uuid"
    )
    stats["words_per_sentence"] = stats["n_tokens"] / stats["n_sents"]
    stats.to_parquet(STATS_PATH, index=False)
    return stats
