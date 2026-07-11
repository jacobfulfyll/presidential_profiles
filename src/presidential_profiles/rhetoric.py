"""Per-speech linguistic statistics via spaCy: modal verbs, pronouns, readability.

Replaces the 2019 NLTK word-by-word DataFrame build (which needed 181 batch
files on AWS) with a single streamed spaCy pass over the corpus.
"""

import re
from collections import Counter

import pandas as pd
import spacy

from .corpus import DATA_DIR, load

STATS_PATH = DATA_DIR / "speech_stats.parquet"

MODALS = ["shall", "will", "must", "should", "can", "may", "would", "could"]
FIRST_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves"}

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def syllables(word: str) -> int:
    """Vowel-group syllable estimate (silent trailing 'e' discounted)."""
    w = word.lower()
    n = len(_VOWEL_GROUPS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(n, 1)


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
        n_syllables = 0
        i_count = 0
        we_count = 0
        for tok in doc:
            if not tok.is_alpha:
                continue
            n_tokens += 1
            low = tok.lower_
            n_syllables += syllables(low)
            if tok.tag_ == "MD":
                modal_counts[low] += 1
            if low in FIRST_SINGULAR:
                i_count += 1
            elif low in FIRST_PLURAL:
                we_count += 1
        n_sents = max(sum(1 for _ in doc.sents), 1)
        n_tokens = max(n_tokens, 1)
        fk_grade = 0.39 * (n_tokens / n_sents) + 11.8 * (n_syllables / n_tokens) - 15.59
        row = {
            "doc_name": df.iloc[i]["doc_name"],
            "n_tokens": n_tokens,
            "n_sents": n_sents,
            "i_count": i_count,
            "we_count": we_count,
            "fk_grade": fk_grade,
        }
        for m in MODALS:
            row[f"modal_{m}"] = modal_counts.get(m, 0)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  tagged {i + 1}/{len(texts)} speeches")

    stats = pd.DataFrame(rows)
    stats = df[["doc_name", "president", "party", "date", "year", "decade", "title"]].merge(
        stats, on="doc_name", validate="one_to_one"
    )
    stats["words_per_sentence"] = stats["n_tokens"] / stats["n_sents"]
    stats.to_parquet(STATS_PATH, index=False)
    return stats
