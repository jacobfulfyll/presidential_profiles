"""Keyword usage trends and era-distinctive vocabulary."""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .corpus import load

WORD_RE = re.compile(r"[a-z']+")

# Term groups tracked per 10k words. Values are the surface forms counted.
TERM_GROUPS = {
    "democracy": ["democracy", "democratic"],
    "freedom / liberty": ["freedom", "freedoms", "liberty", "liberties"],
    "God": ["god"],
    "economy / jobs": ["economy", "economic", "jobs", "unemployment"],
    "war": ["war", "wars"],
    "immigration": ["immigration", "immigrant", "immigrants"],
    "tariff / trade": ["tariff", "tariffs", "trade"],
    "constitution": ["constitution", "constitutional"],
    "border": ["border", "borders"],
}


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def keyword_trends(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rate per 10k words for each term group, by decade."""
    if df is None:
        df = load()
    rows = []
    for decade, group in df.groupby("decade"):
        counts: Counter[str] = Counter()
        total = 0
        for text in group["transcript"]:
            toks = tokens(text)
            total += len(toks)
            counts.update(toks)
        for name, terms in TERM_GROUPS.items():
            rate = sum(counts[t] for t in terms) / total * 10_000
            rows.append({"decade": decade, "term": name, "rate": rate})
    return pd.DataFrame(rows)


def distinctive_terms(
    df: pd.DataFrame | None = None,
    split_date: str = "2019-04-18",
    baseline_start: str = "1989-01-01",
    top_n: int = 15,
) -> pd.DataFrame:
    """Log-odds (informative Dirichlet prior, Monroe et al. 2008) comparing
    vocabulary after `split_date` (the new-era speeches added since the 2019
    capstone) with the modern-era baseline before it."""
    if df is None:
        df = load()
    modern = df[df["date"] >= baseline_start]
    new = modern[modern["date"] > split_date]
    old = modern[modern["date"] <= split_date]

    def word_counts(frame: pd.DataFrame) -> Counter:
        c: Counter[str] = Counter()
        for text in frame["transcript"]:
            c.update(t for t in tokens(text) if len(t) > 2)
        return c

    c_new, c_old = word_counts(new), word_counts(old)
    prior = c_new + c_old
    # Content words only: drop stop words and contraction fragments so the
    # chart reads as topical change rather than register change.
    vocab = [
        w
        for w, n in prior.items()
        if n >= 20 and w not in ENGLISH_STOP_WORDS and "'" not in w
    ]

    n_new, n_old, n_prior = sum(c_new.values()), sum(c_old.values()), sum(prior.values())
    alpha0 = 500.0  # prior strength
    rows = []
    for w in vocab:
        a_w = alpha0 * prior[w] / n_prior
        l_new = (c_new[w] + a_w) / (n_new + alpha0 - c_new[w] - a_w)
        l_old = (c_old[w] + a_w) / (n_old + alpha0 - c_old[w] - a_w)
        delta = np.log(l_new) - np.log(l_old)
        var = 1.0 / (c_new[w] + a_w) + 1.0 / (c_old[w] + a_w)
        rows.append({"term": w, "z": delta / np.sqrt(var), "count": prior[w]})

    scores = pd.DataFrame(rows).sort_values("z")
    top_old = scores.head(top_n).assign(era="1989 - Apr 2019")
    top_new = scores.tail(top_n).assign(era="Apr 2019 - 2026")
    return pd.concat([top_old, top_new]).reset_index(drop=True)
