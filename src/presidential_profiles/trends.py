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


def word_counts(texts) -> Counter:
    """Content-word counts: stop words and contraction fragments dropped so
    downstream comparisons read as topical rather than register change."""
    c: Counter[str] = Counter()
    for text in texts:
        c.update(
            t for t in tokens(text)
            if len(t) > 2 and t not in ENGLISH_STOP_WORDS and "'" not in t
        )
    return c


def log_odds_scores(
    c_a: Counter, c_b: Counter, min_count: int = 20, alpha0: float = 500.0
) -> pd.DataFrame:
    """Log-odds with informative Dirichlet prior (Monroe et al. 2008).
    Positive z = distinctive of side A."""
    prior = c_a + c_b
    vocab = [w for w, n in prior.items() if n >= min_count]
    n_a, n_b, n_prior = sum(c_a.values()), sum(c_b.values()), sum(prior.values())
    rows = []
    for w in vocab:
        a_w = alpha0 * prior[w] / n_prior
        l_a = (c_a[w] + a_w) / (n_a + alpha0 - c_a[w] - a_w)
        l_b = (c_b[w] + a_w) / (n_b + alpha0 - c_b[w] - a_w)
        delta = np.log(l_a) - np.log(l_b)
        var = 1.0 / (c_a[w] + a_w) + 1.0 / (c_b[w] + a_w)
        rows.append({"term": w, "z": delta / np.sqrt(var), "count": prior[w]})
    return pd.DataFrame(rows).sort_values("z")


def distinctive_terms(
    df: pd.DataFrame | None = None,
    split_date: str = "2019-04-18",
    baseline_start: str = "1989-01-01",
    top_n: int = 15,
) -> pd.DataFrame:
    """Vocabulary distinctive of the era after `split_date` (the speeches
    added since the 2019 capstone) vs the modern-era baseline before it."""
    if df is None:
        df = load()
    modern = df[df["date"] >= baseline_start]
    new = modern[modern["date"] > split_date]
    old = modern[modern["date"] <= split_date]

    scores = log_odds_scores(word_counts(new["transcript"]),
                             word_counts(old["transcript"]))
    top_old = scores.head(top_n).assign(era="1989 - Apr 2019")
    top_new = scores.tail(top_n).assign(era="Apr 2019 - 2026")
    return pd.concat([top_old, top_new]).reset_index(drop=True)
