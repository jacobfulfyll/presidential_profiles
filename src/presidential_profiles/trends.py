"""Keyword usage trends and era-distinctive vocabulary."""

import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import word_families as wf
from .corpus import load

WORD_RE = re.compile(r"[a-z']+")

# Seed forms for each tracked concept. These are the editorial groupings (e.g.
# freedom + liberty are synonyms no morphological map would join); the actual
# surface forms counted are these seeds expanded through the word-family map, so
# "immigration" automatically also counts "immigrant"/"immigrants". Seeds keep
# their own literal forms too, so the concept never regresses if the map is
# missing or splits a word.
TERM_SEEDS = {
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


@lru_cache(maxsize=1)
def term_groups() -> dict[str, list[str]]:
    """Each concept's seeds expanded through the word-family map — the surface
    forms actually summed. Lazy (not an import-time dict) so a fresh clone
    without data/word_families.json still imports; unmapped seeds fall back to
    themselves."""
    return {
        name: sorted({f for seed in seeds for f in wf.family_members(seed)})
        for name, seeds in TERM_SEEDS.items()
    }


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def keyword_trends(
    df: pd.DataFrame | None = None, bucket: int = 5, min_words: int = 20_000
) -> pd.DataFrame:
    """Rate per 10k words for each term group, in `bucket`-year periods.
    Periods with fewer than min_words are dropped (the earliest corpus years
    are a handful of speeches)."""
    if df is None:
        df = load()
    df = df.assign(period=(df["year"] // bucket) * bucket)
    rows = []
    for period, group in df.groupby("period"):
        counts: Counter[str] = Counter()
        total = 0
        for text in group["transcript"]:
            toks = tokens(text)
            total += len(toks)
            counts.update(toks)
        if total < min_words:
            continue
        for name, terms in term_groups().items():
            rate = sum(counts[t] for t in terms) / total * 10_000
            rows.append({"period": period, "term": name, "rate": rate})
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


def grouped_word_counts(texts) -> Counter:
    """Content-word counts collapsed through Explore's word-family map.

    This deliberately shares both spelling normalization and family labels
    with the grouped Word Explorer. One occurrence still contributes one
    count; inflected forms such as ``slavery``/``slave``/``slaves`` simply
    accumulate under the same family label.
    """
    c: Counter[str] = Counter()
    for text in texts:
        c.update(
            wf.form_to_node(t)
            for t in wf.tokenize(text)
            if len(t) > 2 and t not in ENGLISH_STOP_WORDS and "'" not in t
        )
    return c


def grouped_word_set(text: str) -> set[str]:
    """Distinct grouped content-word labels present in one speech."""
    return {
        wf.form_to_node(t)
        for t in wf.tokenize(text)
        if len(t) > 2 and t not in ENGLISH_STOP_WORDS and "'" not in t
    }


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
        rows.append({
            "term": w,
            "delta": delta,
            "z": delta / np.sqrt(var),
            "count": prior[w],
        })
    return pd.DataFrame(rows).sort_values("z")


# Spoken-register words: interesting as a register finding, but they crowd
# topical change out of the era-distinctive chart.
REGISTER_WORDS = {
    "going", "said", "really", "yeah", "know", "want", "lot", "just", "got",
    "thing", "things", "kind", "actually", "okay", "gonna", "tell", "told",
    "saying", "talk", "talking", "look", "looking", "way", "right", "think",
    "thank", "great", "president", "didn", "don", "doesn", "isn", "wasn",
    "like", "sir", "did", "doing", "does", "oh", "hey", "guys", "let",
    # corpus-structural noise: split "united states", dates read aloud
    "states", "united", "january", "february", "march", "april", "june",
    "july", "september", "october", "november", "december",
}

# Named eras for the all-history vocabulary comparison.
ERAS = [
    ("The founding", 1789, 1815),
    ("Expansion", 1816, 1849),
    ("Civil War & Reconstruction", 1850, 1877),
    ("The Gilded Age", 1878, 1900),
    ("Progressives & Depression", 1901, 1932),
    ("War & New Deal", 1933, 1945),
    ("The Cold War", 1946, 1988),
    ("Post-Cold War", 1989, 2016),
    ("The present era", 2017, 2026),
]


# Synonym families that split one topic across several tokens ("covid",
# "coronavirus") and hide it from the rankings.
CANON = {"coronavirus": "covid", "vaccines": "vaccine"}


def era_vocabulary(df: pd.DataFrame | None = None, top_n: int = 10) -> list[dict]:
    """Each era's statistically distinctive vocabulary vs all other eras."""
    if df is None:
        df = load()
    counts = {}
    for label, lo, hi in ERAS:
        c = word_counts(df[df["year"].between(lo, hi)]["transcript"])
        for src, dst in CANON.items():
            if src in c:
                c[dst] += c.pop(src)
        counts[label] = c
    total = sum(counts.values(), start=Counter())
    out = []
    for label, lo, hi in ERAS:
        scores = log_odds_scores(counts[label], total - counts[label])
        scores = scores[~scores["term"].isin(REGISTER_WORDS)]
        out.append({"era": label, "years": f"{lo}-{hi}",
                    "words": scores.tail(top_n)["term"].tolist()[::-1]})
    return out


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
    scores = scores[~scores["term"].isin(REGISTER_WORDS)]
    top_old = scores.head(top_n).assign(era="1989 - Apr 2019")
    top_new = scores.tail(top_n).assign(era="Apr 2019 - 2026")
    return pd.concat([top_old, top_new]).reset_index(drop=True)
