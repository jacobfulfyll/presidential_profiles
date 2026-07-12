"""Rhetorical indices per speech and per president.

Certainty vs deliberation (the confidence-sells hypothesis), naming of the
country, nostalgia vs future orientation, civil religion, us-vs-them,
superlatives, NRC emotion scores (hope / fear), and windowed vocabulary
richness. Everything is a rate per 10,000 words unless noted.
"""

import io
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from .corpus import DATA_DIR, load

MARKERS_PATH = DATA_DIR / "speech_markers.parquet"
NRC_URL = "http://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
NRC_CACHE = DATA_DIR / "raw" / "nrc_emolex.txt"

WORD_RE = re.compile(r"[a-z']+")

MARKERS = {
    "boosters": r"\bnever\b|\balways\b|\bcertainly\b|\babsolutely\b|\bdefinitely\b"
                r"|\btremendous\w*\b|\bincredibl\w+\b|\btotally\b|\bcompletely\b",
    "hedges": r"\bmay\b|\bmight\b|\bperhaps\b|\bpossibly\b|\bsomewhat\b|\bseems?\b",
    "concessives": r"\bhowever\b|\balthough\b|\bnevertheless\b|\bon the other hand\b"
                   r"|\bwhereas\b|\byet\b",
    "assertive_modals": r"\bwill\b|\bmust\b",
    "need_to": r"\bneed to\b|\bhave to\b|\bgot to\b|\bgotta\b",
    "united_states": r"\bunited states\b",
    "america": r"\bamericans?\b|\bamerica\b",
    "nostalgia": r"\bagain\b|\brestor\w+\b|\bback to\b",
    "future": r"\bfuture\b|\bforward\b|\btomorrow\b|\bnew era\b",
    "religiosity": r"\bgod\b|\bfaith\b|\bpray\w*\b|\bbless\w*\b|\bsacred\b|\balmighty\b",
    "god_bless": r"\bgod bless\b",
    "us_them": r"\bthey\b|\bthem\b|\bthemselves\b",
    "superlatives": r"\bgreatest\b|\bbest\b|\bfinest\b|\bstrongest\b|\bworst\b"
                    r"|\bbiggest\b|\bmost important\b",
    # Era-safe opponent vocabulary: party-member plurals and explicit opponent
    # words only ("republican government" and "party to a treaty" don't count).
    "opponents": r"\bopponents?\b|\bpartisan\w*\b|\bdemocrats\b|\brepublicans\b"
                 r"|\bpoliticians\b|\bfake news\b|\bradical left\b|\bthe other side\b"
                 r"|\bmy opponent\b|\bcritics\b|\bthe press\b|\bthe media\b",
}

# NRC emotion groupings: hope reads as trust + anticipation + joy;
# fear appeal as fear + anger.
HOPE_EMOTIONS = {"trust", "anticipation", "joy"}
FEAR_EMOTIONS = {"fear", "anger"}


def _nrc_lexicon() -> dict[str, set[str]]:
    """Word -> set of NRC emotions. Downloads and caches the lexicon
    (research use with attribution; see README)."""
    if not NRC_CACHE.exists():
        NRC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading NRC Emotion Lexicon...")
        # saifmohammad.com 406s the default python-requests user agent
        resp = requests.get(
            NRC_URL, timeout=180, headers={"User-Agent": "Mozilla/5.0 (research use)"}
        )
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = next(n for n in zf.namelist() if n.endswith("Wordlevel-v0.92.txt"))
            NRC_CACHE.write_bytes(zf.read(name))
    lex: dict[str, set[str]] = {}
    for line in NRC_CACHE.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2] == "1":
            lex.setdefault(parts[0], set()).add(parts[1])
    return lex


def build_markers(df: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Per-speech marker counts, NRC emotion counts, and token totals."""
    if MARKERS_PATH.exists() and not force:
        return pd.read_parquet(MARKERS_PATH)

    if df is None:
        df = load()
    lex = _nrc_lexicon()
    compiled = {name: re.compile(pat) for name, pat in MARKERS.items()}

    rows = []
    for _, sp in df.iterrows():
        text = sp["transcript"].lower()
        toks = WORD_RE.findall(text)
        row = {"doc_name": sp["doc_name"], "n_words": len(toks)}
        for name, pat in compiled.items():
            row[name] = len(pat.findall(text))
        hope = fear = 0
        for t in toks:
            emos = lex.get(t)
            if emos:
                if emos & HOPE_EMOTIONS:
                    hope += 1
                if emos & FEAR_EMOTIONS:
                    fear += 1
        row["nrc_hope"] = hope
        row["nrc_fear"] = fear
        rows.append(row)

    markers = pd.DataFrame(rows)
    markers = df[["doc_name", "president", "party", "date", "year", "decade", "title"]].merge(
        markers, on="doc_name", validate="one_to_one"
    )
    markers.to_parquet(MARKERS_PATH, index=False)
    return markers


def decade_rates(markers: pd.DataFrame) -> pd.DataFrame:
    """Every marker as a per-10k-words rate, by decade."""
    g = markers.groupby("decade")
    total = g["n_words"].sum()
    out = pd.DataFrame(index=total.index)
    for name in list(MARKERS) + ["nrc_hope", "nrc_fear"]:
        out[name] = g[name].sum() / total * 10_000
    return out.reset_index()


ALL_RATE_COLS = list(MARKERS) + ["nrc_hope", "nrc_fear"]


def _yearly_sums(markers: pd.DataFrame, cols: list[str]):
    g = markers.groupby("year")
    counts = g[cols].sum()
    words = g["n_words"].sum()
    years = pd.RangeIndex(int(markers["year"].min()), int(markers["year"].max()) + 1,
                          name="year")
    return counts.reindex(years, fill_value=0), words.reindex(years, fill_value=0)


def yearly_rates(
    markers: pd.DataFrame, window: int = 5, min_words: int = 20_000
) -> pd.DataFrame:
    """Per-year rates smoothed with a centered rolling window computed on
    word totals (so sparse years are weighted, not averaged). Years whose
    window holds fewer than min_words are masked — the corpus before ~1790
    is a handful of speeches and would otherwise spike every chart."""
    counts, words = _yearly_sums(markers, ALL_RATE_COLS)
    csum = counts.rolling(window, center=True, min_periods=1).sum()
    wsum = words.rolling(window, center=True, min_periods=1).sum()
    rates = csum.div(wsum, axis=0) * 10_000
    rates[wsum < min_words] = np.nan
    return rates


def yearly_raw_rates(markers: pd.DataFrame, min_words: int = 5_000) -> pd.DataFrame:
    """Unsmoothed per-year rates (for texture markers behind the trend line);
    years with very little speech are masked."""
    counts, words = _yearly_sums(markers, ALL_RATE_COLS)
    rates = counts.div(words, axis=0) * 10_000
    rates[words < min_words] = np.nan
    return rates


def president_year_rates(markers: pd.DataFrame, min_words: int = 4_000) -> pd.DataFrame:
    """Unsmoothed rates per (president, year) — 46 years have more than one
    president speaking (1841 and 1881 have three), and the corpus files a few
    famous pre-presidency speeches under the later president, so per-year dots
    must not blend speakers. Rows with under min_words are dropped."""
    g = markers.groupby(["president", "year"])
    counts = g[ALL_RATE_COLS].sum()
    words = g["n_words"].sum()
    out = (counts.div(words, axis=0) * 10_000).reset_index()
    out["n_words"] = words.values
    out["n_speeches"] = g.size().values

    assertive = (g["boosters"].sum() + g["assertive_modals"].sum()).values
    deliberative = (g["hedges"].sum() + g["concessives"].sum()).values
    total_stance = assertive + deliberative
    out["certainty"] = np.where(
        total_stance >= 30, assertive / np.maximum(total_stance, 1), np.nan
    )
    return out[out["n_words"] >= min_words].reset_index(drop=True)


def certainty_yearly(
    markers: pd.DataFrame,
    window: int = 5,
    inaugural_only: bool = False,
    min_markers: int = 150,
) -> pd.Series:
    """Rolling assertive share of stance markers by year."""
    m = markers
    if inaugural_only:
        m = m[m["title"].str.contains("Inaugural", case=False, na=False)]
    counts, _ = _yearly_sums(m, ["boosters", "assertive_modals", "hedges", "concessives"])
    s = counts.rolling(window, center=True, min_periods=1).sum()
    assertive = s["boosters"] + s["assertive_modals"]
    deliberative = s["hedges"] + s["concessives"]
    out = assertive / (assertive + deliberative)
    out[(assertive + deliberative) < min_markers] = np.nan
    return out


def certainty_index(markers: pd.DataFrame) -> pd.Series:
    """Assertive share of stance markers: (boosters + will/must) /
    (all stance markers). 0.5 = balanced; higher = confidence over
    deliberation."""
    assertive = markers["boosters"] + markers["assertive_modals"]
    deliberative = markers["hedges"] + markers["concessives"]
    return assertive / (assertive + deliberative).clip(lower=1)


def certainty_by_decade(markers: pd.DataFrame, inaugural_only: bool = False) -> pd.Series:
    m = markers
    if inaugural_only:
        m = m[m["title"].str.contains("Inaugural", case=False, na=False)]
    g = m.groupby("decade")
    assertive = g["boosters"].sum() + g["assertive_modals"].sum()
    deliberative = g["hedges"].sum() + g["concessives"].sum()
    return assertive / (assertive + deliberative)


def vocabulary_richness(df: pd.DataFrame, window: int = 1000) -> pd.Series:
    """Mean type-token ratio over fixed windows of each president's speech,
    which makes 1-speech and 100-speech presidents comparable."""
    out = {}
    for president, group in df.groupby("president"):
        toks = []
        for text in group.sort_values("date")["transcript"]:
            toks.extend(WORD_RE.findall(text.lower()))
        windows = [toks[i:i + window] for i in range(0, len(toks) - window + 1, window)]
        if not windows:
            windows = [toks]
        out[president] = float(np.mean([len(set(w)) / len(w) for w in windows]))
    return pd.Series(out, name="ttr")


def president_scores(
    markers: pd.DataFrame, stats: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """Per-president rates for every index, plus 0-100 percentile columns
    (pct_*) for the profile radar."""
    g = markers.groupby("president")
    total = g["n_words"].sum()
    scores = pd.DataFrame(index=total.index)
    for name in list(MARKERS) + ["nrc_hope", "nrc_fear"]:
        scores[name] = g[name].sum() / total * 10_000

    assertive = g["boosters"].sum() + g["assertive_modals"].sum()
    deliberative = g["hedges"].sum() + g["concessives"].sum()
    scores["certainty"] = assertive / (assertive + deliberative)

    sg = stats.groupby("president")
    scores["fk_grade"] = sg.apply(
        lambda x: (x["fk_grade"] * x["n_tokens"]).sum() / x["n_tokens"].sum(),
        include_groups=False,
    )
    i_tot, we_tot = sg["i_count"].sum(), sg["we_count"].sum()
    scores["self_reference"] = i_tot / (i_tot + we_tot)
    scores["ttr"] = vocabulary_richness(df)

    radar = {
        "hope": scores["nrc_hope"],
        "fear": scores["nrc_fear"],
        "certainty": scores["certainty"],
        "us_vs_them": scores["us_them"],
        "self_reference": scores["self_reference"],
        "formality": scores["fk_grade"],
        "vocabulary": scores["ttr"],
        "religiosity": scores["religiosity"],
    }
    for name, series in radar.items():
        scores[f"pct_{name}"] = series.rank(pct=True) * 100

    meta = df.groupby("president").agg(
        party=("party", "first"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_speeches=("doc_name", "count"),
        n_words=("word_count", "sum"),
    )
    return meta.join(scores).reset_index()
