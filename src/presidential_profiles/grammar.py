"""Coarse part-of-speech artifacts for descriptive language-change analysis.

This deliberately publishes broad POS shares, not a claim about syntactic
complexity. The spaCy tagger is useful for nouns, verbs, pronouns, and function
words; historical spelling, OCR, quotations, and genre shifts remain important
limitations.
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import spacy

from . import corpus, trends

GRAMMAR_DIR = corpus.DATA_DIR / "grammar"
SPEECH_POS_PATH = GRAMMAR_DIR / "speech_pos.parquet"
ERA_POS_PATH = GRAMMAR_DIR / "era_pos.csv"
MANIFEST_PATH = GRAMMAR_DIR / "manifest.json"

POS_LABELS = (
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET",
    "ADP", "CCONJ", "SCONJ", "NUM",
)


def _era(year: int) -> str:
    for name, start, end in trends.ERAS:
        if start <= year <= end:
            return name
    return "Outside declared eras"


def build(force: bool = False) -> pd.DataFrame:
    if SPEECH_POS_PATH.exists() and not force:
        return pd.read_parquet(SPEECH_POS_PATH)
    df = corpus.load()
    # Keep the attribute ruler: the tagger predicts Penn tags and the ruler
    # maps them onto the coarse Universal POS values published here.
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.enable_pipe("senter")
    nlp.max_length = 2_000_000
    rows = []
    for i, doc in enumerate(nlp.pipe(df.transcript.tolist(), batch_size=16)):
        counts = Counter(
            token.pos_ for token in doc
            if token.is_alpha and token.pos_ in POS_LABELS
        )
        tagged = sum(counts.values())
        rows.append({
            "doc_name": df.iloc[i].doc_name,
            "n_tagged_tokens": tagged,
            **{f"pos_{label.lower()}": counts[label] for label in POS_LABELS},
        })
        if (i + 1) % 100 == 0:
            print(f"  POS-tagged {i + 1}/{len(df)} speeches")
    out = df[["doc_name", "president", "year", "title"]].merge(
        pd.DataFrame(rows), on="doc_name", validate="one_to_one"
    )
    out["era"] = out.year.map(_era)
    GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SPEECH_POS_PATH, index=False)

    count_cols = [f"pos_{label.lower()}" for label in POS_LABELS]
    era = out.groupby("era", sort=False)[["n_tagged_tokens", *count_cols]].sum()
    era_rows = []
    for era_name, row in era.iterrows():
        for label, column in zip(POS_LABELS, count_cols):
            era_rows.append({
                "era": era_name, "part_of_speech": label,
                "share_percent": float(row[column] / max(row.n_tagged_tokens, 1) * 100),
                "n_tagged_tokens": int(row.n_tagged_tokens),
            })
    pd.DataFrame(era_rows).to_csv(ERA_POS_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps({
        "schema_version": "grammar-pos-v1",
        "model": "spaCy en_core_web_sm coarse POS tagger",
        "unit": "speech",
        "labels": list(POS_LABELS),
        "limitations": [
            "Coarse POS shares are not a full syntax or grammar model.",
            "Historical spelling, OCR, quotations, and genre can affect tags.",
            "The artifact describes the formal speech corpus, not all presidential communication.",
        ],
    }, indent=2))
    return out


if __name__ == "__main__":
    built = build(force=True)
    print(f"wrote {SPEECH_POS_PATH.relative_to(corpus.REPO_ROOT)} ({len(built):,} speeches)")
