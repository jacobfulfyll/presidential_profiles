"""Issue topics: what presidents actually cared about.

Hybrid anchored topic model (CorEx) over paragraph-sized chunks:
a curated issue taxonomy is anchored so every president is scored on the
same axes, plus free topics that let the corpus surface themes we didn't
anticipate. Vocabulary is restricted to content words so topics read as
issues rather than register or era.
"""

import json

import numpy as np
import pandas as pd
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .corpus import DATA_DIR, load
from .fetch import PARAGRAPHS_PATH
from .similarity import ERA_WINDOW

ISSUES_PRESIDENT_PATH = DATA_DIR / "issues_president.parquet"
ISSUES_META_PATH = DATA_DIR / "issues_meta.json"
PARA_LABELS_PATH = DATA_DIR / "paragraph_issues.parquet"

# The curated taxonomy. Anchor terms span eras (tariff & broadband both count
# as their issue) — a paragraph about any era's version of the issue lands in
# the same bucket, which is what makes presidents comparable across time.
ISSUE_ANCHORS = {
    "Economy & jobs": ["economy", "jobs", "unemployment", "industry", "business",
                       "prosperity", "wages", "workers"],
    "Taxes & budget": ["tax", "taxes", "budget", "deficit", "debt", "spending",
                       "revenue", "appropriations"],
    "War & military": ["war", "military", "army", "navy", "troops", "defense",
                       "soldiers", "battle"],
    "Foreign policy": ["treaty", "foreign", "diplomatic", "allies", "alliance",
                       "relations", "embassy", "negotiations"],
    "Immigration": ["immigration", "immigrants", "border", "citizenship",
                    "naturalization", "aliens"],
    "Civil rights & race": ["slavery", "race", "racial", "discrimination",
                            "segregation", "equality", "emancipation", "negro"],
    "Health care": ["health", "medicare", "medicaid", "insurance", "hospitals",
                    "disease", "doctors", "patients"],
    "Education": ["education", "schools", "teachers", "students", "colleges",
                  "learning"],
    # Polysemy warning: "energy" (18th-c. vigor), "duties" (obligations), and
    # "justice" (the abstract virtue) are NOT anchors — they drag old speeches
    # into modern issues.
    "Crime & justice": ["crime", "police", "prison", "courts", "drugs",
                        "violence", "criminals"],
    "Energy & environment": ["oil", "environment", "climate",
                             "conservation", "pollution", "forests", "coal"],
    "Trade & tariffs": ["tariff", "trade", "commerce", "exports",
                        "imports", "markets"],
    "Agriculture": ["farmers", "agriculture", "farm", "crops", "livestock"],
    "Religion & values": ["god", "faith", "prayer", "moral", "church",
                          "spiritual"],
    "Money & banking": ["currency", "gold", "silver", "bank", "banking",
                        "inflation", "dollar", "coinage"],
    "Infrastructure": ["railroad", "railroads", "roads", "highways", "canals",
                       "infrastructure", "broadband", "bridges"],
}

N_FREE_TOPICS = 7
ANCHOR_STRENGTH = 6

REGISTER_STOP = {
    "applause", "laughter", "thank", "thanks", "tonight", "today", "going",
    "want", "know", "say", "said", "think", "just", "let", "make", "way",
    "things", "thing", "lot", "got", "america", "american", "americans",
    "united", "states", "people", "country", "nation", "national",
    "president", "congress", "government", "world", "time", "year", "years",
    "great", "good", "new", "man", "men", "day", "fellow", "citizens",
}


def build_issues(force: bool = False):
    """Fit the anchored topic model; return (president_issues, meta)."""
    if ISSUES_PRESIDENT_PATH.exists() and ISSUES_META_PATH.exists() and not force:
        return (
            pd.read_parquet(ISSUES_PRESIDENT_PATH),
            json.loads(ISSUES_META_PATH.read_text()),
        )

    paras = pd.read_parquet(PARAGRAPHS_PATH)
    speeches = load()
    paras = paras.merge(
        speeches[["doc_name", "president", "year", "decade"]], on="doc_name"
    )

    stop = list(ENGLISH_STOP_WORDS | REGISTER_STOP)
    vectorizer = CountVectorizer(
        stop_words=stop, binary=True, min_df=20, max_df=0.3,
        max_features=25_000, token_pattern=r"[a-zA-Z][a-zA-Z]+",
    )
    X = ss.csr_matrix(vectorizer.fit_transform(paras["text"].str.lower()))
    vocab = list(vectorizer.get_feature_names_out())
    vocab_set = set(vocab)

    anchors = [
        [t for t in terms if t in vocab_set] for terms in ISSUE_ANCHORS.values()
    ]
    issue_names = list(ISSUE_ANCHORS.keys())

    model = ct.Corex(n_hidden=len(anchors) + N_FREE_TOPICS, seed=42)
    model.fit(X, words=vocab, anchors=anchors, anchor_strength=ANCHOR_STRENGTH)

    topic_words = {}
    for k in range(model.n_hidden):
        name = issue_names[k] if k < len(issue_names) else f"Discovered {k - len(issue_names) + 1}"
        topic_words[name] = [w for w, *_ in model.get_topics(topic=k, n_words=12)]

    labels = model.labels  # (n_paragraphs, n_topics) booleans
    all_names = issue_names + [f"Discovered {i + 1}" for i in range(N_FREE_TOPICS)]
    para_labels = pd.DataFrame(labels, columns=all_names)
    para_labels.insert(0, "doc_name", paras["doc_name"].values)
    para_labels.insert(1, "para_idx", paras["para_idx"].values)
    para_labels.insert(2, "president", paras["president"].values)
    para_labels.insert(3, "year", paras["year"].values)
    para_labels.to_parquet(PARA_LABELS_PATH, index=False)

    # President x issue: share of paragraphs touching each issue.
    shares = para_labels.groupby("president")[all_names].mean()

    # Era-relative emphasis: president share minus the mean share of
    # contemporaries (first corpus year within ERA_WINDOW), in percentage
    # points. Positive = talked about it more than their era did.
    first_year = paras.groupby("president")["year"].min()
    rel = pd.DataFrame(index=shares.index, columns=all_names, dtype=float)
    for pres in shares.index:
        mask = (first_year - first_year[pres]).abs() <= ERA_WINDOW
        mask[pres] = False
        peers = shares[mask] if mask.sum() >= 2 else shares.drop(pres)
        rel.loc[pres] = (shares.loc[pres] - peers.mean()) * 100

    out = shares.add_prefix("share_").join(rel.add_prefix("rel_"))
    out["n_paragraphs"] = para_labels.groupby("president").size()
    out = out.reset_index()
    out.to_parquet(ISSUES_PRESIDENT_PATH, index=False)

    meta = {"issues": issue_names, "topic_words": topic_words,
            "n_paragraphs": len(paras)}
    ISSUES_META_PATH.write_text(json.dumps(meta, indent=2))
    return out, meta
