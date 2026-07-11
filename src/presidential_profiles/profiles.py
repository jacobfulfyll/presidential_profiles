"""Assemble everything a president profile page needs."""

import re

import numpy as np
import pandas as pd

from .corpus import DATA_DIR, load
from . import indices, issues, rhetoric, similarity, trends

DISTINCTIVE_PATH = DATA_DIR / "president_distinctive.parquet"

MILLER_URL = "https://millercenter.org/the-presidency/presidential-speeches/"

# Conservative invocation patterns: full names / titled surnames only, so
# Jefferson Davis, Hillary Clinton, and Henry Ford don't count. Ambiguous
# surnames (Johnson, Bush, Ford, both Harrisons/Adamses) are skipped.
INVOCATION_PATTERNS = {
    "George Washington": r"George Washington|President Washington",
    "Thomas Jefferson": r"Thomas Jefferson|President Jefferson",
    "James Madison": r"James Madison|President Madison",
    "Andrew Jackson": r"Andrew Jackson|President Jackson",
    "Abraham Lincoln": r"\bLincoln\b",
    "Ulysses S. Grant": r"Ulysses|President Grant",
    "Theodore Roosevelt": r"Theodore Roosevelt|Teddy Roosevelt",
    "Woodrow Wilson": r"Woodrow Wilson|President Wilson",
    "Franklin D. Roosevelt": r"Franklin D?\.? ?Roosevelt|Franklin Delano",
    "Harry S. Truman": r"\bTruman\b",
    "Dwight D. Eisenhower": r"\bEisenhower\b",
    "John F. Kennedy": r"John F\.? Kennedy|President Kennedy|Jack Kennedy",
    "Richard M. Nixon": r"\bNixon\b",
    "Jimmy Carter": r"Jimmy Carter|President Carter",
    "Ronald Reagan": r"\bReagan\b",
    "Bill Clinton": r"Bill Clinton|President Clinton",
    "Barack Obama": r"\bObama\b",
    "Donald Trump": r"\bTrump\b",
    "Joe Biden": r"\bBiden\b",
}

RADAR_AXES = [
    ("hope", "Hope"),
    ("fear", "Fear appeal"),
    ("certainty", "Certainty"),
    ("us_vs_them", "Us vs them"),
    ("self_reference", "Self-reference"),
    ("formality", "Formality"),
    ("vocabulary", "Vocabulary"),
    ("religiosity", "Religiosity"),
]


def slug(president: str) -> str:
    return re.sub(r"[^a-z]+", "-", president.lower()).strip("-")


def build_distinctive(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Top distinctive terms per president (log-odds vs all other presidents)."""
    if DISTINCTIVE_PATH.exists() and not force:
        return pd.read_parquet(DISTINCTIVE_PATH)

    counts = {
        p: trends.word_counts(g["transcript"])
        for p, g in df.groupby("president")
    }
    total = sum(counts.values(), start=trends.Counter())
    rows = []
    for p, c_p in counts.items():
        rest = total - c_p
        n_p = sum(c_p.values())
        min_count = 8 if n_p < 30_000 else 20
        scores = trends.log_odds_scores(c_p, rest, min_count=min_count)
        for rank, (_, r) in enumerate(scores.tail(10)[::-1].iterrows()):
            rows.append({"president": p, "term": r["term"],
                         "z": r["z"], "rank": rank})
    out = pd.DataFrame(rows)
    out.to_parquet(DISTINCTIVE_PATH, index=False)
    return out


def signature_speeches(df: pd.DataFrame, adj: pd.DataFrame, top_n: int = 5) -> dict:
    """Per president: speeches that most express what makes them distinct —
    highest cosine between the speech vector and the president's era-adjusted
    direction."""
    speech_emb = pd.read_parquet(similarity.SPEECH_EMB_PATH)
    vec_cols = [c for c in speech_emb.columns if c.startswith("e")]
    S = speech_emb[vec_cols].to_numpy()
    adj_vecs = {r["president"]: adj.loc[i, vec_cols].to_numpy(dtype=float)
                for i, r in adj.iterrows()}

    merged = df[["doc_name", "president", "title", "year"]].merge(
        speech_emb[["doc_name"]].assign(row=range(len(speech_emb))), on="doc_name"
    )
    out = {}
    for p, group in merged.groupby("president"):
        scores = S[group["row"].to_numpy()] @ adj_vecs[p]
        top = group.assign(score=scores).nlargest(top_n, "score")
        out[p] = [
            {"title": r["title"], "year": int(r["year"]),
             "url": MILLER_URL + r["doc_name"]}
            for _, r in top.iterrows()
        ]
    return out


def invocations(df: pd.DataFrame) -> tuple[dict, dict]:
    """(who each president invokes, how often each is invoked by successors)."""
    first_year = df.groupby("president")["year"].min()
    compiled = {p: re.compile(pat) for p, pat in INVOCATION_PATTERNS.items()}
    invokes: dict[str, list] = {}
    invoked_by: dict[str, int] = {p: 0 for p in INVOCATION_PATTERNS}
    for speaker, group in df.groupby("president"):
        text = " ".join(group["transcript"])
        mentions = []
        for target, pat in compiled.items():
            if target == speaker or first_year[target] >= first_year[speaker]:
                continue
            n = len(pat.findall(text))
            if n:
                mentions.append((target, n))
                invoked_by[target] += n
        invokes[speaker] = sorted(mentions, key=lambda x: -x[1])[:5]
    return invokes, invoked_by


def neighbors(adj: pd.DataFrame, issue_df: pd.DataFrame) -> tuple[dict, dict]:
    """('sounds like' era-adjusted voice neighbors, 'same agenda' issue neighbors)."""
    vec_cols = [c for c in adj.columns if c.startswith("e")]
    V = adj[vec_cols].to_numpy(dtype=float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ V.T
    names = adj["president"].tolist()

    share_cols = [c for c in issue_df.columns if c.startswith("share_")]
    A = issue_df[share_cols].to_numpy(dtype=float)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    agenda_sim = A @ A.T
    agenda_names = issue_df["president"].tolist()

    voice, agenda = {}, {}
    for i, p in enumerate(names):
        order = np.argsort(sim[i])[::-1]
        voice[p] = [(names[j], float(sim[i, j])) for j in order if j != i][:3]
    for i, p in enumerate(agenda_names):
        order = np.argsort(agenda_sim[i])[::-1]
        agenda[p] = [(agenda_names[j], float(agenda_sim[i, j]))
                     for j in order if j != i][:3]
    return voice, agenda


def build_profile_data(force: bool = False) -> dict:
    """Everything the profile pages need, keyed by president."""
    df = load()
    stats = rhetoric.build_stats(df)
    markers = indices.build_markers(df)
    scores = indices.president_scores(markers, stats, df)
    issue_df, issue_meta = issues.build_issues()
    adj = similarity.build_adjusted()
    distinctive = build_distinctive(df, force=force)
    sigs = signature_speeches(df, adj)
    invokes, invoked_by = invocations(df)
    voice, agenda = neighbors(adj, issue_df)

    return {
        "df": df,
        "scores": scores.set_index("president"),
        "issues": issue_df.set_index("president"),
        "issue_meta": issue_meta,
        "adj": adj,
        "distinctive": distinctive,
        "signatures": sigs,
        "invokes": invokes,
        "invoked_by": invoked_by,
        "voice_neighbors": voice,
        "agenda_neighbors": agenda,
    }
