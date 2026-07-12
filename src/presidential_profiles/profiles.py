"""Assemble everything a president profile page needs."""

import html
import re

import numpy as np
import pandas as pd

from .corpus import DATA_DIR, load
from .fetch import PARAGRAPHS_PATH
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
        for rank, (_, r) in enumerate(scores.tail(18)[::-1].iterrows()):
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
    """Who each president invokes, and how each is invoked by successors -
    with the tone of each mention classified from its surrounding words
    (NRC positive/negative), so reverence and criticism are separated."""
    lex = indices._nrc_lexicon()

    def tone(window: str) -> int:
        pos = neg = 0
        for w in re.findall(r"[a-z']+", window.lower()):
            emos = lex.get(w)
            if emos:
                pos += "positive" in emos
                neg += "negative" in emos
        return 1 if pos > neg else (-1 if neg > pos else 0)

    first_year = df.groupby("president")["year"].min()
    compiled = {p: re.compile(pat) for p, pat in INVOCATION_PATTERNS.items()}
    invokes: dict[str, list] = {}
    invoked_by: dict[str, dict] = {
        p: {"total": 0, "pos": 0, "neg": 0} for p in INVOCATION_PATTERNS
    }
    for speaker, group in df.groupby("president"):
        text = " ".join(group["transcript"])
        mentions = []
        for target, pat in compiled.items():
            if target == speaker or first_year[target] >= first_year[speaker]:
                continue
            pos = neg = n = 0
            for m in pat.finditer(text):
                n += 1
                t = tone(text[max(0, m.start() - 130):m.end() + 130])
                pos += t == 1
                neg += t == -1
            if n:
                mentions.append({"target": target, "n": n, "pos": pos, "neg": neg})
                invoked_by[target]["total"] += n
                invoked_by[target]["pos"] += pos
                invoked_by[target]["neg"] += neg
        invokes[speaker] = sorted(mentions, key=lambda x: -x["n"])[:5]
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


# Anchor words for the one discovered topic promoted to the taxonomy display.
_EXTRA_ANCHORS = {"Discovered 5": ["soviet", "nuclear", "weapons", "peace",
                                   "freedom", "forces"]}

# Stance detection: talking about an issue is not the same as taking a
# side on it. Each spec: (side_a regex, side_a label, side_b regex,
# side_b label, mixed label). Applied where the sides are real and the
# vocabulary is era-portable.
_PEACE_RE = re.compile(
    r"\bend(?:ing)? (?:the |this |these )?wars?\b|\bbring(?:ing)? (?:our )?troops home\b"
    r"|\bwithdraw\w*\b|\bpeace\b|\bceasefire\b|\bdiploma\w+\b|\bnegotiat\w+\b"
    r"|\bdisarm\w*\b|\barms control\b|\bnever again\b")
_MARTIAL_RE = re.compile(
    r"\bvictor\w+\b|\bwin (?:the |this )?wars?\b|\bdefeat\w*\b|\bdestroy\w*\b"
    r"|\bcrush\w*\b|\bfight\w*\b|\battack\w*\b|\bstrike\w*\b|\bconquer\w*\b")
_WAR_SPEC = (_PEACE_RE, "mostly about ending wars",
             _MARTIAL_RE, "mostly about waging it",
             "waging and ending in equal measure")

_STANCE_SPECS = {
    "War & military": _WAR_SPEC,
    "Discovered 5": _WAR_SPEC,
    "Immigration": (
        re.compile(r"\bnation of immigrants\b|\bwelcom\w+\b|\bpathway\b|\basylum\b"
                   r"|\brefugees?\b|\bnaturaliz\w+\b|\bcontribut\w+\b|\bopportunit\w+\b"),
        "mostly about welcoming",
        re.compile(r"\billegal\w*\b|\bdeport\w*\b|\bsecure[ds]? (?:the |our )?border\b"
                   r"|\bborder security\b|\bcriminal\w*\b|\binvasion\b|\bexclusion\b"
                   r"|\brestrict\w*\b|\bsmuggl\w+\b"),
        "mostly about restricting",
        "welcoming and restricting in equal measure"),
    "Trade & tariffs": (
        re.compile(r"\bprotect\w* (?:american|our|home) (?:industr|worker|manufactur|labor)\w*"
                   r"|\bprotective tariff\w*|\bunfair\w*\b|\bdumping\b|\btrade deficit\b"
                   r"|\bcheat\w*\b|\bretaliat\w+\b"),
        "mostly protectionist",
        re.compile(r"\bfree trade\b|\bopen markets?\b|\btrade agreements?\b|\bnafta\b"
                   r"|\breciproc\w+\b|\blower\w* (?:the |of )?(?:tariffs?|duties)\b"
                   r"|\bfreer\b|\bliberaliz\w+\b"),
        "mostly free-trade",
        "protection and free trade in equal measure"),
}


def _issue_stance(texts, spec) -> str | None:
    re_a, label_a, re_b, label_b, label_mixed = spec
    joined = " ".join(texts).lower()
    a = len(re_a.findall(joined))
    b = len(re_b.findall(joined))
    if a + b < 8:
        return None
    share = a / (a + b)
    if share >= 0.60:
        return label_a
    if share <= 0.40:
        return label_b
    return label_mixed

_ABBREV_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Gen|Col|Capt|Hon|No|vs|U\.S)\.")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    return [s.replace("\x00", ".") for s in _SENT_SPLIT_RE.split(protected)]


def _pick_sentence(texts, anchor_terms: list[str],
                   extra_terms: list[str]) -> str | None:
    """The most quotable sentence across candidate paragraphs: scored by
    anchor-term hits (x2) plus the president's own distinctive words."""
    best, best_score = None, 0
    for text in texts:
        for sent in _sentences(text):
            if not 70 <= len(sent) <= 300:
                continue
            low = sent.lower()
            score = 2 * sum(
                bool(re.search(rf"\b{re.escape(t)}\w*", low)) for t in anchor_terms
            ) + sum(
                bool(re.search(rf"\b{re.escape(t)}\b", low)) for t in extra_terms
            )
            if score > best_score:
                best, best_score = sent.strip(), score
    return best


def issue_cards(
    df: pd.DataFrame,
    distinctive: pd.DataFrame,
    issue_df: pd.DataFrame,
    issue_meta: dict,
    top_n: int = 4,
) -> dict:
    """Per president: their top era-relative issues, each with the president's
    own distinctive vocabulary for that issue and a verbatim sentence from
    their speeches on it. Remaining distinctive terms become their 'voice'."""
    paras = pd.read_parquet(PARAGRAPHS_PATH).reset_index(drop=True)
    labels = pd.read_parquet(issues.PARA_LABELS_PATH).reset_index(drop=True)
    if len(paras) != len(labels):
        raise RuntimeError("paragraphs and labels out of sync - rerun the pipeline")
    titles = df.set_index("doc_name")[["title", "year"]]

    display = issue_meta["issues"] + ["Discovered 5"]
    anchors = {**issues.ISSUE_ANCHORS, **_EXTRA_ANCHORS}

    out = {}
    for pres, prow in issue_df.iterrows():
        mask = (labels["president"] == pres).to_numpy()
        n_paras = float(prow["n_paragraphs"])
        eligible = [
            n for n in display
            if prow[f"rel_{n}"] >= 0.75 and prow[f"share_{n}"] * n_paras >= 4
        ]
        eligible.sort(key=lambda n: -prow[f"rel_{n}"])
        eligible = eligible[:top_n]

        terms = distinctive[distinctive["president"] == pres].sort_values("rank")
        remaining = terms["term"].tolist()

        all_text = " ".join(paras.loc[mask, "text"]).lower()
        all_words = max(len(all_text.split()), 1)

        cards = []
        for name in eligible:
            issue_mask = mask & labels[name].to_numpy()
            texts = paras.loc[issue_mask, "text"]
            joined = " ".join(texts).lower()
            issue_words = max(len(joined.split()), 1)

            # A term belongs to an issue only if it is CONCENTRATED there
            # (1.5x the president's overall rate) - otherwise a president's
            # general register words would attach to every issue.
            words = []
            for t in remaining:
                n_issue = len(re.findall(rf"\b{re.escape(t)}\b", joined))
                if n_issue < 2:
                    continue
                n_all = len(re.findall(rf"\b{re.escape(t)}\b", all_text))
                if n_issue / issue_words >= 1.5 * n_all / all_words:
                    words.append(t)
                if len(words) == 5:
                    break
            for w in words:
                remaining.remove(w)

            quote = cite = None
            if len(texts):
                hits = texts.str.lower().str.count(
                    "|".join(rf"\b{re.escape(t)}\w*" for t in anchors[name]))
                top_idx = hits.nlargest(3).index
                quote = _pick_sentence(
                    [paras.loc[i, "text"] for i in top_idx], anchors[name], words
                )
                if quote:
                    src = next(i for i in top_idx if quote in paras.loc[i, "text"])
                    t = titles.loc[paras.loc[src, "doc_name"]]
                    cite = f"{t['title'].split(':', 1)[-1].strip()}, {int(t['year'])}"
                    quote = html.escape(quote)

            cards.append({
                "issue": name,
                "rel": float(prow[f"rel_{name}"]),
                "share": float(prow[f"share_{name}"]),
                "words": words,
                "quote": quote,
                "cite": cite,
                "stance": (_issue_stance(texts, _STANCE_SPECS[name])
                           if name in _STANCE_SPECS else None),
            })
        out[pres] = {"cards": cards, "voice": remaining[:8]}
    return out


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
    cards = issue_cards(df, distinctive, issue_df.set_index("president"), issue_meta)

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
        "issue_cards": cards,
    }
