"""Assemble everything a president profile page needs."""

import html
import re
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from .corpus import DATA_DIR, PARTY, load
from .fetch import PARAGRAPHS_PATH
from . import indices, issues, rhetoric, similarity, topic_quality, trends

DISTINCTIVE_PATH = DATA_DIR / "president_distinctive.parquet"

MILLER_ORIGIN = "https://millercenter.org"
MILLER_SPEECH_PATH = "/the-presidency/presidential-speeches/"
MILLER_URL = MILLER_ORIGIN + MILLER_SPEECH_PATH

# --- Issue card thresholds -------------------------------------------------
# These are fixed from first principles, NOT tuned against which presidents
# they happen to flag. Both reuse lines the codebase already draws elsewhere,
# so a card's two axes are judged by one consistent standard.

# The era-relative bar. This is the pre-existing eligibility cutoff: the
# codebase already asserts 0.75 pp is the line between "stands out from their
# era" and "doesn't". We keep it as the positive bar AND reuse it, as a
# two-sided band, for the definition of "not distinctive" - an issue whose
# |rel| falls under it is one the codebase already considers era-noise.
REL_DISTINCT_PP = 0.75

# The raw-attention bar, as a multiple of the issue's own corpus-wide base
# rate. 1.5x is the same concentration ratio issue_cards() already uses to
# decide a term "belongs to" an issue rather than to a president's general
# register - the same question (is this meaningfully above background?) gets
# the same answer here.
RAW_ELEVATED_MULT = 1.5

# An issue needs this many of the president's paragraphs behind it before it
# gets a card, so a one-speech president gets no headline from a stray
# metaphor.
MIN_ISSUE_PARAS = 4

# Below this, a president's rates are too thin to state without a caveat.
# Mirrors the dashboard's sparse-president cutoff (site.py), which drops them
# from per-president graphics entirely; profiles keep them but say so.
SPARSE_MIN_SPEECHES = indices.PRESIDENT_PERCENTILE_MIN_SPEECHES

# Canonical corpus keys remain unchanged in data contracts and URLs. These
# two abbreviated source labels receive their full public display forms.
PRESIDENT_DISPLAY_NAMES = {
    "William Harrison": "William Henry Harrison",
    "William Taft": "William Howard Taft",
}

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
RADAR_VALUE_COLUMNS = {
    "hope": "nrc_hope",
    "fear": "nrc_fear",
    "certainty": "certainty",
    "us_vs_them": "us_them",
    "self_reference": "self_reference",
    "formality": "fk_grade",
    "vocabulary": "ttr",
    "religiosity": "religiosity",
}

FEATURE_SIMILARITY = {
    "ai_topics": {
        "label": "Fine AI topic mix",
        "short": "AI topics",
        "description": "Cosine similarity across all 50 AI-labeled topic shares.",
        "standardized": False,
    },
    "ai_domains": {
        "label": "Broad AI domain mix",
        "short": "AI domains",
        "description": "Cosine similarity across the 17 broad AI topic domains.",
        "standardized": False,
    },
    "ai_rhetoric": {
        "label": "AI rhetoric profile",
        "short": "AI rhetoric",
        "description": (
            "Cosine similarity across six standardized measures: partisan attack, "
            "enemy naming, zero-sum framing, proposals, values, and topic breadth."
        ),
        "standardized": True,
    },
    "rhetorical_fingerprint": {
        "label": "Rhetorical fingerprint",
        "short": "Rhetorical fingerprint",
        "description": (
            "Cosine similarity across eight standardized, named lexical measures: "
            "hope, fear, certainty, us-versus-them, self-reference, formality, "
            "vocabulary, and religiosity."
        ),
        "standardized": True,
    },
    "legacy_issues": {
        "label": "Legacy issue mix",
        "short": "Legacy issues",
        "description": "Cosine similarity across the 16 deterministic CorEx issue shares.",
        "standardized": False,
    },
}


def slug(president: str) -> str:
    return re.sub(r"[^a-z]+", "-", president.lower()).strip("-")


def public_display_name(president: str) -> str:
    """Return the public-facing full name without changing corpus identity."""
    return PRESIDENT_DISPLAY_NAMES.get(president, president)


def president_chronology(
    presidents,
    *,
    require_complete: bool = False,
) -> list[str]:
    """Order profile keys by the repository's declared presidency sequence."""
    supplied = list(presidents)
    if require_complete and len(supplied) != len(set(supplied)):
        raise ValueError("Complete profile president keys must be unique")
    names = list(dict.fromkeys(supplied))
    name_set = set(names)
    if require_complete and (
        len(names) != len(PARTY) or name_set != set(PARTY)
    ):
        missing = sorted(set(PARTY) - name_set)
        extra = sorted(name_set - set(PARTY))
        raise ValueError(
            f"Profile president keys do not match corpus.PARTY; "
            f"expected={len(PARTY)}, observed={len(names)}, "
            f"missing={missing}, extra={extra}"
        )
    ordered = [name for name in PARTY if name in name_set]
    ordered.extend(sorted(name_set - set(PARTY)))
    if len({slug(name) for name in ordered}) != len(ordered):
        raise ValueError("Profile president slugs must be unique")
    return ordered


def format_ordinal(value) -> str:
    """Format a whole-number ordinal with the 11/12/13 exceptions."""
    number = int(round(float(value)))
    remainder_100 = abs(number) % 100
    if 11 <= remainder_100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(abs(number) % 10, "th")
    return f"{number}{suffix}"


def format_percentile(value, *, unavailable: str = "Not ranked") -> str:
    """Format a nullable percentile without manufacturing a midpoint."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return unavailable
    return f"{format_ordinal(value)} percentile"


def format_speech_count(value) -> str:
    """Use the shared singular/plural label for corpus speech counts."""
    count = int(value)
    return f"{count:,} {'speech' if count == 1 else 'speeches'}"


def miller_speech_url(value: str) -> str:
    """Normalize a Miller Center speech slug, route, or absolute URL.

    Corpus rows usually store an already-prefixed route. Absolute URLs are
    never concatenated, and an accidental duplicated Miller route is reduced
    to one canonical prefix.
    """
    raw = str(value or "").strip()
    if not raw:
        return MILLER_URL
    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        if parsed.netloc.lower().removeprefix("www.") != "millercenter.org":
            return raw
        path = parsed.path
        last_prefix = path.rfind(MILLER_SPEECH_PATH)
        if last_prefix > 0:
            path = MILLER_SPEECH_PATH + path[last_prefix + len(MILLER_SPEECH_PATH):]
        return urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
        )

    route = raw.replace("\\", "/")
    prefixed = MILLER_SPEECH_PATH.lstrip("/")
    if route.startswith("/"):
        path = route
    elif route.startswith(prefixed):
        path = "/" + route
    else:
        path = MILLER_SPEECH_PATH + route.lstrip("/")
    last_prefix = path.rfind(MILLER_SPEECH_PATH)
    if last_prefix > 0:
        path = MILLER_SPEECH_PATH + path[last_prefix + len(MILLER_SPEECH_PATH):]
    return urljoin(MILLER_ORIGIN, path)


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
             "url": miller_speech_url(r["doc_name"])}
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


def _named_feature_neighbors(
    frame: pd.DataFrame,
    counts: pd.Series,
    *,
    standardized: bool,
    top_n: int = 6,
) -> dict[str, list[dict]]:
    """Nearest presidents in a small, declared feature space.

    Topic and issue shares remain non-negative and are compared as compositions.
    Rhetorical measures have unlike units, so they are standardized against the
    presidents with at least five speeches before cosine similarity is computed.
    Thin presidents remain queryable, but only adequately sampled presidents can
    be returned as ranked neighbors.
    """
    frame = frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    eligible = counts.reindex(frame.index).fillna(0).ge(SPARSE_MIN_SPEECHES)
    reference = frame.loc[eligible]
    values = frame.to_numpy(float)
    if standardized:
        center = reference.mean(axis=0).to_numpy(float)
        spread = reference.std(axis=0, ddof=0).replace(0, 1).to_numpy(float)
        values = (values - center) / spread
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.where(norms == 0, 1, norms)
    similarities = normalized @ normalized.T
    names = frame.index.tolist()
    candidate_positions = np.flatnonzero(eligible.to_numpy())
    output: dict[str, list[dict]] = {}
    for i, president in enumerate(names):
        ranked = sorted(
            (
                (float(similarities[i, j]), names[j])
                for j in candidate_positions
                if j != i and np.isfinite(similarities[i, j])
            ),
            reverse=True,
        )[:top_n]
        output[president] = [
            {"president": other, "similarity": round(score, 4)}
            for score, other in ranked
        ]
    return output


def feature_neighbors(data: dict, ai_data: dict) -> dict[str, dict[str, list[dict]]]:
    """Five interpretable president-similarity views used across the site."""
    scores = data["scores"]
    names = scores.index.tolist()
    counts = scores["n_speeches"]
    ai_by = ai_data["by_president"]
    topic_names = [entry["name"] for entry in ai_data["taxonomy"]["level2"]]
    domain_names = [entry["name"] for entry in ai_data["taxonomy"]["level1"]]

    def attention_frame(key: str, labels: list[str]) -> pd.DataFrame:
        return pd.DataFrame.from_dict({
            president: {
                item["name"]: float(item["share"])
                for item in ai_by[president][key]
            }
            for president in names
        }, orient="index").reindex(index=names, columns=labels, fill_value=0)

    topic_frame = attention_frame("topic_attention", topic_names)
    domain_frame = attention_frame("domain_attention", domain_names)
    ai_rhetoric = pd.DataFrame.from_dict({
        president: {
            "party_attack": ai_by[president]["flags"]["party_attack"],
            "enemy_naming": ai_by[president]["flags"]["enemy_naming"],
            "zero_sum": ai_by[president]["flags"]["zero_sum"],
            "proposal": ai_by[president]["proposal_values"]["proposal"],
            "values": ai_by[president]["proposal_values"]["values"],
            "topic_breadth": ai_by[president]["ai_radar"]["topic_breadth"]["absolute"],
        }
        for president in names
    }, orient="index")
    fingerprint = scores[
        [RADAR_VALUE_COLUMNS[key] for key, _ in RADAR_AXES]
    ].copy()
    issue_columns = [column for column in data["issues"] if column.startswith("share_")]
    legacy = data["issues"].reindex(names)[issue_columns].copy()

    frames = {
        "ai_topics": topic_frame,
        "ai_domains": domain_frame,
        "ai_rhetoric": ai_rhetoric,
        "rhetorical_fingerprint": fingerprint,
        "legacy_issues": legacy,
    }
    by_category = {
        key: _named_feature_neighbors(
            frame,
            counts,
            standardized=bool(FEATURE_SIMILARITY[key]["standardized"]),
        )
        for key, frame in frames.items()
    }
    return {
        president: {
            key: by_category[key].get(president, [])
            for key in FEATURE_SIMILARITY
        }
        for president in names
    }


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


def issue_base_rates(issue_df: pd.DataFrame, display: list[str]) -> dict:
    """Each issue's corpus-wide base rate: the fraction of all paragraphs in
    the corpus that touch it. Paragraph-weighted, not a mean of per-president
    shares - otherwise Garfield's 29 paragraphs would count as heavily as
    FDR's thousands in defining what 'normal attention' means."""
    n = issue_df["n_paragraphs"].to_numpy()
    return {
        name: float((issue_df[f"share_{name}"].to_numpy() * n).sum() / n.sum())
        for name in display
    }


def issue_strengths(prow, name: str, base: dict) -> tuple[float, float]:
    """An issue's ``(rel_strength, raw_strength)`` for president row ``prow``,
    each normalised so 1.0 is exactly its threshold. ``raw_strength`` is 0 when
    the issue's corpus-wide base rate is 0 (guards against divide-by-zero)."""
    rel = float(prow[f"rel_{name}"]) / REL_DISTINCT_PP
    raw = (float(prow[f"share_{name}"]) / base[name]) / RAW_ELEVATED_MULT \
        if base[name] > 0 else 0.0
    return rel, raw


def is_topic_of_day(raw_strength: float, rel: float) -> bool:
    """A "topic of the day": elevated well above the historical base rate, yet
    indistinguishable from their own era. The subject was in the air; the
    president is not the reason for it."""
    return bool(raw_strength >= 1.0 and abs(rel) < REL_DISTINCT_PP)


def issue_cards(
    df: pd.DataFrame,
    distinctive: pd.DataFrame,
    issue_df: pd.DataFrame,
    issue_meta: dict,
    top_n: int = 4,
) -> dict:
    """Per president: the issues that either defined their agenda in raw terms
    or set them apart from their era, each with the president's own
    distinctive vocabulary for that issue and a verbatim sentence from their
    speeches on it. Remaining distinctive terms become their 'voice'."""
    paras = pd.read_parquet(PARAGRAPHS_PATH)
    labels = pd.read_parquet(issues.PARA_LABELS_PATH)
    merged = paras.merge(
        labels, on=["doc_name", "para_idx"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(paras) or len(merged) != len(labels):
        raise RuntimeError("paragraphs and labels key sets diverge - rerun the pipeline")
    paras = merged.reset_index(drop=True)
    titles = df.set_index("doc_name")[["title", "year"]]

    display = topic_quality.display_issues(issue_meta["issues"])
    anchors = {**issues.ISSUE_ANCHORS, **_EXTRA_ANCHORS}
    base = issue_base_rates(issue_df, display)
    n_speeches = df.groupby("president").size()

    out = {}
    for pres, prow in issue_df.iterrows():
        mask = (paras["president"] == pres).to_numpy()
        n_paras = float(prow["n_paragraphs"])

        # An issue earns a card on EITHER axis. Gating on era-relative alone
        # (as this once did) discards the issues that consumed a presidency
        # but consumed their contemporaries equally - a wartime president
        # talking war at wartime rates scored ~0 rel and vanished from his own
        # profile. Both strengths are normalised so that 1.0 is exactly the
        # threshold, which makes them comparable on one scale for ranking.
        eligible = [
            n for n in display
            if prow[f"share_{n}"] * n_paras >= MIN_ISSUE_PARAS
            and max(issue_strengths(prow, n, base)) >= 1.0
        ]
        # Rank by whichever axis the issue is strongest on, so a defining-but-
        # ordinary issue and a distinctive-but-small one both surface.
        eligible.sort(key=lambda n: -max(issue_strengths(prow, n, base)))
        eligible = eligible[:top_n]

        terms = distinctive[distinctive["president"] == pres].sort_values("rank")
        remaining = terms["term"].tolist()

        all_text = " ".join(paras.loc[mask, "text"]).lower()
        all_words = max(len(all_text.split()), 1)

        cards = []
        for name in eligible:
            issue_mask = mask & paras[name].to_numpy()
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

            rel = float(prow[f"rel_{name}"])
            share = float(prow[f"share_{name}"])
            _, raw_strength = issue_strengths(prow, name, base)
            cards.append({
                "issue": name,
                "rel": rel,
                "share": share,
                "base": base[name],
                "topic_of_day": is_topic_of_day(raw_strength, rel),
                "words": words,
                "quote": quote,
                "cite": cite,
                "stance": (_issue_stance(texts, _STANCE_SPECS[name])
                           if name in _STANCE_SPECS else None),
            })
        out[pres] = {
            "cards": cards,
            "voice": remaining[:8],
            "n_speeches": int(n_speeches.get(pres, 0)),
            "n_paragraphs": int(n_paras),
            "low_confidence": bool(n_speeches.get(pres, 0) < SPARSE_MIN_SPEECHES),
        }
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
    invocation_v2 = {}
    candidate_path = DATA_DIR / "invocations_v2" / "candidates.parquet"
    label_path = DATA_DIR / "invocations_v2" / "classifications.parquet"
    if candidate_path.exists() and label_path.exists():
        candidates = pd.read_parquet(candidate_path)
        labels = pd.read_parquet(label_path)
        classified = candidates.merge(labels, on="candidate_id", validate="one_to_one")
        classified = classified[
            classified.excluded_reason.eq("")
            & classified.speaker.ne(classified.target)
            & classified.target_status.eq("former_president")
        ]
        for president, group in classified.groupby("speaker"):
            summary = (group.groupby(["target", "function", "stance"]).size()
                       .rename("mentions").reset_index()
                       .sort_values("mentions", ascending=False))
            invocation_v2[president] = summary.to_dict("records")

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
        "invocation_v2": invocation_v2,
    }
