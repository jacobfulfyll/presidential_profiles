"""Coherence scoring for CorEx topics, and the display-name registry.

Two jobs, joined because the second depends on the first:

1. **Score** every CorEx topic's NPMI coherence on the *same yardstick* the
   embedding clusters were scored on, so the numbers in `issues_meta.json` and
   `paragraph_clusters_meta.json` are directly comparable.
2. **Decide what the site is allowed to show.** CorEx's 7 free topics are half
   real themes ("nations, peace, soviet, nuclear") and half procedural or
   tokenizer sludge ("ve, don, ll"). The scored, named, surfaceable subset lives
   in an external names file so that display names can change without renaming
   the `Discovered N` columns that `paragraph_issues.parquet`, `issues_meta.json`
   and the tests all key on.

REUSE, NOT REIMPLEMENTATION
    `embed_topics._fit_vectorizer` / `_npmi` are imported, never cloned. A second
    NPMI implementation over a second vocabulary would make CorEx coherence and
    cluster coherence incomparable — which is precisely the property the
    method-triangulation work exists to establish. `embed_topics` is read-only
    here.

THE COHERENCE ASYMMETRY — DO NOT "FIX" THIS
    We compute NPMI for all 22 CorEx topics but only ever *threshold* the 7
    discovered ones. This looks like an inconsistency and is not. Anchored issues
    deliberately span vocabulary that never co-occurs in one paragraph
    ("railroad" and "broadband" are both Infrastructure by design; slavery-era
    and segregation-era wording are both Civil rights & race). Their low NPMI is
    the taxonomy working as intended, not a defect — see the COHERENCE CAVEAT in
    `embed_topics.py`. Thresholding them would delete the taxonomy's whole point.
    The anchored scores exist only to be *published as a caveat table*
    (`notes/topic-comparison-report-v1.md`), never to gate anything.
"""

import json
from datetime import date

import pandas as pd

from .corpus import DATA_DIR
from .embed_topics import N_TERMS, _fit_vectorizer, _npmi
from .fetch import PARAGRAPHS_PATH
from .issues import ISSUES_META_PATH, N_FREE_TOPICS

NAMES_PATH = DATA_DIR / "topic_display_names.json"

# The prefix CorEx's free (unanchored) topics are named with in issues.py.
# Kept as a prefix test rather than a hardcoded list so N_FREE_TOPICS stays the
# single source of truth for how many there are.
DISCOVERED_PREFIX = "Discovered "

# ---------------------------------------------------------------------------
# Pre-registered coherence gate (fixed BEFORE any score was computed)
# ---------------------------------------------------------------------------
# A discovered topic is noise iff EITHER clause fires:
#
#   (a) npmi <= COHERENCE_FLOOR. Zero is not a tuned cutoff — it is NPMI's own
#       null. NPMI == 0 means the topic's top terms co-occur exactly at chance,
#       so a topic at or below it is word soup by construction.
#
#   (b) fewer than MIN_SCORED_FRACTION of its top terms survive into the shared
#       content vocabulary. `topics.EXTRA_STOP` already removes contraction
#       fragments ("ve", "ll", "don", "didn") and honorifics ("mr"), which are
#       exactly what defines CorEx's sludge topics. Without this clause those
#       terms would be silently dropped and the topic scored on its handful of
#       incidental content words — inflating the coherence of the worst topics.
#       A topic whose defining terms are mostly register artifacts is not a
#       theme, however well its leftovers happen to co-occur.
COHERENCE_FLOOR = 0.0
MIN_SCORED_FRACTION = 0.5


def is_discovered(name: str) -> bool:
    """True for CorEx's free topics (`Discovered 1`..`Discovered N`)."""
    return name.startswith(DISCOVERED_PREFIX)


def compute_coherence(
    topic_words: dict[str, list[str]],
    texts: pd.Series,
    n_terms: int = N_TERMS,
) -> dict[str, dict]:
    """NPMI + vocabulary coverage for each topic in `topic_words`.

    `topic_words` maps topic name -> its ranked top terms (as `issues_meta.json`
    stores them). Terms are truncated to `n_terms` so CorEx topics are scored on
    the same number of terms as the embedding clusters in
    `paragraph_clusters_meta.json` — same vocabulary AND same term count is what
    makes the two sets of numbers comparable.

    Returns, per topic: `npmi`, `n_scored` (top terms present in the shared
    vocabulary), `n_terms`, `scored_fraction`, and the truncated `terms`
    themselves, so a caller can publish exactly what was scored. Coverage is
    reported rather than silently absorbed because `_npmi` skips
    out-of-vocabulary terms, and for the sludge topics that is most of them.

    Pure with respect to the corpus: pass any text series (a tiny synthetic one
    in tests) and it scores against that.
    """
    counts, terms = _fit_vectorizer(texts)
    vocab = set(terms.tolist())

    names = list(topic_words)
    truncated = [list(topic_words[n])[:n_terms] for n in names]
    scores = _npmi(counts, terms, truncated)

    out = {}
    for name, words, npmi in zip(names, truncated, scores):
        n_scored = sum(1 for w in words if w in vocab)
        out[name] = {
            "npmi": None if pd.isna(npmi) else float(npmi),
            "n_scored": n_scored,
            "n_terms": len(words),
            "scored_fraction": n_scored / len(words) if words else 0.0,
            "terms": words,
        }
    return out


def classify_discovered(coherence: dict[str, dict]) -> dict[str, dict]:
    """Apply the pre-registered gate to the DISCOVERED topics only.

    Anchored issues are deliberately absent from the result: they are scored (for
    the caveat table) but never classified, because a coherence threshold is
    invalid for them. See the module docstring's asymmetry note before adding
    them here.
    """
    out = {}
    for name, rec in coherence.items():
        if not is_discovered(name):
            continue
        npmi = rec["npmi"]
        fails_floor = npmi is None or npmi <= COHERENCE_FLOOR
        fails_coverage = rec["scored_fraction"] < MIN_SCORED_FRACTION
        reasons = []
        if fails_floor:
            reasons.append(
                f"npmi {npmi if npmi is None else round(npmi, 4)} "
                f"<= floor {COHERENCE_FLOOR}"
            )
        if fails_coverage:
            reasons.append(
                f"only {rec['n_scored']}/{rec['n_terms']} top terms are content "
                f"words (< {MIN_SCORED_FRACTION:.0%}); topic is defined by "
                f"register artifacts"
            )
        out[name] = {
            **rec,
            "status": "noise" if reasons else "coherent",
            "noise_reasons": reasons,
        }
    return out


# ---------------------------------------------------------------------------
# Names file
# ---------------------------------------------------------------------------

class UnanchoredTopicError(ValueError):
    """A surfaced topic has no anchor terms, so a site build would KeyError."""


def load_names(path=NAMES_PATH) -> dict:
    """Read the names file; `{}`-shaped default if it has not been written yet.

    Never raises on a missing file, but the two degraded paths differ and the
    difference matters:

    * a topic present in `surfaced` but missing a `display` value renders under
      its raw column name (`Discovered 5`) — genuinely graceful;
    * a missing FILE yields `surfaced == []`, which DROPS every discovered topic
      from the display list rather than renaming it. That silently removes the
      live "Security & peace" page instead of degrading it. It is a safe failure
      (no crash, no wrong labels) but it is not a lossless one.
    """
    if not path.exists():
        return {"topics": {}, "surfaced": []}
    return json.loads(path.read_text())


def display_name(name: str, names: dict | None = None) -> str:
    """Human-facing label for a topic column, falling back to the column itself.

    The fallback is load-bearing: a topic that is missing from the names file
    (newly discovered, or a hand-edited file) renders as `Discovered 6` rather
    than crashing a page build.
    """
    names = load_names() if names is None else names
    rec = names.get("topics", {}).get(name)
    if not rec:
        return name
    return rec.get("display") or name


def surfaced_discovered(names: dict | None = None) -> list[str]:
    """Discovered COLUMN names cleared for display, in issues.py order.

    Returns column keys (`Discovered 5`), never display labels — callers index
    DataFrames with these. Pair with `display_name` at render time.
    """
    names = load_names() if names is None else names
    return list(names.get("surfaced", []))


def discovered_labels(names: dict | None = None) -> dict[str, str]:
    """`{column -> display label}` for every NAMED discovered topic.

    Covers named-but-not-surfaced topics too, so a caller that already holds a
    column name (e.g. a persisted issue card) still renders a real label. Noise
    topics have no display name and are absent, which makes `.get(name, name)`
    fall back to the column name for them.
    """
    names = load_names() if names is None else names
    return {
        col: rec["display"]
        for col, rec in names.get("topics", {}).items()
        if rec.get("display")
    }


def display_issues(issue_names: list[str], names: dict | None = None) -> list[str]:
    """The 15 anchored issues plus every surfaceable discovered topic.

    This replaces the `issue_names + ["Discovered 5"]` literal that was copied
    into six call sites across five modules. Still column names, so it remains a
    valid DataFrame selector.
    """
    return list(issue_names) + surfaced_discovered(names)


def validate_surfaced(surfaced: list[str]) -> None:
    """Raise unless every surfaced topic has anchor terms to render from.

    `issues_site.py` and `profiles.py` both index an `anchors_all` dict built
    from `issues.ISSUE_ANCHORS | profiles._EXTRA_ANCHORS`, so surfacing a topic
    without anchors raises a bare `KeyError` from deep inside a page build. The
    names file is designed to be hand-edited, which makes that a likely mistake;
    this turns it into a named error naming the fix.

    Enforced on the WRITE path (`build_names`) rather than on read: raising at
    read time would convert a hand-edit typo into a hard site-build failure, and
    `display_issues`'s job is to degrade rather than crash.

    The import is local — `profiles` imports this module.
    """
    from .issues import ISSUE_ANCHORS
    from .profiles import _EXTRA_ANCHORS

    anchored = set(ISSUE_ANCHORS) | set(_EXTRA_ANCHORS)
    missing = [t for t in surfaced if t not in anchored]
    if missing:
        raise UnanchoredTopicError(
            f"surfaced topics with no anchor terms: {missing}. Add them to "
            f"profiles._EXTRA_ANCHORS (and give them a stance spec) before "
            f"surfacing, or a site build will KeyError in issues_site."
        )


def write_names(
    topics: dict[str, dict],
    surfaced: list[str],
    provenance: dict,
    path=NAMES_PATH,
    anchored_coherence_caveat: dict[str, dict] | None = None,
) -> dict:
    """Write the provenance-stamped names file.

    `anchored_coherence_caveat` carries the anchored issues' scores, which are
    published as a caveat table and never gate anything (see the module
    docstring's asymmetry note). It is omitted from the payload entirely when
    absent rather than written as a null.
    """
    payload = {"version": 1, "provenance": provenance,
               "topics": topics, "surfaced": surfaced}
    if anchored_coherence_caveat is not None:
        payload["anchored_coherence_caveat"] = anchored_coherence_caveat
    path.write_text(json.dumps(payload, indent=2))
    return payload


def score_corex_topics(texts: pd.Series | None = None) -> dict[str, dict]:
    """Score every CorEx topic in `issues_meta.json` against the real corpus.

    Reads the persisted meta rather than refitting: `issues.build_issues()` fits
    a full CorEx model over 36k x 25k and must never be called just to read back
    its top terms.
    """
    meta = json.loads(ISSUES_META_PATH.read_text())
    if texts is None:
        texts = pd.read_parquet(PARAGRAPHS_PATH)["text"]
    coherence = compute_coherence(meta["topic_words"], texts)
    expected = len(meta["issues"]) + N_FREE_TOPICS
    if len(coherence) != expected:
        raise ValueError(
            f"scored {len(coherence)} topics but issues_meta.json declares "
            f"{expected} ({len(meta['issues'])} anchored + {N_FREE_TOPICS} free)"
        )
    return coherence


# ---------------------------------------------------------------------------
# In-session authored names for the discovered topics
# ---------------------------------------------------------------------------
# Authored by the implementing model from each topic's top c-TF-IDF terms and its
# NPMI score. NO Anthropic API call was made to produce these (pipeline scope
# decision D1) — a paid batch pass for seven short labels would be pure waste.
#
# `surface` is a SEPARATE, EDITORIAL decision from `status`, and deliberately so.
# `status` is the pre-registered quantitative gate and nothing else. `surface`
# asks the different question "is this a policy issue a reader should see
# alongside the curated 15?" — a topic can be perfectly coherent and still be a
# register artifact rather than an issue ("Debate & Interview Exchanges" is a
# real, tight cluster of transcript genre, not something a president cared
# about). Keeping the two apart stops editorial judgment from being laundered
# through a number.
AUTHORED_NAMES = {
    "Discovered 1": {
        "display": "Fiscal Administration & Appropriations",
        "surface": False,
        "rationale": "Treasury/appropriations housekeeping of 19th-c. annual "
                     "messages. Coherent, but it duplicates Taxes & budget and "
                     "reads as procedural register, not a distinct issue.",
    },
    "Discovered 2": {
        "display": "Executive Departments & Official Reports",
        "surface": False,
        "rationale": "Cabinet-report boilerplate (secretary/department/session). "
                     "Coherent administrative genre, not a policy issue.",
    },
    "Discovered 3": {
        "display": None,
        "surface": False,
        "rationale": "NOISE by the pre-registered gate (NPMI below the "
                     "independence floor). Contraction fragments (ve/don/ll) "
                     "with no thematic core. Deliberately left unnamed.",
    },
    "Discovered 4": {
        "display": "Debate & Interview Exchanges",
        "surface": False,
        "rationale": "Televised-debate transcript genre (mr/senator/governor/"
                     "kennedy/sir). Coherent — those terms genuinely co-occur — "
                     "but it is a medium artifact, not an issue. The clearest "
                     "case for why `status` and `surface` must stay separate.",
    },
    # The name below is load-bearing and MUST NOT change: it is taxonomy.
    # SECURITY_PEACE, a key of crosswalk_v1.json, and it slugifies to the live
    # page docs/issues/security-and-peace.html.
    "Discovered 5": {
        "display": "Security & peace",
        "surface": True,
        "rationale": "A genuine substantive theme the curated 15 never named: "
                     "collective security, nuclear arms, alliances. Highest NPMI "
                     "of the discovered set. Already surfaced site-wide.",
    },
    "Discovered 6": {
        "display": "Territory, Sovereignty & Constitutional Powers",
        "surface": False,
        "rationale": "Early-republic territorial expansion and federal-power "
                     "disputes (constitution/territory/powers/mexico/spain). "
                     "Substantively real and arguably a missing issue — but "
                     "surfacing it needs an anchor-term list "
                     "(issues_site.py indexes anchors_all[name]) and would add a "
                     "page to the live site, which is beyond this task's "
                     "display-name-wiring boundary. Flagged as a candidate.",
    },
    "Discovered 7": {
        "display": "Law, Duty & Public Obligation",
        "surface": False,
        "rationale": "Generic legal/constitutional register (public/shall/duty/"
                     "laws). Coherent but too diffuse to be an issue.",
    },
}


def build_names(texts: pd.Series | None = None, path=NAMES_PATH) -> dict:
    """Score, classify, name, and persist — the full names-file pipeline.

    A noise topic never receives a display name and is never surfaced, no matter
    what `AUTHORED_NAMES` says, so the pre-registered gate cannot be overridden
    by editorial judgment.
    """
    coherence = score_corex_topics(texts)
    classified = classify_discovered(coherence)

    topics: dict[str, dict] = {}
    surfaced: list[str] = []
    for name, rec in classified.items():
        authored = AUTHORED_NAMES.get(name, {})
        is_noise = rec["status"] == "noise"
        display = None if is_noise else authored.get("display")
        surface = bool(authored.get("surface")) and not is_noise
        topics[name] = {
            "display": display,
            "status": rec["status"],
            "surface": surface,
            "npmi": rec["npmi"],
            "n_scored": rec["n_scored"],
            "n_terms": rec["n_terms"],
            "terms": rec["terms"],
            "noise_reasons": rec["noise_reasons"],
            "rationale": authored.get("rationale", ""),
        }
        if surface:
            surfaced.append(name)

    # Anchored issues: scored for the caveat table, never classified, never
    # gated. Recorded separately so no consumer can mistake them for candidates.
    anchored = {
        name: {"npmi": rec["npmi"], "n_scored": rec["n_scored"],
               "n_terms": rec["n_terms"], "terms": rec["terms"]}
        for name, rec in coherence.items() if not is_discovered(name)
    }

    validate_surfaced(surfaced)

    prov = default_provenance(
        "top c-TF-IDF terms per CorEx topic from data/issues_meta.json, scored "
        "with embed_topics._npmi over embed_topics._fit_vectorizer's shared "
        "paragraph vocabulary (the same yardstick as "
        "data/paragraph_clusters_meta.json)"
    )
    # Ordered by the trailing integer, NOT lexicographically: sorted() would put
    # "Discovered 10" before "Discovered 2" the moment N_FREE_TOPICS reaches 10,
    # silently reordering every display list built from this file.
    surfaced.sort(key=lambda n: int(n.rsplit(" ", 1)[1]))
    return write_names(topics, surfaced, prov, path,
                       anchored_coherence_caveat=anchored)


def default_provenance(evidence: str) -> dict:
    """Provenance stamp for a names file authored in-session (no paid API call)."""
    return {
        "authored_by": "claude-opus-4-8",
        "method": "in-session naming by the implementing model; NO Anthropic API "
                  "call was made (see pipeline scope decision D1)",
        "date": date.today().isoformat(),
        "evidence": evidence,
        "gate": {
            "coherence_floor": COHERENCE_FLOOR,
            "min_scored_fraction": MIN_SCORED_FRACTION,
            "pre_registered": True,
            "applies_to": "discovered topics only — anchored issues are exempt "
                          "by design (see embed_topics.py COHERENCE CAVEAT)",
        },
    }
