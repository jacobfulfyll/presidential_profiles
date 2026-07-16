"""Tests for the word-family map (word_families.py).

The map groups inflected and derived surface forms into one node so the word
explorer can chart "immigration" / "immigrants" / "immigrant" as a single line.
The design's hard constraint is asymmetric: a MISSED merge is a visible nuisance
(two lines instead of one), but a FALSE merge is invisible — it renders as a
plausible trend line a reader could misinterpret. So the load-bearing tests here
are the guard rails: words that must never be fused.

These run against the built data/word_families.json (the shipped artifact), and
against pure helpers that need no corpus. The full rebuild (a ~3-minute spaCy
pass) is not exercised here; it is covered by the build's own print/validation.
"""

import json

import pytest

from presidential_profiles import word_families as wf

FAMILIES = (
    json.loads(wf.FAMILIES_PATH.read_text())["families"]
    if wf.FAMILIES_PATH.exists() else None
)
needs_map = pytest.mark.skipif(FAMILIES is None,
                               reason="word_families.json not built")


def _same(a: str, b: str) -> bool:
    return a in FAMILIES and b in FAMILIES and FAMILIES[a] == FAMILIES[b]


# --- Guard rails: these must NEVER share a node -----------------------------
# Each is a homograph or a stem-collision of two genuinely different words. A
# regression here means the threshold or floor-rescue was loosened without
# thought; it must fail loudly.
MUST_SPLIT = [
    ("will", "willing"),          # the future auxiliary vs the adjective
    ("states", "stated"),         # "United States" vs "the President stated"
    ("state", "stated"),          # floor-rescue must not bridge these
    ("united", "unit"),
    ("government", "govern"),      # the institution vs the act (owner ruling)
    ("being", "beings"),
    ("according", "accord"),      # "according to" (particle) vs the verb
    ("plain", "plains"),
    ("good", "goods"),
    ("credit", "creditable"),     # financial vs praiseworthy (cos 0.799)
    ("security", "secure"),       # safety vs 19thc "obtain"
    ("wage", "waged"),            # pay vs war (spikes at the wars)
    ("conviction", "convicted"),  # belief vs legal
    ("patient", "patients"),      # forbearing vs medical
    ("tear", "tears"),            # rip vs weeping
    ("marshal", "marshall"),      # officer vs Marshall Plan
    # cross-stem bridge rejections (adversarially refuted, 2026-07-16)
    ("democracy", "democratic"),  # concept vs the Democratic Party
    ("defense", "defenseless"),   # national defense vs the vulnerable
    ("four", "fourth"),           # the number vs the Fourth of July
    ("tax", "taxpayers"),         # the levy vs the people who pay it
    ("means", "meant"),           # financial means vs the verb "to mean"
]

# --- The whole point: these MUST share a node -------------------------------
MUST_MERGE = [
    ("immigration", "immigrants"),   # the motivating case
    ("immigration", "immigrant"),
    ("tariff", "tariffs"),
    ("state", "states"),             # override fixes the PROPN floor hole
    ("tax", "taxing"),               # override fixes cos=0.231
    ("report", "reported"),          # floor-rescue: cross-POS inflection
    ("work", "worked"),
    ("plan", "planned"),
    ("person", "people"),            # irregular, stem-blind
    ("man", "men"),
    ("child", "children"),
    # cross-stem bridge merges (audited 2026-07-16): suffix rewrites the stem
    ("tax", "taxation"),             # the motivating question for this feature
    ("religion", "religious"),
    ("economy", "economic"),
    ("china", "chinese"),            # a "China" chart should include "Chinese"
    ("destroy", "destruction"),
]


@needs_map
@pytest.mark.parametrize("a,b", MUST_SPLIT)
def test_dangerous_words_stay_separate(a, b):
    assert not _same(a, b), f"{a!r} and {b!r} must not be in the same family"


@needs_map
@pytest.mark.parametrize("a,b", MUST_MERGE)
def test_families_group_their_forms(a, b):
    assert _same(a, b), f"{a!r} and {b!r} should be one family"


@needs_map
def test_immigration_family_is_exactly_the_three_forms():
    assert set(wf.family_members("immigrants")) == {
        "immigration", "immigrants", "immigrant"}


@needs_map
def test_labels_are_real_words_not_stems():
    # The node label is the most frequent surface form, never the Snowball stem.
    assert wf.form_to_node("immigrants") == "immigration"
    assert "immigr" not in set(FAMILIES.values())


@needs_map
def test_security_verb_node_excludes_the_safety_noun():
    # The audit's one internal contradiction, resolved: secure/securing/secures/
    # secured are the verb node; security (and securities) split off.
    assert _same("secure", "securing")
    assert not _same("secure", "security")


def test_normalize_folds_archaic_spellings():
    # NORMALIZE runs before tokenizing, so a hyphenated archaic form never
    # survives as its own token (it would otherwise fragment to "to" + "day").
    assert wf.tokenize("To-day and co-operation") == ["today", "and", "cooperation"]
    assert wf.tokenize("national defence") == ["national", "defense"]


def test_form_to_node_is_identity_for_ungrouped_words():
    assert wf.form_to_node("zxqwerty") == "zxqwerty"
