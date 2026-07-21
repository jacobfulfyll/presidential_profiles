"""v1 rubrics + JSON Schemas for the two annotation passes.

Two passes, because masking and speech-typing want opposite things:

* **paragraph-judgment** (MASKED, unit=paragraph): per-paragraph multi-label
  topic assignment on the FROZEN taxonomy (``taxonomy_v1.json`` level-2 names),
  three binary combativeness flags, a proposal-vs-values class, and entities
  (name + type + stance). The model sees the paragraphs and the speech's DECADE
  only — never the president's name, party, or the speech title. So the obscure
  middle of the corpus is judged on its words, not its author's reputation.
* **speech-factual** (UNMASKED, unit=speech): per-speech speech_type / audience
  / medium, from the title + year + the first few paragraphs. The title is the
  single best signal for speech typing; masking it would make a factual field
  worse for no bias benefit.

Design invariants (see the IMPLEMENT brief and the annotation-provenance layer):

* Topic labels are structurally constrained to the frozen level-2 names via an
  ``enum`` in the schema — invalid labels are *impossible*, not merely rejected
  (the ``annotation-pass-label-validation`` requirement). The names/definitions
  are loaded from ``taxonomy_v1.json`` at import; nothing is hardcoded.
* Output is terse — flags as booleans, topics as an array of enum strings, no
  rationale fields. Output tokens scale over 36,229 paragraphs, so every field
  that isn't consumed downstream is money spent for nothing.
* Combativeness is judged against the speech's own era, not 2026 norms; the
  rubric anchors that with real corpus passages drawn from six periods (each
  passage is cited by its real ``(doc_name, para_idx)`` so a reviewer can pull
  it back up).

``taxonomy_v1.json`` is a frozen paid artifact — this module only READS it.
"""

from __future__ import annotations

import json

from ..corpus import DATA_DIR

PROMPT_VERSION = "v1"

# --------------------------------------------------------------------------
# frozen taxonomy — the label space (never hardcoded here)
# --------------------------------------------------------------------------

_TAXONOMY_PATH = DATA_DIR / "llm_annotations" / "taxonomy_v1.json"


def _load_level2() -> list[dict]:
    data = json.loads(_TAXONOMY_PATH.read_text())
    level2 = data["level2"]
    names = [t["name"] for t in level2]
    if len(names) != len(set(names)):
        raise ValueError("taxonomy_v1 level-2 names are not unique — schema enum would be ambiguous")
    return level2


LEVEL2: list[dict] = _load_level2()
TOPIC_NAMES: list[str] = [t["name"] for t in LEVEL2]

# --------------------------------------------------------------------------
# closed enums for the judgment pass (booleans elsewhere)
# --------------------------------------------------------------------------

PROPOSAL_VALUES_CLASSES = ["proposal", "values", "mixed", "neither"]
ENTITY_TYPES = ["person", "group", "nation", "institution", "other"]
ENTITY_STANCES = ["adversarial", "favorable", "neutral"]

# --------------------------------------------------------------------------
# closed enums for the factual pass (each carries "other" + "uncertain")
# --------------------------------------------------------------------------

SPEECH_TYPES = [
    "inaugural_address",
    "state_of_the_union_or_annual_message",
    "special_message_to_congress",
    "proclamation",
    "veto_or_signing_statement",
    "campaign_or_debate",
    "eulogy_or_commemoration",
    "press_conference_or_interview",
    "public_remarks_or_address",
    "other",
    "uncertain",
]
AUDIENCES = [
    "congress",
    "general_public",
    "press",
    "foreign_or_diplomatic",
    "military",
    "specific_organization_or_group",
    "other",
    "uncertain",
]
MEDIUMS = [
    "written_message",
    "spoken_address",
    "broadcast_radio_or_tv",
    "debate",
    "press_conference",
    "other",
    "uncertain",
]

# --------------------------------------------------------------------------
# JSON Schemas (structured outputs: additionalProperties:false + explicit
# required everywhere; NO minItems/maxItems/minLength/minimum — Sonnet-5
# structured outputs reject those constraints)
# --------------------------------------------------------------------------

_ENTITY_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string", "enum": ENTITY_TYPES},
        "stance": {"type": "string", "enum": ENTITY_STANCES},
    },
    "required": ["name", "type", "stance"],
    "additionalProperties": False,
}

_JUDGMENT_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "para_idx": {"type": "integer"},
        # empty array == "none" (no level-2 topic applies)
        "topics": {"type": "array", "items": {"type": "string", "enum": TOPIC_NAMES}},
        "party_attack": {"type": "boolean"},
        "enemy_naming": {"type": "boolean"},
        "zero_sum": {"type": "boolean"},
        "proposal_values": {"type": "string", "enum": PROPOSAL_VALUES_CLASSES},
        "entities": {"type": "array", "items": _ENTITY_ITEM_SCHEMA},
    },
    "required": [
        "para_idx", "topics", "party_attack", "enemy_naming",
        "zero_sum", "proposal_values", "entities",
    ],
    "additionalProperties": False,
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "annotations": {"type": "array", "items": _JUDGMENT_ITEM_SCHEMA},
    },
    "required": ["annotations"],
    "additionalProperties": False,
}

FACTUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "speech_type": {"type": "string", "enum": SPEECH_TYPES},
        "audience": {"type": "string", "enum": AUDIENCES},
        "medium": {"type": "string", "enum": MEDIUMS},
    },
    "required": ["speech_type", "audience", "medium"],
    "additionalProperties": False,
}

# The nested judgment field ingest explodes into paragraph_entities.parquet.
JUDGMENT_ENTITY_FIELD = "entities"

# --------------------------------------------------------------------------
# per-request sizing
# --------------------------------------------------------------------------

# max_tokens per judgment request. The pilot MEASURED ~112 output tokens per
# paragraph (see JUDGMENT_EST_OUTPUT_TOKENS_PER_PARA), so the original 80/para
# slope would truncate any speech past ~62 paragraphs once the count contract
# makes the model emit all N items — and a truncated structured array is
# incomplete JSON that would strand the batch. 130/para gives headroom over the
# measured 112. The largest speech (Lincoln, Peoria 1854 — 295 paragraphs)
# lands at 2000 + 130*295 = 40,350, comfortably under Sonnet 5's 128K output
# ceiling. max_tokens is a ceiling only — it does not change billed cost (actual
# output tokens do), nor the submit-gating estimate (which uses the per-paragraph
# output estimate, not max_tokens).
def judgment_max_tokens(n_paragraphs: int) -> int:
    return 2000 + 130 * n_paragraphs


# MEASURED from the v1 pilot (run pilot-v1-20260720, cached results.jsonl): the
# 16 non-degenerate judgment responses emitted 47,014 output tokens over 419
# paragraphs = 112.2 tok/paragraph — far above the pre-pilot 40 guess (the
# terse-JSON serialization guess ignored full topic arrays + per-paragraph
# entity objects). Rounded up conservatively to 115 so the offline estimate is
# an upper bound. The 3 degenerate responses (1-2 items then end_turn) are
# excluded because the effort="medium" + count-contract fixes target exactly
# that failure mode; a well-producing run emits at the ~112 rate.
JUDGMENT_EST_OUTPUT_TOKENS_PER_PARA = 115

# How many opening paragraphs the (unmasked) factual pass sees.
FACTUAL_INTRO_PARAS = 5

# --------------------------------------------------------------------------
# per-speech context builders (the masking policy lives here)
# --------------------------------------------------------------------------


def judgment_context(speech, paras) -> str:
    """MASKED body: the speech's decade + every paragraph tagged with the
    para_idx the model must echo. No president, party, or title.

    Works over a WHOLE speech or a single CHUNK of one (the caller passes the
    subset of paragraphs for this request), so the count contract is phrased over
    "the paragraphs below" rather than "this speech" — a chunk request must return
    exactly its own paragraphs, not the speech's.

    Carries an explicit COUNT CONTRACT (opening and closing) because the pilot
    showed the model sometimes emitting 1-2 items then stopping: the exact number
    of paragraphs and the exact para_idx list are stated so the array length is
    unambiguous."""
    decade = int(speech.decade)
    ordered = paras.sort_values("para_idx")
    idxs = [int(i) for i in ordered["para_idx"]]
    n = len(idxs)
    idx_list = ", ".join(str(i) for i in idxs)
    tagged = "\n\n".join(f"[para_idx={int(r.para_idx)}]\n{r.text}" for r in ordered.itertuples())
    return (
        f"SPEECH ERA: the {decade}s. The speaker's name, party, and the speech "
        f"title are withheld. Judge every paragraph only from the words below, "
        f"against the norms of the {decade}s — not today's.\n\n"
        f"COUNT CONTRACT: you are given {n} paragraph(s) below, with para_idx "
        f"values [{idx_list}]. Return EXACTLY {n} annotation object(s) — exactly "
        f"one per para_idx listed, no more, no fewer, and no duplicates.\n\n"
        f"PARAGRAPHS:\n\n{tagged}\n\n"
        f"END OF PARAGRAPHS. Remember: the `annotations` array must contain "
        f"exactly {n} object(s), one for each of para_idx [{idx_list}]."
    )


def factual_context(speech, paras) -> str:
    """UNMASKED body: title + year + the first few paragraphs. The title is the
    strongest signal for speech typing, so it is deliberately shown."""
    intro = paras.sort_values("para_idx").head(FACTUAL_INTRO_PARAS)
    body = "\n\n".join(r.text for r in intro.itertuples())
    return (
        f"TITLE: {speech.title}\n"
        f"YEAR: {int(speech.year)}\n\n"
        f"OPENING PARAGRAPHS (first {FACTUAL_INTRO_PARAS} of the speech):\n\n{body}"
    )


# --------------------------------------------------------------------------
# rubrics (cached system blocks)
# --------------------------------------------------------------------------


def _topic_catalog() -> str:
    """Render the frozen level-2 topics grouped by their level-1 domain, with a
    compact era tag so the model doesn't apply an anachronistic label."""
    by_l1: dict[str, list[dict]] = {}
    for t in LEVEL2:
        by_l1.setdefault(t["level1"], []).append(t)
    lines: list[str] = []
    for l1, topics in by_l1.items():
        lines.append(f"### {l1}")
        for t in topics:
            birth, death = t.get("era_of_birth"), t.get("era_of_death")
            era = f" [{birth or '?'}–{death or 'present'}]" if (birth or death) else ""
            lines.append(f'- "{t["name"]}"{era}: {t["definition"]}')
    return "\n".join(lines)


# Era-anchored few-shots. Each PASSAGE is verbatim real corpus text, cited by
# its real (doc_name, para_idx) so a reviewer can pull it back up. They span six
# periods and jointly exercise every flag, the proposal/values classes, and the
# entity stance field.
_JUDGMENT_FEWSHOTS = r"""
EXAMPLE A — 1780s  (april-30-1789-first-inaugural-address, para_idx 4)
PARAGRAPH: "You will join with me I trust in thinking, that there are none under the influence of which, the proceedings of a new and free Government can more auspiciously commence."
ANNOTATION: {"para_idx": 4, "topics": ["Presidential Humility & Reflection on Office"], "party_attack": false, "enemy_naming": false, "zero_sum": false, "proposal_values": "values", "entities": []}
WHY: an ideals/reflection statement with no policy ask and no adversary.

EXAMPLE B — 1860s  (december-3-1860-fourth-annual-message, para_idx 15)
PARAGRAPH: "Such a doctrine, from its intrinsic unsoundness, can not long influence any considerable portion of our people, much less can it afford a good reason for a dissolution of the Union."
ANNOTATION: {"para_idx": 15, "topics": ["Constitutional Union & Federalism"], "party_attack": false, "enemy_naming": false, "zero_sum": false, "proposal_values": "values", "entities": []}
WHY: argues a constitutional principle. It attacks a "doctrine," not a named actor, so enemy_naming is false.

EXAMPLE C — 1900s  (december-3-1900-fourth-annual-message, para_idx 35)
PARAGRAPH: "For the real culprits, the evil counselors who have misled the Imperial judgment and diverted the sovereign authority to their own guilty ends, full expiation becomes imperative within the rational limits of retributive Justice."
ANNOTATION: {"para_idx": 35, "topics": ["Treaties, Diplomacy & International Arbitration"], "party_attack": false, "enemy_naming": true, "zero_sum": false, "proposal_values": "values", "entities": [{"name": "the evil counselors", "type": "group", "stance": "adversarial"}, {"name": "China", "type": "nation", "stance": "neutral"}]}
WHY: names a specific foreign adversary ("the evil counselors") as culprits -> enemy_naming true. Combative even by 1900 diplomatic norms. It is a foreign, not partisan, target, so party_attack is false.

EXAMPLE D — 1930s  (october-28-1932-campaign-speech-indianapolis-indiana, para_idx 7)
PARAGRAPH: "The Governor dismisses the agreements brought about between the leaders of industry and labor under my assistance less than 1 month after the crash by which wages of literally millions of men and women were, for the first time in 15 depressions of a century, held without reduction until after profits had ceased and the cost of living had decreased."
ANNOTATION: {"para_idx": 7, "topics": ["Labor, Wages & Working Conditions"], "party_attack": true, "enemy_naming": true, "zero_sum": false, "proposal_values": "neither", "entities": [{"name": "the Governor", "type": "person", "stance": "adversarial"}]}
WHY: attacks a partisan opponent (referred to as "the Governor") and imputes fault -> party_attack AND enemy_naming true, even though the speaker is not named to you. It is a rebuttal, not a policy ask or an ideals statement -> proposal_values "neither".

EXAMPLE E — 1950s  (january-7-1954-state-union-address, para_idx 45)
PARAGRAPH: "I recommend enactment of legislation to strengthen agricultural conservation and upstream flood prevention work, and to achieve a better balance with major flood control structures in the down-stream areas."
ANNOTATION: {"para_idx": 45, "topics": ["Conservation & the Environment", "Agriculture & Farm Prices"], "party_attack": false, "enemy_naming": false, "zero_sum": false, "proposal_values": "proposal", "entities": []}
WHY: a concrete legislative ask ("I recommend enactment of legislation ...") -> proposal. "farmers" is a generic group, not a named actor, so no entity.

EXAMPLE F — 2010s  (january-12-2016-2016-state-union-address, para_idx 6)
PARAGRAPH: "But such progress is not inevitable. It's the result of choices we make together. And we face such choices right now. Will we respond to the changes of our time with fear, turning inward as a nation, turning against each other as a people? Or will we face the future with confidence in who we are, in what we stand for, in the incredible things that we can do together?"
ANNOTATION: {"para_idx": 6, "topics": ["Providence, Faith & American Ideals"], "party_attack": false, "enemy_naming": false, "zero_sum": false, "proposal_values": "values", "entities": []}
WHY: it MENTIONS division but argues AGAINST "turning against each other," so its own framing is unity, not us-vs-them -> zero_sum false. No named target -> enemy_naming false.
""".strip()


JUDGMENT_RUBRIC = f"""\
You are a careful annotator of United States presidential rhetoric. You are given
several consecutive paragraphs from a single speech, and the DECADE in which the
speech was delivered. The speaker's name, party, and the speech title are
withheld on purpose: judge each paragraph on its own words, not on any guess
about who wrote it.

ERA-ANCHORING (important): judge combativeness against the norms of the speech's
OWN era, not against 2026 norms. Formal 19th-century invective and plain-spoken
21st-century attacks can both be "combative" for their time.

For EACH paragraph return exactly one annotation object with these fields:

1. topics: an array of zero or more topic names, chosen ONLY from the closed
   list below (exact strings). A paragraph may carry several topics or none.
   If no listed topic applies, return an empty array — do NOT invent a label and
   do NOT force a poor fit. Respect each topic's era tag: do not apply a topic to
   a paragraph from well before that topic existed.

2. party_attack (boolean): TRUE only if the paragraph explicitly criticizes,
   blames, or disparages a political PARTY, a partisan faction, an opposing
   administration, or a named/clearly-referenced partisan opponent AS SUCH, and
   imputes fault, failure, or bad motive to them. A neutral statement of policy
   disagreement is NOT enough. Falsifiable test: can you quote the words that
   name or unambiguously point at a partisan actor and impute fault? If there is
   no such target, FALSE.

3. enemy_naming (boolean): TRUE only if the paragraph identifies a SPECIFIC
   adversary — a person, group, nation, or institution, named or unambiguously
   referenced (e.g. "the Governor", "the trusts", "the evil counselors",
   "Iran") — and frames them as an opponent, threat, or wrongdoer. Test: is
   there at least one entity below you would mark stance="adversarial"? If the
   paragraph attacks no identifiable target, FALSE. (This cross-checks the
   entities field: enemy_naming=true with no adversarial entity is a
   contradiction to avoid.)

4. zero_sum (boolean): TRUE only if the paragraph's OWN framing is that one
   side can gain only at another's expense — a contest of winners vs losers,
   us-vs-them, "they will take what is ours", a fight that must be won. Merely
   MENTIONING conflict or division is not enough; arguing AGAINST division, or
   describing mutual benefit and shared progress, is FALSE.

5. proposal_values: exactly one of:
   - "proposal": primarily advances or urges a concrete policy, program, law,
     appropriation, or course of action ("I recommend...", "Congress should
     enact...", "we will build...").
   - "values": primarily articulates principles, ideals, national identity,
     faith, gratitude, or civic virtue, without a concrete policy ask.
   - "mixed": substantial elements of BOTH a concrete ask and a values appeal.
   - "neither": narrative, procedural, ceremonial, factual reportage, or an
     argument/rebuttal that is neither a policy ask nor an ideals statement.

6. entities: an array of the SPECIFIC named entities the paragraph refers to as
   actors. For each: name (as written), type (one of {ENTITY_TYPES}), and stance
   (one of {ENTITY_STANCES}) — how the paragraph frames them. Include only
   named or unambiguously referenced actors; skip generic collectives like "the
   people", "Americans", or "farmers" unless the paragraph frames them as a
   specific actor. Return an empty array if there are none. Keep names short.

Be terse. Do not add fields, rationales, or commentary — only the object above.

============================ TOPIC LIST (closed) ============================
Choose topic strings EXACTLY as written (including punctuation). The bracketed
range is the era the topic belongs to.

{_topic_catalog()}

============================ WORKED EXAMPLES ============================
Real corpus passages across six eras, with the annotation each should receive:

{_JUDGMENT_FEWSHOTS}
"""


JUDGMENT_INSTRUCTION = (
    "Annotate EVERY paragraph below — the `annotations` array must contain "
    "exactly one object per para_idx shown (the count contract states how many), "
    "with no paragraph skipped and none duplicated. Echo each paragraph's "
    "para_idx exactly as given, and judge combativeness against the paragraph's "
    "own era."
)


FACTUAL_RUBRIC = f"""\
You classify a United States presidential speech by three factual attributes,
from its title, its year, and its opening paragraphs. Unlike the judgment pass,
this pass is UNMASKED: the title is shown because it is the single best signal
for speech typing (e.g. "First Inaugural Address", "Fourth Annual Message").

Return exactly one object with these fields, each a single value from its
closed list:

speech_type — the kind of speech. One of:
  {SPEECH_TYPES}
  Era note: a 19th/early-20th-century written "Annual Message to Congress" and a
  modern spoken "State of the Union" are both
  "state_of_the_union_or_annual_message". Use "special_message_to_congress" for
  a message to Congress on a single subject. Use "campaign_or_debate" for
  stump speeches and debates. Use "uncertain" only when the opening genuinely
  does not let you tell; use "other" for a real kind not in the list.

audience — the primary intended audience. One of:
  {AUDIENCES}
  A message addressed to Congress is "congress"; a broadcast address to the
  nation is "general_public"; a message to another government is
  "foreign_or_diplomatic".

medium — how it was delivered. One of:
  {MEDIUMS}
  A written message transmitted to Congress is "written_message"; a speech read
  aloud is "spoken_address"; a radio/TV address is "broadcast_radio_or_tv".
  Era note: before broadcasting, annual messages were typically written and read
  by a clerk — prefer "written_message" unless the text signals live delivery.

Be terse: return only the three fields. Prefer a specific value over "other"/
"uncertain" when the evidence supports it.
"""


FACTUAL_INSTRUCTION = (
    "Classify this speech's type, audience, and medium from the title, year, and "
    "opening paragraphs below. Return a single object with the three fields."
)
