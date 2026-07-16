"""Word-family map: group inflected and derived forms of a word into one node.

The word explorer indexes exact surface forms, so "immigration", "immigrants",
and "immigrant" are three separate series. This module groups such forms into
family nodes so they can be charted as one line (behind an opt-in toggle in the
explorer). Nothing here calls an API — the whole pipeline is offline and
deterministic.

Pipeline
--------
1. NORMALIZE      archaic/British spellings folded before tokenizing
                  ("to-day" -> "today"), so era-correlated variants don't split.
2. lemma floor    spaCy in-context (lemma, POS) key per surface form. Computed
                  over the real speeches, not a bare word list, so the tagger can
                  tell the noun "state" from the verb "stated".
3. candidates     Snowball stem groups. Two forms are only ever considered for
                  merging if they share a stem, which keeps grouping morphological
                  (so "war" can never merge with "conflict").
4. adjudicate     within a stem group, a form joins a node if it shares a
                  POS-ignored lemma with any admitted member (the "floor-rescue"
                  rule) OR model2vec cosine to the node head >= FAMILY_THRESHOLD.
5. overrides      FORCE_SPLIT isolates and FORCE_MERGE unions win last. These are
                  the residue no threshold can get right (cos(tax,taxing)=0.231
                  wants merge; cos(credit,creditable)=0.799 wants split).
6. stem-blind     IRREGULAR_MERGE (man/men) and the ACRONYMS split, which the
                  stem gate structurally cannot see.

Node labels are the most frequent surface form in the node ("immigration", not
the stem "immigr").

The override lists were produced by a 21-agent adversarial audit of all 2,154
merge/split decisions the algorithm makes (4.5% error rate, 96 confirmed errors,
each refuted by three independent skeptics before it counted). See
notes/word-families-audit-*.{md,json} for the audit and its rationale.
"""

import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .corpus import DATA_DIR, load
from .figures import REPO_ROOT

FAMILIES_PATH = DATA_DIR / "word_families.json"
REVIEW_PATH = REPO_ROOT / "notes" / "word-families-review.md"

WORD_RE = re.compile(r"[a-z']+")
FAMILY_THRESHOLD = 0.70
# Floor-rescue joins a form to a node when it shares the head's lemma but spaCy
# tagged them different parts of speech (so the strict (lemma, POS) floor missed
# it) — e.g. the noun "report" and the verb "reported". Gated by a cosine floor
# because that same lemma-sharing is how homographs sneak in: the noun "state"
# and the verb "stated" both lemmatize to "state", but cos(state, stated)=0.17
# exposes the divergence, while cos(report, reported)=0.65 confirms one word.
# 0.50 sits in the wide, empty gap between the two: every audited homograph
# scores below 0.30, every genuine cross-POS inflection above 0.50.
FLOOR_RESCUE_MIN = 0.50
MIN_UNIGRAM = 30
MIN_BIGRAM = 15
EMBED_MODEL = "minishlab/potion-base-8M"

# --- Spelling / hyphenation normalization (applied before tokenizing) --------
# Archaic hyphenations and British spellings that split a word across eras. The
# current tokenizer strips hyphens, so "to-day" would otherwise fragment into
# "to" + "day"; folding it to "today" keeps the pre-1930 uses on the line.
NORMALIZE = {
    "to-day": "today", "to-morrow": "tomorrow", "to-night": "tonight",
    "co-operation": "cooperation", "co-operate": "cooperate",
    "co-operative": "cooperative", "co-ordinate": "coordinate",
    "co-ordination": "coordination", "co-ordinated": "coordinated",
    "defence": "defense", "offence": "offense",
}
_NORMALIZE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in NORMALIZE) + r")\b")

# --- Case-collision acronyms (detected case-sensitively, before lowercasing) --
# A lexicalized upper-case sense that pollutes a common lower-case word. Handled
# in explorer.py: these occurrences are counted as their own entity series and
# removed from the colliding word, so "salt" isn't 76% Cold-War arms control.
ACRONYMS = {
    "SALT": "SALT (arms limitation)",
    "START": "START (arms treaties)",
    "AIDS": "AIDS",
}

# --- Overrides: the residue the algorithm provably cannot get right ----------
# FORCE_MERGE: forms that must share one node (cosine undervalues these, or the
# lemma floor missed a participle). Deduped; each set is unioned regardless of
# stem. From the 2026-07-14 audit (84 verified sets).
FORCE_MERGE: list[set[str]] = [
    # participle / past-tense orphaned from its own verb (the ~60-of-96 class;
    # floor-rescue retires most of these, the rest are belt-and-suspenders)
    {"work", "working", "works", "worked"},
    {"opened", "open", "opening", "opens"},
    {"planned", "plan", "plans", "planning"},
    {"reported", "report", "reports", "reporting"},
    {"result", "results", "resulted", "resulting"},
    {"end", "ending", "ends", "ended"},
    {"hope", "hopes", "hoping", "hoped"},
    {"study", "studies", "studying", "studied"},
    {"care", "caring", "cares", "cared"},
    {"decline", "declining", "declined"},
    {"notice", "noticed"},
    {"convince", "convinced"},
    {"supply", "supplies", "supplying", "supplied"},
    {"funds", "fund", "funding", "funded"},
    {"exports", "export", "exported"},
    {"deposits", "deposit", "deposited"},
    {"remedy", "remedies", "remedied"},
    {"finance", "finances", "financing", "financed"},
    {"suffering", "suffer", "sufferings", "suffered"},
    # plural / inflection stranded from a form already in the node
    {"individual", "individually", "individuals"},
    {"criminal", "criminals"},
    {"importation", "importations"},
    {"fort", "forts"},
    {"navy", "navies"},
    {"congratulate", "congratulation", "congratulations"},
    {"conviction", "convictions"},
    {"year", "years", "year's"},
    {"deterrent", "deterrence"},
    {"incorporated", "incorporation"},
    # verb + its nominalization (the immigration/immigrant relation cosine misses)
    {"commitment", "commitments", "committed", "commit", "committing"},
    {"employer", "employers", "employment"},
    {"laborer", "laborers", "labor", "labors", "laboring", "labored"},
    {"accomplished", "accomplish", "accomplishment", "accomplishing",
     "accomplishments"},
    {"manufactures", "manufacture", "manufacturers", "manufacturer",
     "manufacturing", "manufactured"},
    {"regulations", "regulation", "regulated", "regulate", "regulating"},
    {"prosecution", "prosecuted", "prosecutions", "prosecute", "prosecuting"},
    {"management", "manage", "managers", "managed"},
    {"participation", "participate", "participating", "participated"},
    {"interference", "interfere", "interfering", "interfered"},
    {"distribution", "distributing", "distribute", "distributed"},
    {"construction", "constructing", "constructed", "construct"},
    {"developed", "development", "develop", "developing", "developments"},
    {"discharge", "discharged", "discharging"},
    {"recommend", "recommended", "recommending", "recommends",
     "recommendation", "recommendations"},
    {"residence", "reside", "residing", "residents", "resident"},
    {"consultation", "consult", "consultations", "consulted", "consulting"},
    {"exertions", "exert", "exertion", "exerted"},
    {"inauguration", "inaugurated", "inaugural", "inaugurate"},
    {"reorganization", "reorganized", "reorganize"},
    {"mobilization", "mobilized", "mobilize"},
    {"expiration", "expire", "expired"},
    {"advocates", "advocate", "advocated"},
    {"accumulation", "accumulate", "accumulated", "accumulating"},
    {"domination", "dominate", "dominated", "dominant"},
    {"illustration", "illustrated", "illustrate"},
    {"industry", "industries", "industrial", "industrialized"},
    {"independence", "independent", "independently"},
    {"advance", "advances", "advanced", "advancement", "advancing"},
    {"liberation", "liberated", "liberate"},
    {"oppression", "oppressive", "oppressed"},
    {"perform", "performed", "performing", "performance"},
    {"enact", "enacted", "enacting", "enactment", "enactments"},
    {"information", "inform", "informed"},
    {"undertake", "undertaking", "undertakings"},
    {"vaccine", "vaccines", "vaccinated"},
    {"acquiescence", "acquiesced", "acquiesce"},
    {"alter", "altered", "alteration"},
    {"postponed", "postpone", "postponement"},
    {"devastating", "devastation"},
    {"transportation", "transport", "transporting", "transported"},
    # embedding artifact: cosine ~0 or negative on one lexeme, no threshold reaches
    {"excite", "excited", "excitement", "exciting"},              # cos 0.019
    {"embarrass", "embarrassed", "embarrassing", "embarrassment",
     "embarrassments"},                                          # cos 0.056
    {"extravagance", "extravagant"},                             # cos -0.015
    {"prosperity", "prosperous", "prosper"},                     # cos 0.192
    {"cooperation", "cooperate", "cooperating"},                 # cos 0.243
    {"waste", "wasteful", "wasted"},
    {"witness", "witnesses", "witnessed"},
    # high-frequency singular/plural / spelling variant
    {"state", "states"},           # cos 0.605 - PROPN floor hole; NOT stated
    {"toward", "towards"},         # cos 0.175 - US/UK spelling
    {"tax", "taxes", "taxed", "taxing"},   # cos(tax,taxing)=0.231
    {"emigration", "emigrants"},   # cos 0.381 - the motivating case, one over
    # adjective / adverb of one lexeme, no sense drift
    {"mere", "merely"},
    {"previous", "previously"},
]

# IRREGULAR_MERGE: families the Snowball stem gate cannot see because the forms
# stem differently (man/men). Verified against the corpus 2026-07-14. Only noun
# irregulars whose plural is unambiguous are included; verb irregulars are
# deferred pending the same per-word corpus check (see review report).
IRREGULAR_MERGE: list[set[str]] = [
    {"man", "men"},
    {"person", "people"},
    {"child", "children"},
    {"woman", "women"},
    {"life", "lives"},
    {"foot", "feet"},
]

# FORCE_SPLIT: forms peeled OUT of their stem group into their own node. Each set
# is severed from its stem-mates and kept together. These are true homographs
# where cosine is high but the senses differ — the invisible, sometimes
# time-correlated false merges the audit exists to catch.
# CROSS_STEM_MERGE: derivational families the Snowball stem gate cannot see
# because the suffix rewrites the stem (tax/taxat, religion/religi), so cosine
# never got to judge them. Found by a prefix+cosine bridge and confirmed by a
# 3-lens adversarial audit (see notes/word-families-audit-bridge.md).
# Each pair names two forms whose whole nodes are unioned.
CROSS_STEM_MERGE: list[set[str]] = [
    # 348 derivational families the Snowball stem gate cannot see because the
    # suffix rewrites the stem (tax/taxat, religion/religi), so cosine never got
    # to judge them. Found by a prefix+cosine bridge (shared prefix >=4 or
    # containment, cos >=0.70), classified, then adversarially verified by 3
    # independent lenses each (semantic / corpus-drift / prefix-trap). 371
    # candidates -> 358 survived (<2 of 3 refutations) -> 10 pulled on the
    # completeness critic (agent-noun & drifted-sense: means/meant,
    # command/commander, combinations/combined, reclaim/reclamation,
    # collected/collector, tax/taxpayers, debt/debtors, ...). See
    # notes/word-families-audit-bridge.md.
    {"absence", "absent"},
    {"accuracy", "accurate"},
    {"acquired", "acquisition"},
    {"add", "addition"},
    {"afghan", "afghanistan"},
    {"africa", "african"},
    {"america", "american"},
    {"anxiety", "anxious"},
    {"argue", "argument"},
    {"arise", "arisen"},
    {"asia", "asian"},
    {"asserted", "assertion"},
    {"associated", "associates"},
    {"assume", "assumption"},
    {"attained", "attainment"},
    {"avoid", "avoidance"},
    {"bank", "bankers"},
    {"belief", "believe"},
    {"big", "bigger"},
    {"big", "biggest"},
    {"bless", "blessings"},
    {"bombers", "bombing"},
    {"brave", "bravery"},
    {"brazil", "brazilian"},
    {"bright", "brighter"},
    {"britain", "british"},
    {"broke", "broken"},
    {"build", "builders"},
    {"build", "buildings"},
    {"build", "built"},
    {"burden", "burdensome"},
    {"bureau", "bureaus"},
    {"canada", "canadian"},
    {"charitable", "charity"},
    {"cheap", "cheaper"},
    {"chile", "chilean"},
    {"china", "chinese"},
    {"chose", "chosen"},
    {"claimants", "claims"},
    {"clean", "cleaner"},
    {"clear", "clearer"},
    {"coin", "coinage"},
    {"colombia", "colombian"},
    {"command", "commanded"},
    {"communism", "communist"},
    {"comparative", "comparison"},
    {"compared", "comparison"},
    {"compassion", "compassionate"},
    {"compete", "competition"},
    {"compete", "competitive"},
    {"compete", "competitors"},
    {"competition", "competitors"},
    {"complain", "complaint"},
    {"conciliation", "conciliatory"},
    {"concluded", "conclusion"},
    {"concur", "concurrence"},
    {"condemnation", "condemned"},
    {"confidence", "confidently"},
    {"confirmation", "confirmed"},
    {"congress", "congressional"},
    {"congress", "congressmen"},
    {"congressional", "congressman"},
    {"coordinate", "coordination"},
    {"correct", "correction"},
    {"create", "creation"},
    {"crime", "criminal"},
    {"crises", "crisis"},
    {"cuba", "cuban"},
    {"dark", "darkest"},
    {"decided", "decision"},
    {"deep", "deeper"},
    {"deep", "deepest"},
    {"deeper", "deepest"},
    {"deeper", "deeply"},
    {"defend", "defenders"},
    {"defend", "defense"},
    {"defined", "definition"},
    {"delegates", "delegation"},
    {"delivered", "delivery"},
    {"demonstrated", "demonstrations"},
    {"depart", "departure"},
    {"department", "departmental"},
    {"depositories", "deposits"},
    {"depositors", "deposits"},
    {"destroy", "destruction"},
    {"detailed", "details"},
    {"deter", "deterrent"},
    {"dictator", "dictatorship"},
    {"differences", "different"},
    {"difficult", "difficulties"},
    {"diplomacy", "diplomatic"},
    {"disagree", "disagreement"},
    {"disarm", "disarmament"},
    {"discovered", "discovery"},
    {"draw", "drawn"},
    {"earliest", "early"},
    {"easier", "easiest"},
    {"east", "eastern"},
    {"economic", "economy"},
    {"efficiency", "efficiently"},
    {"emphasis", "emphasize"},
    {"employed", "employment"},
    {"encourage", "encouragement"},
    {"environment", "environmental"},
    {"erected", "erection"},
    {"europe", "european"},
    {"exclude", "exclusion"},
    {"existence", "existing"},
    {"expand", "expansion"},
    {"experience", "experienced"},
    {"experiment", "experimental"},
    {"explain", "explanation"},
    {"exploration", "explore"},
    {"extended", "extension"},
    {"extremism", "extremists"},
    {"fail", "failure"},
    {"fall", "fallen"},
    {"farm", "farmers"},
    {"fast", "faster"},
    {"fast", "fastest"},
    {"faster", "fastest"},
    {"finance", "financial"},
    {"fish", "fisheries"},
    {"fish", "fishermen"},
    {"forbade", "forbid"},
    {"force", "forcibly"},
    {"forced", "forcibly"},
    {"formation", "formed"},
    {"fort", "fortifications"},
    {"founded", "founders"},
    {"fraud", "fraudulent"},
    {"free", "freely"},
    {"friends", "friendship"},
    {"full", "fully"},
    {"generosity", "generous"},
    {"gentleman", "gentlemen"},
    {"german", "germany"},
    {"government", "governmental"},
    {"graduate", "graduation"},
    {"granted", "grants"},
    {"grateful", "gratitude"},
    {"gratification", "gratifying"},
    {"greece", "greek"},
    {"growing", "grown"},
    {"growing", "growth"},
    {"happily", "happy"},
    {"hard", "harder"},
    {"hard", "hardest"},
    {"harder", "hardest"},
    {"hawaii", "hawaiian"},
    {"heaviest", "heavy"},
    {"heroic", "heroism"},
    {"historic", "historically"},
    {"historic", "history"},
    {"historically", "history"},
    {"honest", "honesty"},
    {"hundred", "hundredth"},
    {"hunger", "hungry"},
    {"idealism", "ideals"},
    {"indebted", "indebtedness"},
    {"induced", "inducement"},
    {"inflation", "inflationary"},
    {"influence", "influential"},
    {"injured", "injury"},
    {"inspection", "inspectors"},
    {"intended", "intention"},
    {"intervene", "intervention"},
    {"invaded", "invasion"},
    {"investment", "investors"},
    {"iran", "iranian"},
    {"iraq", "iraqi"},
    {"israel", "israeli"},
    {"italian", "italy"},
    {"japan", "japanese"},
    {"jealous", "jealousy"},
    {"judicial", "judiciary"},
    {"justification", "justify"},
    {"korea", "korean"},
    {"large", "larger"},
    {"large", "largest"},
    {"leaders", "leadership"},
    {"lebanese", "lebanon"},
    {"legislation", "legislators"},
    {"legislators", "legislature"},
    {"lobby", "lobbyists"},
    {"long", "longest"},
    {"low", "lower"},
    {"low", "lowest"},
    {"lower", "lowest"},
    {"loyal", "loyalty"},
    {"machine", "machinery"},
    {"maintain", "maintenance"},
    {"manifest", "manifestation"},
    {"marriage", "married"},
    {"medical", "medicine"},
    {"meet", "meetings"},
    {"members", "membership"},
    {"mexican", "mexico"},
    {"mistake", "mistaken"},
    {"modification", "modified"},
    {"monthly", "months"},
    {"nation", "nationwide"},
    {"necessaries", "necessity"},
    {"necessary", "necessity"},
    {"new", "newly"},
    {"nicaragua", "nicaraguan"},
    {"nomination", "nominee"},
    {"north", "northern"},
    {"notification", "notified"},
    {"observation", "observed"},
    {"old", "older"},
    {"optimism", "optimistic"},
    {"owned", "owners"},
    {"owned", "ownership"},
    {"owners", "ownership"},
    {"partial", "partly"},
    {"partners", "partnership"},
    {"pay", "payable"},
    {"pay", "payment"},
    {"peace", "peaceable"},
    {"peace", "peacefully"},
    {"peace", "peacetime"},
    {"percent", "percentage"},
    {"political", "politicians"},
    {"poor", "poorest"},
    {"postage", "postal"},
    {"pray", "prayer"},
    {"precisely", "precision"},
    {"preparatory", "prepared"},
    {"present", "presented"},
    {"president", "presidential"},
    {"produce", "producers"},
    {"produce", "production"},
    {"producers", "production"},
    {"prompt", "promptitude"},
    {"provinces", "provincial"},
    {"publication", "published"},
    {"qualifications", "qualified"},
    {"rail", "railroad"},
    {"rail", "railway"},
    {"railroad", "railway"},
    {"rapidity", "rapidly"},
    {"ratification", "ratified"},
    {"rebel", "rebellion"},
    {"rebuild", "rebuilt"},
    {"recognition", "recognize"},
    {"recover", "recovery"},
    {"reduce", "reduction"},
    {"register", "registration"},
    {"regulations", "regulatory"},
    {"religion", "religious"},
    {"repeat", "repeatedly"},
    {"repeat", "repetition"},
    {"representation", "represented"},
    {"respect", "respectfully"},
    {"respond", "response"},
    {"restrain", "restraint"},
    {"revolution", "revolutionary"},
    {"ruin", "ruinous"},
    {"russia", "russian"},
    {"sacrifice", "sacrificed"},
    {"safe", "safer"},
    {"safe", "safest"},
    {"safe", "safety"},
    {"safer", "safety"},
    {"sanitary", "sanitation"},
    {"satisfaction", "satisfied"},
    {"science", "scientific"},
    {"science", "scientists"},
    {"scientific", "scientists"},
    {"secrecy", "secret"},
    {"secure", "secured"},
    {"see", "seen"},
    {"seven", "seventh"},
    {"shipping", "ships"},
    {"short", "shorter"},
    {"show", "shown"},
    {"silence", "silent"},
    {"simple", "simplicity"},
    {"six", "sixth"},
    {"slave", "slavery"},
    {"slow", "slowly"},
    {"small", "smaller"},
    {"small", "smallest"},
    {"smaller", "smallest"},
    {"south", "southern"},
    {"sovereign", "sovereignty"},
    {"speed", "speedy"},
    {"stability", "stable"},
    {"statesman", "statesmanship"},
    {"strategic", "strategy"},
    {"strict", "strictest"},
    {"strong", "stronger"},
    {"strong", "strongest"},
    {"stronger", "strongest"},
    {"submission", "submitted"},
    {"succeed", "success"},
    {"sufferers", "suffering"},
    {"supervision", "supervisors"},
    {"suspended", "suspension"},
    {"sympathize", "sympathy"},
    {"syria", "syrian"},
    {"tax", "taxation"},
    {"teach", "teachers"},
    {"tech", "technology"},
    {"temporarily", "temporary"},
    {"temptation", "tempted"},
    {"terror", "terrorists"},
    {"testify", "testimony"},
    {"threat", "threatened"},
    {"throw", "thrown"},
    {"tough", "tougher"},
    {"tough", "toughest"},
    {"tougher", "toughest"},
    {"traditional", "traditions"},
    {"tragedy", "tragic"},
    {"treated", "treatment"},
    {"tribal", "tribes"},
    {"turkey", "turkish"},
    {"ukraine", "ukrainian"},
    {"uncertain", "uncertainty"},
    {"valuable", "value"},
    {"venezuela", "venezuelan"},
    {"verification", "verify"},
    {"viet", "vietnam"},
    {"viet", "vietnamese"},
    {"vietnam", "vietnamese"},
    {"view", "viewpoint"},
    {"violence", "violent"},
    {"visit", "visitors"},
    {"voluntarily", "voluntary"},
    {"watchful", "watching"},
    {"weak", "weaken"},
    {"weak", "weaker"},
    {"weaken", "weaker"},
    {"wealth", "wealthy"},
    {"wealthiest", "wealthy"},
    {"west", "western"},
    {"wise", "wisest"},
    {"withdrawal", "withdrew"},
    {"withdrawn", "withdrew"},
    {"world", "worldwide"},
    {"worse", "worst"},
    {"write", "written"},
    {"year", "years'"},
    {"young", "younger"},
    {"yourself", "yourselves"},
    {"zeal", "zealous"},
]

FORCE_SPLIT: list[set[str]] = [
    {"specie"},        # hard money vs "species" (biology)
    {"tears"},         # weeping vs "tear" (rip / tear down)
    {"patients"},      # medical vs "patient" (forbearing)
    {"marshall"},      # Marshall Plan / person vs "marshal" (officer)
    {"competency"},    # capability vs "compete" / "competing"
    {"creditable"},    # praiseworthy vs "credit" (financial) - cos 0.799
    {"waged"},         # war (spikes at the wars) vs "wage" / "wages" (pay)
    {"security"},      # safety vs "secure" / "securing" (19thc "obtain")
    {"securities"},    # financial instruments vs the "secure" verb node
    {"convicted"},     # legal vs "conviction" / "convictions" (belief)
    {"borne"},         # carried vs "born" (birth)
    {"according"},     # "according to" (particle) vs "accord" / "accorded"
    {"owing"},         # "owing to" (particle) vs "owe" / "owes" (debt)
]


def tokenize(text: str) -> list[str]:
    """Lowercase, fold archaic spellings, then split into word tokens. The one
    tokenizer the explorer and trends should share so their counts agree with
    the family map."""
    t = _NORMALIZE_RE.sub(lambda m: NORMALIZE[m.group(0)], text.lower())
    return WORD_RE.findall(t)


def _nlp():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "senter"])
    nlp.max_length = 2_000_000
    return nlp


def _pos_keys(texts) -> dict[str, tuple[str, str]]:
    """Majority (lemma, POS) key for each surface form, tagged in context."""
    nlp = _nlp()
    keys: dict[str, Counter] = defaultdict(Counter)
    for doc in nlp.pipe(texts, batch_size=8):
        for tok in doc:
            if tok.is_alpha:
                keys[tok.text.lower()][(tok.lemma_.lower(), tok.pos_)] += 1
    return {w: c.most_common(1)[0][0] for w, c in keys.items()}


def build_families(df: pd.DataFrame | None = None, force: bool = False) -> dict:
    """Build (or load) the word-family map. Returns {surface_form: node_label}.

    Writes data/word_families.json and notes/word-families-review.md. The spaCy
    pass over the whole corpus is the slow step (~3 min), so the result is cached
    and only rebuilt with force=True or when the JSON is missing."""
    global _CACHE
    if FAMILIES_PATH.exists() and not force:
        _CACHE = json.loads(FAMILIES_PATH.read_text())["families"]
        return _CACHE

    from model2vec import StaticModel
    import snowballstemmer

    if df is None:
        df = load()

    # Vocabulary: kept unigrams, plus every word appearing in a kept bigram (a
    # bigram at 15 uses can contain a word below the 30-use unigram cutoff).
    uni, bi = Counter(), Counter()
    for text in df["transcript"]:
        toks = tokenize(text)
        uni.update(toks)
        bi.update(" ".join(p) for p in zip(toks, toks[1:]))
    keep_uni = {w for w, n in uni.items() if n >= MIN_UNIGRAM}
    keep_bi = {b for b, n in bi.items() if n >= MIN_BIGRAM}
    vocab = sorted(keep_uni | {w for b in keep_bi for w in b.split()})

    raw_keys = _pos_keys(df["transcript"].tolist())
    poskey = {w: raw_keys.get(w, (w, "X")) for w in vocab}
    lemma_only = {w: poskey[w][0] for w in vocab}

    st = snowballstemmer.stemmer("english")
    stem_groups: dict[str, list[str]] = defaultdict(list)
    for w in vocab:
        stem_groups[st.stemWord(w)].append(w)

    model = StaticModel.from_pretrained(EMBED_MODEL)
    vecs = model.encode(vocab)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ix = {w: i for i, w in enumerate(vocab)}

    def cos(a: str, b: str) -> float:
        return float(vecs[ix[a]] @ vecs[ix[b]])

    # Adjudicate within each stem group: head-anchored cosine + floor-rescue.
    node_of: dict[str, int] = {}
    members: dict[int, list[str]] = {}
    reason: dict[str, str] = {}
    nid = 0
    for stem in sorted(stem_groups):
        ordered = sorted(stem_groups[stem], key=lambda w: (-uni[w], w))
        local: list[list[str]] = []          # each entry: [head, member, ...]
        for w in ordered:
            for grp in local:
                head = grp[0]
                strict = poskey[w] == poskey[head]
                rescue = (lemma_only[w] == lemma_only[head]
                          and cos(w, head) >= FLOOR_RESCUE_MIN)
                if strict or rescue or cos(w, head) >= FAMILY_THRESHOLD:
                    grp.append(w)
                    reason[w] = ("floor" if strict else
                                 "floor-rescue" if rescue else "cosine")
                    break
            else:
                local.append([w])
                reason[w] = "head"
        for grp in local:
            members[nid] = grp
            for m in grp:
                node_of[m] = nid
            nid += 1

    def new_node(forms: list[str]) -> int:
        nonlocal nid
        members[nid] = list(forms)
        for f in forms:
            node_of[f] = nid
        nid += 1
        return nid - 1

    # FORCE_SPLIT: peel each isolate set out of its current node(s).
    for iso in FORCE_SPLIT:
        forms = [f for f in iso if f in node_of]
        if not forms:
            continue
        for f in forms:
            members[node_of[f]].remove(f)
        new_node(forms)
        for f in forms:
            reason[f] = "override-split"

    # FORCE_MERGE + IRREGULAR_MERGE: union each set into one node (any stem).
    # A form that FORCE_SPLIT just isolated is never re-absorbed — otherwise a
    # later merge set that happened to name it would silently undo the split.
    isolated = set().union(*FORCE_SPLIT) if FORCE_SPLIT else set()
    for grp in FORCE_MERGE + IRREGULAR_MERGE + CROSS_STEM_MERGE:
        forms = [f for f in grp if f in node_of and f not in isolated]
        if len(forms) < 2:
            continue
        target = node_of[forms[0]]
        for f in forms[1:]:
            src = node_of[f]
            if src == target:
                continue
            for m in members[src]:
                node_of[m] = target
                members[target].append(m)
                reason[m] = "override-merge"
            members[src] = []

    # Relabel: node -> most frequent surface form.
    families: dict[str, str] = {}
    node_members: dict[str, list[str]] = {}
    for mem in members.values():
        if not mem:
            continue
        label = max(mem, key=lambda w: (uni[w], w))
        node_members[label] = sorted(mem, key=lambda w: (-uni[w], w))
        for m in mem:
            families[m] = label

    n_nodes = len({families[w] for w in keep_uni})
    shrink = 100 * (1 - n_nodes / len(keep_uni)) if keep_uni else 0.0
    payload = {
        "meta": {
            "threshold": FAMILY_THRESHOLD,
            "n_forms": len(vocab),
            "n_unigrams": len(keep_uni),
            "n_nodes": n_nodes,
            "embed_model": EMBED_MODEL,
            "built_from": f"{len(df)} speeches",
        },
        "families": families,
    }
    FAMILIES_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    _write_review(node_members, uni, cos, reason, len(keep_uni), n_nodes)
    print(f"  word_families: {len(keep_uni):,} unigrams -> {n_nodes:,} nodes "
          f"({shrink:.0f}% shrink)")
    _CACHE = families
    return families


def _write_review(node_members, freq, cos, reason, n_uni, n_nodes) -> None:
    """Rank the groups most likely to be wrong so warts are findable."""
    multi = {lab: ms for lab, ms in node_members.items() if len(ms) > 1}

    def head_cos(ms):
        head = ms[0]
        return [(m, round(cos(head, m), 3)) for m in ms[1:] if reason.get(m) == "cosine"]

    # marginal: a cosine-merged member within 0.1 of the line
    marginal = []
    for lab, ms in multi.items():
        for m, c in head_cos(ms):
            if 0.60 <= c < 0.80:
                marginal.append((lab, m, c))
    marginal.sort(key=lambda x: x[2])

    lines = ["# Word-family review\n",
             f"{n_uni:,} unigrams grouped into {n_nodes:,} nodes "
             f"(threshold {FAMILY_THRESHOLD}). Overrides and irregulars applied.\n",
             f"Multi-form nodes: {len(multi):,}. Marginal cosine merges "
             f"(0.60-0.80, eyeball these): {len(marginal):,}.\n",
             "\n## Marginal cosine merges (closest to the line)\n"]
    for lab, m, c in marginal[:60]:
        lines.append(f"- `{m}` -> **{lab}** (cos {c})")
    lines.append("\n## Largest merged families\n")
    for lab, ms in sorted(multi.items(), key=lambda kv: -sum(freq[w] for w in kv[1]))[:40]:
        tot = sum(freq[w] for w in ms)
        lines.append(f"- **{lab}** (n={tot:,}): {', '.join(ms)}")
    lines.append("\n## Known gaps (deferred)\n")
    lines.append("- Verb irregulars (go/went, buy/bought) not yet merged — need "
                 "the same per-word corpus check as the noun irregulars.")
    lines.append("- Case-collision acronyms beyond SALT/START/AIDS — add to "
                 "`ACRONYMS` after a case-sensitive frequency scan.")
    REVIEW_PATH.write_text("\n".join(lines) + "\n")


_CACHE: dict | None = None


def _load() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(FAMILIES_PATH.read_text())["families"] \
            if FAMILIES_PATH.exists() else {}
    return _CACHE


def form_to_node(form: str) -> str:
    """Family label for a surface form (the form itself if ungrouped)."""
    return _load().get(form, form)


def family_members(form: str) -> list[str]:
    """All surface forms in the same family node as `form`."""
    fam = _load()
    node = fam.get(form, form)
    return sorted(f for f, lab in fam.items() if lab == node) or [form]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build the word-family map")
    p.add_argument("--force", action="store_true", help="rebuild from corpus")
    build_families(force=p.parse_args().force)


if __name__ == "__main__":
    main()
