# Word-family map audit — completeness critique

I have what I need. Writing the critique.

---

# Completeness critique: what this audit still missed

I ran corpus counts to keep this concrete; numbers below are from the 1,057-speech corpus.

## PRIORITY 1 — A confirmed override pair is internally contradictory (fix before anything ships)

**security / secure / secured / securing / secures.** Two confirmed findings, produced in different slices, directly contradict each other:

- The A-slice (false-merge) finding says **FORCE_SPLIT** `security` away from `secure/securing/secures`, on the grounds that the verb forms mean "obtain / bring about" (a different, 19th-c. concept) and inject a spurious 1800s "security" plateau.
- A B-slice (missed-merge) finding says **FORCE_MERGE** `secured` *into* a family it explicitly lists as `{security, secure, securing, securities, secures, secured}` — i.e. it re-lists `security` together with the exact verbs the other finding wants split off.

Applied naively, the FORCE_MERGE undoes the FORCE_SPLIT. This is an artifact of partitioning the audit into independent lenses that never reconciled a shared word. **The correct resolution neither override expresses:** `secure + secured + securing + secures` as one verb node, `security` (+ possibly `securities`, itself a lexicalized financial plural — see the A_00 note) kept separate. **Action:** sweep the 96 findings for every word that appears in both an A finding and a B finding and reconcile them by hand. `security` is the one I can see; there may be others hidden across slice boundaries (candidates to check: `dominate/domination`, `accumulate`, `advocate`, `illustrate` all appear in both an A "cleared" note and a B FORCE_MERGE — those happen to agree, but they prove the overlap exists).

## PRIORITY 2 — Confirm the override list is actually wired into the build

Two A-slice auditors independently flagged that `credit + creditable` is **merged at cos=0.799 in the map they reviewed**, even though the project's own task doc names that exact pair at that exact score as the canonical must-split. Either the hand-curated override list was never applied to this build, or this map predates it. If true, then *none* of the "settled" decisions (credit/creditable split, state/states merge, taxes fix) are reflected in the live map either, and layering these 96 new overrides on top of an unpinned base will produce inconsistent results. **Action:** before acting on any finding, verify (a) the pre-existing override list is applied, (b) the map is rebuilt after overrides, and (c) whether several of the 96 are already pinned. This is a go/no-go check.

## PRIORITY 3 — An entire error class is structurally invisible to BOTH lenses: irregular / suppletive pairs

Both A (nodes with a cosine merge) and B (fractured *stem groups*) only ever see words that **share a Snowball stem**. Irregular morphology stems differently, so these pairs are never candidates and were never eligible to be flagged as a missed merge. They are high-frequency and split across two lines right now:

| form | n | form | n |
|---|---|---|---|
| man | 2,029 | men | 3,455 |
| person | 797 | people | 13,475 |
| child | 601 | children | 1,721 |
| woman | 250 | women | 1,087 |
| life | 2,396 | lives | 1,257 |
| foot | 87 | feet | 128 |

`person`/`people` (14k combined) and `man`/`men` (5.5k) are among the most-charted words a user could type, and each currently returns half the data. The B_00 auditor noticed `life/lives` is "unreachable by the current candidate generator" but treated it as a one-off; it is a whole class. **Action:** these cannot come from the stem gate — they need an explicit irregular-pair override list (man/men, woman/women, child/children, person/people, life/lives, foot/feet, plus verbs: go/went/gone, buy/bought, think/thought, bring/brought, seek/sought, be/is/are/was/were/been). Nobody was asked to build it.

## PRIORITY 4 — Era-correlated spelling / hyphenation variants (the dangerous kind — invisible AND time-skewed)

The stem gate never proposes these because the hyphen/spelling changes the token, and they are **100% front- or back-loaded in time**, so the split silently amputates one era from the trend line:

| 19th-c. form | n | modern form | n |
|---|---|---|---|
| to-day | 125 | today | 2,769 |
| to-morrow | 14 | tomorrow | 312 |
| to-night | 8 | tonight | 1,594 |
| co-operation | 33 | cooperation | 732 |
| co-ordinate* | 2 | coordinate* | 183 |

A user charting `cooperation` loses 33 pre-1920 hits; `today` loses 125 pre-1930 hits. Because the missing mass is entirely in one era, this is exactly the "plausible-looking but false trend" failure the brief exists to prevent — here as a false *absence* in early data. **Action:** a normalization/override pass for hyphenated archaic forms (also check `war-time`, `to-wit`, `any one/anyone`, `some one/someone`, and British spellings like `defence`, `labour`, `honour`, `connexion` if present).

## PRIORITY 5 — Case-folding collisions were found but never swept systematically

B_02 caught `AIDS`(97, disease) folding into `aids`(43, help), and A_05 caught `PATRIOT` Act in `patriot`. Nobody ran the general sweep. Confirmed live collisions (case-sensitive counts):

- **SALT** 48 (arms-limitation treaties, 1970s–80s) folds into **salt** 15 → a "salt" line that is 76% Cold War arms control.
- **START** 20 (arms treaties, 1990s–2010s) folds into **start** 497 → small but perfectly era-correlated contamination.
- **AIDS** 97 vs **aids** 43 (already noted).

**Action:** case-sensitive frequency scan for any all-caps token (≥ ~10 hits) that also exists lowercased, and blocklist/split the lexicalized-acronym sense. This is cheap and mechanical.

## PRIORITY 6 — The BIGRAM map has failure modes a unigram audit cannot surface

The 35,417 bigrams are grouped by applying the per-word map to each half. Concrete consequences nobody audited:

1. **Every unigram error is multiplied, not copied.** One bad unigram merge corrupts every phrase containing either form. The confirmed `tear`/`tears` false merge (destruction vs. grief) means "tear down", "tear gas", "in tears" are all now grouped through one contaminated token; `waged`→`wage` propagates into every "waged war" / "wage war" / "wage increase" phrase. **A single bad override on the unigram side is far more expensive on the bigram side** — which raises the stakes on Priority 1 and on any questionable override (Priority 8).

2. **Unresolved build-order question — do overrides even propagate?** If the bigram map was materialized once from the pre-override unigram map, then all 96 fixes (and the pre-existing override list) must be *re-derived into ~35k phrases*, and none of that has been checked. If bigrams are rebuilt live from the unigram map, fine — but nobody stated which, and it is the difference between the phrase explorer being correct or being frozen at the pre-override state. **This must be answered explicitly.**

3. **Missed unigram merges fracture phrases too.** Until `recommend`/`recommendation`, `work`/`worked`, `immigration`/`immigrants` etc. are merged, every bigram containing the stranded form is also a separate phrase line ("recommended legislation" vs "recommend legislation"). The 96 unigram fixes have a large, uncounted phrase-level payoff — and a large phrase-level risk if any is wrong.

4. **Prepositional-participle contamination is amplified.** `according`/`accorded` and `owing`/`owe` (confirmed floor leaks) surface on the bigram side as high-frequency phrases "according to", "owing to" — function-word phrases that swamp the substantive verb. If bigrams aren't stop-word filtered, the top of the phrase map is "of the / to the / in the" noise; if they are, the participle-preposition phrases dominate their nodes.

5. **Bigrams could *disambiguate* head polysemy the unigram map can't — and the audit never proposed using them.** "White House" (48% of the `white` node's mass) is a distinct bigram from "white men"; "New Deal"/"Square Deal" are distinct from "good deal". The phrase layer naturally separates senses that are hopelessly fused in the unigram line. The audit treated head polysemy (`white`, `deal`, `means`, `thought`) as unfixable, but the bigram data is the fix for the *unigram* chart's caveat. Nobody connected these. **Action:** at minimum, verify override propagation to bigrams; ideally, use bigram context to flag or annotate polysemous unigram heads.

## PRIORITY 7 — The 3-lens refutation structurally disfavors the most dangerous findings

I can't see the killed findings, but the surviving-finding metadata plus the auditors' "considered and did not flag" notes show the pattern: the hardest errors to confirm are **archaic sense-drift** cases (`touch`=concerning, `calculated`=apt-to, `instant`=current-month, `according`, `owing`, `specie`) that require a corpus concordance to see and look like innocent inflections to anyone judging from the surface form. A refuting lens that does *not* re-run the corpus check will wave these through as "same word" — so the refutation vote is biased *against* precisely the era-correlated false merges the project most wants caught. Only the auditors who explicitly grepped the corpus found this class. **Action:** any finding in the archaic-drift class should not be refutable by intuition alone; require the refuter to cite corpus counts. Re-examine near-threshold sense-drift pairs that a single refuter killed.

Separately, two whole populations were **punted, not resolved**: the `-ly` adverb class (dozens: `mere/merely` got filed but `efficiently`, `particularly`, `entirely` did not) and the deverbal `-ment/-tion/-ance` nominalizations (`enforcement` 0.679, `achievement` 0.617, `procurement` 0.311, many scoring *negative*). The map currently renders these **inconsistently** (`perfectly` merged, `absolutely` not; `peacefully` in, `generally` out). This is a live inconsistency, not a clean deferral — a user gets an arbitrary answer depending on which word they type. **Action:** the owner must make one policy call and apply it uniformly; right now the chart's behavior is undefined for hundreds of common words.

## PRIORITY 8 — Confirmed overrides that may themselves be wrong

- **FORCE_MERGE `criminal + criminals`.** The auditor filed this with an explicit caveat: `criminal` is heavily adjectival and system-oriented ("criminal justice", "criminal law", "criminal code") while `criminals` denotes people. This is closer to the `patient`/`patients` homograph they *split* than to the `immigrant`/`immigration` case they cite as precedent. Merging risks inflating a "criminals" (people) line with "criminal justice" (institution) usage. **Re-examine before applying** — and note the bigram multiplier makes a wrong call here expensive ("criminal justice", "criminal aliens", "career criminals" all inherit it).
- **FORCE_SPLIT `convicted` from `conviction/convictions`.** The auditor admitted the head is itself polysemous (belief vs. legal) and that the split "does not fully purify" — it strands the one unambiguously-legal form (n=60) as a tiny orphan line while leaving the head mixed. Low value, possible bad override; confirm it buys more than it costs.
- **FORCE_MERGE `fort + forts`.** A large share of `fort` is proper-name components (Fort Sumter, Fort Pickens) clustered in the 1860s. The auditor argues "still forts," which is defensible, but the merged line carries an era-correlated Civil-War bump that a user will read as a surge in talk about forts generally. Borderline; flag for a second eye.

## Two structural things nobody owns

- **Label-selection is unaudited.** "Node label = most frequent surface form" means a contaminated node can be *named* after its contaminating sense (a node that is 48% "White House" could surface as `white`). No lens checked whether labels mislead.
- **The `government`/`govern` consistency principle was never applied as a sweep.** Auditors flagged `assembly/assemble`, `establishment/establish`, `organization/organize`, `representatives/represent` as the same act-vs-institution shape but left each alone individually. Either the principle is a general rule (then sweep for it) or it's a one-off (then say so) — currently it's applied ad hoc.

**Bottom line on ranking:** fix the `security` contradiction and confirm overrides are actually applied+propagated to bigrams (P1–P2, P6.2) before touching anything else — those determine whether the other 94 fixes even land correctly. Then build the two override lists the stem gate can never generate (irregular pairs, hyphenated era-variants — P3–P4), which are pure missing coverage of high-frequency, time-skewed words. The acronym sweep (P5) is cheap. The adverb/nominalization policy (P7) and the three questionable overrides (P8) need the owner, not another automated pass.