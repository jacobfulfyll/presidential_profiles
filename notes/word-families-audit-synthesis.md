# Word-family map audit — override synthesis

Below is the consolidated deliverable. This is a synthesis of the 96 verified overrides recorded across all 21 audited slices — no new corpus work was needed; every entry already survived 3-lens adversarial verification. I resolved one genuine contradiction in the raw findings (the `security` node — see the note in-line and in the summary).

```python
# ============================================================================
# word_families.py override lists
# 96 verified overrides from an audit of 2,154 merge/split decisions.
# Ordered high severity first; grouped by defect class.
# ============================================================================

FORCE_MERGE: list[set[str]] = [

    # --- HIGH: plural / inflection of a form ALREADY inside the node -------
    # (mechanically indefensible; the singular is already merged elsewhere)
    {"individual", "individually", "individuals"},                 # cos 0.694 - bare sing/plural split
    {"commitment", "commitments", "committed", "commit", "committing"},  # cos 0.631 - plural of a member
    {"employer", "employers", "employers'", "employment"},          # cos 0.645 - plural split; singular already in node
    {"laborer", "laborers", "labor", "labors", "laboring", "labored"},   # cos 0.687 - plural of a member
    {"accomplished", "accomplish", "accomplishment", "accomplishing", "accomplishments"},  # cos 0.63 - plural of a member
    {"criminal", "criminals"},                                      # cos 0.687 - sing/plural, largest split in slice
    {"patriotism", "patriot", "patriotic", "patriots"},             # cos - PROPN floor hole; one concept
    {"importation", "importations"},                                # cos 0.695 - bare sing/plural, siblings merged
    {"manufactures", "manufacture", "manufacturers", "manufacturer", "manufacturing", "manufactured"},  # 3-way fracture of one verb
    {"regulations", "regulation", "regulated", "regulate", "regulating"},  # cos 0.543 - verb paradigm fractured
    {"prosecution", "prosecuted", "prosecutions", "prosecute", "prosecuting"},  # cos 0.224 - paradigm cut in half
    {"management", "manage", "managers", "managed"},                # cos 0.601 - past tense orphaned
    {"participation", "participate", "participating", "participated"},  # cos 0.578 - past tense orphaned
    {"interference", "interfere", "interfering", "interfered"},     # cos 0.697 - near-miss on inflection
    {"distribution", "distributing", "distribute", "distributed"},  # cos 0.688 - past tense orphaned
    {"construction", "constructing", "constructed", "construct"},   # cos 0.649 - keep 'constructive' OUT
    {"developed", "development", "develop", "developing", "developments"},  # cos 0.584 - participle orphaned
    {"discharge", "discharged", "discharging"},                     # cos 0.244 - participle split, embedding artifact

    # --- HIGH: stranded past tense / participle of a verb in the node ------
    {"work", "working", "works", "worked"},                         # cos 0.649 - 'worked' orphaned
    {"opened", "open", "opening", "opens"},                         # cos 0.698 - participle near-miss
    {"planned", "plan", "plans", "planning"},                       # cos 0.693 - past tense near-miss
    {"reported", "report", "reports", "reporting"},                 # cos 0.648 - NOUN/VERB floor miss
    {"supply", "supplies", "supplying", "supplied"},                # cos 0.565 - plain inflection
    {"exports", "export", "exported"},                              # cos 0.677 - load-bearing trade term
    {"funds", "fund", "funding", "funded"},                         # cos 0.681 - budget term fractured
    {"suffering", "suffer", "sufferings", "suffered"},              # cos - past tense orphaned

    # --- HIGH: high-freq singular/plural or spelling variant --------------
    {"state", "states"},                                            # cos 0.605 - PROPN floor hole; keep 'stated' OUT
    {"toward", "towards"},                                          # cos 0.175 - US/UK spelling variant
    {"tax", "taxes", "taxed", "taxing"},                            # cos(tax,taxing)=0.231 - inflection
    {"navy", "navies"},                                             # cos 0.239 - sing/plural, bad vector

    # --- HIGH: verb + its nominalization (the immigration/immigrant case) --
    {"recommend", "recommended", "recommending", "recommends", "recommendation", "recommendations"},  # cos 0.691
    {"emigration", "emigrants"},                                    # cos 0.381 - the motivating case, one letter over
    {"residence", "reside", "residing", "residents", "resident"},   # act/place/people of one concept
    {"consultation", "consult", "consultations", "consulted", "consulting"},  # cos 0.496 - verb + nominalization
    {"exertions", "exert", "exertion", "exerted"},                  # verb paradigm + nominalization
    {"inauguration", "inaugurated", "inaugural", "inaugurate"},     # cos 0.039 - high-salience, embedding artifact
    {"reorganization", "reorganized", "reorganize"},                # cos 0.286 - verb + nominalization
    {"mobilization", "mobilized", "mobilize"},                      # keep 'mobile' OUT - different concept
    {"expiration", "expire", "expired"},                            # verb + nominalization
    {"advocates", "advocate", "advocated"},                         # tense split + agent noun
    {"accumulation", "accumulate", "accumulated", "accumulating"},  # cos 0.698 - threshold artifact
    {"congratulate", "congratulation", "congratulations"},          # sing/plural of noun across nodes
    {"domination", "dominate", "dominated", "dominant"},            # cos 0.68 - threshold artifact
    {"deterrent", "deterrence"},                                    # cos 0.636 - Cold War policy term
    {"illustration", "illustrated", "illustrate"},                  # cos 0.64 - present/past of one verb
    {"industry", "industries", "industrial", "industrialized"},     # cos 0.577 - keep 'industrious' OUT
    {"independence", "independent", "independently"},               # cos 0.653 - noun + adjective
    {"advance", "advances", "advanced", "advancement", "advancing"},  # 3-way verb fracture
    {"liberation", "liberated", "liberate"},                        # cos 0.697 - keep 'liberal' OUT

    # --- HIGH: embedding artifact (cos ~0 or negative on ONE lexeme) -------
    {"excite", "excited", "excitement", "exciting"},                # cos 0.019 - same verb, both agitation sense
    {"embarrass", "embarrassed", "embarrassing", "embarrassment", "embarrassments"},  # cos 0.056
    {"convince", "convinced"},                                      # cos 0.603 - present/past
    {"waste", "wasteful", "wasted"},                                # cos 0.578 - ordinary participle
    {"witness", "witnesses", "witnessed"},                          # cos 0.588 - keep 'wit' OUT
    {"decline", "declining", "declined"},                           # cos 0.656 - tense split of one lemma
    {"prosperity", "prosperous", "prosper"},                        # cos 0.192 - embedding artifact
    {"extravagance", "extravagant"},                                # cos -0.015 - broken vector; fiscal concept
    {"cooperation", "cooperate", "cooperating"},                    # cos 0.243 - keep 'cooperative' OUT

    # --- SECURITY: resolves a conflict in the raw findings (see note) ------
    {"secure", "secured", "securing", "secures"},                   # verb 'obtain/make-safe'; 'security'/'securities' split off below

    # --- MEDIUM: verb + nominalization / adj + adverb ---------------------
    {"hope", "hopes", "hoping", "hoped"},                           # cos 0.56 - 'hoped' orphaned
    {"end", "ending", "ends", "ended"},                             # cos 0.664 - past tense orphaned
    {"result", "results", "resulted", "resulting"},                 # cos 0.613 - participle orphaned
    {"information", "inform", "informed"},                          # cos 0.492 - past tense of a member
    {"study", "studies", "studying", "studied"},                   # cos 0.542 - past tense orphaned
    {"finance", "finances", "financing", "financed"},               # cos 0.561 - past tense orphaned
    {"deposits", "deposit", "deposited"},                           # cos 0.627 - noun/verb inflections
    {"remedy", "remedies", "remedied"},                             # cos 0.347 - past tense of head
    {"perform", "performed", "performing", "performance"},           # cos 0.656 - verb + -ance noun
    {"enact", "enacted", "enacting", "enactment", "enactments"},     # cos 0.601 - verb + -ment noun
    {"mere", "merely"},                                             # cos 0.603 - same lexeme, no drift
    {"previous", "previously"},                                     # cos 0.617 - adj + -ly adverb
    {"oppression", "oppressive", "oppressed"},                      # cos 0.45 - immigration/immigrants pattern
    {"notice", "noticed"},                                          # cos 0.342 - tense split
    {"undertake", "undertaking", "undertakings"},                   # cos 0.536 - verb + gerund/nominalization
    {"vaccine", "vaccines", "vaccinated"},                          # verb form of same concept; COVID-era spike
    {"acquiescence", "acquiesced", "acquiesce"},                    # cos 0.64 - verb + noun
    {"alter", "altered", "alteration"},                             # cos 0.675 - verb + deverbal noun
    {"postponed", "postpone", "postponement"},                      # cos 0.376 - verb + -ment noun
    {"incorporated", "incorporation"},                              # cos 0.651 - verb + nominalization
    {"devastating", "devastation"},                                 # cos 0.685 - participle + nominalization
    {"fort", "forts"},                                             # cos ~0.70 - sing/plural, boundary
    {"conviction", "convictions"},                                  # keep belief sense together; 'convicted' split off below

    # --- LOW: small-n but mechanical -------------------------------------
    {"care", "caring", "cares", "cared"},                           # cos 0.64 - inflections; keep 'careful' OUT
    {"year", "years", "year's", "years'"},                          # cos 0.549 - apostrophe artifact
    {"transportation", "transport", "transporting", "transported"}, # cos 0.627 - pure inflection
]

FORCE_SPLIT: list[set[str]] = [

    # --- HIGH: homograph collision (two unrelated words, one stem) --------
    {"specie", "species"},                                          # money vs kind; era-disjoint false trend
    {"tear", "tears"},                                             # cos 0.824 - rip/destroy vs weeping
    {"patient", "patients"},                                        # cos 0.845 - forbearing ADJ vs medical NOUN
    {"marshall", "marshal", "marshals"},                            # cos 0.717 - Marshall Plan/person vs officer
    {"competency", "compete", "competing"},                         # cos 0.717 - capability vs competition

    # --- HIGH: shared stem, drifted to a different concept ----------------
    {"credit", "creditable"},                                       # cos 0.799 - project's canonical must-split
    {"wages", "wage", "waged"},                                     # isolate 'waged' (warfare); keep wage+wages
    {"security", "secure", "securing", "secures"},                  # isolate 'security' (+'securities') from the verb family

    # --- MEDIUM: sense split / grammaticalized function word --------------
    {"conviction", "convictions", "convicted"},                     # cos 0.776 - belief vs legal 'convicted'
    {"born", "borne"},                                             # birth vs carried/endured
    {"according", "accorded"},                                      # 'according to' preposition vs 'accord/grant'
    {"owing", "owe", "owes"},                                       # isolate 'owing' ('owing to' preposition); keep owe+owes
]
```

Note on the one conflict I resolved: the raw findings contained both a FORCE_SPLIT of `security` from `secure/securing/secures` (verb = "obtain," front-loaded in the 1800s) and a FORCE_MERGE folding the stranded past tense `secured` back into that same verb family. Those are only contradictory on the surface. The coherent resolution, which I encoded above, is: `{secure, secured, securing, secures}` = the verb, one node; `security` (safety/national-security noun) and `securities` (financial instruments, a lexicalized plural) each split off. I dropped the literal six-form merge set as written because it would re-contaminate the line the split exists to clean.

Summary:

- **Measured error rate: 96 confirmed overrides / 2,154 audited decisions = 4.5%.** 160 decisions were flagged on first pass; 40% of flags were refuted under verification, which is the system working as intended (a false accusation writes a permanent bad override).
- **The bail condition is triggered, and not marginally: 96 genuine warts is more than 3x the "~30" threshold.** By the plan's own stated rule, the design is in trouble and should not ship as-is on a hand-override list alone — 96 pins is unmaintainable and will silently rot.
- **But the warts are not random — they collapse into essentially three systematic defects, so the honest read is "one fixable algorithm, not 96 exceptions."** (1) The lemma floor never fires for a verb's own participle/past tense because spaCy tags `reported`/`developed`/`secured` as VERB while the base surface form majority-tags NOUN/ADJ, so the `(lemma, POS)` keys differ and adjudication falls into the 0.55–0.70 cosine dead band; roughly half the FORCE_MERGE list is this one bug. (2) Cosine systematically undervalues `-ment`/`-ion`/`-ance` nominalizations (the exact immigration→immigrants relation the map exists to capture), parking them just under 0.70 or, worse, at negative values (`extravagance` −0.015, `excite/excited` 0.019) where no threshold tuning can reach them. (3) A "plural/inflection of a form already in the node" invariant is violated repeatedly (`individuals`, `commitments`, `laborers`).
- **Recommended fix, in priority order:** add a floor-rescue rule — *a form joins a node if it shares a lemma (POS-ignored) with any member already admitted, not just the head* — which mechanically sweeps defects (1) and (3) without endangering the settled `states/stated`, `united/unit`, or `government/govern` splits (those differ in lemma, not just POS). That alone would retire ~60 of these overrides. Defect (2) and the 12 FORCE_SPLITs (true homographs and sense-drift) are irreducible and must stay as hand overrides regardless of any algorithm change.