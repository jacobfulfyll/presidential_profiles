# Word-family review

7,124 unigrams grouped into 4,423 nodes (threshold 0.7). Overrides and irregulars applied.

Multi-form nodes: 1,696. Marginal cosine merges (0.60-0.80, eyeball these): 223.


## Marginal cosine merges (closest to the line)

- `attainable` -> **attained** (cos 0.602)
- `independently` -> **independence** (cos 0.606)
- `formerly` -> **former** (cos 0.7)
- `speculative` -> **speculation** (cos 0.701)
- `conserve` -> **conservation** (cos 0.702)
- `announcement` -> **announced** (cos 0.703)
- `carefully` -> **careful** (cos 0.703)
- `depreciation` -> **depreciated** (cos 0.703)
- `publicly` -> **public** (cos 0.703)
- `badly` -> **bad** (cos 0.704)
- `subsequently` -> **subsequent** (cos 0.707)
- `elected` -> **election** (cos 0.709)
- `murderers` -> **murder** (cos 0.709)
- `prosperous` -> **prosperity** (cos 0.709)
- `expectation` -> **expect** (cos 0.71)
- `maturity` -> **mature** (cos 0.711)
- `constitutionality` -> **constitution** (cos 0.712)
- `consciousness` -> **conscious** (cos 0.713)
- `determination` -> **determined** (cos 0.713)
- `differ` -> **different** (cos 0.713)
- `declarations` -> **declared** (cos 0.714)
- `engagements` -> **engaged** (cos 0.714)
- `declaration` -> **declared** (cos 0.715)
- `educate` -> **education** (cos 0.715)
- `fruitful` -> **fruits** (cos 0.715)
- `individually` -> **individual** (cos 0.715)
- `opening` -> **open** (cos 0.715)
- `considerably` -> **considerable** (cos 0.719)
- `contributions` -> **contribute** (cos 0.719)
- `fulfilling` -> **fulfill** (cos 0.719)
- `momentous` -> **moment** (cos 0.719)
- `unfortunate` -> **unfortunately** (cos 0.719)
- `valid` -> **validity** (cos 0.72)
- `inaugurated` -> **inauguration** (cos 0.721)
- `anticipations` -> **anticipated** (cos 0.722)
- `realization` -> **realize** (cos 0.722)
- `separation` -> **separate** (cos 0.723)
- `seriously` -> **serious** (cos 0.723)
- `sincerely` -> **sincere** (cos 0.723)
- `intelligent` -> **intelligence** (cos 0.724)
- `promotions` -> **promote** (cos 0.724)
- `savings` -> **save** (cos 0.724)
- `armored` -> **armor** (cos 0.725)
- `completion` -> **complete** (cos 0.725)
- `restrictive` -> **restrictions** (cos 0.726)
- `annually` -> **annual** (cos 0.727)
- `retired` -> **retirement** (cos 0.727)
- `contracting` -> **contract** (cos 0.729)
- `electricity` -> **electric** (cos 0.729)
- `anticipation` -> **anticipated** (cos 0.73)
- `abundance` -> **abundant** (cos 0.731)
- `accumulate` -> **accumulation** (cos 0.731)
- `administrator` -> **administration** (cos 0.731)
- `constitutionally` -> **constitution** (cos 0.732)
- `trading` -> **trade** (cos 0.732)
- `indulgence` -> **indulge** (cos 0.733)
- `discussed` -> **discussion** (cos 0.734)
- `fitting` -> **fit** (cos 0.735)
- `fulfilled` -> **fulfill** (cos 0.735)
- `inevitably` -> **inevitable** (cos 0.735)

## Largest merged families

- **it** (n=42,015): it, it's
- **be** (n=41,881): be, being
- **states** (n=19,023): states, state
- **american** (n=15,829): american, america, americans, america's, americas
- **people** (n=15,629): people, persons, person, peoples, people's
- **government** (n=14,545): government, governments, governmental, government's
- **nation** (n=12,798): nation, nations, national, nation's, nationwide
- **year** (n=11,814): year, years, year's, years'
- **country** (n=10,405): country, countries, country's
- **who** (n=10,194): who, who's
- **there** (n=9,834): there, there's
- **do** (n=9,566): do, doing
- **president** (n=8,891): president, presidency, presidential, presidents, president's
- **one** (n=8,783): one, one's
- **congress** (n=8,782): congress, congressional, congressman, congressmen, congresses
- **other** (n=8,466): other, other's
- **time** (n=8,450): time, times
- **what** (n=7,612): what, what's
- **he** (n=7,450): he, he's
- **world** (n=7,416): world, world's, worldwide
- **make** (n=7,084): make, making, makes
- **law** (n=6,930): law, laws
- **new** (n=6,707): new, newly
- **work** (n=6,505): work, working, works, worked
- **war** (n=6,475): war, wars
- **right** (n=6,373): right, rights, rightly
- **power** (n=6,095): power, powers
- **peace** (n=5,909): peace, peaceful, peacefully, peacetime, peaceable, peaceably
- **going** (n=5,613): going, go
- **your** (n=5,524): your, yours
- **men** (n=5,455): men, man, man's
- **public** (n=5,245): public, publicly
- **know** (n=4,675): know, knows, knowing
- **act** (n=4,646): act, acts, acted
- **interest** (n=4,298): interest, interests
- **want** (n=4,268): want, wants, wanted, wanting
- **need** (n=4,172): need, needs, needed
- **think** (n=4,133): think, thinking, thinks
- **force** (n=4,109): force, forces, forced, forcibly, forcing
- **present** (n=4,026): present, presented, presents, presenting

## Known gaps (deferred)

- Verb irregulars (go/went, buy/bought) not yet merged — need the same per-word corpus check as the noun irregulars.
- Case-collision acronyms beyond SALT/START/AIDS — add to `ACRONYMS` after a case-sensitive frequency scan.
