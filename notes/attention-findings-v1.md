# Issue attention over time, v1 — births, deaths, revivals

**Produced by** `src/presidential_profiles/attention.py` (free local compute; **$0**, no API calls).
**Output** `data/attention/topic_lifecycles.parquet` — 201 rows = (2 levels x 3 genre treatments x
50 level-2 topics / 17 level-1 domains). Byte-reproducible on rerun (seed 20260721, 500
speech-clustered bootstrap replicates).
**Inputs** all frozen: `paragraph_annotations` (36,229 paragraphs), `speech_annotations` (1,057
speeches, 100% coverage), `taxonomy_v1.json`, `crosswalk_v1.json`, `paragraph_issues.parquet`,
`paragraphs.parquet`.

Read `attention.py`'s module docstring for the full method. The one-paragraph version: labels are
normalized casefold-to-canonical (8 case variants, 111 of 52,855 raw label assignments, zero
residue), and repeated (paragraph, topic) pairs are then de-duplicated, leaving **52,133**
assignments; denominators are ALL paragraphs including the 404 with no topic; curves are smoothed
with a centered 5-year window over numerator and denominator separately; a topic-year is
*substantive* when the window holds >= 30 paragraphs and >= 3 topic paragraphs and the smoothed
share clears max(0.2%, 10% of the topic's own peak); everything is computed three ways (raw /
SOTU-only / genre-standardized); CIs resample **speeches**, not paragraphs.

### How to read the numbers

* **Rates are percentage points; the parquet stores fractions.** `rise_rate_per_decade` and
  `fall_rate_per_decade` are fractions of all paragraphs per decade. Every rate in this report is
  that fraction x 100, rounded once from the fraction in a single step (rounding to 2dp and then to
  1dp lands some cells an increment high — four of them did in the first draft of this note).
  Example: Slavery's parquet `fall_rate_per_decade` is 0.318299, reported below as 31.8 pts.
* **Shares are percentages of all paragraphs in the slice**, including the 404 that carry no topic.
* **Topic paragraph counts (`n`) are post-de-duplication**, so they are the count of distinct
  paragraphs carrying the topic, not the count of label assignments. The two differ: Chinese
  Immigration & Exclusion is 116 label assignments but **110** paragraphs; Early Naval Wars is 161
  and **160**. Polygamy in the Territories is 61 either way. Mean topics per paragraph is **1.439**
  post-dedup (52,133 / 36,229), against 1.46 measured on the raw label column.
* **Topics are multi-label**, so per-topic shares do not sum to 1 and must never be normalized as
  if they should.

§9 records exactly which numbers in this report were re-derived from the artifact and which were
not.

---

## 1. Ground-truth falsification checks — RESULTS

Six checks are reported below. **Five are encoded in `attention.py`'s `GROUND_TRUTH` constant** and
evaluated by `run_ground_truth`; of those five, **two failed, one is not testable, and two passed**.
The sixth row — the anachronism check — is not in `GROUND_TRUTH`: it is computed by
`anachronism_report` and printed alongside the other five by `print_checks`. A seventh
pre-registered check, the genre check ("does each claimed death survive inside annual messages
alone?"), is answered in §3 rather than here. Thresholds were fixed before the checks ran and were
not moved afterwards.

| Check | Result | Measured |
|---|---|---|
| Indian affairs peaks 1830s–1870s and dies | **FAIL (peak), PASS (death)** | peak **1790** (The founding), share 35.5%; last substantive **1896** |
| Coinage/currency peaks ~1890s free-silver era | **PASS** | peak **1895**, share 19.3%; Gilded Age era share 7.77% (CI 4.82–11.46) |
| Terrorism born ~2001 | **FAIL** | `Iraq, Gulf Wars, the War on Terror & Interventions` first substantive **1982**; peak 2003 |
| Prohibition is a sharp 1920s spike | **NOT TESTABLE** | taxonomy_v1 contains no Prohibition topic |
| Slavery → civil rights classifies as a RENAME | **PASS** | `rename`, successor `Civil Rights, Voting Rights & Discrimination`, r = −0.14 |
| No topic's birth precedes its real-world existence | **PASS on substantive years, FAIL on raw first appearance** | see §1.4 |

### 1.1 FAIL — Indian affairs peaks in the founding era, not the removal era

The death is unambiguous and survives every genre treatment. The **peak year is not**.

| Era | topic paras | era paras | share | 95% CI |
|---|---|---|---|---|
| The founding (1789–1815) | 139 | 927 | **14.99%** | 9.56–21.43 |
| Expansion (1816–1849) | 232 | 3,708 | 6.26% | 4.65–8.06 |
| Civil War & Reconstruction | 84 | 3,881 | 2.16% | 1.41–2.82 |
| The Gilded Age (1878–1900) | 185 | 3,589 | 5.15% | 3.56–6.66 |
| Progressives & Depression | 33 | 4,718 | 0.70% | 0.30–1.18 |
| War & New Deal | 0 | 1,193 | 0.00% | 0.00–0.00 |
| The Cold War | 9 | 9,143 | 0.10% | 0.02–0.19 |
| Post-Cold War | 4 | 4,950 | 0.08% | 0.00–0.20 |
| The present era | 1 | 4,120 | **0.02%** | 0.00–0.08 |

The founding-era CI (9.56–21.43) does **not overlap** the Expansion CI (4.65–8.06), so the era-level
finding is not a small-sample artifact.

**The 1790 peak specifically, however, rests on one document.** 1790 holds 17 topic paragraphs out
of 32 — but **14 of those 17 come from a single speech**, the *Talk to the Chiefs and Counselors of
the Seneca Nation* (Dec 29, 1790). Only 3 come from the two annual messages (2 from the Second, 1
from the First). This is **single-document concentration**, not annual-message enumeration, and it
is a different artifact class — see §7.

The peak year is 1790 under all three genre treatments, but that agreement is **weaker evidence than
it looks** and should not be read as three independent confirmations:

* Under `raw` and `genre_standardized` the 1790 point is dominated by the same Seneca document.
* Under `sotu` the Seneca speech is excluded outright, so the 1790 estimate (28.6%) is built from
  the 5-year window's 18 SOTU topic paragraphs / 63 SOTU paragraphs — of which only 3 topic
  paragraphs fall in 1790 itself. And the SOTU curve is nearly flat across those years: 1790 28.6%,
  1791 27.6%, 1793 26.3%. The "peak" wins by 0.9 points over the runner-up, which is not a peak
  anyone should quote.

**The era-level finding survives the concentration.** Deleting all 14 Seneca paragraphs outright
leaves the founding era at 125/913 = **13.7%**, still above the Expansion CI ceiling of 8.06. The CI
design already prices this in: the bootstrap resamples **speeches**, not paragraphs, precisely
because a single document can carry this much of a thin era.

**The curve is bimodal at era grain** — and the grain has to be named, because the answer changes
with it. At era grain (the grain at which this report's tables and CIs are computed) the only two
local maxima are **The founding at 14.99%** and **The Gilded Age allotment era at 5.15%**; the
Expansion era (6.26%) sits on the slope between them, not on a peak.

**The removal-era 1830s is not a mode at any grain**, which is the part of the pre-registered
expectation that fails hardest. At decade grain the 1830s (6.63%) is *lower* than the 1820s (7.60%)
and the 1810s (7.53%), and the second decade peak is the **1880s allotment decade at 6.56%**, not
the removal decade: 1800s 19.09 → 1810s 7.53 → 1820s 7.60 → 1830s 6.63 → 1840s 4.82 → … → 1870s
4.80 → 1880s 6.56 → 1890s 3.97. At smoothed-year grain the two largest post-founding local maxima
are **1829 (15.54%)** and **1879 (8.14%)**.

The pre-registered expectation appears to have been about absolute volume or about historiographic
salience; measured as *share of presidential paragraphs* the early republic talked about Indian
affairs more than Jackson did. **This is a finding about the expectation, not a bug** — the annotations are correct
(see the exemplars in §3.1), the taxonomy is correct, and the death claim is untouched.

### 1.2 FAIL — "Terrorism" is born in 1982, because the taxonomy has no terrorism topic

`taxonomy_v1` has no standalone Terrorism topic. The nearest is
`Iraq, Gulf Wars, the War on Terror & Interventions`, which bundles four decades of military
intervention. Its measured birth is **1982**, and the pre-2001 mass is real content, not mislabels:

| year | 1980 | 1983 | 1984 | 1988 | 1990 | 1991 | 2001 | 2002 | 2003 | 2004 |
|---|---|---|---|---|---|---|---|---|---|---|
| paragraphs | 22 | 10 | 23 | 17 | 36 | 37 | 27 | 68 | 53 | 44 |
| smoothed share | 2.7% | 2.7% | 3.5% | 8.1% | 9.1% | 10.5% | 26.4% | 31.0% | **34.5%** | 34.1% |

1980 is the Iran hostage rescue; 1983–84 Lebanon and Grenada; 1990–91 the Gulf War. The **9/11
discontinuity is enormous and correctly located** (3.8% → 26.4% between **1998 and 2001**, peaking
2003), but the *birth* year is set by the bundled interventions. The full run-up is 1997 3.7%,
1998 3.8%, 1999 7.9%, 2000 19.1%, 2001 26.4% — the climb starts in 1999, so 3.8% is the **1998**
share, not the 1999 one. The check as written cannot pass against this taxonomy. Actionable: a
level-2 split of terrorism from conventional intervention would make this measurable — logged in
`TASKS.md` under `### Ungroomed` as **`taxonomy-v1-missing-prohibition-and-terrorism`**.

### 1.3 NOT TESTABLE — there is no Prohibition topic, and the mass is real

178 paragraphs corpus-wide use prohibition vocabulary (`prohibition|intoxicating|liquor|Volstead|
Eighteenth Amendment|bootleg|saloon`); 35 of them fall in 1918–1933, i.e. **2.0% of paragraphs in
that window**. They are absorbed almost entirely by `Crime, Insurrection & Federal Law Enforcement`
(26 of 35), with the remaining **9** paragraphs scattered across **9** other topics. (A further 3
topics appear only as co-labels *on* the Crime paragraphs, so 12 distinct non-Crime topics touch
these 35 paragraphs in all.)

That topic does show the era: smoothed share runs ~1.0–1.7% in 1912–14, then **3.1–5.2% from 1917
to 1931**, falling back to ~0.6% by 1935. The plateau is real but **uneven, not steady** — 1922
(3.9%), 1925 (3.5%), 1926 (3.1%) and 1931 (4.2%) all sit below 4.3%, and 1918 and 1920 sit right on
it at 4.29%. So the underlying attention shift is present and correctly timed — but it is a *bumpy
plateau inside a persistent topic*, not a nameable spike, and the pre-registered check cannot be
evaluated. This is a **taxonomy gap**, not an annotation error.

### 1.4 Anachronism check — the threshold is doing exactly the work it exists to do

Measured on **substantive** years, no topic's birth precedes its real-world existence. Measured on
**raw first appearance**, seven topics do, and they are stray single-paragraph mislabels:

| Topic | first raw paragraph | first substantive | gap |
|---|---|---|---|
| Cold War & Great-Power Rivalry | 1847 | 1945 | 98 yrs |
| Great Depression Recovery | 1838 | 1928 | 90 yrs |
| World War II Military Operations | 1861 | 1917 | 56 yrs |
| World War I Mobilization & War Aims | 1864 | 1914 | 50 yrs |
| Middle East & Israel | 1896 | 1945 | 49 yrs |
| Iraq / Gulf Wars / War on Terror | 1966 | 1982 | 16 yrs |
| Nuclear Weapons & Arms Control | 1921 | 1943 | 22 yrs |

(The 1861 "World War II" paragraph is a Lincoln message; the 1838 "Great Depression Recovery"
paragraph is Van Buren on Canadian border unrest.) The seven rows above are a **curated** set — the cases that are historically impossible — not the
seven largest gaps. The full `ANACHRONISM CHECK` block that a rerun prints ranks every topic by gap,
and its top two rows are *not* errors at all: `Partisan Combat` (1805→1930, 125 yrs) and
`Taxes, Budget Deficits & Federal Spending` (1791→1907, 116 yrs) have real early paragraphs that
simply never reach substantive volume until much later. The lesson is the
same either way: **raw first appearance is not a birth year**, and a lifecycle table built on it
would have shipped seven historically impossible claims.

---

## 2. Headline numbers

**The classification rule, stated numerically.** Without this, "died, last substantive 1973" is
uninterpretable. Over the observed record [1789, 2026], **first match wins**
(`attention.py::lifecycle_class`):

* **died** — last substantive year <= 2026 − 40, i.e. **1986 or earlier**;
* else **revived** — a **>= 30-year** gap between consecutive substantive years;
* else **born** — first substantive year >= 1789 + 40, i.e. **1829 or later**;
* else **persistent**.

Precedence matters: a topic that both died and had a gap is reported `died`. "Died" therefore means
*the record has been silent for four decades*, not *the topic is gone forever* — `Constitutional
Union & Federalism` is classed `died` on a last substantive year of 1973.

**Level-2 lifecycle classes (raw treatment):** died 17 · born 14 · persistent 12 · revived 7.

**Level-1 roll-up:** exactly one of the 17 domains dies — **Indian & Tribal Affairs** (peak 1790,
last substantive 1896). Two are born: `Economy, Labor & Social Welfare` (first 1902) and
`Partisan & Media Combat` (first 1930). Four revive (`Civil Rights & Immigration`,
`Law, Crime & Justice`, `Money, Banking & Currency`, `Personal Narrative & Reflection`); the
remaining ten persist.

**Under-powered guard (< 50 paragraphs corpus-wide):** fires for **zero** topics under `raw` and
`genre_standardized` — the thinnest level-2 topic is `Polygamy in the Territories` at 61
paragraphs, exactly as the pre-run staleness check predicted. It fires **exactly once in the whole
201-row table** — level 2, `sotu`, `Chinese Immigration & Exclusion`, which holds only **45**
paragraphs inside annual messages. Nothing fires at level 1 under any treatment. That single firing
is the guard's value: it is implemented and demonstrably non-vacuous on a real slice, rather than
dead code nobody could tell was dead. It is **not** evidence that claimed deaths generally fail the
annual-message check — the result runs the other way (§3), and the one topic the guard catches is
itself one of the two `sotu` exceptions, so the guard and the robustness result are the same fact
seen twice rather than two independent findings.

**Per-era sample sizes** (denominators for every share in this report):

| Era | paragraphs |
|---|---|
| The founding (1789–1815) | 927 |
| Expansion (1816–1849) | 3,708 |
| Civil War & Reconstruction (1850–1877) | 3,881 |
| The Gilded Age (1878–1900) | 3,589 |
| Progressives & Depression (1901–1932) | 4,718 |
| War & New Deal (1933–1945) | 1,193 |
| The Cold War (1946–1988) | 9,143 |
| Post-Cold War (1989–2016) | 4,950 |
| The present era (2017–2026) | 4,120 |

The founding era (927) and War & New Deal (1,193) are thin; their CIs are correspondingly wide and
should be read as such rather than smoothed.

---

## 3. The clearest deaths

Seventeen level-2 topics die. **Fourteen of the seventeen die under all three genre treatments** —
raw, genre-standardized, *and* inside annual messages alone. Taken one treatment at a time, 15 of
the 17 still classify `died` under `sotu` and 16 of 17 under `genre_standardized`. The annual
message's own decline is therefore not doing the work; these are attention deaths, not format
deaths.

Fall rates are **percentage points per decade** (the parquet stores the fraction — see *How to read
the numbers*).

| Topic | first | peak (era) | peak share | last | fall /decade | n | rename class | successor |
|---|---|---|---|---|---|---|---|---|
| World War I Mobilization & War Aims | 1914 | 1919 (Progressives) | 34.4% | 1922 | 97.0 pts | 227 | unresolved | — |
| World War II Military Operations | 1917 | 1942 (War & New Deal) | 61.5% | 1947 | 69.2 | 624 | unresolved | — |
| Great Depression Recovery | 1928 | 1935 (War & New Deal) | 39.9% | 1940 | 65.9 | 412 | unresolved | — |
| Early Naval Wars: Barbary & 1812 | 1800 | 1814 (founding) | 43.0% | 1820 | 64.4 | 160 | unresolved | — |
| Mexican War | 1844 | 1848 (Expansion) | 24.9% | 1851 | 60.8 | 203 | unresolved | — |
| Vietnam War | 1962 | 1968 (Cold War) | 43.3% | 1977 | 41.4 | 1,201 | unresolved | — |
| **Slavery, Emancipation & Sectionalism** | 1849 | 1856 (Civil War) | 42.7% | 1868 | 31.8 | 866 | **rename** | Civil Rights, Voting Rights & Discrimination |
| Reconstruction & Southern Self-Government | 1863 | 1865 (Civil War) | 34.8% | 1878 | 23.3 | 350 | unresolved | — |
| Territorial Organization & Statehood | 1802 | 1900 (Gilded Age) | 17.4% | 1928 | 5.5 | 751 | unresolved | — |
| Coinage, Currency & Specie | 1790 | 1895 (Gilded Age) | 19.3% | 1935 | 3.9 | 569 | unresolved | — |
| Neutral Rights & Maritime Depredations | 1791 | 1810 (founding) | 42.0% | 1919 | 3.3 | 458 | **rename** | World Order, Foreign Aid & Democracy Promotion |
| Relations with Spain, Mexico & Territorial Claims | 1796 | 1845 (Expansion) | 26.5% | 1916 | 3.2 | 832 | **rename** | World Order, Foreign Aid & Democracy Promotion |
| Indian Affairs, Removal & Allotment | 1790 | 1790 (founding) | 35.5% | 1896 | 3.0 | 687 | unresolved | — |
| Polygamy in the Territories | 1857 | 1883 (Gilded Age) | 2.8% | 1892 | 2.7 | 61 | unresolved | — |
| Chinese Immigration & Exclusion | 1873 | 1884 (Gilded Age) | 6.3% | 1907 | 2.3 | 110 | unresolved | — |
| Constitutional Union & Federalism | 1789 | 1862 (Civil War) | 28.0% | 1973 | 2.2 | 1,328 | unresolved | — |
| **Public Lands & Homestead Settlement** | 1800 | 1802 (founding) | 7.1% | 1938 | 0.5 | 438 | **rename** | Conservation & the Environment |

The three that **do not** survive the genre check, and should be quoted with that caveat:

* `Constitutional Union & Federalism` — dies under raw and SOTU, **revives** under
  genre-standardization. The reweighting lifts modern press-conference and campaign paragraphs about
  federalism that the raw corpus mix buries. This topic is also the one whose classification is
  sensitive to a known, deliberately-deferred rule choice in successor detection — see §6.3.
* `Great Depression Recovery` — dies under raw and genre-standardized, **revives** inside annual
  messages (later presidents invoke the Depression in SOTUs).
* `Chinese Immigration & Exclusion` — **under-powered** inside annual messages alone: 110 paragraphs
  corpus-wide but only **45** in SOTUs, below the 50-paragraph floor, so it is not charted rather
  than judged. The death is real in the other two treatments.

### 3.1 Exemplars

Each quote is attributed `year · president, document title (para N)`. `para N` is the paragraph's
`para_idx`; that pair locates the paragraph exactly, and `exemplar_quotes()` returns the underlying
`doc_name`/`para_idx` directly (§8).

**Indian Affairs, Removal & Allotment** — founding-era peak, 1830s removal, Gilded Age allotment,
then silence.

> "In the first place I observe to you, and I request it may sink deep in your minds, that it is my
> desire, and the desire of the United States that all the miseries of the late war should be
> forgotten and buried forever. That in future the United States and the six Nations should be truly
> brothers, promoting each other's prosperity by acts of mutual friendship and justice."
> — 1790 · Washington, *Talk to the Chiefs and Counselors of the Seneca Nation* (Dec 29, 1790), para 1

> "It gives me pleasure to announce to Congress that the benevolent policy of the government,
> steadily pursued for nearly 30 years, in relation to the removal of the Indians beyond the white
> settlements is approaching to a happy consummation."
> — 1830 · Jackson, *Second Annual Message to Congress* (Dec 6, 1830), para 79

> "I indorse the recommendation made by the present Secretary of the Interior, as well as his
> predecessor, that a permanent commission, consisting of three members, one of whom shall be an
> army officer, be created to perform the duties now devolving upon the Commissioner and Assistant
> Commissioner of Indian Affairs."
> — 1896 · Cleveland, *Fourth Annual Message (Second Term)* (Dec 7, 1896), para 91. After this the
> topic falls to 0.10% of Cold War paragraphs and 0.02% today (1 paragraph).

**Coinage, Currency & Specie** — the cleanest single-era death in the corpus. Gilded Age 7.77%
(CI 4.82–11.46), Post-Cold War **0.00%** (CI 0.00–0.00), present era **0.00%**.

> "There are many plans proposed as a remedy for the evil. Before we can find the true remedy we
> must appreciate the real evil. It is not that our currency of every kind is not good, for every
> dollar of it is good; good because the Government's pledge is out to keep it so, and that pledge
> will not be broken."
> — 1897 · McKinley, *First Annual Message* (Dec 6, 1897), para 5

> "…it is to be earnestly hoped that their labors may result in an international agreement which
> will bring about recognition of both gold and silver as money upon such terms…"
> — 1897 · McKinley, *First Annual Message* (Dec 6, 1897), para 60

**Polygamy in the Territories** — the thinnest topic in the taxonomy (61 paragraphs) and a complete
death: 0.75% in the Civil War era, 0.86% in the Gilded Age, **0.00% in every era after 1932**.

> "The fact that adherents of the Mormon Church, which rests upon polygamy as its corner stone, have
> recently been peopling in large numbers Idaho, Arizona, and other of our Western Territories is
> well calculated to excite the liveliest interest and apprehension."
> — 1881 · Arthur, *First Annual Message* (Dec 6, 1881), para 95

**Reconstruction & Southern Self-Government** — 8.63% of Civil War-era paragraphs (CI 5.45–12.83),
0.33% in the Gilded Age, and **exactly zero paragraphs in every era from 1933 onward**.

> "Under existing conditions the negro votes the Republican ticket because he knows his friends are
> of that party. Many a good citizen votes the opposite, not because he agrees with the great
> principles of state which separate parties, but because, generally, he is opposed to negro rule.
> This is a most delusive cry. Treat the negro as a citizen and a voter, as he is and must remain…"
> — 1874 · Grant, *Sixth Annual Message* (Dec 7, 1874), para 64

**Chinese Immigration & Exclusion** — dies as a named topic in 1907 while immigration itself revives
**ninety-two years later**, in 1999, under a different name (see §5, and note that the rename test
does *not* connect them; the handoff windows do not overlap).

> "In pursuance of the concurrent resolution of October 1, 1890, I have proposed to the Governments
> of Mexico and Great Britain to consider a conventional regulation of the passage of Chinese
> laborers across our southern and northern frontiers."
> — 1890 · Benjamin Harrison, *Second Annual Message* (Dec 1, 1890), para 11

---

## 4. The clearest births

Fourteen level-2 topics are born. Ranked by rise rate (percentage points per decade from first
substantive year to peak; the parquet stores the fraction):

| Topic | first | peak | peak share | rise /decade | present-era share (CI) |
|---|---|---|---|---|---|
| Cold War & Great-Power Rivalry | 1945 | 1959 | 34.9% | 20.9 pts | 9.56% (6.76–13.39) |
| Partisan Combat, Press Conferences & Media Attacks | 1930 | 1950 | 39.6% | 15.5 | 11.75% (8.81–15.13) |
| Iraq, Gulf Wars, War on Terror & Interventions | 1982 | 2003 | 34.5% | 14.7 | 5.53% (3.16–8.99) |
| World Order, Foreign Aid & Democracy Promotion | 1915 | 1948 | 28.1% | 7.3 | 4.10% (2.44–6.12) |
| Space Exploration | 1956 | 1960 | 4.5% | 6.2 | 0.68% (0.37–1.03) |
| Labor, Wages & Working Conditions | 1863 | 1935 | 22.0% | 2.7 | 1.50% (0.92–2.23) |
| Taxes, Budget Deficits & Federal Spending | 1907 | 1980 | 19.8% | 2.4 | 5.17% (3.88–6.62) |
| Middle East & Israel | 1945 | 1981 | 9.4% | 2.3 | 3.47% (2.16–5.17) |
| Conservation & the Environment | 1878 | 1906 | 7.3% | 2.2 | 2.14% (1.21–3.11) |
| Nuclear Weapons & Arms Control | 1943 | 1986 | 12.7% | 2.1 | 1.65% (1.03–2.43) |
| Health Care, Medicare & Social Security | 1934 | 2008 | 13.5% | 1.6 | 9.90% (4.91–15.31) |
| Welfare, Poverty & Economic Opportunity | 1920 | 1999 | 13.9% | 1.5 | 1.99% (1.03–3.12) |
| Prosperity, Jobs & the Middle Class | 1890 | 2012 | 15.0% | 1.0 | 7.45% (5.72–9.18) |
| Civil Rights, Voting Rights & Discrimination | 1865 | 1996 | 11.4% | 0.4 | 3.91% (2.18–6.31) |

Two of these are **absolute** births — literally zero paragraphs in every era before the Cold War:
`Space Exploration` (first 148 paragraphs, Cold War era) and
`Iraq, Gulf Wars, the War on Terror & Interventions` (first 94, Cold War era).

Two more are often described that way but are not, and the difference matters:

* `Health Care, Medicare & Social Security` is zero only **through the Gilded Age**. It carries 11
  paragraphs in Progressives & Depression, before its 1934 first substantive year.
* `Cold War & Great-Power Rivalry` is **not** an absolute birth at all. It carries 35 paragraphs
  before 1946 — 1 in Expansion, 16 in Progressives & Depression, 18 in War & New Deal — the
  earliest being the 1847 stray that §1.4 flags as an anachronism. Any claim that it appears from
  nothing contradicts §1.4 of this same report.

**Space Exploration** — the fastest-rising and fastest-fading birth. Cold War era 1.62%
(CI 1.08–2.30), Post-Cold War 0.95%, present era 0.68%. Born 1956, peaked 1960, has been declining
ever since without ever dying.

> "This Nation belongs among the first to explore it, and among the first—if not the first—we shall
> be. We are offering our know-how and our cooperation to the United Nations. Our satellites will
> soon be providing other nations with improved weather observations."
> — 1962 · Kennedy, *State of the Union Address* (Jan 11, 1962), para 42

**Health Care, Medicare & Social Security** — the slowest-burning birth that is still burning:
0.23% of Progressives-era paragraphs, 1.42% War & New Deal, 2.87% Cold War, **10.04% Post-Cold War**,
9.90% today.

> "It is a testimony also to an enlightened public policy, established by Franklin Roosevelt and
> strengthened by every administration since his death. That policy has freed Americans for more
> hopeful and more productive lives. It has relieved their fears of growing old by social security
> and by medical care."
> — 1966 · Johnson, *Remarks on Receiving the National Freedom Award* (Feb 23, 1966), para 8

**Iraq / Gulf Wars / War on Terror** — the sharpest discontinuity anywhere in the corpus.

> "My fellow citizens, for the last nine days, the entire world has seen for itself the state of our
> Union - and it is strong. Tonight, we are a country awakened to danger and called to defend
> freedom. Our grief has turned to anger, and anger to resolution."
> — 2001 · G. W. Bush, *Address on the U.S. Response to the Attacks of September 11* (Sep 20,
> 2001, nine days after the attacks), para 2

**Civil Rights, Voting Rights & Discrimination** — the slowest birth, and the one that carries the
rename finding. Zero paragraphs in the first two eras; 1.80% in the Civil War era; then a **68-year
trough** — 1878 to 1945 (0.81% Gilded Age, 1.04% Progressives, 0.17% War & New Deal) — before 5.33%
in the Cold War.

> "This State, this city, this campus, have stood long for both human rights and human
> enlightenment—and let that forever be true. This Nation is now engaged in a continuing debate
> about the rights of a portion of its citizens."
> — 1963 · Kennedy, *90th Anniversary of Vanderbilt University* (May 18, 1963), para 6

---

## 5. Revivals

Seven topics come back after a gap of 33–73 non-substantive years.

| Topic | first | peak | peak share | longest gap | era of relevance | n |
|---|---|---|---|---|---|---|
| Personal Narrative, Storytelling & Boasting | 1850 | 1950 | 15.2% | 73 yrs | Civil War → present | 580 |
| National Bank & Banking Crises | 1813 | 1838 | 30.3% | 70 yrs | founding → Post-Cold War | 1,056 |
| Trusts, Corporations & Antitrust | 1885 | 1909 | 16.8% | 58 yrs | Gilded Age → present | 567 |
| Civil Service Reform & the Merit System | 1801 | 1889 | 13.6% | 56 yrs | founding → present | 510 |
| Elections & Voting Integrity | 1828 | 1878 | 14.5% | 56 yrs | Expansion → present | 568 |
| Immigration & Border Security | 1873 | **2017** | 15.1% | 43 yrs | Civil War → present | 685 |
| Crime, Insurrection & Federal Law Enforcement | 1790 | 1793 | 29.2% | 33 yrs | founding → present | 1,418 |

**Immigration & Border Security** is the most striking: 1.63% of Progressives-era paragraphs, **zero
in War & New Deal**, 0.35% through the whole Cold War — and then 3.07% Post-Cold War and **8.83%
(CI 6.13–12.31) in the present era**, its all-time peak, in 2017. A 43-year silence followed by the
largest share the topic has ever held.

> "I am glad to inform you that the immigration of paupers and criminals from certain of the Cantons
> of Switzerland has substantially ceased and is no longer sanctioned by the authorities."
> — 1882 · Arthur, *Second Annual Message* (Dec 4, 1882), para 9

> "…Over the last 24 months, Agent Ortiz and his team have seized more than 200,000 pounds of
> poisonous narcotics, arrested more than 3,000 human smugglers, and rescued more than 2,000
> migrants."
> — 2020 · Trump, *State of the Union Address* (Feb 4, 2020), para 53

**National Bank & Banking Crises** — 13.65% of Expansion-era paragraphs (the Bank War, CI
9.03–19.55), down to 0.59% across the Cold War, back to 1.76% Post-Cold War (2008), and 0.12% today.

> "It is my duty to acquaint you with an arrangement made by the Bank of the United States with a
> portion of the holders of the 3 percent stock, by which the government will be deprived of the use
> of the public funds longer than was anticipated."
> — 1832 · Jackson, *Fourth Annual Message to Congress* (Dec 4, 1832), para 33

> "The financial crisis was ignited when booming housing markets began to decline. As home values
> dropped, many borrowers defaulted on their mortgages, and institutions holding securities backed
> by those mortgages suffered serious losses."
> — 2008 · G. W. Bush, *Speech on Financial Markets and the World Economy* (Nov 13, 2008), para 7

**Elections & Voting Integrity** — 1877's disputed-election settlement and 2020's ballot fights are
the same measured topic, 143 years apart.

> "The fact that two great political parties have in this way settled a dispute in regard to which
> good men differ as to the facts and the law no less than as to the proper course to be pursued in
> solving the question in controversy is an occasion for general rejoicing."
> — 1877 · Hayes, *Inaugural Address* (Mar 5, 1877), para 18 — on the Compromise of 1877

---

## 6. Rename vs death — and the honest limits of the LLM↔CorEx test

Four of seventeen deaths classify as **renames** (a named, in-domain, anti-correlated successor
whose birth window overlaps the dying topic's decline):

| Dying topic | Successor | r | Handoff |
|---|---|---|---|
| Slavery, Emancipation & Sectionalism | Civil Rights, Voting Rights & Discrimination | −0.14 | successor first substantive 1865; slavery's last substantive year 1868 |
| Public Lands & Homestead Settlement | Conservation & the Environment | −0.25 | 1878 vs last 1938 — disposal gives way to conservation |
| Neutral Rights & Maritime Depredations | World Order, Foreign Aid & Democracy Promotion | −0.28 | 1915 vs last 1919 — neutral rights ends at Versailles, Wilsonian internationalism begins |
| Relations with Spain, Mexico & Territorial Claims | World Order, Foreign Aid & Democracy Promotion | −0.42 | 1915 vs last 1916 |

**The verification target passes.** Slavery → civil rights is classified as a rename, not a death.

**The successor correlations are weak, and the rule does not pretend otherwise.** The four winning
correlations span just **−0.14 to −0.42**. The rule is a **sign gate plus a ranking** — a candidate
must clear four structural filters and be negatively correlated, and among the survivors the *most*
anti-correlated wins — and it is deliberately **not** a magnitude threshold. Any magnitude cut would
have had to be chosen after seeing this distribution, which is exactly the post-hoc tuning the
pre-registration exists to prevent. So read `rename` as "the taxonomy names a plausible heir that
passes four independent structural tests", not as "these two curves are strongly anti-correlated".

### 6.1 `true_death` is empty, and that is the finding

**Not one** of the seventeen dying topics has all of its legacy CorEx parents collapse alongside it.

Two different quantities get conflated here, so this section states them separately.

**(a) Closeness to the collapse threshold.** The pre-registered rule calls a curve collapsed when
its decline ratio is <= **0.25**. The lowest CorEx ratio anywhere in the table — across every dying
topic and every one of its parents — is **0.2785**, `Civil rights & race` after slavery dies. It
misses the line by 0.03. **Slavery is genuinely the closest call in the table** on this measure;
the next-closest is Reconstruction's `Civil rights & race` at 0.31.

**That closeness is not fragility, though — and this is worth stating because the arithmetic points
the opposite way from the intuition.** Raising `COREX_COLLAPSE_RATIO` anywhere from 0.25 up to
**0.3089** changes **no** row's `rename_class` (measured by re-running `classify_deaths` at
thresholds 0.26 / 0.2785 / 0.28 / 0.30 / 0.3089 / 0.309 / 0.31 / 0.45 / 0.50). The first flip is at
**0.309**, and it is *Reconstruction*, not Slavery, becoming `true_death`; Coinage follows at 0.45.
Slavery never flips at **any** threshold, because a named successor makes it `rename` and `rename`
outranks `true_death`. The emptiness of `true_death` is **structural, not knife-edge**.

**(b) Divergence between the two labelers** — CorEx retention ÷ LLM retention. This is the different
quantity, and it is the one the LLM↔CorEx comparison is actually for. Across the **15 of 17** dying
topics that have a crosswalk parent (the other 2 have none — see below), every LLM curve retains
**0.37–5.34%** of its peak level while its legacy parent retains **27.8–632.2%**, a divergence of
**18.07x to 560.08x**:

| | Topic | LLM retention | CorEx ratio | divergence |
|---|---|---|---|---|
| least divergent | **Vietnam War** | 0.042 | 0.759 | **18.07x** |
| 5th-least | Slavery, Emancipation & Sectionalism | 0.013 | 0.278 | 21.29x |
| | Indian Affairs, Removal & Allotment | 0.015 | 3.845 | 262.16x |
| most divergent | **Polygamy in the Territories** | 0.004 | 2.229 | **560.08x** |

Slavery is **not** the least divergent case — it is 5th-least, at 21.29x. Vietnam is least divergent
at 18.07x. Slavery is the closest call on (a) and an unremarkable case on (b); the two rankings are
not the same ranking and should not be reported as if they were.

**How to read the reported ratio.** `corex_decline_ratio` is the **max** over a topic's parent
ratios — the *most-persisting* parent — because that is the scalar the "persists if ANY parent
persists" rule turns on, so the column and the verdict can never disagree. This matters when quoting
divergence multiples: Polygamy's 560.08x is `2.229 / 0.00398`, and its numerator is
`Religion & values`, a parent the module itself classifies as **non-tracking**. Per-parent ratios
are preserved in `corex_parent_ratios`, so any other reduction can be recomputed without rerunning.

The reason no parent collapses is structural: the 15 legacy issues are broad, anchored on
time-spanning vocabulary, and alive in 2026 by construction. Several are *higher* after the topic
dies than during its peak — `Civil rights & race` sits at **3.85x** its 1780–1800 level after Indian
affairs dies, and `Energy & environment` at **6.32x** after homesteading dies. A crosswalk parent
with a ratio above 1 is not tracking the topic at all and its verdict carries no information — yet
such a parent still **votes "persists"**, deliberately. Dropping it would let a topic reach
`true_death` on the strength of a cross-check that was never informative; absence of a usable signal
must not be read as agreement between the labelers.

**The CorEx arm is therefore one-sided by construction.** Fifty level-2 topics are cross-checked
against 15 legacy buckets, so a parent is almost always broader than the topic hanging off it and
can persist on vocabulary unrelated to the topic that died. This arm can **veto** a death far more
readily than it can **confirm** one. The two signals are not symmetric evidence: signal 2 (successor
detection) is the affirmative one.

Consequence: **the "a death requires BOTH labelers to collapse" criterion is unreachable against
this crosswalk**, and all of the discrimination in the final classification comes from successor
detection. That is exactly the blindness the corpus-native taxonomy was built to escape, showing up
one level down. Each dying topic's per-parent ratios are stored in `corex_parent_ratios` so the rule
can be re-litigated without recomputation.

A second structural limit: **seven level-2 topics have no legacy crosswalk parent at all** —
`Civil Service Reform & the Merit System`, `Constitutional Union & Federalism`,
`Executive Departments, Postal & Administrative Housekeeping`, `Partisan Combat…`,
`Personal Narrative…`, `Presidential Humility & Reflection on Office`,
`Territorial Organization, Statehood & Insular Governance`. `crosswalk_v1` guarantees every legacy
*issue* maps to ≥1 topic, not the reverse. For those topics the CorEx cross-check is simply
unavailable; the module records `legacy_parents = null` and refuses to claim agreement.

### 6.2 Renames the test does *not* find, and why

`Reconstruction & Southern Self-Government` → `Civil Rights, Voting Rights & Discrimination` is the
obvious candidate rename, and it is rejected: the two curves are **positively** correlated because
civil rights first becomes substantive in 1865, while Reconstruction is still peaking, and the two
run together through 1878. The handoff test cannot distinguish "succeeded" from "co-occurred" here.
Reported as `unresolved_death` rather than forced.

`Chinese Immigration & Exclusion` → `Immigration & Border Security` likewise fails, and it fails for
**the same co-occurrence reason**, not for the opposite one. The two topics did not hand off — they
ran concurrently for the whole of exclusion's life. `Immigration & Border Security` is substantive
in **10 of the 24 years** of exclusion's decline window [1884, 1907] — 1889, 1890, 1891, 1895,
1902–1906, and **1907, the death year itself**.

The filter that rejects it is **filter 2, the handoff test**: the candidate's first substantive year
(**1873**) *precedes* exclusion's peak (**1884**), so it was already established before the topic it
would supposedly succeed began declining. Filters 1, 3 and 4 all pass (same domain; outlives it;
shares the `Immigration` legacy parent), and at r = **−0.02** even the sign gate would have let it
through. A successor has to arrive *during* the decline; this one was already there. Calling it a
rename would be an assumption, not a measurement.

Immigration's later revival is real but is a **separate fact about a different span**: exclusion
dies in 1907 and immigration does not return to substantive attention until **1999 — 92 years
later** (§3.1). Do not confuse that with the **43**-year figure in §5's table, which is
`Immigration & Border Security`'s own longest internal silence, **1956–1998**, and has nothing to do
with exclusion's death.

### 6.3 Two limits of the successor search that a reader cannot see from the results

**Successor search is structurally empty for five of the seventeen level-1 domains.** Candidates are
restricted to the dying topic's own level-1 domain, so a domain holding exactly one level-2 topic can
never yield a successor for it. Five domains are singletons: **`Indian & Tribal Affairs`**,
`Ceremonial & Commemorative Address`, `Procedural & Administrative`, `Faith & National Values`, and
`Partisan & Media Combat`. (The rest range from 2 to 8 topics; `Foreign Relations & Diplomacy` is
the largest at 8.)

This bites hardest on the report's marquee death: `Indian Affairs, Removal & Allotment` is the only
topic in `Indian & Tribal Affairs`, so **the search space was empty**. "No successor found" there
means "there was nothing to search", not "we looked at candidates and rejected them" — a materially
weaker statement, and one that generalizes: for any topic in those five domains, an absence of
successor is a property of the taxonomy's shape, not evidence about history.

**Filter 4 is conditionally skipped, and closing that gap was measured and deferred.** A candidate
successor must share a legacy crosswalk parent with the dying topic (filter 4). When the *dying*
topic has no crosswalk parent at all, the implementation currently **skips** filter 4 rather than
rejecting every candidate, so `Constitutional Union & Federalism` and
`Territorial Organization, Statehood & Insular Governance` are judged on filters 1–3 only. This
contradicts the docstring's "four independent filters" and was flagged as a defect.

Making it unconditional **was implemented and measured**, and it is not a cleanup — it moves
published numbers. Under `sotu` it flips `Constitutional Union & Federalism` from `rename` to
`unresolved_death`, blanks its `successor_topic` and `successor_corr` (−0.278), and clears six
`cross_domain_candidate` diagnostics. It was therefore **reverted pending a method decision**,
because the choice is substantive rather than mechanical: making filter 4 unconditional means
*absence of crosswalk coverage becomes affirmative evidence against a rename*, and treating
no-information as no-constraint may well be the better rule. Tracked as
`find-successor-filter4-skip-moves-published-numbers`. The affected topic is the one discussed in
§3's caveat list.

---

## 7. Where the genre treatments disagree (read these before quoting a peak)

Genre standardization moves the peak year by more than 20 years for **ten** topics. That is all ten,
sorted by the size of the shift:

| Topic | raw peak | genre-standardized peak | shift | SOTU-only peak |
|---|---|---|---|---|
| Public Debt, Revenue & Treasury Finance | 1793 | **1934** | 141 yrs | 1934 |
| Military Preparedness, Armed Forces & Veterans | 1809 | **1940** | 131 | 1807 |
| Commemoration, Eulogy & National Mourning | 1800 | **1925** | 125 | 1971 |
| Providence, Faith & American Ideals | 1939 | **1822** | 117 | 1986 |
| Tariffs, Reciprocity & Navigation Laws | 1828 | **1909** | 81 | 1828 |
| Monroe Doctrine & Latin American Policy | 1903 | **1823** | 80 | 1854 |
| Civil Service Reform & the Merit System | 1889 | **1948** | 59 | 1880 |
| Internal Improvements, Energy & Infrastructure | 1929 | **1975** | 46 | 1929 |
| Nuclear Weapons & Arms Control | 1986 | **1955** | 31 | 1960 |
| Middle East & Israel | 1981 | **1956** | 25 | 1981 |

**Three topics that are not deaths under `raw` become deaths under another treatment:**
`Public Debt, Revenue & Treasury Finance` (persistent raw → **died** under both other treatments),
`Civil Service Reform` (revived raw → died genre-standardized → born SOTU), and
`Treaties, Diplomacy & International Arbitration` (persistent raw → **died** inside annual messages,
where treaty enumeration was a 19th-century annual-message genre convention that modern SOTUs
dropped).

More broadly, **14 of the 50 level-2 topics change lifecycle class under at least one treatment**.
The three above are the subset where the change creates a death that `raw` does not show; the rest
shuffle among `persistent` / `revived` / `born`. Treatment is not a robustness footnote at this
granularity — always check which treatment a class came from.

**Structural caveat for every early peak.** The founding-era corpus is 927 paragraphs across 27
years, and any topic that appears at all in 1790 gets a large share of a 32-paragraph year. **Two
distinct artifacts produce early peaks, and genre standardization absorbs neither.**

1. **Annual-message enumeration.** The early corpus is *already* 55% annual messages — a genre that
   enumerates every department in turn — so there is no other genre to reweight against. Post-
   stratification can only redistribute weight among genres that are present.

2. **Single-document concentration.** A thin year's share can be set by *one speech*, and this is a
   different failure with a different fix. Genre standardization cannot touch it, because the
   document sits *inside* a genre stratum rather than being a property of the genre mix — and it can
   strike a non-SOTU genre just as easily. **This, not enumeration, is what produces the Indian
   Affairs 1790 peak**: 14 of that year's 17 topic paragraphs come from a single speech, the *Talk
   to the Chiefs and Counselors of the Seneca Nation*, which is `public_remarks_or_address` and not
   an annual message at all (§1.1). Attributing that peak to annual-message enumeration names the
   wrong mechanism.

   The defence against this one is the CI design, not the genre treatments: the bootstrap resamples
   **speeches**, so a share carried by a single document gets a correspondingly wide interval.

Ten topics peak before 1815 (`Indian Affairs` 1790, `Crime & Insurrection` 1793, `Public Debt` 1793,
`Treaties` 1797, `Presidential Humility` 1799, `Commemoration` 1800, `Public Lands` 1802,
`Military Preparedness` 1809, `Neutral Rights` 1810, `Early Naval Wars` 1814). For Indian affairs
the founding *era* share survives a CI check even after deleting the concentrated document (§1.1);
for the others, check the era CIs in `bootstrap_era_shares()` — and the document spread behind the
peak year — before treating an early peak year as a claim.

---

## 8. Reproducing

```bash
cd <repo-or-worktree>

# ALWAYS confirm which source tree you are importing FIRST. The editable install's .pth
# hardcodes the MAIN repo's src, so a bare python in a worktree silently runs main-repo code.
PYTHONPATH=$PWD/src arch -x86_64 <repo>/.venv/bin/python \
    -c "import presidential_profiles as m; print(m.__file__)"

PYTHONPATH=$PWD/src arch -x86_64 <repo>/.venv/bin/python -m presidential_profiles.attention

# exemplar quotes for one topic:
PYTHONPATH=$PWD/src arch -x86_64 <repo>/.venv/bin/python -m presidential_profiles.attention \
    --quotes "Coinage, Currency & Specie" --quote-years 1890 1900
```

`<repo>` is the main checkout (the `.venv` lives there); `$PWD` is whichever tree you actually want
to run. If the confirmation line does not print the path you expect, stop — every number below it
will come from the wrong source tree.

`main()` rewrites `data/attention/topic_lifecycles.parquet` byte-identically (sha256
`825fbe66…f0e`) and prints the ground-truth and anachronism checks. Note that `corpus.DATA_DIR` is
package-relative, so a `PYTHONPATH=<worktree>/src` run writes into the **worktree's** `data/`.

The full 9-era share + CI + sample-size grid (used for every era table above) is not written to
disk; regenerate it with
`attention.bootstrap_era_shares(paragraphs, assignments, taxonomy, level, treatment)`, and the
year-level curves with `attention.attention_curves(...)`. `attention.exemplar_quotes(...)` returns
each quote's `doc_name` and `para_idx` alongside the text, which is how the §3.1 attributions were
resolved.

---

## 9. Verification status of the numbers in this report

**Sixteen defects in the first draft of this note have been corrected — fourteen of them numeric or
substantive claims that did not reproduce against the artifact.** They were found in three passes:

| Pass | Found | Examples |
|---|---|---|
| Independent verification of the note against the parquet | **7** | a self-contradicting divergence range; a peak-shift table not sorted by the quantity it ranked; four double-rounded rate cells |
| A documentation sweep of the numbers that pass had *not* listed | **4** | "four absolute births" (two); "three lifecycle classes flip" (fourteen); "eighty years later" (ninety-two); a pre-dedup assignment count |
| Final re-verification gate | **5** | §6.2 naming the wrong rejecting filter *and* the wrong gap (43 vs 92); a mode claim true at no grain; a 70-year trough (68); two presentational defects |

**Every one was invisible to a reader of the note alone.** All sixteen surfaced only by recomputing
from `topic_lifecycles.parquet` and the corpus — none from re-reading the prose. That is the single
most important thing to know about this document: **its internal consistency is not evidence.** The
sweep pass is the reason to distrust spot-checking; it found four errors in cells nobody had flagged,
in a note that had already been through a full verification pass.

This section therefore records what has since been re-derived and what has not, because a report
that says what it did not check is more trustworthy than one that implies it checked everything.

**Re-derived from the artifact and the corpus (independently recomputed, not copied between
sections):**

* §1 — all 6 check rows; §1.1 — all 9 era rows × 4 columns; the 1790 document breakdown (17
  paragraphs / 32; 14 Seneca, 3 annual-message); the SOTU 1789–1793 window counts; the 1830s
  second-mode decade share (96/1,447 = 6.63%) and the Gilded Age third mode (5.15%).
* §1.2 — all 10 paragraph counts and all 10 smoothed shares in the year table; the 1997–2001 run-up.
* §1.3 — 178 corpus-wide vocabulary hits, 35 in 1918–1933 (2.018%), the 26/9 split, the 9 scattered
  topics, and the full 1910–1937 smoothed curve.
* §1.4 — all 7 rows (raw first appearance, first substantive, gap) plus the two large-gap
  non-anachronisms; the "single stray paragraph" claim verified for all 7.
* §2 — all 9 era denominators; all level-1 classes and counts; the under-powered guard's single
  firing (45 paragraphs).
* §3 — all 17 rows × 9 columns; the 14-of-17 / 15-of-17 / 16-of-17 genre-robustness counts; the
  3-topic caveat list.
* §4 — all 14 rows × 6 columns including every present-era CI; the absolute-birth era counts.
* §5 — all 7 rows × 7 columns; the 33–73 year gap range.
* §6 — all 4 rename rows including successor correlations and handoff years; §6.1's full 15-topic
  divergence table, the LLM and CorEx retention ranges, the per-parent ratio minimum, and the
  threshold sweep (9 thresholds).
* §6.2 — both rejected candidates traced **filter by filter**, not asserted: Reconstruction
  (r = +0.2153, sign gate) and Chinese Immigration (filter 2, candidate first year 1873 < peak 1884;
  filters 1/3/4 and the sign gate all pass; 10 substantive candidate years inside the 24-year
  decline window).
* §6.3 — level-1 domain sizes for all 17 domains, giving the 5 singletons.
* §7 — the full 10-topic peak-shift table with shifts; the 3-vs-14 flip counts; the founding-era
  paragraph/year/genre-mix figures; all 10 pre-1815 peak years.
* §1.1 modality — decade, era, and smoothed-year grains computed separately, since the mode claim
  is true at only one of them.
* §3.1/§4/§5 exemplars — every quote's source document located and its `para_idx` recorded; all 17
  re-collated verbatim against `paragraphs.parquet`, including the elided ones.

**NOT re-derived in this pass:**

* **Quote text was not re-verified verbatim here.** A prior verification pass confirmed 17/17
  exemplar quotes verbatim with correct year, president and topic attribution; this pass re-derived
  only each quote's *source document and paragraph index*, and did not re-collate the wording.
* **Historical-context sentences are not measurements** and carry no verification status: e.g. "1980
  is the Iran hostage rescue", "1983–84 Lebanon and Grenada", the Bank War, Versailles, the
  Compromise of 1877. They are editorial gloss on the measured curves.
* **Only the `raw` treatment's lifecycle table was swept cell by cell.** `sotu` and
  `genre_standardized` were re-derived only where this report quotes them (the §7 table, the genre
  robustness counts, the §3 caveat list, the under-powered guard, and §6.3's filter-4 measurement).
* **CIs were regenerated at level2/`raw` only.** The level-1 and non-`raw` bootstrap grids were not
  recomputed; no CI from those grids is quoted in this report.
**Known open items deliberately left in the numbers**, rather than silently smoothed: the filter-4
skip (§6.3), the singleton-domain successor search (§6.3), and the era-grid vs year-curve genre
eligibility question logged as `era-grid-vs-year-curve-genre-eligibility`.
