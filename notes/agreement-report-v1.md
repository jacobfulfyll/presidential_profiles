# Inter-model annotation agreement — v1

- primary model: `claude-sonnet-5` (the committed annotation tables, all run_ids)
- second-opinion model: `claude-opus-4-8` (byte-identical prompts; the model is the only variable)
- sample: `/Users/jacobpress/Desktop/Projects/presidential_profiles/.worktree/inter-model-agreement-check/data/llm_annotations/agreement_sample_v1.json` — 266 speeches, seed 20260721, drawn 2026-07-21
- join coverage: 8,570/9,048 sample paragraphs (94.72%); 262/266 speeches
- speech-factual pass included: yes
- Cohen's kappa < 0.4 is FLAGGED low-confidence below. It is flagged, NEVER gated: disagreement is published, not hidden.

> WARNING: Opus coverage is below 100% — 4 sample speech(es) are missing from the join. Metrics below cover the overlap only; re-run the Opus pass to convergence before treating these as final.

## Agreement by field and era

Every era bin reports its n; nothing is suppressed. `overall` is the corpus-wide (non-stratified) value.

**topics — jaccard**

| era | value | n |
|---|---|---|
| overall | 0.678 | 8,570 |
| 1789-1818 | 0.752 | 242 |
| 1819-1848 | 0.740 | 978 |
| 1849-1878 | 0.682 | 897 |
| 1879-1908 | 0.681 | 1,209 |
| 1909-1938 | 0.631 | 972 |
| 1939-1968 | 0.672 | 1,220 |
| 1969-1998 | 0.649 | 1,313 |
| 1999-2028 | 0.680 | 1,739 |

**party_attack — cohen_kappa**

| era | value | n | flag |
|---|---|---|---|
| overall | 0.716 | 8,570 | |
| 1789-1818 | 0.000 | 242 | LOW (kappa<0.4) |
| 1819-1848 | 0.000 | 978 | LOW (kappa<0.4) |
| 1849-1878 | 0.538 | 897 | |
| 1879-1908 | 0.684 | 1,209 | |
| 1909-1938 | 0.763 | 972 | |
| 1939-1968 | 0.790 | 1,220 | |
| 1969-1998 | 0.651 | 1,313 | |
| 1999-2028 | 0.755 | 1,739 | |

**enemy_naming — cohen_kappa**

| era | value | n | flag |
|---|---|---|---|
| overall | 0.779 | 8,570 | |
| 1789-1818 | 0.761 | 242 | |
| 1819-1848 | 0.694 | 978 | |
| 1849-1878 | 0.681 | 897 | |
| 1879-1908 | 0.568 | 1,209 | |
| 1909-1938 | 0.807 | 972 | |
| 1939-1968 | 0.844 | 1,220 | |
| 1969-1998 | 0.776 | 1,313 | |
| 1999-2028 | 0.817 | 1,739 | |

**zero_sum — cohen_kappa**

| era | value | n | flag |
|---|---|---|---|
| overall | 0.489 | 8,570 | |
| 1789-1818 | 0.121 | 242 | LOW (kappa<0.4) |
| 1819-1848 | 0.315 | 978 | LOW (kappa<0.4) |
| 1849-1878 | 0.497 | 897 | |
| 1879-1908 | 0.403 | 1,209 | |
| 1909-1938 | 0.271 | 972 | LOW (kappa<0.4) |
| 1939-1968 | 0.556 | 1,220 | |
| 1969-1998 | 0.292 | 1,313 | LOW (kappa<0.4) |
| 1999-2028 | 0.548 | 1,739 | |

**proposal_values — exact_match**

| era | value | n |
|---|---|---|
| overall | 0.720 | 8,570 |
| 1789-1818 | 0.690 | 242 |
| 1819-1848 | 0.735 | 978 |
| 1849-1878 | 0.786 | 897 |
| 1879-1908 | 0.752 | 1,209 |
| 1909-1938 | 0.728 | 972 |
| 1939-1968 | 0.693 | 1,220 |
| 1969-1998 | 0.720 | 1,313 |
| 1999-2028 | 0.675 | 1,739 |

**speech_type — cohen_kappa**

| era | value | n | flag |
|---|---|---|---|
| overall | 0.954 | 266 | |
| 1789-1818 | 0.872 | 20 | |
| 1819-1848 | 1.000 | 23 | |
| 1849-1878 | 0.960 | 32 | |
| 1879-1908 | 1.000 | 27 | |
| 1909-1938 | 0.807 | 31 | |
| 1939-1968 | 0.940 | 45 | |
| 1969-1998 | 1.000 | 44 | |
| 1999-2028 | 1.000 | 44 | |

**speech_type — exact_match**

| era | value | n |
|---|---|---|
| overall | 0.962 | 266 |
| 1789-1818 | 0.900 | 20 |
| 1819-1848 | 1.000 | 23 |
| 1849-1878 | 0.969 | 32 |
| 1879-1908 | 1.000 | 27 |
| 1909-1938 | 0.839 | 31 |
| 1939-1968 | 0.956 | 45 |
| 1969-1998 | 1.000 | 44 |
| 1999-2028 | 1.000 | 44 |

**audience — cohen_kappa**

| era | value | n | flag |
|---|---|---|---|
| overall | 0.950 | 266 | |
| 1789-1818 | 0.922 | 20 | |
| 1819-1848 | 1.000 | 23 | |
| 1849-1878 | 1.000 | 32 | |
| 1879-1908 | 1.000 | 27 | |
| 1909-1938 | 0.836 | 31 | |
| 1939-1968 | 0.907 | 45 | |
| 1969-1998 | 1.000 | 44 | |
| 1999-2028 | 0.929 | 44 | |

**audience — exact_match**

| era | value | n |
|---|---|---|
| overall | 0.966 | 266 |
| 1789-1818 | 0.950 | 20 |
| 1819-1848 | 1.000 | 23 |
| 1849-1878 | 1.000 | 32 |
| 1879-1908 | 1.000 | 27 |
| 1909-1938 | 0.903 | 31 |
| 1939-1968 | 0.933 | 45 |
| 1969-1998 | 1.000 | 44 |
| 1999-2028 | 0.955 | 44 |

**medium — cohen_kappa**

| era | value | n | flag |
|---|---|---|---|
| overall | 0.802 | 266 | |
| 1789-1818 | 0.783 | 20 | |
| 1819-1848 | n/a | 23 | |
| 1849-1878 | 1.000 | 32 | |
| 1879-1908 | 1.000 | 27 | |
| 1909-1938 | 0.692 | 31 | |
| 1939-1968 | 0.800 | 45 | |
| 1969-1998 | 0.642 | 44 | |
| 1999-2028 | 0.489 | 44 | |

**medium — exact_match**

| era | value | n |
|---|---|---|
| overall | 0.865 | 266 |
| 1789-1818 | 0.900 | 20 |
| 1819-1848 | 1.000 | 23 |
| 1849-1878 | 1.000 | 32 |
| 1879-1908 | 1.000 | 27 |
| 1909-1938 | 0.806 | 31 |
| 1939-1968 | 0.867 | 45 |
| 1969-1998 | 0.795 | 44 |
| 1999-2028 | 0.705 | 44 |

## Entities

Name-match (Jaccard over normalized `(doc, para, name)` keys) is reported SEPARATELY from stance agreement among matched entities, plus each side's unmatched-entity rate — so a granularity difference (one model splitting an entity in two) shows up as unmatched entities, not as a stance disagreement. Names are normalized by lowercasing and collapsing whitespace.

**entities — name_match_rate**

| era | value | n |
|---|---|---|
| overall | 0.476 | 8,704 |
| 1789-1818 | 0.481 | 208 |
| 1819-1848 | 0.506 | 706 |
| 1849-1878 | 0.471 | 917 |
| 1879-1908 | 0.489 | 962 |
| 1909-1938 | 0.363 | 691 |
| 1939-1968 | 0.504 | 1,613 |
| 1969-1998 | 0.439 | 1,537 |
| 1999-2028 | 0.507 | 2,070 |

**entities — stance_agreement**

| era | value | n |
|---|---|---|
| overall | 0.881 | 4,147 |
| 1789-1818 | 0.910 | 100 |
| 1819-1848 | 0.857 | 357 |
| 1849-1878 | 0.822 | 432 |
| 1879-1908 | 0.851 | 470 |
| 1909-1938 | 0.912 | 251 |
| 1939-1968 | 0.902 | 813 |
| 1969-1998 | 0.880 | 674 |
| 1999-2028 | 0.901 | 1,050 |

**entities — unmatched_primary_rate**

| era | value | n |
|---|---|---|
| overall | 0.399 | 6,897 |
| 1789-1818 | 0.363 | 157 |
| 1819-1848 | 0.367 | 564 |
| 1849-1878 | 0.380 | 697 |
| 1879-1908 | 0.399 | 782 |
| 1909-1938 | 0.536 | 541 |
| 1939-1968 | 0.377 | 1,304 |
| 1969-1998 | 0.431 | 1,184 |
| 1999-2028 | 0.371 | 1,668 |

**entities — unmatched_opus_rate**

| era | value | n |
|---|---|---|
| overall | 0.303 | 5,954 |
| 1789-1818 | 0.338 | 151 |
| 1819-1848 | 0.285 | 499 |
| 1849-1878 | 0.337 | 652 |
| 1879-1908 | 0.277 | 650 |
| 1909-1938 | 0.374 | 401 |
| 1939-1968 | 0.275 | 1,122 |
| 1969-1998 | 0.344 | 1,027 |
| 1999-2028 | 0.277 | 1,452 |

## Low-confidence flags (kappa < 0.4)

6 field x era cell(s) flagged: party_attack/1789-1818, zero_sum/1789-1818, party_attack/1819-1848, zero_sum/1819-1848, zero_sum/1909-1938, zero_sum/1969-1998.
These are the cells where the two models most disagree after correcting for chance — candidates for the anachronism / reputation-contamination risks the design set out to make visible. Flagged, not gated.

## 10 highest-disagreement paragraphs (qualitative)

Disagreement score = flag mismatches (0-3) + (1 - topic Jaccard) + proposal_values mismatch + (1 - entity-name Jaccard). Spot-read these: genuine ambiguity is expected and fine; a systematic schema misread by one model would be a prompt bug to fix before trusting the table.

### 1969-1998 · score 5.80 · `/the-presidency/presidential-speeches/august-19-1976-remarks-republican-national-convention` para 4

- topics — primary: ['Cold War & Great-Power Rivalry', 'Nuclear Weapons & Arms Control', 'Partisan Combat, Press Conferences & Media Attacks'] | opus: ['Early Naval Wars: Barbary & the War Of 1812', 'National Bank & Banking Crises', 'Nuclear Weapons & Arms Control']
- flags — primary: party=True enemy=True zero_sum=True | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: mixed | opus: values
- entities — primary: ['Democratic rule (adversarial)'] | opus: ['Democratic (adversarial)']

> And then as I tried to write-let your own minds turn to that task. You're going to write for people a hundred years from now who know all about us, we know nothing about them. We don't know what kind of world they'll be living in. And suddenly I thought to myself, "If I write of the problems, they'll be the domestic problems of which the President spoke here tonight; the challenges confronting us, the erosion of freedom taken place under Democratic rule in this country, the invasion of private rights, the controls and restrictions on the vitality of the great free economy that we enjoy." These are the challenges that we must meet and then again there is that challenge of which he spoke that […]

### 1939-1968 · score 5.50 · `/the-presidency/presidential-speeches/april-11-1951-report-american-people-korea` para 2

- topics — primary: ['Cold War & Great-Power Rivalry'] | opus: ['Cold War & Great-Power Rivalry', 'World Order, Foreign Aid & Democracy Promotion']
- flags — primary: party=False enemy=True zero_sum=True | opus: party=True enemy=False zero_sum=False
- proposal_values — primary: values | opus: neither
- entities — primary: ['dictators (adversarial)'] | opus: ['Greece (neutral)', 'Soviet Union (adversarial)', 'United Nations (favorable)']

> If they don't act together, they are likely to be picked off, one by one. If they had followed the right policies in the 1930's - if the free countries had acted together to crush the aggression of the dictators, and if they had acted in the beginning when the aggression was small - there probably would have been no World War II. If history has taught us anything, it is that aggression anywhere in the world is a threat to the peace everywhere in the world. When that aggression is supported by the cruel and selfish rulers of a powerful nation who are bent on conquest, it becomes a dear and present danger to the security and independence of every free nation.

### 1969-1998 · score 5.33 · `/the-presidency/presidential-speeches/august-29-1996-remarks-democratic-national-convention` para 20

- topics — primary: ['Prosperity, Jobs & the Middle Class', 'Public Debt, Revenue & Treasury Finance', 'Taxes, Budget Deficits & Federal Spending'] | opus: ['Prosperity, Jobs & the Middle Class', 'Taxes, Budget Deficits & Federal Spending']
- flags — primary: party=True enemy=True zero_sum=True | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: mixed | opus: values
- entities — primary: ['Clinton/Gore administration (favorable)'] | opus: []

> Audience: No-o-o! President Clinton: Do we really want to stop economic growth again? Audience: No-o-o! President Clinton: Do we really want to start piling up another mountain of debt? Audience: No-o-o! President Clinton: Do we want to bring back the recession of 1991 and '92? Audience: No-o-o! President Clinton: Do we want to weaken our bridge to the 21st century? Audience: No-o-o! President Clinton: Of course we don't. We have an obligation, you and I, to leave our children a legacy of opportunity, not a legacy of debt. Our budget would be balanced today, we would have a surplus today, if we didn't have to make the interest payments on the debt run up in the 12 years before the Clinton/Go […]

### 1999-2028 · score 5.00 · `/the-presidency/presidential-speeches/december-19-2008-remarks-plan-assist-automakers` para 6

- topics — primary: ['Executive Power, Vetoes & the Courts', 'Taxes, Budget Deficits & Federal Spending'] | opus: ['Labor, Wages & Working Conditions', 'Prosperity, Jobs & the Middle Class', 'Providence, Faith & American Ideals']
- flags — primary: party=True enemy=True zero_sum=False | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: proposal | opus: mixed
- entities — primary: ['Congress (adversarial)'] | opus: []

> Unfortunately, despite extensive debate and agreement that we should prevent disorderly bankruptcies in the American auto industry, Congress was unable to get a bill to my desk before adjourning this year. This means the only way to avoid a collapse of the U.S. auto industry is for the executive branch to step in. The American people want the auto companies to succeed, and so do I. So today, I'm announcing that the federal government will grant loans to auto companies under conditions similar to those Congress considered last week.

### 1999-2028 · score 5.00 · `/the-presidency/presidential-speeches/february-5-2019-state-union-address` para 6

- topics — primary: ['Labor, Wages & Working Conditions', 'Prosperity, Jobs & the Middle Class'] | opus: ['Labor, Wages & Working Conditions', 'Prosperity, Jobs & the Middle Class']
- flags — primary: party=True enemy=True zero_sum=True | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: mixed | opus: neither
- entities — primary: ['leaders of both parties (adversarial)'] | opus: []

> We must choose between greatness or gridlock, results or resistance, vision or vengeance, incredible progress or pointless destruction. Tonight, I ask you to choose greatness. Over the last 2 years, my administration has moved with urgency and historic speed to confront problems neglected by leaders of both parties over many decades. In just over 2 years since the election, we have launched an unprecedented economic boom - a boom that has rarely been seen before. We have created 5.3 million new jobs and importantly added 600,000 new manufacturing jobs - something which almost everyone said was impossible to do, but the fact is, we are just getting started.

### 1999-2028 · score 5.00 · `/the-presidency/presidential-speeches/february-5-2019-state-union-address` para 44

- topics — primary: ['World Order, Foreign Aid & Democracy Promotion'] | opus: ['Partisan Combat, Press Conferences & Media Attacks', 'Providence, Faith & American Ideals']
- flags — primary: party=True enemy=True zero_sum=True | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: values | opus: values
- entities — primary: ['socialism (adversarial)'] | opus: []

> Here, in the United States, we are alarmed by new calls to adopt socialism in our country. America was founded on liberty and independence--not government coercion, domination, and control. We are born free, and we will stay free. Tonight, we renew our resolve that America will never be a socialist country. One of the most complex set of challenges we face is in the Middle East.

### 1969-1998 · score 5.00 · `/the-presidency/presidential-speeches/january-29-1993-press-conference-gays-military` para 15

- topics — primary: ['Civil Rights, Voting Rights & Discrimination', 'Partisan Combat, Press Conferences & Media Attacks'] | opus: ['Military Preparedness, Armed Forces & Veterans']
- flags — primary: party=False enemy=True zero_sum=True | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: values | opus: neither
- entities — primary: ['opponents of policy change (adversarial)'] | opus: []

> The ban will be issued, or will be lifted, rather? The President. That is my position. My position is that I still embrace the principle, and I think it should be done. The position of those who are opposed to me is that they think that the problems will be so overwhelming everybody with good sense will change their position. I don't expect to do that. Q. So you definitely expect to do it. And secondly - The President. I don't expect to change my position, no. Q. What do you think is going to happen in the military? There have been all sorts of dire predictions of violence, of mass comingsout, whatever. What do you think the impact of this is going to be, practically?

### 1999-2028 · score 5.00 · `/the-presidency/presidential-speeches/june-6-2002-address-nation-department-homeland-security` para 0

- topics — primary: ['Iraq, Gulf Wars, the War on Terror & Interventions', 'Military Preparedness, Armed Forces & Veterans'] | opus: ['Iraq, Gulf Wars, the War On Terror & Interventions']
- flags — primary: party=False enemy=False zero_sum=False | opus: party=False enemy=True zero_sum=True
- proposal_values — primary: mixed | opus: values
- entities — primary: [] | opus: ['al Qaeda (adversarial)']

> Good evening. During the next few minutes, I want to update you on the progress we are making in our war against terror, and to propose sweeping changes that will strengthen our homeland against the ongoing threat of terrorist attacks. Nearly nine months have passed since the day that forever changed our country. Debris from what was once the World Trade Center has been cleared away in a hundred thousand truckloads. The west side of the Pentagon looks almost as it did on September the 10th. And as children finish school and families prepare for summer vacations, for many, life seems almost normal.

### 1969-1998 · score 5.00 · `/the-presidency/presidential-speeches/september-23-1976-debate-president-gerald-ford-domestic-issues` para 57

- topics — primary: ['Taxes, Budget Deficits & Federal Spending'] | opus: []
- flags — primary: party=True enemy=True zero_sum=False | opus: party=False enemy=False zero_sum=False
- proposal_values — primary: mixed | opus: neither
- entities — primary: ['Jimmy Carter (adversarial)'] | opus: ['Carter (adversarial)']

> We feel that in education we can have a slight increase, not a major increase. It's my understanding that Governor Carter has indicated that he approves of a $30 billion expenditure by the Federal Government, as far as education is concerned. At the present time we are spending roughly $3,500 million. I don't know where that money would come from.

### 1999-2028 · score 4.75 · `/the-presidency/presidential-speeches/june-6-2002-address-nation-department-homeland-security` para 15

- topics — primary: ['Cold War & Great-Power Rivalry', 'Executive Departments, Postal & Administrative Housekeeping'] | opus: ['Cold War & Great-Power Rivalry', 'Executive Power, Vetoes & the Courts', 'World Order, Foreign Aid & Democracy Promotion']
- flags — primary: party=False enemy=False zero_sum=False | opus: party=False enemy=True zero_sum=True
- proposal_values — primary: mixed | opus: values
- entities — primary: ['Harry Truman (favorable)'] | opus: []

> What I am proposing tonight is the most extensive reorganization of the federal government since the 1940s. During his presidency, Harry Truman recognized that our nation's fragmented defenses had to be reorganized to win the Cold War. He proposed uniting our military forces under a single Department of Defense, and creating the National Security Council to bring together defense, intelligence, and diplomacy. Truman's reforms are still helping us to fight terror abroad, and now we need similar dramatic reforms to secure our people at home.

