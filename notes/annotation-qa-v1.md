# Annotation QA — v1

- mode: CONVERGED (post-convergence gate)
- run_id: None
- scope: 1057 speeches (corpus-wide)

## Amendment (pre-registration)

- Amendment #1: the coverage gate was redefined from single-pass >= 99% to post-convergence == 100% (corpus-wide, for BOTH real specs).
- Amendment #2: "resubmitted once" -> "resubmitted to convergence across 7 round(s)". Chunk-escalated stragglers (judged with partial, not whole-speech, context): 32 chunk-escalated straggler(s): ['/the-presidency/presidential-speeches/april-16-1964-press-conference-state-department', '/the-presidency/presidential-speeches/april-22-1886-message-regarding-us-labor-force', '/the-presidency/presidential-speeches/april-27-1983-address-central-america', '/the-presidency/presidential-speeches/april-4-1917-republic-must-awaken', '/the-presidency/presidential-speeches/december-3-1912-fourth-annual-message', '/the-presidency/presidential-speeches/december-5-1876-eighth-annual-message', '/the-presidency/presidential-speeches/december-5-1898-second-annual-message', '/the-presidency/presidential-speeches/december-7-1835-seventh-annual-address-congress', '/the-presidency/presidential-speeches/december-7-1847-third-annual-message', '/the-presidency/presidential-speeches/december-8-1925-third-annual-message', '/the-presidency/presidential-speeches/december-8-2003-address-signing-medicare-legislation', '/the-presidency/presidential-speeches/december-9-1908-eighth-annual-message', '/the-presidency/presidential-speeches/february-2-1953-state-union-address', '/the-presidency/presidential-speeches/february-23-2018-remarks-conservative-political-action', '/the-presidency/presidential-speeches/january-11-1944-fireside-chat-28-state-union', '/the-presidency/presidential-speeches/january-15-2009-farewell-address-nation', '/the-presidency/presidential-speeches/january-22-1917-world-league-peace-speech', '/the-presidency/presidential-speeches/january-29-1984-address-announcing-his-candidacy-reelection', '/the-presidency/presidential-speeches/january-7-1954-state-union-address', '/the-presidency/presidential-speeches/july-19-2013-remarks-trayvon-martin', '/the-presidency/presidential-speeches/july-2-1964-remarks-upon-signing-civil-rights-bill', '/the-presidency/presidential-speeches/july-22-1920-high-wages-high-production', '/the-presidency/presidential-speeches/june-13-1870-message-regarding-insurrection-cuba', '/the-presidency/presidential-speeches/june-6-1961-report-american-people-returning-europe', '/the-presidency/presidential-speeches/june-7-1961-remarks-graduating-class-us-naval-academy', '/the-presidency/presidential-speeches/march-4-1853-inaugural-address', '/the-presidency/presidential-speeches/march-4-1901-second-inaugural-address', '/the-presidency/presidential-speeches/march-4-1921-inaugural-address', '/the-presidency/presidential-speeches/may-18-1963-90th-anniversary-vanderbilt-university', '/the-presidency/presidential-speeches/may-31-1870-message-regarding-dominican-republic-annexation', '/the-presidency/presidential-speeches/september-12-2002-remarks-un-general-assembly', '/the-presidency/presidential-speeches/september-23-2010-address-united-nations'].

- paragraph coverage: 100.00% (36,229/36,229)
- speech coverage: 100.00% (1,057/1,057)
- unlabeled-paragraph share (topics==[]): 1.12%

## Gated checks

- [PASS] speech_coverage==100%: 100.00% (1,057/1,057)
- [PASS] paragraph_coverage==100%: 100.00% (36,229/36,229)
- [PASS] party_attack_non_degenerate: 6.56%
- [PASS] enemy_naming_non_degenerate: 21.07%
- [PASS] zero_sum_non_degenerate: 7.68%
- [PASS] entity_name_in_paragraph>=80%: 85.81%

## Per-flag corpus-wide rates

- party_attack: 6.56%
- enemy_naming: 21.07%
- zero_sum: 7.68%

## Flag co-occurrence

- any_flag: 23.09%
- party+enemy: 5.85%
- enemy+zero_sum: 6.29%
- all_three: 1.65%

## Enemy-naming <-> adversarial-entity consistency (reported)

- 7393/7634 enemy-naming paragraphs carry >=1 adversarial entity (96.84%)

## Entities

- rows: 27214
- entity-name-appears-in-paragraph rate: 85.81%

## Top topics by decade

- 1780s: Providence, Faith & American Ideals (8), Presidential Humility & Reflection on Office (5), Constitutional Union & Federalism (3), Executive Power, Vetoes & the Courts (1)
- 1790s: Treaties, Diplomacy & International Arbitration (82), Indian Affairs, Removal & Allotment (61), Neutral Rights & Maritime Depredations (58), Constitutional Union & Federalism (44), Providence, Faith & American Ideals (44)
- 1800s: Indian Affairs, Removal & Allotment (69), Military Preparedness, Armed Forces & Veterans (64), Treaties, Diplomacy & International Arbitration (52), Neutral Rights & Maritime Depredations (49), Relations with Spain, Mexico & Territorial Claims (39)
- 1810s: Early Naval Wars: Barbary & the War of 1812 (89), Treaties, Diplomacy & International Arbitration (69), Neutral Rights & Maritime Depredations (64), Military Preparedness, Armed Forces & Veterans (64), Relations with Spain, Mexico & Territorial Claims (55)
- 1820s: Treaties, Diplomacy & International Arbitration (125), Military Preparedness, Armed Forces & Veterans (115), Tariffs, Reciprocity & Navigation Laws (95), Public Debt, Revenue & Treasury Finance (77), Monroe Doctrine & Latin American Policy (66)
- 1830s: National Bank & Banking Crises (347), Constitutional Union & Federalism (258), Treaties, Diplomacy & International Arbitration (235), Executive Power, Vetoes & the Courts (189), Public Debt, Revenue & Treasury Finance (132)
- 1840s: Treaties, Diplomacy & International Arbitration (273), Relations with Spain, Mexico & Territorial Claims (229), Mexican War (188), Public Debt, Revenue & Treasury Finance (169), National Bank & Banking Crises (148)
- 1850s: Slavery, Emancipation & Sectionalism (511), Treaties, Diplomacy & International Arbitration (203), Constitutional Union & Federalism (181), Monroe Doctrine & Latin American Policy (145), Relations with Spain, Mexico & Territorial Claims (140)
- 1860s: Constitutional Union & Federalism (318), Executive Power, Vetoes & the Courts (287), Reconstruction & Southern Self-Government (272), Slavery, Emancipation & Sectionalism (253), Treaties, Diplomacy & International Arbitration (116)
- 1870s: Treaties, Diplomacy & International Arbitration (191), Executive Departments, Postal & Administrative Housekeeping (124), Monroe Doctrine & Latin American Policy (90), Military Preparedness, Armed Forces & Veterans (89), Public Debt, Revenue & Treasury Finance (84)
- 1880s: Treaties, Diplomacy & International Arbitration (218), Executive Departments, Postal & Administrative Housekeeping (151), Civil Service Reform & the Merit System (150), Tariffs, Reciprocity & Navigation Laws (139), Military Preparedness, Armed Forces & Veterans (138)
- 1890s: Treaties, Diplomacy & International Arbitration (366), Coinage, Currency & Specie (196), Tariffs, Reciprocity & Navigation Laws (178), Military Preparedness, Armed Forces & Veterans (174), Monroe Doctrine & Latin American Policy (158)
- 1900s: Treaties, Diplomacy & International Arbitration (258), Military Preparedness, Armed Forces & Veterans (244), Trusts, Corporations & Antitrust (218), Tariffs, Reciprocity & Navigation Laws (211), Monroe Doctrine & Latin American Policy (202)
- 1910s: Treaties, Diplomacy & International Arbitration (191), World War I Mobilization & War Aims (171), Tariffs, Reciprocity & Navigation Laws (156), Trusts, Corporations & Antitrust (154), Military Preparedness, Armed Forces & Veterans (131)
- 1920s: Treaties, Diplomacy & International Arbitration (146), Providence, Faith & American Ideals (139), Internal Improvements, Energy & Infrastructure (137), Military Preparedness, Armed Forces & Veterans (92), Agriculture & Farm Prices (91)
- 1930s: Great Depression Recovery (321), Providence, Faith & American Ideals (114), Labor, Wages & Working Conditions (102), National Bank & Banking Crises (95), Agriculture & Farm Prices (82)
- 1940s: World War II Military Operations (462), Military Preparedness, Armed Forces & Veterans (141), Providence, Faith & American Ideals (133), World Order, Foreign Aid & Democracy Promotion (106), Treaties, Diplomacy & International Arbitration (98)
- 1950s: Cold War & Great-Power Rivalry (191), Military Preparedness, Armed Forces & Veterans (107), World Order, Foreign Aid & Democracy Promotion (90), Providence, Faith & American Ideals (67), Nuclear Weapons & Arms Control (67)
- 1960s: Vietnam War (916), Cold War & Great-Power Rivalry (864), Providence, Faith & American Ideals (407), World Order, Foreign Aid & Democracy Promotion (393), Treaties, Diplomacy & International Arbitration (341)
- 1970s: Taxes, Budget Deficits & Federal Spending (207), Providence, Faith & American Ideals (200), Vietnam War (197), Internal Improvements, Energy & Infrastructure (183), Cold War & Great-Power Rivalry (165)
- 1980s: Cold War & Great-Power Rivalry (439), Providence, Faith & American Ideals (400), Taxes, Budget Deficits & Federal Spending (369), Nuclear Weapons & Arms Control (288), Military Preparedness, Armed Forces & Veterans (240)
- 1990s: Providence, Faith & American Ideals (231), Health Care, Medicare & Social Security (200), Welfare, Poverty & Economic Opportunity (185), Taxes, Budget Deficits & Federal Spending (172), Treaties, Diplomacy & International Arbitration (171)
- 2000s: Iraq, Gulf Wars, the War on Terror & Interventions (359), Providence, Faith & American Ideals (224), World Order, Foreign Aid & Democracy Promotion (181), Health Care, Medicare & Social Security (169), Military Preparedness, Armed Forces & Veterans (122)
- 2010s: Providence, Faith & American Ideals (342), Prosperity, Jobs & the Middle Class (333), Partisan Combat, Press Conferences & Media Attacks (257), Immigration & Border Security (257), Taxes, Budget Deficits & Federal Spending (218)
- 2020s: Health Care, Medicare & Social Security (358), Partisan Combat, Press Conferences & Media Attacks (351), Cold War & Great-Power Rivalry (321), Providence, Faith & American Ideals (301), Military Preparedness, Armed Forces & Veterans (276)
