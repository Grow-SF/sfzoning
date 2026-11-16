# The 2000 solve's big mistakes (2026-11-16)

An overnight investigation of the 44 incidents of 20+ lots in the solve `final` (5,114 of its 7,100 wrong lots).
Investigation only: no solver code, input or stage output changed. Scripts are in `pipeline/scripts/incidents/`
(local commits 01368b8, 59e1903, 64f737e, e144830, 2c5e55b, 48bfe4c on `solver`, not pushed). Outputs are in
`pipeline/data/solver/experiments/incidents-*`. Where a diagnosis and its adversarial check differ, this note uses
the check.

## The answer

Most of the big mistakes are the search, not the inputs. In 13 of the 44 (2,873 lots) the run's own energy scores the
right map lower than the solved one; the search stops before it. A new search move, `compound` (the compound move
made a QPBO fusion move), reaches 8 of the 13, lowers the energy in every quarter and takes wrong lots from 7,100 to
4,543 with 14 lots broken. The rest are walls (1,368 lots) and placement and labels (873 lots).

Every lot, no data errors left out (the one measure), each quarter solved on its own held-out energy and joined:

| Measure | final | compound | all (energy's pick) | starts-compound | J7 (33 data errors left out) |
|---|---|---|---|---|---|
| Table district lines drawn | 96.3% | 96.4% | 96.4% | 96.5% | 96.3% |
| Drawn lines that are table lines | 96.6% | 96.6% | 96.7% | 96.7% | 96.9% |
| Labels right | 94.5% | 94.8% | 94.5% | 94.9% | 95.0% |
| Holes | 48 | 48 | 47 | 48 | 15 |
| Incidents | 868 | 862 | 853 | 848 | 824 |
| Wrong lots | 7,100 | 4,543 | 4,850 | 4,250 | 8,625 |
| Lots fixed / broken | – | 2,571 / 14 | 2,779 / 529 | 3,015 / 165 | – |
| Energy change, four quarters | 0 | −231.1 | −826.9 | −520.4 | – |

- J7 has only a data-errors-left-out score. This run scored that way: 96.4%, 96.7%, 94.5%, 12 holes, 868, 7,100.
  J7's answer on this build's lots, every lot: 8,630 wrong lots.
- **4,250 is a best-of-nine pick by table score.** Energy alone picks `all`: 4,850 wrong lots, 853 incidents, labels
  94.5%. Across the nine variants the rank correlation of energy and wrong lots is 0.30.
- **`compound` alone is the cleanest:** 14 broken, net +4 outside the 44 incidents, no new incident of 20+ lots,
  50–72 s per quarter. `starts-compound` breaks 165, is net −28 outside the 44, makes 5 new incidents of 20+ lots and
  costs five extra search runs per quarter.
- **#1 (the Mission-strip district, 1,838 lots) is 72% of compound's net gain and 64% of starts-compound's.** Without
  it compound gains 719 lots.
- **By incidents the gain is smaller:** 868 → 862 / 848; labels +0.3 / +0.4 points; lines +0.1 / +0.2.
- The variants start from the stage's answer and add moves. A solver change would run inside the search from scratch
  and could end elsewhere.
- **Gate before shipping:** a measure fixed in advance (energy per quarter, then incidents and labels) and another
  year's map, not the 2000 wrong-lot count the variant was picked by.

## Causes of the 44 incidents

| Cause | Incidents | Wrong lots |
|---|---|---|
| **Search** (the energy prefers the right map) | 13 | **2,873** (56%) |
| A: the expansion bound blocks a downhill one-code move | #4/#23, #14, #18, part of #15 | 445 + 57 |
| B: two or three codes must change at once | #1/#7/#34, #9/#44, #12, #13, #21 | 2,371 |
| **Walls** | 15 | **1,368** (27%) |
| SUD overlay boundaries read (rightly) as lines between two areas of one code | #3, #6 | 457 |
| P district printed across the Junipero Serra roadway; the line is at the curb | #2 | 451 |
| Roadway symbols read as a line, the real line read weak | #8 | 123 |
| Smaller, not diagnosed one by one | 11 | 337 |
| **Placement and labels** | 16 | **873** (17%) |
| Street label charged on lots it does not name (stay + far-side copy; leader dashes name a box) | #5, #30 | 297 |
| Misread label | #19, #20, #26, #28, #42 | 178 |
| Unlabelled area (#11: a rotated label no reader found) | #11, #25, #40 | 148 |
| Coupled with a walls incident | #10 (#6), #24 (#8) | 135 |
| False print | #32, #35 | 54 |
| Move chances on a wrong lot | #27 | 31 |
| Search, in #12's set | #31 | 30 |
| **All** | 44 | **5,114** |

- Search = the 13 incidents whose lower answer was rebuilt from scratch and priced on the whole city (matches the
  census to 0.001). #14 moved from walls to search on that check; #42's lower answer is a swap beside it, not a fix.
- The far-side copy carries a wrong charge in 8 of the 16 placement incidents (485 lots).
- **Near-ties:** 10 incidents (656 lots) whose cheapest right answer costs 0 to +2 more: #5, #11, #14, #19, #24, #27,
  #30, #31, #41, #43. #6 is close too (+9.1).

**How measured:**
- Whole city, exact: the search's lower answers and the search experiments (full quarter solves, joined, scored).
- Confined search (the solver's search on D plus two rings, from the solved and the table's answer, the rest at
  final; 37 sets): the census's cheapest right answers, every walls ceiling, every placement number, the 1801
  deletion. The placement city-wide check searches one solved district at a time (up to 3,000 lots).
- Hand arithmetic: charge-once against the 11-11 leak cases.

**Adversarial corrections:**
- The walls "same-code" ceiling (+1,170) is mostly other causes; only #3 (+276) and #6 (+178), ~454, are overlay
  lines.
- The census's "J7's walls fix #3 and #6" is a trade: the J7-walls solve turns 552 Tenderloin lots to HAYES-GOUGH,
  95 worse on the 1,010-lot RC-4 district.
- The J7-walls solve's −186 is a reshuffle (+1,430 / −1,244); ~770 of its gains are final's search misses; only #8
  (+122) is a walls win. J7's walls read better per pair (0.0451 vs 0.0531 held out), mostly heavy lines across
  streets, but in the solve the two cancel.
- #8's false line is driven by enclosure, not street crossing.
- Search: leakage clean, scoring reproduced, deterministic (byte-identical reruns, seed-independent in se). The gain is
  narrow: #1 dominates, nw gains nothing.
- #10's labels 381 and 400 are read right (all four readers) and printed on neighbouring districts' edges.
- False print 1801 deepens the Mission basin by ~3 but does not explain the miss.

## The nine incidents of 100+ lots

Pictures are 2x the grid (1 px = 2 ft). Paths are under `pipeline/data/solver/experiments/`.

| # | Lots, table → solved, Q | Place | Cause, driving term | J7 | Adversarial | Fixed by |
|---|---|---|---|---|---|---|
| 1 | 1,838 RH-1 → RH-1(S), sw | Outer Mission, the RH-1 district behind the Mission St NC-2 strip | Search B: district → RH-1 (+144.8 alone) and the 42-lot tip → NC-2 (+2.6) only together (−21.9). The tip joins a 13,494-lot RH-1 region across 12 weak borders (median p 0.045). Right after the first stage (2,068/2,080), lost in the bounded same-code expansion. True borders as walls: still 0% right, right answer 67.7 lower. | wrong (NC-2) | fix holds in all four quarter solves; 72% of compound's gain; 12 tip lots broken | compound, compound-qpbo, fusion, all, starts, starts-compound |
| 2 | 451 RH-1(D) → RH-1, sw | Junipero Serra Blvd, west of the roadway | Walls, missing line: the roadway is printed as P, bold line at the west curb; the table's line falls mid-road (2,456 ft blank, 732 ft lettering), p 0.24. Cheapest right +26.8 (walls +43.2). Missed-line oracle +455. Appeared with the 11-15 walls. | wrong (C-2), J7 p 0.06 | not checked separately; diagnoses agree | none |
| 3 | 276 RC-4 → RM-1, ne | Lower Nob Hill, Bush to Post St | Walls, SUD line: North of Market Res. SUD boundary on Post St, 46 borders at p 0.99 charged +258 against labels −17.6. Cheapest right +199.5. | wrong (C-3-O); J7-walls = Tenderloin trade | AGREE; `sud` changes on 36 of 46; RM-1 enters via a POLK lot | none |
| 4 | 275 RH-1(D) → RH-1, se | Crocker-Amazon, Winding Way and Prague St | Search A: true dE −18.6 for 286 lots, the bound sees +15.8 (34.4 overcharge on 16 borders). Start had it right (303/326). −41.7 with #23. | right (walls) | lower answer reproduced; not separately challenged | every moving variant |
| 5 | 267 RM-1 → NC-1, nw | California St, 24th–27th Ave | Placement near-tie (+1.9): NC-1 labels in the street for leader-dash boxes, charged stay + far-side copy + box. 728 alone +10.1; 685's hand move only trains the fit (stay 0.42 vs box 0.37). | wrong (NC-1) | no separate check; the walls check classes its +275 as this near-tie | confined: hand_place, nofar, far_split, one_lot |
| 6 | 181 RC-4 → RH-2, ne | Van Ness Ave corridor | Walls, SUD line: Van Ness S.U.D. boundary, 10 borders at p 0.97. Cheapest right +9.1. Shares D with #10. | wrong (RH-2) | AGREE, fragile; `sud` changes on all 10; RH-2 enters via three unzoned lots | none |
| 7 | 175 RH-1 → RH-1(D), se | the #1 district from the se solve | Search B, 39 tip lots: +64.3 / +6.1 alone, −18.6 together. | wrong (NC-2) | as #1; without 1801 se ends right, sw does not | compound, compound-qpbo, fusion, all, starts-compound |
| 8 | 123 RH-2 → P, se | Glen Park, Chenery St at San Jose Ave / I-280 | Walls, roadway misread: false San Jose Ave line p 0.80–0.96 (enclosure +4.2, context +1.9, crossing +1.1; J7 0.01–0.13); real line 0.45 (J7 0.95). No labels. | right (walls) | AGREE; fix by a crossing profile, not charge-once | starts, starts-compound |
| 9 | 100 RH-2 → RH-1, se | Richland Ave and College Ave, off Mission St | Search B: 101 → RH-2 (+73.9), 20 → RH-3 (+1.6), 6 → NC-1 (+2.6), together −29.0. The compound move's "code no neighbour holds" rule bars it (+98.0). | right (walls) | lower answer reproduced; in compound's checked breadth | fusion, compound, compound-qpbo, starts, starts-compound, all |

Pictures: `incidents-search-diagnosis/pictures/search-1-sw_answers*.png`, `search-1-sw_walls_mission-tip.png`,
`search-7-se_answers.png`; `incidents-walls/pictures/2_lines.png`, `3_lines.png`, `3_ink.png`, `6_lines.png`,
`8_lines.png`, `8_walls*.png`, `rc4_district_by_sud.png` (the one to check by eye: magenta read lines should run where
the `sud` tint changes); `incidents-placement/pictures/5_placement.png`; `incidents-census/pictures/9_lines.png`. #4's
picture: `uv run python scripts/incidents/render_incident.py 4 --other incidents-search-compound`.

## What to try

### Search

- **C1 (top): the compound move as a QPBO fusion move.** Keep the trigger (a district whose labels favour another
  code); drop the two restrictions; the district offers its labels' code, each lot of the touching regions holding
  it offers its own next code; QPBO keeps only the lots that lower the energy. One move in `search.py` (~30 lines
  proposal, ~60 lines QPBO on PyMaxflow). Measured as `compound`. Needs the synthetic case "a strip's tip joined
  across weak borders to a same-code district beyond it", numbers drawn from the seven chain incidents.
- **C2: exact expansion (QPBO instead of the bound)** for mechanism A. `qpbo` alone: 6,638 (fixes #4/#23, #14, #18,
  445 lots). `compound-qpbo`: 4,485, 854 incidents, 92 broken (near-ties), 5–9.5 min more per quarter.
- **C3: cut-and-relabel (Kernighan–Lin, Levinkov et al. 2017) or exact small windows.** Not measured.
- Not recommended: annealing (no gain at 40 sweeps), many starts as the main fix, finer regions at fixed join levels.
- **Ceiling:** the search oracle (every verified lower answer put in; uses the table as a start, never an input):
  4,307 wrong lots, 856 incidents.
- **Variants by energy:** all −826.9 (4,850), fusion −652.1 (4,842), compound-qpbo −602.9 (4,485), starts-compound
  −520.4 (4,250), qpbo −473.1 (6,638), regions −466.8 (6,746), starts −437.3 (4,513), compound −231.1 (4,543),
  anneal 0 (7,100).
- **A lower energy exposes walls errors:** fusion and all break 328 RH-1 lots in se (blocks 6328/6324/6325); putting
  final back costs +25.0 (40 no-line borders on the table's line, median p 0.27). compound leaves it right, partly
  because its search is weaker there.
- **Past tries:** `spread-solve-tried` (11-11) was on old inputs and judged by wrong lots (`past-rejections-confounded`);
  J7's write-up blamed a line read as no line and the labels; the solver's design note already names the chain
  problem.
- **Reading:** Boykov, Veksler & Zabih 2001; Kolmogorov & Zabih 2004; Rother et al. 2005; Kolmogorov & Rother 2007;
  Rother, Kolmogorov, Lempitsky & Szummer 2007; Lempitsky, Rother, Roth & Blake 2010; Levinkov et al. 2017; Beier,
  Hamprecht & Kappes 2015; Gorelick et al. 2014/2017; Kappes et al. 2015; Haller, Swoboda & Savchynskyy 2018.
- **Libraries:** PyMaxflow GPL-3.0 (used; tonight's QPBO checked against brute force on 400 small energies);
  thinqpbo GPL-3.0 (production choice); OpenGM 2 MIT (unmaintained); OR-Tools Apache-2.0 / HiGHS MIT;
  bjoern-andres/graph licence not verified; gco-v3 / pygco research-only, not compatible.

### Walls

- **Lead: label-anchored forgiveness** (the walls adversarial's design). Forgive "same code across a read line" only
  on the film between two distinct regions of the current answer that each hold their own trusted label of that
  code; the amount learned from the table held out by quarter, as a function of the two sides' trusts. Unit = region
  pair (bubble design), so no ink matching or line purity; the 11-11 leak cases are untouched by construction (a hole
  merges A and B, and A's label names another code). Needs its synthetic case and a misread-label counter-case
  (trust 0.7–0.9). Risks: a misread on one side makes a wrong merge cheaper; the term depends on connectivity.
- **Rejected: charge a printed line once.** With a per-chain cap c, a hole makes the leak pay 2c: hole3_bad leaks for
  c < 5.6, hole3_smeared for c < 4.9; the proposed c ≈ 3.1–3.6 gives leaks of 9.4 and 10.8 against 13.8. #3 needs
  its line below ~10.6, #6 below ~3.4 per unit. It also needs the never-built junction-split unit, and capping a
  repulsive term does not make the expansion exact.
- **Crossing-profile feature** for roadways and offset lines (#2, #8): where the heaviest stroke sits along each
  crossing, how many strokes, their widths. Judge by held-out log loss per kind of print.
- **`text_discounted_enclosure` per kind of print:** the 11-15 rejection used all-pairs log loss.
- **Ceilings (confined, 37 sets):** same-code lines charged nothing +1,170 (only ~454 overlay); missed table lines
  charged nothing +876 (#2 +455 real; #18, #7, #1, #14 search); both +1,972; label-only flag +276 (#3; recall 0.9%);
  search from the table's answer alone +573. Benefit-only and partly table-started. Next check: a full-solve oracle
  (odds 0 on same-code read lines).
- **Class size:** 408 of 2,860 one-code read pairs have an SUD or height change; only 4.0% of one-code pairs with an
  SUD change read as a line. The table's `sud` field is a check, never an input.
- **Past tries:** same-code charge to full odds on 11-11 (leak fix, 10/10 cases); whole-line judging failed four
  times on ink-to-border matching; SUD names dropped as codes (+78 Van Ness lots in the bubble solver); J7's walls on
  the cleaned page, dropped 11-13.
- **Reading:** Kohli, Ladický & Torr 2009 (robust Pⁿ); Delong et al. 2012 (label costs); Chen et al. ICDAR 2021;
  Jiao, Heitzler & Hurni 2022; Chiang, Leyk & Knoblock 2014.
- **Libraries:** PyMaxflow GPL-3.0; scikit-image BSD-3, OpenCV Apache-2.0; MapReader MIT; OpenGM 2 MIT (not
  verified); soduco vectorization licence not verified; gco-v3 research-only and mapKurator CC BY-NC 2.0 rejected.

### Placement and labels

- **Lead: the owner's leader ends in `label_fixes.csv` as places** (hand_place): charge the label once on the lot at
  the fix point, drop its far-side copy, join by position and fixed code. +310 on the 37 sets with no loss (#5,
  #30); +58 city-wide (75 / −17, worst −6). Today hand moves only train the fit (median chance 0.40 for the hand
  answer).
- **Code-key bug (not fixed):** `placement.leader_moves` keys each hand move by the row's old `code`, not `fix_code`;
  18 of 101 hand moves never match. Repro: label 238 (p02 at 2254,1552; RC-2 → C-2). Failing integration test in
  `.superpowers/sdd/2026-11-16-big-incidents/placement-report.md`.
- one_lot / nofar / far_split fix five incidents on their sets but lose city-wide (−220 / −178 / −81): 82% of street
  labels name both sides (355 of 431).
- The code reader into the trust fit after its fix; Dawid–Skene confusions (crowd-kit, Apache-2.0). The pre-fix
  reader: 4 of 12 hard labels right; add_codereader −46 city-wide.
- **label_fixes as input:** fixes −887 city-wide, almost all from one row: p12 (2504,1938) "RH-1 → RM-1, code check:
  Terra and Sol both read RM-1", 4.6 px from print 1723, read RH-1 by all four readers (trust 0.97); the table says
  RH-1 for all 988 lots of the district at 6774023. fixes_people +123 with one wrong join (BROADWAY, −65). Code fixes
  amplify placement errors (728's fix loses #5), so code and placement fixes go together. Deleting 1801 flips the
  Mission sets in confined search (869 → 2,897, 884 → 2,920) but not the sw search from scratch.
- **Reading:** Weinman et al. ICDAR 2017; Budig, van Dijk & Wolff 2016; Dawid & Skene 1979; Rother, Kohli, Feng & Jia
  2009; mapKurator, LIGHT (no code licence confirmed).
- **Libraries:** crowd-kit Apache-2.0; scipy.ndimage BSD; PyMaxflow GPL-3.0; gco-v3 research-only.

## Decisions for the owner

1. Build `compound` (C1) as the next solver change, with its synthetic case first and the gate above? C2 with it
   (−58 wrong lots, −8 incidents, +78 broken near-ties, 5–9.5 min per quarter)? Accept that better search exposes
   walls errors (se 6328/6324/6325)?
2. `label_fixes.csv` as solver input, at least the leader ends; and the p12 (2504,1938) row: stale or wrong?
3. Fix the leader-moves code-key bug.
4. The SUD forgiveness design: should the solver know about printed SUD boundaries between two areas of one code?
5. Re-test `text_discounted_enclosure` per kind of print?
6. Tonight's scripts (`pipeline/scripts/incidents/`, six local commits): join #62 or their own branch?
7. Smaller: update J7's write-up on the Mission incident; unlabelled areas and rotated labels (148 lots) in scope;
   the crossing-profile walls feature.

## The code reader

Task 3 (the posterior) is in a fix round: an extent term for point labels and an honest pruning bound. Task 4, its
score on the owner's 717 answers, runs after it and was not in when this note was written.

## Sources

`.superpowers/sdd/2026-11-16-big-incidents/`: `progress.md` (rulings), `task-1-report.md`, `search-report.md`,
`search-adversarial.md`, `walls-report.md`, `walls-adversarial.md`, `placement-report.md`.
`pipeline/data/solver/experiments/`: `incidents-census/census.md`, `incidents.json`;
`incidents-search-summary.md`, `incidents-search-*/measures.md`; `incidents-walls/`; `incidents-placement/`.
