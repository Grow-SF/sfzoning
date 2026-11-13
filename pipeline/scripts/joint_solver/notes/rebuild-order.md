# Rebuild on the new merged page: the order of steps (for the owner to check)

Every step is redone unless marked CARRY.

## Carried over: these never saw the merged page
- Each sheet's scan and its fit to the map.
- Each sheet's cleaned page (block lines and letters erased, per sheet).
- Each sheet's text boxes.
- The label reads (cached answers).
- The parcels and the April 2000 zoning table, with the scored-lot list.
- The city outline: the grid's extent only.

To confirm: two numbers measured on sheets, not on the merged page.
- The heavy/thin stroke split, from p06.
- The letter-O size, for removing small loops.

Keep, or re-measure on the new page?

## The old process (the chain)
1. **Merged page:** every sheet cropped to its frame, notes cut out, ink from any sheet (zmerge.py; replaces mosaic.py).
2. **Graph of the page:** strokes found, then heavy vs thin lines (vectorize.py, zonegraph.py x2).
3. **Small loops removed:** loops smaller than a letter O (small.py).
4. **Line ends joined across gaps** (graph7.py).
5. **Letter probe:** the strokes that lie inside printed text boxes, to learn what a letter stroke looks like (textlines2.py).
   - TO CONFIRM: the whole-line step also reads a saved letter-stroke measurement (meas8t_city.npz). I believe it comes
     from this probe and must be remade; I haven't found its maker yet.
6. **Whole lines judged with their neighbours** (cityedges8.py, then graph7.py).
7. **Closed-area rule** (cityloops2.py, then graph7.py).
8. **Text-label step:** lines that are really letters dropped (textlines2.py, then graph7.py). Result: the city graph
   "city5".
9. **Wall grid:** each lot border and whether a seen line runs along it (lotwalls.py, from city5 and step 6's lines).
10. **Labels to lots:** each label read placed on its lot (labread2.py).
11. **Print classifier** (lotclf6.py).
12. **Street side of each print** (labside.py).
13. **First solve** (ceil.py "base").
14. **Gap closings:** each loose line end's shortest way to another heavy line (gapfix.py).
15. **Area odds, round 1:** areafix.py with the step 13 solve, then solve again (ceil.py).
16. **Round 2:** the same with the step 15 solve.
17. **Street names:** a text box read as a street name between two lots (nameink.py).
18. **Lines lost by the earlier steps that cross a border** (cross.py, onepair.py).
19. **Round 3:** areafix.py with steps 16-18 -> loarea5.
20. **Lost heavy lines measured per border** (cross2.py).
21. **Line odds, each quarter held out** (pairfit.py) -> loarea6wx{q}.
22. **Print trust, each quarter held out** (addprints.py) -> the print sets.
    - TO CONFIRM: the base print set "cw" was itself built by earlier addprints steps; I'm still tracing them.
23. **Each quarter held out:** street sides -> solve -> link offers for empty boxes (relink3.py) -> linked solve.
    - The solve starts from a saved earlier answer (INIT); the rebuild makes that answer anew, never the old one.
    - The count of real prints (OWNL, 1906 today) is recounted, not copied.
24. **Join the quarters and score** -> the chain's number (old page: 3,146 wrong).

## REVISED with the owner (2026-11-12): the graph's judgements move into the joint inference
- **Measured, held out by quarter on the old page** (log-loss):
  - direct ink only: .0909;
  - graph + gap closings: .0674;
  - + area rounds: .0567;
  - + names and crossing: .0553;
  - + pairfit: .0528;
  - today's full model: .0472.
- **Prints carry:** labread2, lotclf6, labside and addprints use only the lot grid and the table (checked in the code).
- **Order:**
  1. merged page;
  2. strokes: a description only, nothing removed;
  3. evidence on each stroke and each possible closing, as likelihoods, one fit per quarter;
  4. joint inference per quarter: lines <-> codes, ink evidence fixed, calibration checked each pass;
  5. score.
- Build on the OLD page first and compare with today; then the new page.
- **Step 2 done (zstrokes.py; nothing dropped, short dead ends tagged):**
  - old page: 84,130 strokes, 1,315 mi;
  - new page: 83,934 strokes, 1,323 mi.
  - Map tool layers "so" and "sn".

## (superseded below) The new process (the bubble solver), on top of steps 9-23
25. **Stroke width and steadiness per border** (zstraight.py); the border evidence table (zborder.py).
26. **Border odds with lot character and the whole wall** (zborderfit2.py), each quarter held out.
27. **The bubble's own fits, each held out by quarter:**
    - push calibration (zpushcal2.py);
    - naming of areas no label reaches (zfitnolabel.py);
    - lot size per code (zlotchar.py);
    - the film judgement (zpairfit pairs -> zpairfit4b.py);
    - the park prior (fitted inside the run).
28. **Bubble run** with v21's settings, quarters joined and scored.

## How I'll know the rebuild is honest
- First replay steps 2-28 on the OLD page into a separate folder. It must reproduce today's numbers (chain 3,146; bubble
  4,904) before anything runs on the new page.
- Then the same recipe on the new page, with no file from the old folders read.

## Step 4: the loop (built 2026-11-12, owner: "sg")
- **Holdouts:** 4 quarters (scored) + 6 pairs.
  - Everything in a holdout's run is fitted outside it: print trust (cwfwx{H}; pair sets by zpairprints.sh), odds
    (zloop_odds.py), push calibration (zpushcal2 HOLD mode), no-label naming (znolabel + zfitnolabel HOLD mode), merge
    decision (zpairfit4 HOLD mode, ridge 0.1) and the park prior (zbubreal's fitm; fitarea now takes pairs).
- **Codes feature (pass 2+):**
  - For quarter Q, a training border in quarter X takes its codes from the run that held Q and X out; Q's own borders
    from Q's run.
  - For a pair, from its own run (a second-order approximation).
- **Leaks found in v21's setup:**
  - The push calibration was fitted on all four quarters (w = fit on all rows).
  - The no-label naming was fitted on all rows.
  - znolabel's majority codes counted every quarter's table.
  - The loop fixes all three.
- **Bootstrap:** pass 1 = stroke odds only (no earlier solve, no old fit).
- Driver: k8/zloop_pass.sh EV N [PREV]; runs tagged L{N}.
- **Pass 1 (L1, stroke + line + nearest-closing evidence): 11,800 wrong** (NW 1,255, NE 4,552, SW 4,123, SE 1,870).
  - Odds held out per quarter: nw .054, ne .084, sw .046, se .054.
  - Broke 7,835 vs v21: RH-1 -> RM-1 1,791 and RH-1 -> RH-2 1,164 in the Sunset.
  - Traced (ztracelots, zwallpic wall_L1_rm1.png): a 32-lot RM-1 district by the zoo is not closed.
    - Its west line runs across the beach and the Great Highway with no ink.
    - The old chain closed it with gapfix closings (shortest way over street ground and lot lines to another heavy line).
    - The RM-1 label's push took 1,544 RH-1 lots in the cut.
- **zclose.py:** gapfix-style closings on the strokes (MCP over street ground and lot lines); 46,020 closings, median
  36 ft.
- **zenclose.py:** separation at 16 levels (stroke width x closing length) + the nearest closing path.
  - Odds: strokes .0782 -> .0736; +wall .0628 -> .0617 (= the old graph+gapfix stage .0614).
- Pass 1 again with them: M1.
- **M1: 7,874 wrong** (NW 1,263, NE 2,533, SW 1,998, SE about 2,080).
  - Vs v21: broke 4,055 (spread out), fixed 1,085.
- **M2 (pass 2, with M1's codes):** held-out odds improve modestly.
  - nw .0542 -> .0524, ne .0808 -> .0772, sw .0424 -> .0395, se .0529 -> .0507.
  - But 9,097 wrong: the zoo RM-1 district took 1,206 Sunset lots again (SW 1,998 -> 3,164).
  - Its west side is open, so small changes to the fits flip a 1,200-lot area.
  - The loop cannot settle while one label can claim a region through an open side.
- **Fix: the city limit is a hard edge** (owner's rule). Closings may end on the city outline (zclose target S), and the
  outline is a wall at every enclosure level (zenclose).
  - Rebuilt as zclose2_old / zstrokeev4_old.

## Gaps shown to the owner (2026-11-12), and a bug they exposed
- N1's three largest incidents each leaked through a stretch of table line with no wall (zgapfind.py, zgappic.py;
  pics/gaps/gap1-3.png):
  1. 1,202 lots RH-1 -> RH-2 (blocks 7103/7094): 220 ft at -122.4506, 37.7226; a heavy 25-32 ft line printed along it.
  2. 449 lots RH-1 -> RH-2 (blocks 7069/7092): 692 ft along a heavy street line at -122.4641, 37.7140.
  3. 258 lots RH-2 -> NC-2 (blocks 5882/6050): one 152-ft border at -122.4060, 37.7321, fully inked, 33 ft line.
- At the loosest enclosure level all three are separated (zleakpath.py), yet zenclose said 0, 0 and 8 of 16 levels.
- **Bug (zrepcheck.py):** zenclose placed each lot by the pixel farthest from any street = on the lot's own edge for 97%
  of lots (median 0 ft in), i.e. on the back lot line where mid-block district lines are drawn.
  - Loosest level, table-line borders separated: 61.8% (old placing), 91.6% (own innermost), 95.0% (majority region).
  - Borders with no table line separated: 4.9%, 3.8%, 3.4%.
  - Fixed: majority region.  Rebuilt zstrokeev5_old; pass 1 again = P1 (run_p1.sh).
- Odds split (zoddsparts.py): on all three gaps "lot character" (touching lots of similar size) pulls -1.4 to -2.1.

## Owner, 2026-11-12: one model learned from the table, applied to every year (notes/one-model-design.md)
- The loop must not learn from its own solve; P1 was stopped (a whole-city run with no synthetic case).
- Case A numbers measured (zcaseA_measure.py), on table lines between touching lots (334 of 632 mi):
  - lot width median 25 ft (10-90%: 25-59), depth median 100 ft (74-138);
  - kind by length: back to back 47%, side by side 21%, between 33%;
  - printed line from the lot line: median 4 ft (75%: 8, 90%: 11), spread along a border median 2 ft;
    91% of the length has it within 16 ft;
  - printed line width median 24 ft (24-32);
  - stray ink: 11% of lots with no table line have a stroke >= 6 ft wide inside, median 24 ft long;
  - rates the sampler must reproduce: on lines the majority placing finds closed, the old placing separates side by side
    41%, between 49%, back to back 39%, across a street 88%.

## Case A result (2026-11-12, zcasesA.py; real 1,500-ft windows, made-up printed ink, 300 draws)
- Inputs: street classes from a refitted width mixture (zstreetclass.py; boulevards 120 / 125 ft kept, better BIC);
  offsets per situation as fitted mixtures (zcaseA_offsets.py); lines drawn by moving the true district map border by
  border (closed by construction).  Placing through zplace.py (zenclose.py uses the same code).
- Realism: the old placing reproduces the real rates for touching lots (side by side 42.9% vs 41.1%, between 51.5% vs
  48.5%, back to back 39.5% vs 38.5%).  Across streets, 2-6 points low in most classes; 10-16 low on 82, 104, 125 ft.
- District-line borders separated: old 56.2%, majority region 98.6%, best achievable 99.1%.
- Other borders wrongly separated: old 6.23%, majority 0.42%, best 0.15%.
- Without stray ink: majority 98.5% / 0.24% (best 99.1% / 0.15%).  So stray ink is about half the extra false
  separation; the rest is not yet traced (window edges likely part of it).

## Case 1 (2026-11-12): a mostly walled area flipped through a short gap
- Q1's two big flips (3,527 and 1,237 lots) traced (ztracelots.py, the SE replay's dissolve log):
  - Films between groups of DIFFERENT codes dissolved after merges.  The decisive one: 84 borders, 98% printed, trusts
    0.98 / 0.98, odds 0.068.
  - Merged groups take their single best label's code, so the 8,735-lot group was named RH-3 on a 0.98 tie.
- Cause in the film judgement (zfilmdiag.py): the term 'trust x mean -odds of the OPEN borders' (-9.5 here) ignores how
  much of the film is open.
- First-drawn films (zfilmrate.py): mostly printed with 1-2 open borders are 100% real, judged 88%, 9% dissolved.
- Fix tried: feats_whole (the open borders' -odds averaged over the whole film), refitted per holdout (zpw_W1x*).
  - On the training pairs (min cuts) it ties: .2721 vs .2729 held out.
  - Case 1 replay (zcase1.py; 8,783 real films, no P land):
    - case-1 films 92.0% -> 96.7% right (best 98.1%); the 12 the real solve dissolved: 0% -> 75% kept;
    - but everything else 97.7% -> 96.7%: it broke 80 (72 fully printed, real), fixed 15 (zcase1_broke.py).
- Root cause: the judgement is trained on min cuts inside one walled area, which ALWAYS cross an open border.  It has
  never seen a fully printed film or a long merged one, yet the solve asks it about thousands; both fits extrapolate.
- Not adopted; W1 whole-city not run.  Proposed: train the film judgement on the films the solve actually judges
  (truth from the table, held out by quarter).
- Owner: "do that training; build your synthetic test cases on real examples in the data".
  - zfilmtrain.py: the film judgement (zbubble.FILMW, all recorded film features) trained on the films the solve judged
    in Q1.  Each film is from its own quarter's run, truth from the table; quarter q trained on the other three.
    - Held-out log-loss .353 -> .084; decisions 95.6% -> 97.6%.
    - Case 1 films 93.1% -> 98.3%; fully printed 97.4% -> 98.7%; unprinted 83.9% -> 92.3%.
  - Replay of held-out real films (zcase1.py INQ=1; zones sized to the real bubbles; scored on whether the two labels end
    in one district):
    - case 1: 73.9% -> 91.3% (long films 69.2% -> 100%); unprinted 93.0% -> 94.0%; the rest 97.7% -> 98.2%;
    - P films 95.6% -> 97.5%; all films without P land 96.8% -> 97.7%.
  - Whole-city pass W2 (Q1 + the film-trained judgement, run_w2.sh).
- W2 (film-trained judgement): 820 incidents, lines found 86.6%, false 25.5 mi (best so far); Q1's two big flips gone.
  - But 12 incidents of 100+ lots (6,508): RH-1 -> P 1,891 / 1,492 / 839 / 558 lots.
  - Traced: a P label (0.98) on an RH-1 lot.  Its film with four RH-1 labels (0.98-0.95) dissolved, correctly (both
    sides RH-1), then the merged district was named by its single best label, a P 0.98 tie.
- Owner (design correction): never flip on a single most trusted label; two trusted labels of different codes mean a
  missing wall.  Restated, waiting for the go.  Red team: two sure labels (0.95+) with no ink are separate districts
  68% of the time (19 pairs), 22% at 0.80-0.95 (106).  But a wrongly kept wall costs one bubble, a wrong merge a district.
- Owner: "a good new direction. Go validate it."  Built as zbubble.RULE = "cost" (BUBRULE=cost):
  - for a film of different codes judged unlikely to be a line (p < 0.5), take the cheapest of: keep the wall, g1
    absorbs g2, g2 absorbs g1.  The cost is the lots expected wrong, from p (the film-trained judgement), qB (which
    label is the misread, from the two trusts) and the two group sizes;
  - with equal trusts a merge never wins; a group keeps its own code, so a joining label never renames it.
- Case 2 on real examples (zcase2_set.py): W2's 'mixed' districts (labels of different codes merged), about 145 per
  quarter, 96k lots, holding 7,596 of W2's 10,085 wrong lots.  Re-solved alone (SUBLOTS) with W2's rule (C2o) and the
  cost rule (C2n); scored with the best achievable from the bubbles (run_case2.sh, zcase2_score.py).
- Case 2 result (W2's mixed districts re-solved alone; scored lots wrong): W2 7,596; C2o (W2's rule) 7,657; C2n (cost
  rule) 4,741; best achievable from the bubbles 888.
  - By quarter C2o -> C2n: NW 661 -> 630, NE 573 -> 572, SW 3,986 -> 1,136, SE 2,437 -> 2,403.
  - The SW single-label P flips (1,891, 839) are gone.
  - Left: 1,491 RH-1 -> P (SE) and 558 (SW).  A misread label's bubble (NC-1 on RH-1 lots) absorbed by a trusted P
    district; the true code has no label there (close to the owner's deferred 'walled areas with no label').
- Whole-city W3 = W2 + cost rule (run_w3.sh).
- W3 (W2 + cost rule), whole city: 6,292 wrong in 817 incidents (7 of 100+ holding 2,869), lines found 86.9%, false
  26.9 mi.  Best bubble run on incidents and lines; W2 -> W3 removed 3,793 wrong lots.
  - Left: RH-1 -> P 1,492 / 558 / 217 lots: a misread label's bubble absorbed by a trusted P district.  The per-lot park
    chance (0.31% for a 2,500 sq ft lot off the list) enters the film judgement only on a squashed log scale, and the
    cost rule assumes absorbed lots come out right when the film is not real.
  - Proposed next case: cost a merge by the per-lot chance that the absorbed lots carry the keeper's code.

## Root cause analysis (owner: "one label ... takes over an entire other area ... crazy and wrong"), 2026-11-12
- zleakanalysis.py over the 34 flips of 100+ lots in W3, W2, Q1, P1:
  - 27 flips (15,300 lots, 91%) were correctly claimed first by their own right-code label (own labels present in 25),
    then renamed when ONE film with a neighbour dissolved: an all-or-nothing group decision.  The neighbour was P in
    13; open edge often large where the 2000 print draws no park line (73%, 84%); some small (1-12%); some 0% (a
    misplaced / misread label inside).
  - 7 flips (1,580 lots): push through an opening into an area with no right-code label.  The random-walk shares never
    fade, so the label behind the opening claims everything, whatever the opening's width.
  - 8 of 34: the taking label's own lot carries a different table code (wrong where printed).
- Owner agreed direction 1: replace the merge.  First form, nomerge (zbubble.RULE): labelled groups of different
  codes never merge; each keeps its own cut.  Case: run_case2x.sh (C2n2 cost rule vs C2x nomerge, merged spellings),
  scoring what nomerge breaks (misread bubbles merging used to fix).
- Direction 2 (later): push that fades by metres travelled.
- nomerge result (case, mixed districts): C2n2 4,740 wrong -> C2x 18,775.  Fixed 1,391, broke 15,426, ALL in bubbles whose
  label does not match most of their lots.  Not adopted.
- Why those bubbles are wrong (zoverflow.py on C2x):
  - 266 bubbles (17,194 lots): the label's code is not even its own lot's table code, and the bubble holds 0 lots of
    the label's code.
  - Overflow of a label that is right where it sits: 13 bubbles, 679 lots.
  - 197 of the 266 labels sit within 400 ft of a real lot of their code (94 within 150 ft), holding 16,006 of the
    17,194 lots: labels linked to a lot in the neighbouring district, then pushing equally (trust plays no part) until
    they meet another label.  Median trust 0.56; 110 under 0.5 hold 7,283.
- Root cause chain: label placement -> equal push floods the wrong district -> merges patch it -> wrong merges = big flips.
- Proposed (owner to confirm): 1. place each label in the district it names (its lot vs the lots just across the
  nearest lines), learned from the table; 2. a label that still does not fit should not flood its area.

## Step 1 of the cutoff clean-up (branch bubble-rnd/raw-profile), 2026-11-12
- zprofile.py: each border sees every nearby stroke (distance to the nearest stroke at each width level 0/6/12/18/24/32 ft,
  with letter share and length); NOWALL=1 in zloop_odds.py drops the whole-wall stage.
- Held-out log-loss nw / ne / sw / se:
  - cleaned, no wall: .0489 / .0754 / .0351 / .0499; + profile: .0484 / .0746 / .0348 / .0498;
  - raw, no wall: .0499 / .0812 / .0385 / .0607; + profile: .0484 / .0795 / .0370 / .0597;
  - with the whole-wall stage (earlier): cleaned .0446 / .0710 / .0358 / .0462.
- Gate (raw >= cleaned in every quarter) fails; SE gap .010.  Next: the profile for whole lines and closings too.
