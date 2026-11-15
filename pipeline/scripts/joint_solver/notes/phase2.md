# Phase 2 (2026-11-12): where the bubble's errors come from, and what to fix

Bubble v13 4,663 wrong; chain (c_link2k_kfcv) 3,146; target 1,524.

## Question 1: why is the bubble 1,500 worse than the chain?
Assumption B1: the gap sits in a few mechanisms that the solve's own pieces can name (dark area, merge rename, lower
reading, lost in its area, right label beyond a line, no right label).  Check: zdecomp.py on a dumped v13 run, with
the chain as the second run (bubble wrong where the chain is right, per mechanism).

## Finding so far
- The 736 "right label inside, neighbour's code" lots under perfect borders are mostly park (P) lots: the table's P
  "district" is a chain of parks and freeway lots across the south-east (475 lots); each park walled area with no P
  label inside is named after a neighbour (RH-1, NC-S, RH-2).  So it is the no-label naming again, not a new mechanism.
- zdecomp v13d (4,663 wrong; chain right on how many of each):
  dark 873 (chain right 347); renamed by a merge 704 (389); lower reading 363 (189); LOST IN THE AREA 1,797 (1,150);
  right label beyond a line 428 (4); no right label 498 (12).  Chain wrong where the bubble is right: 574.
  => the competition inside a walled area is where the bubble loses most.  Lines-beyond and no-label are as hard for
  the chain (it gets ~none), so those are input problems; competition, merges and dark areas are method problems.

## Question 2: inside one walled area, what decides which label a lot belongs to?
Assumption B2 (the bubble's): geometry - every label pushes equally (random walk), faint lines only damp the push; the
cut sums -log(share) over every lot.  The chain's: the ink along the cheapest path (each border -log P(no line)),
minus log trust.  Check: zcomp.py - same lots, same inputs (v13d's line odds), which rule picks a label reading the
table code.
Result: the cut alone 77-88%, push alone 73-82%, ink path alone 69-85%, ink path - log trust 90-98% - but the bubble's
FINAL answer on those lots is 96.7-98.5% (films fix misreads), chain final 98.5-99.0%.  The gap is SE: 820 lots
bubble-wrong/chain-right, 589 of them in ONE walled area of 26,032 lots (missed lines everywhere).
Trace (ztrace6309): 126 RH-1 lots of 6309 go to an M-1 label (6309004A, 0.96, a real 5-lot M-1 district).  The cut gave
them to it directly - the RH-1 label is across a street (6305036).  The M-1 district's heavy line has a gap border
(-0.34); push passes 58% there, and the push unary is summed over all 126 lots, so the film leaves the heavy line.
WRONG ASSUMPTION A5 (confirmed): one geometric comparison is counted once per lot.  The ink says the film belongs on
the line (cutting there costs 0.34 nats; cutting through open RH-1 ground costs several nats per border).
Wrong question in the first test: "does the label read the code" mixes extent with misreads.  The extent question is
"is the claiming label printed in the lot's own district" (zcomp4).
zcomp4 (extent, 85,379 contested lots): wrong - bubble cut 3,539; ink path alone 4,684; push alone 5,835; per-lot fit
4,086.  So the per-lot ink path is NOT a better unary; the joint cut wins.  (Hypothesis "swap push for ink path"
rejected by the data.)

## Question 3: the cut's energy
Found in zcore: the bubble calls solve_directed(repw=0): one label on both sides of a SEEN line costs nothing.  Films pay
-odds on open borders but get no credit for following printed ink.  With calibrated odds (A2 validated) the likelihood
says: cutting border e changes the energy by -odds_e, for every border (repw = 1).  The chain uses 0.3.
Assumption C1: repw=1 (the calibrated likelihood) puts films on printed lines (6309).  Caveat (A3): runs along one line
are correlated, so summing per border overcounts long lines (a run of 10 printed borders: sum ~30 nats vs the measured
whole-run rate ~8 nats).
Assumption C2: the push weight 0.967 is a per-lot calibration made without the ink; given the ink, the per-lot fit says
~0.5 (zcomp4 fit: log push 0.43-0.65).
Check: ztestcut.py on the big SE area (26k lots), extent wrong, and the 6309 blocks.
Result: repw 0 -> 1 changes nothing that matters (596 -> 588 extent wrong; 6309 unchanged); push weight 0.5 is worse
(931).  C1 and C2 REJECTED as the cause.  The real cause at 6309 (zborders_at): the line between the 184,000 sq ft M-1
lot and the small RH-1 lots reads -0.6 to -1.5 per border (thin ink, 1-5.5 px).  A line-reading problem.

## Question 4: is there evidence for lines that is not ink?
Assumption D1: districts separate lots of different character (big industrial/institutional vs small homes).  Check:
zborderfit2 - per-border model + |log area difference|, smaller lot's log area (held out by quarter).
Result: log-loss 0.0510 -> 0.0476 (+wall 0.0472); missed district borders 2,817 -> 2,642, false 2,382 -> 2,254.
At 6309 the M-1/RH-1 borders go from -1.5..-0.6 to +1.3..+2.1.  Validated as evidence.  (Whole-wall mean odds: small.)
Run v14lots: bubble with these odds (weights fitted without the run's quarter).

## Question 5: dark lots (no label's push reaches) - 984 scored, bubble wrong 873 (89%)
ink-path naming 883, + park 865 - no better.  Chain base (no links) 775; chain with links 571.  The chain's links
(relink3: a fitted choice model per empty box over every print within reach + "none") fix ~200.
Dark wrong lots: area 6-30 lots 412, 2-5 lots 332; a label of the right code within 150 ft for 229, 150-400 ft 326,
none within 400 ft 318.  Run v14m0: bubble with the chain's link-offer print set (M0).

## Runs
v14lots 4,502 (-161); v14m0 4,490 (-173); v15 both 4,371 (-292).
v15 breakdown: dark 256 (was 873), renamed by a merge 1,187 (chain right 783), lower reading 457, lost in the area
1,353, beyond a line 496, no right label 622.

## Question 6 (A12): is the film judgement calibrated on the films the solve judges?
zfilmcal v15: NO.  Stated 0.20-0.35 -> 75% real, 0.35-0.50 -> 85%, under 0.02 -> 17% (21,504 borders).  The pair fit
was made on label pairs, not on the bubble's films (different distribution, and the odds changed since).
Fix: zfilmfit - refit on the solve's own films (features at the moment of judging + park evidence + whole-film ink sums
+ film length + bubble sizes), outcome = share of the film's borders where the code changes, held out by run quarter.
Held-out log-loss per border: nw 0.188 -> 0.109, ne 0.182 -> 0.114, sw 0.425 -> 0.128, se 0.416 -> 0.160.
Real lines dissolved (borders): 488/599/449/614 -> 102/92/46/103.  Run v16 with it.
v16: 11,633 wrong - NE better (1,587 -> 1,350), NW collapsed (6,075).  Trace: a 5,033-lot RH-2 group merged with a
small P bubble and took the name P (P's trust highest).  MISTAKE: I turned the exact park evidence (a log-likelihood
ratio over thousands of lots, worth thousands of nats) into a clipped feature (+-3) and fitted it on films the old
judge produced, which never included such a merge.  Off-policy: a fitted judge meets the films its own decisions make.
Fix (v15g -> ...): park evidence back as an exact offset (weight 1, never clipped); lot-size evidence of the lots that
would be renamed (evc) as a feature; fit on pooled runs (the old judge's films and the new judge's) and iterate until
the films judged stop changing; check calibration each round.
v17 (park ev as exact offset, then as log feature): NW collapses again (6,075).  Log trace: films P 0.99 vs RH-2 0.98
along 2-6 blank borders dissolved (odds 0.05-0.6).  CAUSE: wrong EVENT in the refit ("code changes across the film's
borders"): a misplaced film between two real districts scores 'not real'.  Re-fitted on the decision's consequence
(merging renames the losing side's lots: more of them carry their own code than the keeper's?).  On that event the
OLD judge is not badly calibrated; refit: held-out lots lost to wrong calls 2,769 -> 2,546, mixed by quarter (nw/ne
worse, sw/se better).  Plateau rule: stopped on the judge.  The earlier "miscalibration" (zfilmcal) was mostly the
wrong event.
Lost in the area (v15, 1,353): P->NC-S 245 (blocks 7153-7177, 5376-7), P->RH-1 172, RM-1->M-1 101 (4591B), NC-3->RH-2
91, RC-4->VAN NESS S.U.D. 78.
- Overlays: 114 lots named with codes the table never uses (VAN NESS S.U.D. 82, O.S. 29); chain 5.  The legend list
  (37 codes) misses the named NCDs (2,857 lots), so it cannot filter.  Rule used: a Special Use District name (SUD,
  S.U.D.) is an overlay, never a base code -> those readings dropped (NOSUD).  Run v18.
- 7174: the P label's lot is NOT walled (all borders -3..-6.5); its P neighbours go to the NC-S label because the per-lot
  park prior (small lot off the park list: ~0.5% -> ~5 nats against P each) is summed lot by lot - the same correlated
  stacking again.  Isolated test: park weight 1 / 0.5 / 0.25 / 0 (ztestcut).
  Result: park weight 0.25-1 identical (87 P lots -> NC-S); weight 0 gives the P label all 90 P lots + 32 RH-1 lots.
  A switch, not a weight.  City-wide PZW=0 (v19): 4,292 -> 5,258 (fixed 155 incl. P->NC-S 119; broke 1,121: RH-1->RH-2
  574, RH-1->P 309).  Reverted.  The principled fix is park evidence judged once per bubble (correlated lots, sublinear),
  not per lot in the cut - a larger change, not done yet.
v18 (NOSUD): 4,371 -> 4,292.  Kept.
Alternate reading (the weaker print also reads the stronger's code): real signal (two districts 32% vs 63%) but only
135 pairs; held-out log-loss worse in every form (0.458 -> 0.473-0.492).  Rejected.
Found: the pair-fit judgement is over-fitted (ridge 0.1 -> 10: held-out 0.458 -> 0.437) AND the runs used weights fitted
on all four quarters (a leak).  v20 (ridge 10, per quarter): 4,893 - fixed 394 / broke 995, of which 804 RH-1->RH-2 in
7103/7094/7062: ONE single-label bubble (RH-2 0.94 on a split RH-1/RH-2 lot, 573 lots) whose film the old judge merged
(right by luck) and the new one keeps.  Fragility: one weak decision moves ~800 lots; the city total swings by hundreds
on it.  Not tuned around (plateau rule).  v21: the original judge, fitted without each quarter = the honest number.
Twin scan (1996-2000-zoning, same drawing, 20+ sheets): independent ink, the strongest unused line evidence - but not
registered in this checkout (page thumbnails only).  Not started.
Also: lot size as a per-lot unary in the cut adds nothing (ztestcut 238 -> 246/241); with the v15 line odds 6309 is
already right (139/139 RH-1).

## Owner's area 1 (2026-11-12): Golden Gate Park / Kezar is RH-2
- The P/RH-2 line along Fulton (blocks 1662-1687, about 10,200 ft) is on both sheets and survives the clip, but is
  mostly gone from the cleaned merged page (k8/seam_fulton_3panel.png). The sheet edges run along it.
- All 139 park borders there read open: ink odds about -4.8, final odds -1 to -3.
- The cut: keeping the park P would cost 236 nats of open borders, against a cap of 50 for a label lot to give up its own
  label. So park lot 1700001 (unscored, own label P 0.98) went to the RH-2 label on 1676001D, and the P bubbles merged
  into the Richmond's RH-2 district.
- Kezar lots 1264001 and 1263011 (table P) were dark. The no-label rule joined them to that district (chance 0.75); it
  ignores the park evidence (83-88% P).
- Three weaknesses:
  1. A line lost in cleaning at a sheet join (input).
  2. Open borders along one missing line summed as independent votes against a label lot (correlated stacking again).
  3. Dark naming without park evidence.
- How the Fulton line went missing (k8/zjoinzoom.py, join_fulton_zoom.png; stacks: stack_fulton_raw.png, _clean.png):
  - The line is on p04's raw page and survives p04's cleaning.
  - p04's drawn-area edge runs down the middle of the heavy line. p05's drawn area starts about 50-60 ft further south,
    and p05 does not print the line. The strip between them is covered by no sheet.
  - mosaic.py keeps a sheet's ink only inside its drawn area, tested on a 32-ft grid (STEP=8 at 4 ft/px).
  - Of p04's ink in the window: 57% inside its drawn area exactly, 42% as the 32-ft grid sees it, 40% kept in the merged
    page. What survives is a half-width, broken line; the line reader gives about -4.8 there.
