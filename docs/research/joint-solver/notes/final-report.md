# Bubble solver - final report (2026-11-12)

Gate: leave-one-quarter-out, 152,351 scored lots.  Target 99% = 1,524 wrong.  Chain (c_link2k_kfcv): 3,146.

## Where it stands (honest numbers)
- Bubble today (v21): 4,904 wrong, 96.78%.  Same setup as of yesterday (v13h): 5,718.  Gain today: 814 lots
  (fixed 1,289, broke 475).
- Earlier numbers (v13 4,663, v18 4,292) were too good: the film judgement had been fitted on all four quarters,
  including the one being scored.  Fitted without it, the totals above are what's real.
- Not at 99%, and not yet level with the chain.

## What worked
1. **Lot character as line evidence.** A big lot next to small lots is likely a district line.
   - Held-out per-border log-loss 0.0510 → 0.0472; 175 fewer missed district borders, 128 fewer false lines.
   - Fixed Visitacion Valley (6309): 289 RH-1 lots that came out M-1.
2. **The chain's choice model for empty boxes**, used as link-offer prints. Lots with no label's push fell from
   873 wrong to ~260.
3. **Special Use District names dropped as codes.** "VAN NESS S.U.D." and similar are overlays printed over a base
   district. Fixed 78 RC-4 lots on Van Ness.
4. **Error breakdown by what happened inside the solve** (zdecomp). This found every lever above. Current split:
   - lost inside a walled area: 1,989
   - renamed by a merge: 1,059
   - no right label anywhere: 610
   - right label only beyond a line: 497
   - lower reading: 488
   - unreached: 261

   The no-label and beyond-a-line groups (~1,100) are input problems: the chain gets almost none of them either.

## What failed (each measured, then dropped)
- **Push replaced by the ink path, or charging seen lines in the cut:** no gain. The 6309 cause was the line odds,
  not the energy.
- **Lot size as a per-lot term in the cut:** no gain.
- **Per-lot park prior off in the cut:** fixed 155, broke 1,121.
  - Park lots next to four P labels (blocks 7153–7177) still go to the NC-S label, because the per-lot prior is
    summed lot by lot.
  - The real fix is park evidence judged once per bubble. Not built.
- **Film judgement refitted on the solve's own films:** collapsed the north-west (11,600 wrong; the Richmond became
  park). Two causes, both mine:
  1. Wrong event: I asked "does the code change across the film" instead of "what does a merge do to the lots".
  2. Fitted on films the old judgement produced (off-policy).

  On the right event the old judgement is roughly calibrated, and the refit was mixed.
- **"Weaker label also reads the stronger's code":** a real signal (32% two districts vs 63%), but only 135 pairs, and
  worse held out.
- **Stronger ridge on the pair judgement:** better held out (0.458 → 0.437), same map.

## The structural problem I found
A single label with no ink around it can claim hundreds of lots. Then one film decision moves ~800 lots: in blocks
7103/7094, one RH-2 print on a split lot claims 573 lots. The city total swings by hundreds on such coin tosses, so
small changes cannot be judged by the total alone. They need run-vs-run diffs (zdiff), which is how I judged them
today.

## Biggest mistakes today
1. **Fitted the film judgement to the wrong event.** "Code changes across the film's borders" calls a misplaced film
   between two real districts "not real". It cost two full runs and a false "miscalibration" finding.
2. **Did not check that every fitted input was held out.** The pair judgement leaked. I found it late; all earlier
   bubble totals were optimistic.
3. **Handled summed evidence over correlated lots three ways without first checking how it behaves:** clipped,
   exact, then log.

## Next, in order of expected gain
1. **Lines from the twin scan** (1996-2000-zoning, same drawing, 20+ sheets): independent ink. Not registered in this
   checkout.
2. **Park evidence judged once per bubble**, replacing the per-lot prior in the cut.
3. **Junction-split line pieces**, so a whole line is judged once (the plan from yesterday).
4. **The ~800-lot single-label bubbles:** make a bubble's claim depend on ink support, not distance alone.
