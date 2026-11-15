# One model for district lines (design for the owner to check, 2026-11-12)

## Goal
- **One model, learned once** from the April 2000 table on the 2000-digital-zoning map. It gives every lot border the
  chance that it is a district line.
- **Applied unchanged to every other map year.** On each map, codes and lines are worked out together with the model's
  parameters fixed. Nothing is re-fitted on a map's own solve.

## Where everything comes from
- **Truth:** the April 2000 table. A border is a district line where its two lots' codes differ (both lots in the table).
- **Judging:** held out by quarter. The final fit uses all of 2000.
- **Transfer check:** the 1996-2000 scan of the same drawing, judged by the same table, with no re-fit.

## The model, per border (two lots touching, or across a street)
**Hidden:**
- L: district line or not.
- k: the way the border shows on the page, a mixture component.
  - The number of ways is learned.
  - For a line I'd expect ways such as: solid along the border; broken; under printed text; not printed or lost in the
    merge.
  - For no line: blank; a letter or number; another line passing near.
  - None of these are hand-set: the fit finds them.

**Prior, P(L | context), learned:**
- Context:
  - touching, or across a street of a given class. The class comes from the street-width mixture (zsizemix.py street:
    peaks at 35, 40, 50, 55, 60, 64, 66, 68.75, 70, 80, 82.5 and 100 ft; alleys around 20 ft; wide avenues spread above
    ~85 ft).
    - Boulevards are kept as their own class (owner, 2026-11-12), though they cover only a couple of streets: the
      ~120 and ~125 ft peaks, likely Market Street and Van Ness Avenue.
    - The first fit merged them into the wide spread. Refit on every street border, starting from those peaks, and keep
      the class even if the fit score drops it.
  - back or side lot line;
  - both lots' sizes;
  - the map's own ink around the border: how dense the lines are nearby and how thick.
- Owner, 2026-11-12: the chance that a lot is its own district changes by era, and follows the density and thickness of
  the lines on the paper. So the ink-density context is what lets one prior serve every year.

**Mixture, P(ink | L) = sum over ways k of share(k | L) x pattern_k(ink), learned:**
- The shares and each way's ink pattern are both learned.
- Within one way, the ink measurements are taken as independent. What ties them together is the way itself.
- **The shares depend on the border's situation** (owner, 2026-11-12: learn the chance that dark ink on a street of a
  given size is a real border).
  - Situations: touching side by side / between / back to back, or across a street of each width class, boulevards
    included.
  - Each way's ink pattern is shared across situations. Only how often each way occurs differs.
  - Small classes borrow strength from neighbouring widths, so rare ones like the boulevards (about 600 table-line
    borders) still get stable shares.
  - Output to show: per situation, the held-out chance that a border is a district line given dark ink along it, and
    given none.
- This stops one printed line being counted three times. Today it is: strokes near, whole line, and enclosure.

**Ink measurements per border:** a short list in place of today's 160 hat features.
1. Coverage: the share of the border's length with a stroke near it.
2. That stroke's offset from the border.
3. Its width.
4. Its letter share (from the text boxes).
5. Continuity: one line along the whole border, or pieces.
6. Enclosure: the share of the 16 wall levels at which the two lots end up in different regions (majority-region
   placing, after case A passes).
7. The nearest possible closing: its length and whether it carries the line on.

**Result per border:**
- Posterior odds = prior odds x the ink's likelihood ratio.
- An ink pattern that non-lines almost never show gets a large ratio, and outweighs the lot prior. Today's fit can't do
  that: it adds a fixed weight per feature, so a strong prior can beat solid ink.

## Fitting
- In training L is known (from the table) and k is hidden, so it's a mixture fitted within each class (EM). That is well
  posed.
- The prior and the mixture are fitted together, by maximising the joint likelihood of (L, ink) given context.
- The number of ways and the context terms are kept only if they improve the held-out likelihood.

## Reading a map with it (inference, no learning)
- **Lines follow from codes:** a border is a line exactly where the codes differ.
- **A map's reading** is the set of codes that best explains, together:
  - each border's ink and context, through the model;
  - the labels' readings, through the print trust (also learned from the 2000 table).
- **The solver** is the bubble solver, as now. It takes the border odds from this model.
- **Dropped:**
  - the "whole wall" second stage, which fed the model's own odds back in;
  - the "codes differ from the last pass" feature, which learned from its own solve.
  - Continuity along the whole line stays, as an ink measurement.

## Judged by
- **Held out by quarter on 2000-digital:**
  - first, the odds' log-loss and calibration by situation (for example, a solid printed line between two similar
    touching lots: does the model give it the real rate?);
  - then the solve's error incidents and lines.
- **Transfer:** the 1996-2000 scan, same table, no re-fit.

## Tests before any whole-city run (synthetic, numbers from the real map)
- **A. A lot placed on the wrong side of its own district line** (geometry; the bug the gap pictures exposed).
  - Numbers: zcaseA_measure.py.
  - Fix: each lot goes to the region holding most of its pixels.
- **B. The learner recovers what made the data.**
  - Borders are generated from known priors and mixture shares, with values taken from a real-map fit.
  - The fit must return those values and give calibrated odds, including in the gap situation (solid line between
    similar lots).
- **The earlier solver suite (zcases.py) still passes.**

## Left to me (checked by held-out likelihood)
- The simple distribution within each way (for example Beta for coverage, log-normal for width).
- The starting context terms (the list above); a term that doesn't help held out is dropped.

## Not now (owner, 2026-11-12)
- Scaling the inputs to each map's print: the maps are consistent year to year. Revisit if the transfer check fails.
- Era differences in district size beyond what the ink-density context carries.

## Measured 2026-11-12 (held out by quarter, log-loss on table borders; lower is better)
- The enclosure fix alone (today's fit, zstrokeev4 -> zstrokeev5):
  - NW .0519 -> .0456, NE .0810 -> .0705, SW .0409 -> .0361, SE .0514 -> .0462.
- The generative one model as designed (prior x mixture of ways, 8 measurements, K=8; trained NE+SW, tested SE): .0956.
  - Found and fixed a double count: the prior's ink density included the border's own line. Fixed, it scores .0931 and
    is better calibrated, but still about twice today's .0462.
  - Trees (HistGradientBoosting) on the same 8 measurements + context: .0517.
  - So the loss is the generative form, not the inputs.
- Trees on today's 160 features + context, plus the whole-wall step, against today's fit:
  - NW .0443 vs .0456; NE .0722 vs .0705; SW .0318 vs .0361; SE .0450 vs .0462.
  - Missed 2,780 vs 2,690; false 2,196 vs 2,501.  A near tie.
- The three gap borders: gap 2 -2.78 -> +2.62 (today's fit) / +1.80 (trees); gap 3 -1.95 -> +1.43 / +1.84.
  Gap 1 is still open: -3.79 -> -2.84 / -1.95.

## Revised design (owner: "try it", 2026-11-12)
- One model = the plain average of two learners, both learned from the table with the whole-wall step:
  - today's logistic fit;
  - gradient-boosted trees on the 160 features + situation (street class / touching kind), lot sizes, ink density.
  - Weight checked, not tuned: 0.4 / 0.5 / 0.6 on the trees give .0457 / .0456 / .0457.
- Held out, all quarters: .0456 vs today's .0494; missed 2,652 vs 2,690; false 2,189 vs 2,501.
- Case B (revised): per situation, the held-out chance given dark ink / no ink matches the real rate within a few
  points (pics/onemodel_bysit.png).  Overall calibration is slightly over-sure at the low end (says 0.09%, real 0.21%).
- Gap 1 traced (zgap1.py): printed text sits on the district line.  The nearest ink is an 8-ft stroke with letter share
  0.63, and the two lots are tiny (2,071 / 3,054 sq ft), cut into pieces by text and line; separated at 3 of 16 levels.
  -> its own failure kind: needs a synthetic case (case C) before any fix.
- Whole-city pass 1 on the fixed evidence: P1 (today's fit) and Q1 (the one model), judged with N1 by incidents and
  lines (run_wholecity.sh).
