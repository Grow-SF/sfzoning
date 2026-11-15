# Lessons learned (for my own use) - 2026-11-11

## The mistakes, in the order I made them

1. Built things the owner did not ask for and presented them as his: light that leaks through seen walls; per-parcel fade;
   a textbook discretisation I never said was mine.
2. Jumped to building when he wanted to talk; long multi-topic messages; questions he could not tell I was asking.
3. Hand-picked numbers in tests that contradicted my own measurements (every gap = -4.6, "clearly no line").
4. Derived a judgement by multiplying rates (whole-line miss model) instead of measuring the outcome directly. The
   measured outcome said the opposite (91% vs my 0.4 odds).
5. Fitted straight-line (additive) models where the effects interact; then a kNN that washed out the one corner that
   mattered. Only a model shaped like the data (trust curve x film) fitted.
6. Calibrated the wrong question (does a label's push reach a lot) instead of the one the cut uses (which of two pushes
   wins). Flattened the push, broke things, measured again, reversed.
7. Chased a symptom (parks swallowing homes) through four patches instead of the cause (a missed line between them).
8. Judged lines border by border - two pixels at a time - so a plain 300-pixel stroke read as a row of coin flips and the
   wall was full of holes. The owner said "whole line" many times before I built it; then I built it with the wrong unit
   (long strokes that run past where districts change), without first checking the unit was pure.
9. Guessed limits again (5 px stroke cut), which silently dropped the Fulton line.
10. Claimed results from flawed measurements ("89% missed as whole stretches" from a walk that cannot see corners; the
    stretch list mixed table artifacts - Clarendon condos - with printed-but-dropped lines).
11. Guessed causes instead of tracing them: sheet seam at Fulton (wrong), clipping (wrong); at Visitacion Valley the
    county line, an "undrawn strip", a route around the outside - three wrong stories before I traced the actual path
    (one coin-flip border on the heavy line + the push counted on every lot).
12. Did not understand the map: treated blank paper beyond the county line as land; drew the solve's filled-in space
    beyond the city as if it were lots. Every lot is in San Francisco; the city limit is a hard edge; blank = not city.
13. A feedback loop that replaced ink evidence with film verdicts and snowballed; I had not red-teamed it.
14. Asked again about things already decided (whole-line judging).
15. Pictures: wrong sheet, unreadable scale, fans from a park's centroid, overlays without the run's answer, pictures
    sent with no stated purpose.
16. Sign error (lot evidence) - caught only because the test went to 0%.
17. Many city-wide reruns with one changed number each (v5..v13): small, noisy moves = chasing the wrong thing.

## What the owner keeps telling me (his model of the problem)

- Probabilities, not cut-offs. Evidence stacks: many weak, correlated signals along a whole line = one strong signal.
  "It would be extremely unlikely for 300 dark pixels in a row not to be a wall."
- Judge things at the level where they are one thing: a whole line, a whole district, a whole film.
- Codes and lines inform each other (holistic): labels meeting across a faint line make it a line.
- The map has structure I know: blocks, streets, thin lines around blocks, heavy district lines, labels, arrows, the city
  limit. Use it - with distributions, not certainties.
- Test each idea in isolation with cases drawn from measured data; then apply; then check the real map.
- Trust is about the label's code, not the district's extent.
- When stuck: re-examine assumptions; don't tweak numbers.

## Rules for myself

1. Before building anything: write the assumptions down; check each against the data (table, scan pixels, a picture I
   actually look at). Only then build.
2. Define every probability as an event: "P(table code differs across this border | ink evidence along its whole line)".
   Check it is calibrated. Never say "held 20% of the time".
3. Choose the unit where evidence is one piece and the outcome is pure (whole line piece, whole film, whole district).
   Check purity first. Never sum or multiply correlated evidence per border or per lot.
4. Trace the mechanism (the actual path, the actual numbers) before forming a hypothesis. One hypothesis, one check.
   Say "I don't know yet" instead of a story.
5. Plateau rule: if a change moves the city total by a few hundred either way, stop. Re-analyse errors and assumptions.
6. Red-team every plan before running it: double counting, wrong unit, wrong event, distribution shift, feedback.
7. Domain sanity: all lots are in SF; blank paper = no information; the solve may never use space outside the city.
8. Pictures: right sheets stitched, readable, overlays fully labelled (table vs run), one stated purpose.
9. Messages: one point, plain words, no repeating settled questions.

## Phase 2 additions (2026-11-12)
18. Fitted the film judgement to the wrong event ("the code changes across the film's borders").  A misplaced film between
    two real districts scores 'not real', so the judge learned to merge real districts; one merge renamed 5,033 lots.
    The event must be the decision's consequence: what merging does to the lots it renames.  (Rule 2 again.)
19. Turned exact-looking evidence into a clipped feature, then into an exact offset - both wrong.  The park evidence is a
    sum over correlated lots: neither flat nor face value.  Check how a summed quantity behaves on held-out data first.
20. A judge fitted on the old judge's films meets new films once it decides differently (off-policy).  Pool rounds and
    re-check calibration on each new run before trusting a score.
21. Good: measured before swapping the push for the ink path (rejected by data), and before adding a lot-size unary (no
    gain).  The 6309 cause turned out to be the line odds, not the energy.
22. (2026-11-12) Called the Fulton line "p04's frame line" without looking at the page as scanned, because the drawn-area
    outline was traced along it.
    - p04's real frame is the outer rectangle around the whole sheet, with the legend box inside it at the bottom.
    - The Fulton line is where p04's drawing stops: Fulton's south side, the park's edge. Blank paper lies below it, down
      to the legend.
    - Before naming what a printed line is, show the sheet as scanned.
