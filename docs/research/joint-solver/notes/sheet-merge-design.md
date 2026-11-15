# Probabilistic sheet merging: design (2026-11-12, for the owner's review; nothing built)

## What is wrong now (traced at Fulton, not guessed)
`mosaic.py` gives every map pixel to whichever sheet's drawn-area polygon contains it, and keeps that sheet's ink there.
Three hard cut-offs:

1. **One sheet per place.** Inside the drawn area a sheet is trusted completely. One pixel outside, it is ignored
   completely, whatever it prints.
2. **The edge is a line, and the inside test is coarse.** The test is done on a 32-ft grid.
   - At Fulton, p04's edge runs down the middle of the heavy line, so half the line is cut.
   - The 32-ft test then cuts whole stretches of what is left.
3. **Gaps fall to nothing.** A strip that no drawn area covers (50–60 ft at Fulton) comes out blank paper, even where a
   sheet has ink.

Of p04's Fulton line, 40% reaches the page the solver reads.

## The model: every sheet is a noisy view of one drawing
Unknown at each map pixel x: does the drawing have ink here (a line or text)? Every sheet that has a pixel at x adds
evidence. No sheet is chosen.

For sheet s at x:

- **Position.** Its pixel comes through its own fitted transform. Its position uncertainty `σ_s(x)` is measured where
  sheets overlap. The ink is spread by `σ_s`, so a line printed 30 ft apart on two sheets still reinforces.
- **Coverage.** `c_s(x)` is the chance that s draws the map at x. It is a smooth curve in the signed distance to s's
  drawn-area edge, measured, not a polygon test. It lets a sheet that prints right to its edge (p04 at Fulton) count
  there.
- **Ink rates.**
  - Where s draws: `h_s` = P(ink | ink in the drawing) and `f_s` = P(ink | none).
  - Where it does not: `m_s` = P(ink), from frames, legends and margin text, whatever the drawing has.
- **Evidence from s.**
  - If s shows ink: `LR = (c·h + (1−c)·m) / (c·f + (1−c)·m)`
  - If s shows paper: `LR = (c(1−h) + (1−c)(1−m)) / (c(1−f) + (1−c)(1−m))`
  - Where c → 0, both LRs → 1. **Paper is transparent where a sheet does not draw** (the stacking picture's rule). It
    becomes evidence against ink only where the sheet is known to draw.
  - Frames and legends land where c ≈ 0. They add only what `m` says.
- **Combined.** `log-odds(ink at x) = prior + Σ_s log LR_s(x)`.
- **Output: a probability page** (4 ft/px) replacing the binary merged page, plus a layer saying which sheets contribute
  at each pixel.
  - The line reader reads probabilities instead of 0/1 ink.
  - Its border model is refitted on them, held out by quarter.
  - Pixels along one border stay correlated: the reader's fitted aggregation sums them, never the merge.

Every number comes from the sheets themselves, measured where two or more overlap, so the answer key is not needed to
fit the merge. The table is used only to check, held out by quarter.

## Measurements first (before any build)
- **M1. Position.** Offsets between sheet pairs where both print the same ink: their distribution, and how they vary
  with distance to an edge. This gives `σ_s(x)`.
- **M2. Coverage.** Where another sheet has ink, does s have ink within `σ`? Measured against signed distance to s's edge
  (−300…+300 ft), per sheet. This gives `c_s(d)`, and shows which sheets print past their edge.
- **M3. Ink rates.**
  - `h`, `f` inside sheets, from agreement at overlaps.
  - `m` in each sheet's margins: frames, legend boxes, text.
- **M4. Frames.** How often a sheet's frame line coincides with another sheet's line (Fulton: yes; the p03/p06 frames at
  Park Presidio: no). This tells whether `m` must depend on the frame.

## Assumptions, each checked against the data
- **A1. The sheets err independently given the drawing.**
  - Check: at overlaps, the agreement the model predicts vs what is seen.
  - "2000-digital" may be one rendering cut into sheets; omissions would then be shared. M3 shows it.
- **A2.** Coverage changes smoothly with distance to the edge. Check: M2 curves.
- **A3.** Position error is local and smooth. Check: M1 offsets vs distance between measuring points.
- **A4.** Margin ink is unrelated to lines. Check: M4. Where frames follow district lines, `m` is not constant.
- **A5.** The drawn-area polygons are right to within the coverage curve's width. Check: M2 shows the edge where coverage
  actually falls.

## Tests
**Synthetic cases.** Each draws its numbers from M1–M4, each with its expected answer:
1. One sheet prints a line at its edge; the neighbour does not print it (Fulton). Expect: line.
2. Two sheets print the same line about 30 ft apart. Expect: one line, stronger than either sheet alone.
3. A frame line with no district line under it (Park Presidio). Expect: weak, no wall.
4. A legend box lying over another sheet's map. Expect: no effect.
5. A strip no drawn area covers, with ink on one sheet. Expect: that ink counts by its coverage.

**Real checks.**
- Pictures: stacked sheets (paper transparent) beside the probability page at Fulton, Park Presidio, and the five worst
  joins by dropped ink.
- City-wide:
  - the share of table-line length near joins with ink, old vs new;
  - the line reader's held-out log-loss on borders within 100 ft of a join vs far from one.
- Then a bubble run (v22) on the honest gate, diffed lot by lot against v21.

## In the map tool
The scan layer becomes switchable:
- the old merged page;
- the probability page;
- each sheet alone;
- all sheets stacked, paper transparent, one colour per sheet.

The twin scan (1996-2000-zoning, same drawing) fits the same model as more sheets once it is registered.

## Build log (2026-11-12, after the owner's go-ahead)
- **zstack.py:** all 13 sheets laid on the solve's grid, unclipped (cleaned ink, raw ink, on-page, signed distance to the
  drawn edge).
- **M2 coverage (zcover.py, block outlines, no answer key):** sheets print fully from 40-120 ft inside their edge.
  - Half coverage sits 2-66 ft inside the edge (p05 +2, p07 +66).
  - p04, p05, p12 and p13 print 12-40 ft past their edge.
  - RED TEAM: the frame line at the edge puts ink near both block edges and background, so this measure reads frames
    as low coverage. Coverage at the edge is underestimated.
- **Border form of the merge (zsheetev.py), held out by quarter (zborderfit3.py, on top of +lots+wall):**

  | Set | Log-loss | Near joins (<100 ft) | Missed (near) | False (near) |
  |---|---|---|---|---|
  | now | .0472 | .0917 | 2,642 (390) | 2,254 (248) |
  | +merged (control: new ink-distance reader on the old page) | .0454 | .0814 | 2,605 (359) | 2,163 (197) |
  | +cover (coverage-weighted sheet ink) | .0454 | .0814 | 2,614 (373) | 2,163 (205) |
  | +merged+wm (control + width from the old page) | .0450 | .0816 | 2,613 (359) | 2,154 (196) |
  | +merged+ws (+ width from unclipped sheets) | .0448 | .0745 | 2,565 (314) | 2,141 (185) |
  | +cover+ws | .0448 | .0759 | 2,551 (296) | 2,145 (192) |
  | +cover+wc (coverage-weighted width) | .0448 | .0766 | 2,575 (326) | 2,144 (188) |

  - Coverage-weighted ink alone = the control. The merge's value is in the stroke width: unclipped sheets show a line's
    full width where the clip halved it.
  - zframew.py: near joins, ink 24+ ft wide is about 14x likelier on a table line.
  - Thin or no ink under a table line: 9.9% from the old page, 4.7% from the unclipped sheets.
  - Coverage-weighted width is worse at Fulton (-0.05 vs +0.40). The coverage curve is confounded at the edge (above),
    so it under-trusts p04's edge.
- **Fulton (park, 175 borders):** share called line 23% (now) -> 76% (+cover+ws).
- **Park Presidio frames (blocks 1590-95, 1660-68):** false lines 46 -> 48 in every set, control included. The frames
  cross lots and add nothing.
- **Runs:**
  - v22: +cover+ws.
  - v22m (control): +merged+wm.
  - Same setup as v21 otherwise.

## Runs and red team (continued)
- **v22m (control: new reader on the old page):** 4,313 wrong; v21 4,904. Vs v21 it fixed 733 and broke 142. 617 of the
  fixes are the fragile 7103/7094 single-label bubble flipping back (a coin toss), so judge against v22m, not v21.
- **v22 (+cover+ws):** 4,414. Vs v22m it fixed 51 (incl. Kezar's two P lots) and broke 152:
  - RM-1 -> RH-3 79 (blocks 1152/1130/1125);
  - NC-2 -> RM-1 54 (blocks 1770-72).
- **CAUSE (zlodiff.py):** p03's title block and legend strip lie over the Western Addition. The widest-over-sheets width
  read legend rules as strokes, moving odds there by 1-3 points without crossing even odds.
  - The earlier legend check (zlegendcheck.py) counted only crossings of even odds, so it missed this.
  - LESSON: red-team how far odds move, not only threshold crossings.
- **Stack picture (stack_city_raw_all.png):** every sheet carries a title and legend strip below its frame, lying over the
  next sheet's map. Sheets draw only inside their own frame; neighbours meet frame to frame.
- **Fix (zsheetz.py):** evidence added over sheets, split by four zones of distance to each sheet's own edge (far outside
  where legends sit / just outside / edge / inside). The fit learns each zone's weight.
  - Odds moved on non-lines under legends vs elsewhere: +cover+ws 0.88 vs 0.50; +zx+zy 0.54 vs 0.42; +merged+zy
    0.55 vs 0.46.
  - +merged+zy: log-loss .0449 (near joins .0775), missed 2,574 (near 326), false 2,146 (near 190).
  - Fulton 45% called line (now 23%, control 39%).
- **v23:** +merged+zy.
