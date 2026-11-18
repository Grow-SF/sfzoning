# Task 3 report: baselines, picture panel, missed-text audit

Commit 837c994. Outputs (not committed): `pipeline/data/solver/experiments/textlines/baselines/` (`<source>/{scores.json,scores.md,blocks.npz}`, `summary.md`, `panel.png`, `audit/{r328,old-quarters}.{png,json}`).

## Built
- `scripts/textlines/panel.py`: `WINDOWS`, `render`, `page_ink`. Window = (sheet, centre row, centre col, size). Red = ink with chance >= 0.5, grey = other ink, a row per window, a column per source.
- `scripts/textlines/audit.py`: `pieces` (redteam_missed.py's letter-sized rule, run-rule ink removed, > 10 px from every Terra/other box), `missed` (mean chance over the piece's own px < 0.5), `candidates`, `draw`, `sample`, `sheet` (8x8 contact sheet).
- `scripts/textlines/baselines.py`: copies r328 (nested, with held.json) and all-sheets, then gates 7 sources in ONE process with ONE `CityGeo` and one load of the sheets: r328, all-sheets, trial, old-nw/ne/sw/se (each its own quarter only). Writes summary, the panel (4th column "old quarters" = each px from the model of its own quarter) and both audits. Resumable: a source with scores written is skipped.

## TDD
Test written first; red (ModuleNotFoundError audit/panel), then green. `tests/test_textlines_panel.py`: 6 tests (zeros draw no red; red only on called ink; window past page edge; audit picks letter-sized missed pieces; skips pieces near a box or run line; sheet writes a picture). A first test run showed my test glyph "B" exceeded the 0.7 fill rule, so the test uses "K". `uv run pytest -q`: 1486 passed, 3 xfailed.

## Baselines (called >= 0.5, 95% interval)
| source | m1a_in_box | m1a | m1b_in_box | m1c | m2 | overlap_band |
|---|---|---|---|---|---|---|
| r328 | 0.987 [0.983, 0.992] | 0.668 [0.651, 0.686] | 0.975 [0.959, 0.984] | 0.366 [0.291, 0.471] | 0.999 [0.998, 0.999] | 0.977 [0.963, 0.985] |
| all-sheets (in-sample) | 0.579 [0.558, 0.600] | 0.046 [0.044, 0.049] | 0.804 [0.671, 0.879] | 0.036 [0.033, 0.039] | 0.983 [0.975, 0.988] | 0.807 [0.699, 0.872] |
| trial | 0.643 [0.623, 0.664] | 0.059 [0.056, 0.062] | 0.821 [0.695, 0.890] | 0.041 [0.037, 0.045] | 0.985 [0.977, 0.989] | 0.823 [0.723, 0.884] |
| old-nw | 0.683 [0.644, 0.718] | 0.043 [0.038, 0.048] | 0.845 [0.727, 0.919] | 0.019 [0.017, 0.022] | 0.981 [0.973, 0.986] | 0.847 [0.751, 0.916] |
| old-ne | 0.629 [0.595, 0.663] | 0.038 [0.035, 0.041] | 0.818 [0.775, 0.855] | 0.025 [0.020, 0.031] | 0.973 [0.968, 0.980] | 0.831 [0.794, 0.863] |
| old-sw | 0.733 [0.673, 0.788] | 0.039 [0.033, 0.044] | 0.795 [0.600, 0.875] | 0.034 [0.031, 0.039] | 0.983 [0.967, 0.991] | 0.800 [0.640, 0.877] |
| old-se | 0.627 [0.581, 0.672] | 0.047 [0.042, 0.052] | 0.859 [0.591, 0.928] | 0.035 [0.031, 0.041] | 0.965 [0.937, 0.980] | 0.846 [0.619, 0.922] |

all-sheets m1b_in_box 0.804 vs the red team's ~0.82: close (inside the interval 0.671-0.879). Smoke values reproduced (r328 m1a_in_box 0.987, all-sheets 0.579).

## What the panel shows (looked at it)
- r328 (the held-out detectors) colours line ink red heavily: block outlines, street lines and district lines through and around labels in all three RM-2 windows, residential (p02), downtown (partly: lines near the M-1 label), and the thick district line plus lot lines on sw_street (p12, top). The letters stay red too. This is the failure the gate was built for (m1a 0.668 of all district-border ink).
- all-sheets, trial and the old quarter models leave nearly all line ink grey and keep the letters red. Their errors: line ink that touches a label (e.g. the underlines/strokes in the hatched small lot, the thick ink where a label is crossed) is red in places; the dashed pier line is grey, with a stray red mark at its left end and a red dot in a small circle (all but the old models). Hill-street lettering (sw_dense) is red for letters and for the street-name text along curves in all columns; the old quarter models give the street name along the curve slightly less red.
- Lot numbers on sw_street are red in every column.

## Audits (looked at both sheets)
Letter-sized pieces in the pool: 5121 (all four quarters). Missed (mean chance < 0.5): r328 2017; old quarter models (each own quarter) 4586.
- r328, 64 drawn: plainly text 0/64 (nearly all are short diagonal or curved street-line fragments, line corners and junction stubs). Clopper-Pearson 95%: [0, 0.056].
- old quarter models, 64 drawn: plainly text about 4/64 (tiles 33, 34, 42, 45: a letter of a street name or lot label); 0.063, 95% about [0.017, 0.15]. The rest are line fragments, corners and ink bridges. My count is by eye and the borderline ones (34, 45) are judgement calls.
So the letter-sized ink outside the boxes that the detectors do not call text is almost all line ink, not missed text; the old models miss a few street-name letters.

## Window coordinates (sheet, centre row, centre col, size)
rm2a p02 1850,2505,240; rm2b p02 1745,2175,240; rm2c p02 1868,2218,240; downtown p01 3330,1730,240; residential p02 1750,1680,240; dashed p02 890,2050,240; hatched p01 3430,1690,240; sw_dense p12 1500,1470,240; sw_street p12 2770,1170,240. Sheet quarters (by sampled map points): p01, p02 (73%), p07, p08 are NE; p12, p13 are SW.

## Concerns
- "hatched" is small hatch-filled lots (p01 near 3751 Tangang Bora), not a broad hatched area; I did not find a larger one.
- r328 m1c 0.366 and m1a 0.668 are very high: r328 calls large amounts of ink text outside boxes too.
- Old-quarter px differ from r328's (own-quarter only): not paired-comparable across those rows; fine as separate baselines.
- The old-quarter panel column and its audit use per-px quarter assignment through `CityGeo.quarter`.
- Audit counts of "plainly text" are my eye only; no second judge.
- Each gate call re-computes the geo rasters per sheet (about a minute per source); total run about 12 minutes.
