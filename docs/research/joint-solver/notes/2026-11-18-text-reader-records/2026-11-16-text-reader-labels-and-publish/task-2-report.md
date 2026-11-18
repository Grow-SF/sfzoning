# Task T2 report: `solver-boxes`

Status: DONE_WITH_CONCERNS. Commits: 5e5b0c5 (code, tests, README, dvc.yaml), 6d00181 (dvc.lock).

## What was implemented

Word boxes for every 2000 sheet (13 sheets), from the per-quarter held-out chance (`data/solver/text-reader/<sheet>/{chance.png, kind.npz}`) and the page's ink. No outside input. Terra's saved boxes are read only to choose the level and to score.

- `pipeline/src/sfzmaps/solver/wordboxes.py` (new): the port of S3's `group` at the ink-gated level, plus the choice and scoring.
  - Boxes are 8-connected components of px with chance at or above the level and within 4 px of ink, holding at least 20 ink px. The axis is the major axis weighted by chance, mod 180 degrees counter-clockwise.
  - Each box records the grouped extent and the ink-trimmed extent (same axes, extent of the component's ink px only).
  - Kind probabilities and mean chance are means over the box's ink px.
  - Placement uses `prints._place` (centre in footprint and on the mapped area; others dropped and counted). The quarter is `grid.quarter_of` of the placed point.
  - `choose_levels`: per quarter, the level with the highest recall (IoU 0.3) of Terra's zoning-code words on the other three quarters. Ties go to the higher level.
  - `keep_chosen`: a box is kept only at its own quarter's level.
  - Constants (`BOXES_DIR` etc.) are in this module, so `paths.py` and `runs.py` are untouched. Every other solver stage stays up to date.
- `cli.py`: `sfzmaps solver boxes [--workers N]` (4 fork workers, one thread each, run pinned).
- `build.py`: stage `solver-boxes` (upstreams solver-lots, solver-prints, solver-page, solver-text-reader; `cut=text_cut`). `dvc.yaml` regenerated.
- `pipeline/README.md`: entry after the text-reader paragraph.
- `tests/test_wordboxes.py`: 12 tests.

## Output format (`data/solver/boxes/`)

`boxes.parquet`, one row per kept box (8,794 rows):

| column | meaning |
|---|---|
| `sheet`, `i` | sheet name; index of the box among that sheet's boxes at its level (before placement drops) |
| `level` | chance level of the box's quarter |
| `quarter` | 0..3 = nw, ne, sw, se, from the box centre's map point |
| `page_x`, `page_y` | box centre, page px (edge coordinates) |
| `axis` | degrees, mod 180, counter-clockwise from left-to-right |
| `length`, `height` | grouped box along and across the axis, px |
| `t_length`, `t_height` | ink-trimmed box, px |
| `mean_chance`, `ink_px` | mean chance over the box's ink px; their count |
| `p_zoning`, `p_block`, `p_street`, `p_other` | mean kind probabilities over the ink px (order of `data.KINDS`) |
| `x`, `y`, `lab_lot`, `street` | map point (EPSG:2227 ft), lot under it, in a street (from `prints._place`) |
| `g{0..3}{x,y}` | the grouped box's 4 corners, page px |
| `t{0..3}{x,y}` | the trimmed box's 4 corners, page px |

`levels.json` holds the chosen level per quarter and the recall on the other quarters at each level. `summary.json` holds the measures. Nothing in the outputs is a time.

## Tests

`tests/test_wordboxes.py`, 12 tests on small synthetic chance and ink arrays:
- two words become two boxes, and gaps under the ink radius stay one box;
- the axis is mod 180 and counter-clockwise (0, 30, 90, 150);
- the trimmed box is the ink and the grouped box runs past it;
- specks are dropped and blank paper makes no box;
- the level is a threshold;
- kind and chance means are over ink px only;
- IoU and axis error (180 equals 0);
- the level is chosen on the other three quarters only, ties go higher;
- each box is kept at its own quarter's level;
- the per-quarter measures.

Full suite: 1444 passed, 3 xfailed.

## The level, and how it was chosen

The chance from the four held-out detectors is near 1.0 on almost all ink: median 1.00 on ink inside code words, and similar elsewhere. S3's levels (0.5 to 0.9, from in-sample or validation chance) therefore find far too little. Recall on the other quarters at levels 0.5 to 0.9 was only 23 to 53%. I extended the candidates to (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0). The thresholds are on uint8, so 0.995 is 254/255 and 1.0 is 255/255.

Choice per quarter by Terra zoning-code recall (IoU 0.3) on the other three quarters (Terra used only here and to score):

| quarter | chosen level | recall on the other three (at each level 0.95, 0.98, 0.99, 0.995, 1.0) |
|---|---|---|
| nw | 0.995 | .442 .534 .607 .651 .567 |
| ne | 0.99 | .618 .720 .752 .732 .496 |
| sw | 0.995 | .441 .545 .616 .649 .519 |
| se | 0.995 | .453 .558 .626 .666 .568 |

## Measures per quarter (the quarter's own zoning-code words, at its chosen level)

| quarter | words | found at IoU 0.3 | trimmed box found | touched at IoU 0.1 | found with axis within 10 degrees |
|---|---|---|---|---|---|
| nw | 524 | 73.5% | 73.1% | 88.0% | 55.2% |
| ne | 812 | 46.8% | 48.3% | 71.9% | 30.5% |
| sw | 340 | 78.8% | 79.4% | 90.6% | 53.2% |
| se | 474 | 69.0% | 68.8% | 89.5% | 44.1% |

The "axis within 10 degrees" share is of all code words.

Answered labels (the owner's code-check answers; 731 reach the sheets, 668 of them with a Terra box):
- 418 of the 668 (63%) have a box meeting their Terra box at IoU 0.3.
- 641 of the 731 (88%) have a box holding their point.

Boxes: 8,794 kept (nw 1,811, ne 2,452, sw 2,026, se 2,505). Boxes dropped off the footprint or mapped area at each level: 873 (level 1.0) to 2,523 (0.98); at the chosen levels about 1,6xx to 1,9xx.

## DVC

- `build --write-dvc-yaml`, then code committed (the build refuses a dirty tree), then `pixi run build solver-boxes`: 42 s. The upstream solver stages (page, lots, prints, text-reader) reported up to date, and nothing in `paths.py` or `runs.py` changed.
- The six warp stages' lock entries re-recorded a new md5 for `src/sfzmaps/cli.py`, which is in their code closure. Their outputs are unchanged: the `dvc.lock` diff is only those six cli.py hashes plus the new `solver-boxes` entry. `dvc.yaml` and `dvc.lock` are committed.

## Concerns

1. **Recall is well under S3's roughly 90%.** The held-out detectors' chance is saturated near 1.0 on ink, so one level separates text from non-text poorly, and glyph groups merge with their neighbours. NE is worst (46.8% found, 71.9% touched), where codes are dense. "Touched" runs 72 to 91%, so most misses overlap a box without reaching IoU 0.3: merges and splits. The axis from the second moments is usable on only 30 to 55% of codes (S3 had about 75%).
   - The likely cause is the large unknown disc (radius of the longest code) around Sol's points when `solver-text-reader` trained: little direct training signal on code words. That is a property of the chance, not of the grouping. I did not change the detector.
   - Possible next steps for the owner or T3/T4: join and split by the direction output (`direction.npz`), or use the kind probabilities and a component tree.
2. **Trim barely matters for recall** (it moves it by under 2 points), as expected. Its value is the crop size for T3.
3. **The level choice reads Terra's boxes inside the stage**, so `terra` is a stage input. The boxes depend on it only through the five-way level choice (recorded in `levels.json`). Choosing outside the stage would need a committed input.
4. **A level at 0.995 sits at the uint8 resolution of the chance.** A change of detector calibration would move the best level; the choice is recomputed on each build.
5. **Editing `cli.py` makes the warp stages stale on paper**; their lock entries were re-recorded with unchanged outputs, as above.
6. **Terra's zoning-code words in the answered blocks** are included in the level choice and the scores. Those blocks are held out of all four detectors, so the numbers are honest.
