# Task R1 report: the reader's detector (stage `solver-text-reader`)

**Status: DONE_WITH_CONCERNS.** The stage is built to the brief, tested, and run once on the real data. The quality
of that run is **not right** for R2; see "Quality on the training labels" at the end. Two problems, both from
settings the brief fixed, so I did not change them:

1. **The 328-px unknown disc around every Sol point teaches the detectors that district lines near labels are text.**
2. **The reading direction is useless.**
   - It points the wrong way on most labels, for this recipe and for the walls' detector alike.
   - The old quarter models, trained by the old recipe, get it right: 90% of box labels within 15°.

Details and options are under "Concerns".

Commit: `ce828df` (local, not pushed). No later commit: the quality checks are scratch scripts and data, not code in
the branch.

## What was built

- `textnet/data.py`
  - `SheetData.held` is a tuple of map areas, replacing `hold_out: bool`. There are three kinds of area:
    - `MapRect`: the validation area is `VALIDATION = MapRect(*VALIDATION_AREA)`.
    - `Quarter(q)`: uses `grid.quarter_of`.
    - `GridArea`: a bool window of the solver's grid, which may be shared by many sheets.
  - Each area answers in two ways:
    - for a whole page window, from its placement samples, when that settles it (`decided`);
    - otherwise px by px (`contains`).
  - `in_area` is the general test. `in_validation` is now `in_area(VALIDATION)`, with the same numbers as before.
  - `centres` evaluates the hold-out only at its strided px, which gives exactly the values of the full band.
  - Other additions:
    - `SheetData.point_radius` (default `READ_POINT_RADIUS` = 30);
    - a `point_radius` argument to `sheet_data` and `load_sheets`;
    - `data.hold_out(sheets, areas)`.
  - Trials and `solver-text` behave exactly as before. All 47 old textnet tests pass unchanged except for renaming
    `hold_out` to `held`, including the recorded-weights guard and the reference-page and validation tests.
- `textnet/predict.py`: `outputs(net, ink, device, only=None) -> Outputs(chance, kind, kind_p, direction)`.
  - Its chance is bit-identical to `chance()`.
  - `kind` is the argmax of the float32 logits; `kind_p` is the float16 softmax, shape (4, h, w).
  - `direction` is the float16 unit (cos, sin), shape (2, h, w).
  - Tiles that are not scored get chance 0, kind 255, probabilities 0 and direction (0, 0).
- `textnet/reader_stage.py`: `build_reader_text(device=None, recipe=None, seed=0)`.
  - **Held-out blocks:** `held_blocks()` gives `labels.held_out_blocks(load_labels(crops=False))`, which is 715
    blocks. `block_area()` turns them into a `GridArea`: each grid px's nearest lot (`lots.npz` `grown`) and that
    lot's block, from `table.parquet`.
  - **Unknown radius:** `measure_radius()` takes the longest Terra "zoning code" word rectangle, measured in px on
    `SheetData.words`. Words whose centre lies in an answered block are left out.
  - **Training:** four fits of `stage.DEFAULT_RECIPE`, seed 0, each holding out `(Quarter(q), area)`, all in one
    exclusive `gpu_turn`.
  - **Scoring:** each 2000 px takes the outputs of the model for its quarter (`page_quarters`, `sheet_outputs`).
  - **Files:** `.npz` files are written with a fixed zip date (`save_npz`), so they are byte-stable.
  - **Scores:** `quarter_scores` uses the scores module's counts (ink px, and words found by kind) on each quarter's
    own px. The long lines are left out because finding them takes minutes per page.
- **Outputs** (`data/solver/text-reader/`):
  - `models/{nw,ne,sw,se}.pt`;
  - `<sheet>/chance.png`, `kind.npz` (`kind`, `p`) and `direction.npz` (`direction`);
  - `held.json`: the quarters, the split, the blocks, `block_grid_px`, the radius with its measurement, the recipe and
    the seed;
  - `scores.json`.
- `paths.READER_TEXT_DIR`, the CLI `sfzmaps solver text-reader [--device]`, and a README section. The README module
  list also gains `reader_stage`.
- `build.py` stage `solver-text-reader`:
  - **Upstream:** `solver-lots`, `solver-prints`, and `warp-<s>` for the six training maps.
  - **Reads:** each training sheet's `page.png` and `district_labels.json`, plus its warp `transform.json` and
    `correction.npz`. Also Terra's answers, the other passes, `prints.parquet`, `terra_boxes.json`, `lots.npz`,
    `table.parquet`, and the two code-check files.
  - **Code:** the import closure of `reader_stage`, cut by gpu and runs.
  - `dvc.yaml` was regenerated; only the new stage was added. `dvc.lock` was not touched.

## Tests

The full suite passes: `cd pipeline && uv run pytest -q` gives 1143 passed and 3 xfailed (both counts were already
there before this task).

New tests:
- `test_reader_detector_never_trains_on_its_quarter_or_held_blocks` uses the two synthetic maps, with the quarter
  split moved through both pages and synthetic lots (every third block answered). For each q:
  - no crop centre is in the quarter or in an answered block, while without the hold-out some would be;
  - a real `train.fit` (4 steps × 6 crops) draws crops that never centre there;
  - both `known` and the loss weight `w` are 0 on every px of the quarter and of the blocks, judged against a plain
    whole-page reference (`page_world`, the split, and grid lookup);
  - those px would have carried loss without the hold-out, so the test bites.
  - Mutation check: when only the first area is held out, or the radius is ignored, this test and the radius test fail.
- `test_unknown_around_sol_points_covers_the_word`:
  - The radius is the longest code word (100, R-1-D). A word whose centre is held out is left out (giving 80, RH-1).
    With no word left, it raises.
  - Every px within R−1 of a point is unknown unless it is in a word.
  - Every px R+1 or more from all points is known, unless it is in another pass's box.
  - At 30 px the same ring is known.
- `test_outputs_returns_chance_kind_and_direction`:
  - shapes and dtypes are right, and the chance equals `chance()`;
  - the probabilities sum to 1 and `kind` is their argmax;
  - the direction has unit length;
  - on one tile, the outputs equal the network's own softmax and normalised (cos, sin);
  - unscored tiles follow the convention above;
  - a text-only network raises.
- `test_the_reader_stage_trains_four_detectors_in_one_turn_and_gives_each_px_its_quarters` (TINY recipe, CPU):
  - there is one exclusive turn, with the load before it;
  - each fit gets the right `held` and radius;
  - the exact file tree is written, and each px's chance, kind, p and direction equal its own quarter model's outputs;
  - `held.json` and `scores.json` have the expected content;
  - a second run gives identical bytes.
- `test_the_text_reader_command_runs_its_stage` covers the CLI.
- `test_reader_stage_deps` (test_build) checks the command, outs, upstream and order. It also checks that the code
  deps equal the import closure exactly, without gpu, runs, trial or score_stage, and with `transform.py`. The data
  deps are exactly what is listed above.
- `test_one_detector_stage_...` now expects the third `solver-text*` stage.

## The real run

`uv run sfzmaps solver text-reader` on mps, in one exclusive GPU turn, in the foreground process. The log is
`pipeline/data/build-logs/solver-text-reader-run1.log`.

- **Minutes:** 28.0 wall-clock in total.
  - load: 0.5 (labels, lots, 87 sheets with placements, radius, quarters);
  - train: 25.8 (about 6.4 per quarter);
  - predict and write: 1.7;
  - score: under 0.1.
  - Load average stayed at about 4–6.
- **Answered blocks held out:** 715 of 5,302.
  - They cover 44.3 M grid px, about 38% of the grid. `grown` carries coastal blocks out over the water.
  - They hold 3,821 of the 10,018 Terra code words on the training maps, which is 38%.
  - With the quarter also held out, one model's crop centres on text drop from 7.90 M to 4.90 M (62%).
- **Unknown radius: 328 px.**
  - It is set by Terra's box for "SH" on 1968 sheet `20180102110513967-p18`: a 328 × 293 px box for a two-letter
    word, i.e. a Terra mis-box.
  - The rule, applied literally, takes it.
  - The longest real code word outside the answered blocks is INNER CLEMENT NCD, at 312 px.
  - Including the answered blocks, the longest is UPPER MARKET NCD, at 363 px (both the 1990–96 and 2000 scans).
  - Of the 6,197 words measured, the median is 85 px, the 90th percentile 114 px and the 99th percentile 169 px.
- **Words found on each model's own held-out quarter** (the scores measure, on the 2000 sheets, from `scores.json`):

  | held-out quarter | all | zoning code | block number | street name | other |
  |---|---|---|---|---|---|
  | nw | 99.40% (2,155/2,168) | 523/524 | 1,020/1,020 | 505/515 | 107/109 |
  | ne | 99.70% (2,695/2,703) | 812/812 | 1,170/1,172 | 529/530 | 184/189 |
  | sw | 99.86% (2,152/2,155) | 340/340 | 1,067/1,067 | 607/609 | 138/139 |
  | se | 99.90% (3,000/3,003) | 474/474 | 1,368/1,368 | 973/976 | 185/185 |

  These numbers are high because the detectors call most ink near labels text. See concern 1.
- **Output size:** 2.0 GB in total.
  - kind.npz: about 100 MB per sheet, 1.3 GB in all.
  - direction.npz: about 47 MB per sheet, 555 MB in all.
  - chance.png: about 9 MB per sheet.
  - Keeping kind and direction only on the ink and the px next to it would cut this to about 0.5 GB. I didn't do
    that, because the brief says every px.

### Second run

The controller stopped the second run before it finished: reproducibility waits until the quality is right.
- The pipeline's tests show the stage gives the same bytes twice on the CPU.
- Up to the kill, the second run's logged losses matched the first run's exactly: 24 log lines, all of nw's training
  and ne's first 2,000 steps.

## Pictures

All are in `pipeline/data/solver/experiments/text-reader-pictures/`.

1. `direction-dense.png`: 2000 sheet p11, px (1400, 800)–(2600, 1700). This is the window of 1200 × 900 px with the
   most Terra words (235): Glen Park and Mission Terrace, around Balboa Park.
   - **Left:** the reader detectors' reading direction on ink called text. Hue shows the direction, with a wheel in the
     corner, and there is an arrow every 20 px.
   - **Right:** Terra's boxes with their own direction, for comparison.
   - **What it shows:**
     - Block numbers (read left to right) mostly come out blue/cyan, i.e. pointing about 180–250° the wrong way.
     - Codes are mixed.
     - Most lot lines and district lines are coloured, i.e. called text.
   - **Action needed:** yes. It is the evidence for both concerns.
2. `chance-walls-vs-reader.png`: the same window, comparing the walls' detector (left) with the reader's detectors
   (right). Ink called text is red, Sol's points blue.
   - **What it shows:** the reader's detectors call the heavy district lines text almost everywhere near labels. The
     walls' detector does not.
   - **Action needed:** yes (concern 1).

## Concerns

### 1. The disc around every Sol point removes the not-text signal near labels, and the detectors call district lines text

At 328 px around every Sol point, 23–47% of each training map's px are unknown, against 0.1–1% at 30 px:

| map | unknown share |
|---|---|
| 2000 | 42.7% |
| 1980–90 | 46.6% |
| 1990–96 | 42.9% |
| 1960 | 32.1% |
| 1964 | 27.7% |
| 1968 | 23.0% |

District lines run beside labels, so they are almost never taught as not-text.

I scored both detectors on the same 2000 px with the walls' convention (30-px discs left out; long lines from the
scores module). The reader's detectors are each on their own held-out quarter; the walls' detector is in-sample, so
its numbers are flattering:

| quarter | ink outside boxes called text (reader / walls) | long-line ink called text (reader / walls) | AUC (reader / walls) |
|---|---|---|---|
| nw | 24.9% / 1.6% | 22.9% / 5.7% | 0.966 / 0.992 |
| ne | 51.3% / 1.9% | 47.6% / 6.0% | 0.884 / 0.993 |
| sw | 24.8% / 2.8% | 28.2% / 9.7% | 0.972 / 0.987 |
| se | 28.5% / 3.0% | 29.4% / 8.9% | 0.953 / 0.987 |

On the px outside the 328-px discs (the stage's own `scores.json` convention) the reader's detectors look fine:

- ink outside boxes called text: 2.2–3.5%, against the walls' detector's 1.4–3.2% on the same px;
- long-line ink called text: 19–32%, against 17–28%.

So the damage sits exactly near the labels, which is where R2 reads.

For R2 this means the profile of text chance along a word's line runs straight onto district lines. The fused-grouping
alternatives would be everywhere.

Option, not implemented and not decided: put the disc only around Sol points that lie in no Terra word, the point
labels Terra may have missed. That is 96 of the 1,653 points on 2000, and 62, 99, 2, 15 and 20 on the other maps.

- At 328 px this leaves 9.4% of the 2000 map unknown (6.8%, 9.6%, 0.1%, 0.8% and 0.8% on the others) instead of
  42.7%.
- Points inside a Terra box already have their word taught as text.

This needs the owner's or the spec's call, since the spec says "around every Sol point".

### 2. The reading direction is not usable as trained, by either detector

Agreement with Terra's box angle on the ink inside the boxes, pooled over all 13 sheets. Each reader px comes from
its own quarter's model; the walls' detector is in-sample.

| kind | detector | within 45° | within 90° | reversed (> 135°) |
|---|---|---|---|---|
| zoning code (2,150 words) | reader | 16.8% | 31.0% | 38.2% |
| | walls | 2.8% | 60.4% | 8.3% |
| block number (4,627 words) | reader | 8.3% | 17.3% | 60.9% |
| | walls | 91.3% | 98.6% | 0.4% |
| street name (2,630 words) | reader | 46.0% | 61.6% | 21.9% |
| | walls | 64.4% | 76.9% | 15.1% |
| other (622 words) | reader | 51.0% | 91.6% | 3.6% |
| | walls | 85.3% | 90.2% | 6.3% |

The angle loss is still falling at the end of the 4,000 steps:

- the reader's four models end at 0.71–0.82, i.e. mean cosine 0.18–0.29;
- the walls' stage ends at 0.61.

The angle loss stays near 0.99 for the first ~2,000 steps and only then starts to fall. The recipe was chosen on
words found and long-line ink, never on the direction.

R2's premise ("the detector's direction gives the frame, no turn search") does not hold with these outputs. It needs
a recipe that trains the direction head; candidates are more steps or a higher `ANGLE_WEIGHT`. That should be
measured with this direction measure as a trial, and is again the controller's or owner's call.

### Smaller points

- **The radius is set by a mis-boxed "SH"** (328 × 293 px). The literal rule takes it; the longest real code word
  outside the answered blocks is 312 px.
- **The radius excludes words in answered blocks**, per the global constraint. Including them gives 363 px.
- **The answered blocks cover 38% of the grid and 38% of the code words.** This is because `grown` extends coastal
  blocks over the water and answered labels sit where labels are dense. With a quarter out, each model trains on
  about 62% of the text centres.
- **The stage imports `codereader.labels`**, so `codereader/__init__.py` (a long module-list docstring) is in its code
  deps. An edit there in R2 or R4 would mark this ~28-minute stage stale. The outputs would be byte-identical, so
  nothing downstream reruns, but the time is spent.
  - Making that `__init__` one line, like `solver/` and `textnet/`, would avoid this. I didn't touch it; it is out of
    my files.
- **Extra files in the trials' cache.** My comparison script wrote the 13 sheets' long-line masks into the trials'
  cache, `data/solver/experiments/line-masks/2000-digital-zoning/`. They are not stage outputs.
- **Scratch scripts** used for the measurements above (not committed) are in
  `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/`:
  - `compare_quarters.py`
  - `direction_all.py`
  - `cover2.py`
  - `direction_picture.py`
  - `chance_picture.py`
  - `quality.py`
  - `label_pictures.py`

## Quality on the training labels (the controller's request, CPU only, first run's outputs)

**Labels.** `labels.training(load_labels())` gives 959 labels: 905 box and 54 point. `refuse_held` passes, so no
held-out label or block is among them. All 959 have Task 2's fit.

**Fits.** The fits come from `scratchpad/codereader/task2/fits_real.pkl`, with `font_real.npz`. That is Task 2's final
tight-prior alignment with the learned font. Each crop is checked to be the same as Task 2's: 959 of 959 are.

**Detectors compared:**
- *reader*: this stage. Each px comes from its own quarter's model.
- *walls*: `solver-text`, trained on every px, so in-sample.
- *old*: the old quarter models (`data/solver/text-{nw,ne,sw,se}`, `Recipe()`: full size, 8,000 steps). Each label is
  read by the model of its quarter.

**Word ink.** A label's word ink is the page px under the aligned glyphs' strokes (`glyphs.render` ≥ 0) that are ink
and on no line (rule "run").

Script: `scratchpad/quality.py`. Per-label rows are in `pipeline/data/solver/experiments/text-reader-quality/quality.json`.

### 1. Words found, each model on its own held-out quarter, against `solver-text` on the same px

The measure is `scores`, on the 2000 sheets.

| quarter | reader | solver-text |
|---|---|---|
| nw | 99.40% | 96.36% |
| ne | 99.70% | 96.49% |
| sw | 99.86% | 95.59% |
| se | 99.90% | 94.81% |

So holding out costs nothing on recall. The reason is that the reader's detectors call nearly all ink near labels
text: under the 30-px convention they call 25–51% of the ink outside boxes text, against 1.6–3.0% for solver-text
(concern 1). The words-found gain is that over-calling, not a better detector.

### 2. Reading direction against Task 2's fitted direction

The direction is the chance-weighted circular mean over the word ink. The table gives the share within 15°, the share
within 45°, the share reversed (more than 135° off) and the median error.

| labels | reader | walls | old | Terra's box |
|---|---|---|---|---|
| box, all (905) | 2% / 15% / 41% rev / 123° | 1% / 3% / 10% rev / 86° | **90% / 95% / 3% rev / 3°** | 96% / 96% / 4% rev |
| box upright (781) | 1% / 15% / 43% / 123° | 1% / 3% / 0% / 86° | 94% / 98% / 1% / 3° | 100% / 100% / 0% |
| box sideways (99) | 5% / 10% / 29% / 117° | 4% / 10% / 85% / 175° | 83% / 92% / 2% / 3° | 90% / 90% / 10% |
| box upside down (25) | 4% / 36% / 12% / 58° | 0% / 0% / 16% / 104° | 4% / 4% / 84% / 170° | 0% / 0% / 100% |
| point, all (54) | 0% / 17% / 26% / 116° | 7% / 7% / 17% / 88° | **81% / 81% / 11% / 4°** | n/a |
| point upright (37) | 0% / 22% / 35% / 123° | 3% / 3% / 0% / 86° | 97% / 97% / 0% / 4° | n/a |
| point sideways (13) | 0% / 0% / 8% / 111° | 23% / 23% / 69% / 171° | 62% / 62% / 23% / 5° | n/a |
| point upside down (4) | 0% / 25% / 0% / 65° | 0% / 0% / 0% / 96° | 0% / 0% / 75% / 174° | n/a |

A "near" variant uses all ink within 40 px of the label's point, chance-weighted. Within 45°: reader 13% box and 4%
point; walls 6% and 9%; old 94% and 57%.

What the table shows:

- **The stage recipe's direction head has not learned to read.** Over a whole area it outputs nearly one direction,
  so the two pictures show near-uniform colour. The angle loss stays near 1.0 for the first ~2,000 of the 4,000 steps.
- **The walls' detector is the same.** It gives every code about 270° ("down"), but block numbers right.
- **The old recipe learns it.**
- The "upside-down" fits disagree with Terra's angle 100% of the time, and with the old models 84% of the time. Those
  25 + 4 fits may be wrong themselves; the Task 2 review discussed bottom-to-top prints.

### 3. Point labels: share of the word's ink the detector calls text

| detector | point labels: mean | point labels: share of labels with ≥ 90% called | box labels: mean | box labels: share ≥ 90% |
|---|---|---|---|---|
| reader | 100.0% | 100% | 99.9% | 100% |
| solver-text | 89.5% | 78% | 95.1% | 89% |
| old | 87.1% | 81% | 94.4% | 89% |

So the reader's detectors no longer drop a point word's ends. They also call the surrounding lines text (pictures
below), so this is not yet evidence of a better detector.

### 4. Kind: share of training code labels whose detector kind is "zoning code"

The kind is the chance-weighted mean of the kind probabilities over the word ink.

| detector | box | point |
|---|---|---|
| reader | 96.0% | 94.4% |
| solver-text | 97.0% | 94.4% |
| old | 98.7% | 92.6% |

Kind is fine.

### Picture

`pipeline/data/solver/experiments/text-reader-pictures/training-labels-direction.png` shows eight training labels:

| kind | labels |
|---|---|
| box upright | P (p01 print 2); RH-1(D) (p06 913) |
| box sideways | RM-2 (p01 5); RH-3 (p07 1106) |
| box upside down | RM-4 (p01 8) |
| point upright | RINCON HILL RES. SUD (p01 170); RH-1 (p06 1877) |
| point sideways | RM-1 (p01 161) |

For each label, the left panel is the reader's detectors and the right panel the walls' detector. Text ink is
coloured by direction, with small arrows. The green arrow is Task 2's fitted direction; the red arrow is the
detector's mean over the word.

- **Reader's detectors.** Every lot line and district line around the labels is called text, all in one colour (blue,
  about 230°). The red arrow is off by 107–157° on all but two of the eight; the exceptions are RH-3 at 11° and the
  upside-down RM-4 at 15°.
- **Walls' detector.** Lines stay grey. Codes are violet ("down", about 270°) whatever their real direction; block
  numbers are red (0°, right).
- **Action needed:** yes. It shows both problems directly.

### Verdict

**Not right yet.** R2 cannot build frames on these outputs:

- the text chance near labels includes the district lines;
- the direction is nearly constant and wrong.

Kind is fine, and holding out the quarters and blocks works as specified.

Two fixes are needed before re-running the stage. Each should be measured first with these checks as a trial; nothing
here was changed:

1. **The unknown disc.** For example, only around Sol points in no Terra word: 96 of 1,653 points on 2000, leaving
   9.4% of that map unknown instead of 42.7%. Or keep 30 px around points inside a Terra box.
2. **A recipe whose direction head learns.** The old recipe's quarter models reach 90% of box labels within 15°
   (median 3°). The stage recipe reaches 2% (reader) and 1% (walls). Candidates: more steps, a longer warm-up, full
   size, or a higher `ANGLE_WEIGHT`. They should be chosen by this direction measure on training labels, as well as
   words found and long-line ink.

Determinism can then be confirmed with one repeat.
