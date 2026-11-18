# Task 3 report: reader crops from our own boxes, sloppy crops and masking

Commit 59b9cf7 (branch solver):
- `pipeline/src/sfzmaps/solver/wordread/readercrops.py` (new)
- `textread/crops.py`, `textread/train.py`, `textread/read.py`
- `pipeline/tests/test_wordread_readercrops.py` (new, 22 tests)

## What was built

### `wordread/readercrops.py` (no stage imports it)
- **`terra_frame(rows)`:** Terra's 2000 words, taken from crops.parquet rows. Each row has:
  - Terra's string;
  - `rectify`'s geometry (`cx`, `cy` as the page px index; `length`, `height`, `angle`);
  - the TOUCH flags;
  - Task 1's `c0x..c3y` (`word_rect`).
- **`with_neighbours(terra)`:** each word's nearest other word in the given frame on the same sheet. It uses shapely's STRtree on the rectangles. The columns are `n_word` (-1 when there is none) plus the neighbour's geometry.
- **`matches(boxes, terra, iou=0.5)`:** Task 1's rule, through `measures._overlaps` and `_best_word`. Each box goes to the word it meets best.
- **`own_box_crops(boxes, terra, iou=0.5, quarter=None, pages=None, touch=None, word_masks=None)`:**
  - Each matched box is cut exactly as `box_crop` cuts it.
  - The crop is turned a half turn when Terra's word reads more than 90° from the box axis, so it reads the way Terra's word does.
  - The target is Terra's string.
  - A crop is dropped when its own footprint OR its Terra word touches the answered blocks, or touches `quarter` when one is given (the union of the two flags).
  - With `word_masks`, each meta also carries the crop-space mask, turned with the crop.
  - `HeldAreas` gives the real footprint check: placements, `data.Quarter`, and the block area from `reader_stage.held_blocks()`, the same way `solver-text-crops` sets its touch flags.
- **`offset_distribution(boxes, terra, iou=0.5, quarter=None)`:**
  - Per pair, in Terra word heights, measured in our box's own frame: `start`, `end`, `cross`, `angle` (our axis minus Terra's, counter-clockwise, in [-90, 90]) and `height_ratio`.
  - Also `line_share` (matched words with at least `composite.CROSS_PX` line-ink px inside Terra's rectangle), `neighbour_share` and `neighbour_part`.
  - Answered-block words are always left out. With `quarter`, that quarter's boxes and words are left out too.
  - The frame is chosen so that cutting Terra's word with one pair's offsets gives back that box's own crop (tested).
- **`sloppy_crop(page, terra_word, dist, pieces, rng, info=None)`:**
  - Draws one measured pair's offsets jointly, so their correlations are kept.
  - With probability `line_share`, ORs a real line piece through a letter px. The piece is turned a random number of quarter turns, and one of its px lands on an ink px of the word.
  - With probability `neighbour_share`, ORs in the nearest other word. That word is moved straight toward this one (by bisection on the overlap) until a drawn `neighbour_part` of its rectangle lies inside the crop rectangle.
  - It works on a copy of the window, so the page is never changed. `info` records what was drawn.
- **`sloppy_crops(...)`:** makes sloppy crops for one reader.
  - Only words that touch neither the held-out quarter nor the answered blocks are used, and their neighbours come from the same set.
  - Each crop has its own rng: `default_rng([seed, hash_name(sheet), word, copy])`.
  - A crop whose footprint touches a held-out area is dropped.
- **`piece_patches(bank)`:** turns a `composite.Bank`'s line-piece references into grey patches.
- **`line_ink(terra)`:** counts line ink per word with `composite.crossings` on cached `zones` line px.
- **`build_offsets()`:** writes `offsets.json`; run with `python -m sfzmaps.solver.wordread.readercrops`.
- `mask_outside` is re-exported from `crops`.

### `textread/crops.py` (defaults byte-identical)
- `frame_of` was split out of `cut`.
- `cut` gained `fill` and `origin`, so it can cut from a window, and it can now cut bool masks too.
- New: `box_cut` (box_crop plus its frame), `box_mask` and `mask_outside` (shapes are checked).

### `textread/train.py` (no new import)
- **`READER_OPTIONS = {"own": 0.0, "sloppy": 0.0, "mask": False}`.** A recipe's own keys override these. I kept them out of `READER` itself, so `held.json` and the `.pt` "about" metadata stay byte-identical if the stage is ever rebuilt. This is a deliberate reading of "READER options".
- **`Pool.joined(extra, mask)`:** extra items are `(crop, target, meta)`, with weight 1. With `mask`, an own crop whose meta holds `"mask"` gets `np.where(mask, crop, 255)`.
- **`Pool.batches(..., own, sloppy)`:** when a share is non-zero, one multinomial draw splits the batch among real, synth, own and sloppy crops. When both shares are 0, the old code path runs unchanged.
- **`fit(..., extra=None)`:** stops with an error if a share has no crops passed in, or if the shares sum past 1. The log gains `own`/`sloppy` counts only when `extra` is passed.

### `textread/read.py`
- **`word_masks(sheet, rows, page=None, chance=None)`:** rebuilds each box's component exactly as `wordboxes.group` does (box `i` is the i-th kept component in label order), then grows it by `INK_R` with a disc. It raises an error if a component's ink-px count differs from the box's `ink_px`.
- **`box_crops(page, rows, masks=None)`** is the new cutting function. `_box_job` uses it.
- **`build_read(workers, mask=False)`.** The only new import is `grid`, which was already in the stage's import closure; `test_build` passes.

## TDD evidence
1. **Pinned first.** I pinned `fit`'s first 20 batches (reader and classifier) on a synthetic pool, with train.py as at d627640 (unchanged since 5b0b9d6).
   - Text hashes, valid on every platform: reader `080864c3…c8b3e`, direction `6def2540…713b3`.
   - Full byte hashes, Darwin arm64 only: reader `699abc2e…af9ab`, direction `403dbefb…7a0db`.
   - This passed against today's code before any change.
2. **RED:** collection failed with `ImportError: cannot import name 'readercrops'`.
3. **GREEN:** 21 passed. A 22nd test (tilted boxes: angle sign and half turns) first failed on a wrong expectation in my test (axis 177° on a word read at 180° needs no turn), not on the code. I fixed the test case.
4. **Mutation check:** forcing the multinomial path at zero shares fails 3 pinned tests (reader, direction, and extra at share 0).
5. **Final runs:**
   - my file: 22 passed;
   - iterate set (my file, wordread_measures, solver_textnet incl. GUARD_WEIGHTS, solver_textread, build, textnet_composite): 170 passed;
   - full `uv run pytest -q -n 2`: 1613 passed, 3 xfailed;
   - `pgrep` was empty before the commit.
6. **What the tests cover:**
   - Own crops carry Terra's string, keep only training-quarter boxes, are turned for words read at 180°, and equal `rectify` when our box equals Terra's. A word whose own row touches the blocks is excluded.
   - Masks are turned with their crops.
   - `offset_distribution` matches hand-computed numbers.
   - A pair's own offsets cut back its own crop: exactly at quarter turns, within 1 grey level at 3°.
   - The means of 500 draws are within 10% of `dist`; the line and neighbour rates are within 4 sd.
   - A line piece lands on a letter px.
   - A neighbour fragment is placed at the drawn share (within 0.05).
   - `sloppy_crops` respects the hold-outs and is deterministic.
   - `mask_outside` is exact.
   - `box_mask` equals the crop's ink, from the whole page or a window.
   - `word_masks` equals the component grown by `INK_R`, and a speck ahead of the boxes in label order doesn't shift the numbering.
   - Reading unmasked gives today's crops; reading masked blanks the line ends past a word.
   - On real data, `word_masks` finds every box of sheet p01.
   - `fit` draws at the given shares, and with extra crops at share 0 its batches are identical to the pin.

## offsets.json (`data/solver/experiments/wordread/reader/offsets.json`, not committed)
- Our boxes (8,794) were matched to Terra's 2000 words (10,036) at IoU ≥ 0.5.
- There is one distribution per held-out quarter: the pairs whose box is outside that quarter and whose word touches neither it nor an answered block.
- The file is 835 KB and holds every pair's offsets. Means are shown with their sd in brackets.

| held out | pairs | start | end | cross | angle ° | height ratio | line_share (words) | neighbour_share | turned |
|---|---|---|---|---|---|---|---|---|---|
| nw | 2878 | 0.088 (0.238) | -0.100 (0.240) | -0.003 (0.058) | 2.11 (8.00) | 0.925 (0.142) | 0.208 (598) | 0.030 | 881 |
| ne | 2978 | 0.096 (0.208) | -0.104 (0.241) | -0.001 (0.056) | 1.77 (8.12) | 0.929 (0.140) | 0.204 (608) | 0.031 | 866 |
| sw | 2581 | 0.095 (0.238) | -0.095 (0.250) | -0.004 (0.061) | 1.51 (9.48) | 0.948 (0.155) | 0.214 (552) | 0.040 | 986 |
| se | 2382 | 0.090 (0.248) | -0.089 (0.233) | -0.002 (0.057) | 2.30 (9.44) | 0.950 (0.153) | 0.188 (448) | 0.037 | 623 |

Quantiles for NE held out (5%, 25%, 50%, 75%, 95%):
- start: -0.076, 0.006, 0.068, 0.119, 0.398
- end: -0.461, -0.126, -0.067, 0.000, 0.075
- cross: -0.089, -0.032, 0.000, 0.031, 0.082
- angle: -4.0, -0.4, +1.3, +3.1, +7.9
- height ratio: 0.74, 0.85, 0.92, 1.00, 1.14

Tails and neighbours (NE held out):
- The angle offset exceeds 10° in 4.0% of pairs and 30° in 2.1% (maximum 87.9°).
- Ends beyond half a height occur in 0.2% of pairs.
- 93 boxes hold part of a neighbour. The share of the neighbour inside the box has deciles 0.004 (10%), 0.175 (50%) and 0.997 (90%).

So our trimmed boxes are usually tighter than Terra's: they start about 0.09 heights late, end about 0.10 early, and are about 7% shorter. They are also tilted about 1.3–2° counter-clockwise of Terra's integer angles.

## Own-box crop counts (real footprints; answered blocks out)
- 4,402 boxes matched at IoU 0.5. Task 1 counted 4,401 matched words; one word is matched by two boxes.
- 3,585 remain after the answered blocks are removed.

| held out | crops | block number | street name | zoning code | other | turned |
|---|---|---|---|---|---|---|
| nw | 2850 | 2243 | 436 | 160 | 11 | 872 |
| ne | 2959 | 2438 | 352 | 164 | 5 | 858 |
| sw | 2550 | 2017 | 322 | 196 | 15 | 976 |
| se | 2357 | 1879 | 274 | 190 | 14 | 617 |

Masks blank a mean 4.1% of an own crop's px (median 1.2%). Building all of them, with masks, took 9 s on 1 thread.

## Contact sheet (to check by eye; not committed)
`pipeline/data/solver/experiments/wordread/reader/crops_ne.png` shows 24 random own crops with NE held out. The columns are Terra's crop | own | own masked | sloppy as drawn | sloppy with a line forced | sloppy with a neighbour forced.
- The own crops show our real faults: GREENWICH loses its G, and 993 loses half of its last 3.
- The masked crops are nearly identical to the unmasked ones.
- The sloppy crops look right, apart from the two concerns below.

## Concerns
1. **The bank's line pieces are thicker than real crossings.**
   - I measured it with NE and the blocks held out, as twice the distance transform's maximum: 3,782 bank pieces (10/50/90/99%: 4, 6, 12, 26 px) against the line ink inside 608 crossed matched words (2, 4, 8.4, 10.8 px).
   - 18% of the pieces are thicker than the 99th percentile of real crossings. Some are solid blobs that hide a whole word while the target is still its string (6307 and 6731 on the contact sheet).
   - At sloppy 0.3 × line 0.2 this is a small but real source of label noise.
   - A measured fix would be to draw pieces with weights that match the crossing-thickness distribution. That is a design call for Task 4 or the controller; I did not add it.
2. **A forced neighbour can land on the letters.**
   - The fragment's share is drawn separately from the box geometry. When the crop rectangle doesn't extend toward the neighbour, a large share (the 90th percentile is about 1.0) can only be reached by overlapping the word.
   - At the measured 3–4% rate the effect is small.
   - An option is to draw each pair's neighbour share jointly with its offsets (one more per-row column), so the neighbour sits in the measured extension.
3. **Few zoning codes.** Own crops are mostly block numbers (about 80%); zoning codes are only 160–196 per quarter. Task 4 may want to weight by kind.
4. **Stages made stale on paper.** `crops.py` is in the closure of `solver-text-crops` and `solver-text-read`, and `train.py` in `solver-text-train`, so all three are stale on paper (the brief named only text-train). None was rebuilt, and their outputs would be byte-identical if they were.
5. **`read.build_read(mask=True)` reads files the stage doesn't declare.** It reads `data/solver/text-reader/<sheet>/chance.png`, which is not among the `solver-text-read` stage's declared reads. The stage itself never sets `mask=True`; it is for trials only.
6. **Small choices I made:**
   - Offsets are measured per held-out quarter, not once overall.
   - The touch flags are the union of the crop's footprint and Terra's word row.
   - Masking in training applies to own-box crops only. Sloppy crops and Terra's crops have no component to mask by.
   - A box exactly 90° from Terra's angle is not turned.


## Fix round 1 (the controller's five rulings)

Commit b19c933 (branch solver). The changes are in `wordread/readercrops.py`, `textread/train.py`, `textread/read.py` and `pipeline/tests/test_wordread_readercrops.py`. `crops.py` is untouched this round.

### 1. Line pieces no longer hide words
- **Thickness filter.** `piece_patches(bank, *, quarter, max_thickness, ...)` drops any piece thicker than the measured crossings' q99. The cap is `offsets.json` `line_thickness.q99`.
  - Thickness is `thickness(mask)`: twice the largest L2 distance to paper.
  - For crossings it is taken on the sheet's whole line ink, so a line only partly inside the word still gets its full width.
  - q99 per held-out quarter: nw 12.0, ne 10.79, sw 10.99, se 12.0 px.
- **Cover check.** `_line_through` rejects any draw whose piece px exceed `line_cover.q99` of the ink inside the word's rectangle, and redraws.
  - The measure on real words is line ink over word ink, on crossed matched words.
  - q99 per quarter: nw 0.75, ne 0.69, sw 0.75, se 0.78.
  - Redraws are capped at `LINE_TRIES` = 100. That is a loop guard only, and I count how often it is hit.
- **Real run, NE held out:**
  - 3,782 bank pieces, 632 (16.7%) dropped as too thick, 3,150 kept.
  - Over 5,393 sloppy crops, 1,176 got a line: a rate of 0.218 against `line_share` 0.204. The per-word generators' own draws give 0.2178, so this is the seeds, not the code.
  - 352 draws were rejected. 300 of them came from 3 crops that gave up after 100 tries, so 0.25% of line crops are left without a line. The other 52 were spread over 51 crops, a 4.2% rejection rate per draw outside the give-ups.

### 2. The neighbour comes from the same pair and never lands on the word
- **Joint draw.** Each pair's neighbour share is now a per-row `offsets["neighbour"]` column (0 for none), drawn together with that pair's offsets. There is no separate coin flip any more.
- **Placement.** The nearest other word is moved, and its old place is turned to paper wherever that is off the word. It moves along the word's reading axis or across it, trying its own side first. It goes to the farthest place where the pair's share of it lies inside the crop's rectangle, and never onto the word's rectangle. When the crop has no room beside the word, it goes as near as the cap allows.
- **Ruling deviation: the cap.** The measured `neighbour_on_word` (the held neighbour's rectangle on the matched word's rectangle, as a share of the neighbour) is not about 0.
  - With NE held out its quantiles are: median 0.03, q75 0.15, q95 1.0, q99 1.0.
  - The cause is loose Terra rectangles. Block numbers lie wholly inside slanted zoning codes' rectangles (RH-3/1265, RH-1/5879, RH-2/1290). None of the 13 pairs with overlap above 0.5 has the neighbour's text inside the word's.
  - A q99 cap of 1.0 would allow pasting onto the letters. I followed the ruling's "about 0" and "never on the word" literally: `ON_WORD = 0`. The measured distribution is kept in `offsets.json` for the record.
- **Real run, NE held out:**
  - 137 crops got a neighbour (2.5%), against 3.1% of pairs measured. The gap comes from words with no training neighbour, and from dropped crops.
  - The largest overlap with the word was 0.0.
  - The drawn share was reached (within 0.05) in 84 of 137 (61%); the rest were placed as near as the cap allows.
  - 90 went across the word and 47 along it.

### 3. The held-out quarter is now required
- `own_box_crops`, `sloppy_crops`, `offset_distribution` and `piece_patches` take a keyword-only `quarter`, and raise `ValueError` on `None`.
- `piece_patches` also raises unless every sheet of the bank holds out `data.Quarter(q)` and a `GridArea` equal to the answered blocks' area.

### 4. Kind mix
- `extra_pool(own, sloppy, kind_weights=None | "terra", kind_shares, real_masks)` builds `fit`'s `extra`; `own_draw_weights` does the weighting.
- `train.Pool` draws own crops with `rng.choice(p=own_p)` when it is given, and with `rng.integers` otherwise, as before. The default stays off; the pinned batches and the share-0 test still pass.
- `offsets.json` `kind_shares` are measured on the training quarters' Terra words, answered blocks out:

| held out | zoning code | block number | street name | other |
|---|---|---|---|---|
| nw | 0.134 | 0.540 | 0.276 | 0.051 |
| ne | 0.116 | 0.554 | 0.286 | 0.044 |
| sw | 0.141 | 0.556 | 0.263 | 0.040 |
| se | 0.142 | 0.567 | 0.247 | 0.045 |

- With NE held out, the kind-weighted draws match Terra exactly. Zoning codes go from 5.5% of own crops to 11.6% of draws.

### 5. Masks for every crop under `mask=True`
- **Component helpers.** `read.components` and `read.grown` now hold the component rebuild that `word_masks` used; `word_masks` is unchanged in behaviour.
- **Masks of Terra's rectangles.** `rect_masks`, `terra_masks` and `real_masks` take the components that meet Terra's rectangle, at the box level of the word's quarter (levels.json).
- **Training.** `Pool.joined(mask=True)` masks:
  - the real crops, by `extra["real_masks"]` (keyed by the crops.parquet row; `Pool.row_of`);
  - the own and sloppy crops, by their meta `mask`.
- **Real run:**
  - 8,500 of 12,043 2000-sheet crops got a mask, including 8,482 of the 10,036 words. 1,554 words meet no component and stay as cut.
  - The other two maps' 22,894 crops stay unmasked; there is no chance image for them.
  - 4,599 of 5,393 sloppy crops (NE held out) got a mask.

### Contact sheet
Regenerated at `data/solver/experiments/wordread/reader/crops_ne.png`. The rows are own crops drawn with the kind weights. The columns are Terra | Terra masked | own | own masked | sloppy | sloppy masked | line forced | held neighbour forced. What it shows:
- **Kinds.** The rows now mix codes (RH-2, RH-1), street names (ELMIRA, SAN, EDGAR, ST), an "other" (19 C) and block numbers.
- **Neighbours.** Forced neighbours sit beside the word, never on it: "ENW" under 939, "5957" above 5958, "12" left of ST.
- **Lines.** The solid blobs from round 0 are gone. But the q99 caps still allow heavy lines: a forced line hides most of 1547, and diagonals cross EDGAR and SAN.
- **Masks.** They remove nearby clutter, such as RH-2's frame and ST's border. They also remove a letter our detector missed: Terra's "6795A" becomes "795A", while the target keeps "6795A".

### Tests
- `test_wordread_readercrops.py`: 27 pass. New tests cover:
  - rejected-and-redrawn lines, and giving up at a cap of 0;
  - the neighbour moved beside the word (the exact shift and an exact page comparison);
  - neighbour rate and offsets over 500 draws, with `on_word` always 0;
  - the thickness filter and the bank's held-areas check;
  - kind-weighted draws through `fit`;
  - masked training of real, own and sloppy crops;
  - Terra-rectangle masks against the grown component;
  - `quarter=None` raising in all four functions.
- With `test_solver_textread.py` and `test_build.py`: 86 passed.
- Full `uv run pytest -q -n 2`: 1654 passed, 3 xfailed. Nothing failed in other agents' files.

### Concerns
1. **The q99 caps still allow heavy lines** (1547, EDGAR). Matching the whole measured cover distribution would be closer to the real crossings: accept each draw with the ratio of the measured cover density to the proposed one, rather than capping at q99. That is a call for the controller.
2. **The neighbour cap of 0 is not the measured q99 (1.0)**, for the reasons in section 2.
3. **Masking Terra's crops is heavy, and can remove real letters.**
   - Ink blanked in masked Terra word crops: median 4%, q75 24%, q90 48%.
   - More than a quarter of the ink is blanked in 24% of words.
   - Some of that is clutter, but some is letters our detector missed, as in 6795A, while the target string is kept. Task 4's mask arm should be read with this in mind.
4. **The other maps' crops stay unmasked under `mask=True`.**
5. **The neighbour reaches its drawn share in only 61% of crops.** In the rest, the crop's rectangle has no room beside the word.


## Fix round 2 (whole-distribution matching, masks of touching ink, no hand-set caps)

Commit 5ec4ced (branch solver). The changes are in `wordread/readercrops.py`, `textread/read.py`, `textread/train.py` (one docstring line) and `pipeline/tests/test_wordread_readercrops.py`. The round-1 thickness filter and `ON_WORD` cap are removed, not kept alongside. All numbers below are with NE held out.

### 1. Line pieces follow the whole measured distributions
**Real cover, re-measured.** `line_measures` now finds the line ink inside Terra's rectangle that belongs to line-ink components running on outside the rectangle. The rectangle is grown by `zones.GROW_PX` first, which removes the line pixels' own growth rather than adding a new number. A letter's straight stroke stays inside the rectangle and no longer counts.
- A word is crossed when it holds any such ink. The old 3-pixel threshold is gone.
- Cover is that ink over the word's ink.
- Thickness is measured on the sheet's whole line ink, where it crosses.

| measure (q10 / 50 / 90 / 99) | new: crossing components | old: every line px, ≥ 3 |
|---|---|---|
| `line_share` | 0.217 | 0.204 |
| cover | 0.021 / 0.227 / 0.512 / 0.690 | 0.045 / 0.242 / 0.516 / 0.691 (q05 / 50 / 90 / 99 of the old summary) |
| thickness | 2.0 / 4.39 / 8.79 / 10.79 px | (unchanged in kind) |

Letter strokes were not inflating the cover: real crossed words really do have about half their ink as line ink at the 90th percentile.

**Calibrated draws (`Lines`, `calibrate_lines`).**
- **Thickness.** Each piece is weighted by the ratio of measured to bank share for its thickness. The distance transform gives discrete thickness values, so they are matched value for value. 352 of the 3,782 pieces have a thickness no measured crossing has, so their weight is 0: the data decide what is excluded, not a filter.
- **Cover.** Each draw is kept with the chance that turns the proposed cover histogram into the measured one. The bins are Freedman–Diaconis on [0, 1], 14 of them.
- **Raking.** Keeping draws by cover shifts the thicknesses that survive, so the piece weights are raked 4 times against the kept thicknesses. Each round simulates 20,000 draws over the 5,417 training words. Calibration takes 28 s.
- **Redraws.** A rejected draw is drawn again (2.29 draws per line), so the line rate stays `line_share`.

**Realised against measured** (sloppy crops for every training word with a line forced: 5,403 crops):

| q10 / 50 / 90 / 99 | measured | realised |
|---|---|---|
| thickness | 2.0 / 4.39 / 8.79 / 10.79 | 2.0 / 4.39 / 8.79 / 10.79 |
| cover | 0.021 / 0.227 / 0.512 / 0.690 | 0.052 / 0.249 / 0.520 / 0.701 |

- Calibration total-variation distance: thickness 0.018, cover about 0.
- The low tail of cover stays a little high: a piece anchored on a letter always covers at least a few pixels.
- Give-ups: 3 of 5,403 crops (0.056%) went 100 draws without a keep. The 100 is a loop guard only.
- The realised line rate is 0.231 against `line_share` 0.217. That is the per-word seeds' own draws (checked in round 1 with the same generators), not the code.

### 2. Masks never blank ink that touches the word
- **The rule.** `read.kept_ink` keeps every ink component that holds an ink pixel inside the crop's own rectangle. Components are grouped as in `wordboxes` (ink grown by `INK_R`, 8-connected) but from all binarised ink, with no chance level. Only components wholly outside the rectangle are blanked.
- **Which rectangle.** Our box's trimmed rectangle for own crops and at read time (`read.box_crops(mask=True)`). Terra's rectangle for Terra's crops (`real_masks`, now on every map). The drawn rectangle for sloppy crops, computed after the paste, so pasted lines are kept.
- **Removed.** The chance-based masks (`word_masks`, `components`, `grown`, `rect_masks`, `terra_masks`). At mask=False, `read.py` is byte-identical: `box_crops` calls `crops.box_crop` as before.
- **Real results:**
  - Ink lost inside Terra's rectangle: 0 of 1,434 checked words lose any, so the share losing more than 25% is 0. This holds by construction.
  - Masks cover 34,921 of 34,937 crops on all three maps, including 29,144 of 29,160 words; the 16 left have no ink in their rectangle.
  - The mean share of a crop blanked is 10.7% of its pixels, mostly paper already. Only 0.24% of its ink is blanked on average (q99 7.8%).
  - Own crops: 4.1% of pixels blanked.

### 3. Neighbours placed from the real pairs, ink included, with no cap
- **Two measures per pair, from ink rather than rectangles** (`ink_overlaps`, the same function on real pairs and realised pastes):
  - **touch:** the share of the neighbour's ink in ink components that also hold the word's own ink;
  - **on:** the share of the neighbour's ink inside the extent of the word's own ink along and across its reading direction.
- **The word's own ink.** On a real page it is the word's rectangle's ink off the neighbour's rectangle (the best one ink layer allows). For a paste it is the word's ink before the paste.
- **Joint draw.** Both are drawn with each pair's offsets and share, as `offsets["neighbour_ink"]` and `["neighbour_on"]`.
- **Placement.** The neighbour is moved (its old place made paper) to the whole-pixel shift along or across the word's reading axis whose (touch, on) is nearest the pair's. Among those equally near, it takes the shift whose share inside the box is nearest. Ties go to the own side, then the farthest shift.
  - The ink comes first because, with an equal-weighted sum, filling the share pulled neighbours onto the letters.
  - This ordering is a choice of objective, not a cap.

**Measured against realised** (137 neighbours in 5,393 sloppy crops):

| q10 / 50 / 90 / 99 | measured | realised |
|---|---|---|
| touch | 1.0 / 1.0 / 1.0 / 1.0 | 0.0 / 1.0 / 1.0 / 1.0 |
| on | 0.0 / 0.0 / 0.56 / 1.0 (share at 0: 0.53) | 0.0 / 0.0 / 0.28 / 0.72 (share at 0: 0.77) |
| share inside the box | 0.004 / 0.175 / 0.997 / 1.0 | 0.0 / 0.019 / 0.247 / 0.638 |

- Measured `on` is not about 0. Its median is 0 and q90 0.56. A loose rectangle's line ink or a nested label widens the extent of a word's own ink.
- The realised `on` sits below the measured, and the realised share inside the box well below. A sloppy word's own neighbour (its size and side) often differs from the measured pair's, so the drawn share can't always be reached without overprinting.

### Contact sheet (`data/solver/experiments/wordread/reader/crops_ne.png`, regenerated)
- **Masks.** The Terra-masked and own-masked columns now equal their unmasked neighbours. 6795A keeps its "6" in the Terra column; the own column lacks it before masking, because our box itself cut it off.
- **Lines.** They now span the measured range: thin strikes (1436, 5958, 5865, 2778) up to heavy diagonals (EDGAR, SAN, 19 C, 6795A). The heavy ones are within the real crossings' cover distribution (q90 0.51).
- **Neighbours.** They sit beside the word, touching it: "CH939", "2778" with a trailing 2, "5958" with "59", "6300" with a mark at its left.

### Tests
- `test_wordread_readercrops.py`: 28 pass. New or rewritten tests cover:
  - `kept_ink`: the line kept however far it runs, the neighbour and stray mark blanked;
  - reading masked against unmasked;
  - `ink_overlaps` on loose nested, joined and overprinted neighbours;
  - `offset_distribution`'s crossing line measures and touch/`on`;
  - the calibration: no 21 px band drawn, thickness TV < 0.05, cover TV < 0.1, and kept draws in the measured shares through `sloppy_crop`;
  - draws kept by cover, redrawn, and giving up;
  - the neighbour placed by (touch, `on`) then share: the exact shift and page; with no room, it stays off the letters;
  - a sloppy crop's mask keeps its pasted line;
  - Terra crops masked on every map;
  - the bank's held-areas check, with no filter any more.
- With `test_solver_textread.py` and `test_build.py`: 87 passed.
- Full `uv run pytest -q -n 2`: 1677 passed, 3 xfailed, 17 failed.
  - All 17 failures are in `tests/test_wordread_words.py` (Task 6's committed test). It calls `shrink.prepare([sd], R)`, but the committed `wordread/shrink.py` (Task 5) now takes one argument (`TypeError: prepare() takes 1 positional argument but 2 were given`).
  - None of my paths is involved.

### Concerns
1. **Neighbours under-reach their drawn share** (median 0.02 against 0.17 measured), and `on` comes out lower than measured. The ordered objective trades share for ink realism. A likelihood under the measured joint distribution would weigh the three together, but needs a density choice.
2. **`on` on real pairs is inflated by loose rectangles** (line ink or a nested label widening a word's ink extent). So "measured `on`" includes some cases that are not really on letters.
3. **Heavy real crossings remain heavy.** Matching the real cover distribution means about 10% of line crops cover half or more of the word's ink, as real crossed words do.
4. **The cover distribution's low tail is a little high** (q10 0.05 against 0.02): a piece anchored on a letter always covers some pixels.


## Fix round 3 (neighbours from the empirical joint)

Commit 2c2b9a9 (branch solver), in `wordread/readercrops.py` and its test file. All numbers below are with NE held out.

### What changed
- **Poses.** `offset_distribution` (with pages) keeps, for each training pair that holds a neighbour, that neighbour's pose (`neighbour_pose`, in `offsets["neighbour_pose"]`):
  - the neighbour's own ink sampled on a whole-pixel grid in the word's frame (along its reading direction, measured from its nearer end, and down its letters, in the real page's pixels);
  - its rectangle in that frame, and the word's height.
  - A pair whose neighbour touches the held-out quarter or the answered blocks keeps no pose, so it is never drawn.
- **Drawing.** `sloppy_crop` draws one pair (row k). When that pair has a pose, the real neighbour is ORed in at its own place, angle and scale in the sloppy word's frame, scaled by the height ratio and placed past the same end (`_paste_neighbour`, nearest pixel). Its share inside the box and its ink on the letters therefore come together from the same real example, with no search, fitting or conditioning.
- **Removed:** the round-2 move-and-search (`_neighbour_into`), `with_neighbours`, and the moving of the sloppy word's own nearest neighbour.
- **Coverage.** 69 of the 93 pairs holding a neighbour have a pose, which is 2.32% of the 2,978 pairs. The other 24 have their neighbour in a held-out area, or have no ink of their own.

### Realised against measured (5,393 sloppy crops; 98 drew a neighbour)

| q10 / 50 / 90 / 99 | measured (posed pairs) | measured (all holding pairs) | realised |
|---|---|---|---|
| ink on letters (`on`) | 0 / 0 / 0.30 / 0.82 | 0 / 0 / 0.56 / 1.0 | 0 / 0.02 / 0.53 / 0.74 |
| share inside our box | 0.005 / 0.32 / 0.99 / 1.0 | 0.004 / 0.17 / 0.997 / 1.0 | 0 / 0.09 / 0.99 / 1.0 |

| share of draws not touching the word's ink | measured (posed) | measured (all) | realised |
|---|---|---|---|
| zero-touch | 0.029 | 0.022 | 0.032 |

- **Neighbour rate.** 1.82% of crops drew a neighbour, against 2.32% of rows having a pose. The per-word seeds also gave the line rate +2.5 sd over `line_share` (0.231 against 0.217; checked in round 1).
- **Scale ratios** (sloppy word over the pair's word, q10/50/90/99): 0.27 / 0.60 / 1.46 / 3.72.

### Contact sheet (`data/solver/experiments/wordread/reader/crops_ne.png`, regenerated)
- **The neighbour column** (last) shows real neighbours at their real offsets. They are mostly small lot or block numbers tucked under or beside the word: "1375" under 1436, "1695" under 2778, "1683" under ELMIRA, a "10" after 6795A, fragments left of 2099 and 5746. A few are heavier: a bar above 939 and 7064A, and a strongly rotated draw for 7239A. None sits squarely on the letters.
- **The other columns** are as in round 2.

### Tests
- `test_wordread_readercrops.py`: 28 pass. New tests cover:
  - the pose kept in A's frame (E's ink cells, 2 px past A's end), and none when E is in the answered blocks while the measured share stays;
  - A's pair drawn for D: E comes 2 px past D's end, page-exact, and its share and ink come out as measured on A;
  - the neighbour scaling with the word's height (doubled rectangle and gap);
  - the neighbour coming past a 180° word's end on its left.
- With `test_solver_textread.py` and `test_build.py`: 87 passed.
- Full `uv run pytest -q -n 2`: 1679 passed, 3 xfailed, 17 failed, 2 errors. None involves my paths:
  - the 17 failures are all in `tests/test_wordread_words.py`, another agent's rework;
  - the 2 errors are import errors from other agents' untracked work in progress (`tests/test_wordread_boxmodel.py` and `tests/test_wordread_doubt.py`; the `wordread.boxmodel` and `wordread.doubt` modules are untracked).

### Concerns
1. **Anchoring at the nearer end trades share for faithfulness.** Where the sloppy word's length differs from the pair's word's, a neighbour that sat above or below the middle of a long word lands past a short word's end and outside its box. That lowers the realised box share (median 0.09 against 0.32).
2. **Realised `on` runs higher than measured in the middle** (q90 0.53 against 0.30). The sloppy word's letters lie elsewhere than the pair's word's, and scaling by the height ratio (0.27–3.7) enlarges some neighbours.
3. **Measured `on` is still inflated by loose rectangles**, as reported in round 2.


## Fix round 4 (no draw cap; the empty pose)

Commit 38b5e4f (branch solver), in `wordread/readercrops.py` and its test file.

### `LINE_TRIES` removed: a line draw is redrawn until one is kept
- **Lines.** `Lines` now refuses an `accept` with no chance of 1. `calibrate_lines`' acceptance always has one, so some cover is always kept.
- **The no-ink words.** The 3 NE crops that gave up at 100 draws in round 3 would have looped forever. Their Terra rectangles hold no ink ("755", "ST", "5898"), and the anchor fell back to any pixel of the rectangle, so every draw's cover was 1.0. That is a cover no real crossing has, and it is kept never; a million draws were checked.
- **The fix, with no cap.** A word with no ink in its rectangle has no letters to cross, so it gets no line. The anchor is a letter pixel only, and calibration skips such words too.
- **Real check** (NE held out, a line forced on all 5,403 crops): 5,399 get a line and the 4 without letter ink get none. Draws per line average 2.05 and never exceed 15.

| q10 / 50 / 90 / 99 | measured | realised |
|---|---|---|
| thickness | 2.0 / 4.39 / 8.79 / 10.79 | 2.0 / 4.39 / 8.40 / 10.79 |
| cover | 0.021 / 0.227 / 0.512 / 0.690 | 0.052 / 0.248 / 0.513 / 0.717 |

### The empty pose
- `neighbour_pose` now returns None when its sampled grid holds no ink cell, so no pose is stored and `_paste_neighbour` never sees empty cells.
- That case can't actually arise for a neighbour with ink of its own. The grid's first column runs exactly through its first ink pixel along the word, and its nearest row is within half a pixel across, which rounds back onto that pixel at any slant. So the guard is defensive.
- The new test checks the property over 200 random slants and layouts: a pose exists exactly when the neighbour has ink of its own, holds at least one cell, and pastes. It also checks None for no ink, and None when the grid is forced off the window.

### Tests
- `test_wordread_readercrops.py`: 29 pass.
- Full `uv run pytest -q -n 2`: 1730 passed, 1 failed, 3 skipped, 3 xfailed.
  - The failure is `tests/test_wordread_boxmodel.py::test_the_gap_is_along_the_words_reading_direction_whatever_the_boxs_axis`, inside another agent's untracked `wordread/boxmodel.py` (`held_pair_gaps`).
  - None of my paths is involved.
