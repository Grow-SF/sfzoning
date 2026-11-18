# Task 5 report: detector word heads and their targets

Status: **DONE_WITH_CONCERNS**. The heads, targets, losses and shrink code are built, tested and off by default. **But
the brief's rule for r has no answer on Terra's words:** at every r in [0.3, 0.7], 12-18% of neighbouring pairs have
overlapping cores. The cause is Terra box errors, not r (see "r"). A ruling is needed before Task 6 or 8 uses r.

Commit: 87f3d82 (branch `solver`): `wordread/shrink.py`, `textnet/{data,model,train}.py`,
`tests/test_wordread_heads.py`.

## What I built

**`pipeline/src/sfzmaps/solver/wordread/shrink.py`** (new):
- `core(rect, r)` and `unclip(core_rect, r)`, on one rectangle (4, 2) or many (..., 4, 2):
  - `core` shrinks by DBNet's D = A(1 - r^2)/L on every side. The corners keep their order, so the first edge still
    runs along the reading direction.
  - `unclip` is the exact inverse: the positive root of 4(1 + r^2)D^2 + 2r^2(w' + h')D - (1 - r^2)w'h' = 0.
  - Both treat any 4 corners in cyclic order as a rectangle (centre, axes, extents), so `cv2.boxPoints` output works
    too.
- `cores_px(words, r)`: the cores' corners rounded to int32.
- `prepare(sheets, r)`: fills `SheetData.cores` through `dataclasses.replace`, sharing everything else (the
  `zones.prepare` pattern).
- `measure_r(words_by_sheet) -> dict`:
  - Neighbours: two words of one sheet whose rectangles are at most the smaller word height apart (gap 0 if they
    touch or overlap). Found by an axis-box cut, then the exact gap (SAT test plus corner-to-edge distances).
  - Overlapping cores: the two cores, painted as the detector learns them (rounded corners, `cv2.fillPoly`), share a
    px.
  - `curve` per r of 0.3, 0.35, ..., 0.7: overlapping count and share, plus a `touching` count and share (a px of one
    on or next to a px of the other, for Task 6's components).
  - `r`: the largest grid r with share at most 1%, or None.
- `write_r()` (`python -m sfzmaps.solver.wordread.shrink`):
  - Measures each fold on the training maps (`paths.TEXT_TRAIN_SERIES`, 87 sheets).
  - Leaves out words whose centre px is in the fold's quarter or an answered block (715 blocks). This is
    `reader_stage.measure_radius`'s rule.
  - Writes `pipeline/data/solver/experiments/wordread/detector/shrink.json` (not committed).

**`textnet/data.py`** (no new import):
- `SheetData.cores`: int32 (n, 4, 2), default an empty (0, 4, 2) array. Added to `BOX_FIELDS`.
- `crop_targets(..., words=False)`. With `words=True`, it adds:
  - `tag_id` (int32): 1 + the word index, painted from `sd.cores` with the existing `_paint`, later over earlier;
  - `core` (uint8) = `tag_id > 0`;
  - `geo` (float32 (5, h, w)), computed in numpy at page px (`_geometry`):
    - the distances to the top, bottom, start and end edges of the word's **full** rectangle (`sd.words`). Edges
      come from `sd.angles`: along = (cos, -sin), up = (-sin, -cos). Each edge is the furthest corner that way.
    - the reading angle.

    `geo` is resized nearest with the other maps, its distances scaled by size/src, and turned on axes (1, 2). The
    angle turns with the crop, as `angle` does.
  - With `words=True`, cores not matching the words (e.g. empty) raise a ValueError that names `shrink.prepare`.
- The `words=False` path is the old code exactly; the new branch comes after it.

**`textnet/model.py`**:
- `CORE = 7`, `GEO = 8-11`, `TAG = 12-15`, `WORD_OUTPUTS = 16`.
- `HEAD_OUTPUTS = {"all": 7, "text": 1, "words": 16}`.
- `distances(raw) = GEO_UNIT · exp(clamp(raw, ±20))` with `GEO_UNIT = 8` px.
- `OUTPUTS = 7` is unchanged.

**`textnet/train.py`** (no new import):
- `CHOICES["heads"]` gains `"words"`. `build_net` reads `model.HEAD_OUTPUTS`. No new `Recipe` field, so the pinned
  field dict is untouched.
- `heads="words"` with `composites > 0` is refused: composites don't make word targets.
- `draw` passes `words=True` only for `heads="words"`. The default call is the same as before.
- `word_tensors(crops)` returns core, geo, tag, known and groups:
  - Each crop's known core tags are renumbered 1..K.
  - `groups = arange(1, K_max + 1)`, so the loss knows K from a shape, without reading the device.
- `word_loss(out, core, geo, tag, known, groups)` returns the total and `WORD_PARTS = (core, dice, iou, direction,
  pull, push)`. Total = core + dice + iou + 10·direction + 0.25·(pull + push):
  - core: BCE over known px, all positives plus the top k = min(#neg, 3·#pos) negatives over the batch (DBNet's
    batch-level OHEM). The k-th loss is found by a no-grad `sort` and `index_select`, so there is no device read.
    Negatives tied at the k-th value share the remaining places, which is deterministic and gives the same value as
    top-k. No positives gives 0.
  - dice: 1 - 2Σptm/(Σpm + Σtm + 1e-6) over known px.
  - iou: -log((I+1)/(U+1)) of the predicted vs the target distances, the mean over known core px.
  - direction: 1 - cos on channels 5-6 at known core px.
  - pull and push, PAN's terms (Apache PAN++ code read for the formulation, re-expressed):
    - pull: log(1 + max(0, |t - mean| - 0.5)^2) per core px, averaged per word, then over each crop's words, then
      over crops;
    - push: log(1 + max(0, 3 - |mean_i - mean_j|)^2) over each crop's pairs, then over crops.

    Each crop's words come from a dense one-hot (n, K, h·w) and `bmm`: no gather or scatter in the backward, which
    the deterministic mode refuses or doesn't cover on mps.
- `_batches` appends `word_tensors` for `heads="words"`.
- `fit`:
  - unpacks `x, t, k, a, w, *words`;
  - computes `out = out.float()` once, then `loss_fn(out, t, k, a, w)` exactly as before, then adds `word_loss` when
    there are word tensors;
  - logs and checks the parts by name. For the default recipe the log and error strings are the same as before.
- The per-ink head (channels 0-6) is trained by the unchanged `loss_fn`. The fit test patches `loss_fn` and sees it
  called with the 16-channel output on every step.

## TDD evidence

**Tests:** `pipeline/tests/test_wordread_heads.py`, 15 tests:
- `unclip(core)` comes back within 0.5 px, for 4 values of r and 6 rectangles (angles 0, 90, 30, -45, 170, and a tiny
  one), one at a time and batched.
- `core` matches the paper's worked example (60×20 at r = 0.4 gives D = 6.3) and keeps the reading angle (90° stays
  90°).
- `measure_r`:
  - it gives the largest r at most 1%: 2 of 100 pairs overlapping from r = 0.55 gives r = 0.5, and 1 of 100 gives 0.7;
  - the exact curve;
  - words 201 px apart are not neighbours, 200 px apart are.
- Two touching words, a touching upward street name and a line of text 1 px below get **disjoint, non-touching cores
  at the measured r**: 4 separate 8-connected components.
- `geo` at a core px of a 0° word is [15, 15, 50, 50, 0] and [10, 20, 20, 80, 0]. For a 90° word (top on the left,
  start at the bottom) it is [10, 20, 20, 80, 90].
- `geo` doubles at scale 2. Under a quarter turn the distances turn with the maps and the angle becomes 90.
- `words=True` without cores raises. `words=False` targets are byte-identical with or without cores. With
  `words=True` the shared keys are equal.
- The recipe choice and its name (`crop=32-heads=words-width=4`), `HEAD_OUTPUTS`, 16-channel output at full and half
  size, and the composites refusal.
- `word_tensors` renumbering, with a held-out crop giving no tags.
- Hard negatives: the top 6 for 2 positives, with a 3-way tie at the boundary (the loss equals the sort-and-take-6
  value), and 0 with no positive.
- `word_loss`:
  - is about 0 for a perfect answer;
  - push = log 10 when all tags are equal;
  - the total equals the parts times their weights.
- A 2-step `heads=words` fit on CPU:
  - gives identical weights twice;
  - calls `loss_fn` with the (3, 16, 48, 48) output each step;
  - the word loss's gradient reaches every word output channel (7-15).

**Red:** `uv run pytest -q tests/test_wordread_heads.py` before any implementation: `ImportError: cannot import name
'shrink' from 'sfzmaps.solver.wordread'`, collection error.

**Green:** 15 passed.

**Mutation check** (scratch script; each wrong variant must fail a test). All three are caught:
- top and bottom swapped in `_geometry`: caught by the geo test;
- plain BCE in place of hard-negative mining: caught by the hard-negatives test;
- half of DBNet's D: caught by 3 tests.

## r

**Rule (the brief's):** the largest r in [0.3, 0.7], step 0.05, at which at most 1% of neighbouring Terra-word pairs on
the training quarters have overlapping cores.

**Measured** (`shrink.json`): r = **None in every fold** (`"r": null`).

Each cell gives the share of neighbouring pairs with overlapping cores, then in brackets the part added beyond those
already overlapping at r = 0.3:

| fold held out | pairs | words | r 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nw | 7200 | 34135 | 12.5% (+0) | 12.8% (+0.35) | 13.4% (+0.96) | 14.1% (+1.61) | 14.4% (+1.97) | 15.0% (+2.56) | 15.4% (+2.94) | 15.9% (+3.43) | 16.5% (+4.08) |
| ne | 6029 | 28468 | 13.6% (+0) | 13.9% (+0.35) | 14.6% (+1.01) | 15.2% (+1.59) | 15.6% (+2.01) | 16.1% (+2.55) | 16.5% (+2.95) | 17.0% (+3.42) | 17.8% (+4.26) |
| sw | 6180 | 32072 | 12.0% (+0) | 12.4% (+0.36) | 13.0% (+0.95) | 13.6% (+1.59) | 14.0% (+2.01) | 14.6% (+2.59) | 15.1% (+3.07) | 15.6% (+3.56) | 16.4% (+4.37) |
| se | 5847 | 29405 | 12.5% (+0) | 12.8% (+0.32) | 13.4% (+0.89) | 13.9% (+1.44) | 14.3% (+1.81) | 14.9% (+2.41) | 15.2% (+2.74) | 15.7% (+3.18) | 16.3% (+3.78) |

Pairs with touching or overlapping cores add only about 0.5-1 point (`touching_share` in the json).

**Why: the overlapping pairs are Terra box errors, not neighbouring words.** All 87 sheets, no fold held out: 1,617
pairs have cores overlapping at r = 0.3 (scratch `overlaps.py`):
- 56% are other/other (titles and legends), 20% street/street, 8% street/code;
- the rectangles' IoU median is 0.18 (90th percentile 0.27), all under `boxes.SAME_TEXT_IOU` = 0.3, so
  `terra_boxes`' dedup keeps them;
- only 2.9% have the same text.

A random 25 shows two error types:
- **Double reads at tile seams:** "OF THE CIT" with "CITY AND", "AND COUNTY O" with "NTY OF SAN FRAN",
  "INDUSTRIAL DIS" with "L DISTRICTS", "FAIR" with "FAIRBANKS".
- **An oversized box holding the next word:** "ANZA ST" with "ST.", "42ND AVE" with "AVE.", "WINSTON DR" with "DR.",
  "RH-2" with "P", "JULY 10, 1964" with "54".

Each pair is one piece of print read twice, so its cores overlap at any r. Ten percent of these pairs involve a word at
the forced 14 px height of `word_rect`'s near-45° case.

**What r does decide** is the increment, about 0.3-0.4 points per 0.05 step. Read with a 1% bound, r = 0.4 passes in
nw, sw and se (0.96, 0.95, 0.89%) and fails in ne by 0.01 point (1.01%), so ne gets 0.35. DBNet's paper default is
0.4; PAN uses 0.5 or 0.7.

**I did not change the rule or pick an r.** For the controller to rule:
- **(a)** r = 0.4, the DBNet default, which the increment reading also lands on (1.01% in NE).
- **(b)** Measure on words with Task 7's doubt flags removed. "contradicted" and "ink_outside" would likely catch the
  seam double reads, but Task 7 comes later.
- **(c)** Restate the rule as "at most 1% of pairs newly overlapping beyond r = 0.3": 0.35 for NE, 0.4 for the
  others.

All four folds behave alike, so one r for every fold looks safe.

## Defaults unchanged (guard outputs)

- `uv run pytest -q -n 2 tests/test_solver_textnet.py tests/test_textnet_*.py tests/test_build.py`: **157 passed**.
- Run one by one, each PASSED and none skipped:
  - `test_crop_targets_equal_full_page_drawing`;
  - `test_sheet_data_holds_no_full_page_target_rasters`;
  - `test_default_recipe_trains_as_today` (the Recipe field dict is unchanged; no new field);
  - `test_the_stages_recipe_trains_the_weights_recorded` (`GUARD_WEIGHTS` and `GUARD_LOSSES`, on Darwin arm64, key
    "1","1");
  - test_build.py: `test_committed_dvc_yaml_is_the_generated_one`, `test_dvc_yaml_is_the_same_with_no_data`;
  - test_textnet_composite.py: `test_composites_zero_leaves_batches_unchanged` (`PINNED_BATCHES_SEED3`),
    `test_composites_zero_draws_what_no_bank_draws`.
- The guard tests' files are untouched.
- Iterate set: `tests/test_wordread_*.py` and `tests/test_solver_textread.py`, 94 passed.
- Full suite (`uv run pytest -q -n 2`): 1622 passed, 12 failed, 3 xfailed, in 197 s. All 12 failures are in
  `tests/test_wordread_readercrops.py`, another agent's uncommitted work. Its tests call `readercrops` functions and
  arguments not written yet (`own_draw_weights`, `real_masks`, `sloppy_crops(masks=)`, `piece_patches(quarter=)`), so
  they don't come from this task.

## Concerns

1. **r has no answer under the brief's rule** (see above). `shrink.json` holds `"r": null`, the curves, and
   `"r_by_fold"`. Task 6 and Task 8 need a ruling: (a), (b) or (c).
2. **Not run on mps** (the brief said CPU only). Every word-loss op is dense: sigmoid, BCE, `sort` and `index_select`
   under no_grad, `bmm`, `eye`, masked sums. None is on the deterministic-mode list of ops that raise on mps. Task 8's
   first step should run a few steps of `heads=words` on mps inside the GPU turn and check that two runs repeat.
3. **Tag one-hot memory.** Words per training crop at r = 0.3 (1,200 draws of the default recipe): mean 7, median 4,
   99th percentile 35, max 49. The one-hot (12, K, 320²) float32 is up to about 240 MB in a K = 49 batch, about 35 MB
   typically. That is small beside the U-Net's activations, but it scales with the batch's densest crop.
4. **`predict.outputs` refuses a 16-channel network** (`o.shape[0] != model.OUTPUTS`). Word-head prediction (core
   chance, `model.distances` of GEO, the TAG vectors) is Task 6 or Task 8's to add. The distance decoding lives in
   `model.distances`.
5. **The distance form is my choice; the brief doesn't give one.** I used GEO_UNIT · exp(raw), FCOS-style, unbounded
   above where EAST's sigmoid × 512 is not, clamped to ±20 so float32 stays finite. GEO_UNIT is 8 px.
6. **PAN's pull covers the whole text region; ours covers only the core px.** This follows the brief ("tag_id at core
   px") and Task 6's split of core components by their tags. If Task 6 grows cores into text px by tag (PAN's pixel
   aggregation), the pull would need the word's whole rectangle.
7. **The word terms share `known`, rules included.** Under `labels="lines-unknown"`, line px inside a word are
   unknown for the core, geometry and tag too. A held-out word gives no tag. The core is read over all known px, not
   `ink_only`'s px, as the brief says.
8. **Hard negatives and push are pooled per batch.** OHEM pools the batch, as DBNet's code does. Push averages per
   crop, then over crops that have two or more words; PAN's code averages a crop with none as 0.
9. **Stages look stale on paper.** `data.py`, `model.py` and `train.py` are dependencies of the text stages, so
   `dvc status` will call them stale. No stage was rebuilt, and the outputs for the default recipe are unchanged (the
   guards above).
10. **Other agents' work in the tree.** `tests/test_wordread_readercrops.py` and others carry other agents' work.
    Only my 5 paths are committed.

## Follow-up: the controller's ruling on r, option (c), pooled (commit da3a149)

**Rule now:**
- `measure_r` counts per pair the neighbouring pairs whose cores overlap at r but not at the grid's first value,
  0.3. Those are `new` and `new_share` in each curve entry.
- The pairs already overlapping at 0.3 are reported apart as `floor` and `floor_share`. They overlap at every r:
  Terra's box errors, which are Task 7's doubt candidates.
- `choose(curve)` takes the largest r whose new share is at most 1%.
- `pool(measured)` sums the folds' pairs, floors and per-r counts, and chooses r from the sums. `write_r` writes the
  pooled r as `"r"`. Each fold's curve and r are kept under `"folds"` and `"r_by_fold"`.
- Module and function docstrings are updated.
- `data.py`, `model.py` and `train.py` are untouched.

**Test added:** `test_r_is_chosen_by_the_pairs_newly_overlapping_beyond_the_first_r_and_folds_pool`.
- Setup: 10 pairs overlap at every r (a 10% floor), plus 2 pairs that newly overlap from 0.55, among 100.
- r = 0.5. With 1 new pair, r = 0.7.
- Pooling with a 100-pair fold that has no new overlaps gives 2 new of 200, so r = 0.7.
- The test fails when `choose` uses the total share (the old rule); checked by patching it back.
- `tests/test_wordread_heads.py` plus `tests/test_build.py`: 59 passed.

**Re-measured** (`shrink.json` rewritten, 87 sheets, folds pooled): **r = 0.40**.

| | pairs | floor (overlap at 0.3) | new at 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 25,256 | 3,191 (12.6%) | 87 (0.34%) | **241 (0.95%)** | 394 (1.56%) | 493 (1.95%) | 639 (2.53%) | 740 (2.93%) | 859 (3.40%) | 1,042 (4.13%) |
| nw | 7,200 | 12.5% | 0.35% | 0.96% | 1.61% | 1.97% | 2.56% | 2.94% | 3.43% | 4.08% |
| ne | 6,029 | 13.6% | 0.35% | 1.01% | 1.59% | 2.01% | 2.55% | 2.95% | 3.42% | 4.26% |
| sw | 6,180 | 12.0% | 0.36% | 0.95% | 1.59% | 2.01% | 2.59% | 3.07% | 3.56% | 4.37% |
| se | 5,847 | 12.5% | 0.32% | 0.89% | 1.44% | 1.81% | 2.41% | 2.74% | 3.18% | 3.78% |

- By fold alone, r would be nw 0.4, ne 0.35, sw 0.4 and se 0.4. The pooled r used for every fold is 0.40.
- Counted pair by pair, the new counts equal the earlier curves' differences from 0.3 exactly, so cores nest as r
  grows.

## Follow-up 2: the owner's amendment, no picked core size (commit fd0e71e)

**Changed**, in `wordread/shrink.py` and `tests/test_wordread_heads.py` only:
- `prepare(sheets)` takes no r. It fills `SheetData.cores` with each word's own labelled rectangle (`sd.words`,
  shared, not copied), with no shrink.
- The module doc says this, and that touching words are told apart by the learned tags, the geometry votes (Task 6)
  and the learned box model (Task 15).
- `core`, `unclip`, `cores_px`, `measure_r`, `choose`, `pool`, `write_r` and `shrink.json` stay only as a record of
  the overlap measurement. The doc says so and says nothing uses r.
- Tests:
  - `test_touching_words_get_disjoint_cores_at_the_measured_r` is replaced by
    `test_prepare_fills_each_core_with_its_own_rectangle`: the cores equal `sd.words` (int32), the rest of the sheet is
    shared, and the input is left unchanged. On a crop of two touching words, an upward street name and a slanted
    line, the core target equals the text target, and `tag_id` equals the word map `data._words`, so the words are
    told apart only by their tags.
  - The other tests call `prepare([...])` with no r.
  - The `HEAD_OUTPUTS` dict restatement is removed.
- `tests/test_wordread_heads.py` plus `tests/test_build.py`: 59 passed.
- `data.py`, `model.py` and `train.py` are untouched; Task 7 is editing them.

**Concerns for the controller:**
1. **This breaks Task 6's committed tests.** `wordread/words.py` and `tests/test_wordread_words.py` (ac8c0f4) call
   `shrink.prepare([sd], R)` and use `shrink.unclip(..., r)` for `refine="unclip"`. With the new signature, 17 of
   `test_wordread_words.py`'s tests fail with a TypeError. I left their files alone, as told. Under the amendment,
   Task 6's unclip path, which relies on shrunk cores and r, needs a ruling and an update. Its votes path can work on
   full-rectangle cores.
2. **Stale docs in files I may not touch.** `data.py`'s docstring (line 22, "its rectangle shrunk") and the
   `SheetData.cores` comment still describe shrunk cores. A one-line fix for whoever holds `data.py`.
3. **Hand-set constants in the built word loss**, under the brief's new first constraint ("a published method's
   constant is not exempt; stop and report"). Built in 87f3d82, all in `train.py` and `model.py`:
   - `OHEM_RATIO = 3` (DBNet's 3:1);
   - PAN's `PULL = 0.5` and `PUSH = 3.0`;
   - `TAG_WEIGHT = 0.25`, `GEO_ANGLE_WEIGHT = 10`;
   - `GEO_UNIT = 8` px and the ±20 clamp in `model.distances`.

   The brief's own text gives the first five, so they were built as specified. They are reported here for the
   owner's rule; I did not change them (the files are off limits now). `shrink`'s record-only `MAX_SHARE`,
   `TOUCH_GAP`, `R_GRID` and the one-word-height neighbour rule decide nothing now.

## Follow-up round 2: no hand-set constants in the word loss (commit 6d1af69)

**How the doubt test file was committed:** another agent had uncommitted doubt-feature edits in
`tests/test_wordread_doubt.py`. The committed version is HEAD's file plus only my word-section edits, staged as a
blob with `git update-index --cacheinfo`. Their edits remain unstaged in the working tree.

Full suite (`uv run pytest -q -n 2`, before the commit): 1725 passed, 13 failed, 3 skipped, 3 xfailed. The 13 are all
in the other agents' uncommitted work:
- 5 in `test_wordread_boxmodel.py`: `boxmodel` has no `draw_gaps`, `held_pair_gaps` or `join` yet;
- 8 doubt-feature tests in `test_wordread_doubt.py`: their `doubt.py` rewrite needs `line_bits` and an `'edge'` key.

The controller's rulings, in `textnet/train.py`, `model.py`, `data.py`, `tests/test_wordread_heads.py`, and
`tests/test_wordread_doubt.py`'s word-target and loss tests. `wordread/shrink.py` and `wordread/doubt.py` are
untouched.

1. **Learned loss weights.** Homoscedastic uncertainty (Kendall, Gal & Cipolla 2018):
   `L_words = sum_i ( exp(-s_i) L_i + s_i / 2 )` over `WORD_PARTS = (core, dice, iou, direction, tag)`, one learned
   log-variance s_i a term.
   - This is their eq. 7 with s = log σ², the classification form. The regression form's ½ is a shift of s by log 2,
     so one form serves every term. The module docstring says so.
   - `train.WordLoss(nn.Module)` holds `log_var` (5) and `log_temp`, all starting at 0. `fit` trains them in their own
     AdamW group with no weight decay, under the same one-cycle schedule.
   - `fit` logs the learned weights and temperature at each log, and keeps them on the returned net as the plain
     attribute `learned_word_loss` (not in `state_dict`, so `predict.load_model` is unaffected). `save_model` writes
     them as `"word_loss"` only when present.
   - `TAG_WEIGHT` and `GEO_ANGLE_WEIGHT` are gone.
2. **Measured class balance, no mining.** `_balanced_bce`: BCE with each class weighed by the inverse of its weight in
   the batch, **per batch from the batch's own counts**: (weighted mean over core px + weighted mean over other px)
   divided by the number of classes the batch holds. Dice stays. `OHEM_RATIO` and `_ohem_bce` are gone.
3. **No margins in the tag term.** `_tag_ce` computes:
   - per crop, the classes are the background (the mean tag of the known px off the cores) and each word (its core
     px's mean tag);
   - each known core px gets the softmax of -|tag - mean|²/T over the classes its crop holds, T = exp(`log_temp`)
     learned, and the cross-entropy to its own word;
   - the result is the weighted mean over core px.

   It is dense: one-hot `bmm` for the means, |a-b|² expanded, a missing class masked to -inf, non-core rows zeroed,
   then `F.cross_entropy`. `PULL`, `PUSH`, `NORM_EPS` and `_tag_parts` are gone.
4. **`GEO_UNIT` and `GEO_LIMIT` are kept**, now documented in `model.py` as the distances' parameterisation and a float32
   guard, not decisions. `DICE_EPS` and `MASS_EPS` are documented as numerical guards.
5. **From Task 7's review:**
   - `crop_targets(words=True)` gains `geo_known` (uint8): 1 at the core px of words whose rectangle holds no held-out
     px, 0 elsewhere. The distances of the other words stay 0 in the targets, so the targets carry no information
     about the held-out area.
   - The IoU term reads `core·known·geo_known`. The direction term reads `core·known`, so the angle is kept for words
     reaching into a held-out area.
   - `_held_words(sd)` runs `_touch_held` for every word once per sheet and keeps the result on the instance next to
     its fields. It is keyed on the `held` and `words` objects, so `hold_out`/`replace` (a new SheetData) or a reset
     field finds it again.
   - `word_tensors` returns 6 values: core, geo, tag, known, geo_known, groups.
   - The pinned key set and the 5-tuple unpacks are updated deliberately.
6. **Defaults byte-identical.** `heads="all"` takes the unchanged path (`opt = AdamW(net.parameters(), ...)`, no
   `WordLoss`, the same `save_model` dict). All pass:
   - `test_the_stages_recipe_trains_the_weights_recorded` (`GUARD_WEIGHTS` and `GUARD_LOSSES`);
   - `test_default_recipe_trains_as_today`;
   - `test_crop_targets_equal_full_page_drawing`;
   - `test_sheet_data_holds_no_full_page_target_rasters`;
   - `test_composites_zero_leaves_batches_unchanged` and `test_composites_zero_draws_what_no_bank_draws`;
   - test_solver_textnet + test_textnet_* + test_build: 157 passed.

**Tests:**
- `tests/test_wordread_heads.py`, 19 passed:
  - new: the held-word cache (`_touch_held` called once for three crops, again for a replaced sheet, `geo_known` and
    zero distances on the reaching word);
  - new: class balance (the value; a heavier or larger background class changes nothing; with no core px, the
    background's mean);
  - new: perfect answer ≈ 0 on every term; with all tags equal the tag term = log(classes) exactly; the tag CE rises
    with the temperature;
  - new: Kendall's sum for given s, and the plain sum at s = 0; gradients reach every `log_var`, `log_temp` and word
    output;
  - **new fixed-value test** `test_word_loss_fixed_values`: a hand-built 6×8 crop (two words, one without distances,
    one px not known), sine outputs, s = (0.1, -0.2, 0.3, 0, 0.5), log T = 0.2. Parts and total are pinned (rel
    1e-5). The pinned values equal a separate float64 numpy reference written from the docstring formulas to within
    1e-7 relative (scratch `fixed_value.py`);
  - updated: the 2-step fit also checks the learned weights and temperature moved from their start, repeat run to
    run, and are saved; a default model saves no `word_loss`.
- `tests/test_wordread_doubt.py` word tests, 7 passed:
  - the 6-tuple unpacks;
  - `geo_known` checks in the held-out geometry test;
  - the pull/push weighting test becomes `test_the_tag_term_weighs_each_core_px_by_its_words_weight`: two weightings
    solve each word's summed CE, a third must agree, and scaled weights change nothing;
  - the no-geometry test now drives `geo_known` and expects direction gradients on that word (the angle kept);
  - the hard-negatives test becomes `test_the_core_cross_entropy_counts_a_weighted_core_px_by_its_weight`.

  These go beyond "only where geo_known changes", because rulings 2-3 removed the functions those tests called.
- The doubt file's 8 feature tests (`ink_outside`, double reads, `fit_doubt`, `prepare`) fail on the other agent's
  uncommitted `doubt.py` rewrite (`line_bits` required, `'edge'` key). Not from this change.

**Concerns:**
- The tag term's dense softmax builds (n, P, 1+K) tensors. With K up to 49 words in a crop, that is about 250 MB
  each, a few of them in the backward, so it is heavier than PAN's. Typical K is about 7. Task 8 should watch mps
  memory on the densest batches.
- Not run on mps (CPU only). `F.cross_entropy` with -inf entries and `bmm` are the same kinds of op the per-ink loss
  already uses on mps.
- Kendall's s_i on a term near 0 (e.g. a perfectly separated tag CE) keeps falling, which raises that term's weight.
  This is the method's known behaviour, logged each LOG_EVERY.

## Round 3: sign-free word geometry (commit 512a1dc)

Ruled on Task 8's measured evidence: under the quarter turns the reading sign is not learnable, and the signed
1 − cos loss gave the axis no gradient. Applies to `heads="words"` only; `heads="all"` is byte-identical.

**The channel layout of a `heads="words"` network** (`model.WORD_OUTPUTS` = 16; `model.AXIS`, `CORE`, `GEO`, `TAG`):

| ch | name | meaning |
|---|---|---|
| 0 | text | text logit (as "all") |
| 1-4 | kind | kind logits (as "all") |
| **5-6** | **AXIS** | **(cos 2t, sin 2t)** of the reading axis t (the reading angle mod 180°, counter-clockwise from left-to-right, image y down). Any length; only the direction counts. The angle is t = ½·atan2(ch6, ch5) mod 180. This applies to the per-ink and the word terms alike (it was (cos t, sin t), signed). |
| 7 | CORE | word-core logit |
| **8** | dx | offset from the px to its word's centre, **x** (image px, right +), **divided by the word's height** |
| **9** | dy | the same, **y** (image px, down +), over the height |
| **10** | log length | log of the word's length in px (its extent along the axis) |
| **11** | log height | log of the word's height in px (its extent across the axis) |
| 12-15 | TAG | the tag vector (unchanged) |

The old layout (8-11 as log(distance / 8 px) to top, bottom, start, end) is gone. So is `model.distances`, and
`GEO_UNIT` with it.

**Decoding:** `model.word_geometry(out)` takes `out` (..., 16, h, w), channels third from last, and returns
(..., 5, h, w):
- 0, 1: the centre offset in px, ch8·height and ch9·height (centre = (col + [0], row + [1]));
- 2, 3: length and height in px, exp of ch10 and ch11 (read within ±`GEO_LIMIT` = 20, a float32 guard);
- 4: the axis in degrees, in [0, 180).

**Targets** (`crop_targets(words=True)["geo"]`, float32 (5, h, w), at core px, 0 elsewhere):
- 0, 1: dx, dy as above, from the centre of the word's whole rectangle (`SheetData.words`; the mid of its extents
  along and across its reading angle). Divided by the height, so free of the crop's scale.
- 2, 3: log length and log height in crop px: page px plus log(size/src). The extents are floored at 1 px, a guard
  for log 0; such a word is physically impossible, since rectangles are at least 2 px each way.
- 4: the axis in degrees, in [0, 180), turned with the crop: (t + 90·turns) mod 180.

Under a quarter turn counter-clockwise (`np.rot90`), (dx, dy) turns as a vector, to (dy, −dx) per turn
(`data._turn_offsets`); the sizes stay. `geo_known` is kept. A word reaching a held-out area still has 0-3 zeroed and
its axis kept.

**Loss** (`WORD_PARTS = (core, dice, centre, size, axis, tag)`, 6 learned log-variances now):
- centre, size: **Gaussian NLL with the learned Kendall variance, one per group.** L = the mean over the group's 2
  coordinates and the known px with known geometry of (pred − target)²/2. Then exp(−s)·L + s/2 is exactly the mean
  per-coordinate Gaussian negative log-likelihood (up to its constant) with variance e^s. No hand-set weight.
- axis: 1 − cos 2(t' − t) at the known core px, Kendall-weighted as before.
- core, dice and tag are unchanged.
- The per-ink terms under `heads="words"`: `loss_fn` is unchanged and is given **2 × the reading angle**, so its
  1 − cos is 1 − cos 2Δt on the same channels 5-6. `heads="all"` passes `a` itself.

**For the owners who adapt** (I did not edit these files):
- `predict.word_outputs` calls `model.distances(o[GEO])` and stores o[5:7] as the direction. It must use
  `model.word_geometry(o)` and read 5-6 as the axis. Until then `tests/test_wordread_run_detector.py`'s word-output
  tests fail with AttributeError.
- `wordread/words.py` reads the 4 distances and a signed direction.
- `run_detector.py` and `variant.py` go through `predict.word_outputs`.

**Tests** (`tests/test_wordread_heads.py`: 28 passed):
- exact targets at 0° and 90°;
- every core px voting for the word's centre, with its length, height and axis, at 0, 30, 90, 179 and 181°;
- continuity across 180°: 179 and 181 differ by < 0.05 in 0-3, axes 179 and 1, a chord < 0.08 on the 2t circle;
- quarter turns 1-3: votes land on the turned centre, sizes stay, axis +90 mod 180; at scale 2, sizes + log 2 and
  offsets stay;
- `word_geometry` decoding (centre in px, sizes, axis either side of 0/180, batched);
- the fixed-value loss test, re-pinned and equal to a float64 numpy reference within 8e-8 relative (scratch
  `fixed_value3.py`);
- the fit test now also checks that `loss_fn` receives 2 × the batch's angle.

A mutation check confirmed it: the offsets turned the wrong way are caught at 1 and 3 turns (a half turn is the same
either way).

`tests/test_wordread_doubt.py`: 2 tests adapted to the new layout. "A word's size at each px" replaces the old
distance-sum check, and the weighted-mean check now covers (centre, size, axis). 20 passed.

**Guards** (all PASSED): `GUARD_WEIGHTS`/`GUARD_LOSSES`, the default recipe, crop-target equality, no full-page
rasters, both composite pinned batches. test_solver_textnet + test_textnet_* + test_build: 157 passed.

**Full suite**, run with `--ignore=tests/test_wordread_words.py` (its WIP hangs, as instructed): 1738 passed, 3
xfailed, 2 errors. Both errors are one collection error in `tests/test_wordread_run_detector.py`: `KeyError:
'centre'` at its line 182. That is another agent's uncommitted edit of that test, building `words.Spreads` from their
uncommitted `words.py`, which has a new `centre` field. It is not from this change. That file's word-output tests
will also need `predict.word_outputs` adapted to `model.word_geometry`.

## Centre head: why it learned nothing (the controller's test; CPU, one thread, no code changed)

**Cause:** half the core px in a training crop belong to words that the crop's border cuts. Their centre is not in
the input, so their large, unpredictable offsets dominate the centre term. With every px read under one variance,
the network learns nothing, not even for the words wholly in view.

It is not the targets, the wiring or the Kendall weighting. The centre overfits under Kendall and under fixed weights,
and on fresh crops it fails under both. The Kendall weight falling to 0.355 is a symptom: s settles at log(2L) for
the flat loss of predicting 0.

Scripts and outputs are in the scratchpad (`centre_head.py`, `centre_tail.py`, `box_vs_ink.py`, `*.log`,
`centre_head_*.json`). Data: sheet 2000-digital-zoning-p07, `crop_targets(words=True)`, the default network (width
24), AdamW at a constant `LEARNING_RATE` 2e-3.

1. **Targets checked independently.** At all 36,507 core px of 8 real crops, I took each px back through the turn
   (np.rot90 inverted) and the nearest resize (floor(x · src/size)) to its page px. I then took Terra's rectangle
   centre as the mean of its corners and its height as the second edge, and turned the offset with the crop. The
   largest |target − independent| is 2.3e-7 (corr 1.000). Channels 8-9 are dx and dy in the loss; there is no
   activation or clamp on them, and the height divides once. Correct.
2. **Overfit, 8 fixed crops (crop 160), 300 steps:** both arms fit.

   | arm | centre loss: step 1 → 100 → 300 | final corr dx / dy | slope dx / dy |
   |---|---|---|---|
   | Kendall | 1.018 → 0.060 → 0.032 | 0.982 / 0.980 | 0.98 / 0.95 |
   | fixed (weights 1) | 1.018 → 0.058 → 0.005 | 0.997 / 0.997 | 0.99 / 0.99 |

   The Kendall weights stayed between 0.93 and 2.0. Neither of the controller's two cases holds.
3. **Fresh crops each step** (`train.draw`: half on text, scale 0.6-1.6, turns; crop 160, batch 8). Scored on 16
   other crops, where predicting 0 costs 1.96. "Whole" means the 26% of held px whose word lies wholly inside its
   crop.

   | arm (steps) | held centre | corr dx / dy, all px | slope | corr dx / dy, whole words | slope |
   |---|---|---|---|---|---|
   | Kendall (250) | 2.00 | −0.00 / −0.03 | 0.00 / −0.00 | not measured | — |
   | fixed (400) | 2.10 | −0.11 / −0.04 | −0.01 / −0.00 | 0.01 / 0.11 (rerun, 300) | 0.00 / 0.01 |
   | centre term alone (400) | 2.04 (best 1.85) | ≤ 0.26 | ≤ 0.05 | not measured | — |
   | offsets over the length, bounded (300) | 1.97 | 0.01 / 0.09 | 0.00 / 0.03 | not measured | — |
   | per-px learned spread, all px (300) | 1.99 | 0.04 / 0.01 | 0.00 / 0.00 | −0.05 / −0.03 | −0.01 / −0.00 |
   | **centre read only on words wholly in the crop (200)** | **1.74** | **0.30 / 0.38** | 0.07 / 0.14 | **0.87 / 0.86** (step 50: 0.79 / 0.82) | **0.89 / 0.71** |

   The log length learns in every arm (corr 0.5-0.7 by step 300). The per-px spread did not specialise: its median
   sd is 1.69 on whole words and 1.76 on cut words, growing everywhere. As a fix on its own it is too slow at best.
4. **At the training's own crop (320 px, 400 draws):** 49.6% of core px with geometry are in words the crop border
   cuts, and they carry **85.3%** of the centre loss of predicting 0. Their |offset| q50/q90/q99 is 1.45 / 5.6 / 10.4
   heights, against 0.81 / 1.67 / 5.05 for words wholly inside.
5. **Terra's rectangles are centred on their ink** (`box_vs_ink.py`, p07): rectangle centre minus ink centre along the
   word has sd 0.03 heights for codes, 0.06 for block numbers, and 0.31 for street names (median 0.05). Not label
   noise.

Caveats: CPU, crop 160, one sheet, 300-400 steps. Correlations on a 16-crop held set move by ±0.1-0.2 between
logs. The visible-words arm's gain (0.8-0.9 against about 0.1) is far outside that.

**Proposed fix (not made; for the ruling):** in `data.crop_targets(words=True)`, `geo_known` is also 0 for a word
whose rectangle reaches outside the crop's page window [x0, x0 + src) × [y0, y0 + src).
- Its far end is not in the input, so its centre and length cannot be known from it: the same physical reason as a
  word reaching a held-out area. It is an exact geometric test, not a tuned cut.
- The axis stays read, since it is visible.
- Centre and size share `geo_known`, so the size term stops reading cut words' lengths too.
- Expected: the visible-words arm's result, centre corr 0.8-0.9 within 50-200 steps.
- At reading time `predict.word_outputs` stitches tiles at the network's reach, so a px sees its whole word unless
  the word is longer than the reach or crosses the page edge. Words wholly inside but longer than the reach remain
  (|offset| q99 5 heights): measure them after the change before adding anything such as a per-px spread.
- Tests: a cut word gets `geo_known` 0 and keeps its axis; a word wholly inside is unchanged; the default recipe and
  every guard are untouched.

**Fix (commit dadc5d1, on the controller's go-ahead):**
- **The rule:** `data.crop_targets(words=True)` sets `geo_known` to 1 only at the core px of a word that is neither
  held out nor cut. A word is cut when any of its rectangle's px, as `_paint` fills them (corners lo..hi
  inclusive), lies outside the crop's page window [x0, x0 + src) × [y0, y0 + src).
- **What a cut word keeps:** its geo 0-3 (the crop only hides them; it is not a held-out word, whose values stay
  zeroed), its log sizes scaled to the crop as before, and its axis, which the loss still reads.
- **Docstring:** the module's `geo_known` entry now says this and why.
- **Tests:** `test_wordread_heads.py` 31 passed, `test_wordread_doubt.py` 23 passed, test_build included: 93 passed
  together.
  - New: `test_a_word_the_crop_cuts_has_no_geometry_to_read_but_keeps_its_axis` (turns 0 and 1). It is exact at the
    window edge: a word ending on the window's last column is whole, one column further is cut. A word read upward
    is cut by the top edge. Cut words keep length 52 and their axis. A larger crop showing them whole reads them all
    (`geo_known` = core). At scale 2 the rule follows the same page px.
  - Updated: the held-word cache test uses a crop showing both words whole. The `word_tensors` test now expects
    `geo_known` only on the one word its first crop shows whole.
- **The doubt tests** (beyond "data.py and the heads tests": their crops cut words, so three failed on the rule):
  - the held-out geometry test now also checks that neither word's geometry is read in a crop that cuts both, and
    that in a crop showing both whole word 1's is read and word 0's (held) is not;
  - the weighted-mean test weighs centre and size by the px whose geometry is known;
  - the no-geometry gradient test checks the distance gradient at word 1's px whose geometry is known.

  The file was clean, with no other agent's edits in it.
- **Guards** all PASSED (157 passed): `GUARD_WEIGHTS`/`GUARD_LOSSES`, the default recipe, crop-target equality, no
  full-page rasters, both composite pinned batches. `heads="all"` never reaches this code.
