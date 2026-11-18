# Task 15 report: the learned box model

Status: DONE_WITH_CONCERNS. Commit 7d2f553 (3 new files, nothing else). $0, CPU only (one thread), no training run,
no stage touched.

## What was built

`pipeline/src/sfzmaps/solver/wordread/boxmodel.py` (new; no stage imports it), `pipeline/scripts/wordread/run_boxmodel.py`,
`pipeline/tests/test_wordread_boxmodel.py`.

- **Crops** (`box_crop`, `SheetInputs`, `load_sheet`): every box is cut by `textread.crops.box_cut`, so it gets the
  reader crops' own PAD_X/PAD_Y margin and no new constant. One channel per layer: ink (page < INK_BELOW, cut as float
  so the resize area-averages it exactly), the fold detectors' text chance (`text-reader/<sheet>/chance.png` / 255),
  and when given the word heads' `core` (h, w) and `tag` (k, h, w). Without heads: 2 channels; with heads: 3 + k.
  Each layer's window is made float before `crops.cut(origin=...)`, so the pages stay compact (bool/uint8).
- **Count target** (`ink_shares`, `poisson_binomial`, `box_targets`): a Terra word's share is its ink px (centres
  inside its `word_rect`) inside the box's trimmed rectangle, divided by its ink px. A word with no ink px falls back to
  its rectangle's area share; this is a definitional fallback for 0/0. The count is the exact Poisson-binomial of the
  shares, with no "mostly inside" cut.
- **Boundary target** (`boundary_target`): per crop column, the exact integral over the column of
  P(a held word ends at or before x) × P(a held word starts at or after x) × P(no held word covers x). Words are held
  independently with their shares; spans are the rectangles' extents along the box axis, mapped through `crops.cut`'s
  geometry (`layout`). With all shares 1 these are exactly the gap columns, and a column half in a gap counts 0.5.
- **Weights**: `weights=None` means 1. With Task 7's per-word w, an example weighs prod_i (1 - s_i (1 - w_i)) over its
  words; a joined pair weighs w_a·w_b.
- **Held out**: `box_examples` raises on a word touching the held quarter or an answered block (TOUCH flags from
  `solver-text-crops`), on missing flags, on `quarter=None`, on a box in the held quarter, and on a box whose crop
  footprint touches either (`touch`, `readercrops.HeldAreas` by default). `training_words` and `training_boxes` give
  what may go in. `training_boxes` also drops a box meeting a held word's rectangle, because its count would miss that
  word.
- **Joined neighbours** (`join`, `same_line_gaps`, `gap_distribution`, `draw_gaps`): two training words drawn
  uniformly (seeded) are cut as the reader cuts them (`crops.cut` at `rectify`'s centre and the floored length and
  height) and laid on one line at a gap drawn from the measured gap distribution:
  - the first word's crop runs up to its end, the gap is paper, the second word's crop runs from its start, and the
    outer margins are real;
  - on a negative gap, each overlapping px comes from the crop with more ink;
  - the result is squeezed to MAX_WIDTH as `cut` squeezes long crops;
  - it is turned a half turn with the measured share of real boxes' words that read against the box axis (0.355);
  - the targets are count 2 and the realised gap's columns;
  - how many: by default the measured expected number of real 2+-word boxes (sum of P(count ≥ 2)).
- **K**: measured as the largest count with any chance in the examples (`Examples.K`).
- **`BoxNet`**: the reader's MobileNetV3 blocks (`textread.model.InvertedResidual`/`ConvNormAct`), smaller (6 blocks,
  16-48 channels, 96 at the end). Height goes to 1 and width to W/4. Then 4 dilated, masked 1-D convs over the frames,
  a boundary logit per frame, and count logits over 0..K from the masked mean, max and sum of the frames.
  216k parameters.
- **`fit_boxmodel(examples, recipe)`**: runs inside `gpu.gpu_turn(True, device)`, seeded (torch seed and
  `default_rng([seed, 0])`). Loss: the weighted mean of soft count cross-entropy plus `boundary_weight` × the column
  BCE. AdamW, one-cycle schedule, clip 5, fixed steps. `RECIPE` holds the defaults; Task 8 screens them.
- **`box_logp(net, crops) -> (count_logp (n, K+1), [per crop (W, 2): log P(no boundary), log P(boundary)])`**: reads
  crops in batches of one padded width, so a crop's numbers are its own (tested). It runs in a GPU turn on mps.
  `box_crop(inputs, corners, length, height, axis)` cuts Task 9's candidate rectangles the same way.
- **`Examples`**: flat storage, with `save`/`load` (fixed-date npz plus parquet meta).
- **Run script**: `measure [--examples]` writes `counts.json`, `gaps.json` and `examples_<q>`. `fit --quarter q
  --seed s [--device mps]` is for Task 8 and was not run.

Two deviations from the reader's style, both measured:
- BatchNorm momentum is 0.1 (torch's default), not docTR's 0.01. With 0.01 a short fit's running statistics stay far
  from its batches: the CPU test fit read 1.0 in train mode and 0.5 (chance) in eval mode. With 0.1 it read 0.98.
- NNPACK is off around CPU work (`_cpu_convs`). NNPACK convolves each depthwise group separately; turning it off cut
  CPU step time about 3x (130 → 55 ms at batch 16).

## TDD evidence

1. I wrote 15 tests first. They failed with `ImportError: cannot import name 'boxmodel'`.
2. After the implementation: 11 passed and 4 failed. The failures were a synthetic-width bug in a test helper and a
   float32 tolerance; both were fixed in the tests.
3. The fit test then failed at accuracy 0.5. The cause was batch-norm momentum (see above); the fix went into the
   module, after which it read 1.0 at 100 steps × batch 8, about 4 s on an idle CPU.
4. While measuring real data, the gap definition changed twice (see Concerns). Each change came with its test: page
   next word, held pairs on one line, own reading frame, overlap ink union. A quarter turn now gives exactly 0, after a
   failing test caught cos 90° = 6e-17.
5. Final: `tests/test_wordread_boxmodel.py` 19 passed (the last added is a `save_boxnet`/`load_boxnet` round trip).
   With `tests/test_build.py`: 62 passed. Full suite (`uv run pytest -q -n 2`, run before that last test was added):
   1734 passed, 3 skipped, 3 xfailed, 0 failed.

Tests cover:
- the exact Poisson-binomial, including brute-force enumeration;
- the count from ink shares, ink not rectangle;
- doubt weights;
- the boundary equals 1 minus the ink fraction between two solid words, through `cv2`'s INTER_AREA, independent of my
  column formula;
- the boundary scaled by the held chance;
- the boundary integral on a 3-word case;
- the channels with heads;
- the raises (held word, block word, held box, a crop touching a block, no quarter, missing flags);
- `training_boxes` and `training_words`;
- draws matching a weighted distribution, within 0.02;
- `same_line_gaps` (next word, a soft partial-line word in between, a word across the line);
- `held_pair_gaps` (consecutive weights, stacked words count nothing, the gap taken along the words' own direction for
  a box at 30°);
- joined examples (drawn gaps, count 2, boundary sum equal to the realised gap, the gap equal to the drawn gap to half
  a px, paper in the gap, different words, seed-deterministic);
- overlapping joins keeping both words' ink;
- the CPU fit separating 1 word from 2 (≥ 0.9; chance 0.5) with boundary chance higher in the gap;
- `box_logp` batch independence;
- `Examples` save/load.

## Example counts per quarter (model holding q out; today's boxes, 2 channels)

| held out | training words | training boxes | joined | examples | K | crop columns |
|---|---|---|---|---|---|---|
| nw | 5,806 | 5,063 | 128 | 5,191 | 6 | 449,317 |
| ne | 5,417 | 4,742 | 106 | 4,848 | 6 | 417,172 |
| sw | 5,433 | 4,633 | 164 | 4,797 | 6 | 421,446 |
| se | 4,874 | 4,336 | 135 | 4,471 | 6 | 385,150 |

The training boxes' expected counts (0 / 1 / 2 / 3+):
- nw: 1,534.6 / 3,400.2 / 100.0 / 28.2
- ne: 1,418.5 / 3,217.8 / 85.2 / 20.6
- sw: 1,395.9 / 3,072.8 / 133.0 / 31.4
- se: 1,351.5 / 2,849.2 / 115.3 / 19.9

Weights are all 1 (no Task 7 weights yet). Building each fold's examples takes 14-17 s on the CPU. They are saved in
`pipeline/data/solver/experiments/wordread/boxmodel/examples_<q>.npz/.parquet`, ready for Task 8.

## Today's boxes against Terra: how many words they hold

These are all 8,794 of today's boxes (`data/solver/boxes`) against all 10,036 Terra words. "Expected" is the sum of
each box's Poisson-binomial distribution, which is exact. "Likeliest" counts each box at its most likely count and is
shown for display only.

| | boxes | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|---|
| all, expected | 8,794 | 2,868.7 | 5,363.3 | 381.4 | 180.6 |
| all, likeliest | | 2,974 | 5,257 | 380 | 183 |
| nw | 1,811 | 616.9 | 1,081.0 | 95.2 | 17.9 |
| ne | 2,452 | 790.3 | 1,362.7 | 176.3 | 122.8 |
| sw | 2,026 | 671.5 | 1,305.4 | 38.0 | 11.1 |
| se | 2,505 | 790.0 | 1,614.2 | 72.0 | 28.8 |
| Mission sheet p07, expected | 995 | 303.8 | 576.6 | 71.0 | 43.6 |
| Mission sheet p07, likeliest | | 302 | 577 | 71 | 45 |

The largest count with any chance is 25: one long merged box. It is in no fold's training set, since every fold's
K is 6.

## Gap distribution (gaps.json; in mean word heights, weighted quantiles)

Drawn from: `same_line_gaps` on each fold's training words. Each gap runs from a word's end to the start of the next
word on its line of text. Each pair is weighted by the chance that the other word is the next one on that line.

| fold | pairs | weight | q01 | q05 | q25 | q50 | q75 | q95 | q99 |
|---|---|---|---|---|---|---|---|---|---|
| nw | 19,128 | 4,230.9 | -4.28 | 0.12 | 3.46 | 10.22 | 27.47 | 84.46 | 143.77 |
| ne | 19,421 | 3,978.5 | -4.34 | 0.00 | 3.45 | 9.03 | 24.43 | 83.16 | 146.69 |
| sw | 16,637 | 3,874.9 | -3.93 | 0.21 | 3.62 | 9.60 | 26.68 | 84.26 | 142.14 |
| se | 13,600 | 3,383.6 | -3.59 | 0.62 | 3.44 | 7.80 | 21.95 | 74.70 | 134.43 |

For comparison, reported in counts.json and not drawn from: gaps between consecutive words on one line of text that
one of our training boxes holds together (`held_pair_gaps`).
- nw: 158 pairs, weight 48.8. Quantiles: q05 -3.40, q25 -1.78, q50 -0.49, q75 0.09, q95 2.96.
- ne: 146 pairs, weight 48.7. Quantiles: q05 -3.40, q25 -1.90, q50 -0.67, q75 -0.06, q95 1.15.
- sw: 179 pairs, weight 66.2. Quantiles: q05 -3.23, q25 -1.52, q50 -0.43, q75 0.08, q95 2.96.
- se: 129 pairs, weight 49.4. Quantiles: q05 -2.71, q25 -1.08, q50 -0.36, q75 0.08, q95 3.69.

## Concerns

1. **Which gaps the joined neighbours should take is the controller's call; I built the literal one.**
   - **Built:** the "real same-line gaps" as measured. On the page, the next word on a line is mostly another label
     far along the street: median 8-10 heights, 95th percentile 75-85. So most joined crops show two words with wide
     paper between them. Those over about 40 heights get squeezed to MAX_WIDTH, and their letters squash (seen in a
     sample sheet of crops).
   - **Tried and not built:** the hard "nearest word ahead in the strip" (median 4.3-6.2, 95th percentile 40-50).
     Gaussian mixtures in asinh(gap) with 1-5 components, chosen by BIC: none isolates a 0-1-height phrase population;
     the low tail is a continuum. And gaps along the box's own axis.
   - **What real merged boxes look like:** 2-D clusters, mostly a code with its block number tucked beside or below
     it, not words side by side. Pairs held together and on one line are mostly overlapping (median -0.4 to -0.7) and
     few: weight 49-66 per fold.
   - **The choice is open:** `box_examples(gaps=...)` takes either distribution, so Task 8 can screen. Neither looks
     much like a real merged box.
2. **Few real positives.** About 85-133 expected 2-word and 20-31 3+-word training boxes per fold, against 1,400+
   zero-word and about 3,000 one-word boxes. The joined examples add 106-164. The count head may underfit merges.
   Class weighting or more joined examples are recipe options for Task 8; I set neither.
3. **Count 0 means two different things.** About 30% of boxes expect 0 words. That includes junk and text Terra missed
   (its misses become count-0 label noise). It also includes fragments: a single letter of a letter-spaced street name
   holds a small share of the word, so it gets count ≈ 0. That follows exactly from the specified target. For Task 9's
   trims and splits it means "half a word" reads as P(1) ≈ share.
4. **Loose Terra rectangles leak into the targets.** Boundary spans and shares use `word_rect`, while the joined crops
   use `rectify`'s floored geometry; the two differ for slanted words. Overlapping Terra rectangles (Terra double reads
   such as 'YORK ST' and 'ST.') inflate counts. Task 7's weights should temper both.
5. **Not done here:**
   - No augmentation in `fit_boxmodel`. The reader jitters width; doing that here would need the boundary targets
     resampled.
   - mps determinism is unchecked (CPU only here). Task 8 should repeat two short runs, as for Task 5.
   - K differs by run (measured), so Task 9 must read `count_logp` over the model's own 0..K.
6. The CPU fit test runs about 4 s idle and 15 s under a load of 125, since other agents were running at the time.

## Files

- `pipeline/src/sfzmaps/solver/wordread/boxmodel.py`
- `pipeline/scripts/wordread/run_boxmodel.py`
- `pipeline/tests/test_wordread_boxmodel.py`
- Outputs (not committed): `pipeline/data/solver/experiments/wordread/boxmodel/{gaps.json, counts.json, examples_<q>.npz/.parquet}`

## Fix round 1 (the controller's ruling on concerns 1 and 2): commit 600d723; full suite 1744 passed, 3 skipped, 3 xfailed

The joined examples are now the merges our boxes really make, drawn from a fitted merge model. The same-line gaps
remain only as a record (`gaps.json`, `same_line_gaps`). `draw_gaps`, `join`, `_word_crop`, `held_pair_gaps` and
`pair_gaps` are gone, along with the `gaps` and `turned` arguments.

**What changed (`boxmodel.py`):**

- **`pair_geometry(terra)`** takes every pair of training words on one sheet: 1.3–2.0M pairs per fold, with no distance
  cut. The geometry is taken in the first word's reading frame. It records the offsets `along` and `across`: the space
  between the rectangles in their mean height, negative by the overlap. Nested extents are handled: the code uses
  max(lo) − min(hi).
- **The merge model's features** (MERGE_FEATURES), in the same frame:
  - `along_gap` and `across_gap`: those spaces where positive;
  - `along_overlap` and `across_overlap`: the overlap as a share of the shorter extent;
  - `along_overlap_h` and `across_overlap_h`: the overlap in heights;
  - `angle`: the angle between readings, divided by 90;
  - `height`: |log| of the height ratio.
- **Why this encoding.** My first fit used the controller's four quantities as signed linear terms (along, across,
  angle, height). That is mis-specified: overlap entered linearly, so a pair overlapping 25 heights along got high P
  even 5 heights away across. As a result 24–26% of the P mass lay beyond 5 heights and 7.5–8.7% beyond 10. The
  realised joins had along q05 between −19 and −27, against today's −3.4 to −3.6, and K rose to 9–17. Splitting each
  offset into its gap and its overlap makes the chance fall with distance either way, and lets the fit decide how
  overlap counts. The NLL on fold ne fell from 887 to 686.
- **`merge_labels`**: a pair's label is the chance that one of the fold's training boxes holds both words,
  1 − Π_b (1 − s_a s_b), from the ink shares.
- **`fit_merge`**: weighted soft-label logistic regression, maximum likelihood with L-BFGS on standardised features; the
  ridge is 1e-6 and numerical only. Pair weights are w_a·w_b from Task 7 (1 for now).
- **`merge_chance`**: the fitted chance per pair.
- **`union_box`**: the box `wordboxes.group` would make of the two words' ink: the chance-weighted major axis and the
  trimmed extents.
- **`box_examples`**:
  - It fits the merge model on its own boxes and words (`merge=None`), or takes one passed in.
  - It draws real pairs with chance ∝ P(merge), seeded, and cuts each pair's union box from the real page at the
    pair's real place: the empirical joint.
  - The targets are exact (`box_targets`). Both words have share 1, so the count is 2, or more where another word's ink
    is inside the box.
  - The boundary columns are where the two extents separate along the box's axis; a vertical stack has none.
  - A pair is drawn again when its box would touch the held quarter or an answered block, meet a held word
    (`held_words`; the script passes the held-out words), or include a word with no ink. A guard (JOIN_TRIES) stops it
    after 100 draws per wanted example.
  - `Examples.info` carries the fit, the tail mass and the geometry of both today's merges and the joins; it is saved
    in the npz.
- **Docstring:** count 0 also covers text Terra missed, which Task 7's weights can't reach (concern 3).

**Tests (TDD):**
- New tests failed first: KeyError on the new features, and a broadcast error in my own count comparison, which I
  fixed in the test.
- The new tests:
  - pair geometry: side-by-side and nested-stack offsets and features, and angle;
  - merge labels as 1 − Π(1 − s_a s_b), checked exactly;
  - `fit_merge` recovers a known 8-coefficient logistic from exact soft labels to within 0.01;
  - `union_box` gives axis 0 and the exact extents, and None for a word with no ink;
  - joined pairs are drawn ∝ P(merge): 2,000 draws match within 0.04 per pair;
  - a side-by-side pair gets boundary = 1 − ink between the words and count exactly 2; a code over its block number
    gets no boundary column and count 2;
  - drawing is seed-deterministic;
  - joined boxes never touch a held area or a held word;
  - `box_examples` fits the model on its own boxes (one merged pair, so one joined example by default).
- 21 passed in the file; 64 with `test_build`.

**Fitted coefficients** (per unit of each feature; intercept first):

| fold | pairs | merged (Σ labels) | intercept | along_gap | across_gap | along_overlap | across_overlap | along_overlap_h | across_overlap_h | angle | height | NLL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nw | 1,958,989 | 221.6 | −3.007 | −0.574 | −1.008 | 1.676 | 1.372 | −0.270 | −1.234 | −0.480 | 0.248 | 914.8 |
| ne | 1,777,514 | 171.6 | −3.150 | −0.715 | −1.141 | 1.818 | 1.467 | −0.271 | −0.999 | −1.003 | 0.319 | 686.2 |
| sw | 1,621,897 | 261.3 | −2.582 | −0.619 | −1.132 | 1.700 | 1.550 | −0.224 | −1.307 | −0.669 | 0.138 | 950.9 |
| se | 1,336,743 | 191.1 | −2.516 | −0.697 | −1.322 | 2.428 | 2.793 | −0.648 | −2.813 | −0.988 | 0.012 | 669.7 |

All four fits converged and the signs agree across folds. The chance falls about 0.6–0.7 logits per height of gap
along the line and 1.0–1.3 logits per height across it. Overlap raises the chance by its share of the shorter word and
lowers it with its length.

**Tail mass** (share of the total P(merge) beyond a separation, max(along, across), in heights; report only). Every
pair is a candidate, so no bound was needed.

| fold | > 1 | > 2 | > 5 | > 10 |
|---|---|---|---|---|
| nw | 0.398 | 0.182 | 0.0227 | 0.0010 |
| ne | 0.337 | 0.133 | 0.0094 | 0.0002 |
| sw | 0.359 | 0.162 | 0.0159 | 0.0006 |
| se | 0.316 | 0.137 | 0.0109 | 0.0003 |

**Today's merge geometry against the joined pairs'.** Offsets in mean word heights at q05 / q25 / q50 / q75 / q95.
Today's are weighted by the merge labels; the realised are the drawn pairs.

| fold | offset | today | realised |
|---|---|---|---|
| nw | along | −3.20 / −1.36 / −0.81 / 0.05 / 4.15 | −3.10 / −1.56 / −0.89 / 0.53 / 3.28 |
| nw | across | −1.00 / −0.51 / −0.04 / 0.37 / 2.92 | −0.98 / −0.63 / 0.15 / 0.83 / 2.23 |
| ne | along | −3.23 / −1.44 / −1.00 / −0.11 / 3.58 | −3.00 / −1.33 / −0.88 / 0.97 / 3.58 |
| ne | across | −1.07 / −0.58 / −0.10 / 0.26 / 2.75 | −0.96 / −0.87 / −0.18 / 0.50 / 1.57 |
| sw | along | −2.95 / −1.35 / −0.81 / 0.00 / 4.17 | −2.63 / −1.35 / −0.79 / 0.47 / 3.38 |
| sw | across | −1.00 / −0.54 / −0.06 / 0.26 / 2.75 | −1.05 / −0.50 / −0.06 / 0.55 / 1.89 |
| se | along | −2.59 / −1.28 / −0.80 / −0.11 / 4.15 | −2.69 / −1.26 / −0.73 / 0.21 / 3.38 |
| se | across | −0.96 / −0.51 / −0.05 / 0.21 / 2.61 | −0.96 / −0.49 / 0.04 / 0.48 / 1.47 |

The medians and lower quantiles agree within about 0.1–0.2 heights. The realised q75 sits a little further apart (by
0.3–1.1 heights along), and the realised across q95 is narrower (1.5–2.2 against 2.6–2.9). Angle and height
quantiles, and q01 and q99, are in counts.json. The realised quantiles rest on only 106–164 draws per fold, so they are
noisy.

**New joined-example counts per fold** (the default is unchanged: the expected number of real 2+-word boxes):

| fold | boxes | joined | examples | pairs drawn (incl. redrawn) | K | joined boxes' expected extra words (Σ P(count ≥ 3)) |
|---|---|---|---|---|---|---|
| nw | 5,063 | 128 | 5,191 | 164 | 10 | 42.1 |
| ne | 4,742 | 106 | 4,848 | 121 | 9 | 31.6 |
| sw | 4,633 | 164 | 4,797 | 196 | 12 | 40.4 |
| se | 4,336 | 135 | 4,471 | 170 | 8 | 33.5 |

**Remaining concerns:**
- K rose from 6 to 8–12. The joined union boxes have exact counts, and 25–33% of them hold a third word's ink. Among
  today's real boxes, 3+-word boxes are 17–22% of the 2+-word ones.
- Positives are still few: 106–164 joined plus about 105–165 real 2+-word boxes per fold. Task 8 can raise `joined`.
- The realised across spread is narrower at the top than today's merges.

## Fix round 2 (the review's rulings)

1. **The count head is ordinal.**
   - **Form:** one latent score f(x) read through K + 1 increasing thresholds: θ_1 plus softplus increments, starting
     one logit apart, all learned. P(count ≥ j) = σ(f − θ_j). This is the controller's σ(θ_j − f) with f's sign flipped
     and θ increasing.
   - **Output:** `BoxNet.count_logp` and `box_logp` return (n, K + 2): log P(count = 0..K), then log P(count > K) as
     the last column, which is modelled and not capped. P(count = j) is computed stably as
     σ(a_j)σ(−a_{j+1})(1 − e^{−(θ_{j+1}−θ_j)}).
   - **Loss:** cross-entropy of the exact Poisson-binomial target over 0..K, with no mass above K.
   - **Different K:** `collapse_counts(count_logp, k)` reads a model at any k ≤ its K, summing the columns above k
     exactly. Task 9 compares models at the smallest K among them; reading above a model's own K raises.
   - **Tests:**
     - P(count ≥ j) of the output equals σ(f − θ_j), with the last column included and the thresholds increasing;
     - collapsing gives the same shape for K = 5 and K = 2, sums to 1, keeps columns 0..k, sums the rest exactly,
       leaves a model unchanged at its own K, and raises above it.
   - **Learning rate:** a short fit learns the ordinal head more slowly than the softmax head. At 100 steps:
     max_lr 3e-3 gives accuracy 0.0 (mass leans the right way but argmax lands on the outer columns), 1e-2 gives 0.03,
     3e-2 gives 1.0. The CPU test fit now uses 3e-2. Task 8 should screen max_lr and steps (RECIPE keeps 2e-3 ×
     6,000 steps).
2. **`box_examples(held_words=...)` is required.** It raises when None (pass an empty frame when there are none). It
   now also raises when a real box meets a held word's rectangle, as well as refusing joined boxes that do. Tested.
3. **Squeezed crops** (`squeezed(length, height)`, `meta["squeezed"]`, counts.json `examples.squeezed`): 0 real and 0
   joined in every fold (nw, ne, sw, se). No training box or joined box is cut wider than MAX_WIDTH.
4. **Merge-model form, chosen held out.**
   - **Method:** both forms were fitted on each fold's training boxes and words, then scored by NLL on the held-out
     quarter's pairs: the quarter's words outside the answered blocks, labelled by its boxes whose crops touch no
     answered block and meet no answered-block word. `merge_data`, `merge_nll`; the script's `merge_forms`.

     | fold | held-out pairs | held-out merged (Σ labels) | NLL, 4 offsets | NLL, 8 features | per merged, 4 / 8 |
     |---|---|---|---|---|---|
     | nw | 207,886 | 59.2 | 259.2 | 168.2 | 4.38 / 2.84 |
     | ne | 391,126 | 106.4 | 486.4 | 393.0 | 4.57 / 3.69 |
     | sw | 545,551 | 23.5 | 174.9 | 147.4 | 7.46 / 6.28 |
     | se | 824,102 | 92.7 | 461.6 | 434.5 | 4.98 / 4.69 |
     | total | | | 1,382.2 | 1,143.0 | |

   - **Result:** the 8-feature form (gaps, overlap shares, overlap lengths, angle, height ratio) wins in every fold and
     overall, so it is kept. The coefficients, tail mass and joined counts are unchanged from round 1.
   - **Record:** counts.json has `merge_form` per fold and overall (`kept: split`); `OFFSET_FEATURES` stays as the
     alternative scored.
5. **Removed** the `net.k == 2 and net.channels == 2` assertion.

**Tests:** 24 passed in my file; 67 with `test_build`; full suite (`-n 2`) 1749 passed, 3 skipped, 3 xfailed, 0 failed. Commit 36af02a.

## Training (the box model on the word heads, four folds, seed 0)

**Inputs and stacking.**
- **Channels:** 9:
  - ink;
  - the text chance;
  - the core chance (sigmoid of the core logit);
  - the axis as cos 2d and sin 2d, where d is each px's word axis in degrees relative to the crop's own axis. The cos
    and sin of 2θ are cut, then turned by twice the crop's angle, so the encoding is sign-free and turns with the crop;
  - the four tags.

  The chance, core, axis and tags come from Task 8's arm (a) runs `detector/runs/a-<q>-s0/words/<sheet>.npz`, decoded by
  `model.word_geometry` (Task 5's sign-free layout).
- **Stacking:** `run_boxmodel.py stack` builds each 2000 page once. Every px's inputs come from the detector run that
  held out that px's quarter, a-p-s0 for a px in quarter p, by its map point (`reader_stage.page_quarters`, the same
  rule as the shipped text-reader chance). So every training example in quarter p sees a-p-s0's outputs, and the
  held-out quarter's boxes see a-q-s0's: every input is a held-out input. A crop near a quarter line mixes two
  detectors, each held out on its own px.
- **Storage:** `boxmodel/stacked/<sheet>/{chance,core,axis,tag,quarter}.npy`, memory-mapped when read; stacked.json
  records the px per quarter per sheet. Building them took 2.5 min on the CPU.
- **Axis check:** the stacked axis matches Terra's reading angle in the same convention. On p07, the core-weighted
  axis on each word's ink is within a median 1.36° of Terra's angle modulo 180. On the 71 slanted words the median
  error is 6.4°, against 52° with the sign flipped.

**Examples:** the training quarters' boxes (today's, against Terra) plus the merge-drawn joined pairs (fix round 1's
model), with `held_words` = every word that touches the quarter or an answered block.

**Training:** `fit_boxmodel`, the RECIPE as committed (6,000 steps, batch 32, max_lr 2e-3, one-cycle with 15% warmup,
AdamW wd 1e-4, clip 5, boundary weight 1), mps, each fit in its own exclusive GPU turn. The fits queued behind Task 8's
detector runs through the lock.

**Scoring** (`run_boxmodel.py evaluate`, heldout.json and heldout.parquet per run):
- **Boxes:** the held-out quarter's boxes whose crop touches no answered block and that meet no answered-block word.
- **Targets:** the exact ink-share targets against every Terra word off the answered blocks.
- **Reading:** the model is read at 0, 1, 2, 3+ (`collapse_counts`).
- **Report-only thresholds:** the 0.5 boundary call, and P(count ≥ 2) > P(count = 1).

**The runs** (`boxmodel/runs/<q>-s0/`: model.pt, fit.json, prior.json, examples.parquet, heldout.json,
heldout.parquet; logs in `boxmodel/logs/`):

| fold | examples (boxes + joined) | K | train s (mps) | count CE at 250 / 1k / 2k / 4k / 6k steps | boundary BCE at 6k | θ1 / θ2 / θ3 |
|---|---|---|---|---|---|---|
| nw | 5,191 (5,063 + 128) | 12 | 366 | 1.233 / 0.899 / 0.592 / 0.378 / 0.337 | 0.0017 | −0.11 / 4.41 / 7.63 |
| ne | 4,848 (4,742 + 106) | 9 | 341 | 1.234 / 0.897 / 0.607 / 0.397 / 0.358 | 0.0011 | −0.13 / 4.37 / 7.59 |
| sw | 4,797 (4,633 + 164) | 12 | 315 | 1.246 / 0.902 / 0.598 / 0.379 / 0.340 | 0.0021 | −0.13 / 4.36 / 7.93 |
| se | 4,471 (4,336 + 135) | 8 | 326 | 1.222 / 0.883 / 0.574 / 0.357 / 0.322 | 0.0014 | −0.12 / 4.42 / 8.02 |

**The curve:** the ordinal head learns steadily, without stalls.
- **Count cross-entropy:** it falls from about 1.23 to 0.32–0.36. It drops 0.035–0.041 over the last 2,000 steps and
  0.002–0.011 over the last 1,000, while the one-cycle learning rate anneals to 0. It flattens only with the schedule,
  so it is not clearly converged.
- **Thresholds:** they move apart for the whole run. θ2 − θ1 goes from 1 at the start to about 4.5 logits.

  The recipe was not changed: any change chosen now would be chosen on these results. More steps or a higher max_lr
  are for Task 8 to screen.

**Count calibration on the held-out quarter** (boxes off the answered blocks, against their ink-share targets; at 0 /
1 / 2 / 3+):

| fold | boxes | mean predicted | mean target | count CE (prior's) | P(≥2) top decile: predicted / target | 9th decile |
|---|---|---|---|---|---|---|
| nw | 1,197 | 0.290 / 0.611 / 0.089 / 0.010 | 0.306 / 0.652 / 0.038 / 0.004 | 0.515 (0.792) | 0.491 / 0.328 | 0.142 / 0.045 |
| ne | 1,520 | 0.318 / 0.527 / 0.135 / 0.020 | 0.317 / 0.636 / 0.038 / 0.009 | 0.575 (0.819) | 0.761 / 0.353 | 0.366 / 0.096 |
| sw | 1,643 | 0.306 / 0.615 / 0.072 / 0.007 | 0.312 / 0.679 / 0.008 / 0.001 | 0.479 (0.717) | 0.356 / 0.085 | 0.100 / 0.007 |
| se | 1,927 | 0.314 / 0.606 / 0.075 / 0.005 | 0.285 / 0.692 / 0.016 / 0.007 | 0.507 (0.732) | 0.364 / 0.195 | 0.121 / 0.000 |

- **Better than the prior:** the cross-entropy beats the training prior's in every fold, by 0.23–0.28 nats a box.
- **Ranking:** the predicted P(≥2) ranks boxes the right way. The targets' 2+ mass sits in the top deciles, and the
  bottom eight deciles all have targets below 0.02.
- **Calibration:** P(count = 2) is too high in every fold, by 2–9× on average. Every decile of P(≥2) lies above its
  target. Mass the targets put on "1" goes to "2".
- **0 words:** P(count = 0) is calibrated (0.29–0.32 against 0.29–0.32).
- **For Task 9:** these are log-probabilities for a fitted choice, which can correct a scale. Read raw, P(≥2) > 0.5 is
  not a calibrated call.

**Boundary columns on the held-out quarter** (soft targets; the 0.5 call is report only):

| fold | columns | boundary share | BCE (base rate's) | Brier | recall / precision at 0.5 | in boxes likelier 2+: BCE (base) | recall / precision |
|---|---|---|---|---|---|---|---|
| nw | 82,162 | 0.00014 | 0.00241 (0.00138) | 0.00048 | 0.00 / — | 0.0194 (0.0227) | 0.00 / — |
| ne | 106,211 | 0.00112 | 0.01190 (0.00872) | 0.00304 | 0.46 / 0.12 | 0.0869 (0.1212) | 0.47 / 0.64 |
| sw | 130,973 | 0.00008 | 0.00091 (0.00080) | 0.00017 | 0.00 / — | 0.0645 (0.0600) | 0.00 / — |
| se | 147,841 | 0.00041 | 0.00171 (0.00360) | 0.00026 | 0.44 / 0.56 | 0.0551 (0.0956) | 0.45 / 0.74 |

The boundary head is weak:
- Boundary columns are rare: 0.01–0.11% of the held-out quarters' columns.
- Over all columns, it beats the base rate's BCE only in SE. Inside boxes likelier to hold two or more words, it beats
  it in NE, SE and NW.
- In NE and SE it finds about 45% of the boundary mass at 0.5, with precision 0.56–0.74 inside merged boxes.
- In NW and SW, where held-out merges are few (51 and 14 boxes), it never calls a boundary at 0.5.
- Its training BCE (0.001–0.002) is far below its held-out BCE in the merged boxes, so it fits the training quarters'
  few boundaries and doesn't carry them over.

**P(count ≥ 2) > P(count = 1)** (report only, not a rule):

| fold | held-out boxes whose target says 2+ | whose target says fewer | Mission p07 merged_with_word boxes in the quarter |
|---|---|---|---|
| nw | 31 / 51 | 19 / 1,146 | 0 / 1 |
| ne | 59 / 71 | 115 / 1,449 | 86 / 98 |
| sw | 13 / 14 | 28 / 1,629 | 1 / 1 |
| se | 33 / 44 | 14 / 1,883 | 21 / 23 |
| total | 136 / 180 (76%) | 176 / 6,107 (2.9%) | **108 / 123 (88%)** |

On the Mission sheet (p07), the 293 Terra words missed as merged_with_word (review/baseline/missed.json) sit in 123
distinct boxes (their `best_box`). Each box is scored by the model that held out its quarter, on that quarter's
stacked inputs; answered blocks are included, since no model saw them. For 108 of the 123, P(count ≥ 2) exceeds
P(count = 1). Their own ink-share targets put a mean 0.90–1.0 on two or more words.

**Concerns:**
- P(2) is over-predicted, as above.
- The boundary head doesn't generalise in NW and SW.
- The count curve is still falling when the learning rate anneals. Task 8 should screen steps and max_lr, and Task 9
  should rely on a fitted choice rather than raw thresholds.
- A single seed per fold; no verdict.

**Commit and tests:** b27aa1c (`boxmodel.py`, `run_boxmodel.py`, `test_wordread_boxmodel.py`).
- My file: 26 passed; with `test_build`, 69 passed.
- Full suite (`-n 2`): 1865 passed, 7 skipped, 3 xfailed, 1 failed. The failure is `test_wordread_words.py::test_a_dimension_the_net_knows_sharpens_the_boxes`, in another agent's uncommitted `words.py` work, and is not mine.
- No stage was rebuilt; $0.
