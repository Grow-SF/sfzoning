# Our text pipeline, traced step by step (2026-11-17)

This note traces how the solver finds and reads text on the scans, step by step, so it can be set beside published
text detectors. It was a read-only trace of branch `solver` at `a2b1dc7`.

- File:line references are under `pipeline/src/sfzmaps/solver/` unless a path says otherwise.
- Numbers come from the outputs as built today (`pipeline/data/solver/...`) and from
  `2026-11-17-text-vs-line-results.md`.
- **(inferred)** marks something I worked out rather than read in code or data.
- I made no LLM call, no GPU run and no training run. I counted parameters by building the networks on the CPU.

Names used below:
- **Terra** and **Sol** are two OpenAI vision models whose answers are saved in `pipeline/saved-reads/answers/`.
  Terra boxed every piece of text on 179 sheets, shown in 768-px tiles (`text-on-every-sheet`, 14,829 answers). Sol
  put a point on each zoning label.
- **r328**: today's held-out detectors (stage `solver-text-reader`), named for their 328-px unknown discs.
  Everything downstream reads them.
- **old-ne / old-sw**: the earlier quarter models (`Recipe()`, 30-px discs).
- **control**: tonight's label rule `lines-unknown` on `DEFAULT_RECIPE`, measured but not adopted.

## At a glance

```
page.png (grey scan, before any warp) --(< 128)--> 1-bit ink: the detector's only input
  detector targets: Terra's word rectangles, filled  ->  per-px text / kind / angle
  4 U-Nets, one per held-out quarter (solver-text-reader)  ->  chance, kind, direction at every page px
    |-- walls: the chance laid on the 4-ft grid; stroke evidence weighed by (1 - chance)     [read by the solve]
    |-- word boxes: chance >= ~0.995 near ink, connected components, second-moment axis    [solver-boxes]
          |-- crop along the axis, 32 px tall -> CTC reader, read both ways, + 0/180 classifier
          |-- posterior over 95 printed code forms, "other", "none"                       [solver-text-read: measured only]
The labels the solve reads are still five saved LLM reads per print (prints.py, trust.py).
```

## 1. Inputs

- **The page scan.** `data/work/<series>/<sheet>/page.png`, 8-bit grey, as scanned, before any warp
  (`textnet/data.py:237-244`). A 2000 sheet is 3300 x 5100 px at 3.98 ft/px (`textnet/linerules.py:20`); a 1960
  sheet is 5257 x 6600 px.
- **The ink.** A px darker than `grid.INK_BELOW = 128` is ink (`grid.py:15`). It is packed to bits
  (`data.py:301`) and unpacked by `page_ink` (`data.py:352-354`).
  - This 1-bit mask is the detector's only input: no grey levels, no colour (`data.py:3`).
  - The frame, the merged page and the walls' grid use the same threshold (`textgrid.py:67`).
- **The warp never touches the detector's px.** A sheet's corrected placement (`data/warp/<series>/<sheet>/
  transform.json`) only maps page px to city feet for held-out areas (sampled every 32 px, bilinear between;
  `data.py:206-226`), gives the scale for the line rules, lays the chance on the 4-ft grid (`textgrid.py:72`) and
  places boxes on the map (`wordboxes.py:237`). The all-sheets stage reads none (`textnet/stage.py:41`).
- **Terra's word boxes** (`textnet/boxes.py`). Each item has its text, a kind (zoning code, block number, street
  name, other), an axis-aligned box in tile px, a direction and an angle (`boxes.py:25-33`).
  - `terra_boxes` (`boxes.py:54-104`) moves each box into page px and sorts the boxes by distance from the tile edge,
    farthest first. It then drops:
    - a box overlapping a kept one at IoU > 0.3 (the same text, read from another tile);
    - a word lying more than 70% inside a longer label that holds its text;
    - boxes outside the window that was asked about.
  - `word_rect` (`boxes.py:146-164`) turns the axis box and angle into a rotated rectangle. Near 45 degrees (about
    38-52) that solve is ill-conditioned, so the height is set to 14 px.
- **Sol's label points**: x, y per label in `district_labels.json` (`saved.py:48-54`).
- **The other passes' boxes**: `saved-reads/answers/experiments-2000/text-positions`, on the 2000 sheets only
  (`data.py:56`, `:265-268`). 274 of their 11,094 boxes touch no Terra word: "Terra's miss rate", 2.5%
  (`textread/crops.py:228-244`; `text-crops/summary.json`).
- **The maps.**
  - Training: 87 sheets of 2000, 1980-1990, 1990-1996 (one drawing, scanned three times), 1960, 1964 and 1968
    (`paths.py:20`).
  - Test: 92 sheets of 1980, 1972 and 1975, never trained on (`paths.py:21`).

## 2. Detector training data (`textnet/data.py`)

- **`SheetData`** (`data.py:167-190`) keeps each sheet compact: packed ink bits; Terra's rectangles (int32 (n, 4,
  2)) with kinds and angles; the other passes' rectangles; Sol's points; placement samples, scale, held-out areas,
  unknown radius and line bits. Targets are drawn for each crop alone, never for a whole page.
- **Painting** (`_paint`, `data.py:379-403`). Rotated rectangles are filled with `cv2.fillPoly` (later over earlier),
  discs with `cv2.circle`, on a canvas that holds every shape reaching the window, so a crop's targets are the page's,
  cropped. `_words` (`:451-453`) gives each px 1 + the index of the last word drawn on it, or 0.
- **The targets** (`crop_targets`, `data.py:475-501`):
  - **text**: 1 on every px of a word rectangle, ink or paper, letters or the lines through them; 0 elsewhere.
  - **kind**: the word's kind on text px; 255 off text.
  - **angle**: the word's reading angle (degrees counter-clockwise from left-to-right, turned with the crop) on text
    px; 0 off text.
  - **known**: not unknown and not held out. The loss reads only known px.
- **Unknown discs** (`_unknown`, `data.py:456-460`). A px within `point_radius` of a Sol point, or in another pass's
  box, and in no Terra word, is taught neither way. They exist to cover text Terra missed. The radius:
  - 30 px in `solver-text` (`READ_POINT_RADIUS`, `data.py:57`);
  - 328 px in `solver-text-reader`: the longest Terra code word on the training maps (`measure_radius`,
    `reader_stage.py:96-122`). One mis-boxed "SH" on 1968 p18 set it. The median code word is 85 px and the 99th
    percentile 169 px (`text-reader/held.json`).
- **Held out** (`_held`, `data.py:446-448`; the areas, `:81-165`). Held px carry no loss and no crop centres on one,
  though a crop may overlap them. Each px is placed through its map point.
  - The recipe trials hold out `VALIDATION_AREA`, the Mission / Noe Valley / Glen Park strip, about 10% of the area
    (`data.py:60-72`).
  - The reader's detectors hold out quarter q and the 715 answered blocks.
- **Tonight's rule, `labels="lines-unknown"`** (`data.py:17-19`, `:490-491`; `textnet/zones.py`).
  - **The in-box rule.** Px inside a word that are line px become unknown. Line px are `linerules.line_px`
    (`linerules.py:39-51`), grown by 1 px (`zones.py:45`, `:110-114`): ink on a straight run of at least 200 ft
    (about 50 px) in any of 32 directions, found by an opening with a 1-px line.
  - **Zones** (`zones.py:53-93`, `:160-172`). A Sol point that a Terra word holds, or lies within half that word's
    height of, loses its disc. The rest keep a disc of q99/2 of the code-word lengths (84 px). On the 2000 map, 58 of
    1,653 points keep one.
  - **Size.** It makes 24% of the in-box ink unknown on the 2000 map: codes 26%, block numbers 9%, street names 25%,
    other 29% (`experiments/textlines/labels_rule.json`).
  - **Status.** It is off by default (the shipped stages use "boxes"), and the rule written in advance did not adopt
    it (section 10).

## 3. Batch drawing (`textnet/train.py`)

- **Centres** (`data.py:504-518`; `train.py:188-197`): every 4th px each way that is on text or on ink and not held
  out, in two pools.
- **`draw`** (`train.py:200-217`). Each crop takes the text pool with chance 0.5 (TEXT_SHARE), else the ink pool,
  and a uniform centre from it. Its scale is log-uniform in [0.6, 1.6]: a window of 320/scale page px (200-533 px) is
  resized to 320 px, nearest neighbour. It is turned 0-3 quarter turns, the angle target with it. Crop 320 px, batch
  12 (`train.py:56-57`).
- **What is not done** (**inferred** from `draw` and `crop_targets`): no noise, no blur, no turn but a quarter turn,
  no flips, no contrast change (the input is 1-bit anyway).
- **`loss_weight`** (`train.py:231-238`; `_next_to`, `:220-228`) reads every known px ("off"), the known ink
  ("ink"), or the known px that are ink or one of the 8 around an ink px ("ink+1px"). The shipped recipe is
  "ink+1px", so paper more than 1 px from ink never carries loss.
- **Composites** (`textnet/composite.py:1-73`; `Recipe.composites`, `train.py:116`; `draw`, `:213-214`). A share p of
  the crops is made from a bank built before training (`Bank`, `composite.py:524`; `build` `:542`; `crop` `:612`).
  Half (A_SHARE 0.5) are of each kind:
  - **(a)** A clean Terra word (all its ink pieces inside its box, no line px within 3 px) is ORed onto the crop's
    real line ink, on or off a line by its kind's measured crossing share. Up to 20 places are tried for each of 8
    words.
  - **(b)** The crop is moved onto a clean word, and a real line piece is ORed across it, turned a random quarter
    turn. A line piece is connected line ink with a bounding diagonal of at least 100 px, cut on a 256-px grid, taken
    far from any text.
  - The masks are exact: text is only the word's ink, the paper inside the box is unknown, the line ink is not text.
  - The shipped recipe has p = 0. Tonight's variant used p = 0.3396 (34% of crops) and changed nothing beyond the seed
    noise.

## 4. The model (`textnet/model.py`)

- **Layout.** A U-Net on the ink mask (`model.py:48-80`).
  - 5 levels, with channels 24, 48, 96, 192 and 192 (WIDTH 24 x min(2^i, 8)).
  - Each down block is two 3x3 convolutions, each with batch norm and ReLU. Levels are joined by 2x2 max-pooling.
  - Each of the 4 up blocks upsamples (nearest), concatenates the skip and applies two 3x3 conv-BN-ReLU ("full"
    decoder; "light" has one). A 1x1 head ends it.
- **`half_size`** (`model.py:22-30`, `:69-80`). The input is folded 2x2 into 4 channels at half resolution
  (`pixel_unshuffle`). The head gives 4 x 7 channels, unfolded back (`pixel_shuffle`). So there is one output per page
  px, but every layer runs at half resolution. A page side is padded to a multiple of 32.
- **The heads**, 7 channels: 0 the text logit; 1-4 the kind logits; 5-6 the reading direction (cos, sin; only the
  direction counts, not the length).
- **Size**: 2.76 M parameters (2,759,740 at half size, 2,758,567 at full size; counted today).
- **Receptive field** (**inferred**, by hand from the layers): about 400 page px at half size and 200 at full size,
  several times the median code word (85 px).
- **What it does not output**: no instance, link, affinity, centre, border or box regression. There is one binary
  text map, with a kind and a direction at each px.

## 5. The loss and training (`train.py`, `stage.py`, `reader_stage.py`)

- **The loss** (`loss_fn`, `train.py:255-271`) has three terms:
  - binary cross-entropy on text, averaged over the weighted px;
  - 0.5 x the cross-entropy of the kind;
  - 0.5 x (1 - cosine) of the angle.

  The last two are averaged over the weighted text px (`train.py:64-65`). There is no class balancing, no focal or
  dice term and no boundary weighting (**inferred**: none appears in `loss_fn`).
- **`DEFAULT_RECIPE`** (`stage.py:30`): 4,000 steps x 12 crops (48,000 crops); AdamW, peak learning rate 2e-3,
  weight decay 1e-4, one cycle rising over the first 10%; half size, loss on "ink+1px", every head, full decoder,
  float32; label rule "boxes", no composites. Seed 0, deterministic algorithms, 4 threads, the Mac's GPU (mps) in an
  exclusive turn (`fit`, `train.py:381-428`): about 6.2 min.
- **Why** (`pipeline/README.md:695-760`). Trials on the validation area kept a candidate no worse on "words found" and
  "long-line ink called text". Half size cut training from 45.7 to 12.5 min at equal scores; 4,000 steps is the
  shortest schedule all of whose seeds pass. The old `Recipe()` was 8,000 steps at full size, loss on every known px,
  30% rise.
- **Stage `solver-text`** (`stage.py:33-69`): trained on all 87 training sheets (validation area in, 30-px discs, no
  placement); scores every 2000 sheet. Only `solver-text-scores` (the test drawings) and the line gate's "all-sheets"
  reference read it. No solver stage reads its chance (**inferred**, by grep).
- **Stage `solver-text-reader`** (`reader_stage.py:215-272`): four models, each with quarter q and the answered blocks
  held out, 328-px discs, the same recipe and seed 0. These are r328.

## 6. Inference (`textnet/predict.py`)

- **Tiles.** The page is scored in 1024-px tiles, each seen with 64 px of page around it (`predict.py:25-26`) and
  padded with paper to the network's multiple. A tile with no ink is skipped and left at 0. Scoring runs in float32.
- **Outputs** (`outputs`, `predict.py:81-116`): the chance, uint8 round(255 x sigmoid); the kind's argmax and
  softmax (float16); the direction as a unit (cos, sin) vector (float16).
- **Four quarter models.** Each px is scored by the model of the quarter its map point lies in; nothing is averaged.
  `page_quarters` (`reader_stage.py:127-137`) finds the quarters, and `sheet_outputs` (`:140-155`) has each model
  score only the tiles holding px of its quarter.
- **Files**: `text-reader/<sheet>/chance.png`, `kind.npz`, `direction.npz`.
- **(inferred)** With a receptive field of about 400 px and 64 px of context, outputs near a tile edge depend on the
  tiling. At a quarter split, two models meet without blending.

## 7. Word boxes (`wordboxes.py`, stage `solver-boxes`)

- **Inputs**: the r328 chance and kind probabilities of each 2000 sheet, and the page ink. Terra's boxes only choose
  the level and score the result (`wordboxes.py:4-5`).
- **The ink gate** (`ink_gate`, `:71-74`; INK_R 4, `:51`): only px within 4 px of ink (a 9x9 ellipse dilation)
  count, since the chance on paper is no evidence.
- **The threshold**: chance >= ceil(level x 255) (`:84-85`).
- **The level** (`choose_levels`, `:266-278`), from 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995 and 1.0 (`:49`).
  - For each quarter it is the level with the highest recall of Terra's zoning codes (IoU >= 0.3) on the other three
    quarters; ties go to the higher level.
  - Chosen: nw 0.995, ne 0.99, sw 0.995, se 0.995 (`boxes/levels.json`), i.e. chance >= 254/255 or 253/255.
  - Recall climbs all the way to 0.99-0.995 and falls at 1.0, because the chance is saturated near 1 on most ink
    (`:49-50`).
- **Components**: 8-connected, kept with at least 20 ink px (`:86`, `:113`; MIN_PX `:52`).
- **The axis** (`:102-106`, `:136`): from the chance-weighted centre and second moments, the major axis
  theta = 1/2 atan2(2 sxy, sxx - syy), stored modulo 180 degrees.
- **The extents** (`:108-119`): the grouped box is the min and max along and across the axis over all the
  component's px (+-0.5); the trimmed box is the same over its ink px only.
- **Fields per box** (COLUMNS, `:60-62`): sheet, i, level, quarter, page_x, page_y; axis, length, height, t_length,
  t_height; mean_chance and ink_px (over its ink); p_zoning, p_block, p_street, p_other (the mean kind probability over
  its ink; no later stage reads these, **inferred** by grep); map x, y, lab_lot, street; the g and t corners.
- **Placement** (`:237`): a box is kept only when its centre is inside the sheet's footprint and on the mapped area
  (`prints._place`).
- **Not done**: no splitting of touching words, no joining of separate letters, no non-maximum suppression. One
  component is one box: 8,794 boxes.

## 8. Reading (`textread/`)

**8a. The crop** (`crops.py`).
- **Our box** (`box_crop`, `crops.py:99-103`): the trimmed rectangle, read at the box's axis.
- **Terra's box** (`rectify`, `crops.py:91-96`): Terra's rectangle at Terra's angle. A slanted word's height is
  floored at the 10% quantile of its kind's quarter-turn heights (21 px for codes).
- **`cut`** (`crops.py:79-88`) pads 0.4 word heights at each end and 0.2 above and below; samples through an affine
  map (an exact copy at a quarter turn, bilinear otherwise, white off the page); resizes to 32 px tall, aspect kept
  (INTER_AREA when shrinking), at most 1024 px wide. The 180-degree copy is the crop turned a half turn.

**8b. The recogniser** (`textread/model.py`).
- **Layout**: docTR's `crnn_mobilenet_v3_small`, re-expressed in plain torch (`model.py:113-127`).
  - MobileNetV3-small features, with the depthwise stride set to (2, 1) at layers 2, 4 and 9. A 32-px-tall crop comes
    out one row tall, with one frame per 4 px of width.
  - 576 channels feed a two-layer bidirectional LSTM of 128 units, then a linear layer to 46 classes.
- **The alphabet** (`model.py:21`): space & ' ( ) , - . / 0-9 A-Z (45 characters) and the CTC blank.
- **The input** is grey, not 1-bit: divided by 255, normalised (mean 0.694, sd 0.299), repeated over 3 channels and
  padded on the right with paper (`model.py:164-184`).
- **Size**: 2.06 M parameters (0.93 M features, 1.12 M LSTM). Trained from scratch, not from docTR's weights
  (`textread/train.py:3`).

**8c. Its training data.**
- **Real crops** (`sheet_crops`, `crops.py:170-225`): every Terra word of every kind on the 2000 drawing's three
  scans, with Terra's string upper-cased as the target (`norm_text`, `:56-60`).
  - Plus non-text windows, 0.2 per word: a Terra word's size and angle centred on ink away from every word and unknown
    px. Their target is "", at weight 1 - 2.5% = 0.975.
  - 34,937 crops: 6,064 codes, 13,726 block numbers, 7,884 street names, 1,486 other, 5,777 non-text.
  - Quarter q's reader gets the crops touching neither q nor the answered blocks, 17.9-21.9 k (`store.py:80-88`).
- **Synthetic crops** (`textread/synth.py`), 50,000 per quarter.
  - Printed code forms, in the template reader's font, learned from 959 training labels (`codereader.glyphs`). Half
    are drawn by frequency, half uniformly.
  - Measured size, spacing jitter, a residual turn of N(0, 1.5 degrees), measured ink noise and speck rate.
  - With a 47-58% chance, a real non-text window of the 2000 scan is laid over the crop (the darker px wins).
  - Known faults are kept: the box offsets' sign, and windows centred on ink (`synth.py:31-35`).
- **Training** (`textread/train.py:45-60`; `fit`, `:154-218`): 12,000 steps in batches of 64 (eight drawn at once,
  sorted by width), 40% synthetic. Augmentation: width scaled up to 12% either way, 0-6 px of paper at each end, up to
  2 px shift up or down. Loss: CTC negative log likelihood over the target's length, weighted mean. AdamW 1.5e-3, one
  cycle rising over 15%, weight decay 1e-4, gradients clipped at 5, no early stopping.
- **Monitor** (logged only): best-path character error on 600 held-out words of every kind, 11.1-13.2% at the last
  step (`text-train/held.json`).

**8d. The 0/180 classifier** (`Direction`, `model.py:130-144`).
- Four 3x3 conv blocks (32, 64, 64, 128 channels; batch norm, ReLU, max-pool), a global mean and one logit: 130 k
  parameters.
- Trained 3,000 steps on real word crops only, each turned a half turn on a fair coin, with binary cross-entropy.
- q = p(the 0-degree crop is upside down), read on the 0-degree crop only (`read.py:129-132`).

**8e. The posterior** (`ctc.py`, `read.py`).
- Each box is read as cut and turned a half turn (`read_one`, `read.py:91-102`).
- For each read, CTC's exact forward algorithm on the frames' log-softmax gives log p(s | x) for each of the code
  list's 95 printed forms (57 codes; prior.json `forms`) and for "" ("none") (`ctc.py:32-51`). "other" is 1 - the
  forms - none, floored at 0.
- The mixture: p(h | box) = (1 - q) p(h | x0) + q p(h | x180), over forms, "other" and "none" (`ctc.py:54-60`).
- The closed posterior sums the forms by code and renormalises over the codes alone (`ctc.py:63-72`).
- It runs on the CPU, batch 1, one thread, so a crop's numbers never depend on batching.

## 9. Downstream use

- **The walls' text input.**
  - `textgrid.text_chance` (`textgrid.py:52-82`) reads each 2000 sheet's r328 chance through its placement onto the
    4-ft grid, at grid px whose page px is ink. Where sheets overlap, the largest chance wins.
  - `text_weighted` (`wallevidence.py:32-35`, `:671-676`) takes, for each border px, the chance t of its nearest
    stroke px. The stroke and profile features are weighted by (1 - t), t is a feature of its own, and the street
    crossing weighs its ink by (1 - t).
  - A ridge logistic regression is fitted per held-out quarter (`walls.py`). `text_weighted` is in
    `DEFAULT_CANDIDATES` (`walls.py:77`) so that the walls read no LLM output; its held-out log loss ties with
    Terra's boxes (`walls.py:72-76`).
  - The root-cause note (§5) infers that line ink called text is discounted as lettering, which removes wall evidence
    where labels sit on lines.
- **Labels.**
  - The solve's labels come from `prints.parquet`: Terra's code boxes joined with Sol's points, plus the turned pass,
    the third read and its re-check (`prints.py:1-30`).
  - `trust.py` fits a logistic regression per quarter over 23 features of those five LLM reads (`trust.py:48`,
    `:59-83`). Its `readings_{q}.npz` go through `labels.py` to the energy.
  - **Our reader's readings (`text-read`) are read by no solver stage on this branch** (**inferred**, by grep). They
    are measured only.

## 10. Measures and today's numbers

**The detector (`textnet/scores.py`).** The measures are on ink px, with the unknown px left out; "called text" is a
chance >= 0.5.
- The measures: AUC and log loss of text ink against other ink; text ink found; ink outside every box called text;
  Terra's words found (mean chance over a word's ink >= 0.5, `scores.py:194-213`); long-line ink called text (strokes
  >= 200 ft long and <= 48 ft wide, `scores.py:91-119`).
- **r328, each model on its own quarter** (`text-reader/scores.json`, 328-px discs left out): AUC 0.990-0.998; text
  ink found 98.96-99.58%; outside-box ink called text 2.2-3.5%; words found 99.4-99.9% (codes 523/524, 812/812,
  340/340, 474/474).
- **All-sheets model on the test drawings** (`text-scores/scores.json`): AUC 0.991; text ink found 94.0%; words found
  93.3%; outside-box ink called text 2.4%; long-line ink called text 3.7% (4.9% of long-line ink lies in a box; 0.27%
  of the rest is called text).
- **These flatter.** They count in-box line ink as correct text and leave the unknown discs out, so they can't see
  the line over-call (`2026-11-16-text-target-root-cause.md`, §3).

**The text-vs-line gate (`textnet/linegate.py`, tonight)**: each value is the share called text, with
block-bootstrap intervals.
- **M1a** is district-border ink, weighted against the table's cracks and split in box / near / far.
- **M1b** is run-rule line ink.
- **M1c** is all ink outside every box.
- **M2** is glyph ink in boxes, away from lines.
- **Clean** is all the ink in clean words.

| source | M1a in box | M1b in box | M2 | M1c | clean words |
|---|---|---|---|---|---|
| r328, all quarters | 0.987 | 0.975 | 0.999 | 0.366 | 0.995 |
| r328, NE | 0.992 | 0.969 | 0.9994 | 0.547 | 0.985 |
| old-ne | 0.629 | 0.818 | 0.973 | 0.025 | 0.957 |
| control, NE, seeds 0-2 | 0.39-0.41 | 0.51-0.59 | 0.962-0.976 | 0.041-0.062 | 0.927-0.935 |

- The composites changed nothing beyond the seed noise.
- By the rule written in advance, nothing is adoptable. The control's line gain costs 3.3 points of code letters off
  lines on NE.

**The boxes (`boxes/summary.json`; Terra's zoning codes, IoU >= 0.3).**
- Recall: nw 73.5%, ne 46.8%, sw 78.8%, se 69.0%.
- Touched at IoU >= 0.1: 72-91%. Found with the axis within 10 degrees: 31-55%.
- Of the owner's answered labels, 641 of 731 points lie inside a box, and 418 of 668 Terra boxes are met at IoU 0.3.

**The reader (`text-read/measures.json`; the owner's 719 scored answered labels).**
- **At Terra's boxes** (n 662): closed top-1 96.1% [94.4, 97.4], top-3 99.2%; open top-1 91.8%. The LLM chain
  scored 96.9% and spike S2 95.6-97.3%.
- **At our boxes**: 633 of 719 labels get a box.
  - Closed top-1 on those is 46.2% [42.4, 50.1]: nw 53.7%, ne 33.5%, sw 61.3%, se 57.6%.
  - Over every label it is 41.1%; open top-1 is 24.6%.
- **On the same 595 labels**: 96.3% at Terra's boxes against 47.7% at ours (difference -52.6 to -44.7 points).
- **Held-out words at Terra's boxes**: code words put 0.90-0.93 on the code list (closed top-1 against Terra
  92.0-98.9%); block numbers and street names put >= 97% on "other"; non-text windows put >= 97% on "none".
- **All 8,794 boxes**: the top hypothesis is "other" for 7,106, "none" for 1,081 and a code for 607. Mean q is 0.384.

## Design choices and their consequences

1. **A per-px binary chance of text, with no word-instance output** (`textnet/model.py:3-5`).
   - *Why:* the target "text = inside a Terra box" came from an R&D script. The detector was first built so the
     walls could discount text strokes, which needs a per-px chance (root-cause note §1-2; `walls.py:7-8`).
   - *Consequence:* words have to be rebuilt afterwards (choice 2). No link, affinity or region-score output separates
     neighbouring words or joins gapped letters.
2. **Boxes from thresholded connected components** (`wordboxes.py:77-139`; a port of spike S3).
   - *Why:* the level is chosen only by Terra's code recall on the other quarters.
   - *Consequence:* the level lands at 0.99-0.995, where the chance is saturated. Code recall is 47-79%, and the
     reader falls from 96% to 47% on the same labels: **the box step, not the reader, loses about 49 points.**
   - The gap between touched (72-91%) and found (47-79%) suggests merged or split components (**inferred**; not broken
     down by cause).
3. **An axis but no reading direction from the boxes.**
   - *Why:* the axis comes from second moments. The detector's direction head is trained and saved (`direction.npz`),
     but nothing reads it. 0 against 180 is settled by a mixture weighted by a classifier.
   - *Consequence:* the axis is within 10 degrees for only 31-55% of codes. At Terra's boxes the axis alone plus the
     mixture scores the same as Terra's own direction (96.07% both), so the mixture is not the problem.
   - **(inferred)** Short or square words, such as "P", give a poorly defined second-moment axis.
4. **A closed vocabulary at read time, over an open CTC network.**
   - *Why:* the code list's printed forms are known (the template reader's prior), and an exact CTC forward gives p
     over forms, "other" and "none" (`ctc.py:1-19`).
   - *Consequence:* 96.1% closed top-1 at Terra's boxes, close to the LLM chain's 96.9%. A code outside the list (5
     answer strings, e.g. "MB-OS", "CMNC") can't be named; its mass goes to "other".
5. **Training labels from LLM boxes, filled as rectangles.**
   - *Why:* the owner chose "our own detector/reader trained on Terra's saved boxes", but what a box means px by px
     was never decided (root-cause note §2).
   - *Consequence:*
     - Lines through labels are taught as text: r328 calls 97-99% of in-box line ink text, and the scores, built from
       the same boxes, counted that as right.
     - The unknown discs meant for Terra's misses grew to 328 px from one mis-box. With them r328 calls 37% of all
       outside-box ink text (55% on NE).
     - `lines-unknown` cuts in-box line calls to 0.39-0.59 but costs code letters, and was not adopted.
     - Terra's 2.5% miss rate and its loose slanted boxes (`textread/crops.py:10-13`) enter the targets unchecked.
6. **A 1-bit ink input** (`data.py:3`; `stage.py:62`).
   - *Why:* not stated. **(inferred)** The detector's job was to sort ink, and ink is what the walls read.
   - *Consequence:* the detector can't use grey level, stroke texture or the scan's tone, while the reader reads grey
     crops.
7. **Half size** (`DEFAULT_RECIPE`).
   - *Why:* speed. Training fell from 45.7 to 12.5 min at equal validation scores (README trials).
   - *Consequence:* no measured loss, but the trial measures couldn't see the line over-call. Its effect on thin
     strokes at a 2-px resolution is untested (**inferred**).
8. **The ink-only loss ("ink+1px")**.
   - *Why:* it was the best of three in the trials (95.01% words found against 94.80%), within the tolerance.
   - *Consequence:* the chance on paper is unconstrained, so the boxes need the 4-px ink gate. Px 2-4 px from ink, such
     as the gaps between letters, carry no lesson, so whether a word's letters join into one component rests on
     untrained px (**inferred**).
9. **Four held-out quarter models, chosen per px with no averaging** (`reader_stage.py:1-7`).
   - *Why:* the reader and the walls should read only outputs of a detector that never saw the place.
   - *Consequence:* honest held-out numbers, but no ensemble gain, and the models meet at the quarter splits without
     blending (**inferred**).
10. **Narrow augmentation for the detector**: scale 0.6-1.6 and quarter turns only; no synthetic text (composites
    off) and no turn but a quarter turn. *Consequence:* slanted words are learned only from the real examples Terra
    boxed (**inferred**).

## Caveats

- The r328 outputs, the boxes and the reads were built before tonight's code changes. Under the defaults they are
  byte-identical (results note, "Stages left stale").
- The reader's measures are on the owner's 719 answered labels, which all lie in held-out blocks.
- Every detector and reader number above is held out by quarter, except the all-sheets model's test-drawing scores,
  which come from whole drawings it never saw.
