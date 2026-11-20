# A learned map-area / frame finder for every map (issue #71)

## Why

Every scanned sheet prints a frame around its map (thick bars, thin rules beside them, corner ornaments, ruled-off
legend and title boxes) plus margins and scan background. None of it is the city. Today three hand-tuned pieces of code
decide where the map ends, one per family of series, and each fails somewhere (issue #71: 1921 read its frame rule as
district C; a duplicate 1921 crop ran 700 px past its frame; 2000 p05's legend rule counted as map). #69's placement
cannot place real sheets until this exists.

The owner's decisions:

- **One approach for every map, learned, not hand-described** ("Listen to me. One approach."; "I prefer options that
  learn automatically").
- **Detect and erase:** the same result tells reading and placement which ink is the city, and lets the paper layer
  erase printed border rules (AGENTS.md).
- **Approach A** (below), trained on a few hand-drawn sheets. The owner allowed hand labels because there are few;
  Claude drew them and the owner reviewed them.
- **Corner ornaments are frame**, cut out (owner, 2026-11-19).
- **Start small:** 12 labelled sheets, 4 each from 1921, 1960 and 2000; more only if the results ask for it.
- **Which pages are maps** is a hand label (`pipeline/inputs/page_kinds.csv`, 59 non-map pages, owner-confirmed);
  non-map pages are never read for frames.

## Goals, non-goals, anti-goals

Goals

- Per sheet, each page pixel's probability of lying in the printed map area, at the scan's own resolution.
- The same network, weights and code for every series (1921 through 2000).
- Held-out scores on the labelled sheets, and a picture of every map sheet for the owner's eye.

Non-goals (this design)

- Wiring the output into reading, placement or the paper layer. That follows once the owner has seen the pictures
  ("pictures before anything builds on it"); each is its own change.
- More labelled sheets, synthetic frames (approach B) or street outputs (#69): only if the results ask for them.

Anti-goals (failure even if it "works")

- Any per-series or per-year code path, window, inset or threshold written by hand.
- A model scored on a sheet it trained on.
- Labels or inputs from LLM reads, today's placement or the old crops (`frame.py`, `neatline_mask`, the warp outline).
- Non-determinism: two runs on the same inputs must give the same output.

## Constraints

- $0: no paid reads. One GPU (MPS) job at a time, under `/tmp/sfzmaps-mps.lock` as the detector does. Load around 20.
- Shipped code licences: MIT/BSD/Apache only (no GPL-family code).
- Data under `pipeline/data/` is never committed; labels and page kinds are inputs in git.

## Approaches considered

1. **Shared print across an edition (tried, failed on the early maps).** Line up the sheets of an edition page to page;
   ink most siblings share is the printed template, i.e. frame. Clean on 2000 p05. On 1921, 1929, 1935 and 1948 it
   found nothing. Root cause (measured): those frames are drawn to fit each section, so they are not at the same page
   position from sheet to sheet, and only a few siblings agree on any side; the fit preferred "no frame"
   (log-likelihood −3,173,305 vs −3,236,324 with the frame placed). Spike code: `pipeline/scripts/spikes/issue71_*`.
2. **A: a few labelled sheets and a two-scale network (chosen).** The ICDAR 2021 MapSeg task-2 winner (UWB) ran a
   fully convolutional network over the whole downsampled page, trained on 26 labelled sheets, and beat rule-based
   entries 4–6× (HD95 19 px vs 85–126 px on 1894–1937 Paris atlas sheets). A 2023 cadastral-frame study found the same
   split: networks alone localise the border poorly, rules alone fail on new layouts.
3. **B: A plus synthetic sheets** (real frames cut from labelled sheets, drawn at new sizes over real map interiors).
   Kept for later, if A's held-out pictures show it needs more layouts.
4. **C: no labels, self-training from a rule-based "outermost closed loop" detector.** Rejected: hand rules and
   thresholds come back through the detector, and the network learns their failures.

## Design

### Labels

`pipeline/inputs/frame-labels/<series>/<sheet>.json`: one or more polygons in `page.png` pixels. Conventions are in
that folder's `README.md`:

- the edge sits just inside the innermost frame rule, never on it;
- corner ornaments are cut out;
- ruled-off legend and title boxes are cut out;
- floating notes stay inside;
- a side without a frame follows the page edge.

The 12 sheets:

| Series | Sheets |
|---|---|
| 1921 | 094149543, 094325427, 094142210, 094355582 |
| 1960 | 1, 3, 7, 12 |
| 2000-digital-zoning | p01, p03, p05, p09 |

The target is the polygon mask: 1 inside, 0 outside.

**Gate:** the four 1921 labels were first drawn with the corner ornaments inside. They are being redrawn to cut them
out. The owner sees the redrawn 1921 labels before anything trains on them.

### The network: two scales

1. **Page pass (coarse).** The whole page, downsampled so its longer side is a fixed size, goes through a U-Net deep
   enough that every output pixel sees the whole page. It outputs the map-area logit per coarse pixel. This is where
   "this rule is the edge of the map" is decided: it sees that the city stops there.
2. **Pixel pass (fine).** The scan at full resolution, as printed, tiled. The input is the luminance plus the page
   pass's map-area probability, upsampled to full resolution. It goes through the PR #70 U-Net form (`hatchnet/model.py`:
   5 levels, about 120 px reach, batch norm, nearest upsampling). It outputs the map-area logit per pixel. This places
   the edge between the thin rule and the map's first line, which the coarse pass cannot resolve.

Both passes are trained on the same masks. The page pass trains first. The pixel pass then trains on top of the page
pass's held-out output for each training sheet (see Folds), so it never learns to trust an over-confident in-sample
coarse prediction.

**Input as printed:**

- nothing erased, thresholded or cropped;
- no series or scale input (PR #70 ruling R6: a per-sheet scale input let the network treat sheets differently);
- pages stored turned are fed as stored.

**Augmentation:**

- quarter turns and mirror flips, so pages stored turned are covered;
- small scale changes;
- brightness, contrast and blur changes, covering 1-bit, grey and photocopy scans.

### Folds and scores

**Folds.** Four folds by sheet. Each fold holds out one sheet from each of 1921, 1960 and 2000 (3 sheets) and trains on
the other 9. Every labelled sheet is predicted by a model that never trained on it. Sheets that are the same scan (the
1921 Section 12 pair) are never split across a fold; neither is in the 12.

**Scores, fixed now and reported per held-out sheet and per series.** All are probabilities, with no cutoff (owner,
2026-11-19). "Ink" is soft: each pixel counts with weight 1 − luminance (the scan as stored; luma for the colour 1948
photos). The border band is ink within 60 px of the label's outline: "frame ink" outside it (the frame rules,
ornaments and box rules, including 1921's thin rule) and "map ink" inside it (the map's first lines).

1. **Held-out log-loss (the primary score).** The ink-weighted mean of −log(the
   probability given to the right answer), reported separately for frame ink and map ink in the border band, and
   over all ink on the page. A proper scoring rule: it rewards being right and being calibrated, and charges heavily
   for a confident mistake (the thin rule called map at 99%).
2. **Frame ink called map.** The ink-weighted mean predicted map probability on the frame ink. Lower is better. The
   easy-to-read companion of score 1.
3. **Map ink called frame.** The ink-weighted mean of 1 − the predicted map probability on the map ink. Lower is
   better.

**On the side, for comparison with the ICDAR 2021 MapSeg competition only** (owner: "keep it on the side so we can
compare to the competition"): the distance from the probability-½ contour to the label outline, in page pixels
(median and HD95, as MapSeg reports). It is never used to choose anything.

**Pictures, for the owner, every map page of every series** (about 370) from the final model trained on all 12:

- the scan, with each ink pixel coloured by its map probability;
- the ½ contour drawn on top.

The 12 labelled sheets are shown from their held-out folds. The odd cases are called out:

- 1948 sheet 7, which has no frame;
- the multi-map photo;
- the cropped 1980s scans;
- the half-page 1935 scan;
- the Section 12 duplicates.

**No pass bar.** This is an experiment. The plan ends with the scores and the pictures, and the owner decides what
comes next: more labels, approach B, or wiring the output in (the owner's rule: experiments report measurements and
are not cut off at a pass margin).

**The border test** (`pipeline/scripts/spikes/issue69_border_test*.py`) is re-run with today's ink weighted by the map
probability, on 2000 p03, p05, p09 and p13 and on 1921 094149543 and 094325427. It lives on #69's unmerged branch and
imports #69's placement code, so it runs there, reading this work's per-sheet output: a check, not part of this
build (Claude's call). Expected:

- lines beyond the printed map get no evidence;
- 1921's thin rule carries none.

### Where the code goes

- A package `pipeline/src/sfzmaps/frame/`, holding the network, the label masks, training, prediction and pictures.
  The U-Net from PR #70 (`hatchnet/model.py`, our own code on the draft `hatch-cnn` branch, which won't merge) moves
  to one shared module on `main`, `pipeline/src/sfzmaps/nets/unet.py`. This package uses it, and #69's street finder
  imports the same module later instead of keeping a copy (Claude's call).
- A `sfzmaps frame` command.
- Tests with synthetic pages for the parts that are logic: the polygon-to-mask conversion, the folds never training on
  a held-out sheet, and the deterministic same-input/same-output rule.
- The model and the outputs go under `pipeline/data/frame/`.
- It becomes an `sfzmaps build` stage only after the owner has seen the pictures.

## Left to the builder

- The coarse page size and the page pass's depth. It must see the whole page; a longer side of about 1024 px is
  expected.
- The pixel pass's tile size and overlap.
- Loss weighting between ink and paper pixels.
- Training steps. Checkpoints are chosen by log-loss on sheets held back from each fold's own training sheets (an
  inner split), never on the fold's scored sheets or their pictures, so the reported scores stay untouched (Claude's
  call).
- How the page pass's held-out predictions for training sheets are produced (nested folds, or out-of-fold predictions
  inside each fold).
