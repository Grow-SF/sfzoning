# A learned reader for the hatched zoning maps (1921, 1929, 1935, 1948)

Status: the design, agreed with the owner in conversation on 2026-11-18. It replaces the hand-built 1921 patch
pipeline of `docs/superpowers/plans/2026-11-18-1921-region-hatching.md` (paused at its Task 3; see "What came before").

## Intent and why

The hatched editions (1921, 1929, 1935, 1948) draw each zoning district as a fill pattern. The legend, the same on all
four, has six districts:

| District | Pattern |
|---|---|
| 1R | one diagonal family, at a fixed map angle |
| 2R | blank white |
| C | solid black |
| LI | diagonal cross-hatch |
| HI | square grid |
| U | checkerboard |

Today's published 1921 map was built from paid LLM reads, and it is right on 199 of the owner's 240 random 1921 lots
(82.9%). The owner's rule is no LLM output in the pipeline: LLM outputs may only train and validate. So the hatched
editions need a deterministic reader that beats today's map.

The owner wants that reader to be a **learned statistical model**, not a stack of hand-made rules. They have said so
from the start: on 11-13, drop the cleaning step and learn from the original scan as printed; and memory
"probabilities, not cutoffs". The pipeline this replaces failed for that reason. Its hand-built mask erased the
pattern ink of whole checkerboard blocks, and every fix round added another special-case rule.

## What the owner said (their words, 2026-11-18)

- "I'm looking at the hatching, which always falls inside blocks and never crosses streets."
- "blocks are not uniformly zoned - they can be multiple zones (multiple hatchings)."
- "there's no line - it just changes." Inside a block, a district boundary is only a change of pattern.
- "There is sometimes text printed in the block, and the text will always get a plain white background so it can
  interrupt the printed hatching." And: "a real white patch won't have text in it."
- "why are you focused on lots, and not blocks? lots will get what their block zone is. and with this bad warping,
  the lot alignment may be off."
- On the model: "a CNN that downweights printed text influencing the zone".
- Scope: all four editions; "train on 1921 and we can come back to the later years". Don't wait on answers for 1929,
  1935 and 1948.

## Goals

1. **Read the scan as printed.** Each sheet's `page.png` goes in at the scan's own resolution, with no erasing,
   cleaning, masking or hand-made special cases. Whatever separates districts, the model learns from data.
2. **The drawing's own units.** Work in each scan's own pixel coordinates.
   - **Find the printed blocks:** the areas bounded by streets.
   - **Split each block into pieces** wherever its pattern changes, and give each piece one of the six districts,
     as probabilities.
   - Finding blocks and splitting them are both learned, not rule-based.
3. **What the model learns to see past:**
   - text and its white knockout, which belongs to the pattern around it;
   - block outlines, block numbers and street names, which appear in every district;
   - the gaps between 1R's wide lines.

   The owner's suggestion is a CNN that learns how much each part of the scan should count (text, knockouts and outlines
   little), pooled over the unit being labelled.
4. **Lots last.** Each lot takes the district of the piece it falls in. Lots are placed on the scan through the sheet's
   placement, so misalignment matters only for lots at a piece's edge.
5. **Labels:**
   - the reader's 1921 reads (`data/work/1921/<sheet>/district_pattern.json`: blklot, code, confidence), weighted by
     their confidence, attached to the piece each read lot's centre falls in;
   - the legend swatches;
   - the owner's 1921 answers (661 usable, 240 of them random lots). These are split by area: one part trains, the
     rest scores. The owner said on 11-18 that their answers are for both.
6. **One measure.** The score is the owner's held-out random 1921 lots, compared lot by lot with today's published map
   by Task 1's scorer `pipeline/src/sfzmaps/solver/answers1921.py`. That scorer has paired differences, block-bootstrap
   intervals, and footprint matching for stacked lots.
   - **The target:** clearly beat 82.9%. With about 240 random lots, about 89% or more shows a clear win.
   - **The reference:** the paused pipeline's per-lot evidence reached 219/240 (91.2%). The new model should be judged
     against that too.
7. **Pictures first.** Every step's output is painted over real sheets and shown to the owner before anything builds
   on it, and before any review. That means the found blocks, the pieces, and the predicted districts, on one dense
   sheet and one southern sheet at least, whole and zoomed.
8. **All four editions.** Version 1 trains and scores on 1921, then runs on 1929, 1935 and 1948 for the owner to look
   at. Those three get scored when the owner answers a random-lot page for each.

## Non-goals

- The joint solver for these editions, unless the pictures show a need. Its walls look for printed lines, and
  boundaries inside a block have none. (My call; the owner accepted it in the playback.)
- Fixing the sheet alignment or the warp. That is its own project, with measurements running now in
  `scratchpad/align/`. This design reads each scan in its own coordinates, so it does not depend on the warp.
- Publishing. A follow-up plan does stages, `sfzmaps build` and `ship` once the score wins.
- 1929, 1935 and 1948 answers or scores in version 1.

## Anti-goals (a version that does any of these fails, even if it scores well)

- **Any hand-made cleaning, mask, threshold, window or special-case rule** in the reading. Engineering settings are
  allowed, stated in comments: the network's size, the training schedule, and the input tile size, which is set to
  cover enough context.
- **LLM output used at inference.** The reads are training labels only. This also means each sheet's scale or period
  must not be measured on the reader's lots, as the paused pipeline did.
- **Learning from our own solves or predictions:** no pseudo-labels.
- **Scoring on answers from the same area as training answers.** Neighbouring blocks share districts, so the split has
  to be by area.
- **Reporting a score the owner hasn't been shown pictures for.**

## Constraints

- $0: no paid or LLM calls.
- PyTorch is already a dependency. One GPU process at a time, through `/tmp/sfzmaps-mps.lock`.
- Load around 20–80, one thread per worker.
- **Deterministic outputs:** the same inputs and device give the same bytes. Fixed seeds and the device are recorded,
  as the warp does.
- Licences: no GPL-family code in shipped code.
- Experiment outputs go under `pipeline/data/solver/experiments/hatch-cnn/`, never committed; code goes on a branch.
- **The sheets differ:**
  - 1921 scans run at about 2.5–4.7 ft/px, and two are at 2.72;
  - one sheet is scanned sideways;
  - one section was scanned three times (…341084, …727628, …100525362) and others twice;
  - 7 of the 23 1921 page folders are not map sheets.

  The model has to handle these by learning, or by being given the scan's measured scale. It must never rescale or
  clean the scan by hand.
- Overlapping scans of the same place: label each place once, on the sheet the published draw order picks (as
  `sampling.sheet_order`). Comparing the duplicates is a free consistency check worth reporting.

## What came before (reference only)

- **The paused hatch-regions branch**, 62ff52f..b078350:
  - Task 1's scorer (keep);
  - Task 2's hand-built patch measures and outline mask (superseded);
  - Task 3's patch classifier and per-lot evidence (superseded), which reached 219/240 on random lots.

  Uncommitted R16 work sits in the saved-reads worktree, parked.
- **What not to repeat:** judging steps on summary numbers without pictures; adding rules to fix rules.
- **Branches:** `label-checks-regions` holds the region check-page builder, which a later version can use to ask the
  owner one question per block piece. `hatch-depth15` holds the Section 15 depth geometry, unused here.

## Settled after the builder check

**The owner's answer: how the model learns where streets are.** Option (a), as the owner chose. Today's streets
(DataSF, the pipeline's reference data `data/ref/streets.parquet`, with the year's blocks from
`data/ref/parcels.parquet` where they help) are placed on each scan through the sheet's placement. They are the
training target for a learned "street" output.

The labels are positionally noisy: the placement is off by about 3–13 ft, more far from control points, and some
streets changed after 1921. The model learns through that noise. It does not use our rule-traced `blocks_px` and
takes no hand rule.

From the outputs:
- a block is an area between the learned streets;
- a piece is a connected area of one learned district inside a block;
- each lot takes the district of the piece it falls in.

The first picture shown to the owner is the learned streets and blocks on two sheets, before any district
training builds on them.

The per-lot labels (reads and answers) supervise the district output by pooling over each lot's pixels with the
learned weights. That makes no use of the model's own pieces during training, so the labels never loop on the
model's output.

**Answered already:**
- **When version 1 is done.** It is done when it clearly beats today's published map on the owner's random 1921 lots:
  the paired interval against published excludes zero, and the owner has seen the pictures. The 219/240 of the paused
  pipeline is a reference, not a gate. Owner: "it's okay to just be better"; the aim is a published map with fewer
  errors.
- **Later years' reads.** "train on 1921": the 1929, 1935 and 1948 reads are not training labels in version 1.

**My calls** (each changeable):
- **The score.** k-fold by area: each random 1921 lot is scored by a model that saw neither its area's answers nor its
  area's reads, so all 240 random lots are scored. The areas are the solver's quarters (nw, ne, sw, se), as the
  reference did. Withholding the reads too keeps the model from learning the published map's own mistakes on the lots
  it is scored on.
- **The deliverable.** The model that runs on 1929, 1935 and 1948 is trained on everything in 1921, and is judged
  through the fold models' scores.
- **A fair comparison.** One extra run trains on the reads only, without the owner's answers, so the comparison with
  the paused pipeline's 219/240 (which never trained on answers) is like for like.
- **The 1948 input.** 1948 is the same kind of black-ink drawing, but its page.png is a full-tone colour scan (RGB, the
  David Rumsey copy: cream paper, grey ink tones, some pencil marks), while the 1921, 1929 and 1935 scans are 1-bit
  black and white. The network takes each scan's luminance. That is a choice of input representation, not a cleaning
  of the ink. A model trained only on 1-bit 1921 scans may not carry over to full-tone ink, so version 1's 1948 run is,
  like 1929's and 1935's, for looking only until 1948 has answers to score it.
- **The placement.** Reads and owner answers attach, and lots are placed, through `data/work/<series>/<sheet>/
  transform.json`. That is the placement the reads were made through. The alignment measurements (scratchpad/align)
  found it keeps straight print straight, where the warp's correction bends it by up to 12.6 ft (p90) on five 1921
  sheets.
- **U can't be scored.** None of the owner's answers is U. The score cannot measure U, and the report says so.
- **The scorer's docstring** ("never training data") gets updated to the owner's 11-18 rule: answers train and score,
  split by area.

## Left to the builder

- **The network's form.** For example a fully convolutional network giving per-pixel district evidence plus a learned
  per-pixel weight, pooled per piece; or per-pixel segmentation trained through pooled losses.
- **How blocks and pieces come out of it.** For example a learned street/boundary map and its connected areas. Neither
  may be a hand-made rule.
- **Tile size, augmentation, training schedule and the area split** for the owner's answers. The split must be by area
  and stated.
- **Calibration** of the per-piece probabilities.
- **How the duplicate scans are used** beyond labelling each place once.
