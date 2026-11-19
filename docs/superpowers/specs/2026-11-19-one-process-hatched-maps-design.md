# One process for every map: the hatched maps on the 2000 chain

Status: DRAFT, not yet approved by the owner. Written from the conversation on 2026-11-19. Placement (one process, the
warp's line fit as the control points) moved to issue #69. It supersedes the input and placement parts of
`2026-11-18-hatched-maps-learned-reader-design.md` (per-sheet scans, `data/work` placement, learned streets and blocks,
the reader's own sheet-ownership rule). That spec's district reader (pooled lot losses, the read-error table, folds,
the score) carries over, with the changes below.

## Intent and why

The owner's rule: "THE PROCESS WE DESIGNED FOR 2000 SHOULD APPLY TO ALL MAPS. When the processes fail, we should fix
them." The one part that differs for the hatched maps (1921, 1929, 1935, 1948) is **reading the hatching**. The
solver for these maps changes or goes away, "since there are no zoning boundaries on these old maps": the hatching
gives each lot its district directly.

Why now: the 1921 reader built on 11-18/19 read each sheet separately, and every bug found came from that. The bugs
were:
- the printed frame read as solid C (Section 12, lot 6974019), because 1921 fell back to the neatline crop when the
  2000 frame finder failed on every 1921 sheet;
- blank "Join Sec." corners claiming 1,842 lots;
- a duplicate scan (094727628 is the same image as 100525362);
- an ownership rule that handed 25,000 lots to a darker scan;
- lots pooled across sheets.

The owner: "You're going to have problems at sheet boundaries, right? That's why we need to use the fully formed
warped city image." And on the fallback: "This neat line fallback thing is the source of the error that we started
with and here you are deferring it for later."

## The process (agreed: "That's correct")

Every map goes through the same code. Where a stage fails on a map, that stage is fixed.

1. **Place each sheet: one process, on every map, 2000 included.** The owner: "Use it everywhere for the control point
   finding. I don't understand why the warp process and the control point finding process are different. They are the
   same process."

   Today the code splits placement in two:
   - `refine` matches traced block corners (`linework.extract_blocks`) to today's corners and fits a placement;
   - `sheetwarp` then fits a correction on top.

   The 11-18 analysis (`scratchpad/align/report.md`) found the 1921 trouble is **too few control points**:
   - placement error grows with distance from the nearest control point, from 2.9 ft within 100 ft to 12.8 ft beyond
     1,600 ft;
   - 1921 sheets have 79–1,402 control points;
   - the five sheets where the warp fell back to its "first" correction bend straight print by up to 12.6 ft (p90).

   The corner tracer reads 1R's single-line hatch as street channels, so 1R blocks give only corner slivers
   (`scratchpad/blocks1921/`; 25–53% of each checked sheet's map area traced).

   After the change:
   - **Start:** the rough position from text anchors (block numbers, street-name pairs; `rough`), as today.
   - **Placement:** the warp's line fit (`sheetwarp/probabilistic.py`, with `street_edges_and_later_blocks.py`).
     Today's block sides, and street edges where blocks are sparse, are fitted straight to the scan's printed lines, a
     point every 2 ft. That fit *is* the control points. No traced blocks are needed.
   - **Overlapping sheets:** a joint fit, so they line up.
   - **What goes away:** the corner-matching step and its tracer as a source of control points, and the "first"
     fallback.
   - **2000:** its placement changes too. The gate is that 2000 measures no worse than today on the 11-18 measures
     (agreement where scans overlap, straightness of printed lines).
3. **The city image** (solver `page`):
   - each sheet is cropped to **its printed frame** (`frame.find_frame`);
   - read through its corrected placement onto one city grid;
   - a grid px is ink where any sheet has ink there.

   There is no sheet-ownership rule: a blank corner never erases another sheet's ink, and a duplicate scan adds the
   same ink. **Fix the frame finder** so it finds every map's frame. Today it is written for the 2000 page (fixed
   search windows, the 2000 legend along the bottom, the 2000 frame size) and fails on all 16 sheets of 1921. The
   1921 frame is a thick bar, a thin rule 3–5 px inside it, and an inner border with rounded corners about 70–85 px
   further in, where the hatching stops. The 1921 legend is a box in a top corner. The neatline fallback
   (`sheetcrop.neatline_crop` for series outside `FRAME_SERIES`) goes away: every map's crop is its printed frame.
4. **Lots on the grid** (solver `lots`), at the city image's resolution.
5. **Reading.**
   - The 2000 map and the printed-boundary maps: printed codes and borders, as today (prints, text reader, labels,
     walls, solve).
   - The hatched maps: the hatching reader (below) on the city image, then each lot's district. There is no walls or
     solve stage.

**The published paper layer** takes the same frame crop as the city image.

**Scope** (owner, after the builder check: "Every map, 2000 paper too"):
- The frame finder is fixed for every series that uses the neatline fallback today: 1921, 1929, 1935, 1948, 1960,
  1964, 1968, 1972, 1975 and 1980.
- Every map's city image and paper layer use the printed frame. That includes 2000's published paper layer, which is a
  published 2000 change, checked by picture.
- 2000's solver crop stays byte-identical.
- Sheets with no printed frame (1948/7) or several frames on one photo (1948 "2-and-3") are handled explicitly by the
  finder, with pictures. There is no silent fallback.

**One scan per printed sheet** (owner):
- A fixed, hand-made list in `pipeline/inputs/` names the duplicate scans, found by looking. No duplicate detector is
  built.
- Duplicates are left out of every stage's input set.
- For 1921, Section 12 keeps 100525362 (2.72 ft/px). The list drops 094727628, the same image framed differently,
  and 094341084, a coarser re-scan at 4.56 ft/px.
- The other series' sheet lists are checked by eye for duplicates too.

**Seams** (owner: "Doesn't this get solved by fixing the warp and then stacking pixels?"). Yes. Fix the warp so the
sheets line up, then stack the pixels into the city image. Nothing is combined per sheet.

**Resolution** (agreed). The 2000 city grid is 4 ft/px. A hatched map's grid is at its finest scan's resolution:
2.72 ft/px for 1921, the Section 12 scans. So no sheet is shrunk. The step that reads sheets onto the grid stays the
existing one (the corrected placement at grid nodes, bilinear between, then the nearest page px), so 1-bit scans stay
1-bit. The owner: "Stop designing your own shitty upsampling and downsampling algorithms." No new resampling code is
written anywhere.

## Lots

Today's lots go on the city grid (solver `lots`), exactly as for 2000. AGENTS.md: borders come from modern lot lines,
never from tracing the scan. No blocks or streets are traced for reading.
- **Lot placement:** a lot landing in its own printed block is placement's job (step 1).
- **The learned street networks** (v1, v2a, v2c on branch `hatch-cnn`) are dropped. They memorised today's streets in
  their training areas wherever 1921 differs (`scratchpad/streetdbg2/findings.md`). Their code stays on the branch,
  unused.
- **Re-placing 1921:** this rewrites `warped.tif`, from which the pattern reader's cached reads were cut. The current
  reads are kept as a labelled copy of the training labels, and no paid re-read happens without the owner.

## The hatching reader (agreed)

What carries over from the 11-18 spec and the overnight runs:
- **The network:** a fully convolutional network on the city image's ink. It gives six district logits and one weight
  logit per px.
- **Lot pooling:** a lot's district probability is the weight-softmax pooling of its px's district probabilities on
  the city image.
- **Labels:**
  - the pattern reader's reads (`district_pattern.json`), through the read-error table T. T is fitted per fold by the
    empirical-Bayes Dirichlet prior (ruling R6), from the training areas' random-stratum answers only;
  - the owner's answers, taken as exact;
  - the legend swatches.
- **No LLM output at inference:** reads and answers only train and score.
- **Deterministic and $0:** one GPU job at a time; load kept around 20–60.

Changed (agreed):
- **Held-out areas:** scattered groups of blocks, not the four quarters. Quarters hid LI, two-thirds of which is in
  quarter ne, from the model that has to predict quarter ne. Each random lot is predicted by a model that never saw its
  area's reads or answers.
- **Rare districts:** the legend swatches get real weight (overnight they were 7–11 of 48,000 tiles). The training
  step is chosen by a held-out measure that counts each district equally. U was learned by no run, and the kept steps
  sat near the learning-rate peak.
- **The deliverable** is trained on all of 1921.
- **Later maps:** 1929, 1935 and 1948 run through the same process. The reader trained on 1921 reads them, for pictures
  only until they have answers.

## The score and done

- Each of the owner's 240 random 1921 lots is predicted by a model that never saw its area. Each is compared lot by
  lot with today's published map, with a block bootstrap (`answers1921`).
- **Done** means all of the following:
  - the paired interval against the published map excludes zero;
  - the owner has seen the pictures;
  - every sheet's fitted frame lines have been checked;
  - placement measures no worse than today on the 11-18 measures, on 1921 and on 2000.
- The owner's standing aim: "it's okay to just be better". Publish a map with fewer errors than today's.

## Constraints

- **What 2000 keeps:** 2000's solver crop stays byte-identical.
- **What 2000 may change:** its placement (the one placement process) and its paper layer (the frame crop) change by
  the owner's decision. Each 2000 change must measure no worse than today and is shown to the owner by picture before
  it is published.
- **No per-series fallback,** and no quietly different path for one map.
- **No hand-set cutoffs** in the reader: evidence as probabilities.
- **No learning from the reader's own predictions,** and no 2000 table codes in a hatched map's reading.
- **Data:**
  - `pipeline/data/` is never committed;
  - rebuild with `sfzmaps build` and commit `dvc.lock`;
  - no commits while a build runs;
  - pixi tools only;
  - no paid re-reads without the owner.
- **Pictures before building on any step:**
  - every sheet's fitted frame lines;
  - each placement against the scan's printed lines;
  - the city image's hatching at full resolution, next to the scan.

## Anti-goals

- **Reading per-sheet scans,** or any rule that decides which sheet owns a place.
- **Resampling code of our own.**
- **Learning streets or blocks from today's map,** or tracing blocks from the scan for reading.
- **A second, separate control-point process.**
- **Deferring a failing shared stage** behind a fallback.

## First checks

1. **Traced blocks on 1921:** done 11-19 (`scratchpad/blocks1921/`).
   - White, cross-hatch, grid and checkerboard blocks trace well.
   - 1R blocks give only corner slivers; 25–53% of each checked sheet's map area is traced.
   - This is why corner matching finds few control points on 1921. It leads to step 1's change.
2. **The frame finder on 1921:** done 11-19 (`scratchpad/frame1921/`). It finds 0 of 16 frames; the causes are listed
   under step 3.
3. **The line fit as the placement**, on every 1921 sheet and the 2000 sheets, before and after, by the 11-18
   measures, with pictures of the placed printed lines.
4. **The hatching on the full-resolution city image,** next to the scan, for 1R, LI, HI and U, once the city image is
   rebuilt.

## Left to the builder

- **The frame finder's method:** how it finds each side's innermost frame line relative to the page and fits it,
  handles rounded corners and the legend box, and checks sizes per map. 2000's crops must stay identical.
- **How the line fit becomes the one placement step,** and how the joint fit across overlapping sheets is built.
- **The network's form and training schedule** (engineering settings, stated with their reasons).
- **How block groups are drawn** for the held-out areas, and the per-district selection measure.
- **Display overrides** for water and parks: later, out of scope.
