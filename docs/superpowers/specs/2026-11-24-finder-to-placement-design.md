# From the finder to placement: the smallest path

**Status:** draft for the owner's approval, written 2026-11-24 from that day's conversation and a code audit of the
placement steps (`placement-audit-2026-11-24.md`, kept with the work log). Nothing in it is built. Lines marked
*(my call)* are the writer's, not the owner's, and can each be changed.

## Intent and why

The old maps place badly because the first placement step cannot read them. That step traces the printed blocks off
the scan; the next step matches the traced blocks' corners to today's block corners, and those matches are the
control points the warp starts from. On hatched maps the tracing breaks, so there are few control points and the
warp drifts between them.

The owner, 2026-11-24: "We needed this branch because we couldn't reliably detect the street/block grid on old maps,
app couldn't find control points, so couldn't warp correctly, right?" Yes. The finder built on this branch now finds
streets on 95 to 99% of street ground in every era and draws block edges on real block boundary
(`docs/research/street-and-block-finder-first-chain-2026-11-23.md`).

This stage gives placement the finder's blocks in place of the traced ones and changes nothing else, to see whether
the old maps then get the control points they lacked. The owner: "let's do the smallest viable path now and then
prepare for the handoff to the placement branch."

## Goals

1. Any sheet can be read whole by the finder, with a picture of the read.
2. A sheet's found blocks can be written in the form the corner matching reads.
3. On four trial sheets, corner matching and the warp run unchanged from the finder's blocks, in a folder of their
   own, and the result is compared with today's: control points, and the scan warped onto today's map.
4. The finder learns from every lot-labelled sheet, and a lot-labelled sheet in the trial is read by a model that
   did not learn from that sheet's ground.
5. What the placement branch will need from the finder is written down, and the files it needs can be made.

## Not in this stage

- Any change to corner matching or to the warp, including their existing hand-set numbers.
- The placement branch's own route (block edges fed into the line fit, no control points). Prepared for, not built.
- Placing every sheet of every series, or replacing any placement the pipeline uses today.
- Build stages (`sfzmaps build`), publishing, the site.

## What would make it a failure even if it runs

- The trial overwrites today's placements. The finder's training labels and its held-out ground are laid through
  them, and other code reads the traced blocks.
- A lot-labelled sheet is judged from a read by a model that learned that ground's lots.
- A new hand-picked number decides anything.
- Results are shown as today's map drawn on the scan. Placement is judged on the scan warped onto today's map, with
  today's lines thin on top.

## Constraints

The standing rules of this code: the finder is shown only the scan and the frame cut; no LLM output is read
(`SFZMAPS_MAX_USD=0`; the label step in the trial must use its saved answers and call nothing); $0; deterministic;
duplicate scans never read; nothing under `pipeline/data/` committed; no `sfzmaps build`; one way for every series;
"measurements and probabilistic learned behavior, not fixed thresholds and hand tuned limits".

## The ways considered

- **A. The finder's blocks into the existing corner matching, then the existing warp.** One seam: where the traced
  blocks are written today. Everything after it is untouched. It tests directly the reason this branch exists.
- **B. The finder's block edges into the warp's line fit, with no control points.** This is the placement branch's
  route (`place start`, `place fine`). It removes the corner matching and its hand-set numbers, but that branch is
  unpushed and its start step has open problems on 1921.
- **C. Both at once.** Two changes whose effects could not be told apart.

**Picked: A, then prepare for B** (the owner, above).

## What the audit found

- Corner matching reads only a list of graded corners. The warp and the rough placement read the block outlines for
  one thing, the typical street width. So the seam is one place: the blocks file of a sheet
  (`blocks_px.geojson`: outlines in page px and the corners graded good).
- The tracing step does four things to its blocks that the finder's areas do not get by themselves: it drops blocks
  cut by the page's edge; it cleans and simplifies each outline and restores its corners; it puts the outline on the
  centre of the printed line; and its blocks are separate across streets. Its corner grading marks down outlines
  with many corners, so two blocks joined into one area give no good corners.
- A separate data folder (`SFZMAPS_DATA_DIR`) keeps a trial away from today's placements. The warp command takes a
  whole series, so the trial folder must hold only the trial sheets.
- Today's numbers (blocks traced, corners graded good, control points): Section 12 of 1921: 751, 371, 261. 1921
  sheet 094348509: 682, 135, 79. 1921 sheet 094142210: 584, 1,557, 1,402. 2000 zoning sheet 7: 597, 1,167, 1,011.
  On Section 12, 8.7% of traced corners grade good.
- The pipeline's own check passes every automatic 1921 sheet. "Places badly" is by other measures (issue #69):
  error grows with distance from the nearest control point, and overlapping scans disagree.

## The design, in stages, each shown before the next

### 0. The trial folder, proved first

- Before anything of the finder is needed: a trial data folder (`SFZMAPS_DATA_DIR`) holding only the four trial
  sheets' existing work files, with the saved page and label answers so nothing is called and nothing is paid. In
  it corner matching and the warp are run from the traced blocks as they are today. They must reproduce today's
  control points and warp scores. If they do not, the reason is found before the trial means anything.
- *(my call)* This comes first because it needs nothing from the later stages and it is where a trial most easily
  goes wrong for reasons that have nothing to do with the finder.

### 1. Learn from every lot-labelled sheet

- Labels are made, by the existing command, for every sheet of the twelve series the lots label (about 150 sheets),
  under the rule decided on 2026-11-24 (ground with no lot is street only between blocks).
- *(my call)* "Every" means every sheet whose placement passes the warp's own check: the lots are laid on a scan
  through its placement, and a label laid through a placement the warp itself doubts may not match the print. Five
  lot-labelled sheets fail it today; they are left out and named. The same rule already picks the sheets whose print
  sizes are learned.
- Two models are trained with the current recipe:
  - **all**: every lot-labelled sheet and the thirteen hand training pieces. It reads the old maps.
  - **all but sheet 7's ground**: the same, with the ground of 2000 zoning sheet 7 left out of every sheet of every
    series, by the mechanism that already keeps test ground out, used as it is (the ground is carried from sheet to
    sheet through today's placements). It reads the trial's control sheet.
- **What the model that reads the 1921 trial sheets has seen.** No 1921 sheet has lot labels. The "all" model has
  learned the 2000-era lots of the same ground from the 1980s-2000 scans, never from a 1921 scan; it is shown only
  the scan and is not told where a sheet lies. Of the three 1921 trial sheets, two carry no hand-marked piece at
  all; Section 12's sheet carries one training piece (well away from Section 12) and the Section 12 test piece,
  whose ground is held out. This is allowed, and said here so the comparison is read knowing it.
- *(my call)* Two models now, not leave-out rounds for every lot-labelled sheet. Only one lot-labelled sheet is in
  the trial. The fuller option, a round for each part of the city, is needed when every lot-labelled sheet is placed
  from its read, and belongs with that work.
- *(my call)* About 150 sheets do not fit in memory the way six do (64 GB). Sheets are kept on disk and read as
  tiles are drawn. An engineering change; a tile drawn from disk must equal the tile drawn from memory.
- Shown: the same measures as the first chain on the 14 test and 8 choosing pieces, beside the current recipe's; the
  Embarcadero read again (2000 zoning sheet 1 is now a training sheet, so this picture is of learned ground and is
  only a check that the labels took).

### 2. Read whole sheets

- `sfzmaps find read`: a run and a series (or one sheet) in; per sheet, the two answers at the page's own size, as
  reads are stored today, and a picture: the scan with street tinted and block edges drawn.
- The finder reads the same page image the placement steps use (`data/work/<series>/<sheet>/page.png`), and the
  frame cut is on that same grid, so a read's px are the blocks file's px. No offset and no scale.
- The reader has only ever been run on small pieces. A test and a look on a real page check that the joins between
  the squares it reads in do not show.
- Shown: the four trial sheets' pictures.

### 3. Blocks for the corner matching

- A sheet's areas (the connected pieces of ground the finder calls not street) go through the tracing step's own
  outline code in place of that step's own pieces: drop the ones cut by the page's edge, clean, simplify, restore
  corners, grade corners. The result is written as that sheet's blocks file in the trial folder.
- *(my call)* Which areas are "cut by the page's edge": those touching the rim of the frame cut. The tracing step
  uses a mask made from the saved page classification, which is a vision model's answer; the finder's side reads no
  such answer. The traced run keeps its own mask, as today.
- *(my call)* The tracing step's existing numbers for cleaning and simplifying apply as they are. They are hand-set,
  they are not new, and changing them is outside this stage. The simpler-looking option, writing the areas' raw
  outlines, would not give the corner grading what it expects.
- **Where the outline sits.** Whether the finder's areas run to the centre of the printed line or to its outer edge
  is measured, per era, against the traced outlines on a sheet where tracing works. If they run to the outer edge
  they are brought in by half the line's measured width, as the tracing step already does for outlined blocks.
- **Joined blocks are counted, not worked around.** An area holding two blocks gives no good corners. On the test
  pieces plain pieces join few blocks on most maps but many where blocks stand in open land (Section 12's test
  piece: 27 of 34), because nothing but a street separates areas. This is the known weak point of the trial.
- Shown: for each trial sheet, the outlines and graded corners drawn on the scan (a debugging picture, labelled so),
  beside the traced ones; the counts of blocks, good corners and joined areas.

### 4. The trial

- The four sheets of the trial folder: **Section 12** (1921 `100525362`), **1921 `094348509`** (fewest control
  points today), **1921 `094142210`** (most), **2000 zoning sheet 7** (the control).
- In that folder corner matching and the warp are run from the finder's blocks, beside stage 0's run from the
  traced blocks.
- **No pass line.** The trial reports; it does not stop itself on a number. What is hoped for is more control
  points, better spread, on the 1921 sheets and no loss on sheet 7; whatever comes out is measured and shown, and
  what is worse is looked into.
- Compared, per sheet: blocks, good corners, control points; how far points along today's block sides lie from the
  nearest control point (median and the distance holding 90% of them); the existing check's held-out error; which
  correction the warp chose and its share of block-side points within one px of the print.
- Pictures: the scan warped onto today's map with today's block lines thin on top, before and after, at the same
  three places on each sheet, one of them where today's placement is worst.
- The owner looks before anything further.

### 5. Prepare the hand-off

- A note in the repository: what `place start` and `place fine` read from the scan today, and where the finder's
  block-edge answer would take the place of raw printed lines (a thinned centre line with a direction and a
  weight); the function that reads a sheet's stored answers; what has to be on one branch for it to work.
- The per-sheet reads from stage 2 are the files that route needs. No change is made on the placement branch.

## What is left to the builder

Names of functions and options; how sheets are kept on disk; the exact layout of the trial folder and how the four
sheets' existing work files are copied or linked into it; how the debugging pictures are drawn; test cases, within
the rule that each new behaviour gets one with a known answer.

## Calls that are mine

1. Two models now, not a leave-out round for every part of the city.
2. Sheets kept on disk for training.
3. The tracing step's own outline code and its existing numbers are reused for the finder's areas.
4. The four trial sheets.
5. Joined blocks are counted and shown, not split by a new rule.
6. One training of each model (one seed). The first chain measured that streets and block edges barely move between
   seeds.
7. The trial folder is proved on today's blocks before anything else.
8. A lot-labelled sheet whose placement fails the warp's own check is left out of training.
9. Edge-cut areas are those touching the frame cut's rim.
10. The trial has no pass line.

## Risks

- **Section 12 may gain little.** Its blocks stand in open land, where areas join. If so the trial says so with
  numbers, and the remedy (the block-edge answer, or the placement branch's route) is a later decision.
- **The outline code's numbers were set for the tracing step's own pieces** and may treat the finder's areas badly.
  The debugging pictures of stage 3 are there to catch it.
- **Learning from 150 sheets is a larger change than the trial needs.** The owner chose it; its effect is read
  separately in stage 1, before the trial.
- **The control sheet's left-out ground** is the ground of one sheet through today's placement. A wrong placement
  would leave the wrong ground out; sheet 7 passes the warp's own check.
