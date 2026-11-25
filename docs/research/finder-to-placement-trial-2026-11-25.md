# The finder's blocks in placement: the trial (2026-11-25)

Step 3 of `docs/superpowers/specs/2026-11-24-finder-to-placement-design.md` (plan:
`docs/superpowers/plans/2026-11-24-finder-to-placement.md`). On four sheets, corner matching and the warp
were run unchanged, once from today's traced blocks and once from the finder's blocks (model run `all-pieces`), each
in a data folder of its own. Nothing else changed. $0: no label or page step ran; both took their saved answers.

Every figure here comes from `pipeline/data/find/trial/all-pieces/` (`numbers.json`, `compare.json`,
`blocks/<series>/<sheet>.json`, the arms' `work/` records and `logs/`), or from the two cause tests' scripts and logs
named below. Pictures are local and not in git: `pipeline/data/find/trial/all-pieces/look/<series>/`.

## In short

A sheet **placed** means it passed the pipeline's own check (`score.py`): the held-out corner error is within its
limits, and the sheet is not **misplaced**. "Misplaced" means almost none of its corners found a partner in today's
map, or its block numbers and street names lie off the blocks and streets they name.

- **Three sheets placed.** On Section 12 (1921 `100525362`) and on the control (2000 zoning p08) the finder's blocks
  gave many more control points; on downtown (1921 `094142210`) somewhat fewer. By eye the warped scans look the
  same as today's at every pictured place on those three sheets. The gain there is in control points and, on Section
  12, in held-out error; it is not yet a visible one.
- **One sheet failed.** 1921 `094348509` was misplaced (13 of its 837 good corners found a partner). What was
  tested: with the same label matches, the rough fit's result depends only on its scale hint. The finder's blocks
  give that hint (street width 13.3 px against the traced blocks' 16.8), so the blocks feed the failure through the
  hint. The true scale given as the hint also gives the false fit. From the traced arm's rough placement the
  finder's blocks place the sheet (633 control points against today's 79, passed); that is a separate check, kept
  apart below.
- **The finder's blocks lose ground.** Blocks are joined where the read breaks a street, and every area touching the
  frame cut's rim is dropped with all the blocks joined to it.

## What was compared

Per sheet: **blocks** (outlines in the blocks file); **good corners** (corners the grading keeps for matching);
**control points** (good corners matched to today's corners); the pipeline's own **check** (passed or failed, and
its **held-out error**: the distance in feet between held-out control points and where the placement puts them,
median and 95th percentile); **reach** (from points along the year's lots' block sides, placed by today's
placement, the distance in feet to the nearest control point: median and 90th percentile); the **warp's choice** of
correction; and **within 1 px** (the warp's share of block-side points lying within one px of a printed line).

The traced arm gave exactly today's figures on every sheet: for each of the 4 sheets, the transform, refine, score,
warp scores and correction records matched today's (20 comparisons, all "same"). So the "today" column is also the
traced arm.

| Sheet | Blocks | Good corners | Control points | Check | Held-out p50 / p95 ft | Reach p50 / p90 ft | Warp | Within 1 px |
|---|---|---|---|---|---|---|---|---|
| Section 12 `100525362`, today | 751 | 371 | 261 | passed | 5.22 / 13.21 | 427 / 1,133 | first | 0.603 |
| Section 12, finder | 374 | 904 | 669 | passed | 4.06 / 10.03 | 262 / 873 | first | 0.575 |
| `094348509`, today | 682 | 135 | 79 | passed | 7.11 / 13.46 | 1,088 / 2,664 | probabilistic | 0.500 |
| `094348509`, finder | 311 | 837 | 13 | **failed: misplaced** | 14.54 / 29.59 | 2,239 / 5,522 | first | 0.079 |
| `094142210` (downtown), today | 584 | 1,557 | 1,402 | passed | 5.78 / 11.92 | 108 / 311 | probabilistic | 0.462 |
| `094142210`, finder | 492 | 1,446 | 1,186 | passed | 6.29 / 13.38 | 121 / 384 | first | 0.426 |
| 2000 p08 (control), today | 292 | 556 | 492 | passed | 5.01 / 12.28 | 323 / 1,448 | probabilistic | 0.627 |
| 2000 p08, finder | 325 | 930 | 848 | passed | 5.44 / 11.48 | 232 / 1,501 | probabilistic | 0.628 |

No warp came out uncorrected: every sheet's correction was made ("first" or "probabilistic").

**Trained ground.** The share of each sheet's ground lying under the lot-labelled sheets the model trained on, and
the share of its control points on that ground:

| Sheet | Ground under training sheets | Control points on it, today | Control points on it, finder |
|---|---|---|---|
| Section 12 | 0.792 (and one hand training piece) | 0.912 | 0.834 |
| `094348509` | 0.027 | 0.025 | 0.308 (of 13) |
| `094142210` | 0.701 | 0.689 | 0.659 |
| 2000 p08 (control) | 0.111 | 0.140 (69 of 492) | 0.138 (117 of 848) |

The control's training ground is parts of five other sheets' footprints (2000 height p07 0.060, 2000 zoning p01
0.032, 2000 su p10 0.020, 2000 zoning p07 0.001, 1980-1990 p07 0.001); p08 itself was never trained on. The owner
kept it: "Keep sheet 8. It's fine." The 1921 sheets' trained ground is ground the model learned from later maps'
lots, never from a 1921 scan.

## The failure on 1921 `094348509`

What its records show (finder arm, against today's): refine ran 30 rounds without settling and its vote was
"ambiguous" (today: 13 rounds, settled, "ok"); the final scale is 6.43 ft per scan px (today 4.70); of 837 good
corners, 820 were marked "no equivalent" (no matching corner in today's map); the rough placement's scale hint was
4.82 ft/px (today 3.81).

**The cause, tested** (`pipeline/scripts/spikes/finder-to-placement-2026-11-25/rough_hint.py`). The rough placement uses
label matches only (block numbers and street names). It takes a scale hint from the blocks' typical street width and
accepts fits between the hint divided by 1.5 and the hint times 1.5 (`anchors.SCALE_HINT_FACTOR`). The test ran the
rough fit on the same 106 label candidates with only the hint changed:

- Traced blocks: street width 16.8 px, hint 3.81 ft/px, fit 4.90 ft/px, 0.4 degrees, 6 label matches agreeing: the
  right placement.
- Finder blocks: street width 13.3 px, hint 4.82 ft/px, fit 6.34 ft/px, -9.3 degrees, 7 label matches agreeing: a
  false one. A false group of 7 beat the right group of 6.
- Hints 4.5, 4.70 (the true scale), 4.82 and 5.2 all gave the false fit. Hints 3.0 and 4.2 gave no fit. With no hint
  the fit was 9.49 ft/px.

So today's pass on this sheet rests on a hint that is off the true scale but happens to shut the false group out. The
finder's blocks reach the failure only through the hint they give. The street widths (16.8 and 13.3 px) are in the
warp logs; the hints in the arms' `rough.json`; the rough fits per hint are as the script printed them (recorded in
the work log; the script kept no log file).

**A separate check from the right rough start** (`pipeline/scripts/spikes/finder-to-placement-2026-11-25/` (`make_root.py`, `mark_rough.py`, `proofs.py`, `snap.sh`);
its own folder `pipeline/data/find/trial/rough-check/finder`). Refine, score and warp were run from the finder's
blocks with the traced arm's rough placement copied in. Proofs (`proofs.py`): the blocks file's sha equals the
finder arm's; `rough.json` is byte-equal to the traced arm's; refine's record names both. Today's data was untouched
(`snap.sh`, before and after). This is not the trial: the trial's rule is that only the blocks differ, and here the rough start was
taken from the other arm.

| `094348509` | Blocks | Good corners | Control points | Check | Held-out p50 / p95 ft | Refine | Warp | Within 1 px |
|---|---|---|---|---|---|---|---|---|
| today (traced) | 682 | 135 | 79 | passed | 7.11 / 13.46 | 13 rounds, settled, ok | probabilistic | 0.500 (Mac GPU; not comparable) |
| finder blocks, right rough start | 311 | 837 | 633 | passed | 6.96 / 12.93 | 14 rounds, settled, ok | probabilistic | 0.535 (CPU; not comparable) |

The two within-1 figures came from different devices (mps, the Mac's GPU, and the CPU), so they are not like for
like. On the other three sheets the rough transforms are identical between the arms; only the hint differs (Section
12 2.71 against 3.19; downtown 4.78 against 4.45; p08 4.17 against 3.64 ft/px).

## What the finder's blocks lose

Counts from each sheet's blocks file. "Areas" are the connected pieces of ground the finder calls not street.

| Sheet | Finder areas | Cut at the rim | Cleaned away | Blocks kept | Areas holding 2+ of the year's blocks (blocks in them) |
|---|---|---|---|---|---|
| Section 12 | 853 | 68 | 411 | 374 | 47 (318) |
| `094348509` | 524 | 55 | 158 | 311 | 69 (394) |
| `094142210` | 648 | 35 | 121 | 492 | 30 (117) |
| 2000 p08 | 713 | 1 | 387 | 325 | 27 (269) |

- **Joined blocks.** Where the read breaks a street, one area holds several blocks; a joined outline gives no good
  corners.
- **Dropped at the rim.** An area touching the frame cut's rim gets no outline (the design's rule: an area touching
  the rim counts as cut by the page's edge). Open land, parks, water and the margin touch it, and so does every
  block joined to them. When the finder's blocks were shown to the owner before the trial, a scratch count
  (`pipeline/scripts/spikes/finder-to-placement-2026-11-25/lost_areas.py`) found that areas touching the rim held 41%
  (Section 12), 33%, 52% and 69% (p08) of the not-street ground. On p08 that is one area: the Bay, the margin,
  Mission Bay and the waterfront joined.
- **Cleaned away** is mostly specks ("too small"; 0.0 to 0.1% of the not-street ground in the same scratch count).
  Not all of it: one area "too ragged" on `094348509` (the ring of blocks round Urbano Drive), and one or two areas
  "taken for lettering" on every sheet, among them two blocks at 31st and 32nd on `094348509` and an area of
  17,542 px on p08.

**What became of each block** inside the frame cut:

| Sheet | Blocks counted | Its own outline | Joined with others | In an area cut at the rim | Dropped by cleaning | On street |
|---|---|---|---|---|---|---|
| Section 12 | 636 | 283 (44.5%) | 152 (23.9%) | 185 (29.1%) | 0 | 16 |
| `094348509` | 657 | 230 (35.0%) | 237 (36.1%) | 160 (24.4%) | 11 | 19 |
| `094142210` | 491 | 353 (71.9%) | 59 (12.0%) | 59 (12.0%) | 0 | 20 |
| 2000 p08 | 547 | 259 (47.3%) | 70 (12.8%) | 196 (35.8%) | 3 | 19 |

These count today's blocks: one point per block of the year's lots as today's parcel records give them, placed by
today's placement. They are not blocks read off the scan. On 1921 they include land that was open then and blocks
reshaped since. The owner, shown these counts: "You are dropping a lot of blocks, but maybe it's fine."

**Section 12's joined areas.** 47 areas hold two or more of the year's blocks, 318 of the 861 on the page. By eye in
`look/1921/20180102100525362__areas.small.png`: the largest joined area covers many rows of the diagonal grid beside
Mission Street, and most of the sheet's open land is red (cut at the rim), with the blocks standing in it. Today's
traced blocks against the finder's areas: 224 alone in one area, 485 sharing 144 areas (tracing's own fragments count
here too), 42 on street.

## The per-block confidence figure

Each block's properties hold four figures: the rim's px count, the street chance just outside the outline (10th and
50th percentile) and the block-edge chance on it (median). The street figure says little. An area is the px whose
street chance is at most one half, so the px just outside it are where the chance has only just passed one half.
Its low end therefore sits near one half on every block, whatever the read thought of the street there. Measured in
the four blocks files, the 10th-percentile figure has a median of 0.56 to 0.58 per sheet, and 90% of blocks lie
between 0.52 and 0.64 on the 1921 sheets (0.51 to 0.77 on p08). The read's confidence is its per-px chances, stored
with each sheet's read.

## What the pictures show

Eight pictures in `look/` (the places that coincide are written once): left placed from today's traced blocks,
right from the finder's; the scan warped onto today's map, today's block lines (DataSF) in magenta on top.
Each was compared by eye, not measured.

- **Section 12** `__worst.png` (= centre; Mission Street, the Alemany diagonals) and `__busiest.png` (the hill
  streets round Diamond, Chenery and Bosworth): the two panels look the same. On the printed grid the lines sit on
  the streets on both sides; across open land both draw today's blocks over ground printed as open.
- **`094142210`** `__worst.png` (worst = busiest = centre, downtown): the same on both sides; lines on the streets.
- **2000 p08** `__worst.png` (Mission Bay), `__busiest.png` (Potrero Hill, 17th to 24th) and `__centre.png` (the
  Central Basin): the same on both sides. At Mission Bay today's blocks differ from the printed proposed-use areas,
  in both panels alike.
- **`094348509`** `__worst.png` (St Francis Wood, Ocean Avenue) and `__busiest.png` (= centre), by eye: left, the
  lines sit on the printed streets. Right, the scan is turned and the printed streets cross the lines everywhere.
  The right panel is smaller. The panels are at each warped scan's own resolution, and the arms' `warped.tif` px
  are 6.43 ft (finder) and 4.70 ft (traced).

The debugging pictures shown to the owner before the trial, by eye: in
`look/1921/20180102100525362__outlines.small.png` and `look/1921/20180102094348509__outlines.small.png` the finder's
outlines lie on clean grids where tracing's are broken up on the hatching; in
`look/1921/20180102094142210__outlines.small.png` they match tracing's downtown, and tracing has outlines on the Bay
and the piers where the finder has none; the `__areas.small.png` pictures show the red areas named above.

## What is worse, looked into

- **`094348509` failed**: through the rough placement's scale hint, tested above.
- **Downtown (`094142210`)**: fewer control points (1,402 to 1,186), held-out error up (5.78 / 11.92 to 6.29 /
  13.38 ft), within 1 px down (0.462 to 0.426), and the warp chose "first" where today it chose "probabilistic".
  Its rough transform is identical in both arms. The blocks lost: 59 joined and 59 cut at the rim, and good corners
  fell from 1,557 to 1,446. Whether those lost blocks explain the higher error was not tested: unknown. Why the warp
  chose "first" was not looked into. The picture shows no difference.
- **Section 12, within 1 px** 0.603 to 0.575 while its held-out error fell. Not looked into; the pictures show no
  difference.
- **p08**: held-out median 5.01 to 5.44 ft (95th percentile 12.28 to 11.48) and reach p90 1,448 to 1,501 ft. Not
  looked into; the pictures show no difference.

## Not known, not done

- No other sheet was tried. Four sheets, one model run.
- The rough step was not changed (this plan changes no corner matching or warp).
- The within-1 figure of the rough-check run was computed on the CPU and today's on the GPU: not like for like.
- Whether the joins and the rim cut cost placement anything beyond the counts above is not measured.

## Decisions waiting for the owner

1. How the rough start should be made robust to a false group of label matches (it is outside this plan).
2. Whether to do anything about joined blocks and the rim cut, or leave them.
3. Whether to try more sheets, and which.
4. Pushing the branch.

## Reproducing

```
cd pipeline
# the finder arm (data/find/trial/all-pieces/logs/finder/run.sh): four sheets in parallel, then the warp
F=$PWD/data/find/trial/all-pieces/finder
SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year 1921 --sheet 20180102100525362
SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year 1921 --sheet 20180102094348509
SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year 1921 --sheet 20180102094142210
SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year 2000-digital-zoning --sheet 2000-digital-zoning-p08
SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps warp 1921 2000-digital-zoning
uv run sfzmaps find trial compare --trial all-pieces
```

Commits: the trial ran at 6656acf; the pictures were redrawn at 7839903 (`compare` run again, figures unchanged).
The two cause tests are scratch scripts outside the repository (paths above). $0.
