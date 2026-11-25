# The finder and the placement branch: the hand-off (2026-11-25)

Step 4 of `docs/superpowers/specs/2026-11-24-finder-to-placement-design.md` (plan Task 27). What the placement
branch's `place start` and `place fine` read from the scan today, where the finder's block-edge answer would take the
place of the printed lines, how a sheet's stored answers are read, and what has to be on one branch. Nothing was
changed on the placement branch.

Code cited from branch `placement-one-process`, a local branch, not pushed when this was written; the line numbers
are as of commit 0f9a1ab. Paths are under `pipeline/src/sfzmaps/`. Line numbers checked against that code on
2026-11-25.

## What the two steps read from the scan

Both read the page through the frame step's files, on the grid of `data/work/<series>/<sheet>/page.png`:
`place/inputs.py` `erased_page` (line 206; `data/frame/<series>/<sheet>/erased.png`) and `frame_cut` (line 200;
`outside.png`, True where the page is kept). `_frame_image` (line 183) refuses either if its shape is not
`page.png`'s.

**`place start`** (`place/start.py` `find_starts`, line 863):

- The label reads of `data/work/<series>/<sheet>/labels.json` (`place/inputs.py` `reads`, line 405): block numbers
  matched to the year's lots, street names to today's street lines.
- The page's thin ink, once per sheet (`place/start.py` `_ink`, lines 512-519): the erased page, each px's chance of
  ink (`place/inputs.py` `ink_probability`, line 62), then `strokes` (`place/inputs.py` lines 159-174): the skeleton
  of `p_ink > 0.5` inside the frame cut (`sheetwarp/targets.py` `ink_centres`), each centre's direction and half
  width, and its chance of being a thin block line (`probabilistic.stroke_width_groups`).
- `_Ink` (`place/start.py` lines 503-509) holds `x, y` (centre px), `phi` (direction, radians 0 to pi), `w` (thin
  chance times ink chance, centres with `w > 0` only) and `keep` (the frame cut).

**`place fine`** (`place/finefit.py` `fine_from_start`, line 346):

- `PlacedSheet` (`place/finefit.py` line 138): `page` (the erased page), `keep` (the frame cut), `ink` (grey below
  `measures.INK_BELOW`, 128, on kept paper), and `block_line_half_width` (line 183: the widest half width the thin
  group of stroke widths explains, from `ink_centres` over the kept paper).
- `sheetwarp/targets.py` `fit_targets` (line 145): `ink_centres` (line 116: skeleton, half width, direction) on a
  crop round today's block lines, then `direction_maps` (line 131): per direction (18, every 10 degrees), each px's
  distance to the nearest centre running within 20 degrees of it whose `half_width <= block_line_half_width`.
- `sheetwarp/probabilistic.py` `block_line_weights` (line 133): each centre px's weight is `thin_share`, its chance
  of the thin group of stroke widths (`stroke_width_groups`), per direction.
- Not from the scan: the block-side points and the street width come from today's lots and streets
  (`targets.block_side_points`, `finefit.street_width_px`).

## Where the finder's answer would take the place of the printed lines

The finder's block-edge chance is a per-px chance that a block's edge lies there. Its thinned centre line is the
nearest thing it gives to a printed line's centre.

- **For `place start`:** an `_Ink` whose `x, y` are the px of the thinned centre line of the edge chance (inside the
  frame cut), `w` the edge chance there, `phi` the direction there, and `keep` the frame cut. `_Ink` is the only
  thing `_ink` hands on, so the change is inside that one function.
- **For `place fine`:** an `InkCentres` built from the edge chance's centre line (skeleton, half width, direction),
  handed to `direction_maps`, and the edge chance in place of `thin_share` as each centre's weight in
  `block_line_weights`.
- `direction_maps` keeps a centre only where `half_width <= block_line_half_width`, and `block_line_half_width` is
  measured on the printed ink. With the finder's centre line there is no ink width for that cut to measure. What
  takes its place is not decided.
- No offset or scale: the finder reads the same `page.png` grid (`find/pages.py` `load_scan`, and `load_keep` reads
  the same `outside.png` and refuses another shape).

## Reading a sheet's stored answers

```
from sfzmaps.find import read, wholesheet
chances = read.load(wholesheet.paths(run_folder, (series, sheet))[0])
```

- `wholesheet.paths(run_folder, sid)` gives `<run_folder>/sheets/<series>/<sheet>.npz` and its `.png` picture.
- `read.load` gives float32 `(2, h, w)` in the page's own px: `[0]` the street chance, `[1]` the block-edge chance
  (`find.model.ANSWERS = ("street", "edge")`). Stored as levels round(255 p), so 256 values; 0 off the frame cut.
- Made by `sfzmaps find read --run all-pieces --series S` (or `--sheet <series>/<sheet>`, repeatable), from
  `data/find/runs/<run>/model.pt`. Duplicate scans are refused and named. It runs on the GPU (mps) by default.
- Read today with run `all-pieces`: 1921 `094142210`, `094149543`, `094348509`, `100525362`; 1960 `1`, `7`;
  2000 zoning p07, p08.

## The per-block confidence fields in the blocks file

`sfzmaps find blocks` writes `blocks_px.geojson` in the form the corner matching reads (outlines in page px, the
good corners as an N x 2 list). Each outline's `properties` hold `area` (the area's id) and four figures
(`find/placeblocks.py` `outline_confidence`):

- `outline_px`: the count of the area's rim px;
- `outline_street_p10`, `outline_street_p50`: the 10th and 50th percentile of the street chance on the px just
  outside the rim, inside the frame cut;
- `outline_edge_p50`: the median block-edge chance on the rim px.

The street figures sit near one half on every block by construction (an area ends where the street chance passes one
half). On the four trial sheets the 10th-percentile figure has a median of 0.56 to 0.58. Refine and the warp do not
read `properties` (checked on this branch: the only readers of the blocks file in placement take `corners` for refine,
`runner.py` lines 605-608, and the outlines' geometry for the street width, `runner.py` lines 552-553,
`sheetwarp/first.py` lines 154-155 and `sheetwarp/probabilistic.py` lines 84-85). Per-corner confidence is not kept.
The read's own confidence is the per-px chances above.

## What has to be on one branch

- `sfzmaps/find/` (the finder; kept out of the build stages' dependencies by `build.CODE_EXCLUDE`).
- **`sfzmaps/solver/hatching.py` and `sfzmaps/solver/hatchnet/`**: `find` imports both (`find/model.py` imports
  `solver.hatchnet.model`; `find/wholesheet.py` and `find/pages.py` import `solver.hatching` for the duplicate
  scans). Neither exists on `placement-one-process`. (Since this note: the few pieces `find` used were moved into
  `find/` on branch `street-block-finder`, which has neither file: `find/pagegrid.py`, `find/streetlabels.py`,
  `find/cldice.py`, `find/device.py`, `find/model.py` and `find/pages.py`.)
- `pipeline/inputs/find/` and `pipeline/inputs/find-labels/` (the finder's label sources and hand pieces).
- `pipeline/inputs/<series>/duplicate_scans.csv`, which the finder reads through `solver/hatching.py`
  `duplicate_scans` (line 733 on this branch; moved to `find/pages.py`) to refuse duplicate scans: on this branch for 1921, 1929, 1935, 1975,
  1980 and 1996-2000-su. Commit 243af8c added 1921's; none is on `placement-one-process`. That branch has its own
  reader of the same file name (`place/inputs.py` `duplicate_scans`, line 211), which finds no file there today.
- The run folder `pipeline/data/find/runs/all-pieces/` (`model.pt`, `log.json`; also `sheets/` with the stored
  reads). It is under `pipeline/data/`, not in git.

## The trial's result, in brief

1. Section 12 and the 2000 p08 control placed from the finder's blocks with more control points (261 to 669; 492 to
   848); downtown with fewer (1,402 to 1,186). By eye the warped scans look the same as today's.
2. 1921 `094348509` failed: the rough label step's scale hint, which comes from the blocks' street width and allows
   fits from the hint / 1.5 to the hint x 1.5 (`anchors.SCALE_HINT_FACTOR`), let a false group of 7 label matches
   win. Tested by a one-off check kept in the repository,
   `pipeline/scripts/spikes/finder-to-placement-2026-11-25/rough_hint.py`. A separate check that used the traced arm's rough
   placement, so not like for like with the trial, gave 633 control points from the same finder blocks against
   today's 79, and passed.
3. The finder's blocks join where the read breaks a street and are dropped where they touch the frame cut's rim.
   Details: `docs/research/finder-to-placement-trial-2026-11-25.md`.
