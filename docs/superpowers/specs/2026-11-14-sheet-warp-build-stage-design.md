# Sheet warp as a build stage

2026-11-14. Branch `prob-warp`, PR #58.

## Why

Each scanned sheet is placed on the city by a fitted placement (`transform.json`). Small errors remain: printed
block lines miss the real ones by a few pixels. The sheet warp corrects each sheet's placement so its printed
block lines land on the block lines of its year, and warps the sheet with the correction. The owner approved its
results on 2026-11-09.

Today it is a side project: 13 step scripts under `pipeline/scripts/sheet_warp/`, driven by `warp.py`, with the
two PyTorch fits in a second Python environment, the correction applied by a `sitecustomize` patch, and the
corrected sheets written to a "stand-in" copy of the data folder. That shape is an accident of how it was rescued
from R&D. The owner wants it to be a normal part of the project:

- **Rewarping is routine.** Anyone working on the project will need to rewarp (after a placement changes, a scan is
  added, or the fit improves). They run `warp`, optionally with one set of scans (a series, e.g. `1921`,
  `2000-digital-zoning`).
- **`sfzmaps build` is the dependency chain.** It checks whether each stage is stale and reruns only what is: text
  detection, the warp, and the solve. This design makes the warp one of those stages and sets the pattern the
  other two follow.
- **The corrected sheets are the sheets.** Whatever comes after the warp (the text detector, the joint solver,
  anything new) reads the corrected placement and warped sheet. There is no side copy.
- **The old pipeline is deprecated.** `sfzmaps`'s old district reader, and the maps it publishes, are being
  replaced part by part. Effects of this work on the old way (for example paid re-reads the old reader would need
  after a rewarp) don't count against it.

## Goals

1. `sfzmaps warp [SERIES]` corrects and warps every placed sheet of a series (default: every series), skipping a
   sheet whose inputs haven't changed since it was last done.
2. One build stage per series, `warp-<series>`, in `sfzmaps build`'s stage table and the generated `dvc.yaml`, so
   `sfzmaps build` reruns a series' warp when it is stale, records it in `dvc.lock`, and `sfzmaps pull` brings the
   results to another machine instead of refitting.
3. The warp's code is part of the `sfzmaps` package: plain functions in one place, no step scripts, no subprocesses
   between steps, no environment variables carrying the series and sheet, no separate environment.
4. PyTorch is a regular dependency in `pipeline/pyproject.toml` (not optional, not a group): "we'll probably use
   PyTorch or something else too".
5. The placement code (`sfzmaps.transform.SheetTransform`) applies a correction itself when a placement carries
   one. The `sitecustomize` patch goes.
6. Every placed sheet is corrected: 357 sheets across all 22 series (today: 174 in 13 series), including the
   1980–2000 height and special-use series and the 1996–2000 zoning scan. "Placed" = the sheet folder in
   `data/work/<series>/` has `page.png`, `transform.json`, `refine.json`, `blocks_px.geojson` and `warped.tif`
   (and so `warped.footprint.geojson`). The sheet list no longer comes from the old reader's label roles.
7. Names say what things are. No `p1`, `prob6`, `reg2`, `confit`, `plus`. (The owner: "make sure each file has a
   HUMAN-READABLE name.")

## Non-goals

- Making text detection or the solve into stages. They slot into the same chain later, by their own work.
- Replacing the old pipeline's placement (`sfzmaps run`): the warp still starts from its `transform.json`.
- Publishing tiles or district maps from the corrected sheets.
- Changing the joint solver (#57). It currently reads its own copy of 2000 placements (`SFZ_RND/data_ay2`), so this
  work doesn't break it; moving it onto `data/warp/` is its own change.

## Anti-goals (would make this a failure even if it works)

- A side copy of the data (the stand-in folder) instead of the corrected sheets being the sheets.
- A second Python environment, a monkeypatch, or another orchestrator (shell scripts, a Makefile) beside
  `sfzmaps build`.
- Any call to a paid model. The warp needs none.
- Writing into another stage's outputs (`data/work`, `data/out`).
- Committing anything under `pipeline/data/`.

## What the warp does (the method stays as it is)

Per sheet, unchanged from the approved method except where this section says so. The current scripts in
`pipeline/scripts/sheet_warp/steps/` are the reference implementation; each was cleaned up on 2026-11-14 and
checked to give byte-identical results to the approved run's rebuild.

1. **What to aim at.** Points every 2 ft along each side of each block of the sheet's year, inside its drawn map
   (the old warp's footprint less the Presidio), placed by today's placement. On the scan, the centre line of every
   stroke no wider than a block line, with its direction; per direction (18, every 10°) a distance map. A point fits
   when it is within 1 px of a printed line running its way.
2. **The first correction.** Sides the print already holds stay put; every other side moves onto the nearest
   printed line running its way, carried outward from the held sides; a smooth field (grid every 50 px) is fitted
   to that (PyTorch).
3. **The probabilistic correction.** Each point either lies on a printed block line (a Gaussian across the line,
   ink weighted by how likely its stroke width says it is a block line) or isn't printed; the smooth field that
   makes all points most probable, with a stretch prior and a fold penalty, blurred to half a street width at first
   and sharpened in stages (PyTorch).
4. **Street edges and later blocks** (formerly "extra evidence", today used on 2000 p09 and p10 only): the
   probabilistic correction also fitted to every street's two edges and to the outlines of blocks recorded after the
   map's year, each added point counted by how little the year's own blocks already explain it. See "The street
   edges and later blocks test" below.
5. **The choice.** The probabilistic correction where it puts at least as many block-side points within 1 px of the
   print as the first and its smallest cell determinant is above 0 (no fold); otherwise the first. There is no fallback to
   today's placement when it scores higher: today's placements aren't known to be right (the owner: "we don't know
   that today's fits are optimal ... earlier re-warp work produced a better map"), so the warp doesn't treat them as
   a floor.
6. **The warp.** The scan resampled by the chosen correction, then warped exactly as the old pipeline warps it
   (`sfzmaps.warp.warp_sheet`, today's control-point grid, gdalwarp), plain and paper; the footprint traced on the
   page and taken to the map with the corrected placement.

Details that must be kept, because each changes the fitted numbers:

- Block-side points are placed with one `to_pixel` call per array (points, tangent ends) on the whole array:
  `to_pixel` stops on the batch's largest step, so other batching changes the values.
- The block-side area cuts the Presidio (block 1300, buffered 8 ft) out of the footprint twice; the street-edges
  points cut it once. Each cut rounds the outline slightly differently. (Checked on all 236 sheets of the 13
  series: one cut instead of two changes every sheet.)
- The fit crop's extent comes from candidate extreme points placed exactly (`fit_targets.crop`).
- PyTorch threads: 8 for the first correction's fit, 3 for the probabilistic one.
- The general rule for 2000 (footprint area, measured scale), as of 2026-11-14: the 2000-only setup the approved
  run used (its own measured area, a fixed 3.99 ft/px) is gone. Scored on the same points, the old and new 2000
  corrections are equal (mean within 1 px 0.749 vs 0.748; within 2 px 0.812 both).

## Where things go

- **Code:** a subpackage of `sfzmaps` (name and module split are the builder's call; names say what they do).
  `sfzmaps/warp.py` already exists (the old pipeline's gdalwarp wrapper) and stays.
- **Command:** `sfzmaps warp [SERIES ...]` in `sfzmaps/cli.py`; `sfzmaps stage warp-<series>` runs the same thing for
  the build.
- **Outputs:** `pipeline/data/warp/<series>/<sheet>/`, one folder per sheet, written only by the warp stage:
  - `transform.json`: today's placement plus `"correction": {"file": "correction.npz"}` (relative to the folder).
    This is the corrected placement everything downstream loads with `SheetTransform.from_json`.
  - `correction.npz`: the chosen correction (`Gc`, `spacing`, `origin`), and which correction it is.
  - `warped.tif`, `warped_paper.tif` and their footprints.
  - `scores.json`: the scores the choice used (within 1 and 2 px for today / first / probabilistic, control points'
    miss, smallest cell, largest correction) and the choice.
  - Whatever else is needed to skip unchanged sheets. Intermediate arrays don't need saving; the old scripts saved
    them only to hand them from one process to the next.
- **The stand-in folder, `correction_patch/`, `fits_environment/`, `scripts/sheet_warp/`** are removed. The README's
  explanation moves to wherever the builder documents the stage (the pipeline README, or a docstring the README
  points to).

## Sheets the warp can't fit, and failed placements

*(My calls, after review.)* `data/warp` promises every placed sheet, so no sheet is left out:

- **A sheet that can't be fitted** (no block-side point lands on the page, every side already held, any other step
  that has nothing to work on) is written with today's placement uncorrected and warped as today, and
  `scores.json` says why. One such sheet doesn't fail the stage. The current step code fails on some of these cases
  (`propagate_shifts` takes `.max()` of an empty array when every side is held; the crop fails with no points on the
  page); the port handles them.
- **Sheets whose old placement check failed** (`score.json` `passed: false`; 12 of the 357, among them 1929 p14,
  1935 p10 and 1980–1990 special-use p02 and p05, "misplaced") are corrected like any other. The warp doesn't judge
  placements; fixing a wrong placement is the placement step's job.
  `scores.json` carries the old check's result (passed, reason) so whatever reads `data/warp` can see it.
- **Accepting the 183 sheets never fitted before:** their scores, and the seams by eye.

## The build stage

- One stage per series: `warp-<series>`. Upstream: the old stage that makes `data/work/<series>`
  (`edition-<year>` or `period-<period>`) and `refdata`. Outputs: `data/warp/<series>` (persist, like every
  stage).
- Inputs it reads: `data/work/<series>/<sheet>/` (`page.png`, `transform.json`, `refine.json`, `blocks_px.geojson`,
  `warped.footprint.geojson`, `page.json`), `data/ref/parcels.parquet` and the lot history computed from it,
  `data/ref/streets.parquet`, gdal from the pixi environment.
- The warp reads the old pipeline's uncorrected placement and footprint, never its own outputs, so rerunning it
  gives the same result rather than correcting a correction.
- Inside a stage, a sheet whose inputs are unchanged since its outputs were made is not refitted (a fit takes
  minutes per sheet). How that is recorded is the builder's call.
- Fitting is parallel across sheets; the two fits' thread counts above are kept.
- **The old stages don't run again** (owner: "don't waste time re-running the whole pipeline ... no need to waste
  cpu cycles on things getting replaced"). They list every file in `src/sfzmaps/` (except `build.CODE_EXCLUDE`) and
  `uv.lock` as dependencies, so adding PyTorch and the warp code would make every one of them stale. The old
  pipeline's stages (`edition-*`, `period-*`, `today`, `shared`) are therefore frozen, as `refdata` and
  `saved-reads` already are: `sfzmaps build` never runs them, `build --force NAME` still can. Their recorded outputs
  stay what `sfzmaps pull` brings and what the warp reads.
- **No money is spent.** The warp calls no model; every stage runs with `SFZMAPS_MAX_USD=0`.

## The street edges and later blocks test

Today this fit runs on 2000 p09 and p10 only, chosen by name. The owner: the problem it solves is general, not
2000's. Maps of every year print things the year's lot records lack, among them streets and lots planned under water
for landfill that never happened.

**What the data holds** (checked 2026-11-14):
- DataSF's street centre lines (`3psu-pn9h`, the dataset `refdata` already fetches) mark each segment's `layer`,
  which the pipeline doesn't fetch yet. Two layers are unbuilt streets: `PAPER` (128 segments, about 6 mi) and
  `PAPER_WATER` (149 segments, about 12.6 mi, under water: mostly Bayview and Hunters Point, some Potrero Hill and
  Mission Bay). Other layers include `FREEWAYS` (28 retired segments) and `STREETS_HUNTERSP` (110 retired).
- Its history starts in 1998: every `date_added` is 1998 or later, and the 798 retired segments were retired since
  then. A street that moved or closed earlier isn't in it.
- No historic street dataset by year was found (OpenHistoricalMap has partial volunteer coverage; MIT's 2000
  snapshot is restricted). Lot retirements are recorded from 1956.

**The risk.** AGENTS.md: "Where a street really moved, keep the historic position; don't warp the scan to fit
today's streets." Each added point counts by how far it is from the year's own block outlines, so where a built
street moved, today's edges count in full and pull the print toward today's street. Unbuilt and underwater streets
carry no such risk: they were planned and never built. The block-side score can't see this, because the added
evidence acts only where block sides are absent.

**The test** (agreed with the owner):
1. Fetch the street `layer` (with `accepted` and the dates), so unbuilt and underwater streets are marked. Get it
   without fetching the rest of `data/ref` again: a fresh fetch of the lots changes every fit.
2. Fit the probabilistic correction with street edges and later blocks on every placed sheet.
3. Measure, per sheet, against the plain probabilistic correction:
   - the block-side score (within 1 and 2 px), exactly as now;
   - how far the correction moves the print where only the added evidence acts (where it differs from the plain
     fit, and by how much), split by kind: unbuilt or underwater streets, other streets, later blocks;
   - pictures, each saying what it shows, at places where a built street is known to have changed: retired DataSF
     segments (removed freeways, the Hunters Point shipyard and the like) and any place the owner names.
4. Bring the numbers and pictures to the owner, who decides: everywhere, by a rule (not a list of sheet names), or
   not at all.

Until the owner decides, the port keeps today's behaviour (p09, p10 only) so the port can be checked on its own.

## How we know it works

1. **The port changes nothing.** For every sheet the approved run's rebuild covers (174 sheets; reference outputs
   in the `cache-maybe` worktree, `pipeline/data/prob-warp/rebuild-2026-11-13/run-notext/`), the new code's first
   and probabilistic corrections are byte-identical (`Gc` arrays) to the current scripts' (1921–1990) or equal to
   the 2026-11-14 run of the current scripts (2000). Then the test above runs as a separate step.
2. **Unit tests** for what is ours: `SheetTransform` with a correction (to_pixel adds the field; to_world inverts it;
   a placement without one is unchanged); the choice rule; the stage table and the generated `dvc.yaml` (an
   existing test keeps the committed file equal to the generated one).
3. **The seams, by eye** (AGENTS.md: passing tests aren't enough for warping): render where corrected sheets meet,
   and look.
4. **A fresh clone** (empty folder, `pixi install`, `uv sync`, `sfzmaps pull`): `sfzmaps build` reports the warp
   stages fresh and `data/warp` matches the recorded hashes; then `sfzmaps warp 1921` in a fresh clone with
   `data/warp` removed rebuilds byte-identical corrections.
5. Before merge: `sfzmaps build` (runs only the warp stages; the old ones are frozen), commit `dvc.lock`,
   `sfzmaps ship` (AGENTS.md). Nothing in the build or the checks may call a paid model.

## Constraints

- Never commit `pipeline/data/`; never run a bare `dvc gc`; DVC cache type stays `reflink,copy`.
- No paid model calls (a build stage runs with `SFZMAPS_MAX_USD=0`; the warp makes none).
- Dependencies must be AGPL-compatible (PyTorch: BSD-3).
- The pipeline already uses Python 3.12, numpy 2.5.4 and scipy 1.18.1, the versions the fits were made with;
  PyTorch 2.15.0 is the version they were made with.
- The owner merges.

## Order of work

*(My call.)* The port lands in #58 first with today's street edges and later blocks behaviour (2000 p09 and p10):
checked byte-identical, built, recorded in `dvc.lock`, shipped. The street edges and later blocks question is
settled after it, as its own change, so the port is judged on its own.

## Left to the builder

- The subpackage's name and how it is split into modules.
- How a sheet's "inputs unchanged" is recorded.
- Logging and progress output.
- Whether `sfzmaps warp` with no series runs in table order or in parallel across series.
- Where the documentation lives.
- Whether code changes count as a sheet's inputs for skipping (they should: a better fit is a reason to rewarp).
- Whether `shared` depends on the warp stages.
- How `correction.npz` is found from `transform.json` (`from_json` gets a dict, not a folder).
- Worker processes, so each fit keeps its thread count; CPU-only PyTorch on Linux (keeps CUDA out of CI).
