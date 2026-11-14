# Sheet warp

The pipeline places each scanned sheet on the city (its fitted placement, `transform.json`). Small errors remain: a
sheet's printed block lines can miss the real block lines by a few pixels. The sheet warp corrects each sheet's
placement so its printed block lines land on the block lines of its year, and warps the sheets with the corrections.
The owner approved it on 2026-11-09 ("wow! that's just about perfect"). The published maps don't use it yet; the
joint solver (#57) reads its stand-in data folder.

## Running it

From a fresh clone:

```
pixi install && cd pipeline && uv sync && uv run sfzmaps pull
uv run python scripts/sheet_warp/warp.py all
```

`warp.py` is the only entry point:

| Command | What it does |
|---|---|
| `warp.py all [YEAR ...]` | fit every sheet of those years (default: every year), choose each sheet's correction, warp the sheets |
| `warp.py fit YEAR [SHEET ...]` | fit one year's sheets (default: all it corrects) |
| `warp.py again YEAR [SHEET ...]` | fit the probabilistic correction a second time and print how far the two differ: is it the same every run? |

A sheet already fitted is skipped. The two fits run in a PyTorch environment that uv makes on first use from
`fits_environment/` (locked in its `uv.lock`); everything else runs in the pipeline's own environment. It reads the
pipeline's data and writes only under `pipeline/data/sheet-warp/`.

## How it works, for each sheet

1. **Which sheets.** Every fitted sheet of the pattern maps (1921–1948), and every fitted sheet the pipeline's label
   reader classes as a zoning-use sheet in the other years (not height-and-bulk, special-use or blank): 174 sheets.
   From the pipeline's saved answers only; nothing new is asked.
2. **What to aim at** (`steps/block_side_points.py`, `fit_targets.py`, `measure_fit.py`). Points every 2 ft along the
   side of every block of the sheet's year, inside the sheet's drawn map. On the scan, the centre line of every stroke
   no wider than a block line, and for each direction a map of the distance to the nearest such line running that
   way. A point fits the print when it lies within 1 px of a printed line running its way.
3. **The first correction** (`held_sides.py`, `first_correction_inputs.py`, `propagate_shifts.py`,
   `first_correction_fit.py`). Sides the print already holds stay where they are. Every other side is moved onto the
   nearest printed line running its way, carried outward from the held sides so neighbouring sides move together.
   The correction is a smooth field (a grid every 50 px) fitted to all of that.
4. **The probabilistic correction** (`probabilistic_inputs.py`, `block_line_weights.py`, `probabilistic_fit.py`).
   Each block-side point either lies on a printed block line running its way, or isn't printed (missing, or hidden
   under heavy ink). Ink counts as a block line by how likely its stroke width says it is. The fit is the smooth field
   that makes all the points together most probable, held to stretching about as much as today's placement does, with a
   steep penalty wherever it would fold the scan over. It starts blurred to half a street width and sharpens in stages.
5. **The choice** (`warp.py`, with `score_fits.py`). The probabilistic correction where it puts at least as many
   points within 1 px of the print as the first correction and never folds; otherwise the first correction.
6. **Extra evidence, 2000 sheets p09 and p10** (`extra_evidence_points.py`). Every other sheet is fitted to the blocks
   of its own year. On these two, much of the printed street grid has no blocks in the year's lot records (land held
   whole then and split up later, such as the Hunters Point shipyard on p09), so the fit has little to hold on to there. They
   also use the outlines of later blocks and the edges of every street, each point counted by how little the year's
   own blocks already explain it. Otherwise the fit is the same probabilistic correction.
7. **The warp** (`warp_with_correction.py`, `correction_patch/`). Each sheet warped with its chosen correction, exactly
   as the pipeline warps it otherwise, into the stand-in data folder: a copy of the pipeline's data folder in which
   every file is a link to the real one, except each corrected sheet's folder, whose `transform.json` names its
   correction. `correction_patch/` makes the pipeline's placement code apply that correction. The pipeline's own data
   is never written.

## Files

```
sheet_warp/
  README.md
  warp.py                    the entry point: which sheets, the order of the steps, the choice, the stand-in folder
  fits_environment/          the PyTorch environment's packages (pyproject.toml, uv.lock)
  correction_patch/          applies a correction inside the pipeline's placement code while the warp runs
  steps/                     one script per step above; warp_common.py holds what they share
```

## What it reads, and where each comes from

Every script it runs is in this folder; everything else it reads is one of these.

| What | Where | Comes from |
|---|---|---|
| each sheet's scan, placement and pipeline outputs: `page.png`, `page.json`, `transform.json`, `refine.json`, `blocks_px.geojson`, `warped.footprint.geojson`, `warped.tif`, `district_labels.json` (with its saved answers) | `pipeline/data/work/<year>/<sheet>/` | the pipeline's build from the scans (`sfzmaps pull` brings them; `sfzmaps fetch` then `sfzmaps build` remakes them) |
| the lots of each map's year, and every lot ever recorded | `pipeline/data/ref/parcels.parquet` (the lot history is computed from it into `data/cache`) | DataSF's parcels, the pipeline's reference data (`sfzmaps pull`) |
| street centre lines (extra evidence only) | `pipeline/data/ref/streets.parquet` | DataSF, the pipeline's reference data (`sfzmaps pull`) |
| hand corrections to label reads (in choosing the sheets) | `pipeline/inputs/<year>/label_fixes.csv` | git |
| the PyTorch environment's packages | `fits_environment/pyproject.toml`, `uv.lock` | git |
| gdal, for the warps | the pixi environment | `pixi.toml`, `pixi.lock` (`pixi install`) |

## What it writes (`pipeline/data/sheet-warp/`)

| Where | What |
|---|---|
| `fits/<year>/<sheet>/` | each sheet's corrections (`first_correction.npz`, `probabilistic_correction.npz`, and on 2000 p09 and p10 `extra_evidence_correction.npz`), and every file they were fitted from, each named for the step that makes it |
| `fits/scores.jsonl` | every sheet's scores: points within 1 and 2 px of the print, control points' miss, the smallest cell (below 0 folds) |
| `fits/choice.json` | which correction each sheet got, and its scores |
| `stand-in-data/` | the stand-in data folder: the chosen corrections, and the warped sheets (`warped.tif`, `warped_paper.tif`) |
| `fits-environment/`, `logs/` | the PyTorch environment; each sheet's log |

## Results

- **The same corrections as the approved run's rebuild** wherever nothing below changed them: a 1921 sheet fitted
  with this code gives byte-identical arrays in every file the steps still save.
- **No text removal.** The approved run took the printed zoning codes out of the ink before fitting. Fitted with and
  without on the 13 2000 sheets, the scores were the same (within 1 px 0.759 both), and where the two differed most
  the fit without sat on the print as well or better. The owner dropped it, so the warp reads no text.
- **2000 fitted like every other year.** The approved run fitted the 2000 sheets on their own measured area with a
  fixed scale. Scored on the same points, those corrections and this code's are equal (within 1 px 0.749 against
  0.748, within 2 px 0.812 both).
- **Every year, 2026-11-14:** all 174 sheets fitted and warped, none failed (before the two changes above).

## Next

1. **Look at the seams** where warped sheets meet (AGENTS.md).
2. **The same every run, on every year** (`warp.py again`).
3. **Share the warps** as a build stage, so `sfzmaps pull` brings them instead of every machine re-running the fits.
4. **Into production:** apply the correction in the pipeline's own placement step instead of `correction_patch/`, and
   regenerate the published maps.
