# Probabilistic sheet warp ("prob6")

Corrects each sheet's fit to the city (its `transform.json`), so the printed block lines land on the 2000 block lines
before anything is read from the scan. The owner approved it on 2026-11-09 ("wow! that's just about perfect"), and it
ran on every series' sheets that night. This folder holds the scripts exactly as they ran, so the work can be made
reproducible and put into the production pipeline.

## How it works (`probfit.py`)

- **The points:** every 2000 block-line point, as placed by the sheet's current fit.
- **How the print explains each point,** as a probability:
  - *printed:* the point lies on a printed block line running its way, with a Gaussian density across that line;
  - *not printed:* the point is missing from the print or hidden under heavy ink, with a uniform density over every
    possible distance.
- **The fit:** a smooth displacement field (grid every 50 px, bilinear) that makes all the points together as probable
  as possible.
  - Smoothness and anti-rip terms keep it from tearing.
  - The spread starts at half a street width and halves each stage.
  - The outlier share and spread are re-estimated (EM) after each stage.
- **Runs in PyTorch.**

## Status

- **Unchanged copies:**
  - 15 scripts from the R&D scratch folder in `/tmp`, which were never in git before 2026-11-13;
  - 21 scripts from the local branch `zoning-lines-rnd`.
- **Hard-coded paths throughout:** the scratch folder and the `cache-maybe` worktree.
- **Never re-run from these copies.** Rebuilding one sheet and comparing it with `manifest.csv` is the first check.

## Run order, per sheet

1. **Pipeline:** the sheet's `page.png` and fitted transform (`refine`).
2. **The earlier correction, "p1"** (`allyears/corr.sh`; all sheets: `corr_series.sh`):
   - side points, targets and distance measures (`sides.py`, `regprep2.py`, `distmeasure.py`, `heldrule.py`, ...);
   - the fit (`confit6.py`);
   - output: `confit_<sheet>_p1.npz`, plus the `reg2_*` and `dm_sides_*` inputs that step 3 needs.
3. **prob6** (`allyears/prob.sh`; all sheets: `prob_all.sh`):
   - `ay_probprep.py` and `ay_regprep3.py` (inputs);
   - `probfit.py <sheet> prob6 800` under the torch environment (the fit);
   - `distmeasure.py prob6` (measure) and `ay_probsum.py` (summary);
   - output: `confit_<sheet>_prob6.npz`.
4. **"plus"** (2000 sheets p09 and p10; `allyears/prob_plus.sh`):
   - `sides_plus.py` adds later blocks and street edges as weighted evidence;
   - `mk_probfit_plus.py` made that edit to `probfit.py`;
   - `probfit_before_plus.py` is the version step 3 ran with.
5. **Choice** (`allyears/mirror_prob.py`):
   - prob6 where its share of block-line points within 1 px is at least p1's and it never folds, otherwise p1;
   - each chosen sheet's `transform.json` in the mirror gets `"correction": {"file": ...}`;
   - 2000 p06 and p07 used a prob6 fitted on a page with every fitted letter erased.
6. **Apply** (`corrpatch/sitecustomize.py`, run with `PYTHONPATH=corrpatch SFZ_CORR=1`):
   - loads the correction into `SheetTransform`;
   - `allyears/warp_prob.sh` (`warpK.py`) writes `warped.tif`.

**Environment for `probfit.py`:** a separate venv with Python 3.12, torch 2.15.0, numpy 2.5.4 and scipy 1.18.1. The
scratch copy (`torchenv`) was damaged by the `/tmp` cleaner, so reinstall it.

## What exists (`manifest.csv`)

174 sheets across all series got a correction. Per sheet, the manifest gives:
- the chosen fit and the correction file;
- whether that file still exists, with its sha256, size and modified time;
- the measures behind the choice: share of block-line points within 1 px for the current fit, p1 and prob6; the 99th
  percentile move of the control points; the smallest determinant (a fold check).

| | Sheets |
|---|---|
| prob6 field kept | 64 |
| p1 field kept | 1 |
| p1 field deleted by the `/tmp` cleaner on 2026-11-13 | 107 |
| 2000 p06, p07 (letter-erased prob6) deleted | 2 |

- **2000-digital-zoning:**
  - kept: p01–p05, p08 and p11–p13 (prob6), and p09–p10 (plus);
  - lost: p06 and p07. Their plain prob6 fits survive, plus one variant for p06 (`confit_06_prob6lm.npz`).
- **The lost p1 fields:** their logs and distance maps survive, and `corr.sh` recomputes them.
- **Rescued copy (not in git):** in the `cache-maybe` worktree's git-ignored
  `pipeline/data/prob-warp/rescued-2026-11-13/`, with timestamps kept.
  - `allyears-run/`: every sheet's outputs and logs;
  - `allyears-chain/`: the years chain's text-erased pages, which prob6 reads (`ay_regprep3.py`);
  - `data_ay2/`: the mirror with the chosen transforms.

## Running it

- `rebuild.sh [SERIES ...]` runs the six steps in order. Settings are environment variables, listed at its top. All
  paths default to this repo; outputs go to `pipeline/data/prob-warp/run` (git-ignored).
- The torch environment: `uv venv --python 3.12 DIR && uv pip install --python DIR/bin/python torch==2.15.0
  numpy==2.5.4 scipy==1.18.1`, then `PW_TPY=DIR/bin/python`.
- `check/`: rebuild one sheet (`rebuild_one.sh`) and compare its outputs with the rescued copy (`compare_rebuild.py`).

## Rebuild status

- **2000 p01, rebuilt with these scripts: not yet identical.** It used 429,825 block-line points against the
  original's 428,227. The 2000 sheets were run in `SAME2000` mode (`ftpx` 3.99, and each sheet's measured area from
  `stepview/city_m2/parts.json`). That file was deleted by the `/tmp` cleaner, and none of the surviving older
  `parts.json` files, nor one rebuilt from the pipeline's footprints, reproduces the original's points
  (`check/try_parts.sh`).
- **Not recreatable yet:** 2000 p06 and p07's letter-erased fits, and the warps built from them.

## Next

1. **Make it run from the repo:** no hard-coded paths, one entry point, torch as a declared dependency.
2. **Check a rebuild:** rebuild one sheet that has a surviving prob6 field, and compare it with the manifest.
3. **Recompute what was lost:** the p1 fields, and 2000 p06 and p07. The owner has dropped page cleaning for reading
   zoning, so decide whether the letter-erased variant is still wanted.
4. **Into production:** move the correction into the pipeline's transform stage instead of the `sitecustomize` patch,
   then regenerate the production maps.
