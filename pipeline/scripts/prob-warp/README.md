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

- **Where the scripts came from:** 15 from the R&D scratch folder in `/tmp` (never in git before 2026-11-13) and 21
  from the local branch `zoning-lines-rnd`. They now run from the repo; paths are settings (`check/rebuild_env.sh`).
- **Rebuilt from scratch, identical to the approved run:** all 13 2000 sheets, p1 and prob6 (`check/rebuild_series.sh`),
  with the years chain's pages and the 2000 code list it needed at the time.
- **No text removal since 2026-11-13** (below). prob6 now fits the page's ink as printed, so its fields differ from the
  approved run's by up to 8.2 px on 2000 (median 0.03–0.13 px per sheet).

## Run order, per sheet

1. **Pipeline:** the sheet's `page.png` and fitted transform (`refine`).
2. **The earlier correction, "p1"** (`allyears/corr.sh`; all sheets: `corr_series.sh`):
   - side points, targets and distance measures (`sides.py`, `regprep2.py`, `distmeasure.py`, `heldrule.py`, ...);
   - the fit (`confit6.py`);
   - output: `confit_<sheet>_p1.npz`, plus the `reg2_*` and `dm_sides_*` inputs that step 3 needs.
3. **prob6** (`allyears/prob.sh`; all sheets: `prob_all.sh`):
   - `ay_probprep.py` and `ay_regprep3.py` (inputs: the page's ink as printed, text included);
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
   - every sheet is chosen the same way (the approved run used a letter-erased prob6 for 2000 p06 and p07).
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
  - `allyears-chain/`: the years chain's text-erased pages, which prob6 read until 2026-11-13;
  - `data_ay2/`: the mirror with the chosen transforms.

## Running it

- `rebuild.sh [SERIES ...]` runs the six steps in order. Settings are environment variables, listed at its top. All
  paths default to this repo; outputs go to `pipeline/data/prob-warp/run` (git-ignored).
- The torch environment: `uv venv --python 3.12 DIR && uv pip install --python DIR/bin/python torch==2.15.0
  numpy==2.5.4 scipy==1.18.1`, then `PW_TPY=DIR/bin/python`.
- `check/`:
  - `rewarp_all.sh`: the whole warp, every series: fits, choice, plus, warps (settings in `rebuild_env.sh`; set
    `PW_PIPELINE` when running from a copy outside the repo);
  - `rebuild_one.sh` / `rebuild_series.sh` / `rebuild_all.sh`: rebuild sheets and compare them with the rescued copy
    (`compare_rebuild.py`);
  - `fit_twice.sh`: fit a series' sheets a second time and compare the fields (is prob6 the same every run?);
  - `field_diff.py`, `field_diff_spot.py`: how far two runs' fields differ per sheet, and a close-up of the largest.
- The scripts that made the saved reads the warper once used (`textboxes.py`, `codes2000/`) and the owner's code-check
  answers moved to the saved-reads PR (#59): `pipeline/scripts/saved_reads/sources/` and
  `pipeline/inputs/code-check-2000/`.

## Text removal dropped (2026-11-13)

The approved run took printed zoning codes out of the ink before fitting: the years chain fitted a letter shape to each
code (from Terra's boxes and the 2000 code list) and erased it, and 2000 p06 and p07 used a page with every fitted
letter erased. The owner asked why the warper needed to know about text, so prob6 was run on the 13 2000 sheets with
and without it (`check/text_ablation.sh`, commit b5bea0d; removed with the chain after):

| Mean over 13 sheets | With text removal | Without |
|---|---|---|
| block-line points within 1 px of a printed line | 0.759 | 0.759 |
| within 2 px | 0.822 | 0.822 |
| control points' miss, 99th percentile (px) | 2.956 | 2.951 |
| smallest cell determinant (folds below 0) | 0.911 | 0.909 |

At the four largest field differences (p02 8.2 px, p06 4.4, p11 3.4, p01 3.3) the fit without text removal sits on
the print as well or better. On p06 the erased "2" of RH-2 took part of a block line with it. The owner dropped text
removal from the warper. The years chain's scripts (`flow.py`, `fitboxes.py`, `chain_pages.sh`, ...) and the p06/p07
variant (`erased2000/`) were removed from this folder with it; they are in this branch's history and on
`zoning-lines-rnd`.

## Next

1. **One entry point and declared dependencies:** the warper as a package with torch in the pipeline's `rnd` extra.
2. **Determinism:** `check/fit_twice.sh` on every series. Done for 2000 (2026-11-13): all 13 sheets' prob6 outputs are
   byte-identical run to run (THREADS=3), and identical to the fit-test run without text removal.
3. **Recompute what was lost:** the p1 fields of the other series (`corr.sh`), then prob6 on them.
4. **Into production:** move the correction into the pipeline's transform stage instead of the `sitecustomize` patch,
   then regenerate the production maps.
