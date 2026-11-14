# Probabilistic sheet warp ("prob6")

Corrects each sheet's fit to the city (its `transform.json`), so the printed block lines land on the 2000 block lines
before anything is read from the scan. The owner approved it on 2026-11-09 ("wow! that's just about perfect"), and it
ran on every series' sheets that night. This folder holds those scripts, made to run from a fresh clone, so the work is
reproducible and can move into the production pipeline.

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
- **Runs in PyTorch** (`requirements-torch.txt`, made by `check/torchenv.sh`).

## Running it, from a fresh clone

Everything it reads is in this checkout or comes with `sfzmaps pull`; it writes only under `pipeline/data/prob-warp`.

```
pixi install && (cd pipeline && uv sync && uv run sfzmaps pull)
zsh pipeline/scripts/prob-warp/check/rewarp_all.sh            # every series: fits, choice, plus, warps
```

- **The sheets:** `allyears/sheets.py SERIES`, from the pipeline's data: every fitted sheet of a pattern-map series
  (1921–1948), and every fitted sheet the pipeline's label reader classes as a zoning-use sheet in the others (not
  height-and-bulk, special-use or blank sheets). 174 sheets, the same as the approved run.
- **The torch environment:** made on first use at `pipeline/data/prob-warp/torchenv` (Python 3.12, the pinned versions).
- **Settings:** `check/rebuild_env.sh` (outputs in `pipeline/data/prob-warp/rebuild-2026-11-13/run-notext`, its
  `data_ay` / `data_ay2` stand-in data folders, and the 2000 sheets in `SAME2000` mode as the approved run had them).
- **One sheet:** `check/rebuild_one.sh SERIES SHEET`; one series: `check/rebuild_series.sh SERIES`.
- **Checks:** `check/fit_twice.sh SERIES` fits a series again and compares the fields (the same every run?);
  `field_diff.py` / `field_diff_spot.py` compare two runs. Where a machine still has the approved run's outputs
  (`pipeline/data/prob-warp/rescued-2026-11-13`, never in git), `rebuild_series.sh` also compares each sheet with them;
  without that folder it skips the comparison.

## Run order, per sheet

1. **Pipeline:** the sheet's `page.png` and fitted transform (`refine`).
2. **The earlier correction, "p1"** (`allyears/corr.sh`):
   - side points, targets and distance measures (`sides.py`, `regprep2.py`, `distmeasure.py`, `heldrule.py`, ...);
   - the fit (`confit6.py`);
   - output: `confit_<sheet>_p1.npz`, plus the `reg2_*` and `dm_sides_*` inputs that step 3 needs.
3. **The stand-in data folder entry** (`allyears/build_mirror_ay.py`): `data_ay` links every file to the pipeline's
   data except the sheet's folder, whose `transform.json` points at its correction and whose outline is moved through
   it. The warp reads this folder, so nothing in the pipeline's own data is written.
4. **prob6** (`allyears/prob.sh`):
   - `ay_probprep.py` and `ay_regprep3.py` (inputs: the page's ink as printed, text included);
   - `probfit.py <sheet> prob6 800` under the torch environment (the fit);
   - `distmeasure.py prob6` (measure) and `ay_probsum.py` (summary);
   - output: `confit_<sheet>_prob6.npz`.
5. **Choice** (`allyears/mirror_prob.py`): prob6 where its share of block-line points within 1 px is at least p1's and
   it never folds, otherwise p1, written into `data_ay2`.
6. **"plus"** (2000 p09 and p10; `allyears/apply_plus.sh`, `prob_plus.sh`): `sides_plus.py` adds later blocks and
   street edges as weighted evidence.
7. **Apply** (`corrpatch/sitecustomize.py`, run with `PYTHONPATH=corrpatch SFZ_CORR=1`): loads the correction into
   `SheetTransform`; `allyears/warp_prob.sh` (`warpK.py`) writes `warped.tif`.

## Status

- **Rebuilt from scratch, identical to the approved run:** all 13 2000 sheets, p1 and prob6, with the inputs that run
  used for text removal (since removed).
- **No text removal since 2026-11-13** (below). prob6 fits the page's ink as printed; on 2000 its fields differ from
  the approved run's by up to 8.2 px (median 0.03–0.13 px per sheet).
- **Determinism:** fitting the 13 2000 sheets twice gives byte-identical outputs (THREADS=3).
- **Full re-warp, every series (2026-11-14, `check/rewarp_all.sh`):** 174 sheets fitted and warped, none failed. The
  choice took prob6 on 66 sheets and p1 on 108 (prob6 mostly on 1921 and 1980–2000, p1 on 1935–1975).
  - Every p1 fit is byte-identical to the approved run's; so is every prob6 fit of 1921–1948, which never had text
    removal. Every warped sheet whose chosen fit didn't change is byte-identical to the approved run's warp (all of
    1921–1948, and every sheet that keeps p1).
  - Without text removal, each year's share of block-line points within 1 px of the chosen fit is the same as with
    it, to 0.001 (1960 0.705, 1964 0.720, 1968 0.704, 1972 0.725, 1975 0.742, 1980 0.774, 1980–1990 0.774,
    1990–1996 0.718, 2000 0.759). The choice changes on one sheet: 1990–1996 p28 (prob6 0.648 vs p1 0.649) keeps p1.

## The approved run (`manifest.csv`)

174 sheets across all series got a correction on 2026-11-08/09. Per sheet, the manifest gives the chosen fit, its
correction file, its sha256, size and modified time, and the measures behind the choice. Many of those files were
deleted by a `/tmp` cleaner on 2026-11-13; `rewarp_all.sh` remakes every one.

## Text removal dropped (2026-11-13)

The approved run took printed zoning codes out of the ink before fitting: a years chain fitted a letter shape to each
code (from Terra's boxes and the 2000 code list) and erased it, and 2000 p06 and p07 used a page with every fitted
letter erased. The owner asked why the warper needed to know about text, so prob6 was run on the 13 2000 sheets with
and without it (commit b5bea0d, `check/text_ablation.sh`):

| Mean over 13 sheets | With text removal | Without |
|---|---|---|
| block-line points within 1 px of a printed line | 0.759 | 0.759 |
| within 2 px | 0.822 | 0.822 |
| control points' miss, 99th percentile (px) | 2.956 | 2.951 |
| smallest cell determinant (folds below 0) | 0.911 | 0.909 |

At the four largest field differences (p02 8.2 px, p06 4.4, p11 3.4, p01 3.3) the fit without text removal sits on
the print as well or better. On p06 the erased "2" of RH-2 took part of a block line with it. The owner dropped text
removal from the warper, and the scripts that served only it are gone from this folder (they are in this branch's
history). The scripts that made the saved reads it used, and the owner's code-check answers, are in the saved-reads
PR (#59).

## Next

1. **One entry point as a package**, with torch declared in the pipeline's dependencies.
2. **Determinism on every series:** `check/fit_twice.sh`.
3. **Look at the seams** where re-warped sheets meet (AGENTS.md), against the approved run's.
4. **Into production:** move the correction into the pipeline's transform stage instead of the `sitecustomize` patch,
   then regenerate the production maps.
