# Sheet warp

The pipeline places each scanned sheet on the city (its fitted placement, `transform.json`). Small errors remain: a
sheet's printed block lines can miss the real block lines by a few pixels. The sheet warp corrects each sheet's
placement so its printed block lines land on the 2000 block lines, before anything is read from the scan. The owner
approved it on 2026-11-09 ("wow! that's just about perfect"). The published maps don't use it yet.

## Running it, from a fresh clone

```
pixi install && (cd pipeline && uv sync && uv run sfzmaps pull)
zsh pipeline/scripts/sheet_warp/warp_every_year.sh          # every year: fits, choice, warps
```

It writes only under `pipeline/data/sheet-warp/`. One year: `fit_one_year.sh YEAR`; one sheet:
`fit_one_sheet.sh YEAR SHEET` (then `steps/choose_correction.py` and `steps/warp_corrected_sheets.sh` make the warps).

## The steps, for each sheet

1. **Which sheets** (`steps/sheets_to_correct.py`): every fitted sheet of the pattern maps (1921–1948), and every
   fitted sheet the pipeline's label reader classes as a zoning-use sheet in the other years (not height-and-bulk,
   special-use or blank sheets): 174 sheets.
2. **The first correction** (`steps/first_correction.sh`): a smooth correction that pulls the sheet's block sides onto
   the printed lines nearest them.
   - `block_side_points.py`: points along every side of every 2000 block, placed by today's fit;
   - `fit_targets.py`: the printed block lines near each point (from the scan);
   - `measure_fit.py`: how far each point is from a printed line running its way;
   - `held_sides.py`: the sides the print holds well, kept in place;
   - `first_correction_inputs.py`, `propagate_shifts.py`: the shifts to fit, spread along each side;
   - `first_correction_fit.py`: the fit (PyTorch);
   - `control_points_picture.py`: a picture of how far the placement's control points moved.
3. **The stand-in data folder** (`steps/stand_in_data_folder.py`): a copy of the pipeline's data folder in which every
   file is a link to the real one, except each corrected sheet's folder, which holds its placement with the correction
   added. The warp runs on it, so the pipeline's own data is never written.
4. **The probabilistic correction** (`steps/probabilistic_correction.sh`), the one the owner approved:
   - `probabilistic_inputs.py`: every 2000 block-line point, placed by today's fit;
   - `block_line_weights.py`: how likely each inked pixel is to be a block line, from its stroke width;
   - `probabilistic_fit.py` (PyTorch): each point either lies on a printed block line running its way (a Gaussian
     across the line) or isn't printed (missing, or hidden under heavy ink); the fit is the smooth displacement field
     (a grid every 50 px) that makes all the points together most probable. Smoothness and anti-tearing terms keep it
     from folding; the spread starts at half a street width and halves each stage;
   - `measure_fit.py`, `score_fits.py`: the share of block-line points within 1 and 2 px of the print, how far the
     control points moved, and the smallest cell area (below 0 means a fold), for both corrections.
5. **The choice** (`steps/choose_correction.py`): the probabilistic correction where it puts at least as many points
   within 1 px as the first correction and never folds, otherwise the first correction.
6. **Extra evidence for 2000 sheets p09 and p10** (`steps/extra_evidence_2000.sh`, `extra_evidence_correction.sh`,
   `extra_evidence_points.py`, `extra_evidence_inputs.py`): these sheets have few blocks of their own year, so the fit
   also uses later years' blocks and the street edges.
7. **The warp** (`steps/warp_corrected_sheets.sh`, `warp_one_corrected_sheet.sh`, `warp_with_correction.py`): each
   sheet warped with its chosen correction, exactly as the pipeline warps it otherwise. `correction_patch/` makes the
   pipeline's placement code apply the correction while the warp runs.

Shared by the steps: `warp_common.py` (paths and settings), `corrected_transform.py` (today's fit plus a correction),
`crop_extent.py`, `threaded_spline.py`. The 2000 sheets are fitted as the approved run fitted them: each on its own
measured area (`measured_area_2000.py`) with the scan's scale taken as 3.99 ft per px.

## Files

```
sheet_warp/
  README.md
  warp_every_year.sh            the whole warp, every year
  fit_every_year.sh             step 1-4 for every year
  fit_one_year.sh               ... for one year
  fit_one_sheet.sh              ... for one sheet
  settings.sh                   where it reads and writes
  make_torch_environment.sh     the PyTorch environment, made on first use
  torch_requirements.txt        its exact packages
  steps/                        every step above
  correction_patch/             applies a correction inside the pipeline's placement code during the warp
  checks/
    fit_twice_and_compare.sh    fit a year again and compare: the same every run?
    field_differences.py        how far two runs' corrections differ, per sheet
    field_difference_closeup.py a picture of where two runs' corrections differ most
```

## What it reads, and where each comes from

Every script it runs is in this folder; everything else it reads is one of these.

| What | Where | Comes from |
|---|---|---|
| each sheet's scan, placement and pipeline outputs: `page.png`, `page.json`, `transform.json`, `refine.json`, `blocks_px.geojson`, `warped.footprint.geojson`, `warped.tif`, `district_labels.json` (with its `.answers`) | `pipeline/data/work/<year>/<sheet>/` | the pipeline's build from the scans (`sfzmaps pull` brings them; `sfzmaps fetch` then `sfzmaps build` remakes them) |
| the lots of each map's year, and every lot ever recorded | `pipeline/data/ref/parcels.parquet` (the lot history is computed from it into `data/cache`) | DataSF's parcels, the pipeline's reference data (`sfzmaps pull`) |
| street centrelines (extra evidence only) | `pipeline/data/ref/streets.parquet` | DataSF, the pipeline's reference data (`sfzmaps pull`) |
| hand corrections to label reads (in choosing the sheets) | `pipeline/inputs/<year>/label_fixes.csv` | git |
| the PyTorch environment's packages | `torch_requirements.txt` | git |
| gdal, for the warps | the pixi environment | `pixi.toml` / `pixi.lock` (`pixi install`) |

## What it writes (`pipeline/data/sheet-warp/`)

| Where | What |
|---|---|
| `fits/<year>/<sheet>/` | each sheet's corrections (`confit_<sheet>_p1.npz`: the first; `_prob6.npz`: the probabilistic; `plus/`: extra evidence), their measures and pictures |
| `fits/probsum.jsonl`, `fits/choice.json` | every sheet's scores, and which correction it got |
| `stand-in-data/`, `stand-in-data-chosen/` | the stand-in data folders: with each sheet's first correction, and with its chosen one and its warps (`warped.tif`) |
| `measured-area-2000.json` | the 2000 sheets' measured areas |
| `torch-environment/`, `logs/` | the PyTorch environment; every run's log |

## Results

- **Reproduces the approved run.** Rebuilt from scratch, all 13 2000 sheets came out byte-identical to the approved
  run's, and so does every fit and warp the change below doesn't touch.
- **No text removal.** The approved run took the printed zoning codes out of the ink before fitting. Run with and
  without on the 13 2000 sheets, the fit scored the same, and where the two differed most the fit without sat on the
  print as well or better (on 2000 p06, erasing the "2" of RH-2 took part of a block line with it). The owner dropped
  it, so the warp needs no text reads.

  | Mean over the 13 2000 sheets | With text removal | Without |
  |---|---|---|
  | block-line points within 1 px of a printed line | 0.759 | 0.759 |
  | within 2 px | 0.822 | 0.822 |
  | control points' miss, 99th percentile (px) | 2.956 | 2.951 |
  | smallest cell area (folds below 0) | 0.911 | 0.909 |

- **Every year, 2026-11-14.** All 174 sheets fitted and warped, none failed. The choice took the probabilistic
  correction on 66 sheets and the first correction on 108 (the probabilistic one mostly on 1921 and 1980–2000, the
  first on 1935–1975). Without text removal, each year's share of points within 1 px of its chosen correction is the
  same as with it, to 0.001; one sheet's choice changed (1990–1996 p28, 0.648 against 0.649, keeps the first).
- **The same every run:** the 13 2000 sheets fitted twice gave byte-identical corrections.

## Next

1. **Look at the seams** where warped sheets meet, against the approved run's (AGENTS.md).
2. **The same every run, every year** (`checks/fit_twice_and_compare.sh`).
3. **Share the warps** as a build stage, so `sfzmaps pull` brings them instead of every machine re-running the fits.
4. **Into production:** apply the correction in the pipeline's own placement step instead of `correction_patch/`, and
   regenerate the published maps.
