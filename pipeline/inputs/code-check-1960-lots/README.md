# The 1960 lot check

The cases of the owner's lot check of the 1960 zoning map: which district is this lot in? It follows the 1960 label
check (`code-check-1960`), where 55 of the first 56 printed labels were read right: the reads are mostly right, and
the open question is where the district lines run. Nothing in the build reads this folder.

| File | What |
|---|---|
| `cases.json` | The 300 cases, in the order the page shows them, how they were drawn (`design`) and from what (`sources`, each file's sha256). |
| `answers/` | The owner's 299 answers so far (all given 2026-11-18), one file per answered case, exactly as the page's store exports them. |

## Provenance

Made by `scripts/label_checks/build_label_checks.py --series 1960 --page lots` (seed 7) on the `label-checks`
branch, from the `solver` branch at bb40bc4, its data as `pipeline/dvc.lock` there records it; no API calls. It reads
`data/solver/1960/lots/` (the lots of the April 2000 table, their neighbours, blocks and quarters),
`data/solver/1960/solve/runs/final.npz` (the solve's code per lot), `data/out/1960/districts/parcels.fgb` (the
published 1960 map, at each lot's inside point), `data/warp/1960/<sheet>/` (each sheet's placement and footprint) and
`data/work/1960/<sheet>/page.png` (the scans, 300 dpi).

## A case

One lot. The page shows a 640 px crop of the scan (2.7 ft per px: about 1,730 ft, a few blocks with their printed
labels and district lines; 2x zoom shows the middle half) centred on the lot, on the sheet that holds the lot deepest
inside its footprint, the lot outlined and ringed. The answer pad offers the candidate codes, the published map's and
the solve's, as `options`, in a random order drawn per case and without saying which map gave which (one button when
they agree). The owner answers `pick` (one of them), `other` (another 1960 code, from the quick picks or typed) or
`canttell`.

Fields: `id`, `order`, `stratum`, `disagree`, `options` (as shown: key 1, key 2), `published`, `solve`, `pi` (the
case's chance of being drawn), `area` (the solve area: a connected run of lots the solve gives one code, numbered for
this solve), `patch` (the disagreement patch: a connected run of lots with the same pair of codes; -1 where the maps
agree), `lot` (row of the lot table), `blklot`, `block`, `quarter`, `x`, `y` (EPSG:2227 ft, a point inside the lot),
`sheet`, `crop` (x0, y0, size on the page), `outline` (the lot's rings in crop px), `sprite`, `slot`.

## The sample

The population is the 153,029 lots of the solve on a sheet (263 lots no sheet shows are left out). The solve and the
published map give 41,303 of them different codes. Two strata:

- `random`, 75 lots: a proportionally stratified random sample of all lots (cells: quarter x the solve's code; each
  lot's chance 75 / 153,029). 20 of them are disagreements. Their plain mean estimates the whole map.
- `disagree`, 225 lots where the maps differ, not in `random`: split over the quarters in proportion to their
  disagreeing lots (nw 51, ne 69, sw 74, se 106 cases in all, both strata). Within a quarter no solve area gives more
  than 3 lots and the areas share the rest in proportion to their disagreeing lots; the lots of an area are spread over
  its disagreement patches. The 225 lots come from 143 of the 655 solve areas that hold a disagreement. Weight each
  answer by 1 / `pi` to estimate all the disagreeing lots.

The page interleaves the strata by rank, so any prefix of answers is a fair sample of both. To score a solve held out
by area, group by `quarter`, `area` or `block`.

## The answers

The page saves one document per case in its store, collection `answers-1960-lots`, id = the case's `id`:
`{case, order, answer, code, options, stratum, at}`. `answer` is pick / other / canttell; `code` is the district the
owner gives the lot ("" for canttell); `options` the candidates as shown.
