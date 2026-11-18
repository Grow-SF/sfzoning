# Joint solver inputs: where each comes from (2026-11-13)

Goal: every input is either a fixed input or made by preserved package code from fixed inputs (bootstrap rule).

## Fixed inputs (allowed)

| Input | Where | Status |
|---|---|---|
| Sheet scans (`page.png`, 1-bit, 300 dpi) | pipeline `data/work/2000-digital-zoning/*/page.png` | pipeline-made from the raw scans (DVC) |
| April 2000 table | `zon2000/april2000.geojson` (DataSF download) -> `truth_prep.py` -> parquet | the download file must be kept; prep script preserved (snapshot-w3) |
| Parcels, street centrelines, OSM | pipeline data, `osm_dl/` | fixed |
| Terra text boxes (`textboxes_gpt-6.1-terra.json`, `textboxes2_*`) | `k8/pNN/` | saved reads; **no provenance record yet** (cache keys are sha256 of provider, model, image, prompt, schema; entries hold only the answer; no timestamps beyond file times) |
| Label reads (sol, `district_labels.json` + `.answers`) | pipeline data/work | saved reads with answer sidecars |
| Hand leader moves (101 rows of `label_fixes.csv`) | owner's working copy | **not committed** |

## Made inputs

| Input (package reads) | Made by | From | Script preserved? | Problem |
|---|---|---|---|---|
| Sheet transforms (`data_ay2/.../transform.json`) | zoning-lines all-years chain (confit6 etc.), 2026-11-09 | pipeline refine + street/block control points | `zoning-lines-rnd` branch (~40 scripts, shell drivers, drivers patch code text); `probfit.py` in no branch | not packaged; never rebuilt to check |
| Merged page `MALLP_c1/raw.png` | `zmerge.py` | transforms, `page.png`, frame crop `zframe2.json`, cuts `zframe2_cuts.json` | yes (snapshot-w3) | not packaged |
| Lot grid `truth_grid.npz` | `truth.py` | april2000, `k8/city/footprint.geojson` (graph7 city chain) | yes | footprint from an old chain run |
| Scored lots `lotcode_run.npz` | `lotcode.py` | **an old chain run's closed areas** | yes | depends on an old run; also sets the quarter split |
| Table corrections `judge2/final2.csv` | **LLM subagents judging crops** (242 groups, 888 lots set to our code) | crops of an old run's disputed lots (crops deleted) | crop maker yes, verdicts are LLM answers | LLM answers in the yardstick; biased toward an old solve |
| Label readings `lotclf6/labread2/labside_cwfwx{q}` | `lotclf6.py`, `labread2.py`, `addprints.py`, `labside.py` (+ relink), run by shell drivers per holdout | terra boxes, sol reads, lot grid, old odds | yes | not packaged; driver runs not recorded |
| Wall odds `zodds_Q1x{q}` | strokes -> lines -> closings -> evidence -> enclosure -> fit + trees -> blend | **old cleaned page**, terra letter share | yes | the chain itself (owner dropped it); cleaned page |
| Placement strokes `zstrokes_old` | `zstrokes.py` | old cleaned page | yes | cleaned page |
| Lot-size fits `zlotchar2.npz` | `zlotchar2.py` | table, scored lots | yes | inherits scored |
| Data-error list (checked into PR #57) | package `holes` | all of the above | yes | derived from non-reproducible inputs |

## Port order (bottom-up)

1. Fixed inputs with provenance: terra replay in cache-only mode (recompute each call's key from model, prompt, tile
   image; record them with the cache file's time), keep the table download, commit the leader moves.
2. Sheet warp, then the probabilistic merge.
3. Lot grid and scored lots from the parcels, the table and the mapped footprint only (no old run).
4. Label readings from the saved reads.
5. Placement (strokes from the original scan).
6. Walls inside the solver, learned from the original scan with the text detector (no chain, no cleaned page).
7. Re-score; regenerate the data-error list.
