# The 1921 label check

The cases of the owner's check of the 1921 zoning map, the earliest: the first ground truth for a map before 1980 (no
digitized zoning of these years exists), to score a solve held out by area and to train readers of the pattern maps.
No build stage reads this folder. The 1921 measure (`sfzmaps.solver.answers1921`) and the hatched-map
reader's lot table (`sfzmaps.solver.hatchnet.data`) read its cases and answers.

| File | What |
|---|---|
| `cases.json` | The 400 cases, in the order the page shows them, and how they were drawn (`design`) and from what (`sources`, each file's sha256). |
| `answers/` | The owner's 400 answers so far (all given 2026-11-18), one file per answered case, exactly as the page's store exports them. |

## How the map shows districts

The 1921 map (Ordinance No. 5464) prints no codes: each district is a fill pattern keyed to a legend, printed on
section 1 (sheet 20180102094142210): First Residential (1R) diagonal hatch, Second Residential (2R) blank, Commercial
(C) solid black, Light Industrial (LI) diamond cross-hatch, Heavy Industrial (HI) square grid, Unrestricted (U)
checkerboard. The pipeline's pattern reader (reader A, `districts/pattern.py`) gave a district to each lot or whole
block it could see, saved per sheet in `data/work/1921/<sheet>/district_pattern.json` (`via`: block or parcel). The
1921 solve takes each read as a print standing for its own lot (`data/solver/1921/prints/prints.parquet`, 148,623 reads
on lots of the April 2000 table) and solves every lot from them.

## Provenance

Made by `scripts/label_checks/build_label_checks.py --series 1921` (seed 7) on the `label-checks` branch, from the `solver` branch at
bb40bc4, its data as `pipeline/dvc.lock` there records it; no API calls. It reads:

- `data/solver/1921/prints/prints.parquet` and the sheets' `district_pattern.json`: the reads;
- `data/solver/1921/lots/table.parquet`, `lots.npz`: the lots, their blocks and quarters;
- `data/solver/1921/solve/runs/final.npz`: the solve's code per lot;
- `data/out/1921/districts/parcels.fgb`: the published 1921 map;
- `data/warp/1921/<sheet>/transform.json`: each sheet's placement, to draw the lot on its page;
- `data/work/1921/<sheet>/page.png`: the scans (300 dpi, 1-bit, the scan's own resolution); the legend and its
  swatches (`src/sfzmaps/districts/legends.json`) are cut from section 1's.

## A case

One lot the reader read. The page shows a 256 px crop of the read's sheet centred on the lot, the lot outlined (and
ringed when small), with `code`: the reader's district, its legend swatch, and the printed legend. The owner answers
`ok`, `fix` (the district shown, picked from the legend) or `canttell`.

Fields: `id`, `order` (on the page), `stratum`, `disagree`; `code`, `confidence`, `via` (the read); `solve`,
`published` (the two maps' codes at the lot); `sheet`, `print` (row of prints.parquet); `x`, `y` (EPSG:2227 ft, a
point inside the lot); `quarter`, `block`, `blklot`, `lot` (row of the lot table); `crop` (x0, y0, size on the
page), `outline` (the lot's rings in crop px: page px = crop origin + outline), `sprite`, `slot`.

## The sample

The population is the 148,623 reads. Two strata:

- `random`, 240 cases: a proportionally stratified random sample of all reads. Cells are quarter (`paths.QUARTERS`)
  x `code`; within each cell the reads are shuffled and read k of N is keyed (k + u) / N; the sample is the 240
  smallest keys. It is self-weighting (each read had a 240 / 148,623 chance): the plain mean of its answers estimates
  the whole map.
- `disagree`, 160 cases: lots where the solve and the published map differ (29,828 of the reads), not already drawn
  in `random` (52 were), drawn the same way. Every case with `disagree` true (212, both strata) is a uniform sample of
  the disagreements: their plain mean estimates the disagreements.

The page interleaves the strata by rank (case k of a stratum of n at (k + u) / n), so any prefix of answers is a fair
sample of both. To score a solve held out by area, group by `quarter` (or by `block`).

## The answers

The page saves one document per case in its store, collection `answers-1921`, id = the case's `id`:
`{case, order, answer, code, shown, stratum, at}`. `answer` is ok / fix / canttell; `code` is the district there by the
owner (the shown code for ok, the picked one for fix, "" for canttell); `shown` the district the page showed.

## "unzoned" answers

For a while (2026-11-18) the 1921 pages also offered "Unzoned" (key N, saved as `{answer: "unzoned", code: ""}`) for
wilderness and water no district seemed to cover. The owner then ruled, as the 1921 ordinance says, that every part
of the city is in one of the six districts and white means 2R, and the answer was taken off the pages. Answers saved
as `unzoned` stay in the store: they mean "2R on the map; real-world water or wild land, for the display rule".
