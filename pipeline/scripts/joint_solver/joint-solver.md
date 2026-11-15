# The joint solver for the 2000 zoning map

R&D, not wired into the build. It reads the district of every parcel from the 2000-digital-zoning scan. It grew out of
the "bubble" solver of earlier R&D. It runs on a data folder (see `README.md`) and makes no paid reads.

## How it works

One energy over every parcel's zoning code, minimised once for the whole city.

1. **Walls, priced both ways.** Each pair of neighbouring parcels has odds that a district line runs between them, read
   from the printed ink and held out by quarter.
   - Different codes across a border with no line cost those odds.
   - The same code across a printed line costs them too.
2. **Labels, charged once.** A printed label claims its code for the lots it stands for: the lot under its text, the far
   side of a street, or a small box its leader points into.
   - The chances come from a placement model learned from the 101 hand-checked leader moves and the table.
   - Trust is calibrated (trust 0.99 means right 99.4% of the time).
   - Readings that name no 2000 district (overlay text such as "NORTH OF", "RES.", "SUD") are dropped.
3. **Nothing charged per lot.** A per-lot "distance to the nearest label" and a per-lot park prior were removed. Both
   made large districts pay the same cost on every lot, so the energy preferred wrong answers.
4. **Search.** Graph-cut expansion (PyMaxflow) runs first without the same-code-across-a-line cost, then with it.
   - Then come whole-district moves: recode a district; swap two districts' codes; let a district take its labels' code
     while the neighbour holding that code moves to another.
   - Each move is kept only if the exact energy falls.
   - A solve takes 2–3 minutes per quarter, the four quarters side by side.

The check used at every step: with the table's own borders as walls (`solve --true-borders`), the solve must find
nearly the table's answer. It does: 100% of the table's lines, 99.9% of its drawn lines right, labels 97.8%, no holes.
The energy itself still rates the table's answer above the solve's, even then: misread and misplaced labels charge it.

## Results

Held out by quarter and scored against the April 2000 table. The 33 data-error parcels (below) are left out.

| | Published (pipeline's current 2000 output) | **Joint solver (J7)** |
|---|---|---|
| Table district lines the map draws (count) | 73.6% | **96.3%** |
| Lines the map draws that are table lines | 51.8% | **96.9%** |
| Zoning labels whose district comes out right | 56.5% | **95.0%** |
| Holes (short breaks in a drawn district line) | 1,070 | **15** |
| Wrong-lot incidents (one connected mistake = 1) | 7,900 | **824** |
| Wrong lots | 51,063 | **8,625** |

- **Published:** `data/work/2000-digital-zoning/districts_raw.parquet` from a main checkout. It leaves 10,083 scored
  parcels without a code.
- **Labels:** labels whose reading names a 2000 district (1,880 of 1,892).
- **Holes:** missed table-line stretches up to 100 ft long, with the line found on two or more sides.

## Where it still fails

- **Misread labels in narrow strips.** A vertical "NC-1" between two heavy lines was read as RM-2, so the strip took its
  neighbours' codes. The letters of the 94 labels no text box covers can also read as walls (5 labels, 10 borders).
- **The two largest incidents.**
  - 2,020 RH-1 lots take NC-3 through a heavy printed line read as no wall (wall chance 0.08).
  - 2,011 RH-1 lots behind the Mission Street NC-2 strip take NC-2: a mid-block line read as no line lets the strip
    trade codes with the district behind it. With the true borders as walls, 1,844 of them still go wrong (they take
    NC-1), so there the labels (or their absence) decide wrongly too.
- **Big flips.** 7 incidents hold 5,327 of the 8,625 wrong lots.
- **15 holes remain (692 ft):**
  - 4 edges of larger wrong areas;
  - 3 lines printed beside the table's line;
  - 3 printed lines read as no wall;
  - 3 where the wall reads right but the energy prefers the step;
  - 1 label pulling a lot;
  - 1 that only counts as a hole once the data-error parcels are left out.
- **Tried and not adopted:** reading walls along whole borders (an ink model per border point, and learned walls
  coupled at junctions). It closed small gaps in the walls but did not improve the solve. That code stays on the local
  R&D branch `bubble-rnd/wall-field`, not in this PR.

## Where it needs manual checks

- **Data errors.** At 33 parcels the printed district line is drawn over the parcel, and the table puts the parcel on the
  other side. They are left out of every score. `holes --save-data-errors` writes the list; it is not checked
  in, because the inputs it is derived from cannot yet be rebuilt from this branch (below).
- **Three lines printed beside the table's line.** These may also be data errors; they need a person to decide.
- **Districts of 100+ lots whose code differs from their own labels' reading.** These are the big flips.
- **Narrow strips with a single label,** where a misread names the whole strip wrong.

## Not done

- Not wired into `sfzmaps` or the published tiles.
- **The inputs cannot be rebuilt from this branch.** The wall odds, label readings, sheet warp, merged page, lot
  grid and scored lots were made by R&D scripts (local branches `bubble-rnd/snapshot-w3` and `zoning-lines-rnd`,
  not pushed). The wall odds were fitted on per-sheet cleaned pages, and the scored lots come from an old R&D run.
  Making every input reproducible from the scans, the DataSF tables and the saved reads is in progress.
- **The scores use a corrected table.** At 888 lots in 216 groups, LLM helpers judged that the map prints the
  run's code, not the table's, and those lots count as the run's code (`k8/judge2/final2.csv`).
- The 101 hand-checked leader moves placement learns from are rows of `inputs/2000-digital-zoning/label_fixes.csv`
  that are not committed yet.
- Extra package for these scripts only: `PyMaxflow` (GPL-3.0, which can be combined with AGPL-3.0).
