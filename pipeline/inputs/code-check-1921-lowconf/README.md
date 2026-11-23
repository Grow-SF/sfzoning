# The 1921 least-sure lot reads

The lots of the 1921 map whose pattern read is least sure, least sure first, for the owner to correct the reads that
are wrong. There is no cutoff in confidence: the owner stops when the reads look right. No build stage reads this
folder. The 1921 measure (`sfzmaps.solver.answers1921`) and the hatched-map reader's lot table
(`sfzmaps.solver.hatchnet.data`) read its cases and answers.

| File | What |
|---|---|
| `cases.json` | The 1,000 cases, in the order the page shows them, and from what they were made (`sources`, each file's sha256). |
| `answers/` | The owner's 68 answers so far (all given 2026-11-18), one file per answered case, exactly as the page's store exports them. |
| `cases-retired.json` | The page's first 1,000 cases (`label-checks` dd900aa). 10 of the 68 answers are for cases retired since; an answer's case is looked up in `cases.json` first, then here. |

## Provenance

Made by `scripts/label_checks/build_label_checks.py --series 1921 --page patternlow --answered <ids>` on the `label-checks` branch,
from the `solver` branch at bb40bc4, its data as `pipeline/dvc.lock` there records it; no API calls. It reads:

- `data/work/1921/<sheet>/district_pattern.json`: the pattern reader's reads (`parcels`: blklot, code, confidence,
  via), on the 16 sheets the warp placed; reads with no tile behind them (`off_sheet`, `no_tile`) are left out;
- `data/out/1921/districts/parcels.fgb`: the published 1921 map, whose parcels are the lots the reader read (the
  outline, and `published`, the code it publishes);
- `data/warp/1921/<sheet>/transform.json`: each sheet's placement, to draw the lot on its page;
- `data/work/1921/<sheet>/page.png`: the scans (300 dpi, 1-bit), and section 1's legend.

## The cases

The reader read 180,044 lots, each on one sheet (a lot read on several sheets would show its least confident read,
with the others in `other_reads`). Condo lots stacked on one parcel shape (the same outline, on the same sheet) are
one case, its least confident read: 180,044 lots make 150,702 footprints, 3,916 of them stacks. A case's `blklots`
lists every lot of its footprint, and its answer applies to all of them. Every read would be gigabytes of crops, so
the page stops at 1,000 footprints.

Low confidence clusters by block, so the cases come in rounds over the blocks rather than in one list by confidence.
Round 1 is the least confident footprint of every block, ordered by confidence; round 2 is each block's second least
confident footprint, ordered by confidence; and so on until there are 1,000 cases. There is no fixed number per block.
The cases the owner had already answered stay first, in their order (`front`), and they count as their block's first
rounds. The page covers 976 blocks; past the front, confidence runs 0.50 to 0.65.

Case ids are stable, because the owner's answers are saved by id. A retired id (`retired_ids`) is never used again:
answers saved under it match no case, and the builder reads the previous cases file and keeps the list.

- The first version (2026-11-18) had one case per lot, in order of confidence. 110 of its 1,000 cases repeated an
  earlier case's outline. The second version kept the other 890 under their ids and in their order, retired the 110
  repeats, and refilled the page with the next footprints, ids c1001 to c1110.
- The third version (block rounds, `--answered`) kept the 50 answered cases still on the page at the front. It kept
  400 unanswered cases that stayed among the 1,000 under their ids, retired the 550 that dropped out, and gave the
  new footprints ids c1111 to c1660.

A case is the same view as the 1921 label check (`code-check-1921`): the lot outlined on a 256 px crop of its
sheet's page, two wider views of the stitched sheets, the read's district and swatch, and the printed legend; the
owner answers `ok`, `fix` (a legend district) or `canttell`. The page shows the confidence under "About this case".

Fields: `id`, `order`, `stratum` (`all`), `code`, `confidence`, `via` (block or parcel), `published`, `other_reads`, `blklots`,
`sheet`, `blklot`, `block`, `quarter`, `x`, `y` (EPSG:2227 ft, a point inside the parcel), `crop`, `outline`
(crop px), `sprite`, `slot`.

## The answers

The page saves one document per case in its store, collection `answers-1921-lowconf`, id = the case's `id`:
`{case, order, answer, code, shown, stratum, at}`, as on the 1921 label check.

## "unzoned" answers

For a while (2026-11-18) the 1921 pages also offered "Unzoned" (key N, saved as `{answer: "unzoned", code: ""}`) for
wilderness and water no district seemed to cover. The owner then ruled, as the 1921 ordinance says, that every part
of the city is in one of the six districts and white means 2R, and the answer was taken off the pages. Answers saved
as `unzoned` stay in the store: they mean "2R on the map; real-world water or wild land, for the display rule".
