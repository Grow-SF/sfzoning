# The 1921 likely-wrong lots

The cases of the owner's check of the 1921 lots that the hatching model thinks the old pattern reader got wrong.
No build stage reads this folder. The 1921 measure (`sfzmaps.solver.answers1921`) and the hatched-map
reader's lot table (`sfzmaps.solver.hatchnet.data`) read its cases and answers.

| File | What |
|---|---|
| `cases.json` | The 1,000 cases, in the order the page shows them. |
| `answers/` | The owner's 203 answers so far (all given 2026-11-18), one file per answered case, exactly as the page's store exports them. |

Made by `scripts/label_checks/build_label_checks.py --series 1921 --page patternlikely --likely
data/solver/experiments/hatch1921/next_cases.json` on the `label-checks` branch; no API calls. The cases are that
file's, in its order: one per block, ranked by the model's chance that the reader's read is wrong (1.00 down to
0.93). The file is made by `scripts/spikes/hatch1921/model.py` and its sha256 is in `sources`. Each case is one
footprint, with condo lots stacked on one outline in `blklots`; the builder groups footprints again.

The page shows the model's district (`code`, with `probs`, `p_model`) on the lot outlined on its read's sheet, in the
views of the 1921 pages. "About this case" gives the old reader's read (`reader_code`, `reader_confidence`) and
`p_wrong`. The owner answers `ok`, `fix` (a legend district) or `canttell`. Answers are saved in collection
`answers-1921-likely`, one document per case id: `{case, order, answer, code, shown, stratum, at}`, where `shown` is
the model's code.

## "unzoned" answers

For a while (2026-11-18) the 1921 pages also offered "Unzoned" (key N, saved as `{answer: "unzoned", code: ""}`) for
wilderness and water no district seemed to cover. The owner then ruled, as the 1921 ordinance says, that every part
of the city is in one of the six districts and white means 2R, and the answer was taken off the pages. Answers saved
as `unzoned` stay in the store: they mean "2R on the map; real-world water or wild land, for the display rule".
