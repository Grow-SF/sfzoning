# Finder-to-placement checks (2026-11-25)

One-off checks behind `docs/research/finder-to-placement-trial-2026-11-25.md` and
`docs/research/finder-to-placement-handoff-2026-11-25.md`. They are not pipeline code, were run from `pipeline/`
against the data of the day, and are kept so the numbers in the notes can be traced. They need `pipeline/data/`
(the all-pieces trial run), so they will not run on a fresh clone.

| Script | Question | Result recorded in the trial note |
|---|---|---|
| `rough_hint.py` | Does the rough placement of 1921/094348509 fail because of the scale hint the finder's blocks give? | Yes: with the finder's hint (4.82 ft/px) a false group of 7 label matches beat the right group of 6; the traced hint (3.81) gave the right fit. |
| `lost_areas.py` | Of the finder's areas on a trial sheet, which become blocks, which are cut at the rim, which do the outline checks drop? | Areas touching the rim held 41% (Section 12), 33%, 52% and 69% (p08) of the not-street ground; specks cleaned away were 0.0 to 0.1%. |
| `blocks_fate.py` | Of the year's blocks on a trial sheet, how many does the finder give alone, joined, or lost at the rim? | The finder's blocks join where the read breaks a street and are lost at the rim (see "What the finder's blocks lose"). |
| `make_root.py` | (rough-check step 1) Build a trial root with the finder's blocks and the traced arm's rough placement copied in. | 094348509 from the right rough start: 633 control points against today's 79, check passed. |
| `mark_rough.py` | (step 2) Record the copied `rough.json` as fresh so `sfzmaps run` takes it as current. | Part of the same check. |
| `proofs.py` | (step 4) Prove the rough-check's inputs are byte-equal to the arms', and print its numbers. | Blocks file sha equals the finder arm's; `rough.json` byte-equal to the traced arm's. |
| `snap.sh` | Did the check leave today's data folders untouched? (mtime and size listing, run before and after and diffed.) | Today's data was untouched. |
