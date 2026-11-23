# What the `hatch-cnn` branch holds: an audit from the code (2026-11-23)

Asked for by the owner ("audit the work we've done in this branch (line by line!) and summarize what we've done and
what the current status is"). Five readers went through all 35 files the branch adds to `main` (10,577 lines) line
by line, checking every docstring against the code under it. The claims marked **[checked]** were then re-read in
the code, or re-measured on the files on disk, by the session that wrote this note. The branch's own tests were run
(102 passed, one expected failure now passes). No training, prediction or pipeline run was made for this.

Paths are under `pipeline/src/sfzmaps/solver/`: `D` = `hatchnet/data.py`, `T` = `hatchnet/train.py`,
`P` = `hatchnet/predict.py`, `A` = `hatchnet/areas.py`, `H` = `hatching.py`, `F` = `hatchfit.py`,
`S` = `answers1921.py`. `EXP` = `pipeline/data/solver/experiments/hatch-cnn/` (git-ignored).

## Where things stand

- **Branch:** 34 commits ahead of `main`, 0 behind. Two commits are not pushed: the approved design (`6dbfbe9`) and
  the first look at areas (`810d5cf`). Draft PR #70 is "not for merge"; its tests pass at `243af8c`.
- **Worktree:** `.claude/worktrees/saved-reads` is the only checkout of the branch and holds its data (`EXP`,
  3.6 GB; the 16 frame cuts of 1921).
- **The approved design** is `docs/superpowers/specs/2026-11-22-print-decides-parcel-splits-design.md`: read the
  printed areas first, give each area a district, lots last. Of it, only the first look at areas exists (`A`,
  131 lines). Everything else on the branch belongs to the two designs it replaced.
- **Last word from the owner (11-22 evening),** on the areas pictures of Section 12: the street ends fan out past
  the thin printed line that closes them; "we're back to street finding."
- **Open question, never answered:** build the street-and-block finder here and hand it to placement (#69), or
  build it under #69. The placement notes of 11-23 record the owner on PR #70's finder becoming the one finder:
  "probably" (try it and show him).
- **No reading by the network has ever been scored.** The only scores that exist are: today's published map
  199 of 240, the 2000-style solve 201 of 240, and the abandoned patch model 219 of 240 (11-18, before the frame
  cut and the duplicate list).

## The findings that matter most

1. **The sheet-ownership rule is still the core of the data builder.** Where two scans show the same block, the
   scan with more ink over the block owns it (`D:296-336`, `D:852-857`; called "R9" in the code) **[checked]**.
   PR #70's description says this rule is wrong, and the 11-19 and 11-22 designs both say there is no ownership
   rule. In the data built on 11-22, 857 blocks go to a scan other than the top one, and Section 12 gives up
   2,152,854 block px to its neighbours **[checked]**. Every raster, label, prediction and the areas measure pass
   through it.
2. **Every street finder on disk was trained on the sheets it is then judged on, with both duplicate scans and the
   old crop.** Each of v1, v2, v2c is one model over 16 sheets, both duplicates of Section 12 included
   **[checked]**. One 1,024 px cell in five is "held out" (`T:211-213`) **[checked]**, and those same cells choose
   the kept training step (`T:532`) and give the reported "held-out" numbers. v2 and v2c kept step 3,000, the last
   one, so the step budget chose them **[checked]**. The 11-19 debugging found these finders copy today's street
   map on ground they trained on (Alemany Blvd drawn across printed blocks 91% in trained cells, 17% deep in
   held-out ones; from the work log, not re-measured).
3. **The areas measure of 11-22 cannot be trusted to rank finders.** (`A`, read in full **[checked]**.)
   - It was taken on Section 12, a training sheet of all three finders, with no split by held-out cell.
   - A block the finder calls all street drops out of every count (`A:48-84`); a finder that calls everything
     street scores no merges and no shattered blocks. The block counts on disk differ by finder (525, 524, 522).
   - Only blocks this scan owns under finding 1 are counted (`A:112`).
   - Open land that 1921 really draws as one area is counted as a merge, because today's blocks divide it.
   - An area is decided by a hard cut at one half on single px (`T:640`), and "v1 or v2c" is the larger of two
     models' outputs (`A:117`).
   - The command only works on Section 12: a Section 12 pixel window is cut from whatever sheet is given
     (`A:100`, `A:124-125`), which is empty on the 12 sheets that are 2,550 px wide; `table.json` is overwritten by
     each run.
   - "v2" in that table is the model with the scale input (the work log's "v2a"), which the code elsewhere says
     treats each sheet differently **[checked]**.
4. **Every street and block label is today's map, not the print.** Blocks are April-2000 lots filled in; streets
   are corridors round DataSF centre lines (`D:660-662`, `D:781`). Ground inside one of today's blocks is "not
   street" at full weight even where 1921 prints a street. One loss term penalises a predicted street anywhere
   outside today's corridors (`T:414-415`). Nothing labels a printed block outline, a double-line road, or the
   line that closes a street end.
5. **The district reader reads lot by lot,** which is what the 11-22 design replaces: `T:853-1496` and all of `P`.
   Its nine saved models date from 11-19 (old crop, duplicates in). The 11-22 district pictures applied them to the
   new data; the only check tying a model to what it predicts is its fold name (`P:204-212`).
6. **The owner's answers exist only in a git-ignored folder on this machine.** 671 answer files for 1921 (and the
   1960 ones) are in `pipeline/data/solver/experiments/label-checks/answers/` (`S:39`) **[checked]**. They are on
   no branch of the remote. The case files they refer to were exported by hand from the `label-checks` branches,
   which are local only **[checked]**. With the folder missing, the data builder writes a lot table with no
   answers and raises nothing. *(Done after this audit, 11-23, on the owner's yes: the answers and case files are
   checked in under `pipeline/inputs/code-check-<page>/` and the loader reads them there. The line numbers in this
   finding are from before that change.)*
7. **What the 199 of 240 is.** On the random page, the code shown to the owner was the published map's, and the
   truth is "the shown code" when he pressed OK (`S:91-92`). So 199 is the count of OK answers and 41 the fixes
   **[checked]**. Every other reader is judged against answers given while looking at the published code.
   - An answer the reader gives no code for leaves the count (`S:181-186`) **[checked]**, so giving no answer on a
     hard lot raises the share.
   - The interval is about 5 points either way. It resamples by block, but the 240 answers sit in 235 blocks.
   - Of the 240: 121 are 2R, 87 1R, 22 C, 6 HI, 4 LI, 0 U. It says almost nothing about the industrial patterns.
   - The scorer's own command pools all 661 answers; the "240 random" row is made only inside the abandoned patch
     model (`F:503-519`). The score command the plan promised for the network was never written.
8. **The commands cannot rebuild what is on disk.**
   - `streets` writes to `streets-v2/`; `districts` starts from `streets-v2c/` (`T:856`), which no documented
     command writes **[checked]**. A default `streets` run today would overwrite the v2 model on disk with a
     different one.
   - Output folders carry no series name (`T:755-757`, `P:50`), so a second series overwrites 1921's models.
   - `districts/summary.json` merges earlier runs under the newest settings (`T:1421-1428`).
   - Nothing here is a build stage or an `sfzmaps` command **[checked]**. The frame cuts it needs come from a
     hand-run `sfzmaps frame`, also not a stage.
   - The comparison script `scripts/spikes/hatchnet_r9_compare.py` reads a folder that no longer exists.
9. **Things are left out without a word.**
   - A map sheet with no frame cut is dropped from the sheet list before the refusal in `load_crop` can fire
     (`D:238-245`, `H:786-795`) **[checked]**. The plan's test covers the function, not the run.
   - Training and prediction read every `*.npz` in the data folder (`T:780`, `P:179`) and the data builder never
     clears old ones, so a raster left from before the duplicate list would be trained on. The folder is clean
     today (14 files) **[checked]**.
   - `data --sheets` overwrites the lot table with counts from those sheets only (`D:1242-1257`).
   - A series with no `duplicate_scans.csv` has no duplicates (`H:733-741`) **[checked]**; only 1921 has one, and
     the "kept" column is never read.
10. **The old patch pipeline is still on the branch and is dead.** `F` (555 lines) has no caller. Of `H`'s 1,223
    lines, 117 are used by the network's data builder (the frame cut, the sideways test, the duplicate list, the
    page-to-grid lookups); the other 1,106 are the pipeline the owner abandoned on 11-18. It also used the paid
    reader's 1R reads to set each sheet's patch size (`H:810-817`), which is LLM output at inference.
11. **Picked numbers that decide something.** The main ones: the one-half cut that makes areas (`T:640`); a 95%
    test for whether a street type keeps its own widths (`D:399`); the three street-loss terms added with weight 1
    each (`T:392`); the start value 0.25 of the two label-noise rates, which the code's own comment says the
    result depends on (`T:135-148`); 3,000 training steps; held-out cells of 1,024 px, one in five; a lot is
    "held" when over half its px are (`T:1049`); "wider than tall means scanned sideways" (`H:498-511`), a 1921
    fact **[checked]**.
12. **Docs point at things that are not in the repo.** The rulings the code cites (R3', R4, R5, R6, R8, R9) are
    defined only in `.superpowers/sdd/`, which is git-ignored; two labels are used for two different rulings each.
    The specs cite `scratchpad/...` notes and `/tmp/rt3_areas.png`; one of those notes
    (`streetdbg/findings.md`) is already gone. Stale docstrings: "the owner's answers are ... never training
    data" (`S:12`; the network trains on them, by the owner's 11-18 word), "one per cell" for the read-error table
    (`T:56-57`), "each lot's px are pooled on one sheet only" (`D:21-22`; 1,886 lots are on two to five sheets).

## What was built, in order

| When | What | How it ended |
|---|---|---|
| 11-18 day | The measure on the owner's answers (`S`); hand-built patch measures (`H`) and a patch model per lot (`F`) | 219 of 240. Abandoned by the owner the same day: hand rules, no pictures shown |
| 11-18 evening | The learned reader: per-sheet training data (`D`), a small network, a street finder trained on today's blocks (v1) | Owner: "aren't quite learning the proper street-block separation" |
| 11-18 night | Street finder v2: corridor labels with learned widths, a label-noise model, a connectivity loss (`topo.py`); v2c drops the scale input | Stopped after three versions: on trained ground they copy today's streets |
| 11-19 night | District training through lot-pooled losses, nine models; lot predictions (`P`); the ownership rule R9 | No score computed. R9 found wrong the same morning |
| 11-19 | Owner: one process for every map. Draft design; issues #68, #69; draft PR #70 | This reader parked |
| 11-22 | Frame cut adopted (#73, #75); duplicate scans dropped; district pictures on three scans | Owner: why do 2000 parcels override a 1921 map? |
| 11-22 | Design "read the print, then the lots", approved after three red teams; first look at areas (`A`) | Owner: street ends fan out; "we're back to street finding" |

## What each file does

**`hatchnet/data.py` (1,434 lines).** Builds the training data per scan.
1. Lists the sheets: a page, a placement, a frame cut, less the duplicate list.
2. Makes today's blocks from the solver's lot grid; measures each scan's ink over each block and assigns each block
   to one scan (finding 1).
3. Makes street labels on the 4 ft grid from DataSF centre lines, with half-widths learned per street type.
4. For each scan, reads lot, block and street labels at every page px through the placement in `data/work`, blanks
   what the scan does not own, and turns a sideways scan upright.
5. Builds the lot table: the paid pattern reader's reads (training labels), the owner's answers, the legend
   swatches (hand-entered boxes in `districts/legends.json`).
6. Writes `EXP/data/<series>/<sheet>.npz`, `lots.parquet`, `summary.json`.

Reads: `data/work/<series>/<sheet>/{page.png, transform.json, district_pattern.json, score.json}`,
`data/ref/{streets,parcels}.parquet`, `data/solver/<series>/lots/`, `data/frame/<series>/<sheet>/outside.png`,
`inputs/<series>/duplicate_scans.csv`, and the answers and case files of finding 6.

**`hatchnet/train.py` (1,496 lines).** Lines 1-850 train the street finder (v1: cross-entropy against today's
blocks; v2: the label-noise likelihood, plus connectivity recall against today's centre lines, plus connectivity
precision against today's corridors) and write `street_prob` per sheet. Lines 853-1496 train the districts per lot:
a read-error table fitted from the random answers, lot-pooled losses, four folds by city quarter with and without
answers, and one model on everything. Fold q really is kept out of its labels, its read-error table and its choice
of step; it is not kept out of the starting weights, which come from a street model that saw every quarter.

**`hatchnet/predict.py` (369).** Each lot's district from the fold that never saw its quarter; per-lot pictures
and the contact sheet of the 240 random lots.

**`hatchnet/areas.py` (131).** Areas from a street finder's output, the merged and shattered counts against
today's blocks, a picture (finding 3).

**`hatchnet/model.py` (102), `topo.py` (67), `look.py` (469), `__main__.py` (159).** The network (a five-level
U-Net, 16 channels wide, outputs: street, six districts, a pooling weight); the connectivity loss; pictures; the
command line (`data`, `streets`, `districts`, `predict`).

**`answers1921.py` (318).** Loads the owner's answers (used by `D`) and scores a per-lot map against them (used
only by its own command and by `F`). The scoring function takes any lot-to-code table, so it can score any reader.

**`hatching.py` (1,223), `hatchfit.py` (555).** The abandoned patch pipeline, apart from the 117 lines of `H` in
finding 10.

## Tests

102 pass; `test_theta_recovered` is marked as an expected failure and now passes (it cannot fail the suite either
way). No test checks numpy, scipy or torch themselves. Gaps that would have caught findings above: no test runs
`areas` on a second sheet; none covers a block swallowed by "street"; none runs the data builder with one sheet's
frame cut missing; none checks which side the read-error table is applied on (the test's table is symmetric);
`look`, `__main__` and the command chain have no test; the synthetic towns are all hatched 1R blocks with
axis-aligned streets and hand-picked sizes, so "a blank block beside a white street" is in no test.

## On disk (git-ignored, this machine only)

| Folder under `EXP` | What | Made |
|---|---|---|
| `data/1921` | Training data, 14 sheets, frame cut | 11-22 15:01 |
| `data/1921-neatline-2026-11-18`, `data/1921-r9` | The 16-sheet data the saved models trained on; the first R9 data | 11-18, 11-19 |
| `streets`, `streets-v2`, `streets-v2c`, `run1`, `run2`, `models` | Street models and maps (2.8 GB) | 11-18, 11-19 |
| `districts` | Nine district models | 11-19 |
| `predict`, `predict-frame` | Lot predictions on the old and the new data | 11-19, 11-22 |
| `areas` | The first look at areas | 11-22 17:55 |

Also local only: the work logs in `.superpowers/sdd/2026-11-18-*` (rulings, measured results), the debugging notes
under `/private/tmp/.../scratchpad/` (`streetdbg2`, `align`, `blocks1921`, `frame1921`, `edge`; access times
refreshed 11-23 so the nightly cleaner keeps them), and the branches `hatch-regions`, `hatch-regions-r16-parked`,
`hatch-depth15`, `label-checks*`.

## Not built

Street maps made by models that never trained on the ground they predict; labels for boundaries the print draws
that are not today's streets; texture evidence trained on ground with the area map as an input; one district per
area; divided areas; ties between neighbouring areas; split lots; a score command for any of it.
