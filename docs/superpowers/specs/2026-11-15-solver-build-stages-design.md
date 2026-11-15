# The 2000 joint solver as `sfzmaps build` stages

Design, 2026-11-15. It carries out the reproducible-rebuild plan's parts C to G
(`docs/superpowers/plans/2026-11-13-reproducible-rebuild.md`), adjusted to what the owner decided after that plan was
written. The owner asked for this to be designed and built autonomously ("use superpowers to autonomously brainstorm
and implement with SDD the solver work"). Every decision below that the owner did not make is marked **(my call)**.

## Why

The joint solver (PR #57, run "J7") reads the district of every lot from the 2000 zoning map better than anything
before it. Held out by quarter, with the 33 data-error lots left out:

- table district lines the map draws: 96.3%;
- lines the map draws that are table lines: 96.9%;
- zoning labels whose district comes out right: 95.0%;
- holes: 15;
- wrong-lot incidents: 824;
- wrong lots: 8,625.

But none of its inputs can be rebuilt. They were made by R&D scripts on local branches that were never pushed, from an
old warp, a cleaned page and an old solver run, and they sit in one person's git-ignored folder. The owner wants
everything "fully reproducible from a fresh clone of the repo" with "deterministic outputs (small epsilon
notwithstanding)". They also want `sfzmaps build` to be "a dependency chain checking if it needs to re-detect text,
re-warp, and resolve". The sheet warp is now such a stage (PR #58). This design adds the rest of the chain.

## Goals

- Every input the solver reads is made by code in the `sfzmaps` package from fixed inputs only:
  - the scans and the pipeline's sheet pages;
  - the warp (`data/warp`);
  - the April 2000 table, checked into git;
  - the saved model reads (`pipeline/saved-reads`);
  - decisions people made, in git under `pipeline/inputs/`.
- `uv run sfzmaps build` runs the chain and reruns only what is stale: merged page → lots → prints → text detector →
  labels → walls → solve.
- The same answer every run, within each output's stated epsilon.
- Each part is judged by its own measure, and the solve by the six measures above. Every change from J7's definitions
  is listed with its effect.

## Non-goals

- Publishing anything. The published maps don't change. Nothing replaces what they use without the owner's yes.
- Other years. The owner's longer aim is one model learned from the 2000 table and applied to every year. Here, only
  the 2000 zoning map is solved. Code takes the series as a parameter only where that costs nothing (the merged page).
- Matching J7's files. J7's numbers are the yardstick to report against, not a test.

## Anti-goals (a failure even if it works)

- Any step reads `pipeline/data/rnd`, a scratch or `/tmp` folder, or another worktree. A fresh clone must suffice.
- A new paid or model read. Saved reads are fixed input; nothing asks a model anything.
- The cleaned page, or the old chain of graph steps, in any form.
- A fit that saw the quarter it is scored on.
- New per-item freshness stamps (`*.inputs.json` sidecars). The owner called them footguns (issue #61). The new stages
  rely on DVC alone: a stage reruns whole when any of its inputs or its code changes.

## Constraints (from the owner's decisions; see the decisions log)

- No text removal anywhere. Text is evidence. Walls are read from the original scan, with text discounted by our own
  detector (E7, agreed 11-15). The detector reads each sheet's original, unwarped page (D5).
- Probabilities, not cutoffs. A cutoff carried over from J7 is named where it stays.
- No priors or areas from earlier solves. Fits use the table, held out by quarter.
- The owner's hand labels (`inputs/2000-digital-zoning/label_fixes.csv`, PR #60) include the 101 leader moves that
  placement learns from.
- The judge2 table corrections stay in the solver's score for now (D1, 11-15). They are 888 lots in 216 groups where
  helpers judged that the map prints another code than the table.
- PyTorch and PyMaxflow are regular dependencies. PyMaxflow is GPL-3.0, which can be combined with AGPL-3.0.
- Run with pixi's tools (`pixi run build`), as every stage does.
- Determinism, as the plan states: seeded, deterministic algorithms, pinned thread counts.

## The chain

```
data/warp/2000-digital-zoning ─┐
data/work/2000-digital-zoning ─┼─► solver-page ─► solver-lots ─► solver-prints ─┬─► solver-labels ─┐
inputs/merged-page             ┘                       │                        │                  ├─► solver-solve
inputs/zoning-2000 ────────────────────────────────────┘                        └─► solver-text-{q} ─► solver-walls ─┘
saved-reads/answers ─► solver-prints, solver-text-{q}
inputs/2000-digital-zoning/label_fixes.csv ─► solver-solve
```

| Stage | Command | Writes | Main inputs |
|---|---|---|---|
| `solver-page` | `sfzmaps solver page` | `data/solver/page/` | warp transforms, sheet pages, cut boxes |
| `solver-lots` | `sfzmaps solver lots` | `data/solver/lots/` | the table, the page's sheet coverage |
| `solver-prints` | `sfzmaps solver prints` | `data/solver/prints/` | saved reads, Sol label reads, lots |
| `solver-text-nw` … `-se` | `sfzmaps solver text Q` | `data/solver/text-Q/` | sheet pages, Terra boxes (from prints), quarters |
| `solver-text-scores` | `sfzmaps solver text-scores` | `data/solver/text-scores/` | the four quarters' chances, lots |
| `solver-labels` | `sfzmaps solver labels` | `data/solver/labels/` | prints, lots, streets, merged page |
| `solver-walls` | `sfzmaps solver walls` | `data/solver/walls/` | merged page, text chances, lots |
| `solver-solve` | `sfzmaps solver solve` | `data/solver/solve/` | everything above, hand labels, table corrections |

- **Stage code deps.** A stage's code deps are the import closure of its entry module (`fresh.code_files`), not every
  file of `src/sfzmaps` as the old stages have. So editing the solver's search doesn't retrain the text detector.
  `pyproject.toml` and `uv.lock` are deps too, since a library change can change numbers.
- **Four text stages.** The detector is the slow step (about 1.7 h per quarter on the GPU in R&D), so there is one
  stage per held-out quarter. A build can start them together (each trains and scores in its own exclusive turn on
  the GPU: two processes on `mps` at once get wrong numbers), and changing one quarter's input reruns only that
  quarter. Their scores on the held-out quarters are one more stage, `solver-text-scores`, so the detectors' code
  holds no lots, table or page code. **(my call)**
- **Code and data.** The solver moves from `pipeline/scripts/joint_solver/` into `sfzmaps.solver` (one subpackage,
  one CLI group `sfzmaps solver ...`). The `SFZ_RND` folder goes. Its R&D notes move to
  `docs/research/joint-solver/`. **(my call)**

## The parts

### Fixed inputs (plan A2–A4)

- **The April 2000 table** (A2; owner: "download it from datasf and check it in alongside a provenance file").
  - Dataset `cfhj-8c6b`, "Historical Zoning Districts - April 2000", SF Planning, license PDDL, "Not updated
    (historical only)".
  - `sfzmaps solver fetch-table` downloads the whole table as GeoJSON in one request (the export endpoint) into
    `pipeline/inputs/zoning-2000/april2000.geojson.gz`. Gzip keeps the downloaded bytes exactly: about 98 MB raw,
    near GitHub's limit. **(my call)**
  - It writes `pipeline/inputs/zoning-2000/PROVENANCE.md` with:
    - the URL, dataset id, name, publisher and license;
    - the fetch time;
    - the row count;
    - the sha256 of the uncompressed bytes;
    - the command.
  - Check, reported once: the fetched table equals the R&D download (`april2000.geojson`) row for row (blklot, zoning,
    geometry).
- **The table corrections** (D1): `k8/judge2/final2.csv` checked in as `pipeline/inputs/zoning-2000/table-corrections.csv`.
  Its README note says what it is: helpers' verdicts on crops, verdict A means the map prints column A. It is used
  only in the score. **(my call: fits use the uncorrected table, since the corrections came from judging an old
  solve's disputes; J7 used them in three fits, which is listed as a definition change.)**
- **The cut boxes** (A3): `pipeline/inputs/merged-page/2000-digital-zoning.json`. These are the 6 page-px boxes `zcuts.py`
  made from regions picked by eye:
  - p01: `[163,3830,713,3991]`, `[2898,3737,3146,3954]`, `[2228,526,3118,641]`;
  - p02: `[118,505,1008,631]`;
  - p08: `[2223,555,3124,675]`;
  - p09: `[2287,3874,3178,3995]`.

  Each box gets a note saying what it covers (title, legend, etc., found by looking).
- **PyMaxflow** (A4), a regular dependency.

### Merged page (plan C)

- **What it is:** a port of `zmerge.py` (`bubble-rnd/snapshot-w3:pipeline/scripts/bubble-rnd/k8/`), reading the
  warp's corrected sheets. It is a union of ink, as J7's `raw.png` was. The probabilistic merge in
  `sheet-merge-design.md` is not built. **(my call: the plan chose the union, and J7 ran on it.)**
- **Grid:** EPSG:2227 feet, 4 ft/px, 11,900 × 9,825 px, top-left corner (5979300, 2124100), north up (row = (2124100 −
  y) / 4). The lots and walls use the same grid.
- **Per sheet** (the 13 placed 2000-digital-zoning sheets):
  - The crop is the frame quad minus the cut boxes. The frame is found on `page.png` by a port of `zframe2.py`
    (measured, inset 4 px as J7's saved frames were, which keeps the legend box's top line out; its search windows
    stay as named constants).
  - The page is sampled through `SheetTransform.from_json(data/warp/.../transform.json, folder=...)`: `to_pixel`
    every 8 grid px, bilinear between, then the nearest page px.
  - Ink is page < 128 inside the crop.
- **Outputs:**
  - `ink.png`: 0 = ink, 255 = paper;
  - `sheets.png`: the number of sheets whose crop covers each pixel. **(my call: J7's `nsheets` counted the cleaned
    ink of each sheet; coverage is what "mapped area" needs.)**
- **Epsilon:** byte-identical.

### Lots, mapped area, quarters (plan E1–E3)

- **The table:** read from the checked-in file and put in EPSG:2227.
  - `zoning` is stripped and upper-cased, the geometry made valid. Row order is file order, and that order is the lot
    index everywhere.
  - Code spellings use `joint_solver/codes.py`.
- **The lot grid** (a port of `truth.py`, on the page's grid):
  - `lot`: polygon exteriors filled largest first;
  - `grown`: every pixel takes its nearest lot;
  - `pairs`: 4-neighbour pairs of `grown`, a < b, with touching px (both pixels inside lots) and street px.
- **Mapped area:** `sheets.png > 0`.
- **Scored lots:** a known code (non-blank, no "/") and more than half the lot's pixels in the mapped area.
  - J7's definition also needed "inside a closed area of an old run". That can't be rebuilt, so it goes.
  - Report: the count against J7's 152,351, and the lots that differ by kind.
- **Quarters:** the city splits east–west and north–south at two constants.
  - The constants are J7's own split: the median centroid of J7's 152,351 scored lots, x = 6000180.818639896 ft and
    y = 2101727.4853113447 ft (EPSG:2227). **(my call)**
  - Why J7's values: every held-out number here (wall log-loss, detector, solve) is then comparable with J7's, and
    J7's held-out fits can be scored on new pairs without leaking.
  - They are two numbers that cut the city into four, not evidence or a prior.
  - The stage logs the median of the new scored lots beside them.
- **Outputs:** `lots.npz` (`lot`, `grown`, `pairs`, `scored`, `quarter` per lot, `known`, `table` codes) and `table.parquet`.
- **Epsilon:** byte-identical.

### Prints: every printed zoning label and its reads (plan E4, first half)

- **Inputs:**
  - the Terra boxes (kind "zoning code"), decoded from `saved-reads/answers/text-on-every-sheet.jsonl.gz` exactly as
    `saved_reads/scripts_as_they_ran/textboxes.py` decodes them;
  - Sol's label points (kind "use") from `data/work/2000-digital-zoning/<sheet>/district_labels.json`;
  - the turned pass, `experiments-2000/text-positions/sideways-text`;
  - the third reads, `third-code-read` and `third-code-read-recheck`.
- **Ports:**
  - `labread2.py`, first half only (matching boxes to points; prints kept inside the sheet's warped footprint and the
    mapped area; the lot under the text; in a street or not);
  - `read3.py` and `read3r.py`;
  - the print-joining half of `addprints.py`: turned boxes join a print by containing its point, or by IoU > 0.3 with
    its Terra box (a named cutoff, kept). A turned box matching no print, with a known code, becomes a new print.
- **Third reads by cache key.** The saved third reads carry no image or sheet. They are lined up by rebuilding each
  crop and prompt exactly as `read3.py`/`read3r.py` made them and computing the cache key.
  - A key that doesn't match is reported. That print then has no third read, rather than a guessed one.
  - If fewer than 95% match, the builder stops and reports instead of going on. **(my call)**
- **J7's readers only, on purpose** **(my call)**:
  - Not used in this rebuild:
    - the red-circle code reads, which the saved-reads README recommends from now on;
    - the owner's 743 code-check and 50 leader-check answers;
    - the other 247 rows of `label_fixes.csv` (code fixes and deletions).
  - The rebuild first has to reproduce J7's quality from fixed inputs. Then each of these is added as its own measured
    step, with the human answers as held-out checks or as inputs (the owner decides which).
  - The report lists them as next steps.
- **Positions:** page px → map feet through the corrected warp transform.
- **Outputs:** `prints.parquet` (one row per print: sheet, page and map position, kind, every read with its code and
  confidence, the lot under its text, in a street), and the decoded Terra boxes of every sheet (`terra_boxes.json`,
  which the text stages read).
- **Reported against J7:** the count (1,906), the share of prints within 4 ft of a J7 print, and agreement of reads.

### Text detector (plan D)

- **What it is:** a port of `joint_solver/textnet` with D5: it trains on each sheet's original page. Sheet pages are
  also about 4 ft/px.
  - Input: the ink mask.
  - Targets: the Terra boxes in page px.
  - "Unknown" (no loss): the other text passes' boxes and 30-px discs at Sol label points, minus anything inside a
    Terra box.
  - Each page pixel's quarter comes from the corrected `to_world`. Held-out pixels carry no loss, and crops are
    centred only on training pixels.
- **Model and training: as R&D.**
  - A U-Net of width 24 and 5 levels, with 7 outputs (text, 4 kinds, reading angle).
  - The same losses.
  - Batches of 12 crops of 320 px, half on text; scale 0.6–1.6 and quarter turns.
  - AdamW at 2e-3 with OneCycle, 8,000 iterations.
- **Determinism first.**
  - A short spike: train 200 iterations twice on the GPU (`mps`) and twice on the CPU, with
    `torch.use_deterministic_algorithms(True)` and pinned threads. Compare the weights and the predictions, and time
    an iteration.
  - The device that gives predictions within 0.01 between runs is used.
  - If only the CPU does, and a quarter would take over 8 h, the builder reports and asks the owner to choose
    rather than lowering the iterations. **(my call)**
- **Outputs per quarter:**
  - the model;
  - the text chance of every sheet's page (uint8 PNG per sheet), predicted by that quarter's model;
  - `evaluation.json`, scored on its held-out quarter: held-out text found by kind, and the share of table
    district-line ink called text. R&D for NE: 94.9% of words found, 0.55% of district-line ink.
- **Epsilon:** predictions within 0.01 between runs.

### Labels: trust, street sides, lot size, strokes (plan E4 second half, E5, E6)

- **Trust:** a port of the fitting half of `addprints.py` in its `K3=pqf` mode.
  - It is a logistic regression on rows of (print, candidate code); the target is that the candidate equals the table
    code of the lot under the text.
  - Fitted by Newton with ridge 1e-3. The distance and in-a-street weights get ridge 1000 and the single-letter
    weight 1e-3.
  - There is one fit per held-out quarter, trained on prints outside it.
  - The 6 place-answer features were all zero in J7 and are dropped.
  - Output per quarter: every reading (print, code, trust).
- **Street sides:** a port of `labside.py` per held-out quarter.
  - Inputs: the nearest street centreline (`data/ref/streets.parquet`), the far-side lot and the four fitted rates.
  - The uncorrected table is used **(my call, as above)**.
- **Lot size:** a port of `zlotchar2.py` per held-out quarter.
  - Log-area bins `arange(5, 14.01, 0.25)`, each blended with half its neighbours, α = 36.
  - The uncorrected table is used.
- **Strokes for leader lines** (E5): a port of `zstrokes.py` on the merged `ink.png` (raw, nothing erased), in place
  of the cleaned page.
- **Outputs:** `readings_{q}.npz`, `streetside_{q}.npz`, `lotsize.npz`, `strokes.npz`, in the shapes the solver's
  `labels.py` and `placement.py` read today.
- **Epsilon:** identical on one machine (numpy fits, fixed order).

### Walls (plan E7): the design note

- **What it is:** per pair of neighbouring lots, the log odds that a printed district line runs between them. Learned
  from the table (the target is that the two lots' table codes differ), and held out by quarter.
- **Evidence:** read from the merged `ink.png`. Each ink pixel is weighted by (1 − text chance).
  - The text chance comes from the detector that held out the pair's quarter.
  - It is sampled onto the grid through the same mapping as the merge, taking the largest chance among the sheets
    that inked the pixel.
- **Features:** hat bases with fitted weights. No feature is a decision.
  - **Ink along the border:**
    - weighted ink at offsets 0–16 ft from the border (touching lots; R&D measured the printed line's median offset
      at 4 ft, 90% within 11 ft);
    - its width and continuity along the border;
    - the share of nearby ink that is text.
  - **Across a street:** ink over the whole street width, the street edges included. The chain looked only near
    the middle, so a line printed along an edge counted against a wall there.
  - **Enclosure:** whether text-discounted ink separates the two lots' majority regions, at a small grid of stroke
    widths. This was the largest single gain in R&D, with held-out log-loss .052 → .046.
    - The city limit is a wall at every level (owner's rule).
    - The levels are features, not decisions.
  - **Context:** touching kind (back to back, side by side, between) or street class, and lot character (the
    difference in log area, the smaller log area).
- **Model:** ridge logistic per held-out quarter, trained on pairs with both lots known outside the quarter.
  - No whole-wall stage: coupling along a wall is left to the energy.
  - No hand weights.
- **Gate 1:** held-out log-loss per quarter, scored on the pairs the new lots and J7's share (matched by lot index;
  the table's row order is the same), for both the new walls and J7's `zodds_Q1x{q}`.
  - J7's odds on those pairs are the target. R&D's raw page without the cleaned page (.0484 / .0795 / .0370 / .0597
    on J7's pairs) is context.
  - **On a miss** **(my call):**
    - The loop works through the feature list. Each feature is kept only if it lowers held-out log-loss in at least
      three quarters.
    - Then the solve runs whatever gate 1 says, and the report gives both numbers. Gate 2 decides.
    - The loop never stops the chain, and it never runs past the feature list without a new idea written down first.
- **Gate 2:** the solve's six measures against J7. Breaks are looked at before anything is rejected
  ([[errors-stack-dont-discard]]).
- **How it is developed:** as R&D in the principled loop. Each feature is added only when it improves the held-out
  log-loss, and each step's number is recorded. The note's approach is the owner's (11-13, 11-15); its feature list
  is **my call**.

### Solve (plan F)

- **Port:** `joint_solver` energy, search, solve, measures, borders, holes, placement, labels and published, unchanged
  in method, reading the new stages' outputs.
- **Placement:**
  - It learns from the 101 hand leader moves (`label_fixes.csv` rows with a fix point) and the table.
  - Its walled areas come from the new walls (odds ≤ 0, a named cutoff from J7).
- **The four held-out solves** run side by side, then the join, the six measures and the data-error list (F3). The
  data-error list is written to `data/solver/solve/` and not checked in.
- **Report** (F1): the six measures against J7, with every definition change and its effect where it can be isolated:
  - the new warp;
  - scored lots without the old run;
  - fits on the uncorrected table;
  - the dropped zero features;
  - strokes from the raw page;
  - text-discounted walls.
- **Epsilon:** every lot's code identical between runs (F2).

### Fresh-clone check (plan G)

- **The script:** `pipeline/scripts/fresh_clone_check.sh STAGE...`. It:
  - clones the branch into an empty folder;
  - runs `pixi install`, `uv sync`, `sfzmaps pull`;
  - reruns the named stages (`sfzmaps build --force`) on the pulled upstream outputs;
  - compares each output with the shipped one within that stage's own epsilon.

  Each stage is held to its own epsilon on the same upstream outputs. The epsilons are not chained end to end.
  **(my call)**
- **When it runs:** before each PR of this work is called ready, for the stages that PR adds or changes.
  - For the text detector, one quarter (nw) is retrained, not four.
  - It runs on the owner's Mac (the GPU), as the warp's check did. CI runs the unit tests only.

## How it is judged

| Part | Measure |
|---|---|
| Merged page | ink agreement with J7's `raw.png` (reported); the seams were judged on the warp |
| Lots | pair and scored-lot counts against J7's (reported, differences explained) |
| Prints | count and read agreement with J7's 1,906 prints |
| Text detector | held-out text found by kind; district-line ink called text |
| Labels | held-out log-loss of the trust fit against J7's (same prints where they match) |
| Walls | held-out log-loss per quarter (gate 1) |
| Solve | the six measures against J7 (gate 2), each definition change listed |

## Order of work

1. Fixed inputs, `sfzmaps.solver` package (the move), the stage machinery, `solver-page`, `solver-lots`.
2. `solver-prints`.
3. The text detector, determinism spike first, then the four quarters trained in the background.
4. `solver-labels`.
5. `solver-walls` (R&D loop against gate 1).
6. `solver-solve` and the report against J7; then the fresh-clone check.

Parts 1–4 are ports with checks against J7's files; part 5 is new. One PR per step where it can stand alone, each
based on the one before.

## Left to the builder

- Module boundaries inside `sfzmaps.solver`, file formats beyond those named, and test fixtures.
- How the R&D scripts are simplified, as long as the method and named constants stay.
- Comparison scripts against J7's files are one-offs: they stay out of the package and out of git. They may read the
  R&D folder; package code never does.
