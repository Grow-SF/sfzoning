# Joint Solver Inputs Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every file the joint solver (PR #57) reads is either a fixed input or is made from fixed inputs by code on a
branch built on #57, through one entry point. Each rebuilt file is checked against the file the solver reads today.

**Architecture:**
- **Code:** a subpackage `joint_solver/inputs/` with one module per input, and one command,
  `python -m joint_solver inputs build|check NAME`.
- **Where outputs go:** rebuilt files go to `pipeline/data/rnd/inputs/`, beside today's `pipeline/data/rnd/{zon2000,k8,data_ay2}`.
  The solver gets one setting that chooses which folder it reads, so old and rebuilt inputs run side by side.
- **Ports:** a port keeps the method and must reproduce today's file.
- **Method changes:** each change you decided on is its own task. It is re-scored with the solver's held-out score and
  adopted only on your yes.

**Tech Stack:**
- The pipeline's Python (`pipeline/.venv`), with numpy, scipy, geopandas and opencv as in #57.
- PyTorch only for the sheet warp and the text detector. Its environment is still to be declared as a dependency.

**Spec:**
- Your decisions, listed below under "Decisions already made".
- `pipeline/scripts/joint_solver/joint-solver.md` (PR #57).
- `pipeline/scripts/joint_solver/notes/joint-solver-inputs.md`, the morning-of-11-13 inventory. It is partly out of
  date; this plan's table below supersedes it.

## Decisions already made (from your decisions log; nothing below re-proposes what these rule out)

- 11-11: no LLM question steps. Saved reads are fixed input; no new paid reads, and no prompt, image or model changes.
- 11-12: bootstrap from primary sources only. No priors or areas from earlier solves or old chain runs.
- 11-12: the sheet merge combines ink probabilistically. Priors and the mixture are learned from the table as one
  model, held out by quarter.
- 11-13: no cleaned page anywhere and no hard-cutoff whole-wall stage. Walls belong inside the solver.
- 11-13: Terra outputs stay as training data, each with provenance: model, prompt, input image, timestamps.
- 11-13: our own text detector is trained on Terra's boxes. Text feeds the solver and is not erased. Walls are learned
  from the original scan, never from the cleaned page or from Terra at use.
- 11-13: work goes on a branch built on PR #57, in the `joint_solver` package, with one entry point and sensible names.
- 11-13: no file made by a one-off or unpreserved script gets checked in or used.

## Global Constraints

- Fixed inputs are only:
  - the sheet scans (`page.png`, pipeline-made);
  - the DataSF April 2000 table download (`april2000.geojson`, data set cfhj-8c6b);
  - DataSF parcels and street centrelines;
  - saved LLM reads, replayed from the cache only;
  - inputs a person made, committed to git.
- Every replay of a saved read runs cache-only. A miss stops the step; `llm.CallCounter.live` must stay 0.
- Every built file gets a sidecar `NAME.provenance.json` with:
  - each input's path and sha256;
  - the git commit of the code and the command line;
  - the held-out quarter, for fitted files;
  - the build time.
- Every fitted step is fitted with its quarter held out, and every score names the held-out area.
  The quarter split is frozen (Task 0).
- Every compared run is scored with the solver's own `score` command and the same six measures:
  - lines found;
  - drawn lines right;
  - labels right;
  - holes;
  - wrong-lot incidents;
  - wrong lots.
- Evidence is modelled as probabilities, not hard cutoffs. An existing cutoff is ported as-is and listed in the task.
- No built file may contain a path under `/private/tmp` or the home folder.
- Code goes on the branch as it is written; data goes under `pipeline/data/rnd/`.
- Nothing replaces today's solver inputs without your explicit yes.

## Decisions needed from you (each one changes what gets built)

| | Question | Options | My recommendation |
|---|---|---|---|
| D1 | The table corrections my helper agents judged (`k8/judge2/final2.csv`: 888 lots counted as the run's code) | keep in the score / drop / a person checks them | **Drop them from the score**, and list the 216 groups as "needs a person". They are LLM verdicts inside the yardstick, which 11-11 rules out. |
| D2 | How built and rescued data is tracked | git-ignored folders (today) / DVC (`.dvc` pointers committed, data shipped with the rest) | **DVC** for the rescued originals, the trained models and every built input. |
| D3 | Your uncommitted `label_fixes.csv` rows (the 101 leader moves placement learns from, plus 266 others) | commit in their own PR / leave uncommitted | **Commit them.** They are a fixed input, and the solver can't be rebuilt without them. |
| D4 | Which lots are scored (today: lots inside an old city chain run's closed areas, "city5") | lots with a known table code mostly inside the mapped area (where any sheet's crop shows the map) / keep city5 | **The mapped area.** It removes the old run; the denominator changes, so J7 is re-scored on it for comparison. |
| D5 | Label-evidence features that come from old runs (path costs from `lotdist_run.npz`, trusts fitted on the west half, `relink3`'s boxes from an earlier solve) | reproduce exactly first, then replace one at a time / replace directly | **Reproduce first, then replace one at a time.** Reproducing proves we know how today's numbers were made; each replacement is measured alone. |

## What the solver reads today, and where each comes from

Today's files are in `pipeline/data/rnd/` (copied out of `/tmp` on 2026-11-13). The R&D makers are on the local branch
`bubble-rnd/snapshot-w3` (`pipeline/scripts/bubble-rnd/k8/`); none of the R&D branches are pushed.

| Input (path under `rnd/`) | Made by | From | What must change | Task |
|---|---|---|---|---|
| `zon2000/april2000.parquet` | `truth_prep.py` | the DataSF download (kept, `zon2000/april2000.geojson`) | port only | 1 |
| Terra first-pass boxes `k8/pNN/textboxes_gpt-6.1-terra.json` | `textboxes.py` | page + cache | Provenance record. Replay already proven identical (p01–p13, PR #58) | 2 |
| Terra turned pass `k8/pNN/textboxes2_*.json` | `textpass.py` | page + cache | replay + provenance | 2 |
| Sol label reads `district_labels.json` (+ `.answers`) | pipeline | pipeline cache | provenance record only | 2 |
| Third read `read3.npz` | `read3.py` | scratch data folders' caches (rescued: `scratch-2026-11-13/data_*/cache/llm`) | replay + provenance | 2 |
| Sheet transforms `data_ay2/work/2000-digital-zoning/*/transform.json` | the prob6 warp (PR #58) | pipeline fits + block lines | Finish PR #58's checks (plus, choice, warps). Correction paths point into `/tmp` | 4 |
| Frame crop `zon2000/zframe2.json`, cuts `zframe2_cuts.json` | `zframe2.py`, `zcuts.py` | page; cut regions picked by eye (constants in the code) | port only | 5 |
| Merged scan `k8/MALLP_c1/raw.png` | `zmerge.py` | pages + transforms + crop | port only (its cleaned-page output is dropped) | 6 |
| Lot grid `zon2000/truth_grid.npz` | `truth.py` | table + the grid frame of old run city5 | grid frame from fixed constants | 7 |
| Mapped area `k8/city/footprint.geojson` | an old city chain run | that run | from the sheet crops (D4) | 8 |
| Scored lots `zon2000/lotcode_run.npz` (`scored`; also sets the quarter split) | `lotcode.py` | old run city5's closed areas | D4 | 0, 8 |
| Label evidence `zon2000/{lotclf6,labread2,labside}_cwfwx{q}.npz` | `labread2`, `addprints`, `lotclf6`, `labside`, `relink3` + shell drivers | reads, transforms, lot grid, `lotdist_run.npz` and `lotcode_run.npz` (old runs) | reproduce, then D5 | 9, 10 |
| Leader strokes `zon2000/zstrokes_old.npz` | `zstrokes.py` | the old **cleaned** merged page | from `raw.png` | 11 |
| Lot-size fits `zon2000/zlotchar2.npz` | `zlotchar2.py` | table + scored lots | rebuild on the new scored lots | 12 |
| Your leader moves | you | uncommitted rows of `label_fixes.csv` | D3 | 3 |
| Wall odds `zon2000/zodds_Q1x{q}.npz` | the wall chain (strokes → … → logistic fit + trees, blended) | the old **cleaned** page + a Terra letter feature | replaced by walls inside the solver | 13, 14 |
| Table corrections `k8/judge2/final2.csv` | LLM helper verdicts | crops of an old run | D1 | 15 |

**Not finished:** the text detector, which feeds the new walls. Two of its four held-out models are trained (NE and
NW). It isn't wired into anything.

## Order

```
fixed inputs: 1 table, 2 reads, 3 your rows            (no dependencies; run first, in parallel)
0 freeze quarters  ─────────────────────────────────────── before anything fitted
4 transforms (PR #58) → 5 frame/cuts → 6 merged page → 7 lot grid → 8 mapped + scored → 12 lot size
                                  6 + 2 → 9 label evidence (reproduce) → 10 remove old runs   (own plan)
                                  6 → 11 strokes from raw page
                       6 + 2 → 13 text detector (finish) → 14 walls in the solver           (design, own plan)
everything → 15 re-score
```

Tasks 0–9, 11 and 12 are ports and checks, detailed below. Tasks 10, 13 and 14 are R&D. Each gets its own plan or
design note for your yes when it is reached. Task 15 waits for all the others.

## Review Focus

1. **`/tmp` paths inside data:**
   - Today's `transform.json` files name their correction files by `/private/tmp/...` paths, so the solver still
     depends on `/tmp`.
   - Expected: no built file contains such a path. Test in Task 0: `check_no_tmp_paths`.
2. **A fitted input that saw its own held-out quarter:**
   - This happened before with `zpairfit3`; the honest totals were about 600 lots worse.
   - Expected: every fitted builder takes the quarter and writes it to provenance. The `check` command refuses a
     fitted file without one. Test in Task 0.
3. **A cache miss turning into a paid call:**
   - Expected: replay raises on a miss and reports `live 0`. Test in Task 2: `test_replay_refuses_miss`.
4. **Files that differ in bytes but not in data, or the other way round:**
   - Expected: `check` compares values exactly (zero tolerance unless a task says otherwise) and reports which
     arrays differ and by how much. Test in Task 0: `test_compare`.
5. **A different `label_fixes.csv` than J7 used:**
   - Placement would change silently.
   - Expected: its sha256 goes into the provenance of every file placement makes, and a mismatch is reported.
     Test in Task 3.

---

### Task 0: Branch, entry point, checks, frozen quarters

**Files:**
- Create: `pipeline/scripts/joint_solver/inputs/__init__.py` (registry `BUILDERS`, `REFERENCES`)
- Create: `pipeline/scripts/joint_solver/inputs/compare.py`, `inputs/provenance.py`
- Modify: `pipeline/scripts/joint_solver/__main__.py` (subcommand `inputs build|check NAME...`)
- Modify: `pipeline/scripts/joint_solver/data.py:30-36` (`SFZ_INPUTS_ROOT`; frozen quarter split)
- Test: `pipeline/tests/joint_solver/test_inputs_core.py`, plus `pipeline/tests/joint_solver/conftest.py`, which
  puts `pipeline/scripts` on `sys.path` so the package imports

**Interfaces:**
- Produces:
  - `compare(a: Path, b: Path) -> tuple[str, str]`, returning ("identical" | "same data" | "DIFFERS", detail).
  - `write_provenance(out: Path, inputs: list[Path], command: list[str], quarter: str | None) -> Path`
  - `BUILDERS: dict[str, Callable[[Path], list[Path]]]`, where each builder takes the output root and returns the
    files it wrote.
  - `REFERENCES: dict[str, list[tuple[str, str]]]`, which pairs each built path (relative to `inputs/`) with today's
    file (relative to `rnd/`).
  - `QUARTER_SPLIT = (x_ft, y_ft)` in `data.py`: today's median centroid of the scored lots, frozen.
    `quarter_of(xy)` uses it from now on.

- [ ] **Step 1: Branch.** Create branch `solver-rebuild` from `text-detector` (that is #57 plus the detector and
  notes). Commit this plan to it.
- [ ] **Step 2: Write the failing tests.**
  - `test_compare`: three cases on tiny npz, png and json files in `tmp_path`. Same bytes give "identical"; a re-saved
    npz with the same arrays gives "same data"; one changed array element gives "DIFFERS", with the array name in the
    detail.
  - `test_check_no_tmp_paths`: a json file containing `"/private/tmp/x"` is reported.
  - `test_fitted_needs_quarter`: `check` on a fitted file whose provenance has no quarter fails.
- [ ] **Step 3: Run the tests.**
  - Run: `cd pipeline && uv run pytest tests/joint_solver/test_inputs_core.py -v`
  - Expected: FAIL (module missing).
- [ ] **Step 4: Implement `compare`, `write_provenance`, the registry and the CLI.** The compare logic is
  `prob-warp/check/compare_rebuild.py`'s (PR #58).
- [ ] **Step 5: Freeze the quarter split.**
  - Compute today's split from `rnd/zon2000/lotcode_run.npz` with the current `quarter_of`, and write it as
    `QUARTER_SPLIT`.
  - Check: every lot's quarter is unchanged (a one-off check printed in the commit message, `0 lots moved`).
- [ ] **Step 6: Run the tests.** Expected: PASS. Then `uv run pytest` for the whole pipeline. Expected: PASS.
- [ ] **Step 7: Commit.**

### Task 1: The April 2000 table

**Files:** Create `inputs/table.py` (a port of `k8/truth_prep.py`). Move the download to
`rnd/fixed/april2000.geojson`, with a provenance record: data set cfhj-8c6b and the file time 2026-11-09 23:24.

**Interfaces:** `build_table(root: Path) -> list[Path]` writes `root/zon2000/april2000.parquet`.

- [ ] **Step 1: Port `truth_prep.py` as `build_table`,** with no change in method.
- [ ] **Step 2: Build and check.**
  - Run: `python -m joint_solver inputs build table && python -m joint_solver inputs check table`
  - Expected: `april2000.parquet  identical` or `same data`.
- [ ] **Step 3: Commit.**

### Task 2: Saved reads, replayed with provenance

**Files:** Create `inputs/reads.py`, built on `prob-warp/check/run_as_ran.py`'s cache-only runner (PR #58). Copy
`textpass.py` and `read3.py` from `bubble-rnd/snapshot-w3`, unchanged.

**Interfaces:**
- `replay(kind: str, sheet: str, root: Path) -> Path`, where kind is one of `terra`, `terra-turned`, `read3`.
- `provenance_of_reads(path: Path) -> list[dict]`. Each record holds:
  - the cache key;
  - the model;
  - the prompt's sha256 and the prompt text file;
  - the image's sha256;
  - the cache file's time, as the call time.

- [ ] **Step 1: Write the failing test `test_replay_refuses_miss`.** An empty cache folder makes `replay` raise
  `LLMCacheMiss`, and no client call happens.
- [ ] **Step 2: Implement `replay` for `terra`.** This is already proven identical for p01–p13.
- [ ] **Step 3: Add `terra-turned`.**
  - Find which cache holds `textpass.py`'s answers: the pipeline's `data/cache/llm` or a rescued scratch cache.
  - If it is a scratch cache, copy only the needed entries into `pipeline/data/cache/llm`. Adding answers is how the
    pipeline stores them; nothing is deleted.
  - Check against `rnd/k8/pNN/textboxes2_*.json`: identical.
- [ ] **Step 4: Add `read3`.**
  - Its answers are in the rescued `scratch-2026-11-13/data_*/cache/llm`; copy the entries it needs as in Step 3.
  - Check against `rnd/zon2000/read3.npz` and `read3r.npz` (`read3r.py`, the re-read, is ported alongside):
    identical.
- [ ] **Step 5: Provenance for the sol reads.** These are the pipeline's own `district_labels.json` and `.answers`;
  write the records only.
- [ ] **Step 6: Build all reads.**
  - Run: `python -m joint_solver inputs build reads && python -m joint_solver inputs check reads`
  - Expected: every file identical, and `live calls 0`.
- [ ] **Step 7: Commit.**

### Task 3: Your hand-made inputs (needs D3)

- [ ] **Step 1: Commit your `label_fixes.csv` rows** in their own PR. Run `sfzmaps build`, commit `dvc.lock`, then
  `sfzmaps ship`, as AGENTS.md says.
- [ ] **Step 2: Record the sha256** of the `label_fixes.csv` that J7 used. That is the working copy: last changed
  2026-11-09 22:23, before `joint/placement.npz` was fitted (2026-11-13 10:52).
  - Write the test `test_label_fixes_recorded`: placement's provenance carries the sha256, and `check` reports a
    mismatch.
- [ ] **Step 3: Commit.**

### Task 4: Sheet transforms (finish PR #58)

**Files:**
- Modify the `prob-warp` branch: the `check/` scripts and the README.
- Create here: `inputs/transforms.py`, which copies `data_ay2` with each `correction.file` pointing at the rebuilt
  file under `pipeline/data/prob-warp/`.

- [ ] **Step 1: The 2000 p09/p10 plus fits** (`apply_plus.sh`). Check against the rescued
  `plus/confit_*_plus.npz`: identical.
- [ ] **Step 2: The choice** (`mirror_prob.py`), for the 2000 sheets.
  - Check `choice.json` (2000 entries) against the rescued copy: identical.
- [ ] **Step 3: The warps for the 13 2000 sheets** (`warp_prob.sh`).
  - Check each `warped.tif` against `rnd/data_ay2`: identical.
  - The p06/p07 fits rebuilt without LOWMEM have not been checked; this step checks them.
  - If p06/p07 differ: recover `probfit.py` as of 2026-11-09 07:08 UTC from the session transcript, rebuild, and
    check again.
- [ ] **Step 4: `build_transforms`.**
  - It writes `inputs/data_ay2` with in-repo correction paths.
  - Check: the 13 corrected transforms give the same `to_pixel` and `to_world` (exactly) as today's on 1,000 points
    per sheet.
  - `check_no_tmp_paths` passes.
- [ ] **Step 5: Commit both branches.**

### Task 5: Frame crop and cuts

**Files:** Create `inputs/frame.py`, porting `k8/zframe2.py` and `k8/zcuts.py`. The cut regions stay as constants,
named as hand-picked. Port `zframe2.py`'s 3 px refit cutoff as-is and list it here.

**Interfaces:** `build_frame(root) -> [zon2000/zframe2.json, zon2000/zframe2_cuts.json]`

- [ ] **Step 1: Port both.**
- [ ] **Step 2: Build and check.** Expected: both identical.
- [ ] **Step 3: Commit.**

### Task 6: Merged scan

**Files:** Create `inputs/merged.py`, porting `k8/zmerge.py` without its cleaned-page output.

**Interfaces:** `build_merged(root) -> [k8/MALLP_c1/raw.png, k8/MALLP_c1/nsheets.png]`. It consumes Task 4's
`inputs/data_ay2` and Task 5's frame files.

- [ ] **Step 1: Port.**
- [ ] **Step 2: Build and check.** Expected: `raw.png` and `nsheets.png` identical to `rnd/k8/MALLP_c1/`.
- [ ] **Step 3: Commit.**

### Task 7: Lot grid

**Files:** Create `inputs/grid.py`, porting the grid part of `k8/truth.py`. The grid frame comes from constants:
4 ft/px, top-left (5979300, 2124100), 11900 × 9825 (`zmerge.py`'s grid), not from `lotwalls.build("city5")`.

**Interfaces:** `build_grid(root) -> [zon2000/truth_grid.npz]` with arrays `lot`, `grown`, `pairs` and `inside`.

- [ ] **Step 1: Port.**
- [ ] **Step 2: Build and check.**
  - Expected: all four arrays identical.
  - If they differ, city5's frame was not these constants. Report how, and stop for your call.
- [ ] **Step 3: Commit.**

### Task 8: Mapped area and scored lots (needs D4)

**Files:** Create `inputs/mapped.py`. Test: `tests/joint_solver/test_mapped.py`.

**Interfaces:**
- `mapped_area(nsheets_png: Path) -> shapely.MultiPolygon`: the grid cells where any sheet shows the map.
- `scored_lots(lots, mapped, known) -> np.ndarray[bool]`: a lot with a known table code and more than half its area
  inside `mapped`.
- Writes `zon2000/scored.npz` (`scored`) and `k8/city/footprint.geojson`.

- [ ] **Step 1: Write the failing test** on a 3-lot synthetic grid: a lot 60% inside is scored, 40% inside is not,
  and a lot with an unknown code is not.
- [ ] **Step 2: Implement it.** Run the test: PASS.
- [ ] **Step 3: Compare with today's sets.**
  - Build, then report: today's scored count; the new count; lots added and lost, by quarter; and a picture of the
    differences for you.
- [ ] **Step 4: Re-score J7 on the new scored set.**
  - Run `score` with `SFZ_INPUTS_ROOT` pointing at a tree that differs only in `scored`.
  - Report the six measures for J7 on old and new scored sets side by side.
- [ ] **Step 5: Commit, and wait for your yes** before later tasks use the new scored set.

### Task 9: Label evidence, reproduced as it is

**Files:**
- Create `inputs/labels/`, porting `labread2.py`, `addprints.py`, `lotclf6.py`, `labside.py` and `relink3.py`.
- Recover the shell driver that made `cwfwx{q}` from the session transcript (`check/recover_from_transcript.py`,
  PR #58).

**Interfaces:**
- `build_labels(root, quarter) -> [zon2000/lotclf6_cwfwx{q}.npz, labread2_cwfwx{q}.npz, labside_cwfwx{q}.npz]`
- Old-run inputs (`lotdist_run.npz`, `lotcode_run.npz` and `relink3`'s run files) are read from `rnd/` and named in
  provenance as "old run".

- [ ] **Step 1: Recover the driver.**
  - List each step's command and settings, with the transcript line for each.
- [ ] **Step 2: Port each script and build for nw, ne, sw and se.**
- [ ] **Step 3: Check.** Expected: all 12 files identical.
- [ ] **Step 4: Commit.**
  - Task 10 (removing the old runs, D5) then gets its own plan.
  - That plan lists one replacement per old-run input, each re-scored alone.

### Task 11: Leader strokes from the original merged scan

**Files:** Create `inputs/strokes.py`, porting `k8/zstrokes.py`. It reads `raw.png`, not the cleaned page.

- [ ] **Step 1: Port.**
  - Check: run on the old cleaned page, it reproduces `zstrokes_old.npz` exactly (this proves the port).
- [ ] **Step 2: Build on `raw.png`.** Refit placement (`python -m joint_solver placement`).
  - Report placement's held-out hit rate on the 101 hand moves, old against new.
- [ ] **Step 3: Re-score with the new placement.** Report the six measures. Commit, and wait for your yes.

### Task 12: Lot-size fits

**Files:** Create `inputs/lotsize.py`, porting `k8/zlotchar2.py`. It is fitted per quarter, with the quarter in
provenance.

- [ ] **Step 1: Port.**
  - Check: built on today's `lotcode_run.npz`, it reproduces `zlotchar2.npz` exactly.
- [ ] **Step 2: Build on Task 8's scored set.** Report how the fits change. Commit.

### Task 13: Finish the text detector (R&D)

- [ ] Move `joint_solver/textnet/` paths to the package settings.
- [ ] Train the SW and SE held-out models, about 100 minutes each, side by side.
- [ ] Report all four quarters' held-out numbers in one table.
- [ ] Make a page of the unboxed text pieces for you to look at.
- [ ] Then write a short design note: where its text odds enter the walls and the labels. Wait for your yes.

### Task 14: Walls inside the solver (R&D; replaces `zodds_Q1x`)

- [ ] Design note first. Walls are a probability along each lot border, learned from the table and held out by quarter:
  - read from `raw.png` ink;
  - text ink discounted by Task 13;
  - no chain, no cleaned page, no cutoffs.
- [ ] The comparisons, all with the same score:
  - today's `zodds_Q1x` (cleaned page);
  - the existing raw-page variant `zodds_Enrx` (`k8/run_rawclean.sh`: near-equal except SE, where it misses
    1,301 against 753);
  - the new walls.
- [ ] Your yes on the design, then its own plan.

### Task 15: Re-score on the rebuilt inputs

- [ ] Solve all four quarters on the rebuilt inputs.
- [ ] Report the six measures against J7. Every definition change from Tasks 8, 10, 11 and 14 is listed with its own
  effect.
- [ ] Apply D1 to the table corrections.
- [ ] Rebuild the data-error list from the rebuilt inputs.
- [ ] Update `joint-solver.md`. Commit.
