# 1921 Region-Level Hatching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read the 1921 hatching over each drawn region instead of each lot. Each lot's evidence comes from the
scan patches around it, the joint solver turns it into whole regions, and every step is scored on the owner's answers
with one measure.

**Why (owner, 11-18):** "are you learning this stuff at the lot level instead of the drawn-region level? these old
maps were way more coarse so learning at the lot/parcel level may be wrong". The lot classifier is stuck at 75%
(rounds 1 and 2: 195 and 196 of 261 answers). Its 36 misses of true 1R:
- a small lot sits in the white gap between 1R's wide hatch lines;
- text covers the lot;
- the lot outline sits a few pixels off;
- the lot is clearly hatched but the classifier still under-weights it.

**Architecture:**
- **Evidence (Task 3):** square patches of each 1921 scan are measured with the spike's texture measures and
  classified by a model fitted on the old pattern reader's reads (training only) and the legend swatches. A lot's cost
  for each district is the area-weighted sum of its patches' log-likelihood ratios. Because costs add, a solved region
  sums the evidence of every patch inside it. A lot between hatch lines or under text takes the ink of the patches
  around it.
- **The solve (Task 4):** the existing joint solver (`solver/solve.py`: lot costs, wall odds, region/compound/expansion
  moves) decides where the regions end. Lots take their region's district.
- **Wall terms (Tasks 5 and 6):** hatching that changes across a border, and the ordinance's Section 15 boundary
  depths. Each is added only after the step before it is scored.

**Tech Stack:**
- Python, numpy/scipy, geopandas/shapely, opencv/scikit-image (all already dependencies);
- the solver package `sfzmaps.solver`;
- pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-general-solver-design.md`:
- Goal 3, hatching evidence for class C;
- Goal 4, the owner's answers;
- Goal 5, one measure;
- the Anti-goals;
- the section "Wild land and water on the early maps".

**Spike to port from:** branch `worktree-agent-ab10cd06a6d7b4538`, `pipeline/scripts/spikes/hatch1921/`:
- `measure.py`: `spectrum`, `morph_stack`, `morph_feats`, `spec_feats`, `dir_1r`, `periodicity`, `blocks`,
  `measure_swatches`, `footprint_keys`;
- `model.py` / `model_r2.py`: the multinomial logistic fit and `owner_answers`;
- `contact.py`: the contact sheet.

## Global Constraints

- **Cost:** $0. No paid or LLM call. Set `SFZMAPS_MAX_USD=0` on every run.
- **The pattern reader's reads are training labels only, never inference input.** These are the reads in
  `pipeline/data/work/1921/*/district_pattern.json`. The 1921 solves here take no readings, only the patch evidence.
- **The owner's answers are never trained on.** They are the score.
- **No hand-set cutoffs.** No picked thresholds, windows, caps or grids. Every number is fitted, or derived from a
  measurement of the scan, and the code comment says which.
- **No April 2000 table codes** in a 1921 solve. No labels or priors from our own solves.
- **Deterministic:** the same inputs give the same bytes. Sort parallel results before writing, and keep times out of
  every output.
- **Load:** one thread per worker (`OMP_NUM_THREADS=1`) and at most two solves at once. Check `uptime` and
  `memory_pressure -Q` before each run, and keep the load around 20–80.
- **Where outputs go:**
  - every output goes under `pipeline/data/solver/experiments/hatch-regions/`, which is git-ignored and never
    committed;
  - no dvc stages and no change to the `final` run in this plan;
  - stages and publishing come in a follow-up plan, only if the measure says the new map wins.
- **Commits:** none while an `sfzmaps build` runs (`pgrep -f "sfzmaps build"` first).
- **Branch and merging:** work on branch `hatch-regions`, made from `origin/main`. Task 7 works on branch
  `label-checks`. The owner merges.
- **Checks:** `cd pipeline && uv run pytest`.
- **Pictures:** when a step ends with a picture, the report says what the picture is for.

## Review Focus

1. **A lot no patch covers** (off its sheet, in a gap between sheets, no tile) gets a zero cost row (no evidence).
   It never defaults to 2R. Test in Task 3.
2. **A patch at a sheet seam** is measured once, on the sheet whose neatline holds the patch centre. Test in Task 2.
3. **An answered blklot that is a stacked condo, or is missing from the solver's lot table,** is matched through the
   footprint grouping. The scorer reports the count it could not match and never drops answers silently. Test in
   Task 1.
4. **Per-sheet measuring run in parallel** writes byte-identical files when run again. Test in Task 2.
5. **An answered lot that an old district line cuts** (PR #56) is reported apart in the score: it is printed in its
   own row, not left out. Test in Task 1.

---

### Task 1: The measure first, and the baselines

**Files:**
- Create: `pipeline/src/sfzmaps/solver/answers1921.py`
- Test: `pipeline/tests/test_answers1921.py`

**Interfaces:**
- Produces:
  - `owner_answers(answer_dirs: list[Path], case_files: list[Path]) -> pd.DataFrame`, with columns `blklot`, `block`,
    `truth`, `page`, `case`;
  - `predictions_from_run(run_npz: Path) -> pd.Series` (blklot → code; lot index → blklot through
    `data/solver/1921/lots/table.parquet`);
  - `score(pred: pd.Series, answers: pd.DataFrame) -> dict`, with keys `n`, `right`, `share`, `ci95`, `unmatched`,
    `split_lots`;
  - `compare(preds: dict[str, pd.Series], answers: pd.DataFrame) -> dict`, giving per method `score()`, plus paired
    differences against `"published"` with block-bootstrap 95% intervals;
  - CLI `uv run python -m sfzmaps.solver.answers1921 --run NAME [...]`, which prints the table and writes
    `experiments/hatch-regions/score_<name>.json`.

**Truth rules** (the owner's rulings, 11-18):
- `ok`: the case's `code` field, the code the page showed;
- `fix`: the answer's `code`;
- `unzoned`: `2R` (the ordinance: white is 2R);
- `canttell` and `unsure`: dropped, and counted.

**Bootstrap:** resample blocks, 2000 resamples, `numpy.random.default_rng(0)`. That is a fixed seed for byte-identical
reports, not a tuning choice.

- [ ] **Step 1: Branch.** Work on branch `hatch-regions`. It starts from `origin/main` with this plan and the spec's
  11-18 additions committed. In a fresh worktree, set up data with `pipeline/scripts/worktree-data.sh`.
- [ ] **Step 2: Write the failing tests.**
  - `test_truth_rules`: four answers on three blocks (ok shown `1R`; fix → `LI`; unzoned; canttell) give truths `1R`,
    `LI`, `2R`, and one dropped.
  - `test_score_counts_and_interval`: a prediction right on 2 of 3 gives `right == 2`, and `ci95[0] <= 2/3 <= ci95[1]`.
  - `test_unmatched_reported`: an answer whose blklot is absent from the prediction counts in `unmatched`, not in `n`.
  - `test_split_lot_listed`: a blklot in the PR #56 split list appears in `split_lots`.
- [ ] **Step 3:** Run `uv run pytest tests/test_answers1921.py -v`. Expect a FAIL (module missing).
- [ ] **Step 4:** Implement it. Stacked condos: match an answer's blklot to any blklot of its footprint (the spike's
  `footprint_keys`).
- [ ] **Step 5:** Run the tests. Expect a PASS. Also run the full `uv run pytest`.
- [ ] **Step 6: Baselines.** Export the case files to `experiments/hatch-regions/cases/` with
  `git show label-checks:pipeline/inputs/code-check-<page>/cases.json` for pages `1921-likely`, `1921-lowconf` and
  `1921`. Score:
  - `published`: `data/out/1921/districts/parcels.fgb`;
  - `solve-final`: `data/solver/1921/solve/runs/final.npz`;
  - `round1` and `round2`: the spike's `owner_answers_cv*.csv` next to `next_cases_r2.json`.

  Expect round 2 near 196 of 261. Write `score_baselines.json`.
- [ ] **Step 7:** Commit `solver: 1921 answers measure (owner answers, block bootstrap) and baselines`.

### Task 2: Patch measures at the scan's resolution

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatching.py` (measures ported from the spike, plus the patch grid)
- Test: `pipeline/tests/test_hatching.py`

**Interfaces:**
- Consumes:
  - each sheet's `data/work/1921/<sheet>/page.png` (300 dpi, 1-bit) and its sheet transform;
  - `data/solver/1921/lots/lots.npz` (`lot` raster, 4 ft/px, `grid.FT_PER_PX`);
  - `data/solver/1921/page/frames.json` (neatlines).
- Produces:
  - `sheet_period(ink: np.ndarray, mask_1r: np.ndarray) -> float`: the dominant hatch period, in px, of the sheet's ink
    inside the reader's 1R lots;
  - `patch_grid(shape, period) -> list[tuple[int, int, int, int]]`: squares of side `2*period` at stride `period`;
  - `measure_patch(ink, inblock, box, frame) -> dict`;
  - `patch_lots(box, sheet_transform, lot_raster) -> list[tuple[int, float]]`, giving (solver lot, share of the patch's
    in-block pixels);
  - CLI `uv run python -m sfzmaps.solver.hatching --series 1921 --workers N`, which writes
    `experiments/hatch-regions/patches/<sheet>.parquet` (`sheet`, `patch`, `cx`, `cy` world ft, features) and
    `patch_lots/<sheet>.parquet` (`patch`, `lot`, `share`).

**Derived numbers (no picks):**
- **Patch side:** `2 * period` (two periods: the least that shows a period), with `period` from `sheet_period`.
- **Fallback period:** a sheet with no 1R reads takes the median period of the other sheets.
- **Erosion inside the block outline:** the sheet's measured outline stroke width, which replaces the spike's
  `ERODE_PX`.
- **Spectral bins:** the spike's `BANDS`, binning only. The hatch band runs from the legend's finest period ÷ 2 to
  `2 * period`.
- **Orientation:** in both frames, page and block major axis (the spike's `blocks`). Each angle term is weighted by
  the patch's periodicity strength `|m2|` (`|m4|` for cross-hatch).

- [ ] **Step 1: Write the failing tests.**
  - `test_period_and_angle`: a synthetic 1-bit 200×200 image of −45° lines, 3 px wide, at period 20 gives a period of
    20 ± 1 and a line angle of −45 ± 3°.
  - `test_blank_not_periodic`: a blank patch has a periodicity under 1/5 of the hatched patch's.
  - `test_lot_between_lines_borrows`: a lot 12 px wide lying between two lines 20 px apart gets `share > 0` from a
    patch whose features show the hatch.
  - `test_seam_once`: a patch whose centre lies inside sheet A's neatline is written for A only.
  - `test_rerun_identical`: running the CLI twice on a two-sheet fixture gives identical bytes.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement. Port the spike's functions with their docstrings.
- [ ] **Step 4:** Run them. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run on 1921.** `OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatching --series
  1921 --workers 6`. Check `uptime` while it runs. Report the patch count and each sheet's period.
- [ ] **Step 6: Look.** On two sheets (one dense, one with wide 1R), draw the patches coloured by dominant line angle
  over the scan: `experiments/hatch-regions/look_patches.png`. It is for checking the periods and angles by eye.
- [ ] **Step 7:** Commit `solver: 1921 hatch patch measures at scan resolution`.

### Task 3: Patch model and per-lot evidence

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchfit.py`
- Test: `pipeline/tests/test_hatchfit.py`

**Interfaces:**
- Consumes: Task 2's parquet files; the reads (the spike's `reads()`); the legend swatches (the spike's
  `measure_swatches`, measured with `measure_patch`); `lots.npz` `quarter`; `transfer.series_vocabulary()`.
- Produces:
  - `fit_quarter(q: str) -> Model`;
  - `lot_evidence(model, patches, patch_lots, n_lots, names) -> np.ndarray` (n_lots × K costs);
  - CLI `uv run python -m sfzmaps.solver.hatchfit --series 1921`, which writes
    `experiments/hatch-regions/evidence/evidence_{q}.npz` (`U` float32, `names`, `tau`) for q in nw, ne, sw, se.

**Training and evidence:**
- **Training rows:** each patch is labelled by every read lot it overlaps, with weight = the read's confidence × the
  lot's share of the patch. Patches that straddle a border therefore weigh less (my call on the open spec question:
  straddling patches are kept, weighted by share). Each legend swatch is a row of weight 1 for its district.
- **Model:** multinomial logistic. Its features are the patch measures plus `|m2|·cos 2θ`, `|m2|·sin 2θ` (and the
  same for `m4`) in both the page frame and the block frame, so the fit learns how far each district follows the
  block axis (spec Goal 3: a learned mixture). The L2 strength is fitted by held-out log loss, leaving each of the
  three training quarters out in turn.
- **Per held-out quarter q:**
  - fit only on patches whose lots lie outside q;
  - `U_q[lot, k] = −τ_q Σ_p share_lp (log p_k(p) − log π_k)`, where π is the weighted class shares in training;
  - subtract each row's minimum;
  - `τ_q` is the slope of a weighted logistic calibration of the pooled lot log-ratios against the reads, on the
    training quarters.
- **Lots with no patch:** a zero row.

- [ ] **Step 1: Write the failing tests.**
  - `test_hatched_patch_beats_blank`: a lot covered by one hatch patch and one blank patch costs less for 1R than a lot
    covered by two blank patches.
  - `test_no_patch_zero_row`: a lot with no patches has a zero row.
  - `test_quarter_held_out`: no patch of quarter q is among the rows `fit_quarter(q)` trained on.
  - `test_names_order`: the columns of `U` follow `names` exactly.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement, porting the spike's logistic fit.
- [ ] **Step 4:** Run them. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run it, then score the evidence alone.** Score the argmin of `U` per lot, with no solve, as `patches`
  with Task 1's measure. Write `score_patches.json`.
- [ ] **Step 6:** Commit `solver: 1921 patch hatch model and per-lot evidence`.

### Task 4: Solve with the patch evidence

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/solve.py` (`costs`, `solve_joint`, `solve_quarter`, `solve_all`,
  `solve_and_report`)
- Modify: `pipeline/src/sfzmaps/cli.py` (`solver solve` and `solver solve-quarter`: a new `--evidence PATH` next to
  `--walls`)
- Test: `pipeline/tests/test_solve_evidence.py`

**Interfaces:**
- Consumes: `evidence_{q}.npz` from Task 3.
- Produces:
  - `costs(..., evidence: np.ndarray | None = None)` adds `evidence` (in `rd.names` order) to U before the row minimum;
  - `solve_quarter(..., evidence_dir: Path | None = None)` loads `evidence_{q}.npz`. If its `names` differ from
    `rd.names`, or its shape from (n_lots, K), it raises `ValueError` naming the file. With evidence, the readings are
    empty (a zero-row `labels.Readings` with the series vocabulary's names);
  - `check_run` refuses `evidence_dir` for `runs.FINAL_RUN`.

- [ ] **Step 1: Write the failing tests** on a 5-lot chain.
  - `test_weak_white_lot_joins_region`: wall odds −5 (no line) everywhere. Lots 0, 1, 3 and 4 cost 0 for 1R and 3 for
    2R; lot 2 costs 0.5 for 1R and 0 for 2R. Expect all five lots to be 1R.
  - `test_wall_splits_regions`: odds +5 between lots 2 and 3, and lots 3 and 4 strongly favour 2R. Expect
    `[1R, 1R, 1R, 2R, 2R]`.
  - `test_evidence_names_mismatch_raises`.
  - `test_final_run_refuses_evidence`.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expect a PASS. Then run the full suite: the existing solver tests must still pass.
- [ ] **Step 5: Run.** `OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run sfzmaps --series 1921 solver solve hatch-r1
  --evidence data/solver/experiments/hatch-regions/evidence`. Score `hatch-r1` against the baselines.
- [ ] **Step 6: Look.**
  - Rerun the spike's contact sheet on the owner's 36 true-1R lots that round 2 missed, now under `hatch-r1`:
    `experiments/hatch-regions/miss_r1.png`.
  - Render one dense area with the published map and `hatch-r1` side by side.

  The pictures are for judging whether the misses are fixed and whether regions look like the drawing.
- [ ] **Step 7:** Commit `solver: solve with per-lot evidence (1921 hatch patches)`.

### Task 5: Walls where the hatching changes

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchwalls.py`
- Test: `pipeline/tests/test_hatchwalls.py`

**Interfaces:**
- Consumes: the evidence and patch posteriors (Task 3), `lots.npz` `pairs`, and the transferred
  `data/solver/1921/walls/walls_{q}.npz`.
- Produces:
  - `change_logodds(pa: np.ndarray, pb: np.ndarray) -> np.ndarray`, which is `log P(differ) − log P(same)` with
    `P(same) = Σ_k pa_k pb_k`; `pa` and `pb` are each lot's pooled patch posterior (softmax of `−U/τ`);
  - `fit_wall_terms(q, features) -> coefficients`;
  - CLI `uv run python -m sfzmaps.solver.hatchwalls --series 1921 [--depth]`, which writes
    `experiments/hatch-regions/walls-hc/walls_{q}.npz` (`lo_ab` = transferred odds + fitted terms).

**The fit:** per held-out quarter, a weighted logistic regression with the transferred `lo_ab` as offset, on pairs
whose two lots both have reads outside q. The label is "the two reads differ"; the weight is the product of their
confidences.

- [ ] **Step 1: Write the failing tests.**
  - `test_same_vs_different`: both lots strongly 1R gives a log-odds below 0; 1R against 2R gives one above 0.
  - `test_fit_recovers_sign`: on synthetic pairs whose labels follow the feature, the fitted coefficient is above 0.
  - `test_quarter_pairs_held_out`.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run and score.** `solver solve hatch-r2 --walls .../walls-hc --evidence .../evidence`, then score
  `hatch-r2`.
  - If it is not worse than `hatch-r1` beyond noise (paired interval), keep it.
  - If it is worse, look at what it broke before dropping it.
- [ ] **Step 6:** Commit `solver: 1921 walls from hatching change`.

### Task 6: The ordinance's boundary depths (Section 15) as a likelihood

**Ordinance No. 5464, Section 15:** a district boundary inside a block lies on the street line, or 100 ft back (blocks
200 ft wide or less), or 137½ ft back or on the rear lot line (wider blocks).

**Files:**
- Create: `pipeline/src/sfzmaps/solver/depth15.py`
- Modify: `pipeline/src/sfzmaps/solver/hatchwalls.py` (`--depth` adds this feature to the same fit)
- Test: `pipeline/tests/test_depth15.py`

**Interfaces:**
- Produces:
  - `pair_depths(pairs, lots_table) -> pd.DataFrame`, giving per pair the shared edge's depth `d` (ft) from its block's
    nearest street line, and the block width `W` (the short side of the block's minimum rotated rectangle; blocks as
    in the spike's `blocks`);
  - `ordinance_distance(d, W, rear_depth) -> np.ndarray`, which is `|d − 100|` where W ≤ 200, and
    `min(|d − 137.5|, |d − rear_depth|)` where W > 200;
  - `depth_logratio(x_line, w_line, x_all, w_all) -> callable`, the log of two weighted Gaussian kernel densities
    (Silverman bandwidth), "district line" over "any pair", fitted on the training quarters' read pairs.

  The ordinance gives the feature. The data give how much it counts.

- [ ] **Step 1: Write the failing tests.**
  - `test_ordinance_distance`: a 200 ft block, d = 100 gives 0 and d = 60 gives 40. A 260 ft block, d = 137.5 gives 0.
    With a rear lot line at 120, d = 120 gives 0.
  - `test_logratio_peaks_at_lines`: lines labelled at x ≈ 0 give a higher log-ratio at 0 than at 40.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run, solve and score.** Run `hatchwalls --depth` → `walls-hc-depth`, solve `hatch-r3`, and score it.
  Report where the fitted density peaks, as a check against the ordinance.
- [ ] **Step 6:** Commit `solver: 1921 Section 15 boundary depths as a wall likelihood`.

### Task 7: Region check page (branch `label-checks`)

**Files:**
- Modify: `pipeline/scripts/label_checks/build_label_checks.py` (new page kind `region`)
- Modify: `pipeline/scripts/label_checks/page.html`
- Create: `pipeline/inputs/code-check-1921-regions/` (`cases.json`, README)
- Test: `pipeline/tests/test_label_checks_regions.py`

**The page:**
- **Cases:** a case is one region of the best run: the lots joined over pairs where the run gave the same code.
- **Order:** ranked by 1 − the region's area-weighted mean posterior (`solve.posterior`), one case per block,
  spread in rounds (the builder's existing spreading). 300 cases.
- **What's shown:** the crop shows the region's outline (the union of its lots), with the pill beside the outline
  giving the region's code.
- **Answers:** OK (O), Fix with a code (F), Border wrong (B), Can't tell (C). They are saved to `answers-1921-regions`.
- **What the answers are for:** training patches answered whole (spec Goal 4), for the next round. They are never
  this plan's score. The score's random lots are the existing `1921` page's 240 random cases.

- [ ] **Step 1: Write the failing tests.**
  - `test_region_outline_is_union`: the outline equals the union of the member lots' outlines.
  - `test_one_case_per_block_per_round`.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expect a PASS.
- [ ] **Step 5:** Build the page from the best run of Tasks 4–6. Commit
  `label-checks: 1921 region page (one answer per drawn region)`. The controller publishes it.

### After the tasks: report to the owner

- **The score table:**
  - methods: published, solve-final, round 1, round 2, patches, hatch-r1, hatch-r2, hatch-r3;
  - for each: right / n, share, 95% interval, and the paired difference against published.
- **The two pictures,** each saying what it is for.
- **The ask:** the 240 random lots on the 1921 page are what decides publishing (spec Goal 5: the owner's held-out
  random answers). Publishing is a follow-up plan: stages, `sfzmaps build`, `ship`.
