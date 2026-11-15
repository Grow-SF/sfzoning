# Solver Walls and Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The joint solver runs as `sfzmaps build` stages on the rebuilt inputs:
- walls learned from the table and read from the raw merged page, with text discounted by our detector;
- the four held-out solves, joined and scored with the six measures against J7.

**Architecture:**
- J7's solver (`pipeline/scripts/joint_solver/`) moves into `sfzmaps.solver`. Its method is unchanged; it reads the
  stage outputs of the first plan.
- The wall evidence is a port of R&D's raw-page model ("S1": stroke evidence, closings, enclosure, profile, ridge
  logistic per held-out quarter, no whole-wall stage). The detector's text chance is then added as measured features.
- Two new stages: `solver-walls` and `solver-solve`.

**Tech Stack:** numpy, scipy, scikit-image/OpenCV, PyMaxflow, PyTorch (prediction only, already done by the text
stages), Typer, DVC, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-solver-build-stages-design.md`: sections "Walls (plan E7)", "Solve
(plan F)", "Fresh-clone check (plan G)". The first plan is `docs/superpowers/plans/2026-11-15-solver-inputs.md`, and
its outputs are this plan's inputs.

## Global Constraints

Everything in the first plan's Global Constraints applies here too:
`.superpowers/sdd/2026-11-15-solver-inputs/global-constraints.md`. That covers:
- no reads outside a fresh clone;
- no model calls;
- no per-item stamps;
- determinism;
- the lot index;
- the grid;
- the quarter split;
- style;
- commits.

In addition:

- **The inputs this plan reads** (first plan's outputs, all under `pipeline/data/solver/`):
  - `page/ink.png`, `sheets.png`;
  - `lots/lots.npz` (lot, grown, pairs, scored, known, quarter, table) and `table.parquet`;
  - `prints/prints.parquet` and `terra_boxes.json`;
  - `text-{q}/chance/<sheet>.png`;
  - `labels/readings_{q}.npz`, `streetside_{q}.npz`, `lotsize.npz` and `strokes.npz`.

  Read each module that writes them before using them.
- **J7's numbers are the yardstick**, quoted from `pipeline/scripts/joint_solver/joint-solver.md`:

  | Measure | J7 |
  |---|---|
  | lines drawn | 96.3% |
  | drawn that are table lines | 96.9% |
  | labels right | 95.0% |
  | holes | 15 |
  | incidents | 824 |
  | wrong lots | 8,625 |
  | wall odds held-out log-loss (nw / ne / sw / se) | .0422 / .0676 / .0313 / .0412 |
  | R&D raw page, no cleaned page ("S1") | .0484 / .0795 / .0370 / .0597 |
- **Comparing with J7's files.**
  - J7's arrays are in J7's lot order. Translate them with
    `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/solver/j7_lot_of_new.npy`
    (new lot index → J7 lot index).
  - J7's wall odds: `/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/rnd/zon2000/zodds_Q1x{q}.npz['lo_ab']`,
    on J7's pairs (`zon2000/truth_grid.npz['pairs']`).
  - Comparison scripts stay in the scratch folder.
- **The wall gate (spec):**
  - held-out log-loss per quarter on the pairs common to the new lots and J7's, for both the new walls and J7's odds;
  - a feature is kept only if it lowers held-out log-loss in at least three quarters;
  - after the feature list, the solve runs whatever the gate says, and both numbers are reported.

## Review Focus

1. **A wall fit that sees its own quarter.** The training pairs for q have both lots outside q. Test: changing a pair
   inside q changes nothing in the q fit.
2. **Pair alignment.** Wall odds are indexed like `lots.npz['pairs']` rows. Test: `walls_{q}.npz` has one value per
   pair row, and a synthetic pair with ink on its border gets higher odds than one without.
3. **The solver's behaviour drifting in the move.**
   - Test: on a synthetic city, the moved `solve_joint` with true-border walls returns the table's codes.
   - Check: with J7's odds mapped onto the new pairs and the new labels, the measures land near J7's.
4. **Text discount leaking the held-out quarter.** A pixel's text chance comes from the model that held out that
   pixel's quarter. Test: the out-of-fold assembly picks model q for quarter-q pixels.
5. **Nondeterminism in the solve.** It is discrete and seeded. Two runs give identical codes for every lot.

---

### Task 10: The solver moves into `sfzmaps.solver`

**Files:**
- Create, from `pipeline/scripts/joint_solver/`:
  - `pipeline/src/sfzmaps/solver/energy.py`, `search.py`, `measures.py`, `borders.py`, `holes.py`, `placement.py`,
    `published.py`, all unchanged in method;
  - `labels.py`, reading the new outputs;
  - `solve.py`, with four held-out solves side by side as subprocesses of `sfzmaps solver solve-quarter`;
  - `city.py`, replacing `data.py`.
- Move `pipeline/scripts/joint_solver/notes/` and `joint-solver.md` to `docs/research/joint-solver/`.
- Delete `pipeline/scripts/joint_solver/`, including its R&D `textnet/`, which is now ported.
- Modify `cli.py`: `sfzmaps solver solve RUN [--walls DIR] [--true-borders] [--keep-costs]`, `solve-quarter`,
  `placement [--walls DIR]`, `score RUN...`, `holes RUN`, `published`.
- Test: `pipeline/tests/test_solver_solve.py`.

**Interfaces:**
- **`city.City`**, built from `lots.npz`, `table.parquet` and `table.corrected_codes`:
  - `n`, `a`, `b`, `border_ft`;
  - `known`; `table` (corrected codes, for scoring); `table_raw` (uncorrected);
  - `scored`, `quarter`, `centroid`;
  - `lot_raster`, `grown_raster`;
  - `scan()` → page ink on the SAME grid. J7's origin offsets go: the page and the lots share one grid now.
- **`city.wall_odds(q, walls_dir=WALLS_DIR) -> np.ndarray`**: `walls_dir/walls_{q}.npz['lo_ab']`, NaN → 10 (as J7).
- **`labels.readings(q, vocabulary)`** is built from:
  - `readings_{q}.npz` (print, code, trust);
  - `prints.parquet` `lab_lot`;
  - `streetside_{q}.npz` (far-side copies at `trust × min(rB_st / rA_st, 1)`, as J7).

  It returns J7's `Readings` dataclass.
- **`placement`:**
  - inputs: `prints.parquet` (positions, sheet), `strokes.npz`, `lotsize.npz`, `readings_{q}.npz`, the walls;
  - leader moves: the rows of `paths.LABEL_FIXES` with a fix point, through the corrected warp transforms (not
    `data_ay2`).
  - It writes `SOLVE_DIR/placement.npz`.
- **The outputs:**
  - `paths.WALLS_DIR = SOLVER_DIR / "walls"`;
  - `paths.SOLVE_DIR = SOLVER_DIR / "solve"`;
  - runs go to `SOLVE_DIR/runs/`;
  - the data-error list goes to `SOLVE_DIR/data_errors.npz`.

- [ ] **Step 1: Write failing tests** on a synthetic city (about 40 lots, 3 districts, each with labels):
  - `test_true_border_walls_recover_the_table`;
  - `test_energy_charges_both_wall_directions`;
  - `test_readings_add_far_side_copies_at_the_ratio`;
  - `test_join_takes_each_lot_from_its_quarters_solve`.
- [ ] **Step 2:** Port. See the tests pass. No module may read `SFZ_RND`, `data_ay2` or `pipeline/data/rnd`.
- [ ] **Step 3: One-off check (scratch).**
  - Write J7's odds, mapped onto the new pairs, as `walls_{q}.npz` into a scratch walls folder. A pair missing in J7
    gets NaN.
  - Run `sfzmaps solver placement --walls DIR` and `sfzmaps solver solve CHECK --walls DIR`, then `score CHECK`.
  - Report the six measures against J7's. This isolates what the new warp, lots and labels change.
- [ ] **Step 4:** Update the README. Run the full suite. Commit.

### Task 11: Wall evidence from the raw page (R&D's "S1" ported)

**Files:**
- Create: `pipeline/src/sfzmaps/solver/wallevidence.py` (stroke evidence per border, closings, enclosure levels,
  majority-region placement, profile) and `pipeline/src/sfzmaps/solver/walls.py` (hat bases, ridge logistic per
  held-out quarter, `build_walls()`).
- Ports from `bubble-rnd/raw-profile:pipeline/scripts/bubble-rnd/k8/`: `zstrokeev.py`, `zclose.py`, `zenclose.py`,
  `zplace.py`, `zprofile.py`, `zborderfit.py`, `zloop_odds.py` (the NOWALL path only).
- Test: `pipeline/tests/test_solver_walls.py`.

**Interfaces:**
- **`wallevidence.border_features(lots, ink, strokes, letter_share) -> np.ndarray`**: (n pairs, k) in pair-row order.
  - Each named feature group stays as in R&D.
  - `letter_share` is per stroke, from the Terra boxes (as R&D) in this task.
- **`walls.fit_quarter(F, y, train_mask, ridge=0.1) -> np.ndarray`**: weights. `y` = the two lots' uncorrected table
  codes differ; train = both lots known and both outside q.
- **`walls.build_walls() -> dict`** writes `WALLS_DIR/walls_{q}.npz` (`lo_ab` float32 per pair row) and
  `WALLS_DIR/evaluation.json` (held-out log-loss per quarter, and the number of pairs).

- [ ] **Step 1: Write failing tests:**
  - `test_inked_border_scores_higher_than_bare_border` (Review Focus 2);
  - `test_fit_ignores_pairs_in_its_quarter` (Review Focus 1);
  - `test_enclosure_separates_two_walled_regions` (two lots in regions separated by a drawn ring);
  - `test_city_limit_is_a_wall` (the mapped-area edge closes like ink).
- [ ] **Step 2:** Port. See the tests pass.
- [ ] **Step 3: Run it.**
  - Run `build_walls()`.
  - One-off: the held-out log-loss per quarter on the common pairs, the new walls against J7's odds, with S1 as
    context.
  - Report the table and the run time.
- [ ] **Step 4:** Update the README. Run the full suite. Commit.

### Task 12: The detector's text discount in the walls

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textgrid.py`: out-of-fold text chance on the merged grid. Each grid pixel
  takes the chance from the model of its own quarter, sampled through each sheet's corrected transform as the merge
  samples ink. Where several sheets inked a pixel, the largest chance wins.
- Modify: `wallevidence.py` and `walls.py`.
- Test: `pipeline/tests/test_solver_walls.py`.

**Interfaces:**
- **`textgrid.text_chance() -> np.ndarray`** (uint8 HEIGHT × WIDTH), cached to `WALLS_DIR/text_chance.png`.
- **Candidate features, in this order.** Each is kept only if held-out log-loss falls in at least three quarters;
  each result is recorded in `evaluation.json` and the report.
  1. ink weighted by (1 − text chance) in the stroke-evidence and profile features, replacing Terra's letter share;
  2. the same, added beside Terra's letter share instead of replacing it;
  3. across a street: weighted ink over the whole street width, the edges included.

- [ ] **Step 1: Write failing tests:**
  - `test_out_of_fold_text_takes_each_pixels_own_quarter_model` (Review Focus 4);
  - `test_text_ink_counts_less_than_line_ink`.
- [ ] **Step 2:** Implement. See the tests pass.
- [ ] **Step 3:** Run each candidate and report the per-quarter table. Keep what passes. Rerun `build_walls()` with
  the kept set.
- [ ] **Step 4:** Update the README (the features kept and why). Run the full suite. Commit.

### Task 13: The `solver-walls` and `solver-solve` stages and the report

**Files:**
- Modify: `build.py`:
  - `Stage("solver-walls", (), (), ("solver-labels", "solver-text-nw", "solver-text-ne", "solver-text-sw",
    "solver-text-se"), command=("solver", "walls"), reads=("data/solver/page", "data/solver/lots",
    "data/solver/labels/strokes.npz", "data/solver/prints/terra_boxes.json", "data/solver/text-nw/chance",
    "data/solver/text-ne/chance", "data/solver/text-sw/chance", "data/solver/text-se/chance",
    "data/warp/2000-digital-zoning"), writes=("data/solver/walls",), code=("sfzmaps.solver.walls",))`;
  - `Stage("solver-solve", (), (), ("solver-walls",), command=("solver", "solve", "final"),
    reads=("data/solver/lots", "data/solver/prints", "data/solver/labels", "data/solver/walls",
    "data/warp/2000-digital-zoning", "inputs/2000-digital-zoning/label_fixes.csv", "inputs/zoning-2000"),
    writes=("data/solver/solve",), code=("sfzmaps.solver.solve",))`.
- `solve final` runs:
  - placement;
  - the four solves;
  - the join;
  - holes and the data-error list;
  - the score.

  It writes `SOLVE_DIR/report.json` and `report.md`: the six measures with and without the data-error lots, against
  J7's numbers.
- Test: in `pipeline/tests/test_build.py`, the two stages' deps.

- [ ] **Step 1:** Write the failing tests. Implement.
- [ ] **Step 2: Run** `uv run sfzmaps solver walls` and `uv run sfzmaps solver solve final`.
  - Run the solve twice: every lot's code must be identical (Review Focus 5).
  - Report the six measures against J7, and every definition change with its effect where Task 10's check isolates
    it.
- [ ] **Step 3:** Regenerate `dvc.yaml`. Update the README. Run the full suite. Commit.

### Task 14: The fresh-clone check

**Files:**
- Create: `pipeline/scripts/fresh_clone_check.sh STAGE...`. It:
  - clones the current branch from `origin` into an empty folder;
  - runs `pixi install`, `uv sync` and `uv run sfzmaps pull`;
  - runs `uv run sfzmaps build --force STAGE` for each named stage;
  - compares each stage's outputs with this checkout's, byte for byte, except the text stages' chance PNGs (within
    0.01 × 255) and model files (not compared).
- Modify: the README.

- [ ] **Step 1: Run it** after the controller has shipped (`sfzmaps ship` is the owner's; if nothing is shipped
  yet, the check runs from a clone that pulls nothing and builds every stage).
  - Stages: `solver-page solver-lots solver-prints solver-text-nw solver-labels solver-walls solver-solve`.
  - Report each stage: identical, within epsilon, or the differences.
- [ ] **Step 2:** Commit the script and the README.
