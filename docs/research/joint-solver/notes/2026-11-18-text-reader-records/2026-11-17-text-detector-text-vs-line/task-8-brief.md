### Task 8: Preview downstream for the owner (no stage change)

Run only if Task 7 found an adoptable recipe. Otherwise skip, and say why in the results note.

**Files:**
- Create: `pipeline/scripts/textlines/preview.py`

**Interfaces:**
- Consumes: the adoptable recipe; `reader_stage` (its training loop, `sheet_outputs`); `walls.build_walls`
  (candidates, its own folder); the solve's entry point as `criteria_check` runs it.
- Produces: `experiments/textlines/preview/`:
  - `text-reader/`: four held-out detectors with the recipe, chance as the stage writes it;
  - `walls/`: fitted on that chance as an experiment;
  - `solve/`: today's solve inputs with those walls.
  - `before-t8/`: today's `data/solver/solve/` and `walls/`, copied first.

- [ ] **Step 1: Copy `before-t8`.**
- [ ] **Step 2: Train and predict the four detectors** (one GPU turn, about 28 min), then gate them on all four
  quarters and render the panel.
- [ ] **Step 3: Fit the walls and solve** into the preview folder.
- [ ] **Step 4: Measure M5:**
  - word boxes via `wordboxes` on the preview chance (T2's measures);
  - the walls' held-out log loss;
  - wall evidence on district borders inside word boxes;
  - the solve's six measures with `criteria_check --baseline before-t8`, plus counted mistakes.

  Wrong lots are reported, but are not the verdict. Add all of it to the results note.
- [ ] **Step 5: Write the adoption path into the note.** Adopting would take one commit: the recipe in
  `reader_stage`, with `held.json` keeping `unknown_radius` for `solver-text-crops` and adding `zone_radius`; then
  `pixi run build solver-solve solver-boxes`.
  - List what that would rebuild, from `uv run sfzmaps build --dry-run`.
  - State that T3's crops and training stay untouched.
  - State that `solver-text` (no placements by design) keeps the boxes rule until the owner decides.

  Commit the script (after the T3 check).

Stop here. Adoption, and T4 and T5 of the text-reader plan, wait for the owner.

## Global Constraints

- **T3 first, and no commits during builds.**
  - T3 of the text-reader plan builds stages in this same worktree. A commit, or a dirty or untracked file under
    `pipeline/src`, `inputs`, `dvc.yaml`, `pyproject.toml` or `uv.lock`, makes a running `sfzmaps build` refuse to
    record (`src/sfzmaps/build.py:509,624,665,706-711`).
  - Before creating or editing any such file, and before any commit: `pgrep -fl 'sfzmaps (build|stage)'` must be
    empty, and the T3 ledger (`.superpowers/sdd/2026-11-16-text-reader-labels-and-publish/progress.md`) must show
    T3's builds recorded.
- **No stage's import closure changes.**
  - `textnet/data.py` and `train.py` gain no new imports. `linerules.py` imports nothing from `sfzmaps.solver` but
    `grid`. The label rule's data reach `crop_targets` through `SheetData` fields filled before training. Composites
    reach `draw` through an object passed to `fit`.
  - Check after every task: `uv run pytest -q tests/test_build.py`.
  - Don't edit `build.py`, `dvc.yaml`, `dvc.lock`, `pipeline/README.md`, any file directly in `src/sfzmaps/` (e.g.
    `cli.py`), `pyproject.toml` or `uv.lock`.
- **Staleness is expected.** After Task 4, six stages (`solver-text`, `solver-text-scores`, `solver-text-reader`,
  `solver-boxes`, `solver-text-crops`, `solver-text-read`) show stale in `dvc status` because their code changed. The
  default recipe is unchanged. Nobody builds them in this plan; the morning report lists them.
- **$0.** No OpenAI/LLM calls. Never print, log, copy or commit `pipeline/.env`, `site/.env.local` or any token.
- **One GPU (mps) process on the machine at a time:** every training and prediction runs inside
  `gpu.gpu_turn(True, device)`. The pytest session holds a GPU turn too (`conftest.py:44-60`), so don't run the
  suite while a variant trains.
- **Load near 20:** ≤ 4 worker processes × 1 thread.
- **Determinism:** seeded. The batch thread's rng draws every crop and composite; `Bank.build` uses its own rng
  `np.random.default_rng([seed, 1])`.
- **Held out by quarter** (`data.Quarter`). The owner's answered blocks (`reader_stage.held_blocks()` /
  `load_block_area`) are never trained on, never used for composites, and never in a bank.
- **No LLM output at inference.** Terra's and Sol's saved answers only train and score.
- **Measures:**
  - never define line ink by Terra boxes or a training labeller;
  - nothing is excluded near labels;
  - every line rule prints its exact-truth accuracy beside its numbers;
  - no verdict from wrong-lot counts or one seed.
- **Apache-2.0 code;** no GPL-family dependency.
- **Tests:**
  - CI has no `pipeline/data`. Tests use the synthetic `maps` fixture (`tests/test_solver_textnet.py:244`) or arrays
    built in the test.
  - Real data are injected (`geo=`, `sheets=`, `areas=`, `score=` arguments). A check needing real files skips when
    they are absent, as `tests/test_solver_textread.py:152` does.
  - Glyphs in unit tests are drawn with `cv2.putText`.
- **Test command,** while iterating:
  `uv run pytest -q tests/test_solver_textnet.py tests/test_codereader.py tests/test_solver_textread.py
  tests/test_build.py tests/test_textnet_*.py tests/test_textlines_*.py`. Before each commit, `uv run pytest -q`
  (no variant training at the time).
- **Data:** outputs under `pipeline/data/solver/experiments/textlines/`, never committed. No bare `git stash`, no
  `dvc gc`.
- **Git:** `git add <paths>` and `git commit` as separate plain commands. Every commit message ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A chance file missing** for a sheet, or covering part of it: the gate stops with the sheet's name. It never
   scores fewer px (Task 2 test).
2. **A Sol point whose Terra word sits on another sheet** (seams): matching is per sheet (Task 4 test).
3. **A turned word box running off the page:** the in-box rule clips it as `_paint` does (Task 4 test).
4. **A clean word that fits no place in its crop:** the next candidate is drawn, from the same rng (Task 6 test).
5. **A new recipe field leaking into today's runs:** `labels="boxes"` and `composites=0.0` give byte-identical
   batches and the same run name (Tasks 4 and 6 tests).

