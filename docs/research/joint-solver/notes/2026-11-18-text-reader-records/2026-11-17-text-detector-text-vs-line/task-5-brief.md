### Task 5: One held-out quarter, trained and gated: control 0, value first

**Files:**
- Create: `pipeline/scripts/textlines/variant.py`
- Test: `pipeline/tests/test_textlines_variant.py`

**Interfaces:**
- Consumes: `train.fit`, `train.save_model`, `predict.load_model`, `data.hold_out`, `reader_stage.held_blocks` /
  `load_block_area`, `zones.prepare`, `linegate.gate` / `write`, `panel.render`, `audit.sample`.
- Produces:
  - `run(name: str, recipe: train.Recipe, quarter: str, seed: int, device=None, sheets=None, areas=None,
    score=None, bank=None, extra_series=()) -> Path`:
    - it trains with `quarter` and the answered blocks held out;
    - it predicts chance on the 2000 sheets for that quarter's px (0 elsewhere), and on `extra_series` sheets' px of
      that quarter, if any;
    - it gates the result and renders the panel and the audit;
    - it writes `experiments/textlines/runs/<name>-<quarter>-s<seed>/{model.pt, chance/, scores.*, blocks.npz,
      panel.png, audit.png, run.json}`. `run.json` holds the recipe, seed, device, training minutes, git head and
      `git status --porcelain -- src` at the start.
  - CLI: `uv run python scripts/textlines/variant.py <name> --quarter ne --seed 0 [--labels lines-unknown]
    [--composites P] [--extra 1990-1996-zoning]`.

- [ ] **Step 1: Write the failing test** on the synthetic `maps` fixture, with `score=` replaced by a stub and a
  2-step CPU recipe: the folder is `t-ne-s0` and holds `model.pt`, `scores.json` and `run.json`.

- [ ] **Step 2: Run it to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement.** The base recipe is `stage.DEFAULT_RECIPE` with `labels` (and `composites`) from the CLI.
  `zones.prepare` and the bank run before the turn; training and prediction run inside one `gpu.gpu_turn`.

- [ ] **Step 4: Run the test, then the control, value first, one run at a time.**
1. `variant.py control --quarter ne --seed 0 --labels lines-unknown --extra 1990-1996-zoning`.
   - Gate it against `baselines/r328` and the old NE quarter model with `linegate.paired`.
   - Look at the panel.
   - **If M1a-in-box and M1b-in-box are not lower than both baselines** (paired interval excluding zero), stop the
     plan's training here and write up why.
2. `variant.py control --quarter ne --seed 1 ...`, then `--seed 2`.
   - The noise is each measure's largest pairwise difference over the three seeds.
   - Report the three runs, the noise, and the 1990-96 NE numbers (M1b, M1c, M2; there is no M1a without a table).

- [ ] **Step 5: Commit** (after the T3 check; not during a run): `git add pipeline/scripts/textlines/variant.py
  pipeline/tests/test_textlines_variant.py`, then commit "textlines: train and gate one held-out quarter; control 0 on
  NE".


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

