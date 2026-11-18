### Task 7: Variant 1, the decision as written, and SW

**Files:**
- Create: `pipeline/scripts/textlines/decide.py`, `docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md`
- Test: `pipeline/tests/test_textlines_decide.py`

**Interfaces:**
- Consumes: `linegate.paired`, the run folders.
- Produces: `decide.verdict(control: list[Path], variant: list[Path], baselines: dict[str, Path], eye_ok: bool |
  None) -> dict`. It returns one of the verdicts below, with every condition's numbers.

**The rule, written before the runs**
- **Adoptable,** for the control or the variant, against `baselines/r328` and the old quarter model, on NE and on SW:
  - M1a-in-box and M1b-in-box are lower, with the paired interval excluding zero;
  - M2 is not lower by more than the noise;
  - M1c is not higher by more than the noise;
  - the audit's count of text pieces called text is not lower beyond its binomial interval.

  The all-sheets detector is a reference only (in-sample).
- **Noise** = max(the control's largest pairwise seed difference, the paired interval's half-width).
- **Variant over control** requires all of:
  - M1a-in-box lower, with the paired interval excluding zero. M1a carries the verdict, since M1b shares the
    training's line rule.
  - M2 not lower by more than the noise;
  - M1c not higher by more than the noise;
  - `eye_ok` True: the controller looked at both panels and wrote what they show.
- **The verdicts:**
  - `pending-eye` when `eye_ok` is None;
  - `mixed` if M1b-in-box falls and M1a-in-box does not, or on any other split;
  - `control` or `variant` otherwise.

- [ ] **Step 1: Write the failing tests** for `verdict` with fake run folders (synthetic `blocks.npz`):
  - better lines with the same text and `eye_ok` True → `variant`;
  - less text beyond noise → `mixed`;
  - `eye_ok` None → `pending-eye`;
  - M1b down with M1a flat → `mixed`.

- [ ] **Step 2: Run them to verify they fail.** Expected: FAIL.

- [ ] **Step 3: Implement, then run one at a time.**
  1. `variant.py composites --quarter ne --seed 0 --labels lines-unknown --composites P`, with P =
     `distributions.json`'s word-weighted `cross_by_kind` over all kinds;
  2. the same at `--seed 1` and `--seed 2`;
  3. look at the panels, then `decide.verdict(...)` on NE;
  4. the NE winner and the control on SW, at seed 0 each, decided again on SW.

  Write the results note:
  - the baselines and every run's tables;
  - the noise, and the verdicts, with the controller's description of each panel;
  - the audits' counts with their binomial bounds;
  - the 1990-96 numbers;
  - the paths of every picture.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** (after the T3 check): `git add pipeline/scripts/textlines/decide.py
  pipeline/tests/test_textlines_decide.py docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md`, then
  commit "textlines: variant 1 against control 0 on NE and SW, the decision as written, results note".


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

