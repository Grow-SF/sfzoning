### Task 4: One `solver-text` stage, a scores stage, a trial command, `textgrid` with one model

**Files:**
- Modify: `pipeline/src/sfzmaps/build.py`, `pipeline/src/sfzmaps/cli.py`,
  `pipeline/src/sfzmaps/solver/textnet/stage.py`, `pipeline/src/sfzmaps/solver/textgrid.py`,
  `pipeline/src/sfzmaps/solver/walls.py` (only where it reads the text chance), `pipeline/README.md`, `dvc.yaml`.
- Test: `pipeline/tests/test_build.py`, `test_solver_textnet.py`, `test_solver_walls.py`

**Interfaces:**
- **Stages:**
  - `solver-text`: command `("solver", "text")`. Reads the named page, label and box inputs of the training sheets.
    Writes `data/solver/text/` (`model.pt`, `chance/<2000 sheet>.png`).
  - `solver-text-scores`: command `("solver", "text-scores")`. Upstream `solver-text`. Writes
    `data/solver/text-scores/scores.json`, the test drawings' scores.
  - The four `solver-text-{q}` stages and the out-of-fold assembly go.
- **`stage.build_text(recipe=DEFAULT_RECIPE)`** trains on every training sheet, the validation area included, then
  predicts the 2000 sheets, in one GPU turn.
- **`DEFAULT_RECIPE`** lives in `textnet/stage.py`: today's recipe, until Task 5 sets it.
- **`sfzmaps solver text-trial [--set key=value ...]`** trains with the validation area held out. It scores the
  validation area and writes to `data/solver/experiments/text-<recipe name>/` (not a stage output, not shipped):
  `scores.json`, `recipe.json` and `minutes.json`.
- **`textgrid.text_chance()`** samples `data/solver/text/chance/` (one model). The walls stage's reads follow.

- [ ] **Step 1: Write failing tests:**
  - the stage deps: no quarter stages; the named files; no `paths` / `runs` churn;
  - `text-trial` writes only under `experiments/`;
  - `textgrid` with one model;
  - the walls deps updated.
- [ ] **Step 2:** Implement; the tests pass. Regenerate `dvc.yaml`. Full suite. Commit.
  - Don't run the stages (Task 5 and the controller do).
  - The old `data/solver/text-<q>/` folders are left for the controller to remove.

