# One Text Detector, Trained in Minutes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One text detector, trained on the 2000-style and the 1960s-style maps and tested on whole drawings it
never saw, in a few minutes per training run, built by one `solver-text` stage.

**Architecture:**
- **Training data:** `sfzmaps.solver.textnet` learns from many series' original pages, held out by a fixed area of
  the city.
- **Recipe:** training is driven by a `Recipe` (today's settings by default) whose options are the speed candidates.
- **Measurement:** a trial command measures a recipe on the validation area. A loop task picks the recipe by the
  spec's keep rule.
- **The build:** `solver-text` trains the shipped model, and `solver-text-scores` scores it on the test drawings.

**Tech Stack:** PyTorch 2.15 on the Mac's GPU (mps), numpy, OpenCV, Typer, DVC, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-text-detector-training-design.md` (read it first; it is the authority).

## Global Constraints

- **The maps:**
  - **train:** `2000-digital-zoning`, `1980-1990-zoning`, `1990-1996-zoning`, `1960`, `1964`, `1968`;
  - **test:** `1980`, `1972`, `1975`.
  - Only sheets Terra read are used. The sheet lists are fixed constants in code; `dvc.yaml` must not depend on data
    being present.
- **Validation area:** one named rectangle in map feet, about 10% of the mapped area. A page pixel is held out when
  its map point (through `data/warp/<series>/<sheet>/transform.json`) falls inside. Held-out pixels carry no loss,
  and crops don't centre on them.
- **Same answer every run:** two runs predict within 0.01.
  - Seeded; `torch.use_deterministic_algorithms(True)`; masked-sum losses (never boolean-index a tensor that needs a
    gradient on mps).
  - The run takes `sfzmaps.gpu.gpu_turn(True, device)` around training and prediction. Never run two mps jobs.
- **Scores**, the same measure for every comparison, on ink pixels:
  - AUC and log-loss;
  - text ink found;
  - words found, by kind;
  - ink outside every box called text;
  - long straight lines called text: strokes ≥ `LONG_LINE_FT = 200` long and ≤ `THIN_LINE_FT = 12` wide.
- **Keep rule:** validation words found and long-line ink called text are no worse within the tolerance. The
  tolerance is 2 × the two-seed spread of the fast recipe, with a floor of 0.5 points on words found.
- **Clock:** the training loop ≲ 5 min, the whole `solver-text` stage ≲ 10 min. If kept changes stall above that,
  stop and report; never loosen the quality bar.
- **Fresh clone only:** no paid or model reads; nothing from `pipeline/data/rnd`, scratch or another worktree; no
  per-item freshness stamps.
- **Process:**
  - Tests run on CPU (`cd pipeline && uv run pytest -q`; the conftest guard keeps tests off a busy GPU).
  - Never run `sfzmaps warp/build/solver text*` outside the task that says to.
  - Commits end with a `Co-Authored-By:` line naming the model you are, then
    `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.
  - Stage only your own files.

## Review Focus

1. **The validation area leaking through crop context.** A crop centred outside the area overlaps it, and those
   pixels must carry no loss. Test in Task 1.
2. **Maps at different scales:** the 1960s pages are about twice the pixels. The long-line constants are in feet and
   must convert per sheet through its transform. Test in Task 3.
3. **Memory:** 87 training sheets must not become full-page per-pixel target rasters (13 GB). Test in Task 1
   (target bytes per sheet bounded); measured in Task 5.
4. **Prefetching changes the batches.** With the background thread on, the weights must be bit-identical to off.
   Test in Task 2.
5. **Half-size folding at odd sizes.** The fold needs even sizes, so crops and prediction tiles must be even and the
   page's odd edge handled. Test in Task 2.

---

### Task 1: Training data from many maps, held out by an area

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/data.py`
- Modify: `pipeline/src/sfzmaps/solver/paths.py` (the series and sheet lists)
- Create: `pipeline/src/sfzmaps/solver/textnet/boxes.py` (Terra boxes for any series, from the saved answers)
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Produces:**
  - **In `paths`:** `TEXT_TRAIN_SERIES`, `TEXT_TEST_SERIES` (tuples as in Global Constraints) and
    `TEXT_SHEETS: dict[str, tuple[str, ...]]`, the fixed sheet names Terra read per series.
  - **`boxes.terra_boxes(series, sheet) -> list[dict]`:** page-px boxes, decoded as `prints.terra_boxes` decodes
    2000. Reuse it; generalise it, don't copy it.
  - **In `data`:**
    - `VALIDATION_AREA = (x0, y0, x1, y1)` in map feet. Pick a block of about 10% of the mapped area that every
      training map covers, and say how in a comment.
    - `SheetData`, compact per sheet: ink (bool or packed), the box list, unknown regions, the held-out mask (or a
      way to compute it per crop), and `ft_per_px`.
    - `load_sheets(series_list, hold_out_validation: bool) -> list[SheetData]`.
    - `crop_targets(sheet, y, x, size, scale, turns) -> dict` draws the text / kind / angle / known targets for one
      crop from the boxes. No full-page target rasters.

- [ ] **Step 1: Write failing tests**, on synthetic sheets:
  - `test_terra_boxes_decode_for_another_series`: a 1960-style record with a page_box gives page-px boxes.
  - `test_validation_area_pixels_carry_no_loss_even_inside_a_crop_centred_outside` (Review Focus 1).
  - `test_crop_targets_equal_full_page_drawing` (the per-crop drawing equals drawing the whole page and cropping).
  - `test_sheet_data_holds_no_full_page_target_rasters`: per-sheet bytes ≤ 1.2 × the ink bytes plus the boxes
    (Review Focus 3).
  - `test_text_sheet_lists_match_the_saved_answers`: built from a tmp saved-answers file, not real data.
- [ ] **Step 2:** Implement; the tests pass. Keep the 2000-only path working until Task 4 removes the quarter
  stages.
- [ ] **Step 3:** Real check, CPU only: `load_sheets(TEXT_TRAIN_SERIES, True)`. Report the seconds, the peak memory,
  the sheets per series, and the validation share of the text pixels per series.
- [ ] **Step 4:** Full suite. Commit.

### Task 2: The training recipe and its options

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/train.py`, `model.py`, `predict.py`
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Consumes:** Task 1's `SheetData`, `load_sheets` and `crop_targets`.
- **Produces:** a frozen dataclass `Recipe`, whose defaults are today's settings:
  - `iterations=8000`, `max_lr=2e-3`, `pct_start=0.3`, `batch=12`, `crop=320`, `width=24`;
  - `ink_only: str = "off"` (`"off" | "ink" | "ink+1px"`);
  - `half_size: bool = False`;
  - `heads: str = "all"` (`"all" | "text"`);
  - `weight_average: float = 0.0` (EMA decay; 0 is off);
  - `prefetch: bool = False`;
  - `decoder: str = "full"` (`"full" | "light"`).

  Also `fit(sheets, recipe, device=None, seed=0) -> TextNet` and `Recipe.name()`, a stable short id used in folder
  names.
- **Details:**
  - **`half_size`:** `pixel_unshuffle(2)` on the ink (4 channels), the same U-Net, a head with 4× the outputs, then
    `pixel_shuffle(2)`. Crops and prediction tiles stay even.
  - **`prefetch`:** a background thread builds batches from the one seeded rng, in order.
  - **No per-step `.item()`:** losses accumulate on the device and are logged every `LOG_EVERY`. The finite check
    runs at log time.
  - **`heads="text"`:** a 1-channel output, with no kind or angle loss.
  - **Prediction** reads `Recipe` from the saved model, so `predict` works for any recipe.

- [ ] **Step 1: Write failing tests**, on synthetic data, CPU:
  - `test_default_recipe_trains_as_today`: the default recipe's loss on a fixed batch equals the current `loss_fn`'s.
  - `test_half_size_fold_is_lossless_and_shapes_round_trip`, including an odd page size at prediction (Review
    Focus 5).
  - `test_ink_only_loss_ignores_paper`: changing the targets on paper pixels changes nothing.
  - `test_prefetch_gives_bit_identical_weights` over 5 iterations (Review Focus 4).
  - `test_text_head_only_predicts_text`.
  - `test_weight_average_is_the_ema_of_weights`.
  - `test_recipe_round_trips_through_the_saved_model`.
- [ ] **Step 2:** Implement; the tests pass. Then real-data timing, CPU only: 20 iterations of the default recipe
  and of `half_size=True`. Report the seconds per iteration for both.
- [ ] **Step 3:** Full suite. Commit.

### Task 3: Scores on the validation area and the test drawings

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textnet/scores.py`, replacing `evaluate.py`'s scoring. `evaluate.py` goes,
  along with its callers.
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Consumes:** Task 1's data, Task 2's `predict` (chance per sheet), and `sfzmaps.solver.strokes` (the skeleton
  strokes, reused for the long-line measure).
- **Produces:**
  - `score_sheets(chances: dict[(series, sheet), np.ndarray], where: "validation" | "whole") -> dict`, per series
    and overall: AUC, log-loss, text ink found, words found by kind, ink outside boxes called text, long-line ink
    called text.
  - `LONG_LINE_FT = 200.0`, `THIN_LINE_FT = 12.0`, converted per sheet with `ft_per_px`.

- [ ] **Step 1: Write failing tests:**
  - `test_long_line_measure_uses_feet_per_sheet` (Review Focus 2): the same drawing at 2 scales gives the same
    lines.
  - `test_validation_scores_use_only_validation_pixels`.
  - `test_words_found_counts_a_box_with_half_its_ink_called_text`: the rule today's evaluate uses, kept.
- [ ] **Step 2:** Implement; the tests pass. Full suite. Commit.

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

### Task 5: The measured loop and the chosen recipe

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/stage.py` (`DEFAULT_RECIPE`), `pipeline/README.md` (the results
  table)
- Scratch: the per-run results and the minutes stay in `data/solver/experiments/` and the report.

The GPU runs one job at a time. Run each `text-trial` in the foreground or wait on it, never two at once.

- [ ] **Step 1: Baseline.** `text-trial` with the default recipe (today's). Record the minutes and the validation
  scores.
- [ ] **Step 2: The big levers first.** These make every later run cheaper:
  - `half_size=True`;
  - `ink_only=ink` and `ink_only=ink+1px`;
  - then the best of those together.

  Keep each by the keep rule against the current best.
- [ ] **Step 3: The tolerance.** Two seeds of the current fast recipe give the spread; tolerance = max(2 × spread,
  0.5 points on words found). Re-judge Step 2's decisions with it if any were within it.
- [ ] **Step 4: The schedule.**
  - `iterations` ∈ {2000, 4000, 8000} × `max_lr` ∈ {2e-3, 6e-3}, with `pct_start=0.1`;
  - then `weight_average=0.999` on the best.

  Keep by the rule, preferring the shortest that passes.
- [ ] **Step 5: The rest.** In order: `heads="text"`, then `prefetch=True` (it must give identical scores; it's
  speed only), then `decoder="light"` only if the clock isn't met.
- [ ] **Step 6: The mixed-style check.** Train the chosen recipe on the modern-lettering maps only and compare it
  with the both-styles model on the 1980 test drawing. If both-styles is worse beyond the tolerance, report it and
  stop before Step 7.
- [ ] **Step 7: Lock it in.**
  - Set `DEFAULT_RECIPE` to the chosen recipe.
  - Run `uv run sfzmaps solver text` twice: the chances agree within 0.01 and the minutes are within the clock.
  - Run `uv run sfzmaps solver text-scores`.
  - Write the README table: each step's minutes and validation scores, the test scores, and today's quarter models
    for context.
  - Full suite. Commit.
