# Task 4 report: one `solver-text` stage, a scores stage, `text-trial`, `textgrid` with one model

Status: done. Branch `solver`, three commits (not pushed):

- `b0c5821` text detector: one solver-text stage, its scores on the test drawings, a trial command; the quarter models go
- `e61230f` fresh-clone check: the one solver-text stage, retrained in the clone (no quarters copied from the reference)
- `a3e425f` README: one text detector for both lettering styles, its validation area, its stages and text-trial

Tests: full suite `cd pipeline && uv run pytest -q`: 1082 passed, 3 skipped (the `mps` tests; a training holds the
GPU). Nothing ran on the GPU. No stage, warp, build or text-trial was run for real. `pipeline/dvc.lock` is not staged.

## The stages as generated (`pipeline/dvc.yaml`)

`solver-text` (`sfzmaps stage solver-text` → `sfzmaps solver text`). Upstream: the frozen stages that made the training
maps' pages (`period-2000-digital`, `period-1980-1990`, `period-1990-1996`, `edition-1960`, `edition-1964`,
`edition-1968`). Out: `data/solver/text`. Deps:

```
src/sfzmaps/__init__.py, config.py, transform.py
src/sfzmaps/solver/__init__.py, grid.py, paths.py, saved.py
src/sfzmaps/solver/textnet/__init__.py, boxes.py, data.py, model.py, predict.py, stage.py, train.py
pyproject.toml, uv.lock
data/work/<series>/<sheet>/page.png and district_labels.json   for each of the 87 sheets of paths.TEXT_SHEETS of
                                                              the 6 training maps (174 files), by series then sheet
saved-reads/answers/text-on-every-sheet.jsonl.gz
saved-reads/answers/experiments-2000/text-positions
```

No `data/warp` file: training with the validation area included reads no placement. I added
`data.load_sheets(..., placement=False)` (and `sheet_data`): `world` is empty `(0, 0, 2)`, `ft_per_px` nan;
`hold_out_validation=True` without a placement and `in_validation` on such a sheet raise. The crops and centres are
the same as with the placement (a test checks this). So no warp retrains the detector. The 2000 prediction reads only
the 13 pages, which are already among the reads.

`solver-text-scores` (`sfzmaps solver text-scores`). Upstream: `solver-text`, `warp-1980`, `warp-1972`, `warp-1975`.
Out: `data/solver/text-scores`. Deps:

```
src/sfzmaps/__init__.py, config.py, transform.py
src/sfzmaps/solver/__init__.py, frame.py, grid.py, page.py, paths.py, saved.py, strokes.py
src/sfzmaps/solver/textnet/__init__.py, boxes.py, data.py, model.py, predict.py, score_stage.py, scores.py, train.py
pyproject.toml, uv.lock
data/solver/text/model.pt
data/work/<series>/<sheet>/page.png, district_labels.json   for the 92 test sheets (1980, 1972, 1975)
data/warp/<series>/<sheet>/transform.json, correction.npz   for the same 92 sheets (for ft_per_px: long lines in feet)
saved-reads/answers/text-on-every-sheet.jsonl.gz
```

That makes 368 per-sheet files.

`solver-walls`: its upstream is now `("solver-labels", "solver-text")`, and it reads `data/solver/text/chance` where it
used to read the four `text-<q>/chance` folders. Everything else is unchanged.

Code cut: both text stages use `cut=(sfzmaps.gpu, sfzmaps.solver.runs)`. Ruling 1 asked for the atomic write through
`runs.replace_folder`, but importing `runs` would put `runs.py` (the walls' and solve's folders) in the detector's
closure. The old test and the README both say an edit there retrains nothing. So I cut `runs` on these two stages
only: they use nothing from it but the folder swap. If you'd rather keep `cut = gpu`, drop `"sfzmaps.solver.runs"`
from `text_cut` in `build.py` and the `in st.cut` assertions in two tests. The tests check that neither closure holds
`runs.py`, `gpu.py`, `walls.py`, `wallevidence.py`, `textgrid.py`, `solve.py`, `lots.py`, `table.py`, `codes.py`,
`prints.py`, `llm.py`, `cli.py` or `trial.py`. `solver-text` also holds none of `scores.py`, `score_stage.py`,
`strokes.py` or `page.py`.

## Code

- `textnet/stage.py`: `build_text(recipe=None → DEFAULT_RECIPE, device, seed)`.
  - It loads the 6 training maps (`hold_out_validation=False, placement=False`) before the turn.
  - In one `gpu_turn(True, device)` it fits, saves `model.pt` into `runs.fresh_folder(TEXT_DIR)`, reloads it with
    `predict.load_model`, and writes `chance/<sheet>.png` for `paths.SHEETS`.
  - Then `runs.replace_folder`.
  - It returns iterations, device, trained_on, scored, and seconds {load, train, predict}.
- `textnet/score_stage.py`: `build_scores(device)`.
  - It refuses without `paths.TEXT_MODEL`, then loads the test maps with their placements.
  - It predicts every sheet in one exclusive turn and runs `scores.score_sheets(chances, "whole", sheets)` on the CPU.
  - It writes `scores.json` alone into a fresh folder, so the old `evaluation_<q>.json` files go. The file holds no
    time.
- `textnet/trial.py`: `parse_recipe`, `folder`, `run_trials(recipes, device, seed, reuse)`.
  - Sheets are read once with the area held out; each sheet's validation px are found once (packed bits).
  - Per recipe: one exclusive turn (fit, save, predict only the tiles holding validation px), then score on the CPU
    (`line_ink` is memoized, so a sheet's long lines are found once for all the recipes).
  - It writes `model.pt`, `recipe.json` (`asdict(recipe)`, which round-trips), `scores.json` and `minutes.json`
    {load, train, predict, score}.
  - A trained recipe goes in a fresh folder that is then renamed into place. With `--reuse`, model.pt is checked to
    match the recipe's name and the seed before anything is read, and the folder's old `train` minutes are kept.
- `predict.chance(net, ink, device, only=None)`: tiles holding none of `only` are skipped (left 0). The px asked for
  get exactly what a whole-page prediction gives them (tested).
- `paths`: `TEXT_DIR`, `TEXT_MODEL`, `TEXT_CHANCE_DIR` replace `text_dir(q)` / `chance_dir(q)`.
- `textgrid.text_chance()`: one model, no `quarters` argument. It takes the sheets from `paths.check_sheets()` (so a
  sheet the warp placed but the detector didn't score gets a clear error), and each grid px takes the largest chance
  among the sheets that inked it, as before.
- `Recipe.name()` leaves out `prefetch` (`train.NOT_IN_NAME`).
- Guard: `test_the_stages_recipe_trains_the_weights_recorded` runs 3 steps of `DEFAULT_RECIPE` on the two synthetic
  maps on the CPU (about 12 s).
  - It checks the per-step losses within 1e-5 everywhere.
  - It checks a sha256 of the state_dict where one was recorded: `(Darwin, arm64)` with `OMP_NUM_THREADS` /
    `VECLIB_MAXIMUM_THREADS` at "1"/"1" (the pytest-xdist workers) and unset (`-n0`). The bits differ between those
    two thread settings, and CI runs on Linux x86, so a single hash can't hold everywhere.
  - When Task 5 sets `DEFAULT_RECIPE`, this test fails on purpose. Re-record it by printing the digest and losses.

## `text-trial` usage

```
uv run sfzmaps solver text-trial                                   # today's recipe → experiments/text-default/
uv run sfzmaps solver text-trial --set half_size=True --set ink_only=ink    # one recipe from settings
uv run sfzmaps solver text-trial --recipe "half_size=True ink_only=ink" --recipe "iterations=2000" --recipe default
uv run sfzmaps solver text-trial --recipe "half_size=True" --seed 1          # → text-half_size-seed=1/
uv run sfzmaps solver text-trial --reuse                            # score experiments/text-default/model.pt
```

- `--set` (repeatable) makes one recipe; each `--recipe` makes another, with its settings separated by spaces
  (`default` or `""` for today's). With both, the `--set` recipe runs first. Settings sit on top of `train.Recipe()`,
  the same base as the folder names.
- Switches accept true/false, yes/no, on/off or 1/0. Unknown or repeated names and refused values exit with a usage
  error before anything is read.
- Two recipes with the same folder (they differ only in `prefetch`) are refused.
- `--device cpu|mps`.

## Deleted

- Stages `solver-text-nw`, `-ne`, `-sw`, `-se`; `textnet/evaluate.py` (district lines, the per-quarter
  `evaluation_<q>.json`).
- `train.quarter_targets`, `_quarter_centres`, `train.crop` and the `SheetTargets` type check (fit takes `SheetData`
  only).
- `data.SheetTargets`, `targets`, `_fill`, `quarter_index`, `page_quarters`, `sheet_targets`, and `data.terra_boxes`
  (the prints-file version).
- `predict.predict_sheets`; `paths.text_dir`, `chance_dir`.
- The out-of-fold assembly and `quarters` argument in `textgrid`; the `solver text Q --iterations` CLI.
- Their tests:
  - the quarter-target, held-out-quarter-loss, quarter-stage, evaluate and district-line tests;
  - the out-of-fold textgrid test (replaced by a one-model test).
  - Two quarter-fixture tests were ported to the multi-map fixture: box, kind, angle and unknown marking, and the
    reading angle turning with the crop.
- `scripts/compare_outputs.py` and `fresh_clone_check.sh` (and their test) now name `solver-text`. The clone retrains
  it; the step that copied ne/sw/se from the reference is gone.
- Kept: `data.page_world`, `load_transform`, `warp_dir`, `other_boxes` (used by tests or `sheet_data`);
  `paths.QUARTERS` and `check_sheets` (the walls and solve use them).
- Old `data/solver/text-<q>/` folders are left for the controller.

## Concerns / notes

1. `solver text` lost `--iterations`. The stage trains `DEFAULT_RECIPE` only; experiments go through `text-trial`.
2. The scores stage keeps all 92 test chances in memory as uint8, about 2.8 GB. `score_sheets` takes them all at once,
   and I didn't edit `scores.py`. That's fine at 64 GB.
3. `prefetch` runs share the folder of the same recipe without prefetch (ruling 4), so a prefetch trial overwrites that
   folder's `minutes.json`. Copy the earlier minutes first if both are wanted.
4. Task 5 step 6 (modern maps only, compared on the 1980 drawing) isn't covered by `text-trial`. It scores validation
   only, on the fixed `TEXT_TRAIN_SERIES`, so that step needs a small addition (a `--maps` option, say) or a one-off
   in Task 5.
5. The spec says `solver-text-scores` scores "the validation sheets and the test drawings". Per ruling 2 it scores only
   the test drawings (whole): the shipped model trained on the validation area, so its validation score would be
   in-sample.
6. Any edit to `paths.py` reruns every solver stage. It already differs from the lock (Task 1), so this task adds no new
   rerun.
