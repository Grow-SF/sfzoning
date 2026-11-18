# Final review: one text detector, trained in minutes (8f0103b..a6a4c6d)

Scope: the detector plan's commits only (b5abd6d, 78c74c1, ccfe2b5, cb12116, 6f6f4be, 9960f19, e7550a4, b0c5821,
e61230f, a3e425f, d8044f9, ca8984f, 1bcd48d, a201d3d, ad9db9f, a6a4c6d). The code-reader commits (66e1c9a, f37c575,
5209120, fda4c42) were skipped. I checked only that they don't touch detector files: they import `textnet.data` and
`textnet.scores` read-only and never use torch or the GPU. I read the final versions of every textnet module,
`gpu.py`, `textgrid.py`, the `build.py`/`paths.py`/`runs.py`/`cli.py` diffs, the generated `dvc.yaml` entries, the
conftest, the detector tests, the README's detector section, the spec, the plan and the ledger. No tests or GPU work
were run.

## Strengths

- **Compact data, per-crop targets.** `SheetData` keeps the ink as packed bits and the boxes as short arrays.
  `_paint` draws each crop on a canvas holding every shape that reaches it. Tests check that the crop equals the
  whole page drawn and cropped, and that no full-page target rasters are kept (Review Focus 1 and 3).
- **One definition of the validation area.** The training mask (`_held`), the crop centres (`centres`) and the
  scoring scope (`scores.scored_px`) all go through `data.in_validation`. Its fast paths are sound: a px's map point
  is a convex mix of the four samples around it, and the 1-ft margin covers rounding. Words on the area's edge are
  counted by their centre and judged only on ink inside the area, so no trained px is scored.
- **No leakage into the test drawings.** `training_maps` only accepts `TEXT_TRAIN_SERIES`, and the test drawings are
  never in a fit. The stage's shipped model is scored only on the test drawings, as the plan says, since it trained
  on the validation area.
- **GPU turns, checked by tests.**
  - The stage holds one exclusive turn around fit, save, reload and predict.
  - The scores stage and each trial recipe hold their own turn.
  - CPU scoring happens outside the turn.
  - The tests record the turn sequence and which calls happen inside it (`_turns`, `_inside`).
  - Every test that trains or predicts passes `device="cpu"`, so the conftest's shared session turn is never unsafe.
- **Non-blocking copies done right.** `Copies` matches how mps queues a non-blocking copy: it wraps the host memory
  without copying and reads it later. Each CPU batch is held until an event recorded after its copy has passed.
  `close()` runs on errors too, through `contextlib.closing`.
- **Prefetch.** The worker thread is the only user of the one numpy generator, so the batches come out in the same
  order. There is a bit-identity test and a test of which thread drew each batch.
- **Byte-stable outputs.**
  - Stage outputs are swapped into place whole (`fresh_folder` and `replace_folder`).
  - `scores.json` holds no time, and its tallies are integer counts divided at the end.
  - `model.pt` holds the recipe, iterations, device and seed, but no time.
  - Two stage runs were measured bit-identical.
- **Narrow, tested build deps.**
  - `solver-text` reads, file by file: each training sheet's `page.png` and `district_labels.json`, Terra's answers
    file, and the other passes' folder.
  - Its code is the textnet closure without `scores`, `trial` or `score_stage`. Cutting `sfzmaps.transform` is
    sound: that module imports only numpy and scipy, and no code path runs it when `placement=False`.
  - Tests assert both what is in the deps and what is not.
- **Honest README.** The trial tables give each verdict. The seed instability at 2,000 and 3,000 steps is shown. The
  missed 5-minute training target is reported, not hidden behind a looser bar.
- **Default-recipe guard.** It checks the recipe's name, the per-step losses everywhere, and a weight digest per
  platform, so a change in how the detector trains shows up.

## Issues

### Critical (Must Fix)

None found.

### Important (Should Fix)

**1. The mixed-style gate was run on the rejected 2,000-step recipe, one seed, not on the shipped recipe.**
`pipeline/README.md:672-675`; ledger Task 5 BLOCKED entry; `task-5-results.md` rows 16–17.

- **What's wrong.**
  - The spec ("the same recipe is also trained on the modern-lettering maps only") and plan Step 6 ("Train the
    chosen recipe…") make this a ship gate. If both-styles is worse on 1980 beyond the tolerance, it goes to the
    owner before anything ships.
  - The check was run once, seed 0, while 2,000 steps was briefly the choice. The choice then moved to 4,000 steps,
    because single seeds at 2,000 steps proved unreliable: up to 1.5 points apart on words found, by the README's own
    account.
  - The measured margin was +0.67 points on words (92.72 vs 92.05). That is within that seed noise. No modern-only
    run exists for the 4,000-step recipe.
- **Why it matters.** The shipped detector's gate rests on a recipe and a single seed that the loop itself judged
  not good enough to decide on.
- **Fix.** One trial: `sfzmaps solver text-trial --recipe "half_size=True ink_only=ink+1px iterations=4000
  pct_start=0.1 prefetch=True" --train-series 2000-digital-zoning,1980-1990-zoning,1990-1996-zoning --test-series
  1980`. On three maps it trains faster than the full stage. Compare it with the 4,000-step both-styles row already
  in the README (92.67% / 4.83%). Then update the "One detector for both styles" paragraph to the shipped recipe.

**2. "Today's recipe" now names the retired recipe, and `text-trial` builds on it.**
`pipeline/src/sfzmaps/cli.py:1038, 1041, 1056`; `trial.py:27, 111-112`; `train.py:1, 73, 134`; `stage.py:28-29`;
`pipeline/README.md:605-606` (and the table rows "today's, `Recipe()`").

- **What's wrong.**
  - After the lock-in, the recipe in use today is `stage.DEFAULT_RECIPE`. Yet the CLI help, the README and the
    docstrings still call `Recipe()`, the old 8,000-step full-size recipe of the quarter models, "today's".
  - So `sfzmaps solver text-trial` with no settings ("today's recipe") runs the 45-minute baseline.
  - `--set iterations=3000` varies that baseline, not the shipped recipe.
- **Why it matters.** A reasonable user reading "today's" expects the current recipe. An accidental run costs about
  45 GPU-minutes and gives numbers for the wrong recipe. This is the kind of claim from before the speed work that
  focus area 5 asks about.
- **Fix.** At minimum, rename the wording to "the quarter models' recipe (`Recipe()`)" or "the old recipe" in the
  help, docstrings and README. Optionally, add `--recipe stage` (or make `default` mean the stage's recipe). Keep
  `Recipe.name()` relative to `Recipe()` so existing experiment folders keep their names.

### Minor (Nice to Have)

1. **Stale speed claim in a docstring.** `train.py:27-28` says float16 was "measured 22% faster per step on the M1
   Max". The ledger and the README (`:692`, and the candidate table) found no gain in the real loop: 104 vs 103–107
   ms. The 22% came from a fixed-batch micro-benchmark. Say so, or say "no faster in the training loop", so nobody
   re-tries it on the docstring's word.
2. **README timing that predates the speed work.** `README.md:698-700` gives "0.39 s a step … about 52 min against
   6.9 h for 8,000 steps". That is the old full-size `Recipe()`, but it reads as the detector's speed today. Label it
   as the old recipe (the current step is about 0.1 s, `:690`).
3. **Ambiguous tolerance sentence.** `README.md:627`: "(twice the spread, 0.12, is below the floor)" reads as if
   twice the spread were 0.12. The spread is 0.12 (95.01 − 94.89), and twice it is 0.24. Write "twice the spread of
   0.12, 0.24, is below the floor", matching the long-line clause.
4. **Wording.** `README.md:707-708`: "its scores stage and each trial's recipe predict in one". Each trial recipe
   trains and predicts in one turn.
5. **The turn rule is enforced only by convention.** `fit` and `predict.chance` say "the caller holds the turn" but
   don't check. Every in-repo entry point is correct. A direct call on mps from a script or notebook (the Task 5 loop
   ran several timing scripts) would compute beside a stage, which is the silent-corruption case `gpu.py` exists for.
   A cheap guard: `gpu` keeps a process-local "exclusive held" flag, and `fit` / `chance` raise on mps without it.
6. **Training-only edits rerun the walls.** `textgrid.py:53, 65` imports `textnet.data` only for `load_page`. That
   puts the training-data code (`VALIDATION_AREA`, crop drawing, centres) in `solver-walls`' code deps, so a
   training-only edit reruns the walls even when the chance bytes don't change. Fix: move `load_page` to a light
   module, or read the page with `cv2` in textgrid.
7. **Test-only helpers in production modules.** `train.to_tensors` and `data.page_world` are used only by tests
   (`fit` uses `Copies`; the mask uses `_world_at`). Keep them as reference implementations if intended, but say so,
   or move them into the tests.
8. **Test file imports another test module.** `tests/test_solver_textnet_precision.py` imports `OLD`, `SMALL` and the
   `maps` fixture from `test_solver_textnet`. Fold its three tests into that file, or move the fixture to a conftest.
9. **Private dict reached from outside.** `trial.read_line_masks` and `keep_line_masks` write and read
   `scores._LINE_INK` directly. A small public put/get in `scores` would keep the cache's key rules in one module.
10. **Unprotected cache folder name.** `experiments/line-masks` isn't covered by `runs.solve_dir`'s prefix refusal,
    so a solve run named `line-masks` would replace it. Only a cache, minutes to rebuild.
11. **Kept options from rejected candidates.** `heads="text"`, `weight_average`, `decoder="light"` and
    `precision="float16"` are kept as tested recipe options for `text-trial`. That seems reasonable for the owner's
    "iterate on approach" goal: each is small and tested, and with the defaults the float16 `GradScaler` is a no-op.
    The parked float16 minors (the schedule advances on skipped steps; the load test doesn't assert that precision
    round-trips) should be addressed when the option is next used.

## Declined to judge

- **`dvc.lock`, `sfzmaps ship`, the fresh-clone run and the PR.** The spec puts them in "Afterwards", and the ledger
  carries `dvc.lock` to the end. The worktree's uncommitted `dvc.lock` is outside the range.
- **Rerunning the walls' text candidates on the new chance.** A spec non-goal (afterwards).
- **The code-reader commits.** Out of scope; I checked only that they don't touch detector files or the GPU.
- **Several warp processes on mps at once.** Shared turns let them compute side by side (`gpu.py`, the warp's turn
  in `cli.py:814`), which `gpu.py`'s own docstring says can give wrong numbers. This predates the range; worth the
  owner's attention separately.
- **pytest-xdist workers running `mps`-marked tests at once under shared turns.** The same hazard, and the conftest
  design predates this plan.
- **A run on a machine without mps** (training on CPU: different weights, hours long). Per-platform outputs were
  accepted in the ledger.
- **The losses-only guard within 1e-5 on Linux CI.** The ledger defers it to the PR's CI.
- **Whether 1980 and 1972/1975 are truly separate drawings** from the training ones. The spec's data table asserts
  it, and the code can't show it.
- **Leakage at the validation rectangle's edge** where reprints register a few feet apart. This is inherent to the
  spec's area design, and the band is a small share of the area.
- **The 5-minute training target missed** (6.2 min idle). The spec says stop and report, which the README does.
- **The keep rule changed to "every seed passes against today's recipe".** A deliberate ledger ruling.
- **Test-drawing scores seen mid-loop.** The ledger ruling forbids choosing on them, and the choice was made on
  validation seeds.
- **Tools from pixi (OpenCV) and `config.py`, `grid.py`, `pyproject.toml`, `uv.lock` in every solver stage's deps.**
  A pre-existing pattern shared by all the solver stages.
- **`correction.npz` named as a dep of each test sheet.** It fails loudly if a sheet is ever left uncorrected; all 92
  test sheets have one now. The pattern predates this plan.
- **The spec's "scores stage predicts the validation sheets".** The plan narrowed the stage to the test drawings, and
  a model trained on the validation area can't be scored there.
- **`THIN_LINE_FT` at 48 instead of 12.** A ledger ruling, and the spec was amended (9960f19).

## Assessment

**Ready to merge? With fixes.**

**Reasoning:** The code is sound:

- runs are deterministic end to end, with bit-identical stage runs;
- the GPU turns are correct on every in-repo path;
- the build deps are narrow and tested;
- nothing leaks between training, validation and test.

Two things should be fixed first. The mixed-style gate has to be rerun on the shipped 4,000-step recipe (one short
trial). And "today's recipe" should stop naming the retired recipe in the CLI help, docstrings and README.
