# The detector plan's final fix round

The whole-branch review is at `.superpowers/sdd/2026-11-15-text-detector-training/final-review.md`. Read its Issues
section; the item numbers below are its own.

## Fix these

**Important 2: "today's recipe" now names the retired recipe.**
- `Recipe()` is the old quarter models' recipe: 8,000 steps, full size, 45 minutes. The build's recipe is
  `stage.DEFAULT_RECIPE`.
- Rename the wording everywhere the review lists to "the quarter models' recipe" (`Recipe()`), or "the old recipe"
  where shorter reads better:
  - `cli.py:1038, 1041, 1056`;
  - `trial.py:27, 111-112`;
  - `train.py:1, 73, 134`;
  - `stage.py:28-29`;
  - `pipeline/README.md:605-606`, and the README table rows "today's, `Recipe()`".
  - grep for any other "today's" that means `Recipe()`.
- Keep the README's comparisons meaningful. "Today's recipe" was the baseline every candidate was held to. Call it
  "the old recipe (the quarter models')" consistently, and keep the numbers.
- Make `text-trial` start from the stage's recipe:
  - A `--recipe` value of `stage` (alone, or as the first token, e.g. `"stage iterations=2000"`) starts from
    `stage.DEFAULT_RECIPE` and applies the settings after it.
  - A recipe without `stage` keeps today's behaviour, starting from `Recipe()`, so existing trial folder names don't
    change.
  - Say so in the CLI help.
  - Test: `"stage"` parses to `DEFAULT_RECIPE`; `"stage iterations=2000"` changes only the iterations.

**Minor 1:** `train.py:27-28` claims float16 was "measured 22% faster per step". Measured in the training loop, it
gives 0–8%, within the noise; 22% held only for one fixed batch with nothing else running. Say that, briefly.

**Minor 2:** `README.md:698-700` gives a timing from before the speed work ("0.39 s a step … about 52 min"). Update it
to the shipped recipe: about 0.1 s a step, about 6–7 min of training. Use the README's own numbers.

**Minor 3:** the tolerance sentence (`README.md:627`) is ambiguous. Make it plain: "0.5 points on words found (the
floor; twice the measured spread was 0.12) and 0.11 points on long-line ink (twice the spread of 0.055)".

**Minor 4:** the wording at `README.md:707-708`; follow the review.

**Minor 6:** `textgrid.py:53, 65` imports `textnet.data` only for `load_page`, so an edit to the detector's training
code reruns the walls.
- Read the page without that import: move `load_page` to a small shared place, or read the page the way the other
  solver stages do. Keep the bytes read identical.
- Check the walls stage's code deps (the import closure) no longer include `textnet.data`, `train` or `model`. Adjust
  the deps test that asserts them, if any.

**Minor 8:** `tests/test_solver_textnet_precision.py` imports another test module. Move the shared fixtures or
helpers to `conftest.py` or a small helper module under `tests/`, as the repo does elsewhere, or inline them.

**Minor 10:** `experiments/line-masks` isn't covered by `runs.solve_dir`'s prefix refusal. Add it, so no solve run
can be named into the cache, with a test, following the existing refusal tests.

## Not now

The controller is running Important 1 (the both-styles check on the shipped recipe) on the GPU and will update the
"One detector for both styles" paragraph (`README.md:672-675`) itself. Leave that paragraph alone.

Parked, don't do: Minor 5, 7, 9, 11.

## Rules

- **Run nothing on the GPU,** not even `text-trial` or `solver text`. Tests run on the CPU only.
- **Stage only the files you change.**
  - Another agent (the code reader) commits on this branch: never stage, revert or stash `codereader/*`,
    `tests/test_codereader.py` or `dvc.lock`.
  - No bare `git stash`. Never commit `pipeline/data/`.
- **Full suite:** `cd pipeline && uv run pytest -q`. Failures only in `test_codereader.py` may belong to the other
  agent's work in progress: report them, don't fix them.
- **Commit message:** ends with:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`
- **Report:** write it to `.superpowers/sdd/2026-11-15-text-detector-training/final-fix-report.md`, listing each item
  and what you did. Reply with only: status, commit(s), a one-line test summary, concerns.
