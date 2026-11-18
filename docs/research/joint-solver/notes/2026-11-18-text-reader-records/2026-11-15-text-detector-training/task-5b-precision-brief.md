# Task 5b: a half-precision option for the detector's recipe

Part of the text detector's speed work (spec `docs/superpowers/specs/2026-11-15-text-detector-training-design.md`,
plan Task 5). Measured on the owner's M1 Max, one training step of 12 half-size crops:
- float32: 136 ms;
- float16 autocast: 106 ms (−22%);
- bfloat16: 120 ms.

Five float16 steps from the same seed gave the same weights twice. The spec had excluded half precision on a review's
word, never measured; the measurement overrides it. The controller runs the quality trials. You add the option.

## What to build

In `pipeline/src/sfzmaps/solver/textnet/train.py`:

- **`Recipe.precision: str = "float32"`,** one of `"float32"` or `"float16"`. Add it to `CHOICES` so the existing
  validation and `Recipe.name()` handle it: it appears in the name only when not the default. Document it in the
  Recipe docstring beside the other options, in the file's style.
- **`fit`:** with `"float16"`, the network's forward runs under `torch.autocast(device_type, dtype=torch.float16)`.
  - The loss stays float32: cast the output with `.float()` before `loss_fn`, outside the autocast.
  - Scale the loss against float16 gradient underflow with `torch.amp.GradScaler(device)` if torch 2.15 supports it
    for that device type: scale, backward, step, update.
  - Where it isn't supported, train without a scaler and say so in the docstring and the report. Check the real
    behaviour; don't assume.
  - The scaler must keep the run deterministic. Don't use a dynamic check that reads the device each step: a read
    waits for the GPU. GradScaler's own inf check is fine, since it doesn't sync on mps until `update`; but report if
    `update` forces a wait each step, and measure the step time with and without the scaler on the CPU.
  - A non-finite loss is still caught at the log, as today.
- **The module docstring:** one sentence on the option.
- **`predict` and `save_model`:** prediction stays float32. The weights are float32 either way; say so in one comment
  where predict builds the network. `save_model` keeps the recipe as it does, `precision` included, and
  `predict.load_model` must still load such a file.

## Tests

In a NEW file `pipeline/tests/test_solver_textnet_precision.py`. Don't edit `test_solver_textnet.py`: it holds
another agent's uncommitted work. Reuse that file's fixtures by importing them, or build small synthetic sheets the
way its tests do.

- `test_precision_is_a_recipe_choice`:
  - `Recipe(precision="float16").name()` contains `precision=float16`;
  - the default's name doesn't contain `precision`;
  - `Recipe(precision="half")` raises ValueError.
- `test_float16_fit_is_repeatable`: two `fit` runs of a tiny float16 recipe on the CPU, same seed, give identical
  weights. If CPU autocast doesn't support float16 for these ops in torch 2.15, test what does run on the CPU and say
  so in the report. Never run tests on the GPU; the conftest guard keeps them off it.
- `test_saved_float16_model_loads`: `save_model` then `predict.load_model` round-trips a float16 recipe.

## Rules

- **The GPU is busy with the controller's trials.** Run nothing on the GPU: no `text-trial`, no `solver text`.
  Tests run on the CPU (`cd pipeline && uv run pytest -q`).
- **Stage only your own files:** `train.py`, the predict file if you touch it, and the new test file.
  - The working tree holds other uncommitted changes that are not yours: `stage.py`, `README.md`,
    `test_solver_textnet.py`, `dvc.lock`, `test_codereader.py`, `codereader/glyphs.py`. Never stage, revert or
    stash them.
  - No bare `git stash`.
- **The full suite must pass,** except failures in `test_codereader.py`, which another agent is mid-way through. Report
  those; don't fix them.
- **Commit message:** ends with:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`
- **Report:** write the full report to `.superpowers/sdd/2026-11-15-text-detector-training/task-5b-precision-report.md`
  (what changed, the GradScaler finding, the test output). Reply with only: status, commit, a one-line test summary,
  concerns.
