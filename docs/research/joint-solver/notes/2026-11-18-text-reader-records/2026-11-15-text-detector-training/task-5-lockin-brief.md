# Task 5, Step 7: lock in the detector's recipe

The plan's last detector step (`docs/superpowers/plans/2026-11-15-text-detector-training.md`, Task 5 Step 7). The
measured loop is finished, and the owner chose the recipe:

**4,000 steps:** `train.Recipe(iterations=4000, pct_start=0.1, ink_only="ink+1px", half_size=True, prefetch=True)`.

It was chosen by the keep rule as applied. Both seeds of a candidate must pass the plan's tolerance against today's
recipe: words found at least 93.65%, long-line ink called text at most 4.21%. Each shorter or lighter candidate had
a seed that failed.

## Steps

1. **`stage.DEFAULT_RECIPE`** in `pipeline/src/sfzmaps/solver/textnet/stage.py`: set it to the recipe above, with a
   short comment in the file's style. Say what it is, how it was chosen (trials with the validation area held out;
   steps and scores in `pipeline/README.md`), and its validation scores against today's recipe.
2. **Re-record the guard** in `pipeline/tests/test_solver_textnet.py`: `GUARD_RECIPE` (the recipe's `name()`),
   `GUARD_WEIGHTS` (both digests) and `GUARD_LOSSES`, as the comment above them describes.
   - A previous attempt did the same for a 2,000-step recipe. Its diff is a worked example of what changes:
     `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t5/stale-lockin-2000.patch`.
     Its values are wrong for this recipe; don't copy them.
   - Record the new values by running what the test runs, on the CPU, the way the comment says (in a worker and not).
3. **Run the stage twice, in the foreground, one after the other:** `cd pipeline && uv run sfzmaps solver text`.
   - The chances must agree within 0.01; earlier stage runs gave identical chances and weights.
   - Record each run's minutes. The clock is a whole stage of at most about 10 min, and training of at most about
     5 min. Training is expected to take about 6.1 min idle, missing the training clock by about a minute. The owner
     accepted that; report the measured minutes.
4. **`uv run sfzmaps solver text-scores`.**
5. **The README** (`pipeline/README.md`, the text detector's section): rewrite the recipe bullet. "Its defaults are
   the quarter models' recipe…" stays; add what the stage's recipe is. Write the results table the plan asks for:
   - each step's minutes and validation scores;
   - the seed table: 2,000 / 3,000 / 4,000 steps × seeds 0–2, words found / long-line ink called text;
   - the test drawings' scores: 1980 and 1972/1975, words found and long-line ink called text, for today's recipe,
     the chosen recipe with the validation area held out, and the stage's model;
   - today's quarter models, for context;
   - the candidates measured and not kept, each with its numbers and one plain reason: today's recipe, half size,
     the loss on ink, the schedule lengths and the higher learning rate, weight averaging, the text output alone,
     batches drawn ahead, bigger batches (24 and 48 crops), the lighter decoder, 256-px crops, half precision
     (float16).
   - One short paragraph on speed. The step is bound by the GPU's memory traffic, in proportion to the px trained on,
     not by drawing the crops or the batch size. A memory-heavy job on the CPU slows it, since the Mac's CPU and GPU
     share memory.

   Write for a reader of the repo. No scratch paths, no session story, no "R&D". Short sentences, as the rest of the
   README.
6. **Full suite:** `cd pipeline && uv run pytest -q`. Commit.

## Where the numbers are

- **The per-trial table:** `.superpowers/sdd/2026-11-15-text-detector-training/task-5-results.md`, rows 1–23.
- **The ledger:** `.superpowers/sdd/2026-11-15-text-detector-training/progress.md`. Its last entries hold the seed-2,
  batch, lighter-decoder, 256-px and half-precision numbers and timings.
- **Each trial's own scores:** under `pipeline/data/solver/experiments/text-*/`. Folder names are the recipe names,
  with a seed suffix for seeds other than 0.
- **The logs:** `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t5/*.log`
  (`seed2.log`, `batch.log`, `light_crop_seed0.log`, `light_crop_seed1.log`, `bound_timing.log`, `scaler_timing.log`).
  Read them for numbers. Never cite them in the README.

Training minutes from the later trials were measured while a heavy CPU job ran (the 4,000-step recipe took 7.8 min
then, 6.1 idle). Compare candidates within the same conditions and say which conditions a number comes from.

## Rules

- **The GPU runs one job at a time.** Run each GPU command in the foreground and wait for it. Never two at once,
  never in the background. Tests run on the CPU (the conftest guard keeps them off the GPU).
- **Stage only** `stage.py`, `test_solver_textnet.py` and `README.md`. Never stage, revert or stash `dvc.lock`,
  `test_codereader.py` or anything under `codereader/`. No bare `git stash`. Never commit `pipeline/data/`.
- **Commit message:** ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`
- **Report:** write the full report to `.superpowers/sdd/2026-11-15-text-detector-training/task-5-lockin-report.md`.
  Include the two runs' minutes and chance comparison, the text-scores numbers and the test output. Reply with only:
  status, commit, a one-line test summary, concerns.
