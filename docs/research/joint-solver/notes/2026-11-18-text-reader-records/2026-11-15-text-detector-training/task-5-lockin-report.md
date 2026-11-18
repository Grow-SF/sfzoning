# Task 5 Step 7: lock-in report

**Status:** DONE_WITH_CONCERNS. **Commit:** `ad9db9f` (base `fda4c42`). It holds only `stage.py`,
`test_solver_textnet.py` and `README.md`. `dvc.lock` is still modified and unstaged, as it was before I started.

## 1. `stage.DEFAULT_RECIPE`

`train.Recipe(iterations=4000, pct_start=0.1, ink_only="ink+1px", half_size=True, prefetch=True)`, precision float32
(the default). `name()` is `half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1`. The comment gives what the recipe
is, says it was chosen by trials with the validation area held out (steps and scores in the README), and gives its
validation scores: three seeds, 94.03–94.92% words found and 4.12–4.15% long-line ink called text, against today's
94.15% / 4.10%. It also gives the training minutes: 6.2 for its trial, 45.7 for today's.

## 2. The guard (re-recorded on the CPU, in a worker and not)

I recorded the values by running the test itself. A temporary line, now removed, wrote the digest and losses: once
with `-n 2` (a worker, OMP/VECLIB = 1) and once with `-n 0` (not a worker, both None).

- `GUARD_RECIPE = "half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1"`
- `GUARD_WEIGHTS`: `("Darwin","arm64","1","1")` → `4cae67dd…4266`; `(…, None, None)` → `226bca5d…165b`
- `GUARD_LOSSES`: from the run outside a worker. The worker's losses differ in the 7th significant digit, well within
  the 1e-5 tolerance.
- The digests and losses are the same as the stale 2,000-step patch's. That is expected: the test replaces
  `iterations` with 3, which makes the 2,000- and 4,000-step recipes the same recipe, schedule included. Only
  `GUARD_RECIPE` tells them apart. So the weights guard can't see a change in step count; the name check is what does.
- The test passed both ways after the values went in.

## 3. The stage, twice (`uv run sfzmaps solver text`, one after the other)

| Run | Load | Train | Predict (13 sheets) | Whole stage (wall) |
|---|---|---|---|---|
| 1 | 15 s | 448 s (7.5 min) | 8 s | 473 s (7.9 min) |
| 2 | 15 s | 413 s (6.9 min) | 8 s | 438 s (7.3 min) |

- **Chances:** the largest difference is 0 counts of 255 on every one of the 13 sheets (0 px differ), so they agree
  within 0.01. The weights are bit-identical, and every 250-step loss line is the same in both logs.
- **Conditions:**
  - macOS `mediaanalysisd` used about 115–120% CPU through both runs.
  - In run 1 I also ran a stray `find /` (a disk-wide search for the old quarter scores) for about 2 min, around
    steps 500–1250, then stopped it. Steps there slowed from about 25 s to 27–33 s per 250. Run 2 had nothing of mine
    beside it.
  - The trial of the same recipe trained in 6.2 min.
- **Against the clock:** the whole stage is within about 10 min. Training (6.9 / 7.5 min) misses the target of about
  5 min, as accepted.

## 4. `uv run sfzmaps solver text-scores`

Wall 485 s: load 23 s, predict 66 s, score 394 s. The stage's model, every px of the test drawings:

| Map | AUC | Words found | Long-line ink called text | Ink outside boxes called text | Text ink found |
|---|---|---|---|---|---|
| 1980 | 0.9907 | 92.4% (18,375/19,896) | 4.83% | 2.13% | 92.8% |
| 1972 | 0.9904 | 94.1% (13,660/14,512) | 3.47% | 2.72% | 94.4% |
| 1975 | 0.9909 | 93.8% (13,632/14,527) | 3.30% | 2.32% | 94.8% |
| all | 0.9906 | 93.32% (45,667/48,935) | 3.74% | 2.43% | 94.0% |

1972 and 1975 pooled: 93.98% / 3.39%. The 2,000-step stage model it replaces scored 92.39% / 3.64% over all.

**One extra GPU run, not in the brief's steps.** The README needs the test scores of the chosen recipe with the
validation area held out, and no run had produced them for 4,000 steps. So I scored the existing seed-0 trial model:
`sfzmaps solver text-trial --recipe "half_size=True ink_only=ink+1px prefetch=True pct_start=0.1 iterations=4000"
--reuse --test-series 1980,1972,1975`. It trained nothing and took 153 s.

- Results: 1980 92.67% / 4.83%; 1972 94.2% / 3.46%; 1975 94.1% / 3.34%; pooled 1972/1975 94.16% / 3.40%; all 93.56% /
  3.75%, AUC 0.9899.
- Its validation `scores.json` came out byte-identical to before, and `minutes.json` kept its training minutes (6.197).
  The run added `test_scores.json` and a `test` minute.

## 5. README (`pipeline/README.md`, the text detector's section)

- **The recipe bullet:** "its defaults (`Recipe()`) are the quarter models' recipe…" stays. It now says the defaults
  rise over the first 30%, lists the batch and crop sizes and `precision="float16"` among the options, and names the
  stage's recipe (half size, `ink+1px`, 4,000 steps rising over 10%, batches drawn ahead, float32).
- **New paragraph "How the stage's recipe was chosen":**
  - the keep rule and the tolerance;
  - the steps table: minutes, words found, long-line ink called text, AUC and a verdict for each trial, rows 1–15;
  - the seed table: 2,000 / 3,000 / 4,000 × seeds 0–2. 3,000 seed 2 is "not run", since seeds 0 and 1 both fail;
  - the bar against today's recipe (93.65% / 4.21%);
  - the later candidates (batches of 24 and 48 crops, the lighter decoder, 256-px crops, float16), each with both seeds
    and a reason. The text says their minutes were measured beside a memory-heavy CPU job, when the 4,000-step recipe
    took 7.8–8.0 min (6.2 idle);
  - the mixed-style check;
  - the test-drawings table (today's recipe held out, the chosen recipe held out, the stage's model);
  - one line on the quarter models;
  - the speed paragraph, which closes with the stage's measured minutes and its bit-identical repeat.
- **How the brief's list of candidates is covered:** some items on it are in the chosen recipe: half size, the loss on
  the ink, and batches drawn ahead. Rather than call those "not kept", every candidate measured gets a row with its
  numbers and a one-line verdict.
- No scratch paths, logs or session story.

## 6. Tests

`cd pipeline && uv run pytest -q`: **1105 passed, 1 warning** (a shapely deprecation in `test_reviewpage`) in 60 s.
After a later comment-only edit to `stage.py`, I reran `tests/test_solver_textnet.py`: 43 passed.

## Concerns

1. **Training minutes.** Training took 6.9 min in the clean run (7.5 in the other) against the brief's expected 6.1 idle.
   Both runs had `mediaanalysisd` at about 115% CPU; run 1 also had my stray `find` beside it.
2. **The quarter models' line.** I found no scores for them except the spec's north-west figures (96.6% words; 0.16% of
   district-line ink). The old `evaluation_<Q>.json` files are gone. So the README cites only the north-west model and
   says the measure differs.
3. **The text output alone (row 14).** It was rejected against the best recipe at the time, #7 (94.97%), at 0.94 points
   fewer words. Its seed-0 score of 94.03% / 3.90% would pass the final bar against today's recipe (93.65% / 4.21%).
   It was never run on more seeds or at 4,000 steps. The README states its reason as measured, fewer words than with
   every output. Flagging it in case the owner wants that candidate run on the final rule.
4. **`dvc.lock`.** The `solver-text` and `solver-text-scores` outputs in `data/solver/` now come from the new recipe,
   but I didn't record them in `dvc.lock`, as the rules say. The controller's build or `dvc commit` records them.
5. Run 1's model and chances are kept for comparison in
   `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t5/lockin-run1/text`.
   The logs are in that `t5/logs/` folder: `17-lockin-stage-run1.log`, `18-lockin-stage-run2.log`,
   `19-lockin-text-scores.log`, `20-lockin-heldout-test.log` and `21-lockin-pytest.log`.
