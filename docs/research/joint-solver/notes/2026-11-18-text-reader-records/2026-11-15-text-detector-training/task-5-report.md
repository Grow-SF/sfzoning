# Task 5 report: the measured loop and the chosen recipe

(in progress; the per-trial table is task-5-results.md)

## Small fixes carried from the reviews (commit d8044f9)

- `runs.solve_dir` refuses `text-*` names (`runs.TEXT_PREFIX`, which `trial.PREFIX` now is): a solve built afresh
  would delete a trial's folder. Test: `test_a_solve_never_takes_the_folder_of_a_text_detector_trial`.
- The default-recipe guard also holds `stage.DEFAULT_RECIPE.name()` (`GUARD_RECIPE = "default"` until Step 7).
- `sfzmaps.transform` is out of `solver-text`'s code deps: the stage's cut (a lazy import would not do it:
  `fresh._imports` counts function-level imports too). `solver-text-scores` keeps it (it reads placements).
  dvc.yaml regenerated (one line gone); the deps tests assert both.
- Stale docs: scores.py's `evaluate.py` (now "their evaluate.py, since deleted"); textnet/__init__ (one line, as
  `test_the_solver_packages_hold_no_module_list` requires); `compare_outputs.LOOSE_JSON` = {"solver-text-scores"}
  (+ a test that a JSON in solver-text is held byte for byte).
- text-trial: `--train-series` (comma list of training maps, normalised to the training maps' order so one set trains
  one way; folder `-train=<maps>`; a test drawing refused), `--test-series` (also scores every px of those test
  drawings into `test_scores.json`, for the mixed-style check: the validation area can't compare a modern-only model
  with the both-styles one); recipes that are one (prefetch aside) are a usage error (exit 2) before anything runs;
  `minutes.json` is keyed by run (`plain` / `prefetch`), a flat file read as the plain run's.

## Controller's speed-ups (commit ca8984f)

- prefetch=True on every trial from #5 on. Measured alone (300 steps, mps): no gain (28.5 s without, 28.1-28.8 s
  with): the GPU's work dominates a half-size step (~0.09 s), the draw is ~0.01 s.
- Non-blocking copies (`train.Copies`): adopted. ABBA-alternated 300-step mps runs (4 + 4, plus an earlier 2 + 2):
  non-blocking 27.4-28.4 s, blocking 28.1-28.8 s, about 2% faster; every run's weights the same digest.
- Long-line masks kept on disk for the trials (data/solver/experiments/line-masks, 192 MB for the 59 validation
  sheets), labelled with the sheet's key and a digest of the code and libraries; the scores stage doesn't read them.
- Check on the full run (#5): the current best rerun with prefetch + non-blocking copies gave bit-identical weights
  to the plain run (#4) and byte-identical scores.json. That is also seed 0's two-run determinism check.

## Decisions in order

1. Baseline (#1): today's recipe on the six maps, validation held out: words 94.15%, long-line ink 4.10%, AUC
   0.9911, train 45.7 min (0.34 s/step).
2. Ruling (mine): the ink-only variants were measured on top of half_size (the current best once it passed),
   not at full size: each full-size run costs 46 min, and every later step builds on half size anyway; the keep
   rule is "against the current best".
3. half_size (#2): words 94.80 (+0.65), long-line 4.15 (+0.05): kept (provisional until the tolerance); train
   12.5 min (3.6x).
4. half_size + ink_only=ink (#3): 94.91 / 4.06, every measure better than #2: kept.
5. half_size + ink_only=ink+1px (#4): 95.01 / 4.05, every measure better than #3 (tiny): the current best.
6. Tolerance (#6, seed 1 of the current best, against #5, its seed 0): words 94.89 vs 95.01 (spread 0.12 pt),
   long-line 4.00 vs 4.05 (spread 0.055 pt). Tolerance: words max(2 x 0.12, 0.5) = 0.5 pt; long-line ink
   2 x 0.055 = 0.11 pt (the spec's floor is on words only). Re-judged Step 2: every keep stands (half_size's
   long-line +0.05 pt is within 0.11). Keep rule from here: words >= 95.01 - 0.5 and long-line <= 4.05 + 0.11,
   against the current best's seed 0.
7. Schedule (#7-#12, on half_size + ink+1px, pct_start 0.1), words / long-line vs the bar (>= 94.51, <= 4.16):
   2000 @ 2e-3 94.97 / 4.12 pass (3.1 min); 2000 @ 6e-3 93.76 / 3.90 fail (words); 4000 @ 2e-3 94.92 / 4.15 pass;
   4000 @ 6e-3 95.49 / 4.36 fail (long-line); 8000 @ 2e-3 94.95 / 4.08 pass; 8000 @ 6e-3 94.82 / 4.05 pass.
   Chosen: 2000 @ 2e-3, the shortest that passes. Note: the measures outside the keep rule are worse at 2000 steps
   (AUC 0.9881 vs 0.9923, ink outside boxes called text 2.94% vs 2.33%, log loss 0.110 vs 0.090): the detector is
   less sure, though it finds the words and calls the lines the same.
8. weight_average 0.999 on the 2000-step schedule (#13): words 91.63, fails (an average with a ~1000-step horizon
   over a 2000-step run keeps half-trained weights).
9. heads="text" (#14): words 94.03 (-0.94 pt vs #7), fails the words bar, though every other measure is better
   (long-line 3.90, AUC 0.9900, outside 2.35%, log loss 0.099). With the weight average too (#15): 92.93, fails.
   heads stays "all".
10. prefetch=True: kept (the same weights bit for bit, #5); no measurable speed gain alone.
11. decoder="light": not tried; the clock is met without it (training 3.1 min; the whole stage in Step 7).
12. Chosen recipe: Recipe(iterations=2000, pct_start=0.1, ink_only="ink+1px", half_size=True, prefetch=True),
    name half_size-ink_only=ink+1px-iterations=2000-pct_start=0.1: words 94.97%, long-line ink 4.12%, AUC 0.9881,
    train 3.1 min (baseline 94.15%, 4.10%, 0.9911, 45.7 min).

## The mixed-style check (Step 6)

The chosen recipe trained on the modern-lettering maps only (2000-digital-zoning, 1980-1990-zoning, 1990-1996-zoning;
38 sheets) against the same recipe on both styles (87 sheets), both scored on every px of the 1980 test drawing (32
sheets, modern lettering, never trained on): both-styles words 92.72% vs 92.05% (+0.67 pt), long-line ink 4.73% vs
4.78%, AUC 0.9891 vs 0.9854, ink outside boxes called text 2.28% vs 3.10%. Training on both styles doesn't cost the
modern drawing anything (it is better on every measure but block numbers, 96.0% vs 97.9%), so Step 7 goes ahead.

## Lock-in (Step 7)

- `stage.DEFAULT_RECIPE = train.Recipe(iterations=2000, pct_start=0.1, ink_only="ink+1px", half_size=True,
  prefetch=True)`; the guard re-recorded: GUARD_RECIPE "half_size-ink_only=ink+1px-iterations=2000-pct_start=0.1",
  its 3-step losses and the weight digests in both Mac thread setups (in a pytest-xdist worker and not).
- The clock: `uv run sfzmaps solver text` twice: 215 s and 219 s in all (load 15 / 18 s, train 190 / 191 s, predict
  the 13 2000 sheets 9 / 8 s). Training loop 3.2 min (<= 5), whole stage 3.6 min (<= 10).
- Determinism: the second run's chances equal the first's on every px of the 13 sheets (largest difference 0 counts
  of 255; the bar was 2) and the weights are the same bits.
- `uv run sfzmaps solver text-scores` (the stage's model, trained on every px of the six maps, on every px of the
  test drawings; 500 s: load 25, predict 67, score 406):

  | test map | AUC | words found | zoning code | block number | street name | other | text ink found | ink outside boxes called text | long-line ink called text |
  |---|---|---|---|---|---|---|---|---|---|
  | 1980 (32 sheets, modern) | 0.9886 | 91.7% (18,237/19,896) | 95.5% | 94.5% | 88.3% | 99.0% | 91.7% | 2.15% | 4.72% |
  | 1972 (30, older) | 0.9882 | 93.1% (13,505/14,512) | 97.0% | 2/2 | 91.6% | 99.1% | 93.3% | 2.67% | 3.36% |
  | 1975 (30, older) | 0.9892 | 92.7% (13,470/14,527) | 96.7% | 2/2 | 91.3% | 98.2% | 93.6% | 2.29% | 3.22% |
  | all | 0.9886 | 92.4% (45,212/48,935) | 96.3% | 94.5% | 90.4% | 98.8% | 92.9% | 2.40% | 3.64% |
- Like-for-like on the test drawings (both trial models, validation held out): today's recipe 92.65% words /
  3.72% long-line / AUC 0.9911; the chosen 2000-step recipe 93.35% / 3.67% / 0.9892 (results #18-19).

## STOPPED BEFORE LOCK-IN: the 2000-step pick is within its own noise

- Seed 1 of the chosen 2000-step recipe (#20): validation words 94.65, long-line 4.35, which fails the bar (4.16);
  seed mean 94.81 / 4.23. The tolerance (Step 3) was measured on the 8000-step recipe (long-line spread 0.055 pt);
  the 2000-step recipe's own spread is 0.23 pt, so its seed-0 pass is not evidence it keeps quality.
- My ruling: a recipe is locked in only if the mean of two seeds passes. 3000 steps seed 0 (#21): 96.45 / 4.36
  (fails long-line). Seed 1 of 3000 and of 4000 steps were still running when I had to hand back
  (data/solver/experiments/text-...-iterations=3000-...-seed=1 and -iterations=4000-...-seed=1; log
  scratchpad/t5/logs/16-seed1-3000-4000.log).
- Pattern: the shorter schedules trade long-line ink for words (the 0.5 threshold drifts): 8000 steps is the only
  length whose two seeds both pass (95.01/4.05, 94.89/4.00), at 12.3-12.8 min of training (clock: 5).
- Uncommitted in the working tree (NOT committed, on purpose): stage.DEFAULT_RECIPE = the 2000-step recipe, the
  guard re-recorded for it, the README recipe bullet. data/solver/text and data/solver/text-scores hold that
  recipe's stage outputs (run twice: identical; 3.6 min).
- Next, for the controller / owner: decide between (a) 4000 steps if its seed mean passes (6.2 min, over the 5 min
  clock; then decoder="light" to bring it under), (b) 8000 steps half size (12.5 min, clock missed), or (c) accept
  the 2000-step recipe with the measured noise (its test-drawing scores match today's: 93.35 / 3.67 vs 92.65 / 3.72;
  a third seed would firm this up). The quality bar was not loosened.
