# Task 5 report: control NE, lines-unknown, seeds 0-2

## Built
- `pipeline/scripts/textlines/variant.py`: `run(...)` (signature per the brief), CLI, `ExtraGeo`, default `score`.
  - `run` loads the training sheets, holds out the quarter plus the answered blocks, calls `zones.prepare` before the GPU turn (lines cache `experiments/textlines/lines/`), then trains and predicts inside one `gpu.gpu_turn`.
  - It predicts the quarter's px on the 2000 sheets (0 elsewhere) and on `--extra` sheets, then gates (CityGeo for 2000; `ExtraGeo` for extras: quarter by map point, block -1, weight 0), renders the panel and the audit, writes `run.json`.
  - The panel columns are this run, r328, and the old quarter model.
  - `--composites` and `bank` are refused (Task 6; `Recipe` has no such field).
- `pipeline/tests/test_textlines_variant.py`: 2-step CPU recipe, `lines-unknown`, stub `score=`, on the `maps` fixture, with an extra series. It checks the folder name, `model.pt`, `scores.json`, `run.json`, the chance shape, zeros outside the quarter, and the extra chance file.
- Commit 318ad8b.
- TDD evidence: the test was written alongside the implementation and passed on its first run (4.96 s); I did not capture a separate red run. Full `uv run pytest -q`: 1501 passed, 3 xfailed (before the runs; none ran during training).

## Runs (NE held out, DEFAULT_RECIPE + labels=lines-unknown, mps)
Minutes: load 0.6, train 6.3, predict 0.1-0.2, gate+panel+audit 1.1 (each run). `git_status_src` was empty.

Scores on NE of the 2000 sheets (share called text >= 0.5; 95% CI over blocks for s0):

| measure | s0 | s1 | s2 | r328 (NE blocks) | old-ne |
|---|---|---|---|---|---|
| m1a_in_box (lower better) | 0.3925 [0.358,0.428] | 0.4095 | 0.4105 | 0.9922 | 0.6294 |
| m1b_in_box (lower) | 0.5102 [0.472,0.538] | 0.5936 | 0.5320 | 0.9694 | 0.8177 |
| m1c (lower) | 0.0405 | 0.0616 | 0.0545 | 0.5474 | 0.0252 |
| m2 (higher) | 0.9653 [0.950,0.977] | 0.9756 | 0.9620 | 0.9994 | 0.9727 |
| m1a (all border ink) | 0.0389 | 0.0538 | 0.0506 | - | 0.038 |

## Paired (run minus baseline, 95% interval, NE common blocks)
Seed 0 alone:
- vs r328: m1a_in_box -0.600 [-0.634,-0.563]; m1b_in_box -0.459 [-0.515,-0.411]; m2 -0.034 [-0.049,-0.022]; m1c -0.507 [-0.641,-0.365].
- vs old-ne: m1a_in_box -0.237 [-0.271,-0.205]; m1b_in_box -0.308 [-0.323,-0.284]; m2 -0.007 [-0.027,+0.009] (no change); m1c +0.015 [+0.008,+0.021] (slightly worse).

Mean of 3 seeds (arm = mean over runs):
- vs r328: m1a_in_box -0.588 [-0.620,-0.554]; m1b_in_box -0.424 [-0.473,-0.383]; m2 -0.032 [-0.045,-0.021]; m1c -0.495 [-0.632,-0.350].
- vs old-ne: m1a_in_box -0.225 [-0.257,-0.194]; m1b_in_box -0.272 [-0.291,-0.245]; m2 -0.005 [-0.023,+0.010]; m1c +0.027 [+0.017,+0.037].

## Stop rule
Passed: m1a_in_box and m1b_in_box are lower than both baselines, intervals exclude zero. s1 and s2 were run.

## Noise (largest pairwise difference over 3 seeds)
m1a_in_box 0.018; m1b_in_box 0.083; m1c 0.021; m2 0.014; m1a 0.015.
The m1b_in_box noise (0.083) is large against its gain over old-ne (0.27) but small against r328's. The m1c loss against old-ne (+0.027) and the m2 loss are within about one noise width: not clearly real. The seeds differ on every measure.

## 1990-96 (NE px of sheets 22-33; no table, so no M1a)
| | m1b_in_box | m1c | m2 |
|---|---|---|---|
| s0 | 0.5253 | 0.0442 | 0.9687 |
| s1 | 0.6044 | 0.0680 | 0.9773 |
| s2 | 0.5432 | 0.0611 | 0.9632 |

These track the 2000 NE numbers closely. Px: m1b_in_box 289,966, m1c 4.85M, m2 1.05M. The intervals printed in the extra scores.json are degenerate (one block, -1), so no interval is reported for these. No baselines exist for them.

## Panels (control-ne-s0/panel.png; this run is NE-only, so the two p12 SW rows are blank for it)
- Lines through labels and street/district lines: GREY in every NE window (rm2a/b/c, downtown, residential, hatched). r328 colours them heavily red. The old quarter model is mostly grey but reddens some.
- Letters: RED, mostly. RH-2, UNION NC, RM-2 (rm2b), RC-3, block numbers, SLI, 3-S, M- are red.
- Misses: the large vertical "RM-2" in rm2a is grey, a miss; some small lot numbers (e.g. "549", "523") are partly grey. In the hatched window the street-name letters "TANGANG SORA" are grey too.
- Overcalls: the dashed pier-head line (dashed window) is red, and some hatching and thin lines at the small-lot clusters (downtown, hatched windows) are red.
- Audit: `audit.png` and `audit.json` are written per run (not examined in detail).

## Concerns
- m1b_in_box is still about 0.5-0.6: line ink in words is called text half the time. The rule leaves it unknown in training, so the net has no push against it.
- The seed spread on m1b_in_box is large. Verdicts between variants need a gap larger than about 0.08, or multi-seed arms.
- The missed big letters (RM-2 in rm2a) hint that m2 loses on large glyphs.
- Predict time: only the quarter's tiles are scored.
- `paired` against r328 uses NE blocks only; r328's headline (all quarters) differs from its NE value.
- Outputs are in `pipeline/data/solver/experiments/textlines/runs/control-ne-s{0,1,2}/` (not committed).
