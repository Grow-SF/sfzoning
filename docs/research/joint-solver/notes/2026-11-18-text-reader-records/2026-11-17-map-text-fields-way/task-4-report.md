# Task 4 report (INTERIM): reader experiments on NE, arms base and (a); arm (b) PENDING

Per the coordinator's change: arms base and (a) at seeds 0 and 1 on NE are done; arm (b) (mask) was NOT started (the GPU
goes to Task 8). No four-fold confirmation and no page rebuild, because nothing is kept yet.

## Built
- `pipeline/scripts/wordread/run_reader.py` (`run`, `decide`, `assemble`), `pipeline/tests/test_wordread_run_reader.py` (5 tests).
- `run(name, quarter, seed, own, sloppy, mask, boxes_dir)` writes `experiments/wordread/runs/<name>-<q>-s<seed>/`
  (labels.parquet, headline.json, box_numbers.json, read_numbers.json, run.json, reader.pt).
- LINE_TRIES: gone from readercrops.py before the first training.
- Check: base NE s0 reproduces the stage's NE reader exactly (loss 0.07645; 92/285, 0.3228 = baseline NE).

## Arms (no picked share)
- base: today's recipe.
- (a): every training Terra word of the 2000 maps cut with our boxes' measured offsets (`sloppy_crops`, 1 copy; its clean
  Terra crop leaves the real pool, since near-zero offsets in the draw stand for it: 5,417 rows replaced), plus every
  real own-box crop (2,959; kinds drawn by `kind_weights="terra"`). Each pool drawn by its count: share =
  (1 - synthetic share 0.4, today's recipe's own constant) x count / (real + own + sloppy). Real 14,630, own 2,959,
  sloppy 5,393 (s0) / 5,403 (s1) -> own 0.0773, sloppy 0.1408, real 0.3819, synthetic 0.4.
- The 0/180 classifier is the stage's NE classifier for every arm (same recipe, unchanged), so it adds no difference.
- Keep rule (written in the script before the runs): headline rise with paired block-bootstrap 95% interval (mean of the two
  seeds' per-label results) excluding zero AND each seed rising; reading at Terra's boxes not below base by more than
  base's two-seed spread; then the controller's look.

## Results, NE (285 labels; headline = right at our boxes, missed box wrong)
| arm | seed | headline | CI (blocks) | at Terra's boxes (263 labels) | loss | monitor CER |
|---|---|---|---|---|---|---|
| base | 0 | 92/285 = 0.3228 | [0.269, 0.377] | 0.9468 [0.916, 0.973] | 0.0765 | 0.1113 |
| base | 1 | 88/285 = 0.3088 | [0.257, 0.363] | 0.9620 [0.935, 0.981] | 0.0868 | 0.1169 |
| (a) | 0 | 96/285 = 0.3368 | [0.283, 0.394] | 0.9468 [0.920, 0.970] | 0.1085 | 0.1221 |
| (a) | 1 | 94/285 = 0.3298 | [0.274, 0.384] | 0.9658 [0.943, 0.985] | 0.1408 | 0.1209 |

Mean over seeds: base 0.3158, (a) 0.3333.

Paired (a) minus base (label bootstrap over whole blocks, 2000 draws):
- mean of the two seeds: +0.0175, 95% [-0.0121, +0.0477]
- seed 0: +0.0140 [-0.0209, +0.0495]; seed 1: +0.0211 [-0.0176, +0.0612]

Box numbers (IoU 0.5, NE) are identical in every run (the boxes are the same): 2,705 Terra words, 2,452 boxes, 850 matched,
1,855 missed. Read numbers (matched boxes 850): same_as_terra / differs: base s0 165/42, s1 159/48; (a) s0 165/42, s1 157/50.
Codes right vs answers at matched boxes (n 77-78): base 63, 58; (a) 60, 61 (Terra's string 77/78).
Reading at Terra's boxes: (a) minus base mean +0.0019, base seed spread 0.0152 -> within noise.

## Decision by the rule (interim)
(a): the headline rises at both seeds (+1.4, +2.1 points) but the paired interval includes zero (lo -0.012). Condition 1 FAILS:
(a) is NOT kept. Condition 2 passes. (b): pending, not run.

## Concerns
- The gain (+1.8 points, ~5 labels of 285) is inside the noise; the two seeds of base differ by 1.4 points themselves.
- The headline is limited by the boxes: only 278 of 285 NE labels have a box, and only 850 of 2,705 Terra words are matched at
  IoU 0.5, so most training-distribution change cannot help where the box is wrong. Most of the 41% gap is boxes, not reader.
- Choices I made (may need the owner's eye): in (a) the clean Terra crops of the 2000 words are replaced by their sloppy
  crops (not kept beside them); the synthetic share stays 0.4; the classifier is shared, not retrained per seed; the bank for
  line pieces is built from the 2000 sheets only; sloppy crops use one copy per word.
- Terra reading's 263 labels give an interval of about +-0.03, wider than the seed spread used as the "noise".
