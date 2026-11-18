# Task T3 report: the text reader as stages, in plain torch

Status: DONE_WITH_CONCERNS.

Commits (branch `solver`, local):
- 90b1069: code, tests, README entries, dvc.yaml;
- 11de9e5: dvc.lock for solver-text-crops;
- e48d3b7: dvc.lock for solver-text-train;
- 92c8215: dvc.lock for solver-text-read;
- c070d80: README with the measures.

No `sfzmaps build` is running now, and all three stages show up to date in `dvc status`.

Headline:
- **The reader is as good as the spike, held out in every quarter:** closed top-1 0.962 [0.947, 0.976] at Terra's
  boxes on 662 answered labels.
- **Our boxes lose half of it:** 0.464 [0.425, 0.503] on the 633 labels a box holds; 86 more labels have no box.
- **The loss is box error, not reader error.** On the same 595 labels: 0.965 at Terra's box, 0.477 at ours.

## What was implemented

Three stages instead of one stage plus its inputs stage: the training is its own stage, so that a change to the word
boxes (T2, or the pending detector retrain) rereads the boxes in two minutes and never retrains (about 85 min of GPU).
The work is the same; only the split differs.

| stage | command | code | writes |
|---|---|---|---|
| `solver-text-crops` | `sfzmaps solver text-crops [--workers N]` | `solver/textread/crops.py`, `synth.py`, `inputs.py` | `data/solver/text-crops/` |
| `solver-text-train` | `sfzmaps solver text-train [--device cpu\|mps]` | `solver/textread/model.py`, `train.py` | `data/solver/text-train/` |
| `solver-text-read` | `sfzmaps solver text-read [--workers N]` | `solver/textread/ctc.py`, `read.py` | `data/solver/text-read/` |

Path constants live in `solver/textread/store.py` (with `training_rows` and the flat crop store), so `paths.py` and
`runs.py` are untouched. The train stage's code closure is only `textread/{ctc,model,store,train}.py` plus
`paths.py`/`config.py` (a test pins this): nothing but the training code can trigger a retrain.

Files changed: `src/sfzmaps/solver/textread/` (new: `__init__`, `store`, `model`, `ctc`, `crops`, `synth`, `inputs`,
`train`, `read`), `src/sfzmaps/build.py` (three stages), `src/sfzmaps/cli.py` (three commands; re-records the six
warp stages' cli.py hash, outputs unchanged), `dvc.yaml`, `dvc.lock`, `README.md` (an entry per stage),
`tests/test_solver_textread.py` (new), `tests/test_build.py` (a deps test), `tests/test_solver_textnet.py` (the stage
list). No new dependency: no torchvision, no doctr.

### The model (`model.py`, ~150 lines)

docTR's `crnn_mobilenet_v3_small` re-expressed: torchvision's MobileNetV3-small feature layers written out
(`ConvNormAct`, `SqueezeExcite` with hard sigmoid, `InvertedResidual`, the 11-block table, the 576-channel last
layer, torchvision's init), the depthwise convolutions of layers 2, 4 and 9 at stride (2, 1); then
`nn.LSTM(576, 128, 2 layers, bidirectional)` and `Linear(256, 46)`. Parameter names are docTR's
(`feat_extractor.*`, `decoder.*`, `linear.*`), so `C.pt` loads with `strict=True`. The 0/180 classifier `Direction`
is S2's, names kept (`body.*`, `head.*`). The input is S2's (`padded`, `to_input`: docTR's mean 0.694 / sd 0.299 over
three channels).

### The code list, and where it comes from

The printed forms are `solver-text-crops`' `prior.json` `forms` (95 forms over 57 codes):
`codereader.posterior.printed_forms(codereader.labels.code_list(training labels))`, made inside
`Prior.learn`: the April 2000 table's code NAMES (`lots.npz` `table` where `known`; the list, never a code at a
label's spot), `labels.ALIAS` and `codes.ALIAS` spellings, " NCD" after each named district's forms, and every
printed form the training labels' saved reads wrote that names a table code. It is identical to S2's `prior3.json`
`forms` (checked: equal dicts).

### The output format (for T4)

`data/solver/text-read/reads.npz`, one entry per row of `data/solver/boxes/boxes.parquet`, in its order:

| array | shape | meaning |
|---|---|---|
| `row` | (n,) | the box's row in boxes.parquet |
| `forms`, `form_code`, `codes` | (F,), (F,), (K,) | the printed forms (sorted), each form's code, the codes (sorted); F = 95, K = 57 |
| `log_forms` | (n, 2, F) | log p(form \| crop) read at 0 and at 180 degrees (exact CTC forward) |
| `log_none` | (n, 2) | log p(none \| crop) at 0 and 180: the empty string, the all-blank path (S2's definition) |
| `p_other` | (n, 2) | p(other \| crop) = 1 − Σ forms − none, at 0 and 180 (floored at 0) |
| `p_classifier` | (n,) | the quarter's 0/180 classifier's p(the 0-degree crop is upside down), balanced classes |
| `prior_turn` | (n,) | the direction prior at the box's axis: p(it reads at 180) |
| `p_turn` | (n,) | q = p(180°), the classifier's odds on that prior |
| `post` | (n, F + 2) | the mixture (1 − q)·read(0) + q·read(180): each form, then "other", then "none"; sums to 1 |
| `log_closed` | (n, K) | log p(code \| box, a code): the mixture's forms summed by code, renormalised (log space) |

"0 degrees" is the box's axis angle as T2 gives it (counter-clockwise from left-to-right, modulo 180): the crop
reads along `labels.frame(…, axis)`; 180 is that crop turned a half turn. `reads.parquet` has one summary row per box
(`row, sheet, i, quarter, axis, best_0, best_180, p_turn, p_codes, p_other, p_none, top, p_top, closed_top,
closed_p`). `answered.parquet` has every answered label's readings at Terra's box and at our box (`box_row`, −1 when
no box holds its point).

### The direction

Each box is read both ways; the posterior is the mixture with q = p(180°), never the max. q is the quarter's 0/180
classifier's balanced-class probability taken as a likelihood ratio on a measured prior: by axis, the share of that
quarter's training words (every kind, Terra's reading angle) that read the axis's half turn
(`ctc.axis_prior`: a Gaussian kernel of 5° in the axis, not wrapped at 0/180 because the two ends of the axis read
opposite ways, Jeffreys' half count of each direction; table every 0.1°, in `axis_prior.npz`). Without this prior
the axis's two ends would get 50/50, and S2 found the classifier alone flips about 3% of upright crops. The raw
classifier output and the prior are stored, so T4 can recombine.

## How the synthetic inputs are now made

Found in the scratchpad recon (`codereader/task2/real.py`, `codereader/task3/build_prior2.py`, the code-reader plan):
the font was `glyphs.learn_font(labels.training(labels), texts=consensus, observed=~line_mask(rules=glyphs.RULES),
charset=code list's characters)` and the prior `posterior.Prior.learn(training labels, font, font.noise)`;
`labs.pkl` was `labels.load_labels()` pickled. `solver-text-crops` now runs exactly these on today's labels
(`textread/synth.learn_template`), from the prints, the code-check answers (to hold their blocks out) and the pages,
on 8 threads with the maths libraries pinned to one thread (`_in_one_thread`):

- **the font**: identical to `font_real.npz`, array for array (round log-likelihoods −9,235,416 … −9,137,146 as in
  the original run; 959 training labels; 711 s);
- **the prior**: identical to `prior3.json` (forms, p(code), p(form | code), the pose prior; 91 s). Its code prior
  counts the training labels' saved reads (`glyphs.consensus`), never the table's code at a label's spot;
- **`labs.pkl`** is no longer needed: the answered labels come from `labels.load_labels(crops=False)` (S2 checked the
  two agree).

One font and one prior for the map (ruling P4). The synthetic crops are drawn per quarter (`synth_<q>`, 50,000 each,
seed `[20261117, q]`): S2's recipe, with the code heights, the line share and the line windows taken from that
quarter's reader's own training crops, so no reader's training set holds a px of its own quarter. S2's known faults
are kept so the recipe is C's (box offsets drawn with the opposite sign; line windows centred on ink).

The real crops are S2's recipe (`crops.py`) on the three scans, with five flags per crop (`touch_nw … touch_se`,
`touch_blocks`) instead of one held quarter; the validation area is not held out (as in `solver-text-reader`). A
non-text window touching an answered block is not kept. 34,937 crops (13,726 block numbers, 7,884 street names,
6,064 codes, 1,486 other, 5,777 non-text); training crops per reader nw 21,892, ne 20,047, sw 20,392, se 17,868
(S2's se-held-out C had 17,584). 731 answered labels, 668 with Terra's box.

## Regression check

`model.load("…/spike-s2/models/C.pt")` into the plain-torch `Reader` gives S2's saved CPU logits
(`frames_C.npz`, made with docTR's model) **bit for bit** (max |Δ| = 0.0 on the 7 crops checked by hand, from 7 to 41
frames); `D.pt` into `Direction` gives `direction.npz`'s p(upside down) exactly. The tests
`test_the_plain_torch_reader_gives_spike_s2s_reader_c_its_outputs` (12 crops, atol 1e-4) and
`test_the_plain_torch_classifier_gives_spike_s2s_classifier_d_its_outputs` (12 crops, 1e-6) skip when the spike's
files are absent.

## Tests

`tests/test_solver_textread.py`, 15 tests, all passing:

- **The CTC sums:**
  - the exact CTC forward over printed forms equals a brute-force sum over every frame path (5 frames, alphabet
    "RH1" + blank, 4^5 paths);
  - `read` splits the mass into the forms, "none" (the empty string) and "other" (the brute-force rest);
  - a form too long for its frames has no mass.
- **The direction:**
  - the mixture is (1 − q)·read(0) + q·read(180), sums to 1, and is neither read alone;
  - the closed posterior sums a code's forms and renormalises;
  - the classifier's odds on the prior (three exact cases);
  - the axis prior reads each end of the axis its own way (no wrap at 0/180) and the axis/turn split of an angle.
- **The held-out split:**
  - on synthetic rows: `training_rows(q)` keeps exactly the rows touching neither q nor an answered block, with a
    target; `text_only` drops the windows;
  - on the built crops, by a second route: no training crop's centre map point is in its reader's quarter, none
    sits on an answered block's lot, and no 2000 training word is an answered label's Terra box.
- **The crop:** a word cut at axis 90 on a page turned a quarter turn equals the upright word cut at axis 0, and
  axis 180 is its half turn.
- **Matching:** the box holding a label's point is the nearest of those that hold it; −1 when none does.
- **Scoring:** a "no code" answer is right on none or other, and the closed and open ranks.
- **The weight-loading regression:** C.pt and D.pt give S2's saved outputs (these skip without the spike's files),
  and the reader turns a 50 px crop into 13 frames of 46 classes.
- **Elsewhere:** `tests/test_build.py::test_text_reader_stages_deps` pins the three stages' commands, upstreams,
  outputs and code closures (the train stage's is only `textread/{ctc,model,store,train}`), and its reads (the crops
  folder alone).

Full suite: 1460 passed, 3 xfailed (`cd pipeline && uv run pytest -n 4`).

## Training (solver-text-train, mps, one exclusive GPU turn per training, seed 0 for all eight)

| quarter held out | reader: real + synthetic crops | final loss | held-out quarter CER (logged only) | seconds | classifier: crops | final loss | seconds |
|---|---|---|---|---|---|---|---|
| nw | 21,892 + 50,000 | 0.137 | 0.132 | 961 | 17,207 | 0.094 | 178 |
| ne | 20,047 + 50,000 | 0.076 | 0.111 | 966 | 15,533 | 0.053 | 186 |
| sw | 20,392 + 50,000 | 0.095 | 0.116 | 964 | 15,956 | 0.134 | 180 |
| se | 17,868 + 50,000 | 0.071 | 0.124 | 974 | 14,209 | 0.109 | 181 |

- Readers: 12,000 steps each; classifiers: 3,000 steps each.
- Total 76.6 min of GPU, against the plan's 90.
- The CER is the best path's character error on 600 words of the held-out quarter on the 2000 scan, every kind;
  nothing is chosen by it.
- For comparison, spike C's final loss was 0.097 over 1,278 s.
- Reproducibility: seeds are fixed; S2 found its GPU loop identical step for step (300 steps); this run did not
  repeat a training.

## Measures

The owner's answered labels: 731 in all.
- 719 are scored: a code in the list, or "no code" (20).
- 7 have no answer.
- 5 answers are outside the code list: 3RD, CMNC, MB-OS, NC, NOE VALLEY.

Each label is read by the reader of its point's quarter. Closed means p(code | a code), the codes renormalised, T = 1
(no temperature fitted; T4 calibrates); 95% label-bootstrap intervals.

**At Terra's boxes (S2's crop; reader error alone), 662 labels:**

| | n | closed top-1 | closed top-3 | closed −log p | open top-1 | open −log p |
|---|---|---|---|---|---|---|
| all | 662 | **0.962** [0.947, 0.976] | 0.992 | 0.142 [0.091, 0.202] | 0.934 | 0.300 |
| nw | 184 | 0.984 [0.962, 1.0] | 0.995 | 0.052 | 0.973 | 0.136 |
| ne | 263 | 0.947 [0.916, 0.973] | 0.996 | 0.193 | 0.905 | 0.455 |
| sw | 89 | 0.978 [0.944, 1.0] | 1.0 | 0.064 | 0.955 | 0.106 |
| se | 126 | 0.952 [0.913, 0.984] | 0.976 | 0.221 | 0.921 | 0.353 |
| unflagged | 560 | 0.980 [0.968, 0.991] | 0.996 | 0.080 | 0.959 | 0.211 |

- With Terra's direction (S2's setting) and with the axis alone (ours: the classifier on the measured axis prior),
  every number is the same to the third decimal.
- **Against:**
  - spike C, se held out only: 0.956–0.973 over three seeds, −log p 0.102–0.126;
  - the LLM chain: 96.9%.
- Our readers are inside the spike's range overall, held out in all four quarters. ne, the densest quarter, is the
  lowest.

**At our boxes (the box holding the label's point), 719 labels:**

| | labels | box found | box missed | closed top-1 on found | closed top-3 | closed −log p | open top-1 | every label, closed top-1 (missed = wrong) |
|---|---|---|---|---|---|---|---|---|
| all | 719 | 633 | 86 | **0.464** [0.425, 0.503] | 0.540 | 5.35 | 0.283 | 0.412 [0.376, 0.448] |
| nw | 191 | 163 | 28 | 0.537 | 0.624 | 4.44 | 0.399 | 0.458 |
| ne | 295 | 286 | 9 | 0.335 | 0.396 | 6.92 | 0.192 | 0.326 |
| sw | 96 | 75 | 21 | 0.627 | 0.720 | 3.79 | 0.280 | 0.500 |
| se | 137 | 109 | 28 | 0.576 | 0.660 | 3.74 | 0.349 | 0.469 |
| unflagged | 591 | 523 | 68 | 0.520 [0.476, 0.563] | 0.594 | 4.64 | 0.306 | 0.464 |

**Box error against reader error, on the same 595 labels (a Terra box and one of ours):**
- closed top-1 0.965 at Terra's box against 0.477 at ours (difference −0.49 [−0.53, −0.45]).
- By what our box is like against Terra's word (a diagnostic, scratch `box_errors.py`):

| our box | n | at Terra's box | at ours |
|---|---|---|---|
| axis off by more than 10° | 272 | 0.945 | 0.371 |
| axis within 10°, much larger (merged with linework) | 108 | 0.991 | 0.185 |
| axis within 10°, much shorter (split) | 41 | 0.976 | 0.366 |
| axis within 10°, size near Terra's word | 174 | 0.977 | 0.851 |

- 95 of the 105 one- and two-letter codes (mostly "P") get an axis more than 10° off.
- Re-reading our boxes' extent with the axis snapped to the nearest quarter turn (0.445) or set to Terra's (0.466)
  does not recover anything (0.477 as built; scratch `axis_diag.py`). The axis error is a symptom of components
  merged with linework, not a separate cause.

**Explaining away (each reader on its own quarter's words at Terra's boxes, 2000 scan, Terra's direction):**
- p(other) is 0.996-0.999 on block numbers and 0.973-0.997 on street names.
- p(none) is 0.975-0.990 on non-text windows.
- On code words, the code list's mass is 0.91-0.94, and closed top-1 against Terra's string is 0.920-0.989.
- So "other" and "none" separate the non-codes cleanly at Terra's boxes.

**Every box (8,794):**
- The open posterior's top is "other" on 7,031 boxes, "none" on 1,129, and a code on 634.
- Mean q (p of 180°) is 0.35.

## DVC

- **dvc.yaml:** `build --write-dvc-yaml` adds the three stages; committed in 90b1069.
- **`pixi run build solver-text-crops`** ran the stage (936 s: crops 31 s, font 717 s, prior 91 s, synthetic crops
  74 s). The warp stages re-recorded their cli.py hash; outputs unchanged.
  - The build refused to record: "HEAD moved during the build". Another session committed ba36061, a spike script
    outside the stage's deps.
  - I recorded it with `dvc commit --force solver-text-crops` (11de9e5), the same command the build runs.
- **`pixi run build solver-text-train`** ran 76.6 min on mps. Refused for the same reason: the controller's commits
  dc32995 and be37735 (docs and spike scripts).
  - `git diff 11de9e5..HEAD --stat -- pipeline/src pipeline/inputs pipeline/dvc.yaml pipeline/pyproject.toml
    pipeline/uv.lock` was empty.
  - Recorded with `dvc commit --force solver-text-train` (e48d3b7), as the controller asked. Not rerun.
- **`pixi run build solver-text-read`** ran 2.1 min, recorded by the build (92c8215).
- `dvc status solver-text-crops solver-text-train solver-text-read`: up to date. No build is running.
- **Data sizes:** text-crops 345 MB (four synthetic sets of 74 MB), text-train 34 MB, text-read 22 MB.

## Concerns

1. **Our boxes, not the reader, decide T4.**
   - At our boxes the reader gets 46% of the found answered labels right (41% of all, counting missed boxes).
   - At Terra's boxes it gets 96%, with the same networks and the same posterior.
   - Of 8,794 boxes, only 634 have a code as the open top; the solve has 1,904 prints today.
   - T4 as planned (labels from these boxes) will very likely be far worse than today's final, and plausibly than
     the published map in label accuracy.
   - The lever is the grouping or the detector (T2's merges; the controller's new text-vs-line detector design), not
     the reader. Snapping the axis does not help.
   - **This needs the owner's or the controller's decision before T4.**
2. **The crop rule for our boxes is the ruling's** (the ink-trimmed extent). On boxes whose size and axis match
   Terra's word it reads 0.851 against 0.977 at Terra's crop, so some loss remains even on good boxes; I did not
   chase it.
3. **Three stages, not two.** The training is split from the reading so that box changes never retrain. Same work;
   say if you want them merged.
4. **The direction prior is my construction.**
   - A Gaussian kernel of 5° over the training words' axes, Jeffreys counts, and Terra's direction as the label.
   - Terra's direction is itself noisy for vertical words: the template reader measured a 50% half-turn rate for
     its "top-to-bottom".
   - The 5° width is a smoothing choice, not fitted.
   - The raw classifier probability and the prior are stored for T4. At Terra's boxes this route scores exactly as
     Terra's own direction does.
5. **S2's known synthetic-stream faults are kept**, so the recipe is C's: Terra-box offsets with the opposite sign;
   line windows centred on ink.
6. **One font and one prior for all four quarters** (ruling P4). Its p(code) counts training labels in every
   quarter. The answered blocks are out, so the answered-label measures are unaffected; the held-out-quarter word
   stats are slightly optimistic.
7. **The validation area is not held out** (as in `solver-text-reader`). The monitor is the held-out quarter's words,
   logged only.
8. **Twice the build guard refused to record because other sessions committed during my builds.** Both were recorded
   by `dvc commit` after checking that no dep changed.
9. **T = 1 throughout.** S2's closed numbers used a cross-fitted temperature. Top-1 hardly moves with T, but −log p
   would.
10. **The LLM chain's 96.9% is the plan's figure.** I did not recompute it on these 662 labels.

---

# Fix round 1 (review: three Important findings; the controller's rulings)

Status: DONE. This round supersedes everything above about a direction prior (`axis_prior`, `prior_turn`, `p_turn`,
`axis_prior.npz`, concern 4).

## Changes (commit 7509ac7)

- **(3) and (1): the classifier alone.**
  - The direction mixture's weight is q = the quarter's 0/180 classifier's p(the 0-degree crop is upside down),
    alone. `read.weight(got)` is used for every box, both Terra-box variants and the held-out-quarter words.
  - Removed from `ctc.py`: `axis_prior`, `prior_at` (the wrap bug), `flip_probability`, `AXIS_SD`, `AXIS_STEP`,
    `PRIOR_COUNT`, `D_CLIP`.
  - Removed from `train.py`: the per-quarter table and `axis_prior.npz`.
  - Removed from `read.py`: the prior's loading. Its Terra-direction variant no longer uses the template reader's
    half-turn rates either.
  - `reads.npz` drops `prior_turn` and `p_turn`. **`p_classifier` is the mixture's weight q**, and is kept.
    - `reads.parquet`'s `p_turn` column is now `p_classifier`.
    - `answered.parquet`'s `<prefix>p_turn` columns are now `<prefix>p_classifier`.
    - `measures.json`'s `mean_p_turn` is now `mean_p_classifier`.
  - The per-direction reads are unchanged: `log_forms`, `log_none`, `p_other`, each (n, 2).
- **(2): the reading maths is out of the training stage.**
  - `ctc.py` now holds only the reading maths: `log_softmax`, `string_log_probs`, `read`, `mixture`, `closed_log`,
    `axis_and_turn`. Only `read.py` imports it.
  - `best_path` moved to `model.py`, with the same body; training's `monitor_cer` uses it.
  - `train.py` imports `model` and `store` only. `held.json`'s `direction_words_turned` is computed inline by the same
    formula (angle mod 360 ≥ 180).
  - The closure test (`tests/test_build.py::test_text_reader_stages_deps`) now pins the train stage's code as
    `textread/{__init__,model,store,train}.py` (+ `paths`, `config`), and checks that `ctc.py` is in the read
    stage's deps.
  - `dvc.yaml` regenerated.
- **Docs:** the docstrings (`__init__`, `store`, `train`, `read`, `ctc`) and the README entries say "the
  classifier alone", and the README measures were updated (8157a97).
- **Tests:**
  - removed `test_flip_probability_…` and `test_axis_prior_…`;
  - added `test_the_mixture_weight_is_the_classifier_alone_on_the_first_crop`,
    `test_an_angle_is_an_axis_and_a_turn` and `test_the_best_path_merges_repeats_and_drops_blanks`.

## Tests

```
$ uv run pytest -q -n 0 tests/test_solver_textread.py tests/test_build.py
59 passed in 7.08s
$ uv run pytest -q -n 4
1467 passed, 3 xfailed, 1 warning in 108.86s
```

## R8 evidence

**solver-text-train: re-recorded, not retrained (a6fce98).**
- `dvc status solver-text-train`: changed deps `textread/__init__.py`, `model.py`, `store.py`, `train.py`; `ctc.py`
  left the deps.
- What `train.py` executes, `git diff 90b1069 7509ac7`:
  - the import `ctc, model, store` becomes `model, store`;
  - `ctc.best_path` becomes `model.best_path`, the same body, moved;
  - `tables` and the two `ctc.axis_prior` lines are removed;
  - `turned = np.mod(angles, 360) >= 180` replaces `ctc.axis_and_turn(...)[1]`, the same formula;
  - the `axis_prior.npz` write is removed;
  - in `__init__.py` and `store.py`, docstrings only.
- `fit`, `Pool`, `augment`, `batches`, `targets`, the losses, `monitor_cer`'s logic, the recipes and the seeds are
  unchanged.
- `held.json`'s `direction_words_turned` recomputed by the new code equals the stored values: nw 1683, ne 1575,
  sw 1372, se 672.
- The one output change: `axis_prior.npz` is no longer written. I removed it from `data/solver/text-train/` and ran
  `dvc commit --force solver-text-train`. The lock shows the code hashes plus the output dir going from 11 files to
  10 (35,577,237 → 35,524,760 bytes). The models, `held.json` and `log.json` are the trained ones.

**solver-text-crops: an unplanned on-paper record (d40baae).**
- The first `pixi run build solver-text-read` (from HEAD a6fce98, no build running) planned `solver-text-crops`
  too. Its code closure holds `textread/__init__.py`, `model.py` and `store.py`, so it ran it. A rerun takes 16 min,
  and any byte difference would have cascaded into a retrain.
- I stopped that build within about 5 min: the build, its `sfzmaps stage`, and the pinned `solver text-crops`
  re-exec, all killed. Its output folder was untouched (a stage swaps its `.part` folder in only at the end; no
  `.part` was left), and no process remains.
- `git diff 90b1069 HEAD -- textread/crops.py synth.py inputs.py` is empty. `best_path` is not used by the crops
  stage. The other changes are docstrings.
- `dvc commit --force solver-text-crops`: the lock diff is the three code hashes only; outputs unchanged.

**solver-text-read: rebuilt.**
- `pgrep -fl 'sfzmaps (build|stage)'` was empty. HEAD was d40baae.
- `pixi run build solver-text-read`: 2.1 min, recorded by the build itself (2f01c4b). HEAD did not move.
- `dvc status` for all three stages: up to date. No build is running.

## The new measures (`measures.json`, classifier alone; the old values in brackets)

**At Terra's boxes (662 labels), closed:**

| | top-1 | top-3 | −log p | open top-1 |
|---|---|---|---|---|
| all | 0.961 [0.944, 0.974] (0.962) | 0.992 | 0.143 (0.142) | 0.918 (0.934) |
| nw | 0.984 | 0.995 | 0.054 | 0.962 |
| ne | 0.947 | 0.996 | 0.189 | 0.901 |
| sw | 0.966 (0.978) | 1.0 | 0.074 | 0.933 |
| se | 0.952 | 0.976 | 0.224 | 0.881 |
| unflagged | 0.979 [0.966, 0.989] (0.980) | | | |

- With Terra's direction and with the axis alone, the scores are again equal.
- **Against:** the spike's 0.956–0.973; the LLM chain's 96.9%.

**At our boxes (719 labels):**

| | found / missed | closed top-1 on the found | −log p | every label (missed = wrong) |
|---|---|---|---|---|
| all | 633 / 86 | 0.462 [0.424, 0.501] (0.464) | 5.12 | 0.411 [0.376, 0.446] |
| nw | 163 / 28 | 0.537 | 4.23 | 0.458 |
| ne | 286 / 9 | 0.335 | 6.52 | 0.326 |
| sw | 75 / 21 | 0.613 (0.627) | 3.89 | 0.489 |
| se | 109 / 28 | 0.576 | 3.67 | 0.469 |
| unflagged | 523 / 68 | 0.520 [0.476, 0.563] | | |

**Box error against reader error:**
- On the same 595 labels: 0.963 at Terra's box against 0.477 at ours (difference [−0.53, −0.45]).
- The diagnostic by box type is unchanged:
  - axis off by more than 10°: 0.371 (Terra's box 0.941);
  - merged with linework: 0.185;
  - split: 0.366;
  - axis and size near Terra's word: 0.851.

**Other:**
- **Explaining away:** unchanged in substance. p(other) is 0.97–0.999 on block numbers and street names, p(none)
  0.97–0.99 on non-text windows, and the code mass 0.90–0.93 on code words.
- **Every box:**
  - the open top is "other" on 7,106, "none" on 1,081, and a code on 607 (was 634);
  - mean q is 0.384.

The headline concern (concern 1, box error decides T4) stands unchanged.
