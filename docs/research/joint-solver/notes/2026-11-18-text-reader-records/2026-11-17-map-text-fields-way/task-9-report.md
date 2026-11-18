# Task 9 report: box choice by the reader

Commit edb6f33 (only `pipeline/src/sfzmaps/solver/wordread/choose.py` and `pipeline/tests/test_wordread_choose.py`).

## Built
- `candidates(box, ink, neighbours, widths=None)`: list of (name, corners (k,4,2)): as_found (first), split_gap (widest interior empty gap of the ink projected on the box axis, two halves), join_nearest (nearest neighbour whose gap along the axis <= one word height, centre across the axis within half a height, axis within 10 degrees), trim_start/trim_end (median char width of the box's kind; only when length > 2 widths). Versions that do not exist are left out.
- `choose(box, cands, read_fn, prior_fn=None) -> (name, {"scores", "words"})`. `read_fn(corners, box)` returns {hypothesis: log posterior}, "none" and "other" among them. `prior_fn(hypothesis, box)` returns a log prior, added to every hypothesis (none/other too) and renormalised; None uses the reader's posterior as returned. Does not import context.py.
- My call on fairness: a word's score is the log of its best hypothesis' posterior; a version's score is the SUM over its words (each includes its own none/other mass). Documented in the docstring. Ties keep the earlier version (strict > with 1e-9 slack), as_found first.
- `measure_char_widths(crops)` / `write_char_widths()` / `load_char_widths(f, held_out)`.

## TDD
Tests written first (18 tests); first run failed on the missing module/behaviour; one real failure (renormalising without a prior changed an unnormalised stub's score) led me to renormalise only when a prior is given. Final: tests/test_wordread_choose.py + test_build.py 61 passed; full `uv run pytest -q`: 1654 passed, 3 xfailed.

## Measured char widths (px per character; Terra words in crops.parquet, outside touch_blocks, kind != none)
Saved to `pipeline/data/solver/experiments/wordread/choose/char_width.json` (not committed): all = zoning 22.5, block 8.25, street 9.33, other 19.67; plus a `held_out` variant per quarter (fit without that quarter's words), values within about 1 px for zoning/block/street, other 17 to 21.

## Concerns
- Fold-honest scoring should use `load_char_widths(held_out=<quarter>)`; the pooled fit has seen every quarter.
- The sum rule penalises splits: a split wins only if both halves read clearly better than the whole.
- Word-score of "none"/"other" as best hypothesis is allowed (a candidate whose best is none scores by its none mass).
- "Other" widths are noisy (mixed content, spaces counted in string length).
- join_nearest requires neighbours' axis and line alignment; vertical (axis 90) boxes work via the box's own frame but untested on real data. No real-data run of choose with the reader was done (stub tests only).
- ruff is not installed in the venv, so lint was not run.

## Fix round 1 (one follow-up commit on top of edb6f33)
- Fitted choice: `choice_features(box, cands, read_fn, widths)` (cand, score, gap in the kind's char widths), `label_of(cands, terra_words)` (best mean IoU, a split matches two Terra words), `fit_choice(rows)` (ML by L-BFGS; P(c) ~ exp(beta score + alpha_c + gamma_c gap_c), as_found alpha=gamma=0; refuses rows flagged held_out/answered; boxes without exactly one label skipped), `choose(..., params=None, widths=None)` (params: argmax of the logit, returns `probs`/`prob`). Candidates are `Cand` arrays carrying `.gap`.
- Widths required (`candidates` raises ValueError on None). Unknown kind: no trims. Join excludes the box itself and overlapping neighbours.
- Owner addendum (no hand-set thresholds): join_nearest has no window: the nearest rectangle (2-D distance in the box frame) however far, its distance is gamma's feature; no axis/same-line filter; split exists at any ink gap (existence only), gap is the feature; trim needs only length > one char width (was 2 widths); removed AXIS_OK, MIN_CHARS and MATCH_IOU (label_of has no minimum IoU, None only at zero overlap). Remaining constants are numerical: tie tolerance EPS, ridge 1e-6 (keeps separable fits finite; decides nothing).
- Tests: 72 passed with test_build; full suite 1665 passed, 3 xfailed. New: fit recovers gamma>0, choice flips with gap, confident merged "other" loses to a large-gap split under fitted params, 1 px split loses, vertical box, no-window join, unknown kind, required widths, fit refuses held-out/answered.
- Concern: the fit was only run on synthetic data (Task 11 does the real one); the nearest-box rule can pick a box on another line, which the fit should learn to give a large distance gap.

## Fix round 2
1. Join features: `along` (space along the axis, in the kind's char widths) and `across` (space across the axis, in word heights), both clipped at 0, are separate features; the logit has gamma (along; split, join) and gamma_across (join). The nearest box is still chosen by 2-D distance in px. Test: fitted on synthetic data where only same-line joins are right, a next-line box (0 along, 6 px across) has lower join odds than a same-line box at 6 px.
2. `fit_choice` raises when `held_out` or `answered` is missing (and when any is true).
3. Ridge (1e-6) now also on beta; test of a score-separated fit staying finite.
4. `Candidate` frozen dataclass (name, rects, along, across; iterates as (name, rects)); tests of pickling and of the features going to parquet. Gap features are normalised inside `candidates`, so `choose` and `choice_features` no longer take `widths` (a change from round 1's signatures).
5. `label_of` uses shapely rotated-polygon IoU (Terra words: (x0,y0,x1,y1) upright or (4,2) corners for slanted). I did not reuse Task 1's measures helper (it works on boxes.parquet rows, not on candidates).
Tests: 76 passed with test_build. Full suite: 4 failed + 1 error, all in other agents' WIP files (test_wordread_readercrops.py, test_wordread_words.py), none in mine.

## Fix round 3
- Labels by ink, soft: `label_of` and its rotated-IoU helpers are gone (no IoU anywhere). `label_weights(cands, terra_words, ink)` gives each version P(holds exactly one Terra word): per rectangle, `boxmodel.ink_shares` (reused) gives each overlapping word's share of its ink inside it, `boxmodel.poisson_binomial` gives P(count = 1); a split multiplies its two rectangles' values. Terra words: (x0,y0,x1,y1) upright or (4,2) rotated corners.
- `fit_choice` now takes a `weight` column (soft target) instead of `label`: the box's targets are weights over their sum, and the box counts with min(1, sum of weights), so a box whose versions all weigh about 0 adds about nothing through its weights (my reading of "normalised" plus "contribute nothing"). Refusals (held_out, answered, missing columns) kept.
- Tests added: short box beside another word labels as_found, not the join; word split across two boxes labels the join; two words in a box label the split; empty box weighs 0; slanted word by rotated rectangle; the soft fit recovers beta 0.8, alpha -1.5/-0.7, gamma 1.2 (within 0.03); an all-zero box leaves the fit unchanged. 81 passed with test_build.
- choose.py now imports `wordread.boxmodel` (torch-heavy; it is a new module, no existing stage imports choose).
- Full suite (-n 2, test_wordread_words.py left out): 1800 passed, 3 failed, none in my files: two in test_wordread_context.py (another agent's WIP) and tests/test_wordread_run_choose_pack.py::test_fold_fit_uses_only_the_other_quarters_outside_the_answered_blocks, which calls fit_choice with a `label` column: Task 11's caller must pass `weight` (label_weights) now. Commit: see git log.

## Fix round 4 (commit 91ad2a1): each version's own quality, no shares
The controller's ruling after Task 11's "Why joins": the conditional logit's argmax split as_found's share with its two
trims (near-copies) while a lone join stood alone. Replaced:
- `fit_quality(rows)`: a fractional logistic regression of each version's label weight w (`label_weights`, unchanged: P it
  holds exactly one whole Terra word, by ink) on its features; every version is its own observation. A word's eta =
  intercept + alpha_version + gamma_version along + gamma'_join across + kind offset (block/street/other/unknown; zoning 0)
  + beta score_word (+ count1 log P(count = 1)_word, Task 15's box model, fitted only when the rows carry `word_count1`).
  A version's quality is the product over its words of sigmoid(eta), so a split's is its halves' product, as its label is;
  fitted by quasi-likelihood w log q + (1 - w) log(1 - q), L-BFGS with the exact gradient, ridge 1e-6 (numerical).
  Returns the params plus boxes, versions, nll, converged, iterations. Refusals kept (held_out/answered missing or true;
  missing weight/word_scores/kind).
- `choose(..., params, count_fn=None)`: the version of the highest predicted quality (`predict_quality`), as_found first
  on ties; returns `quality` (every version's) and `prob` (the chosen one's) in place of `probs`. Params with count1 need
  `count_fn` (ValueError). `choice_features` adds `word_scores` (each word's log posterior), `kind`, and `word_count1`
  with `count_fn`. The plain arm (params None) is unchanged. `fit_choice` is gone; candidates and label_weights unchanged.
- run_choose_pack.py: `fold_fit` calls fit_quality (CLI unchanged); run.json gains `calibration` (mean predicted quality vs
  mean weight per version, training and held-out rows); `picks.parquet` holds every box's pick per arm (the chosen-boxes
  table keeps a mutual join once, so it under-counts joins).
- Tests (tests/test_wordread_choose.py, 42): near-copies don't lose to a lone join of equal or lower quality (hand params:
  a tie keeps as_found; and a fit on data where as_found's family is right in 55% of boxes and the join in 45%, which by
  per-box shares would favour the join: as_found's quality 0.55 wins, and each version's quality is the same with the
  trims left out); the fit recovers 16 known coefficients within 0.02 with splits as products; a split wins when both
  halves are single words and its quality equals the product of its halves' (end to end: label_weights, stub reads and a
  stub box model); a join wins only for a word cut across two boxes (end to end); count1 absent without the column;
  refusals; the earlier gap / next-line / separated-fit tests ported. With test_wordread_run_choose_pack.py and
  test_build.py: 96 passed. Full suite (-n 2, test_wordread_words.py included, no hang): 1849 passed, 7 skipped, 3 xfailed.
- No box model was trained yet (no BoxNet file under experiments/wordread/boxmodel), so the run below has no count1 term.

### NE preliminary (today's boxes and readers; runs quality-ne-s0/s1, reads from the version cache)
Fold NE's fit: 4288 boxes, 19284 versions, converged in 39 iterations. beta 1.607, intercept 0.103; alpha split -1.356,
join -0.387, trim_start -0.722, trim_end -0.699; gamma along split -0.445, join -0.360; across join -0.031; kind block
+1.117, street -0.140, other -1.324. Calibration: in sample exact per version (as the offsets force); held-out NE mean
quality vs weight: as_found 0.571 vs 0.549, join 0.386 vs 0.304, split 0.113 vs 0.143, trims 0.43 vs 0.46.
| seed | without | plain | fitted | fitted - without (paired, blocks resampled) |
|---|---|---|---|---|
| 0 | 92 of 285 | 51 | 81 | -0.0386 [-0.0650, -0.0139] |
| 1 | 88 | 69 | 86 | -0.0070 [-0.0309, +0.0174] |
| mean of seeds | | | | -0.0228 [-0.0412, -0.0036] |
(Before this round: logit with IoU labels 77 / 84, with ink labels 46 / 62.) The headline still does not rise; by the
keep rule (5) is not kept on these boxes and readers.
- Chosen kinds, NE fitted: s0 as_found 2225, join 191, trim_end 20, trim_start 16, split 0; s1 as_found 2234, join 164,
  trim_end 36, trim_start 18 (the logit with ink labels: join 1559 / 1484).
- Joins whose Terra word is split across the two boxes (Task 11's joins.py measure: both boxes meet one Terra word at IoU
  >= measures.SPLIT_PART), over every box's pick: s0 11 of 191 (5.8%), s1 7 of 164 (4.3%).
- Labels changed against without: s0 13 right to wrong (11 held by a chosen join, 1 trim_start, 1 trim_end), 2 wrong to
  right; s1 7 right to wrong (5 join, 2 trims), 5 wrong to right (4 join, 1 trim).
Script for the join numbers: scratchpad quality_joins.py (not committed).
