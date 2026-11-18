# Task 3 review (opus): Needs fixes

## Spec compliance

Partially compliant.

**What is in place:**
- the interfaces: `score_label`, `evidence` / `posterior_of`, `code_prior`, `Constants`, `printed_forms`,
  `Prior.learn`;
- carried items 1, 2, 4 and 5 (item 6, optional, skipped);
- the four Review Focus tests, which drive the real engine;
- clean leakage.

**What fails:**
- **No hard cutoffs:** the coarse pruning changes posteriors, though its docstring says it can't.
- **"A reading point off the label can still find it":** for point labels, a prefix code or "other" wins over the
  full code when the point is off the label.

## Strengths

- **One engine for every hypothesis.** Codes, "other" and "none" are each a log-likelihood ratio against bare
  background, over the same observed px. Line px are zero in every channel: the ratio, the claimed ink and the word
  ink (posterior.py:343, 480-484, 495-496). `test_line_through_h_does_not_make_it_m` checks it exactly.
- **Code evidence is exact sums:** logsumexp over cells, both placement components and every chunk (833-836, 858).
  The per-character jitter is a proper marginal (730-749), normalised over codes, none and other (1175-1195).
- **No FFT wrap-around aliasing:** patch-size padding plus the `_gather` validity mask (479, 521).
- **Leakage is clean.**
  - Every fitted piece takes training labels and `_guard`s held-out ones (189, 235, 272, 1067; glyphs.py:718).
  - The NCD forms come from the code names (208-228).
  - Nothing reads the table's code at a label's spot.
- **The priors are measured, and recorded** (`PosePrior.measured`).
- **The Evidence / posterior split** lets Task 4 fit the masses without rescoring.
- **The runner's merge order doesn't matter.**

## Critical

**1. Pruning is not a bound and moves posteriors** (posterior.py:67-69, 99, 790-817).

The coarse score:
- uses only the core placement component (`rows[0]`, 802), while the full pass also sums the tail (up to ~50 nats
  more for a text one word height off);
- scores codes only, with no "other";
- scores the likeliest residual angle only (649-650);
- is judged only at the lowest temperature (794, 805).

Measured with the tests' setup (reader font, TRUE_NOISE, `_prior`), PRUNE_LOG 40 against 1e9:
- **off-point RM-2,** point 12 px (0.5 word heights) past the middle, one temperature: unpruned 0.829, pruned 0.913
  (|Δp| 0.085);
- **clean box label at temperatures (5, 30):** log E of RH-1(D) is 22.7 nats low at T=5, and 6.65 at T=30;
- **a point label at T=30** in the (5, 30) grid: 11.7 nats low.

`prune_check.py` (5 labels, one temperature, |Δposterior|) can't see changes in −log p for an unlikely true code.

**Fix:** drop the pruning, or make the proxy honest:
- both components, with "other";
- keep a chunk if any grid temperature is within the margin.

**Test:** pruned equals unpruned on off-point and multi-temperature synthetic cases. This must be in place before
Task 4's temperature grid.

**2. Point labels have no extent term: a reading that stops short of the ink is free** (posterior.py:347-353,
767-776, 849-855).

With no Terra word, `_log_other_text` reduces to softplus(log ρ + gG). Every hypothesis scores about L/(ρF) whatever
it covers, so only the placement prior separates a full code from its prefix. "Other" also sums every partial
string.

**Synthetic results** (unpruned):

| Label | Point | Reads |
|---|---|---|
| clean RH-1(D) | 6 px (0.25 word heights) before its middle | RH-1 0.574 |
| clean RH-1(D) | 12 px before | RH-1 0.912 |
| clean RH-1(D) | 24 px before | RH-1 0.999 |
| clean RH-1(D) | 24 px after | "other" 0.854 |
| clean RM-2 | 6 px before | "other" 0.608 |

**Real run** (`full_cpu8.records`, default constants):

| | Labels | Read "other" | Median top p |
|---|---|---|---|
| Point labels | 136 | 13 (9.6%) | 0.63 |
| Box labels | 1,769 | 8 (0.5%) | 0.99 |

**Exposure:** 63 of the 731 held-out labels are point labels. Sol's measured tail (share 0.34, sd 0.38 word heights)
puts about a fifth of points 0.25 word heights or more off.

**Fix:** from training and synthetic evidence only, now. Either charge unexplained adjacent text ink along the line
(forward-backward word segmentation), or estimate a word extent from the ink around the point.

**Test:** an off-centre point, 0.25 to 1 word heights on both sides, reads the full code.

## Important

**3. A code's printed forms are weighted equally** (posterior.py:1183-1190, `- math.log(len(vals))`).
- "RH2" is 1 of 140 consensus strings for RH-2, yet it halves RH-2's evidence. 24TH-NOE VALLEY and HAYES-GOUGH are
  divided by 6.
- Measured on the 120 training labels (`cmp_rule_run.pkl`; `review3/forms_weight2.py`): with p(form | code) from
  training consensus counts + 0.5, agreement goes from 110 to 108. RM-2 #635 (0.66) and #990 (0.57) flip to RH-2.
- **Fix:** measure p(form | code), smoothed. Report the RH-2/RM-2 lean honestly.

**4. Speed.** 48,327 worker-seconds (93 min on 8 workers). Point labels are 7% of the labels and half the work. Plan
this before Task 4.

## Minor

5. **`REACH_LOG` is an unmeasured window** (40-41, 98, 399-425). Check 12 against ~20 on off-point cases.
6. **The "other" Viterbi path is chosen without the end-placement weight** (839-848). For box labels the full-word
   bonus (up to −log ρ ≈ 12 nats) is invisible to the path choice.
7. **Determinism isn't enforced in the package:** no `torch.use_deterministic_algorithms(True)` and no thread pins.
   `evidence(device="mps")` doesn't take `gpu_turn` (71-72, 779-780).
8. **Two measures of the same size spread:** `Font.scale_sd` is persisted but unused; the posterior uses
   `PosePrior.ratio` (1108-1111). Say which one is authoritative.
9. **Test gaps:**
   - nothing checks that pruning leaves the sums unchanged;
   - no off-centre point test;
   - `test_text_that_is_no_code_reads_other` asserts only the top hypothesis (test_codereader.py:539);
   - the guards in `Prior.learn` and `measure_pose_prior` are untested;
   - `_guard` checks `held_out` only, not block-mates (201-205).
