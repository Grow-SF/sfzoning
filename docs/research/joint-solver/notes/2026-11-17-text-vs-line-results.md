# Text detector, text vs line: results of the deciding runs (2026-11-17)

The plan is `docs/superpowers/plans/2026-11-17-text-detector-text-vs-line.md` (Task 7) and the spec is
`docs/superpowers/specs/2026-11-17-text-detector-text-vs-line-design.md`. The verdicts come from
`pipeline/scripts/textlines/decide.py`; its rule was written before the runs. Run outputs are in
`pipeline/data/solver/experiments/textlines/` and are not committed. Cost: $0, with no LLM calls.

## Summary for the owner

- **What "control 0" is, and what "against the old model" means.** Control 0 is the label rule: zones (84 px, around
  unmatched points only) plus the in-box rule (line ink inside a word box is "unknown" in training, not "text"). Every
  "control vs old-ne" or "control vs old-sw" number in this note also carries a change of recipe: the old quarter
  model (`data/solver/text-ne/model.pt`) is the old `Recipe()` (8000 steps, full size, loss on all known px), while
  the control uses `stage.DEFAULT_RECIPE` (4000 steps, half size, ink+1px). So those differences mix three changes:
  the recipe, the zones and the in-box rule. See "Post hoc decomposition" below for a partial split on NE.
- **What the label fix does (control 0 against the old NE model; recipe, zones and in-box rule together).**
  - On the held-out NE quarter it calls much less line ink inside labels text than the old NE quarter model does.
  - District-border ink in word boxes (M1a in box): 0.63 to 0.40, a change of -0.225 [-0.257, -0.194].
  - All straight-line ink in word boxes (M1b in box): 0.82 to 0.55, a change of -0.272 [-0.291, -0.245].
  - Against today's held-out detectors (r328), which call nearly all in-box ink text (its M1c is 0.37-0.55), the
    changes are -0.59 and -0.42.
  - In the pictures, in the panel windows, block, street and district lines through labels turn grey and the letters
    stay red. Over all boxes M1b in box is still about 0.55.
  - On SW (one seed, the control only), against the old SW model:
    - M1a in box: 0.73 to 0.41, a change of -0.319 [-0.384, -0.255];
    - M1b in box: 0.80 to 0.52, a change of -0.275 [-0.297, -0.214].
- **What it costs, against the old NE model** (again recipe, zones and in-box rule together):
  - **Letters of clean words** (words no line touches) called text: 0.957 to 0.930, a change of -0.027
    [-0.040, +0.003]. That is beyond the noise (0.021).
    - Post hoc: most of this is in words of the kind "other" (-0.050).
    - Post hoc: zoning-code letters off the lines (M2 for codes) fall 0.984 to 0.951, a change of -0.033
      [-0.039, -0.028].
  - **Ink outside every box** called text: 0.025 to 0.052, a change of +0.027 [+0.017, +0.037]. That is beyond the
    noise (0.021).
  - **The dashed pier-head line** is red in all six runs. It is grey in r328, old-ne, all-sheets and the trial.
  - Glyph ink in boxes away from lines (M2) is within the noise against the old model: -0.005 [-0.023, +0.010].
    M2's base is, by construction, the complement of what the control leaves unknown (glyph ink off the run rule's
    line px and more than 1 px from them), so M2 cannot see the in-box rule's cost. That cost shows in
    `m2_clean_words` and in the per-kind numbers (codes, below).
- **What it costs on SW, against the old SW model** (same mix of recipe, zones and in-box rule):
  - Clean-word letters: 0.981 to 0.909, a change of -0.072 [-0.098, +0.001] (noise 0.049).
    - Post hoc: nearly all of it is in "other" words in 5 blocks.
  - M2: -0.014 [-0.027, -0.008]. That is just beyond its noise (0.0136).
    - Post hoc: street-name letters -0.048 and zoning codes -0.018.
    - In the picture, street names written inside curved road bands are grey, as in the old SW model.
  - Ink outside boxes does not rise on SW: +0.001.
- **The composites (variant 1): no change beyond the seed noise.** The block intervals of M1c (-0.0094) and M2
  (+0.0017) exclude 0 but sit inside the seed noise. M1c is what lets the variant pass old-ne as written (+0.018, inside
  the noise, against the control's +0.027, beyond it).
  - Variant less control, three seeds each:
    - M1a in box +0.006 [-0.006, +0.018];
    - M1b in box -0.004 [-0.012, +0.002];
    - M2 +0.002;
    - M1c -0.009;
    - clean words +0.0005.
  - All of these are inside the noise. The controller saw no visible difference.
- **By the rule written in advance, nothing is adoptable**, so the downstream preview (Task 8) does not run.
  - On NE both recipes fail "M2 not lower than r328's beyond the noise"; on SW the control (the only run there) does.
    That condition did not pass for any line-aware detector here, and is unlikely to: r328 calls 99.9% of the glyph
    ink and 97% of the line ink in boxes text. See the post-hoc section.
  - Against the old quarter model alone (post hoc):
    - on NE the control also fails ink outside boxes and the clean-word guard;
    - on NE the variant fails only the clean-word guard;
    - on SW the control fails M2 by a hair and the clean-word guard.
- **What remains open** is listed at the end:
  - the dashed-line over-call;
  - half the line ink in boxes is still called text;
  - the code-letter loss;
  - a fair adoption bar.

## The rule (written before the runs; `decide.py`)

- Each measure is the share of its ink called text (chance >= 0.5). An arm's value is the mean over its runs.
  - A difference is the variant less the control (or an arm less a baseline), on the blocks both scored.
  - It comes with the 95% paired bootstrap interval over blocks (`linegate.paired`, 1000 draws).
- **Noise** = max(the control's largest pairwise seed difference, the paired interval's half-width).
- **Variant over control** requires all of:
  - M1a-in-box lower, with the interval below 0;
  - M2 not lower by more than the noise;
  - M1c not higher by more than the noise;
  - m2_clean_words not lower by more than its noise (the controller's ruling for this task);
  - `eye_ok`.

  The verdict is `pending-eye` without the eye check. It is `mixed` on any split (for example M1b falls and M1a
  does not), and `control` or `variant` otherwise.
- **Adoptable**, for an arm against r328 and the old model of its quarter:
  - M1a-in-box and M1b-in-box lower, with the intervals below 0;
  - M2 not lower, and M1c not higher, by more than the noise;
  - the audit's text pieces called text not lower beyond the binomial interval. Concretely, the arm's estimated missed
    text pieces are at most the baseline's Clopper-Pearson 95% upper bound.

  The table reports this rule both as written and with the clean-word guard added. The all-sheets detector is a
  reference only, being in-sample.
- **SW:** the NE winner and the control at seed 0 each. The NE winner is the control, so only the control runs. The
  seed noise comes from the NE control seeds.

## Baselines (Task 3; NE values are on NE blocks)

| source | m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | m1a |
|---|---|---|---|---|---|---|
| r328, all quarters | 0.987 [0.983, 0.992] | 0.975 [0.959, 0.984] | 0.999 | 0.366 | 0.995 | 0.668 |
| r328, NE blocks | 0.9922 | 0.9694 | 0.9994 | 0.5474 | 0.9845 | 0.8110 |
| old-ne (own quarter) | 0.6294 | 0.8177 | 0.9727 | 0.0252 | 0.9565 | 0.0380 |
| r328, SW blocks | 0.9811 | 0.9734 | 0.9987 | 0.2736 | 0.9985 | 0.5167 |
| old-sw (own quarter) | 0.7333 | 0.7950 | 0.9832 | 0.0336 | 0.9812 | 0.0387 |
| all-sheets (in-sample, reference) | 0.579 | 0.804 | 0.983 | 0.036 | 0.9956 | 0.046 |

## NE: every run

Recipe: DEFAULT_RECIPE with labels `lines-unknown`. The variant adds composites 0.3396 (P is the ink-only crossing
rate, NE held out). Each run trains 6.4 min on mps. The variant's runs had `git_status_src` empty and head f5bc3aa.
Each composite run made about 8,100 (a) and 8,100 (b) composites out of 48,000 crops (33.8%).

| run | m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | m1a | overlap_band |
|---|---|---|---|---|---|---|---|
| control-ne-s0 | 0.3925 | 0.5102 | 0.9653 | 0.0405 | 0.9271 | 0.0389 | 0.5450 |
| control-ne-s1 | 0.4095 | 0.5936 | 0.9756 | 0.0616 | 0.9352 | 0.0538 | 0.6272 |
| control-ne-s2 | 0.4105 | 0.5320 | 0.9620 | 0.0545 | 0.9275 | 0.0506 | 0.5643 |
| composites-ne-s0 | 0.3992 | 0.5364 | 0.9717 | 0.0402 | 0.9288 | 0.0407 | 0.5738 |
| composites-ne-s1 | 0.4197 | 0.5540 | 0.9705 | 0.0412 | 0.9329 | 0.0462 | 0.5894 |
| composites-ne-s2 | 0.4123 | 0.5328 | 0.9660 | 0.0472 | 0.9297 | 0.0561 | 0.5673 |

**Noise** is the control's largest pairwise seed difference on NE:

| m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | m1a |
|---|---|---|---|---|---|
| 0.0180 | 0.0834 | 0.0136 | 0.0211 | 0.0082 | 0.0149 |

**Variant less control** (three seeds each, 713 blocks):

| measure | diff [95%] | half-width | noise | outcome |
|---|---|---|---|---|
| m1a_in_box | +0.0063 [-0.0060, +0.0184] | 0.0122 | 0.0180 | does not fall |
| m1b_in_box | -0.0042 [-0.0119, +0.0018] | 0.0069 | 0.0834 | does not fall |
| m2 | +0.0017 [+0.0003, +0.0038] | 0.0018 | 0.0136 | held |
| m1c | -0.0094 [-0.0147, -0.0039] | 0.0054 | 0.0211 | held (falls) |
| m2_clean_words | +0.0005 [-0.0009, +0.0010] | 0.0010 | 0.0082 | held |
| m1a (diagnostic) | -0.0001 [-0.0031, +0.0032] | 0.0031 | 0.0149 | |
| m1a_in_box_context (diagnostic) | +0.0058 [-0.0002, +0.0128] | 0.0065 | 0.0572 | |
| m1b_near (diagnostic) | -0.0058 [-0.0078, -0.0040] | 0.0019 | 0.0157 | |
| m1b_far (diagnostic) | -0.0126 [-0.0227, +0.0015] | 0.0121 | 0.0251 | |

**Verdict NE: `control`.**
- The variant gains on neither line measure, and no guard fails.
- With `eye_ok` True (the controller's) the rule gives `control`.

**Adoptable on NE** (arm less baseline, 95% interval; the noise of each guard in brackets):

| arm vs baseline | m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | audit | as written | with clean guard |
|---|---|---|---|---|---|---|---|---|
| control vs r328 | -0.588 [-0.620, -0.554] | -0.424 [-0.473, -0.383] | -0.0317 [-0.0454, -0.0207] (0.0136) **fails** | -0.495 [-0.632, -0.350] (0.141) | -0.0545 [-0.0806, +0.0006] (0.0406) **fails** | held | no | no |
| control vs old-ne | -0.225 [-0.257, -0.194] | -0.272 [-0.291, -0.245] | -0.0051 [-0.0231, +0.0104] (0.0167) | +0.0270 [+0.0167, +0.0366] (0.0211) **fails** | -0.0266 [-0.0401, +0.0025] (0.0213) **fails** | held | no | no |
| variant vs r328 | -0.582 [-0.614, -0.546] | -0.428 [-0.478, -0.386] | -0.0300 [-0.0430, -0.0196] (0.0136) **fails** | -0.505 [-0.638, -0.365] (0.136) | -0.0540 [-0.0796, +0.0003] (0.0400) **fails** | held | no | no |
| variant vs old-ne | -0.219 [-0.247, -0.188] | -0.277 [-0.294, -0.252] | -0.0033 [-0.0205, +0.0113] (0.0159) | +0.0176 [+0.0116, +0.0227] (0.0211) | -0.0261 [-0.0391, +0.0022] (0.0206) **fails** | held | yes | no |

Neither recipe is adoptable on NE: each needs every baseline, and both fail M2 against r328. In the "vs old-ne" rows
the arm and the old model differ in recipe as well as in label rule (see the summary), so those rows measure the whole
change, not the label rule alone.

## SW (control only, seed 0; noise from the NE control seeds)

- **The run.** `runs/control-sw-s0` trained 6.4 min on mps, at head f5bc3aa, with `git_status_src` empty.
- **Its values.**

  | m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | m1a | overlap_band |
  |---|---|---|---|---|---|---|
  | 0.4138 | 0.5203 | 0.9690 | 0.0350 | 0.9093 | 0.0314 | 0.5431 |

- **Adoptable on SW** (control less baseline; one seed; the noise of each guard in brackets, from the NE seeds or the
  half-width, whichever is larger):

| baseline | m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | audit | as written | with clean guard |
|---|---|---|---|---|---|---|---|---|
| r328 | -0.567 [-0.630, -0.498] | -0.453 [-0.580, -0.404] | -0.0297 [-0.0549, -0.0166] (0.0191) **fails** | -0.239 [-0.382, -0.137] (0.123) | -0.0892 [-0.1209, -0.0002] (0.0604) **fails** | **fails** (37.5 missed text pieces vs r328's bound 24.9) | no | no |
| old-sw | -0.319 [-0.384, -0.255] | -0.275 [-0.297, -0.214] | -0.0142 [-0.0274, -0.0082] (0.0136) **fails** | +0.0014 [-0.0048, +0.0052] (0.0211) | -0.0719 [-0.0977, +0.0006] (0.0491) **fails** | held | no | no |

- **Verdict SW: the control does not pass the rule on one seed (M2 -0.0142 vs noise 0.0136).** It fails on M2
  against both baselines (by a hair against old-sw), on the clean-word guard, and on the audit against r328.
- **Reading this.**
  - The line gain repeats on SW: M1a in box -0.319 (NE -0.225) and M1b in box -0.275 (NE -0.272).
  - M1c does not rise on SW.
  - The control on SW is one seed, so a verdict that hinges on M2 -0.0142 against a noise of 0.0136 is a near-tie.
  - The audit against r328 rests on 3 against 0 text pieces in 64, by one eye.

## The controller's eye notes (verbatim)

The controller looked at `runs/compare-ne.png`: all six runs, r328 and old-ne.

- **eye_ok = True for the variant's own panels.** Line ink through and under labels is grey (rm2b, rm2c, the block
  line through RH-2), and the letters stay red. But the variant shows NO visible difference from the control. The
  verdict stays `control`, by the numbers, as you computed.
- **The big vertical "RM-2" (rm2a) is red in all six runs, the control included.** The controller's earlier note
  (Task 5) that it went grey was a misreading of a downscaled picture. Record that correction plainly.
- **The dashed pier-head line (window "dashed") is red in all six runs,** and grey in r328, old-ne, all-sheets and
  the trial. This is a new over-call, consistent with m1c rising against old-ne. Its cause is not established; see
  "What remains open".
- **Control s0 only:** parts of "UNION" (residential) are grey. The other runs keep it red.

**Correction, stated plainly:**
- The Task 5 report and the controller's Task 5 note said the big vertical "RM-2" in window rm2a was missed (grey)
  by the control.
- That was wrong. It is red in `runs/control-ne-s0/panel.png` (the file is unchanged since its run) and in every
  other run.
- The Task 6 review's caution ("composites do not restore long letter strokes of real codes") therefore had no
  regression to explain in this window.

## Noticed after the runs (post hoc; the verdicts above are as written and unchanged)

- **The r328 M2 condition is uninformative.**
  - "M2 not lower beyond the noise against r328" did not pass for any line-aware detector here, and is unlikely to.
    r328 calls nearly all in-box ink text: M2 0.9994, M1b in box 0.97, M1c 0.55 on NE.
  - A detector that leaves line ink grey also loses some glyph ink next to lines. Passing would need M2 >= 0.986 on
    NE, which no line-aware detector here reaches (the old NE model has 0.973).
  - So against r328, M2 says nothing about whether letters were lost.
- **Against the old quarter model alone:**
  - NE, control: fails M1c by +0.027 (noise 0.021) and the clean-word guard by -0.027 (noise 0.021). It passes M1a,
    M1b, M2 and the audit.
  - NE, variant: fails only the clean-word guard, by -0.026 (noise 0.021). M1c is +0.018, inside the noise. It is
    adoptable against old-ne by the rule as written, which has no clean-word guard.
  - SW, control:
    - fails M2 by -0.0142 (noise 0.0136) and the clean-word guard by -0.072 (noise 0.049);
    - passes M1a, M1b, M1c (+0.001) and the audit.

    So, against the old quarter models alone:
    - with the clean-word guard, nothing passes;
    - as written, only the variant on NE passes, and the variant was not run on SW. The NE winner was the control.
- **Where the clean-word loss lies** (NE, control less old-ne, by the kind of the word):
  - "other" -0.050 [-0.057, +0.023]. Half of the clean-word ink is in "other" words; the bootstrap interval is skewed
    over 32 blocks.
  - Zoning codes -0.009 [-0.013, +0.000].
  - Block numbers -0.0002.
  - Street names -0.0001.
  - On SW the loss is "other" words -0.171 (0.955 to 0.784, in 5 blocks; 42% of the clean-word ink). Codes are
    -0.004, block numbers -0.0004 and street names -0.002.
  - So the clean-word guard mostly measures a few large "other" words (titles, legends or notes: not looked at word
    by word). The clean code and number letters lose under 1%.
- **Zoning-code letters away from lines** (m2_by_kind for codes) fall from 0.984 to 0.951: -0.033
  [-0.039, -0.028] for the control and -0.029 for the variant. Block numbers fall -0.008 [-0.012, -0.005].
  - These losses are on code letters, the text the reader needs.
  - The pictures do not show them as whole missed words.
  - On SW: codes -0.018 [-0.035, -0.010], block numbers -0.010, street names -0.048 [-0.057, -0.039]. The street
    names written inside curved road bands are grey in `runs/compare-sw.png` (window sw_dense), as they are for old-sw.

## Post hoc decomposition of control against old-ne (NE units; the trial baseline)

"Control vs old-ne" changes three things at once (see the summary). `baselines/trial` is a third point between them:
the new recipe (`DEFAULT_RECIPE`), the boxes rule (no in-box rule) and 30-px zones. It was trained on every sheet, so
it is mostly in-sample on NE: read its numbers as an upper bound on what the recipe does for NE, not as held-out
values. Each row is a `linegate.paired` difference (95% interval over the NE blocks; 1000 draws, seed 0). The three
columns are exact: trial less old-ne, control less trial, and their sum (control less old-ne).

| measure | trial less old-ne (recipe, boxes rule, 30-px zones) | control less trial (84-px zones + in-box rule) | control less old-ne |
|---|---|---|---|
| m1c (ink outside boxes) | +0.0161 [+0.0131, +0.0195] | +0.0109 [-0.0015, +0.0230] | +0.0270 [+0.0167, +0.0366] |
| m1a_in_box | -0.0060 [-0.0334, +0.0249] | -0.2193 [-0.2466, -0.1925] | -0.2253 [-0.2567, -0.1943] |
| m1b_in_box | -0.0147 [-0.0273, -0.0057] | -0.2577 [-0.2806, -0.2275] | -0.2724 [-0.2911, -0.2447] |
| m2_clean_words | +0.0286 [-0.0012, +0.0421] | -0.0552 [-0.0821, +0.0017] | -0.0266 [-0.0401, +0.0025] |
| m2_by_kind codes | +0.0012 [-0.0017, +0.0045] | -0.0344 [-0.0396, -0.0294] | -0.0332 [-0.0389, -0.0280] |

Reading:
- **Over half of the rise outside boxes goes with the recipe:** +0.016 of the +0.027. The trial has the old rule's
  boxes and the new recipe and already rises beyond its interval; the remaining +0.011 has an interval that includes 0.
- **The line gain and the code-letter loss are the label rule's** (zones plus the in-box rule, not separated here):
  M1a in box -0.219 and M1b in box -0.258 of the -0.225 and -0.272, and codes -0.034 of -0.033. The recipe alone
  moves M1a in box by nothing measurable and M1b in box by -0.015.
- **Clean words** move the other way under the recipe (+0.029, interval includes 0) and fall under the label rule
  (-0.055, interval includes 0), so the control's -0.027 is a small net of two larger moves.
- The decomposition is partial: the trial's zones (30 px) differ from the control's (84 px), so "the label rule" here
  is zones plus the in-box rule. Which of the two does what is not separated. The trial is in-sample on NE.

## Audits (missed letter-sized pieces outside every box, judged by eye)

- **How they were made.** A run's own audit draws 64 of its missed pieces (mean chance below 0.5) on its quarter. The
  baselines' audits on one quarter come from `decide.py audit <q>`, with the same pool and draw.
- **What was counted.** "Plainly text" means the boxed piece is a letter or digit. Pieces of digits fused into a line
  were counted as borderline, not as text. One person counted (the Task 7 implementer); there is no second judge.
- **The estimate.** Missed text pieces = missed × the share judged text, with Clopper-Pearson 95% bounds. The seeds of
  an arm are pooled.

| source | pool | missed | text / drawn | share [95%] | missed text pieces [95%] | borderline tiles |
|---|---|---|---|---|---|---|
| control NE s0/s1/s2 | 1459 | 1279 / 1246 / 1246 | 1/192 (s0 tile 22, the "0" of "40") | 0.005 [0.000, 0.029] | 6.5 [0.2, 36.0] | s0: 5, 8, 29; s1: 30, 42; s2: 41, 47 |
| composites NE s0/s1/s2 | 1459 | 1275 / 1284 / 1248 | 0/192 | 0.000 [0.000, 0.019] | 0.0 [0.0, 24.1] | s0: 15, 26; s1: 8, 23, 41, 45, 63; s2: 15, 24, 25 |
| r328 on NE | 1459 | 449 | 0/64 | 0.000 [0.000, 0.056] | 0.0 [0.0, 25.1] | none |
| old-ne on NE | 1459 | 1379 | 1/64 (tile 42, the "T" of a large "STR...") | 0.016 [0.000, 0.084] | 21.5 [0.5, 115.9] | 9 |
| control SW s0 | 1032 | 799 | 3/64 (tiles 25, 38, 44: street-name letters) | 0.047 [0.010, 0.131] | 37.5 [7.8, 104.6] | 27, 50, 51 |
| r328 on SW | 1032 | 445 | 0/64 | 0.000 [0.000, 0.056] | 0.0 [0.0, 24.9] | none |
| old-sw on SW | 1032 | 821 | 3/64 (tiles 25, 39, 63: street-name letters, the "T" of "...CTS") | 0.047 [0.010, 0.131] | 38.5 [8.0, 107.5] | 7, 10, 18, 47, 49, 50 |

- The audit condition holds everywhere except control SW against r328. There the control has 37.5 estimated missed
  text pieces, against r328's upper bound of 24.9: 3 in 64 against 0 in 64.
- Almost every missed letter-sized piece is a line fragment, a corner or a junction stub, for every detector.
- The detectors that leave lines grey miss more such pieces than r328 does. Ratios of the missed pools, per quarter
  (recomputed from the audits' `missed_below_0.5`; the seeds of an arm averaged):
  - NE, against r328's 449: control 1257 (2.8x), composites 1269 (2.8x), old-ne 1379 (3.1x);
  - SW, against r328's 445: control 799 (1.8x), old-sw 821 (1.8x).

  So it is about 2.8x to 3.1x on NE and 1.8x on SW, not a flat 3x. That "they are lines too" is one person's count of
  64 tiles per source (see the table), not a measured share of the pool.

## 1990-96 (sheets 22-33, the quarter's px; there is no table, so no M1a; no baselines)

- The NE control runs' `extra/` folders were gated again with `decide.py regate-extra` so they carry
  m2_clean_words. Their older measures came out identical, which was checked before writing.
- The variant tracks the control: it is a little lower on M1c and a little higher on M2.

| run | m1b_in_box | m1c | m2 | m2_clean_words |
|---|---|---|---|---|
| control-ne-s0 | 0.5253 | 0.0442 | 0.9687 | 0.9542 |
| control-ne-s1 | 0.6044 | 0.0680 | 0.9773 | 0.9569 |
| control-ne-s2 | 0.5432 | 0.0611 | 0.9632 | 0.9534 |
| composites-ne-s0 | 0.5514 | 0.0441 | 0.9731 | 0.9535 |
| composites-ne-s1 | 0.5712 | 0.0468 | 0.9738 | 0.9574 |
| composites-ne-s2 | 0.5539 | 0.0525 | 0.9702 | 0.9535 |
| control-sw-s0 (SW px) | 0.5061 | 0.0387 | 0.9774 | 0.8994 |

## Pictures

All paths are under `pipeline/data/solver/experiments/textlines/`.
- Panels, one per run:
  - `runs/control-ne-s{0,1,2}/panel.png`;
  - `runs/composites-ne-s{0,1,2}/panel.png`;
  - `runs/control-sw-s0/panel.png`.

  Each panel shows the run, r328 and the old quarter model.
- Side by side:
  - `runs/compare-ne.png`: control s0-s2, composites s0-s2, r328 and old-ne;
  - `runs/compare-sw.png`: control-sw-s0, r328 and old-sw.
- Audits:
  - `runs/<run>/audit.png`;
  - `baselines/audit/{r328,old-ne}-ne.png`;
  - `baselines/audit/{r328,old-sw}-sw.png`.
- Baselines: `baselines/panel.png`, `baselines/audit/{r328,old-quarters}.png`.
- Composites: `composites.png` (Task 6).

## What remains open

- **The dashed pier-head line is a new over-call.** What is seen:
  - The "dashed" window (p02 rows 770-1010, cols 1930-2170) holds no Terra box, other-pass box, zone or Sol point.
  - The dashes are red in all six runs, and grey in r328, old-ne, all-sheets and the trial, so the cause is not the
    recipe's.
  - One untested hypothesis: dashes inside boxes elsewhere are still taught as text (the run rule's recall on
    dashes is 0.02, so the label rule does not mark them unknown there).
  - It is unknown which control change (the zones or the in-box rule) causes it. M1c rises by +0.027 against
    old-ne, of which the recipe's share is shown above.
- **Half the straight-line ink in boxes is still called text** (M1b in box 0.51-0.59).
  - The label rule leaves that ink unknown, with no loss on it, so nothing pushes the detector off it.
  - The composites were meant to give that push. They did not move it (-0.004).
- **Letters away from lines are lost** against the old models:
  - zoning codes: -0.033 on NE, -0.018 on SW;
  - street names: -0.048 on SW;
  - clean "other" words: -0.05 on NE, -0.17 on SW.

  Which strokes these are has not been looked at word by word. The clean-word guard is dominated by a few large
  "other" words.
- **The adoption bar.** The rule as written requires parity with r328 on M2, which no line-aware detector here met
  and none is likely to.
  Whether to judge against the old quarter models alone is the owner's call. Even then, the clean-word guard fails on
  NE for both recipes.
- **One-seed SW.** SW is one seed, with the noise taken from the NE seeds.
- **The audits** were counted by one eye.
- **Composites, deferred from Task 6:**
  - the realised crossed share is about 0.72 against 0.34-0.38 on real maps;
  - A_SHARE 0.5 is not measured;
  - line pieces are cut on a 256-px grid.

  - the composites' domain check compared unlike sets (known text ink against all text ink);
  - `baselines/r328` is a copy that `copy_today` will not refresh after a rebuild of `solver-text-reader`.

  A composite recipe closer to the real rates is untested.

## Stages left stale by this plan

`solver-text`, `solver-text-scores`, `solver-text-reader`, `solver-boxes`, `solver-text-crops` and `solver-text-read`
list `textnet/data.py` and `textnet/train.py` as code deps, and those files changed (new defaults for the label rule
and composites). Under the defaults the behaviour is byte-identical: the `GUARD_WEIGHTS` digest in
`tests/test_solver_textnet.py` and the pinned-batch tests show it. Nothing was rebuilt.
- A rebuild would retrain on the GPU and cascade downstream.
- `asdict(recipe)` now carries `labels` and `composites`, so a rebuilt `held.json` and `model.pt` would differ in
  bytes even with identical weights, and `held.json` feeds `solver-text-crops`.

The owner's options (none taken):
- (a) re-record without rerunning (the R8 path), with the digest as the evidence;
- (b) rebuild;
- (c) keep these hooks off the branch until a recipe is adopted.

## How to reproduce (from `pipeline/`, $0)

```
uv run python scripts/textlines/variant.py composites --quarter ne --seed N --labels lines-unknown \
    --composites 0.3396 --extra 1990-1996-zoning          # N = 0, 1, 2, one at a time
uv run python scripts/textlines/variant.py control --quarter sw --seed 0 --labels lines-unknown \
    --extra 1990-1996-zoning
uv run python scripts/textlines/decide.py audit ne        # and sw: the baselines' audits on one quarter
uv run python scripts/textlines/decide.py regate-extra control-ne-s0 control-ne-s1 control-ne-s2
uv run python scripts/textlines/decide.py panel ne control-ne-s0 control-ne-s1 control-ne-s2 \
    composites-ne-s0 composites-ne-s1 composites-ne-s2
uv run python scripts/textlines/decide.py panel sw control-sw-s0
uv run python scripts/textlines/decide.py verdict ne --control control --variant composites --eye-ok yes
uv run python scripts/textlines/decide.py verdict sw --control control --noise ne
```

`verdict` writes `runs/verdict-<name>-<q>.{json,md}`. The eye counts of the audits are `decide.EYE_AUDIT`.
