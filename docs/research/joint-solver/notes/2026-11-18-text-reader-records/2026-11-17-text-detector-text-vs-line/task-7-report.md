# Task 7 report: variant 1 (composites) against control 0, the decision as written

## Built
- `pipeline/scripts/textlines/decide.py` (commit f5bc3aa; edited since, uncommitted until the final commit):
  - `verdict(control, variant, baselines, eye_ok, *, quarter, noise_runs, audit_counts, boot, seed) -> dict`: the
    brief's rule plus the controller's m2_clean_words ruling; every condition's numbers; `numbers_say` (the verdict
    with eye_ok True) beside `verdict`.
  - Noise per measure and comparison = max(the noise runs' largest pairwise difference, the paired interval's
    half-width), recomputed inside. `noise_runs` defaults to the control (on SW: the NE control seeds).
  - "mixed" = the variant gains on a line measure (m1a_in_box or m1b_in_box falls, interval below 0) and a condition
    fails; "control" = it gains on neither.
  - Adoptable per arm against r328 and old-<q> on that quarter's blocks: the brief's conditions (`as_written`) and,
    separately, with the clean-word guard (`adoptable`). The audit condition: the arm's estimated missed text pieces
    (mean missed x pooled eye share) at most the baseline's Clopper-Pearson 95% upper bound; None until counted.
  - CLI: `audit <q>` (r328's and old-<q>'s audits on one quarter: the same pool, draw and picture as a run's own),
    `panel <q> RUN...` (side by side), `regate-extra RUN...` (1990-96 gated again so older runs carry
    m2_clean_words; refuses to write if any older measure changes), `verdict <q> ...` (json + md).
  - `EYE_AUDIT`: the eye counts, in git.
- `pipeline/tests/test_textlines_decide.py`: 9 tests on synthetic blocks.npz (the brief's four cases; the clean-word
  ruling; gains nothing -> control; adoptable incl. the pending audit; the audit bound both ways; SW noise from the
  NE seeds).
- TDD: RED = `ModuleNotFoundError: No module named 'decide'`; GREEN 9 passed; full suite 1539 passed, 3 xfailed
  (before any run).

## Smoke: decide reproduces Task 5 (control vs baselines, NE)
vs r328 m1a_in_box -0.5880 [-0.6200, -0.5539]; vs old-ne -0.2253 [-0.2567, -0.1943]; seed noise m1a_in_box 0.0180,
m1b_in_box 0.0834, m2 0.0136, m1c 0.0211, m2_clean_words 0.0082: the same as the Task 5 and 2b reports.

## NE runs (variant = DEFAULT_RECIPE + labels lines-unknown + composites 0.3396; extra 1990-1996-zoning)
- `runs/composites-ne-s{0,1,2}`: one at a time, mps. Minutes per run: load 0.6, bank 0.6-0.7, train 6.4-6.5,
  predict 0.1, score 1.1. `git_head` f5bc3aa, `git_status_src` empty in all three.
- Composites per run: about 8,100 (a) + 8,100 (b) of 48,000 crops (33.8%; P = 0.3396); (a) on lines about 3,450;
  left as is 28-36; words redrawn 3,070-3,240; new-lesson px 1.56-1.58 M.

Share called text (chance >= 0.5) on NE; baselines on NE blocks:

| run | m1a_in_box | m1b_in_box | m2 | m1c | m2_clean_words | m1a |
|---|---|---|---|---|---|---|
| control-ne-s0 | 0.3925 | 0.5102 | 0.9653 | 0.0405 | 0.9271 | 0.0389 |
| control-ne-s1 | 0.4095 | 0.5936 | 0.9756 | 0.0616 | 0.9352 | 0.0538 |
| control-ne-s2 | 0.4105 | 0.5320 | 0.9620 | 0.0545 | 0.9275 | 0.0506 |
| composites-ne-s0 | 0.3992 | 0.5364 | 0.9717 | 0.0402 | 0.9288 | 0.0407 |
| composites-ne-s1 | 0.4197 | 0.5540 | 0.9705 | 0.0412 | 0.9329 | 0.0462 |
| composites-ne-s2 | 0.4123 | 0.5328 | 0.9660 | 0.0472 | 0.9297 | 0.0561 |
| r328 (NE) | 0.9922 | 0.9694 | 0.9994 | 0.5474 | 0.9845 | 0.8110 |
| old-ne | 0.6294 | 0.8177 | 0.9727 | 0.0252 | 0.9565 | 0.0380 |

Variant less control (each arm the mean of 3 seeds; 95% paired interval over 713 blocks; noise = max(control seed
spread, half-width)):

| measure | diff [95%] | noise | role |
|---|---|---|---|
| m1a_in_box | +0.0063 [-0.0060, +0.0184] | 0.0180 | must fall: does not |
| m1b_in_box | -0.0042 [-0.0119, +0.0018] | 0.0834 | a gain if it falls: does not |
| m2 | +0.0017 [+0.0003, +0.0038] | 0.0136 | held (slightly up) |
| m1c | -0.0094 [-0.0147, -0.0039] | 0.0211 | held (falls: better) |
| m2_clean_words | +0.0005 [-0.0009, +0.0010] | 0.0082 | held |
| m1a (diagnostic) | -0.0001 [-0.0031, +0.0032] | 0.0149 | |
| m1b_near (diagnostic) | -0.0058 [-0.0078, -0.0040] | 0.0157 | |
| m1a_in_box_context (diagnostic) | +0.0058 [-0.0002, +0.0128] | 0.0572 | |

Verdict NE: **pending-eye; the numbers say `control`** (the variant gains on neither line measure; no guard fails).
Under the rule the eye cannot make this `variant`: with eye_ok True or False the verdict is `control`.

Adoptable on NE (arm less baseline; noise in brackets after the guard measures):

| arm vs | m1a_in_box | m1b_in_box | m2 (noise) | m1c (noise) | m2_clean_words (noise) | audit | as written | + clean guard |
|---|---|---|---|---|---|---|---|---|
| control vs r328 | -0.588 [-0.620, -0.554] | -0.424 [-0.473, -0.383] | -0.0317 [-0.0454, -0.0207] (0.0136) FAIL | -0.495 (0.141) | -0.0545 (0.0406) FAIL | held | False | False |
| control vs old-ne | -0.225 [-0.257, -0.194] | -0.272 [-0.291, -0.245] | -0.0051 (0.0167) | +0.0270 [+0.0167, +0.0366] (0.0211) FAIL | -0.0266 (0.0213) FAIL | held | False | False |
| variant vs r328 | -0.582 [-0.614, -0.546] | -0.428 [-0.478, -0.386] | -0.0300 [-0.0430, -0.0196] (0.0136) FAIL | -0.505 (0.136) | -0.0540 (0.0400) FAIL | held | False | False |
| variant vs old-ne | -0.219 [-0.247, -0.188] | -0.277 [-0.294, -0.252] | -0.0033 (0.0159) | +0.0176 [+0.0116, +0.0227] (0.0211) held | -0.0261 (0.0206) FAIL | held | True | False |

Neither arm is adoptable on NE as written: both fail m2 against r328, which calls 99.9% of glyph ink (and 97-99% of
line ink) text. The variant passes against old-ne as written; the clean-word guard fails against old-ne for both arms.

## Audits (eye counts by me; "plainly text" = the boxed piece is a letter or digit; digits fused into a line are
borderline and not counted)
Pool on NE: 1,459 letter-sized pieces, the same for every source.

| source | missed | plainly text / drawn | share [CP 95%] | missed text pieces [95%] | borderline tiles |
|---|---|---|---|---|---|
| control s0/s1/s2 | 1279 / 1246 / 1246 (mean 1257) | 1/192 (s0 tile 22, the 0 of "40") | 0.005 [0.000, 0.029] | 6.5 [0.2, 36.0] | s0: 5, 8, 29; s1: 30, 42; s2: 41, 47 |
| composites s0/s1/s2 | 1275 / 1284 / 1248 (mean 1269) | 0/192 | 0.000 [0.000, 0.019] | 0.0 [0.0, 24.1] | s0: 15, 26; s1: 8, 23, 41, 45, 63; s2: 15, 24, 25 |
| r328 (NE) | 449 | 0/64 | 0.000 [0.000, 0.056] | 0.0 [0.0, 25.1] | none |
| old-ne (NE) | 1379 | 1/64 (tile 42, the T of a large "STR...") | 0.016 [0.000, 0.084] | 21.5 [0.5, 115.9] | 9 |

The audit holds for every arm against every baseline. Almost all missed letter-sized pieces outside the boxes are
line fragments, corners and junction stubs.
SW baseline audits written ahead (`baselines/audit/{r328,old-sw}-sw.*`): pool 1,032; missed r328 445 (0/64 text),
old-sw 821 (3/64 text: tiles 25, 39, 63, street-name letters and the T of "...CTS"; borderline 7, 10, 18, 47, 49, 50).

## 1990-96 (NE px of sheets 22-33)
The control's extra folders were gated again (`decide.py regate-extra`) so they carry m2_clean_words; their older
measures came out identical (checked before writing).

| run | m1b_in_box | m1c | m2 | m2_clean_words |
|---|---|---|---|---|
| control-ne-s0 | 0.5253 | 0.0442 | 0.9687 | 0.9542 |
| control-ne-s1 | 0.6044 | 0.0680 | 0.9773 | 0.9569 |
| control-ne-s2 | 0.5432 | 0.0611 | 0.9632 | 0.9534 |
| composites-ne-s0 | 0.5514 | 0.0441 | 0.9731 | 0.9535 |
| composites-ne-s1 | 0.5712 | 0.0468 | 0.9738 | 0.9574 |
| composites-ne-s2 | 0.5539 | 0.0525 | 0.9702 | 0.9535 |

## Pictures (under pipeline/data/solver/experiments/textlines/)
- `runs/composites-ne-s0/panel.png`, `runs/composites-ne-s1/panel.png`, `runs/composites-ne-s2/panel.png`
- `runs/control-ne-s0/panel.png`
- `runs/compare-ne.png`: control s0-s2 and composites s0-s2 side by side, then r328 and old-ne (`decide.py panel`)
- audits: `runs/{control,composites}-ne-s{0,1,2}/audit.png`, `baselines/audit/{r328,old-ne}-ne.png`,
  `baselines/audit/{r328,old-sw}-sw.png`

What I see (not the eye check, which is the controller's):
- The big vertical "RM-2" in rm2a is RED in control-ne-s0's own panel (unchanged since 03:06) and in all six runs.
  The Task 5 note that it is grey does not match this picture.
- Lines through labels are grey in all six (rm2a-c, residential). "UNION NC": control s0 leaves "U", "I", "O" partly
  grey; the other five call it red.
- The dashed pier-head line is red in all six, a little less in composites s1 and s2.
- Small stray red marks differ by seed (e.g. near "M-1" in downtown, composites s1).

## Stop: pending-eye
The verdict needs eye_ok from the controller. Next, after the reply: the NE winner is the control by the numbers, so
only `variant.py control --quarter sw --seed 0 --labels lines-unknown --extra 1990-1996-zoning` runs on SW, then
`decide.py verdict sw --control control --noise ne`.

## After the controller's eye check (eye_ok True; notes verbatim in the results note)
- NE verdict, written with `--eye-ok yes`: **control** (`runs/verdict-composites-ne.{json,md}`).
- **SW:** `runs/control-sw-s0` (with `--extra 1990-1996-zoning`, as on NE). Train 6.35 min, head f5bc3aa,
  `git_status_src` empty. Values: m1a_in_box 0.4138, m1b_in_box 0.5203, m2 0.9690, m1c 0.0350, m2_clean_words
  0.9093.
- **SW adoptable** (NE seed noise; one seed):
  - vs old-sw: m1a_in_box -0.319 [-0.384, -0.255] and m1b_in_box -0.275 [-0.297, -0.214] fall. But m2 -0.0142
    [-0.0274, -0.0082] fails (noise 0.0136), and m2_clean_words -0.0719 [-0.0977, +0.0006] fails (noise 0.0491).
    m1c +0.0014 holds, as does the audit.
  - vs r328: fails m2 (-0.0297, noise 0.0191), m2_clean_words (-0.089, noise 0.060) and the audit. The control's
    3/64 text gives 37.5 missed text pieces, against r328's 0/64 upper bound of 24.9.
  - **Not adoptable** on SW as written.
- Eye counts on SW: control-sw-s0 3/64 (tiles 25, 38, 44: street-name letters; borderline 27, 50, 51); r328-sw
  0/64; old-sw-sw 3/64.
- 1990-96 on SW px: m1b_in_box 0.5061, m1c 0.0387, m2 0.9774, m2_clean_words 0.8994.
- `runs/compare-sw.png`: control-sw-s0, r328, old-sw. In sw_dense and sw_street the control looks like old-sw: lines
  grey, lot numbers red, and the street names inside curved road bands grey in both. r328 calls the lines red.
- **Post hoc (in the note, marked as such; the verdicts are unchanged):**
  - The r328 M2 condition cannot pass for a line-aware detector: it needs M2 >= 0.986 on NE, and old-ne has 0.973.
  - Against the old models alone: NE control fails m1c (+0.027 vs 0.021) and clean words (-0.027 vs 0.021); NE
    variant fails clean words (-0.026 vs 0.021) only; SW control fails m2 by a hair and clean words.
  - The clean-word loss is mostly "other" words: NE -0.050 over 32 blocks, SW -0.171 over 5 blocks, 42% of SW
    clean-word ink. Clean codes and numbers lose under 1%.
  - Code letters off lines (m2_by_kind): NE -0.033 [-0.039, -0.028], SW -0.018. SW street names -0.048.
- The results note's owner summary: the line gain against the old models with intervals; the costs (clean-word and
  code letters, ink outside boxes on NE, the dashed line); composites add nothing; nothing adoptable, so Task 8 does
  not run; what remains open.

## Commits
- f5bc3aa: decide.py and its tests (the rule, the verdict).
- d489e38: decide.py's final version (audits per quarter, the panel, regate-extra, markdown, EYE_AUDIT) and
  `docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md`.
- Full `uv run pytest -q` before each: 1539 passed, 3 xfailed (no training running). `pgrep` empty before each.

## Concerns
- **The audits were counted by one eye (mine).** They are subjective at the margin; the borderline tiles are listed.
  The SW audit failure against r328 hinges on 3 against 0 in 64.
- **The SW verdict is one seed.** The m2 failure against old-sw is -0.0142 against a noise of 0.0136: a near-tie.
- **The clean-word guard is dominated by a few large "other" words** (5 blocks on SW). It says little about code
  letters, and which words they are has not been looked at.
- **Interpretation choices in decide.py:**
  - "mixed" = a line gain plus any failed condition;
  - "control" = no line gain;
  - the audit condition = the arm's missed-text estimate at most the baseline's Clopper-Pearson upper bound;
  - adoptable needs every baseline;
  - the clean-word guard is reported both in and out of adoptable (`as_written`).
- **Data and side runs:**
  - The control's 1990-96 `extra/` folders were rewritten with m2_clean_words; the older measures were checked
    identical first.
  - The SW control run added `--extra 1990-1996-zoning` beyond the controller's command line, for the 1990-96
    numbers.
- **Stale reports:** the Task 5 report's RM-2 "grey" claim is wrong (corrected in the note). The Task 5 report
  itself is not edited.
