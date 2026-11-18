# Task 3 report: the posterior

Status: DONE_WITH_CONCERNS (listed at the end). Branch `solver`, commits cfd5253 (the reader) and d80cc7d (docstring:
pruning and the GPU). CPU only for the results; one GPU
attempt (below). No LLM, no `.env`, no stage run. Only my files staged (`dvc.lock` left as it was).

## What was built

`pipeline/src/sfzmaps/solver/codereader/posterior.py` (new), small changes in `glyphs.py` and `__init__.py`, 12 new
tests in `pipeline/tests/test_codereader.py`.

- **`score_label(label, font, noise, prior, constants) -> dict`**: the posterior over every code of `prior`, `"none"`
  and `"other"` (sums to 1). Keyword arguments go to `evidence` (`observed=`, `rules=`, `device=`).
- **`evidence(label, font, noise, prior, temperatures, ...) -> Evidence`** and **`posterior_of(ev, prior,
  constants)`**: the expensive part once (log E of every printed form and of "other", at any number of temperatures
  in one pass), the masses applied after. Task 4 can fit the two masses without rescoring, and T over a grid
  computed in one pass.
- **`Constants(temperature=None, none_mass=0.02, other_mass=0.03)`**: None = the prior's measured temperature. The two
  masses are placeholders until Task 4's cross-fit.
- **`code_prior(training_labels, forms=None) -> dict`**: one count per training label (its `glyphs.consensus`
  string's code), add-1/2 over every code of the forms. P 0.229, RH-2 0.146, RM-1 0.102, RH-1 0.100, RH-3 0.080, ...
- **`printed_forms(code_list)`** (carried item 2): the code list, `codes.ALIAS`/`labels.ALIAS` spellings, and
  "<form> NCD" for every named district, derived from the table's names only (`named_district`: a word of 4+ letters
  with a vowel, so POLK and UNION yes, CRNC/CCB/SLI no). 95 forms over 57 codes.
- **`char_bigram(training_labels, chars, forms)`**: "other"'s bigram, p(next | previous character's kind: START,
  letter, digit, space, each punctuation mark), from the training labels whose own string is no code (25 texts: SUD
  names, MARKET, NCD, ...), add-1/2, no space first/last/twice. Kinds pool the previous character because those
  texts are few; it also lets the Viterbi pass maximise per kind (8 rows), not per character.
- **`PosePrior`** and **`measure_pose_prior(training_labels, font, noise)`**: everything below measured on the 959
  training labels (109 s); **`Prior`** (codes, forms, bigram, chars, pose) with `save`/`load` (JSON) and
  **`Prior.learn(training_labels, font, noise)`** for Task 5's stage.
- **Carried items**: (1) half turn and quarter turns in the pose grid with measured rates; (2) NCD forms from names;
  (3) measured position priors per source and size, other text explaining foreign ink; (4) `learn_font`,
  `code_prior` and `char_bigram` raise on a held-out label; (5) `consensus` docstring fixed (alphabetical tie);
  the EM trace keeps each label's weight, and the synthetic EM test now asserts the two misplaced labels' weights
  are < 0.05 and every other > 0.9; `Font.scale_sd` persisted (save/load, old fonts load with 0). (6) I did not move
  the pose search out of `glyphs.py`: the posterior has its own engine and only calls `glyphs._patch`, `_ab`,
  `_axes`, `_linear`, `lay`, `_warp`, `align`, `search_for`.

## The model (the module docstring has it in full)

p(h | ink) oc p(h) E_h, E_h = p(ink | h) / p(ink | none), E_none = 1.

- **One engine for every hypothesis (Document Image Decoding / Ocular).** Per scale and angle, each character's
  log-likelihood ratio against the bare background at every centre is one FFT correlation of its rendered patch
  (`glyphs`' noise, line px unobserved) with the crop. A string's ratio is the sum of its characters' at their pens.
  On Task 2's training fits the per-character sums equal the whole string's ratio to 0.3% (787 = 787; 1459 vs 1462).
  Codes are rigid layouts of each printed form. "other" is the best string from each cell by a max-product (Viterbi)
  pass over the bigram.
- **Each character shifted along the line** (`_jitter`), its shift Gaussian with Task 2's gap spread / sqrt 2, summed
  over (log-sum, so a well-placed character pays only its Occam factor). Without it the rigid layout missed real
  spacing ("1(" and ")" after "D") and "other" won by spelling "R.H-1( S" for RH-1(D).
- **Placement prior: a measured core + tail mixture about the print** (`PosePrior`), cells where even the tail is
  below e^-12 of its peak not scored. With Terra's box: the ink's two ends at the word's ends (a centre prior times a
  length likelihood) and the line's middle at the word's middle; the cap height log-normal in the word height. Without
  a box: the ink's centre about the point, along and across the line, per source (Sol's point; the turned pass's box
  centre); cap height by the map's spread. **No uniform anywhere-in-the-crop component**: in the measurement it holds
  the strings matched on other ink (the size bias), not where labels are.
- **Slanted boxes**: Terra's axis box gives a word rectangle far thinner than the letters (training median 14 px
  against 27 for quarter-turned words; 5% at 3 px), so the box-relative scale grid could not reach the letters
  (#915: a 5.8 px rectangle). A slanted box keeps its position; its letters' size takes the map's spread.
- **Angles**: Terra's angle and its half turn (rate per Terra direction) or every quarter turn without a box (rate per
  source), each with 5 residuals (-6..+6 degrees, measured rates).
- **Other text in the crop (carried item 3).** The ink a hypothesis claims (its characters' cells: advance x
  lettering height, leaned) could instead be other text, at the measured density of text in the crops, explaining it
  as well as the label's own print is explained (each character's measured gain per ink px x the label's quality,
  q = the best code's ratio over its claimed ink at the core placement). Terra's word counts as one word of ink:
      E_h = L_h (1 + rho F(word \ h)) / [(1 + rho F(word)) (1 + rho F(h \ word))]
  so a code that stops short of its word or shrinks inside it pays for the leftover as an extra word; a code that
  matches a neighbour outside the word gets about what the label gets at its own place, and the placement prior
  decides. Without a box the word term drops out (see concerns). `PosePrior.foreign = 0` turns it all off.
- **Temperature**: every px ratio / T before anything is summed. Default = the design effect of the training fits'
  residuals within one character (px pairs up to half the median cap height, 15 px, apart): **32.7**.
- **Speed**: a coarse pass (every other scale, each base angle's likeliest residual) gives the label's quality and
  lets the full pass skip a scale and base angle whose coarse best is e^-40 below the label's best (a numerical
  tolerance: on 5 labels the posteriors moved by at most 8e-11).

## Measured on the 959 training labels (`prior.json` in scratch)

Broad-prior fits of each training string (`glyphs.align`, the first-round prior), then mixtures:

| what | core | tail | notes |
|---|---|---|---|
| ink's left end - word's left end (word heights) | mean +0.023, sd 0.050 | sd 0.239 | quarter-turned boxes (815) |
| ink's right end - word's right end | mean -0.032, sd 0.065 | sd 0.325 | |
| line's middle - word's middle | mean -0.007, sd 0.028 | sd 0.079 | |
| box weights core / tail / uniform | 0.679 / 0.114 / 0.206 | | the uniform 21% = strings matched on other ink, left out |
| log(cap / word height) | mean +0.014, sd 0.134 | | (Font.scale_sd from the EM: kept too) |
| Sol's point vs Terra's word centre, along / across | sd 0.096 / 0.064 | sd 0.383 / 0.176 | 734 prints both read; tail share 0.33 |
| turned box centre vs Terra's word centre | sd 0.035 / 0.036 | sd 0.301 / 0.139 | 838 prints; tail share 0.26 |
| half turn per Terra direction | top-to-bottom 0.50, bottom-to-top 0.023, left-to-right 0.0025, slanted 0.22 | | slanted from censored fits (concern) |
| quarter turns without a box (0/90/180/270) | Sol 0.68/0.11/0.11/0.11; turned 0.60/0.29/0.06/0.06 | | mostly Jeffreys counts: 54 point labels |
| residual angle -6/-3/0/+3/+6 | 0.024/0.094/0.448/0.345/0.089 | | |
| map's cap height | log mean 3.381 (29.4 px), sd 0.284 | | |
| other text density | 1.22e-4 per px^2 | | 5,394 Terra boxes of every kind in the crops |
| character gain (ratio per ink px of its cell) | 0.79 (")") .. 1.89 ("P") | | rare ones shrunk to the mean (2 instances) |
| design effect, reach 1/2/4/8/15/16/32 px, whole crop | 5.3/9.8/14.9/19.4/32.7/34.6/59.7, 93 | | 32.7 kept |

## Comparisons on training labels (the same measure throughout)

Truth = the training string's code (`glyphs.consensus`: LLM reads, not the owner's answers; it is itself wrong on
some labels, e.g. #670 is printed RH-1(S), read RH-3). The same 120 labels (every 8th training label), top-1 and
mean -log p(true):

| variant | T=15 | T=30 | T=45 | T=90 |
|---|---|---|---|---|
| per-hypothesis other-text term (cells only) | 103, 1.72 | 106, 1.33 | 106, 1.42 | 93, 1.84 |
| no other-text term (bare background) | 93, 4.36 | 108, 1.42 | 106, 1.37 | 93, 1.81 |
| **word-level other-text term** | 105, 2.10 | **110, 1.28** | 104, 1.45 | 93, 2.22 |

Kept: the word-level term. With the final prior (quarter-turned measurement, slanted boxes on the map's scales,
T = 32.7): **110/120, mean -log p 0.310, median 0.007**. Every reading with top p >= 0.8 agrees with the string (85
of 85); p 0.5-0.8: 21 labels, mean p 0.65, agree 0.90 (under-confident: Task 4's fit should lower T); p < 0.5: 14,
mean 0.35, agree 0.43.

**Line rule** (carried from Tasks 1-2: choose by training likelihood), final prior, T = 32.7, same 120 labels:

| rule | top-1 | mean -log p |
|---|---|---|
| **run** | **110** | **0.310** |
| long + run | 107 | 0.366 |
| long | 100 | 0.858 |
| width | 94 | 0.877 |

`RULES = "run"` (the font's too).

Earlier attempts I measured and dropped (each documented in the module or here):
- a uniform anywhere-in-the-crop placement component: lets codes wander onto bigger ink, and makes every label score
  the whole crop (51 s a label);
- explaining away normalised per cell by the sum over hypotheses there: a wrong code saturates at cells where the
  true one is misaligned (RH-1 0.50 / RM-1 0.25 on clean synthetic ink);
- one quality for the whole label: the poorly learned "(", "D", ")" made RH-1 beat RH-1(D); per-character gains fix
  it;
- claimed ink = the strokes' footprint instead of the cells: a code at a smaller scale escaped the charge for the
  letters' tops and bottoms (RM-1 at 0.689 vs RH-1 at 0.755);
- sum-product "other" with a per-cell combination of two sums: broke when strings of very different ink mixed
  ("other" 0.94 on RH-1); replaced by Viterbi with each path's own ink.

## Timing (every label, CPU)

Every one of the 1,905 labels scored once (`prior.json`, T = 32.7, rules "run"), 8 worker processes x 1 thread:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` in the
environment before numpy/torch load, and in each worker `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`,
`cv2.setNumThreads(1)` (torch's FFT runs on torch's one thread). Load average with my run alone: 10-14 (it rose to
80-230 twice while another agent's `sfzmaps solver solve` and macOS's `mediaanalysisd` ran; not mine). The runner
(`full_run.py` in scratch) appends each label's evidence as it finishes, so it resumes after a stop, and it stops
taking labels while `sfzmaps build` runs (it did: 17 labels, then resumed after the controller's build).

| label kind | labels | per label (one worker) | work |
|---|---|---|---|
| quarter-turned box | 1,620 | median 12.5 s, max 31 s | 21,086 s |
| slanted box | 149 | median 24.0 s, max 36 s | 3,614 s |
| point (no Terra box: 4 quarter turns x 29 scales) | 136 | median 170 s, max 319 s | 23,627 s |
| every label | 1,905 | | 48,327 worker-s |

Wall: 5,573 s (1 h 33 min) for 1,888 labels after the resume, plus 17 before it. With 8 workers each label runs ~35%
slower than alone (point labels 170 s vs 122 s, memory bandwidth). The point labels are 7% of the labels and half the
work. Readings (no answers looked at): a code on top for 1,875, "other" 21, "none" 9; median top probability 0.982.
Records: `full_cpu8.records` in scratch (key, kind, point?, Evidence, seconds), for Task 4.

The first attempt (10 workers, thread pools not pinned) put the load at 110-140; the controller stopped it at 500
labels.

## Pictures (scratch)

- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task3/misses_final.png`
  - **What it shows:** the 10 of the 120 training labels the final reader reads differently from their training
    string: the crop, Terra's word (red box) or the point (red cross), the string, the reading and both
    probabilities.
  - **What to look at:** all are hard: hand-lettering variation (C-3-S read C-3-G 0.32), small or slanted words crossed
    by heavy lines (RH-2/RM-2 swaps under a line, 0.55-0.58), and two Sol-only points beside bigger text (P read RM-1,
    NC-1 read P). Every miss is at p < 0.6. No action needed; it is the reader's error profile for Task 4.
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task3/misses_word120.png`
  - the same for the run before the slanted-box fix and the final prior (T = 30); #915 there went to a long NCD name.
- `crop_527.png` in the same folder: an "NC-1" printed on two lines ("NC" over "-1"), which the one-line model can't
  render (concern 4).

## Tests

`pipeline/tests/test_codereader.py`, 12 new (24 in the file):
- `test_posterior_sums_to_one`; Task 4's constants change the masses and the sum stays 1.
- `test_line_through_h_does_not_make_it_m` (Review Focus 2): RH-1 crossed by a 6 px bar through the H, the bar's px
  unobserved, scores **exactly** like the clean label with those px unread (the line's ink neither helps nor hurts),
  RH-1 on top (> 0.8); the same bar read as ink pulls toward RM-1 (odds x3 or more).
- `test_sideways_code_reads` (Review Focus 3): RM-2 printed bottom-to-top at a Sol point reads RM-2 (> 0.9), and
  RH-2 that Terra read top-to-bottom (crop upside down) reads RH-2 (> 0.9).
- `test_lines_only_window_prefers_none` (Review Focus 4): two heavy masked lines and background specks: none > 0.9.
- `test_slanted_box_reads_whatever_its_rectangle_height`.
- `test_code_is_not_chosen_for_covering_more_ink`: a P at Terra's word with RH-1(D) on the next line and RM-2 on the
  same line reads P (> 0.9).
- `test_text_that_is_no_code_reads_other` ("SUD").
- `test_printed_variant_reads_as_its_code` (Review Focus 5): "NOE VALLEY" ink reads 24TH-NOE VALLEY (> 0.9).
- `test_printed_forms_come_from_the_code_names`, `test_code_prior_and_bigram_count_training_strings`,
  `test_mixture_fit_separates_core_tail_and_stray`, `test_learning_refuses_held_out_labels`.
- The tests' synthetic ink is drawn px by px independently, so their prior uses T = 5 (the map's is measured).

Full suite (`cd pipeline && uv run pytest -q`): 1,119 passed (the count includes other agents' changes on the
branch), 75 s, at cfd5253; after d80cc7d (docstring only) `tests/test_codereader.py`: 24 passed. `ruff check --line-length 120` clean on posterior.py and the test file (glyphs.py's existing B023
findings in `learn_font` are Task 2's).

## Concerns

1. **Speed is not "minutes".** Every label: 48,327 worker-seconds, 1 h 33 min wall on 8 workers x 1 thread (see
   Timing); half of it is the 136 point labels (170 s each: 4 base angles x 29 scales). Threads scale poorly (16.6 s at 1 thread,
   ~10-13 s at 10), so the run uses processes. The GPU (mps, one exclusive turn) was **not** faster: the first label
   ran 14 minutes at 100% GPU without finishing (it sat in a Metal `waitUntilCompleted`; probably the FFTs at
   non-power-of-2 sizes), so I stopped it and released the lock. FFTs are ~1/3 of the time; most of the rest is
   per-form sums and log-sum-exps. Task 4 needs a temperature grid: `evidence(..., temperatures=grid)` does it in one
   pass, but the per-temperature work (shifts, codes, Viterbi) repeats.
2. **Training-string agreement is 110/120 (92%)**, below the LLM readers. The misses are the hard labels (slanted,
   line-crossed, small, hand-lettering variants, Sol-only points beside bigger text), all at p < 0.6. Judged on the
   owner's answers in Task 4.
3. **The other-text term is an approximation.** It compares each hypothesis's claimed ink with other text claiming
   the same ink, not a decoding of the crop's text. Terra's word supplies the word extent; **point labels have no word
   term**, so a code that stops short of its word there is not charged for the rest. The exact model (a forward-
   backward segmentation of the line into words) is the next step if Task 4 shows point labels need it.
4. **Two-line labels** (e.g. "NC" over "-1", #527; big NCD names) are not modelled: one line per hypothesis.
5. **Slanted boxes**: their half-turn rate (0.22) comes from fits made on the box-relative grid, which can't reach
   their letters (censored); their placement spreads are the quarter-turned ones, in the map's word height.
6. **The temperature**: 32.7 is the design effect within one character; over the whole crop it is 93 (a word's own
   misfit is shared by every code, so it cancels between them). On the 120 labels the reader is under-confident at
   32.7. Task 4 fits it.
7. **In-sample**: the pose prior, character gains, bigram and temperature are measured on the same training labels the
   font was learned on. None of it touches the owner's answers or their blocks (`_guard` refuses held-out labels).
8. **The two masses** (none 0.02, other 0.03) are placeholders.
9. Point labels' quarter-turn rates are mostly Jeffreys counts (54 training point labels, many fitted off their
   point), and the "other" placement for point labels judges the best string's own end, not every end.

## Fix round 1

Commit df61853 (only my files: `posterior.py`, `glyphs.py`, `labels.py`, `tests/test_codereader.py`). Everything
below was measured on training labels and synthetic cases only. No owner's answer was read.

**Tests:** the full suite passed, 1,124 tests in 95 s on 4 workers (`tests/test_codereader.py`: 29). ruff
(line length 120) is clean on posterior.py, labels.py and the test file. glyphs.py has the same 14 findings as
before my change.

### 1. Critical: pruning moved posteriors. ADDRESSED (pruning dropped)

- `PRUNE_LOG`, the coarse scores and `chunks(keep=)` are gone. The full pass now scores every scale, angle and cell
  within the placement prior's reach. Pass 1 stays only to give the label's print quality q, a statistic used as
  defined at each temperature.
- **Why not an honest bound:** a true bound on a chunk needs that chunk's correlation maps (the FFT part) and a
  per-form upper bound. The cheap bound (each character's best anywhere in the chunk) is loose: a wrong quarter turn
  still scores 50-70% of the right one in per-character maxima. It could prune at T=5 but not at T=32.7, so nothing
  would be saved over a grid that includes 32.7.
- **Test:** `test_temperature_grid_gives_each_temperature_as_alone`. On a box label, and on a point label whose
  text sits off its point, a (5, 30) grid's log E of every form and of "other" equals each temperature run alone
  (to 1e-9).
- **Cost** (one thread, one temperature; the same 4 box labels and 1 point label, run one after another on a shared
  machine):

  | reader | 4 box labels | point label #161 |
  |---|---|---|
  | committed (pruned) | 67 s | 173 s |
  | committed, unpruned | 135 s | 174 s |
  | fixed, no end-break term | 130 s | 182 s |
  | fixed | 143 s | 184 s |

  Dropping the pruning about doubles box labels. It costs nothing on that point label: at T=32.7 the coarse margin
  never pruned it. The end-break term adds about 1% on point labels and nothing on boxes (the 130 vs 143 s gap is
  machine noise).

### 2. Critical: point labels had no extent term. PARTLY addressed

**The model.** Without a box, each end of a hypothesis (code or "other") multiplies E_h by
P(a word ends | the ink density past that end) / P(end) (`end_breaks`, `end_factor`):
- The density is the observed ink in a window of the lettering band past the end, leaned like the glyphs.
- **Measured** on the 959 training fits, the same quarter-turned boxes the gains use: each string's own two ends
  (word ends) against every boundary between its characters (inside a word; boundaries at a space left out).
  5,198 samples, 1,294 ends (weighted), base rate 0.288.
- **Fitted** by isotonic regression (non-increasing in the density), one base-rate sample added per step.
- **Window:** skip and width chosen from 12 by two-fold log loss. Kept: skip 0.1, width 0.9 cap heights
  (loss 0.253; base entropy 0.600; skip 0 width 0.4: 0.325).
  - A logistic regression on all windows' densities did no better (0.286 against 0.289 in a check with flat
    weights).
- **The table:**

  | density | P(end) | factor (nats) |
  |---|---|---|
  | 0 | 0.99 | +1.23 |
  | 0.06 | 0.67 | +0.84 |
  | 0.1 | 0.29 | 0 |
  | 0.2 | 0.05 | -1.69 |
  | 0.3+ | 0.03 | -2.15 to -2.31 |

  At most about 3.5 nats separate a clean end from one with a letter beside it. About 3% of true ends have dense ink
  just past them: neighbours, and the fit's rigid layout off the print's spacing. Codes are scored with the same
  rigid layouts, so that noise is real for the reader too.
- **Applied:** codes get both ends' factors. The best "other" string takes its last end's factor inside the Viterbi
  pass (its path choice) and its first end's at its start.
- **Tests:**
  - the tests' prior measures its own gains and table on 60 synthetic training labels (`_measured`; four in ten
    have a neighbouring word a word gap away);
  - `test_end_breaks_measure_where_words_end`;
  - `test_off_centre_point_reads_the_full_code`.

**Off-centre RH-1(D)** (the reviewer's setup; `offset_try.py`). The point is moved along the line, in word heights;
"before" is toward "RH", "after" toward "(D)":

| point | committed | fixed, the tests' table | fixed, the map's table |
|---|---|---|---|
| 0 | RH-1(D) 0.974 | RH-1(D) 0.999 | RH-1(D) 0.999 |
| 0.25 before | RH-1 0.574 | RH-1(D) 0.960 | RH-1(D) 0.948 |
| 0.5 before | RH-1 0.912 | RH-1(D) 0.672 | RH-1(D) 0.703 |
| 0.75 before | RH-1 0.981 | other 0.666 | RH-1 0.725 |
| 1.0 before | RH-1 0.999 | RH-1 0.901 | RH-1 0.994 |
| 0.25 after | - | RH-1(D) 0.996 | RH-1(D) 0.998 |
| 0.5 after | RH-1(D) 0.902 | RH-1(D) 0.990 | RH-1(D) 0.996 |
| 0.75 after | - | RH-1(D) 0.867 | RH-1(D) 0.991 |
| 1.0 after | other 0.854 | RH-1(D) 0.709 | RH-1(D) 0.961 |

- **What reads now:** the full code at 0.25 and 0.5 on both sides, and up to 1.0 after. The test asserts exactly
  those six offsets.
- **What still fails:** 0.75 and 1.0 before the middle still read the prefix RH-1, which is centred nearer the
  point. There the measured end evidence (one end, about 3.3 nats) loses to the measured placement prior. At
  0.75-1 word heights the full code's centre is 2-2.6 tail spreads off (tests: tail sd 0.4, share 0.3; Sol: 0.38,
  0.34), while RH-1's centre is about 0.4 word heights off. On the map the code prior (RH-1 0.100, RH-1(D) 0.028)
  leans the same way.
- I did not find a statistic that separates ends more strongly from the training fits. Making the test pass by
  hand-setting a stronger table would be tuning to the test.
- **RM-2** (the reviewer's other case: "other" 0.608 at 6 px before): now RM-2 0.982. At 0.75 and 1.0 on either
  side: RM-2 0.95-0.998. One offset, 0.5 word heights after, reads "other" 0.710.

**Real point labels:** "other" is 13 of 136 (9.6%), as before (12 the same labels). The three training ones are
right (SUD names). On the 54 training point labels, agreement with the training string is 24 (committed: 26). See
"Full run".

### 3. Important: printed forms weighted equally. ADDRESSED

- `form_prior` gives p(form | code) from the training strings' counts, with a half count added to each form
  (`Prior.form_weights`, saved with the prior). `posterior_of` weights a code's forms by it, renormalised over the
  forms the font can lay out. With no weights, a code's forms count alike.
- Measured: RH-2 0.989 / RH2 0.011; HAYES-GOUGH NCD 0.375; WEST PORTAL NCD 0.625; BROADWAY NCD 0.75.
- **The RH-2/RM-2 lean, as the reviewer measured:** on the committed run's evidence, agreement goes 110 → 108; on
  the fixed reader's, 109 → 107. Both times it is the same two labels:
  - RM-2 #635: 0.66 → RH-2 0.50 (RM-2 0.497);
  - RM-2 #990: 0.57 → RH-2 0.485 (RM-2 0.433).

  RH-2 no longer splits its evidence with RH2, so it gains log 2 (about 0.69 nats) against every other code. Both
  labels were within that of RH-2. Not tuned away.

### 4. Important: speed. ADDRESSED (measured)

The split by fix is in finding 1's table. The full run's worker-seconds by label kind are in "Full run" below.

Temperatures cost a lot. The per-temperature work (shifts, form sums, Viterbi) repeats for each one: box #788 took
20 s at one temperature and 54 s at five; point #161 took 162 s and 515 s. That is why the full run gives the
held-out labels Task 4's grid and the others only the default.

### Minor findings

5. **REACH_LOG. ADDRESSED.** 12 against 20 on the nine off-centre RH-1(D) points: the top posteriors are equal to 3
   decimals (largest change 0.001). 20 is 2.7 times slower (5.2 → 14 s). Kept 12.
6. **"Other"'s path chosen without its end weight. ADDRESSED.**
   - The Viterbi pass takes an end weight per (t, line, last character):
     - with Terra's word, the two components' end weights mixed by their rows' weights. It only chooses the path:
       it is taken off after, and each component's exact weight is applied at the end the path carries;
     - without a box, the end factor of finding 2, exactly.
   - **Effect on box labels:** "other"'s log E rose 15-30 nats on the labels I checked (#1414 -9.9 → 20.4; #635
     -17.7 → -3.7; #990 -15.8 → -1.0). Its best string had been ending off Terra's word and paying the end weight.
   - One training label now reads "other": #1414, printed M-2, "other" 0.975. "Other" can spell a code plus a
     little, and M-2's prior is small. "Other"'s mass is Task 4's to fit.
7. **Determinism. ADDRESSED.** `evidence` calls `settle_torch()` (deterministic algorithms, THREADS = 1 thread) and
   takes `gpu.gpu_turn(True, device)`, a no-op on the CPU.
8. **Two size spreads. ADDRESSED.** `Font.scale_sd` is removed (save writes 4 meta values; load reads the first 4,
   so older fonts load). `PosePrior.ratio` is the one size spread, measured on the scoring's own population. The
   font on disk had scale_sd 0.0 anyway.
9. **Tests. ADDRESSED.**
   - `test_learning_refuses_held_out_labels`, parametrised held_out / held_block: it covers learn_font,
     code_prior, form_prior, char_bigram, measure_pose_prior and Prior.learn.
   - Block-mates: `make_labels` marks `Label.held_block` (a held-out block, or no block). `labels.refuse_held`
     (used by every learner) refuses them. `test_training_excludes_answered_labels_and_their_blocks` checks
     labels 1, 3 and 6 are marked and refused.
   - `test_text_that_is_no_code_reads_other` now asserts p(other) > 0.9.
   - New tests: the grid test, the off-centre test, the end-break test and
     `test_posterior_weighs_a_codes_forms_by_how_often_they_are_printed`.

### Agreement on the 120 training labels (training strings, not answers)

Committed reader at T=32.7: 110/120, top-3 116, mean -log p 0.310. Fixed reader (`cmp_fix1.pkl`):

| T | top-1 | top-3 | mean -log p |
|---|---|---|---|
| 16 | 99 | 116 | 1.646 |
| 23 | 104 | 116 | 0.642 |
| **32.7** | **107** | **116** | **0.358** |
| 46 | 105 | 115 | 0.430 |

With a code's forms counted alike, T=32.7 gives 109/120 and 0.356.

The three new misses against the committed reader are #635 and #990 (finding 3) and #1414 (finding 6). The other
ten misses are the committed reader's.

Reliability at T=32.7:

| top p | labels | mean p | agree |
|---|---|---|---|
| < 0.5 | 17 | 0.37 | 0.35 |
| 0.5-0.8 | 19 | 0.65 | 0.95 |
| 0.8-0.95 | 16 | 0.90 | 1.00 |
| 0.95-0.99 | 11 | 0.98 | 0.91 (#1414) |
| ≥ 0.99 | 57 | 0.998 | 1.00 |

### Full run

Fixed reader (df61853), `prior3.json`. Records, one per label (key, kind, point?, Evidence, seconds), in
`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task3/`:
- `full_fix1held.records`: the 731 held-out labels at Task 4's grid, T = 16, 23, 32.7, 46;
- `full_fix1rest.records`: the other 1,174 labels at T = 32.7 only. Each extra temperature repeats most of the
  work, and Task 4 needs the grid only on the answers.

**Settings.**
- 8 worker processes x 1 thread. OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, MKL_NUM_THREADS
  and NUMEXPR_NUM_THREADS are all 1. In each worker: torch.set_num_threads(1), set_num_interop_threads(1),
  cv2.setNumThreads(1), and `evidence`'s own `settle_torch`.
- The runner (`full_run.py`) pauses while `sfzmaps build` runs and resumes after it (no build ran).
- **Load:** 9-14 for the whole run (the other agents were light).
  - Once, at 02:03, I added a second runner of 4 workers. The load went to 155 within a minute and I stopped it.
  - With more than about 10 busy processes on the 10 cores, the load average climbs into the hundreds: every app's
    waking threads queue.
  - An earlier 8-worker comparison beside another agent's 6 solver processes also hit 256, and I stopped it.

| run | kind | labels | median | mean | worker-s |
|---|---|---|---|---|---|
| held out, 4 T | box | 625 | 36.0 s | 41.8 s | 26,122 |
| | slanted box | 43 | 83.9 s | 84.8 s | 3,646 |
| | point | 63 | 607 s | 613 s | 38,586 |
| rest, 1 T | box | 995 | 14.8 s | 16.7 s | 16,621 |
| | slanted box | 106 | 34.5 s | 35.2 s | 3,726 |
| | point | 73 | 181 s | 194 s | 14,151 |

**Wall:**
- held out: 8,620 s (2 h 24 min);
- the rest: 4,340 s (1 h 12 min);
- 3 h 36 min and 102,852 worker-seconds in all.

The committed run took 48,327 worker-seconds at one temperature, pruned. One temperature, unpruned, is about 1.4x
that.

**Readings at the default constants** (no answers looked at):

| | labels | a code | "other" | "none" | median top p | committed reader |
|---|---|---|---|---|---|---|
| point labels | 136 | 123 | 13 (9.6%) | 0 | 0.63 | "other" 13 (9.6%), 12 the same labels |
| box labels | 1,769 | 1,749 | 16 (0.9%) | 4 | 0.99 | "other" 8 (0.5%) |

On the held-out labels alone, 9 of 63 point labels read "other" (the committed reader: 8).

**What the real point labels say** (training labels only):
- The 3 training point labels that read "other" (#171, #490, #491) are SUD names: "NORTH OF MARKET RES. SUD",
  "VAN NESS S.U.D.". Their training string is other text, so "other" is right there. A 9.6% "other" rate is not in
  itself the extent problem.
- **Point labels are read badly overall.** On the 54 training point labels the top reading agrees with the training
  string 24 times (the committed reader: 26). On the 905 training box labels it agrees 816 times (committed: 818).
- **Point-label misses are mostly unsure** (top p 0.2-0.4). Four are confident:
  - #1875, string P, read RH-2 0.72;
  - #1880, string P, read RM-1 0.95;
  - #1899, string P, read RH-1 0.99;
  - #1903, string P, read RH-1 0.98.

  All four are the turned pass's points (kind 3).
- That is a larger problem than the extent term. I have not chased it in this round: Task 4 will show it on the
  answers.

### Concerns after fix round 1

1. **Point labels.**
   - **Off-centre, partly fixed:** a prefix code nearer the point still wins at 0.75-1 word heights. The measured
     end evidence (at most ~3.5 nats an end) is the limit.
   - **On the map they read badly:** 24 of 54 training point labels agree with their string (box labels 816/905).
     Four of the misses are confident: turned-pass points read RH-1/RM-1/RH-2 where the string is P.
   - The real "other" rate did not move (13/136), but the training ones are real SUD names.
   - Next step if Task 4 confirms it: a forward-backward segmentation of the line into words, with the measured gap
     and letter-spacing distributions, and a look at what the turned pass's points sit on.
2. **"Other" is stronger on box labels now** that it chooses its end. It can spell a code with a little more, which
   costs #1414 (M-2 → other 0.975). "Other" should exclude code strings. The other mass (Task 4) will set how often
   this bites.
3. **Agreement on the 120 dropped from 110 to 107:** two are from p(form | code) (the RH-2 lean, finding 3), one
   from "other" (concern 2).
4. **Speed:** box labels cost twice what they did unpruned. Each temperature repeats most of the work. A grid of
   four costs about 2.7× (boxes) to 3.2× (points) one temperature.
5. The earlier concerns still stand: two-line labels; the censored slanted half-turn; in-sample measurement; the
   none/other masses being placeholders.

## Fix round 2

Commit c626e46 (only `posterior.py` and `tests/test_codereader.py`). Training labels and synthetic cases only; no
owner's answer read.

**The gate fails by one box label, so there was no rescore** (as instructed).

| measure (training labels whose string is a code; default constants) | committed | fix round 1 | **round 2** | round 2, a code's forms alike |
|---|---|---|---|---|
| box labels top-1 (of 885; gate ≥ 815) | 815 | 811 | **814** | 815 |
| box labels with p(other) > 0.5 (gate ≤ 4) | 4 | 8 | **4** | 4 |
| box mean -log p(code) | 0.399 | 0.429 | **0.395** | 0.399 |
| point labels top-1 (of 50; gate ≥ 23) | 23 | 21 | **24** | 23 |
| point mean -log p(code) | 2.265 | 2.551 | **2.263** | 2.265 |

- **The one box label short is p(form | code)** (Important 3, kept). RM-2 #388, #635 and #990 go to RH-2; #1165
  (RM-2) and #1439 (M-1) come right. With a code's forms counted alike the gate passes exactly (815 / 4 / 23).
- The form weights lower box -log p (0.395 against 0.399) and help the point labels (24 against 23).
- I did not drop them to pass the gate: that would be choosing a model on 1 of 885 top-1 against the likelihood.
  Your call. Either rescore takes the same ~3.6 h.

Gate runs: `full_g2point_noends.records` (54 training point labels, 1,243 s) and `full_g1box_nochoose.records` (905
training box labels, 2,156 s), both 8 workers × 1 thread, load ≤ 11. The code those runs used is the committed
round-2 reader, apart from float32 against float64 shift weights in `_jitter`.

### 1. "Other" beat clean printed codes. PARTLY

**What it was** (#1414, the crisp M-2; `other_diag*.py`):
- **Not a spaced spelling of the code.** Scored as extra forms, "M- 2" got 16.6 and "M -2" 14.6 against "M-2" 18.6.
- **First cause: one inserted character, double-counting.** "1M-2" scored log E 38.9 against "M-2" 20.1. Its rigid
  fit is far worse (ratio 1,902 against 2,715; picture `diag_1414.png`, below). Each character's shift is
  independent, so the "1" slides onto the M's left leg and that ink counts twice.
- **I tried keeping each character's shift inside its own cell** (no overlap). "1M-2" fell to -4.7, but "M-.2"
  (23.0), "M 2" (19.1) and "M--2" (17.8) still beat "M-2" (1.5) in the same run.
- **Second cause: the print is spaced wider than the font's advances.** "Other" widens its layout with fillers. A
  code cannot: it has no letter-spacing freedom, and the shift limit made that worse.

**Synthetic check** (`spaced_try.py`; codes alone on paper, letters spread apart):

| extra per gap (cap heights) | M-2: end choice on / off | RH-1(D): end choice on / off |
|---|---|---|
| 0 | 1.000 / 1.000 | 1.000 / 1.000 |
| 0.15 | 0.617 / 0.850 | 0.000 / 0.066 |
| 0.3 | "other" / "other" | "other" / "other" |

The cell limit makes it worse (M-2 0.000 at 0.15). The committed reader already had this gap. Fix round 1's end
choice let "other" find it on 4 real labels.

**Done:**
- With Terra's word, "other"'s path is chosen without its end weight again (the committed choice). The exact end
  weights still apply at the end it carries.
- The four flips are gone: p(other) > 0.5 is back to 4.
- I dropped the cell limit.
- `test_crisp_code_on_clean_paper_reads_that_code_not_other` passes for the 7 test codes and M-2 (p > 0.9).
- `test_widely_spaced_code_reads_that_code` (M-2 with letters 0.3 cap heights apart) is a strict xfail with the
  measured reason.

**Not done: equal footing in letter spacing.** Codes and "other" need the same spacing model. Options:
- one letter-spacing (tracking) value per label, summed over a measured prior;
- a random walk of gaps with the measured gap spread, letters never overlapping.

Either changes codes and the Viterbi pass. The tracking prior must be measured on the training fits. The cost is
about the number of spacing values times the per-temperature work. Too much to build, check and rescore this round.

### 2. Critical 2 remainder. NOT addressed in the reader (the term is off)

The decision was made by training likelihood: the measured end-break term hurts the real point labels.

| | top-1 / 50 | mean -log p(code) |
|---|---|---|
| end term on (fix round 1) | 21 | 2.551 |
| end term off | 24 | 2.263 |
| committed | 23 | 2.265 |

- Example: #1705, string NC-1. Its NC-1 evidence fell from 10.1 to 5.9 nats with the term on, and it read RH-2.
- `END_BREAKS = False`. The measurement (`end_breaks`, `PosePrior.end_break`) and the term stay in the code, and
  the module says why the term is off.
- **The test exercises the term with `END_BREAKS` on.** It reads the full code at ±0.25, ±0.5, +0.75 and +1.0 word
  heights. Two offsets are strict xfails with the measured reasons:
  - −0.75: "other" 0.666 (RH-1(D) 0.267);
  - −1.0: RH-1 0.901.

  I found no principled fix for −0.75 this round. "Other" sums partial strings, and one end's factor costs it less
  than the full code's placement tail.
- **Synthetic off-centre RH-1(D) as the reader now stands** (term off):

  | point | reads |
  |---|---|
  | 0 | RH-1(D) 0.950 |
  | 0.25 before | "other" 0.602 |
  | 0.5 before | "other" 0.941 |
  | 0.75 before | "other" 0.977 |
  | 1.0 before | RH-1 0.762 |
  | 0.25 after | RH-1(D) 0.872 |
  | 0.5 after | RH-1(D) 0.814 |
  | 0.75 after | "other" 0.683 |
  | 1.0 after | "other" 0.745 |

  So off-centre points still read "other" or a prefix in the reader.
- **Real map:** training point labels with a code string are 24/50 (committed 23/50; fix round 1 21/50).

### 3. GPU turn. ADDRESSED

`evidence` no longer takes a GPU turn: the caller takes one exclusive turn around its whole run (turns don't nest).
The docstring says so, and that `settle_torch` is process-wide.

### Re-review minor 4 (old pickles)

Labels from `make_labels` carry `held_block`. Labels pickled before it read False until `labels.mark_held_blocks`
is called on the full list; every scratch script here calls it.

### Tests

The full suite: 1,137 passed and 3 strict xfails, 93 s on 6 workers. ruff is clean on posterior.py and the test file.

### Picture

`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task3/diag_1414.png`:
- **What it shows:** #1414's crop (grey), Terra's word (green), and the outlines of the rigid best fits of "M-2"
  (red) and "1M-2" (blue). The rigid "1" sits on paper left of the word; the jittered one slid onto the M.
- **Action:** none. It shows the mechanism.

### Concerns after round 2

1. **The gate fails by one box label** because of p(form | code) (see the table). It passes exactly with forms alike.
2. **Letter spacing:** a code printed wider than the font's advances reads "other", in the committed reader too.
   Equal footing needs a shared spacing model; it is not built.
3. **Off-centre point labels are not handled in the reader:** the end term helps synthetic points but hurt the real
   training point labels, so it is off. Point labels read badly overall (24/50 training).
4. **The fix-1 records** (`full_fix1held.records`, `full_fix1rest.records`) are from fix round 1's code, which has
   the "other" over-strength and the end term. Task 4 should not use them; a rescore of c626e46 (with or without the
   form weights) is needed first.
