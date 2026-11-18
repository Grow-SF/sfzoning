# Task 2 report: glyph templates and the noise model

Status: DONE_WITH_CONCERNS (listed at the end).
- Branch `solver`, commit fda4c42.
- CPU only: numpy, OpenCV and scipy's FFT on 10 threads.  No GPU, no LLM, no `.env`, no pipeline stage run.

## What was built

`pipeline/src/sfzmaps/solver/codereader/glyphs.py` (plus one line in the package docstring); tests in
`pipeline/tests/test_codereader.py`.

- **`Font`**:
  - One template per character: p(ink) at each px of a 52-row grid (cap height `SIZE` = 32 px, 10 rows above and
    below), in the upright frame.
  - Per character: `origin` (the pen's column), `advance` (the pen's move after it) and `extent` (the first font's
    ink columns, which decide the px a character owns).
  - The font as a whole: `slant` (the lettering's lean, put back at layout), `size`, `counts`, `spacing_sd`,
    `position_sd` (the measured spread of the drawn text about Terra's word centre, in word heights), `noise` and
    `trace`.
  - Methods: `width(c)` (inked width), `layout`, `laid` (cached), and `save`/`load` (.npz).
- **`initial_font(chars, slant, face, stroke)`**: the plain sans first font is **OpenCV's Hershey Simplex**.
  - It is a single-stroke sans built into OpenCV, so it needs no font files.
  - Stroke 0.18 of the cap height, drawn at 4x with anti-aliasing and averaged down; p = 0.02 + 0.96 x coverage.
- **`render(font, text, pose, shape) -> array`**: the signed distance from the stroke edge of the laid-out text, in
  crop px, clipped to +-6 px.
  - The text's shape is the template px at p >= 0.5, united over the characters and leaned.
  - A **`Pose`** places it: centre x, y in the crop; scale (crop px per template px); angle (degrees
    counter-clockwise, 180 = upside down).
  - It takes the brief's signature plus `shape`, the crop's size.
- **`NoiseModel`**: Kanungo-style, measured rather than parametric.
  - `p` holds p(ink) at 25 knots of signed distance, -6..+6 crop px.  It is linear between knots, constant beyond,
    and made non-decreasing (pool adjacent violators).
  - Methods: `fit(rendered, crops, observed, weights)` (soft-binned counts, observed px only),
    `log_lik(rendered, crop, observed)` (observed px only), `p_ink`, `background`, `flips()` (paper->ink off
    strokes, ink->paper on them) and `default()`.
- **`learn_font(labels, iterations, *, texts, observed, charset, rules, residuals, ratios, workers) -> Font`**:
  hard EM, detailed below.
- **Also public:** `consensus(label)` (the training string), `Search`/`search_for` (the pose grid and prior),
  `align(font, noise, text, ink, observed, search) -> Fit`, `refine`, `label_weights`, `signed_distance`, `lay`,
  `word_heights`.

### The EM (my calls; say if you want them otherwise)

- **Training strings** (`consensus`):
  - Terra's text as written, when its code (`codes.canon`) has the most reads (Terra wins ties).
  - Otherwise Sol's as written; otherwise the plurality read.
  - The string differs from Terra's text on 22 box labels; counting the 54 point labels, which have no Terra text,
    76.
- **Round 0 picks the first font's slant** from 0/5/10/15/20 degrees by the likelihood of every 4th label: **15
  degrees** (log-lik -2,368,138 / -2,355,897 / -2,346,862 / **-2,343,136** / -2,345,999).
- **E-step** (`align`): every scale x angle of the label's grid and every position in the crop.
  - Correlation by FFT: two correlations folded into one inverse FFT per pose, in scipy, float32.  The chosen
    pose's likelihood is then summed exactly in float64.
  - Grid with a Terra box: cap height 0.60-1.60 x the word's height (14 steps of 7.8%).  Angles: Terra's and its half
    turn, each +-6 degrees in 3 degree steps.  That is 140 poses.
  - Grid without a box (point labels): cap heights over the training words' 5-99% heights at the same steps, and all
    four quarter turns.
  - Position prior: a Gaussian on the **centre of the drawn ink's box** (what Terra's box measures), not on the
    layout centre.
- **The position prior is measured.**
  - Round 0 uses a broad prior (sd 1 word height).  Under it (an earlier run), 20% of the box labels' strings went
    more than a word height from their word (picture `far_v2.png`).
  - The reason: a likelihood summed over px pays more for a match on *larger* ink, such as a bigger "RM-2" for a
    small "P", or the "LI" of "JAMES LICK" for "SLI".
  - So after round 0 the word labels' prior sd becomes the spread measured in round 0, the "good" part of the
    escape's mixture: **0.099 word heights** along x (0.033 down).
  - With it, 4.2% of box labels' fits are more than a word height away; the median is 0.056 h.
- **The "mislabelled" escape** (`label_weights`): a two-part mixture, fitted by EM per group (word labels apart from
  point labels).
  - What it models: where the string was found (its drawn ink's centre, against the word's centre or the point, in
    the prior's sds), and for word labels how large (the log of cap height over word height).
  - "Good" is a Gaussian; "bad" is any pose tried alike.  The bad share is at most a half, and no sd goes below the
    grid's rounding (step / sqrt 12).
  - Each label counts in the M-step by its share of "good".
  - The scale axis was added because, with the grid to 1.6, 7.2% of the fits sit at its top.  Most of those are
    strings swollen over other ink (picture `bigscale_v4.png`).
  - A first version scored the fit's log-likelihood ratio per px, with a "bad" Gaussian at 0.  It never fired (bad
    share 0 every round), because that ratio grows with the letter size.
- **M-step**:
  - Each character is re-placed alone within +-0.25 cap heights of its laid-out spot (`refine`, with a Gaussian
    prior on the shift).
  - **The other characters' strokes are unobserved meanwhile.**  Without that, a neighbour's ink in the
    character's expected paper pushed it away every round.  In the synthetic test the hyphen's glyph drifted 7 px
    in its template over 6 rounds.
  - Each template px becomes the weighted mean observed ink there, line px unobserved, with the first font counting
    as 2 instances.
  - A character owns the px on its side of the middles of the gaps to its neighbours' ink (the first font's extents,
    upright frame).
  - Advances: weighted least squares on the gaps between the shifted pens, ridge 2 towards the last.  A gap measures
    the first character's advance alone, plus any spaces.  An earlier layout, with characters centred in cells, only
    identified sums of neighbouring advances and drifted.
  - Then each label is re-placed (at its round's scale and angle) under the new layout, and the noise refitted
    there.
- **The font returned is the best round's** by training log-likelihood, after round 0.  Hard EM wanders slightly
  once settled, because the M-step's per-character shifts are not the E-step's rigid layout.
- **Line masks**: `rules="run"` alone, so the unobserved px are `labels.line_mask(label, rules="run")`.
  - "long+run", as the brief suggested, hides large letters: `long` leaves 82-100% of a big P's or SSO's ink
    unobserved (picture `masks_worst.png`).
  - Share of a word's own ink masked, over the 905 word labels:

    | rule | mean | words over half masked |
    |---|---|---|
    | long | 0.14 | 8% |
    | run | 0.26 | 13.9% |
    | long+run | 0.34 | 25% |

    For run, that is mostly lines crossing the word; letters are masked only where they touch a straight line.
  - With long+run (the first real run), the strings of fully masked words went to other ink, and the templates
    learned from it.
  - Task 3 still chooses its rule by likelihood.

## Real run (959 training labels, CPU)

**Timings:**
- Labels load 2.4 s; run-rule masks 8 s (38 s for long+run).
- Slant choice 240 s.
- **learn_font 1,124 s** for 6 rounds; each E-step 115-141 s on 10 threads.
- The whole script, with a final fit of every label, 1,399 s.
- Other processes shared the CPU during this run.  An earlier run without them, on the 11-step grid, took 81-87 s
  an E-step and 720 s for learn_font.
- One label on one thread takes 0.13 s (11-step grid).  OpenCV's matchTemplate took 0.35 s, so the FFT path is 2.7x
  faster.

Training log-likelihood per round (every label at its best pose; "kept" = sum of the escape's weights):

| round | log-likelihood | vs bare background | kept of 959 | upside-down fits | E-step s |
|---|---|---|---|---|---|
| 0 (first font, broad prior) | -9,235,416 | +936,323 | 613.6 | 110 | 141 |
| 1 (measured prior from here) | -9,177,510 | +1,127,299 | 821.1 | 43 | 132 |
| 2 | -9,142,048 | +1,084,553 | 799.5 | 39 | 116 |
| 3 | -9,148,092 | +1,073,718 | 810.2 | 37 | 116 |
| 4 | -9,148,178 | +1,074,582 | 823.1 | 39 | 115 |
| 5 | -9,142,146 | +1,078,860 | 823.8 | 39 | 119 |
| **6 (returned)** | **-9,137,146** | +1,086,148 | 814.7 | 39 | 117 |

- **Shape of the trace:** most of the gain comes in rounds 1-2; after that it moves within 0.12%.  Round 1's jump
  in "vs background" comes partly from the tighter prior: fits held at their word give up larger ink elsewhere.
- **Earlier runs, for the record:**
  - long+run masks, broad prior, ratio-based escape: -7,818,414 -> -7,625,354 (round 5) -> -7,627,890 (round 8).
  - run masks, measured prior, 11-step grid: -9,257,205 -> -9,159,140 (round 5).
  - The totals differ between mask rules because the observed px differ.

**Noise** (p(ink) by signed distance from the drawn stroke edge, in crop px):

| d | <= -2 (background) | -1.5 | -1 | -0.5 | 0 | +0.5 | +1 | +1.5 | +2 | +3 | +6 (deep) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p(ink) | **0.073** | 0.101 | 0.159 | 0.287 | 0.465 | 0.655 | 0.792 | 0.878 | 0.916 | 0.951 | 0.972 |

- Paper->ink flips: 0.29 half a px off a stroke, and the background's 0.073 beyond 2 px.
- Ink->paper flips: 0.35 half a px inside, and 0.03-0.05 deeper in.

**The font:**
- Slant 15 degrees.  The gaps spread 3.4 template px (0.11 cap) about the advances.
- Weighted instances per character:

  | characters | instances |
  |---|---|
  | - | 539 |
  | R | 437 |
  | H | 272 |
  | 1 | 212 |
  | P | 181 |
  | M, 2 | 174 each |
  | C | 122 |
  | 3 | 102 |
  | N | 98 |
  | S | 43 |
  | D | 36 |
  | L | 21 |
  | O | 20 |
  | ( | 19 |
  | ) | 18 |
  | E | 17 |
  | I | 15 |
  | T, 4 | 13 each |
  | A | 11 |
  | U | 8 |
  | G, . | 6 each |
  | F | 4 |
  | B, W | 3 each |
  | K, Y | 2 each |
  | V | 0.8 |

- Every character of the 70-form code list is in the font.  The rare ones stay close to the first font, which
  counts as 2 instances.

**Turns** (fits upside down, by Terra's direction):

| Terra's direction | upside down |
|---|---|
| top-to-bottom | 7 of 11 (all 11 are printed bottom-to-top, per Task 1) |
| left-to-right | 16 of 750 |
| bottom-to-top | 1 of 75 |
| slanted | 11 of 69 |
| point labels | 4 of 54 |

The left-to-right ones are mostly escaped fits on other ink (`alignments_real.png`).

**Scale:**
- Cap height / Terra's word height, 5-95%: 0.81-1.60, median 1.02.
- 0.2% sit at the grid's bottom and 7.2% at its top.  On the 0.60-1.25 grid it was 10.4% at the top, many of them
  good fits to words taller than Terra's box.

## Extra text (block numbers and street names): not kept

- **Held-back slice:** a fixed 20% of the training labels' blocks (167 of 833 blocks; seed 20261116), so 190 labels
  held back and 769 to fit on.
- **The extra text:** Terra's block numbers and street names on the 13 sheets, plain characters only.
  - 7,262 items; 1,225 dropped as in a held-out or held-back block.
  - Every 4th of the rest is used, 1,510 items, to keep the runtime down.
  - Crops and run-rule masks are made the same way as the codes'.
- **The fonts:** each learned for 5 rounds, on the 0.60-1.25 grid without the scale axis in the escape (the code
  then).  Each is scored by the held-back labels' log-likelihood at their best poses, with its own noise and the
  same measured position prior.

| font | held-back log-likelihood | vs A | held-back labels better / worse than A | time |
|---|---|---|---|---|
| A: codes only | -1,930,651 | | | 530 s |
| B1: + extra text as printed | -1,944,298 | -13,647 (-71.8 a label) | 52 / 138 | 1,132 s |
| B2: + extra text leaned 15 degrees like the codes | -1,943,690 | -13,039 (-68.6 a label) | 59 / 131 | 1,148 s |

Both lower the held-back likelihood, so the extra text stays out and nothing for it goes into the package.  The
block numbers and street names are an upright, smaller hand with other letter shapes (`peek_extras.png`).  Leaning
them to the codes' slant doesn't make them the same letters.

## Pictures (scratch)

Each path is under
`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/`.

- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/glyphs_real.png`
  - **What it shows:** the learned 2000 glyphs.  Each cell holds three panels:
    - the first font (Hershey Simplex);
    - the learned p(ink), darker meaning more ink;
    - the learned shape (p >= 0.5), leaned 15 degrees as it is rendered.

    Captions: n = weighted instances, adv = advance, w = inked width (template px).
  - **What to look at:** the learned letters are the map's heavy italic lettering, its "1" a plain bar and its "-"
    short and thick.  Rare letters (B, K, V, W, Y) are mostly the first font.  The hyphen and the B carry a stray
    speck.
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/alignments_real.png`
  - **What it shows:** each training label's best alignment, drawn over its crop:
    - which labels: 16 at random, the 12 the escape weighs least, and 12 fitted upside down;
    - the colours: green = rendered stroke on ink, pink = rendered stroke on paper, red = its edge, blue = line px
      (unobserved);
    - the caption: the string, the escape weight w, the angle and the cap height.
  - **What to look at:** the random ones sit on their words, RH-1(D) with its parentheses.  The least-weighted are
    strings found on a neighbouring label.  Upside-down fits on real bottom-to-top prints keep weight 1; the rest
    are weighted 0.
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/far_v2.png`
  - **What it shows:** why the prior is measured.  Under the broad prior, each string's free fit (red) is drawn
    against its fit held at its word (green).  Terra's word is the magenta box.
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/bigscale_v4.png`
  - **What it shows:** why the scale is in the escape.  These are fits at the top of the scale grid; most are
    swollen or elsewhere.
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/masks_worst.png`
  - **What it shows:** why the masks are "run".  These are the 24 words the long+run union hides most; red = long
    only, blue = run only, purple = both.
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task2/peek_extras.png`
  - **What it shows:** sample block numbers and street names.

Scratch scripts, in the same folder:
- `real.py` (the run), with its outputs `real.log`, `trace_real.json`, `font_real.npz` and `fits_real.pkl`.  The
  earlier runs' outputs are `*_v2`, `*_v4`.
- The experiment: `extra_text.py`, `extras.py`, `extra_text.log` and `extra_text.json`.
- The analyses: `pictures.py`, `offsets.py`, `far.py`, `bigscale.py`, `masks.py`, `synth_debug.py`,
  `synth_iou.py` and `prof.py`.

## Tests

`pipeline/tests/test_codereader.py` has 3 new tests.

**`test_em_recovers_templates_from_synthetic_text`**
- The synthetic crops come from a known font: **Hershey Complex** (serif), with a heavier stroke, a 12 degree lean
  and 10% wider spacing.
- Noise is Kanungo-like (background 0.04, edge scale 0.6 px).  Caps are 20-28 px, angles +-2.5 degrees.
- Every 5th crop is upside down; every 3rd is crossed by a 5 px line whose px are unobserved.  Two labels' word
  boxes are 50 px off.
- After 4 rounds, starting from Hershey Simplex, the test checks:
  - each character's shape IoU with the truth is > 0.8, and more than the first font's + 0.05 (0.90-0.95 against
    0.70-0.81);
  - the background rate is within 0.015 of the truth, and p at d = -2, 0, +2 within 0.12;
  - the returned round is the best, and better than the first font;
  - exactly the two misplaced labels were let go;
  - each word is found to 2.5 px, with the upside-down ones upside down;
  - each word renders where the truth drew it: IoU min > 0.65, median > 0.8 (the small grid's 10% scale steps
    limit it);
  - the font, noise and position spread come back from `save`/`load` unchanged.

**`test_label_weights_let_go_of_strings_found_far_from_their_label`**

**`test_unobserved_pixels_change_nothing`**
- Scribbling under the lines leaves the likelihood unchanged, and changes it when those px are observed.
- It also leaves one EM round's templates, advances, noise and trace bit-identical.  That checks the threaded FFT
  path for determinism as well.

Full suite (`cd pipeline && uv run pytest -q`): **1,102 passed, 3 skipped** in 93 s, before commit.  That run
includes the other implementer's uncommitted files in the worktree.  `ruff check --line-length 120` is clean on both
files.

## Concerns

1. **A likelihood summed over px rewards larger ink**, both larger text elsewhere and a swollen scale.  Pixels are
   not independent, so the log-likelihood ratios run to hundreds of nats and a few-nats prior can't hold a pose.
   - For training I measured a tight position prior (0.099 word heights) and put position and scale into the
     escape.
   - **Task 3 will meet this when every code competes over a window**: a code can win by matching a neighbouring
     label.  The spec's "spread of a few letter heights" prior will not hold a code at its label.
   - The measured spreads are now on the font: `position_sd` for position; for scale, about 0.14 in log cap/word
     height, in the trace.  Task 3's temperature (spec) also bears on this.
2. **The training log-likelihood is not monotone after round 2.**  It wanders within 0.12%, because the M-step uses
   per-character shifts and the E-step a rigid layout.  The font returned is the best round's.  An elastic layout in
   both steps would fix it; it is left out.
3. **7.2% of word-label fits sit at the scale grid's top (1.6)**, mostly failures that the escape now weighs out.
   Widening the grid further would cost time for those failures.
4. **Point labels (54):** their escape mixture runs at its cap (bad share 0.5) every round, so their offsets are
   spread wide.  They add little.  Many are Sol strings of multi-line names laid out on one line.
5. **Rare letters** (B, K, V, W, Y have under 4 instances) are mostly the first font; F has 4 and G 6.  The extra
   text didn't help, as above.  The hyphen and the B templates carry a stray speck from clutter.
6. **No kerning**: one advance per character.  ")" after "D" and "1" before "(" are the visible cases.  The gaps
   spread 3.4 template px.
7. **`render` takes a 4th argument `shape`** (the crop size) beyond the brief's signature.
8. **The line rule is "run"**, not long+run (measured above).  Task 1's coverage table counted lines missed near
   labels.  Hiding letters is the bigger harm for template learning, and Task 3 should weigh the same for scoring.
9. **Runtime:** learn_font takes 12-19 min on the CPU depending on the load (spec: "minutes").  The E-step is FFT
   correlation; the GPU wasn't used.
10. **The default ITERATIONS is 6**: the gains end by round 2 and the best round varies (2 or 6).
