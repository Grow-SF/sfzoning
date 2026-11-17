# Teaching the text detector letters from line ink: design (2026-11-17)

## Status

- **Origin:** written overnight by the controller, on the owner's instruction: "Autonomously brainstorm with super
  powers, then red team & correct the plan, then proceed with implementation of this text reader improvement."
- **Checked by:**
  - a builder check (six questions; answered here as "my call");
  - a red team: `pipeline/data/solver/experiments/redteam-text-vs-line/findings.md`, F1-F12, with scripts in
    `pipeline/scripts/spikes/redteam_text_vs_line/`.
- **This version is the corrected one.** The first draft's gate would have rejected a detector that fixed the
  owner's RM-2 case (F1).
- **The owner reviews this in the morning.**

## Why

- **The owner, on the detector's output** ("RM-2" with its block lines red): it has "learned how to detect the
  location of a label but not reliably detect the contents of the label text … You've got to actually learn to
  discriminate the text from the background ink."
- **The owner's idea:** "make some synthetic tests with the letter shapes that you've learned where you draw lines
  that look like the street lines and block lines, etc., on the image, and the detector learns to not detect those
  pixels but to detect the text pixels … this might mean learning to read the letters & words, which is good too."
- **Root cause** (`docs/research/joint-solver/notes/2026-11-16-text-target-root-cause.md`, fact-checked):
  - **The text target is the whole Terra word rectangle** (`textnet/data.py:469`). Every line through a word is
    taught as text.
  - **The scores were built from the same boxes,** so they counted that error as right.
  - **The held-out detectors' 328-px unknown discs** (one giant "SH") left 98% of the district-line ink near labels
    untaught.
- **Measured** (red team, F1/F2):
  - today's 30-px detectors call about 82% of straight-line ink inside word boxes "text";
  - the held-out ones call 31% of plain line ink near labels "text".

## Goals

1. **Text means characters.** The detector calls a px text when it belongs to a printed character, and not text when
   it belongs to line work. That includes where a line touches or crosses a character, right next to labels, on
   district borders, and on dashed lines and hatching.
2. **Independent measures.** They can't repeat the root cause: line ink is found by rules that use no Terra box and
   no training labeller, each validated on cases with exact truth, and nothing is excluded near labels.
3. **The smallest steps that show value come first** (value before verification). Each is measured against a
   control with measured noise.
4. **One fix for every consumer:** the walls, the word boxes and the text reader, on the 2000 map first. The method
   stays usable for the other years in training (86% of the training text is not from 2000).

## Non-goals (tonight)

- Heights, and T4/T5 of the text-reader plan (on hold for the owner).
- New LLM reads, other cities.
- A per-px character head: deferred (F11). The text reader (T3) is the part that reads.
- The alignment labeller and font extension: deferred until a measured gap needs them (F4, F10).

## Anti-goals

- A measure that reuses the training target's definition, or one built from Terra boxes for "line".
- A parameter set from the maximum of an LLM-labelled set.
- A verdict from wrong-lot counts or from one seed.
- Synthetic text the detector can tell apart by texture (F3).
- LLM output at inference.
- A GPL-family dependency.

## Design

### 1. Measures first (step 0, $0, CPU)

These are written before any training run, then used to score the existing detectors. Every rule below reports its
own accuracy on exact-truth synthetic cases (line-ink recall, glyph-ink false-line rate) beside every number it
produces.

- **M1a. District-border ink called text: the owner's own measure, restored.**
  - Ink on the zoning table's district borders (`borders.border_paths()`: lot borders where the table's codes
    differ), carried to page px by the placement.
  - No exclusion near boxes or labels.
  - Ink-to-border distance is weighed by the placement's measured residuals, a likelihood, not a band.
  - It depends on no ink rule and no Terra box, and it covers the stair-stepped district lines.
- **M1b. Straight-line ink called text.**
  - The run rule (`codereader/labels.py` `run_lines`: junctions don't cut a run), plus a chain rule for bent lines.
  - Split inside boxes / near boxes / far, and by distance to label points.
- **M1c. All ink outside every Terra box called text,** with no exclusions (F9).
- **M2. Text found.**
  - The base: ink inside Terra word boxes that is on no M1 rule and not within 1 px of one, overall and by kind
    (code / block number / street name), plus the owner's answered labels.
  - The overlap and adjacent band is reported on its own, never inside M2.
- **Readouts:**
  - every M number at chance ≥ 0.5 and ≥ 0.9, with the mean chance and log loss (probabilities, not cutoffs);
  - intervals by bootstrap over blocks.
- **M3. Missed-text audit.**
  - 64 random letter-sized ink pieces outside every box, per build, looked at by the controller.
  - The binomial bound on "text taught as not text" is stated beside M1c (F7: 11-25% today).
- **M4. A fixed picture panel:**
  - the RM-2 window;
  - picture 0's area;
  - a downtown block;
  - a residential block;
  - a real dashed line and a real hatched area.

  The controller looks at it for every candidate before accepting one, and it goes in the owner's morning report.
- **M5. Downstream, after a rebuild:**
  - the word boxes (T2's measures);
  - wall evidence on district borders inside word boxes, specifically (F8);
  - the walls' held-out log loss;
  - the solve's counted mistakes. Wrong lots are reported, but are not the verdict.
- **Baselines:** M1-M4 on the all-sheets detector (r30), the held-out detectors (r328) and the recipe trial (r30).

### 2. The control: fix the labels without new data (step 2)

- **Control (0)** is today's recipe with two label changes:
  - **The zones:** unknown only around Sol points with no matching Terra word, with radius equal to half the
    measured 99th-percentile code-word length (about 85 px). Points with a Terra word get no zone.
  - **Inside a Terra box:** text, except straight-line ink (run rule + chain rule), which is unknown, no loss (F5).

  This applies to every series. It needs no font, and it never teaches the run rule's px as "not text", so M1b
  measures what was learned elsewhere.
- **Runs:**
  - NE quarter, 2 seeds, plus one same-seed repeat (the noise, and whether mps is deterministic);
  - about 7 min of GPU each.
- **The control is the baseline** every later variant is compared against. Not r328: that comparison would mix the
  zone fix into the label fix (F6).

### 3. Composites with real glyph ink: the owner's idea, done with real letter shapes (step 3)

- **Clean words.**
  - These are Terra words whose ink pieces lie wholly inside the box, with no straight line within 3 px. Their ink is
    an exact text mask with real texture.
  - On 2000 that gives 718 codes, 3,397 block numbers and 604 street names. Other series give their own sprites.
- **Line windows:** real windows of scan line work with no text.
  - They are sampled near labels, from the measured distribution of line-to-label distances.
  - They are checked by the missed-text audit, and outside-box pieces are weighted by a measured p(text) (F7).
- **Composites by OR** (1-bit overprint is a union), in both directions so that texture predicts neither label (F3):
  - a clean word onto a line window, placed to cross or touch lines at the measured rates for its kind;
  - real line pieces and drawn lines (widths from `strokes.npz`; the drawn lines degraded to match real edges) onto
    a clean word.
- **Exact masks:** text = the word's own ink; line ink = not text. An overlap px is text, with the line px beside it
  weighted up, on composites only.
- **The domain check before training:** the composites' ink statistics (isolated px, pinholes, pieces per word) must
  match real words. Any rendered font glyphs, if used later, must match too.
- **Mixing:**
  - Composites enter `textnet/train.py` `draw()` at share p, drawn from the batch thread's single rng.
  - p is the measured share of line-crossed words: one value, no sweep tonight.
  - Each held-out detector draws sprites and windows only from its training quarters, and never from the owner's
    answered blocks.
- **Variant (1) = (0) + composites:** NE, 2 seeds.

### 4. The decision rule (written before the runs)

- **(1) replaces (0)** if, on held-out NE:
  - M1a and M1b inside boxes fall, with block-bootstrap intervals that exclude zero;
  - M2 on the clean base falls by no more than the control's seed spread;
  - M1c rises by no more than the control's seed spread;
  - M4's pictures agree, by the controller's eye.
- **The best of (0)/(1) is then confirmed on SW** (step 4), one seed.
- **Mixed results** are reported with pictures of the losses and are not adopted silently. "Better is enough": a
  clear M1 win with M2 inside noise is kept.

### 5. Rebuild (step 5, only if a variant is kept)

- **Retrain:** the chosen recipe becomes the recipe of `solver-text-reader` (the four held-out detectors) and of
  `solver-text` (the all-sheets detector), one recipe for both. The same stage names rebuild through
  `pixi run build`, one GPU job at a time.
- **Downstream rebuilds through DVC:** the walls (T1), the word boxes (T2) and T3's reading of the boxes.
- **Measure M5.**
- **Stop before T4.** T4 and T5 wait for the owner.

### 6. Later, only if a measured gap remains (not tonight)

- **The alignment labeller** (`glyphs.align` on Terra strings). It is validated first on composites, by per-px
  precision and recall by case. It labels actual ink by stroke, not the template footprint. (F4)
- **Font extension** (0, 5-9, J, Q, X, Z) with a texture-matched renderer, for rare strings and heights.
- **Two outputs, glyph ink and line ink,** with the walls reading the line output (F8). Or the walls' per-stroke
  statistic leaves out junction px.
- **A p sweep.** A per-px character head as an auxiliary loss on composites.

## Constraints

- **$0:** no LLM calls.
- **One GPU (mps) process at a time** (`gpu.gpu_turn`). Load near 20 (≤ 4 workers × 1 thread).
- **Deterministic:** seeded, one rng in the batch thread. Whether mps runs repeat is measured by the control's
  same-seed repeat.
- **Held out by quarter,** and the owner's answered blocks never trained on.
- **Apache-2.0 code;** no GPL-family dependency (scikit-image and OpenCV are fine).
- **Code** in `pipeline/src/sfzmaps/solver/` for stages and measures that stay, `scripts/spikes/` for one-off checks.
  Data in `pipeline/data/`, never committed. Avoid editing `paths.py`, which re-hashes every solver stage.
- **Rebuilds via `pixi run build`;** `dvc.lock` is committed. `dvc commit --force` only after an output-identity
  check, and never as acceptance.

## The builder's questions, answered (my calls)

1. **Overlap px under M1:** M1 counts line ink as found by its rules. Overlap and adjacent px are reported apart and
   left out of M2. Composites (exact) are the overlap check.
2. **If alignment fails on the hard cases:** alignment is deferred. The control's "line ink in boxes = unknown" plus
   the composites carry the hard cases.
3. **Scope:** build through step 5's rebuild and M5, then stop before T4.
4. **Which detector:** one recipe for both the held-out four and the all-sheets one. The 328 and 30 radii are
   retired for the measured zone rule.
5. **Other years:** the control's label rule applies to every series, and the composites use each series' own clean
   words. No 2000 font is drawn into other years.
6. **"Beyond noise":** the control's measured seed spread and block-bootstrap intervals (§4).

## How this mistake got in, and what changes

From the root-cause note and its fact-check:
- **An inherited training target gets decided:** written down with what it labels, and checked by a dense-area
  picture.
- **A gate states what it would miss,** and never shares the training target's definition.
- **No parameter comes from a maximum** over an LLM-labelled set.
- **A dense-area picture of every detector is looked at** before its stage is accepted. A caption says only what the
  picture shows: the controller's caption on picture 0, "call only the lettering", was wrong.
- **Recording a stage after an inputs-unchanged check** is bookkeeping, not acceptance.
