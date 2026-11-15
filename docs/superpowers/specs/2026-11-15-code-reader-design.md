# A probabilistic reader for the printed zoning codes

Design, 2026-11-15. The owner's idea, the owner approved the outline in conversation; calls the owner did not make are
marked **(my call)**. Builds after the text detector's speed work (`2026-11-15-text-detector-training-design.md`).

## Why

The joint solver's worst mistakes start from misread zoning labels.

- A misread code in a narrow strip renames the strip, and the big flips come from labels.
- The readings today are LLM reads (Terra, Sol, several passes), combined by a fitted trust.
- On the owner's 717 hand-checked codes, J7's combined reading is right 96.7%, and 98.1% on the fair, unflagged
  sample. At least one reader is right on 99.4%.
- The misses are hard labels:
  - small (under about 23 px tall);
  - sideways or slanted (39% of hard labels against 6% of easy ones);
  - fused with heavy boundary or lot lines, where a thick line through an H reads as M and 1/2/3 blur;
  - or read at the wrong place (the reading point off the label, or a box swallowing a neighbour).

The owner's model, in their words: a prior over what the text says, set by where it is, times the likelihood of the
ink given the text, "the text … that makes the existing block of noisy ink most likely". It is probabilistic, not
"try each word and pick" (owner: "that's not probabilistic"). It was inspired by deblurring: ask which letter, passed
through the blur, best explains the smudge.

Prior art does exactly this:

- Xerox's Document Image Decoding (Kopec & Chou 1994) and Berkeley's Ocular (Berg-Kirkpatrick et al. 2013): glyph
  templates learned from the document, scored against the ink under a pixel-noise model, times a prior over strings.
- For maps, Weinman 2017 used location priors from centrelines and cut word errors by about a third.

## Goals

- **A reader that gives, for each printed zoning label on the 2000 map, a posterior over the codes** (about 60, plus
  "no code here" and "some other text"), from the original scan.
- **Better than J7 where J7 fails:** top-1 on the owner's answers above 96.7% overall (98.1% fair). Calibrated: a
  posterior of 0.9 is right about 90% of the time.
- **Fed to the solver as a reading,** a principled probability beside or instead of the LLM reads. Judged then by
  the solver's "labels right" and the six measures.
- **Deterministic and fast:** scoring every label in minutes.

## Non-goals (now)

- Street names and block numbers. They come next, reusing the machinery with their own priors (85% of 2000 street
  names sit within 50 ft of the same-named centreline; 92% of block numbers equal their block's number).
- Other years' codes (later: the serif years need their own glyphs).
- Leader lines (placement already handles them).

## Anti-goals

- **Using the table's code at a label's spot,** in the prior or anywhere else. That is the answer key the solver is
  scored against.
- **Training on the owner's 717 answers.** They are the held-out test. The labels in the same blocks as an answered
  label are left out of training too.
  - The only exception is the three calibration constants, fitted by two-fold cross-fitting (below).
  - **(my call)**
- **Hard cutoffs as decisions.** The pose window and line masks are probabilities, or "unobserved"; nothing is
  thrown away.
- **New paid reads.**

## The model

For one label, p(code | ink) ∝ p(ink | code) × p(code).

- **Glyph templates (the likelihood's alphabet).** One template per character of the 2000 lettering: R, H, M, C, N,
  P, S, D, digits 1–4, "-", "(", ")" and the other letters in the code list. Learned from the page itself, by
  aligning Terra's read strings to their crops (hard EM, as DID/Ocular do).
  - Train on labels outside the test set and its blocks.
  - Block numbers and street names on the same sheets may help the digits and letters they share. **(my call: try
    it, keep it if held-out training likelihood improves)**
- **Rendering a code:** the code's string laid out from the templates, with per-character spacing, at a pose. The
  pose is a position, a size and an angle, summed over a grid; the grid is forward-only, which keeps it
  deterministic on the GPU.
  - Prior over position: centred on the print's point (Terra box centre or Sol point), with a spread of a few letter
    heights, so a reading point off the label can still find it.
  - Prior over angle: from the box's angle, or upright/sideways when there is no box.
  - Prior over size: the measured spread of 2000 code heights.
- **Noise:** pixel flips more likely near stroke edges (the Kanungo degradation model), fitted on training labels.
  - **Pixels on long lines are unobserved:** they come from the long-line masks the detector's scoring already
    computes, dilated slightly. A boundary line through a label then neither helps nor hurts any code, so a thick
    line can't turn H into M.
- **Hypotheses:** every code in the list; "no code here" (the ink explained by background and lines); "some other
  text", scored by an open-vocabulary pass with a character bigram, so non-code text such as "UPPER MARKET" or an
  SUD name is not forced into a code.
- **Prior p(code):** how often each code is printed on the map, counted from the readings of training labels (not
  the table, not the test labels), smoothed. "No code" and "other" get small fixed masses, fitted on training labels.
  **(my call)** A district-shape or neighbour prior is left to the solver, which already couples labels and walls.
- **Calibration:** one temperature on the log-likelihood, fitted on a held-back slice of training labels. Pixels
  aren't independent, so raw likelihoods are overconfident; Ocular needed the same.

## Evaluation

The same measure for every comparison.

- **On the owner's 717 answered codes**, and the 593 unflagged as the fair sample; the 20 "no label" answers check
  "no code here":
  - top-1 and top-3;
  - mean −log p(true);
  - a reliability curve;
  - results for the hard subset (small, sideways, line-touching).
- **The same numbers for each existing reader and J7's trust,** so the comparison is like for like.
- **Then in the solver:** the reader's posterior as the reading set (alone, and beside the LLM reads). Report "labels
  right" and the six measures against the last solve.

## Determinism, speed

Seeded; deterministic algorithms. Pose search is forward-only. Training (template EM) and scoring each take minutes,
on the CPU or the GPU in one exclusive turn.

## The build

- Code: `sfzmaps.solver.codereader` (templates, likelihood, posterior).
- A stage `solver-codes` writes `data/solver/codes/`: the templates and a posterior per print. It reads the pages,
  prints, Terra's answers and the long-line masks.
- The labels stage gains the reader's posteriors as a reading set. **(my call: the builder decides beside vs
  instead, by the solver measure)**

## Settled after the builder check (my calls)

- **Truth for each fit:**
  - **Glyph templates, spacing and noise** learn from the LLM reads of training labels. Their strings are good enough
    to learn letter shapes from.
  - **The calibration constants** (the temperature and the "no code" / "other" masses, three numbers) are fitted by
    two-fold cross-fitting on the owner's answers: fit on one half, score the other, then swap. Every reported number
    then comes from answers not used to fit it.
  - Nothing else is tuned on the answers.
  - The owner may answer more of the 905 unanswered code-check cases as extra calibration data. This is optional.
- **Passing:**
  - The reader ships if adding it improves the solver's "labels right" and the six measures, without worsening the
    others.
  - Reader-alone accuracy is reported against J7's 96.7% / 98.1%, but it is not a gate.
- **Joining the solver:** the reader's posterior log-odds per candidate code become extra features of the existing
  trust fit (`trust.py`). That fit already combines the reads, and it learns from the table held out by quarter, as
  today.
  - The rule against the table is about the reader. It never sees the table code at a label's spot.
  - Also reported: the reader's posterior alone as the reading set.
- **Line masks:** start with the detector's long-line masks (≥ 200 ft strokes). Measure how much line ink they cover
  in label windows. If they miss lines broken by junctions, the builder may use a width-based or line-layer rule,
  chosen by likelihood on training labels.
- **The code list:**
  - The hypotheses are the table's code names (the list only, no locations), plus printed variants seen in training
    reads (e.g. "NOE VALLEY", "UPPER MARKET NCD"), each mapped to its code.
  - The 6 answers outside the list (NOE VALLEY, CMNC, 3RD, NC, RC-1, MB-OS) are reported separately.

## Left to the builder

- Template resolution and parameterisation.
- The pose grid's steps.
- The hard-EM details.
- How "other text" is scored.
- Module layout.
