# The code reader, read along the word

Design, 2026-11-16. It revises the pose handling of `2026-11-15-code-reader-design.md`, which stays the authority on
everything this note doesn't change.

The owner set the direction in conversation. Calls the owner did not make are marked **(my call)**.

## Why

The reader works out, for each printed zoning label, the probability of each code given the ink:
p(code | ink) ∝ p(ink | code) p(code). The unknown is where and how the code sits: its position, size and angle. That
pose was summed over by brute force:
- every code;
- every position near the label, at several sizes;
- the box's angle and its half turn, or four quarter turns for point labels, each with a few degrees of play;
- at full scan resolution, on the CPU.

A full run of the 1,905 labels took 3.6 hours, about 170 s per point label and 25 s per box label. Point labels still
read badly (24 of 50 training labels), because their direction and extent were searched blindly.

The owner: "Every rotation? That doesn't sound like probabilistic recognition!!" The pose should come from evidence,
not a sweep. And the owner on the reader itself: a posterior, not "try each word". The pose is no different.

The map-text literature and OCR practice do the same:
- detect the word;
- straighten it along its line;
- read it with a left-to-right model that sums over where each letter starts (Kopec & Chou's Document Image Decoding;
  Berg-Kirkpatrick et al.'s Ocular, a semi-Markov model over glyph widths and offsets);
- for map words, add priors from where the word sits (Weinman 2017, with gazetteer and location priors).

## Goals

- **The same output as before:** for each label, a posterior over the code list plus "no code" and "other text". It
  is judged as the reader spec says: on the owner's answers (Task 4) and then in the solver (Task 5).
- **The pose comes from evidence the project already has.** Not a grid.
- **Point labels get their direction and extent from the ink,** and read as well as box labels of the same kind of
  print.
- **Codes and "other text" are scored by the same reading model,** with the same freedom of letter spacing, so neither
  wins by layout alone.
- **A full run takes minutes on the CPU.** Deterministic as before.

## Non-goals (now)

- Street names and block numbers (next, with the same machinery and their own priors).
- Labels printed on two lines.
- Other years' lettering.

## Anti-goals

- A brute-force sweep over angles, positions or sizes.
- Hard cutoffs as decisions. Any truncation is a numerical tolerance with a measured bound on what it drops, never a
  choice about what the text is.
- Using the table's code at a label's spot, or the owner's answers for anything but the three calibration constants
  (cross-fitted).
- New LLM reads.

## The model

### 1. A detector for the reader, trained by quarter

The text detector predicts three things per px:
- the chance it is text;
- the kind (zoning code, block number, street name, other);
- the reading direction as cos and sin, which tells upright from upside down.

The `solver-text` stage's detector can't serve the reader. It was trained on every 2000 sheet, so at each label,
including the answered labels and their blocks, its kind and direction are fitted to Terra's box, not predicted. And
around a Sol point, only 30 px were marked unknown, so the ends of a point label's word were taught as "not text".

So the reader gets its own stage, `solver-text-reader`. **(my call)**
- **Four detectors, one per quarter.** Each trains on every training map with that quarter of the 2000 sheets held
  out, by the quarter split the solver uses. Each label is read with the detector that never saw its quarter.
- **The answered labels' blocks** are held out of all four.
- **Around every Sol point,** the whole likely word is marked unknown: a measured word length each way along any
  direction, not 30 px.
- **Same recipe as the stage's.** About 7 min of GPU each, in one exclusive turn. It saves all three outputs for the
  2000 sheets.
- **`solver-text` and the walls are untouched.**

The spreads in step 2 are measured from these out-of-quarter outputs, so they're honest for both box and point labels.

### 2. Find the word

For each label, from its Terra box or its Sol point, work out a distribution over the word's frame:
- **Direction:** from the detector's direction on the text px near the label, weighted by their text chance. It is a
  full 360°, so there is no turn search.
- **Letter height:** from the text px's extent across the direction.
- **Position across the line.**
- **Where the word starts and ends along the direction:** from the profile of text chance along the line.

Each comes with a spread measured on the training labels, against the pose Task 2's alignment found for them. The
frame is a small weighted set of hypotheses, not one guess:
- a fused neighbour word, or a heavy line through the label, can make the grouping ambiguous, so the alternative
  groupings carry their probabilities;
- line px are unobserved, as before;
- a box, where Terra gives one, is evidence about the frame (its angle and extent), weighed by a measured spread like
  the rest. **(my call)**
- short words with a weak direction (a lone "P") fall back toward the box's angle, by the same weighting.

### 3. Straighten, and score every hypothesis on the same px

**One common window per label.** It is the union of all its frame hypotheses' extents plus a margin. Every hypothesis
is scored on that window's original page px: each code, "other text" and "no code", under every frame. **(my call)**

**Inside the window:**
- the reading explains its own extent;
- every other px is explained by background, or by "other text nearby" at the density measured around training labels
  (today's ρF term, kept);
- line px stay unobserved.

So a reading that stops short of its word pays to explain the leftover text ink as "other text nearby". A reading
that runs onto a neighbour's word gains nothing that "other text nearby" didn't already explain.

**Straightening is a change of coordinates for the reading pass, not a resampling of the evidence.**
- Each page px belongs to one column of the frame's strip.
- The likelihood is computed on the page px.
- The noise model (Task 2's Kanungo noise by distance from the stroke edge) applies unchanged.

**The frame hypotheses are summed.** Their prior is the evidence of step 2, times the reading's own likelihood, so the
ink can favour one grouping over another.

### 4. Read along the strip

A semi-Markov forward pass along the strip:
- the state is the position in the string;
- each letter's width and the gap after it are drawn from measured distributions;
- the emissions come from Task 2's letter shapes, under the noise model.

It sums exactly over every segmentation:
- **for codes,** the string is fixed to each printed form, weighted by p(form | code) from training counts;
- **for "other text",** it is the character bigram over the font's letters;
- **for "no code",** it is background only.

All three use the same widths, gaps and noise, so letter spacing gives none of them an edge.

Ink in the word's measured extent that a reading leaves unexplained is charged under the same noise model, so a prefix
("RH-1" inside "RH-1(D)") pays for the text it ignores.

### 5. Prior

- p(code) from the training reads' consensus counts, smoothed, as now.
- **The detector's kind is evidence for "a zoning code is printed here" against "other text"** (street names, SUD
  names). Its likelihood is measured on training labels. **(my call)**
- Location priors for other kinds of text wait for the street-name and block-number work.

### 6. Calibration and evaluation

- **Calibration:** as the reader spec says. A temperature, and the "no code" and "other" masses, cross-fitted on the
  owner's answers in Task 4.
- **The temperature only rescales the summed log-likelihood,** so each frame's and segmentation's terms are computed
  once and summed per temperature. More temperatures cost almost nothing.
- **Evaluation:** Task 4 unchanged, with point and box labels reported separately.
- **Synthetic cases first** (owner's rule), each with numbers drawn from measured distributions:
  - a code upside down;
  - a code sideways;
  - a point placed off the word's middle by the measured spread;
  - a code spaced wider than the font's advances;
  - a heavy line through a letter;
  - a code beside a street name;
  - a lone "P".

  Each must read right before the real run.

## When it is done (before Task 4)

**(my call)**
1. Every synthetic case reads right.
2. Then, on the training labels, box labels and point labels are each no worse than the current reader's saved records
   (commit c626e46). That means top-1 against the training reads, and mean −log p, measured the same way.
3. Only then does Task 4 score it on the owner's answers.

The old pose-grid code goes: it lives in git history, and its saved records are the comparison.

## What stays

From Tasks 1–3:
- the labels and their held-out blocks;
- the code list and printed forms;
- the line masks;
- the letter shapes and the noise model;
- the guards against training on held-out labels;
- the Evidence / posterior split and the record format;
- the measures.

## What goes

The 2-D pose grid and its FFT correlation over every position, the placement prior over 2-D positions, and the
end-break term. The word's measured frame and extent replace all three. The `solver-text` detector's outputs are
never used by the reader (section 1).

## Left to the builder

- How text px are grouped into a word (connected text along the direction, or a 1-D profile), and how many frame
  hypotheses to keep.
- The straightening (resampling) method.
- The discretisation of the forward pass.
- CPU or GPU.
- Module layout.
