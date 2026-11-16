# The map's labels from the scan alone

Design, 2026-11-16. It supersedes `2026-11-16-code-reader-read-along-the-word-design.md`, which used Terra's boxes and
Sol's points as evidence for each word.

The prior-times-likelihood reader of `2026-11-15-code-reader-design.md` stays: its letter shapes, noise model, code
list, guards and measures. The owner set the goal in conversation. Calls the owner did not make are marked
**(my call)**.

## Why

The owner, 2026-11-16:
- "We should be able to say, 'Here's an old map from some year of some city.' The tool should be able to automatically
  run every part of the scanned map to a modern correctly coded map, running locally without any external calls."
- "responses from a paid LLM or even a free LLM should not be part of this deterministic pipeline." Terra's saved
  responses may train and validate the text detector, and nothing more.
- "I don't actually care what the text is. It's only useful so that we can do the zoning boundary assignment correctly
  and, I guess, for the skewing. … The text reading and the text box finding are only in service of the final map."
- A code reader that scores somewhat worse than the LLM reads is an acceptable price.

Today the 2000 solver takes its printed labels and their readings from saved LLM reads:
- the prints from Terra's and Sol's answers;
- five saved reads per print;
- a trust fit combining them.

That has to go. The solver's labels must come from the scan, through a detector and a reader that run locally.

Two lessons from 11-15/16 shape the method:
- **Score every output something will read.** The speed loop judged only the text chance. The 360° reading-direction
  output was left half-trained at 4,000 steps, and unusable.
- **Step back to what is needed before tuning.** A word's box needs its line's orientation (an axis, modulo 180°) and
  its extent. Which end the word starts at can be left to the reader.

## Goals

1. **Find the words on the scan:** a rotated box per word (centre, axis modulo 180°, length, height), with its kind
   (zoning code, block number, street name, other) and a probability, alternatives kept.
2. **Read the zoning labels:**
   - for each box, a posterior over the code list, "other text" and "no text";
   - read both ways along its axis;
   - street names and block numbers are modelled only so that they explain their ink and are not taken for codes.
3. **Feed the solver:** its printed labels (where, and the posterior over codes) come from goals 1 and 2. The saved LLM
   reads, Terra's and Sol's prints and the trust fit over several readers leave the solver's inputs.
4. **Judge it by the map:**
   - the solve's six measures, held out by quarter as today, against the current solve on the LLM reads;
   - the code reader on the owner's answers;
   - the detector against Terra's boxes, allowing for Terra's misses.

   Somewhat worse than the LLM reads is acceptable.
5. **Deterministic and local:** no external calls. Detector trials in minutes; reader and solve runs in minutes.

## Non-goals (now)

- Writing the text out as a product (a text dump). Text is internal to the map.
- **A new map's first placement.** The warp itself uses no text: it fits ink to DataSF street geometry and block
  outlines. But it starts from each sheet's placement by the old pipeline, whose anchors were LLM-read block numbers
  and street names. On a map with no LLM, the first placement must come from geometry alone (matching the street grid)
  or from this reader's street-name and block-number hypotheses. That decision comes later.
- Other years' lettering and code lists. The letter shapes and code list come from the 2000 map; the serif years need
  their own, later. The detector is trained on both styles already.
- Labels on two lines.
- Leader lines beyond what the placement already learns.

## Anti-goals

- **Any LLM output when the pipeline runs on a map:** Terra, Sol, the saved reads, paid or free.
- **The owner's hand corrections as an input to the solve.** `label_fixes.csv` and its leader moves are training and
  scoring data, like Terra's. **(my call)** A new map has none.
- **Treating a detection outside Terra's boxes as wrong** where the other passes and the eye say there is text.
- **Brute-force sweeps** over angles, positions or sizes, and hard cutoffs as decisions.
- **Training on what is scored:**
  - the held-out quarter;
  - the validation area and the test drawings;
  - the owner's answered labels and their blocks.
- **The zoning table's code at a label's spot,** anywhere but in scoring.

## 1. The detector

The same U-Net on the ink, at half size. Its outputs per px, learned from Terra's word boxes:

1. **Text:** the chance a px is in a word, as now.
2. **Kind:** as now.
3. **The line's axis:** the doubled angle (cos 2θ, sin 2θ), the same for a word and its upside-down copy. **(my call)**
   It replaces the 360° direction output.
4. **The word's box, seen from each of its px:** the distances to the box's four sides along its axis, EAST-style.
   Each text px votes for one rotated box. **(my call)**

**Training** uses the generalised hold-outs built in R1 (`SheetData.held`: a quarter, the validation area, the answered
blocks). **Terra's misses** are unlabelled, not "not text":
- ink outside every box counts as "not text" in the loss only with the probability that unboxed ink is not text;
- that probability is measured from Terra's miss rate: the share of the other passes' boxes and Sol's points with no
  Terra box, by kind and map. **(my call)**

Any printed word or number is text.

**The schedule** is chosen by every output's own measure, on the validation area and the test drawings:
- words found and long-line ink called text;
- the axis error against Terra's angle modulo 180°;
- the box error: the overlap of the voted box with Terra's box;
- kind agreement.

"The shortest schedule that passes" means passing all of them.

## 2. Word boxes

Each text px's vote is a rotated box. Votes that agree are grouped into a word, as EAST's merge does, but keeping
probabilities instead of a hard suppression:
- each box carries its probability and the spread of its votes;
- where votes split (two words touching, a word and a line), the alternative boxes are kept with their weights.

## 3. The reader

The reader takes only detected boxes. For each box and each of its alternatives:
- **Both directions along the axis.** Upright and upside down are two hypotheses, and the likelihood picks.
- **The read-along-the-word pass** (11-16 design, sections 3–4):
  - the left-to-right sum over letter positions and spacing;
  - one common window of page px;
  - "other text nearby" at the measured density;
  - line px unobserved;
  - all hypotheses on equal footing.
- **The hypotheses:**
  - **A zoning code:** the code list (the April 2000 table's code names, not their places), with p(code) and each
    printed form weighted by counts from the training reads.
  - **A street name:** the names of the street centrelines (DataSF / OSM) the box runs along, weighted by distance and
    angle, plus their printed forms.
  - **A block number:** the block the box sits in and its neighbours (parcels' block numbers).
  - **Other text:** a character bigram over the font's letters.
  - **No text.**

  The street name and block number hypotheses exist to explain their ink, so a street name is not read as a code. Only
  the code posterior is used downstream.
- **The kind is part of the hypothesis:** p(kind) comes from the detector's kind output, measured on training boxes.
  Every kind keeps some mass everywhere.
- **Calibration:** a temperature, and the "other" and "none" masses, cross-fitted on the owner's answers.

## 4. The solver's labels

A printed label is a detected box whose posterior gives a zoning code real mass. The labels stage builds, from the
boxes and their posteriors alone:
- the readings, per held-out quarter as today;
- the placement features (where each label belongs, from its box);
- the street-side and lot-size features.

With one reader there is nothing to combine. The trust fit becomes the reader's calibration, learned from the table and
held out by quarter as the trust fit is today. **(my call)** The rest of the solve (walls, the energy, the search) is
unchanged.

## Scoring

The scoring sets are not all correct. The owner: "If two approaches have similar scores but one is barely ahead of or
barely behind another one, it doesn't necessarily say something about the accuracy of the method you're testing. It
may, in fact, just say something about our validation set accuracy." So:
- **Error rates are measured.** Each set's own error rate (Terra's boxes and strings, the table, and the owner's
  answers' flagged cases) is measured where a second source allows: the other passes, the owner's answers, the
  eye-check sample. It is reported beside every score.
- **Differences are reported with their noise:** a bootstrap interval over labels or areas. A difference inside both
  the interval and the set's measured error is a tie, not a win.
- **Ties are broken by looking.** The cases where two methods disagree go on a page, with the scan, and the owner or
  the eye decides which is right. Otherwise the more principled method is kept (owner's rules), never the one barely
  ahead on the number.

The same measures are used for every comparison, on what the fits didn't see:

- **The map:** the solve's six measures (every-lot column), with labels from the pipeline against the current solve on
  the LLM reads. Reported, not a stop gate (owner).
- **The code reader:** top-1, top-3, mean −log p and a reliability table on the owner's answers (matched to boxes by
  position), with box-found and box-missed answers reported separately. The saved LLM reads' 96.7% is the yardstick.
- **The detector:**
  - **Recall:** of Terra's words, by kind, as a curve over the overlap threshold.
  - **Precision:** allowing for Terra's misses. Unseen boxes are judged once, on a sample page by the owner's eye, on
    the final detector.
  - **The axis and box errors** on matched boxes.

## What stays, what changes

- **Stays:**
  - the detector's network, recipe options and trial machinery;
  - the generalised hold-outs and the reader-stage scaffold from R1 (commit ce828df);
  - the code reader's letter shapes, noise model, code list and printed forms, guards and records;
  - the read-along-the-word pass;
  - the solver's walls, energy and search.
- **Changes:**
  - the detector's direction output becomes the axis and box outputs, with a loss that treats Terra's misses as
    unlabelled;
  - the reader reads detected boxes;
  - the labels stage builds from boxes and posteriors.
- **Goes from the pipeline's inputs:**
  - the saved LLM reads, the prints from Terra's and Sol's answers, the trust fit over several readers;
  - `label_fixes.csv` as an input (it stays as training and scoring data).

## Left to the builder

- The box parameterisation, and the losses for the axis and box outputs.
- How votes are grouped into words, and how alternatives are weighted.
- Thresholds that are numerical tolerances (bounded, reported), not decisions.
- Whether the four quarter models stay, or one held-out scheme serves both the detector's scores and the reader's.
- The order of the work, with the detector first.
- Module layout.
