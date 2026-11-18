# Red team: "The map's labels from the scan alone" (spec + plan P1–P6)

Independent and adversarial. Reviewed:
- `docs/superpowers/specs/2026-11-16-text-pipeline-from-the-scan-design.md` (the spec);
- `docs/superpowers/plans/2026-11-16-labels-from-the-scan.md` (the plan);
- the two earlier reader specs;
- the code in `solver/` (prints, trust, labels, placement, energy, solve, walls, wallevidence), `solver/textnet/`,
  `solver/codereader/`, `build.py`, `anchors.py` and `sheetwarp/first.py`;
- the incident note `docs/research/joint-solver/notes/2026-11-16-big-incidents.md`;
- the SDD ledgers;
- the owner's memory rules.

Nothing in the repo was changed. Three read-only measurements were run. Their scripts are in
`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam/`:
`terra_tight.py`, `terra_sheet.py` (makes `terra_codes.png`) and `axis_pca.py`. Copy them somewhere permanent if they
are worth keeping, because the scratchpad gets cleaned.

## Verdict

The aim is right: the labels come from a local detector and reader, and the LLM outputs only train and score. Using
a segmentation detector trained on Terra's boxes is also right, and it already works (98.5% of Terra's zoning codes
are found on the validation area).

Four things are wrong:
- The plan does not deliver its own headline, "no LLM output anywhere in the pipeline". The walls and the first
  placement still use LLM output.
- It bets the reader on a one-off method that has not yet worked here. Standard methods would serve the "any old map"
  goal better.
- It adds a regression head trained on noisy LLM geometry that the existing outputs may already make unnecessary.
- It builds six tasks before measuring the cheapest things that decide whether any of it is needed.

Most of this can be fixed by changing the order and swapping two components. The goal and the scoring rules can
stay.

## Top findings (ranked)

### 1. The pipeline still reads LLM output after P6, and P1 silently changes the walls

- **The walls read Terra's boxes when they run.** The stage's walls use `DEFAULT_CANDIDATES = (street_crossing,
  context)` (`walls.py:72`). That set does not include `text_weighted`. So `wallevidence.border_features` sets
  `terra = TEXT_WEIGHTED not in options` (`wallevidence.py:655`), and `walls.load_inputs` reads
  `prints/terra_boxes.json` as the "Terra letter share" feature (`walls.py:206-213, 260-262`). The `solver-walls` stage
  declares that read (`build.py:185-191`).
  - The plan says "the rest of the solve (walls, the energy, the search) is unchanged" (spec line 146).
  - The detector-based option `text_weighted` exists, but it lost on held-out log loss. Moving the walls off Terra is
    therefore a measured change of its own, and nobody has measured it.
- **The first placement is LLM-anchored.** The warp starts from each sheet's `transform.json`/`refine.json`
  (`sheetwarp/first.py:83-90`). Those come from `anchors.py`'s RANSAC over vision-model block numbers and street-name
  pairs.
  - The spec defers this (lines 55-58). But the end goal ("here's an old map … run every part") cannot start a new map
    without it.
  - Text for "skewing" means exactly this step, because the warp itself takes no text (decision 11-13 evening).
- **The solve stage still reads the prints.** `solver-solve` reads `prints.parquet` (`build.py:197`).
  `solver/labels.py:32` and `streetside.py:141` load it too, but P5's file list leaves out `labels.py`,
  `streetside.py`, `walls.py` and `energy.py`.
- **P1 retrains the walls' detector.** P1 changes `model.OUTPUTS` (7 to 11) and re-records the weights guard.
  `solver-text`'s code dependency is the import closure of `textnet.stage` (`build.py:149-154`), so the walls'
  detector retrains: new head, new initialisation, a new text chance. Its chance feeds `street_crossing`, so the walls
  move.
  - P5 then compares "labels from the scan" with `final` while the walls have also changed. That is two changes in
    one comparison, against the one-problem rule.
- **Fix:**
  - Pin the walls' detector by giving it its own model class or output count, or by freezing its stage.
  - Make "walls off Terra" a separate measured step, using `text_weighted` or the new detector's chance.
  - Add `labels.py`, `streetside.py`, `walls.py` and `energy.py` to P5.
  - Split every stage into a training stage, which may read saved LLM output, and an inference stage, which reads only
    the scan, DataSF data and saved model files. Then the DVC graph of the inference path proves LLM-freedom: a test
    that walks `build.stages()` from `solver-solve` and finds no `saved-reads/` or `solver-prints` path.
  - The plan's monkeypatch test does not prove this. The font, noise model, p(code) and p(form|code) are all learned
    from Terra strings, and the plan doesn't say where they are saved. If `read-labels` learns them in the same
    command, inference does read LLM output.

### 2. The reader bets on a one-off template decoder when the standard recognizer fits the goal better

- **The existing reader's record.** The spec keeps the 11-15 reader's letter shapes, noise model, code list and
  guards, and adds the read-along-the-word pass on top. That reader:
  - learned its font from 959 training labels on one map;
  - read 92% of training box labels the way the LLM reads did (814/885) and 24 of 50 point labels;
  - took 3.6 h for 1,905 labels;
  - has known failures the new pass inherits: wide letter spacing reads as "other", off-centre points fail.
- **The plan reads far more.** It would read every detected box: about 10,000 Terra words on the 13 sheets of 2000
  alone, counted from `terra_boxes.json`. Each box is read both ways, against codes, nearby street names, nearby block
  numbers, a bigram and "none". The only speed target is the spec's "minutes"; nothing measures seconds per box
  before P4 is built.
- **The training data it leaves unused.** Terra's saved answers hold 194,553 word reads (tile duplicates included)
  across 9 maps: 31,703 zoning codes, 108,990 street names, about 37,000 block numbers, in both lettering styles. A
  standard CNN + CTC recognizer could be trained on these crops, with rotation and line-through augmentation; a docTR
  or PaddleOCR recognizer fine-tuned the same way also works (both Apache-2.0). It needs only Terra's strings, which
  the owner allows as training labels.
- **It is still a posterior, and the owner already chose it.**
  - CTC's forward algorithm gives p(string | crop) summed over every alignment. That is the same "sum over every
    segmentation" the read-along pass hand-builds.
  - Scoring each printed form of each code gives a normalised posterior over the code list. Dividing by the training
    prior and multiplying by a location prior is Weinman 2017/2019's map-reading recipe.
  - The 11-13 decision already reads "our own text detector/reader (CNN + CTC) trained on terra's saved boxes". The
    template route was a 11-15 recommendation the owner accepted ("if you think that is the best approach, sure"), not
    an owner rule.
- **It fits "any old map".** One recognizer covers both lettering styles. Changing a map's code list means changing a
  list of strings, not relearning a font. The spec instead defers "other years' lettering" (spec Non-goals).
- **Alternatives:** CRNN (Shi, Bai & Yao 2017) with CTC (Graves 2006); ASTER (TPS rectification, for curved street
  names); PARSeq (2022). Libraries: docTR (DBNet + CRNN/PARSeq, PyTorch), PaddleOCR (DB + SVTR + a 0/180 direction
  classifier), MMOCR, Kraken. All are Apache-2.0, which is compatible with AGPL-3.0. mapKurator is CC BY-NC, so it is
  out, but its method (detector, then recogniser, then gazetteer linking) is the template to follow.
- **What I would do:** spike S2 (below) before writing `reading.py`, scored on the same owner answers. If it reaches
  the 96.7% yardstick's neighbourhood, P4 becomes a recognizer, lexicon scoring and the existing calibration.
  - Keep the generative "lines unobserved" idea as an input: feed the long-line mask as a second channel to the
    recognizer.

### 3. P1's axis and box heads probably aren't needed, and they would learn from noisy LLM geometry

- **Terra's geometry is a guess.** Each box is an LLM-guessed axis-aligned rectangle in tile coordinates plus an
  integer `angle_deg` (`textnet/boxes.py:24-33`). For 30–60° words, `word_rect` can't recover the height and puts in
  14 px (`boxes.py:18, 154-158`). Terra never reports "upside-down" (0 of 10,036 boxes on 2000). Across all kinds,
  1,230 of the 10,036 are "slanted", with LLM-estimated angles.
- **Measured (`terra_tight.py`, 0°/90° boxes):** the overlap (IoU) of each Terra box with the box around the
  letter-sized ink inside it:

  | Kind | Median IoU | Share under 0.5 |
  |---|---|---|
  | Zoning code | 0.80 | 38% |
  | Street name | 0.82 | 18% |
  | Block number | 0.85 | 12% |

  The zoning-code tail is partly letters fused with district lines, which my filter drops, so it is an upper bound on
  the noise. The contact sheet (`terra_codes.png`, 48 random code boxes) shows several boxes covering only part of a
  turned word, and about a quarter of "zoning code" boxes are legend or key entries.
  - So the "box overlap ≥ 0.5 / 0.7" measures in P1 and P2 have a noise floor that must be measured first
    (validation-sets-are-noisy). EAST-style regression to these targets learns the LLM's box habits.
- **Measured (`axis_pca.py`), with no new head:** the axis taken from the second moments of today's held-out text
  chance (`data/solver/text-reader/<sheet>/chance.png`), in a round window that carries no orientation, compared with
  Terra's angle mod 180:

  | Words | Median error | Within 15° |
  |---|---|---|
  | Zoning codes, 3–4 characters | 4.9° | 78% |
  | Zoning codes, 5+ characters | 3.5° | 84% |
  | Block numbers, 3–4 characters | 2.2° | 90% |
  | Street names, 5+ characters | 3.7–4.4° | 80–85% |
  | Slanted street names | 3.7° | 85% |

  These numbers include Terra's own angle errors and neighbouring text inside the window.
  - Where it fails is 1–2 character words: a lone "P" is 4% within 15°. A word that short has no line axis, so an
    axis head trained on Terra's guessed angle won't fix it either. What such a word needs is its letter orientation
    (0/90/180/270), which a recognizer can score as four weighted hypotheses, or a small orientation classifier can
    give (PaddleOCR's approach).
- **Mixed granularity.** 11.5% of street-name boxes and 59% of "other" boxes are multi-word phrases, while words like
  "ST." are boxed alone. A per-pixel "distance to my box's sides" target is inconsistent across words and phrases.
- **Instead:**
  - Get boxes the DBNet/CRAFT way, from the segmentation that already exists. Add one shrunk-text "kernel" channel if
    touching words merge, which is the DBNet/PSENet fix and robust to loose boxes. Take the components'
    minimum-area rectangles or moments.
  - Train any new geometry target from the ink inside Terra's box, not from Terra's numbers.
  - Measure S3 and S4 (below) before P1. Note also that the 4,000-step schedule already left the direction head
    half-trained, and that 8,000 steps fixed it (86–92% within 45°, R1 debug). A new head adds another output to
    train, schedule and score.

### 4. The plan works on the smallest error source, on the one map with an answer key, and its comparison is swamped by search noise

- **Where the errors are.** The census of the 44 incidents of 20+ lots:

  | Cause | Wrong lots | Share |
  |---|---|---|
  | Search | 2,873 | 56% |
  | Walls | 1,368 | 27% |
  | Placement and labels | 873 | 17% |

  Within placement and labels, misreads are 178 lots and unlabelled areas 148. Energy and wrong lots rank-correlate
  only 0.30 across variants, and one 1,838-lot search incident (#1) flips on small changes.
  - P5 judges "labels from the scan" against `final` by the six measures. A reader change that moves #1 either way
    will swamp the label signal.
  - Land the compound search (C1) first, or compare under the oracle search. Report label-level measures (labels
    right, label-term energy per label, incidents) with bootstrap intervals, not wrong lots.
- **"Somewhat worse" has no unit.** The owner accepts a somewhat worse reader, but the plan never measures how map
  quality depends on reader quality. S1 (below) gives that curve. Without it, P6 can't tell the owner whether 90%
  reader accuracy costs 50 lots or 3,000.
- **Built only for 2000, with no transfer test.** Every learned piece is fitted on 2000:
  - the code list is the 2000 table's (`solve.py:93`: `readings(q, table_vocabulary(C.lots))` drops readings outside
    it);
  - p(code) and p(form | code) come from 2000's LLM reads;
  - the font is 2000's;
  - the calibration is cross-fitted on the 2000 owner answers and the table;
  - `energy.TOP_READING_FIT = (0.128, 1.085)` was fitted on the LLM trust (`energy.py:12`) and must be refitted for the
    new reader, but the plan doesn't list `energy.py`;
  - lot-size PMI and placement were fitted on the 2000 table.

  Learning once on 2000 is the owner's rule. Never checking transfer is not.
- **The free transfer test.** `1996-2000-zoning` is a second scan of the same drawing (`labels.TWIN_SERIES`). The
  table judges it, it has warped sheets, and Terra never read it. It is the "new map with no LLM text" test. The
  plan's solver is hard-wired to `paths.SERIES` = 2000 and never runs on it.
- **Other eras.** The 1960–1975 maps print almost no block numbers (Terra found 1–5 per map) and use other codes. The
  block-number hypothesis and the 2000 code list don't carry over. The map's own legend lists its codes, and the
  detector already finds legend text (see the contact sheet). Reading the legend is a local, deterministic source of
  each map's code list, and the plan should at least name it.

### 5. Design gaps against the owner's rules

- **A cutoff decides what is a label.** "A printed label is a detected box whose posterior gives a zoning code real
  mass" (spec line 138) is an in/out rule. The energy has no "this box is not a label" term: `energy.label_costs`
  (`energy.py:30-63`) treats every reading's label as a printed code, and P always sums to 1 over codes. With about
  10,000 boxes, the choice is a threshold or thousands of spurious charges.
  - **The principled form:** charge each box −log((1 − p_lab)/K + p_lab·(π·P + (1 − π)/K)), with p_lab the box's
    posterior mass on "a zoning code is printed here". A box that is surely a street name then costs nothing.
    (`add_codereader` used p ≥ 0.05, also a cutoff.)
- **The location priors need the placement.** "The centrelines within reach" and "the block the box sits in" assume
  the sheet is already placed. On a new map that is circular, so this reader can't supply the first placement (finding
  1).
  - **Invert the order:** read block numbers as digits with no prior, and street names against the city's whole name
    list. Feed those reads to the existing `anchors.match_labels` / `rough_fit` (RANSAC) for the first placement. Add
    the location priors only after that.
  - **A measure that serves the end goal:** place each 2000 sheet from local reads alone and report the residual
    against today's placement.
- **The miss rate can't be measured "by kind and map".** MISS_WEIGHT is to come from "the other passes and Sol's
  points with no Terra box". The other passes exist only for 2000, and Sol labels only zoning codes. Say which numbers
  are measured and which are carried over.
- **The legend must stay out.** About a quarter of random Terra "zoning code" boxes are legend or key entries. P3 must
  apply the mapped-area mask that `prints.py` uses (`sheets.png > 0`), or those entries become labels. That mask rests
  on the hand frame cuts (`inputs/merged-page/2000-digital-zoning.json`), a per-map hand input the end goal will need
  to automate.

## The attack questions, briefly

**1. Right approach, or a reinvention?**
- The detector is fine: a U-Net segmentation, the CRAFT/DBNet family.
- The P1 geometry head is EAST (2017), the older regression design that DBNet and CRAFT replaced because regression
  struggles with long, spaced and curved text.
- The reader re-derives Kopec & Chou's Document Image Decoding and Ocular (Berg-Kirkpatrick 2013), one font at a time.
- A professional OCR or GIS engineer would:
  - fine-tune docTR or PaddleOCR (DBNet + CRNN/PARSeq + an orientation classifier) on Terra's crops from all 9 maps;
  - score lexicons by CTC;
  - link to a gazetteer (DataSF names and blocks) only after a RANSAC placement;
  - use location priors à la Weinman;
  - follow ICDAR MapText 2024's tasks: detection, linking, recognition.

**2. Does it serve the end goal?**
- Partly. It removes the LLM from the labels of one map.
- It leaves the LLM in the walls and the first placement.
- It runs nothing on a map without LLM reads.
- It works on 17% of today's errors while search is 56%.

**3. Hidden dependencies**
- Terra boxes in the walls (`walls.py:260`).
- The LLM-anchored start placement (`anchors.py`, `sheetwarp/first.py:83-90`).
- `prints.parquet` in labels, streetside and placement, and in the solve stage's reads.
- The 2000 table as the code vocabulary (`solve.py:93`, `trust.vocabulary`).
- `TOP_READING_FIT` (fitted on LLM trust).
- The reader's font, priors and forms, learned from Terra strings, with no saved-model boundary.
- The location priors, which need a placement.
- The calibration and placement models, fitted on the 2000 table and owner answers and never checked on another map.
- `solver-text-reader`'s stage deps include `solver-prints` and the code-check answers (`build.py:168-176`), so a new
  map without them can't build the inference outputs.

**4. Feasibility and risk**
- Noisy box and angle targets (measured above).
- Mixed word/phrase granularity.
- The reader's speed on about 10,000 boxes, both ways, against a 3.6 h history, with no budget.
- A single font.
- 1–2 character labels ("P"), which have no axis.
- Two seeds per schedule trial on the GPU.
- Validating against search noise.
- What's fine: determinism (the same seeded discipline works for a CTC model) and treating line px as unobserved
  (keep it, as a mask channel).

**5. Ordering:** below.

## What to measure first (no new model needed for S0, S1, S3 and S4)

| # | Spike | What it decides | Measure |
|---|---|---|---|
| S0 | Solve with the existing template reader's posteriors as the ONLY reading. All 1,905 labels at Terra/Sol positions are already scored: `scratchpad/codereader/task3/full_fix1held.records` + `full_fix1rest.records`. It is one more variant in `scripts/incidents/placement_labels.py` (which already has `add_codereader`). | Does the map survive a ~92% reader? | The six measures + labels right, every lot, with intervals, held out by quarter. Wrong lots reported but not trusted (search noise). |
| S1 | A degradation curve: the LLM readings with (a) prints dropped at the detector's measured miss rate by kind and size, and (b) top readings swapped at q = 2/5/10/15% using a confusion matrix measured on training labels. Run under C1 search if it has landed. | What "somewhat worse" means in lots and labels. Is the reader even the lever? | The same measures against q. |
| S2 | A CNN + CTC recognizer (a small CRNN, or docTR's crnn/parseq) trained on Terra word crops from the six training maps. 2000's held-out quarter and the answered blocks are left out. Crops are made upright from Terra's box and angle (spike only). Read the owner-answer crops at 0/90/180/270, score the code list by CTC, cross-fit a temperature. Also read the 1972/1975/1980 Terra strings for transfer. | Template decoder or standard recognizer? | Top-1, top-3, mean −log p, reliability on the owner's answers (box and point separately), with intervals, against 96.7% and the template reader. Seconds per crop. |
| S3 | The axis from the second moments of the existing chance (done roughly here); then boxes from connected components or minimum-area rectangles of the 8,000-step reader-detector chance. | Are P1's axis and box heads needed? | Recall of Terra's words by kind (overlap curve); axis error against angles re-measured from the ink. |
| S4 | Terra's geometric noise against the ink (done roughly here; refine by handling line-fused letters). | The noise floor of every box and axis measure. | IoU and edge offsets by kind; an eye sample. |
| S5 | Walls with `text_weighted` (detector chance) in place of Terra's letter share. | The cost of taking the LLM out of the walls. | Walls held-out log loss and the six measures. |

Then build, in this order:
1. The search fix (C1), so that later comparisons aren't noise.
2. The detector boxes the S3/S4 way.
3. The S2 recognizer, if it wins, with lexicon scoring and a "not a label" mass in the energy.
4. Labels and walls moved off the LLM, as two separate measured steps, each with its stage split into training and
   inference.
5. TOP_READING_FIT and the calibration refitted.
6. The first placement from local block-number and street-name reads through `anchors.rough_fit`.
7. One end-to-end run on `1996-2000-zoning`, the map nobody's LLM read.

Each step gets its synthetic case first (one-problem-synthetic-tests), with numbers from the measurements above.

## What the plan gets right

- Terra and Sol as training and scoring only.
- Terra's misses treated as unlabelled.
- Every detector output scored.
- Hold-outs by quarter and answered blocks.
- Noisy-set reporting with intervals and tie-breaking by eye.
- Value before verification.
- The leader-move key bug fixed with a failing test.
- `label_fixes.csv` kept out of inference.

Keep all of these in any revision.
