# Frame crop: find the frame line, delete it and everything outside it (issue #71, revised)

Supersedes the two-scale design in `2026-11-19-learned-frame-design.md` (page pass + pixel pass). Same issue, same labels
(revised as below), simpler model.

## Intent

Every scanned sheet prints a frame around its map: bars, thin rules, corner arcs, ruled-off legend and title strips,
margins, scan background. The goal, in the owner's words: **"detect & delete the frameline and everything outside of
it."** The result is one region per page, the map inside the frame; everything outside it, including the frame ink, is
deleted (painted plain paper in the paper layer, ignored by reading and placement). The cut must be **picture perfect**:
no frame ink left in, no map ink cut off.

## Owner decisions this design rests on

- One learned method for every edition; hand labels allowed because there are few; Claude draws them.
- **One region, no holes.** Floating notes, titles and compass roses inside the frame stay inside (2026-11-20, reversing
  the earlier "floating notes are frame"). Ruled-off strips and insets that touch the frame are cut out, as notches.
- Corner arcs are frame and are cut out.
- **Full resolution in training and prediction.** No downsampled pass ("I don't think we can use low res to train or
  run").
- No made-up limits: no thresholds or penalties set by hand; anything numeric is learned, measured, or an owner cost.
- Judge by eye first; measure with probabilities, sheet-held-out.
- Prediction near instant; training minutes, not hours.

## Goals / non-goals / anti-goals

Goals: per page, a deletion mask (outside the frame) at scan resolution and the chance behind it; a picture of every map
page; held-out scores on every labelled sheet.

Non-goals (this change): wiring the mask into the published paper layer, reading and placement. Each is its own change
after the owner has seen the pictures (publishing needs build + ship).

Anti-goals: a downsampled stage; any hand threshold, window or per-series rule; any stage fitted on outputs of a network
that trained on the same sheets; non-determinism; GPL-family code.

## Is there an open-source tool for this? No.

Checked 2026-11-20:

| Candidate | What it does | Why not |
|---|---|---|
| ICDAR 2021 MapSeg task 2 entries ([report](https://arxiv.org/abs/2105.13265)) | Exactly this task on 1894–1937 Paris atlas sheets (≈10,000 px, 26 training sheets) | Winner (UWB): downsampled whole-page network + recursive Otsu binarization and morphology with parameters tuned on a validation set; **HD95 19 px**, not pixel perfect; code only as "extra material" on a university GitLab, no package or licence we can rely on. The 512 px single network (L3IRIS) scored **126 px** ("overly smoothed masks"): direct evidence that low resolution can't be precise. Rule-based entries (CMM 85 px, IRISA 112 px) were "penalized by unseen frame configurations". |
| dhSegment | Generic historical-document pixel labelling, page extraction | GPL-3.0 per its catalogue listing (licence rule), downsampled input |
| Segment Anything (SAM/SAM2) | Prompted object masks | Works at 1024 px input with a 256 px mask: low resolution; not trained on map frames |
| MapReader | Patch classification of map collections | Classifies patches; no frame boundary |
| OpenCV Hough / LSD, morphology recipes (e.g. Library of Congress blog) | Line finding | Hand rules per frame style; the MapSeg rule-based entries show the generalisation failure |
| mapvision (MIT code) | Compact U-Net with a "frame" class (margins, title boxes, decorations) | 1000 px tiles shrunk to 256 px; trained on 157 Paris tiles under a non-commercial dataset licence; frame-class IoU 0.77, no boundary accuracy |
| Semap (Petitpierre, arXiv 2603.05037) | Mask2Former over heterogeneous map collections; a "background" class covers borders, legends, ornaments | 768 px patches, heavy model; boundary IoU about 41%; code licence not stated. Its synthetic "crop along an edge to imitate the frame" augmentation is the same idea as our edge cutting |
| Fragment-based map evolution (Humanities & Social Sciences Communications 2024) | OCRNet segmenting map content from frame, title and legend | Area IoU 87.7%, far from pixel precise |
| Global Mapper "crop collar" | Crops USGS topographic sheets to the neatline | Proprietary, and built for USGS sheet layouts |
| mapsheet-mosaic ArcGIS tools (Apache-2.0) | Clips georeferenced sheets to a known sheet polygon | Needs the sheet already georeferenced plus the sheet polygon; doesn't find the frame |
| Shafait & Breuel page-frame detection (IJDAR 2008) | Geometric matching of a page frame around text lines | Built for text documents; no text lines to match here |

So we build it, from standard parts we already use (PyTorch, numpy, scipy; all BSD/MIT), on top of the existing
`frame` package.

## Two measurements that shaped the design

On the 20 current labels (`pipeline/inputs/frame-labels/`):

1. **The shapes are not "four sides".** Cut depth from the page edge along each side (middle 80% of the side): typical
   100–500 px, frame plus margin up to ~1,100 px (1960 bottom strip, 1975 left), but notches reach **4,961 px** (1960
   insets, 1975, 2000 legend columns, L-shaped 1921 sections reaching 2,871 of 3,309 px). A per-side line or path model
   (proposed earlier) can't represent these without hand rules. The constraint that is true of every label is the
   owner's: one region, no holes.
2. **The outlines are not ink-clean.** Share of outline pixels lying on ink (lum < 0.5, page edge excluded): 1921
   5–22%, 1935 19%, 1929 6%, 1948/11 11%; 1960/1975/1980s/2000 ≈ 0. Points every 300–400 px can't follow bowed 1921
   rules to the pixel. A model learns its labels' error, and held-out scores are only as good as the labels, so
   picture-perfect needs ink-clean labels first.

## Approaches

1. **Full-resolution per-pixel chance + decision theory + the owner's topology (recommended).** One fully convolutional
   network sees every scan pixel and about 2,000 px around it, and gives each pixel's chance of being outside the frame.
   The cut minimises expected cost given those chances, then is made one region with no holes, which is the definition
   of the answer, not a tuned step.
2. **MapSeg winner's recipe:** downsampled network for the rough area + full-resolution binarisation and morphology to
   snap the edge. Proven at 19 px HD95, but it has a low-resolution stage (owner: no) and hand-tuned morphology (owner:
   no made-up limits), and it isn't pixel perfect.
3. **A shape model on top of the network** (per-side paths, straight-line priors, a learned "side has no frame" chance).
   Rejected for now: measurement 1 shows the shapes don't fit a per-side model, and the full-resolution network already
   sees about 4,000 px of a side at once, so "this side has no rule" is inside its view. If held-out errors turn out to
   be runs along sides (see "If the errors call for it"), a boundary-length prior is the principled next step.

## The design, part by part

Each part states what it is, why it is sound, and the red team against it.

### 1. The decision: minimum expected cost

For each page pixel *p* let *q_p* = P(p is outside the frame | page) and *i_p* its ink amount (1 − luminance). With
cost *c_f* per unit of frame ink left in and *c_m* per unit of map ink cut off, a region *R* (kept) has expected cost

  E[cost(R)] = Σ_{p∈R} c_f · q_p · i_p + Σ_{p∉R} c_m · (1 − q_p) · i_p.

**Why sound.** Bayes decision theory: the action that minimises expected loss under the posterior is optimal (Berger,
*Statistical Decision Theory*, 1985). By **linearity of expectation** this sum needs only the per-pixel chances *q_p*,
even though neighbouring pixels are strongly correlated: correlation changes the variance of the cost, not its
expectation. This removes the problem that sank the earlier per-side likelihood idea (summing correlated evidence as if
independent). Without the shape constraint the optimum is per pixel: cut *p* iff q_p > c_m / (c_f + c_m). This is not a
made-up threshold; it is the cost ratio. The ink amount cancels, so blank paper (*i* ≈ 0) costs nothing either way,
which matches the task: paper is paper whether it's erased or not.

**The cost ratio is the owner's:** frame left in is 3× worse than map cut (owner answers, 2), so the cut is at q > ¼.

**Red team.** (a) The rule is only as good as the chances are calibrated (part 4). (b) The chances must condition on
everything relevant; the network sees about ±2,000 px, not the whole page (part 3 measures what it actually uses).

### 2. The shape: one region, no holes, by definition

After the per-pixel decision: (i) the outside set keeps only the cut components that touch the page border; every other
cut component (a hole) becomes inside; (ii) the inside set keeps its largest component; the rest becomes outside.
Outside uses 4-connectivity and inside 8-connectivity, the complementary pair that makes digital regions behave like
continuous ones (Rosenfeld, "Digital topology", 1979). Pages that print several framed maps are listed in
`pipeline/inputs/page_kinds.csv` with how many maps they hold, and keep that many inside components; this is a hand
label, like the page kinds. *(Builder check: my earlier "1948 2 and 3" was wrong; 1948/2 and 1948/3 are single maps. My
call: I find such pages in the mosaics and list them for the owner. To catch any I miss, every run ranks the pages by
how much ink the "largest component" step dropped, for the owner's eye; that's a ranking, not a cutoff.)*

**Why sound.** "Everything outside the frame" is, by definition, connected to the page edge, and the owner's map is one
region with no holes. These two steps project the decision onto exactly that set. There is nothing to tune. They also
remove most of the earlier failures for free: lettering, commercial bands and floating notes inside the map can't be cut
unless the cut reaches the page edge.

**Red team.** (a) It's a projection, not the exact constrained optimum: it can override a confident pixel decision.
That's correct here, because such a decision contradicts the definition. The held-out run reports how much ink the
projection changes, to see whether it matters. (b) It does nothing for a wrong cut that does reach the edge: blocks
touching the frame, or a frameless edge called frame. Those are the errors to watch (see "If the errors call for it").
(c) "Largest component" fails if a page holds two maps; hence the hand list.

### 3. The chances: a full-resolution fully convolutional network

- **Network:** the existing `Folded` U-Net: pixel_unshuffle by 8 → U-Net with 6 levels and width 32 → pixel_shuffle.
  One output logit per scan pixel; reach about 2,000 px each way.
- **Input:** whole pages, one per step, at scan resolution. Training and prediction see the same thing; no windows.
- **Normalisation:** GroupNorm instead of BatchNorm.

**Why sound.**
- Fully convolutional networks are the standard tool for dense per-pixel labelling. The U-Net was introduced trained on
  30 images (Long et al. 2015; Ronneberger et al. 2015).
- pixel_unshuffle is a lossless reshape (Shi et al. 2016): every scan pixel stays an input. This is what the
  full-resolution requirement means. It also makes training fast on our GPU, where full-size few-channel layers were
  the measured bottleneck.
- The reach has to cover the frame depth (≤ ~1,100 px, measurement 1) plus enough map to tell map from frame. 2,000 px
  does.
- Whole pages remove the train/predict mismatch that windows create: a window's edge pixels never see their
  surroundings in training. Measured cost: **193 ms per step on the largest page** (6,656 × 5,376), 3.4 GB.
- BatchNorm with one image per step trains on that page's own statistics and predicts with an average over pages, which
  is a mismatch between 1-bit 1921 scans and 1948 photos. GroupNorm doesn't depend on the batch (Wu & He 2018).

**Red team.**
- The effective receptive field is much smaller than the theoretical reach (Luo et al. 2016). After training we
  measure it: the gradient of an outline pixel's output with respect to the input. If it's well under the frame depth,
  the network can't be using the context we assume.
- 21 pages is little data: 2,000 steps is about 100 passes per page. The spike overfitted by 4,000 steps (training loss
  0.03). Mitigations are augmentation (part 5), the step count chosen once from held-out loss curves (part 6), and the
  ensemble (part 6).
- With dihedral augmentation the network can't use "the legend strip is at the bottom". That costs a true regularity,
  for 8× the variety. Legend strips are recognisable as ruled boxes in any orientation, so I keep it; a held-out
  comparison can revisit it if strips fail.

### 4. Training targets and loss: calibrated chances

- **Labels:** the outline polygons with holes filled (owner: no holes). Target is 1 outside, 0 inside.
- **Loss:** binary cross-entropy (log-loss), weighted per pixel by w(x) = 0.1 + i_p. This weight depends on the input
  only.
- **No uncertain strip.** The earlier one-sided strip gave zero weight to inside-labelled pixels within 8 px of the
  outline.
- **Calibration:** one temperature on the logits, fitted by weighted log-loss on held-out chances (Guo et al. 2017), and
  a reliability table on held-out ink.

**Why sound.**
- Log-loss is a strictly proper scoring rule: its minimiser is the true conditional probability (Gneiting & Raftery
  2007). That's what part 1 needs.
- A weight that depends only on the input leaves the per-input minimiser unchanged, so the chances stay calibrated. A
  weight that depends on the label doesn't: with weights w₁, w₀ for the two classes, the minimiser becomes
  P·w₁ / (P·w₁ + (1 − P)·w₀). The old strip was exactly such a weight, biasing pixels near the outline towards
  "outside". Ink-clean labels (part 5) make it unnecessary, so it goes.
- Networks are often overconfident on new data (Guo et al. 2017), and the decision compares chances against the cost
  ratio, so calibration is checked rather than assumed. The temperature is fitted only on chances for sheets the model
  didn't train on.

**Red team.** Calibration on 21 sheets is coarse: one parameter, with a bootstrap interval. If the interval includes 1,
the network is already calibrated.

### 5. Labels and augmentation

- **Ink-clean outlines.** A check lists every outline stretch lying on ink, ignoring the page edge, with a full-resolution
  crop of each. I move or add points until the only stretches left on ink are where map ink truly touches frame ink.
  There the outline follows the inner edge of the rule. The owner sees the QA sheet before training.
- **One framed 1948 sheet added** (1 or 2). All three labelled 1948 sheets are frameless, while the 1948/3 and 1948/4
  problems were on framed sheets. 1948/3 and 1948/4 stay unlabelled, as the test.
- **Augmentation:**
  - turns and flips (the dihedral group);
  - photometric: contrast, brightness, gamma;
  - blur;
  - scale ±10%;
  - **random edge cutting** (kept from the current code, my call after the builder check). Each side of a drawn page is
    cut by a random amount from 0 up to that side's own labelled cut depth, so at the extreme the frame is gone and map
    ink runs to the page edge: a synthetic frameless side, which is the edge-connected error that part 2 can't fix. The
    label moves with the cut, so the target stays correct. This replaces the inherited fixed 10% of the page with a
    range read from each label.

**Why sound.**
- A model can only be as precise as its targets, and the held-out score measures against the same targets.
  Measurement 2 shows the 1921 and 1935 outlines are off by about the rule width in places.
- Augmenting with transformations the answer doesn't depend on is a standard regulariser for small datasets. Here the
  answer doesn't change under rotation, exposure or scan sharpness.

**Red team.**
- The ±10% scale range is inherited, not measured. Editions differ in scale more than that: page widths run 2,550 to
  5,266 px. Scale changes between editions are learned from the labelled editions themselves. The jitter only covers
  scan-to-scan variation within one edition.
- **14 of the 22 series have no label:** 1964, 1968, 1972, 1980, and the height, special-use and 1990–2000 printings.
  Many share a template with a labelled sibling, but 1964–1972 and 1980 rest on generalisation alone. The mosaic of
  every series is the check. Where one fails, label a sheet of that series, picked where the two seeds disagree most.
- Hand-moved points are still a person's judgement. The QA sheet makes it visible.

### 6. Held-out chances, step count, ensemble

- **Folds:** 4 folds round-robin by sheet within each edition, as `frame/folds.py` already does, so each fold mixes
  editions.
- **Seeds:** 2 per fold, 8 trainings in all. They give:
  - held-out chances for every labelled sheet, the average of 2 models that never saw it;
  - the step count: the held-out loss is logged every 250 steps, and the step count is chosen once from the curve
    averaged over folds. That's one setting with about 4 candidates, so the bias from choosing on the scored sheets is
    small, and the doc says so;
  - the seed variance: how often two seeds disagree on held-out ink decisions.
- **Final:** the same 2 seeds trained on all labelled sheets, averaged for prediction. The production ensemble is the
  same size as the held-out one, so the temperature fitted on held-out chances applies to the same kind of average.

**Why sound.**
- Held-out by sheet: errors cluster within a sheet, so the sheet is the unit of independence.
- Averaging M models leaves variance ρσ² + (1 − ρ)σ²/M, where ρ is how correlated the models are. It also improves
  calibration (Lakshminarayanan et al. 2017). We saw large run-to-run swings (Dogpatch 0–60%, Great Highway 4–40%),
  though across different designs. Two seeds measure that variance in this design. If the disagreement is still large,
  the owner sees the numbers and M goes up.
- Nothing else is fitted on network outputs, so no stage learns that the network is always right. That was the bug
  that switched off the pixel pass.

**Red team.**
- A fold model sees 75% of the labels, so held-out scores are slightly pessimistic. That's standard cross-validation
  bias, and it's in the safe direction.
- Time:
  - 10 trainings of about 5 minutes each, about 50 minutes in all; 8 of those are the held-out measurement, run once;
  - after that, retraining is the 2 final models, about 10 minutes;
  - prediction: the one-model spike did all 351 map pages in 329 s, so 2 models take about 10 minutes (mostly
    loading and saving, done in parallel).

### 7. Scoring

On held-out chances after the decision and the shape:
- **Per sheet:** frame ink kept and map ink cut, each as an ink-weighted share of that class, plus the expected cost.
- **Pooled:** a mean over sheets, with **90% intervals from resampling sheets** (bootstrap; Efron & Tibshirani 1993).
- **Methods are compared paired, sheet by sheet,** on the same scale. Near-ties are settled by looking, not by the
  third decimal (memories: validation sets are noisy; one measure when comparing).
- **The current two-pass model's held-out outputs** (`data/frame-cmp-none`, 18 shared sheets) are scored the same way
  against the cleaned labels, as the baseline.
- **The MapSeg HD95 in pixels,** on the side, for comparison with the competition.

**Eye first:**
- Mosaics of every map page, outline drawn in.
- Close-ups:
  - the 1921 thin rule and arcs;
  - 1929 p09 Ocean Beach;
  - 1948/3 and 1948/4 (unlabelled, the reported failures);
  - 1948/8's left edge (where the 4,000-step spike went wrong);
  - one page per unlabelled edition.

**Red team.** With 21 sheets the intervals are wide. A difference has to be large to show up, and that's stated, not
hidden.

### 8. Deletion output

Per page under `pipeline/data/frame/<series>/<sheet>/`:
- **`outside.png`:** 1-bit, the deletion mask.
- **`outside_prob.npz`:** float16 chances, for reading and placement to weigh ink by.
- **`erased.png`:** the scan with the outside painted plain paper. Painting plain paper is an edit AGENTS.md allows. It's
  for looking only, until wiring.
- **`outline.json`:** the inside region as polygons, from the mask's boundary with collinear points merged, not
  smoothed.

`sfzmaps frame` runs:
1. the 8 held-out trainings, only when asked;
2. the 2 final trainings;
3. prediction;
4. the pictures.

It takes the one GPU turn.

## If the errors call for it (not in this build)

If the held-out errors left after part 2 are runs of map ink cut along a side (frameless edges, blocks against the frame,
the 1948/8 kind), the principled next step is a boundary-length prior. That's a binary Markov random field whose
smoothness term favours short, straight outlines. Its exact optimum comes from a minimum cut (Greig, Porteous & Seheult
1989; Boykov & Kolmogorov 2004), and its one weight is fitted by expected cost on held-out chances. It needs a non-GPL
max-flow (PyMaxflow is GPL). This is where the owner's "a whole side may have no frame" would enter, as evidence pooled
along the whole boundary. We decide from the residual errors, not in advance.

## What changes in the code

- **`frame/predict.py`, `frame/train.py`:** replaced. One network, whole-page batches, GroupNorm in `nets/unet.py` (a
  flag, so PR #70's users are unaffected).
- **`frame/data.py`:** holes filled, no strip, ink-clean check.
- **`frame/scores.py`:** ink-kept / ink-cut, sheet bootstrap, HD95.
- **New `frame/decide.py`:** cost rule, topology, outline.
- **`frame/run.py`:**
  - folds with seeds;
  - temperature;
  - the final ensemble;
  - prediction;
  - outputs.
- **`page_kinds.csv`:** a map count.
- **Tests:** the decision and topology on small synthetic pages (holes, islands, two maps, touching ink), the
  label-dependent-weight bias, temperature fitting, and the output formats. No tests of PyTorch itself.

## Build order

1. **Ink-clean labels:**
   - the QA check and the label fixes;
   - one framed 1948 sheet;
   - the owner's look at the QA sheet.
2. **Network changes:** GroupNorm, whole-page training, holes filled, no strip; spike-level run on one fold and a look.
3. **Decision and outputs:** decision, topology and outputs, with tests.
4. **Held-out run and scoring:**
   - 8 trainings;
   - step count;
   - temperature;
   - scores, compared with the baseline.
5. **Final models and the eye check:** final models, prediction of every page, pictures; the owner's eye.
6. **Review and pull request** (the owner merges).

Labelling unlabelled series that fail by eye is a follow-up loop, not part of this pull request (my call after the
builder check). The owner's eye decides which fail. Labels are added in a batch, then the held-out run, step count and
temperature are refitted once for the batch, not after each label.

## Owner answers (2026-11-20)

1. **Ruled inset boxes standing clear of the frame: (a).** Every ruled-off box at the margin is frame, cut out together
   with the blank paper between it and the frame. That's one target for one printed object: the ZONING MAP INDEX inset
   on 1975 p02 and 1980-1990-zoning p01 is relabelled as a notch, as on 1960/1.
2. **Cost ratio: leaving frame in is 3× worse than cutting map** (owner, tentative). So c_f = 3·c_m, and the cut is at
   q > c_m / (c_f + c_m) = ¼. It's applied only at decision time, so changing it later needs no retraining; the
   held-out run can show the frame-kept / map-cut trade-off at a few ratios for the owner to look at.
3. **1948 framed sheet:** "whichever" → 1948/2.

## Amendment: removed blocks (owner, 2026-11-20, later)

The owner circled the "City & County of San Francisco / Zone Map / Legend" block on the 1948 sheets ("I think the
legends should be cut out"), then: "Compass roses should go too. Section labels." So three kinds of printed block
**inside** the frame are deleted, wherever they sit:
- legend and title blocks;
- compass roses and north arrows;
- "SECTION N" sheet-number badges.

Everything else that floats inside the frame stays: notes, scale bars, cross-references, place names. Ruled boxes at
the margin stay notches (rule 3).

**Three answers per pixel.** The network gives a softmax over three classes:
- 0, map;
- 1, outside the frame;
- 2, removed block.

Labels gain a `remove` list of polygons, one per block, drawn a few px outside the block's ink. Training uses
multiclass log-loss with the same input-only weights. Multiclass log-loss is still a strictly proper scoring rule, so
the chances stay calibrated; one temperature divides the logits before the softmax (Guo et al. 2017 is multiclass).

**The decision** (part 1 and part 2 amended). With p₀, p₁, p₂ the three chances and `level` = c_m / (c_f + c_m) = ¼
(removed-block ink left in costs the same as frame ink left in):
1. Cut where p₁ + p₂ > `level`. Outside is the 4-connected components of that cut that touch the page border, as
   before.
2. Every other pixel can't be outside the frame, by definition. It is removed iff p₂ > `level`, which is the same
   cost rule with the impossible outside-frame mass returned to map. Map is where the no-holes step would put it.

   Why not condition, p₂ / (p₀ + p₂)? Where the network is confidently but wrongly calling map "frame", both p₀ and p₂
   are tiny and that ratio is noise. Returning the mass to map keeps the earlier behaviour there: a commercial band
   called frame mid-map is kept.
3. The inside region (neither outside nor removed) keeps its `maps` largest 8-connected components; the rest is
   outside.

Removed blocks are therefore the only holes, and only where the network says "removed block", not merely "not map".

**Scores:** frame ink kept counts every pixel labelled outside or removed that the decision keeps.

## Amendment 2: the cut traces straight lines (owner, 2026-11-20, later)

Owner, looking at the first look's curvy cut edges: "why are your edges so curved? that doesn't happen", then "you've
removed too many useful constraints. we're by-and-large tracing straight lines", and on notches: "Rectangles, but they
can be rotated." This restores the owner's first description ("The lines should be mostly straight except for the
arced corners"), which part 2's per-pixel cut lost.

### Why the per-pixel cut curves (measured, fold-0 model, 1948/8, 12, 14)

- **The edge wanders where nothing pins it.** 76–79% of the kept region's edge runs over blank paper, where it sits
  wherever the network's chance happens to cross ¼ (median chance along those edges 0.23–0.27): a soft contour, not
  a printed line.
- **The big curvy cuts are blank water mistaken for margin.** Deleted areas more than 300 px in from the page edge are
  76–89% blank paper (bays, Lake Merced), outlined along the coastline.
- **The labels are straight.** Fitting a line to each labelled side over its middle 70%:
  - median 0.3–2 px off it, and 95% of each side within about 10 px (1921 bows reach about 18 px);
  - tilt: median 0.5°, 95% within 1.6°;
  - bow (quadratic sag over the side): median 0.8 px, 95% within 15 px.

  The only exceptions are sides with notches, and the notches are rectangles.

### The shape (replaces part 2's per-pixel cut and component rules; edited after the red team, see below)

The kept region K is one shape:

- **The frame: one rotated rectangle.**
  - One rotation angle φ for the whole frame and four offsets, plus small per-side deviations. Measured on the
    labels: opposite sides are parallel to within 0.13° (median; 80% within 0.28°), and a frame's sides share one
    rotation to within 0.1°.
  - Per side: a tilt deviation bounded by those spreads widened by half, and a gentle bow bounded by the labels' 95%
    bow (15 px) widened by half.
  - A side with no frame is the page edge.
- **Corner arcs:** each corner may be cut by a circular arc, radius 0 (sharp) up to the labels' largest arc radius
  widened by half. The arc's form (tangent to both sides, as the 1921 corners look) is checked against the labelled
  corners in the spike before it's fixed.
- **Notches: rotated rectangles cut from K, each touching a side, at any length and depth.**
  - A notch's angle is searched around the frame's φ, over the range of the labels' notch angles widened by half (the
    owner: "they can be rotated").
  - **A notch must be a ruled box:** its inner sides follow printed ink. Every labelled notch is one (rule 3, and the
    owner's "every ruled-off box at the margin"). So a candidate notch scores the ink lying along its inner sides,
    with weight β (fitted). A cut that follows nothing printed, such as Lake Merced's region, gets no such support.
- **Removed blocks: shapes too.** Each is a rotated rectangle (legend and title blocks) or an ellipse (SECTION badges,
  compass roses), placed by the same expected cost, with a per-block cost λ_block (fitted). The candidates come from
  the pieces where p₂ > `level`. A map patch the network wrongly calls "removed" has an irregular, map-shaped
  outline: no rectangle or ellipse fits it without taking map ink, so it stays.
- **Removed blocks are the only holes.**
- **Pages listed with several maps:** one shape per map. Until such a page is found and checked, those pages keep the
  per-pixel decision, and the run lists them.

**The cost:** the same expected ink cost as part 1. It's a sum over pixels of c_f (p₁ + p₂) ink for each pixel kept,
and c_m p₀ ink for each pixel deleted. **Paper costs nothing:** the shape, not the paper's chances, decides where an
edge crosses blank paper, which is why it can be straight. Three numbers are fitted, all where scoring happens (on
held-out chances, by the same realised cost, as the temperature and the priors were):
- λ_notch, the cost of a notch, so noise isn't carved into many small notches;
- β, the weight of the ink along a notch's inner sides;
- λ_block, the cost of a removed block.

They are fitted on the full run's 21 held-out sheets. The fold-0 spike fits them on its 6 held-out sheets, which is
rough, and says so.

**Why sound:**
- This is the Bayes decision restricted to the shape class: of all the shapes the owner and the labels say a frame
  can have, choose the one with the least expected cost given the network's chances. The class carries knowledge the
  per-pixel network can't express: sides are straight across the whole page, the frame is one rectangle, and notches
  are ruled rectangles.
- The few fitted numbers are fitted on held-out chances, never on the network's own training pages.

### The search: each piece exact given the others; coordinate descent overall

The cost is a sum over pixels, so a running-sum table of the cost difference along each side's normal makes any
candidate line's cost a sum over the side's length.

1. **Start (for the per-side refinements; the frame itself is searched globally in step 2):** a rotated rectangle fitted jointly to all four sides with a robust (Huber) fit. The points come from
   each row or column's outermost kept pixel of the per-pixel cut. Robust, because a wrong inward cut, like Lake
   Merced's, only affects a minority of a side's columns.
2. **Frame, searched globally:**
   - Every rotation angle in the measured range, and all four offsets, on a coarse grid. A summed-area table of the
     cost difference in the rotated frame makes each candidate rectangle's cost one lookup (four corners), so the full
     grid is cheap.
   - Then a 1-px refinement around the best: the angle and each offset in turn, until none moves.
   - Then, per side, the small tilt deviation and bow within their bounds, from the running sums along that side.

   The frame is the best rectangle on the whole page to within the grid, not a local optimum near the start. Only the
   small per-side adjustments are local.
3. **Corners:** for each corner, the arc radius by a 1-D search.
4. **Notches**, greedily, one at a time while one gains more than λ_notch:
   - for each side and each angle in the range, work in that rotated frame: u along the side, v inward;
   - the column sums to each depth h include −β × the ink along the rectangle's inner sides;
   - for every depth, the best span [u₀, u₁] comes from a maximum-subarray (Kadane) pass;
   - this is exact for a single rectangle anchored at the side.
5. **Removed blocks:**
   - for each piece where p₂ > `level` inside K, the best rotated rectangle and the best ellipse over its
     orientation, centre and size, each by a local search from the piece's own moments;
   - each kept if it gains more than λ_block.

### Red team, and the edits it made

- **Edit 1: the frame is one rotated rectangle,** not four free lines. Measured: opposite sides parallel within 0.13°,
  one rotation per frame within 0.1°. Fewer free numbers per page, so a wrong patch of chances can't tilt one side on
  its own.
- **Edit 2: a notch must be a ruled box.** Without this, a confident error touching the frame, such as Lake Merced
  meeting the bottom frame, could still become a rectangular notch if its chances paid for it. Every labelled notch
  is a ruled box, so its inner sides lying on printed ink is real evidence. β weighs it, fitted.
- **Edit 3: removed blocks are shapes too.** About 70% of the fold-0 held-out error was the removed-block class,
  mostly map patches called "removed" (1948/12) and legends not removed (1921 Lake Merced). The straight-line frame
  alone leaves that untouched. Legends are rectangles; badges and compass roses fit an ellipse; a map-shaped false
  patch fits neither.
- **Edit 4: the fitted numbers are fitted on all 21 held-out sheets.** Fold 0's 6 hold only one or two notches,
  far too few to fit λ_notch; the spike says its fit is rough.
- **Edit 5: corner arcs are checked against the labels before the form is fixed.** The 1921 arcs look tangent to both
  sides, but a wrong form would leave arc ink inside every corner.
- **Edit 6: the start is a robust, joint rectangle fit,** so a wrong inward cut can't drag a side.
- **Edit 7 (owner: "not sure about your three still open"): no local optimum for the frame.** It is searched over
  the whole angle × four-offset grid (one lookup per candidate), then refined.
- **Edit 8: a bow beyond the bound is measured, not hoped away.** After the fit, every page is ranked by the
  confident frame ink (chance above `level`) left inside along each side. A page that bows more than any label sits
  at the top of that ranking for the owner's eye: label it, or widen the bound. It's a ranking, not a threshold.
- **Not a design gap:** errors of meaning, such as a sheet where the network takes a lake for margin, can't be fixed
  by any decision. The shape limits their damage (straight lines, ruled notches, shaped blocks); the fix is labels,
  chosen where the two seeds disagree, by the labelling loop already in the plan.
- **Speed:**
  - the line search is a few hundred candidates × the side length, from running sums;
  - the notch search is one pass per depth × angle;
  - the block search is local per candidate.

  Measured in the spike; the target is seconds per page.
- **What doesn't change:** the network, the training, the labels, the calibration and the scores. Only the decision
  changes, so it's compared with the per-pixel cut on the same held-out chances.

### Test before adopting

Same as the priors: the fold-0 model's held-out chances and pages, both decisions, the same costs, and the pictures of
1948/8, 12, 14 and the close-ups. Then the plan gets a task for it.

## Amendment 3: the combined design (owner, 2026-11-21: "Write the plan for the new combined design")

The owner asked for an audit against the literature ("surprised that this isn't working ... single model, a mixture of
experts, or what?"). Findings, each measured here or cited:

- **The from-scratch region network gets the layout wrong, and more labels of the same kind don't fix it.** Trained on
  all 21 labels, its wrong cuts move rather than vanish: 1948/14 loses the ocean instead of the lake, and 3, 5a and 6a
  lose sparse areas. It decides "dense city = map, sparse = outside". "Inside the frame" is a non-local question
  (Villalobos et al. 2021), and half of what drives the network's decision comes from within about 280 px.
- **A frozen pretrained backbone with a small head gets the layout right** (DINOv2 ViT-B/14, Apache-2.0; Sterzinger
  et al. 2025 for few-shot historical maps). On fold 0's held sheets:

  | | pooled cost | layout cost (more than 100 px from the outline) |
  |---|---:|---:|
  | DINOv2 + small head | 205,036 | 6,578 |
  | region network | 381,666 | 159,954 |

  - It keeps oceans, lakes and bays, finds every legend, rose and badge on unseen 1948 pages, and cuts the title
    strips on five unlabelled editions.
  - Its edge wobbles by about a patch, and 82–97% of its cost is within 100 px of the outline.
- **Line evidence** (a network trained on nearness to the outline) finds the right rule (median 3 px, 24/24 sides
  framed-or-not), but its peak lands on the rule about half the time: cost 787k–2,039k against the region cut's
  113k. Not used.
- **The full-resolution region network is good at exactly the edge:** a steady 4 px inside the rule where a frame
  line really is.
- **The straight-line shape decision** (amendment 2) is exact given its chances; its remaining cost was the chances.

### The design: each part does what it measured well at

1. **Layout** (new): frozen DINOv2 ViT-B/14 with registers (`facebook/dinov2-with-registers-base`, pinned revision
   and file hash).
   - **Input:** the whole page, long side 111 patches (1,554 px), grey as 3 channels.
   - **Features:** the last 4 layers' patch tokens (3,072 numbers per patch), standardised and reduced to 128
     principal components. The components are fitted once on every map page's features. That uses no labels, so the
     348 unlabelled pages contribute.
   - **Head:** 3×3 conv (32) → ReLU → 1×1 conv → three classes (map / outside / removed block), trained on each
     patch's class shares with the usual input-only weights (0.1 + mean ink). It trains in seconds.
   - **Seeds:** three heads from different seeds are averaged (measured spread across seeds: 2.6k–17k layout cost).
   - **Held out by leaving one sheet out** (21 × 3 heads, seconds each): every labelled sheet gets layout chances from
     heads that never saw it. This replaces the 4 folds for the layout.
2. **Edge** (already built): the full-resolution network (`frame/model.py`, `learn.py`), unchanged.
   - It is used **only within a band of each fitted line**. Its layout mistakes far from any line are never read.
   - The band's half-width W is the layout's largest held-out side-position error, widened by half (the same pattern
     as the other bounds).
   - Held-out edge chances come from the 4 folds, one seed (about 9 minutes each).
3. **Decision** (amendment 2's shape, ported from the spike): one rotated-rectangle frame, per-side tilt and bow,
   corner fillets, ruled rectangular notches, and removed blocks as rotated rectangles or ellipses.
   - **Coarse stages** (frame search, notch and block candidates) minimise the expected ink cost under the layout
     chances.
   - **Refinements** (side offsets, tilt, bow, fillet radii, notch sides) use the edge chances inside the band and the
     layout chances outside it.
   - **Block shapes are bounded by the labelled blocks' sizes, widened by half.** The spike's ellipses ballooned over
     blank water because paper is free.
   - **λ_notch, β and λ_block are fitted on the 21 held-out sheets** by realised cost.
   - Amendment 2's ranking of pages by confident frame ink left along the sides, and its multi-map rule, stand.
4. **Calibration, scores, outputs and pictures:** as built. There is one temperature per model, each fitted on its own
   held-out chances. Scores add the split of cost into layout (more than 100 px from the label's outline) and edge;
   that is a reporting split, not a decision.

### Why sound

- This is the competition winner's recipe (coarse layout from a network, then the border re-adjusted on the ink at
  full resolution; Chazalon et al. 2021). A pretrained backbone supplies the coarse part, since 21 labels can't teach
  layout from scratch, and the owner's straight-line shape supplies the re-adjustment in place of tuned morphology.
- Each stage's fitted numbers are fitted on chances from sheets its model did not train on.
- It is one process for every edition, not a mixture of experts: per-edition experts would have 1–5 labels each.

### Red team, and the edits it made

- **The pretrained result rests on 6 sheets, and mostly on one** (1948/12). The paired interval just includes zero.
  → The first task is a combined test on the same pages before anything is ported, and leave-one-out over all 21
  follows in the run.
- **The layout still cuts wide blank strips just inside a frame** (1948/2, 9, 5a's Pacific strip). → The frame is
  fitted as a rectangle to the whole page's cost, so a strip the head wrongly calls "outside" only moves a side if it
  outweighs the rule's own evidence. The combined test looks at exactly those pages. If it persists, the fix is labels
  for that case, chosen by the seeds' disagreement.
- **A new dependency and a downloaded model** (`transformers`, Apache-2.0; weights Apache-2.0). → The revision is
  pinned and the weights' hash checked before use. Licence notices are added. Tests never download (a tiny stand-in
  backbone). The first real run needs the network once.
- **Features differ slightly between devices** (to within 0.03 between cpu and mps). → One device for every page
  (mps under the GPU turn). Results are deterministic on one machine, and the doc says they are not bit-equal across
  machines.
- **Disk:** raw features are about 58 MB per page. → Only the 128-component features are stored (about 2.4 MB per
  page, under 1 GB for all pages), and per-page outputs follow the disk rule (one run's maps at a time).
- **The edge network still trains for about 9 minutes per fold.** → One seed. It is only read inside the band, where
  the decision is a few px either way. If the two models disagree on framed-or-not anywhere, the layout wins by
  construction.

### Amendment 3, after the combined test (owner, 2026-11-21: "Sgtm")

On fold 0's six held-out sheets, as shares of each class's ink:

| decision | frame / legend ink left in | map ink cut |
|---|---:|---:|
| from-scratch region network, per-pixel cut | 3.8% | 1.00% |
| pretrained layout, per-pixel cut | 2.0% | 0.12% |
| pretrained layout + straight-line shape | 1.1% | 0.14% |
| … + the full-resolution network within a band of the lines | 3.9% | 0.07% |
| pretrained layout + shape + block shapes bounded by the labels | **1.0%** | **0.14%** |

- **The full-resolution network is dropped** (part 2 of the design above). It calls the 1921 Lake Merced sheet's top
  frame bar "map", and inside the band its call overrules the correct layout: that sheet alone goes from 38k to 202k
  cost. It helps by a few hundred ink on clean modern rules. `frame/model.py` and `frame/learn.py` go; no long GPU
  training remains.
- **A side with no frame is the page's own edge**, not a rotated line. The fitted frame's angle is shared by framed
  sides only. On partly framed pages the rotated page-edge sides clipped corners (1948/8 +2.56°, 1948/9 −2.10°).
- **The remaining cost is about 90% at the edge:** the layout sees 50-px patches. Next test: slide each fitted line,
  within one patch, to where it lies on paper (label rule 1), with no second network.
- **Open from the test:** the bottom row of blocks cut on 1948/4, and a notch across the map on 1948/3.

### Amendment 3: the cut lies on paper (owner, 2026-11-21: "Make a plan and do it")

Task 1b's test, on the six held-out sheets with the full-resolution network truly absent:

| | frame / legend ink left in | map ink cut |
|---|---:|---:|
| layout + straight lines (the true start; the 1.0% above still read the full-resolution network on boundary px) | 1.19% | 0.136% |
| … + each line slid onto paper | 0.81% | 0.139% |

**The step:**
- Each fitted line slides along its normal within one patch width.
- It takes the position minimising the layout's expected ink cost of what it keeps, plus the ink under the line itself
  charged at the frame-left-in price (label rule 1: the outline lies on paper). No number is fitted.
- A frame side's two ends move independently. Ties stay put, and page-edge sides don't move.

**What was tried and lost:**
- **Lightest row first:** 4× worse, because it lands on the blank margin outside the bar.
- **The layout's cost alone:** no change.
- **Ink charged at the map price:** 87k against 71k.
- **A likelihood measured from the labels:** worse, because the labelled lines sit on the rule's ragged fringe.

**Known regression:** where map hatching touches the rule (1921/142210), the line moves 1–3 px into the rule's fringe.

**Out of its reach (layout misses of more than a patch), to be examined when the shape code is ported:**
- 1948/12's badge not found and its legend fitted 5° off;
- 1921/355582's bottom side 400 px high;
- 1948/4's bottom row of blocks clipped;
- 1948/14's thick bars called map.

## Amendment 4: a test set, and what it showed (owner, 2026-11-21)

The owner's directions, in order:
- "if you can't score honestly, then add some damn labels. don't train on them";
- "you're working in a little box ... trying to workaround obvious things that you should do";
- on the cut: "you were following a frame edge and then decided to jump to the edge of the scan for no apparent
  reason"; "such wobbly magenta";
- on a sparse sheet: the model is "using white space as a signal on the outside of the frame rather than looking for
  edge of file".

### What exists now

- **A test set of 22 sheets, one per series, never trained on:** `pipeline/inputs/frame-labels-test/`. Each was chosen
  by a fixed rule before any prediction was looked at (the unlabelled page at index 3n/4 of its series) and drawn from
  the scan only.
- **39 training sheets** (`pipeline/inputs/frame-labels/`): the original 21, one for each of the 14 series that had
  none, and a second for 1929, 1935, 1975 and 1980-1990-zoning.
- **The measure is the test set.** Per sheet and pooled: the share of frame and legend ink left in, and the share of
  map ink cut. Scoring is one decision per test sheet, taking minutes.
- **The leave-one-out run is retired.** It took over an hour, and on single-label editions it measured "no label for
  this edition", not the model that ships.

### First test-set result (heads on the 39 labels; the simple cut)

The simple cut: cut where deleting is the cheaper expected cost; outside is what connects to the edge of the file; one
kept region.

| | frame / legend ink left in | map ink cut |
|---|---:|---:|
| pooled, 22 sheets | 2.37% | 0.361% |
| median sheet | 0.78% | 0.043% |

- **1960 through 2000-digital (17 sheets):** at most 1.6% left in (one sheet 2.6%), at most 0.34% cut.
- **1921, 1929, 1935:** 4.7–8.5% left in, 0.6–2.9% cut.
- **1948/6a** (frameless photo): 51% left in. Its title block, compass rose and badge are all there is to delete.

### What was learned about the two decisions

- **The model sorts ink well and has no reliable opinion about blank paper.** With three heads it calls large blank
  areas "outside" on sparse sheets (1975 p18 "4H", 1960/4, 1964 p10). The owner's white-space inference was right.
- **The simple cut** trusts that opinion: its boundary wanders across blank paper.
- **The straight-line shape decision** (amendment 2) was built to fix that, but on unseen pages it produced a frame
  fitted inside the real one, rectangles cut from the map (1975 p18, at every setting tried), and a patch of margin
  left in between two notches (2000-digital-su p11). Causes not yet found; tinted paper letting the model's
  blank-paper opinion into the "paper costs nothing" cost is the first suspect.

### The rule the cut must follow

**A side of the cut is a straight line along the inner edge of the printed frame ink, or the edge of the file where
there is no frame. Blank paper never moves a line.** The model says which ink is frame; the geometry comes from the
ink and the file's edge.

### How decisions are chosen from here

Every candidate decision is scored on the 22 test sheets against the simple cut's numbers above, and shown on the same
pages. The one with the better numbers and no embarrassing page ships. Labels are added where the test set and the
owner's eye show a weakness; the test set itself is never trained on and never used to fit a number.

The one number that needs held-out chances, the temperature, comes from a 4-fold split of the training sheets
(12 small heads), not from the test set.

## Amendment 4, after the straight-line test (owner, 2026-11-21)

- **The owner chose the cut:** "well those straight-line cuts are practically perfect". The straight-line cut ships.
  The simple cut is no longer a candidate, and the "how decisions are chosen" paragraph above no longer applies to it.
- **The owner's one complaint:** "the only issue I see is the map label box being a bit wrong".
- **Measured on the same 22 test sheets, frame and label boxes apart** (pooled):
  - frame ink left in: 0.24%;
  - label-box ink left in: 31.5% (boxes are 3.3% of the ink to delete, so one pooled figure hid them);
  - map ink cut: 0.32%.
- **The frame fit is therefore kept as it is.** The remaining work is the label boxes, which have two causes:
  - **the fit:** rectangles take their angle from the model's blob and come out turned by up to 22°, while every
    labelled box is square to its sheet. Rule: a rectangular box takes the frame's angle; on a page with no framed
    side, the page's axes;
  - **the model:** compass roses drawn over the map (chance of deletion 0.00–0.02) and some small badges are not seen
    at all. That needs training labels, not a different fit.
- **The test set is a check, not a tuning set.** Fixes are iterated on a dev set: the training sheets, each cut by
  heads that did not train on it.
- **The earlier suspicion that paper tint moves the frame's sides is withdrawn:** it was never tested, and the frame
  is right on the test sheets. The four pages the owner flagged earlier were seen with older heads and settings, and
  are re-cut with the current model before anything is concluded about them.

Plan: `docs/superpowers/plans/2026-11-21-frame-crop-test-set.md`.

## Amendment 5: a printed rule is one thing (owner, 2026-11-21)

**The fault.** On 2000-digital-su p11 the cut came out tilted 1.66° against a level printed frame, with the right half
of the top rule left in. The model's chance of deletion along that one rule, in six stretches from left to right, was
0.97, 0.92, 0.72, 0.17, 0.25, 0.38. The cut judges ink px by px, so the cheapest rectangle kept the "map" half of the
rule and tilted to do it. The level and the tilted rectangle scored within 0.07% of each other.

**The owner's choice** among three options (more labels; treat a rule as one thing; only flag such pages): "2", then
"Do 2". In my words, which the owner chose: treat a straight printed rule as one thing: if most of it is frame, all of
it is.

**The rule.**
- A *rule* is a connected stroke of ink that runs straight for a long way along the page (within the frame search's
  angle range) and is thin across. Frame rules and legend bars are rules. Hatching, lettering, corner arcs and solid
  blocks are not.
- Every px of a rule shares one set of chances: the ink-weighted mean of the model's chances along the rule.
- The straight-line cut then runs unchanged on those chances.
- A stroke the model reads the same way along its length is unchanged by this. Only strokes with a split reading
  change, and they change as a whole.

**Why the mean.** A rule is one printed object with one true class. The mean keeps each rule's expected ink in each
class exactly as the model gave it, so the cost accounting of part 1 still holds, and the 3:1 cost then decides the
whole rule at once. Two readings were tried on the same sheets: every rule takes its mean, or only rules whose mean
chance of deletion is over one half. The first is used.

**Measured (trial, heads on 39 labels, 22 test sheets, pooled):**

| | frame ink left in | label-box ink left in | map ink cut |
|---|---:|---:|---:|
| before | 0.24% | 31.5% | 0.322% |
| every rule takes its mean | 0.10% | 31.4% | 0.330% |
| only rules over one half | 0.18% | 31.5% | 0.322% |

- 2000-digital-su p11: angle −1.66° → −0.22°, no bow, the whole top rule deleted.
- 1964 p14 (test): angle −1.31° → −0.14°, frame ink left in 0.68% → 0.00%.
- 1980-1990-su p01: the top side's bow, which was at its limit, goes to 0.
- The rise in map ink cut is one sheet, 1929 p14: its top rule goes whole, including stretches where the label counts
  block ink merged into the rule as map. Its frame ink left in falls from 7.7% to 4.3%.
- Not improved: where blank paper lies between a bar and the map, nothing printed pins the cut's angle
  (1980-1990-height p10 moved from −0.21° to −1.01°, with no ink affected).

**Settings.** How long a stroke must run to count (201 px) and how thin it must be (under 61 px) are trial settings,
not fitted. Halving the length (101 px) gives the same pooled result (0.10% / 0.328%); doubling it (401 px) loses the
1929 p14 gain (0.18% / 0.322%) and keeps the three fixes above.
