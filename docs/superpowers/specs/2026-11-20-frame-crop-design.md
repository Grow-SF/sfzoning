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

1. **Start:** a rotated rectangle fitted jointly to all four sides with a robust (Huber) fit. The points come from
   each row or column's outermost kept pixel of the per-pixel cut. Robust, because a wrong inward cut, like Lake
   Merced's, only affects a minority of a side's columns.
2. **Frame:**
   - the shared angle φ by a 1-D search around the start;
   - for each side in turn, holding the others fixed, every offset and a fine grid of tilt deviation and bow within
     the bounds, each scored from the running sums;
   - repeat until none moves. The cost can only fall, so this ends.
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
- **Still true:**
  - coordinate descent finds a local optimum, and the held-out scores show it if the start misleads it;
  - a bow beyond the bound leaves a sliver of rule inside, and the close-ups check the thin rule;
  - errors of meaning on sheets unlike any label still need labels.
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
