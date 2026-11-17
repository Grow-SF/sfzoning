# Text detectors and spotters, traced step by step (2026-11-17)

Purpose: trace what the strongest general text detectors and spotters do, step by step, and set them against our
map-text chain to find what we are missing. No code changed. This does not repeat
`2026-11-16-text-graphics-separation-literature.md` (synthetic data, missing labels, loss weights).

Marks:
- **[paper]**: read in the paper (the ar5iv HTML render of the arXiv version).
- **[code]**: read in the official repo's source, README or LICENSE on GitHub, 2026-11-17.
- **[not checked]**: from memory or inferred, not confirmed against a source.
- "Inference": my reading for our case, not the source's.

Our chain (from the brief):
1. A U-Net gives each ink px a text chance, a kind and a direction, learned from an LLM's word rectangles.
2. Threshold the chance, then group connected components into rotated boxes.
3. A CRNN/CTC reader reads each box against a closed code list.

Our problems:
- (P1) line ink crossing labels is called text;
- (P2) boxes merge with lines or split (47-79% of code words found);
- (P3) the reader is 96% right on good boxes and 46% on ours.

## The one thing every approach below does differently from us

- None forms a word from **ink connectivity**. They use a **shrunk region** (DBNet, PSENet, PAN, Mask TextSpotter v3),
  **character blobs plus links** (CRAFT), **per-px box regression** (EAST, ABCNet) or **one query per word** (TESTR,
  DeepSolo).
- Their targets cover the word's **area** (paper px included), not its ink. Line ink inside the area is "word region",
  which is harmless; line ink outside it is background, even where it touches a letter.
- Inference: our ink-level target makes the net settle P1 px by px. Grouping by connected components then turns each
  P1 mistake into a P2 merge.

## 1. CRAFT (Baek et al., CVPR 2019)

Sources: https://arxiv.org/abs/1904.01941 [paper]; https://github.com/clovaai/CRAFT-pytorch [code].

**Steps**
1. **Input and backbone:** an RGB image (the demo resizes it to a 1280 canvas at 1.5× magnification) [code].
   - The backbone is VGG-16 with batch norm and a U-Net-like decoder [paper].
   - The output is at half size (`ratio_net = 2`) [code].
2. **Heads** [paper]:
   - **region score**: the chance that a px is a character centre;
   - **affinity score**: the chance that a px is the centre of the gap between two adjacent characters of one word.
3. **Targets** [paper]:
   - Region: a 2D Gaussian warped by a perspective transform into each character box.
   - Affinity: a box joins the centres of the upper and lower triangles (cut along the diagonals) of two neighbouring
     character boxes, and gets the same Gaussian.
   - Affinity is zero across a space (appendix).
4. **Loss:** squared L2 on both maps, weighted per px by a confidence map S_c; OHEM at 1:3. 50k iterations on
   SynthText, then 25k on real data with SynthText mixed in at 1:5 [paper].
5. **Post-processing** (`getDetBoxes_core`) [code]:
   - take px with region > 0.4 (`low_text`) OR affinity > 0.4 (`link_threshold`), then 4-connected components;
   - drop components under 10 px, or with a peak region score under 0.7 (`text_threshold`);
   - remove link-only px, dilate a little, and fit `cv2.minAreaRect`: one rotated box per word, no NMS.
   - An optional polygon follows the character maxima [paper].
6. **Recognition:** none.

**Supervision.** It needs character boxes, but real sets give only word boxes and transcriptions, so it makes its own
[paper]:
- It crops the word, predicts region scores with the interim model, splits them into characters by watershed, and maps
  them back.
- It scores the split as s_conf = (l − min(l, |l − l_c|)) / l, where l is the transcription length and l_c the
  characters found.
- If s_conf < 0.5, it splits the box evenly into l characters and sets s_conf = 0.5.
- S_c = s_conf inside the word and 1 elsewhere. Do-not-care words get 0.
- In short, it checks its own character guesses against the **spelling length**.
- Limits [paper]: line-level sets needed an extra LinkRefiner, and cursive scripts split badly.

**Text vs touching graphics; splitting words.**
- The targets are blobs at character centres, not ink. A line through a letter adds nothing. Line px between words or
  past a word's end have target 0 although they are ink (inference).
- A word is region blobs **bridged by affinity blobs**, and words split where affinity is low [paper, code].
- Clutter is dropped by the 0.7 peak and 10 px tests [code].
- Inference: a line touching two words merges them only if affinity fires along it. Affinity is trained only between
  character pairs; the paper does not test this.

**Strengths and failures.**
- Strengths: long and curved words (a chain of characters, so the receptive field does not cap length), rotation, any
  shape [paper].
- Failures [paper]: words split by meaning rather than spacing; cursive scripts; blurred regions when the region score
  is poor.
- On maps, MapText 2024 (Rumsey): EasyOCR's CRAFT reached word-detection F 58.3, against 92.0 for the winner (li24
  Table 2). The report does not say whether it was fine-tuned.

**Practicality.**
- The model code is **MIT**, but the official repo has **no training code** ("cannot release ... for IP reason") [code].
- EasyOCR's CRAFT trainer is **Apache-2.0**, with watershed pseudo-labels, Gaussian targets and MSE
  (https://github.com/JaidedAI/EasyOCR/tree/master/trainer/craft) [code]. Its accuracy claim was not verified.
- Inference:
  - The two maps fit as extra heads on our U-Net, with targets made on the CPU from boxes; VGG-16 is not needed.
  - The targets ignore the image, so 1-bit scans are no obstacle.

## 2. DBNet / DBNet++ (Liao et al., AAAI 2020; TPAMI 2022)

Sources: https://arxiv.org/abs/1911.08947 and https://arxiv.org/abs/2202.10304 [paper]; https://github.com/MhLiao/DB
[code].

**Steps**
1. **Input and backbone:** 640×640 crops [paper].
   - ResNet-18 or -50 with deformable conv in stages 3-5, plus an FPN (levels upsampled and concatenated).
   - The head is a 3×3 conv and two stride-2 deconvs.
2. **Heads** [paper]:
   - **probability map** P: the chance that a px is inside a word's shrunk kernel;
   - **threshold map** T: a learned per-px threshold.
   - The binary map B = 1/(1+e^(−50(P−T))) is differentiable, so the threshold can be learned.
3. **Targets:**
   - Shrink each polygon (Vatti clipping) by D = A(1−r²)/L, with A the area, L the perimeter and r = 0.4 [paper].
   - In the band grown by D, T's target is the distance to the edge scaled into [0.3, 0.7] (`make_border_map.py`)
     [code].
   - Ignore-tagged words, and words under 8 px on their short side, are cut from the loss mask [code].
   - Worked example (my arithmetic): a 60×20 px box gives D = 6.3 px and a kernel of about 47×7.
4. **Loss:** Ls + 1.0·Lb + 10·Lt. Ls and Lb are BCE with hard-negative mining at 1:3; Lt is L1 inside the grown
   polygon [paper].
5. **Post-processing** [paper, code]:
   - use P only (T is dropped at test time);
   - binarize (0.2 in the paper, 0.3 in the code) and take connected components;
   - drop a box if the mean P inside it is under 0.7 (`box_thresh`);
   - unclip by D′ = A′·1.5/L′. Worked example (my arithmetic): the 47×7 kernel grows to about 57×17.
6. **DBNet++** adds Adaptive Scale Fusion, a per-px attention over the 4 levels [paper].
   - Gains: +2.3 F on MSRA-TD500 and +1.9 on CTW1500, at 32 → 29 FPS.
7. **Recognition:** none.

**Supervision.** Word polygons only. Doubtful words are handled only by the ignore mask (loss weight 0) [code].

**Text vs touching graphics; splitting words.**
- Neighbouring kernels are at least 2D plus the word gap apart. Components run on kernels, never on ink [paper, code].
- The threshold map lights up at borders even without supervision. The paper: it helps "separate text instances which
  are closely jointed" [paper].
- Inference:
  - A border crossing a code lies inside its kernel and does no harm.
  - A line running on beyond the code is outside the kernel, so it cannot drag the box along.

**Strengths and failures.**
- Strengths: fast (ResNet-18 at 62 FPS, MSRA-TD500 at 736 px).
  - DB adds +3.2 to +4.9 F; deformable conv adds +1.5 to +5.0 [paper].
- Failures:
  - "text inside text" [paper];
  - small text: for long words D ≈ 0.42·h, so a word 12 px tall keeps a kernel about 2 px tall (inference from the
    formula).
- On maps, MapText 2024 (Rumsey): a DBNet-based entry (MapDet) got F 47.2. That is one team's effort, not a ranking of
  the method.

**Practicality.**
- The official repo has **no LICENSE file** (all rights reserved) [code], so don't copy from it.
- Apache-2.0 ports:
  - **MMOCR**: DBNet, DBNet++, PSENet, PANet and an ABCNet project [code]. Not checked on mps.
  - **docTR**: DBNet, LinkNet and FAST, plus the CRNN, SAR, MASTER, ViTSTR and PARSeq readers [code].
    - Detection training takes `--device mps` (with `PYTORCH_ENABLE_MPS_FALLBACK=1`) and polygons in several classes.
    - `--mask-empty-classes` handles partly labelled data.
- Deformable conv on mps: not checked. It can be dropped (−1.5 F on ResNet-18).
- Inference: the heads are two small conv branches on our decoder. Shrink and unclip are closed-form for rotated
  rectangles. Fits 6-7 min.

## 3. EAST (Zhou et al., CVPR 2017)

Source: https://arxiv.org/abs/1704.03155 [paper].

**Steps**
1. **Input and backbone:** 512×512 crops, batch 24; PVANet (or VGG-16) with features at 1/32 to 1/4 merged U-Net style;
   output at 1/4 size (implied, not stated) [paper].
2. **Heads** [paper]:
   - a **score map**: the chance that a px is inside a shrunk word;
   - **geometry**: RBOX gives the distances from the px to the 4 sides of its word's rotated box, plus the angle (5
     channels); QUAD gives offsets to the 4 corners (8).
3. **Targets:** each edge moves in by 0.3·r_i, r_i being the shorter edge at that corner; geometry is set at every
   positive px [paper]. Worked example (my arithmetic): a 60×20 box gives a 48×8 core.
4. **Loss:** class-balanced cross-entropy on the score (β = 1 − positive share), −log IoU on the box plus
   10·(1 − cos Δθ); a scale-normalised smooth L1 for QUAD [paper].
5. **Post-processing:** threshold the score; each positive px proposes a box. **Locality-aware NMS** merges neighbours
   row by row (score-weighted average of geometries), then runs standard NMS; O(n) at best [paper].
6. **Recognition:** none.

**Supervision.** Word quadrilaterals only. No device for weak labels is given [paper].

**Text vs touching graphics; splitting words.**
- The extent is **regressed, never traced**: a core px of "R-4" predicts R-4's sides whatever ink touches it
  (inference).
- Line px extend nothing unless they score high themselves; even then they make separate proposals for NMS (inference).
- Neighbours are split by the shrunk cores and NMS [paper].

**Strengths and failures.**
- Strengths: one stage, rotated boxes [paper].
- Failures: the longest word is capped by the receptive field and vertical text is weak (rare in training) [paper]; no
  curved text (inference).
- MapTD (Weinman 2019), an EAST-based map detector, is covered in the 2026-11-16 note.

**Practicality.**
- No official code found in one search [not checked thoroughly].
- argman/EAST is TensorFlow and **GPL-3.0** [code], so not usable. SakuraRiven/EAST is PyTorch and **MIT** [code].
- Inference:
  - We already predict direction; four distance channels, an IoU loss and a merge loop are small.
  - Codes are short and straight, so the receptive-field cap matters only for long street names.

## 4. PSENet and PAN (Wang et al., CVPR 2019; ICCV 2019)

Sources:
- PSENet: https://arxiv.org/abs/1903.12473 [paper]; https://github.com/whai362/PSENet [code].
- PAN: https://arxiv.org/abs/1908.05900 [paper]; https://github.com/whai362/pan_pp.pytorch [code].

**PSENet steps**
1. **Backbone:** ResNet with an FPN. The concatenated levels (1024 channels) are cut to 256 and fed to n sigmoid
   branches [paper].
2. **Heads:** maps S1..Sn, the same words at n scales. S1 is the smallest kernel and Sn the full word [paper].
   - The official configs use 7 kernels with a minimum scale of 0.4 [code].
3. **Targets:** shrink by d_i = A(1−r_i²)/L, with r_i rising linearly from m to 1 [paper].
4. **Loss:** 0.7·L_c + 0.3·L_s, both Dice [paper].
   - L_c (the full map) uses OHEM at 3:1.
   - L_s counts only px where Sn ≥ 0.5, so non-text px are ignored for the kernels.
5. **Progressive scale expansion:** seeds are the components of S1; a 4-connected BFS grows them through S2..Sn; a
   contested px goes to the first seed to arrive [paper]. Test settings: `min_score` 0.85, `min_area` 16 [code].

**PAN steps**
1. **Backbone:** ResNet-18 with cascaded FPEMs (light U-shaped blocks at about 1/5 an FPN's FLOPs) and an FFM giving
   stride-4, 512-channel features [paper].
2. **Heads:** text region, kernel (shrunk at r = 0.5 on IC15, 0.7 elsewhere) and a **4-d similarity vector** per px
   [paper].
3. **Loss:** L_tex + 0.5·L_ker + 0.25·(L_agg + L_dis); OHEM on L_tex only [paper]. L_agg pulls each text px's vector
   to within 0.5 of its kernel's mean; L_dis pushes different kernels' means at least 3 apart.
4. **Pixel aggregation:** kernel components grow into neighbouring text px whose vector is within d = 6 of theirs,
   repeated until stable; 84 FPS at 320 px [paper].
5. **Recognition:** none. PAN++ adds an attention reader with a "Masked RoI" (https://arxiv.org/abs/2105.00405,
   abstract only).

**Supervision.** Word polygons only. Ignore handling as in DB [not checked in this code].

**Text vs touching graphics; splitting words.**
- Seeds are the smallest kernels, which neighbours never share, and growth stops at non-text px [paper].
- PAN adds a learned test: a px joins a word only if its vector matches the word's kernel [paper].
  - Inference: line ink touching a word could be refused even when called "text region". Our component step cannot
    ask that question.
- PSENet's own trade-off [paper]: with m too large, close words merge; with m too small, a word splits.
  - This is our P2 in miniature. They escape it by growing from small seeds, not by tuning one threshold.

**Strengths and failures.**
- Strengths: any shape, close words split, PAN real-time [paper].
- Failures: PSENet's expansion is slow at full resolution; PAN fails on wide character spacing, symbols and false
  positives [paper]. Inference: letter-spaced street names would break into islands.

**Practicality.**
- Both official repos are **Apache-2.0** [code]. The PSE/PA post-processing is Cython/C++ [not checked for a Python
  fallback].
- Inference: a kernel head, a 4-d embedding head and a CPU BFS over a few hundred words are all cheap. Fits the budget.

## 5. End-to-end spotters

### 5a. Mask TextSpotter v3 (Liao et al., ECCV 2020)

Sources: https://arxiv.org/abs/2007.09482 [paper]; v2's character branch https://arxiv.org/abs/1908.08207 [paper];
https://github.com/MhLiao/MaskTextSpotterV3 [code].

**Steps**
1. **Backbone:** ResNet-50 [paper].
2. **Segmentation Proposal Network** [paper]:
   - a U-Net head on 1/4-scale features predicts shrunk kernels (Vatti, r = 0.4), trained with Dice;
   - binarize at 0.5, take components, and grow each by A·3.0/L into a polygon proposal.
3. **Hard RoI masking:** the 32×32 RoI features are multiplied by the polygon mask, zeroing px outside it, to
   "suppress background noise or neighboring text instances". Total-Text: 71.2 hard, 69.1 soft, 67.3 without [paper].
4. **Heads on the masked features:** box refinement, a word mask, and two readers [paper; v2]:
   - **character segmentation**: 37 maps (36 letters and digits plus background) at 32×128.
     - Targets: character boxes shrunk to 1/4 of their sides.
     - Decoding: binarize at 0.75 and vote a class per region.
   - a **spatial attention module** trained on words alone.
   - The more confident reader wins. Lexicon matching uses an edit distance weighted by character probability.
5. **Loss:** L_s + L_rcnn + L_mask, all with weight 1 [paper].

**Supervision.** Character boxes come from SynthText. Word-only data sets the character targets to −1, which the loss
ignores [paper; v2].

**Text vs graphics.**
- Proposals are kernels, as in DB.
- Hard masking keeps the readers from seeing anything outside the word [paper].
- Inference: a direct answer to P3. Our reader sees every bit of line ink inside a loose box.

**Strengths and failures** [paper]:
- Strengths: rotation (RoIC13 at 45°: detection F 84.2 vs 62.2 for v2), long words (MSRA-TD500 83.5), shapes
  (Total-Text 71.2). Failure: near-90° text, whose reading direction is unclear.

**Practicality.** **CC BY-NC 4.0** [code], so not usable. The ideas are easy to rebuild.

### 5b. ABCNet v2 (Liu et al., TPAMI 2021)

Sources: https://arxiv.org/abs/2105.03620 [paper]; https://github.com/aim-uofa/AdelaiDet [code];
https://github.com/open-mmlab/mmocr/tree/main/projects/ABCNet [code].

**Steps** [paper]
1. **Backbone:** ResNet-50; FPN or BiFPN (+1.6 to 1.9); CoordConv, i.e. x and y maps concatenated (+2.8 to 2.9).
2. **Detector:** single-shot and anchor-free (FCOS-style). Each location predicts a score and **16 numbers**: the 8
   control points of two cubic Bézier curves (top and bottom edges).
3. **Targets:** a least-squares fit of each polygon on the Bernstein basis. Four-point boxes get two interpolated points
   on each long side.
4. **BezierAlign:** samples an 8×32 grid between the two curves, along the word's own axis, from the 1/4 to 1/16 maps.
   - 61.9 vs 44.7 F against quadrilateral sampling.
5. **Reader:** 6 convs, height pooling, a BiLSTM and an attention GRU. v1 used CTC; v2 calls attention "more
   powerful".
6. **Adaptive end-to-end training (AET):** detections above a score threshold (after NMS) are matched to the
   ground-truth word with the nearest control points and **added to the reader's training set** with that word's text.
   +1.2 on Total-Text, +1.7 on CTW1500.
7. **Inference:** NMS, then BezierAlign and the reader.

**Supervision.** Polygons and transcriptions, no character boxes; 150k synthetic images [paper].

**Text vs graphics.** Boxes come from regression, as in EAST. The reader samples only between the curves, so lines
outside the band never reach it (inference).

**Failures** [paper]: multi-line reading order; calligraphic fonts; curves with more than three crests; weaker than Mask
TextSpotter on Chinese, for lack of character boxes.

**Practicality.**
- AdelaiDet is **non-commercial only** [code].
- The MMOCR port is Apache-2.0 [code], not checked on mps.
- Training took 4 V100s and up to 260k iterations [paper], far beyond our budget. AET alone is cheap.

### 5c. TESTR (Zhang et al., CVPR 2022) and DeepSolo (Ye et al., CVPR 2023)

Sources:
- TESTR: https://arxiv.org/abs/2204.01918 [paper]; https://github.com/mlpc-ucsd/TESTR [code].
- DeepSolo: https://arxiv.org/abs/2211.10772 [paper]; https://github.com/ViTAE-Transformer/DeepSolo [code].

**TESTR steps** [paper]
1. **Backbone and encoder:** ResNet-50 (last 3 stages, no FPN) and a Deformable-DETR encoder (6 layers, 8 heads, 4
   points).
2. **Box guidance:** the encoder proposes coarse boxes; the top 100 seed the queries.
3. **Location decoder:** one query per word, made of 16 polygon-vertex sub-queries (or 8 Bézier points). Attention runs
   within and across words.
4. **Character decoder:** 25 character queries per word. They share the location query's reference points and read
   the image by deformable cross-attention, with **no RoI crop**.
5. **Matching and loss:**
   - Hungarian matching on score and point L1 (characters are not in the match cost).
   - Losses: focal 2.0, point L1 5.0, character CE 4.0, and encoder GIoU 2.0.
6. **Inference and training:** no NMS or thresholds named.
   - Training: 440k iterations on 8 RTX 2080 Ti (about 3 days), then a 20k fine-tune.

**DeepSolo steps** [paper]
1. The encoder proposes a **Bézier centre line** per word and keeps the top 100.
2. **25 point queries** are sampled on each line. Each predicts a refined centre point, top and bottom edge offsets, a
   **character class** (37, background included) and a score; the word's score is the mean over its points.
3. **CTC** maps the 25 per-point classes to the transcription. Text also enters the match cost: focal + 0.5·CTC + L1.
4. It also trains from **centre lines alone** (edge head dropped): 81.6 vs 82.5 with full labels on Total-Text.
   - It stays robust to line shifts up to 50% and shrinkage up to 20%.
   - Speed: 17 FPS with ResNet-50.

**Text vs graphics; splitting words.**
- No px grouping: each word is one query.
- Hungarian matching trains each word to be claimed once, so merges and splits are learned, not post-processed.
- Reading happens at the word's own points; there is never a crop (inference).
- Failures [paper]: TESTR's fixed point count limits very curved text. DeepSolo has unstable edges (invalid polygons),
  reading-order errors, and a word-length cap from N (CTW1500 60.1 → 63.2 going from N = 25 to 50).

**On maps this family wins** [paper: li24, lin25]:
- 2024, Rumsey word detection F: DeepSolo (ViTAEv2-S) 92.0; Mask DINO 87.3; TESTR checkpoint 69.3; CRAFT 58.3;
  DBNet-based 47.2.
- 2025, Rumsey word detection F: DNTextSpotter (DeepSolo plus denoising) 90.2; DeepSolo with NMS 88.9; **fine-tuned
  TESTR baseline 88.5**. Detection plus recognition, same three: 91.1, 90.3, 89.5.
- The 2025 report: "DeepSolo-based methods dominated", and "DETR-based methods are the top performing" for detection.
- The 2024 report: the winner was "very sensitive" to small, low-contrast text on the French cadastre.

**Practicality.**
- TESTR's repo says **Apache-2.0** [code]. But it ships an `adet/` package and installs through AdelaiDet
  (non-commercial). Whether `adet/` carries AdelaiDet's terms is **not checked**.
- DeepSolo's LICENSE **is AdelaiDet's non-commercial licence** [code], so not usable.
- Both need CUDA and compiled ops (TESTR: CUDA 11.3 and `setup.py build develop`) [code].
- Deformable-DETR has a pure-PyTorch fallback [not checked on mps].
- Inference: far beyond a 6-7 min mps budget from scratch. A checkpoint fine-tune might take hours.

## 6. Comparison table

| | Output | Word instances from | Labels | Text touching lines | Compute | Licence (official) |
|---|---|---|---|---|---|---|
| **Ours** | text chance per ink px, kind, direction | thresholded ink → components | LLM word rectangles | line ink must be called non-text px by px, or the component follows it | tiny U-Net, minutes on mps | ours |
| **CRAFT** | char-centre Gaussians and between-char affinity, 1/2 size | components of region OR affinity → rotated box | word boxes + transcriptions | targets ignore ink; merging needs affinity along the line | VGG-16 U-Net | MIT, no trainer; EasyOCR trainer Apache-2.0 |
| **DBNet/++** | shrunk-kernel chance and a learned border threshold | kernel components, unclipped by D′ | word polygons | the kernel follows the word's area; lines past the word stay 0 | ResNet-18, 62 FPS | no licence file; MMOCR and docTR Apache-2.0 |
| **EAST** | shrunk-core score and per-px rotated box (4 distances + angle) | per-px votes → locality-aware NMS | word quads | extent regressed, not traced | PVANet, 1/4 output | none found; argman GPL-3.0; SakuraRiven MIT |
| **PSENet** | n nested kernels | smallest-kernel seeds → BFS outward | word polygons | seeds never reach lines outside the word | ResNet-50 FPN; slow expansion | Apache-2.0 |
| **PAN** | region, kernel, 4-d embedding | kernel seeds + embedding-gated growth (d < 6) | word polygons | a touching px must also match the word's embedding | ResNet-18, 84 FPS | Apache-2.0 |
| **Mask TS v3** | kernel proposals, word mask, 37 char maps, attention text | kernel proposals → R-CNN refine | polygons + text; char boxes from synthetic | hard RoI mask zeros all outside the word | ResNet-50, 250k iters | CC BY-NC 4.0 |
| **ABCNet v2** | per-location score and 8 Bézier points | regression → NMS | polygons + text | the reader samples only inside the curve band | 4 V100s, 260k iters | AdelaiDet non-commercial; MMOCR Apache-2.0 |
| **TESTR** | 100 queries × (16 points + 25 chars) | one Hungarian-matched query per word | polygons + text | no grouping; reads at its own points | 8 GPUs, about 3 days | Apache-2.0 (check `adet/`) |
| **DeepSolo** | 100 queries × 25 centre points (point, edges, char each) | one query per word | centre lines or polygons + text | as TESTR; won MapText on Rumsey 2024/25 | 17 FPS; heavy training | AdelaiDet non-commercial |

## 7. What we are missing, by problem (inference unless cited)

**P1 (line ink called text).**
- Everywhere above, line ink **inside** a word's area is just word region; nobody asks the detector to split those px.
- The reader copes, and Mask TextSpotter v3 and ABCNet keep it from seeing anything **outside** the word.
- So P1 hurts us only because our boxes come from ink components.
- Caveat: the walls read the detector's per-px chance (`2026-11-16-text-reader-survey.md`, premise 3). If they need "is
  this line ink?", keep an ink-level head for them, but do not form words from it.

**P2 (merges and splits).**
- Shrunk kernels separate neighbours by construction and cannot follow a line past the word. Per-px regression makes
  the extent a vote, not a trace. Links (CRAFT affinity, PAN embeddings) replace "touching" with "learned to belong".
- PSENet's m trade-off is our threshold dial. They escape it by growing from small seeds.

**P3 (reader at 46%).**
- ABCNet v2's AET trains the reader on the detector's own matched boxes; Mask TextSpotter v3 zeros everything outside
  the predicted region; TESTR and DeepSolo never crop.
- Our reader has seen only good boxes, never our detector's errors.

**Labels.**
- CRAFT checks its pseudo-labels against the spelling length, and our LLM gives both rectangles and words.
- DB-style methods mask doubtful words out of the loss.
- DeepSolo tolerates rough centre lines (shifts up to 50%). Rough LLM rectangles may hurt a kernel or centre-line
  target less than an ink target.

## 8. Ideas ranked for our three problems (inference, built on the cited mechanisms)

1. **Replace "ink px is text" with a word-kernel target over all px, plus unclip** (DB, PSENet, PAN, SPN).
   - Shrink each LLM rectangle by D = A(1−r²)/L with r ≈ 0.4, and mask doubtful words.
   - Take components of the kernel map, not of the ink, and grow each back by D′.
   - Addresses P1 and P2; one more head on our U-Net.
2. **Regress the rotated box from each kernel px** (EAST RBOX: 4 side distances plus angle; we already predict
   direction), and merge the votes by score (locality-aware NMS).
   - Extents come from votes, not from where ink goes. Addresses P2, and P3 through tighter boxes.
3. **Add a word-membership signal:** PAN's 4-d embedding (pull 0.5, push 3, join under 6) or CRAFT's affinity between
   character centres.
   - It splits a code from an adjacent block number or street name where kernels touch. Addresses P2.
4. **Train the reader on the detector's own boxes** (ABCNet v2 AET), and **mask px outside the predicted word** before
   reading (Mask TextSpotter v3: 71.2 vs 67.3).
   - Addresses P3 directly, with no detector change.
5. **Use the LLM's transcriptions as a label check, CRAFT-style.**
   - Take watershed pseudo character centres from our interim map, and weight each word by s_conf (count against
     spelling length).
   - Fall back to an even split at 0.5; give doubtful words 0.
   - Gives character and affinity targets that never depend on line ink. Supports ideas 1-3.
6. Reference bar, not a fix: query spotters top the map contests.
   - TESTR says Apache-2.0, but `adet/` needs a provenance check.
   - Worth a fine-tune only if ideas 1-4 stall and a longer budget is allowed.

## 9. Not checked, or open

- DBNet and PSENet training schedules: not read.
- Whether Mask TextSpotter v3 trains its readers on SPN proposals or on ground-truth boxes: not stated in what was read.
- PAN++'s Masked RoI: abstract only.
- On mps, none of these were checked: torchvision deformable conv, MMOCR's mmcv ops, the PSE/PA builds, the
  Deformable-DETR fallback.
- An official EAST release: none found in one search.
- Whether TESTR's `adet/` files carry AdelaiDet's non-commercial terms.
- MapText entries differ in effort, data and backbone. Their ranks compare teams, not methods.
- No paper or contest report above measures text crossed by lines. Every statement about lines here is inference from
  the target design.
- The worked examples (60×20 px box → kernels) are my arithmetic.

## Sources

Each section's paper and repo links are at its head. Cited by short name elsewhere:
- li24, MapText 2024 report (Li et al.): https://weinman.cs.grinnell.edu/pubs/li24icdar.pdf
- lin25, MapText 2025 report (Lin et al.): https://weinman.cs.grinnell.edu/~weinman/pubs/lin25icdar.pdf
- DNTextSpotter: https://arxiv.org/abs/2408.00355
- Toolkits: MMOCR https://github.com/open-mmlab/mmocr ; docTR https://github.com/mindee/doctr
