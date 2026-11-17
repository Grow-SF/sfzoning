# Map-text approaches traced step by step (2026-11-17)

Purpose: compare published systems for text on maps and line drawings with ours, step by step, and pick
the ideas most likely to fix our three problems. No code changed. Companion to
`2026-11-16-text-graphics-separation-literature.md` (losses, missing labels, synthetic data); that note's
material is referred to here, not repeated.

Our system: U-Net gives each ink px a text chance, kind and reading direction (labels = LLM word
rectangles) -> threshold + connected components -> rotated word boxes -> CRNN/CTC reader over a closed list
of zoning codes. Problems:
- **P1** line ink that crosses labels is called text;
- **P2** word boxes merge with lines or split (47-79% of code words found);
- **P3** the reader gets 96% right on good boxes but 46% on ours.

Marks:
- **[read]** full text read this session (PDF parsed, or the arXiv HTML); **[code]** official repo README,
  licence file or source read.
- **[abstract only]** / **[secondary]** only the abstract, or another paper's description.
- **[memory]** my own knowledge, not checked; **[unchecked]** could not verify.
- Licences: GitHub API plus the licence file, 2026-11-17.

---

## 1. ICDAR MapText competitions (2024, 2025)

Sources: 2024 report, Li et al. (https://weinman.cs.grinnell.edu/pubs/li24icdar.pdf) **[read]**; 2025 report,
Lin et al. (https://weinman.cs.grinnell.edu/~weinman/pubs/lin25icdar.pdf) **[read]**; platform
https://rrc.cvc.uab.es/?ch=28 and ?ch=32; data https://zenodo.org/communities/icdar-maptext; evaluation
https://github.com/icdar-maptext/evaluation **[code]**.

### Tasks, data, metrics
- **Tasks**: (1) word detection; (2) + linking words into phrases; (3) detection + recognition; (4) all
  three. Output: one polygon per word, plus text and phrase groups.
- **Data**:
  - Rumsey: 2000x2000 tiles from 936 maps (1623-2012); 200 train / 40 val / 700 test tiles (34,518 train
    words, 128,457 test words).
  - French Land Registers (FLR): 19th-century cadastral plans, mostly handwritten, many parcel numbers.
    80/15/50 tiles in 2024, 228/25/144 in 2025.
  - 2025 added Taiwanese topographic maps.
  - 2025 also gave synthetic train tiles: 35,000 Rumsey-style (SynthMap+, section 2), 18,073 FLR-style.
    The FLR-style tiles were rendered with QGIS from today's French cadastre, with names sampled from the
    test area "to integrate contemporary place names in learned language models".
- **Matching**: one-to-one, IoU > 0.5; illegible or truncated words are "don't care".
- **Metrics**:
  - 2024: tightness T = mean IoU of the true positives; PDQ = T x F (detection); PWQ = the same with the
    exact text required; PCQ = T x F x (1 - mean normalised edit distance) for phrases.
  - 2025: harmonic mean of P, R, T, character accuracy and link precision/recall. They changed it because
    the 2024 task-2 winner "predicted no links at all".

### Organisers' baselines
- 2024 "TESTR Checkpoint": TESTR (Deformable DETR, one decoder for polygon control points and one for
  characters) with scene-text weights (SynthText + real scene sets, TotalText fine-tune). No map training.
- 2025 "TESTR Finetuned + Heuristic MST": the same model fine-tuned per set; words linked by a minimum
  spanning tree over distance and similarity. It ranked 4th on Rumsey detection (HMean 88.5) and 1st on
  FLR (80.5).

### Top entries, step by step (the reports; the 2025 appendix F is the teams' own text)
- **"MapText (Detection) Strong Pipeline"** (Bilibili + QUST). Won Rumsey tasks 1/3/4 in 2024 and Rumsey
  tasks 1/3 in 2025.
  1. Augment: crop, scale, saturation, contrast.
  2. DeepSolo, later DNTextSpotter (DeepSolo with denoising training), ViTAEv2-S backbone.
     - The encoder predicts a Bezier centre curve per word.
     - Decoder queries are points sampled along that curve; they output the polygon and the characters.
  3. Pretrain on real scene text only (TextOCR, MLT17, TotalText, IC15, IC13; 35,109 images); fine-tune on
     Rumsey.
  4. Unspecified "post-processing". No lexicon.
  - On FLR (small, faint, handwritten) it fell to 5th in detection; the organisers suspect its training
    lacked such text.
  - Code: https://github.com/yyyyyxie/MapTextPipeline, https://github.com/yyyyyxie/DNTextSpotter **[code]**.
- **MapTest** (Tianjin U.). Won every FLR end-to-end task in 2024.
  1. PP-YOLOE-R rotated-box detector (PaddleDetection **[memory]**), pretrained on COCO-Text v2, fine-tuned.
  2. ABINet reads the crops. It pairs a vision model with a learned language model that corrects it
     **[memory]**. Pretrained on MJSynth + SynthText, then fine-tuned.
  3. Link heuristically: words over 50 px tall, by distance, height and angle.
  - So a two-stage pipeline (rotated box, then a crop reader), like ours, won the cadastral, number-heavy
    set: FLR task 3 PWQ 40.1, against 26.1 for DeepSolo++ and 8.6 for the Rumsey winner.
- **DINO MAP** (FLR detection winner, 2024): MaskDINO (DETR-style detection plus masks), fine-tuned. Each
  tile is cut into 4 overlapping patches and detections are merged by overlap ratio.
  - Report: "with proper fine-tuning, the general object detection models can achieve quite good
    performance".
- **DS-LP** (2024): DeepSolo++, NMS on the polygons, and a LayoutLMv3 relation model for links.
- **MapTextSpotter** (QUST): one decoder with point and character queries. An LLM "replaces the lexicon
  matching process" (no numbers on what it adds).
- **Self-Sequencer** (EPFL; 2025 joint task-4 winner).
  1. DeepSolo reads the words.
  2. Duplicates are merged when the Frechet distance between their Bezier curves is below a threshold.
  3. A transformer pointer orders local word sequences; strongly connected components of the graph become
     phrases.
  - Training data: MapText, mapKurator human labels, SynthMap, and a Paris/Jerusalem set.
  - Code: https://github.com/SesamePaste233/ToponymExtractor **[code]**: patch pyramid (384-2048 px),
    "flattening" of stacked detections, then grouping.
- **CREPE + BezierCurve** (NAVER / LINE WORKS; 2025 joint task-4 winner).
  - A Donut-style model: Swin encoder, BART decoder. It writes `<x><y> text </ocr>`, and a small FFN regresses
    8 Bezier control points.
  - `<s_toponym>` tokens group the words into phrases.
  - Pretrained on ArT, fine-tuned on Rumsey.

### Results (2024, %)
| Entry | Rumsey task 1 PDQ (T / F) | Rumsey task 3 PWQ | FLR task 1 PDQ | FLR task 3 PWQ |
|---|---|---|---|---|
| Strong Pipeline | 76.1 (82.7 / 92.0) | 60.1 | 42.3 | 8.6 |
| MapTest | 73.1 (81.8 / 89.3) | 52.3 | 64.2 | 40.1 |
| DINO MAP | 73.4 (84.0 / 87.3) | — | 64.7 | — |
| TESTR checkpoint (baseline) | 55.1 (79.6 / 69.3) | 27.8 | 20.6 | 2.2 |

2025, Rumsey HMean: task 1 Strong Pipeline 90.2 (P 95.9, R 91.8); task 3 91.1 (character accuracy 94.0); task 4
CREPE and Self-Sequencer 80.8 each. FLR end-to-end: only the baseline, 83.1.

### Touching lines, priors, practicality
- **Touching lines**: no entry models line work explicitly.
  - The reports list the hard cases: "text-like graphical structures" inside spaced words, interposed text,
    binding artefacts, and "large text often overlaps with other elements".
  - The Rumsey labels split a word into several polygons where other words cross it (section 3).
- **Priors**: none in the top entries beyond ABINet's language model and MapTextSpotter's LLM.
- **Licences**:
  - DeepSolo, DNTextSpotter and MapTextPipeline carry AdelaiDet's "for non-commercial purposes" licence
    **[code]**, so none is usable.
  - ToponymExtractor is MIT, but it bundles DeepSolo as `WordSpotter` **[code; files unchecked]**.
  - TESTR declares Apache-2.0 but ships an `adet/` package taken from AdelaiDet **[code]**.
  - MaskDINO: Apache-2.0. PaddleDetection: Apache-2.0, but PaddlePaddle. ABINet: non-commercial.
    PARSeq: Apache-2.0. Evaluation code: GPL-3.0 (run it, don't ship it).
- **mps**: the DETR-family spotters use CUDA deformable-attention ops. Deformable DETR has a slow pure-PyTorch
  path **[memory]**; mps is **[unchecked]**.
- **1-bit**: colour augmentation and colour backgrounds don't apply to us.

---

## 2. mapKurator (USC, now U. Minnesota Knowledge Computing Lab)

Sources:
- Kim et al. 2023, https://arxiv.org/abs/2306.17059 **[read via HTML]**.
- Docs https://knowledge-computing.github.io/mapkurator-doc/ (module pages read in the repo) **[code]**.
- https://github.com/knowledge-computing/mapkurator-system and .../mapkurator-spotter **[code]**.
- Li et al. KDD 2020, https://arxiv.org/abs/2112.01671 **[skimmed]**.
- PALETTE, Lin & Chiang KDD 2024, https://arxiv.org/abs/2506.15010 **[read via HTML]**.

### Execution steps
1. **Input**: full scans (6K-15K px), plus metadata and, where known, ground control points.
2. **Crop**: 1000x1000 tiles. The stitch step's `--shift_size` must equal the tile size, so there is no
   overlap and a word on a tile edge is cut **[code]**.
3. **Spot** each tile end to end: a 16-point polygon plus the text.
   - TESTR.
   - Spotter-v2: TESTR plus feature sampling around the predicted boundary points.
   - PALETTE: "hyper-local" sampling around each boundary point and each character centre.
4. **Train**: pretrain on SynthText (40k text-free COCO backgrounds) plus synthetic maps (SynthMap, Li et
   al. 2021, see the 2026-11-16 note; or SynthMap+), then fine-tune on human-labelled Rumsey tiles.
5. **Merge**: shift each tile's polygons by its offset into a map GeoJSON. No duplicate removal described.
6. **Geo-convert**: pixel coordinates -> longitude/latitude, from the control points (affine, GDAL).
7. **PostOCR** (`m4_post_ocr/post_ocr.py`) **[code]**:
   - strings containing any digit are skipped; for the rest, only letters are kept;
   - Elasticsearch fuzzy match against an OSM name list, edit distance 0 first, then 1;
   - ties go to the most frequent OSM name.
8. **Link**: a word links to an OSM entity when the corrected text is a substring of the entity's name and
   the georeferenced polygon intersects (or lies inside) its geometry.
9. **Sanborn variant** (`m_sanborn/`) **[code]**: geocode the recognised names (ArcGIS, Google, OSM,
   GeoNames); DBSCAN (eps 1.5 km) locates the sheet.
- **First version (KDD 2020)**: Google Vision OCR, a word-linking network, the Google Geocoding API on the
  concatenated words, then a LinkedGeoData match.

### SynthMap+ (from the PALETTE paper)
- **Backgrounds** come from real Rumsey maps: 8x8 cells near detected text; k-means on colour separates
  background from foreground; the background cells are clustered into style profiles.
- **Text**: OSM names placed by the QGIS labelling engine, curved along lines or along polygon edges. Font,
  size and spacing are random.
- **Size**: 73,657 images, 1.1M word annotations. Licence: CC BY 4.0
  (https://zenodo.org/records/14480731) **[code: Zenodo API]**.
- **Self-training**: character centres exist only on synthetic data. On real data, predicted centres that
  fall inside the true polygon become labels, and training repeats until their count is stable.

### Touching lines, priors, results, practicality
- **Touching lines**: no explicit handling. PALETTE fails "when other text or linear features overlap the
  target" and "when many lines surround the text"; the authors' remedy is more varied synthetic data.
- **Priors**: only after OCR (the OSM name list, then a spatial test).
  - Digits are never corrected; codes like "RH-1" and block numbers would pass through untouched.
  - Geography accepts or links a reading; it never narrows what the reader may say.
- **Results**: the demo paper gives scale only (60,000+ maps, 100M+ labels).
  - PALETTE pretrained, on Rumsey-309 (detection F / end-to-end, no lexicon): TESTR 82.3 / 66.0;
    DeepSolo 72.8 / 60.1; PALETTE 84.8 / 69.8. Fine-tuned PALETTE: 90.3 / 75.9.
  - Ablation (end-to-end): SynthText + human labels 55.9; SynthText + SynthMap+ 64.6; all three 69.8.
    Map-like synthetic data beat human labels.
- **Licences**: mapkurator-spotter is CC BY-NC 2.0; the PALETTE weights are "modified from PolyForm
  Noncommercial"; mapkurator-system has no licence **[code]**. Not usable; CUDA 11.3 + Docker.
  Take the ideas only: overlapped tiling, real-scan backgrounds, geographic name lists.

---

## 3. Weinman et al. (Grinnell): MapTD, map text synthesis, gazetteer alignment

Sources:
- ICDAR 2019, https://weinman.cs.grinnell.edu/pubs/weinman19deep.pdf **[read]**.
- ICDAR 2017, https://weinman.cs.grinnell.edu/pubs/weinman17geographic.pdf **[read]**.
- ICDAR 2013, "Toponym recognition in historical maps by gazetteer alignment": slides
  https://weinman.cs.grinnell.edu/~weinman/files/icdar13slides.pdf and https://weinman.cs.grinnell.edu/research/maps.shtml
  **[secondary]**.
- Code: https://github.com/weinman/MapTD, .../MapTextSynthesizer, .../cnn_lstm_ctc_ocr, .../CTCWordBeamSearch
  **[code]**.

### 2019: MapTD + CRNN, step by step
1. **Input**: grey scans. Training uses random 512x512 tiles (no rescaling). Testing uses overlapping
   4096x4096 tiles (stride half a tile), then NMS.
2. **Detector (EAST-like)**: ResNet-50 with a U-Net-style merge. Outputs at 1/4 resolution: a text score, 4
   distances to the edges of the word's rotated box, and the angle.
   - Edges are "semantic": "top" is the top of the letters, "left" is where reading starts.
   - Plain geometric edges cut end-to-end F by 3.6%.
3. **Labels and loss**: score target = the word box shrunk by 0.3; the ring around it, and px where two
   boxes overlap, are "don't care". Dice (score), IoU (box) and cosine (angle) losses.
4. **To words**: threshold the score; every surviving px proposes a whole box; locality-aware NMS merges the
   proposals row by row, weighted by score. **Words come from box votes, not from connected ink.**
5. **Reader**: crop, rotate to horizontal, scale to 32 px high (at least 8 px wide). CRNN variant:
   - paired 3x3 convolutions, time axis downsampled once (2 characters fit in 10 px);
   - two BLSTMs, CTC, beam width 128.
6. **Lexicon inside the decoder**: the beam is limited to a trie of allowed words.
   - Why: map "distractor ink that often looks much like text (e.g., city markers, ... lines)" means the
     free reading may not be near the truth.
   - "Hybrid" mode takes the lexicon word if lambda·P(lexicon best) > (1-lambda)·P(free best), lambda = 0.999.
   - Lexicon: SCOWL English (237,873 words) plus GNIS names for the map's region (median 34,400 per map).
7. **Reader synthesizer**: on the fly, about 200 parameters, 112M images.
   - Distractors: thick boundaries; lines and curves; grids and parallel lines; splotches, points,
     textures; sometimes overlapping text.
   - Text: curved baselines, spacing, holes, opacity.
   - **"Image padding/cropping is added to simulate poor localization."**
   - Noise, blur, JPEG. The repo warns it is not for detector training (2026-11-16 note).

### Results (30 Rumsey maps, 1866-1927, 32,659 words; 10-fold, 27 train / 3 test maps)
- Detection: about 88% precision and recall; leave-one-atlas-out costs 9% F.
- Reading ground-truth crops with the synthetic-trained reader:
  - word error: open 37.9%, closed lexicon 24.1%, hybrid 22.1%;
  - character error: 13.6 / 9.9 / 9.0%;
  - a reader trained on the real maps instead reaches 21.5% (hybrid).
- End-to-end: 62% precision and recall. Their diagnosis: "poor horizontal word beginning/ending
  localization increases end-to-end word recognition error rate". This is our P3 in their words.

### 2013 / 2017: geography inside recognition ("GeoOCR")
1. OCR each word crop; keep its top strings and scores.
2. **Alignment by MLESAC**:
   - sample 3 words, and for each one a gazetteer feature matching one of its top-2 strings;
   - fit an affine (shear <= 0.05, axis-scale ratio <= 1.2);
   - disagreeing words are outliers; repeat for each candidate projection.
3. **EM refinement**: refine the alignment, the outlier rate and the OCR "temperature" (how much to trust
   OCR against GIS).
   - Placement likelihoods: Gaussian around points; uniform along a polyline and falling off with distance;
     the same for polygons.
4. **Re-read with a pixel-specific lexicon**: run the OCR again in closed mode, allowing only names whose
   aligned feature is near the word (inlier test, kappa = 0.95).
5. **Style**: LDA clusters of letter texture learn which categories share a style (rivers in italic).
- **Results**:
  - 2017: word error -36% against OCR alone, -20% against 2013; style cuts toponym-match error 32%. Tested on
    20 maps (8,590 words).
  - 2013: "eliminates 37% of OCR errors" on 20 maps **[secondary]**.

### Touching lines, priors, practicality
- **Touching lines**: the detector learns line ink is not text, and the box votes ignore ink connectivity.
  The reader is trained on line distractors and sloppy crops; the lexicon vetoes readings made of line ink.
- **Priors**: the strongest use of geography in this literature. Location decides which strings the
  reader may output.
- **Licences**: MapTD (TensorFlow 1.x), the synthesizer and cnn_lstm_ctc_ocr are GPL-3.0;
  CTCWordBeamSearch is MIT **[code]**. Reimplement in PyTorch; a trie-constrained CTC beam is small.

---

## 4. Strabo (Chiang & Knoblock) and classical touching-text separation

Sources:
- "Recognizing text in raster maps", GeoInformatica 2015,
  https://spatial-computing.github.io/papers/chiang14geoinformatica.pdf **[read]**.
- Map processing survey, https://spatial-computing.github.io/papers/chiang14survey.pdf **[read in part]**.
- Code: https://github.com/spatial-computing/strabo-text-recognition (no licence) and
  .../strabo-text-recognition-deep-learning (Apache-2.0) **[code]**.

### Strabo, step by step
1. Quantise colours. The user marks 1-12 rectangles of text and non-text.
2. Pick the text colours: for each colour in an example, apply RLSA (closing); the colour that survives best
   is text. Extract it.
3. **Clean**: drop components smaller than half or larger than twice the mean character size.
   - This also deletes characters that touch a large component.
   - The paper accepts it ("most overlapping features in a map have different colors"). It points to Cao &
     Tan for touching text, warning that recovered characters "might be damaged".
4. **Group into strings (CDA, conditional dilation)**: grow components. A px may join only if:
   - it touches 1-2 characters;
   - the two are of similar size (ratio <= 2);
   - curvature stays within 30%;
   - growth stays under 1/5 of the character size.
5. **Orientation (SSOD)**: rotate 0-179 degrees; closing and erosion pick the flat angle.
6. **Read**: send the string and its 180-degree turn to ABBYY FineReader 10; **keep the version the OCR is
   more confident in**. Only ABBYY's built-in dictionary.
- **Deep variant** (2019): EAST + Tesseract 4 on TensorFlow 1.4. Apache-2.0, but it credits GPL-3.0
  argman/EAST.
- **Results** (15 maps, 10 sources, 3 scanned): characters P/R 92.7 / 87.9%; words P/R 82.0 / 77.5%
  (ABBYY alone: words 46.1 / 20.6%).
- **Error causes**: poor scans; look-alike symbols ("Pl" read as "PI", "99" upside down as "66");
  pixelated text; the CDA joining an end character to a nearby non-text object of similar size.

### Cao & Tan 2001-2002, Li et al. 2000, stroke width
- **Cao & Tan** **[secondary only; no full text found]**:
  - Papers: GREC 2001 / LNCS 2390, https://doi.org/10.1007/3-540-45868-9_14; ICDAR 2001,
    https://doi.org/10.1109/ICDAR.2001.953752.
  - Method: thin the image; split components into line segments; "character strokes are short, graphic
    lines are long", plus line continuity and width, settle the overlap; a size filter recovers strokes.
  - Assumes only a few characters touch a line; strings were rotated by hand for OCR.
  - Described by the survey and by Ghafari-Beranghar et al. 2018,
    https://thesai.org/Downloads/Volume9No6/Paper_32-Text_Separation_from_Graphics.pdf **[read]**.
- **Li, Liu & Tan 2000** (IJDAR 2(4)) **[secondary, via the survey]**: extend each line straight through a
  label, using its continuity. Px shared by line and text are **given to both**, so lines stay unbroken and
  letters stay whole. Fails on curves.
- **Ghafari-Beranghar 2018** (Persian city maps) **[read]**: stroke width per px has two histogram peaks;
  px finer than the threshold are line. P > 80%, R > 90%.
- **Practicality**: colour layers are useless on 1-bit scans, and the size filter fails exactly where our
  labels touch borders. What carries over: CDA's "at most two string neighbours, of similar size";
  "keep the most confident reading"; "shared px belong to both".

---

## 5. Engineering drawings and floor plans

- **Moreno-Garcia, Johnston & Garkuwa, IJCNN 2020**, https://doi.org/10.1109/IJCNN48605.2020.9207479
  **[abstract only; the RGU PDF returned 403]**.
  - A CNN classifies each P&ID px as symbol, text or connector, from a window around it.
  - Labels come from their own semi-automatic, heuristic tool (with human help).
  - The window size "strongly" changes accuracy per class. No numbers in the abstract.
- **Jamieson, Moreno-Garcia & Elyan, IJCNN 2020**, https://doi.org/10.1109/IJCNN48605.2020.9207127
  **[abstract only]**.
  - Off-the-shelf deep detector and reader on real P&IDs: 90% of strings found, but "certain non text
    diagram elements were detected as text". 86% of detections read.
  - Text "in close proximity to other drawing elements" is still open.
- **Schlagenhauf, Netzer & Hillinger, CIRP 2022**, https://arxiv.org/abs/2205.02659 **[read]**; code
  https://github.com/2Obe/Text-Detection-on-Technical-Drawings (MIT) **[code]**.
  1. **Generator type 1**: random part geometry with 10-16 dimension labels.
  2. **Generator type 2**: 150 real elements (hatching, bearings, holes) cut from 9 real drawings, scattered
     so they intersect, with 40 targets on top. This teaches what is not text.
  3. **Generator type 3**: 50x35 px crops of 2-4 characters from the closed alphabet (digits, F G H K M,
     +, -, diameter sign), for the reader.
  4. Faster R-CNN detector on 2,000-7,000 generated images; keras-ocr recogniser retrained on 30,000 crops.
  - Results on 9 real drawings (171 targets): generated data 81.9% found, 79.3% characters right; 9 real
    drawings only 4.7% found; out-of-the-box keras-ocr 77.2% / 46.8%.
- **Schonfelder et al., Automation in Construction 2024**, https://doi.org/10.1016/j.autcon.2023.105156
  **[abstract only]**.
  - Tile, detect, read. YOLOv7 was the best of Tesseract, EAST, CRAFT, Faster R-CNN, YOLOv5/v7/v8 and YOLOR
    (97.5 wmAP).
  - PARSeq was the best reader ("85.2% CER"; the wording is ambiguous).
- **Touching lines**: per-px classes from heuristic labels (Moreno-Garcia); real line-work cut-outs as hard
  negatives (Schlagenhauf). Prior: a closed alphabet.
- **Practicality**: Schlagenhauf is MIT and tiny; reuse the idea.

---

## How each family treats text crossed by lines

1. **Ignore it and train on more data** (MapText, mapKurator). They name it as the remaining failure.
2. **Separate by colour or component size** (Strabo). Characters touching big components are lost.
3. **Geometric surgery on the overlap** (Cao & Tan; Li 2000 "both"; stroke width). These assume few
   touches, straight lines or two stroke widths.
4. **Train the models on distractors** (Weinman's reader synthesizer; Schlagenhauf's cut-outs). Labels are
   exact because everything is rendered.
5. **Let the reader and the lexicon veto line-made readings** (trie beam; GeoOCR; Strabo's
   pick-by-confidence).
6. **Boxes from geometry votes, not ink connectivity** (MapTD/EAST; DETR queries). A touching line cannot
   glue two words together.

## Comparison table

| Approach | Detector | Synthetic data | Touching lines | Priors | Results reported | Licence |
|---|---|---|---|---|---|---|
| MapText "Strong Pipeline" | DeepSolo / DNTextSpotter (Bezier centre line, point queries) | none (scene text only) | none explicit | none | Rumsey 2024 PDQ 76.1, PWQ 60.1; 2025 task 3 HMean 91.1 | AdelaiDet non-commercial |
| MapTest (2024 FLR winner) | PP-YOLOE-R rotated boxes + ABINet crop reader | MJSynth / SynthText (reader) | none | ABINet language model | FLR PWQ 40.1 (next 26.1) | Apache-2.0 (Paddle); ABINet non-commercial |
| DINO MAP | MaskDINO | none | none | none | FLR PDQ 64.7 (1st) | Apache-2.0 |
| Self-Sequencer / CREPE (2025) | DeepSolo / Donut-style sequence model | SynthMap (Self-Sequencer) | none | none | task 4 HMean 80.8 | MIT + DeepSolo / none |
| TESTR (baseline) | Deformable DETR, dual decoders | SynthText | none | none | 2024 PDQ 55.1; fine-tuned 2025 HMean 88.5 | Apache-2.0 declared; `adet/` from AdelaiDet |
| mapKurator + PALETTE | TESTR / Spotter-v2 / PALETTE | SynthMap (CycleGAN OSM), SynthMap+ (real backgrounds, QGIS labels) | none; named as failure | OSM names (edit distance <= 1, letters only), spatial link | PALETTE Rumsey-309 det F 90.3, E2E 75.9 | CC BY-NC / PolyForm-NC; data CC BY |
| Weinman 2019 MapTD + CRNN | EAST-like per-px score + box + angle, LANMS | reader only: lines, grids, overlapping text, crop jitter | box votes; distractor reader; trie | SCOWL + GNIS trie, hybrid lambda | det ~88% P/R; word error 37.9 -> 22.1% with lexicon; E2E 62% | GPL-3.0 (beam search MIT) |
| Weinman 2013/2017 GeoOCR | given word crops | none | lexicon veto | gazetteer + alignment -> per-place lexicon | word error -36% against OCR | none released |
| Strabo | colour layer + CC filter + conditional dilation | none | drops touching chars | ABBYY dictionary | word P/R 82.0 / 77.5% | none / Apache-2.0 + GPL EAST |
| Cao & Tan; Li 2000 | thinning + segments; line extension | none | split or share the overlap | none | qualitative **[secondary]** | papers only |
| Moreno-Garcia 2020 | per-px CNN (symbol / text / connector) | heuristic labels | per-px class | none | window size matters **[abstract]** | paper CC BY-NC |
| Schlagenhauf 2022 | Faster R-CNN | generator + real cut-out line work | hard negatives | closed alphabet | 81.9% found, 79.3% chars (9 real drawings) | MIT |

## Ideas for our three problems (ranked; each an inference from the sources)

1. **Word boxes from per-px box votes, not connected components** (P2, also P1).
   - Add MapTD-style heads to the U-Net: 4 edge distances and a semantic angle per text px (we already
     predict a reading direction). Then score-weighted locality-aware NMS.
   - A line touching a code no longer merges or splits the word: every glyph px votes for the same box.
   - Support: MapTD; MapText's lesson that fine-tuned box detectors reach about 90% HMean; MapTest won FLR
     with rotated boxes.
2. **A per-box lexicon from the geography, inside the CTC beam** (P3, also P2).
   - Allowed strings per box: the zoning codes of the lots it lies on or next to (zoning table); the street
     names whose centreline it lies along; the number of the block under it.
   - Keep the hybrid lambda, so a far more likely off-list reading can still win.
   - Support: GeoOCR (-36% word error); the trie beam (37.9% -> 22.1%).
3. **Train the reader on crops like the detector's, not like the LLM's** (P3).
   - Jitter box ends, size and angle; draw lot lines, borders, dashes and hatching through the letters; add
     pieces of neighbouring words.
   - Support: the synthesizer's "padding/cropping to simulate poor localization" and its distractors;
     Weinman's word-end diagnosis; Schlagenhauf's closed-alphabet reader.
4. **Render synthetic sheets from the real geography** (P1, P2).
   - Draw DataSF lots, blocks and street edges as 1-bit line work; place codes, block numbers and street
     names the way the print does; add real line-work cut-outs from text-free crops.
   - Keep exact text / line / both masks and word boxes.
   - Support: the MapText 2025 cadastral generator renders today's cadastre in the old style
     (https://github.com/solenn-tl/synthetic_maps_generator; LGPL-3.0 per Zenodo; PostGIS + QGIS; take the
     idea, not the code); SynthMap+ beat human labels (64.6 vs 55.9); Schlagenhauf's cut-outs.
   - New against the 2026-11-16 note: the layout and crossings come from real lot geometry, not drawn
     guesses.
5. **Choose between box hypotheses by what the reader can read** (P2, P3).
   - Per cluster: merge, split, trim each end, both directions.
   - Score each by the constrained reader's probability times the prior from idea 2.
   - Support: Strabo keeps the reading the OCR is most confident in; GeoOCR re-reads with a narrowed lexicon;
     end-to-end spotters tie box and text in one loss.
6. **(Next) A line prior from the georeferenced lot lines and centrelines** (P1).
   - Project DataSF lot lines and street edges through the existing warp, as a soft extra input channel.
   - Px a projected line shares with a glyph are "both" (Li 2000).
   - Historic lines differ from today's, so feed a likelihood, never a mask.
   - No paper does exactly this; the nearest are Weinman's placement likelihoods and Li 2000.

## What I could not check
- No full text for Cao & Tan, Li 2000, Moreno-Garcia 2020, Jamieson 2020 or Schonfelder 2024; their steps
  come from abstracts or other papers' summaries.
- GeoOCR's 37% (2013) comes from the author's web page, not the paper.
- What the "post-processing" in the Strong Pipeline was.
- Whether TESTR's `adet/` files carry AdelaiDet's non-commercial terms, and which ToponymExtractor files are
  DeepSolo's.
- Whether deformable-attention ops run on mps. PP-YOLOE-R and ABINet internals are from memory.
- No source measures "line ink called text" per px as we do; every number above is word-level (IoU > 0.5).
