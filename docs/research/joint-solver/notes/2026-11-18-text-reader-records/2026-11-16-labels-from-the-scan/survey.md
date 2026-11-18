# Survey: finding and reading the zoning labels on the scans, locally

2026-11-16. Literature and library survey for `docs/superpowers/specs/2026-11-16-text-pipeline-from-the-scan-design.md`
(the detector of word boxes, the reader of every box, the solver's labels from them). Nothing was installed or run;
licences and maintenance were read from the repositories on 2026-11-16 (GitHub API) and from the papers. Numbers are the
authors' own unless marked. Where I infer rather than cite, it says **(inference)**.

## 0. Bottom line

1. **No published work reads zoning-district labels on historic zoning maps.** The nearest are the ICDAR MapText
   competitions (Rumsey general maps; 19th-c. French cadastral land registers), Weinman's gazetteer-prior work on US
   maps, mapKurator's Rumsey/Sanborn runs, and EPFL's Venice cadastre parcel numbers. Evidence for our exact case
   (closed ~70-code list, one typeface per year, heavy boundary lines through labels) is thin. Every number below
   comes from neighbouring problems.
2. **Detection is the easier half; recognition is where systems fail.** In MapText 2024, the best Rumsey detector
   reached F 92.0, but the best word-level end-to-end (exact string) only F 71.3 (PWQ 60.1). Fine-tuned general
   detectors (MaskDINO, PP-YOLOE-R, DBNet variants, TESTR) all did well on detection. Off-the-shelf generic OCR did badly
   on maps: CRAFT+TrOCR, PWQ 12.2; PaddleOCR with preprocessing, detection recall 40% in 2025.
3. **The map-specific spotters that win (DeepSolo, Spotter-v2, PALETTE) can't go into an AGPL-3.0 tool.** DeepSolo
   carries AdelaiDet's non-commercial licence, Spotter-v2 is CC BY-NC 2.0, and PALETTE is a modified PolyForm
   Noncommercial licence. They are also CUDA/Detectron2 stacks trained on 4×A100. TESTR is Apache-2.0, but its weights
   were trained on SynthText, whose terms are non-commercial research only.
4. **The prior × likelihood idea is well supported for maps.** Weinman 2017 jointly inferred toponyms and map
   alignment against a gazetteer. Word error fell from 22.29% to 14.23% (−36%). Weinman 2019 used a CNN-LSTM-CTC
   recogniser with a mixed open/lexicon vocabulary. It beat lexicon-only decoding by 8%.
5. **Recommendation:**
   - **Keep our own U-Net detector.** MapTD, an EAST-style detector on maps, is the direct precedent.
   - **Add a small CTC line recogniser (CRNN-style) trained in-repo on our own crops.** Use Terra's word boxes and
     strings, plus synthetic crops rendered from the existing glyph templates, noise model and measured line overlays.
   - **Use it as the reader's likelihood:** score every candidate string exactly with the CTC forward algorithm; let
     the CTC remainder be "other text" and the empty string be "no text"; add a 0°/180° crop-orientation classifier.
     Combine with the location priors already in the spec (code list, nearby centreline names, nearby block numbers).
   - **Calibrate** with a temperature and a prior weight, cross-fitted.
   - Off-the-shelf code to start from: **docTR** (Apache-2.0, pure PyTorch, supports `mps` and CPU, CRNN/PARSeq models,
     multi-class rotated-polygon training). Its pretrained weights' training data is not disclosed. So the spike tests
     training from scratch against fine-tuning from docTR's weights, and the shipped tool should prefer from-scratch
     weights if they are as good.
   - **The spike (section 4.6):** crops from the 2000 map's training quarters, a CRNN-CTC model, scored on the held-out
     quarter and on the owner's answers. Compare top-1, −log p and reliability with the generative reader's records,
     plus bitwise determinism and crops per second on CPU.

## 1. Research on map text

### 1.1 Weinman (Grinnell): gazetteer priors, MapTD, CNN-LSTM-CTC

- **Toponym recognition by gazetteer alignment (ICDAR 2013).** OCR words were aligned to gazetteer locations, giving
  pixel-specific lexicons. The authors' slides report about 20% better word accuracy from the gazetteer.
- **Geographic and style models (ICDAR 2017, best paper).**
  - The model is a Bayesian network over the strings, the map projection/alignment and the label↔feature links.
  - The gazetteer is USGS GNIS: points, plus distance-transform likelihoods for rivers and regions.
  - Inference: RANSAC, then EM, then rescoring.
  - Text style is clustered (LDA over LBP features) and linked to feature category.
  - Data: 20 maps, 6,949 words (where the right string was in the OCR candidates and in the gazetteer).
  - Results: word error 22.29% → 14.23%, character error 10.19% → 6.75% (removes 37% of OCR errors). The abstract
    reports −36% word error over OCR alone and −32% toponym-matching error from the style links.
  - **For us:** the prior from nearby centrelines and parcels in our spec is this model, with streets as line
    features, as in Weinman's rivers.
- **Deep networks for map text (ICDAR 2019).**
  - **Detector MapTD:** EAST adapted with a ResNet-50, sigmoid output and Dice loss. It predicts the *semantic*
    orientation (the text's baseline and reading direction), not EAST's geometric angle. That raised end-to-end F by
    3.6%.
  - Detection: P 90.45 / R 86.56 / F 88.46 under ten-fold cross-validation on 30 maps (~30k labels).
  - Leave-one-atlas-out costs 9% F, so style familiarity matters.
  - **Recogniser:** a CRNN-like CNN + BiLSTM with CTC.
    - Trained on a *dynamic synthetic map-text generator* (MapTextSynthesizer: random captions, letter spacing,
      distractor lines and other words in the background), more than 112M distinct images.
    - MJSynth (scene text) was "ill-suited" to map text.
    - Cropped-word results: 9% character error / 22% word error. Fine-tuned on real map crops: 7.07% CER, 18.29% WER.
    - 512 LSTM units were needed: 256 raised open-vocabulary CER by 43%.
  - **Lexicon:**
    - Decoding uses CTC word beam search (Scheidl et al. 2018) restricted to a trie.
    - "Hybrid vocabulary": pick the best lexicon word if λP(ĉ_L|x) > (1−λ)P(ĉ_U|x), else the best open word, with
      λ = 0.999 (a −3 log-prob bias against non-lexicon words).
    - Mixed beat closed-vocabulary by 8%.
    - The authors note the assumption P(x|L) = P(x) "is false in general but works well in practice".
  - **Annotation convention:** "Interrupted or highly spaced words are annotated in multiple rectangles."
  - Code: MapTD, cnn_lstm_ctc_ocr and MapTextSynthesizer are all **GPL-3.0** (AGPL-compatible) but TensorFlow 1.x.
    The map images are CC BY-NC-SA 3.0.
- **What carries over:**
  - Text-specific detectors trained on a few dozen maps reach high-80s F.
  - Recognition needs either lots of synthetic map-like data or in-domain crops.
  - Lexicon restriction must keep an escape to open vocabulary, because distractor ink makes the open best path
    unreliable. A pure closed list is worse than a mixed one.

### 1.2 Chiang / Knowledge Computing Lab (Minnesota, formerly USC): Strabo, multi-period maps, mapKurator, PALETTE

- **Yu, Luo, Chiang (ICPR 2016).**
  - Method: combine OCR of the same area across maps from several time periods, plus OSM names.
  - Results against Strabo: +153% precision, +31% recall, +75% F at word level.
  - **For us (inference):** San Francisco has about 10 printed years of the same streets and blocks. Street-name and
    block-number hypotheses can be shared across years (codes change, streets mostly don't). This helps the "explain
    the ink" role and a new map's first placement.
- **mapKurator system (2023 demo paper).**
  - Pipeline: tile → spot (Deformable-DETR/TESTR-based) → merge → post-OCR → geocoordinates → link to OSM.
  - Post-OCR is a nearest edit-distance word in an OSM-derived vocabulary, ties broken by popularity. It is a
    post-processor, not a joint prior, and the paper gives no accuracy numbers.
  - Run on 60k+ Rumsey maps and "thousands" of Sanborn sheets.
  - **Licences:** Spotter-v2 code and weights are **CC BY-NC 2.0** (repository LICENSE and HF model card). The system
    repository has no licence.
- **PALETTE (arXiv 2506.15010, 2025).**
  - Architecture: Deformable DETR with "hyper-local" sampling (boundary-point and character-centre queries), aimed at
    long, curved and widely spaced text.
  - Training data: SynthMap+ (about 73,657 synthetic map images, 1.1M annotations; QGIS label placement along OSM
    features over real Rumsey textures), plus SynthText, ICDAR15, Total-Text and MLT.
  - Rumsey-309 fine-tuned: detection F 90.31, end-to-end with no lexicon 75.87 (TESTR 89.21 / 73.44; DeepSolo
    83.28 / 68.52).
  - Trained on 4×A100 for 400k pretraining iterations.
  - **Licence:** "Palette Pre-trained Model License (modified from PolyForm Noncommercial)", shipped as a CUDA 11.3
    Docker image. Not usable here.

### 1.3 ICDAR MapText 2024 and 2025

**Data:**
- Rumsey: 936 maps, 2k×2k tiles. Train 200 tiles / 34,518 words; test 700 tiles / 128,457 words.
- French Land Registers (cadastral, mostly handwritten): 8,096 training words in 2024, 25,564 in 2025.
- 2025 added Taiwanese maps and synthetic tiles (35k Rumsey, 18k French, 45k Chinese).
- Licences: Rumsey annotations **CC BY-NC-SA 4.0**, images CC BY-NC-SA 3.0. IGN/French data CC BY-SA 4.0.

**2024 results (Li et al.):**
- **Detection, Rumsey:** won by Bilibili's "Strong Pipeline" (DeepSolo + ViTAEv2-S, pretrained on about 35k scene-text
  images): PDQ 76.1, F 92.0. Close behind: MaskDINO fine-tuned (F 87.3) and PP-YOLOE-R fine-tuned (F 89.3).
- **Detection, French:** won by MaskDINO.
- **End-to-end, exact string:**
  - Rumsey best: PWQ 60.1 (P 76.0 / R 67.2).
  - French best: MapTest (PP-YOLOE-R detector + ABINet recogniser fine-tuned): 40.1.
  - The DeepSolo pipeline collapsed on the low-contrast French data (PWQ 8.6). The organisers note sensitivity to small
    and low-contrast text.
  - CRAFT (EasyOCR) + TrOCR ranked last (PWQ 12.2).
  - Organisers' conclusion: "transcription is far from being solved"; greater success "may require advanced
    contextualization", for example jointly reading sequences of plot numbers.
- **Linking:** the winner by the 2024 PQ metric predicted no links at all, so the metric was changed.

**2025 results (Lin et al.):**
- Metrics are now harmonic means with character accuracy. Detection P/R/T are no longer tied to an exact string match.
- **Detection, Rumsey:** best HMean 90.2 (DNTextSpotter: DeepSolo with denoising training). A fine-tuned TESTR baseline
  reached 88.5 and was best on the French data (80.5).
- **Character accuracy on matched words:** 94–95.5% for the leaders.
- **PaddleOCR** with preprocessing scored detection recall 40.0.
- **Linking:** CREPE (a Donut-style generator with Bézier control points) and EPFL's Self-Sequencer tied at HMean
  80.8. The heuristic MST baseline reached link precision 27.1.

**Lessons:**
- Rotated and curved text is the norm on maps, and the best systems output polygons from Bézier curves or control
  points.
- Letter-spaced words with other ink between the letters are the named detection failure.
- Detection is "close to solved" with fine-tuning. Recognition and linking are not.
- Training sets of about 200 tiles (about 35k words) suffice to fine-tune DETR spotters that were pretrained on scene
  text.
- No competitor used a geographic prior. One 2024 entry (MapTextSpotter) used an LLM to refine recognition.

### 1.4 Cadastral, large-scale city and zoning maps

- **French Land Registers (MapText):**
  - Cadastral plans full of parcel numbers.
  - Best 2024 end-to-end, from fine-tuning a separate detector and recogniser, was 40.1 PWQ. That is much lower than
    Rumsey, owing to handwriting and low contrast.
  - Numbers there are read without context. The organisers point at sequence context, which for us is the block and
    lot structure.
- **EPFL Venice Napoleonic cadastre (Ares Oliveira et al., DH 2019):**
  - Training data: about 1/3 of one sheet (about 800 parcels).
  - Parcel identifiers were rotated to horizontal and split into digits. Only about 10% of identifiers were read
    fully right, blamed on digit segmentation.
  - The authors suggest sequence recognisers (LSTM/CTC) instead. This supports a segmentation-free CTC reader.
- **Sanborn maps:**
  - mapKurator ran on them. A UT Libraries project OCRs street labels at detected intersections and cross-checks them
    with OSM.
  - No published accuracy for Sanborn street or number reading was found.
- **Hamburg large-scale maps (Schlegel, AGILE 2021):**
  - Method: Strabo detection plus Tesseract, matched to current street names by Levenshtein distance.
  - Rotating the input by ±45° and ±90° raised detection recall by about 50% relative (for example 41% → 58% → 66%).
    This shows how badly upright-only OCR does on maps.
- **Zoning:** the Urban Institute's zoning-data work OCRs zoning *documents* (Textract) and matched 55% of districts.
  It does not read maps. Nothing found reads printed district codes on zoning maps.

### 1.5 Linking words into phrases

- **Our need:** multi-word district names (e.g. "UPPER MARKET … NCD"), and labels on two lines, which the spec makes a
  non-goal.
- **Published linkers:**
  - LIGHT (2025: LayoutLMv3 + polygon encoder + image) reaches link F 83.7 on ground-truth words.
  - Self-Sequencer and CREPE: see 1.3.
  - All are heavy transformers trained on Rumsey (NC-SA data).
- **Inference:** with a closed list of district names, "does the concatenation of nearby boxes, in reading order,
  spell a known name?" is a much stronger signal than anything those general linkers have. A lexicon-scored link
  (section 3) is the practical route. Evidence is thin; no paper tests it.

### 1.6 How the literature handles our specific difficulties

| Difficulty | What's done | Evidence |
|---|---|---|
| Any rotation, upside down | Detect a rotated box, rectify the crop, read upright. Direction: predict a semantic orientation (MapTD: +3.6% e2e), or classify 0/180 per crop (docTR crop-orientation classifier, PaddleOCR text-line orientation classifier), or try rotations (EasyOCR `rotation_info`). | MapTD; library docs; Schlegel 2021 (rotation raised recall by ~50% relative) |
| Curved street names | Bézier/polygon outputs (ABCNet, DeepSolo, TESTR, PALETTE), TPS rectification (ASTER). | MapText winners |
| Letter spacing | Annotate as several boxes (Weinman) or one polygon (MapText). Synthetic training with random spacing. Character-centre queries (PALETTE). For CTC, wide crops just add blank frames. | Weinman 2019; PALETTE |
| Lines through text | Train with distractor lines (MapTextSynthesizer). Classical text/graphics separation. Generative models mark line pixels unobserved (our current reader). | Weinman 2019; Chiang et al. 2014 survey |
| Tiny text | Fails first. MapText organisers single out small text, and the French collapse came from it. Read crops at full resolution. | MapText 2024 |
| Little labelled data | Synthetic map text (MapTextSynthesizer, SynthMap+, IGN generator) plus fine-tuning. With one typeface, few lines suffice: Reul et al. 2017 cut errors 43% (60 lines) and 26% (150 lines) by fine-tuning a mixed model instead of training from scratch. | Weinman 2019; PALETTE; Reul 2017 |
| Lexicons and priors | Gazetteer/location priors in a joint model (Weinman 2017: −36% WER). Trie-constrained CTC beam with an open escape (Weinman 2019: mixed −8% vs closed). Edit-distance post-correction (mapKurator: no numbers). | as cited |

## 2. Libraries usable locally

AGPL-3.0 fit uses the FSF licence list:
- Apache-2.0, MIT and BSD code can go into GPLv3/AGPLv3 work.
- GPLv3 code combines with AGPLv3 (GPLv3 §13).
- CC BY-SA 4.0 is one-way compatible with GPLv3.
- Non-commercial or use-restricted terms (CC BY-NC, AdelaiDet's licence, PolyForm NC, OpenRAIL-M) add restrictions
  AGPL forbids, so they are **incompatible**.
- **Weights trained on non-commercial data** (SynthText's terms: non-commercial research and education) are a legal
  grey area. The libraries ship such weights under Apache-2.0 anyway. For a published AGPL tool, from-scratch weights
  trained on our own crops avoid the question.

| Library | Code licence | Pretrained weights: licence, data | Maintained? (2026-11-16) | Rotation | Fine-tune on our rotated boxes | p(string\|image) over a candidate list | CPU, determinism | AGPL fit |
|---|---|---|---|---|---|---|---|---|
| **docTR** (Mindee) | Apache-2.0 | Served from Mindee's static host; training data not disclosed in docs ("trained on our french vocab") | Active: v1.1.0 Aug 2026, pushes Nov 2026 | Rotated quads; page and crop orientation classifiers (0/90/180/270) | Yes. Reference train scripts with `--device mps`/`cpu`. Detection takes 4-point polygons, **multi-class** (our kinds). Recognition takes crop + string, custom vocab. | CRNN models give CTC logits: score any list exactly with `ctc_loss` (the model's own loss uses `F.ctc_loss`). PARSeq/ViTSTR/MASTER: teacher-force. | crnn_mobilenet_v3_small 0.05 s / batch of 64 on a laptop i7 (docs); vgg16 0.6 s. Pure PyTorch: CPU inference reproducible with fixed threads (verify). ONNX via OnnxTR (Apache-2.0). | **Good** (code); weights' data provenance unknown |
| **PaddleOCR** (PP-OCRv5) | Apache-2.0 | Apache-2.0; trained on Baidu's mix (public + synthetic + internal, not fully listed) | Very active: v3.7.0 Jun 2026 | DB detector gives quads; optional text-line 0/180 classifier; v5 claims better rotated/vertical text | Yes (YAML configs, det + rec). PaddlePaddle on macOS is **CPU-only** (arm64 wheels exist). | Recogniser deploys the CTC branch: log-probs available, score with any CTC implementation. | Fast mobile models; ONNX via RapidOCR (Apache-2.0). A second DL framework next to torch. | Good licence; heavy framework; MapText 2025 entry did badly unadapted |
| **MMOCR** | Apache-2.0 | Apache; public datasets (incl. SynthText/MJ) | **Stale:** last release v1.0.1 Jul 2023; mmcv last release Apr 2024 | DBNet++, FCENet, TextSnake, PSENet | Yes in principle; mmcv builds on Apple Silicon are a known pain | CTC models (CRNN, SVTR) can be scored; attention models by teacher forcing | | Avoid as a dependency; useful reference code |
| **EasyOCR** | Apache-2.0 | CRAFT (MIT code; weights trained on SynthText+IC13/IC17); recogniser VGG-BiLSTM-CTC on synthetic data | Active pushes; last release v1.7.2 Sep 2024 | `rotation_info=[90,180,270]` re-reads rotated copies; CRAFT gives free quads | Recogniser via a fork of clovaai deep-text-recognition-benchmark; CRAFT trainer in `trainer/` | Decoders: greedy, beam, `wordbeamsearch` (dictionary); CTC logits reachable in code | OK on CPU | Code OK; weights touch SynthText; CRAFT+TrOCR did badly in MapText 2024 |
| **keras-ocr** | MIT | CRAFT + CRNN | **Dead:** last release 2020 | | TF/Keras | | | Avoid |
| **Tesseract 5** | Apache-2.0 | Apache-2.0 traineddata | Active: 5.5.4 Oct 2026 | Upright lines only (page OSD 0/90/180/270) | `tesstrain` on line images | No API for p(given string); per-character alternatives only via choice iterators | Fast, deterministic | Licence fine; poor on map text (Schlegel 2021) |
| **TrOCR** (microsoft/unilm) | MIT | HF cards list no licence; small-printed = DeiT + UniLM, fine-tuned on SROIE | Code maintained in unilm | None (upright line images) | HF fine-tuning | Teacher-forced log-prob of any string; strong internal LM | 61M–558M params, slow autoregressive on CPU | Code OK; weights' licence unstated; worst MapText 2024 entry (with CRAFT) |
| **PARSeq** (baudm) | Apache-2.0 (ABINet BSD, CRNN MIT) | Synthetic (MJ+ST) or "real" mix (COCO-Text, TextOCR, ArT, LSVT, MLT19, ReCTS, Uber…; mixed terms) | Repo quiet (v1.0.0 2022, last push 2024); maintained inside docTR and PaddleOCR | Abstract claims robustness to arbitrarily oriented text (no numbers given) | Hydra configs; custom charset | AR teacher forcing gives exact log p(s\|x) under the model; permutation-LM ensemble; internal LM | ~24M params | Code OK; weights' data mixed |
| **Kraken** | Apache-2.0 | Zenodo models, various CC licences | Active: 7.1 Aug 2026 | Baseline/polygon line extraction (curved lines) | `ketos train` on line image + text | CTC: per-char confidences; full logits in code | CPU-friendly | Good licence; built for manuscript lines, not map words |
| **Calamari** | GPL-3.0 | Various | Active (push Jun 2026) | Lines only | Yes (TF) | CTC | | Compatible (GPLv3), TensorFlow |
| **Weinman MapTD + cnn_lstm_ctc_ocr + CTCWordBeamSearch** | GPL-3.0 / GPL-3.0 / MIT | Checkpoints from synthetic map text | Dormant since 2021 (TF 1.x); CTCWordBeamSearch active | MapTD: semantic orientation | TF 1.x | CTC with trie lexicon (word beam search) | | Compatible; obsolete stack; ideas reusable |
| **TESTR** | Apache-2.0 (Detectron2, Deformable DETR) | Google-Drive weights: pretrained on SynthText-family synthetic data + human-annotated scene sets, fine-tuned on Total-Text (MapText 2024 report); non-commercial terms in the mix | Dormant (last push Jan 2023) | Polygons / Bézier | Detectron2 training; CUDA deformable attention (CPU fallback exists, slow) | Per-query character classification at fixed slots (max 25): candidate scoring awkward | Heavy on CPU | Code OK; weights grey; MapText baseline 88.5 detection after fine-tuning |
| **DeepSolo / DNTextSpotter / MapTextPipeline** | **AdelaiDet licence: non-commercial** | | | | | | | **Incompatible** |
| **mapKurator Spotter-v2** | **CC BY-NC 2.0** | CC BY-NC 2.0 | Last push Jun 2024 | | | | | **Incompatible** |
| **PALETTE** | **Modified PolyForm NC**, Docker/CUDA | | | | | | | **Incompatible** |
| **Surya** | Apache-2.0 | **Modified OpenRAIL-M** (revenue-capped commercial use) | Active | | | | | Weights **incompatible** |

**Practical reading of the table:**
- Only docTR, PaddleOCR, Kraken and a hand-written CRNN are realistic on this Mac with permissive licences.
- docTR is the closest fit:
  - PyTorch, the framework already used here;
  - `mps`/CPU training scripts;
  - multi-class rotated-polygon detection data (an alternative to, or a check on, our U-Net);
  - CTC recognisers whose logits can be scored exactly.
- PaddleOCR is stronger on generic benchmarks but brings a second framework that is CPU-only on macOS.
- The rest are dead, mis-licensed, CUDA-bound or built for document lines.

## 3. Prior-constrained recognition: methods and practicality

### 3.1 Exact scoring of a candidate string

**CTC (CRNN, docTR crnn_*, PaddleOCR rec, EasyOCR, Kraken, Weinman's recogniser):**
- The model gives per-frame label distributions. p(s|x) for any string s is the sum over all alignments, computed
  exactly by the forward algorithm (Graves et al. 2006). In PyTorch that is `-F.ctc_loss(log_probs, s,
  reduction='none')`. Scoring 70 codes × printed forms plus a few dozen street and block candidates per crop is cheap.
- Three properties that suit this project **(inference from the CTC definition)**:
  - **CTC is a proper distribution over all label sequences.** So p(other text | x) = 1 − Σ_{s∈candidates} p(s|x),
    exactly. This turns Weinman's heuristic "mixed vocabulary" into an exact mass. The open-vocabulary best string
    (prefix beam search, Hannun et al. 2014) is only needed to *show* what "other" looks like.
  - **p(""|x), the all-blank path, is a "no text" score**, provided training includes non-text crops with empty
    targets. CTC allows empty targets.
  - **Frames are conditionally independent given the image.** So CTC's internal language model is weak, and an external
    prior multiplies in cleanly (see 3.2).
- Constrained search for large lexicons: CTC word beam search (Scheidl, Fiel, Sablatnig 2018; MIT code, used by
  Weinman 2019). Our lists are small enough to enumerate exactly.

**Attention/autoregressive (PARSeq, TrOCR, ABINet, MASTER):**
- log p(s|x) = Σ_t log p(s_t | s_<t, x) by teacher forcing, exact under the model, one decoder pass per candidate.
- These models learn a strong internal LM from their training strings. On short codes that is a risk: they can "read"
  a frequent code from context. It also has to be divided out before applying our prior (internal-LM estimation,
  Meng et al. 2021). **Inference:** for a ~70-code closed list, that cost buys little.

**Whole-word classification (Jaderberg et al. 2016, 90k-word dictionary classifier):**
- Possible for 70 codes.
- Can't express "other", needs examples of every code, and doesn't share letters across rare codes such as C-3-O(SD).
  CTC's compositionality handles rare codes better **(inference)**.

**Generative (Document Image Decoding, Kopec & Chou 1994; Ocular, Berg-Kirkpatrick et al. 2013; our current reader):**
- Computes p(ink | string) directly, so the prior multiplies in without correction, and line pixels can be marked
  unobserved.
- Ocular beat Tesseract by 47% relative WER on historical print in 2013.
- Since then, discriminative CTC line models fine-tuned on a few hundred lines reach about 98% character accuracy on
  early printed books (Reul et al. 2017; Calamari/OCRopus line). They are the standard for one-typeface historical
  print.
- Our generative reader's measured limits (11-15/16 ledger):
  - hours per full run;
  - sensitivity to layout freedom ("other" beating crisp codes);
  - box top-1 814/885 on training labels.
  - These match the usual weaknesses of hand-built likelihoods **(inference)**.

### 3.2 Combining a discriminative likelihood with a location prior

We want p(s | x, location) ∝ p(x | s) · p(s | location). A discriminative model gives p_train(s | x) ∝ p(x | s) ·
p_train(s), so:

  log score(s) = log p_model(s | x) − β·log p_train(s) + λ·log p(s | location)

This is the "scaled likelihood" of hybrid NN/HMM speech recognition (Bourlard & Morgan 1994) and the density-ratio
fusion of end-to-end ASR (McDermott, Sak, Variani 2019). Weinman 2019's mixed vocabulary is the special case β = 0 with
a fixed λ, and he notes that the needed independence "is false in general but works well in practice".

**Practical recipe (inference, built from those papers):**
- β and λ, plus a temperature on the logits (Guo et al. 2017), are three calibration constants. Fit them by
  cross-fitting, as the spec already allows for the temperature and the "other"/"none" masses.
- With CTC, β ≈ 0 is a reasonable start. That is Weinman's assumption, and CTC's per-frame independence keeps the
  learned string prior small.
- **The priors themselves are the spec's:**
  - code-list frequencies and printed forms;
  - street names of centrelines along the box, weighted by distance and angle, including **fragments**, because
    letter-spaced names get split (Weinman's annotation note);
  - block numbers of the containing and neighbouring blocks;
  - a character bigram for "other";
  - "no text" from the detector.
- **Direction:** the reading direction is a latent variable d ∈ {0°, 180°} along the box axis:
  p(s|x) = Σ_d p(d|x) p(s | rot_d(x)), with p(d|x) from a small crop classifier. docTR and PaddleOCR ship exactly
  this kind of classifier. It is trained from Terra's angles, which carry the reading direction even though the
  detector's new axis output drops it.
- **Kinds:** the posterior is normalised jointly over codes, street candidates, block candidates, "other" and "none".
  A street-name crop is then not forced into the code list, which is the "explain the ink" requirement.

### 3.3 Where priors have been shown to pay on maps

| Work | Prior | Effect |
|---|---|---|
| Weinman 2013 | gazetteer, after automatic alignment | about +20% word accuracy (authors' slides) |
| Weinman 2017 | gazetteer + joint alignment + style↔category | WER 22.29% → 14.23% (−36%) |
| Weinman 2019 | map-specific lexicon, mixed with open vocabulary | mixed −8% error vs closed-only |
| Yu, Luo, Chiang 2016 | same area in other years + OSM names | +75% word F vs Strabo |
| mapKurator 2023 | OSM vocabulary, edit-distance post-correction | not reported |

Our priors should be stronger than any of these **(inference)**: the code list is closed (~70 strings), centrelines
and parcels are exact modern geometry, and the warp already places each sheet.

## 4. Recommendation for this project

### 4.1 Detector: keep the U-Net, with the spec's axis and box outputs

- **Precedent:** MapTD is the same design family (per-pixel text score + box geometry + orientation), trained on about
  30 maps to F 88.5.
- **Literature agrees** detection is the solvable part with fine-tuning.
- **Mod-180 axis:** the spec's choice (axis modulo 180°, direction left to the reader) gives up MapTD's measured +3.6%
  from semantic orientation. But the reader scores both directions with a direction classifier (3.2), which recovers
  that information where it matters.
- **Optional cross-check:** fine-tune docTR's `db_resnet50` or `fast_base` on Terra's polygons with our kinds as
  classes, on the same held-out quarters. If it is much better on recall, the U-Net recipe is the problem. It needs only
  data conversion and docTR's own scripts.
- **Read crops from the full-resolution scan, not the half-size detector input.** Small text is the documented
  first failure (MapText 2024), and our hard labels are under about 23 px tall.

### 4.2 Reader: a CTC recogniser as the likelihood, inside the spec's prior × likelihood frame

- **Model:**
  - A small CRNN: a conv stack, then a BiLSTM (Weinman found 512 units necessary on diverse maps; one typeface may need
    fewer), then CTC.
  - Vocabulary: A–Z, 0–9, `-()/.&'` and space.
  - Input: rectified crops at a fixed height (32–48 px), aspect kept, padded by a fraction of the height. Variable
    width means letter spacing becomes blank frames, not squashed letters.
  - Write it in-repo (a few hundred lines of PyTorch), or use docTR's `crnn_mobilenet_v3_*` / `crnn_vgg16_bn`
    classes and training script.
  - Start the spike with docTR's code and weights. Ship from-scratch weights if they are as good: provenance, plus
    the owner's "bootstrap from primary sources" rule.
- **Training data:**
  - Terra's word boxes and strings on all years' training quarters, all kinds (codes, streets, block numbers, lot
    dimensions, other). Crops in both directions train the 0°/180° classifier. Non-text crops get empty strings.
  - **Plus a synthetic stream** with exactly known truth: strings from the code list, street names and block numbers;
    rendered with the existing learned glyph templates and Kanungo noise model; measured sizes, spacings and line
    overlays.
  - This is MapTextSynthesizer / SynthMap+ made in-domain. It also corrects for Terra's misreads, which are
    concentrated in the hard cases (fused lines, tiny, sideways).
- **Line interference:**
  - Learn it from real crops, which contain the lines, and from synthetic overlays.
  - Optional ablation: give the long-line mask as a second input channel, so the network can learn to discount those
    pixels. That is the discriminative counterpart of "line pixels unobserved". It is an idea, not literature-backed.
- **Scoring:**
  - For each detected box and its alternatives, and each direction: CTC log-likelihoods of all candidates.
  - "Other" is the CTC remainder; "none" is the empty string.
  - The detector's kind prior and the location prior combine as in 3.2. Temperature, β and λ are cross-fitted.
  - Only the code posterior goes to the solver.
- **Determinism:**
  - Inference on CPU in float32 with fixed `torch.set_num_threads` and `torch.use_deterministic_algorithms(True)`.
    Check bitwise across two runs and two thread counts in the spike.
  - Training may use `mps` for speed, one process at a time (ledger: concurrent mps runs gave NaN gradients). The
    shipped weights are a DVC-tracked stage output.
  - PyTorch documents nondeterminism for CTC backward on CUDA, and MPS makes no determinism promise. So the
    reproducible artefact is the weights file plus CPU inference.
- **Runtime:** docTR's own benchmark puts a mobile CRNN at about 1 ms per crop on a laptop CPU, and candidate scoring is
  a batched `ctc_loss`. So a year's worth of boxes (tens of thousands × 2 directions × alternatives) is well within the
  spec's "minutes".

### 4.3 What happens to the generative reader

- Its learned glyph templates, noise model and measured layout statistics become the **synthesiser** for the CTC
  reader's training data.
- Its guards, records and scoring harness stay.
- Its slow pose search leaves the inference path.
- If the spike shows the two disagree in useful ways, a product of the two likelihoods is possible. That has no
  evidence behind it; decide by the measures, not by default.

### 4.4 Licence check (for what is recommended)

- PyTorch (BSD-3), docTR (Apache-2.0), OnnxTR/onnxruntime (Apache-2.0/MIT) and CTCWordBeamSearch (MIT, only if
  needed) are all compatible with AGPL-3.0.
- docTR's pretrained weights are Apache-labelled, but their training data is undisclosed. Use them only as a spike
  baseline or initialisation, and prefer from-scratch shipped weights.
- **Don't use:** DeepSolo, DNTextSpotter or MapTextPipeline (AdelaiDet non-commercial), Spotter-v2 (CC BY-NC),
  PALETTE (PolyForm NC), or Surya weights (OpenRAIL-M).
- **Don't train shipped weights on MapText Rumsey data** (CC BY-NC-SA). The IGN French data (CC BY-SA 4.0) is allowed
  but of little use (handwritten French).
- SynthText (non-commercial terms) sits under most scene-text weights. That is another reason for from-scratch weights.

### 4.5 Main risks

1. **Terra's misreads become training labels.** They are concentrated in exactly the hard cases.
   - Mitigations: synthetic crops with known truth; weighting by agreement across the saved passes (training data
     only); and judging only on the owner's answers and the held-out quarter.
   - **Policy point for the owner:** the spec says Terra's responses "may train and validate the text detector, and
     nothing more". The glyph templates were already learned from Terra's strings, but training a recogniser on them
     should be confirmed explicitly.
2. **Overconfidence.** Neural posteriors are miscalibrated (Guo et al. 2017), and a confident wrong code in a narrow
   strip renames the strip. Report the reliability table and fit the temperature before the solver sees anything.
3. **Fragmented and letter-spaced street names.** The detector splits them, and fragments must still be explained, by
   fragment candidates in the street prior. Otherwise "other" or a code soaks them up.
4. **Style shift across years.** Weinman's leave-one-atlas-out cost 9% detection F. The serif years need their own
   synthetic font and some crops from those years.
5. **Determinism.** Verify bitwise CPU inference. Keep training reproducible enough for DVC, or treat the weights as
   the versioned input.
6. **The scoring sets are noisy** (spec). Report bootstrap intervals and look at the disagreements.

### 4.6 The cheap spike to run first

Goal: does a CTC reader on Terra-box crops match or beat the generative reader on the same labels, quickly and
deterministically? It uses data and harness that already exist, and no new reads.

1. **Crops:**
   - Cut rectified crops from Terra's boxes on the 2000 map (all kinds), at full resolution, both directions.
   - Hold out the quarter and the answered blocks exactly as R1's `SheetData.held` does.
   - Add empty-target crops sampled from ink outside boxes, with the probability the spec measures for Terra's misses.
2. **Train three readers, one at a time on `mps` or CPU:**
   - A: docTR `crnn_mobilenet_v3_small`, fine-tuned from its weights, our vocabulary.
   - B: the same architecture from scratch.
   - C: B plus about 50k synthetic crops from the glyph renderer.
   - Also a 0°/180° crop classifier (a tiny CNN).
3. **Measure on the held-out quarter** (the same measures for every variant):
   - character error and word error on Terra strings, by kind;
   - for code boxes, top-1, top-3, mean −log p and a reliability table on the owner's answers, box-found subset, with
     bootstrap intervals;
   - beside the generative reader's c626e46 records and the 96.7% LLM yardstick;
   - p(other) and p(none) on street, block and non-text crops (explaining away).
4. **Determinism and speed:**
   - Run CPU inference twice and with 1 vs 8 threads; check the outputs are byte-identical.
   - Crops per second on CPU.
5. **Decision:** adopt the CTC likelihood if B or C is at least tied with the generative reader on box-found top-1 and
   better on −log p, with a tie judged by the intervals and the scoring set's error. Then do it by eye on the
   disagreements, per the spec's scoring rules. If only A wins, the gap is pretraining data, and the synthetic stream
   is the fix to try before shipping docTR's weights.

## Sources

**Map text research:**
- Weinman, research page and data: https://weinman.cs.grinnell.edu/research/maps.shtml
- Weinman 2017, Geographic and style models (ICDAR): https://weinman.cs.grinnell.edu/pubs/weinman17geographic.pdf
- Weinman et al. 2019, Deep neural networks for text detection and recognition in historical maps (ICDAR):
  https://weinman.cs.grinnell.edu/pubs/weinman19deep.pdf (code: https://github.com/weinman/MapTD,
  https://github.com/weinman/cnn_lstm_ctc_ocr, https://github.com/weinman/MapTextSynthesizer)
- Li et al. 2024, ICDAR 2024 MapText competition report: https://weinman.cs.grinnell.edu/pubs/li24icdar.pdf
- Lin et al. 2025, ICDAR 2025 MapText competition report: https://weinman.cs.grinnell.edu/~weinman/pubs/lin25icdar.pdf
- MapText Rumsey data (CC BY-NC-SA): https://zenodo.org/records/10656556; IGN data:
  https://zenodo.org/records/10987299
- mapKurator system paper: https://arxiv.org/html/2306.17059v2; repo:
  https://github.com/knowledge-computing/mapkurator-system; Spotter-v2 card:
  https://huggingface.co/ReadingMaps/Spotter-v2-english; spotter repo (LICENSE: CC BY-NC 2.0):
  https://github.com/knowledge-computing/mapkurator-spotter
- PALETTE (Hyper-local deformable transformers): https://arxiv.org/html/2506.15010v1; licence:
  https://github.com/kartta-foundation/mapkurator-palette-doc
- LIGHT linking: https://arxiv.org/html/2506.22589
- MapTextPipeline (DeepSolo/AdelaiDet-based): https://github.com/yyyyyxie/MapTextPipeline
- Yu, Luo, Chiang 2016, Recognizing text in historical maps using maps from multiple time periods (ICPR):
  https://spatial-computing.github.io/papers/yu16icpr.pdf
- Schlegel 2021, Automated extraction of labels from large-scale historical maps (AGILE):
  https://agile-giss.copernicus.org/articles/2/12/2021/agile-giss-2-12-2021.pdf
- Ares Oliveira et al. 2019, A deep learning approach to cadastral computing (DH):
  https://infoscience.epfl.ch/record/268282
- Urban Institute, ML for zoning data:
  https://urban-institute.medium.com/the-potential-of-machine-learning-for-compiling-standardized-zoning-data-c0f4697a9f0
- Chiang, Leyk, Knoblock 2014, A survey of digital map processing techniques (ACM CSUR)

**Recognition and decoding:**
- Graves et al. 2006, CTC (ICML)
- Hannun et al. 2014, prefix beam search with LM (arXiv 1408.2873)
- Scheidl, Fiel, Sablatnig 2018, Word beam search (ICFHR): https://github.com/githubharald/CTCWordBeamSearch
- Shi, Bai, Yao 2017, CRNN (TPAMI)
- Jaderberg et al. 2016, Reading text in the wild (IJCV)
- Bautista & Atienza 2022, PARSeq (ECCV): https://arxiv.org/abs/2207.06966
- Bourlard & Morgan 1994, Connectionist speech recognition (hybrid scaled likelihoods)
- McDermott, Sak, Variani 2019, density-ratio LM fusion (ASRU)
- Meng et al. 2021, internal LM estimation (SLT)
- Guo et al. 2017, On calibration of modern neural networks (ICML)
- Kopec & Chou 1994, Document image decoding (TPAMI)
- Berg-Kirkpatrick, Durrett, Klein 2013, Unsupervised transcription of historical documents (ACL):
  https://nlp.cs.berkeley.edu/pubs/BergKirkpatrick-Durrett-Klein_2013_OCR_paper.pdf
- Reul et al. 2017, Transfer learning for OCRopus model training on early printed books:
  https://arxiv.org/abs/1712.05586

**Libraries:**
- docTR: https://github.com/mindee/doctr; model docs: https://mindee.github.io/doctr/using_doctr/using_models.html;
  training README (`references/detection`, multi-class polygons, `--device mps`)
- OnnxTR: https://github.com/felixdittrich92/OnnxTR
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR; PP-OCRv5 docs:
  https://www.paddleocr.ai/v3.1.1/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html; PaddleOCR 3.0 report:
  https://arxiv.org/pdf/2507.05595; Paddle on macOS (CPU only):
  https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html
- RapidOCR: https://github.com/RapidAI/RapidOCR
- MMOCR: https://github.com/open-mmlab/mmocr
- EasyOCR: https://github.com/JaidedAI/EasyOCR (custom_model.md, `trainer/`)
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- keras-ocr: https://github.com/faustomorales/keras-ocr
- Tesseract: https://github.com/tesseract-ocr/tesseract; tesstrain: https://github.com/tesseract-ocr/tesstrain
- TrOCR: https://huggingface.co/microsoft/trocr-small-printed; https://github.com/microsoft/unilm
- PARSeq: https://github.com/baudm/parseq
- Kraken: https://github.com/mittagessen/kraken
- Calamari: https://github.com/Calamari-OCR/calamari
- TESTR: https://github.com/mlpc-ucsd/TESTR
- DeepSolo (AdelaiDet licence): https://github.com/ViTAE-Transformer/DeepSolo; AdelaiDet:
  https://github.com/aim-uofa/AdelaiDet
- Surya (weights OpenRAIL-M): https://github.com/datalab-to/surya

**Licences and determinism:**
- FSF licence list (Apache-2.0 → GPLv3 compatible): https://www.gnu.org/licenses/license-list.html
- CC BY-SA 4.0 one-way compatible with GPLv3:
  https://creativecommons.org/2015/10/08/cc-by-sa-4-0-now-one-way-compatible-with-gplv3/
- SynthText terms (non-commercial research):
  https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c
- PyTorch CTCLoss (CUDA nondeterminism note): https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
