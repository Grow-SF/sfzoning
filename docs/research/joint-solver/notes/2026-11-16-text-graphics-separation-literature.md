# Text/graphics separation: literature notes (2026-11-16)

Purpose: inform the design of synthetic training images (learned glyphs plus
drawn street/block lines) for the per-px TEXT detector. No code changed.

How to read this: "Source says" = read in the paper or its abstract/page.
"Inference" = my reading for our case. Items marked **[memory]** were NOT
re-opened in this session (cited from memory; check before quoting numbers).
Items marked **[abstract only]** were seen only through abstracts or
summaries. Search was done 2026-11-16; full PDFs were mostly not read
(the Li et al. numbers come from an HTML render of the paper).

## Annotated bibliography

### A. Text vs lines (classical and deep)

1. **Fletcher & Kasturi 1988**, "A robust algorithm for text string separation
   from mixed text/graphics images", IEEE TPAMI 10(6). **[memory]** Look it up
   by title (IEEE Xplore / Semantic Scholar).
   - Source says (as commonly summarised): connected components are
     classified by size/shape; strings are rebuilt by Hough transform; text
     touching graphics is not handled well.
   - Inference: the classical baseline. Its failure case (touching) is exactly
     ours. Size/shape cues on whole components are useless for us because
     lettering and lines are merged into one blob.

2. **Cao & Tan 2002**, "Text/Graphics Separation in Maps", GREC 2001, LNCS
   2390. https://unpaywall.org/10.1007%2F3-540-45868-9_14 (DOI
   10.1007/3-540-45868-9_14). **[abstract only]**
   - Source says: targets characters touching graphics in maps; assumes only a
     few characters of a string touch a line; uses the fact that character
     strokes are short while line strokes are long (and thinning).
   - Inference: "text strokes are short, line strokes are long" is a real
     cue a CNN can learn at the receptive-field level. The "few chars touch"
     assumption is weaker on our maps, where district borders run through
     whole codes.

3. **Moreno-Garcia, Johnston & Garkuwa 2020**, pixel-level CNN layer
   separation (text / connector / other) in P&ID engineering diagrams.
   https://rgu-repository.worktribe.com/OutputFile/972848 **[abstract only]**
   - Source says: per-pixel CNN classes; training patches came from a
     semi-automatic, heuristic layer tool.
   - Inference: closest deep precedent for per-px text vs line on line
     drawings. Their labels were noisy heuristics; ours would be exact.

4. **Jamieson, Moreno-Garcia & Elyan 2020**, deep text detection and
   recognition on P&IDs. https://rgu-repository.worktribe.com/OutputFile/972849
   **[abstract only]**
   - Source says: older text/graphics separation worked only moderately on
     these drawings; off-the-shelf deep detectors need adapting.
   - Inference: do not expect a generic scene-text detector to be a fix.

5. **Hoque et al. 2022**, segmenting technical drawings (U-Net, HR-Net, MedT,
   DETR compared; masks made from boxes with heuristics).
   https://www.cs.odu.edu/%7ejwu/downloads/pubs/hoque-2022-sdu/hoque-2022-sdu.pdf
   **[abstract only]**
   - Source says: box-only ground truth forced heuristic pixel masks.
   - Inference: a warning. Box-derived masks inside a box include the lines
     crossing it. Exact masks from rendering avoid that.

6. **Zhang et al. 2022**, component segmentation of mechanical drawings with
   graph convolutional networks. https://arxiv.org/abs/2212.00290
   - Source says: pixel sparsity hurts image-based methods; they vectorise and
     classify components (text / dimension / contour) as a graph.
   - Inference: a different route (vectorise first). Not suited to scans where
     touching merges components; noted only as the alternative.

### B. Synthetic data for text

7. **Gupta, Vedaldi & Zisserman 2016**, "Synthetic Data for Text Localisation
   in Natural Images" (SynthText). https://arxiv.org/abs/1604.06646
   **[memory]**
   - Source says (as I recall): renders words onto real photos, placed on
     regions chosen by depth/segmentation so text sits on plausible surfaces,
     then blended; detector pre-trained on it transfers to real sets.
   - Inference: supports "real background plus rendered glyphs" and "place text
     where it plausibly goes". Their backgrounds were text-free real images,
     which is what gave them exact labels.

8. **Li, Guan, Yu, Chiang & Knoblock 2021**, "Synthetic Map Generation to
   Provide Unlimited Training Data for Historical Map Text Detection",
   GeoAI 2021. https://arxiv.org/abs/2112.06104 (read via ar5iv render)
   - Source says: OSM tiles are restyled to Ordnance Survey look with CycleGAN;
     labels from OSM placed with QGIS (point labels around points, line labels
     along lines, 3 size tiers, 16 fonts); 13,892 tiles of 512x512 with 45,375
     text regions. PSENet F1 on 31 Rumsey maps: ICDAR only 57.3, synthetic
     only 60.6, both 64.3; recall rose most (45.0 -> 55.3 with both).
   - Source says: failures on unseen background colours, vertical text, wide
     spacing; 3 of 9 Rumsey series did not improve.
   - Inference: gain is real but modest, and the lines are whatever CycleGAN
     makes of OSM roads; they did not model lines as hard negatives and did not
     measure line-called-text. Uses boxes, not per-px labels.

9. **Weinman et al. 2019**, "Deep Neural Networks for Text Detection and
   Recognition in Historical Maps", ICDAR 2019 (MapTD, MapTextSynthesizer).
   https://github.com/weinman/MapTD and
   https://github.com/weinman/MapTextSynthesizer **[abstract/repo only]**
   - Source says: MapTD is an EAST-like detector for mixed text/graphics;
     trained on 33,868 annotated polygons over 30 maps with ten-fold CV; the
     synthesizer renders map-like grayscale text on the fly for training a
     RECOGNIZER, and the repo warns it is not meant for detector training.
   - Inference: the synthesizer's glyph/noise modelling is reusable for our
     glyph shapes; their own warning says nobody there trusted it for the
     text-vs-background problem.

10. **Kim et al. 2023, mapKurator** (demo) and **Lin & Chiang 2025, PALETTE
    with SynthMap+**. https://arxiv.org/abs/2306.17059 ;
    https://arxiv.org/abs/2506.15010 **[abstract only]**
    - Source says: mapKurator trains on synthetic maps in the Li et al. style,
      run on 60,000+ Rumsey maps; SynthMap+ is a newer synthetic set aimed at
      long and angled text.
    - Inference: synthetic-first training is the accepted route in this
      community, but all of it is box/polygon-level.

### C. Imperfect labels

11. **Zhang et al. 2020**, "Solving Missing-Annotation Object Detection with
    Background Recalibration Loss", ICASSP. https://arxiv.org/abs/2002.05274
    - Source says: when many true objects are unlabeled, they act as false
      background; a focal-loss-like term down-weights background samples that
      the model confidently calls object (using an IoU threshold); gains on
      PASCAL VOC / COCO with labels removed.
    - Inference: the key trick is to soften the loss only where the model
      itself is confident it sees an object, not in a big fixed disc.

12. **Niitani et al. 2019**, "Sampling Techniques for Large-Scale Object
    Detection from Sparsely Annotated Objects", CVPR. https://arxiv.org/abs/1811.10862
    **[memory]**
    - Source says (as I recall): down-weight/ignore negatives that overlap
      positives of related classes ("part-aware sampling"), per-image sampling
      of negatives.
    - Inference: negatives near a labelled thing are the suspect ones; weight
      by evidence, not area.

13. **Liu & Zhong 2020**, "Robust Scene Text Detection for Partially Annotated
    Training Data", Neurocomputing. Listing:
    https://www.bohrium.com/paper-details/robust-scene-text-detection-for-partially-annotated-training-data/817402567335084033-3832
    **[summary page only; full text not found]**
    - Source says (per summary): detector accuracy drops significantly when
      some text in training images is unannotated; a two-tier scheme makes
      pseudo-labels with a hybrid loss to refine text regions in tier two.
    - Inference: this is exactly our failure, and the fix is pseudo-labels
      from the detector itself, which conflicts with our "no LLM/own-solve
      labels as inference" rules only if used at inference; as a training aid
      it is allowed by the notes.

14. **Sun et al. 2019**, ICDAR 2019 RRC-LSVT (50k full + 400k weak labels).
    https://arxiv.org/abs/1909.07741
    - Source says: partial/weak labels are a recognised regime; "do not care"
      regions are excluded from scoring.
    - Inference: "do not care" regions are the standard device, but they are
      boxes around things nobody could judge, not discs around every label.

15. **Kiryo et al. 2017**, "Positive-Unlabeled Learning with Non-Negative Risk
    Estimator" (nnPU). https://arxiv.org/abs/1703.00593 **[memory]**
    - Source says (as I recall): with a known class prior, the negative risk is
      estimated from unlabeled data minus positives; clipping at zero stops
      deep nets overfitting.
    - Inference: needs the fraction of text among "unlabeled" px (the prior).
      We could measure it on a hand-checked sample. Mostly relevant if we keep
      real-map training at all.

16. **Zhang et al. 2019**, "Curved Text Detection with Semi- and Weakly-
    Supervised Learning". https://arxiv.org/abs/1908.09990
    - Source says: baseline from a small pixel-annotated set, then
      high-confidence pseudo-labels; for weak boxes, false positives are
      filtered by overlap with ground truth.
    - Inference: confirms confidence-gated self-training; the filter uses
      known true boxes, like our Terra boxes.

17. **Tian et al. 2017**, "WeText: Scene Text Detection under Weak
    Supervision". https://arxiv.org/abs/1710.04826 **[abstract only]**
    - Source says: detector learned from few char-level labels plus weak/
      unlabeled data via iterative re-labelling.
    - Inference: same family; iterate, keep only high-confidence re-labels.

### D. Pixel losses and boundaries

18. **Ronneberger et al. 2015**, U-Net. https://arxiv.org/abs/1505.04597
    **[memory]**
    - Source says (as I recall): a precomputed weight map gives extra loss on
      thin gaps between touching objects, plus class-balance weights.
    - Inference: the same device can weight the thin text-on-line overlap px.

19. **Lin et al. 2017**, Focal Loss. https://arxiv.org/abs/1708.02002 ;
    **Shrivastava et al. 2016**, OHEM. https://arxiv.org/abs/1604.03540
    **[memory]**
    - Source says (as I recall): focal loss down-weights easy negatives; OHEM
      keeps the highest-loss examples.
    - Inference: hard-negative mining only helps if the "hard negatives" are
      truly negative. With missed Terra text, mining picks the unlabeled text
      first. Mine on synthetic data, where negatives are exact.

20. **Baek et al. 2019**, CRAFT. https://arxiv.org/abs/1904.01941
    **[abstract only]**
    - Source says: per-character region/affinity maps; real word-level data is
      converted to character-level pseudo-labels by the model itself, with
      confidence-based filtering of the pseudo-labels.
    - Inference: per-px (not box) targets are normal for text; confidence-based
      pseudo-labels are an accepted way to get them from boxes.

## Answers to the six questions (short)

1. Text/graphics separation. Classical methods use component size, string
   alignment and stroke length (Fletcher 1988, Cao & Tan 2002); they handle
   touching badly or by strong assumptions. Deep per-px nets (Moreno-Garcia
   2020) work on engineering drawings but I found no paper that measures the
   overlap case directly. The usable cue is local stroke geometry
   (short, curved, dense strokes vs long straight ones), which a U-Net's
   receptive field can learn if it sees enough touching examples.
2. Synthetic data. SynthText renders on real text-free photos; Li et al. 2021
   restyles OSM and places labels by QGIS; Weinman renders glyph noise for
   recognisers. None drew line work as an explicit hard negative. Gain: Li
   F1 57.3 -> 64.3 (PSENet, with real out-of-domain data). Domain gap: unseen
   colours, vertical text, a third of target series not improved.
3. Overlap labels. I found no paper that labels the overlap px as a separate
   class in maps. The closest practice: U-Net weight maps, class-balanced
   losses. With synthetic rendering you know every overlap px exactly, so a
   three-class or "text over line" mask is cheap to make (inference).
4. Incomplete positives. Remedies in the literature: do-not-care regions
   (RRC-LSVT), confidence-gated background down-weighting (Background
   Recalibration Loss), part-aware negative sampling (Niitani), nnPU with a
   class prior, and detector-made pseudo-labels (Liu & Zhong; Zhang 2019;
   CRAFT). None masks a fixed disc around each label.
5. Metrics. Detection papers use box precision/recall/F1 (Wolf's IoU
   protocol, as in Li et al.; ICDAR do-not-care rules). Per-px metrics for
   "line called text" are not standard; see Recommendations.
6. Real backgrounds vs drawn lines. SynthText-style real backgrounds beat
   pure graphics for photos; for maps, Li et al. used generated (GAN-restyled)
   backgrounds. I found no head-to-head comparison.

## Recommendations for our case (ranked)

1. **Train the first fix on synthetic images with exact per-px labels, and give
   the line work enough variety to matter.** Render glyphs, draw lines over,
   under and through them, and keep three masks (text px, line px, overlap
   px). Sources: Li 2021 (synthetic helps, recall most), SynthText 2016 (exact
   labels from rendering), Weinman 2019 (their own warning that glyph-only
   rendering was not fit for detector training). Inference: our gap is the
   line class, which no prior synthetic set models.
2. **Make the line drawing look like the map, from measured line statistics.**
   Take widths, spacing, dash patterns, hatch pitch, border weight and how
   often lines touch or cross lettering from the real scans; draw heavy borders
   through codes, not just near them. Sources: Cao & Tan 2002 (stroke length
   is the cue), Li 2021 (failures when the style is unseen). Inference: the
   gap is closed by matching the line statistics, not by more of the same.
3. **Mix in real text-free crops as extra negatives, and as backgrounds under
   the rendered glyphs, in a ratio we choose by a held-out check.** Sources:
   SynthText (real backgrounds); Li 2021 (mixing synthetic with real
   gave the best F1). Inference: drawn lines teach the shape; real crops teach
   the scan noise (1-bit grain, binding shadow, tint). Needs a crop selector
   that is sure the crop has no text (see 5).
4. **Weight the overlap px and the thin gaps in the loss; keep hard-negative
   mining on synthetic data only.** Sources: U-Net weight maps; focal loss;
   OHEM. Inference: on real data mining would pick up the text Terra missed
   first.
5. **Replace the big disc on real data by a small ignore band plus a
   confidence-gated rule.** Keep "unknown" only where the current model is
   confident of text and no Terra box covers it (the Background
   Recalibration Loss idea), and size any disc from measured Terra box
   misses. Sources: Zhang 2020; Niitani 2019; RRC-LSVT do-not-care.
   Inference: the disc was the cause; once the synthetic data teaches
   "line px are not text", the real-data term can be gentle.
6. **Then self-train: let the model make pseudo-labels on the real maps, keep
   only high-confidence text far from lines, and add them as positives.**
   Sources: Liu & Zhong 2020; Zhang 2019; CRAFT; WeText. Inference: use
   only as training labels; the notes say no model output enters inference.
7. **If real-map training stays, check nnPU as a second option.** Needs the
   share of text px among unlabeled px, estimated on a hand-checked sample.
   Source: Kiryo 2017. Inference: lowest rank because the prior is hard to
   measure and the synthetic route removes most of the need.
8. **Measure line-called-text directly (see "Measuring" below) next to the
   usual box F1, and gate every change by both.** Inference; no paper gives
   a standard for this.

### Measuring (inference, not from a source)

- On synthetic: per-px precision/recall for text, and the **line false-text
  rate** = line px (outside the overlap) with text chance above the
  threshold / all line px. Report also on overlap px alone.
- On real: a small hand-labelled set of crops with line-only and text-over-line
  zones; report the share of street/district line length called text.
- Keep the box-level Wolf/IoU F1 (Li 2021) for comparison with the literature.
  Per-px IoU or Dice for text, plus a boundary-band score (px within 2 of a
  line) so the overlap is not hidden by the large background count.

## Pitfalls the literature reports

- Unlabeled text taught as background lowers accuracy sharply (Liu & Zhong
  2020 summary; Zhang 2020).
- Synthetic styles do not cover unseen colours, orientations, spacing; a third
  of Rumsey series gained nothing (Li 2021).
- Glyph-only rendering is not good enough for detector training (Weinman's
  repo caveat).
- Pseudo-labels reinforce the model's own errors if the gate is loose; keep a
  confidence gate and a held-out check (CRAFT, Zhang 2019).
- Hard-negative mining on partly unlabeled data mines the missed positives
  (inference from OHEM plus the missing-annotation papers).
- Box-derived masks include the crossing lines (Hoque 2022 had to patch it
  with heuristics).
- Heuristic layer labels are noisy (Moreno-Garcia 2020).

## What the literature does NOT settle

- Whether drawn lines or real text-free crops transfer better for maps: no
  head-to-head found.
- Whether a three-class (text / line / overlap) target beats a binary one on
  maps: no paper tested it.
- How big an ignore region should be: no rule found; the papers gate by model
  confidence or by do-not-care boxes.
- A standard per-px metric for line ink wrongly called text.
- How much synthetic data is enough and in what ratio with real: Li 2021 used
  one mix; no curve.
- Whether classical stroke-length cues survive 1-bit scans with heavy lines
  touching every letter; Cao & Tan assumed few touches.
- Several items above were not re-read this session (marked **[memory]** or
  **[abstract only]**); confirm before quoting.
