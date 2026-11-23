# Finding blocks, streets and lines on scanned maps: what the literature does (2026-11-23)

A review made for `docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md`. A reader fetched and read
the sources; nothing here is from memory alone. Read status: **[F]** full text read; **[A]** abstract or project
page only; **[N]** could not be opened, not verified.

## 1. Blocks from scanned city maps (the ICDAR 2021 competition and the EPITA work)

- **Chazalon et al., ICDAR 2021 competition on historical map segmentation, task 1** [F]
  https://arxiv.org/abs/2105.13265
  - Paris atlas sheets, about 8,000 px square: 1 to train, 1 to validate, 3 to test. Labels are block masks. The
    measure is COCO PQ (shapes matched at IoU over 0.5).
  - Winner (L3IRIS, PQ 74.1): a DenseNet-121 labels the ground that is NOT block (the streets) on 480 px tiles, with
    semi-supervised use of the unlabelled sheet; morphology then opens thin bridges and fills holes. Second (CMM,
    62.6): U-Net edges plus morphology. A U-Net with Dice loss scored 6.4 and distance-transform regression plus
    watershed 4.2; both lost to thin links between shapes.
  - For us: labelling street ground beat edge-finding; bridges between shapes, not pixel accuracy, decide the score.
- **Chen et al., ICDAR 2021, "Vectorization of historical maps using deep edge filtering and closed shape
  extraction"** [F] https://www.lre.epita.fr/dload/papers/chen.21.icdar.pdf (precursor [F]
  https://arxiv.org/abs/2101.02144)
  - A network gives an edge-probability map; a watershed on it gives closed shapes with 1 px borders. Two settings
    (minimum dynamic, minimum area), grid-searched on a validation sheet.
  - Target: hand-drawn polygon borders drawn 3 px wide, the printed width; text, railways and other lines are plain
    negatives. Loss: class-rebalanced cross-entropy. About 200 hours of labelling.
  - Test PQ: cut at a level plus connected pieces 16.2; watershed 38.4. A U-Net scored 56.6 on validation but 18.3
    on test: crisp edges with crisp gaps.
  - For us: the closing step bought more than the choice of network.
- **Chen et al., PLOS ONE 2024, benchmark** [F] https://doi.org/10.1371/journal.pone.0298217
  - Same pipeline. Training fixes took a U-Net from 8.1 to 41.2 test PQ; plus watershed 46.7; choosing the epoch and
    the watershed settings together on validation PQ 47.1; plus contrast and thin-plate-spline augmentation 51.1.
  - Losses added to cross-entropy, test PQ: none 47.1, BALoss 45.6, TopoLoss 36.9, Mosinska 36.0. BALoss led on
    validation (63.1) but did not carry to the other sheet. Transformers 34.7-36.6; a learned deep watershed 28.5.
    clDice was not tested.
  - For us: recipe, closure, selection on a shape measure and augmentation mattered most; topology losses did not.

## 2. Losses that reward unbroken lines

- **clDice** (Shit et al. 2021) [F] https://arxiv.org/abs/2003.07311: soft skeletons by repeated min/max pooling;
  rewards each skeleton lying inside the other mask. On roads it cut breaks (Betti-0 error 0.491 to 0.375). Others
  measured +88% time and +52% memory. This is the loss on the branch (`hatchnet/topo.py`).
- **TopoLoss** (Hu et al. 2019) [F] https://arxiv.org/abs/1906.05404: matches persistence diagrams; slow (65 px
  patches).
- **Mosinska et al. 2018** [F] https://arxiv.org/abs/1712.02190: feature difference plus iterative refinement.
- **BALoss** (Ngoc et al. 2021) [F] https://www.lrde.epita.fr/dload/papers/movn.21.bmvc.pdf: one seed per true
  region; a minimum-barrier cut on the predicted map finds the leak pixels.
- **Homotopy warping** (Hu 2022) [F] https://arxiv.org/abs/2112.07812; **skeleton recall** (Kirchhoff et al. 2024)
  [F] https://arxiv.org/abs/2404.03010 (+8% time, small gains). PathLoss [A; preprint withdrawn].
- For us: these reduce breaks on aerial roads; the only test on scanned maps (Chen 2024) shows no gain on a sheet
  not trained on. Optional, not the main lever.

## 3. Several kinds of map feature in one model

- **Petitpierre et al. 2021** [F] https://ceur-ws.org/Vol-2989/long_paper27.pdf: region classes (roads, blocks,
  water, non-built, frame) from hand-labelled patches (7 weeks). Good with three classes, clearly worse with five;
  binarising the input cost 7.2%.
- **Petitpierre 2026, Semap** [F] https://arxiv.org/abs/2603.05037: five classes plus an auxiliary "boundary" class;
  1,439 patches, about 400 hours. The boundary class is the weakest (40.7) and absorbs thin roads and water; heavy
  hatching hides thin features. Pretraining on made-up maps drawn from modern vectors (random hatching, line widths,
  text, grids) added 5.1.
- **Can et al. 2021** [F] https://eprints.gla.ac.uk/364716/1/364716.pdf: seven road line types, one model each; F
  about 0.5 for common types, 0.10 where the line style is hard to tell from other ink.
- **Muhlematter et al. 2024** [F] https://arxiv.org/abs/2410.02250: find roads first, then classify the symbol on
  the known geometry; grid lines mistaken for one road class.
- Oliveira et al. (Venice cadastre) [N]; Lin et al. 2023 (Sanborn, relies on colour) [F]; mapKurator (text only) [F].
- For us: **no source tests giving a confusable line its own class against leaving it unnamed.** EPITA treats such
  lines as negatives; Semap's shared boundary class is its weakest.

## 4. Labels that are shifted or out of date

- **Mnih & Hinton 2012** [F] https://www.cs.toronto.edu/~vmnih/docs/noisy_maps.pdf: the loss models missing labels
  and a hidden shift of each label patch. This is the noise model on the branch.
- **Girard et al. 2019** [F] https://arxiv.org/abs/1903.06529: a network predicts the label's displacement; re-align
  the labels, retrain. Works because the noise is zero-mean and independent of the image.
- **Duan et al. 2020, 2021** [F] https://doi.org/10.1080/13658816.2019.1698742 , https://arxiv.org/abs/2112.05794:
  move modern vector points onto nearby ink; delete points with no ink near them. Label precision 37-65% to
  95-99.8%. Fails by snapping to a neighbouring line.
- **Wu et al. 2023** [F] https://doi.org/10.3929/ethz-b-000599929: a widened-label Dice plus a learned mask that
  withholds supervision where the object changed.
- **Lopez-Rauhut et al. 2025** [F] https://arxiv.org/abs/2505.24824: training directly on modern labels drew roads
  and buildings the old map lacks (26-39 against 61-76 from 470 hand-labelled tiles).
- Held-out ("cross-fitted") predictions against label noise: used in confident learning [F]
  https://arxiv.org/abs/1911.00068 for classification. **No use found on maps or remote-sensing segmentation.**

## 5. Separating regions without a cut level

- **Deep watershed** (Bai & Urtasun 2017) [F]: learned energy cut at a fixed level; 28.5 against 47.1 on maps.
- **StarDist** [F], **Cellpose** [F]: per-pixel shape or flow targets made from instance masks.
- **Mutex watershed** (Wolf et al.) [F] https://arxiv.org/abs/1904.12654: short-range "join" and long-range "must
  not join" evidence, no seeds and no level; from nerve-cell imaging.
- **MALIS** [F] https://arxiv.org/abs/1709.02974: trains exactly the weakest pixel on the best path between two
  regions.
- Unproven on maps.

## 6. Made-up training sheets

- Jiao et al. 2022 [A]; Semap (above): +5.1; Arzoumanidis et al. 2025 [F] https://arxiv.org/abs/2511.15875: trained
  only on made-up sheets, 0.87 accuracy on real ones; learned style transfer moved geometry off its labels.
- Text: Li et al. 2021 [F]; SynthMap+ [F].

## What the best approaches do that a plain network on shifted labels does not

1. Close shapes by construction: a watershed on the probability map.
2. Choose the epoch and settings on a shape measure from a different sheet.
3. Move labels onto the ink first; drop labels with no ink under them.
4. Put shift, omission and change into the loss.
5. Draw the target at the printed line width, with confusable ink as negatives.
6. Fix the training recipe, augment, pretrain on drawn maps.
7. Label street ground, not only edges.
8. Train the leak pixel or add long-range evidence: unproven on maps.

## What the literature does not solve for us

- Labels both shifted and wrong for the era, on 1-bit hatched city maps. The one study that reproduces our failure
  fixes it only with hand labels.
- A thick border down a street or through a block: no source tests it as its own class.
- Held-out label cleaning for segmentation where the noise comes in whole stretches.
- Snapping labels to ink can lock onto the wrong parallel line (block edge against border).
- Roads drawn as two open lines are not closed shapes.
- The best cross-sheet scores are 47-51 by their measure: other sheets are where these methods usually fail.
