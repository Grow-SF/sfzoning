# One solver for every drawing style: literature notes (2026-11-18)

The question (owner, 2026-11-18): "can we train a general solver on the set of these three and have it infer the
parameters of line width, hatching, etc based on the statistics observed in the scan itself? ... start with the 2000
solver and then add the other two classes to restart training (is this fine-tuning? another theory?)"

This note collects published work on that question and says what it predicts for us. It changes no code. $0.

How to read it. Every work below was looked up on 2026-11-18 and exists at the link given. The tag after each says how
much of it was read: **[full text]** = the relevant sections were read; **[abstract]** = only the abstract, or a
summary of it; **[secondary]** = what another paper says about it. Lines starting "Found:" are what the source
says. Lines starting "For us:" are my reading for our maps.

## Our setup, in one paragraph

The joint solver gives every modern lot a zoning code. It weighs two kinds of evidence: the printed labels (read once
by an LLM, saved), and "walls". A wall is the odds that a district border runs between two neighbouring lots, from a
ridge logistic regression on hand-built ink features (line ink along the shared border, stroke widths, street
crossings, context). The 2000 wall model has 463 features. It was fitted on about 430,000 lot pairs against the April
2000 zoning table. No other year has an answer key. There are three kinds of drawing:

- **Class A, modern line-and-label drawings (1980-2000).** The 2000 walls carry over, and wrong lots fall 2-8x
  against the old pipeline.
- **Class B, the 1960-1980 editions.** Thinner strokes and another drafting style. Scaling stroke widths by one
  measured ratio raised the share of the map's own labels the solve keeps from 41% to about 81%; the old LLM-region
  map keeps 93%. On the 1996-2000 scan, the same drawing as 2000, a 5% width scale *raised* the table's wrong lots
  from 15,974 to 20,169 (`transfer.width_scale`), so it is switched off for scans of the 2000 drawing.
- **Class C, 1921-1948.** Districts are drawn as hatching patterns keyed by a legend, not as lines plus labels.

Coming labels: a few hundred lot-level answers from the owner on one class B map (1960) and one class C map (1921).

## Short answer

1. **The name for "start from 2000, add the other two classes, train again" depends on what is trained on.**
   - Continuing from the 2000 weights on the new answers alone is *fine-tuning*.
   - Fitting once on all three classes pooled is *joint* (multi-domain) training.
   - The version that suits us is a *hierarchical* model, also called *partial pooling* or *random effects*. Every
     class shares one set of weights. Each class may move a few of them, and the data decide how far.
2. **Plain fine-tuning does nothing useful for our model.** Our wall fit is convex: ridge logistic regression, run by
   Newton's method to convergence. A convex fit ends at the same answer whatever weights it starts from. So "start
   from 2000" keeps nothing from 2000 unless the 2000 knowledge goes in as data (the 2000 pairs stay in the fit) or as
   a prior (a penalty pulling toward the 2000 weights). This is L2-SP in deep learning and MAP adaptation for logistic
   models. Daumé's feature-copying trick does it inside the ridge fit we already have.
3. **"Infer the parameters from the scan's own statistics" splits in two.**
   - Measuring the nuisance parameters (pen width, ink contrast, hatch spacing and angle) needs no labels. Each
     feature can then be expressed relative to them. This is classic scale normalisation and per-domain normalisation,
     and the width ratio already does a crude version of it.
   - Learning how the evidence should be *weighed* as those parameters change needs labels from many styles. We have
     answers on only three drawings, which is three data points for that function. The many styles can only come
     from re-drawing known answers synthetically.
4. **Normalisation is enough only when the style change is a pure re-scaling of the same evidence.** It cannot fix a
   change in what the evidence means. It cannot turn hatching into lines. It can hurt when the base rates differ, and
   the 1921 districts are larger and fewer than 2000's. For class B it should close most of the gap, and the
   41% → 81% jump agrees. For class C it cannot help much.
5. **Hatching (class C) needs its own evidence.** That means texture matched against the legend's swatches, and
   walls from a change of texture across a border. The best published legend-guided systems, from the DARPA mineral
   maps work, are built around colour. A 2026 thesis from that group still lists hatching patterns as future work. A
   one-colour hatched map is not a solved problem.
6. **A few hundred answers give few border examples.** The answers should come in contiguous patches, so that both
   lots of many pairs are known. A random, area-blocked part should be kept back for scoring only. Expect roughly
   ±3-4 points of accuracy at 95% from 300 scored lots. Compare methods paired on the same lots.

## The owner's question in standard terms

| Owner's words | Standard name | What it means for our ridge logistic walls |
|---|---|---|
| start with 2000, train on the new classes | fine-tuning (sequential transfer) | Refit on 1960/1921 answers only, starting at the 2000 weights. Convex, so the start is forgotten; with a few hundred answers, 463 weights overfit. |
| restart training on all three | joint / multi-domain training | Pool 2000 pairs with answered pairs. 2000 dominates (≈430k vs ≈1k pairs) unless weighted. |
| keep 2000 but let the new classes move it a little | L2-SP / MAP adaptation | Ridge penalty centred on the 2000 weights instead of zero. The width of the penalty is the one tuning knob. |
| one shared model, each class its own small deviations, learned how big | hierarchical Bayes / partial pooling / random effects | Shared weights plus a per-class offset on a few chosen parameters; offset size picked on held-out answers. Contains the three rows above as limiting cases. |
| infer line width / hatching from the scan | per-domain normalisation; conditioning (FiLM); hypernetworks | Measure scan statistics, then either divide features by them (no labels) or let weights depend on them (needs many labelled styles). |
| don't lose what 2000 learned | avoiding catastrophic forgetting | Automatic in joint training, because the 2000 pairs stay in the loss; check by re-scoring 2000 quarter by quarter. |

## 1. Historical map processing across map styles

1. **Petitpierre, Kaplan & di Lenardo 2021**, "Generic Semantic Segmentation of Historical Maps", CHR 2021,
   CEUR-WS 2989. https://ceur-ws.org/Vol-2989/long_paper27.pdf **[full text: abstract, discussion, conclusion]**
   - Found: a CNN trained on city maps from many cultures and styles held up well across "figurative diversity". Most
     of its skill came from abstract cues (shape, topology, which class sits inside which) rather than from colour or
     texture. Pre-training on the diverse world corpus helped the rarest class most: water rose 18% on Paris and 36%
     on the world set. Transfer was poor where a class meant different things in the two corpora (their "non-built").
   - For us: our solver already leans on the cues that carried over there: lot geometry, street crossings, and the
     joint energy. The ink features are the style-bound part. A class whose *meaning* differs between eras (a 1921
     district class against a 2000 code) will not transfer by weights alone.

2. **Petitpierre 2026**, "Generalizable Multiscale Segmentation of Heterogeneous Map Collections", arXiv 2603.05037.
   https://arxiv.org/abs/2603.05037 **[full text, first ~85%]**
   - Found: the Semap benchmark (1,439 annotated patches). Pre-training on *procedurally rendered* maps with random
     fills (plain colour, dots, hatchings), random line widths and scan noise raised test mIoU from 69.1% to 74.2%.
     The rendered maps were modern geodata drawn in varied historical styles. Metadata (institution, country, scale,
     year) explained almost none of the remaining error (R² = 0.043). The author calls synthesis "a controlled
     approximation of the real domain".
   - For us: the closest published precedent for re-drawing known answers in each era's style, hatching included
     (see §6). The gain was real but moderate, and mostly in recall.

3. **Chazalon et al. 2021**, "ICDAR 2021 Competition on Historical Map Segmentation" (MapSeg), arXiv 2105.13265.
   https://arxiv.org/abs/2105.13265 **[abstract]**
   - Found: nine Paris atlases from 1894-1937, one publisher. The winners: a weakly supervised DenseNet for building
     blocks, and U-Net plus binarisation for map content and graticules.
   - For us: the nearest benchmark to "one city, one series, decades of editions". Even there, every task was
     trained on the series itself; nothing was learned on one style and used unchanged on another.

4. **Chiang, Leyk & Knoblock 2014**, "A Survey of Digital Map Processing Techniques", ACM Computing Surveys 47(1).
   https://spatial-computing.github.io/papers/chiang14survey.pdf **[abstract]**; and **Uhl, Leyk, Chiang, Duan &
   Knoblock 2020**, "Automated extraction of human settlement patterns from historical topographic map series using
   weakly supervised CNNs", IEEE Access 8. https://doi.org/10.1109/ACCESS.2019.2963213 **[abstract]**
   - Found: the survey is the standard map of the field. Uhl et al. trained on USGS maps from 1893-1954 with *no hand
     labels*: modern geodata chose the training samples. Recall reached up to 0.96, F up to 0.79.
   - For us: modern data used as weak labels on old maps is common practice. We already do it in the strong form,
     with modern lot lines as the geometry.

5. **Wu, Schindler, Heitzler & Hurni 2023**, "Domain adaptation in segmenting historical maps: A weakly supervised
   approach through spatial co-occurrence", ISPRS J. Photogramm. 197. https://doi.org/10.1016/j.isprsjprs.2023.01.021
   **[abstract]**; and **Yuan, Thiemann & Sester 2025**, "Semantic Segmentation for Sequential Historical Maps by
   Learning from Only One Map", arXiv 2501.01845. https://arxiv.org/abs/2501.01845 **[abstract]**
   - Found: both move a model from one map series or year to neighbouring ones. Neither uses new hand labels. Both
     take what probably did not change at the same place as weak labels (Wu: objects that co-occur across series;
     Yuan: predictions on the labelled map, passed to the next year). Yuan reports 77.3% mIoU, about 20 points above
     their baselines.
   - For us: this is the main published route between neighbouring years. It conflicts with two owner rules: never
     learn from our own solve, and no priors from earlier solves. It is listed under "not recommended" below.

6. **The DARPA/USGS AI for Critical Mineral Assessment competition (2022-23).** **Goldman et al. 2025**, "Extracting
   data from maps: Lessons learned from the AI for Critical Mineral Assessment competition", Applied Computing and
   Geosciences 27. https://doi.org/10.1016/j.acags.2025.100274 **[abstract]**
   - Found: legend-prompted extraction from geologic maps of very varied style reached median F1 0.77 for polygons,
     0.56 for lines and 0.35 for points.
   - For us: about 1 in 4 polygon pixels wrong at the median, even with the legend in hand. Cross-style extraction is
     hard even for funded teams with labelled training maps.

7. **Kim et al. 2023**, "The mapKurator System: A Complete Pipeline for Extracting and Linking Text from Historical
   Maps", arXiv 2306.17059. https://arxiv.org/abs/2306.17059 **[abstract]**
   - Found: one text pipeline ran over 60,000 David Rumsey maps. Recognised words are corrected against a vocabulary.
   - For us: it bears on labels (classes A and B), not on walls. The vocabulary step matches our era code lists.
     Text was covered in the 11-17 notes.

**What area 1 predicts for us.** Abstract cues carry across styles; the drawing-specific ink cues do not. Every
system that crossed styles well did it by one of three routes: training on many styles (real or rendered), giving
the model the legend, or labelling the target. No one carried a model trained on one style unchanged to a very
different one. Expect class B to need a little target data, and class C new evidence.

## 2. Label-free per-scan normalisation

1. **Li et al. 2016**, "Revisiting Batch Normalization for Practical Domain Adaptation" (AdaBN), arXiv 1603.04779.
   https://arxiv.org/abs/1603.04779 **[abstract]**; and **Schneider et al. 2020**, "Improving robustness against
   common corruptions by covariate shift adaptation", NeurIPS 2020. https://arxiv.org/abs/2006.16971 **[abstract]**
   - Found: recomputing each layer's feature means and spreads on target images, with no labels, adapts a network to
     a new domain. For corruption-type shifts (blur, noise, contrast), 32 target images were enough to beat the
     previous best.
   - For us: the per-scan width ratio is a one-feature AdaBN. The literature says this kind of fix works when the
     shift is "the same evidence, rescaled": a thinner pen, lighter ink, another scan threshold.

2. **Pan et al. 2018**, "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net", ECCV 2018.
   https://arxiv.org/abs/1807.09441 **[abstract]**
   - Found: normalising each image by its *own* statistics (instance normalisation) removes appearance (colour,
     style). Normalising over the training batch keeps content. The best networks use both.
   - For us: "own statistics" here means per scan, not per page or per sheet. A feature normalised by the whole
     page's mean width mixes border lines, lot lines, letters and hatching. The 1996-2000 result, where a 5% scale
     cost 4,195 lots, says that mix is fragile. Normalise against a reference from the *same kind of ink*, such as
     each scan's lot-line widths.

3. **Sun, Feng & Saenko 2016**, "Return of Frustratingly Easy Domain Adaptation" (CORAL), AAAI 2016.
   https://arxiv.org/abs/1511.05547 **[abstract]**; and **Wang et al. 2021**, "Tent: Fully Test-time Adaptation by
   Entropy Minimization", ICLR 2021. https://arxiv.org/abs/2006.10726 **[abstract]**
   - Found: CORAL matches the target features' means and covariances to the source with no labels. Tent nudges the
     normalisation parameters at test time so the model's own predictions become more confident.
   - For us: CORAL is the multi-feature version of what the 11-18 go-ahead calls per-feature distribution matching.
     Tent is self-training on our own output, which the owner has ruled out.

4. **Saerens, Latinne & Decaestecker 2002**, "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A
   Simple Procedure", Neural Computation 14(1). https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf
   **[abstract]**; and **Lipton, Wang & Smola 2018**, "Detecting and Correcting for Label Shift with Black Box
   Predictors", ICML 2018. https://arxiv.org/abs/1802.03916 **[abstract]**
   - Found: when only the class base rates change, they can be re-estimated on unlabelled target data. Saerens uses
     an EM loop that maximises the target likelihood; Lipton uses the confusion matrix. The classifier's odds are
     then shifted, with no refit.
   - For us: the share of neighbouring-lot pairs with a border differs by era, and 1921 has far fewer, larger
     districts. That is label shift. It moves only the wall model's intercept, and it can be estimated per scan with
     no labels.

5. **Zhao, Tachet des Combes, Zhang & Gordon 2019**, "On Learning Invariant Representations for Domain Adaptation",
   ICML 2019. https://proceedings.mlr.press/v97/zhao19a.html **[abstract]**; and **Ben-David et al. 2010**, "A
   theory of learning from different domains", Machine Learning 79. https://doi.org/10.1007/s10994-009-5152-4
   **[abstract]**
   - Found: Zhao et al. prove that making source and target features look alike cannot fix a change in how features
     relate to the answer. When base rates differ, it can make things worse. Ben-David et al. bound the target error
     by the source error plus a domain-distance term. They also show how to weight a little labelled target data
     against plentiful source data.
   - For us: this is the formal "when is normalisation enough".

**When normalisation is enough.** All three conditions must hold:

- the same evidence is printed in the new era, only rescaled (width, darkness);
- the statistic is measured on that same evidence;
- base rates are handled separately (item 4).

For class B the first holds roughly, which matches 41% → 81%. Whatever gap remains toward 93% is either
mis-normalisation (test: normalise against lot-line ink only and see whether own-labels-kept rises) or a real change
in drafting meaning (test: the 1960 answers). For class C the first condition fails, because the evidence is a
texture, not a line.

## 3. Few-shot transfer with a prior toward the source model

1. **Li, Grandvalet & Davoine 2018**, "Explicit Inductive Bias for Transfer Learning with Convolutional Networks"
   (L2-SP), ICML 2018. https://arxiv.org/abs/1802.01483 **[abstract]**
   - Found: when fine-tuning, an explicit L2 penalty toward the pre-trained weights beats relying on the starting
     point plus early stopping. The authors recommend it as the baseline for transfer.
   - For us: in our ridge fit this is one change: the penalty pulls toward the 2000 weights instead of toward zero.

2. **Chelba & Acero 2004/2006**, "Adaptation of Maximum Entropy Capitalizer: Little Data Can Help a Lot", EMNLP 2004
   (https://aclanthology.org/W04-3237/), journal version Computer Speech & Language 20(4), 2006.
   **[full text, EMNLP version]**
   - Found: this is the logistic-regression form of L2-SP, published 14 years earlier. A Gaussian prior is centred on
     the background model's weights; in their words, "the effect of the prior is to keep the model parameters close
     to the background ones". The background was a model trained on 20 million words; the new domain had 25-70
     thousand words. The adapted model was 20-25% better (relative) than the background alone. The prior's width was
     chosen on held-out data.
   - For us: the closest published match to our setup: a log-linear model, a big source and a small target.

3. **Daumé III 2007**, "Frustratingly Easy Domain Adaptation", ACL 2007. https://aclanthology.org/P07-1033/
   **[abstract]**; and **Finkel & Manning 2009**, "Hierarchical Bayesian Domain Adaptation", NAACL 2009.
   https://aclanthology.org/N09-1068/ **[full text, searched]**
   - Found: Daumé copies every feature three times: shared, source-only and target-only. An ordinary regularised fit
     on the copies then shares what is common and keeps what differs. Finkel & Manning show this is *equivalent* to
     a hierarchical Bayesian model: each domain's weights are drawn around shared top-level weights. Each domain's
     variance is a knob. A very large variance is the same as separate models; a variance near zero ties the weights
     together. Training took about as long as one pooled model.
   - For us: the most decision-relevant finding here. Our `walls.fit_quarter` is a ridge logistic regression already,
     so partial pooling needs no new solver, only extra feature columns. Copy the chosen columns per class; the ridge
     strength on the class copies is how far a class may differ from the shared weights.

4. **Gelman 2006**, "Multilevel (Hierarchical) Modeling: What It Can and Cannot Do", Technometrics 48(3).
   https://doi.org/10.1198/004017005000000661 **[abstract]**
   - Found: giving the regression coefficients their own model, fitted from data, is "almost always an improvement"
     over pooling everything or fitting each group alone. The gain is largest for prediction.
   - For us: with only three classes (A, B, C), the between-class spread is poorly determined by the data. Choose it
     by held-out answers, and keep the per-class offsets to a few parameters.

5. **Bilen & Vedaldi 2017**, "Universal representations: The missing link between faces, text, planktons, and cat
   breeds", arXiv 1701.07275. https://arxiv.org/abs/1701.07275 **[abstract]**; with **Rebuffi, Bilen & Vedaldi 2017**,
   "Learning multiple visual domains with residual adapters", NeurIPS 2017 (https://arxiv.org/abs/1705.08045)
   **[abstract]**, and **Chang et al. 2019**, "Domain-Specific Batch Normalization for Unsupervised Domain
   Adaptation", CVPR 2019 (https://arxiv.org/abs/1906.03950) **[abstract]**
   - Found: one network handles very different domains almost as well as specialist networks, *if* each domain gets
     its own rescaling of the internal activations. Everything else stays shared.
   - For us: the multi-domain-learning form of "shared walls plus a per-class scale on each evidence block". It is
     the deep-learning cousin of item 3.

6. **Kirkpatrick et al. 2017**, "Overcoming catastrophic forgetting in neural networks" (EWC), PNAS 114(13).
   https://doi.org/10.1073/pnas.1611835114 **[abstract]**; and **Li & Hoiem 2016**, "Learning without Forgetting",
   ECCV 2016. https://arxiv.org/abs/1606.09282 **[abstract]**
   - Found: a model trained on a new task forgets the old one. EWC slows learning on the weights the old task needed,
     which amounts to an L2-SP penalty weighted by how well each weight was determined. Li & Hoiem measure themselves
     against training on old and new data together, the reference these methods try to approach *when the old data
     is gone*.
   - For us: we still have every 2000 pair, so joint training avoids forgetting by construction and EWC is not
     needed. The EWC idea has one trap here. Using the 2000 fit's own certainty (its Newton Hessian) as the prior
     treats 1960 as *more 2000 data*. With 430k pairs behind it the prior would barely move. A different drafting
     style needs the extra between-class spread of item 3, not the 2000 posterior alone.

7. **Peduzzi et al. 1996**, "A simulation study of the number of events per variable in logistic regression
   analysis", J. Clin. Epidemiol. 49(12). https://pubmed.ncbi.nlm.nih.gov/8970487/ **[abstract]**
   - Found: logistic regression coefficients become biased and unstable below about 10 "events" (the rarer outcome)
     per fitted parameter. Later work finds 10 sometimes too strict.
   - For us: suppose 300 answered lots in patches give about 700 known pairs, of which perhaps 50-100 are borders.
     That supports roughly 5-10 freely fitted parameters, not 463. Refitting the full wall model on the answers would
     overfit badly. The answers can set an intercept and a few per-block scales; the rest must stay shared.

Also noted: **Sterzinger, Peer & Sablatnig 2025**, "Few-Shot Segmentation of Historical Maps via Linear Probing of
Vision Foundation Models", ICDAR 2025. https://arxiv.org/abs/2506.21826 **[abstract]** It reports state of the art on
the Siegfried map from 5-10 labelled examples by training only a thin layer over frozen general-purpose image
features. This is the other few-shot route: keep a strong general model frozen and fit a tiny head. It needs a large
pretrained network, which would be a new dependency to check for licence and determinism.

**What area 3 predicts for us.**

- Fine-tuning on answers alone will forget 2000 and overfit.
- Joint training on everything, with no class terms, lets the 430k 2000 pairs drown the answers.
- The hierarchical form fits our numbers: shared weights, a per-class intercept and a few per-class block scales,
  chosen on held-out answers. Built with Daumé's copies, it is a small change to the existing fit, and forgetting is
  checked by re-scoring 2000 by quarter.

## 4. Conditioning on scan-level statistics

1. **Perez et al. 2018**, "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
   https://arxiv.org/abs/1709.07871 **[abstract]**; and **Ha, Dai & Le 2016**, "HyperNetworks", arXiv 1609.09106.
   https://arxiv.org/abs/1609.09106 **[abstract]**
   - Found: FiLM scales and shifts each feature by amounts computed from a side input. A hypernetwork goes further and
     computes the whole weight set from a side input.
   - For us: in a logistic model, FiLM means interaction terms: weight = shared weight + (coefficient × scan
     statistic). To learn those coefficients you need labelled scans spread across the statistic. We have three
     labelled drawings, so three points to fit a function on. That is not enough without synthetic styles (§6).

2. **Huang & Belongie 2017**, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (AdaIN),
   ICCV 2017. https://arxiv.org/abs/1703.06868 **[abstract]**
   - Found: an image's "style" is carried mostly by the mean and spread of its feature channels. Swapping those
     transfers the style.
   - For us: a scan's "drafting style" can likewise be summarised by a short vector: pen-width distribution, ink
     darkness, line-spacing spectrum, legend textures. That vector is what to normalise by, or condition on.

3. **Lindeberg 1998**, "Feature Detection with Automatic Scale Selection", IJCV 30(2)
   (https://people.kth.se/~tony/papers/cvap198.pdf), and its companion "Edge Detection and Ridge Detection with
   Automatic Scale Selection", IJCV 30(2) (https://people.kth.se/~tony/papers/cvap191.pdf). **[abstract]**
   - Found: the image itself picks the scale at which to measure each structure. The ridge detector returns a line's
     width as the scale where its normalised response peaks. Features are then measured *at* that scale.
   - For us: the textbook version of "infer the line width from the scan and express features relative to it". The
     reference should be *local*: the lot lines next to a candidate border, on the same sheet. That handles sheets
     drawn with different pens, and old re-inked corrections.

4. **Epshtein, Ofek & Wexler 2010**, "Detecting Text in Natural Scenes with Stroke Width Transform", CVPR 2010.
   https://doi.org/10.1109/CVPR.2010.5540041 **[abstract]**; and **Su, Lu & Tan 2013**, "Robust Document Image
   Binarization Technique for Degraded Document Images", IEEE TIP 22(4) **[secondary: arXiv 1505.00529 describes its
   use]**
   - Found: the stroke width transform estimates stroke width at every pixel and groups pixels whose widths agree.
     Su et al. estimate the document's stroke width and size their local window from it.
   - For us: measuring pen width per scan and then setting every other scale from it is routine in document
     analysis.

5. **Moon, Lee, Mun & Lim 2021**, "Deep Learning-Based Method to Recognize Line Objects and Flow Arrows from
   Image-Format Piping and Instrumentation Diagrams for Digitization", Applied Sciences 11(21).
   https://doi.org/10.3390/app112110054 **[abstract]**
   - Found: line *types* (continuous, dashed and other signed lines) are recognised by combining classical line
     detection with a learned detector for the type marks. Precision was 96% and recall 90% over nine diagrams.
   - For us: if the 1960-1980 drafting style marks district lines by *type* (dash pattern, double line) rather than
     by width, a width ratio will never capture it. A line-type feature would be needed. Worth checking by eye on a
     1960 crop.

**What area 4 predicts for us.** Use scan statistics in two ways, in this order. First, divide features by measured
reference quantities, which needs no labels: local lot-line width, ink-to-paper contrast, and for class C the hatch
spacing and angle read from the legend. Second, and only with many styles available, let some weights depend on the
statistics (FiLM-style interactions). With three labelled drawings the second step can only be learned from
synthetic re-renders.

## 5. Hatched and patterned areas, texture boundaries, legend-guided segmentation

1. **Lin, Knoblock, Shbita, Vu, Li & Chiang 2023**, "Exploiting Polygon Metadata to Understand Raster Maps — Accurate
   Polygonal Feature Extraction", SIGSPATIAL 2023. https://doi.org/10.1145/3589132.3625659 **[abstract]**; and
   **Lin 2026**, PhD thesis "Learning to Interpret Historical Maps by Exploiting Polygon Metadata", USC.
   https://usc-isi-i2.github.io/papers/lin26-thesis.pdf **[full text, searched]**
   - Found: the map key (legend) is turned into bitmaps that guide a CNN to that key's polygons, so one model serves
     every key without retraining. Its steps copy how a person reads a map:
     - colour match;
     - tolerance for symbols and textures printed over the fill;
     - each pixel belongs to exactly one key;
     - boundaries separate same-coloured units.

     The thesis's outlook says the method is "primarily color-based". It also says that "intricate hatching patterns
     and overlapping translucent textures" are a direction still to handle.
   - For us: the legend-guided recipe fits class C exactly: a per-map legend, every area one key. But the published
     strength is colour, and ours are one-colour hatchings.

2. **Luo et al. 2023**, "Critical Minerals Map Feature Extraction Using Deep Learning", IEEE GRSL.
   https://doi.org/10.1109/LGRS.2023.3310915 **[abstract]**; and **Saxton et al. 2024**, "Accurate Feature Extraction
   from Historical Geologic Maps Using Open-Set Segmentation and Detection", Geosciences 14(11) 305.
   https://doi.org/10.3390/geosciences14110305 **[abstract]**; and **Duan et al. 2025**, "DIGMAPPER: A Modular System
   for Automated Geologic Map Digitization", arXiv 2506.16006. https://arxiv.org/abs/2506.16006 **[full text, polygon
   section only]**
   - Found:
     - Luo et al. stack the legend swatch onto the map patch as extra input channels, so a U-Net segments "wherever
       this swatch is". Median F1 was 0.67.
     - Saxton et al. use the legend units as one-shot prompts. Median F1 was 0.91 for polygons, but only for units
       with plenty of training annotations.
     - DIGMAPPER's polygon model scores by map quality class: F1 0.98 on "excellent" maps, 0.77 on "good" and 0.28 on
       "fair" (its Table 2). It has no separate mechanism for hatching.
   - For us: "legend swatch as exemplar" is now the standard design for many-style maps. Its results hinge on
     training data and scan quality. Our 1921/1929/1935/1948 scans should be put in a quality class by eye before
     choosing.

3. **Martin, Fowlkes & Malik 2004**, "Learning to Detect Natural Image Boundaries Using Local Brightness, Color, and
   Texture Cues", IEEE TPAMI 26(5). https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/mfm-pami-boundary.pdf
   **[abstract]**
   - Found: a boundary is scored by comparing histograms of texture elements in the two half-discs on either side of
     a point (the "texture gradient"). Plain edge detectors fire all over inside textured areas, and treating texture
     explicitly was required.
   - For us: the direct model for a class-C wall. Compare the hatching statistics (orientation, spacing, density) on
     the two lots of a pair, near their shared border, and measure the hatch-line ink in between. Hatch lines are
     exactly what makes a line-ink wall feature fire everywhere. The lot interior on each side is the "half-disc".

4. **Ustyuzhaninov, Michaelis, Brendel & Bethge 2018**, "One-shot Texture Segmentation", arXiv 1807.02654.
   https://arxiv.org/abs/1807.02654 **[abstract]**
   - Found: given an image and one reference texture patch, segment where that texture occurs. The model was trained
     only on synthetic collages of textures, and it carried over to natural images without fine-tuning.
   - For us: the task is legend swatch → lots, and synthetic collages are a cheap way to train it.

5. Hatching on maps and drawings, classical:
   - **Brügelmann 1996**, "Recognition of Hatched Cartographic Patterns", ISPRS Archives XXXI-B3 (Vienna).
     https://goobi.tib.eu/viewer/fulltext/1667440764/105 **[page text]**
   - **Wu et al. 2019**, "A New Gabor Filter-Based Method for Automatic Recognition of Hatched Residential Areas",
     IEEE Access 7. https://doi.org/10.1109/ACCESS.2019.2907114 **[full text, abstract and introduction]**
   - **Wu, Xiong, Zhao & Hu 2021**, "An Automatic Extraction Method for Hatched Residential Areas in Raster Maps
     Based on Multi-Scale Feature Fusion", ISPRS IJGI 10(12) 831. https://doi.org/10.3390/ijgi10120831
     **[abstract]**
   - **Ablameyko et al. 1998**, "Knowledge-based recognition of crosshatched areas in engineering drawings", LNCS.
     https://doi.org/10.1007/BFb0033267 **[title only]**

   - Found: hatched areas are found from the run lengths of black and white pixels along rows and columns
     (Brügelmann), from Gabor filters tuned to the hatch direction (Wu 2019, which assumes the one 45° hatch of its
     map series), or with a U-Net. The U-Net reached 94% IoU on a single map style, and its authors note that older
     methods need hand-set parameters. Broken hatch lines, uneven spacing and over-printing are the usual troubles.
   - For us: the classical methods hand-set the hatch spacing and angle per map series. The legend lets us *measure*
     them per scan: take the autocorrelation or Fourier peak of each swatch. That is "infer the hatching parameters
     from the scan", with the legend as a free, labelled sample of every class.

**What area 5 predicts for us.** Class C wants two new evidence models, plugged into the same solver energy. The
first is a **label** model: how likely is each lot's interior texture under each legend swatch? The second is a
**wall** model: does the texture change across the pair, beyond what a lot line explains? The trouble spots will be
lots smaller than a couple of hatch periods (too few lines to judge), hatching interrupted by streets, lettering and
lot lines, and legend swatches printed at another density than the map. Per-lot evidence should be weighted by how
much hatching the lot actually holds, as a likelihood, not a cutoff.

## 6. Synthetic, style-randomised training data for maps

1. **Tobin et al. 2017**, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real
   World", IROS 2017. https://arxiv.org/abs/1703.06907 **[abstract]**; and **Zhou, Yang, Qiao & Xiang 2021**, "Domain
   Generalization with MixStyle", ICLR 2021. https://arxiv.org/abs/2104.02008 **[abstract]**
   - Found: if training renders vary the nuisance factors widely enough, a model trained only on renders works on
     real images. MixStyle makes new "styles" by mixing feature statistics between training images.
   - For us: render the 2000 answer with pen widths, line types and ink levels drawn from the measured spread of the
     class B scans. A wall model trained that way has to rely on *relative* evidence. It should also stop a 5% width
     error from costing thousands of lots.

2. **Jiao, Heitzler & Hurni 2022**, "A fast and effective deep learning approach for road extraction from historical
   maps by automatically generating training data with symbol reconstruction", Int. J. Appl. Earth Obs. Geoinf.
   https://doi.org/10.1016/j.jag.2022.102980 **[abstract]**; and **Mühlematter et al. 2024**, "Probabilistic road
   classification in historical maps using synthetic data and deep learning", arXiv 2410.02250.
   https://arxiv.org/abs/2410.02250 **[abstract]**
   - Found: Jiao et al. re-draw known vector roads in the map's own symbols ("imitation maps"). With little real
     labelled data, training on imitation maps alone gave satisfactory results, and mixing imitation with real
     beat real-only on some measures. Mühlematter et al. re-paint road segments in each class's symbol to train a
     classifier with no hand labels.
   - For us: the closest precedent to re-drawing the 2000 districts (or 1921-like districts built from modern lots)
     in 1921 hatch symbols. The scan's own background, streets and lettering stay underneath; only the district fill
     is painted.

3. **Li, Guan, Yu, Chiang & Knoblock 2021**, "Synthetic Map Generation to Provide Unlimited Training Data for
   Historical Map Text Detection", SIGSPATIAL GeoAI 2021. https://arxiv.org/abs/2112.06104 **[abstract]**; and
   Petitpierre 2026 (§1 item 2)
   - Found: a style-transfer network turned modern map tiles into historical-looking ones, which beat out-of-domain
     training for text detection. Petitpierre's renders with random hatchings and line widths gave +5 mIoU, and the
     author treats the renders as an approximation to be checked against real data.
   - For us: synthetic data helps in every case, but always as a supplement that is checked on real labelled maps,
     never alone. Our few hundred owner answers are that check.

**What area 6 predicts for us.** Re-rendering is the only way to get *many labelled styles*. Learning how evidence
should change with pen width or hatch density needs many styles, and we have three. The renders must take every
number (widths, spacings, noise, district sizes) from measurements on the real scans, as the owner's synthetic-test
rule already requires. They must be scored on the owner's real answers.

## Ranked approaches for our situation

What we have: about 430k labelled pairs on class A (2000), a few hundred lot answers each on class B (1960) and
class C (1921), and per-scan statistics including the legends. "Score" below always means the same measure, so
methods can be compared. The measure has three parts:

- the share of held-out answered lots the solve gets right, where the held-out lots are blocked by area and never
  used in any fit, compared lot by lot between methods;
- held-out pair log loss for wall fits;
- 2000 wrong lots, with each quarter scored by a fit that did not see it, as the check that 2000 has not been
  forgotten.

**1. Scan-relative features and a per-scan base rate, all label-free (classes A and B now; base for everything).**

- What: express each ink feature relative to a reference taken from the same kind of ink on the same scan, such as
  the local lot-line width and the scan's ink-to-paper contrast. Alternatively, quantile-map each feature to its 2000
  distribution, measured on lot-line ink only, not the whole page. Re-estimate the border base rate per scan with
  Saerens' EM.
- Data: none beyond the scan.
- Risks:
  - a reference statistic that mixes ink kinds (the 1996-2000 lesson);
  - hiding a real change in meaning (dashed lines, double lines);
  - EM on a badly calibrated model drifting.
- It worked if:
  - on the 1996-2000 twin, wrong lots against the table are no worse than unscaled (15,974);
  - 1960 own-labels-kept rises above about 81%;
  - then, on the 1960 answers, held-out lot accuracy beats the single width ratio, paired.

**2. Joint hierarchical wall model (partial pooling, built with Daumé's copies) (class B, then class C).**

- What: one ridge logistic fit over the 2000 pairs plus every answered pair. Give each class its own intercept and a
  few per-class scales, one per evidence block (ink, widths, context), and keep the other weights shared. The ridge
  strength on the class copies is picked by area-held-out log loss on the answers. This is the principled form of
  "start from 2000 and restart training"; L2-SP toward 2000 is its one-knob special case and the first thing to try.
- Data: the 2000 pairs (have), and answers given in contiguous patches so that both lots of many pairs are known.
- Risks:
  - few border pairs (expect 50-100), so keep the number of per-class parameters to about 10 or fewer;
  - answered patches unlike the rest of the map;
  - the pooling strength chosen on a noisy set.
- It worked if, on the 1960 held-out answers, it beats approach 1 alone, paired. The 2000 leave-one-quarter-out
  wrong lots must not change.

**3. New evidence for class C: legend-exemplar texture labels and texture-change walls.**

- What:
  - measure each legend swatch's hatch statistics on that scan: orientation, spacing, line width and density;
  - give each lot a likelihood under each swatch, weighted by how much hatching the lot holds;
  - compare the two lots' textures in a pair, plus any printed line, for the wall;
  - keep the solver's energy, lot geometry and context terms as they are.

  Calibrate the new terms' weights through approach 2, with per-class intercept and scales.
- Data: legend crops (free); the 1921 answers for calibration and scoring.
- Risks:
  - small lots that hold too little texture;
  - hatching broken by streets, lettering and lot lines;
  - legend swatches printed differently from the map body;
  - classes whose hatchings differ only in spacing.
- It worked if:
  - held-out lot accuracy on the 1921 answers, with a per-legend-class confusion table, beats the old pattern-map
    reading on the same lots;
  - the same code, with no refit, then does as well on 1929/1935/1948 by eye on the review page.

**4. Synthetic re-rendering of known answers in each class's style (domain randomisation) (supports 1-3).**

- What: re-draw the 2000 districts on the 2000 lot geometry, or 1921-like districts built from modern lots. Use
  class B pen widths and line types, or class C hatch fills, with every parameter drawn from measurements on the real
  scans. Then use the renders to:
  - make the walls depend on relative evidence;
  - fit FiLM-style "weight depends on scan statistic" terms, if wanted;
  - train the class C texture models.
- Data: the 2000 table, and style measurements from the B and C scans. No new answers.
- Risks:
  - the gap between renders and reality (missing scan defects, wear, registration);
  - time spent on a renderer;
  - renders whose statistics are too clean.
- It worked if, on the real held-out answers, the version trained with renders beats the version without, paired,
  on the same lots.

**5. How to spend the owner's answers (applies to 2 and 3).**

- What: split each map's few hundred answers in two:
  - about half chosen at random within held-out areas, for scoring only;
  - about half in contiguous patches around the walls the current solve is least sure of, so that most answered
    lots have answered neighbours. This is uncertainty sampling; see **Settles 2009**, "Active Learning Literature
    Survey", UW-Madison TR 1648, https://research.cs.wisc.edu/techreports/2009/TR1648.pdf **[abstract]**.

  For 1921, cover every legend class.
- Risks: answers chosen by uncertainty are a biased sample, so never score on them; patches that are too small yield
  few border pairs.
- It worked if the score set's 95% interval is narrow enough to separate the methods compared. With 300 lots near
  90% right, it is about ±3.4 points; the paired, lot-by-lot comparison is sharper.

**Not recommended.**

- Plain fine-tuning on the answers alone. The fit is convex, so the 2000 start is forgotten; 463 weights on 50-100
  border pairs overfit; 2000 gets worse.
- A hypernetwork or FiLM trained only on real labelled drawings. Three labelled drawings are three points.
- Test-time entropy minimisation (Tent), self-training, and neighbouring-year pseudo-labels (Wu 2023, Yuan 2025). All
  learn from our own or another year's solve, which the owner has ruled out. They are listed so they are not
  re-proposed.
- Whole-feature alignment (CORAL) across classes with different border rates. Zhao et al. show this can make things
  worse.

## Open points for the owner

- Whether "restart training on all three" should mean approach 2 (one shared model with small per-class offsets).
  The literature says yes. The alternative is three separate models, which the answers cannot support.
- Whether 1960's district lines differ by *type* (dash pattern, double line) as well as width. One look at a 1960
  crop beside a 2000 crop would settle whether a line-type feature is needed (§4 item 5).
- Whether the 1921 legend swatches can be cropped reliably on each pattern map. They are the free labelled sample
  that makes class C feasible.
