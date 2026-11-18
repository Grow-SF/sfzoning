# The text-reader research on the 2000 map: the full record (2026-11-15 to 2026-11-18)

**Status: parked on 2026-11-18, not abandoned.** The owner chose to ship the 2000 map from the solve on the saved
Terra and Sol reads ("pivot to shipping with the saved reads from Terra and Sol. I want to get the published map
updated"). The shipping plan is `docs/superpowers/plans/2026-11-18-ship-2000-saved-reads.md`. This note records
everything tried on the way to reading the map's labels ourselves, what each attempt showed, and the controller's
analysis that led to the pivot. The working records it draws on are archived next to it, in
`docs/research/joint-solver/notes/2026-11-18-text-reader-records/` (see section 11).

All numbers below come from the ledgers, notes and result files named beside them. Times are Pacific.

## Contents

1. [In one page](#1-in-one-page)
2. [The goal, and why](#2-the-goal-and-why)
3. [Words used in this note](#3-words-used-in-this-note)
4. [Timeline: the plans in order and the owner's turns](#4-timeline-the-plans-in-order-and-the-owners-turns)
5. [The headline table](#5-the-headline-table)
6. [Every attempt](#6-every-attempt)
7. [Why the best version still gets labels wrong: the five-way causes](#7-why-the-best-version-still-gets-labels-wrong-the-five-way-causes)
8. [The analysis (the controller's, given to the owner on 11-18)](#8-the-analysis-the-controllers-given-to-the-owner-on-11-18)
9. [The owner's working rules this research taught](#9-the-owners-working-rules-this-research-taught)
10. [If we resume](#10-if-we-resume)
11. [Where every output lives, and what is reusable](#11-where-every-output-lives-and-what-is-reusable)

## 1. In one page

- **The aim.** Turn a scanned map into a coded map on the owner's Mac, with no LLM output at inference. Terra and
  Sol (the saved LLM reads) are only for training and scoring.
- **The measure that ran the last two days.** Of the owner's 699 answered zoning-code labels, how many our own chain
  (our detector finds the word box, our reader reads it) reads right. A label with no box counts as wrong.
- **Where it went.**

  | Version | Read right of 699 | Interval |
  |---|---|---|
  | Today's chain (the shipped stages) | 286 = 40.9% | 37.2-44.6% |
  | crude-map: the new detector's word patches, boxed | 343 = 49.1% | 45.3-52.8% |
  | crude-kind: those patches split where the detector's kind changes | **414 = 59.2%** | 55.5-63.1% |
  | learned-links: patches joined by a learned same-word link | 351 = 50.2% | 46.5-54.0% |
  | The same reader at Terra's boxes (for scale) | about 96% | 94.4-97.4% (662 labels) |

- **What worked:** the simplest uses of the new detector. Plain word patches (+8.2 points) and splitting them by
  the detector's own kind call (+10.2 more).
- **What failed:** six more elaborate add-ons: the same-word affinity, code spotting, box choice, the vote mixture,
  the context pack (for codes) and learned links. Words merged into one box stayed the biggest cause of wrong labels
  (123 of crude-kind's 285).
- **The core mistake.** For two days a stand-in score (reading at our boxes on 699 labels) was optimised, and the
  map itself was never measured with our own reads. Whether 59.2%, or even 46%, already makes a better map than the
  published one (35,455 wrong lots) is unknown.
- **The decision.** Ship the map from the saved reads (35,455 wrong lots down to about 4,543). The research is
  parked. Its first step on return is to measure the map with our own reads (section 10).

## 2. The goal, and why

The owner's end goal (11-16, about 11:15): "Here's an old map from some year of some city", and the tool runs every
part of the scanned map to a modern, correctly coded map, locally, with no external calls. Text matters only in
service of the map: the zoning labels (which code each district gets) and maybe street names for the skewing. There
is no text dump.

Three of the owner's decisions frame everything below:

- **No LLM output at inference** (11-16, memory `deterministic-no-llm-answers`). Paid or free, saved or fresh: none
  of it enters the pipeline that makes a map. This later reached the saved reads too, once our reader could replace
  them.
- **Terra and Sol are training labels and the scoring set only** (11-16 about 10:50, memory
  `terra-outputs-as-training-data`). "For future maps I don't want to run paid calls. And I know it misses text."
  So our detector must find text boxes without outside information, and Terra's reads score it.
- **A somewhat worse reader than the LLM reads is acceptable** (11-16 about 11:15), and "it's okay to just be
  better" (memory `ship-better-not-perfect`): the published map has many errors.

On 11-16 about 18:30 the owner held the 2000 publish until the solve's labels came from our own detector and reader.
On 11-18 the owner reversed that for this release.

## 3. Words used in this note

- **Terra, Sol.** The two saved LLM reads of the 2000 sheets. Terra gives word boxes and strings; Sol gives label
  points. About 195,000 Terra word reads exist across the maps.
- **The owner's answered labels.** 719 labels the owner checked by hand (`pipeline/inputs/code-check-2000/`): 699
  zoning-code labels and 20 "no label". They are never trained on, nor is anything in the same block.
- **Read right at our boxes** (the headline). The code label's point falls in one of our boxes, and our reader's
  top reading of that box is the owner's code. No box at the point counts as wrong.
- **The detector.** Our U-Net (a convolutional network, a CNN) that gives, for every pixel of a scan, the chance it
  is text, and later also word-shape outputs (section 6). "Held out by quarter" means a detector trained on three
  quarters of the city and scored on the fourth.
- **The reader.** "The text reader": a small CTC recogniser (a standard way to read a word image into letters)
  trained from scratch on Terra's word crops plus synthetic crops. It replaced "the template reader" on 11-16.
- **Paired interval.** The difference between two versions on the same labels, with a 95% interval from resampling
  whole blocks (labels in one block move together). An interval that excludes zero means the change is beyond noise.
- **MAP, even odds.** Calling a pixel by whichever answer the network gives more than half the chance. It is the
  network's own decision, not a picked level.
- **The solve, wrong lots.** The 2000 joint solver turns label readings and wall evidence into a code for every lot.
  Wrong lots count lots whose code differs from the table. The published map has 35,455; the solve on the saved
  reads has 4,543.

## 4. Timeline: the plans in order and the owner's turns

| When | Plan or event | The owner's turn that steered it |
|---|---|---|
| 11-15 08:55 | **One text detector, trained in minutes** (spec a60e1fc, plan b56a07c) | "Faster at same quality is good too"; judge by words found and district-line ink called text. Later: lock in 4,000 steps; try bigger batches first. |
| 11-15 12:49 | **The template code reader** (spec 66e1c9a, plan f37c575) | The owner's idea: a probabilistic reader, prior over what text says at a place times the likelihood of the ink. "If you think that is the best approach, sure." |
| 11-15 21:42 | Template reader's full run | "You should interrupt it; load too high." |
| 11-16 08:20 | Template reader's 3.6 h rescore stopped | "Good grief that's nuts. Every rotation? That doesn't sound like probabilistic recognition!!" |
| 11-16 08:47 | **Read along the word** (spec b81a063, plan 7aab4e7, R1 ce828df) | "Sounds right." Then (10:10): "Why are you proving reproducibility before you even know if the result is any good?" |
| 11-16 10:50-11:15 | Goal correction | No Terra boxes or Sol points at inference; Terra only to train and score; no LLM output anywhere in the pipeline; a somewhat worse reader is fine. |
| 11-16 11:54 | **Labels from the scan alone** (spec 24fa2ce, plan 63fd9a5) | "Did you red team this? Is this the right approach? There should be published libraries or research papers." A red team and a library survey followed; the plan was replaced. |
| 11-16 13:03 | **The compound search move and the reader spikes S0-S3** (plan 9358c9b; results ee2a59e) | Acceptance criteria fixed before the runs ("defining your acceptance criteria to avoid bias. ok"). On S2: "Why would I prefer to be 800 times slower...?" The CTC reader is adopted. "It's okay to just be better." Then: hold the publish until labels come from our reader. |
| 11-16 21:00 | **The text reader's labels, then publish** (plan 2552f04; T0-T5) | "Ok go on the plan. For release, we will regenerate all maps when we're ready." On T1's jump in wrong lots: "small errors can result in huge shifts in the number of wrong lots": keep the plan as written. |
| 11-16 23:20 | Detector over-call | "Did you reach for 'likely cause' without validating your hunch?" Then: the detector "learned the location of a label, not the label text"; "not realizing the background ink isn't text". Overnight go: debug, brainstorm, red team, plan, implement. |
| 11-17 00:09 | **Text vs line** (spec be37735, plan aac94be) | Ran overnight; nothing adoptable by the rule written in advance. |
| 11-17 morning | **The comparison report** (notes bf5723d) | Asked for a literature review and a step-by-step trace of our pipeline against the best methods. |
| 11-17 08:00 | **Map text the field's way** (spec 2c51657, plan 928aa5a; 15 tasks) | "Brainstorm an approach that implements and tests all of those." A general model, with optional context packs at read time; judged on the Mission Street review page. |
| 11-17 09:05 | Throughput | "Could we be doing more?" Implementers ran in parallel on separate files. |
| 11-17 10:35 | Cutoffs | "I KEEP TELLING YOU TO DO THIS AND NOT DO ARBITRARY THRESHOLDS." Spec and plan fixed (26985e3). |
| 11-17 11:10-11:25 | Cores and the box model | "The R core size measure is still arbitrary numbers that you've picked." "There is a hidden function that you should be able to learn automatically." Plan amended (eb0cf7e); Task 15 added. |
| 11-17 12:45-16:00 | Pictures | "Do we have an updated map ... with the detections?" "Can I see what we have so far? Can you make any kind of map?" |
| 11-17 17:10-17:20 | **End to end first** | "I can't believe you don't have any intermediate things you can show me." "You're telling me that after all of that work today we're at 40.9%?" "You absolutely should have validated the ideas before spending 8 hours on them." |
| 11-17 18:00-22:45 | The crude chains | First end-to-end result (crude-map). On a tag split: "It sounds like you're going to split them based on some kind of arbitrary cutoff. Shouldn't you instead learn that an individual word is an individual word?" Then "Yes try affinity"; "Do some superpowers brainstorming on next approaches. You're not thinking"; "Why not go direct to the 'detect straight from pixel'?"; "Run the test"; "Yes" (kind split). |
| 11-17 22:45-23:30 | The review page | "That text reader progress artifact doesn't show me any boxes." "Bro this is unusable. Apply some damn UI principles." The page was redesigned. |
| 11-17 late | Learned links | "Explain to me why the text isn't clearly detected." "'Agrees' sounds like something that should be learned." "Try it." |
| 11-18 00:30 | Tangent? | "Is this an unnecessary tangent? ... brainstorm what we should do next (finishing the model, pivoting to a CNN, or using the Terra reads to ship)." |
| 11-18 00:40 | **Pivot** | "Save all of our work for this tangent, as well as a full write-up of everything we've attempted and your analysis just now, and then we should pivot to shipping with the saved reads from Terra and Sol." Ship plan b23e280. |

## 5. The headline table

All 699 answered code labels, each quarter read by a detector and a reader that never saw it, seed 0. Intervals
resample whole blocks. Sources: `pipeline/data/solver/experiments/wordread/detector/runs/<run>/vs_*.json`.

"Today" is the shipped stages: `solver-boxes` read by the text reader.

| Version | Right | Share [95%] | Paired change [95%] |
|---|---|---|---|
| Today | 286 | 40.9% [37.2, 44.6] | |
| crude-map | 343 | 49.1% [45.3, 52.8] | +8.2 pt [+3.7, +12.6] vs today |
| crude-kind | **414** | **59.2% [55.5, 63.1]** | +18.3 pt [+13.9, +22.7] vs today; +10.2 pt [+7.3, +13.1] vs crude-map |
| learned-links | 351 | 50.2% [46.5, 54.0] | +9.3 pt [+4.8, +13.8] vs today; -9.0 pt [-12.0, -6.1] vs crude-kind |

By quarter (right of the quarter's labels; Map, Kind and Links are crude-map, crude-kind and learned-links):

| Quarter | Labels | Today | Map | Kind | Links |
|---|---|---|---|---|---|
| NW | 190 | 87 | 91 | 113 | 91 |
| NE | 285 | 92 | 142 | 167 | 150 |
| SW | 94 | 46 | 50 | 60 | 50 |
| SE | 130 | 61 | 60 | 74 | 60 |

crude-kind beat crude-map in every quarter (NW +11.6, NE +8.8, SW +10.6, SE +10.8 points, each interval above
zero).

**For scale:**
- The same text reader at Terra's boxes reads 0.961 [0.944, 0.974] of the 662 answered labels that have a Terra box
  (T3, 8157a97). At our boxes in T3 it read 0.462 of the 633 labels that got a box, 0.412 counting the 86 with none.
- On the Mission Street sheet (p07, 122 answered labels): today 46, crude-map 71, crude-kind 79, Terra's own reads
  107.

## 6. Every attempt

Each entry gives the idea, what was built, the measure, the result, the verdict and the lesson. Commits are on
`solver`. Code paths are under `pipeline/` unless they start with `docs/`.

### 6.1 Before the fields way (11-15 to 11-17 morning)

#### The detector speed recipe (11-15)

- **Idea.** One text detector for every map year, trained in minutes instead of the four quarter models' roughly
  50 minutes each, at the same held-out quality.
- **Built.** Training data from every map Terra read, held out by one dense area (Mission, Noe, Glen Park); the
  recipe options (half size, loss on the ink and the pixel next to it, schedule length, batch, light decoder, crop
  size, float16, weight averaging); `sfzmaps solver text-trial`; one `solver-text` stage. Commits 8f0103b..915ab84
  (b5abd6d, 78c74c1, cb12116, 6f6f4be, e7550a4, b0c5821, 1bcd48d, ad9db9f). Files `src/sfzmaps/solver/textnet/`
  (`data.py`, `train.py`, `scores.py`, `trial.py`, `stage.py`).
- **Measure.** Words found and long-line ink called text on the validation area, three seeds per length; then the
  test drawings (1972, 1975, 1980), never trained on.
- **Result.** 4,000 steps at half size passed (for example seed 2: 94.08 words / 4.12 long-line, against the old
  recipe's 94.15 / 4.10); 2,000 steps failed one seed. Training about 6.1 minutes idle. Bigger batches, the light
  decoder, 256-px crops and float16 gave no reliable gain: the step is bound by GPU memory traffic, linear in
  pixels. Both lettering styles together beat modern-only on 1980 by about 1.1 points of words.
- **Verdict.** Adopted (the `solver-text` stage).
- **Lesson.** Two faults hid inside it and surfaced a day later. The 4,000-step schedule left the reading-direction
  head half trained (found in R1), and the training target counted line ink inside a word box as text (found on
  11-16). Neither showed, because the gates measured with the same rectangles the detector learned from, and nobody
  read the direction output. A gate built from the training target can't see the target's errors.

#### The template code reader (11-15 to 11-16)

- **Idea.** The owner's: a probabilistic reader. A prior over the roughly 60 codes at a place times the
  likelihood of the ink given each code in the map's own lettering, with a little mass for "other" and "none".
- **Built.** Labels and the code list (5209120: 1,905 labels; 717 answered); glyph templates learned by hard EM
  from a plain sans font (fda4c42); the posterior over every code, "none" and "other", summed exactly over a grid
  of poses (cfd5253, d80cc7d, df61853, c626e46). Files `src/sfzmaps/solver/codereader/` (`labels.py`, `glyphs.py`,
  `posterior.py`).
- **Measure.** Agreement on training labels first; the owner's answers were to be the test (Task 4).
- **Result.** On training labels: boxes 814 of 885 top-1 (about 92%), point labels 24 of 50. A full run took 93
  minutes, then 3.6 hours after the fixes. At Terra's boxes (measured later in S2): top-1 0.949, -log p 0.186.
- **Verdict.** Stopped by the owner during the 3.6 h rescore ("Every rotation?"). Retired as a reader on 11-16; kept
  as the generator of synthetic crops.
- **Lesson.** A brute-force sum over every pose is not what the owner means by probabilistic recognition: infer the
  pose from cheap evidence and sum only where the mass is. Also, a hand-built model lost to a standard trained one
  (S2) at 800 times the cost.

#### Read along the word (11-16 morning)

- **Idea.** Infer each label's pose from a reader-only detector (its text chance and reading direction), then read
  along that frame.
- **Built.** R1 only: four held-out detectors, one per quarter (stage `solver-text-reader`, ce828df, 25.8 minutes
  of training). R2-R4 were not built.
- **Measure.** Direction agreement with the template reader's fitted poses; word ink called text.
- **Result.** The direction head was nearly useless: within 45° on 15% of boxed labels (41% reversed), against 95%
  for the old 8,000-step quarter models. Root cause: the 4,000-step schedule leaves that head half trained (every
  8,000-step model 86-92%). Separately, a 328-pixel "unknown" disc around every Sol point left 43% of the map
  untrained, so district lines were called text.
- **Verdict.** Superseded at 11:00 by the owner's goal correction (it used Terra boxes and Sol points as evidence).
  R1's held-out stage stayed.
- **Lesson.** Look at an output before building on it. And the owner's "quality first, reproducibility later".

#### Labels from the scan alone, P1-P6 (11-16 midday)

- **Idea.** A detector with axis and box outputs, a reader of every box, the solve's labels from them.
- **Built.** Spec 24fa2ce and plan 63fd9a5 only.
- **Measure.** A red team and a library survey (archived: `2026-11-16-labels-from-the-scan/redteam.md`,
  `survey.md`).
- **Result.** The red team found the plan would still read LLM output (the walls read Terra boxes; the warp's first
  placement used LLM anchors), reinvented standard detection and recognition, worked on the smallest error source,
  and had a "real mass" rule that was a cutoff. The survey found no published work on reading zoning codes;
  detection on historical maps is mostly solved, recognition is hard; usable permissive libraries are docTR
  (Apache-2.0, PyTorch, mps and CPU), PaddleOCR (CPU only on macOS) and Kraken, or an in-repo CRNN.
- **Verdict.** Not built. Replaced by the CTC reader, boxes from the existing segmentation, and spikes S0-S3.
- **Lesson.** Red-team a plan and check the literature before writing it (the owner's question).

#### Spikes S0-S3 (11-16 afternoon)

Results: `docs/research/joint-solver/notes/2026-11-16-search-move-and-reader-spikes.md`.

- **S3, word boxes from the detector's chance.**
  - Built: `scripts/spikes/s3_boxes.py` (efcdccb..99b3b4b).
  - Result: about 90% of zoning codes found on the three test drawings (89.9% [88.2, 91.6] at level 0.8), about three
    quarters with the axis within 10°. Short, stacked and one-letter codes flip a quarter turn. Not held out on 2000.
  - Verdict: the basis of T2's `solver-boxes`. Lesson: "found" here meant overlap with a Terra box; it said nothing
    about whether the box held one word.
- **S2, the CTC text reader** (the attempt the task list calls "the CTC text reader").
  - Built: `scripts/spikes/s2_ctc/` (1874d9a..ca54663); later the stages `solver-text-crops`, `solver-text-train`,
    `solver-text-read` (T3).
  - Measure: the owner's answered labels with a Terra box (662), three seeds, against the template reader.
  - Result: top-1 0.956-0.973 and -log p 0.102-0.126, against 0.949 / 0.186. About 800 times faster (0.02 s against
    16 s a label), deterministic on CPU. docTR's pretrained reader stayed a little ahead.
  - Verdict: adopted (the owner). Lesson: weigh a large cost gap plainly (memory `weigh-cost-in-near-ties`).
- **S0, the template reader as the only reading** (`scripts/spikes/s0_reader_only.py`, 02dd797, 7e62d73).
  - Measure: the solve with the template reader's posterior in place of the saved readings, label positions still
    from the saved reads; old search (final 7,100 wrong lots).
  - Result: labels right 88.78% against 94.47% (-5.69 pt [-6.99, -4.40]); incidents +116; wrong lots -369
    [-1,201, +507], not distinguishable.
  - Lesson: a reader with about 9% misreads, at the right label positions, did not visibly hurt wrong lots.
- **S1, what reader error costs the map** (`scripts/spikes/s1_degraded.py`).
  - Measure: the saved readings degraded at known rates (misses 5-40%, misreads 2-20%), three seeds, old search.
  - Result: each point of confident misread costs about 1,132 wrong lots [756, 1,508]; each point of miss about 428
    [235, 621]; a misread at the reader's own low sureness about 462 [308, 616]. At 40% misses the map had 26,878
    wrong lots; at 20% confident misreads, 32,467.
  - Lesson: how sure a reader is when it is wrong matters about as much as how often.

#### The heights spike (11-16 21:14, d9e3987)

- **Idea.** Can the zoning-trained detector and reader read height labels?
- **Result.** The detector and S3 boxes found 218 of 221 saved height labels; the reader read 2 (8% with a height
  grammar). It never saw "NN-X" or "OS", and read "OS" upside down as "SO". Outputs
  `pipeline/data/solver/experiments/spike-heights/`.
- **Verdict.** Not pursued; heights ship as published.

#### The text reader's labels, T0-T3 (11-16 21:00 to 11-17 01:26)

- **T0, the scoring set pinned** (a19ca2d..adcfbfa): the score's label set separated from the labels the solve reads.
  The rebuilt report was byte-identical. Done.
- **T1, the walls without Terra** (700f40c..1e6b64d): the walls read the held-out detectors' text chance instead of
  Terra's boxes. Labels right 94.79 to 94.05% [-1.43, -0.11]; incidents 862 to 921; wrong lots 4,543 to 7,001, mostly
  three patches that lost a table border. Kept by the owner. The ship plan restores Terra-box walls for this release.
- **T2, word boxes** (5e5b0c5, 6d00181; stage `solver-boxes`): 8,794 boxes; the level per quarter came out at
  0.99-0.995 because the held-out chance was saturated near 1. Terra code words found at IoU 0.3: 47-79% by
  quarter. Misses were mostly merges and splits.
- **T3, the text reader as stages** (90b1069..8157a97): at Terra's boxes 0.961 [0.944, 0.974]; at ours 0.462 on the
  633 labels with a box, 0.412 counting misses. Merged-with-line boxes read 0.185.
- **T4 (the solve's labels from our boxes and reader) and T5 (publish wiring)** were put on hold: box error
  dominated, and the detector work ran first. T4 was never run. T5 moved to the ship plan.
- **Lesson.** T4 was held at 46% on the belief that it wasn't good enough, without measuring the map (section 8).

#### The detector over-call and its root cause (11-16 night)

- **Idea.** Test the T2 implementer's guess that the held-out detectors over-call line ink because of the
  328-pixel unknown disc, before retraining (the owner's "do the cheap test").
- **Built.** `scripts/spikes/detector_overcall.py` (ba36061); results
  `pipeline/data/solver/experiments/detector-overcall/results.md`; the root-cause note
  `docs/research/joint-solver/notes/2026-11-16-text-target-root-cause.md` (dc32995); the literature note
  `2026-11-16-text-graphics-separation-literature.md`.
- **Result.** Non-text line ink called text: 31% inside the discs for the held-out detectors, 0.1-0.3% for 30-pixel
  models. The disc came from one giant "SH" word on a 1968 sheet. The deeper cause, which the owner pointed to: the
  training target was the whole Terra word rectangle (`textnet/data.py:469`, `text = (word > 0)`), so any block,
  street or district line crossing a label was taught as text, in every detector since the R&D script. Every score
  used the same rectangles, so the error counted as right.
- **Lesson.** The target was never decided, only carried forward ("as R&D"). Check what a pixel label means,
  pixel by pixel, and don't build the measure from the target.

#### Text vs line labels (11-17 overnight)

Results: `docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md`.

- **Idea.** Teach letters, not line ink: mark line ink inside word boxes as unknown (the "control"), and add
  composites of real clean words pasted on real lines (the "variant").
- **Built.** A text-vs-line gate (`textnet/linegate.py`, `linerules.py`), baselines, the label rule
  (`zones.py`, `data.py`), composites (`composite.py`), `scripts/textlines/` (aac94be..a2b1dc7). The plan's ledger
  is archived as `2026-11-17-text-detector-text-vs-line/`.
- **Measure.** District-border ink in word boxes called text (M1a), all line ink in boxes (M1b), glyph ink (M2),
  ink outside boxes (M1c), clean-word letters; three seeds; a keep rule written before the runs.
- **Result.** Against the old NE model, the control cut border ink in boxes called text from 0.63 to 0.40 (-0.225
  [-0.257, -0.194]) and all line ink in boxes from 0.82 to 0.55. It cost code letters away from lines (-0.033) and
  raised ink outside boxes called text (+0.027). The composites changed nothing beyond noise. A dashed pier-head line
  became a new over-call.
- **Verdict.** Not adopted (nothing passed the rule written in advance). The control's label rule
  (`labels="lines-unknown"`) became the per-ink head in the fields-way runs.
- **Lesson.** Half the line ink in boxes was still called text: leaving it unknown gives the network no push. The
  adoption bar (parity with a detector that calls all ink text) could not be met by any line-aware detector.

#### The comparison report and its eight items (11-17 morning)

- **Idea.** Trace our pipeline and the best published methods step by step, and list what we miss.
- **Built.** Notes `2026-11-17-our-text-pipeline-trace.md`, `2026-11-17-text-detectors-traced.md`,
  `2026-11-17-map-text-approaches-traced.md` (bf5723d); the report https://claude.ai/artifact/BAGwr4yy8Bfsf1pqWVvVhC.
- **Result.** No published method forms words from connected ink; the box step cost about 49 points of reading
  (96% at Terra's boxes against 48% at ours); the reader had only practised on good boxes; the map's geography was
  unused.
- **The eight items and what became of them:**

  | # | Item | Tasks | Outcome |
  |---|---|---|---|
  | 1 | Words from word cores and box votes, not connected ink | 5, 6, 8 | Cores gave crude-map (+8.2) and crude-kind (+18.3). The vote mixture was parked (too slow; degenerate on real pages). |
  | 2 | Separate jobs: a per-ink line head, word-area heads | 5 | Built on the one U-Net; the detector behind every crude chain. |
  | 3 | The reader trained on our own boxes, sloppy crops, masking | 3, 4 | +1.75 pt [-1.2, +4.8] on today's boxes: not kept. The mask arm and a retrain on new boxes never ran. |
  | 4 | Context packs (streets, blocks, the era's codes) | 10, 11 | Codes not kept (headline -1.0 / -0.4); street and block reads clearly better. |
  | 5 | The reader picks between box versions | 9, 11 | -2.3 pt [-4.1, -0.4]: not kept. |
  | 6 | Terra's labels cleaned (doubt as weights) | 7 | Weights fitted; detector arm (b) trained, never scored end to end. |
  | 7 | Grey input and full resolution | 12 | Code only; never compared. |
  | 8 | Map-like synthetic scenes from DataSF geometry | 13 | Built and checked by eye; paused; never trained with. |

### 6.2 Map text the field's way, Tasks 1-15 (11-17)

Spec `docs/superpowers/specs/2026-11-17-map-text-fields-way-design.md`, plan
`docs/superpowers/plans/2026-11-17-map-text-fields-way.md`, ledger `2026-11-17-map-text-fields-way/progress.md`.
Every item was to be screened on NE at two seeds and kept by a rule written before its runs.

#### Task 1: the end-to-end measures

- **Built.** `src/sfzmaps/solver/wordread/measures.py` (2ad9644, 5b0b9d6): the headline, box numbers against
  Terra's words (matched at IoU 0.5 and 0.3, missed, ours unmatched, the matched-to-missed ratio), read numbers,
  paired label-level intervals.
- **Result.** Today's baseline 286/699 = 0.4092 [0.372, 0.446].
- **Verdict.** Done; the measure every later entry uses. **Lesson.** It came first, but it was then used on parts
  for most of the day instead of on the whole chain.

#### Task 2: the Mission Street review page

- **Built.** `wordread/review.py`, `scripts/wordread/review_page.py` (d627640..0473912). Sheet p07 (16,117 ft of
  Mission St). Red = ink called text; outlines = our boxes; green / purple = our reading agrees / differs from
  Terra's; yellow = Terra words we missed, with a reason.
- **Result.** On p07: Terra 1,106 words, ours 995, matched 447, missed 659. Miss reasons: merged with another word
  293, below the level 156, wrong size 116, outside the map 64, split 18, merged into a line 12. Published
  https://claude.ai/artifact/Ckx4CuXsFwgne4TsNkzJEN.
- **Verdict.** Done. **Lesson.** Merging words was the top cause from the first page, and stayed so.

#### Task 3: reader crops from our own boxes, sloppy crops, masking

- **Built.** `wordread/readercrops.py` and `textread` options (59b9cf7..38b5e4f; four fix rounds). Offsets of our
  boxes against Terra's measured (our trimmed boxes start 0.096 and end 0.104 word heights inside Terra's); line
  pieces and neighbours drawn from measured distributions; masking keeps every ink cluster touching the box.
- **Result.** The contact sheet looked like the real thing; masking removed no letter (0 of 1,434 words).
- **Verdict.** Done (input to Task 4). **Lesson.** Four review rounds went into matching distributions before any
  number said the crops would help.

#### Task 4: the reader experiments (item 3)

- **Built.** `scripts/wordread/run_reader.py` (92b0c40). Outputs `experiments/wordread/runs/{base,a}-ne-s{0,1}/`.
- **Measure.** NE's 285 code labels on today's boxes, two seeds.
- **Result.** Base 0.316 (mean of seeds), arm (a) 0.333; paired +0.0175 [-0.012, +0.048]. Reading at Terra's boxes
  was unchanged.
- **Verdict.** Not kept. Arms (b) (masking) and (a') never ran; the GPU went to the detector.
- **Lesson.** On boxes this bad (850 of 2,705 NE Terra words matched), reader training can't show.

#### Task 5: the detector's word heads

- **Built.** Off-by-default `heads="words"` on the U-Net: a word-core chance, box geometry, reading axis, a
  four-number word tag per pixel (`textnet/model.py`, `data.py`, `train.py`, `wordread/shrink.py`; 87f3d82..6d1af69,
  512a1dc, dadc5d1). The owner removed the picked core size: cores are the labelled rectangles (fd0e71e). Loss
  weights learned (Kendall uncertainty weights), class-balanced loss in place of fixed ratios (6d1af69).
- **Verdict.** Kept as the detector of every crude chain. Two faults were found and fixed (next two entries).

#### The sign-free axis (Task 8 finding, Task 5 fix 512a1dc)

- **Idea.** Test why the first word-head detector never learned reading direction.
- **Result.** The axis was often roughly right, but where it was right it pointed 180° wrong on 99.8% of pixels:
  under quarter-turn training the sign can't be learned, and its loss starved the axis. The fix learns the axis
  without a sign (as 2θ) and leaves which way up to the reader's 0/180 classifier. After it, every fold had 90-95% of
  held-out word pixels within 10° (median 1.1° on NE), against chance before.
- **Verdict.** Kept. **Lesson.** Test the cause first; the cheap check (axis error modulo 180°) found it.

#### The centre fix (Task 5 dadc5d1)

- **Idea.** Find why the head that votes for each word's centre learned nothing (correlation 0.04).
- **Result.** At 320-pixel training crops, 49.6% of core pixels belonged to words cut by the crop's edge, whose
  centre isn't visible; they carried 85.3% of the loss. Restricted to whole words it learned fast (correlation
  0.87). The fix gives cut words no geometry target. Retrained (af-ne-s0): centre loss 0.286 against 1.78, word length
  ratio 1.06 against 1.44.
- **Verdict.** Kept in the code. It did not move the crude chain (crude-map on the fixed detector's cores 140/285
  against 142).
- **Lesson.** The receptive field was not the cause (the first guess); the overfit test on 8 crops and the
  whole-word split found it.

#### Task 6: words from the vote mixture

- **Idea.** Each core pixel votes for a box and carries a tag; words are the components of a mixture over the
  votes, with measured spreads and a fitted per-word prior, no level, no merge rule.
- **Built.** `wordread/words.py`, `scripts/wordread/page_synthetic.py` (ac8c0f4..21f4f06; five fix rounds).
- **Result.** On a synthetic 1500x1000 page every painted word became exactly one box. On a real page the search was
  estimated at about 7 hours; the fitted spreads were degenerate (the dead centre head), and a draft formed 0
  words.
- **Verdict.** Parked: correct but too slow, and never run end to end.
- **Lesson.** The most principled formation step consumed five review rounds before anyone knew whether new boxes
  moved the headline. The crude chain answered that question in 3 minutes.

#### Task 7: doubt weights for Terra's words (item 6)

- **Idea.** Terra makes mistakes; count doubtful words less in training, with weights learned, not flags.
- **Built.** `wordread/doubt.py` (38b5e4f, 1d610d3). Terra read 27,219 words twice where its tiles overlap; the two
  reads agree 86.8% of the time, boxes at a median IoU of 0.88. A logistic model of agreement and a beta regression
  of box overlap give each word a weight (boxes 0.52-0.92, strings 0.49-0.99).
- **Result.** Held-out log-likelihood improved in every quarter. Detector arm (b) (b-ne-s0, b-ne-s1, b-se-s0) trained
  with the weights.
- **Verdict.** Built; never scored end to end. **Lesson.** Reusable for any training on Terra labels.

#### Task 8: the detector experiments

- **Built.** `scripts/wordread/run_detector.py`, `textnet/predict.py` word outputs, `scripts/textlines/variant.py`
  options (2e3964b, 84639d8, 29e0286, 0884d89, 658d390, 0a27ced, 2661bf3, e24aa90). The arms trained: (a) on all four
  folds and NE seed 1, (a') with the centre fix, (b) with doubt weights, base controls. Every crude chain, code
  spotting and learned links ran here (section 6.3). The report is `task-8-report.md` in the archive.
- **Verdict.** Stopped at the pivot. The keep rule's arms (a), (b), (c) were never compared end to end with Task 6's
  formation, because formation never ran.

#### Tasks 9 and 11: box choice (item 5)

- **Idea.** For each box, read its versions (as found, split, joined with the nearest neighbour, trimmed) and keep the
  best.
- **Built.** `wordread/choose.py` (edb6f33..91ad2a1; four fix rounds), `scripts/wordread/run_choose_pack.py`
  (10df504, ea969cf).
- **Result.** On today's NE boxes it failed three ways before it was understood: labels taken from rectangle overlap
  rewarded joining a neighbour (only 29% of "join" labels had a split word); soft labels by ink made it pick joins 64%
  of the time (-16 pt); near-copies of one version split their vote so a lone join won. The final form (each version
  scored on its own) gave -2.3 pt [-4.1, -0.4].
- **Verdict.** Not kept. It was to be retried with the box model on new boxes; never was.
- **Lesson.** A choice among near-duplicates needs a model of the duplicates; and box choice can't fix boxes that
  hold two words.

#### Tasks 10 and 11: the context pack (item 4)

- **Idea.** What a place knows (street names with centrelines, block numbers with outlines, the era's code list),
  given to the reader at read time as a prior that shifts probabilities and never filters. The 2000 table's code at a
  lot is never in it.
- **Built.** `wordread/context.py`, `scripts/wordread/build_pack.py` (cd2bf9b..be2acd9; five fix rounds). Pack:
  17,176 centrelines (2,713 streets), 5,302 blocks, 57 codes in 95 forms. Prior strength, escape mass and a read-time
  temperature fitted held out.
- **Result.** Held-out agreement with Terra 0.584 (ink alone) to 0.669 (with the pack). On today's NE boxes
  (5116bed): the headline -1.05 / -0.35 pt (seeds 0 / 1); street names read as Terra did 65 to 85 of 153, block
  numbers 373 to 432 of 478, codes 100 to 110 of 203. It also caught Terra misreads ("417B" read as "4178").
- **Verdict.** Not kept for codes (the headline didn't rise). Street and block gains noted. One concern parked: in
  small blocks the prior can override off-list ink (193 of 5,682 pairs).
- **Lesson.** The code list is flat, so it can't help codes; the pack helps the text the solver doesn't use yet.

#### Task 12: grey input and full resolution (item 7)

- **Built.** Code only (8b83a08, 6c8de6b): `SheetData.grey_bits`, `Recipe.input="grey"`, predict reads grey.
- **Verdict.** Never compared (it waited for a kept detector).

#### Task 13: map-like synthetic scenes (item 8)

- **Built.** `wordread/scenes.py`, `scripts/wordread/run_scenes.py` (a7debd6..cd9bc5f; three fix rounds). Scenes drawn
  from DataSF blocks, lots and streets at measured widths and inked shares, with real words pasted in, and exact masks.
- **Result.** Drawn line shares match the maps (2000 block edges 0.758 drawn, 0.755 measured); words per crop 11.6
  against 13.7 real. Gaps at corners remained.
- **Verdict.** Paused at "end to end first"; never trained with.

#### Task 14: the final model on every fold, the final page, the results note

- **Verdict.** Not reached. This note stands in for its results note.

#### Task 15: the learned box model

- **Idea.** The owner's: learn from a box's own pixels how many words it holds and where they divide, with no hand
  features.
- **Built.** `wordread/boxmodel.py`, `scripts/wordread/run_boxmodel.py` (7d2f553..36af02a, b27aa1c). Trained on all
  four folds with stacked held-out detector outputs.
- **Result.** On p07 it flags 108 of today's 123 merged boxes as holding two or more words; on held-out boxes 136 of
  180 real multi-word boxes, with 176 false flags among 6,107 single-word boxes. Word-count loss 0.48-0.58 against
  0.72-0.82 for a guess without pixels. Where the words divide: weak (about 45% found in NE and SE).
- **Verdict.** Built; never used in a chain. **Lesson.** It sees merges well; it was the missing signal for box
  choice, and never got its run.

### 6.3 End to end, 11-17 evening

After "you absolutely should have validated the ideas before spending 8 hours on them", the ruling was end to end
first: new boxes into today's reader into the headline, before any more polish. Every run below is in
`pipeline/data/solver/experiments/wordread/detector/runs/`.

#### crude-map

- **Idea.** The crudest chain: each patch the new detector calls more likely word core than not (even odds), boxed
  along its learned axis. No tags, no votes. A lower bound on what new boxes can do.
- **Built.** `run_detector.py crude` (658d390). 3 minutes for NE.
- **Result.** NE 142/285 = 49.8% against 92 (+17.5 pt [+11.3, +23.9]). All quarters 343/699 = 49.1% (+8.2 pt
  [+3.7, +12.6]). Outside NE, patches merged more words than today's boxes (no tags to split them). On p07 the owner's
  labels went 46 to 71 of 122. Page https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf (v1 NE, v2 whole sheet).
- **Verdict.** Kept as the base. **Lesson.** The first end-to-end number, after a day of parts, was the biggest gain
  of the day.

#### The tag split, crude-tags (proposed, rejected)

- **Idea.** Split each patch where the per-pixel word tags differ, by a measured tag spread.
- **Built.** `run_detector.py crude --tags` (committed); stopped before forming words, never scored.
- **Verdict.** Rejected by the owner: "It sounds like you're going to split them based on some kind of arbitrary
  cutoff. Shouldn't you instead learn that an individual word is an individual word?"
- **Lesson.** A split by a spread is still a decision by a number; the owner wanted the decision itself learned.

#### The same-word affinity

- **Idea.** A learned head: for each pixel and its right and lower neighbours, the chance they are one word. Words
  are pixels linked by more-likely-than-not links.
- **Built.** `heads="words_aff"` (7f67fca), crude-aff in `run_detector.py`, a Bayes prior correction (0a27ced).
  Report `affinity-report.md` in the archive.
- **Result.** Raw, it shattered NE into 373,503 boxes. With the measured prior correction: NE 144/285, +0.7 pt
  [-4.5, +5.7] over crude-map. Pairs of different touching words were called "same" 41-78% of the time: rare in
  training, so not learned.
- **Verdict.** Not kept. **Lesson.** Touching different words are a few hundred pairs among millions; a pixel-pair
  loss barely sees them.

#### Code spotting

- **Idea.** The owner's "detect straight from pixel": a head that names each pixel's code directly (57 codes, other
  text, background).
- **Built.** `heads="words_code"`, `wordread/codes.py`, `scripts/wordread/spot_page.py` (a483bc5).
- **Result.** NE 38/285 = 13.3%, -18.9 pt [-24.7, -12.5] against today. It kept other text apart from codes (94% of
  other-text pixels right) but couldn't name codes (19% on its own training words); 754 training code words, 18 codes
  never seen.
- **Verdict.** Not kept. **Lesson.** Naming a code needs the reading of a whole word; a pixel classifier with a few
  hundred examples per class can't learn it.

#### crude-kind

- **Idea.** Split crude-map's patches wherever the detector's own per-pixel kind (code, block number, street name,
  other) changes, then read each part. No numbers: each pixel takes the network's most likely kind.
- **Built.** `run_detector.py crude --kind map` (2661bf3).
- **Result.** 414/699 = 59.2% [55.5, 63.1]; +18.3 pt against today, +10.2 pt against crude-map, every quarter up.
  Merged boxes against Terra fell from 1,588 to 446; the cost is fragments where the kind flickers (80,834 boxes,
  mostly read as empty). The kind agrees with Terra's on 78% of word ink. On p07: 79 of 122. Page v3 at
  https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf.
- **Verdict.** The best version. Proposed next (not run): the reader retrained on crude-kind's boxes.
- **Lesson.** The second simple use of an output the detector already had gave the second big gain.

#### The review-page redesign

- **Idea.** The owner judges by looking at boxes on the map, on a phone; the page must be usable.
- **Built.** `wordread/review.py` rewritten (ae1d18e): a calm default (grey ink, boxes as thin outlines coloured by
  kind), layer chips, upright readings that avoid each other, tap for details in a bottom sheet, a navigator through
  the wrong labels with their causes, versions as a segmented control. Numbers byte-identical. Published v4 at
  https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf; learned-links later added as a fourth version.
- **Verdict.** Done. **Lesson.** Every result needs its map page, and the page must work on a phone (memory
  `show-each-step-on-page`).

#### Learned links and the boundary diagnostic

- **Idea.** The owner: "'agrees' sounds like something that should be learned". A small network (MLP) on pairs of
  neighbouring core pixels decides "same word", from all 16 detector outputs of both pixels; words are the linked
  components.
- **Built.** `scripts/wordread/links.py`, `run_detector.py crude --links`, `tests/test_wordread_links.py` (e24aa90).
  Each fold's MLP trained on the other three quarters (37.95 M same pairs, 2.36 M different, 120,820 of them between
  two words). Outputs `experiments/wordread/detector/links/`.
- **Result.** Held out, the MLP called same pairs right about 79% and different pairs about 85% (balanced). With the
  measured prior (+2.8 log odds) it cut only 1-2% of word-word pairs. End to end 351/699 = 50.2%, -9.0 pt [-12.0,
  -6.1] against crude-kind; merged stayed the top cause (221 of 348 wrong).
- **The boundary diagnostic** (the controller's question: is the evidence there if summed along a whole boundary
  instead of decided pair by pair?). 866 boundaries between touching, merged Terra words against 9,752 same-word
  halves. Summed evidence separates them only weakly: AUC 0.695 (0.766 length-matched); for the pairs that matter,
  code+block 0.63, code+street 0.67, code+code 0.58. And 2,536 more merged pairs touch only through core pixels that
  belong to no word, which no word-to-word boundary can see.
- **Verdict.** Not adopted; region merging on these links is not promising.
- **Lesson.** One surviving link merges two words, so a per-pair call is the wrong aggregation. And much of the
  merging runs through non-word ink, which is a detector problem, not a grouping one.

## 7. Why the best version still gets labels wrong: the five-way causes

Each wrong label is put in one cause by Terra's words and their ink inside the box at the label's point
(`scripts/wordread/label_causes.py`; a box "holds" a word when more than half its ink is inside). Results
`pipeline/data/solver/experiments/wordread/causes/`; page https://claude.ai/artifact/SC6Ut8UjjpNSVHR6iBANLk.

| Cause | Today | Map | Kind | Links |
|---|---|---|---|---|
| No box | 78 | 14 | 11 | 14 |
| Merged | 206 | 227 | **123** | 221 |
| Partial | 35 | 3 | 21 | 3 |
| Clean misread | 26 | 39 | 55 | 36 |
| Other | 68 | 73 | 75 | 74 |
| **Wrong in all** | 413 | 356 | **285** | 348 |

Map is crude-map, Kind is crude-kind, Links is learned-links. The causes:
- **No box:** no box holds the label's point.
- **Merged:** the box holds the code word and at least one other word.
- **Partial:** the box holds half the code word's ink or less.
- **Clean misread:** the box holds the code word alone and matches Terra's box; the reader got it wrong.
- **Other:** mostly a loose box (it holds the code word alone but matches Terra's box at IoU under 0.5); also a box
  holding one other word, or no Terra word at the point.

crude-kind in detail:
- **Merged (123):** code with a block number 74 (25 along the code, 49 stacked above or below it), code with another
  code 28, code with a street name 16, code with other text 1, the box holds two other words 4.
- **Partial (21):** where the kind flickers inside a code.
- **Clean misread (55):** 36 of them in NE; Terra's own reading is wrong too in 9.
- **Other (75):** a loose box holding the code alone (IoU under 0.5) 61; no Terra word at the point 14.
- **Against crude-map:** 94 labels fixed (69 of them were merges), 23 broken.

So the next levers were the merges (mostly code with block number), then the reader on our own boxes (clean
misreads, loose and partial boxes).

## 8. The analysis (the controller's, given to the owner on 11-18)

- **The real gains came from the simplest uses of the new detector.** Today's 40.9% went to 49.1% with crude
  components and 59.2% with the kind split.
- **Six more elaborate add-ons all failed:** the affinity, code spotting, box choice, the vote mixture, the context
  pack and learned links. Merged boxes stayed the biggest cause (123 of 285).
- **The core mistake: a stand-in score was optimised for two days, and the map itself was never measured with our
  own reads.**
  - The stand-in was reading at our boxes on 699 labels.
  - T4 was put on hold at 46% on the assumption that this wasn't good enough, without testing it.
  - Stretching S1's costs (measured at a few points of error) to about 40 points of error gives roughly 22,000 to
    50,000 wrong lots (4,543 plus 40 points times about 430 to 1,130 a point). That range straddles the published
    map's 35,455, so whether we were already shippable is unknown.
- **Options weighed:**
  1. **More box work.** Stop until the map says boxes matter.
  2. **A "CNN pivot".** The detector already is a CNN (a U-Net). A real pivot means a proven detector's word
     grouping.
  3. **Ship with the saved reads.** 35,455 to about 4,543 wrong lots, but it reverses the 11-16 decisions.
  4. **Measure the map with our own reads** (the recommendation): T4 with crude-kind, T5's publish wiring alongside.
- **The owner chose 3:** "pivot to shipping with the saved reads from Terra and Sol. I want to get the published map
  updated." The research is parked, not abandoned.

**A cross-check from the records (added while writing this note).** S1's per-point costs were fitted over misread
rates of 2-20% and miss rates of 5-40%, on the old search (7,100 wrong lots before the compound move), so the range
above rests on measured points, not only on stretching. S1's own grid reached 26,878 wrong lots at 40% misses and
32,467 at 20% confident misreads, both inside 22,000-50,000. S0 points the other way: a reader with 9% misreads, at
the right label positions, gave wrong lots not distinguishable from the saved reads. Neither spike measured the kind
of error our chain makes (labels merged or lost at the box step, which the solve sees as a missing or wrong print),
and neither ran under the compound search. So the conclusion stands: the map with our reads was never measured, and
it could land on either side of the published map.

## 9. The owner's working rules this research taught

Each rule is in memory or the decisions log; this list points to them and to the moment that taught it.

- **No hand-set cutoffs** (memory `probabilities-not-cutoffs`). Taught on 11-17 at 10:35 ("I KEEP TELLING YOU..."),
  again for the core size (11:10) and the tag split (18:35). The spec had reworded the rule as "thresholds are
  fitted or measured", which let measured cutoffs in; fixed in 26985e3.
- **Validate end to end first** (memory `principled-loop`; also `test-the-cause-first`). Taught on 11-17 at 17:20
  ("you absolutely should have validated the ideas before spending 8 hours on them"), and on 11-16 at 23:20 ("Did
  you reach for 'likely cause' without validating your hunch?"). Quality before reproducibility: 11-16 at 10:10
  (decisions log).
- **Show results on usable map pages** (memory `show-each-step-on-page`, `pictures-need-a-purpose`). Taught on 11-17 at
  17:10, 22:45 and 22:50 ("Bro this is unusable"). Publish intermediate results as they land.
- **No LLM at inference** (memory `deterministic-no-llm-answers`, `terra-outputs-as-training-data`). Set on 11-16;
  set aside for this release on 11-18 (decisions log).
- Also from this stretch: weigh a large cost gap plainly (`weigh-cost-in-near-ties`, S2); a blank answer is not a
  choice (`blank-answer-is-not-a-choice`, T1 at 22:37); a correction of a proposal is not a go
  (`wait-for-go-ahead`).

The decisions log (`decisions-log` in memory) holds every turn in order.

## 10. If we resume

In this order:

1. **Measure the map first (T4).** Our reads (crude-kind's boxes and the text reader) as the solve's labels, then the
   solve, then the six measures and wrong lots, against the published 35,455 and the saved reads' 4,543. This needs
   T4's wiring from `docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md` (boxes to prints,
   readings, calibration). If the map is already better than the published one, the owner's "better, not perfect"
   applies. Only the map's own errors should pick what to work on next.
2. **A calibrated reader.** S1: a confident misread costs about 1,130 wrong lots a point, a miss about 430, a
   misread at low sureness about 460. A reader that is honest about doubt may help the map as much as one that reads
   more. The CTC reader is overconfident: Task 10's held-out fits softened its ink scores (a power of about 0.6-0.7)
   and still needed a read-time temperature of about 2.2-2.9 on the combined posterior.
3. **The reader trained on our own boxes.** Retrain per fold on crude-kind's boxes (Task 4's runner, at natural
   counts). It targets the clean misreads (55), loose boxes (61), partial boxes (21) and some merges.
4. **A proven detector, only if the map shows boxes are the bottleneck.** For example DBNet in docTR (Apache-2.0,
   PyTorch, mps and CPU), fine-tuned on the map. Its word grouping is what our crude chains lack.
5. **The boundary diagnostic's verdict.** Summed link evidence separates merged words only weakly (AUC 0.70), and
   2,536 merged pairs are joined through non-word ink. More grouping on our current outputs is not the next step.

Parked, with their state: Task 4 arms (b) and (a'); Task 12's comparison; Task 13's scenes (corner gaps); Task 6's
speed; Task 10's small-block concern; Task 14; the text-vs-line dashed-line over-call; the stages left stale on
paper by the text-vs-line plan (`solver-text`, `solver-text-scores`, `solver-text-reader`, `solver-boxes`,
`solver-text-crops`, `solver-text-read`; byte-identical under the defaults).

## 11. Where every output lives, and what is reusable

### Records (archived with this note)

`docs/research/joint-solver/notes/2026-11-18-text-reader-records/` holds the git-ignored working records, copied as
they were on 11-18:

| Folder or file | What it is |
|---|---|
| `2026-11-15-code-reader/` | The template reader's ledger, briefs, reports and review notes |
| `2026-11-15-text-detector-training/` | The speed recipe's ledger, trials and reviews |
| `2026-11-16-code-reader-read-along-the-word/` | R1's ledger and report |
| `2026-11-16-labels-from-the-scan/` | The red team (`redteam.md`) and the library survey (`survey.md`) |
| `2026-11-16-text-reader-labels-and-publish/` | T0-T3's ledger, reports and reviews |
| `2026-11-17-text-detector-text-vs-line/` | The text-vs-line ledger (from `experiments/textlines/sdd-records/`) |
| `2026-11-17-map-text-fields-way/` | The fields-way ledger (`progress.md`), every task's brief and report, `affinity-report.md`, review notes |
| `redteam-text-vs-line-design.md`, `redteam-text-vs-line-plan.md` | The red teams of the text-vs-line design and plan |

The `review-*.diff` files in them are the code packages the reviewers read; each is a `git diff` of the commits
named in its file name.

### Notes, specs and plans (in git)

- Notes in `docs/research/joint-solver/notes/`: `2026-11-16-search-move-and-reader-spikes.md`,
  `2026-11-16-text-reader-survey.md`, `2026-11-16-text-target-root-cause.md`,
  `2026-11-16-text-graphics-separation-literature.md`, `2026-11-17-text-vs-line-results.md`,
  `2026-11-17-our-text-pipeline-trace.md`, `2026-11-17-text-detectors-traced.md`,
  `2026-11-17-map-text-approaches-traced.md`.
- Specs and plans in `docs/superpowers/`: code reader, text detector training, read along the word, labels from the
  scan (`2026-11-16-text-pipeline-from-the-scan-design.md`), search move and reader spikes, text reader labels and
  publish, text detector text vs line, map text fields way.

### Results (not in git; under `pipeline/data/solver/experiments/`)

| Path | What |
|---|---|
| `wordread/detector/runs/` | Every detector run: `a-*-s0`, `a-ne-s1`, `af-ne-s0`, `aff-ne-s0`, `b-*`, `code-ne-s0` (models, chance, words), and the chains `crude-map-all-s0`, `crude-kind-map-all-s0`, `crude-links-all-s0`, `crude-aff-*` (boxes, reads, `vs_today.json` and paired comparisons) |
| `wordread/detector/links/` | Learned links: models, pair counts, `boundaries.json` |
| `wordread/causes/` | The five-way causes: `causes-<run>.json`, `labels-<run>.parquet`, crops |
| `wordread/review/` | Review pages: `crude-kind/review.html` (four versions), `crude-map/`, `baseline/`, `preview-heads/`, `pack-prelim/`, `choose-prelim/`, `code-spotting/` |
| `wordread/runs/` | Task 4's reader runs |
| `wordread/packs/`, `wordread/choose*/`, `wordread/boxmodel/`, `wordread/scenes/`, `wordread/reader/` | Tasks 10, 9/11, 15, 13, 3 |
| `wordread/pages/progress/` | The progress page's source (`progress.src.html`, `inline.py`, the built page), copied from the scratchpad |
| `wordread/writeup/index.html` | This note as a phone-readable page |
| `textlines/` | Text vs line: runs, baselines, panels, `sdd-records/` |
| `spike-s0/`, `spike-s1/`, `spike-s2/`, `spike-s3/`, `spike-heights/`, `detector-overcall/` | The spikes |
| `text-reader-plan/final-before/` | The solve before T1 (the saved-reads baseline, 4,543 wrong lots) |

Stage outputs: `pipeline/data/solver/text/` (`solver-text`), `text-reader/` (the four held-out detectors),
`boxes/`, `text-crops/`, `text-train/`, `text-read/` (the text reader).

### Published pages

- Mission review, crude-kind with four versions: https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf
- Causes: https://claude.ai/artifact/SC6Ut8UjjpNSVHR6iBANLk
- Progress: https://claude.ai/artifact/PnbodJhmTTyTJjPKXgKzfE
- Comparison report: https://claude.ai/artifact/BAGwr4yy8Bfsf1pqWVvVhC
- Text-vs-line morning report: https://claude.ai/artifact/9GPpC6Y7ZvE3LAH7AyVNZU
- Baseline review (today's boxes): https://claude.ai/artifact/Ckx4CuXsFwgne4TsNkzJEN
- Word-heads preview: https://claude.ai/artifact/SqpLkTJ1rRtfR2Uwt2RH9g; pack preliminary:
  https://claude.ai/artifact/VQra75hKcAEdMJBquyy6x3; the agreed mockup:
  https://claude.ai/artifact/2ytdcmDwbya7t72JMTxg5F

### What is reusable

- **The end-to-end measures and the causes** (`wordread/measures.py`, `scripts/wordread/label_causes.py`): any future
  chain can be scored the same way in minutes.
- **The review page** (`wordread/review.py`, `scripts/wordread/review_page.py`): phone-usable, versions side by side.
- **The text reader** (`textread/`, stages `solver-text-crops`, `-train`, `-read`): 96% at good boxes, fast,
  deterministic.
- **The detector's word heads** (`textnet` `heads="words"`, with the sign-free axis and the centre fix) and the
  four held-out `a-*-s0` detectors.
- **The crude-kind chain** (`scripts/wordread/run_detector.py crude --kind map`): the best boxes we have, from
  outputs we already compute.
- **Doubt weights** (`wordread/doubt.py`): an error model of Terra itself, for any training on Terra labels.
- **The context pack** (`wordread/context.py`, `scripts/wordread/build_pack.py`): better street names and block
  numbers, if the solver ever uses them (for example for the warp's skewing).
- **The box model** (`wordread/boxmodel.py`): flags merged boxes well.
- **The S1 harness** (`scripts/spikes/s1_degraded.py`): what reader error costs the map; worth re-running under the
  compound search.
- **The text-vs-line gate** (`textnet/linegate.py`, `scripts/textlines/`), the synthetic scenes
  (`wordread/scenes.py`) and the template reader's glyphs (`codereader/glyphs.py`, as a crop generator).
