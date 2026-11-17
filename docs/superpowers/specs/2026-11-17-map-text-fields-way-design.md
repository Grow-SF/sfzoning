# Finding and reading map text the way the field does: design (2026-11-17)

## Why

The comparison report (`https://claude.ai/artifact/BAGwr4yy8Bfsf1pqWVvVhC`; notes
`docs/research/joint-solver/notes/2026-11-17-{our-text-pipeline-trace,text-detectors-traced,map-text-approaches-traced}.md`)
found:
- **We build words from connected ink.** Every strong published method forms them from shrunk word cores, box votes or
  learned word objects.
- **The box step costs us about 49 points.** Our reader reads 96% of the owner's answered labels right at Terra's
  boxes but 48% at ours, on the same labels.
- **The reader has only ever practised on good boxes.**
- **The map's geography is unused,** though it is a better prior than any published map system had.

The owner asked for an approach that implements and tests all eight items in the report.

## What the owner said

- "I would like to train a general model so this can be applied to other old maps. But my primary interest is San
  Francisco and launching this map. So maybe these are optional things that are supplied when the model is reading
  the map so that it can reference data to alter its priors or posteriors."
- **Agreed playback:**
  - The detector and reader learn only what old maps look like, trained across every era we hold.
  - What a place knows arrives at read time as an optional context pack: street names with centrelines, block numbers
    with outlines, and the codes of that year. It shifts probabilities, never filters, and without a pack the reader
    reads the ink alone.
  - The 2000 zoning table's code at a lot is never in the pack for the 2000 map: that is the answer key.
  - San Francisco first; the general part is tested on maps the model never trained on.
- **How the owner wants to judge it:** the text reader's output on one full sheet, the one along Mission Street, as
  one page. Agreed against a mockup, `https://claude.ai/artifact/2ytdcmDwbya7t72JMTxg5F`.
  - **Red:** the ink px the detector calls text. The owner confirmed red means "text", not "ink". Grey is other ink.
  - **Our boxes:** every detected box, with its reading direction and the metadata found for it.
  - **Readings against Terra:**
    - green when our reading matches Terra's read of that word;
    - purple when it doesn't, with Terra's reading shown beside it in red.
  - **Green shading:** text we found that Terra missed. The shading never covers the red ink.
  - **Yellow shading:** boxes Terra saw that we missed, with debug: how close to certain a box there was (e.g. "a 38%
    chance there was a box here"), and why it didn't register.
  - **Top-line numbers, for boxes:**
    - the number of boxes Terra saw;
    - the number we detected;
    - the number we missed;
    - the ratio of shared boxes to the ones we didn't detect;
    - an improvement number against Terra or the last run.
  - **The same top-line breakdown for the text read.**
- The owner said "Continue" to the anti-goal question. The anti-goals below are the controller's, drawn from the
  owner's standing rules.

## Goals

1. **Raise "read right at our own boxes"** on the owner's answered labels, toward the 96% the reader gets at Terra's
   boxes. A label with no box counts as wrong. This is the headline measure.
   - Today: **41.1% over all 719 labels**.
   - It is 46.2% on the 633 labels that get a box, and 47.7% on the 595 that also have a Terra box.
2. **Implement and measure all eight items** of the report, each kept or dropped by a rule written before its runs:
   1. words from word cores and box votes, not connected ink;
   2. separate jobs: a per-ink line head for the walls, word-area heads for finding words;
   3. the reader trained on our own boxes and sloppy crops, and masked outside the word;
   4. context packs: geography narrows street names and block numbers, and codes stay on the era list;
   5. the reader picks between box versions;
   6. LLM labels cleaned: doubtful words left out of the loss;
   7. grey input and full resolution tested against 1-bit at half size;
   8. map-like synthetic scenes built from DataSF geometry.
3. **A general model:** trained across every era we hold, with no San Francisco knowledge in the weights. It is also
   scored on the test drawings (1972, 1975, 1980), which were never trained on.
4. **The Mission Street review page,** as specified above, built after the baseline and after every kept step.

## Non-goals

- Wiring readings into the solve's labels, and publishing (T4/T5 of the text-reader plan stay on hold for the owner).
- Heights (a later step: its forms join the reader's vocabulary).
- Context packs for other cities. The interface is general; only San Francisco's pack is built.
- Rebuilding the shipped stages. Adoption is the owner's call at the end, as before.

## Anti-goals (the controller's, from the owner's standing rules)

- **The answer key in a prior:** the 2000 table's code at a lot in any 2000 prior, or any measure that reads the same
  rule it trains on.
- **A parameter guessed or taken from a maximum:** prior strengths and thresholds are fitted on held-out quarters or
  measured.
- **A verdict from one seed, from wrong-lot counts, or changed after seeing the results.**
- **A claim with no picture behind it:** the review page and the panel go with every kept step.
- **LLM output at inference; GPL-family code.**

## Design

### 0. Measures and the review page (built first; baseline today)
- **The headline:** read right at our boxes over all 719 answered labels, with a missed box counted wrong, and an
  interval. Also reported split by box found or missed.
- **Box numbers** against Terra's words, per quarter and on the Mission sheet:
  - matched at IoU 0.5, and at 0.3;
  - missed;
  - ours unmatched, split into "read as text" and "read as none";
  - the matched-to-missed ratio, and the change against the previous run.
- **Read numbers:** on matched boxes, same as Terra or different; zoning codes right against the owner's answers.
- **The review page (an Artifact, updated in place each run):**
  - **The sheet:** the 2000 sheet holding the longest stretch of Mission Street (the controller's call: find it from
    the DataSF centreline and the warp), at full resolution with pan and zoom, plus the agreed overlays.
  - **Our unmatched boxes:** those read as "none" are drawn as thin grey outlines (junk, counted apart). Those read as
    text get the green shade. (The controller's call; the owner asked for green on "text Terra missed".)
  - **Yellow debug** comes from the detector's own outputs at Terra's box: the highest word-core chance, the share of
    its ink above the level, which component swallowed it (with that component's size), and the reader's reading at
    Terra's box. It also gives one reason code:
    - below level;
    - merged into a line;
    - split;
    - wrong size;
    - outside the mapped area.
  - **Tapping a box** shows its kind probabilities, angle, chance, top three readings, Terra's reading and the IoU.
  - **The top-line panels,** as in the mockup.
- **The text-versus-line gate** (`textnet/linegate.py`) stays the measure for the walls' per-ink head.

### 1. The reader on our own boxes (no detector change)
- **Training crops:**
  - our detector's boxes matched to Terra words, carrying Terra's string as the target, ABCNet v2 style;
  - plus deliberately sloppy crops of Terra's words: jittered ends, angle and height, line pieces through the letters,
    fragments of neighbouring words (Weinman's synthesizer).
- **Masking:** px outside the predicted word (its core, grown back) are blanked before reading (Mask TextSpotter v3).
- **Held out:** by quarter as now; the answered blocks are never trained on.

### 2. The detector, with separate jobs
- **One U-Net** (today's), with four heads:
  - **(a) per-ink text vs line** for the walls, trained as today with the best measured label rule;
  - **(b) word core:** each Terra rectangle shrunk inward (DBNet's D = A(1−r²)/L; r measured on our word sizes, not
    assumed), over every px, paper included;
  - **(c) box geometry** at core px: the distances to the word's four sides and its reading angle, with semantic edges
    (the top of the letters, where reading starts), as in EAST and MapTD;
  - **(d) a 4-number word tag per px** (PAN), so that touching words can be told apart.
- **Words** are core components grown back out (unclip), with each grown word refined by its geometry votes
  (locality-aware NMS). Both forms are built; the measures pick between them.
- **Item 6, label cleaning:** each Terra word gets a measured doubt score. A word whose box holds ink running far
  outside it, a forced-height slant box, or a word the other passes contradict counts less in the loss, or not at all.
  The rule and its counts are reported, as DBNet's ignore mask does. A CRAFT-style check of letter count against
  spelling, if the builder finds a sound letter count.

### 3. Context packs and choosing the box
- **A context pack** (optional; one file per map) holds:
  - street names with centrelines;
  - block numbers with outlines;
  - the era's code list.

  All in map coordinates, carried to the sheet by the warp. San Francisco's pack is built from DataSF. A pack for the
  2000 map never holds the zoning table's lot codes.
- **The reader's posterior:** p(string | ink) × p(string | kind, place), with an escape mass for strings off the
  list. Today's closed code list becomes the pack's era list. The prior's strength and the escape mass are fitted on
  held-out quarters.
- **Choosing the box:** for each detected word, try candidate boxes (as found, split at the widest gap, joined with its
  nearest neighbour, each end trimmed, both directions). Keep the one with the highest posterior. Report how often
  each candidate wins.
- **Tested** with and without the pack, on held-out quarters and on a test drawing, whose streets are mostly today's.

### 4. Input and resolution (item 7)
- **One controlled comparison** of grey against 1-bit input, and of full against half resolution, on the held-out
  quarter, after step 2.

### 5. Map-like synthetic scenes (item 8)
- **Rendered from real DataSF geometry** at a sheet's scale:
  - lots, blocks and street edges as 1-bit lines at the measured widths;
  - block numbers at block centres;
  - street names along centrelines;
  - codes inside lots, placed as the print places them (measured from Terra's boxes).
- **Letters** are cut from real clean words (real texture), with real line pieces over them.
- **Exact masks for every head.**
- **Tested** as an addition to real crops, against the step-2 model.

### Testing (every item)
- **Each item is an ablation** against the best so far, trained with the same held-out quarter and seeds (at least
  two), and scored on the headline and the box and read numbers, with paired block-bootstrap intervals.
- **The keep rule is written before each item's runs.** Better on the headline beyond noise, with no loss beyond noise
  on the other measures, and the review page agreeing by eye.
- **Order:** 0 → 1 → 2 → 3 → 4 → 5. Each step builds on the kept form of the last. A dropped item is written up with
  its numbers and picture.
- **The final model** is scored on all four quarters and on the test drawings.

## Constraints

- **$0:** no LLM calls.
- **One GPU (mps) job at a time.** Load near 20.
- **Deterministic and seeded.**
- **Held out by quarter;** the owner's answered blocks are never trained on.
- **Apache-2.0:** our own code. MMOCR and docTR (Apache-2.0) may be read and borrowed from. Nothing from AdelaiDet,
  DeepSolo, MapTD (GPL) or mapKurator.
- **Training budget per run:** minutes on the Mac's GPU (today 6-7 min for a detector, about 20 min for a reader).
- **No commits while an `sfzmaps build` runs.** No stage is rebuilt without the owner's go.
- **Code** in `pipeline/src/sfzmaps/solver/`; experiments under `pipeline/data/solver/experiments/`. Don't edit
  `paths.py`.

## The builder's questions, answered (the controller's calls)

1. **Folds and seeds.**
   - Each item is first screened on one fold, NE (the hardest; 302 of 731 answered labels), at two seeds, with paired
     intervals on NE's labels.
   - A kept item is then confirmed on all four folds at one seed, so the headline is over all 719 labels.
   - The final model runs all four folds at two seeds.
2. **The reader follows the detector.**
   - Whenever the detector changes, the reader is retrained on the new detector's boxes. A detector item is judged
     end to end (detector plus retrained reader), and the walls' gate (below) also binds.
   - Box choice (item 5) and context packs (item 4) get separate keep rules. Box choice comes first, scored with the
     ink-only posterior; packs are added after.
3. **The review page is honest.** Every px and box on it comes from the fold model that held out that px's quarter,
   as `solver-text-reader` does. The page says which models drew it.
4. **The test drawings** (1972, 1975, 1980) have no owner answers. There the numbers are "agreement with Terra" (boxes
   and strings), named as agreement, never "right".
   - Their context pack is today's DataSF streets and blocks. Its mismatch with old names is reported, not hidden.
5. **Matching is fixed across runs.** An answered label belongs to the box that holds its point (T3's
   `text-read/measures.json` rule). When several boxes hold it, the label goes to the highest-posterior box. A merged
   box that holds several labels is read once and scored against each.
6. **The prior is fitted without a leak.** A pack's strength and escape mass are fitted on the other three quarters,
   on Terra-matched words (not the owner's answers), and scored on the held-out quarter's answers.
   - Block numbers in the pack are meant to be near look-ups. They are context, not what is scored.
   - "Never filters" is checked by a test: an off-list string with clear ink wins over the list.
7. **The walls' head is trained jointly** with the new heads. Step 2's keep rule also requires no loss beyond noise
   on the gate's measures for that head (m1a in box, m2, clean words, m1c).

## Left to the builder

- Module layout, file formats, and the review page's rendering (one large image plus SVG overlays, or tiles).
- How the doubt score is computed, within "measured, reported".
- The exact candidate set for box choice.
