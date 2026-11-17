# Map Text the Field's Way: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise "read right at our own boxes" on the owner's 719 answered labels (41.1% today) by implementing and measuring the eight items of the comparison report. Judge each by a rule written before its runs and by the Mission Street review page.

**Architecture:**
- A new package `solver/wordread/` holds:
  - the end-to-end measures and the review page's data;
  - word formation from cores and votes;
  - box choice;
  - context packs;
  - synthetic scenes.
- The detector (`textnet`) gains word heads and doubt weights behind off-by-default recipe fields.
- The reader (`textread`) gains own-box, sloppy and masked crops behind off-by-default options.
- Experiments run one fold at a time through `scripts/wordread/`, and are decided by `scripts/textlines/decide.py`'s
  paired machinery, extended to label-level pairs.

**Tech Stack:** Python 3, numpy, OpenCV, PyTorch (mps), shapely/pyproj if already in the lock (check `uv.lock`; add nothing GPL), pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-map-text-fields-way-design.md`, including its "builder's questions, answered" section.
Background:
- `docs/research/joint-solver/notes/2026-11-17-our-text-pipeline-trace.md` (our steps, with file:line);
- `…/2026-11-17-text-detectors-traced.md` and `…/2026-11-17-map-text-approaches-traced.md` (the methods).

## Global Constraints

- **No hand-set cutoffs (the owner's standing rule; reviewers treat a breach as Important):**
  - nothing that decides is a fixed number: no match, split, join, filter, cap, search window, flag or in/out rule;
  - a percentile (q99, "the 99th") or a value read off a measured histogram and used as a cut is STILL a cutoff;
  - a published method's constant (PAN's 3, DBNet's 0.4, a 1% rule) is not exempt;
  - model each decision as a probability: a likelihood with measured spreads, a fitted logit or prior, a per-item
    weight, or a draw matching a whole measured distribution;
  - a setting that can't be a probability (e.g. a shrink ratio) is screened by the held-out outcome;
  - hard rules only for the physically impossible;
  - report-only measures (IoU 0.5 / 0.3 for the top-line, the review page's reason labels) are display, not decisions;
  - if this brief's text gives a cutoff, stop and report it, don't build it.
- **$0:** no LLM calls. Never print, log, copy or commit `pipeline/.env`, `site/.env.local` or any token.
- **One GPU (mps) process on the machine at a time:** every training or prediction inside `gpu.gpu_turn(True, device)`.
  Load near 20: ≤ 4 workers × 1 thread. Don't run pytest while a training runs (the session takes a GPU turn).
- **Deterministic:** seeded. New rngs are derived from the run seed (`np.random.default_rng([seed, k])`).
- **Held out:** by quarter (`data.Quarter`). The owner's answered blocks (`reader_stage.held_blocks()`) are never
  trained on, never used for composites, scenes or banks, and never used to fit a prior.
- **No answer key in a prior:** the 2000 zoning table's code at a lot is never in a 2000 context pack, and no prior is
  fitted on the owner's answers.
- **Defaults stay byte-identical:** every new recipe or reader option defaults off, so today's training and reading
  are unchanged. The pinned-batch tests and `GUARD_WEIGHTS` must keep passing.
- **No import-closure change for the existing stages:**
  - don't import new modules from `textnet/data.py`, `train.py`, `textread/*` stage paths, `wordboxes.py`;
  - pass new behaviour in as objects or options;
  - check with `uv run pytest -q tests/test_build.py`.
- **Files not to touch:** `build.py`, `dvc.yaml`, `dvc.lock`, `paths.py`, `cli.py`, `pyproject.toml`, `uv.lock`.
  No stage is rebuilt.
- **No commit and no new or edited file under `pipeline/src`, `inputs`, `dvc.yaml`, `pyproject.toml` or `uv.lock`
  while an `sfzmaps build` runs:** `pgrep -fl 'sfzmaps (build|stage)'` must be empty.
- **Experiments:**
  - outputs under `pipeline/data/solver/experiments/wordread/`, never committed;
  - each item is screened on fold NE at seeds 0 and 1, and a kept item is confirmed on all four folds at seed 0;
  - the keep rule is written in the task before its runs;
  - no verdict from one seed, from wrong-lot counts, or changed after the results.
- **The headline:** read right at our boxes over all answered labels of the scored folds, with a missed box counted
  wrong. Matching is T3's rule (`textread/read.py` `holding_box`): the box holding the label's point, and the
  highest-posterior box when several hold it.
- **Licences:** Apache-2.0 code only. Ideas from MMOCR or docTR may be re-expressed; nothing copied from AdelaiDet,
  DeepSolo, MapTD (GPL) or mapKurator.
- **Tests:** synthetic arrays in tests. Real data are injected; a real-data check skips without `pipeline/data`.
  Iterate set:
  `uv run pytest -q tests/test_wordread_*.py tests/test_solver_textnet.py tests/test_solver_textread.py tests/test_build.py`.
  Before a commit: `uv run pytest -q`.
- **Git:** `git add <paths>` and `git commit` as separate plain commands. Messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A label point held by two boxes, or by none:** the headline counts it once, by T3's rule, and a missed box is
   wrong, never dropped (Task 1 test).
2. **A Terra word split across two of our boxes, or two Terra words in one of ours:** the box numbers count each Terra
   word once (matched, or missed with reason "split" / "merged"). A merged box counts once as matched, to the
   best-IoU word (Task 1 test).
3. **A context pack with no entry near a box, or a kind with no list:** the posterior falls back to the ink alone
   (escape mass), never to zero probability (Task 10 test).
4. **Word formation on a core that touches the tile edge or the page edge:** boxes are clipped, never wrapped or
   dropped (Task 6 test).
5. **New recipe or reader options left at their defaults:** batches, weights and reads are byte-identical to today
   (Tasks 3, 5 and 12 tests).

---

### Task 1: The end-to-end measures (`wordread/measures.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/__init__.py`, `measures.py`; Test
`pipeline/tests/test_wordread_measures.py`.

**Interfaces:**
- **Consumes:**
  - `textread/read.py`: `holding_box`, `label_scores`, `CodeList`, `boot_interval`;
  - `wordboxes.py`: `poly_area`, `pairs`;
  - `textnet/boxes.py`: `terra_boxes`.
- **Produces:**
  - `headline(boxes: pd.DataFrame, reads: dict, answers: pd.DataFrame, quarters: tuple[str, ...]) -> dict`:
    - keys `n`, `right`, `share`, `ci`, `with_box`, `right_with_box`, and `by_quarter`;
    - `reads` maps a box row to its posterior, as `text-read` stores it;
    - `answers` is the owner's labels, with their sheet, point, truth and quarter.
  - `box_numbers(boxes, terra: pd.DataFrame, reads, iou=(0.5, 0.3)) -> dict`, with keys:
    - `terra`, `ours`, `matched`, `missed`, `ours_unmatched_text`, `ours_unmatched_none`, `matched_to_missed`;
    - `missed_reasons` (a counter of the reason codes from Task 2);
    - per IoU and per quarter.
  - `read_numbers(boxes, terra, reads) -> dict`: on the matched boxes, `same_as_terra`, `differs`, and
    `codes_right_vs_answers`.
  - `compare(now: dict, before: dict) -> dict`: the differences, for "vs last run".
  - `paired_labels(a: pd.DataFrame, b: pd.DataFrame, boot=2000, seed=0) -> dict`:
    - a and b give right or wrong per answered label for two arms;
    - the bootstrap resamples blocks (clusters of labels);
    - returns `diff`, `lo` and `hi`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_label_with_no_box_counts_wrong():
    r = measures.headline(BOXES_ONE, READS_ONE, ANSWERS_TWO_ONE_UNBOXED, ("ne",))
    assert r["n"] == 2 and r["right"] == 1 and r["with_box"] == 1

def test_a_point_in_two_boxes_goes_to_the_higher_posterior_box():
    r = measures.headline(BOXES_OVERLAP, READS_OVERLAP_SECOND_RIGHT, ANSWERS_ONE, ("ne",))
    assert r["right"] == 1

def test_a_terra_word_split_over_two_boxes_is_missed_with_reason_split():
    b = measures.box_numbers(BOXES_SPLIT, TERRA_ONE, READS_SPLIT)
    assert b[0.5]["missed"] == 1 and b[0.5]["missed_reasons"]["split"] == 1

def test_a_merged_box_matches_one_word_and_misses_the_other():
    b = measures.box_numbers(BOXES_MERGED, TERRA_TWO, READS_MERGED)
    assert b[0.5]["matched"] == 1 and b[0.5]["missed_reasons"]["merged"] == 1

def test_paired_labels_of_identical_arms_is_zero():
    p = measures.paired_labels(ARM, ARM, boot=50)
    assert p["diff"] == 0 and p["lo"] == 0 == p["hi"]
```

- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/test_wordread_measures.py -v`. Expected: FAIL
  (no module).
- [ ] **Step 3: Implement** to the interfaces. A Terra word is "split" when two or more of our boxes each meet it at
  IoU ≥ 0.1 and none at ≥ 0.5. It is "merged" when the box that meets it best also meets another Terra word better.
- [ ] **Step 4: Run the tests to verify they pass.** Then write today's baseline to
  `experiments/wordread/baseline/{headline,boxes,reads}.json`, from `data/solver/boxes` and `data/solver/text-read`
  (the headline should come out at 41.1% over 719). Expected: PASS, and a baseline of about 0.411.
- [ ] **Step 5: Commit** (after the pgrep check): the two module files and the test.

### Task 2: The Mission Street review page (`wordread/review.py`, `scripts/wordread/review_page.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/review.py`, `pipeline/scripts/wordread/review_page.py`; Test
`pipeline/tests/test_wordread_review.py`.

**Interfaces:**
- **Consumes:** Task 1, `textnet/data.py` (page, ink, Terra words), and the fold models' outputs (chance, kind,
  direction).
- **Produces:**
  - `mission_sheet() -> str`: the 2000 sheet holding the longest length of the DataSF centreline named MISSION ST
    inside its footprint, mapped through the warp. Find the centreline loader the walls use (`STREET_CROSSING`).
  - `miss_reason(word_rect, chance, ink, components, level) -> tuple[str, dict]`. The reason is one of:
    - `"below_level"`;
    - `"merged_into_line"`: the component holding its ink has a length > 3× the word's length;
    - `"split"`;
    - `"wrong_size"`;
    - `"outside_map"`.

    The details hold the highest chance on the word's ink, the share of its ink above the level, the swallowing
    component's size, and the reading at Terra's box when it is given.
  - `colour(ours: str, terra: str | None) -> str`: "green" when equal, "purple" when different, "none" with no Terra
    word.
  - `page_data(sheet, boxes, reads, terra, chance, before: dict | None) -> dict`: everything the page draws, with the
    top-line panels from Task 1.
  - `render(data, out: Path) -> Path`: one self-contained HTML page.
    - The full-resolution sheet image: grey ink, red where chance ≥ 0.5 (text), white paper.
    - SVG overlays:
      - every box, with a reading-direction arrow and a tag (kind · chance · angle);
      - the reading in green or purple, with Terra's string in red after a purple one;
      - green-shaded boxes for ours unmatched but read as text;
      - thin grey outlines for ours unmatched and read as "none";
      - yellow-shaded dashed Terra boxes for misses, with the reason and details.

      Shading sits under the red ink and never over it.
    - Tapping a box shows its metadata.
    - The two top-line panels.
    - A line naming the models that drew it. Every px and box comes from the fold model that held out that px's
      quarter.
    - Pan and zoom.
    - It works on a phone, and in both themes.
    - Under 16 MB.

- [ ] **Step 1: Write the failing tests**

```python
def test_colour_rules():
    assert review.colour("RH-2", "RH-2") == "green" and review.colour("RM-2", "RH-2") == "purple"
    assert review.colour("RH-2", None) == "none"

def test_a_word_inside_a_long_line_component_is_merged_into_line():
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD_ON_LINE, COMPONENTS_LINE, 0.99)
    assert r == "merged_into_line" and d["component_px"] > 3 * d["word_length_px"]

def test_a_word_below_the_level_says_how_close():
    r, d = review.miss_reason(WORD, CHANCE_062, INK_WORD, COMPONENTS_NONE, 0.99)
    assert r == "below_level" and 0.6 <= d["max_chance"] <= 0.63

def test_shading_never_covers_red_ink(tmp_path):
    out = review.render(PAGE_DATA_TINY, tmp_path / "p.html")
    assert review.red_px_after_overlay(out) == review.red_px_of(PAGE_DATA_TINY)
```

- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** The script `review_page.py` builds the page for the Mission sheet from the latest run, or
  from the baseline. It writes it to `experiments/wordread/review/<run>/review.html`, and keeps `before` from the
  previous run's numbers. The controller publishes it to one Artifact URL and updates it in place.
- [ ] **Step 4: Run the tests, then build the baseline page.** Open the HTML and check the overlays against the agreed
  mockup (`https://claude.ai/artifact/2ytdcmDwbya7t72JMTxg5F`). Expected: PASS, and a page.
- [ ] **Step 5: Commit** `review.py`, `review_page.py` and the test.

### Task 3: Reader crops from our own boxes, sloppy crops and masking (`textread` options)

**Files:**
- Modify `pipeline/src/sfzmaps/solver/textread/crops.py` and `textread/train.py`: off-by-default options; no new
  import into `train.py`'s stage closure.
- Create `pipeline/src/sfzmaps/solver/wordread/readercrops.py`.
- Test `pipeline/tests/test_wordread_readercrops.py`.

**Interfaces:**
- **Consumes:** Task 1's matching; `crops.box_crop`, `crops.rectify`; `composite.Bank`'s real line pieces (Task 6 of
  the textlines plan).
- **Produces:**
  - `own_box_crops(boxes, terra, iou=0.5) -> list[tuple[np.ndarray, str, dict]]`: our boxes matched to a Terra word
    at IoU ≥ 0.5, cut as `box_crop` does, with Terra's string as the target. Only training-quarter boxes outside the
    answered blocks.
  - `offset_distribution(boxes, terra) -> dict`: the measured spread of our matched boxes against Terra's (end
    offsets in word heights, the angle difference, the height ratio).
  - `sloppy_crop(page, terra_word, dist, pieces, rng) -> np.ndarray`:
    - Terra's word cut with ends, angle and height drawn from `dist`;
    - with probability `dist["line_share"]` (the measured share of matched words holding line ink), a real line piece
      ORed through the letters;
    - with the measured share of boxes holding a neighbour, a fragment of the nearest other word.
  - `mask_outside(crop, word_mask) -> np.ndarray`: px outside `word_mask` (here the box's own component, dilated by
    `wordboxes.INK_R`) are set to paper.
  - `textread/train.py`:
    - `READER` options `own=0.0`, `sloppy=0.0`, `mask=False` (default off);
    - `fit` draws own-box and sloppy crops at those shares from a pool passed in (`extra=None`).
  - `read.py` gains `mask=False` for reading, behaviour unchanged at the default.

- [ ] **Step 1: Write the failing tests:**
  - own-box crops carry the matched Terra string and only training-quarter boxes;
  - `sloppy_crop`'s offsets follow `dist` (mean within 10% over 500 draws);
  - `mask_outside` blanks exactly the px outside the mask;
  - with the default options, `fit`'s first 20 batches are byte-identical to today (pin a hash first).
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** Measure `offset_distribution` on the training quarters from today's boxes, and save it
  to `experiments/wordread/reader/offsets.json`.
- [ ] **Step 4: Run the tests to verify they pass.** Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 4: The reader experiments: own boxes, sloppy crops, masking (items 3)

**Files:** Create `pipeline/scripts/wordread/run_reader.py`; Test `pipeline/tests/test_wordread_run_reader.py` (the
folder contents with a 2-step CPU recipe on synthetic crops).

**Interfaces:**
- **Consumes:** Tasks 1 and 3; `textread/train.fit`, `read.build_read`'s reading path, `gpu.gpu_turn`.
- **Produces:** `run(name, quarter, seed, own, sloppy, mask, boxes_dir) -> Path`.
  - It trains the quarter's reader (and its 0/180 classifier, unchanged) with the options, and reads the quarter's
    boxes from `boxes_dir`.
  - It writes `experiments/wordread/runs/<name>-<q>-s<seed>/` with `labels.parquet` (right or wrong per answered
    label), `headline.json`, `box_numbers.json`, `read_numbers.json` and `run.json`.

**The keep rule** (written now): a variant replaces the current reader if, on NE over seeds 0 and 1:
- the headline rises, with the paired label-bootstrap interval (`measures.paired_labels`) excluding zero;
- the reading at Terra's boxes doesn't fall by more than its seed noise;
- the controller's look at the review page agrees.

Variants, each against the current reader (run first as `base`):
- (a) `own=0.3`;
- (b) (a) + `sloppy=0.3`;
- (c) the best of (a)/(b) + `mask=True`.

A kept variant is then confirmed on all four folds at seed 0.

- [ ] **Step 1: Write the failing test.**
- [ ] **Step 2: Run it to verify it fails.**
- [ ] **Step 3: Implement, then run the arms one at a time:** `base`, (a), (b), (c) on NE at seeds 0 and 1.
- [ ] **Step 4: Decide by the rule,** confirm the kept variant on all four folds, and rebuild the review page.
  Write `experiments/wordread/reader/decision.md` with the numbers.
- [ ] **Step 5: Commit** the script and its test.

### Task 5: Detector word heads and their targets (`textnet` options)

**Files:**
- Modify `pipeline/src/sfzmaps/solver/textnet/model.py` (`OUTPUTS` per recipe), `data.py` (`crop_targets(...,
  words=False)`), `train.py` (`Recipe.heads="all"` gains `"words"`; the loss terms).
- Create `pipeline/src/sfzmaps/solver/wordread/shrink.py`.
- Test `pipeline/tests/test_wordread_heads.py`.

**Interfaces:**
- **Produces:**
  - **Amended (owner, 11-17: no picked core size):** NO shrink ratio. `shrink.prepare(sheets)` fills
    `SheetData.cores` with each word's own labelled rectangle. Touching words are separated by the learned tags, the
    geometry votes (Task 6) and the learned box model (Task 15). `measure_r`, `core` and `unclip` (built in
    87f3d82/da3a149) stay only as a record of the overlap measurement; nothing uses r.
  - **`crop_targets(..., words=True)`** adds:
    - `core` (uint8, all px in a word's rectangle);
    - `geo` (float32 (5, h, w) at core px: distances to the word's top, bottom, start and end edges in crop px, plus
      the reading angle, the semantic edges taken from the word's reading angle);
    - `tag_id` (int32 word id at core px, 0 elsewhere).
  - **`Recipe.heads="words"`** gives `OUTPUTS` = 7 + 1 (core) + 4 (geometry distances) + 4 (the PAN tag). The angle
    reuses channels 5-6.
  - **The loss adds:**
    - for the core: BCE over all known px with hard-negative mining at 3:1, plus Dice;
    - for geometry at core px: IoU loss on the distances, 10 × (1 − cos) on the angle;
    - for the tag: PAN's aggregation (pull to ≤ 0.5) and discrimination (push ≥ 3), each × 0.25.

    The per-ink text head (channels 0-6) is trained exactly as today.
  - `heads="all"` (the default) stays byte-identical.
- [ ] **Step 1: Write the failing tests:**
  - (amended) `prepare` fills each word's core with its own rectangle;
  - `geo` at a core px of a 0° word gives the right four distances;
  - the default recipe's batches and the `GUARD_WEIGHTS` are unchanged;
  - `test_build.py` passes.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** (The overlap curve stays recorded in `experiments/wordread/detector/shrink.json`.)
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 6: Word formation from cores, votes and tags (`wordread/words.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/words.py`; Test `pipeline/tests/test_wordread_words.py`.

**Interfaces:**
- **Consumes:** Task 5's outputs (predicted core chance, geo distances, tag vectors).
- **Produces (amended, owner 11-17: no level, no IoU merge, no r, no one-box-per-component):**
  - `core_words(core, geo, tag, ink, spreads, word_prior) -> pd.DataFrame`, in `wordboxes.COLUMNS` with the same
    meaning, plus `p_word`, `core_mass` and `tag_spread`.
    - Each px votes for a box (its four distances and angle, Task 5's `_geometry` inverted exactly) and carries its
      tag. A vote's weight is the px's core chance; there is no level.
    - Words are the components of a mixture over the weighted votes: each word is a box and a tag mean, and its votes
      are Gaussian about them with the spreads `spreads` (corner and tag spreads, measured in Task 8 on the
      training quarters' predictions against Terra's boxes). Spatial correlation between neighbouring px is
      accounted for (an effective number of independent votes per word, measured), so the likelihood does not grow
      with word size alone.
    - The number of words is decided by that one likelihood with `word_prior` (a fitted log prior per extra word):
      one model for splitting and joining, no split-then-rejoin.
    - Every box keeps `p_word` (its share of the weighted vote mass); weak boxes are kept with their probability,
      never discarded by a rule.
  - `measure_spreads(...)` and `fit_word_prior(...)`: for Task 8, on the training quarters' predictions against
    Terra's words (never the owner's answers).
- [ ] **Step 1: Write the failing tests** on synthetic maps, one per failure kind:
  - two touching words become two boxes;
  - a noisy single word stays one word;
  - a line touching a word doesn't extend its box;
  - a strong minority cluster of votes (a real second word) becomes its own box;
  - a core at the page edge gives a clipped box;
  - rotated words (30, 90, 150, −45, 180°) come back exactly from unrounded votes.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 7: Doubt weights for Terra words (item 6, `wordread/doubt.py`, `crop_targets` option)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/doubt.py`; Modify `textnet/data.py` (`SheetData.weights`,
default empty = 1) and `textnet/train.py` (the word losses weight each word's px by it); Test
`pipeline/tests/test_wordread_doubt.py`.

**Interfaces (amended, owner 11-17: weights, not a q99 cut or flags):**
- **Produces:**
  - `doubt_features(sd) -> pd.DataFrame`, per Terra word: `ink_outside` (the share of its ink pieces' px outside its
    box), `forced_height` (`boxes.word_rect`'s near-45° case), kind, and, for words Terra read twice (tile seams),
    the other read: the overlapping word of the other pass with the highest IoU (argmax, no IoU cut), the pair's
    IoU, and whether the normalised strings agree.
  - `fit_doubt(features) -> dict`: on the training quarters' double-read words outside the answered blocks,
    a logistic model of P(strings agree) and a beta regression of the pair's box IoU, from `ink_outside`,
    `forced_height` and kind.
  - `doubt_weights(sd, params) -> (box_weight, string_weight)` per word: the expected box IoU (for the detector's
    word targets) and P(string right) (for the reader's targets). No cut, no in/out flag.
  - Weights and fitted parameters are saved to `experiments/wordread/detector/doubt.json`.
- Also (Task 5's review): geometry targets are unknown for words whose rectangle touches a held px; remove the test
  that restates `HEAD_OUTPUTS`.
- [ ] **Step 1: Write the failing tests:**
  - more ink outside the box gives a lower weight, smoothly (no step);
  - the fit recovers known coefficients on synthetic double reads;
  - the word losses scale with the weights (weight 0 = no gradient from that word);
  - `fit_doubt` refuses answered-block or held-quarter words;
  - with `words=False`, the targets are unchanged.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement,** then write the counts.
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 8: The detector experiments (items 1, 2, 6)

**Files:** Modify `pipeline/scripts/textlines/variant.py` (`--heads words`, `--doubt`, and word formation after
prediction); Create `pipeline/scripts/wordread/run_detector.py` (detector run, then words, then the reader retrained
on its boxes via Task 4's `run`, then the measures); Test `pipeline/tests/test_wordread_run_detector.py`.

**The keep rule** (written now): the detector variant replaces today's if, on NE over seeds 0 and 1, end to end
(detector plus a reader retrained on its boxes with Task 4's kept options):
- the headline rises, with the paired label interval excluding zero;
- the gate's measures for the per-ink head (`linegate`: m1a_in_box, m2, m2_clean_words, m1c) lose nothing beyond the
  control's noise against today's best per-ink run;
- the controller's look at the review page agrees.

Arms (amended, owner 11-17):
- `base`: today's recipe, with words from connected ink and the Task 4 reader;
- (a) `heads=words`, words by Task 6's weighted vote mixture;
- (b) (a) plus Task 7's doubt weights;
- (c) the better of (a)/(b) plus Task 15's box model in Task 9's choice.

Before (a): measure Task 6's spreads and fit its word prior on the training quarters' predictions; place the boxes
(`quarter`, x/y, lot) with the code `wordboxes.sheet_boxes` uses (shared, not copied); stitch tiles so a word on a
tile edge is one box. The first mps run checks that two short runs repeat and records peak memory (Task 5's review).

A kept variant is confirmed on all four folds at seed 0.

- [ ] **Step 1: Write the failing test** (the run folder's contents on synthetic data, with a 2-step recipe).
- [ ] **Step 2: Run it to verify it fails.**
- [ ] **Step 3: Implement, then run the arms one at a time.**
- [ ] **Step 4: Decide,** confirm, rebuild the review page, and write `experiments/wordread/detector/decision.md`.
- [ ] **Step 5: Commit.**

### Task 9: Box choice by the reader (item 5, `wordread/choose.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/choose.py`; Test `pipeline/tests/test_wordread_choose.py`.

**Interfaces:**
- **Produces:**
  - `candidates(box, ink, neighbours) -> list[tuple[str, np.ndarray]]`, the candidate kinds:
    - `as_found`;
    - `split_gap` (at the widest gap along the axis, both halves);
    - `join_nearest` (with the nearest box within one word height along the axis);
    - `trim_start` and `trim_end` (each by the measured median character width of its kind).
  - `choose(box, cands, read_fn, prior_fn=None) -> tuple[str, dict]`: the candidate with the highest log posterior,
    with `prior_fn` (Task 10) or the ink alone.
  - **As built (d4b62c2, ea52c04):** a fitted conditional logit over the candidates (no windows or minimum gaps).
    **Amended (owner 11-17):** `choice_features` also takes Task 15's box-model log-probabilities (the word count of
    each candidate box, and the boundary at a split) as optional inputs, fitted like the others; added when Task 15
    lands.
- [ ] **Step 1: Write the failing tests:**
  - a box holding two words is split when the halves read better;
  - a word cut in two is joined;
  - with equal reads, `as_found` wins (a stable tie-break).
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 10: Context packs and the reader's posterior with them (item 4, `wordread/context.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/context.py`, `pipeline/scripts/wordread/build_pack.py`; Test
`pipeline/tests/test_wordread_context.py`.

**Interfaces:**
- **Produces:**
  - `Pack` (one JSON or parquet file per map), with:
    - `streets` (name, centreline linestring, EPSG:2227);
    - `blocks` (number, outline);
    - `codes` (the era's code list).
    - It has NO lot codes for the 2000 map; the builder raises if asked to add them.
  - `build_sf_pack(year) -> Pack`, from the DataSF centrelines and blocks the solver already loads (find the
    loaders; `lots/table.parquet` has `block_num`), with `codes` from `codereader.labels.table_codes`' era list.
  - `prior(pack, box_map: dict, kind_p: np.ndarray) -> dict[str, float]`: log prior over the candidate strings near
    the box:
    - street names: by the distance from the box's axis to each centreline and the angle difference;
    - the block number: the block holding the box centre;
    - codes: the era list, flat.

    An `escape` term covers every other string.
  - `fit_prior(words: pd.DataFrame) -> dict`: the weight w and escape mass e, by maximum likelihood on Terra-matched
    words of the other three quarters (never the owner's answers).
  - `posterior(ink_logp: dict[str, float], prior: dict, w, e) -> dict`: log p(s | ink) + w · log prior, with mass e
    on "other". The reader computes `ink_logp` exactly by CTC (`ctc.string_log_probs`) for every candidate string.
- [ ] **Step 1: Write the failing tests:**
  - a string off the list with clearly stronger ink wins;
  - with no pack, the posterior equals the ink-only one;
  - a box with no street within range falls back to the escape mass;
  - `build_sf_pack(2000)` refuses lot codes;
  - `fit_prior` never sees answered-block words (it raises if given one).
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement,** then build the SF 2000 pack to `experiments/wordread/packs/sf-2000.parquet`.
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 11: The box-choice and pack experiments (items 5, 4: separate keep rules)

**Files:** Create `pipeline/scripts/wordread/run_choose_pack.py` (reads only: no training); Test
`pipeline/tests/test_wordread_run_choose_pack.py`.

**Keep rules** (written now), on NE over the kept models' two seeds:
- **(5) Box choice** (ink only) is kept if the headline rises with the paired label interval excluding zero, and the
  review page agrees.
- **(4) The pack**, after (5), is kept if:
  - the headline rises with the interval excluding zero;
  - the read numbers against Terra on street names and block numbers rise;
  - codes don't fall by more than the noise.

  It is also reported on the 1980 test drawing as agreement with Terra, with the pack of today's streets and the
  mismatch of old names counted.

Kept steps are confirmed on all four folds.

- [ ] **Step 1: Write the failing test.**
- [ ] **Step 2: Run it to verify it fails.**
- [ ] **Step 3: Implement, then run.**
- [ ] **Step 4: Decide,** confirm, rebuild the review page, and write `experiments/wordread/choose-pack/decision.md`.
- [ ] **Step 5: Commit.**

### Task 12: Grey input and full resolution (item 7)

**Files:** Modify `textnet/data.py` (`SheetData.grey_bits`, default empty) and `train.py` (`Recipe.input="ink"`, which
gains `"grey"`; the default is byte-identical; under `"grey"` a crop is page/255 resized bilinearly); `predict.py`
reads grey when the model's recipe says so; Test `pipeline/tests/test_wordread_input.py`.

**The keep rule:** with the kept detector and reader, on NE at two seeds, grey (and separately full size) is kept if
the headline rises beyond the interval and the per-ink gate loses nothing beyond noise.

- [ ] **Step 1: Write the failing tests:** the default recipe is unchanged; a grey crop equals page/255 at the crop.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement, then run the two comparisons.**
- [ ] **Step 4: Decide,** and write `experiments/wordread/input/decision.md`.
- [ ] **Step 5: Commit.**

### Task 13: Map-like synthetic scenes from DataSF geometry (item 8, `wordread/scenes.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/scenes.py`; Modify `textnet/train.py` (`Recipe.scenes=0.0`;
scenes come in through the bank object, as composites do); Test `pipeline/tests/test_wordread_scenes.py`.

**Interfaces:**
- **Produces:** `SceneBank.build(sheets, areas, seed)` and `SceneBank.crop(rng, size, scale, turns) -> dict`, with the
  keys of `crop_targets(..., words=True)`.
  - **Geometry:** lots, blocks and street edges are drawn as 1-bit lines at a sheet's scale, from DataSF geometry, at
    the measured widths (`strokes.npz`).
  - **Labels:**
    - block numbers at block centres;
    - street names along centrelines;
    - codes inside lots, at offsets measured from Terra's code boxes.

    Each label is a real clean word of that string from the bank (`linegate.clean_word_mask`). If no clean word of
    that string exists, the label is skipped.
  - **Masks** are exact for every head: ink, core, geo and tag.
  - **Held out:** none of a held-out quarter's or the answered blocks' geometry, and none of their words.
- [ ] **Step 1: Write the failing tests:**
  - the masks are exact (text is ⊆ the pasted words' ink; the cores are disjoint);
  - the same seed gives the same scene;
  - `scenes=0.0` leaves batches byte-identical;
  - a held-out area never appears.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement,** then run kept recipe + scenes at the share of crops measured to match real word density.
  Two seeds on NE, with the same keep rule as Task 8.
- [ ] **Step 4: Decide,** and write `experiments/wordread/scenes/decision.md`.
- [ ] **Step 5: Commit.**

### Task 14: The final model on every fold, the final page, the results note

**Files:** Create `docs/research/joint-solver/notes/2026-11-17-map-text-fields-way-results.md`.

- [ ] **Step 1: Run the kept recipe on all four folds at seeds 0 and 1:** detector, words, reader, box choice and
  pack.
- [ ] **Step 2: Score:**
  - the headline over all 719 labels, with intervals, against today's 41.1%;
  - the box and read numbers;
  - the per-ink gate;
  - the 1980, 1972 and 1975 test drawings as agreement with Terra.
- [ ] **Step 3: Rebuild the review page from the final model.**
- [ ] **Step 4: Write the results note.** It covers:
  - each item's arms and numbers;
  - its keep or drop;
  - the pictures;
  - what is still open;
  - the adoption path (which stages a rebuild would touch, from `uv run sfzmaps build --dry-run`). Adoption waits for
    the owner.
- [ ] **Step 5: Commit the note** (after the pgrep check).

Stop here. Adoption into the shipped stages, and T4/T5 of the text-reader plan, wait for the owner.

### Task 15: The learned box model (added by the owner, 11-17; built alongside Tasks 6-7, run in Task 8 arm (c))

The owner: "there are things you can learn about boxes and how many words they hold ... Don't just hardcode these
... There is a hidden function that you should be able to learn automatically." So no hand features (ink density,
the ink centroid against the box centre, gaps): a small network learns them from the box's own pixels.

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/boxmodel.py`, `pipeline/scripts/wordread/run_boxmodel.py`;
Test `pipeline/tests/test_wordread_boxmodel.py`.

**Interfaces:**
- **Consumes:** boxes in `wordboxes.COLUMNS` form (today's, Task 6's, or Task 9's candidates); the page ink; the
  detector's per-ink text chance and, when present, its word-head outputs (core chance, tags); Terra's words
  (`textnet/boxes.py` `terra_boxes`); Task 7's weights when present; `textread/crops.py`'s rectification.
- **Produces:**
  - `box_examples(boxes, terra, sheets, quarter, weights=None) -> Examples`: per box, the rectified crop with the
    reader crops' own margin (`crops.py`; no new margin constant). Channels: ink, text chance, and the word-head
    outputs when present. Targets, with no cut:
    - the word count as a distribution: each Terra word contributes its share of ink inside the box, so the count is
      the Poisson-binomial of those shares;
    - per column along the box, whether it lies in the gap between two consecutive held Terra words (exact columns
      from their ends).
    - Plus joined real neighbours: two real Terra word crops on one line, the gap drawn from the measured
      distribution of real same-line gaps, the targets known by construction.
    - Only training-quarter words outside the answered blocks; it raises otherwise.
  - `BoxNet`: plain torch, like the reader's backbone. Outputs: count logits over 0..K (K = the largest count in the
    training examples, measured) and a boundary logit per column.
  - `fit_boxmodel(examples, recipe) -> model` inside `gpu.gpu_turn`; deterministic seeds.
  - `box_logp(model, crops) -> (count_logp, boundary_logp)`, for Task 9's `choice_features`.
- **The keep rule** (written now): on NE over seeds 0 and 1, Task 9's choice with the box model's log-probabilities
  against without:
  - the headline rises with the paired label interval excluding zero;
  - the held-out box numbers' `merged_with_word` count falls (Mission sheet: 293 today);
  - the review page agrees.

  A kept model is confirmed on all four folds at seed 0.
- [ ] **Step 1: Write the failing tests:**
  - the count target is the exact Poisson-binomial of the ink shares;
  - the boundary columns are exactly the gaps between held Terra words;
  - `box_examples` raises on held-quarter or answered-block words;
  - a short CPU fit on synthetic one-word and two-word crops separates them well above chance;
  - joined-neighbour gaps follow the measured distribution.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** No training runs in this step (Task 8 arm (c) trains it).
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**
