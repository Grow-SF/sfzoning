# Learned Hatched-Map Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CNN that reads the 1921 zoning scans as printed. It finds the printed blocks from a learned street output,
splits each block into pieces of one district, and gives each lot its piece's district. It is scored on the owner's
held-out random 1921 lots against today's map.

**Architecture:** One small fully convolutional network (`sfzmaps.solver.hatchnet`) on page tiles, with three outputs
per pixel:
- a street logit, trained on today's blocks placed on the scan;
- six district logits;
- a pooling-weight logit.

The district and weight outputs are trained only through per-lot pooled losses: the reads go through a measured
read-noise matrix, and the owner's answers count as exact. There are four folds by quarter, each withholding that
quarter's reads and answers.

Blocks are the connected areas the street output calls "not street". Pieces are the connected areas of one argmax
district inside a block. A lot takes the pooled district of its pieces.

**Tech Stack:** Python, PyTorch (already a dependency, `mps` through `sfzmaps.gpu`), numpy/scipy/OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-hatched-maps-learned-reader-design.md` (34f1db1). Read it first,
especially "What the owner said", "Anti-goals" and "Settled after the builder check".

## Global Constraints

- **No hand rules.** No hand-made cleaning, mask, threshold, window or special-case rule in the reading. The engineering
  settings allowed are the network's size, the training schedule and the input tile size, each stated in a comment.
  - A two-class decision takes the most likely class (probability > 1/2): that is the argmax, not a cutoff.
- **No LLM output used at inference.** The reads (`district_pattern.json`) are training labels only. No sheet scale,
  period or orientation may be measured on them.
- **No pseudo-labels.** No learning from our own predictions.
- **Scoring is by area.** A random lot is scored only by a model that saw neither its quarter's answers nor its
  quarter's reads.
- **Pictures before scores.** No score is reported before the owner has seen that step's pictures. Every task with a
  visual output ends with pictures: the controller sends them with SendUserFile and waits for the owner's OK.
- **The score:** the owner's random lots on the `1921` page, against published (199/240, 82.9%), with
  `sfzmaps.solver.answers1921`. Version 1 is done when the paired interval against published excludes zero.
- **Placement:** `data/work/1921/<sheet>/transform.json` (not `data/warp`). The scan goes in at its own resolution,
  never rescaled or cleaned. A sheet scanned sideways is turned back by its exact quarter turn
  (`hatching.paper_turn`), which is lossless. Each sheet's measured ft/px is an input.
- **Cost and load:** $0 (`SFZMAPS_MAX_USD=0`). One GPU process at a time through `sfzmaps.gpu.take_turn`. Keep the
  load around 20–80, `OMP_NUM_THREADS=1`.
- **Deterministic:** fixed seeds, `torch.use_deterministic_algorithms(True)`, and the device recorded. The same inputs
  and device give the same bytes.
- **Where things go:** outputs under `pipeline/data/solver/experiments/hatch-cnn/`, never committed. Code on branch
  `hatch-cnn` from `hatch-regions` at 34f1db1.
- **Before every commit:** run `pgrep -f "sfzmaps build"` first. Messages end with these two lines:
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q

## Review Focus

1. **A lot that no learned block covers** (on a learned street, off the sheet, outside the neatline) gets no district.
   It counts as unmatched in the score and never defaults to 2R. Test in Task 4.
2. **Overlapping scans** (section 12 is scanned three times): each place is labelled and predicted once, on the sheet
   the published draw order picks (`hatching.draw_order` + `hatching.owned`). Test in Task 1.
3. **Fold leakage:** fold q's training never sees quarter q's reads or answers. Test by perturbation in Task 3.
4. **Determinism:** two CPU trainings of the test network with the same seed give bit-identical weights. Test in
   Task 2.
5. **The turned sheet:** turning it back gives the same tile as the upright scan of the same place. Test in Task 1.

---

### Task 1: Training rasters and the lot table

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchnet/__init__.py`
- Create: `pipeline/src/sfzmaps/solver/hatchnet/data.py`
- Create: `pipeline/src/sfzmaps/solver/hatchnet/__main__.py`
- Test: `pipeline/tests/test_hatchnet_data.py`

**Interfaces:**
- Produces:
  - `load_scan(series: str, sheet: str) -> np.ndarray`: float32 luminance in [0, 1] (0 = ink), the page as scanned,
    turned upright by its quarter turn.
  - `sheet_rasters(series: str, sheet: str) -> SheetRasters`, with fields:
    - `ink` (float32);
    - `block` (uint8; 1 today's blocks, 0 street, 255 unknown outside the neatline or on another sheet's owned area);
    - `lot` (int32, the solver lot index, −1 none);
    - `ft_per_px` (float).

    Blocks are `binary_fill_holes(lots >= 0)` from the solver lot raster, carried onto the page through the
    `data/work` placement with `hatching.PageWorld` / `hatching.page_values`.
  - `lot_table(series: str) -> pd.DataFrame`, one row per solver lot: `lot`, `blklot`, `quarter`, `read_code`,
    `read_conf`, `answer` (truth via `answers1921`, or empty), `answer_page`, `stratum`.
  - `swatches(series: str) -> list[tuple[str, str, tuple[int, int, int, int]]]`: (sheet, code, box) from
    `districts/legends.json`.
  - CLI `python -m sfzmaps.solver.hatchnet data --series 1921`, which writes
    `experiments/hatch-cnn/data/<sheet>.npz` and `lots.parquet`.

- [ ] **Step 1: Branch.**
  - Commit the parked, uncommitted R16 work in `hatchfit.py` / `test_hatchfit.py` to a side branch
    `hatch-regions-r16-parked` (a WIP commit), so it isn't lost.
  - Run `git switch hatch-regions`.
  - Run `git switch -c hatch-cnn`.
- [ ] **Step 2: Write the failing tests.**
  - `test_turned_sheet_matches_upright`: a synthetic page and the same page `np.rot90`'d with `paper_turn` = 1 give
    the same `load_scan` array.
  - `test_overlap_owned_once`: two synthetic sheets overlapping in world space. A world point in the overlap is
    `block` ≠ 255 on the draw-order owner only.
  - `test_lot_table_answers_and_reads`: a fixture with one read and one owner answer per lot gives the expected
    `read_code`, `read_conf` and `answer` (`unzoned` → `2R`, canttell dropped, as in `answers1921`).
  - `test_no_rule_masks`: `ink` equals the scan's luminance everywhere inside the neatline. Nothing is erased.
- [ ] **Step 3:** Run `cd pipeline && uv run pytest tests/test_hatchnet_data.py -v`. Expect a FAIL (module missing).
- [ ] **Step 4:** Implement it. Reuse `hatching.PageWorld`, `page_values`, `owned`, `draw_order`, `paper_turn` and
  `load_crop` for geometry only, and none of its masks. The placement comes from `data/work/<series>/<sheet>/
  transform.json` through `SheetTransform.from_json`.
- [ ] **Step 5:** Run the tests. Expect a PASS. Then run the full suite.
- [ ] **Step 6:** Run `OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchnet data --series
  1921`. Report the sheets, pixels and lots per quarter, the reads per code, and the answers per page.
- [ ] **Step 7:** Commit `hatchnet: 1921 training rasters (scan as printed, today's blocks via placement) and lot
  table`.

### Task 2: The network and the learned streets

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchnet/model.py`
- Create: `pipeline/src/sfzmaps/solver/hatchnet/train.py`
- Create: `pipeline/src/sfzmaps/solver/hatchnet/look.py`
- Test: `pipeline/tests/test_hatchnet_model.py`

**Interfaces:**
- Consumes: Task 1's `SheetRasters` and npz files.
- Produces:
  - `HatchNet(width: int)`, a fully convolutional U-Net style network. Its input is 2 channels (ink, and ft/px as a
    constant plane). Its output is 8 channels: [street, 6 districts in `NAMES` order, weight]. `NAMES = sorted(["1R",
    "2R", "C", "LI", "HI", "U"])`.
  - `TILE`: the training tile side in px, stated with its reason. It must hold several of the widest 1R spacings
    (~30 px) and a whole text knockout.
  - `train_streets(rasters, seed, device) -> state_dict`: BCE on `block` (255 ignored), all 1921 sheets. Today's
    blocks are not owner answers, so this is no leak.
  - `predict_streets(net, sheet) -> np.ndarray` (float32 probability).
  - `blocks(street_prob, crop) -> np.ndarray` (int32): connected areas where street_prob ≤ 1/2, inside the neatline.
  - `look.streets(sheet, street_prob, blocks, out) -> Path`: the scan with learned streets tinted and each block
    outlined, whole and zoomed.

**Augmentation:** no flips or rotations, because 1R has a fixed map angle. Intensity jitter only, stated.

- [ ] **Step 1: Write the failing tests.**
  - `test_shapes`: a 2×2×256×256 input gives 2×8×256×256.
  - `test_deterministic_cpu`: two 5-step CPU trainings with seed 0 on a synthetic two-sheet fixture give
    bit-identical state dicts.
  - `test_streets_learnable`: on a synthetic page of white streets between hatched blocks, 200 CPU steps reach street
    BCE below the BCE of the constant prior (the share of street px). That proves learning, not a picked number.
  - `test_blocks_are_components`: two blank areas separated by a street-prob ridge give 2 blocks.
- [ ] **Step 2:** Run the tests. Expect a FAIL.
- [ ] **Step 3:** Implement. Use `torch.use_deterministic_algorithms(True)`, `sfzmaps.gpu.take_turn` around GPU work,
  and record the device in the saved checkpoint's metadata.
- [ ] **Step 4:** Run the tests. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run.** `python -m sfzmaps.solver.hatchnet streets --series 1921` trains, predicts every 1921 sheet,
  and writes `experiments/hatch-cnn/streets/<sheet>.npz` plus the pictures for 20180102094149543 (dense) and
  20180102100525362 (south), whole and zoomed.
- [ ] **Step 6:** Commit `hatchnet: network, learned streets and blocks`.
- [ ] **Step 7: Owner gate.** The controller sends the two pictures and waits for the owner's OK before Task 3.

### Task 3: District training through pooled lot losses (four folds, plus reads-only)

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/hatchnet/train.py`
- Test: `pipeline/tests/test_hatchnet_train.py`

**Interfaces:**
- Consumes: Task 1's lot table and rasters; Task 2's network with the street head trained.
- Produces:
  - `read_noise(lots: pd.DataFrame, train_quarters) -> np.ndarray` (6×6): T[j, k] = P(read k | true j), counted from
    the training quarters' RANDOM-stratum owner answers against their reads, plus one per cell (a uniform prior).
    The likely-wrong and least-sure pages were chosen where the reader was probably wrong, so they would overstate
    its errors; they still train as exact labels. Without answers, the identity.
  - `lot_logprob(district_logits, weight_logits, lot_px) -> Tensor` (n_lots × 6): the log of Σ_px softmax(weight) ·
    softmax(districts) over each lot's pixels.
  - Loss per lot:
    - for a read k: −log Σ_j p(j) T[j, k];
    - for an owner answer j: −log p(j);
    - for a swatch box: each pixel's district cross-entropy.

    Each lot counts once.
  - `train_fold(q: str | None, with_answers: bool, seed) -> state_dict`. `q=None` means all quarters (the deliverable);
    otherwise quarter q's reads and answers are withheld. The fine-tune starts from Task 2's street weights.
  - CLI `python -m sfzmaps.solver.hatchnet districts --series 1921`, which trains folds nw, ne, sw and se with answers,
    the same four reads-only, and the all-quarters model, one GPU job at a time. It writes
    `experiments/hatch-cnn/models/<name>.pt`.

- [ ] **Step 1: Write the failing tests.**
  - `test_read_noise_counts`: answers {1R: read 1R ×3, read 2R ×1} give T[1R] ∝ [counts + 1].
  - `test_pool_weights`: a lot whose weight logits favour its left half gets the left half's district.
  - `test_fold_withholds_quarter`: perturb quarter q's reads and answers at random. `train_fold(q)` gives bit-identical
    weights on CPU (3 steps).
  - `test_reads_only_ignores_answers`: with `with_answers=False`, changing the answers changes nothing.
- [ ] **Step 2:** Run the tests. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expect a PASS. Then run the full suite.
- [ ] **Step 5:** Run the CLI. Report each model's training loss curve, and T per fold.
- [ ] **Step 6:** Commit `hatchnet: district training through pooled lot losses, read-noise matrix, folds by quarter`.

### Task 4: Pieces, lot districts and the pictures

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchnet/predict.py`
- Modify: `pipeline/src/sfzmaps/solver/hatchnet/look.py`
- Test: `pipeline/tests/test_hatchnet_predict.py`

**Interfaces:**
- Consumes: the Task 3 models and the Task 2 blocks.
- Produces:
  - `pieces(district_prob, blocks) -> np.ndarray` (int32): connected areas of one argmax district inside each block.
  - `lot_districts(district_prob, pieces, lot_px) -> pd.DataFrame` (lot, code, p[6]): each lot's probability is the
    mean of its pieces' mean probabilities, weighted by the lot's pixels in each piece. A lot with no pixels in any
    block has no row.
  - CLI `python -m sfzmaps.solver.hatchnet predict --series 1921`. Each quarter's lots are predicted by the fold that
    withheld that quarter. The output is `experiments/hatch-cnn/predict/{with_answers,reads_only}.parquet`.
  - `look.districts(sheet, ...)`: the predicted districts tinted in the legend's six colours over the scan, with piece
    outlines, whole and zoomed. There is also a contact sheet of the owner's random 1921 lots, each a crop with the
    prediction and the owner's answer.

- [ ] **Step 1: Write the failing tests.**
  - `test_pieces_split_on_change`: one block, left half 1R and right half 2R, gives 2 pieces.
  - `test_lot_from_pieces`: a lot 70% in a 1R piece and 30% in a 2R piece gets 1R.
  - `test_uncovered_lot_has_no_row`: a lot whose pixels are all street gives no row.
- [ ] **Step 2:** Run the tests. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run.** Run predict for 1921, then the pictures for the two Task 2 sheets plus the random-lot contact
  sheet. Do NOT score yet.
- [ ] **Step 6:** Commit `hatchnet: pieces, lot districts, pictures`.
- [ ] **Step 7: Owner gate.** The controller sends the pictures and waits for the owner's OK before Task 5.

### Task 5: The score, and the later editions

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchnet/score.py`
- Modify: `pipeline/src/sfzmaps/solver/answers1921.py` (docstring only: answers train and score, split by area, the
  owner's 11-18 rule)
- Test: `pipeline/tests/test_hatchnet_score.py`

**Interfaces:**
- Consumes: Task 4's parquets; `answers1921.score` / `compare` / `published`.
- Produces:
  - CLI `python -m sfzmaps.solver.hatchnet score --series 1921`, which writes `experiments/hatch-cnn/score.json`. It
    compares four methods on the random `1921` page and on all answers: with-answers, reads-only, published, and the
    paused pipeline's per-lot evidence (`experiments/hatch-regions/evidence`).
  - CLI `python -m sfzmaps.solver.hatchnet run --series 1929|1935|1948`: the all-quarters model on those editions,
    with Task 1 rasters and no labels needed. It writes district pictures for two sheets each.

- [ ] **Step 1: Write the failing test.** `test_score_uses_heldout_predictions`: a fixture where the with-answers
  parquet would be perfect only if it leaked must score with the fold predictions given.
- [ ] **Step 2:** Run the tests, implement, and run them again. Expect a PASS. Then run the full suite.
- [ ] **Step 3:** Run the score. Report the table: right/n, share, 95% interval, paired difference against published,
  and the per-quarter split. Say plainly that U can't be scored, since no answer is U.
- [ ] **Step 4:** Run 1929, 1935 and 1948. The controller sends their pictures, which are for looking only.
- [ ] **Step 5:** Commit `hatchnet: score against the owner's held-out random lots; later editions`.
