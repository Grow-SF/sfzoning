# Learned Streets v2 (quick test) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the hatchnet street output so it learns streets as long, thin, connected lines. Narrow and
track-filled printed streets should separate blocks, and the Bay, lakes and other big blank areas should stop
being called street. Then compare the result with the Task 2 run on the same measures and pictures.

**Architecture:**
- **New street labels.** A pixel is street if it is off today's blocks and inside a street corridor. A corridor is
  traced from today's street centre lines (`data/ref/streets.parquet`): a pixel belongs to its nearest centre line's
  corridor if it is no farther from that line than the line's own distance to the nearest block.
  - Today's blocks are not-street.
  - Every other pixel is ignored (no gradient). That covers the Bay, lakes, ocean, unparcelled land, and 1921 streets
    gone today.
- **A soft-clDice loss alongside the BCE** (Shit et al., CVPR 2021):
  - skeleton recall: every centre line must be covered by a connected predicted street;
  - skeleton precision: the skeleton of every predicted street must lie in a corridor. It is computed over all
    pixels, labelled or not, so a "street" across open water is penalised.
- **Checkpoints and selection:** a checkpoint every 250 steps, and the kept one is the lowest held-out loss of the SAME
  BCE + clDice sum. The clDice part of that measure sees the thin streets.

**Tech Stack:** PyTorch, numpy/scipy (distance transforms), the existing `sfzmaps.solver.hatchnet` package.

**Why:** the debugging findings `scratchpad/streetdbg/findings.md` (RC1–RC4). The owner: "streets are consistent -
long and thin. Things that aren't streets like the Bay or lakes etc very much aren't long and thin"; "connectivity
loss is great".

**Spec:** `docs/superpowers/specs/2026-11-18-hatched-maps-learned-reader-design.md` (unchanged goals). This plan
replaces the street training of `docs/superpowers/plans/2026-11-18-hatched-maps-learned-reader.md` Task 2. Its
blocks rule, `look.streets` and the CLI stay.

## Global Constraints

- **No hand rules:** no picked thresholds, widths or windows. The corridor half-width comes from the data (each
  centre line's distance to today's nearest block). Ignored is ignored, never a ½ target. The clDice skeleton depth
  (its iteration count) is an engineering setting stated with its reason, as the widest street's half-width in px.
- **One change set, measured against the Task 2 run** with the same measures. No other change: same network, TILE,
  ft/px input, augmentation and seed.
- **Determinism and cost:**
  - deterministic (fixed seed, `torch.use_deterministic_algorithms(True)`);
  - `mps` through `sfzmaps.gpu.take_turn`;
  - $0, `OMP_NUM_THREADS=1`, load under about 60.
- **Where things go:** outputs under `pipeline/data/solver/experiments/hatch-cnn/streets-v2/` (models, maps, look),
  leaving the Task 2 run's files untouched for comparison. Code on branch `hatch-cnn`.
- **Pictures first:** pictures go to the owner (SendUserFile) before anything builds on this.
- **Commits:** run `pgrep -f "sfzmaps build"` first. Messages end with these two lines:
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q

## Review Focus

1. **A shoreline street:** its corridor stops at the street's own half-width on the water side; the water beyond is
   ignored. Test in Task A.
2. **A 1921 street with no centre line today:** it is ignored, not labelled not-street. Test in Task A.
3. **A blob of predicted street away from every centre line** raises the precision loss, even where every pixel is
   ignored. Test in Task B.
4. **A one-pixel gap** in a predicted street along a centre line raises the recall loss. Test in Task B.
5. **Selection picks by the same summed loss on held-out cells.** Test in Task B.

---

### Task A: Street labels from centre lines and today's blocks

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/hatchnet/data.py`. Add the street labels to each sheet's npz as `street`
  (uint8: 1 street, 0 not street, 255 ignored) and `centre` (uint8: 1 on a centre line), keeping `block`.
- Test: `pipeline/tests/test_hatchnet_data.py`

**Interfaces:**
- Produces:
  - `street_labels(block_world: np.ndarray, centre_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]`. It works on
    the solver's 4 ft grid, before carrying the result to the page as `block` is today.
    - With d_c the distance to the nearest centre-line px, c(x) that px, and w(c) the distance from c to the nearest
      block px: street = (not block) & (d_c(x) ≤ w(c(x))); not-street = block; ignored = everything else. Use
      `scipy.ndimage.distance_transform_edt` with `return_indices`.
    - It also returns the centre-line raster.
  - Centre lines come from `data/ref/streets.parquet` (EPSG:2227 LineStrings), rasterised onto the solver grid
    (`sfzmaps.solver.grid`). Use every row; the 798 rows with `active=False` are kept too, since 1921 streets are
    older than today's list.

- [ ] **Step 1: Write the failing tests.**
  - `test_corridor_between_two_blocks`: two blocks 10 px apart with a centre line between them give street on all 10
    px between them and not-street on the blocks.
  - `test_shoreline_stops_at_half_width`: a block, a centre line 5 px from it, then open space. Street extends
    exactly 5 px past the centre line on the open side; beyond that is 255.
  - `test_no_centre_line_is_ignored`: a gap between blocks with no centre line is 255, not 0.
- [ ] **Step 2:** Run `uv run pytest tests/test_hatchnet_data.py -v`. Expect a FAIL.
- [ ] **Step 3:** Implement, and carry `street` and `centre` to the page through the same placement and ownership as
  `block`.
- [ ] **Step 4:** Run the tests. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Run.** Regenerate the data for 1921. Render the new labels (street orange, not-street blue, ignored
  grey) over the scan for 20180102100525362 (the same zoom as before), 20180102094149543, and Lake Merced
  20180102094355582 (whole). Write `streets-v2/look/labels_*.png`.
- [ ] **Step 6:** Commit `hatchnet: street labels from centre-line corridors; everything else ignored`.

### Task B: Soft-clDice, checkpoint selection, retrain and compare

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchnet/topo.py`: `soft_skeleton(p: Tensor, iters: int) -> Tensor` and
  `cldice_terms(p, centre, corridor, iters) -> tuple[Tensor, Tensor]` (recall, precision), as in Shit et al.: iterated
  min/max pooling.
- Modify: `pipeline/src/sfzmaps/solver/hatchnet/train.py`. The street loss becomes BCE on labelled px + (1 − recall)
  + (1 − precision), with precision over all px. Save a checkpoint every 250 steps up to 3000, and keep the lowest
  held-out value of that same sum.
- Test: `pipeline/tests/test_hatchnet_topo.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_skeleton_of_thin_line`: a 5-px-wide bar's soft skeleton lies within its middle 3 px and keeps 95% or more
    of its length.
  - `test_recall_penalises_gap`: a predicted street covering a centre line except for a 1-px gap has a lower recall
    than one with no gap.
  - `test_precision_penalises_blob_off_centre`: adding a 40×40 predicted blob far from any centre line, on ignored
    px, lowers precision.
  - `test_selection_uses_same_sum`: with fake per-checkpoint held-out sums, the kept checkpoint is the argmin.
- [ ] **Step 2:** Run them. Expect a FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expect a PASS. Then run the full suite.
- [ ] **Step 5: Retrain and predict** into `streets-v2/`. Measure against the Task 2 run with the debugging scripts
  (`scratchpad/streetdbg/crossings.py`, `blank.py`, `specks.py`, `merges.py`):
  - corridor breaches by width, trained and held-out;
  - the 20 worst merges from `worst.csv`, re-scored;
  - the blank areas' street share (the Bay, Lake Merced lands, McLaren);
  - the speck count;
  - Twin Peaks block 112.

  Render the same pictures as Task 2 (the same windows) plus a side-by-side of v1 and v2.
- [ ] **Step 6:** Commit `hatchnet: soft-clDice street loss, checkpoint selection by the same held-out sum`.
- [ ] **Step 7: Owner gate.** The controller sends the side-by-sides and the table and waits.
