# Printed areas, first look (stage 1a of "read the print, then the lots")

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and picture, on Section 12, the printed areas the branch's existing street outputs give, so the owner sees the areas before anything is read from them and the next step (how to improve the finder) is chosen on evidence.

**Architecture:** One small module, `hatchnet/areas.py`: areas from a boundary probability, a measure of merged and shattered areas against today's blocks, and a picture. No training. It reads the saved street outputs of v1, v2 and v2c.

**Tech stack:** Python, numpy, scipy.ndimage, OpenCV; `uv run` from `pipeline/`.

**Spec:** `docs/superpowers/specs/2026-11-22-print-decides-parcel-splits-design.md` (stage 1a).

## Global constraints

- Section 12 only: sheet `20180102100525362`. No training, no GPU, no `sfzmaps build`, $0.
- Deterministic.
- No hand-set cutoffs beyond "most probable" (boundary probability above 1/2).
- Pictures at the scan's own resolution, with a caption saying what each is for.
- Nothing under `pipeline/data/` is committed.

## Review focus

- An empty boundary map (no streets found): one area, reported as merged, no crash (Task 1 test).
- An area that touches the frame cut's edge: still an area (Task 1 test).
- Today's block with no lot px inside the frame cut: left out of the measure, not counted as shattered (Task 1 test).

---

### Task 1: `areas.py`: areas, the merge and shatter measure

**Files:**
- Create: `pipeline/src/sfzmaps/solver/hatchnet/areas.py`
- Test: `pipeline/tests/test_hatchnet_areas.py`

**Interfaces:**
- Produces:
  - `areas(boundary_prob: np.ndarray, crop: np.ndarray) -> np.ndarray` (int32, 0 = boundary or off the frame cut, 1..n each 4-connected area where `boundary_prob <= 0.5` inside `crop`). Same rule as `train.blocks`; it calls it.
  - `against_blocks(area: np.ndarray, block: np.ndarray) -> dict` where `block` is today's blocks numbered on the page (int32, 0 off them). Returns:
    - `areas`, `blocks`: counts of each that hold px of the other;
    - `merged_areas`: areas that hold more than half of the px of two or more blocks;
    - `merged_px_share`: their share of all area px;
    - `blocks_in_merged`: blocks most of whose px lie in a merged area;
    - `shattered_blocks`: blocks whose px are spread so that no single area holds more than half;
    - `largest_area_px_share`.
  - `picture(lum: np.ndarray, area: np.ndarray, merged: np.ndarray) -> np.ndarray` (BGR uint8): the scan in grey, each area tinted its own colour, merged areas in red, boundaries untinted.

- [ ] **Step 1: Write the tests** (`tests/test_hatchnet_areas.py`), synthetic 60 x 80 pages:
  - `test_two_blocks_split_by_a_street`: boundary column between two blocks; `against_blocks` gives `areas == 2`, `merged_areas == 0`, `shattered_blocks == 0`.
  - `test_missing_street_merges_blocks`: no boundary between them; `merged_areas == 1`, `blocks_in_merged == 2`, `merged_px_share == 1.0`.
  - `test_invented_street_shatters_a_block`: three boundary columns through one block so no part holds over half; `shattered_blocks == 1`.
  - `test_no_boundary_at_all_is_one_area`: `boundary_prob` all zero; one area, no crash.
  - `test_area_at_the_crop_edge_counts` and `test_block_outside_the_crop_is_left_out`.
- [ ] **Step 2: Run** `uv run pytest -q -n 0 tests/test_hatchnet_areas.py`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** the three functions.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 2: Section 12, three finders, table and pictures

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/hatchnet/areas.py` (a `main` that runs the comparison)

**Interfaces:**
- Consumes: Task 1's functions; `data.read_rasters`; today's blocks on the page from the sheet's rasters (`ndimage.label(r.lot >= 0)` is today's lots, so blocks come from `data.block_ids` of the page's `block == 1`).
- Produces: `python -m sfzmaps.solver.hatchnet.areas --series 1921 --sheet 20180102100525362` writes, under `data/solver/experiments/hatch-cnn/areas/`: `table.json` (one `against_blocks` dict per finder) and `<finder>_<sheet>.png`, plus `<finder>_<sheet>_6220002.png` for the window rows 2800-4750, cols 2560-3650.

Finders compared, each from its saved `street_prob`:
- `v1` (`hatch-cnn/streets/`), `v2` (`streets-v2/streets/`), `v2c` (`streets-v2c/streets/`);
- `v1_or_v2c`: boundary where either says street (the larger of the two probabilities).

- [ ] **Step 1: Add `main`** and run it. Expected: four rows in `table.json`, eight pictures.
- [ ] **Step 2: Look at all four `6220002` window pictures and the whole-sheet picture of the best finder.** Check by eye: outline-bounded blocks, the double-line roads (Visitacion, Sunnydale), open land.
- [ ] **Step 3: Send the owner** the table and the pictures, saying what each is for and which finder is nearest to usable, with what is still wrong.
- [ ] **Step 4: Commit** the code.

### After this plan

The owner and the pictures decide the next plan: which finder to build on, and how outlines and double-line roads become boundaries (the spec's synthetic cases for each come first).
