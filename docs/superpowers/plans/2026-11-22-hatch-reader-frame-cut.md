# Hatch reader on the frame finder's cut: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The 1921 hatch reader (PR #70) takes every sheet outline from the frame finder (#73) as it ships, and the owner sees the reader's output on three scans.

**Architecture:** `hatching.load_crop` reads `data/frame/1921/<sheet>/outside.png` (keep = zero). The crop, sheet list, sideways check, grid window and street-label area all derive from that one mask. The frame finder is not changed.

**Tech stack:** Python, `uv run` from `pipeline/`, torch on mps (one GPU process at a time).

**Spec:** `docs/superpowers/specs/2026-11-19-one-process-hatched-maps-design.md` (crop = printed frame; neatline fallback goes away).

## Global constraints

- Do not edit `pipeline/src/sfzmaps/frame/` or the frame labels.
- No `sfzmaps build`, no `sfzmaps frame` run, no other series. Only the 16 pages of 1921.
- No before/after comparison against the old crop.
- No paid calls. No retraining in this plan.
- Pictures at the scan's own resolution, sent with a caption saying what each is for.
- Do not merge; the owner merges.

## Scope: 1921 only

The hatched maps are 1921, 1929, 1935 and 1948, but the hatch reader has lots, reads and answers only for 1921 today. This plan touches 1921's 16 pages and nothing else.

## Why `--dev` was run, and why it is not in this plan

The frame finder cuts its own training sheets with heads that did not train on them, so those cuts are honest. That is what `sfzmaps frame --dev` does; plain `sfzmaps frame` skips training sheets and copies their `--dev` cuts. Seven of 1921's 16 pages are training sheets, so without `--dev` they had no cut. Both runs are finished: all 16 cuts are in `pipeline/data/frame/1921/` and match #73's run pixel for pixel. Nothing in this plan runs the frame finder again.

## Review focus

- A sheet with no frame cut: `load_crop` raises `FileNotFoundError` naming `sfzmaps frame` (in code; no test yet, add in Task 1).
- A cut whose shape differs from its page: `load_crop` raises `ValueError` (in code; add test in Task 1).
- A sheet scanned on its side: covered by `test_turned_sheet_matches_upright`.

---

### Task 1: Finish adopting the cut (code)

Already done in commit `3e17293`: `hatching.py`, `hatchnet/data.py`, picture legends, both test fixtures.

**Files:**
- Test: `pipeline/tests/test_hatching.py`
- Delete: `pipeline/scripts/spikes/hatch_frame_compare.py` (untracked; before/after only)

- [ ] **Step 1: Add `test_load_crop_refuses_missing_or_misshapen_cut`** in `tests/test_hatching.py`: with a `tmp_path` frame folder, `hatching.load_crop(frame, "s", (10, 10))` raises `FileNotFoundError` when `s/outside.png` is absent, and `ValueError` when the file is 8 x 8.
- [ ] **Step 2: Run** `uv run pytest -q tests/test_hatching.py tests/test_hatchnet_data.py`. Expected: all pass.
- [ ] **Step 3: Delete the comparison leftovers:** the untracked script above, and under `data/solver/experiments/`: `hatch-frame/`, `hatch-cnn/old-crop-as-cut/`, `hatch-cnn/data/1921-oldcrop/`. Keep `hatch-cnn/data/1921-neatline-2026-11-18/` (the data the saved models were trained on).
- [ ] **Step 4: Commit** the test.

### Task 2: The reader's targets on three scans

The 1921 training data is already rebuilt on the cut (`hatch-cnn/data/1921/`, 77 s, 16 sheets).

Sheets: `20180102094149543` (north, legend swatches kept by the cut), `20180102094254223` (dense downtown), `20180102100525362` (scanned on its side).

- [ ] **Step 1: Draw the targets:** `uv run python -m sfzmaps.solver.hatchnet data --series 1921 --look 20180102094149543 20180102094254223 20180102100525362`. Expected: `targets_<sheet>_whole.png` and `_zoom.png` per sheet under `hatch-cnn/look/`.
- [ ] **Step 2: Look at each** (seams, sheet edges, the legend corner on `094149543`), then send them to the owner.

### Task 3: The reader's districts on the same scans

Uses the saved models (`hatch-cnn/districts/models`, trained 11-19 on the old crop). This shows what the reader says on the new cut today; retraining is the owner's next decision.

- [ ] **Step 1: Check no other torch mps process is running** (`pgrep -fl 'hatchnet|sfzmaps frame'`).
- [ ] **Step 2: Predict:** `uv run python -m sfzmaps.solver.hatchnet predict --series 1921 --root data/solver/experiments/hatch-cnn/predict-frame`. Expected: predictions and the district-by-lot pictures under `predict-frame/`.
- [ ] **Step 3: Look at the three sheets' pictures,** then send them to the owner with what each shows.

### Task 4: Publish to the PR

- [ ] **Step 1: Push** `hatch-cnn` (merge of `main`, `3e17293`, Task 1's test, this plan).
- [ ] **Step 2: Edit PR #70's description from the live text:** add that the reader's outlines come from #73's cut, and that #74 covers the rest of the pipeline.

## Not in this plan

- Retraining streets and districts on the new cut.
- 1929, 1935, 1948.
- The frame finder's follow-ups (#72) and the other `frames.json` readers (#74).
