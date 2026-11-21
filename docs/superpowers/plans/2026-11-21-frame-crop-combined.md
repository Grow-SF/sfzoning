# Frame Crop, Combined Design: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every map page, delete the printed frame line, everything outside it, and the legend / compass-rose /
SECTION blocks. Layout comes from a frozen pretrained model and the shape is straight lines.

**Architecture:**
- A frozen DINOv2 backbone plus a small trained head gives each page's layout at coarse scale: map, outside the frame,
  or removed block.
- The straight-line shape decision fits one rotated-rectangle frame with fillets, ruled notches and block shapes to
  that layout.
- Held-out chances come from leaving one sheet out. The decision's three numbers and the temperature are fitted on
  those.
- **After Task 1 (owner, 2026-11-21: "Sgtm"):** the full-resolution network is dropped from the design. In the
  combined test it raised frame ink left in from 1.0% to 3.9%. `frame/model.py` and `frame/learn.py` are deleted in
  Task 6. The exact edge is the subject of Task 1b (the cut lies on paper).

**Tech Stack:** Python 3, PyTorch (mps/cpu), `transformers` (new, Apache-2.0) for DINOv2 ViT-B/14 with registers,
numpy, scipy, opencv, rasterio, shapely, Pillow, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-frame-crop-design.md`. Read parts 1, 4 and 7, "Amendment: removed
blocks", "Amendment 2" (the shape, with its red-team edits) and "Amendment 3" (this design). This plan replaces Tasks
10–13 of `docs/superpowers/plans/2026-11-20-frame-crop.md`; that plan's Tasks 1–9 and 5c are built and stay.

## Global Constraints

- $0: no paid reads, no LLM output. Inputs are only `page.png`, `pipeline/inputs/frame-labels/`,
  `pipeline/inputs/page_kinds.csv`, and the pinned public DINOv2 weights.
- Licences: MIT/BSD/Apache only. `transformers` and the DINOv2 weights are Apache-2.0; add both to `NOTICE` and
  `LICENSING.md`.
- The backbone is `facebook/dinov2-with-registers-base` at a pinned revision. Its safetensors file's sha256 is
  checked before use. Tests never download: they use a tiny stand-in backbone.
- One process for every edition: no per-series rule, window or threshold. Every number is an owner cost, a
  measurement, a bound "the labels' range widened by half", a published default or a computational setting, named
  with a comment saying which.
- Costs: frame or removed-block ink left in is 3× map ink cut (`decide.COST_FRAME = 3.0`, `COST_MAP = 1.0`). Blank
  paper costs nothing.
- Every fitted number (the temperature, λ_notch, β, λ_block) is fitted on chances from sheets the heads did not train
  on.
- Deterministic on one machine: seeded, features extracted on one device for every page. Not bit-equal across
  machines; the code says so.
- One GPU job at a time (`sfzmaps.gpu.gpu_turn(True, "mps")`, one turn per process). At most 4 threads per job.
- Disk: store only the 128-component features (about 2.4 MB per page). Keep one run's per-page maps at a time, with
  chances as uint8. Check `df -h /System/Volumes/Data` before a run over every page.
- Never commit `pipeline/data/`. `frame/` and `nets/` stay in `build.py`'s `CODE_EXCLUDE`.
- Tests test our behaviour only. Commands run from `pipeline/`. The full suite is `uv run pytest -q -n 6`.
- Commit messages end with the session's attribution lines.

## Review Focus

1. **A page with no frame on any side** (1948 photos): the frame is the page rectangle and nothing is cut but blocks.
   Test in Task 4.
2. **A blank or near-blank page, and a page that is all ink:** the decision returns without error; every cost ties, so
   the simplest shape wins. Test in Task 4.
3. **A layout head that calls a wide blank strip inside the frame "outside"** (1948/2, 9, 5a): the fitted side stays
   on the rule. Synthetic test in Task 4; real pages in Tasks 1 and 8.
4. **Page sizes whose long side isn't a multiple of 14 at 111 patches, portrait and landscape:** the feature grid and
   the upsampled chances have exactly the page's aspect and shape. Test in Task 2.
5. **Weights file missing, wrong revision or wrong hash:** a clear error before any work, and no silent download of
   another revision. Test in Task 2.

---

### Task 1: The combined test, before anything is ported (controller or one agent; spike level)

**Files:**
- Create: `pipeline/scripts/spikes/issue71_combined.py` (throwaway; imports `issue71_shape.py` and
  `issue71_dino.py`)

The test covers fold 0's 6 held-out sheets and the 1948 pages 14, 3, 5a, 6a, 8, 2, 9, 4 (4 is unlabelled; 2 and 9
are in fold 0's training set, so their pictures are in-sample and are labelled so).

- [ ] **Step 1: Layout chances.** For each page: the DINOv2 head's chances (1554 px, last 4 layers, conv3 head trained
  on fold 0's 16 sheets, 3 seeds averaged) from the cached features in `data/frame-dino/feat/`, upsampled to the
  page.
- [ ] **Step 2: Decide three ways**, at the shape spike's fitted numbers (λ_notch 1e4, β 3, λ_block 1e3):
  - **(a)** the per-pixel cut on the layout chances;
  - **(b)** the shape decision on the layout chances;
  - **(c)** the shape decision with the full-resolution fold-0 network's chances used within a band of half-width W
    of (b)'s fitted lines, and (b)'s removed blocks bounded by the labelled block sizes × 1.5. Use
    `data/frame-shape/cache/` and, for unlabelled pages, the snapshot at
    `data/frame-arch-fold16/fold-0/folds/fold-0/seed-0/step-04000.pt` with `model.BUCKET = 256`, at T 2.086.

  W is the largest side-position error of (b) against the labels on the 6 held sheets × 1.5; report it.
- [ ] **Step 3: Report** to `.superpowers/sdd/2026-11-20-frame-crop/combined-test-report.md`:
  - per held sheet and pooled: cost, frame_kept, map_cut, and the cost split into layout and edge (100 px from the
    label's outline), for (a), (b), (c) and the region network's per-pixel cut (381,666 pooled; layout 159,954);
  - a side-by-side picture of (a) | (b) | (c) for the 1948 pages;
  - a full-resolution close-up of the 1921 thin rule;
  - seconds per page.
- [ ] **Step 4: Show the owner** the picture and the table (SendUserFile with a purpose caption). **Wait for the
  owner** before Task 2. If (c) doesn't beat (b) at the edge, say so: the edge model's place in the design is then
  the owner's call.

---

### Task 1b: The cut lies on paper (spike; runs beside Tasks 2-5)

**Files:**
- Create: `pipeline/scripts/spikes/issue71_on_paper.py` (throwaway; builds on `issue71_combined.py`)

After Task 1, 90% of the remaining cost is at the edge: the layout sees 50-px patches, so a fitted side can sit a few
px off the rule. Label rule 1 says the outline lies on paper between the innermost rule and the first map ink.

- [ ] **Step 1:** Start from Task 1's best decision (layout + shape + bounded blocks, "W = 0"). For each fitted frame
  side, notch side and block outline, slide it along its normal within one patch width of where the layout put it
  (page px per patch = the page's long side / 111). Choose the position that minimises the layout's expected ink
  cost plus the ink lying under the line itself. Report exactly how the two terms are combined and why; no
  hand-tuned weight (if one is unavoidable, fit it on the held sheets by realised cost and say so).
- [ ] **Step 2:** The same six held sheets and eight 1948 pages. Report frame/legend ink left in and map ink cut, as
  shares of each class's ink, before and after; full-resolution close-ups of the 1921 thin rule, a 1960 rule and
  a 1948 legend's edge, before | after.
- [ ] **Step 3:** Show the owner. If it helps, it becomes a refinement stage in Task 5 (a plan edit, then built with
  tests).

---

### Task 2: The layout backbone and features (`frame/layout.py`, part 1)

**Files:**
- Create: `pipeline/src/sfzmaps/frame/layout.py`
- Modify: `pipeline/pyproject.toml` (add `transformers`), `uv.lock`, `NOTICE`, `LICENSING.md`
- Test: `pipeline/tests/test_frame_layout.py`

**Interfaces:**
- Produces:
  - `MODEL_ID = "facebook/dinov2-with-registers-base"`.
  - `REVISION` and `WEIGHTS_SHA256`: read them from the copy in the HF cache that the spike used, and record both.
  - `PATCH = 14`; `LONG_PATCHES = 111` (spike: 1,554 px); `LAYERS = 4`; `COMPONENTS = 128`.
  - `grid(shape: tuple[int, int]) -> tuple[int, int]`: (rows, cols) of patches. The long side is `LONG_PATCHES`, the
    short side `max(1, round(short / long * LONG_PATCHES))`.
  - `load_backbone(device: str)`: the frozen model in eval mode. Raises `RuntimeError` naming the file if the weights'
    sha256 differs from `WEIGHTS_SHA256`. Downloads only that pinned revision.
  - `raw_features(backbone, lum: np.ndarray, device: str) -> np.ndarray`: float32 `(768 * LAYERS, rows, cols)`. The
    page is resized with Pillow BOX to `grid * PATCH`, grey ×3, ImageNet normalisation. The last `LAYERS` hidden
    states' patch tokens each go through the model's final layer norm, with register and class tokens dropped.
  - `class Reducer`: `fit_update(feats)` accumulates count, sum and outer products over patches; `finish()` gives the
    mean, the standard deviation and the top `COMPONENTS` principal directions of the standardised features;
    `transform(feats) -> float16 (COMPONENTS, rows, cols)`; `save(path)` / `load(path)`.
  - `features_dir(out) -> Path` = `out / "features"`.
  - `extract_all(sids, device, out) -> None`: pass 1 fits the reducer over every page, and pass 2 writes
    `<series>/<sheet>.npy` (float16). It skips pages already written when `reducer.npz` exists and its recorded page
    list matches.

- [ ] **Step 1: Write the failing tests** (a stand-in backbone fixture `tiny_backbone` in `tests/conftest.py`: 2
  layers, 8 dims, patch 14, same output structure; monkeypatch `layout.load_backbone`):

```python
def test_grid_keeps_the_aspect():
    assert layout.grid((6600, 5266)) == (111, 89) and layout.grid((2550, 3312)) == (85, 111)
    assert layout.grid((10, 5000)) == (1, 111)

def test_raw_features_shape(tiny_backbone):
    f = layout.raw_features(tiny_backbone, np.random.default_rng(0).random((700, 500), dtype=np.float32), "cpu")
    assert f.shape == (8 * layout.LAYERS, 111, 79) and f.dtype == np.float32

def test_reducer_whitens_and_orders_components():
    rng = np.random.default_rng(0); x = rng.normal(size=(6, 40, 30)).astype(np.float32); x[0] *= 10; x[1] += 5
    r = layout.Reducer(components=3); r.fit_update(x); r.finish()
    y = r.transform(x).astype(np.float32).reshape(3, -1)
    assert abs(y.mean(1)).max() < 1e-2 and y.var(1)[0] >= y.var(1)[1] >= y.var(1)[2]

def test_reducer_round_trip(tmp_path): ...                      # save/load gives the same transform
def test_wrong_weights_hash_is_refused(monkeypatch, tmp_path): ...   # RuntimeError naming the file, nothing loaded
def test_extract_all_is_resumable(tiny_backbone, tmp_path): ...  # second call extracts nothing
```

- [ ] **Step 2: Run; expect** `ModuleNotFoundError: sfzmaps.frame.layout`.
- [ ] **Step 3: Implement.** Add `transformers` to the dependencies and lock, and add the licence notices.
- [ ] **Step 4: Run the full suite.** Expected: PASS, with no network access in tests.
- [ ] **Step 5: Measure.** On mps: `extract_all` for the 21 labelled pages, seconds per page, and the disk used.
  Report both.
- [ ] **Step 6: Commit** `frame: frozen DINOv2 layout features (pinned weights, 128 components over every page)`.

---

### Task 3: The layout head (`frame/layout.py`, part 2)

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/layout.py`
- Test: `pipeline/tests/test_frame_layout.py`

**Interfaces:**
- Consumes: Task 2's features; `data.class_mask`.
- Produces:
  - `CLASSES = 3` (0 map, 1 outside the frame, 2 removed block) and `combine(zs, temperature=1.0)`, both moved here
    from `frame/model.py` unchanged (with their tests); `calibrate.py` imports `combine` from here.
  - `HIDDEN = 32`; `HEAD_SEEDS = (0, 1, 2)` (spike: layout cost 2.6k–17k across seeds).
  - `HEAD_LR = 1e-3`, `HEAD_MAX_STEPS = 3000`, `HEAD_FLAT = 0.002`: stop when the mean loss over 100 steps improves
    by less than this share. These are computational convergence settings.
  - `patch_targets(cls: np.ndarray, lum: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]`: `(shares (3, rows,
    cols) float32, weight (rows, cols) float32)`. Shares are each patch's area fractions of the three classes (Pillow
    BOX resize of the one-hot planes); weight = 0.1 + the patch's mean ink.
  - `new_head() -> nn.Module`: 3×3 conv (`COMPONENTS` → `HIDDEN`, padding 1) → ReLU → 1×1 conv → 3.
  - `train_head(feats: Sequence[np.ndarray], targets: Sequence[tuple], seed: int, device="cpu") -> nn.Module`: one
    page per step in a seeded order; soft-target cross-entropy weighted by `weight`; Adam.
  - `head_logits(head, feat) -> np.ndarray` `(3, rows, cols)` float32.
  - `page_logits(z: np.ndarray, shape) -> np.ndarray`: the patch logits bilinearly upsampled to the page `(3, h,
    w)`, with align_corners False.
  - `leave_one_out(feats: dict[SheetId, np.ndarray], targets: dict, seeds=HEAD_SEEDS) -> dict[SheetId,
    list[np.ndarray]]`: for each sheet, the patch logits from heads trained on all the others, one per seed.

- [ ] **Step 1: Write the failing tests**

```python
def test_patch_targets_are_area_shares():
    cls = np.zeros((28, 28), np.uint8); cls[:, 14:] = 1; cls[:7, :7] = 2
    s, w = layout.patch_targets(cls, np.ones((28, 28), np.float32), (2, 2))
    np.testing.assert_allclose(s[:, 0, 0], [0.75, 0, 0.25]); np.testing.assert_allclose(s[:, 0, 1], [0, 1, 0])
    np.testing.assert_allclose(w, 0.1)

def test_weight_ignores_the_label(): ...                       # same lum, different cls -> equal weights
def test_head_learns_a_separable_layout():                      # feature 0 marks "outside": held-out page right
    ...
def test_same_seed_same_head(): ...
def test_leave_one_out_never_trains_on_the_held_sheet(monkeypatch): ...   # spy on train_head's inputs
def test_page_logits_shape_and_orientation(): ...               # (3, h, w); a left/right split stays left/right
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: layout head on frozen features (soft patch targets, leave-one-sheet-out)`.

---

### Task 4: The shape decision, frame (`frame/shape.py`, part 1)

Port from `pipeline/scripts/spikes/issue71_shape.py` (the measured, working spike). Its report,
`.superpowers/sdd/2026-11-20-frame-crop/spike-shape-report.md`, section "What I did differently from the brief",
lists ten findings; each is a requirement here.

**Files:**
- Create: `pipeline/src/sfzmaps/frame/shape.py`
- Test: `pipeline/tests/test_frame_shape.py` (synthetic pages: port the spike's `make_synth`)

**Interfaces:**
- Produces:
  - `ANGLE_MAX = 2.5` degrees (labels' 95% tilt 1.56°, widened by half and rounded).
  - `TILT_MAX = 0.45` degrees (labels' 80% parallelism 0.28°, × 1.5).
  - `BOW_MAX = 22` px (labels' 95% bow 15 px, × 1.5).
  - `ARC_MAX = 332` px (largest labelled fillet 221 px, × 1.5).
  - `NOTCH_SPAN_MAX = 0.5` (labelled notches span 23–29% of a side; the spike's bound).
  - `cost_field(p: np.ndarray, lum: np.ndarray) -> np.ndarray`: D = `ink * (COST_MAP * p[0] − COST_FRAME * (p[1] +
    p[2]))`, float32. D > 0 means "cheaper kept". Fix the sign once here and state it in the docstring; the spike
    used the opposite sign in places.
  - `@dataclass class Frame`: `phi` (degrees), the four offsets, per-side `tilt` and `bow`, and four fillet radii.
    `mask(shape) -> bool` gives the kept region before notches and blocks.
  - `frame_search(D, notch_aware: bool) -> Frame`:
    - every angle in ±`ANGLE_MAX`, all four offsets on a 16-px grid, exact per angle: for fixed (top, bottom) the best
      (left, right) separates;
    - refined to 1 px;
    - `notch_aware` scores each rectangle with its best single notch per side (span ≤ `NOTCH_SPAN_MAX`).
  - `refine_sides(frame, D) -> Frame`: tilt and bow per side within the bounds, the offset re-optimised, with the
    columns within a fitted fillet's radius left out; uses forward binning at full resolution (finding 4).
  - `fit_arcs(frame, D) -> Frame`: tangent fillets. One histogram per corner gives every radius: a px at inward
    distances (a, b) is cut iff a + b + √(2ab) < r.
  - **Ties** go to zero tilt, zero bow, the angle nearest 0 and the outermost offset (finding 5).
  - **A side with no frame is the page's own edge** (label rule 5): a side whose best offset is 0 is not rotated by
    φ, tilted or bowed. φ is shared by the framed sides only, and with no framed side φ = 0. (Task 1: with rotated
    page-edge sides, 1948/8 and 1948/9 came out at +2.56° and −2.10° and lost their corners.)

- [ ] **Step 1: Write the failing tests**

```python
def test_recovers_a_rotated_frame():                      # synthetic A without its notch: 1.2° frame, sides ±2 px
    ...
def test_fillets_recovered_within_3px(): ...              # radius 120 -> 117..123; sides not bowed by the arcs
def test_frameless_page_is_the_page_rectangle():          # D >= 0 everywhere -> offsets 0, phi 0, nothing cut
    ...
def test_all_ties_take_the_simplest_shape():              # D == 0 (blank page): phi 0, tilt 0, bow 0, offsets 0
    ...
def test_blank_strip_called_outside_does_not_move_the_side():
    # a framed page whose chances call a 300-px blank strip inside the left rule "outside" (ink 0 there): left side
    # stays at the rule (paper costs nothing; the rule's ink decides)
    ...
def test_all_ink_page_does_not_crash(): ...
def test_partly_framed_page_keeps_its_corners():
    # a rule along the top and left only, tilted 1°: top and left follow it; right and bottom are the page edge
    # exactly (no px cut along them, corners kept); phi within 0.1° of 1°
    ...
```

- [ ] **Step 2: Run; expect** `ModuleNotFoundError`.
- [ ] **Step 3: Implement** by porting the spike's `Canvas`, `_rect_best`, `frame_search`, `refine_sides` and
  `fit_arcs`.
- [ ] **Step 4: Run.** Expected: PASS. Print the seconds per synthetic page; the spike's global search took 1.7 s on
  real pages.
- [ ] **Step 5: Commit** `frame: straight-line frame fit (global rotated rectangle, tilt, bow, fillets)`.

---

### Task 5: The shape decision, notches and blocks (`frame/shape.py`, part 2)

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/shape.py`
- Test: `pipeline/tests/test_frame_shape.py`

**Interfaces:**
- Consumes: Task 4; `decide.Decision`, `decide.cut_level`.
- Produces:
  - `STRIP = 4` px: the strip just inside a notch's inner sides whose ink supports it (spike finding 7). A notch is
    at least `2 * STRIP + 1` px wide.
  - `NOTCH_CAP = 6` notches per page (computational; the labels have at most 3).
  - `NOTCH_ANGLE = 2.46` degrees: a notch's angle around φ (the labels' 1.64° × 1.5).
  - `@dataclass class Params: lam_notch: float; beta: float; lam_block: float`: the three fitted numbers.
  - `@dataclass class BlockBounds: rect_max: tuple[float, float]; ellipse_max: tuple[float, float]`: the labelled
    blocks' largest sides and axes × 1.5. `block_bounds(labelled) -> BlockBounds` measures them from the labels'
    `remove` polygons: a polygon counts as an ellipse when an ellipse fits it better than a rectangle, by area
    overlap.
  - `greedy_notches(D, ink, frame, params) -> list[Notch]`: rotated rectangles anchored at a side. The gain is the Σ
    of −D over the notch plus β × ink in the strip inside its three inner sides; a notch is added while the gain
    exceeds λ_notch.
  - `block_shapes(p, D, kept, frame, bounds, params) -> list[Block]`:
    - candidates are the 8-connected pieces of `p[2] > cut_level()` inside the frame;
    - each gets the better of a rotated rectangle and an ellipse within `bounds`;
    - a block is kept if its gain exceeds λ_block.
  - `decide(p, lum, params, bounds, maps=1) -> tuple[decide.Decision, Frame, list, list]`, in the spike's measured
    order (finding 2), all on the layout's cost field:
    1. `frame_search` from both starts (plain and notch-aware); the lower joint objective wins;
    2. blocks, then notches;
    3. the rectangle re-fitted within ±0.5° and ±32 px, then sides, tilt, bow and fillets, with notch and block px
       zeroed;
    4. blocks re-scored in the final kept region;
    5. `Decision(outside, removed, filled_ink=0.0, dropped_ink, removed_ink)`.

    With `maps > 1`, fall back to `decide.decide(p, lum, maps)` and say so in the log.
  - `side_ink_left(p, lum, frame, strip=STRIP) -> dict[str, float]`: per side, the ink with `p[1] + p[2] >
    cut_level()` in the strip just inside the fitted side (amendment 2's ranking).

- [ ] **Step 1: Write the failing tests** (the spike's synthetic pages A and B give the expected values)

```python
def test_ruled_notch_found_and_unruled_blob_is_not():
    # synthetic A: a ruled box at the bottom side is cut (its three inner sides within 2 px); the same-size region of
    # "outside" chances with no rule along its sides is not, at beta > 0 and lam_notch between the two gains
    ...
def test_notch_may_be_rotated(): ...                      # box turned 1.5° from the frame: found, angle within 0.2°
def test_blocks_are_bounded_by_the_labels():
    # a p2 blob on blank paper 5x the largest labelled block: the fitted ellipse is no larger than bounds.ellipse_max
    ...
def test_map_shaped_false_block_is_refused(): ...         # an L-shaped p2 patch over map ink: no shape gains
def test_decision_is_one_region_with_only_block_holes(): ...
def test_frameless_page_only_loses_its_blocks(): ...
def test_side_ink_left_ranks_a_missed_rule(): ...
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement** by porting `kadane`, `best_notch`, `greedy_notches`, `_rect_on`, `_ellipse_on`,
  `block_shapes`, `accept_blocks` and `pipeline`.
- [ ] **Step 4: Run the full suite.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: ruled notches, bounded block shapes, the full shape decision`.

---

### Task 6: The run, reworked (`frame/run.py`, `calibrate.py`, `cli.py`)

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/run.py`, `frame/scores.py`, `frame/look.py`, `cli.py`
- Test: `pipeline/tests/test_frame_run.py`, `test_frame_scores.py`, `test_cli.py`

**Interfaces:**
- Consumes: Tasks 2–5; the existing `calibrate.*`, `decide.outline`, `write_outputs`.
- Deletes: `frame/model.py`, `frame/learn.py`, `frame/folds.py`, their tests, the `small_model` fixture, and the old
  fold logic in `run.py` (the full-resolution network is out of the design). `nets/unet.py` stays (PR #70 uses it).
- Produces, in `scores.py`: `SPLIT_PX = 100` (a reporting split only). `sheet_scores` gains `cost_layout` and
  `cost_edge`: the cost on ink further than / within `SPLIT_PX` of any class change in the label.
- Produces, in `run.py` (replacing the fold logic where it differs):
  - `GRID = {"lam_notch": (0, 1e3, 1e4, 1e5), "beta": (0, 1, 3, 10), "lam_block": (0, 1e3, 1e4)}`: the search grid
    from the spike (computational).
  - `held_out(labelled, device, out) -> dict`:
    1. `layout.extract_all` over every map page;
    2. `layout.leave_one_out`, giving each sheet's patch logits per seed. Upsample each with `layout.page_logits`,
       then `layout.combine`. One temperature is fitted on ink-weighted samples of them (`calibrate.sample`);
    3. `Params` are chosen by minimum pooled realised cost over `GRID` on the 21 held-out sheets (the chances are
       cached in memory per sheet);
    4. per sheet, the scores of two decisions: the per-pixel cut on the layout chances (`decide.decide`) and the
       shape decision; plus the share of each class's ink left in or cut, pooled means with sheet-bootstrap
       intervals, paired differences against `BASELINE`, and `side_ink_left`;
    5. everything goes to `folds.json` (`params`, `temperature`, `sheets`, `pooled`, `baseline`).
  - `train_final(labelled, device, out, held: dict) -> None`: `HEAD_SEEDS` heads on all labelled sheets. Writes
    `model/` (heads, `reducer.npz` reference, `model.json` with params, temperature and the block bounds).
  - `predict_all(device, out) -> None`: for every unlabelled map page, the layout chances, the decision and
    `write_outputs`. Writes `pages.json`, ranked by `side_ink_left` (highest first) and by dropped ink.
    It checks the free disk first, and refuses below 20 GB with a clear message (a safety margin, computational).
- `cli.py`: `sfzmaps frame [--held-out | --predict-only]` with the same guards as now.
- `look.py`: the fitted frame, notches and blocks are drawn as straight outlines from `outline.json`. The close-ups
  list adds 1948/2 and 1948/9's blank strips.

- [ ] **Step 1: Write the failing tests** (synthetic pages; stub `layout.load_backbone` with the tiny stand-in):

```python
def test_layout_chances_are_held_out(fake_world): ...          # each sheet's head never trained on it
def test_params_minimise_the_pooled_cost(fake_world): ...       # a rigged grid point wins
def test_scores_split_layout_and_edge(): ...                    # ink 150 px from the outline -> layout; 20 px -> edge
def test_predict_refuses_a_full_disk(monkeypatch, fake_world): ...
def test_cli_frame_needs_the_held_out_run_first(...): ...
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement.** Delete the code and tests of the old per-pixel fold path that nothing uses any more.
- [ ] **Step 4: Run the full suite.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: the run on the pretrained layout (leave-one-out, fitted shape numbers, every page)`.

---

### Task 7: Held-out run (controller)

- [ ] **Step 1:** `uv run sfzmaps frame --held-out` on mps. Log the time of each stage (features, heads, grid).
- [ ] **Step 2:** Report to the owner, numbers with their 90% sheet-bootstrap intervals:
  - the share of frame/legend ink left in and of map ink cut, and the cost with its layout/edge split, for both
    decisions and the old region baseline;
  - the fitted params;
  - the temperature and its reliability near ¼;
  - the five worst sheets with pictures;
  - the top of the side-ink ranking.

  Near-ties are judged by looking.
- [ ] **Step 3: Wait for the owner.** Sheets the owner calls wrong become labelling candidates: a batch, then one
  rerun.

### Task 8: Final models and every page (controller)

- [ ] **Step 1:** Check the free disk and delete the previous run's per-page maps. Then run `uv run sfzmaps frame`.
- [ ] **Step 2:** Send the owner every series mosaic, the close-ups, and the tops of both rankings in `pages.json`
  (frame ink left along a side; ink dropped by "one region").

### Task 9: Final review and pull request

- [ ] **Step 1:** A whole-branch review:
  - the deferred minors in both ledgers;
  - dead code from the superseded designs (the per-pixel run path, unused spikes);
  - licence notices;
  - the fresh-clone check (pinned weights download, no local-only paths).
- [ ] **Step 2:** Run the full suite; push; open the PR against `main` with the held-out numbers and pictures. The
  owner merges.
