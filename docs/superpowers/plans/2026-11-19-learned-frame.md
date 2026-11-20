# Learned map-area / frame finder: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a two-scale U-Net on the 12 hand-drawn map-area labels, score it on held-out sheets, and draw a
picture of every map page's predicted map area.

**Architecture:**
- **Two passes.**
  - A page pass: a whole-page U-Net at a long side of 1024 px gives each coarse pixel's chance of being map.
  - A pixel pass: a full-resolution U-Net, the reach of PR #70's finder, reads the scan plus the page pass's answer
    and gives each pixel's chance.
- **Held out by sheet, at two levels.**
  - Four outer folds hold out one sheet per series, and every score comes from them.
  - Inside each outer fold, three inner folds do two jobs:
    - they give the page-pass answers the pixel pass trains on;
    - they choose how many training steps to take.
- **Outputs** go to `pipeline/data/frame/`. Nothing is wired into reading, placement or the paper layer.

**Tech Stack:** Python 3.12, PyTorch 2.15 (cpu/mps), numpy, Pillow, scipy.ndimage, typer; pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-learned-frame-design.md` (read it first).

## Global Constraints

- $0: no paid reads, no LLM output as input.
- Inputs are only `page.png` (as scanned: nothing erased, thresholded or cropped), `pipeline/inputs/frame-labels/` and
  `pipeline/inputs/page_kinds.csv`. Never the old crops (`solver/frame.py`, `pages.neatline_mask`, warp outlines),
  never placements.
- One process for every series: no per-series or per-year branch, window, inset or threshold anywhere in `sfzmaps/frame`.
- No series or scale input to the network (PR #70 ruling R6).
- Deterministic: `torch.use_deterministic_algorithms(True)`, seeded numpy and torch, the same output bits on reruns
  (on cpu; on mps alone on the GPU).
- On mps, training and prediction run inside `sfzmaps.gpu.gpu_turn(True, device)`, one exclusive turn per process.
- CPU: `torch.set_num_threads(THREADS)` with `THREADS = 4`; keep machine load around 20.
- Never commit `pipeline/data/`. Code, tests and docs only.
- Licences: no GPL-family dependencies (torch, numpy, scipy and Pillow are fine).

## Review Focus

1. **Pages stored turned:** 1964–1975 are landscape, and some 1921 sections are upside down. Prediction must return a
   map that lines up pixel for pixel with the page as stored. Task 5 test `test_predict_shape_matches_page`.
2. **Colour and grey scans:** the 1948 photos are RGB; others are 8-bit grey. Luminance must be luma in [0, 1] for
   both. Task 2 test `test_load_luminance_rgb_and_grey`.
3. **Odd page sizes:** 1935 p15 is 5100×2480, with sides that aren't multiples of the tile or of the U-Net's stride.
   Task 5 `test_predict_shape_matches_page` uses a 517×1283 page.
4. **Label polygons with notches, many vertices or several pieces** (1960/1's inset notch; 1921's ~70-vertex arcs):
   masks must fill exactly. Task 2 `test_map_mask_notch_and_two_pieces`.
5. **A sheet missing from `page_kinds.csv`:** an error naming the sheet, not a silent skip. Task 2
   `test_map_pages_unknown_sheet_raises`.

---

## File Structure

- `pipeline/src/sfzmaps/nets/__init__.py`, `pipeline/src/sfzmaps/nets/unet.py`: the shared U-Net, moved from PR #70.
- `pipeline/src/sfzmaps/frame/__init__.py`: the package docstring, a short version of the spec.
- `pipeline/src/sfzmaps/frame/data.py`: pages, luminance, labels, masks, the map-page list and the labelled-sheet list.
- `pipeline/src/sfzmaps/frame/folds.py`: outer and inner folds.
- `pipeline/src/sfzmaps/frame/scores.py`: log-loss, the two mean probabilities, edge distance.
- `pipeline/src/sfzmaps/frame/predict.py`: network sizes and whole-page prediction (page pass, then tiled pixel pass).
- `pipeline/src/sfzmaps/frame/train.py`: augmentation, the page-pass and pixel-pass trainers, step choice.
- `pipeline/src/sfzmaps/frame/run.py`: orchestration (folds, final model, all pages), outputs under `data/frame/`.
- `pipeline/src/sfzmaps/frame/look.py`: pictures.
- `pipeline/src/sfzmaps/cli.py`: the `frame` command.
- Tests: `pipeline/tests/test_nets_unet.py`, `test_frame_data.py`, `test_frame_folds.py`, `test_frame_scores.py`,
  `test_frame_train.py`, `test_frame_predict.py`, `test_frame_run.py`; plus `test_cli.py` gains one test.

Run tests from `pipeline/`: `uv run pytest tests/<file> -v`.

---

### Task 1: The shared U-Net

**Files:**
- Create: `pipeline/src/sfzmaps/nets/__init__.py` (empty docstring), `pipeline/src/sfzmaps/nets/unet.py`
- Test: `pipeline/tests/test_nets_unet.py`

**Interfaces:**
- Produces: `UNet(in_ch: int, out_ch: int, width: int = 16, levels: int = 5)`, an `nn.Module` with
  `forward(x: Tensor[n, in_ch, h, w]) -> Tensor[n, out_ch, h, w]`. It needs h and w to be multiples of
  `2 ** (levels - 1)`.
- Produces: `reach(levels: int) -> int`, how far (px) an output sees each way.
- Produces: `multiple(levels: int) -> int` = `2 ** (levels - 1)`.

The body is the form of `HatchNet` in `pipeline/src/sfzmaps/solver/hatchnet/model.py` on the branch `origin/hatch-cnn`
(read it with `git show origin/hatch-cnn:pipeline/src/sfzmaps/solver/hatchnet/model.py`):
- channels `width * min(2**i, 8)`;
- two 3×3 conv + BatchNorm + ReLU per block;
- max-pool down, nearest upsample, skip concat;
- a 1×1 head;
- `reach` is the same formula.

Leave out the hatching-specific constants (NAMES, DISTRICTS, scale inputs). The module docstring says it came from PR
#70's finder.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from sfzmaps.nets.unet import UNet, reach, multiple

def test_shapes():
    net = UNet(in_ch=2, out_ch=1, width=4, levels=3)
    y = net(torch.zeros(2, 2, 16, 24))
    assert y.shape == (2, 1, 16, 24)

def test_reach_matches_hatchnet():
    assert reach(5) == 120          # PR #70's REACH at LEVELS = 5
    assert multiple(9) == 256

def test_same_input_same_output():
    torch.manual_seed(0); a = UNet(1, 1, width=4, levels=3).eval()
    torch.manual_seed(0); b = UNet(1, 1, width=4, levels=3).eval()
    x = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a(x), b(x))
```

(If `reach(5)` in hatchnet is not 120, use the value its formula gives and say so in the commit message.)

- [ ] **Step 2: Run, and expect the import to fail.** `uv run pytest tests/test_nets_unet.py -v` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement `nets/unet.py`** as described above.
- [ ] **Step 4: Run, and expect a pass.** `uv run pytest tests/test_nets_unet.py -v` → 3 passed.
- [ ] **Step 5: Commit.** `git add pipeline/src/sfzmaps/nets pipeline/tests/test_nets_unet.py && git commit -m "nets: shared U-Net, moved from PR #70's street finder"`

---

### Task 2: Pages, labels and masks

**Files:**
- Create: `pipeline/src/sfzmaps/frame/__init__.py`, `pipeline/src/sfzmaps/frame/data.py`
- Test: `pipeline/tests/test_frame_data.py`

**Interfaces:**
- Produces: `SheetId = tuple[str, str]` (series, sheet).
- Produces: `load_luminance(path: Path) -> np.ndarray`. Float32 [0, 1], 1 = paper. Grey is value/255. RGB is Rec. 601
  luma (`PIL convert("L")`, then /255).
- Produces: `ink_weight(lum: np.ndarray) -> np.ndarray` = `1 - lum`.
- Produces: `load_label(series, sheet, root: Path = LABELS) -> list[list[tuple[float, float]]]`, the polygons.
- Produces: `labelled_sheets(root: Path = LABELS) -> list[SheetId]`, sorted; every `<series>/<sheet>.json`.
- Produces: `map_mask(polygons, shape: tuple[int, int], scale: float = 1.0) -> np.ndarray`. Uint8, 1 inside. The
  polygons are drawn in page px times `scale` (PIL `ImageDraw.polygon`, fill 1, outline 1).
- Produces: `map_pages(kinds_csv: Path = PAGE_KINDS, work: Path = WORK) -> list[SheetId]`. Every sheet under `work`
  whose kind is `map`. Raises `KeyError` naming any sheet that has a `page.png` but no row.
- Produces: `page_path(series, sheet, work: Path = WORK) -> Path`.
- Constants:
  - `LABELS = ROOT / "inputs" / "frame-labels"`
  - `PAGE_KINDS = ROOT / "inputs" / "page_kinds.csv"`
  - `WORK = ROOT / "data" / "work"`
  - `OUT = ROOT / "data" / "frame"`
  - `ROOT` is the `pipeline/` directory, as `sfzmaps.config` defines it; reuse it if it exists.

- [ ] **Step 1: Write the failing tests** (synthetic files in `tmp_path`):

```python
def test_load_luminance_rgb_and_grey(tmp_path):
    Image.new("L", (3, 2), 51).save(tmp_path / "g.png")
    Image.new("RGB", (3, 2), (255, 0, 0)).save(tmp_path / "c.png")
    assert np.allclose(load_luminance(tmp_path / "g.png"), 0.2)
    assert np.allclose(load_luminance(tmp_path / "c.png"), 76 / 255, atol=1 / 255)   # PIL luma of pure red

def test_map_mask_notch_and_two_pieces():
    notch = [(0, 0), (6, 0), (6, 2), (10, 2), (10, 10), (0, 10)]
    m = map_mask([notch, [(12, 0), (14, 0), (14, 2), (12, 2)]], (11, 16))
    assert m[1, 8] == 0 and m[5, 8] == 1 and m[1, 3] == 1 and m[1, 13] == 1 and m[5, 13] == 0
    assert map_mask([notch], (6, 8), scale=0.5)[2, 2] == 1

def test_labelled_sheets_and_load_label(tmp_path):
    (tmp_path / "1921").mkdir()
    (tmp_path / "1921" / "a.json").write_text('{"polygons": [[[0,0],[4,0],[4,4]]], "notes": "", "drawn_by": ""}')
    assert labelled_sheets(tmp_path) == [("1921", "a")]
    assert load_label("1921", "a", tmp_path) == [[(0, 0), (4, 0), (4, 4)]]

def test_map_pages_unknown_sheet_raises(tmp_path):
    # work/1921/{a,b}/page.png exist; the csv lists a (map) only  -> KeyError mentioning "1921/b"
    ...
```

Write the last test out fully: make the two `page.png` files, a two-line csv with the header
`series,sheet,kind,checked`, and use `pytest.raises(KeyError, match="1921/b")`. Add
`test_map_pages_keeps_only_maps`: a `text` row is dropped.

- [ ] **Step 2: Run, and expect failures.** `uv run pytest tests/test_frame_data.py -v` → FAIL.
- [ ] **Step 3: Implement `frame/data.py`, and `frame/__init__.py`'s docstring.** The docstring says in about 10
  lines what the package does, points to the spec, and states the global constraints above.
- [ ] **Step 4: Run, and expect a pass.** `uv run pytest tests/test_frame_data.py -v` → 5 passed.
- [ ] **Step 5: Commit.** `git commit -m "frame: pages, labels and masks"` (both files and the test).

---

### Task 3: Folds

**Files:**
- Create: `pipeline/src/sfzmaps/frame/folds.py`
- Test: `pipeline/tests/test_frame_folds.py`

**Interfaces:**
- Consumes: `SheetId` (Task 2).
- Produces: `@dataclass(frozen=True) class Fold: train: tuple[SheetId, ...]; held: tuple[SheetId, ...]`.
- Produces: `split(sheets: list[SheetId], k: int) -> list[Fold]`. The sheets are grouped by series, each series' sheets
  sorted. Fold i holds out each series' i-th sheet (counting modulo that series' size, so a series with fewer than k
  sheets is held out more than once and still never trains on its held-out sheet in that fold). Fold i trains on the
  rest. Raises `ValueError` if any series has fewer than 2 sheets.
- Produces: `OUTER_K = 4`, `INNER_K = 3`.
- Use: outer folds are `split(labelled, OUTER_K)`; the inner folds of an outer fold are `split(list(fold.train), INNER_K)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_outer_folds_hold_each_sheet_once`: 12 sheets, 4 per series in 3 series; `split(..., 4)`. Every sheet is in
    exactly one `held`; each `held` has one sheet per series; `set(train) & set(held) == ∅`; `train ∪ held == all`.
  - `test_inner_folds_of_nine`: `split(nine_sheets, 3)`. Every sheet is held exactly once, with 3 held and 6 trained
    per fold.
  - `test_split_is_deterministic`: two calls give equal results, whatever the order of the input list.
- [ ] **Step 2: Run, and expect failures.** `uv run pytest tests/test_frame_folds.py -v` → FAIL.
- [ ] **Step 3: Implement `folds.py`.**
- [ ] **Step 4: Run, and expect a pass.** → 3 passed.
- [ ] **Step 5: Commit.** `git commit -m "frame: outer and inner folds by sheet"`

---

### Task 4: Scores

**Files:**
- Create: `pipeline/src/sfzmaps/frame/scores.py`
- Test: `pipeline/tests/test_frame_scores.py`

**Interfaces:**
- Produces: `BAND_PX = 60` (the spec's border band).
- Produces: `bands(mask: np.ndarray, band_px: int = BAND_PX) -> tuple[np.ndarray, np.ndarray]`, two boolean arrays:
  - `frame_band`: outside the mask, within `band_px` of it;
  - `map_band`: inside the mask, within `band_px` of its edge.
  - Euclidean distance comes from `scipy.ndimage.distance_transform_edt`.
- Produces: `log_loss(p, mask, w, where) -> float`. The weighted mean over `where` of `-log(p)` where mask = 1 and
  `-log(1 - p)` where mask = 0. `p` is clipped to `[1e-6, 1 - 1e-6]`, a numerical guard only.
- Produces: `frame_called_map(p, w, frame_band) -> float` = the weighted mean of p.
- Produces: `map_called_frame(p, w, map_band) -> float` = the weighted mean of `1 - p`.
- Produces: `edge_distance(p, mask) -> dict`, with keys `median` and `hd95`. These are the symmetric distances between
  the ½-contour of p (the boundary pixels of `p >= 0.5`) and the boundary of `mask`. HD95 follows MapSeg: the 95th
  percentile of the pooled two-way distances. This number is reported on the side only.
- Produces: `sheet_scores(p, mask, lum) -> dict`, with keys:
  - `log_loss_frame_band`, `log_loss_map_band`, `log_loss_all_ink`;
  - `frame_called_map`, `map_called_frame`;
  - `edge_median_px`, `edge_hd95_px`.
  - `w = ink_weight(lum)`.

- [ ] **Step 1: Write the failing tests** (small synthetic arrays):

```python
def test_perfect_prediction():
    mask = np.zeros((100, 100), np.uint8); mask[20:80, 20:80] = 1
    lum = np.full(mask.shape, 0.0, np.float32)           # all ink
    s = sheet_scores(mask.astype(np.float32), mask, lum)
    assert s["log_loss_all_ink"] < 1e-5 and s["frame_called_map"] < 1e-5 and s["edge_hd95_px"] == 0

def test_confident_mistake_costs_more_than_unsure():
    mask = np.zeros((50, 50), np.uint8); mask[10:40, 10:40] = 1
    lum = np.zeros(mask.shape, np.float32)
    wrong, unsure = mask.astype(np.float32), mask.astype(np.float32)
    wrong[5, 5:45] = 0.99; unsure[5, 5:45] = 0.5          # a "rule" outside the map
    fb, _ = bands(mask)
    w = 1 - lum
    assert log_loss(wrong, mask, w, fb) > log_loss(unsure, mask, w, fb) > log_loss(mask.astype(np.float32), mask, w, fb)

def test_edge_distance_of_a_shift():
    mask = np.zeros((100, 100), np.uint8); mask[20:80, 20:80] = 1
    p = np.zeros((100, 100), np.float32); p[23:83, 20:80] = 1
    assert edge_distance(p, mask)["median"] == pytest.approx(3, abs=1)

def test_paper_does_not_count():
    # same mask; a wrong p on pixels with lum == 1 (paper) leaves frame_called_map at 0
    ...
```

Write the last test out: lum is 1 on a strip where p is wrong, and 0 elsewhere.

- [ ] **Step 2: Run, and expect failures.** `uv run pytest tests/test_frame_scores.py -v` → FAIL.
- [ ] **Step 3: Implement `scores.py`.**
- [ ] **Step 4: Run, and expect a pass.** → 4 passed.
- [ ] **Step 5: Commit.** `git commit -m "frame: probabilistic scores (log-loss, mean probabilities) and the MapSeg edge distance"`

---

### Task 5: Whole-page prediction

**Files:**
- Create: `pipeline/src/sfzmaps/frame/predict.py`
- Test: `pipeline/tests/test_frame_predict.py`

**Interfaces:**
- Consumes: `UNet`, `reach`, `multiple` (Task 1).
- Produces: `COARSE_SIDE = 1024`, `COARSE_LEVELS = 9` (`reach(9)` must be at least 1024, so every coarse output
  pixel sees the whole page; assert it in a test), `COARSE_WIDTH = 8`, `FINE_LEVELS = 5`, `FINE_WIDTH = 16`,
  `TILE = 512`.
- Produces: `coarse_input(lum) -> tuple[np.ndarray, float]`. The page resized (Pillow BOX) so its long side is
  `COARSE_SIDE`, then padded with paper (1.0) at the right and bottom to a multiple of `multiple(COARSE_LEVELS)`.
  Returns the array and the scale factor.
- Produces: `new_coarse_net() -> UNet` = `UNet(1, 1, COARSE_WIDTH, COARSE_LEVELS)` and `new_fine_net() -> UNet` =
  `UNet(2, 1, FINE_WIDTH, FINE_LEVELS)`.
- Produces: `predict_coarse(net, lum, device="cpu") -> np.ndarray`. The page pass's probability, resized (bilinear)
  back to the page's full shape. Padding is cropped off first.
- Produces: `predict_tiles(net, lum, prior, device="cpu") -> np.ndarray`. The pixel pass's probability at full
  resolution, the page's shape.
  - Tiles are `TILE`×`TILE`, read with a margin of `reach(FINE_LEVELS)` px on each side (paper-padded at the page
    edges).
  - Only each tile's centre is kept, so tile seams can't show.
  - The page is padded to a multiple of `multiple(FINE_LEVELS)`.
- Produces: `predict_page(coarse_net, fine_net, lum, device="cpu") -> tuple[np.ndarray, np.ndarray]`, returning
  `(coarse_prob, map_prob)`, both float32 of the page's shape.
- Both networks run in `eval()` under `torch.no_grad()`, in float32.

- [ ] **Step 1: Write the failing tests.**
  - `test_predict_shape_matches_page`: lum 517×1283; untrained tiny nets (monkeypatch `TILE = 64`). Both outputs have
    shape (517, 1283) and values in [0, 1].
  - `test_tiling_has_no_seams`: on a random page, `predict_tiles` with `TILE = 64` equals running the network on the
    whole padded page in one piece, to 1e-5. A constant-width tile margin of at least `reach` makes them the same.
  - `test_coarse_reach_covers_page`: `reach(COARSE_LEVELS) >= COARSE_SIDE`.
  - `test_coarse_input_pads_to_multiple`: for a 3300×5100 page, the longest side is a multiple of
    `multiple(COARSE_LEVELS)`, the width is at least 1024, and the factor ≈ 1024/5100.
- [ ] **Step 2: Run, and expect failures.** `uv run pytest tests/test_frame_predict.py -v` → FAIL.
- [ ] **Step 3: Implement `predict.py`.**
- [ ] **Step 4: Run, and expect a pass.** `uv run pytest tests/test_frame_predict.py -v` → 4 passed.
- [ ] **Step 5: Commit.** `git commit -m "frame: whole-page prediction, page pass then seamless tiled pixel pass"`

---

### Task 6: Training both passes

**Files:**
- Create: `pipeline/src/sfzmaps/frame/train.py`
- Test: `pipeline/tests/test_frame_train.py`

**Interfaces:**
- Consumes: `SheetId`, `map_mask`, `ink_weight` (Task 2); `scores.log_loss`, `scores.BAND_PX` (Task 4); from
  Task 5, `coarse_input`, `new_coarse_net`, `new_fine_net`, `TILE`, `predict_coarse` and `predict_tiles` (validation
  curves use them).
- Produces: `@dataclass(frozen=True) class Recipe`, with fields:
  - `steps: int = 4000`, `check_every: int = 250`, `lr: float = 1e-3`, `batch: int = 4`;
  - `paper_weight: float = 0.1`;
  - `seed: int = 0`.

  The step and learning-rate defaults are a starting point. Step choice (below) picks the count actually used.
- Produces: `@dataclass class Page: id: SheetId; lum: np.ndarray; mask: np.ndarray | None; prior: np.ndarray | None`.
  `prior` is the page pass's probability at full resolution: needed by the pixel pass, None for the page pass.
- Produces: `augment(rng: np.random.Generator, *arrays) -> tuple`. The same random quarter turn and mirror flip on every
  array, then a brightness/contrast/gamma change and an optional Gaussian blur (sigma ≤ 1.5 px) on the first array (the
  luminance) only. Scale jitter of ±10% is applied before cropping in the samplers.
- Produces: `train_coarse(pages: list[Page], recipe: Recipe, device: str, val: list[Page] | None = None) -> tuple[UNet, dict]`.
  - Each step draws `batch` whole pages (with replacement, from `rng = np.random.default_rng(recipe.seed)`), augmented.
  - The loss is BCE-with-logits against the mask resized (BOX) to the coarse grid, as a soft target, weighted by
    `paper_weight + ink_weight`.
  - Padding pixels get weight 0.
  - When `val` is given, every `check_every` steps it records the ink-weighted log-loss on `val`
    (`scores.log_loss` over all ink).
  - The dict returned is `{"val_curve": [(step, loss), ...]}`.
- Produces: `train_fine(pages: list[Page], recipe: Recipe, device: str, val: list[Page] | None = None) -> tuple[UNet, dict]`.
  - Input channels are `[lum, prior]`.
  - Each step draws `batch` crops of `TILE`×`TILE` at full resolution. Half are centred on a random pixel within
    `scores.BAND_PX * 4` of the label outline, and half are uniform over the page. All are augmented.
  - The loss is BCE-with-logits against the mask, weighted by `paper_weight + ink_weight`.
  - The validation curve works as in `train_coarse`, with `predict_tiles` from Task 5.
- Produces: `choose_steps(curves: list[list[tuple[int, float]]]) -> int`. The step with the lowest mean validation
  loss across the inner folds' curves. Ties go to the earlier step.
- Determinism: at the start of each trainer, call `torch.manual_seed(recipe.seed)`,
  `torch.use_deterministic_algorithms(True)` and `torch.set_num_threads(THREADS)`. Every draw comes from the one numpy
  generator. On mps, use masked sums rather than boolean indexing in the loss; that is the text detector's rule, see
  `sfzmaps/solver/textnet/train.py`.

- [ ] **Step 1: Write the failing tests** (cpu, tiny synthetic pages):

```python
def toy_page(seed, h=96, w=128):
    """Paper with a 3-px frame rule 10 px in from each edge, plus random 'streets' inside; the mask is the inside of
    the rule."""
    ...

def test_augment_keeps_arrays_aligned():
    rng = np.random.default_rng(0)
    lum = np.random.default_rng(1).random((6, 8)).astype(np.float32); mask = (lum > 0.5).astype(np.uint8)
    l2, m2 = augment(rng, lum, mask.astype(np.float32))
    assert l2.shape == m2.shape  # and m2's pattern equals the same rot/flip of mask (check with np.rot90/np.flip)

def test_fine_pass_learns_the_toy_frame():
    pages = [toy_page(s) for s in range(4)]           # prior = mask blurred, as a stand-in page-pass answer
    net, _ = train_fine(pages, Recipe(steps=300, batch=2, check_every=100), "cpu")
    pg = toy_page(9)
    p = predict_tiles(net, pg.lum, pg.prior)          # Task 5's tiled prediction (small TILE via monkeypatch)
    assert p[rule_px].mean() < 0.2 and p[inside_ink_px].mean() > 0.8

def test_training_is_deterministic():
    a, _ = train_coarse([toy_page(0)], Recipe(steps=5, batch=1), "cpu")
    b, _ = train_coarse([toy_page(0)], Recipe(steps=5, batch=1), "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))

def test_choose_steps():
    assert choose_steps([[(250, 0.5), (500, 0.3), (750, 0.4)], [(250, 0.6), (500, 0.2), (750, 0.2)]]) == 500
```

Shrink `TILE` and `COARSE_SIDE` with `monkeypatch` for the toy tests. `toy_page` returns a `Page` whose `prior` is
the mask blurred (a stand-in for the page pass's answer).

- [ ] **Step 2: Run, and expect failures.** `uv run pytest tests/test_frame_train.py -v` → FAIL.
- [ ] **Step 3: Implement `train.py`.**
- [ ] **Step 4: Run, and expect a pass.** → all pass (the toy learning test may take up to a minute on cpu).
- [ ] **Step 5: Commit.** `git commit -m "frame: page-pass and pixel-pass training, augmentation, step choice"`

---

### Task 7: The run, outputs, pictures and the command

**Files:**
- Create: `pipeline/src/sfzmaps/frame/run.py`, `pipeline/src/sfzmaps/frame/look.py`
- Modify: `pipeline/src/sfzmaps/cli.py`, adding a `frame` command next to `warp` (around line 811)
- Test: `pipeline/tests/test_frame_run.py`, and one test in `pipeline/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_folds(labelled: list[SheetId], recipe: Recipe, device: str, out: Path = OUT) -> dict`. For each outer
  fold:
  1. **Inner folds:**
     - train the page pass on each inner fold's 6 sheets, with validation on its 3;
     - predict those 3, which gives an out-of-fold `prior` for every one of the outer fold's 9 training sheets;
     - choose the page-pass step count with `choose_steps` over the 3 inner curves.
  2. Train the page pass on all 9 for that step count.
  3. **Inner folds again, for the pixel pass:** train on 6 with their out-of-fold priors, validate on 3, choose the
     pixel-pass step count.
  4. Train the pixel pass on all 9 for that step count.
  5. **Predict the outer fold's 3 held-out sheets** with this fold's two networks. Write per sheet:
     - `out/<series>/<sheet>/map_prob.npz`, holding `coarse` and `map` as float16;
     - `scores.json`, from `sheet_scores`.

  Returns and writes `out/folds.json`, containing:
  - per sheet: its scores and its fold;
  - per series and overall: ink-weighted means of each score;
  - each fold's chosen step counts.
- Produces: `train_final(labelled, recipe, device, steps: dict, out: Path = OUT) -> None`.
  - The page pass trains on all 12 sheets for the median of the folds' chosen page-pass steps.
  - The pixel pass trains on all 12. Its priors are each sheet's held-out page-pass prediction from `run_folds`, read
    back from the `.npz` files. Its step count is the median of the folds' chosen pixel-pass steps.
  - Saves `out/model/coarse.pt`, `out/model/fine.pt` and `out/model/recipe.json`.
- Produces: `predict_all(device, out: Path = OUT) -> None`.
  - For every sheet in `data.map_pages()` that is not labelled, writes `out/<series>/<sheet>/map_prob.npz` from the
    final model.
  - Labelled sheets keep their held-out prediction.
- Produces: `look.sheet_picture(lum, map_prob) -> Image`:
  - paper stays white;
  - each ink pixel is coloured from blue (map, p = 1) to red (frame, p = 0), its strength set by the ink weight;
  - the ½ contour is drawn in magenta;
  - longest side 1400 px.
- Produces: `look.series_mosaics(out: Path = OUT) -> list[Path]`. One PNG per series under `out/look/`, every map page
  of the series, each tile labelled with the sheet name. Labelled sheets carry their held-out `log_loss_all_ink`.
- Produces: `look.odd_cases(out: Path = OUT) -> Path`. One PNG at `out/look/odd_cases.png`, larger tiles of:
  - 1948/7;
  - the 1948 page with several framed maps (find it in the mosaics, then list it in `ODD_CASES` with a comment);
  - 1935 p15;
  - 1980 p18 and 1980-1990-zoning p07;
  - 1921 094727628 and 100525362, the duplicate scans.
- Produces: the CLI command
  `sfzmaps frame [--device cpu|mps] [--seed 0] [--steps N] [--only-folds] [--skip-folds]`.
  - It runs `run_folds`, `train_final`, `predict_all` and the pictures in one exclusive GPU turn
    (`gpu.gpu_turn(True, device)`).
  - The default device follows `textnet.train`'s rule: mps if available.
  - `--steps` overrides `Recipe.steps`, the most steps any trainer may take.
  - It prints the overall and per-series scores from `folds.json`.

- [ ] **Step 1: Write the failing tests.**
  - `test_run_folds_never_scores_a_trained_sheet`: monkeypatch the trainers so they record the sheets they were given,
    and return a constant network. Run `run_folds` on 12 toy sheets in 3 toy series, with `tmp_path` as `out`. For
    every fold, the sheets passed to any trainer (outer or inner) are disjoint from that fold's held-out sheets. Every
    sheet has a `scores.json`.
  - `test_final_priors_are_held_out_predictions`: `train_final` reads each sheet's prior from its own fold's `.npz`.
    Check by giving each toy `.npz` a distinct constant and asserting the fine trainer received it.
  - `test_cli_frame_listed` in `test_cli.py`: `sfzmaps --help` lists `frame`.
- [ ] **Step 2: Run, and expect failures.** `uv run pytest tests/test_frame_run.py tests/test_cli.py -k frame -v` → FAIL.
- [ ] **Step 3: Implement `run.py`, `look.py` and the CLI command.**
- [ ] **Step 4: Run, and expect a pass.** → pass. Then run the full suite: `uv run pytest -q` → no new failures.
- [ ] **Step 5: Commit.** `git commit -m "frame: held-out folds, final model, every map page, pictures; sfzmaps frame"`

---

### Task 8: The real run, and showing the owner

No new code, unless the run exposes a bug. A fix for that bug follows Tasks 1–7's test-first steps.

- [ ] **Step 1: Check no other GPU job and no build is running.**
  - Run `pgrep -fl "sfzmaps"` and `uptime`.
  - Wait if a build is running (no commits during a build).
  - Wait if load is above about 20.
- [ ] **Step 2: Run the folds.** `cd pipeline && uv run sfzmaps frame --only-folds 2>&1 | tee data/frame/folds.log`
  (background; it takes the GPU turn).
  - Expected: `data/frame/folds.json` and 12 `scores.json`.
  - Read the per-series log-losses, and look at the 12 held-out pictures before anything else.
- [ ] **Step 3: Check determinism on one fold.**
  - Run the same command for fold 0 alone (add a hidden `--fold 0` option if needed) a second time.
  - The `.npz` outputs of fold 0 must be equal bit for bit.
- [ ] **Step 4: Train the final model and draw every page.** `uv run sfzmaps frame --skip-folds`.
  - Expected: `data/frame/model/`, a `map_prob.npz` for every map page (369), and `data/frame/look/*.png`.
- [ ] **Step 5: Look at every series mosaic and `odd_cases.png` yourself.** Write down what's wrong where; don't fix
  anything yet.
- [ ] **Step 6: Report to the owner.**
  - Send the pictures with `SendUserFile` (a caption on each says what it shows and what to look for).
  - Give the score table: per series, log-loss for frame and map ink in the band, the two mean probabilities, and the
    MapSeg edge distance on the side.
  - List what you saw wrong.
  - The owner decides the next step: more labels, approach B, or wiring in.
- [ ] **Step 7: Commit any fixes.** Commit no data.
