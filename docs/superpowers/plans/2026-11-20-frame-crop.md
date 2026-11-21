# Frame Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every map page, find the printed frame line and mark it and everything outside it for deletion, at full
scan resolution, from one learned network.

**Architecture:** One fully convolutional network (`Folded` U-Net, pixel-unshuffle 8, GroupNorm) reads whole pages at
full resolution and gives each pixel's chance of being outside the frame. A decision step cuts where that chance beats
the owner's cost ratio (frame left in is 3× worse than map cut, so it cuts at q > ¼). It then makes the result one
region with no holes. Four folds × two seeds give held-out chances. Those chances choose the step count and fit one
temperature, and they score every labelled sheet. Two final models, averaged, predict every page.

**Tech Stack:** Python 3, PyTorch (mps/cpu), numpy, scipy, rasterio, shapely, Pillow, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-frame-crop-design.md` (approved by the owner 2026-11-20). Read it first;
each task's "why" is there, part by part.

## Global Constraints

- $0: no paid reads, no LLM output anywhere. Inputs are only `page.png`, `pipeline/inputs/frame-labels/` and
  `pipeline/inputs/page_kinds.csv`.
- Licences: MIT/BSD/Apache only; no GPL-family dependency (PyMaxflow is GPL: not used).
- Full resolution: no step downsamples a page for training or prediction. The ±10% scale jitter is the only
  resampling.
- No hand thresholds, windows or per-series rules. Every number is an owner cost, a measurement, a published default
  or a computational setting, and each is a named constant whose comment says which it is.
- Deterministic: seeded torch and numpy, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(4)`. Same
  bits on rerun.
- One GPU job at a time: every mps run happens inside `sfzmaps.gpu.gpu_turn(True, "mps")`, one turn per process. Keep
  load around 20.
- Never commit `pipeline/data/`. `frame/` and `nets/` stay in `build.py`'s `CODE_EXCLUDE`; `dvc.yaml` unchanged.
- No commits while an `sfzmaps build` runs (`pgrep -f "sfzmaps build"` first).
- Tests test our behaviour only; no tests of PyTorch, numpy or scipy themselves.
- Pipeline commands run from `pipeline/`: `uv run pytest tests/test_frame_*.py tests/test_nets_unet.py -q`; the full
  suite is `uv run pytest -q -n 8`.

## Review Focus

1. **Colour and 1-bit scans** (1948 photos are RGB; 1921 are 1-bit). The network reads luminance, but `erased.png`
   keeps the scan's own mode and colours. Test in Task 8.
2. **Nothing cut, or everything cut** (a frameless photo; a blank page). `decide` returns an empty or full `outside`
   without error, and `outline` returns the page rectangle or `[]`. Test in Task 6.
3. **Page sizes that aren't multiples of 256, and small pages** (2550 × 3312). Outputs have exactly the page's shape.
   Paper padding never gets weight in training. Tests in Tasks 2 and 5.
4. **Non-square pages under quarter turns.** Target and weight turn with the luminance, and padding is added after
   the turn. Test in Task 5.
5. **Pages with several maps, and `maps` missing from the CSV.** A blank `maps` means 1, and `maps = 2` keeps two
   inside components. Tests in Tasks 3 and 6.

---

### Task 1: GroupNorm option in the U-Net

**Files:**
- Modify: `pipeline/src/sfzmaps/nets/unet.py`
- Test: `pipeline/tests/test_nets_unet.py`

**Interfaces:**
- Produces:
  - `UNet(in_ch, out_ch, width=16, levels=5, norm: str = "batch")`
  - `Folded(in_ch, out_ch, fold, width, levels, norm: str = "batch")`
  - `norm` is `"batch"` or `"group"`. `GROUPS = 32` is Wu & He 2018's default; a layer with fewer channels uses one
    group per channel.

- [ ] **Step 1: Write the failing tests**

```python
def test_group_norm_is_batch_independent():
    torch.manual_seed(0)
    net = Folded(1, 1, fold=2, width=4, levels=2, norm="group").train()
    a, b = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    alone = net(a)
    together = net(torch.cat([a, b]))[:1]
    assert torch.allclose(alone, together, atol=1e-6)

def test_batch_norm_default_keeps_its_parameters():
    keys = set(UNet(1, 1, width=4, levels=2).state_dict())
    assert any("running_mean" in k for k in keys)   # PR #70's checkpoints still load

def test_unknown_norm_refused():
    with pytest.raises(ValueError):
        UNet(1, 1, width=4, levels=2, norm="layer")
```

- [ ] **Step 2: Run them; expect a failure** with `TypeError: unexpected keyword argument 'norm'`.
  Run: `uv run pytest tests/test_nets_unet.py -q`
- [ ] **Step 3: Implement.** Thread `norm` through `_conv` and `_block`. For `"group"`, use
  `nn.GroupNorm(min(GROUPS, o), o)`. Raise `ValueError` for any other value. Add a module docstring line.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `nets: GroupNorm option (batch-independent, for one page per step)`.

---

### Task 2: The single-network model; remove the two-pass code

**Files:**
- Create: `pipeline/src/sfzmaps/frame/model.py`
- Delete:
  - `frame/predict.py`, `frame/train.py`, `frame/run.py`, `frame/look.py`;
  - `tests/test_frame_predict.py`, `tests/test_frame_train.py`, `tests/test_frame_run.py`;
  - the `frame` command in `cli.py` and its tests in `tests/test_cli.py` (Task 9 re-adds them).
- Modify:
  - `frame/data.py`: remove `POSITION_CHANNELS`, `EDGE_MAX`, `position_planes`, `stacked_planes`.
  - `tests/test_frame_data.py`: remove their tests.
  - the `position_channels` fixture wherever it lives (grep `tests/`).
  - `frame/__init__.py`: the docstring now describes this design and points to the new spec.
- Test: `pipeline/tests/test_frame_model.py`
- Spike scripts under `scripts/spikes/` that import removed modules stay as they are: they're throwaway, and no test
  runs them.

**Interfaces:**
- Produces, in `frame/model.py`:
  - `FOLD = 8`, `WIDTH = 32`, `LEVELS = 6`: the spike's network, with a reach of 2,000 px (spec part 3).
  - `PAPER = 1.0`.
  - `new_net() -> Folded`: `Folded(1, 1, FOLD, WIDTH, LEVELS, norm="group")`.
  - `multiple() -> int`: `folded_multiple(FOLD, LEVELS)`, which is 256.
  - `pad(lum: np.ndarray) -> np.ndarray`: `lum` paper-padded on the bottom and right up to multiples of `multiple()`.
  - `logits(net: nn.Module, lum: np.ndarray, device: str) -> np.ndarray`: float32, shape `lum.shape`. Eval mode,
    `no_grad`, the net's mode and device restored afterwards.
  - `combine(zs: Sequence[np.ndarray], temperature: float = 1.0) -> np.ndarray`: float32 mean over `zs` of
    `sigmoid(z / temperature)`, which is P(outside).
  - `prob(nets: Sequence[nn.Module], lum, device, temperature=1.0) -> np.ndarray`:
    `combine([logits(n, lum, device) for n in nets], temperature)`.

- [ ] **Step 1: Write the failing tests** (monkeypatch `model.FOLD = 2`, `model.LEVELS = 2`, `model.WIDTH = 4` so
  they run on cpu in seconds):

```python
def test_logits_have_the_page_shape(small_model):          # 300 x 457: not a multiple
    lum = np.random.default_rng(0).random((300, 457), dtype=np.float32)
    assert model.logits(model.new_net(), lum, "cpu").shape == (300, 457)

def test_pad_is_paper_and_a_multiple(small_model):
    p = model.pad(np.zeros((5, 9), np.float32))
    assert p.shape[0] % model.multiple() == 0 and p.shape[1] % model.multiple() == 0
    assert (p[:5, :9] == 0).all() and (p[5:, :] == model.PAPER).all() and (p[:, 9:] == model.PAPER).all()

def test_prob_is_the_mean_of_tempered_sigmoids(small_model):
    torch.manual_seed(0); a, b = model.new_net(), model.new_net()
    lum = np.full((40, 40), 0.7, np.float32)
    za, zb = model.logits(a, lum, "cpu"), model.logits(b, lum, "cpu")
    expect = (1 / (1 + np.exp(-za / 2)) + 1 / (1 + np.exp(-zb / 2))) / 2
    np.testing.assert_allclose(model.prob([a, b], lum, "cpu", temperature=2.0), expect, rtol=1e-5)

def test_logits_restore_train_mode(small_model):
    net = model.new_net().train()
    model.logits(net, np.ones((8, 8), np.float32), "cpu")
    assert net.training
```

- [ ] **Step 2: Run; expect** `ModuleNotFoundError: sfzmaps.frame.model`.
  Run: `uv run pytest tests/test_frame_model.py -q`
- [ ] **Step 3: Implement `model.py`.** Then delete the listed files and the `frame` CLI command, and strip the
  position planes from `data.py`.
- [ ] **Step 4: Run the full suite.** Expected: everything passes; nothing imports a deleted module.
  Run: `uv run pytest -q -n 8`
- [ ] **Step 5: Commit** `frame: one full-resolution network (model.py); remove the two-pass code`.

---

### Task 3: Label check, page map counts, labels without holes

**Files:**
- Create: `pipeline/src/sfzmaps/frame/labels.py`
- Modify:
  - `frame/data.py`;
  - `pipeline/inputs/page_kinds.csv`: add the column `maps`, blank in every row;
  - `cli.py`: new command `frame-labels`.
- Test:
  - `pipeline/tests/test_frame_labels.py`
  - `pipeline/tests/test_frame_data.py`

**Interfaces:**
- Produces, in `frame/data.py`:
  - `load_label(series, sheet, root=LABELS) -> list[Polygon]`: polygons only.
  - `label_holes(series, sheet, root=LABELS) -> list[Polygon]`: the file's `holes`, for Task 4's review; Task 4
    deletes it.
  - `inside_mask(polygons, shape) -> np.ndarray`: bool, `map_mask(polygons, shape) == 1`.
  - `map_counts(kinds_csv=PAGE_KINDS) -> dict[SheetId, int]`: from column `maps`, where blank means 1.
- Produces, in `frame/labels.py`:
  - `STRETCH_PX = 50`: a presentation unit for listing; it decides nothing.
  - `@dataclass(frozen=True) class Stretch: series: str; sheet: str; start: tuple[int, int]; end: tuple[int, int];
    ink: float`, with `start`/`end` as (x, y) page px.
  - `outline_stretches(lum, polygons, series, sheet) -> list[Stretch]`:
    - each polygon ring sampled at 1 px steps (shapely `LineString.interpolate`), rounded to px;
    - px on the page's outermost row or column dropped (a frameless side follows the page edge);
    - consecutive runs of `STRETCH_PX` px, `ink` = Σ(1 − lum) over the run's px;
    - sorted by `ink`, highest first.
  - `qa_sheet(lum, polygons, stretches, path) -> Path`: one full-resolution crop per stretch (the stretch's bounding
    box plus 40 px each way, never shrunk), with the outline drawn 1 px in magenta and the stretch's ink as a caption,
    stacked vertically.
- Produces, in `cli.py`: `sfzmaps frame-labels [--top N]` (default 30). For every labelled sheet it:
  - writes `data/frame/labels-qa/<series>/<sheet>.png` from the top N stretches;
  - prints `series/sheet total-outline-ink top-stretch-ink`, one line per sheet, highest first.

  It uses no GPU.

- [ ] **Step 1: Write the failing tests**

```python
def test_stretch_on_a_rule_ranks_first():
    lum = np.ones((200, 200), np.float32); lum[:, 100] = 0.0              # a printed rule at x = 100
    poly = [(20, 20), (100, 20), (100, 180), (20, 180)]                   # right edge on the rule
    st = labels.outline_stretches(lum, [poly], "s", "h")
    assert st[0].ink > 40 and min(x for x, _ in (st[0].start, st[0].end)) >= 99
    assert st[-1].ink == 0                                                # edges on paper

def test_page_edge_is_not_listed():
    lum = np.zeros((100, 100), np.float32)                                # all ink
    st = labels.outline_stretches(lum, [[(0, 0), (99, 0), (99, 99), (0, 99)]], "s", "h")
    assert st == []

def test_map_counts_blank_means_one(tmp_path):
    p = tmp_path / "k.csv"
    p.write_text("series,sheet,kind,checked,maps\nA,1,map,x,\nA,2,map,x,2\nA,3,text,x,\n")
    assert data.map_counts(p) == {("A", "1"): 1, ("A", "2"): 2, ("A", "3"): 1}

def test_load_label_returns_polygons_and_holes_separately(tmp_path):  # adapt the existing load_label test
    ...
```

- [ ] **Step 2: Run; expect failures** (`labels` missing; `map_counts` missing).
- [ ] **Step 3: Implement**, update the old `load_label` tests, and add the `maps` column.
- [ ] **Step 4: Run.** Expected: PASS. Then run `uv run sfzmaps frame-labels --top 5` once and paste its ranking into
  the task report (no GPU).
- [ ] **Step 5: Commit** `frame: label outline check (ranked ink under each outline), map counts per page`.

---

### Task 4: Fix the labels (hand work, by the controller; runs beside Tasks 5–9)

**Files:**
- Modify: every `pipeline/inputs/frame-labels/*/*.json`, `pipeline/inputs/frame-labels/README.md`
- Create: `pipeline/inputs/frame-labels/1948/2.json`
- Modify: `frame/data.py` (delete `label_holes`; `load_label` raises `ValueError` if a file has `holes`)
- Test: `pipeline/tests/test_frame_data.py`

**Interfaces:**
- Produces: labels with `polygons` only. Every ruled-off box at the margin is a notch together with the blank paper
  between it and the frame (owner answer 1). Floating blocks inside the frame are inside.

- [ ] **Step 1: Update `README.md` rules 3 and 4.**
  - **Rule 3:** every ruled-off box at the margin, touching the frame or standing clear of it, is outside, cut
    together with the paper between it and the frame.
  - **Rule 4:** floating blocks inside the frame (notes, titles, compass roses, place names) are inside. There are no
    holes.
  - **Rule 1:** the edge lies on paper between the innermost rule and the first map ink; it touches ink only where
    map ink touches the rule, and there it follows the rule's inner edge.
- [ ] **Step 2: Review every label's holes** with the label tool
  (`uv run python scripts/spikes/issue71_label_tool.py zoom ...`).
  - A ruled box at the margin becomes a notch: 1975 p02 and 1980-1990-zoning p01 (the ZONING MAP INDEX inset), and
    any others found.
  - Every other hole is deleted.
  - Record each decision in the file's `notes`.
- [ ] **Step 3: Fix the outlines on ink.** Run `uv run sfzmaps frame-labels --top 30` and work through each sheet's
  QA sheet from the top. Move or add points until the stretches left on ink are only where map ink truly touches
  frame ink. Rerun until the top stretches are all of that kind.
- [ ] **Step 4: Label 1948/2** (framed photo) by the README rules, then run its QA.
- [ ] **Step 5: Make `load_label` refuse holes.** Add the test, then delete `label_holes`.

```python
def test_no_label_has_holes():
    for series, sheet in data.labelled_sheets(REPO_LABELS):   # the repo's real labels
        assert "holes" not in json.loads((REPO_LABELS / series / f"{sheet}.json").read_text())
```

  `REPO_LABELS` is `Path(__file__).parents[1] / "inputs" / "frame-labels"`, because conftest points `INPUTS_DIR` into
  `tmp_path`.
- [ ] **Step 6: Show the owner.** Send the QA sheets of the five worst sheets before and after (SendUserFile, with a
  purpose caption) and the new 1948/2 outline. **Wait for the owner's look before Task 10.**
- [ ] **Step 7: Commit** the labels, the README and the test, separately from code:
  `issue #71: ink-clean labels; margin insets are notches, no holes; label 1948/2`.

---

### Task 5: Whole-page training (`learn.py`)

**Files:**
- Create: `pipeline/src/sfzmaps/frame/learn.py`
- Test: `pipeline/tests/test_frame_learn.py`

**Interfaces:**
- Consumes: `model.new_net`, `model.multiple`, `model.PAPER`, `model.prob` (Task 2); `data.SheetId`.
- Produces, in `frame/learn.py`:
  - Constants copied from the deleted `train.py` with their values and comments: `THREADS = 4`,
    `SCALE_JITTER = 0.1`, `BLUR_SIGMA = 1.5`, `BLUR_SHARE = 0.5`, `BRIGHTNESS = 0.1`, `CONTRAST = 0.2`,
    `GAMMA = 1.5`, `BLUR_TRUNCATE = 4.0`, `WEIGHT_EPS = 1e-12`, `PREFETCH = 2`.
  - Augmentation and plumbing copied unchanged from the deleted `train.py`: `Aug`, `draw_aug`, `_blur`, `apply_aug`,
    `_settle`, `_loss`, `_batches`. Use `git show HEAD~N:pipeline/src/sfzmaps/frame/train.py` from before Task 2.
  - `@dataclass(frozen=True) class Recipe: steps: int = 4000; check_every: int = 1000; lr: float = 1e-3;
    paper_weight: float = 0.1; seed: int = 0; prefetch: bool = True`. The learning rate is constant, with no schedule:
    the held-out score at step s is then exactly the score of "train s steps", which the final model repeats (spec
    part 6). Checks every 1,000 steps give the spec's four candidate step counts (1,000 to 4,000).
  - `@dataclass class Page: id: SheetId; lum: np.ndarray; inside: np.ndarray | None`. `lum` is float32; `inside` is
    bool, the label's map area.
  - `side_depths(inside: np.ndarray) -> tuple[int, int, int, int]`: (top, bottom, left, right). Each is the median
    over that side's lines of the distance in px from the page edge to the first inside px; lines with no inside px
    are left out, and a side with none gives 0.
  - `draw_cut(rng, depths) -> tuple[int, int, int, int]`: `rng.integers(0, d + 1)` per side, in the order top,
    bottom, left, right.
  - `make_sample(page, cut, scale, aug, paper_weight, device) -> tuple[Tensor, Tensor, Tensor]`: `(x, target, weight)`,
    each `(1, 1, H, W)`. The steps:
    1. cut the page by `cut`;
    2. scale by `scale` (bilinear for luminance, nearest-exact for `inside`);
    3. `apply_aug(aug, lum, target, weight)`;
    4. pad to multiples of `model.multiple()`.

    `target` is `1 - inside`. `weight` is `paper_weight + (1 - lum)` from the luminance before the photometric
    changes, 0 on padding. It never depends on the label.
  - `held_loss(net, held: Sequence[Page], device) -> tuple[float, float]`: (pooled ink-weighted log-loss of
    `model.prob([net], ...)` against `~inside` over all px weighted by `1 - lum`, total ink).
  - `train(pages, recipe, device, held=(), snapshots: Path | None = None, label: str = "") -> tuple[nn.Module, dict]`:
    - Each step draws one page uniformly, then its cut, scale and `Aug`, in that order from one generator; one Adam
      step on `_loss`.
    - At every multiple of `check_every`, it logs `"<label> step s/n: train <loss>[, held <loss>]"` and saves
      `snapshots / f"step-{s:05d}.pt"` when `snapshots` is set.
    - It returns the net (eval, cpu) and `{"train_curve": [[s, loss]], "held_curve": [[s, loss, ink]]}`.

- [ ] **Step 1: Write the failing tests** (small model via monkeypatch, as in Task 2; cpu):

```python
def test_weight_ignores_the_label():
    lum = np.random.default_rng(0).random((60, 70), dtype=np.float32)
    a = learn.Page(("s", "a"), lum, np.zeros((60, 70), bool))
    b = learn.Page(("s", "b"), lum, np.ones((60, 70), bool))
    aug = learn.Aug(1, True, 1.0, 1.0, 0.0, 0.0)
    wa = learn.make_sample(a, (3, 0, 2, 5), 1.0, aug, 0.1, "cpu")[2]
    wb = learn.make_sample(b, (3, 0, 2, 5), 1.0, aug, 0.1, "cpu")[2]
    assert torch.equal(wa, wb)

def test_padding_weighs_nothing_after_a_turn():          # non-square page, quarter turn
    pg = learn.Page(("s", "h"), np.full((60, 70), 0.5, np.float32), np.ones((60, 70), bool))
    x, t, w = learn.make_sample(pg, (0, 0, 0, 0), 1.0, learn.Aug(1, False, 1, 1, 0, 0), 0.1, "cpu")
    assert int((w > 0).sum()) == 60 * 70 and w[0, 0, :70, :60].min() > 0

def test_target_turns_with_the_page():
    inside = np.zeros((60, 70), bool); inside[10:30, 5:20] = True
    pg = learn.Page(("s", "h"), np.where(inside, 0.2, 1.0).astype(np.float32), inside)
    x, t, _ = learn.make_sample(pg, (0, 0, 0, 0), 1.0, learn.Aug(1, True, 1, 1, 0, 0), 0.1, "cpu")
    np.testing.assert_array_equal((t[0, 0] == 0).numpy()[:70, :60], np.rot90(inside[:, ::-1], 1))

def test_side_depths_and_cut():
    inside = np.zeros((100, 120), bool); inside[10:80, 5:120] = True
    assert learn.side_depths(inside) == (10, 20, 5, 0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = learn.draw_cut(rng, (10, 20, 5, 0))
        assert 0 <= c[0] <= 10 and 0 <= c[1] <= 20 and 0 <= c[2] <= 5 and c[3] == 0

def test_same_seed_same_weights(tmp_path): ...       # 3 steps twice -> equal state_dicts
def test_snapshots_and_curves(tmp_path): ...         # steps 4, check_every 2 -> step-00002.pt, step-00004.pt;
                                                     # held_curve has 2 rows
def test_learns_a_printed_frame():                   # 64 x 64 pages, a 3 px dark rule 6 px in, inside = within it;
    ...                                              # 150 steps: last held loss < half the first
```

- [ ] **Step 2: Run; expect** `ModuleNotFoundError: sfzmaps.frame.learn`.
  Run: `uv run pytest tests/test_frame_learn.py -q`
- [ ] **Step 3: Implement `learn.py`.** The cpu side (a prefetch thread via `_batches`) draws the page index, cut,
  scale and `Aug`, and cuts the arrays. The device side scales, augments, pads and steps.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Measure one real step** (mps, inside `gpu_turn`): 20 steps on the labelled pages, ms per step logged.
  Put the number in the report; the spec expects about 120–200 ms.
- [ ] **Step 6: Commit** `frame: whole-page training at full resolution, constant rate, snapshots and held-out curve`.

---

### Task 6: The decision (`decide.py`)

**Files:**
- Create: `pipeline/src/sfzmaps/frame/decide.py`
- Test: `pipeline/tests/test_frame_decide.py`

**Interfaces:**
- Produces, in `frame/decide.py`:
  - `COST_FRAME = 3.0` (owner 2026-11-20: frame ink left in is 3× worse than map ink cut), `COST_MAP = 1.0`.
  - `cut_level(cost_frame=COST_FRAME, cost_map=COST_MAP) -> float`: `cost_map / (cost_frame + cost_map)`.
  - `@dataclass(frozen=True) class Decision: outside: np.ndarray; filled_ink: float; dropped_ink: float`.
  - `decide(q, lum, maps=1, level=None) -> Decision`, with `level=None` meaning `cut_level()`:
    1. `cut = q > level` (strict);
    2. `outside` = the 4-connected components of `cut` touching the page border. The rest of `cut` becomes inside,
       and `filled_ink` = Σ(1 − lum) of those px;
    3. of the inside px's 8-connected components, keep the `maps` largest by px count (ties go to the lower
       `scipy.ndimage.label` index). The others become outside, and `dropped_ink` = their Σ(1 − lum);
    4. with no inside px, everything is outside.
  - `outline(outside: np.ndarray) -> list[list[tuple[float, float]]]`: the inside region's exterior rings in px-corner
    (x, y) coordinates. Built with `rasterio.features.shapes(..., connectivity=8)` on `~outside`, each ring passed
    through shapely `simplify(0)` to drop collinear vertices.

- [ ] **Step 1: Write the failing tests**

```python
def test_cut_level_is_the_cost_ratio():
    assert decide.cut_level(3.0, 1.0) == 0.25

def test_level_itself_is_not_cut():
    q = np.full((5, 5), 0.25); assert not decide.decide(q, np.zeros((5, 5)), level=0.25).outside.any()

def test_hole_inside_the_map_is_filled():
    q = np.zeros((50, 50)); q[:3] = 1; q[20:25, 20:25] = 1          # top margin + a blob mid-map
    lum = np.zeros((50, 50))
    d = decide.decide(q, lum)
    assert d.outside[:3].all() and not d.outside[20:25, 20:25].any() and d.filled_ink == 25

def test_island_in_the_margin_is_dropped():
    q = np.ones((50, 50)); q[10:40, 10:40] = 0; q[2, 2] = 0           # one stray px called inside
    d = decide.decide(q, np.zeros((50, 50)))
    assert d.outside[2, 2] and d.dropped_ink == 1

def test_two_maps_kept_only_when_listed():
    q = np.ones((40, 80)); q[5:35, 5:35] = 0; q[5:35, 45:75] = 0
    assert decide.decide(q, np.ones((40, 80)), maps=1).outside[20, 60]
    assert not decide.decide(q, np.ones((40, 80)), maps=2).outside[20, 60]

def test_diagonal_touch_does_not_reach_the_border():
    q = np.zeros((10, 10)); q[1, 1] = 1                               # touches the corner px only diagonally
    assert not decide.decide(q, np.zeros((10, 10))).outside[1, 1]

def test_all_cut_and_none_cut():
    assert decide.decide(np.ones((8, 8)), np.zeros((8, 8))).outside.all()
    assert not decide.decide(np.zeros((8, 8)), np.zeros((8, 8))).outside.any()

def test_outline_reproduces_the_region():
    outside = np.ones((40, 50), bool); outside[5:35, 5:45] = False; outside[5:15, 30:45] = True   # L-shape
    rings = decide.outline(outside)
    raster = rasterio.features.rasterize([(Polygon(r), 1) for r in rings], out_shape=outside.shape)
    np.testing.assert_array_equal(raster.astype(bool), ~outside)
    assert len(rings[0]) <= 7                                         # 6 corners (+ closing point), no collinear ones
    assert decide.outline(np.ones((8, 8), bool)) == []
    assert len(decide.outline(np.zeros((8, 8), bool))[0]) <= 5          # nothing cut: the page rectangle
```

- [ ] **Step 2: Run; expect** `ModuleNotFoundError`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: the cut (owner cost ratio) made one region with no holes; outline polygons`.

---

### Task 7: Scores and calibration

**Files:**
- Rewrite: `pipeline/src/sfzmaps/frame/scores.py`
- Create: `pipeline/src/sfzmaps/frame/calibrate.py`
- Test:
  - `pipeline/tests/test_frame_scores.py` (rewrite)
  - `pipeline/tests/test_frame_calibrate.py`

**Interfaces:**
- Consumes: `decide.COST_FRAME`, `decide.COST_MAP` (Task 6); `model.combine` (Task 2).
- Produces, in `frame/scores.py`:
  - `EPS = 1e-6`: numerical clip for log-loss.
  - `BOOT_N = 10_000`: Monte Carlo resamples; computational.
  - `sheet_scores(q, outside, inside, lum, cost_frame=COST_FRAME, cost_map=COST_MAP) -> dict`, with keys:
    - `frame_kept`: Σ(1 − lum) over `~inside & ~outside` (ink px);
    - `map_cut`: Σ(1 − lum) over `inside & outside`;
    - `cost`: `cost_frame * frame_kept + cost_map * map_cut`;
    - `frame_kept_share`, `map_cut_share`: each over all of its class's ink, NaN when the class has none;
    - `log_loss`: of `q` against `~inside`, weighted by `1 - lum`, over all px;
    - `hd95_px`: MapSeg's HD95, the 95th percentile of the pooled two-way distances between the boundaries of
      `~outside` and `inside`, using the old `edge_distance` method. NaN when either has no boundary.
  - `bootstrap_mean(values, n=BOOT_N, seed=0) -> tuple[float, float, float]`: mean and 5th/95th percentiles of the
    means of sheets resampled with replacement. NaNs are dropped first.
  - `paired_difference(a: dict[SheetId, float], b: dict[SheetId, float], n=BOOT_N, seed=0) -> tuple[float, float, float]`:
    `bootstrap_mean` of a − b over the sheets in both.
- Produces, in `frame/calibrate.py`:
  - `N_CAL = 200_000`: px per sheet. A computational setting: its sampling error is far below the spread between
    sheets.
  - `CAL_BOOT = 200`: computational.
  - `sample(zs: Sequence[np.ndarray], outside_label: np.ndarray, lum, rng, n=N_CAL) -> tuple[np.ndarray, np.ndarray]`:
    `(z (seeds, n) float32, y (n,) bool)`, px drawn with replacement with probability proportional to `1 - lum`.
  - `fit_temperature(samples: list[tuple[np.ndarray, np.ndarray]]) -> float`: minimises the pooled mean log-loss of
    `model.combine(z, T)` against `y`, via `scipy.optimize.minimize_scalar(method="bounded")` on log T in [−3, 3].
    The search bounds are numerical; hitting one is logged as a warning.
  - `temperature_interval(samples, n=CAL_BOOT, seed=0) -> tuple[float, float]`: 5th/95th percentiles of refits on
    sheets resampled with replacement.
  - `reliability(samples, temperature, bins=10) -> list[dict]`: one row per equal-width bin of `combine(z, T)`, with
    `lo`, `hi`, `mean_q`, `share_outside`, `n`. A reporting format only.

- [ ] **Step 1: Write the failing tests**

```python
def test_sheet_scores_counts_kept_and_cut_ink():
    inside = np.zeros((10, 10), bool); inside[2:8, 2:8] = True
    outside = ~inside; outside[2, 2:8] = True; outside[0, 0] = False      # cut a map row; keep one frame px
    lum = np.zeros((10, 10), np.float32)
    s = scores.sheet_scores(np.where(outside, 1.0, 0.0), outside, inside, lum)
    assert s["map_cut"] == 6 and s["frame_kept"] == 1 and s["cost"] == 3 * 1 + 6

def test_log_loss_of_a_constant_quarter():
    inside = np.zeros((4, 4), bool); inside[:2] = True
    s = scores.sheet_scores(np.full((4, 4), 0.25), np.zeros((4, 4), bool), inside, np.zeros((4, 4)))
    assert s["log_loss"] == pytest.approx(-(np.log(0.75) + np.log(0.25)) / 2)

def test_hd95_of_a_shifted_square():
    inside = np.zeros((60, 60), bool); inside[10:50, 10:50] = True
    outside = np.ones((60, 60), bool); outside[13:53, 10:50] = False
    assert scores.sheet_scores(np.zeros((60, 60)), outside, inside, np.zeros((60, 60)))["hd95_px"] == pytest.approx(3, abs=0.5)

def test_paired_difference_sign_and_interval():
    a = {("s", str(i)): 1.0 + 0.1 * i for i in range(20)}; b = {k: v - 0.5 for k, v in a.items()}
    m, lo, hi = scores.paired_difference(a, b)
    assert m == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)

def test_temperature_is_recovered():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 4, (2, 200_000)).astype(np.float32); z[1] = z[0]          # two identical "seeds"
    y = rng.random(200_000) < 1 / (1 + np.exp(-z[0] / 2))
    assert calibrate.fit_temperature([(z, y)]) == pytest.approx(2.0, rel=0.05)

def test_reliability_of_calibrated_chances():
    rng = np.random.default_rng(1); z = rng.normal(0, 2, (1, 100_000)).astype(np.float32)
    y = rng.random(100_000) < 1 / (1 + np.exp(-z[0]))
    for row in calibrate.reliability([(z, y)], 1.0):
        if row["n"] > 2000:
            assert abs(row["mean_q"] - row["share_outside"]) < 0.03
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement both modules.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: ink scores (kept/cut/cost, HD95), sheet bootstrap, temperature and reliability`.

---

### Task 8: The run (`run.py`)

**Files:**
- Create: `pipeline/src/sfzmaps/frame/run.py`
- Test: `pipeline/tests/test_frame_run.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 5, 6 and 7 by the names above; `folds.split`, `folds.OUTER_K`.
- Produces, in `frame/run.py`:
  - `SEEDS = (0, 1)`;
  - `RATIOS_SHOWN = (2.0, 3.0, 10.0)`: the cost ratios reported to the owner; the decision itself uses
    `decide.COST_FRAME`;
  - `BASELINE = DATA_DIR / "frame-cmp-none"`: the two-pass model's held-out `map_prob.npz` (key `map` = P(inside));
  - file names: `FOLDS_FILE = "folds.json"`, `MODEL_DIR = "model"`, `HELD_FILE = "held_logits.npz"` (keys
    `z0`, `z1`, float16), `OUTSIDE_PNG = "outside.png"` (mode `"1"`), `PROB_FILE = "outside_prob.npz"` (key `q`,
    float16), `ERASED_PNG = "erased.png"`, `OUTLINE_FILE = "outline.json"` (`{"polygons": [...], "maps": n}`),
    `PAGES_FILE = "pages.json"`.
  - `sheet_dir(out, sid) -> Path`: `out / series / sheet`.
  - `labelled_pages(labelled) -> dict[SheetId, learn.Page]`.
  - `choose_step(curves: Sequence[Sequence[tuple[int, float, float]]]) -> int`: argmin over check steps of the
    ink-pooled held log-loss across all curves. Ties go to the earlier step.
  - `run_folds(labelled, recipe, device, out=OUT, only: int | None = None) -> dict`:
    - For each fold k and seed s, it trains on the fold's sheets with `held` = its held sheets and
      `snapshots = out/"folds"/f"fold-{k}"/f"seed-{s}"`.
    - It then sets `step = choose_step(...)` and saves `HELD_FILE` from each seed's snapshot at `step`. It fits T
      (with its interval and the reliability table) on all held sheets, and per held sheet runs `decide` (with
      `map_counts`), `write_outputs` and `sheet_scores`, at the 3:1 cut and at each of `RATIOS_SHOWN`.
    - It also computes per-sheet seed disagreement: the ink-weighted share of px where seed 0 alone and seed 1 alone
      decide differently at T.
    - Baseline: q = 1 − `map` for the labelled sheets found under `BASELINE`, scored the same way with no
      temperature, giving paired differences (new − old).
    - Everything goes to `folds.json`: `recipe`, `step`, `curves`, `temperature`, `temperature_interval`,
      `reliability`, `sheets`, `pooled`, `ratios`, `seed_disagreement`, `baseline`.
    - With `only=k`: fold k and seed 0 only, everything under `out/f"fold-{k}"`, the step being the recipe's last.
  - `train_final(labelled, recipe, device, out, step: int, temperature: float) -> None`: `SEEDS` trained on all
    labelled sheets for `step` steps. Saves `out/MODEL_DIR/seed-<s>.pt` and `model.json` (`step`, `temperature`,
    `seeds`, `sheets`, `level`).
  - `load_model(out) -> tuple[list[nn.Module], float]`.
  - `load_snapshot(path) -> nn.Module`.
  - `write_outputs(out, sid, q, decision, maps) -> None`: the four files. `erased.png` is `page.png` in its own mode
    (`"1"` is converted to `"L"`), with outside px set to the per-channel median of the inside px.
  - `predict_pages(nets, temperature, sids, device, out) -> None`: saves on `SAVE_THREADS = 4` threads, as the old
    `predict_all` did.
  - `predict_all(device, out) -> None`: every unlabelled map page; labelled sheets keep their held-out outputs. Then
    `PAGES_FILE`: per page `filled_ink` and `dropped_ink`, sorted by `dropped_ink`, highest first.
  - `format_scores(result) -> str`.

- [ ] **Step 1: Write the failing tests.** Synthetic pages in `tmp_path`: monkeypatch `data.WORK`, `data.LABELS`,
  `data.PAGE_KINDS` and `run.BASELINE`. `learn.train` is stubbed to save tiny real nets as snapshots.

```python
def test_choose_step_pools_ink_and_breaks_ties_early():
    a = [(500, 0.30, 10.0), (1000, 0.20, 10.0)]; b = [(500, 0.10, 30.0), (1000, 0.20, 30.0)]
    assert run.choose_step([a, b]) == 500          # pooled: 0.15 vs 0.20
    assert run.choose_step([[(500, 0.2, 1.0), (1000, 0.2, 1.0)]]) == 500

def test_every_labelled_sheet_held_out_once(fake_world): ...   # folds.json sheets == labelled; each has HELD_FILE
def test_write_outputs_keeps_colour(tmp_path): ...             # RGB page: erased.png RGB, outside px = inside median,
                                                               # outside.png mode "1" equals decision.outside
def test_predict_all_skips_labelled_and_ranks_dropped(fake_world): ...
def test_baseline_only_on_shared_sheets(fake_world): ...       # BASELINE holds 1 of 3 sheets -> 1 paired value
```

- [ ] **Step 2: Run; expect** `ModuleNotFoundError`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: held-out folds x seeds, step choice, temperature, outputs, final model, every page`.

---

### Task 9: Pictures and the `frame` command

**Files:**
- Create: `pipeline/src/sfzmaps/frame/look.py`
- Modify: `pipeline/src/sfzmaps/cli.py`
- Test:
  - `pipeline/tests/test_frame_look.py`
  - `pipeline/tests/test_cli.py`

**Interfaces:**
- Consumes: `run` file names and `sheet_dir` (Task 8).
- Produces, in `frame/look.py`:
  - `LONG_SIDE`, `MOSAIC_SIDE`, `MOSAIC_COLUMNS` and the colours, as in the deleted `look.py`.
  - `sheet_picture(lum, q, outside, side=LONG_SIDE) -> Image`: ink coloured by q (blue for inside, red for outside),
    the inside region's boundary in magenta. Pages are never enlarged.
  - `series_mosaics(out) -> list[Path]`: one per series. A labelled sheet's caption shows its held-out `frame_kept`
    and `map_cut`.
  - `CLOSEUPS`: a list of (series, sheet, (x0, y0, x1, y1) page px, caption):
    - `("1921", "20180102094204401", (54, 82, 354, 382), "1921 thin rule and arc, top-left")`
    - `("1921", "20180102094204401", (54, 1506, 354, 1806), "1921 thin rule, mid-left")`
    - `("1929", "20180102100804566-p09", (1562, 740, 2179, 840), "1929 p09 Ocean Beach bar")`
    - `("1948", "3", (563, 5702, 1263, 6402), "1948/3 commercial bands")`
    - `("1948", "4", (680, 5000, 1160, 5800), "1948/4 Great Highway")`
    - `("1948", "4", (4334, 5905, 4834, 6305), "1948/4 blocks 1724-1726 and frame bar")`
    - `("1948", "8", (0, 1500, 600, 4500), "1948/8 left edge")`
  - `closeups(out, items=CLOSEUPS) -> Path`: each crop at full scan resolution (never shrunk), outside ink tinted
    red, the outline drawn, captioned, stacked into `out/look/closeups.png`. A page with no outputs is skipped with a
    log line.
- Produces, in `cli.py`: `sfzmaps frame [--device cpu|mps] [--folds] [--only-fold K] [--predict-only]`, with the
  flags mutually exclusive (`BadParameter`) and one `gpu_turn` around the GPU work.
  - **No flag:** needs `folds.json`, otherwise error and exit 1 before any training. Then `train_final` with its
    `step` and `temperature`, `predict_all`, `series_mosaics`, `closeups`.
  - **`--folds`:** `run_folds` (all), then print `format_scores`.
  - **`--only-fold K`:** `run_folds(only=K)`. Then the `CLOSEUPS` pages and every page of 1948 predicted with that
    fold's last seed-0 snapshot at T = 1 into `out/fold-K/`, and the pictures there.
  - **`--predict-only`:** needs `out/model/`, then `predict_all` and the pictures.

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_frame_needs_the_folds_first(monkeypatch, tmp_path):
    # stub run.train_final / run_folds / predict_all to raise; folds.json absent -> exit 1, nothing trained
def test_cli_frame_flags_exclude_each_other(): ...
def test_closeups_are_full_resolution(tmp_path): ...    # a 300 x 300 box comes out 300 px wide (+ caption)
def test_sheet_picture_never_enlarges(): ...
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: PASS.
  Run: `uv run pytest -q -n 8`
- [ ] **Step 5: Commit** `frame: pictures (mosaics, full-resolution close-ups) and the frame command`.

---

### Task 10: First look on one fold (controller; needs Tasks 4 and 9)

- [ ] **Step 1:** `uv run sfzmaps frame --only-fold 0` (mps). Log the time per step and the total.
- [ ] **Step 2:** Look at:
  - the held sheets' pictures;
  - `closeups.png`;
  - the 1948 mosaic in `data/frame/fold-0/look/`.

  Check the thin rule, the arcs, the margin insets, Ocean Beach, 1948/3, 1948/4 and 1948/8.
- [ ] **Step 3: Measure the effective view** (spec part 3, red team). Write a throwaway
  `scripts/spikes/issue71_erf.py` that takes the gradient of the summed output over 20 outline px of a held page with
  respect to the input, using the fold's net on cpu. Report the radius holding 95% of the |gradient| mass, against the
  frame depths measured in the spec (up to about 1,100 px).
- [ ] **Step 4:** Send the owner the close-ups and the 1948 mosaic (SendUserFile, each with its purpose), with the
  effective view. **Wait for the owner** before Task 11.

### Task 11: Held-out run (controller)

- [ ] **Step 1:** `uv run sfzmaps frame --folds`. Expect about 8 trainings of 4,000 steps; log the total time.
- [ ] **Step 2:** Report to the owner, numbers with their 90% intervals:
  - the chosen step;
  - T with its interval, and the reliability table near ¼;
  - pooled `frame_kept`, `map_cut` and `cost`;
  - the cost-ratio table (2×, 3×, 10×);
  - seed disagreement;
  - paired differences against the two-pass baseline;
  - the five worst sheets with pictures.

  Near-ties are judged by looking, not decimals.
- [ ] **Step 3: Wait for the owner.** If seed disagreement is large next to the errors, the owner decides whether to
  raise the seed count (spec part 6).

### Task 12: Final models and every page (controller)

- [ ] **Step 1:** `uv run sfzmaps frame`: final models, every page, mosaics and close-ups.
- [ ] **Step 2:** Send the owner:
  - every series mosaic;
  - the close-ups;
  - the top of `pages.json` (the pages where "keep the largest region" dropped the most ink: candidates for `maps = 2`
    in `page_kinds.csv`).

  The owner's eye decides.

### Task 13: Final review and pull request

- [ ] **Step 1:** A whole-branch review: deferred minors from the ledger, spike scripts left importing removed modules
  (list them in the PR), the spec's anti-goals.
- [ ] **Step 2:** `uv run pytest -q -n 8`; push; open the PR against `main` with the held-out numbers and pictures.
  The owner merges.
