# Sheet Warp, Fast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sheet warp (`sfzmaps.sheetwarp`) many times faster on the owner's Mac: both fits on the GPU with
a deterministic field and sampler, the probabilistic stages stopping when settled, and faster CPU preparation —
judged by the same scores as today's method and the same answer every run.

**Architecture:** A new module `sfzmaps/sheetwarp/device.py` holds the device choice and the three deterministic
building blocks the fits use (the field at fixed points, the map sampler, the separable blur). `first.py` and
`probabilistic.py` use them; the preparation in `targets.py`, `probabilistic.py`, `first.py`, `sheet.py` and
`transform.py` gets the measured CPU speed-ups; `run.py` records the device.

**Tech Stack:** Python 3.12, PyTorch 2.15 (MPS on the Mac, CPU elsewhere), OpenCV, numpy, scipy.

**Spec:** `docs/superpowers/specs/2026-11-14-sheet-warp-fast-design.md` (read it: the spike's numbers and the
reference code paths are there). Earlier design: `docs/superpowers/specs/2026-11-14-sheet-warp-build-stage-design.md`.

## Global Constraints

- Corrections need NOT be byte-identical to today's. Acceptance per sheet against
  `pipeline/data/warp-baseline-2026-11-14/<series>/<sheet>/scores.json`: the chosen correction's `within1_*` share
  ≥ today's − 0.005, `smallest_cell_probabilistic` > 0; over the sheets checked, the mean within-1 ≥ today's − 0.001.
- The same answer every run on the same device: fitting a sheet twice gives bit-identical correction grids.
- No atomic-add / scatter-based backward in the fits (that is what breaks repeatability on the GPU): the field's
  gradient is a fixed-order sum per grid node; the sampler's gradient is analytic.
- The loss terms, stage schedule (start at half a street width, halve each stage, last at sigma), learning rates,
  the first fit's 1500 steps, and the probabilistic cap of 800 steps per stage stay as they are. The only change to
  the optimization: a probabilistic stage stops when its loss fell by less than 1e-4 (relative) over the last 25
  steps, after at least 50 steps.
- Device: `mps` when available, else `cpu`; the environment variable `SFZMAPS_WARP_DEVICE=cpu|mps` overrides (tests
  and CI use the CPU). No `torch.set_num_threads` change on MPS; on the CPU keep today's thread counts (8 first,
  3 probabilistic).
- Preparation speed-ups must give identical outputs to today's functions, except point placement through the grid,
  which must stay within 1e-3 px of exact `to_pixel` (measured on big sheets).
- No new dependencies. No paid model calls. Never commit `pipeline/data/`. Don't run the old pipeline's stages.
  Warps need pixi's GDAL first on PATH (`/Users/sbuss/workspace/sfzoning/.claude/worktrees/saved-reads/.pixi/envs/default/bin`).
- Names say what things are. Reference code (throwaway, read it, don't copy it in blindly):
  `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/spike/`
  `fits.py`, `fast_prep.py`, `common.py`, `exp5_cpu.py`.
- Tests: `cd pipeline && uv run pytest -q` passes without `pipeline/data/`, on a machine without MPS (CPU path).

## Review Focus

1. A machine without MPS (CI, Linux) → the CPU path runs and its tests pass. Tests in Tasks 1 and 2.
2. Fitting the same sheet twice → bit-identical grids on the same device. Test in Task 1 (building blocks, CPU) and a
   check in Task 2 (MPS, real sheet).
3. The field and sampler match `grid_sample` (values and gradients) so the model is unchanged. Test in Task 1.
4. A sheet whose points don't all fall inside the grid / map (points at the crop edge, `padding_mode` behaviour
   today: border for the field, zeros for the maps) → same behaviour as `grid_sample`. Test in Task 1.
5. Grid-placed points near the page edge or where `to_pixel` fails today (non-finite) → same non-finite rows, within
   tolerance elsewhere. Test in Task 3.

---

### Task 1: The device and the deterministic building blocks

**Files:**
- Create: `pipeline/src/sfzmaps/sheetwarp/device.py`
- Test: `pipeline/tests/test_sheetwarp_device.py`

**Interfaces (produces):**
- `fit_device() -> torch.device`: `SFZMAPS_WARP_DEVICE` if set, else `mps` if `torch.backends.mps.is_available()`,
  else `cpu`.
- `class PointField`: built once per point set — `PointField(points: np.ndarray (n, 2) crop px, grid_shape:
  tuple[int, int], spacing: float, device)`; `field(G: torch.Tensor (1, 2, gh, gw)) -> torch.Tensor (n, 2)`, the same
  values as today's `F.grid_sample(G, gp(points), mode="bilinear", padding_mode="border", align_corners=True)` read
  per point, differentiable in G with a deterministic backward (fixed-order sum per node).
- `class MapSampler`: `MapSampler(bins: np.ndarray (n,) direction bin per point, h: int, w: int, device)`;
  `sample(maps: torch.Tensor (B, 1, h, w) or flattened, xy: torch.Tensor (n, 2)) -> torch.Tensor (n,)`, the same
  values as today's per-bin `grid_sample(..., padding_mode="zeros", align_corners=True)`, differentiable in xy with
  an analytic gradient (maps take none).
- `blur(maps: np.ndarray (B, h, w) float32, sigma: float, device) -> torch.Tensor (B, 1, h, w)`: separable Gaussian
  matching `scipy.ndimage.gaussian_filter(map, sigma)` (truncate 4.0, mode "reflect") to within 1e-6 of the map's
  peak.

- [ ] **Step 1: Write the failing tests** (CPU device; synthetic data): `test_point_field_matches_grid_sample`
  (values exact to 1e-6, gradient into G within 1e-5 of grid_sample's, including points outside the grid → border);
  `test_point_field_repeats_bit_for_bit`; `test_map_sampler_matches_grid_sample` (values within 1e-6, gradient in xy
  within 1e-5, points outside the map → 0); `test_blur_matches_scipy`; `test_fit_device_honours_the_override`
  (monkeypatch the env var).
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement `device.py`** (see `fits.py` in the spike for the formulation: bilinear weights W per point,
  nodes bucketed by number of contributions for the fixed-order backward; a gather of the 4 neighbours from the
  flattened maps with the analytic gradient).
- [ ] **Step 4: Run the tests** (`uv run pytest tests/test_sheetwarp_device.py -q`), then, on this Mac, the same
  tests with `SFZMAPS_WARP_DEVICE=mps` where they apply (say which pass).
- [ ] **Step 5: Commit** — `git commit -m "sheetwarp: the fit device, a deterministic field, sampler and blur"`

### Task 2: The fits on the device; probabilistic stages stop when settled

**Files:**
- Modify: `pipeline/src/sfzmaps/sheetwarp/first.py` (`first_correction`'s fit), `pipeline/src/sfzmaps/sheetwarp/probabilistic.py`
  (`probabilistic_correction`)
- Test: `pipeline/tests/test_sheetwarp_first.py`, `pipeline/tests/test_sheetwarp_probabilistic.py`

**Interfaces:** unchanged signatures; both return `Correction` with a float32 CPU numpy grid. They use
`device.fit_device()`, `PointField`, `MapSampler`, `blur` (Task 1). The stop rule (Global Constraints) in the
probabilistic stages.

- [ ] **Step 1: Tests:** `test_a_stage_stops_when_its_loss_settles` (a synthetic fit whose loss flattens stops
  before the cap and after ≥ 50 steps); a small synthetic probabilistic fit gives a finite grid on the CPU device and
  the same grid when run twice.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement** (move tensors to the device once per fit; keep the loss terms and stage schedule; bring
  the result back to CPU numpy float32).
- [ ] **Step 4: Verify on real sheets** (scratch script outside the repo; do not write `pipeline/data/warp`): for
  1948 sheet 11, 1921 20180102094212101, 1968 20180102110513967-p21, 2000-digital-zoning p03 and p09 (street edges and
  later blocks): `run.fit_sheet(Sheet(series, name))` on MPS, twice — grids bit-identical; scores against the
  baseline within the Global Constraints; fit times. Also once with `SFZMAPS_WARP_DEVICE=cpu` on one small sheet
  (the CPU path works).
- [ ] **Step 5: Commit** — `git commit -m "sheetwarp: both fits on the GPU; probabilistic stages stop when settled"`

### Task 3: Faster preparation

**Files:**
- Modify: `pipeline/src/sfzmaps/transform.py` (a grid placement helper), `pipeline/src/sfzmaps/sheetwarp/targets.py`
  (`block_side_points`, `fit_targets`, `_crop` placement), `pipeline/src/sfzmaps/sheetwarp/probabilistic.py`
  (`block_line_weights`), `pipeline/src/sfzmaps/sheetwarp/first.py` (`held_sides`, `propagated_shifts`),
  `pipeline/src/sfzmaps/sheetwarp/sheet.py` (`block_line_half_width`)
- Test: the matching `pipeline/tests/test_sheetwarp_*.py`, `pipeline/tests/test_transform.py`

**Interfaces:**
- `SheetTransform.grid_to_pixel(w: np.ndarray, page_shape: tuple[int, int], step: int = 32) -> np.ndarray`: placement
  through `to_world` sampled every `step` px over the page (cubic spline), today's iteration from that start, then
  one exact step; non-finite where today's `to_pixel` is. Used for the block-side points (pts, ta, tb, still one call
  per whole array) and the other large batches in the warp's preparation; exact `to_pixel` everywhere else.
- `Targets` keeps the crop's skeleton, ink half width and ink direction (computed once in `fit_targets`) for
  `block_line_weights`; the page's ink distance (EDT) is computed once per sheet (`Sheet`) for the half width and
  held sides.
- Distance transforms: `cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)` with scipy's float64 value
  recovered as `sqrt(round(d*d))`.
- `block_line_weights`: the stroke-width EM on the width histogram (`np.unique` counts), identical float16 output.
- `propagated_shifts`: each side's members by one stable sort of the side ids.

- [ ] **Step 1: Tests** (synthetic): `test_grid_to_pixel_matches_to_pixel` (within 1e-3 px; non-finite rows kept);
  the cv2 distance equals scipy's on random masks; histogram EM equals per-pixel EM; grouped members equal masked
  members.
- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify on real sheets** (scratch, outside the repo): on 1948 sheet 11 and 1964
  20180102105639444-p17, today's functions (from git `HEAD` before this task) against the new ones: fit targets,
  block-line weights, held sides, half width, propagated shifts identical; grid placement max error and the number of
  rounded pixels that change; preparation times before and after.
- [ ] **Step 5: Commit** — `git commit -m "sheetwarp: faster preparation (OpenCV distances, shared crop features, grid placement)"`

### Task 4: The device in the record; sheets side by side

**Files:**
- Modify: `pipeline/src/sfzmaps/sheetwarp/run.py`, `pipeline/src/sfzmaps/cli.py` (the `--jobs` default if it
  changes), tests in `pipeline/tests/test_sheetwarp_run.py`

- [ ] **Step 1:** `numeric_versions()` adds the fit device and the macOS version (`platform.mac_ver()[0]`, empty
  elsewhere); `scores.json` records `"device"`. Test both.
- [ ] **Step 2: Measure** throughput of `sfzmaps warp 1929 --force --jobs N` (12 sheets; pixi's GDAL first on PATH)
  for N = 3, 6, 9: wall time and peak memory. Set `correct_series`' default `jobs` (and the CLI default) to the best
  that keeps memory well under 64 GB; report the numbers.
- [ ] **Step 3:** `uv run pytest -q`; commit — `git commit -m "sheetwarp: the fit device in the record; more sheets at once"`

### Task 5: Documentation

**Files:** Modify `pipeline/README.md` (the sheet warp section)

- [ ] **Step 1:** Say, in plain words: the fits run on the Mac's GPU when there is one (CPU otherwise, slower; the
  two give slightly different corrections, each the same every run); probabilistic stages stop when settled; the
  measured speed (per sheet and whole job, from Tasks 2–4); `--jobs` advice. Keep the rest accurate.
- [ ] **Step 2:** Commit — `git commit -m "sheet warp README: the GPU, the speed"`

### Task 6 (controller): rebuild and judge

- [ ] Full `sfzmaps build` (warp stages only); compare every baseline sheet (Global Constraints), list uncorrected
  sheets and any that got worse with pictures; the 208 sheets without a baseline: their scores against today's
  placement; seams by eye; a fresh clone; commit `dvc.lock`; push; update PR #58 from its live text.
