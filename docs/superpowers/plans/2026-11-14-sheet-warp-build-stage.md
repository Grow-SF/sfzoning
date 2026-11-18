# Sheet Warp Build Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the sheet warp from `pipeline/scripts/sheet_warp/` (13 step scripts, a second environment, a
`sitecustomize` patch, a stand-in data folder) into the `sfzmaps` package as `sfzmaps warp [SERIES ...]` and one
`warp-<series>` build stage per series, writing every placed sheet's corrected placement and warp to
`pipeline/data/warp/`, with the same corrections the scripts make today.

**Architecture:** `SheetTransform` learns to carry a correction (a displacement field on a page-pixel grid). A new
subpackage `sfzmaps.sheetwarp` holds the method as plain functions, ported step by step from the scripts with the
same numerics, in memory (no intermediate files). `sfzmaps.sheetwarp.run` fits, chooses, writes and warps one sheet
and skips sheets whose inputs haven't changed (`sfzmaps.fresh`); `sfzmaps.build` gains a stage per series and
freezes the deprecated old stages.

**Tech Stack:** Python 3.12, numpy 2.5.4, scipy 1.18.1, PyTorch 2.15.0 (CPU), shapely, geopandas, OpenCV, gdal (pixi),
typer, DVC (via `sfzmaps build`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-sheet-warp-build-stage-design.md` (read it first; it says why).

## Global Constraints

- Corrections must be byte-identical to the reference (below): the port changes no numbers. Keep, exactly:
  one `to_pixel` call per whole array when placing block-side points and tangent ends; the Presidio cut twice for
  block-side work and once for street edges and later blocks; the crop from candidate extreme points; PyTorch
  threads 8 for the first correction's fit and 3 for the probabilistic one; float32 grids; the same iteration
  counts, learning rates and loss terms.
- No model calls, no money spent. Nothing here may call `sfzmaps.llm` live; the warp uses no reads.
- Never commit anything under `pipeline/data/`. Never run a bare `dvc gc`. Don't run the old pipeline's stages
  (`edition-*`, `period-*`, `today`, `shared`): they are deprecated and get frozen in Task 6.
- No monkeypatching, no second environment, no subprocesses between steps, no environment variables carrying the
  series or sheet.
- Names say what things are: never `p1`, `prob6`, `reg2`, `confit`, `plus`, `dm_`. "Street edges and later blocks"
  is the name of what the scripts call extra evidence.
- PyTorch is a regular dependency (`torch==2.15.0` in `[project] dependencies`), CPU wheels on Linux.
- Every placed sheet of every series (357) is warped. Placed = the sheet folder in `data/work/<series>/` has
  `page.png`, `page.json`, `transform.json`, `refine.json`, `blocks_px.geojson`, `warped.tif`,
  `warped.footprint.geojson`.
- Street edges and later blocks stays on 2000 p09 and p10 only (`2000-digital-zoning-p09`, `-p10`) in this plan.
- No fallback to today's placement when it scores higher (spec, "The choice").
- Run tests from `pipeline/`: `uv run pytest -q` (all must pass; CI runs them without `pipeline/data/`).

## Reference

The scripts at commit 7d208d6 (`pipeline/scripts/sheet_warp/steps/*.py`, `warp.py`) are the reference
implementation: port each function from the script named in the task, keeping its arithmetic. Their outputs on this
machine (git-ignored data, worktree `saved-reads`):

- `pipeline/data/sheet-warp/fits/1921/20180102094142210/` and `.../fits/2000-digital-zoning/<13 sheets>/`: every
  intermediate file (`block_side_points.npz`, `block_side_points_on_page.npz`, `fit_targets.npz`,
  `measure_today.npz`, `held_sides.npz`, `first_correction_inputs.npz`, `propagated_shifts.npz`,
  `first_correction.npz`, `probabilistic_inputs.npz`, `probabilistic_correction.npz`, `measure_first.npz`,
  `measure_probabilistic.npz`, `scan_measures.json`; on p09 and p10 also `extra_evidence_points.npz`,
  `extra_evidence_inputs.npz`, `extra_evidence_correction.npz`), and `fits/scores.jsonl`, `fits/choice.json`.
- `pipeline/data/sheet-warp/stand-in-data/work/<series>/<sheet>/warped.tif`, `warped_paper.tif` and footprints:
  the warps made with the chosen corrections (the 1921 sheet above and the 13 2000 sheets).
- For every other 1921–1990 sheet, the approved run's rebuild, old file names:
  `/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/prob-warp/rebuild-2026-11-13/run-notext/<series>/<sheet>/confit_<sheet>_p1.npz`
  and `confit_<sheet>_prob6.npz` (`Gc` arrays). Read only; never write there.

**Port check:** `uv run python /private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/check_port.py SERIES SHEET STAGE`
from `pipeline/` calls the new functions on one sheet and compares every array with the reference, printing
`SAME`/`DIFF` per array and `ALL SAME` at the end. STAGE is `targets`, `first`, `probabilistic`, `street` or `warp`
(each includes the earlier ones). It is verification tooling outside the repo; don't commit it. Use sheets
`1921 20180102094142210` and `2000-digital-zoning 2000-digital-zoning-p09` (p09 is the only one with `street`).

## Review Focus

1. A sheet the method can't fit (no block-side point on the page, every side already held, nothing to propagate)
   → written uncorrected with a reason in `scores.json`; the series still succeeds. Tests in Tasks 3 and 5.
2. `sfzmaps warp` run twice on unchanged inputs → the second run fits nothing. Test in Task 5.
3. A corrected `transform.json` names its correction relative to its own folder, so `data/warp` works after
   `sfzmaps pull` on another machine. Tests in Tasks 1 and 5.
4. A sheet interrupted mid-write (the warp raises after the fits) → not fresh next time; it is redone. Test in
   Task 5.
5. `sfzmaps warp NOPE` (no such series) → a clear error naming the known series, exit code nonzero. Test in Task 5.

---

### Task 1: Corrections in `SheetTransform`

**Files:**
- Modify: `pipeline/src/sfzmaps/transform.py`
- Test: `pipeline/tests/test_transform.py`

**Interfaces:**
- Produces:
  - `class Correction` (dataclass): `grid: np.ndarray` (2, gh, gw) float32, x then y displacement in page px at
    each grid node; `spacing: float` (px between nodes); `origin: np.ndarray` (2,) float, page px of node (0, 0);
    `file: str | None = None`.
    - `Correction.u(self, p: np.ndarray) -> np.ndarray`: (n, 2) page px → (n, 2) displacement; bilinear
      (`scipy.ndimage.map_coordinates(grid[k], [(p[:,1]-o[1])/spacing, (p[:,0]-o[0])/spacing], order=1,
      mode="constant", cval=0.0)` for k = 0, 1), 0 outside the grid.
    - `Correction.save(self, path: Path, **extra) -> None`: `np.savez(path, Gc=grid, spacing=spacing,
      origin=origin, **extra)`.
    - `Correction.load(cls, path: Path) -> Correction`: `grid=Z["Gc"]`, `spacing=float(Z["spacing"])`,
      `origin=np.asarray(Z["origin"], float)`, `file=path.name`.
  - `SheetTransform.correction: Correction | None = None` (new dataclass field, after `smoothing`).
  - `SheetTransform.uncorrected(self) -> SheetTransform`: the same placement without the correction.
  - `SheetTransform.from_json(cls, d: dict, folder: Path | None = None) -> SheetTransform`: when
    `d["correction"]["file"]` is present, loads it from `folder` (relative names only; `ValueError` if `folder`
    is None).
  - `SheetTransform.to_json()` adds `"correction": {"file": correction.file}` when a correction is set.
  - With a correction: `to_pixel(w)` = uncorrected `to_pixel(w)` plus `u` there (rows with a non-finite pixel left
    as they are); `to_world(q)` = uncorrected `to_world(p)`, `p` solving `p + u(p) = q` by the algorithm below.
    Without one, every method behaves exactly as now.

- [ ] **Step 1: Write the failing tests** in `tests/test_transform.py` (reuse that file's existing helpers for a
  fitted synthetic `SheetTransform`):
  - `test_a_correction_adds_its_field_to_to_pixel`: a `Correction` with a constant field (1.5, -2.0) over a 3 x 3
    grid, spacing 50, origin (0, 0); for world points whose uncorrected pixel lies inside (0..100, 0..100),
    `t.to_pixel(w) - t.uncorrected().to_pixel(w)` equals (1.5, -2.0) to 1e-6; outside the grid it equals (0, 0).
  - `test_to_world_inverts_a_corrected_to_pixel`: a smooth field (amplitude ≤ 3 px) on a 6 x 6 grid; for 200
    points inside it, `t.to_pixel(t.to_world(q))` is within 1e-3 px of `q`.
  - `test_without_a_correction_nothing_changes`: `from_json(d)` of a dict without `"correction"` gives the same
    `to_pixel`, `to_world` and `to_json()` as before (`to_json()` has no `"correction"` key).
  - `test_from_json_reads_the_correction_beside_the_placement(tmp_path)`: `Correction.save(tmp_path /
    "correction.npz")`, a dict with `"correction": {"file": "correction.npz"}`; `from_json(d, folder=tmp_path)`
    has a correction with the same grid, and `to_json()["correction"] == {"file": "correction.npz"}`.
  - `test_a_correction_needs_its_folder`: the same dict with `folder=None` raises `ValueError`.

- [ ] **Step 2: Run them to see them fail**
  Run: `uv run pytest tests/test_transform.py -q` → FAIL (`Correction` not defined).

- [ ] **Step 3: Implement** in `sfzmaps/transform.py`. `to_world` with a correction must be this algorithm (from
  `pipeline/scripts/sheet_warp/correction_patch/sitecustomize.py`, which the reference warps used):

  ```python
  q = np.atleast_2d(np.asarray(px, float))
  p = q - u(q)
  best, err = p.copy(), np.linalg.norm(p + u(p) - q, axis=1)
  for _ in range(60):
      p = p + 0.5 * (q - u(p) - p)
      e = np.linalg.norm(p + u(p) - q, axis=1)
      better = e < err
      best[better], err[better] = p[better], e[better]
      if err.max() < 1e-4:
          break
  return <uncorrected to_world>(best)
  ```

- [ ] **Step 4: Run the tests to see them pass**
  Run: `uv run pytest tests/test_transform.py tests/test_warp.py -q` → PASS.

- [ ] **Step 5: Commit**
  `git add pipeline/src/sfzmaps/transform.py pipeline/tests/test_transform.py && git commit -m "transform: a placement can carry a correction"`

### Task 2: The sheet and what the fits aim at

**Files:**
- Create: `pipeline/src/sfzmaps/sheetwarp/__init__.py` (a docstring: what the sheet warp is, in plain words, and
  the modules), `pipeline/src/sfzmaps/sheetwarp/sheet.py`, `pipeline/src/sfzmaps/sheetwarp/targets.py`
- Port from: `steps/warp_common.py`, `block_side_points.py`, `fit_targets.py`, `measure_fit.py`
- Test: `pipeline/tests/test_sheetwarp_targets.py`

**Interfaces:**
- Consumes: Task 1's `Correction`.
- Produces:
  - `sheet.py`:
    - `class NothingToFit(Exception)`: raised by any step that has nothing to work on; its message is the reason.
    - `PLACED_FILES: tuple[str, ...]` = the seven file names in Global Constraints.
    - `placed_sheets(series: str) -> list[str]`: sorted names of the placed sheets in `config.WORK_DIR / series`
      (empty list when the folder doesn't exist).
    - `@dataclass class Sheet`: `series: str`, `name: str`; `folder` property (`config.WORK_DIR / series / name`);
      cached properties `transform` (today's placement, `SheetTransform.from_json` of `transform.json`), `page`
      (grayscale uint8), `ink` (`page < 128`), `year` (`read.series_year(series)`), `lots`
      (`read.load_parcels(year=year)`), `ft_per_px` (`transform.median_scale_ft_per_px(page.shape)`),
      `block_line_half_width` (warp_common.half on the whole page's ink); method `area(self, cuts: int = 2)`
      (the warped footprint less the Presidio, cut `cuts` times, as `lots_and_area`).
  - `targets.py`:
    - `@dataclass class SidePoints`: `pts, sid, ta, tb` (map ft: points every 2 ft, their side, tangent ends);
      `px, pa, pb` (the same placed by today's placement: exactly one `to_pixel` call on each whole array);
      `wt: np.ndarray | None = None`, `kind: np.ndarray | None = None` (Task 4 fills them).
    - `block_side_points(sheet: Sheet) -> SidePoints`; raises `NothingToFit` when there are no points.
    - `@dataclass class Targets`: `maps` (18, h, w) uint8, `origin` (2,) int, `page_shape` (H, W).
    - `fit_targets(sheet: Sheet) -> Targets` (with `crop` from the script); raises `NothingToFit` when no
      candidate point lands on the page.
    - `distances(targets: Targets, px, pa, pb) -> np.ndarray`: per point, the distance (px) to the nearest
      printed centre running its way; `inf` off the crop.
    - `measure(targets: Targets, sides: SidePoints, correction: Correction | None = None) -> np.ndarray`:
      `distances` of today's placement, or of today's plus the correction (`px + correction.u(px)`, same for
      `pa`, `pb`).

- [ ] **Step 1: Write the failing tests** (synthetic, no `pipeline/data/`):
  - `test_placed_sheets_needs_every_placed_file(tmp_path, monkeypatch)`: point `config.WORK_DIR` at `tmp_path`;
    a sheet folder with all seven files is listed, one missing `refine.json` isn't; an unknown series gives `[]`.
  - `test_distances_reads_the_map_of_each_points_direction`: a `Targets` with two direction maps distinguishable
    by value; a horizontal and a vertical point get their own map's value; a point off the crop gets `inf`.
  - `test_measure_with_a_correction_moves_the_points_first`: a constant correction of (+3, 0) px makes `measure`
    equal `distances` at `px + (3, 0)`.

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_sheetwarp_targets.py -q` → FAIL.

- [ ] **Step 3: Implement** `sheet.py` and `targets.py`, porting the scripts' arithmetic unchanged.

- [ ] **Step 4: Run the tests and the port check**
  `uv run pytest tests/test_sheetwarp_targets.py -q` → PASS.
  Port check `targets` on both sheets → `ALL SAME` (block-side points and their placement, fit targets, today's
  measure, ft per px, block-line half width).

- [ ] **Step 5: Commit** — `git commit -m "sheetwarp: the sheet, its block-side points and the fit targets"`

### Task 3: The first correction, and PyTorch as a dependency

**Files:**
- Modify: `pipeline/pyproject.toml`, `pipeline/uv.lock`
- Create: `pipeline/src/sfzmaps/sheetwarp/first.py`
- Port from: `held_sides.py`, `first_correction_inputs.py`, `propagate_shifts.py`, `first_correction_fit.py`
- Test: `pipeline/tests/test_sheetwarp_first.py`

**Interfaces:**
- Consumes: Task 2's `Sheet`, `SidePoints`, `Targets`, `measure`, `NothingToFit`; Task 1's `Correction`.
- Produces:
  - `held_sides(sheet: Sheet, sides: SidePoints, d_today: np.ndarray) -> np.ndarray` (bool per side).
  - `first_correction(sheet: Sheet, sides: SidePoints, targets: Targets, held: np.ndarray) -> Correction`
    (inputs, propagated shifts and the fit inside; `torch.set_num_threads(8)` before the fit).

- [ ] **Step 1: Add PyTorch.** `torch==2.15.0` in `[project] dependencies`; on Linux from the CPU index
  (`[[tool.uv.index]] name = "pytorch-cpu"`, `url = "https://download.pytorch.org/whl/cpu"`, `explicit = true`;
  `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]`). `uv lock`, then
  check `uv.lock` still pins numpy 2.5.4 and scipy 1.18.1 (`grep -A1 'name = "numpy"' uv.lock`). `uv sync`.
- [ ] **Step 2: Write the failing tests:**
  - `test_propagation_with_every_side_held_has_nothing_to_move`: synthetic sides all held → the first correction's
    propagation raises `NothingToFit` (the script took `.max()` of an empty array here).
  - `test_held_sides_leaves_out_points_near_a_sides_ends`: a synthetic sheet stub (`ft_per_px`, `ink`,
    `block_line_half_width` set) with one side whose only distant points are within 2 px of its ends → held.
- [ ] **Step 3: Run them to see them fail** — `uv run pytest tests/test_sheetwarp_first.py -q` → FAIL.
- [ ] **Step 4: Implement** `first.py`, porting the scripts unchanged (drop their prints except one summary line
  through `logging`).
- [ ] **Step 5: Run the tests and the port check**
  `uv run pytest tests/test_sheetwarp_first.py -q` → PASS. Port check `first` on both sheets → `ALL SAME` (held
  sides, the first correction's `Gc`, `spacing`, `origin`, and its measure).
- [ ] **Step 6: Commit** — `git commit -m "sheetwarp: the first correction; PyTorch a pipeline dependency"`

### Task 4: The probabilistic correction, and street edges and later blocks

**Files:**
- Create: `pipeline/src/sfzmaps/sheetwarp/probabilistic.py`,
  `pipeline/src/sfzmaps/sheetwarp/street_edges_and_later_blocks.py`
- Port from: `probabilistic_inputs.py`, `block_line_weights.py`, `probabilistic_fit.py`, `extra_evidence_points.py`
- Test: `pipeline/tests/test_sheetwarp_probabilistic.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
  - `@dataclass class ProbabilisticInputs`: `P` (float32), `bin`, `street_px`, `origin`, `gbase`, `gpx`, `tau_j`,
    `wt: np.ndarray | None`, `kind: np.ndarray | None` (the fields of the script's npz).
  - `probabilistic_inputs(sheet: Sheet, points: SidePoints, targets: Targets) -> ProbabilisticInputs` (with `wt`
    and `kind` when `points` has them, as the script's `extra_evidence` mode).
  - `block_line_weights(sheet: Sheet, targets: Targets) -> np.ndarray` (18, h, w) float16.
  - `probabilistic_correction(inputs: ProbabilisticInputs, targets: Targets, weights: np.ndarray) -> Correction`
    (`torch.set_num_threads(3)` before the fit).
  - `street_edges_and_later_blocks(sheet: Sheet, sides: SidePoints) -> SidePoints` (block sides plus later-block
    and street-edge points, with `wt` and `kind`: 0 block side, 1 later block, 2 street edge; placed with one
    `to_pixel` call per whole array; the Presidio cut once).
- [ ] **Step 1: Write the failing tests:**
  - `test_block_line_weights_favour_the_thin_strokes`: a synthetic page with thin (1 px) and heavy (6 px) lines;
    centre pixels on the thin lines weigh more than on the heavy ones, in the map of their direction.
  - `test_street_edge_points_count_by_how_little_blocks_explain_them`: the weight formula `1 - exp(-d^2 / (2 s^2))`
    gives 0 at a block side and nearly 1 far from any (call the module's weight function on synthetic distances).
- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_sheetwarp_probabilistic.py -q` → FAIL.
- [ ] **Step 3: Implement** both modules, porting the scripts unchanged.
- [ ] **Step 4: Run the tests and the port check**
  `uv run pytest tests/test_sheetwarp_probabilistic.py -q` → PASS. Port check `probabilistic` on both sheets and
  `street` on p09 → `ALL SAME` (inputs, the correction's `Gc`, its measure; on p09 the street-edge points, their
  inputs and correction).
- [ ] **Step 5: Commit** — `git commit -m "sheetwarp: the probabilistic correction; street edges and later blocks"`

### Task 5: One sheet end to end, a series, and `sfzmaps warp`

**Files:**
- Modify: `pipeline/src/sfzmaps/warp.py` (an optional argument, no behaviour change without it),
  `pipeline/src/sfzmaps/cli.py`
- Create: `pipeline/src/sfzmaps/sheetwarp/scores.py`, `pipeline/src/sfzmaps/sheetwarp/apply.py`,
  `pipeline/src/sfzmaps/sheetwarp/run.py`
- Port from: `score_fits.py`, `warp_with_correction.py`, `warp.py` (`fit_sheet`, `choose`, `EXTRA_EVIDENCE`)
- Test: `pipeline/tests/test_sheetwarp_run.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `scores.scores(targets, sides, inputs: ProbabilisticInputs, first: Correction, probabilistic: Correction) -> dict`
    with exactly the keys `score_fits.py` writes (`within1_today` … `largest_correction_px_probabilistic`).
  - `scores.choose(s: dict) -> str`: `"probabilistic"` if `within1_probabilistic >= within1_first` and
    `smallest_cell_probabilistic > 0`, else `"first"`.
  - `sfzmaps.warp.warp_sheet(..., resample: Callable[[np.ndarray], np.ndarray] | None = None)`: when given, the
    RGBA from `make_rgba` passes through it before warping (replaces the script's monkeypatch of `make_rgba`).
  - `apply.warp_with_correction(sheet: Sheet, correction: Correction | None, out_dir: Path) -> None`: writes
    `warped.tif`, `warped_paper.tif` and their footprints into `out_dir`, as `warp_with_correction.py` did
    (scan resampled by the field: `cv2.remap`, linear for plain, cubic for paper; then the old warp with today's
    control-point grid and output resolution, footprint through the corrected `to_world`). With `None`, copy the
    sheet's four warp files from `data/work` (today's placement, already warped).
  - `run.OUT = config.DATA_DIR / "warp"`; `run.STREET_EDGES_AND_LATER_BLOCKS = {"2000-digital-zoning":
    {"2000-digital-zoning-p09", "2000-digital-zoning-p10"}}`.
  - `run.correct_sheet(series: str, name: str, force: bool = False) -> str`: `"skipped"` when fresh; else fits,
    chooses, writes `OUT/series/name/`: `correction.npz` (`Correction.save(..., kind=<"first" | "probabilistic" |
    "street edges and later blocks">)`), `scores.json` (the scores, `"chosen"`, and `"placement_check"`:
    `passed`/`reason` from the sheet's `data/work` `score.json` when it has one), the warps, and LAST
    `transform.json` (today's json plus `"correction": {"file": "correction.npz"}`), then `fresh.record`.
    Returns `"corrected (<kind>)"`. On `NothingToFit`: no `correction.npz`, `transform.json` without a correction,
    the copied warps, `scores.json` with `"uncorrected": <reason>`; returns `"uncorrected: <reason>"`.
    Freshness inputs: the seven placed files, `config.REF_DIR / "parcels.parquet"`, `config.REF_DIR /
    "streets.parquet"`, and `fresh.code_files(["sfzmaps.sheetwarp.run"])`.
  - `run.correct_series(series: str, jobs: int = 3, force: bool = False) -> dict[str, str]`: every placed sheet,
    `jobs` at a time in worker processes (spawn context), name → result; raises after all finish if any sheet
    raised, naming them.
  - CLI: `sfzmaps warp [SERIES ...] [--jobs N] [--force]`; no series = every series in `SERIES` with placed
    sheets; an unknown series → `typer.BadParameter` listing the known ones.
- [ ] **Step 1: Write the failing tests** (monkeypatch `config.WORK_DIR`, `config.DATA_DIR`, `run.OUT` to
  `tmp_path`; stub the fits with fakes so no real fitting runs):
  - `test_choose_takes_the_probabilistic_correction_only_when_it_fits_as_well_and_never_folds`.
  - `test_a_sheet_with_nothing_to_fit_is_written_uncorrected`: a fake that raises `NothingToFit("no block sides
    on the page")` → `transform.json` without `"correction"`, warps copied, `scores.json["uncorrected"]` is the
    reason, result starts with `"uncorrected"`.
  - `test_a_second_run_on_unchanged_inputs_fits_nothing`: two `correct_sheet` calls; the fake fit is called once,
    the second returns `"skipped"`.
  - `test_the_correction_is_named_relative_to_its_folder`: `transform.json["correction"] == {"file":
    "correction.npz"}` and `SheetTransform.from_json(d, folder=out)` loads it.
  - `test_a_sheet_interrupted_before_its_placement_is_written_is_redone`: the fake warp raises once; the next call
    fits again.
  - `test_warp_rejects_an_unknown_series`: `CliRunner` on `warp NOPE` → nonzero exit, message lists a known series.
- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_sheetwarp_run.py -q` → FAIL.
- [ ] **Step 3: Implement** `scores.py`, `apply.py`, `run.py`, the `warp_sheet` argument and the command.
- [ ] **Step 4: Run the tests, the port check, and one real sheet**
  `uv run pytest -q` → PASS (all). Port check `warp` on both sheets → `ALL SAME` (scores equal the reference
  `scores.jsonl` line; `warped.tif` and `warped_paper.tif` byte-identical to the stand-in folder's).
  `uv run sfzmaps warp 1921 --jobs 3` → every 1921 sheet corrected; then for each, its `correction.npz` `Gc`
  equals the reference `confit_<sheet>_<chosen>.npz` (`cache-maybe` rebuild; `p1` = first, `prob6` =
  probabilistic) and its `scores.json["chosen"]` matches the rebuild's `choice.json` (`p1`/`prob6`). Run
  `uv run sfzmaps warp 1921` again → every sheet `skipped`.
- [ ] **Step 5: Commit** — `git commit -m "sfzmaps warp: correct and warp every placed sheet of a series"`

### Task 6: Build stages

**Files:**
- Modify: `pipeline/src/sfzmaps/build.py`, `pipeline/dvc.yaml` (regenerated), `pipeline/src/sfzmaps/cli.py` (the
  `stage` command's help text), `pipeline/tests/test_build.py`

**Interfaces:**
- Consumes: Task 5's `sfzmaps warp SERIES`.
- Produces:
  - `Stage.warp: str | None = None` (the series a warp stage corrects).
  - In `stages()`: one `warp-<series>` stage per series in `SERIES`, after the old stages and before `shared`,
    upstream `(<the old stage that owns the series>, "refdata")`. The old stages (`edition-*`, `period-*`,
    `today`, `shared`) get `frozen=True`. `shared`'s upstream stays the old stages only.
  - `stage_commands("warp-S") == [["warp", "S"]]`; `stage_deps` of a warp stage = `_code_deps()` +
    `["pyproject.toml", "uv.lock", "../pixi.toml", "../pixi.lock", "data/ref", "data/work/S"]`; `stage_outs ==
    ["data/warp/S"]`.
  - `run_build` calls `prepare()` only when a stage that isn't a warp stage is in the order.
- [ ] **Step 1: Update the tests first:** keep every existing assertion that still holds; rewrite those that
  assumed the old stages run (`test_plan_adds_downstream_and_limits_to_targets` with warp stages:
  `plan_run({"warp-1948"}, None, t) == ["warp-1948"]`; a stale old stage plans nothing:
  `plan_run({"edition-1948"}, None, t) == []`); add
  `test_every_series_has_one_warp_stage_after_the_stage_that_places_it`,
  `test_a_warp_stage_reads_its_series_work_and_writes_data_warp`,
  `test_the_old_pipeline_stages_are_frozen`.
- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_build.py -q` → FAIL.
- [ ] **Step 3: Implement**, then `uv run sfzmaps build --write-dvc-yaml`.
- [ ] **Step 4: Run** `uv run pytest -q` → PASS; `uv run sfzmaps build --list` lists 22 `warp-*` stages;
  `uv run sfzmaps build --dry-run` (after committing) plans only `warp-*` stages.
- [ ] **Step 5: Commit** (`build.py`, `dvc.yaml`, `cli.py`, `test_build.py`) —
  `git commit -m "build: a warp stage per series; the old pipeline's stages frozen"`

### Task 7: Remove the scripts; document the stage

**Files:**
- Delete: `pipeline/scripts/sheet_warp/` (all of it)
- Modify: `pipeline/README.md` (a section "The sheet warp (`sfzmaps warp`)"), `AGENTS.md` (one line under
  "Data and tiles": the warp's corrected sheets live in `data/warp/`, made by `sfzmaps warp` / the `warp-*`
  stages)

- [ ] **Step 1:** `git rm -r pipeline/scripts/sheet_warp`; `grep -rn "sheet_warp\|stand-in\|correction_patch\|SFZ_CORR" pipeline/src pipeline/tests AGENTS.md pipeline/README.md` → nothing.
- [ ] **Step 2:** Write the README section from `scripts/sheet_warp/README.md` at 7d208d6 (what it does, the
  steps in plain words, what it reads and where each comes from, what it writes, results), updated for
  `sfzmaps warp`, `data/warp/`, every placed sheet, and the stage. No R&D history, no /tmp, no scratch.
- [ ] **Step 3:** `uv run pytest -q` → PASS.
- [ ] **Step 4: Commit** — `git commit -m "sheet warp: the scripts are gone; the README documents sfzmaps warp"`

### Task 8 (controller, not a subagent): build, verify, hand over

- [ ] `uv run sfzmaps build -j 4` (only `warp-*` run; 357 sheets). Then `uv run sfzmaps build --dry-run` → nothing
  to build.
- [ ] Compare: every 1921–1990 sheet in the cache-maybe rebuild has the same chosen correction (`Gc`) as before;
  the 13 2000 sheets equal the 2026-11-14 run; list the 183 new sheets' scores and any uncorrected sheets with
  reasons.
- [ ] Seams by eye: render where corrected sheets meet for a few series (pictures saying what each shows).
- [ ] Fresh clone (empty folder, `pixi install`, `uv sync`, `sfzmaps pull`): commands and data from the branch
  only; `sfzmaps warp 1921` in it gives the same corrections.
- [ ] Commit `dvc.lock`; push; update PR #58's description from its live text. `sfzmaps ship` is left to the owner
  (it uploads).
