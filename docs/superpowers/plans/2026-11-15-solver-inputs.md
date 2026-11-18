# Solver Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The 2000 joint solver's inputs are made by `sfzmaps build` stages from fixed inputs: the table, the merged
page, the lots, the prints, the text detector and the label evidence (spec order, steps 1–4).

**Architecture:**
- A new subpackage, `sfzmaps.solver`.
- Each part is one module with a `build_*()` entry point, run by `sfzmaps solver <part>` and by one DVC stage.
- Stage code deps are the import closure of the entry module.
- Outputs go to `pipeline/data/solver/<part>/`.
- R&D scripts are ported, not imported.

**Tech Stack:** Python 3.12+, numpy, geopandas/shapely, OpenCV, scipy, PyTorch (detector), Typer (CLI), DVC (stages),
pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-solver-build-stages-design.md` (read it first; it is the authority).
Walls, the solve and the fresh-clone check are a second plan.

## Global Constraints

- **No reads outside a fresh clone.**
  - Package code and tests never read `pipeline/data/rnd`, another worktree, a scratch folder, or `/tmp`.
  - One-off comparisons against J7's files may read
    `/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/rnd` (read-only). They live in the
    scratch folder `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/solver/`,
    never in git, and their results go in the task report.
- **No model calls, ever.** Saved answers are read from `pipeline/saved-reads/answers/*.jsonl.gz` by `cache_key`.
  Never call `llm.ask_json` or anything that can reach the API. Never print or read `pipeline/.env`.
- **No per-item freshness stamps** (`fresh.record` / `*.inputs.json`), per issue #61. A stage recomputes its whole
  output.
- **Determinism:**
  - page, lots and prints are byte-identical between runs;
  - label fits are identical on one machine;
  - the detector's predictions are within 0.01.
  - Seed everything. Torch uses `torch.use_deterministic_algorithms(True)`.
- **Port sources:**
  - R&D scripts: `git show bubble-rnd/snapshot-w3:pipeline/scripts/bubble-rnd/k8/<name>.py` (`zstrokes.py` and
    `zlotchar2.py` are on `bubble-rnd/wall-field`).
  - The detector: `pipeline/scripts/joint_solver/textnet/` on this branch.
  - The solver's readers of these files: `pipeline/scripts/joint_solver/{data,labels,placement}.py`.
  - Keep each method and its named constants. Drop dead branches and unused outputs.
- **Lot index:** the lot index is the row order of the checked-in table file, everywhere.
- **The grid** (`sfzmaps.solver.grid`):
  - EPSG:2227 ft, 4 ft/px, width 11900, height 9825;
  - top-left corner x = 5979300.0, y = 2124100.0;
  - north up: col = (x − 5979300) / 4, row = (2124100 − y) / 4.
- **Quarter split:** x = 6000180.818639896, y = 2101727.4853113447. Index nw 0, ne 1, sw 2, se 3; east is x ≥ split,
  north is y ≥ split.
- **Style:**
  - Match `sfzmaps.sheetwarp`: plain functions, docstrings that say what a thing is.
  - Human-readable names, no R&D history in code or docs.
  - The pipeline README gets a section "The 2000 solver (`sfzmaps solver`)". Each task adds its part.
- **Running things:**
  - Tests: `cd pipeline && uv run pytest tests/<file> -q`, and the full suite before each commit.
  - Real runs: `cd pipeline && uv run sfzmaps solver <part>`. pixi's GDAL is not needed for these parts.
- **Commits** end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q
  ```
- **Don't touch** the old pipeline's frozen stages, `pipeline/data/` contents other than `data/solver/`, or the
  `joint_solver` package (the second plan moves it).

## Review Focus

1. **Lot order drift.** A table re-read in another order misaligns every code. Test: `lots.npz['table']` equals the
   tidied `zoning` of `table.parquet` row for row.
2. **Grid flips and half-pixel shifts** between the page, the lots and the prints. Test: a square lot's raster
   pixels sit where `grid.to_grid` puts its corners, and a page pixel sampled at a known map point lands on that
   point's grid pixel.
3. **A fit that sees its own quarter.** Each fit's training rows must exclude quarter q. Tests in tasks 8–9 assert
   that a row in q changes nothing in the q fit.
4. **A saved answer silently missing.** A key lookup that misses must be counted and reported, never guessed or
   asked live. Test: a print whose crop key is absent gets no third read and is counted.
5. **Stages too coarse or too fine.** A stage must rerun when its code or inputs change and not otherwise. Test in
   task 1: a solver stage's deps are its import closure plus `pyproject.toml`, `uv.lock` and its named reads, and no
   other `src/sfzmaps` file.

---

### Task 1: Stage machinery, the `sfzmaps solver` command group, PyMaxflow

**Files:**
- Modify: `pipeline/src/sfzmaps/build.py` (`Stage`, `stage_deps`, `stage_outs`, `stage_commands`, `needs_prepare`)
- Create: `pipeline/src/sfzmaps/solver/__init__.py`, `pipeline/src/sfzmaps/solver/paths.py`
- Modify: `pipeline/src/sfzmaps/cli.py` (a `solver` Typer group)
- Modify: `pipeline/pyproject.toml`, `pipeline/uv.lock` (PyMaxflow, exact version pinned)
- Test: `pipeline/tests/test_build.py`, `pipeline/tests/test_solver_paths.py`

**Interfaces:**
- Produces:
  - **`Stage` fields:**
    - `command: tuple[str, ...] = ()`: the `sfzmaps` argv tail;
    - `reads: tuple[str, ...] = ()`: data and input deps, relative to `pipeline/`;
    - `writes: tuple[str, ...] = ()`: outs;
    - `code: tuple[str, ...] = ()`: entry modules.

    A stage with `command` is a "command stage":
    - deps = the `fresh.code_files(list(code))` paths relative to `pipeline/` + `["pyproject.toml", "uv.lock"]` + reads;
    - outs = writes;
    - commands = `[list(command)]`;
    - it needs no `prepare()`.
  - **`sfzmaps.solver.paths`:**
    - `SERIES = "2000-digital-zoning"`, `QUARTERS = ("nw", "ne", "sw", "se")`;
    - `SOLVER_DIR = config.DATA_DIR / "solver"`;
    - `PAGE_DIR`, `LOTS_DIR`, `PRINTS_DIR`, `LABELS_DIR` (= `SOLVER_DIR / "page"` etc.), `text_dir(q) -> Path`
      (`SOLVER_DIR / f"text-{q}"`);
    - `TABLE_DIR = config.INPUTS_DIR / "zoning-2000"`;
    - `CUTS_FILE = config.INPUTS_DIR / "merged-page" / "2000-digital-zoning.json"`;
    - `LABEL_FIXES = config.INPUTS_DIR / "2000-digital-zoning" / "label_fixes.csv"`;
    - `SAVED_ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"`;
    - `sheet_names() -> list[str]`: the placed sheets of SERIES, via `sheetwarp.sheet.placed_sheets`.
  - **`cli.solver_app`:** a `typer.Typer` registered as `sfzmaps solver`. Later tasks add commands to it.

- [ ] **Step 1: Write failing tests.**
  - In `test_build.py`, `test_a_command_stage_depends_on_its_import_closure_and_reads`: build a
    `Stage("x", (), (), command=("solver", "page"), reads=("data/warp/2000-digital-zoning",),
    writes=("data/solver/page",), code=("sfzmaps.solver.paths",))`. Assert:
    - `stage_deps(st)` contains `src/sfzmaps/solver/paths.py`, `src/sfzmaps/config.py`, `pyproject.toml`, `uv.lock`
      and `data/warp/2000-digital-zoning`;
    - it does not contain `src/sfzmaps/districts` or `src/sfzmaps/sheetwarp/probabilistic.py`;
    - `stage_outs(st) == ["data/solver/page"]`;
    - `stage_commands` returns `[["solver", "page"]]` for it, with the stage injected into the table the way the
      existing tests do.
  - `test_needs_prepare_is_false_for_command_stages`.
- [ ] **Step 2:** Run them and see them fail.
- [ ] **Step 3: Implement.**
  - `fresh.code_files` takes module names. Make the paths relative to `config.PIPELINE_DIR`, sorted.
  - `sheet_names` needs `sheetwarp.sheet`, so make `paths.py` import it lazily inside the function. Otherwise every
    solver stage's closure would include the warp.
- [ ] **Step 4: Add PyMaxflow.**
  - Run `uv add "PyMaxflow==<latest>"` in `pipeline/`.
  - Check `uv run python -c "import maxflow; print(maxflow.__version__)"`.
  - Note its license (GPL-3.0) in the README's dependency notes, if there are any; otherwise in the new solver
    section.
- [ ] **Step 5:**
  - Run the full suite.
  - Check that `uv run sfzmaps build --write-dvc-yaml` leaves `dvc.yaml` unchanged, since no solver stage is
    registered yet.
  - Commit.

### Task 2: The April 2000 table, its provenance, the table corrections

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codes.py`, a copy of `pipeline/scripts/joint_solver/codes.py`.
- Create: `pipeline/src/sfzmaps/solver/table.py`.
- Create, by running the fetch: `pipeline/inputs/zoning-2000/april2000.geojson.gz` and
  `pipeline/inputs/zoning-2000/PROVENANCE.md`.
- Create: `pipeline/inputs/zoning-2000/table-corrections.csv`, copied byte for byte from the R&D
  `k8/judge2/final2.csv`.
- Create: `pipeline/inputs/zoning-2000/README.md`: what each file is, and that the corrections are used only in
  scoring.
- Modify: `pipeline/src/sfzmaps/cli.py` (`sfzmaps solver fetch-table`).
- Test: `pipeline/tests/test_solver_table.py`.

**Interfaces:**
- **`table.py`:**
  - `DATASET = "cfhj-8c6b"`.
  - `EXPORT_URL`: the Socrata whole-table GeoJSON export. Try
    `https://data.sfgov.org/api/geospatial/cfhj-8c6b?method=export&format=GeoJSON`. If that endpoint doesn't serve
    it, use the resource API in one request with `$limit` above the row count, and say which in PROVENANCE.md.
  - `fetch_table(dest: Path = paths.TABLE_DIR) -> dict` downloads once with httpx and writes
    `april2000.geojson.gz` (gzip, mtime 0, so the file is reproducible from the same bytes) and `PROVENANCE.md`. It
    returns the provenance:
    - url, dataset, name;
    - publisher "San Francisco Planning", license "PDDL 1.0";
    - fetched (UTC ISO);
    - rows;
    - sha256 of the uncompressed bytes;
    - the command.
  - `load_table() -> gpd.GeoDataFrame`:
    - reads the gz file;
    - `to_crs("EPSG:2227")`;
    - `zoning` = `fillna("").str.strip().str.upper()`;
    - geometry `shapely.make_valid`;
    - an `area` column;
    - row order kept.
  - `known_and_codes(table) -> tuple[np.ndarray, np.ndarray]`: `known` (non-blank, no "/") and each lot's code in
    the one spelling (`codes.canon(codes.tidy(z))`, "" where not known).
  - `corrected_codes(table) -> np.ndarray` applies `table-corrections.csv` the way `joint_solver/data.py
    _table_codes` does (verdict "A" sets column A on the listed blklots). It is used only for scoring.

- [ ] **Step 1: Write failing tests**, with a 3-lot GeoJSON fixture written to `tmp_path`, gzipped, and
  `paths.TABLE_DIR` monkeypatched:
  - `test_load_table_projects_tidies_and_keeps_order`;
  - `test_known_and_codes` (blank and "RH-1/RM-1" are unknown);
  - `test_corrected_codes_apply_only_verdict_a`;
  - `test_fetch_table_writes_gz_and_provenance`, with the download function monkeypatched to return fixture bytes;
    it asserts the sha256 and row count in `PROVENANCE.md`.
- [ ] **Step 2:** Run them and see them fail. Then implement. Then see them pass.
- [ ] **Step 3: The real fetch.**
  - Run `uv run sfzmaps solver fetch-table`.
  - One-off check, in the scratch folder: the fetched table equals R&D's `zon2000/april2000.geojson` row for row
    (same row count and order; blklot, zoning, and geometry equal by `shapely.equals_exact(tol=1e-9)` after both are
    read).
  - Report the result. If rows differ, report how, and don't paper over it.
  - Check the gz file is under 50 MB. If it isn't, report its size (it still goes in).
- [ ] **Step 4:** Add the README section's first lines (inputs). Run the full suite. Commit the code and all three
  input files.

### Task 3: The merged page (`solver-page`)

**Files:**
- Create: `pipeline/src/sfzmaps/solver/grid.py` (the grid constants, `to_grid(xy) -> (col,row)` float array,
  `to_map(colrow) -> xy`).
- Create: `pipeline/src/sfzmaps/solver/frame.py`, a port of `zframe2.py`.
- Create: `pipeline/src/sfzmaps/solver/page.py`, a port of `zmerge.py`.
- Create: `pipeline/inputs/merged-page/2000-digital-zoning.json`.
- Modify: `build.py` `stages()`: register `solver-page`.
- Modify: `cli.py`: `sfzmaps solver page`.
- Test: `pipeline/tests/test_solver_page.py`.

**Interfaces:**
- **`frame.find_frame(page: np.ndarray) -> np.ndarray`**: (4, 2) float corners TL, TR, BR, BL in page px, inset
  `FRAME_INSET_PX = 2.0`. The search windows and fit rules stay as named constants, with zframe2's values.
- **The cut-boxes file:**
  `{"2000-digital-zoning-p01": [{"box": [163,3830,713,3991], "what": "..."}, ...], ...}`.
  - p01: `[163,3830,713,3991]`, `[2898,3737,3146,3954]`, `[2228,526,3118,641]`;
  - p02: `[118,505,1008,631]`;
  - p08: `[2223,555,3124,675]`;
  - p09: `[2287,3874,3178,3995]`.

  Look at each box on its `page.png` and write what it covers.
- **`page.build_page() -> dict`** writes:
  - `PAGE_DIR/ink.png` (uint8, 0 = ink, 255 = paper, HEIGHT × WIDTH);
  - `sheets.png` (uint8, the count of sheets whose crop covers the pixel);
  - `frames.json` (each sheet's found quad).

  It returns a summary (ink px, covered px, seconds).
- **Per sheet:**
  - `SheetTransform.from_json(json.load(data/warp/<series>/<sheet>/transform.json), folder=that folder)`.
  - `to_pixel` every `STEP = 8` grid px, then `scipy.ndimage.map_coordinates(order=1, mode="nearest")` between,
    then round to the nearest page px.
  - Ink is `page.png < 128` inside the crop (the frame quad minus the cut boxes).
  - Combine sheets by union in bands of 512 rows.
- **`page.load_ink() -> np.ndarray`** (bool, True = ink) and **`page.load_sheets() -> np.ndarray`**.
- **The stage:** `Stage("solver-page", (), (), ("warp-2000-digital-zoning",), command=("solver", "page"),
  reads=("data/warp/2000-digital-zoning", "data/work/2000-digital-zoning", "inputs/merged-page"),
  writes=("data/solver/page",), code=("sfzmaps.solver.page",))`.

- [ ] **Step 1: Write failing tests.**
  - `test_find_frame_on_a_drawn_frame`: a synthetic 4200 × 3300 page with dark bands where zframe2's windows look.
    Corners within 1 px.
  - `test_grid_round_trip_and_north_up`: `to_grid` of the top-left corner is (0, 0); 4 ft south is row 1.
  - `test_a_sheet_lands_where_its_transform_says`: a fake transform object with `to_pixel` = a known affine. Ink
    pixels on the page appear at the expected grid px. Cut boxes and outside-the-frame ink don't appear. `sheets`
    counts 1 inside the crop and 0 outside.
  - `test_two_sheets_union_and_count`.
- [ ] **Step 2: Implement.** The per-sheet sampler takes any object with `to_pixel(world (n, 2)) -> page (n, 2)`, so
  the tests can pass a fake.
- [ ] **Step 3: Run it.**
  - Run `uv run sfzmaps solver page` and note the time.
  - Run it again: `ink.png` and `sheets.png` must be byte-identical (`cmp`).
  - One-off comparison with R&D `k8/MALLP_c1/raw.png`:
    - the share of R&D ink px that are ink here, and the reverse;
    - a picture (scratch folder) of one 2000 × 2000 px window with R&D ink red, new ink blue and both black. Say
      what it shows.
- [ ] **Step 4:**
  - Regenerate `dvc.yaml` (`uv run sfzmaps build --write-dvc-yaml`).
  - Add to the README.
  - Run the full suite.
  - Commit. Don't commit `dvc.lock` changes from a real build. The controller runs builds.

### Task 4: Lots, mapped area, scored lots, quarters (`solver-lots`)

**Files:**
- Create: `pipeline/src/sfzmaps/solver/lots.py`.
- Modify: `build.py`: `solver-lots`.
- Modify: `cli.py`: `sfzmaps solver lots`.
- Test: `pipeline/tests/test_solver_lots.py`.

**Interfaces:**
- **`QUARTER_SPLIT = (6000180.818639896, 2101727.4853113447)`**, with a comment saying where it came from (J7's
  scored lots' median centroid, kept so held-out numbers compare with J7's).
- **`quarter_of(xy: np.ndarray) -> np.ndarray`** (int8; nw 0, ne 1, sw 2, se 3).
- **`lot_rasters(table) -> tuple[np.ndarray, np.ndarray]`**: `lot` (int32, −1 outside lots; polygon exteriors
  filled with `cv2.fillPoly`, largest area first, on the grid) and `grown` (int32: every pixel's nearest lot, through
  `scipy.ndimage.distance_transform_edt(lot < 0, return_indices=True)`).
- **`neighbour_pairs(lot, grown) -> np.ndarray`** (int64 (n, 4): a, b, touching px, street px).
  - Built from 4-neighbour pixel pairs of `grown` with different values, a < b.
  - A pair of pixels counts as touching if both `lot` values are ≥ 0, as street otherwise.
  - Rows are sorted by (a, b).
- **`scored_lots(lot, mapped, known) -> np.ndarray`**: known, and more than half of the lot's `lot` pixels have
  `mapped`.
- **`build_lots() -> dict`** writes:
  - `LOTS_DIR/lots.npz`: `lot`, `grown`, `pairs`, `scored`, `known`, `quarter` (per lot, from the centroid), `table`
    (codes, as `known_and_codes`);
  - `LOTS_DIR/table.parquet`: the loaded table.

  It returns a summary that includes the recomputed median of the scored lots' centroids.
- **`load_lots()`**: a small dataclass over `lots.npz` (`n`, `a`, `b`, `border_ft = (touching + street) * 4`, ...).
- **The stage:** `Stage("solver-lots", (), (), ("solver-page",), command=("solver", "lots"),
  reads=("inputs/zoning-2000", "data/solver/page"), writes=("data/solver/lots",), code=("sfzmaps.solver.lots",))`.

- [ ] **Step 1: Write failing tests**, on a synthetic 3-lot table on a small grid (monkeypatch the grid size, or
  pass the shape):
  - `test_lot_raster_fills_largest_first_and_grows_to_nearest`;
  - `test_pairs_count_touching_and_street_pixels` (two lots sharing an edge, a third across a 3-px gap);
  - `test_scored_needs_known_code_and_majority_mapped`;
  - `test_quarter_of_uses_the_split`;
  - `test_lots_table_codes_follow_table_row_order` (Review Focus 1).
- [ ] **Step 2:** Implement. See the tests pass.
- [ ] **Step 3: Run it.**
  - Run `uv run sfzmaps solver lots`, twice, and `cmp` the npz files. Write npz with fixed key order; if np.savez
    isn't byte-stable, compare the arrays instead and say so.
  - One-off comparison with R&D `zon2000/truth_grid.npz` and `lotcode_run.npz`:
    - pair count, against 437,252;
    - pairs common by (a, b);
    - scored count, against 152,351;
    - lots scored here but not in J7, and the reverse, by count;
    - a picture of where the scored lots differ.
- [ ] **Step 4:** Update `dvc.yaml` and the README. Run the full suite. Commit.

### Task 5: Prints (`solver-prints`)

**Files:**
- Create: `pipeline/src/sfzmaps/solver/prints.py`.
- Create: `pipeline/src/sfzmaps/solver/saved.py`: reading saved answers by sheet and by `cache_key`.
- Modify: `build.py`: `solver-prints`.
- Modify: `cli.py`: `sfzmaps solver prints`.
- Test: `pipeline/tests/test_solver_prints.py`.

**Interfaces:**
- **`saved.answers(read: str) -> list[dict]`** reads `SAVED_ANSWERS/<read>.jsonl.gz`. `read` is e.g.
  `"text-on-every-sheet"` or `"experiments-2000/third-code-read"`. The `answer` strings are Python-literal dicts:
  parse them with `ast.literal_eval`.
- **`saved.by_key(read) -> dict[str, dict]`** keyed by `cache_key`.
- **`prints.terra_boxes(sheet) -> list[dict]`**: the sheet's Terra items in page px, decoded exactly as
  `pipeline/scripts/saved_reads/scripts_as_they_ran/textboxes.py` does (tile `page_box` offset, any de-duplication it
  does).
- **`prints.build_prints() -> dict`** writes:
  - `PRINTS_DIR/prints.parquet`, one row per print:
    - `sheet`, `page_x`, `page_y`, `x`, `y` (map ft);
    - `kind` (0 box + point, 1 box only, 2 point only, 3 turned-pass only);
    - the Terra box and read (code, confidence);
    - Sol's read;
    - the turned-pass read and edge;
    - the third read and the re-read (code, confidence, or empty);
    - `lab_lot` (the `grown` lot at the point);
    - `street` (`lot` < 0 at the point);
  - `PRINTS_DIR/terra_boxes.json` (every sheet's decoded Terra boxes, all kinds, for the text stages);
  - `PRINTS_DIR/summary.json` (counts by kind; third-read keys found and missing).
- **The method,** ported:
  - from `labread2.py`, first half: matching, and keeping a print inside the sheet's warp footprint
    (`data/warp/.../warped.footprint.geojson`) and the mapped area;
  - from `addprints.py`: joining the turned pass. Its box-to-page conversion is given in the positions file. The IoU
    > 0.3 cutoff and "the largest edge wins" stay, named.
  - from `read3.py` and `read3r.py`: the crops and prompts. Each key is `llm._key(llm._encode(crop), prompt, schema)`,
    with the scripts' own pydantic schema copied.
- **Positions:** page px → map ft with the corrected `SheetTransform.to_world`.
- **The stage:** `Stage("solver-prints", (), (), ("solver-lots",), command=("solver", "prints"),
  reads=("saved-reads/answers", "data/work/2000-digital-zoning", "data/warp/2000-digital-zoning",
  "data/solver/lots", "data/solver/page"), writes=("data/solver/prints",), code=("sfzmaps.solver.prints",))`.

- [ ] **Step 1: Write failing tests**, all on synthetic records and pages:
  - `test_box_and_point_match_as_labread2` (kinds 0, 1 and 2; the nearest point to the box centre wins);
  - `test_turned_box_joins_by_point_or_iou`;
  - `test_unmatched_turned_box_with_a_code_becomes_a_print`;
  - `test_third_read_found_by_recomputed_key`: build a record whose `cache_key` comes from `llm._key` of a synthetic
    crop.
  - `test_missing_third_read_is_counted_not_guessed` (Review Focus 4);
  - `test_terra_decode_offsets_tiles`.
- [ ] **Step 2:** Implement. See the tests pass. Nothing may make a network call: run the tests with `llm.cache_only()`
  in effect, or just never call into the client.
- [ ] **Step 3: Run it.**
  - Run `uv run sfzmaps solver prints`, twice. The parquet is identical (compare with pandas `equals`).
  - Report the third-read key match rate. **Below 95% for either read: stop and report (status BLOCKED)** with what
    differs (crop size, prompt, encoding).
  - One-off comparison with R&D `labread2_cwfwx` (nw file):
    - the count, against 1,906;
    - prints within 4 ft of an R&D print;
    - agreement of the top reads on matched prints;
    - the decoded Terra boxes against R&D `k8/pNN/textboxes_gpt-6.1-terra.json`, by count and exact equality.
- [ ] **Step 4:** Update `dvc.yaml` and the README. Run the full suite. Commit.

### Task 6: The text detector (`solver-text-{q}`)

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textnet/__init__.py`, `data.py`, `model.py`, `train.py`, `predict.py`,
  `evaluate.py`. Port `pipeline/scripts/joint_solver/textnet/`.
- Modify: `build.py`: four stages `solver-text-nw` … `solver-text-se`.
- Modify: `cli.py`: `sfzmaps solver text Q [--iterations N] [--device cpu|mps]`.
- Test: `pipeline/tests/test_solver_textnet.py`.

**Interfaces:**
- **`data.sheet_targets(sheet, q) -> SheetTargets`**: page ink (bool) and the text target (uint8 0/1). Also:
  - kind (uint8 0–3, 255 none);
  - the reading angle (cos and sin);
  - `known` (bool: loss applies; False on unknown areas and on pixels of quarter q);
  - `centre_ok` (pixels crops may centre on).

  Targets are the sheet's Terra boxes from `PRINTS_DIR/terra_boxes.json`, drawn as rotated rectangles in page px.
  Unknown areas are:
  - the boxes in `experiments-2000/text-positions/*.jsonl.gz` for that sheet;
  - discs of radius 30 px at Sol label points;
  - minus every Terra box.

  A pixel's quarter is `lots.quarter_of(transform.to_world(page px))`, computed on a coarse grid and interpolated.
- **`model.TextNet`**: the R&D U-Net, width 24, 5 levels, 7 output channels.
- **`train.train_quarter(q, iterations=8000, device=None, seed=0) -> Path`**:
  - the R&D losses, batches (12 × 320 px, half on text), augmentation (scale 0.6–1.6, quarter turns) and AdamW 2e-3
    with OneCycle;
  - `torch.manual_seed(seed)`, numpy `default_rng(seed)`, `torch.use_deterministic_algorithms(True)`,
    `torch.set_num_threads(4)`;
  - the device defaults to `mps` if available, else `cpu`; Task 7 settles it.
  - It writes `text_dir(q)/model.pt`.
- **`predict.predict_sheets(q) -> None`** writes `text_dir(q)/chance/<sheet>.png` (uint8, text chance × 255) for
  every sheet, tiled with overlap. The tiling is deterministic.
- **`evaluate.evaluate(q) -> dict`** writes `text_dir(q)/evaluation.json`, scored on quarter q's pixels only:
  - AUC and log-loss;
  - text ink found;
  - words found by kind;
  - pieces found where Terra drew no box;
  - the share of ink on table district lines called text (lines from `lots.npz` pairs whose table codes differ).
- **The command** runs train, predict, evaluate.
- **Stages:** `Stage(f"solver-text-{q}", (), (), ("solver-prints",), command=("solver", "text", q),
  reads=("data/work/2000-digital-zoning", "data/warp/2000-digital-zoning", "data/solver/prints",
  "data/solver/lots", "saved-reads/answers/experiments-2000/text-positions"), writes=(f"data/solver/text-{q}",),
  code=("sfzmaps.solver.textnet.train", "sfzmaps.solver.textnet.predict", "sfzmaps.solver.textnet.evaluate"))`.

- [ ] **Step 1: Write failing tests**, on a synthetic 640 × 640 page with two drawn words and their boxes:
  - `test_targets_mark_boxes_and_unknowns` (unknown minus Terra; quarter q pixels not known);
  - `test_model_output_shape`;
  - `test_three_iterations_twice_give_identical_weights_on_cpu`;
  - `test_heldout_pixels_carry_no_loss`: the loss's gradient is zero for a batch made only of quarter-q pixels.
- [ ] **Step 2:** Implement. See the tests pass on CPU, in under 30 s in total.
- [ ] **Step 3:** Don't run the real training (Task 7 does). Run `sfzmaps solver text nw --iterations 20 --device
  cpu` once, to see the whole path work on real pages, then delete that output folder.
- [ ] **Step 4:** Update `dvc.yaml` and the README. Run the full suite. Commit.

### Task 7: Detector determinism and device, then training

**Files:**
- Modify, as the spike decides: `pipeline/src/sfzmaps/solver/textnet/train.py` (the device default) and the README.
- The measurements go in the task report, not in git.

- [ ] **Step 1: The spike.** Run `train_quarter("nw", iterations=200)` twice on `mps` and twice on `cpu` (into the
  scratch folder, by monkeypatching `text_dir`). For each device, report:
  - seconds per iteration;
  - the maximum weight difference between the two runs;
  - the maximum difference in predicted chance on one sheet.

  If `use_deterministic_algorithms` raises on `mps`, report which op.
- [ ] **Step 2: Rule.**
  - Use `mps` if its two runs predict within 0.01; otherwise `cpu`.
  - If only `cpu` qualifies and 8,000 iterations would take over 8 h, stop and report (BLOCKED). Don't cut
    iterations.
  - Set the default and write the measurement in one README line.
- [ ] **Step 3:** Run the full suite. Commit.
- [ ] **Step 4 (controller):** Start `pixi run build` for the four text stages in the background (`-j 2` on `mps`),
  and go on with Task 8 meanwhile.

### Task 8: Label trust (`labels/readings_{q}.npz`)

**Files:**
- Create: `pipeline/src/sfzmaps/solver/trust.py`, a port of the fitting half of `addprints.py` in mode
  `K3=pqf`.
- Test: `pipeline/tests/test_solver_trust.py`.

**Interfaces:**
- **`trust.features(prints, lots) -> FeatureTable`**: rows of (print, candidate code), the 23 features J7 used. That
  is lotclf6's 17, the turned pass's 5 and the single-letter one; the 6 always-zero place-answer columns are dropped.
  The constants stay:
  - `lcf = -log(max(1-conf, 1e-3))`;
  - log height relative to the median;
  - distance `log(max(d,1)) - log(100)`.
- **`trust.fit(table, q) -> np.ndarray`**: weights by Newton, 200 iterations, tolerance 1e-8, ridge 1e-3; ridge 1000
  on the distance and in-a-street weights, 1e-3 on single-letter.
  - Training rows are prints whose lot has a known (uncorrected) table code and lies outside quarter q.
  - The target is that the candidate equals that code.
- **`trust.build_readings() -> None`** writes `LABELS_DIR/readings_{q}.npz` for each q:
  - `names` (code vocabulary);
  - `print_index`, `code_index`, `trust`: every (print, code) row with its fitted probability.

  These are the shapes `joint_solver/labels.py` reads as `lotclf6` (`cj`, `ck`, `cr`, `names`). Name them as they
  are, and note the mapping in the docstring for the second plan.

- [ ] **Step 1: Write failing tests:**
  - `test_feature_columns_match_names` (23 columns, no all-zero column on a synthetic set);
  - `test_newton_fit_matches_sklearn_with_same_ridge`: on a synthetic separable-ish set, compare with
    `sklearn.linear_model.LogisticRegression` with an equivalent C, within 1e-4;
  - `test_fit_ignores_rows_in_its_own_quarter` (Review Focus 3).
- [ ] **Step 2:** Implement. See the tests pass.
- [ ] **Step 3: Run it** on the real prints, twice: the outputs are identical.
  - One-off: the held-out log-loss of the trust per quarter, against J7's `lotclf6_cwfwx{q}` on prints matched
    within 4 ft. Report both.
- [ ] **Step 4:** Update the README. Run the full suite. Commit. The stage is registered in Task 9.

### Task 9: Street sides, lot size, strokes, and the `solver-labels` stage

**Files:**
- Create: `pipeline/src/sfzmaps/solver/streetside.py` (a port of `labside.py`), `lotsize.py` (a port of
  `zlotchar2.py`), `strokes.py` (a port of `zstrokes.py`) and `labels_stage.py` (`build_labels()` runs trust, street
  sides, lot size and strokes).
- Modify: `build.py`: `solver-labels`.
- Modify: `cli.py`: `sfzmaps solver labels`.
- Test: `pipeline/tests/test_solver_labels.py`.

**Interfaces:**
- **`streetside.build_streetsides() -> None`** writes `LABELS_DIR/streetside_{q}.npz`:
  - `lotB`, `onst`, `isfw`;
  - the four rates `rA_st`, `rB_st`, `rA_fw`, `rB_fw`, fitted on prints outside q against the uncorrected table.

  The method stays: the nearest centreline from `data/ref/streets.parquet`, freeway = `st_type` HWY or RAMP or a
  name matching `^(HWY|I-)\s*\d`, the side at ±5 ft along the normal, and the far-side lot searched from 100 ft,
  doubling up to 3,200 ft.
- **`lotsize.build_lotsize() -> None`** writes `LABELS_DIR/lotsize.npz` (`codes`, `edges` (37), `PMI` (4, n codes,
  36)).
  - Log-area edges `np.arange(5, 14.01, 0.25)`; half-neighbour blending; α = 36.
  - Per quarter, trained on scored lots with a known uncorrected code outside q.
- **`strokes.build_strokes() -> None`** writes `LABELS_DIR/strokes.npz` (`xy`, `w`, `off`, and per stroke `a`, `b`,
  `length`, `wmed`, `wiqr`, `spur`, `dega`, `degb`), from `page.load_ink()`.
  - Skeletonize. Nodes are skeleton pixels of degree ≠ 2.
  - Width = 2 × distance transform × 4 ft. A spur is a dead end shorter than its median width.
  - Positions in map ft.
- **The stage:** `Stage("solver-labels", (), (), ("solver-prints",), command=("solver", "labels"),
  reads=("data/solver/prints", "data/solver/lots", "data/solver/page", "data/ref"),
  writes=("data/solver/labels",), code=("sfzmaps.solver.labels_stage",))`.

- [ ] **Step 1: Write failing tests:**
  - `test_streetside_finds_the_far_lot_across_a_street` (two lots facing across a synthetic centreline);
  - `test_streetside_rates_exclude_quarter_q`;
  - `test_lotsize_pmi_shape_and_smoothing` (a code with all lots in one bin gets positive PMI there);
  - `test_strokes_of_an_L_and_a_spur` (a drawn L has 2 strokes, widths match the drawn width, a short stub is a spur).
- [ ] **Step 2:** Implement. See the tests pass.
- [ ] **Step 3: Run it.**
  - Run `uv run sfzmaps solver labels`, twice: identical outputs.
  - One-off comparisons:
    - street-side rates against J7's (nw 0.828 / 0.810);
    - lot-size PMI against `zlotchar2.npz`, the largest difference on shared codes;
    - stroke count and miles against R&D's raw-page strokes, if they exist (otherwise against the cleaned page's
      84,130 strokes / 1,315 mi; say which).
- [ ] **Step 4:** Update `dvc.yaml` and the README. Run the full suite. Commit.
