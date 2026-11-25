# From the Finder to Placement: Implementation Plan (steps 0, 2, 3, 4)

> **For agentic workers:** executed task by task by subagents (superpowers:subagent-driven-development), with a
> review after each task. Steps use checkbox (`- [ ]`) syntax. Tasks 1-19 are the finder's own plan
> (`2026-11-23-street-and-block-finder-first-chain.md`); this plan starts at Task 20.

**Goal:** Put the finder's blocks in place of the traced ones, run today's corner matching and warp on them, unchanged,
in a separate trial folder for four sheets, and show the owner the result beside a proved rerun of today's blocks.

**Architecture:** Two data roots under `data/find/trial/<trial>/`, `traced` and `finder`. Each one is a whole
`SFZMAPS_DATA_DIR` holding copies of the four sheets' input files. Today's `sfzmaps run` and `sfzmaps warp` run in
each root without any change to their code. The `finder` root differs in one file per sheet, `blocks_px.geojson`,
which `sfzmaps find blocks` writes from the stored read (`find.areas`, then the tracing step's outline and
corner-grading code). New code goes only in `sfzmaps/find/` (`trial.py`, `placeblocks.py`, `look.py`) and `cli.py`.

**Spec:** `docs/superpowers/specs/2026-11-24-finder-to-placement-design.md`. Read it first. Then read the code audit
`.superpowers/sdd/2026-11-23-street-and-block-finder-first-chain/placement-audit-2026-11-24.md`, sections A, E and F.

## Global Constraints

- **The trial never writes into today's data.** Nothing under `data/work`, `data/warp` or `data/out` changes. Trial
  roots live under `data/find/trial/`. Work files are copied into them, never linked. Only `raw/`, `ref/` and `cache/`
  are links.
- **No change to corner matching, the warp or the tracing step.** `linework.py`, `runner.py`, `refine.py`,
  `anchors.py` and `sheetwarp/` stay byte-identical. `linework.py` is a freshness input of the blocks stage and a DVC
  dependency of every edition and warp stage.
- **$0, no LLM/API calls.** Every pipeline command in the trial runs with `SFZMAPS_MAX_USD=0`. The label and page
  steps must be skipped as fresh from the copied saved answers (`*.answers` sidecars). A runner log line saying
  classify or labels "done" means something is wrong: stop.
- No `sfzmaps build`. Nothing under `pipeline/data/` is committed. Never push. Before every commit run
  `pgrep -f "sfzmaps build"` and commit only if it prints nothing. Commit with `git add <paths>`, never `-A`. Commit
  messages end with the two attribution lines the controller gives.
- **No new hand-picked number decides anything.** The tracing step's existing numbers are reused as they are. A
  yes/no taken from a chance is "the more likely answer". Picture sizes are picture settings, and a comment must say
  so.
- **Nothing is retrained. One run per change, no repeats with only a seed changed.** The model is run `all-pieces`.
- **Duplicate scans are never read** (`solver.hatching.duplicate_scans`).
- **Placement results are shown as the scan warped onto today's map, with today's block lines thin on top.** Never
  show today's map drawn onto the scan.
- **Result pictures are at the scan's own resolution and titled "RESULT:"** (a debugging picture is titled
  "RESULT (debugging picture):"). Each picture goes to the owner with SendUserFile. The caption says first that it is
  a result, not a training image, and then what it is for.
- **Targeted tests only:** `cd pipeline && OMP_NUM_THREADS=1 uv run pytest -q tests/test_find_trial.py
  tests/test_find_placeblocks.py`, plus `tests/test_find_*.py` for any finder module you change and
  `tests/test_build.py` when `cli.py` changes. Do not run the whole suite; CI runs it.
- **Run the pipeline with the pinned tools:** `pixi run ...` from `pipeline/`, or inside `pixi shell`. Run
  independent sheets in parallel. Only one GPU job runs at a time: check `pgrep -fl "sfzmaps|hatchnet"` before
  `find read` or `warp`.
- **No pass line.** The trial reports and does not stop itself on a number. Anything worse is looked into, not
  explained away.
- Before a run that writes whole-page files, check `df -h /System/Volumes/Data`. About 100 GB was free on 2026-11-25.
  Each trial root takes about 50 MB per sheet.

## Review Focus

1. **A trial root that reaches today's files.** A root inside `data/work` or `data/warp`, or a work file that is a
   link (hard or symbolic) to today's, would let the runner's in-place writes change today's placements. Test in
   Task 20.
2. **The runner re-traces the finder's blocks.** The blocks stage is skipped only while `page.png`, `page.json` and
   `linework.py` are unchanged (audit E.1), and `--force` re-traces. The comparison must refuse a `finder`-root
   blocks file whose sha256 is not the one `find blocks` recorded. Test in Task 25.
3. **Half-pixel shifts.** Outline vertices must sit at pixel centres (cv2 contour convention, audit F). A polygon
   that uses pixel edges moves every corner by half a pixel. Test in Task 22.
4. **A blocks file placement cannot read.** The corners must parse as `runner.do_refine` parses them: N x 2, with
   `null` for a value that is not finite. The polygons must parse as `shapely.geometry.shape`. Extra properties and
   the top-level `finder` key must be ignored by both readers. Test in Task 22.
5. **The control judged on ground the model learned.** The share of the control's ground under the run's
   lot-labelled training sheets, and the run's hand crops on each trial sheet, must be measured and reported apart.
   Test in Task 20.

## Files

| File | Responsibility |
|---|---|
| `pipeline/src/sfzmaps/find/trial.py` (new) | Trial roots, the numbers read from a root, reproduction check, trained-ground share, comparison measures and placement pictures |
| `pipeline/src/sfzmaps/find/placeblocks.py` (new) | Areas into placement's blocks file: edge-cut areas, outlines (the tracing step's per-piece code, mirrored), per-block read confidence, joined areas, where the outline sits |
| `pipeline/src/sfzmaps/find/look.py` (modify) | `outlines_picture` |
| `pipeline/src/sfzmaps/find/__init__.py` (modify) | One docstring paragraph: `trial` and `placeblocks` are the placement side (they read placement outputs only to compare, never `labels.json`, and nothing of theirs reaches the model) |
| `pipeline/src/sfzmaps/cli.py` (modify) | `sfzmaps find blocks`, `sfzmaps find outline-offset`, and a sub-app `sfzmaps find trial {root, numbers, compare}` (`find_app.add_typer(trial_app, name="trial")`) |
| `pipeline/tests/test_find_trial.py`, `tests/test_find_placeblocks.py` (new) | Tests |
| `docs/research/finder-to-placement-trial-<date>.md` (new, Task 26) and `docs/research/finder-to-placement-handoff-<date>.md` (new, Task 27) | The trial's report; the hand-off note |

Trial layout (`trial = "all-pieces"`):
`data/find/trial/all-pieces/{traced,finder}/` are the two data roots, each holding `raw ref cache` links,
`work/<series>/<sheet>/` and `warp/<series>/<sheet>/`. Beside them: `numbers.json`, `outline_offsets.json`,
`blocks/<series>/<sheet>.json` (counts and the blocks file's sha256), `compare.json`, and
`look/<series>/<sheet>__outlines.png` and `look/<series>/<sheet>__<place>.png`.

Trial sheets (`SHEETS` in `trial.py` holds no ids; the commands take `--sheet`): `1921/20180102100525362`
(Section 12), `1921/20180102094348509`, `1921/20180102094142210`, and the control
`2000-digital-zoning/2000-digital-zoning-p08`.

---

### Task 20: Trial roots and the numbers read from them

**Files:** Create `find/trial.py` and `tests/test_find_trial.py`. Modify `cli.py` and `find/__init__.py`.

**Interfaces (produces):**
- `TRIAL = pages.OUT / "trial"`, `ARMS = ("traced", "finder")`, and `arm_root(trial: str, arm: str) -> Path`.
- `KEPT = ("page.png", "page.json", "labels.json", "blocks_px.geojson", "warped_paper.tif",
  "warped_paper.footprint.geojson")`. Each file is copied with its `.inputs.json` and `.answers` sidecars where they
  exist.
- `prepare(root: Path, sids: list[SheetId], source: Path) -> list[Path]` does the following:
  - It makes `root`, with `raw`, `ref` and `cache` as symlinks to `(source / x).resolve()`.
  - It copies (`shutil.copy2`) the `KEPT` files and their sidecars of each sheet into
    `root/work/<series>/<sheet>/`. Nothing made after the blocks is copied (`rough.json`, `refine*.json`,
    `changes.geojson`, `transform.json`, `score.json`, `warped.tif`, `warped.footprint.geojson`), so both arms
    compute those again. It returns the copied files.
  - `ValueError` names why when `root` is `source`, or lies inside `source/work`, `source/warp` or `source/out`. It
    does the same for a sheet that is a duplicate scan or has a review or a seed (`inputs.review_path`,
    `inputs.seed_path`: such a sheet does not test the seam, audit E.3), or that lacks a `KEPT` file. In every case
    nothing is written.
  - `FileExistsError` if `root/work` exists. A root is prepared once; delete the arm folder to redo it.
- `numbers(root: Path, sid: SheetId) -> dict` reads only `root/work/...` and `root/warp/...` and returns these keys:
  `blocks` (features), `good_corners` (corners), `control_points` (enabled GCPs of `transform.json`), `passed`,
  `reason`, `holdout_p50_ft`, `holdout_p95_ft`, `warp` (`chosen`, or the `uncorrected` reason) and `within1` (the
  chosen kind's `within1_*`, else None). Today's numbers are `numbers(config.DATA_DIR, sid)`: the same function for
  every column.
- `reproduced(a: Path, b: Path, sid) -> dict[str, bool]` compares the sheet in two roots. Keys: `transform` (GCPs),
  `refine` (matches), `score`, `warp_scores` (ignoring `device`), and `correction` (the arrays of `correction.npz`).
  The test is exact equality, with no tolerance.
- `trained_ground(sid, run_folder: Path, source: Path) -> dict` returns `lot_labelled_share`: the share of the sheet's
  `source/warp/.../warped.footprint.geojson` area that lies under the union of the warped footprints of the run's
  lot-labelled sheets (`log.json` items whose name has no `__`). It also returns `hand_crops`: the run's training
  items on this sheet.
- CLI: `sfzmaps find trial root --trial T --arm traced|finder --sheet S/N ...` calls `prepare` and prints the files
  copied. `sfzmaps find trial numbers --trial T [--run all-pieces]` prints one row per sheet and root (today, each
  arm present), the `reproduced` flags of `traced` against today, and the control's `trained_ground`. It writes
  `TRIAL/T/numbers.json`.

- [ ] **Step 1: Write the failing tests.** Fixture `src`: a data root in `tmp_path` holding `raw/ ref/ cache/` and
  `work/s/a/` with every `KEPT` file, the sidecars and every downstream file.
  ```python
  def test_a_trial_root_holds_copies_of_the_inputs_and_nothing_after_them(src, tmp_path):
      root = tmp_path / "t" / "traced"; trial.prepare(root, [("s", "a")], source=src)
      d = root / "work/s/a"
      assert {p.name for p in d.iterdir()} == {*KEPT, *sidecars_present_in_src}
      assert not any((d / f).is_symlink() or os.stat(d / f).st_ino == os.stat(src / "work/s/a" / f).st_ino for f in KEPT)
      (d / "page.json").write_text("x"); assert (src / "work/s/a/page.json").read_text() != "x"
      assert all((root / x).is_symlink() and (root / x).resolve() == (src / x).resolve() for x in ("raw", "ref", "cache"))
      assert not any((d / f).exists() for f in ("rough.json", "refine.json", "transform.json", "score.json", "warped.tif"))
  def test_a_root_in_todays_data_is_refused(src):            # src, src/work/x, src/warp/x, src/out/x -> ValueError, nothing made
  def test_a_sheet_that_does_not_test_the_seam_is_refused(src, tmp_path, monkeypatch):  # duplicate / review / seed: ValueError naming s/a and why; root has no work/
  def test_a_root_is_prepared_once(src, tmp_path):            # second prepare -> FileExistsError
  def test_numbers_are_read_from_the_root(src):               # 3 features, corners [[1,2],[None,3]] -> 2; 2 of 3 GCPs enabled -> 2;
                                                              # warp scores chosen "probabilistic", within1_probabilistic 0.627 -> 0.627;
                                                              # an "uncorrected" scores.json -> warp "uncorrected: ...", within1 None
  def test_reproduced_says_which_file_differs(src, tmp_path): # one GCP moved -> transform False, the rest True; only "device" differs -> warp_scores True
  def test_trained_ground_is_the_share_under_lot_labelled_sheets(tmp_path):
      # sheet footprint box(0,0,10,10); a lot-labelled item's footprint box(5,0,15,10) -> 0.5;
      # item "train/s/a__0_0_64_64" -> in hand_crops; "train/s/b__..." -> not
  ```
- [ ] **Step 2:** Run the tests. Expected: they FAIL (no module `sfzmaps.find.trial`).
- [ ] **Step 3:** Implement the interfaces above in `find/trial.py`, the `trial` sub-app in `cli.py`, and the
  docstring paragraph in `find/__init__.py`.
- [ ] **Step 4:** Run the tests and `tests/test_build.py`. Expected: PASS.
- [ ] **Step 5:** Commit: `find: trial data roots for placement (copies, never links), the numbers read from a root`.

### Task 21: Step 0: rerun today's placement in the trial folder (run, no code)

- [ ] **Step 1:** Run `pgrep -fl "sfzmaps|hatchnet"` and `df -h /System/Volumes/Data`. Then, from `pipeline/`:
  `uv run sfzmaps find trial root --trial all-pieces --arm traced --sheet 1921/20180102100525362
  --sheet 1921/20180102094348509 --sheet 1921/20180102094142210 --sheet 2000-digital-zoning/2000-digital-zoning-p08`.
- [ ] **Step 2:** Run the four sheets in parallel, one process each, with `R=$PWD/data/find/trial/all-pieces/traced`:
  `SFZMAPS_DATA_DIR=$R SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year <series> --sheet <sheet>`. Passing
  looks like this: exit 0; render, classify, labels and blocks log "skipped"; rough, refine, changes, score and warp
  log "done"; and `git status` shows nothing under `pipeline/` changed. If classify or labels says anything else,
  stop and report.
- [ ] **Step 3:** `SFZMAPS_DATA_DIR=$R SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps warp 1921 2000-digital-zoning`.
  Passing: every one of the four sheets prints `corrected (...)` or `uncorrected: ...`, and no other sheet is
  touched.
- [ ] **Step 4:** `uv run sfzmaps find trial numbers --trial all-pieces`. Report the table: today against `traced`
  for each sheet, with the `reproduced` flags. Today's values from the audit and from disk are: Section 12 751 / 371
  / 261; 094348509 682 / 135 / 79; 094142210 584 / 1,557 / 1,402; p08 292 / 556 / 492 (passed, held-out
  5.01 / 12.28 ft, probabilistic, within1 0.627). Also report the control's `trained_ground`. A read-only check on
  2026-11-25 gave 0.111 of p08's ground under lot-labelled training sheets (p01 0.032, height-p07 0.060, su-p10 0.020)
  and no hand crop on p08. Section 12 has one training crop, `train/1921/20180102100525362__3100_2570_1024_1024`.
- [ ] **Step 5:** If any flag is False, find the reason before Gate A. First compare the versions recorded in today's
  `data/warp/<s>/<n>/transform.json.inputs.json` with `sheetwarp.run.numeric_versions()`. Write the reason down with
  its evidence. Do not change code to force a match.
- [ ] **Owner gate A.** Send the table and any reason found. Name the control with today's numbers and its 11%
  trained-ground share, and ask whether the control stands. Wait for a plain go.

### Task 22: Outlines from the finder's areas

**Files:** Create `find/placeblocks.py` and `tests/test_find_placeblocks.py`.

**Where "not street" is taken:** only in `areas.areas(street, keep)`, as a street chance of at most one half (the
more likely answer). `placeblocks` takes no other cut on a chance. It keeps the chances themselves in each block's
`properties` (below), so a later step can weigh blocks by how sure the read was.

**Interfaces (produces):**
- `edge_cut(lab: np.ndarray, keep: np.ndarray) -> np.ndarray` returns the sorted ids of areas with a px 4-adjacent to
  a px off `keep`, or on the page's edge (the design's call 5).
- `outlines(lab: np.ndarray, drop: np.ndarray, ink: np.ndarray, inset_px: float) -> tuple[list[Polygon], list[int]]`
  applies the tracing step's per-piece outline code to every area not in `drop`. The code is mirrored from
  `linework._extract` (the loop after `bad = ...`), because `linework.py` must not change. It works like this:
  - It uses `linework`'s own constants and helpers (`_odd`, `_disk`, `_polygon`, `_drop_lettering`, `sheet_scale`,
    `street_width`, `estimate_street_px`, `CLEAN_PX`, `MIN_BLOCK_AREA`, `MIN_BLOCK_SOLIDITY`,
    `POLY_EPS_STREET_FRAC`, `CLOSE_STREET_FRAC`, `SIMPLIFY_PX`, `SMALL_EPS_FRAC`).
  - It sets the scale as `extract_blocks` does: a first pass at scale 1, then `sheet_scale(estimate_street_px)` when
    measured, and a second pass if the scale is not 1.
  - `sw = street_width(~ink & keep)`.
  - It then insets every polygon by `inset_px` (`buffer(-inset_px, join_style="mitre")`, dropping what is not a
    `Polygon`). This replaces tracing's white/dark choice (Task 23).
  - It returns the polygons and the area id of each. A comment names the mirrored lines.
- `outline_confidence(lab: np.ndarray, ids: list[int], chances: np.ndarray) -> list[dict]` returns, for each id:
  - `outline_px`: the area's rim px, meaning its own px that are 4-adjacent to a px outside it.
  - `outline_street_p10` and `outline_street_p50`: the street chance on the px 4-adjacent outside the rim, within the
    frame cut. This says how sure the read is that a street parts this block from the next.
  - `outline_edge_p50`: the block-edge chance on the rim px.
- `joined(lab: np.ndarray, points_px: np.ndarray) -> dict` returns `areas_with_two_or_more` and `blocks_in_them`. It
  is for reporting only: the points are the representative points of the year's lots dissolved by block, placed on
  the page by today's placement (Task 24).
- `blocks_file(polys, ids, confidence, finder: dict) -> dict` builds today's schema (audit A). Each feature's
  `properties` are `{"area": id, **confidence}`; `corners` holds `linework.good_corners_px(polys)` with `null` for a
  value that is not finite (as `runner._num`); and a top-level key `finder` holds the read, run, `inset_px` and
  `"not_street": "street chance <= 0.5 (find.areas)"`.

- [ ] **Step 1: Write the failing tests.** Synthetic pages are drawn in the test. Two 200 x 120 px blocks are
  outlined in 2 px ink with a 24 px street between them; `lab` is their areas.
  ```python
  def test_corners_sit_at_pixel_centres():          # area px cols 50..249, rows 40..159 -> vertices within 1 px of
                                                     # (50,40),(249,40),(249,159),(50,159); never at 250 or 160
  def test_an_area_on_the_rim_or_the_page_edge_is_cut():   # areas 1 (touches ~keep), 3 (row 0) -> edge_cut == [1, 3]
  def test_inset_moves_every_side_in():              # inset_px=2 -> vertices within 1 px of (52,42),(247,42),(247,157),(52,157)
  def test_two_blocks_joined_by_a_neck_are_one_outline_with_fewer_good_corners():
      # the two blocks apart: 2 polygons, 8 good corners; joined by a 10 px neck: 1 polygon, fewer than 8
  def test_the_confidence_carries_the_chance():      # street 0.9 round area 1, 0.6 round area 2 (both called street);
      # edge 0.8 / 0.3 on their rims -> outline_street_p50 0.9 / 0.6 and outline_edge_p50 0.8 / 0.3 (within 1/255)
  def test_joined_counts_areas_holding_two_or_more_blocks():  # points 2 in area 1, 1 in area 2 -> (1, 2)
  def test_the_blocks_file_reads_as_placement_reads_it():
      d = json.loads(json.dumps(placeblocks.blocks_file(polys, ids, conf, {...})))
      c = np.array([[np.nan if v is None else v for v in c] for c in d["corners"]], float).reshape(-1, 2)  # as runner.do_refine
      np.testing.assert_allclose(c, np.asarray(linework.good_corners_px(polys), float).reshape(-1, 2))
      assert [shape(f["geometry"]).equals(p) for f, p in zip(d["features"], polys)] == [True] * len(polys)
      assert d["finder"]["not_street"].startswith("street chance <= 0.5")
  ```
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the interfaces above in `find/placeblocks.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Confirm `git diff --stat main -- pipeline/src/sfzmaps/linework.py`
  is empty.
- [ ] **Step 5:** Commit: `find: the finder's areas as placement's blocks (the tracing step's outline code), with each
  block's read confidence`.

### Task 23: Where the outline sits

**Interfaces (produces), in `placeblocks.py`:**
- `outline_offset(street: np.ndarray, keep: np.ndarray, traced: list[Polygon]) -> dict` returns `median_px`,
  `p25_px`, `p75_px` and `points`. Each traced exterior is sampled every px, with pixel centres at integers. At each
  sample it measures the signed distance to the boundary of `areas`' ground (street chance <= 0.5, in `keep`): plus
  half a px less than the distance transform inside the ground, minus the same outside it. Positive means the
  finder's area reaches past the traced outline.
- `inset_px(offset_px: float, half_stroke_px: float) -> float` takes the nearer of the design's two answers. It
  returns `half_stroke_px` when `|offset - half_stroke| < |offset|`, else 0. A tie keeps 0.
- CLI: `sfzmaps find outline-offset --run R --trial T --sheet S/N ...` reads each sheet's stored read
  (`wholesheet.paths`), its frame cut, and today's traced polygons (`data/work/.../blocks_px.geojson`). It groups the
  sheets by era (`pages.source(series).era`) and writes `TRIAL/T/outline_offsets.json`. Each era entry holds the
  sheet, the offset dict, the sheet's `half_stroke_px = linework.stroke_width(linework.binarize(img)) / 2`, and
  `outer_edge: inset_px(...) > 0`. It refuses a sheet with no stored read, naming the `find read` command.

- [ ] **Step 1: Write the failing tests.**
  ```python
  def test_an_outline_inside_the_area_has_a_positive_offset():  # ground px 40..139 square; traced box(43,43,136,136) -> median 3.5 ± 0.5
  def test_the_same_outline_has_no_offset():                    # traced box(40,40,139,139) -> |median| <= 0.5
  def test_an_outline_outside_the_area_is_negative():           # traced box(37,37,142,142) -> median -3.5 ± 0.5
  def test_inset_takes_the_nearer_answer():
      assert placeblocks.inset_px(2.8, 3.0) == 3.0 and placeblocks.inset_px(0.4, 3.0) == 0.0
      assert placeblocks.inset_px(1.5, 3.0) == 0.0
  ```
- [ ] **Step 2:** Run (FAIL), implement, run (PASS). Commit:
  `find: where the finder's block outlines sit against the traced ones, per era`.
- [ ] **Step 3: Reads (GPU, one job).** `uv run sfzmaps find read --run all-pieces --sheet 1921/20180102094142210
  --sheet 1921/20180102094149543 --sheet 2000-digital-zoning/2000-digital-zoning-p07`. Expected: one line per sheet
  and exit 0. The other three trial sheets were read in step 1; check their `.npz` files exist under
  `data/find/runs/all-pieces/sheets/`.
- [ ] **Step 4: Measure.** `uv run sfzmaps find outline-offset --run all-pieces --trial all-pieces
  --sheet 1921/20180102094149543 --sheet 2000-digital-zoning/2000-digital-zoning-p07`. These sheets were picked
  because tracing works on them and neither is a trial sheet. Report, per era: the median offset with p25 and p75,
  the half stroke, and `outer_edge`. The numbers are shown at Gate B with the pictures.

### Task 24: `sfzmaps find blocks`, the outlines picture, and the four sheets (step 2)

**Interfaces (produces):**
- `look.outlines_picture(lum, traced: list[Polygon], traced_corners: np.ndarray, found: list[Polygon],
  found_corners: np.ndarray, title: str) -> np.ndarray` returns two panels at the page's own size, side by side and
  separated by `GAP`: traced on the left, finder on the right. Each panel shows outlines 1 px wide and good corners
  as dots, under a key, with the title starting "RESULT (debugging picture):".
- `placeblocks.write_sheet(sid, run_folder: Path, root: Path, inset_px: float) -> dict` does the following:
  - It reads the stored read (`read.load`), the frame cut (`pages.load_keep`) and `page.png` (as `runner` reads it:
    `cv2.IMREAD_UNCHANGED`, ink by `linework.binarize`).
  - It computes `lab = areas.areas(street, keep)`, drops `edge_cut`, runs `outlines`, `outline_confidence` and
    `blocks_file`, and writes `root/work/<series>/<sheet>/blocks_px.geojson`. It leaves that file's copied
    `.inputs.json` as it is, so the runner does not re-trace.
  - It returns the counts: `areas`, `edge_cut`, `dropped_by_cleaning`, `blocks`, `good_corners`,
    `areas_with_two_or_more`, `blocks_in_them`, `inset_px`, and `sha256` of the written file.
- CLI: `sfzmaps find blocks --run R --trial T --sheet S/N ...` uses the `finder` arm root and `outline_offsets.json`.
  For each sheet: inset = the sheet's own half stroke if its era's `outer_edge`, else 0. It refuses, naming the
  missing step, when the arm root, the read or the era's offset is missing. It writes
  `TRIAL/T/blocks/<series>/<sheet>.json` (the counts) and `TRIAL/T/look/<series>/<sheet>__outlines.png`. It prints
  one row per sheet: today's traced blocks and good corners beside the finder's, and the joined areas.

- [ ] **Step 1: Write the failing tests** (in `test_find_placeblocks.py`):
  `test_the_outlines_picture_shows_each_side_on_its_own_panel` checks that the shape is `(band + h, 2w + GAP, 3)`,
  that a px on a found outline is the outline colour in the right panel, and that the same px in the left panel is
  not when no traced polygon is there. `test_blocks_refuses_an_unprepared_arm` (CliRunner) checks for exit 1 and a
  message naming `sfzmaps find trial root`. `test_write_sheet_keeps_the_blocks_sidecar` checks that the copied
  `.inputs.json` is byte-identical after the write, and that `sha256` is the sha of the file on disk.
- [ ] **Step 2:** Run (FAIL), implement, run (PASS, plus `tests/test_build.py`). Commit:
  `find: sfzmaps find blocks writes a sheet's blocks file from its read, with a debugging picture beside the traced`.
- [ ] **Step 3: Run.** `uv run sfzmaps find trial root --trial all-pieces --arm finder --sheet ...` (the four
  sheets), then `uv run sfzmaps find blocks --run all-pieces --trial all-pieces --sheet ...` (the four sheets).
- [ ] **Step 4: Look first.** Open each `__outlines.png` before reading the counts. For each sheet, write down where
  the finder's outlines are right and where they are wrong (joined blocks, outlines off the print, blocks lost to
  the cleaning or to the rim). After that, read the counts.
- [ ] **Owner gate B.** Send the four outlines pictures, each captioned as a result (a debugging picture) and what
  it shows. Send the counts table: traced against finder blocks, good corners, joined areas and the blocks in them,
  edge-cut and cleaned-away areas. Send the outline offsets from Task 23. Say what is wrong. Wait for a plain go.

### Task 25: The comparison: measures and placement pictures

**Interfaces (produces), in `trial.py`:**
- `reach_ft(points_world: np.ndarray, gcps_world: np.ndarray) -> dict` returns `p50` and `p90`: the distance in feet
  from each point to its nearest control point (`scipy.spatial.cKDTree`).
- `side_points(sid) -> np.ndarray` returns the world points of `sheetwarp.targets.block_side_points(Sheet(*sid)).pts`
  under today's data root. They are the same points the warp's within-1-px score uses, and the same set is used for
  every column.
- `places(sid, source: Path) -> dict[str, tuple[float, float, float, float]]` returns three boxes in feet, each
  `WINDOW_PX` (a picture setting, 1200) times the resolution of today's `warped.tif`, on a grid over the sheet's
  footprint:
  - `worst`: the box holding the most side points more than 1 px from the print under today's corrected placement
    (`targets.measure(targets.fit_targets(sheet), sides, Correction.load(...))`: the warp's own measure);
  - `busiest`: the box holding the most side points;
  - `centre`: the box holding the footprint's centroid.
- `placed_picture(before_tif: Path, after_tif: Path, box_ft, lines, title) -> np.ndarray` reads each `warped.tif`
  inside the box at its own resolution (`rasterio`, a window), draws today's block lines (`datasf` reference blocks'
  boundaries) 1 px over each, and puts the two side by side, before on the left. The title begins "RESULT:".
- `check_blocks(trial: str, sid) -> None` raises `ValueError` naming the sheet ("re-traced or changed") when the
  `finder` root's `blocks_px.geojson` sha256 is not the one in `blocks/<series>/<sheet>.json`.
- CLI: `sfzmaps find trial compare --trial T` does the following:
  - It runs `check_blocks` for every sheet first.
  - It writes `compare.json`. For each sheet it holds `numbers` for today, `traced` and `finder`; `reach_ft` for
    each, from the enabled GCPs' `wx, wy`; and, for the control, each arm's share of control points on trained
    ground (`trained_ground`'s union).
  - It writes `look/<series>/<sheet>__<place>.png` for the three places, with `traced` before and `finder` after.
  - It prints the table.

- [ ] **Step 1: Write the failing tests.**
  ```python
  def test_reach_is_the_distance_to_the_nearest_control_point():   # points x=0..100 (y=0), one GCP at (0,0) -> p50 50, p90 90
  def test_compare_refuses_a_retraced_blocks_file(tmp_path):       # record a sha, rewrite the file -> ValueError naming s/a
  def test_places_takes_the_box_with_most_points_off_the_print(): # synthetic distances: 30 points > 1 px in box B, 10 in A -> worst == B
  def test_placed_picture_is_at_the_scans_own_resolution(tmp_path):
      # two 300x200 GeoTIFFs (EPSG:2227, 2.7 ft/px) and one line across: picture width == 2 * window px + GAP,
      # the line's px are the line colour in both panels
  ```
- [ ] **Step 2:** Run (FAIL), implement, run (PASS, plus `tests/test_build.py`). Commit:
  `find: the trial comparison: control-point reach, the three places and the warped-scan pictures`.

### Task 26: Step 3: the trial (run, no code)

- [ ] **Step 1:** With `F=$PWD/data/find/trial/all-pieces/finder`, run the four sheets in parallel:
  `SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year <series> --sheet <sheet>`. Never pass
  `--force`. Passing looks like this: blocks "skipped (fresh)"; rough, refine, changes, score and warp "done";
  classify and labels "skipped". Then run
  `SFZMAPS_DATA_DIR=$F SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps warp 1921 2000-digital-zoning`.
- [ ] **Step 2:** `uv run sfzmaps find trial compare --trial all-pieces`. Passing: no sheet refused, and 12 pictures
  written. If a sheet's warp came out `uncorrected`, its `warped_paper.tif` is the copied old one. Say so; the
  pictures use `warped.tif` only.
- [ ] **Step 3: Look first.** Open all 12 pictures before the table. For each sheet and place, write down whether
  the scan sits on today's block lines better, the same, or worse, and where.
- [ ] **Step 4:** Write `docs/research/finder-to-placement-trial-<date>.md` with the following, and no conclusion
  from a number without its picture:
  - per sheet, today / traced / finder: blocks, good corners, control points, reach p50 and p90 in feet, held-out p50
    and p95, the warp's choice and within-1;
  - for the control, its trained-ground share and control points on it;
  - Section 12's joined areas;
  - what the pictures show;
  - what is worse, looked into, with the evidence.

  Commit the note.
- [ ] **Owner gate C.** Send the table and, for each sheet, the `worst` place's picture (more on request), each
  captioned as a result with what it is for. Say what is worse and why, as far as is known. Wait for a plain go.

### Task 27: Step 4: the hand-off note (after Gate C)

- [ ] **Step 1:** Re-read what the placement branch reads today, on the `placement-one-process` branch: `place/start.py` `_ink` and
  `place/inputs.py:159-174`, `place/finefit.py` `PlacedSheet`, and `sheetwarp/targets.py` `fit_targets`,
  `ink_centres`, `block_line_weights`. Cite them by function and check every line number against the code (audit
  H).
- [ ] **Step 2:** Write `docs/research/finder-to-placement-handoff-<date>.md` covering the following:
  - what `place start` and `place fine` read from the scan;
  - where the finder's block-edge answer would take the place of printed lines: an `_Ink` whose `x, y` is the thinned
    centre line of the edge chance, `w` the edge chance, `phi` its direction, and `keep` the frame cut; and an
    `InkCentres` for `direction_maps` with the edge chance in place of `thin_share`;
  - the function that reads a sheet's stored answers (`find.read.load(wholesheet.paths(run_folder, sid)[0])`, giving
    float32 (2, h, w) in the page's own px) and how it is made (`sfzmaps find read --run all-pieces --series S`);
  - the per-block confidence fields in the blocks file;
  - what has to be on one branch: `sfzmaps.find`, `inputs/find*`, `inputs/<series>/duplicate_scans.csv` (from commit
    243af8c, absent on the placement branch), and the run folder `data/find/runs/all-pieces/` (`model.pt`,
    `log.json`), which is not in git;
  - the trial's result in three lines.
- [ ] **Step 3:** Commit the note. No change is made on the placement branch.

---

## Decisions the plan made

1. **New code lives in `sfzmaps/find/`, not a new package.** `build.CODE_EXCLUDE` leaves `find` out of the build
   stages' dependencies, and a new top-level package would become a dependency of every stage. Cost if wrong: `find`'s
   "no LLM output is read" rule gets a written exception for the placement side. `trial`/`placeblocks` read placement
   outputs to compare and never open `labels.json`.
2. **The tracing step's per-piece outline loop is mirrored in `placeblocks.outlines` (about 30 lines), calling
   `linework`'s own helpers and constants. `linework.py` is not refactored.** A change there re-traces every sheet,
   stales every edition and warp stage, and in the trial would overwrite the finder's blocks. Cost: a later change to
   the loop in `linework` will not reach the mirror; a comment names the lines.
3. **The seam is used without a code change.** This is audit F's "no-code alternative": the finder's file goes into a
   copied work folder whose blocks sidecar says fresh. `runner.do_blocks` stays as it is. The quirk is guarded by the
   sha check (Review Focus 2). Cost: if the guard fires, the arm must be prepared again.
4. **Trial layout.** Two full data roots (`traced`, `finder`) under `data/find/trial/all-pieces/`, because the warp
   takes a whole series and places by folder. Inputs are copied; everything after the blocks is left out so both
   arms compute it. Cost: `cache/` is shared through its link. Its only write path seen, the lot-lives cache, is
   keyed by the parcels' sha and is a hit.
5. **Step 0 tests exact equality with today's files.** There is no tolerance. A difference gets its reason found
   (library versions and device first). Cost: a harmless rounding difference costs a look.
6. **Sheets with a review or a seed are refused in trial roots** (audit E.3). None of the four has one.
7. **The control is `2000-digital-zoning-p08`.** It has work and warp files and an automatic placement: 292 blocks,
   556 good corners, 492 control points, passed (5.01 / 12.28 ft), probabilistic warp, within1 0.627. It has no hand
   crop. But 11.1% of its ground lies under lot-labelled training sheets' footprints (p01, height-p07, su-p10), so
   that share is reported apart. Cost if the owner wants none: pick another sheet and redo Task 21's root.
8. **The outline's place is measured on 1921 `094149543` and 2000 `p07`.** Tracing works on both and neither is a
   trial sheet. "Outer edge or centre" is the nearer of the design's two answers. The inset is each sheet's own half
   stroke (`linework.stroke_width`), as tracing does. Cost: three more whole-sheet reads, and p07 is training ground,
   where the read is better than on unseen ground.
9. **The per-block confidence is four numbers in each feature's `properties`.** These are the rim px count, the
   street chance outside the outline (p10, p50) and the edge chance on it (p50). Refine and the warp ignore
   `properties`. The corners list stays N x 2. Cost: per-corner confidence is not kept; it can be read back from a
   corner's block.
10. **Joined areas are counted with the year's lots placed by today's placement**, for reporting only. Cost: on
    1921, blocks renumbered or reshaped since then can miscount.
11. **The comparison's points are `targets.block_side_points` under today's placement**, the same as the warp's
    within-1 score. Those are the year's lots' block sides, not DataSF's current blocks. Cost: the design said
    "today's block sides"; on 1921 the two differ where lots have changed.
12. **Picture places are `worst` (most side points off the print under today's corrected placement, by the warp's
    own measure), `busiest` and `centre`**, with 1200-px windows. "Before" is the `traced` arm, not today's files, so
    both sides come from the same rerun. Cost: the owner may want named places, such as Visitacion or Sunnydale on
    Section 12.
13. **Names:** trial `all-pieces` (the run's name), commands `find blocks`, `find outline-offset`,
    `find trial {root,numbers,compare}`.
