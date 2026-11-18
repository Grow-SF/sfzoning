# Text Detector: Letters, Not Line Ink. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The text detector calls printed characters text and line work not text, including where a line crosses a
label. This is shown on independent measures against a control with measured noise. The candidate is then previewed
downstream for the owner, who decides adoption.

**Architecture:**
- Measures come first: `solver/textnet/linerules.py`, `solver/textnet/linegate.py` and `scripts/textlines/`. They
  are validated on exact-truth cases and scored on today's detectors.
- Then two training changes are measured on one held-out quarter:
  - **control 0, a label rule:** zones only around unmatched label points, and line ink inside word boxes unknown;
  - **variant 1:** control 0 plus composites built inside real crops from real clean words and real line pieces.
- A decision rule written in advance picks between them. The winner is previewed through the walls and the solve in
  an experiment folder. No stage is rebuilt.

**Tech Stack:** Python 3, numpy, OpenCV, PyTorch (mps), pytest. No DVC change in this plan.

**Spec:** `docs/superpowers/specs/2026-11-17-text-detector-text-vs-line-design.md`.
- Spec red team: `pipeline/data/solver/experiments/redteam-text-vs-line/findings.md`.
- Plan red team, folded in here: `.superpowers/sdd/redteam-text-vs-line-plan.md`.
- Root cause: `docs/research/joint-solver/notes/2026-11-16-text-target-root-cause.md`.

## Global Constraints

- **T3 first, and no commits during builds.**
  - T3 of the text-reader plan builds stages in this same worktree. A commit, or a dirty or untracked file under
    `pipeline/src`, `inputs`, `dvc.yaml`, `pyproject.toml` or `uv.lock`, makes a running `sfzmaps build` refuse to
    record (`src/sfzmaps/build.py:509,624,665,706-711`).
  - Before creating or editing any such file, and before any commit: `pgrep -fl 'sfzmaps (build|stage)'` must be
    empty, and the T3 ledger (`.superpowers/sdd/2026-11-16-text-reader-labels-and-publish/progress.md`) must show
    T3's builds recorded.
- **No stage's import closure changes.**
  - `textnet/data.py` and `train.py` gain no new imports. `linerules.py` imports nothing from `sfzmaps.solver` but
    `grid`. The label rule's data reach `crop_targets` through `SheetData` fields filled before training. Composites
    reach `draw` through an object passed to `fit`.
  - Check after every task: `uv run pytest -q tests/test_build.py`.
  - Don't edit `build.py`, `dvc.yaml`, `dvc.lock`, `pipeline/README.md`, any file directly in `src/sfzmaps/` (e.g.
    `cli.py`), `pyproject.toml` or `uv.lock`.
- **Staleness is expected.** After Task 4, six stages (`solver-text`, `solver-text-scores`, `solver-text-reader`,
  `solver-boxes`, `solver-text-crops`, `solver-text-read`) show stale in `dvc status` because their code changed. The
  default recipe is unchanged. Nobody builds them in this plan; the morning report lists them.
- **$0.** No OpenAI/LLM calls. Never print, log, copy or commit `pipeline/.env`, `site/.env.local` or any token.
- **One GPU (mps) process on the machine at a time:** every training and prediction runs inside
  `gpu.gpu_turn(True, device)`. The pytest session holds a GPU turn too (`conftest.py:44-60`), so don't run the
  suite while a variant trains.
- **Load near 20:** ≤ 4 worker processes × 1 thread.
- **Determinism:** seeded. The batch thread's rng draws every crop and composite; `Bank.build` uses its own rng
  `np.random.default_rng([seed, 1])`.
- **Held out by quarter** (`data.Quarter`). The owner's answered blocks (`reader_stage.held_blocks()` /
  `load_block_area`) are never trained on, never used for composites, and never in a bank.
- **No LLM output at inference.** Terra's and Sol's saved answers only train and score.
- **Measures:**
  - never define line ink by Terra boxes or a training labeller;
  - nothing is excluded near labels;
  - every line rule prints its exact-truth accuracy beside its numbers;
  - no verdict from wrong-lot counts or one seed.
- **Apache-2.0 code;** no GPL-family dependency.
- **Tests:**
  - CI has no `pipeline/data`. Tests use the synthetic `maps` fixture (`tests/test_solver_textnet.py:244`) or arrays
    built in the test.
  - Real data are injected (`geo=`, `sheets=`, `areas=`, `score=` arguments). A check needing real files skips when
    they are absent, as `tests/test_solver_textread.py:152` does.
  - Glyphs in unit tests are drawn with `cv2.putText`.
- **Test command,** while iterating:
  `uv run pytest -q tests/test_solver_textnet.py tests/test_codereader.py tests/test_solver_textread.py
  tests/test_build.py tests/test_textnet_*.py tests/test_textlines_*.py`. Before each commit, `uv run pytest -q`
  (no variant training at the time).
- **Data:** outputs under `pipeline/data/solver/experiments/textlines/`, never committed. No bare `git stash`, no
  `dvc gc`.
- **Git:** `git add <paths>` and `git commit` as separate plain commands. Every commit message ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A chance file missing** for a sheet, or covering part of it: the gate stops with the sheet's name. It never
   scores fewer px (Task 2 test).
2. **A Sol point whose Terra word sits on another sheet** (seams): matching is per sheet (Task 4 test).
3. **A turned word box running off the page:** the in-box rule clips it as `_paint` does (Task 4 test).
4. **A clean word that fits no place in its crop:** the next candidate is drawn, from the same rng (Task 6 test).
5. **A new recipe field leaking into today's runs:** `labels="boxes"` and `composites=0.0` give byte-identical
   batches and the same run name (Tasks 4 and 6 tests).

---

### Task 1: Line rules, checked on exact-truth cases

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textnet/linerules.py`
- Test: `pipeline/tests/test_textnet_linerules.py`

**Interfaces:**
- Consumes: nothing from `sfzmaps.solver` but `grid`. `text-crops/font.npz` is read only by `rule_accuracy`'s
  report, when present.
- Produces:
  - `run_lines(ink: np.ndarray, length: int) -> np.ndarray`: a copy of `codereader.labels.run_lines` (with its
    `_line_kernel` and `RUN_ANGLES`); `codereader` is not edited;
  - `run_px(ft_per_px: float) -> int`: a copy of `codereader.labels.run_px`;
  - `line_px(ink: np.ndarray, ft_per_px: float) -> np.ndarray`: bool, the run rule only (`run_lines` at
    `run_px(ft_per_px)`);
  - `chain_lines(ink: np.ndarray, length: int, run: int) -> np.ndarray`: bool, ink on chains of straight runs of at
    least `run` px joined where their ends meet within 2 px, with the chain's end-to-end extent ≥ `length`. It is
    built and measured, and NOT part of `line_px`;
  - `TRUTH_CASES: tuple[str, ...]`:
    - "line through the middle", "line along the baseline", "block corner", "two street lines, word between",
      "vertical line through the 2nd letter";
    - "stair-step border", "dashed line", "hatching";
  - `truth_case(name: str, text: str, cap_px: int, seed: int) -> dict`: keys `ink`, `glyph`, `line` (bool);
  - `rule_accuracy(rule, texts=("RM-2","RH-1(D)","NC-3","C-3-O","RM-4","P","NC-1","M-1","RTO-M"), caps=(20, 27),
    seeds=range(4)) -> dict`: per case and per text, `line_recall` and `glyph_false_line`. Glyph px within 1 px of
    the line are not counted.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_copy_matches_codereaders_run_rule():
    rng = np.random.default_rng(0)
    ink = rng.random((120, 160)) < 0.2
    ink[60, :] = True
    assert (linerules.run_lines(ink, 40) == codereader_labels.run_lines(ink, 40)).all()

def test_a_line_through_a_word_is_line_and_the_letters_are_not():
    c = linerules.truth_case("line through the middle", "RM-2", cap_px=27, seed=0)
    found = linerules.line_px(c["ink"], ft_per_px=3.98)
    line = c["line"] & ~c["glyph"]
    assert (found & line).sum() / line.sum() >= 0.95
    far = c["glyph"] & ~dilate(c["line"], 1)
    assert (found & far).sum() / far.sum() <= 0.02

def test_rule_accuracy_reports_every_case_and_text():
    acc = linerules.rule_accuracy(linerules.line_px, texts=("RM-2", "P"), caps=(27,), seeds=range(1))
    assert set(acc) == set(linerules.TRUTH_CASES)
    assert all({"line_recall", "glyph_false_line"} <= set(v["RM-2"]) for v in acc.values())
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd pipeline && uv run pytest tests/test_textnet_linerules.py
  -v`. Expected: FAIL (no module).

- [ ] **Step 3: Implement `linerules.py`**
- The cases are ported from `scripts/spikes/redteam_text_vs_line/redteam_measure.py` (`synth`).
- Glyphs are drawn with `cv2.putText` (Hershey simplex, thickness scaled to cap/8), and lines with `cv2.line` at 4
  and 6 px.
- "stair-step border" is lot jogs of 20-40 px; "dashed line" is 12 on, 8 off; "hatching" is 45° lines 10 px apart.

- [ ] **Step 4: Run the tests to verify they pass, and report.** Expected: PASS.
  - Write `rule_accuracy(line_px)` and `rule_accuracy(lambda ink, f: chain_lines(ink, run_px(f), q99_cap_px))` to
    `experiments/textlines/rules.json`, with the learned font as well when `font.npz` is present.
  - Put both tables in the report. `chain_lines` is adopted nowhere unless its glyph false-line is ≤ 2% on every
    text.

- [ ] **Step 5: Commit** (after the T3 check in Global Constraints)

```bash
git add pipeline/src/sfzmaps/solver/textnet/linerules.py pipeline/tests/test_textnet_linerules.py
git commit -m "textnet: line rules (the run rule; the chain rule measured, not used) checked on exact-truth cases"
```

### Task 2: The gate: M1a, M1b, M1c and M2, with per-block tallies

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textnet/linegate.py`
- Test: `pipeline/tests/test_textnet_linegate.py`

**Interfaces:**
- Consumes: `linerules.line_px`, `rule_accuracy`; `data.SheetData`, `data._world_at`, `grid.to_grid`,
  `grid.quarter_of`.
- Produces:
  - `ChanceSource = Callable[[str], np.ndarray]`: a sheet name → uint8 chance, the page's shape.
  - `folder_source(folder: Path, layout: str) -> ChanceSource`: `layout` is `"flat"` (`<sheet>.png`) or `"nested"`
    (`<sheet>/chance.png`). A missing file raises `FileNotFoundError` naming the sheet, and a wrong shape raises
    `ValueError`.
  - `Geo`, a protocol per sheet: `quarter(sd) -> uint8 page raster` (index into `paths.QUARTERS`, 255 off the map),
    `block(sd) -> int32 page raster` (−1 none), `border_weight(sd) -> float32 page raster` (M1a's w).
  - `CityGeo`: the real `Geo`.
    - Page px → map point by `data._world_at(sd.world, rows, cols)` → `grid.to_grid` → the grid px.
    - The quarter is `grid.quarter_of`.
    - The block is `C.grown_raster` → the lot → its `block_num` (`lots/table.parquet`).
    - The weight is w(d) at the grid px's distance d to the nearest district crack. A district crack is a
      `border_paths()` point with `C.known[a] & C.known[b] & (C.table[a] != C.table[b])`, as in
      `measures.Table.line`.
    - w(d) = max(0, f_D(d) − f_L(d)) / f_D(d) for |d| ≤ `J`, else 0. Here f_D and f_L are `prof`'s ink shares at
      offset ±d over district cracks and over the other both-known cracks, computed separately for `touch` True and
      False, using the class of the nearest district crack.
    - It is computed in bands of `data.BAND` rows, at ink px only.
  - `gate_sheets(sheets, source, quarters: tuple[str, ...], geo: Geo, boot: int = 1000, seed: int = 0) -> dict`:
    - keys:
      - `m1a`, `m1a_in_box`, `m1a_near`, `m1a_far` (M1a: px weighted by w);
      - `m1b_in_box`, `m1b_near`, `m1b_far`, `m1b_by_point` (`{"0-30", "30-100", "100+"}`, distance to the nearest
        Sol point);
      - `m1c` (all ink outside every Terra box);
      - `m2`, `m2_by_kind`, `m2_answered`, `overlap_band`;
      - `rules`.
    - Each measure gives `{"px", "called_05", "called_09", "mean", "log_loss", "ci_05": [lo, hi]}`.
    - The splits:
      - "in box" = inside a Terra word rectangle; "near" = within 30 px of one; "far" = farther;
      - M2's base = in-box ink not `line_px` and not within 1 px of it;
      - overlap band = in-box ink that is `line_px` or within 1 px of it.
  - `gate(source, quarters, geo=None, boot=1000, seed=0) -> dict`: loads the 2000 sheets (placement, words, points)
    and calls `gate_sheets` with `CityGeo()` by default.
  - `write(result: dict, tallies: dict, out: Path)`: writes `scores.json`, `scores.md` and `blocks.npz`. Per measure,
    `blocks.npz` holds the block ids and, per block, px, called_05, called_09, the chance sum and the log-loss sum
    (float64). `gate_sheets` returns `(result, tallies)`.
  - `paired(a: list[Path], b: list[Path], measure: str, boot=1000, seed=0) -> dict`:
    - it reads the runs' `blocks.npz` files;
    - each draw resamples the blocks once, the same draw for both arms;
    - each arm's ratio is the mean over its runs;
    - it returns `{"diff", "lo", "hi"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_line_ink_crossing_a_word_counts_against_a_detector_that_calls_it_text(crossed_word):
    # crossed_word: SheetData with one word box crossed by a 4-px line (masks known), a FakeGeo with one quarter "ne"
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    assert r["m1b_in_box"]["called_05"] == 1.0
    assert r["m2"]["px"] == crossed_word.glyph_far_from_line_px     # counted from the fixture's own masks
    assert r["m2"]["called_05"] == 1.0
    assert r["overlap_band"]["px"] > 0

def test_border_ink_inside_a_box_is_split_out(crossed_word):
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    assert r["m1a_in_box"]["px"] > 0 and r["m1a"]["px"] >= r["m1a_in_box"]["px"]

def test_a_missing_chance_file_stops_the_gate(tmp_path, crossed_word):
    with pytest.raises(FileNotFoundError, match=crossed_word.sd.sheet):
        linegate.gate_sheets([crossed_word.sd], linegate.folder_source(tmp_path, "flat"), ("ne",), crossed_word.geo)

def test_paired_difference_of_identical_runs_is_zero(tmp_path, crossed_word):
    r, t = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    linegate.write(r, t, tmp_path / "a"); linegate.write(r, t, tmp_path / "b")
    p = linegate.paired([tmp_path / "a"], [tmp_path / "b"], "m1b_in_box", boot=50)
    assert p["diff"] == 0 and p["lo"] == 0 and p["hi"] == 0

@pytest.mark.skipif(not REAL_DATA, reason="needs pipeline/data")
def test_page_to_grid_round_trip_within_one_px():
    ...  # 10,000 random ink px of one 2000 sheet: page -> grid (CityGeo's path) -> page (page.sheet_px / to_pixel) within 1 px
```

- [ ] **Step 2: Run them to verify they fail.** Expected: FAIL.

- [ ] **Step 3: Implement `linegate.py`** to the interfaces. Print the w(d) table into `scores.md`.

- [ ] **Step 4: Run the tests to verify they pass.** Expected: PASS (the real-data test passes locally, and is
  skipped in CI).

- [ ] **Step 5: Commit** (after the T3 check)

```bash
git add pipeline/src/sfzmaps/solver/textnet/linegate.py pipeline/tests/test_textnet_linegate.py
git commit -m "textnet: the text-vs-line gate (M1a district-border ink weighted by the measured profile, split by box; M1b run-rule ink; M1c; M2 on a clean base) with per-block tallies and paired intervals"
```

### Task 3: Baselines, the picture panel and the missed-text audit

**Files:**
- Create: `pipeline/scripts/textlines/baselines.py`, `panel.py`, `audit.py`
- Test: `pipeline/tests/test_textlines_panel.py`

**Interfaces:**
- Consumes: `linegate.gate`, `folder_source`, `write`.
- Produces:
  - `panel.WINDOWS: dict[str, tuple[str, int, int, int]]`: name → (sheet, row, col, size).
    - `rm2a`, `rm2b`, `rm2c`: the RM-2 zoom centres (1850, 2505), (1745, 2175) and (1868, 2218) on p02, size 240.
    - `downtown`, `residential`, `dashed`, `hatched`: NE, chosen once by looking.
    - `sw_dense`, `sw_street`: SW.
  - `panel.render(sources: dict[str, ChanceSource], out: Path, title: str) -> Path`:
    - a row per window, a column per source;
    - red where the chance is ≥ 0.5 on ink, grey for other ink;
    - a title line saying what the picture is for.
  - `audit.sample(source, quarters, n=64, seed=0) -> list[dict]` and `audit.sheet(items, out) -> Path`.
    "Letter-sized" is `scripts/spikes/redteam_text_vs_line/redteam_missed.py`'s definition.

- [ ] **Step 1: Write the failing test**

```python
def test_a_source_of_zeros_draws_no_red(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "WINDOWS", {"w": ("s", 0, 0, 64)})
    monkeypatch.setattr(panel, "page_ink", lambda sheet: np.ones((64, 64), bool))
    out = panel.render({"zeros": lambda s: np.zeros((64, 64), np.uint8)}, tmp_path / "p.png", "t")
    img = cv2.imread(str(out))
    assert not ((img[..., 2] > 200) & (img[..., 1] < 80) & (img[..., 0] < 80)).any()
```

- [ ] **Step 2: Run it to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement.** `baselines.py` does the following.
1. First, copy today's outputs into `experiments/textlines/baselines/`:
   - `data/solver/text-reader/*/chance.png` and `held.json` → `r328/`;
   - `data/solver/text/chance/` → `all-sheets/`.
2. Gate four sources, over all four quarters:
   - `r328`, nested;
   - `all-sheets`, flat (in-sample: a reference only);
   - the recipe trial: `data/solver/experiments/detector-overcall/chance/trial/<sheet>.png`, flat;
   - the old r30 quarter models: `data/solver/text-<q>/chance/`, flat, each scored on its own quarter only.
3. Write `baselines/<source>/{scores.json, scores.md, blocks.npz}` and `baselines/summary.md`.
4. Render the panel for the four sources, and an audit sheet for r328 and for the old quarter models.

- [ ] **Step 4: Run the test, then the baselines.**
  - `uv run pytest tests/test_textlines_panel.py -v`: PASS.
  - `OMP_NUM_THREADS=1 uv run python scripts/textlines/baselines.py`.
  - Expected: `summary.md`. The all-sheets detector's `m1b_in_box.called_05` should be near the red team's ~0.82;
    if it is far off, say why.
  - Look at the panel and the audits: say what they show, and count the text pieces in each audit.

- [ ] **Step 5: Commit** (after the T3 check): `git add pipeline/scripts/textlines/baselines.py
  pipeline/scripts/textlines/panel.py pipeline/scripts/textlines/audit.py pipeline/tests/test_textlines_panel.py`,
  then commit "textlines: baselines of today's detectors on the gate, the picture panel, the missed-text audit".

### Task 4: The control's label rule

**Files:**
- Modify:
  - `pipeline/src/sfzmaps/solver/textnet/data.py`: `SheetData.line_bits`, and `crop_targets(..., labels=)`;
  - `pipeline/src/sfzmaps/solver/textnet/train.py`: `Recipe.labels`, `CHOICES`, and `draw` passing `recipe.labels`.
  - Neither file gains an import.
- Create: `pipeline/src/sfzmaps/solver/textnet/zones.py`
- Modify (on purpose): `tests/test_solver_textnet.py`
  - `test_default_recipe_trains_as_today` gets the new default fields;
  - `test_sheet_data_holds_no_full_page_target_rasters` allows the empty `line_bits`.
- Test: `pipeline/tests/test_textnet_labels_rule.py`

**Interfaces:**
- Consumes: `linerules.line_px`.
- Produces:
  - **`Recipe.labels: str = "boxes"`**, with `CHOICES["labels"] = ("boxes", "lines-unknown")`. The run name is
    unchanged at the default (as `NOT_IN_NAME` / default handling does).
  - **`SheetData.line_bits: np.ndarray`**, default `np.zeros((0, 0), np.uint8)`: the sheet's `line_px` grown by 1 px
    (8-neighbours), packed like `ink_bits`.
  - **Under `labels="lines-unknown"`:** `crop_targets` sets `known` False on word px that are set in `line_bits`, and
    raises `ValueError` when `line_bits` is empty.
  - **`zones.matched_points(sd) -> np.ndarray`:** bool per Sol point, True when a Terra word of the same sheet holds it
    or lies within half that word's height of it.
  - **`zones.zone_radius(sheets, areas) -> dict`:** `{"rule": "q99/2", "px", "q50", "q90", "q99", "words"}`, from the
    code words' lengths outside `areas`.
  - **`zones.prepare(sheets, areas, cache: Path | None) -> list[SheetData]`:**
    - for each sheet, `replace(sd, points=sd.points[~matched_points(sd)], point_radius=zone_radius(...)["px"],
      line_bits=...)`;
    - lines are cached per sheet in `experiments/textlines/lines/<series>/<sheet>.npz`, keyed by the hash of
      `linerules.py`;
    - it is called before `gpu.gpu_turn`. `_unknown` is unchanged.
  - **`zones.label_shares(sheets) -> dict`:** per series × kind, the share of in-box ink made unknown.

- [ ] **Step 1: Write the failing tests** on the synthetic `maps` fixture and arrays built in the test.

```python
def test_line_ink_inside_a_word_is_unknown_and_the_letters_stay_text(crossed_word_sd):
    sd = zones.prepare([crossed_word_sd], (), cache=None)[0]
    t = data.crop_targets(sd, *CENTRE, 128, 1.0, 0, labels="lines-unknown")
    assert not t["known"][LINE_IN_BOX].any()
    assert t["text"][GLYPH_FAR].all() and t["known"][GLYPH_FAR].all()

def test_lines_unknown_without_lines_is_an_error(crossed_word_sd):
    with pytest.raises(ValueError):
        data.crop_targets(crossed_word_sd, *CENTRE, 128, 1.0, 0, labels="lines-unknown")

def test_a_point_with_its_own_word_gets_no_zone_and_an_unmatched_one_does(two_points_sd):
    assert zones.matched_points(two_points_sd).tolist() == [True, False]

def test_a_word_on_another_sheet_does_not_match_a_point(seam_sds):
    assert not zones.matched_points(seam_sds[0]).any()

def test_a_turned_word_running_off_the_page_is_clipped(edge_word_sd):
    sd = zones.prepare([edge_word_sd], (), cache=None)[0]
    data.crop_targets(sd, 5, 5, 128, 1.0, 1, labels="lines-unknown")

def test_the_boxes_rule_and_its_run_name_are_unchanged(maps):
    assert train.Recipe().name == RECIPE_NAME_BEFORE      # pinned from today's code before the change
```

The existing guards `test_crop_targets_equal_full_page_drawing` and the GUARD weights digest must pass unchanged.

- [ ] **Step 2: Run them to verify they fail.** Expected: FAIL.

- [ ] **Step 3: Implement** to the interfaces. Then, on real data, write `zones.label_shares` for all 87 training
  sheets and `zone_radius` to `experiments/textlines/labels_rule.json`.

- [ ] **Step 4: Run the test command, then `uv run pytest -q`.** Expected: PASS, with `tests/test_build.py` passing
  (no closure change).

- [ ] **Step 5: Commit** (after the T3 check): `git add` the four source and test files and the modified
  `tests/test_solver_textnet.py`, then commit "textnet: the lines-unknown label rule (zones only around unmatched label
  points at the measured radius; line ink in word boxes unknown)".

### Task 5: One held-out quarter, trained and gated: control 0, value first

**Files:**
- Create: `pipeline/scripts/textlines/variant.py`
- Test: `pipeline/tests/test_textlines_variant.py`

**Interfaces:**
- Consumes: `train.fit`, `train.save_model`, `predict.load_model`, `data.hold_out`, `reader_stage.held_blocks` /
  `load_block_area`, `zones.prepare`, `linegate.gate` / `write`, `panel.render`, `audit.sample`.
- Produces:
  - `run(name: str, recipe: train.Recipe, quarter: str, seed: int, device=None, sheets=None, areas=None,
    score=None, bank=None, extra_series=()) -> Path`:
    - it trains with `quarter` and the answered blocks held out;
    - it predicts chance on the 2000 sheets for that quarter's px (0 elsewhere), and on `extra_series` sheets' px of
      that quarter, if any;
    - it gates the result and renders the panel and the audit;
    - it writes `experiments/textlines/runs/<name>-<quarter>-s<seed>/{model.pt, chance/, scores.*, blocks.npz,
      panel.png, audit.png, run.json}`. `run.json` holds the recipe, seed, device, training minutes, git head and
      `git status --porcelain -- src` at the start.
  - CLI: `uv run python scripts/textlines/variant.py <name> --quarter ne --seed 0 [--labels lines-unknown]
    [--composites P] [--extra 1990-1996-zoning]`.

- [ ] **Step 1: Write the failing test** on the synthetic `maps` fixture, with `score=` replaced by a stub and a
  2-step CPU recipe: the folder is `t-ne-s0` and holds `model.pt`, `scores.json` and `run.json`.

- [ ] **Step 2: Run it to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement.** The base recipe is `stage.DEFAULT_RECIPE` with `labels` (and `composites`) from the CLI.
  `zones.prepare` and the bank run before the turn; training and prediction run inside one `gpu.gpu_turn`.

- [ ] **Step 4: Run the test, then the control, value first, one run at a time.**
1. `variant.py control --quarter ne --seed 0 --labels lines-unknown --extra 1990-1996-zoning`.
   - Gate it against `baselines/r328` and the old NE quarter model with `linegate.paired`.
   - Look at the panel.
   - **If M1a-in-box and M1b-in-box are not lower than both baselines** (paired interval excluding zero), stop the
     plan's training here and write up why.
2. `variant.py control --quarter ne --seed 1 ...`, then `--seed 2`.
   - The noise is each measure's largest pairwise difference over the three seeds.
   - Report the three runs, the noise, and the 1990-96 NE numbers (M1b, M1c, M2; there is no M1a without a table).

- [ ] **Step 5: Commit** (after the T3 check; not during a run): `git add pipeline/scripts/textlines/variant.py
  pipeline/tests/test_textlines_variant.py`, then commit "textlines: train and gate one held-out quarter; control 0 on
  NE".

### Task 6: Composites inside real crops

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textnet/composite.py`
- Modify: `pipeline/src/sfzmaps/solver/textnet/train.py`: `Recipe.composites: float = 0.0`; `fit(..., bank=None)`;
  `draw(..., bank=None)`. No new import: `bank` is any object with `.crop(rng, sd, y, x, size, scale, turns)`.
- Test: `pipeline/tests/test_textnet_composite.py`

**Interfaces:**
- Consumes: `linerules.line_px` (via `SheetData.line_bits`), `data.crop_targets`, `zones.prepare`.
- Produces:
  - **`measure(sheets) -> dict`,** written to `experiments/textlines/distributions.json` from the training quarters
    of the 2000 sheets:
    - `cross_by_kind`: the share of words of each kind with ≥ 3 `line_bits` px inside the box;
    - `line_texture`: the SD and lag-1/2/4 autocorrelation of each edge's offset from a straight fit, on run-rule line
      pieces ≥ 100 px long, by width class.
  - **`Bank.build(sheets, areas, seed) -> Bank`,** with its own rng `default_rng([seed, 1])`. It holds references
    only:
    - clean words, as (sheet index, rectangle, kind, angle): Terra words whose ink pieces all lie inside the box,
      with no `line_bits` within 3 px;
    - real line pieces, as (sheet index, piece box): run-rule components ≥ 100 px, outside every word.
    - None comes from held-out quarters or answered blocks.
  - **`Bank.crop(rng, sd, y, x, size, scale, turns) -> dict`:** the keys of `crop_targets`. It starts from
    `crop_targets(sd, y, x, ..., labels="lines-unknown")` and changes it one of two ways, chosen by rng:
    - **(a) word onto lines.**
      - A clean word's ink is ORed in at page scale, before the resize.
      - It is placed on the crop's `line_bits` ink with probability `cross_by_kind[kind]`, else off line ink.
      - It goes only where its turned rectangle + 3 px holds no Terra word, no other box and no unmatched zone; after
        20 failed tries the next clean word is drawn.
      - `text` is set on its ink; `kind` and `angle` on those px; `known` True.
    - **(b) line onto a word.** A crop centred on a clean word, with a real line piece ORed across the word. The
      line's px are not text and known; overlap px are text. Drawn lines are not used tonight.
  - **`texture_stats(masks) -> dict`:** isolated share, pinholes per 1,000 ink px, pieces per word.
  - **`line_texture(pieces) -> dict`:** as `measure`'s.
  - **In `draw`:**
    - with `recipe.composites > 0`, one rng draw decides per crop and composites come from `bank.crop`;
    - with `recipe.composites == 0` no extra rng draw happens, so batches are byte-identical to today's.

- [ ] **Step 1: Write the failing tests** on synthetic sheets.

```python
def test_a_word_onto_lines_keeps_exact_masks(bank_fx, rng):
    c = bank_fx.bank.crop(rng, bank_fx.sd, *bank_fx.centre, 128, 1.0, 0)
    assert (c["text"].astype(bool) <= c["ink"]).all()
    assert (c["ink"] & ~c["text"].astype(bool) & c["known"]).any()

def test_composites_zero_leaves_batches_unchanged(maps):
    assert batches_equal(draw_n(maps, train.Recipe(composites=0.0), seed=3), PINNED_BATCHES_SEED3)

def test_composites_repeat_from_the_seed(bank_fx):
    r = train.Recipe(composites=0.5)
    assert batches_equal(draw_n(bank_fx.sheets, r, 7, bank_fx.bank), draw_n(bank_fx.sheets, r, 7, bank_fx.bank))

def test_composite_word_ink_is_real_word_ink(bank_fx, rng):
    c = bank_fx.bank.crop(rng, bank_fx.sd, *bank_fx.centre, 128, 1.0, 0)
    t = composite.texture_stats([c["ink"] & c["text"].astype(bool)])
    assert t["pinholes_per_1000"] <= 3 * max(bank_fx.real_texture["pinholes_per_1000"], 1)

def test_a_word_that_fits_nowhere_takes_the_next_candidate(bank_with_one_huge_word, rng):
    bank_with_one_huge_word.bank.crop(rng, bank_with_one_huge_word.sd, 32, 32, 64, 1.0, 0)
```

- [ ] **Step 2: Run them to verify they fail.** Expected: FAIL.

- [ ] **Step 3: Implement.**
  - First write `distributions.json` from real data.
  - Then, on real data, time 100 batches of `draw` at composites P and at 0. The GPU step takes about 95 ms a batch;
    say if the draw is slower.
  - Save 24 composites to `experiments/textlines/composites.png` and look at them.
  - Overlap px get weight 1: there is no measured weight, so none is invented.

- [ ] **Step 4: Run the test command, then `uv run pytest -q`.** Expected: PASS, including `test_build.py`.

- [ ] **Step 5: Commit** (after the T3 check): `git add` the composite module, `train.py` and the test, then commit
  "textnet: composites inside real crops (real clean words onto real lines, real line pieces across clean words) at the
  measured crossing rate".

### Task 7: Variant 1, the decision as written, and SW

**Files:**
- Create: `pipeline/scripts/textlines/decide.py`, `docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md`
- Test: `pipeline/tests/test_textlines_decide.py`

**Interfaces:**
- Consumes: `linegate.paired`, the run folders.
- Produces: `decide.verdict(control: list[Path], variant: list[Path], baselines: dict[str, Path], eye_ok: bool |
  None) -> dict`. It returns one of the verdicts below, with every condition's numbers.

**The rule, written before the runs**
- **Adoptable,** for the control or the variant, against `baselines/r328` and the old quarter model, on NE and on SW:
  - M1a-in-box and M1b-in-box are lower, with the paired interval excluding zero;
  - M2 is not lower by more than the noise;
  - M1c is not higher by more than the noise;
  - the audit's count of text pieces called text is not lower beyond its binomial interval.

  The all-sheets detector is a reference only (in-sample).
- **Noise** = max(the control's largest pairwise seed difference, the paired interval's half-width).
- **Variant over control** requires all of:
  - M1a-in-box lower, with the paired interval excluding zero. M1a carries the verdict, since M1b shares the
    training's line rule.
  - M2 not lower by more than the noise;
  - M1c not higher by more than the noise;
  - `eye_ok` True: the controller looked at both panels and wrote what they show.
- **The verdicts:**
  - `pending-eye` when `eye_ok` is None;
  - `mixed` if M1b-in-box falls and M1a-in-box does not, or on any other split;
  - `control` or `variant` otherwise.

- [ ] **Step 1: Write the failing tests** for `verdict` with fake run folders (synthetic `blocks.npz`):
  - better lines with the same text and `eye_ok` True → `variant`;
  - less text beyond noise → `mixed`;
  - `eye_ok` None → `pending-eye`;
  - M1b down with M1a flat → `mixed`.

- [ ] **Step 2: Run them to verify they fail.** Expected: FAIL.

- [ ] **Step 3: Implement, then run one at a time.**
  1. `variant.py composites --quarter ne --seed 0 --labels lines-unknown --composites P`, with P =
     `distributions.json`'s word-weighted `cross_by_kind` over all kinds;
  2. the same at `--seed 1` and `--seed 2`;
  3. look at the panels, then `decide.verdict(...)` on NE;
  4. the NE winner and the control on SW, at seed 0 each, decided again on SW.

  Write the results note:
  - the baselines and every run's tables;
  - the noise, and the verdicts, with the controller's description of each panel;
  - the audits' counts with their binomial bounds;
  - the 1990-96 numbers;
  - the paths of every picture.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** (after the T3 check): `git add pipeline/scripts/textlines/decide.py
  pipeline/tests/test_textlines_decide.py docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md`, then
  commit "textlines: variant 1 against control 0 on NE and SW, the decision as written, results note".

### Task 8: Preview downstream for the owner (no stage change)

Run only if Task 7 found an adoptable recipe. Otherwise skip, and say why in the results note.

**Files:**
- Create: `pipeline/scripts/textlines/preview.py`

**Interfaces:**
- Consumes: the adoptable recipe; `reader_stage` (its training loop, `sheet_outputs`); `walls.build_walls`
  (candidates, its own folder); the solve's entry point as `criteria_check` runs it.
- Produces: `experiments/textlines/preview/`:
  - `text-reader/`: four held-out detectors with the recipe, chance as the stage writes it;
  - `walls/`: fitted on that chance as an experiment;
  - `solve/`: today's solve inputs with those walls.
  - `before-t8/`: today's `data/solver/solve/` and `walls/`, copied first.

- [ ] **Step 1: Copy `before-t8`.**
- [ ] **Step 2: Train and predict the four detectors** (one GPU turn, about 28 min), then gate them on all four
  quarters and render the panel.
- [ ] **Step 3: Fit the walls and solve** into the preview folder.
- [ ] **Step 4: Measure M5:**
  - word boxes via `wordboxes` on the preview chance (T2's measures);
  - the walls' held-out log loss;
  - wall evidence on district borders inside word boxes;
  - the solve's six measures with `criteria_check --baseline before-t8`, plus counted mistakes.

  Wrong lots are reported, but are not the verdict. Add all of it to the results note.
- [ ] **Step 5: Write the adoption path into the note.** Adopting would take one commit: the recipe in
  `reader_stage`, with `held.json` keeping `unknown_radius` for `solver-text-crops` and adding `zone_radius`; then
  `pixi run build solver-solve solver-boxes`.
  - List what that would rebuild, from `uv run sfzmaps build --dry-run`.
  - State that T3's crops and training stay untouched.
  - State that `solver-text` (no placements by design) keeps the boxes rule until the owner decides.

  Commit the script (after the T3 check).

Stop here. Adoption, and T4 and T5 of the text-reader plan, wait for the owner.
