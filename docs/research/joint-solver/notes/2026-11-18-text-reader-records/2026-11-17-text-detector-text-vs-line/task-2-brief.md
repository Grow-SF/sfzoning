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

