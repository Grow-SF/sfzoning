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

