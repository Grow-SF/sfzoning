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

