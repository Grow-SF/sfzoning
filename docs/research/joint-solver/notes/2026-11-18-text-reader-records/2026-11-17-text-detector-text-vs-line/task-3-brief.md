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

