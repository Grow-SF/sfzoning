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

