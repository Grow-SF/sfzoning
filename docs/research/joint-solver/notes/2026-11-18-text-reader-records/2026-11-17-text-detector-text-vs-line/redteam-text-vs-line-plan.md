# Red team: plan "Text Detector: Letters, Not Line Ink" (2026-11-17)

Target: `docs/superpowers/plans/2026-11-17-text-detector-text-vs-line.md` ("the plan"), against its spec, the spec's
red team and the code at be37735. Read-only, CPU, one thread. Scratch measurement scripts (scratchpad,
`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/`):
`time_runlines.py`, `scales.py`, `chain_check.py`, `windows.py`, `prof_check.py`.

**Verdict: run as corrected, with two tasks restructured.**
- The order is sound: measures, then the control, then composites, then a written decision.
- Three things break the night as written:
  - the plan starts while T3's build runs in the same worktree;
  - Tasks 4 and 6 change the code that six DVC stages hash;
  - the chain rule calls most code letters "line".
- Task 6's line windows cannot be near labels.
- Task 8 collides with T3's chain and with an open owner question.

Fix those and the rest is detail.

---

## CRITICAL

### C1. Starting now breaks T3's running build (shared worktree)

**Problem.**
- T3's `pixi run build solver-text-train` has been running since 23:36 in this same worktree (`ps`: `sfzmaps build
  solver-text-train`; about 85 min of GPU per T3's report). It finishes around 01:00, then T3 builds
  `solver-text-read`.
- `sfzmaps build` refuses to record a stage when:
  - any file under `src`, `inputs`, `dvc.yaml`, `pyproject.toml` or `uv.lock` is dirty, untracked files included;
  - HEAD has moved.
- So Task 1 creating `linerules.py`, or any commit of this plan, during T3's build throws away T3's 85 min, or forces
  a hand `dvc commit` with identity checks. That already happened once (11de9e5).

**Evidence.**
- `src/sfzmaps/build.py:509` `git_dirty` (`git status --porcelain -- src inputs dvc.yaml …`; untracked files are
  listed: `status.showUntrackedFiles` is unset).
- `build.py:706-711` guard ("HEAD moved during the build" / "code or inputs changed during the build").
- `build.py:624`: the guard is checked before each commit, and a failure means "not recorded".
- `build.py:665`: a dirty tree refuses to start a build.

**Correction.**
- Add to Global Constraints:
  > "**T3 first.** Before creating or editing any file under `pipeline/src`, `pipeline/inputs`, `dvc.yaml`,
  > `pyproject.toml` or `uv.lock`, and before any commit, check that no `sfzmaps build` / `sfzmaps stage` process is
  > running (`pgrep -fl 'sfzmaps (build|stage)'` is empty) and that the T3 ledger
  > (`.superpowers/sdd/2026-11-16-text-reader-labels-and-publish/progress.md`) records `solver-text-read` as built and
  > recorded. Never leave such files uncommitted while either plan might start a build. No commit by anyone while a
  > build runs."
- Replace Task 8 Step 3's "Before this, check that the T3 plan's tasks have committed…" with "(see Global
  Constraints: T3 first)". The hazard starts at Task 1, not at Task 8.

### C2. Tasks 4 and 6 change six stages' code and can invalidate `dvc.yaml`

**Problem.**
- `textnet/data.py` and `train.py` are code deps of six stages:
  - `solver-text`, `solver-text-scores`, `solver-text-reader`, `solver-boxes`;
  - T3's `solver-text-crops` and `solver-text-read`.

  They are listed at `dvc.yaml:1735-1739, 1935-1940, 2338-2344, 2728, 2846-2852, 3065-3071`.
- Editing them marks all six stale. Any build that touches them then retrains the held-out detectors (about 28 min
  of GPU) and may pull in T3's crops, then T3's reader training (85 min).
- **The import closure.** A new import from `data.py` or `train.py` (`zones`, `linerules`, `composite`) changes the
  closure that `dvc.yaml` lists.
  - `sfzmaps build` then refuses every build: "pipeline/dvc.yaml is out of date" (`build.py:660`).
  - `tests/test_build.py:32` (`test_committed_dvc_yaml_is_the_generated_one`) fails in CI.
  - The plan forbids touching `dvc.yaml` until Task 8.
- **A circular import.**
  - If `linerules` imports `codereader.labels.run_lines` and `data.py` imports `linerules`, the chain is
    `textnet.data` → `linerules` → `codereader.labels` → `textnet.scores` → `textnet.data`.
  - `textnet.data` is still half-initialised at that point. `scores.py:122` evaluates the annotation
    `sd: data.SheetData` when the function is defined (there is no `from __future__ import annotations`), so the
    import fails.
  - It also bloats `solver-text`'s lean closure (`dvc.yaml:1724-1739`) with pandas, `prints`, `strokes` and the rest.

**Correction.**
- **Task 1.** Replace "Consumes: `codereader.labels.run_lines`…" with:
  > "`linerules` holds its own copy of `_line_kernel` and `run_lines` (numpy and cv2 only; it imports nothing from
  > `sfzmaps.solver` but `grid`). A test asserts `linerules.run_lines == codereader.labels.run_lines` on random ink.
  > `codereader.labels` is not edited."
- **Task 4.** Add:
  > "`data.py` imports nothing new. `crop_targets` only unpacks `sd.line_bits`, which `zones.prepare` filled (see M3).
  > `train.py` imports nothing new: the composite source is passed into `fit(..., bank=None)` and `draw(..., bank)` as
  > an object with `.crop(rng, sd, y, x, size, scale, turns)`. So no stage's import closure changes before Task 8.
  > Check with `uv run pytest tests/test_build.py -q`."
- **Global Constraints.** Add:
  > "After Task 4 commits, six stages show stale in `dvc status` until Task 8. Nobody builds them in between. The
  > morning report lists them, and says why: code changed, default recipe unchanged."
- If an import really must be added before Task 8: regenerate `dvc.yaml` in the same commit
  (`uv run sfzmaps build --write-dvc-yaml`), only after T3 is done, and never touch `dvc.lock`.

### C3. The chain rule calls most code letters "line" (F2's failure again)

**Problem.**
- Task 1's chain rule joins straight runs of at least `length // 4` (12 px on 2000) at junctions, and calls a chain of
  50 px or more "line".
- Capital letters are made of 12-27 px straight strokes meeting at corners (M, E, N, H, R, 1, 4). So whole letters
  become chains of 50 px or more.

**Evidence.**
- The learned font, letters only, no line drawn (`chain_check.py`; approximation: runs at 12 px, joined within 2 px,
  chain length = skeleton px). Share of glyph ink the chain rule calls line:

  | text | cap 27 | cap 20 |
  |---|---|---|
  | RM-2 | **72%** | 68% |
  | RH-1(D) | 79% | 80% |
  | RM-4 | **100%** | 95% |
  | NC-3 | 64% | 43% |
  | NC-1 | 79% | 58% |
  | M-1 | 75% | 0% |
  | RTO-M | 99% | 54% |
  | C-3-O | 37% | 0% |
  | P | 0% | 0% |
- The run rule alone, on the red team's exact-truth cases, calls 2-6% of glyph ink line on average (up to 17.5% for
  "line through the middle"), counting glyph px touching the line (`redteam-out/synth.json`).

**Consequence.**
- Task 1's own test (glyph false-line ≤ 5%) cannot pass with the chain rule, so an implementer will tune parameters
  until one case passes.
- If `line_ink` keeps the chain rule:
  - the control makes most code letters unknown in training;
  - M2's clean base loses most letters;
  - M1b counts correct letter calls as line called text.

**Correction.**
- Task 1, Produces:
  > "`line_ink(ink, ft_per_px)` = the run rule (`run_lines` at `run_px(ft_per_px)`). `chain_lines` is built and
  > measured, but stays out of `line_ink` tonight. The stair-stepped district lines are M1a's job (spec F1/F2)."
- Task 1, Step 1. Change the second assertion of `test_a_line_through_a_word_is_line_ink_and_the_letters_are_not` to
  count only glyph px more than 1 px from the line:

  ```python
  far = c["glyph"] & ~dilate(c["line"], 1)
  assert (found & far).sum() / far.sum() <= 0.02
  ```

  This matches M2's base, which leaves out px within 1 px of line ink.
- Add:
  > "`rule_accuracy` also reports `chain_lines`' glyph false-line rate on every truth text (RM-2, RH-1(D), NC-3,
  > C-3-O, RM-4, P, NC-1, M-1, RTO-M at caps 20 and 27). Use it anywhere only if it stays ≤ 2% on all of them. For
  > example: runs of at least the q99 cap height of code words at the sheet's scale, and a chain whose end-to-end
  > extent is at least `length`."
- Draw the unit-test glyphs with `cv2.putText`, so CI runs the tests without data. The learned font is used only in
  `rule_accuracy`'s report, when `font.npz` is present (see M2).

---

## HIGH

### H1. M1a's page mapping, "`SheetData.world` inverted by nearest grid cell", is too coarse, and the weight is undefined

**Problem.**
- **The mapping.**
  - `SheetData.world` samples the placement every 32 px (`data.py:53, 176`). Inverting it to the nearest sample is off
    by up to about 22 px.
  - The district line's ink band is about 6 px each side (measured below).
  - So M1a would score mostly the wrong px.
- **The weight.** "The likelihood that ink at its distance belongs to the border" names no formula.

**Evidence.**
- `border_paths()` builds in 14 s: 8.17 M cracks, of which 364 k touching-lot and 470 k across-street cracks are
  district cracks.
- Ink share at |offset| d along the normal (`prof_check.py`), touching lots:

  | d (px) | 0 | 2 | 3 | 4 | 6 | 8 |
  |---|---|---|---|---|---|---|
  | district | 0.894 | 0.789 | 0.616 | 0.429 | 0.229 | 0.191 |
  | other lot cracks | 0.153 | 0.152 | 0.153 | 0.157 | 0.173 | 0.172 |

- Across a street: 0.816 against 0.216 at 0, 0.397 against 0.230 at 4.
- So `prof` is usable and sharp.

**Correction.** Replace the M1a bullets of Task 2, Step 3 with:
> "**M1a.**
> - **District cracks:** `border_paths()` points whose pair has `C.known[a] & C.known[b] & (C.table[a] != C.table[b])`
>   (`measures.Table.line`, `measures.py:84-85`; `C = city()`). Not re-derived from `table.py`.
> - **Weight by distance:** w(d) = max(0, f_D(d) − f_L(d)) / f_D(d), for |d| ≤ `J` and 0 beyond.
>   - f_D and f_L are the ink shares of `prof` at offset ±d, over district cracks and over the other cracks with both
>     lots known.
>   - Computed separately for `touch` True and False; at each grid px the class of the nearest district crack is
>     used.
>   - The table of w goes into `scores.md`.
> - **Distance:** d is the distance in grid px from a grid px to the nearest district crack (a distance transform of
>   the district cracks rasterised on the grid).
> - **Page px to grid**, the direction already used for the quarter split:
>   - each ink px's map point is `data._world_at(sd.world, rows, cols)`, then `grid.to_grid`, then the grid px holding
>     it;
>   - computed in bands of `data.BAND` rows, at ink px only;
>   - the same grid px gives each px its quarter (`grid.quarter_of`), its block (`C.grown_raster` → the lot → the
>     lot's `block_num` in `lots/table.parquet`) and w(d).
> - **Mapping test:** for 10,000 random ink px of one sheet, page → grid → page through `page.sheet_px` (the merged
>   page's own grid → page map, `to_pixel`) lands within 1 px."

### H2. M1a as one number hides the improvement it is meant to show

**Problem.**
- Most district-border ink is far from any label, where today's r30 detectors already call about 0.1-0.3% text
  (`detector-overcall/results.md`). The in-box share, where about 82% is called text, is small.
- A detector that fixes RM-2 moves overall M1a by (in-box share) × (gain): a point or less, about the size of the seed
  noise.
- The rule asks M1a's interval to exclude zero, so the fix is likely judged "mixed". That is F1 again.

**Correction.**
- Task 2 Produces: replace `m1a` with `m1a_in_box`, `m1a_near`, `m1a_far`, split exactly as M1b (in a Terra word box,
  within 30 px of one, farther). This is a split, not an exclusion. `m1a` stays as the total.
- Task 7's decision rule: "M1a-in-box and M1b-in-box fall…".

### H3. The decision rule cannot be computed from what the gate writes

**Problem.**
- Task 7 needs block-bootstrap intervals of the difference between runs.
- Task 2 writes only per-run `ci_05`. Two separate intervals cannot give the paired interval.
- The by-eye condition cannot come from `decide.verdict` at all.

**Correction.**
- Task 2 Produces, add:
  > "`score_sheets` also writes `blocks.npz`: for each measure, the block ids and per block the px, `called_05`,
  > `called_09`, sum of chance and sum of log loss (integers and float64), so runs can be compared block by block."
- Task 7 Produces, replace with:
  > "`decide.paired(a_runs, b_runs, measure, boot=1000, seed=0)`:
  > - for each draw, resample the blocks once (the same draw for both arms);
  > - each arm's ratio is the mean over its seeds;
  > - returns the difference and its 95% interval.
  >
  > `decide.verdict(control_runs, variant_runs, eye_ok: bool | None)` returns `"variant"` only when `eye_ok` is
  > True (the controller looked at the panels and wrote what they show), `"pending-eye"` when None, and `"control"` or
  > `"mixed"` otherwise; with every condition's numbers."
- Fix the `fake_runs` tests to pass `eye_ok`.

### H4. The noise estimate is weak, and the repeat run proves reproducibility before value

**Problem.**
- Two seeds give one range. |s0 − s1| falls between 0.09σ and 2.3σ in 90% of cases. So "M2 falls by no more than the
  spread" is close to a coin toss either way.
- **The same-seed repeat measures mps determinism, not noise, and it is already known:**
  - `gpu.py:1-5`: "alone on the GPU it gives the same bits every run";
  - `data/build-logs/solver-text-reader-run1.log` and `run2.log` give identical losses (0.1383 / 0.1818 / 0.7492).
- Owner, 11-16 10:10: "Why are you proving reproducibility before you even know if the result is any good?"

**Correction.** Task 5, Step 4: replace the three runs with:
> 1. control NE s0. Gate it against `baselines` and look at the panel (the value first). If it does not beat today's
>    held-out detectors on M1a-in-box and M1b-in-box (H6/M5), stop and write it up.
> 2. control NE s1 and s2 (the noise: the largest pairwise difference of the three seeds per measure).
> 3. the same-seed repeat only at the end, and only if a recipe is adopted.

Task 7's rule then reads: "noise = max(the control's largest pairwise seed difference, the half-width of the paired
interval)".

### H5. Task 6's line windows cannot be near labels

**Problem.**
- `line_windows` asks for crop-size windows with no Terra word, no other box and no unmatched zone, centred at the
  measured line-to-label distances (tens of px).
- A crop-size window that close to a label holds that label's word. So nearly every candidate is rejected, and the
  sampler falls back to windows far from labels: F7's bias.

**Evidence** (`windows.py`, 2000 p02 / p07 / p09 / p12):

| | 5th percentile of centre-to-nearest-Sol-point distance | median |
|---|---|---|
| word-free 320-px windows | 248-309 px | 404-931 px |
| word-free 200-px windows | 184-235 px | — |

**Correction.** Replace `line_windows` and the two composite bullets of Task 6 with:
> "**Composites inside real crops.** A composite crop is a real crop (`crop_targets`, same rng, same scale and turns),
> changed one of two ways.
> - **(a) Word onto lines.** A clean word is ORed in at page scale before the resize. It is placed by the batch rng:
>   - on the crop's run-rule line ink with the measured crossing probability for its kind, else off line ink;
>   - only where its turned rectangle plus 3 px holds no Terra word, no other box and no unmatched zone;
>   - if no place fits after 20 tries, the next candidate word is drawn.
>
>   `text` |= the word's ink, and `kind` and `angle` are set on those px. The rest of the crop keeps its own targets.
> - **(b) Line onto a word.** A real crop centred on a clean word, with a real line piece ORed across the word. The
>   line piece's px are not text and known; overlap px are text.
>
> The distance distribution is not needed."

This also gives the real label context near lines for free.

### H6. Task 8 collides with T3 and with an open owner question

**(a) The all-sheets detector cannot use the rule as built.**
- `stage.build_text` reads no placement (`stage.py:41`, `placement=False`), so `ft_per_px` is nan and
  `labels.run_px(nan)` raises.
- That is by design: `build.py:139-147` cuts `sfzmaps.transform` so that "a change there must not retrain the
  detector".
- Its only consumer is `solver-text-scores`.

**(b) T3's chain.**
- `solver-text-crops` reads `text-reader/held.json` `unknown_radius.px` for its non-text windows
  (`textread/crops.py:282-286`, `_unknown` at `:213`).
- A new radius changes the crops. Then `solver-text-train` retrains (about 85 min of GPU), and the reader outputs
  change. The owner has not reviewed them yet.
- Task 8 never mentions crops or training.

**(c) The owner.**
- The T3 ledger records an open owner question: "retrain the per-quarter detectors with solver-text's recipe? … Waits
  for the owner's words."
- Memory "show each step on the page": the owner looks before the next change is stacked. Here a new detector would
  sit under T1's walls (c), which the owner chose to keep.

**Correction.** Replace Task 8 with **Task 8 (preview; adoption waits for the owner)**:
> "If Task 7 kept a recipe that beats today's detectors (M5):
> 1. Copy today's solve, `data/solver/solve/` and `walls/`, to `experiments/textlines/before-t8/`.
>    `text-reader-plan/final-before` is from before T1, so it is not the right baseline here.
> 2. In a script (`scripts/textlines/preview.py`), train the four held-out detectors with the chosen recipe, as
>    `reader_stage.build_reader_text` does, with `paths.READER_TEXT_DIR` pointed at
>    `experiments/textlines/preview/text-reader/`.
> 3. Fit the walls as an experiment (`walls.build_walls(candidates=[...])` writes its own folder, `walls.py:306-318`)
>    on that chance, then run the solve on them into the experiment folder.
> 4. Measure M5 against `before-t8` and against `final-before`, and render the panel.
>
> Change neither `DEFAULT_RECIPE` nor `dvc.yaml` / `dvc.lock`. The morning report says that adopting is one commit
> (the recipe in `reader_stage`, with `held.json` keeping `unknown_radius` for `solver-text-crops` and adding
> `zone_radius`) plus `pixi run build solver-solve solver-boxes`. It also says what that would rebuild, from
> `uv run sfzmaps build --dry-run`, and that T3's crops and training stay untouched.
>
> `solver-text` (the all-sheets detector) keeps the boxes rule until the owner decides whether it should read
> placements."

If the controller rebuilds the stages anyway:
- run `--dry-run` first, and stop if `solver-text-train` or any warp or district stage is planned;
- keep `held.json["unknown_radius"]` byte-stable;
- copy `before-t8` first.

---

## MEDIUM

### M1. The test commands skip the existing textnet suite, and several existing tests will break

**Problem.**
- The glob `tests/test_textnet*.py` does not match the existing `tests/test_solver_textnet.py` (52 tests), nor
  `test_codereader.py`, `test_solver_textread.py` or `test_build.py`. So "including every existing textnet test" is
  false.
- These tests break, and must be changed on purpose:
  - `test_default_recipe_trains_as_today`: it pins `asdict(Recipe())` exactly (`test_solver_textnet.py:493-499`), and
    breaks when `labels` and `composites` are added;
  - `test_sheet_data_holds_no_full_page_target_rasters`: every field must be an ndarray / str / int / float / bool
    (`:403-411`), so a `line_bits=None` field fails;
  - `test_the_stages_recipe_trains_the_weights_recorded`: `GUARD_RECIPE` and the pinned losses and digest (`:982-1011`),
    when the recipe changes;
  - `test_build.py:32`, if an import closure changes.
- CI already guards "boxes unchanged" on synthetic maps: `test_crop_targets_equal_full_page_drawing` (`:387`) and the
  GUARD weights digest (`:991`).

**Correction.**
- Every "Run the tests" step:
  > "`uv run pytest -q tests/test_solver_textnet.py tests/test_codereader.py tests/test_solver_textread.py
  > tests/test_build.py tests/test_textnet_*.py tests/test_textlines_*.py`; before each commit, `uv run pytest -q`."
- Task 4: drop the real-sheet `EXPECTED_*` pin in favour of the two existing synthetic guards, which must still pass
  unchanged. Store `line_bits` as an empty `uint8 (0, 0)` array, not None, and extend the field test to allow it.

### M2. Tests that need real data cannot run in CI

**Problem.**
- CI runs `uv run pytest -q` without `pipeline/data` (`.github/workflows/pipeline.yml:49`).
- `conftest.py` points `config.INPUTS_DIR` at `tmp_path`, so `reader_stage.held_blocks()` → `labels.load_labels` fails
  even locally.
- The tests affected:
  - Task 1's glyphs need `font.npz`;
  - Task 2's `tiny_sheet` reaches `border_paths()` / `city()`;
  - Task 3 needs `panel.page_shape`;
  - Task 4 needs `any_real_sheet`;
  - Task 5's `variant.run` loads sheets and blocks;
  - Task 6 needs `real_clean_words`.

**Correction.** Add to Global Constraints:
> "Tests use the synthetic `maps` fixture (`tests/test_solver_textnet.py:244`) or arrays built in the test. Real data
> are injected:
> - `score_sheets(..., geo)`, where `geo` gives each sheet's page rasters of quarter, block and M1a weight; `CityGeo`
>   is the real one;
> - `variant.run(..., sheets=None, areas=None, score=None)`.
>
> A check that needs real data skips when its file is absent, as `test_solver_textread.py:152` does."

### M3. The wiring of "lines-unknown" is unspecified, and the slow path is the easy path

**Problem.**
- No interface says who calls `with_lines`, where the unmatched points and the zone radius live, or where the bank is
  built.
- Computing `matched_points` per crop costs points × words for every crop, about 10-50 ms. That slows the single
  prefetch thread past the GPU step: about 95 ms per 12-crop batch (4,000 steps in 382-416 s, per the build logs).

**Correction.** Task 4 Produces, replace the `line_bits` and `_unknown` bullets with:
> "`zones.prepare(sheets, areas) -> list[SheetData]`, called before `gpu.gpu_turn`. For each sheet it returns
> `replace(sd, points=sd.points[~matched_points(sd)], point_radius=zone_radius(sheets, areas)["px"],
> line_bits=packbits(dilate(line_ink(ink, sd.ft_per_px), 1)))`. So `_unknown` stays as it is.
>
> `crop_targets(..., labels="lines-unknown")` sets `known` False on word px that are `line_bits`, and raises
> ValueError when `line_bits` is empty.
>
> Cost, measured: `run_lines` takes 0.8 s for a 2000 page (17 Mpx), 0.9 s for 1980-90 and 1990-96, and 2.4 s for
> 1960/64/68 (35 Mpx, `run_px` 74). That is about 2.5 min for the 87 sheets on one thread, and 0.3 GB packed.
>
> Lines are cached per sheet in `experiments/textlines/lines/<series>/<sheet>.npz`, keyed by the hash of
> `linerules.py`."

The bank (Task 6) is likewise built before the turn and passed in.

### M4. Make the in-box unknown line ink 1 px wider

**Problem.**
- The run opening leaves a line's ragged edge px outside the run, so they stay text targets.
- `codereader` grows line ink by `LINE_DILATE_PX = 1` before treating it as unobserved (`labels.py:67`). M2 leaves out
  px within 1 px of line ink.

**Correction.** Task 4: "ink inside a word that is `line_ink` grown by 1 px (8-neighbours) is unknown". This is
already part of M3's `prepare`.

### M5. "Beats today's detectors" is undefined, and so is the case where the control is worse

**Correction.** Insert before Task 7's rule:
> "**Adoptable** means that the kept recipe, on held-out NE and on SW, against the held-out detectors
> (`text-reader/`, copied to `baselines/r328/` in Task 3), and against the old r30 quarter models
> (`data/solver/text-<q>/chance/`, held out by quarter):
> - M1a-in-box and M1b-in-box are lower, with the paired interval excluding zero;
> - M2 is not lower by more than the noise;
> - M1c is not higher by more than the noise;
> - of the 64 audit pieces judged text by eye, the number called text is not lower beyond the binomial interval.
>
> The all-sheets detector is in-sample on every quarter: a reference only.
>
> If the control is not adoptable, the variant may still be compared with it, but Task 8 does not run. Write why."

### M6. Task 3's baselines point at the wrong path, and miss a fair held-out baseline

**Problem.**
- The trial's chance is at `experiments/detector-overcall/chance/trial/<sheet>.png` (13 files), not `chance/`. As
  written, the script falls back to predicting it.
- The old quarter models (`data/solver/text-<q>/chance/`) are held out by quarter with 30-px discs: the fairest r30
  baseline on NE.
- Task 8 will overwrite today's held-out detectors.

**Correction.**
- Task 3: fix the trial path, and add `old quarter models` (layout flat, per quarter, scored on its own quarter) as a
  fourth source.
- Add: "First copy today's `data/solver/text-reader/*/chance.png` and `held.json`, and `data/solver/text/chance/`,
  to `experiments/textlines/baselines/{r328,all-sheets}/`."

### M7. The composite texture test compares unlike things, and "edge jitter" is undefined

**Problem.**
- `texture_stats` of whole 320-px composite crops (mostly real line work and context) is compared against
  word-only masks.
- A composite's text ink is a copied real word, so its texture matches by construction. The risk F3 named is the
  drawn lines.

**Correction.** Replace the test with:
> "`texture_stats([c["ink"] & c["text"]])` against the clean words, and `line_texture(drawn line ink)` against
> `line_texture(real run-rule line pieces of the same width class)`. `line_texture` is the SD of each edge's offset
> from a straight fit, and its lag-1/2/4 autocorrelation."

Define in Task 6, Step 3:
- **Edge jitter:** that SD and autocorrelation, measured on run-rule line pieces at least 100 px long on training
  quarters. Drawn edges get Gaussian offsets filtered to match.
- **Crossing probability by kind:** the share of words of that kind with at least 3 run-rule px inside the box. On
  2000 the red team found 62% of codes, 47% of street names and 20% of block numbers.

If the jitter fit has no matching statistic by the time Task 6 runs: real line pieces only tonight, and drawn lines
deferred.

### M8. M1b is not independent for variant 1

**Problem.**
- Variant 1 teaches exactly `line_ink` px as not text (the line pieces and window negatives), and M1b scores
  exactly `line_ink` px.
- That breaks the spec's anti-goal "a measure that reuses the training target's definition", and M1b's fall is partly
  by construction.

**Correction.** Task 7's rule, add:
> "M1a-in-box (independent of every ink rule) carries the line verdict. If M1b-in-box falls but M1a-in-box does not,
> the verdict is `mixed`."

### M9. Spec items no task does

- **(a)** Spec §3: "an overlap px is text, with the line px beside it weighted up, on composites only". There is no
  per-px weight in `loss_weight`.
  - Correction, Task 6: "Not done tonight: no measured weight exists (it would be a guessed parameter). Overlap px are
    text, weight 1."
- **(b)** F5: label-source shares per series.
  - Correction, Task 4, Step 4: "Report per series × kind the share of in-box ink made unknown by the line rule, on
    all 87 sheets (`labels_rule.json`)."
- **(c)** F5: "score one held-out quarter of an old map".
  - Correction, Task 5: "Also predict the 1990-1996 sheets' NE px and report M1b, M1c and M2 there. There is no M1a,
    because no table exists for that map."
- **(d)** Spec §1: M1b by distance to label points. It was dropped silently.
  - Correction: add `m1b_by_point_distance` (0-30, 30-100, 100+ px), or say why it was dropped.

---

## LOW

- **L1. Name clashes.** `scores.line_ink(sd)` (the stroke rule F1 rejected) and `scores.score_sheets` already exist.
  - Name the new ones `linerules.line_px` and `linegate.gate_sheets`, or always qualify them by module.
- **L2. Tautological tests.**
  - `clean_base_px` computed by the code it checks is circular. Assert against counts built from the fixture's own
    glyph and line masks.
  - The panel-keys test checks constants. Assert instead that a source of zeros draws no red.
- **L3. The test session holds the GPU.** `conftest.py:44-60` holds a shared GPU turn for the whole pytest session.
  A long test run makes a variant run wait.
  - `run.json` records the training minutes inside the turn, and `git status --porcelain -- src` at the start.
- **L4. The CLI.** `--name control-repeat` conflicts with the positional name. Use `variant.py control-repeat …`.
- **L5. The panel windows.**
  - `rm2` and `picture0` are the same window (p02 y1600 x1920 n640, `redteam_measure.py:150`). Use the RM-2 zoom
    centres (1850, 2505), (1745, 2175) and (1868, 2218) for `rm2`.
  - Add SW windows: an SW run predicts SW px only, so its NE windows come out blank.
- **L6. Don't touch shared deps.** Add to Global Constraints: "Don't edit any file directly in `src/sfzmaps/`
  (`cli.py` and the rest are deps of every warp stage; the cli.py re-hash precedent), nor `pyproject.toml` / `uv.lock`
  (deps of every stage)."
- **L7. Determinism.** `Bank.build` uses its own rng (`np.random.default_rng([seed, 1])`). `draw` reads the batch rng
  for the composite decision only when `composites > 0`.
- **L8. Memory and definitions.**
  - The bank holds references (sheet index, rectangle, line-piece box), not px crops. About 40 k words as bool crops
    would be hundreds of MB.
  - "Letter-sized" in the audit: reuse `scripts/spikes/redteam_text_vs_line/redteam_missed.py`'s definition.
- **L9. Timing.** Before a composites run, time 100 batches of `draw` with composites at P and at 0. The GPU step
  takes about 95 ms per batch; if the draw is slower, say so, because the run will be draw-bound.

---

## Questions asked, answered

1. **M1a mapping.**
   - Not workable as written (H1).
   - Reuse page → grid through `data._world_at` + `grid.to_grid`, the quarter split's path, checked against
     `page.sheet_px` (`to_pixel`).
   - `prof` is usable and sharp, with the weight formula in H1.
2. **Task 4.**
   - Lines precomputed per sheet: about 2.5 min on one thread for 87 sheets, 0.3 GB packed, microseconds per crop.
     The chain rule is not local, so it must be full-page.
   - "Byte for byte" holds for the targets, but two existing tests break: the None field and the exact recipe dict.
     `held.json` / `model.pt` bytes also change (`asdict(recipe)` gains keys).
   - The real-sheet pin is CI-skipped. Use the existing synthetic guards (M1, M2).
3. **GPU time.**
   - About 6.5-7 min per held-out training (382-416 s in the build logs). Predicting one quarter: under 1 min for 13
     sheets.
   - This plan's runs: Task 5 about 25 min, Task 7 about 25-30 min, Task 8 as written about 36 min (28 + 8).
   - T3's training (about 85 min) holds the lock until about 01:00. If Task 8's radius change reaches T3's crops,
     another 85 min.
4. **Task 6.**
   - Drawing composites from the prefetch thread's rng is feasible.
   - `Bank.build` is affordable once the lines are precomputed (clean words: about 10 s).
   - Windows near labels are infeasible (H5).
   - The jitter and crossing definitions are underspecified (M7).
   - The texture test compares unlike things (M7).
5. **The decision.**
   - Not implementable without per-block tallies (H3).
   - n = 2 is not defensible alone (H4).
   - A control worse than today's is unhandled (M5).
6. **Task 8.**
   - It conflicts with `solver-text`'s no-placement design and with T3's crops / training through `held.json`.
   - DVC hazards: the HEAD / dirty-tree guard, an out-of-date `dvc.yaml`, and `cli.py` / `pyproject.toml` re-hashes.
   - It should preview, and wait for the owner to adopt (H6).
7. **Missing pieces.** M9 (a)-(d), M1a split by box (H2), the baseline copies (M6).
8. **Order and parallelism.**
   - Tasks 1-3 cannot run beside T3's build in this worktree (C1).
   - Measures first is right.
   - Inside Task 5, value comes before the repeat (H4).
   - Once T3 is done: Task 2 can run beside Task 4 (disjoint files after Task 1), and Task 6's code beside Task 5's GPU
     runs. Commit one at a time (git index lock).
