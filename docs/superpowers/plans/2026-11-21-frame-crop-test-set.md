# Frame Crop, Test-Set-Driven Finish: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the frame crop against an honest measure: every decision and every new label is judged on the 22
test sheets that are never trained on, and the cut traces straight lines on printed ink or the file's edge.

**Architecture:**
- The layout comes from frozen DINOv2 features plus small heads trained on the training labels (built).
- A test command trains the heads, decides the test sheets and prints, per sheet and pooled, the share of frame and
  legend ink left in and of map ink cut.
- Two decisions are scored the same way: the simple cut (built) and the straight-line cut (built; it fails on some
  unseen pages).
- The straight-line cut's failures are debugged to root causes on named pages, fixed with tests, and re-scored.
- The better decision ships, and labels are added where the test set shows weakness.

**Tech Stack:** Python 3, PyTorch (mps/cpu), `transformers` (DINOv2), numpy, scipy, opencv, rasterio, shapely,
Pillow, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-frame-crop-design.md`: read "Amendment 3" with its two follow-up notes,
and "Amendment 4: a test set, and what it showed". This plan replaces Tasks 7–9 of
`docs/superpowers/plans/2026-11-21-frame-crop-combined.md`; that plan's Tasks 1–6 are built and stay.

## Global Constraints

- The test set (`pipeline/inputs/frame-labels-test/`) is never trained on and never used to fit a number: no head,
  temperature, parameter or bound reads it. Code enforces this, not habit.
- The measure is, per test sheet and pooled: `frame_left` = frame + removed-block ink kept / all such ink, and
  `map_cut` = map ink deleted / all map ink. Reports print percentages first.
- First test-set numbers (heads on 39 labels, T 1.54), median sheet: simple cut 0.78% left in / 0.043% cut;
  straight-line cut 0.00% / 0.007%. The straight-line cut was better on 20 of 22 sheets and leaves 0.00% on most
  1960-2000 sheets. Both are poor on 1921-1935 (3.8-8.5% left in, 0.6-3.0% cut) and on 1948/6a (50% left in). Pooled
  shares come from Task 1's command: the spike's pooled figures dropped sheets with nothing left in and are not to
  be quoted.
- The rule of the cut: a side is a straight line along the inner edge of the printed frame ink, or the file's edge
  where there is no frame. Blank paper never moves a line.
- Costs: frame or removed-block ink left in is 3× map ink cut (`decide.COST_FRAME = 3.0`, `COST_MAP = 1.0`).
- $0, no LLM output. Licences MIT/BSD/Apache. One process for every edition: no per-series rule.
- No run without a stated time budget. At the first check where it is badly over, or its result is surprising, stop
  and report what is known. Long work runs in the background and prints progress as it goes.
- Disk: keep pictures and score files. One run's per-page maps at a time, with chances as uint8.
- One GPU job at a time (`sfzmaps.gpu.gpu_turn`). At most 4 worker processes.
- Never commit `pipeline/data/`. Tests test our behaviour only. Commands run from `pipeline/`. The full suite is
  `uv run pytest -q -n 6`.
- Commit messages end with the session's attribution lines.

## Review Focus

1. **A test sheet that also has a training label** (someone adds one later): the run refuses with both paths named.
   Test in Task 1.
2. **A test label with no `remove` blocks, or a frameless sheet whose only deletions are blocks** (1948/6a):
   `frame_left` is still defined and reported. Test in Task 1.
3. **A tinted scan** (paper luminance about 0.93, not 1.0) with blank areas the model calls "outside": no side moves
   off the frame ink. Test in Task 3.
4. **A page with a frame on three sides and a legend bar on the fourth:** the bar's inner edge is the fourth side,
   one straight line, not notches. Test in Task 3.
5. **Pictures:** ink the cut keeps is never drawn in the deleted colour because the model was unsure of it. Test in
   Task 2.

---

### Task 1: The test command

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/data.py`, `frame/run.py`, `frame/calibrate.py`, `cli.py`
- Test: `pipeline/tests/test_frame_run.py`, `test_frame_data.py`, `test_cli.py`

**Interfaces:**
- Produces, in `data.py`:
  - `TEST_LABELS = INPUTS_DIR / "frame-labels-test"`;
  - `test_sheets(root=TEST_LABELS) -> list[SheetId]`;
  - `check_split(train, test) -> None`: raises `ValueError` naming every sheet that is in both.
- Produces, in `run.py` (replacing `held_out`, `search_params`, `GRID`, `START`, `SEARCH_ROUNDS`, `leave_one_out`
  and their tests, which are deleted):
  - `TEMPERATURE_FOLDS = 4` (computational).
  - `PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)`: the first spike's numbers. They are constants
    until the test set shows a reason to change them, and they are not searched.
  - `fold_temperature(feats, targets, device) -> tuple[float, tuple[float, float]]`:
    - the training sheets are dealt round-robin in sorted order into `TEMPERATURE_FOLDS` folds;
    - for each fold, `HEAD_SEEDS` heads are trained on the others;
    - the temperature is fitted on ink-weighted samples of the held folds' logits (`calibrate.fit_temperature`), with
      its interval.
  - `fit(device, out, turn) -> dict`:
    1. extract features (as now);
    2. `fold_temperature`;
    3. train the final heads on every training sheet;
    4. save `model/` (heads, `model.json` with the temperature, block bounds, the reducer record and the training
       sheets).

    `check_split` runs first.
  - `DECISIONS = {"simple": ..., "lines": ...}`: the per-pixel `decide.decide`, and `shape.decide` at `PARAMS`.
  - `score_test(out, decisions=("simple", "lines")) -> dict`:
    - for each test sheet and decision, on `PROCESSES` workers: the decision, `write_outputs` under
      `out/test/<decision>/`, and `scores.sheet_scores` against the test label;
    - written to `out/test/scores.json`: per sheet, pooled and median, for each decision, plus the paired difference
      lines − simple with a sheet-bootstrap interval;
    - progress is logged per sheet.
  - `format_test(result) -> str`: one line per sheet, `series/sheet: simple left in x% cut y% | lines left in x% cut
    y%`, then the pooled and median lines.
- Produces, in `cli.py`: `sfzmaps frame --test` (fit if `model/` is stale or missing, then `score_test`, then the
  pictures of Task 2). `--held-out` is removed. The plain command and `--predict-only` use `fit`'s model.

- [ ] **Step 1: Write the failing tests** (synthetic pages, `tiny_backbone`, heads stubbed where the test is about
  orchestration, and say so):

```python
def test_split_overlap_is_refused(tmp_path):
    with pytest.raises(ValueError, match="1960/6"):
        data.check_split([("1960", "6"), ("1960", "1")], [("1960", "6")])

def test_no_test_sheet_reaches_training(fake_world, spy_train_head):
    run.fit("cpu", fake_world.out, turn=contextlib.nullcontext)
    assert not (spy_train_head.sheets_seen & set(data.test_sheets()))

def test_temperature_comes_from_folds_not_the_test_set(fake_world, spy_fit_temperature): ...
    # every sample passed to fit_temperature is of a training sheet, from heads that did not train on it

def test_score_test_reports_both_decisions(fake_world): ...   # scores.json has simple and lines for every test sheet

def test_frameless_test_sheet_with_only_blocks_is_scored(fake_world): ...   # frame_left defined; equals kept/all

def test_cli_frame_test_refuses_an_overlap_before_any_gpu_work(...): ...
```

- [ ] **Step 2: Run them.** Expect failures.
- [ ] **Step 3: Implement.** Delete the leave-one-out and search code and its tests.
- [ ] **Step 4: Run the full suite.** Expected: PASS.
- [ ] **Step 5: Run it for real** (budget: 8 minutes; it prints per sheet): `pixi run uv run sfzmaps frame --test
  --device mps`. Paste the printed table into the report. The "simple" pooled line should be near 2.4% / 0.36%; say
  why if it isn't. One cause is known: the temperature now comes from folds.
- [ ] **Step 6: Commit** `frame: the test command (never-trained test sheets, fold temperature); leave-one-out retired`.

---

### Task 2: Pictures that show the cut, not the doubt

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/look.py`
- Test: `pipeline/tests/test_frame_look.py`

**Interfaces:**
- Produces:
  - `sheet_picture(lum, deleted, side=LONG_SIDE, polygons=None, label=None) -> Image`: ink the cut keeps in
    `KEPT_RGB`, ink it deletes in `DELETED_RGB`, with no in-between colours. The cut's outline is drawn straight from
    `polygons` in magenta, and the test label's outline in green when `label` is given.
  - `doubt_picture(lum, q, side) -> Image`: ink shaded by the chance of deletion, for debugging only, never in the
    mosaics.
  - `test_mosaic(out, decision) -> Path`: every test sheet for one decision, captioned `left in x% cut y%`.
  - `compare_mosaic(out) -> Path`: simple | lines side by side per test sheet.

- [ ] **Step 1: Write the failing tests**

```python
def test_kept_ink_is_never_the_deleted_colour():
    lum = np.full((40, 40), 1.0, np.float32); lum[10:30, 10:30] = 0.0
    deleted = np.zeros((40, 40), bool); deleted[:, :20] = True
    im = np.asarray(look.sheet_picture(lum, deleted, side=40))
    assert (im[20, 25] == look.KEPT_RGB).all() and (im[20, 15] == look.DELETED_RGB).all()

def test_label_outline_is_drawn_in_green(): ...
def test_compare_mosaic_has_both_decisions(tmp_path): ...
```

- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement.** Update the callers in `run.py` and `cli.py`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: pictures show the cut in two plain colours; test and compare mosaics`.

---

### Task 3: Debug the straight-line cut to root causes (REQUIRED SUB-SKILL: superpowers:systematic-debugging)

**Files:**
- Modify: whichever of `pipeline/src/sfzmaps/frame/shape.py`, `notches.py`, `blocks.py`, `paper.py`, `geometry.py`
  the causes are in
- Test: the matching `pipeline/tests/test_frame_*.py`
- Create: `pipeline/scripts/spikes/issue71_debug_lines.py` (the evidence script; committed)

The known failures, all reproducible from the saved heads (`data/frame-look39/heads.pt`) and features:

| page | what is wrong |
|---|---|
| 1975/20180102114142444-p18 ("4H") | the frame is fitted inside the real one, about −1.3° off; up to 6 "blocks" are cut from the street grid and the ocean; the same at β = 10, 3 and 0 |
| 2000-digital-su/2000-digital-su-p11 | the top side goes to the file's edge and two notches that don't meet remove the margin: a box of margin and a stretch of the top rule stay in |
| 1948/4 | the bottom row of blocks is clipped |
| 1980-1990-su/1980-1990-su-p01 | a large rectangle is cut from the map |
| any test sheet where `lines` is worse than `simple` in Task 1's table | from the table |

- [ ] **Step 1: Evidence before any fix.** For each page, the script saves pictures of each stage's output: the
  layout's three chances, the cost field's sign on paper and on ink, the plain and notch-aware starts, the blocks and
  notches accepted with their gains, the re-fit, and the on-paper slide. Write down, per page, the first stage whose
  output is wrong and the numbers that make it choose wrongly.
- [ ] **Step 2: Name each root cause in one sentence, with its evidence.** Then show the owner, before fixing. The
  suspects to confirm or rule out, not to assume:
  - tinted paper: the cost field is ink × chances with ink = 1 − luminance, so paper at 0.93 carries weight and the
    model's opinion of blank paper moves lines;
  - removed-block candidates from a few high-p₂ patches growing over map ink;
  - a framed side losing to "file edge plus notches" in the joint objective.
- [ ] **Step 3: For each confirmed cause, a failing synthetic test first.** Each test builds the smallest page that
  reproduces the cause, including:

```python
def test_blank_paper_never_moves_a_side_on_a_tinted_scan():
    # paper at 0.93, a frame rule, a wide blank band inside it that the chances call "outside": the side stays on
    # the rule's inner edge (within 2 px)
    ...
def test_a_ruled_side_is_one_line_not_notches():
    # a full-width rule 300 px in from the file's edge on one side: the side is on the rule; no notch on that side
    ...
```

- [ ] **Step 4: Fix each cause at its source.** Run the tests, then the full suite.
- [ ] **Step 5: Re-score** with `sfzmaps frame --test` (budget: 8 minutes). Report both decisions' tables and the
  pictures of the five pages above, before | after.
- [ ] **Step 6: Commit** one commit per cause: `frame: <cause> (test, fix)`.

If three fixes in a row don't move the test numbers or the pages, stop and bring the evidence to the owner. Do not
try a fourth.

---

### Task 4: Choose the cut (controller, with the owner)

- [ ] **Step 1:** Send the owner `compare_mosaic` and the two tables, with any page where either decision is
  embarrassing called out.
- [ ] **Step 2:** The owner picks. Record the choice as `run.DEFAULT_DECISION` with a comment giving the date and the
  numbers. If neither is acceptable, stop: the owner decides what to change.

### Task 5: Labels where the test set shows weakness (hand work; agents draw, the controller checks)

**Files:**
- Create: new files under `pipeline/inputs/frame-labels/` only.

- [ ] **Step 1:** Six more training sheets from 1921, 1929, 1935 and 1948, including two frameless 1948 photos with
  their title block, compass rose and badge. In each series, take the unlabelled, non-test pages where the three
  heads disagree most: the ink-weighted share of ink on which their cuts differ, computed by a short script. Use the
  labelling brief `.superpowers/sdd/2026-11-21-frame-crop-combined/test-labels-brief.md`, pointed at the training
  folder.
- [ ] **Step 2:** `sfzmaps frame --test` again (budget: 10 minutes). Report the table before | after per series. A
  series that got worse is a finding, not noise to average away: look at its pictures.
- [ ] **Step 3:** Commit the labels: `issue #71: six more training labels (1921-1948, where the heads disagreed)`.

### Task 6: Every page (controller)

- [ ] **Step 1:** Time the first 20 pages and tell the owner the total before continuing. Check the free disk, and
  delete the previous run's per-page maps.
- [ ] **Step 2:** `pixi run uv run sfzmaps frame` in the background, printing per page.
- [ ] **Step 3:** Send the owner every series' mosaic in the two plain colours, and the top of each ranking in
  `pages.json`. The page list fix (`1990-1996-zoning-p05` is a text page, not a map) goes in
  `pipeline/inputs/page_kinds.csv` with the owner's OK.

### Task 7: Cleanup, review, pull request

- [ ] **Step 1:** Delete dead code and stale spikes of superseded designs. A whole-branch review covers:
  - the deferred minors in the three ledgers under `.superpowers/sdd/`;
  - the licence notices;
  - the fresh-clone check.
- [ ] **Step 2:** Run the full suite; push; open the PR against `main` with the test-set table and the pictures. The
  owner merges.
