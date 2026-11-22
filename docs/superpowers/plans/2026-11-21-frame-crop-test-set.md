# Frame Crop, Finish: Label Boxes and Shipping. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the straight-line cut the owner approved, with the one thing they flagged fixed: the label boxes (title
and legend blocks, compass roses, section badges).

**Architecture:**
- The layout (frozen DINOv2 features plus small heads) and the straight-line cut (`shape.decide`) are built and stay
  as they are. The frame fit is not reworked.
- A scoring command reports the frame and the label boxes separately, on two sets:
  - a dev set for iterating: the training sheets, each cut by heads that did not train on it;
  - the 22 test sheets, never trained on, looked at only as a check.
- The label-box work has two parts with separate causes: the box fit (code) and boxes the model does not see (labels).

**Tech Stack:** Python 3, PyTorch (mps/cpu), `transformers` (DINOv2), numpy, scipy, opencv, rasterio, shapely,
Pillow, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-frame-crop-design.md`: amendments 2, 3 and 4, and "Amendment 4, after the
straight-line test". This plan replaces Tasks 7–9 of `docs/superpowers/plans/2026-11-21-frame-crop-combined.md`.

## What the owner said, and what was measured (2026-11-21)

- **Owner, positive:** "well those straight-line cuts are practically perfect"; "we're getting close".
- **Owner, negative:** "the only issue I see is the map label box being a bit wrong". Earlier: "such wobbly magenta"
  (the simple cut), and kept ink that looked half-deleted in the pictures.
- **Measured on the saved straight-line cuts of the 22 test sheets** (heads on 39 labels, T 1.54,
  `Params(1e4, 3, 1e3)`), pooled:

| | left in | note |
|---|---:|---|
| frame ink | 0.24% | 0.00% on 18 sheets; 1929 p14 7.7% (its label has a stated doubt), 1964 p14 0.7% |
| label-box ink | 31.5% | label boxes are only 3.3% of the ink to delete, so the pooled figure hid them |
| map ink cut | 0.32% | 1921 3.0%, 1935 1.1%, 1929 0.6%, others at most 0.41% |

- **Per labelled box on the test sheets:**

| sheet | box | deleted | model's chance of deletion on its ink |
|---|---|---:|---:|
| 1921/…348509 | title and legend | 100% | 0.98 |
| 1921/…348509 | compass rose (over the map) | 0% | 0.02 |
| 1921/…348509 | badge | 13% | 0.17 |
| 1929 p14 | title, compass (in open water), badge | 95–100% | 0.88–0.96 |
| 1935 p12 | title and legend | 78% | 0.62 |
| 1935 p12 | badge | 32% | 0.27 |
| 1948/6a | title and legend | 78% | 0.60 |
| 1948/6a | compass rose (over the map) | 0% | 0.00 |
| 1948/6a | badge | 99% | 0.74 |
| 1980-1990-su p05 | small block | 0% | 0.01 |

- **Fitted boxes that are wrong:**
  - rectangles at 22.4° (1935 p12), 19.3° (1980-1990-su p05), −19.3° (1929 p14), 5.5° and −2.9° (1948/6a);
  - an ellipse over the square title block on 1921, spilling into the map;
  - an ellipse on 1990-1996-zoning p30, which has no labelled box (0.41% of its map cut).
- **Code fact:** `blocks.py` takes a rectangle's angle candidates from the model blob's own moment angle
  (`MOMENT_ANGLES`) as well as from the frame's angle (`FRAME_ANGLES`). In the 39 training labels, every rectangular
  box is drawn square to the page, within 1.7° of its frame's angle.
- **Two causes, so two tasks:**
  - where the model sees the box (chance 0.6 or more), the fit is skewed: code, Task 3;
  - where the model does not see it (chance under 0.3: compass roses drawn over the map, small badges), no fit can
    delete it: labels, Task 4.

## Global Constraints

- **The frame is right; keep it.** A change for label boxes must leave every test and dev sheet's fitted frame (its
  angle, four sides, arcs and notches) exactly as it was. The scoring command records the fit and reports any
  difference.
- **The test set is not a tuning set.** `pipeline/inputs/frame-labels-test/` is never trained on and never used to fit
  a number. Fixes are iterated on the dev set and the synthetic tests. The test set is run when a task says so, and
  its result is reported, not optimised.
- **The measure,** per sheet and pooled, as percentages:
  - frame ink left in;
  - label-box ink left in, and per labelled box the share deleted;
  - map ink cut by the frame, and map ink cut by fitted boxes.

  Pooled means summed ink over summed ink (a sheet with nothing left in still counts).
- Costs: frame or label-box ink left in is 3× map ink cut (`decide.COST_FRAME = 3.0`, `COST_MAP = 1.0`).
- $0, no LLM output. Licences MIT/BSD/Apache. One process for every edition: no per-series rule.
- **No run without a stated time budget.** At the first check where it is badly over, or surprising, stop and report
  what is known. Long work runs in the background and prints progress per sheet. The controller relays progress to
  the owner every few minutes.
- **Nothing starts without the owner's plain go.** Praise or a question is not a go.
- Disk: keep pictures and score files. One run's per-page maps at a time, chances as uint8.
- One GPU job at a time (`sfzmaps.gpu.gpu_turn`). At most 4 worker processes.
- Never commit `pipeline/data/`. Tests test our behaviour only. Commands run from `pipeline/`. The full suite is
  `uv run pytest -q -n 6`.
- Commit messages end with the session's attribution lines.

## Review Focus

1. **A model blob that covers three quarters of a square box, along a diagonal:** the fitted box is square to the
   frame, not turned to the blob. Test in Task 3.
2. **A page with no framed side:** boxes are square to the page. Test in Task 3.
3. **A test sheet that also has a training label:** the run refuses, naming both paths. Test in Task 1.
4. **A sheet with no labelled box, or a frameless sheet whose only deletions are boxes** (1948/6a): every measure is
   still defined. Test in Task 1.
5. **Pictures:** kept ink is never drawn in the deleted colour because the model was unsure of it. Test in Task 2.

---

### Task 0: Re-cut the pages the owner flagged, with the current model (controller; no new code)

The four bad pages were seen with the old heads (21 labels) and other settings. Nobody has looked at them with the
model whose test cuts the owner approved.

- [ ] **Step 1:** With the saved heads (`data/frame-look39/heads.pt`) and `Params(1e4, 3, 1e3)`, cut 1975 p18 ("4H"),
  2000-digital-su p11, 1948/4 and 1980-1990-su p01, by pointing `scripts/spikes/issue71_test_shape.py`'s page list at
  them (unscored: they have no labels). Budget: 3 minutes.
- [ ] **Step 2:** Send the owner the four pictures in plain colours. For each page, say whether the earlier fault is
  still there.
- [ ] **Step 3:** A page that is still wrong is recorded in the ledger with its picture. If its fault is in the frame
  or notches, it becomes a debugging item for the owner to rank against the label boxes. It is not fixed in passing.

### Task 1: The scoring commands

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/data.py`, `frame/scores.py`, `frame/run.py`, `cli.py`
- Test: `pipeline/tests/test_frame_data.py`, `test_frame_scores.py`, `test_frame_run.py`, `test_cli.py`

**Interfaces:**
- Produces, in `data.py`:
  - `TEST_LABELS = INPUTS_DIR / "frame-labels-test"`;
  - `test_sheets(root=TEST_LABELS) -> list[SheetId]`;
  - `check_split(train, test) -> None`: raises `ValueError` naming every sheet that is in both.
- Produces, in `scores.py`:
  - `split_scores(outside, removed, polygons, remove, lum) -> dict` with keys:
    - `frame_ink`, `frame_left`: ink of class 1, and how much of it is kept;
    - `box_ink`, `box_left`: ink of class 2, and how much of it is kept;
    - `boxes`: one entry per labelled box, `{"kind", "size", "deleted_share"}`;
    - `map_ink`, `map_cut_frame` (map ink in `outside`), `map_cut_boxes` (map ink in `removed`).
  - `pooled_split(rows) -> dict`: the four shares from summed ink.
- Produces, in `run.py` (replacing `held_out`, `search_params`, `GRID`, `START`, `SEARCH_ROUNDS`, `leave_one_out`
  and their tests, which are deleted):
  - `TEMPERATURE_FOLDS = 4` (computational).
  - `PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)`: the numbers behind the cuts the owner approved.
    Constants, not searched.
  - `fit(device, out, turn) -> dict`:
    1. `check_split`;
    2. features (reused when the page list matches);
    3. training sheets dealt round-robin in sorted order into `TEMPERATURE_FOLDS` folds; per fold, `HEAD_SEEDS` heads
       trained on the others; the held sheets' logits saved to `model/dev_logits.npz`;
    4. the temperature fitted on ink-weighted samples of those held logits, with its interval;
    5. the final heads on every training sheet;
    6. `model/model.json`: temperature, block bounds, reducer record, training sheets.

    The model is stale when `model.json` is missing or its training sheets or reducer record differ.
  - `score_dev(out) -> dict`: each training sheet cut from its held logits; written under `out/dev/`.
  - `score_test(out) -> dict`: each test sheet cut by the final heads; written under `out/test/`.
  - Both use `PROCESSES` workers, log one line per finished sheet, and write `scores.json` (per sheet and pooled
    `split_scores`) and `fits.json` (each sheet's frame angle, sides, arcs, notches and boxes).
  - `fit_changes(before: Path, after: Path) -> list[str]`: the sheets whose frame, arcs or notches differ between
    two `fits.json` files.
  - `format_split(result) -> str`: one line per sheet (`series/sheet: frame left x% | boxes left y% (n of m deleted)
    | map cut by frame a%, by boxes b%`), then the pooled line.
- Produces, in `cli.py`: `sfzmaps frame --dev` and `sfzmaps frame --test` (each fits first if the model is stale).
  `--held-out` is removed. The plain command and `--predict-only` use `fit`'s model and `PARAMS`.

- [ ] **Step 1: Write the failing tests** (synthetic pages, `tiny_backbone`; heads stubbed where the test is about
  orchestration, and say so):

```python
def test_split_overlap_is_refused():
    with pytest.raises(ValueError, match="1960/6"):
        data.check_split([("1960", "6"), ("1960", "1")], [("1960", "6")])

def test_split_scores_separates_frame_from_boxes():
    # a page with a frame rule (all deleted) and one labelled box (half its ink kept): frame_left == 0,
    # box_left == half the box's ink, boxes[0]["deleted_share"] == 0.5
    ...

def test_map_cut_is_split_by_what_cut_it(): ...          # map ink in `removed` counts under map_cut_boxes only
def test_sheet_without_boxes_is_scored(): ...            # box_ink == 0; pooled shares still defined
def test_no_test_sheet_reaches_training(fake_world, spy_train_head): ...
def test_dev_sheets_are_cut_by_heads_that_did_not_train_on_them(fake_world, spy_train_head): ...
def test_fit_changes_names_a_sheet_whose_frame_moved(tmp_path): ...
def test_cli_frame_test_refuses_an_overlap_before_any_gpu_work(...): ...
```

- [ ] **Step 2: Run them.** Expect failures.
- [ ] **Step 3: Implement.** Delete the leave-one-out and search code and its tests.
- [ ] **Step 4: Run the full suite.** Expected: PASS.
- [ ] **Step 5: Run for real**, in the background, printing per sheet:
  - `pixi run uv run sfzmaps frame --dev --device mps`. Budget: 18 minutes (15 heads at about 50 s each, then 39
    sheets on 4 workers).
  - `pixi run uv run sfzmaps frame --test`. Budget: 6 minutes.

  Report both tables, the fitted temperature, and the test table beside the approved run's (frame 0.24%, boxes 31.5%,
  map cut 0.32%). The temperature now comes from folds, so say which sheets changed and by how much.
- [ ] **Step 6: Commit** `frame: dev and test scoring, frame and label boxes scored apart; leave-one-out retired`.

### Task 2: Pictures in two plain colours

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/look.py` (and its callers in `run.py`, `cli.py`)
- Test: `pipeline/tests/test_frame_look.py`

**Interfaces:**
- Produces:
  - `sheet_picture(lum, deleted, side=LONG_SIDE, polygons=None, label=None) -> Image`: kept ink in `KEPT_RGB`, deleted
    ink in `DELETED_RGB`, nothing in between. The cut's outline from `polygons` in magenta. The label's frame and
    boxes in green when `label` is given.
  - `set_mosaic(out, name) -> Path`: every sheet of the dev or test set, captioned with its `format_split` line.

- [ ] **Step 1: Write the failing tests**

```python
def test_kept_ink_is_never_the_deleted_colour():
    lum = np.full((40, 40), 1.0, np.float32); lum[10:30, 10:30] = 0.0
    deleted = np.zeros((40, 40), bool); deleted[:, :20] = True
    im = np.asarray(look.sheet_picture(lum, deleted, side=40))
    assert (im[20, 25] == look.KEPT_RGB).all() and (im[20, 15] == look.DELETED_RGB).all()

def test_label_is_drawn_in_green(): ...
def test_set_mosaic_captions_each_sheet(tmp_path): ...
```

- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `frame: pictures show the cut in two plain colours, the label in green`.

### Task 3: Label boxes sit square (REQUIRED SUB-SKILL: superpowers:systematic-debugging)

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/blocks.py` (and `shape.py` only if the evidence points there)
- Test: `pipeline/tests/test_frame_blocks.py`, `tests/synthetic_frame.py`

**The rule (owner: the label box is wrong; labels: every rectangular box is square to its sheet):** a rectangular box
takes the frame's angle. A page with no framed side takes the page's axes. The model's blob says where a box is, not
which way it is turned. Ellipses keep their own angle: a compass needle points where it points.

- [ ] **Step 1: The dev table before any change.** From Task 1's `--dev` run: per labelled box, the share deleted and
  the fitted box's kind and angle. This is the number each fix is judged on.
- [ ] **Step 2: Failing tests for the known cause**

```python
def test_a_box_is_square_to_the_frame_when_the_blob_is_diagonal():
    # a ruled 600 x 400 box inside a frame turned 0.5 degrees; the chances call three quarters of it "removed",
    # along a diagonal: the fitted rectangle's angle is within 0.5 of the frame's and it covers the whole box
    ...

def test_a_box_on_a_frameless_page_is_square_to_the_page(): ...
```

- [ ] **Step 3: Fix.** Rectangle angle candidates are `frame.phi + FRAME_ANGLES` only. `MOMENT_ANGLES` stays for
  ellipses.
- [ ] **Step 4: Run the tests, the full suite, then `sfzmaps frame --dev`** (budget: 6 minutes; the model is not
  refitted). Report the per-box table before | after, `fit_changes` (expected: none), and the dev mosaic.
- [ ] **Step 5: Commit** `frame: rectangular label boxes take the frame's angle (test, fix)`.
- [ ] **Step 6: What is still wrong, evidence first.** For each dev box still under-deleted, and each fitted box that
  matches no label, save pictures of the stages (the "removed" chance, the candidate pieces, each candidate shape's
  gain, the one kept). Write the first stage that is wrong and its numbers. Known items to explain, not assume:
  - an ellipse chosen for a square title block (1921);
  - a box found where no box is labelled (1990-1996-zoning p30);
  - a box the model is sure of on three quarters of its ink being cut to three quarters (1935 p12, 1948/6a);
  - 1948/6a's frame angle of −1.89° on a sheet labelled as having no frame.
- [ ] **Step 7: Show the owner the causes before fixing.** Then one failing synthetic test and one fix per cause, each
  re-scored on the dev set, one commit each.
- [ ] **Step 8: Stop rule.** If two fixes in a row don't raise the dev boxes' deleted share, stop and bring the
  evidence to the owner.

### Task 4: Labels for the boxes the model does not see (hand work; runs alongside Tasks 1–3)

**Files:**
- Create or modify: files under `pipeline/inputs/frame-labels/` only.

The model gives a chance of 0.00–0.02 to compass roses drawn over the map and 0.01–0.27 to some small badges. Those
cannot be fixed in the fit.

- [ ] **Step 1: Check the labels we train on.** A picture per training sheet from 1921–1948, with its label in green.
  The controller looks for a compass rose, badge or title block left unmarked: an unmarked one teaches the model to
  keep it. Fix any found, with the labelling brief's ink check.
- [ ] **Step 2: Ten more training sheets,** chosen by what is on the page, from thumbnails, never from predictions and
  never a test sheet:
  - two each from 1921, 1929, 1935 and 1948 that have a compass rose drawn over the map and a section badge;
  - two from the 1980–1990 special-use series with the small block that was missed on test sheet p05.

  Drawn by agents with the brief `.superpowers/sdd/2026-11-21-frame-crop-combined/test-labels-brief.md`, pointed at
  the training folder. The controller checks each against its scan.
- [ ] **Step 3: Commit the labels.** Then `sfzmaps frame --dev` (budget: 18 minutes, the model is refitted). Report
  the per-box table before | after.
- [ ] **Step 4: A known limit, stated up front.** One layout patch covers about 45–55 scan px, so an 80 px badge is
  under two patches wide. If badges are still missed after the labels, stop and bring it to the owner with the
  pictures. Do not build a workaround.

### Task 5: The test check and the owner's eye (controller)

- [ ] **Step 1:** `sfzmaps frame --test` (budget: 6 minutes). Report the table beside the approved run's, with
  `fit_changes`. A frame that moved, or a sheet that got worse, is shown with its picture.
- [ ] **Step 2:** Send the test mosaic and Task 0's four pages again, in plain colours with labels in green.
- [ ] **Step 3:** The owner says whether it ships. Anything the owner flags goes back through Task 3 or Task 4.

### Task 6: Every page (controller)

- [ ] **Step 1:** Time the first 20 pages and tell the owner the total before continuing. Check the free disk, and
  delete the previous run's per-page maps.
- [ ] **Step 2:** `pixi run uv run sfzmaps frame` in the background, printing per page.
- [ ] **Step 3:** Send every series' mosaic in the two plain colours. The page list fix (`1990-1996-zoning-p05` is a
  text page, not a map) goes in `pipeline/inputs/page_kinds.csv` with the owner's OK.

### Task 7: Cleanup, review, pull request

- [ ] **Step 1:** Delete dead code and the spikes of superseded designs. A whole-branch review covers:
  - the deferred minors in the three ledgers under `.superpowers/sdd/`;
  - the licence notices;
  - the fresh-clone check.
- [ ] **Step 2:** Run the full suite; push; open the PR against `main` with the test table and the pictures. The owner
  merges.

## Red team of the first draft, and the edits it caused

| # | Problem in the first draft | Edit |
|---|---|---|
| 1 | It treated the straight-line cut as broken (a five-page debugging task, suspects about paper tint moving the sides) after the owner called it practically perfect. Measured: frame ink left in is 0.24%. | The frame fit is not touched. A constraint and `fit_changes` guard it. |
| 2 | It asked the owner to choose between the simple and straight-line cuts. The owner had already chosen. | The choice task and the simple cut's scoring are gone. |
| 3 | The label box, the owner's one complaint, was a note in the ledger, not a task. | It is the centre of the plan: Tasks 3 and 4, with the measurements above. |
| 4 | Its measure pooled frame and label-box ink. Boxes are 3.3% of that ink, so 31.5% of box ink left in showed up as 1.3%. | `split_scores`: frame, boxes (per box) and map cut by frame or by boxes, reported apart. |
| 5 | Fixes were to be re-scored on the 22 test sheets again and again. Only 5 of them have boxes; tuning on them would make the test set dishonest. | A dev set (training sheets cut by heads that did not train on them) for iterating; the test set is a check. |
| 6 | The four pages the owner flagged were listed as known failures of this model. They were seen with older heads and settings. | Task 0 re-cuts them first, before any code. |
| 7 | One cause was assumed for the label box. There are two, with different fixes. | Code for boxes the model sees (Task 3); labels for boxes it does not (Task 4). |
| 8 | New labels were to be picked "where the heads disagree most": my invention, and blind to the boxes the model confidently misses. | Picked by what is on the page: compass roses over the map, badges. Existing labels are checked for unmarked boxes first. |
| 9 | The temperature changes when it moves to folds, which changes the approved cuts. | Task 1 reports the test table beside the approved run's and names the sheets that changed. |
| 10 | Extras nobody asked for: a doubt picture, a side-by-side of two cuts, a paired bootstrap between them. | Removed. |
| 11 | Small badges may be below what the layout can resolve; more labels might not fix them. | Stated up front in Task 4, with a stop instead of a workaround. |
| 12 | The label work waited on the code work, though they are independent. | Task 4 runs alongside Tasks 1–3. |
