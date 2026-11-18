# Task 4 report: the control's label rule ("lines-unknown")

Status: DONE_WITH_CONCERNS. Commit `97dbfdc` on `solver` (base 837c994).

## What I built

- **`textnet/data.py`** (no new import):
  - `SheetData.line_bits` defaults to an empty `uint8 (0, 0)` array (`dataclasses.field(default_factory=...)`).
  - `LABEL_RULES = ("boxes", "lines-unknown")`.
  - `crop_targets(..., labels="boxes")`. Under `"lines-unknown"` it sets `known &= ~((word > 0) & line_window(...))`
    at page resolution, before the resize and the turn.
    - An unknown rule raises `ValueError`.
    - Empty `line_bits` (or a shape other than `ink_bits`') raise `ValueError` that names the sheet.
  - `_ink_window` now calls `_bits_window(packed, ...)` (same arithmetic); public `line_window(sd, ...)` is added.
  - The "boxes" path computes exactly what it did before.
- **`textnet/train.py`** (no new import):
  - `Recipe.labels: str = "boxes"` (last field) and `CHOICES["labels"] = data.LABEL_RULES`.
  - `draw` passes `labels=recipe.labels`.
  - Run name at the default is unchanged (`name()` lists only fields that differ from the defaults).
  - `Recipe(labels="lines-unknown").name() == "labels=lines-unknown"`.
- **`textnet/zones.py`** (new; imported by no stage):
  - `matched_points(sd)`: vectorised point-to-rotated-rectangle distance, chunked by 256 points. A point matches
    when it is inside a word or within half that word's height of it (height = the second edge). Matching is per
    sheet only.
  - `zone_radius(sheets, areas)` returns `{"rule": "q99/2", "px": round(q99/2), "q50", "q90", "q99", "words"}`.
    It uses the same inputs as `reader_stage.measure_radius`: code-word length is the first edge, and a word is left
    out when its rounded centre is in an area. Its quantiles match measure_radius's on the real data (85.0 / 114.4 /
    169.0, 6197 words).
  - `prepare(sheets, areas, cache, threads=1)` returns, per sheet,
    `replace(sd, points=sd.points[~matched], point_radius=zone_radius(...)["px"], line_bits=...)`.
    - `line_bits = packbits(dilate(linerules.line_px(page_ink, ft_per_px), 3x3))`.
    - The cache is `<cache>/<series>/<sheet>.npz`, written atomically and compressed.
    - The cache key is the sha256 of linerules.py (`rule_key()`) plus `GROW_PX`, `run_px` and the sha256 of
      `ink_bits`. The page and run-length parts are additions to the brief, so a changed page or scale is never
      served stale lines.
    - A sheet without its scale (`ft_per_px` nan, i.e. read with `placement=False`) raises a ValueError naming it.
    - `threads` is an optional extra (default 1). `_unknown` is unchanged.
  - `label_shares(sheets)` gives per series × kind (plus "all") `{ink, unknown, share}`. In-box ink is drawn by
    `_words`/`_paint` in BAND rows; each px takes the kind of the word drawn last, as the detector learns it.
  - Additions: `point_counts`, `write_rule` (atomic JSON), and `main()`
    (`uv run python -m sfzmaps.solver.textnet.zones`), which writes `experiments/textlines/labels_rule.json`.
    `reader_stage` is imported only inside `main`.
- **Tests:**
  - New file `tests/test_textnet_labels_rule.py`, 14 tests:
    - the six from the brief;
    - the half-height boundary on a 30° word;
    - lines = run rule grown by 1 and packed like the ink;
    - a missing scale is an error;
    - every sheet gets the measured radius;
    - the zone_radius values, with an area left out;
    - the cache is reused, and found again when the rule or the page changes;
    - `label_shares` counts;
    - filled `line_bits` never change a "boxes" batch, and `draw` passes the recipe's rule.
  - `tests/test_solver_textnet.py`, changed on purpose:
    - `test_default_recipe_trains_as_today` gets `"labels": "boxes"`;
    - `test_sheet_data_holds_no_full_page_target_rasters` pops `line_bits` and asserts it is the empty `uint8 (0, 0)`
      array.
  - **One unlisted change, mechanical:** in `test_reader_detector_never_trains_on_its_quarter_or_held_blocks`, the
    lambda that monkeypatches `crop_targets` now takes and forwards `**k`. It took no keywords, and `draw` now passes
    `labels=`. What the test asserts is unchanged.

## TDD evidence

- **RED** (before any source change):
  `uv run pytest -q tests/test_textnet_labels_rule.py tests/test_solver_textnet.py -k "labels_rule or default_recipe or sheet_data_holds"`
  ```
  FAILED tests/test_solver_textnet.py::test_sheet_data_holds_no_full_page_target_rasters
  FAILED tests/test_solver_textnet.py::test_default_recipe_trains_as_today   (Right contains 1 more item: {'labels': 'boxes'})
  ERROR tests/test_textnet_labels_rule.py - ImportError (no textnet.zones)
  2 failed, 1 error
  ```
- **First GREEN attempt:** 15 passed, 1 failed. In `test_every_sheet_gets_the_measured_radius` the fixture's radius
  happened to be 30 = READ_POINT_RADIUS. I added a 200 px code word so the test tells the two apart.
- **Test command** (`test_solver_textnet`, `test_codereader`, `test_solver_textread`, `test_build`, `test_textnet_*`,
  `test_textlines_*`): **192 passed, 3 xfailed**. That includes `tests/test_build.py`, so no closure changed.
- **Full suite** `uv run pytest -q`: **1500 passed, 3 xfailed, 1 warning** (pre-existing shapely warning).

## Proof that "boxes" is unchanged

The guards are byte-for-byte the same in the diff, and both pass under both digest keys:
```
# xdist workers (OMP/VECLIB = "1": GUARD_WEIGHTS[("Darwin","arm64","1","1")])
PASSED tests/test_solver_textnet.py::test_crop_targets_equal_full_page_drawing
PASSED tests/test_solver_textnet.py::test_the_stages_recipe_trains_the_weights_recorded
# one plain process, thread env unset (GUARD_WEIGHTS[("Darwin","arm64",None,None)])
tests/test_solver_textnet.py::test_crop_targets_equal_full_page_drawing PASSED
tests/test_solver_textnet.py::test_the_stages_recipe_trains_the_weights_recorded PASSED
```
- `test_the_boxes_rule_and_its_run_name_are_unchanged` pins two names:
  - `Recipe().name() == "default"`;
  - `stage.DEFAULT_RECIPE.name() == "half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1"`.

  Both values are from 837c994, before the change.
- `test_lines_on_a_sheet_never_change_the_boxes_rules_batches`: `train.draw` with the same seed gives byte-identical
  batches whether or not the sheet's `line_bits` are filled.
- Expected side effects:
  - Saved `model.pt` and `held.json` gain the `"labels": "boxes"` key the next time they are built.
  - The six code-dependent stages show stale; I did not build them and did not touch dvc.yaml or dvc.lock.

## Real data: `data/solver/experiments/textlines/labels_rule.json` (87 training sheets, not committed)

Run: `OMP_NUM_THREADS=1 uv run python -m sfzmaps.solver.textnet.zones`. CPU only, 4 line threads with
`cv2.setNumThreads(1)`. Load average stayed about 5–8. $0.

### Zone radius

The answered blocks are left out, as `measure_radius` does at the stage.

| | q50 | q90 | q99 | px | words |
|---|---|---|---|---|---|
| zone_radius (blocks out) | 85.0 | 114.4 | 169.0 | **84** | 6197 |
| + NW held | 85.0 | 116.0 | 166.0 | 83 | 4749 |
| + NE held | 86.0 | 115.0 | 166.41 | 83 | 4160 |
| + SW held | 84.0 | 111.9 | 164.69 | 82 | 5032 |
| + SE held | 85.0 | 116.0 | 172.51 | 86 | 4650 |

Today's unknown radius (`measure_radius`, the maximum) is **328 px**. It is set by Terra's word "SH" on 1968 p18,
328 px long. So the zones' radius falls from 328 to 84. Task 5 (NE held) will get 83 px.

### Sol's points

Of 8246 points, 8069 are matched (97.9%), so **177 keep a zone**:

| Series | Zones / points |
|---|---|
| 2000 | 58 / 1653 |
| 1980-90 | 38 / 1671 |
| 1990-96 | 62 / 1477 |
| 1960 | 2 / 1035 |
| 1964 | 7 / 1191 |
| 1968 | 10 / 1219 |

### Shares of in-box ink made unknown, per series × kind

Each cell is ink px made unknown / in-box ink px = share.

| series | zoning code | block number | street name | other | all |
|---|---|---|---|---|---|
| 2000-digital-zoning | 509,396 / 1,966,893 = **0.259** | 81,839 / 874,527 = 0.094 | 182,237 / 737,112 = 0.247 | 545,446 / 1,861,543 = 0.293 | 1,318,918 / 5,440,075 = **0.242** |
| 1980-1990-zoning | 431,813 / 1,967,685 = **0.219** | 107,492 / 1,109,027 = 0.097 | 223,666 / 978,869 = 0.228 | 253,348 / 1,093,499 = 0.232 | 1,016,319 / 5,149,080 = **0.197** |
| 1990-1996-zoning | 513,074 / 1,918,719 = **0.267** | 67,792 / 809,413 = 0.084 | 149,255 / 643,244 = 0.232 | 538,267 / 1,729,303 = 0.311 | 1,268,388 / 5,100,679 = **0.249** |
| 1960 | 212,802 / 1,720,916 = **0.124** | 0 / 137 = 0.000 | 222,525 / 2,392,928 = 0.093 | 724,310 / 2,358,684 = 0.307 | 1,159,637 / 6,472,665 = **0.179** |
| 1964 | 184,677 / 1,579,721 = **0.117** | 0 / 39 = 0.000 | 178,176 / 2,632,614 = 0.068 | 1,162,078 / 3,409,547 = 0.341 | 1,524,931 / 7,621,921 = **0.200** |
| 1968 | 324,954 / 1,901,149 = **0.171** | 96 / 129 = 0.744 | 352,364 / 3,660,224 = 0.096 | 1,762,267 / 4,751,949 = 0.371 | 2,439,681 / 10,313,451 = **0.237** |

Over all maps: 8,727,874 of 40,097,871 in-box ink px, **0.218**.

### Cost of `prepare` on 87 sheets

| Step | Wall time |
|---|---|
| Load (`load_sheets`, 8 loader threads) | 26.9 s |
| Lines, cold, 4 threads | **47.6 s** |
| Lines, read back from the cache, 1 thread | **0.7 s** |
| `label_shares` | 2.3 s |

- The cold lines' single-thread sum is 178.1 s (about 3.0 min). Per sheet:
  - 2000: 1.02 s
  - 1980-90: 1.09 s
  - 1990-96: 1.03 s
  - 1960: 2.85 s
  - 1964: 2.80 s
  - 1968: 2.82 s
- Memory: `line_bits` take 293 MB in memory and 16 MB on disk (87 compressed .npz under
  `experiments/textlines/lines/`). Peak RSS 3.6 GB.
- Whole command: 82 s wall.

## Concerns

1. **The run rule calls the letters of big or bold 2000-era labels "line", so those letters lose their lesson.**
   I looked at the pictures, which are not committed:
   - `data/solver/experiments/textlines/labels_rule_examples.png` (what for: what the rule makes unknown on
     typical and top-share code words, 2000 p05 and 1968 p10);
   - `data/solver/experiments/textlines/labels_rule_high_share.png` (what for: a seeded random 24 of the 260 code
     words on 2000 with share ≥ 0.5).

   What they show:
   - **Typical words** (share ~0.2) lose only the line running along or through the box. This is right.
   - **High-share words:** roughly half are real lines through the word. The other half are the letters themselves:
     - big P, M and I strokes longer than run_px = 50 px;
     - bold letters merged with the hyphen into a horizontal run (RH-2, RM-2 entirely red).

   Per-word shares of code words, measured on all 87 sheets:

   | Series | Median | Words ≥ 0.5 | Their part of the unknown ink | Words at 0 |
   |---|---|---|---|---|
   | 2000 | 0.200 | 12.1% | 35.7% | 36.7% |
   | 1990-96 | 0.205 | 13.9% | 39.9% | — |
   | 1980-90 | 0.185 | 5.2% | 15.0% | — |
   | 1960s | ≤ 0.016 | ≤ 1.3% | — | — |

   On the exact-truth cases the rule calls 0.7% (putText) / 1.6% (font.npz) of glyph ink line, max 11%
   (`rules.json`). Real bold labels are worse.
   - The rule only makes these px unknown, never not-text, so it removes lessons and teaches nothing wrong.
   - But the control may learn less about the 2000 map's biggest labels. Task 5's M2 (and M2 by kind) should show
     it if so.
   - The fix is not mine to make here: line_px is fixed by Task 1, and the measures say never to define line ink by
     a training labeller. The controller and owner should know before reading Task 5's numbers.
2. **Matching takes any Terra word, as the brief says, not only code words.** Of 8069 matched points, 113 (1.4%)
   are matched only by a word of another kind, for example a street name beside the label:
   - 2000: 21
   - 1980-90: 16
   - 1990-96: 13
   - 1960: 1
   - 1964: 22
   - 1968: 40

   Those points lose their zone, though Terra may have missed their code word. (Scratch count, not committed.)
3. **One existing test changed beyond the brief's two:** the monkeypatch lambda in
   `test_reader_detector_never_trains_on_its_quarter_or_held_blocks` now forwards keywords. The two guards are
   untouched.
4. **"In-box" uses the kind of the word drawn last** on overlapping boxes, as the detector learns it. The
   "block number" cells on 1960–68 are tiny (39–137 px), so 1968's 0.744 there means nothing.
5. **Optional extras beyond the brief's interface:** `prepare(..., threads=1)`; `zones.main` / `write_rule` /
   `point_counts`; `data.line_window`; and the cache key includes the ink's hash and `run_px`.
6. **Stale stages as expected:** `solver-text`, `-text-scores`, `-text-reader`, `-boxes`, `-text-crops` and
   `-text-read`. Not built.
