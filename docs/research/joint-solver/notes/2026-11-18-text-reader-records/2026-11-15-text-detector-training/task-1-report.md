# Task 1 report: training data from many maps, held out by an area

Commit `b5abd6d` on `solver` (base 8f0103b). Status: done. Full suite 1068 passed.

## What was built

- **`paths`:**
  - `TEXT_TRAIN_SERIES` and `TEXT_TEST_SERIES`, as in Global Constraints.
  - `TEXT_SHEETS`: the fixed sheet names Terra read, per series, in `boxes.sheet_order` (numbers in a name compare as
    numbers). Counts: 13/13/12/13/16/20 train and 32/30/30 test.
  - Terra's `sheet` field is the data/work and data/warp folder name in every series. All 179 sheets have
    `page.png`, `district_labels.json` and `transform.json`.
- **`textnet/boxes.py`:**
  - `terra_boxes(series, sheet, records=None)`: the decoding from `prints.terra_boxes`, moved here and filtered by
    series as well as sheet. `prints.terra_boxes(sheet)` now calls it with `paths.SERIES`, so there is one
    implementation.
  - The two pairwise loops are vectorised with the same arithmetic. On all 179 sheets (101,221 boxes) the output
    equals the original pure-Python code exactly; time went from 56 s to 10 s.
  - `TextItem`/`TextItems` (unchanged, no docstrings) and `iou` moved here too.
  - `text_sheets(records=None)` and `check_text_sheets(records=None)`: the check that the constants equal the saved
    answers. It names the series and sheets that differ. `load_sheets` calls it.
- **`textnet/data.py`:**
  - `VALIDATION_AREA` (see below).
  - `SheetData`: series, sheet, height, width, `ink_bits`, `words` (int32 (n,4,2) rounded rectangle corners),
    `kinds`, `angles`, `other` (the other passes' rectangles, 2000 only), `points` (Sol's points), `world` (placement
    samples every 32 px, float64), `ft_per_px` (`median_scale_ft_per_px`), `hold_out`. `BOX_FIELDS` names the box
    arrays.
  - `load_sheets(series_list, hold_out_validation)`: 8 threads, deterministic order (series, then `TEXT_SHEETS`
    order).
  - `crop_targets(sd, y, x, size, scale, turns)`: keys and dtypes as `train.crop` (ink, text, kind, angle, known),
    drawn from the boxes for that crop only.
  - `centres(sd, stride) -> (on_text, on_ink)`: (row, col) int32 at stride points, never held out, drawn band by
    band.
  - `in_validation(sd, y0, x0, h, w)`: for Task 3's scores.
  - `page_ink(sd)`.
  - `page_world` now shares `placement_samples` / `_world_at` with the per-window path and gives the same numbers.
- **The 2000 quarter path is unchanged:** `SheetTargets`, `targets`, `sheet_targets`, `page_quarters`,
  `terra_boxes(sheet)` from `terra_boxes.json`, and train/predict/evaluate.

### Drawing per crop equals drawing the whole page

- `cv2.fillPoly` is not shift-invariant where the raster edge clips a polygon. Measured: 398 of 5,000 random word
  rectangles differed between a clipped window and the full page. Unclipped it is exact: 0 of 5,000.
- So `_paint` fills on a canvas holding the window plus every shape that reaches it. This is the page as filled on an
  unbounded page.
- The only difference from the old full-page targets is for a box that crosses the page's own edge.
- On real 2000 sheets, 600 crops (320 px, three sheets) equal the old `data.targets` path on ink, text, kind, angle
  and unknown: 0 of 600 differ.

### Held-out mask

- It is not stored. Each crop interpolates the stored placement samples bilinearly, with the same arithmetic as
  `page_world`, so a window gets exactly the full page's numbers.
- A window whose samples all lie on one side of an edge of the area (by a 1 ft margin) is answered without
  per-pixel work.

## VALIDATION_AREA, and why

`(5983192, 2089227, 5999918, 2097590)` in EPSG:2227 ft: the city's south-west (Parkside, Lakeshore, Ingleside, Ocean
Avenue), 16,726 × 8,363 ft = 5.0 sq mi.

- **The rule (also in the code comment):**
  - The rectangle covers 10% of the training maps' mapped area: the union of their Terra sheets' warped footprints
    is 50.2 sq mi.
  - It lies inside every training map's sheets: at least 99.5% inside their intersection, which is 38.7 sq mi.
  - Searched over corners on a 250 ft grid, with aspect ratios from 1:2 to 2:1.
  - The score is how close each training map's share of Terra box area in the rectangle comes to 10%: in all and, at
    half weight, for each kind the map prints (5% of its boxes or more; the older maps print no block numbers).
- **The winner:** score 0.022, with 8.7–12.2% of each training map's box area and 6–15% of each kind.
- **Runner-up:** the central Mission / Noe / Glen Park strip `(6000442, 2091727, 6008805, 2108453)`, score 0.030.
- Scripts: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/coverage.py`
  and `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/choose_area.py`.

## Ink storage

Packed bits (`np.packbits` along rows, uint8 of shape (h, ceil(w/8))).

- **Why:** 1 bit per px instead of 1 byte. The 87 training sheets are 2.34 Gpx: 0.293 GB packed against 2.34 GB as
  bool.
- **Cost:** a crop unpacks only its own rows and bytes, which is cheap.
- **Whole page:** `page_ink(sd)` unpacks the whole page for prediction and scoring.

## Tests (pipeline/tests/test_solver_textnet.py)

- **RED:** the file failed to import (`cannot import name 'boxes'`). Before the implementation, none of the
  interfaces the new tests use existed: `boxes.*`, `load_sheets`, `crop_targets`, `centres`, `VALIDATION_AREA`,
  `page_ink`, `BOX_FIELDS`.
- **GREEN:** 22 passed in the file (16 old, 6 new). The full suite gives 1068 passed.
- **The new tests:**
  - `test_terra_boxes_decode_for_another_series`: 1960-style tiles. Page-px offsets; the same word read twice is kept
    from the tile it is farther inside; another map's sheet of the same name is ignored.
  - `test_validation_area_pixels_carry_no_loss_even_inside_a_crop_centred_outside` (Review Focus 1): a crop centred
    outside the area overlaps it, and the overlapping px are not known. Flipping their targets leaves the loss and
    every gradient unchanged. `centres` never lands in the area; without the hold-out it does.
  - `test_crop_targets_equal_full_page_drawing`: two sheets (2000 at 4 ft/px with other passes, 1960 at 2 ft/px),
    8 centres including page corners and area edges, 4 sizes and scales, all turns. Every key equals an independent
    whole-page reference: plain cv2 on a padded page, `page_world` hold-out, then crop, resize and turn.
  - `test_sheet_data_holds_no_full_page_target_rasters` (Review Focus 3): every field is an array or a scalar; no
    array is a view of something bigger; total bytes ≤ 1.2 × ink bytes + box bytes; `page_ink` equals the page's ink.
  - `test_each_map_is_read_at_its_own_scale_with_its_own_unknowns`:
    - `ft_per_px` is 4 and 2.
    - Other-pass boxes count as unknown on 2000 only. A same-named 1960 sheet is untouched.
    - A `TEXT_SHEETS` entry the answers don't have stops `load_sheets`.
  - `test_text_sheet_lists_match_the_saved_answers`: built from a temporary saved-answers file. It checks the rule,
    including "2" before "10". The check raises on a missing sheet, a wrong order, an extra map and a left-out map.
    The real constants list train then test, are disjoint, and are in rule order.

## Real check (CPU only, no torch)

`load_sheets(TEXT_TRAIN_SERIES, True)`:

- **Time:** 87 sheets in **26.3 s**. It was 43.6 s before the decode was vectorised.
- **Peak RSS:** **2.03 GB**, mostly transient: 8 pages being decoded at once.
- **Held:** **0.333 GB** of arrays: ink bits 0.293 GB, placement samples 37 MB, boxes 2.2 MB.

| series | sheets | words | text px | validation share of text px | of ink px | centres (stride 4) text / ink | ft/px |
|---|---|---|---|---|---|---|---|
| 2000-digital-zoning | 13 | 10,036 | 14.7M | 0.107 | 0.092 | 818,871 / 1,218,181 | 3.98–4.04 |
| 1980-1990-zoning | 13 | 10,463 | 13.5M | 0.102 | 0.088 | 757,800 / 1,133,364 | 3.37–4.01 |
| 1990-1996-zoning | 12 | 8,661 | 13.1M | 0.082 | 0.088 | 747,667 / 1,196,232 | 3.92–3.98 |
| 1960 | 13 | 5,988 | 22.4M | 0.108 | 0.092 | 1,245,494 / 2,123,048 | 2.70–2.73 |
| 1964 | 16 | 8,036 | 27.8M | 0.088 | 0.077 | 1,585,881 / 2,483,752 | 2.70–2.75 |
| 1968 | 20 | 9,034 | 35.0M | 0.102 | 0.085 | 1,962,936 / 3,215,721 | 2.67–2.72 |

- **All series:** the validation share of text px is 0.098.
- **Other timings:**
  - Finding the centres of all 87 sheets takes 4.3 s.
  - `crop_targets` at 320 px with random scale and turns takes **0.72 ms a crop**, so 8.6 ms for a batch of 12, on
    one CPU thread.
- **Scripts:**
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/real_check.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/crop_speed.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/decode_same.py`

## Files changed

- `pipeline/src/sfzmaps/solver/paths.py`: the lists.
- `pipeline/src/sfzmaps/solver/textnet/boxes.py`: new.
- `pipeline/src/sfzmaps/solver/textnet/data.py`
- `pipeline/tests/test_solver_textnet.py`
- `pipeline/src/sfzmaps/solver/prints.py`: delegates to `boxes` (outside the brief's list; see concern 1).
- `pipeline/src/sfzmaps/solver/saved.py`: `sol_labels(sheet, series=paths.SERIES)`.
- `pipeline/dvc.yaml`: regenerated.
- `pipeline/dvc.lock` is not committed.

## Concerns

1. **`prints.py` and `saved.py` changed, beyond the brief's file list.**
   - "Reuse it, don't copy it" could only be met by moving the decoding into `boxes.py`.
   - Having `boxes` import `prints` would have pulled `prints`, `llm`, `lots`, `page` and `codes` into the detector's
     dependencies. The existing deps test forbids that, and it would retrain on every prints change.
   - `prints`' behaviour is unchanged: its tests pass, and the output is identical on all sheets.
2. **dvc.yaml was regenerated by `build.dvc_yaml_path().write_text(build.dvc_yaml())`,** which is what
   `--write-dvc-yaml` does, without running `sfzmaps build`.
   - Seven stages gain `textnet/boxes.py`: solver-prints, the four solver-text-{q} stages, solver-text-scores and
     solver-walls.
   - Those stages, and all stages importing `paths.py`, `saved.py` or `data.py`, will rerun at the next build on
     their code hash. Their outputs should be byte-identical.
3. **The validation area is the residential south-west.** It is what the stated rule picks: the closest to 10% per
   map and per kind. If the owner prefers denser lettering in validation, the runner-up central strip scores 0.030;
   swapping it is a one-line change.
4. **Drawing per crop costs 0.72 ms a crop,** against slicing precomputed rasters before. Task 2's prefetch thread
   should hide it. Precomputing each shape's bounds would trim about another 20% if needed.
5. **The pools are large:** 7.1M text and 11.4M ink centres at stride 4. That is about 150 MB as (row, col) int32, or
   about 445 MB if Task 2 keeps the old (sheet, row, col) int64 format. Task 2 should keep them compact.
6. **`in_validation` on a whole page builds a float64 map-point raster** (16 B/px, about 0.55 GB for a 1960 page).
   Task 3's scores should call it band by band, as `centres` does.
7. **One 1980-1990-zoning sheet is at 3.37 ft/px;** the others are about 4. That is real, from its placement. Task 3's
   per-sheet conversion of feet to px covers it.
8. **Sheets touching the area are trained only outside it.** The area holds out the same map region on every map,
   which the spec asks for.

## Fix (controller ruling): the validation area is now the central strip

Commit `78c74c1`. `VALIDATION_AREA = (6000442, 2091727, 6008805, 2108453)`: the central strip through the Mission,
Noe Valley and Glen Park. It replaces the south-west given above.

- **Why:** the rule's best rectangle, the residential south-west, has sparse, simple lettering. An easy area can't
  tell training recipes apart, and the owner's rule is to score on dense areas, never only the simple west.
- **The code comment** keeps the selection rule and gives this reason. The strip is the rule's runner-up (score
  0.030).
- **Coverage:** 5.02 sq mi, 10.0% of the training maps' union. 99.69% of it is inside the training maps'
  intersection, so the ≥ 99.5% rule holds. It is 99.8% inside each of the three modern scans and 100% inside the
  1960 / 1964 / 1968 maps.
- **Real check rerun** (CPU): load 25.7 s, peak RSS 2.14 GB, 0.333 GB held. Per-crop drawing still matches the old
  2000 path: 0 of 600 crops differ.

Validation share of text px, per series and by kind (zoning code / block number / street name / other; "–" where the
map prints too little of that kind):

| series | all text | zoning code | block number | street name | other |
|---|---|---|---|---|---|
| 2000-digital-zoning | 0.117 | 0.137 | 0.149 | 0.148 | 0.081 |
| 1980-1990-zoning | 0.107 | 0.123 | 0.127 | 0.120 | 0.059 |
| 1990-1996-zoning | 0.123 | 0.144 | 0.160 | 0.134 | 0.090 |
| 1960 | 0.121 | 0.149 | – | 0.139 | 0.093 |
| 1964 | 0.110 | 0.148 | – | 0.110 | 0.095 |
| 1968 | 0.101 | 0.147 | – | 0.091 | 0.092 |

- **Overall:** 0.111 of text px. The validation share of ink px ranges from 0.092 to 0.137.
- **Kinds:** zoning codes and block numbers are slightly over-represented (12–16%). "Other" text is
  under-represented in 1980-1990 (5.9%).
- **Tests:** `tests/test_solver_textnet.py` 22 passed; full suite 1068 passed.
