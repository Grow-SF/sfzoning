### Task 1: Training data from many maps, held out by an area

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/data.py`
- Modify: `pipeline/src/sfzmaps/solver/paths.py` (the series and sheet lists)
- Create: `pipeline/src/sfzmaps/solver/textnet/boxes.py` (Terra boxes for any series, from the saved answers)
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Produces:**
  - **In `paths`:** `TEXT_TRAIN_SERIES`, `TEXT_TEST_SERIES` (tuples as in Global Constraints) and
    `TEXT_SHEETS: dict[str, tuple[str, ...]]`, the fixed sheet names Terra read per series.
  - **`boxes.terra_boxes(series, sheet) -> list[dict]`:** page-px boxes, decoded as `prints.terra_boxes` decodes
    2000. Reuse it; generalise it, don't copy it.
  - **In `data`:**
    - `VALIDATION_AREA = (x0, y0, x1, y1)` in map feet. Pick a block of about 10% of the mapped area that every
      training map covers, and say how in a comment.
    - `SheetData`, compact per sheet: ink (bool or packed), the box list, unknown regions, the held-out mask (or a
      way to compute it per crop), and `ft_per_px`.
    - `load_sheets(series_list, hold_out_validation: bool) -> list[SheetData]`.
    - `crop_targets(sheet, y, x, size, scale, turns) -> dict` draws the text / kind / angle / known targets for one
      crop from the boxes. No full-page target rasters.

- [ ] **Step 1: Write failing tests**, on synthetic sheets:
  - `test_terra_boxes_decode_for_another_series`: a 1960-style record with a page_box gives page-px boxes.
  - `test_validation_area_pixels_carry_no_loss_even_inside_a_crop_centred_outside` (Review Focus 1).
  - `test_crop_targets_equal_full_page_drawing` (the per-crop drawing equals drawing the whole page and cropping).
  - `test_sheet_data_holds_no_full_page_target_rasters`: per-sheet bytes ≤ 1.2 × the ink bytes plus the boxes
    (Review Focus 3).
  - `test_text_sheet_lists_match_the_saved_answers`: built from a tmp saved-answers file, not real data.
- [ ] **Step 2:** Implement; the tests pass. Keep the 2000-only path working until Task 4 removes the quarter
  stages.
- [ ] **Step 3:** Real check, CPU only: `load_sheets(TEXT_TRAIN_SERIES, True)`. Report the seconds, the peak memory,
  the sheets per series, and the validation share of the text pixels per series.
- [ ] **Step 4:** Full suite. Commit.

