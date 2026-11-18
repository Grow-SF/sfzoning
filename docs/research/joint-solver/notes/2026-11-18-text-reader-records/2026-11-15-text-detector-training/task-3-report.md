# Task 3 report: scores on the validation area and the test drawings

Commit `6f6f4be` on `solver` (base cb12116). Status: done.

- **Tests:** full suite 1077 passed, 3 skipped. The 3 skips are the conftest's GPU-busy guard (another job held the
  GPU) and are unrelated. `tests/test_solver_textnet.py`: 34 passed (5 new).
- **Not committed:** `pipeline/dvc.lock`. `evaluate.py` is left in place, as the ledger ruled.

## What was built

### `textnet/scores.py` (new)

- **The entry point:** `score_sheets(chances, where, sheets=None) -> dict`.
  - `chances` maps (series, sheet) to the page's chance: uint8 ×255 as `predict.chance` writes it, or float 0..1,
    which is rounded the same way.
  - `where` is `"validation"` or `"whole"`.
  - `sheets` defaults to `data.load_sheets` of the maps named in `chances`, taken in `paths.TEXT_SHEETS` order. The
    load uses `hold_out_validation=False`; the hold-out flag doesn't affect scoring.
- **Output:** JSON-able, with no times or paths. The same chances give the same bytes (checked on real sheets).

  ```
  {"where", "called": 128, "long_line_ft": 200.0, "thin_line_ft": 12.0,
   "series": {series: S, ...},            # in the sheets' order
   "all": S}                              # pooled over all sheets
  S = {"sheets", "ink_px", "text_ink_px", "auc", "log_loss", "text_ink_found",
       "ink_outside_boxes_called_text", "called_text_in_a_box",
       "words": {kind: {"found", "of"}, ..., "all": {...}}, "words_found",
       "long_lines": {"ink_px", "called_text", "mean_chance", "in_a_box", "rest_called_text"}}
  ```

  - Every count is an integer, divided only at the end, so the order of the sheets doesn't matter.
  - A share of nothing is `null`, where evaluate gave 0.
- **Checks (ValueError):**
  - a sheet with px to score but no chance;
  - a chance for a sheet not among those scored;
  - a chance of the wrong shape;
  - a bad `where`.

  In `"validation"`, sheets with no px in the area are skipped and need no chance. In `"whole"`, every sheet needs
  one.
- **`scored_px(sd, where)`:** the validation px, computed `data.BAND` rows at a time, as Task 1's concern 6 asked.
  Task 4 can use it to predict only the sheets that touch the area.

### The rules

- **Which px:** ink px that are not unknown, using the training targets' own definitions through the new
  `data.page_text`.
  - Validation px are exactly the held-out px: those whose map point is in `VALIDATION_AREA`.
  - "Whole" means every px of the page, legend and margins included. Evaluate restricted itself to the warped
    footprint; this doesn't.
- **"Called text":** a chance ≥ 0.5, which is uint8 ≥ `CALLED` = 128. A float chance of 0.5 rounds to 128 and is
  called; 0.498 is not.
- **Words:** each Terra word is its own rectangle, filled as in training. Its ink is the ink px inside it, and ink
  where two words overlap counts for each.
  - A word is **found when the mean chance over its ink is ≥ CALLED**. This is `evaluate.py`'s rule, kept.
  - A word with no ink is not counted, as in evaluate.
- **Words straddling the validation edge (the rule carried over from Task 1):**
  - A word is counted in validation when its **centre px** is in the area. The centre px is the mean of its
    rectangle's corners, rounded.
  - It is judged on **its ink inside the area only**, so no px the detector trained on counts.
  - A word whose centre is outside is not counted, but its ink inside still counts as text ink in the pixel scores.
  - On "whole", every word counts and is judged on all its ink.
- **Long lines:** `long_lines(ink, ft_per_px)` takes the strokes of the whole page's ink (`strokes.strokes`).
  - A long line is a stroke at least `LONG_LINE_FT` = 200 ft long and at most `THIN_LINE_FT` = 12 ft wide (median),
    both in the sheet's own feet through `sd.ft_per_px`.
  - Each ink px belongs to its nearest centre-line px (`cv2.distanceTransformWithLabels`), within its own connected
    piece of ink. A speck whose centre line is no stroke is no line's.
  - Strokes are computed on the whole page, so a line crossing the area's edge is long by its whole length, and only
    its ink inside the area counts.
  - **Box ink is included.** `in_a_box` gives the share inside words, which is the floor a box-perfect detector hits.
    `rest_called_text` gives the share of the rest that is called text.
  - Unknown px are left out.

### Small changes outside the new file

- **`strokes.strokes(ink, ft_per_px=grid.FT_PER_PX, to_map=grid.to_map)`:** two optional parameters, so a sheet's
  page gives its own scale and keeps its points in page px.
  - The default output is **identical, array for array**, to HEAD on a real 2000 page (13,745 strokes) and on a
    3000×3000 window of the merged page (26,261 strokes).
  - solver-labels already reruns at the next build because of Task 1's `paths.py` change.
- **`data.page_text(sd)`:** returns the text and unknown px of a whole page, public so scores doesn't reach into
  `data._words` / `data._known`.
  - `_known` now uses a shared `_unknown` with the same arithmetic.
  - Task 1's crop-equality tests still pass.
- **`pipeline/README.md`:** the `textnet/` module-list line names `scores`. Task 4 rewrites the rest.

## Tests (`pipeline/tests/test_solver_textnet.py`)

**RED:** the module failed to import (`scores` didn't exist). **GREEN:** 5 new tests, all on in-memory `SheetData`
except the last.

- **`test_long_line_measure_uses_feet_per_sheet`** (Review Focus 2):
  - The same drawing in feet at 2 and at 4 ft/px. At both scales, ≥ 90% of the long thin line's ink is marked, and
    none of the short line, the thick line, a speck beside the long line, or the paper.
  - Read at the grid's fixed 4 ft/px, the 2 ft drawing's thin line is not found.
  - Through `score_sheets`, the long-line ink area in ft² is the same at both scales, within 10%.
- **`test_validation_scores_use_only_validation_pixels`:**
  - The scored ink px equal an independently drawn count: ink, in the area, not unknown. Text ink is checked the same
    way.
  - Changing the chance anywhere outside the area gives an identical result. Changing it on in-area ink changes the
    result.
  - A line of 1,040 ft whose 160 ft in the area alone would be short: its in-area ink is the long-line ink.
  - "Whole" counts every known ink px.
  - Each missing, extra, wrong-shape and bad-`where` case raises.
- **`test_a_word_across_the_validation_edge_counts_by_its_centre_and_only_its_ink_inside`:** this is the carried
  straddle rule, tested both ways round.
  - With one chance, word P is not found in validation (inside ink at 100) but found on the whole sheet (mean 136).
  - Word Q, whose centre is outside, is not counted, but its in-area ink is text ink.
- **`test_words_found_counts_a_box_with_half_its_ink_called_text`:**
  - A mean of exactly CALLED is found; CALLED−1 is not. The float chance is read as described above.
  - Half the ink at 255 and half at 1 is found. So is 40% at 255 with the rest at 100, which pins the mean rather than
    the share of px.
  - With 60% / 40% of the ink at 255 and the rest at 0, the word is found / not found.
  - Paper in the box and ink outside every box don't count.
  - Overlapping words are each judged on their own ink.
  - A word with no ink is not counted.
- **`test_scores_read_the_sheets_of_the_maps_scored_and_add_them_up`** (the `maps` fixture):
  - The default loading and the per-map order.
  - "All" equals the sum of the maps.
  - The JSON is byte-stable.
  - A map not in `TEXT_SHEETS` raises.

**Mutation check:** each of these 10 changes made the tests fail:

- a fixed grid scale;
- lines found on clipped ink;
- no speck check;
- words judged on all their ink in validation;
- words counted by any px inside;
- a share-of-px rule;
- unknown px kept;
- the scope ignored;
- float chances truncated;
- words with no ink counted.

Script: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/t3/mutate_scores.py`.

## Sanity check on real sheets (CPU)

A "perfect" chance: 255 on Terra's words' px, 0 elsewhere. It was run on the 2000 and 1960 sheets with the most
validation px.

| sheet | where | ink px | words found | text ink found | outside-box called | AUC | log loss | long-line ink px | long-line called (= in a box) |
|---|---|---|---|---|---|---|---|---|---|
| 2000 p07 (3.99 ft/px) | validation | 1.37M | 760/760 | 1.0 | 0.0 | 1.0 | 1e-4 | 32,308 | 2.96% |
| 2000 p07 | whole | 2.06M | 1105/1105 | 1.0 | 0.0 | 1.0 | 1e-4 | 81,818 | 2.14% |
| 1960 "7" (2.70 ft/px) | validation | 1.93M | 403/403 | 1.0 | 0.0 | 1.0 | 1e-4 | 802,241 | 3.49% |
| 1960 "7" | whole | 3.56M | 654/654 | 1.0 | 0.0 | 1.0 | 1e-4 | 1,417,008 | 3.18% |

- `rest_called_text` is 0 throughout.
- **The long-line floor:** about 2–3.5% of long-line ink lies inside Terra's boxes. A detector that called exactly
  the boxes would call that much. For context, today's 0.16% district-line figure left out ink near boxes.
- **Time per sheet:** 2.9 s for 2000 and 4.7 s for 1960, almost all of it `long_lines` (the strokes' Python loop).
- **Memory:** peak RSS 2.7 GB with all 87 training sheets loaded.
- Script and output:
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/t3/sanity.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/t3/sanity.json`

**Which training sheets touch the area:**

- 59 of the 87 training sheets have validation px: 53 have more than 1% of the page in the area, 41 more than 5%, and
  12 more than 20%.
- By map: 2000 9, 1980–90 8, 1990–96 8, 1960 9, 1964 12, 1968 13.
- Scoring them takes about 4 min of CPU per trial. Scoring the 92 test sheets on "whole" takes about 6–7 min.
- Four threads sped `long_lines` up only 1.34×, since the GIL is held by the strokes loop, so scoring stays
  sequential.
- The long-line mask doesn't depend on the chance. Task 4 or 5 could compute it once per sheet and reuse it across
  trials if the clock needs it.

## Concerns

1. **`THIN_LINE_FT = 12` against the strokes' width measure on the 2000 map. This needs the controller's call.**
   - The width is 2 × the distance from the centre px to the paper, so a 3-px line reads as 4 px. At 4 ft/px that is
     16 ft.
   - On 2000 p07, the ink on strokes ≥ 200 ft splits by median width as follows: ≤ 12 ft 5.7%, 12–16 ft **46%**,
     20–24 ft 20%, > 32 ft 21%.
   - On 1960 "7", 66% of that ink is 8–12 ft wide.
   - So on the modern-lettering maps the measure covers few lines: 4% of the page's ink against 40% on 1960.
   - 16 ft would take in the 2000 map's ordinary lines. The brief gives 12, so I kept it.
   - Script: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/t3/widths.py`.
2. **The words rule:** I kept evaluate's mean-chance rule. Read literally, "half its ink called text" would mean
   "at least half its px called", which differs on chances that aren't 0 or 1.
   - At exactly half the ink at 1 and half at 0, the mean is 127.5/255, so the word is not found.
   - The test pins the mean rule.
3. **Words are counted per box (rectangle),** where evaluate counted connected regions of box px. The counts aren't
   comparable with the quarter models' 96.6%, which the spec already reports as context only.
4. **"Whole" includes legends and margins.** There is no footprint mask, so the stage needs no footprint files.
   - Validation px whose map point lands in the area through a sheet's margin are included. They were held out of
     training too.
5. **Files beyond the brief's list:** `strokes.py` and `data.py`, both small changes that keep behaviour, and one
   README line.
   - `auc` and `log_loss` are duplicated in `scores.py` until Task 4 deletes `evaluate.py`.
   - The text-scores stage still runs `evaluate` until Task 4 rewires it.

## Fix (controller rulings): commit `e7550a4`

- **`THIN_LINE_FT = 48.0`** (it was 12). The constant's comment now says why:
  - the measure is there to catch district lines called text;
  - the 2000 map's district lines are heavy, 24–32 ft wide, and its ordinary lines about 16 ft;
  - letters already fall out by length, so the width limit only keeps out solid fills.
- **`line_ink(sd)`:** the page's long-line ink, kept for the rest of the process as packed bits.
  - It is keyed by series, sheet, shape, `ft_per_px`, a blake2b digest of the ink bits, and the two constants. So
    another page under the same name, or the same page at another scale, never gets this one's answer. The tests
    reuse sheet names, so name alone would not be safe.
  - `score_sheet` uses it. `long_lines(ink, ft_per_px)` stays as the plain function behind it.
- **Docstring:** the found-word rule is evaluate's (mean chance ≥ 0.5), not "half its px called". The quarter models'
  96.6% counted connected regions as words, so it isn't comparable.
- **Tests:**
  - **The drawing:** it gains a 40 ft "heavy" line, which counts, and a 64 ft solid fill, which doesn't.
  - **Wrong-scale check:** read at the grid's 4 ft/px, the 2 ft drawing's short line becomes long and its heavy line
    too wide. Both the length and the width follow the sheet's scale.
  - **New test, `test_a_sheets_long_lines_are_found_once_per_process`:** a call counter on `long_lines`.
    - Two chances, scored on both validation and whole, plus an equal copy of the sheet: the lines are found once.
    - The results are identical to scoring with an empty cache.
    - A changed page under the same name, and the same page at 2 ft/px, are each found afresh.
  - **Mutation check:** 12 of 12 caught, including "no cache" and "cache keyed by name only".
- **Test results:** `tests/test_solver_textnet.py` 35 passed. Full suite: 1078 passed, 3 skipped (the GPU-busy
  guard).

### Real data, rerun at 48 ft

Ink on strokes ≥ 200 ft, by median width, as the ink px `long_lines` assigns. Script:
`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/t3/bands.py`

| sheet | ≤12 ft | 12–16 | 16–32 | 32–48 | >48 | long-line ink at 48 ft, share of page ink |
|---|---|---|---|---|---|---|
| 2000 p07 (3.99 ft/px) | 6.5% | 39.9% | 29.9% | 13.0% | 10.7% | 52.7% |
| 1960 "7" (2.70 ft/px) | 62.4% | 2.1% | 21.4% | 9.8% | 4.4% | 61.0% |

**The perfect-chance sanity check again:**

- Words found are 760/760 and 1105/1105 on 2000 (validation and whole), and 403/403 and 654/654 on 1960. Text ink
  found is 1.0, outside-box 0.0, AUC 1.0 and log loss 1e-4.
- Long-line ink called text, equal to the share in a box:

  | sheet | validation | whole |
  |---|---|---|
  | 2000 | 6.9% of 697k px | 6.1% of 1.09M px |
  | 1960 | 4.2% of 1.21M px | 4.1% of 2.17M px |

  `rest_called_text` is 0.
- The floor for a box-perfect detector is now about 4–7%, since wider lines pass through more boxes.
- The cache works on real data: whole scoring of the same sheet after validation took 0.1–0.2 s, against 3–8 s the
  first time.
- Output: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/t3/sanity48.json`.
