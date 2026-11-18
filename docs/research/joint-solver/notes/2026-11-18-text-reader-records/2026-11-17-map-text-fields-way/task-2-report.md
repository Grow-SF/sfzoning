# Task 2 report: the Mission Street review page

Commit d627640 (branch solver):
- `pipeline/src/sfzmaps/solver/wordread/review.py`
- `pipeline/scripts/wordread/review_page.py`
- `pipeline/tests/test_wordread_review.py`

## What I built

`wordread/review.py`:
- `mission_sheet(streets=None, footprints=None)` and `mission_lengths`. The centreline loader is `streetside.load_streets`
  (DataSF `data/ref/streets.parquet`, EPSG:2227). The walls' STREET_CROSSING doesn't load centrelines; this is the
  solver's only centreline loader. Each sheet's length is the MISSION ST line inside the warp's
  `warped.footprint.geojson`.
- `Components` and `components(chance, ink, level)`. They group the px exactly as `wordboxes.group` does: chance at or
  above the level, within INK_R of ink, 8-connected. Each component's length is measured along its chance-weighted
  major axis.
- `miss_reason(word_rect, chance, ink, components, level, *, on_map=True, reading=None) -> (reason, details)`. The
  reasons are checked in this order:
  1. outside_map
  2. below_level: under 50% of the word's ink is at or above the level, or it has no ink.
  3. merged_into_line: the main component is more than 3× the word's length.
  4. split: two or more components each hold at least 10% of its above-level ink.
  5. wrong_size

  The details hold:
  - the highest and the mean chance on its ink;
  - `share_above` and `ink_px`;
  - the word's length and height;
  - the components it meets, and the main one's length (`component_px`), ink px and share;
  - the reading at Terra's box.

  `miss_reasons` adds `shares_component_with`: the other Terra words whose ink sits mostly in that component.
- `colour(ours, terra, forms=None)`. Both strings are normalised with `crops.norm_text`. A printed form of the code
  list counts as its code.
- `page_data(sheet, boxes, reads, terra, chance, before, *, ink, levels, on_map, terra_reads, answers, forms, models)`:
  - Matching follows Task 1's rule, at IoU 0.5 on the page.
  - The panels use `measures.box_numbers` (with `miss_reason` as the `reason_fn`), `read_numbers` and `headline`.
  - It also returns `delta = measures.compare(numbers, before)`.
- `render(data, out)`. It writes one HTML file with no `<html>`, `<head>` or `<body>`:
  - `<title>Mission Sheet Review</title>`, the viewport-fit meta, and colour tokens on `:root` with both dark-mode
    blocks.
  - The image is 6 palette-PNG tiles (2048 px each) as data URIs, with `image-rendering: pixelated`.
  - One SVG overlay with `vector-effect: non-scaling-stroke`.
  - Pan and zoom use a CSS transform: pointer drag, pinch, wheel, double-click, and +/−/Fit buttons.
  - The readings show from 0.8×. The tags and the miss details show from 2×.
  - Tapping hit-tests the polygons in JS and lists every box and missed word under the finger.
  - A `#view=x,y,s&tap=x,y` hash opens a given view (I used it for the screenshots).
- The shading is painted into the palette under the ink, never over it. Paper inside green boxes and missed Terra
  words is tinted; ink stays grey or red. The overlay's paths all have `fill="none"`.
- `red_px_of`, `page_image` and `red_px_after_overlay` read the page back. They decode the tiles and composite any
  filled SVG polygon, rect or path at its fill and opacity.
- Loaders:
  - `reads_for_page`: Task 1's reads, plus the best paths, p(180°) and the open hypotheses from `reads.npz`.
  - `terra_on_sheet`
  - `terra_box_reads`: the fold reader of the word's quarter reads a `crops.rectify` crop. It runs on the CPU, one
    thread, deterministic, weighted by the 0/180 classifier, as `textread.read` reads at Terra's box.
  - `on_the_map`: the same footprint and mapped-area rule as `prints._place`.

`scripts/wordread/review_page.py`:
- It builds the page for the Mission sheet from the baseline (`data/solver/boxes`, `text-read`, `text-reader` and
  `text-train`), or from a run given with `--run/--boxes/--reads/--levels/--chance/--readers`.
- It writes to `data/solver/experiments/wordread/review/<run>/`: `review.html`, `numbers.json` and `missed.json`.
- It keeps the build order in `review/runs.json`. "Before" is the previous run's `numbers.json`, or the run named with
  `--before`.

What the page shows, per the controller's calls:
- Our unmatched boxes read as "none" are thin grey outlines. Those read as text are shaded green, with their reading in
  black.
- A box read as a code shows its top closed code. Any other box shows the fold reader's open best path in the
  classifier's direction.
- Every box has a reading-direction arrow and a tag: kind · mean chance · angle.
- A matched box's reading is green or purple. A purple one has "(Terra: …)" after it in red.
- Missed Terra words are yellow, dashed and shaded. They carry "Terra: X" in red and a reason line, e.g. "wrong size ·
  max 1.00 · 100% ≥ 0.99 / with 3596 · reader: NC-3 0.99".
- The panels follow the mockup: Boxes, Text read, Legend, plus a "why we missed" table.
- A line names the models that drew the page, with paths relative to `pipeline/`.

## TDD evidence

- **RED:** `uv run pytest -q tests/test_wordread_review.py` → ImportError (no `review`).
- **GREEN:** 15 passed. That is the brief's four, verbatim, plus:
  - colours: normalised strings, a printed form counts as its code, "" is purple;
  - one test per remaining reason: split, wrong_size, and outside_map with the reading kept;
  - a bare label image gives the same reason as `Components`;
  - a word merged with its neighbour names it, at its quarter's level;
  - `mission_sheet` on synthetic streets and footprints;
  - `page_data`'s sorting and numbers (matched/green, text, none, missed with a reason, IoU 0.5 counts);
  - the shading tints paper green and yellow while keeping text ink red and other ink grey;
  - the "vs last run" delta;
  - the page is self-contained and meets the Artifact contract.
- **Mutation check (scratch, not committed).** Each of these makes `red_px_after_overlay` give 243 of 324 instead of
  324, so the brief's test catches both:
  - painting the green shade over the ink in the palette;
  - adding a translucent filled SVG polygon over the ink.
- `-W error`: the new tests raise no warnings.
- `tests/test_build.py`: 43 passed.
- Full `uv run pytest -q`: 1568 passed, 3 xfailed.
- `pgrep -fl 'sfzmaps (build|stage)'` was empty before writing under `src` and before the commit.
- ruff (uvx, line length 120) is clean on `review.py` and the test.
- **Consistency check (scratch).** I re-read all 995 boxes on the sheet with the fold readers on the CPU, one thread.
  `best_0`, `best_180` and `p_classifier` equal `text-read/reads.parquet` for 995/995. So the stored best paths the
  page shows are the fold readers' CPU best paths.

## The Mission sheet

`mission_sheet()` picks **2000-digital-zoning-p07**, which holds 16,117 ft of DataSF's MISSION ST inside its footprint.

| Sheet | MISSION ST inside the footprint |
|---|---|
| p07 | 16,117 ft |
| p11 | 15,081 ft |
| p01 | 12,070 ft |
| p12 | 3,307 ft |
| p02 | 84 ft |

On the page, Mission St runs down x ≈ 2270 from y 1450 to 3270.
- The page is 3,300 × 5,100 px.
- Our boxes by quarter: ne 684, se 273, nw 28, sw 10.
- Terra's words by quarter: ne 709, se 331, nw 52, sw 14.
- Levels: ne 0.99, the rest 0.995.

## Top-line numbers (baseline, whole sheet)

**Boxes:**
- Terra saw 1,106 words; we found 995 boxes (−111 against Terra).
- At IoU ≥ 0.5: 447 matched (40%) and 659 missed (60%). Matched : missed = 0.68 : 1.
- At IoU ≥ 0.3: 602 matched and 504 missed (1.19 : 1).
- Ours not in Terra: 384 read as text (green) and 163 read as none (grey).

**Text read:**
- 448 boxes matched. They meet 447 words: one word is met by two boxes.
  - Same as Terra: 313 (70%, green).
  - Different: 135 (purple).
- Zoning-code words: 98 matched, 58 same, 40 different.
- The owner's answered labels on this sheet: 46 / 122 right at our boxes (0.377; block CI 0.29–0.46; 115 have a box).
  Terra's strings get 107 right.
- The text-ink red covers 1,499,057 px: 68% of the sheet's 2.2 M ink px.

## Reason codes (the 659 words missed at IoU 0.5)

| Reason | Words |
|---|---|
| wrong size | 255 |
| merged into line | 160 |
| below level | 156 |
| outside map | 64 |
| split | 24 |

The most common pairs of kind and reason:

| Kind | Reason | Words |
|---|---|---|
| zoning code | wrong size | 135 |
| street name | below level | 99 |
| block number | wrong size | 96 |
| block number | merged into line | 85 |
| zoning code | merged into line | 61 |
| zoning code | outside map | 29 |
| other | outside map | 29 |

Missed words by kind: 254 zoning codes, 208 block numbers, 159 street names, 38 other. At Terra's box, the fold
reader's best path equals Terra's string for 420 of the 659 missed words.

## The page

`/Users/sbuss/workspace/sfzoning/.claude/worktrees/saved-reads/pipeline/data/solver/experiments/wordread/review/baseline/review.html`
- 2.11 MB (2,107,552 bytes). No external resources; no local paths in it.
- Its numbers and misses: `numbers.json` and `missed.json` beside it.
- Rebuild: `cd pipeline && uv run python scripts/wordread/review_page.py`, about 18 s on the CPU.

What I looked at, with headless Chrome (screenshots in the scratchpad `t2/` folder):
- **Desktop overview** (`desk.png`): the four panels, then the whole sheet. Red covers most of the line work as well as
  the words, green shading clusters in the dense north-east, and the title block is shaded yellow.
- **Mission St at 3×** (`zoom2c.png`):
  - Large rotated green boxes: level-0.99 components that join several words through the lines.
  - Yellow missed codes (NC-3, NC-1, RH-2), each with a reason line and the reader's reading at Terra's box.
  - Matched block numbers in green, with arrows and tags.
- **Purple box tapped at 3×** (`desk_tapc.png`): "RH-1 (Terra: RH-3)". The panel shows the reading, the top three (RH-1
  0.60, RH-3 0.22, RH-2 0.15), the best paths, the direction, the kind probabilities, the chance and level, the size,
  Terra with IoU 0.67, and the quarter.
- **A 390 px iframe, dark theme** (`phone2.png`, tapping the missed NC-3):
  - The panels stack with no horizontal scroll.
  - The map stays white.
  - The bottom sheet shows "wrong size", 1 component 187 px long (word 97 × 38), "also holds 3596, RH-2, MISSION",
    best box IoU 0.17, and the reader at Terra's box: NC-3 0.99.
- **Desktop, dark theme, the quieter south-east** (`desk-dark.png`):
  - green block numbers;
  - "656 (Terra: 6562)" purple;
  - "6585" missed, below level (19% ≥ 0.995);
  - "ST (Terra: ST.)".

## Concerns

1. **merged_into_line is the brief's length rule (> 3×), not "a line".** Some of these words are merged with a
   neighbouring word rather than a line. For example, block "3610" (32 px) shares a 101 px component with "RM-1". The
   details and the reason line now name the other Terra words in the component (`shares_component_with`), so these are
   visible. A stricter rule ("the component's extra length isn't other words") would be a change to the brief's
   definition.
2. **wrong_size is the largest bucket (255).** It covers:
   - several words joined through lines into one component that is under 3× long (e.g. NC-3 with 3596, RH-2 and
     MISSION, IoU 0.17);
   - boxes that fit at IoU 0.3–0.5.

   The details tell them apart: `best_iou`, `shares_component_with`, `component_px`.
3. **Comparison is strict after norm_text.** For example, our "ST" against Terra's "ST." is purple, so Terra's
   punctuation counts as a difference.
4. **Red ≥ 0.5 marks most line ink on this sheet** (68% of ink). That is r328's state, and it means the red doesn't
   separate text from lines here. The dense Mission/NE area is cluttered: large green boxes, and labels overlapping at
   2–3×. The tap panel is the reliable way to read a spot.
5. **The image is the binarised ink** (page < 128, the detector's own input) in grey and red, not the scan's grey
   levels.
6. **Not checked on a real phone or in Safari.** Pinch and tap use Pointer Events. The phone check was a 390 px iframe in
   headless Chrome, because headless Chrome's window is at least 500 px wide.
7. **The script stops on a run with no best paths.** It needs best paths in `reads.parquet`, which `textread.read`
   writes. A later run's store without them would stop with a message, rather than recompute them.
8. **I read Task 1's private helpers.** `review.matching` reads `measures._overlaps` and `measures._best_word` directly,
   to stay on Task 1's matching rule.

## Fix round 1 (commit ce14d96)

### 1. The `merged_with_word` reason

`review.miss_reason` now checks the reasons in the ruling's order:
1. `merged_with_word`
2. `merged_into_line`: longer than 3× the word, and no other word in the component
3. `outside_map`
4. `below_level`
5. `split`
6. `wrong_size`: the remainder

A word is `merged_with_word` when its main component also holds another Terra word, by either of two tests:
- `merged_with`: 10% or more (MERGE_PART) of that word's ink at the level lies in the component. Only the ink outside
  this word's own rectangle counts, so a loose neighbouring Terra rectangle that overlaps the word is not a merge.
- `shares_component_with`: that word's own main component is this one.

To support this:
- `miss_reason` takes new keyword options: `others`, `other_names` and `shares`.
- `miss_reasons` passes every other word on the sheet.
- `Components` gains `bbox`, so only nearby words are checked.
- The details gain `merged_with` (at most 5 names) and `on_map`.
- REASONS, REASON_SHORT, REASON_TEXT, the module docstring, the overlay line ("with 3596, RH-2 (187 px)") and the tap
  panel ("that component also holds") are all updated.

Task 1's geometric reasons are unchanged.

### 2. The two text-read lines

The text panel now has two lines.
- **"All matched boxes, any kind, string vs Terra's":** 448 boxes; 330 same (74%), 118 different.
- **"Zoning-code forms only"** (Task 1's `read_numbers`: Terra's string a printed form, against our likeliest code):
  83 same, 16 different, 349 other matched boxes. On the owner's labels whose box is matched: ours 31 / 35, Terra's
  35 / 35.

Both are in `numbers.json` (`reads.all` and `codes`), and both count in "vs last run". I dropped the old
per-Terra-kind line from the panel; it is still in `numbers.json` as `reads["zoning code"]`.

### 3. The cheap extras

- `colour` compares the strings through `_fold`: `norm_text`, then a trailing "." dropped and runs of spaces made one.
  This applies to the green/purple comparison only. "ST" now matches Terra's "ST.", which moves 17 boxes from purple
  to green (313 → 330).
- `text_ink_of(ink, chance)` makes the red rule a function, and a test pins it: chance 0.49 (125/255) is not red, 0.5
  (128/255) is red, and paper is never red.
- The page now says that the image is the binarised ink (px darker than 128), drawn grey or red, not the scan's grey
  levels.

### Tests

- 5 new tests:
  - two words joined by one component are `merged_with_word`, even off the map and on a long line;
  - a word on a long line with no other word on it is `merged_into_line`;
  - a neighbouring rectangle that overlaps the word, with no ink of its own, is not a merge;
  - the colour fold;
  - the red threshold.
- The neighbour test now also asserts `merged_with_word` and `merged_with`.
- `tests/test_wordread_review.py`: 20 passed (also with `-W error`).
- `test_build.py`: 43 passed.
- Full `uv run pytest -q`: 1589 passed, 3 xfailed, 1 error. The error is only another agent's work-in-progress file
  `tests/test_wordread_readercrops.py`, which imports `wordread.readercrops`, a module that doesn't exist yet.
- `pgrep` was empty before the commit, and the commit holds only `review.py` and `test_wordread_review.py`.
- ruff is clean on my files, apart from the script's `# noqa: E402` markers (the same pattern as `decide.py`).

### Rebuilt baseline page

`data/solver/experiments/wordread/review/baseline/review.html` is 2.14 MB. The box numbers are unchanged: 447 matched,
659 missed.

New reason counts for the 659 misses (before → after):

| Reason | Before | After |
|---|---|---|
| merged with word | — | 310 |
| below level | 156 | 145 |
| wrong size | 255 | 116 |
| outside map | 64 | 58 |
| split | 24 | 18 |
| merged into line | 160 | 12 |

The most common pairs now:

| Kind | Reason | Words |
|---|---|---|
| zoning code | merged with word | 153 |
| block number | merged with word | 115 |
| street name | below level | 93 |
| block number | wrong size | 62 |
| zoning code | wrong size | 44 |
| street name | merged with word | 35 |

I looked at the rebuilt page in headless Chrome. The panels show the two text lines and the binarised-ink sentence.
Tapping NC-3 now gives "merged with word: … also holds 3596, RH-2, MISSION".

### Concern

Taking "check it first" literally puts `merged_with_word` ahead of `outside_map` and `below_level`. Of the 310:
- 6 are off the map (boxes there are dropped anyway);
- 16 have less than half their ink at the level;
- 148 are in components longer than 3× the word, so they were `merged_into_line` before.

If those first two groups should keep their old reasons, the fix is to move `outside_map` and `below_level` ahead in
`miss_reason`.

### Ruling on the order (commit 0473912)

The controller ruled that `outside_map` and `below_level` come first, since a merge matters only once the ink was
called text. The order is now:
1. `outside_map`
2. `below_level` (under half the word's ink at the level)
3. `merged_with_word`
4. `merged_into_line`
5. `split`
6. `wrong_size`

REASONS and the module docstring follow the same order.

Tests:
- The merge test no longer runs off the map. It is now `test_two_words_joined_by_one_component_are_merged_with_word_before_line`:
  the component is more than 3× the word's length, and the result is still `merged_with_word`.
- A new test, `test_off_the_map_and_below_the_level_come_before_a_merge`, covers a word that both shares a component
  with "12" and has only 1 of its 7 glyphs at the level. It gives `below_level`; the same word off the map gives
  `outside_map`. Both cases still record `merged_with == ["12"]`. Under the old order this test fails.
- `tests/test_wordread_review.py`: 21 passed (also with `-W error`).
- `test_build.py`: 43 passed.
- Full `uv run pytest -q`: 1590 passed, 3 xfailed, 1 error. The error is only another agent's work-in-progress
  `tests/test_wordread_readercrops.py`, which imports a missing `wordread.readercrops`.
- `pgrep` was empty before the commit, and the commit holds only `review.py` and `test_wordread_review.py`.

The rebuilt baseline page (`review/baseline/review.html`, 2.14 MB) has these reason counts for the 659 misses:

| Reason | Words |
|---|---|
| merged with word | 293 |
| below level | 156 |
| wrong size | 116 |
| outside map | 64 |
| split | 18 |
| merged into line | 12 |

None of the 293 `merged_with_word` words is now off the map or below the level. 148 of them sit in components longer
than 3× the word.

The most common pairs of kind and reason:

| Kind | Reason | Words |
|---|---|---|
| zoning code | merged with word | 150 |
| block number | merged with word | 113 |
| street name | below level | 99 |
| block number | wrong size | 62 |
| zoning code | wrong size | 44 |

The box and read numbers are unchanged: 447 matched, 659 missed; 330 same and 118 different in any kind; 83 same and
16 different on code forms.
