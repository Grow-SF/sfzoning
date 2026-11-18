# Task 1 report: the label set, the code list and the line masks

Status: DONE (with concerns, below). Branch `solver`, commit 5209120. CPU only; no LLM, no `.env`, no GPU, no
pipeline stage run.

## What was built

`pipeline/src/sfzmaps/solver/codereader/` (`__init__.py`, `labels.py`); tests in `pipeline/tests/test_codereader.py`.

- **`Label`**: sheet, `point` (page px), `box` (Terra's, or None), `direction`, `angle` (Terra's reading angle),
  `crop` (uint8, straightened), `to_page` (2x3 affine), `centre` (the point in the crop), `code_rect` (Terra's word
  in the crop), `terra_text`/`sol_text` (as written), `reads` (terra, sol, turned, third, recheck; canon spelling),
  `lab_lot`, `block`, `cases`, `answers` ((id, owner's code, flagged) per answered case), `answer`, `held_out`,
  `flagged`, `case_blocks`, `fold`. Properties: `ink`, `word` (length, height).
- **`load_labels(crops=True)`**: one label per print (prints.parquet order), plus one per answered case that joins
  no print (kind -1). Pure core: `cases_from`, `join_cases`, `make_labels`, `assign_folds`.
- **`training(labels)`**, **`held_out_blocks(labels)`**.
- **`code_list(labels=None, names=None)`**, **`code_of(text, names)`**, **`table_codes()`**, `ALIAS`.
- **`place_crop` / `add_crops` / `measure_sizes` / `Sizes` / `frame` / `sample`**.
- **`line_mask(label, lines=None, rules="long")`**, with three rules in `LINE_RULES`: `long` (the detector's
  `scores.line_ink`, called, not edited), `width` (`width_lines`), `run` (`run_lines`). `sheet_lines(sheet, rule)`
  caches per process.

### Decisions (my calls; say if you want them otherwise)

- **The join** is the brief's rule exactly: the kind-2 print at the case's very point (exact float equality; Sol
  points and case points are whole px), else the print whose Terra box holds the point (edges included; nearest
  print point of several). No 20 px fallback (the recon had one; it only joined unanswered r1321 to a kind-3 print).
- **Held out = any answer joined, including "unsure" and "no label"** (conservative). Unanswered cases (905) don't
  hold anything out.
- **Blocks held out**: each held-out label's block **and the block under each answered case's own point** (page point
  -> sheet placement -> `grown` lot -> `block_num`). This adds 36 blocks the label's own block misses (case points in
  streets, nearest lot across the street), and it catches r0236, the one answered case on no print (outside its
  sheet's footprint; its block 6965A is held out, so the same code on the neighbouring sheet isn't trained on).
  Labels with no block are left out of training too (none among the prints).
- **A label's `answer`**: the code when its answered cases agree; None when they disagree. One print does: #1287, a
  278x245 px Terra box "UPPER MARKET NCD" swallowing two separate answered labels (RH-2, P). Its `answers` keeps both,
  so Task 4 can score per case (717) or per label (704).
- **Kind -1 labels**: r0236 becomes a point label at its case point with Sol's read (the case's code) as its only
  read, so all 717 answered codes are on a label.
- **Fold**: labels with an owner's code or "no label"; strata = (flagged, any "no label" answer), in sorted order;
  within each, labels ordered by (sheet, print index, point), permuted by one `np.random.default_rng(20261115)`, and
  assigned 0/1 alternately. Label-level, not block-level (three constants; noted in case you want block grouping).
- **Crops**: Terra's word rectangle (`textnet.data.word_rect`) + a margin of `REACH` = 3 letter heights on every side
  (letter height = median word height of the *training* labels: 27 px -> margin 81 px). Quarter turns are an exact
  px copy (integer affine, gathered with numpy); other angles are bilinear (`cv2.warpAffine` INTER_LINEAR,
  WARP_INVERSE_MAP, white border). No box (kinds 2, 3, -1): an upright square of half-side ceil(L95/2) + margin =
  144 px (L95 = 95th percentile word length of training labels, 125.8 px), so upright and sideways both fit. Sizes are
  measured on training labels only (box geometry of answered labels never fits anything).
- **Code list** (`code_list`): the 57 table names (lots.npz `table[known]`, list only) as themselves, plus `ALIAS`
  ("NOE VALLEY" -> "24TH-NOE VALLEY"; explicit, world knowledge of the district's name, since its only read is on a
  held-out label), plus every printed form in *training* reads (Terra's and Sol's as written, the others canon)
  that `code_of` (canon, then ALIAS) takes to a table name.

## Real run (CPU)

`load_labels()` takes 2.4 s (13 pages read, 1,905 crops).

| | count |
|---|---|
| labels | 1,905: kind 0 1,525, kind 1 244, kind 2 94, kind 3 41, kind -1 1 |
| code-check cases joined to a print | 1,645 of 1,648 (unjoined: r0236 answered -> kind -1; r0817, r1321 unanswered) |
| held-out labels | 731 (704 with a code, 582 of them unflagged; 20 "no label"; 6 unsure-only; 1 disagreeing) |
| answered cases with a code on a label | **717, 593 unflagged** (matches the spec); 20 "no label" |
| held-out blocks | 715 (36 only from a case's own point) |
| training labels | **959** (kind 0 734, 1 171, 2 28, 3 26); 215 unanswered labels left out for sharing a held-out block |
| folds | fold 0: 364 labels, fold 1: 361 (unflagged codes 292/291, flagged codes 61/61, no-label 11/9) |
| crops | 1,620 quarter-turn (exact), 149 resampled, 136 point windows; median 189 x 229 px, max 288 x 523 |

**Code list: 70 forms** (57 table names + 13): BROADWAY NCD, CASTRO NCD, FILLMORE, HAIGHT NCD, HAYES-GOUGH NCD,
INNER CLEMENT NCD, NOE VALLEY (alias), OUTER CLEMENT NCD, RH2, SACRAMENTO NCD, UPPER FILLMORE NCD,
WEST PORTAL AVENUE NCD, WEST PORTAL NCD.

**Answered codes outside the table's list: 5, not 6**: NC (fix), CMNC (fix), NOE VALLEY (ok), MB-OS (ok), 3RD
("3RD & A. A.F. SUD", fix). RC-1 *is* in the table's list (2 lots; answer r0430, read NC-1 by Terra/Sol/turned and RC-1
by the third read). With the code list, NOE VALLEY maps to 24TH-NOE VALLEY; 4 stay outside (NC, CMNC, MB-OS, 3RD).

### Line ink near labels

There is no marked line ink, so "line ink" = ink outside every Terra word rectangle of the sheet (all kinds): lines,
plus any lettering Terra didn't box (some Terra boxes sit off their text, so this is a noisy proxy). Measured in every
crop, pooled. "near" = within one letter height (27 px) of the label's word rectangle; "code ink" = ink inside the
word rectangle (letters, and lines crossing it, so the share masked is an upper bound on letters masked); "letters
masked" = code ink masked on the 45 box labels with < 20 px of line ink near them (no line touches them; mostly large
lettering in open areas, so a small, biased sample).

| rule (dilated 1 px) | near | whole crop | point windows | labels < half near covered | code ink masked | letters masked (45) | 13 sheets |
|---|---|---|---|---|---|---|---|
| `long` (scores.line_ink) | **0.616** | 0.664 | 0.624 | 30.2% | 0.148 | **0.580** | 32 s |
| `width` (long + strokes >= 24 ft) | 0.797 | 0.821 | 0.796 | 5.2% | 0.393 | 0.896 | 32 s |
| `run` (straight runs >= 200 ft) | **0.870** | 0.869 | 0.843 | 1.7% | 0.307 | **0.000** | 6 s |
| `long` + `run` | 0.924 | 0.930 | 0.910 | 0.3% | 0.384 | 0.580 | |

Scratch prototypes of `run` at 40 / 54 / 81 px: near 0.94 / 0.80 / 0.55, code ink masked 0.42 / 0.27 / 0.13 (40 px
starts eating tall letters' stems). The package's `run` uses the long lines' own 200 ft (50 px at 4 ft/px).

What I saw (pictures below):
- **`long` misses lines broken by junctions**, as the spec feared: the dense lot-line grid around labels is cut into
  strokes under 200 ft, so 30% of labels have less than half the line ink near them covered. It also **masks large
  lettering**: a big "P" or "RH-1(D)" has strokes over 200 ft, so on the isolated labels 58% of the letters' ink is
  unobserved.
- **A width rule can't separate letters from lines on this map**: of the training crops' centre-line px (width =
  twice the distance to paper), letters inside their own word are 2-2.5 px 8%, 4 px 43%, 5.5-6 px 34%, 7-8.5 px 13%;
  lines (outside every word) 2-2.5 px 22%, 4 px 43%, 5.5-6 px 21%, 7-8.5 px 13%. Heavy letters go with heavy lines
  (90% of the isolated labels' letters masked). At the same near coverage as `run` at 54 px it masks more code ink
  (0.39 against 0.27). It's in the package because the brief asked for it; the measurement says it is dominated.
- **`run`** catches straight lines through junctions and leaves letters alone. It misses curved lines (contours,
  cul-de-sacs, park outlines), staircase boundaries with short steps, and the small boxes under 50 px drawn around
  some "P" codes.

The choice is Task 3's, by training likelihood: candidates `rules="long"`, `"run"`, `("long", "run")`, `"width"`.
My expectation: `run` or `long+run`.

Pictures (scratch; red = masked ink, blue = line ink not masked, black = text ink observed, green = Terra's word;
each grid shows the 40 labels with the lowest near coverage, then 24 at random; captions are coverage):
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task1/final_long.png`
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task1/final_width.png`
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task1/final_run.png`
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task1/final_long+run.png`
- crops by direction (check the straightening):
  `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task1/crops_by_direction.png`
- every "top-to-bottom" label:
  `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/task1/top_to_bottom.png`

Scratch scripts (same folder): `counts.py`, `coverage.py`, `coverage_rules.py`, `letters_masked.py`, `widths.py`,
`rules.py` (prototypes), `crops_picture.py`, `ttb_picture.py`, `one.py`.

## Tests

`pipeline/tests/test_codereader.py`, 9 tests:
- `test_training_excludes_answered_labels_and_their_blocks` (Review Focus 1): the join (in a box / at a kind-2
  point / neither), ok, fix, nolabel, unsure, an answered case on no print (kind -1, with its own block held out),
  an unanswered case holding nothing out; training = exactly the two labels apart from every answer.
- `test_folds_split_each_stratum_in_half_with_a_seed`.
- `test_alias_maps_to_its_code` (Review Focus 5): NOE VALLEY -> 24TH-NOE VALLEY; a training form joins; a form seen
  only on a held-out label and a stray word stay out.
- `test_crop_is_straightened_by_the_box_angle`: 0/90/-90/180 exact (the crop equals the upright word, padded);
  30 degrees bilinear (IoU > 0.7 with the upright word, no ink in the margin).
- `test_point_crop_is_a_square_window_on_the_point`, `test_line_mask_is_dilated_and_follows_the_crop`.
- `test_run_rule_keeps_lines_that_junctions_break`, `test_width_rule_keeps_heavy_strokes_of_any_length`,
  `test_line_mask_unites_the_rules_asked_for`.

Full suite (`cd pipeline && uv run pytest -q`): 1,096 passed, 3 skipped, in 84 s. `ruff check --line-length 120` clean on the new files.

## Concerns

1. **Terra's "top-to-bottom" is wrong on the 2000 map.** All 15 labels Terra read top-to-bottom at -90 degrees are
   printed bottom-to-top, so their crops come out upside down (the frame is right: `one_245.png` shows the raw page
   beside the crop). The 4 at -72/-75 read correctly. Tasks 2 and 3 must search the half turn (the quarter-turn grid
   of Review Focus 3 covers it). This matters for template EM in particular: a fixed-pose alignment would learn from
   upside-down letters.
2. **NCD forms missing from the code list.** 7 NCD names (24TH MISSION, 24TH-NOE VALLEY, NORTH BEACH, POLK, UNION,
   UPPER MARKET, VALENCIA) have no "<name> NCD" form, because every read of them sits in a held-out block. The table
   doesn't mark NCDs. If Task 3 renders printed forms, it may want "<name> NCD" for every name the training reads show
   with NCD (9 of the 16 named districts). That is a rule for you to decide; I didn't add it.
3. **`long` line masks hide large letters** (strokes over 200 ft): 58% of the isolated labels' letter ink. If Task 3
   keeps `long`, big "P"s are scored on little ink.
4. **The crop margin (3 letter heights = 81 px) is my call**; Task 3's position prior must sit inside it. `place_crop`
   takes any `Sizes` if Task 3 wants another margin.
5. **The code-check answers become a build input** of the code reader (training exclusion). The folder's README says
   "Nothing in the build reads this folder"; Task 5 should update it and the stage's deps.
6. The brief's "6 outside the list" counted RC-1; RC-1 is in the table's list (2 lots).
7. `line_mask` finds a sheet's lines in-process (`long` 2-3 s a sheet, `run` 0.5 s); no disk cache of my own, and
   the detector's coming disk cache for `scores.line_ink` will apply automatically (same `SheetData`, same key).
