# Survey: where the 2000 solver takes LLM output, and what replacing it touches (2026-11-16, read only)

Abbreviations: P/ = pipeline/; solver/ = P/src/sfzmaps/solver/; districts/ = P/src/sfzmaps/districts/.

## 0. Premises to correct

1. **The solver's lots are the April 2000 table's**, not today's: `city.lots` = `P/data/solver/lots/table.parquet`
   (solver/table.py:61-69, solver/lots.py:87-101).
   - 153,292 rows, 153,247 unique blklots (45 duplicated IDs).
   - The published map is on DataSF's lots valid in mid-2000 (districts/read.py:24-43): `districts_raw` has 184,241,
     2,769 with no code.
   - 1,637 table IDs are not in `districts_raw`; 32,631 publish lots are not in the table.
   - In a sample of 3,000 of those 32,631, 2,970 lie more than half inside one table lot, so a majority-area map
     covers them.
2. **S3's 2000 numbers are not held out.** It read `paths.TEXT_CHANCE_DIR`, the `solver-text` detector, trained
   with `hold_out_validation=False` (solver/textnet/stage.py:1-4,41). Only its test-drawing numbers are honest.
3. **The walls already read a detector's chance,** but the in-sample `solver-text` one.
4. **S0 and S1 compare against the old final** (7,100 wrong lots / 94.47%), not today's 4,543 / 94.79%.

## 1. The label chain today

- **prints** (`solver-prints`).
  - Code: solver/prints.py:333-375. Sol is joined to Terra at 125-150; the turned pass is 153-172; the third read
    and the recheck are found by their rebuilt cache key at 228-274, and the stage stops below 95% found at 281-287.
    `under` (the lot under the point) is 290-302; `_place` keeps a print inside the footprint and on the mapped area
    (305-314).
  - Reads:
    - Terra: `saved-reads/answers/text-on-every-sheet.jsonl.gz`, through textnet/boxes.py:54-104;
    - Sol: `data/work/2000-digital-zoning/<sheet>/district_labels.json`, kind "use" (solver/saved.py:48-54);
    - the sideways, third and recheck reads;
    - the page, the warp, `lots.npz`, `sheets.png`.
  - Writes `prints.parquet`: 1,904 rows (kind 0: 1,525; kind 1: 244; kind 2: 94; kind 3: 41; 135 with no box; 455 on
    a street). Also `terra_boxes.json` and `summary.json`.
- **trust** (`solver-labels`).
  - Code: solver/labels_stage.py:11-19 and solver/trust.py. 23 features at 59-83, built at 136-200. Training rows
    at 228-231; the fit at 234-238; the output at 260-279.
  - Every feature comes from the five reads; the height feature comes from Terra's box.
  - One fit per quarter. It trains on rows whose `lab_lot` is known and lies outside q. Target: the reading's code
    equals the table-as-downloaded code at `lab_lot` (198-199).
  - Writes `readings_{q}.npz` (2,363 readings, 118 names) and `trust.json`.
- **street side** (solver/streetside.py:137-158). The other-side lot is 68-103; the rates are fitted per quarter at
  118-134. Writes `streetside_{q}.npz`.
- **lot size** (solver/lotsize.py:58-74; PMI per quarter, 52-55) and **strokes** (solver/strokes.py): no labels.
- **`labels.readings`** (solver/labels.py:37-59).
  - Adds a far-side copy at trust × min(rB_st / rA_st, 1) when `onst & ~isfw & lotB >= 0`.
  - Drops overlay names, and names outside `table_vocabulary` (solve.py:126).
- **placement** (in `solver-solve`; solver/placement.py).
  - Code: options 78-181, answers 149-170, fit 195-240, loading 243-257.
  - Reads `x`, `y`, `sheet`, `lab_lot`, `street`, the top code and its trust (105-111), `walls_{q}`, strokes, lot
    size, the table, and `label_fixes.csv` leader moves (56-75).
  - Quirk: line 74 matches `canon(fx["code"])`, not `fix_code`; the two differ in 21 of 101 move rows.
  - A row's quarter comes from `lab_lot`. Model qi trains on the other quarters; the solve for q reads `holdout == q`.
  - Writes `placement.npz` (32,640 options for 1,904 labels; 1,723 with answers, 80 of them hand moves).
- **`energy.label_costs`** (solver/energy.py:30-63, called at solve.py:48-55,105-141).
  - Per (label, far): T[code] = the highest trust any reading gives that code; q = logistic(0.128 + 1.085 ·
    logit(T_top)) (`TOP_READING_FIT`, energy.py:12, fitted on J7's trust); rest = (T_others + 1/K) / (ΣT_others + 1);
    P = q·e_top + (1 − q)·rest.
  - Then U[lot] −= log(π·P + (1 − π)/K) over the placement options. Options are used only when the stay option's lot
    is the label's own lot, and never for far-side copies.
  - A tie-break of 1e-4 × `path_costs` is added.
- **Scoring:** "labels right" (1,881 labels) comes from `top_readings(q)`, the same readings the solve used
  (measures.py:69-75, report.py:95-145).

## 2. Formats a replacement must produce

- **`prints.parquet`.** Its row number is the label id everywhere.
  - Columns: `sheet`; `page_x`, `page_y`; `x`, `y` (EPSG:2227 feet); `kind` (int8 0-3); `box_x0..box_y1`; the five
    reads (`terra_code`, `sol_code`, `sol_confidence`, `turned_*`, `third_*`, `recheck_*`); `lab_lot` (int32, ≥ 0 for
    kept prints); `street` (bool, no lot at the point).
  - The solve needs only `sheet`, `x`, `y`, `lab_lot`, `street`, plus whatever the calibration reads.
- **`readings_{q}.npz`:**
  - `names` (str, spellings `canon` accepts);
  - `print_index` (int64);
  - `code_index` (int64);
  - `trust` (float64 in [1e-6, 1]);
  - several readings per print allowed.
- **`streetside_{q}.npz`:** `lotB` (int64 per print, −1 for none), `onst`, `isfw` (bool per print), and `rA_st`,
  `rB_st`, `rA_fw`, `rB_fw` (0-d float64).
- **Placement reads per label:** `x`, `y`, `sheet`, `lab_lot`, `street`, the top code and its trust. Labels with no
  reading are skipped (121-123).
- **Direct from boxes:** `sheet`, `page_x`/`page_y`, the box, `x`, `y`, `lab_lot`, `street` (through `_place` /
  `under`), and `lotB`/`onst`/`isfw` (`far_side` unchanged).
- **Lossy:** trust = posterior mass (as S0 did).
- **A new fit:** the calibration replacing `trust.py`. Same rows, target and quarter rule; reuse `FeatureTable`,
  `newton_fit`, `training_rows` and `build_readings`. Street-side rates and placement refit on their own.

## 3. Walls

- **Terra's letter share.**
  - `DEFAULT_CANDIDATES = (STREET_CROSSING, CONTEXT)` (walls.py:72). Because `text_weighted` is not among them,
    `load_inputs` builds Terra's share (walls.py:260-265).
  - `terra_text` (206-225) draws every Terra box from `prints/terra_boxes.json` (`wallevidence.text_rings` 278-292,
    then `text_raster` 295-304). `letter_share` (257-269) gives each stroke its share of points inside a word.
  - In `border_features`, `terra = TEXT_WEIGHTED not in options` (wallevidence.py:655). The share fills 12 of the 463
    features: stroke evidence `per = [letter[k]] + per` (686, 6 knots) and profile `close * letter[k]` (766, 6
    levels). walls.py:322 records "strokes touching text" (106,761).
- **The detector's chance is already read.** `STREET_CROSSING` is in `READS_TEXT` (wallevidence.py:104), so
  walls.py:266 calls `textgrid.text_chance()`. That reads `data/solver/text/chance/<sheet>.png` (paths.py:43,
  textgrid.py:31-38) and is a DVC dep (build.py:185-193).
- **Switching:**
  1. **`text_weighted`** (wallevidence.py:32-35,671-694,749-763): the same columns, with the detector's chance in
     place of Terra's share.
     - README 868-877: .0982/.1026/.0452/.0612 against .0978/.1027/.0452/.0611, a tie.
     - Change `DEFAULT_CANDIDATES`, drop `terra_boxes.json` from `solver-walls`' reads (build.py:188), regenerate
       dvc.yaml.
  2. **A letter share from our own boxes:** `text_rings` would need corners in place of box plus angle.
  3. **Honest chance:** `textgrid` reads `data/solver/text-reader/<sheet>/chance.png`. That needs a source switch in
     `_chance_files`, and `solver-walls` gains `solver-text-reader` upstream, which depends on `solver-prints` for
     the answered-block hold-out.
- Any walls change moves `walls_digest`, so placement refits.

## 4. Other LLM uses

- **Prints, trust and readings:** block a no-LLM solve.
- **Walls letter share:** blocks it; fix with option 1.
- **`solver-text` chance:** not LLM output, but in-sample. Use the per-quarter chance.
- **Placement answers** (the table, `label_fixes`): training only.
- **`solver-page` / `solver-lots`:** only page.png is read (page.py:36,141).
- **The warp** (anchors.py:1-20, vision-read seeds): later.
- **`solver-text` / `solver-text-reader`:** training and hold-out only.
- **Scoring "labels right":** must be fixed (see §9, item 1).

## 5. Held-out structure

- **Today:** trust, street side, lot size, placement and walls are fitted per quarter (by lot centroid).
  `solver-text` is one model with nothing held out. `solver-text-reader` (textnet/reader_stage.py:215-272) has four
  detectors with the quarter and the answered blocks held out, keyed by each px's map point.
- **Recommendation: four text readers,** one per quarter, with the answered blocks held out of all four. A reader
  that read a crop with Terra's own string as training would reproduce Terra; and the calibration needs the same
  error profile everywhere.
- **Reuse from `reader_stage.py`:**
  - `held_blocks` (69-71), `load_block_area` (87-93);
  - `data.Quarter` (data.py:102-124), `hold_out` (333), `in_area` (401);
  - `page_quarters` (127-137), `sheet_outputs` (140-155), `measure_radius` (96-122);
  - `save_npz` (158-166), `runs.fresh_folder` / `replace_folder`, `gpu.gpu_turn`, `held.json` / `scores.json`,
    `quarter_scores` (190-210).
- **Cost:** about 85 min of GPU for four readers, plus about 3.5 min for each 0/180 classifier.

## 6. Publish path

- **Today:**
  - `period-2000-digital` runs `districts --series` (build.py:348-349), reading `districts_raw.parquet` (blklot,
    code, height_code, confidence, source_sheet; districts/read.py:21).
  - `assemble.publish` (districts/assemble.py:569-616) applies, in order: unzoned blocks; `overrides.csv` (273-313: 5
    rows, all P, keyed through `era_lot_of`, districts/lots.py:101-121); the height join (from the LLM-read
    2000-digital-height series, 582-589); parks; categories (`docs/research/zoning-districts.csv`).
  - It writes `data/out/2000-digital-zoning/districts/{parcels.fgb, districts.fgb, stats.json}`.
  - `shared` merges them (districts/shared.py:85-108). The site reads `stats.json` and the shared files
    (site/scripts/publish-tiles.mjs:112-119).
- **No per-lot path from the solver to the site exists** (solver/published.py goes the other way).
- **Blockers:**
  - DVC refuses overlapping outputs (build.py:10-11): `period-2000-digital` owns `data/out/2000-digital-zoning`
    (dvc.yaml:720).
  - Lots must map to DataSF's 2000 lots: `shared.merge` stops if one blklot has different geometry in two eras
    (shared.py:92-95).
  - Spellings: 17 of final's 51 names are not in `districts_raw` (NORTH BEACH against NORTH BEACH NCD, HAIGHT against
    HAIGHT ST NCD, C-3-R, SPD).
  - Heights are LLM-read; there is no per-lot confidence; `source_sheet`; the twin 1996-2000 series
    (corrections.py:14-15).
- **Options:**
  - **A:** a generated file in `inputs/`, plus forced period and shared rebuilds. The smallest, but no DVC link.
  - **B:** a new publish folder, with every reader repointed: shared.py:62, parks.py:135, downloads/__init__.py:62,
    ship.py:188, reviewpage.py:102, cli.py:437,444, and the site script.
  - **C:** narrow the period stage's outputs.

## 7. Dependencies

- torch==2.15.0 is pinned, with a CPU index on Linux. torchvision would need a matching pin and the same source
  line.
- Every stage depends on pyproject.toml and uv.lock. The precedent is 44dffb1 (`dvc commit --force` after a digest
  check).
- python-doctr cannot be a dependency (opencv-python against opencv-python-headless, onnx, h5py and more).
- Plain torch is feasible: MobileNetV3-small `.features` with stride (2,1) at `features.{2,4,9}.block.1.0`, then a
  2-layer BiLSTM 128, then Linear(256 → 46). About 150 lines. Keep the parameter names.
- New code goes under `solver/`, which is excluded from the old stages' deps (build.py:51-54). Edits under
  `districts/` or to cli.py re-hash every warp stage (precedent 7cce81a: recorded unchanged).

## 8. Minimal sequence (each step measured against final's six measures with the paired block bootstrap)

0. Fix the scoring labels; re-solving `final` gives a byte-identical report.
1. Walls without Terra (`text_weighted`).
2. Per-quarter chance in `textgrid`.
3. `solver-boxes` from S3's `group` on the text-reader outputs.
4. The reader stage in plain torch (four readers and four D classifiers).
5. Labels from boxes, with the new calibration.
6. Publish.

## 9. Risks

1. "Labels right" changes with the label source.
2. Don't overwrite `prints.parquet`.
3. Missed boxes cost lots: S1 measured about 483 wrong lots per point of miss (interval under review).
4. The open posterior ties at defaults; use the closed-list ranking, with p(other) and p(none) as calibration
   features.
5. The synthetic inputs (`font_real.npz`, `prior3.json`, `labs.pkl`) are made by no stage.
6. S2 used Terra's reading direction; our boxes give only an axis.
7. A distribution shift: trained at Terra's boxes, reading ours.
8. Placement's and `far_side`'s loops scale with the label count.
9. The `fix_code` quirk.
10. `TOP_READING_FIT` is shared.
11. Publishing (overlap, lots, spellings, heights, re-hashing).
