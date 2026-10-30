# Zoning Districts (Step Two) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every zoning-map era, assign each modern parcel the district the historic map shows, join it to plain-language rules, and publish per-era vector tiles and stats, validated against DataSF's current zoning.

**Architecture:** A new `sfzmaps.districts` subpackage reads the step-one warped scans (`work/<series>/<sheet>/warped.tif`, EPSG:2227). It uses one of two readers per series: A, pattern classification by vision over batched parcel-crop mosaics; or B, district-label reading plus heavy-line region segmentation. Each parcel is assigned a code with a confidence; overrides apply next, then the research-table rules join, then outputs are published.

**Tech Stack:** Python 3.12 (existing uv project), numpy/opencv/scikit-image, shapely 2, geopandas, rasterio, pyogrio (FlatGeobuf), GDAL `ogr2ogr -f PMTiles`, and the existing cached vision client `llm.ask_json` (OpenAI `gpt-6.1-sol`).

**Spec:** `docs/superpowers/specs/2026-10-30-zoning-districts-design.md` (read the "Builder-check questions, settled" section).

## Global Constraints

- Secrets: the OpenAI key comes from `pipeline/.env` via llm.py and is never printed, logged, cached or committed. Run `git diff --cached | grep -c 'sk-'` before each commit.
- Every vision call goes through `llm.ask_json`, so it's cached. Minimise image tokens: crops and mosaics are downscaled to the smallest size that still classifies correctly, calibrated per series.
- The unit is modern active parcels: `ref/parcels.parquet` with `active` true, keyed by `blklot`.
- Series → reader → role, exactly as in the spec table:
  - A: 1921, 1929, 1935, 1948;
  - B use: 1960, 1964, 1968, 1972, 1975, 1980, and the *-zoning series;
  - B height-only: the *-height series;
  - *-su: skipped.
- The research CSV `docs/research/zoning-districts.csv` is the source of truth for rules and categories.
- Story categories: `single_family`, `two_unit`, `three_unit`, `multi_family`, `mixed`, `commercial`, `industrial`, `public`, `unrestricted`, `unknown`.
- Validation target: story-category agreement ≥ 85% between 2000-digital-zoning and DataSF today (`3i4a-hu95`).
- TDD; tests use synthetic maps and FakeClient. Real data runs never touch step-one artifacts except to read them.

## Review Focus

1. A parcel straddling two sheets, or a sheet edge: use the stitch priority, never average codes. Test in D3.
2. A parcel with no warped coverage (unplaced sheet, or outside the mapped area) → `unknown`, never a neighbour's code. Test in D3.
3. A district code the model returns that isn't in the CSV (e.g. "R-1-D" variants, OCR noise like "R1") → normalised by the alias table, else `unknown` and listed. Test in D1.
4. A label-and-boundary region with two conflicting labels → split by nearest label. A region with no label and disagreeing neighbours → `unknown`. Test in D5.
5. An override for a `blklot` that no longer exists → warn, don't crash. Test in D6.

---

## File structure

```
pipeline/src/sfzmaps/districts/
  __init__.py
  series.py       # series → reader/role/era table, research era lookup
  rules.py        # load research CSV, code normaliser + alias table, rules join
  categories.py   # code/rule → story category; code family for validation
  today.py        # fetch DataSF current zoning, assign to parcels
  sampling.py     # parcel ↔ sheet coverage/priority, parcel crops, texture stats
  pattern.py      # reader A: legend swatches, mosaics, vision classify
  labels.py       # reader B: district-label read, heavy-line regions, assign
  assemble.py     # overrides, height join, dissolve, outputs, stats
  validate.py     # agreement vs today, answer-key accuracy
pipeline/tests/test_districts_*.py, tests/synthetic_districts.py, tests/fixtures/district_keys/
```

### Task D1: Series table, rules, categories

**Files:** create `districts/__init__.py`, `series.py`, `rules.py`, `categories.py`; tests `tests/test_districts_rules.py`.

**Interfaces (produces):**
- `series.SeriesInfo(series: str, reader: Literal["A","B","skip"], role: Literal["use","height","skip"], year: int, period: str)`. Example period values: `"1980-1990"` for `1980-1990-height`/`-zoning`; the series name itself for single-year series.
- `series.SERIES: dict[str, SeriesInfo]`, covering all 22 series in `pages.discover()`.
- `rules.load_rules(path=…) -> pandas.DataFrame`.
- `rules.normalise_code(raw: str, year: int) -> str | None`.
- `rules.rule_for(code: str, year: int) -> dict | None`: the CSV row whose era range contains the year. 1935 uses 1929–1946.
- `categories.category_for(code: str, year: int) -> str`.
- `categories.family_for(code: str) -> str`: e.g. RH, RM, RC, NC, C3, M, P, PDR, other.

- [ ] Tests:
  - `test_known_categories` (parametrised):
    - (1921, "1R") → single_family;
    - (1921, "2R") → multi_family;
    - (1960, "R-1") → single_family;
    - (1960, "R-2") → two_unit;
    - (1960, "R-4") → multi_family;
    - (1960, "C-2") → commercial;
    - (1980, "RH-2") → two_unit;
    - (1980, "RH-3") → three_unit;
    - (1980, "RM-1") → multi_family;
    - (1990, "NC-2") → mixed;
    - (1960, "M-1") → industrial;
    - (1960, "P") → public;
    - (1921, "U") → unrestricted.
    Adjust only where the CSV says otherwise; the CSV wins, so document each adjusted case in the test.
  - `test_normalise_aliases`: "R1" → "R-1", "rh 2" → "RH-2", " C-3-O " → "C-3-O", garbage → None.
  - `test_series_table_complete`: every series from `pages.discover()` has an entry.
  - `test_unknown_code_is_unknown`.
- [ ] Implement. Build the alias table from the CSV's compound code strings plus common OCR variants. Commit `feat(districts): series table, rules join, story categories`.

### Task D2: Today's zoning from DataSF

**Files:** create `districts/today.py`; test `tests/test_districts_today.py`; modify `sources/datasf.py` (add a fetch for `3i4a-hu95` → `ref/zoning_current.parquet`, using the existing paging helper).

**Interfaces:**
- `today.assign_today(parcels: GeoDataFrame, zoning: GeoDataFrame) -> GeoDataFrame` with columns `blklot`, `code`, `category`, `confidence=1.0`. The code comes from the zoning polygon covering the largest share of the parcel's area, using the DataSF field `zoning`, with `zoning_sim` as a fallback.

- [ ] Tests:
  - `test_majority_overlap`: a parcel 70/30 across two zones gets the 70% zone.
  - `test_today_categories`: uses categories.py.
- [ ] Implement. Then run it live, fetching and assigning, and report the category counts. Commit `feat(districts): today's zoning from DataSF`.

### Task D3: Parcel sampling and sheet priority

**Files:** create `districts/sampling.py`; test `tests/test_districts_sampling.py`.

**Interfaces:**
- `sampling.sheet_order(series: str) -> list[str]`: reuse `stitch.load_scores`/the VRT priority (passed first, then lower held-out p50).
- `sampling.assign_sheets(parcels, series) -> pandas.Series`: blklot → sheet_id, the highest-priority sheet whose valid-alpha footprint (`warp.footprint_polygons` / the stitch cutlines) covers ≥ 50% of the parcel, else None.
- `sampling.parcel_crop(warped_tif: Path, geom, *, inset_ft=3.0, px=64) -> np.ndarray | None`: an RGBA crop of the shrunk parcel, masked outside the parcel. Returns None when the parcel is < 50% alpha-covered.
- `sampling.texture_stats(crop) -> np.ndarray`: ink fraction, gradient orientation histogram (4 bins), and a 2-D FFT peak ratio.

- [ ] Tests on a synthetic warped GeoTIFF (built with `tests/synthetic_districts.py`):
  - `test_priority_two_sheets`: overlapping sheets → the higher-priority one wins.
  - `test_uncovered_parcel_none` (Review Focus 2).
  - `test_crop_masks_outside`.
  - `test_texture_stats_distinguish_hatch_vs_solid`.
- [ ] Implement. Commit `feat(districts): parcel crops, sheet priority, texture stats`.

### Task D4: Reader A (pattern maps)

**Files:** create `districts/pattern.py`; create `tests/synthetic_districts.py` (a pattern-map generator: parcels filled with white, hatch, solid, stipple and crosshatch per a fake legend); test `tests/test_districts_pattern.py`.

**Interfaces:**
- `pattern.legend_swatches(series) -> dict[str, np.ndarray]`.
  - Crop the legend from the series' page that has one, using `page.json` mask_boxes plus a vision call asking for the legend entries with their swatch boxes (cached).
  - Fallback: a hand-entered per-series config in `districts/legends.json` listing (code, label, swatch box on a given sheet's page.png). Fill in 1921 and 1948 by inspection if the vision crop fails.
- `pattern.build_mosaic(crops: list[np.ndarray], swatches) -> tuple[np.ndarray, list[int]]`: the legend strip on top, numbered tiles of ≤ 48 crops, each tile ≤ 96 px. Calibrate down.
- `pattern.classify(parcels_with_crops, swatches, *, client=None) -> pandas.DataFrame` with `blklot`, `code`, `confidence`. Pydantic schema: `PatternResult(items: list[{index:int, code:str, confidence:float}])`.
- Block-uniform shortcut: within a block, if every parcel's `texture_stats` is within a tolerance, classify one representative crop and apply it to all parcels in the block, with confidence × 0.9.

- [ ] Tests with FakeClient:
  - `test_mosaic_layout_and_index_map`;
  - `test_uniform_block_shortcut_one_call`;
  - `test_mixed_block_per_parcel`;
  - `test_results_mapped_back_by_index`.
- [ ] Live: classify one 1948 sheet (sheet "1"). Render the category fills over the scan into the scratch dir and view them. Check the legend mapping by eye (black = Commercial, etc.). Record cost. Commit `feat(districts): pattern-map reader`.

### Task D5: Reader B (labels and boundaries)

**Files:** create `districts/labels.py`; extend `tests/synthetic_districts.py` (a label-and-boundary map: thin block outlines, heavy district lines, codes printed in regions); test `tests/test_districts_labels.py`.

**Interfaces:**
- `labels.read_district_labels(series, sheet_id) -> list[DistrictLabel]`, where `DistrictLabel(code: str, kind: Literal["use","height"], x: float, y: float, confidence: float)`.
  - Positions are page px. Reuse the `reading` tiling, skip and scale machinery (`reading.select_tiles`, `config.label_scale`) with a new prompt constant and schema.
  - Cache artifact: `work/<series>/<sheet>/district_labels.json`.
- `labels.heavy_line_mask(page_img) -> np.ndarray`. Separate heavy from thin strokes by local stroke width (distance transform), relative to the sheet's measured stroke width.
- `labels.regions(page_img, exclude) -> list[Polygon]` (page px): polygonise the connected non-heavy-line regions inside the neatline.
- `labels.assign_regions(regions, labels) -> list[tuple[Polygon, str|None, float]]`:
  - one label → that code;
  - conflicting labels → split the region by a Voronoi of its labels;
  - no label → inherit only if all adjacent regions agree, else None.
- `labels.assign_parcels(parcels, sheet, transform) -> DataFrame` with `blklot`, `code`, `height_code`, `confidence`. Transform the regions to EPSG:2227 via `SheetTransform.to_world`; each parcel takes the region covering the majority of its area.
- Fallback when < 5 closed regions: nearest same-block label at confidence 0.3.

- [ ] Tests:
  - `test_heavy_vs_thin`;
  - `test_regions_closed`;
  - `test_conflicting_labels_split` (Review Focus 4);
  - `test_unlabelled_region_inherits_only_if_agree`;
  - `test_parcel_majority_region`;
  - `test_fallback_nearest_label`.
- [ ] Live: one 1960 sheet and one 2000-digital-zoning sheet. Render the fills over the scan and view them. Commit `feat(districts): label-and-boundary reader`.

### Task D6: Assembly, outputs, CLI, review layer

**Files:** create `districts/assemble.py`; modify `cli.py` (`sfzmaps districts [--series S | --all] [--validate]`), `reviewpage.py` + its template (add a district layer: fill by category, low confidence hatched or outlined), and `share.py` (`SHARED_GLOBS` += `out/*/districts/*`, `work/*/*/district_labels.json`, `overrides/*.csv`); test `tests/test_districts_assemble.py`.

**Interfaces:**
- `assemble.apply_overrides(df, series) -> df`: reads `overrides/<series>.csv` (`blklot|block,code`). Applied before the join. Unknown blklots warn (Review Focus 5).
- `assemble.run_series(series) -> Path`.
  1. Dispatch by `SERIES[series]`.
  2. A `role=height` series writes `out/<period-zoning-series>/districts/height.parquet` and is joined by blklot into the period's zoning output.
  3. Then: overrides → rules join → category.
  4. Writes `parcels.fgb`, `districts.fgb` (dissolve by code), `<series>-districts.pmtiles` (two layers, `ogr2ogr -f PMTiles`, EPSG:3857, minzoom 11, maxzoom 16), and `stats.json` (`{category: {parcels, area_sqft}}`, plus `unknown_share` and `mapped_area_sqft`).
- `today` is published to `out/today/districts/` the same way.

- [ ] Tests:
  - `test_override_applied_before_join`;
  - `test_override_missing_blklot_warns`;
  - `test_height_join_by_blklot`;
  - `test_stats_sum`;
  - `test_pmtiles_written` (skip if ogr2ogr lacks PMTiles).
- [ ] Implement and commit `feat(districts): assembly, outputs, CLI, review layer`.

### Task D7: Full run, validation, answer keys, tuning

**Files:** create `districts/validate.py`, `tests/test_districts_validate.py`, `tests/fixtures/district_keys/{1921,1948,1960,1980}.csv`; modify `docs/superpowers/notes/2026-10-29-tuning.md` (new "Districts" section).

**Interfaces:**
- `validate.agreement(a: DataFrame, b: DataFrame) -> dict`: category and code agreement over the common blklots, plus per-family confusion.
- `validate.key_accuracy(df, key_csv) -> float`.

- [ ] Tests: `test_agreement_math`, `test_key_accuracy`.
- [ ] Answer keys: for each of 1921, 1948, 1960 and 1980, pick ~40 parcels spread across districts. Read their district by eye from the scan (render crops with the parcel outlined, view them with Read), and write the CSV (blklot, code). Do this BEFORE looking at the pipeline's output for those parcels.
- [ ] Run `sfzmaps districts --all` and `--validate`. Targets:
  - 2000-digital-zoning vs today: category agreement ≥ 85%;
  - answer-key accuracy reported per era;
  - unknown share < 10% per era.
  Diagnose and tune any that miss, recording it in the notes.
- [ ] Spot-check images for 1921, 1948, 1960, 1980 and 2000-digital (Sunset, Mission, downtown), viewed by eye.
- [ ] Commit `feat(districts): validation, answer keys, full run`. Report the per-series table: parcels, unknown share, category shares, key accuracy, spend.
