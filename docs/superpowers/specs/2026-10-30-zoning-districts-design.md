# Zoning district overlays and district rules: design (step two)

Date: 2026-10-30
Status: written autonomously overnight at the owner's request ("use your best
judgment… if you make a mistake… we can just fix it later"). Every decision
below that the owner didn't make is marked **[call]**.

## Why

Step one aligned every archived zoning map sheet (1921–2000s) to real-world
coordinates and published the scans as web tiles (`pipeline/data/out/<series>/`).
The story on sfzoning.com needs more than pictures of scans. For each era it
needs to show **what was allowed where**: which parcels allowed apartments,
which allowed only single-family houses, and where commerce and industry could
go, in clean colours a reader can compare across decades. The owner's
priorities from step one still hold: zoning boundaries must be correct
(10/10), and they come from snapping to modern lot lines, not from tracing the
scan.

## Goals

1. For each zoning-map era, a **parcel-level district assignment**: every
   modern parcel gets the district code the historic map shows for it, with a
   confidence value. The parcels are then dissolved into **district polygons**
   that follow modern lot lines.
2. A **rules table** joining each (era, district code) to plain-language
   categories: `allowed_uses`, `max_units_per_lot`, `height_limit_ft`, and the
   other fields in the research table
   (`docs/research/zoning-districts.csv`, built tonight from primary sources).
3. Derived **story categories** per parcel per era, used for map colour:
   - single-family only;
   - 2-unit (flats/duplex);
   - multi-family / apartments allowed;
   - commercial (housing allowed or not, per era);
   - industrial (housing allowed or not);
   - public / open space;
   - unrestricted;
   - unknown.
4. Outputs the website can use directly: per-era **vector PMTiles** (parcels
   with code + category + confidence, plus dissolved districts) and a compact
   per-era stats JSON (parcel counts / land area by category) for charts.
5. **Validation against ground truth**: today's zoning comes straight from
   DataSF (`3i4a-hu95`, Zoning Map – Zoning Districts). The most recent scanned
   era (2000-digital-zoning) is classified by the same pipeline and compared
   against DataSF's current zoning, reporting per-parcel agreement. Expect high
   but not perfect agreement, since zoning changed after the 2000s maps.
6. Review: the step-one review page gains a district layer (colour by code,
   with low-confidence parcels highlighted). Corrections go into a small
   override file, `pipeline/data/overrides/<series>.csv` (parcel `blklot` or
   block → code), which reruns apply. No new editor UI **[call]**.

## Non-goals

- The website itself (separate spec; built after this).
- Lot lines as they existed historically. We use today's parcels (DataSF
  `acdm-wktn`, active) as the unit **[call]**. Where historic lots were
  different, the assignment follows the map at today's lot, which is fine for
  the story.
- Special-use / height-and-bulk overlay semantics beyond extracting the
  codes. Height districts (1972+) are extracted as a second code field where
  the map series is a height map, but they aren't interpreted beyond the
  research table **[call]**.

## How it works

### Inputs
- Warped scans per sheet: `work/<series>/<sheet>/warped.tif` (RGBA, EPSG:2227,
  transparent paper), plus `transform.json`, `page.json` (with legend
  `mask_boxes`) and `page.png`.
- Modern parcels: `ref/parcels.parquet` (active). Blocks: `ref/` as in step one.
- Research rules: `docs/research/zoning-districts.csv`.
- DataSF current zoning (fetch into `ref/zoning_current.parquet`).

### Two drawing styles, two readers [call]
The maps encode districts in one of two ways. The pipeline chooses the reader
per series from config:

**A. Pattern maps (1921, 1929, 1935, 1948).** The district is the fill pattern
inside each block or lot: white, hatched, solid black, stippled,
cross-hatched or checkered, per each era's legend. There are no per-block
labels.
- **Reader:** for each parcel, crop the warped scan inside the parcel polygon
  (shrunk inward ~3 ft so boundary strokes don't dominate) and classify the
  texture.
- **Method [call]:** the vision model (`gpt-6.1-sol`, cached through
  `llm.ask_json`, as in step one) classifies **batched mosaics**. Each request
  is one image containing the era's legend swatches (cropped from the legend
  `mask_boxes`) plus a numbered grid of ~48 parcel crops. It returns
  `{index: code, confidence}`.
- **Batching cost:** classify blocks first. A block whose parcels are visually
  uniform (low variance of simple texture statistics, i.e. ink fraction and
  dominant gradient orientation, across its parcels) is sent as one crop; only
  mixed blocks send per-parcel crops. Estimated cost is well under $50 for all
  pattern eras.
- Respect the minimise-image-tokens rule from step one: crops downscaled to the
  smallest size that preserves the pattern (calibrate once per era, as with
  labels).

**B. Label-and-boundary maps (1960 onward).** Districts are regions bounded by
heavy lines, each carrying a printed code (R-1, C-2, RH-2, 40-X…).
- **Reader:**
  1. **District labels:** a new vision prompt reads district codes (and, on
     height maps, height-district codes) with pixel positions from page tiles,
     reusing the step-one tiling, cache and `read_labels` machinery with a
     different prompt and schema. The labels map to EPSG:2227 through the
     sheet's transform.
  2. **District boundary mask:** in the warped scan, separate heavy district
     lines from thin block and lot lines by stroke width (distance transform);
     the step-one linework code already measures stroke and street widths.
     Polygonise the regions enclosed by heavy lines.
  3. **Assign:** each region takes the label(s) inside it. If labels conflict,
     the region splits by nearest label within it. A region with no label
     inherits from its neighbours only when they agree; otherwise it's
     `unknown`.
  4. **Parcels:** each parcel takes the region covering the majority of its
     area; the confidence reflects that majority share and label agreement.
- **Fallback [call]:** where boundary extraction fails on a sheet (too few
  closed regions), assign each parcel the nearest district label within the
  same block, else the same street-bounded area, at low confidence.

### Sheets, overlaps and review status
- Multiple sheets can cover a parcel. Use the same priority as the step-one
  stitch (passed over needs_review, then lower held-out error).
- Parcels from sheets that aren't placed (no transform) are `unknown`.
- Parcels from `needs_review` sheets are kept, but their confidence is capped
  at "medium" **[call]**.

### Rules join and story category
- Join (era, code) to the research table by the code and the era range that
  contains the series year.
- Map rules to the story categories with a small, explicit mapping table
  (`pipeline/src/sfzmaps/districts/categories.py`), unit-tested on known codes:
  - 1921 "1R" → single-family;
  - 1921 "2R" → multi-family;
  - 1960 "R-1" → single-family;
  - 1960 "R-2" → 2-unit;
  - "R-3".."R-5" → multi-family;
  - "C-x" → commercial;
  - 1978+ "RH-1" → single-family;
  - "RH-2" → 2-unit;
  - "RH-3" → 3-unit;
  - "RM-x" → multi-family;
  - "NC-x" → commercial (housing allowed);
  - "M-x" → industrial;
  - "P" → public.
- Codes not in the research table → category `unknown`, listed in a report
  for follow-up.

### Outputs (per series)
- `out/<series>/districts/parcels.fgb`: one row per parcel, with `blklot`,
  `code`, `height_code`, `category`, `confidence`, `source_sheet`.
- `out/<series>/districts/districts.fgb`: dissolved by `code`.
- `out/<series>/districts/<series>-districts.pmtiles`: vector tiles with both
  layers (via `ogr2ogr -f PMTiles` or tippecanoe if installed) **[call:
  ogr2ogr first]**.
- `out/<series>/districts/stats.json`: parcels and land area by category,
  plus the unknown share.
- `out/today/…`: the same, from DataSF current zoning.
- The step-one `bundle` shares these too (add to `SHARED_GLOBS`).

### Validation (required before calling it done)
1. **2000-digital-zoning vs DataSF today:** the share of parcels with the same
   code family, overall and by family. Target ≥ 85% **[call]**. Below that,
   diagnose before shipping.
2. **Visual spot checks** of 1921, 1948, 1960, 1980 and 2000-digital:
   - render district fills over the scan for 2–3 neighbourhoods each (e.g.
     Sunset, Mission, downtown);
   - check them by eye, with images saved and viewed.
3. **Unknown share per era** reported, aiming for < 10% of parcels inside each
   era's mapped area.

### CLI
- `sfzmaps districts --year Y` and `--all`, which read, assign, join,
  publish and update the review page.
- `sfzmaps districts --validate`.

## Constraints

- Same secrets and caching rules as step one: OpenAI via `pipeline/.env`,
  never logged; every vision response cached; image tokens minimised.
- The owner is the only operator. No multi-user features.
- Same test standards: TDD, synthetic fixtures for the readers (a synthetic
  pattern map and a synthetic label-and-boundary map built like
  `tests/synthetic.py`), plus real-crop regression fixtures.

## Builder-check questions, settled [all calls]

1. **Series → reader → role.** The CLI takes `--series S` (and `--all`), not
   `--year`.

   | Series | Reader | Role |
   |---|---|---|
   | 1921, 1929, 1935, 1948 | A (pattern) | use districts |
   | 1960, 1964, 1968, 1972, 1975, 1980 | B (labels) | use districts; also extract height codes (SHD, 40-X…) where printed |
   | 1980-1990-zoning, 1990-1996-zoning, 1996-2000-zoning, 2000-digital-zoning | B | use districts |
   | 1980-1990-height, 1990-1996-height, 1996-2000-height, 2000-digital-height | B | height_code only |
   | *-su (special-use) | skipped | out of scope for now |

   - A height series is joined onto its period's zoning series by `blklot`, so
     each period publishes ONE parcels layer with `code` + `height_code`.
   - The reader-per-series map is a config table. Before running, the builder
     verifies each assignment by looking at one legend and sheet per series.
     If a "pattern" series actually prints codes, it switches to B.
   - 1935 uses the research CSV's 1929–1946 era.
2. **Mid-block splits (1921-era depth rules).** Classification is per parcel,
   so depth splits follow lot lines naturally. The block-uniform shortcut only
   applies when the per-parcel texture statistics are uniform, so a split block
   is never collapsed.
3. **The research CSV is the source of truth for categories.**
   - The story categories gain **mixed (housing + commercial)**: NC, RC, NCT,
     MUR, UMU and similar map there, and spec examples that say otherwise
     yield to the CSV.
   - "special" rows (height letters, SHDs) feed `height_code`, not `category`.
   - The builder writes a code normaliser with an alias table for compound CSV
     codes ("R / T / V", "Named NCDs"…). Unmatched codes → `unknown`, listed in
     a report.
4. **Validation metric.**
   - The primary measure is **story-category agreement** between
     2000-digital-zoning and DataSF today (target ≥ 85%).
   - Exact-code agreement is reported as secondary information.
   - The family and category tables are written by the builder, documented,
     and tested.
5. **Earlier-era ground truth.** The builder hand-labels a small answer key:
   ~40 parcels each for 1921, 1948, 1960 and 1980, read visually from the
   scans, stored as `tests/fixtures/district_keys/<series>.csv`, and reports
   accuracy against it.
6. **Model and budget.** `gpt-6.1-sol` is wired into `llm.py`. No hard cap
   (the owner waived cost gates). Log the call count and estimated spend.
7. **Overrides** are applied right after reading and before the rules join, so
   they survive reruns.

## Left to the builder

The exact texture statistics, crop sizes, batching, stroke-width thresholds,
region-split rule, confidence formula and the exact PMTiles tooling, all
validated on real sheets.
