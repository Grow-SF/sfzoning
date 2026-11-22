# Downloads: separate data and text change checks

2026-11-18. Status: approved by the owner. Follow-up to `2026-11-18-gis-download-names-design.md` (PR #65).

## Why

Editing README text re-ran `gdal_translate` on every citywide image and `ogr2ogr` on every vector
file, because each download has one change check covering all exporter code. The citywide image also
embedded "last updated <date>" in a TIFF tag, so any rebuild changed its bytes and hash, which under
the versioned names would mean "new versions" of identical pixels.

## Goals

- A README/LICENSE-only change never runs GDAL or ogr2ogr.
- Rebuilding the same data gives byte-identical images.
- No new naming or versioning concepts.

## Non-goals

- Separate versions for README files. (Owner: README-only changes will be rare; when one happens,
  the citywide image and its README publish together as a new version.)
- Reusing files from before this change: the first export after it builds everything once.

## Approach (one sensible way)

**Two checks per item**, both stored in `index.json`:
- `data_fingerprint`: the item's sources, the code that writes the data (image or GIS layers), and
  `EXPORTER_VERSION`.
- `fingerprint`: `data_fingerprint` plus the README/LICENSE code (template, `readme.py`, and the code
  that assembles the zip/sidecars).

**What runs:**
- `data_fingerprint` changed (or output missing): full rebuild with GDAL/ogr2ogr, as today.
- Only `fingerprint` changed, citywide: rewrite `<stem>.README.txt` and `<stem>.LICENSE.txt`; the
  `.tif` is left as is.
- Only `fingerprint` changed, lots/districts: rebuild the zip from the existing zip's data members
  plus the new README and LICENSE; no ogr2ogr.
- Neither changed: nothing (as today).

**Code layout:** the data-writing code and the text/packaging code live in separate files, so the
data check names only the data files. *(Builder's call: exact split, e.g. `rasters.py` image only,
`vectors.py` layers only, packaging/sidecars in the existing `__init__`/a new `package.py`.)*

**No date in the image:** `TIFFTAG_IMAGEDESCRIPTION` becomes `"<series> edition"`;
`TIFFTAG_COPYRIGHT` unchanged. The date lives in the README beside it. *(My call, owner approved.)*

**`made` date:** the day the item's files last changed (either check), as today.

**Publishing (`site/scripts/publish-downloads.mjs`):** a citywide entry is unchanged only if the
image and both sidecars have the same SHA-256 as the manifest; otherwise all three are versioned
anew under the new `made` date (same image bytes are fine). Vector zips are versioned by their own
SHA-256 as today.

## Constraints

- `downloads` stays outside `build.CODE_EXCLUDE`-tracked build code: `dvc.lock` unchanged.
- The determinism goal is checked, not assumed: two exports of the same data give the same image
  SHA-256 (a test with a small fixture raster).
- Checks: `cd pipeline && uv run pytest`; site `npm test`, `npx tsc --noEmit`, `npm run lint`.
