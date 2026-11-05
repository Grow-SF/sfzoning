# GIS downloads and site navigation

Status: design, 2026-11-04.

## Intent

Hobbyists, researchers and government workers should be able to take the reconstructed historic
zoning as real GIS data, not only look at it on the site: the lot-by-lot zoning, the district
shapes, and the aligned scans, in the formats their tools open. Files change when the pipeline
improves, so every file carries the date it last changed and consumers can tell when to update.

At the same time the site becomes four pages with a shared top menu.

## Goals

1. Per edition, vector downloads in three formats:
   - **Lots**: every parcel with `blklot`, `code`, `height_code`, `category`, `confidence`,
     `source_sheet`, `note` (the fields of `out/<series>/districts/parcels.fgb`).
   - **Districts**: one shape per `code`+`category` (the fields of `districts.fgb`).
   - Formats: GeoJSON (RFC 7946, WGS84, 6 decimals), GeoPackage (WGS84 layer plus the same layer
     in EPSG:2227, California State Plane zone 3 ftUS, the city's own system), zipped ESRI
     Shapefile (WGS84, `.prj` included).
2. Aligned GeoTIFFs of the scans (owner request).
3. A README and license file in every zip; license and attribution on the download page.
4. A "last updated" date for every file, on the page and in a machine-readable manifest.
5. Files hosted on Vercel Blob (owner's call, instead of GitHub).
6. Site: top menu on every page — Story (`/`), All Maps (`/maps`), Download the Data (`/data`),
   About (`/about`) — collapsing to a hamburger on mobile.

## Non-goals

- No new data: exports are conversions of what the pipeline already publishes.
- No geometry simplification, no vectorising of scans (AGENTS.md faithfulness rules).
- No API or query service; static files only.

## Anti-goals

- A download that disagrees with the map on the site (exports must come from the same build
  `dvc.lock` records, never a hand run).
- A published file overwritten in place (breaks links and open downloads; AGENTS.md).
- Losing the attribution, the David Rumsey credit on 1948 imagery, or the CC BY-NC-SA notice.
- A build that calls the OpenAI API (exports are $0, like every build stage).

## Editions

The fifteen eras in `site/lib/eras.ts` (1921 … 2000s, today). Vectors for every era with a
`districtSeries`. GeoTIFFs for every era with scans; the height and special-use series of
1980–2000 get GeoTIFFs too (they are scans with no vectors of their own; their heights are already
in the lot files' `height_code`). Today has no scan.

## The files

### Vectors

Sizes measured on 1948: lots ≈ 7 MB per format compressed (117 MB raw GeoJSON), districts ≈ 3 MB
zipped GeoJSON, 8 MB zipped shapefile. Every download is a zip (GeoJSON and GeoPackage too), so
each one carries its README and LICENSE.

### GeoTIFFs

- **One citywide image per edition** (owner): a Cloud-Optimized GeoTIFF of the series' paper
  layer, the same stitch the site shows as "the original map" (`<series>-paper.vrt`: backing,
  sheet priority, erased border rules). RGBA, lossless DEFLATE, EPSG:2227, with overviews; a bare
  `.tif` so GIS tools can read it over HTTP, with its README and LICENSE as sidecar files
  (`<name>.README.txt`, `<name>.LICENSE.txt`) and the attribution and license also written into the
  TIFF's metadata tags. Resolution (my call): the edition's finest sheet, as the site's VRT
  already uses (`-resolution highest`), so no sheet loses detail; coarser sheets are resampled up
  with no detail added, exactly as on the site.
- **Every individual sheet** (owner), with no second copy (owner): each sheet's
  `warped_paper.tif` as the pipeline wrote it (native resolution, unresampled) is already on Blob,
  stored alone by `sfzmaps ship` at its public DVC path (`dvc/files/md5/…`; objects over 8 MB are
  never packed). Downloads point straight at those objects:
  - one sheet: `/data/sheets/<series>/sheet/<id>`, a site route that streams the object from Blob
    as an attachment named `<series>-sheet-<id>.tif`;
  - all sheets of a series: `/data/sheets/<series>/zip`, a site route that builds
    `<series>-sheets.zip` while it streams (README, LICENSE, then each sheet fetched in turn), with
    an exact Content-Length, so the browser shows an ordinary download (owner, 2026-11-04: the first
    version built the zip in the page, and readers couldn't tell it was downloading);
  - for scripts: a per-series sheet list, `<series>-sheets.csv` (sheet, readable name, URL, size,
    md5, last updated), so `wget`/`curl` can fetch them all.
  - Those cache objects must never be deleted: they back public downloads. Write that into
    `pipeline/README.md` next to the existing never-overwrite rule. The publish fails if a listed
    object is missing on Blob (checked with one listing call).
  - The README for a sheets zip comes from the manifest, with the same text as the pipeline's
    READMEs (shared template).
- Every scanned series gets its downloads, including the height and special-use series of 1980–90,
  1990–96, 1996–2000 and 2000. In the table their row's GeoTIFF cell lists Zoning · Height ·
  Special use. For the 1964–1980 books, which bind height and special-use sheets in with the
  zoning, the sheet downloads hold every sheet the pipeline placed, not only the paper layer's.
- The 1948 READMEs and the page carry the David Rumsey Map Collection credit.
- Size: measured at export and shown on the page (1948's sheets are about 0.7 GB).

### README and LICENSE in every zip

- What the file is (edition, layer, fields with meaning, CRS).
- How lots were assigned (read from the scan onto modern DataSF parcels; boundaries are modern
  lot lines, not traced from the scan); what `confidence` means; that it is a reconstruction, not
  an official record of the Planning Department.
- Last updated date and the source commit.
- License: CC BY-NC-SA 4.0 (content and data, per `NOTICE`); attribution text "Based on
  sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)"; Rumsey credit where 1948 imagery
  is included; DataSF parcels credit.

## Dates and updates

- Each file's **last updated** is the date its content last changed, not the date of the last
  publish: the publish step compares a content hash of the file's *data* (not the zip bytes, which
  carry timestamps) with the previous manifest and keeps the old date when it is unchanged.
- `downloads.json`, published with the files: for every file its edition, layer, format, size,
  sha256, last updated, and URL; plus the manifest's own generated date. Consumers poll it.
- The page shows the date beside each file and a "data last updated" date at the top.
- Stable links (my call): `/data/latest/<edition>/<file>` on the site redirects to the current
  Blob URL, so scripts can use one URL forever and the Blob files stay immutable.

## Pipeline

- A new `sfzmaps downloads` command writes everything into `data/downloads/` (outside `out/`)
  from `out/<series>/districts/*.fgb`, `work/<series>/*/warped_paper.tif` and the paper VRT (rebuilt by the stitch code if absent), using the GDAL pinned in
  `pixi.toml`. Zips are written deterministically (fixed timestamps, sorted entries) so unchanged
  data gives unchanged bytes.
- It is not a DVC stage (my call): its outputs are conversions of data `dvc.lock` already records,
  so caching them would only make `sfzmaps ship` and `pull` move another gigabyte per change.
  The publish step runs it, after the same DVC check that guards tile publishing, so every
  download comes from the recorded build. It is $0, and per series it redoes only what changed
  (vectors when the `.fgb` files changed, a citywide image or sheet list when its sheets changed).
- Rollout (owner): build it now; the first upload of downloads, and the Download page going
  live, wait until the owner finishes the current big iteration.
- Tests: field set and CRS of each output, deterministic zips, README contents (attribution,
  Rumsey credit for 1948).

## Publishing

- `site/scripts/publish-tiles.mjs` (or a sibling script sharing its code) uploads the downloads to
  Blob under content-hashed names, never overwriting, with the same DVC check; it writes
  `site/lib/downloads.json` (committed, deployed) and publishes `downloads.json` to Blob.
- **Only changed files upload** (owner). Every publish uploads exactly the files whose data
  changed, GeoTIFFs included; unchanged ones keep their URL and date. No batching or hold: the
  owner will turn downloads on after the current big iteration, when the scans rarely change.
  Downloads are made only in the publish (not in every build), so ordinary builds and PRs never
  pay for them. Sheets are never uploaded by the publish; they are already shipped.
- **Old versions** (my call): a superseded file this publish uploaded (never a cache object) stays on Blob for 30 days, so downloads in
  progress and old manifests keep working, then the publish deletes it (one batch delete, driven
  by the manifest's history). The stable `/data/latest/...` links always point at the current one.
- Upload size: ~15 × (lots + districts) × 3 formats ≈ 0.5 GB plus the citywide images on the first publish; after that only what changed. Large files use
  multipart upload. Uploads are one request per file plus a single list call to find what exists
  (minimise remote requests).

- **`sfzmaps release`** (owner): one command after a merge, from `main`, that runs the checks,
  the downloads export, the tile publish and the downloads publish, then opens a PR with the
  updated manifests (merging it deploys). The scripts stay runnable on their own.

## Site

- **Top menu** in `app/layout.tsx`: site title at left, the four links at right; under a
  hamburger below a mobile breakpoint; current page marked. Existing fonts and colours. On the
  story page it stays slim and must not cover the map steps.
- **Story** (`/`): the scrolling story; ends with a link to All Maps instead of the embedded
  explorer. The footer (draft note, licenses, sources, credits) stays.
- **All Maps** (`/maps`): the current `Explorer` component on its own page. `/#explore` links keep
  working (client redirect from the hash to `/maps`).
- **Download the Data** (`/data`): short intro (who it's for, lots vs districts vs GeoTIFF, it's a
  reconstruction), license and attribution, "data last updated", then a table with one row per
  edition: Lots (GeoJSON · GeoPackage · Shapefile), Districts (same), Scans (citywide GeoTIFF · sheets zip), each with size
  and last updated date; a link to `downloads.json`. Built statically from `lib/downloads.json`.
- **About** (`/about`): the current "How the maps were made" page, unchanged apart from the menu.
- No browser tests exist; the owner checks the pages by eye. Unit tests cover the table built
  from the manifest and the redirect lookup.

## Left to the builder

- Exact menu styling and breakpoint, page copy for the download intro (marked draft for the
  owner).
- Whether the downloads publish shares `publish-tiles.mjs` or is a sibling.
- Zip layout inside the browser-built sheet zips (folder per sheet or flat).
