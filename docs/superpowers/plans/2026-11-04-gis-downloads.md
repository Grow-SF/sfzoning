# GIS Downloads and Site Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish every edition's lots, districts, citywide scan image and individual aligned sheets as GIS downloads with a per-file "last updated" date, and split the site into four pages under a top menu.

**Architecture:** `sfzmaps release` (run after a merge, from `main`) bundles the public steps: checks, downloads export, tile and download publishing, and a PR with the updated manifests. A new `sfzmaps downloads` command (Python, pinned GDAL) writes lot/district zips and citywide COGs into `data/downloads/` plus `index.json`, which records each file's source fingerprint and every sheet's DVC md5. A new `site/scripts/publish-downloads.mjs` uploads only files whose fingerprint changed to Blob (`downloads/<fp12>/<readable name>`), checks the shipped sheet objects exist, keeps dates, and writes `site/lib/downloads.json`. The site renders `/data` from that manifest, serves `downloads.json`, the sheet CSVs and `/data/latest/...` redirects, and builds sheet zips in the browser from the shipped DVC objects.

**Tech Stack:** Python 3 + typer + GDAL CLI (pixi); Node ESM + `@vercel/blob` 2.8.2; Next.js 16 App Router, React 19, vitest; `client-zip` (MIT) for browser zips.

**Spec:** `docs/superpowers/specs/2026-11-04-gis-downloads-design.md`

## Global Constraints

- Attribution, verbatim: "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)".
- 1948 imagery credit: `config.RUMSEY_CREDIT` ("David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries").
- Data license: CC BY-NC-SA 4.0 (per `NOTICE`); credit DataSF parcels for lot lines.
- No geometry simplification; no vectorising; sheets are shipped unresampled.
- Never overwrite a Blob file (`allowOverwrite: false`); never delete a `dvc/files/md5/...` object.
- $0: nothing here calls the OpenAI API. Not a DVC stage.
- Remote requests: one listing call per 1,000 names, one request per uploaded file, one batch delete.
- New dependencies must be AGPL-3.0 compatible (MIT/BSD/Apache fine).
- Use the GDAL from `pixi.toml` (`toolchain` checks as in other commands).
- Rollout: build everything now; do not run a real publish or `sfzmaps release` (only `--dry-run`). `/data` and its menu entry appear only when `lib/downloads.json` lists files.
- Draft copy is marked `// DRAFT COPY — NOT REVIEWED BY THE OWNER.` like `app/about/page.tsx`.

## Review Focus

- A sheet under 8 MB lives in a data pack, not alone on Blob: the publish must upload it itself under `downloads/` instead of linking a missing object.
- Re-running `sfzmaps downloads` with nothing changed must rewrite nothing and leave every date alone.
- A browser without the File System Access API (Firefox, Safari) must still get the sheets zip (service-worker stream), not a tab that buffers 0.7 GB in memory.
- Shapefile field names truncate at 10 characters: use the explicit short names below, not GDAL's truncation.
- `/#explore` links from before the split must land on `/maps`.

---

## File Structure

- `pipeline/src/sfzmaps/downloads/__init__.py`: `catalog()`, fingerprints, `index.json`.
- `pipeline/src/sfzmaps/downloads/vectors.py`: lot/district zips.
- `pipeline/src/sfzmaps/downloads/rasters.py`: citywide COG, sheet entries.
- `pipeline/src/sfzmaps/downloads/readme.py` + `templates/download-readme.txt`: README/LICENSE text.
- `pipeline/src/sfzmaps/cli.py`: `downloads` command.
- `pipeline/tests/test_downloads.py`.
- `site/scripts/publish-downloads.mjs`, `site/tests/publish-downloads.test.ts`.
- `site/lib/downloads.json` (starts as `{"files": {}, "sheets": {}, "superseded": []}`), `site/lib/downloads.ts`.
- `site/components/SiteMenu.tsx` + `siteMenu.module.css`; `site/app/layout.tsx`.
- `site/app/maps/page.tsx`; `site/app/page.tsx`; `site/components/Explorer.tsx`.
- `site/app/data/page.tsx`, `site/components/DownloadTable.tsx`, `site/components/downloads.module.css`.
- `site/app/data/downloads.json/route.ts`, `site/app/data/sheets/[series]/route.ts`, redirects in `site/next.config.mjs`.
- `site/lib/sheetZip.ts`, `site/public/download-sw.js`, `site/components/SheetDownload.tsx`.
- `site/tests/downloads.test.ts`.

---

### Task 1: Catalog, fingerprints and README text

**Files:**
- Create: `pipeline/src/sfzmaps/downloads/__init__.py`, `pipeline/src/sfzmaps/downloads/readme.py`, `pipeline/src/sfzmaps/templates/download-readme.txt`
- Test: `pipeline/tests/test_downloads.py`

**Interfaces:**
- Produces:
  - `DOWNLOADS_DIR = config.DATA_DIR / "downloads"`
  - `@dataclass(frozen=True) class Item: key: str; series: str; layer: Literal["lots","districts","citywide"]; fmt: Literal["geojson","gpkg","shp","cog"]; filename: str; sources: list[Path]` where key is `"<series>/<layer>.<fmt>"` and filename is e.g. `1948-lots.geojson.zip`, `1948-districts.shp.zip`, `1948-citywide.tif`.
  - `catalog() -> list[Item]`: vectors for every `SERIES` entry with role `"use"` plus `"today"` that has `out/<s>/districts/parcels.fgb`; `citywide` for every series (any role, including `-su`) with at least one `work/<s>/*/warped_paper.tif`.
  - `fingerprint(item: Item, exporter_version: str) -> str`: sha256 over sorted (relpath, file sha256) of `item.sources` plus `exporter_version`; reuses `stitch._sha256`.
  - `EXPORTER_VERSION = "1"` (bump when output format changes).
  - `readme(series: str, layer: str, updated: str, commit: str) -> str` and `LICENSE_TEXT: str`.

- [ ] **Step 1: Write failing tests**

```python
def test_catalog_lists_three_formats_per_vector_layer(tmp_data):  # fixture builds fake out/1948/districts/*.fgb and work/1948/1/warped_paper.tif
    keys = {i.key for i in downloads.catalog()}
    assert {"1948/lots.geojson", "1948/lots.gpkg", "1948/lots.shp",
            "1948/districts.geojson", "1948/districts.gpkg", "1948/districts.shp", "1948/citywide.cog"} <= keys

def test_su_series_gets_citywide_only(tmp_data_su):
    keys = {i.key for i in downloads.catalog() if i.series == "1980-1990-su"}
    assert keys == {"1980-1990-su/citywide.cog"}

def test_fingerprint_changes_with_source_bytes_and_version(tmp_data): ...  # assert fp differs after rewriting a source and after a different exporter_version; equal on rerun

def test_readme_has_attribution_license_and_rumsey_only_for_1948():
    r = readme("1948", "citywide", "2026-11-04", "abc1234")
    assert "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)" in r
    assert "CC BY-NC-SA 4.0" in r and config.RUMSEY_CREDIT in r and "2026-11-04" in r
    assert config.RUMSEY_CREDIT not in readme("1960", "lots", "2026-11-04", "abc1234")
    assert "not an official record" in readme("1960", "lots", "2026-11-04", "abc1234")
```

- [ ] **Step 2: Run `cd pipeline && uv run pytest tests/test_downloads.py -v`** — expect FAIL (module missing).
- [ ] **Step 3: Implement** the interfaces above. Template sections: what the file is (edition, layer, CRS), field table (lots/districts only, with shapefile names), how lots were assigned (read from scan onto modern DataSF parcels; boundaries are modern lot lines, not traced), what `confidence` means, "a reconstruction, not an official record of the San Francisco Planning Department", last updated + commit, license, attribution, DataSF credit, Rumsey credit when series is 1948. Mark the template file as draft copy for the owner.
- [ ] **Step 4: Run the tests** — expect PASS.
- [ ] **Step 5: Commit** `feat(downloads): catalog, fingerprints and README text`.

### Task 2: Vector exports

**Files:**
- Create: `pipeline/src/sfzmaps/downloads/vectors.py`
- Test: `pipeline/tests/test_downloads.py`

**Interfaces:**
- Consumes: `Item`, `readme`, `LICENSE_TEXT`.
- Produces: `export_vector(item: Item, out: Path, updated: str, commit: str) -> None` writing a zip at `out`.

Field mapping (lots): `blklot, code, height_code, category, confidence, source_sheet, note`; shapefile names `blklot, code, height, category, confidence, src_sheet, note`. Districts: `code, category`. Layer names `lots` / `districts`; the GeoPackage also gets `lots_ca_spcs` / `districts_ca_spcs` in EPSG:2227. GeoJSON: `-lco RFC7946=YES -lco COORDINATE_PRECISION=6`, EPSG:4326. Zip members: data file(s), `README.txt`, `LICENSE.txt`; fixed member timestamps (1980-01-01), sorted entries, DEFLATE.

- [ ] **Step 1: Write failing tests** on a two-polygon synthetic `parcels.fgb` in EPSG:2227:
  - `test_geojson_zip_is_wgs84_with_all_lot_fields`: open with `ogrinfo`/`pyogrio` via `/vsizip/`; fields equal the lot list; coordinates within SF bounds (−122.6..−122.3, 37.6..37.9).
  - `test_gpkg_has_both_crs_layers`: layers `lots` (4326) and `lots_ca_spcs` (2227), same feature count.
  - `test_shapefile_uses_short_names_and_prj`: dbf fields equal the shapefile list; `.prj` member present.
  - `test_zip_contains_readme_and_license`.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** with `ogr2ogr` subprocess calls (`-sql` with `AS` for shapefile renames) into a temp dir, then `zipfile` with fixed `ZipInfo` dates.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(downloads): lot and district exports`.

### Task 3: Citywide image and sheet entries

**Files:**
- Create: `pipeline/src/sfzmaps/downloads/rasters.py`
- Test: `pipeline/tests/test_downloads.py`

**Interfaces:**
- Consumes: `stitch.build_vrt(series, paper=True)`, `stitch.draw_order`, `stitch.load_scores`, `config.RUMSEY_CREDIT`.
- Produces:
  - `export_citywide(item: Item, out: Path, updated: str, commit: str) -> None`: `gdal_translate -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER -co OVERVIEWS=AUTO -mo TIFFTAG_COPYRIGHT=<license + attribution [+ Rumsey]> -mo TIFFTAG_IMAGEDESCRIPTION=<edition, last updated>` from the paper VRT; writes sidecars `<stem>.README.txt` and `<stem>.LICENSE.txt` next to it. `item.sources` for citywide = every placed sheet's `warped_paper.tif` and footprint plus `stitch.py`, so a stitch change refreshes it.
  - `sheet_entries(series: str) -> list[dict]`: one per sheet with `warped_paper.tif` (every placed sheet, not only `layer_sheets`): `{"sheet": id, "name": f"{series}-sheet-{id}.tif", "md5": <md5 hex of file>, "bytes": int, "sha256": str}`, sorted by sheet id.

- [ ] **Step 1: Write failing tests** on two tiny synthetic RGBA `warped_paper.tif` sheets (use helpers in `tests/synthetic.py` if suitable):
  - `test_citywide_is_cog_with_license_tags`: `gdalinfo` shows `LAYOUT=COG`, EPSG 2227, 4 bands, and the attribution in `TIFFTAG_COPYRIGHT`; sidecars exist.
  - `test_sheet_entries_md5_matches_file`: md5 equals `hashlib.md5(bytes)`, name format as above.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(downloads): citywide COG and sheet entries`.

### Task 4: `sfzmaps downloads` command and index

**Files:**
- Modify: `pipeline/src/sfzmaps/downloads/__init__.py`, `pipeline/src/sfzmaps/cli.py` (new command beside `build`), `pipeline/README.md` (commands table; "Shipping the cache": shipped `dvc/files/md5/...` objects back public sheet downloads and are never deleted)
- Test: `pipeline/tests/test_downloads.py`, `pipeline/tests/test_cli.py`

**Interfaces:**
- Produces: `run(series: list[str] | None = None) -> dict` writing `DOWNLOADS_DIR/index.json`:

```json
{"commit": "<git short sha>", "files": {"1948/lots.geojson": {"series": "1948", "layer": "lots", "fmt": "geojson",
  "path": "1948/1948-lots.geojson.zip", "fingerprint": "<sha256>", "bytes": 0, "sha256": "<file sha256>",
  "extra": ["1948/1948-citywide.README.txt"]}},
 "sheets": {"1948": [{"sheet": "10", "name": "1948-sheet-10.tif", "md5": "...", "bytes": 0, "sha256": "..."}]},
 "readme": {"1948": "<sheets README text>"}}
```

  `extra` lists citywide sidecars only. An item whose fingerprint equals the previous `index.json` entry and whose file exists is not rewritten. `updated` passed to exporters is today's date for rewritten items.
- CLI: `sfzmaps downloads [--series S ...]`; refuses tools outside pixi as `build` does; prints one line per item (`same`/`wrote`) and totals.

- [ ] **Step 1: Write failing tests**: `test_rerun_rewrites_nothing` (second `run()` leaves every mtime unchanged and index identical); `test_changed_source_rewrites_only_that_series`; `test_cli_downloads_listed` (command appears in `--help`).
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `uv run pytest`** (whole suite) — expect PASS.
- [ ] **Step 5: Run it for real**: `pixi run uv run sfzmaps downloads --series 1948`; check sizes and open one zip of each kind with `ogrinfo /vsizip/...` and the COG with `gdalinfo`. Report sizes.
- [ ] **Step 6: Commit** `feat(downloads): sfzmaps downloads command`.

### Task 5: Publish script

**Files:**
- Create: `site/scripts/publish-downloads.mjs`, `site/lib/downloads.json`
- Test: `site/tests/publish-downloads.test.ts`

**Interfaces:**
- Consumes: `index.json` (Task 4); `dvcProblem`/DVC check from `publish-tiles.mjs` (export `checkPipelineData` from there and import it).
- Produces `site/lib/downloads.json`:

```json
{"generated": "2026-11-04", "base": "<blob base url>",
 "files": {"1948/lots.geojson": {"series": "1948", "layer": "lots", "fmt": "geojson", "url": "<base>/downloads/<fp12>/1948-lots.geojson.zip",
   "bytes": 0, "sha256": "...", "fingerprint": "...", "updated": "YYYY-MM-DD", "extra": {"README.txt": "<url>"}}},
 "sheets": {"1948": {"updated": "YYYY-MM-DD", "readme": "...", "items": [{"sheet": "10", "name": "1948-sheet-10.tif", "url": "<base>/dvc/files/md5/ab/cdef...", "bytes": 0, "md5": "...", "updated": "YYYY-MM-DD"}]}},
 "superseded": [{"url": "...", "since": "YYYY-MM-DD"}]}
```

- Pure, exported, tested functions:
  - `planDownloads(index, manifest, today) -> {upload: [{key, local, to}], files, superseded}`: unchanged fingerprint keeps url and `updated`; changed gets `downloads/<fp[:12]>/<filename>`, `updated = today`, old url pushed to `superseded` with `since = today`.
  - `planSheets(index, manifest, listing, today) -> {upload, sheets}`: a sheet whose `dvc/files/md5/<2>/<30>` is in `listing` links there; otherwise it uploads to `downloads/sheets/<md5>/<name>`. Per-sheet `updated` changes only when its md5 changes; series `updated` is the latest of its sheets.
  - `expired(superseded, today, days = 30) -> string[]`.
- `main()`: DVC check (same `--dry-run`/`--skip-dvc-check` flags), one `list()` pass over `downloads/` and `dvc/files/md5/` prefixes, `put` each upload (`allowOverwrite: false`, `multipart` over 50 MB, `cacheControlMaxAge` one year), verify by listing size, then one `del(expiredUrls)`, write manifest (after each upload, like `publish-tiles.mjs`). `--dry-run` prints the plan and request count. `--local <base>` writes a manifest whose URLs use `<base>` (the local tile server) and uploads nothing, for checking the page.

- [ ] **Step 1: Write failing tests** for the three pure functions: `keeps url and date when fingerprint unchanged`; `new fingerprint gets new hashed path and today's date and supersedes the old url`; `packed sheet (not in listing) is uploaded under downloads/sheets`; `sheet date unchanged when md5 unchanged`; `expired returns only entries 30+ days old`.
- [ ] **Step 2: Run `cd site && npm test -- publish-downloads`** — expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — expect PASS; then `node --env-file=.env.local scripts/publish-downloads.mjs --dry-run` (read-only listing) and report the plan. Do not run a real publish (rollout).
- [ ] **Step 5: Commit** `feat(site): publish-downloads script`.

### Task 6: Top menu

**Files:**
- Create: `site/components/SiteMenu.tsx`, `site/components/siteMenu.module.css`
- Modify: `site/app/layout.tsx`
- Test: `site/tests/downloads.test.ts` (menu items)

**Interfaces:**
- Produces: `menuItems(hasDownloads: boolean): {href: string; label: string}[]` returning `/` "Story", `/maps` "All Maps", `/data` "Download the Data" (only when `hasDownloads`), `/about` "About".
- `SiteMenu` (client component): site title "A History of San Francisco’s Zoning" linking `/` at left; links at right; `aria-current="page"` on the current one (`usePathname`); below 720px a hamburger `<button aria-expanded aria-controls>` toggles the list; Escape and link clicks close it. Slim (≤ 48px), existing fonts/colours from `globals.css`. Check the story's sticky map and step cards still start below it.

- [ ] **Step 1: Failing test**: `menuItems(false)` has no `/data`; `menuItems(true)` order Story, All Maps, Download the Data, About.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `npm test`, `npm run lint`, `npx tsc --noEmit` — PASS.
- [ ] **Step 5: Commit** `feat(site): top menu with mobile hamburger`.

### Task 7: All Maps page and story home

**Files:**
- Create: `site/app/maps/page.tsx` (metadata title "All maps · A History of San Francisco’s Zoning", canonical `/maps`; renders `<Explorer eager />` and the licence/credits footer)
- Modify: `site/components/Explorer.tsx` (prop `eager?: boolean` skips the near-scroll lazy mount), `site/app/page.tsx` (remove `<Explorer />`; add a short "Explore every map →" link to `/maps` before the footer), `site/components/SiteMenu.tsx` (on `/` with `location.hash === "#explore"`, `router.replace("/maps")`)
- Test: none beyond existing suites (layout change; owner checks by eye).

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Run** `npm test`, `npm run lint`, `npx tsc --noEmit` — PASS; `grep -rn '#explore' site/app site/components site/content` returns only the redirect.
- [ ] **Step 3: Commit** `feat(site): All Maps page`.

### Task 8: Download page, manifest routes, stable links

**Files:**
- Create: `site/lib/downloads.ts`, `site/app/data/page.tsx`, `site/components/DownloadTable.tsx`, `site/components/downloads.module.css`, `site/app/data/downloads.json/route.ts`, `site/app/data/sheets/[series]/route.ts`
- Modify: `site/next.config.mjs`
- Test: `site/tests/downloads.test.ts`

**Interfaces:**
- Consumes: `lib/downloads.json`, `ERAS`.
- Produces (`lib/downloads.ts`):
  - `rows(manifest, eras = ERAS): Row[]`, one per era in ERAS order: `{era, lots: Cell[], districts: Cell[], scans: ScanCell[]}`, `Cell = {fmt, label: "GeoJSON"|"GeoPackage"|"Shapefile", url, bytes, updated}`; `ScanCell = {series, label: "Zoning"|"Height"|"Special use"|"", citywide?: Cell, sheets?: {series, count, bytes, updated}}`. Scan series per era: `era.scanSeries` plus, when it ends `-zoning`, the same period's `-height` and `-su`; for 1921–1980 just the series. Missing files give empty cells.
  - `formatBytes(n): string` ("3.2 MB", "0.7 GB").
  - `lastUpdated(manifest): string` (latest `updated`).
  - `sheetsCsv(manifest, series): string` header `sheet,name,url,bytes,md5,updated`.
  - `latestRedirects(manifest): {source, destination, permanent: false}[]`: `/data/latest/<series>/<filename>` → url, for every file and sheet.
- `/data` page: kicker + h1 "Download the data"; draft intro (who it's for; lots vs districts vs scans; GeoJSON/GeoPackage/Shapefile in WGS84, GeoPackage also in California State Plane; a reconstruction, not an official record); license + attribution + Rumsey + DataSF; "Data last updated <date>"; table from `rows`, each link with size and date; links to `/data/downloads.json` and each sheet CSV. Responsive: stacks per era below 720px. Empty manifest → `notFound()`.
- `downloads.json/route.ts`: `export const dynamic = "force-static"`; returns the manifest.
- `sheets/[series]/route.ts`: static, `generateStaticParams` from manifest; `text/csv` from `sheetsCsv`.
- `next.config.mjs`: `redirects: async () => latestRedirects(manifest)` (import the JSON; keep config plain JS — put `latestRedirects` in a small `.mjs` helper both import if TS import is awkward).

- [ ] **Step 1: Failing tests** with a fixture manifest: `rows` gives 15 rows in ERAS order; 1980–90 row has Zoning/Height/Special use scan cells; today has no scans; `formatBytes(3_182_670) === "3.2 MB"`; `sheetsCsv` header and one line; `latestRedirects` maps `/data/latest/1948/1948-lots.geojson.zip`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `npm test`, lint, tsc — PASS.
- [ ] **Step 5: Commit** `feat(site): Download the Data page`.

### Task 9: Sheet downloads in the browser

**Files:**
- Create: `site/lib/sheetZip.ts`, `site/public/download-sw.js`, `site/components/SheetDownload.tsx`
- Modify: `site/components/DownloadTable.tsx`, `site/package.json` (add `client-zip`, exact current version, MIT; check the license on npm before adding)
- Test: `site/tests/downloads.test.ts`

**Interfaces:**
- Produces:
  - `zipEntries(manifest, series, fetchFn = fetch): AsyncIterable<{name, input, lastModified}>`: `README.txt` and `LICENSE.txt` first (from `manifest.sheets[series].readme`; LICENSE text shared via `lib/downloads.ts`), then each sheet fetched in order as `<name>`.
  - `saveSheetsZip(manifest, series): Promise<void>`: `downloadZip(zipEntries(...))` from `client-zip`; if `window.showSaveFilePicker` exists, pipe to the chosen file; otherwise register `/download-sw.js` (scope `/data/`), `postMessage` the ReadableStream (transferable) with filename and size, and navigate a hidden iframe to the worker's download URL, which responds with `Content-Disposition: attachment; filename="<series>-sheets.zip"`.
  - `saveSheet(item)`: fetch → `Blob` → object URL → `<a download=name>` (one sheet ≤ ~60 MB).
- `SheetDownload`: "All N sheets (X GB)" button with progress (sheets done / total, bytes), cancel, error message; per-sheet list behind a `<details>` with one button per sheet. Also show the CSV link.

- [ ] **Step 1: Failing tests**: `zipEntries` yields README, LICENSE, then sheet names in order and calls `fetchFn` with each sheet url; a failed fetch (status 404) throws an error naming the sheet.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `npm test`, lint, tsc — PASS.
- [ ] **Step 5: Commit** `feat(site): browser-built sheet zips`.

### Task 11: `sfzmaps release`

One command for everything that goes public after a merge, so nothing has to be remembered.

**Files:**
- Create: `pipeline/src/sfzmaps/release.py`
- Modify: `pipeline/src/sfzmaps/cli.py`, `pipeline/README.md` (commands table + a "Releasing" section), `site/README.md` (publishing section points at `sfzmaps release`; the scripts stay runnable alone)
- Test: `pipeline/tests/test_release.py`

**Interfaces:**
- Consumes: `downloads.run()` (Task 4); `site/scripts/publish-tiles.mjs` and `publish-downloads.mjs` (Task 5), run as subprocesses with the env file `ship` uses (`site/.env.local`, or `SFZMAPS_ENV_FILE`); the git helpers in `ship.py`.
- Produces: `sfzmaps release [--dry-run] [--no-pr]`, which in order:
  1. `checks() -> list[str]` (problems; any one stops the release): tree clean; `HEAD` equals `origin/main` after a fetch; DVC data matches `dvc.lock`; this commit's cache is shipped (the same public checks as the `Cache shipped` workflow).
  2. `downloads.run()`.
  3. `publish-tiles.mjs`, then `publish-downloads.mjs` (each with `--dry-run` under `--dry-run`).
  4. If `site/lib/published.json` or `site/lib/downloads.json` changed: commit them on branch `release/<YYYY-MM-DD>-<sha7>`, push, and open a PR (`gh pr create`, title `Release <date>`, body listing what was uploaded), unless `--no-pr`, which only commits. Merging that PR deploys. Nothing changed: print "nothing to release".
  5. Print a summary: files uploaded, bytes, superseded files deleted, PR URL.
- The download page goes live on the first release that lists files (Task 6/8 gating), so the owner's first `sfzmaps release` after the big iteration is the launch.

- [ ] **Step 1: Failing tests** (subprocess and VCS calls injected): `test_release_stops_on_dirty_tree`; `test_release_stops_when_not_at_origin_main`; `test_release_stops_when_cache_not_shipped`; `test_dry_run_passes_dry_run_to_both_scripts_and_commits_nothing`; `test_no_changes_makes_no_branch`.
- [ ] **Step 2: Run** `uv run pytest tests/test_release.py -v` — FAIL. **Step 3: Implement.** **Step 4: Run** the whole suite — PASS.
- [ ] **Step 5:** `uv run sfzmaps release --dry-run` from this worktree should stop at "HEAD is not origin/main" (expected); report the output.
- [ ] **Step 6: Commit** `feat: sfzmaps release`.

### Task 12: End-to-end check (no publish)

- [ ] **Step 1:** `pixi run uv run sfzmaps downloads` for all series in parallel batches (`--series` per group); report total sizes.
- [ ] **Step 2:** Write a local test manifest with `publish-downloads.mjs --local http://127.0.0.1:<tiles port>` (serving `data/downloads` and the DVC cache objects through `npm run tiles`), start `npm run dev`, open `/`, `/maps`, `/data`, `/about` at desktop and 375px width; download one lots zip, one citywide image, one sheet and a sheets zip in Chrome and Firefox; open each in QGIS-compatible `ogrinfo`/`gdalinfo`. Screenshot the menu (both widths) and `/data` for the owner.
- [ ] **Step 3:** Restore `site/lib/downloads.json` to the empty manifest; run all suites (`uv run pytest`; `npm run lint`, `npx tsc --noEmit`, `npm test`).
- [ ] **Step 4:** Commit; open the PR describing what changed on the site for the owner to check by eye, and that the first real publish waits for the owner.
