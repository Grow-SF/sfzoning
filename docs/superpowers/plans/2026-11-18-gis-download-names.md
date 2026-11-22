# GIS Download Names and Versions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every GIS download gets a permanent, readable, versioned link on sfzoning.com plus a `latest` link, nothing published is ever deleted, and every release is listed on static versions pages.

**Architecture:** Files live on Vercel Blob at `downloads/<series>/<versioned name>`. One site route, `/data/files/[series]/[name]`, redirects a versioned name to Blob by rule and a `latest` name to the current versioned name (from `lib/downloads.json`), and streams the `latest` sheet zip. The publish script writes the current manifest, an append-only `lib/download-versions.json`, and `lib/sheet-numbers.json`; versions pages are static, generated from the history at deploy.

**Tech Stack:** Python pipeline (`sfzmaps downloads`, pytest), Next.js App Router (TypeScript, vitest), Node publish script with `@vercel/blob` (`put`, `copy`, `list`).

**Spec:** `docs/superpowers/specs/2026-11-18-gis-download-names-design.md` (read it first).

## Global Constraints

- Versioned name: `sfzoning-<series>-<layer>-<YYYY-MM-DD>-<hash8>.<ext>`; latest: `sfzoning-<series>-<layer>-latest.<ext>`. `<hash8>` = first 8 hex of the file's own SHA-256.
- Layers: `lots`, `districts`, `citywide`, `sheet-<n>`, `sheets` (zip, latest only). Exts: `geojson.zip`, `gpkg.zip`, `shp.zip`, `tif`, `README.txt`, `LICENSE.txt` (citywide sidecars, image's date and hash), `zip` (sheet zip); any dated file + `.sha256`.
- Sheet `<n>`: `p<NN>` when the sheet ID ends `-p<NN>`, else two-digit position in natural ID order; once given, fixed to its ID forever, never reused.
- Blob path by rule: `downloads/<series>/<name>`. Never overwrite (`allowOverwrite: false`), never delete.
- Dates ISO `YYYY-MM-DD` everywhere.
- Versioned redirect: 308; latest redirect: 307, not cached past a publish (`cache-control: no-store` is fine).
- No download bytes through a function, except the `latest` sheet zip (built while streaming, as today).
- README section (exact text, after LAST UPDATED):
  ```
  GETTING THE NEWEST VERSION
  For the newest version of this file, use:
    https://sfzoning.com/data/files/<series>/<latest name>
  Every version released so far:
    https://sfzoning.com/data/versions/<series>/<layer>
  ```
  The sheet zip's README: same heading, the zip's latest link, and `https://sfzoning.com/data/versions/<series>`.
- Page note (exact): "Use the latest links to always get the newest data. A dated link always returns exactly that version."
- Requests: one listing pass per publish; uploads sequential; no per-file HEAD/GET requests.
- License and credit text unchanged. `--local` manifests stay local; production never points at local files.
- Checks: `cd pipeline && uv run pytest`; in `site/`: `npm run lint`, `npx tsc --noEmit`, `npm test`.

## Review Focus

1. A request for a malformed name, an unknown series, `..` or an encoded `/` must 404 and never redirect to an arbitrary Blob path. (Task 4)
2. Re-running a publish after it failed partway must skip names already on the store, add no duplicate history entries, and write no manifest until every upload is confirmed by the listing. (Task 3)
3. A publish with nothing changed must upload nothing, add no history entries and leave every committed file byte-identical. (Task 3)
4. A sheet added or removed in a series must leave every other sheet's number unchanged; a removed sheet's number is never reused, and its `latest` link 404s. (Tasks 2, 3)
5. Two releases of the same file on the same day differ only by hash; both stay in the history, and `latest` and "(latest)" follow the newer one (history order, not date). (Tasks 3, 5)

---

### Task 1: Pipeline: README section and `made` date

**Files:**
- Modify: `pipeline/src/sfzmaps/downloads/readme-template.txt`, `pipeline/src/sfzmaps/downloads/readme.py`, `pipeline/src/sfzmaps/downloads/__init__.py`, `pipeline/src/sfzmaps/downloads/vectors.py:58`, `pipeline/src/sfzmaps/downloads/rasters.py:41`
- Test: `pipeline/tests/test_downloads.py`

**Interfaces:**
- Produces: `latest_name(series: str, layer: str, fmt: str) -> str` in `sfzmaps/downloads/__init__.py` (`fmt` in `geojson|gpkg|shp|cog`, ext from `_EXT`); `readme(series, layer, updated, commit, fmt: str | None = None) -> str` (`fmt` required unless `layer == "sheets"`); each `index.json` `files` entry gains `"made": "YYYY-MM-DD"` (kept unchanged when the entry is reused).

- [ ] **Step 1: Write the failing tests**

```python
def test_latest_name():
    from sfzmaps.downloads import latest_name
    assert latest_name("1948", "lots", "gpkg") == "sfzoning-1948-lots-latest.gpkg.zip"
    assert latest_name("1921", "citywide", "cog") == "sfzoning-1921-citywide-latest.tif"

def test_readme_points_to_latest_and_versions():
    t = readme_mod.readme("1948", "lots", "2026-11-20", "abc1234", "gpkg")
    assert ("GETTING THE NEWEST VERSION\nFor the newest version of this file, use:\n"
            "  https://sfzoning.com/data/files/1948/sfzoning-1948-lots-latest.gpkg.zip\n"
            "Every version released so far:\n  https://sfzoning.com/data/versions/1948/lots\n") in t
    assert t.index("LAST UPDATED") < t.index("GETTING THE NEWEST VERSION") < t.index("LICENSE\n")

def test_sheets_readme_points_to_latest_zip_and_series_versions():
    t = readme_mod.readme("1921", "sheets", "2026-11-20", "abc1234")
    assert "https://sfzoning.com/data/files/1921/sfzoning-1921-sheets-latest.zip" in t
    assert "https://sfzoning.com/data/versions/1921\n" in t

def test_index_records_made_date_and_keeps_it(fake_exports, tmp_data, monkeypatch):
    # first run on day A writes made=A; a rerun on day B with nothing changed keeps made=A
```

- [ ] **Step 2:** Run `cd pipeline && uv run pytest tests/test_downloads.py -q`. Expected: the four new tests fail.
- [ ] **Step 3:** Add `{{NEWEST}}` to the template after the LAST UPDATED block; fill it in `readme()`; pass `item.fmt` from `vectors.py` and `rasters.py`; add `latest_name`; write `"made": updated` in `run()`'s new-entry dict. Update existing tests that call `readme()` for vector/citywide layers to pass `fmt`.
- [ ] **Step 4:** Run `uv run pytest -q` (whole suite). Expected: all pass; `test_fingerprint_covers_the_exporter_code` still passes (the template is in `EXPORTER_CODE`).
- [ ] **Step 5:** Confirm `dvc.yaml`/`dvc.lock` are unchanged (`git status pipeline/dvc.*` clean: `downloads` is in `build.CODE_EXCLUDE`). Commit: `pipeline: downloads READMEs link latest and versions; index records made date`.

### Task 2: Site: names and sheet numbers

**Files:**
- Create: `site/lib/downloadNames.mjs` (plain ESM, imported by the publish script and the site), `site/lib/downloadNames.d.mts` if lint/tsc needs types
- Test: `site/tests/download-names.test.ts`

**Interfaces:**
- Produces:
  - `EXTS`: `{ geojson: "geojson.zip", gpkg: "gpkg.zip", shp: "shp.zip", cog: "tif" }`
  - `versionedName({ series, layer, date, sha256, ext }) -> string`
  - `latestName({ series, layer, ext }) -> string`
  - `parseName(series, name) -> { layer, ext, latest: boolean, date: string | null, hash8: string | null, checksum: boolean } | null` (null for anything not matching the global name rules for that series; `checksum` = ends `.sha256`)
  - `blobPath(series, name) -> "downloads/<series>/<name>"`
  - `assignSheetNumbers(ids: string[], known: Record<string, string>) -> Record<string, string>`: id → number for every id; keeps every `known` entry (including ids no longer present); new `-pNN` ids get `pNN`; other new ids get the next two-digit number after the largest numeric one in `known` (positions assigned in natural ID order when `known` is empty)

- [ ] **Step 1: Write the failing tests**

```ts
expect(versionedName({ series: "1948", layer: "lots", date: "2026-11-04", sha256: "3f9a2c1b" + "0".repeat(56), ext: "gpkg.zip" }))
  .toBe("sfzoning-1948-lots-2026-11-04-3f9a2c1b.gpkg.zip");
expect(latestName({ series: "1980-1990-zoning", layer: "sheet-p22", ext: "tif" })).toBe("sfzoning-1980-1990-zoning-sheet-p22-latest.tif");
expect(parseName("1921", "sfzoning-1921-citywide-2026-11-04-ce47cd18.README.txt")).toMatchObject({ layer: "citywide", ext: "README.txt", date: "2026-11-04", hash8: "ce47cd18", latest: false });
expect(parseName("1921", "sfzoning-1921-lots-2026-11-04-ce47cd18.gpkg.zip.sha256")?.checksum).toBe(true);
for (const bad of ["../x", "sfzoning-1948-lots-latest.gpkg.zip", "sfzoning-1921-lots-2026-13-04-ce47cd18.gpkg.zip", "sfzoning-1921-lots-2026-11-04-CE47CD18.gpkg.zip", "sfzoning-1921-lots-latest.exe", "sfzoning-1921-sheets-2026-11-04-ce47cd18.zip"])
  expect(parseName("1921", bad)).toBeNull();
expect(assignSheetNumbers(["10", "2", "5a", "1"], {})).toEqual({ "1": "01", "2": "02", "5a": "03", "10": "04" });
expect(assignSheetNumbers(["x-p05", "x-p22"], {})).toEqual({ "x-p05": "p05", "x-p22": "p22" });
// removing "2" and adding "11" keeps the rest and never reuses "02"
expect(assignSheetNumbers(["1", "5a", "10", "11"], { "1": "01", "2": "02", "5a": "03", "10": "04" }))
  .toEqual({ "1": "01", "2": "02", "5a": "03", "10": "04", "11": "05" });
```

(A real-calendar date check: `2026-02-30` is rejected too. The sheet zip has no dated name.)

- [ ] **Step 2:** `cd site && npx vitest run tests/download-names.test.ts`. Expected: fails (module missing).
- [ ] **Step 3:** Implement the module. Natural order: digit runs compare as numbers (same as `rasters._natural`).
- [ ] **Step 4:** Run the test again. Expected: PASS.
- [ ] **Step 5:** Commit: `site: download names, parsing and stable sheet numbers`.

### Task 3: Publish script: versioned uploads, history, no deletion

**Files:**
- Modify: `site/scripts/publish-downloads.mjs`, `site/lib/downloadRows.ts` (types only)
- Create (written by the script, committed): `site/lib/download-versions.json` (`{ "versions": [] }` initially), `site/lib/sheet-numbers.json` (`{}` initially)
- Test: `site/tests/publish-downloads.test.ts` (rewrite the planDownloads/planSheets/expired cases)

**Interfaces:**
- Consumes: Task 2's functions; `index.json` from Task 1 (`files[k].made`, `sheets[s][i].sha256`).
- Produces (types in `downloadRows.ts`):
  - `ManifestFile = { series; layer; fmt; name; bytes; sha256; fingerprint; updated; extra?: Record<"README.txt" | "LICENSE.txt", string> }`. `name` and `extra` values are versioned names; `url` is gone.
  - `ManifestSheet = { sheet; number; name; bytes; md5; sha256; updated }`
  - `Manifest = { files; sheets: Record<string, { updated; readme; items: ManifestSheet[] }>; license?; generated?; base?; local? }` (no `superseded`)
  - `Version = { series; layer; fmt; name; date; bytes; sha256 }`; history file `{ versions: Version[] }`, oldest first. Sidecars and `.sha256` files are not history entries.
  - `plan(index, manifest, history, numbers, listing: Map<string, number>, today) -> { files, sheets, numbers, newVersions: Version[], uploads: Upload[], copies: { from: string; to: string }[] }`, where `Upload = { to: string; local?: string; root?: "downloads" | "work"; text?: string; bytes: number }` (`text` for `.sha256` bodies)

- [ ] **Step 1: Write the failing tests** (pure `plan()`, no network)
  - A new vector file gets `name = versionedName({ date: made, ... })`, an upload to `blobPath`, a `.sha256` upload whose text is `"<sha256>  <name>\n"`, and one `Version`. The same applies to citywide plus its README/LICENSE sidecars, which share the image's date and hash.
  - An unchanged file (same `sha256` as the manifest) gives no upload and no `Version`.
  - A sheet whose `md5` is a DVC object in `listing` gives a `copy` from `dvc/files/md5/<2>/<30>` to `blobPath` (no upload). A sheet that isn't gives an upload from `work/<series>/<sheet>/warped_paper.tif`. Both get `.sha256` uploads.
  - A sheet keeps `updated` while its md5 is unchanged; a changed one gets `today`, a new versioned name, and a `Version`.
  - Review Focus 2 and 3: a name already in `listing` at the right size is not uploaded again. A `Version` whose `name` is already in `history` is not added twice. With nothing changed, `uploads`, `copies` and `newVersions` are empty and `files`/`sheets`/`numbers` deep-equal their inputs.
  - Review Focus 4: dropping a sheet from the index removes it from `sheets` but keeps its entry in `numbers`.
  - A name listed at a different size throws, naming the file.
  - `--local`: `files[k].name` stays versioned, and the site resolves it to `<base>/downloads/<index path>` (keep the existing local test's intent).
- [ ] **Step 2:** `npx vitest run tests/publish-downloads.test.ts`. Expected: new cases fail.
- [ ] **Step 3:** Implement `plan()`. Then rewrite `main()`:
  - list `downloads/` and `dvc/files/md5/` once;
  - print counts: uploads, copies, total bytes, and requests (listing + uploads + copies + one confirming listing);
  - `--dry-run` stops there;
  - otherwise upload sequentially (`put`, `allowOverwrite: false`, `addRandomSuffix: false`, `cacheControlMaxAge: ONE_YEAR`), then `copy(from, to, { access: "public", addRandomSuffix: false })`;
  - confirm every name by one listing;
  - only then write `downloads.json`, append `newVersions` to `download-versions.json`, and write `sheet-numbers.json`.

  Delete `expired`, `deletable`, `pruneSuperseded`, `resolveListed`'s move logic, and every `del` import and call. Keep `manifestProblem` (checking `base`/`local`) and `sameManifest`.
- [ ] **Step 4:** `npm test && npx tsc --noEmit && npm run lint`. Expected: all pass. Type errors in Tasks 4–6 files are expected only if those still read `url`. Fix them minimally here (derive with a helper `fileUrl(series, name)`, defined in `downloadRows.ts` and returning `/data/files/<series>/<name>`), so the build stays green.
- [ ] **Step 5:** Commit: `site: publish downloads under versioned names, keep a version history, never delete`.

### Task 4: The files route

**Files:**
- Create: `site/app/data/files/[series]/[name]/route.ts`, `site/lib/filesRoute.ts` (handler logic, testable)
- Delete: `site/app/data/latest/`, `site/app/data/sheets/[series]/sheet/`, `site/app/data/sheets/[series]/zip/`, `site/lib/latestRedirects.mjs` (and its exports in `downloadRows.ts`)
- Modify: `site/lib/sheetZip.ts` (zip filename and README section; members named by `ManifestSheet.name`; fetch from `${base}/${blobPath(series, name)}`), `site/app/data/sheets/[series]/route.ts` (CSV), `site/app/data/downloads.json/route.ts`
- Test: `site/tests/downloads.test.ts`

**Interfaces:**
- Consumes: `parseName`, `blobPath`, `latestName` (Task 2); the `Manifest` types (Task 3).
- Produces: `filesResponse(manifest: Manifest, series: string, name: string, fetchFn?, signal?) -> Promise<Response>`; `fileUrl(series, name)` (Task 3). CSV columns: `sheet,number,name,url,latest_url,bytes,md5,sha256,updated,license,credit`, where the URLs are absolute `https://sfzoning.com/data/files/...`.

- [ ] **Step 1: Write the failing tests**
  - A versioned name of a known series returns 308, with `location` = `${manifest.base}/downloads/<series>/<name>`; a `.sha256` name behaves the same.
  - `sfzoning-1948-lots-latest.gpkg.zip` returns 307, with `location` = `/data/files/1948/<current versioned name>` and `cache-control: no-store`.
  - `sfzoning-1921-sheets-latest.zip` streams the zip: an attachment under that name, the predicted length, and a README containing the GETTING THE NEWEST VERSION section.
  - Review Focus 1: an unknown series, a malformed name, `..`, a name containing `/` or `%2F`, and the `latest` name of a sheet no longer in the manifest each return 404 without fetching.
  - `downloads.json` lists `url` (versioned) and `latest` (latest) for each current file and sheet, as absolute sfzoning.com URLs.
- [ ] **Step 2:** `npx vitest run tests/downloads.test.ts`. Expected: new cases fail.
- [ ] **Step 3:** Implement. The route uses `export const dynamic = "force-dynamic"` and `maxDuration = 800` (the zip).
- [ ] **Step 4:** `npm test && npx tsc --noEmit && npm run lint`. Expected: pass.
- [ ] **Step 5:** Commit: `site: /data/files route for versioned and latest links`.

### Task 5: Versions pages

**Files:**
- Create: `site/app/data/versions/[series]/page.tsx`, `site/app/data/versions/[series]/[layer]/page.tsx`, `site/lib/versions.ts`
- Test: `site/tests/versions.test.ts`

**Interfaces:**
- Consumes: `download-versions.json`, `downloads.json`, `fileUrl`.
- Produces: `versionRows(history, manifest, series, layer) -> { date; latest: boolean; files: { fmt; label; name; url; bytes; sha256; checksumUrl }[] }[]`, newest first by history order. A row groups the formats released together (same `date` and adjacent in history). `latest` is true for the row holding the manifest's current names. Also `seriesLayers(history, series) -> string[]`.

- [ ] **Step 1: Write the failing tests**
  - Two releases on different dates give two rows, newest first, with `latest` only on the newer.
  - Review Focus 5: two releases on the same day give two rows ordered by history position, with `latest` on the later one.
  - `checksumUrl` = `fileUrl(series, name + ".sha256")`; the full 64-hex `sha256` is present.
  - `seriesLayers` lists `lots`, `districts`, `citywide` and `sheet-<n>`, in that order (sheets natural).
- [ ] **Step 2:** Run it. Expected: fails.
- [ ] **Step 3:** Implement `versions.ts` and the two pages (static: `generateStaticParams` from the history, `dynamicParams = false`). The layer page shows one line per version: date, "(latest)", per format a link with size, the full SHA-256 and a checksum link. It ends with the page note. The series page links each layer's page. Reuse `components/endmatter.module.css` and `downloads.module.css`; no new visual design.
- [ ] **Step 4:** `npm test && npx tsc --noEmit && npm run lint && npm run build`. Expected: pass; the build lists `/data/versions/...` as static.
- [ ] **Step 5:** Commit: `site: static versions pages from the version history`.

### Task 6: /data page links

**Files:**
- Modify: `site/components/DownloadTable.tsx`, `site/components/SheetDownload.tsx`, `site/app/data/page.tsx`, `site/lib/downloadRows.ts` (`Cell` gains `latestUrl`, `versionedUrl`, `versionsUrl`)
- Test: `site/tests/downloads.test.ts` (`rows` cases)

- [ ] **Step 1:** Update the `rows` tests. Each cell's `latestUrl` is `/data/files/<series>/<latest name>`, `versionedUrl` is `/data/files/<series>/<name>`, and `versionsUrl` is `/data/versions/<series>/<layer>`.
- [ ] **Step 2:** Run it. Expected: fails.
- [ ] **Step 3:** Implement:
  - buttons link `latestUrl`;
  - each file shows its dated link (text: the versioned file name) and a "versions" link;
  - sheets list each sheet's latest link and its dated name;
  - the zip button links `/data/files/<series>/sfzoning-<series>-sheets-latest.zip`;
  - add the exact page note under "What is here";
  - drop the `download=` attributes, since the redirect gives the name.

  Check the page's DRAFT COPY text still matches what's on screen.
- [ ] **Step 4:** `npm test && npx tsc --noEmit && npm run lint`. Expected: pass.
- [ ] **Step 5:** Commit: `site: /data links latest, shows dated and versions links`.

### Task 7: Export, publish, PR

- [ ] **Step 1:** Set up data in this worktree: `pipeline/scripts/worktree-data.sh`. Then run `cd pipeline && pixi run uv run sfzmaps downloads`. Expected: every vector and citywide file re-exported ($0), and `index.json` carries `made` on every file.
- [ ] **Step 2:** `cd site && node --env-file=.env.local scripts/publish-downloads.mjs --dry-run`. Report the counts it prints, uploads/copies/bytes/requests, to the owner before going on. Expected: copies equal the DVC-linked sheets (80 today), and there's no delete line.
- [ ] **Step 3:** Real publish (owner approved publishing to a PR branch). Expected: "wrote lib/downloads.json". `download-versions.json` gains one entry per file, sheet and format, and `sheet-numbers.json` fills.
- [ ] **Step 4:** Spot-check with a handful of requests: `curl -sI` a versioned link, a latest link and a `.sha256` link against `npm run dev`. Expected: 308/307/308 with the right `location`. Then `curl -sL` the `.sha256` and run `sha256sum -c` on the downloaded file.
- [ ] **Step 5:** Commit the three JSON files. Push the `gis-download-names` branch, open a PR (body: what changed, counts from step 2, and that the owner checks /data and a versions page by eye on the preview), then stop. The owner merges.
