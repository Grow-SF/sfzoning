# Data Releases Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `/data` into a dense releases page: dated whole-dataset snapshots, each with its own
page, filterable file table, bulk `urls.txt` / `SHA256SUMS` / `manifest.json` / `CITATION.cff`,
ready-made zips on Blob, and a citation switch.

**Architecture:** A new committed record, `site/lib/download-releases.json`, written by
`scripts/publish-downloads.mjs`, lists each release's file names (looked up in the existing version
history for details) and its zips. Pure release logic shared by the script and the site lives in
`.mjs` files under `site/lib/` (the script can't import `.ts`). Static Next.js pages render releases;
small dynamic routes return filtered text built from the committed JSON; zips are built once by the
publish script and served from Blob by redirect.

**Tech Stack:** Next.js App Router (TypeScript), Node `.mjs` publish script, `@vercel/blob`,
`client-zip` (already a dependency; ZIP64 for the 6.6 GB zip), Vitest.

**Spec:** `docs/superpowers/specs/2026-11-18-data-releases-page-design.md` (read it with this plan).
Agreed look: `.superpowers/brainstorm/13322-1795048131/content/releases-v6.html` (git-ignored, this
checkout only); match its layout, colours and behaviour.

## Global Constraints

- Every download is a plain link to a server route or file; nothing is built or saved in page JS.
- A published release never changes: its page, `urls.txt`, `SHA256SUMS`, `manifest.json`,
  `CITATION.cff` and zips list the same files and hashes forever. Nothing on Blob is overwritten or
  deleted (`allowOverwrite: false`).
- Without JavaScript every row shows, the Chicago citation shows, and bulk links are unfiltered.
- Credit line verbatim: `Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)`; keep
  the David Rumsey credit (`RUMSEY_CREDIT`, CC BY-NC-SA 3.0) wherever 1948 imagery is offered.
- New user-facing copy carries `// DRAFT COPY — NOT REVIEWED BY THE OWNER` at the top of its file.
- Dates are ISO `YYYY-MM-DD`; release ids match `^\d{4}-\d{2}-\d{2}(-[2-9]|-[1-9]\d+)?$`.
- Blob requests: listings and multipart uploads only, parts sent one at a time (no bursts).
- Never print or copy `BLOB_READ_WRITE_TOKEN`; run the script with
  `--env-file=<main checkout>/site/.env.local`, never copy that file.
- Tests test our behaviour only (no assertions about Next, MapLibre or client-zip internals).
- Checks: `cd site && npm run lint && npx tsc --noEmit && npm test`.

## Review Focus

1. **`latest` alias with a query string** (`/data/releases/latest/urls.txt?format=shp`): the redirect
   must keep the query, or a script silently downloads everything. Test in Task 6.
2. **Empty or junk filter values** (`format=`, `format=SHP`, `edition=1999`, repeated params): 400 with
   a one-line message naming the parameter, never a silently empty or full list. Tests in Task 5.
3. **A second publish on the same day** gets id `2026-11-18-2`, and every route and page accepts that
   id. Tests in Tasks 1 and 6.
4. **An older release page** links no current-only route (`/data/sheets/<series>`, the
   `sheets-latest.zip`). Test in Task 7.
5. **A filter that empties the table** shows "Nothing matches" and the zip/format buttons still make
   sense (whole-release button always shows; no format button). Covered by `formatZip` tests in Task 5
   and checked by eye in Task 9.

---

### Task 1: Release record logic (`lib/releases.mjs`)

**Files:**
- Create: `site/lib/releases.mjs`
- Test: `site/tests/releases.test.ts`

**Interfaces:**
- Produces (JSDoc-typed, exported):
  - `type Extra = { series: string; name: string; sha256: string }`
  - `type ZipKind = "all" | "geojson" | "gpkg" | "shp"`
  - `type ZipEntry = { name: string; bytes: number; sha256: string }`
  - `type Release = { id: string; date: string; names: string[]; extras: Extra[]; zips: Record<ZipKind, ZipEntry> }`
  - `type ReleaseRecord = { releases: Release[] }`
  - `RELEASE_ID: RegExp` (the Global Constraints pattern)
  - `currentNames(manifest) -> { names: string[]; extras: Extra[] }` — every `files[*].name` and
    `sheets[*].items[*].name`, sorted; extras from each file's `extra` + `extraSha256`, sorted by name.
  - `releaseChanged(record, names, extras) -> boolean` — false only when the last release has the same
    names and extras.
  - `nextReleaseId(record, date) -> string` — `date`, or `date-2`, `date-3`… if taken.
  - `zipNames(id) -> Record<ZipKind, string>` — `sfzoning-<id>.zip`, `sfzoning-<id>-geojson.zip`,
    `sfzoning-<id>-gpkg.zip`, `sfzoning-<id>-shp.zip`.
  - `releaseZipPath(id, name) -> string` — `downloads/releases/<id>/<name>`.
  - `zipMembers(release, history) -> Record<ZipKind, string[]>` — `all`: every name plus every extra
    name; `geojson`/`gpkg`/`shp`: names whose history entry has that `fmt`.
  - `sha256Sums(entries: {name, sha256}[]) -> string` — `<sha256>  <name>\n` per entry, in given order.

- [ ] **Step 1: Write the failing tests** in `tests/releases.test.ts`, using a two-file manifest fixture
  (one gpkg lots file, one citywide cog with README/LICENSE extras, one sheet):
  - `currentNames` returns the three names sorted and two extras with their `extraSha256` hashes.
  - `releaseChanged({releases: []}, …)` is true; same names+extras as last → false; one extra hash
    differs → true.
  - `nextReleaseId` gives `"2026-11-18"`, then `"2026-11-18-2"`, then `"2026-11-18-3"`; ids all match
    `RELEASE_ID`; `"2026-11-18-1"` and `"2026-11-18-02"` do not.
  - `zipNames("2026-11-18-2").shp === "sfzoning-2026-11-18-2-shp.zip"`.
  - `zipMembers`: `all` has 3 names + 2 extras; `gpkg` has only the lots file; `shp` is `[]`.
  - `sha256Sums([{name:"a.tif", sha256:"ab"}]) === "ab  a.tif\n"`.
- [ ] **Step 2: Run** `cd site && npx vitest run tests/releases.test.ts` — Expected: FAIL (module not found).
- [ ] **Step 3: Implement** `site/lib/releases.mjs` with the signatures above (plain JS + JSDoc, like
  `lib/downloadNames.mjs`).
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `git add site/lib/releases.mjs site/tests/releases.test.ts && git commit -m "site: release record logic"`

### Task 2: Citations (`lib/citation.mjs`)

**Files:**
- Create: `site/lib/citation.mjs`
- Test: `site/tests/citation.test.ts`

**Interfaces:**
- Produces: `citations({ id, date }) -> { url: string; apa: string; chicago: string; bibtex: string; cff: string }`,
  with `url = https://sfzoning.com/data/releases/<id>` and `year = date.slice(0, 4)`. Plain text (no
  HTML); the page italicises the title itself.

- [ ] **Step 1: Write the failing tests** for `citations({ id: "2026-11-18", date: "2026-11-18" })`:
  - `apa === "Bacio, S. (2026). A History of San Francisco’s Zoning (Version 2026-11-18) [Data set]. GrowSF. https://sfzoning.com/data/releases/2026-11-18"`
  - `chicago === "Bacio, Steven. 2026. A History of San Francisco’s Zoning. Data release 2026-11-18. GrowSF. https://sfzoning.com/data/releases/2026-11-18."`
  - `bibtex` equals exactly:
    ```
    @misc{bacio_sfzoning_2026,
      author    = {Bacio, Steven},
      title     = {A History of San Francisco’s Zoning},
      year      = {2026},
      version   = {2026-11-18},
      publisher = {GrowSF},
      url       = {https://sfzoning.com/data/releases/2026-11-18},
      note      = {Data set, CC BY-NC-SA 4.0}
    }
    ```
  - `cff` contains the lines `cff-version: 1.2.0`, `type: dataset`, `title: "A History of San Francisco’s Zoning"`,
    `  - family-names: Bacio` and `    given-names: Steven`, `version: "2026-11-18"`,
    `date-released: 2026-11-18`, `url: "https://sfzoning.com/data/releases/2026-11-18"`,
    `license: CC-BY-NC-SA-4.0`, and a `message:` line containing the verbatim credit line.
  - For id `2026-11-18-2` the url and version use `2026-11-18-2` and the year stays `2026`.
- [ ] **Step 2: Run** `npx vitest run tests/citation.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** `citations` in `site/lib/citation.mjs` (DRAFT COPY marker at top).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "site: release citations (APA, Chicago, BibTeX, CFF)"` (with both files added).

### Task 3: Publish script makes releases and their zips

**Files:**
- Modify: `site/scripts/publish-downloads.mjs` (`plan()`, `main()`, header comment)
- Modify: `site/README.md` (the publish section: the new file to commit, the zips, disk needed)
- Test: `site/tests/publish-downloads.test.ts`

**Interfaces:**
- Consumes: Task 1 (`currentNames`, `releaseChanged`, `nextReleaseId`, `zipNames`, `zipMembers`,
  `releaseZipPath`, `sha256Sums`), Task 2 (`citations`).
- Produces (exported from the script):
  - `plan(...)` result gains `locals: Record<string, { root: "downloads" | "work"; rel: string }>`
    for every current file, sidecar and sheet (kept or new), keyed by versioned name.
  - `releaseStep(record, manifestNext, newVersionCount, today, previousGenerated) -> { id, date, names, extras } | null`
    — null when `releaseChanged` is false. Date: `previousGenerated` when the record is empty and
    `newVersionCount === 0` (the backfill: today's set was published on that day), else `today`.
  - `zipReadme(release, kind, memberCount, bytes) -> string` — the zip's README.txt (DRAFT COPY):
    release id and date, what the zip holds, `sha256sum -c SHA256SUMS`, the credit line, the
    Rumsey credit (for `all`, which holds 1948 scans), the release page URL.
  - `writeZip(outFile, entries) -> Promise<{ bytes: number; sha256: string }>` where `entries` is
    `{ name, file?: string, text?: string, sha256?: string, lastModified: Date }[]`; streams
    `client-zip`'s `makeZip` to disk (no compression), hashing each file member as it streams and
    rejecting if a member's sha256 differs from the expected one.
- Release write order in `main()`: numbers → history → `lib/download-releases.json` → manifest.

- [ ] **Step 1: Write the failing tests**:
  - `plan` (existing fixtures): `p.locals[<new gpkg name>]` is `{ root: "downloads", rel: <index path> }`;
    a kept file and a sheet appear too (sheet: `{ root: "work", rel: "<series>/<sheet>/warped_paper.tif" }`);
    each sidecar appears.
  - `releaseStep`: empty record, 0 new versions → id/date = `previousGenerated`; empty record, 3 new
    versions → today; unchanged set → null; changed set on a day with a release → `<today>-2`.
  - `writeZip` (temp dir): two small files + a text entry → the zip's sha256 matches a re-hash of
    the written file and `makeZip`'s listing order is README.txt first; a member whose expected
    sha256 is wrong → rejects with its name in the message.
- [ ] **Step 2: Run** `npx vitest run tests/publish-downloads.test.ts` — Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `locals` in `plan`, `releaseStep`, `zipReadme`, `writeZip`.
- [ ] **Step 4: Wire `main()`**: after uploads are confirmed, compute `releaseStep`. If non-null:
  for each `ZipKind`, members = `zipMembers` (resolved via `p.locals` to absolute paths with
  `localPath`), plus text entries `README.txt`, `LICENSE.txt` (`index.license`), `CITATION.cff`
  (`citations(release).cff`), `SHA256SUMS` (`sha256Sums` of the members); write to
  `<DOWNLOADS>/releases/<id>/<zip name>`; reuse an existing local file only if a same-named zip is
  already on the store at the same size (rerun after a crash). Upload each zip with the existing
  multipart path to `releaseZipPath(id, name)` plus `<name>.sha256` (sha256sum format), skipping
  names the listing (now also listing `downloads/releases/`) already shows at the right size. Then
  append `{ ...release, zips }` to the record. `--dry-run` prints the release id, members per zip
  and total bytes, and uploads nothing; `--local` skips releases entirely.
- [ ] **Step 5: Run** `npx vitest run tests/publish-downloads.test.ts` — Expected: PASS.
- [ ] **Step 6: Update** the script header and `site/README.md` publish section: commit
  `lib/download-releases.json` with the others; the zips need ~8 GB free disk under
  `pipeline/data/downloads/releases/`.
- [ ] **Step 7: Commit** `git commit -m "publish: record releases and upload their zips"`

### Task 4: Publish the first release (real run)

**Files:**
- Create (by the script): `site/lib/download-releases.json`
- Test: `site/tests/releases.test.ts` (committed-data guards)

No new code beyond the guard tests; this is the backfill the spec calls for (release `2026-11-18`).

- [ ] **Step 1: Set up pipeline data** in this worktree:
  `pipeline/scripts/worktree-data.sh /Users/sbuss/workspace/sfzoning` — Expected: `pipeline/data/downloads/index.json`
  and `pipeline/data/work/` exist; `pgrep -fl "sfzmaps build"` shows nothing running.
- [ ] **Step 2: Dry run**: `cd site && node --env-file=/Users/sbuss/workspace/sfzoning/site/.env.local scripts/publish-downloads.mjs --dry-run`
  — Expected: `new versions: 0`, `to upload: 0 file(s)`, release `2026-11-18`, four zips
  (all ≈ 6.6 GB, geojson ≈ 135 MB, gpkg ≈ 744 MB, shp ≈ 317 MB). Anything else (new versions, a
  different id): stop and report to the owner.
- [ ] **Step 3: Real run** with the same command minus `--dry-run`. Expected: four zips and four
  `.sha256` uploaded, `lib/download-releases.json` written, `lib/downloads.json` and the history unchanged
  (`git diff --stat` shows only the new file).
- [ ] **Step 4: Write guard tests** in `tests/releases.test.ts` against the committed JSON: the last
  release's names/extras equal `currentNames(lib/downloads.json)`; ids unique, match `RELEASE_ID`,
  dates non-decreasing; every name is in `download-versions.json`; every release has all four zips.
- [ ] **Step 5: Run** `npm test` — Expected: PASS.
- [ ] **Step 6: Commit** `git add site/lib/download-releases.json site/tests/releases.test.ts && git commit -m "data: first release, 2026-11-18, with its zips"`

### Task 5: Release files, filter and text outputs (site side)

**Files:**
- Create: `site/lib/releaseFiles.ts`, `site/lib/releaseFilter.ts`, `site/lib/releaseData.ts`
- Test: `site/tests/releaseFiles.test.ts`

**Interfaces:**
- Consumes: Task 1 types and `sha256Sums`, `zipNames`; `Version`, `fileUrl`, `versionsUrl` from
  `lib/downloadRows.ts`; `latestName`, `EXTS` from `lib/downloadNames.mjs`; `ERAS` from `lib/eras.ts`.
- Produces:
  - `releaseFiles.ts`:
    - `type Fmt = "geojson" | "gpkg" | "shp" | "cog"`; `type Kind = "lots" | "districts" | "scans" | "sheets"`
    - `type ReleaseFile = { series; era; layer; kind: Kind; fmt: Fmt; name; bytes; sha256; date }`
    - `eraOf(series) -> string | undefined` — era id whose `districtSeries` is the series or whose
      `scanSeries` (with `-zoning` expanded to `-height`, `-su`, as `scanSeries()` in
      `downloadRows.ts` does) contains it.
    - `releaseFiles(release, history) -> ReleaseFile[]` — in ERAS order, then lots, districts
      (geojson, gpkg, shp), then per scan series citywide then sheets by number. Throws naming any
      name missing from history.
    - `extraFiles(release) -> (Extra & { era: string })[]`
    - `releaseChanges(prev: ReleaseFile[] | null, cur: ReleaseFile[]) -> { added: ReleaseFile[]; changed: { from: ReleaseFile; to: ReleaseFile }[]; removed: ReleaseFile[] }`
      keyed by `series/layer/fmt`.
    - `urlsTxt(files, extras) -> string` — absolute `https://sfzoning.com` + `fileUrl(series, name)`, one per line.
    - `releaseManifest(release, files, extras, license) -> object` — `{ id, date, license, credit, files: [{ series, layer, fmt, name, bytes, sha256, url, latest }], extras: [...], zips }`.
  - `releaseFilter.ts`:
    - `type Filter = { format: Set<Fmt>; layer: Set<Kind>; edition: Set<string> }`; `ALL: Filter`
    - `parseFilter(params: URLSearchParams) -> { filter: Filter } | { error: string }` — missing
      param = all; comma lists; empty value, unknown value or a repeated param → error naming it.
    - `filterQuery(f) -> string` — `""` when everything; else `?` + only narrowed groups in the
      order format, layer, edition, values in canonical order (`format=shp`).
    - `matches(file: { fmt; kind; era }, f) -> boolean`; extras match as `{ fmt: "cog", kind: "scans" }`.
    - `formatZip(f) -> "geojson" | "gpkg" | "shp" | null` — one vector format, layers include lots and
      districts, every edition.
  - `releaseData.ts` (reads the committed JSON): `allReleases(): Release[]` (oldest first),
    `findRelease(id): Release | undefined` (`"latest"` → newest), `previousRelease(id): Release | undefined`,
    `isLatest(id): boolean`.

- [ ] **Step 1: Write the failing tests** with a fixture history/release spanning 1921 lots (3
  formats), 1980–90 zoning + height citywide, and two 1921 sheets:
  - `eraOf("1980-1990-height") === "1980-1990"`, `eraOf("today") === "today"`, `eraOf("nope") === undefined`.
  - `releaseFiles` order and `kind`s; a missing name throws with that name.
  - `releaseChanges`: a new gpkg version → `changed` one entry; a dropped series → `removed`.
  - `urlsTxt` first line `https://sfzoning.com/data/files/1921/<name>`; extras included.
  - `parseFilter(new URLSearchParams("format=shp"))` → format `{shp}`, others all;
    `"format="`, `"format=SHP"`, `"edition=1999"`, `"format=shp&format=gpkg"` → `{ error }` naming the param.
  - `filterQuery(parseFilter("format=shp,gpkg").filter) === "?format=gpkg,shp"`; ALL → `""`.
  - `formatZip`: `format=shp` → `"shp"`; `format=shp&edition=1921` → null; `format=cog` → null;
    `format=shp&layer=lots` → null.
- [ ] **Step 2: Run** `npx vitest run tests/releaseFiles.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** the three files.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "site: release files, filter and text outputs"`

### Task 6: Bulk routes

**Files:**
- Create: `site/app/data/releases/[id]/urls.txt/route.ts`, `…/SHA256SUMS/route.ts`,
  `…/manifest.json/route.ts`, `…/CITATION.cff/route.ts`, `…/[zip]/route.ts`
- Create: `site/lib/releaseRoutes.ts` (the handlers, so tests call them without Next)
- Test: `site/tests/releaseRoutes.test.ts`

**Interfaces:**
- Consumes: Task 5 everything; Task 1 `zipNames`, `releaseZipPath`; Task 2 `citations`; manifest `base` and `license`.
- Produces: `releaseTextResponse(kind: "urls" | "sums" | "manifest" | "cff", id: string, url: URL) -> Response`,
  `releaseZipResponse(id: string, file: string) -> Response`. Each `route.ts` is a one-line
  `GET` delegating to these, `dynamic = "force-dynamic"`.

Behaviour: unknown id → 404. `id === "latest"` → 307 to the same path under the newest id, query
string kept, `cache-control: no-store`. Bad filter → 400, body the parse error. `urls`/`sums` →
`text/plain; charset=utf-8`; `manifest` → JSON (unfiltered fields, filtered `files`); `cff` →
`text/plain` (ignores the filter). Zip: `file` must be one of `zipNames(id)` values → 308 to
`<base>/<releaseZipPath(id, file)>?download=1`, `cache-control: public, max-age=31536000, immutable`; else 404.

- [ ] **Step 1: Write the failing tests** against the committed record:
  - `urls` for `2026-11-18` with `?format=shp` → 30 lines, all ending `.shp.zip`.
  - `sums` lines match `/^[0-9a-f]{64}  \S+$/` and count = files + extras.
  - `latest` + `?format=shp` → 307, `location` ends `/data/releases/2026-11-18/urls.txt?format=shp`.
  - `?format=` → 400; id `2026-11-17` → 404; id `2026-11-18-2` (not in record) → 404, but
    `RELEASE_ID.test("2026-11-18-2")` routes reach the lookup (no 400 for the id shape).
  - zip `sfzoning-2026-11-18-shp.zip` → 308 to `…/downloads/releases/2026-11-18/sfzoning-2026-11-18-shp.zip?download=1`;
    `sfzoning-2026-11-18-geotiff.zip` → 404.
- [ ] **Step 2: Run** `npx vitest run tests/releaseRoutes.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** `lib/releaseRoutes.ts` and the five `route.ts` files.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "site: release bulk routes and zip redirects"`

### Task 7: The release page (`/data` and `/data/releases/<id>`)

**Files:**
- Create: `site/components/ReleasePage.tsx` (server), `site/components/ReleaseFilter.tsx` (`"use client"`),
  `site/components/releases.module.css`, `site/app/data/releases/[id]/page.tsx`
- Modify: `site/app/data/page.tsx` (render `ReleasePage` for the newest release)
- Delete: `site/components/DownloadTable.tsx`, `site/components/SheetDownload.tsx` (only `/data` used them)
- Test: `site/tests/releasePage.test.ts`

**Interfaces:**
- Consumes: Tasks 2, 5, 6 (URLs only).
- Produces: `releaseView(release, history, manifest) -> ReleaseView` in `site/lib/releaseView.ts`
  (pure; the page renders it): `{ id, date, latest: boolean, previousId?, totals: { files, bytes, editions },
  citations, editions: { era, label, rows: Row[] }[] }`, where a `Row` is a file row
  (`{ type: "file", file, latestUrl, historyUrl }`) or a sheet group
  (`{ type: "sheets", series, label, count, bytes, date, sheets: ReleaseFile[], zipUrl?, listUrl }`).
  On the newest release `zipUrl` is the existing sheet zip and `listUrl` the CSV; on older ones
  `zipUrl` is absent and `listUrl` is `/data/releases/<id>/urls.txt?layer=sheets&edition=<era>`.

Page per spec "The page" and the mockup: header with release card (id dropdown navigating to
`/data/releases/<id>`; older releases also show a banner linking the latest), cite box (Chicago
default), sticky filter panel, jump bar, table, notes. Every file and sheet row carries
`data-fmt`, `data-kind`, `data-era`, `data-bytes`, `data-n`. `ReleaseFilter` does what the mockup's
script does, using `filterQuery`, `matches`, `formatZip` from Task 5: hides rows/editions, updates
counts, sets bulk link hrefs, shows the format-zip button only when `formatZip` is non-null, always
shows "Download the whole release as zip (<size>)", writes the filter to the page URL with
`history.replaceState`, and reads it on load (invalid → ignored, all on). Copy buttons use
`navigator.clipboard`. `generateStaticParams` on `[id]` lists every release id; metadata title
`Release <id> · Download the data · A History of San Francisco’s Zoning`.

- [ ] **Step 1: Write the failing tests** for `releaseView` against the committed data: 469 files /
  15 editions totals for `2026-11-18`; 1921's first rows are lots geojson, gpkg, shp; a sheet group
  for 1921 with 16 sheets and the current `zipUrl`; for a fixture *older* release, no row's URL
  contains `/data/sheets/` or `sheets-latest`.
- [ ] **Step 2: Run** `npx vitest run tests/releasePage.test.ts` — Expected: FAIL.
- [ ] **Step 3: Implement** `lib/releaseView.ts`, then the components, CSS (port the mockup's CSS to
  the module, using the site's `--paper`, `--ink`, `--accent`, `--font-body`, `--font-ui` tokens) and pages;
  delete the two old components.
- [ ] **Step 4: Run** `npm run lint && npx tsc --noEmit && npm test` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "site: /data as the release page, with filter and citations"`

### Task 8: Releases index and changes page

**Files:**
- Create: `site/app/data/releases/page.tsx`, `site/app/data/releases/[id]/changes/page.tsx`
- Test: `site/tests/releasePage.test.ts` (extend)

**Interfaces:**
- Consumes: Task 5 `allReleases`, `previousRelease`, `releaseFiles`, `releaseChanges`; Task 7 CSS module.

Index: newest first, one row per release: id (link), date, files, total size, "Changes" link.
Changes page: three sections (Added / Changed / Removed) in the release table's columns; the first
release says "This is the first release." Both static (`generateStaticParams`).

- [ ] **Step 1: Write the failing test**: `releaseChanges(null, files)` for the first release returns
  everything as `added`, and the changes page's helper `changesView(id)` (in `lib/releaseView.ts`)
  returns `{ first: true }` for `2026-11-18`.
- [ ] **Step 2: Run** — Expected: FAIL. **Step 3: Implement.** **Step 4: Run** `npm test` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "site: releases index and changes pages"`

### Task 9: Check it on the running site and open the PR

- [ ] **Step 1:** `cd site && npm run lint && npx tsc --noEmit && npm test && npm run build` — all pass.
- [ ] **Step 2:** `npm run dev`; in Chrome open `/data`, `/data?format=shp`, `/data/releases`,
  `/data/releases/2026-11-18/changes`, `/data/releases/2026-11-18/urls.txt?format=shp`,
  `/data/releases/latest/SHA256SUMS`; click a zip link (expect the Blob download to start). Compare
  `/data` with the mockup; check the narrow layout at 390 px; check without JavaScript (all rows).
  Screenshot `/data` and `/data?format=shp` for the owner, saying what each shows.
- [ ] **Step 3:** Push the branch and open a PR (owner merges; don't merge). Body: what changed,
  the screenshots, the four zips uploaded and their sizes, the storage cost (~7.8 GB per release,
  ~$0.18/month), and that the copy is DRAFT.
