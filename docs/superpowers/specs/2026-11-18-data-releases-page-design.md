# The data page as a releases page

2026-11-18. Status: design, for the owner's approval.

## Why

The person this page serves is an academic (or planner, or journalist) who wants the data, often all
of it. They need every file visible with its details, a direct link to each, a way to fetch a whole
set in one go, and a fixed, citable version. Today `/data` is a roomy one-row-per-edition table:
three formats stacked in each cell, sheets hidden behind "One sheet at a time", checksums only on
the per-layer versions pages and in `downloads.json`. There are 469 files (6.6 GB); clicking through
them is not a way to get everything.

The owner's model is a GitHub releases page or a Linux distro's release listing: dated releases,
each with its files, individual downloads and bulk downloads. Not zero design: denser, not plainer.
The look was agreed on a mockup (see "Look").

## Goals

- `/data` shows the newest release as a dense table: one row per file with layer, format, dated
  file name (the link), size, date, SHA-256 (short, full on copy), a `latest` link and a `history`
  link.
- Each release is a **dated snapshot of the whole dataset** with its own page, citable by URL.
- Bulk downloads for a release: `urls.txt` (for `wget -i`), `SHA256SUMS` (for `sha256sum -c`),
  `manifest.json`, `CITATION.cff`, and ready-made zips (the whole release, and each format).
- A **filter** (format, layer, edition) that narrows the table and every bulk download, e.g. "only
  Shapefiles".
- **Cite this release** with an APA / Chicago / BibTeX switch and a copy button.
- **What changed** between a release and the one before it.
- Per-file history stays: each row links to the existing `/data/versions/<series>/<layer>` page.

## Non-goals

- Changing how files are named, stored or published on Blob (the
  [download names design](2026-11-18-gis-download-names-design.md) stands).
- Browser tests (the repo has none; the owner checks the page by eye).
- Search over file names; the filter plus the browser's find is enough.

## Anti-goals

- A download built or saved in page JavaScript. Every download is a plain link to a server route
  (owner rule); the filter only hides rows and rewrites link query strings.
- A release whose contents change after it is published. A release URL, its `urls.txt` and
  `SHA256SUMS` always list the same files and hashes.
- A page that only works with JavaScript: without it, every row shows and the bulk links are the
  unfiltered ones.

## Approaches (owner picked 1)

1. **`/data` is the current release; each release has its own page (picked).** `/data/releases`
   lists releases; `/data/releases/<id>` shows one; bulk files are routes under it. Gives each
   release a citable URL and one checksum file; `/data` stays fast.
2. **All releases on `/data`, one collapsible block each.** Nothing new to route, but the page grows
   by ~469 rows per release and no single release has a clean URL to cite. Rejected.
3. **A bare directory listing per release** (like releases.ubuntu.com). Densest and most scriptable,
   but loses the edition/layer/format columns and the license context; the owner wants "not zero
   design". Rejected.

## What a release is

A **release** is one publish run of `scripts/publish-downloads.mjs` that changed anything. It
contains every current file at that moment, unchanged files included (a snapshot, like a distro
release), not only the files that changed.

The version history (`lib/download-versions.json`) cannot tell runs apart: unchanged sheets keep
their own older date (the 357 sheets read `2026-11-04`; everything else `2026-11-18`), and a
dropped series would still look current. So releases get their own record:

- **`site/lib/download-releases.json`** (new, committed, append-only like the history):
  `{ "releases": [ { "id": "2026-11-18", "date": "2026-11-18", "names": ["<versioned name>", …],
  "extras": { "<sidecar name>": "<sha256>", … }, "zips": { … } } ] }`.
  `names` lists every current file in the release: vectors, citywide images and single sheets.
  Everything a row shows (series, layer, format, bytes, sha256) is looked up from the version
  history by name. `extras` holds the citywide images' README/LICENSE sidecars with their hashes,
  because the history has no sidecar entries and the manifest forgets them once it moves on;
  `urls.txt`, `SHA256SUMS` and `manifest.json` include them, so `sha256sum -c` checks every file
  `wget` fetched. `zips` records the release's ready-made zips (see Zip).
- The publish script appends a release when the set of current names differs from the last
  release; a run that changes nothing appends nothing.
- **Id:** the publish date. A second release on the same day gets `-2`, `-3` (`2026-11-18-2`).
  *(My call: dates are what people cite; same-day repeats are rare and need only stay unique.)*
- **Made at publish time,** dated the publish day, in the branch that publishes. Branches must be
  up to date with `main` to merge (already enforced), so two branches that both publish are
  reconciled when one rebases: keep both sides' releases, re-id if they collide. A test checks ids
  are unique and in date order. A release may be dated a little before its deploy goes live; that
  is accepted. *(My call.)*
- **Backfill:** one release, `2026-11-18`, holding the current manifest's names. There is no
  2026-11-04 release: those files were published before versioned names existed.
- The newest release always equals the current manifest; a test enforces it.

## Pages and routes

All new pages are static, generated at deploy from the committed JSON, like the versions pages.

| Path | What |
|---|---|
| `/data` | The newest release, rendered by the same component as a release page, with the page intro. Canonical stays `/data`. |
| `/data/releases` | Every release, newest first: id, file count, total size, link to "what changed". |
| `/data/releases/<id>` | One release: the full page (header, cite, filter panel, table, notes). A banner on older releases says it is not the latest and links to it. |
| `/data/releases/<id>/changes` | Files added, changed (same series/layer/format, new version) and removed versus the previous release, as a table with the same columns. The first release says it is the first. |
| `/data/releases/<id>/urls.txt` | One absolute dated URL per line (`https://sfzoning.com/data/files/<series>/<name>`), README/LICENSE sidecars of citywide images included. |
| `/data/releases/<id>/SHA256SUMS` | `<sha256>  <name>` per line, for `sha256sum -c` in the download folder (see below). |
| `/data/releases/<id>/manifest.json` | The release as JSON: id, date, license, credit, and each file with series, layer, format, name, bytes, sha256, url, latest. |
| `/data/releases/<id>/CITATION.cff` | Citation File Format 1.2.0 for the release. |
| `/data/releases/<id>/sfzoning-<id>[-<format>].zip` | 308 redirect to the ready-made zip on Blob (format: `geojson`, `gpkg`, `shp`, `geotiff`; see Zip); 404 for any other name. |
| `/data/releases/latest/…` | Each bulk file above under `latest` resolves to the newest release (307, not cached past a deploy), so scripts can always ask for the newest. |

**Filtering.** The bulk routes take `format`, `layer` and `edition`, each a comma-separated list;
a missing parameter means all. Values: format `geojson,gpkg,shp,cog`; layer
`lots,districts,scans,sheets` (`scans` = citywide images, `sheets` = single sheets); edition = the
era ids in `lib/eras.ts` (`1921` … `today`), an edition covering all its series (1980–90 includes
its height and special-use scans). Unknown values are a 400 with a one-line message. These routes
read the query string, so they are dynamic, but they only return text built from committed JSON.

**`SHA256SUMS` paths.** `wget -i urls.txt` saves flat files under their versioned names, which are
unique across series, so `SHA256SUMS` lists bare file names (`<sha256>  <name>`), and
`sha256sum -c SHA256SUMS` works in the download folder. *(My call: names are unique across series
because they carry the series.)*

**Existing routes.** `/data/downloads.json` and `/data/sheets/<series>` (CSV) stay as they are,
and serve only the current sheets, so only the newest release links them. On an older release the
sheet row's "sheet list" points to that release's `urls.txt?layer=sheets&edition=<era>` and
"all as zip" is hidden. *(My call: an old release must not silently show today's sheets.)*
`/data/versions/…` pages stay and are linked from each row's `history`.

## The page

Built as in the agreed mockup (`.superpowers/brainstorm/…/releases-v6.html`, generator kept in the
session scratchpad), in the site's own fonts (Newsreader, Libre Franklin), paper colours and the
ultramarine accent.

- **Header.** Kicker "Open data", title "Download the data", a one-paragraph lede (what the data is,
  that it is a reconstruction, link to how it was made). Right: a release card with the release id
  as a dropdown of all releases (navigates to that release's page), a "Latest" tag on the newest,
  `<n> files · <size> · <n> editions · CC BY-NC-SA 4.0`, and links to "Changes since <previous>",
  "All releases", `manifest.json`.
- **Cite this release.** Segmented switch APA · Chicago · BibTeX (Chicago selected), the citation in
  a box with a copy button, a `CITATION.cff` link. Texts, for release `<id>`:
  - APA: `Bacio, S. (<year>). A History of San Francisco’s Zoning (Version <id>) [Data set]. GrowSF. <url>`
  - Chicago: `Bacio, Steven. <year>. A History of San Francisco’s Zoning. Data release <id>. GrowSF. <url>.`
  - BibTeX: `@misc{bacio_sfzoning_<year>, author = {Bacio, Steven}, title = {…}, year, version = {<id>}, publisher = {GrowSF}, url, note = {Data set, CC BY-NC-SA 4.0}}`
  One function makes all three and `CITATION.cff`, so they never disagree. Without JavaScript the
  Chicago text shows.
- **Filter and download panel** (sticky at the top while scrolling the table). Left: pill toggles
  for Format, Layer, Edition (with "all" / "none" for editions), all on by default. Right: the live
  count (`30 files · 316.8 MB`, "Everything in this release" or "Your selection"), a primary
  zip button when the selection matches a ready-made zip ("Download everything as zip",
  "Download all Shapefiles as zip", with its size; otherwise no button and a line saying a zip
  exists for everything and for each whole format), `urls.txt` and `SHA256SUMS` links, and a
  one-line `curl -s '<urls.txt?…>' | wget -nc -i -` command with a copy button. Only narrowed
  groups go in the query string. The filter state is mirrored in the page URL
  (`/data?format=shp`) so a filtered view can be shared; the page reads it on load.
- **Jump bar** of editions (hides editions the filter emptied).
- **Table.** Fixed column widths (they must not jump when filtering). Grouped by edition (header row
  with the edition's filtered file count and size). Columns: Layer (named once per run of the same
  layer among visible rows), Format (coloured tag: GeoJSON green, GeoPackage blue, Shapefile
  orange, GeoTIFF purple), File (dated name, the link), Size, Updated, SHA-256 (first 8, click
  copies the full hash), and `↓ latest` · `history`. Each scan series' sheets are one row
  ("16 sheets", "all as zip" = the existing sheet zip, "sheet list (CSV)"; on older releases see
  "Existing routes") that expands to one row
  per sheet in the same columns; collapsed by default. "Nothing matches" when the filter empties it.
- **Notes** in three short columns: Formats, Links (dated vs latest vs history; single sheets have
  no README), License and credit (keeps the exact credit line and the David Rumsey credit).
- **Narrow screens:** one column header/panel; date and hash columns hide.
- New copy carries the repo's `// DRAFT COPY — NOT REVIEWED BY THE OWNER` marker.

The table, header and notes are server-rendered; the filter, citation switch, sheet expanders and
copy buttons are one small client component working on `data-` attributes, so the 469 rows are not
re-rendered by React on each toggle.

## Zip (owner picked A)

Ready-made zips, built by the publish script and stored on Blob, so downloads come straight from
storage, resume, and never pass through a function.

- Per release: the whole release, `sfzoning-<id>.zip` (6.6 GB today), and one per format:
  `sfzoning-<id>-geojson.zip` (135 MB), `-gpkg.zip` (744 MB), `-shp.zip` (317 MB), `-geotiff.zip`
  (citywide images and sheets, 5.4 GB). Each holds `README.txt` (the release, what is inside, how to
  verify), `LICENSE.txt`, `CITATION.cff`, `SHA256SUMS` for its own contents, and the files flat under
  their versioned names (names are unique, so the zip's `SHA256SUMS` works from its top level).
  Stored (no compression: the members are already zips and TIFFs).
- Stored on Blob at `downloads/releases/<id>/<zip name>` with a `<zip name>.sha256` beside it, never
  overwritten or deleted. `download-releases.json` records each zip's name, bytes and sha256.
- Built once, when the release is made. Entries are read from the local outputs the publish
  script already has; a member whose bytes do not match its recorded sha256 stops the publish.
  Upload uses Blob multipart, one upload per zip (no bursts of requests).
- The page's zip button links the route, which 308-redirects to the Blob copy with `?download=1`.
- About 12 GB of extra storage per release today, kept forever (owner accepted).

Rejected: a zip built while it downloads (B), because its run time depends on the reader's
connection (at 800 s a 20 Mbit/s reader is cut off near 2 GB) and every click would push the bytes
through a function from an open link; and no zip at all (C).

## Testing

Vitest, our behaviour only:
- releases: snapshot lookup, newest release equals the manifest, same-day ids, append-only.
- publish script: appends a release only when the name set changes.
- filter parsing (valid, missing, unknown values) and edition → series mapping.
- `urls.txt` / `SHA256SUMS` / `manifest.json` / `CITATION.cff` contents for a fixture release,
  filtered and not; `sha256sum -c` format.
- changes: added / changed / removed.
- citation texts (one fixture per style).
- zip planning: which zips a release gets, their members and `SHA256SUMS`; the button's
  selection → zip matching; the zip route's redirect and 404s.
- releases: sidecar hashes carried in `extras`; ids unique and in order.

Then `npm run lint`, `npx tsc --noEmit`, `npm test`. The owner checks the page on the running site.

## Left to the builder

- Component and file layout under `site/app/data/` and `site/components/`.
- Cache headers within the rules above (static pages; `latest` aliases not cached past a deploy).
- Exact `CITATION.cff` fields beyond author, title, version, date-released, url, license.
