# GIS download names and versions

2026-11-18. Status: design, for the owner's approval.

## Why

People who download the GIS files (lots, districts, citywide images, sheets) should be able to tell
from the link and from the saved file what it is: which map, which layer, which version. Anyone who
needs a fixed version can always fetch exactly that version; anyone who wants the current data uses a
`latest` link, and the guidance on the site and in every README says to.

Today a link looks like `…blob.vercel-storage.com/downloads/680046dfd523/1921-lots.gpkg.zip`: the
folder is a hash of the build inputs (shared by all three vector formats), the name carries no date or
version, the link is on the storage domain, and superseded files are deleted after 30 days.

## Goals

- Every published file has a permanent, human-readable, versioned link on sfzoning.com that always
  returns the same bytes.
- Every file also has a `latest` link, same name with `latest` in place of date and hash, that
  always leads to the current version.
- Saved files keep their versioned name, so a copy on disk says which version it is.
- Nothing published is ever deleted.
- Every date is ISO `YYYY-MM-DD` (names, manifest, READMEs, CSVs).

## Non-goals

- Keeping the links published on 2026-11-04 working. The project has not been distributed; those
  links (raw Blob URLs, `/data/latest/...`, `/data/sheets/<series>/sheet/<id>`) can go. The 11-04
  files stay on Blob, unlinked (not deleted), and get no versioned names: versions start with the
  first publish under this design. (The README change below changes the downloads fingerprint, so
  every vector and citywide file is rebuilt with a new date and hash anyway.)
- Alerting people who pin a version. Pinning is a deliberate choice; the guidance points to `latest`.

## Anti-goals

- A versioned link that ever returns different bytes, or stops working.
- Download bytes passing through a Vercel function (slow and costly for 130 MB images).
- Permanent links on the storage domain (tied to one Blob store).

## Names

```
sfzoning-<series>-<layer>-<date>-<hash8>.<ext>      versioned
sfzoning-<series>-<layer>-latest.<ext>              latest
```

- `<series>`: the edition as the pipeline names it (`1921`, `1980-1990-zoning`, `today`).
- `<layer>`: `lots`, `districts`, `citywide`, or `sheet-<n>` for one sheet, `sheets` for a series'
  sheet zip.
- `<date>`: ISO `YYYY-MM-DD`, the day this version was made (see Dates).
- `<hash8>`: the first 8 hex digits of the file's own SHA-256, so `sha256sum` confirms it.
- `<ext>`: `geojson.zip`, `gpkg.zip`, `shp.zip`, `tif`; the citywide image's sidecars are
  `.README.txt` and `.LICENSE.txt` after the image's own stem (same date and hash as the image).

Examples (dates and hashes are illustrative):

```
/data/files/1948/sfzoning-1948-lots-2026-11-04-3f9a2c1b.gpkg.zip
/data/files/1948/sfzoning-1948-lots-latest.gpkg.zip
/data/files/1921/sfzoning-1921-citywide-2026-11-04-ce47cd18.tif
/data/files/1921/sfzoning-1921-citywide-2026-11-04-ce47cd18.README.txt
/data/files/1921/sfzoning-1921-sheet-01-2026-11-04-c103ba02.tif
/data/files/1990-1996-zoning/sfzoning-1990-1996-zoning-sheet-p22-2026-11-04-1a2b3c4d.tif
/data/files/1921/sfzoning-1921-sheets-latest.zip
```

**Sheet numbers.** `p<NN>` where the sheet ID ends in a page number (`…-p22` → `p22`); otherwise the
sheet's two-digit position in the series in natural order of sheet ID (`01`, `02`, …). This covers
1921 (scanner-timestamp IDs) and 1948/1960 (`1`, `2`, `5a`, …). The original sheet ID stays in the
sheet CSV and the README. (Owner approved this over scan dates in names.)

A number, once given, stays with its sheet ID for good and is never reused, so a `sheet-05-latest`
link always means the same physical sheet. The publish script keeps the ID → number map in a
committed file (`site/lib/sheet-numbers.json`): existing sheets keep their numbers, a new sheet gets
the next unused number in its series. *(My call: the alternative, recomputing positions every run,
would silently repoint links if a sheet were ever added or removed.)*

**Sheet zip.** Only a `latest` zip: `sfzoning-<series>-sheets-latest.zip`, built while it streams
(as today), holding README.txt, LICENSE.txt and each current sheet under its versioned name. No
versioned zip: it is a convenience bundle, and each sheet in it is permanently available on its own.
*(My call: a versioned zip would need its own hash scheme and a record of past member lists.)*

## Dates

- Vector files and citywide images: the day `sfzmaps downloads` made the file, which is also the
  README's LAST UPDATED date. `index.json` gains a per-file `made` date to carry it (today it records
  none; the publish script stamps its own day).
- Sheets: the day the publish first saw that sheet's content (as now: a sheet's date moves only when
  its content does).
- Every date everywhere is `YYYY-MM-DD`.

## Approach (owner picked A)

- **A. Site links that redirect to Blob (picked).** Each file is stored on Blob under its versioned
  readable name. `/data/files/<series>/<name>` looks the name up: a versioned name redirects to its
  Blob copy; a `latest` name redirects to the current versioned name on our site. Bytes come straight
  from Blob, browsers save under the readable name, and the permanent links live on sfzoning.com.
- **B. The site streams each file.** Full control of headers, but every download goes through a Vercel
  function: slow, costly, can time out on large files. Rejected.
- **C. Versioned links straight to Blob, only `latest` on the site.** Least code, but the permanent
  links would be tied to this storage account. Rejected.

## How it works

**Storage.** `downloads/<series>/<versioned name>` on Blob. Never overwritten, never deleted (the
publish script's 30-day deletion goes). Large sheets now link to their DVC objects; they get a copy
under the readable name instead (Blob's server-side `copy()`, no re-upload), because a redirect to a
DVC object would save the file under a hash name.

**Manifest.** `site/lib/downloads.json` stays the current set: what the /data page shows and what
`latest` links resolve to. There is no history file to look versioned names up in: a versioned name
maps to its Blob path by rule (`<series>/<name>` → `downloads/<series>/<name>`), so the route needs no
lookup. That makes the store itself the record of every version, and a publish from a PR branch
works the same as one from `main` (its versioned links work everywhere at once; its `latest` links
follow that branch's manifest until it merges). *(My call, from the builder check: a committed
history file would conflict between branches and lag behind the store.)*

**Routes.**
- `/data/files/[series]/[name]`: a well-formed versioned name (pattern above, known series) → 308
  (permanent) redirect to its Blob URL, by rule; if no such file was published, Blob answers 404.
  A `latest` name → 307 (temporary) redirect to the current versioned name on sfzoning.com, from the
  manifest; anything else → 404. The `latest` redirect must not be cached past a publish. *Builder's call:* cache headers
  within that rule.
- The `latest` sheet zip streams from a route (it is built, not stored).
- Removed: `/data/latest/[series]/[file]`, `/data/sheets/[series]/sheet/[sheet]` (and
  `lib/latestRedirects.mjs`). `/data/sheets/[series]` (the CSV) stays, with `latest` and versioned
  URLs per sheet.
- `/data/downloads.json` lists both links for every current file.

**/data page.** Download buttons use the `latest` links (the saved file still gets the versioned name
through the redirect). Each file also shows its versioned link for citing. A short note: "Use the
`latest` links to always get the newest data; a dated link always returns exactly that version."

**READMEs** (pipeline `downloads/readme.py`, outside the build's code deps so `dvc.lock` doesn't
change): a line giving the file's `latest` link and saying a dated link never changes. The README
can't name its own hash.

**Publishing.** `site/scripts/publish-downloads.mjs` computes names, uploads (or copies) what the
store's listing lacks, updates the current manifest and `sheet-numbers.json`. Still one
listing pass, no per-file requests, never overwrites.

## Constraints

- No overwrite of any published name (AGENTS.md); no bursts of requests (list/batch only).
- Production must never point at local files (`scripts/check-tiles-env.mjs`); `--local` manifests stay
  local.
- License and credit notices unchanged (CC BY-NC-SA 4.0; Rumsey credit on 1948).
- Site checks: `npm run lint`, `npx tsc --noEmit`, `npm test`; pipeline: `uv run pytest`. No browser
  tests: the owner checks the /data page by eye.

## Left to the builder

Marked *Builder's call* above, plus: test layout, how `index.json` carries `made`, and the exact
wording of the page note and README line (short, plain).
