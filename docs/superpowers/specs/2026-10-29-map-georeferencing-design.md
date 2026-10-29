# Automatic georeferencing of historic SF zoning maps — design

Date: 2026-10-29
Status: draft for review

## Why

sfzoning.com will host a NYT-style scrollytelling feature on the history of San
Francisco zoning: as the reader scrolls, the story advances through time and the
map changes to show what each era's zoning looked like on today's streets.

The source material is a set of archived zoning maps, most of them scanned by the
owner on the SF Planning Department's large-format scanner. To show them on a web
map they have to be aligned (georeferenced) to real-world coordinates. The owner
did 1948 and 1960 by hand in QGIS. Picking matching control points (find an
intersection on the scan, click the same intersection on the DataSF centerlines)
took forever, and with ~8 points per sheet the result is "OK, not perfect."

This project (step one) replaces that with a fully automatic pipeline. Step two,
a separate later project, will classify each historic district (height, use,
etc.) and redraw each era's zoning as clean colored shapes. The finished story
will show **both** the actual scans and the redrawn zoning.

## Goals

- A Python command-line pipeline that takes raw scans and produces, with no
  manual control-point picking and no QGIS:
  1. one aligned (georeferenced) image per sheet,
  2. one stitched citywide layer per map year/series,
  3. web map tiles the site can load directly,
  4. a saved alignment per sheet (the transform from scan pixels to real-world
     coordinates) that step two can reuse to map any point on a scan to a real
     location,
  5. an accuracy score per sheet and a review page per year,
  6. a browser-based control-point editor for sheets that fall below 95%.
- Cover every era in the source folder (see Inputs), including the 1980–2000s
  zoning, height & bulk, and special-use series, each as its own layer.
- Rerunnable end to end: same inputs → same outputs; adding a new scan means
  running the pipeline again, not doing manual work.

## Accuracy target (owner's priorities)

- Zoning boundaries: must be correct (10/10). Delivered in step two by snapping
  boundaries to modern block/lot lines — no warp of a hand-drafted scan can be
  more accurate than the drawing itself. Step one's job is to make that snapping
  possible and reliable.
- Block outlines: 9/10. Where a block hasn't really changed, the scan's block
  edges should land within a few feet of today's DataSF block outlines. Lines a
  few feet off are noticeable and bother the owner, so this is a real target,
  not a nice-to-have.
- **Stay true to the historical record (owner's decision).** Where an old block
  edge differs from today's for real reasons (street widenings, cut-back block
  faces, filled shoreline, redevelopment), the overlay keeps the historic
  geometry. The warp corrects paper and drafting distortion smoothly over
  areas; it must not pull each old edge onto its modern counterpart. Matches
  that are consistently off by more than drafting error, in a locally coherent
  way, are classified as **real change**: excluded from the warp fit and from
  the accuracy score, and recorded (with their offsets) for step two. Telling
  real change apart from drafting error is the hard part; the builder tunes the
  thresholds and validates against known cases (e.g. the Van Ness / Geary /
  Embarcadero changes, Western Addition).
- The pipeline measures this itself: for each sheet, distance from the aligned
  scan's block corners/edges to the matching modern ones (excluding real
  change), reported as summary stats (e.g. "95% of block corners within 5 ft,
  worst 18 ft near the Embarcadero"), plus how much was classified as real
  change. **Review threshold (owner's decision): any sheet below 95% accuracy
  goes to human review** in the control-point editor (see below), never
  silently published. "95% accuracy" is taken to mean 95% of a sheet's matched
  block corners (excluding real change) land within the distance tolerance.
  Designer's interpretation: the owner should confirm it. The tolerance
  itself (a few feet) is the builder's call, tuned against the answer key.

## Non-goals

- Classifying districts or redrawing zoning as vectors (step two). Step one only
  saves what step two needs.
- The scrollytelling site itself.
- Any interactive GIS / QGIS workflow. (The control-point editor below replaces
  it for the cases that need a human.)

## Anti-goals

- A bad overlay published without warning. Failure must be loud (flagged on the
  review page), not quiet.
- A process that needs the owner to place control points by hand.
- A pure even-stretch (affine/low-order) warp per sheet — that's what the manual
  attempt did and it can't correct paper shrinkage or drafting distortion.

## Inputs

Drive folder `old-planning-maps`
(https://drive.google.com/drive/folders/10p74RwqmzeQFBrBqjxXdmEOUREx0BLgO).
It's shared with steven@growsf.org and also readable via its public link
(`https://drive.google.com/embeddedfolderview?id=<folder id>` lists a folder;
`https://drive.usercontent.google.com/download?id=<file id>&export=download&confirm=t`
downloads a file).

| Folder | Contents |
|---|---|
| 1921 | 23 single-page PDFs (cover, ordinance text, ~14+ map sheets). 1-bit 300 dpi. Map sheets print **assessor block numbers** in each block, sheet numbers, and "Join Sec. N" edge notes. |
| 1929, 1935, 1964, 1968, 1972, 1975, 1980 | One multi-page PDF each (e.g. 1968: 23 pages, 1-bit 300 dpi, ~6600×5300 px). Includes covers/text pages. 1960s+ sheets print **street names**, district labels (R-1, C-2…), "See Sheet N" edge notes. |
| 1948 | `originals/` (JPG photos of a bound atlas, ~5900×7500 px, color, from the **David Rumsey Map Collection** — not the owner's scans; must be credited, likely CC BY-NC-SA), plus `index.jpg`, `setback.jpg`, `planning-code.jpg`, some sheets split/combined (`2-and-3.jpg` etc.). `aligned/` holds the owner's manual QGIS control points (`N.jpg.points`) and warped TIFFs. Prints block numbers. |
| 1960 | `originals/1960.pdf`, `aligned/` with per-sheet PDFs and manual `.points` files. |
| 1980 Plus | Large PDFs per series: Zoning, Height & Bulk, Special Use (SU-SS / Coastal / Preservation) for 1980–90, 1990–96, 1996–2000, 2000–digital. Not yet inspected. |

Manual `.points` files are QGIS GCP CSVs: `mapX,mapY,pixelX,pixelY,enable`, with
`mapX/mapY` in California State Plane Zone III, US feet (EPSG:2227) — verify from
the 1948 `.qgs`. These are the **answer key** for evaluating the automatic
results (they're imperfect too; treat them as a sanity check, not ground truth).

Reference data from DataSF (builder verifies exact datasets/IDs):
- Street centerlines (active and retired).
- Parcels / assessor blocks (block number + geometry). Assessor block numbers
  have been stable since before 1921, which is what makes the block-number
  anchors work.

Raw scans and large outputs do not go in git.

## How it works (decided at this level; details are the builder's)

Per sheet:

1. **Sort pages.** Separate map sheets from covers, legends, and text pages;
   skip the latter (log them). Handle oddities like the 1948 combined/split
   sheets and the 1921 "Join Sec." pages.
2. **Rough placement — automatic anchors.** Read printed text with the
   **OpenAI vision API** (owner's decision, replacing the earlier choice of
   Claude because the owner has OpenAI credits; tesseract isn't good enough on this
   lettering). Tile the sheet as needed for resolution. Cache every API
   response on disk keyed by input image + prompt, so reruns are free and
   deterministic. Match the text to DataSF:
   - block numbers (1921, 1948, any sheet that has them) → modern block
     centroids;
   - street names (1960s+) → centerlines/intersections.
   Sheet numbers, "See Sheet N"/"Join Sec. N" notes, and already-placed
   neighboring sheets can narrow down the search. Fit a robust initial
   transform, discarding anchors that disagree (OCR misreads).
3. **Fine alignment — use everything.** Render today's block outlines at the
   sheet's estimated scale and position. Iteratively match the scan's street /
   block geometry (e.g. intersections or edges extracted from the line drawing)
   to the nearest modern features, rejecting outliers, until it converges. Use
   hundreds to thousands of matches per sheet, not ~8.
4. **Flexible but smooth warp.** Warp with a regularized non-linear transform
   (e.g. smoothed thin-plate spline or a coarse piecewise model) that corrects
   paper and drafting distortion over areas, not per edge. It must not snap
   individual old edges onto modern ones (see "Stay true to the historical
   record"). Guard against overfitting to bad matches.
5. **Change classification.** Two categories, recorded per sheet for step two:
   - *No modern equivalent*: areas where the old layout doesn't exist today at
     all (Western Addition redevelopment, Golden Gateway, Diamond Heights,
     freeways, Mission Bay, filled shoreline). Excluded from matching.
   - *Real change*: features that exist in both but moved (widened streets,
     cut-back block faces), with their measured offsets. Excluded from the warp
     fit and the score.
6. **Score.** Measure residuals as in "Accuracy target" and against the manual
   1948/1960 points where available. Flag sheets that fall below threshold.

Per year/series:

7. **Stitch** sheets into one citywide layer, resolving overlaps between
   neighboring sheets (crop margins, legends, title blocks).
8. **Tile** for the web, in a single-file format the site can load from static
   hosting (PMTiles is the expected choice), one file per year/series. White
   paper is made transparent so today's map can show through. An opaque "paper"
   look can still be restored by drawing a background underneath, so this
   keeps both options open (designer's call). 1948 is color and needs its own
   background-removal handling.
9. **Review page** (static HTML): each sheet over today's blocks, with its
   accuracy score, its real-change / no-modern-equivalent areas, and a flag if
   it failed. The owner reviews here, not in a GIS.

### Human review: control-point editor (in scope; owner's decision)

Sheets below 95% go into a **needs review** queue. The pipeline keeps working
on the other sheets; it doesn't stop and wait. The builder also builds the
editor:

- Runs locally in the browser (started from the pipeline CLI). No QGIS.
- Shows the scan next to (and optionally overlaid on) today's blocks and
  centerlines, with the automatic control points drawn on both.
- Points are colored by how far they are from where the warp puts them, and
  the worst areas of the sheet are highlighted, so the reviewer knows where to
  look.
- The reviewer can drag, add, delete, or disable points. They can also mark an
  area as real change / no modern equivalent.
- Shows a live (or near-live) preview of the warped result and the updated
  accuracy score as points change.
- Saving writes the edited points back as the sheet's alignment, re-runs warp,
  score, stitch and tile for just that sheet/year, and records that a human
  reviewed it (who/when). Human edits are never overwritten by a later
  automatic rerun unless explicitly requested.
- The reviewer can accept a sheet that's still under 95% (with a note), for
  cases where the drawing itself is just that inaccurate.

Until reviewed, a sheet is still in the stitched layer and tiles (so the year
has no hole), but the machine-readable manifest marks it `needs_review` so the
site can hide or caveat it (designer's call). For 1948 and 1960, the editor can
start from the owner's manual `.points` if they score better than the automatic
ones.

Step one is **done** when every era is processed, every sheet is either
passing (≥95%) or human-reviewed, and the review pages and editor exist.

Saved per sheet (for step two): the transform (as GCPs + method, or an
equivalent serializable form), the changed-streets mask, the accuracy report,
and the page classification.

## Constraints

- Runs locally on the owner's Mac (Apple Silicon). Python. GDAL, OpenCV,
  tesseract, ImageMagick, and poppler are already installed.
- Repo is currently a bare Vite + React template (`sfzoning`); the pipeline
  lives alongside it in its own directory.
- 1948 imagery must carry David Rumsey Collection credit wherever it's shown.
- OpenAI API key required (`OPENAI_API_KEY`); all inference uses OpenAI (owner's decision). Expected one-time cost:
  tens of dollars (unverified estimate); caching means reruns are free.
- **Nothing large goes in git (owner's decision).** Scans and generated files
  live in the git-ignored `pipeline/data/`. Generated files that save other
  people work or money are shared in a new `processed/` folder inside the Drive
  folder `old-planning-maps`, so anyone can fully recreate the results without
  redoing the expensive or manual steps. That means:
  - the vision-model response cache (the paid step);
  - the per-sheet alignments, change areas, scores and human `review.json`
    files (the manual step);
  - the exact DataSF reference snapshot the results were computed against
    (DataSF changes over time);
  - the per-year PMTiles, manifests and review pages.
  Rendered PNGs and warped GeoTIFFs are not shared, since they are cheap to
  regenerate from the scans plus the alignments. `sfzmaps pull` restores the
  shared files into `pipeline/data/` with no Google login, via the folder's
  public link. `sfzmaps push` uploads them, which needs write access (the
  owner).
- Hosting is undecided and deferred to the site project (designer's call). Build
  tiles locally, one file per year/series, and report sizes per zoom level so
  hosting can be chosen with real numbers.

## Left to the builder

- Choice of libraries and algorithms within the approach above (OCR engine or a
  vision model for text, feature extraction method, transform model, matching
  strategy).
- Output CRS for intermediate files and the exact tile format/zoom levels.
- Accuracy thresholds, tuned using the manual 1948/1960 points and residuals.
- Directory layout, caching, and how raw scans are downloaded and stored.
- Order of eras. Suggested: start with 1948 (block numbers plus answer key),
  then 1960 (street names plus answer key), then the rest.
