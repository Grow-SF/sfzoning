# sfzmaps: georeferencing San Francisco's historic zoning maps

`sfzmaps` turns the scanned zoning maps in the Drive folder `old-planning-maps`
(1921 to 2000-digital) into georeferenced, transparent-paper overlays: one PMTiles
file per year/series, a machine-readable manifest, and a review page. Sheets that
miss the accuracy bar go to a local control-point editor for human review.

Design: `docs/superpowers/specs/2026-10-29-map-georeferencing-design.md`.
Tuning history and final results: `docs/superpowers/notes/2026-10-29-tuning.md`.

## Setup

The system tools (GDAL, poppler, uv, git, Node) are pinned in the repository's `pixi.toml` and
locked in `pixi.lock`; see [Toolchain](#toolchain). Install [pixi](https://pixi.sh) once
(`brew install pixi` on macOS), then, from the repository root:

```sh
pixi install           # the pinned tools, into .pixi/ inside the repository (no admin rights)
pixi shell             # a shell with them first on PATH (or prefix commands with `pixi run`)
cd pipeline
uv sync
uv run sfzmaps doctor  # every tool should say "pixi"
```

Page classification and label reading use the OpenAI API. Put the key in
`pipeline/.env` (git-ignored; `chmod 600`):

```
OPENAI_API_KEY=...
```

The code loads it itself. Every answer is cached on disk under
`data/cache/llm/`, so a rerun over the same pages makes no API calls. District pattern
answers are also cached per tile, so re-placing a sheet re-reads only the tiles that changed.

## Commands, in order

All commands run from `pipeline/` as `uv run sfzmaps <command>`.

| Command | What it does |
|---|---|
| `build [-j N] [--dry-run] [STAGE...]` | Rebuild whatever is stale, in parallel, at $0: see [Builds](#builds). The usual way to run the pipeline. |
| `stage NAME` | Run one build stage (what `build` and `dvc repro` call). |
| `fetch [--only FOLDER]` | Mirror the Drive tree into `data/raw/`. |
| `refdata [--force]` | Download DataSF parcels and street centerlines; build blocks, corners and intersections in `data/ref/`. Also the park outlines: DataSF's Recreation and Parks Properties and the National Park Service's land tracts in the city. |
| `parks [--out CSV] [--maps]` | The park lots `data/ref/` shows (at least half the lot inside a Rec and Park property, or an NPS tract the NPS owns outright that today's zoning calls P), with the NPS conveyance year: the candidates for `inputs/parks.csv`. |
| `run --year S` / `run --all` | Run every page of a series: classify, read labels, rough fit, extract linework, refine, classify real change, score, warp; then stitch, tile, write the manifest and review page. Fresh stages are skipped; human reviews are kept (`--force-auto` ignores them, `--force` reruns everything). `--sheet ID` and `--until STAGE` make a partial run that does not republish the series. |
| `tiles --year S` | Rebuild the stitched VRT, PMTiles, manifest and review page for one series from existing warps. |
| `paper --year S` / `paper --all` | Warp each placed sheet as it looks on paper (opaque, colours untouched, cubic resampling) to `warped_paper.tif` and publish `out/<series>/<series>-paper.pmtiles`: the layer the site shows as the original map. The transparent-paper `<series>.pmtiles` stays for the review page and the district readers. The PDFs are 1-bit black and white at 300 dpi; their white is multiplied onto `warp.PAPER_TINT`, the 1948 photos' cream paper, so every era sits on the same paper (ink stays black; 1948 is untouched). Each sheet keeps its map up to its frame; the printed border rule and the 1948 photos' binding shadows are erased (`warp.erase_border_rules`). The stitch stacks a plain-paper backing (`stitch.paper_backing`, for narrow gaps), every sheet's whole paper (legend and title boxes filled with paper), then every sheet's inked content in priority order, so blank paper never hides a neighbour's map. |
| `review --year S` | Serve `data/out/` on 127.0.0.1 and open the series review page. Makes the page's districts layer (`out/<S>/districts/<S>-districts.pmtiles`) first if the series' districts changed; builds don't make it, since the site reads the shared archive. |
| `edit [--year S] [--sheet ID]` | Start the control-point editor on 127.0.0.1:8765 (`--demo` for a synthetic sheet). |
| `downloads [--series S ...]` | Write the public GIS downloads into `data/downloads/` (git-ignored) and its `index.json`: per edition, lot and district zips (GeoJSON, GeoPackage, Shapefile), the citywide Cloud Optimized GeoTIFF, and the per-sheet list. An item whose sources are unchanged is not rewritten, so its date stays. $0, no API calls; not a DVC stage; needs the pixi GDAL. |
| `bundle` | Write the processed files into `data/share/` as zips plus `PROCESSED.json`, for copying by hand into the Drive folder `old-planning-maps/processed/`. |
| `pull` | This commit's cache from the Blob store: its answer packs and the data its `dvc.lock` names. See [Shipping the cache](#shipping-the-cache). |
| `ship [--dry-run]` | When a branch is ready to merge: upload what its commit needs that `main` hasn't shipped, and commit the records. |
| `release [--dry-run] [--no-pr]` | After a merge, from an up-to-date `main`: check, run `downloads`, publish tiles and downloads to the Blob store, and open a PR with the updated manifests. See [Releasing](#releasing). |
| `answers [--check\|--baseline]` | Which artifacts know the answers they were made from; record the rest. |
| `pull-drive` | The older way: the processed zips from the public Drive `processed/` folder, extracted into `data/`. |
| `crossref --year S` / `--all` | Re-place flagged sheets by matching each to the same printed sheet number in another edition that passed (SIFT + RANSAC on the drawings, composed with that sheet's transform), then rerun refine and score. A result is kept only if it scores better, and is then pinned as the sheet's seed (`inputs/<series>/seeds/`; commit it); otherwise the sheet's files and seed are restored. Republishes improved series (`--no-publish` to skip). No API calls. |
| `eval --year 1948\|1960` | Per-sheet score next to the distance from our transform to the owner's manual `.points` (the answer key). |
| `estimate --year S` / `--all` | Estimate API calls, tokens and USD for the uncached pages (no API calls). It is an estimate: image tokens use OpenAI's patch formula, prompt text is a flat `config.EST_TEXT_TOKENS_PER_CALL` (~350) per call, and output is a flat `EST_OUT_TOKENS` per call. |
| `calibrate --year S --sheet ID` | Find the smallest label-reading scale that keeps 95% of full-resolution recall. |
| `districts --check [--against DIR...]` | The $0 accuracy check of the district reads: see [Checking the district reads](#checking-the-district-reads). |

`push` also exists: it bundles and uploads with rclone, if you have set rclone up.
It's optional; `bundle` plus a manual copy does the same job.

## Inputs

Everything a person decides is in git, under `inputs/`, so every checkout has it. `data/`
holds only what the pipeline can rebuild. Per series, `inputs/<series>/`:

- `reviews/<sheet>.json`: a reviewed fit, written by the editor. Commit it.
- `seeds/<sheet>.json`: a pinned rough fit, used instead of the automatic one; refine still
  runs from it. These are the sheets placed by matching another edition (`crossref`) or by
  hand-guided seeding; without the seed, a `--force` run would refit them from scratch. Delete
  a seed to let the pipeline place that sheet on its own again.
- `pages/<sheet>.json`: fields that override the machine `page.json`, e.g. `{"kind": "text"}`
  for a page the classifier took for a map.
- `overrides.csv`: zoning corrections (`blklot,block,code`); the research behind them is in
  `docs/research/zoning-overrides/`.

and `inputs/label_scale.json`, the calibrated label-reading scale per series, and
`inputs/parks.csv` (`name,blklot,since,source`): park lots, each `public` in every series whose
year is `since` (the year it became a park) or later, whatever its map zones it. Its lots come
from `sfzmaps parks`; a park without a documented year stays out. `inputs/park_maps.csv`
(`name,blklot,series`) is generated by `sfzmaps parks --maps` from the published maps: the first
map showing each park lot of `data/ref/` as public. That lot is `public` on every later map; with
a `since` too, from the earlier of the two. The shared step fails when the file no longer matches
the maps: run `sfzmaps parks --maps` and build again. Each series' `stats.json` counts the parks
in effect (`parks: {count, acres}`).

## Builds

`uv run sfzmaps build` rebuilds exactly what is stale, up to `-j` stages at once (default 4),
and records what it built. [DVC](https://dvc.org) keeps the record: `dvc.lock` (committed)
holds the content hash of every input and output of every stage, so a checkout can tell
whether its data matches its code and inputs (`uv run sfzmaps build --dry-run`, or
`uv tool run --from dvc==3.67.1 dvc status`).

- **Stages** (`sfzmaps build --list`): one per edition (`edition-1948`: run, paper,
  districts), one per 1980–2000 period (`period-1980-1990`: its zoning, height and
  special-use series, heights read before zoning), `today`, and `shared`, the merged
  archive the site reads. `pipeline/dvc.yaml` is generated from `sfzmaps.build`
  (`sfzmaps build --write-dvc-yaml`); a test keeps them equal.
- **Stale** means a stage's code (`src/sfzmaps`, except the editor), its inputs
  (`inputs/<series>`), the scans, the reference data or its own outputs changed. Inside a
  stage, the pipeline's own checks decide what is recomputed: sheets whose placement is
  current are skipped, and a layer is re-stitched and re-tiled only when its sheets, their
  warped rasters (by content), the stitch or tiling code, or the tiling settings changed (the
  fingerprint of each build is kept beside the layer as `<name>.inputs.json`). Districts work
  the same way: a series is read again only when its sheet files, its hand-made inputs, the
  parcels, the research table or the code the read runs changed (`districts_raw.parquet.inputs.json`),
  and republished only when its read, heights, overrides, rules or the joining code changed
  (`stats.json.inputs.json`); `districts --force` reads and publishes anyway. So most code
  changes rebuild in minutes. The dry run shows which stages would run.
- **Downstream only when it changed.** A stage that is not stale itself but comes after one that
  is waits for that one to run and be recorded, then is asked again (`dvc status STAGE`): when
  what its upstream stages wrote is unchanged where it reads, it doesn't run ("up to date after
  its upstreams"). Stale stages, and stages named with `--force`, always run.
- **Never spends money.** Stages run with `SFZMAPS_MAX_USD=0`: cached answers are free, and a
  read that would call the API fails the stage instead. Run that read yourself
  (`SFZMAPS_MAX_USD=5 uv run sfzmaps run --year S`, or `districts --series S`), then build again.
- **Fails loudly.** A stage fails when a command fails, when any of its sheets is in error
  (`score.json` reason `error: ...`, also on a retry), or when `paper` skips a series. A failed stage isn't recorded and the stages after it don't run.
- **Records what it was built from.** `build` refuses to start with uncommitted changes in
  `src/`, `inputs/`, `dvc.yaml` or the lock files, and stops recording if they change while it
  runs. One build at a time per data folder (`data/.build.lock`). Logs: `data/build-logs/`.
  Commit `dvc.lock` after a build.
- **Publishing checks it.** `site/scripts/publish-tiles.mjs` refuses to publish unless DVC says
  the data matches `dvc.lock`.
- `dvc repro` also works, one stage at a time.

DVC caches each recorded version in `.dvc/cache` as copy-on-write clones (`cache.type =
reflink,copy`), which cost no disk on APFS; never switch it to `hardlink` or `symlink`, because
the pipeline rewrites files in place. Stitch intermediates and the editor's caches are in
`.dvcignore`. To reclaim cache space, only ever run
`dvc gc --workspace --all-commits --all-branches`: a plain `dvc gc` deletes versions other
branches and worktrees still use.

**Worktrees.** In a new git worktree, run `pipeline/scripts/worktree-data.sh <main checkout>`.
It links `data/raw`, `data/ref` and `data/cache` to the main checkout and checks out `data/work`
and `data/out` exactly as that worktree's `dvc.lock` records them, from the shared cache. Never
link `work/` or `out/` between checkouts.

**Sharing** is [Shipping the cache](#shipping-the-cache): each commit's data and answers on the
Blob store.

## Shipping the cache

Each branch or worktree builds and works locally; nothing leaves the machine until the branch is
ready to merge. Then `uv run sfzmaps ship` publishes what that commit needs, so `main` (and
anyone) gets it with `uv run sfzmaps pull` and builds at $0 with nothing to do.

- **What a commit needs**: its processed data (the DVC objects its `dvc.lock` names: `work/`,
  `out/` and `ref/`), and the vision answers that data was made from. Every artifact made from
  answers keeps a sidecar, `<artifact>.answers`, listing the answer keys it was made from,
  written before the artifact and versioned with it. Only live artifacts count (every page's
  `page.json`, map pages' `labels.json`, each series' district reads, which fold in the legend
  and district labels they used), so answers from abandoned or superseded experiments never
  ship.
- **Shipped objects back public downloads.** The single `dvc/files/md5/…` objects (the large
  sheets) are also what the site's public sheet downloads link to. Never delete one, and never
  `dvc gc` them away.
- **Data packs.** Data objects under 8 MB travel in *data packs*: uncompressed zips of at most
  256 MB, named by their sha256 (`cache/datapacks/<sha>.zip`). Larger objects are stored alone
  (`dvc/files/md5/…`). Today that's 13 packs (3.5 GB, about 10,250 objects) plus 204
  single objects (7 GB), so a fresh `pull` makes 219 requests instead of one per object (about
  10,450): the manifest, the answer pack, 13 data packs and 204 single objects; measured into an
  empty DVC cache, it took 11.5 minutes, and `dvc checkout` then took 3 seconds. The shipped
  manifest (version 2) says which pack holds each small object; an object a pack on `main`
  already holds keeps that pack, so a ship packs only new small objects.
- **`sfzmaps ship`** (needs `site/.env.local`, or `SFZMAPS_ENV_FILE=<path>` naming another, e.g.
  the main checkout's from a worktree; never calls the vision API) refuses a dirty tree or
  data that doesn't match `dvc.lock`. It uploads the data the Blob store lacks, one new
  *answer pack* of exactly the needed answers `main` hasn't shipped (a zip named by its sha256,
  with an index of every answer's sha256), and the full list of needed answers. Then it commits
  the pack's record (`pipeline/cache-packs/<sha>.json`) and `pipeline/cache-needed.json`: push
  them with the branch. Uploads come first and records last, and nothing is ever overwritten:
  every name is the hash of its bytes. Requests stay few: one listing call per 1,000 names,
  then one file at a time, one request per file up to 100 MB; a larger file goes up with the
  SDK's multipart upload (`put(..., { multipart: true })`, which `@vercel/blob` 2.8.2 splits
  into 8 MB parts, sends 8 at once in Node and retries part by part). `--dry-run` says what would upload and how many requests it takes; more than
  500 need `--yes`. `--repack` (run once, for PR #28) also moved the small objects that older
  ships stored alone into packs; it shipped 13 packs (3.46 GB) in about 95 requests.
- **`sfzmaps pull`** applies the packs recorded at this commit (sha-checked, put-if-absent) and
  fetches the data its `dvc.lock` names that the local DVC cache lacks, then checks it out:
  each data pack holding a missing object once (4 at a time; every member md5-checked, written
  atomically, present ones skipped, so an interrupted pull resumes), then single objects (also
  4 at a time, md5-checked). Commits shipped before data packs (version 1 manifests) still pull, one
  request per object. Public URLs, no token. (`pull-drive` is the older Drive download.)
- **The merge gate.** The `Cache shipped` workflow checks every PR touching the pipeline, with
  public reads only: every recorded pack and needed answer, and every data object (in a data
  pack `ship` confirmed, or alone), is on the Blob store; no two packs disagree on an answer;
  and `dvc.lock` matches the code (a commit added after shipping, or a rebase without a
  rebuild, fails). With GitHub's "require branches
  to be up to date before merging", the merged tree is exactly the shipped one, so ordinary
  merge commits work; `git bisect --first-parent` walks the shipped states.
- **Old artifacts.** A build replays artifacts made before sidecars existed from the cache, at
  $0, and writes their sidecars without touching them. `uv run sfzmaps answers --baseline`
  records whatever couldn't be replayed (its answers aren't cached under today's prompt) in
  `pipeline/cache-baseline.json`, by exact md5, accepted as it is; `--check` lists live
  artifacts still without a record.
- **The store's limits.** These settings come from Vercel's Blob docs, not from tests against
  the store (testing there risks its DDoS protection). Pro rate limits are 4,500 advanced
  operations a minute (75/s) and 7,200 simple ones (120/s), counted per blob: "Deleting 100
  blobs in a batch counts as 100 operations, not one." Deletes are free but count as advanced
  operations. Multipart parts must be at least 5 MB and equal in size except the last, and the
  docs recommend multipart for files over 100 MB.
- **Never delete an answer to force a re-read**: packs keep one answer per key. Change the prompt
  or its context instead (the key changes).

Costs: Blob storage is $0.023/GB-month (about 13 GB of data plus packs); a contributor's first
full `pull` is about $0.65 at $0.05/GB, then only what changed.

## Releasing

`uv run sfzmaps release` does everything public after a merge, in order. It stops at the first
problem:

1. Checks: the tree is clean, `HEAD` is `origin/main` (after a fetch), DVC data matches
   `dvc.lock`, and this commit's cache is shipped (the `Cache shipped` workflow's check).
2. `sfzmaps downloads` (the GIS export).
3. `site/scripts/publish-tiles.mjs`, then `publish-downloads.mjs`, with the Blob token from
   `site/.env.local` or `SFZMAPS_ENV_FILE`, like `ship`.
4. If `site/lib/published.json` or `site/lib/downloads.json` changed: commit them on
   `release/<date>-<sha7>` (Pacific date), push, and open a PR (`gh`). Merging it deploys. With
   `--no-pr` it only commits. Nothing changed prints "nothing to release".

`--dry-run` runs the checks and export and passes `--dry-run` to both scripts; nothing is
uploaded, committed or pushed. The scripts stay runnable alone (see `site/README.md`). The
download page goes live on the first release that lists files.

## Toolchain

`pixi.toml` declares every system tool the pipeline runs, from conda-forge, and `pixi.lock`
pins each package's exact build and hash: GDAL 3.13.3 (`gdalwarp`, `gdal_translate`,
`gdaladdo`, `gdalbuildvrt`, `gdal_rasterize`, `gdalinfo`, `ogr2ogr`), poppler 26.10.0
(`pdftoppm`, `pdfinfo`), uv and git, plus Node 22 for the site. They install into `.pixi/`
inside the repository. Tile archives are written with the `pmtiles` Python library (in
`uv.lock`), so the separate pmtiles command-line tool isn't needed.

Every build stage depends on `pixi.toml` and `pixi.lock`, and `sfzmaps build` refuses to run
unless every tool comes from the pixi environment, so `dvc.lock` names the toolchain as well as
the code (`--system-tools` overrides it, loudly). Shortcuts from the repository root:
`pixi run build`, `pixi run doctor`, `pixi run doctor-renders`.

`sfzmaps doctor` lists each tool's version and where it comes from, and the GDAL and GEOS
versions the Python wheels bundle (rasterio, pyogrio, shapely; pinned by `uv.lock`, separately
from the GDAL tools). `--renders N` re-renders N PDF pages per series into a temporary folder
and compares them with the `page_sha256` their `page.json` recorded.

**Before changing any version in `pixi.toml`** (or running `pixi update`), run
`pixi run doctor-renders`. Page renders feed the vision-answer cache keys, so a poppler (or
font or image library) that renders different bytes makes labels and classification re-read,
and bill, every page that changed. With the current lock, 63 sampled pages (3 per series)
rendered byte for byte as the existing ones, which were made with Homebrew's poppler 26.10.0.

## Recreate without re-running

```sh
uv run sfzmaps fetch && uv run sfzmaps pull && uv run sfzmaps build
```

`fetch` mirrors the scans from Drive; `pull` brings this commit's processed data, reference data
and the answers it was made from ([Shipping the cache](#shipping-the-cache)); the hand-made
inputs are already in git. `build` then finds nothing to do. Re-reading costs **zero API spend
only if your renders are byte-identical** to the ones that filled the cache (the pixi toolchain
makes them so; `pixi run doctor-renders` checks). `page.png` is not shared; `run`
re-renders it from the raw scan, and the cache keys hash the image pixels.
`page.json` and `labels.json` store the `page_sha256` of the `page.png` they came
from. If your fresh render has the same hash, classification and labels count as
current and nothing is re-read. A different poppler (PDF render) or OpenCV version
can change the bytes. Then the labels are re-read and **billed**. Run
`uv run sfzmaps estimate --all` after `fetch` and `pull`, before `run`: it shows how
many calls would miss the cache. Either way, `run` re-warps and re-tiles locally
where outputs are missing.

Hand-made inputs (reviews, seeds, page corrections, zoning overrides, the label scale) are in
git under `inputs/` (see [Inputs](#inputs)), so `pull` never touches them; it skips any an older
bundle still carries.

## The pass rule

A sheet **passes** when, measured on **held-out blocks** (blocks left out of the
warp fit, 5 folds):

- median corner error ≤ `PASS_MEDIAN_FT × k` (7 ft × k), and
- 95th-percentile corner error ≤ `PASS_P95_FT × k` (15 ft × k), and
- the sheet is not misplaced, the fit converged, placement was unambiguous, at
  least 40 corners were scored, and real change is at most 15% of matches.

**Scale factor (owner's decision).** Corner error is roughly constant in scan
pixels, and the 1980+ PDFs are coarse office-copier scans (~4 ft/px against
~2.7 ft/px for 1929–75). So `k = max(1, ft_per_px / 2.7)`
(`config.PASS_REF_FT_PER_PX`), where `ft_per_px` is the sheet's final transform
scale (the median local scale over the page). A 4 ft/px sheet passes at median
≤ 10.4 ft and p95 ≤ 22.2 ft.

**Bar history (owner's decisions).** 5 ft / 12 ft on 2026-10-29; relaxed to
7 ft / 15 ft on 2026-10-30, once the zoning colors followed today's lot lines,
because the sheets between 5 and 7 ft (at most 4 screen pixels at z18) didn't
look misaligned. Each `score.json` records `ft_per_px`, `pass_scale_k`,
`pass_median_limit_ft` and `pass_p95_limit_ft`. The review page and the editor
show the limit that was actually applied.

Every other map sheet is `needs_review`, with the first failing reason
(`misplaced`, `not_converged`, `ambiguous_placement`, `too_few_matches`,
`excess_real_change`, `holdout_median`, `holdout_p95`, `insufficient_anchors`, or
`error: ...`). `sfzmaps crossref` retries flagged sheets by matching them to the
same sheet in another edition that passed; see the commands table. Non-map pages (covers, text, indexes, legends) are skipped and keep
their kind in `page.json`.

## The sheet warp (`sfzmaps warp`)

The pipeline places each scanned sheet on the city. Small errors remain: a sheet's printed block lines can miss
the real block lines by a few pixels. The sheet warp corrects each sheet's placement so its printed block lines land
on the block lines of its year, then warps the sheet with that correction. It works on every placed sheet of a series.
The published maps don't use the warped sheets yet.

### Running it

```
uv run sfzmaps warp [SERIES ...] [--jobs N] [--force] [--system-tools]
```

With no series it does every series. A sheet is skipped when all its outputs are there and nothing it is made from
has changed since its last warp: the files in "What it reads", the code the warp runs, and the versions of the
libraries that compute it (PyTorch, NumPy, SciPy, OpenCV, Shapely, scikit-image, and the GDAL of `gdalwarp`).
`--force` redoes it. `uv run sfzmaps build` runs one `warp-<series>` stage per series when its inputs are stale, like
every other stage. Warping needs the pinned GDAL, so run inside the pixi environment (`pixi shell` or `pixi run`),
as for the rest of the pipeline; like `sfzmaps build`, it refuses tools from elsewhere unless you pass
`--system-tools`. The fits use PyTorch, which comes with the pipeline's own dependencies.

The fits run on the Mac's GPU when there is one, and on the CPU otherwise, which is slower.
`SFZMAPS_WARP_DEVICE=cpu` or `=mps` (the GPU) picks one. The method is deterministic: a sheet fitted twice on the same
device gets the same correction. The GPU and the CPU give slightly different corrections from each other, so each sheet's
`scores.json` records the device that fitted it. The sheet's freshness record also holds the macOS version, so a
different device or macOS version refits the sheet.

Mind memory: each warp fits `--jobs` sheets at once (6 by default), and a large sheet's fit takes several GB.
`sfzmaps build -j N` runs N series at once and shares one budget of 6 sheets among them: each warp stage fits
`max(1, 6 // N)` sheets at a time. So `-j 1` does one series with 6 sheets at once, `-j 2` two series with 3 each, and
`-j 4` four series with 1 each. The total is at most 6 sheets while N is 6 or less (each series gets at least 1),
which is safe on a 64 GB Mac.

### How fast it is

Each probabilistic stage stops when its loss has settled (it fell by less than 1e-4 of itself over 25 steps, after at
least 50 steps), with 800 steps as the most it takes. The preparation is faster too: distance maps by OpenCV, the
crop's features computed once, and block-side points placed through a precomputed grid that agrees with the exact
placement to within 0.001 px.

- The fits on five real sheets run 6 to 25 times faster than before (1948 sheet 11: 245 s down to 10 s), with the same
  scores: the mean share of block-side points within 1 px changed by +0.0001.
- Preparing the two biggest sheets went from 152 s to 36 s and from 286 s to 58 s.
- Whole sheets, 1929 (12 sheets) on the GPU: 2.95 sheets a minute with 3 sheets at a time, 4.07 with 6 (the default),
  4.24 with 9.

### The steps, for each sheet

1. **What to aim at.** Points every 2 ft along the side of every block of the sheet's year, inside the sheet's drawn
   map. On the scan, the centre line of every stroke no wider than a block line, and for each direction a map of the
   distance to the nearest such line running that way. A point fits the print when it lies within 1 px of a printed
   line running its way.
2. **The first correction.** Sides the print already holds stay where they are. Every other side is moved onto the
   nearest printed line running its way, carried outward from the held sides so neighbouring sides move together. The
   correction is a smooth field (a grid every 50 px) fitted to all of that.
3. **The probabilistic correction.** Each block-side point either lies on a printed block line running its way, or
   isn't printed (missing, or hidden under heavy ink). Ink counts as a block line by how likely its stroke width says
   it is. The fit is the smooth field that makes all the points together most probable, held to stretching about as
   much as today's placement does, with a steep penalty wherever it would fold the scan over. It starts blurred to half
   a street width and sharpens in stages.
4. **The choice.** The probabilistic correction where it puts at least as many block-side points within 1 px of the
   print as the first correction and never folds the scan; otherwise the first correction.
5. **Street edges and later blocks (2000 sheets p09 and p10 only).** These two don't go through the choice: they
   get the probabilistic correction, fitted with this extra evidence as well. Every other sheet is fitted to the blocks of
   its own year. On these two, much of the printed street grid has no blocks in the year's lot records (land held
   whole then and split up later, such as the Hunters Point shipyard on p09), so the fit has little to hold on to
   there. They also use the outlines of later blocks and the edges of every street, each point counted by how little
   the year's own blocks already explain it. Otherwise the fit is the probabilistic correction.
6. **The warp.** The scan is first resampled by the correction, then warped as the pipeline warps it otherwise (its
   control-point grid and resolution), so wherever the correction is zero the result is the same as before. A sheet
   with nothing to fit is written with today's placement, uncorrected.

### What it reads

| What | Where | Comes from |
|---|---|---|
| each sheet's scan, placement and outputs (`page.png`, `page.json`, `transform.json`, `refine.json`, `blocks_px.geojson`, `warped.tif`, `warped_paper.tif` and their footprints) | `data/work/<series>/<sheet>/` | the pipeline's build from the scans (`sfzmaps pull` brings them; `sfzmaps build` remakes them) |
| a page's hand-made patch, if it has one (the warp takes the mask boxes from the patched `page.json`) | `inputs/<series>/pages/<sheet>.json` | made by hand |
| the lots of each map's year, and every lot ever recorded | `data/ref/parcels.parquet` | DataSF's parcels, the pipeline's reference data |
| street centre lines (street edges only) | `data/ref/streets.parquet` | DataSF, the pipeline's reference data |
| the old placement check's verdict, which `scores.json` carries | `data/work/<series>/<sheet>/score.json` | the pipeline's build |
| GDAL, for the warps | the pixi environment | `pixi.toml`, `pixi.lock` |
| PyTorch, for the fits | the pipeline's Python environment | `pyproject.toml`, `uv.lock` |

### What it writes

`pipeline/data/warp/<series>/<sheet>/`:

| File | What |
|---|---|
| `transform.json` | today's placement plus `"correction": {"file": "correction.npz"}`; load it with `SheetTransform.from_json(d, folder=...)`, which applies the correction itself |
| `correction.npz` | the chosen correction and which kind it is |
| `scores.json` | the scores, the choice, the old placement check's verdict, or why the sheet was left uncorrected |
| `warped.tif`, `warped_paper.tif` | the sheet warped with its correction, and the paper layer, each with its footprint |

### Results

Only what has been checked. "The approved run" means the corrections the owner looked at and approved on
2026-11-09, made by earlier scripts.

- **Every sheet, 2026-11-14:** all 357 placed sheets corrected and warped on the Mac's GPU in 81 minutes
  (`sfzmaps build -j 2`); none was left uncorrected.
- **As good as before it got faster.** On the 149 sheets the earlier (CPU) method had corrected, the share of
  block-side points within 1 px of the print is the same on average (0.6786 both); no sheet is more than 0.004 lower,
  and 148 of the 149 choose the same correction. None folds.
- **The 208 sheets corrected for the first time** (the rest of 1972–1980, and the 1980–2000 height, special-use and
  1996–2000 scans) all fit the print better than the pipeline's own placement: per series, the share within 1 px
  rises from 0.39–0.67 to 0.46–0.80, and every sheet gains.
- **Sheets printed from the same drawing agree better once corrected** (a period's zoning, height and special-use
  scans, and the 1996–2000 scan against the 2000 map): the share of one sheet's ink within 4 ft of the other's rises
  on nearly every pair (zoning against height, every pair in every period: about 0.50 → 0.55). The exception is 2000
  p09, fitted with street edges and later blocks, against the scans of the same drawing that aren't (its special-use
  sheet 0.81 → 0.68, the 1996–2000 scan 0.89 → 0.78): see "Next".
- **The same every run:** a sheet fitted twice in separate processes on the same Mac gives byte-identical corrections.
- **Before the speed-up, the port changed no numbers:** 1921's corrections were identical to the approved run's
  rebuild of 2026-11-13 (which reproduced the approved run byte for byte), and every correction checked was identical
  to what the earlier scripts made.
- **No text removal.** The approved run took the printed zoning codes out of the ink before fitting. Fitted with and
  without on the 13 2000 sheets, the scores were the same (within 1 px 0.759 both), so the warp reads no text.
- **2000 is fitted like every other year**, so its corrections differ from the approved run's, which fitted the 2000
  sheets on their own measured area with a fixed scale. Scored on the same points, the old 2000 setup's corrections
  and the new ones are equal (within 1 px 0.749 against 0.748, within 2 px 0.812 both).
- **The faster method does not reproduce the approved run byte for byte.** Its corrections are judged by the same
  scores instead.

### Next

1. Look at the seams where warped sheets meet, by eye, across every series (spot checks on 1948 and 2000 show no new
   jogs or doubled lines).
2. Decide whether street edges and later blocks should apply by ground rather than by sheet name: today only 2000 p09
   and p10 use them, so p09 now disagrees with the other scans of the same drawing. Test it on every sheet; the owner
   decides.
3. Move whatever reads the old pipeline's sheets onto `data/warp`.

## The 2000 solver (`sfzmaps solver`)

The inputs of the 2000 joint solver are made by `sfzmaps solver <part>` commands, each one a stage of
`sfzmaps build` (a "command stage": a `Stage` with `command`, `reads`, `writes` and `code`). A command stage's
dependencies are the source files in the import closure of its `code` modules, `pyproject.toml`, `uv.lock` and its
`reads`; it recomputes its whole output when any of them changes, and needs no `prepare()`. Of a checked-in inputs
folder a stage reads only the file it uses, so a README or `PROVENANCE.md` reruns nothing, and
`table-corrections.csv` (read only in scoring) reruns only the solve. A stage's code deps leave out the modules of its
`cut` (`build.OUTPUT_NEUTRAL`: `sfzmaps.gpu`, whose turns decide when a stage computes on the GPU, never what), and
the two package `__init__.py` files are one line each, so an edit to the GPU's turns or to the module list below
reruns nothing. Outputs go under `data/solver/`: `sfzmaps.solver.paths` names where the earlier stages and the text
detector read and write, and `sfzmaps.solver.runs` the walls' and the solve's folders, so a change to the latter
reruns neither the detector nor the stages before it.

The modules (`src/sfzmaps/solver/`):

- `paths`, `runs`: the locations above; `threads`: one thread for the maths libraries where the walls, the
  placement, the report and the solve compute;
- `grid`: the raster over the city that the page, the lots and the walls share; `codes`: zoning codes in one spelling;
- `frame`, `page`: a scanned 2000 page's printed frame, measured, and the merged page (`sfzmaps solver page`);
- `table`, `lots`: the April 2000 table and its lots on the grid (`sfzmaps solver lots`);
- `saved`, `prints`: the saved vision answers, read by file, sheet or cache key (nothing asks a model), and every
  printed zoning code with its saved reads (`sfzmaps solver prints`);
- `textnet/`: the text detector (`data`, `model`, `train`, `predict`, `stage`: `sfzmaps solver text Q`) and its
  scores (`evaluate`: `sfzmaps solver text-scores`);
- `labels_stage` with `trust`, `streetside`, `lotsize`, `strokes`: what the solver knows about the labels and the ink
  (`sfzmaps solver labels`);
- `textgrid`, `wallevidence`, `walls`: the detectors' chance of text on the grid, out of fold, and the wall odds
  between neighbouring lots (`sfzmaps solver walls`);
- `city`, `labels`, `placement`, `energy`, `search`, `solve`: the city as the solve reads it, each quarter's readings
  of the labels, where each label belongs (`sfzmaps solver placement`), the energy over every lot's code, its
  search, and the four held-out solves (`sfzmaps solver solve`);
- `borders`, `measures`, `holes`, `published`, `report`: scoring a run against the table (`sfzmaps solver score`,
  `holes`, `published`) and the solve's report against J7's.

Hand-made and downloaded inputs are in `inputs/zoning-2000/` (see its README): the April 2000 table
(`april2000.geojson.gz`, downloaded from DataSF by `sfzmaps solver fetch-table`, with `PROVENANCE.md`) and
`table-corrections.csv`, which is used only in scoring. The lot index is the row order of the table file.

The solver's grid (`sfzmaps.solver.grid`) is one raster over the city in EPSG:2227 feet, 4 ft per px, 11,900 ×
9,825 px, north up, its top-left corner at (5979300, 2124100): col = (x − 5979300) / 4, row = (2124100 − y) / 4. The
merged page, the lots and the walls all sit on it. The grid module also holds what every part reads alike: a page px
is ink below 128 (`INK_BELOW`), and the city's four quarters, which the fits hold out in turn, split at
`QUARTER_SPLIT` (`quarter_of`: nw 0, ne 1, sw 2, se 3).

**The merged page** (`sfzmaps solver page`, stage `solver-page`, after `warp-2000-digital-zoning`) lays every warped
2000 zoning sheet onto the grid as one picture of the printed ink. Each sheet's crop is its printed frame, found on
`page.png` as four straight, slightly tilted lines and taken 4 px inside them, so the legend box's slightly bowed top
line stays out (`solver/frame.py`; its search windows are fixed page px), minus the cut boxes in `inputs/merged-page/` (notes and the north arrow printed inside the
frame). The sheet's corrected placement (`data/warp/2000-digital-zoning/<sheet>/transform.json`) is evaluated every
8 grid px, read bilinearly between, and each grid px takes the nearest page px: the only resampling, so the 1-bit
scans stay 1-bit. A grid px is ink where any sheet shows ink (page < 128) inside its crop. It writes
`data/solver/page/ink.png` (0 = ink, 255 = paper), `sheets.png` (how many sheets' crops cover each px; 0 = not
mapped) and `frames.json` (each sheet's crop quad, page px). The same inputs give the same bytes.

**The lots** (`sfzmaps solver lots`, stage `solver-lots`, after `solver-page`) paint the table's lots on the same
grid. Each polygon's exterior is filled with `cv2.fillPoly`, largest lot first, with its corners at `to_grid - 0.5`
to 1/16 px (fillPoly takes integer coordinates as px centres), so a lot's px are the grid px whose centres are
inside it and line up with `ink.png`. `grown` gives every px its nearest lot (`distance_transform_edt`), so lots
facing each other across a street are neighbours. `pairs` holds each two lots whose `grown` regions meet:
`a < b`, the px pairs where both px are inside lots (touching) and the others (street), sorted by (a, b); a
pair's border is (touching + street) × 4 ft. A lot is **scored** when its code is known (one district, not blank)
and more than half of its px are on the mapped area (`sheets.png > 0`). `quarter` (nw 0, ne 1, sw 2, se 3) comes
from the lot's centroid and `QUARTER_SPLIT`, the median centroid of the first run's scored lots, kept so held-out
numbers compare with it. It writes `data/solver/lots/lots.npz` (`lot`, `grown`, `pairs`, `scored`, `known`,
`quarter`, `table`: each lot's code, as `known_and_codes`) and `table.parquet` (the loaded table, in lot order).
The same inputs give the same bytes; `sfzmaps.solver.lots.load_lots()` reads it back.

**The prints** (`sfzmaps solver prints`, stage `solver-prints`, after `solver-lots`) are the zoning codes printed on
the sheets, each with every saved read of it. Nothing asks a model: every answer comes from `saved-reads/answers/`
(`sfzmaps.solver.saved`). Terra's text boxes (`text-on-every-sheet`) are decoded as the script that asked them did
(each tile's items moved by its `page_box`; the copy farthest from its tile's edge kept, tiles taken row by row; a
word inside a longer label that holds it dropped). A zoning-code box with Sol's label points (`district_labels.json`,
kind "use") inside it is a print read by both (kind 0; the point nearest the box's centre gives Sol's read, and all
the points inside are used up), a box with none is Terra's alone (kind 1), and a point in no box is Sol's alone
(kind 2). A print sits at its box's centre (or Sol's point), goes to the map through the sheet's corrected placement,
and is kept inside the sheet's warped footprint and on the mapped area; `lab_lot` is `grown` at the grid px holding
it and `street` is `lot < 0` there. The turned pass (`experiments-2000/text-positions/sideways-text`, page px) adds a
reading to every print whose point is inside a turned box (edges included) or whose Terra box it overlaps by IoU > 0.3,
the box farthest from its tile's edge winning; a turned box on no print with a known code (the table's, or one Sol
reads on a print) is a print of its own (kind 3). The third read and its turned re-check name no sheet or picture:
each print's crop and question are made again as `read3.py` and `read3r.py` made them, and the answer is found by its
cache key (`llm._key`; the answer formats keep the scripts' class names and fields, and no docstrings, since the JSON
schema is in the key). The keys are made with the model, picture limit and point-box size the reads were asked with
(Sol at low effort, 2,048 px, 64 px; pinned in `solver/prints.py`), never the pipeline's settings of today. A key not found is counted, and that print has no such read; with fewer than 95% of a read's
keys found the stage stops. It writes `data/solver/prints/prints.parquet` (one row per print; the columns are listed
in `solver/prints.py`), `terra_boxes.json` (every sheet's decoded Terra boxes, every kind, page px) and
`summary.json` (prints by kind; keys asked, found and missing per read). The same inputs give the same bytes.

**The text detector** (`sfzmaps solver text Q [--iterations N] [--device cpu|mps]`, stages `solver-text-nw`,
`-ne`, `-sw` and `-se`, after `solver-prints`) scores every px of every sheet's page as printed text or not, so the
walls can weigh ink by how likely it is to be a district line rather than lettering. It is a small U-Net
(`solver/textnet/`: width 24, 5 levels, 7 outputs: text, 4 kinds, the reading direction's cosine and sine) over the
ink mask (`page.png` < 128) of each sheet's own page, unwarped. It learns only from Terra's boxes
(`data/solver/prints/terra_boxes.json`, every kind, each drawn as the word's rotated rectangle in page px). Text Terra
may have missed is taught neither way: the boxes of the other text passes
(`saved-reads/answers/experiments-2000/text-positions/`) and 30-px discs at Sol's label points, outside every Terra
box. Each quarter's model holds that quarter out: a page px's quarter is `grid.quarter_of` of its map point through
the sheet's corrected placement (evaluated every 32 px, bilinear between); held-out px carry no loss and no crop
centres on one. Training: 8,000 steps of 12 crops of 320 px (half centred on text, the rest on any ink; each scaled
by 0.6–1.6 and turned by quarter turns, the reading angle with it); loss = text cross-entropy + 0.5 × kind
cross-entropy + 0.5 × (1 − cosine) of the reading direction; AdamW at 2e-3 with a one-cycle schedule. Every draw comes
from one seeded numpy generator and the weights from seeded torch, with deterministic algorithms on 4 threads. The
model then scores every sheet's page in 1,024-px tiles seen with 64 px of context. A stage
(`solver/textnet/stage.py`) makes its folder afresh and writes only `data/solver/text-<Q>/model.pt` (the weights,
with the iterations, device and seed) and `chance/<sheet>.png` (uint8, the chance of text × 255, page px). Its code is
the detector's alone: the lots, the table, the merged page, the scoring, the walls' and the solve's folders and the
GPU's turns are not in it, so a change to them retrains nothing. Its reads are named file by file for each of the 13
sheets of `paths.SHEETS` (a fixed list, so `dvc.yaml` is the same in a fresh clone before anything is pulled):
`transform.json` and `correction.npz` in `data/warp/2000-digital-zoning/<sheet>/`, `page.png` and
`district_labels.json` in `data/work/2000-digital-zoning/<sheet>/`; no other file there retrains it. Before reading
them it checks that the warp placed exactly those sheets (`paths.check_sheets`).

**The text scores** (`sfzmaps solver text-scores`, stage `solver-text-scores`, after the four text stages) score each
quarter's model on the quarter it held out, from the chances it wrote (`solver/textnet/evaluate.py`), on the map only
(inside the sheet's warped footprint: the legend, alike on every sheet, is left out). It reads the detector's files
of each sheet and its `warped.footprint.geojson`, file by file, and the lots. It writes
`data/solver/text-scores/evaluation_<Q>.json`: AUC and log loss of text ink against other ink, the share of text ink
found, Terra's words found by kind, pieces of text found where Terra drew no box (listed for a look), and the share
of ink on the table's district lines (borders between lots whose known codes differ, read on the grid, away from
every box) called text.

The detector trains on `mps` where there is one (else `cpu`; `--device` overrides): two 200-step runs on `mps` gave
bit-identical weights and predicted chance (max difference 0), at 0.39 s a step against 3.1 s on `cpu` (about 52 min
against 6.9 h for 8,000 steps), once the loss avoided boolean-mask indexing, whose backward `mps` has no deterministic
kernel for (`index_put_with_accumulate_mps`). That holds only for one process on the GPU at a time: with two
processes computing on `mps` at once, the same step on the same input gives different numbers each time (outputs off
by up to 2.5 in a logit, training-mode batch norm by up to 3,000) and now and then a gradient that is all NaN. So
work on `mps` takes a turn (`sfzmaps.gpu`, an `flock` on `/tmp/sfzmaps-mps.lock`, one file for the whole machine).
macOS deletes files in `/tmp` that nobody has touched for 3 days, so every turn taken sets the file's times to now,
and a turn whose file was deleted (and maybe made again by another process) while it waited is taken again on the
file now at that path. A detector's stage trains and scores in one exclusive turn: nothing else computes on the GPU
meanwhile. The sheet warp's fits (`sfzmaps warp`) take a shared turn: warps run beside each other, never beside a
detector. The tests hold a shared turn for their whole run, taken without waiting; while a detector has the GPU they
never use it (the warp's fits run on the CPU and the tests marked `mps` are skipped). A loss that is not finite stops
the training.

**The trust of each reading** (`solver/trust.py`, run by the `solver-labels` stage) is the chance that a code read
on a print is the district printed there. Each different code among a print's five reads (Sol, Terra, the third
read, the re-check, the turned pass) is one reading. One logistic regression on 23 features (listed in
`trust.FEATURES`: which reads name the code, Sol's and the third read's confidence as −log(max(1 − c, 0.001)), the
print's height as the log of its ratio to the median, in a street, a leader line, printed sideways, and for prints
found only by the turned pass the log of the distance to the nearest other print over 100 ft (at least 1 ft), in a
street, a single letter) gives each reading its trust. The target is that the code is the table's code of the lot
under the print. The fit is Newton's method from zero (at most 200 steps, stopping when no weight moves 1e-8), with a
ridge of 1e-3 on each weight and 1,000 on the turned-pass-only distance and street weights. There is one fit per
quarter, trained only on readings whose lot (the lot under the print's text) has a known code in the uncorrected
table and lies outside that quarter, by the lot's quarter, not the print's position. It writes `data/solver/labels/readings_<Q>.npz` (`names`, and per reading `print_index` into `prints.parquet`,
`code_index` into `names` and `trust`) and `trust.json` (each fit's weights and held-out log loss). The same inputs
give the same bytes.

**The labels** (`sfzmaps solver labels`, stage `solver-labels`, after `solver-prints`) are everything the solver
knows about the prints and the ink around them, in `data/solver/labels/`: the trust above, then the street sides, the
lot sizes and the strokes (`solver/labels_stage.py`). Nothing asks a model, and the same inputs give the same bytes.

- **Street sides** (`solver/streetside.py`): a code printed on street ground also speaks, at a lower trust, for the
  lot across the street. For each print with a lot under its text (side A), the nearest street centreline in
  `data/ref/streets.parquet` gives the sides: a lot's side is where its nearest point to the print falls along the
  centreline's normal (its direction from 5 ft before the print's foot to 5 ft after). Side B is the nearest lot
  other than A on the other side, searched within 100 ft of the print, the reach doubling up to 3,200 ft. A
  centreline is a freeway when its `st_type` is HWY or RAMP or its name is a numbered highway or interstate
  (`^(HWY|I-)\s*\d`). Four rates, fitted once per held-out quarter on the uncorrected table, are how often the
  table's code at side A, or at side B, is the print's top reading (its most trusted in `readings_<Q>.npz`), on
  ordinary streets (`rA_st`, `rB_st`) and freeways (`rA_fw`, `rB_fw`). A print counts for a side when it is on street
  ground and that side's lot has a known code outside the quarter, so no fit reads a code of its own quarter. It
  writes `streetside_<Q>.npz` (per print `lotB`, `onst`, `isfw`, and the four rates).
- **Lot size** (`solver/lotsize.py`): what a lot's size says about its code. Lots are binned by log area at
  `arange(5, 14.01, 0.25)` (36 bins; smaller and larger lots join the end bins). Per code, each bin's count is
  blended with half of each neighbour's (no wraparound), smoothed toward the overall size distribution with 36
  pseudo-lots, and turned into log p(bin | code) − log p(bin): a code with no training lots carries no evidence. One
  fit per held-out quarter, on the scored lots outside it with a known code in the uncorrected table. It writes
  `lotsize.npz` (`codes`, `edges`, `PMI` [quarter, code, bin]).
- **Strokes** (`solver/strokes.py`): the merged `ink.png`, raw (text and lines alike), thinned to centre lines
  (`skeletonize`). Nodes are runs of centre-line px with other than two neighbours (ends and junctions); a stroke runs
  between two nodes, or round a closed loop. Each keeps its path (map ft, at grid px centres), its width (twice the
  distance to the paper, × 4 ft), its length and its end nodes' degree; a dead end shorter than its median width is
  tagged a spur. It writes `strokes.npz` (`xy`, `w`, `off`, and per stroke `a`, `b`, `length`, `wmed`, `wiqr`,
  `spur`, `dega`, `degb`). On the 2000 page: about 172,000 strokes and 3,240 miles, in about 25 s and 3 GB.

**The walls** (`sfzmaps solver walls [--workers N] [--candidates A,B] [--evaluate A,B ...]`, stage `solver-walls`,
after `solver-labels` and the four text stages; `solver/wallevidence.py` and `solver/walls.py`) give every pair of
neighbouring lots (each row of `lots.npz` `pairs`) the log odds that a printed district line runs between them. The
stage reads what they read: the merged page (with `frames.json`), the cut boxes, the lots, `strokes.npz`,
`terra_boxes.json` and the warp's placements, and for the text discount the four detectors' `chance/` and the sheets'
pages (`data/work/2000-digital-zoning`) they are read on.
Nothing in them decides: every measure is a feature, and one fit weighs them all. The ink is read through the raw
page's strokes (`strokes.npz`; no cleaned page, no line chain). A pair's border px are both px of every 4-neighbour
px pair where `grown` changes between its lots, and each feature is a mean over them; distances enter as hat
functions over 0 / 6 / 14 / 26 / 42 / 64 ft. The 208 border features, in `wallevidence.GROUPS` order:

- **stroke evidence** (60): the nearest stroke (the widest on top where strokes cross): its distance × hats of its
  median width (0–40 ft), and its distance × its letter share, log length, steadiness (width spread over median
  width, capped at 3) and spur tag. The letter share is the share of the stroke's points inside Terra's text boxes
  (`prints/terra_boxes.json`, every kind, each the word's rectangle turned to its reading angle), kept when the box's
  centre is inside the sheet's crop (`page/frames.json`) and in no cut box (`inputs/merged-page/`), and put on the
  grid through the sheet's corrected placement;
- **whole line** (30): the nearest stroke's whole line (strokes joined at each node, the straightest pairing first):
  its length, whether it closes on itself, the strokes of other lines ending on it, its sharpest turn, its width;
- **straight closing** (30): from every loose stroke end, the straight gap to the nearest point of another stroke
  within 250 ft (a search reach; the fit weighs the length): its length, how well it carries on the end's direction,
  the smaller width at its ends, its share over lots;
- **enclosure** (16): at each level of stroke width ≥ 6 / 12 / 18 / 26 ft × closing paths ≤ 0 / 60 / 150 / 400 ft,
  the walls are those strokes, those closing paths and the city limit (the edge of the mapped area, `sheets.png` >
  0: a wall at every level); a pair is apart when its lots are in different 4-connected regions, each lot in the
  region holding most of its px. A closing path is the shortest way (`MCP_Geometric`) from a loose end over street
  ground and lot lines, never through a lot, to a stroke sharing no node with it or to the city limit, within 512 ft;
- **closing path** (24): the nearest closing path: its length, how well it carries on, the smaller width at its ends
  (the city limit counts 40 ft);
- **profile** (48): for each width level 0 / 6 / 12 / 18 / 24 / 32 ft, the distance to the nearest stroke at least
  that wide, and, while it is within 26 ft, its letter share and log length.

The fit adds the same 208 times "the pair faces across a street" (no touching px) and the lots' character: hats of
the difference of their log areas, that times across a street, and hats of the smaller log area. It is one ridge
logistic regression (ridge 0.1, Newton's method from zero) per quarter, trained on the pairs with both lots known and
both outside the quarter; the target is that the two lots' codes in the table as downloaded differ. There is no
whole-wall stage: how a line holds together along its length is left to the solve's energy. It writes
`data/solver/walls/walls_<Q>.npz` (`lo_ab`, float32 per pair row, the odds of the fit without quarter Q) and
`evaluation.json` (per quarter the log loss on the held-out pairs, both lots known and in the quarter, and their
number). It writes them into a fresh folder beside `data/solver/walls` and renames that into its place, so nothing of
an earlier fit (a `text_chance.png` its candidates no longer read) is left. The closing paths run on `--workers`
processes in bands of grid rows; the result is the same for any number, and the same inputs give the same odds. On
the 2000 page: 437,000 pairs over 16.3 million border px, 60,716 closing paths from 68,363 loose ends, in about 4
minutes and 12 GB with 6 workers.

**Candidate wall features** (`walls.CANDIDATES`). Each is tried against the set above and kept only if it lowers the
held-out log loss in at least three quarters; the stage fits with those kept, `walls.DEFAULT_CANDIDATES` in code
(none yet). `--candidates A,B` fits with others as an experiment: into
`data/solver/experiments/walls-<the names sorted, joined by +>/` (`runs.walls_experiment_dir`), for a solve to read
with `--walls DIR`, never over the stage's walls. `--evaluate A,B` (repeatable, `none` for the set above;
`walls.evaluate_candidates`) scores several configurations from inputs read once and writes nothing. Most candidates
read the text detector's chance of text on the grid (`solver/textgrid.py`, computed afresh each time; a fit writes the
one it fitted with to its folder's `text_chance.png`, never read back).
Each sheet's chance is read through its corrected placement at the same page px as the merged page reads its ink
(`page.sheet_px`, the same crop), from the detector that held out that page px (the quarter of its map point, as the
detector's own held-out mask has it), so no px is scored by a model that learned from it; a grid px takes the
largest chance among the sheets that inked it, 0 where none did. It needs all four quarters' detectors.

- `text_weighted`: the stroke evidence and the profile weigh each border px's nearest stroke px by (1 − its text
  chance), and read that chance where Terra's letter share was (the same 208 columns);
- `text_weighted_beside_letters`: the same, with Terra's letter-share columns kept beside it (12 more);
- `street_crossing`: for every border px pair, the crossing from the nearest px of one lot to the nearest px of the
  other (the street's whole width, both edges included; for touching lots the two px themselves), and hats over
  0–40 ft of the ink met along it, weighted by (1 − text chance) (6 more). A line printed along a street's edge is
  far from the street's middle, where the border px are;
- `context`: touching pairs get hats of their kind (side by side, between, back to back: how the line between the
  lots' centroids runs along their long axes); pairs across a street get hats of the street's width at the table's
  peaks (25 to 125 ft, then parks and water);
- `no_city_limit` (an ablation): the city limit is no closing's end and no enclosure wall.

**The solve** (`solver/city.py`, `labels.py`, `placement.py`, `energy.py`, `search.py`, `solve.py`) reads the
district of every lot from the page: one energy over every lot's code, minimised for the whole city and held out by
quarter. How it works and how well it did as R&D (run J7) is written up in `docs/research/joint-solver/`, with the
R&D notes behind it. What it reads:

- **The lots** (`city.City`): `lots.npz` and `table.parquet`. Scores count against the table with its by-eye
  corrections (`City.table`, from `table.corrected_codes`); every fit learns from the table as downloaded
  (`City.table_raw`). The true borders of the ceiling check and the table's answer the log prices use the scoring
  table.
- **The walls**: `data/solver/walls/walls_<Q>.npz` `lo_ab`, one log odds per row of `lots.npz` `pairs` that a printed
  district line runs between the two lots, fitted without quarter Q (`city.wall_odds`; NaN is read as 10, a line).
  `--walls DIR` reads them from another folder.
- **The labels** (`labels.readings(Q)`): each reading of `readings_<Q>.npz` speaks for the lot under its print's text
  (`prints.parquet` `lab_lot`); a print on an ordinary street (`streetside_<Q>.npz`: on street ground, not by a
  freeway, a lot across) also speaks for the lot across at trust × min(`rB_st` / `rA_st`, 1). Overlay (SUD) names and
  names of no table code are dropped.

**Runs and their folders** (`runs.solve_dir`). The `solver-solve` stage's run is `final`, in `data/solver/solve/`
(shipped). Every other run is an experiment, in its own `data/solver/experiments/RUN/` (no stage's output, never
shipped), with its own placement, runs, data errors and report; a run's name can't start with `walls-` (those folders
hold the walls of `walls --candidates`). `final` is refused with `--walls` or `--true-borders`, and `placement` and
`holes --save-data-errors` never write into it. A solve builds its run's folder afresh beside it (`.NAME.part`) and
renames it into place, so nothing of an earlier solve is left, and nothing it writes holds a time.

**One thread for the maths libraries** (`solver/threads.py`). The walls' fit, the placement, the report and each
quarter's solve compute in processes whose BLAS and OpenMP run one thread (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `MKL_NUM_THREADS` = 1), so their numbers don't depend on how many
cores the machine has. The libraries read these only as they load, before any command runs, so `sfzmaps solver
walls`, `placement` and `solve` start themselves again in such a process when theirs isn't one, and the solve starts
its quarters' processes so.

The commands, run from `pipeline/`:

- `sfzmaps solver placement RUN [--walls DIR]` fits where each label belongs (`solver/placement.py`): under its text, or
  in a walled area within 500 ft (lots joined by borders with odds ≤ 0) that its leader points into. A conditional
  logit over each label's options (12 features: distance, area size, a stroke of `strokes.npz` from the text into
  the area and its width, labels already there, how well the area's lot sizes fit the code by `lotsize.npz`, …),
  with ridge 1, fitted four times, each without the labels whose lot is in one quarter. Its answers are the owner's
  hand leader moves (the rows of `inputs/2000-digital-zoning/label_fixes.csv` with a fix point, placed on the map
  through each sheet's corrected warp, matched to a print of the same code within 40 page px), and elsewhere the
  table: stay when the lot under the text has the code, else the nearest option whose lots mostly have it. It writes
  `placement.npz` into the experiment's folder (with `walls`, the digest of the walls it was fitted with:
  `city.walls_digest`, of every quarter's odds) and prints each model's held-out hit rate, the hand moves' among
  them. A solve refuses a placement fitted on other walls.
- `sfzmaps solver solve RUN [--walls DIR] [--true-borders]` (stage `solver-solve`, after `solver-walls`, runs
  `solve final`) fits the placement as above, solves the city, lists the data errors among the run's holes and writes
  the report. The solve runs four times side by side, one `sfzmaps solver solve-quarter RUN Q [--keep-costs]` process
  per held-out quarter (one BLAS thread each), each with that quarter's walls, readings and placement model; every lot
  then takes its code from the solve that held its quarter out. The energy charges each label once over the lots it
  may stand for (its top reading's trust calibrated: logit = 0.128 + 1.085 logit(trust), `energy.TOP_READING_FIT`), a
  different code across a border with odds < 0 its −odds, and the same code across a border with odds > 0 its odds;
  a tiny tie-break (1e-4 × the cheapest path from each code's labels) settles lots no label reaches. The search runs
  graph-cut expansion moves (PyMaxflow), then region moves, compound moves and expansion again until the energy stops
  falling; a move is kept only if the exact energy falls. It is discrete: every lot gets the same code every run. Into
  the run's folder it writes `placement.npz`, `runs/RUN-<Q>.npz`, `.log` (the quarter's log: what its process
  prints, with no times; each solve's time and anything a process writes to stderr, a library's warning or a
  traceback, go to the console only) and each solve's costs (`RUN-<Q>-costs.npz`, for `holes`), the join
  `runs/RUN.npz`, the data errors (`data_errors.npz` and `data-errors-2000-digital.csv`, as `holes
  --save-data-errors`) and the report (`solver/report.py`, `report.json` and `report.md`): the six measures (below)
  against J7's, with the data-error parcels left out and with every lot; wrong lots by quarter, the largest
  incidents, the holes by cause, the walls' held-out log loss per quarter (against the scoring table, beside J7's
  quoted numbers, and against the table as downloaded, from their `evaluation.json`), the placement's hit rates; a
  check of `energy.TOP_READING_FIT` on this build's held-out trust (each label's top reading, from the fit that held
  its lot's quarter out, against the table as downloaded there: the log loss of the energy's calibration and of one
  fitted afresh, and stated against right by trust); and what differs from J7's definitions. Each run leaves out its
  own data errors, so runs of this build compare on the every-lot column; the left-out column is there for J7. The
  report holds no times: the same run gives the same bytes. `--true-borders` is the ceiling check: the table's own
  borders as walls (odds ±6) must give back nearly the table.
  The stage reads the lots, the merged page's `ink.png` (the holes' causes), `prints.parquet`, the labels, the
  walls, the warp's placements (the hand leader moves), `label_fixes.csv` and `table-corrections.csv`. On the 2000
  page it takes about 4 minutes (the four solves about 165 s each, side by side), up to 17 GB in one process, and
  writes about 110 MB, most of it the solves' costs.
- `sfzmaps solver score RUN... [--with-data-errors]` scores runs by their mistakes (`solver/measures.py`): the share
  of the table's district lines the run draws and of the lines it draws that are table lines (by count and length),
  the share of zoning labels whose district comes out right, holes (missed runs of table line up to 100 ft that meet
  found line at two or more places), and wrong lots grouped into incidents. Each run's own data-error parcels are
  left out (none for a run that wrote no list, such as `published`).
- `sfzmaps solver holes RUN [--picture PNG] [--save-data-errors]` gives each hole its cause (`solver/holes.py`,
  `solver/borders.py`); `--save-data-errors` writes the parcels where the printed line is drawn over the parcel and
  the table puts it on the other side into the run's folder (`data_errors.npz` and `data-errors-2000-digital.csv`;
  not for `final`, whose solve writes its own).
- `sfzmaps solver published [--districts PATH]` scores the pipeline's published 2000 map
  (`data/work/2000-digital-zoning/districts_raw.parquet`) the same way, as the run `published`.

Dependency notes: the max-flow cuts use PyMaxflow (pinned in `pyproject.toml`), which is GPL-3.0 licensed.

**The fresh-clone check** (`scripts/fresh_clone_check.sh [--clone DIR] [--dry-run] [STAGE...]`) shows the stages
rebuild from a clone. Push the branch first; it clones `origin`'s copy of it into `data/fresh-clone/` (git-ignored; the
folder must be empty), runs `pixi install`, `uv sync --frozen` and `sfzmaps pull`, builds
`warp-2000-digital-zoning` if the pull did not bring it, then runs each stage (default `solver-page solver-lots
solver-prints solver-text-nw solver-labels solver-walls solver-solve`) and compares its outputs with this checkout's
(`scripts/compare_outputs.py`, tested in `tests/test_compare_outputs.py`). Each stage is held to its own epsilon on the
same upstream outputs, not chained end to end: after each stage is compared (or fails), its outputs in the clone are
replaced by this checkout's (`compare_outputs.py --use-reference`), so the next stage runs on exactly the inputs this
checkout's run of it had, and the summary says so. The warp, page, lots, prints, labels, walls and solve must be
byte-identical (an `.npz` by its arrays, a `.parquet` by its table); a text stage's `chance/*.png` may differ by 0.01 × 255
(2 counts) and its `.json` numbers by 1%, and `model.pt` is not compared; `solver-text-scores` numbers match within
1%. Only `nw` is retrained (it trains on the GPU, so run the check when the GPU is free); `text-ne`, `-sw` and `-se`
are copied from this checkout before the walls, which read all four. It runs each stage with `sfzmaps stage`, not
`sfzmaps build --force`: a build also plans every stale upstream stage, and until `sfzmaps ship` has recorded the
solver stages in `dvc.lock` that would retrain the other three quarters. It prints a table (identical, within
epsilon, differs, with the first differences) and exits 1 if any stage differs. CI doesn't run it.

## Where outputs land

Everything is under `pipeline/data/` (git-ignored; set `SFZMAPS_DATA_DIR` to move it):

- `raw/`: the Drive mirror.
- `ref/`: the reference data, the `refdata` stage's output (DVC-tracked and shipped like any
  output): DataSF parcels, streets, today's zoning, Rec and Park properties
  (`rec_park_properties.parquet`), and NPS land tracts (`nps_tracts.parquet`).
- `cache/llm/`: cached API answers.
- `work/<series>/<sheet>/`: per-sheet stages, i.e. `page.png`, `page.json`,
  `labels.json`, `rough.json`, `refine.json`, `changes.geojson`, `transform.json`,
  `score.json`, `warped.tif`, `warped_paper.tif` (after `paper`). Machine output only:
  everything a person decides is in `inputs/`.
- `warp/<series>/<sheet>/`: corrected placements and warped sheets (see The sheet warp).
- `out/<series>/`: `<series>.pmtiles`, `<series>-paper.pmtiles`, `manifest.json` (per-sheet status,
  `needs_review` reasons, scores, limits, bounds), `review.html`.
- `out/ref/blocks.pmtiles`: today's blocks for the review pages.
- `out/run-<timestamp>.log`: tracebacks for any sheet that errored.
- `tmp/`: scratch while tiling (outside `out/`, which DVC versions); safe to delete when no
  build is running.
- `share/`: the `bundle` output.

## Reviewing and editing

1. `uv run sfzmaps review --year 1968` opens the series over today's blocks. The
   sidebar lists each sheet's status, reason, held-out p50/p95 and the limit
   applied. Click a sheet to zoom to it.
2. For a `needs_review` sheet, follow "Open in editor", or run
   `uv run sfzmaps edit --year 1968 --sheet <id>`. The editor shows the scan beside
   today's blocks, with the control points colored by residual and the worst
   areas highlighted. Drag, add, delete or disable points, and mark areas as
   real change or no modern equivalent. The preview and the held-out score update
   as you edit.
3. Save. This writes `inputs/<series>/reviews/<sheet>.json` (commit it) and re-runs warp, score, stitch and tile for that
   sheet. A later automatic run never overwrites a human review unless you pass
   `--force-auto`. You can accept a sheet that still fails, with a note.
   The review records the reviewer as `$SFZMAPS_REVIEWER`, or `"owner"` if that
   is unset. It never records your git email, because reviews are committed to the public repository.
   An edited set of 100 or more enabled points (usually the automatic points the
   editor starts from) is fit with the automatic warp's smoothing,
   `TPS_SMOOTHING`. A sparser hand-placed set uses `MANUAL_TPS_SMOOTHING`.
   The review stores the smoothing it used. The held-out score refits the warp
   on your edited points, leaving out one group of blocks at a time, so moving or
   disabling points changes it.
4. **Auto-fill from my points** (needs 4 or more enabled points of your own; `auto`
   points don't count) seeds the automatic matcher from your points and fills in the
   matched block corners. It never saves: review the result, then Save. Saving an
   auto-fill fits it with usually `TPS_SMOOTHING` (the result is dense; under 100 enabled
   points it is `MANUAL_TPS_SMOOTHING`) and records `autofill` in the review.
5. For 1948 and 1960, the editor offers the owner's manual `.points` and adopts
   them when they score better.

## Checking the district reads

Measure every change to the district readers before shipping it. Nothing here calls the API.

1. Before the change: `uv run sfzmaps districts --snapshot data/check/before`.
2. Make the change and rebuild.
3. Compare: `uv run sfzmaps districts --check --against data/check/before`. It prints one column
   per source:
   - **Review cases** (`inputs/district_cases.csv`): known artifacts, each a bbox with the code
     wrongly read there and the right ones. The counts are wrong / right lots in the box. `seam`
     and `other` cases should go down. `harm` cases mark areas that an earlier change broke, and
     they must stay right.
   - **Answer keys** (`tests/fixtures/district_keys/`).
   - **Same base drawing**. 1960 and 1964 share their drawings, and 1996-2000 p25-p36 and
     2000-digital p01-p12 are two scans of the same twelve drawings. Agreement between them needs
     no key, so it is the best single measure of how repeatable the reader is.
   - **Adjacent eras**. Real rezonings lower these too, so look for drops rather than maximising them.
   - **The largest clusters where the same-base maps disagree**: where to look for new cases. Check
     one against the scans, then add it to `inputs/district_cases.csv` (commit it).
4. Look at the whole city: `uv run sfzmaps districts --map S --against data/check/before` writes
   `data/check/S.png`. It shows before | now, with lots coloured by zoning class and sheet
   footprints outlined. Regressions no case covers show up here.

An experiment can write `<series>_raw.parquet` files (columns `blklot`, `code`, `source_sheet`)
into a directory and pass it to `--against`, with no rebuild.

## Credit: 1948

The 1948 atlas photographs come from the **David Rumsey Map Collection**, not the
owner's scans. Anything published from the 1948 series must carry the credit
"David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries" (`config.RUMSEY_CREDIT`;
likely CC BY-NC-SA). The pipeline writes it into the 1948 PMTiles attribution,
manifest and review page. Keep it wherever the overlay is shown.
