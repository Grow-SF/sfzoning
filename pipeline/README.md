# sfzmaps: georeferencing San Francisco's historic zoning maps

`sfzmaps` turns the scanned zoning maps in the Drive folder `old-planning-maps`
(1921 to 2000-digital) into georeferenced, transparent-paper overlays: one PMTiles
file per year/series, a machine-readable manifest, and a review page. Sheets that
miss the accuracy bar go to a local control-point editor for human review.

Design: `docs/superpowers/specs/2026-10-29-map-georeferencing-design.md`.
Tuning history and final results: `docs/superpowers/notes/2026-10-29-tuning.md`.

## Setup

```sh
brew install gdal pmtiles poppler      # gdalwarp/ogr2ogr (GDAL >= 3.13), pmtiles, pdftoppm
cd pipeline
uv sync
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
| `refdata [--force]` | Download DataSF parcels and street centerlines; build blocks, corners and intersections in `data/ref/`. |
| `run --year S` / `run --all` | Run every page of a series: classify, read labels, rough fit, extract linework, refine, classify real change, score, warp; then stitch, tile, write the manifest and review page. Fresh stages are skipped; human reviews are kept (`--force-auto` ignores them, `--force` reruns everything). `--sheet ID` and `--until STAGE` make a partial run that does not republish the series. |
| `tiles --year S` | Rebuild the stitched VRT, PMTiles, manifest and review page for one series from existing warps. |
| `paper --year S` / `paper --all` | Warp each placed sheet as it looks on paper (opaque, colours untouched, cubic resampling) to `warped_paper.tif` and publish `out/<series>/<series>-paper.pmtiles`: the layer the site shows as the original map. The transparent-paper `<series>.pmtiles` stays for the review page and the district readers. The PDFs are 1-bit black and white at 300 dpi; their white is multiplied onto `warp.PAPER_TINT`, the 1948 photos' cream paper, so every era sits on the same paper (ink stays black; 1948 is untouched). Each sheet keeps its map up to its frame; the printed border rule and the 1948 photos' binding shadows are erased (`warp.erase_border_rules`). The stitch stacks a plain-paper backing (`stitch.paper_backing`, for narrow gaps), every sheet's whole paper (legend and title boxes filled with paper), then every sheet's inked content in priority order, so blank paper never hides a neighbour's map. |
| `review --year S` | Serve `data/out/` on 127.0.0.1 and open the series review page. |
| `edit [--year S] [--sheet ID]` | Start the control-point editor on 127.0.0.1:8765 (`--demo` for a synthetic sheet). |
| `bundle` | Write the processed files into `data/share/` as zips plus `PROCESSED.json`, for copying by hand into the Drive folder `old-planning-maps/processed/`. |
| `pull` | Download the processed zips from the public Drive `processed/` folder and extract them into `data/`. The `processed/` folder must be shared as "Anyone with the link" (Viewer); `pull` uses Drive's public download, so a private folder gives an error page, which `pull` rejects. |
| `crossref --year S` / `--all` | Re-place flagged sheets by matching each to the same printed sheet number in another edition that passed (SIFT + RANSAC on the drawings, composed with that sheet's transform), then rerun refine and score. A result is kept only if it scores better, and is then pinned as the sheet's seed (`inputs/<series>/seeds/`; commit it); otherwise the sheet's files and seed are restored. Republishes improved series (`--no-publish` to skip). No API calls. |
| `eval --year 1948\|1960` | Per-sheet score next to the distance from our transform to the owner's manual `.points` (the answer key). |
| `estimate --year S` / `--all` | Estimate API calls, tokens and USD for the uncached pages (no API calls). It is an estimate: image tokens use OpenAI's patch formula, prompt text is a flat `config.EST_TEXT_TOKENS_PER_CALL` (~350) per call, and output is a flat `EST_OUT_TOKENS` per call. |
| `calibrate --year S --sheet ID` | Find the smallest label-reading scale that keeps 95% of full-resolution recall. |

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

and `inputs/label_scale.json`, the calibrated label-reading scale per series.

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
  stage, the pipeline's own checks still decide which sheets are recomputed, but every series
  in it is re-stitched and re-tiled, so a code change costs about an hour of tiling. The dry
  run shows what would run first.
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

**Sharing (not set up yet).** A DVC remote would let `dvc push` / `dvc pull` replace the Drive
bundle. Cloudflare R2 (S3-compatible, no download fees) suits it:
`dvc remote add -d storage s3://<bucket>/sfzoning` with
`dvc remote modify storage endpointurl https://<account>.r2.cloudflarestorage.com`, and
credentials in `.dvc/config.local` (never committed). Until then, `bundle` and `pull` stay
the way to share data.

## Recreate without re-running

```sh
uv run sfzmaps fetch && uv run sfzmaps pull && uv run sfzmaps build
```

`pull` brings back the LLM cache, the reference data and every sheet's processed
files (transforms, scores); the hand-made inputs are already in git. It costs **zero API spend only if your renders
are byte-identical** to the ones that filled the cache. `page.png` is not shared; `run`
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

## Where outputs land

Everything is under `pipeline/data/` (git-ignored; set `SFZMAPS_DATA_DIR` to move it):

- `raw/`: the Drive mirror.
- `ref/`: DataSF parquet files.
- `cache/llm/`: cached API answers.
- `work/<series>/<sheet>/`: per-sheet stages, i.e. `page.png`, `page.json`,
  `labels.json`, `rough.json`, `refine.json`, `changes.geojson`, `transform.json`,
  `score.json`, `warped.tif`, `warped_paper.tif` (after `paper`). Machine output only:
  everything a person decides is in `inputs/`.
- `out/<series>/`: `<series>.pmtiles`, `<series>-paper.pmtiles`, `manifest.json` (per-sheet status,
  `needs_review` reasons, scores, limits, bounds), `review.html`.
- `out/ref/blocks.pmtiles`: today's blocks for the review pages.
- `out/run-<timestamp>.log`: tracebacks for any sheet that errored.
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

## Credit: 1948

The 1948 atlas photographs come from the **David Rumsey Map Collection**, not the
owner's scans. Anything published from the 1948 series must carry the credit
"David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries" (`config.RUMSEY_CREDIT`;
likely CC BY-NC-SA). The pipeline writes it into the 1948 PMTiles attribution,
manifest and review page. Keep it wherever the overlay is shown.
