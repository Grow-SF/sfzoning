# Historic Zoning Map Georeferencing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI (`sfzmaps`) that automatically aligns every archived SF zoning map sheet to real-world coordinates, scores each sheet, stitches and tiles each year, and gives a browser editor for sheets scoring under 95%.

**Architecture:** Each sheet flows through a fixed set of stages, and each stage writes a JSON or GeoTIFF artifact under `pipeline/data/work/<series>/<sheet_id>/`. Stages are idempotent and skip work whose inputs haven't changed. The stages are: render page → classify page (Claude) → read labels (Claude) → extract block shapes from the scan → rough fit from label anchors → iterative refine against DataSF block corners → smooth warp → score → GeoTIFF. Per year: stitch → PMTiles → review page. Human edits live in `review.json` and take precedence over automatic results.

**Tech Stack:** Python 3.12 (uv-managed), numpy, scipy, opencv-python-headless, scikit-image, shapely 2, geopandas, pyproj, rasterio, pydantic 2, openai SDK, typer, FastAPI + uvicorn (editor), pytest. GDAL CLI 3.13 (`gdalwarp`, `gdalbuildvrt`, `gdal_translate`, `gdaladdo`), poppler `pdftoppm`, `pmtiles` CLI (`brew install pmtiles`). Front end: plain HTML + MapLibre GL + pmtiles JS from CDN (no build step).

**Spec:** `docs/superpowers/specs/2026-10-29-map-georeferencing-design.md`

## Global Constraints

- No QGIS in any workflow; the owner never places control points except in the editor for sheets under threshold.
- Working CRS: **EPSG:2227** (CA State Plane III, US ft), matching the owner's `.points` files. Tiles: EPSG:3857.
- Pass threshold: **95%** of a sheet's matched block corners (excluding real change and no-modern-equivalent areas) within `TOLERANCE_FT`. `TOLERANCE_FT` starts at **6.0** and is tuned against the 1948/1960 answer key (Task 13). Both values are constants in `sfzmaps/config.py`.
- **Vision provider: OpenAI (owner's decision, superseding Claude; the owner has OpenAI credits).** Text reading and any other inference use the official `openai` Python SDK only, with structured outputs parsed into pydantic models. The model is the constant `config.VISION_MODEL`, set to OpenAI's current flagship vision-capable model, verified against OpenAI's docs or `client.models.list()` at implementation time (Task 4b) and recorded in the report. Every response is cached on disk keyed by sha256(provider + model + prompt + schema JSON + image bytes). A cache hit makes no network call. Credentials: `OPENAI_API_KEY`.
- Images sent to the model are sized to the provider's actual image handling, so the server never silently downscales them below legibility. For OpenAI "high" detail, an image is fit within 2048×2048 and then scaled so its short side is ≤ 768 px. So label tiles must have a short side ≤ 768 (`VISION_TILE`/`VISION_MAX_EDGE` in config; Task 4b sets the values). Prompts state the image's pixel size and ask for integer pixel coordinates; returned coordinates are clamped to the tile.
- **Minimize image tokens (owner's instruction).** Image tokens scale with pixel area (≈ width × height / 750), so every image is downscaled to the smallest size that still reads correctly:
  - page classification uses a ≤ 1024 px long-edge thumbnail;
  - label reading runs at a per-series `label_scale` calibrated in Task 5 (the smallest scale whose label recall is ≥ 95% of full resolution on a calibration sheet);
  - tiles that are blank (ink < 0.5% of pixels), fully outside the neatline, or fully inside a mask box are never sent;
  - `estimate` computes tokens from actual tile pixel areas, not a per-image constant.
- The warp must be smooth (regularized). It must never interpolate every control point exactly or snap individual historic edges onto modern ones.
- Human-reviewed alignments (`review.json` present with `status` in {`edited`,`accepted`}) are never overwritten by an automatic rerun unless `--force-auto` is passed.
- Sheets under threshold are still stitched and tiled, but `manifest.json` marks them `needs_review`.
- Anything showing 1948 imagery (review page, editor, manifest) carries the credit string `"© Cartography Associates, David Rumsey Map Collection"`.
- Raw scans, reference data, cache, work and output dirs are git-ignored (`pipeline/data/`). Nothing large goes in git. Shareable generated files go to the Drive folder `old-planning-maps/processed/` via `sfzmaps push`/`pull` (Task 19).
- Before spending API money on a full run, `sfzmaps estimate` must print the projected call count, image tokens and dollar cost, using the provider's real image-token formula and prices from config (Task 4b sets the OpenAI values).

## Review Focus

1. **A sheet where the vision model reads few or no usable labels** (e.g. Golden Gate Park or the Presidio, or a page mislabeled as a map). Expect `needs_review` with reason `insufficient_anchors`, not a crash and not a wild fit. Test in Task 8.
2. **A sheet that is rotated, mirrored-looking, or has north not up** (the 1968 pages are landscape with the map turned 90°). The rough fit must solve for any rotation. Test in Task 8.
3. **Combined or split source images** (1948 `2-and-3.jpg` alongside `2.jpg`/`3.jpg`; a 1921 page that is only text). The page registry must not align the same map area twice, and must skip non-map pages with a logged reason. Test in Task 3.
4. **Rerun after a human edit.** The automatic run must leave `review.json`'s points alone and must use them. Test in Task 12.
5. **Areas that truly changed** (Western Addition superblocks, the widened Embarcadero, Golden Gateway). These matches must be classified as real change, not counted as error, and must not drag the warp. Test in Task 10 (synthetic shifted block) and Task 13 (real 1948 eval).

---

## File Structure

```
pipeline/
  pyproject.toml              # uv project, console script `sfzmaps = sfzmaps.cli:app`
  .python-version             # 3.12
  src/sfzmaps/
    config.py                 # paths, CRS, thresholds, model id, credit strings
    cli.py                    # typer app: fetch, refdata, estimate, run, eval, review, edit
    sources/drive.py          # list/download public Drive folders
    sources/datasf.py         # parcels/streets → blocks, block corners, intersections
    pages.py                  # Page/Sheet registry, PDF/JPG → PNG rendering
    llm.py                    # cached vision client (ask_json); OpenAI implementation
    reading.py                # page classification + label reading prompts/schemas
    linework.py               # scan → block polygons/corners in pixel space
    transform.py              # SheetTransform: affine + smoothed TPS, (de)serialize
    anchors.py                # labels × reference → rough transform
    refine.py                 # iterative matching, change classification
    score.py                  # SheetScore
    warp.py                   # GeoTIFF with transparent paper
    stitch.py                 # per-year VRT + masks
    tiles.py                  # VRT → PMTiles, size report
    manifest.py               # per-year manifest.json
    reviewpage.py             # static review HTML
    pointsfile.py             # read QGIS .points (answer key)
    editor/server.py          # FastAPI app
    editor/static/index.html, editor.js, editor.css
  tests/
    conftest.py               # fixtures: tiny reference dataset, fake LLM
    synthetic.py              # render a fake map sheet from real block polygons with known distortion
    fixtures/                 # small real crops (1948, 1968), tiny GeoParquet subset
    test_*.py
  data/                       # git-ignored: raw/ ref/ cache/ work/ out/
```

Stage artifacts per sheet in `data/work/<series>/<sheet_id>/`: `page.png`, `page.json`, `labels.json`, `blocks_px.geojson`, `transform.json`, `changes.geojson`, `score.json`, `warped.tif`, optional `review.json`.

`<series>` values: `1921`, `1929`, `1935`, `1948`, `1960`, `1964`, `1968`, `1972`, `1975`, `1980`, and for "1980 Plus": `1980-1990-zoning`, `1980-1990-height`, `1980-1990-su`, `1990-1996-zoning`, … `2000-digital-zoning`, `2000-digital-height`, `2000-digital-su` (derived from file names; the "Introduction" PDF is a non-map document).

---

### Task 1: Project scaffold and Drive download

**Files:**
- Create: `pipeline/pyproject.toml`, `pipeline/.python-version`, `pipeline/src/sfzmaps/__init__.py`, `config.py`, `cli.py`, `sources/drive.py`
- Create: `pipeline/tests/test_drive.py`, `pipeline/tests/fixtures/drive_folder.html` (save the real listing HTML of the 1948 folder)
- Modify: `.gitignore` (add `pipeline/data/`, `pipeline/.venv/`)

**Interfaces:**
- Produces: `config.DATA_DIR, RAW_DIR, REF_DIR, CACHE_DIR, WORK_DIR, OUT_DIR: Path`; `config.WORK_CRS = "EPSG:2227"`; `config.TOLERANCE_FT = 6.0`; `config.PASS_FRACTION = 0.95`; `config.MODEL = "claude-opus-5-5"`; `config.ROOT_FOLDER_ID = "10p74RwqmzeQFBrBqjxXdmEOUREx0BLgO"`; `config.RUMSEY_CREDIT`.
- Produces: `drive.DriveEntry(id: str, title: str, is_folder: bool)`; `drive.list_folder(folder_id: str) -> list[DriveEntry]`; `drive.parse_listing(html: str) -> list[DriveEntry]`; `drive.sync(folder_id: str, dest: Path) -> list[Path]` (recursive, skips files already present with non-zero size).
- Produces CLI: `sfzmaps fetch` → mirrors the Drive tree into `data/raw/`.

- [ ] **Step 1: Scaffold.** `uv init --package --python 3.12 pipeline`. Add the dependencies listed in Tech Stack, plus dev dependency pytest. Wire the console script.
- [ ] **Step 2: Write failing tests** `test_parse_listing_finds_entries` (the fixture HTML yields entries including title `"aligned"` with `is_folder=True` and `"1948.tif"` with `is_folder=False`) and `test_parse_listing_unescapes_titles` (`"1980 - 1990  Height &amp; Bulk Maps.PDF"` → `"1980 - 1990  Height & Bulk Maps.PDF"`).
- [ ] **Step 3: Run** `cd pipeline && uv run pytest tests/test_drive.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement `drive.py`.** List via `https://drive.google.com/embeddedfolderview?id=<id>`. Parse entries with the regex `id="entry-([^"]+)".*?flip-entry-title">([^<]+)<` (DOTALL). A folder's entry link contains `/drive/folders/`. Download via `https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t` using `httpx`, streaming to a `.part` file and then renaming.
- [ ] **Step 5: Run tests.** Expected: PASS. Then run `uv run sfzmaps fetch` and confirm `data/raw/1948/originals/1.jpg` exists (about 7.6 MB).
- [ ] **Step 6: Commit** `feat(pipeline): scaffold sfzmaps and Drive mirror`.

### Task 2: Reference data from DataSF

**Files:**
- Create: `src/sfzmaps/sources/datasf.py`, `tests/test_datasf.py`, `tests/fixtures/parcels_sample.geojson` (about 30 real parcels covering 3 blocks near Lombard/Hyde, fetched once with `$where=block_num in('0045','0046','0047')`)

**Interfaces:**
- Produces: `datasf.fetch_reference(dest: Path = REF_DIR) -> None`. It downloads `https://data.sf.gov/resource/acdm-wktn.geojson` (parcels; active and retired) and `https://data.sf.gov/resource/3psu-pn9h.geojson` (streets), paging with `$limit=50000&$offset=` until a page comes back empty, then writes `parcels.parquet` and `streets.parquet` in EPSG:2227.
- Produces: `datasf.build_blocks(parcels: GeoDataFrame) -> GeoDataFrame` with columns `block_num: str` (4-char, zero-padded, any letter suffix dropped, e.g. `"0045"`), `geometry` (Polygon, EPSG:2227). Built from active parcels (`active == true`) dissolved by block, with holes and slivers under 50 sq ft removed.
- Produces: `datasf.block_corners(blocks) -> GeoDataFrame` with `block_num`, `corner_idx: int`, `geometry: Point`. Corners are vertices of the simplified exterior (`simplify(2.0)`) whose turning angle is > 30°.
- Produces: `datasf.intersections(streets) -> GeoDataFrame` with `node_cnn`, `streets: list[str]` (normalized names, e.g. `"LOMBARD"`), `geometry: Point`. Built from active segments' `f_node_cnn`/`t_node_cnn` endpoints.
- Produces: `datasf.normalize_street(name: str) -> str` (uppercase; drop ST/AVE/BLVD/etc.; `"FIRST"`→`"1ST"`; strip punctuation).
- Produces: `datasf.load_reference() -> Reference` (dataclass with `blocks`, `corners`, `intersections`, `streets`, all GeoDataFrames).
- Produces CLI: `sfzmaps refdata`.

- [ ] **Step 1: Write failing tests:**
  - `test_build_blocks_dissolves_by_block` (the fixture gives exactly 3 rows, `set(block_num) == {"0045","0046","0047"}`)
  - `test_block_corners_rectangular_block_has_4` (block `"0045"` yields 4 corners)
  - `test_normalize_street` (`"Lombard St"`→`"LOMBARD"`, `"First Street"`→`"1ST"`, `"Van Ness Ave"`→`"VAN NESS"`)
- [ ] **Step 2: Run** `uv run pytest tests/test_datasf.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with geopandas: `read_file(url)` per page, `to_crs(2227)`, `dissolve(by="block_num")`.
- [ ] **Step 4: Run tests** (PASS). Then run `uv run sfzmaps refdata` and check: blocks count between 2,500 and 4,000, and `Reference.blocks.total_bounds` within x 5.97e6–6.03e6, y 2.08e6–2.14e6.
- [ ] **Step 5: Commit** `feat(pipeline): DataSF blocks, corners, intersections`.

### Task 3: Page registry and rendering

**Files:**
- Create: `src/sfzmaps/pages.py`, `tests/test_pages.py`

**Interfaces:**
- Produces: `pages.Page(series: str, sheet_id: str, source: Path, page_index: int | None)`. `sheet_id` is stable: the PDF stem plus `-p{NN}` for multi-page PDFs; for 1948 it's the JPG stem.
- Produces: `pages.discover(raw_dir: Path = RAW_DIR) -> list[Page]`. Rules:
  - `1948`: use `originals/*.jpg` except `index`, `setback`, `planning-code`, and any `X-and-Y.jpg` whose single sheets `X.jpg`/`Y.jpg` are also present (only `1.jpg`, `14.jpg` and the combined ones exist as JPGs in `originals/`). The individual sheets 2–13 come from `aligned/N.jpg`.
  - `1960`: `originals/1960.pdf` pages.
  - "1980 Plus": series from the file name per the File Structure list.
  - Ignore `aligned/`, `cropped/` and `georeferenced/` everywhere except as the 1948 sheet source just described.
- Produces: `pages.render(page: Page, out: Path) -> Path`. Writes a 300 dpi grayscale PNG for PDFs (`pdftoppm -r 300 -gray -f N -l N`) and an RGB PNG for JPGs (1948 stays color). Skips the work if `out` is newer than the source.

- [ ] **Step 1: Write failing tests** against a temp tree mimicking the real layout:
  - `test_discover_1948_prefers_single_sheets` (with `2-and-3.jpg`, `aligned/2.jpg`, `aligned/3.jpg` present, sheets `2` and `3` appear and `2-and-3` does not)
  - `test_discover_skips_index_and_setback`
  - `test_discover_1980plus_series_names` (`"1990 to 1996 Height Maps.PDF"` → series `"1990-1996-height"`)
  - `test_sheet_ids_unique`
  - `test_neatline_mask_synthetic` (a white page with a 20 px black rectangular border inset 100 px → the mask is True strictly inside and False in the margin)
- Also produces: `pages.neatline_mask(img: np.ndarray) -> np.ndarray` (bool: inside the map's outer border, found as the largest near-rectangular closed contour covering > 40% of the page; returns all-True if none is found). Used by label tile skipping (Task 5), block extraction (Task 6) and warping (Task 11).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** Page counts come from `pdfinfo`.
- [ ] **Step 4: Run tests** (PASS). Then `uv run python -c "from sfzmaps.pages import discover; print(len(discover()))"` and record the count in the commit message.
- [ ] **Step 5: Commit** `feat(pipeline): page discovery and rendering`.

### Task 4: Cached Claude client

**Files:**
- Create: `src/sfzmaps/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Produces: `llm.ask_json(image: np.ndarray, prompt: str, schema: type[T], *, client: LLMClient | None = None) -> T`, where `T` is a pydantic `BaseModel`. It encodes the image as PNG and asserts the long edge is ≤ 2576. The cache file is `CACHE_DIR/llm/<sha256>.json`.
- Produces: `llm.LLMClient` protocol with `def call(self, png: bytes, prompt: str, schema: type[T]) -> T`; `llm.AnthropicClient` implements it with `anthropic.Anthropic().beta.messages.parse(model=MODEL, max_tokens=16000, output_format=schema, betas=["server-side-fallback-2026-07-01"], fallbacks="default", output_config={"effort": "medium"}, messages=[user: image block (base64 png) then text])`.
  - Check `stop_reason`. On `"refusal"`, raise `LLMRefusal`. Otherwise return `parsed_output`.
  - If `messages.parse` rejects the beta params in the installed SDK, use `beta.messages.create` with `output_config.format` JSON schema and validate with pydantic. Check the SDK, don't guess.
- Produces: `llm.CallCounter` (a module-level count of live calls and of cache hits) for `estimate` and logging.

- [ ] **Step 1: Write failing tests** using a `FakeClient` that counts calls:
  - `test_cache_hit_skips_client` (two identical calls → `FakeClient.calls == 1`, and the results are equal)
  - `test_cache_key_changes_with_prompt`
  - `test_rejects_oversize_image` (a 3000×100 array raises `ValueError`)
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests** (PASS). Then run a manual smoke check: `uv run python -m sfzmaps.llm --smoke data/raw/1948/originals/1.jpg` downsizes the image, asks "what is the sheet number", and prints `7`. Run it twice; the second run must print `cache hit`.
- [ ] **Step 5: Commit** `feat(pipeline): cached Claude vision client`.

### Task 4b: Switch the vision client to OpenAI (owner's decision)

**Files:**
- Modify: `src/sfzmaps/llm.py` (replace `AnthropicClient` with `OpenAIClient`; drop the `anthropic` dependency), `src/sfzmaps/config.py` (`VISION_PROVIDER = "openai"`, `VISION_MODEL`, `VISION_TILE`, `VISION_OVERLAP`, `VISION_MAX_EDGE`, `VISION_THUMB`, `estimate_image_tokens`, `PRICE_IN_PER_M`, `PRICE_OUT_PER_M`), `pyproject.toml`
- Test: `tests/test_llm.py`, `tests/test_config.py`

**Interfaces:**
- `llm.ask_json` and the `LLMClient` protocol are unchanged, so reading.py doesn't change. The cache key adds the provider name.
- `llm.OpenAIClient.call(png, prompt, schema)` uses the installed `openai` SDK's structured-output parse helper (verify the current one in the installed SDK, e.g. `client.responses.parse(model=VISION_MODEL, input=[{role:"user", content:[{type:"input_image", image_url:"data:image/png;base64,...", detail:"high"}, {type:"input_text", text:prompt}]}], text_format=schema)` → `output_parsed`). Don't guess: confirm the names in the installed SDK source.
  - A refusal raises `LLMRefusal`. Incomplete output (max tokens) raises a clear RuntimeError.
  - A missing `OPENAI_API_KEY` gives a one-line error.
  - Corrupt-cache recovery is unchanged.
- `config.VISION_MODEL`: OpenAI's current flagship vision model, verified from the OpenAI docs (WebFetch platform.openai.com docs on models and vision) and, if a key is present, `client.models.list()`. Record the source in the report.
- `config.VISION_TILE = 768`, `VISION_OVERLAP = 96`, `VISION_MAX_EDGE = 2048`, `VISION_THUMB = 1024`. These come from OpenAI high-detail handling (fit 2048², then short side → 768; 512-px tiles). Tiles at or below 768×768 are never downscaled by the server. The builder verifies against the current OpenAI vision docs and adjusts if the docs differ, recording the reasoning.
- `config.estimate_image_tokens(w, h) -> int` implements OpenAI's documented high-detail formula for `VISION_MODEL` (base tokens + per-512-tile tokens after the scaling steps; verify the per-model constants in the docs). `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` are the model's current prices from the docs.

- [ ] **Step 1: Write failing tests:**
  - `test_openai_client_builds_request` (monkeypatch the SDK client: the request carries a base64 PNG data URL with detail "high", `text_format` is the schema, and the model is `VISION_MODEL`)
  - `test_openai_refusal_raises`
  - `test_cache_key_includes_provider`
  - `test_estimate_tokens_768_tile` (equals the documented value for one 768×768 high-detail image)
- [ ] **Step 2: Run, implement, pass.**
- [ ] **Step 3: Live smoke** (`python -m sfzmaps.llm --smoke data/raw/1948/originals/1.jpg` prints the sheet number `7`, then `cache hit` on the second run). Deferred if there is no `OPENAI_API_KEY`.
- [ ] **Step 4: Commit** `feat(pipeline): OpenAI vision client`.

### Task 5: Page classification and label reading

**Files:**
- Create: `src/sfzmaps/reading.py`, `tests/test_reading.py`

**Interfaces:**
- Consumes: `llm.ask_json`, `pages.Page`.
- Produces pydantic schemas:
  - `PageInfo(kind: Literal["map","cover","legend","text","index","other"], sheet_number: str | None, neighbor_refs: list[NeighborRef], mask_boxes: list[Box])`, where `NeighborRef(text: str, sheet: str, side: Literal["n","s","e","w","unknown"])` and `Box(x0:int,y0:int,x1:int,y1:int)` are full-page pixel boxes of legends, title blocks and margins to exclude.
  - `Label(kind: Literal["block_number","street_name","place_name"], text: str, x: int, y: int, confidence: float)`, with pixel coordinates in full-page space.
- Produces: `reading.classify_page(png_path: Path) -> PageInfo`. Sends one whole-page thumbnail at ≤ 1024 px long edge and scales the boxes back to full resolution.
- Produces: `reading.read_labels(png_path: Path, *, scale: float, tile: int = 1568, overlap: int = 160, skip: np.ndarray | None = None) -> list[Label]`.
  - Downscales the page by `scale` (area-averaging, so 1-bit lettering stays legible), then tiles it.
  - Skips tiles with ink < 0.5%, or fully excluded by `skip` (the neatline/mask-box exclusion mask).
  - Maps coordinates back to full-res page space (÷ `scale`, + tile offset) and dedupes labels with the same `kind`+`text` within 40 full-res px.
- Produces: `reading.calibrate_scale(png_path: Path, scales=(1.0, 0.75, 0.6, 0.5, 0.4, 0.33)) -> float`. Reads labels at each scale on one representative map sheet and returns the smallest scale whose recall of block-number + street labels is ≥ 95% of the scale-1.0 result. Matching is by text, within 40 px. The result is stored in `config.LABEL_SCALE[series]`, which defaults to 0.5 until calibrated. CLI: `sfzmaps calibrate --year Y --sheet S`. Calibration costs a few dollars per series; run it on one sheet per drafting style (1921, 1948, 1960s, 1980+), not every series.
- Produces: `reading.save/load` helpers writing `page.json` / `labels.json`.
- Prompts are module constants. The label prompt asks for every printed block number (digits, possibly with a letter suffix) and every street name, each with the pixel center of the text, and says to ignore zoning district codes (R-1, C-2…) and legend text.

- [ ] **Step 1: Write failing tests** with a FakeClient returning canned schema objects:
  - `test_read_labels_offsets_tile_coords` (at `scale=0.5`, a label at (10,10) in the tile at col 1/row 0 maps to ((10 + 1568 − 160) / 0.5, 10 / 0.5))
  - `test_read_labels_dedupes_overlap`
  - `test_blank_tiles_not_sent` (a page that is white except one corner → FakeClient receives exactly the tiles touching that corner)
  - `test_masked_tiles_not_sent`
  - `test_classify_thumbnail_max_1024` (FakeClient sees an image with long edge ≤ 1024; boxes are scaled back to full-res)
  - `test_calibrate_picks_smallest_adequate_scale` (FakeClient returns full recall down to 0.5 and 80% at 0.4 → returns 0.5)
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests** (PASS). Then run `uv run sfzmaps calibrate --year 1948 --sheet 7` and record the chosen scale and the per-scale recall in the tuning notes. Then a live check: `uv run sfzmaps run --year 1948 --sheet 7 --until labels`. Expected: `page.json` has kind `map` and sheet_number `7`, and `labels.json` contains block numbers including `"26"`, `"27"`, `"45"` (visible top-left of the sheet).
- [ ] **Step 5: Add `sfzmaps estimate [--year | --all]`.** It computes, without calling the API, the exact tiles that would be sent (after scaling and blank/mask skipping) minus cache hits. It prints calls, image tokens (Σ w×h/750), and USD at $4/M input + $20/M output, assuming 800 output tokens per call (the builder adjusts this after the live check using observed `usage`). Commit `feat(pipeline): page classification, label reading, scale calibration, cost estimate`.

### Task 6: Block shapes from the scan

**Files:**
- Create: `src/sfzmaps/linework.py`, `tests/synthetic.py`, `tests/test_linework.py`
- Create fixtures: `tests/fixtures/crop_1948.png` (about 1500×1500 crop of `1948/7` around Lombard–Union with white, black and hatched blocks), `tests/fixtures/crop_1968.png` (about 1500×1500 crop of 1968 page 8 including a heavy district boundary line running down a street). Count blocks by eye and record the counts in the test.

**Interfaces:**
- Produces: `linework.extract_blocks(img: np.ndarray, mask: np.ndarray | None = None) -> list[Polygon]`, returning block outlines in pixel coordinates. It must handle white, solid-black, cross-hatched and stippled blocks, and street-width gaps from about 15 px to 120 px at 300 dpi. A heavy district line drawn inside a street must not merge the blocks on either side.
  - Approach (builder may change it if the tests pass): binarize (Otsu on 1948; threshold on 1-bit pages), close small gaps, fill holes inside closed outlines so every block becomes solid, then open with a structuring element smaller than the narrowest street to cut thin bridges, then take connected components filtered by area and rectangularity.
- Produces: `linework.block_corners_px(polys: list[Polygon]) -> np.ndarray` (N×2), using the same > 30° turning-angle rule as `datasf.block_corners`.
- Produces in `tests/synthetic.py`: `make_sheet(blocks: GeoDataFrame, *, px_per_ft: float, rotation_deg: float, distortion: Callable[[np.ndarray], np.ndarray], style: Literal["outline","filled"], shift_blocks: dict[str, tuple[float,float]] = {}) -> SyntheticSheet` with fields `img`, `true_px_to_world: Callable`, `labels: list[Label]`. It renders the given modern blocks as a fake scan with a known transform. `shift_blocks` moves named blocks by (dx, dy) ft to simulate real change.

- [ ] **Step 1: Write failing tests:**
  - `test_extract_blocks_synthetic_outline` (a 3×3 synthetic grid yields 9 polygons, each IoU > 0.9 with truth)
  - `test_extract_blocks_synthetic_filled` (same, `style="filled"`)
  - `test_extract_blocks_real_1948_crop` (count within ±1 of the hand count)
  - `test_extract_blocks_real_1968_crop` (count within ±1 of the hand count; no polygon's area > 1.5× the median, which catches merged blocks)
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** `linework.py` and `synthetic.py`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): extract block shapes from scans`.

### Task 7: SheetTransform

**Files:**
- Create: `src/sfzmaps/transform.py`, `tests/test_transform.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class GCP: px: float; py: float; wx: float; wy: float; source: Literal["anchor","auto","manual","answer_key"]; enabled: bool = True
  @dataclass
  class SheetTransform:
      gcps: list[GCP]
      smoothing: float            # RBF smoothing; 0 is forbidden (exact interpolation)
      def fit(self) -> "SheetTransform"      # affine by least squares + thin-plate RBF on residuals (scipy RBFInterpolator, kernel="thin_plate_spline", smoothing=self.smoothing)
      def to_world(self, px: np.ndarray) -> np.ndarray     # (N,2) pixel → EPSG:2227
      def to_pixel(self, w: np.ndarray) -> np.ndarray      # inverse via fitting the reverse direction the same way
      def residuals_ft(self) -> np.ndarray                  # per enabled GCP
      def dense_grid(self, shape: tuple[int,int], step: int = 150) -> list[GCP]  # for gdalwarp
      def to_json(self) -> dict; @classmethod from_json(cls, d) -> "SheetTransform"
  ```
- `smoothing` default: `config.TPS_SMOOTHING = 50.0`, tuned in Task 13. Values are in ft² units of the residual field. The builder documents the units in the code.

- [ ] **Step 1: Write failing tests:**
  - `test_affine_recovered_exactly` (GCPs from a pure affine give residuals < 0.01 ft)
  - `test_smooth_warp_does_not_chase_outlier` (on a grid of 100 GCPs from a gentle distortion, moving one GCP by 40 ft moves `to_world` at that pixel by < 10 ft)
  - `test_roundtrip` (`to_pixel(to_world(p))` within 0.5 px)
  - `test_json_roundtrip`
  - `test_zero_smoothing_rejected`
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): smooth sheet transform`.

### Task 8: Rough fit from anchors

**Files:**
- Create: `src/sfzmaps/anchors.py`, `tests/test_anchors.py`

**Interfaces:**
- Consumes: `reading.Label`, `datasf.Reference`, `transform.SheetTransform`, `synthetic.make_sheet`.
- Produces: `anchors.match_labels(labels: list[Label], ref: Reference) -> list[Candidate]`, where `Candidate(label: Label, world: tuple[float,float], kind: str, weight: float)`:
  - A block number maps to its modern block's centroid. The label text is normalized to 4-digit zero-padded, and labels whose text matches no modern block are dropped.
  - Two street-name labels within 400 px of each other whose normalized names meet at a modern intersection produce a candidate at the midpoint of the two labels, matched to that intersection's location, with weight 0.5.
- Produces: `anchors.rough_fit(cands: list[Candidate], *, min_inliers: int = 6, ransac_tol_ft: float = 150) -> RoughFit | None`. `RoughFit(transform: SheetTransform, inliers: list[Candidate], scale_ft_per_px: float, rotation_deg: float)`. It runs RANSAC over a similarity transform (any rotation, uniform scale), then refits an affine on the inliers. It returns `None` when there are fewer than `min_inliers` inliers or the scale is outside 0.3–10 ft/px.

- [ ] **Step 1: Write failing tests** using `make_sheet` with labels at block centroids:
  - `test_rough_fit_recovers_rotation_90` (`rotation_deg=90`; the fitted rotation is within 1° and the median block-centroid error < 30 ft)
  - `test_rough_fit_survives_misreads` (30% of labels replaced with random existing block numbers; still recovers)
  - `test_rough_fit_none_when_few_labels` (3 labels → `None`)
  - `test_street_pair_candidates`
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): rough placement from label anchors`.

### Task 9: Iterative refine

**Files:**
- Create: `src/sfzmaps/refine.py`, `tests/test_refine.py`

**Interfaces:**
- Consumes: `RoughFit`, `linework.block_corners_px`, `Reference.corners`, `SheetTransform`.
- Produces: `refine.refine(corners_px: np.ndarray, ref_corners: GeoDataFrame, start: SheetTransform, *, radii_ft: tuple[float, ...] = (120, 60, 30, 15), max_iter: int = 30) -> RefineResult`, with `RefineResult(transform: SheetTransform, matches: list[Match], iterations: int)` and `Match(px: float, py: float, wx: float, wy: float, dist_ft: float, label: Literal["inlier","real_change","no_equivalent","outlier"])`.
  - Loop: map the scan corners to the world through the current transform; take mutual nearest neighbours to modern corners within the current radius; drop matches whose residual is > 3× the MAD; refit with `SheetTransform` (source `"auto"`); shrink the radius once the transform stops moving (median shift < 1 ft).
  - Scan corners with no modern corner within the final radius → `no_equivalent`.
- Classification of real change happens in Task 10; here every surviving match is `inlier` and the rejected ones are `outlier`.

- [ ] **Step 1: Write failing tests** on `make_sheet` with `distortion` set to a smooth paper-stretch field of up to 25 ft, started from a rough fit perturbed by 40 ft and 1°:
  - `test_refine_converges_on_synthetic` (95% of inlier block corners within 3 ft of truth)
  - `test_refine_uses_many_points` (`len(inliers) > 200` on a 20×20-block synthetic sheet)
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): iterative refinement against modern block corners`.

### Task 10: Change classification and scoring

**Files:**
- Create: `src/sfzmaps/score.py`; extend `refine.py` with `classify_changes`
- Test: `tests/test_score.py`, add to `tests/test_refine.py`

**Interfaces:**
- Produces: `refine.classify_changes(result: RefineResult, *, tol_ft: float = TOLERANCE_FT, min_cluster: int = 3, coherence_ft: float = 4.0) -> RefineResult`. After a smooth fit, a match with residual > `tol_ft` becomes `real_change` when at least `min_cluster` such matches lie within 600 ft of each other and their residual vectors agree within `coherence_ft` (same direction and size). The step then refits excluding `real_change` matches, repeating until labels stop changing (at most 5 rounds). Isolated large residuals stay `outlier`.
- Produces: `refine.changes_geojson(result) -> dict`. Holds `real_change` matches with offset vectors, plus convex hulls of `no_equivalent` clusters, all in EPSG:2227.
- Produces: `score.SheetScore(n_scored: int, frac_within: float, p50_ft: float, p95_ft: float, max_ft: float, n_real_change: int, n_no_equivalent: int, passed: bool, reason: str | None)`, and `score.score(result: RefineResult, tol_ft=TOLERANCE_FT, pass_fraction=PASS_FRACTION, min_scored: int = 40) -> SheetScore`.
  - `n_scored` counts inlier + outlier matches (real change and no-equivalent are excluded).
  - `passed = n_scored >= min_scored and frac_within >= pass_fraction`.
  - `reason` is one of `"below_threshold"`, `"too_few_matches"` or `"insufficient_anchors"`.

- [ ] **Step 1: Write failing tests:**
  - `test_shifted_block_is_real_change` (`make_sheet(shift_blocks={"0045": (30, 0)})`: block 0045's 4 corners are labeled `real_change`, `score.passed` is True, and the transform near block 0045 moved < 5 ft)
  - `test_isolated_outlier_not_real_change`
  - `test_score_threshold_boundary` (frac 0.95 passes, 0.949 fails)
  - `test_score_too_few_matches`
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): real-change classification and sheet score`.

### Task 11: Warp to GeoTIFF with transparent paper

**Files:**
- Create: `src/sfzmaps/warp.py`, `tests/test_warp.py`

**Interfaces:**
- Produces: `warp.warp_sheet(png: Path, t: SheetTransform, mask_boxes: list[Box], out: Path) -> Path`.
  1. Build an RGBA image where alpha = 0 for paper and for `mask_boxes`. For 1-bit pages, paper is white. For 1948 color, paper is pixels whose luminance is > Otsu threshold and whose saturation is < 0.15; the gutter and binding outside the neatline also get alpha 0.
  2. Write a temporary GeoTIFF carrying `t.dense_grid()` as GCPs (EPSG:2227).
  3. Run `gdalwarp -tps -r bilinear -t_srs EPSG:2227 -co COMPRESS=DEFLATE -co TILED=YES`. A dense grid from the smooth transform makes GDAL's TPS reproduce our transform, not re-fit to raw points.
- Consumes: `pages.neatline_mask` (Task 3) to zero alpha outside the map border.

- [ ] **Step 1: Write failing tests:**
  - `test_warp_places_pixel_correctly` (warp a synthetic sheet; the warped raster's world coordinate of a known block corner is within 1 px-equivalent of truth, checked with rasterio)
  - `test_paper_is_transparent` (a white region has alpha 0)
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): warp sheets to GeoTIFF`.

### Task 12: Sheet orchestration, reruns and human review state

**Files:**
- Modify: `src/sfzmaps/cli.py`
- Create: `src/sfzmaps/runner.py`, `tests/test_runner.py`

**Interfaces:**
- Produces: `runner.run_sheet(page: Page, ref: Reference, *, until: str | None = None, force_auto: bool = False) -> SheetScore`. Stages, in order: `render` → `classify` → `labels` → `blocks` → `rough` → `refine` → `changes` → `score` → `warp`. Each stage skips work when its artifact is newer than its inputs. A non-map page stops after `classify`, and the run logs `skipped: <kind>`.
- `review.json` schema: `{"status": "edited"|"accepted", "reviewer": str, "at": ISO8601, "note": str, "gcps": [GCP...], "change_areas": GeoJSON}`. If it is present and `force_auto` is False, the run uses its GCPs as the transform (skipping `rough`/`refine`), still recomputes `score` for display, and warps from it. If the automatic run fails (`rough_fit` returns `None`) on a 1948/1960 sheet, it falls back to the owner's `.points` (Task 13 loader), with source `"answer_key"` and `reason` kept.
- Produces: `runner.run_series(series: str, ...) -> dict[str, SheetScore]`.
- CLI: `sfzmaps run [--year Y | --all] [--sheet S] [--until STAGE] [--force-auto]`. Prints a table of sheet, passed, frac_within, p95_ft and reason.

- [ ] **Step 1: Write failing tests** using a fake pipeline (monkeypatched stages):
  - `test_rerun_keeps_human_points` (with a `review.json` present, `run_sheet` doesn't call `refine` and `transform.json` equals the review GCPs)
  - `test_force_auto_ignores_review`
  - `test_non_map_page_skipped`
  - `test_stage_skip_when_fresh` (second run calls no stage functions)
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): sheet runner with review-aware reruns`.

### Task 13: Evaluate on 1948 and 1960 against the answer key; tune

**Files:**
- Create: `src/sfzmaps/pointsfile.py`, `tests/test_pointsfile.py`; add the `eval` command in `cli.py`
- Create: `docs/superpowers/notes/2026-10-29-tuning.md` (results table)

**Interfaces:**
- Produces: `pointsfile.read_points(path: Path) -> list[GCP]`. Reads the QGIS CSV `mapX,mapY,pixelX,pixelY,enable`, skipping a leading `#CRS:` line if one is present. QGIS stores `pixelY` negative, so `py = -pixelY`. Source is `"answer_key"`.
- CLI: `sfzmaps eval --year 1948|1960`. For each sheet with a `.points` file, it reports the distance between our `to_world(px)` and the owner's `(mapX,mapY)` at each answer-key point (median, max), next to our own `SheetScore`. The owner's points are imperfect, so a large disagreement means "look at it", not automatically "we're wrong".

- [ ] **Step 1: Write a failing test** `test_read_points_flips_y`, using the first two rows of the real `1.jpg.points` copied into a fixture.
- [ ] **Step 2: Run, implement, pass.**
- [ ] **Step 3: Run the live pipeline:** `uv run sfzmaps estimate --year 1948`, then `uv run sfzmaps run --year 1948`, then `uv run sfzmaps eval --year 1948`. Expected: every map sheet either passes or has a reason. The median distance to the answer key is < 30 ft (the manual points are coarse).
- [ ] **Step 4: Tune** `TOLERANCE_FT` (keep it ≤ 8), `TPS_SMOOTHING` and the `classify_changes` parameters. Pick values so that known-unchanged downtown/Marina grids pass, and so the Embarcadero, Van Ness and Western Addition show up as real change rather than as error. Record each tried combination and its per-sheet results in the tuning notes. Change values only in `config.py`.
- [ ] **Step 5: Repeat for 1960.**
- [ ] **Step 6: Commit** `feat(pipeline): answer-key eval; tuned thresholds` (include the tuning notes).

### Task 14: Stitch, tiles, manifest

**Files:**
- Create: `src/sfzmaps/stitch.py`, `tiles.py`, `manifest.py`, `tests/test_manifest.py`, `tests/test_stitch.py`

**Interfaces:**
- Produces: `stitch.build_vrt(series: str) -> Path` → `OUT_DIR/<series>/<series>.vrt`, built from every map sheet's `warped.tif`. Ordering for overlaps: passing sheets on top of `needs_review` sheets, then higher `frac_within` on top. Uses `gdalbuildvrt -srcnodata` with alpha honored.
- Produces: `tiles.make_pmtiles(vrt: Path, out: Path, *, min_zoom: int = 11, max_zoom: int = 19) -> TileReport`. Steps: `gdalwarp -t_srs EPSG:3857` to a COG, then `gdal_translate -of MBTILES -co TILE_FORMAT=WEBP -co QUALITY=85`, then `gdaladdo` down to `min_zoom`, then `pmtiles convert`. `TileReport(bytes_total: int, bytes_by_zoom: dict[int,int])` is read back from the MBTiles before conversion.
- Produces: `manifest.write(series: str, scores: dict[str, SheetScore]) -> Path` → `OUT_DIR/<series>/manifest.json`, shaped `{"series", "generated_at", "credit": str|null, "pmtiles": "<series>.pmtiles", "tile_report": {...}, "sheets": [{"sheet_id","status": "passed"|"needs_review"|"edited"|"accepted","frac_within","p95_ft","reason","bounds_wgs84"}]}`. `credit` is `RUMSEY_CREDIT` for `1948`, otherwise null.
- CLI: `sfzmaps run` runs stitch, tiles and manifest automatically after its sheets finish.

- [ ] **Step 1: Write failing tests:**
  - `test_manifest_marks_needs_review`
  - `test_manifest_1948_credit`
  - `test_vrt_order_passed_on_top` (two overlapping synthetic GeoTIFFs; the VRT lists the passing one last)
- [ ] **Step 2: Run, implement, pass.**
- [ ] **Step 3: Live check:** `uv run sfzmaps run --year 1948`. Expected: `out/1948/1948.pmtiles` exists and `pmtiles show` reports bounds over SF; record the tile report in the tuning notes.
- [ ] **Step 4: Commit** `feat(pipeline): stitch, PMTiles, manifest`.

### Task 15: Review page

**Files:**
- Create: `src/sfzmaps/reviewpage.py`, `src/sfzmaps/templates/review.html`, `tests/test_reviewpage.py`

**Interfaces:**
- Consumes: the manifest, the PMTiles file, each sheet's `changes.geojson` (reprojected to WGS84 on write), and DataSF blocks (exported once to `OUT_DIR/ref/blocks.pmtiles` with `ogr2ogr -f PMTiles`, since GDAL 3.13 writes vector PMTiles directly).
- Produces: `reviewpage.write(series: str) -> Path` → `OUT_DIR/<series>/review.html`. It's a static page using MapLibre GL + pmtiles protocol from a CDN with pinned versions, opened via `sfzmaps review --year Y` (serves `OUT_DIR` on localhost with HTTP range support and opens the browser). Contents:
  - the scan layer with an opacity slider;
  - modern block outlines;
  - real-change areas in one color and no-equivalent areas in another;
  - a sidebar table of sheets sorted failing-first, where clicking a row zooms to that sheet's bounds and shows its score;
  - an "Open in editor" link per `needs_review` sheet;
  - the 1948 credit in the corner when present.

- [ ] **Step 1: Write failing tests:**
  - `test_review_html_lists_failing_first`
  - `test_review_html_has_credit_for_1948`
  - `test_review_html_editor_link_only_for_needs_review`
  These parse the generated HTML; the sheet data is embedded as a JSON `<script>` block.
- [ ] **Step 2: Run, implement, pass.**
- [ ] **Step 3: Live check:** `uv run sfzmaps review --year 1948`. In the browser, the scan aligns visibly with block outlines at zoom 17 on a known unchanged area (e.g. Russian Hill). Take a screenshot into the tuning notes.
- [ ] **Step 4: Commit** `feat(pipeline): per-year review page`.

### Task 16: Editor backend

**Files:**
- Create: `src/sfzmaps/editor/server.py`, `tests/test_editor_api.py`

**Interfaces:**
- Produces: a FastAPI `app` launched by `sfzmaps edit [--year Y] [--sheet S]` on `127.0.0.1:8765`. Endpoints:
  - `GET /api/queue` → `[{series, sheet_id, status, frac_within, p95_ft, reason}]`, `needs_review` first.
  - `GET /api/sheet/{series}/{sheet_id}` → `{gcps: [GCP + residual_ft], matches: [Match], score: SheetScore, image_url, image_size, world_bounds}`. Starts from `review.json` if present, else `transform.json`. It also includes the answer-key GCPs if they exist and score better.
  - `GET /api/sheet/{series}/{sheet_id}/image/{z}/{x}/{y}.png` → scan tiles (Deep Zoom–style pyramid generated lazily with Pillow and cached under `work/.../dz/`).
  - `POST /api/sheet/{series}/{sheet_id}/preview` with body `{gcps, change_areas}` → `{score, residuals, preview_png_url}`. Fits a `SheetTransform` in memory, scores against the same matches (excluding points inside `change_areas`), and renders a low-res warped preview (≤ 2048 px) as a PNG with a world-file sidecar.
  - `POST /api/sheet/{series}/{sheet_id}/save` with body `{gcps, change_areas, status: "edited"|"accepted", note, reviewer}` → writes `review.json`, runs `runner.run_sheet` from `warp`, rebuilds that series' stitch/tiles/manifest/review page in a background task, and returns the new score. `status="accepted"` with `frac_within < PASS_FRACTION` requires a non-empty `note` (400 otherwise).
- Reviewer defaults to `git config user.email`.

- [ ] **Step 1: Write failing tests** with FastAPI `TestClient` on a synthetic sheet written into a temp `WORK_DIR`:
  - `test_queue_needs_review_first`
  - `test_preview_moving_bad_point_improves_score` (a GCP corrupted by 50 ft gets corrected → `frac_within` rises)
  - `test_save_writes_review_json_and_reruns`
  - `test_accept_below_threshold_requires_note`
- [ ] **Step 2: Run, implement, pass.**
- [ ] **Step 3: Commit** `feat(editor): control-point editor API`.

### Task 17: Editor front end

**Files:**
- Create: `src/sfzmaps/editor/static/index.html`, `editor.js`, `editor.css`

**Interfaces:**
- Consumes: Task 16 endpoints.
- Layout: a left pane shows the scan (OpenLayers or Leaflet with `L.CRS.Simple`, using the image tile endpoint). A right pane shows MapLibre with OSM-free modern block outlines (`OUT_DIR/ref/blocks.pmtiles`), centerlines, and an optional semi-transparent preview overlay. A top bar holds the sheet picker (from `/api/queue`), score readout, status, note field, and Save / Accept buttons.
- Behavior:
  - GCPs are drawn in both panes with matching IDs, colored by residual: green < `TOLERANCE_FT`, amber < 3×, red beyond. Real-change matches are blue and no-equivalent ones gray. Selecting a point in one pane highlights it in the other.
  - Drag a point in either pane to move that side. "Add point" mode: click the scan, then click the map. Delete and disable are keyboard shortcuts (`Del`, `D`).
  - "Mark change area" draws a polygon on the map.
  - "Worst areas" jumps through the 5 highest-residual clusters.
  - Every edit calls `/preview`, debounced 400 ms, and updates the score readout and overlay.
  - Leaving the page with unsaved edits prompts the user; that prompt lives inside the page, not as a browser dialog.
- Verification is manual (no JS unit tests): a scripted checklist in the Step below.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Manual check** on a deliberately corrupted synthetic sheet (`sfzmaps edit --demo` generates one into a temp work dir):
  - the queue shows it first;
  - dragging the red point onto the right corner turns it green and raises the score;
  - Save writes `review.json`;
  - rerunning `sfzmaps run --sheet demo` keeps the edit.
- [ ] **Step 3: Live check** on the worst-scoring real 1948 sheet. Fix it in under 10 minutes and record the result in the tuning notes.
- [ ] **Step 4: Commit** `feat(editor): browser control-point editor`.

### Task 19 (runs before Task 18): Share processed files via Drive

**Files:**
- Create: `src/sfzmaps/share.py`, `tests/test_share.py`
- Modify: `cli.py` (add `push`, `pull`), `sources/drive.py` (reuse `list_folder`/`sync` for pull)

**Interfaces:**
- Produces: `share.SHARED_GLOBS: list[str]`, relative to `DATA_DIR`:
  - `cache/llm/**`
  - `ref/*.parquet`
  - `work/*/*/{page.json,labels.json,transform.json,changes.geojson,score.json,review.json}`
  - `out/*/{*.pmtiles,manifest.json,review.html}`
  - `out/ref/blocks.pmtiles`
  It never includes `raw/`, PNGs or GeoTIFFs.
- Produces: `share.bundle(dest: Path) -> list[Path]`. Writes one `processed-<series>.zip` per series (that series' `work/` and `out/` files), plus `processed-cache.zip` (the LLM cache) and `processed-ref.zip` (the reference snapshot plus `ref/SNAPSHOT.json` with the fetch date and row counts). It also writes `PROCESSED.json` with sha256, size and created_at for each zip. Zipping is deterministic: entries sorted, fixed timestamps. Re-bundling unchanged data gives identical sha256 values, so `push` skips the upload.
- Produces: `share.pull(folder_id: str = ROOT_FOLDER_ID) -> None`. Finds the `processed` subfolder via the public listing (`drive.list_folder`), downloads `PROCESSED.json` and any zip whose sha256 differs from the local copy, and extracts into `DATA_DIR`. Local `review.json` files that are newer than the zipped copy are never overwritten; it logs a warning instead.
- Produces: `share.push(remote: str = "sfzoning-drive:old-planning-maps/processed") -> None`. Runs `share.bundle` into `DATA_DIR/share/`, then `rclone copy --checksum` to the remote. `rclone` is the upload tool (`brew install rclone`; the owner runs `rclone config` once to create the `sfzoning-drive` remote with Google Drive access). If rclone or the remote is missing, it exits with a one-line instruction, not a traceback.
- CLI: `sfzmaps push`, `sfzmaps pull`.

- [ ] **Step 1: Write failing tests:**
  - `test_bundle_includes_only_shared` (a temp DATA_DIR with raw/, PNG, TIF and the shared files → the zips contain exactly the shared files)
  - `test_bundle_deterministic` (two bundles have equal sha256)
  - `test_pull_keeps_newer_local_review` (fake downloader)
  - `test_push_missing_rclone_message` (monkeypatched `shutil.which` → `SystemExit` with the instruction text)
- [ ] **Step 2: Run, implement, pass.**
- [ ] **Step 3: Live check (owner present):** run `sfzmaps push` after Task 14's 1948 run, then `sfzmaps pull` into a fresh `SFZMAPS_DATA_DIR`, then `sfzmaps run --year 1948`. The run must make zero live model calls (`CallCounter` shows only cache hits) and reproduce the same `score.json` values.
- [ ] **Step 4: Commit** `feat(pipeline): share processed files via Drive`.

### Task 18: Full run across all eras

**Files:**
- Modify: `docs/superpowers/notes/2026-10-29-tuning.md` (final per-series table)
- Create: `pipeline/README.md` (setup: `brew install pmtiles`, `uv sync`, `OPENAI_API_KEY`; commands in order; where outputs land; how to review/edit)

- [ ] **Step 1:** Run `uv run sfzmaps fetch && uv run sfzmaps estimate --all`, and **stop to show the owner the dollar estimate before continuing.**
- [ ] **Step 2:** Run `uv run sfzmaps run --all`. Expected: every discovered page is either skipped with a kind, passed, or `needs_review` with a reason. No crashes; failures in one sheet don't stop the others (logged in `out/run-<timestamp>.log`).
- [ ] **Step 3:** Fill in the tuning notes with a per-series table: sheets, passed, needs_review, median p95_ft, PMTiles size, and size per zoom.
- [ ] **Step 4:** Write `pipeline/README.md`. Include "Recreate without re-running": `sfzmaps fetch && sfzmaps pull && sfzmaps run --all` (zero API spend).
- [ ] **Step 4b:** Run `sfzmaps push` to publish the full run's processed files to Drive.
- [ ] **Step 5: Commit** `docs(pipeline): full-run results and README`.
