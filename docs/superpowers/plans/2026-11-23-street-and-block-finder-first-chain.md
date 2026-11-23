# Street-and-Block Finder, First Whole Chain: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the street-and-block finder as far as its first whole chain (the design's stages 1 to 3): labels from lots and the 2000 table, hand crops, one model with three answers, closed blocks, counts, and pictures on test crops for the owner.

**Architecture:** A new package `sfzmaps.find` with a command group `sfzmaps find`. Labels reach the trainer in one shape (`Labelled`) whether they come from a lot-labelled sheet or a hand crop. Lines from lots and the table are taught as statements about short cuts ("a line crosses this cut" / "none does"), hand crops as per-pixel labels. Blocks are the basins of the "street or block edge" chance.

**Tech stack:** Python 3.12, PyTorch (mps through `sfzmaps.gpu.gpu_turn`), numpy, scipy.ndimage, scikit-image (`h_minima`, `watershed`, `threshold_otsu`; already a dependency), OpenCV, geopandas/shapely, Typer; `uv run` from `pipeline/`.

**Spec:** `docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md`. Read it first, and `docs/research/hatch-cnn-code-audit-2026-11-23.md` for what went wrong before.

**This plan stops at stage 3.** Stage 4 (every series) and stage 5 (hand-off to #69) get their own plan after the owner has seen stage 3's pictures. Also left for that plan: hand-traced zoning borders on the 1960-1980 and other 1980s-2000 maps, and the fitted share of table borders the drawing does not show (stage 3 first looks at the misses).

## Global Constraints

- **One model.** It is shown only the scan and the frame cut: no placement, lots, saved reads, series name, scale or "which way is up".
- **Never train on a label that might not match the print.** Lots label only the series marked `lots` in `inputs/find/label_sources.csv`; the April 2000 table labels zoning borders only on the three `2000-digital` series; 1921-1948 get "no zoning border" everywhere; everything else comes from hand crops.
- **The ground under test and choosing crops is left out of all training, in every year's scan.**
- **No hand-set cutoff, window or special case in the reading; no per-series path; no erasing or cleaning of ink.** A yes/no taken from a chance is "the more likely answer". Engineering settings (network size, schedule, tile size) are stated as such in a comment.
- **Nothing is counted, chosen or handed on from a crop the model trained on. Pictures before counts.** Every picture goes to the owner with SendUserFile and a caption saying what it is for.
- **Owner gates are stops:** after Task 3 (label pictures), Task 5 (hand crops) and Task 9 (the chain). Send the pictures and wait for a plain go.
- $0. No LLM output is read anywhere in `sfzmaps.find`. No `sfzmaps build`. Nothing under `pipeline/data/` is committed.
- Deterministic: fixed seeds, `torch.use_deterministic_algorithms(True)`, the device recorded; same inputs and device, same bytes. One GPU job at a time (`gpu.gpu_turn(True, device)` around all GPU work of a command).
- Scans are read at their own resolution, as stored. No resampling code of our own.
- Duplicate scans are never read (`solver.hatching.duplicate_scans`).
- Outputs under `pipeline/data/find/`, with the series in every path. Check `df -h /System/Volumes/Data` before a run that writes per-page maps; delete a superseded run's per-pixel maps, keep its pictures and counts.
- Say "zoning border" (never bare "border") in everything the owner reads.
- Before every commit run `pgrep -f "sfzmaps build"`; commit only if nothing is found. Messages end with:
  `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UjTgrnFxxUqpDGdyiFtNFS`
- Tests: `cd pipeline && OMP_NUM_THREADS=1 uv run pytest -q tests/test_find_*.py`; before a task's commit also the full `uv run pytest -q`.

## Review Focus

1. **A map page with no frame cut** is refused with the page named, never skipped (audit finding 9). Test in Task 1.
2. **A series with no row in `label_sources.csv`** is refused, never read with a default. Test in Task 1.
3. **A training crop on ground a test or choosing crop covers in another year's scan** has that ground unlabelled. Test in Task 4.
4. **A quarter turn of a tile with its labels gives the same loss.** Test in Task 7.
5. **A printed block the read calls all street is "lost", and a read that calls everything street finds no block** (audit finding 3). Test in Task 8.

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/src/sfzmaps/find/__init__.py` | What the finder is and the constraints that hold in the package |
| `find/pages.py` | Which pages are read; the scan; the frame cut; the label sources file |
| `find/lotlabels.py` | Street-or-not labels from a series' lots, on the grid and on a page |
| `find/cuts.py` | Cut statements for block edges and zoning borders |
| `find/handlabels.py` | Hand crop files, their labels, the reserved ground |
| `find/town.py` | Made-up towns drawn from measured numbers (tests and the border-label check) |
| `find/model.py`, `find/losses.py` | The network; the likelihoods |
| `find/data.py`, `find/train.py` | `Labelled`, tiles, the trainer |
| `find/read.py`, `find/areas.py`, `find/counts.py` | Reading a page; closed areas; the counts |
| `find/look.py` | Pictures |
| `pipeline/src/sfzmaps/cli.py` | `sfzmaps find ...` |
| `pipeline/inputs/find/label_sources.csv`, `town-measures.json` | Hand-decided sources; measured town numbers |
| `pipeline/inputs/find-labels/{train,choose,test}/<series>/*.json`, `README.md` | Hand crops |
| `pipeline/tests/test_find_*.py`, `tests/find_fixtures.py` | Tests |

---

### Task 1: The package, the pages, the label sources

**Files:**
- Create: `pipeline/src/sfzmaps/find/__init__.py`, `find/pages.py`, `pipeline/inputs/find/label_sources.csv`
- Modify: `pipeline/src/sfzmaps/cli.py` (a Typer sub-app `find_app`, `app.add_typer(find_app, name="find")`, with one command for now: `sfzmaps find sheets [--series S]` printing `series/sheet` per line)
- Test: `pipeline/tests/test_find_pages.py`

**Interfaces (produces):**
- `SheetId = tuple[str, str]`; `OUT = config.DATA_DIR / "find"`; `SOURCES = config.INPUTS_DIR / "find" / "label_sources.csv"`.
- `sheets(series: str | None = None) -> list[SheetId]`: every map page (`frame.data.map_pages()`) of the series (None: all), less its duplicate scans. Raises `FileNotFoundError` naming every such page with no `data/frame/<series>/<sheet>/outside.png`.
- `load_scan(sid) -> np.ndarray`: float32 (h, w), 1 paper to 0 ink, the page as stored (`frame.data.load_luminance`).
- `load_keep(sid, shape) -> np.ndarray`: bool, True where `outside.png` is 0; `ValueError` if its shape is not `shape`.
- `Source(series: str, era: str, lots: bool, borders: str)` and `sources(path=None) -> dict[str, Source]`; `source(series) -> Source` raises `KeyError` naming the file for a series with no row.
- `label_sources.csv`, columns `series,era,lots,zoning_borders`: `era` is one of `hatched-grey` (1921, 1929, 1935), `hatched-colour` (1948), `lined-grey` (1960, 1964, 1968, 1972, 1975, 1980), `lined-1bit` (every `1980-1990-*`, `1990-1996-*`, `1996-2000-*`, `2000-digital-*`); `lots` is `year` for the `lined-1bit` series and blank otherwise; `zoning_borders` is `zoning`, `height_lim`, `sud` for `2000-digital-zoning`, `-height`, `-su`, `none` for the four hatched series, blank otherwise. One row for each of the 22 series under `data/work`.

- [ ] **Step 1: Write the failing tests** (pages built in `tmp_path` with monkeypatched `frame.data.WORK`, `OUT`, `PAGE_KINDS` and `config.INPUTS_DIR`):
  - `test_sheets_leaves_out_duplicates_and_non_maps`: two map pages, one text page, one listed duplicate; `sheets("s") == [("s", "a")]`.
  - `test_a_map_page_without_a_frame_cut_is_refused`: `pytest.raises(FileNotFoundError, match="s/b")`.
  - `test_keep_is_the_frame_cut_and_its_shape_is_checked`: zeros kept, non-zero not; an 8 x 8 cut for a 10 x 10 page raises `ValueError`.
  - `test_colour_scan_is_luminance`: a pure red page loads as 0.299 within 1/255.
  - `test_sources_rows_and_refusals`: the checked-in file has 22 rows; `source("2000-digital-height").borders == "height_lim"`; `source("1921") == Source("1921", "hatched-grey", False, "none")`; an unknown `zoning_borders` value or a repeated series raises `ValueError`; `source("nope")` raises `KeyError`.
- [ ] **Step 2:** Run `uv run pytest -q tests/test_find_pages.py`. Expected: FAIL (no module).
- [ ] **Step 3:** Implement. `__init__.py` is a docstring: what the finder answers and the constraints above, in the manner of `frame/__init__.py`.
- [ ] **Step 4:** Run the tests (PASS), then `uv run sfzmaps find sheets --series 1921` (expected: 14 lines).
- [ ] **Step 5:** Commit `find: the package, the pages it reads and the label sources`.

### Task 2: Street-or-not labels from a series' lots

**Files:**
- Create: `find/lotlabels.py`; Modify: `solver/hatchnet/data.py` (`learn_half_widths`, `street_labels`, `street_grid` take `per_type: bool = True`; False gives every street type the pooled half-widths, with no 95% test); `cli.py`
- Test: `pipeline/tests/test_find_lotlabels.py`, one test added to `tests/test_hatchnet_data.py`

**Interfaces:**
- Consumes: `pages.source`, `districts.read.load_parcels`, `districts.read.series_year`, `solver.table.load_table`, `solver.lots.lot_rasters`, `hatchnet.data.centre_lines` / `street_grid`, `hatching.PageWorld` / `page_values`, `transform.SheetTransform`.
- Produces:
  - `series_lots(series) -> gpd.GeoDataFrame`: the April 2000 table (`load_table()`) when the series' `borders` is a table column, else `load_parcels(series_year(series))`. `ValueError` for a series whose `lots` is False.
  - `Ground(lot: int32, block: bool, street: uint8, weight: float16)`, all on the solver grid: `lot` from `lot_rasters`; `block = ndimage.binary_fill_holes(lot >= 0)`; `street`, `weight` from `street_grid(lines, block, per_type=False)` (`STREET`, `NOT_STREET`, `UNKNOWN`).
  - `ground(series) -> Ground`, cached in `OUT / "labels" / series / "grid.npz"`.
  - `placement(sid, which: str = "warp") -> SheetTransform`: `data/warp/<series>/<sheet>/transform.json` with its correction, or for `"work"` `data/work/.../transform.json`.
  - `PageGround(street: uint8 (h, w) 1 street / 0 block / 255 no label, weight: float32)`, and `page_ground(g, tr, shape, keep) -> PageGround`: the grid rasters read at every page px; 255 and weight 0 off `keep`.

- [ ] **Step 1: Write the failing tests** (a 60 x 80 grid world built by `tests/find_fixtures.py`: `lots_world()` returns a GeoDataFrame of four lots in two blocks either side of a 16 ft street, a centre line down the street, and an identity-like placement `Flat(ft_per_px)` with `to_world` / `to_pixel`):
  - `test_block_px_are_not_street_and_the_gap_is`: on the page, px inside either block are 0 with weight 1; px in the street are 1 with weight in (0, 1]; px beyond the blocks with no centre line are 255.
  - `test_off_the_frame_cut_is_unlabelled`: `keep` False on the left half gives 255 and weight 0 there.
  - `test_a_series_without_lot_labels_is_refused`: `series_lots("1921")` raises `ValueError`.
  - in `test_hatchnet_data.py`, `test_pooled_half_widths_have_no_type_test`: with `per_type=False`, `learned.own == {}` and every type's samples are the pooled ones.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests (PASS) and the full suite.
- [ ] **Step 5:** Add `sfzmaps find labels --series S [--sheet X ...] [--placement warp|work]` (it grows in Task 3; for now it builds `ground(S)` and prints the px per label). Run it for `2000-digital-zoning`. Expected: a `grid.npz` and three counts.
- [ ] **Step 6:** Commit `find: street-or-not labels from a series' lots; pooled street half-widths`.

### Task 3: Cut statements for block edges and zoning borders, and the label pictures

**Files:**
- Create: `find/cuts.py`, `find/look.py`; Modify: `find/lotlabels.py` (the per-sheet label file), `cli.py`
- Test: `pipeline/tests/test_find_cuts.py`

**Interfaces:**
- `EDGE, BORDER = 0, 1`.
- `Cuts(px: int64 (m,), cut: int32 (m,), prior: float32 (m,), answer: uint8 (n,), says: bool (n,))`: `px` is the flat page index (`row * w + col`) of every cut px, cut after cut; `prior` is 1 where the line is most likely and falls off; `says` True is "a line crosses this cut", False "none does". `Cuts.joined(a, b) -> Cuts`.
- `spread_px(sid, which="warp") -> float`: the sheet's measured placement spread in page px: `scores.json` `control_miss_p50_<chosen>` / 1.1774 (the median radius of a 2-D normal is 1.1774 sigma), combined in quadrature with the grid's own 4 ft / sqrt(12) in page px.
- `edge_cuts(g: Ground, tr, shape, keep, sigma: float) -> Cuts`: one cut for every 4-neighbour pair of grid px (u in a block, v not). Centre: their midpoint, through `tr.to_pixel`. Direction: from u to v, as it lands on the page. Px: the centre and `R = ceil(4 * sigma)` px each way (numerics only: the prior there is 3e-4 of its peak). `prior = exp(-t**2 / (2 * sigma**2))`. `says` True.
- `border_cuts(lot: int32, codes: np.ndarray, tr, shape, keep, sigma: float) -> Cuts`, `codes[i]` the code of lot i, "" unknown:
  - touching lots: every 4-neighbour pair (u, v) in two different lots with known codes: built as an edge cut; `says = codes differ`;
  - across a street: every 4-neighbour pair (u, v) off the lots whose nearest lots a, b differ (`ndimage.distance_transform_edt(lot < 0, return_indices=True)`) with known codes: the page px on the straight line from u's nearest lot px to v's nearest lot px, extended `R` px at both ends; prior 1 between the two lot px and the Gaussian fall-off beyond them; `says = codes differ`.
- A cut with any px off the page or off `keep` is dropped whole.
- `lotlabels.sheet_labels(sid, which="warp") -> dict` writes and returns `OUT / "labels" / series / f"{sheet}.npz"`: `street`, `weight` (Task 2), `cuts` (edge cuts for a `lots` series; border cuts where the series has a table column, with `codes` that column of the table, "" where it is missing), `edge_none` (bool: px with a street-or-not label on no edge cut), `border_none` (bool: px inside a lot with a known code on no border cut; for a `none` series, all of `keep`), `sigma`, `placement`.
- `look.labels(sid, lum, lab, box=None) -> np.ndarray` (BGR): the scan with street orange, block blue, edge cuts as thin cyan ticks, "a zoning border crosses" cuts red, "none" cuts green; a key and the sheet's name drawn on it.

- [ ] **Step 1: Write the failing tests** (the Task 2 world, codes `["A", "A", "B", "B"]` so that the two blocks differ and each block's two lots agree, plus a variant with `["A", "B", "B", "B"]`):
  - `test_an_edge_cut_crosses_each_block_side`: every edge cut's px run perpendicular to the block side under it, are `2 * R + 1` long, and their prior peaks at the middle px.
  - `test_touching_lots_same_code_say_none_and_different_say_one`: the variant's cuts across the shared lot line have `says` True; the first world's have `says` False.
  - `test_a_street_cut_spans_the_street`: a cut between the two blocks covers every px of the 16 ft street on its line, with prior 1 on all of them, and `says` True when the codes differ.
  - `test_unknown_code_makes_no_statement`: with one lot's code "", no border cut touches that lot.
  - `test_a_cut_leaving_the_frame_cut_is_dropped`: with `keep` False on a strip through the street, no cut has a px in the strip.
  - `test_hatched_series_say_no_zoning_border_everywhere`: `sheet_labels` for a fixture series whose `borders` is `none` has `border_none == keep` and no border cuts.
  - `test_doubling_the_cut_length_changes_nothing_that_matters`: the summed prior of a cut built with `R` and with `2 * R` differ by under 1e-3 of the sum.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests (PASS) and the full suite.
- [ ] **Step 5: Build and draw the labels.** `sfzmaps find labels --series 2000-digital-zoning --sheet 2000-digital-zoning-p05 --sheet 2000-digital-zoning-p07 --placement warp --look`, the same with `--placement work`, and `--series 2000-digital-height --sheet 2000-digital-height-p07 --look`. Pictures go to `OUT / "look" / "labels" / <placement>/`, whole sheet and a 640 px close-up at 2x. Look at every picture: do the cyan ticks sit on the thin block lines, the red ones on the heavy lines, and does either placement put block-edge ticks on a zoning border?
- [ ] **Step 6:** Commit `find: cut statements for block edges and zoning borders; label pictures`.
- [ ] **Step 7: Owner gate.** Send the close-ups for both placements with a caption saying which placement sits better on the block lines and why. Wait for the go, and for which placement to use (the default stays `warp` unless the pictures say otherwise).

### Task 4: Hand crops: the file, the labels, the reserved ground

**Files:**
- Create: `find/handlabels.py`, `pipeline/inputs/find-labels/README.md`; Modify: `find/look.py`, `cli.py`
- Test: `pipeline/tests/test_find_handlabels.py`

**Interfaces:**
- `SETS = ("train", "choose", "test")`; `ROOT = config.INPUTS_DIR / "find-labels"`.
- A crop file `ROOT/<set>/<series>/<sheet>__<x0>_<y0>_<w>_<h>.json`:
  ```json
  {"box": [x0, y0, w, h], "street": [[x, y]], "blocks": [[x, y]], "open": [[x, y]],
   "strokes": [[[x, y], [x, y]]], "limits": [[[x, y], [x, y]]], "notes": "", "drawn_by": ""}
  ```
  Coordinates are `page.png` px as stored. `street`, `blocks`, `open` are seed points. A stroke is a printed line (it closes a gap in it, or separates two blocks that share a line). A limit stops the fill where no line is printed.
- `Crop(set, series, sheet, box, street, blocks, open, strokes, limits, path)`; `crops(sets=SETS, root=None) -> list[Crop]`, sorted by path. `ValueError` for a file whose name and `box` disagree.
- `CropLabels(street: uint8 1/0/255, area: int32, kinds: dict[int, str], edge: uint8 1/0/255, warnings: list[str])` and `make_labels(lum_crop: np.ndarray, crop: Crop) -> CropLabels`, by these rules in this order:
  1. paper = luminance above `skimage.filters.threshold_otsu` of the crop;
  2. strokes and limits are drawn 3 px wide as barriers; barrier px are not paper;
  3. street = paper 4-connected to a street seed (a street seed not on paper: `ValueError` naming the file and the seed);
  4. islands = 8-connected pieces of what is neither street nor barrier. An island holding seeds is an area, numbered in the order its first seed appears (`blocks` then `open`), its kind from the seed; an island holding two seeds: `ValueError` "draw a stroke between them"; an island with no seed is street (lettering);
  5. `street` label: 1 street, 0 area, 255 barrier;
  6. `edge` label: 1 on area px 8-adjacent to anything that is not the same area, and on stroke px; 255 on px 8-adjacent to a limit, on limit px, and on the crop's outermost rows and columns; 0 elsewhere;
  7. `warnings` lists unseeded islands larger than the smallest seeded one ("possible missed block at x, y").
- `reserved(sid, shape, margin: int, root=None, which="warp") -> np.ndarray`: bool (h, w), True on ground under any `choose` or `test` crop of any sheet (its box corners through that sheet's placement to the ground and through this sheet's onto its page, filled), grown by `margin` px.
- `look.crop(lum_crop, labels) -> np.ndarray`: street orange, each area its own colour (open land striped), edge px black on white, unlabelled px grey, seeds as dots.
- `sfzmaps find crops [--set S] --look` draws every crop to `OUT / "look" / "crops" / <set>/` and prints each crop's warnings.

- [ ] **Step 1: Write the failing tests** (crops drawn with numpy: white paper, 2 px black lines):
  - `test_the_fill_stops_at_a_line_closing_a_street`: a street closed by a line; px beyond the line are not street.
  - `test_letters_in_the_street_are_street`: a small unseeded ink blob in the street has `street == 1`.
  - `test_a_stroke_closes_a_gap_and_is_an_edge`: a block outline with a 4 px gap; without the stroke the block's seed island is the whole crop's (two seeds, `ValueError`); with it the block is its own area and the stroke px are `edge == 1`.
  - `test_a_limit_stops_the_fill_and_labels_no_edge`: an open-ended street; with a limit the land beyond is not street and px beside the limit are `edge == 255`.
  - `test_two_blocks_sharing_a_line_need_a_stroke`: two seeds in one island raise `ValueError` matching "stroke"; with a stroke along the shared line there are two areas.
  - `test_open_land_is_its_own_kind`: an `open` seed gives `kinds[id] == "open"`.
  - `test_the_crop_border_is_not_an_edge`: a block cut by the crop's side has `edge == 255` on the outermost column, `1` on its printed sides.
  - `test_reserved_ground_reaches_other_scans`: a `test` crop on sheet A covers ground that sheet B (a `Flat` placement shifted by 100 px) also shows; `reserved(B)` is True on exactly that box grown by `margin`, False elsewhere; a `train` crop reserves nothing.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Write `inputs/find-labels/README.md`: the file format, the seven rules, the three sets and what each is for, and the reading of the maps from the design (a line closing a street is a block edge; a road drawn as two lines is a street; open land is its own kind; railway lines, shorelines and contours are neither edge nor zoning border).
- [ ] **Step 4:** Run the tests (PASS) and the full suite.
- [ ] **Step 5:** Commit `find: hand crops (seeds and strokes), their labels and the reserved ground`.

### Task 5: Draw the first hand crops (labelling, no code)

**Files:** `pipeline/inputs/find-labels/{test,choose,train}/<series>/*.json`

**Before a crop is drawn on a series, check the series for duplicate scans:** list its map pages with their sizes, look at small copies side by side, and send any two scans of one sheet to the owner as a proposed row for `inputs/<series>/duplicate_scans.csv`. No crop is drawn on either until he answers. (Only 1921 has been checked.)

Each crop is 1,024 x 1,024 px unless the sheet is smaller. To draw one: cut the crop to a PNG with a coordinate grid along its edges, look at it, write the seeds and strokes, run `sfzmaps find crops --look`, look at the picture, and correct until every block is its own area, every street is street and no warning is unexplained. Record anything undecided in `notes`.

- [ ] **Step 1: The test set, 12 crops, three per era,** chosen for what is hard and one of plain ground each:
  - `hatched-grey`: 1921 Section 12 at Visitacion and Sunnydale (outline-only blocks, double-line roads, street ends: `20180102100525362`, box about `[880, 3030, 1024, 1024]`); a 1935 crop with dead-end alleys in cross-hatched blocks; a 1929 crop of plain hatched blocks.
  - `hatched-colour`: three 1948 crops (hatched blocks with printed dimensions; solid commercial strips; an institution's grounds).
  - `lined-grey`: 1960 (zoning borders down street centres and through blocks), 1975 (borders beside block lines), 1980 (the coarse scan, stair-step borders).
  - `lined-1bit`: `2000-digital-zoning-p07` (borders with jogs, labels over lines), `2000-digital-height-p07` (curving streets), a `1980-1990-zoning` crop (borders drawn beside block lines).
- [ ] **Step 2: The choosing set, 8 crops, two per era,** on other sheets or far from the test crops.
- [ ] **Step 3: The first training crops, 16:** six on `hatched-grey` (four sheets of 1921, one each of 1929 and 1935), two on 1948, six on `lined-grey` (1960, 1964, 1968, 1972, 1975, 1980), two on `lined-1bit`. No training crop on ground a test or choosing crop covers (check with `reserved`).
- [ ] **Step 4:** Commit `find: the first hand crops (12 test, 8 choosing, 16 training)`.
- [ ] **Step 5: Owner gate.** Send the pictures of every test crop and four training crops, each with the scan beside its labels, and the list of what was undecided. Wait for his corrections and his go.

### Task 6: The network, the likelihoods, and the zoning-border label on a made-up town

**Files:**
- Create: `find/model.py`, `find/losses.py`, `find/town.py`, `pipeline/inputs/find/town-measures.json`; Modify: `cli.py`
- Test: `pipeline/tests/test_find_model.py`, `tests/test_find_losses.py`, `tests/test_find_town.py`

**Interfaces:**
- `model.STREET, EDGE, BORDER = 0, 1, 2`; `ANSWERS = ("street", "edge", "border")`; `FindNet(width: int = 16)`: the branch's U-Net (`hatchnet.model._block`, `LEVELS`, `REACH`), input (n, 1, h, w) luminance, output (n, 3, h, w) logits.
- `losses.dense_nll(logit, target, weight) -> Tensor`: the sum over px of `weight * BCE`; target 0/1, weight 0 where there is no label.
- `losses.cut_nll(logit_flat, cuts_px, cuts_cut, prior, says) -> Tensor`: the sum over cuts of minus the log chance of its statement. "A line crosses": `1 - prod_i (1 - prior_i * p_i)`. "None does": `prod_i (1 - p_i)`. Computed in logs.
- `losses.statements(...) -> int`: how many statements a batch held (labelled px plus cuts); the trainer divides the summed loss by it. The three answers add as one likelihood with no weights between them.
- `town.measures() -> dict` reads `town-measures.json`: for street width, block side, block-line width and zoning-border width, 101 quantiles in px each; `town.measure(series, sheets) -> dict` makes them from built label files (street width: the between-lots run of across-street cuts; block side: block runs along rows and columns; line widths: the ink run under the best px of each edge cut and of each "crosses" border cut). `sfzmaps find measure-town` writes the file from `2000-digital-zoning` p05 and p07, with the command, date and git commit in it.
- `town.draw(rng, m: dict, shape=(192, 192)) -> Town(lum, lot, codes, block_line: bool, border_line: bool)`: axis-aligned blocks of two to four lots, streets, block outlines and zoning borders (down a street's middle, beside a block line, or along a lot line inside a block, in equal shares) with every size drawn from `m`; lot lines are not printed.

- [ ] **Step 1: Write the failing tests:**
  - `test_shapes`: `FindNet(8)` maps (2, 1, 64, 64) to (2, 3, 64, 64).
  - `test_dense_nll_ignores_unlabelled_px`: changing the logit where weight is 0 changes nothing.
  - `test_a_crossing_statement_needs_one_px`: one cut of three px with priors (0.5, 1, 0.5): with p = (0, 1, 0) the loss is 0 within 1e-6; with p = (0, 0, 0) it is over 10; with p = (1, 0, 0) it is `-log(0.5)`.
  - `test_a_none_statement_needs_every_px_off`: p = (0, 0, 0) gives 0; p = (0, 0.5, 0) gives `-log(0.5)`.
  - `test_measures_file_is_complete`: the checked-in file has the four keys, each of 101 non-decreasing numbers, and names its command and commit.
  - `test_town_borders_follow_the_codes`: in a drawn town, every "crosses" border cut (built by `cuts.border_cuts` with a `Flat(4.0)` placement) has a `border_line` px on it, and no "none" cut does.
  - `test_the_border_answer_learns_which_ink_is_the_border` (the design's made-up case; CPU, `FindNet(8)`, tile 64, batch 8, 400 steps, seed 0; trained on 16 towns with edge and border cuts only): on a fresh town, more than half of the "crosses" cuts have a px whose border chance is over one half, fewer than half of the "none" cuts do, and the mean border chance on `border_line` px is above the mean on `block_line` px that are not `border_line`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the model and losses; then `town.measure` and the command; run `sfzmaps find measure-town` and commit its file with the code; then `town.draw`.
- [ ] **Step 4:** Run the tests (PASS; the last one within about a minute) and the full suite. **If the made-up case fails after two honest fixes, stop and report to the owner with pictures of what the border answer lights up** (the design's known weakness; do not patch a third time).
- [ ] **Step 5:** Commit `find: the network, the likelihoods, and the zoning-border label learned on made-up towns`.

### Task 7: `Labelled`, tiles and the trainer

**Files:**
- Create: `find/data.py`, `find/train.py`; Modify: `cli.py`
- Test: `pipeline/tests/test_find_train.py`

**Interfaces:**
- `data.Labelled(name: str, era: str, lum: float32 (h, w), street: uint8 1/0/255, street_w: float32, edge: uint8 1/0/255, border: uint8 1/0/255, cuts: Cuts | None)`.
- `data.sheet_labelled(sid, which="warp") -> Labelled`: from Task 3's file; `edge` is 0 on `edge_none`, 255 elsewhere; `border` is 0 on `border_none`, 255 elsewhere; everything under `handlabels.reserved(sid, shape, model.REACH)` set to 255 / weight 0, and cuts touching it dropped.
- `data.crop_labelled(crop) -> Labelled`: from `make_labels`; `street_w` 1 where labelled; `border` 0 everywhere when the series' `borders` is `none`, else 255; no cuts. Raises for a `train` crop any px of which is reserved.
- `data.turned(tile: Labelled, k: int) -> Labelled`: the tile and every label turned k quarter turns (`np.rot90`), cut px re-indexed.
- `train.Recipe(steps=3000, batch=16, tile=256, width=16, max_lr=2e-3, pct_start=0.3, weight_decay=1e-4, jitter=0.25, every=250, answers=ANSWERS, unbroken=False, seed=0)`: the branch's engineering settings, stated as such. An answer left out of `answers` gets no loss.
- `train.batch_loss(net_out, tiles: list[Labelled], recipe) -> Tensor`.
- `train.fit(items: list[Labelled], recipe, device, on_checkpoint=None) -> Fitted(state_dict, step, curve, device)`: each batch takes equal numbers of tiles from every era present among `items` (the remainder to the eras first in sorted order), a tile's centre drawn uniformly from that era's labelled px, each tile given a random quarter turn and the branch's intensity jitter. `on_checkpoint(step, state_dict) -> float` is called every `every` steps; the step kept is the one with the highest return, the earliest of a tie; with no callback, the last. With `unbroken`, hand-crop tiles add the branch's `topo` recall and precision terms between the edge chance and the hand edge label.
- `sfzmaps find train --run NAME [--sheet series/sheet ...] [--without-borders] [--unbroken] [--device D]`: trains on the named lot-labelled sheets and every `train` crop; writes `OUT / "runs" / NAME / {model.pt, log.json}`. (It keeps the last step until Task 8 gives it the chooser.)

- [ ] **Step 1: Write the failing tests:**
  - `test_a_quarter_turn_keeps_the_loss`: for a random logit array and a tile with dense labels and cuts, `batch_loss(z, [t])` equals `batch_loss(rot90(z, k), [turned(t, k)])` to 1e-5 for k = 1, 2, 3.
  - `test_reserved_ground_has_no_label`: a sheet's `Labelled` with a reserved box has 255 and weight 0 there and no cut px in it.
  - `test_a_training_crop_on_reserved_ground_is_refused`.
  - `test_each_era_gets_its_share_of_a_batch`: with items of three eras and batch 16, the tiles drawn per era are 6, 5, 5.
  - `test_an_answer_left_out_gets_no_gradient`: with `answers=("street", "edge")`, the border head's weights are bit-equal before and after 3 steps.
  - `test_same_seed_same_bits_cpu`: two 5-step CPU fits give identical state dicts.
  - `test_the_kept_step_is_the_best_by_the_callback`: a callback returning 0.2, 0.9, 0.9, 0.1 keeps the second checkpoint.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests (PASS) and the full suite.
- [ ] **Step 5:** Commit `find: Labelled, tiles by era and quarter turn, the trainer`.

### Task 8: Reading a page, closed areas, the counts

**Files:**
- Create: `find/read.py`, `find/areas.py`, `find/counts.py`; Modify: `find/look.py`, `cli.py`
- Test: `pipeline/tests/test_find_areas.py`, `tests/test_find_counts.py`, `tests/test_find_read.py`

**Interfaces:**
- `read.read_page(net, lum, keep, device) -> np.ndarray`: float32 (3, h, w) chances, 0 off `keep`; in 1,024 px squares with 128 px of context (engineering: memory only).
- `read.save(path, chances)` / `read.load(path)`: uint8 levels `round(255 p)` under keys `street`, `edge`, `border`.
- `areas.height(street, edge) -> np.ndarray`: `1 - (1 - street) * (1 - edge)`.
- `areas.areas(street, edge, keep, depth: float) -> np.ndarray`: int32, 0 none. Markers: the connected pieces of `skimage.morphology.h_minima(height, depth)`; basins: `skimage.segmentation.watershed(height, markers, mask=keep)`; an area is a basin's px where `street < 0.5` (not street is the more likely answer); a basin with no such px is dropped.
- `counts.blocks(found: int32, truth: CropLabels) -> dict`: every hand area of kind `block` is exactly one of, tested in this order: `lost` (more than half its px have no found area), `found` (one found area holds more than half its px and more than half of that area's px are it), `joined` (the found area holding most of it is mostly outside it), `cut_up` (otherwise). Also `extra`: found areas more than half of whose px lie on hand street. Open-land areas are not counted and make nothing `extra`.
- `counts.found_share(dicts_by_era: dict[str, list[dict]]) -> float`: the mean over eras of `found / blocks`.
- `counts.fit_depth(reads, truths, eras) -> tuple[float, float]`: the depth among `i / 255, i = 1..254` (every level the stored chances can tell apart) with the highest `found_share`, the smallest of a tie, and that share.
- `counts.edge_feet(found: int32, truth: CropLabels, ft_per_px: float) -> dict`: for hand edge px of found blocks, the distance to the nearest found-area boundary px, in feet: median and 90th percentile.
- `counts.borders(border_chance, cuts: Cuts) -> dict`: of the "crosses" border cuts, how many have a px over one half; of the "none" cuts, how many do.
- `counts.chooser(crops: list[Crop], device) -> Callable[[int, dict], float]`: reads every `choose` crop with the checkpoint's weights and returns `fit_depth(...)[1]`. `sfzmaps find train` now passes it as `on_checkpoint`.
- `sfzmaps find count --run NAME`: reads every `test` crop with the kept model at the depth fitted on the `choose` crops; writes `OUT / "runs" / NAME / counts.json` (per era and pooled; border counts from the table's cuts on test crops of table series; edge feet with each crop's ft per px taken from its sheet's placement, which is read here for scoring only) and pictures `look/test/<crop>.png` (the scan, the three chances, the areas with found blocks outlined green, joined red, cut up orange, lost purple).

- [ ] **Step 1: Write the failing tests:**
  - `test_a_faint_stretch_still_separates_two_blocks`: two blocks split by an edge ridge of 0.9 that dips to 0.6 for 3 px: one area at `depth=0.5`, two at `depth=0.2`.
  - `test_a_block_reaches_only_where_not_street_is_likelier`: basins run to mid-street; area px are exactly those with `street < 0.5`.
  - `test_everything_street_finds_no_block`: `street` all 1 gives no area, and `counts.blocks` gives every hand block `lost`, `found == 0`.
  - `test_each_block_is_exactly_one_thing`: hand-built reads for found, joined (two blocks in one area), cut up (three areas, none over half) and lost; the four counts are 1 each and sum to the blocks.
  - `test_open_land_is_not_counted`: an `open` hand area changes no count.
  - `test_fit_depth_takes_the_smallest_best`: on the faint-stretch case with one hand block either side, the fitted depth is the smallest level that finds both.
  - `test_tiled_reading_is_one_pass`: `read_page` in 64 px squares equals one pass within 1e-5 on a 200 x 150 page (`FindNet(4)`, CPU).
  - `test_border_counts_are_per_cut`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests (PASS) and the full suite.
- [ ] **Step 5:** Commit `find: reading a page, closed areas by watershed, the block and zoning-border counts`.

### Task 9: The first whole chain (stage 3)

Sheets with lot labels for this run: `2000-digital-zoning-p05`, `-p07`, `-p11`, `2000-digital-height-p07`, `2000-digital-su-p10`, `1980-1990-zoning-p07`. Hand crops: all 16 training crops.

- [ ] **Step 1:** `df -h /System/Volumes/Data`; `pgrep -fl "sfzmaps|hatchnet"` (no other GPU job). Build the six sheets' labels (`sfzmaps find labels ...`).
- [ ] **Step 2: Run A, the design as written:** `sfzmaps find train --run a-three-answers --sheet ...` then `sfzmaps find count --run a-three-answers`. Note the wall time of the training. **If it runs past twice the branch's 25 minutes for 3,000 steps, or the loss is not finite, stop and report.**
- [ ] **Step 3: Look** at every test-crop picture of run A before reading its counts. Write down, per era, what is right and what is wrong.
- [ ] **Step 4: Run B, without the zoning-border answer:** `--run b-no-borders --without-borders`, then count. The comparison the owner asked for: on the test crops of table series, of the table's "crosses" cuts between touching lots (a zoning border inside a block, where no block edge is drawn), the share with an edge chance over one half somewhere, run A against run B.
- [ ] **Step 5: Run C, with the unbroken-line loss:** `--run c-unbroken --unbroken`, then count.
- [ ] **Step 6:** Write `docs/research/street-and-block-finder-first-chain-<date>.md`: per run and era, the block counts and edge feet; the border counts; the A-against-B comparison; training time; how many hand crops each era had and what the pictures say it lacks; the fitted depth. No conclusion from a number without its picture.
- [ ] **Step 7:** Commit the note. Delete the per-page maps of runs not kept.
- [ ] **Step 8: Owner gate.** Send, per era, one test crop of run A (scan, three answers, areas), the Section 12 crop at Visitacion and Sunnydale, the A-against-B picture on one 2000 crop, and the table of counts, each with what it is for. Say what is still wrong. Wait.
