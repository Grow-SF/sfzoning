# Hand crops for the street-and-block finder

Hand-made labels on crops of the scans, the only teacher of the street-and-block finder (`sfzmaps.find`) for every
map before the 1980s. A person marks a crop, one of two ways (below); `sfzmaps.find.handlabels` makes the per-pixel
labels from the marks. Design: `docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md`.

One file per crop: `<set>/<series>/<sheet>__<x0>_<y0>_<w>_<h>.json`, the sheet named as in
`data/work/<series>/<sheet>/page.png`, the name's numbers the same as the file's `box`.

## Two ways to draw a crop

- **Seeds and strokes**, for maps whose streets are open paper: the hatched maps of 1921-1948, where a thin outline
  closes each block and the street between is white. A few seed points and strokes; the fill does the rest.
- **Outlines**, for maps where heavy lines and letters join the blocks: 1960 on, where zoning borders run down the
  streets and letters cross them, so the ink of every block is joined and no fill can tell the blocks apart. Every
  printed block in the box is typed as a polygon. **Outline crops are for scoring only for now:** they may be in
  `choose` or `test`, never `train` (a corner typed by eye is a pixel or two off the printed line, and teaching edges
  from that waits until that spread is measured).

A file holds marks of one kind only; a file with both, or with neither, is refused.

A seed crop:

```json
{
 "box": [x0, y0, w, h],
 "street": [[x, y]],
 "blocks": [[x, y]],
 "open": [[x, y]],
 "strokes": [[[x, y], [x, y]]],
 "limits": [[[x, y], [x, y]]],
 "openings": [[[x, y], [x, y]]],
 "notes": "what was ambiguous and how it was decided",
 "drawn_by": "who, how, date"
}
```

Coordinates are `page.png` pixels at full resolution, as stored (x right, y down, the page as scanned, not turned).
`street`, `blocks` and `open` are seed points. `strokes`, `limits` and `openings` are polylines (two or more points):

- a **stroke** is a printed line: it closes a gap in a printed line, or separates two blocks that share one;
- a **limit** stops the fill where no line is printed (a street running out of the crop into open land, say);
- an **opening** lets a street through printed ink where the owner has said it runs through.

An outline crop:

```json
{
 "box": [x0, y0, w, h],
 "polygons": [[[x, y], [x, y], [x, y], ...]],
 "open_polygons": [[[x, y], [x, y], [x, y], ...]],
 "notes": "...",
 "drawn_by": "..."
}
```

`polygons` holds one outline per printed block, its corners on the outer edge of the block's printed line (at least
3 corners); `open_polygons` the same for land with no outline of its own. A block cut by the crop's side gets its
corners on the box's edge (or up to 2 px past it; it is clipped to the box).

Check every crop by eye: `uv run sfzmaps find crops --look` (in `pipeline/`) draws the scan and the labels side by side,
twice size, with page-pixel rulers to read coordinates off, under `data/find/look/crops/<set>/`, and prints each crop's
warnings and its areas by kind.

To read coordinates off before typing marks, `uv run sfzmaps find ruled --sheet <series>/<sheet> --box x0,y0,w,h
[--zoom 4] [--tiles 2]` cuts the box into tiles, each enlarged (4x by default) with page-pixel rulers and a light grid
every 50 px, plus the whole box at 2x, under `data/find/look/ruled/<series>__<sheet>__<x0>_<y0>_<w>_<h>/`.

## The three sets

- **`test`**: drawn first; used for nothing but the pictures and counts shown to the owner.
- **`choose`**: used only to choose the training step kept and the one setting of the area step.
- **`train`**: the only hand labels the model learns from; added to wherever the pictures show misses.

The ground under every `test` and `choose` crop is left out of all training, in every year's scan of that ground
(`handlabels.reserved`): 1921 and 1929 draw the same streets, so a training crop on one would give away a test crop on
the other. A crop must lie on a sheet that has a placement, and never on a duplicate scan.

## How the labels are made

### Outline crops

Each polygon is an area, numbered in file order (`polygons`, then `open_polygons`), of kind `block` or `open`, filled on
the crop (corners in page px less the box's corner; px centres at integers). Two polygons sharing any px, or a corner
more than 2 px outside the box, is an error. The **street** label is 1 outside every polygon and 0 inside; the
**edge** label 1 on area px 8-adjacent to anything not the same area, 255 (no label) on the crop's outermost rows and
columns, 0 elsewhere. No warnings.

### Seed crops

In this order:

1. **Paper** is the crop's px lighter than its Otsu threshold.
2. **Openings** are drawn 3 px wide and count as paper whatever the scan shows: a place where the map's owner has said
   a street runs through printed ink (a line drawn across a street's mouth, lettering across a street). A stroke or
   limit over the same px wins.
3. **Strokes and limits** are drawn 3 px wide as barriers; a barrier px is not paper.
4. **Street** is the paper 4-connected to a street seed. A street seed not on paper is an error.
5. **Islands** are the 8-connected pieces of what is neither street nor barrier. An island holding a seed is an
   **area**, numbered in the order its first seed appears in the file (`blocks`, then `open`), of the seed's kind
   (`block` or `open`). Two seeds in one island is an error: draw a stroke between them. A block or open seed the
   street reaches is an error: close the gap with a stroke. An island with no seed is street (lettering in the street).
6. **Pockets** join their block. A thin outline often has pin-holes, and next to cross-hatching a pin-hole lets the
   street fill run into the white cell (or the whole strip between two hatch lines) behind it, so the block's side
   gets a bite of street. A pocket is street paper the street reaches only through a neck narrower than `PINHOLE`
   (3 px): the paper the street reaches through passages at least 3 px wide is found first (from the street seeds),
   and every other piece of street paper is a pocket. Each pocket:
   - holding a street seed stays street (a narrow alley someone marked);
   - touching exactly one area joins it (a hatch cell or strip behind a pin-hole; a crevice between the outline and a
     mark touching it);
   - touching no area, or two or more, stays street (paper among lettering in the street; a narrow lane between two
     blocks).

   Nothing else moves: an inner corner of a block, a wide alley or dead end, and strokes and limits are as without
   this step. `PINHOLE` is a tool setting, checked by eye on the pictures. The cost: a real dead end narrower than
   3 px at some point, lying inside one block and holding no street seed, is taken into the block; the warnings name
   the five largest pockets taken, so a real one can be given a street seed.
7. The **street** label: 1 street, 0 an area, 255 a barrier.
8. The **edge** label: 1 on area px 8-adjacent to anything not the same area, and on stroke px; 255 (no label) on px
   8-adjacent to a limit, on limit px, and on the crop's outermost rows and columns; 0 elsewhere.
9. **Warnings** list each island with no seed that is larger than the smallest seeded one ("possible missed block at
   x, y", page px, a point well inside it), and then always the five largest islands with no seed, whatever was seeded
   ("unseeded island of N px at x, y: a block left unseeded is labelled street"), then the five largest pockets
   that joined a block ("pocket of N px at x, y joined its block: ...").

The 3 px barriers and openings, the 4-connected street, the 8-connected islands and the 2 px an outline's corner may
lie past the box belong to this drawing aid, which is checked by eye on its picture. Nothing like it runs when a sheet
is read.

## Reading the maps

- The line closing a street is a block edge, unless the map's owner has said the street runs through there; then an
  `opening` marks it and the crop's `notes` quote him.
- A road drawn as two lines is a street.
- Open land (land with no outline of its own) is its own kind, `open`, so it is never counted as blocks that failed to
  separate.
- Railway lines, shorelines and contours are neither a block edge nor a zoning border.

The owner corrects these labels on the pictures, not a description of them.
