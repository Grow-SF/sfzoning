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
  printed block in the box is typed as a polygon. Outline crops may be in any set (`train`, `choose` or `test`). A
  corner typed by eye sits within about a pixel of the printed line: the 1960-1980 test pieces, typed this way, have
  90% of their boundary within 3.1 to 3.8 ft of the model's edge, about one pixel.

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
 "notes": "what was ambiguous and how it was decided",
 "drawn_by": "who, how, date"
}
```

Coordinates are `page.png` pixels at full resolution, as stored (x right, y down, the page as scanned, not turned).
`street`, `blocks` and `open` are seed points. `strokes` and `limits` are polylines (two or more points):

- a **stroke** is a printed line: it closes a gap in a printed line, or separates two blocks that share one;
- a **limit** stops the fill where no line is printed (a street running out of the crop into open land, say).

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
- **`choose`**: fit nothing; they give the training trace (scored at each checkpoint; it chooses nothing, the trainer
  keeps its last step) and are where two recipes are compared (`sfzmaps find count` measures them as it does the test
  crops).
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
2. **Strokes and limits** are drawn 3 px wide as barriers; a barrier px is not paper.
3. **Street** is the paper 4-connected to a street seed. A street seed not on paper is an error.
4. **Islands** are the 8-connected pieces of what is neither street nor barrier. An island holding a seed is an
   **area**, numbered in the order its first seed appears in the file (`blocks`, then `open`), of the seed's kind
   (`block` or `open`). Two seeds in one island is an error: draw a stroke between them. A block or open seed the
   street reaches is an error: close the gap with a stroke. An island with no seed is street (lettering in the street).
5. **Sides are straightened.** The scan's ink edge is ragged: a block's side wobbles by a px or two, so the labels
   have teeth and pits along every side. Each area in turn (in area-number order) has its outline replaced by the
   fewest straight sides that stay within `STRAIGHT` (2 px) of it (`cv2.approxPolyDP`, on the outline traced on the
   mask enlarged 4 times so that a straight side keeps its place to the px and a concave corner is not cut); its
   holes are straightened the same way. The straightened shape takes only px that are street now (never another
   area's, a stroke's or a limit's); px it leaves out become street. `STRAIGHT` is a tool setting, checked by eye on
   the pictures. An area that straightening would empty or whose seed it would drop (a strip about 2 px wide or
   less, often a sliver of a block at the crop's side) is kept whole as the fill made it. What it cannot do:
   - lettering that touches a block still counts as the block;
   - a dent deeper than 2 px stays, so the cell-sized bites on cross-hatched blocks (a cell the street reached
     through a hole in the outline) stay street;
   - a spur about 2 px wide or less (a whisker of line) is thinned to a wedge, not removed;
   - a curved side may be flattened by up to 2 px;
   - a street gap narrower than about twice the tolerance between two blocks can be narrowed by the first-numbered
     block's straightened side (a gap of 3 px or more is never closed below 1 px).
6. The **street** label: 1 street, 0 an area, 255 a barrier.
7. The **edge** label: 1 on area px 8-adjacent to anything not the same area, and on stroke px; 255 (no label) on px
   8-adjacent to a limit, on limit px, and on the crop's outermost rows and columns; 0 elsewhere.
8. **Warnings** list each island with no seed that is larger than the smallest seeded one ("possible missed block at
   x, y", page px, a point well inside it), and then always the five largest islands with no seed, whatever was seeded
   ("unseeded island of N px at x, y: a block left unseeded is labelled street").

The 3 px barriers, the 4-connected street, the 8-connected islands, `STRAIGHT` and the 2 px an outline's
corner may lie past the box belong to this drawing aid, which is checked by eye on its picture. Nothing like it runs
when a sheet is read.

## Reading the maps

- The line closing a street is a block edge.
- A road drawn as two lines is a street.
- Open land (land with no outline of its own) is its own kind, `open`, so it is never counted as blocks that failed to
  separate.
- Railway lines, shorelines and contours are not block edges.

The owner corrects these labels on the pictures, not a description of them.
