# Hand crops for the street-and-block finder

Hand-made labels on crops of the scans, the only teacher of the street-and-block finder (`sfzmaps.find`) for every
map before the 1980s. A person marks a few seed points and strokes on a crop; `sfzmaps.find.handlabels` makes the
per-pixel labels from them. Design: `docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md`.

One file per crop: `<set>/<series>/<sheet>__<x0>_<y0>_<w>_<h>.json`, the sheet named as in
`data/work/<series>/<sheet>/page.png`, the name's numbers the same as the file's `box`.

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

Check every crop by eye: `uv run sfzmaps find crops --look` (in `pipeline/`) draws the scan and the labels side by side,
twice size, with page-pixel rulers to read coordinates off, under `data/find/look/crops/<set>/`, and prints each crop's
warnings and its areas by kind.

## The three sets

- **`test`**: drawn first; used for nothing but the pictures and counts shown to the owner.
- **`choose`**: used only to choose the training step kept and the one setting of the area step.
- **`train`**: the only hand labels the model learns from; added to wherever the pictures show misses.

The ground under every `test` and `choose` crop is left out of all training, in every year's scan of that ground
(`handlabels.reserved`): 1921 and 1929 draw the same streets, so a training crop on one would give away a test crop on
the other. A crop must lie on a sheet that has a placement, and never on a duplicate scan.

## How the labels are made

In this order:

1. **Paper** is the crop's px lighter than its Otsu threshold.
2. **Strokes and limits** are drawn 3 px wide as barriers; a barrier px is not paper.
3. **Street** is the paper 4-connected to a street seed. A street seed not on paper is an error.
4. **Islands** are the 8-connected pieces of what is neither street nor barrier. An island holding a seed is an
   **area**, numbered in the order its first seed appears in the file (`blocks`, then `open`), of the seed's kind
   (`block` or `open`). Two seeds in one island is an error: draw a stroke between them. A block or open seed the
   street reaches is an error: close the gap with a stroke. An island with no seed is street (lettering in the street).
5. The **street** label: 1 street, 0 an area, 255 a barrier.
6. The **edge** label: 1 on area px 8-adjacent to anything not the same area, and on stroke px; 255 (no label) on px
   8-adjacent to a limit, on limit px, and on the crop's outermost rows and columns; 0 elsewhere.
7. **Warnings** list each island with no seed that is larger than the smallest seeded one ("possible missed block at
   x, y", page px, a point well inside it).

The 3 px barriers, the 4-connected street and the 8-connected islands belong to this drawing aid, which is checked by
eye on its picture. Nothing like it runs when a sheet is read.

## Reading the maps

- The line closing a street is a block edge.
- A road drawn as two lines is a street.
- Open land (land with no outline of its own) is its own kind, `open`, so it is never counted as blocks that failed to
  separate.
- Railway lines, shorelines and contours are neither a block edge nor a zoning border.

The owner corrects these labels on the pictures, not a description of them.
