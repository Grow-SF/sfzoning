# merged-page

What the merged page (`uv run sfzmaps solver page`) leaves out of each 2000 zoning sheet besides everything outside
its printed frame.

- `2000-digital-zoning.json`: per sheet, the cut boxes: page furniture printed inside the frame (notes and the north
  arrow), not map. `box` is `[x0, y0, x1, y1]` in px of the sheet's `page.png`, both ends inside the box; `what` says
  what it covers. Each box is the bounding box of the ink pieces lying wholly inside a corner region picked by eye on
  the scan, plus 10 px; map lines that run out of the region are never cut.
