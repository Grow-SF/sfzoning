"""The sheet warp: a per-sheet correction of where each scanned map sheet is placed.

The pipeline places every scan with a thin-plate-spline transform fitted to matched block corners.  That placement
is good but a few feet off in places.  The sheet warp lines today's placement up with the printed block lines: it
fits a small displacement field (a `sfzmaps.transform.Correction`) that moves the modern block sides onto the ink
of the scan, then warps the scan with it.  Nothing is traced from the scan and nothing is drawn that is not in it.

Modules:
  sheet     one placed sheet: its files, scan, placement, year's lots and the area the warp measures
  targets   the block-side points of a sheet, the maps of distance to the printed block lines, and how well a
            placement (with or without a correction) fits them
"""
