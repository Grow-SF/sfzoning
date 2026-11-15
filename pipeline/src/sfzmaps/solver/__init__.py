"""The 2000 solver: its inputs, made as `sfzmaps build` stages by the `sfzmaps solver` commands, and the solve.

Modules:
  paths       where the solver's inputs and outputs live
  grid        the raster over the city that the page, the lots and the walls share
  codes       zoning codes in one spelling
  frame       a scanned 2000 page's printed frame, measured
  page        the merged page: every warped sheet's ink on the grid (`sfzmaps solver page`)
  table, lots the April 2000 table and its lots on the grid (`sfzmaps solver lots`)
  saved       the saved vision answers, read by file, sheet or cache key (nothing asks a model)
  prints      every printed zoning code and its saved reads (`sfzmaps solver prints`)
  textnet     the text detector (`sfzmaps solver text Q`) and its scores (`sfzmaps solver text-scores`)
  labels_stage
              what the solver knows about the labels and the ink (`sfzmaps solver labels`):
  trust, streetside, lotsize, strokes
              each reading's trust, the street sides, the lot sizes and the ink's strokes
  textgrid    the detectors' chance of text on the grid, out of fold
  wallevidence, walls
              the wall odds between neighbouring lots (`sfzmaps solver walls`)
  city        the lots, their neighbours, the table's codes, the wall odds and the runs, as the solve reads them
  labels      each held-out quarter's readings of the labels
  placement   where each label belongs (`sfzmaps solver placement`)
  energy, search, solve
              the energy over every lot's code, its search, and the four held-out solves (`sfzmaps solver solve`)
  borders, measures, holes, published
              scoring a run against the table (`sfzmaps solver score`, `holes`, `published`)
  report      a run's six measures against J7's (`sfzmaps solver solve`)
"""
