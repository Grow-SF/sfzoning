"""The 2000 solver: its inputs, made as `sfzmaps build` stages by the `sfzmaps solver` commands, and the solve.

Modules:
  paths       where the solver's inputs and outputs live
  grid        the raster over the city that the page, the lots and the walls share
  frame       a scanned 2000 page's printed frame, measured
  page        the merged page: every warped sheet's ink on the grid (`sfzmaps solver page`)
  table, lots the April 2000 table and its lots on the grid (`sfzmaps solver lots`)
  prints      every printed zoning code and its saved reads (`sfzmaps solver prints`)
  textnet     the text detector (`sfzmaps solver text Q`)
  trust, streetside, lotsize, strokes
              what the solver knows about the labels and the ink (`sfzmaps solver labels`)
  city        the lots, their neighbours, the table's codes, the wall odds and the runs, as the solve reads them
  labels      each held-out quarter's readings of the labels
  placement   where each label belongs (`sfzmaps solver placement`)
  energy, search, solve
              the energy over every lot's code, its search, and the four held-out solves (`sfzmaps solver solve`)
  borders, measures, holes, published
              scoring a run against the table (`sfzmaps solver score`, `holes`, `published`)
"""
