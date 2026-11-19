"""A learned reader for the hatched zoning maps (1921, 1929, 1935, 1948): a small fully convolutional network on the scans
as printed, with a learned street output (today's blocks placed on the scan are its target), six district outputs and
a pooling weight, trained through per-lot pooled losses (spec: docs/superpowers/specs/
2026-11-18-hatched-maps-learned-reader-design.md).

  data   the training rasters (the scan, today's blocks, the street labels from today's centre-line corridors, the
         solver lot of every px) and the lot table (`data`); `--look-labels` draws the street labels' pictures

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchnet data --series 1921

  streets  the network (`model`) and its street output trained on today's blocks placed on the scans; every sheet's
           street probability and blocks, the connected areas it calls not street (`train`); their pictures (`look`)

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchnet streets --series 1921

$0: no API calls."""
