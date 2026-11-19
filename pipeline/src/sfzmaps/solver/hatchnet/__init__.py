"""A learned reader for the hatched zoning maps (1921, 1929, 1935, 1948): a small fully convolutional network on the scans
as printed, with a learned street output (today's blocks placed on the scan are its target), six district outputs and
a pooling weight, trained through per-lot pooled losses (spec: docs/superpowers/specs/
2026-11-18-hatched-maps-learned-reader-design.md).

  data   the training rasters (the scan, today's blocks, the street labels from today's centre-line corridors, the
         solver lot of every px) and the lot table (`data`); `--look-labels` draws the street labels' pictures

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchnet data --series 1921

  streets  the network (`model`) and its street output, trained (`train`) on the street labels with a label-noise
           model and the clDice terms (`topo`; `--street-loss v2`, the default, into hatch-cnn/streets-v2), or on
           today's blocks placed on the scans (`--street-loss v1`, into hatch-cnn); every sheet's street probability
           and blocks, the connected areas it calls not street; their pictures (`look`)

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchnet streets --series 1921

  districts  the six district outputs and the pooling weight, trained through pooled lot losses (the reader's reads
             through a read-noise matrix, the owner's answers, the legend swatches; each lot once), from the streets
             v2c model and with no street loss (ruling R5): the four folds by quarter with answers, the same four
             reads-only and the all-quarters model, one GPU turn at a time, into hatch-cnn/districts (models/,
             summary.json); then the all-quarters model's per-px pictures (look/)

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchnet districts --series 1921

$0: no API calls."""
