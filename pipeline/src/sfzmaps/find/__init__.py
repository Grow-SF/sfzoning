"""Street-and-block finder.

One model looks at a scanned historic zoning map and says, per spot, two things: street or not, block edge or not.  It
does not look for zoning borders.  Its labels come only from what matches the print: modern lots for the series that
mark them (`inputs/find/label_sources.csv`) and hand crops for the rest.
Design: docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md

Constraints that hold everywhere in this package:
- One model, shown only the scan and the frame cut: no placement, lots, saved reads, series name, scale or "which
  way is up".
- Never train on a label that might not match the print; the ground under test and the choosing crops is left out of
  all training, in every year's scan.
- No hand-set cutoff or window in the reading, no per-series path, no erasing or cleaning of ink.
- No LLM output is read.  $0.
- Deterministic: fixed seeds, deterministic algorithms, the device recorded; GPU work only inside
  sfzmaps.gpu.gpu_turn.
- When a sheet or crop is read, its scan is read at its own resolution, as stored, with no resampling code of our
  own (training tiles are resized, `data.resized`, to learn every print size).  A duplicate scan is never read.
- Outputs go under data/find/, with the series in every path.

The placement side (`trial`) tries the finder's blocks in placement, in separate data roots under data/find/trial/
that never touch today's data.  It reads placement outputs (control points, the placement check, the warp's scores)
only to report and compare them, and copies `labels.json` with a sheet's other work files without ever opening it.
Nothing it reads or writes reaches the model.
"""
