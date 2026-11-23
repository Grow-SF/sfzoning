"""Street-and-block finder.

One model looks at a scanned historic zoning map and says, per spot, three things: street or not, block edge or not,
zoning border or not.  Its labels come only from what matches the print: modern lots for the series that mark them
(`inputs/find/label_sources.csv`), the April 2000 table for the three `2000-digital` series' zoning borders, "no zoning
border" everywhere on the hatched 1921-1948 maps, and hand crops for the rest.
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
- Scans are read at their own resolution, as stored.  A duplicate scan is never read.
- Outputs go under data/find/, with the series in every path.
"""
