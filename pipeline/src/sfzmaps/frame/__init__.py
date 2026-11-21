"""Learned map-frame finder (issue #71).

One full-resolution network looks at a whole scanned page and gives, per pixel, the chance it is outside the map
area (the printed frame, notches and all); a later step cuts the page where that chance says.  This package holds
the data layer (pages, hand-drawn labels, masks) and the network (`model`); training, the cut decision, scores and
pictures are added on top of them.
Design: docs/superpowers/specs/2026-11-20-frame-crop-design.md

Constraints that hold everywhere in this package:
- $0: no paid reads, no LLM output as input.
- Inputs are only page.png (as scanned), inputs/frame-labels/ and
  inputs/page_kinds.csv; never the old crops or placements.
- One process for every series: no per-series or per-year branch or threshold.
- No series or scale input to the network.
- Deterministic: seeded, deterministic algorithms, same bits on rerun.
- On mps, run inside sfzmaps.gpu.gpu_turn; on cpu use THREADS = 4.
"""
