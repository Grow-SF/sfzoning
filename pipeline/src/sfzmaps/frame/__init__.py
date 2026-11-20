"""Learned map-frame finder (issue #71).

A U-Net looks at one scanned page and predicts which pixels belong to the map
area (inside the printed frame, notches and all), replacing the hand-built
frame crops. This package holds the data layer (pages, hand-drawn labels,
masks), and later the training, prediction and evaluation code.
Design: docs/superpowers/specs/2026-11-19-learned-frame-design.md

Constraints that hold everywhere in this package:
- $0: no paid reads, no LLM output as input.
- Inputs are only page.png (as scanned), inputs/frame-labels/ and
  inputs/page_kinds.csv; never the old crops or placements.
- One process for every series: no per-series or per-year branch or threshold.
- No series or scale input to the network.
- Deterministic: seeded, deterministic algorithms, same bits on rerun.
- On mps, run inside sfzmaps.gpu.gpu_turn; on cpu use THREADS = 4.
"""
