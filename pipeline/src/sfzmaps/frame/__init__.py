"""Map-frame finder (issue #71).

A frozen pretrained backbone's features (`layout`, DINOv2 with small heads) give each page's layout as three chances
per pixel: map, outside the frame (the printed frame, notches and all) and removed block (legend and title blocks,
compass roses and "SECTION N" badges, deleted wherever they sit inside the frame).  The decision (`shape`) fits
straight lines to them: one turned rectangle with per-side tilt and bow and filleted corners, ruled notches and
removed blocks as rectangles or ellipses (`geometry`, `notches`, `blocks`), each line slid onto paper (`paper`); the
per-pixel cut (`decide`) is the fallback for pages of several maps.  `run` orchestrates the held-out run, the final
heads and every page; `scores`, `calibrate` and `look` score and picture it.
Design: docs/superpowers/specs/2026-11-20-frame-crop-design.md (amendment 3)

Constraints that hold everywhere in this package:
- $0: no paid reads, no LLM output as input.
- Inputs are only page.png (as scanned), inputs/frame-labels/ and
  inputs/page_kinds.csv; never the old crops or placements.
- One process for every series: no per-series or per-year branch or threshold.
- No series or scale input to the layout heads.
- Deterministic: seeded, deterministic algorithms, same bits on rerun.
- On mps, run inside sfzmaps.gpu.gpu_turn; at most 4 threads per job.
"""
