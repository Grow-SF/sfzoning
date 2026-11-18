# T1 report: the walls without Terra's boxes, on the held-out text chance

Status: DONE_WITH_CONCERNS (kept per ruling; measurably worse than before).

## Implemented
- `walls.DEFAULT_CANDIDATES` = (text_weighted, street_crossing, context). `load_inputs` therefore never builds Terra's letter share (only an experiment without `text_weighted` still does, via `terra_text`).
- `textgrid._chance_files` reads `data/solver/text-reader/<sheet>/chance.png` (each px from the detector that never saw its quarter), constant `textgrid.CHANCE_FILE`.
- `build.py`: `solver-walls` upstream is now (solver-labels, solver-text-reader); reads are the 13 `chance.png` files instead of `terra_boxes.json` and `data/solver/text/chance`. `dvc.yaml` regenerated.
- `scripts/incidents/criteria_check.py`: new `--baseline FOLDER` (compare against a copy of an earlier `final`; use with `--no-pictures`).
- README walls text and log-loss table updated.
- I did NOT touch `paths.py` or `reader_stage.py` (both are code deps of solver-text-reader; editing them would have forced an 85 min rebuild).

Files: src/sfzmaps/solver/{walls,textgrid}.py, src/sfzmaps/build.py, dvc.yaml, dvc.lock, README.md, scripts/incidents/criteria_check.py, tests/{test_solver_walls,test_build}.py.

## Tests
Full suite: 1432 passed, 3 xfailed. New: stage deps carry no `terra_boxes` and no `solver-text` upstream; `load_inputs` with the default candidates never calls `terra_text` (an experiment without text_weighted does); `text_chance` takes each px from the reader's chance and ignores `solver-text`'s (also tied to `reader_stage.CHANCE_FILE`).

## solver-text-reader staleness
Stale only on paper. The stage was never in `dvc.lock` (built by hand 2026-11-16 09:30-09:56, right after commit ce828df, never recorded). Since ce828df no file in its code import set changed except `pyproject.toml`/`uv.lock` (pymaxflow replaced by ortools; uv.lock diff only adds absl-py, immutabledict, ortools, removes pymaxflow). Every data input (data/work, data/warp, lots, prints) is dated 2026-11-15 or earlier, before the build; `inputs/` and saved-reads unchanged. Did `dvc commit --force solver-text-reader` (commit 700f40c).

## Walls held-out log loss (nw / ne / sw / se)
| | nw | ne | sw | se |
|---|---|---|---|---|
| (a) today's walls (street_crossing + context) | .0978 | .1027 | .0452 | .0611 |
| (b) + text_weighted on solver-text's chance (re-measured) | .0982 | .1026 | .0452 | .0612 |
| (c) + text_weighted on the per-quarter chance (final) | .1011 | .1069 | .0487 | .0658 |

(b) reproduces the README tie. (c) is worse in every quarter, and worse than the no-candidate walls (.0978/.1049/.0486/.0635) in ne/sw/se. Each fit took ~5 min.

## Six measures, new final vs final-before (every lot; paired block bootstrap, 1000 draws, 5,113 blocks, `--across-label-sources`)
| Measure | before | after | difference (95%) |
|---|---|---|---|
| Lines drawn | 96.43% | 95.69% | -0.74 pts |
| Drawn that are table lines | 96.65% | 96.13% | -0.52 pts |
| Labels right | 94.79% | 94.05% | -0.74 pts [-1.43, -0.11]; worse beyond noise (table known error 0.48) |
| Holes | 48 | 46 | -2 |
| Incidents | 862 | 921 | +59 [+30.5, +89.1]; not beyond the 220 known error |
| Wrong lots | 4,543 | 7,001 | +2,458 [+1,815, +3,069] |

Criterion 2 verdict: fail (labels right). Energy not comparable (walls changed). Only holes improved.

## Three largest changed areas (before -> after; pictures in data/solver/experiments/t1-walls/pictures/changed-{1,2,3}_lines_window.png)
1. sw, blocks 7161/7169/7145 (641 lots): 584 RH-1 -> P, 57 P -> NC-2. 626 right before, 14 after. The new walls draw lines through the freeway ground and drop the line along the lot edge, so the P of the freeway spills over the RH-1 blocks.
2. se, blocks 4868/4899/4890 (594 lots): 580 RH-1(D) -> P. The old border between P (west) and RH-1(D) is gone in the new walls; only a small loop round the printed "RH-1(D)" letters is drawn, so the whole area takes P.
3. ne, blocks 0212/0221/0213 (589 lots): RM-1 -> CRNC 275, RM-3 -> RM-1 246, RM-4 -> RM-1 63. The table's lines near blocks 221/246/253 (California/Clay) are not drawn; blue lines cut through block interiors instead.
Common cause (my read, not tested): the held-out detector's chance is weaker on the big printed district letters, so letters count as wall ink and real printed borders are discounted less well. Not chased, per ruling.

## DVC actions
`dvc commit --force solver-text-reader` (700f40c); `dvc.yaml` regenerated (`build --write-dvc-yaml`); `pixi run build solver-solve` rebuilt solver-walls and solver-solve (10.7 min); `dvc.lock` committed (1e6b64d); `dvc status` clean for the three stages. Baseline copy of the old solve: `pipeline/data/solver/experiments/text-reader-plan/final-before/` (never committed); scratch run copies `experiments/t1-walls` (new) and `experiments/t1-before` (old, renamed files) exist for criteria_check/rendering and can be deleted.

## Concerns
- Kept per the no-LLM ruling, but the cost is large: wrong lots +2,458 and labels right -0.74 pts (beyond noise). T4's baseline (`final-before`) is the old final; the new final is the new reference for the walls.
- Wrong lots are dominated by three big patches (about 1,800 lots), so the loss is concentrated, not diffuse.
- Options for the owner (not done): weaken the detector's weight in walls, or a letter share from our own boxes (survey option 2).
- criteria_check `--baseline` pictures still draw the stage's final; I rendered the pictures with render_incident directly.
- Commits: 700f40c, 7b61feb, 1e6b64d.
