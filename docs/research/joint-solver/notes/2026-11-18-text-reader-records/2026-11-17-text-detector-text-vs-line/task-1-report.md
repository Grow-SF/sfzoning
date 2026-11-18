# Task 1 report: line rules on exact-truth cases

Status: DONE. Commit d008d0a on `solver`.

## Built
- `pipeline/src/sfzmaps/solver/textnet/linerules.py`: `run_lines`, `_line_kernel`, `RUN_ANGLES`, `run_px` (copies, numpy/cv2 only,
  no `sfzmaps` imports at all, not even `grid`), `line_px`, `chain_lines(ink, length, run)`, `TRUTH_CASES` (8),
  `truth_case(name, text, cap_px, seed, draw=None)`, `rule_accuracy(rule, texts, caps, seeds, draw=None)`.
- `pipeline/tests/test_textnet_linerules.py`: the brief's 3 tests plus a case/seed test and a chain test.
- `data/solver/experiments/textlines/rules.json` (not committed): all four tables below, per case and per text.
- No edits to `textnet/data.py`, `train.py`, `codereader`, or any file in the Global Constraints' no-edit list.

## Choices
- `draw` argument: lets the learned font supply glyphs without `linerules` importing `glyphs` (rule: no `sfzmaps.solver` imports).
  The script that wrote rules.json (scratchpad `mkrules.py`) passes `glyphs.render(font.npz)`; default is `cv2.putText`.
- The chain rule is called as `chain_lines(ink, run_px(f), 27)`: chain extent >= 50 px, runs >= 27 px (q99 cap taken as 27, the larger cap).
  A chain = runs dilated by 3x3 (ends within 2 px), connected components, extent = bounding-box diagonal.
- Hatching: "10 px apart" taken perpendicular (14 px along x). With 10 px along x and 4-6 px lines there is 0-1 px gap, so no glyph px were "far" from a line and the case measured nothing.
- Line width 4 or 6 px by seed parity (through/vertical/corner...); baseline, corner and street cases use 4 px.
- Stair-step: monotone staircase, jogs 20-40 px, through the word.

## TDD
RED: `uv run pytest tests/test_textnet_linerules.py -q` -> `ImportError: cannot import name 'linerules' from 'sfzmaps.solver.textnet'`, 1 error.
GREEN: same command -> `5 passed in 3.05s`.
Before commit: `tests/test_solver_textnet.py tests/test_codereader.py tests/test_build.py tests/test_textnet_linerules.py` -> 142 passed, 3 xfailed
(no `test_textnet_*.py` other than mine exists). Full `uv run pytest -q` -> 1465 passed, 3 xfailed. `pgrep -fl 'sfzmaps (build|stage)'` was empty before commit.

## rule_accuracy: means over 9 texts x 2 caps x 4 seeds; false-line is the glyph px >1 px from the line that the rule calls line
Cell: line recall mean (min over texts) / glyph false-line mean (max over texts, which text).

### Run rule (`line_px`), cv2.putText glyphs
| case | recall mean (min) | glyph false-line mean (max) |
|---|---|---|
| line through the middle | 1.000 (1.000) | 0.001 (0.011 RM-4) |
| line along the baseline | 1.000 (1.000) | 0.000 (0.000) |
| block corner | 1.000 (1.000) | 0.001 (0.004 M-1) |
| two street lines, word between | 1.000 (1.000) | 0.000 (0.000) |
| vertical line through the 2nd letter | 1.000 (1.000) | 0.003 (0.012 RM-2) |
| stair-step border | 0.092 (0.085) | 0.007 (0.020 RH-1(D)) |
| dashed line | 0.016 (0.000) | 0.006 (0.017 RM-4) |
| hatching | 1.000 (1.000) | 0.014 (0.047 NC-1) |

### Run rule, learned font (`font.npz`)
| case | recall mean (min) | glyph false-line mean (max) |
|---|---|---|
| line through the middle | 1.000 (1.000) | 0.012 (0.089 RM-4) |
| line along the baseline | 1.000 (1.000) | 0.000 (0.004 RM-2) |
| block corner | 1.000 (1.000) | 0.000 (0.004 RM-2) |
| two street lines, word between | 1.000 (1.000) | 0.000 (0.000) |
| vertical line through the 2nd letter | 1.000 (1.000) | 0.011 (0.029 NC-1) |
| stair-step border | 0.093 (0.088) | 0.005 (0.013 RM-4) |
| dashed line | 0.024 (0.000) | 0.014 (0.064 RM-4) |
| hatching | 1.000 (1.000) | 0.049 (0.114 RM-4) |

### Chain rule (`chain_lines(ink, run_px(f), 27)`), cv2.putText glyphs
| case | recall mean (min) | glyph false-line mean (max) |
|---|---|---|
| line through the middle | 1.000 (1.000) | 0.207 (0.282 C-3-O) |
| line along the baseline | 1.000 (1.000) | 0.073 (0.147 NC-3) |
| block corner | 1.000 (1.000) | 0.092 (0.179 C-3-O) |
| two street lines, word between | 1.000 (1.000) | 0.317 (0.484 NC-1) |
| vertical line through the 2nd letter | 1.000 (1.000) | 0.026 (0.083 RM-4) |
| stair-step border | 0.958 (0.958) | 0.020 (0.045 RH-1(D)) |
| dashed line | 0.043 (0.000) | 0.052 (0.127 RM-4) |
| hatching | 0.992 (0.992) | 0.514 (0.587 RM-4) |

### Chain rule, learned font
| case | recall mean (min) | glyph false-line mean (max) |
|---|---|---|
| line through the middle | 1.000 (1.000) | 0.302 (0.438 RM-4) |
| line along the baseline | 1.000 (1.000) | 0.282 (0.411 M-1) |
| block corner | 1.000 (1.000) | 0.235 (0.357 NC-1) |
| two street lines, word between | 1.000 (1.000) | 0.267 (0.568 P) |
| vertical line through the 2nd letter | 1.000 (1.000) | 0.239 (0.414 RM-4) |
| stair-step border | 0.958 (0.958) | 0.022 (0.043 RH-1(D)) |
| dashed line | 0.058 (0.000) | 0.134 (0.283 RM-4) |
| hatching | 0.992 (0.992) | 0.654 (0.825 RM-4) |

## Verdicts
- Chain rule: glyph false-line far above 2% on every text in most cases (up to 82%); NOT adopted anywhere, not in `line_px`. It does catch stair steps (96% vs 9%), at the cost of calling letters line.
- Run rule: exact on straight lines through, along or touching words (recall 1.0, false-line <= 1.2% on the main cases with putText, 0.4-8.9% worst text with the learned font). It misses stair-steps (9%) and dashed lines (2%) by design (runs shorter than 50 px).

## Concerns
- Run rule false-line with the learned font reaches 8.9% (RM-4, line through middle) and 11.4% (hatching): above 2% for that text. Letters that line ink touches along a long straight stroke (M, 4) get joined into a run in some placements. Hatching is a dense case.
- Run rule cannot see stair-step district lines or dashed lines; that is for the measure built on the owner's district borders (M1a), as the findings F1/F2 say.
- The learned-font glyph mask is `render(...) > 0`, as in the red team's script; its extent includes faint tails, so "glyph px" there is a slightly generous set.
- (Superseded by fix round 1: the generator is now the committed `pipeline/scripts/textlines/rules.py`.)

## Fix round 1
Changes (second commit, 97ec46b, on `solver` after d008d0a):
1. `pipeline/scripts/textlines/rules.py` (new, committed) writes `data/solver/experiments/textlines/rules.json`: both rules x putText, plus the learned font when `font.npz` exists. The scratchpad `mkrules.py` is deleted.
2. Stair-step: the staircase is placed so the step under the word's centre column runs through the word's middle row (every seed, both caps). New test `test_the_stair_step_steps_through_the_word_on_every_seed` (seeds 0-3, caps 20 and 27: glyph px within 1 px of the line). I wrote it after the fix, so there is no RED run against the old staircase.

Tests: `uv run pytest tests/test_textnet_linerules.py -q` -> 6 passed. Targeted set (solver_textnet, codereader, build, linerules) -> 143 passed, 3 xfailed.
Full `uv run pytest -q` -> 2 failed, 1464 passed. The 2 failures are `tests/test_build.py`'s dvc.yaml check and a textread test, caused by another agent's uncommitted edits in the shared working tree (`dvc.yaml`, `README.md`, `textread/*`, `test_build.py`, `test_solver_textread.py`, all modified). They are not mine, and I committed only my three paths. The full suite was green (1465 passed) before those edits appeared. `pgrep` was empty before the commit.

New numbers, only the stair-step rows changed (recall mean (min) / glyph false-line mean (max)):
| rule / glyphs | recall | false-line |
|---|---|---|
| run / putText | 0.080 (0.074) | 0.030 (0.066) |
| chain / putText | 0.968 (0.966) | 0.256 (0.461) |
| run / font.npz | 0.086 (0.079) | 0.039 (0.098) |
| chain / font.npz | 0.969 (0.967) | 0.336 (0.607) |

The other cases are identical to the tables above. With the staircase really through the word, the earlier stair-step false-line figures (0.7%, 2.0%, 2.0%) were an artifact of the line missing the word: the run rule's false-line there is 3.0-3.9% mean (up to 9.8%), the chain rule's 26-34%.
