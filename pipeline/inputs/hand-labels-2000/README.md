# The owner's hand labels for the 2000 map

Decisions a person made, kept as input: 329 labels of the 2000 digital zoning map checked by hand (2026-11-14), in
the format of `inputs/<series>/label_fixes.csv` (`sheet,x,y,code,fix_code,fix_x,fix_y,note`), 101 of them leader
moves (a fix point where the label's leader line points).

| File | What |
|---|---|
| `label_fixes.csv` | The 329 rows. |

Used by: the 2000 solver alone. Its placement learns the leader moves from the rows with a fix point
(`solver.placement.leader_moves`, `solver.runs.LABEL_FIXES`; stage `solver-solve`).

Not read by the old district reader. That reader applies `inputs/<series>/label_fixes.csv`, and through its twin fuse
the 2000 sheets' fixes also reach the 1996-2000 scan. These rows were written for the solver's labels, and applied
there they made the 1996-2000 map worse against the April 2000 table (38,507 → 39,034 wrong lots). They moved here
from `inputs/2000-digital-zoning/label_fixes.csv` on 2026-11-18, which keeps its 19 rows of 2026-11-04.
