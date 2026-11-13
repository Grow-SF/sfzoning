# Joint solver (R&D, 2000-digital-zoning)

How it works, how well, where it fails: [`joint-solver.md`](joint-solver.md). Not part of the build; no paid reads.

## Running it

One entry point, run from `pipeline/`:

```
uv run python -m scripts.joint_solver placement               # where each printed label belongs
uv run python -m scripts.joint_solver solve J7 --keep-costs   # four held-out solves side by side, joined and scored
uv run python -m scripts.joint_solver published               # the published 2000 map, scored the same way
uv run python -m scripts.joint_solver score J7 published      # scores (data-error parcels left out)
uv run python -m scripts.joint_solver holes J7 --picture holes.png
```

- `solve RUN --true-borders` is the ceiling check: with the table's own borders as walls, the solve must find
  (nearly) the table's answer.
- `holes RUN --save-data-errors` rewrites `data-errors-2000-digital.csv`.
- Needs `PyMaxflow` (`uv pip install PyMaxflow`; GPL-3.0, which can be combined with AGPL-3.0).

## Files

| File | What it does |
|---|---|
| `__main__.py` | The commands above |
| `data.py` | The data folder, the lots and the April 2000 table's codes |
| `codes.py` | One spelling per zoning code |
| `labels.py` | The printed labels and their readings |
| `placement.py` | Where each label belongs: under its text, or in a box its leader points into |
| `energy.py` | The energy: label costs and wall costs |
| `search.py` | Expansion, region and compound moves |
| `solve.py` | The four held-out solves and the join |
| `borders.py` | The network of lot borders (used by the holes measure) |
| `measures.py` | Borders, labels, holes and incidents |
| `holes.py` | Why each hole is there; the data-error list |
| `published.py` | The published map as a run |

## Data

One folder, `SFZ_RND` (default `pipeline/data/rnd`, git-ignored). `data.py` lists the inputs; they come from the
earlier R&D chain (local branch `bubble-rnd/snapshot-w3`, not pushed). Outputs go to `SFZ_RND/joint/`.

Placement also learns from the hand-checked leader moves: the rows of
`inputs/2000-digital-zoning/label_fixes.csv` with a fix point. The 101 rows behind the results are not committed yet;
`placement --label-fixes PATH` reads them from another copy of the file.
