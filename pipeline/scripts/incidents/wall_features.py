"""Which features make the wall odds on an incident's borders: each pair's log odds split by feature group.

    uv run python scripts/incidents/wall_features.py design [--walls DIR] [--workers 2]
    uv run python scripts/incidents/wall_features.py split ID [--walls DIR]

`design` computes the walls' design matrix (walls.candidate_design) for the candidate features the walls folder's
evaluation.json names (default the stage's walls, runs.WALLS_DIR), refits each quarter's weights as the walls fit
them (walls.fit_quarter on the table as downloaded), checks that they give the folder's saved odds, and keeps X and
the weights in data/solver/experiments/incidents-wall-design/<candidates>/ (X is about 0.8 GB).  CPU only, a few
minutes; `--workers` border-feature processes.

`split` reads the census (census.py) for incident ID and, in the solve of its quarter, splits the log odds of its
pairs (the table lines the solve leaves out, the lines it draws that the table lacks, every pair touching the
incident) into: each border-feature group (wallevidence.columns, alone and times across a street, summed), the lots'
character (area hats) and the context columns (touching kind, street width).  Prints medians and sums per group and
the four longest pairs.  Reads only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from common import CENSUS_DIR, mask
from sfzmaps.solver import paths, runs, walls, wallevidence
from sfzmaps.solver.city import city, run_file

DESIGN_DIR = runs.EXPERIMENTS_DIR / "incidents-wall-design"
CHARACTER = 3 * len(walls.AREA_DIFFERENCE_KNOTS)       # difference, difference x street, the smaller lot's area
KIND, WIDTH = len(walls.KIND_KNOTS), len(walls.STREET_WIDTH_KNOTS)


def candidates_of(walls_dir: Path) -> list[str]:
    return list(json.loads((Path(walls_dir) / "evaluation.json").read_text())["candidates"])


def folder_of(cands) -> Path:
    return DESIGN_DIR / ("+".join(sorted(cands)) or "none")


def column_groups(cands) -> dict[str, list[np.ndarray]]:
    """design column indices per named group (a border-feature group's own and times-street columns together)"""
    options = [c for c in walls.check_candidates(cands) if c != walls.CONTEXT]
    cols = wallevidence.columns(options)
    nF = sum(k for _, k in wallevidence.groups(options))
    out = {name: np.r_[np.arange(s.start, s.stop), nF + np.arange(s.start, s.stop)] for name, s in cols.items()}
    out["lots' character"] = np.arange(2 * nF, 2 * nF + CHARACTER)
    if walls.CONTEXT in cands:
        out["context: touching kind"] = np.arange(2 * nF + CHARACTER, 2 * nF + CHARACTER + KIND)
        out["context: street width"] = np.arange(2 * nF + CHARACTER + KIND, 2 * nF + CHARACTER + KIND + WIDTH)
    return out


def design(walls_dir: Path, workers: int) -> None:
    cands = walls.check_candidates(candidates_of(walls_dir))
    out = folder_of(cands)
    out.mkdir(parents=True, exist_ok=True)
    inputs = walls.load_inputs([cands])
    X = walls.candidate_design(inputs, cands, workers=workers)
    np.save(out / "X.npy", X)
    y = walls.table_lines(inputs.lots)
    weights = {}
    for qi, q in enumerate(paths.QUARTERS):
        w = walls.fit_quarter(X, y, walls.training_pairs(inputs.lots, qi))
        saved = np.load(Path(walls_dir) / f"walls_{q}.npz")["lo_ab"]
        err = float(np.nanmax(np.abs(X @ w - saved)))
        print(f"{q}: refit odds against {walls_dir}: max |difference| {err:.2g}", flush=True)
        if err > 1e-3:
            raise SystemExit(f"{q}: the refit does not give the saved odds")
        weights[q] = w
    np.savez(out / "weights.npz", **weights)
    (out / "source.json").write_text(json.dumps({"walls": str(walls_dir), "candidates": list(cands),
                                                 "columns": int(X.shape[1])}, indent=1) + "\n")
    print(f"wrote {out}")


def split(incident: int, walls_dir: Path) -> None:
    cands = walls.check_candidates(candidates_of(walls_dir))
    folder = folder_of(cands)
    X = np.load(folder / "X.npy", mmap_mode="r")
    W = np.load(folder / "weights.npz")
    groups = column_groups(cands)
    rec = next(t for t in json.loads((CENSUS_DIR / "incidents.json").read_text())["list"] if t["id"] == incident)
    run, q = json.loads((CENSUS_DIR / "incidents.json").read_text())["run"], rec["quarter"]
    C = city()
    a, b = C.a, C.b
    z = np.load(run_file(run, f"-{q}-costs", runs.solve_dir(run)))
    codes = z["codes"]
    w = W[q]
    inI = mask(C.n, rec["lots"])
    P = np.flatnonzero(inI[a] | inI[b])
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    drawn = codes[a] != codes[b]
    blklot = C.lots["blklot"].astype(str).values
    touching = np.load(paths.LOTS_DIR / "lots.npz")["pairs"][:, 2] > 0
    print(f"incident #{incident}: {rec['lots_count']} lots {rec['table']} -> {rec['solved']} ({q}); walls {walls_dir} "
          f"({', '.join(cands) or 'no candidates'})")
    for label, S in (("table lines the solve leaves out", P[tline[P] & ~drawn[P]]),
                     ("lines the solve draws that the table lacks", P[both[P] & ~tline[P] & drawn[P]]),
                     ("every pair touching the incident", P)):
        if not len(S):
            print(f"  {label}: none")
            continue
        x = np.asarray(X[S], float)
        lo = x @ w
        print(f"  {label}: {len(S)} pairs, {C.border_ft[S].sum():.0f} ft; log odds median {np.median(lo):+.2f} "
              f"(p {np.median(expit(lo)):.3f}), sum {lo.sum():+.1f}")
        for name, cols in groups.items():
            part = x[:, cols] @ w[cols]
            print(f"    {name:28} median {np.median(part):+6.2f}  sum {part.sum():+8.1f}")
        for i in S[np.argsort(-C.border_ft[S])][:4]:
            xi = np.asarray(X[i], float)
            parts = ", ".join(f"{name} {xi[cols] @ w[cols]:+.2f}" for name, cols in groups.items()
                              if abs(xi[cols] @ w[cols]) > 0.05)
            print(f"      {blklot[a[i]]}|{blklot[b[i]]} table {C.table[a[i]]}|{C.table[b[i]]} "
                  f"{'touching' if touching[i] else 'street'} {C.border_ft[i]:.0f} ft: log odds {xi @ w:+.2f} = "
                  f"{parts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("--walls", type=Path, default=runs.WALLS_DIR)
    d.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("split")
    s.add_argument("id", type=int)
    s.add_argument("--walls", type=Path, default=runs.WALLS_DIR)
    args = ap.parse_args()
    if args.cmd == "design":
        design(args.walls, args.workers)
    else:
        split(args.id, args.walls)
