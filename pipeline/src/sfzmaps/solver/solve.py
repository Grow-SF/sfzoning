"""Solving the city: one energy over every lot's code (energy.py), searched as in search.py.

Held out by quarter: the city is solved four times, each time with wall odds and a placement model fitted without one
quarter, and every lot takes its answer from the solve that did not see its quarter.  The four solves run side by side,
each in its own process (`sfzmaps solver solve-quarter`), with one thread each for the linear algebra.

Writes, under data/solver/solve/runs/:
  {run}-{q}.npz        the solve that held out quarter q: `names` and `pred` (city.save_run)
  {run}-{q}.log        its log
  {run}-{q}-costs.npz  with keep_costs: its label costs, odds, codes and readings (for holes.py)
  {run}.npz            the join: every lot from its own quarter's solve"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sfzmaps.solver import energy as E
from sfzmaps.solver import paths, search
from sfzmaps.solver.city import city, run_file, save_run, wall_odds
from sfzmaps.solver.codes import canon, table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.placement import load_options

TRUE_BORDER_ODDS = 6.0      # the ceiling check: the table's own borders as near-certain walls (+) or none (-)
ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
              "MKL_NUM_THREADS": "1"}


def solve_joint(n_lots, a, b, odds, rd, options=None, label_weight=1.0, wall_weight=1.0, same_weight=1.0,
                tie_break=1e-4, table_codes=None, log=print):
    """the lowest-energy code per lot (an index into rd.names).
    tie_break x each code's cheapest-path cost only breaks ties in areas no label stands for; it must stay tiny, as
    it is charged per lot.  table_codes (index per lot, -1 unknown): also report the energy of the table's answer
    (lots the table leaves blank take the solve's codes).  A diagnostic: misread walls and misread or misplaced labels
    charge the table's answer, so it can cost more than the solve's even when the solve is mostly right."""
    K = len(rd.names)
    U, n_labels, n_terms = E.label_costs(n_lots, K, rd, options, label_weight)
    U = U + tie_break * E.path_costs(n_lots, K, a, b, odds, rd)
    U -= U.min(1, keepdims=True)
    odds = wall_weight * odds
    same = same_weight / wall_weight
    steps = []
    p = U.argmin(1)
    e_start = E.energy(U, a, b, odds, p, same)
    if same > 0:                                         # first without the same-code-across-a-line cost (an easy cut)
        p = search.expansion_moves(U, a, b, odds, 0.0, p, log=steps)
    p = search.expansion_moves(U, a, b, odds, same, p, log=steps)
    for _ in range(4):                                   # whole-region moves and the cut, until nothing improves
        e0 = E.energy(U, a, b, odds, p, same)
        p = search.region_moves(U, a, b, odds, same, p, log=steps)
        p = search.compound_moves(U, a, b, odds, same, p, log=steps)
        p = search.expansion_moves(U, a, b, odds, same, p, log=steps)
        if E.energy(U, a, b, odds, p, same) >= e0 - 1e-6:
            break
    e_solve = E.energy(U, a, b, odds, p, same)
    log(f"labels {n_labels:,}, label terms on {n_terms:,} lots; energy at the start {e_start:,.1f}, "
        f"solved {e_solve:,.1f}")
    log("search steps (kind, round, moved, energy): " + ", ".join(f"{s[0]} {s[1]} {s[2]} {s[3]:,.1f}" for s in steps))
    if table_codes is not None:
        t = np.where(table_codes >= 0, table_codes, p)
        e_table = E.energy(U, a, b, odds, t, same)
        log(f"energy of the table's answer {e_table:,.1f} "
            f"({'lower' if e_table < e_solve else 'higher'} than the solve's)")
    return p, U


def solve_quarter(run: str, q: str, walls_dir: Path | None = None, true_borders: bool = False,
                  keep_costs: bool = False) -> None:
    """Solve the whole city with holdout q's inputs (the walls of `walls_dir`, default WALLS_DIR); writes
    runs/{run}-{q}.npz.  The scoring table (with the by-eye corrections) gives the true borders of the ceiling check
    and the table's answer the log prices."""
    t0 = time.time()

    def log(s):
        print(f"{q}: {s}", flush=True)

    C = city()
    odds = wall_odds(q, walls_dir)
    if len(odds) != len(C.a):
        raise ValueError(f"walls_{q}.npz has {len(odds):,} odds for {len(C.a):,} neighbour pairs: rerun "
                         "`sfzmaps solver walls` after `sfzmaps solver lots`")
    if true_borders:
        both = C.known[C.a] & C.known[C.b]
        odds = np.where(both, np.where(C.table[C.a] != C.table[C.b], TRUE_BORDER_ODDS, -TRUE_BORDER_ODDS), odds)
    rd = readings(q, table_vocabulary(C.lots))
    index = {}
    for i, n in enumerate(rd.names):
        index.setdefault(n, i)
    table_codes = np.array([index.get(canon(c), -1) if k else -1 for c, k in zip(C.table, C.known)])
    log(f"readings {len(rd.lot):,} on {len(np.unique(rd.lot)):,} lots")
    p, U = solve_joint(C.n, C.a, C.b, odds, rd, load_options(q), table_codes=table_codes, log=log)
    names = rd.names + [""]
    save_run(run_file(f"{run}-{q}"), np.array(names)[p])
    if keep_costs:                                       # for hole causes
        np.savez_compressed(run_file(f"{run}-{q}-costs"), U=U.astype(np.float32), odds=odds.astype(np.float32),
                            codes=p, names=np.array(rd.names), table_codes=table_codes, lot=rd.lot, code=rd.code,
                            trust=rd.trust)
    log(f"done in {time.time() - t0:.0f} s")


def join(run: str) -> None:
    """Every lot takes the answer of the solve that held out its quarter; writes runs/{run}.npz."""
    C = city()
    out = np.full(C.n, "", dtype=object)
    for k, q in enumerate(paths.QUARTERS):
        R = np.load(run_file(f"{run}-{q}"), allow_pickle=True)
        m = C.quarter == k
        out[m] = R["names"].astype(str)[R["pred"]][m]
    save_run(run_file(run), out)


def _sfzmaps() -> list[str]:
    """The command that runs sfzmaps in this environment."""
    exe = Path(sys.executable).with_name("sfzmaps")
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "sfzmaps.cli"]


def solve_all(run: str, walls_dir: Path | None = None, true_borders: bool = False, keep_costs: bool = False) -> None:
    """The four holdouts side by side (one `sfzmaps solver solve-quarter` process each), then the join."""
    runs = run_file(run).parent
    runs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **ONE_THREAD)
    flags = (["--true-borders"] * true_borders + ["--keep-costs"] * keep_costs
             + (["--walls", str(walls_dir)] if walls_dir else []))
    procs = []
    for q in paths.QUARTERS:
        logf = open(runs / f"{run}-{q}.log", "w")
        cmd = _sfzmaps() + ["solver", "solve-quarter", run, q] + flags
        procs.append((q, logf, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)))
    failed = []
    for q, logf, p in procs:
        if p.wait() != 0:
            failed.append(q)
        logf.close()
        print((runs / f"{run}-{q}.log").read_text().rstrip())
    if failed:
        raise RuntimeError(f"solve failed for {', '.join(failed)}; see {runs}/{run}-<quarter>.log")
    join(run)
    print(f"joined: {run_file(run)}")
