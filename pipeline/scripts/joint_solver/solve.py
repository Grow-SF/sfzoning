"""Solving the city: one energy over every lot's code (energy.py), searched as in search.py.

Held out by quarter: the city is solved four times, each time with wall odds and a placement model fitted without one
quarter, and every lot takes its answer from the solve that did not see its quarter."""
import sys
import time

import numpy as np

from . import energy as E
from . import search
from .codes import canon
from .data import QUARTERS, RUNS, city, save_run, wall_odds
from .labels import readings
from .placement import load_options


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


def solve_quarter(run, q, true_borders=False):
    """solve the whole city with holdout q's inputs; writes runs/{run}-{q}.npz"""
    t0 = time.time()

    def log(s):
        print(f"{q}: {s}", flush=True)

    C = city()
    odds = wall_odds(q)
    if true_borders:                                     # the ceiling: the table's own borders as near-certain walls
        both = C.known[C.a] & C.known[C.b]
        odds = np.where(both, np.where(C.table[C.a] != C.table[C.b], 6.0, -6.0), odds)
    rd = readings(q)
    index = {}
    for i, n in enumerate(rd.names):
        index.setdefault(n, i)
    table_codes = np.array([index.get(canon(c), -1) if k else -1 for c, k in zip(C.table, C.known)])
    log(f"readings {len(rd.lot):,} on {len(np.unique(rd.lot)):,} lots")
    p, U = solve_joint(C.n, C.a, C.b, odds, rd, load_options(q), table_codes=table_codes, log=log)
    names = rd.names + [""]
    save_run(RUNS / f"{run}-{q}.npz", np.array(names)[p])
    log(f"done in {time.time() - t0:.0f} s")


def join(run):
    """every lot takes the answer of the solve that held out its quarter; writes runs/{run}.npz"""
    C = city()
    out = np.full(C.n, "", dtype=object)
    for k, q in enumerate(QUARTERS):
        R = np.load(RUNS / f"{run}-{q}.npz", allow_pickle=True)
        m = C.quarter == k
        out[m] = R["names"].astype(str)[R["pred"]][m]
    save_run(RUNS / f"{run}.npz", out)


def solve_all(run, true_borders=False):
    """the four holdouts side by side (one process each), then join"""
    import os
    import subprocess
    RUNS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               MKL_NUM_THREADS="1")
    flags = ["--true-borders"] * true_borders
    procs = []
    for q in QUARTERS:
        logf = open(RUNS / f"{run}-{q}.log", "w")
        cmd = [sys.executable, "-m", __package__, "solve", run, "--quarter", q] + flags
        procs.append((q, logf, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)))
    failed = []
    for q, logf, p in procs:
        if p.wait() != 0:
            failed.append(q)
        logf.close()
        print((RUNS / f"{run}-{q}.log").read_text().rstrip())
    if failed:
        sys.exit(f"solve failed for {', '.join(failed)}; see {RUNS}/{run}-<quarter>.log")
    join(run)
    print(f"joined: {RUNS / (run + '.npz')}")
