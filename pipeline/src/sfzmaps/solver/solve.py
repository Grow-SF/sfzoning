"""Solving the city: one energy over every lot's code (energy.py), searched as in search.py.

Held out by quarter: the city is solved four times, each time with wall odds and a placement model fitted without one
quarter, and every lot takes its answer from the solve that did not see its quarter.  The four solves run side by side,
each in its own process (`sfzmaps solver solve-quarter`), with one thread each for the linear algebra (`threads`).

Writes, under runs/ in the run's folder (`runs.solve_dir`: data/solver/solve for the stage's run `final`,
data/solver/experiments/<run> for any other):
  {run}-{q}.npz        the solve that held out quarter q: `names` and `pred` (city.save_run)
  {run}-{q}.log        its log: what its process prints, with no times (the same inputs give the same bytes); its
                       stderr goes to the console
  {run}-{q}-costs.npz  with keep_costs: its label costs, odds, codes and readings (for holes.py)
  {run}.npz            the join: every lot from its own quarter's solve

`solve_and_report` is the `solver-solve` stage (`sfzmaps solver solve final`, a process pinned to one thread like
the quarters'): the placement, the four solves, the join, the data errors among the run's holes and the report, built
in a fresh folder that then takes the run folder's place."""
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from sfzmaps.solver import energy as E
from sfzmaps.solver import paths, runs, search, threads
from sfzmaps.solver.city import city, load_run, run_file, save_run, wall_odds
from sfzmaps.solver.codes import canon, table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.placement import load_options

TRUE_BORDER_ODDS = 6.0      # the ceiling check: the table's own borders as near-certain walls (+) or none (-)
ROUNDS = 4                  # the search's loop of region, compound and expansion moves: rounds at most
COMPOUND_FUSION_ROUNDS = 8  #   and with the compound fusion move in it (solve_joint's compound_fusion)


def costs(n_lots, a, b, odds, rd, options=None, label_weight=1.0, tie_break=1e-4):
    """(U, labels, label terms): U[lot, code], the label costs (energy.label_costs) plus tie_break x each code's
    cheapest-path cost (energy.path_costs), less each lot's row minimum: the costs solve_joint minimises"""
    K = len(rd.names)
    U, n_labels, n_terms = E.label_costs(n_lots, K, rd, options, label_weight)
    U = U + tie_break * E.path_costs(n_lots, K, a, b, odds, rd)
    U -= U.min(1, keepdims=True)
    return U, n_labels, n_terms


def polish(U, a, b, odds, same, p, compound_fusion=False, steps=None):
    """solve_joint's loop from the answer p: region, compound (with compound_fusion, then compound fusion) and
    expansion moves, round after round until the energy stops falling, at most ROUNDS rounds
    (COMPOUND_FUSION_ROUNDS with compound_fusion).  steps: the moves' log, appended to."""
    for _ in range(COMPOUND_FUSION_ROUNDS if compound_fusion else ROUNDS):
        e0 = E.energy(U, a, b, odds, p, same)
        p = search.region_moves(U, a, b, odds, same, p, log=steps)
        p = search.compound_moves(U, a, b, odds, same, p, log=steps)
        if compound_fusion:
            p = search.compound_fusion_moves(U, a, b, odds, same, p, log=steps)
        p = search.expansion_moves(U, a, b, odds, same, p, log=steps)
        if E.energy(U, a, b, odds, p, same) >= e0 - 1e-6:
            break
    return p


def solve_joint(n_lots, a, b, odds, rd, options=None, label_weight=1.0, wall_weight=1.0, same_weight=1.0,
                tie_break=1e-4, table_codes=None, log=print, compound_fusion=False):
    """the lowest-energy code per lot (an index into rd.names).
    tie_break x each code's cheapest-path cost only breaks ties in areas no label stands for; it must stay tiny, as
    it is charged per lot.  table_codes (index per lot, -1 unknown): also report the energy of the table's answer
    (lots the table leaves blank take the solve's codes).  A diagnostic: misread walls and misread or misplaced labels
    charge the table's answer, so it can cost more than the solve's even when the solve is mostly right.
    compound_fusion: the search's loop also runs search.compound_fusion_moves after the compound moves, for up to
    COMPOUND_FUSION_ROUNDS rounds instead of ROUNDS (off in the stage's run: the owner decides when it goes on)."""
    U, n_labels, n_terms = costs(n_lots, a, b, odds, rd, options, label_weight, tie_break)
    odds = wall_weight * odds
    same = same_weight / wall_weight
    steps = []
    p = U.argmin(1)
    e_start = E.energy(U, a, b, odds, p, same)
    if same > 0:                                         # first without the same-code-across-a-line cost (an easy cut)
        p = search.expansion_moves(U, a, b, odds, 0.0, p, log=steps)
    p = search.expansion_moves(U, a, b, odds, same, p, log=steps)
    p = polish(U, a, b, odds, same, p, compound_fusion, steps)
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
                  keep_costs: bool = False, folder: Path | None = None, compound_fusion: bool = False) -> None:
    """Solve the whole city with holdout q's inputs (the walls of `walls_dir`, default WALLS_DIR, and the placement
    in the run's folder) and write runs/{run}-{q}.npz into `folder` (default the run's own,
    `runs.solve_dir`).  The scoring table (with the by-eye corrections) gives the true borders of the ceiling check
    and the table's answer the log prices.  compound_fusion: solve_joint's.  What it prints is the quarter's log: no
    times."""
    runs.check_final(run, walls_dir, true_borders, compound_fusion)
    folder = Path(folder or runs.solve_dir(run))

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
    options = load_options(q, folder, walls_dir)
    p, U = solve_joint(C.n, C.a, C.b, odds, rd, options, table_codes=table_codes, log=log,
                       compound_fusion=compound_fusion)
    names = rd.names + [""]
    save_run(run_file(run, f"-{q}", folder), np.array(names)[p])
    if keep_costs:                                       # for hole causes
        np.savez_compressed(run_file(run, f"-{q}-costs", folder), U=U.astype(np.float32),
                            odds=odds.astype(np.float32), codes=p, names=np.array(rd.names), table_codes=table_codes,
                            lot=rd.lot, code=rd.code, trust=rd.trust)
    log("done")


def join(run: str, folder: Path | None = None) -> None:
    """Every lot takes the answer of the solve that held out its quarter; writes runs/{run}.npz into `folder`
    (default the run's own)."""
    C = city()
    out = np.full(C.n, "", dtype=object)
    for k, q in enumerate(paths.QUARTERS):
        R = np.load(run_file(run, f"-{q}", folder), allow_pickle=True)
        m = C.quarter == k
        out[m] = R["names"].astype(str)[R["pred"]][m]
    save_run(run_file(run, folder=folder), out)


def solve_all(run: str, walls_dir: Path | None = None, true_borders: bool = False, keep_costs: bool = False,
              folder: Path | None = None, compound_fusion: bool = False) -> None:
    """The four holdouts side by side (one `sfzmaps solver solve-quarter` process each, pinned to one thread
    (`threads`), writing into `folder`, default the run's own), then the join.  A quarter's log (runs/RUN-Q.log) is
    what its process prints; what it writes to stderr (a library's warning, a traceback) goes to the console, never
    into the run's folder.  It prints each quarter's log and how long each took (the times only here, never in the
    run's folder).  compound_fusion: solve_joint's, in every quarter."""
    runs.check_final(run, walls_dir, true_borders, compound_fusion)
    folder = Path(folder or runs.solve_dir(run))
    logs = run_file(run, folder=folder).parent
    logs.mkdir(parents=True, exist_ok=True)
    env = threads.environment()
    flags = (["--true-borders"] * true_borders + ["--keep-costs"] * keep_costs
             + ["--compound-fusion"] * compound_fusion
             + (["--walls", str(walls_dir)] if walls_dir else []) + ["--folder", str(folder)])
    t0 = time.perf_counter()
    procs = []
    for q in paths.QUARTERS:
        logf = open(logs / f"{run}-{q}.log", "w")
        cmd = threads.sfzmaps_command() + ["solver", "solve-quarter", run, q] + flags
        procs.append((q, logf, subprocess.Popen(cmd, stdout=logf, env=env)))
    seconds = {}
    while len(seconds) < len(procs):                     # each solve's time, as it ends
        for q, _, p in procs:
            if q not in seconds and p.poll() is not None:
                seconds[q] = time.perf_counter() - t0
        time.sleep(0.5)
    failed = []
    for q, logf, p in procs:
        if p.returncode != 0:
            failed.append(q)
        logf.close()
        print((logs / f"{run}-{q}.log").read_text().rstrip())
    print("solved in " + ", ".join(f"{q} {seconds[q]:.0f} s" for q in paths.QUARTERS), flush=True)
    if failed:
        raise RuntimeError(f"solve failed for {', '.join(failed)}; see {logs}/{run}-<quarter>.log and the errors "
                           "above")
    join(run, folder)
    print(f"joined: {run_file(run, folder=folder)}")


def solve_and_report(run: str, walls_dir: Path | None = None, true_borders: bool = False,
                     compound_fusion: bool = False) -> dict:
    """Everything a solve needs and makes, from the walls of `walls_dir` (default WALLS_DIR), built in a fresh
    folder beside the run's (`runs.fresh_folder`) that then takes its place (`runs.replace_folder`), so nothing of
    an earlier solve is left: fit the placement (placement.npz), solve the four held-out quarters with their costs
    and join them (`solve_all`), list the data errors among the run's holes (`holes.write_data_errors`:
    data_errors.npz and data-errors-2000-digital.csv), and write the report (`report.write_report`: report.json and
    report.md), which scores the run with those parcels left out and with every lot.  compound_fusion: solve_joint's.
    The stage's run (`final`) is refused with other walls, the true borders or the compound fusion
    (`runs.check_final`).  The placement and the report are computed in
    this process: `sfzmaps solver solve` runs it pinned to one thread (`threads`).  Returns the report."""
    import logging

    from sfzmaps.solver import holes, placement, report
    runs.check_final(run, walls_dir, true_borders, compound_fusion)
    target = runs.solve_dir(run)
    threads.warn_unless_pinned(logging.getLogger(__name__), "the placement and the report")
    folder = runs.fresh_folder(target)
    summary = placement.fit(folder, walls_dir)
    solve_all(run, walls_dir, true_borders, keep_costs=True, folder=folder, compound_fusion=compound_fusion)
    rows = holes.causes(run, folder)
    holes.write_data_errors(holes.data_errors(rows), load_run(run, folder), folder)
    found = report.build_report(run, summary, dict(Counter(r["cause"] for r in rows)), walls_dir, folder)
    report.write_report(found, folder)
    runs.replace_folder(folder, target)
    return found
