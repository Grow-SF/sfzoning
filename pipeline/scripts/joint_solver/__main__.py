"""The joint solver's one entry point.  Run from pipeline/:

  uv run python -m scripts.joint_solver placement           fit where each printed label belongs
  uv run python -m scripts.joint_solver solve RUN           solve the four held-out quarters side by side, join, score
  uv run python -m scripts.joint_solver score RUN [RUN ...] score runs against the April 2000 table
  uv run python -m scripts.joint_solver published           the published 2000 map as a run, scored the same way

Data: SFZ_RND (default pipeline/data/rnd); see data.py.  No paid reads."""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="joint_solver", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("placement", help="fit where each printed label belongs (joint/placement.npz)")
    p.add_argument("--label-fixes", help="label_fixes.csv holding the hand-checked leader moves (default: the repo's)")
    p = sub.add_parser("solve", help="solve the four held-out quarters, join them and score the run")
    p.add_argument("run", help="a name for the run (joint/runs/RUN.npz)")
    p.add_argument("--quarter", choices=("nw", "ne", "sw", "se"), help="solve this holdout only (no join)")
    p.add_argument("--true-borders", action="store_true", help="the ceiling: the table's own borders as walls")
    p = sub.add_parser("score", help="score runs against the April 2000 table")
    p.add_argument("runs", nargs="+")
    p = sub.add_parser("published", help="the published 2000 map as the run 'published', scored the same way")
    p.add_argument("--districts", help="districts_raw.parquet of a main checkout (default: this checkout's)")
    args = ap.parse_args(argv)
    os.environ.setdefault("SFZMAPS_DATA_DIR", str(_rnd() / "data_ay2"))

    if args.command == "placement":
        from .data import LABEL_FIXES
        from .placement import fit
        fit(args.label_fixes or LABEL_FIXES)
    elif args.command == "solve":
        from .solve import solve_all, solve_quarter
        if args.quarter:
            solve_quarter(args.run, args.quarter, args.true_borders)
        else:
            solve_all(args.run, args.true_borders)
            _score([args.run])
    elif args.command == "score":
        _score(args.runs)
    elif args.command == "published":
        from .published import DEFAULT, save_published
        _score([save_published(args.districts or DEFAULT)])


def _rnd():
    from .data import RND
    return RND


def _score(runs):
    from .data import load_run
    from .measures import report
    for r in runs:
        report(r, load_run(r))


if __name__ == "__main__":
    sys.exit(main())
