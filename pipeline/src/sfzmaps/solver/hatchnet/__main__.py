"""python -m sfzmaps.solver.hatchnet <command>: see the package's docstring."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from sfzmaps.solver.hatchnet import __doc__ as doc
    ap = argparse.ArgumentParser(prog="python -m sfzmaps.solver.hatchnet", description=doc.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data", help="the training rasters and the lot table (hatchnet.data)")
    d.add_argument("--series", required=True)
    d.add_argument("--workers", type=int, default=4, help="sheets at once (one maths thread each)")
    d.add_argument("--sheets", nargs="*", help="only these sheets (the draw order still takes every sheet)")
    d.add_argument("--out", type=Path, help="default: data/solver/experiments/hatch-cnn/data/<series>")
    d.add_argument("--look", nargs="+", metavar="SHEET",
                   help="draw the street-target pictures of these written sheets (hatch-cnn/look) and stop")
    a = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")                 # the workers start with one maths thread each
    if a.command == "data":
        from sfzmaps.solver.hatchnet import data
        if a.look:
            for f in data.look_targets(a.series, a.look, a.out):
                print(f)
            return
        s = data.run(a.series, a.out, a.workers, only=a.sheets)
        print(f"{len(s['sheets'])} sheets: " + ", ".join(s["sheets"]))
        for q, v in s["quarters"].items():
            print(f"  {q}: {v['lots']} lots, {v['lots_with_px']} with px ({v['px']} px), {v['lots_read']} read, "
                  f"{v['lots_answered']} answered ({v['lots_answered_random']} random)")
        print(f"  reads per code: {s['reads_per_code']}")
        print(f"  answers per page and stratum: {s['answers_per_page_stratum']}")


if __name__ == "__main__":
    main()
