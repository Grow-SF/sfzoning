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
    st = sub.add_parser("streets", help="train the street output; every sheet's streets and blocks (hatchnet.train)")
    st.add_argument("--series", required=True)
    st.add_argument("--device", help="default: mps where available, else cpu")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--steps", type=int, help="default: train.STEPS")
    st.add_argument("--root", type=Path, help="default: data/solver/experiments/hatch-cnn (models/, streets/, look/)")
    st.add_argument("--look", nargs="*", metavar="SHEET", help="the sheets to draw (default: train.LOOK_SHEETS)")
    st.add_argument("--redraw", action="store_true", help="only draw the pictures again, from the written streets")
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
    elif a.command == "streets":
        import logging

        from sfzmaps.solver.hatchnet import data, train
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        root = a.root or data.EXPERIMENT
        sheets = tuple(a.look) if a.look is not None else train.LOOK_SHEETS
        if a.redraw:
            for f in train.redraw(a.series, root, sheets):
                print(f)
            return
        recipe = train.Recipe(**({"steps": a.steps} if a.steps else {}))
        s = train.run(a.series, root, a.device, a.seed, recipe, sheets)
        h = s["held_out"]
        print(f"device {s['device']}, {s['weights']} weights, kept step {s['best_step']} of {recipe.steps}; "
              f"{s['seconds']['train']:.0f} s training, {s['seconds']['predict']:.0f} s predicting")
        print(f"held out: {h['px']} px, agreement {h['agree']:.4f} (all one class: {h['majority_agree']:.4f}), "
              f"BCE {h['bce']:.4f} (constant prior {h['constant_prior_bce']:.4f})")
        z, t = s["last_step"]["held_out"], s["last_step"]["trained"]
        print(f"the last step's weights: held-out agreement {z['agree']:.4f}, BCE {z['bce']:.4f}; trained areas "
              f"{t['agree']:.4f}, BCE {t['bce']:.4f} (kept: {s['trained']['agree']:.4f}, {s['trained']['bce']:.4f})")
        for sheet, v in s["sheets"].items():
            print(f"  {sheet}: {v['blocks']['count']} blocks (target {v['blocks']['target_blocks']}), held-out "
                  f"agreement {v['held_out']['agree']}")
        for f in s["pictures"]:
            print(f)


if __name__ == "__main__":
    main()
