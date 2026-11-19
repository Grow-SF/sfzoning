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
    d.add_argument("--look-labels", action="store_true",
                   help="draw the street-label pictures (data.LABEL_WHOLE, data.LABEL_ZOOMS) into "
                        "hatch-cnn/streets-v2/look and stop")
    st = sub.add_parser("streets", help="train the street output; every sheet's streets and blocks (hatchnet.train)")
    st.add_argument("--series", required=True)
    st.add_argument("--device", help="default: mps where available, else cpu")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--steps", type=int, help="default: train.STEPS")
    st.add_argument("--street-loss", choices=("v1", "v2"), default="v2",
                    help="v1: Task 2's cross-entropy on today's blocks; v2: the street labels with the noise model and "
                         "clDice (default)")
    st.add_argument("--inputs", type=int, choices=(1, 2), default=1,
                    help="the network's input channels: 1 the scan only (default), 2 the scan and its ft/px (Task 2's "
                         "and streets v2a's)")
    st.add_argument("--root", type=Path, help="default: data/solver/experiments/hatch-cnn for v1, its streets-v2/ for "
                                              "v2 (models/, streets/, look/)")
    st.add_argument("--look", nargs="*", metavar="SHEET", help="the sheets to draw (default: train.LOOK_SHEETS)")
    st.add_argument("--redraw", action="store_true", help="only draw the pictures again, from the written streets")
    ds = sub.add_parser("districts", help="train the district and weight outputs through pooled lot losses: four folds "
                                          "with answers, four reads-only, all quarters (hatchnet.train)")
    ds.add_argument("--series", required=True)
    ds.add_argument("--device", help="default: mps where available, else cpu")
    ds.add_argument("--seed", type=int, default=0)
    ds.add_argument("--steps", type=int, help="default: train.STEPS")
    ds.add_argument("--init", type=Path, help="the street model to start from (default: train.STREETS_INIT, streets "
                                              "v2c)")
    ds.add_argument("--root", type=Path, help="default: data/solver/experiments/hatch-cnn/districts (models/, look/, "
                                              "summary.json)")
    ds.add_argument("--runs", nargs="+", metavar="NAME",
                    help="only these runs (answers_<q>, reads_<q>, answers_all; default: all nine, train.RUNS)")
    ds.add_argument("--look-only", action="store_true", help="only draw the pictures of the written all-quarters model")
    a = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")                 # the workers start with one maths thread each
    if a.command == "data":
        from sfzmaps.solver.hatchnet import data
        if a.look:
            for f in data.look_targets(a.series, a.look, a.out):
                print(f)
            return
        if a.look_labels:
            for f in data.look_labels(a.series, data=a.out):
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

        from sfzmaps.solver.hatchnet import train
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        root = a.root or train.default_root(a.street_loss)
        sheets = tuple(a.look) if a.look is not None else train.LOOK_SHEETS
        if a.redraw:
            for f in train.redraw(a.series, root, sheets):
                print(f)
            return
        recipe = train.Recipe(street_loss=a.street_loss, inputs=a.inputs, **({"steps": a.steps} if a.steps else {}))
        s = train.run(a.series, root, a.device, a.seed, recipe, sheets)
        h = s["held_out"]
        print(f"device {s['device']}, {s['weights']} weights, {recipe.street_loss} loss, kept step {s['best_step']} "
              f"of {recipe.steps}; {s['seconds']['train']:.0f} s training, {s['seconds']['predict']:.0f} s predicting")
        if s["rates"]:
            print(f"noise rates kept: theta {s['rates']['theta']:.4f}, phi {s['rates']['phi']:.4f} (last step: "
                  f"theta {s['last_rates']['theta']:.4f}, phi {s['last_rates']['phi']:.4f})")
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
    elif a.command == "districts":
        import logging

        from sfzmaps.solver.hatchnet import train
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        if a.look_only:
            for f in train.look_districts(a.series, a.root, device=a.device):
                print(f)
            return
        names = {train.run_name(q, w): (q, w) for q, w in train.RUNS}
        unknown = sorted(set(a.runs or ()) - set(names))
        if unknown:
            ap.error(f"unknown runs {unknown}: one of {sorted(names)}")
        runs = tuple(names[n] for n in a.runs) if a.runs else train.RUNS
        recipe = train.DistrictRecipe(**({"steps": a.steps} if a.steps else {}))
        s = train.run_districts(a.series, a.root, a.init or train.STREETS_INIT, a.device, a.seed, recipe, runs)
        for name, v in s["runs"].items():
            k = v["kept"]
            print(f"{name}: kept step {v['best_step']} of {recipe.steps}, held-out loss {k['val_loss']:.4f} (reads "
                  f"{k['val_read_loss']:.4f}, read agreement {k['val_read_agree']:.4f}); {v['seconds']:.0f} s; "
                  f"lot units per quarter {v['units_per_quarter']}")
        for f in s.get("pictures", []):
            print(f)


if __name__ == "__main__":
    main()
