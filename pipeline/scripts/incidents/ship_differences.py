"""A zoning map published from its series' solve against the map the site shows: the page of differences for the
owner's eye (`sfzmaps.solver.differences`).  $0: no model is asked, and nothing under data/out is written: the new map
is published the stage's way into data/solver/experiments/ship[-<series>]/new-map/.

    uv run python scripts/incidents/ship_differences.py [--series S] [--root DIR] [--published shared|PARCELS.fgb]
                                                       [--page-dir DIR]

--series: the zoning series (default 2000-digital-zoning); --root: its solver's outputs in 2000's layout (default
data/solver for 2000, data/solver/<series> for another); --published: the map to compare against, the site's archive
(data/out/shared, default) or a parcels.fgb; --page-dir: where the page goes (default the experiment's differences/).
After a publish into data/out, `sfzmaps solver publish --series S --page DIR` writes the same page from it."""
import argparse
from pathlib import Path

from sfzmaps.solver import differences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="2000-digital-zoning")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--published", default="shared")
    ap.add_argument("--page-dir", type=Path, default=None)
    args = ap.parse_args()
    print(differences.make(args.series, root=args.root, published=args.published, page_dir=args.page_dir))


if __name__ == "__main__":
    main()
