"""Two reads of a series scored against the April 2000 table with the solve's measures (report.six_measures on every
lot, each read reaching the table's lots by lot ID, as `ship_differences` scores the published 2000 map): for the
1996-2000 scan, which is the same drawing as the 2000 one, before and after a re-read.

    uv run python scripts/incidents/twin_measures.py BEFORE_districts_raw.parquet AFTER_districts_raw.parquet

Saves each as a run under data/solver/experiments (twin-before, twin-after); prints the measures.
"""
import json
import sys

import numpy as np

from sfzmaps.solver.city import city, load_run
from sfzmaps.solver.published import save_published
from sfzmaps.solver.report import six_measures

KEYS = ("wrong_lots", "labels_right", "incidents", "lines_drawn", "drawn_are_table_lines", "holes")

if __name__ == "__main__":
    before, after = sys.argv[1:3]
    nothing = np.zeros(city().n, bool)
    out = {}
    for name, path in (("twin-before", before), ("twin-after", after)):
        save_published(path, name=name)
        m = six_measures(load_run(name), nothing)
        out[name] = {k: m[k] for k in KEYS}
    print(json.dumps(out, indent=1))
