"""The published 2000 map as a run, for comparison: the pipeline's district reader's code per parcel
(data/work/2000-digital-zoning/districts_raw.parquet)."""
import numpy as np

from sfzmaps import config
from sfzmaps.solver import paths
from sfzmaps.solver.city import city, run_file, save_run
from sfzmaps.solver.codes import canon


def save_published(path=None, name="published"):
    """Write the published map (`path`, default data/work/<series>/districts_raw.parquet) as the run `name`."""
    import pandas as pd
    d = pd.read_parquet(path or config.WORK_DIR / paths.SERIES / "districts_raw.parquet")
    code = dict(zip(d["blklot"].astype(str), d["code"].fillna("").astype(str)))
    lots = city().lots["blklot"].astype(str).values
    codes = np.array([canon(code[x]) if code.get(x, "") else "" for x in lots])
    print(f"parcels {len(lots):,}; with a published code {int((codes != '').sum()):,}")
    save_run(run_file(name), codes)
    return name
