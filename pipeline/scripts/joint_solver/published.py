"""The published 2000 map as a run, for comparison: the pipeline's district reader's code per parcel
(data/work/2000-digital-zoning/districts_raw.parquet from a main checkout)."""
import numpy as np

from .codes import canon
from .data import PIPELINE, RUNS, city, save_run

DEFAULT = PIPELINE / "data" / "work" / "2000-digital-zoning" / "districts_raw.parquet"


def save_published(path=DEFAULT, name="published"):
    import pandas as pd
    d = pd.read_parquet(path)
    code = dict(zip(d["blklot"].astype(str), d["code"].fillna("").astype(str)))
    lots = city().lots["blklot"].astype(str).values
    codes = np.array([canon(code[x]) if code.get(x, "") else "" for x in lots])
    print(f"parcels {len(lots):,}; with a published code {int((codes != '').sum()):,}")
    save_run(RUNS / f"{name}.npz", codes)
    return name
