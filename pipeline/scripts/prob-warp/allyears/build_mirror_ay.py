"""The all-years mirror data folder (SFZMAPS_DATA_DIR=<scratch>/data_ay): everything linked to the real data, except
each corrected sheet of SERIES: its transform.json (today's fit + "correction": its allyears confit) and its
warped.footprint.geojson (today's footprint carried through the corrected fit).  Nothing in the real data folder
is written.
  build_mirror_ay.py SERIES             every sheet with a correction (as before)
  build_mirror_ay.py SERIES --prepare   only the series folder: every entry a link (run once before per-sheet use)
  build_mirror_ay.py SERIES SHEET       only that sheet (its link replaced by a folder); safe to run for different
                                        sheets at once after --prepare (speed E3)"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.transform import SheetTransform

SERIES = sys.argv[1]
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
AY = Path(__file__).resolve().parent
RUN = Path(os.environ.get("PW_RUN", AY.parents[2] / "data" / "prob-warp" / "run"))
M = Path(os.environ.get("PW_MIRROR", AY.parents[2] / "data" / "prob-warp" / "data_ay"))
sys.path.insert(0, str(AY))
from composed import Composed  # noqa: E402

REAL = config.DATA_DIR
assert "data_ay" not in str(REAL), REAL
SER, MS = REAL / "work" / SERIES, M / "work" / SERIES


def prepare():
    M.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(REAL):
        if name != "work" and not (M / name).exists():
            os.symlink(REAL / name, M / name)
    (M / "work").mkdir(exist_ok=True)
    for name in os.listdir(REAL / "work"):
        if name != SERIES and not os.path.lexists(M / "work" / name):
            os.symlink(REAL / "work" / name, M / "work" / name)
    if MS.is_symlink():
        MS.unlink()
    MS.mkdir(exist_ok=True)
    for name in os.listdir(SER):
        if not os.path.lexists(MS / name):
            os.symlink(SER / name, MS / name)


def one(name):
    src, dst = SER / name, MS / name
    corr = RUN / SERIES / name / f"confit_{name}_p1.npz"
    if not (src.is_dir() and corr.exists()):
        return False
    if dst.is_symlink():
        dst.unlink()
    dst.mkdir(exist_ok=True)
    for f in os.listdir(src):
        if f not in ("transform.json", "warped.footprint.geojson") and not os.path.lexists(dst / f):
            os.symlink(src / f, dst / f)
    TJ = json.loads((src / "transform.json").read_text())
    TJ["correction"] = {"file": str(corr)}
    (dst / "transform.json").write_text(json.dumps(TJ))
    T0 = SheetTransform.from_json(json.loads((src / "transform.json").read_text()))
    Z = np.load(corr)
    C = Composed(T0, Z["Gc"], Z["spacing"], Z["origin"])
    fp = shapely.from_geojson((src / "warped.footprint.geojson").read_text())
    def move(c):
        """A footprint vertex through the corrected fit.  Where today's fit can't be inverted (to_pixel gives NaN: a
        vertex far off the drawn map, 1935 p15) the vertex stays: the correction is zero there, so it wouldn't move."""
        c = np.asarray(c, float)[:, :2]
        out = C.to_world(T0.to_pixel(c))
        bad = ~np.isfinite(out).all(1)
        out[bad] = c[bad]
        return out
    def ring(r):
        """A ring moved and closed again: the batch inverse can give its first and last point (the same input) values a
        last bit apart, and the ring then isn't closed (1935 p15)."""
        out = move(np.asarray(r.coords))
        out[-1] = out[0]
        return out

    def poly(p):
        return shapely.Polygon(ring(p.exterior), [ring(i) for i in p.interiors])
    if fp.geom_type == "Polygon":
        moved = poly(fp)
    else:
        parts = [poly(p) for p in shapely.get_parts(fp) for p in
                 (shapely.get_parts(p) if p.geom_type != "Polygon" else [p]) if p.geom_type == "Polygon"]
        try:
            moved = shapely.union_all(parts)
        except shapely.errors.GEOSException:     # a moved ring crosses itself (1960 13): mend each piece, then join
            moved = shapely.union_all([shapely.make_valid(p) for p in parts])
            print(f"{name}: footprint pieces mended (a moved ring crossed itself)", flush=True)
    if not os.path.lexists(dst / "warped.footprint.geojson") or (dst / "warped.footprint.geojson").is_symlink():
        if (dst / "warped.footprint.geojson").is_symlink():
            (dst / "warped.footprint.geojson").unlink()
        (dst / "warped.footprint.geojson").write_text(shapely.to_geojson(moved))
    print(f"{name}: transform + correction, footprint moved", flush=True)
    return True


if ONLY == "--prepare":
    prepare()
elif ONLY:
    assert MS.is_dir() and not MS.is_symlink(), "run --prepare first"
    one(ONLY)
else:
    prepare()
    for name in sorted(os.listdir(SER)):
        one(name)
print("mirror", M)
