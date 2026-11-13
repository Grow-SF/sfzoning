"""A mirror of the pipeline's data folder for the corrected-fit runs (SFZMAPS_DATA_DIR=<mirror>): everything linked
to the real data, except each 2000 sheet's transform.json (today's fit + "correction": its confit_NN_p1.npz) and its
warped.footprint.geojson (today's footprint carried through the corrected fit: world -> today's page -> corrected
world).  Nothing in the real data folder is written.  usage: build_mirror.py MIRROR"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.transform import SheetTransform

M = Path(sys.argv[1])
S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
sys.path.insert(0, S)
from composed import Composed  # noqa: E402

REAL = config.DATA_DIR
M.mkdir(parents=True, exist_ok=True)
for name in os.listdir(REAL):
    if name == "work":
        continue
    if not (M / name).exists():
        os.symlink(REAL / name, M / name)
(M / "work").mkdir(exist_ok=True)
for name in os.listdir(REAL / "work"):
    if name != "2000-digital-zoning" and not (M / "work" / name).exists():
        os.symlink(REAL / "work" / name, M / "work" / name)
SER = REAL / "work" / "2000-digital-zoning"
MS = M / "work" / "2000-digital-zoning"
MS.mkdir(exist_ok=True)
for name in os.listdir(SER):
    src = SER / name
    if name.startswith("2000-digital-zoning-p") and src.is_dir():
        nn = name[-2:]
        dst = MS / name
        dst.mkdir(exist_ok=True)
        for f in os.listdir(src):
            if f in ("transform.json", "warped.footprint.geojson"):
                continue
            if not (dst / f).exists():
                os.symlink(src / f, dst / f)
        TJ = json.loads((src / "transform.json").read_text())
        corr = S + f"confit_{nn}_p1.npz"
        TJ["correction"] = {"file": corr}
        (dst / "transform.json").write_text(json.dumps(TJ))
        T0 = SheetTransform.from_json(json.loads((src / "transform.json").read_text()))
        Z = np.load(corr)
        C = Composed(T0, Z["Gc"], Z["spacing"], Z["origin"])
        fp = shapely.from_geojson((src / "warped.footprint.geojson").read_text())

        def move(coords):
            c = np.asarray(coords, float)[:, :2]
            return C.to_world(T0.to_pixel(c))
        moved = shapely.transform(fp, lambda c: move(c))
        (dst / "warped.footprint.geojson").write_text(shapely.to_geojson(moved))
        print(f"p{nn}: transform + correction, footprint moved", flush=True)
    elif not (MS / name).exists():
        os.symlink(src, MS / name)
print("mirror", M)
