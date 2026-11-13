"""Shared settings for the all-years runs of the correction chain.  Each script is the 2000 script with its
2000-only constants taken from here instead (env SERIES, SHEET = the sheet's folder name):
  NN      the sheet's folder name (file tag)
  S       this sheet's output folder (allyears/run/<series>/<sheet>/), trailing slash
  D       the sheet's real work folder (inputs; never written)
  YEAR    the series' lot year (read.series_year)
  PART()  the area measured: the sheet's drawn map (its warped footprint; 2000 used its city part) less the Presidio
  FTPX    the scan's ft per px (today's fit, median)
  HALF    block-line weight: the 75th percentile of the skeleton half-width (px) of the ink lying where today's fit
          puts the block outlines (blocks_px.geojson, within 1 px).  On the 2000 sheets this gives 2.0, the setting
          the 2000 run used.
Light imports only at module level: confit6 runs in the torch env, which has no sfzmaps."""
import json
import os
import sys
from pathlib import Path

SCR = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
AY = SCR / "allyears"
SERIES, NN = os.environ["SERIES"], os.environ["SHEET"]
OUT = Path(os.environ["AY_OUT"]) if os.environ.get("AY_OUT") else AY / "run" / SERIES / NN   # AY_OUT: profiling runs
OUT.mkdir(parents=True, exist_ok=True)
S = str(OUT) + "/"
REAL = Path("/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/work")
D = REAL / SERIES / NN
sys.path.insert(0, str(AY))
_SET = OUT / "settings.json"
if int(os.environ.get("RBF_THREADS", "1")) > 1:      # speed E7: today's fit's thin-plate spline on threads, bit-identical
    import rbfthreads
    rbfthreads.install()


def settings():
    return json.loads(_SET.read_text()) if _SET.exists() else {}


def save_setting(**kw):
    s = settings()
    s.update(kw)
    _SET.write_text(json.dumps(s, indent=1))


def side_pixels(T0, SD):
    """Today's fit's to_pixel of the block-side points (dm_sides: pts, ta, tb), computed once per sheet and kept in
    pix_NN.npz (speed E4).  Each array is the same call on the same batch as before, so the values are bit-identical
    (to_pixel's iteration stops on the batch's largest step: never split or merge these batches)."""
    import numpy as np
    f = S + f"pix_{NN}.npz"
    if not os.path.exists(f):
        np.savez(f, pts=T0.to_pixel(SD["pts"]), ta=T0.to_pixel(SD["ta"]), tb=T0.to_pixel(SD["tb"]))
    Z = np.load(f)
    return Z["pts"], Z["ta"], Z["tb"]


def year():
    from sfzmaps.districts import read
    return read.series_year(SERIES)


def PART():
    import shapely
    from sfzmaps.districts import read
    fp = shapely.from_geojson((D / "warped.footprint.geojson").read_text())
    if os.environ.get("SAME2000"):          # check run: the 2000 run's own area (its city part), to compare outputs
        fp = shapely.from_wkt(json.load(open(SCR / "stepview/city_m2/parts.json"))[NN[-2:]])
    lots = read.load_parcels(year=year())
    return fp.difference(lots[lots["block_num"] == "1300"].union_all().buffer(8.0))


def ftpx():
    if os.environ.get("SAME2000"):          # check run: the 2000 scripts' own constant (the measured scale is 4.006-4.038
        save_setting(ftpx=3.99)             # on p06, p08, p13, which moves the held rule's near-end cut by one point)
        return 3.99
    s = settings()
    if "ftpx" not in s:
        import cv2
        from sfzmaps.transform import SheetTransform
        h, w = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE).shape
        t = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
        save_setting(ftpx=float(t.median_scale_ft_per_px((h, w))))
        s = settings()
    return s["ftpx"]


def half(ink=None):
    s = settings()
    if "half" not in s:
        import cv2
        import numpy as np
        import shapely
        from scipy import ndimage
        from skimage.morphology import skeletonize
        if ink is None:
            ink = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE) < 128
        m = np.zeros(ink.shape, np.uint8)
        for f in json.loads((D / "blocks_px.geojson").read_text())["features"]:
            for q in shapely.get_parts(shapely.geometry.shape(f["geometry"])):
                if q.geom_type == "Polygon":
                    cv2.polylines(m, [np.round(np.asarray(q.exterior.coords)[:, :2]).astype(np.int32)], True, 1, 3)
        hw = ndimage.distance_transform_edt(ink)[skeletonize(ink) & (m > 0)]
        save_setting(half=float(np.percentile(hw, 75)) if len(hw) else 2.0, half_n=int(len(hw)))
        s = settings()
    return s["half"]
