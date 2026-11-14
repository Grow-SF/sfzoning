"""What every step shares.  Each step runs for one sheet: env SERIES (the year's folder name in data/work) and SHEET
(the sheet's folder name).
  S   the sheet's fit folder, data/sheet-warp/fits/<year>/<sheet>/ (or under WARP_FITS), with a trailing slash
  D   the sheet's folder in the pipeline's data/work: read, never written
Only light imports at the top: the two fits run in the PyTorch environment, which has no sfzmaps."""
import json
import os
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[3]
FITS = Path(os.environ.get("WARP_FITS", PIPELINE / "data" / "sheet-warp" / "fits"))
SERIES, SHEET = os.environ["SERIES"], os.environ["SHEET"]
OUT = FITS / SERIES / SHEET
OUT.mkdir(parents=True, exist_ok=True)
S = str(OUT) + "/"
D = PIPELINE / "data" / "work" / SERIES / SHEET


def year():
    """the year of the lots the sheet is fitted to"""
    from sfzmaps.districts import read
    return read.series_year(SERIES)


def lots_and_area(cuts=2):
    """the year's lots, and the area the warp measures: the sheet's drawn map (its warped footprint) less the Presidio
    (block 1300, and 8 ft round it).  The block-side steps cut the Presidio out twice and the extra evidence once: each
    cut rounds the outline's corners a little differently, and every fit was made this way."""
    import shapely
    from sfzmaps.districts import read
    lots = read.load_parcels(year=year())
    presidio = lots[lots["block_num"] == "1300"].union_all().buffer(8.0)
    area = shapely.from_geojson((D / "warped.footprint.geojson").read_text())
    for _ in range(cuts):
        area = area.difference(presidio)
    return lots, area


def side_pixels(T0, SD):
    """Today's placement (to_pixel) of the block-side points and their tangents' ends, computed once per sheet and kept
    in block_side_points_on_page.npz.  Always the same call on the same batch: to_pixel's iteration stops on the
    batch's largest step, so placing the points in other batches would change the values slightly."""
    import numpy as np
    f = S + "block_side_points_on_page.npz"
    if not os.path.exists(f):
        np.savez(f, pts=T0.to_pixel(SD["pts"]), ta=T0.to_pixel(SD["ta"]), tb=T0.to_pixel(SD["tb"]))
    Z = np.load(f)
    return Z["pts"], Z["ta"], Z["tb"]


def _kept(key, compute):
    """a number measured once per sheet and kept in scan_measures.json"""
    f = OUT / "scan_measures.json"
    s = json.loads(f.read_text()) if f.exists() else {}
    if key not in s:
        s[key] = compute()
        f.write_text(json.dumps(s, indent=1))
    return s[key]


def ftpx():
    """the scan's scale: map feet per page pixel (today's placement, median over the page)"""
    def compute():
        import cv2
        from sfzmaps.transform import SheetTransform
        h, w = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE).shape
        t = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
        return float(t.median_scale_ft_per_px((h, w)))
    return _kept("ft_per_px", compute)


def half(ink):
    """the block lines' half width (px): the 75th percentile of the ink's half width along its centre line, where
    today's placement puts the block outlines (blocks_px.geojson, within 1 px).  Wider ink is heavier than a block line."""
    def compute():
        import cv2
        import numpy as np
        import shapely
        from scipy import ndimage
        from skimage.morphology import skeletonize
        m = np.zeros(ink.shape, np.uint8)
        for f in json.loads((D / "blocks_px.geojson").read_text())["features"]:
            for q in shapely.get_parts(shapely.geometry.shape(f["geometry"])):
                if q.geom_type == "Polygon":
                    cv2.polylines(m, [np.round(np.asarray(q.exterior.coords)[:, :2]).astype(np.int32)], True, 1, 3)
        hw = ndimage.distance_transform_edt(ink)[skeletonize(ink) & (m > 0)]
        return float(np.percentile(hw, 75)) if len(hw) else 2.0
    return _kept("block_line_half_width_px", compute)
