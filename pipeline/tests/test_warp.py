import cv2
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from sfzmaps import warp
from sfzmaps.reading import Box
from sfzmaps.transform import GCP, SheetTransform
from synthetic import make_sheet

PX_PER_FT = 0.55


def _blocks():
    rows, geoms = [], []
    for i in range(5):
        for j in range(5):
            rows.append(f"{i * 5 + j + 1:04d}")
            x, y = 6_005_000.0 + j * 460, 2_115_000.0 + i * 330
            geoms.append(box(x, y, x + 400, y + 270))
    return gpd.GeoDataFrame({"block_num": rows}, geometry=geoms, crs="EPSG:2227")


def _stretch(p):
    a = 8.0
    x, y = p[..., 0], p[..., 1]
    return np.stack([x + a * np.sin(x / 700 + 0.3) * np.cos(y / 900),
                     y + a * np.cos(x / 800) * np.sin(y / 600 + 1.0)], axis=-1)


@pytest.fixture(scope="module")
def warped(tmp_path_factory):
    d = tmp_path_factory.mktemp("warp")
    sheet = make_sheet(_blocks(), px_per_ft=PX_PER_FT, rotation_deg=3.0, distortion=_stretch,
                       style="filled", seed=1)
    h, w = sheet.img.shape
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 8), np.linspace(0, h - 1, 8))
    P = np.c_[xs.ravel(), ys.ravel()]
    W = sheet.true_px_to_world(P)
    t = SheetTransform([GCP(*p, *q, "auto") for p, q in zip(P, W)], smoothing=1.0).fit()
    png = d / "page.png"
    cv2.imwrite(str(png), sheet.img)
    box_ = Box(x0=0, y0=0, x1=300, y1=250)
    sheet.img[0:250, 0:300] = 0  # ink under the mask box to prove masking, not paper
    cv2.imwrite(str(png), sheet.img)
    out = warp.warp_sheet(png, t, [box_], d / "out.tif")
    return sheet, t, out


def test_warp_places_pixel_correctly(warped):
    sheet, t, out = warped
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 2227
        assert ds.count == 4
        res = ds.res[0]
        alpha = ds.read(4)
        for g in _blocks().geometry[7:18:5]:  # interior blocks
            for cx, cy, sx, sy in [(g.bounds[0], g.bounds[1], 1, 1), (g.bounds[2], g.bounds[3], -1, -1)]:
                r_in, c_in = ds.index(cx + sx * 3 * res, cy + sy * 3 * res)
                r_out, c_out = ds.index(cx - sx * 4 * res, cy - sy * 4 * res)
                assert alpha[r_in, c_in] == 255
                assert alpha[r_out, c_out] == 0


def test_paper_is_transparent(warped):
    sheet, t, out = warped
    with rasterio.open(out) as ds:
        rgba = ds.read()
        # world point in a street between blocks (paper)
        r, c = ds.index(6_005_000.0 + 430, 2_115_000.0 + 135)
    assert rgba[3, r, c] == 0
    assert (rgba[3] == 0).mean() > 0.2


def test_no_white_halo(tmp_path):
    """Soft (anti-aliased) ink on white paper: partially transparent pixels stay dark."""
    sheet = make_sheet(_blocks(), px_per_ft=PX_PER_FT, rotation_deg=3.0, distortion=_stretch,
                       style="filled", seed=1)
    soft = cv2.GaussianBlur(sheet.img, (0, 0), 1.5)
    png = tmp_path / "soft.png"
    cv2.imwrite(str(png), soft)
    h, w = soft.shape
    P = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    t = SheetTransform([GCP(*p, *sheet.true_px_to_world(p), "auto") for p in P], smoothing=1.0).fit()
    out = warp.warp_sheet(png, t, [], tmp_path / "o.tif")
    with rasterio.open(out) as ds:
        rgba = ds.read()
    semi = (rgba[3] > 0) & (rgba[3] < 255)
    assert semi.sum() > 1000
    assert rgba[0][semi].max() < 150, rgba[0][semi].max()  # paper white is 255
    assert rgba[0][semi].mean() < 120


def test_mask_box_is_transparent(warped):
    sheet, t, out = warped
    wx, wy = t.to_world(np.array([[150.0, 120.0]]))[0]
    with rasterio.open(out) as ds:
        r, c = ds.index(wx, wy)
        assert ds.read(4)[r, c] == 0


def test_resolution_is_median_scale(warped):
    sheet, t, out = warped
    with rasterio.open(out) as ds:
        assert ds.res[0] == pytest.approx(1 / PX_PER_FT, rel=0.05)


def test_color_paper_rule(tmp_path):
    img = np.full((200, 200, 3), (225, 235, 240), np.uint8)  # warm paper tone, BGR
    img[50:100, 50:100] = (40, 40, 200)  # red ink
    img[120:160, 20:60] = (190, 120, 60)  # blue-ish colored area, luminance < otsu
    a = warp.paper_alpha(img)
    assert a[10, 10] == 0
    assert a[75, 75] == 255
    assert a[140, 40] == 255


def test_gray_alpha_soft_edge():
    g = np.full((10, 10), 255, np.uint8)
    g[:, 5:] = 0
    g[:, 4] = 175  # anti-aliased edge pixel
    a = warp.paper_alpha(g)
    assert a[0, 0] == 0 and a[0, 8] == 255
    assert 0 < a[0, 4] < 255


def _edge_crossing(profile, lo, hi):
    """Sub-pixel index (pixel-centre coords) where alpha crosses 128 within [lo, hi)."""
    for i in range(lo, hi - 1):
        a, b = float(profile[i]), float(profile[i + 1])
        if (a - 128) * (b - 128) <= 0 and a != b:
            return i + (128 - a) / (b - a)
    raise AssertionError("no edge found")


def test_warp_edges_within_one_pixel_of_truth(tmp_path):
    """Warped block edges land at their true world position (no half-pixel GCP offset).

    make_sheet uses the pixel-centre convention (cv2.polylines/remap: integer coordinate =
    pixel centre, and true_px_to_world is evaluated at those indices), so t.to_world of a
    pixel index is the pixel-centre world position and the test is not circular.
    The source is binary (edges quantised to a source pixel), so besides a per-edge bound of
    one output pixel we assert the mean signed error over many edges is tiny, which is what
    a missing +0.5 shift (0.5 source px = 0.9 ft) would violate.
    """
    blocks = _blocks()
    sheet = make_sheet(blocks, px_per_ft=PX_PER_FT, rotation_deg=0.0, distortion=lambda p: p,
                       style="filled", seed=1)
    h, w = sheet.img.shape
    P = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    t = SheetTransform([GCP(*p, *sheet.true_px_to_world(p), "auto") for p in P], smoothing=1.0).fit()
    png = tmp_path / "p.png"
    cv2.imwrite(str(png), sheet.img)
    out = warp.warp_sheet(png, t, [], tmp_path / "o.tif")
    errs_x, errs_y = [], []
    with rasterio.open(out) as ds:
        alpha = ds.read(4)
        res, left, top = ds.res[0], ds.bounds.left, ds.bounds.top
        for g in blocks.geometry:
            x0, y0, x1, y1 = g.bounds
            cx, cy = (x0 + x1) / 2 + 60, (y0 + y1) / 2 + 40
            r, c = ds.index(cx, cy)
            for xe in (x0, x1):
                ce = int((xe - left) / res)
                f = _edge_crossing(alpha[r], ce - 4, ce + 5)
                errs_x.append(left + (f + 0.5) * res - xe)
            for ye in (y0, y1):
                re_ = int((top - ye) / res)
                f = _edge_crossing(alpha[:, c], re_ - 4, re_ + 5)
                errs_y.append(top - (f + 0.5) * res - ye)
    for e in errs_x + errs_y:
        assert abs(e) < res, e
    assert abs(np.mean(errs_x)) < 0.3 * res and abs(np.mean(errs_y)) < 0.3 * res, \
        (np.mean(errs_x) / res, np.mean(errs_y) / res)


def _bordered(h=1600, w=1800, margin=100, stroke=12):
    img = np.full((h, w), 255, np.uint8)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), 0, stroke)
    img[600:700, 700:900] = 0   # ink well inside
    return img


def test_neatline_stroke_is_transparent():
    img = _bordered()
    rgba = warp.make_rgba(img, [])
    a = rgba[..., 3]
    assert a[100:112, 300:1500].max() == 0      # top border stroke
    assert a[800, 95:115].max() == 0            # left border stroke
    assert a[600:700, 700:900].min() == 255     # interior ink untouched
    assert a[:warp.NEATLINE_INSET_PX + 100].max() == 0   # whole inset band is clear


def test_no_neatline_means_no_inset():
    img = np.full((600, 600), 255, np.uint8)
    img[0:20, :] = 0
    assert warp.make_rgba(img, [])[..., 3][0:20].min() == 255


def test_warp_writes_footprint_and_clears_stroke(tmp_path):
    import json
    from shapely.geometry import shape
    img = _bordered()
    png = tmp_path / "p.png"
    cv2.imwrite(str(png), img)
    h, w = img.shape
    P = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    W = P * 2.0 * np.array([1, -1]) + np.array([6_000_000.0, 2_120_000.0])
    t = SheetTransform([GCP(*p, *q, "auto") for p, q in zip(P, W)], smoothing=1.0).fit()
    out = warp.warp_sheet(png, t, [], tmp_path / "w.tif")
    with rasterio.open(out) as ds:
        a = ds.read(4)
        r, c = ds.index(6_000_000 + 2 * 600, 2_120_000 - 2 * 105)   # on the top stroke
        assert a[r, c] == 0
        r, c = ds.index(6_000_000 + 2 * 800, 2_120_000 - 2 * 650)   # in the ink well
        assert a[r, c] == 255
    fp = json.loads((tmp_path / "w.footprint.geojson").read_text())
    g = shape(fp["features"][0]["geometry"])
    assert g.contains(__import__("shapely.geometry").geometry.Point(6_000_000 + 2 * 800, 2_120_000 - 2 * 650))
    assert not g.contains(__import__("shapely.geometry").geometry.Point(6_000_000 + 2 * 105, 2_120_000 - 2 * 800))


def test_two_framed_maps_on_one_page_are_both_cleared():
    img = np.full((1600, 3400), 255, np.uint8)
    for x0 in (100, 1800):
        cv2.rectangle(img, (x0, 100), (x0 + 1400, 1500), 0, 12)
        img[600:700, x0 + 600:x0 + 800] = 0
    a = warp.make_rgba(img, [])[..., 3]
    for x0 in (100, 1800):
        assert a[100:112, x0 + 200:x0 + 1200].max() == 0     # top stroke
        assert a[800, x0 - 5:x0 + 15].max() == 0             # left stroke
        assert a[600:700, x0 + 600:x0 + 800].min() == 255    # ink kept
    assert a[:, 1550:1750].max() == 0                        # gutter between the frames


def test_footprint_skips_blank_corner_of_frame():
    """Content only in the left part of the frame: the blank right part is not in the
    footprint (a neighbour's data must show there)."""
    img = np.full((1600, 1800), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (1700, 1500), 0, 12)
    for y in range(250, 1400, 60):
        cv2.line(img, (250, y), (800, y), 0, 6)          # content block at x 250-800
    h, w = img.shape
    P = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    W = P * np.array([1, -1]) + np.array([6_000_000.0, 2_120_000.0])   # 1 ft/px
    t = SheetTransform([GCP(*p, *q, "auto") for p, q in zip(P, W)], smoothing=1.0).fit()
    excl = warp.inset_exclusion(img, [])
    g = warp.footprint_polygons(excl, warp.make_rgba(img, [], excl)[..., 3], t)
    from shapely.geometry import Point
    assert g.contains(Point(6_000_000 + 500, 2_120_000 - 800))
    assert not g.contains(Point(6_000_000 + 1400, 2_120_000 - 800))
    assert g.bounds[2] < 6_000_000 + 900


def test_combine_rings_repairs_bowtie():
    from shapely.geometry import Polygon
    bow = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])      # self-intersecting
    hole = Polygon([(2, 4), (3, 4), (3, 6), (2, 6)])
    assert not bow.is_valid
    g = warp._combine_rings([bow, hole])
    assert g.is_valid and not g.is_empty and g.area > 0


def test_footprint_falls_back_to_neatline_on_failure(monkeypatch, caplog):
    img = np.full((400, 500), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (450, 350), 0, 6)
    cv2.line(img, (100, 100), (300, 200), 0, 4)
    h, w = img.shape
    P = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    W = P * np.array([1, -1]) + np.array([6_000_000.0, 2_120_000.0])
    t = SheetTransform([GCP(*p, *q, "auto") for p, q in zip(P, W)], smoothing=1.0).fit()
    excl = warp.inset_exclusion(img, [])

    def boom(rings):
        raise RuntimeError("TopologyException")
    monkeypatch.setattr(warp, "_combine_rings", boom)
    with caplog.at_level("WARNING"):
        g = warp.footprint_polygons(excl, warp.make_rgba(img, [], excl)[..., 3], t)
    assert g.is_valid and g.area > 0
    assert "falling back" in caplog.text


def test_paper_mode_keeps_the_page_as_scanned():
    """Paper mode: paper is opaque inside the neatline, gray tones are multiplied onto the
    cream tint and mask boxes are filled with plain paper; the frame and inset band are cut
    (stitch's paper backing fills the gap between abutting sheets)."""
    img = _bordered()
    img[400:420, 400:420] = 200   # light grey paper tone, would be transparent without paper
    rgba = warp.make_rgba(img, [Box(x0=1000, y0=900, x1=1100, y1=1000)], paper=True)
    a, rgb = rgba[..., 3], rgba[..., :3]
    assert a[400:420, 400:420].min() == 255
    assert tuple(rgb[300, 300]) == warp.PAPER_TINT                                # white -> cream
    assert tuple(rgb[410, 410]) == tuple(round(200 * c / 255) for c in warp.PAPER_TINT)
    assert a[600:700, 700:900].min() == 255 and rgb[600:700, 700:900].max() == 0   # ink stays black
    assert a[:150].max() == 0                         # frame stroke (rows 94-106) and inset cut away
    assert a[900:1000, 1000:1100].min() == 255                                    # mask box: paper,
    assert (rgb[900:1000, 1000:1100] == warp.PAPER_TINT).all()                    # not a hole


def test_paper_mode_leaves_colour_pages_untouched():
    img = np.full((1600, 1800, 3), (231, 242, 246), np.uint8)   # BGR paper of a 1948 photo
    cv2.rectangle(img, (100, 100), (1700, 1500), (0, 0, 0), 12)
    rgba = warp.make_rgba(img, [], paper=True)
    assert tuple(rgba[800, 900, :3]) == (246, 242, 231)


def test_paper_mode_unframed_blank_margin_is_not_content():
    """The paper is opaque over the whole page area, but the inked content (what stitch draws
    over other sheets) excludes a blank margin far from any ink, and a legend box's own ink."""
    img = np.full((1600, 1800), 255, np.uint8)
    for x in range(200, 900, 60):
        img[200:1000, x:x + 4] = 0
    for y in range(200, 1000, 60):
        img[y:y + 4, 200:900] = 0
    img[1300:1500, 1300:1700] = 0                       # legend ink, away from the map
    boxes = [Box(x0=1250, y0=1250, x1=1750, y1=1550)]
    excl = warp.inset_exclusion(img, [])
    a = warp.make_rgba(img, boxes, excl, paper=True)[..., 3]
    assert (a == np.where(excl, 0, 255)).all()          # paper: the whole non-excluded page
    c = warp.paper_content(img, boxes, excl)
    assert c[500, 500] == 255                           # between the drawn lines
    assert c[1400, 1500] == 0                           # the legend is not content
    assert c[1450, 200] == 0                            # blank margin


def test_erase_border_rules_keeps_crossing_streets():
    """The printed frame rule goes; a street crossing the frame band and the map just inside
    the old 60 px trim stay."""
    img = _bordered()
    img[95:400, 800:806] = 0              # a street running out to the frame, across the band
    out = warp.erase_border_rules(img)
    assert out[100:112, 300:700].min() == 255          # top rule erased
    assert out[130:200, 800:806].max() == 0            # street crossing the band kept (beyond the rule's 15 px)
    assert out[600:700, 700:900].max() == 0            # interior ink untouched
    excl = warp.inset_exclusion(img, [], inset=warp.PAPER_EDGE_INSET_PX)   # from the page as printed
    assert not excl[130, 800]                           # map kept up to near the frame
    assert excl[50, 800] and excl[800, 50]              # margin outside the frame still clear


def test_erase_border_rules_handles_a_tilted_frame():
    """A frame rotated 2 degrees (a photographed page) is still erased."""
    img = np.full((1600, 1800), 255, np.uint8)
    box = cv2.boxPoints(((900, 800), (1600, 1400), 2.0)).astype(np.int32)
    cv2.polylines(img, [box], True, 0, 14)
    img[600:700, 700:900] = 0
    out = warp.erase_border_rules(img)
    edge = np.zeros(img.shape, np.uint8)
    cv2.polylines(edge, [box], True, 1, 14)
    assert (out[edge.astype(bool)] == 0).mean() < 0.05     # >95% of the rule gone
    assert out[600:700, 700:900].max() == 0


def test_erase_border_rules_unframed_page_heavy_edge_rule_only():
    """No neatline: a heavy rule near the page edge goes; a thin line there and the map stay."""
    img = np.full((1600, 1800), 255, np.uint8)
    img[200:1500, 120:140] = 0                          # heavy edge rule, 20 px, ~120 px in
    img[200:1500, 1700:1704] = 0                        # thin line near the other edge
    for x in range(300, 1500, 60):                      # the map (no closed frame)
        img[300:1400, x:x + 4] = 0
    out = warp.erase_border_rules(img)
    assert out[300:1400, 120:140].min() == 255
    assert out[300:1400, 1700:1704].max() == 0
    assert out[300:1400, 300:304].max() == 0


def test_erase_border_rules_binding_shadow_on_framed_page():
    """A framed page photographed in a bound atlas: the dark binding shadow along the page edge
    (outside the frame band, 30 px wide) is erased too."""
    img = _bordered()
    img[:, 1770:1800] = 30
    out = warp.erase_border_rules(img)
    assert out[300:1300, 1772:1798].min() == 255
    assert out[600:700, 700:900].max() == 0


def test_erase_border_rules_double_frame_inner_line():
    """A double frame: the inner line ~80 px in is erased; a shorter street line at the same
    depth stays."""
    img = _bordered()
    img[180:186, 300:1500] = 0          # inner frame line, 80 px below the 100 px frame
    img[1400:1406, 400:700] = 0         # a short street line at a similar depth near the bottom
    out = warp.erase_border_rules(img)
    assert out[180:186, 400:1400].min() == 255
    assert out[1400:1406, 450:650].max() == 0
