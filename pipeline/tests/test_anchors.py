import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, box

from sfzmaps import anchors
from sfzmaps.reading import Label
from sfzmaps.sources.datasf import Reference
from synthetic import make_sheet


def _grid_blocks(n=6, w=400.0, h=260.0, street=60.0, x0=6_005_000.0, y0=2_115_000.0):
    rows, geoms = [], []
    for i in range(n):
        for j in range(n):
            x = x0 + j * (w + street)
            y = y0 + i * (h + street)
            rows.append(f"{i * n + j + 1:04d}")
            geoms.append(box(x, y, x + w, y + h))
    return gpd.GeoDataFrame({"block_num": rows}, geometry=geoms, crs="EPSG:2227")


def _ref(blocks, inter=None):
    if inter is None:
        inter = gpd.GeoDataFrame({"node_cnn": [], "streets": []}, geometry=[], crs="EPSG:2227")
    return Reference(blocks, None, inter, None)


def _sheet(rotation_deg):
    blocks = _grid_blocks()
    sheet = make_sheet(blocks, px_per_ft=0.55, rotation_deg=rotation_deg,
                       distortion=lambda p: p, style="outline")
    return blocks, sheet


def _median_err(blocks, sheet, fit):
    px = np.array([[l.x, l.y] for l in sheet.labels], float)
    truth = np.array([[g.centroid.x, g.centroid.y] for g in blocks.geometry])
    return np.median(np.linalg.norm(fit.transform.to_world(px) - truth, axis=1))


def _angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_rough_fit_recovers_rotation_90():
    blocks, sheet = _sheet(90.0)
    fit = anchors.rough_fit(anchors.match_labels(sheet.labels, _ref(blocks)))
    assert fit is not None
    # rotation_deg is the world-frame angle of the pixel +x axis (y flipped);
    # the synthetic sheet rotates world by +90 to get pixels, so that is -90.
    assert _angle_diff(fit.rotation_deg, -90.0) < 1.0
    assert abs(fit.scale_ft_per_px - 1 / 0.55) < 0.05
    assert _median_err(blocks, sheet, fit) < 30.0
    assert all(g.source == "anchor" for g in fit.transform.gcps)


def test_rough_fit_survives_misreads():
    blocks, sheet = _sheet(15.0)
    rng = np.random.default_rng(1)
    nums = list(blocks["block_num"])
    labels = list(sheet.labels)
    bad = rng.choice(len(labels), size=int(0.3 * len(labels)), replace=False)
    for i in bad:
        labels[i] = labels[i].model_copy(update={"text": str(rng.choice(nums))})
    fit = anchors.rough_fit(anchors.match_labels(labels, _ref(blocks)))
    assert fit is not None
    assert _angle_diff(fit.rotation_deg, -15.0) < 1.0
    assert _median_err(blocks, sheet, fit) < 30.0
    assert len(fit.inliers) < len(labels)


def test_rough_fit_deterministic():
    blocks, sheet = _sheet(40.0)
    c = anchors.match_labels(sheet.labels, _ref(blocks))
    a, b = anchors.rough_fit(c), anchors.rough_fit(c)
    assert a.rotation_deg == b.rotation_deg and a.scale_ft_per_px == b.scale_ft_per_px
    assert [x.label.text for x in a.inliers] == [x.label.text for x in b.inliers]


def test_rough_fit_none_when_few_labels():
    blocks, sheet = _sheet(0.0)
    assert anchors.rough_fit(anchors.match_labels(sheet.labels[:3], _ref(blocks))) is None


def test_match_labels_normalizes_and_drops_unknown():
    blocks = _grid_blocks(2)
    labels = [Label(kind="block_number", text=t, x=10, y=10, confidence=1.0)
              for t in ["2", "0003A", "9999", "xx"]]
    c = anchors.match_labels(labels, _ref(blocks))
    assert [x.label.text for x in c] == ["2", "0003A"]
    cen = blocks.geometry.iloc[1].centroid
    assert c[0].world == (cen.x, cen.y) and c[0].weight == 1.0


def test_multipolygon_block_uses_largest_part():
    small, big = box(0, 0, 10, 10), box(1000, 0, 1400, 300)
    blocks = gpd.GeoDataFrame({"block_num": ["0007"]}, geometry=[MultiPolygon([small, big])],
                              crs="EPSG:2227")
    c = anchors.match_labels([Label(kind="block_number", text="7", x=0, y=0, confidence=1)],
                             _ref(blocks))
    assert big.contains(Point(c[0].world))


def test_street_pair_candidates():
    inter = gpd.GeoDataFrame(
        {"node_cnn": ["1", "2", "3"],
         "streets": [["LOMBARD", "VAN NESS"], ["LOMBARD", "VAN NESS"], ["1ST", "MARKET"]]},
        geometry=[Point(100, 200), Point(5000, 6000), Point(9, 9)], crs="EPSG:2227")
    labels = [
        Label(kind="street_name", text="Lombard St", x=1000, y=1000, confidence=1),
        Label(kind="street_name", text="Van Ness Ave", x=1200, y=1100, confidence=1),
        Label(kind="street_name", text="Market St", x=5000, y=5000, confidence=1),  # too far
    ]
    c = anchors.match_labels(labels, _ref(_grid_blocks(2), inter))
    assert len(c) == 2 and all(x.kind == "street_pair" and x.weight == 0.5 for x in c)
    assert {x.world for x in c} == {(100.0, 200.0), (5000.0, 6000.0)}
    assert all((x.label.x, x.label.y) == (1100, 1050) for x in c)
