import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

from sfzmaps import placement
from sfzmaps.reading import Label
from sfzmaps.sources.datasf import Reference
from sfzmaps.transform import GCP, SheetTransform

X0, Y0, S = 6_005_000.0, 2_118_000.0, 2.5   # world = (X0 + S px, Y0 - S py)


def _ref():
    blocks = gpd.GeoDataFrame({"block_num": [f"{i + 1:04d}" for i in range(4)]},
                              geometry=[box(X0 + 100 + 460 * i, Y0 - 500, X0 + 500 + 460 * i, Y0 - 240)
                                        for i in range(4)], crs="EPSG:2227")
    streets = gpd.GeoDataFrame({"street": ["MAIN"], "streetname": ["MAIN ST"], "active": ["true"]},
                               geometry=[LineString([(X0, Y0 - 200), (X0 + 2000, Y0 - 200)])],
                               crs="EPSG:2227")
    return Reference(blocks, None, None, streets)


def _labels():
    out = [Label(kind="block_number", text=str(i + 1), x=int((300 + 460 * i) / S), y=int(370 / S),
                 confidence=1) for i in range(4)]
    out += [Label(kind="street_name", text="Main St", x=int(x / S), y=int(200 / S), confidence=1)
            for x in (300, 900)]
    out.append(Label(kind="block_number", text="9999", x=10, y=10, confidence=1))  # no such block
    return out


def _t(dx=0.0, dy=0.0):
    g = [GCP(px, py, X0 + S * px + dx, Y0 - S * py + dy, "anchor")
         for px in (0, 1000, 2000) for py in (0, 1000)]
    return SheetTransform(g, 1000.0)


def test_label_agreement_placed_and_misplaced():
    a = placement.label_agreement(_t(), _labels(), _ref())
    assert (a.fraction, a.n, a.n_block, a.n_street) == (1.0, 6, 4, 2)
    far = placement.label_agreement(_t(3000.0, 0.0), _labels(), _ref())
    assert far.fraction == 0.0
    # one street over (460 ft): block labels land in the neighbouring block, not their own
    one_over = placement.label_agreement(_t(460.0, 0.0), _labels(), _ref())
    assert one_over.fraction <= 2 / 6


def test_label_agreement_nothing_checkable():
    a = placement.label_agreement(_t(), [Label(kind="place_name", text="X", x=1, y=1, confidence=1)], _ref())
    assert a.n == 0 and np.isnan(a.fraction)
