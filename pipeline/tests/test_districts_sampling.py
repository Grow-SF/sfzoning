import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from sfzmaps import config
from sfzmaps.districts import sampling
from synthetic_districts import write_sheet


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    return tmp_path


def _parcels(*items):
    return gpd.GeoDataFrame({"blklot": [b for b, _ in items]}, geometry=[g for _, g in items], crs=config.WORK_CRS)


def test_priority_two_sheets(work):
    write_sheet("1948", "lo", 0, 0, 200, 100, passed=False, p50=9.0)
    write_sheet("1948", "hi", 100, 0, 200, 100, passed=True, p50=3.0)
    assert sampling.sheet_order("1948") == ["hi", "lo"]
    p = _parcels(("a", box(10, 10, 50, 50)), ("b", box(120, 10, 160, 50)), ("c", box(250, 10, 290, 50)))
    s = sampling.assign_sheets(p, "1948")
    assert s["a"] == "lo" and s["b"] == "hi" and s["c"] == "hi"   # b in the overlap: hi wins


def test_uncovered_parcel_none(work):
    write_sheet("1948", "s", 0, 0, 100, 100)
    p = _parcels(("in", box(10, 10, 30, 30)), ("out", box(500, 500, 520, 520)),
                 ("edge", box(90, 10, 130, 30)))   # 25% covered
    s = sampling.assign_sheets(p, "1948")
    assert s["in"] == "s" and s["out"] is None and s["edge"] is None


def test_crop_masks_outside(work):
    tif = write_sheet("1948", "s", 0, 0, 100, 100, fill="solid")
    crop = sampling.parcel_crop(tif, box(20, 20, 60, 40), inset_ft=3.0, px=64)
    assert crop.shape == (64, 64, 4) and crop.dtype == np.uint8
    assert crop[..., 3].min() == 0 and crop[..., 3].max() == 255
    assert (crop[crop[..., 3] == 0][:, :3] == 0).all()
    tri = sampling.parcel_crop(tif, Polygon([(20, 20), (80, 20), (20, 80)]))
    assert 0.3 < (tri[..., 3] > 0).mean() < 0.7   # the triangle's far corner is masked


def test_crop_none_when_mostly_uncovered(work):
    tif = write_sheet("1948", "s", 0, 0, 100, 100, alpha_box=(0, 0, 100, 30), footprint=(0, 0, 100, 30))
    assert sampling.parcel_crop(tif, box(20, 20, 60, 80)) is None
    assert sampling.parcel_crop(tif, box(500, 500, 520, 520)) is None


def test_texture_stats_distinguish_hatch_vs_solid(work):
    def stats(fill):
        tif = write_sheet("1948", fill, 0, 0, 100, 100, fill=fill)
        return sampling.texture_stats(sampling.parcel_crop(tif, box(10, 10, 80, 80)))
    h, s, b = stats("hatch"), stats("solid"), stats("blank")
    assert h.shape == (6,)
    assert s[0] > 0.95 and b[0] < 0.05 and 0.1 < h[0] < 0.5
    assert h[5] > s[5] + 3 and h[1:5].max() > 0.5   # peaked FFT and one dominant orientation
    assert not np.isnan(s).any() and not np.isnan(b).any()


def test_crop_transparent_paper_reads_white(work):
    """Real warps make paper alpha 0 with black premultiplied RGB: a blank parcel inside the
    footprint is a white crop, not None and not black."""
    tif = write_sheet("1948", "s", 0, 0, 100, 100, fill="blank", paper="transparent")
    crop = sampling.parcel_crop(tif, box(20, 20, 60, 60))
    assert crop is not None
    inside = crop[crop[..., 3] > 0][:, :3]
    assert inside.min() == 255
    assert sampling.texture_stats(crop)[0] < 0.01
    tif = write_sheet("1948", "k", 0, 0, 100, 100, fill="solid", paper="transparent")
    assert sampling.texture_stats(sampling.parcel_crop(tif, box(20, 20, 60, 60)))[0] > 0.99


def test_crop_keep_aspect_pads_not_squashes(work):
    tif = write_sheet("1948", "s", 0, 0, 200, 100, fill="solid")
    crop = sampling.parcel_crop(tif, box(10, 10, 130, 40), inset_ft=0.0, px=64, keep_aspect=True)
    assert crop.shape == (64, 64, 4)
    rows = np.where(crop[..., 3].any(axis=1))[0]
    cols = np.where(crop[..., 3].any(axis=0))[0]
    assert cols.size >= 62 and 14 <= rows.size <= 20      # 120 x 30 ft stays 4:1
    native = sampling.parcel_crop(tif, box(10, 10, 130, 40), inset_ft=0.0, px=None)
    assert abs(native.shape[1] - 121) <= 2 and abs(native.shape[0] - 31) <= 2
