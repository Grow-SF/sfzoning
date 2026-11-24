"""The finder's pages: which pages are read, the scan and the frame cut, the label sources."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sfzmaps import config
from sfzmaps.find import pages
from sfzmaps.frame import data as fdata

CHECKED_IN = Path(__file__).parents[1] / "inputs" / "find" / "label_sources.csv"


def _page(work, series, sheet, arr, mode="L"):
    d = work / series / sheet
    d.mkdir(parents=True)
    Image.fromarray(arr, mode).save(d / "page.png")


@pytest.fixture
def world(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "frame"
    kinds = tmp_path / "page_kinds.csv"
    monkeypatch.setattr(fdata, "WORK", work)
    monkeypatch.setattr(fdata, "OUT", out)
    monkeypatch.setattr(fdata, "PAGE_KINDS", kinds)
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    return work, out, kinds, tmp_path


def _kinds(kinds, rows):
    kinds.write_text("series,sheet,kind\n" + "".join(f"{s},{h},{k}\n" for s, h, k in rows))


def _cut(out, series, sheet, arr):
    d = out / series / sheet
    d.mkdir(parents=True)
    Image.fromarray(arr, "L").save(d / "outside.png")


def test_sheets_leaves_out_duplicates_and_non_maps(world):
    work, out, kinds, tmp = world
    px = np.full((10, 10), 255, np.uint8)
    for sh in "abcd":
        _page(work, "s", sh, px)
        _cut(out, "s", sh, px * 0)
    _kinds(kinds, [("s", "a", "map"), ("s", "b", "text"), ("s", "c", "map"), ("s", "d", "text")])
    (tmp / "inputs" / "s").mkdir(parents=True)
    (tmp / "inputs" / "s" / "duplicate_scans.csv").write_text("sheet,kept,reason\nc,a,same scan\n")
    assert pages.sheets("s") == [("s", "a")]


def test_a_map_page_without_a_frame_cut_is_refused(world):
    work, out, kinds, _ = world
    px = np.full((10, 10), 255, np.uint8)
    for sh in "ab":
        _page(work, "s", sh, px)
    _cut(out, "s", "a", px * 0)
    _kinds(kinds, [("s", "a", "map"), ("s", "b", "map")])
    with pytest.raises(FileNotFoundError, match="s/b"):
        pages.sheets("s")


def test_keep_is_the_frame_cut_and_its_shape_is_checked(world):
    _, out, _, _ = world
    cut = np.full((10, 10), 255, np.uint8)
    cut[:5] = 0
    _cut(out, "s", "a", cut)
    keep = pages.load_keep(("s", "a"), (10, 10))
    assert keep.dtype == bool and keep[:5].all() and not keep[5:].any()
    small = np.zeros((8, 8), np.uint8)
    _cut(out, "s", "b", small)
    with pytest.raises(ValueError):
        pages.load_keep(("s", "b"), (10, 10))


def test_colour_scan_is_luminance(world):
    work, _, _, _ = world
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    _page(work, "s", "a", red, "RGB")
    scan = pages.load_scan(("s", "a"))
    assert scan.dtype == np.float32 and scan.shape == (4, 4)
    assert abs(float(scan[0, 0]) - 0.299) < 1 / 255


def test_sources_rows_and_refusals(tmp_path):
    table = pages.sources(CHECKED_IN)
    assert len(table) == 22
    assert pages.sources(CHECKED_IN)["1921"] == pages.Source("1921", "hatched-grey", "")
    assert table["2000-digital-height"].lots == "table" and table["1990-1996-su"].lots == "year"
    assert not table["1960"].lots
    bad = tmp_path / "bad.csv"
    bad.write_text("series,era,lots\nx,lined-grey,parcels\n")
    with pytest.raises(ValueError, match="unknown lots 'parcels'"):
        pages.sources(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text("series,era,lots\nx,lined-grey,\nx,lined-grey,\n")
    with pytest.raises(ValueError):
        pages.sources(dup)
    empty = tmp_path / "e.csv"
    empty.write_text("series,era,lots\nx,lined-grey,\n")
    with pytest.raises(KeyError, match="e.csv"):
        pages.source("nope", empty)
