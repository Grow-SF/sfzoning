"""Reading a whole sheet (`find.wholesheet`, `sfzmaps find read`): the read stored at the page's own size (0 off the
frame cut) and its RESULT picture, one line a sheet; a px's answer the same whatever window round it is read, when the
window's corner is on the page's MULTIPLE px grid; and the refusals (a duplicate scan, no page, no frame cut, no
weights)."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from find_fixtures import invoke
from sfzmaps.find import model, pages, read
from sfzmaps.frame import data as fdata
from sfzmaps.solver import hatching

SHAPE = (150, 120)


def _sheet_world(tmp_path, monkeypatch, sheets=("A", "B"), frames=("A", "B"), dups=None):
    """Series `s` with map pages `sheets` (each a white page with two dark lines), frame cuts for `frames` (all but a
    10 px strip on the left), the duplicates `dups` ({sheet: kept}), and a run `r` with weights."""
    work, frame = tmp_path / "work", tmp_path / "frame"
    page = np.full(SHAPE, 255, np.uint8)
    page[40:42] = page[60:62] = 0
    cut = np.zeros(SHAPE, np.uint8)
    cut[:, :10] = 255
    for h in sheets:
        (work / "s" / h).mkdir(parents=True)
        Image.fromarray(page, "L").save(work / "s" / h / "page.png")
    for h in frames:
        (frame / "s" / h).mkdir(parents=True)
        Image.fromarray(cut, "L").save(frame / "s" / h / "outside.png")
    monkeypatch.setattr(fdata, "WORK", work)
    monkeypatch.setattr(fdata, "OUT", frame)
    monkeypatch.setattr(fdata, "map_pages", lambda kinds=None, work=None: [("s", h) for h in sorted(set(sheets) |
                                                                                                    set(dups or {}))])
    monkeypatch.setattr(hatching, "duplicate_scans", lambda series: dict(dups or {}))
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    (tmp_path / "find" / "runs" / "r").mkdir(parents=True)
    torch.manual_seed(0)
    torch.save({"state_dict": model.FindNet(4).state_dict(), "step": 1}, tmp_path / "find" / "runs" / "r" / "model.pt")
    return tmp_path / "find" / "runs" / "r"


def test_a_sheet_is_read_whole_stored_and_pictured(tmp_path, monkeypatch):
    run = _sheet_world(tmp_path, monkeypatch)
    r = invoke(["find", "read", "--run", "r", "--sheet", "s/A", "--device", "cpu"])
    assert r.exit_code == 0, r.output
    f = run / "sheets" / "s" / "A.npz"
    got = read.load(f)
    assert got.shape == (2, *SHAPE) and (got[:, :, :10] == 0).all() and got[:, :, 10:].any()
    picture = Image.open(run / "sheets" / "s" / "A.png")
    assert picture.width == SHAPE[1] and picture.height > SHAPE[0]           # the page's own size, a key above
    line = [ln for ln in r.output.splitlines() if ln.startswith("s/A:")]
    assert len(line) == 1 and "150 x 120 px" in line[0] and "called street" in line[0] and " s" in line[0]
    assert not (run / "sheets" / "s" / "B.npz").exists()                     # nothing read that was not asked for
    again = invoke(["find", "read", "--run", "r", "--sheet", "s/A", "--device", "cpu"])
    assert again.exit_code == 0 and np.array_equal(read.load(f), got)        # the same bits on the same device


def test_a_px_reads_the_same_whatever_window_round_it_on_the_page_grid():
    torch.manual_seed(0)
    net = model.FindNet(4).eval()
    rng = np.random.default_rng(2)
    lum = np.ones((300, 260), np.float32)
    for _ in range(40):                                      # printed lines at random
        y, x = rng.integers(0, 290), rng.integers(0, 250)
        lum[y:y + 2, x:x + rng.integers(5, 60)] = 0
        lum[y:y + rng.integers(5, 60), x:x + 2] = 0
    keep = np.ones(lum.shape, bool)
    whole = read.read_page(net, lum, keep, "cpu", tile=64)
    for y0, x0, h, w in ((40, 50, 60, 70), (100, 30, 90, 40), (150, 170, 80, 60)):       # each across square joins
        oy, ox = (y0 - read.CONTEXT) // model.MULTIPLE * model.MULTIPLE, (x0 - read.CONTEXT) // model.MULTIPLE * \
            model.MULTIPLE                                   # the window's corner on the page's grid
        oy, ox = max(oy, 0), max(ox, 0)
        ey, ex = min(y0 + h + read.CONTEXT, lum.shape[0]), min(x0 + w + read.CONTEXT, lum.shape[1])
        win = read.read_page(net, lum[oy:ey, ox:ex], keep[oy:ey, ox:ex], "cpu", tile=32)
        np.testing.assert_allclose(win[:, y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w], whole[:, y0:y0 + h, x0:x0 + w],
                                   atol=1e-5, rtol=0)


def test_each_refusal_names_the_sheet_and_why(tmp_path, monkeypatch):
    _sheet_world(tmp_path, monkeypatch, sheets=("A", "B"), frames=("A",), dups={"D": "A"})
    for sheet, why in (("s/D", "s/D is a duplicate scan of A"), ("s/B", "s/B: no frame cut"),
                       ("s/C", "s/C: no page image")):
        r = invoke(["find", "read", "--run", "r", "--sheet", sheet, "--device", "cpu"])
        assert r.exit_code == 1 and why in r.output, (sheet, r.output)
    r = invoke(["find", "read", "--run", "r", "--sheet", "s/A", "--sheet", "s/B", "--device", "cpu"])
    assert r.exit_code == 1 and not (tmp_path / "find" / "runs" / "r" / "sheets").exists()    # nothing read
    r = invoke(["find", "read", "--run", "none", "--sheet", "s/A", "--device", "cpu"])
    assert r.exit_code == 1 and "no model" in r.output
    r = invoke(["find", "read", "--run", "r", "--device", "cpu"])
    assert r.exit_code == 2 and "--sheet" in r.output and "--series" in r.output


def test_a_series_is_read_with_its_duplicates_skipped_and_named(tmp_path, monkeypatch):
    run = _sheet_world(tmp_path, monkeypatch, dups={"D": "A"})
    r = invoke(["find", "read", "--run", "r", "--series", "s", "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert "s/D: skipped, a duplicate scan of A" in r.output
    assert sorted(p.name for p in (run / "sheets" / "s").glob("*.npz")) == ["A.npz", "B.npz"]
