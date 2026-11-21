import json

import numpy as np
from PIL import Image

from sfzmaps.frame import data, look, run


def _page(work, sid, h, w):
    p = data.page_path(*sid, work=work)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((h, w), 255, np.uint8)
    a[::7] = 0
    Image.fromarray(a).save(p)


def _outputs(out, sid, h, w):
    d = run.sheet_dir(out, sid)
    d.mkdir(parents=True, exist_ok=True)
    deleted = np.zeros((h, w), bool)
    deleted[:, : w // 4] = True
    Image.fromarray(deleted).save(d / run.OUTSIDE_PNG)
    np.savez_compressed(d / run.PROB_FILE, q=np.where(deleted, 255, 0).astype(np.uint8))
    x = w // 4
    (d / run.OUTLINE_FILE).write_text(json.dumps(
        {"polygons": [{"exterior": [[x, 0], [w, 0], [w, h], [x, h], [x, 0]], "holes": []}], "maps": 1}))


def test_sheet_picture_never_enlarges():
    lum = np.ones((30, 50), np.float32)
    q = np.zeros((30, 50), np.float32)
    deleted = np.zeros((30, 50), bool)
    assert look.sheet_picture(lum, q, deleted, side=1400).size == (50, 30)
    big = look.sheet_picture(np.ones((300, 500), np.float32), np.zeros((300, 500), np.float32),
                             np.zeros((300, 500), bool), side=100)
    assert big.size == (100, 60)


def test_sheet_picture_colours_ink_by_chance_of_deletion():
    lum = np.zeros((20, 40), np.float32)            # all ink
    q = np.zeros((20, 40), np.float32)
    q[:, :20] = 1.0
    deleted = q > 0.5
    a = np.asarray(look.sheet_picture(lum, q, deleted))
    assert a[10, 5, 0] > 200 and a[10, 5, 2] < 50       # deleted ink: red
    assert a[10, 35, 2] > 200 and a[10, 35, 0] < 50     # kept ink: blue
    assert (a[10, 19:21] == look.CONTOUR_RGB).all(axis=1).any()    # the kept region's boundary


def test_closeups_are_full_resolution(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "out"
    monkeypatch.setattr(data, "WORK", work)
    sid = ("s", "p")
    _page(work, sid, 500, 600)
    _outputs(out, sid, 500, 600)
    path = look.closeups(out, [("s", "p", (10, 20, 310, 320), "a crop"), ("s", "missing", (0, 0, 9, 9), "none")])
    assert path == out / "look" / "closeups.png"
    with Image.open(path) as im:
        assert im.width >= 300 and im.height > 300      # 300 px crop, unshrunk, plus its caption


def test_series_mosaic_caption_has_held_out_scores(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "out"
    monkeypatch.setattr(data, "WORK", work)
    kinds = tmp_path / "kinds.csv"
    kinds.write_text("series,sheet,kind,maps\ns,p,map,1\ns,q,map,1\n")
    monkeypatch.setattr(data, "PAGE_KINDS", kinds)
    for sid in (("s", "p"), ("s", "q")):
        _page(work, sid, 60, 80)
    _outputs(out, ("s", "p"), 60, 80)
    (out / "folds.json").write_text(json.dumps({"sheets": [
        {"series": "s", "sheet": "p", "scores": {"frame_kept": 12.5, "map_cut": 3.25}}]}))
    seen = []
    monkeypatch.setattr(look, "_mosaic", lambda tiles, side, cols, path: seen.append((tiles, path)) or path)
    paths = look.series_mosaics(out)
    assert paths == [out / "look" / "s.png"]
    (tiles, _), = seen
    assert len(tiles) == 1 and "12.5" in tiles[0][1][1] and "3.25" in tiles[0][1][1]
    assert (out / "look" / "s" / "p.png").exists()
