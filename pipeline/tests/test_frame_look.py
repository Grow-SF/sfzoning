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
    deleted = np.zeros((30, 50), bool)
    assert look.sheet_picture(lum, deleted, side=1400).size == (50, 30)
    big = look.sheet_picture(np.ones((300, 500), np.float32), np.zeros((300, 500), bool), side=100)
    assert big.size == (100, 60)


def test_kept_ink_is_never_the_deleted_colour():
    lum = np.full((40, 40), 1.0, np.float32); lum[10:30, 10:30] = 0.0
    deleted = np.zeros((40, 40), bool); deleted[:, :20] = True
    im = np.asarray(look.sheet_picture(lum, deleted, side=40))
    assert (im[20, 25] == look.KEPT_RGB).all() and (im[20, 15] == look.DELETED_RGB).all()
    assert (im[5, 30] == look.PAPER_RGB).all() and (im[5, 5] == look.PAPER_RGB).all()     # paper stays white


def test_shrunk_ink_takes_one_colour_shaded_by_its_darkness():
    # ink every other column, lighter in the top half; the page shrunk 4 times, one cell holding kept and deleted ink:
    # every px lies on the line from paper to the kept colour or to the deleted colour, never between the two
    lum = np.full((80, 80), 1.0, np.float32); lum[:, ::2] = 0.0; lum[:40] = 0.5 + 0.5 * lum[:40]
    deleted = np.zeros((80, 80), bool); deleted[:, 40:] = True; deleted[:, 18] = True     # cell 4: half deleted ink
    im = np.asarray(look.sheet_picture(lum, deleted, side=20)).astype(float)
    paper = np.array(look.PAPER_RGB, float)
    on_a_line = (im == look.CONTOUR_RGB).all(axis=2)                  # the cut's outline, at cells 9-11
    assert not on_a_line[:, :8].any()
    for colour in (look.KEPT_RGB, look.DELETED_RGB):
        d = np.array(colour, float) - paper
        w = ((im - paper) @ d) / (d @ d)
        on_a_line |= (np.abs(paper + w[..., None] * d - im).max(axis=2) <= 1.0) & (w >= -0.01) & (w <= 1.01)
    assert on_a_line.all()
    assert (im[2, 2] != im[12, 2]).any()                                 # lighter ink above, darker below


def test_label_is_drawn_in_green():
    lum = np.ones((100, 100), np.float32)
    deleted = np.zeros((100, 100), bool)
    square = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
    im = np.asarray(look.sheet_picture(lum, deleted, side=100, label=[square]))
    green = (im == look.LABEL_RGB).all(axis=2)
    assert green[20, 30:70].all() and green[30:70, 80].all()             # its edges, closed
    assert not green[50, 50] and not green[5, 5]
    assert not (np.asarray(look.sheet_picture(lum, deleted, side=100)) == look.LABEL_RGB).all(axis=2).any()


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


def test_series_mosaic_caption_has_the_split_scores(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "out"
    monkeypatch.setattr(data, "WORK", work)
    kinds = tmp_path / "kinds.csv"
    kinds.write_text("series,sheet,kind,maps\ns,p,map,1\ns,q,map,1\n")
    monkeypatch.setattr(data, "PAGE_KINDS", kinds)
    for sid in (("s", "p"), ("s", "q")):
        _page(work, sid, 60, 80)
    _outputs(out, ("s", "p"), 60, 80)
    split = {"frame_ink": 80.0, "frame_left": 1.0, "box_ink": 10.0, "box_left": 5.0, "boxes": [], "map_ink": 400.0,
             "map_cut_frame": 1.0, "map_cut_boxes": 0.3}
    (out / run.SCORES_FILE).write_text(json.dumps({"sheets": [{"series": "s", "sheet": "p", "split": split}]}))
    seen = []
    monkeypatch.setattr(look, "_mosaic", lambda tiles, side, cols, path: seen.append((tiles, path)) or path)
    paths = look.series_mosaics(out)
    assert paths == [out / "look" / "s.png"]
    (tiles, _), = seen
    assert len(tiles) == 1 and "1.25%" in tiles[0][1][1] and "50%" in tiles[0][1][1] and "0.325%" in tiles[0][1][1]
    assert (out / "look" / "s" / "p.png").exists()


def test_sheet_picture_draws_the_outline_straight():
    lum = np.zeros((200, 300), np.float32)
    deleted = np.ones((200, 300), bool)
    # a kept region with a tilted left side: a staircase in px, one straight segment once drawn
    rows = np.arange(200)
    for r in rows:
        deleted[r, 50 + r // 10:250] = False
    polygons = run_outline(deleted)
    a = np.asarray(look.sheet_picture(lum, deleted, side=300, polygons=polygons))
    drawn = (a == look.CONTOUR_RGB).all(axis=2)
    for r in (20, 100, 180):                    # the drawn left side runs along the tilted edge, one px wide
        xs = np.flatnonzero(drawn[r, :150])
        assert len(xs) >= 1 and abs(xs.mean() - (50 + r / 10)) <= 1.5


def run_outline(deleted):
    from sfzmaps.frame import decide
    return decide.outline(deleted)


def test_closeups_list_the_blank_strips():
    assert {(s, h) for s, h, _, _ in look.CLOSEUPS} >= {("1948", "2"), ("1948", "9")}


def test_set_mosaic_captions_each_sheet(tmp_path, monkeypatch):
    work, out, labels = tmp_path / "work", tmp_path / "out", tmp_path / "labels-test"
    monkeypatch.setattr(data, "WORK", work)
    monkeypatch.setattr(data, "TEST_LABELS", labels)
    sheets = [("s", "p"), ("t", "q")]
    rows = []
    for i, sid in enumerate(sheets):
        _page(work, sid, 60, 80)
        _outputs(out / run.TEST_DIR, sid, 60, 80)
        (labels / sid[0]).mkdir(parents=True, exist_ok=True)
        (labels / sid[0] / f"{sid[1]}.json").write_text(json.dumps(
            {"polygons": [[[25, 5], [75, 5], [75, 55], [25, 55]]], "remove": [[[40, 20], [50, 20], [50, 30]]]}))
        rows.append({"series": sid[0], "sheet": sid[1], "split": {
            "frame_ink": 80.0, "frame_left": 1.0 + i, "box_ink": 10.0, "box_left": 5.0,
            "boxes": [{"ink": 10.0, "deleted_share": 0.5}], "map_ink": 400.0, "map_cut_frame": 1.0,
            "map_cut_boxes": 0.3}})
    (out / run.TEST_DIR / run.SCORES_FILE).write_text(json.dumps(
        {"set": "test", "temperature": 1.5, "temperature_interval": [1.4, 1.6], "sheets": rows,
         "pooled": run.scores.pooled_split([r["split"] for r in rows])}))
    seen = []
    real = look._mosaic
    monkeypatch.setattr(look, "_mosaic", lambda tiles, side, cols, path, title=():
                        seen.append((tiles, title)) or real(tiles, side, cols, path, title))
    path = look.set_mosaic(out, run.TEST_DIR)
    assert path == out / run.TEST_DIR / "look" / "test-set.png" and path.exists()
    (tiles, title), = seen
    assert len(tiles) == len(sheets)
    for (img, caption), row in zip(tiles, rows):
        want = [t for t in run.sheet_line(row).replace(": ", " ").split() if t != "|"]
        assert " ".join(caption).replace(": ", " ").split() == want
        assert (np.asarray(img) == look.LABEL_RGB).all(axis=2).any()   # the test label, in green
    assert any(line.startswith("pooled (2 sheets)") for line in title)
