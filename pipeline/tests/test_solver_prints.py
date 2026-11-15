import gzip
import json
import math

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

from sfzmaps import llm
from sfzmaps.solver import grid, prints, saved

SHEET = "2000-digital-zoning-p01"


def _box(text, x0, y0, x1, y1, kind="zoning code", edge=100):
    return {"text": text, "kind": kind, "box": [x0, y0, x1, y1], "edge": edge}


def _label(code, x, y, confidence=0.9):
    return {"code": code, "kind": "use", "x": float(x), "y": float(y), "confidence": confidence}


def test_box_and_point_match_as_labread2():
    boxes = [_box("rh-1", 100, 100, 200, 140),       # two of Sol's points inside: the one nearest the centre wins
             _box("RM-2", 300, 100, 380, 140),       # no point inside: Terra only
             _box("1234", 500, 500, 540, 520, kind="block number")]   # not a zoning code: no print
    labels = [_label("RH-2", 110, 105, 0.5), _label("rh-1", 152, 121, 0.8),
              _label("NC-3", 800, 800, 0.7),          # inside no box: Sol only
              _label("P", 520, 510, 0.6),             # inside the block number's box only: still Sol only
              {**_label("RH-3", 120, 130), "kind": "height"}]  # not a use label: ignored
    rows = prints.match_prints(boxes, labels)
    assert [r["kind"] for r in rows] == [0, 1, 2, 2]
    a, b, c, d = rows
    assert (a["page_x"], a["page_y"]) == (150.0, 120.0) and a["box"] == (100, 100, 200, 140)
    assert (a["terra_code"], a["sol_code"], a["sol_confidence"]) == ("RH-1", "RH-1", 0.8)
    assert (b["terra_code"], b["sol_code"], b["box"]) == ("RM-2", "", (300, 100, 380, 140))
    assert math.isnan(b["sol_confidence"])
    # the farther point inside the first box is used up too: it makes no print of its own
    assert (c["page_x"], c["page_y"], c["sol_code"], c["terra_code"], c["box"]) == (800.0, 800.0, "NC-3", "", None)
    assert (d["sol_code"], d["sol_confidence"]) == ("P", 0.6)


def _print(x, y, box=None, kind=0):
    return {"page_x": float(x), "page_y": float(y), "box": box, "kind": kind,
            "turned_code": "", "turned_edge": -math.inf}


def test_turned_box_joins_by_point_or_iou():
    by_point = _print(1000, 1000, kind=2)                     # Sol only: joins only by its point
    by_iou = _print(50, 10, box=(0, 0, 100, 20))              # Terra box (0, 0)-(100, 20)
    alone = _print(400, 400, box=(380, 390, 420, 410))
    on_edge = _print(2000, 2000, kind=2)                      # a point on a box's edge is inside it
    turned = [
        _box("RM-2", 1950, 1960, 2000, 2040, edge=20),        # on_edge's point is on its right edge
        _box("RH-2", 990, 960, 1010, 1040, edge=50),          # holds by_point's point
        _box("RH-3", 55, 0, 120, 20, edge=40),                # IoU 900 / 2400 = 0.375 with by_iou's box, point outside
        _box("RM-1", 300, 380, 395, 420, edge=500),           # IoU 300 / 4300 = 0.07 with alone's box, point outside
        _box("RH-1", 980, 950, 1020, 1050, edge=80),          # holds by_point's point too, farther from its tile's edge
        _box("RH-2", 985, 955, 1015, 1045, edge=60),          # and again, nearer the edge than the last: it loses
    ]
    new = prints.join_turned([by_point, by_iou, alone, on_edge], turned, codes_known={"RH-1", "RH-2", "RH-3"})
    assert (on_edge["turned_code"], on_edge["turned_edge"]) == ("RM-2", 20)
    assert (by_point["turned_code"], by_point["turned_edge"]) == ("RH-1", 80)
    assert (by_iou["turned_code"], by_iou["turned_edge"]) == ("RH-3", 40)
    assert (alone["turned_code"], alone["turned_edge"]) == ("", -math.inf)
    assert new == []                                          # RM-1 joins nothing, and is not a known code


def test_unmatched_turned_box_with_a_code_becomes_a_print():
    have = _print(50, 10, box=(0, 0, 100, 20))
    turned = [_box("rh2", 500, 500, 520, 580, edge=70),       # a known code on no print: a new print
              _box("XQ-9", 600, 500, 620, 580, edge=70),      # not a code anyone uses: dropped
              _box("ELM", 700, 500, 720, 580, kind="street name", edge=70),
              _box("RH-2", 40, 0, 60, 20, edge=30)]           # on the print we have: a reading, not a print
    new = prints.join_turned([have], turned, codes_known={"RH-2"})
    assert len(new) == 1
    p = new[0]
    assert (p["kind"], p["page_x"], p["page_y"], p["turned_code"], p["turned_edge"]) == (3, 510.0, 540.0, "RH-2", 70)
    assert (p["box"], p["terra_code"], p["sol_code"]) == (None, "", "")
    assert (have["turned_code"], have["turned_edge"]) == ("RH-2", 30)


class CodeRead(BaseModel):   # read3.py's answer format, copied here apart from the module (its name is in the key too)
    code: str
    confidence: float
    leader: bool


def _page():
    """A 1-bit page with a few black bars to crop."""
    pg = np.full((900, 700), 255, np.uint8)
    pg[300:330, 200:300] = 0
    pg[600:700, 500:520] = 0
    pg[100:120, 50:90] = 0
    return pg


def _crop_as_read3(pg, box):
    """The crop read3.py made around a page box, and its box in the crop's px (floats)."""
    bx0, by0, bx1, by1 = box
    sz = max(bx1 - bx0, by1 - by0)
    cx, cy, half = (bx0 + bx1) / 2, (by0 + by1) / 2, sz * 1.5
    r0, r1 = int(max(0, cy - half)), int(min(pg.shape[0], cy + half))
    c0, c1 = int(max(0, cx - half)), int(min(pg.shape[1], cx + half))
    crop = pg[r0:r1, c0:c1]
    z = max(1, int(np.ceil(640 / max(crop.shape))))
    z = min(z, max(1, llm.MAX_EDGE // max(crop.shape)))
    if z > 1:
        crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    return crop, ((bx0 - c0) * z, (by0 - r0) * z, (bx1 - c0) * z, (by1 - r0) * z)


def _key(crop, prompt):
    return f"llm/{llm._key(llm._encode(crop), prompt, CodeRead)}.json"


def _write_answers(root, read, records):
    path = root / f"{read}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _prompt_as_read3(crop, x0, y0, x1, y1):
    return prints.THIRD_PROMPT.format(w=crop.shape[1], h=crop.shape[0], x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1))


def _on(sheet_print, terra, sol):
    return {**sheet_print, "sheet": SHEET, "terra_code": terra, "sol_code": sol}


def test_third_read_found_by_recomputed_key(tmp_path, monkeypatch):
    pg = _page()
    upright = _on(_print(250, 315, box=(200, 300, 300, 330)), "RH-2", "RH-2")
    sideways = _on(_print(510, 650, box=(500, 600, 520, 700)), "RM-1", "")
    point = _on(_print(70, 110, kind=2), "", "P")
    # the median code size of the prints with a box is (100 + 100) / 2 = 100: a point gets a box 100 wide, 50 high
    want = {}
    for name, box in (("upright", upright["box"]), ("sideways", sideways["box"]),
                      ("point", (20.0, 85.0, 120.0, 135.0))):
        crop, cbox = _crop_as_read3(pg, box)
        want[name] = crop, cbox, _prompt_as_read3(crop, *cbox)
    third = [{"cache_key": _key(want[n][0], want[n][2]), "answer": repr(a)} for n, a in (
        ("upright", {"code": "rh2", "confidence": 0.9, "leader": False}),
        ("sideways", {"code": "RM-1", "confidence": 0.6, "leader": True}),
        ("point", {"code": "P", "confidence": 0.7, "leader": False}))]
    # the sideways code is read again turned both ways; the surer answer wins
    crop, (X0, Y0, X1, Y1), _ = want["sideways"]
    h, w = crop.shape
    recheck = []
    for rot, f, ans in (
            (cv2.ROTATE_90_CLOCKWISE, lambda x, y: (h - 1 - y, x), {"code": "RM-2", "confidence": 0.5, "leader": False}),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y: (y, w - 1 - x),
             {"code": "RM-3", "confidence": 0.8, "leader": False})):
        turned = cv2.rotate(crop, rot)
        (xa, ya), (xb, yb) = f(X0, Y0), f(X1, Y1)
        prompt = _prompt_as_read3(turned, min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)) + prints.TURNED_NOTE
        recheck.append({"cache_key": _key(turned, prompt), "answer": repr(ans)})
    monkeypatch.setattr(saved.paths, "SAVED_ANSWERS", tmp_path)
    _write_answers(tmp_path, prints.THIRD_READ, third)
    _write_answers(tmp_path, prints.RECHECK_READ, recheck)
    rows = [upright, sideways, point]
    counts = prints.third_reads(rows, {SHEET: pg}, saved.by_key(prints.THIRD_READ),
                                saved.by_key(prints.RECHECK_READ))
    assert [(r["third_code"], r["third_confidence"], r["third_leader"], r["third_found"]) for r in rows] == [
        ("RH-2", 0.9, False, True), ("RM-1", 0.6, True, True), ("P", 0.7, False, True)]
    assert [r["recheck_asked"] for r in rows] == [False, True, False]
    assert (sideways["recheck_code"], sideways["recheck_confidence"]) == ("RM-3", 0.8) and sideways["recheck_found"]
    assert upright["recheck_code"] == "" and math.isnan(upright["recheck_confidence"])
    assert not upright["recheck_found"]
    assert counts == {"third_read": {"asked": 3, "found": 3, "missing": 0, "saved": 3, "unused": 0},
                      "recheck": {"asked": 2, "found": 2, "missing": 0, "saved": 2, "unused": 0}}


def test_missing_third_read_is_counted_not_guessed():
    pg = _page()
    a = _on(_print(250, 315, box=(200, 300, 300, 330)), "RH-2", "RH-2")
    b = _on(_print(70, 110, box=(50, 100, 90, 120)), "P", "P")
    crop, cbox = _crop_as_read3(pg, a["box"])
    prompt = _prompt_as_read3(crop, *cbox)
    other = _key(np.zeros((8, 8), np.uint8), "not a third read")
    table = {_key(crop, prompt): {"answer": {"code": "RH-2", "confidence": 0.9, "leader": False}},
             other: {"answer": {"code": "C-2", "confidence": 1.0, "leader": False}}}
    counts = prints.third_reads([a, b], {SHEET: pg}, table, {})
    assert (a["third_code"], a["third_found"]) == ("RH-2", True)
    assert (b["third_code"], b["third_found"], b["third_leader"]) == ("", False, False)
    assert math.isnan(b["third_confidence"])
    assert counts["third_read"] == {"asked": 2, "found": 1, "missing": 1, "saved": 2, "unused": 1}
    with pytest.raises(prints.SavedReadsMissing, match="third_read: 1 of 2"):
        prints.check_key_match(counts)


def _tile(px, py, items, window="0,0,3300,5100", size=768):
    return {"series": "2000-digital-zoning", "sheet": SHEET, "page_box": [px, py, px + size, py + size],
            "answer": {"items": items},
            "script": {"args": ["gpt-6.1-terra", "2000-digital-zoning", SHEET, window, "<out>"]}}


def _item(text, x0, y0, x1, y1, kind="zoning code"):
    """A Terra item as saved: its fields in alphabetical order."""
    return {"angle_deg": 0, "direction": "left-to-right", "kind": kind, "text": text, "x0": x0, "x1": x1, "y0": y0,
            "y1": y1}


def test_terra_decode_offsets_tiles():
    # the tiles are listed column by column; the script read them row by row, which decides ties in the edge distance
    records = [
        _tile(0, 0, [_item("RH-1", 700, 300, 760, 330)]),           # 8 px from this tile's right edge
        _tile(0, 600, [_item("NC-3", 640, 40, 700, 100),            # 40 px from this tile's top edge
                       _item("RM-2", 300, 100, 400, 140),           # a word inside the longer label beside it
                       _item("RM-2 NCD", 300, 100, 700, 140)]),
        _tile(0, 1200, [_item("RH-9", 100, 200, 150, 230)]),
        _tile(600, 0, [_item("RH-1", 100, 300, 160, 330),           # the same RH-1, 100 px from every edge: kept
                       _item("NC-3", 40, 640, 100, 700)]),          # the same NC-3, 40 px from this tile's left edge
        _tile(600, 600, [_item("ELM ST", 10, 10, 60, 20, kind="street name")]),
    ]
    got = prints.terra_boxes(SHEET, records)
    by_text = {}
    for b in got:
        by_text.setdefault(b["text"], []).append(b)
    assert sorted(by_text) == ["ELM ST", "NC-3", "RH-1", "RH-9", "RM-2 NCD"]
    rh1 = by_text["RH-1"]
    assert len(rh1) == 1 and rh1[0]["box"] == [700, 300, 760, 330] and rh1[0]["tile"] == [600, 0]
    assert rh1[0]["edge"] == 100 and rh1[0]["x0"] == 100
    assert list(rh1[0]) == ["text", "kind", "x0", "y0", "x1", "y1", "direction", "angle_deg", "box", "edge", "tile"]
    # NC-3 is 40 px from an edge in both tiles: tile (600, 0) comes first row by row, though (0, 600) is listed first
    assert len(by_text["NC-3"]) == 1 and by_text["NC-3"][0]["tile"] == [600, 0]
    assert by_text["NC-3"][0]["box"] == [640, 640, 700, 700]
    # the word RM-2 lies inside the longer label "RM-2 NCD" that contains it: it is part of that label
    assert by_text["RM-2 NCD"][0]["box"] == [300, 700, 700, 740]
    assert by_text["ELM ST"][0]["kind"] == "street name" and by_text["ELM ST"][0]["box"] == [610, 610, 660, 620]
    assert by_text["RH-9"][0]["box"] == [100, 1400, 150, 1430]
    # a box outside the window the script was asked about is dropped
    small = [_tile(0, 0, [_item("RH-1", 100, 100, 160, 130), _item("RM-1", 500, 500, 560, 530)], window="0,0,400,400")]
    assert [b["text"] for b in prints.terra_boxes(SHEET, small)] == ["RH-1"]


def test_saved_answers_read_literal_strings_and_key_by_cache_key(tmp_path, monkeypatch):
    monkeypatch.setattr(saved.paths, "SAVED_ANSWERS", tmp_path)
    _write_answers(tmp_path, "experiments-2000/x", [
        {"cache_key": "llm/aa.json", "sheet": SHEET, "answer": "{'code': 'RH-1', 'confidence': 0.5, 'leader': False}"},
        {"cache_key": "llm/bb.json", "sheet": "other", "answer": {"code": "P"}},
        {"sheet": SHEET, "text": "RH-1"}])
    got = saved.answers("experiments-2000/x")
    assert got[0]["answer"] == {"code": "RH-1", "confidence": 0.5, "leader": False}
    assert got[1]["answer"] == {"code": "P"} and "answer" not in got[2]
    assert [r.get("text") for r in saved.by_sheet("experiments-2000/x", SHEET)] == [None, "RH-1"]
    assert set(saved.by_key("experiments-2000/x")) == {"llm/aa.json", "llm/bb.json"}
    assert saved.cache_key("aa") == "llm/aa.json"


def test_lot_under_a_print_is_the_grid_pixel_holding_it():
    lot = np.full((6, 8), -1, np.int32)
    lot[2, 3] = 7
    grown = np.full((6, 8), 7, np.int32)
    grown[4, 5] = 9
    mapped = np.ones((6, 8), bool)
    mapped[0, 0] = False
    # map points inside grid px (3, 2), (5, 4) and (0, 0), and one off the grid
    xy = grid.to_map(np.array([[3.9, 2.1], [5.5, 4.5], [0.5, 0.5], [-0.5, 1.0]]))
    on, lab, street = prints.under(xy, lot, grown, mapped)
    assert on.tolist() == [True, True, False, False]
    assert lab[:2].tolist() == [7, 9] and street[:2].tolist() == [False, True]
