import math

import cv2
import numpy as np
import pandas as pd

from sfzmaps.solver.codereader import labels

SHEET = "2000-digital-zoning-p01"


def _prints(rows):
    """A prints table (prints.parquet's columns that the labels read) from short rows."""
    out = []
    for r in rows:
        box = r.get("box") or (math.nan,) * 4
        x, y = r.get("point") or ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        out.append({"sheet": SHEET, "page_x": float(x), "page_y": float(y), "kind": r["kind"],
                    "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3],
                    "terra_code": r.get("terra", ""), "sol_code": r.get("sol", ""), "turned_code": "",
                    "third_code": "", "recheck_code": "", "lab_lot": r["lot"]})
    return pd.DataFrame(out)


def _case(cid, x, y, code="RH-1", flag=False):
    return {"id": cid, "sid": "p01", "code": code, "x": float(x), "y": float(y), "flag": flag}


def test_training_excludes_answered_labels_and_their_blocks():
    lot_block = np.array(["100", "100", "200", "300", "400", "500", "600"])
    prints = _prints([
        {"kind": 0, "box": (100, 100, 180, 130), "lot": 0, "terra": "RH-1"},   # 0: an answered case in its box
        {"kind": 0, "box": (400, 100, 480, 130), "lot": 1, "terra": "RH-2"},   # 1: same block as 0
        {"kind": 2, "point": (700, 700), "lot": 2, "sol": "P"},                # 2: "no label" at its very point
        {"kind": 1, "box": (1000, 1000, 1060, 1030), "lot": 3},                # 3: in an off-print case's block
        {"kind": 0, "box": (1500, 100, 1580, 130), "lot": 4},                  # 4: only an unanswered case in it
        {"kind": 2, "point": (3000, 3000), "lot": 5},                          # 5: "unsure" at its point
        {"kind": 2, "point": (3500, 3500), "lot": 5},                          # 6: same block as 5
        {"kind": 1, "box": (2000, 100, 2060, 130), "lot": 6},                  # 7: apart from every answer
    ])
    published = [_case("a", 150, 120), _case("b", 700, 700, "P", flag=True), _case("c", 2500, 2500, "RM-1"),
                 _case("d", 1510, 110), _case("e", 3000, 3000), _case("f", 701, 700)]   # f: no box, no equal point
    answers = {"a": {"answer": "ok"}, "b": {"answer": "nolabel"}, "c": {"answer": "fix", "code": "rm2"},
               "e": {"answer": "unsure"}}
    cases = labels.cases_from(published, answers)
    labs = labels.make_labels(prints, {}, {}, cases, lot_block, case_lots={"a": 0, "b": 2, "c": 3, "e": 5})
    by_print = {lab.index: lab for lab in labs if lab.index >= 0}
    alone = [lab for lab in labs if lab.index < 0]

    # the join: a case inside a Terra box, or at a Sol-only print's point; no other
    assert by_print[0].cases == ("a",) and by_print[0].answer == "RH-1"
    assert by_print[2].cases == ("b",) and by_print[2].answer == ""            # "no label": no code there
    assert by_print[4].cases == ("d",) and not by_print[4].held_out and by_print[4].answer is None
    assert by_print[5].held_out and by_print[5].answer is None                 # unsure: held out, no code
    # an answered case on no print is a label of its own, at its point, with Sol's read and its owner's code
    assert len(alone) == 1 and alone[0].cases == ("c",) and alone[0].answer == "RM-2"
    assert alone[0].point == (2500.0, 2500.0) and alone[0].reads["sol"] == "RM-1" and alone[0].held_out

    train = labels.training(labs)
    assert sorted(lab.index for lab in train) == [4, 7]
    held = labels.held_out_blocks(labs)
    assert held == {"100", "200", "300", "500"}
    assert not any(lab.held_out or lab.block in held for lab in train)


def test_folds_split_each_stratum_in_half_with_a_seed():
    lot_block = np.array([str(i) for i in range(40)])
    rows = [{"kind": 2, "point": (100 * i, 100), "lot": i} for i in range(40)]
    published = [_case(f"c{i}", 100 * i, 100, flag=i < 11) for i in range(40)]
    answers = {f"c{i}": {"answer": "nolabel" if 30 <= i < 35 else "ok"} for i in range(37)}
    cases = labels.cases_from(published, answers)
    one = labels.make_labels(_prints(rows), {}, {}, cases, lot_block, case_lots={})
    two = labels.make_labels(_prints(rows), {}, {}, cases, lot_block, case_lots={})
    assert [lab.fold for lab in one] == [lab.fold for lab in two]       # seeded: the same split every time
    assert [lab.fold for lab in one[37:]] == [-1, -1, -1]               # unanswered: no fold
    for stratum in (range(11), [*range(11, 30), 35, 36], range(30, 35)):     # flagged; unflagged codes; no label
        folds = [one[i].fold for i in stratum]
        assert set(folds) <= {0, 1} and abs(folds.count(0) - folds.count(1)) <= 1


def _label(index, block, terra="", sol="", held_out=False, **kw):
    return labels.Label(index=index, sheet=SHEET, kind=0, point=(0.0, 0.0), terra_text=terra, sol_text=sol,
                        block=block, held_out=held_out, **kw)


def test_alias_maps_to_its_code():
    names = ["24TH-NOE VALLEY", "VALENCIA", "RH-2", "POLK", "RM-1"]
    labs = [_label(0, "1", terra="VALENCIA NCD", sol="RH2"),         # training reads: their printed forms join
            _label(1, "2", terra="NCD", sol="RM-1"),                  # a stray word that names no code stays out
            _label(2, "3", terra="POLK NCD", held_out=True)]          # seen only on an answered label: left out
    got = labels.code_list(labs, names)
    assert all(got[n] == n for n in names)
    assert got["NOE VALLEY"] == "24TH-NOE VALLEY"                     # the printed name, not "other"
    assert got["VALENCIA NCD"] == "VALENCIA" and got["RH2"] == "RH-2"
    assert "NCD" not in got and "POLK NCD" not in got
    assert labels.code_of("Noe  Valley NCD", names) == "24TH-NOE VALLEY"
    assert labels.code_of("NORTH OF MARKET", names) == ""


def _word(text="RH-2"):
    """A word's ink, read left to right, cut to its ink."""
    canvas = np.zeros((40, 120), np.uint8)
    cv2.putText(canvas, text, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_8)
    ys, xs = np.nonzero(canvas)
    return canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0


def test_crop_is_straightened_by_the_box_angle():
    word = _word()
    h, w = word.shape
    sizes = labels.Sizes(height=10.0, length=60.0)                    # a margin of REACH x 10 px
    m = sizes.margin
    # quarter turns: the crop is the page's px, turned exactly so the word reads left to right
    for angle, direction, turns in ((0, "left-to-right", 0), (90, "bottom-to-top", 1), (-90, "top-to-bottom", -1),
                                    (180, "upside-down", 2)):
        printed = np.rot90(word, turns)
        page = np.full((400, 500), 255, np.uint8)
        x0, y0 = 151, 170
        page[y0:y0 + printed.shape[0], x0:x0 + printed.shape[1]][printed] = 0
        box = (x0, y0, x0 + printed.shape[1], y0 + printed.shape[0])
        lab = _label(0, "1", box=box, angle=float(angle), direction=direction)
        labels.place_crop(lab, sizes, page)
        assert lab.crop.shape == (h + 2 * m, w + 2 * m)
        assert np.array_equal(lab.crop < 128, np.pad(word, m)), angle
        assert lab.code_rect == (m, m, m + w, m + h) and lab.centre == ((w + 2 * m) / 2, (h + 2 * m) / 2)
        cu, cv = (lab.crop.shape[1] - 1) / 2, (lab.crop.shape[0] - 1) / 2
        assert np.allclose(lab.to_page @ [cu, cv, 1], [(box[0] + box[2] - 1) / 2, (box[1] + box[3] - 1) / 2])
    # any other angle: resampled (bilinear) about the box's centre, the word upright again
    angle = 30.0
    pad = np.pad(word, 60).astype(np.uint8) * 255
    ph, pw = pad.shape
    turned = cv2.warpAffine(pad, cv2.getRotationMatrix2D(((pw - 1) / 2, (ph - 1) / 2), angle, 1.0), (pw, ph),
                            flags=cv2.INTER_LINEAR) >= 128
    page = np.full((400, 500), 255, np.uint8)
    page[100:100 + ph, 120:120 + pw][turned] = 0
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    bw, bh = w * c + h * s, w * s + h * c                              # the turned word's axis box
    cx, cy = 120 + pw / 2, 100 + ph / 2
    box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    lab = _label(0, "1", box=box, angle=angle, direction="slanted")
    labels.place_crop(lab, sizes, page)
    u0, v0, u1, v1 = (round(v) for v in lab.code_rect)
    got = lab.crop[v0:v1, u0:u1] < 128
    assert got.shape == word.shape
    assert (got & word).sum() / (got | word).sum() > 0.7
    assert not (lab.crop < 128)[:v0 - 3].any() and not (lab.crop < 128)[v1 + 3:].any()


def test_point_crop_is_a_square_window_on_the_point():
    page = np.full((400, 500), 255, np.uint8)
    page[200, 300] = 0
    lab = labels.Label(index=0, sheet=SHEET, kind=2, point=(300.0, 200.0))     # Sol's point alone, no box
    sizes = labels.Sizes(height=10.0, length=60.0)
    labels.place_crop(lab, sizes, page)
    half = sizes.point_half
    assert lab.crop.shape == (2 * half, 2 * half) and lab.code_rect is None
    assert lab.centre == (half, half) and (lab.crop[half, half] < 128) and (lab.crop < 128).sum() == 1


def test_line_mask_is_dilated_and_follows_the_crop():
    word = _word()
    printed = np.rot90(word, 1)                                       # read bottom to top
    page = np.full((400, 500), 255, np.uint8)
    x0, y0 = 151, 170
    page[y0:y0 + printed.shape[0], x0:x0 + printed.shape[1]][printed] = 0
    box = (x0, y0, x0 + printed.shape[1], y0 + printed.shape[0])
    lab = _label(0, "1", box=box, angle=90.0, direction="bottom-to-top")
    labels.place_crop(lab, labels.Sizes(height=10.0, length=60.0), page)
    lines = np.zeros(page.shape, bool)
    lines[y0 + 10, :] = True                                          # a one px line across the page, through the word
    mask = labels.line_mask(lab, lines)
    assert mask.shape == lab.crop.shape
    # on the page the line runs left to right; turned with the word, it runs down the crop, three px wide
    cols = np.flatnonzero(mask.any(axis=0))
    assert len(cols) == 3 and cols[2] - cols[0] == 2 and mask[:, cols].all()
    u = lab.to_page @ np.c_[cols, np.zeros(3), np.ones(3)].T
    assert np.allclose(sorted(u[1]), [y0 + 9, y0 + 10, y0 + 11])


def _broken_line_page():
    """A page (4 ft a px) with a 3 px line 320 px long crossed every 30 px by short ticks (so junctions cut it into
    strokes of 30 px, 120 ft), a heavy bar 8 px wide and 30 px long, and a letter H 27 px tall."""
    ink = np.zeros((300, 400), bool)
    ink[99:102, 40:360] = True
    for x in range(55, 360, 30):
        ink[90:111, x - 1:x + 2] = True
    bar = np.zeros_like(ink)
    bar[196:204, 40:70] = True
    letter = np.zeros_like(ink)
    letter[180:207, 200:203] = letter[180:207, 215:218] = letter[192:195, 200:218] = True
    return ink | bar | letter, bar, letter


def test_run_rule_keeps_lines_that_junctions_break():
    from sfzmaps.solver.textnet import scores
    ink, _, _ = _broken_line_page()
    line = np.zeros_like(ink)
    line[99:102, 40:360] = True
    long = scores.long_lines(ink, 4.0)
    run = labels.run_lines(ink, labels.run_px(4.0))
    assert long[line].mean() < 0.05                       # the detector's long lines: cut into short strokes
    assert run[line].mean() > 0.95                        # a straight run of 200 ft goes through the junctions
    assert not (run & ~line).any()                        # the ticks, a letter's strokes and a short bar are no runs


def test_width_rule_keeps_heavy_strokes_of_any_length():
    from sfzmaps.solver.textnet import scores
    ink, bar, letter = _broken_line_page()
    width = labels.width_lines(ink, 4.0)
    assert width[bar].mean() > 0.95 and not scores.long_lines(ink, 4.0)[bar].any()   # 32 ft wide, 120 ft long
    assert not width[letter].any()                        # 12 ft strokes: not heavy
    assert (width & scores.long_lines(ink, 4.0)).sum() == scores.long_lines(ink, 4.0).sum()   # the long lines too


def test_line_mask_unites_the_rules_asked_for(monkeypatch):
    page = np.full((400, 500), 255, np.uint8)
    lab = labels.Label(index=0, sheet=SHEET, kind=2, point=(250.0, 200.0))
    labels.place_crop(lab, labels.Sizes(height=10.0, length=60.0), page)
    a, b = np.zeros(page.shape, bool), np.zeros(page.shape, bool)
    a[200, :], b[:, 250] = True, True
    pages = {"long": a, "run": b}
    monkeypatch.setattr(labels, "sheet_lines", lambda sheet, rule: (np.packbits(pages[rule], axis=1), *page.shape))
    one, both = labels.line_mask(lab), labels.line_mask(lab, rules=("long", "run"))
    assert np.array_equal(one, labels.line_mask(lab, a))
    assert np.array_equal(both, one | labels.line_mask(lab, b))
