import dataclasses
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver.codereader import glyphs, labels, posterior

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
    # each label knows whether its block is held out, so a list to learn from can be checked on its own: the block
    # mates of answered labels (1: with 0; 3: an off-print case's lot; 6: with 5) are refused
    assert [lab.index for lab in labs if lab.held_block and not lab.held_out] == [1, 3, 6]
    labels.refuse_held(train)
    with pytest.raises(ValueError, match="held-out block"):
        labels.refuse_held([by_print[1]])


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


# --- the glyph templates and the noise --------------------------------------------------------------------------------

WORDS = ("RH-1", "RH-2", "RM-1", "RM-2", "M-1", "M-2", "RH-12", "H-21")
TRUE_NOISE = glyphs.NoiseModel(0.04 + 0.92 / (1 + np.exp(-glyphs.KNOTS / 0.6)))   # Kanungo-like: flips near edges
SMALL = {"residuals": (-3.0, 0.0, 3.0), "ratios": (0.85, 0.95, 1.05, 1.15)}      # a small pose grid, for speed


def _true_font():
    """The known font the synthetic crops are drawn from: Hershey Complex (serifs; the first font is a sans), a
    heavier stroke, leaning 12 degrees, wider spacing, crisp templates."""
    font = glyphs.initial_font("RHM-12", slant=12.0, face=cv2.FONT_HERSHEY_COMPLEX, stroke=0.22)
    font.templates = {c: np.where(t >= 0.5, 0.97, 0.03).astype(np.float32) for c, t in font.templates.items()}
    font.advance = {c: 1.1 * a for c, a in font.advance.items()}
    return font


def _synthetic(n, seed, true=None, misplaced=()):
    """n crops of codes drawn from the true font with TRUE_NOISE: caps 20-28 px, a few degrees off level, every fifth
    upside down (as Terra's top-to-bottom labels come out), every third crossed by a heavy line whose px (grown by
    one) are unobserved.  Returns the labels (Terra's word rectangle on the drawn ink, or 50 px beside it for those
    `misplaced`), strings, observed px and the true poses."""
    true = true or _true_font()
    rng = np.random.default_rng(seed)
    shape = (96, 176)
    labs, texts, observed, poses = [], [], [], []
    for i in range(n):
        text = WORDS[i % len(WORDS)]
        angle = (180.0 if i % 5 == 4 else 0.0) + rng.uniform(-2.5, 2.5)
        pose = glyphs.Pose(88 + rng.uniform(-4, 4), 48 + rng.uniform(-4, 4), rng.uniform(20, 28) / glyphs.SIZE, angle)
        d = glyphs.render(true, text, pose, shape)
        ink = rng.random(shape) < TRUE_NOISE.p_ink(d)
        obs = np.ones(shape, bool)
        if i % 3 == 0:
            bar = np.zeros(shape, np.uint8)
            y = round(pose.y + rng.uniform(-8, 8))
            cv2.line(bar, (0, y), (shape[1] - 1, y + round(rng.uniform(-20, 20))), 1, 5)
            ink |= bar > 0
            obs &= cv2.dilate(bar, np.ones((3, 3), np.uint8)) == 0
        ys, xs = np.nonzero(d > 0)
        off = 50.0 if i in misplaced else 0.0
        labs.append(labels.Label(index=i, sheet=SHEET, kind=0, point=(0.0, 0.0),
                                 crop=np.where(ink, 0, 255).astype(np.uint8),
                                 code_rect=(xs.min() + off, float(ys.min()), xs.max() + 1 + off, float(ys.max() + 1))))
        texts.append(text)
        observed.append(obs)
        poses.append(pose)
    return labs, texts, observed, poses


def _shape_iou(a, b, char):
    """How well two fonts' shapes of a character agree, drawn alone at scale 1: the best IoU over shifts of 5 px."""
    pose = glyphs.Pose(40, 40, 1.0, 0.0)
    sa, sb = glyphs.render(a, char, pose, (80, 80)) > 0, glyphs.render(b, char, pose, (80, 80)) > 0
    best = 0.0
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            m = np.roll(sb, (dy, dx), axis=(0, 1))
            best = max(best, (sa & m).sum() / (sa | m).sum())
    return best


def test_em_recovers_templates_from_synthetic_text(tmp_path):
    true = _true_font()
    labs, texts, observed, poses = _synthetic(48, seed=7, true=true, misplaced=(5, 17))
    font = glyphs.learn_font(labs, 4, texts=texts, observed=observed, **SMALL)
    first = glyphs.initial_font("RHM-12", font.slant)
    for c in "RHM-12":
        got, before = _shape_iou(font, true, c), _shape_iou(first, true, c)
        assert got > 0.8 and got > before + 0.05, (c, got, before)       # the serif font, learned from a sans
    # the noise: the background's ink, and flips near the edges, as drawn
    assert abs(font.noise.background - 0.04) < 0.015
    p = font.noise.p_ink(np.array([-2.0, 0.0, 2.0]))
    want = TRUE_NOISE.p_ink(np.array([-2.0, 0.0, 2.0]))
    assert np.all(np.abs(p - want) < 0.12), (p, want)
    # the rounds explain the ink better than the first font did, and the font returned is the best round's; every
    # label is kept but the two whose word was found far from where Terra's box put it (the "mislabelled" escape)
    rounds = [r for r in font.trace if isinstance(r["round"], int)]
    chosen = font.trace[-1]["chosen"]
    assert rounds[chosen]["loglik"] == max(r["loglik"] for r in rounds[1:]) > rounds[0]["loglik"]
    assert abs(rounds[-1]["kept"] - (len(labs) - 2)) < 0.5
    weights = np.array(rounds[-1]["weights"])
    assert weights[[5, 17]].max() < 0.05 and np.delete(weights, [5, 17]).min() > 0.9
    # each word, templates and spacing together, found and drawn where the true font drew it (to the small grid's
    # scale steps of 10%); the upside-down crops aligned upside down (the half turn is part of the pose)
    ious = []
    for lab, text, obs, pose in zip(labs, texts, observed, poses):
        fit = glyphs.align(font, font.noise, text, lab.ink, obs, glyphs.search_for(lab, np.array([24.0]), **SMALL))
        assert abs((fit.pose.angle - pose.angle + 180) % 360 - 180) < 5
        assert math.hypot(fit.pose.x - pose.x, fit.pose.y - pose.y) < 2.5
        got = glyphs.render(font, text, fit.pose, lab.crop.shape) > 0
        want = glyphs.render(true, text, pose, lab.crop.shape) > 0
        ious.append((got & want).sum() / (got | want).sum())
    assert min(ious) > 0.65 and np.median(ious) > 0.8
    # the font, its noise and its measured position spread come back from disk as they were
    font.save(tmp_path / "font.npz")
    back = glyphs.Font.load(tmp_path / "font.npz")
    assert back.chars == font.chars and all(np.array_equal(back.templates[c], font.templates[c]) for c in "RHM-12")
    assert back.advance == font.advance and np.array_equal(back.noise.p, font.noise.p)
    assert 0 < back.position_sd == font.position_sd < glyphs.POSITION_SD
    pose = glyphs.Pose(80, 40, 0.8, 0.0)
    assert np.array_equal(glyphs.render(back, "RH-12", pose, (80, 160)), glyphs.render(font, "RH-12", pose, (80, 160)))


def test_label_weights_let_go_of_strings_found_far_from_their_label():
    rng = np.random.default_rng(1)
    z = np.vstack([rng.normal(0.05, 0.1, (60, 2)), [[2.5, 1.0], [-3.0, 0.5]]])     # two found elsewhere in the crop
    w, mix = glyphs.label_weights(z, np.full(len(z), 60.0))
    assert np.all(w[:60] > 0.95) and np.all(w[60:] < 0.01)
    assert abs(mix["bad_share"] - 2 / 62) < 0.02
    w, mix = glyphs.label_weights(z[:60], np.full(60, 60.0))                       # none: every label kept
    assert np.all(w > 0.95)


def test_unobserved_pixels_change_nothing():
    labs, texts, observed, poses = _synthetic(12, seed=3)
    rng = np.random.default_rng(0)
    scribbled = []
    for lab, obs in zip(labs, observed):
        crop = lab.crop.copy()
        crop[~obs] = np.where(rng.random((~obs).sum()) < 0.5, 0, 255)       # anything at all under the lines
        scribbled.append(labels.Label(index=lab.index, sheet=SHEET, kind=0, point=(0.0, 0.0), crop=crop,
                                      code_rect=lab.code_rect))
    assert any((a.crop != b.crop).any() for a, b in zip(labs, scribbled))
    # the likelihood reads the observed px only
    d = glyphs.render(_true_font(), texts[0], poses[0], labs[0].crop.shape)
    assert TRUE_NOISE.log_lik(d, labs[0].ink, observed[0]) == TRUE_NOISE.log_lik(d, scribbled[0].ink, observed[0])
    every = np.ones(labs[0].crop.shape, bool)
    assert TRUE_NOISE.log_lik(d, labs[0].ink, every) != TRUE_NOISE.log_lik(d, scribbled[0].ink, every)
    # and so does learning: the same font, noise and fits from crops that differ only under the lines
    a = glyphs.learn_font(labs, 1, texts=texts, observed=observed, **SMALL)
    b = glyphs.learn_font(scribbled, 1, texts=texts, observed=observed, **SMALL)
    assert sorted(a.templates) == sorted(b.templates)
    assert all(np.array_equal(a.templates[c], b.templates[c]) for c in a.templates)
    assert a.advance == b.advance and np.array_equal(a.noise.p, b.noise.p)
    assert [r.get("loglik") for r in a.trace] == [r.get("loglik") for r in b.trace]


@pytest.mark.parametrize("mark", ["held_out", "held_block"])
def test_learning_refuses_held_out_labels(mark):
    labs, texts, observed, _ = _synthetic(3, seed=1)
    setattr(labs[1], mark, True)            # an owner's answer joined it, or one joined a label in its block
    font = _reader_font("RHM-12")
    for learn in (lambda: glyphs.learn_font(labs, 1, texts=texts, observed=observed, **SMALL),
                  lambda: posterior.code_prior(labs, {"RH-1": "RH-1"}),
                  lambda: posterior.form_prior(labs, {"RH-1": "RH-1"}),
                  lambda: posterior.char_bigram(labs, "RH-1 ", {"RH-1": "RH-1"}),
                  lambda: posterior.measure_pose_prior(labs, font, TRUE_NOISE),
                  lambda: posterior.Prior.learn(labs, font, TRUE_NOISE, {"RH-1": "RH-1"})):
        with pytest.raises(ValueError, match="held-out"):
            learn()


# --- the posterior ----------------------------------------------------------------------------------------------------

READ_CODES = ("RH-1", "RH-2", "RM-1", "RM-2", "P", "M-1", "RH-1(D)")


def _reader_font(chars):
    """The font the posterior tests read with, and draw with: Hershey Simplex leaning 12 degrees, crisp templates, a
    gap spread of 2 template px."""
    font = glyphs.initial_font(chars, slant=12.0)
    font.templates = {c: np.where(t >= 0.5, 0.97, 0.03).astype(np.float32) for c, t in font.templates.items()}
    font.spacing_sd = 2.0
    return font


def _prior(font, forms, **pose):
    """A prior over `forms` (printed form -> code): codes alike, the bigram from no reads (smoothing alone), and a
    pose prior like the 2000 map's measured one, on a small grid.  The synthetic ink is drawn px by px independently
    given the rendering, so its temperature is small (the map's px are not: there it is measured)."""
    defaults = {"left": (0.0, 0.05, 0.3), "right": (0.0, 0.07, 0.4), "middle": (0.0, 0.03, 0.1), "box_tail": 0.15,
                "ratio": (0.0, 0.1), "half_turn": {"left-to-right": 0.01, "top-to-bottom": 0.6},
                "along": {"sol": (0.0, 0.1, 0.4)}, "across": {"sol": (0.0, 0.07, 0.2)}, "point_tail": {"sol": 0.3},
                "point_turns": {"sol": [0.7, 0.15, 0.05, 0.1]}, "residuals": [0.02, 0.1, 0.5, 0.3, 0.08],
                "cap": (math.log(24.0), 0.2), "height": 24.0, "point_heights": (20.0, 24.0, 28.0), "foreign": 1.2e-4,
                "temperature": 5.0, "ratios": (0.85, 0.95, 1.05, 1.15), "point_quantiles": (0.5, 0.5),
                "char_gain": {c: 1.5 for c in font.templates}}
    defaults.update(pose)
    codes = sorted(set(forms.values()))
    return posterior.Prior(codes={c: 1 / len(codes) for c in codes}, forms=dict(forms),
                           bigram=posterior.char_bigram([], font.chars, forms), chars=font.chars,
                           pose=posterior.PosePrior(**defaults))


def _draw(font, items, shape, seed):
    """Ink of texts drawn by the font at their poses with TRUE_NOISE, and the first one's ink box (px edges)."""
    rng = np.random.default_rng(seed)
    ink = np.zeros(shape, bool)
    box = None
    for text, pose in items:
        d = glyphs.render(font, text, pose, shape)
        ink |= rng.random(shape) < TRUE_NOISE.p_ink(d)
        if box is None:
            ys, xs = np.nonzero(d > 0)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    return ink, box


def _box_label(ink, box, direction="left-to-right"):
    """A label read by Terra's box `box` (the crop already reads along Terra's angle)."""
    return labels.Label(index=0, sheet=SHEET, kind=0, point=(0.0, 0.0), box=(0.0, 0.0, 1.0, 1.0),
                        direction=direction, crop=np.where(ink, 0, 255).astype(np.uint8), code_rect=box,
                        centre=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def _top(post):
    return max(post, key=post.get)


def _measured(font, n=60, seed=7):
    """What `measure_pose_prior` measures from the ink, measured on synthetic training labels drawn as the tests
    draw (the true poses as the fits): each character's gain and where a word ends.  Each label a code at a cap of
    19-27 px, a little off level; four in ten have another word on its line after it and two in ten before it, a word
    gap (0.5-1.5 cap heights) away."""
    rng = np.random.default_rng(seed)
    shape = (120, 420)
    texts, fits, labs = [], [], []
    for i in range(n):
        text = READ_CODES[i % len(READ_CODES)]
        s, angle = rng.uniform(0.6, 0.85), rng.normal(0.0, 1.0)
        pose = glyphs.Pose(210 + rng.uniform(-10, 10), 60 + rng.uniform(-4, 4), s, angle)
        items = [(text, pose)]
        d = glyphs.render(font, text, pose, shape)
        xs = np.flatnonzero((d > 0).any(axis=0))
        for side, p in ((1, 0.4), (-1, 0.2)):
            if rng.random() < p:
                other = READ_CODES[rng.integers(len(READ_CODES))]
                width = glyphs.render(font, other, glyphs.Pose(210, 60, s, angle), shape)
                ox = np.flatnonzero((width > 0).any(axis=0))
                gap = rng.uniform(0.5, 1.5) * s * glyphs.SIZE
                edge = xs[-1] + 1 + gap if side > 0 else xs[0] - gap
                centre = edge + side * (ox[-1] + 1 - ox[0]) / 2
                items.append((other, glyphs.Pose(centre, pose.y + (centre - pose.x) * math.tan(math.radians(-angle)),
                                                 s, angle)))
        ink, box = _draw(font, items, shape, seed=100 + i)
        texts.append(text)
        fits.append(type("Fit", (), {"pose": pose})())
        labs.append(_box_label(ink, box))
    observed = [np.ones(shape, bool)] * n
    weights = np.ones(n)
    return (posterior.char_gains(font, TRUE_NOISE, texts, fits, labs, observed, weights),
            posterior.end_breaks(font, texts, fits, labs, observed, weights))


@pytest.fixture(scope="module")
def reader():
    font = _reader_font("".join(sorted(set("".join(READ_CODES)) | set("SUD"))))
    gains, ends = _measured(font)
    return font, _prior(font, {c: c for c in READ_CODES}, char_gain=gains, end_break=ends)


def test_posterior_sums_to_one(reader):
    font, prior = reader
    ink, box = _draw(font, [("RM-2", glyphs.Pose(100, 55, 0.75, 0.0))], (110, 200), seed=2)
    post = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                 observed=np.ones(ink.shape, bool))
    assert set(post) == set(READ_CODES) | {"none", "other"}
    assert all(p >= 0 for p in post.values()) and abs(sum(post.values()) - 1) < 1e-9
    assert _top(post) == "RM-2" and post["RM-2"] > 0.95
    # Task 4's constants change the masses, and the posterior still sums to one
    other = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior,
                                  posterior.Constants(temperature=30.0, none_mass=0.3, other_mass=0.2),
                                  observed=np.ones(ink.shape, bool))
    assert abs(sum(other.values()) - 1) < 1e-9 and other["none"] > post["none"]


def test_line_through_h_does_not_make_it_m(reader):
    """Review Focus 2: a heavy line through the H, its px unobserved, scores RH-1 no worse than without the line."""
    font, prior = reader
    shape = (110, 200)
    ink, box = _draw(font, [("RH-1", glyphs.Pose(100, 55, 0.75, 0.0))], shape, seed=3)
    clean = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                  observed=np.ones(shape, bool))
    # a vertical bar 6 px wide through the middle of the H, the length of the crop: M's middle stroke
    _, pens = font.layout("RH-1")
    h_mid = round(100 + 0.75 * (pens[1] + font.advance["H"] / 2))
    bar = np.zeros(shape, np.uint8)
    bar[:, h_mid - 3:h_mid + 3] = 1
    crossed = ink | (bar > 0)
    unobserved = cv2.dilate(bar, np.ones((3, 3), np.uint8)) > 0
    masked = posterior.score_label(_box_label(crossed, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                   observed=~unobserved)
    # the line's ink neither helps nor hurts: exactly the clean label with the line's px unread
    unread = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                   observed=~unobserved)
    assert masked == unread
    assert _top(clean) == _top(masked) == "RH-1" and masked["RH-1"] > 0.8
    # the same line read as ink (not unobserved) is what pulls towards M
    seen = posterior.score_label(_box_label(crossed, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                 observed=np.ones(shape, bool))
    assert seen["RM-1"] / seen["RH-1"] > 3 * masked["RM-1"] / masked["RH-1"]


def test_sideways_code_reads(reader):
    """Review Focus 3: a code printed bottom-to-top at Sol's point (no box) reads right, and so does a label Terra
    read top-to-bottom whose crop therefore shows it upside down."""
    font, prior = reader
    shape = (220, 220)
    ink, _ = _draw(font, [("RM-2", glyphs.Pose(110, 112, 0.75, 90.0))], shape, seed=4)
    point = labels.Label(index=0, sheet=SHEET, kind=2, point=(0.0, 0.0), crop=np.where(ink, 0, 255).astype(np.uint8),
                         centre=(110.5, 113.0))
    post = posterior.score_label(point, font, TRUE_NOISE, prior, posterior.Constants(), observed=np.ones(shape, bool))
    assert _top(post) == "RM-2" and post["RM-2"] > 0.9
    ink, box = _draw(font, [("RH-2", glyphs.Pose(100, 55, 0.75, 180.0))], (110, 200), seed=5)
    post = posterior.score_label(_box_label(ink, box, "top-to-bottom"), font, TRUE_NOISE, prior,
                                 posterior.Constants(), observed=np.ones(ink.shape, bool))
    assert _top(post) == "RH-2" and post["RH-2"] > 0.9


def test_slanted_box_reads_whatever_its_rectangle_height(reader):
    """A slanted Terra box gives a word rectangle far thinner than the letters (its median on the map is 14 px
    against 27 for words read a quarter turn); the letters' size is then the map's, not the rectangle's."""
    font, prior = reader
    ink, box = _draw(font, [("RH-2", glyphs.Pose(100, 55, 0.75, 0.0))], (110, 200), seed=10)
    mid = (box[1] + box[3]) / 2
    lab = _box_label(ink, (box[0], mid - 3, box[2], mid + 3), "slanted")
    lab.angle = 30.0
    post = posterior.score_label(lab, font, TRUE_NOISE, prior, posterior.Constants(), observed=np.ones(ink.shape, bool))
    assert _top(post) == "RH-2" and post["RH-2"] > 0.9


def test_lines_only_window_prefers_none(reader):
    """Review Focus 4: a window of lines (unobserved) and background specks holds no code."""
    font, prior = reader
    shape = (110, 200)
    rng = np.random.default_rng(6)
    ink = rng.random(shape) < TRUE_NOISE.background
    lines = np.zeros(shape, np.uint8)
    cv2.line(lines, (0, 40), (199, 70), 1, 6)
    cv2.line(lines, (120, 0), (110, 109), 1, 5)
    ink |= lines > 0
    observed = cv2.dilate(lines, np.ones((3, 3), np.uint8)) == 0
    post = posterior.score_label(_box_label(ink, (70.0, 43.0, 130.0, 67.0)), font, TRUE_NOISE, prior,
                                 posterior.Constants(), observed=observed)
    assert _top(post) == "none" and post["none"] > 0.9


def test_code_is_not_chosen_for_covering_more_ink(reader):
    """A small code at Terra's word with larger codes printed beside it (the next line, and further along the same
    line) reads as itself: other text explains the neighbours' ink, so moving onto more ink gains nothing."""
    font, prior = reader
    shape = (150, 260)
    ink, box = _draw(font, [("P", glyphs.Pose(110, 60, 0.75, 0.0)), ("RH-1(D)", glyphs.Pose(130, 105, 0.85, 0.0)),
                            ("RM-2", glyphs.Pose(205, 60, 0.85, 0.0))], shape, seed=7)
    post = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                 observed=np.ones(shape, bool))
    assert _top(post) == "P" and post["P"] > 0.9


def test_text_that_is_no_code_reads_other(reader):
    font, prior = reader
    ink, box = _draw(font, [("SUD", glyphs.Pose(100, 55, 0.75, 0.0))], (110, 200), seed=8)
    post = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                 observed=np.ones(ink.shape, bool))
    assert _top(post) == "other" and post["other"] > 0.9


@pytest.mark.parametrize("code", [*READ_CODES, "M-2"])
def test_crisp_code_on_clean_paper_reads_that_code_not_other(reader, code):
    """A printed code alone on paper, at Terra's word: the code, not "other" spelling it (or it with a filler)."""
    font, prior = reader
    prior = dataclasses.replace(prior, forms={**prior.forms, "M-2": "M-2"},
                                codes={c: 1 / (len(READ_CODES) + 1) for c in [*READ_CODES, "M-2"]})
    ink, box = _draw(font, [(code, glyphs.Pose(110, 55, 0.75, 0.0))], (110, 230), seed=11)
    post = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, observed=np.ones(ink.shape, bool))
    assert post[code] > 0.9, (code, post)


def test_temperature_grid_gives_each_temperature_as_alone(reader):
    """Evidence at a grid of temperatures is each temperature's alone (nothing pruned by one temperature's scores
    or shared between them): on a box label and on a point label whose text sits off its point (the placement's tail)."""
    font, prior = reader
    ink, box = _draw(font, [("RM-2", glyphs.Pose(100, 55, 0.75, 0.0))], (110, 200), seed=2)
    off, _ = _draw(font, [("RM-2", glyphs.Pose(150, 112, 0.75, 0.0))], (220, 260), seed=4)
    point = labels.Label(index=0, sheet=SHEET, kind=2, point=(0.0, 0.0), crop=np.where(off, 0, 255).astype(np.uint8),
                         centre=(126.5, 113.0))
    for lab in (_box_label(ink, box), point):
        obs = np.ones(lab.crop.shape, bool)
        grid = posterior.evidence(lab, font, TRUE_NOISE, prior, (5.0, 30.0), observed=obs)
        for i, temp in enumerate((5.0, 30.0)):
            alone = posterior.evidence(lab, font, TRUE_NOISE, prior, (temp,), observed=obs)
            assert abs(alone.other[0] - grid.other[i]) < 1e-9
            assert max(abs(alone.forms[f][0] - grid.forms[f][i]) for f in grid.forms) < 1e-9


@pytest.mark.parametrize("off", [
    -0.5, -0.25, 0.25, 0.5, 0.75, 1.0,
    pytest.param(-0.75, marks=pytest.mark.xfail(strict=True, reason=(
        "measured: 'other' 0.666 (RH-1(D) 0.267): a string over the part nearest the point pays one end's factor, "
        "less than the full code's placement tail costs"))),
    pytest.param(-1.0, marks=pytest.mark.xfail(strict=True, reason=(
        "measured: RH-1 0.901: the prefix centred near the point pays one end (~3 nats on the map's table), the full "
        "code's centre is 2.5 tail spreads off")))])
def test_off_centre_point_reads_the_full_code_with_the_end_term(reader, monkeypatch, off):
    """Without a box, the print's point off the middle of its text (word heights along the line; negative: toward
    "RH"): with the end-break term on (`END_BREAKS`; measured here on the synthetic training labels), a reading that
    stops short of the ink pays for the ink left beside its end.  The term is off in the reader: on the map's
    training point labels it lowered the likelihood (see the module)."""
    monkeypatch.setattr(posterior, "END_BREAKS", True)
    font, prior = reader
    shape = (220, 260)
    ink, box = _draw(font, [("RH-1(D)", glyphs.Pose(130, 112, 0.75, 0.0))], shape, seed=4)
    mid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    lab = labels.Label(index=0, sheet=SHEET, kind=2, point=(0.0, 0.0), crop=np.where(ink, 0, 255).astype(np.uint8),
                       centre=(mid[0] + off * prior.pose.height, mid[1]))
    post = posterior.score_label(lab, font, TRUE_NOISE, prior, observed=np.ones(shape, bool))
    assert _top(post) == "RH-1(D)", (off, post)


@pytest.mark.xfail(strict=True, reason=(
    "measured: a code printed wider spaced than the font's advances reads 'other' (M-2 with its letters 0.3 cap "
    "heights further apart: other 0.989), which spells fillers into the gaps; codes have no letter-spacing freedom"))
def test_widely_spaced_code_reads_that_code(reader):
    font, prior = reader
    prior = dataclasses.replace(prior, forms={**prior.forms, "M-2": "M-2"},
                                codes={c: 1 / (len(READ_CODES) + 1) for c in [*READ_CODES, "M-2"]})
    s, extra = 0.75, 0.3 * glyphs.SIZE * 0.75
    chars, xs = font.layout("M-2")
    rng = np.random.default_rng(3)
    ink = np.zeros((110, 260), bool)
    boxes = []
    for i, (c, x) in enumerate(zip(chars, xs)):
        d = glyphs.render(font, c, glyphs.Pose(120 + (x + font.advance[c] / 2) * s + (i - 1) * extra, 55, s, 0.0),
                          ink.shape)
        ink |= rng.random(ink.shape) < TRUE_NOISE.p_ink(d)
        ys, xx = np.nonzero(d > 0)
        boxes.append((xx.min(), ys.min(), xx.max() + 1, ys.max() + 1))
    b = np.array(boxes, float)
    box = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
    post = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, observed=np.ones(ink.shape, bool))
    assert post["M-2"] > 0.9, post


def test_end_breaks_measure_where_words_end():
    """`end_breaks` on synthetic training labels: past a string's own ends the window holds paper or a word a gap
    away, past a boundary inside it the next letter; P(end) falls as the ink past it grows, and a window is chosen
    by the other half's log loss."""
    font = _reader_font("".join(sorted(set("".join(READ_CODES)) | set("SUD"))))
    _, ends = _measured(font)
    assert tuple(ends["window"]) in posterior.END_WINDOWS and len(ends["log_loss"]) == len(posterior.END_WINDOWS)
    p = np.array(ends["p_end"])
    assert np.all(np.diff(p) <= 1e-12) and p[0] > 0.6 and p[-1] < 0.15
    assert 0.2 < ends["base"] < 0.35                         # two ends per string among its boundaries


def test_printed_variant_reads_as_its_code():
    """Review Focus 5: the printed name "NOE VALLEY" is a form of 24TH-NOE VALLEY, so its ink reads as that code."""
    forms = posterior.printed_forms({"24TH-NOE VALLEY": "24TH-NOE VALLEY", "RH-1": "RH-1", "P": "P",
                                     "NOE VALLEY": "24TH-NOE VALLEY"})
    font = _reader_font("".join(sorted(set("".join(forms)))))
    prior = _prior(font, forms)
    ink, box = _draw(font, [("NOE VALLEY", glyphs.Pose(150, 55, 0.6, 0.0))], (110, 300), seed=9)
    post = posterior.score_label(_box_label(ink, box), font, TRUE_NOISE, prior, posterior.Constants(),
                                 observed=np.ones(ink.shape, bool))
    assert _top(post) == "24TH-NOE VALLEY" and post["24TH-NOE VALLEY"] > 0.9


def test_printed_forms_come_from_the_code_names():
    """Carried from Task 1: each named district may print "<name> NCD", derived from the names, not from reads."""
    forms = posterior.printed_forms({"24TH-NOE VALLEY": "24TH-NOE VALLEY", "POLK": "POLK", "UNION": "UNION",
                                     "CRNC": "CRNC", "RH-1": "RH-1", "C-3-O(SD)": "C-3-O(SD)", "SLI": "SLI",
                                     "NOE VALLEY": "24TH-NOE VALLEY", "UPPER MARKET": "UPPER MARKET"})
    assert forms["POLK NCD"] == "POLK" and forms["UNION NCD"] == "UNION"
    assert forms["NOE VALLEY NCD"] == forms["24TH-NOE VALLEY NCD"] == "24TH-NOE VALLEY"
    assert forms["UPPER MARKET NCD"] == forms["UPR MARKET"] == "UPPER MARKET"     # codes.canon's spelling too
    assert not any(f.startswith(("CRNC ", "RH-1 ", "C-3-O(SD) ", "SLI ")) for f in forms)


def test_code_prior_and_bigram_count_training_strings():
    labs = [_label(i, str(i), terra=t, sol=t) for i, t in enumerate(["P", "P", "RH-1", "SUD", "RH-1", "P"])]
    forms = {"P": "P", "RH-1": "RH-1", "RM-2": "RM-2"}
    prior = posterior.code_prior(labs, forms)
    assert set(prior) == {"P", "RH-1", "RM-2"} and abs(sum(prior.values()) - 1) < 1e-12
    assert prior["P"] > prior["RH-1"] > prior["RM-2"] > 0            # 3, 2 and 0 counts, smoothed
    # p(form | code): how often the training strings print each form of a code, a half count added to each
    weights = posterior.form_prior(labs, {"RH-1": "RH-1", "RH1": "RH-1", "P": "P"})
    assert weights == {"P": 1.0, "RH-1": 2.5 / 3, "RH1": 0.5 / 3}
    chars = "-1DHMPRSU "
    big = posterior.char_bigram(labs, chars, forms)
    kinds = posterior.kinds_of(chars)
    start, letter, space = kinds.index("^"), kinds.index("A"), kinds.index(" ")
    assert np.allclose(np.exp(big).sum(axis=1), 1)
    assert np.isneginf(big[start, chars.index(" ")]) and np.isneginf(big[space, -1])     # no space first or last
    assert big[start, chars.index("S")] > big[start, chars.index("R")]   # "SUD" is the only other text; codes don't count
    assert big[letter, chars.index("U")] > big[letter, chars.index("1")]


def test_posterior_weighs_a_codes_forms_by_how_often_they_are_printed():
    font = _reader_font("RH-12")
    prior = _prior(font, {"RH-2": "RH-2", "RH2": "RH-2", "RH-1": "RH-1"})
    ev = posterior.Evidence(temperatures=(5.0,), forms={"RH-2": np.array([1.0]), "RH2": np.array([4.0]),
                                                        "RH-1": np.array([3.0])}, other=np.array([0.0]),
                            gain=np.array([1.0]))
    c = posterior.Constants()
    alike = posterior.posterior_of(ev, prior, c)                     # no weights: a code's forms alike
    prior.form_weights = {"RH-2": 0.9, "RH2": 0.1, "RH-1": 1.0}
    weighed = posterior.posterior_of(ev, prior, c)
    ratio = lambda p: p["RH-2"] / p["RH-1"]
    assert math.isclose(ratio(alike), (0.5 * math.e + 0.5 * math.e ** 4) / math.e ** 3)
    assert math.isclose(ratio(weighed), (0.9 * math.e + 0.1 * math.e ** 4) / math.e ** 3)


def test_mixture_fit_separates_core_tail_and_stray():
    rng = np.random.default_rng(3)
    z = np.vstack([rng.normal(0.02, 0.05, (300, 2)), rng.normal(0.02, 0.4, (60, 2)), rng.uniform(-4, 4, (40, 2))])
    fit = posterior.mixture_fit(z, np.full(len(z), 64.0), [0.01, 0.01])
    assert np.allclose(fit["mean"], 0.02, atol=0.02)
    assert np.allclose(fit["core"], 0.05, atol=0.01) and np.allclose(fit["tail"], 0.4, atol=0.12)
    assert np.allclose(fit["weights"], [0.75, 0.15, 0.10], atol=0.05)
