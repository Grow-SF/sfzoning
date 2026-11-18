"""The control's label rule, "lines-unknown" (`textnet.zones`, `data.crop_targets(..., labels=)`,
`train.Recipe.labels`): zones only around the label points no Terra word holds, at the measured radius, and the line
ink inside word boxes left unknown.  Today's rule ("boxes") stays as it was."""
import dataclasses
import json

import cv2
import numpy as np
import pytest

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, linerules, stage, train, zones

FT_PER_PX = 4.0           # the run rule's length is round(200 / 4) = 50 px here
PAGE = 256
CENTRE = (128, 128)       # (row, col): a crop of 128 px at scale 1 centred here is page rows and cols 64..191
CROP0 = 64
SHEET = "2000-digital-zoning-p01"
ORIGIN = (6000000.0, 2100000.0)                 # the map point (EPSG:2227 ft) of page px (0, 0)
RECIPE_NAME_BEFORE = "default"                  # train.Recipe().name() at 837c994, before the label rule
STAGE_NAME_BEFORE = "half_size-ink_only=ink+1px-iterations=4000-pct_start=0.1"   # stage.DEFAULT_RECIPE.name() then


def _sheet(gray, words=(), points=(), sheet=SHEET, ft_per_px=FT_PER_PX) -> data.SheetData:
    """A 2000 SheetData made in memory: the page `gray`, placed north up at `ft_per_px` ft per page px with page px
    (0, 0) at ORIGIN; Terra's `words` (axis box, kind, reading angle) and Sol's `points` (x, y)."""
    h, w = gray.shape
    step = data.PLACEMENT_STEP
    gx, gy = np.meshgrid(np.arange(0, w - 1 + step, step, dtype=float), np.arange(0, h - 1 + step, step, dtype=float))
    world = np.stack([ORIGIN[0] + ft_per_px * gx, ORIGIN[1] - ft_per_px * gy], -1)
    rects = [data.word_rect(box, angle) for box, _, angle in words]
    return data.SheetData(
        series=paths.SERIES, sheet=sheet, height=h, width=w, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.array([data.KINDS[k] for _, k, _ in words], np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.array(points, np.int32).reshape(-1, 2), world=world, ft_per_px=float(ft_per_px))


def _blank() -> np.ndarray:
    return np.full((PAGE, PAGE), 255, np.uint8)


def _glyphs(text: str, org) -> np.ndarray:
    """A word's glyphs alone (cv2.putText, Hershey simplex 0.8, 2 px), bool page."""
    pg = _blank()
    cv2.putText(pg, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return pg < grid.INK_BELOW


def _hline(row: int) -> np.ndarray:
    """A printed line across the whole page at `row`, 2 px thick, bool page."""
    pg = _blank()
    cv2.line(pg, (0, row), (PAGE - 1, row), 0, 2)
    return pg < grid.INK_BELOW


def _box_of(mask: np.ndarray, margin: int = 2) -> list[int]:
    """The axis box (x0, y0, x1, y1) of a mask's px, grown by `margin`: a word box as Terra gives one."""
    rows, cols = np.nonzero(mask)
    return [int(cols.min()) - margin, int(rows.min()) - margin, int(cols.max()) + margin, int(rows.max()) + margin]


def _filled(rects: np.ndarray, shape) -> np.ndarray:
    """Word rectangles (n, 4, 2) filled as cv2 fills them on a padded page, cut to `shape`: the words' px."""
    pad = 200
    canvas = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad), np.uint8)
    for r in rects:
        cv2.fillPoly(canvas, [(r + pad).reshape(-1, 1, 2)], 1)
    return canvas[pad:pad + shape[0], pad:pad + shape[1]] > 0


def _crop(mask: np.ndarray, turns: int = 0, y0: int = CROP0, x0: int = CROP0, size: int = 128) -> np.ndarray:
    return np.rot90(mask[y0:y0 + size, x0:x0 + size], turns)


def _grown(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(mask.view(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


@pytest.fixture
def crossed():
    """A code word, RM-2, with a printed line through its middle and Sol's point at its centre; the glyphs, the
    line and the word's box as page masks."""
    glyph = _glyphs("RM-2", (98, 142))
    box = _box_of(glyph)
    line = _hline((box[1] + box[3]) // 2)
    gray = np.where(glyph | line, 0, 255).astype(np.uint8)
    centre = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    sd = _sheet(gray, [(box, "zoning code", 0)], [centre])
    return {"sd": sd, "glyph": glyph, "line": line, "box": _filled(sd.words, gray.shape)}


@pytest.fixture
def crossed_word_sd(crossed):
    return crossed["sd"]


@pytest.fixture
def two_points_sd():
    """A code word with one of Sol's points on it and one far from every word."""
    glyph = _glyphs("RH-1", (40, 60))
    box = _box_of(glyph)
    return _sheet(np.where(glyph, 0, 255).astype(np.uint8), [(box, "zoning code", 0)],
                  [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), (200, 200)])


@pytest.fixture
def seam_sds():
    """Two sheets of the 2000 drawing meeting at a seam: Sol's point on the first, its word read on the second at
    the same page px."""
    glyph = _glyphs("RH-2", (98, 142))
    box = _box_of(glyph)
    centre = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    gray = np.where(glyph, 0, 255).astype(np.uint8)
    return [_sheet(_blank(), [], [centre], sheet="2000-digital-zoning-p01"),
            _sheet(gray, [(box, "zoning code", 0)], [], sheet="2000-digital-zoning-p02")]


EDGE_BOX = [-46, -35, 66, 39]      # a word read at 30 degrees (120 x 16 px) running off the page's top-left corner


@pytest.fixture
def edge_word_sd():
    """A turned code word running off the page's top-left corner, a printed line through it."""
    gray = np.where(_hline(10), 0, 255).astype(np.uint8)
    return _sheet(gray, [(EDGE_BOX, "zoning code", 30)], [(5, 5)])


# --- the label rule in the crops -------------------------------------------------------------------------------------

def test_line_ink_inside_a_word_is_unknown_and_the_letters_stay_text(crossed, crossed_word_sd):
    line_in_box = _crop(crossed["line"] & crossed["box"])
    glyph_far = _crop(crossed["glyph"] & ~_grown(crossed["line"], 2))
    line_out = _crop(crossed["line"] & ~crossed["box"])
    assert line_in_box.any() and glyph_far.sum() > 100 and line_out.any()
    sd = zones.prepare([crossed_word_sd], (), cache=None)[0]
    t = data.crop_targets(sd, *CENTRE, 128, 1.0, 0, labels="lines-unknown")
    assert not t["known"][line_in_box].any()
    assert t["text"][glyph_far].all() and t["known"][glyph_far].all()
    # the line outside the box stays a lesson of not-text: Sol's point sits on its own word, so it has no zone
    assert len(sd.points) == 0
    assert t["known"][line_out].all() and not t["text"][line_out].any()
    # today's rule on the same sheet: the line in the box is taught as text
    b = data.crop_targets(sd, *CENTRE, 128, 1.0, 0)
    assert b["known"][line_in_box].all() and b["text"][line_in_box].all()
    # only `known` differs, and only where the rule makes px unknown
    assert all(np.array_equal(t[k], b[k]) for k in ("ink", "text", "kind", "angle"))
    assert not (t["known"] & ~b["known"]).any()


def test_lines_unknown_without_lines_is_an_error(crossed_word_sd):
    with pytest.raises(ValueError, match="line"):
        data.crop_targets(crossed_word_sd, *CENTRE, 128, 1.0, 0, labels="lines-unknown")
    with pytest.raises(ValueError, match="one of"):
        data.crop_targets(crossed_word_sd, *CENTRE, 128, 1.0, 0, labels="lines")


def test_a_sheets_lines_are_its_run_rule_grown_by_one_px_and_packed_like_its_ink(crossed, crossed_word_sd):
    sd = zones.prepare([crossed_word_sd], (), cache=None)[0]
    ink = crossed["glyph"] | crossed["line"]
    want = _grown(linerules.line_px(ink, FT_PER_PX), 1)
    assert sd.line_bits.dtype == np.uint8 and sd.line_bits.shape == sd.ink_bits.shape
    assert np.array_equal(np.unpackbits(sd.line_bits, axis=1, count=PAGE).view(bool), want)
    # the letters, more than a px from the line, are no line
    assert not (want & crossed["glyph"] & ~_grown(crossed["line"], 2)).any()


def test_a_sheet_without_its_scale_has_no_lines(crossed_word_sd):
    with pytest.raises(ValueError, match=SHEET):
        zones.prepare([dataclasses.replace(crossed_word_sd, ft_per_px=float("nan"))], (), cache=None)


def test_a_turned_word_running_off_the_page_is_clipped(edge_word_sd):
    sd = zones.prepare([edge_word_sd], (), cache=None)[0]
    t = data.crop_targets(sd, 5, 5, 128, 1.0, 1, labels="lines-unknown")
    # the window is the page's top-left 128 px (the crop moved inside the page), turned a quarter
    box = _filled(sd.words, (PAGE, PAGE))
    line = _hline(10)
    assert box[0, 0] and (line & box).any() and (line & ~box).any()
    assert not t["known"][_crop(line & box, 1, 0, 0)].any()
    assert t["known"][_crop(line & ~box, 1, 0, 0)].all()
    assert np.array_equal(t["text"] > 0, _crop(box, 1, 0, 0))


# --- the zones: only around label points no Terra word holds ---------------------------------------------------------

def test_a_point_with_its_own_word_gets_no_zone_and_an_unmatched_one_does(two_points_sd):
    assert zones.matched_points(two_points_sd).tolist() == [True, False]
    sd = zones.prepare([two_points_sd], (), cache=None)[0]
    assert sd.points.tolist() == [[200, 200]]


def test_a_point_matches_a_word_within_half_its_height(two_points_sd):
    # a word read at 30 degrees, 100 px long and 24 px high, centred on (128, 128); its first edge along the reading
    # direction (as `data.word_rect` gives it)
    a = np.deg2rad(30)
    v, n = np.array([np.cos(a), -np.sin(a)]), np.array([np.sin(a), np.cos(a)])
    c, half_length, half_height = np.array([128.0, 128.0]), 50.0, 12.0
    rect = np.array([c - v * half_length - n * half_height, c + v * half_length - n * half_height,
                     c + v * half_length + n * half_height, c - v * half_length + n * half_height])
    points = [c, c + n * (2 * half_height - 2), c + n * (2 * half_height + 2),        # beside its long edge
              c - v * (half_length + half_height - 2), c - v * (half_length + half_height + 2)]   # past its start
    sd = dataclasses.replace(two_points_sd, words=np.round(rect).astype(np.int32)[None],
                             kinds=np.array([data.KINDS["zoning code"]], np.uint8),
                             angles=np.array([30], np.int16), points=np.round(points).astype(np.int32))
    assert zones.matched_points(sd).tolist() == [True, True, False, True, False]


def test_a_word_on_another_sheet_does_not_match_a_point(seam_sds):
    assert not zones.matched_points(seam_sds[0]).any()
    first, second = zones.prepare(seam_sds, (), cache=None)
    assert len(first.points) == 1 and first.point_radius == second.point_radius


def test_the_zone_radius_is_half_the_99th_percentile_of_the_code_words_outside_the_areas_held():
    words = [([20, 20 + 30 * i, 20 + length, 40 + 30 * i], "zoning code", 0)
             for i, length in enumerate((40, 50, 60, 70, 100))]
    words.append(([20, 200, 220, 220], "street name", 0))       # no code word: not measured
    sd = _sheet(_blank(), words)
    got = zones.zone_radius([sd], ())
    # lengths 40 50 60 70 100: the 99th percentile (linear) is 70 + 0.96 x 30 = 98.8, so 49 px
    assert got == {"rule": "q99/2", "px": 49, "q50": 60.0, "q90": 88.0, "q99": 98.8, "words": 5}
    # the 100 px word's centre (70, 150) is in an area held out: it is not measured
    x, y = ORIGIN[0] + FT_PER_PX * 70, ORIGIN[1] - FT_PER_PX * 150
    held = data.MapRect(x - 10, y - 10, x + 10, y + 10)
    got = zones.zone_radius([sd], (held,))
    assert (got["px"], got["q99"], got["words"]) == (35, 69.7, 4)
    with pytest.raises(ValueError, match="zoning code"):
        zones.zone_radius([_sheet(_blank(), words[-1:])], ())


def test_every_sheet_gets_the_measured_radius(seam_sds, crossed_word_sd):
    long_word = _sheet(_blank(), [([20, 20, 220, 44], "zoning code", 0)], sheet="2000-digital-zoning-p03")
    sheets = [*seam_sds, crossed_word_sd, long_word]
    radius = zones.zone_radius(sheets, ())["px"]
    assert radius > 90 and radius != data.READ_POINT_RADIUS          # the 200 px word sets the 99th percentile
    assert [sd.point_radius for sd in zones.prepare(sheets, (), cache=None)] == [radius] * 4


# --- the lines kept on disk ------------------------------------------------------------------------------------------

def test_lines_are_kept_per_sheet_and_found_again_when_the_rule_or_the_page_changes(crossed_word_sd, tmp_path,
                                                                                      monkeypatch):
    first = zones.prepare([crossed_word_sd], (), cache=tmp_path)[0]
    f = tmp_path / paths.SERIES / f"{SHEET}.npz"
    assert f.is_file()

    def refuse(*a, **k):
        raise AssertionError("the lines were found again")

    real = zones.linerules.line_px
    monkeypatch.setattr(zones.linerules, "line_px", refuse)
    again = zones.prepare([crossed_word_sd], (), cache=tmp_path)[0]
    assert np.array_equal(again.line_bits, first.line_bits)
    calls = []

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(zones.linerules, "line_px", counted)
    monkeypatch.setattr(zones, "rule_key", lambda: "another rule")      # linerules.py changed
    zones.prepare([crossed_word_sd], (), cache=tmp_path)
    assert len(calls) == 1
    blank = dataclasses.replace(crossed_word_sd, ink_bits=np.zeros_like(crossed_word_sd.ink_bits))   # another page
    assert not zones.prepare([blank], (), cache=tmp_path)[0].line_bits.any() and len(calls) == 2


# --- the shares of in-box ink made unknown ---------------------------------------------------------------------------

def test_label_shares_count_the_in_box_ink_made_unknown_per_series_and_kind(crossed, crossed_word_sd):
    sd = zones.prepare([crossed_word_sd], (), cache=None)[0]
    ink = crossed["glyph"] | crossed["line"]
    lines = np.unpackbits(sd.line_bits, axis=1, count=PAGE).view(bool)
    in_box = ink & crossed["box"]
    got = zones.label_shares([sd])
    code = got[paths.SERIES]["zoning code"]
    assert (code["ink"], code["unknown"]) == (int(in_box.sum()), int((in_box & lines).sum()))
    assert code["share"] == pytest.approx(code["unknown"] / code["ink"]) and 0 < code["share"] < 0.5
    assert got[paths.SERIES]["all"] == code
    assert got[paths.SERIES]["street name"] == {"ink": 0, "unknown": 0, "share": None}
    json.dumps(got)
    with pytest.raises(ValueError, match="line"):
        zones.label_shares([crossed_word_sd])


# --- today's rule and its name ---------------------------------------------------------------------------------------

def test_the_boxes_rule_and_its_run_name_are_unchanged():
    assert train.Recipe().name() == RECIPE_NAME_BEFORE
    assert stage.DEFAULT_RECIPE.name() == STAGE_NAME_BEFORE
    assert train.Recipe().labels == "boxes" and train.CHOICES["labels"] == ("boxes", "lines-unknown")
    assert train.Recipe(labels="lines-unknown").name() == "labels=lines-unknown"
    with pytest.raises(ValueError, match="one of"):
        train.Recipe(labels="lines")


def test_lines_on_a_sheet_never_change_the_boxes_rules_batches(crossed_word_sd):
    lined = dataclasses.replace(crossed_word_sd,
                                line_bits=zones.prepare([crossed_word_sd], (), cache=None)[0].line_bits)
    pools = train.centre_pools([crossed_word_sd])
    recipe = train.Recipe(batch=8, crop=64)
    plain = train.draw([crossed_word_sd], pools, recipe, np.random.default_rng(3))
    with_lines = train.draw([lined], pools, recipe, np.random.default_rng(3))
    for a, b in zip(plain, with_lines, strict=True):
        assert sorted(a) == sorted(b)
        assert all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a)
    # the recipe's rule reaches the crops: the same draws, fewer known px
    ruled = train.draw([lined], pools, dataclasses.replace(recipe, labels="lines-unknown"), np.random.default_rng(3))
    assert all(not (r["known"] & ~a["known"]).any() for r, a in zip(ruled, plain, strict=True))
    assert sum(int(a["known"].sum() - r["known"].sum()) for r, a in zip(ruled, plain)) > 0
    with pytest.raises(ValueError, match="line"):
        train.draw([crossed_word_sd], pools, dataclasses.replace(recipe, labels="lines-unknown"),
                   np.random.default_rng(3))
