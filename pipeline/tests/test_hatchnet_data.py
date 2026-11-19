"""The learned reader's training data (`sfzmaps.solver.hatchnet.data`): the scan as printed (a sheet scanned on its side
turned back exactly), today's blocks placed on it and labelled once where scans overlap, nothing erased from the ink,
and the lot table's one read per lot and the owner's answers."""
import dataclasses
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import shapely

from sfzmaps.solver import grid
from sfzmaps.solver.hatchnet import data
from sfzmaps.transform import GCP, SheetTransform

SERIES = "1921"
PAGE = (150, 100)          # an upright sheet's page px (rows, cols): portrait, as every 1921 section
FT_PER_PX = 1.1            # not a divisor of the grid's 4 ft: no page px centre lands on a grid px edge
Y_FT = 30.21               # the sheets' top edge, ft below the grid's top
INSET = 5                  # the neatline: px inside the page edge
LOTS = [                   # (rows, cols) of the solver grid (4 ft px), blklot, quarter
    ((15, 25), (8, 12), "0001001", 0),      # on A only
    ((15, 25), (24, 28), "0002001", 1),     # in the overlap of A and C
    ((15, 25), (38, 42), "0003001", 2),     # on C only
    ((28, 36), (14, 18), "0004001", 3),     # on A only
    ((28, 36), (43, 46), "0005001", 1),     # on C only
    ((28, 36), (24, 28), "0006001", 3),     # in the overlap of A and C: C owns its centre
]
TODAY = {                  # today's parcels (DataSF) that the solver's April 2000 table lacks: world box (ft from X0, Y1)
    "0003001B": (154.0, 62.0, 166.0, 98.0),     # inside lot 2
    "0003001C": (156.0, 64.0, 164.0, 96.0),     # inside lot 2
    "0004001Z": (58.0, 116.0, 70.0, 140.0),     # inside lot 3
    "0009001": (73.0, 62.0, 79.0, 98.0),        # on a street (no solver lot)
}


def placement(shape, x_ft: float, y_ft: float = Y_FT, ft_per_px: float = FT_PER_PX) -> SheetTransform:
    """North up: page px (x, y) -> (X0 + x_ft + x ft_per_px, Y1 - y_ft - y ft_per_px), fitted from the 4 corners."""
    h, w = shape
    gcps = [GCP(px, py, grid.X0 + x_ft + px * ft_per_px, grid.Y1 - y_ft - py * ft_per_px, "manual")
            for px, py in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))]
    return SheetTransform(gcps, 1.0)


def placement_turned(shape, x_ft: float, y_ft: float = Y_FT, ft_per_px: float = FT_PER_PX) -> SheetTransform:
    """The placement of `np.rot90(upright, -1)` (scanned a quarter turn clockwise: north to the right), `shape` the
    upright page's: turned px (x, y) is upright px (y, h - 1 - x)."""
    h, w = shape
    gcps = [GCP(px, py, grid.X0 + x_ft + py * ft_per_px, grid.Y1 - y_ft - (h - 1 - px) * ft_per_px, "manual")
            for px, py in ((0, 0), (h - 1, 0), (h - 1, w - 1), (0, w - 1))]
    return SheetTransform(gcps, 1.0)


def scan(seed: int = 0) -> np.ndarray:
    """A grey page as printed: diagonal hatch, a solid block, block outlines, a white text knockout with letters in
    it, and grey px of every value (a scanned edge's anti-aliasing)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:PAGE[0], 0:PAGE[1]]
    pg = np.full(PAGE, 255, np.uint8)
    pg[np.mod(x + y, 9) < 2] = 0                       # 1R-like hatch
    pg[100:130, 10:40] = 0                             # a solid (C) block
    cv2.rectangle(pg, (8, 8), (90, 60), 0, 2)          # a block outline
    pg[30:42, 30:70] = 255                             # a text knockout
    cv2.putText(pg, "AB", (32, 41), cv2.FONT_HERSHEY_PLAIN, 0.8, 0, 1)
    noise = rng.integers(0, 256, PAGE).astype(np.uint8)
    edge = rng.random(PAGE) < 0.05
    pg[edge] = noise[edge]
    return pg


def write_sheet(root, sheet: str, page: np.ndarray, tr: SheetTransform, quad, crop: np.ndarray, reads=()):
    d = root / "work" / SERIES / sheet
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "page.png"), page)
    (d / "transform.json").write_text(json.dumps(tr.to_json()))
    parcels = [{"blklot": b, "code": c, "confidence": p, "via": v} for b, c, p, v in reads]
    (d / "district_pattern.json").write_text(json.dumps({"series": SERIES, "sheet": sheet, "parcels": parcels}))
    crops = root / "solver" / SERIES / "page" / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crops / f"{sheet}.png"), crop.astype(np.uint8) * 255)
    f = root / "solver" / SERIES / "page" / "frames.json"
    frames = json.loads(f.read_text()) if f.is_file() else {}
    frames[sheet] = [[float(a), float(b)] for a, b in quad]
    f.write_text(json.dumps(frames))


def write_lots(root):
    """The solver's lot raster (a 80 x 80 corner of its grid), quarters and table (geometry in EPSG:2227 ft)."""
    lot = np.full((80, 80), -1, np.int32)
    geoms, rows = [], []
    for i, ((r0, r1), (c0, c1), blklot, _) in enumerate(LOTS):
        lot[r0:r1, c0:c1] = i
        geoms.append(shapely.box(grid.X0 + 4 * c0, grid.Y1 - 4 * r1, grid.X0 + 4 * c1, grid.Y1 - 4 * r0))
        rows.append({"blklot": blklot, "block_num": blklot[:4]})
    d = root / "solver" / SERIES / "lots"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "lots.npz", lot=lot, quarter=np.array([q for *_, q in LOTS], np.int8))
    pd.DataFrame(rows).assign(geometry=shapely.to_wkb(np.array(geoms))).to_parquet(d / "table.parquet")
    return lot


def write_today(root):
    """Today's parcels (blklot, WKB geometry in EPSG:2227 ft), one with no geometry."""
    geoms = [shapely.box(grid.X0 + x0, grid.Y1 - y1, grid.X0 + x1, grid.Y1 - y0) for x0, y0, x1, y1 in TODAY.values()]
    wkb = list(shapely.to_wkb(np.array(geoms))) + [None]
    pd.DataFrame({"blklot": list(TODAY) + ["0009003"], "geometry": wkb}).to_parquet(root / "parcels.parquet")


STREETS = [                # today's centre lines: world (x, y) ft from (X0, Y1), x east and y south
    ((78.0, 30.0), (78.0, 190.0)),          # north-south down grid column 19, on sheet A only
    ((10.0, 162.0), (190.0, 162.0)),        # east-west along grid row 40, across both sheets
]


def write_streets(root):
    """Today's street centre lines (WKB LineStrings in EPSG:2227 ft, as data/ref/streets.parquet), one not active."""
    lines = [shapely.LineString([(grid.X0 + x, grid.Y1 - y) for x, y in pts]) for pts in STREETS]
    pd.DataFrame({"cnn": np.arange(len(lines)), "active": [True, False],
                  "geometry": list(shapely.to_wkb(np.array(lines)))}).to_parquet(root / "streets.parquet")


def sources(root, order=None) -> data.Sources:
    parcels, streets = root / "parcels.parquet", root / "streets.parquet"
    return data.Sources(series=SERIES, work=root / "work" / SERIES, solver=root / "solver" / SERIES,
                        legends=root / "legends.json", answers=root / "answers", cases=root / "cases",
                        parcels=parcels if parcels.is_file() else None, order=tuple(order) if order else None,
                        streets=streets if streets.is_file() else None)


def upright_quad(shape=PAGE):
    h, w = shape
    return [(INSET, INSET), (w - 1 - INSET, INSET), (w - 1 - INSET, h - 1 - INSET), (INSET, h - 1 - INSET)]


def upright_crop(shape=PAGE):
    c = np.zeros(shape, bool)
    c[INSET:shape[0] - INSET, INSET:shape[1] - INSET] = True
    return c


def two_sheets(root, reads_a=(), reads_c=()):
    """Sheets A (world x 20-130 ft) and C (80-190 ft) side by side, overlapping 50 ft; C is drawn above A."""
    write_lots(root)
    write_sheet(root, "sheetA", scan(0), placement(PAGE, 20.37), upright_quad(), upright_crop(), reads_a)
    write_sheet(root, "sheetC", scan(1), placement(PAGE, 80.37), upright_quad(), upright_crop(), reads_c)
    return sources(root, order=("sheetA", "sheetC"))


# --- the tests ------------------------------------------------------------------------------------------------------

def test_turned_sheet_matches_upright(tmp_path):
    # the same sheet scanned upright and on its side (a quarter turn clockwise: north to the right, as
    # 20180102100525362) gives the same scan and the same rasters once turned back
    page = scan(0)
    up, side = tmp_path / "up", tmp_path / "side"
    for root in (up, side):
        write_lots(root)
        write_streets(root)
    write_sheet(up, "sheet", page, placement(PAGE, 20.37), upright_quad(), upright_crop())
    h, w = PAGE
    turned_quad = [(h - 1 - y, x) for x, y in upright_quad()]          # upright px (x, y) -> turned (h-1-y, x)
    turned_quad = turned_quad[3:] + turned_quad[:3]                      # still top-left first
    write_sheet(side, "sheet", np.ascontiguousarray(np.rot90(page, -1)), placement_turned(PAGE, 20.37),
                turned_quad, np.rot90(upright_crop(), -1))
    a = data.load_scan(SERIES, "sheet", sources(up))
    b = data.load_scan(SERIES, "sheet", sources(side))
    assert a.dtype == np.float32 and a.shape == PAGE
    assert np.array_equal(a, b)
    ra, rb = data.sheet_rasters(SERIES, "sheet", sources(up)), data.sheet_rasters(SERIES, "sheet", sources(side))
    assert (ra.turn, rb.turn) == (0, 1)
    for k in ("ink", "block", "lot", "crop", "street", "centre"):
        assert np.array_equal(getattr(ra, k), getattr(rb, k)), k
    assert (ra.block == 1).sum() > 0 and (ra.block == 0).sum() > 0 and (ra.lot >= 0).sum() > 0
    assert (ra.street == 1).sum() > 0 and ra.centre.sum() > 0
    assert np.isclose(ra.ft_per_px, rb.ft_per_px) and np.isclose(ra.ft_per_px, FT_PER_PX)
    # a box on the scanned page lands on the same px of the turned-back page
    assert data.turn_box((10, 20, 30, 50), (w, h), 1) == (20, h - 30, 50, h - 10)
    box = data.turn_box((10, 20, 30, 50), (w, h), 1)
    side_page = np.rot90(page, -1)
    assert np.array_equal(np.rot90(side_page, 1)[box[1]:box[3], box[0]:box[2]],
                          np.rot90(side_page[20:50, 10:30], 1))


def test_overlap_owned_once(tmp_path):
    src = two_sheets(tmp_path)
    ra = data.sheet_rasters(SERIES, "sheetA", src)
    rc = data.sheet_rasters(SERIES, "sheetC", src)
    ta, tc = placement(PAGE, 20.37), placement(PAGE, 80.37)

    def at(r, tr, world):
        x, y = np.round(tr.to_pixel(np.array([world]))[0]).astype(int)
        return int(r.block[y, x]), int(r.lot[y, x])

    overlap = (grid.X0 + 105.3, grid.Y1 - 80.3)                  # inside both neatlines: C is drawn above A
    assert at(ra, ta, overlap)[0] == 255 and at(ra, ta, overlap)[1] == -1
    assert at(rc, tc, overlap)[0] != 255 and at(rc, tc, overlap)[1] == 1
    only_a = (grid.X0 + 40.3, grid.Y1 - 80.3)
    assert at(ra, ta, only_a)[0] != 255 and at(ra, ta, only_a)[1] == 0
    # outside the neatline: unknown; inside it on no other sheet: labelled
    assert (ra.block[~ra.crop] == 255).all() and (ra.lot[~ra.crop] == -1).all()
    assert (rc.block[rc.crop] != 255).all()                      # C is on top: it keeps its whole neatline
    # each place is labelled on one sheet: A's labelled px reach no further east than C's neatline begins
    ys, xs = np.nonzero(ra.block != 255)
    east = ta.to_world(np.c_[xs, ys].astype(float))[:, 0].max()
    c_west = tc.to_world(np.array([[INSET, 50.0]]))[0, 0]
    assert east < c_west and east > c_west - 2 * FT_PER_PX


def test_lot_table_answers_and_reads(tmp_path):
    write_today(tmp_path)
    src = two_sheets(tmp_path,
                     reads_a=[("0001001", "1R", 0.8, "parcel"), ("0002001", "2R", 0.6, "parcel"),
                              ("0004001", "", 0.5, "parcel"), ("0003001", "HI", 0.9, "off_sheet"),
                              ("0006001", "HI", 0.7, "parcel"), ("0004001Z", "1R", 0.6, "parcel"),
                              ("0009001", "C", 0.9, "parcel"), ("0009002", "C", 0.9, "parcel"),
                              ("0009003", "C", 0.9, "parcel")],
                     reads_c=[("0002001", "C", 0.7, "parcel"), ("0003001", "LI", 0.9, "block"),
                              ("0005001", "U", 0.5, "parcel"), ("0003001B", "2R", 0.95, "parcel"),
                              ("0003001C", "1R", 0.5, "parcel")])
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "1921.json").write_text(json.dumps({"cases": [
        {"id": "c0001", "stratum": "random", "code": "1R", "blklot": "0001001", "block": "0001"},
        {"id": "c0002", "stratum": "disagree", "code": "2R", "blklot": "0002001", "block": "0002"},
        {"id": "c0003", "stratum": "random", "code": "2R", "blklot": "0004001", "block": "0004"}]}))
    (cases / "1921-likely.json").write_text(json.dumps({"cases": [
        {"id": "c0001", "stratum": "all", "code": "LI", "blklot": "0003001", "blklots": ["0003001"], "block": "0003"},
        {"id": "c0002", "stratum": "all", "code": "1R", "blklot": "0009009", "blklots": ["0009009", "0005001"],
         "block": "0005"}]}))
    answers = {"answers-1921": [("c0001", "ok", "1R"), ("c0002", "fix", "LI"), ("c0003", "canttell", "")],
               "answers-1921-likely": [("c0001", "unzoned", ""), ("c0002", "fix", "HI")]}
    for page, rows in answers.items():
        d = tmp_path / "answers" / page
        d.mkdir(parents=True)
        for case, ans, code in rows:
            (d / f"{case}.json").write_text(json.dumps({"case": case, "answer": ans, "code": code}))
    t = data.lot_table(SERIES, src)
    assert list(t["lot"]) == [0, 1, 2, 3, 4, 5]
    assert list(t["blklot"]) == [b for *_, b, _ in LOTS]
    assert list(t["quarter"]) == ["nw", "ne", "sw", "se", "ne", "se"]
    # one read per lot (R3'): a lot read on one sheet keeps it whichever sheet that is (lot 5: A read it, C owns its
    # centre); read on two, the draw order picks (lot 1: C, drawn above A, owns its centre).  Empty codes and the
    # reader's off-sheet marks are no reads.  R4: a read of today's parcel the solver's table lacks lands on the
    # solver lot under the parcel's representative point (lots 2 and 3); several on one lot from one sheet keep the
    # most confident (lot 2: 0.95 over its own 0.9 and 0.5); one on no solver lot, not in today's parcels or with no
    # geometry stays unattached
    assert list(t["read_code"]) == ["1R", "C", "2R", "1R", "U", "HI"]
    assert np.allclose(t["read_conf"].to_numpy(), [0.8, 0.7, 0.95, 0.6, 0.5, 0.7])
    assert list(t["read_sheet"]) == ["sheetA", "sheetC", "sheetC", "sheetA", "sheetC", "sheetA"]
    assert list(t["read_blklot"]) == ["0001001", "0002001", "0003001B", "0004001Z", "0005001", "0006001"]
    assert list(t["read_recovered"]) == [False, False, True, True, False, False]
    assert list(t["n_reads"]) == [1, 2, 1, 1, 1, 1]
    r = t.attrs["reads"]
    assert (r["direct"], r["recovered"]) == (6, 3)
    assert r["unattached"] == {"no_geometry": 1, "no_solver_lot": 1, "not_in_parcels": 1}
    assert (r["lots_several_reads_one_sheet"], r["lots_read_on_several_sheets"]) == (1, 1)
    # the owner's answers: ok -> the code shown, fix -> the fix, unzoned -> 2R, canttell dropped; a case whose own
    # blklot is not a solver lot lands on the lot of its footprint
    assert list(t["answer"]) == ["1R", "LI", "2R", "", "HI", ""]
    assert list(t["answer_page"]) == ["1921", "1921", "1921-likely", "", "1921-likely", ""]
    assert list(t["stratum"]) == ["random", "disagree", "likely", "", "likely", ""]


def test_no_rule_masks(tmp_path):
    # the ink is the scan's luminance, every px: hatch, solid ink, outlines, knockouts and text all kept
    src = two_sheets(tmp_path)
    page = scan(0)
    r = data.sheet_rasters(SERIES, "sheetA", src)
    assert r.ink.dtype == np.float32
    expect = page.astype(np.float32) / np.float32(255)
    assert np.array_equal(r.ink[r.crop], expect[r.crop])
    assert np.array_equal(r.ink, expect)


def test_full_tone_luminance(tmp_path):
    # a colour scan (1948) goes in as its luminance (Rec. 601), 0 = ink, not thresholded
    rgb = np.array([[[255, 255, 255], [0, 0, 0], [200, 100, 50], [12, 200, 240]]], np.uint8)
    f = tmp_path / "page.png"
    cv2.imwrite(str(f), rgb[..., ::-1])
    lum = data.load_luminance(f)
    r, g, b = (rgb[..., k].astype(np.float32) for k in range(3))
    expect = (np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * b) / np.float32(255)
    assert lum.dtype == np.float32 and lum.shape == (1, 4)
    assert np.allclose(lum, expect, atol=1e-6) and lum[0, 0] == 1.0 and lum[0, 1] == 0.0


def gap_blocks(width: int, *spans) -> np.ndarray:
    """A grid (20 rows) whose blocks are the column spans given (ends exclusive), every row: a 1-D map."""
    block = np.zeros((20, width), bool)
    for c0, c1 in spans:
        block[:, c0:c1] = True
    return block


def centre_cols(shape, *cols) -> np.ndarray:
    centre = np.zeros(shape, bool)
    for c in cols:
        centre[:, c] = True
    return centre


def test_corridor_between_two_blocks():
    # blocks at columns 0-9 and 20-29, a 10 px street between them, its centre line at column 14 (5 px from the west
    # block, 6 from the east): every px between the blocks is street, the blocks are not street
    block = gap_blocks(40, (0, 10), (20, 30))
    centre = centre_cols(block.shape, 14)
    street, c = data.street_labels(block, centre)
    assert street.dtype == np.uint8 and c.dtype == np.uint8
    assert (street[:, 10:20] == 1).all()
    assert (street[block] == 0).all()
    assert np.array_equal(c == 1, centre)
    assert set(np.unique(street)) <= {0, 1, data.UNKNOWN}


def test_shoreline_stops_at_half_width():
    # a block (columns 0-9), a centre line 5 px from it (column 14), then open water: the corridor reaches the line's
    # own distance to the block on both sides, 5 px past the line on the open side (columns 15-19), and no further
    block = gap_blocks(60, (0, 10))
    street, _ = data.street_labels(block, centre_cols(block.shape, 14))
    assert (street[:, :10] == 0).all()
    assert (street[:, 10:20] == 1).all()
    assert (street[:, 20:] == data.UNKNOWN).all()


def test_no_centre_line_is_ignored():
    # a street between two blocks with no centre line today (columns 30-39) is ignored, not labelled not-street; the
    # one with a centre line (columns 10-19) is street.  With no centre line at all, every px off the blocks is ignored
    block = gap_blocks(60, (0, 10), (20, 30), (40, 50))
    street, _ = data.street_labels(block, centre_cols(block.shape, 14))
    assert (street[:, 10:20] == 1).all()
    assert (street[:, 30:40] == data.UNKNOWN).all()
    assert (street[block] == 0).all()
    street, c = data.street_labels(block, np.zeros_like(block))
    assert (street[block] == 0).all() and (street[~block] == data.UNKNOWN).all() and not c.any()


def test_centre_on_a_block_is_not_street():
    # a centre line crossing today's block (a park road, a street built over): the block stays not-street, the line is
    # no centre there, and its corridor is the line itself, so it labels nothing off the block
    block = gap_blocks(40, (0, 20))
    centre = centre_cols(block.shape, 10)
    street, c = data.street_labels(block, centre)
    assert (street[block] == 0).all() and (street[~block] == data.UNKNOWN).all()
    assert not c.any()


def test_centre_lines_rasterised_on_the_grid():
    # today's centre lines (EPSG:2227 ft) on the solver's 4 ft grid, 8-connected, one px wide, wherever they run;
    # a line off the grid is left out
    lines = [shapely.LineString([(grid.X0 + 2, grid.Y1 - 42), (grid.X0 + 158, grid.Y1 - 42)]),       # row 10
             shapely.LineString([(grid.X0 + 30, grid.Y1 - 2), (grid.X0 + 70, grid.Y1 - 158)]),        # diagonal
             shapely.LineString([(grid.X0 - 500, grid.Y1 + 500), (grid.X0 - 100, grid.Y1 + 100)])]    # off the grid
    c = data.centre_raster(np.array(lines, dtype=object), (40, 40))
    assert c.dtype == bool
    assert c[10].all()
    n, _ = cv2.connectedComponents(c.astype(np.uint8), connectivity=8)
    assert n == 2                                                  # one piece (and the background)
    assert c[39, 17]                                               # the diagonal reaches the bottom row at x 70 ft


def test_street_and_centre_carried_like_block(tmp_path):
    # the labels go to each page through the same placement and ownership as `block`: not-street exactly where block is
    # 1, ignored wherever block is unknown; the centre lines are drawn on the page through the placement, connected,
    # on street px only, and only on the sheet that owns them
    write_streets(tmp_path)
    src = two_sheets(tmp_path)
    for name, x_ft in (("sheetA", 20.37), ("sheetC", 80.37)):
        r = data.sheet_rasters(SERIES, name, src)
        assert r.street.dtype == np.uint8 and r.centre.dtype == np.uint8
        assert np.array_equal(r.street == 0, r.block == 1), name
        assert (r.street[r.block == data.UNKNOWN] == data.UNKNOWN).all(), name
        assert (r.street == 1).any() and (r.street[r.block == 0] == data.UNKNOWN).any(), name
        assert r.centre.any() and (r.street[r.centre == 1] == 1).all(), name
        assert (r.centre[r.block == data.UNKNOWN] == 0).all(), name
        # the drawn centre px lie on the lines (STREETS), within a page px; the north-south line is A's alone
        ys, xs = np.nonzero(r.centre)
        world = placement(PAGE, x_ft).to_world(np.c_[xs, ys].astype(float))
        on_ew = np.abs(world[:, 1] - (grid.Y1 - 162.0)) <= FT_PER_PX
        on_ns = np.abs(world[:, 0] - (grid.X0 + 78.0)) <= FT_PER_PX
        assert (on_ew | on_ns).all() and on_ew.any(), name
        assert on_ns.any() == (name == "sheetA"), name
        assert cv2.connectedComponents(r.centre, connectivity=8)[0] == 2, name
    # with no centre lines every px off today's blocks is ignored
    bare = data.sheet_rasters(SERIES, "sheetA", dataclasses.replace(src, streets=None))
    assert not (bare.street == 1).any() and not bare.centre.any()
    assert np.array_equal(bare.street == 0, bare.block == 1)


def test_run_writes_same_bytes(tmp_path):
    write_streets(tmp_path)
    src = two_sheets(tmp_path, reads_a=[("0001001", "1R", 0.8, "parcel")])
    (tmp_path / "legends.json").write_text(json.dumps({SERIES: {"sheet": "sheetA", "swatches": {
        "1R": [10, 20, 30, 40]}}}))

    def digest(folder):
        return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}

    one = data.run(SERIES, tmp_path / "one", workers=1, src=src)
    two = data.run(SERIES, tmp_path / "two", workers=1, src=src)
    assert digest(tmp_path / "one") == digest(tmp_path / "two")
    assert one["sheets_used"] == ["sheetA", "sheetC"]
    r = data.read_rasters(tmp_path / "one" / "sheetA.npz")
    direct = data.sheet_rasters(SERIES, "sheetA", src)
    for k in ("block", "street", "centre"):
        assert np.array_equal(getattr(r, k), getattr(direct, k)), k
    assert (r.street == 1).any() and r.centre.any()
    lab = one["street_labels"]
    assert lab["lines"] == len(STREETS) and lab["centre_grid_px"] > 0
    assert data.swatches(SERIES, src) == [("sheetA", "1R", (10, 20, 30, 40))]
