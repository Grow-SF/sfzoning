"""The 1921 patch measures (`sfzmaps.solver.hatching`): the hatch period and line angle, periodicity, a lot between two
hatch lines borrowing its patch's evidence, a patch at a seam written once, and a rerun writing the same bytes."""
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import pytest
import shapely

from sfzmaps.solver import grid, hatching
from sfzmaps.transform import GCP, SheetTransform


def hatch(shape, period, width=3, angle_deg=-45.0, offset=0.0) -> np.ndarray:
    """1-bit lines (1 = ink) at `angle_deg` (page px, y down: -45 rises to the right), `period` px apart measured across
    them, `width` px wide."""
    t = np.radians(angle_deg)
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    across = -x * np.sin(t) + y * np.cos(t) + offset
    return (np.mod(across, period) < width).astype(np.uint8)


def affine(shape, x_ft: float, y_ft: float, ft_per_px: float = 1.0) -> SheetTransform:
    """A placement: page px (x, y) -> (X0 + x_ft + x ft_per_px, Y1 - y_ft - y ft_per_px), fitted from the 4 corners."""
    h, w = shape
    gcps = [GCP(px, py, grid.X0 + x_ft + px * ft_per_px, grid.Y1 - y_ft - py * ft_per_px, "manual")
            for px, py in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))]
    return SheetTransform(gcps, 1.0)


def test_period_and_angle():
    ink = hatch((200, 200), 20)
    period = hatching.sheet_period(ink, np.ones(ink.shape, bool))
    assert abs(period - 20) <= 1
    box = hatching.patch_grid(ink.shape, period)[60]
    f = hatching.measure_patch(ink, np.ones(ink.shape, bool), box, hatching.Frame(period=period))
    assert abs(f["line_angle"] + 45) <= 3


def test_blank_not_periodic():
    ink = hatch((200, 200), 20)
    blank = np.zeros_like(ink)
    rng = np.random.default_rng(0)
    blank[rng.integers(0, 200, 12), rng.integers(0, 200, 12)] = 1        # a few specks of dirt
    frame = hatching.Frame(period=20.0)
    box = (80, 80, 120, 120)
    on = np.ones(ink.shape, bool)
    hatched = hatching.measure_patch(ink, on, box, frame)["periodicity"]
    empty = hatching.measure_patch(blank, on, box, frame)["periodicity"]
    assert hatched > 0.5
    assert empty < hatched / 5


def test_lot_between_lines_borrows():
    # vertical lines 3 px wide, 20 px apart (x = 4-6, 24-26, ...); the lot is page x 8-19 (12 px), between two of them
    shape = (80, 80)
    ink = hatch(shape, 20, angle_deg=90.0, offset=6.0)
    assert ink[:, 4:7].all() and ink[:, 24:27].all() and not ink[:, 7:24].any()
    tr = affine(shape, 0.5, 0.5)                     # 1 ft per page px: a grid px (4 ft) is 4 x 4 page px
    lot = np.full((20, 20), -1, np.int32)
    lot[:, 2:5] = 7                                  # grid cols 2-4: page x 8-19
    lot[:, 6:] = 8
    box = (0, 0, 40, 40)
    shares = dict(hatching.patch_lots(box, tr, lot))
    assert not ink[:, 8:20].any()                    # the lot's own pixels hold no ink
    assert shares[7] > 0
    assert abs(sum(shares.values()) - (12 + 16) / 40) < 1e-9
    f = hatching.measure_patch(ink, np.ones(shape, bool), box, hatching.Frame(period=20.0))
    assert f["periodicity"] > 0.5                    # the patch shows the hatch the lot sits in


def test_seam_once():
    # two sheets 1 ft per px; A spans world x 0-239 ft, B 160-399 ft; A is drawn above B
    shape = (240, 240)
    ta, tb = affine(shape, 0.0, 0.0), affine(shape, 160.0, 0.0)
    crop = np.zeros(shape, bool)
    crop[10:230, 10:230] = True
    period = 12.0
    boxes = np.array(hatching.patch_grid(shape, period), float)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2 - 0.5
    above_a = hatching.Above.of(ta, crop)
    own_a = hatching.owned(centres, ta.to_world(centres), crop, [])
    own_b = hatching.owned(centres, tb.to_world(centres), crop, [above_a])
    wa = ta.to_world(centres[own_a])
    wb = tb.to_world(centres[own_b])
    a_x = (grid.X0 + 10, grid.X0 + 229)              # A's neatline in world x
    assert own_a.sum() > 0 and own_b.sum() > 0
    # B writes no patch whose centre lies inside A's neatline; A writes every patch centred inside it
    inside_a = (wb[:, 0] >= a_x[0] - 0.5) & (wb[:, 0] <= a_x[1] + 0.5)
    assert not inside_a.any()
    xi = np.round(centres[:, 0]).astype(int)
    yi = np.round(centres[:, 1]).astype(int)
    assert (own_a == crop[yi, xi]).all()
    # every world centre is measured once
    keys = {tuple(np.round(p, 3)) for p in wa} | {tuple(np.round(p, 3)) for p in wb}
    assert len(keys) == len(wa) + len(wb)


def white_rings(ink: np.ndarray) -> list[np.ndarray]:
    """Every white region the ink encloses, as the line work traces them (blocks_px.geojson): each connected region
    of paper with its territory into the lines around it (here one px), its outer boundary, page px."""
    n, lab = cv2.connectedComponents((ink == 0).astype(np.uint8), connectivity=4)
    rings = []
    for i in range(1, n):
        grown = cv2.dilate((lab == i).astype(np.uint8), np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2)
        if len(c) >= 3:
            rings.append(np.vstack([c, c[:1]]).astype(float))
    return rings


def test_lots_off_the_print_keep_the_outline_out():
    # a blank block: its printed outline 3 px wide around page px 39-120, its paper traced as one ring; the lots that
    # mark the block sit k px off the print, so they reach over the outline on two sides
    shape = (170, 170)
    ink = np.zeros(shape, np.uint8)
    ink[39:121, 39:121] = 1
    ink[42:118, 42:118] = 0
    rings = white_rings(ink)
    stroke = hatching.stroke_width(ink, rings)
    assert stroke == 3
    for k in (4, 8, 16):
        in_blocks = np.zeros(shape, np.uint8)
        in_blocks[40 + k:120 + k, 40 + k:120 + k] = 1
        inblock = hatching.inblock_mask(in_blocks, ink, rings, stroke, np.ones(shape, bool))
        seen = 0
        for box in hatching.patch_grid(shape, 12.0):
            f = hatching.measure_patch(ink, inblock, box, hatching.Frame(period=12.0, taper=stroke))
            if f["area"]:
                seen += 1
                assert f["periodicity"] < 0.1, (k, box)
                assert np.hypot(f["m2_re"], f["m2_im"]) < 0.2, (k, box)
        assert seen


@pytest.mark.parametrize("traced", ["every strip", "every other strip"])
def test_hatched_block_keeps_its_hatch(traced):
    # a 1R block: outline 3 px wide around page px 39-160, -45 degree hatch 12 px apart inside; the line work traces
    # each strip of paper between two hatch lines as a ring of its own (and the street round the block as another),
    # or on the scans only some of them (the rest fail its size and shape tests)
    shape = (200, 200)
    ink = np.zeros(shape, np.uint8)
    ink[39:161, 39:161] = 1
    ink[42:158, 42:158] = hatch(shape, 12)[42:158, 42:158]
    rings = white_rings(ink)
    assert len(rings) > 10
    stroke = hatching.stroke_width(ink, rings)
    if traced == "every other strip":
        rings = rings[::2]
    on = np.ones(shape, bool)
    frame = hatching.Frame(period=12.0, taper=stroke, wave_1r=np.pi / 4)
    for k in (0, 6):                                          # lots on the print, and 6 px off it
        in_blocks = np.zeros(shape, np.uint8)
        in_blocks[40 + k:160 + k, 40 + k:160 + k] = 1
        mask = hatching.inblock_mask(in_blocks, ink, rings, stroke, on)
        plain = cv2.erode(in_blocks, hatching.disk(stroke)).astype(bool)     # the lots alone, no outline taken out
        hatched = hatch(shape, 12) > 0
        outline = (ink > 0) & ~hatched & plain
        # the outline goes (a px may stay where a hatch line meets it: its end notches the paper's edge)
        assert (mask & outline).sum() <= 0.02 * outline.sum()
        inside = plain & hatched & (ink > 0)
        assert (mask & inside).sum() / inside.sum() > 0.85                   # the hatch stays (the old band: 0)
        if k == 0:                                                           # lots on the print: all of it inside
            inner = np.zeros(shape, bool)
            inner[42 + 2 * stroke:158 - 2 * stroke, 42 + 2 * stroke:158 - 2 * stroke] = True
            assert (mask[inner] == plain[inner]).all()
        got, ref = [], []
        for box in hatching.patch_grid(shape, 12.0):
            x0, y0, x1, y1 = box
            if plain[y0:y1, x0:x1].all():
                got.append(hatching.measure_patch(ink, mask, box, frame))
                ref.append(hatching.measure_patch(ink, plain, box, frame))
        assert len(got) > 10
        for key in ("periodicity", "dir1r_acf", "dir1r_share"):
            a, b = np.median([g[key] for g in got]), np.median([r[key] for r in ref])
            assert abs(a - b) < 0.05, (k, key, a, b)


def test_turned_sheet_same_grid():
    # one family of lines at -30 degrees; the same patch scanned a quarter turn round reads the same in the page
    # frame (the sheet upright)
    ink = hatch((48, 48), 12, angle_deg=-30.0)
    on = np.ones(ink.shape, bool)
    up = hatching.measure_patch(ink, on, (0, 0, 48, 48), hatching.Frame(period=12.0))
    turned = hatching.measure_patch(np.ascontiguousarray(np.rot90(ink)), on, (0, 0, 48, 48),
                                    hatching.Frame(period=12.0, turn=1))
    keys = [k for k in up if k.startswith("g") or k in ("m2_re", "m2_im", "m4_re", "m4_im", "line_angle")]
    a, b = np.array([up[k] for k in keys]), np.array([turned[k] for k in keys])
    assert np.allclose(a, b, rtol=1e-6, atol=1e-9 * np.abs(a).max())


def test_two_outlines_not_periodic():
    # two block outlines 17 px apart in a patch of side 24 (period 12, taper one 3 px stroke) are not a hatch; lines
    # 12 px apart are
    frame = hatching.Frame(period=12.0, taper=3.0)
    on = np.ones((24, 24), bool)
    two = np.zeros((24, 24), np.uint8)
    two[:, 4:7] = two[:, 21:24] = 1
    hatched = hatch((24, 24), 12, angle_deg=90.0)
    assert hatching.measure_patch(hatched, on, (0, 0, 24, 24), frame)["periodicity"] > 0.75
    assert hatching.measure_patch(two, on, (0, 0, 24, 24), frame)["periodicity"] < 0.25


def test_fallback_period():
    # periods measured on other sheets, carried to this sheet's resolution; the legend's sheet takes its 1R swatch
    measured = {"a": (10.0, 5.0), "b": (20.0, 2.5), "c": (12.0, 5.0), "d": (30.0, 2.5)}      # (period px, ft per px)
    assert hatching.fallback_period("x", 5.0, measured, "leg", 13.0) == (11.0, "median")
    assert hatching.fallback_period("y", 2.5, measured, "leg", 13.0) == (22.0, "median")
    assert hatching.fallback_period("leg", 5.0, measured, "leg", 13.0) == (13.0, "legend 1R swatch")


# --- a two-sheet run of the command -------------------------------------------------------------------------------

SHEETS = ("sheetA", "sheetB")
SHIFT_FT = 160.0           # B's left edge in world x (ft) from A's: they overlap 80 ft
PAGE = (300, 240)          # page px, 1 ft per px: the map in rows 0-239, the legend below it
LEGEND = {"1R": [12, 250, 60, 290], "HI": [70, 250, 118, 290]}     # boxes below A's map (outside its crop)


def _fixture(root):
    """Two sheets of a 1921-like map laid side by side in world feet, with their pages, placements, neatlines,
    reads, printed blocks, the lots and the legend."""
    work, warp, solver = root / "work" / "1921", root / "warp" / "1921", root / "solver" / "1921"
    # blocks: 40 ft squares every 52 ft (12 ft streets), each split into two 20 x 40 ft lots; lots west of
    # world x 200 ft are 1R (hatched), the rest 2R (blank)
    lots, rows = [], []
    k = 0
    for bx in range(0, 400 - 40, 52):
        for by in range(8, 240 - 40, 52):
            for j in range(2):
                x0 = grid.X0 + bx + 4 + 20 * j
                y1 = grid.Y1 - by - 4
                lots.append(shapely.box(x0, y1 - 40, x0 + 20, y1))
                rows.append({"blklot": f"{k // 2:04d}{j:03d}", "block_num": f"{k // 2:04d}",
                             "code": "1R" if bx < 200 else "2R"})
                k += 1
    table = pd.DataFrame(rows)
    lot = np.full((60, 100), -1, np.int32)
    for i, g in enumerate(lots):
        c0, r0 = grid.to_grid([g.bounds[0], g.bounds[3]])
        c1, r1 = grid.to_grid([g.bounds[2], g.bounds[1]])
        lot[int(round(r0)):int(round(r1)), int(round(c0)):int(round(c1))] = i
    (solver / "lots").mkdir(parents=True)
    np.savez(solver / "lots" / "lots.npz", lot=lot)
    table.assign(geometry=shapely.to_wkb(np.array(lots))).drop(columns="code").to_parquet(
        solver / "lots" / "table.parquet")
    (solver / "page" / "crops").mkdir(parents=True)
    frames = {}
    for s, x_ft in zip(SHEETS, (0.0, SHIFT_FT)):
        tr = affine(PAGE, x_ft + 0.5, 0.5)          # page px centres mid grid px
        ink = np.zeros(PAGE, np.uint8)
        hatched = hatch(PAGE, 12, offset=x_ft * np.sqrt(0.5))          # the same lines in world on both sheets
        for g, code in zip(lots, table["code"]):
            px = tr.grid_to_pixel(np.asarray(g.exterior.coords), PAGE)
            x0, y0 = np.floor(px.min(0)).astype(int)
            x1, y1 = np.ceil(px.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, PAGE[1]), min(y1, PAGE[0])
            if x1 <= x0 or y1 <= y0:
                continue
            if code == "1R":
                ink[y0:y1, x0:x1] = hatched[y0:y1, x0:x1]
            cv2.rectangle(ink, (x0, y0), (x1 - 1, y1 - 1), 1, 2)             # the printed outline, 2 px
        if s == SHEETS[0]:
            x0, y0, x1, y1 = LEGEND["1R"]
            ink[y0:y1, x0:x1] = hatch(PAGE, 12)[y0:y1, x0:x1]
            cv2.rectangle(ink, (x0, y0), (x1 - 1, y1 - 1), 1, 2)
            x0, y0, x1, y1 = LEGEND["HI"]
            sq = np.zeros(PAGE, np.uint8)
            sq[:, ::8] = sq[::8, :] = 1
            sq[:, 1::8] = sq[1::8, :] = 1
            ink[y0:y1, x0:x1] = sq[y0:y1, x0:x1]
            cv2.rectangle(ink, (x0, y0), (x1 - 1, y1 - 1), 1, 2)
        d = work / s
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "page.png"), np.where(ink > 0, 0, 255).astype(np.uint8))
        parcels = [{"blklot": b, "code": c, "confidence": 0.9, "via": "parcel"}
                   for b, c in zip(table["blklot"], table["code"])]
        (d / "district_pattern.json").write_text(json.dumps({"series": "1921", "sheet": s, "parcels": parcels}))
        feats = []
        for g in lots:
            px = tr.grid_to_pixel(np.asarray(g.exterior.coords), PAGE)
            feats.append({"type": "Feature", "properties": {},
                          "geometry": {"type": "Polygon", "coordinates": [px.tolist()]}})
        (d / "blocks_px.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        (warp / s).mkdir(parents=True)
        (warp / s / "transform.json").write_text(json.dumps(tr.to_json()))
        crop = np.zeros(PAGE, np.uint8)
        crop[12:228, 12:228] = 255
        cv2.imwrite(str(solver / "page" / "crops" / f"{s}.png"), crop)
        frames[s] = [[12.0, 12.0], [227.0, 12.0], [227.0, 227.0], [12.0, 227.0]]
    (solver / "page" / "frames.json").write_text(json.dumps(frames))
    legends = root / "legends.json"
    legends.write_text(json.dumps({"1921": {"sheet": SHEETS[0], "swatches": LEGEND}}))
    return work, warp, solver, legends


def _digest(folder) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def two_sheet_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("hatch")
    work, warp, solver, legends = _fixture(root)
    mp = pytest.MonkeyPatch()
    mp.setattr(hatching, "WORK", work)
    mp.setattr(hatching, "WARP", warp)
    mp.setattr(hatching, "SOLVER_1921", solver)
    mp.setattr(hatching, "LEGENDS_FILE", legends)
    mp.setattr(hatching, "draw_order", lambda sheets: [SHEETS[1], SHEETS[0]])     # A on top
    outs = []
    try:
        for k in range(2):
            out = root / f"out{k}"
            hatching.main(["--series", "1921", "--workers", "2", "--out", str(out)])
            outs.append(out)
    finally:
        mp.undo()
    return outs


def test_rerun_identical(two_sheet_run):
    first, second = (_digest(o) for o in two_sheet_run)
    assert first and first == second
    assert {f"patches/{s}.parquet" for s in SHEETS} <= set(first)
    assert {f"patch_lots/{s}.parquet" for s in SHEETS} <= set(first)


def test_run_writes_each_seam_patch_once(two_sheet_run):
    out = two_sheet_run[0]
    a = pd.read_parquet(out / "patches" / f"{SHEETS[0]}.parquet")
    b = pd.read_parquet(out / "patches" / f"{SHEETS[1]}.parquet")
    assert len(a) and len(b)
    # A's neatline: its page px 12-227 each way, i.e. (11.5, 227.5) on its page (1 ft per px, from X0 + 0.5, Y1 - 0.5);
    # B writes no patch centred strictly inside it
    ax, ay = b["cx"] - grid.X0 - 0.5, grid.Y1 - 0.5 - b["cy"]
    eps = 1e-6
    assert not ((ax > 11.5 + eps) & (ax < 227.5 - eps) & (ay > 11.5 + eps) & (ay < 227.5 - eps)).any()
    pl = pd.read_parquet(out / "patch_lots" / f"{SHEETS[0]}.parquet")
    assert set(pl["patch"]) <= set(a["patch"])
    assert (pl["share"] > 0).all() and (pl.groupby("patch")["share"].sum() <= 1 + 1e-9).all()
    # the 1R side's period is measured from its 12 px hatch
    sheets = json.loads((out / "sheets.json").read_text())
    assert abs(sheets["sheets"][SHEETS[0]]["period"] - 12) <= 1
