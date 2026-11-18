import numpy as np

from sfzmaps.solver import frame, grid, page

X0, Y1 = 5979300.0, 2124100.0     # the grid's top-left corner, EPSG:2227 ft


def _meet(xline, yline):
    """Where x = a + b*y meets y = c + d*x."""
    a, b = xline
    c, d = yline
    x = (a + b * c) / (1 - b * d)
    return [x, c + d * x]


TILT = 0.004   # the scan turned by 0.004 rad: horizontal lines go down 4 px per 1000, vertical ones left


def _frame_page(bars: bool = False) -> np.ndarray:
    """A 4200 x 3300 page with the frame drawn where find_frame looks: the left, right and top bands 8 px wide, the
    legend box's top line 3 px thick.  `bars`: 3 px bars of map ink 55-95 px above the legend line along the bottom's
    first 1800 px, each 120 px long at its own row, so the legend line is found at fewer than 60% of the points."""
    h, w = 4200, 3300
    pg = np.full((h, w), 255, np.uint8)
    ys, xs = np.arange(h), np.arange(w)
    left = np.rint(120 - TILT * ys).astype(int)       # the left band's inner (last dark) column
    right = np.rint(3170 - TILT * ys).astype(int)     # the right band's inner (first dark) column
    top = np.rint(500 + TILT * xs).astype(int)        # the top band's inner (last dark) row
    legend = np.rint(4000 + TILT * xs).astype(int)    # the legend line's first dark row
    for y in ys:
        pg[y, left[y] - 7:left[y] + 1] = 0
        pg[y, right[y]:right[y] + 8] = 0
    for x in xs:
        pg[top[x] - 7:top[x] + 1, x] = 0
        pg[legend[x]:legend[x] + 3, x] = 0
    if bars:
        for k, x0 in enumerate(range(400, 2200, 120)):
            r = 3905 + (k * 13) % 40
            pg[r:r + 3, x0:x0 + 120] = 0
    return pg


def _drawn_corners() -> np.ndarray:
    inset = frame.FRAME_INSET_PX
    L, R = (120 + inset, -TILT), (3170 - inset, -TILT)
    T, B = (500 + inset, TILT), (4000 - inset, TILT)
    return np.array([_meet(L, T), _meet(R, T), _meet(R, B), _meet(L, B)])


def test_find_frame_on_a_drawn_frame():
    got = frame.find_frame(_frame_page())
    assert got.shape == (4, 2)
    assert np.abs(got - _drawn_corners()).max() < 1.0


def test_find_frame_takes_the_bottom_at_the_common_gap_below_the_top_when_map_ink_hides_the_legend_line():
    got = frame.find_frame(_frame_page(bars=True))
    assert np.abs(got - _drawn_corners()).max() < 1.0


def test_grid_round_trip_and_north_up():
    assert np.allclose(grid.to_grid([X0, Y1]), [0, 0])
    assert np.allclose(grid.to_grid([X0, Y1 - 4.0]), [0, 1])        # 4 ft south: one row down
    assert np.allclose(grid.to_grid([X0 + 4.0, Y1]), [1, 0])        # 4 ft east: one column right
    pts = np.array([[6000000.5, 2100000.25], [5990000.0, 2110001.0]])
    assert np.allclose(grid.to_map(grid.to_grid(pts)), pts)
    assert (grid.WIDTH, grid.HEIGHT, grid.FT_PER_PX) == (11900, 9825, 4.0)


class Shifted:
    """A sheet laid one page px per grid px, north up: the centre of grid px (col, row) is page px
    (col - dx, row - dy).  Written out in feet, not through the grid module."""

    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy

    def to_pixel(self, w):
        w = np.atleast_2d(np.asarray(w, float))
        return np.c_[(w[:, 0] - X0) / 4.0 - 0.5 - self.dx, (Y1 - w[:, 1]) / 4.0 - 0.5 - self.dy]

    def to_world(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        return np.c_[X0 + (p[:, 0] + self.dx + 0.5) * 4.0, Y1 - (p[:, 1] + self.dy + 0.5) * 4.0]


def _on_grid(a, dx, dy, shape):
    """A page-sized array moved to where Shifted(dx, dy) lays it on a grid of `shape` (False / 0 elsewhere)."""
    out = np.zeros(shape, a.dtype)
    out[dy:, dx:] = a[:shape[0] - dy, :shape[1] - dx]
    return out


QUAD = np.array([[50.0, 40.0], [350.0, 40.0], [350.0, 250.0], [50.0, 250.0]])   # page px: TL, TR, BR, BL


def test_a_sheet_lands_where_its_transform_says():
    shape = (300, 420)
    pg = np.full((300, 400), 255, np.uint8)
    pg[150, 200] = 0                  # in the crop
    pg[115, 120] = 0                  # in the cut box
    pg[20, 20] = 0                    # outside the frame
    pg[60:70, 300:303] = 90           # dark grey: ink too (below 128)
    pg[80, 80] = 200                  # light grey: paper
    cut = [100, 100, 140, 130]        # x0, y0, x1, y1, inclusive
    crop = page.crop_mask(pg.shape, QUAD, [cut])
    assert crop[40, 50] and crop[250, 350] and not crop[39, 50] and not crop[40, 351]
    assert not crop[100, 100] and not crop[130, 140] and crop[131, 140]

    tr = Shifted(30, 20)
    ink, sheets = np.zeros(shape, bool), np.zeros(shape, np.uint8)
    page.lay_sheet(tr, pg, crop, ink, sheets)
    assert ink[170, 230]                                    # page (200, 150) at grid col 230, row 170
    assert ink.sum() == 1 + 30                              # nothing from the cut box or outside the frame
    assert np.array_equal(sheets, _on_grid(crop, 30, 20, shape).astype(np.uint8))
    assert np.array_equal(ink, _on_grid(crop & (pg < 128), 30, 20, shape))

    box = page.grid_box(tr, QUAD, shape)                     # only where the crop can reach: the same result
    assert box != (0, shape[0], 0, shape[1])
    ink2, sheets2 = np.zeros(shape, bool), np.zeros(shape, np.uint8)
    page.lay_sheet(tr, pg, crop, ink2, sheets2, box)
    assert np.array_equal(ink2, ink) and np.array_equal(sheets2, sheets)


def test_two_sheets_union_and_count():
    shape = (300, 420)
    a = np.full((300, 400), 255, np.uint8)
    b = np.full((300, 400), 255, np.uint8)
    a[150, 200] = 0                   # grid (170, 230), where b is blank: still ink
    b[150, 200] = 0                   # grid (190, 240)
    a[160, 240] = 0                   # grid (180, 270) ...
    b[140, 230] = 0                   # ... and b at the same grid px: one ink px
    crop = page.crop_mask(a.shape, QUAD, [])
    ink, sheets = np.zeros(shape, bool), np.zeros(shape, np.uint8)
    page.lay_sheet(Shifted(30, 20), a, crop, ink, sheets)
    page.lay_sheet(Shifted(40, 40), b, crop, ink, sheets)
    assert ink[170, 230] and ink[190, 240] and ink[180, 270] and ink.sum() == 3
    want = _on_grid(crop, 30, 20, shape).astype(np.uint8) + _on_grid(crop, 40, 40, shape)
    assert np.array_equal(sheets, want) and sheets.max() == 2
