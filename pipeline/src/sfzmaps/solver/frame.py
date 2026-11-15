"""The printed frame of a 2000 zoning page, measured on its scan.  Every sheet prints the same rectangle; only the
scan's small rotation and offset differ, so each side is found as a straight, slightly tilted line.

Along each side, every ALONG_STEP_PX page px, the frame's inner edge is found inside a fixed search window: the inner
end of the first dark band (a run of at least MIN_BAND_PX dark px from the page edge inwards) for the left, right and
top sides; at the bottom, the top line of the legend box (the first row, from above, that is dark across more than
LEGEND_DARK_SHARE of a LEGEND_STRIP_PX wide strip).  A line is fitted to each side's points by least squares, dropping
points more than FIT_KEEP_PX off and refitting.  When the legend line's own fit is poor (fewer than LEGEND_MIN_KEPT of
its points kept, or a spread above LEGEND_MAX_SD_PX), the bottom is fitted to the points at the most common gap below
the top line instead.  The crop is the quadrilateral inside the four lines, FRAME_INSET_PX further in."""
import numpy as np

DARK_BELOW = 128                       # a page px darker than this is ink
# Where each side's inner edge is searched for: page px, the same on every page (x for left and right, y for top and
# bottom), and where along the side it is sampled.
WINDOWS = {"left": (40, 220), "right": (3080, 3260), "top": (400, 600), "bottom": (3900, 4100)}
ALONG_SIDES = (700, 3800)              # rows sampled for the left and right sides
ALONG_TOP_BOTTOM = (400, 2900)         # columns sampled for the top and bottom
ALONG_STEP_PX = 20
MIN_BAND_PX = 4                        # a dark run at least this long is the frame's band
LEGEND_STRIP_PX = 60                   # width of the strip the legend line is looked for in
LEGEND_DARK_SHARE = 0.7                # a row of the strip with more than this share dark is the legend line
FIT_KEEP_PX = 3.0                      # points further than this from the fitted line are dropped
FIT_ROUNDS = 5                         # fits at most, each without the points the last one left far off
LEGEND_MIN_KEPT = 0.6                  # the legend line's own fit is used when it keeps this share of its points ...
LEGEND_MAX_SD_PX = 1.0                 # ... with no more spread than this
GAP_BIN_PX = 2.0                       # histogram bins of the gap below the top line (the fallback bottom)
GAP_KEEP_PX = 3.0                      # the fallback bottom's points: those this close to the most common gap
FRAME_INSET_PX = 2.0                   # the crop's distance inside the fitted lines


def band_inner(dark: np.ndarray) -> int:
    """`dark`: dark flags along a scan line from the page edge inwards.  The index of the inner end of the first run
    of at least MIN_BAND_PX dark px, or -1 when there is none."""
    i, n = 0, len(dark)
    while i < n:
        if dark[i]:
            j = i
            while j < n and dark[j]:
                j += 1
            if j - i >= MIN_BAND_PX:
                return j - 1
            i = j
        else:
            i += 1
    return -1


def fit_line(s: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float, int]:
    """The least-squares line p = a + b*s through the points, refitted without points more than FIT_KEEP_PX off, up to
    FIT_ROUNDS times.  Returns (a, b), the spread (standard deviation) of the kept points' residuals, and how many
    points were kept."""
    keep = np.ones(len(s), bool)
    for _ in range(FIT_ROUNDS):
        A = np.c_[np.ones(keep.sum()), s[keep]]
        co, *_ = np.linalg.lstsq(A, p[keep], rcond=None)
        res = p - (co[0] + co[1] * s)
        new = np.abs(res) <= FIT_KEEP_PX
        if (new == keep).all():
            break
        keep = new
    return co, float(np.std(res[keep])), int(keep.sum())


def _side_points(dark: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions along the side and the inner edge found at each (x for left and right, y for top and bottom)."""
    a, b = WINDOWS[side]
    along = range(*ALONG_SIDES, ALONG_STEP_PX) if side in ("left", "right") else \
        range(*ALONG_TOP_BOTTOM, ALONG_STEP_PX)
    ss, pp = [], []
    for s in along:
        if side == "left":
            k = band_inner(dark[s, a:b])
            pos = a + k if k >= 0 else None
        elif side == "right":
            k = band_inner(dark[s, a:b][::-1])
            pos = b - 1 - k if k >= 0 else None
        elif side == "top":
            k = band_inner(dark[a:b, s])
            pos = a + k if k >= 0 else None
        else:
            rows = np.flatnonzero(dark[a:b, s:s + LEGEND_STRIP_PX].mean(1) > LEGEND_DARK_SHARE)
            pos = a + int(rows[0]) if len(rows) else None
        if pos is not None:
            ss.append(s)
            pp.append(pos)
    return np.array(ss, float), np.array(pp, float)


def _bottom_line(dark: np.ndarray, top: np.ndarray) -> np.ndarray:
    """The legend box's top line: its own fit where that is clean, else the fit through the points at the most common
    gap below the top line."""
    s, p = _side_points(dark, "bottom")
    co, sd, kept = fit_line(s, p)
    if kept >= LEGEND_MIN_KEPT * len(s) and sd <= LEGEND_MAX_SD_PX:
        return co
    gaps = p - (top[0] + top[1] * s)
    hist, edges = np.histogram(gaps, bins=np.arange(gaps.min() - 1, gaps.max() + 3, GAP_BIN_PX))
    common = edges[np.argmax(hist)] + GAP_BIN_PX / 2
    near = np.abs(gaps - common) <= GAP_KEEP_PX
    return fit_line(s[near], p[near])[0]


def _meet(xline, yline) -> list[float]:
    """Where the line x = a + b*y meets the line y = c + d*x."""
    a, b = xline
    c, d = yline
    x = (a + b * c) / (1 - b * d)
    return [float(x), float(c + d * x)]


def find_frame(page: np.ndarray) -> np.ndarray:
    """The crop of a scanned page (grayscale, as page.png holds it): the frame's four corners, FRAME_INSET_PX inside
    its fitted lines, as a (4, 2) float array of page px (x, y) in the order top-left, top-right, bottom-right,
    bottom-left."""
    dark = np.asarray(page) < DARK_BELOW
    left, right, top = (fit_line(*_side_points(dark, side))[0] for side in ("left", "right", "top"))
    bottom = _bottom_line(dark, top)
    # left and right: x = a + b*y; top and bottom: y = a + b*x; each moved FRAME_INSET_PX inwards
    left = (left[0] + FRAME_INSET_PX, left[1])
    right = (right[0] - FRAME_INSET_PX, right[1])
    top = (top[0] + FRAME_INSET_PX, top[1])
    bottom = (bottom[0] - FRAME_INSET_PX, bottom[1])
    return np.array([_meet(left, top), _meet(right, top), _meet(right, bottom), _meet(left, bottom)])
