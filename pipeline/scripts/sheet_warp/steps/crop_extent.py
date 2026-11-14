"""regprep2's crop extent without placing every block-line point (speed E8).  The extent is the box round the points
that land on the page (with a usable direction), plus 60 px; only its extreme points matter.  Candidates: each point
placed with the fit's linear part alone (to_pixel's own first guess), those within M px of the page; the K most
extreme along x and along y each way; then those candidates placed exactly (to_pixel), filtered as before.
extent(T0, pts, ta, tb, W, H) -> (x0, y0, x1, y1, sel): sel = the candidates' indices (exactly placed)."""
import numpy as np

K, M = 3000, 200


def extent(T0, pts, ta, tb, W, H):
    T0._ensure_fit()
    lin = T0._pm + (pts - T0.to_world(T0._pm[None])) @ T0._lin_inv.T
    near = np.flatnonzero((lin[:, 0] > -M) & (lin[:, 0] < W + M) & (lin[:, 1] > -M) & (lin[:, 1] < H + M))
    ln = lin[near]
    pick = set()
    for c in (0, 1):
        o = np.argsort(ln[:, c], kind="stable")
        pick.update(o[:K].tolist()); pick.update(o[-K:].tolist())
    sel = near[np.array(sorted(pick))]
    px = T0.to_pixel(pts[sel])
    dv = T0.to_pixel(tb[sel]) - T0.to_pixel(ta[sel])
    pdir = np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180
    ok = np.isfinite(px).all(1) & np.isfinite(pdir) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    p = px[ok]
    x0, y0 = max(0, int(p[:, 0].min()) - 60), max(0, int(p[:, 1].min()) - 60)
    x1, y1 = min(W, int(p[:, 0].max()) + 60), min(H, int(p[:, 1].max()) + 60)
    return x0, y0, x1, y1, sel[ok], px[ok], pdir[ok]
