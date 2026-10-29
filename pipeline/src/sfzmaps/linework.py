"""Block outlines from a scanned sheet, in pixel space.

A "street-corridor" segmentation rather than a pure ink one:

1. Binarize ink against an estimated paper level (handles the 1948 photos'
   uneven lighting, page curvature and gutter shadow; on clean 1-bit PDF
   renders it reduces to a plain threshold).
2. Classify the *white* space into street and block interiors
   (`classify_white`).  Streets are channels joining into a network (on
   photos, seeded by long straight white runs); interiors are compact white
   regions enclosed by outlines.  Text, district lines and hatching are
   ink, so they never make white look like street; they only split it.
   Ambiguous thin pieces are judged by connectivity and by what lies across
   the ink around them.
3. Anchors = block interiors + solid fills (black, hatched, stippled).  Ink
   counts as block material only near an anchor, so lines and lettering in
   the middle of a street belong to no block (`block_material`).
4. Each anchor claims the material geodesically nearest to it; territories
   that share a long boundary (white and black parts of one block, halves
   split by a district line or a single-line alley) are regrouped, while
   blocks touching only through a letter stroke or line stay apart.
5. Each group is smoothed, polygonised with corners restored by
   intersecting lines fitted to the straight edges, and shrunk by half a
   stroke when outlined (the outline's centreline is the block edge).

All sizes are in pixels, tuned for 300 dpi renders; the 1948 colour photos
are at a similar scale.  Street width and stroke width are measured per
sheet.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon
from scipy import ndimage as ndi
from skimage.segmentation import watershed

# Same > 30 deg turning-angle corner rule as the DataSF reference corners.
from sfzmaps.sources.datasf import _ring_corners

# Ink closing that seals 1-2 px breaks in printed outlines (photos only).
SEAL_PX = 3
# White farther than CHANNEL_PX/2 from ink forms "cores"; a white channel
# narrower than CHANNEL_PX (lot-line gap, crack, break in an outline) cannot
# join two cores.
CHANNEL_PX = 11
# A compact white region at least DEFINITE_FACTOR x the sheet's typical
# street half-width thick is a block interior outright; thinner compact
# regions down to MIN_INTERIOR_HALFWIDTH are judged by their surroundings.
DEFINITE_FACTOR = 2.0
MIN_INTERIOR_HALFWIDTH = 10.0
# On clean renders, an elongated region no thicker than STREETLIKE_FACTOR x
# the street half-width is a street segment boxed in by district lines or
# lettering, never a block.  (Not on photos: there street seeds come from
# long runs, and narrow blocks are about as wide as streets.)
STREETLIKE_FACTOR = 1.3
STREETLIKE_ASPECT = 3.0
INTERIOR_SOLIDITY = 0.7      # filled area / convex-hull area
INTERIOR_FILL = 0.75         # pixel area / filled area (rejects rings)
INTERIOR_MAX_ASPECT = 12.0   # longer and thinner is a street piece
# Labels flood only through white at least this far from ink (px).  On
# photos this is high so that hatch gaps behind a faded outline are not
# flooded from the street; on clean renders it is low so that the flood
# squeezes past street names printed inside narrow streets.
FLOOD_MIN_DIST = 2.0
FLOOD_MIN_DIST_PHOTO = 5.0
# Enclosed narrow white up to this area is a counter / pattern cell (block
# material); larger pieces are judged by what lies ACROSS_PX beyond them.
SMALL_CELL_AREA = 150
ACROSS_PX = 12
ACROSS_STREET_FRACTION = 0.25
ACROSS_INTERIOR_FRACTION = 0.25
# A thin candidate with at least this share of its outline touching street
# runs alongside the street (a margin the photo street seeds missed), so it
# is street.  An alley or lot gap touches street only at its mouth.
ALONGSIDE_FRACTION = 0.3
# Solid fills must survive an opening this wide (thicker than bold letters).
SOLID_PX = 25
# Outlines / district lines / lettering count as block only this close to
# a block interior or solid fill (>= outline + heavy line width).
ANCHOR_REACH_PX = 6
# Anchor territories sharing at least this fraction of the smaller one's
# perimeter belong to the same block.
SHARE_FRACTION = 0.12
# Groups smaller than this fraction of the median block are fragments if they
# touch a bigger group.
FRAGMENT_FRACTION = 0.15
# Final clean-up closing + opening of each block mask (px).
CLEAN_PX = 9
MIN_BLOCK_AREA = 400
MIN_BLOCK_SOLIDITY = 0.5
SIMPLIFY_PX = 2.0
# Photos (1948) have tinted, textured paper and faded, dotted outlines.
PHOTO_TINT_FRACTION = 0.3
# In photos, white runs at least this long (along the grid or diagonals)
# mark streets: they cross intersections, while a run inside a block stops
# at the block's far side even when one side is dotted.
STREET_RUN_PX = 400
RUN_ANGLES = (0.0, 90.0, 45.0, 135.0)
RUN_CLEAN_PX = 5             # opening that drops 1-2 px run threads through dotted outlines
# Paper-level estimate in `binarize`.
BG_DOWNSAMPLE = 8            # work on an 8x downsample for speed
BG_MAX_WIN = 61              # max-filter window (x8 = ~500 px, wider than any solid block)
BG_BLUR_WIN = 31             # smoothing of the paper estimate
BG_PAPER_PCT = 90            # percentile taken as the clean paper level
STREET_RUN_RANGE = (8, 150)  # white run lengths (px) considered when measuring street width
STROKE_RANGE_HALFPX = (2, 12)  # ridge values (half-px bins) considered thin strokes
MIN_INTERSECT_DEG = 20.0     # edges closer to parallel than this are not intersected
# Corner extraction (`block_corners_px`), all scaled by the sheet's street width.
STREET_FT = 64.0             # typical SF street right-of-way, ft (sets px-per-ft)
SIMPLIFY_FT = 2.0            # same simplification as datasf.block_corners, in ft
MIN_EDGE_STREET_FRAC = 0.5   # corner edges shorter than this x street width are bumps/notches
POLY_EPS_STREET_FRAC = 0.2   # Douglas-Peucker tolerance for outlines, x street width
CLOSE_STREET_FRAC = 0.6      # per-block closing disk, x street width
SMALL_EPS_FRAC = 0.1         # outline tolerance cap, x sqrt(block area)
# In-and-out notches / bumps on a straight edge (lettering at the block
# edge, line stubs): at most this wide and deep, x street width, with the
# outline resuming within NOTCH_TOL of the same line.
NOTCH_WIDTH_STREET_FRAC = 2.5
NOTCH_DEPTH_STREET_FRAC = 1.2
NOTCH_TOL_STREET_FRAC = 0.15
NOTCH_MAX_VERTS = 4
DEFAULT_STREET_PX = 25.0     # fallback street width when it cannot be measured
# Lettering polygons: small (vs the median block) and not convex.
LETTER_AREA_FRAC = 0.15
LETTER_SOLIDITY = 0.85


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g


def binarize(img: np.ndarray, seal: bool | None = None) -> np.ndarray:
    """Bool ink mask, robust to uneven illumination.

    The paper level is estimated as a large-window max filter on an 8x
    downsample (window ~500 px, wider than any solid block, so it always sees
    some paper), then the page is divided by it and Otsu-thresholded.  A
    plain adaptive threshold would hollow out large solid-black blocks.
    """
    g = _gray(img)
    h, w = g.shape
    f = BG_DOWNSAMPLE
    small = cv2.resize(g, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    bg = cv2.dilate(small, np.ones((BG_MAX_WIN, BG_MAX_WIN), np.uint8))
    bg = cv2.blur(bg, (BG_BLUR_WIN, BG_BLUR_WIN))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    paper = float(np.percentile(bg, BG_PAPER_PCT))
    bg = np.maximum(bg, 0.5 * paper)
    norm = np.clip(g.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink = norm <= t   # <=: Otsu returns t = min value on two-level images
    if seal is None:
        seal = is_photo(img)
    # Photos: seal 1-2 px breaks in faded outlines.  Clean renders keep
    # hairline gaps, which are often the only trace of an alley between two
    # outlined strips.
    if seal and SEAL_PX > 1:
        ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEAL_PX, SEAL_PX))) > 0
    return ink


def is_photo(img: np.ndarray) -> bool:
    """True for camera photos of printed sheets (tinted, textured paper).

    Clean PDF renders are almost pure black/white; the 1948 photos have
    most pixels in the 200-245 band.
    """
    g = _gray(img)[::4, ::4]
    return float(((g >= 200) & (g < 245)).mean()) > PHOTO_TINT_FRACTION


def _hruns(m: np.ndarray) -> np.ndarray:
    """Per-pixel length of the horizontal run of True pixels containing it."""
    h, w = m.shape
    f = np.zeros((h, w + 2), np.int8)
    f[:, 1:-1] = m
    d = np.diff(f, axis=1)
    ys, xs = np.nonzero(d == 1)
    ye, xe = np.nonzero(d == -1)
    lens = (xe - xs).astype(np.int32)
    acc = np.zeros((h, w + 1), np.int32)
    acc[ys, xs] += lens
    acc[ye, xe] -= lens
    return np.cumsum(acc, axis=1)[:, :w] * m


def _runs(m: np.ndarray, angle: float) -> np.ndarray:
    """Run lengths along direction `angle` (degrees, counter-clockwise)."""
    if angle % 180 == 0:
        return _hruns(m)
    if angle % 180 == 90:
        return _hruns(m.T).T
    h, w = m.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(rot[0, 0]), abs(rot[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    rot[0, 2] += nw / 2 - w / 2
    rot[1, 2] += nh / 2 - h / 2
    r = cv2.warpAffine(m.astype(np.uint8), rot, (nw, nh), flags=cv2.INTER_NEAREST)
    rr = _hruns(r > 0).astype(np.float32)
    return cv2.warpAffine(rr, cv2.invertAffineTransform(rot), (w, h), flags=cv2.INTER_NEAREST)


def street_runs(white: np.ndarray) -> np.ndarray:
    """White pixels on long straight white runs (photo street seeds)."""
    long = np.zeros(white.shape, bool)
    for a in RUN_ANGLES:
        long |= _runs(white, a) >= STREET_RUN_PX
    # Drop 1-2 px threads that slip through aligned gaps in dotted outlines.
    return cv2.morphologyEx(long.astype(np.uint8), cv2.MORPH_OPEN, _disk(RUN_CLEAN_PX)) > 0


def street_width(white: np.ndarray) -> float:
    """Typical street width (px): the commonest horizontal/vertical white run
    length between 8 and 150 px.  Runs across streets share one width;
    runs across blocks vary with block size."""
    lens = []
    for m in (white[::2], white.T[::2]):
        f = np.zeros((m.shape[0], m.shape[1] + 2), np.int8)
        f[:, 1:-1] = m
        d = np.diff(f, axis=1)
        lens.append(np.nonzero(d.ravel() == -1)[0] - np.nonzero(d.ravel() == 1)[0])
    ln = np.concatenate(lens)
    lo, hi = STREET_RUN_RANGE
    ln = ln[(ln >= lo) & (ln <= hi)]
    if not len(ln):
        return 2 * MIN_INTERIOR_HALFWIDTH
    hist = np.convolve(np.bincount(ln, minlength=hi + 1).astype(float), np.ones(3) / 3, "same")
    return float(np.argmax(hist))


def stroke_width(ink: np.ndarray) -> float:
    """Typical thin-line width (px), independent of line direction.

    On each line's medial ridge the distance to paper is about half the
    width; the commonest ridge value among thin strokes gives the width.
    """
    d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = (d > 0) & (d >= cv2.dilate(d, np.ones((3, 3), np.uint8)))
    v = np.round(d[ridge] * 2).astype(int)              # half-px bins
    v = v[(v >= STROKE_RANGE_HALFPX[0]) & (v <= STROKE_RANGE_HALFPX[1])]
    if not len(v):
        return 0.0
    return max(1.0, float(np.argmax(np.bincount(v))) - 1.0)


def _disk(d: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def classify_white(ink: np.ndarray, outside: np.ndarray, photo: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Split white space into (street, block interior) bool masks.

    White that is neither (pattern cells, letter counters, dead-end gaps)
    belongs to whatever ink it sits in.  `outside` counts as street.
    """
    white = ~ink & ~outside
    dist = cv2.distanceTransform(white.astype(np.uint8), cv2.DIST_L2, 5)
    edge = np.zeros(white.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    if outside.any():
        edge |= cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    seeds = street_runs(white) if photo else np.zeros(white.shape, bool)
    # Cores: white at least CHANNEL_PX/2 from ink, minus known street.
    cores = ((dist > CHANNEL_PX / 2) & ~seeds).astype(np.uint8)
    n, lab = cv2.connectedComponents(cores, connectivity=4)
    markers = lab + 1
    markers[cores == 0] = 0
    markers[seeds] = 1
    # Grow street seeds and cores over the white each is geodesically
    # closest to, through channels at least 2*FLOOD_MIN_DIST wide.  A gap in
    # an outline is split between the street and the interior behind it; an
    # alley drawn as a narrow channel joins the streets it opens onto.
    fmd = FLOOD_MIN_DIST_PHOTO if photo else FLOOD_MIN_DIST
    flood = (dist >= fmd) | (markers > 0)
    ws = watershed(np.zeros(white.shape, np.uint8), markers=markers, mask=flood, connectivity=1)
    # The thin white rim along ink takes the label of the nearest flooded pixel.
    d, (iy, ix) = ndi.distance_transform_edt(ws == 0, return_indices=True)
    rim = white & (ws == 0) & (d <= fmd + 1.5)
    ws[rim] = ws[iy[rim], ix[rim]]

    street_hw = street_width(white) / 2.0
    definite_hw = DEFINITE_FACTOR * street_hw

    # Every flooded region is street (touches the edge, or is not compact:
    # the network, ring-shaped half-streets), a definite block interior
    # (compact and clearly thicker than a street), or a candidate.
    is_street = np.ones(n + 1, bool)          # index = marker label
    is_interior = np.zeros(n + 1, bool)
    cand = np.zeros(n + 1, bool)
    can_anchor = np.zeros(n + 1, bool)        # thick enough to anchor a block
    touches = np.zeros(n + 1, bool)
    touches[np.unique(ws[edge])] = True
    for lbl, sl in enumerate(ndi.find_objects(ws), start=1):
        if lbl < 2 or sl is None or touches[lbl]:
            continue
        sub = (ws[sl] == lbl).astype(np.uint8)
        area = float(sub.sum())
        hw = float(dist[sl][sub > 0].max())
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        filled = max(cv2.contourArea(c), 1.0)
        hull = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        compact = (filled / hull >= INTERIOR_SOLIDITY and area / filled >= INTERIOR_FILL
                   and aspect <= INTERIOR_MAX_ASPECT)
        if not photo and hw <= STREETLIKE_FACTOR * street_hw and aspect >= STREETLIKE_ASPECT:
            continue                          # a street segment boxed in by district lines
        if compact and hw >= definite_hw:
            is_street[lbl], is_interior[lbl] = False, True
        elif compact or hw < MIN_INTERIOR_HALFWIDTH:
            is_street[lbl], cand[lbl] = False, True
            can_anchor[lbl] = compact and hw >= MIN_INTERIOR_HALFWIDTH
    street = (ws > 0) & is_street[ws]
    interior = (ws > 1) & is_interior[ws]

    # Candidates -- regions about as thin as a street, and white the flood
    # never reached -- are judged by their surroundings.  They include
    # alleys, gaps between adjacent blocks, half-streets beside a district
    # line, street segments boxed in by district lines or by their own
    # printed names, narrow blocks, strips of a block cut off by a district
    # line, hatch gaps, lot-line gaps and letter counters.
    #  * A thin channel opening onto street at two or more separate places
    #    is a street: it separates the shapes on either side.  (Not applied
    #    to compact regions thick enough to be a block: on faded photos a
    #    block's outline is often broken in several places.)
    #  * Small pieces (counters, pattern cells) are block material.
    #  * A thin piece whose outline lies largely against street runs
    #    alongside it: a street margin the photo street seeds missed (they
    #    stop short of the kerb where specks interrupt the long runs).  It is
    #    street; otherwise the block grows a strip of street.
    #  * Otherwise look ACROSS_PX across the surrounding ink.  A block (or a
    #    piece of one cut off by a district line) has street beyond its
    #    outline.  A boxed-in street segment or half-street has block
    #    interiors beyond it but hardly any known street.
    orphan = white & (ws == 0)
    no, olab = cv2.connectedComponents(orphan.astype(np.uint8), connectivity=4)
    clab = np.where(orphan, olab + n, np.where(cand[ws], ws, 0)).astype(np.int32)
    clab[~white] = 0
    nlab = n + no
    anchor_ok = np.zeros(nlab + 1, bool)
    anchor_ok[:n + 1] = can_anchor
    decided = np.zeros(nlab + 1, np.int8)     # 1 street, 2 block
    objs = ndi.find_objects(clab)
    sizes = np.bincount(clab.ravel(), minlength=nlab + 1)
    r_in, r_out = _disk(2 * 3 + 1), _disk(2 * ACROSS_PX + 1)
    h, w = white.shape
    for _ in range(2):
        st8 = cv2.dilate(street.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touch = st8 & (clab > 0)
        nt, tlab = cv2.connectedComponents(touch.astype(np.uint8), connectivity=8)
        contacts = np.zeros(nlab + 1, np.int64)
        if nt > 1:
            pairs = np.unique(np.stack([clab[touch], tlab[touch]]), axis=1)
            contacts = np.bincount(pairs[0], minlength=nlab + 1)
        for i, sl in enumerate(objs, start=1):
            if sl is None or decided[i]:
                continue
            if contacts[i] >= 2 and not anchor_ok[i]:
                decided[i] = 1
                continue
            if sizes[i] <= SMALL_CELL_AREA:
                decided[i] = 2
                continue
            y0, y1 = max(0, sl[0].start - ACROSS_PX), min(h, sl[0].stop + ACROSS_PX)
            x0, x1 = max(0, sl[1].start - ACROSS_PX), min(w, sl[1].stop + ACROSS_PX)
            comp = (clab[y0:y1, x0:x1] == i).astype(np.uint8)
            if not anchor_ok[i]:
                rim_px = comp.astype(bool) & ~(cv2.erode(comp, np.ones((3, 3), np.uint8)) > 0)
                if (rim_px & st8[y0:y1, x0:x1]).sum() >= ALONGSIDE_FRACTION * max(int(rim_px.sum()), 1):
                    decided[i] = 1
                    continue
            ring = (cv2.dilate(comp, r_out) > 0) & ~(cv2.dilate(comp, r_in) > 0) & white[y0:y1, x0:x1]
            n_white = max(int(ring.sum()), 1)
            f_str = (ring & street[y0:y1, x0:x1]).sum() / n_white
            f_int = (ring & interior[y0:y1, x0:x1]).sum() / n_white
            if f_str >= ACROSS_STREET_FRACTION:
                decided[i] = 2
            elif f_int >= ACROSS_INTERIOR_FRACTION:
                decided[i] = 1
        street |= (decided == 1)[clab] & (clab > 0)
        interior |= ((decided == 2) & anchor_ok)[clab] & (clab > 0)
    # Undecided after two passes: thick ones are blocks, the rest material.
    interior |= ((decided == 0) & anchor_ok)[clab] & (clab > 0)
    return street | outside, interior


def street_mask(ink: np.ndarray, outside: np.ndarray, photo: bool = False) -> np.ndarray:
    """Bool mask of street (and outside-the-map) white space."""
    return classify_white(ink, outside, photo)[0]


def block_material(ink: np.ndarray, outside: np.ndarray, photo: bool = False
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(material, anchor labels): everything that belongs to some block.

    Anchors are block interiors and solid fills (black, hatched, stippled:
    ink plus its enclosed cells, thick enough to survive an opening wider
    than any lettering stroke).  Other non-street ink -- outlines, district
    lines through blocks, lettering inside blocks -- is kept only within
    ANCHOR_REACH_PX of an anchor.  So a heavy district line drawn down the
    middle of a street is dropped, and a bold "R-4" or a street name that
    spans a street is cut a few px beyond each outline.
    """
    street, interior = classify_white(ink, outside, photo)
    stuff = ~street & ~interior                       # ink + cells
    solid = cv2.morphologyEx(stuff.astype(np.uint8), cv2.MORPH_OPEN, _disk(SOLID_PX)) > 0
    _, ilab = cv2.connectedComponents(interior.astype(np.uint8), connectivity=4)
    ns, slab = cv2.connectedComponents((solid & ~interior).astype(np.uint8), connectivity=4)
    anchors = np.where(interior, ilab + ns, slab).astype(np.int32)   # disjoint label ranges
    near = cv2.dilate((anchors > 0).astype(np.uint8), _disk(2 * ANCHOR_REACH_PX + 1)) > 0
    return (anchors > 0) | (stuff & near), anchors


def _shared_boundaries(lab: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """4-neighbour contact lengths between labels, and each label's perimeter."""
    pairs: dict[tuple[int, int], int] = {}
    perim = np.zeros(int(lab.max()) + 1, np.int64)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        diff = a != b
        np.add.at(perim, a[diff], 1)
        np.add.at(perim, b[diff], 1)
        m = diff & (a > 0) & (b > 0)
        if m.any():
            lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
            keys, counts = np.unique(np.stack([lo, hi]), axis=1, return_counts=True)
            for (u, v), c in zip(keys.T.tolist(), counts.tolist()):
                pairs[(u, v)] = pairs.get((u, v), 0) + c
    return pairs, perim


def _group_territories(lab: np.ndarray) -> np.ndarray:
    """Merge anchor territories that share a long boundary.

    Two parts of one block (a white part and a black part, two halves split
    by a district line or by a single-line alley) meet along a whole side.
    Two blocks bridged by lettering or a line crossing a street meet only
    where the strokes cross, a small fraction of their perimeter.
    """
    pairs, perim = _shared_boundaries(lab)
    parent = np.arange(len(perim))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (u, v), c in pairs.items():
        if c >= SHARE_FRACTION * min(perim[u], perim[v]):
            parent[find(u)] = find(v)
    # A tiny group touching another one is a fragment of it (a letter
    # counter, a corner boxed off by a district line, the tip of a block
    # that runs off the sheet): fold it into the neighbour it shares the most
    # boundary with.  Real small blocks are surrounded by street and touch
    # nothing, so they are kept.
    roots = np.array([find(i) for i in range(len(perim))])
    area = np.bincount(roots[lab].ravel(), minlength=len(perim))
    area[0] = 0
    big = area[area >= MIN_BLOCK_AREA]
    if len(big):
        small = area < FRAGMENT_FRACTION * np.median(big)
        best: dict[int, tuple[int, int]] = {}
        for (u, v), c in pairs.items():
            ru, rv = roots[u], roots[v]
            if ru == rv:
                continue
            for a, b in ((ru, rv), (rv, ru)):
                if small[a] and area[b] > area[a] and c > best.get(a, (0, 0))[0]:
                    best[a] = (c, b)
        for a, (_, b) in best.items():
            parent[find(a)] = find(b)
        roots = np.array([find(i) for i in range(len(perim))])
    roots[0] = 0
    return roots[lab]


def _fit_line(pts: np.ndarray):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < math.sin(math.radians(MIN_INTERSECT_DEG)):
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def _polygon(contour: np.ndarray, eps: float) -> Polygon | None:
    """Polygon from a dense contour; corners rounded by smoothing are restored
    by intersecting lines fitted to the adjacent straight edges."""
    pts = contour[:, 0, :].astype(float)
    n = len(pts)
    if n < 8:
        return None
    idx = cv2.approxPolyDP(contour, eps, True)
    # approxPolyDP returns points; map them back to contour indices.
    lookup = {tuple(p): i for i, p in enumerate(contour[:, 0, :].tolist())}
    vi = sorted(lookup[tuple(p)] for p in idx[:, 0, :].tolist())
    if len(vi) < 3:
        return None
    k = len(vi)
    trim = CLEAN_PX // 2 + 2
    lines = []
    for j in range(k):
        a, b = vi[j], vi[(j + 1) % k]
        seg = np.arange(a, b + 1) if b > a else np.r_[np.arange(a, n), np.arange(0, b + 1)]
        seg = seg[trim:len(seg) - trim]
        lines.append(_fit_line(pts[seg]) if len(seg) >= 6 else None)
    out = []
    for j in range(k):
        v = pts[vi[j]]
        l_in, l_out = lines[j - 1], lines[j]
        if l_in is not None and l_out is not None:
            x = _intersect(l_in, l_out)
            if x is not None and np.hypot(*(x - v)) <= 2 * CLEAN_PX:
                v = x
        out.append((float(v[0]), float(v[1])))
    poly = Polygon(out)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            return None
    return poly


def extract_blocks(img: np.ndarray, exclude: np.ndarray | None = None) -> list[Polygon]:
    """Block outlines (pixel coords, x right / y down) found on a scanned sheet.

    `exclude` is a bool mask with True = excluded (outside the neatline,
    legend, title...), the same polarity as `reading.exclusion_mask`.
    Handles white (outlined), solid-black, cross-hatched and stippled blocks.
    Blocks touching the image edge or an excluded area are dropped: they are
    partial and would yield false corners.

    Pixel sizes assume ~300 dpi renders (the 1948 photos are of similar
    scale); street and stroke widths are measured per sheet.
    """
    ink = binarize(img)
    outside = np.zeros(ink.shape, bool) if exclude is None else exclude.astype(bool)
    material, anchors = block_material(ink, outside, photo=is_photo(img))
    # Each anchor claims the material geodesically nearest to it, so blocks
    # never merge through lettering or lines; then parts of one block are
    # regrouped by shared boundary length.
    terr = watershed(np.zeros(ink.shape, np.uint8), markers=anchors, mask=material, connectivity=1)
    groups = _group_territories(terr)
    # A printed outline's centreline is the block edge, but the territory
    # includes the whole stroke: outlined (mostly white) blocks are shrunk
    # by half the typical line width.
    half_stroke = stroke_width(ink) / 2.0
    sw = street_width(~ink & ~outside)
    eps = max(SIMPLIFY_PX, POLY_EPS_STREET_FRAC * sw)
    # Closing disk: fills intrusions narrower than ~half a street (street
    # names and lot-line stubs the street flood reached through a faded
    # outline); real streets between blocks are never closed because each
    # block is smoothed on its own.
    close_px = max(CLEAN_PX, int(round(CLOSE_STREET_FRAC * sw)) | 1)
    is_white = np.bincount(groups[(anchors > 0) & (ink == 0)].ravel(), minlength=groups.max() + 1)
    is_dark = np.bincount(groups[(anchors > 0) & ink].ravel(), minlength=groups.max() + 1)
    h, w = ink.shape
    border = np.zeros(ink.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    if outside.any():
        border |= cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    bad = set(np.unique(groups[border]).tolist())
    polys: list[Polygon] = []
    for g, sl in enumerate(ndi.find_objects(groups), start=1):
        if sl is None or g in bad:
            continue
        y0, y1 = max(0, sl[0].start - 1), min(h, sl[0].stop + 1)
        x0, x1 = max(0, sl[1].start - 1), min(w, sl[1].stop + 1)
        pad = close_px + 1
        sub = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), np.uint8)
        sub[pad:-pad, pad:-pad] = groups[y0:y1, x0:x1] == g
        # Smooth the outline: fill dents (stipple, broken outlines), then cut
        # slivers and stroke stubs hanging off the block.
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, _disk(close_px))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, _disk(CLEAN_PX))
        if int(sub.sum()) < MIN_BLOCK_AREA:
            continue
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if area < MIN_BLOCK_AREA or hull <= 0 or area / hull < MIN_BLOCK_SOLIDITY:
            continue
        # Small blocks (triangles at diagonal streets) get a proportionally
        # smaller tolerance so they keep their shape.
        poly = _polygon(c, min(eps, SMALL_EPS_FRAC * math.sqrt(area)))
        if poly is None or poly.area < MIN_BLOCK_AREA:
            continue
        if is_white[g] > is_dark[g]:
            poly = poly.buffer(-half_stroke, join_style="mitre")
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
        polys.append(translate(poly, x0 - pad, y0 - pad))
    return _drop_lettering(polys)


def _drop_lettering(polys: list[Polygon]) -> list[Polygon]:
    """Drop polygons that are bold lettering (R-4, C-2...) boxed off as blocks:
    small next to the median block and clearly non-convex.  Real small blocks
    (triangles at diagonal streets, strips) are convex and are kept."""
    if len(polys) < 3:
        return polys
    med = float(np.median([p.area for p in polys]))
    return [p for p in polys
            if not (p.area < LETTER_AREA_FRAC * med and p.area / p.convex_hull.area < LETTER_SOLIDITY)]


def estimate_street_px(polys: list[Polygon]) -> float:
    """Street width (px) from block polygons: the 75th percentile of each
    block's gap to its nearest neighbour (lower gaps are alleys and hairline
    splits).  Falls back to DEFAULT_STREET_PX."""
    if len(polys) < 2:
        return DEFAULT_STREET_PX
    from shapely import STRtree
    tree = STRtree(polys)
    gaps = []
    for i, p in enumerate(polys):
        best = np.inf
        for j in tree.query(p.buffer(4 * DEFAULT_STREET_PX)):
            if j != i:
                best = min(best, p.distance(polys[j]))
        if np.isfinite(best):
            gaps.append(best)
    gaps = [g for g in gaps if g > 1.0]
    return float(np.percentile(gaps, 75)) if gaps else DEFAULT_STREET_PX


def _clean_ring(pts: np.ndarray, min_edge: float) -> np.ndarray:
    """Remove bumps and notches: repeatedly collapse the shortest edge below
    `min_edge`.  Its two endpoints become the intersection of the adjacent
    edges' lines when that lies close by (a clipped corner is restored);
    otherwise both are dropped (a text bump or in-and-out notch on a
    straight edge, whose neighbours are near-parallel)."""
    pts = [np.asarray(p, float) for p in pts]
    while len(pts) > 3:
        n = len(pts)
        lens = [np.hypot(*(pts[(i + 1) % n] - pts[i])) for i in range(n)]
        i = int(np.argmin(lens))
        if lens[i] >= min_edge:
            break
        a, b = pts[i], pts[(i + 1) % n]
        prev, nxt = pts[i - 1], pts[(i + 2) % n]
        new = None
        if np.hypot(*(a - prev)) > 0 and np.hypot(*(nxt - b)) > 0:
            d1 = (a - prev) / np.hypot(*(a - prev))
            d2 = (nxt - b) / np.hypot(*(nxt - b))
            x = _intersect((prev, d1), (b, d2))
            if x is not None and np.hypot(*(x - (a + b) / 2)) <= min_edge:
                new = x
        keep = [pts[k] for k in range(n) if k not in (i, (i + 1) % n)]
        if new is not None:
            keep.insert(i if i + 1 < n else 0, new)
        pts = keep
    return np.asarray(pts)


def _line_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    n = np.hypot(*d)
    if n == 0:
        return float(np.hypot(*(p - a)))
    return float(abs(d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / n)


def _drop_notches(pts: np.ndarray, max_width: float, max_depth: float, tol: float) -> np.ndarray:
    """Remove in-and-out notches and bumps on straight edges.

    A run of up to NOTCH_MAX_VERTS vertices between i and j is a notch (a
    letter cut into the block edge, a stub of line, a text bump) when the
    edge before i, when extended, passes within `tol` of j (the outline
    resumes on the same straight line), the chord i-j is at most
    `max_width` long, and no vertex of the run strays more than
    `max_depth` from that line.  A genuine jog in a block outline does not
    return to the same line, so it is kept.
    """
    pts = [np.asarray(q, float) for q in pts]
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b = pts[i - 1], pts[i]
            if np.hypot(*(b - a)) == 0:
                continue
            for k in range(2, NOTCH_MAX_VERTS + 2):
                if k >= n - 2:
                    break
                j = (i + k) % n
                c = pts[j]
                if np.hypot(*(c - b)) > max_width or _line_dist(c, a, b) > tol:
                    continue
                # c must lie ahead of b along the edge direction.
                if np.dot(c - b, b - a) <= 0:
                    continue
                run = [pts[(i + m) % n] for m in range(1, k)]
                if all(_line_dist(q, a, b) <= max_depth for q in run):
                    drop = {(i + m) % n for m in range(1, k)}
                    pts = [q for t, q in enumerate(pts) if t not in drop]
                    changed = True
                    break
            if changed:
                break
    return np.asarray(pts)


def block_corners_px(polys: list[Polygon], street_px: float | None = None) -> np.ndarray:
    """N x 2 pixel block corners, by the same > 30 deg turning rule as
    datasf.block_corners, applied to a comparably simplified outline.

    Scale comes from the sheet's street width (measured from the gaps
    between `polys` unless given): px_per_ft = street_px / STREET_FT.  The
    outline is simplified by SIMPLIFY_FT in px, edges shorter than
    MIN_EDGE_STREET_FRAC x street width (ragged print, clipped corners) are
    collapsed, and in-and-out notches on straight edges (lettering, line
    stubs) are removed before turning angles are measured.
    """
    street_px = estimate_street_px(polys) if street_px is None else street_px
    tol = max(1.0, SIMPLIFY_FT * street_px / STREET_FT)
    min_edge = MIN_EDGE_STREET_FRAC * street_px
    out: list[tuple[float, float]] = []
    for p in polys:
        simp = p.simplify(tol)
        parts = [simp] if simp.geom_type == "Polygon" else list(getattr(simp, "geoms", []))
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            ring = _clean_ring(np.asarray(part.exterior.coords)[:-1], min_edge)
            if len(ring) > 4:
                ring = _drop_notches(ring, NOTCH_WIDTH_STREET_FRAC * street_px,
                                     NOTCH_DEPTH_STREET_FRAC * street_px,
                                     NOTCH_TOL_STREET_FRAC * street_px)
            if len(ring) < 3:
                continue
            out += _ring_corners([tuple(q) for q in ring] + [tuple(ring[0])])
    return np.asarray(out, dtype=float).reshape(-1, 2)
