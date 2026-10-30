"""Page registry (which scans are map sheets), rendering, and neatline detection.

Two-page photos (1948 "A-and-B.jpg", an open atlas spread). When both A and B also exist as
single scans the photo is ignored. When only one of them does (1948: "2-and-3", sheet 3
alone as 3.jpg), the photo yields one Page for the missing sheet (id "2"), with `other` set
to the single scan of the page to cut away. render() crops the photo to that page:
neatline_frames finds the framed pages, and the frame whose content matches `other` (ORB
features, homography inliers) is the one dropped; see crop_box. The crop origin is the
offset from the photo's pixels (the owner's answer key "2-and-3.jpg.just-2.points") to
page.png's.
"""
import functools
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sfzmaps.config import RAW_DIR

log = logging.getLogger(__name__)

NON_MAP_1948 = {"index", "setback", "planning-code"}
PLUS_DIR = "1980 Plus"
_COMBO = re.compile(r"^(.+)-and-(.+)$")


@dataclass(frozen=True)
class Page:
    series: str
    sheet_id: str
    source: Path
    page_index: int | None  # 0-based page within a multi-page PDF; None for images / single-page PDFs
    # Two-page photo only: the single scan of the OTHER page, which render() crops away.
    other: Path | None = None


def pdf_page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    if not m:
        raise RuntimeError(f"pdfinfo reported no page count for {pdf}")
    return int(m.group(1))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _pdf_pages(series: str, pdf: Path, sheet_base: str) -> list[Page]:
    n = pdf_page_count(pdf)
    if n == 1:
        return [Page(series, sheet_base, pdf, None)]
    return [Page(series, f"{sheet_base}-p{i + 1:02d}", pdf, i) for i in range(n)]


def _is_pdf(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".pdf"


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _discover_1948(d: Path) -> list[Page]:
    src: dict[str, Path] = {}
    for sub in ("originals", "aligned"):  # aligned overrides originals
        for p in sorted((d / sub).glob("*.jpg")) if (d / sub).is_dir() else []:
            if p.stem not in NON_MAP_1948:
                src[p.stem] = p
    other: dict[str, Path] = {}
    for stem in list(src):
        m = _COMBO.match(stem)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if a in src and b in src:       # both pages scanned alone: the photo adds nothing
            del src[stem]
        elif (a in src) != (b in src):  # one page alone: the photo stands for the other one
            keep, drop = (b, a) if a in src else (a, b)
            src[keep], other[keep] = src.pop(stem), src[drop]
    return [Page("1948", s, src[s], None, other.get(s)) for s in sorted(src, key=_natural)]


def _discover_1960(d: Path) -> list[Page]:
    # Aligned PDFs, so pixel coordinates match the owner's N.pdf.points files.
    out: list[Page] = []
    a = d / "aligned"
    for p in sorted(a.glob("*.pdf"), key=lambda p: _natural(p.stem)) if a.is_dir() else []:
        out += _pdf_pages("1960", p, p.stem)
    return out


def plus_series(name: str) -> str | None:
    """'1990 to 1996 Height Maps.PDF' -> '1990-1996-height'; None for non-maps."""
    stem = Path(name).stem
    low = stem.lower()
    if "introduction" in low:
        return None
    years = re.findall(r"\d{4}", stem)
    if len(years) >= 2:
        span = f"{years[0]}-{years[1]}"
    elif years and "digital" in low:
        span = f"{years[0]}-digital"
    else:
        return None
    if "height" in low:
        kind = "height"
    elif "special use" in low or "su-ss" in low or re.search(r"\bsu\b", low):
        kind = "su"
    elif "zoning" in low:
        kind = "zoning"
    else:
        return None
    return f"{span}-{kind}"


def _discover_plus(d: Path) -> list[Page]:
    out: list[Page] = []
    for p in sorted(x for x in d.iterdir() if _is_pdf(x)):
        series = plus_series(p.name)
        if series is None:
            log.info("skipping non-map document %s", p.name)
            continue
        out += _pdf_pages(series, p, series)
    return out


def discover(raw_dir: Path = RAW_DIR) -> list[Page]:
    out: list[Page] = []
    for d in sorted(x for x in raw_dir.iterdir() if x.is_dir()):
        if d.name == "1948":
            out += _discover_1948(d)
        elif d.name == "1960":
            out += _discover_1960(d)
        elif d.name == PLUS_DIR:
            out += _discover_plus(d)
        elif re.fullmatch(r"\d{4}", d.name):
            for p in sorted(x for x in d.iterdir() if _is_pdf(x)):
                out += _pdf_pages(d.name, p, _slug(p.stem))
    return out


def render(page: Page, out: Path) -> Path:
    """Render to PNG: 300 dpi grayscale for PDFs, RGB for images (a two-page photo cropped
    to crop_box). Skips if fresh."""
    out = Path(out)
    srcs = [page.source] + ([page.other] if page.other is not None else [])
    if out.exists() and all(out.stat().st_mtime_ns >= p.stat().st_mtime_ns for p in srcs):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.png")  # written whole, then renamed: no partial page.png
    try:
        if page.source.suffix.lower() == ".pdf":
            n = (page.page_index or 0) + 1
            subprocess.run(
                ["pdftoppm", "-r", "300", "-gray", "-png", "-singlefile",
                 "-f", str(n), "-l", str(n), str(page.source), str(tmp.with_suffix(""))],
                check=True,
            )
            # Some poppler builds write RGB PNGs even with -gray; enforce one channel.
            g = cv2.imread(str(tmp), cv2.IMREAD_GRAYSCALE)
            if g is None or not cv2.imwrite(str(tmp), g):
                raise RuntimeError(f"cannot post-process {out}")
        else:
            img = cv2.imread(str(page.source), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cannot read {page.source}")
            box = crop_box(page)
            if box is not None:
                x0, y0, x1, y1 = box
                img = img[y0:y1, x0:x1]
            if not cv2.imwrite(str(tmp), img):
                raise RuntimeError(f"cannot write {out}")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def neatline_mask(img: np.ndarray) -> np.ndarray:
    """Bool mask of the area inside the map's outer border (border included).

    The border is the largest near-rectangular closed dark contour covering
    more than 40% of the page; returns all-True if none is found.
    """
    full = np.ones(img.shape[:2], bool)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = gray.shape
    s = min(1.0, 1500 / max(h, w))
    small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    k = max(3, round(max(small.shape) / 300)) | 1
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = small.shape[0] * small.shape[1]
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0.4 * page_area or area <= best_area:
            continue
        rect = cv2.minAreaRect(c)
        if rect[1][0] * rect[1][1] <= 0 or area / (rect[1][0] * rect[1][1]) < 0.9:
            continue
        best, best_area = c, area
    if best is None:
        return full
    mask = np.zeros(small.shape, np.uint8)
    cv2.fillPoly(mask, [cv2.convexHull(best)], 1)
    if s < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _frame_contours(img: np.ndarray, min_frac: float) -> tuple[list, float, tuple[int, int]]:
    """Near-rectangular closed dark contours covering more than `min_frac` (and at most 90%)
    of the page, at the working scale: (contours, scale, working shape)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = gray.shape
    s = min(1.0, 1500 / max(h, w))
    small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    k = max(3, round(max(small.shape) / 300)) | 1
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_area = small.shape[0] * small.shape[1]
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        if area <= min_frac * page_area or rect[1][0] * rect[1][1] <= 0 \
                or area / (rect[1][0] * rect[1][1]) < 0.85 or area > 0.9 * page_area:
            continue
        out.append(c)
    return out, s, small.shape


def neatline_frames_mask(img: np.ndarray, min_frac: float = 0.1) -> np.ndarray:
    """Like neatline_mask, but for pages holding several framed maps side by side (a photo of
    a two-page spread, 1948 "2-and-3"): the union of every near-rectangular closed dark
    contour covering more than `min_frac` (and less than 90%) of the page, border included.  A page-sized
    dark blob (cloth around a photographed spread) is skipped, so all contours are examined.  All-True if none."""
    h, w = img.shape[:2]
    contours, s, shape = _frame_contours(img, min_frac)
    if not contours:
        return np.ones((h, w), bool)
    mask = np.zeros(shape, np.uint8)
    for c in contours:
        cv2.fillPoly(mask, [cv2.convexHull(c)], 1)
    if s < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def neatline_frames(img: np.ndarray, min_frac: float = 0.1) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x0, y0, x1, y1 in full-resolution pixels, x1/y1 exclusive) of the
    separate framed maps on a page (the contours of neatline_frames_mask), left to right. The
    inner and outer lines of one thick border are one frame: a box inside another is dropped."""
    contours, s, _ = _frame_contours(img, min_frac)
    h, w = img.shape[:2]
    boxes = set()
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.add((max(0, int(x / s)), max(0, int(y / s)),
                   min(w, int(np.ceil((x + bw) / s))), min(h, int(np.ceil((y + bh) / s)))))
    tol = round(2 / s)

    def inside(a, b):
        return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol

    outer: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if not any(inside(b, o) for o in outer):
            outer.append(b)
    return sorted(outer)


def _gray_small(img: np.ndarray, edge: int = 1200) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    f = min(1.0, edge / max(g.shape))
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g


def content_match(a: np.ndarray, b: np.ndarray) -> int:
    """How much two images show the same thing: RANSAC homography inliers among ORB feature
    matches (ratio test), both reduced to at most 1200 px. 1948: the right frame of
    2-and-3.jpg against 3.jpg scores ~2750, the left frame ~30."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(_gray_small(a), None)
    kb, db = orb.detectAndCompute(_gray_small(b), None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0
    _, inl = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 5.0)
    return 0 if inl is None else int(inl.sum())


MIN_CONTENT_MATCH = 100   # homography inliers that make a frame "the other page"
CROP_MARGIN_FRAC = 0.02   # margin kept around the frame, so the whole neatline stays on the page


def crop_box(page: Page) -> tuple[int, int, int, int] | None:
    """For a two-page photo (page.other set): (x0, y0, x1, y1) in the photo's pixels of the
    page to keep, the frame (plus a margin, never reaching past half the gutter) whose content
    does NOT match the single scan page.other. None for an ordinary page. Raises when the photo
    does not show exactly two frames or the content does not tell them apart: warping the whole
    spread with one transform is the bug this exists to prevent."""
    if page.other is None:
        return None
    return _crop_box(page.source, page.source.stat().st_mtime_ns,
                     page.other, page.other.stat().st_mtime_ns)


@functools.lru_cache(maxsize=8)
def _crop_box(source: Path, _m1: int, other: Path, _m2: int) -> tuple[int, int, int, int]:
    img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    oth = cv2.imread(str(other), cv2.IMREAD_GRAYSCALE)
    if img is None or oth is None:
        raise RuntimeError(f"cannot read {source if img is None else other}")
    frames = neatline_frames(img)
    if len(frames) != 2:
        raise RuntimeError(f"{source.name}: expected two framed pages, found {len(frames)}")
    score = [content_match(img[y0:y1, x0:x1], oth) for x0, y0, x1, y1 in frames]
    if max(score) < MIN_CONTENT_MATCH or min(score) * 3 > max(score):
        raise RuntimeError(f"{source.name}: cannot tell which frame is {other.name} "
                           f"(feature inliers {score})")
    keep, drop = (frames[1], frames[0]) if score[0] > score[1] else (frames[0], frames[1])
    log.info("%s: keeping frame %s; %s matches the other (inliers %s)", source.name, keep, other.name, score)
    h, w = img.shape
    m = round(CROP_MARGIN_FRAC * max(keep[2] - keep[0], keep[3] - keep[1]))
    x0, y0, x1, y1 = max(0, keep[0] - m), max(0, keep[1] - m), min(w, keep[2] + m), min(h, keep[3] + m)
    if drop[0] >= keep[2]:          # never reach past the middle of the gutter
        x1 = min(x1, (keep[2] + drop[0]) // 2)
    elif drop[2] <= keep[0]:
        x0 = max(x0, (drop[2] + keep[0]) // 2)
    if drop[1] >= keep[3]:
        y1 = min(y1, (keep[3] + drop[1]) // 2)
    elif drop[3] <= keep[1]:
        y0 = max(y0, (drop[3] + keep[1]) // 2)
    return int(x0), int(y0), int(x1), int(y1)
