"""Page registry (which scans are map sheets), rendering, and neatline detection."""
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
    for stem in list(src):
        m = _COMBO.match(stem)
        if m and m.group(1) in src and m.group(2) in src:
            del src[stem]
    return [Page("1948", s, src[s], None) for s in sorted(src, key=_natural)]


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
    """Render to PNG: 300 dpi grayscale for PDFs, RGB for images. Skips if fresh."""
    out = Path(out)
    if out.exists() and out.stat().st_mtime_ns >= page.source.stat().st_mtime_ns:
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


def neatline_frames_mask(img: np.ndarray, min_frac: float = 0.1) -> np.ndarray:
    """Like neatline_mask, but for pages holding several framed maps side by side (a photo of
    a two-page spread, 1948 "2-and-3"): the union of every near-rectangular closed dark
    contour covering more than `min_frac` (and less than 90%) of the page, border included.  A page-sized
    dark blob (cloth around a photographed spread) is skipped, so all contours are examined.  All-True if none."""
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
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_area = small.shape[0] * small.shape[1]
    mask = np.zeros(small.shape, np.uint8)
    found = False
    for c in contours:
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        if area <= min_frac * page_area or rect[1][0] * rect[1][1] <= 0 \
                or area / (rect[1][0] * rect[1][1]) < 0.85 or area > 0.9 * page_area:
            continue
        cv2.fillPoly(mask, [cv2.convexHull(c)], 1)
        found = True
    if not found:
        return full
    if s < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)
