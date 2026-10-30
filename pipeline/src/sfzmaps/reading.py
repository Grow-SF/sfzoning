"""Vision-model page classification and printed label reading (block numbers, street names)."""
import argparse
import json
import sys
from pathlib import Path
from typing import Literal, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel

from sfzmaps import config, llm
from sfzmaps.pages import neatline_mask

INK_MIN = 0.005
INK_LEVEL = 170  # gray < this counts as ink (area-averaged lettering is lighter than 1-bit black)
DEDUPE_PX = 40
MATCH_PX = 40
SCALES = (1.0, 0.75, 0.6, 0.5, 0.4, 0.33)

CLASSIFY_PROMPT = """This is a low-resolution thumbnail of one page from a scanned historic San Francisco zoning map set. It is {w} pixels wide and {h} pixels high. Return all coordinates as integer pixels in this image (x from 0 to {w}, y from 0 to {h}, origin at the top-left).

Return:
- kind: "map" if the page is a map sheet showing city blocks and streets (even if it also has a legend or title block); "cover" for a title/cover page; "legend" for a page that is mainly a legend or key; "text" for pages of prose or tables; "index" for an index or key-map page showing how sheets fit together; "other" otherwise.
- sheet_number: the sheet number printed on the page (for example "7"), or null if none is visible.
- neighbor_refs: for a map, each printed reference to an adjoining sheet (usually small numbers or "See Sheet N" near an edge), with text as printed, sheet as the sheet number, and side as the edge it is on ("n", "s", "e", "w", or "unknown").
- mask_boxes: bounding boxes (x0, y0, x1, y1 in pixels) of every legend, title block, notes panel, or margin area that does not show the map itself and should be excluded from reading street names and block numbers. Do not box the map area. Return an empty list if there are none.

Do not guess; use null or empty lists when unsure."""

LABEL_PROMPT = """This image is a crop of a scanned historic San Francisco zoning map. It is {w} pixels wide and {h} pixels high. Return x and y as integer pixel coordinates within this image (x from 0 to {w}, y from 0 to {h}, origin at the top-left).

Find only these printed labels:
- block_number: assessor block numbers. They are usually 1 to 4 digits, sometimes with a letter suffix (for example "26" or "3721A"), and printed inside a block.
- street_name: street names printed along streets, for example "MARKET ST" or "Van Ness". They may be rotated or curved. Return the text as printed.
- place_name: only large named places such as parks.

For each label give its text and the pixel center (x, y) of the printed text, plus a confidence from 0 to 1.

Ignore zoning district codes (R-1, C-2, M-1, "P" and the like), legend text, sheet titles, notes, and lot dimensions or lot numbers. If a label is cut off by the image edge, skip it. Return an empty list rather than guessing."""


def classify_prompt(img: np.ndarray) -> str:
    return CLASSIFY_PROMPT.format(h=img.shape[0], w=img.shape[1])


def label_prompt(img: np.ndarray) -> str:
    return LABEL_PROMPT.format(h=img.shape[0], w=img.shape[1])


class Box(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class NeighborRef(BaseModel):
    text: str
    sheet: str
    side: Literal["n", "s", "e", "w", "unknown"]


class PageInfo(BaseModel):
    kind: Literal["map", "cover", "legend", "text", "index", "other"]
    sheet_number: str | None = None
    neighbor_refs: list[NeighborRef] = []
    mask_boxes: list[Box] = []


class Label(BaseModel):
    kind: Literal["block_number", "street_name", "place_name"]
    text: str
    x: int
    y: int
    confidence: float


class _Labels(BaseModel):
    labels: list[Label]


M = TypeVar("M", bound=BaseModel)


def save(path: Path, obj: BaseModel | list[Label], page_sha256: str | None = None) -> None:
    """Write a PageInfo (page.json) or labels (labels.json). With page_sha256 (the sha256 of
    the page.png they were read from) it is stored too: a "page_sha256" key in page.json,
    and labels.json becomes {"page_sha256": ..., "labels": [...]} (a bare list without)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [x.model_dump() for x in obj] if isinstance(obj, list) else obj.model_dump()
    if page_sha256 is not None:
        data = {"page_sha256": page_sha256, "labels": data} if isinstance(data, list) \
            else {**data, "page_sha256": page_sha256}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def load_page(path: Path) -> PageInfo:
    return PageInfo.model_validate_json(Path(path).read_text())


def load_labels(path: Path) -> list[Label]:
    d = json.loads(Path(path).read_text())
    return [Label.model_validate(x) for x in (d["labels"] if isinstance(d, dict) else d)]


def stored_page_sha256(path: Path) -> str | None:
    """The page_sha256 stored in a page.json / labels.json, or None (missing, unreadable,
    or an older file without it)."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    v = d.get("page_sha256") if isinstance(d, dict) else None
    return v if isinstance(v, str) else None


def _read_gray(png_path: Path) -> np.ndarray:
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {png_path}")
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def thumbnail(img: np.ndarray) -> tuple[np.ndarray, float]:
    g = _gray(img)
    s = min(1.0, config.VISION_THUMB / max(g.shape))
    if s < 1:
        g = cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    return g, s


def classify_page(png_path: Path, *, client: llm.LLMClient | None = None) -> PageInfo:
    img = _read_gray(png_path)
    h, w = img.shape[:2]
    thumb, _ = thumbnail(img)
    info = llm.ask_json(thumb, classify_prompt(thumb), PageInfo, client=client)
    fx, fy = w / thumb.shape[1], h / thumb.shape[0]
    boxes = []
    for b in info.mask_boxes:
        boxes.append(Box(x0=max(0, round(b.x0 * fx)), y0=max(0, round(b.y0 * fy)),
                         x1=min(w, round(b.x1 * fx)), y1=min(h, round(b.y1 * fy))))
    return info.model_copy(update={"mask_boxes": boxes})


def exclusion_mask(img: np.ndarray, info: PageInfo | None = None) -> np.ndarray:
    """True where tiles should NOT be sent: outside the neatline or inside a mask box."""
    skip = ~neatline_mask(img)
    h, w = skip.shape
    for b in (info.mask_boxes if info else []):
        skip[max(0, b.y0):min(h, b.y1), max(0, b.x0):min(w, b.x1)] = True
    return skip


def select_tiles(img: np.ndarray, *, scale: float, tile: int = config.VISION_TILE, overlap: int = config.VISION_OVERLAP,
                 skip: np.ndarray | None = None
                 ) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Downscale by `scale` and cut tiles; drop blank and fully excluded ones.

    Returns (scaled_gray_page, [(x0, y0, tile_image)]) with offsets in scaled pixels.
    This is the single source of truth for which images get sent (read_labels, estimate).
    """
    g = _gray(img)
    if scale != 1.0:
        g = cv2.resize(g, (max(1, round(g.shape[1] * scale)), max(1, round(g.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = g.shape
    sk = None
    if skip is not None:
        sk = skip if skip.shape == g.shape else cv2.resize(
            skip.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    stride = tile - overlap
    xs = [x for x in range(0, sw, stride) if x == 0 or x + overlap < sw]
    ys = [y for y in range(0, sh, stride) if y == 0 or y + overlap < sh]
    out = []
    for y0 in ys:
        for x0 in xs:
            t = g[y0:y0 + tile, x0:x0 + tile]
            if sk is not None and sk[y0:y0 + tile, x0:x0 + tile].all():
                continue
            if (t < INK_LEVEL).mean() < INK_MIN:
                continue
            out.append((x0, y0, t))
    return g, out


def read_labels(png_path: Path, *, scale: float, tile: int = config.VISION_TILE, overlap: int = config.VISION_OVERLAP,
                skip: np.ndarray | None = None,
                client: llm.LLMClient | None = None) -> list[Label]:
    img = _read_gray(png_path)
    _, tiles = select_tiles(img, scale=scale, tile=tile, overlap=overlap, skip=skip)
    if client is None and tiles and not llm.CACHE_ONLY and not all(llm.is_cached(t[2], label_prompt(t[2]), _Labels) for t in tiles):
        client = llm.OpenAIClient()  # build once; shared by worker threads

    def one(tile):
        return llm.ask_json(tile[2], label_prompt(tile[2]), _Labels, client=client)

    results = llm.pmap(one, tiles, config.VISION_CONCURRENCY)  # in tile order -> deterministic merge
    out: list[Label] = []
    for (x0, y0, t), res in zip(tiles, results):
        th, tw = t.shape[:2]
        for lb in res.labels:
            if not (0 <= lb.x <= tw and 0 <= lb.y <= th):
                continue  # outside the tile: provider coordinate error
            out.append(lb.model_copy(update={"x": round((lb.x + x0) / scale),
                                             "y": round((lb.y + y0) / scale)}))
    return _dedupe(out)


def _dedupe(labels: list[Label]) -> list[Label]:
    kept: list[Label] = []
    for lb in sorted(labels, key=lambda l: -l.confidence):
        if any(k.kind == lb.kind and k.text == lb.text
               and abs(k.x - lb.x) <= DEDUPE_PX and abs(k.y - lb.y) <= DEDUPE_PX for k in kept):
            continue
        kept.append(lb)
    return sorted(kept, key=lambda l: (l.y, l.x))


def _recall(ref: list[Label], got: list[Label]) -> float:
    ref = [l for l in ref if l.kind in ("block_number", "street_name")]
    if not ref:
        return 1.0
    hit = sum(any(g.text.casefold() == r.text.casefold() and abs(g.x - r.x) <= MATCH_PX
                  and abs(g.y - r.y) <= MATCH_PX for g in got) for r in ref)
    return hit / len(ref)


def calibrate_scale(png_path: Path, scales=SCALES, *, client: llm.LLMClient | None = None,
                    report: dict[float, float] | None = None) -> float:
    """Smallest scale whose recall is >= 95% of the scale-1.0 result (stops at the first failing scale)."""
    img = _read_gray(png_path)
    skip = ~neatline_mask(img)
    ref = read_labels(png_path, scale=1.0, skip=skip, client=client)
    best = 1.0
    for s in sorted(scales, reverse=True):
        r = 1.0 if s == 1.0 else _recall(ref, read_labels(png_path, scale=s, skip=skip, client=client))
        if report is not None:
            report[s] = r
        if r < 0.95:
            break
        best = s
    return best


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("png", type=Path)
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    try:
        out: dict = {}
        if a.classify:
            out["page"] = classify_page(a.png).model_dump()
        if a.labels:
            img = _read_gray(a.png)
            info = PageInfo.model_validate(out["page"]) if "page" in out else None
            out["labels"] = [l.model_dump() for l in
                             read_labels(a.png, scale=a.scale, skip=exclusion_mask(img, info))]
    except (llm.LLMAuthError, llm.LLMRefusal, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
