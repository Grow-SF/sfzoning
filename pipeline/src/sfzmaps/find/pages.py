"""The pages the finder reads, and where each series' labels come from.

Reads data/work/<series>/<sheet>/page.png (the scan), data/frame/<series>/<sheet>/outside.png (the frame cut),
inputs/page_kinds.csv and inputs/<series>/duplicate_scans.csv, and the hand-decided inputs/find/label_sources.csv.
Writes nothing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sfzmaps import config
from sfzmaps.frame import data as frame_data
from sfzmaps.solver import hatching

SheetId = tuple[str, str]  # (series, sheet)

OUT = config.DATA_DIR / "find"
SOURCES = config.INPUTS_DIR / "find" / "label_sources.csv"

ERAS = ("hatched-grey", "hatched-colour", "lined-grey", "lined-1bit")
BORDERS = ("zoning", "height_lim", "sud", "none", "")
CUT = "outside.png"


def sheets(series: str | None = None) -> list[SheetId]:
    """Every map page of the series (None: all), less its duplicate scans.  FileNotFoundError names every such page
    with no frame cut."""
    found = frame_data.map_pages(frame_data.PAGE_KINDS, frame_data.WORK)
    skip: dict[str, dict] = {}
    out = []
    for s, h in found:
        if series is not None and s != series:
            continue
        if s not in skip:
            skip[s] = hatching.duplicate_scans(s)
        if h not in skip[s]:
            out.append((s, h))
    missing = [f"{s}/{h}" for s, h in out if not (frame_data.OUT / s / h / CUT).is_file()]
    if missing:
        raise FileNotFoundError(f"map pages with no frame cut ({CUT}): {', '.join(missing)}")
    return out


def load_scan(sid: SheetId) -> np.ndarray:
    """The page as stored: float32 (h, w), 1 paper to 0 ink."""
    return frame_data.load_luminance(frame_data.page_path(*sid, work=frame_data.WORK))


def load_keep(sid: SheetId, shape: tuple[int, int]) -> np.ndarray:
    """Bool, True where the frame cut (outside.png) is 0, the map area.  ValueError if its shape is not `shape`."""
    path = frame_data.OUT / sid[0] / sid[1] / CUT
    with Image.open(path) as im:
        cut = np.asarray(im.convert("L"))
    if cut.shape != tuple(shape):
        raise ValueError(f"{path}: frame cut is {cut.shape}, the page is {tuple(shape)}")
    return cut == 0


@dataclass(frozen=True)
class Source:
    """Where a series' labels come from: its era, whether lots label it, and what labels its zoning borders."""
    series: str
    era: str
    lots: bool
    borders: str


def sources(path: Path | None = None) -> dict[str, Source]:
    """The label sources by series.  ValueError for a repeated series, an unknown era or an unknown zoning_borders."""
    path = SOURCES if path is None else path
    table: dict[str, Source] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = r["series"]
            if s in table:
                raise ValueError(f"{path}: series {s} appears twice")
            if r["era"] not in ERAS:
                raise ValueError(f"{path}: series {s}: unknown era {r['era']!r}")
            if r["zoning_borders"] not in BORDERS:
                raise ValueError(f"{path}: series {s}: unknown zoning_borders {r['zoning_borders']!r}")
            if r["lots"] not in ("year", ""):
                raise ValueError(f"{path}: series {s}: unknown lots {r['lots']!r}")
            table[s] = Source(s, r["era"], r["lots"] == "year", r["zoning_borders"])
    return table


def source(series: str, path: Path | None = None) -> Source:
    """The series' label source; KeyError naming the file when it has no row."""
    table = sources(path)
    if series not in table:
        raise KeyError(f"series {series!r} has no row in {SOURCES if path is None else path}")
    return table[series]
