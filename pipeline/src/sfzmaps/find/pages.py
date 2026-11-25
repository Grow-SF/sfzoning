"""The pages the finder reads, and where each series' labels come from.

Reads data/work/<series>/<sheet>/page.png (the scan), data/frame/<series>/<sheet>/outside.png (the frame cut),
inputs/page_kinds.csv and inputs/<series>/duplicate_scans.csv, and the hand-decided inputs/find/label_sources.csv.
Writes nothing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sfzmaps import config
from sfzmaps.frame import data as frame_data

SheetId = tuple[str, str]  # (series, sheet)

OUT = config.DATA_DIR / "find"
SOURCES = config.INPUTS_DIR / "find" / "label_sources.csv"

ERAS = ("hatched-grey", "hatched-colour", "lined-grey", "lined-1bit")
LOTS = ("year", "table", "")       # a series' lots: those of its year, the April 2000 table's, or none
CUT = "outside.png"


def duplicate_scans(series: str) -> dict[str, str]:
    """The series' scans that repeat a section another scan shows better, each with the scan kept in its place: the
    owner's list, inputs/<series>/duplicate_scans.csv (sheet, kept, reason); none when there is no file.  A duplicate
    is never read: one scan per section."""
    f = config.INPUTS_DIR / series / "duplicate_scans.csv"
    if not f.is_file():
        return {}
    rows = pd.read_csv(f, dtype=str)
    return dict(zip(rows["sheet"], rows["kept"]))


def sheets(series: str | None = None) -> list[SheetId]:
    """Every map page of the series (None: all), less its duplicate scans (`sheets_and_duplicates`).
    FileNotFoundError names every such page with no frame cut."""
    return sheets_and_duplicates(series)[0]


def sheets_and_duplicates(series: str | None = None) -> tuple[list[SheetId], list[tuple[SheetId, str]]]:
    """The map pages of the series (None: all) the finder reads, and its duplicate scans, never read, each with the
    scan kept in its place (`duplicate_scans`).  FileNotFoundError names every page read with no frame
    cut."""
    found = frame_data.map_pages(frame_data.PAGE_KINDS, frame_data.WORK)
    skip: dict[str, dict] = {}
    out, dups = [], []
    for s, h in found:
        if series is not None and s != series:
            continue
        if s not in skip:
            skip[s] = duplicate_scans(s)
        if h in skip[s]:
            dups.append(((s, h), skip[s][h]))
        else:
            out.append((s, h))
    missing = [f"{s}/{h}" for s, h in out if not (frame_data.OUT / s / h / CUT).is_file()]
    if missing:
        raise FileNotFoundError(f"map pages with no frame cut ({CUT}): {', '.join(missing)}")
    return out, dups


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
    """Where a series' labels come from: its era, and the lots that label it (`LOTS`: "year" the lots of its year,
    "table" the April 2000 table's, "" none: hand crops only)."""
    series: str
    era: str
    lots: str


def sources(path: Path | None = None) -> dict[str, Source]:
    """The label sources by series.  ValueError for a repeated series, an unknown era or unknown lots."""
    path = SOURCES if path is None else path
    table: dict[str, Source] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = r["series"]
            if s in table:
                raise ValueError(f"{path}: series {s} appears twice")
            if r["era"] not in ERAS:
                raise ValueError(f"{path}: series {s}: unknown era {r['era']!r}")
            if r["lots"] not in LOTS:
                raise ValueError(f"{path}: series {s}: unknown lots {r['lots']!r}")
            table[s] = Source(s, r["era"], r["lots"])
    return table


def source(series: str, path: Path | None = None) -> Source:
    """The series' label source; KeyError naming the file when it has no row."""
    table = sources(path)
    if series not in table:
        raise KeyError(f"series {series!r} has no row in {SOURCES if path is None else path}")
    return table[series]
