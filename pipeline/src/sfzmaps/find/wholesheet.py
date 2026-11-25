"""Reading a whole sheet with the finder (`sfzmaps find read`): a run's weights over the page the placement steps use,
at the page's own px (no offset, no scale), stored, pictured and summed up in one line.

`check` says why a sheet cannot be read (a duplicate scan, which is never read; no page image; no frame cut).
`read_sheet` reads the page (`pages.load_scan`) with its frame cut (`pages.load_keep`) through `read.read_page`, and
writes the read as a stored read (`read.save`: the two answers as levels round(255 p), 0 off the frame cut) and a
picture at the page's own size (`look.sheet_picture`), both under the run's folder and apart from the crops' reads:
data/find/runs/<run>/sheets/<series>/<sheet>.npz and .png.

A px's answer does not depend on where `read_page`'s squares fall: they are read with their corners on the page's
model.MULTIPLE px grid (`find.read`).  The network's answer does depend on where its grid falls, so every read in this
code is made on the page's grid: a scoring crop (`find.counts`, through `read.window`) gives its px exactly the numbers
this whole-sheet read gives them.

Reads the page, its frame cut and the series' duplicate scans; writes the two files above."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from sfzmaps.find import look, model, pages, read
from sfzmaps.find.pages import SheetId
from sfzmaps.frame import data as frame_data


def check(sid: SheetId) -> str | None:
    """Why the sheet cannot be read, in one line naming it (a duplicate scan, no page image, no frame cut); None when
    it can."""
    series, sheet = sid
    dup = pages.duplicate_scans(series)
    if sheet in dup:
        return f"{series}/{sheet} is a duplicate scan of {dup[sheet]}, which is never read; read {dup[sheet]}"
    page = frame_data.page_path(series, sheet, work=frame_data.WORK)
    if not page.is_file():
        return f"{series}/{sheet}: no page image ({page})"
    cut = frame_data.OUT / series / sheet / pages.CUT
    if not cut.is_file():
        return f"{series}/{sheet}: no frame cut ({cut})"
    return None


def paths(run_folder: Path, sid: SheetId) -> tuple[Path, Path]:
    """The stored read and the picture of a sheet under a run's folder."""
    base = Path(run_folder) / "sheets" / sid[0] / sid[1]
    return base.with_suffix(".npz"), base.with_suffix(".png")


def read_sheet(net: model.FindNet, sid: SheetId, device, run_folder: Path, run: str) -> dict:
    """Read the sheet whole with `net` and write its stored read and picture (`paths`); returns the page's `shape`,
    `street_share` (the share of the frame cut's px called street, the more likely answer), `seconds` and the two
    paths."""
    t0 = time.perf_counter()
    lum = pages.load_scan(sid)
    keep = pages.load_keep(sid, lum.shape)
    chances = read.read_page(net, lum, keep, device)
    f_read, f_pic = paths(run_folder, sid)
    read.save(f_read, chances)
    got = read.stored(chances)
    share = float((got[model.STREET][keep] > 0.5).mean()) if keep.any() else 0.0
    title = (f"RESULT: the finder's read of {sid[0]} / {sid[1]} with run {run} (what the model answers, not training "
             f"labels), at the page's own size")
    f_pic.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f_pic), look.sheet_picture(lum, got, keep, title))
    return {"shape": lum.shape, "street_share": share, "seconds": time.perf_counter() - t0, "read": f_read,
            "picture": f_pic}
