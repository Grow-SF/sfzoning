"""Per-sheet download entries: one per placed sheet, with sizes and hashes (not part of any item's fingerprint)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from sfzmaps import config, stitch


def sheet_entries(series: str, known: list[dict] | None = None) -> list[dict]:
    """One entry per placed sheet (every dir with warped_paper.tif), in natural order of sheet id."""
    entries = []
    old = {e["sheet"]: e for e in known or []}
    for tif in (config.WORK_DIR / series).glob("*/warped_paper.tif"):
        st = tif.stat()
        sheet = tif.parent.name
        k = old.get(sheet)
        if k and k.get("bytes") == st.st_size and k.get("mtime_ns") == st.st_mtime_ns:
            md5, sha = k["md5"], k["sha256"]  # unchanged since the last index: don't read GBs again
        else:
            md5, sha = _md5(tif), stitch._sha256(tif)
        entries.append({"sheet": sheet, "name": f"{series}-sheet-{sheet}.tif", "md5": md5,
                        "bytes": st.st_size, "sha256": sha, "mtime_ns": st.st_mtime_ns})
    return sorted(entries, key=lambda e: _natural(e["sheet"]))


def _natural(name: str) -> list:
    """"2" before "10", "5a" after "5"; digit runs compare as numbers (parts alternate str, int)."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
