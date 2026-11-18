"""Packaging: the README and LICENSE that go with each download, and the zips for the vector files.

This is the text side of a download. Changing it (or readme.py or the template) never re-runs GDAL:
see `fingerprint` and `data_fingerprint` in __init__.py.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from sfzmaps.downloads import Item
from sfzmaps.downloads.readme import LICENSE_TEXT, readme

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)
_TEXT_MEMBERS = ("README.txt", "LICENSE.txt")


def write_zip(out: Path, data_members: dict[str, bytes], item: Item, updated: str, commit: str) -> None:
    """Write the zip for one lots/districts item: the data members plus README and LICENSE.

    Deterministic (fixed member dates, sorted entries): the same inputs give the same bytes."""
    members = dict(data_members)
    members["README.txt"] = readme(item.series, item.layer, updated, commit, item.fmt).encode()
    members["LICENSE.txt"] = LICENSE_TEXT.encode()
    part = out.with_name(out.name + ".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, _FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, members[name])
    part.replace(out)


def data_members(zip_path: Path) -> dict[str, bytes]:
    """Every member of an existing download zip except its README and LICENSE, in zip order."""
    with zipfile.ZipFile(zip_path) as z:
        return {n: z.read(n) for n in z.namelist() if n not in _TEXT_MEMBERS}


def write_sidecars(out: Path, item: Item, updated: str, commit: str) -> list[str]:
    """Write `<stem>.README.txt` and `<stem>.LICENSE.txt` beside the citywide image `out`;
    return their paths relative to the downloads directory (as the index lists them)."""
    stem = out.name.removesuffix(out.suffix)
    out.with_name(f"{stem}.README.txt").write_text(readme(item.series, "citywide", updated, commit, item.fmt))
    out.with_name(f"{stem}.LICENSE.txt").write_text(LICENSE_TEXT)
    return [f"{item.series}/{stem}.README.txt", f"{item.series}/{stem}.LICENSE.txt"]
