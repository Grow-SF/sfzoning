"""GIS downloads: what is published (the catalog) and when it is out of date (fingerprints).

No network and no paid reads. Not a DVC stage.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sfzmaps import config, stitch
from sfzmaps.districts.series import SERIES

DOWNLOADS_DIR = config.DATA_DIR / "downloads"
# Bump by hand only to force a refresh the exporter code below doesn't show (say, a GDAL option
# that changes output while the code stays the same): the fingerprint already covers that code.
EXPORTER_VERSION = "2"
_HERE = Path(__file__).parent
# The code that writes the downloads: a change to any of it (README text included) refreshes them.
EXPORTER_CODE = [_HERE / "readme.py", _HERE / "readme-template.txt", _HERE / "vectors.py",
                 _HERE / "rasters.py"]

VECTOR_FORMATS = ("geojson", "gpkg", "shp")
_EXT = {"geojson": "geojson.zip", "gpkg": "gpkg.zip", "shp": "shp.zip", "cog": "tif"}


@dataclass(frozen=True)
class Item:
    key: str  # "<series>/<layer>.<fmt>"
    series: str
    layer: Literal["lots", "districts", "citywide"]
    fmt: Literal["geojson", "gpkg", "shp", "cog"]
    filename: str
    sources: list[Path]


def _item(series: str, layer: str, fmt: str, sources: list[Path]) -> Item:
    return Item(f"{series}/{layer}.{fmt}", series, layer, fmt,  # type: ignore[arg-type]
                f"{series}-{layer}.{_EXT[fmt]}", sources)


def citywide_sources(series: str) -> list[Path]:
    """Every placed sheet's warped_paper.tif and footprint, plus the stitching code."""
    tifs = sorted((config.WORK_DIR / series).glob("*/warped_paper.tif"))
    out: list[Path] = []
    for t in tifs:
        fp = t.with_name("warped_paper.footprint.geojson")
        out += [t, fp] if fp.exists() else [t]
    return out + [Path(stitch.__file__)] if tifs else []


def catalog() -> list[Item]:
    items: list[Item] = []
    vector_series = [s for s, i in SERIES.items() if i.role == "use"] + ["today"]
    for s in vector_series:
        d = config.OUT_DIR / s / "districts"
        if not (d / "parcels.fgb").exists():
            continue
        for layer, name in (("lots", "parcels.fgb"), ("districts", "districts.fgb")):
            for fmt in VECTOR_FORMATS:
                items.append(_item(s, layer, fmt, [d / name]))
    for s in SERIES:
        src = citywide_sources(s)
        if src:
            items.append(_item(s, "citywide", "cog", src))
    return items


def _rel(p: Path) -> str:
    for base in (config.DATA_DIR, Path(stitch.__file__).parent):
        try:
            return str(p.relative_to(base))
        except ValueError:
            pass
    return p.name


def fingerprint(item: Item, exporter_version: str) -> str:
    h = hashlib.sha256()
    for rel, digest in sorted((_rel(p), stitch._sha256(p)) for p in item.sources):
        h.update(f"{rel}\0{digest}\n".encode())
    for p in EXPORTER_CODE:
        h.update(f"code:{p.name}\0{stitch._sha256(p)}\n".encode())
    h.update(f"exporter:{exporter_version}".encode())
    return h.hexdigest()


def run_tool(*cmd: str) -> None:
    """Run a GDAL command line; on failure raise with GDAL's own stderr in the message."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")


def _commit() -> str:
    repo = Path(__file__).resolve().parent
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def _today() -> str:
    return datetime.date.today().isoformat()


def _sha256_file(p: Path) -> str:
    return stitch._sha256(p)


def run(series: list[str] | None = None, log=lambda line: None) -> dict:
    """Write what is out of date under DOWNLOADS_DIR and return (and write) index.json.

    An item whose fingerprint equals the previous index entry, and whose file exists, is left alone
    (not even its date changes). `series` limits the work; entries of other series are kept.
    """
    from sfzmaps.downloads import rasters, readme as readme_mod, vectors

    root = DOWNLOADS_DIR
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    prev: dict = json.loads(index_path.read_text()) if index_path.exists() else {}
    prev_files = prev.get("files", {})
    commit = _commit()
    items = catalog()
    everything = {i.series for i in items}
    known = everything | set(prev.get("sheets", {})) | {s for s in SERIES if (config.WORK_DIR / s).is_dir()}
    unknown = sorted(set(series or []) - known)
    if unknown:
        raise ValueError(f"unknown series: {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    wanted = set(series) if series else everything
    files: dict = {k: v for k, v in prev_files.items() if v.get("series") not in wanted}
    sheets: dict = {k: v for k, v in prev.get("sheets", {}).items() if k not in wanted}
    readmes: dict = {k: v for k, v in prev.get("readme", {}).items() if k not in wanted}
    updated = _today()
    for item in items:
        if item.series not in wanted:
            continue
        fp = fingerprint(item, EXPORTER_VERSION)
        out = root / item.series / item.filename
        old = prev_files.get(item.key)
        extra_ok = all((root / e).exists() for e in (old or {}).get("extra", []))
        if old and old.get("fingerprint") == fp and out.exists() and extra_ok and old.get("path") == f"{item.series}/{item.filename}":
            files[item.key] = old
            log(f"same   {item.key}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if item.layer == "citywide":
            rasters.export_citywide(item, out, updated, commit)
            stem = out.name.removesuffix(out.suffix)
            extra = [f"{item.series}/{stem}.README.txt", f"{item.series}/{stem}.LICENSE.txt"]
        else:
            vectors.export_vector(item, out, updated, commit)
            extra = []
        files[item.key] = {
            "series": item.series, "layer": item.layer, "fmt": item.fmt,
            "path": f"{item.series}/{item.filename}", "fingerprint": fp,
            "bytes": out.stat().st_size, "sha256": _sha256_file(out), "extra": extra,
        }
        log(f"wrote  {item.key}  ({out.stat().st_size:,} bytes)")
    for s in sorted(wanted & everything):
        entries = rasters.sheet_entries(s, prev.get("sheets", {}).get(s))
        if not entries:
            continue
        sheets[s] = entries
        old_readme = prev.get("readme", {}).get(s)
        new_readme = readme_mod.readme(s, "sheets", updated, commit)
        if (old_readme is not None and prev.get("sheets", {}).get(s) == entries
                and _undated(old_readme) == _undated(new_readme)):
            readmes[s] = old_readme  # nothing changed, text included: keep its date
        else:
            readmes[s] = new_readme
    index = {
        "commit": prev.get("commit", commit) if not _anything_written(prev, files, sheets, readmes) else commit,
        "license": readme_mod.LICENSE_TEXT,
        "files": dict(sorted(files.items())),
        "sheets": dict(sorted(sheets.items())),
        "readme": dict(sorted(readmes.items())),
    }
    text = json.dumps(index, indent=2) + "\n"
    if not index_path.exists() or index_path.read_text() != text:
        part = index_path.with_name("index.json.part")
        part.write_text(text)
        os.replace(part, index_path)  # a reader never sees a half-written index
    return index


def _undated(readme_text: str) -> str:
    """A README without its LAST UPDATED line (date and build), to compare the rest."""
    return re.sub(r"(?m)^(LAST UPDATED\n).*$", r"\1", readme_text)


def _anything_written(prev: dict, files: dict, sheets: dict, readmes: dict) -> bool:
    return (prev.get("files"), prev.get("sheets"), prev.get("readme")) != (files, sheets, readmes) or "commit" not in prev
