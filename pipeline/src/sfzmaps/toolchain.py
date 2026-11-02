"""The system tools the pipeline runs, where they come from, and whether they render the same.

The tools are declared and pinned in the repository's flake.nix (Nix): GDAL, poppler, pmtiles,
uv, git. `sfzmaps doctor` reports each tool's path and version, and whether it comes from the
Nix store; with --renders it re-renders sample pages into a temporary folder and compares them
with the page_sha256 their page.json recorded. A different render means the vision-answer
cache misses for that page (paid re-reads), so check renders before changing the toolchain.

`sfzmaps build` refuses to run unless every tool comes from the Nix store (the flake's dev
shell), so a build records one known toolchain. --system-tools overrides that, loudly.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sfzmaps import config

# Every external program the pipeline (and its build) calls.
TOOLS = ("gdalwarp", "gdal_translate", "gdaladdo", "gdalbuildvrt", "gdal_rasterize", "gdalinfo",
         "ogr2ogr", "pdftoppm", "pdfinfo", "pmtiles", "uv", "git")
NIX_STORE = "/nix/store/"
# How to ask each tool its version (stdout or stderr), and the version pattern.
_VERSION_CMD = {"gdalinfo": ["--version"], "pdftoppm": ["-v"], "pmtiles": ["version"],
                "uv": ["--version"], "git": ["--version"]}


@dataclass
class Tool:
    name: str
    path: str | None
    version: str | None

    @property
    def from_nix(self) -> bool:
        return bool(self.path) and os.path.realpath(self.path).startswith(NIX_STORE)


def _version(path: str, args: list[str]) -> str | None:
    try:
        p = subprocess.run([path, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\d+\.\d+(?:\.\d+)?", p.stdout + p.stderr)
    return m.group(0) if m else None


def tools() -> list[Tool]:
    out = []
    for name in TOOLS:
        path = shutil.which(name)
        args = _VERSION_CMD.get(name)
        out.append(Tool(name, path, _version(path, args) if path and args else None))
    return out


def python_libraries() -> dict[str, str]:
    """The GDAL / GEOS versions the Python wheels bundle (separate from the GDAL tools)."""
    found = {}
    for mod, attr in (("rasterio", "__gdal_version__"), ("pyogrio", "__gdal_version__"),
                      ("shapely", "geos_version_string")):
        try:
            found[f"{mod} {attr.strip('_')}"] = str(getattr(__import__(mod), attr))
        except Exception:  # noqa: BLE001 - a library missing is reported as such
            found[f"{mod} {attr.strip('_')}"] = "not installed"
    return found


def not_from_nix(ts: list[Tool] | None = None) -> list[Tool]:
    return [t for t in (ts if ts is not None else tools()) if not t.from_nix]


def toolchain_id() -> str:
    """What the build ran with: the flake's pin inside its dev shell, else "system"."""
    return os.environ.get("SFZMAPS_TOOLCHAIN") or "system"


# --- render check ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_renders(per_series: int = 3, seed: int = 0) -> tuple[int, list[str]]:
    """Re-render up to per_series PDF pages of each series (those whose page.json records a
    page_sha256) into a temporary folder; (pages checked, the ones that render differently).
    Writes nothing under data/."""
    from sfzmaps import pages

    by_series: dict[str, list] = {}
    for page in pages.discover(config.RAW_DIR):
        if page.source.suffix.lower() != ".pdf":
            continue
        pj = config.WORK_DIR / page.series / page.sheet_id / "page.json"
        try:
            sha = json.loads(pj.read_text()).get("page_sha256")
        except (OSError, ValueError):
            continue
        if sha:
            by_series.setdefault(page.series, []).append((page, sha))
    rng = random.Random(seed)
    checked, differ = 0, []
    with tempfile.TemporaryDirectory(prefix="sfzmaps-doctor-") as tmp:
        for series, items in sorted(by_series.items()):
            for page, sha in rng.sample(items, min(per_series, len(items))):
                out = Path(tmp) / series / f"{page.sheet_id}.png"
                pages.render(page, out)
                checked += 1
                if _sha256(out) != sha:
                    differ.append(f"{series}/{page.sheet_id}")
    return checked, differ
