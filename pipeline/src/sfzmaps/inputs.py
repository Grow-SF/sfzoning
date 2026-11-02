"""Hand-made pipeline inputs, kept in git under pipeline/inputs/ (config.INPUTS_DIR).

Everything a person decides lives here, beside the code, so every checkout has it; data/ holds
only what the pipeline can rebuild. Per series, inputs/<series>/:

  reviews/<sheet>.json   a reviewed fit (written by the editor): GCPs, status, smoothing
  seeds/<sheet>.json     a pinned rough fit ({"transform": ...}), used instead of the automatic
                         rough fit; refine still runs from it. For sheets placed by matching
                         another edition (crossref) or by hand-guided seeding.
  pages/<sheet>.json     fields that override the machine page.json (e.g. {"kind": "text"})
  overrides.csv          zoning corrections (blklot,block,code), applied by districts

and inputs/label_scale.json, the calibrated label-reading scale per series, and inputs/parks.csv
(name,blklot,note), the lots of parks that existed from 1921 to today: category `public` in every
series, whatever the map zones them (owner's decision, 2026-11-02).
"""
from __future__ import annotations

import json
from pathlib import Path

from sfzmaps import config


def series_dir(series: str) -> Path:
    return config.INPUTS_DIR / series


def review_path(series: str, sheet_id: str) -> Path:
    return series_dir(series) / "reviews" / f"{sheet_id}.json"


def seed_path(series: str, sheet_id: str) -> Path:
    return series_dir(series) / "seeds" / f"{sheet_id}.json"


def page_patch_path(series: str, sheet_id: str) -> Path:
    return series_dir(series) / "pages" / f"{sheet_id}.json"


def overrides_path(series: str) -> Path:
    return series_dir(series) / "overrides.csv"


def parks_path() -> Path:
    return config.INPUTS_DIR / "parks.csv"


def label_scale_path() -> Path:
    return config.INPUTS_DIR / "label_scale.json"


def sheet_of(page_json: Path) -> tuple[str, str]:
    """(series, sheet_id) of a work/<series>/<sheet>/page.json path."""
    p = Path(page_json)
    return p.parent.parent.name, p.parent.name


def page_dict(page_json: Path) -> dict:
    """A machine page.json with its hand-made patch (if any) merged over it: top-level keys of
    the patch replace the page's. Every page.json read goes through this or reading.load_page."""
    d = json.loads(Path(page_json).read_text())
    patch = page_patch_path(*sheet_of(page_json))
    if patch.exists():
        d.update(json.loads(patch.read_text()))
    return d
