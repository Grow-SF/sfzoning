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
(name,blklot,since,source), San Francisco's park lots with the year each became a park: category
`public` in every series whose year is `since` or later, whatever the map zones them; earlier
series keep the map's zoning (owner's decision, 2026-11-02). The optional last column
`public_since` is the year the land became public when that was before it became a park (the
Presidio: an Army post from 1776): `public` from then, counted as a park only from `since`. Candidates come from
`sfzmaps parks` (sfzmaps.parks); a park without a documented year is left out.

inputs/park_maps.csv (name,blklot,series) is generated, not hand-made (`sfzmaps parks --maps`):
the first zoning map showing each park lot of the reference data as public. That lot is a park
from that map on; with a `since` too, from the earlier of the two (assemble.park_starts).
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


def park_maps_path() -> Path:
    return config.INPUTS_DIR / "park_maps.csv"


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
