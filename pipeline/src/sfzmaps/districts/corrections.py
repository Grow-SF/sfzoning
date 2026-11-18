"""Readers' corrections from sfzoning.com, imported from a GitHub issue into the hand-made inputs.

The site's lot card lets a reader say what an era's map prints for a lot; the tray sends the list
as a GitHub issue (label `correction`) whose body carries a fenced ```sfzoning-corrections JSON
block (site/lib/corrections.ts).  `sfzmaps corrections N`:

1. reads issue N's block (`gh issue view`) and checks each row: a known district series (not
   today's map, which is DataSF's), an active lot, a code the era knows (categories.category_for);
2. writes scan crops of each lot (data/check/corrections/<N>/) so a person checks the map really
   prints the proposed code before anything is applied (`--apply`);
3. with --apply, sets each lot's code in inputs/<series>/overrides.csv with a note naming the
   issue (shown on the lot card), and adds a review case per series and code change to
   inputs/district_cases.csv, so the reader is measured against what readers found.

The 1996-2000 and 2000-digital maps are two scans of one drawing: a correction to one is applied
to the other too.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sfzmaps import config, inputs
from sfzmaps.districts.series import SERIES

REPO = "Grow-SF/sfzoning"
FENCE = "sfzoning-corrections"
TWINS = {"1996-2000-zoning": "2000-digital-zoning", "2000-digital-zoning": "1996-2000-zoning"}
CASE_PAD_FT = 15.0


@dataclass
class Checked:
    ok: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def parse(body: str) -> list[dict]:
    """The rows of the issue body's fenced sfzoning-corrections block ([] if there is none)."""
    m = re.search(r"```" + FENCE + r"\s*\n(.*?)\n```", body, re.S)
    if not m:
        return []
    rows = json.loads(m.group(1))
    if not isinstance(rows, list):
        raise ValueError("the corrections block is not a list")
    return rows


def fetch(issue: int) -> dict:
    """{number, url, title, body} of the issue, through the GitHub CLI."""
    out = subprocess.run(["gh", "issue", "view", str(issue), "--repo", REPO, "--json", "number,url,title,body"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _norm(code: str) -> str:
    from sfzmaps.districts.labels import normalise

    return normalise(code or "")


def check(rows: list[dict], parcels: pd.DataFrame) -> Checked:
    """Rows kept (series, blklot, shown, proposed, note) and why the others were left out."""
    from sfzmaps.districts.categories import category_for

    known = set(parcels["blklot"].astype(str))
    out = Checked()
    for i, r in enumerate(rows, 1):
        s, b = str(r.get("series") or ""), str(r.get("blklot") or "").strip()
        code = _norm(str(r.get("proposed") or ""))
        info = SERIES.get(s)
        if info is None or s == "today":
            out.problems.append(f"row {i}: series {s!r} isn't a historic map")
        elif b not in known:
            out.problems.append(f"row {i}: lot {b!r} isn't an active parcel")
        elif not code:
            out.problems.append(f"row {i}: no code proposed for lot {b}")
        elif category_for(code, info.year) == "unknown":
            out.problems.append(f"row {i}: {code} isn't a district code the {s} map uses")
        else:
            out.ok.append({"series": s, "blklot": b, "shown": _norm(str(r.get("shown") or "")),
                           "proposed": code, "note": str(r.get("note") or "").strip()})
    return out


def with_twins(rows: list[dict]) -> list[dict]:
    """Rows plus the same correction on the twin scan of the same drawing."""
    out = list(rows)
    have = {(r["series"], r["blklot"]) for r in rows}
    for r in rows:
        t = TWINS.get(r["series"])
        if t and (t, r["blklot"]) not in have:
            out.append({**r, "series": t, "shown": "", "note": (r["note"] + " " if r["note"] else "")
                        + f"(from the {r['series']} scan of the same drawing)"})
    return out


def note_for(issue: int, shown: str) -> str:
    seen = f"; the map reader saw {shown}" if shown else ""
    return f"Corrected from a reader's report (issue #{issue}){seen}"


def write_overrides(rows: list[dict], issue: int) -> dict[str, int]:
    """Set each lot's code in inputs/<series>/overrides.csv (replacing an earlier row for the
    lot), with a note naming the issue.  {series: rows written}."""
    done: dict[str, int] = {}
    for s, grp in pd.DataFrame(rows).groupby("series"):
        path = inputs.overrides_path(s)
        cur = (pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists()
               else pd.DataFrame(columns=["blklot", "block", "code"]))
        if "note" not in cur:
            cur["note"] = ""
        cur = cur[~cur["blklot"].isin(set(grp["blklot"]))]
        new = pd.DataFrame({"blklot": grp["blklot"], "block": "", "code": grp["proposed"],
                            "note": [note_for(issue, sh) for sh in grp["shown"]]})
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([cur, new], ignore_index=True)[["blklot", "block", "code", "note"]].to_csv(path, index=False)
        done[s] = len(new)
    return done


def write_cases(rows: list[dict], issue: int, cent: pd.DataFrame) -> int:
    """One review case per (series, code shown, code proposed): the bbox round those lots.
    Rows the reader left unread (no code shown) give no case: a case needs a wrong code."""
    from sfzmaps.districts.check import CASES_CSV

    df = pd.DataFrame([r for r in rows if r["shown"] and r["shown"] != r["proposed"]])
    if df.empty:
        return 0
    c = cent.set_index("blklot")
    cases = pd.read_csv(CASES_CSV, dtype=str, keep_default_na=False)
    new = []
    for k, ((s, shown, prop), g) in enumerate(df.groupby(["series", "shown", "proposed"]), 1):
        xy = c.loc[c.index.intersection(g["blklot"])]
        if xy.empty:
            continue
        new.append({"id": f"issue{issue}-{k} {s} {shown}->{prop}", "series": s,
                    "x0": round(xy.x.min() - CASE_PAD_FT), "y0": round(xy.y.min() - CASE_PAD_FT),
                    "x1": round(xy.x.max() + CASE_PAD_FT), "y1": round(xy.y.max() + CASE_PAD_FT),
                    "wrong": shown, "right": prop, "kind": "other",
                    "note": f"reader correction, issue #{issue} ({len(g)} lots)"})
    cases = cases[~cases["id"].str.startswith(f"issue{issue}-")]
    pd.concat([cases, pd.DataFrame(new)], ignore_index=True).to_csv(CASES_CSV, index=False)
    return len(new)


def crops(rows: list[dict], dest: Path, parcels) -> list[Path]:
    """A scan crop per corrected lot, the lot outlined in red, on the sheet the reader read it
    from (or any sheet covering it): what a person checks before --apply."""
    import cv2
    import numpy as np

    from sfzmaps.districts import check as chk
    from sfzmaps.districts import sampling
    from sfzmaps.transform import SheetTransform

    dest.mkdir(parents=True, exist_ok=True)
    geom = parcels.set_index("blklot").geometry
    out = []
    for r in rows:
        s, b = r["series"], r["blklot"]
        raw = chk.loader(chk.WORK)(s)
        sheet = None
        if raw is not None:
            hit = raw[raw.blklot == b]
            sheet = hit.source_sheet.iloc[0] if len(hit) and isinstance(hit.source_sheet.iloc[0], str) else None
        if sheet is None:
            got = sampling.assign_sheets(parcels[parcels.blklot == b], s)
            sheet = got.iloc[0] if len(got) and isinstance(got.iloc[0], str) else None
        d = config.WORK_DIR / s / str(sheet) if sheet else None
        if d is None or not (d / "page.png").exists():
            continue
        t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
        g = geom.get(b)
        if g is None:
            continue
        g = g.geoms[0] if hasattr(g, "geoms") else g
        px = t.to_pixel(np.asarray(g.exterior.coords)[:, :2])
        img = cv2.imread(str(d / "page.png"), cv2.IMREAD_GRAYSCALE)
        cx, cy = px.mean(axis=0)
        x0, y0 = int(max(0, cx - 300)), int(max(0, cy - 300))
        crop = cv2.cvtColor(img[y0:y0 + 600, x0:x0 + 600], cv2.COLOR_GRAY2BGR)
        cv2.polylines(crop, [np.round(px - (x0, y0)).astype(np.int32)], True, (0, 0, 255), 2)
        cv2.putText(crop, f"{s} {b}: {r['shown'] or 'none'} -> {r['proposed']}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        p = dest / f"{s}_{b}.png"
        cv2.imwrite(str(p), crop)
        out.append(p)
    return out
