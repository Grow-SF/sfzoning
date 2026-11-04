"""Reading a series' districts: each mapped parcel's code (reader A, the legend patterns; reader B,
the printed labels and their regions), height code and confidence, on its highest-priority sheet.

Kept apart from publishing (assemble): a read is fingerprinted by the code it runs (this module
and what it imports, assemble.read_inputs), so a change to the joins, rules or outputs
republishes without reading every series again.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from sfzmaps import config
from sfzmaps.districts.series import SERIES

log = logging.getLogger(__name__)

MEDIUM = 0.6            # confidence cap for parcels read from needs_review sheets
COLUMNS = ["blklot", "code", "height_code", "confidence", "source_sheet"]


def load_parcels(year: float | None = None, *, every: bool = False) -> gpd.GeoDataFrame:
    """Today's lots (DataSF's active parcels); with `year`, the lots of that year instead
    (districts/lots.py: retired lots back, their successors out); with `every`, all lots ever
    recorded (to look up any lot's place)."""
    p = gpd.read_parquet(config.REF_DIR / "parcels.parquet")
    if every:
        return p.reset_index(drop=True)
    if year is None:
        return p[p["active"]].reset_index(drop=True)
    from sfzmaps.districts import lots

    t = lots.lives().set_index("blklot").reindex(p["blklot"].astype(str))
    return p[lots.valid_in(t.reset_index(), year + 0.5)].reset_index(drop=True)


def series_year(series: str) -> float | None:
    """The year whose lots a series is read and published on (None: today's lots)."""
    info = SERIES.get(series)
    return None if info is None or series == "today" else float(info.year)


def sheet_statuses(series: str) -> dict[str, str]:
    from sfzmaps import manifest, stitch

    return {sid: manifest.status_of(series, sid, sc) for sid, sc in stitch.load_scores(series).items()}


def cap_confidence(df: pd.DataFrame, statuses: dict[str, str]) -> pd.DataFrame:
    """Parcels from needs_review sheets keep their code at confidence <= MEDIUM."""
    df = df.copy()
    bad = df["source_sheet"].map(lambda s: statuses.get(s) == "needs_review")
    df.loc[bad, "confidence"] = df.loc[bad, "confidence"].clip(upper=MEDIUM)
    return df


def merge_use_height(use: pd.DataFrame, height: pd.DataFrame, role: str) -> pd.DataFrame:
    """One row per parcel read on a use sheet (code) and/or a height sheet (height_code);
    confidence and source_sheet come from the series' role."""
    u = use[["blklot", "code", "confidence", "source_sheet"]].drop_duplicates("blklot")
    h = height[["blklot", "height_code", "confidence", "source_sheet"]].drop_duplicates("blklot")
    m = u.merge(h, on="blklot", how="outer", suffixes=("_u", "_h"))
    mine, other = ("_u", "_h") if role == "use" else ("_h", "_u")
    out = pd.DataFrame({
        "blklot": m["blklot"],
        "code": m["code"].astype(object).where(m["code"].notna(), None),
        "height_code": m["height_code"].astype(object).where(m["height_code"].notna(), None),
        "confidence": m["confidence" + mine].fillna(0.0).astype(float),
        "source_sheet": m["source_sheet" + mine].where(m["source_sheet" + mine].notna(),
                                                       m["source_sheet" + other]),
    })
    return out.sort_values("blklot").reset_index(drop=True)[COLUMNS]


def pick_readings(rd: pd.DataFrame, col: str) -> pd.DataFrame:
    """One reading per parcel from its readings on every covering sheet (columns blklot, col,
    confidence, source_sheet, rank = sheet priority, label_ft): one with a code, nearest to the
    printed label backing it, then the highest-priority sheet.  Each sheet also draws the next
    sheets' streets, unzoned, for context; a parcel there takes whatever code its region
    picked up from a label far away (1972: a band of the Sunset as P; 1960: the Tenderloin as
    M-1), while the sheet that zones it prints its code nearby."""
    key = rd.assign(_none=rd[col].isna())
    key = key.sort_values(["_none", "label_ft", "rank"], kind="stable")
    return key.drop_duplicates("blklot").drop(columns="_none").reset_index(drop=True)


def readings(series: str, parcels, kind_order: list[str], kind: str) -> pd.DataFrame:
    """Every parcel's reading of `kind` on every sheet of kind_order that covers it."""
    from sfzmaps.districts import labels, sampling

    col = "code" if kind == "use" else "height_code"
    out = []
    for rank, sid in enumerate(kind_order):
        sub = parcels[(sampling.assign_sheets(parcels, series, sheets=[sid]) == sid).to_numpy()]
        if not len(sub):
            continue
        log.info("%s sheet %s (%s): %d parcels", series, sid, kind, len(sub))
        r = labels.assign_parcels(sub.reset_index(drop=True), config.WORK_DIR / series / sid,
                                  all_parcels=parcels)
        out.append(pd.DataFrame({"blklot": r["blklot"], col: r[col],
                                 "confidence": r[f"{kind}_confidence"], "source_sheet": sid,
                                 "rank": rank, "label_ft": r[f"{kind}_label_ft"]}))
    cols = ["blklot", col, "confidence", "source_sheet", "rank", "label_ft"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def _read_b(series: str, parcels, order: list[str]) -> pd.DataFrame:
    """Reader B per sheet kind: use codes from use sheets, height codes from height sheets
    (labels.sheet_role); each parcel read on every sheet of that kind covering it, keeping
    one reading (pick_readings)."""
    from sfzmaps.districts import labels

    roles = labels.sheet_roles(series, order)
    log.info("%s sheet roles: %s", series, roles)
    parts = {}
    for kind, col in (("use", "code"), ("height", "height_code")):
        kind_order = [s for s in order if roles[s] == kind]
        parts[kind] = pick_readings(readings(series, parcels, kind_order, kind), col)
    return merge_use_height(parts["use"], parts["height"], SERIES[series].role)


def read_raw(series: str, *, sheets: list[str] | None = None, client=None) -> pd.DataFrame:
    """Reader output for every mapped parcel (COLUMNS). Each parcel is read on its
    highest-priority covering sheet (of the right kind, for reader B); `sheets` limits the
    read to those sheets."""
    from sfzmaps.districts import pattern, sampling

    info = SERIES[series]
    parcels = load_parcels(series_year(series))
    if info.reader == "B":
        order = [s for s in sampling.sheet_order(series) if sheets is None or s in sheets]
        return cap_confidence(_read_b(series, parcels, order), sheet_statuses(series))
    parcels["sheet"] = sampling.assign_sheets(parcels, series).to_numpy()
    order = [s for s in sampling.sheet_order(series) if sheets is None or s in sheets]
    swatches = pattern.legend_swatches(series, client=client)
    out = []
    for sid in order:
        sub = parcels[parcels["sheet"] == sid]
        if not len(sub):
            continue
        log.info("%s sheet %s: %d parcels", series, sid, len(sub))
        r = pattern.read_sheet(series, sid, sub, swatches=swatches, client=client)
        r["height_code"] = None
        r["source_sheet"] = sid
        out.append(r[COLUMNS])
    if not out:
        return pd.DataFrame(columns=COLUMNS)
    return cap_confidence(pd.concat(out, ignore_index=True), sheet_statuses(series))
