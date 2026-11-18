"""The April 2000 zoning table: SF Planning's "Historical Zoning Districts - April 2000" lots, the district the table
gives each lot.  Downloaded once from DataSF and checked in (`inputs/zoning-2000/`); the lot index everywhere in the
solver is the row order of that file."""
import csv
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import shapely

from sfzmaps.solver import codes, paths

DATASET = "cfhj-8c6b"
NAME = "Historical Zoning Districts - April 2000"
EXPORT_URL = f"https://data.sfgov.org/api/geospatial/{DATASET}?method=export&format=GeoJSON"
TABLE_FILE = "april2000.geojson.gz"
PROVENANCE_FILE = "PROVENANCE.md"
CORRECTIONS_FILE = "table-corrections.csv"
TABLE_CRS = "EPSG:2227"
COMMAND = "uv run sfzmaps solver fetch-table"


def _download(url: str) -> bytes:
    """The bytes at `url`, in one request."""
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def fetch_table(dest: Path | None = None) -> dict:
    """Download the whole table once and write `april2000.geojson.gz` (gzip with mtime 0, so the same bytes make the
    same file) and `PROVENANCE.md` into `dest`.  Returns the provenance."""
    dest = Path(dest or paths.TABLE_DIR)
    raw = _download(EXPORT_URL)
    rows = len(json.loads(raw)["features"])
    info = {
        "url": EXPORT_URL, "dataset": DATASET, "name": NAME,
        "publisher": "San Francisco Planning", "license": "PDDL 1.0",
        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": rows, "sha256": hashlib.sha256(raw).hexdigest(), "command": COMMAND,
    }
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / TABLE_FILE, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)
    lines = [f"# {NAME}", "",
             "The checked-in `april2000.geojson.gz` is the dataset's whole-table GeoJSON export, compressed with gzip.",
             "The sha256 is of the uncompressed bytes.", "",
             "| field | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in info.items()]
    (dest / PROVENANCE_FILE).write_text("\n".join(lines) + "\n")
    return info


def load_table() -> gpd.GeoDataFrame:
    """The checked-in table in EPSG:2227 feet, in file order: `zoning` upper-cased and stripped ("" where blank),
    valid geometry, and an `area` column (square feet)."""
    with gzip.open(paths.TABLE_DIR / TABLE_FILE, "rb") as f:
        t = gpd.read_file(io.BytesIO(f.read())).to_crs(TABLE_CRS)
    t["zoning"] = t["zoning"].fillna("").str.strip().str.upper()
    t["geometry"] = shapely.make_valid(t.geometry.values)
    t["area"] = t.geometry.area
    return t


def known_and_codes(table: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`known`: lots whose zoning is a single district (not blank, no "/").  `codes`: each lot's code in the one
    spelling, "" where not known.  For a series the table does not describe (`paths.has_table`: every series but the
    2000 map and its 1996-2000 scan) no lot is known: the table gives such a series its lots, never their codes."""
    tidied = table["zoning"].map(codes.tidy)
    known = ((tidied != "") & ~tidied.str.contains("/")).values
    if not paths.has_table():
        known = np.zeros(len(tidied), bool)
    code = np.array([codes.canon(c) for c in tidied.values], dtype=object)
    return known, np.where(known, code, "").astype(str)


def corrected_codes(table: gpd.GeoDataFrame) -> np.ndarray:
    """The table's codes with the by-eye checks of `table-corrections.csv` applied (a verdict "A" sets column A on
    the listed lots).  Used only to score; the solver never reads it."""
    known, code = known_and_codes(table)
    code = code.astype(object)
    row = {b: i for i, b in enumerate(table["blklot"].astype(str).values)}
    with open(paths.TABLE_DIR / CORRECTIONS_FILE, newline="") as f:
        for r in csv.DictReader(f):
            if r["verdict"] == "A":
                for b in r["blklots"].split():
                    if b in row:
                        code[row[b]] = codes.canon(r["A"])
    return np.where(known, code, "").astype(str)
