"""Check that the "published" map the review compares with is the one the site shows today: for each series, the md5
of this checkout's published file (districts/parcels.fgb; a height map's period's districts/height.parquet) against
the entry of the same file in a dvc.lock (main's), through the DVC cache's directory listings.

    uv run python scripts/all_years/verify_old.py MAIN_DVC_LOCK CACHE_DIR [--json OUT]

Prints one line per series; with --json writes {series: {"file", "lock", "here", "match"}}."""
import hashlib
import json
import sys
from pathlib import Path

import yaml

from sfzmaps import config
from sfzmaps.districts.series import SERIES as INFO
from sfzmaps.solver import paths


def published_file(s: str) -> str:
    info = INFO[s]
    if info.role == "height":
        return f"data/out/{info.period}-zoning/districts/height.parquet"
    return f"data/out/{s}/districts/parcels.fgb"


def lock_md5(lock: dict, cache: Path, rel: str) -> str | None:
    """The md5 the lock records for the pipeline file `rel`: a file out of that path, or the file inside a directory
    out (its .dir listing in the cache)."""
    for st in lock["stages"].values():
        for o in st.get("outs", []):
            p, md5 = o["path"], o.get("md5", "")
            if p == rel:
                return md5
            if rel.startswith(p + "/") and md5.endswith(".dir"):
                listing = cache / "files" / "md5" / md5[:2] / md5[2:]
                if not listing.is_file():
                    return None
                inner = rel[len(p) + 1:]
                return next((e["md5"] for e in json.loads(listing.read_text()) if e["relpath"] == inner), None)
    return None


def main():
    lock = yaml.safe_load(Path(sys.argv[1]).read_text())
    cache = Path(sys.argv[2])
    out = {}
    keep = config.DATA_DIR / "solver" / "experiments" / "all-years" / "old-maps"
    for s in (paths.SERIES_2000, *paths.ALL_YEARS):
        rel = published_file(s)
        f = config.PIPELINE_DIR / rel
        here = hashlib.md5(f.read_bytes()).hexdigest() if f.is_file() else None
        want = lock_md5(lock, cache, rel)
        out[s] = {"file": rel, "lock": want, "here": here, "match": bool(want and here == want), "use": str(f)}
        if want and here != want:                   # this checkout holds another map: take the lock's from the cache
            cached = cache / "files" / "md5" / want[:2] / want[2:]
            if cached.is_file():
                keep.mkdir(parents=True, exist_ok=True)
                copy = keep / f"{s}{Path(rel).suffix}"
                copy.write_bytes(cached.read_bytes())
                out[s].update(use=str(copy), match=True, source="the DVC cache, by main's md5")
        print(s, rel, "match" if out[s]["match"] else f"DIFFERS (lock {want}, here {here})", out[s]["use"])
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
