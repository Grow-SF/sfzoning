"""Put the package's answers back into the answer cache (pipeline/data/cache/llm), so a fresh clone can replay every
reading script cache-only (record_all.sh) with no paid call and no download: each record carries its answer and the
cache file name it is stored under (cache_key).  Answers already in the cache are left as they are.
usage: python -m saved_reads restore-cache [--from PACKAGE]   (default pipeline/saved-reads)"""
import gzip
import json
import os
import tempfile
from pathlib import Path

from sfzmaps import config


def main(package: Path) -> None:
    cache = config.CACHE_DIR
    added = kept = 0
    for f in sorted((package / "reads").glob("*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            dest = cache / r["cache_key"]
            if dest.exists():
                kept += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as out:
                json.dump(r["answer"], out)
            os.replace(tmp, dest)
            added += 1
    print(f"answers written to {cache}: {added}; already there: {kept}")
