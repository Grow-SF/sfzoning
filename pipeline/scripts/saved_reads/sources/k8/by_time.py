"""Answers stored in the cache during a time window, grouped by the shape of the answer (its field names), for reads
whose questions cannot be asked again exactly (their inputs came from old solves, or the script was changed after its
paid run): the run that made them is known from its command and time, the answers from when they were stored.
usage: python -m saved_reads by-time FROM TO   (UTC, e.g. 2026-11-10T08:32:00 2026-11-10T09:10:00)"""
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from sfzmaps import config


def stored_between(t0: str, t1: str) -> list[tuple[float, str, dict]]:
    a = datetime.fromisoformat(t0).replace(tzinfo=timezone.utc).timestamp()
    b = datetime.fromisoformat(t1).replace(tzinfo=timezone.utc).timestamp()
    out = []
    for e in os.scandir(config.CACHE_DIR / "llm"):
        if e.name.endswith(".json"):
            m = e.stat().st_mtime
            if a <= m <= b:
                out.append((m, f"llm/{e.name}", json.loads(open(e.path).read())))
    return sorted(out)


def shape(answer: dict) -> str:
    keys = sorted(answer)
    inner = []
    for k in keys:
        v = answer[k]
        if isinstance(v, list) and v and isinstance(v[0], dict):
            inner.append(f"{k}[{','.join(sorted(v[0]))}]")
    return ",".join(keys) + (" " + " ".join(inner) if inner else "")


if __name__ == "__main__":
    rows = stored_between(sys.argv[1], sys.argv[2])
    by = defaultdict(list)
    for m, k, a in rows:
        by[shape(a)].append(m)
    for s, ms in sorted(by.items(), key=lambda kv: kv[1][0]):
        f = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m-%dT%H:%M:%S")
        print(f"{len(ms):5d}  {f(ms[0])} .. {f(ms[-1])}  {s[:150]}")
