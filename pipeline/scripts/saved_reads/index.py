"""Index every saved answer in the cache (llm/*.json and llm/tiles/*.json) with what is known of how it was got.

One record per answer, sorted by key, in OUT/reads.jsonl.gz:
  key        the cache file, "llm/<sha>.json" or "llm/tiles/<sha>.json"
  answer     the stored answer (JSON), exactly as on disk
  sha256     of the answer file's bytes
  stored     the answer file's modification time (UTC): when the answer was written, or when a pack holding it was
             applied to this cache (then "pack" is set)
  pack       the answer pack (pipeline/cache-packs/<sha>.json) that holds it, if any: shipped, so `sfzmaps pull` brings it
  used_by    pipeline outputs whose .answers sidecar lists it (paths under the data folder)
  asked      every recorded way of asking it (record.py, from the --calls folders): model, effort, prompt, schema,
             image, run.  Recorded lookups make the key from those parts exactly as llm does, so a match proves them.
Also OUT/summary.json: counts by kind, how many answers have a recorded way of asking, a pipeline use, a pack, none.
usage: python -m saved_reads index --out OUT [--calls DIR ...]"""
import gzip
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from sfzmaps import config

CACHE = config.CACHE_DIR


def _packs() -> dict[str, str]:
    """answer key -> the pack holding it, from the applied packs' key lists"""
    out = {}
    for f in sorted((CACHE / "pack-index").glob("*.keys.gz")):
        sha = f.name.split(".")[0]
        for line in gzip.open(f, "rt"):
            if line.strip():
                out.setdefault(line.split()[0], sha)
    return out


def _uses() -> dict[str, list[str]]:
    """answer key -> pipeline outputs whose .answers sidecar lists it"""
    out = defaultdict(list)
    data = config.DATA_DIR
    seen = set()
    roots = [config.WORK_DIR, data / "out"]          # where the pipeline writes outputs (and their sidecars)
    for top in roots:
        for root, dirs, files in os.walk(top, followlinks=True):
            real = os.path.realpath(root)
            if real in seen:                         # a folder reached twice through links: read it once
                dirs[:] = []
                continue
            seen.add(real)
            _sidecars(Path(root), files, data, out)
    return out


def _sidecars(root: Path, files, data: Path, out) -> None:
    """add one folder's .answers sidecars: each lists the answer keys its output was made from"""
    for f in files:
        if f.endswith(".answers"):
            p = root / f
            rel = str(p.relative_to(data))[: -len(".answers")]
            for line in p.read_text().splitlines():
                if line.strip():
                    out[line.strip()].append(rel)


def _asked(folders: list[Path]) -> dict[str, list[dict]]:
    """answer key -> distinct recorded ways of asking it"""
    out = defaultdict(list)
    for d in folders:
        for line in open(d / "calls.jsonl"):
            r = json.loads(line)
            way = {k: r[k] for k in ("kind", "model", "params", "effort", "prompt", "schema", "image", "run")
                   if k in r}
            way["record"] = d.name
            if way not in out[r["key"]]:
                out[r["key"]].append(way)
    return out


def main(out: Path, folders: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    packs, uses, asked = _packs(), _uses(), _asked(folders)
    files = sorted([("llm", e) for e in os.scandir(CACHE / "llm") if e.name.endswith(".json")]
                   + [("llm/tiles", e) for e in os.scandir(CACHE / "llm" / "tiles") if e.name.endswith(".json")],
                   key=lambda t: f"{t[0]}/{t[1].name}")
    n = defaultdict(int)
    with gzip.open(out / "reads.jsonl.gz", "wt", compresslevel=9) as g:
        for sub, e in files:
            key = f"{sub}/{e.name}"
            raw = Path(e.path).read_bytes()
            rec = {"key": key, "answer": json.loads(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                   "stored": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.stat().st_mtime))}
            if key in packs:
                rec["pack"] = packs[key]
            if key in uses:
                rec["used_by"] = sorted(uses[key])
            if key in asked:
                rec["asked"] = asked[key]
            g.write(json.dumps(rec, sort_keys=True) + "\n")
            kind = "tile" if sub.endswith("tiles") else "ask"
            n[f"{kind}: answers"] += 1
            n[f"{kind}: asked recorded"] += key in asked
            n[f"{kind}: used by the pipeline"] += key in uses
            n[f"{kind}: in a shipped pack"] += key in packs
            n[f"{kind}: none of these"] += not (key in asked or key in uses or key in packs)
    (out / "summary.json").write_text(json.dumps(dict(n), indent=1))
    print(json.dumps(dict(n), indent=1))
