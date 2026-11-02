"""Content-based freshness for per-sheet steps (render, blocks, rough, ..., warp, paper warp).

A step's output is current when it exists and its fingerprint sidecar,
`<output>.inputs.json`, equals the fingerprint of its inputs now: the sha256 of each input
file's bytes plus VERSION. Modification times are never trusted: a `git switch` or a DVC
checkout rewrites files with new mtimes but the same bytes, and must not rerun anything.

Hashes are memoized per process, keyed by (device, inode, size, mtime), so a big raster read by
several steps is hashed once.

Adoption (outputs made before sidecars existed): an output without a sidecar is adopted (a
sidecar is written from its inputs as they are, and the step skipped) only when DVC proves the
output and every input are exactly the files the last recorded build made or used: each file's
md5 equals the one `dvc.lock` records for the output's stage (inside a recorded output or
directory dependency, through the directory listing in the DVC cache; a file dependency such as
src/sfzmaps/warp.py, directly). A directory dependency whose listing isn't in the cache
(data/raw) counts as unchanged only when `sfzmaps build` says so (SFZMAPS_DVC_CHANGED_DEPS, from
`dvc status`). Anything else is recomputed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from sfzmaps import config

log = logging.getLogger(__name__)

VERSION = 1  # bump to recompute every per-sheet step (a change in what a fingerprint means)
CHANGED_DEPS_ENV = "SFZMAPS_DVC_CHANGED_DEPS"

_memo: dict[tuple, str] = {}


def _digest(p: Path, algo: str) -> str:
    st = p.stat()
    key = (algo, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _memo:
        h = hashlib.new(algo)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _memo[key] = h.hexdigest()
    return _memo[key]


def sha256(p: Path) -> str:
    return _digest(Path(p), "sha256")


def sidecar(out: Path) -> Path:
    return out.with_name(out.name + ".inputs.json")


def fingerprint(inputs: list[Path]) -> dict:
    return {"version": VERSION, "inputs": [[p.name, sha256(p)] for p in inputs]}


def is_fresh(out: Path, inputs: list[Path], force: bool = False) -> bool:
    if force or not out.exists() or not inputs or not all(i.exists() for i in inputs):
        return False
    try:
        have = json.loads(sidecar(out).read_text())
    except (OSError, ValueError):
        have = None
    if have is not None:
        return have == fingerprint(inputs)
    if _adoptable(out, inputs):
        record(out, inputs)
        log.debug("%s: adopted (matches dvc.lock)", out)
        return True
    return False


def record(out: Path, inputs: list[Path]) -> None:
    """Write out's sidecar from its inputs as they are now (unchanged content is not rewritten)."""
    if not out.exists():
        return
    p = sidecar(out)
    text = json.dumps(fingerprint(inputs), indent=1)
    try:
        if p.read_text() == text:
            return
    except OSError:
        pass
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)


# -- adoption from dvc.lock -------------------------------------------------------------------

_lock: dict | None = None
_listings: dict[str, dict | None] = {}


def _load_lock() -> dict:
    global _lock
    if _lock is None:
        import yaml
        try:
            _lock = yaml.safe_load((config.PIPELINE_DIR / "dvc.lock").read_text()) or {}
        except (OSError, ValueError, yaml.YAMLError):
            _lock = {}
    return _lock


def _cache_dir() -> Path:
    dvc_dir = config.PIPELINE_DIR.parent / ".dvc"
    found = dvc_dir / "cache"
    for name in ("config", "config.local"):  # later files win, as in DVC
        section = None
        try:
            lines = (dvc_dir / name).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            s = line.strip()
            if s.startswith("["):
                section = s.strip("[]").strip()
            elif section == "cache" and "=" in s:
                k, v = (x.strip() for x in s.split("=", 1))
                if k == "dir":
                    found = (dvc_dir / v) if not os.path.isabs(v) else Path(v)
    return found


def _listing(md5: str) -> dict | None:
    """{relpath: md5} of a directory recorded as `<md5>.dir`, from the DVC cache; None if absent."""
    if md5 not in _listings:
        h = md5.removesuffix(".dir")
        try:
            items = json.loads((_cache_dir() / "files" / "md5" / h[:2] / (h[2:] + ".dir")).read_text())
            _listings[md5] = {e["relpath"]: e["md5"] for e in items}
        except (OSError, ValueError, KeyError, TypeError):
            _listings[md5] = None
    return _listings[md5]


def _rel(p: Path) -> str | None:
    r = os.path.relpath(os.path.abspath(p), config.PIPELINE_DIR)
    return None if r.startswith("..") else Path(r).as_posix()


def _within(rel: str, entry: str) -> bool:
    return rel == entry or rel.startswith(entry.rstrip("/") + "/")


def _adoptable(out: Path, inputs: list[Path]) -> bool:
    rel_out = _rel(out)
    if rel_out is None:
        return False
    stage = next((s for s in (_load_lock().get("stages") or {}).values()
                  if any(_within(rel_out, o["path"]) for o in s.get("outs") or [])), None)
    if stage is None:
        return False
    entries = sorted((e for e in (stage.get("outs") or []) + (stage.get("deps") or [])
                      if e.get("md5")), key=lambda e: -len(e["path"]))
    changed_env = os.environ.get(CHANGED_DEPS_ENV)
    changed = None if changed_env is None else set(json.loads(changed_env))
    for p in [out, *inputs]:
        rel = _rel(p)
        e = next((e for e in entries if rel is not None and _within(rel, e["path"])), None)
        if e is None:
            return False
        if not e["md5"].endswith(".dir"):
            if rel != e["path"] or _digest(p, "md5") != e["md5"]:
                return False
            continue
        listing = _listing(e["md5"])
        if listing is not None:
            if listing.get(rel[len(e["path"]):].lstrip("/")) != _digest(p, "md5"):
                return False
        elif changed is None or e["path"] in changed:
            return False
    return True
