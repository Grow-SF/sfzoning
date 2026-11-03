"""Data packs: DVC cache objects bundled into a few large, content-addressed zips.

Shipping each DVC object alone costs one request per object, and most objects are tiny. So
objects under SINGLE_MIN go into packs of at most PACK_MAX bytes; larger ones are still stored
alone. A pack is an uncompressed zip whose member names are the objects' md5s (".dir" kept for
folder manifests), built deterministically and named by the sha256 of its bytes.

Applying a pack checks every member before writing any: a member whose bytes don't match its
md5, a name that isn't an md5, or a broken zip raise and nothing is written. Writes are atomic
and members already in the cache are skipped, so an interrupted pull resumes cleanly.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import zipfile
from pathlib import Path

SINGLE_MIN = 8 * 1024 * 1024    # objects this large or larger are stored alone
PACK_MAX = 256 * 1024 * 1024    # a pack holds at most this many bytes of objects

_FIXED_DATE = (2020, 1, 1, 0, 0, 0)
_NAME = re.compile(r"[0-9a-f]{32}(\.dir)?")
_CHUNK = 1 << 20


class DataPackError(Exception):
    pass


def _object_path(md5: str, cache: Path) -> Path:
    # Same layout as ship.object_path; kept here so ship can import this module.
    return cache / "files" / "md5" / md5[:2] / md5[2:]


def plan_packs(objs: dict[str, int]) -> tuple[list[list[str]], list[str]]:
    """Given md5 -> size: (packs, each a sorted list of md5s totalling at most PACK_MAX bytes;
    the md5s stored alone, each at least SINGLE_MIN)."""
    singles = sorted(m for m, n in objs.items() if n >= SINGLE_MIN)
    packs: list[list[str]] = []
    cur: list[str] = []
    total = 0
    for m in sorted(m for m, n in objs.items() if n < SINGLE_MIN):
        if cur and total + objs[m] > PACK_MAX:
            packs.append(cur)
            cur, total = [], 0
        cur.append(m)
        total += objs[m]
    if cur:
        packs.append(cur)
    return packs, singles


def make_pack(md5s: list[str], cache: Path, out: Path) -> str:
    """Write a deterministic, uncompressed zip of these cache objects to `out` (atomically);
    return its sha256."""
    names = sorted(set(md5s))
    if not names:
        raise DataPackError("no objects to pack")
    for m in names:
        if not _NAME.fullmatch(m):
            raise DataPackError(f"{m!r} is not a DVC md5")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for m in names:
                try:
                    body = _object_path(m, cache).read_bytes()
                except FileNotFoundError:
                    raise DataPackError(f"object {m} is not in the DVC cache") from None
                zi = zipfile.ZipInfo(m, _FIXED_DATE)
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o644 << 16
                z.writestr(zi, body)
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return h.hexdigest()


def _member_md5(z: zipfile.ZipFile, info: zipfile.ZipInfo, sink=None) -> str:
    h = hashlib.md5()
    with z.open(info) as src:
        for chunk in iter(lambda: src.read(_CHUNK), b""):
            h.update(chunk)
            if sink is not None:
                sink.write(chunk)
    return h.hexdigest()


def apply_pack(zip_path: Path, cache: Path, want: set[str] | None = None) -> int:
    """Extract this pack's members (only those in `want`, if given) into the DVC cache.
    Every member to write is checked first; any mismatch raises DataPackError and nothing is
    written. Members already present are skipped. Returns the number written."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            todo = []
            for info in z.infolist():
                m = info.filename
                if not _NAME.fullmatch(m):
                    raise DataPackError(f"{zip_path.name}: member {m!r} is not a DVC md5")
                if want is not None and m not in want:
                    continue
                if _object_path(m, cache).exists():
                    continue
                if _member_md5(z, info) != m.removesuffix(".dir"):
                    raise DataPackError(f"{zip_path.name}: member {m} doesn't match its md5")
                todo.append(info)
            for info in todo:
                m = info.filename
                dest = _object_path(m, cache)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + f".{os.getpid()}.{time.time_ns()}.tmp")
                try:
                    with open(tmp, "wb") as f:
                        got = _member_md5(z, info, f)
                    if got != m.removesuffix(".dir"):
                        raise DataPackError(f"{zip_path.name}: member {m} changed while reading")
                    os.replace(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        raise DataPackError(f"{zip_path.name}: not a valid pack ({e})") from None
    return len(todo)
