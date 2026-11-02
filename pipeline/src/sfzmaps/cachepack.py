"""Answer packs: immutable, content-addressed bundles of cached vision answers.

A pack is a zip of answer files at their cache paths ("llm/<key>.json", "llm/tiles/<key>.json"),
built deterministically and named by the sha256 of its bytes. Its index maps every key to the
sha256 of its answer, so two packs can never disagree on a key unnoticed. On Blob a pack is
cache/packs/<sha>.zip with cache/packs/<sha>.keys.gz beside it; in git, each pack shipped is
recorded as pipeline/cache-packs/<sha>.json. The answers on a branch = the union of the packs
recorded on it.

Applying a pack is put-if-absent and checks every byte: a corrupt or tampered download, a
member outside the answer cache, or an answer that differs from the one already stored raise.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sfzmaps import answers, config

_FIXED_DATE = (2020, 1, 1, 0, 0, 0)


class PackError(Exception):
    pass


@dataclass(frozen=True)
class Pack:
    sha: str
    data: bytes           # the zip
    index: dict[str, str]  # key -> sha256 of the answer bytes

    def index_bytes(self) -> bytes:
        text = "".join(f"{k} {h}\n" for k, h in sorted(self.index.items()))
        return gzip.compress(text.encode(), 9, mtime=0)

    def record(self) -> dict:
        return {"sha": self.sha, "keys": len(self.index), "bytes": len(self.data),
                "index_sha": hashlib.sha256(self.index_bytes()).hexdigest(),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def answer_path(key: str, cache_dir: Path | None = None) -> Path:
    """The local file of an answer key, refusing anything outside the answer cache."""
    answers.validate([key])
    root = ((cache_dir or config.CACHE_DIR) / "llm").resolve()
    p = ((cache_dir or config.CACHE_DIR) / key).resolve()
    if root not in p.parents:
        raise PackError(f"{key} resolves outside the answer cache")
    return p


def make_pack(keys) -> Pack:
    """A deterministic zip of these answers (same keys and bytes -> same pack sha)."""
    keys = answers.validate(keys)
    if not keys:
        raise PackError("no answers to pack")
    buf = io.BytesIO()
    index = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for k in keys:
            try:
                body = answer_path(k).read_bytes()
            except FileNotFoundError:
                raise PackError(f"answer {k} is not in the local cache") from None
            zi = zipfile.ZipInfo(k, _FIXED_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, body)
            index[k] = _sha(body)
    data = buf.getvalue()
    return Pack(_sha(data), data, index)


def parse_index(data: bytes) -> dict[str, str]:
    out = {}
    for line in gzip.decompress(data).decode().splitlines():
        k, h = line.split(" ")
        out[k] = h
    answers.validate(out)
    return out


def apply_pack(data: bytes, expect_sha: str) -> tuple[int, int]:
    """Unpack into the local answer cache, put-if-absent; (written, already present).
    PackError on a sha mismatch, a member outside the cache, or a stored answer that differs."""
    if _sha(data) != expect_sha:
        raise PackError(f"pack {expect_sha[:12]}: bytes don't match its name")
    written = present = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        for info in members:                      # validate everything before writing anything
            answer_path(info.filename)
        conflicts = []
        for info in members:
            body = z.read(info)
            p = answer_path(info.filename)
            if p.exists():
                present += 1
                if p.read_bytes() != body:
                    conflicts.append(info.filename)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f".pack{os.getpid()}.tmp")
            tmp.write_bytes(body)
            try:
                os.link(tmp, p)                   # put-if-absent
                written += 1
            except FileExistsError:
                present += 1
                if p.read_bytes() != body:
                    conflicts.append(info.filename)
            finally:
                tmp.unlink(missing_ok=True)
    if conflicts:
        raise PackError(f"pack {expect_sha[:12]}: {len(conflicts)} answer(s) differ from the ones "
                        f"stored here (e.g. {conflicts[0]}); the cache keeps one answer per key")
    return written, present


# --- records in git -----------------------------------------------------------------------------

def records_dir() -> Path:
    return config.PIPELINE_DIR / "cache-packs"


def recorded() -> dict[str, dict]:
    """sha -> record, for every pack recorded in this checkout."""
    out = {}
    for p in sorted(records_dir().glob("*.json")) if records_dir().exists() else []:
        r = json.loads(p.read_text())
        if r.get("sha") != p.stem:
            raise PackError(f"{p.name}: record names pack {r.get('sha')}")
        out[r["sha"]] = r
    return out


def write_record(pack: Pack) -> Path:
    records_dir().mkdir(parents=True, exist_ok=True)
    p = records_dir() / f"{pack.sha}.json"
    p.write_text(json.dumps(pack.record(), indent=1, sort_keys=True) + "\n")
    return p


# --- locally applied packs -----------------------------------------------------------------------

def _applied_path() -> Path:
    return config.CACHE_DIR / ".packs"


def applied() -> set[str]:
    try:
        return {line for line in _applied_path().read_text().split() if line}
    except FileNotFoundError:
        return set()


def mark_applied(sha: str) -> None:
    _applied_path().parent.mkdir(parents=True, exist_ok=True)
    with open(_applied_path(), "a") as f:
        f.write(sha + "\n")
