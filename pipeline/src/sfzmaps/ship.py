"""Ship a commit's cache to the Blob store, and pull a commit's cache from it.

What a commit needs, so a fresh checkout builds at $0 and finds nothing to do:

* its **processed data**: the DVC objects (data/work, data/out, data/ref) its dvc.lock names,
  stored on Blob under the DVC cache layout, dvc/files/md5/<2>/<30>[.dir];
* its **answers**: the cached vision answers its live artifacts were made from (their
  .answers sidecars, sfzmaps.answers), in content-addressed packs (sfzmaps.cachepack) recorded
  in pipeline/cache-packs/.

`sfzmaps ship` runs on a branch when it is ready to merge: it uploads what main lacks and
commits the new pack's record, so merging the branch gives main exactly what it shipped.
Nothing leaves the machine before that. `sfzmaps pull` fetches a commit's packs and data
(public URLs, no token) and checks the data out. Uploads go through site/scripts/blob-put.mjs
(the token stays in site/.env.local); downloads and existence checks here add a query string,
so no read of a missing object is ever cached at its real URL.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sfzmaps import answers, cachepack, config

BASE = (os.environ.get("NEXT_PUBLIC_TILES_BASE_URL")
        or "https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com").rstrip("/")
DVC_PREFIX = "dvc/files/md5"
PACK_PREFIX = "cache/packs"
NEEDED_PREFIX = "cache/needed"
WORKERS = 16


class ShipError(Exception):
    pass


# --- public reads -------------------------------------------------------------------------------

def _client():
    import httpx
    return httpx.Client(timeout=120, follow_redirects=True)


def remote_size(name: str, client=None) -> int | None:
    """Size of the published object, or None; cache-busted, so a miss is never cached."""
    c = client or _client()
    r = c.head(f"{BASE}/{name}", params={"check": str(time.time_ns())})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return int(r.headers.get("content-length", "-1"))


def download(name: str, dest: Path, alg: str, want: str, client=None) -> None:
    """Stream a published object to dest, atomically, if its hash (alg) is `want`; else
    ShipError and nothing is written."""
    c = client or _client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    h = hashlib.new(alg)
    try:
        with c.stream("GET", f"{BASE}/{name}", params={"check": str(time.time_ns())}) as r:
            if r.status_code == 404:
                raise ShipError(f"{name} is not on the Blob store (was the branch shipped?)")
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
        if h.hexdigest() != want:
            raise ShipError(f"{name}: downloaded bytes don't match their name")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


# --- DVC objects ---------------------------------------------------------------------------------

def _repo_root() -> Path:
    return config.PIPELINE_DIR.parent


def dvc_cache() -> Path:
    """The DVC cache this checkout uses (.dvc/config.local may point a worktree elsewhere)."""
    p = subprocess.run(["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc", "cache", "dir"],
                       cwd=_repo_root(), capture_output=True, text=True)
    return Path(p.stdout.strip()) if p.returncode == 0 and p.stdout.strip() else _repo_root() / ".dvc" / "cache"


def object_path(md5: str, cache: Path) -> Path:
    return cache / "files" / "md5" / md5[:2] / md5[2:]


def object_name(md5: str) -> str:
    return f"{DVC_PREFIX}/{md5[:2]}/{md5[2:]}"


def lock_outs(lock_text: str) -> list[str]:
    """md5s (".dir" for folders) of every output a dvc.lock names."""
    import yaml

    lock = yaml.safe_load(lock_text) or {}
    return sorted({o["md5"] for st in (lock.get("stages") or {}).values() for o in st.get("outs") or []})


def objects(outs: list[str], read_dir) -> list[str]:
    """Every object of these outputs: each .dir manifest and the files it lists.
    read_dir(md5) -> the manifest's entries."""
    found = []
    for md5 in outs:
        found.append(md5)
        if md5.endswith(".dir"):
            found += [e["md5"] for e in read_dir(md5)]
    return sorted(set(found))


def _md5_hex(md5: str) -> str:
    return md5.removesuffix(".dir")


# --- answers a commit needs -----------------------------------------------------------------------

def live_artifacts() -> list[Path]:
    """The artifacts whose answers the current data was made from: every page's page.json,
    labels.json of map pages, each read series' districts_raw.parquet, each height series'
    height.parquet. Stale leftovers (a reclassified sheet's labels, partial reads, superseded
    legends) are never in this list; legends and district labels are covered through the
    district read that consumed them."""
    from sfzmaps import pages, reading
    from sfzmaps.districts import assemble
    from sfzmaps.districts.series import SERIES

    arts = []
    for page in pages.discover(config.RAW_DIR):
        d = config.WORK_DIR / page.series / page.sheet_id
        pj = d / "page.json"
        if not pj.exists():
            continue
        arts.append(pj)
        if reading.load_page(pj).kind == "map" and (d / "labels.json").exists():
            arts.append(d / "labels.json")
    for s, info in SERIES.items():
        if info.role == "use" and assemble._raw_path(s).exists():
            arts.append(assemble._raw_path(s))
        elif info.role == "height":
            hp = assemble.out_dir(assemble.zoning_series_of(s)) / "height.parquet"
            if hp.exists():
                arts.append(hp)
    return arts


def needed_answers() -> tuple[set[str], list[Path]]:
    """(answer keys the live artifacts were made from, artifacts with no known provenance)."""
    keys, unknown = set(), []
    for a in live_artifacts():
        got = answers.read(a)
        if got is None:
            if not answers.covered(a):
                unknown.append(a)
            continue
        keys.update(got)
    return keys, unknown


def known_keys(records: dict[str, dict], client=None) -> dict[str, str]:
    """key -> answer sha over the given packs (indexes cached locally, fetched if missing);
    ShipError if two packs disagree on a key."""
    idx_dir = config.CACHE_DIR / "pack-index"
    out: dict[str, str] = {}
    for sha, rec in sorted(records.items()):
        p = idx_dir / f"{sha}.keys.gz"
        if not p.exists():
            download(f"{PACK_PREFIX}/{sha}.keys.gz", p, "sha256", rec["index_sha"], client)
        for k, h in cachepack.parse_index(p.read_bytes()).items():
            if out.setdefault(k, h) != h:
                raise ShipError(f"packs disagree on {k}")
    return out


# --- upload (node, @vercel/blob) ------------------------------------------------------------------

def upload(entries: list[dict]) -> dict[str, str]:
    """Upload manifest entries with site/scripts/blob-put.mjs; name -> status. The token is read
    by node from site/.env.local; it never passes through here."""
    if not entries:
        return {}
    site = _repo_root() / "site"
    if not (site / ".env.local").exists():
        raise ShipError("site/.env.local is missing (BLOB_READ_WRITE_TOKEN, from `vercel env pull`)")
    p = subprocess.run(["node", "--env-file=.env.local", "scripts/blob-put.mjs"], cwd=site,
                       input=json.dumps(entries), capture_output=True, text=True)
    status = {}
    for line in p.stdout.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        status[r["name"]] = r["status"] if r["status"] != "error" else f"error: {r.get('error')}"
    bad = {n: s for n, s in status.items() if s.startswith("error")}
    if p.returncode != 0 or bad or len(status) != len(entries):
        raise ShipError("upload failed: " + "; ".join(f"{n}: {s}" for n, s in list(bad.items())[:5])
                        + (f" {p.stderr.strip()[-300:]}" if p.stderr.strip() else ""))
    return status


# --- what a commit needs, published so CI can check it ----------------------------------------------

def needed_path() -> Path:
    """pipeline/cache-needed.json: which dvc.lock was shipped, and the published list of the
    answers that commit needs (cache/needed/<sha>.keys.gz)."""
    return config.PIPELINE_DIR / "cache-needed.json"


def needed_bytes(keys) -> bytes:
    import gzip
    return gzip.compress("".join(k + "\n" for k in answers.validate(keys)).encode(), 9, mtime=0)


def lock_md5() -> str:
    return hashlib.md5((config.PIPELINE_DIR / "dvc.lock").read_bytes()).hexdigest()


# --- ship -----------------------------------------------------------------------------------------

@dataclass
class ShipReport:
    pack: str | None
    new_answers: int
    objects_total: int
    objects_uploaded: int
    bytes_uploaded: int


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=_repo_root(), check=True, capture_output=True,
                          text=True).stdout


def records_at(ref: str) -> dict[str, dict]:
    """Pack records as they are at a git ref (e.g. origin/main)."""
    out = {}
    try:
        names = _git("ls-tree", "--name-only", f"{ref}:pipeline/cache-packs").split()
    except subprocess.CalledProcessError:
        return out
    for n in names:
        if n.endswith(".json"):
            r = json.loads(_git("show", f"{ref}:pipeline/cache-packs/{n}"))
            out[r["sha"]] = r
    return out


def preflight(base: str = "origin/main") -> None:
    dirty = [l for l in _git("status", "--porcelain", "--", "pipeline", "pixi.toml", "pixi.lock").splitlines()
             if l.strip()]
    if dirty:
        raise ShipError("uncommitted changes; commit first, so what ships is what merges:\n  "
                        + "\n  ".join(dirty[:10]))
    st = subprocess.run(["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc", "status", "--json"],
                        cwd=config.PIPELINE_DIR, capture_output=True, text=True)
    if st.returncode != 0 or json.loads(st.stdout or "{}"):
        raise ShipError("the data doesn't match dvc.lock; run `uv run sfzmaps build` first")
    if not _git("rev-list", f"{base}..HEAD").strip():
        raise ShipError(f"nothing to ship: HEAD is not ahead of {base}")


def ship(base: str = "origin/main", dry_run: bool = False) -> ShipReport:
    """Upload what this commit needs that `base` and this branch haven't shipped, and commit the
    new pack's record."""
    preflight(base)
    client = _client()
    keys, unknown = needed_answers()
    if unknown:
        raise ShipError(f"{len(unknown)} artifact(s) have no answers record and no baseline entry "
                        f"(e.g. {unknown[0]}); rebuild them, or list them in cache-baseline.json")
    records = {**records_at(base), **cachepack.recorded()}
    have = known_keys(records, client)
    new = sorted(keys - set(have))
    cache = dvc_cache()
    outs = lock_outs((config.PIPELINE_DIR / "dvc.lock").read_text())
    objs = objects(outs, lambda md5: json.loads(object_path(md5, cache).read_text()))
    with ThreadPoolExecutor(WORKERS) as ex:
        sizes = list(ex.map(lambda m: remote_size(object_name(m), client), objs))
    missing = [m for m, s in zip(objs, sizes) if s is None]
    nbytes = sum(object_path(m, cache).stat().st_size for m in missing)
    if dry_run:
        return ShipReport(None, len(new), len(objs), len(missing), nbytes)
    pack_sha = None
    entries = [{"file": str(object_path(m, cache)), "name": object_name(m), "alg": "md5",
                "hash": _md5_hex(m)} for m in missing]
    nb = needed_bytes(keys)
    needed = {"lock_md5": lock_md5(), "keys_sha": hashlib.sha256(nb).hexdigest(), "keys": len(keys)}
    with tempfile.TemporaryDirectory(prefix="sfzmaps-ship-") as tmp:
        np_ = Path(tmp) / "needed.keys.gz"
        np_.write_bytes(nb)
        entries.append({"file": str(np_), "name": f"{NEEDED_PREFIX}/{needed['keys_sha']}.keys.gz",
                        "alg": "sha256", "hash": needed["keys_sha"]})
        if new:
            pack = cachepack.make_pack(new)
            zp, ip = Path(tmp) / f"{pack.sha}.zip", Path(tmp) / f"{pack.sha}.keys.gz"
            zp.write_bytes(pack.data)
            ip.write_bytes(pack.index_bytes())
            index_sha = hashlib.sha256(pack.index_bytes()).hexdigest()
            entries += [{"file": str(zp), "name": f"{PACK_PREFIX}/{pack.sha}.zip", "alg": "sha256", "hash": pack.sha},
                        {"file": str(ip), "name": f"{PACK_PREFIX}/{pack.sha}.keys.gz", "alg": "sha256",
                         "hash": index_sha}]
            pack_sha = pack.sha
        upload(entries)                      # data, list and pack first ...
        files = []                           # ... the records only once they're all there
        if pack_sha:
            files.append(cachepack.write_record(pack))
            cachepack.mark_applied(pack_sha)
        old = needed_path().read_text() if needed_path().exists() else None
        new_text = json.dumps(needed, indent=1, sort_keys=True) + "\n"
        if old != new_text:
            needed_path().write_text(new_text)
            files.append(needed_path())
        if files:
            _git("add", *map(str, files))
            _git("commit", "-q", "-m", f"cache: ship {len(objs)} data objects, {len(keys)} answers"
                 + (f" (new pack {pack_sha[:12]}: {len(new)})" if pack_sha else ""))
    return ShipReport(pack_sha, len(new), len(objs), len(missing), nbytes)


# --- pull -----------------------------------------------------------------------------------------

def pull(dvc_checkout: bool = True) -> tuple[int, int]:
    """Fetch this commit's packs and data from the Blob store; (packs applied, objects fetched)."""
    client = _client()
    done = cachepack.applied()
    applied_n = 0
    for sha, rec in sorted(cachepack.recorded().items()):
        if sha in done:
            continue
        with tempfile.TemporaryDirectory(prefix="sfzmaps-pull-") as tmp:
            zp = Path(tmp) / f"{sha}.zip"
            download(f"{PACK_PREFIX}/{sha}.zip", zp, "sha256", sha, client)
            cachepack.apply_pack(zp.read_bytes(), sha)
        cachepack.mark_applied(sha)
        applied_n += 1
    cache = dvc_cache()
    outs = lock_outs((config.PIPELINE_DIR / "dvc.lock").read_text())

    def fetch(md5: str) -> int:
        p = object_path(md5, cache)
        if p.exists():
            return 0
        download(object_name(md5), p, "md5", _md5_hex(md5), client)
        return 1

    with ThreadPoolExecutor(WORKERS) as ex:      # manifests first: they list the files
        n = sum(ex.map(fetch, [m for m in outs if m.endswith(".dir")]))
        objs = objects(outs, lambda md5: json.loads(object_path(md5, cache).read_text()))
        n += sum(ex.map(fetch, [m for m in objs if not m.endswith(".dir")]))
    if dvc_checkout:
        subprocess.run(["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc", "checkout"],
                       cwd=config.PIPELINE_DIR, check=True)
    return applied_n, n
