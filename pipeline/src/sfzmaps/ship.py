"""Ship a commit's cache to the Blob store, and pull a commit's cache from it.

What a commit needs, so a fresh checkout builds at $0 and finds nothing to do:

* its **processed data**: the DVC objects (data/work, data/out, data/ref) its dvc.lock names.
  Objects under 8 MB (datapack.SINGLE_MIN) travel in *data packs* (sfzmaps.datapack): zips of
  at most 256 MB, cache/datapacks/<sha256>.zip. Larger objects are stored alone under the DVC
  cache layout, dvc/files/md5/<2>/<30>[.dir]. For about 10,450 objects (10.6 GB) that is 13
  packs and about 200 single objects, so a fresh pull makes about 220 requests instead of one
  per object;
* its **answers**: the cached vision answers its live artifacts were made from (their
  .answers sidecars, sfzmaps.answers), in content-addressed packs (sfzmaps.cachepack) recorded
  in pipeline/cache-packs/.

`sfzmaps ship` runs on a branch when it is ready to merge: it uploads what main lacks and
commits the new pack's record, so merging the branch gives main exactly what it shipped.
Nothing leaves the machine before that. `sfzmaps pull` fetches a commit's packs and data
(public URLs, no token) and checks the data out.

Requests are kept to the fewest that get the data (a burst of per-object checks once tripped
Vercel's DDoS protection): ship learns what the store holds from its list API (one call per
1,000 names, site/scripts/blob-list.mjs), uploads only what's missing, one file at a time
(site/scripts/blob-put.mjs: one request per file up to 100 MB; a larger one with the SDK's
multipart upload, put(..., { multipart: true }), which @vercel/blob 2.8.2 splits into 8 MB
parts sent 8 at once in Node, retrying failed parts: 2 + parts requests), lists once more to
confirm, then publishes a *shipped
manifest* (cache/shipped/<sha>.json, version 2: every object, the data pack each small one is
in, and every pack it confirmed, for this dvc.lock). An object a pack on main already holds
keeps that pack; only new small objects go into new packs. `ship --dry-run` prints the
requests it would make; more than MAX_REQUESTS need --yes. `ship --repack` (once, from the
version 1 layout where every object was stored alone) packs the small objects too.
pipeline/cache-needed.json names it. The merge gate then reads two files and the pack indexes,
never one request per object; pull reads the manifest and fetches only what it lacks: each
data pack holding a missing object once, then single objects, 4 at a time for both.
Version 1 manifests (older commits) still pull and pass the gate.

Vercel's Blob docs set the limits these settings sit under. Pro rate limits: 4,500 advanced
operations a minute (75/s) and 7,200 simple ones (120/s), counted per blob: "Deleting 100
blobs in a batch counts as 100 operations, not one." Multipart parts are at least 5 MB and
equal in size except the last, and the docs recommend multipart for files over 100 MB.

A name is only ever published after its object is uploaded, so no read is ever made of
a missing object, and downloads need no cache-busting: they are ordinary cacheable CDN reads.
The token stays in site/.env.local, read by node.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sfzmaps import answers, cachepack, config, datapack

BASE = (os.environ.get("NEXT_PUBLIC_TILES_BASE_URL")
        or "https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com").rstrip("/")
DVC_PREFIX = "dvc/files/md5"
PACK_PREFIX = "cache/packs"
NEEDED_PREFIX = "cache/needed"
SHIPPED_PREFIX = "cache/shipped"
DATAPACK_PREFIX = "cache/datapacks"
MAX_REQUESTS = 500                   # a ship making more than this needs --yes
MULTIPART_MIN = 100_000_000          # blob-put.mjs: larger files go up multipart (the SDK's) ...
PART_BYTES = 8 * 1024 * 1024         # ... in the SDK's 8 MB parts (keep both equal to blob-put.mjs)
PULL_WORKERS = 4                     # downloads at once, data packs and single objects alike
RETRY_STATUS = {403, 429, 500, 502, 503, 504}
RETRY_WAIT_S = (2, 8, 30)


class ShipError(Exception):
    pass


# --- public reads -------------------------------------------------------------------------------

def _client():
    import httpx
    return httpx.Client(timeout=120, follow_redirects=True)


def download(name: str, dest: Path, alg: str, want: str, client=None, wait=time.sleep) -> None:
    """Stream a published object to dest, atomically, if its hash (alg) is `want`; else
    ShipError and nothing is written. A refusal or overload (403, 429, 5xx) waits and retries a
    few times, slower each time, then gives up."""
    c = client or _client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        for attempt, pause in enumerate((*RETRY_WAIT_S, None)):
            h = hashlib.new(alg)
            with c.stream("GET", f"{BASE}/{name}") as r:
                if r.status_code == 404:
                    raise ShipError(f"{name} is not on the Blob store (was this commit shipped?)")
                if r.status_code in RETRY_STATUS and pause is not None:
                    wait(pause)
                    continue
                if r.status_code in RETRY_STATUS:
                    raise ShipError(f"{name}: the Blob store keeps refusing ({r.status_code}); "
                                    "try again later")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        h.update(chunk)
                        f.write(chunk)
            break
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


# --- the store's listing and uploads (node, @vercel/blob, token in site/.env.local) ----------------

def _node(script: str, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run a site script; node reads the token from site/.env.local, or from the file
    SFZMAPS_ENV_FILE names (e.g. the main checkout's, from a worktree that has none)."""
    site = _repo_root() / "site"
    env_file = Path(os.environ.get("SFZMAPS_ENV_FILE") or site / ".env.local")
    if not env_file.exists():
        raise ShipError(f"{env_file} is missing (BLOB_READ_WRITE_TOKEN, from `vercel env pull`); "
                        "SFZMAPS_ENV_FILE may name another")
    return subprocess.run(["node", f"--env-file={env_file}", f"scripts/{script}", *args], cwd=site,
                          input=stdin, capture_output=True, text=True)


def listing(prefixes=(f"{DVC_PREFIX}/", "cache/")) -> dict[str, int]:
    """name -> size of everything the store holds under prefixes: one list call per 1,000 names."""
    p = _node("blob-list.mjs", list(prefixes))
    if p.returncode != 0:
        raise ShipError(f"listing the Blob store failed: {p.stderr.strip()[-300:]}")
    out = {}
    for line in p.stdout.splitlines():
        r = json.loads(line)
        out[r["name"]] = r["size"]
    return out


def upload(entries: list[dict]) -> dict[str, str]:
    """Upload manifest entries with site/scripts/blob-put.mjs; name -> status. The token is read
    by node from site/.env.local; it never passes through here."""
    if not entries:
        return {}
    p = _node("blob-put.mjs", [], json.dumps(entries))
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
    """pipeline/cache-needed.json: which dvc.lock was shipped, the published list of the answers
    that commit needs (cache/needed/<sha>.keys.gz) and its shipped manifest
    (cache/shipped/<sha>.json: every object and pack confirmed on the store)."""
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
    packs_new: int = 0                # data packs built for objects nothing placed yet
    singles_uploaded: int = 0         # large objects uploaded alone
    requests: int = 0                 # requests to the store, listings included


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


def _list_calls(held: dict[str, int]) -> int:
    """Requests one listing makes: one per 1,000 names under each prefix."""
    dvc = sum(n.startswith(f"{DVC_PREFIX}/") for n in held)
    return max(1, math.ceil(dvc / 1000)) + max(1, math.ceil((len(held) - dvc) / 1000))


def put_requests(nbytes: int) -> int:
    """Requests blob-put.mjs makes to upload a file of this size."""
    return 1 if nbytes <= MULTIPART_MIN else 2 + math.ceil(nbytes / PART_BYTES)


def datapack_name(sha: str) -> str:
    return f"{DATAPACK_PREFIX}/{sha}.zip"


def _manifest(sha: str, client) -> dict:
    """A shipped manifest by its sha256, kept locally once fetched (it never changes)."""
    p = config.CACHE_DIR / "shipped" / f"{sha}.json"
    if not p.exists():
        download(f"{SHIPPED_PREFIX}/{sha}.json", p, "sha256", sha, client)
    return json.loads(p.read_text())


def prior_manifests(base: str, client) -> list[dict]:
    """The v2 manifests whose data packs this ship may reuse: this branch's own (from an earlier
    ship of it), then each one cache-needed.json named on `base`, newest first, back to the
    first v1 or missing manifest."""
    out = []
    if needed_path().exists():
        sha = json.loads(needed_path().read_text()).get("shipped_sha")
        try:
            m = _manifest(sha, client) if sha else {}
        except ShipError:
            m = {}
        if m.get("version") == 2:
            out.append(m)
    try:
        commits = _git("log", "--format=%H", base, "--", "pipeline/cache-needed.json").split()
    except subprocess.CalledProcessError:
        commits = []
    seen = set()
    for c in commits:
        try:
            sha = json.loads(_git("show", f"{c}:pipeline/cache-needed.json")).get("shipped_sha")
        except (subprocess.CalledProcessError, ValueError):
            continue
        if not sha or sha in seen:
            continue
        seen.add(sha)
        try:
            m = _manifest(sha, client)
        except ShipError:
            break
        if m.get("version") != 2:
            break
        out.append(m)
    return out


def ship(base: str = "origin/main", dry_run: bool = False, yes: bool = False,
         repack: bool = False) -> ShipReport:
    """Upload what this commit needs that the store lacks, confirm it, publish the shipped
    manifest, and commit the records (pack record, pipeline/cache-needed.json).

    A data object stays in the data pack an earlier manifest (this branch's, or one on `base`)
    put it in, or alone if it's already stored alone; the rest go into new data packs (under
    datapack.SINGLE_MIN) or alone. More than MAX_REQUESTS requests need `yes`.

    `repack` also packs the small objects already stored alone (a one-time move from v1, where
    every object was stored alone); those single copies stay on the store."""
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
    size = {m: object_path(m, cache).stat().st_size for m in objs}
    held = listing()
    wrong = [m for m in objs if object_name(m) in held and held[object_name(m)] != size[m]]
    if wrong:
        raise ShipError(f"{len(wrong)} object(s) on the store differ in size from the local ones "
                        f"(e.g. {object_name(wrong[0])}); not overwriting")
    where: dict[str, str] = {}                            # md5 -> data pack sha
    data_packs: dict[str, int] = {}                       # data pack sha -> bytes
    for prior in prior_manifests(base, client):
        for m, sha in (prior.get("where") or {}).items():
            pbytes = (prior.get("data_packs") or {}).get(sha)
            if m in size and m not in where and pbytes is not None \
                    and held.get(datapack_name(sha)) == pbytes:
                where[m] = sha
                data_packs[sha] = pbytes
    missing = [m for m in objs if m not in where and (object_name(m) not in held
                                                       or (repack and size[m] < datapack.SINGLE_MIN))]
    packs, singles = datapack.plan_packs({m: size[m] for m in missing})
    nbytes = sum(size[m] for m in missing)
    nb = needed_bytes(keys)
    keys_sha = hashlib.sha256(nb).hexdigest()
    # a pack's zip adds about 100 bytes per member to the objects' own bytes
    requests = (2 * _list_calls(held)
                + sum(put_requests(sum(size[m] + 100 for m in p)) for p in packs)
                + sum(put_requests(size[m]) for m in singles) + (2 if new else 0) + 1
                + (f"{NEEDED_PREFIX}/{keys_sha}.keys.gz" not in held))

    def report(pack_sha=None) -> ShipReport:
        return ShipReport(pack_sha, len(new), len(objs), len(missing), nbytes, len(packs),
                          len(singles), requests)

    if dry_run:
        return report()
    if requests > MAX_REQUESTS and not yes:
        raise ShipError(f"this ship would make {requests} requests to the store (more than "
                        f"{MAX_REQUESTS}); check with --dry-run, then pass --yes")
    pack_sha = None
    entries = [{"file": str(object_path(m, cache)), "name": object_name(m), "alg": "md5",
                "hash": _md5_hex(m)} for m in singles]
    with tempfile.TemporaryDirectory(prefix="sfzmaps-ship-") as tmp:
        for i, members in enumerate(packs):
            dp = Path(tmp) / f"data-{i}.zip"
            sha = datapack.make_pack(members, cache, dp)
            data_packs[sha] = dp.stat().st_size
            where.update({m: sha for m in members})
            if held.get(datapack_name(sha)) != data_packs[sha]:   # an interrupted ship may have sent it
                entries.append({"file": str(dp), "name": datapack_name(sha), "alg": "sha256",
                                "hash": sha})
        np_ = Path(tmp) / "needed.keys.gz"
        np_.write_bytes(nb)
        if f"{NEEDED_PREFIX}/{keys_sha}.keys.gz" not in held:
            entries.append({"file": str(np_), "name": f"{NEEDED_PREFIX}/{keys_sha}.keys.gz",
                            "alg": "sha256", "hash": keys_sha})
        if new:
            pack = cachepack.make_pack(new)
            zp, ip = Path(tmp) / f"{pack.sha}.zip", Path(tmp) / f"{pack.sha}.keys.gz"
            zp.write_bytes(pack.data)
            ip.write_bytes(pack.index_bytes())
            index_sha = hashlib.sha256(pack.index_bytes()).hexdigest()
            for f, name, h in ((zp, f"{PACK_PREFIX}/{pack.sha}.zip", pack.sha),
                               (ip, f"{PACK_PREFIX}/{pack.sha}.keys.gz", index_sha)):
                if held.get(name) != f.stat().st_size:     # an interrupted ship may have sent it
                    entries.append({"file": str(f), "name": name, "alg": "sha256", "hash": h})
            pack_sha = pack.sha
            records = {**records, pack.sha: pack.record()}
        upload(entries)                                   # only what the listing lacked ...
        held = listing()                                  # ... then one more listing confirms
        absent = [datapack_name(where[m]) for m in objs
                  if m in where and held.get(datapack_name(where[m])) != data_packs[where[m]]]
        absent += [object_name(m) for m in objs
                   if m not in where and held.get(object_name(m)) != size[m]]
        absent += [f"{PACK_PREFIX}/{sha}.zip" for sha, r in records.items()
                   if held.get(f"{PACK_PREFIX}/{sha}.zip") != r["bytes"]]
        if f"{NEEDED_PREFIX}/{keys_sha}.keys.gz" not in held:
            absent.append(f"{NEEDED_PREFIX}/{keys_sha}.keys.gz")
        if absent:
            raise ShipError(f"{len(absent)} object(s) aren't on the store after uploading "
                            f"(e.g. {absent[0]}); run ship again")
        placed = {m: where[m] for m in objs if m in where}
        manifest = json.dumps({"version": 2, "lock_md5": lock_md5(), "outs": outs,
                               "objects": {m: size[m] for m in objs},
                               "where": placed,
                               "data_packs": {sha: data_packs[sha] for sha in sorted(set(placed.values()))},
                               "packs": {sha: r["bytes"] for sha, r in sorted(records.items())}},
                              indent=0, sort_keys=True).encode()
        shipped_sha = hashlib.sha256(manifest).hexdigest()
        if f"{SHIPPED_PREFIX}/{shipped_sha}.json" not in held:
            mp = Path(tmp) / "shipped.json"
            mp.write_bytes(manifest)
            upload([{"file": str(mp), "name": f"{SHIPPED_PREFIX}/{shipped_sha}.json",
                     "alg": "sha256", "hash": shipped_sha, "contentType": "application/json"}])
        files = []                                        # the records last
        if pack_sha:
            files.append(cachepack.write_record(pack))
            cachepack.mark_applied(pack_sha)
        needed = {"lock_md5": lock_md5(), "keys_sha": keys_sha, "keys": len(keys),
                  "shipped_sha": shipped_sha}
        new_text = json.dumps(needed, indent=1, sort_keys=True) + "\n"
        if not needed_path().exists() or needed_path().read_text() != new_text:
            needed_path().write_text(new_text)
            files.append(needed_path())
        if files:
            _git("add", *map(str, files))
            _git("commit", "-q", "-m", f"cache: ship {len(objs)} data objects, {len(keys)} answers"
                 + (f" (new pack {pack_sha[:12]}: {len(new)})" if pack_sha else ""))
    return report(pack_sha)


def shipped_manifest(client=None) -> dict:
    """This commit's shipped manifest (named in cache-needed.json, sha-checked)."""
    if not needed_path().exists():
        raise ShipError("pipeline/cache-needed.json is missing: this commit wasn't shipped")
    sha = json.loads(needed_path().read_text()).get("shipped_sha")
    if not sha:
        raise ShipError("pipeline/cache-needed.json names no shipped manifest: ship again")
    with tempfile.TemporaryDirectory(prefix="sfzmaps-manifest-") as tmp:
        p = Path(tmp) / "m.json"
        download(f"{SHIPPED_PREFIX}/{sha}.json", p, "sha256", sha, client)
        return json.loads(p.read_text())


# --- pull -----------------------------------------------------------------------------------------

def pull(dvc_checkout: bool = True) -> tuple[int, int]:
    """Fetch this commit's packs and data from the Blob store; (answer packs applied, objects
    fetched). Reads the shipped manifest, then fetches only what's missing locally: with a v2
    manifest, each data pack holding a missing object once, then the objects stored alone, both
    PULL_WORKERS at a time. A v1 manifest stores every object alone."""
    client = _client()
    manifest = shipped_manifest(client)
    if manifest["lock_md5"] != lock_md5():
        raise ShipError("this checkout's dvc.lock isn't the one shipped; pull on a shipped commit")
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
    todo = [m for m in manifest["objects"] if not object_path(m, cache).exists()]
    where = (manifest.get("where") or {}) if manifest.get("version") == 2 else {}   # v1: all alone
    by_pack: dict[str, set[str]] = {}
    for m in todo:
        if m in where:
            by_pack.setdefault(where[m], set()).add(m)

    def fetch_pack(sha: str) -> None:
        with tempfile.TemporaryDirectory(prefix="sfzmaps-pull-") as tmp:
            zp = Path(tmp) / f"{sha}.zip"
            download(datapack_name(sha), zp, "sha256", sha, client)
            try:
                datapack.apply_pack(zp, cache, by_pack[sha])
            except datapack.DataPackError as e:
                raise ShipError(str(e)) from None

    def fetch(md5: str) -> None:
        download(object_name(md5), object_path(md5, cache), "md5", _md5_hex(md5), client)

    with ThreadPoolExecutor(PULL_WORKERS) as ex:
        list(ex.map(fetch_pack, sorted(by_pack)))
    lost = sorted(m for ms in by_pack.values() for m in ms if not object_path(m, cache).exists())
    if lost:
        raise ShipError(f"{len(lost)} object(s) weren't in the data pack the manifest names "
                        f"(e.g. {lost[0]})")
    with ThreadPoolExecutor(PULL_WORKERS) as ex:
        list(ex.map(fetch, [m for m in todo if m not in where]))
    if dvc_checkout:
        subprocess.run(["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc", "checkout"],
                       cwd=config.PIPELINE_DIR, check=True)
    return applied_n, len(todo)


# --- the merge gate (CI) ---------------------------------------------------------------------------

DATA_DEPS = ("data/",)   # deps CI doesn't have (scans, reference data, outputs): not checked here


def lock_problems(status_json: str) -> list[str]:
    """From `dvc status --json` on a checkout without data: stages whose git-tracked deps or
    command changed since dvc.lock was recorded (a commit added after the build, or a rebase
    without a rebuild). Missing data deps and outputs are expected in CI and ignored."""
    out = []
    for stage, changes in (json.loads(status_json or "{}") or {}).items():
        if stage.endswith(".dvc"):
            continue
        for ch in changes:
            if ch == "changed command":
                out.append(f"{stage}: its command changed since dvc.lock was recorded")
            elif isinstance(ch, dict):
                for path, how in (ch.get("changed deps") or {}).items():
                    if not path.startswith(DATA_DEPS):
                        out.append(f"{stage}: {path} {how} since dvc.lock was recorded")
    return out


def check_shipped(status_json: str, client=None) -> list[str]:
    """Everything this commit needs is on the Blob store and dvc.lock matches the code: the list
    of problems (empty = mergeable). Public reads only, and few: the shipped manifest, the needed
    list and the pack indexes (ship confirmed every object with the store's listing)."""
    client = client or _client()
    problems = lock_problems(status_json)
    if not needed_path().exists():
        return problems + ["pipeline/cache-needed.json is missing: run `sfzmaps ship`"]
    needed = json.loads(needed_path().read_text())
    if needed.get("lock_md5") != lock_md5():
        return problems + ["dvc.lock changed since `sfzmaps ship` ran: build, then ship again"]
    try:
        manifest = shipped_manifest(client)
    except ShipError as e:
        return problems + [str(e)]
    if manifest.get("lock_md5") != lock_md5() or manifest.get("outs") != lock_outs(
            (config.PIPELINE_DIR / "dvc.lock").read_text()):
        problems.append("the shipped manifest doesn't describe this dvc.lock: ship again")
    if not set(manifest.get("outs", [])) <= set(manifest.get("objects", {})):
        problems.append("the shipped manifest lacks some of dvc.lock's outputs: ship again")
    if manifest.get("version") == 2:
        # Each object is in a data pack whose size ship confirmed by listing, or stored alone
        # (also confirmed by listing). v1 manifests store every object alone.
        where, dpacks = manifest.get("where") or {}, manifest.get("data_packs") or {}
        stray = sorted(set(where) - set(manifest.get("objects", {})))
        if stray:
            problems.append(f"the shipped manifest places {len(stray)} object(s) it doesn't list "
                            f"(e.g. {stray[0]}): ship again")
        unlisted = sorted({sha for sha in where.values() if sha not in dpacks})
        unlisted += sorted(sha for sha, n in dpacks.items()
                           if not isinstance(n, int) or isinstance(n, bool) or n <= 0)
        if unlisted:
            problems.append(f"data pack {unlisted[0][:12]} wasn't confirmed on the store by "
                            "`sfzmaps ship`: ship again")
    elif manifest.get("version", 1) != 1:
        problems.append(f"the shipped manifest's version {manifest.get('version')!r} is unknown")
    records = cachepack.recorded()
    for sha, rec in sorted(records.items()):
        if manifest.get("packs", {}).get(sha) != rec["bytes"]:
            problems.append(f"answer pack {sha[:12]} wasn't confirmed on the store by `sfzmaps ship`")
    with tempfile.TemporaryDirectory(prefix="sfzmaps-check-") as tmp:
        try:
            have = known_keys(records, client)
            np_ = Path(tmp) / "needed.keys.gz"
            download(f"{NEEDED_PREFIX}/{needed['keys_sha']}.keys.gz", np_, "sha256", needed["keys_sha"], client)
            import gzip
            keys = gzip.decompress(np_.read_bytes()).decode().split()
            gap = [k for k in answers.validate(keys) if k not in have]
            if gap:
                problems.append(f"{len(gap)} needed answer(s) are in no shipped pack (e.g. {gap[0]})")
        except ShipError as e:
            problems.append(str(e))
    return problems
