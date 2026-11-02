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
(public URLs, no token) and checks the data out.

Requests are kept to the fewest that get the data (a burst of per-object checks once tripped
Vercel's DDoS protection): ship learns what the store holds from its list API (one call per
1,000 names, site/scripts/blob-list.mjs), uploads only what's missing (one request each,
site/scripts/blob-put.mjs), lists once more to confirm, then publishes a *shipped manifest*
(cache/shipped/<sha>.json: every object and pack it confirmed, for this dvc.lock).
pipeline/cache-needed.json names it. The merge gate then reads two files and the pack indexes,
never one request per object; pull reads the manifest and fetches only what it lacks, a few at
a time. A name is only ever published after its object is uploaded, so no read is ever made of
a missing object, and downloads need no cache-busting: they are ordinary cacheable CDN reads.
The token stays in site/.env.local, read by node.
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
SHIPPED_PREFIX = "cache/shipped"
PULL_WORKERS = 4                     # downloads at once: a browser's tile load, not a burst
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
    site = _repo_root() / "site"
    if not (site / ".env.local").exists():
        raise ShipError("site/.env.local is missing (BLOB_READ_WRITE_TOKEN, from `vercel env pull`)")
    return subprocess.run(["node", "--env-file=.env.local", f"scripts/{script}", *args], cwd=site,
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
    """Upload what this commit needs that the store lacks, confirm it, publish the shipped
    manifest, and commit the records (pack record, pipeline/cache-needed.json)."""
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
    missing = [m for m in objs if object_name(m) not in held]
    nbytes = sum(size[m] for m in missing)
    if dry_run:
        return ShipReport(None, len(new), len(objs), len(missing), nbytes)
    pack_sha = None
    entries = [{"file": str(object_path(m, cache)), "name": object_name(m), "alg": "md5",
                "hash": _md5_hex(m)} for m in missing]
    nb = needed_bytes(keys)
    keys_sha = hashlib.sha256(nb).hexdigest()
    with tempfile.TemporaryDirectory(prefix="sfzmaps-ship-") as tmp:
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
        absent = [object_name(m) for m in objs if held.get(object_name(m)) != size[m]]
        absent += [f"{PACK_PREFIX}/{sha}.zip" for sha, r in records.items()
                   if held.get(f"{PACK_PREFIX}/{sha}.zip") != r["bytes"]]
        if f"{NEEDED_PREFIX}/{keys_sha}.keys.gz" not in held:
            absent.append(f"{NEEDED_PREFIX}/{keys_sha}.keys.gz")
        if absent:
            raise ShipError(f"{len(absent)} object(s) aren't on the store after uploading "
                            f"(e.g. {absent[0]}); run ship again")
        manifest = json.dumps({"lock_md5": lock_md5(), "outs": outs,
                               "objects": {m: size[m] for m in objs},
                               "packs": {sha: r["bytes"] for sha, r in sorted(records.items())}},
                              indent=0, sort_keys=True).encode()
        shipped_sha = hashlib.sha256(manifest).hexdigest()
        mp = Path(tmp) / "shipped.json"
        mp.write_bytes(manifest)
        upload([{"file": str(mp), "name": f"{SHIPPED_PREFIX}/{shipped_sha}.json", "alg": "sha256",
                 "hash": shipped_sha, "contentType": "application/json"}])
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
    return ShipReport(pack_sha, len(new), len(objs), len(missing), nbytes)


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
    """Fetch this commit's packs and data from the Blob store; (packs applied, objects fetched).
    Reads the shipped manifest, then fetches only what's missing locally, PULL_WORKERS at a time."""
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

    def fetch(md5: str) -> None:
        download(object_name(md5), object_path(md5, cache), "md5", _md5_hex(md5), client)

    with ThreadPoolExecutor(PULL_WORKERS) as ex:
        list(ex.map(fetch, todo))
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
