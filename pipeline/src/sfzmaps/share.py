"""Share processed (generated, expensive or manual) files via the Drive folder.

`push` bundles them into deterministic zips under DATA_DIR/share and uploads with rclone;
`pull` fetches the public `processed` folder listing and extracts into DATA_DIR.
"""
import fnmatch
import hashlib
import httpx
import json
import logging
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sfzmaps import config
from sfzmaps.sources import drive

log = logging.getLogger(__name__)

SHARED_GLOBS = [
    "cache/llm/**",
    "ref/*.parquet",
    "work/label_scale.json",
    "work/*/*/page.json",
    "work/*/*/labels.json",
    "work/*/*/transform.json",
    "work/*/*/changes.geojson",
    "work/*/*/score.json",
    "work/*/*/review.json",
    "work/*/*/refine.json",
    "work/*/*/rough.json",
    "out/*/*.pmtiles",
    "out/*/manifest.json",
    "out/*/review.html",
    "out/ref/blocks.pmtiles",
]
# Never shared, whatever the globs match (credentials).
_NEVER = ("*.env", ".env", "*key*", "*secret*")
_FIXED_DATE = (2020, 1, 1, 0, 0, 0)
DEFAULT_REMOTE = "sfzoning-drive:old-planning-maps/processed"
SETUP = (
    "rclone is not set up. Run: brew install rclone, then `rclone config`: create a new remote "
    'named sfzoning-drive, type Google Drive, scope "drive" (full access), then rerun.'
)


def _excluded_rel(rel: str) -> bool:
    """True if any path component matches a never-bundle pattern."""
    return any(
        fnmatch.fnmatch(part.lower(), pat) for part in rel.split("/") for pat in _NEVER
    )


def shared_files(data_dir: Path | None = None) -> list[str]:
    """Relative posix paths of every shared file under data_dir, sorted."""
    root = data_dir or config.DATA_DIR
    real_root = root.resolve()
    found: set[str] = set()
    for pat in SHARED_GLOBS:
        # a trailing "/**" means every file below (Path.glob's "**" alone yields dirs only)
        for p in root.glob(pat + "/*" if pat.endswith("/**") else pat):
            if p.is_symlink() or not p.is_file():
                continue
            if real_root not in p.resolve().parents:
                continue
            rel = p.relative_to(root).as_posix()
            if not _excluded_rel(rel):
                found.add(rel)
    return sorted(found)


def _group(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "cache" or rel == "work/label_scale.json":
        return "cache"
    if parts[0] == "ref" or rel.startswith("out/ref/"):
        return "ref"
    return parts[1]  # work/<series>/... or out/<series>/...


def _write_zip(path: Path, members: list[tuple[str, "bytes | Path"]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, src in sorted(members, key=lambda m: m[0]):
            zi = zipfile.ZipInfo(name, _FIXED_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, src if isinstance(src, bytes) else src.read_bytes())
    tmp.replace(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_counts(paths: list[Path]) -> dict[str, int]:
    try:
        import pyarrow.parquet as pq

        return {p.name: pq.ParquetFile(p).metadata.num_rows for p in paths}
    except Exception as e:  # not a real parquet file / pyarrow missing
        log.warning("could not count parquet rows (%s); recording -1", e)
        return {p.name: -1 for p in paths}


def bundle(dest: Path) -> list[Path]:
    """Write processed-<series>.zip, processed-cache.zip, processed-ref.zip, PROCESSED.json."""
    root = config.DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    for leftover in dest.glob("*.tmp"):
        leftover.unlink()
    groups: dict[str, list[tuple[str, "bytes | Path"]]] = {}
    for rel in shared_files(root):
        groups.setdefault(_group(rel), []).append((rel, root / rel))
    parquets = [p for n, p in groups.get("ref", []) if n.endswith(".parquet")]
    if parquets:
        psha = hashlib.sha256(
            "".join(f"{p.name}:{_sha(p)}" for p in sorted(parquets)).encode()
        ).hexdigest()
        fetched = datetime.fromtimestamp(
            max(p.stat().st_mtime for p in parquets), timezone.utc
        ).strftime("%Y-%m-%d")
        try:  # unchanged parquets keep the previous fetch date so the zip stays identical
            with zipfile.ZipFile(dest / "processed-ref.zip") as oz:
                prev_snap = json.loads(oz.read("ref/SNAPSHOT.json"))
            if prev_snap.get("parquet_sha256") == psha:
                fetched = prev_snap["fetched"]
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            pass
        snap = {"fetched": fetched, "parquet_sha256": psha, "rows": _row_counts(parquets)}
        groups["ref"].append(
            ("ref/SNAPSHOT.json", json.dumps(snap, indent=1, sort_keys=True).encode())
        )
    try:
        old = json.loads((dest / "PROCESSED.json").read_text())
    except (OSError, ValueError):
        old = {}
    now = datetime.now(timezone.utc)
    meta, zips = {}, []
    for g in sorted(groups):
        z = dest / f"processed-{g}.zip"
        _write_zip(z, groups[g])
        sha = _sha(z)
        prev = old.get(z.name, {})
        keep = prev.get("sha256") == sha and "created_at_epoch" in prev
        meta[z.name] = {
            "sha256": sha,
            "size": z.stat().st_size,
            "created_at": prev["created_at"] if keep else now.isoformat(timespec="seconds"),
            "created_at_epoch": prev["created_at_epoch"] if keep else int(now.timestamp()),
        }
        zips.append(z)
    for stale in dest.glob("processed-*.zip"):
        if stale.name not in meta:
            stale.unlink()
    (dest / "PROCESSED.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return zips


PULLED_SHAS = "work/.pulled_shas.json"


def _load_pulled(root: Path) -> dict[str, str]:
    try:
        d = json.loads((root / PULLED_SHAS).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}  # missing/corrupt sidecar: every local review.json counts as edited


def _extract(zpath: Path, root: Path) -> None:
    """Extract zpath into root. A review.json edited locally since the last pull is never
    overwritten: the incoming copy goes to review.json.pulled instead."""
    root = root.resolve()
    pulled = _load_pulled(root)
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        for info in infos:  # validate everything before writing anything
            target = (root / info.filename).resolve()
            if root not in target.parents or _excluded_rel(info.filename):
                raise ValueError(f"unsafe path in {zpath.name}: {info.filename}")
        for info in infos:
            target = root / info.filename
            data = z.read(info)
            if target.name == "review.json":
                new_sha = hashlib.sha256(data).hexdigest()
                if target.exists():
                    local_sha = _sha(target)
                    if local_sha != new_sha and local_sha != pulled.get(info.filename):
                        side = target.with_name("review.json.pulled")
                        side.write_bytes(data)
                        log.warning(
                            "sheet %s: local review.json was edited; kept it, incoming copy "
                            "saved as %s", target.parent.name, side,
                        )
                        continue
                pulled[info.filename] = new_sha
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    sidecar = root / PULLED_SHAS
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(pulled, indent=1, sort_keys=True))


def pull(
    folder_id: str = config.ROOT_FOLDER_ID,
    lister: Callable[[str], list[drive.DriveEntry]] = drive.list_folder,
    downloader: Callable[[str, Path], None] = drive.download,
) -> None:
    """Download changed processed zips from the public Drive folder and extract them."""
    errs = (httpx.HTTPError, OSError, ValueError, zipfile.BadZipFile)

    def fail(what: str, e: Exception):
        return SystemExit(f"could not {what}: {type(e).__name__}: {str(e)[:120]}")

    try:
        folders = [e for e in lister(folder_id) if e.is_folder and e.title == "processed"]
    except errs as e:
        raise fail("list the Drive folder", e) from None
    if not folders:
        raise SystemExit("no 'processed' folder found in the Drive folder (has anyone pushed?)")
    try:
        entries = {e.title: e for e in lister(folders[0].id) if not e.is_folder}
    except errs as e:
        raise fail("list the processed folder", e) from None
    if "PROCESSED.json" not in entries:
        raise SystemExit("processed/PROCESSED.json is missing from the Drive folder")
    share_dir = config.DATA_DIR / "share"
    share_dir.mkdir(parents=True, exist_ok=True)
    meta_path = share_dir / "PROCESSED.json"
    try:
        downloader(entries["PROCESSED.json"].id, meta_path)
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            raise ValueError("manifest is not a JSON object")
    except errs as e:
        raise fail("download PROCESSED.json", e) from None
    for name, info in sorted(meta.items()):
        if not name.endswith(".zip") or "/" in name:
            continue
        local = share_dir / name
        try:
            if local.exists() and _sha(local) == info["sha256"]:
                continue
            if name not in entries:
                log.warning("%s listed in PROCESSED.json but not in the folder", name)
                continue
            downloader(entries[name].id, local)
            if _sha(local) != info["sha256"]:
                local.unlink()
                raise ValueError("sha256 mismatch after download")
            _extract(local, config.DATA_DIR)
        except (KeyError, TypeError) as e:
            raise fail(f"read manifest entry {name}", e) from None
        except errs as e:
            raise fail(f"fetch or extract {name}", e) from None
        log.info("extracted %s", name)


def push(remote: str = DEFAULT_REMOTE) -> None:
    """Bundle into DATA_DIR/share and upload with `rclone copy --checksum`."""
    if shutil.which("rclone") is None:
        raise SystemExit(SETUP)
    name = remote.split(":", 1)[0] + ":"
    r = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if r.returncode != 0 or name not in r.stdout.split():
        raise SystemExit(SETUP)
    dest = config.DATA_DIR / "share"
    bundle(dest)
    r = subprocess.run(["rclone", "copy", "--checksum", str(dest), remote], text=True)
    if r.returncode != 0:
        raise SystemExit(f"rclone copy failed (exit {r.returncode})")
