"""Share processed (generated, expensive or manual) files via the Drive folder.

`push` bundles them into deterministic zips under DATA_DIR/share and uploads with rclone;
`pull` fetches the public `processed` folder listing and extracts into DATA_DIR.
"""
import fnmatch
import hashlib
import json
import logging
import shutil
import subprocess
import time
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


def _excluded(path: Path) -> bool:
    n = path.name.lower()
    return any(fnmatch.fnmatch(n, pat) for pat in _NEVER)


def shared_files(data_dir: Path | None = None) -> list[str]:
    """Relative posix paths of every shared file under data_dir, sorted."""
    root = data_dir or config.DATA_DIR
    found: set[str] = set()
    for pat in SHARED_GLOBS:
        # a trailing "/**" means every file below (Path.glob's "**" alone yields dirs only)
        for p in root.glob(pat + "/*" if pat.endswith("/**") else pat):
            if p.is_file() and not _excluded(p):
                found.add(p.relative_to(root).as_posix())
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
    except Exception:  # not a real parquet file / pyarrow missing
        return {p.name: -1 for p in paths}


def bundle(dest: Path) -> list[Path]:
    """Write processed-<series>.zip, processed-cache.zip, processed-ref.zip, PROCESSED.json."""
    root = config.DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[str, "bytes | Path"]]] = {}
    for rel in shared_files(root):
        groups.setdefault(_group(rel), []).append((rel, root / rel))
    parquets = [p for n, p in groups.get("ref", []) if n.endswith(".parquet")]
    if parquets:
        snap = {
            "fetched": datetime.fromtimestamp(
                max(p.stat().st_mtime for p in parquets), timezone.utc
            ).strftime("%Y-%m-%d"),
            "rows": _row_counts(parquets),
        }
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


def _extract(zpath: Path, root: Path, created_epoch: float) -> None:
    root = root.resolve()
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            target = (root / info.filename).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe path in {zpath.name}: {info.filename}")
            if (
                target.name == "review.json"
                and target.exists()
                and target.stat().st_mtime > created_epoch
            ):
                log.warning("keeping newer local %s (not overwritten)", target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))


def pull(
    folder_id: str = config.ROOT_FOLDER_ID,
    lister: Callable[[str], list[drive.DriveEntry]] = drive.list_folder,
    downloader: Callable[[str, Path], None] = drive.download,
) -> None:
    """Download changed processed zips from the public Drive folder and extract them."""
    folders = [e for e in lister(folder_id) if e.is_folder and e.title == "processed"]
    if not folders:
        raise SystemExit("no 'processed' folder found in the Drive folder (has anyone pushed?)")
    entries = {e.title: e for e in lister(folders[0].id) if not e.is_folder}
    if "PROCESSED.json" not in entries:
        raise SystemExit("processed/PROCESSED.json is missing from the Drive folder")
    share_dir = config.DATA_DIR / "share"
    share_dir.mkdir(parents=True, exist_ok=True)
    meta_path = share_dir / "PROCESSED.json"
    downloader(entries["PROCESSED.json"].id, meta_path)
    meta = json.loads(meta_path.read_text())
    for name, info in sorted(meta.items()):
        local = share_dir / name
        if local.exists() and _sha(local) == info["sha256"]:
            continue
        if name not in entries:
            log.warning("%s listed in PROCESSED.json but not in the folder", name)
            continue
        downloader(entries[name].id, local)
        if _sha(local) != info["sha256"]:
            local.unlink()
            raise SystemExit(f"{name}: sha256 mismatch after download; try again")
        _extract(local, config.DATA_DIR, info.get("created_at_epoch", time.time()))
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
