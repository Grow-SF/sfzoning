import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from sfzmaps import config, share
from sfzmaps.sources.drive import DriveEntry


def _w(root: Path, rel: str, data: bytes = b"x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


SHARED = [
    "cache/llm/ab/abcdef.json",
    "ref/parcels.parquet",
    "work/label_scale.json",
    "work/1948/p1/page.json",
    "work/1948/p1/labels.json",
    "work/1948/p1/transform.json",
    "work/1948/p1/changes.geojson",
    "work/1948/p1/score.json",
    "work/1948/p1/review.json",
    "work/1948/p1/refine.json",
    "work/1948/p1/rough.json",
    "work/1960/p2/score.json",
    "out/1948/tiles.pmtiles",
    "out/1948/manifest.json",
    "out/1948/review.html",
    "out/ref/blocks.pmtiles",
]
EXCLUDED = [
    "raw/1948/scan.tif",
    "work/1948/p1/page.png",
    "work/1948/p1/warp.tif",
    "out/1948/mosaic.tif",
    "out/1948/preview.png",
    ".env",
    "cache/llm/secret.env",
    "cache/llm/my_api_key.json",
    "cache/llm/client_secret.json",
    "work/1948/p1/token.env",
]


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", d)
    for rel in SHARED + EXCLUDED:
        _w(d, rel, rel.encode())
    return d


def _members(dest: Path) -> set[str]:
    names = set()
    for z in dest.glob("processed-*.zip"):
        names |= set(zipfile.ZipFile(z).namelist())
    return names


def test_bundle_includes_only_shared(data, tmp_path):
    dest = tmp_path / "share"
    zips = share.bundle(dest)
    assert {z.name for z in zips} == {
        "processed-1948.zip", "processed-1960.zip",
        "processed-cache.zip", "processed-ref.zip",
    }
    assert _members(dest) == set(SHARED) | {"ref/SNAPSHOT.json"}
    assert json.loads(zipfile.ZipFile(dest / "processed-ref.zip").read("ref/SNAPSHOT.json"))
    meta = json.loads((dest / "PROCESSED.json").read_text())
    z = dest / "processed-1948.zip"
    assert meta["processed-1948.zip"]["sha256"] == hashlib.sha256(z.read_bytes()).hexdigest()
    assert meta["processed-1948.zip"]["size"] == z.stat().st_size


def test_env_file_never_bundled(data, tmp_path):
    _w(data, ".env", b"OPENAI_API_KEY=sk-secret")
    _w(data, "cache/llm/.env", b"SECRET")
    _w(data, "work/1948/p1/.env", b"SECRET")
    share.bundle(tmp_path / "share")
    for z in (tmp_path / "share").glob("*.zip"):
        for n in zipfile.ZipFile(z).namelist():
            assert not n.endswith(".env")
        assert b"sk-secret" not in z.read_bytes()


def test_bundle_deterministic(data, tmp_path):
    a = share.bundle(tmp_path / "a")
    os.utime(data / "work/1948/p1/page.json", (1, 1))  # mtimes must not matter
    b = share.bundle(tmp_path / "b")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert [sha(p) for p in a] == [sha(p) for p in b]


def test_pull_keeps_newer_local_review(data, tmp_path):
    src = tmp_path / "src"
    share.bundle(src)
    meta = json.loads((src / "PROCESSED.json").read_text())
    rev = data / "work/1948/p1/review.json"
    rev.write_text("LOCAL EDIT")
    future = meta["processed-1948.zip"]["created_at_epoch"] + 1000
    os.utime(rev, (future, future))
    (data / "work/1948/p1/score.json").unlink()

    files = {p.name: DriveEntry("id-" + p.name, p.name, False) for p in src.iterdir()}

    def lister(fid):
        return [DriveEntry("F", "processed", True)] if fid == "ROOT" else list(files.values())

    def downloader(fid, dest):
        dest.write_bytes((src / fid[3:]).read_bytes())

    share.pull("ROOT", lister=lister, downloader=downloader)
    assert rev.read_text() == "LOCAL EDIT"
    assert (data / "work/1948/p1/score.json").exists()  # restored from zip

    calls = []

    def counting(f, d):
        calls.append(f)
        downloader(f, d)

    share.pull("ROOT", lister=lister, downloader=counting)  # shas match: only the manifest
    assert calls == ["id-PROCESSED.json"]


def test_push_missing_rclone_message(monkeypatch, data):
    monkeypatch.setattr(share.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as e:
        share.push()
    msg = str(e.value)
    assert "brew install rclone" in msg and "sfzoning-drive" in msg and "rclone config" in msg


def test_push_missing_remote_message(monkeypatch, data):
    monkeypatch.setattr(share.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(
        share.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="other:\n", stderr=""),
    )
    with pytest.raises(SystemExit) as e:
        share.push()
    assert "sfzoning-drive" in str(e.value) and "Google Drive" in str(e.value)


def test_push_runs_rclone_copy(monkeypatch, data):
    monkeypatch.setattr(share.shutil, "which", lambda name: "/usr/bin/rclone")
    ran = []

    def run(cmd, **k):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="sfzoning-drive:\n", stderr="")

    monkeypatch.setattr(share.subprocess, "run", run)
    share.push()
    copy = [c for c in ran if "copy" in c][0]
    assert "--checksum" in copy and copy[-1] == "sfzoning-drive:old-planning-maps/processed"
    assert (data / "share" / "PROCESSED.json").exists()
