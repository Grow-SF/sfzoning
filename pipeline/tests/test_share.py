import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import httpx
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


def _remote(src: Path):
    files = {p.name: DriveEntry("id-" + p.name, p.name, False) for p in src.iterdir()}

    def lister(fid):
        return [DriveEntry("F", "processed", True)] if fid == "ROOT" else list(files.values())

    def downloader(fid, dest):
        dest.write_bytes((src / fid[3:]).read_bytes())

    return lister, downloader


REV = "work/1948/p1/review.json"


def _fresh_pull(data, tmp_path, monkeypatch):
    """Bundle `data` as the remote, then pull it into an empty DATA_DIR. Returns (new, remote)."""
    src = tmp_path / "src"
    share.bundle(src)
    new = tmp_path / "new"
    monkeypatch.setattr(config, "DATA_DIR", new)
    lister, downloader = _remote(src)
    share.pull("ROOT", lister=lister, downloader=downloader)
    return new, src


def _republish(data, src, rel, text, monkeypatch, new):
    (data / rel).write_text(text)
    monkeypatch.setattr(config, "DATA_DIR", data)
    share.bundle(src)
    monkeypatch.setattr(config, "DATA_DIR", new)
    return _remote(src)


def test_pull_keeps_local_edit_after_pull(data, tmp_path, monkeypatch):
    new, src = _fresh_pull(data, tmp_path, monkeypatch)
    assert (new / REV).read_bytes() == REV.encode()
    (new / REV).write_text("LOCAL EDIT")
    lister, downloader = _republish(data, src, REV, "REMOTE V2", monkeypatch, new)
    share.pull("ROOT", lister=lister, downloader=downloader)
    assert (new / REV).read_text() == "LOCAL EDIT"
    assert (new / (REV + ".pulled")).read_text() == "REMOTE V2"


def test_pull_updates_unedited_review(data, tmp_path, monkeypatch):
    new, src = _fresh_pull(data, tmp_path, monkeypatch)
    lister, downloader = _republish(data, src, REV, "REMOTE V2", monkeypatch, new)
    share.pull("ROOT", lister=lister, downloader=downloader)
    assert (new / REV).read_text() == "REMOTE V2"
    assert not (new / (REV + ".pulled")).exists()


def test_pull_missing_sidecar_treated_as_edited(data, tmp_path, monkeypatch):
    new, src = _fresh_pull(data, tmp_path, monkeypatch)
    (new / share.PULLED_SHAS).unlink()
    lister, downloader = _republish(data, src, REV, "REMOTE V2", monkeypatch, new)
    share.pull("ROOT", lister=lister, downloader=downloader)
    assert (new / REV).read_bytes() == REV.encode()  # kept
    assert (new / (REV + ".pulled")).read_text() == "REMOTE V2"


def test_pull_skips_unchanged_zips(data, tmp_path, monkeypatch):
    new, src = _fresh_pull(data, tmp_path, monkeypatch)
    lister, downloader = _remote(src)
    calls = []

    def counting(f, d):
        calls.append(f)
        downloader(f, d)

    share.pull("ROOT", lister=lister, downloader=counting)
    assert calls == ["id-PROCESSED.json"]


def test_pull_network_error_is_one_line(data, tmp_path, monkeypatch):
    src = tmp_path / "src"
    share.bundle(src)
    lister, _ = _remote(src)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "new")

    def boom(fid, dest):
        raise httpx.ConnectError("no route")

    with pytest.raises(SystemExit) as e:
        share.pull("ROOT", lister=lister, downloader=boom)
    assert str(e.value).startswith("could not") and "\n" not in str(e.value)

    def bad_list(fid):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(SystemExit) as e:
        share.pull("ROOT", lister=bad_list, downloader=boom)
    assert str(e.value).startswith("could not")


def test_pull_corrupt_zip_is_one_line(data, tmp_path, monkeypatch):
    src = tmp_path / "src"
    share.bundle(src)
    meta = json.loads((src / "PROCESSED.json").read_text())
    z = src / "processed-1960.zip"
    z.write_bytes(b"not a zip")
    meta["processed-1960.zip"]["sha256"] = hashlib.sha256(b"not a zip").hexdigest()
    (src / "PROCESSED.json").write_text(json.dumps(meta))
    lister, downloader = _remote(src)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "new")
    with pytest.raises(SystemExit) as e:
        share.pull("ROOT", lister=lister, downloader=downloader)
    assert str(e.value).startswith("could not")


def test_extract_validates_before_writing(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("work/ok.json", "1")
        zf.writestr("../evil.json", "2")
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        share._extract(z, root)
    assert not (root / "work/ok.json").exists()
    assert not (tmp_path / "evil.json").exists()


def test_download_failure_leaves_no_part_file(tmp_path, monkeypatch):
    from sfzmaps.sources import drive

    class Boom:
        def __enter__(self):
            raise httpx.ConnectError("x")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(drive.httpx, "stream", lambda *a, **k: Boom())
    (tmp_path / "f.zip.part").write_bytes(b"stale")
    with pytest.raises(httpx.ConnectError):
        drive.download("id", tmp_path / "f.zip")
    assert list(tmp_path.iterdir()) == []


def test_symlink_and_secret_dir_not_bundled(data, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("TOPSECRETDATA")
    (data / "work/1948/p1/labels.json").unlink()
    (data / "work/1948/p1/labels.json").symlink_to(outside)
    _w(data, "cache/llm/secret/x.json", b"x")
    _w(data, "cache/llm/mykeys/y.json", b"y")
    share.bundle(tmp_path / "share")
    names = _members(tmp_path / "share")
    assert "work/1948/p1/labels.json" not in names
    assert "cache/llm/secret/x.json" not in names
    assert "cache/llm/mykeys/y.json" not in names
    for z in (tmp_path / "share").glob("*.zip"):
        assert b"TOPSECRETDATA" not in z.read_bytes()


def test_snapshot_fetched_stable_and_no_tmp(data, tmp_path):
    dest = tmp_path / "share"
    share.bundle(dest)
    first = (dest / "processed-ref.zip").read_bytes()
    os.utime(data / "ref/parcels.parquet", (1_700_000_000, 1_700_000_000))
    (dest / "stray.tmp").write_bytes(b"x")
    share.bundle(dest)
    assert (dest / "processed-ref.zip").read_bytes() == first
    assert not list(dest.glob("*.tmp"))


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


def test_bundle_command_prints_folder_and_instruction(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli, config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "work" / "1948" / "A").mkdir(parents=True)
    (tmp_path / "work" / "1948" / "A" / "page.json").write_text("{}")
    r = CliRunner().invoke(cli.app, ["bundle"])
    assert r.exit_code == 0, r.output
    assert str((tmp_path / "share").resolve()) in r.output
    assert "processed-1948.zip" in r.output and "PROCESSED.json" in r.output
    assert "Copy these files into the Drive folder old-planning-maps/processed/" in r.output
