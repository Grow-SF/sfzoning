from pathlib import Path

from sfzmaps.sources import drive

FIXTURE = Path(__file__).parent / "fixtures" / "drive_folder.html"


def test_parse_listing_finds_entries():
    entries = {e.title: e for e in drive.parse_listing(FIXTURE.read_text())}
    assert entries["aligned"].is_folder is True
    assert entries["1948.tif"].is_folder is False
    assert {"cropped", "georeferenced", "originals"} <= set(entries)


def test_parse_listing_unescapes_titles():
    snippet = (
        '<div class="flip-entry" id="entry-abc123"><div class="flip-entry-info">'
        '<a href="https://drive.google.com/file/d/abc123/view">'
        '<div class="flip-entry-title">1980 - 1990  Height &amp; Bulk Maps.PDF</div>'
        "</a></div></div>"
    )
    [e] = drive.parse_listing(snippet)
    assert e.title == "1980 - 1990  Height & Bulk Maps.PDF"
    assert e.id == "abc123"
    assert e.is_folder is False


class _FakeStream:
    def __init__(self, body: bytes, ctype: str):
        self.body, self.headers = body, {"content-type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self, n):
        yield self.body


def _serve(monkeypatch, body, ctype="application/octet-stream"):
    monkeypatch.setattr(drive.httpx, "stream", lambda *a, **k: _FakeStream(body, ctype))


def test_download_rejects_html_by_content_type(tmp_path, monkeypatch):
    import pytest

    _serve(monkeypatch, b"PK\x03\x04 looks binary", "text/html; charset=utf-8")
    with pytest.raises(drive.DriveHTMLError):
        drive.download("id", tmp_path / "a.zip")
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_html_by_first_bytes(tmp_path, monkeypatch):
    import pytest

    _serve(monkeypatch, b"\n  <!DOCTYPE html><html><title>Google Drive - Quota exceeded</title>")
    with pytest.raises(drive.DriveHTMLError):
        drive.download("id", tmp_path / "a.zip")
    _serve(monkeypatch, b"<HTML><body>Virus scan warning</body></HTML>")
    with pytest.raises(ValueError):  # pull catches ValueError
        drive.download("id", tmp_path / "a.zip")
    assert list(tmp_path.iterdir()) == []
    _serve(monkeypatch, b"PK\x03\x04real")
    drive.download("id", tmp_path / "a.zip")
    assert (tmp_path / "a.zip").read_bytes() == b"PK\x03\x04real"


def test_sync_replaces_saved_html_page(tmp_path, monkeypatch):
    (tmp_path / "scan.jpg").write_bytes(b"<!DOCTYPE html><html>quota</html>")
    (tmp_path / "ok.jpg").write_bytes(b"\xff\xd8good")
    monkeypatch.setattr(drive, "list_folder", lambda fid: [drive.DriveEntry("1", "scan.jpg", False),
                                                           drive.DriveEntry("2", "ok.jpg", False)])
    got = []

    def dl(fid, dest):
        got.append(fid)
        dest.write_bytes(b"\xff\xd8fresh")
    monkeypatch.setattr(drive, "download", dl)
    drive.sync("root", tmp_path)
    assert got == ["1"] and (tmp_path / "scan.jpg").read_bytes() == b"\xff\xd8fresh"
