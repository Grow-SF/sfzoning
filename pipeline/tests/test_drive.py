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
