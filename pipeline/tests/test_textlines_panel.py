"""The picture panel and the missed-text audit (scripts/textlines): synthetic pages, no pipeline data."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
import audit  # noqa: E402
import panel  # noqa: E402


def _red(img):
    return (img[..., 2] > 200) & (img[..., 1] < 80) & (img[..., 0] < 80)


def test_a_source_of_zeros_draws_no_red(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "WINDOWS", {"w": ("s", 0, 0, 64)})
    monkeypatch.setattr(panel, "page_ink", lambda sheet: np.ones((64, 64), bool))
    out = panel.render({"zeros": lambda s: np.zeros((64, 64), np.uint8)}, tmp_path / "p.png", "t")
    img = cv2.imread(str(out))
    assert not _red(img).any()


def test_red_is_ink_called_text_and_only_ink(tmp_path, monkeypatch):
    ink = np.zeros((64, 64), bool)
    ink[10:20, 10:20] = True
    monkeypatch.setattr(panel, "WINDOWS", {"w": ("s", 32, 32, 64)})
    monkeypatch.setattr(panel, "page_ink", lambda sheet: ink)
    full = {"all": lambda s: np.full((64, 64), 255, np.uint8)}
    assert _red(cv2.imread(str(panel.render(full, tmp_path / "p.png", "t")))).any()
    monkeypatch.setattr(panel, "page_ink", lambda sheet: np.zeros((64, 64), bool))
    assert not _red(cv2.imread(str(panel.render(full, tmp_path / "r.png", "t")))).any()


def test_a_window_past_the_page_edge_is_padded(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "WINDOWS", {"w": ("s", 60, 60, 64)})
    monkeypatch.setattr(panel, "page_ink", lambda sheet: np.ones((64, 64), bool))
    out = panel.render({"a": lambda s: np.zeros((64, 64), np.uint8)}, tmp_path / "p.png", "t")
    assert cv2.imread(str(out)) is not None


def _page():
    ink = np.zeros((200, 200), np.uint8)
    for y, x, ch in [(20, 20, "A"), (20, 100, "K"), (100, 20, "E")]:
        cv2.putText(ink, ch, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 2)
    return ink.view(bool)


def test_audit_picks_only_letter_sized_pieces_the_detector_misses():
    ink = _page()
    none = np.zeros((200, 200), bool)
    pieces = audit.pieces(ink, none, none)
    assert len(pieces) == 3
    assert len(audit.missed(pieces, np.zeros((200, 200), np.uint8))) == 3
    assert len(audit.missed(pieces, np.full((200, 200), 255, np.uint8))) == 0


def test_audit_skips_pieces_near_a_box_or_a_run_line():
    ink = _page()
    none = np.zeros((200, 200), bool)
    box = np.zeros((200, 200), bool)
    box[10:40, 10:60] = True
    assert len(audit.pieces(ink, box, none)) == 2
    line = np.zeros((200, 200), bool)
    line[:, 98:120] = True
    assert len(audit.pieces(ink, none, line)) == 2


def test_audit_sheet_writes_a_picture(tmp_path):
    items = [{"sheet": "s", "x": 30, "y": 30, "x0": 20, "y0": 20, "w": 10, "h": 12, "chance": 10}] * 3
    out = audit.sheet(items, tmp_path / "a.png", ink_of=lambda s: _page())
    assert cv2.imread(str(out)) is not None
