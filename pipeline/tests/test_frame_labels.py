import re

import numpy as np

from sfzmaps.frame import labels


def test_stretch_on_a_rule_ranks_first():
    lum = np.ones((200, 200), np.float32)
    lum[:, 100] = 0.0                                                     # a printed rule at x = 100
    poly = [(20, 20), (100, 20), (100, 180), (20, 180)]                   # right edge on the rule
    st = labels.outline_stretches(lum, [poly], "s", "h")
    assert st[0].ink > 40 and min(x for x, _ in (st[0].start, st[0].end)) >= 99
    assert st[-1].ink == 0                                                # edges on paper


def test_page_edge_is_not_listed():
    lum = np.zeros((100, 100), np.float32)                                # all ink
    st = labels.outline_stretches(lum, [[(0, 0), (99, 0), (99, 99), (0, 99)]], "s", "h")
    assert st == []


def test_frame_labels_command_checks_removed_blocks(monkeypatch, tmp_path):
    """The command ranks the ink under the removed blocks' outlines too, not only the map polygons'."""
    import json

    from PIL import Image
    from typer.testing import CliRunner

    from sfzmaps import cli
    from sfzmaps.frame import data as fd

    page = np.full((200, 200), 255, np.uint8)
    page[100, 60:141] = 0                                                 # a printed rule under the block's top edge
    Image.fromarray(page).save(tmp_path / "page.png")
    root = tmp_path / "labels"
    (root / "s").mkdir(parents=True)
    (root / "s" / "h.json").write_text(json.dumps({
        "polygons": [[[20, 20], [180, 20], [180, 180], [20, 180]]],      # on paper
        "remove": [[[60, 100], [140, 100], [140, 140], [60, 140]]],      # top edge on the rule
    }))
    load_label, load_remove = fd.load_label, fd.load_remove
    monkeypatch.setattr(fd, "labelled_sheets", lambda: [("s", "h")])
    monkeypatch.setattr(fd, "load_label", lambda s, h: load_label(s, h, root))
    monkeypatch.setattr(fd, "load_remove", lambda s, h: load_remove(s, h, root))
    monkeypatch.setattr(fd, "page_path", lambda s, h: tmp_path / "page.png")
    monkeypatch.setattr(fd, "OUT", tmp_path / "out")
    res = CliRunner().invoke(cli.app, ["frame-labels"])
    assert res.exit_code == 0, res.output
    row = re.search(r"^s/h\s+(\S+)\s+(\S+)\s*$", res.output, re.M)
    assert row, res.output
    total, best = map(float, row.groups())
    assert total >= 80 and best >= 49
    assert (tmp_path / "out" / "labels-qa" / "s" / "h.png").exists()
