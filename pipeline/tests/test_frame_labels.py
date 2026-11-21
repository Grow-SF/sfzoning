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


def test_qa_sheet_stacks_one_crop_per_stretch(tmp_path):
    from PIL import Image
    lum = np.ones((200, 200), np.float32)
    poly = [(20, 20), (100, 20), (100, 180), (20, 180)]
    st = labels.outline_stretches(lum, [poly], "s", "h")[:2]
    out = labels.qa_sheet(lum, [poly], st, tmp_path / "s" / "h.png")
    with Image.open(out) as im:
        assert im.height >= 2 * (2 * labels.CROP_MARGIN)                  # two crops, never shrunk
