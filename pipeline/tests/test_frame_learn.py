import numpy as np
import torch

from sfzmaps.frame import learn, model


def _cls(inside: np.ndarray) -> np.ndarray:
    """The class mask of a two-class label: 0 (map) inside, 1 (outside the frame) elsewhere."""
    return np.where(inside, 0, 1).astype(np.uint8)


def test_weight_ignores_the_label():
    lum = np.random.default_rng(0).random((60, 70), dtype=np.float32)
    a = learn.Page(("s", "a"), lum, np.zeros((60, 70), np.uint8))
    b = learn.Page(("s", "b"), lum, np.random.default_rng(1).integers(0, 3, (60, 70)).astype(np.uint8))
    aug = learn.Aug(1, True, 1.0, 1.0, 0.0, 0.0)
    wa = learn.make_sample(a, (3, 0, 2, 5), 1.0, aug, 0.1, "cpu")[2]
    wb = learn.make_sample(b, (3, 0, 2, 5), 1.0, aug, 0.1, "cpu")[2]
    assert torch.equal(wa, wb)


def test_padding_weighs_nothing_and_is_outside_after_a_turn():          # non-square page, quarter turn
    pg = learn.Page(("s", "h"), np.full((60, 70), 0.5, np.float32), np.zeros((60, 70), np.uint8))
    x, t, w = learn.make_sample(pg, (0, 0, 0, 0), 1.0, learn.Aug(1, False, 1, 1, 0, 0), 0.1, "cpu")
    assert int((w > 0).sum()) == 60 * 70 and w[0, :70, :60].min() > 0
    assert (t[0, :70, :60] == 0).all() and int((t == 1).sum()) == t.numel() - 60 * 70


def test_target_is_the_class_mask_turned_with_the_page():
    cls = np.ones((60, 70), np.uint8)
    cls[10:30, 5:20] = 0
    cls[40:50, 30:45] = 2
    pg = learn.Page(("s", "h"), np.where(cls == 1, 1.0, 0.2).astype(np.float32), cls)
    x, t, w = learn.make_sample(pg, (0, 0, 0, 0), 1.0, learn.Aug(1, True, 1, 1, 0, 0), 0.1, "cpu")
    assert t.dtype == torch.int64 and t.shape == (1, *x.shape[2:]) and w.shape == t.shape
    np.testing.assert_array_equal(t[0].numpy()[:70, :60], np.rot90(cls[:, ::-1], 1))


def test_sample_is_apply_aug_of_lum_target_and_weight():
    """make_sample moves the bytes before making the floats; the values are those of the plain order: `apply_aug` of
    the luminance, the target and the weight, then the padding."""
    rng = np.random.default_rng(3)
    lum = rng.integers(0, 256, (37, 53)).astype(np.float32) / np.float32(255)
    cls = rng.integers(0, 3, (37, 53)).astype(np.uint8)
    aug = learn.Aug(3, True, 1.3, 0.9, 0.05, 1.2)
    x, t, w = learn.make_sample(learn.Page(("s", "r"), lum, cls), (0, 0, 0, 0), 1.0, aug, 0.1, "cpu")
    lum_t = torch.from_numpy(lum)
    want = learn.apply_aug(aug, lum_t, torch.from_numpy(cls).long(), 0.1 + (1.0 - lum_t))
    for got, expect in zip((x[0], t, w), want):
        assert torch.equal(got[0, :53, :37], expect)


def test_cut_and_scale_shape_the_sample():
    """The cut takes (top, bottom, left, right) px off the page, then the scale resizes what is left."""
    pg = learn.Page(("s", "h"), np.full((60, 70), 0.5, np.float32), np.zeros((60, 70), np.uint8))
    _, _, w = learn.make_sample(pg, (3, 7, 2, 8), 1.1, learn.Aug(0, False, 1, 1, 0, 0), 0.1, "cpu")
    assert int((w > 0).sum()) == round(50 * 1.1) * round(60 * 1.1)


def test_side_depths_and_cut():
    inside = np.zeros((100, 120), bool)
    inside[10:80, 5:120] = True
    assert learn.side_depths(_cls(inside)) == (10, 20, 5, 0)
    assert learn.side_depths(np.where(inside, 2, 1)) == (10, 20, 5, 0)      # a removed block is not outside either
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = learn.draw_cut(rng, (10, 20, 5, 0))
        assert 0 <= c[0] <= 10 and 0 <= c[1] <= 20 and 0 <= c[2] <= 5 and c[3] == 0


def test_side_depths_of_no_map_are_zero():
    assert learn.side_depths(np.ones((30, 40), np.uint8)) == (0, 0, 0, 0)


def test_held_loss_is_the_ink_weighted_multiclass_log_loss(monkeypatch):
    lum = np.array([[0.0, 0.5, 1.0]], np.float32)                           # weights 1, 0.5, 0
    cls = np.array([[2, 0, 1]], np.uint8)
    p = np.array([[[0.2, 0.25, 0.3]], [[0.3, 0.5, 0.4]], [[0.5, 0.25, 0.3]]], np.float32)
    monkeypatch.setattr(learn.model, "prob", lambda nets, lum, device: p)
    loss, ink = learn.held_loss(None, [learn.Page(("s", "h"), lum, cls)], "cpu")
    assert ink == 1.5
    assert abs(loss - (-np.log(0.5) * 1.0 - np.log(0.25) * 0.5) / 1.5) < 1e-6


def _framed_pages(n: int, seed: int) -> list:
    """64 x 64 pages: a 3 px dark rule 6 px in, the map (class 0) within it, short dark map strokes inside it, a dark
    4 x 4 printed block (class 2, removed) at a random place in the map and specks of ink outside (class 1)."""
    rng = np.random.default_rng(seed)
    pages = []
    for i in range(n):
        lum = np.ones((64, 64), np.float32)
        lum[6:58, 6:58] = 0.1
        lum[9:55, 9:55] = 1.0
        cls = np.ones((64, 64), np.uint8)
        cls[9:55, 9:55] = 0
        for _ in range(6):
            r, c = rng.integers(12, 50, size=2)
            lum[r, c:c + 4] = 0.2
        r, c = rng.integers(14, 46, size=2)
        lum[r:r + 4, c:c + 4] = 0.0
        cls[r - 1:r + 5, c - 1:c + 5] = 2
        for r, c in rng.integers(0, 4, size=(3, 2)):
            lum[r, c] = 0.3
        pages.append(learn.Page(("s", str(i)), lum, cls))
    return pages


def test_same_seed_same_weights_with_or_without_prefetch(small_model):
    pages = _framed_pages(3, 0)
    a, _ = learn.train(pages, learn.Recipe(steps=3, check_every=3, prefetch=True), "cpu")
    b, _ = learn.train(pages, learn.Recipe(steps=3, check_every=3, prefetch=False), "cpu")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_snapshots_and_curves(small_model, tmp_path):
    pages = _framed_pages(3, 0)
    net, curves = learn.train(pages[:2], learn.Recipe(steps=4, check_every=2), "cpu", held=pages[2:],
                              snapshots=tmp_path / "snaps")
    assert sorted(p.name for p in (tmp_path / "snaps").iterdir()) == ["step-00002.pt", "step-00004.pt"]
    assert [row[0] for row in curves["train_curve"]] == [2, 4]
    assert [row[0] for row in curves["held_curve"]] == [2, 4]
    assert all(row[2] > 0 for row in curves["held_curve"])
    saved = torch.load(tmp_path / "snaps" / "step-00004.pt")
    assert all(torch.equal(saved[k], v) for k, v in net.state_dict().items())
    assert not net.training and next(net.parameters()).device.type == "cpu"


def test_held_loss_pools_by_ink(small_model):
    net = model.new_net()
    a, b = _framed_pages(2, 1)
    la, ia = learn.held_loss(net, [a], "cpu")
    lb, ib = learn.held_loss(net, [b], "cpu")
    both, ink = learn.held_loss(net, [a, b], "cpu")
    assert abs(ink - (ia + ib)) < 1e-9
    assert abs(both - (la * ia + lb * ib) / (ia + ib)) < 1e-9


def test_learns_a_printed_frame(small_model):
    """The rule (outside), the printed block (removed) and the map's own strokes (inside) are all ink, so dark alone
    is not the answer.  A rate above the recipe's, so the tiny net's learning shows within 150 steps (at 1e-3 it
    needs about 300)."""
    pages = _framed_pages(6, 2)
    _, curves = learn.train(pages[:4], learn.Recipe(steps=150, check_every=10, lr=3e-3), "cpu", held=pages[4:])
    held = [row[1] for row in curves["held_curve"]]
    assert held[-1] < held[0] / 2
