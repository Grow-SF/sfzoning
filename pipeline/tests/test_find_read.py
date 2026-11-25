"""Reading a page with the finder (`find.read`): a page read a square at a time gives what one pass over the whole
page gives, and a read is stored as the levels round(255 p) of its two answers (street, block edge); a read stored
with the zoning-border answer of before still loads."""
import numpy as np
import torch

from sfzmaps.find import model, read


def test_tiled_reading_is_one_pass():
    torch.manual_seed(0)
    net = model.FindNet(4).eval()
    rng = np.random.default_rng(0)
    lum = rng.uniform(0, 1, (200, 150)).astype(np.float32)
    keep = np.ones(lum.shape, bool)
    keep[:, :7] = False
    got = read.read_page(net, lum, keep, "cpu", tile=64)
    h, w = lum.shape
    padded = np.pad(lum, ((0, -h % model.MULTIPLE), (0, -w % model.MULTIPLE)), constant_values=1.0)
    with torch.no_grad():
        whole = torch.sigmoid(net(torch.from_numpy(padded)[None, None]))[0, :, :h, :w].numpy()
    whole[:, ~keep] = 0
    assert got.shape == (2, h, w) and got.dtype == np.float32
    np.testing.assert_allclose(got, whole, atol=1e-5, rtol=0)


def test_a_read_is_stored_as_levels(tmp_path):
    rng = np.random.default_rng(1)
    chances = rng.uniform(0, 1, (2, 9, 7)).astype(np.float32)
    f = tmp_path / "r" / "read.npz"
    read.save(f, chances)
    with np.load(f) as z:
        assert sorted(z.files) == ["edge", "street"]
        assert z["street"].dtype == np.uint8
        np.testing.assert_array_equal(z["edge"], np.rint(chances[model.EDGE] * 255).astype(np.uint8))
    back = read.load(f)
    np.testing.assert_array_equal(back, read.stored(chances))
    assert np.abs(back - chances).max() <= 0.5 / 255 + 1e-7


def test_a_read_stored_with_three_answers_still_loads(tmp_path):
    rng = np.random.default_rng(2)
    lv = {k: rng.integers(0, 256, (5, 6), dtype=np.uint8) for k in ("street", "edge", "border")}
    f = tmp_path / "old.npz"
    np.savez_compressed(f, **lv)
    back = read.load(f)
    assert back.shape == (2, 5, 6)
    np.testing.assert_array_equal(back[model.STREET], lv["street"] / np.float32(255))
    np.testing.assert_array_equal(back[model.EDGE], lv["edge"] / np.float32(255))


def _lined_page(shape=(300, 260), seed=2):
    rng = np.random.default_rng(seed)
    lum = np.ones(shape, np.float32)
    for _ in range(40):                                      # printed lines at random
        y, x = rng.integers(0, shape[0] - 10), rng.integers(0, shape[1] - 10)
        lum[y:y + 2, x:x + rng.integers(5, 60)] = 0
        lum[y:y + rng.integers(5, 60), x:x + 2] = 0
    return lum


def _lively_net():
    """An untrained network whose convolutions' weights are tripled, so that its answers vary with the print as a
    trained one's do (at the weights it starts with they barely vary, and nothing about the grid would show)."""
    torch.manual_seed(0)
    net = model.FindNet(4).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.mul_(3)
    return net


def test_the_network_answers_differently_on_another_grid():
    # Why this test exists: the network pools four times by 2 x 2, so a px's answer depends on where the 16 px grid of
    # those poolings falls.  A window read with its corner off the page's grid (as a crop grown by model.REACH = 122
    # px is) gives other numbers than the whole page at the same px; every read in this code is made on the page's
    # grid (`read.window`) so that a crop reads as its sheet does.  If this ever fails, the network has become
    # shift-invariant and the alignment is no longer needed.
    net = _lively_net()
    lum = _lined_page()
    keep = np.ones(lum.shape, bool)
    whole = read.read_page(net, lum, keep, "cpu")
    y0, x0, h, w = 140, 130, 60, 60
    oy, ox = y0 - model.REACH, x0 - model.REACH                    # off the grid: 18 and 8 mod 16
    assert oy % model.MULTIPLE and ox % model.MULTIPLE
    win = read.read_page(net, lum[oy:y0 + h + model.REACH, ox:x0 + w + model.REACH],
                         keep[oy:y0 + h + model.REACH, ox:x0 + w + model.REACH], "cpu")
    assert np.abs(win[:, y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w] - whole[:, y0:y0 + h, x0:x0 + w]).max() > 1e-3


def test_a_window_on_the_page_grid_reads_as_the_whole_page():
    net = _lively_net()
    lum = _lined_page()
    keep = np.ones(lum.shape, bool)
    whole = read.read_page(net, lum, keep, "cpu")
    for box in ((37, 53, 60, 40), (130, 141, 70, 90), (0, 3, 50, 50), (199, 250, 61, 50)):  # off the grid, to the ends
        outer, inner = read.window(box, lum.shape)
        assert outer[0].start % model.MULTIPLE == 0 and outer[1].start % model.MULTIPLE == 0
        x0, y0, w, h = box
        assert outer[1].start <= max(0, x0 - model.REACH) and outer[0].start <= max(0, y0 - model.REACH)
        assert outer[1].stop >= min(lum.shape[1], x0 + w + model.REACH)
        got = read.read_page(net, lum[outer], keep[outer], "cpu")[(slice(None), *inner)]
        np.testing.assert_allclose(got, whole[:, y0:y0 + h, x0:x0 + w], atol=1e-5, rtol=0, err_msg=str(box))
