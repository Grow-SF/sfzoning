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
