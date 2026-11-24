"""Reading a page with the finder (`find.read`): a page read a square at a time gives what one pass over the whole
page gives, and a read is stored as the levels round(255 p)."""
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
    assert got.shape == (3, h, w) and got.dtype == np.float32
    np.testing.assert_allclose(got, whole, atol=1e-5, rtol=0)


def test_a_read_is_stored_as_levels(tmp_path):
    rng = np.random.default_rng(1)
    chances = rng.uniform(0, 1, (3, 9, 7)).astype(np.float32)
    f = tmp_path / "r" / "read.npz"
    read.save(f, chances)
    with np.load(f) as z:
        assert sorted(z.files) == ["border", "edge", "street"]
        assert z["street"].dtype == np.uint8
        np.testing.assert_array_equal(z["edge"], np.rint(chances[model.EDGE] * 255).astype(np.uint8))
    back = read.load(f)
    np.testing.assert_array_equal(back, read.stored(chances))
    assert np.abs(back - chances).max() <= 0.5 / 255 + 1e-7
