import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from sfzmaps.frame import layout


def test_grid_keeps_the_aspect():
    assert layout.grid((6600, 5266)) == (111, 89) and layout.grid((2550, 3312)) == (85, 111)
    assert layout.grid((10, 5000)) == (1, 111)


def test_raw_features_drop_the_class_and_register_tokens(tiny_backbone):
    lum = np.random.default_rng(0).random((700, 500), dtype=np.float32)
    f = layout.raw_features(tiny_backbone, lum, "cpu")
    assert f.shape == (8 * layout.LAYERS, 111, 79)        # one feature column per patch of the grid
    with torch.no_grad():                                  # other class and register tokens: the same patch features
        tiny_backbone.cls.fill_(100.0)
        tiny_backbone.reg.fill_(-100.0)
    np.testing.assert_array_equal(layout.raw_features(tiny_backbone, lum, "cpu"), f)


def test_reducer_whitens_and_orders_components():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 40, 30)).astype(np.float32)
    x[0] *= 10
    x[1] += 5
    r = layout.Reducer(components=3)
    r.fit_update(x)
    r.finish()
    y = r.transform(x).astype(np.float32).reshape(3, -1)
    assert abs(y.mean(1)).max() < 1e-2 and y.var(1)[0] >= y.var(1)[1] >= y.var(1)[2]


def test_wrong_weights_hash_is_refused(monkeypatch, tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"not the weights")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(weights))
    loaded = []
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *a, **k: loaded.append(1))
    assert hashlib.sha256(weights.read_bytes()).hexdigest() != layout.WEIGHTS_SHA256
    with pytest.raises(RuntimeError, match="model.safetensors"):
        layout.load_backbone("cpu")
    assert not loaded


def _pages(tmp_path, monkeypatch, n=2):
    sids = [("s", f"p{i}") for i in range(n)]
    monkeypatch.setattr(layout.data, "page_path", lambda s, h, work=None: tmp_path / "work" / s / h / "page.png")
    for i, sid in enumerate(sids):
        path = layout.data.page_path(*sid)
        path.parent.mkdir(parents=True)
        Image.fromarray((np.random.default_rng(i).random((70, 50)) * 255).astype(np.uint8)).save(path)
    return sids


def test_extract_all_is_resumable(tiny_backbone, tmp_path, monkeypatch):
    sids = _pages(tmp_path, monkeypatch)
    monkeypatch.setattr(layout, "load_backbone", lambda device: tiny_backbone)
    monkeypatch.setattr(layout, "COMPONENTS", 5)
    out = tmp_path / "out"
    layout.extract_all(sids, "cpu", out)
    files = [layout.features_dir(out) / s / f"{h}.npy" for s, h in sids]
    assert all(f.exists() for f in files)
    f = np.load(files[0])
    assert f.dtype == np.float16 and f.shape[0] == 5
    assert layout.Reducer.load(out / "reducer.npz").pages == ["s/p0", "s/p1"]
    stamp = [p.stat().st_mtime_ns for p in files]
    calls = []
    monkeypatch.setattr(layout, "raw_features", lambda *a, **k: calls.append(1))
    layout.extract_all(sids, "cpu", out)
    assert not calls and [p.stat().st_mtime_ns for p in files] == stamp


# ---- the head ----


def test_combine_is_a_tempered_softmax_mean():
    z = np.random.default_rng(0).normal(size=(3, 4, 5)).astype(np.float32)
    p = layout.combine([z, 2 * z], temperature=2.0)
    np.testing.assert_allclose(p.sum(0), 1.0, rtol=1e-6)
    e1, e2 = np.exp(z / 2), np.exp(z)
    np.testing.assert_allclose(p, (e1 / e1.sum(0) + e2 / e2.sum(0)) / 2, rtol=1e-5)


def test_combine_of_large_logits_is_finite():
    z = np.array([1000.0, 0.0, -1000.0], np.float32)[:, None, None]
    np.testing.assert_array_equal(layout.combine([z]), np.array([1.0, 0.0, 0.0], np.float32)[:, None, None])


def test_patch_targets_are_area_shares():
    cls = np.zeros((28, 28), np.uint8)
    cls[:, 14:] = 1
    cls[:7, :7] = 2
    s, w = layout.patch_targets(cls, np.ones((28, 28), np.float32), (2, 2))
    assert s.shape == (3, 2, 2) and s.dtype == np.float32 and w.dtype == np.float32
    np.testing.assert_allclose(s[:, 0, 0], [0.75, 0, 0.25])
    np.testing.assert_allclose(s[:, 0, 1], [0, 1, 0])
    np.testing.assert_allclose(w, 0.1)


def test_weight_is_floor_plus_mean_ink_and_ignores_the_label():
    lum = np.ones((28, 28), np.float32)
    lum[:14, :14] = 0.0   # black: ink 1 in the top-left patch
    a = np.zeros((28, 28), np.uint8)
    b = np.ones((28, 28), np.uint8)
    _, wa = layout.patch_targets(a, lum, (2, 2))
    _, wb = layout.patch_targets(b, lum, (2, 2))
    np.testing.assert_allclose(wa, [[1.1, 0.1], [0.1, 0.1]], atol=1e-6)
    np.testing.assert_array_equal(wa, wb)


def _toy_pages(n, seed, c=6, rows=8, cols=10):
    """Feature 0 is high on the right half (class 1: outside), low on the left (class 0), plus noise."""
    rng = np.random.default_rng(seed)
    feats, tg = [], []
    for _ in range(n):
        f = rng.normal(size=(c, rows, cols)).astype(np.float16)
        f[0, :, cols // 2:] += 4
        shares = np.zeros((3, rows, cols), np.float32)
        shares[0, :, :cols // 2] = 1
        shares[1, :, cols // 2:] = 1
        feats.append(f)
        tg.append((shares, np.full((rows, cols), 0.5, np.float32)))
    return feats, tg


def test_head_learns_a_separable_layout(monkeypatch):
    monkeypatch.setattr(layout, "COMPONENTS", 6)
    feats, tg = _toy_pages(4, 0)
    head = layout.train_head(feats, tg, seed=0)
    test, _ = _toy_pages(1, 99)
    z = layout.head_logits(head, test[0])
    assert z.shape == (3, 8, 10) and z.dtype == np.float32
    pred = z.argmax(0)
    assert (pred[:, :5] == 0).mean() > 0.9 and (pred[:, 5:] == 1).mean() > 0.9


def test_same_seed_same_head(monkeypatch):
    monkeypatch.setattr(layout, "COMPONENTS", 6)
    monkeypatch.setattr(layout, "HEAD_MAX_STEPS", 20)        # determinism needs no convergence
    feats, tg = _toy_pages(3, 1)
    a = layout.head_logits(layout.train_head(feats, tg, seed=3), feats[0])
    b = layout.head_logits(layout.train_head(feats, tg, seed=3), feats[0])
    c = layout.head_logits(layout.train_head(feats, tg, seed=4), feats[0])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_page_logits_shape_and_orientation():
    z = np.zeros((3, 4, 5), np.float32)
    z[0, :, :2] = 5
    z[1, :, 3:] = 5
    up = layout.page_logits(z, (40, 100))
    assert up.shape == (3, 40, 100) and up.dtype == np.float32
    assert up[0, 20, 5] > up[1, 20, 5] and up[1, 20, 95] > up[0, 20, 95]


def _per_page_loss(head, feats, tg):
    """The loss page by page: each page alone through the head, weighted by its patches' weights."""
    wsum = sum(float(w.sum()) for _, w in tg)
    loss = 0.0
    for f, (sh, w) in zip(feats, tg):
        z = head(torch.from_numpy(np.asarray(f, np.float32))[None])[0]
        loss = loss + (-(torch.from_numpy(sh) * torch.nn.functional.log_softmax(z, 0)).sum(0)
                       * torch.from_numpy(w)).sum() / wsum
    return loss


def _ragged_pages():
    """Three pages of different shapes (landscape, portrait, small), random features, targets and weights."""
    rng = np.random.default_rng(7)
    feats, tg = [], []
    for rows, cols in ((8, 10), (11, 6), (4, 5)):
        feats.append(rng.normal(size=(6, rows, cols)).astype(np.float16))
        tg.append((rng.dirichlet((1, 1, 1), (rows, cols)).transpose(2, 0, 1).astype(np.float32),
                   (0.1 + rng.random((rows, cols))).astype(np.float32)))
    return feats, tg


def test_the_padded_batch_gives_the_per_page_loss_and_gradients(monkeypatch):
    monkeypatch.setattr(layout, "COMPONENTS", 6)
    feats, tg = _ragged_pages()
    torch.manual_seed(0)
    head = layout.new_head()
    want = _per_page_loss(head, feats, tg)
    want.backward()
    g_want = [p.grad.clone() for p in head.parameters()]
    head.zero_grad()
    got = layout.batch_loss(head, *layout.batch(feats, tg))
    got.backward()
    assert float(got.detach()) == pytest.approx(float(want.detach()), rel=1e-6)
    for a, b in zip(head.parameters(), g_want):
        torch.testing.assert_close(a.grad, b, rtol=1e-5, atol=1e-7)
