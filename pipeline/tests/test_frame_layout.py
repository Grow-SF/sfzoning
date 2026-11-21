import hashlib

import numpy as np
import pytest
from PIL import Image

from sfzmaps.frame import layout


def test_grid_keeps_the_aspect():
    assert layout.grid((6600, 5266)) == (111, 89) and layout.grid((2550, 3312)) == (85, 111)
    assert layout.grid((10, 5000)) == (1, 111)


def test_raw_features_shape(tiny_backbone):
    f = layout.raw_features(tiny_backbone, np.random.default_rng(0).random((700, 500), dtype=np.float32), "cpu")
    assert f.shape == (8 * layout.LAYERS, 111, 79) and f.dtype == np.float32


def test_raw_features_take_the_last_layers_through_the_norm(tiny_backbone):
    import torch
    lum = np.random.default_rng(1).random((140, 280), dtype=np.float32)
    f = layout.raw_features(tiny_backbone, lum, "cpu")
    rows, cols = layout.grid(lum.shape)
    small = np.asarray(Image.fromarray(lum).resize((cols * 14, rows * 14), Image.BOX), np.float32)
    x = (np.repeat(small[..., None], 3, -1) - layout.MEAN) / layout.STD
    with torch.no_grad():
        hs = tiny_backbone(pixel_values=torch.from_numpy(x.transpose(2, 0, 1)[None].copy()),
                           output_hidden_states=True).hidden_states[-layout.LAYERS:]
        want = torch.cat([tiny_backbone.layernorm(t)[0, 5:] for t in hs], -1).T.reshape(-1, rows, cols).numpy()
    assert np.allclose(f, want, atol=1e-5)


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


def test_reducer_round_trip(tmp_path):
    x = np.random.default_rng(0).normal(size=(6, 20, 10)).astype(np.float32)
    r = layout.Reducer(components=3)
    r.fit_update(x)
    r.finish()
    r.pages = ["a/b"]
    r.save(tmp_path / "reducer.npz")
    r2 = layout.Reducer.load(tmp_path / "reducer.npz")
    assert np.array_equal(r.transform(x), r2.transform(x)) and r2.pages == ["a/b"]


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
    monkeypatch.setattr(layout.data, "page_path", lambda s, h: tmp_path / "work" / s / h / "page.png")
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
