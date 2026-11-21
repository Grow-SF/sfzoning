"""The page layout's features: a frozen pretrained backbone's patch tokens, reduced to a few principal directions.

DINOv2 ViT-B/14 with registers (`facebook/dinov2-with-registers-base`, Apache-2.0), loaded through Hugging Face
`transformers` from a pinned revision of its safetensors weights, whose sha256 is checked before anything loads.  The
page is shown to it at `LONG_PATCHES` patches on its long side; the last `LAYERS` layers' patch tokens (each through
the final layer norm) are concatenated, then reduced to `COMPONENTS` principal directions fitted over every page.  The
reduced features are float16 `.npy` files, one per page, under `features_dir(out)`."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sfzmaps.frame import data

log = logging.getLogger(__name__)

MODEL_ID = "facebook/dinov2-with-registers-base"
# The snapshot the spike (scripts/spikes/issue71_dino.py) measured with, and the sha256 of its model.safetensors.
REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
WEIGHTS_SHA256 = "7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362"
PATCH = 14
LONG_PATCHES = 111   # 1,554 px on the long side
LAYERS = 4
COMPONENTS = 128
HIDDEN = 768
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def grid(shape: tuple[int, int]) -> tuple[int, int]:
    """(rows, cols) of patches: the long side is `LONG_PATCHES`, the short side keeps the aspect."""
    long = max(shape)
    return tuple(LONG_PATCHES if n == long else max(1, round(n / long * LONG_PATCHES)) for n in shape)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_backbone(device: str):
    """The frozen backbone in eval mode on `device`.  Only the pinned revision is fetched, and its weights' sha256
    must match `WEIGHTS_SHA256` (RuntimeError naming the file otherwise) before the model is loaded."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    weights = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION))
    got = _sha256(weights)
    if got != WEIGHTS_SHA256:
        raise RuntimeError(f"{weights} has sha256 {got}, not the pinned {WEIGHTS_SHA256} ({MODEL_ID} at {REVISION})")
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, use_safetensors=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def raw_features(backbone, lum: np.ndarray, device: str) -> np.ndarray:
    """Float32 (HIDDEN * LAYERS, rows, cols): the last `LAYERS` hidden states' patch tokens, each through the final
    layer norm (class and register tokens dropped), concatenated.  The page goes in resized (Pillow BOX) to
    `grid * PATCH`, grey in all three channels, ImageNet-normalised."""
    import torch

    rows, cols = grid(lum.shape)
    small = np.asarray(Image.fromarray(lum).resize((cols * PATCH, rows * PATCH), Image.BOX), np.float32)
    x = (np.repeat(small[..., None], 3, -1) - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)
    skip = 1 + backbone.config.num_register_tokens
    with torch.no_grad():
        hs = backbone(pixel_values=x, output_hidden_states=True).hidden_states[-LAYERS:]
        toks = [backbone.layernorm(t)[0, skip:] for t in hs]
        f = torch.cat(toks, -1).T.reshape(-1, rows, cols)
    return f.float().cpu().numpy()


class Reducer:
    """Standardise each channel, then project on the top principal directions of the standardised features (ordered by
    variance, so the first components carry the most).  `fit_update` accumulates the count, the
    sums and the outer products of the patches' features (float64); `finish` makes the mean, the standard deviation
    and the directions."""

    def __init__(self, components: int | None = None):
        self.components = COMPONENTS if components is None else components
        self.n = 0
        self.sum = self.outer = None
        self.mean = self.std = self.dirs = None
        self.pages: list[str] = []

    def fit_update(self, feats: np.ndarray) -> None:
        x = feats.reshape(feats.shape[0], -1).astype(np.float64)
        if self.sum is None:
            self.sum = np.zeros(x.shape[0])
            self.outer = np.zeros((x.shape[0],) * 2)
        self.n += x.shape[1]
        self.sum += x.sum(1)
        self.outer += x @ x.T

    def finish(self) -> None:
        mean = self.sum / self.n
        cov = self.outer / self.n - np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(cov), 0)) + 1e-6
        corr = cov / np.outer(std, std)
        _, vecs = np.linalg.eigh(corr)   # ascending eigenvalues: the top directions are the last columns
        self.mean, self.std = mean, std
        self.dirs = vecs[:, ::-1][:, : self.components].copy()

    def transform(self, feats: np.ndarray) -> np.ndarray:
        c, rows, cols = feats.shape
        z = (feats.reshape(c, -1).astype(np.float64) - self.mean[:, None]) / self.std[:, None]
        y = self.dirs.T @ z
        return y.reshape(-1, rows, cols).astype(np.float16)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std, dirs=self.dirs, n=self.n,
                 pages=np.array(self.pages, dtype=str))

    @classmethod
    def load(cls, path: Path) -> "Reducer":
        with np.load(path) as z:
            r = cls(components=z["dirs"].shape[1])
            r.mean, r.std, r.dirs, r.n = z["mean"], z["std"], z["dirs"], int(z["n"])
            r.pages = [str(p) for p in z["pages"]]
        return r


def features_dir(out: Path) -> Path:
    return out / "features"


def _lum(sid: data.SheetId) -> np.ndarray:
    return data.load_luminance(data.page_path(*sid))


def extract_all(sids: list[data.SheetId], device: str, out: Path) -> None:
    """Pass 1 fits the reducer over every page; pass 2 writes `<series>/<sheet>.npy` (float16) under
    `features_dir(out)`.  Skipped when `reducer.npz` exists with this page list and every file is there."""
    pages = [f"{s}/{h}" for s, h in sids]
    path = out / "reducer.npz"
    fdir = features_dir(out)
    if path.exists() and Reducer.load(path).pages == pages and all((fdir / s / f"{h}.npy").exists() for s, h in sids):
        return
    out.mkdir(parents=True, exist_ok=True)
    backbone = load_backbone(device)
    # pass 1 keeps no features: they would not fit in memory for every page; pass 2 extracts again (same cost)
    reducer = Reducer()
    reducer.pages = pages
    t0 = time.perf_counter()
    for sid in sids:
        reducer.fit_update(raw_features(backbone, _lum(sid), device))
    reducer.finish()
    log.info("reducer fitted over %d pages in %.0f s", len(sids), time.perf_counter() - t0)
    for sid in sids:
        target = fdir / sid[0] / f"{sid[1]}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, reducer.transform(raw_features(backbone, _lum(sid), device)))
    reducer.save(path)
