"""Spike S2: what every step shares (the crops on disk, the vocabulary, the CTC code posterior, contact sheets).

Runs in the project's environment and in the docTR overlay (`uv run --with python-doctr --with torch==2.15.0`): it
imports only numpy, pandas, cv2 and (lazily) torch.
"""
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[3]
SPIKE = PIPELINE / "data" / "solver" / "experiments" / "spike-s2"
OUT = Path(os.environ.get("SFZ_S2_OUT", SPIKE))     # SFZ_S2_OUT: another run's folder (v1/), rescored as this one
INPUTS = SPIKE / "inputs"
VOCAB = " &'(),-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = len(VOCAB)               # docTR's CTC blank: the last class
HEIGHT = 32
STRIDE = 4                       # input px per CTC frame (mobilenet_v3_small_r: width / 4)
GPU_LOCK = Path("/tmp/sfzmaps-mps.lock")


class Crops:
    """The crops of crops.py (or synth.py): crop i is `flat[:, offsets[i]:offsets[i + 1]]`, uint8, white paper."""

    def __init__(self, npz: Path, meta: Path | None = None):
        with np.load(npz) as z:
            self.flat, self.offsets, self.widths = z["flat"], z["offsets"], z["widths"]
        self.meta = pd.read_pickle(meta) if meta is not None else None

    def __len__(self):
        return len(self.widths)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.flat[:, self.offsets[i]:self.offsets[i + 1]]


def load_crops() -> Crops:
    crops = Crops(OUT / "crops.npz", OUT / "crops_meta.pkl")
    for col in ("touch_quarter", "touch_validation", "touch_blocks"):     # the answered labels have none
        crops.meta[col] = crops.meta[col].astype(object).where(crops.meta[col].notna(), False).astype(bool)
    return crops


def encode(text: str) -> list[int]:
    return [VOCAB.index(c) for c in text]


def padded(crop: np.ndarray) -> np.ndarray:
    """A crop padded with paper on the right to a whole number of CTC frames (the same px for every batching)."""
    w = crop.shape[1]
    want = max(STRIDE * 2, math.ceil(w / STRIDE) * STRIDE)
    if want == w:
        return crop
    return np.pad(crop, ((0, 0), (0, want - w)), constant_values=255)


WIDTHS = (64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280)


def to_input(batch: list[np.ndarray], quantize: bool = False):
    """A batch of crops (padded, `padded`; right-padded with paper to the longest, or with `quantize` to the next of
    WIDTHS, so the GPU sees few shapes) as the network's input: float32 (n, 3, HEIGHT, W), docTR's normalisation of
    the grey value repeated over its three channels; and each crop's frame count."""
    import torch
    crops = [padded(c) for c in batch]
    w = max(c.shape[1] for c in crops)
    if quantize:
        w = next((q for q in WIDTHS if q >= w), w)
    x = np.full((len(crops), HEIGHT, w), 255, np.uint8)
    for i, c in enumerate(crops):
        x[i, :, :c.shape[1]] = c
    t = torch.from_numpy(x).float().div_(255.0)
    mean, std = 0.694, 0.299
    t = (t - mean) / std
    return t[:, None].expand(-1, 3, -1, -1).contiguous(), [c.shape[1] // STRIDE for c in crops]


def forward(model, x):
    """docTR CRNN logits (n, frames, classes) without its post-processing (its forward wants targets in training)."""
    f = model.feat_extractor(x)
    c, h, w = f.shape[1:]
    f = f.reshape(-1, h * c, w).transpose(1, 2)
    return model.linear(model.decoder(f)[0])


def string_log_probs(log_probs, strings: list[str]):
    """log p(s | x) under CTC for each of `strings` (exact forward algorithm, `torch.nn.functional.ctc_loss`), for one
    crop: `log_probs` (frames, classes) float64 log-softmax.  The empty string's is the all-blank path.  -inf when a
    string can't fit the frames."""
    import torch
    import torch.nn.functional as F
    n = len(strings)
    lens = [len(s) for s in strings]
    tgt = torch.zeros((n, max(1, max(lens))), dtype=torch.long)
    for i, s in enumerate(strings):
        if s:
            tgt[i, :len(s)] = torch.tensor(encode(s))
    lp = log_probs[:, None, :].expand(-1, n, -1)
    nll = F.ctc_loss(lp, tgt, torch.full((n,), log_probs.shape[0], dtype=torch.long),
                     torch.tensor(lens, dtype=torch.long), blank=BLANK, reduction="none", zero_infinity=False)
    return (-nll).numpy()


def best_path(log_probs) -> str:
    """The greedy (best path) reading of one crop's frames."""
    idx = log_probs.argmax(-1).tolist()
    out, last = [], None
    for k in idx:
        if k != last and k != BLANK:
            out.append(VOCAB[k])
        last = k
    return "".join(out)


def code_posterior(log_probs, forms: dict[str, str]) -> dict[str, float]:
    """The posterior over the codes, "none" (the empty string) and "other" (every other string: the CTC remainder)
    from one crop's CTC frames: p(code) = sum of its printed forms' p(s | x), exactly; p(other) = 1 - the rest."""
    strings = sorted(forms)
    lp = string_log_probs(log_probs, strings + [""])
    p_forms = np.exp(lp[:-1])
    out: dict[str, float] = {}
    for s, p in zip(strings, p_forms):
        out[forms[s]] = out.get(forms[s], 0.0) + float(p)
    out["none"] = float(np.exp(lp[-1]))
    out["other"] = max(0.0, 1.0 - sum(out.values()))
    return out


def contact_sheet(crops: list[np.ndarray], captions: list[str], path: Path, cols: int = 4, scale: int = 2,
                  cell_w: int = 300, title: str = "") -> None:
    """Crops on one PNG with a caption under each (scaled `scale` x, nearest px), `cols` per row."""
    cell_h = HEIGHT * scale + 34
    rows = math.ceil(len(crops) / cols)
    top = 30 if title else 0
    sheet = np.full((top + rows * cell_h, cols * cell_w), 255, np.uint8)
    if title:
        cv2.putText(sheet, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    for k, (c, cap) in enumerate(zip(crops, captions)):
        r, q = divmod(k, cols)
        y, x = top + r * cell_h, q * cell_w
        big = cv2.resize(c, (c.shape[1] * scale, c.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        big = big[:, :cell_w - 6]
        sheet[y + 2:y + 2 + big.shape[0], x + 3:x + 3 + big.shape[1]] = big
        cv2.rectangle(sheet, (x + 2, y + 1), (x + 3 + big.shape[1], y + 2 + big.shape[0]), 160, 1)
        for i, line in enumerate(cap.split("\n")[:2]):
            cv2.putText(sheet, line[:46], (x + 4, y + HEIGHT * scale + 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        0, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)
