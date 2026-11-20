"""Issue #71: which op's backward is slow on mps?  Forward vs forward+backward per op, at the page pass's level-0
shape (4 x 8 x 1024 x 1024) and its first layer.

    uv run python scripts/spikes/issue71_speed3.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfzmaps import gpu

DEV = "mps"
import os
CONTIG = os.environ.get("CONTIG") == "1"


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def fb(op, *shape):
    x = torch.rand(*shape, device=DEV, requires_grad=True)

    def fwd():
        with torch.no_grad():
            op(x)

    def both():
        y = op(x)
        y.backward(torch.ones_like(y) if CONTIG else None) if CONTIG else y.sum().backward()
    return bench(fwd), bench(both)


def main():
    torch.manual_seed(0)
    with gpu.gpu_turn(True, DEV):
        rows = {
            "conv 1->8 (first layer)": (nn.Conv2d(1, 8, 3, padding=1).to(DEV), (4, 1, 1024, 1024)),
            "conv 8->8": (nn.Conv2d(8, 8, 3, padding=1).to(DEV), (4, 8, 1024, 1024)),
            "conv 24->8 (up block, concat in)": (nn.Conv2d(24, 8, 3, padding=1).to(DEV), (4, 24, 1024, 1024)),
            "conv 8->1 1x1 head": (nn.Conv2d(8, 1, 1).to(DEV), (4, 8, 1024, 1024)),
            "batch norm": (nn.BatchNorm2d(8).to(DEV).train(), (4, 8, 1024, 1024)),
            "relu": (nn.ReLU(), (4, 8, 1024, 1024)),
            "max_pool2d 2": (lambda t: F.max_pool2d(t, 2), (4, 8, 1024, 1024)),
            "nearest upsample x2": (lambda t: F.interpolate(t, scale_factor=2, mode="nearest"), (4, 16, 512, 512)),
            "nearest upsample to size": (lambda t: F.interpolate(t, size=(1024, 1024), mode="nearest"),
                                         (4, 16, 512, 512)),
            "repeat_interleave x2 (upsample alternative)": (
                lambda t: t.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3), (4, 16, 512, 512)),
            "conv_transpose 2x2 s2 16->16": (nn.ConvTranspose2d(16, 16, 2, stride=2).to(DEV), (4, 16, 512, 512)),
        }
        for name, (op, shape) in rows.items():
            f, b = fb(op, *shape)
            print(f"{name:45s} fwd {f:6.1f} ms   fwd+bwd {b:6.1f} ms   bwd/fwd {b / max(f, 1e-6) - 1:5.1f}x")
        for det in (True, False):
            torch.use_deterministic_algorithms(det)
            f, b = fb(lambda t: F.interpolate(t, size=(1024, 1024), mode="nearest"), 4, 16, 512, 512)
            print(f"nearest upsample, deterministic={det}: fwd+bwd {b:.1f} ms")


if __name__ == "__main__":
    main()
