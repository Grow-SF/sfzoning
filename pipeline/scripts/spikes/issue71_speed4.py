"""Issue #71: candidate fast networks.  Fold k x k px into channels (pixel_unshuffle), run a U-Net at 1/k the size
with more channels, unfold the output (pixel_shuffle): one output per scan px, nothing discarded.  Times a training
step (fwd+bwd+Adam) at the real batch shapes, and each candidate's reach in page px.

    uv run python scripts/spikes/issue71_speed4.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfzmaps import gpu
from sfzmaps.nets.unet import UNet, reach

DEV = "mps"


class Folded(nn.Module):
    def __init__(self, in_ch, k, width, levels):
        super().__init__()
        self.k = k
        self.net = UNet(in_ch * k * k, k * k, width, levels)

    def forward(self, x):
        return F.pixel_shuffle(self.net(F.pixel_unshuffle(x, self.k)), self.k)


def bench_step(net, x, n=10, warm=3):
    net.to(DEV).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t = torch.rand(x.shape[0], 1, *x.shape[2:], device=DEV)

    def f():
        y = net(x)
        loss = F.binary_cross_entropy_with_logits(y, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for _ in range(warm):
        f()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    with gpu.gpu_turn(True, DEV):
        page = torch.rand(4, 1, 1024, 1024, device=DEV)
        print(f"page pass as built  UNet(w8,L9) @1024: {bench_step(UNet(1, 1, 8, 9), page):5.0f} ms, reach {reach(9)}")
        for k, w, lv in [(4, 32, 7), (4, 16, 7), (8, 32, 6), (2, 16, 8)]:
            n = Folded(1, k, w, lv)
            print(f"page pass folded k={k} w={w} L={lv}: {bench_step(n, page):5.0f} ms, "
                  f"reach {k * reach(lv)} page px, {sum(p.numel() for p in n.parameters()) / 1e6:.2f} M weights")
        crop = torch.rand(4, 2, 768, 768, device=DEV)
        print(f"pixel pass as built UNet(w16,L5) @768: {bench_step(UNet(2, 1, 16, 5), crop):5.0f} ms, reach {reach(5)}")
        for k, w, lv in [(2, 32, 4), (2, 32, 5), (4, 32, 4), (4, 64, 3)]:
            n = Folded(2, k, w, lv)
            print(f"pixel pass folded k={k} w={w} L={lv}: {bench_step(n, crop):5.0f} ms, "
                  f"reach {k * reach(lv)} page px, {sum(p.numel() for p in n.parameters()) / 1e6:.2f} M weights")


if __name__ == "__main__":
    main()
