"""Issue #71: what is the GPU doing in a page-pass step?  Micro-benchmarks on synthetic tensors of the real shapes.

    uv run python scripts/spikes/issue71_speed2.py
"""

import time

import torch
import torch.nn as nn

from sfzmaps import gpu
from sfzmaps.nets.unet import UNet

DEV = "mps"


def sync():
    torch.mps.synchronize()


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000


def step_fn(net, x, amp=False, fwd_only=False):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t = torch.rand_like(x[:, :1])

    def f():
        if fwd_only:
            with torch.no_grad():
                net(x)
            return
        with torch.autocast("mps", dtype=torch.float16, enabled=amp):
            y = net(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return f


def no_bn(net):
    for name, m in list(net.named_modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, nn.BatchNorm2d):
                setattr(m, cname, nn.Identity())
    return net


def main():
    torch.manual_seed(0)
    with gpu.gpu_turn(True, DEV):
        x = torch.rand(4, 1, 1024, 1024, device=DEV)
        base = UNet(1, 1, 8, 9).to(DEV).train()
        print(f"page pass as built (4x1024^2, w8, L9): fwd+bwd+adam {bench(step_fn(base, x)):.0f} ms; "
              f"fwd only {bench(step_fn(base, x, fwd_only=True)):.0f} ms")
        torch.use_deterministic_algorithms(True)
        print(f"  deterministic on: {bench(step_fn(base, x)):.0f} ms")
        torch.use_deterministic_algorithms(False)
        print(f"  no batch norm: {bench(step_fn(no_bn(UNet(1, 1, 8, 9).to(DEV).train()), x)):.0f} ms")
        cl = UNet(1, 1, 8, 9).to(DEV).train().to(memory_format=torch.channels_last)
        print(f"  channels_last: {bench(step_fn(cl, x.to(memory_format=torch.channels_last))):.0f} ms")
        print(f"  float16 autocast: {bench(step_fn(UNet(1, 1, 8, 9).to(DEV).train(), x, amp=True)):.0f} ms")
        print(f"  width 4: {bench(step_fn(UNet(1, 1, 4, 9).to(DEV).train(), x)):.0f} ms; "
              f"width 16: {bench(step_fn(UNet(1, 1, 16, 9).to(DEV).train(), x)):.0f} ms")
        for side in (512, 256):
            xs = torch.rand(4, 1, side, side, device=DEV)
            lv = 8 if side == 512 else 7
            print(f"  side {side} (L{lv}): {bench(step_fn(UNet(1, 1, 8, lv).to(DEV).train(), xs)):.0f} ms")
        # one conv layer at full resolution: achieved throughput
        conv = nn.Conv2d(8, 8, 3, padding=1).to(DEV)
        a = torch.rand(4, 8, 1024, 1024, device=DEV)
        ms = bench(lambda: conv(a), n=20)
        flops = 4 * 1024 * 1024 * 8 * 8 * 9 * 2
        byts = 2 * a.numel() * 4
        print(f"one conv 8->8 3x3 on 4x8x1024^2: {ms:.1f} ms = {flops / ms / 1e6:.0f} GFLOP/s, "
              f"{byts / ms / 1e6:.0f} GB/s (in+out)")
        bn = nn.BatchNorm2d(8).to(DEV).train()
        print(f"one batch norm on the same: {bench(lambda: bn(a), n=20):.1f} ms; "
              f"relu: {bench(lambda: torch.relu(a), n=20):.1f} ms")
        conv64 = nn.Conv2d(64, 64, 3, padding=1).to(DEV)
        b = torch.rand(4, 64, 128, 128, device=DEV)
        ms64 = bench(lambda: conv64(b), n=20)
        print(f"one conv 64->64 on 4x64x128^2 (same FLOPs): {ms64:.1f} ms = "
              f"{4 * 128 * 128 * 64 * 64 * 9 * 2 / ms64 / 1e6:.0f} GFLOP/s")
    # CPU, for comparison
    torch.set_num_threads(8)
    xc = torch.rand(4, 1, 1024, 1024)
    netc = UNet(1, 1, 8, 9).train()
    opt = torch.optim.Adam(netc.parameters())
    t0 = time.perf_counter()
    for _ in range(3):
        loss = netc(xc).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"cpu (8 threads) page pass step: {(time.perf_counter() - t0) / 3 * 1000:.0f} ms")


if __name__ == "__main__":
    main()
