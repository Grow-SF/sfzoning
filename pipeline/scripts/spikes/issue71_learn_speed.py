"""Issue #71 (frame crop plan, Task 5 step 5): ms per whole-page training step of `frame.learn.train` on the
labelled pages, on the Mac's GPU, with the cpu side prefetched and not.

A 3-step warm-up first (mps compiles its kernels on first use), then 20 steps with prefetch and without, twice each,
alternating, each checked every 5 steps, so `train` logs the ms per step of steps 1-5 (the first page's cpu side not
yet overlapped) and of three steady stretches.  The same seed draws the same pages in every run, so the runs are paired.
Then where one step's time goes on the largest page (`breakdown`).  Labels that do not load (mid-edit) are skipped.
Throwaway measurement.

    uv run python scripts/spikes/issue71_learn_speed.py
"""
import dataclasses
import logging
import time

import numpy as np
import torch

from sfzmaps import gpu
from sfzmaps.frame import data, learn, model

DEV = "mps"


def pages() -> list[learn.Page]:
    out = []
    for series, sheet in data.labelled_sheets():
        try:
            lum = data.load_luminance(data.page_path(series, sheet))
            out.append(learn.Page((series, sheet), lum, data.inside_mask(data.load_label(series, sheet), lum.shape)))
        except Exception as e:  # noqa: BLE001 - a label mid-edit is skipped, the timing needs no particular page
            print(f"skipped {series}/{sheet}: {type(e).__name__}: {e}")
    return out


def timed(fn, n=5) -> float:
    """Median ms of `fn()` over `n` runs after one warm-up, the GPU synchronised around each."""
    fn()
    ts = []
    for _ in range(n):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def breakdown(pg: learn.Page) -> None:
    """Where one step's time goes on `pg`: the cpu side (cut and scale), the device side (turns, photometric
    changes, blur, padding) and the net's step (forward, backward, Adam), each alone."""
    src = learn._Source.of(pg)
    cut, aug = (40, 30, 20, 10), learn.Aug(1, True, 1.2, 1.1, 0.05, 1.0)
    lum, inside = learn._cut_and_scale(src, cut, 1.07)
    net = model.new_net().to(DEV).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, t, w = learn._on_device(lum, inside, aug, 0.1, DEV)

    def step():
        loss = learn._loss(net(x), t, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"breakdown on {pg.id} {pg.lum.shape} (scaled 1.07 to {lum.shape}, padded to {tuple(x.shape[-2:])}):")
    print(f"  cpu cut + scale 1.07       {timed(lambda: learn._cut_and_scale(src, cut, 1.07)):6.0f} ms")
    print(f"  cpu cut, scale 1           {timed(lambda: learn._cut_and_scale(src, cut, 1.0)):6.0f} ms")
    print(f"  device side (blurred)      {timed(lambda: learn._on_device(lum, inside, aug, 0.1, DEV)):6.0f} ms")
    unblurred = dataclasses.replace(aug, sigma=0.0)
    print(f"  device side (not blurred)  {timed(lambda: learn._on_device(lum, inside, unblurred, 0.1, DEV)):6.0f} ms")
    print(f"  net step                   {timed(step):6.0f} ms")
    device_pieces(lum, inside, aug)


def device_pieces(lum: np.ndarray, inside: np.ndarray, aug: learn.Aug) -> None:
    """The device side's pieces alone, and moving the uint8 arrays instead of the float32 ones."""
    lu, iu = torch.from_numpy(lum).to(DEV), torch.from_numpy(inside).to(DEV)
    lf = lu.float() / 255.0
    tf, wf = 1.0 - iu.float(), 0.1 + (1.0 - lf)

    def move(a):
        return torch.rot90(torch.flip(a, (-1,)), 1, (-2, -1)).contiguous()

    def photometric():
        y = lf.clamp(0.0, 1.0) ** aug.gamma
        return ((y - 0.5) * aug.contrast + 0.5 + aug.brightness).clamp(0.0, 1.0)

    def padded():
        h, w = lf.shape
        m = model.multiple()
        x = torch.full((1, 1, -(-h // m) * m, -(-w // m) * m), 1.0, device=DEV)
        x[0, 0, :h, :w] = lf
        return x
    rows = [
        ("upload 2 uint8 arrays", lambda: (torch.from_numpy(lum).to(DEV), torch.from_numpy(inside).to(DEV))),
        ("float, target, weight", lambda: (lu.float() / 255.0, 1.0 - iu.float(), 0.1 + (1.0 - lu.float() / 255.0))),
        ("turn + flip 3 float32", lambda: (move(lf), move(tf), move(wf))),
        ("turn + flip 2 uint8", lambda: (move(lu), move(iu))),
        ("photometric", photometric),
        ("blur sigma 1", lambda: learn._blur(lf, 1.0)),
        ("pad 1 array", padded),
    ]
    for name, fn in rows:
        print(f"    {name:26s} {timed(fn):6.0f} ms")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    pgs = pages()
    print(f"{len(pgs)} pages loaded in {time.perf_counter() - t0:.0f} s; largest "
          f"{max(p.lum.shape for p in pgs)}, mean {sum(p.lum.size for p in pgs) / len(pgs) / 1e6:.1f} Mpx")
    with gpu.gpu_turn(True, DEV):
        learn.train(pgs, learn.Recipe(steps=3, check_every=3), DEV, label="warm-up")
        for prefetch in (True, False, True, False):
            t0 = time.perf_counter()
            learn.train(pgs, learn.Recipe(steps=20, check_every=5, prefetch=prefetch), DEV,
                        label=f"prefetch {'on' if prefetch else 'off'}")
            print(f"prefetch {prefetch}: {time.perf_counter() - t0:.1f} s for the whole call (setup included)")
        breakdown(max(pgs, key=lambda p: p.lum.size))


if __name__ == "__main__":
    main()
