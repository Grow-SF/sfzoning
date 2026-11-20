"""Issue #71 speed check: where does a training step's time go?

Times, on real labelled pages: building one batch (draw + augment) on cpu, the device step (copy, forward, backward,
Adam) with the device synchronised, and one validation check, for the page pass and the pixel pass.

    uv run python scripts/spikes/issue71_speed.py [--device mps] [--steps 30]
"""

import argparse
import time

import numpy as np
import torch

from sfzmaps import gpu
from sfzmaps.frame import data, predict, train
from sfzmaps.frame import run as frun


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def time_fit(net, draw, device, steps, recipe):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)
    t_draw = t_step = 0.0
    for i in range(steps + 3):
        t0 = time.perf_counter()
        arrs = draw()
        t1 = time.perf_counter()
        x, t, w = (torch.from_numpy(a).to(device) for a in arrs)
        loss = train._loss(net(x), t, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(device)
        t2 = time.perf_counter()
        if i >= 3:  # warm-up excluded
            t_draw += t1 - t0
            t_step += t2 - t1
    return t_draw / steps, t_step / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--nondet", action="store_true", help="deterministic algorithms off (speed comparison only)")
    ap.add_argument("--coarse-side", type=int, default=None)
    ap.add_argument("--coarse-levels", type=int, default=None)
    ap.add_argument("--only-coarse", action="store_true")
    ap.add_argument("--tile", type=int, default=None)
    a = ap.parse_args()
    if a.coarse_side:
        predict.COARSE_SIDE = a.coarse_side
    if a.tile:
        predict.TILE = a.tile
    if a.coarse_levels:
        predict.COARSE_LEVELS = a.coarse_levels
    recipe = train.Recipe()
    sheets = data.labelled_sheets()[:6]
    t0 = time.perf_counter()
    pages = [frun.load_page(s) if hasattr(frun, "load_page") else None for s in sheets]
    if pages[0] is None:
        pages = []
        for s in sheets:
            lum = data.load_luminance(data.page_path(*s))
            mask = data.map_mask(data.load_label(*s), lum.shape)
            pages.append(train.Page(id=s, lum=lum, mask=mask, prior=None))
    print(f"load 6 pages: {time.perf_counter() - t0:.1f} s")
    with gpu.gpu_turn(True, a.device):
        rng = train._settle(0)
        if a.nondet:
            torch.use_deterministic_algorithms(False)
        t0 = time.perf_counter()
        coarse = [train._coarse_arrays(pg, recipe.paper_weight) for pg in pages]
        print(f"coarse arrays: {time.perf_counter() - t0:.1f} s")
        net = predict.new_coarse_net()
        d, s = time_fit(net, lambda: train._coarse_batch(rng, coarse, recipe.batch), a.device, a.steps, recipe)
        print(f"PAGE PASS per step: draw {d * 1000:.0f} ms, device step {s * 1000:.0f} ms "
              f"-> 4000 steps {(d + s) * 4000 / 60:.1f} min")
        t0 = time.perf_counter()
        train._val_loss(pages[:3], lambda pg: predict.predict_coarse(net.eval(), pg.lum, a.device))
        print(f"page-pass validation check (3 pages): {time.perf_counter() - t0:.1f} s")
        if a.only_coarse:
            return
        for pg in pages:
            pg.prior = predict.predict_coarse(net, pg.lum, a.device)
        near = [train._near_outline(pg.mask) for pg in pages]
        fnet = predict.new_fine_net()
        d, s = time_fit(fnet, lambda: train._fine_batch(rng, pages, near, recipe.batch, recipe.paper_weight),
                        a.device, a.steps, recipe)
        print(f"PIXEL PASS per step: draw {d * 1000:.0f} ms, device step {s * 1000:.0f} ms "
              f"-> 4000 steps {(d + s) * 4000 / 60:.1f} min")
        t0 = time.perf_counter()
        train._val_loss(pages[:3], lambda pg: predict.predict_tiles(fnet.eval(), pg.lum, pg.prior, a.device))
        print(f"pixel-pass validation check (3 pages): {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
