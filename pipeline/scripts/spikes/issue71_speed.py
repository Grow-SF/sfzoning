"""Issue #71 speed check: where does a training step's time go?

Times, on real labelled pages, for the page pass and the pixel pass as `sfzmaps.frame.train` trains them (folded
U-Nets, batches built on the device, prefetch):

- draw: the cpu's part of a batch (`_coarse_draw` / `_fine_draw`: the random values and the crops), alone;
- make: the device's part (`_coarse_make` / `_fine_make`: scaling, turns, flips, photometric changes, blur), synced;
- device step: make, forward, backward and Adam, synced;
- loop: the training loop as `_fit` runs it, prefetch on (the draw in a background thread), per step, synced at the
  end only;
- one validation check on 3 pages.

    uv run python scripts/spikes/issue71_speed.py [--device mps] [--steps 20]
"""

import argparse
import time

import torch

from sfzmaps import gpu
from sfzmaps.frame import data, predict, train
from sfzmaps.frame import run as frun

WARM = 3


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def time_pass(name, net, draw, make, device, steps, recipe):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)

    def step(spec):
        with torch.no_grad():
            x, t, w = make(spec)
        loss = train._loss(net(x), t, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t_draw = t_make = t_step = 0.0
    for i in range(steps + WARM):
        t0 = time.perf_counter()
        spec = draw()
        t1 = time.perf_counter()
        make(spec)
        sync(device)
        t2 = time.perf_counter()
        step(spec)
        sync(device)
        t3 = time.perf_counter()
        if i >= WARM:
            t_draw += t1 - t0
            t_make += t2 - t1
            t_step += t3 - t2
    sync(device)
    t0 = time.perf_counter()
    with train._batches(draw, steps, True) as batches:
        for spec in batches:
            step(spec)
    sync(device)
    t_loop = time.perf_counter() - t0
    print(f"{name} per step: draw {t_draw / steps * 1000:.0f} ms (cpu), make {t_make / steps * 1000:.0f} ms, "
          f"device step {t_step / steps * 1000:.0f} ms (make + fwd + bwd + Adam); "
          f"loop with prefetch {t_loop / steps * 1000:.0f} ms -> {recipe.steps} steps "
          f"{t_loop / steps * recipe.steps / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--nondet", action="store_true", help="deterministic algorithms off (speed comparison only)")
    ap.add_argument("--only-coarse", action="store_true")
    a = ap.parse_args()
    recipe = train.Recipe()
    sheets = data.labelled_sheets()[:6]
    t0 = time.perf_counter()
    pages = [frun._labelled_page(s) for s in sheets]
    print(f"load {len(pages)} pages: {time.perf_counter() - t0:.1f} s")
    with gpu.gpu_turn(True, a.device):
        rng = train._settle(0)
        if a.nondet:
            torch.use_deterministic_algorithms(False)
        t0 = time.perf_counter()
        coarse = [train._coarse_arrays(pg, recipe.paper_weight) for pg in pages]
        scales = [s for _, s in coarse]
        on_device = [tuple(torch.from_numpy(arr.copy()).to(a.device) for arr in arrays) for arrays, _ in coarse]
        print(f"coarse arrays: {time.perf_counter() - t0:.1f} s")
        net = predict.new_coarse_net()
        time_pass("PAGE PASS", net, lambda: train._coarse_draw(rng, scales, recipe.batch),
                  lambda spec: train._coarse_make(spec, on_device, a.device), a.device, a.steps, recipe)
        val_loss = train._val_loss(pages[:3])
        t0 = time.perf_counter()
        val_loss(lambda pg: predict.predict_coarse(net, pg.lum, a.device))
        print(f"page-pass validation check (3 pages): {time.perf_counter() - t0:.1f} s")
        if a.only_coarse:
            return
        for pg in pages:
            pg.prior = predict.predict_coarse(net, pg.lum, a.device).astype("float16")
        aux = [train._fine_aux(pg) for pg in pages]
        fnet = predict.new_fine_net()
        time_pass("PIXEL PASS", fnet, lambda: train._fine_draw(rng, pages, aux, recipe.batch),
                  lambda spec: train._fine_make(spec, recipe.paper_weight, a.device), a.device, a.steps, recipe)
        t0 = time.perf_counter()
        val_loss(lambda pg: predict.predict_tiles(fnet, pg.lum, pg.prior, a.device))
        print(f"pixel-pass validation check (3 pages): {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
