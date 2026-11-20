"""Issue #71 speed check: where does a joint training step's time go?

Times, on real labelled pages, for one step of `sfzmaps.frame.train.train_joint` (both passes together: the page
pass's priors, the two batches, forward, backward, Adam):

- draw: the cpu's part of a step (`_coarse_draw` and `_fine_draw`: the random values and the crops), alone;
- priors: the page pass in eval mode on the drawn pages (`_page_priors`), synced;
- make: the device's part of both batches (`_coarse_make`, `_fine_make`: the prior windows, scaling, turns, flips,
  photometric changes, blur), synced;
- device step: priors, make, both forwards, backward and Adam, synced;
- loop: the training loop as `train_joint` runs it, prefetch on (the draw in a background thread), per step, synced
  at the end only;
- one validation check on 3 pages (the pixel pass only near the outline).

    uv run python scripts/spikes/issue71_speed.py [--device mps] [--steps 20]
"""

import argparse
import math
import time

import torch

from sfzmaps import gpu
from sfzmaps.frame import data, predict, train
from sfzmaps.frame import run as frun

WARM = 3


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    recipe = train.Recipe()
    sheets = data.labelled_sheets()[:6]
    t0 = time.perf_counter()
    pages = [frun._labelled_page(s) for s in sheets]
    print(f"load {len(pages)} pages: {time.perf_counter() - t0:.1f} s")
    with gpu.gpu_turn(True, a.device):
        dev = a.device
        rng = train._settle(0)
        t0 = time.perf_counter()
        found = [train._coarse_arrays(pg, recipe.paper_weight) for pg in pages]
        scales = [s for _, s in found]
        arrays = [tuple(torch.from_numpy(x.copy()).to(dev) for x in arr) for arr, _ in found]
        aux = [train._fine_aux(pg) for pg in pages]
        print(f"page arrays and crop aids: {time.perf_counter() - t0:.1f} s")
        coarse, fine = predict.new_coarse_net().to(dev).train(), predict.new_fine_net().to(dev).train()
        opt = torch.optim.Adam([*coarse.parameters(), *fine.parameters()], lr=recipe.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / recipe.steps)))

        def draw():
            return train._coarse_draw(rng, scales, recipe.batch), train._fine_draw(rng, pages, aux, recipe.batch)

        def step(spec):
            cspec, fspec = spec
            with torch.no_grad():
                priors = train._page_priors(coarse, arrays, (c.page for c in fspec), dev)
                xc, tc, wc = train._coarse_make(cspec, arrays, dev)
                xf, tf, wf = train._fine_make(fspec, priors, recipe.paper_weight, dev)
            losses = torch.stack([train._loss(coarse(xc), tc, wc), train._loss(fine(xf), tf, wf)])
            opt.zero_grad(set_to_none=True)
            losses.sum().backward()
            opt.step()
            sched.step()

        t_draw = t_prior = t_make = t_step = 0.0
        for i in range(a.steps + WARM):
            t0 = time.perf_counter()
            spec = draw()
            t1 = time.perf_counter()
            with torch.no_grad():
                priors = train._page_priors(coarse, arrays, (c.page for c in spec[1]), dev)
                sync(dev)
                t2 = time.perf_counter()
                train._coarse_make(spec[0], arrays, dev)
                train._fine_make(spec[1], priors, recipe.paper_weight, dev)
                sync(dev)
            t3 = time.perf_counter()
            step(spec)
            sync(dev)
            t4 = time.perf_counter()
            if i >= WARM:
                t_draw += t1 - t0
                t_prior += t2 - t1
                t_make += t3 - t2
                t_step += t4 - t3
        n = a.steps
        sync(dev)
        t0 = time.perf_counter()
        with train._batches(draw, n, True) as batches:
            for spec in batches:
                step(spec)
        sync(dev)
        t_loop = time.perf_counter() - t0
        print(f"JOINT per step: draw {t_draw / n * 1000:.0f} ms (cpu), priors {t_prior / n * 1000:.0f} ms, "
              f"make {t_make / n * 1000:.0f} ms, device step {t_step / n * 1000:.0f} ms (priors + make + fwd + bwd + "
              f"Adam); loop with prefetch {t_loop / n * 1000:.0f} ms -> {recipe.steps} steps "
              f"{t_loop / n * recipe.steps / 60:.1f} min")
        check = train._validation(pages[:3], dev)
        t0 = time.perf_counter()
        check(coarse, fine)
        print(f"validation check (3 pages, pixel pass near the outline only): {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
