"""Issue #71 Task 5c: ms/step of the real training loop (`learn.train`, prefetch on) on the 21 labelled pages, "fold 16"
net (FOLD 16, LEVELS 5, WIDTH 32), mps, under the exclusive GPU turn: 200 steps, a check every 50 (each check logs the
ms/step since the last one, the check's own work left out; the last checks are the steady state).

    uv run python scripts/spikes/issue71_train_speed.py <tag>

With FRAME_CODE=<dir>, `frame/model.py` and `frame/learn.py` are loaded from that directory instead (an older
version's, say `git show <rev>:pipeline/src/sfzmaps/frame/learn.py`), the rest of the package as checked out, so the
pages load the same way for both.
"""
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import sfzmaps.frame


def _swap(code: Path) -> None:
    for name in ("model", "learn"):             # model first: learn imports it
        spec = importlib.util.spec_from_file_location(f"sfzmaps.frame.{name}", code / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        setattr(sfzmaps.frame, name, mod)
        spec.loader.exec_module(mod)


if os.environ.get("FRAME_CODE"):
    _swap(Path(os.environ["FRAME_CODE"]))

from sfzmaps import gpu                                 # noqa: E402 - after the swap
from sfzmaps.frame import data, learn, model            # noqa: E402
from sfzmaps.frame import run as fr                     # noqa: E402

STEPS, CHECK = 200, 50


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model.FOLD, model.LEVELS, model.WIDTH = 16, 5, 32
    pages = list(fr.labelled_pages(data.labelled_sheets()).values())
    print(tag, "learn", learn.__file__, "pages", len(pages), "bucket", getattr(model, "BUCKET", None), flush=True)
    with gpu.gpu_turn(True, "mps"):
        t0 = time.perf_counter()
        _, curves = learn.train(pages, learn.Recipe(steps=STEPS, check_every=CHECK), "mps", label=tag)
        wall = time.perf_counter() - t0
    print(tag, "train curve", curves["train_curve"], flush=True)
    print(f"{tag} wall {wall:.1f} s for {STEPS} steps = {wall / STEPS * 1000:.0f} ms/step incl. startup", flush=True)


if __name__ == "__main__":
    main()
