"""Write `data/solver/experiments/textlines/rules.json`: `linerules.rule_accuracy` for the run rule and the chain rule,
on cv2.putText glyphs and (when `data/solver/text-crops/font.npz` exists) on the learned font.  CPU, one thread, $0.

    uv run python scripts/textlines/rules.py
"""
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.codereader import glyphs  # noqa: E402
from sfzmaps.solver.textnet import linerules as L  # noqa: E402

CAP = 27    # the chain rule's run length: the larger cap, px


def main() -> None:
    out = runs.EXPERIMENTS_DIR / "textlines"
    out.mkdir(parents=True, exist_ok=True)
    rules = {"run": L.line_px, "chain": lambda ink, f: L.chain_lines(ink, L.run_px(f), CAP)}
    draws = {"putText": None}
    fp = paths.SOLVER_DIR / "text-crops" / "font.npz"
    if fp.exists():
        font = glyphs.Font.load(fp)

        def draw(text, cap, rng):
            jx, jy = rng.integers(-15, 16, 2)
            pose = glyphs.Pose(L.SIZE / 2 + jx, L.SIZE / 2 + jy, cap / font.size, 0.0)
            return glyphs.render(font, text, pose, (L.SIZE, L.SIZE)) > 0
        draws["font.npz"] = draw
    res = {f"{rn} / {dn}": L.rule_accuracy(r, draw=dr) for dn, dr in draws.items() for rn, r in rules.items()}
    (out / "rules.json").write_text(json.dumps(res, indent=1))
    for key, acc in res.items():
        print("==", key)
        for case, per in acc.items():
            rec = [v["line_recall"] for v in per.values()]
            fal = [v["glyph_false_line"] for v in per.values() if v["glyph_false_line"] is not None] or [float("nan")]
            print(f"{case:40s} recall mean {np.mean(rec):.3f} min {min(rec):.3f} | false-line mean "
                  f"{np.mean(fal):.3f} max {max(fal):.3f}")


if __name__ == "__main__":
    main()
