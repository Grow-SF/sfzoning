"""The `solver-labels` stage: what the solver knows about the printed labels and the ink around them, written into
data/solver/labels/.  In order: the trust of every reading (`trust`, readings_{q}.npz and trust.json), the side of
the street a code printed on street ground speaks for (`streetside`, streetside_{q}.npz, from each print's top
reading), what a lot's size says about its code (`lotsize`, lotsize.npz) and the ink's strokes (`strokes`,
strokes.npz)."""
import time

from sfzmaps.solver import lotsize, streetside, strokes, trust


def build_labels() -> dict:
    """Write every file of data/solver/labels/.  Returns each part's summary and the seconds each took."""
    out, seconds = {}, {}
    for name, build in (("trust", trust.build_readings), ("streetside", streetside.build_streetsides),
                        ("lotsize", lotsize.build_lotsize), ("strokes", strokes.build_strokes)):
        t0 = time.perf_counter()
        out[name] = build()
        seconds[name] = round(time.perf_counter() - t0, 1)
    return {**out, "seconds": seconds}
