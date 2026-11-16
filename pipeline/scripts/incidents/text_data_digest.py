"""A digest of everything the text detector's stages read from `textnet.data`, for comparing two versions of the code
(run once per version, `diff` the outputs; CPU only).

The stage `solver-text` trains on `load_sheets(TEXT_TRAIN_SERIES, hold_out_validation=False, placement=False)`
(`centres` per sheet, then `crop_targets`) and `solver-text-scores` scores `load_sheets(TEXT_TEST_SERIES, False)`
(`scored_px`, `page_text`, `page_ink`); this hashes, per sheet, every array of the SheetData (and what it holds out),
its centres, its scored px, its text and unknown px, and a fixed set of crops (several sizes, scales and turns, centred
on its own centres), and prints one line per sheet.

    SFZMAPS_DATA_DIR=<this checkout's data> PYTHONPATH=<src of the version> python text_data_digest.py > digest.txt"""
import dataclasses
import hashlib

import numpy as np

import sfzmaps
from sfzmaps.solver import paths
from sfzmaps.solver.textnet import data

SKIP = {"held", "hold_out", "point_radius"}


def h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return m.hexdigest()[:16]


def digest(sd) -> str:
    parts = []
    for f in dataclasses.fields(sd):
        if f.name in SKIP:
            continue
        v = getattr(sd, f.name)
        parts.append(f"{f.name}={h(v) if isinstance(v, np.ndarray) else v!r}")
    held_none = (not sd.hold_out) if hasattr(sd, "hold_out") else (sd.held == ())
    parts.append(f"holds_nothing={held_none} radius={getattr(sd, 'point_radius', data.READ_POINT_RADIUS)}")
    on_text, on_ink = data.centres(sd, 4)
    parts.append(f"centres={h(on_text, on_ink)}:{len(on_text)}:{len(on_ink)}")
    text, unknown = data.page_text(sd)
    parts.append(f"text={h(text, unknown)} ink={h(data.page_ink(sd))}")
    if sd.world.size:
        scored = np.concatenate([data.in_validation(sd, y0, 0, min(data.BAND, sd.height - y0), sd.width)
                                 for y0 in range(0, sd.height, data.BAND)])
        parts.append(f"validation={h(scored)}")
    rng = np.random.default_rng(0)
    crops = []
    for pool in (on_text, on_ink):
        for i in rng.choice(len(pool), size=min(12, len(pool)), replace=False) if len(pool) else []:
            y, x = pool[i]
            for size, scale, turns in ((256, 1.0, 0), (256, 0.7, 1), (384, 1.4, 3)):
                c = data.crop_targets(sd, int(y), int(x), size, scale, turns)
                crops.append(h(*[c[k] for k in sorted(c)]))
    parts.append(f"crops={h(np.array([int(c, 16) % 2**31 for c in crops]))}:{len(crops)}")
    return " ".join(parts)


def main():
    print("sfzmaps from", sfzmaps.__file__)
    train = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False, placement=False)
    test = data.load_sheets(paths.TEXT_TEST_SERIES, hold_out_validation=False)
    for kind, sheets in (("train", train), ("test", test)):
        for sd in sheets:
            print(kind, sd.series, sd.sheet, digest(sd), flush=True)


if __name__ == "__main__":
    main()
