"""The text reader's inputs (stage `solver-text-crops`, `sfzmaps solver text-crops`): the real crops (`crops`), the
template reader's font and prior, and each quarter's synthetic code crops (`synth`), all from primary sources: the
pages, Terra's saved boxes, the prints and the owner's answers (to hold their blocks out), the lots.

Writes data/solver/text-crops/ (`store.CROPS_DIR`, see `store`): crops.npz and crops.parquet (the words and non-text
windows; columns: series, sheet, word (Terra's word on the sheet, -1 for a window), kind ("none" for a window), terra,
text (the target; null where a character is outside the reader's alphabet), angle, direction, box_x0..box_y1, length,
height, weight, touch_<q> for each quarter, touch_blocks, mx, my, quarter, block); answered.npz and answered.parquet
(the owner's answered labels; `crop` the row of its crop at Terra's box in answered.npz, -1 without one); font.npz
and prior.json; synth_<q>.npz and synth_<q>.parquet (text, code, overlay, overlay_wanted); summary.json.  Built in a
fresh folder and renamed into place.  The .npz files carry no dates.
"""
import json
import logging
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from sfzmaps.solver import paths, runs
from sfzmaps.solver.codereader import labels as L
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import store, synth

log = logging.getLogger(__name__)

TEMPLATE_WORKERS = 8      # threads aligning the template reader's labels (each one thread)

_SYN: dict = {}


def _synth_job(q: int) -> dict:
    return synth.make_synth(q, _SYN["font"], _SYN["prior"], _SYN["rows"], _SYN["specks"])


def save_font(font, f: Path) -> None:
    """`glyphs.Font.save`'s arrays, rewritten without dates (`store.save_npz`)."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "font.npz"
        font.save(t)
        with np.load(t) as z:
            store.save_npz(f, **{k: z[k] for k in z.files})


def build_inputs(workers: int = K.WORKERS, template_workers: int = TEMPLATE_WORKERS) -> dict:
    """Make every input (see the module) and write CROPS_DIR.  Returns the summary."""
    t0 = time.perf_counter()
    labels = L.load_labels(crops=False)
    made = K.make_crops(labels, workers)
    log.info("%d crops, %d answered labels (%d with Terra's box) in %.0f s", len(made["rows"]),
             len(made["answered_rows"]), len(made["answered"]), made["summary"]["seconds"])
    t1 = time.perf_counter()
    font, prior, template = synth.learn_template(L.load_labels(), template_workers)
    log.info("the template reader's font and prior from %d training labels in %.0f s: %d forms over %d codes",
             template["training_labels"], time.perf_counter() - t1, len(prior.forms), len(prior.codes))
    specks = synth.speck_rate(paths.SHEETS)
    _SYN.update(font=font, prior=prior, rows=made["rows"], specks=specks)
    with get_context("fork").Pool(min(workers, len(paths.QUARTERS))) as pool:
        syn = pool.map(_synth_job, range(len(paths.QUARTERS)), chunksize=1)
    part = runs.fresh_folder(store.CROPS_DIR)
    store.Crops.of(made["crops"], K.HEIGHT).save(part / "crops.npz")
    made["rows"].to_parquet(part / "crops.parquet", index=False)
    store.Crops.of(made["answered"], K.HEIGHT).save(part / "answered.npz")
    made["answered_rows"].to_parquet(part / "answered.parquet", index=False)
    save_font(font, part / "font.npz")
    prior.save(part / "prior.json")
    for name, s in zip(paths.QUARTERS, syn):
        store.Crops.of(s["crops"], K.HEIGHT).save(part / f"synth_{name}.npz")
        s["rows"].to_parquet(part / f"synth_{name}.parquet", index=False)
    summary = {"crops": {k: v for k, v in made["summary"].items() if k != "seconds"},
               "template": {k: v for k, v in template.items() if k != "seconds"},
               "specks": specks, "synth": {n: {k: v for k, v in s["summary"].items() if k != "seconds"}
                                           for n, s in zip(paths.QUARTERS, syn)},
               "training_crops": {n: int(len(store.training_rows(made["rows"], q)))
                                  for q, n in enumerate(paths.QUARTERS)}}
    (part / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    runs.replace_folder(part, store.CROPS_DIR)
    seconds = {"crops": made["summary"]["seconds"], **template["seconds"],
               "synth": max(s["summary"]["seconds"] for s in syn), "all": round(time.perf_counter() - t0, 1)}
    return {**summary, "seconds": seconds}
