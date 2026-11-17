"""Task 12's comparison runs (item 7), for Task 8's runner to call when the kept detector exists: the word-heads detector
trained with the scan's grey levels as its input (`Recipe.input="grey"`), and separately at full size (`half_size` off),
each beside the kept recipe on fold NE at seeds 0 and 1.  Nothing here runs by import; it is the same `run_detector.
train_run` with the recipe changed (and, for grey, the sheets loaded with their grey levels).

The keep rule (written before the runs): an item is kept if, with the kept detector and reader on NE at two seeds, the
headline rises beyond the interval and the per-ink gate loses nothing beyond noise.

Needs `textnet.predict` to read grey when the model's recipe says so (the detector's tiles are the page's grey over 255,
not its ink): until that lands, `variant.run`'s prediction of a grey model would be fed ink, so a grey run is not valid.
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_detector                                               # noqa: E402
from sfzmaps.solver import paths                                  # noqa: E402
from sfzmaps.solver.textnet import data, train                    # noqa: E402

ITEMS = ("grey", "full")


def item_recipe(kept: train.Recipe, item: str) -> train.Recipe:
    """The kept recipe with one item changed: "grey" its input the grey levels; "full" the network at full size."""
    if item == "grey":
        return dataclasses.replace(kept, input="grey")
    if item == "full":
        return dataclasses.replace(kept, half_size=False)
    raise ValueError(f"item is one of {', '.join(ITEMS)}, not {item!r}")


def item_run(item: str, quarter: str, seed: int, kept: train.Recipe | None = None, **kwargs) -> Path:
    """Train one item's detector on `quarter` at `seed` into `run_detector.RUNS` (`run_detector.train_run`; `kwargs`
    as it takes them, e.g. device, doubt, root).  The sheets are loaded with their grey levels for "grey"."""
    kept = run_detector.word_recipe() if kept is None else kept
    recipe = item_recipe(kept, item)
    if item == "grey" and kwargs.get("sheets") is None:
        kwargs["sheets"] = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False, grey=True)
    return run_detector.train_run(f"input-{item}", quarter, seed, recipe=recipe, **kwargs)
