"""Where the text reader's three stages write, and the files they share.

  data/solver/text-crops/   (solver-text-crops)
    crops.npz, crops.parquet   every real crop and its row (`crops`)
    answered.parquet           every answered label of the owner's code check: its point, answer, quarter, and the
                               row of its crop at Terra's box (-1 without one)
    font.npz, prior.json       the template reader's font and prior, learned from the training labels (`synth`)
    synth_<q>.npz, .parquet    quarter q's reader's synthetic code crops
    summary.json
  data/solver/text-train/   (solver-text-train)
    <q>.pt, <q>_direction.pt   quarter q's reader and 0/180 classifier
    held.json                  per quarter: what was trained on, the seeds and recipes
    log.json                   the training logs
  data/solver/text-read/    (solver-text-read)
    reads.npz, reads.parquet   every word box's reading (`read`)
    answered.parquet           every answered label's readings, at Terra's box and at our box
    measures.json

A crop is uint8, HEIGHT px tall, paper 255, reading left to right.  A set of crops is stored flat: crop i is
`flat[:, offsets[i]:offsets[i + 1]]`.  Every .npz is written with fixed dates (`save_npz`): the same arrays, the same
bytes.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sfzmaps.solver import paths

CROPS_DIR = paths.SOLVER_DIR / "text-crops"
TRAIN_DIR = paths.SOLVER_DIR / "text-train"
READ_DIR = paths.SOLVER_DIR / "text-read"
NPZ_DATE = (1980, 1, 1, 0, 0, 0)
TOUCH = tuple(f"touch_{q}" for q in paths.QUARTERS) + ("touch_blocks",)   # the held-out areas a crop touches


def save_npz(f: Path, **arrays) -> None:
    """A compressed .npz whose members are dated NPZ_DATE (numpy dates them now)."""
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        for name, a in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", NPZ_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with z.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(a), allow_pickle=False)


class Crops:
    """A set of crops (see the module): `crops[i]` is crop i."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat, self.offsets = flat, offsets

    @classmethod
    def of(cls, crops: list[np.ndarray], height: int) -> "Crops":
        widths = np.array([c.shape[1] for c in crops], np.int64)
        offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        flat = np.concatenate(crops, axis=1) if crops else np.zeros((height, 0), np.uint8)
        return cls(np.ascontiguousarray(flat, np.uint8), offsets)

    @classmethod
    def load(cls, f: Path) -> "Crops":
        with np.load(f) as z:
            return cls(z["flat"], z["offsets"])

    def save(self, f: Path) -> None:
        save_npz(f, flat=self.flat, offsets=self.offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.flat[:, self.offsets[i]:self.offsets[i + 1]]

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.offsets)


def training_rows(rows: pd.DataFrame, q: int, text_only: bool = False) -> np.ndarray:
    """The rows of crops.parquet quarter q's reader may train on: a target (Terra's string in the reader's alphabet,
    or "" for a non-text window), and a footprint touching neither quarter q nor the answered blocks; `text_only`:
    words only."""
    ok = rows["text"].notna().to_numpy() & ~rows[f"touch_{paths.QUARTERS[q]}"].to_numpy(bool) \
        & ~rows["touch_blocks"].to_numpy(bool)
    if text_only:
        ok &= (rows["kind"] != "none").to_numpy()
    return np.flatnonzero(ok)
