"""Where and how torch computes for the finder: the device to use by default and torch settled for repeatable
results.  Moved here unchanged from the 1921 hatching reader's trainer (`solver.hatchnet.train`, branch
`hatch-experiments-archive`).  Reads and writes nothing."""
from __future__ import annotations

import torch

THREADS = 4


def default_device() -> str:
    """"mps" where available, else "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def settle_torch() -> None:
    """Pin torch's threads and make it use deterministic algorithms only (it raises on an op that has none)."""
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
