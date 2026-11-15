"""Turns on the Mac's GPU (PyTorch's `mps`), one machine-wide lock for every process that computes on it.

Two processes computing on `mps` at once can get wrong numbers back: the text detector beside another process on the
GPU gives a different output for the same step on the same input each time (a logit off by up to 2.5) and now and then
a gradient that is all NaN, while alone on the GPU it gives the same bits every run.  So work on `mps` takes a turn
first: an `fcntl.flock` on LOCK, one file for the whole machine (as the GPU is), whatever the shell or worktree.

  exclusive  the GPU alone (LOCK_EX): the text detector's training and scoring
  shared     beside other shared turns only (LOCK_SH): the sheet warp's fits, which run several sheets at once

An exclusive turn waits until no other turn is held, and a shared turn waits while an exclusive one is.  The CPU takes
no turn.
"""
import contextlib
import fcntl
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

LOCK = Path("/tmp/sfzmaps-mps.lock")


def on_gpu(device) -> bool:
    """Whether `device` (a name such as "mps" or "cpu", or a torch.device) is the Mac's GPU."""
    return str(device).split(":")[0] == "mps"


def take_turn(exclusive: bool, wait: bool = True) -> int | None:
    """Open LOCK (made if missing) and lock it, exclusive or shared.  Returns the open file descriptor, which holds
    the turn until `end_turn`; with `wait` False, None at once when the turn can't be had now.  The file is opened
    read-only: a lock needs no write access, so a lock file another user made works too."""
    fd = os.open(LOCK, os.O_RDONLY | os.O_CREAT, 0o666)
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(fd, mode if wait else mode | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


def end_turn(fd: int) -> None:
    """Give back a turn `take_turn` gave."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextlib.contextmanager
def gpu_turn(exclusive: bool, device="mps"):
    """Hold a turn on the GPU while the body runs, if `device` is "mps" (see the module); log and wait while another
    process's turn is in the way.  Nothing happens for any other device.

    Turns don't nest within one process: each takes its own lock on a new open of LOCK, so a process holding a turn
    that asks for another waits for itself forever when either is exclusive.  Take one turn, around everything the
    process does on the GPU."""
    if not on_gpu(device):
        yield
        return
    fd = take_turn(exclusive, wait=False)
    if fd is None:
        log.info("waiting for the GPU: another process's turn holds %s", LOCK)
        t0 = time.perf_counter()
        fd = take_turn(exclusive)
        log.info("the GPU's turn is ours after %.0f s", time.perf_counter() - t0)
    try:
        yield
    finally:
        end_turn(fd)
