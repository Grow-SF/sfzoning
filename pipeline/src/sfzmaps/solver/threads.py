"""One thread for the maths libraries (BLAS: Accelerate, OpenBLAS or MKL; OpenMP) in the processes that fit the walls,
the placement and the report and that solve the quarters, so that their numbers don't depend on how many cores the
machine has.  The libraries read their thread counts once, as they load (before any command runs: the command line
loads numpy), so a process is pinned only when ONE_THREAD is in its environment from its start.  `sfzmaps solver
walls`, `placement` and `solve` start themselves again in such a process (`run_pinned`) when theirs isn't, and the
solve starts its quarters' processes so."""
import os
import subprocess
import sys
from pathlib import Path

ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
              "MKL_NUM_THREADS": "1"}
# Not read by the maths libraries the numbers depend on, so not part of `pinned`; set beside ONE_THREAD so a pinned
# process and its children start no thread pools of their own (numexpr's; OpenCV's and torch's: `limit_pools`).
QUIET_POOLS = {"NUMEXPR_NUM_THREADS": "1"}


def pinned() -> bool:
    """Whether this process's environment holds ONE_THREAD (as it did when the process started, unless something
    has set it since)."""
    return all(os.environ.get(k) == v for k, v in ONE_THREAD.items())


def limit_pools() -> None:
    """In a pinned process, one thread for OpenCV's and (when loaded) torch's own pools, which read no environment
    variable: a worker of a pinned process then uses one core."""
    if not pinned():
        return
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def environment() -> dict[str, str]:
    """This process's environment with ONE_THREAD: for a process to start pinned."""
    return {**os.environ, **ONE_THREAD, **QUIET_POOLS}


def sfzmaps_command() -> list[str]:
    """The command that runs sfzmaps in this environment."""
    exe = Path(sys.executable).with_name("sfzmaps")
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "sfzmaps.cli"]


def run_pinned(argv: list[str]) -> int:
    """Run `sfzmaps <argv>` in a new process started pinned (its output goes where this one's does); returns its
    exit status."""
    return subprocess.call(sfzmaps_command() + list(argv), env=environment())


def warn_unless_pinned(log, what: str) -> None:
    """Log a warning that `what` may come out different in its last bits on another machine, unless this process is
    pinned."""
    if not pinned():
        log.warning("%s: the maths libraries are not pinned to one thread (sfzmaps.solver.threads), so the numbers "
                    "may differ in their last bits on a machine with another number of cores", what)
