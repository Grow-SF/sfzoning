"""Turns on the Mac's GPU (sfzmaps.gpu), on a lock file of the test's own: the detector's turn is the GPU alone, the
warp's turns share it with each other but never with the detector's, and the CPU takes no turn."""
import os
import threading
import time

import pytest

from sfzmaps import gpu

WAIT_S = 0.3       # how long a turn that must wait is given to (wrongly) start
DONE_S = 10        # how long a turn that may start is given to finish


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "mps.lock"
    monkeypatch.setattr(gpu, "LOCK", path)
    return path


def _turn_in_a_thread(exclusive: bool, order: list, name: str) -> threading.Thread:
    """A turn on "mps" taken in a daemon thread (its own open of the lock file): records `name` once it has it."""
    def run():
        with gpu.gpu_turn(exclusive, "mps"):
            order.append(name)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_an_exclusive_turn_waits_for_every_other_turn(lock):
    for first in (True, False):
        order = []
        with gpu.gpu_turn(first, "mps"):
            waiting = _turn_in_a_thread(True, order, "the detector's turn")
            waiting.join(WAIT_S)
            assert waiting.is_alive() and order == []
            order.append("first turn ends")
        waiting.join(DONE_S)
        assert not waiting.is_alive() and order == ["first turn ends", "the detector's turn"]


def test_shared_turns_run_together_but_not_beside_an_exclusive_one(lock):
    order = []
    with gpu.gpu_turn(False, "mps"):
        other = _turn_in_a_thread(False, order, "another shared turn")
        other.join(DONE_S)
        assert not other.is_alive() and order == ["another shared turn"]
    order = []
    with gpu.gpu_turn(True, "mps"):
        shared = _turn_in_a_thread(False, order, "a shared turn")
        shared.join(WAIT_S)
        assert shared.is_alive() and order == []
        order.append("the detector's turn ends")
    shared.join(DONE_S)
    assert order == ["the detector's turn ends", "a shared turn"]


def test_a_turn_ends_when_its_body_raises(lock):
    with pytest.raises(RuntimeError):
        with gpu.gpu_turn(True, "mps"):
            raise RuntimeError("the fit failed")
    held = gpu.take_turn(True, wait=False)
    assert held is not None
    gpu.end_turn(held)


def test_the_cpu_takes_no_turn(lock):
    with gpu.gpu_turn(True, "mps"):
        with gpu.gpu_turn(True, "cpu"):        # would wait for itself if it took one
            pass


def test_a_turn_not_waited_for_is_none_while_another_holds_it(lock):
    held = gpu.take_turn(True)
    try:
        assert gpu.take_turn(False, wait=False) is None and gpu.take_turn(True, wait=False) is None
    finally:
        gpu.end_turn(held)
    shared = [gpu.take_turn(False, wait=False), gpu.take_turn(False, wait=False)]
    try:
        assert None not in shared
    finally:
        for f in shared:
            if f is not None:
                gpu.end_turn(f)


def test_a_lock_file_deleted_and_made_again_while_a_turn_waits_is_locked_again_where_it_is_now(lock, monkeypatch):
    """macOS deletes /tmp files nobody touched for 3 days.  A process that opened the lock file before it was deleted
    and locked it after another process made a new one would hold a turn on a file no one else locks, beside the
    other's turn: it takes its turn again on the file now at the path."""
    real, opened = gpu.fcntl.flock, []

    def flock(fd, op):
        if not opened:                      # between this open and its lock: deleted, and made again by another
            lock.unlink()
            lock.touch()
        opened.append(os.fstat(fd).st_ino)
        return real(fd, op)

    lock.touch()
    first = os.stat(lock).st_ino
    monkeypatch.setattr(gpu.fcntl, "flock", flock)
    held = gpu.take_turn(True)
    monkeypatch.setattr(gpu.fcntl, "flock", real)
    try:
        assert os.fstat(held).st_ino == os.stat(lock).st_ino != first
        assert opened == [first, os.stat(lock).st_ino]
        # the turn is held on the file at the path: it keeps out a turn taken there
        assert gpu.take_turn(False, wait=False) is None
    finally:
        gpu.end_turn(held)


def test_every_turn_taken_touches_the_lock_file(lock):
    lock.touch()
    for exclusive, wait in ((True, True), (False, False)):
        long_ago = time.time() - 5 * 86400               # older than the 3 days after which /tmp files are deleted
        os.utime(lock, (long_ago, long_ago))
        held = gpu.take_turn(exclusive, wait)
        try:
            st = os.stat(lock)
            assert st.st_mtime > long_ago + 86400 and st.st_atime > long_ago + 86400
        finally:
            gpu.end_turn(held)


def test_mps_is_named_by_a_string_or_a_torch_device():
    torch = pytest.importorskip("torch")
    assert gpu.on_gpu("mps") and gpu.on_gpu(torch.device("mps")) and gpu.on_gpu("mps:0")
    assert not gpu.on_gpu("cpu") and not gpu.on_gpu(torch.device("cpu"))
