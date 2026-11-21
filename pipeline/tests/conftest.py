"""Test-wide guards: no test may reach the vision API, or use the Mac's GPU while a text detector has it.

* per-region reads (regionread.ENABLED) are off unless a test switches them on;
* config.INPUTS_DIR points into the test's tmp_path (never the repo's pipeline/inputs/);
* constructing llm.OpenAIClient raises, so a missing FakeClient fails loudly instead of
  calling the network (or reading pipeline/.env).  Tests of the client itself opt out with
  ``@pytest.mark.real_openai_client`` (they build it with __new__ or a fake SDK);
* where there is a GPU (`mps`), each test process holds a shared turn on it (`sfzmaps.gpu`) for the whole session,
  taken without waiting.  When a detector's exclusive turn has it, the session never uses it: the warp's fits run
  on the CPU (SFZMAPS_WARP_DEVICE=cpu) and the tests that need the GPU (marked ``mps``) are skipped.  Either way the
  tests can't give a detector training beside them wrong numbers (two processes on `mps` at once do), and never
  wait for one.
"""
import os

# Parallel workers (pytest-xdist) each get one maths thread; otherwise every worker's
# NumPy/OpenCV grabs all cores and the run gets slower, not faster. Set before they load.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import cv2
    cv2.setNumThreads(1)

import pytest

from sfzmaps import config, gpu, llm
from sfzmaps.districts import regionread


class _NoClient:
    def __init__(self, *a, **k):
        raise RuntimeError("tests must not construct llm.OpenAIClient (pass a FakeClient, or mark "
                           "the test real_openai_client)")


def pytest_configure(config):
    config.addinivalue_line("markers", "real_openai_client: keep the real llm.OpenAIClient class "
                                       "(the test wires a fake SDK itself)")
    config.addinivalue_line("markers", "mps: the test computes on the Mac's GPU (skipped while a detector has it)")


GPU_BUSY = "the GPU is busy (a detector is training)"


@pytest.fixture(scope="session", autouse=True)
def _gpu_turn_for_the_session():
    """True when the GPU is busy (see the module); holds a shared turn for the session otherwise."""
    import torch
    if not torch.backends.mps.is_available():
        yield False
        return
    turn = gpu.take_turn(exclusive=False, wait=False)
    if turn is not None:
        try:
            yield False
        finally:
            gpu.end_turn(turn)
        return
    before = os.environ.get("SFZMAPS_WARP_DEVICE")
    os.environ["SFZMAPS_WARP_DEVICE"] = "cpu"
    try:
        yield True
    finally:
        if before is None:
            os.environ.pop("SFZMAPS_WARP_DEVICE", None)
        else:
            os.environ["SFZMAPS_WARP_DEVICE"] = before


@pytest.fixture(autouse=True)
def _no_gpu_while_a_detector_has_it(request, _gpu_turn_for_the_session):
    if _gpu_turn_for_the_session and request.node.get_closest_marker("mps") is not None:
        pytest.skip(GPU_BUSY)


@pytest.fixture(autouse=True)
def _no_vision_api(request, monkeypatch, tmp_path):
    # hand-made inputs: never the repo's real ones (sfzmaps.inputs reads config.INPUTS_DIR)
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    monkeypatch.setattr(regionread, "ENABLED", False)
    monkeypatch.setattr(llm, "CACHE_ONLY", False)
    monkeypatch.delenv("SFZMAPS_MAX_USD", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if request.node.get_closest_marker("real_openai_client") is None:
        monkeypatch.setattr(llm, "OpenAIClient", _NoClient)
    llm.reset_stop()
    regionread.Stats.reset()
    yield
    llm.reset_stop()


@pytest.fixture
def small_model(monkeypatch):
    """The frame finder's network shrunk (`frame.model` FOLD, LEVELS, WIDTH) so tests run on cpu in seconds, and its
    shape buckets (`model.BUCKET`) with it, so small pages are not padded to 1,024 px."""
    from sfzmaps.frame import model
    monkeypatch.setattr(model, "FOLD", 2)
    monkeypatch.setattr(model, "LEVELS", 2)
    monkeypatch.setattr(model, "WIDTH", 4)
    monkeypatch.setattr(model, "BUCKET", 8)


@pytest.fixture
def tiny_backbone():
    """A stand-in for the frozen DINOv2 backbone with the output structure `frame.layout` relies on: called with
    `pixel_values` and `output_hidden_states=True`, it returns `.hidden_states` (the embeddings, then one entry per
    layer: 4 layers, 8 dims, patch 14), each (1, 1 + registers + patches, 8): a class token, 4 register tokens, the
    patch tokens.  It has `.layernorm` and `.config.num_register_tokens`.  No weights are downloaded."""
    import types

    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.config = types.SimpleNamespace(num_register_tokens=4, patch_size=14, hidden_size=8)
            self.proj = torch.nn.Conv2d(3, 8, 14, stride=14)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(4)])
            self.cls = torch.nn.Parameter(torch.randn(1, 1, 8))
            self.reg = torch.nn.Parameter(torch.randn(1, 4, 8))
            self.layernorm = torch.nn.LayerNorm(8)

        def forward(self, pixel_values, output_hidden_states=False):
            p = self.proj(pixel_values).flatten(2).transpose(1, 2)
            h = torch.cat([self.cls.expand(len(p), -1, -1), self.reg.expand(len(p), -1, -1), p], 1)
            states = [h]
            for lin in self.layers:
                h = torch.tanh(lin(h))
                states.append(h)
            return types.SimpleNamespace(hidden_states=tuple(states), last_hidden_state=self.layernorm(h))

    return Tiny().eval()
