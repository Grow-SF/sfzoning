"""Test-wide guards: no test may reach the vision API.

* per-region reads (regionread.ENABLED) are off unless a test switches them on;
* constructing llm.OpenAIClient raises, so a missing FakeClient fails loudly instead of
  calling the network (or reading pipeline/.env).  Tests of the client itself opt out with
  ``@pytest.mark.real_openai_client`` (they build it with __new__ or a fake SDK).
"""
import pytest

from sfzmaps import llm
from sfzmaps.districts import regionread


class _NoClient:
    def __init__(self, *a, **k):
        raise RuntimeError("tests must not construct llm.OpenAIClient (pass a FakeClient, or mark "
                           "the test real_openai_client)")


def pytest_configure(config):
    config.addinivalue_line("markers", "real_openai_client: keep the real llm.OpenAIClient class "
                                       "(the test wires a fake SDK itself)")


@pytest.fixture(autouse=True)
def _no_vision_api(request, monkeypatch):
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
