import numpy as np
import pytest
from pydantic import BaseModel

from sfzmaps import llm


class Ans(BaseModel):
    n: int


class FakeClient:
    def __init__(self):
        self.calls = 0

    def call(self, png, prompt, schema):
        self.calls += 1
        return schema(n=len(prompt))


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    llm.CallCounter.reset()


IMG = np.zeros((50, 60), np.uint8)


def test_cache_hit_skips_client():
    c = FakeClient()
    a = llm.ask_json(IMG, "hello", Ans, client=c)
    b = llm.ask_json(IMG, "hello", Ans, client=c)
    assert c.calls == 1 and a == b == Ans(n=5)
    assert (llm.CallCounter.live, llm.CallCounter.hits) == (1, 1)


def test_cache_key_changes_with_prompt():
    c = FakeClient()
    llm.ask_json(IMG, "hello", Ans, client=c)
    llm.ask_json(IMG, "hello!", Ans, client=c)
    assert c.calls == 2


def test_cache_key_changes_with_image():
    c = FakeClient()
    llm.ask_json(IMG, "hello", Ans, client=c)
    llm.ask_json(IMG + 1, "hello", Ans, client=c)
    assert c.calls == 2


def test_rejects_oversize_image():
    with pytest.raises(ValueError):
        llm.ask_json(np.zeros((100, 3000), np.uint8), "x", Ans, client=FakeClient())


def test_missing_key_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(llm.LLMAuthError, match="ANTHROPIC_API_KEY"):
        llm.AnthropicClient()


def test_corrupt_cache_recovers(tmp_path):
    c = FakeClient()
    llm.ask_json(IMG, "hello", Ans, client=c)
    f = next((tmp_path / "llm").glob("*.json"))
    f.write_text("garbage{")
    r = llm.ask_json(IMG, "hello", Ans, client=c)
    assert r == Ans(n=5) and c.calls == 2
    assert llm.ask_json(IMG, "hello", Ans, client=c) == r and c.calls == 2
