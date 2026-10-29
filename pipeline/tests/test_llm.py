import base64
import os
import types

import numpy as np
import pytest
from pydantic import BaseModel

from sfzmaps import config, llm


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


def test_missing_key_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / "nope.env")
    with pytest.raises(llm.LLMAuthError, match="OPENAI_API_KEY missing or invalid"):
        llm.OpenAIClient()


def test_env_file_loaded_but_env_wins(monkeypatch, tmp_path):
    f = tmp_path / ".env"
    f.write_text('# c\nOPENAI_API_KEY="fromfile"\nOTHER=1\n')
    monkeypatch.setattr(llm, "ENV_FILE", f)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm._load_env()
    assert os.environ["OPENAI_API_KEY"] == "fromfile"
    monkeypatch.setenv("OPENAI_API_KEY", "fromenv")
    llm._load_env()
    assert os.environ["OPENAI_API_KEY"] == "fromenv"


class FakeResponses:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None

    def parse(self, **kw):
        self.kwargs = kw
        return self.resp


def _client(resp):
    c = llm.OpenAIClient.__new__(llm.OpenAIClient)
    c._client = types.SimpleNamespace(responses=FakeResponses(resp))
    import openai
    c._openai = openai
    return c


def _resp(parsed=None, content=None, status="completed", reason=None):
    content = content if content is not None else []
    return types.SimpleNamespace(
        output_parsed=parsed, status=status,
        incomplete_details=types.SimpleNamespace(reason=reason) if reason else None,
        output=[types.SimpleNamespace(type="message", content=content)])


def test_openai_client_builds_request():
    c = _client(_resp(parsed=Ans(n=3)))
    out = c.call(b"\x89PNGdata", "read it", Ans)
    assert out == Ans(n=3)
    kw = c._client.responses.kwargs
    assert kw["model"] == config.VISION_MODEL and kw["text_format"] is Ans
    content = kw["input"][0]["content"]
    img = next(i for i in content if i["type"] == "input_image")
    assert img["detail"] == "high"
    assert img["image_url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNGdata").decode()
    assert {"type": "input_text", "text": "read it"} in content


def test_openai_refusal_raises():
    r = _resp(content=[types.SimpleNamespace(type="refusal", refusal="no")])
    with pytest.raises(llm.LLMRefusal):
        _client(r).call(b"x", "p", Ans)


def test_openai_incomplete_raises():
    r = _resp(status="incomplete", reason="max_output_tokens")
    with pytest.raises(RuntimeError, match="max_output_tokens"):
        _client(r).call(b"x", "p", Ans)


def test_openai_truncated_json_raises_runtime_error():
    def boom(**kw):
        Ans.model_validate_json('{"n": ')  # what parse() does on truncated output

    c = _client(None)
    c._client.responses.parse = boom
    with pytest.raises(RuntimeError, match="incomplete or invalid JSON"):
        c.call(b"x", "p", Ans)


def test_request_sets_reasoning_effort():
    c = _client(_resp(parsed=Ans(n=1)))
    c.call(b"x", "p", Ans)
    assert c._client.responses.kwargs["reasoning"] == {"effort": config.VISION_REASONING_EFFORT}


def test_concurrent_cache_writes_do_not_collide(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    c = FakeClient()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda _: llm.ask_json(IMG, "same", Ans, client=c), range(16)))
    assert not list((tmp_path / "llm").glob("*.tmp*"))
    assert llm.ask_json(IMG, "same", Ans, client=c) == Ans(n=4)


def test_cache_key_includes_provider(monkeypatch):
    k1 = llm._key(b"x", "p", Ans)
    monkeypatch.setattr(llm, "VISION_PROVIDER", "other")
    assert llm._key(b"x", "p", Ans) != k1


def test_corrupt_cache_recovers(tmp_path):
    c = FakeClient()
    llm.ask_json(IMG, "hello", Ans, client=c)
    f = next((tmp_path / "llm").glob("*.json"))
    f.write_text("garbage{")
    r = llm.ask_json(IMG, "hello", Ans, client=c)
    assert r == Ans(n=5) and c.calls == 2
    assert llm.ask_json(IMG, "hello", Ans, client=c) == r and c.calls == 2
