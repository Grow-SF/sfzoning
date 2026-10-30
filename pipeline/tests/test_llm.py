import base64
import os
import types

import numpy as np
import pytest
from pydantic import BaseModel

from sfzmaps import config, llm

pytestmark = pytest.mark.real_openai_client   # these tests wire a fake SDK into the real class


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
    # the SDK drops usage on this path: a conservative estimate is counted (spend guard)
    assert llm.CallCounter.estimated == 1
    assert llm.CallCounter.out_tokens == llm.MAX_OUT_TOKENS and llm.CallCounter.in_tokens > 0


# --- errors, retries, quota -----------------------------------------------------------

def _status_error(cls, status, *, code=None, message="err", headers=None):
    import httpx
    req = httpx.Request("POST", "https://example.invalid/v1/responses")
    resp = httpx.Response(status, request=req, headers=headers or {})
    body = {"error": {"message": message, "code": code, "type": code}} if code else {"error": {"message": message}}
    return cls(message, response=resp, body=body)


def _seq_client(monkeypatch, outcomes):
    """A client whose parse() raises / returns `outcomes` in turn; sleeps recorded."""
    c = _client(None)
    calls = []

    def parse(**kw):
        calls.append(1)
        o = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(o, BaseException):
            raise o
        return o

    c._client.responses.parse = parse
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    return c, calls, sleeps


@pytest.mark.parametrize("kw", [
    {"code": "insufficient_quota", "message": "You exceeded your current quota"},
    {"message": "Organization spend limit reached"},
    {"message": "You have hit your quota for the month"},
])
def test_quota_429_maps_to_quota_error_without_retry(monkeypatch, kw):
    import openai
    c, calls, sleeps = _seq_client(monkeypatch, [_status_error(openai.RateLimitError, 429, **kw)])
    with pytest.raises(llm.LLMQuotaError):
        c.call(b"x", "p", Ans)
    assert len(calls) == 1 and sleeps == []


def test_ordinary_429_and_5xx_retried_with_backoff(monkeypatch):
    import openai
    outcomes = [_status_error(openai.RateLimitError, 429, message="Rate limit, slow down",
                              headers={"retry-after": "3"}),
                _status_error(openai.InternalServerError, 500),
                _resp(parsed=Ans(n=7))]
    c, calls, sleeps = _seq_client(monkeypatch, outcomes)
    assert c.call(b"x", "p", Ans) == Ans(n=7)
    assert len(calls) == 3
    assert sleeps[0] == 3.0                      # Retry-After honoured
    assert 0.5 * llm.BACKOFF_S * 2 <= sleeps[1] <= llm.BACKOFF_S * 2   # 2nd attempt, jittered


def test_retries_give_up_after_max(monkeypatch):
    import openai
    c, calls, sleeps = _seq_client(monkeypatch, [_status_error(openai.InternalServerError, 503)])
    with pytest.raises(openai.InternalServerError):
        c.call(b"x", "p", Ans)
    assert len(calls) == llm.MAX_RETRIES + 1 and len(sleeps) == llm.MAX_RETRIES


def test_sdk_client_built_without_its_own_retries(monkeypatch):
    import openai
    seen = {}

    class SDK:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(openai, "OpenAI", SDK)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    llm.OpenAIClient()
    assert seen["max_retries"] == 0


def test_backoff_bounds():
    assert llm.backoff(0, 500) == llm.RETRY_AFTER_MAX_S
    for a in range(10):
        assert 0 < llm.backoff(a) <= llm.BACKOFF_MAX_S


class QuotaClient:
    def __init__(self):
        self.calls = 0

    def call(self, png, prompt, schema):
        self.calls += 1
        raise llm.LLMQuotaError("spend limit")


def test_quota_stops_later_calls_and_cancels_pool():
    import time
    q = QuotaClient()
    slow = []

    def one(i):
        if i == 0:
            return llm.ask_json(IMG, "q0", Ans, client=q)
        time.sleep(0.05)
        slow.append(i)
        return llm.ask_json(IMG + i, f"q{i}", Ans, client=FakeClient())

    with pytest.raises(llm.LLMQuotaError):
        llm.pmap(one, range(40), 2)
    time.sleep(0.2)
    assert len(slow) < 40          # queued work cancelled
    with pytest.raises(llm.LLMQuotaError, match="stopped"):
        llm.ask_json(IMG, "later", Ans, client=FakeClient())
    assert q.calls == 1


def test_spend_guard(monkeypatch):
    monkeypatch.setenv("SFZMAPS_MAX_USD", "0.01")
    c = FakeClient()
    llm.ask_json(IMG, "a", Ans, client=c)          # nothing spent yet
    llm.CallCounter.add_usage(types.SimpleNamespace(input_tokens=10_000, output_tokens=0))
    with pytest.raises(llm.LLMQuotaError, match="SFZMAPS_MAX_USD"):
        llm.ask_json(IMG, "b", Ans, client=c)
    assert c.calls == 1
    assert llm.ask_json(IMG, "a", Ans, client=c) == Ans(n=1)   # cache hits still answer


def test_cache_only_never_calls(monkeypatch):
    c = FakeClient()
    llm.ask_json(IMG, "cached", Ans, client=c)
    with llm.cache_only():
        assert llm.ask_json(IMG, "cached", Ans, client=c) == Ans(n=6)
        with pytest.raises(llm.LLMCacheMiss):
            llm.ask_json(IMG, "uncached", Ans, client=c)
        with pytest.raises(llm.LLMCacheMiss):
            llm.OpenAIClient()
    assert c.calls == 1 and not llm.CACHE_ONLY


def test_inflight_cap(monkeypatch):
    import threading
    import time
    monkeypatch.setattr(llm, "_INFLIGHT", threading.BoundedSemaphore(2))
    lock, now, peak = threading.Lock(), [0], [0]

    class Slow:
        def call(self, png, prompt, schema):
            with lock:
                now[0] += 1
                peak[0] = max(peak[0], now[0])
            time.sleep(0.02)
            with lock:
                now[0] -= 1
            return schema(n=1)

    llm.pmap(lambda i: llm.ask_json(IMG, f"p{i}", Ans, client=Slow()), range(12), 8)
    assert peak[0] == 2


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


def _v1_key(png, prompt, schema):
    """The cache key as computed through 2d1bf49 (before effort / tokens / detail joined)."""
    import hashlib
    import json
    h = hashlib.sha256()
    for part in (config.VISION_PROVIDER, config.VISION_MODEL, prompt,
                 json.dumps(schema.model_json_schema(), sort_keys=True)):
        h.update(part.encode())
        h.update(b"\0")
    h.update(png)
    return h.hexdigest()


def test_cache_key_legacy_params_keep_existing_entries():
    assert (config.VISION_REASONING_EFFORT, llm.MAX_OUT_TOKENS, llm.DETAIL) == llm.LEGACY_KEY_PARAMS
    assert llm._key(b"x", "p", Ans) == _v1_key(b"x", "p", Ans)


@pytest.mark.parametrize("name,value", [("VISION_REASONING_EFFORT", "high"),
                                        ("MAX_OUT_TOKENS", 32000), ("DETAIL", "low")])
def test_cache_key_includes_effort_tokens_detail(monkeypatch, name, value):
    k1 = llm._key(b"x", "p", Ans)
    monkeypatch.setattr(llm, name, value)
    assert llm._key(b"x", "p", Ans) != k1


def test_corrupt_cache_recovers(tmp_path):
    c = FakeClient()
    llm.ask_json(IMG, "hello", Ans, client=c)
    f = next((tmp_path / "llm").glob("*.json"))
    f.write_text("garbage{")
    r = llm.ask_json(IMG, "hello", Ans, client=c)
    assert r == Ans(n=5) and c.calls == 2
    assert llm.ask_json(IMG, "hello", Ans, client=c) == r and c.calls == 2
