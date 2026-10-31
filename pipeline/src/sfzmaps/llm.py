"""Disk-cached OpenAI vision client returning structured (pydantic) output."""
import argparse
import base64
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from sfzmaps.config import (CACHE_DIR, EST_TEXT_TOKENS_PER_CALL, PIPELINE_DIR, PRICE_IN_PER_M,
                            PRICE_OUT_PER_M, VISION_MAX_EDGE, VISION_MAX_INFLIGHT, VISION_MODEL,
                            VISION_PROVIDER, VISION_REASONING_EFFORT, estimate_image_tokens)

T = TypeVar("T", bound=BaseModel)
MAX_EDGE = VISION_MAX_EDGE
log = logging.getLogger(__name__)
ENV_FILE = PIPELINE_DIR / ".env"
MAX_OUT_TOKENS = 16000


class LLMRefusal(RuntimeError):
    pass


class LLMAuthError(RuntimeError):
    pass


class LLMQuotaError(RuntimeError):
    """The account cannot pay for more calls (insufficient_quota / spend limit) or the
    SFZMAPS_MAX_USD guard tripped: every further call would fail, so callers stop the run."""


class LLMCacheMiss(RuntimeError):
    """A live call was needed while CACHE_ONLY is on."""


# Errors after which no further call can succeed: callers abort the whole run on these.
FATAL = (LLMQuotaError, LLMAuthError)

DETAIL = "high"            # input_image detail
MAX_RETRIES = 6            # own retries of ordinary 429s / 5xx / connection errors (SDK: 0)
BACKOFF_S = 1.0            # first backoff; doubles per attempt, full jitter in [0.5, 1] x
BACKOFF_MAX_S = 60.0
RETRY_AFTER_MAX_S = 120.0
# (reasoning effort, max output tokens, detail) of every cache entry written before these
# joined the key: with exactly these values the key is computed the old way, so the
# existing cache still matches; any other value adds them to the key.
LEGACY_KEY_PARAMS = ("low", 16000, "high")

CACHE_ONLY = False         # answer from the disk cache only; a miss raises LLMCacheMiss
_STOP = threading.Event()  # set after a fatal error: later live calls refuse at once
_INFLIGHT = threading.BoundedSemaphore(VISION_MAX_INFLIGHT)
_sleep = time.sleep


class CallCounter:
    live = 0
    hits = 0
    tile_hits = 0     # per-tile answers served from the tile store (tile_get)
    in_tokens = 0     # billed usage reported by live OpenAI responses (or estimated, below)
    out_tokens = 0
    estimated = 0     # calls whose usage was estimated (invalid JSON: the SDK drops usage)
    _lock = threading.Lock()

    @classmethod
    def bump(cls, field: str) -> None:
        with cls._lock:
            setattr(cls, field, getattr(cls, field) + 1)

    @classmethod
    def add_usage(cls, usage) -> None:
        if usage is None:
            return
        with cls._lock:
            cls.in_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            cls.out_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    @classmethod
    def add_estimate(cls, in_tokens: int, out_tokens: int) -> None:
        with cls._lock:
            cls.in_tokens += int(in_tokens)
            cls.out_tokens += int(out_tokens)
            cls.estimated += 1

    @classmethod
    def usd(cls) -> float:
        return cls.in_tokens * PRICE_IN_PER_M / 1e6 + cls.out_tokens * PRICE_OUT_PER_M / 1e6

    @classmethod
    def reset(cls) -> None:
        cls.live = 0
        cls.hits = 0
        cls.tile_hits = 0
        cls.in_tokens = 0
        cls.out_tokens = 0
        cls.estimated = 0


class LLMClient(Protocol):
    def call(self, png: bytes, prompt: str, schema: type[T]) -> T: ...


def _load_env() -> None:
    """Fill OPENAI_API_KEY from pipeline/.env if unset (an existing env var wins)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        lines = ENV_FILE.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == "OPENAI_API_KEY" and v.strip().strip("\"'"):
            os.environ["OPENAI_API_KEY"] = v.strip().strip("\"'")


def is_quota_error(e: BaseException) -> bool:
    """A 429 that retrying cannot fix: code/body insufficient_quota, or a spend-limit /
    quota message."""
    code = str(getattr(e, "code", "") or "")
    body = getattr(e, "body", None)
    text = " ".join([code, json.dumps(body, default=str) if body is not None else "", str(e)]).lower()
    return "insufficient_quota" in text or "spend limit" in text or "quota" in text


def retry_after(e: BaseException) -> float | None:
    """Seconds from a Retry-After / retry-after-ms header, if the error carries one."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        if headers.get("retry-after-ms") is not None:
            return float(headers["retry-after-ms"]) / 1000
        if headers.get("retry-after") is not None:
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        return None
    return None


def backoff(attempt: int, after: float | None = None) -> float:
    if after is not None and after >= 0:
        return min(after, RETRY_AFTER_MAX_S)
    return min(BACKOFF_MAX_S, BACKOFF_S * 2 ** attempt) * random.uniform(0.5, 1.0)


def _png_size(png: bytes) -> tuple[int, int]:
    if len(png) >= 24 and png[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    return VISION_MAX_EDGE, VISION_MAX_EDGE


class OpenAIClient:
    def __init__(self) -> None:
        if CACHE_ONLY:
            raise LLMCacheMiss("cache-only run: no live vision client")
        import openai

        _load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMAuthError("OPENAI_API_KEY missing or invalid")
        self._openai = openai
        self._client = openai.OpenAI(max_retries=0)   # retries: OpenAIClient.call

    def _retryable(self, e: BaseException) -> bool:
        o = self._openai
        if isinstance(e, o.RateLimitError):
            return True
        if isinstance(e, o.APIConnectionError):     # includes APITimeoutError
            return True
        return isinstance(e, o.APIStatusError) and getattr(e, "status_code", 0) >= 500

    def _request(self, png: bytes, prompt: str, schema: type[T]):
        return self._client.responses.parse(
            model=VISION_MODEL,
            max_output_tokens=MAX_OUT_TOKENS,
            text_format=schema,
            reasoning={"effort": VISION_REASONING_EFFORT},
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "detail": DETAIL,
                     "image_url": "data:image/png;base64," + base64.b64encode(png).decode()},
                    {"type": "input_text", "text": prompt},
                ],
            }],
        )

    def call(self, png: bytes, prompt: str, schema: type[T]) -> T:
        o = self._openai
        attempt = 0
        while True:
            try:
                resp = self._request(png, prompt, schema)
                break
            except o.AuthenticationError:
                raise LLMAuthError("OPENAI_API_KEY missing or invalid") from None
            except ValidationError:
                # parse() validates the JSON itself; output cut off by max_output_tokens lands
                # here, and the SDK drops the usage: count a conservative estimate instead
                w, h = _png_size(png)
                CallCounter.add_estimate(estimate_image_tokens(w, h) + EST_TEXT_TOKENS_PER_CALL,
                                         MAX_OUT_TOKENS)
                raise RuntimeError("response incomplete or invalid JSON") from None
            except Exception as e:  # noqa: BLE001 - classified below
                if isinstance(e, o.RateLimitError) and is_quota_error(e):
                    raise LLMQuotaError(f"OpenAI quota / spend limit reached: {str(e)[:200]}") from None
                if not self._retryable(e) or attempt >= MAX_RETRIES or _STOP.is_set():
                    raise
                wait = backoff(attempt, retry_after(e))
                log.warning("vision call failed (%s); retry %d/%d in %.1fs",
                            type(e).__name__, attempt + 1, MAX_RETRIES, wait)
                _sleep(wait)
                attempt += 1
        CallCounter.add_usage(getattr(resp, "usage", None))
        for item in resp.output:
            if item.type == "message" and any(c.type == "refusal" for c in item.content):
                raise LLMRefusal("model refused the request")
        if resp.status == "incomplete":
            reason = resp.incomplete_details.reason if resp.incomplete_details else "unknown"
            raise RuntimeError(f"response incomplete ({reason}); raise MAX_OUT_TOKENS if max_output_tokens")
        if resp.output_parsed is None:
            raise RuntimeError(f"no parsed output (status={resp.status})")
        return resp.output_parsed


def _model_parts() -> tuple[list[str], list[str]]:
    """(leading, trailing) key parts naming the provider, model and call parameters; shared by
    _key and tile_key so the two can't drift.  Legacy params keep the v1 key (no trailing)."""
    params = (VISION_REASONING_EFFORT, MAX_OUT_TOKENS, DETAIL)
    trailing = [] if params == LEGACY_KEY_PARAMS else ["params:" + json.dumps(list(params))]
    return [VISION_PROVIDER, VISION_MODEL], trailing


def _key(png: bytes, prompt: str, schema: type[BaseModel]) -> str:
    h = hashlib.sha256()
    lead, trail = _model_parts()
    parts = [*lead, prompt, json.dumps(schema.model_json_schema(), sort_keys=True), *trail]
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    h.update(png)
    return h.hexdigest()


def _encode(image: np.ndarray) -> bytes:
    h, w = image.shape[:2]
    if max(h, w) > MAX_EDGE:
        raise ValueError(f"image {w}x{h} exceeds {MAX_EDGE}px long edge")
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def is_cached(image: np.ndarray, prompt: str, schema: type[BaseModel]) -> bool:
    """True if ask_json would answer from disk (no network call)."""
    return (CACHE_DIR / "llm" / f"{_key(_encode(image), prompt, schema)}.json").exists()


def tile_key(tile_rgb: np.ndarray, context: str) -> str:
    """Key of one tile's answer: its PNG bytes, the caller's context string (prompt, schema,
    anything else the answer depends on) and the provider / model / parameters."""
    h = hashlib.sha256()
    lead, trail = _model_parts()
    for part in (*lead, context, *trail):
        h.update(part.encode())
        h.update(b"\0")
    h.update(_encode(cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)))
    return h.hexdigest()


def _tile_path(key: str) -> Path:
    return CACHE_DIR / "llm" / "tiles" / f"{key}.json"


def tile_get(key: str) -> dict | None:
    """The stored answer for a tile key, or None (a corrupt file is discarded: a miss)."""
    path = _tile_path(key)
    try:
        answer = json.loads(path.read_text())
        if not isinstance(answer, dict):
            raise ValueError("not a JSON object")
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as e:
        log.warning("discarding bad tile cache file %s: %s", path.name, e)
        path.unlink(missing_ok=True)
        return None
    CallCounter.bump("tile_hits")
    return answer


def tile_put(key: str, answer: dict) -> None:
    """Store a tile's answer unless one is already there (first answer wins), atomically."""
    path = _tile_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(answer, f)
        try:
            os.link(tmp_name, path)       # fails if present: never overwrite
        except FileExistsError:
            pass
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def max_usd() -> float | None:
    v = os.environ.get("SFZMAPS_MAX_USD", "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        raise ValueError(f"SFZMAPS_MAX_USD must be a number, got {v!r}") from None


def check_budget() -> None:
    """LLMQuotaError once this process's vision spend (CallCounter) reaches SFZMAPS_MAX_USD."""
    cap = max_usd()
    if cap is not None and CallCounter.usd() >= cap:
        _STOP.set()
        raise LLMQuotaError(f"spend guard: ${CallCounter.usd():.2f} spent >= SFZMAPS_MAX_USD=${cap:g}")


def reset_stop() -> None:
    _STOP.clear()


@contextmanager
def cache_only(on: bool = True):
    """Within: every vision answer must come from the disk cache (LLMCacheMiss otherwise)."""
    global CACHE_ONLY
    prev, CACHE_ONLY = CACHE_ONLY, on
    try:
        yield
    finally:
        CACHE_ONLY = prev


def pmap(fn, items, workers: int) -> list:
    """[fn(x) for x in items] on a thread pool, in order.  A fatal error (quota, auth) cancels
    every queued call and returns at once (in-flight calls see _STOP and refuse to retry);
    any other error lets the submitted calls finish (their answers are cached) and re-raises."""
    items = list(items)
    if not items:
        return []
    ex = ThreadPoolExecutor(max(1, min(workers, len(items))))
    futs = [ex.submit(fn, x) for x in items]
    try:
        out = [f.result() for f in futs]
    except (*FATAL, KeyboardInterrupt) as e:
        if isinstance(e, FATAL):
            _STOP.set()
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:
        ex.shutdown(wait=True)
        raise
    ex.shutdown(wait=True)
    return out


def ask_json(image: np.ndarray, prompt: str, schema: type[T], *,
             client: LLMClient | None = None) -> T:
    png = _encode(image)
    path = CACHE_DIR / "llm" / f"{_key(png, prompt, schema)}.json"
    if path.exists():
        try:
            cached = schema.model_validate_json(path.read_text())
        except (ValueError, OSError) as e:
            log.warning("discarding bad cache file %s: %s", path.name, e)
            path.unlink(missing_ok=True)
        else:
            CallCounter.bump("hits")
            return cached
    if CACHE_ONLY:
        raise LLMCacheMiss(f"cache-only run: no cached answer ({path.name})")
    if _STOP.is_set():
        raise LLMQuotaError("vision calls stopped after an earlier quota / auth error")
    check_budget()
    if client is None:
        client = OpenAIClient()
    try:
        with _INFLIGHT:
            result = client.call(png, prompt, schema)
    except FATAL:
        _STOP.set()
        raise
    CallCounter.bump("live")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(result.model_dump_json())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return result


class _Sheet(BaseModel):
    sheet_number: int


def _smoke(path: str) -> int:
    import openai

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"cannot read image: {path}", file=sys.stderr)
        return 1
    s = 1024 / max(img.shape)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    try:
        out = ask_json(img, "What is the sheet number printed on this map?", _Sheet)
    except (LLMAuthError, LLMRefusal, RuntimeError) as e:  # LLMQuotaError is a RuntimeError
        print(f"error: {e}", file=sys.stderr)
        return 1
    except openai.OpenAIError as e:
        print(f"error: {type(e).__name__}: {e}".splitlines()[0], file=sys.stderr)
        return 1
    print("cache hit" if CallCounter.hits else "live call")
    print(out.sheet_number)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", metavar="IMAGE", required=True)
    sys.exit(_smoke(ap.parse_args().smoke))
