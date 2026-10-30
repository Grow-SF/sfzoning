"""Disk-cached OpenAI vision client returning structured (pydantic) output."""
import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from sfzmaps.config import CACHE_DIR, PIPELINE_DIR, VISION_MAX_EDGE, VISION_MODEL, VISION_PROVIDER, VISION_REASONING_EFFORT

T = TypeVar("T", bound=BaseModel)
MAX_EDGE = VISION_MAX_EDGE
log = logging.getLogger(__name__)
ENV_FILE = PIPELINE_DIR / ".env"
MAX_OUT_TOKENS = 16000


class LLMRefusal(RuntimeError):
    pass


class LLMAuthError(RuntimeError):
    pass


class CallCounter:
    live = 0
    hits = 0
    in_tokens = 0     # billed usage reported by live OpenAI responses
    out_tokens = 0
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
    def reset(cls) -> None:
        cls.live = 0
        cls.hits = 0
        cls.in_tokens = 0
        cls.out_tokens = 0


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


class OpenAIClient:
    def __init__(self) -> None:
        import openai

        _load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMAuthError("OPENAI_API_KEY missing or invalid")
        self._openai = openai
        self._client = openai.OpenAI(max_retries=8)   # backoff on 429 / 5xx under parallel series

    def call(self, png: bytes, prompt: str, schema: type[T]) -> T:
        try:
            resp = self._client.responses.parse(
                model=VISION_MODEL,
                max_output_tokens=MAX_OUT_TOKENS,
                text_format=schema,
                reasoning={"effort": VISION_REASONING_EFFORT},
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_image", "detail": "high",
                         "image_url": "data:image/png;base64," + base64.b64encode(png).decode()},
                        {"type": "input_text", "text": prompt},
                    ],
                }],
            )
        except self._openai.AuthenticationError:
            raise LLMAuthError("OPENAI_API_KEY missing or invalid") from None
        except ValidationError:
            # parse() validates the JSON itself; output cut off by max_output_tokens lands here
            raise RuntimeError("response incomplete or invalid JSON") from None
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


def _key(png: bytes, prompt: str, schema: type[BaseModel]) -> str:
    h = hashlib.sha256()
    for part in (VISION_PROVIDER, VISION_MODEL, prompt, json.dumps(schema.model_json_schema(), sort_keys=True)):
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
    if client is None:
        client = OpenAIClient()
    result = client.call(png, prompt, schema)
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
    except (LLMAuthError, LLMRefusal, RuntimeError) as e:
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
