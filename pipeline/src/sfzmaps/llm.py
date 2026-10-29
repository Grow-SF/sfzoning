"""Disk-cached Claude vision client returning structured (pydantic) output."""
import argparse
import base64
import hashlib
import json
import logging
import os
import sys
from typing import Protocol, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel

from sfzmaps.config import CACHE_DIR, MODEL, VISION_MAX_EDGE

T = TypeVar("T", bound=BaseModel)
MAX_EDGE = VISION_MAX_EDGE
log = logging.getLogger(__name__)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    pass


class LLMAuthError(RuntimeError):
    pass


class CallCounter:
    live = 0
    hits = 0

    @classmethod
    def reset(cls) -> None:
        cls.live = 0
        cls.hits = 0


class LLMClient(Protocol):
    def call(self, png: bytes, prompt: str, schema: type[T]) -> T: ...


class AnthropicClient:
    def __init__(self) -> None:
        import anthropic

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMAuthError("ANTHROPIC_API_KEY is not set; export it to use the Claude API")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def call(self, png: bytes, prompt: str, schema: type[T]) -> T:
        try:
            resp = self._client.beta.messages.parse(
                model=MODEL,
                max_tokens=16000,
                output_format=schema,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": "medium"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png",
                            "data": base64.b64encode(png).decode()}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
        except self._anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Anthropic authentication failed: {e}") from None
        if resp.stop_reason == "refusal":
            raise LLMRefusal("Claude refused the request")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Claude response truncated (stop_reason=max_tokens); raise max_tokens")
        if resp.parsed_output is None:
            raise RuntimeError(f"no parsed output (stop_reason={resp.stop_reason})")
        return resp.parsed_output


def _key(png: bytes, prompt: str, schema: type[BaseModel]) -> str:
    h = hashlib.sha256()
    for part in (MODEL, prompt, json.dumps(schema.model_json_schema(), sort_keys=True)):
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
            CallCounter.hits += 1
            return cached
    if client is None:
        client = AnthropicClient()
    result = client.call(png, prompt, schema)
    CallCounter.live += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json())
    tmp.replace(path)
    return result


class _Sheet(BaseModel):
    sheet_number: int


def _smoke(path: str) -> int:
    import anthropic

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
    except anthropic.APIError as e:
        print(f"error: {type(e).__name__}: {e}".splitlines()[0], file=sys.stderr)
        return 1
    print("cache hit" if CallCounter.hits else "live call")
    print(out.sheet_number)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", metavar="IMAGE", required=True)
    sys.exit(_smoke(ap.parse_args().smoke))
