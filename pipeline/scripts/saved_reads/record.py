"""Record how each saved answer was asked for.

install(out) wraps llm.ask_json and llm.tile_key so that every lookup writes one line to out/calls.jsonl:
  key       the cache file ("llm/<sha>.json" or "llm/tiles/<sha>.json"), recomputed exactly as llm does
  model     llm's model parts (provider, model; and the call parameters when they enter the key)
  effort    the reasoning effort setting
  prompt    sha256 of the prompt (or a tile's context string); the text is in out/prompts.jsonl
  schema    sha256 of the answer format's JSON schema (ask_json only); the schema is in out/schemas.jsonl
  image     sha256 of the exact PNG bytes the key was made from; the bytes are in out/images/<sha>.png
  hit       whether the cache held the answer
  run       which recorded run made the lookup (out/runs.jsonl: the script, its sha256, its arguments)
  caller    the file and line that asked
Recording never changes what the script does; with llm.CACHE_ONLY on, a missing answer still raises."""
import hashlib
import inspect
import json
import os
import threading
import time
from pathlib import Path

import cv2

from sfzmaps import llm

_lock = threading.Lock()
_state = {}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _once(kind: str, sha: str, write) -> None:
    """write a prompt / schema / image the first time its sha is seen in this output folder"""
    seen = _state["seen"].setdefault(kind, set())
    if sha in seen:
        return
    seen.add(sha)
    write()


def _caller() -> str:
    for f in inspect.stack()[2:8]:
        if "saved_reads" not in f.filename and not f.filename.endswith("llm.py"):
            return f"{f.filename}:{f.lineno}"
    return ""


def _write(rec: dict, prompt: str, png: bytes, schema_json: str | None) -> None:
    out = _state["out"]
    p_sha, i_sha = _sha(prompt.encode()), _sha(png)
    rec.update(prompt=p_sha, image=i_sha, run=_state["run"], caller=_caller())
    with _lock:
        _once("prompt", p_sha, lambda: _append(out / "prompts.jsonl", {"sha": p_sha, "text": prompt}))
        if schema_json is not None:
            s_sha = _sha(schema_json.encode())
            rec["schema"] = s_sha
            _once("schema", s_sha, lambda: _append(out / "schemas.jsonl", {"sha": s_sha, "schema": json.loads(schema_json)}))
        img = out / "images" / f"{i_sha}.png"
        _once("image", i_sha, lambda: img.exists() or img.write_bytes(png))
        _append(out / "calls.jsonl", rec)


def _append(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _model() -> dict:
    lead, trail = llm._model_parts()
    return {"model": lead, "params": trail, "effort": llm.VISION_REASONING_EFFORT}


def install(out: Path, script: str, argv: list[str]) -> None:
    """start recording into out (appending); one run line per install"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    seen = {}
    for kind, name, field in (("prompt", "prompts.jsonl", "sha"), ("schema", "schemas.jsonl", "sha")):
        f = out / name
        seen[kind] = {json.loads(line)[field] for line in open(f)} if f.exists() else set()
    seen["image"] = {p.stem for p in (out / "images").glob("*.png")}
    run = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    _state.update(out=out, run=run, seen=seen)
    src = Path(script).read_bytes() if Path(script).exists() else b""
    _append(out / "runs.jsonl", {"run": run, "script": script, "script_sha256": _sha(src), "argv": argv,
                                 "cwd": os.getcwd(), "data_dir": str(llm.CACHE_DIR.parent),
                                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    ask, tile_key = llm.ask_json, llm.tile_key

    def ask_json(image, prompt, schema, *, client=None):
        png = llm._encode(image)
        key = llm._key(png, prompt, schema)
        rel = f"llm/{key}.json"
        rec = {"key": rel, "kind": "ask", "schema_name": schema.__name__, **_model(),
               "hit": (llm.CACHE_DIR / rel).exists()}
        _write(rec, prompt, png, json.dumps(schema.model_json_schema(), sort_keys=True))
        return ask(image, prompt, schema, client=client)

    def tile_key_rec(tile_rgb, context):
        key = tile_key(tile_rgb, context)
        png = llm._encode(cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
        rel = f"llm/tiles/{key}.json"
        rec = {"key": rel, "kind": "tile", **_model(), "hit": (llm.CACHE_DIR / rel).exists()}
        _write(rec, context, png, None)
        return key

    llm.ask_json = ask_json
    llm.tile_key = tile_key_rec
