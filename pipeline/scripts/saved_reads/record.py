"""Record how each saved answer was asked for.

install(out, script, argv) wraps llm.ask_json and llm.tile_key so that every lookup writes one line to out/calls.jsonl:
  key       the cache file ("llm/<sha>.json" or "llm/tiles/<sha>.json"), recomputed exactly as llm does
  model     llm's model parts (provider, model; and the call parameters when they enter the key)
  effort    the reasoning effort setting
  prompt    sha256 of the prompt (or a tile's context string); the text is in out/prompts.jsonl
  schema    sha256 of the answer format's JSON schema (ask_json only); the schema is in out/schemas.jsonl
  image     sha256 of the exact PNG bytes the key was made from; the bytes are in out/images/<sha>.png
  hit       whether the cache held the answer
  context   where the image came from, as the asking code knew it: the small local values (tile corner, code point,
            case, place, variant ...) of each frame of the script's own files, or of the pipeline package, that led
            to this lookup
  run       which recorded run made it; out/runs.jsonl has the script, its sha256 and git blob id, its arguments,
            and (once, at the run's first lookup) the script's small top-level values: series, sheet, window, settings
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

import sfzmaps
from sfzmaps import llm

_lock = threading.Lock()
_state = {}
_PKG = str(Path(sfzmaps.__file__).parent)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _append(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _small(v, depth=0):
    """v if it is a small plain value (number, short text or path, or a short list / dict of them), else None"""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    if type(v).__module__ == "numpy" and getattr(v, "shape", None) == ():
        return v.item()
    if isinstance(v, Path):
        v = str(v)
    if isinstance(v, str):
        return v if len(v) <= 160 else None
    if depth == 0 and isinstance(v, (list, tuple)):
        kept = [x for x in (_small(x, 1) for x in v[:8]) if x is not None]
        return kept or None
    if depth == 0 and isinstance(v, dict) and len(v) <= 16:
        kept = {str(k): _small(x, 1) for k, x in v.items()}
        return {k: x for k, x in kept.items() if x is not None} or None
    return None


def _ours(filename: str) -> bool:
    """a frame of the asking script's own folder, or of the pipeline package (not llm.py or this package)"""
    if "saved_reads" in filename or filename.endswith("llm.py"):
        return False
    return filename.startswith(_state["folder"]) or filename.startswith(_PKG)


def _context(stack) -> dict:
    out = {}
    for f in stack:
        if f.function == "<module>" or not _ours(f.filename):
            continue
        loc = {k: _small(v) for k, v in f.frame.f_locals.items() if not k.startswith("_")}
        loc = {k: v for k, v in loc.items() if v is not None}
        if loc:
            out[f"{Path(f.filename).name}:{f.function}"] = loc
    return out


def watch_globals(g: dict) -> None:
    """the script's own top-level namespace (run.py executes the script in it), read at the first lookup: lookups
    made from worker threads do not have the script's top level on their stack"""
    _state["g"] = g


def _script_globals() -> dict:
    g = {k: _small(v) for k, v in _state.get("g", {}).items() if not k.startswith("_")}
    return {k: v for k, v in g.items() if v is not None and not callable(v)}


def _write(rec: dict, prompt: str, png: bytes, schema_json: str | None) -> None:
    out = _state["out"]
    stack = inspect.stack()[2:14]
    p_sha, i_sha = _sha(prompt.encode()), _sha(png)
    caller = next((f"{f.filename}:{f.lineno}" for f in stack if "saved_reads" not in f.filename
                   and not f.filename.endswith("llm.py")), "")
    rec.update(prompt=p_sha, image=i_sha, run=_state["run"], caller=caller, context=_context(stack))
    first = not _state["globals_written"]
    g = _script_globals() if first else None
    with _lock:
        if first and not _state["globals_written"]:
            _state["globals_written"] = True
            _append(out / "runs.jsonl", {"run": _state["run"], "script_globals": g})
        if p_sha not in _state["seen"]["prompt"]:
            _state["seen"]["prompt"].add(p_sha)
            _append(out / "prompts.jsonl", {"sha": p_sha, "text": prompt})
        if schema_json is not None:
            s_sha = _sha(schema_json.encode())
            rec["schema"] = s_sha
            if s_sha not in _state["seen"]["schema"]:
                _state["seen"]["schema"].add(s_sha)
                _append(out / "schemas.jsonl", {"sha": s_sha, "schema": json.loads(schema_json)})
        if i_sha not in _state["seen"]["image"]:
            _state["seen"]["image"].add(i_sha)
            img = out / "images" / f"{i_sha}.png"
            if not img.exists():
                img.write_bytes(png)
        _append(out / "calls.jsonl", rec)


def _model() -> dict:
    lead, trail = llm._model_parts()
    return {"model": lead, "params": trail, "effort": llm.VISION_REASONING_EFFORT}


def install(out: Path, script: str, argv: list[str]) -> None:
    """start recording into out (appending); one run line per install"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    seen = {}
    for kind, name in (("prompt", "prompts.jsonl"), ("schema", "schemas.jsonl")):
        f = out / name
        seen[kind] = {json.loads(line)["sha"] for line in open(f)} if f.exists() else set()
    seen["image"] = {p.stem for p in (out / "images").glob("*.png")}
    run = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    _state.update(out=out, run=run, seen=seen, script=script, folder=str(Path(script).parent), globals_written=False)
    src = Path(script).read_bytes() if Path(script).exists() else b""
    _append(out / "runs.jsonl", {
        "run": run, "script": script, "script_sha256": _sha(src),
        "script_git_blob": hashlib.sha1(b"blob %d\0" % len(src) + src).hexdigest(),
        "source": os.environ.get("SR_SOURCE", ""), "argv": argv, "cwd": os.getcwd(),
        "env": {k: os.environ[k] for k in ("SERIES", "SHEET", "SPLIT", "THREADS", "SFZ_CORR", "SFZMAPS_DATA_DIR",
                                           "K3", "FITAREA") if k in os.environ},
        "data_dir": str(llm.CACHE_DIR.parent), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    ask, tile_key = llm.ask_json, llm.tile_key
    model, effort = os.environ.get("SR_MODEL"), os.environ.get("SR_EFFORT")

    def _override():
        """SR_MODEL / SR_EFFORT: ask another model or effort than the script names (scripts set it at the top)"""
        if model:
            llm.VISION_MODEL = model
        if effort:
            llm.VISION_REASONING_EFFORT = effort

    def ask_json(image, prompt, schema, *, client=None):
        _override()
        png = llm._encode(image)
        rel = f"llm/{llm._key(png, prompt, schema)}.json"
        rec = {"key": rel, "kind": "ask", "schema_name": schema.__name__, **_model(),
               "hit": (llm.CACHE_DIR / rel).exists()}
        _write(rec, prompt, png, json.dumps(schema.model_json_schema(), sort_keys=True))
        return ask(image, prompt, schema, client=client)

    def tile_key_rec(tile_rgb, context):
        _override()
        key = tile_key(tile_rgb, context)
        png = llm._encode(cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
        rel = f"llm/tiles/{key}.json"
        rec = {"key": rel, "kind": "tile", **_model(), "hit": (llm.CACHE_DIR / rel).exists()}
        _write(rec, context, png, None)
        return key

    llm.ask_json = ask_json
    llm.tile_key = tile_key_rec
