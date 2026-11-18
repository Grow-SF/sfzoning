"""The saved vision answers (`pipeline/saved-reads/answers/`, see its README): one gzipped JSON line per answer, read
by answer file, by sheet, or by the answer's cache key; and Sol's label points of each sheet, which the label stage
keeps in data/work (`sol_labels`).  Nothing here asks a model."""
import ast
import gzip
import json

from sfzmaps import config
from sfzmaps.solver import paths


def answers(read: str) -> list[dict]:
    """Every line of `SAVED_ANSWERS/<read>.jsonl.gz` in file order, e.g. read "text-on-every-sheet" or
    "experiments-2000/third-code-read".  An `answer` held as a string is a Python-literal dict and is parsed."""
    path = paths.SAVED_ANSWERS / f"{read}.jsonl.gz"
    with gzip.open(path, "rt") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    for r in lines:
        if isinstance(r.get("answer"), str):
            r["answer"] = ast.literal_eval(r["answer"])
    return lines


def by_sheet(read: str, sheet: str) -> list[dict]:
    """The lines of `read` about one sheet, in file order."""
    return [r for r in answers(read) if r.get("sheet") == sheet]


def by_key(read: str) -> dict[str, dict]:
    """The lines of `read` by their `cache_key` ("llm/<key>.json"); a key on two lines is an error (ValueError),
    since one would hide the other."""
    out = {}
    for r in answers(read):
        k = r.get("cache_key")
        if not k:
            continue
        if k in out:
            raise ValueError(f"{read}: the cache key {k} is on more than one line")
        out[k] = r
    return out


def cache_key(key: str) -> str:
    """The saved `cache_key` of the answer whose `llm._key` is `key`: its file in the answer cache."""
    return f"llm/{key}.json"


def sol_labels(sheet: str, series: str = paths.SERIES) -> list[dict]:
    """Sol's labels of a sheet: the "labels" of data/work/<series>/<sheet>/district_labels.json (a declared input of
    the stages that read it, so a missing file stops them)."""
    f = config.WORK_DIR / series / sheet / "district_labels.json"
    if not f.is_file():
        raise FileNotFoundError(f"no Sol labels for {sheet}: can't read {f}")
    return json.loads(f.read_text())["labels"]
