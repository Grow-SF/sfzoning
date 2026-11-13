"""Run a script that reads the scan through the vision model, unchanged, with every answer taken from the cache and
every lookup recorded (record.py).  Always cache-only: llm.CACHE_ONLY is on and the client class is replaced by one
that cannot call out (some scripts build a client up front even when every answer is cached), so nothing is paid.

  --out DIR      where the record goes (calls.jsonl, prompts.jsonl, schemas.jsonl, images/, runs.jsonl)
  --scratch DIR  the R&D scripts hard-code the scratch folder they ran in; every mention of it reads DIR instead
                 (the file on disk is not edited)

usage: python -m saved_reads run --out DIR [--scratch DIR] SCRIPT [ARGS ...]"""
import importlib.abc
import importlib.util
import sys
from pathlib import Path

from sfzmaps import llm

from . import record

SCRATCH = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad"


class _Rewriter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports the script's sibling modules (e.g. k8/lotwalls.py) with the scratch folder replaced, as the script."""

    def __init__(self, folder: Path, scratch: str):
        self.folder, self.scratch = folder, scratch

    def find_spec(self, name, path, target=None):
        f = self.folder / f"{name}.py"
        if "." in name or not f.exists():
            return None
        return importlib.util.spec_from_file_location(name, f, loader=self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        src = Path(module.__file__).read_text().replace(SCRATCH, self.scratch)
        exec(compile(src, module.__file__, "exec"), module.__dict__)


class _NoClient:
    """Stands in for llm.OpenAIClient: never reached on a cache hit; a miss raises before a client is used."""

    def __init__(self):
        pass

    def call(self, *a, **k):
        raise llm.LLMCacheMiss("cache-only run: a live call was attempted")


def main(args: list[str]) -> None:
    out, scratch = None, None
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--out":
            out = Path(args.pop(0)).resolve()
        elif flag == "--scratch":
            scratch = str(Path(args.pop(0)).resolve())
        else:
            sys.exit(f"unknown flag {flag}")
    if out is None or not args:
        sys.exit(__doc__)
    script = str(Path(args[0]).resolve())
    llm.CACHE_ONLY = True
    llm.OpenAIClient = _NoClient
    record.install(out, script, args[1:])
    src = Path(script).read_text()
    if scratch:
        src = src.replace(SCRATCH, scratch)
        sys.meta_path.insert(0, _Rewriter(Path(script).parent, scratch))
    sys.argv = [script, *args[1:]]
    sys.path.insert(0, str(Path(script).parent))
    g = {"__name__": "__main__", "__file__": script}
    record.watch_globals(g)
    exec(compile(src, script, "exec"), g)
