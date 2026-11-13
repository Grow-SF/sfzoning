"""Run one of the R&D scripts exactly as it ran, with two optional changes made at load time (the file is not edited):
  --scratch DIR  every mention of the R&D scratch folder (the scripts hard-code it) reads DIR instead, so inputs are
                 read from and outputs written to DIR;
  --cache-only   the vision model answers from the pipeline's answer cache only: llm.CACHE_ONLY is turned on (a
                 question not in the cache raises LLMCacheMiss, no call is made), and the client class is replaced by
                 one that cannot call out, since some scripts build a client up front even when every answer is cached.
usage: run_as_ran.py [--cache-only] [--scratch DIR] SCRIPT [ARGS ...]"""
import sys
from pathlib import Path

SCRATCH = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad"

args = sys.argv[1:]
cache_only, scratch = False, None
while args and args[0].startswith("--"):
    flag = args.pop(0)
    if flag == "--cache-only":
        cache_only = True
    elif flag == "--scratch":
        scratch = str(Path(args.pop(0)).resolve())
    else:
        sys.exit(f"unknown flag {flag}")
script = str(Path(args[0]).resolve())

if cache_only:
    from sfzmaps import llm

    class _NoClient:
        """Stands in for llm.OpenAIClient: never reached on a cache hit; a miss raises before a client is used."""

        def __init__(self):
            pass

        def call(self, *a, **k):
            raise llm.LLMCacheMiss("cache-only run: a live call was attempted")

    llm.CACHE_ONLY = True
    llm.OpenAIClient = _NoClient

src = Path(script).read_text()
if scratch:
    src = src.replace(SCRATCH, scratch)
sys.argv = [script, *args[1:]]
sys.path.insert(0, str(Path(script).parent))
exec(compile(src, script, "exec"), {"__name__": "__main__", "__file__": script})
