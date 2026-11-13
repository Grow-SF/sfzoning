"""Saved reads: every answer the vision models gave this project, and how each was asked for.

  python -m saved_reads run --out DIR [--scratch DIR] SCRIPT [ARGS ...]
      run a reading script cache-only (nothing paid), recording every lookup into DIR
  python -m saved_reads index --out OUT [--calls DIR ...]
      index every answer in the cache with its recorded ways of asking, pipeline uses and shipped pack

Run from pipeline/scripts with the pipeline's python."""
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "index"):
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        from .run import main as run
        run(args)
    else:
        out, calls = None, []
        while args:
            flag = args.pop(0)
            if flag == "--out":
                out = Path(args.pop(0))
            elif flag == "--calls":
                calls.append(Path(args.pop(0)))
            else:
                sys.exit(f"unknown flag {flag}")
        if out is None:
            sys.exit(__doc__)
        from .index import main as index
        index(out, calls)


if __name__ == "__main__":
    main()
