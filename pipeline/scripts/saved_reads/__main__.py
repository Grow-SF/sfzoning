"""Saved reads: every answer the vision models gave this project, and how each was asked for.

  python -m saved_reads run --out DIR [--scratch DIR] SCRIPT [ARGS ...]
      run a reading script cache-only (nothing paid), recording every lookup into DIR
  python -m saved_reads package --calls CALLS_DIR [--out OUT] [--images IMAGES]
      turn the recorded runs into the checked-in package (pipeline/saved-reads) and the named images
  python -m saved_reads verify [--from PACKAGE] [--images IMAGES]
      rebuild every replayed record's cache key from its own prompt, schema, model and image, and compare
  python -m saved_reads restore-cache [--from PACKAGE]
      write the package's answers back into the answer cache (a fresh clone, before replaying)
  python -m saved_reads index --out OUT [--calls DIR ...]
      index every answer in the cache with its recorded ways of asking, pipeline uses and shipped pack

Run from pipeline/scripts (or with it on PYTHONPATH) with the pipeline's python.  See pipeline/saved-reads/README.md."""
import sys
from pathlib import Path


def _flags(args, names):
    got = {n: [] for n in names}
    while args:
        flag = args.pop(0)
        if flag not in got:
            sys.exit(f"unknown flag {flag}\n\n{__doc__}")
        got[flag].append(Path(args.pop(0)))
    return got


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "index", "package", "restore-cache", "verify"):
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        from .run import main as run
        run(args)
    elif cmd == "index":
        f = _flags(args, ("--out", "--calls"))
        if not f["--out"]:
            sys.exit(__doc__)
        from .index import main as index
        index(f["--out"][0], f["--calls"])
    elif cmd == "verify":
        f = _flags(args, ("--from", "--images"))
        pipe = Path(__file__).resolve().parents[2]
        from .verify import main as verify
        verify((f["--from"] or [pipe / "saved-reads"])[0], (f["--images"] or [pipe / "data" / "saved-reads" / "images"])[0])
    elif cmd == "restore-cache":
        f = _flags(args, ("--from",))
        from .restore import main as restore
        restore((f["--from"] or [Path(__file__).resolve().parents[2] / "saved-reads"])[0])
    else:
        f = _flags(args, ("--calls", "--out", "--images"))
        if not f["--calls"]:
            sys.exit(__doc__)
        pipe = Path(__file__).resolve().parents[2]
        from .package import main as package
        package(f["--calls"][0], (f["--out"] or [pipe / "saved-reads"])[0],
                (f["--images"] or [pipe / "data" / "saved-reads" / "images"])[0])


if __name__ == "__main__":
    main()
