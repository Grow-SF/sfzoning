"""CI merge gate: everything this commit needs is on the Blob store, and dvc.lock matches the
code (sfzmaps.ship.check_shipped). Run from pipeline/: `uv run python scripts/check_shipped.py`."""
import subprocess
import sys

from sfzmaps import ship

status = subprocess.run(["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc", "status", "--json"],
                        capture_output=True, text=True)
if status.returncode != 0:
    sys.exit(f"dvc status failed:\n{status.stderr.strip()}")
problems = ship.check_shipped(status.stdout)
for p in problems:
    print(f"::error::{p}")
print("cache shipped: OK" if not problems else f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)
