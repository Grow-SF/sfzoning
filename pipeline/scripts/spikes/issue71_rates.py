"""Issue #71 spike: print each sheet's fitted witness rates (f / chance / d per sibling)."""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "data" / "research" / "issue71" / "shared_ink"
for arg in sys.argv[1:]:
    d = json.load(open(root / f"{arg}.json"))
    print(arg, "pi", round(d["pi"], 3))
    print("  ", " ".join(f"{k[-4:]}:{v['f']:.2f}/{v['chance']:.2f}/{v['d']:.2f}" for k, v in d["siblings"].items()))
