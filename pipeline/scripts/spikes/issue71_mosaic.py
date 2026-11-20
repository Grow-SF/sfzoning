"""Issue #71 spike: mosaic of one series' finished pictures (only sheets fitted by the current model), labelled."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import issue71_shared_ink as m  # noqa: E402

series = sys.argv[1]
size = int(sys.argv[2]) if len(sys.argv) > 2 else 400
od = m.OUT / series
tiles = []
for j in sorted(od.glob("*.json")):
    if j.name.endswith(".align.json"):
        continue
    d = json.load(open(j))
    if not d["siblings"] or "chance" not in next(iter(d["siblings"].values())):
        continue
    im = Image.open(od / f"{d['sheet']}.png")
    im.thumbnail((size, size))
    c = Image.new("RGB", (size + 10, size + 30), "white")
    c.paste(im, (5, 5))
    ImageDraw.Draw(c).text((5, size + 10), f"{d['sheet'][-6:]}  frame share {d['pi']:.3f}", fill="black")
    tiles.append(c)
cols = 6
rows = (len(tiles) + cols - 1) // cols
S = Image.new("RGB", (cols * (size + 10), max(1, rows) * (size + 30)), "white")
for i, t in enumerate(tiles):
    S.paste(t, ((i % cols) * (size + 10), (i // cols) * (size + 30)))
out = m.OUT.parent / f"mosaic_{series}.png"
S.save(out)
print(out, len(tiles))
