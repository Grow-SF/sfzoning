"""Issue #71 spike: overlay sheet A (red) and sibling B moved onto A by the cached alignment (cyan)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import issue71_shared_ink as m  # noqa: E402

series, sa, sb = sys.argv[1:4]
al = json.load(open(m.OUT / series / f"{sa}.align.json"))[sb]
A = m.load_ink(m.WORK / series / sa / "page.png", m.FINE)
B = m.load_ink(m.WORK / series / sb / "page.png", m.FINE)
M = np.array(al["M"], dtype=np.float32)
Bm = cv2.warpAffine(B, M, (A.shape[1], A.shape[0]))
rgb = np.stack([1 - Bm, 1 - A, 1 - A], -1)  # A ink -> red, B ink -> cyan, both -> black
out = m.OUT.parent / f"overlay_{series}_{sa[-5:]}_{sb[-5:]}.png"
Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).save(out)
print(out, {k: al[k] for k in ("k", "scale", "coarse_score", "ecc")})
