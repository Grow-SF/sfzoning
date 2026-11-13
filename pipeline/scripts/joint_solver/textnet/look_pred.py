"""Picture (my own check, then the step page): held-out windows of the merged page with the detector's text chance as a
red tint (darker = surer), terra's boxes outlined blue, unknown areas outlined grey.
  row 1-2: terra words the detector missed (mean chance < 1/2), by kind
  row 3-4: new text pieces (called text, no terra box, no unknown area)
usage: look_pred.py QUARTER OUT.png"""
import sys

import cv2
import numpy as np

S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad"
qname, out = sys.argv[1], sys.argv[2]
q = ("nw", "ne", "sw", "se").index(qname)
D = np.load(f"{S}/textnet/data.npz")
P = np.load(f"{S}/textnet/text_pred_{qname}.npz")
ink, text, kind, unknown, quarter = D["ink"], D["text"], D["kind"], D["unknown"], D["quarter"]
prob = P["prob"]
inq = quarter == q
n, lab, st, _ = cv2.connectedComponentsWithStats(((text > 0) & inq).astype(np.uint8), connectivity=8)
li = lab[ink & inq & (lab > 0)]
s = np.bincount(li, prob[ink & inq & (lab > 0)].astype(float), minlength=n)
c = np.bincount(li, minlength=n)
missed = [i for i in range(1, n) if c[i] > 0 and s[i] / c[i] < 128]
new = np.load(f"{S}/textnet/new_text_{qname}.npy")
rng = np.random.default_rng(0)
W = 160


def panel(cx, cy, cap):
    y0, x0 = int(np.clip(cy - W // 2, 0, ink.shape[0] - W)), int(np.clip(cx - W // 2, 0, ink.shape[1] - W))
    sl = (slice(y0, y0 + W), slice(x0, x0 + W))
    im = np.where(ink[sl], 0, 255).astype(float)
    im = np.dstack([im, im, im])
    a = prob[sl][..., None] / 255.0 * 0.6
    im = im * (1 - a) + a * np.array([30, 30, 230])
    im = cv2.resize(im.astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for m, col in ((text[sl] > 0, (220, 120, 0)), (unknown[sl], (150, 150, 150))):
        cs, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, col, 2)
    cv2.putText(im, cap, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return im


panels = []
for i in rng.choice(missed, min(8, len(missed)), replace=False):
    x, y, w, h, _ = st[i]
    k = int(np.bincount(kind[lab == i]).argmax())
    panels.append(panel(x + w // 2, y + h // 2, f"missed {['', 'code', 'block', 'street', 'other'][k]} {s[i] / c[i] / 255:.2f}"))
for r in new[rng.choice(len(new), min(8, len(new)), replace=False)]:
    x, y, w, h, _ = r
    panels.append(panel(x + w // 2, y + h // 2, "new"))
while len(panels) % 4:
    panels.append(np.full_like(panels[0], 255))
img = np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)])
cv2.imwrite(out, img)
print("wrote", out, img.shape, "missed", len(missed), "new", len(new))
