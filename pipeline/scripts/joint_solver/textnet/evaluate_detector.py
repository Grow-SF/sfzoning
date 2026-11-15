"""Scores the text detector on its held-out quarter (it never trained on these pixels).
  ink pixels   text vs not-text (terra's boxes; unknown areas left out): AUC, log loss, and at chance 1/2 the share of
               text ink found and the share of ink called text that is in a box
  words        terra's words (connected box regions) found: mean text chance over their ink >= 1/2, by kind
  new text     connected pieces of ink called text (chance >= 1/2) touching no terra box and no unknown area: text terra
               missed, or false alarms; saved for a look
  lines        ink on the table's district lines and on other lot borders, away from any box: share called text
usage: evaluate_detector.py QUARTER"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

S = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
os.environ.setdefault("SFZ_RND", str(S / "rnd_final"))
sys.path.insert(0, str(S / "pkg"))
from joint_solver.borders import border_paths  # noqa: E402
from joint_solver.data import city  # noqa: E402
from joint_solver.measures import Table  # noqa: E402

QUARTERS = ("nw", "ne", "sw", "se")
KIND = {1: "zoning code", 2: "block number", 3: "street name", 4: "other"}


def auc(pos, neg):
    """from scores 0..255: chance a random positive scores above a random negative"""
    hp, hn = np.bincount(pos, minlength=256), np.bincount(neg, minlength=256)
    below = np.cumsum(hn) - hn
    return float((hp * (below + 0.5 * hn)).sum() / max(hp.sum() * hn.sum(), 1))


def main(qname):
    q = QUARTERS.index(qname)
    D = np.load(S / "textnet" / "data.npz")
    P = np.load(S / "textnet" / f"text_pred_{qname}.npz")
    ink, text, kind, unknown, quarter = D["ink"], D["text"], D["kind"], D["unknown"], D["quarter"]
    prob = P["prob"]
    inq = quarter == q
    m = ink & inq & ~unknown
    pos, neg = prob[m & (text > 0)], prob[m & (text == 0)]
    p = np.clip(np.r_[pos, neg] / 255.0, 1e-4, 1 - 1e-4)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    called = np.r_[pos, neg] >= 128
    print(f"{qname} held out: ink px {m.sum():,} (text {len(pos):,}); AUC {auc(pos, neg):.4f}; log loss {ll:.4f}; at 1/2: "
          f"text ink found {np.mean(pos >= 128):.1%}, ink called text that is in a box {called[:len(pos)].sum() / max(called.sum(), 1):.1%}")
    # words
    n, lab = cv2.connectedComponents(((text > 0) & inq).astype(np.uint8), connectivity=8)
    li = lab[ink & inq & (lab > 0)]
    s = np.bincount(li, prob[ink & inq & (lab > 0)].astype(float), minlength=n)
    c = np.bincount(li, minlength=n)
    kk = np.zeros(n, int)
    kk[lab[(lab > 0)]] = kind[lab > 0]
    found = (c > 0) & (s / np.maximum(c, 1) >= 128)
    words = c > 0
    print("words found: " + "; ".join(f"{KIND[k]} {found[words & (kk == k)].sum()}/{(words & (kk == k)).sum()}"
                                       for k in KIND) + f"; all {found[words].sum()}/{words.sum()} ({found[words].mean():.1%})")
    # new text: pieces of ink called text away from any box or unknown area
    called_px = (prob >= 128) & ink & inq
    near = cv2.dilate(((text > 0) | unknown).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    pieces = cv2.dilate(called_px.astype(np.uint8), np.ones((9, 9), np.uint8))
    n2, lab2, st, cen = cv2.connectedComponentsWithStats(pieces, connectivity=8)
    touch = np.zeros(n2, bool)
    touch[np.unique(lab2[near & (pieces > 0)])] = True
    cnt = np.bincount(lab2[called_px], minlength=n2)
    new = [i for i in range(1, n2) if not touch[i] and cnt[i] >= 20]
    print(f"new text pieces (no terra box, no unknown area, 20+ ink px): {len(new)}")
    np.save(S / "textnet" / f"new_text_{qname}.npy", np.array([st[i] for i in new]))
    # lines away from any box
    C = city()
    _, row0, col0 = C.scan()
    T = Table(np.zeros(C.n, bool))
    Z = border_paths()
    k = Z["k"].astype(np.int64)
    pr = row0 - np.floor(Z["my"]).astype(np.int64)
    pc = np.floor(Z["mx"]).astype(np.int64) - col0
    ok = (pr >= 0) & (pr < ink.shape[0]) & (pc >= 0) & (pc < ink.shape[1])
    for name, sel in (("table district lines", T.line[k]), ("other lot borders", ~T.line[k])):
        mask = np.zeros(ink.shape, np.uint8)
        g = ok & sel
        mask[pr[g], pc[g]] = 1
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
        lm = mask & ink & inq & ~near
        print(f"{name}: ink px away from boxes {lm.sum():,}; called text {np.mean(prob[lm] >= 128):.2%}; "
              f"mean chance {prob[lm].mean() / 255:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
