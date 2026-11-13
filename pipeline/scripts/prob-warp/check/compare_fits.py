"""Compare two fitboxes.py checkpoints (fits_raw.pkl): the letter fits found for each Terra box, matched by box.
Reports fits only in one, text / source differences, and the largest difference in score and letter points.
usage: compare_fits.py ORIG.pkl NEW.pkl"""
import pickle
import sys

import numpy as np

a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
A = {tuple(f["box"]): f for f in a}
B = {tuple(f["box"]): f for f in b}
only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
text, dscore, dpts, same = [], 0.0, 0.0, 0
for k in sorted(set(A) & set(B)):
    fa, fb = A[k], B[k]
    if (fa["text"], fa.get("source")) != (fb["text"], fb.get("source")):
        text.append((k, fa["text"], fa.get("source"), fb["text"], fb.get("source")))
        continue
    pa, pb = np.asarray(fa["pts"]), np.asarray(fb["pts"])
    ds = abs(float(fa["score"]) - float(fb["score"]))
    dp = float(np.abs(pa - pb).max()) if pa.shape == pb.shape else float("inf")
    dscore, dpts = max(dscore, ds), max(dpts, dp)
    same += ds == 0 and dp == 0
print(f"fits: {len(a)} orig, {len(b)} new; {same} identical; only orig {len(only_a)}, only new {len(only_b)}; "
      f"text/source differs {len(text)}; largest score diff {dscore:.3g}, largest point diff {dpts:.3g} px")
for r in text[:20]:
    print("  text", r)
for k in only_a[:10]:
    print("  only orig", k, A[k]["text"])
for k in only_b[:10]:
    print("  only new", k, B[k]["text"])
