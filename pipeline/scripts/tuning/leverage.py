"""leverage.py SERIES SHEET: how far does one GCP moved 40 ft (or a 3-corner block moved 15 ft) pull the fit, on real inlier density?"""
import sys, json, numpy as np
from sfzmaps import config
from sfzmaps.transform import GCP, SheetTransform
s, sh = sys.argv[1], sys.argv[2]
M = [m for m in json.loads((config.WORK_DIR/s/sh/"refine.json").read_text())["matches"] if m["label"] == "inlier"]
G = [GCP(m["px"], m["py"], m["wx"], m["wy"], "auto") for m in M]
P = np.array([[g.px, g.py] for g in G]); rng = np.random.default_rng(0); ks = rng.choice(len(G), 6, replace=False)
from scipy.spatial import cKDTree
tree = cKDTree(np.array([[g.wx, g.wy] for g in G]))
for sm in (1.0, 0.1, 0.03, 0.01):
    base = SheetTransform(G, sm); b = base.to_world(P[ks])
    one, blk = [], []
    for i, k in enumerate(ks):
        bad = list(G); g = G[k]; bad[k] = GCP(g.px, g.py, g.wx + 40, g.wy, "auto")
        one.append(np.linalg.norm(SheetTransform(bad, sm).to_world(P[k:k+1]) - b[i:i+1]))
        nb = tree.query_ball_point([g.wx, g.wy], 250)  # a block's worth of corners
        bad = list(G)
        for j in nb: q = G[j]; bad[j] = GCP(q.px, q.py, q.wx + 15, q.wy, "auto")
        blk.append((np.linalg.norm(SheetTransform(bad, sm).to_world(P[k:k+1]) - b[i:i+1]), len(nb)))
    print(f"s={sm}: lone 40 ft GCP pulls fit {np.mean(one):.1f} ft (max {np.max(one):.1f});  "
          f"all corners within 250 ft moved 15 ft -> fit follows {np.mean([x for x,_ in blk]):.1f} ft (n~{np.mean([n for _,n in blk]):.0f})")
