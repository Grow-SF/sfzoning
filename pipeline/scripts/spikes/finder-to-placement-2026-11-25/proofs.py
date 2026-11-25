# One-off check, not pipeline code: proofs that the rough-check's inputs are byte-equal to the arms', plus its numbers (step 4 of the rough-check).
"""Step 4 (SFZMAPS_DATA_DIR unset; reads only). Proofs (a)-(c), the numbers of the new root and of both all-pieces arms,
refine's iterations / converged / vote status, final ft/px, and the second question (rough transforms of the other three sheets)."""
import hashlib, json
from pathlib import Path
from sfzmaps.find import trial
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
AP = trial.TRIAL / "all-pieces"; R = trial.TRIAL / "rough-check" / "finder"
sid = ("1921", "20180102094348509")
w = lambda root: Path(root) / "work" / sid[0] / sid[1]
print("(a) blocks sha256  new root", sha(w(R) / "blocks_px.geojson"))
print("              finder arm", sha(w(AP / "finder") / "blocks_px.geojson"))
print("              traced arm", sha(w(AP / "traced") / "blocks_px.geojson"))
print("(b) rough.json sha256 new root", sha(w(R) / "rough.json"), " traced arm", sha(w(AP / "traced") / "rough.json"),
      " finder arm", sha(w(AP / "finder") / "rough.json"))
print("    rough.json bytes equal to traced arm's:", (w(R) / "rough.json").read_bytes() == (w(AP / "traced") / "rough.json").read_bytes())
rec = json.loads((w(R) / "refine_raw.json.inputs.json").read_text())
print("(c) refine_raw.json.inputs.json:", rec["inputs"])
print("    refine's blocks hash == finder sha:", dict(rec["inputs"])["blocks_px.geojson"] == sha(w(AP / "finder") / "blocks_px.geojson"),
      "; refine's rough hash == traced rough sha:", dict(rec["inputs"])["rough.json"] == sha(w(AP / "traced") / "rough.json"))
rough_t = json.loads((w(R) / "rough.json").read_text())["transform"]
raw = json.loads((w(R) / "refine_raw.json").read_text())
print("    refine_raw transform differs from the rough start (refine moved it):", raw["transform"]["gcps"] != rough_t["gcps"])
print("    rough: scale_hint", json.loads((w(R) / "rough.json").read_text())["scale_hint_ft_per_px"], "gcps", len(rough_t["gcps"]))
for name, root in (("rough-check finder", R), ("all-pieces traced", AP / "traced"), ("all-pieces finder", AP / "finder")):
    n = trial.numbers(root, sid)
    rf = json.loads((w(root) / "refine.json").read_text()); sc = json.loads((w(root) / "score.json").read_text())
    print(trial.row(sid, name, n))
    print(f"    refine iterations {rf['iterations']} converged {rf['converged']} vote {rf['vote_status']} vote_shift_ft {rf['vote_shift_ft']} "
          f"final_radius_ft {rf['final_radius_ft']} classify_converged {rf['classify_converged']}; score ft_per_px {sc.get('ft_per_px')} "
          f"label_agreement {sc.get('label_agreement')} inlier_frac {sc.get('inlier_frac')} n_scored {sc.get('n_scored')} n_holdout {sc.get('n_holdout')}")
    ws = json.loads((Path(root) / "warp" / sid[0] / sid[1] / "scores.json").read_text())
    print("    warp scores:", {k: v for k, v in ws.items() if k.startswith(("within1", "chosen", "device", "placement"))})
print("second question: rough.json of the other sheets, traced vs finder arm")
for s in (("1921", "20180102100525362"), ("1921", "20180102094142210"), ("2000-digital-zoning", "2000-digital-zoning-p08")):
    a, b = [json.loads((AP / arm / "work" / s[0] / s[1] / "rough.json").read_text()) for arm in ("traced", "finder")]
    ta, tb = a["transform"], b["transform"]
    print(f"  {s[1]}: file bytes equal {(AP/'traced'/'work'/s[0]/s[1]/'rough.json').read_bytes() == (AP/'finder'/'work'/s[0]/s[1]/'rough.json').read_bytes()};"
          f" transform equal {ta == tb}; scale hints {a['scale_hint_ft_per_px']} vs {b['scale_hint_ft_per_px']}")
    if ta != tb and ta and tb:
        ga, gb = ta["gcps"], tb["gcps"]
        print("    gcps", len(ga), len(gb), "keys differing:", [k for k in ta if ta.get(k) != tb.get(k)])
        if len(ga) == len(gb):
            import numpy as np
            d = np.array([[g[k] - h[k] for k in ("px", "py", "wx", "wy")] for g, h in zip(ga, gb)])
            print("    max |diff| px,py,wx,wy:", np.abs(d).max(axis=0))
