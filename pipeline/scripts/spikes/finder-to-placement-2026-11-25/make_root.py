# One-off check, not pipeline code: builds the rough-check trial root: the finder's blocks with the traced arm's rough placement copied in. Run from pipeline/.
"""Step 1 (SFZMAPS_DATA_DIR unset: today's data is the source). Make the trial root rough-check/finder for one sheet with
trial.prepare, then byte-copy into it the finder arm's blocks_px.geojson (+ its .inputs.json) and the traced arm's
rough.json.  Writes only under data/find/trial/rough-check/."""
import hashlib, shutil, sys
from pathlib import Path
from sfzmaps import config
from sfzmaps.find import trial

sid = ("1921", "20180102094348509")
AP = trial.TRIAL / "all-pieces"
root = trial.TRIAL / "rough-check" / "finder"
assert "SFZMAPS_DATA_DIR" not in __import__("os").environ
made = trial.prepare(root, [sid], config.DATA_DIR)
print("prepared", root, "files", len(made))
src_f, src_t = AP / "finder/work" / sid[0] / sid[1], AP / "traced/work" / sid[0] / sid[1]
dst = root / "work" / sid[0] / sid[1]
for f, s in (("blocks_px.geojson", src_f), ("blocks_px.geojson.inputs.json", src_f), ("rough.json", src_t)):
    shutil.copyfile(s / f, dst / f)   # bytes only
    print("copied", s / f, "->", dst / f)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in trial.KEPT:
    for name in (f, f + ".inputs.json", f + ".answers"):
        if (dst / name).exists():
            print(f"{name:40} root {sha(dst/name)[:16]}  finder-arm {sha(src_f/name)[:16] if (src_f/name).exists() else '-'}"
                  f"  traced-arm {sha(src_t/name)[:16] if (src_t/name).exists() else '-'}")
