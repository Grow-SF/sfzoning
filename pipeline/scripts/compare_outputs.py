"""Compare one stage's outputs in two checkouts, each stage held to its own epsilon (fresh_clone_check.sh).

    uv run python scripts/compare_outputs.py STAGE NEW_PIPELINE_DIR REF_PIPELINE_DIR
    uv run python scripts/compare_outputs.py --use-reference STAGE NEW_PIPELINE_DIR REF_PIPELINE_DIR

NEW and REF are `pipeline/` folders. Prints `STAGE: identical | within epsilon | differs`, then the first
differences; exits 1 only for `differs`. With --use-reference, replaces NEW's outputs of STAGE with a copy of REF's
(after the comparison, so the stages after it run on the reference's inputs); exits 2 if REF has none.

Rules:
  * every file is compared byte for byte; if the bytes differ, an .npz is compared by its arrays' content, a
    .parquet by pandas `equals`, a .png by its decoded pixels (equal content still counts as identical);
  * the text detector stages (solver-text-nw/-ne/-sw/-se): chance/*.png may differ by at most 2 counts (0.01 x 255)
    at any px (within epsilon); model.pt is not compared; .json numbers compare within 1% (a retrained model's
    scores move a little);
  * solver-text-scores: .json numbers within 1% (relative, or 1e-4 absolute near zero).
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PNG_EPS = 2          # counts: 0.01 * 255, rounded down
JSON_REL = 0.01
JSON_ABS = 1e-4
IGNORED = {".DS_Store"}

STAGE_DIRS = {
    "warp-2000-digital-zoning": "data/warp/2000-digital-zoning",
    "solver-page": "data/solver/page",
    "solver-lots": "data/solver/lots",
    "solver-prints": "data/solver/prints",
    "solver-text-nw": "data/solver/text-nw",
    "solver-text-ne": "data/solver/text-ne",
    "solver-text-sw": "data/solver/text-sw",
    "solver-text-se": "data/solver/text-se",
    "solver-text-scores": "data/solver/text-scores",
    "solver-labels": "data/solver/labels",
    "solver-walls": "data/solver/walls",
    "solver-solve": "data/solver/solve",
}
# what each stage must have made, on both sides (patterns under its folder): a pass needs these present
REQUIRED = {
    "warp-2000-digital-zoning": ["*/transform.json", "*/warped.tif"],
    "solver-page": ["ink.png", "sheets.png", "frames.json"],
    "solver-lots": ["lots.npz", "table.parquet"],
    "solver-prints": ["prints.parquet", "summary.json", "terra_boxes.json"],
    "solver-text-nw": ["chance/*.png"],
    "solver-text-ne": ["chance/*.png"],
    "solver-text-sw": ["chance/*.png"],
    "solver-text-se": ["chance/*.png"],
    "solver-text-scores": ["*.json"],
    "solver-labels": ["readings_*.npz", "strokes.npz", "trust.json"],
    "solver-walls": ["walls_*.npz"],
    "solver-solve": ["runs/final.npz", "report.json"],
}
TEXT_STAGES = {"solver-text-nw", "solver-text-ne", "solver-text-sw", "solver-text-se"}
LOOSE_JSON = TEXT_STAGES | {"solver-text-scores"}


@dataclass
class Result:
    identical: int = 0
    within: int = 0
    differs: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "differs" if self.differs else ("within epsilon" if self.within else "identical")


def _json_close(a, b, path="") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys differ"
        for k in a:
            r = _json_close(a[k], b[k], f"{path}/{k}")
            if r:
                return r
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = _json_close(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return None
        if abs(a - b) <= max(JSON_ABS, JSON_REL * max(abs(a), abs(b))):
            return None
        return f"{path}: {a} vs {b}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def _npz_diff(new: Path, ref: Path) -> str | None:
    with np.load(new, allow_pickle=True) as a, np.load(ref, allow_pickle=True) as b:
        if sorted(a.files) != sorted(b.files):
            return f"arrays differ: {sorted(set(a.files) ^ set(b.files))}"
        for k in sorted(a.files):
            x, y = a[k], b[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                return f"array {k}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}"
            same = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)
            if not same:
                return f"array {k} differs"
    return None


def _png(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def compare_file(rel: Path, new: Path, ref: Path, stage: str) -> tuple[str, str]:
    """('identical' | 'within' | 'differs', what differs)."""
    if new.read_bytes() == ref.read_bytes():
        return "identical", ""
    suf = rel.suffix
    if suf == ".npz":
        d = _npz_diff(new, ref)
        return ("identical", "") if d is None else ("differs", d)
    if suf == ".parquet":
        import pandas as pd
        same = pd.read_parquet(new).equals(pd.read_parquet(ref))
        return ("identical", "") if same else ("differs", "table differs")
    if suf == ".png":
        a, b = _png(new), _png(ref)
        if a.shape != b.shape:
            return "differs", f"shape {a.shape} vs {b.shape}"
        if a.dtype != np.uint8 or b.dtype != np.uint8:
            return "differs", f"dtype {a.dtype} vs {b.dtype}, not uint8"
        worst = int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()) if a.size else 0
        if worst == 0:
            return "identical", ""
        if stage in TEXT_STAGES and rel.parts[0] == "chance" and worst <= PNG_EPS:
            return "within", f"max diff {worst} counts"
        return "differs", f"max diff {worst} counts"
    if suf == ".json" and stage in LOOSE_JSON:
        d = _json_close(json.loads(new.read_text()), json.loads(ref.read_text()))
        return ("within", "numbers within 1%") if d is None else ("differs", d)
    return "differs", "bytes differ"


def _files(root: Path, stage: str) -> dict[Path, Path]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in IGNORED and "__pycache__" not in p.parts:
            rel = p.relative_to(root)
            if stage in TEXT_STAGES and rel.name == "model.pt":
                continue
            out[rel] = p
    return out


def compare_stage(stage: str, new_pipeline: Path, ref_pipeline: Path) -> Result:
    sub = STAGE_DIRS[stage]
    new_root, ref_root = Path(new_pipeline) / sub, Path(ref_pipeline) / sub
    res = Result()
    if not new_root.is_dir():
        res.differs.append(f"{new_root} was not made")
        return res
    if not ref_root.is_dir():
        res.differs.append(f"{ref_root} is missing (nothing to compare with)")
        return res
    new, ref = _files(new_root, stage), _files(ref_root, stage)
    for pat in REQUIRED[stage]:
        for side, files in (("made", new), ("reference", ref)):
            if not any(r.match(pat) for r in files):
                res.differs.append(f"no {pat} in the {side} outputs")
    if not set(new) & set(ref):
        res.differs.append("no files compared")
    for rel in sorted(set(ref) - set(new)):
        res.differs.append(f"{rel}: not made")
    for rel in sorted(set(new) - set(ref)):
        res.differs.append(f"{rel}: not in the reference")
    for rel in sorted(set(new) & set(ref)):
        kind, what = compare_file(rel, new[rel], ref[rel], stage)
        if kind == "identical":
            res.identical += 1
        elif kind == "within":
            res.within += 1
        else:
            res.differs.append(f"{rel}: {what}")
    return res


def use_reference(stage: str, new_pipeline: Path, ref_pipeline: Path) -> Path:
    """Put the reference's outputs of `stage` in the place of the new checkout's (those deleted first, nothing of them
    left), so that every later stage runs on the inputs the reference's own run of it had: each stage is held to its
    own epsilon, never to the epsilons of the stages above it as well.  Returns the new checkout's folder."""
    sub = STAGE_DIRS[stage]
    src, dst = Path(ref_pipeline) / sub, Path(new_pipeline) / sub
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is missing: no reference outputs of {stage} to use")
    shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)
    return dst


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[0] == "--use-reference" and argv[1] in STAGE_DIRS:
        try:
            dst = use_reference(argv[1], Path(argv[2]), Path(argv[3]))
        except FileNotFoundError as e:
            print(f"{argv[1]}: {e}")
            return 2
        print(f"{argv[1]}: the reference's outputs copied into {dst} (the stages after it read them)")
        return 0
    if len(argv) != 3 or argv[0] not in STAGE_DIRS:
        print(__doc__)
        print("stages:", " ".join(STAGE_DIRS))
        return 2
    res = compare_stage(argv[0], Path(argv[1]), Path(argv[2]))
    print(f"{argv[0]}: {res.status} ({res.identical} identical, {res.within} within epsilon, "
          f"{len(res.differs)} differ)")
    for d in res.differs[:10]:
        print(f"    {d}")
    if len(res.differs) > 10:
        print(f"    ... and {len(res.differs) - 10} more")
    return 1 if res.differs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
