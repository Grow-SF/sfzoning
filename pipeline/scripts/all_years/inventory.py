"""What exists of each solver input for every series (read-only, $0): the sheets and the warp's corrected sheets,
frame cuts, Terra's word boxes, Sol's label points, the 2000-only saved reads, the page's colour, the published map,
and the hand inputs that could score a solve.

    uv run python scripts/all_years/inventory.py

Writes data/solver/experiments/all-years/inventory.json and inventory.md."""
import collections
import csv
import gzip
import json
from pathlib import Path

import cv2

from sfzmaps import config
from sfzmaps.districts.series import SERIES

OUT = config.DATA_DIR / "solver" / "experiments" / "all-years"
ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"


def _lines(name):
    with gzip.open(ANSWERS / f"{name}.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def _rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def main():
    terra = collections.defaultdict(set)
    for r in _lines("text-on-every-sheet"):
        terra[r["series"]].add(r["sheet"])
    sol_saved = collections.Counter(r["series"] for r in _lines("district-labels"))
    rows = []
    for s, info in SERIES.items():
        work, warp = config.WORK_DIR / s, config.DATA_DIR / "warp" / s
        sheets = sorted(d.name for d in work.iterdir() if (d / "page.png").is_file()) if work.is_dir() else []
        placed = sorted(d.name for d in warp.iterdir() if (d / "transform.json").is_file()
                        and (d / "warped.tif").is_file()) if warp.is_dir() else []
        sol_files = [work / n / "district_labels.json" for n in placed if (work / n / "district_labels.json").is_file()]
        sol_use = sum(sum(1 for lab in json.loads(f.read_text())["labels"] if lab.get("kind", "use") == "use")
                      for f in sol_files)
        colour = None
        if placed:
            img = cv2.imread(str(work / placed[0] / "page.png"), cv2.IMREAD_UNCHANGED)
            colour = bool(img.ndim == 3 and img.shape[2] >= 3 and
                          (abs(img[..., 0].astype(int) - img[..., 2]) > 30).mean() > 0.01)
        inp = config.INPUTS_DIR / s
        rows.append({
            "series": s, "role": info.role, "year": info.year,
            "sheets": len(sheets), "warp_placed": len(placed),
            "cuts_file": (config.INPUTS_DIR / "merged-page" / f"{s}.json").is_file(),
            "page_json_masks": sum(1 for n in placed if json.loads((work / n / "page.json").read_text()).get("mask_boxes")),
            "terra_sheets": len(terra.get(s, set()) & set(placed)), "terra_sheets_any": len(terra.get(s, set())),
            "sol_label_files": len(sol_files), "sol_use_points": sol_use, "sol_saved_answers": sol_saved.get(s, 0),
            "reads_2000_only": s == "2000-digital-zoning",
            "colour_page": colour,
            "published_raw": (work / "districts_raw.parquet").is_file(),
            "published_site": (config.OUT_DIR / s / "districts" / "parcels.fgb").is_file(),
            "overrides": _rows(inp / "overrides.csv"), "label_fixes": _rows(inp / "label_fixes.csv"),
            "reviews": (inp / "reviews").is_dir(), "seeds": (inp / "seeds").is_dir(),
            "table": s in ("2000-digital-zoning", "1996-2000-zoning"),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(rows, indent=1) + "\n")
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (OUT / "inventory.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
