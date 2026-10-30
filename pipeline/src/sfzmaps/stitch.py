"""Stitch a series' warped sheets into one VRT (overlaps resolved by sheet quality)."""
from __future__ import annotations

import json
import math
import subprocess

from pathlib import Path

from sfzmaps import config, manifest
from sfzmaps.score import SheetScore


def load_scores(series: str) -> dict[str, SheetScore]:
    """Scores from each sheet dir's score.json (SheetScore.empty('no_score') when missing)."""
    from sfzmaps import runner

    scores: dict[str, SheetScore] = {}
    base = config.WORK_DIR / series
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        sj = d / "score.json"
        try:
            scores[d.name] = runner._score_load(json.loads(sj.read_text()))
        except (OSError, ValueError, KeyError):
            scores[d.name] = SheetScore.empty("no_score")
    return scores


def draw_order(series: str, scores: dict[str, SheetScore]) -> list[str]:
    """Sheet ids with a warped.tif, bottom first: needs_review under passing/reviewed sheets,
    and within a class the larger holdout_p50_ft first (so the best ends up on top)."""
    def key(sid: str):
        sc = scores[sid]
        top = manifest.status_of(series, sid, sc) != "needs_review"
        p50 = sc.holdout_p50_ft
        return (top, -(p50 if math.isfinite(p50) else math.inf), sid)

    have = [s for s in scores if (config.WORK_DIR / series / s / "warped.tif").exists()]
    return sorted(have, key=key)


def build_vrt(series: str, scores: dict[str, SheetScore] | None = None) -> Path:
    scores = load_scores(series) if scores is None else scores
    order = draw_order(series, scores)
    if not order:
        raise RuntimeError(f"no warped sheets for series {series}; run `sfzmaps run --year {series}`")
    out = config.OUT_DIR / series / f"{series}.vrt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    lst.write_text("\n".join(str(config.WORK_DIR / series / s / "warped.tif") for s in order) + "\n")
    # The RGBA alpha band is honored as the mask; the last listed source wins in overlaps.
    subprocess.run(["gdalbuildvrt", "-overwrite", "-resolution", "highest",
                    "-input_file_list", str(lst), str(out)], check=True, capture_output=True)
    return out
