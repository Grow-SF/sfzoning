"""A page of code spotting on one 2000 sheet (Task 8): each px coloured by the likeliest class of the code head
(`run_detector spot`), each quarter's px from its own held-out run (`code-<quarter>-s<seed>`; a quarter with no run
is left as ink), and the owner's answered labels: a cross at the label's point, the owner's code and the spotted one
beside it, green when they agree, red when not.  Grey: ink off any code; a pale wash: another text.

    uv run python scripts/wordread/spot_page.py [--sheet 2000-digital-zoning-p07] [--name code] [--seed 0]

Writes experiments/wordread/review/code-spotting/<sheet>/index.html (the image embedded) and spot.png."""
from __future__ import annotations

import argparse
import base64
import colorsys
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "textlines")]
import run_detector as RD  # noqa: E402

from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import data, reader_stage  # noqa: E402
from sfzmaps.solver.wordread import codes, measures  # noqa: E402

OUT = RD.ROOT.parent / "review" / "code-spotting"
INK = (110, 110, 110)
OTHER_WASH = (235, 220, 200)          # BGR: another text


def palette(n: int) -> np.ndarray:
    """`n` colours (BGR uint8) spread around the hue circle, alternating two lightnesses so neighbours differ."""
    out = []
    for k in range(n):
        r, g, b = colorsys.hls_to_rgb(k / n, 0.42 if k % 2 else 0.55, 0.85)
        out.append((int(b * 255), int(g * 255), int(r * 255)))
    return np.array(out, np.uint8)


def render(sheet: str, runs: dict, labels) -> tuple[np.ndarray, dict]:
    """The sheet's image (BGR, the page's px) and each code's colour (hex), from `runs` ({quarter number: run folder})
    and the answered `labels` of the sheet (rows with point_x, point_y, truth, code)."""
    sd = RD.load_sheet(sheet)
    ink = data.page_ink(sd)
    quarter = reader_stage.page_quarters(sd)
    names = codes.classes()
    colours = palette(len(names) - codes.FIRST_CODE)
    img = np.full((*ink.shape, 3), 255, np.uint8)
    img[ink] = INK
    for q, run in runs.items():
        o = RD.outputs(run, sheet)
        mine = quarter == q
        other = mine & (o.code == codes.OTHER)
        img[other & ~ink] = OTHER_WASH
        on = mine & (o.code >= codes.FIRST_CODE) & (o.code < len(names))
        tint = colours[o.code[on].astype(int) - codes.FIRST_CODE]
        img[on] = np.where(ink[on][:, None], (tint * 0.55).astype(np.uint8), tint)
    for r in labels:
        p = (int(round(r["point_x"])), int(round(r["point_y"])))
        good = r["code"] == r["truth"]
        colour = (40, 150, 40) if good else (30, 30, 210)
        cv2.drawMarker(img, p, colour, cv2.MARKER_CROSS, 14, 2)
        text = f"{r['truth']}" if good else f"{r['truth']} / {r['code'] or '-'}"
        cv2.putText(img, text, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, text, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    hexes = {names[k + codes.FIRST_CODE]: "#%02x%02x%02x" % tuple(int(v) for v in colours[k][::-1])
             for k in range(len(colours))}
    return img, hexes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="2000-digital-zoning-p07")
    ap.add_argument("--name", default="code")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = {q: RD.run_dir(a.name, qn, a.seed) for q, qn in enumerate(paths.QUARTERS)
            if (RD.run_dir(a.name, qn, a.seed) / "words" / f"{a.sheet}.npz").is_file()}
    answers = measures.answers_from_store()
    answers = answers[(answers["sheet"] == a.sheet) & answers["quarter"].isin(list(runs))]
    names = codes.classes()
    rows, cache = [], {}
    for r in answers.to_dict("records"):
        o = cache.setdefault(r["quarter"], RD.outputs(runs[r["quarter"]], a.sheet))
        c = RD.code_at(o.code, r["point_x"], r["point_y"])
        rows.append({**r, "code": names[c] if codes.FIRST_CODE <= c < len(names) else None})
    img, hexes = render(a.sheet, runs, rows)
    out = OUT / a.sheet
    out.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", img)
    (out / "spot.png").write_bytes(png.tobytes())
    right = sum(r["code"] == r["truth"] for r in rows)
    shown = ", ".join(paths.QUARTERS[q] for q in runs) or "none"
    legend = "".join(f"<span class=k><i style='background:{c}'></i>{html.escape(n)}</span>"
                     for n, c in sorted(hexes.items()))
    page = (f"<!doctype html><meta charset=utf-8><title>Code spotting</title><style>body{{font:14px sans-serif;"
            f"margin:16px;background:#fff;color:#111}}.k{{display:inline-block;margin:2px 8px 2px 0;font-size:12px}}"
            f".k i{{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle}}"
            f"img{{max-width:100%;border:1px solid #ccc}}</style><h1>Code spotting: {html.escape(a.sheet)}</h1>"
            f"<p>{html.escape(RD.SPOT_NOTE)}. Each px coloured by the code head's likeliest class, each quarter's from "
            f"its own held-out run (quarters shown: {shown}); grey ink off any code; a pale wash: another text. "
            f"Crosses: the owner's answered labels, green when the spotted code is the owner's, red when not "
            f"(owner / spotted). Labels right here: {right} of {len(rows)}.</p><p>{legend}</p>"
            f"<img alt='code spotting' src='data:image/png;base64,{base64.b64encode(png.tobytes()).decode()}'>")
    (out / "index.html").write_text(page + "\n")
    print(json.dumps({"page": str(out / "index.html"), "labels": len(rows), "right": int(right),
                      "quarters": shown, "mb": round((out / "index.html").stat().st_size / 1e6, 2)}))


if __name__ == "__main__":
    main()
