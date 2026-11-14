# saved_reads/rerun: a copy of zoning-lines-rnd:pipeline/scripts/zoning-lines/leaders.py with one fix, 2026-11-13: the image goes to llm.ask_json in BGR, the order it
# encodes (cv2.imencode).  The original handed it RGB, so the mark the prompt calls red reached the model blue.
"""Experiment: ask the vision model where each printed code points (leader, arrow, bracket).

usage: leaders.py test [effort]          the known cases, printed with the expected answer
       leaders.py sheet SID [effort]     every use label on one 2000 sheet -> leaders/<SID>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from sfzmaps import config, llm
from sfzmaps.districts import labels as L

S = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
SERIES = "2000-digital-zoning"
HALF = 160          # crop half-size, page px (4 ft/px: ~1,300 ft across)
UP = 2              # upscale for legibility

PROMPT = """This is a crop of a scanned 2000 San Francisco zoning map, {w} x {h} pixels (origin top-left).
The district code "{code}" is printed at the centre, circled in red.

On these maps a code is usually printed inside the area it names. But when that area is too small or
too narrow (a commercial strip along a street, a single small lot), the code is printed beside it and
tied to it by a leader: a thin line, a curved line, an arrow, or a bracket such as "{{" or ")" drawn
next to the letters.

Heavy, thick lines are district boundaries, never leaders: do not follow them. A leader is a thin
line or an arrow starting at the letters. A bracket points into the area immediately beside it,
usually a narrow strip just past the bracket.

Does this code have such a leader? If yes, give the pixel where the leader ends: the point inside the
area the code names (at the arrowhead, or just past the end of the line or the bracket's tip). If the
code simply sits inside its area, answer has_leader false and give the code's own centre.
(effort: {effort})"""


class Leader(BaseModel):
    has_leader: bool
    x: int
    y: int
    note: str


def crop(img: np.ndarray, x: float, y: float) -> tuple[np.ndarray, int, int]:
    h, w = img.shape
    x0, y0 = int(max(0, x - HALF)), int(max(0, y - HALF))
    x1, y1 = int(min(w, x + HALF)), int(min(h, y + HALF))
    c = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    c = cv2.resize(c, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    cv2.circle(c, (int((x - x0) * UP), int((y - y0) * UP)), 34, (0, 0, 255), 2)
    return c, x0, y0


def ask(img, code, x, y, effort, client):
    c, x0, y0 = crop(img, x, y)
    h, w = c.shape[:2]
    a = llm.ask_json(c, PROMPT.format(w=w, h=h, code=code, effort=effort),       # BGR, as ask_json encodes it
                     Leader, client=client)
    return a, (x0 + a.x / UP, y0 + a.y / UP), c


def page(sid):
    return cv2.imread(str(config.WORK_DIR / SERIES / sid / "page.png"), cv2.IMREAD_GRAYSCALE)


TESTS = [   # sheet, code, printed x, y, expected target (None: no leader)
    ("p03", "RH-2", 1832, 3066, (1798, 3089), "Clement at Arguello: arrow to the small box"),
    ("p11", "NC-2", 1002, 1514, (1039, 1522), "Glen Park: bracket to the slim strip"),
    ("p11", "NC-1", 1180, 2493, "west", "Excelsior: leader back into the Mission St strip"),
    ("p12", "NC-1", 2178, 1792, None, "Monterey: printed in its corridor"),
    ("p11", "RH-1", 920, 1256, None, "Glen Park hillside RH-1: no leader"),
]


def main():
    effort = sys.argv[-1] if sys.argv[-1] in ("low", "medium", "high") else "low"
    if effort != config.VISION_REASONING_EFFORT:
        config.VISION_REASONING_EFFORT = effort
        llm.VISION_REASONING_EFFORT = effort
    client = llm.OpenAIClient()
    (S / "leaders").mkdir(exist_ok=True)
    if sys.argv[1] == "test":
        for sh, code, x, y, want, note in TESTS:
            img = page(f"{SERIES}-{sh}")
            a, (tx, ty), c = ask(img, code, x, y, effort, client)
            cv2.circle(c, (int((tx - max(0, x - HALF)) * UP), int((ty - max(0, y - HALF)) * UP)), 8, (255, 0, 0), -1)
            cv2.imwrite(str(S / "leaders" / f"test_{sh}_{code}_{effort}.png"), c)
            print(f"{note}\n   leader={a.has_leader} -> ({tx:.0f}, {ty:.0f})  expected {want}  | {a.note}")
    else:
        sid = sys.argv[2]
        img = page(sid)
        out = []
        for l in L.use_code_labels(L._read_labels(SERIES, sid)):
            a, (tx, ty), _ = ask(img, l.code, l.x, l.y, effort, client)
            out.append({"code": l.code, "x": l.x, "y": l.y, "leader": a.has_leader, "tx": tx, "ty": ty, "note": a.note})
        (S / "leaders" / (f"{sid}.json" if effort == "medium" else f"{sid}_{effort}.json")).write_text(json.dumps(out, indent=1))
        print(sid, len(out), "labels,", sum(o["leader"] for o in out), "with leaders")
    print(f"spent ~${llm.CallCounter.usd():.2f}")


if __name__ == "__main__":
    main()
