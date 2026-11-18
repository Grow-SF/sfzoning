# saved_reads/rerun: a copy of prob-warp:pipeline/scripts/prob-warp/codes2000/reread.py with one fix, 2026-11-13: the image goes to llm.ask_json in BGR, the order it
# encodes (cv2.imencode).  The original handed it RGB, so the mark the prompt calls red reached the model blue.
"""Blind re-read of each code from a full-resolution, enlarged crop (the label pass reads the
2000 sheets at half scale).  The model is NOT told what was read; it picks from the map's codes
or says none.  Scored against the code-check answers (truth2/reads).

usage: reread.py MODEL EFFORT [all]   (default: only the hand-checked codes)
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from pydantic import BaseModel

from sfzmaps import config, llm

SP = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
HALF, UP = 90, 4
CODES = ["P", "RH-1", "RH-1(D)", "RH-1(S)", "RH-2", "RH-3", "RM-1", "RM-2", "RM-3", "RM-4", "RC-1", "RC-2",
         "RC-3", "RC-4", "NC-1", "NC-2", "NC-3", "NC-S", "NCD", "M-1", "M-2", "C-1", "C-2", "C-M", "C-3-G",
         "C-3-O", "C-3-R", "C-3-S", "SLI", "SLR", "SSO", "RED", "CCB", "CVR", "RSD", "SPD", "MB-O"]
PROMPT = f"""This is a crop of a 1-bit scan of San Francisco's 2000 zoning map, enlarged 4x.
Somewhere near the centre (marked by four small red ticks at the edges of the image), a zoning
district code is printed in bold capitals, possibly rotated or italic. Lot numbers (plain digits),
street names and district boundary lines are also printed; they are not codes.

Read the district code printed nearest the centre, character by character. Look closely at digits:
1, 2, 3 and 4 differ; RH and RM differ (H has two straight uprights; M has a V in the middle).
Codes on this map are usually one of: {", ".join(CODES)} — or a named district such as
"UNION NCD". If no district code is printed within about a quarter of the image from the centre,
answer "NONE". Give "code" exactly as printed (normalised to that list's spelling when it matches)."""


class Read(BaseModel):
    letters_seen: str
    code: str
    confidence: float


def crop(sid, x, y, pages={}):
    if sid not in pages:
        pages[sid] = cv2.imread(str(config.WORK_DIR / "2000-digital-zoning" / sid / "page.png"), cv2.IMREAD_GRAYSCALE)
    img = pages[sid]
    h, w = img.shape
    x0, y0 = int(x) - HALF, int(y) - HALF
    c = 255 * __import__("numpy").ones((2 * HALF, 2 * HALF), "uint8")
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + 2 * HALF), min(h, y0 + 2 * HALF)
    c[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    c = cv2.resize(c, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)                  # BGR, as ask_json encodes it
    m, n = c.shape[0] // 2, 14
    for a, b in [((m, 0), (m, n)), ((m, c.shape[0] - n), (m, c.shape[0])),
                 ((0, m), (n, m)), ((c.shape[1] - n, m), (c.shape[1], m))]:
        cv2.line(c, a, b, (0, 0, 255), 3)                        # red, in BGR
    return c


def main():
    model, effort = sys.argv[1], sys.argv[2]
    llm.VISION_MODEL = model
    config.VISION_REASONING_EFFORT = llm.VISION_REASONING_EFFORT = effort
    cases = json.loads((SP / "codecheck/cases.json").read_text())
    if "all" not in sys.argv:
        done = {Path(f).stem for f in glob.glob(str(SP / "truth2/reads/*.json"))}
        cases = [c for c in cases if c["id"] in done]
    client = llm.OpenAIClient()
    p = PROMPT + f"\n(effort: {effort})"

    def one(c):
        a = llm.ask_json(crop("2000-digital-zoning-" + c["sid"], c["x"], c["y"]), p, Read, client=client)
        return {**c, "reread": a.code, "seen": a.letters_seen, "conf": a.confidence}

    with ThreadPoolExecutor(int(os.environ.get("THREADS", 8))) as ex:
        out = list(ex.map(one, cases))
    name = f"reread_{model}_{effort}{'_all' if 'all' in sys.argv else ''}.json"
    (SP / name).write_text(json.dumps(out, indent=1))
    print(name, len(out), f"spent ~${llm.CallCounter.usd():.2f}")


if __name__ == "__main__":
    main()
