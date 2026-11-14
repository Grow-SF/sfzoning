# saved_reads/rerun: a copy of prob-warp:pipeline/scripts/prob-warp/codes2000/leaders2.py with one fix, 2026-11-13: the image goes to llm.ask_json in BGR, the order it
# encodes (cv2.imencode).  The original handed it RGB, so the mark the prompt calls red reached the model blue.
"""Leader prompt iteration against a high-effort reference, on the codes medium and low disagree on.

usage: leaders2.py set                       build the disagreement set -> leaders2/set.json
       leaders2.py ref                       high-effort reference -> leaders2/ref.json
       leaders2.py try PROMPT_FILE [effort]  low (default) with that prompt; scores against ref
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from sfzmaps import config, llm

S = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
D = S / "leaders2"
SERIES = "2000-digital-zoning"
HALF, UP = 160, 2
FAR_PX = 15


class Read(BaseModel):
    code_read: str
    has_leader: bool
    x: int
    y: int
    note: str


def crop(sid, x, y):
    img = cv2.imread(str(config.WORK_DIR / SERIES / sid / "page.png"), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    x0, y0 = int(max(0, x - HALF)), int(max(0, y - HALF))
    x1, y1 = int(min(w, x + HALF)), int(min(h, y + HALF))
    c = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    c = cv2.resize(c, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    cv2.circle(c, (int((x - x0) * UP), int((y - y0) * UP)), 34, (0, 0, 255), 2)
    return c, x0, y0


def set_effort(effort):
    config.VISION_REASONING_EFFORT = effort
    llm.VISION_REASONING_EFFORT = effort


def run(cases, prompt, effort):
    set_effort(effort)
    client = llm.OpenAIClient()

    def one(c):
        im, x0, y0 = crop(c["sid"], c["x"], c["y"])
        h, w = im.shape[:2]
        p = prompt.replace("{w}", str(w)).replace("{h}", str(h)).replace("{code}", c["code"]) + f"\n(effort: {effort})"
        a = llm.ask_json(im, p, Read, client=client)                  # BGR, as ask_json encodes it
        return {**c, "read": a.code_read, "leader": a.has_leader, "tx": x0 + a.x / UP, "ty": y0 + a.y / UP, "note": a.note}

    with ThreadPoolExecutor(int(__import__("os").environ.get("THREADS", 8))) as ex:
        return list(ex.map(one, cases))


def norm(s):
    return s.upper().replace(" ", "").replace("–", "-")


def _eff(c):
    """Where an answer puts the code: the leader's end, else the code itself."""
    return (c["tx"], c["ty"]) if c["leader"] else (c["x"], c["y"])


def score(out, ref):
    r = {(c["sid"], c["x"], c["y"]): c for c in ref}
    lead = pt = txt = 0
    bad = []
    for c in out:
        q = r[(c["sid"], c["x"], c["y"])]
        ok_l = c["leader"] == q["leader"]
        ok_p = ok_l and (not q["leader"] or np.hypot(c["tx"] - q["tx"], c["ty"] - q["ty"]) <= FAR_PX)
        ok_t = norm(c["read"]) == norm(q["read"])
        lead += ok_l
        pt += ok_p
        txt += ok_t
        if np.hypot(*np.subtract(_eff(c), _eff(q))) > FAR_PX or not ok_t:
            bad.append((c, q))
    n = len(out)
    eff = sum(np.hypot(*(np.subtract(_eff(c), _eff(r[(c["sid"], c["x"], c["y"])])))) <= FAR_PX for c in out)
    print(f"vs high: leader-or-not {lead}/{n}, leader and point {pt}/{n}, text {txt}/{n}, "
          f"same place for the code {eff}/{n}")
    return bad


def main():
    D.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "set":
        cases = []
        for n in range(1, 14):
            sid = f"{SERIES}-p{n:02d}"
            m = json.loads((S / "leaders" / f"{sid}.json").read_text())
            lo = {(r["x"], r["y"]): r for r in json.loads((S / "leaders" / f"{sid}_low.json").read_text())}
            for a in m:
                b = lo.get((a["x"], a["y"]))
                if b is None:
                    continue
                if a["leader"] != b["leader"] or (a["leader"] and np.hypot(a["tx"] - b["tx"], a["ty"] - b["ty"]) > FAR_PX):
                    cases.append({"sid": sid, "code": a["code"], "x": a["x"], "y": a["y"]})
        (D / "set.json").write_text(json.dumps(cases, indent=1))
        print(len(cases), "cases")
        return
    if cmd == "map":  # map MODEL PROMPT_FILE effort : every use code on every 2000 sheet
        from sfzmaps.districts import labels as L
        model, pf, effort = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
        llm.VISION_MODEL = model
        out_dir = S / "leaders3"
        out_dir.mkdir(exist_ok=True)
        for n in range(1, 14):
            sid = f"{SERIES}-p{n:02d}"
            f = out_dir / f"{model}_{sid}.json"
            if f.exists():
                continue
            cs = [{"sid": sid, "code": l.code, "x": l.x, "y": l.y}
                  for l in L.use_code_labels(L._read_labels(SERIES, sid))]
            out = run(cs, pf.read_text(), effort)
            f.write_text(json.dumps(out, indent=1))
            print(sid, len(out), "codes;", sum(c["leader"] for c in out), "leaders; spent so far",
                  f"~${llm.CallCounter.usd():.2f}", flush=True)
        print(f"spent ~${llm.CallCounter.usd():.2f}")
        return
    cases = json.loads((D / "set.json").read_text())
    if cmd == "ref":
        pf = Path(sys.argv[2]) if len(sys.argv) > 2 else D / "prompt_ref.txt"
        out = run(cases, pf.read_text(), "high")
        name = "ref.json" if len(sys.argv) <= 2 else f"ref_{pf.stem}.json"
        (D / name).write_text(json.dumps(out, indent=1))
        print(len(out), "ref answers;", sum(c["leader"] for c in out), "leaders")
    elif cmd == "model":  # model MODEL PROMPT_FILE effort : only the hand-answered cases
        import glob
        model, pf, effort = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
        llm.VISION_MODEL = model
        done = {Path(f).stem for f in glob.glob(str(S / "truth/labels/*.json"))}
        sub = [c for i, c in enumerate(cases) if f"c{i:03d}" in done]
        out = run(sub, pf.read_text(), effort)
        (D / f"m_{model}_{pf.stem}_{effort}.json").write_text(json.dumps(out, indent=1))
        print(model, pf.stem, effort, len(out), "answers;", sum(c["leader"] for c in out), "leaders")
    elif cmd == "try":
        pf = Path(sys.argv[2])
        effort = sys.argv[3] if len(sys.argv) > 3 else "low"
        out = run(cases, pf.read_text(), effort)
        (D / f"try_{pf.stem}_{effort}.json").write_text(json.dumps(out, indent=1))
        ref = D / (sys.argv[4] if len(sys.argv) > 4 else "ref.json")
        bad = score(out, json.loads(ref.read_text()))
        (D / f"bad_{pf.stem}_{effort}.json").write_text(json.dumps(bad, indent=1))
    print(f"spent ~${llm.CallCounter.usd():.2f}")


if __name__ == "__main__":
    main()
