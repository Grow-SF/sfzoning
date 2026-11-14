"""How the answers with the marks red (rerun.sh) differ from the originals (mark blue).

Each new answer is paired with the original asked by the same model, at the same effort, about the same crop: the two
images are identical once the mark's coloured pixels are blanked (the crop is the same; only the mark's colour moved).
Reported per read and model:
  leader map    the code read; whether a leader was seen; where it points (moved more than 15 page px)
  re-check      the code read; and, on the cases the owner answered on the code-check page, how often each round's
                code is the owner's (ok: the case's code; fix: the owner's code; nolabel: no code)
  leader reads  whether a leader was seen; where it points
usage: python -m saved_reads.rerun.compare PACKAGE IMAGES CALLS ANSWERS.json CASES.json"""
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config


def blanked(path: Path) -> str:
    """the image's hash with every coloured pixel (the mark) made white"""
    a = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(int)
    mark = (a.max(2) - a.min(2)) > 60
    g = a.mean(2).astype(np.uint8)
    g[mark] = 255
    return hashlib.sha256(g.tobytes() + bytes(str(g.shape), "ascii")).hexdigest()


def norm(code) -> str:
    c = "".join(str(code or "").upper().split())
    return c if c not in ("", "NONE", "NULL") else "NONE"


def kind_of(template: str) -> str:
    return {"code-reads-t1": "leader map", "code-reads-t2": "re-check"}.get(template, "leader reads")


def main(package: Path, images: Path, calls: Path, answers: Path, cases: Path) -> None:
    old = {}                                         # (kind, model, effort, blanked image) -> original record
    for f in ("code-reads", "leader-reads"):
        for line in gzip.open(package / "reads" / f"{f}.jsonl.gz", "rt"):
            r = json.loads(line)
            old[(kind_of(r["prompt"]), r["model"], r["effort"], blanked(images / r["image"]))] = r
    kinds = {"leaders2.py": "leader map", "reread.py": "re-check", "leaders.py": "leader reads"}
    pairs = defaultdict(list)
    unpaired = Counter()
    for folder in sorted(calls.glob("*-fixed*")):
        runs = {}
        for line in open(folder / "runs.jsonl"):
            r = json.loads(line)
            runs.setdefault(r["run"], {}).update(r)
        for line in open(folder / "calls.jsonl"):
            c = json.loads(line)
            ans = config.CACHE_DIR / c["key"]
            if not ans.exists():
                continue                             # asked, not answered yet
            kind = kinds[Path(runs[c["run"]]["script"]).name]
            k = (kind, c["model"][1], c["effort"], blanked(folder / "images" / f"{c['image']}.png"))
            if k not in old:
                unpaired[kind] += 1
                continue
            pairs[(kind, c["model"][1], c["effort"])].append((old[k], json.loads(ans.read_text())))
    truth = {}
    pub = {c["id"]: c for c in json.load(open(cases))}
    for cid, a in json.load(open(answers)).items():
        if a["answer"] in ("ok", "fix", "nolabel") and cid in pub:
            truth[cid] = norm({"ok": pub[cid]["code"], "fix": a.get("code"), "nolabel": None}[a["answer"]])
    for (kind, model, effort), ps in sorted(pairs.items()):
        seen = {}
        for o, n in ps:
            seen[(o["image"])] = (o, n)              # a crop asked twice in a run counts once
        ps = list(seen.values())
        line = f"{kind:12s} {model:14s} {effort:6s} {len(ps):5d} pairs:"
        if kind == "re-check":
            ch = sum(norm(o["answer"]["code"]) != norm(n["code"]) for o, n in ps)
            line += f" code differs {ch} ({ch / len(ps):.1%})"
            on = [(o, n) for o, n in ps if o.get("case") in truth]
            if on:
                ro = sum(norm(o["answer"]["code"]) == truth[o["case"]] for o, n in on)
                rn = sum(norm(n["code"]) == truth[o["case"]] for o, n in on)
                line += f"; on {len(on)} owner-answered cases: blue mark {ro / len(on):.1%} right, red {rn / len(on):.1%}"
        else:
            if kind == "leader map":
                ch = sum(norm(o["answer"]["code_read"]) != norm(n["code_read"]) for o, n in ps)
                line += f" code differs {ch} ({ch / len(ps):.1%});"
            lo = sum(bool(o["answer"]["has_leader"]) for o, n in ps)
            ln = sum(bool(n["has_leader"]) for o, n in ps)
            flip = sum(bool(o["answer"]["has_leader"]) != bool(n["has_leader"]) for o, n in ps)
            moved = sum(o["answer"]["has_leader"] and n["has_leader"] and
                        np.hypot(o["answer"]["x"] - n["x"], o["answer"]["y"] - n["y"]) > 30 for o, n in ps)
            line += (f" leader seen {lo} -> {ln} (changed on {flip}, {flip / len(ps):.1%});"
                     f" both see one but it points elsewhere: {moved}")
        print(line)
    if unpaired:
        print("not paired with an original:", dict(unpaired))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:6]))
