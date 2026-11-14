"""Before a paid rerun: the marks the prompts call red must reach the model red.  Each rerun script's crop is made
for one 2000 code and encoded exactly as llm.ask_json sends it (llm._encode); the coloured pixels of the PNG are read
back as RGB.
usage: python -m saved_reads.rerun.check_marks   (from pipeline/scripts)"""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config, llm

HERE = Path(__file__).parent
SHEET, X, Y = "2000-digital-zoning-p07", 300.0, 2048.0


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = [str(path), "--no-main"]
    spec.loader.exec_module(m)
    return m


def colours(img_sent) -> set:
    png = llm._encode(img_sent)
    a = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]     # back to RGB
    px = a.reshape(-1, 3)
    return {tuple(int(v) for v in p) for p in px[(px.max(1) - px.min(1)) > 100][:5000]}


def as_sent(name: str, m):
    """the image each script hands to llm.ask_json, made as the script makes it"""
    if name == "leaders2":
        im, _, _ = m.crop(SHEET, X, Y)
        return im if "rerun" in m.__file__ else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if name == "leaders":
        page = cv2.imread(str(config.WORK_DIR / "2000-digital-zoning" / SHEET / "page.png"), cv2.IMREAD_GRAYSCALE)
        c, _, _ = m.crop(page, X, Y)
        return c if "rerun" in m.__file__ else cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    return m.crop(SHEET, X, Y)                                                         # reread sends crop() as is


if __name__ == "__main__":
    for name in ("leaders2", "reread", "leaders"):
        fixed = colours(as_sent(name, load(HERE / f"{name}.py", f"fixed_{name}")))
        print(f"{name:9s} rerun copy sends {sorted(fixed)}  {'RED: ok' if fixed == {(255, 0, 0)} else 'NOT RED'}")
