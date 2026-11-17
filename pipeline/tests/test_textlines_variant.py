"""scripts/textlines/variant.py on the synthetic `maps` fixture: a 2-step CPU training and a stub for the gate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import variant  # noqa: E402
from test_solver_textnet import OLD, TINY, maps  # noqa: E402,F401

from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import data, reader_stage  # noqa: E402


def test_a_run_makes_its_folder(maps, monkeypatch):  # noqa: F811
    monkeypatch.setattr(variant, "RUNS_DIR", maps / "runs")
    monkeypatch.setattr(variant, "LINES", maps / "lines")
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    sd = next(s for s in sheets if s.series == paths.SERIES)
    counts = [(reader_stage.page_quarters(sd) == i).sum() for i in range(4)]
    q = int(np.argmax(counts))           # the synthetic pages lie in the quarter their placement puts them in
    quarter = paths.QUARTERS[q]
    seen = {}

    def stub(out, quarter, scored, chances, extra):
        seen.update(quarter=quarter, extra=[sd.series for sd in extra])
        (out / "scores.json").write_text(json.dumps({"stub": True}))

    recipe = variant.dataclasses.replace(TINY, labels="lines-unknown")
    out = variant.run("t", recipe, quarter, 0, device="cpu", sheets=sheets, areas=(data.Quarter(q),), score=stub,
                      extra_series=(OLD,))
    assert out.name == f"t-{quarter}-s0"
    run = json.loads((out / "run.json").read_text())
    assert (out / "model.pt").is_file() and (out / "scores.json").is_file()
    assert run["seed"] == 0 and run["recipe"]["labels"] == "lines-unknown" and "git_head" in run
    assert seen == {"quarter": quarter, "extra": [OLD]}
    chance = cv2.imread(str(out / "chance" / f"{sd.sheet}.png"), cv2.IMREAD_UNCHANGED)
    assert chance.shape == (sd.height, sd.width)
    assert not chance[reader_stage.page_quarters(sd) != q].any()
    assert (out / "chance" / OLD / "3.png").is_file()
