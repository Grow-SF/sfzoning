import json

import numpy as np
import pytest

from sfzmaps import config, evaluate
from sfzmaps.pages import Page
from sfzmaps.transform import GCP, SheetTransform

# pixel (x, y) -> world (6e6 + 0.5 x, 2.1e6 - 0.5 y)
TRUE = [GCP(x, y, 6_000_000 + 0.5 * x, 2_100_000 - 0.5 * y, "auto")
        for x in (0, 1000, 2000) for y in (0, 1000, 2000)]
SCORE = {"n_scored": 100, "frac_within": 0.97, "p50_ft": 2.0, "p95_ft": 5.5, "max_ft": 9.0,
         "n_real_change": 3, "n_no_equivalent": 7, "passed": True, "reason": None}


def _points(rows):
    lines = ["#CRS: EPSG:2227", "mapX,mapY,pixelX,pixelY,enable"]
    lines += [f"{wx},{wy},{px},{-py},{e}" for wx, wy, px, py, e in rows]
    return "\n".join(lines) + "\n"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    for n in ("WORK_DIR", "RAW_DIR"):
        monkeypatch.setattr(config, n, tmp_path / n.lower())
    aligned = config.RAW_DIR / "1948" / "aligned"
    aligned.mkdir(parents=True)
    pages = []
    for sid in ("1", "2", "3"):
        src = aligned / f"{sid}.jpg"
        src.write_bytes(b"x")
        pages.append(Page("1948", sid, src, None))
        (config.WORK_DIR / "1948" / sid).mkdir(parents=True)
    # sheet 1: our transform is exact; the owner's points are 3 ft and 4 ft off; one disabled
    d = config.WORK_DIR / "1948" / "1"
    (d / "score.json").write_text(json.dumps(SCORE))
    (d / "transform.json").write_text(json.dumps({**SheetTransform(TRUE, 1.0).to_json(), "origin": "auto"}))
    (d / "page.png").write_bytes(b"x")
    (aligned / "1.jpg.points").write_text(_points([
        (6_000_000 + 3, 2_100_000, 0, 0, 1),
        (6_000_500, 2_099_500 + 4, 1000, 1000, 1),
        (6_000_500 + 1, 2_099_500, 1000, 1000, 1),
        (6_000_000 + 5000, 2_100_000, 0, 0, 0)]))
    # sheet 2: scored, no answer key, no transform
    (config.WORK_DIR / "1948" / "2" / "score.json").write_text(
        json.dumps({**SCORE, "passed": False, "reason": "below_threshold", "frac_within": 0.6}))
    # sheet 3: never run (empty dir) -> no row
    return pages


def test_eval_table(workdir):
    rows = evaluate.evaluate_series("1948", workdir)
    assert [r.sheet for r in rows] == ["1", "2"]
    r1, r2 = rows
    assert r1.origin == "auto" and r1.n_key == 3
    assert r1.key_median_ft == pytest.approx(3.0, abs=1e-6)
    assert r1.key_max_ft == pytest.approx(4.0, abs=1e-6)
    assert r2.n_key == 0 and np.isnan(r2.key_median_ft) and r2.note == "no answer key"
    table = evaluate.format_eval_table(rows)
    lines = table.splitlines()
    assert lines[0].split()[:4] == ["sheet", "frac_within", "p50", "p95"]
    one = lines[1].split()
    assert one[:9] == ["1", "0.970", "2.0", "5.5", "9.0", "100", "3", "7", "pass"]
    assert one[9:13] == ["auto", "3", "3.0", "4.0"]
    assert "below_threshold" in lines[2] and "no answer key" in lines[2]
    assert lines[-1].startswith("passed 1/2; median p95 5.5 ft; median key distance 3.0 ft")


def test_eval_answer_key_origin_is_flagged(workdir):
    d = config.WORK_DIR / "1948" / "1"
    data = json.loads((d / "transform.json").read_text())
    (d / "transform.json").write_text(json.dumps({**data, "origin": "answer_key"}))
    r1 = evaluate.evaluate_series("1948", workdir)[0]
    assert r1.note == "fit to the key itself"
    # the key's own fit is excluded from the median key distance
    assert "median key distance -" in evaluate.format_eval_table([r1])


def test_eval_cli_rejects_other_years():
    from typer.testing import CliRunner

    from sfzmaps.cli import app

    res = CliRunner().invoke(app, ["eval", "--year", "1921"])
    assert res.exit_code != 0
