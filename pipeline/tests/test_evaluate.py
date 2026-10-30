import json

import numpy as np
import pytest

from sfzmaps import config, evaluate, runner
from sfzmaps.pages import Page
from sfzmaps.transform import GCP, SheetTransform

# pixel (x, y) -> world (6e6 + 0.5 x, 2.1e6 - 0.5 y)
TRUE = [GCP(x, y, 6_000_000 + 0.5 * x, 2_100_000 - 0.5 * y, "auto")
        for x in (0, 1000, 2000) for y in (0, 1000, 2000)]
SCORE = {"n_scored": 100, "insample_frac_within": 0.97, "insample_p50_ft": 2.0,
         "insample_p95_ft": 5.5, "insample_max_ft": 9.0, "n_real_change": 3,
         "n_no_equivalent": 7, "passed": True, "reason": None, "holdout_p50_ft": 3.0,
         "holdout_p95_ft": 8.0, "n_holdout": 100, "score_version": config.SCORE_VERSION, "label_agreement": 0.8, "inlier_frac": 0.5}


def _points(rows):
    lines = ["#CRS: EPSG:2227", "mapX,mapY,pixelX,pixelY,enable"]
    # rows hold our pixel-center coordinates; QGIS files are corner-based (+0.5 px)
    lines += [f"{wx},{wy},{px + 0.5},{-(py + 0.5)},{e}" for wx, wy, px, py, e in rows]
    return "\n".join(lines) + "\n"


KEY = [(6_000_000 + 3, 2_100_000, 0, 0, 1),
       (6_000_500, 2_099_500 + 4, 1000, 1000, 1),
       (6_000_500 + 1, 2_099_500, 1000, 1000, 1),
       (6_000_000 + 5000, 2_100_000, 0, 0, 0)]


def _transform(d, origin="auto"):
    (d / "transform.json").write_text(json.dumps({**SheetTransform(TRUE, 1.0).to_json(), "origin": origin}))


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
    # sheet 1: our transform is exact; the owner's points are 3 ft and 4 ft off; one disabled.
    # No page.png: a .jpg key is in source pixels and needs no render.
    d = config.WORK_DIR / "1948" / "1"
    (d / "score.json").write_text(json.dumps(SCORE))
    _transform(d)
    (aligned / "1.jpg.points").write_text(_points(KEY))
    # sheet 2: scored, no answer key, no transform
    (config.WORK_DIR / "1948" / "2" / "score.json").write_text(
        json.dumps({**SCORE, "passed": False, "reason": "holdout_median", "holdout_p50_ft": 6.0}))
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
    lines = evaluate.format_eval_table(rows).splitlines()
    assert lines[0].split()[:5] == ["sheet", "pass", "reason", "ho_p50", "ho_p95"]
    one = lines[1].split()
    assert one[:15] == ["1", "yes", "-", "3.0", "8.0", "5.0", "12.0", "2.0", "5.5", "0.97", "100", "3",
                        "7", "0.80", "0.50"]
    assert one[15:19] == ["auto", "3", "3.0", "4.0"]
    assert "holdout_median" in lines[2] and "no answer key" in lines[2]
    assert lines[-1].startswith("passed 1/2; median held-out p50 4.5 ft, p95 8.0 ft (in-sample p95 5.5 ft); "
                                "median key distance 3.0 ft")


def test_eval_reads_old_score_files(workdir):
    d = config.WORK_DIR / "1948" / "1"
    (d / "score.json").write_text(json.dumps({"n_scored": 10, "frac_within": 0.5, "p50_ft": 1.0,
                                              "p95_ft": 2.0, "max_ft": 3.0, "n_real_change": 0,
                                              "n_no_equivalent": 0, "passed": False,
                                              "reason": "below_threshold"}))
    r1 = evaluate.evaluate_series("1948", workdir)[0]
    assert r1.score.insample_p95_ft == 2.0 and np.isnan(r1.score.holdout_p50_ft)


def test_eval_answer_key_origin_is_flagged(workdir):
    _transform(config.WORK_DIR / "1948" / "1", origin="answer_key")
    r1 = evaluate.evaluate_series("1948", workdir)[0]
    assert r1.note == "fit to the key itself"
    assert "median key distance -" in evaluate.format_eval_table([r1])


def test_eval_key_without_transform(workdir):
    (config.WORK_DIR / "1948" / "1" / "transform.json").unlink()
    r1 = evaluate.evaluate_series("1948", workdir)[0]
    assert r1.n_key == 3 and r1.note == "no transform" and np.isnan(r1.key_median_ft)


def test_eval_skipped_sheet_has_no_row(workdir):
    (config.WORK_DIR / "1948" / "2" / "score.json").write_text(
        json.dumps({**SCORE, "passed": False, "reason": "skipped: legend"}))
    assert [r.sheet for r in evaluate.evaluate_series("1948", workdir)] == ["1"]


def test_eval_just_2_points_filename(workdir, monkeypatch):
    # Sheet 2 is cut from the two-page photo 2-and-3.jpg (3.jpg is the other page); its key
    # "2-and-3.jpg.just-2.points" is in the photo's pixels, shifted by the crop origin.
    from sfzmaps import pages

    aligned = config.RAW_DIR / "1948" / "aligned"
    src = aligned / "2-and-3.jpg"
    src.write_bytes(b"x")
    page = Page("1948", "2", src, None, other=aligned / "3.jpg")
    monkeypatch.setattr(pages, "crop_box", lambda p: (100, 40, 5000, 7000) if p.other else None)
    d = config.WORK_DIR / "1948" / "2"
    (d / "score.json").write_text(json.dumps(SCORE))
    _transform(d)
    (aligned / "2-and-3.jpg.just-2.points").write_text(
        _points([(wx, wy, px + 100, py + 40, e) for wx, wy, px, py, e in KEY]))
    assert runner.answer_key_path(page) == aligned / "2-and-3.jpg.just-2.points"
    r = evaluate.eval_sheet(page)
    assert r.n_key == 3 and r.key_median_ft == pytest.approx(3.0, abs=1e-6)
    gcps = runner._answer_key_gcps(page, d / "page.png")
    assert (gcps[1].px, gcps[1].py) == (1000, 1000)
    # the photo's key is not the answer key of an uncropped page with another id
    assert runner.answer_key_path(Page("1948", "2-and-3", src, None)) is None


def test_eval_pdf_key_is_rescaled(workdir, tmp_path, monkeypatch):
    # R7: a 1960 PDF key is in GDAL's raster of the PDF; our render is 2x wider here
    import cv2

    aligned = config.RAW_DIR / "1960" / "aligned"
    aligned.mkdir(parents=True)
    src = aligned / "4.pdf"
    src.write_bytes(b"%PDF")
    page = Page("1960", "4", src, None)
    d = config.WORK_DIR / "1960" / "4"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "page.png"), np.zeros((10, 4000), np.uint8))
    monkeypatch.setattr(runner, "_gdal_width", lambda pdf: 2000)
    (d / "score.json").write_text(json.dumps(SCORE))
    _transform(d)
    # key pixels are half of ours, corner-based: (500.25, 500.25) in the file is our
    # pixel center (1000, 1000) (scaled 2x, then -0.5)
    (aligned / "4.pdf.points").write_text("\n".join([
        "mapX,mapY,pixelX,pixelY,enable", "6000500,2099500,500.25,-500.25,1",
        "6000000,2100000,0.25,-0.25,1", "6001000,2099000,1000.25,-1000.25,1"]) + "\n")
    r = evaluate.eval_sheet(page)
    assert r.n_key == 3 and r.key_max_ft == pytest.approx(0.0, abs=1e-6)
    # without page.png a PDF key cannot be rescaled
    (d / "page.png").unlink()
    assert evaluate.eval_sheet(page).note == "answer key unreadable"


def test_eval_cli_rejects_other_years():
    from typer.testing import CliRunner

    from sfzmaps.cli import app

    res = CliRunner().invoke(app, ["eval", "--year", "1921"])
    assert res.exit_code != 0


def test_eval_summary_prints_scaled_limits(workdir):
    d = config.WORK_DIR / "1948" / "2"
    (d / "score.json").write_text(json.dumps({**SCORE, "pass_median_limit_ft": 7.4,
                                              "pass_p95_limit_ft": 17.8, "pass_scale_k": 1.48}))
    last = evaluate.format_eval_table(evaluate.evaluate_series("1948", workdir)).splitlines()[-1]
    assert "held-out median <= 5.0-7.4 ft and p95 <= 12.0-17.8 ft" in last
