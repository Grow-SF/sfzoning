"""Trial data roots for placement (`find.trial`, `sfzmaps find trial`): a root holds copies (never links) of a sheet's
work files up to its blocks, links only raw/ ref/ cache/, refuses a root inside today's data and a sheet that does not
test the blocks seam, and is prepared once; the numbers are read from a root's own files, the same way for every root;
two roots are compared file by file with no tolerance; the share of a sheet's ground under the run's lot-labelled
training sheets and the run's hand crops on it."""
import json
import os
import shutil

import numpy as np
import pytest
from shapely.geometry import box, mapping

from find_fixtures import invoke
from sfzmaps import config, inputs
from sfzmaps.find import pages, trial
from sfzmaps.find.trial import KEPT

DOWNSTREAM = ("rough.json", "refine_raw.json", "refine.json", "changes.geojson", "transform.json", "score.json",
              "warped.tif", "warped.footprint.geojson")
SIDECARS = ("page.png.inputs.json", "page.json.inputs.json", "page.json.answers", "labels.json.inputs.json",
            "labels.json.answers", "blocks_px.geojson.inputs.json", "warped_paper.tif.inputs.json")


def _transform(gcps):
    return {"smoothing": 0.1, "gcps": [{"px": x, "py": y, "wx": x * 2, "wy": y * 2, "source": "auto", "enabled": e}
                                       for x, y, e in gcps], "origin": "auto"}


def _fill(root, series="s", sheet="a", chosen="probabilistic"):
    """A placed sheet in data root `root`: every KEPT file, the sidecars, every downstream work file and a warp
    folder, with known numbers (3 blocks, 2 corners, 2 of 3 control points enabled)."""
    d = root / "work" / series / sheet
    d.mkdir(parents=True)
    for f in (*KEPT, *SIDECARS, *DOWNSTREAM):
        (d / f).write_text(f"{f} of {sheet}")
    blocks = {"type": "FeatureCollection", "corners": [[1, 2], [None, 3]],
              "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(i, 0, i + 1, 1))}
                           for i in range(3)]}
    (d / "blocks_px.geojson").write_text(json.dumps(blocks))
    (d / "transform.json").write_text(json.dumps(_transform([(1.5, 2.5, True), (3, 4, False), (5, 6, True)])))
    (d / "refine.json").write_text(json.dumps({"matches": [{"px": 1.5, "py": 2.5, "label": "inlier"}]}))
    (d / "score.json").write_text(json.dumps({"passed": True, "reason": None, "holdout_p50_ft": 5.01,
                                              "holdout_p95_ft": 12.28}))
    w = root / "warp" / series / sheet
    w.mkdir(parents=True)
    (w / "scores.json").write_text(json.dumps({"within1_first": 0.6, "within1_probabilistic": 0.627,
                                               "chosen": chosen, "device": "mps"}))
    np.savez(w / "correction.npz", Gc=np.arange(6, dtype=np.float32).reshape(2, 3), spacing=np.int64(4),
             kind=np.str_(chosen))


@pytest.fixture
def src(tmp_path):
    root = tmp_path / "src"
    for x in ("raw", "ref", "cache"):
        (root / x).mkdir(parents=True)
    _fill(root)
    return root


def test_a_trial_root_holds_copies_of_the_inputs_and_nothing_after_them(src, tmp_path):
    root = tmp_path / "t" / "traced"
    copied = trial.prepare(root, [("s", "a")], source=src)
    d = root / "work/s/a"
    assert {p.name for p in d.iterdir()} == {*KEPT, *SIDECARS}
    assert sorted(copied) == sorted(d.iterdir())
    assert not any((d / f).is_symlink() or os.stat(d / f).st_ino == os.stat(src / "work/s/a" / f).st_ino
                   for f in (*KEPT, *SIDECARS))
    (d / "page.json").write_text("x")
    assert (src / "work/s/a/page.json").read_text() != "x"
    assert all((root / x).is_symlink() and (root / x).resolve() == (src / x).resolve() for x in ("raw", "ref", "cache"))
    assert not any((d / f).exists() for f in DOWNSTREAM)
    assert not (root / "warp").exists()


@pytest.mark.parametrize("inside", ["", "work/x", "warp/x", "out/x", "work/../work/x", "alias/warp/x"])
def test_a_root_in_todays_data_is_refused(src, tmp_path, inside):
    if inside.startswith("alias/"):                   # a linked parent folder that leads into today's data
        (tmp_path / "alias").symlink_to(src, target_is_directory=True)
        root = tmp_path / inside
    else:
        root = src / inside if inside else src
    before = sorted(p.relative_to(src) for p in src.rglob("*"))
    with pytest.raises(ValueError, match="today's data"):
        trial.prepare(root, [("s", "a")], source=src)
    assert sorted(p.relative_to(src) for p in src.rglob("*")) == before        # nothing made


@pytest.mark.parametrize("why", ["duplicate", "review", "seed", "missing"])
def test_a_sheet_that_does_not_test_the_seam_is_refused(src, tmp_path, why):
    if why == "duplicate":
        (config.INPUTS_DIR / "s").mkdir(parents=True)
        (config.INPUTS_DIR / "s" / "duplicate_scans.csv").write_text("sheet,kept,reason\na,b,same scan\n")
    elif why == "review":
        inputs.review_path("s", "a").parent.mkdir(parents=True)
        inputs.review_path("s", "a").write_text("{}")
    elif why == "seed":
        inputs.seed_path("s", "a").parent.mkdir(parents=True)
        inputs.seed_path("s", "a").write_text("{}")
    else:
        (src / "work/s/a/blocks_px.geojson").unlink()
    root = tmp_path / "t" / "traced"
    with pytest.raises(ValueError, match="s/a") as e:
        trial.prepare(root, [("s", "a")], source=src)
    assert {"duplicate": "duplicate scan", "review": "review", "seed": "seed",
            "missing": "blocks_px.geojson"}[why] in str(e.value)
    assert not (root / "work").exists() and not root.exists()


def test_a_root_is_prepared_once(src, tmp_path):
    root = tmp_path / "t" / "traced"
    trial.prepare(root, [("s", "a")], source=src)
    (root / "work/s/a/blocks_px.geojson").write_text("the finder's")
    with pytest.raises(FileExistsError):
        trial.prepare(root, [("s", "a")], source=src)
    assert (root / "work/s/a/blocks_px.geojson").read_text() == "the finder's"


@pytest.mark.parametrize("held", ["warp", "out"])
def test_a_root_holding_a_link_into_todays_data_is_refused(src, tmp_path, held):
    root = tmp_path / "t" / "traced"
    root.mkdir(parents=True)
    (root / held).symlink_to(src / held, target_is_directory=True)     # a hand-made link, no work/ yet
    before = sorted(p.relative_to(src) for p in src.rglob("*"))
    with pytest.raises(ValueError, match=f"{held}: a link or a path into today's data"):
        trial.prepare(root, [("s", "a")], source=src)
    assert sorted(p.name for p in root.iterdir()) == [held] and (root / held).is_symlink()
    assert sorted(p.relative_to(src) for p in src.rglob("*")) == before


def test_a_failed_copy_leaves_no_root_and_can_be_redone(src, tmp_path, monkeypatch):
    root = tmp_path / "t" / "traced"
    real, calls = shutil.copy2, []

    def failing(a, b, **k):
        calls.append(a)
        if len(calls) == 3:
            raise OSError("no space left on device")
        return real(a, b, **k)
    monkeypatch.setattr(shutil, "copy2", failing)
    with pytest.raises(OSError, match="no space"):
        trial.prepare(root, [("s", "a")], source=src)
    assert not root.exists() and list(root.parent.iterdir()) == []      # nothing half-made left beside it
    monkeypatch.setattr(shutil, "copy2", real)
    assert {p.name for p in trial.prepare(root, [("s", "a")], source=src)} == {*KEPT, *SIDECARS}


def test_a_root_whose_files_are_not_copies_is_removed(src, tmp_path, monkeypatch):
    root = tmp_path / "t" / "traced"
    monkeypatch.setattr(shutil, "copy2", lambda a, b, **k: os.link(a, b))  # the same file under a second name
    with pytest.raises(ValueError, match="not copies of their own"):
        trial.prepare(root, [("s", "a")], source=src)
    assert not root.exists() and (src / "work/s/a/page.json").is_file()


def test_numbers_are_read_from_the_root(src):
    got = trial.numbers(src, ("s", "a"))
    assert got == {"blocks": 3, "good_corners": 2, "control_points": 2, "passed": True, "reason": None,
                   "holdout_p50_ft": 5.01, "holdout_p95_ft": 12.28, "warp": "probabilistic", "within1": 0.627}
    (src / "warp/s/a/scores.json").write_text(json.dumps({"uncorrected": "nothing to fit",
                                                          "placement_check": {"passed": True, "reason": None}}))
    got = trial.numbers(src, ("s", "a"))
    assert got["warp"] == "uncorrected: nothing to fit" and got["within1"] is None


def test_reproduced_says_which_file_differs(src, tmp_path):
    other = tmp_path / "other"
    _fill(other)
    assert trial.reproduced(src, other, ("s", "a")) == {"transform": True, "refine": True, "score": True,
                                                        "warp_scores": True, "correction": True}
    t = other / "work/s/a/transform.json"
    t.write_text(json.dumps(_transform([(1.5, 2.5000001, True), (3, 4, False), (5, 6, True)])))
    s = other / "warp/s/a/scores.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "device": "cpu"}))
    assert trial.reproduced(src, other, ("s", "a")) == {"transform": False, "refine": True, "score": True,
                                                        "warp_scores": True, "correction": True}
    np.savez(other / "warp/s/a/correction.npz", Gc=np.arange(6, dtype=np.float32).reshape(2, 3) + 1e-6,
             spacing=np.int64(4), kind=np.str_("probabilistic"))
    assert trial.reproduced(src, other, ("s", "a"))["correction"] is False


def _footprint(root, series, sheet, geom):
    d = root / "warp" / series / sheet
    d.mkdir(parents=True, exist_ok=True)
    (d / "warped.footprint.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(geom)}]}))


def test_trained_ground_is_the_share_under_lot_labelled_sheets(tmp_path):
    source, run = tmp_path / "data", tmp_path / "run"
    _footprint(source, "s", "a", box(0, 0, 10, 10))
    _footprint(source, "s", "l", box(5, 0, 15, 10))
    _footprint(source, "s", "far", box(50, 50, 60, 60))
    run.mkdir()
    (run / "log.json").write_text(json.dumps({"items": [
        {"name": "s/l"}, {"name": "s/far"}, {"name": "train/s/a__0_0_64_64"}, {"name": "train/s/b__0_0_64_64"},
        {"name": "train/s/ab__0_0_64_64"}]}))
    got = trial.trained_ground(("s", "a"), run, source)
    assert got["lot_labelled_share"] == pytest.approx(0.5)
    assert got["hand_crops"] == ["train/s/a__0_0_64_64"]


def test_the_trial_commands_prepare_a_root_and_write_the_numbers(src, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", src)
    monkeypatch.setattr(trial, "TRIAL", tmp_path / "find" / "trial")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    r = invoke(["find", "trial", "root", "--trial", "T", "--arm", "traced", "--sheet", "s/a"])
    assert r.exit_code == 0, r.output
    root = trial.arm_root("T", "traced")
    assert root == tmp_path / "find" / "trial" / "T" / "traced"
    assert str(root / "work/s/a/blocks_px.geojson") in r.output.splitlines()
    again = invoke(["find", "trial", "root", "--trial", "T", "--arm", "traced", "--sheet", "s/a"])
    assert again.exit_code == 1 and "prepared once" in again.output
    bad = invoke(["find", "trial", "root", "--trial", "T", "--arm", "other", "--sheet", "s/a"])
    assert bad.exit_code != 0
    _fill(tmp_path / "copy")                          # the traced arm's placement, as `sfzmaps run` would leave it
    for x in ("work", "warp"):
        for f in (tmp_path / "copy" / x / "s" / "a").iterdir():
            (root / x / "s" / "a").mkdir(parents=True, exist_ok=True)
            (root / x / "s" / "a" / f.name).write_bytes(f.read_bytes())
    _footprint(src, "s", "a", box(0, 0, 10, 10))
    run = tmp_path / "find" / "runs" / "all-pieces"
    run.mkdir(parents=True)
    (run / "log.json").write_text(json.dumps({"items": [{"name": "train/s/a__0_0_64_64"}]}))
    r = invoke(["find", "trial", "numbers", "--trial", "T"])
    assert r.exit_code == 0, r.output
    rows = [ln for ln in r.output.splitlines() if ln.startswith("s/a ") and " blocks, " in ln]
    assert [ln.split()[1] for ln in rows] == ["today:", "traced:"]
    assert "traced as today: transform same" in r.output and "hand crops on it: train/s/a__0_0_64_64" in r.output
    assert "3 blocks" in rows[0] and "2 control points" in rows[0] and "probabilistic" in rows[0]
    saved = json.loads((tmp_path / "find" / "trial" / "T" / "numbers.json").read_text())
    got = saved["sheets"]["s/a"]
    assert got["today"] == got["traced"] == trial.numbers(src, ("s", "a"))
    assert all(got["reproduced"].values())
    assert got["trained_ground"] == {"lot_labelled_share": 0.0, "by_sheet": {}, "hand_crops": ["train/s/a__0_0_64_64"]}


# The comparison (`find trial compare`): control-point reach, the three places, the warped-scan pictures.

def _tif(path, value=255, left=6_000_000.0, top=2_100_000.0, res=2.7, w=300, h=200):
    """An RGBA GeoTIFF (EPSG:2227, `res` ft/px), every px `value` and opaque, as `sfzmaps warp` writes warped.tif."""
    import rasterio
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((4, h, w), value, np.uint8)
    a[3] = 255
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=4, dtype="uint8", crs="EPSG:2227",
                       transform=from_origin(left, top, res, res)) as d:
        d.write(a)


def test_reach_is_the_distance_to_the_nearest_control_point():
    points = np.stack([np.arange(101.0), np.zeros(101)], axis=1)
    got = trial.reach_ft(points, np.array([[0.0, 0.0]]))
    assert got == {"p50": pytest.approx(50.0), "p90": pytest.approx(90.0)}
    assert trial.reach_ft(points, np.zeros((0, 2))) == {"p50": None, "p90": None}


def test_control_points_on_trained_ground_are_those_under_the_lot_labelled_sheets(tmp_path):
    source, run = tmp_path / "data", tmp_path / "run"
    _footprint(source, "s", "l", box(0, 0, 10, 10))
    run.mkdir()
    (run / "log.json").write_text(json.dumps({"items": [{"name": "s/l"}, {"name": "train/s/a__0_0_64_64"}]}))
    ground = trial.trained_union(run, source)
    assert trial.share_on(np.array([[1.0, 1.0], [5.0, 5.0], [20.0, 5.0], [-1.0, 3.0]]), ground) == 0.5
    assert trial.share_on(np.zeros((0, 2)), ground) is None and trial.share_on(np.ones((3, 2)), None) == 0.0


def _record(tmp_path, monkeypatch, text="the finder's blocks"):
    import hashlib
    monkeypatch.setattr(trial, "TRIAL", tmp_path / "trial")
    f = trial.arm_root("T", "finder") / "work/s/a/blocks_px.geojson"
    f.parent.mkdir(parents=True)
    f.write_text(text)
    rec = tmp_path / "trial/T/blocks/s/a.json"
    rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps({"sheet": "s/a", "sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return f


def test_compare_refuses_a_retraced_blocks_file(tmp_path, monkeypatch):
    f = _record(tmp_path, monkeypatch)
    trial.check_blocks("T", ("s", "a"))                     # as `find blocks` wrote it: no complaint
    f.write_text("traced again")
    with pytest.raises(ValueError, match="s/a") as e:
        trial.check_blocks("T", ("s", "a"))
    assert "re-traced or changed" in str(e.value)
    (tmp_path / "trial/T/blocks/s/a.json").unlink()
    with pytest.raises(ValueError, match="s/a.*find blocks"):
        trial.check_blocks("T", ("s", "a"))


def test_places_takes_the_box_with_most_points_off_the_print(tmp_path, monkeypatch):
    from types import SimpleNamespace
    src, res = tmp_path / "data", 2.7
    side = trial.WINDOW_PX * res
    _tif(src / "warp/s/a/warped.tif", res=res)            # today's warped scan: only its resolution is read
    _footprint(src, "s", "a", box(0, 0, 3.5 * side, side))  # one row of four grid boxes; the centroid in the second

    def at(col, n):                                        # n points inside grid box `col`
        return np.stack([np.linspace(col * side + 1, (col + 1) * side - 1, n), np.full(n, side / 2)], axis=1)
    pts = np.vstack([at(0, 10), at(3, 30), at(2, 50)])
    off = np.r_[np.ones(10, bool), np.ones(30, bool), np.zeros(50, bool)]   # A: 10 off the print, B: 30, C: 50 on it
    monkeypatch.setattr(trial, "_block_sides", lambda sid, source: SimpleNamespace(pts=pts))
    monkeypatch.setattr(trial, "off_print", lambda sid, source: off)
    got = trial.places(("s", "a"), src)

    def cell(col):
        return pytest.approx((col * side, 0.0, (col + 1) * side, side))
    assert got["worst"] == cell(3)                          # B: the most points off the print
    assert got["busiest"] == cell(2)                        # C: the most points
    assert got["centre"] == cell(1)                         # the footprint's centroid
    monkeypatch.setattr(trial, "off_print", lambda sid, source: np.zeros(len(pts), bool))
    got = trial.places(("s", "a"), src)                    # nothing off the print: no worst place, not box 0
    assert "worst" not in got and got["busiest"] == cell(2)


def test_places_sharing_a_box_are_grouped():
    boxes = {"worst": (0, 0, 1, 1), "busiest": (1, 0, 2, 1), "centre": (0, 0, 1, 1)}
    assert trial.shared(boxes) == {"worst": ["worst", "centre"], "busiest": ["busiest"]}
    assert trial.shared({"busiest": (0, 0, 1, 1), "centre": (0, 0, 1, 1)}) == {"busiest": ["busiest", "centre"]}


def test_placed_picture_is_at_the_scans_own_resolution(tmp_path):
    from shapely.geometry import LineString
    from sfzmaps.find import look
    left, top, res = 6_000_000.0, 2_100_000.0, 2.7
    _tif(tmp_path / "before.tif", 255)                     # white paper
    _tif(tmp_path / "after.tif", 0)                        # black ink
    x0, y1 = left + 50 * res, top - 40 * res               # a window 100 x 80 px from px (50, 40)
    box_ft = (x0, y1 - 80 * res, x0 + 100 * res, y1)
    y = top - (40 + 30.5) * res                            # the centre of the window's px row 30
    pic = trial.placed_picture(tmp_path / "before.tif", tmp_path / "after.tif", box_ft,
                               [LineString([(left, y), (left + 300 * res, y)])], "RESULT: s / a")
    assert pic.shape[1] == 2 * 100 + look.GAP
    band = pic.shape[0] - 80
    for x in (40, 100 + look.GAP + 40):                    # each panel
        for r in (29, 30, 31):                             # the line, drawn LINE_PX (3) wide
            assert tuple(pic[band + r, x]) == trial.LINE
        assert tuple(pic[band + 26, x]) != trial.LINE
    assert tuple(pic[band + 10, 40]) == (255, 255, 255) and tuple(pic[band + 10, 100 + look.GAP + 40]) == (0, 0, 0)
    with pytest.raises(ValueError, match="RESULT:"):
        trial.placed_picture(tmp_path / "before.tif", tmp_path / "after.tif", box_ft, [], "s / a")


def test_a_line_lands_on_the_px_the_rasters_georeferencing_says(tmp_path, monkeypatch):
    from shapely.geometry import LineString
    from sfzmaps.find import look
    base, top, res = 6_000_000.0, 2_100_000.0, 2.7
    left = base + 0.7 * res                                # the raster's origin 0.7 px off the box's grid
    _tif(tmp_path / "s.tif", 255, left=left)
    box_ft = (base + 50 * res, top - 60 * res, base + 70 * res, top - 40 * res)
    # the window starts at raster column round(49.3) = 49; a line a quarter of the way into raster column 60
    x = left + 60.25 * res
    monkeypatch.setattr(trial, "LINE_PX", 1)
    pic = trial.placed_picture(tmp_path / "s.tif", tmp_path / "s.tif", box_ft,
                               [LineString([(x, top), (x, top - 200 * res)])], "RESULT: s / a")
    band = pic.shape[0] - 20
    for x0 in (0, 20 + look.GAP):                          # each panel: raster column 60 is window column 11
        cols = [c for c in range(20) if tuple(pic[band + 10, x0 + c]) == trial.LINE]
        assert cols == [11]


def test_a_picture_over_the_viewers_limit_is_refused(tmp_path):
    res = 2.7
    _tif(tmp_path / "s.tif")
    box_ft = (6_000_000.0, 2_100_000.0 - 10 * res, 6_000_000.0 + 4000 * res, 2_100_000.0)   # 4,000 px wide each
    with pytest.raises(ValueError, match="8016 x 10 px"):
        trial.placed_picture(tmp_path / "s.tif", tmp_path / "s.tif", box_ft, [], "RESULT: s / a")


def test_a_large_picture_gets_a_reduced_overview(tmp_path, monkeypatch):
    import cv2
    from sfzmaps.find import look
    monkeypatch.setattr(look, "SMALL_SIDE", 100)
    assert trial.write_picture(np.zeros((50, 80, 3), np.uint8), tmp_path / "a__worst.png") is None
    assert not (tmp_path / "a__worst.small.png").exists()
    small = trial.write_picture(np.zeros((50, 300, 3), np.uint8), tmp_path / "b__worst.png")
    assert small == tmp_path / "b__worst.small.png"
    assert cv2.imread(str(small)).shape[:2] == (17, 100) and cv2.imread(str(tmp_path / "b__worst.png")).shape[1] == 300


def _placed_trial(src, tmp_path, monkeypatch, finder_placed=True):
    """A trial T with today's data `src` and both arms holding sheet s/a, placed (unless `finder_placed` is False for
    the finder arm), with the finder's blocks recorded, and the slow readers of real data replaced."""
    import hashlib
    from shapely.geometry import LineString
    monkeypatch.setattr(config, "DATA_DIR", src)
    monkeypatch.setattr(trial, "TRIAL", tmp_path / "find" / "trial")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    _tif(src / "warp/s/a/warped.tif")
    res = 2.7
    b = (6_000_000.0 + 10 * res, 2_100_000.0 - 110 * res, 6_000_000.0 + 110 * res, 2_100_000.0 - 10 * res)
    for arm in trial.ARMS:
        root = trial.arm_root("T", arm)
        _fill(root)
        # each arm its own control points (map ft = 2 x px): traced on the box's corner, finder 300 ft east of it
        east = 150 if arm == "finder" else 0
        (root / "work/s/a/transform.json").write_text(json.dumps(_transform(
            [(b[0] / 2 + east, b[1] / 2, True), (b[0] / 2 + east, b[3] / 2 + 100, True), (0, 0, False)])))
        if arm == "finder" and not finder_placed:
            (root / "work/s/a/transform.json").unlink()
        else:
            _tif(root / "warp/s/a/warped.tif", 0 if arm == "finder" else 255)
    f = trial.arm_root("T", "finder") / "work/s/a/blocks_px.geojson"
    rec = trial.TRIAL / "T/blocks/s/a.json"
    rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps({"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}))
    run = tmp_path / "find/runs/all-pieces"
    run.mkdir(parents=True)
    (run / "log.json").write_text(json.dumps({"items": [{"name": "train/s/a__0_0_64_64"}]}))
    other = (b[2], b[1], b[2] + 50 * res, b[3])
    monkeypatch.setattr(trial, "side_points", lambda sid: np.array([[b[0] + 1, b[1] + 1], [b[0] + 1, b[3] + 1]]))
    monkeypatch.setattr(trial, "off_print", lambda sid, source: np.array([True, False]))
    monkeypatch.setattr(trial, "places", lambda sid, source: {"worst": b, "busiest": other, "centre": b})
    monkeypatch.setattr(trial, "block_lines", lambda boxes: [LineString([(b[0], b[1]), (b[2], b[3])])])


def _files(*roots):
    import hashlib
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for r in roots for p in sorted(r.rglob("*"))
            if p.is_file()}


def test_compare_writes_only_its_own_files(src, tmp_path, monkeypatch):
    _placed_trial(src, tmp_path, monkeypatch)
    roots = [src, *(trial.arm_root("T", a) for a in trial.ARMS)]
    before = _files(*roots)
    t = trial.TRIAL / "T"
    (t / "look/s").mkdir(parents=True)
    (t / "look/s/a__centre.png").write_text("an earlier run's picture")
    r = invoke(["find", "trial", "compare", "--trial", "T"])
    assert r.exit_code == 0, r.output
    assert _files(*roots) == before                         # today's data and both arms untouched
    written = sorted(str(p.relative_to(t)) for p in t.rglob("*")
                     if p.is_file() and p.relative_to(t).parts[0] not in (*trial.ARMS, "blocks"))
    # worst and centre share a box: one picture, named after worst; the earlier centre picture is gone
    assert written == ["compare.json", "look/s/a__busiest.png", "look/s/a__worst.png"]
    assert "s/a worst (the most block-side points off the print today) and centre (the sheet's middle) share one " \
           "box" in r.output
    got = json.loads((t / "compare.json").read_text())["sheets"]["s/a"]
    assert got["places"]["centre"]["picture"] == got["places"]["worst"]["picture"]
    assert got["numbers"]["finder"] == trial.numbers(trial.arm_root("T", "finder"), ("s", "a"))
    pts = trial.side_points(("s", "a"))                     # one in the worst place's box, one above it
    for col, root in (("today", src), *((a, trial.arm_root("T", a)) for a in trial.ARMS)):
        gcps = np.array([(g["wx"], g["wy"]) for g in json.loads((root / "work/s/a/transform.json").read_text())["gcps"]
                         if g["enabled"]])
        d = np.linalg.norm(pts[:, None] - gcps[None], axis=2).min(axis=1)
        assert got["reach_ft"][col] == {"p50": pytest.approx(np.percentile(d, 50)),
                                        "p90": pytest.approx(np.percentile(d, 90))}
    assert len({got["reach_ft"][c]["p50"] for c in ("today", "traced", "finder")}) == 3
    assert got["places"]["worst"]["side_points"] == 1
    assert got["places"]["worst"]["off_print_today"] == 1
    assert "s/a traced:" in r.output and "s/a finder:" in r.output


def test_compare_refuses_a_data_root_inside_the_trial(src, tmp_path, monkeypatch):
    _placed_trial(src, tmp_path, monkeypatch)
    monkeypatch.setattr(config, "DATA_DIR", trial.arm_root("T", "finder"))
    r = invoke(["find", "trial", "compare", "--trial", "T"])
    assert r.exit_code == 1 and "unset SFZMAPS_DATA_DIR" in r.output
    assert not (trial.TRIAL / "T/compare.json").exists()


def test_compare_refuses_a_sheet_with_no_placement(src, tmp_path, monkeypatch):
    _placed_trial(src, tmp_path, monkeypatch, finder_placed=False)
    r = invoke(["find", "trial", "compare", "--trial", "T"])
    assert r.exit_code == 1
    assert "s/a finder" in r.output and "sfzmaps run --year s --sheet a" in r.output
    assert "sfzmaps warp s" in r.output
    assert not (trial.TRIAL / "T/compare.json").exists() and not (trial.TRIAL / "T/look").exists()
