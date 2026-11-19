"""The 1921 patch model and per-lot evidence (`sfzmaps.solver.hatchfit`): a hatched patch pulls its lot towards 1R, a
lot no patch covers has no evidence, the held-out quarter's patches stay out of its fit, the columns follow the names
asked for, and each patch's training weight is the read's confidence times the lot's share of the patch.

The patches are measured with the real measure (`hatching.measure_patch`) on drawn copies of the six legend patterns."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver import grid, hatchfit, hatching, paths

PERIOD = 12.0
SIDE = 24
WAVE_1R = np.radians(-45.0) + np.pi / 2           # 1R's lines at -45 degrees on the page: its wave across them


def lines(period, angle_deg, offset, width=2, size=SIDE) -> np.ndarray:
    t = np.radians(angle_deg)
    y, x = np.mgrid[0:size, 0:size].astype(float)
    return (np.mod(-x * np.sin(t) + y * np.cos(t) + offset, period) < width).astype(np.uint8)


def pattern(code: str, rng) -> np.ndarray:
    """A patch of one legend pattern, at a random phase, with a few specks of dirt."""
    off = rng.uniform(0, PERIOD)
    if code == "1R":
        ink = lines(PERIOD, -45, off)
    elif code == "2R":
        ink = np.zeros((SIDE, SIDE), np.uint8)
    elif code == "C":
        ink = np.ones((SIDE, SIDE), np.uint8)
    elif code == "LI":
        ink = lines(PERIOD / 2, -45, off) | lines(PERIOD / 2, 45, off)
    elif code == "HI":
        ink = lines(PERIOD / 2, 0, off) | lines(PERIOD / 2, 90, off)
    else:                                          # U: a checkerboard
        y, x = np.mgrid[0:SIDE, 0:SIDE]
        ink = (((x + int(off)) // 3 + (y + int(off)) // 3) % 2).astype(np.uint8)
    specks = rng.integers(0, SIDE, (2, 3))
    ink = ink.copy()
    ink[specks[0], specks[1]] ^= 1
    return ink


def measure(ink: np.ndarray) -> dict:
    frame = hatching.Frame(period=PERIOD, low=2.0, taper=1.0, wave_1r=WAVE_1R, turn=0, phi=0.3)
    return hatching.measure_patch(ink, np.ones(ink.shape, bool), (0, 0, SIDE, SIDE), frame)


# the centre of each quarter, EPSG:2227 ft: nw 0, ne 1, sw 2, se 3 (`grid.quarter_of`)
CENTRE = {0: (-2000.0, 2000.0), 1: (2000.0, 2000.0), 2: (-2000.0, -2000.0), 3: (2000.0, -2000.0)}


def patch_row(code, rng, sheet, k, quarter) -> dict:
    dx, dy = CENTRE[quarter]
    return {"sheet": sheet, "patch": k, "side": SIDE, "period": PERIOD, "cx": grid.QUARTER_SPLIT[0] + dx + k,
            "cy": grid.QUARTER_SPLIT[1] + dy, **measure(pattern(code, rng))}


@pytest.fixture(scope="module")
def data() -> hatchfit.Data:
    """Six lots of each legend district in every quarter, each covered by three patches of its pattern; one read in
    ten names a wrong district at a lower confidence (the reader's noise).  A few patches straddle two lots in two
    quarters.  One legend swatch per district."""
    rng = np.random.default_rng(0)
    rows, links, quarter, reads = [], [], [], []
    lot = 0
    for q in range(4):
        for code in hatchfit.NAMES:
            for _ in range(6):
                read, conf = code, 0.95
                if rng.uniform() < 0.1:
                    read, conf = hatchfit.NAMES[rng.integers(len(hatchfit.NAMES))], 0.6
                quarter.append(q)
                reads.append((read, conf))
                for _ in range(3):
                    k = len(rows)
                    rows.append(patch_row(code, rng, "s", k, q))
                    links.append((k, lot, 1.0))
                lot += 1
    # straddling patches: half in the last lot of quarter q, half in the first lot of the next quarter
    per_q = lot // 4
    for q in range(4):
        a, b = q * per_q + per_q - 1, ((q + 1) % 4) * per_q
        k = len(rows)
        rows.append(patch_row(hatchfit.NAMES[-1], rng, "s", k, q))
        links += [(k, a, 0.5), (k, b, 0.5)]
    patches = pd.DataFrame(rows)
    patch_lots = pd.DataFrame(links, columns=["patch", "lot", "share"]).assign(sheet="s")
    labels = np.zeros((lot, len(hatchfit.NAMES)))
    for i, (code, conf) in enumerate(reads):
        labels[i, hatchfit.NAMES.index(code)] = conf
    swatches = pd.DataFrame([{"code": c, "side": SIDE, "period": PERIOD, **measure(pattern(c, rng))}
                             for c in hatchfit.NAMES])
    return hatchfit.Data(patches, patch_lots, np.array(quarter, np.int8), labels, swatches)


@pytest.fixture(scope="module")
def model(data) -> hatchfit.Model:
    return hatchfit.fit_quarter("nw", data)


def probe(rng, codes_by_lot: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fresh patches: lot i covered by one whole patch of each code in codes_by_lot[i]."""
    rows, links = [], []
    for lot, codes in enumerate(codes_by_lot):
        for code in codes:
            k = len(rows)
            rows.append(patch_row(code, rng, "t", k, 0))
            links.append(("t", k, lot, 1.0))
    return pd.DataFrame(rows), pd.DataFrame(links, columns=["sheet", "patch", "lot", "share"])


def test_hatched_patch_beats_blank(model):
    rng = np.random.default_rng(1)
    patches, patch_lots = probe(rng, [["1R", "2R"], ["2R", "2R"]])
    U = hatchfit.lot_evidence(model, patches, patch_lots, 2, hatchfit.NAMES)
    one_r = hatchfit.NAMES.index("1R")
    assert U[0, one_r] < U[1, one_r]
    assert U[1, hatchfit.NAMES.index("2R")] == 0.0          # two blank patches: 2R is the cheapest


def test_no_patch_zero_row(model):
    rng = np.random.default_rng(2)
    patches, patch_lots = probe(rng, [["1R"], ["C"]])
    U = hatchfit.lot_evidence(model, patches, patch_lots, 3, hatchfit.NAMES)
    assert U.shape == (3, len(hatchfit.NAMES))
    assert (U[2] == 0).all()                                 # no patch: no evidence, no default district
    assert (U[:2] > 0).sum(1).min() > 0                      # the covered lots have evidence
    assert U[0].argmin() == hatchfit.NAMES.index("1R") and U[1].argmin() == hatchfit.NAMES.index("C")


def test_quarter_held_out(data):
    for q in ("ne", "se"):
        m = hatchfit.fit_quarter(q, data)
        qi = paths.QUARTERS.index(q)
        rows = set(m.trained.tolist())
        assert rows
        lots_of = data.patch_lots.groupby("patch")["lot"].apply(list)
        for r in rows:
            p = data.patches.iloc[r]
            assert int(grid.quarter_of(np.array([[p.cx, p.cy]]))[0]) != qi
            assert all(data.quarter[lot] != qi for lot in lots_of[p.patch])
        # every patch outside q with a read lot is trained on (the straddling ones only where neither half is in q)
        touch_q = {k for k, ls in lots_of.items() if any(data.quarter[lot] == qi for lot in ls)}
        assert rows == set(range(len(data.patches))) - touch_q


def test_names_order(model):
    rng = np.random.default_rng(3)
    patches, patch_lots = probe(rng, [["1R", "LI"], ["HI"], ["U", "2R"]])
    U = hatchfit.lot_evidence(model, patches, patch_lots, 3, hatchfit.NAMES)
    turned = list(reversed(hatchfit.NAMES))
    V = hatchfit.lot_evidence(model, patches, patch_lots, 3, turned)
    for j, name in enumerate(turned):
        np.testing.assert_array_equal(V[:, j], U[:, hatchfit.NAMES.index(name)])
    with pytest.raises(ValueError):
        hatchfit.lot_evidence(model, patches, patch_lots, 3, ["1R", "RH-1"])


def test_same_inputs_same_bytes(data, model, tmp_path):
    rng = np.random.default_rng(4)
    patches, patch_lots = probe(rng, [["1R", "2R"], ["LI"]])
    U = hatchfit.lot_evidence(model, patches, patch_lots, 2, hatchfit.NAMES)
    again = hatchfit.lot_evidence(hatchfit.fit_quarter("nw", data), patches, patch_lots, 2, hatchfit.NAMES)
    assert U.tobytes() == again.tobytes()
    for name in ("a", "b"):
        hatchfit.save_npz(tmp_path / f"{name}.npz", U=U.astype(np.float32), names=np.array(hatchfit.NAMES),
                          tau=np.float64(model.tau))
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    with zipfile.ZipFile(tmp_path / "a.npz") as z:              # no clock in the file (numpy dates members now)
        assert {i.date_time for i in z.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_patch_weights_confidence_times_share():
    # one patch: a quarter in a 1R lot read at 0.8, three quarters in a 2R lot read at 1.0; a lot with no read
    patch_lots = pd.DataFrame({"sheet": "s", "patch": [0, 0, 0], "lot": [0, 1, 2], "share": [0.25, 0.5, 0.25]})
    labels = np.zeros((3, len(hatchfit.NAMES)))
    labels[0, hatchfit.NAMES.index("1R")] = 0.8
    labels[1, hatchfit.NAMES.index("2R")] = 1.0
    Y = hatchfit.targets(pd.DataFrame({"sheet": ["s"], "patch": [0]}), patch_lots, labels)
    want = np.zeros(len(hatchfit.NAMES))
    want[hatchfit.NAMES.index("1R")] = 0.25 * 0.8
    want[hatchfit.NAMES.index("2R")] = 0.5 * 1.0
    np.testing.assert_allclose(Y[0], want)


def test_reads_skip_off_sheet_and_other_codes(tmp_path):
    work, warp = tmp_path / "work", tmp_path / "warp"
    for s in ("a", "b"):
        (work / s).mkdir(parents=True)
    (warp / "a").mkdir(parents=True)
    (warp / "a" / "transform.json").write_text("{}")
    parcels = [{"blklot": "0001001", "code": "1R", "confidence": 0.9, "via": "parcel"},
               {"blklot": "0001002", "code": "2R", "confidence": 0.8, "via": "off_sheet"},
               {"blklot": "0001003", "code": "", "confidence": 0.8, "via": "parcel"},
               {"blklot": "0001004", "code": "LI", "confidence": 0.7, "via": "block"}]
    for s in ("a", "b"):                           # b has no placement: the spike's sheets() leaves it out
        (work / s / "district_pattern.json").write_text(json.dumps({"parcels": parcels}))
    r = hatchfit.reads(work, warp)
    assert r[["sheet", "blklot", "code"]].values.tolist() == [["a", "0001001", "1R"], ["a", "0001004", "LI"]]
