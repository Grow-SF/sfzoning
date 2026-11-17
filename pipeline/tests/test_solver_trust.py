import dataclasses
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sfzmaps.solver import lots, paths, prints, runs, trust

NW, NE, SW, SE = range(4)


def _print(kind=0, sol="", sol_confidence=math.nan, terra="", height=None, third=None, third_confidence=math.nan,
           leader=False, recheck=None, turned="", lot=0, street=False, x=0.0, y=0.0):
    """One prints.parquet row: `third` / `recheck` None for no saved answer found, "" for an answer naming no code."""
    box = (0.0, 0.0, 40.0, height) if height is not None else (math.nan,) * 4
    row = {c: None for c in prints.COLUMNS}
    row.update(sheet="2000-digital-zoning-p01", page_x=0.0, page_y=0.0, x=float(x), y=float(y), kind=kind,
               box_x0=box[0], box_y0=box[1], box_x1=box[2], box_y1=box[3], terra_code=terra, sol_code=sol,
               sol_confidence=sol_confidence, turned_code=turned, turned_edge=50.0 if turned else math.nan,
               third_code=third or "", third_confidence=third_confidence, third_leader=leader,
               third_found=third is not None, recheck_asked=recheck is not None, recheck_code=recheck or "",
               recheck_confidence=math.nan, recheck_found=recheck is not None, lab_lot=lot, street=street)
    return row


def _prints(rows):
    return pd.DataFrame(rows, columns=prints.COLUMNS)


def _lots(codes, quarter):
    """A Lots holding only what the trust reads: each lot's code ("" unknown) and quarter."""
    codes = np.array(codes, dtype=str)
    empty = np.zeros(0, np.int64)
    return lots.Lots(lot=np.zeros((1, 1), np.int32), grown=np.zeros((1, 1), np.int32), a=empty, b=empty,
                     touching=empty, street=empty, scored=codes != "", known=codes != "",
                     quarter=np.array(quarter, np.int8), codes=codes)


def _synthetic():
    """Five prints on four lots, between them giving every feature a value somewhere."""
    rows = [
        # read by Sol and Terra, and by the third read, the re-check and the turned pass
        _print(kind=0, sol="RH-1", sol_confidence=0.9, terra="RH-1", height=30.0, third="RH-1",
               third_confidence=0.8, leader=True, recheck="RH-2", turned="RH-1", lot=0, x=0, y=0),
        # Terra alone, in a street; the third read found no code there
        _print(kind=1, terra="RM-2", height=60.0, third="", lot=1, street=True, x=500, y=0),
        # Sol alone, no box
        _print(kind=2, sol="NC-3", sol_confidence=0.5, lot=2, x=0, y=500),
        # found only by the turned pass: a single letter in a street, 300 ft from the nearest print
        _print(kind=3, turned="P", lot=3, street=True, x=800, y=0),
        # found only by the turned pass, 0.5 ft from a print (distance floored at 1 ft)
        _print(kind=3, turned="RH-2", lot=0, x=0.3, y=0.4),
    ]
    return _prints(rows), _lots(["RH-1", "RM-2", "NC-3", "P"], [NW, NE, SW, SE])


def test_feature_columns_match_names():
    table = trust.features(*_synthetic())
    assert len(trust.FEATURES) == 23 and table.values.shape == (6, 23)
    assert (table.values != 0).any(axis=0).all(), "every feature has a value somewhere on this set"
    # one row per (print, candidate), print by print, candidates by name
    assert table.print_index.tolist() == [0, 0, 1, 2, 3, 4]
    assert [table.names[k] for k in table.code_index] == ["RH-1", "RH-2", "RM-2", "NC-3", "P", "RH-2"]
    # the code vocabulary: the table's known codes and Sol's and Terra's reads, sorted; then the third read's and the
    # re-check's other codes, sorted; then the turned pass's
    assert table.names == ["NC-3", "P", "RH-1", "RM-2", "RH-2"]
    col = {name: table.values[:, i] for i, name in enumerate(trust.FEATURES)}
    hmed = 45.0                                               # the median of the box heights 30 and 60
    expect = {
        "base": [1, 1, 1, 1, 1, 1],
        "Sol names it": [1, 0, 0, 1, 0, 0],
        "Terra names it": [1, 0, 1, 0, 0, 0],
        "the third read names it": [1, 0, 0, 0, 0, 0],
        "every read names it": [0, 0, 1, 1, 1, 1],
        "two or more reads name it": [1, 0, 0, 0, 0, 0],
        "Sol's confidence when Sol names it (log)": [-math.log(0.1), 0, 0, -math.log(0.5), 0, 0],
        "the third read's confidence when it names it (log)": [-math.log(0.2), 0, 0, 0, 0, 0],
        "print height (log, against the median)": [math.log(30 / hmed)] * 2 + [math.log(60 / hmed), 0, 0, 0],
        "in a street": [0, 0, 1, 0, 1, 0],
        "no Sol read": [0, 0, 1, 0, 1, 1],
        "no Terra read": [0, 0, 0, 1, 1, 1],
        "the third read found no code": [0, 0, 1, 0, 0, 0],
        "the third read saw a leader line": [1, 1, 0, 0, 0, 0],
        "a code Sol reads on some print": [1, 0, 0, 1, 0, 0],
        "the re-check names it": [0, 1, 0, 0, 0, 0],
        "re-checked (printed sideways)": [1, 1, 0, 0, 0, 0],
        "the turned pass names it": [1, 0, 0, 0, 1, 1],
        "no turned-pass read": [0, 0, 1, 1, 0, 0],
        "found only by the turned pass": [0, 0, 0, 0, 1, 1],
        "turned pass only: distance to the nearest other print (log)":
            [0, 0, 0, 0, math.log(300 / 100), -math.log(100)],
        "turned pass only: in a street": [0, 0, 0, 0, 1, 0],
        "turned pass only: a single letter": [0, 0, 0, 0, 1, 0],
    }
    assert list(expect) == list(trust.FEATURES)
    for name, want in expect.items():
        np.testing.assert_allclose(col[name], want, atol=1e-12, err_msg=name)
    # the target: the candidate is the table's code of the lot under the print
    assert table.target.tolist() == [True, False, True, True, True, False]
    assert table.quarter.tolist() == [NW, NW, NE, SW, SE, NW]


def _logistic_set(rng, n=400, k=5):
    """A separable-ish set: a base column and k features, labels drawn from a strong logistic signal."""
    X = np.c_[np.ones(n), rng.normal(size=(n, k))]
    beta = np.r_[0.3, rng.normal(scale=2.5, size=k)]
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(float)
    return X, y


def test_newton_fit_matches_scipy_minimum_of_log_loss_with_per_weight_ridge():
    # the ridges the trust fit uses (1e-3 and 1000) and one between, on different weights
    X, y = _logistic_set(np.random.default_rng(1))
    ridge = np.array([1e-3, 1000.0, 1e-3, 1000.0, 0.5, 1e-3])

    def loss(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        return np.logaddexp(0, -np.where(y > 0, z, -z)).sum() + 0.5 * (ridge * w * w).sum(), \
            X.T @ (p - y) + ridge * w

    ref = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000}).x
    ours = trust.newton_fit(X, y, ridge)
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def _random_table(rng, n=600):
    k = len(trust.FEATURES)
    values = np.c_[np.ones(n), rng.normal(size=(n, k - 1))]
    target = rng.random(n) < 1 / (1 + np.exp(-values @ rng.normal(size=k)))
    return trust.FeatureTable(print_index=np.arange(n), code_index=np.zeros(n, np.int64), values=values,
                              names=["RH-1"], lot=np.arange(n), known=rng.random(n) < 0.9,
                              quarter=rng.integers(0, 4, n).astype(np.int8), target=target)


def test_fit_ignores_rows_in_its_own_quarter():
    rng = np.random.default_rng(2)
    table = _random_table(rng)
    inside = table.quarter == NE
    changed = dataclasses.replace(table, values=np.where(inside[:, None], rng.normal(size=table.values.shape),
                                                         table.values), target=np.where(inside, ~table.target,
                                                                                        table.target))
    np.testing.assert_array_equal(trust.fit(changed, "ne"), trust.fit(table, "ne"))
    assert not np.array_equal(trust.fit(changed, "nw"), trust.fit(table, "nw")), "the other fits see those rows"
    # rows on lots with no known code train no fit
    unknown = dataclasses.replace(table, target=np.where(table.known, table.target, ~table.target))
    np.testing.assert_array_equal(trust.fit(unknown, "sw"), trust.fit(table, "sw"))
    # through features(): the table's codes of the lots in a quarter change nothing in that quarter's fit
    found, city = _synthetic()
    found = _prints([found.iloc[i].to_dict() for i in [0, 1, 2, 3, 4] * 6])
    table = trust.features(found, city)
    recoded = dataclasses.replace(city, codes=np.where(city.quarter == NW, "RM-2", city.codes))
    np.testing.assert_array_equal(trust.fit(trust.features(found, recoded), "nw"), trust.fit(table, "nw"))
    assert not np.array_equal(trust.fit(trust.features(found, recoded), "ne"), trust.fit(table, "ne"))


def test_build_readings_writes_every_quarter(tmp_path, monkeypatch):
    found, city = _synthetic()
    found = _prints([found.iloc[i].to_dict() for i in [0, 1, 2, 3, 4] * 6])
    for name in ("PRINTS_DIR", "LOTS_DIR", "LABELS_DIR"):
        monkeypatch.setattr(paths, name, tmp_path / name)
    monkeypatch.setattr(runs, "SOLVE_PRINTS_FILE", paths.PRINTS_DIR / "prints.parquet")
    paths.PRINTS_DIR.mkdir()
    paths.LOTS_DIR.mkdir()
    found.to_parquet(paths.PRINTS_DIR / "prints.parquet", index=False)
    np.savez_compressed(paths.LOTS_DIR / "lots.npz", lot=city.lot, grown=city.grown, pairs=np.zeros((0, 4), np.int64),
                        scored=city.scored, known=city.known, quarter=city.quarter, table=city.codes)
    trust.build_readings()
    first = {q: dict(np.load(paths.LABELS_DIR / f"readings_{q}.npz")) for q in paths.QUARTERS}
    trust.build_readings()
    for q in paths.QUARTERS:
        out = first[q]
        assert sorted(out) == ["code_index", "names", "print_index", "trust"]
        again = np.load(paths.LABELS_DIR / f"readings_{q}.npz")
        for key in out:
            np.testing.assert_array_equal(again[key], out[key])
        # each row is one of its print's reads, with a trust in (0, 1]
        reads = found[list(trust.READS)].to_numpy()
        names = out["names"].astype(str)
        assert all(names[k] in reads[j] for j, k in zip(out["print_index"], out["code_index"]))
        assert ((out["trust"] > 0) & (out["trust"] <= 1)).all()
