import pandas as pd

from sfzmaps.districts import check


def _raw(rows):
    return pd.DataFrame(rows, columns=["blklot", "code", "source_sheet"])


def test_a_case_counts_wrong_and_right_lots_inside_its_bbox_only():
    cent = pd.DataFrame({"blklot": ["a", "b", "c", "d"], "x": [1, 2, 3, 50], "y": [1, 2, 3, 50]})
    cases = pd.DataFrame([{"id": "x", "series": "1980", "x0": 0, "y0": 0, "x1": 10, "y1": 10,
                           "wrong": "RH-2", "right": {"RH-1", "P"}, "kind": "seam"}])
    raw = _raw([("a", "RH-2", None), ("b", "RH-1", None), ("c", "RM-1", None), ("d", "RH-2", None)])
    r = check.score_cases(cases, lambda s: raw, cent).iloc[0]
    assert (r.n, r.wrong, r.right) == (3, 1, 1)


def test_same_base_agreement_counts_only_lots_read_on_the_matching_sheets():
    a = _raw([("1", "RH-1", "A-p25"), ("2", "RH-1", "A-p25"), ("3", "RH-1", "A-p26")])
    b = _raw([("1", "RH-1", "B-p01"), ("2", "RH-2", "B-p01"), ("3", "RH-2", "B-p01")])
    share, n = check.agreement(a, b, {"A-p25": "B-p01", "A-p26": "B-p02"})
    assert (share, n) == (0.5, 2)       # lot 3 was read on p26 / p01: not the same drawing


def test_disagreements_cluster_by_place_and_code_pair():
    near = [(f"n{i}", 100.0 + 10 * i, 100.0) for i in range(25)]
    far = [(f"f{i}", 90000.0 + 10 * i, 100.0) for i in range(30)]
    cent = pd.DataFrame(near + far, columns=["blklot", "x", "y"])
    a = _raw([(b, "RH-2", "s") for b, _, _ in near + far])
    b = _raw([(b, "RH-1", "s") for b, _, _ in near + far])
    cl = check.clusters(a, b, cent, min_lots=20)
    assert cl.n.tolist() == [30, 25]
    assert cl.x0.iloc[0] == 90000


def test_the_committed_cases_name_known_use_series_and_ordered_boxes():
    cases = check.load_cases()
    assert set(cases.series) <= set(check.use_series())
    assert ((cases.x0 < cases.x1) & (cases.y0 < cases.y1)).all()
    assert set(cases.kind) <= {"seam", "harm", "other"}
    assert cases.id.is_unique
    assert all(cases.right) and cases.wrong.notna().all()


def test_park_scores_count_acres_of_parks_dated_by_the_map_year(monkeypatch):
    from sfzmaps import parks
    from sfzmaps.districts import assemble

    monkeypatch.setattr(assemble, "_parks_csv", lambda: pd.DataFrame(
        {"name": ["A", "B"], "blklot": ["p1", "p2"], "since": [1900, 1990]}))
    monkeypatch.setattr(check, "use_series", lambda: ["1980"])
    monkeypatch.setattr(parks, "map_year", lambda s: 1980)
    acre = 43560
    cent = pd.DataFrame({"blklot": ["p1", "p2", "h"], "x": 0, "y": 0, "area": [10 * acre, 5 * acre, acre]})
    raw = _raw([("p1", "R-1", "s"), ("p2", "P", "s"), ("h", "P", "s")])
    assert check.park_scores(lambda s: raw, cent) == {"1980": (0.0, 10.0)}   # p2 isn't a park yet
    raw = _raw([("p1", "P", "s")])
    assert check.park_scores(lambda s: raw, cent) == {"1980": (10.0, 10.0)}
