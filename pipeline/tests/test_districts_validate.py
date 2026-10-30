import pandas as pd
import pytest

from sfzmaps.districts import validate


def _df(rows):
    return pd.DataFrame(rows, columns=["blklot", "code", "category"])


def test_agreement_math():
    a = _df([("1", "RH-1", "single_family"), ("2", "RH-2", "two_unit"), ("3", "RM-1", "multi_family"),
             ("4", "NC-2", "mixed"), ("5", None, "unknown"), ("6", "M-1", "industrial"),
             ("9", "P", "public")])
    b = _df([("1", "RH-1(D)", "single_family"), ("2", "RH-2", "two_unit"), ("3", "NCT-3", "mixed"),
             ("4", "NC-2", "mixed"), ("5", "RH-1", "single_family"), ("6", "PDR-1-G", "industrial"),
             ("7", "P", "public")])
    r = validate.agreement(a, b)
    assert r["n_common"] == 6                     # 1-6; 7 and 9 are one-sided
    assert r["n_compared"] == 5                   # 5 is unknown in a
    assert r["category_agreement"] == pytest.approx(4 / 5, abs=1e-4)       # 1, 2, 4, 6
    assert r["category_agreement_all"] == pytest.approx(4 / 6, abs=1e-4)
    assert r["code_agreement"] == pytest.approx(2 / 5, abs=1e-4)           # 2, 4 (normalised codes)
    assert r["family_agreement"] == pytest.approx(3 / 5, abs=1e-4)         # RH, RH, NC
    assert r["by_family"]["RH"] == {"n": 2, "agree": 2, "share": 1.0}
    assert r["family_confusion"]["RM"] == {"NCT": 1}
    assert r["family_confusion"]["M"] == {"PDR": 1}
    assert r["category_confusion"]["multi_family"] == {"mixed": 1}
    assert r["by_category"]["single_family"]["share"] == 1.0


def test_agreement_empty():
    r = validate.agreement(_df([]), _df([]))
    assert r["n_common"] == 0 and r["category_agreement"] == 0.0


def test_key_accuracy(tmp_path):
    key = tmp_path / "k.csv"
    key.write_text("blklot,code\n0001001,1R\n0001002,2R\n0001003,C\n0001004,LI\n0001005,\n")
    df = pd.DataFrame({"blklot": ["0001001", "0001002", "0001003", "0001009"],
                       "code": ["1 R", "2R", "LI", "C"]})
    # 4 scored rows (a blank key code is skipped); 1001, 1002 right, 1003 wrong, 1004 missing
    assert validate.key_accuracy(df, key) == pytest.approx(2 / 4, abs=1e-4)
    rows = validate.key_report(df, key)
    assert list(rows["ok"]) == [True, True, False, False]
