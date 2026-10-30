import pytest

from sfzmaps import pages
from sfzmaps.config import RAW_DIR
from sfzmaps.districts import categories, rules, series

# (year, code, category). The research CSV is the source of truth; cases where it overrides the
# brief's example mapping are marked DEVIATION. (None so far: every brief example matches the CSV.)
KNOWN = [
    (1921, "1R", "single_family"),     # 1R: max 1 unit (two-family only with 5,000 sq ft open area)
    (1921, "2R", "multi_family"),      # unlimited
    (1960, "R-1", "single_family"),
    (1960, "R-2", "two_unit"),
    (1960, "R-4", "multi_family"),     # 1 per 200 sq ft
    (1960, "C-2", "commercial"),
    (1980, "RH-2", "two_unit"),
    (1980, "RH-3", "three_unit"),
    (1980, "RM-1", "multi_family"),
    (1990, "NC-2", "mixed"),           # NC-2 starts 1987, so year 1990 (not 1980) is the in-era check
    (1960, "M-1", "industrial"),
    (1960, "P", "public"),
    (1921, "U", "unrestricted"),
    # Extra CSV-driven cases.
    (1980, "RH-1(S)", "two_unit"),     # CSV max 2 (second unit): two_unit, not single_family
    (1980, "RH-1(D)", "single_family"),
    (1980, "RH-1", "single_family"),
    (1970, "R-4-C", "mixed"),
    (1980, "RC-3", "mixed"),
    (1990, "SLI", "industrial"),
    (2010, "NCT-2", "mixed"),
    (2010, "RTO", "multi_family"),     # "no numeric density limit"
    (1966, "R-3.5", "multi_family"),   # CSV max_units "unknown" -> multi_family default
    (1970, "C-3-O", "commercial"),
    (1921, "HI", "industrial"),
    (1960, "40' SHD", "unknown"),      # "special" rows feed height_code, not category
]


@pytest.mark.parametrize("year,code,expected", KNOWN)
def test_known_categories(year, code, expected):
    assert categories.category_for(code, year) == expected


def test_era_fallback_for_series_start_year():
    # NC-2 only exists from 1987 but a 1980-1990 series looks up at 1980: nearest era row is used.
    assert categories.category_for("NC-2", 1980) == "mixed"
    # strict rule_for stays strict
    assert rules.rule_for("NC-2", 1980) is None


def test_rule_for_era_selection():
    assert rules.rule_for("2R", 1935)["era_start"] == "1929"      # 1935 -> 1929-1946
    assert rules.rule_for("2R", 1921)["era_start"] == "1921"
    assert rules.rule_for("2R", 1929)["era_start"] == "1929"      # overlap: latest start wins
    assert rules.rule_for("2R", 1948)["era_start"] == "1946"
    assert rules.rule_for("R-1", 1980) is None                    # R-1 ended 1978
    assert rules.rule_for("RH-2", 2024)["era_end"] == "present"


def test_load_rules_and_compound_codes():
    df = rules.load_rules()
    assert {"start", "end", "code", "category"} <= set(df.columns)
    assert rules.rule_for("UMU", 2015)["code"] == "UMU / MUR / MUG / MUO"
    assert rules.rule_for("MUO", 2015)["category"] == "mixed"
    assert rules.rule_for("PDR-1-G", 2015)["category"] == "industrial"
    assert rules.rule_for("Upper Market NCD", 1995)["code"] == "Named NCDs"


@pytest.mark.parametrize("raw,expected", [
    ("R1", "R-1"), ("rh 2", "RH-2"), (" C-3-O ", "C-3-O"), ("r-1-d", "R-1-D"),
    ("RH-1 (D)", "RH-1(D)"), ("RH1S", "RH-1(S)"), ("1 R", "1R"), ("2R", "2R"), ("c", "C"),
    ("CM", "C-M"), ("nc 2", "NC-2"), ("NCT-3", "NCT-3"), ("rto-m", "RTO-M"),
    ("M1", "M-1"), ("p", "P"), ("PDR 1 G", "PDR-1-G"), ("R-3.5", "R-3.5"),
    ("garbage", None), ("", None), ("???", None), ("ZZ-9", None),
])
def test_normalise_aliases(raw, expected):
    assert rules.normalise_code(raw, 1980) == expected


def test_unknown_code_is_unknown():
    assert categories.category_for("ZZ-9", 1980) == "unknown"
    assert categories.category_for("garbage", 1960) == "unknown"
    assert rules.rule_for("garbage", 1960) is None


@pytest.mark.parametrize("code,family", [
    ("RH-2", "RH"), ("RM-1", "RM"), ("RC-3", "RC"), ("NC-2", "NC"), ("NCT-1", "NCT"),
    ("C-3-O", "C3"), ("C-2", "C"), ("M-1", "M"), ("P", "P"), ("PDR-1-D", "PDR"),
    ("R-1", "R"), ("1R", "R"), ("UMU", "MU"), ("RTO", "RTO"), ("garbage", "other"), ("OS", "other"),
])
def test_family_for(code, family):
    assert categories.family_for(code) == family


EXPECTED_SERIES = (
    {"1921", "1929", "1935", "1948", "1960", "1964", "1968", "1972", "1975", "1980"}
    | {f"{p}-{k}" for p in ("1980-1990", "1990-1996", "1996-2000", "2000-digital")
       for k in ("zoning", "height", "su")}
)


def test_series_table_matches_spec():
    assert set(series.SERIES) == EXPECTED_SERIES and len(series.SERIES) == 22
    S = series.SERIES
    assert [S[s].reader for s in ("1921", "1929", "1935", "1948")] == ["A"] * 4
    assert all(S[s].reader == "B" and S[s].role == "use" for s in ("1960", "1980", "1996-2000-zoning"))
    assert S["2000-digital-height"].role == "height" and S["2000-digital-height"].reader == "B"
    assert S["1990-1996-su"].reader == "skip" and S["1990-1996-su"].role == "skip"
    assert S["1980-1990-height"].period == S["1980-1990-zoning"].period == "1980-1990"
    assert S["1960"].period == "1960" and S["1935"].year == 1935


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw scans not present")
def test_series_table_complete():
    found = {p.series for p in pages.discover()}
    assert found <= set(series.SERIES)
