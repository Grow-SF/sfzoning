import pytest

from sfzmaps import pages
from sfzmaps.config import RAW_DIR
from sfzmaps.districts import categories, rules, series

# (year, code, category). The research CSV is the source of truth; cases where it overrides the
# brief's example mapping are marked DEVIATION. (None so far: every brief example matches the CSV.)
KNOWN = [
    (1921, "1R", "single_family"),     # 1R: max 1 unit (two-family only with 5,000 sq ft open area)
    (1921, "2R", "homes_top"),      # unlimited
    (1960, "R-1", "single_family"),
    (1960, "R-2", "two_unit"),
    (1960, "R-4", "homes_high"),     # 1 per 200 sq ft
    (1960, "C-2", "commercial_light"),
    (1980, "RH-2", "two_unit"),
    (1980, "RH-3", "three_unit"),
    (1980, "RM-1", "homes_low"),
    (1990, "NC-2", "homes_low"),           # NC-2 starts 1987, so year 1990 (not 1980) is the in-era check
    (1960, "M-1", "industrial"),
    (1960, "P", "public"),
    (1921, "U", "unrestricted"),
    # Extra CSV-driven cases.
    (1980, "RH-1(S)", "two_unit"),     # CSV max 2 (second unit): two_unit, not single_family
    (1980, "RH-1(D)", "single_family"),
    (1980, "RH-1", "single_family"),
    (1970, "R-4-C", "homes_high"),
    (1980, "RC-3", "homes_mid"),
    (1990, "SLI", "industrial"),
    (2010, "NCT-2", "homes_top"),
    (2010, "RTO", "homes_top"),     # "no numeric density limit"
    (1966, "R-3.5", "homes_mid"),   # CSV max_units "unknown": placed by hand, unverified
    (1970, "C-3-O", "commercial_dense"),
    (1921, "HI", "industrial"),
    (1960, "40' SHD", "unknown"),      # "special" rows feed height_code, not category
]


@pytest.mark.parametrize("year,code,expected", KNOWN)
def test_known_categories(year, code, expected):
    assert categories.category_for(code, year) == expected


def test_era_fallback_for_series_start_year():
    # NC-2 only exists from 1987 but a 1980-1990 series looks up at 1980: nearest era row is used.
    assert categories.category_for("NC-2", 1980) == "homes_low"
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


# DataSF today codes outside the original CSV (D7): every one resolves, with the category its
# Planning Code district purpose implies (rows added to the research CSV, confidence medium/low).
TODAY_CODES = [
    ("NCT-MISSION", "homes_top"), ("NCT-24TH-MISSION", "homes_top"), ("NCT-UPPER MARKET", "homes_top"),
    ("NCT-GLEN PARK", "homes_top"), ("RTO-1", "homes_top"), ("RH DTR", "homes_top"),
    ("SB-DTR", "homes_top"), ("TB DTR", "homes_top"), ("RCD", "homes_top"),
    ("SALI", "industrial"), ("RED-MX", "homes_top"), ("WMUG", "homes_top"), ("WMUO", "commercial_light"),
    ("CMUO", "commercial_light"), ("MR-MU", "homes_top"), ("BR-MU", "homes_top"), ("P70-MU", "homes_top"),
    ("PPS-MU", "homes_top"), ("S-MU", "homes_top"), ("MB-RA", "homes_top"), ("MB-O", "commercial_light"),
    ("HP-RA", "homes_top"), ("TI-MU", "homes_top"), ("TI-R", "homes_top"), ("TI-OS", "public"),
    ("TI-PCI", "public"), ("YBI-R", "homes_top"), ("YBI-OS", "public"), ("YBI-PCI", "public"),
    ("YBI-MU", "homes_top"), ("PM-R", "homes_top"), ("PM-OS", "public"), ("PM-MU1", "homes_top"),
    ("PM-MU2", "homes_top"), ("PM-CF", "public"), ("PM-S", "public"), ("Job Corps", "public"),
]


@pytest.mark.parametrize("code,expected", TODAY_CODES)
def test_today_special_codes(code, expected):
    assert categories.category_for(code, 2026) == expected


@pytest.mark.parametrize("code,family", [
    ("NCT-MISSION", "NCT"), ("RCD", "NC"), ("RH DTR", "DTR"), ("WMUG", "MU"), ("CMUO", "MU"),
    ("RTO-1", "RTO"), ("MB-RA", "other"), ("PM-R", "other"),
])
def test_family_for_today_codes(code, family):
    assert categories.family_for(code) == family


def test_mission_bay_open_space():
    assert categories.category_for("MB-OS", 2000) == "public"


# The map's categories: a seven-step housing scale (single_family .. homes_top) plus uses.
MAP_CATEGORY_TABLE = [
    ("RH-1", 2020, "single_family"),
    ("RH-1(D)", 2020, "single_family"),
    ("R-1", 1960, "single_family"),
    ("1R", 1948, "single_family"),
    ("RH-1(S)", 2020, "two_unit"),
    ("RH-2", 2020, "two_unit"),
    ("R-2", 1960, "two_unit"),
    ("RH-3", 2020, "three_unit"),
    ("RM-1", 2020, "homes_low"),
    ("RM-2", 2020, "homes_low"),
    ("NC-1", 2020, "homes_low"),
    ("NC-2", 2020, "homes_low"),
    ("NC-3", 2020, "homes_low"),
    ("NC-S", 2020, "homes_low"),
    ("RC-1", 1990, "homes_low"),
    ("RC-2", 1990, "homes_low"),
    ("RM-3", 2020, "homes_mid"),
    ("R-3", 1960, "homes_mid"),
    ("RC-3", 2020, "homes_mid"),
    ("C-1", 1960, "homes_mid"),
    ("RM-4", 2020, "homes_high"),
    ("R-4", 1960, "homes_high"),
    ("RC-4", 2020, "homes_high"),
    ("SLR", 2000, "homes_high"),
    ("R-5", 1960, "homes_top"),
    ("2R", 1948, "homes_top"),
    ("NCT-3", 2020, "homes_top"),
    ("RTO", 2020, "homes_top"),
    ("UMU", 2020, "homes_top"),
    ("MUR", 2020, "homes_top"),
    ("MUG", 2020, "homes_top"),
    ("RH-DTR", 2020, "homes_top"),
    ("C-3-O", 2020, "commercial_dense"),
    ("C-3-R", 2020, "commercial_dense"),
    ("C-3-G", 2020, "commercial_dense"),
    ("C-3-S", 2020, "commercial_dense"),
    ("C-3", 1960, "commercial_dense"),
    ("C-M", 2020, "commercial_dense"),
    ("C-2", 2020, "commercial_light"),
    ("SSO", 2000, "commercial_light"),
    ("WMUO", 2020, "commercial_light"),
    ("MB-O", 2020, "commercial_light"),
    ("MUO", 2020, "commercial_light"),
    ("CMUO", 2020, "commercial_light"),
    ("C", 1948, "commercial_dense"),
    ("M-1", 2020, "industrial"),
    ("PDR-1-G", 2020, "industrial"),
    ("LI", 1948, "industrial"),
    ("HI", 1948, "industrial"),
    ("P", 2020, "public"),
    ("U", 1948, "unrestricted"),
    # Code forms that old maps print land on the same step through normalisation.
    ("R-1-D", 1960, "single_family"),
    ("R1D", 1960, "single_family"),
    ("C-3-O(SD)", 2020, "commercial_dense"),
    ("C3O(SD)", 2020, "commercial_dense"),
    ("R-4-C", 1970, "homes_high"),
    ("R-5-C", 1970, "homes_top"),
    ("WMUG", 2020, "homes_top"),
    ("NCT-MISSION", 2020, "homes_top"),
    ("NCD", 2000, "homes_low"),
    ("Valencia NCD", 2000, "homes_low"),
    ("NCD-CASTRO", 2026, "homes_low"),
    # Broadway's 1 per 400 is the one NCD limit the research gives that differs from homes_low.
    ("BROADWAY NCD", 2000, "homes_mid"),
    ("NCD-BROADWAY", 2026, "homes_mid"),
    # Form-based, no numeric lot cap, in the research.
    ("RCD", 2026, "homes_top"),
]


@pytest.mark.parametrize("code,year,expected", MAP_CATEGORY_TABLE)
def test_map_category_table(code, year, expected):
    assert categories.category_for(code, year) == expected


def test_map_categories_are_the_agreed_set_in_order():
    assert categories.CATEGORIES == (
        "single_family", "two_unit", "three_unit", "homes_low", "homes_mid", "homes_high",
        "homes_top", "commercial_light", "commercial_dense", "industrial", "public",
        "unrestricted", "unknown")


@pytest.mark.parametrize("text,expected", [
    ("1 per 400 sq ft (6 on a 2,500 sq ft lot)", "homes_mid"),   # the density, not the parenthetical
    ("1 per 3,000 sq ft of lot area", "homes_low"),
    ("1 per 125 sq ft; a one-room apartment counts as 3/4", "homes_top"),
    ("No numeric density limit (form-based)", "homes_top"),
    ("unlimited", "homes_top"),
    ("1 per 249 sq ft", "homes_high"),
    ("1 per 250 sq ft", "homes_mid"),
    ("1 per 500 sq ft", "homes_low"),
])
def test_density_text_picks_the_step(text, expected):
    assert categories.step_from_text(text) == expected
