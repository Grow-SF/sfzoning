"""Code -> story category (from the research CSV) and code family (for validation)."""
import re

from sfzmaps.districts.rules import normalise_code, rule_for, rules_for_any_era

CATEGORIES = ("single_family", "two_unit", "three_unit", "multi_family", "mixed", "commercial",
              "industrial", "public", "unrestricted", "unknown")

_UNITS = {1: "single_family", 2: "two_unit", 3: "three_unit"}


def _residential(rule: dict) -> str:
    """Story category from `max_units_per_lot`: a bare 'N' or 'N per lot' is a per-lot cap;
    'N per <area>' (density), 'unlimited', 'unknown', 'No numeric limit' all mean multi_family."""
    text = rule["max_units_per_lot"].strip().lower()
    m = re.match(r"(\d+)\b(?!\s*per\s+(?!lot\b))", text)
    if m and int(m.group(1)) in _UNITS:
        return _UNITS[int(m.group(1))]
    return "multi_family"


def _from_rule(rule: dict) -> str:
    cat = rule["category"]
    if cat == "residential":
        return _residential(rule)
    if cat in ("mixed", "commercial", "industrial", "public", "unrestricted"):
        return cat
    return "unknown"      # "special" rows (height letters, SHDs, boundary rule, R/T/V)


def category_for(code: str, year: int) -> str:
    """Story category. Strict era match first; if the code exists in the CSV only in another era
    (a series' single lookup year can sit outside the era of a code printed on it), use the
    nearest-era row. Codes absent from the CSV are 'unknown'."""
    rule = rule_for(code, year)
    if rule is None:
        rows = rules_for_any_era(code)
        if not rows:
            return "unknown"
        rule = min(rows, key=lambda r: min(abs(year - r["start"]), abs(year - r["end"])))
    return _from_rule(rule)


_FAMILY_PREFIXES = (("RH", "RH"), ("RM", "RM"), ("RC", "RC"), ("RTO", "RTO"), ("NCT", "NCT"),
                    ("NCD", "NC"), ("NC", "NC"), ("PDR", "PDR"), ("UMU", "MU"), ("MUR", "MU"),
                    ("MUG", "MU"), ("MUO", "MU"))


def family_for(code: str) -> str:
    """Era-independent family: R (1960s R-n), RH, RM, RC, RTO, NC, NCT, C3, C, M, MU, P, PDR, other."""
    norm = normalise_code(code, 0)
    if norm is None:
        return "other"
    if norm.startswith("C-3"):
        return "C3"
    for pre, fam in _FAMILY_PREFIXES:
        if norm.startswith(pre):
            return fam
    if norm in ("1R", "2R") or re.match(r"R-\d", norm):
        return "R"
    if norm in ("C", "C-M") or re.match(r"C-[12]", norm):
        return "C"
    if re.match(r"M-\d", norm):
        return "M"
    if norm == "P":
        return "P"
    return "other"
