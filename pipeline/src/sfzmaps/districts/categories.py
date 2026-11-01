"""Code -> story category (from the research CSV) and code family (for validation)."""
import re

from sfzmaps.districts.rules import normalise_code, rule_for, rules_for_any_era

CATEGORIES = ("single_family", "two_unit", "three_unit", "homes_low", "homes_mid", "homes_high",
              "homes_top", "commercial_light", "commercial_dense", "industrial", "public",
              "unrestricted", "unknown")

_UNITS = {1: "single_family", 2: "two_unit", 3: "three_unit"}

# Zones placed by hand: blue zones, and zones whose research text has no density to read or
# reads misleadingly. Keyed by the canonical code from normalise_code. Entries marked
# "unverified" are listed in docs/research/map-category-notes.md.
_EXPLICIT = {
    # Commercial, by the owner's rule: blue if housing is banned or it is mostly offices.
    "C": "commercial_dense",            # 1921's C held downtown
    "C-2": "commercial_light", "SSO": "commercial_light", "WMUO": "commercial_light",
    "MB-O": "commercial_light", "MUO": "commercial_light", "CMUO": "commercial_light",
    "C-M": "commercial_dense", "C-3": "commercial_dense", "C-3-O": "commercial_dense",
    "C-3-O(SD)": "commercial_dense", "C-3-R": "commercial_dense", "C-3-G": "commercial_dense",
    "C-3-S": "commercial_dense",
    # C-1 allowed homes at R-3 density or more.
    "C-1": "homes_mid",
    # R-3.5 is between R-3 (1 per 400) and R-4 (1 per 200); the CSV has no number. Unverified.
    "R-3.5": "homes_mid",
    # Named NCDs vary; no confirmed limit. Unverified.
    "NCD": "homes_low",
    # Unread limits or per-agreement / per-plan zones: the closest comparable zone. Unverified.
    "CCB": "homes_high", "CRNC": "homes_high", "CVR": "homes_high",
    "RED": "homes_top", "SPD": "homes_top", "RSD": "homes_top", "RED-MX": "homes_top",
    "RCD": "homes_low", "RTO-C": "homes_top",
    "MB-RA": "homes_top", "HP-RA": "homes_top",
}

_DENSITY = re.compile(r"\b1\s*per\s*([\d,]+)")
_NO_LIMIT = re.compile(r"unlimited|no (?:numeric )?(?:density )?limit|form-based|per development agreement"
                       r"|per redevelopment plan|no per-lot cap", re.I)


def step_from_text(text: str) -> str | None:
    """Step from a research `max_units_per_lot` cell: a bare or 'per lot' cap of 1-3 stays as it
    is; else the first '1 per N' (N sq ft) sets the step; 'no limit' and form-based are the top.
    Parentheticals ('6 on a 2,500 sq ft lot') are never read. None when it says nothing."""
    text = text.strip().lower()
    m = re.match(r"(\d+)\b(?!\s*per\s+(?!lot\b))", text)
    if m and int(m.group(1)) in _UNITS:
        return _UNITS[int(m.group(1))]
    d = _DENSITY.search(text)
    if d:
        n = int(d.group(1).replace(",", ""))
        if n >= 500:
            return "homes_low"
        if n >= 250:
            return "homes_mid"
        if n >= 150:
            return "homes_high"
        return "homes_top"
    if _NO_LIMIT.search(text):
        return "homes_top"
    return None


def _from_rule(rule: dict, norm: str | None) -> str:
    if norm in _EXPLICIT:
        return _EXPLICIT[norm]
    cat = rule["category"]
    if cat in ("residential", "mixed"):
        return step_from_text(rule["max_units_per_lot"]) or "homes_top"
    if cat == "commercial":
        return "commercial_light"
    if cat in ("industrial", "public", "unrestricted"):
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
    return _from_rule(rule, normalise_code(code, year))


_FAMILY_PREFIXES = (("RH", "RH"), ("RM", "RM"), ("RC", "RC"), ("RTO", "RTO"), ("NCT", "NCT"),
                    ("NCD", "NC"), ("NC", "NC"), ("PDR", "PDR"), ("UMU", "MU"), ("MUR", "MU"),
                    ("MUG", "MU"), ("MUO", "MU"))


_FAMILY_EXACT = {"RCD": "NC", "RH-DTR": "DTR", "SB-DTR": "DTR", "TB-DTR": "DTR", "CMUO": "MU",
                 "WMUG": "MU", "WMUO": "MU", "RED-MX": "MU", "SALI": "PDR"}


def family_for(code: str) -> str:
    """Era-independent family: R (1960s R-n), RH, RM, RC, RTO, NC, NCT, C3, C, M, MU, P, PDR,
    DTR, other."""
    norm = normalise_code(code, 0)
    if norm is None:
        return "other"
    if norm in _FAMILY_EXACT:
        return _FAMILY_EXACT[norm]
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
