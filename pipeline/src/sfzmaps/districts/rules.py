"""Research CSV loader, code normaliser with alias table, era-aware rules join."""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from sfzmaps.config import PIPELINE_DIR

RULES_CSV = PIPELINE_DIR.parent / "docs" / "research" / "zoning-districts.csv"
OPEN_END = 9999  # "present" and "unknown" era ends are open


def load_rules(path: Path = RULES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["start"] = df["era_start"].astype(int)
    df["end"] = [int(e) if e.isdigit() else OPEN_END for e in df["era_end"]]
    return df


@lru_cache(maxsize=1)
def _default_rules() -> pd.DataFrame:
    return load_rules()


# Codes whose digits/suffix canonical form is a plain CSV code after normalisation.
_ALPHA_ONLY = {"C", "U", "LI", "HI", "P", "OS", "S", "X", "SLR", "SLI", "SSO", "RED", "SPD",
               "RSD", "CCB", "CRNC", "CVR", "RTO", "UMU", "MUR", "MUG", "MUO", "NCT", "NCD",
               "RH", "RM", "RC", "NC", "PDR", "SHD"}
_HAS_DIGIT_PREFIX = {"R", "RH", "RM", "RC", "NC", "NCT", "C", "M", "PDR"}
_ALPHA_ALIAS = {"CM": "C-M", "C-M": "C-M", "UNRESTRICTED": "U", "PUBLIC": "P", "NCS": "NC-S",
                "RTOM": "RTO-M", "RTO-M": "RTO-M", "RTOC": "RTO-C", "RTO-C": "RTO-C",
                "NC-S": "NC-S"}

# Canonical normalised code -> CSV `code` cell (compound cells cover several codes).
_CSV_KEY = {
    "RTO": "RTO / RTO-M", "RTO-M": "RTO / RTO-M",
    "NCT": "NCT-1/2/3 + named NCTs", "NCT-1": "NCT-1/2/3 + named NCTs",
    "NCT-2": "NCT-1/2/3 + named NCTs", "NCT-3": "NCT-1/2/3 + named NCTs",
    "UMU": "UMU / MUR / MUG / MUO", "MUR": "UMU / MUR / MUG / MUO",
    "MUG": "UMU / MUR / MUG / MUO", "MUO": "UMU / MUR / MUG / MUO",
    "PDR-1-B": "PDR-1-B / PDR-1-D / PDR-1-G / PDR-2", "PDR-1-D": "PDR-1-B / PDR-1-D / PDR-1-G / PDR-2",
    "PDR-1-G": "PDR-1-B / PDR-1-D / PDR-1-G / PDR-2", "PDR-2": "PDR-1-B / PDR-1-D / PDR-1-G / PDR-2",
    "CCB": "CCB / CRNC / CVR", "CRNC": "CCB / CRNC / CVR", "CVR": "CCB / CRNC / CVR",
    "RED": "RED / SPD / RSD", "SPD": "RED / SPD / RSD", "RSD": "RED / SPD / RSD",
    "NCD": "Named NCDs",
}

_SUFFIX_ALIAS = {"(D)": "(D)", "D": "(D)", "(S)": "(S)", "S": "(S)"}


def normalise_code(raw: str, year: int) -> str | None:
    """Canonical code ("R1" -> "R-1", "rh 2" -> "RH-2"), or None for anything unrecognised.

    `year` is accepted for era-specific aliases; none are needed so far."""
    s = re.sub(r"\s+", " ", str(raw or "").strip().upper())
    if not s:
        return None
    s = s.replace("–", "-").replace("—", "-")
    if re.search(r"\bNCD\b", s) or s.endswith(" NCD"):
        return "NCD"
    if re.fullmatch(r"[12]\s*R", s):                      # 1921-era "1R", "2R"
        return s.replace(" ", "")
    s = re.sub(r"[\s.]*$", "", s)
    flat = s.replace(" ", "").replace("-", "")
    if flat in _ALPHA_ALIAS:
        return _ALPHA_ALIAS[flat]
    if s in _ALPHA_ONLY:
        return s
    m = re.fullmatch(r"([A-Z]{1,3})[\s-]*(\d(?:\.\d)?)(?:[\s-]*(\([A-Z]+\)|[A-Z]))?(\(SD\))?", s)
    if not m or m.group(1) not in _HAS_DIGIT_PREFIX:
        m2 = re.fullmatch(r"(PDR)[\s-]*(1|2)(?:[\s-]*([BDG]))?", s)
        if not m2:
            return None
        return "PDR-" + "-".join(g for g in m2.groups()[1:] if g)
    prefix, num, suf, sd = m.groups()
    code = f"{prefix}-{num}"
    suffix = suf
    if suffix:
        if prefix in ("RH",):
            suffix = _SUFFIX_ALIAS.get(suffix, suffix)
            if suffix in ("(D)", "(S)"):
                return f"{code}{suffix}"
        suffix = suffix.strip("()")
        code = f"{code}-{suffix}"
    return code + "(SD)" if sd and code == "C-3-O" else code


def rule_for(code: str, year: int, rules: pd.DataFrame | None = None) -> dict | None:
    """The CSV row for `code` whose era range contains `year` (latest start wins an overlap)."""
    rules = _default_rules() if rules is None else rules
    norm = normalise_code(code, year)
    if norm is None:
        return None
    key = _CSV_KEY.get(norm, norm)
    hit = rules[(rules["code"] == key) & (rules["start"] <= year) & (rules["end"] >= year)]
    if hit.empty:
        return None
    return hit.sort_values("start").iloc[-1].to_dict()


def rules_for_any_era(code: str, rules: pd.DataFrame | None = None) -> list[dict]:
    """All CSV rows for `code` regardless of era, nearest-first not guaranteed (used for fallback)."""
    rules = _default_rules() if rules is None else rules
    norm = normalise_code(code, 0)
    if norm is None:
        return []
    return rules[rules["code"] == _CSV_KEY.get(norm, norm)].to_dict("records")
