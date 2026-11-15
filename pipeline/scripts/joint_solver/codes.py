"""Zoning codes: one spelling per code, whether printed on the map or written in the April 2000 table."""
import re

ALIAS = {
    "NO BEACH": "NORTH BEACH", "FILLMORE": "UPPER FILLMORE", "WEST PORTAL": "WEST PORTAL AVENUE", "CR-NC": "CRNC",
    "UPR MARKET": "UPPER MARKET", "24TH-NOE": "24TH-NOE VALLEY", "24TH-MISSION": "24TH MISSION", "HAYES": "HAYES-GOUGH",
    "HAIGHT ST": "HAIGHT", "GOUGH": "HAYES-GOUGH",
}
_DASH = re.compile(r"^(NC|RH|RM|RC|M|C)([0-9]+|S)$")


def tidy(code):
    """upper case, single spaces, a dash after the district letters: 'rh1' -> 'RH-1'"""
    code = re.sub(r"\s+", " ", str(code or "").strip().upper())
    return re.sub(r"^(RH|RM|RC|M|C)(\d)", r"\1-\2", code)


def canon(code):
    """the one spelling of a code.  A label that prints an overlay after the district ('HAIGHT ST NCD & ALCOHOL RES.')
    names the district first; dashed and undashed spellings are one code (NC2 = NC-2)."""
    code = tidy(code).split(" & ")[0].strip().removesuffix(" NCD")
    code = ALIAS.get(code, code)
    return _DASH.sub(r"\1-\2", code)


def is_overlay(name):
    """a Special Use District name: an overlay printed over a base district, never its code"""
    return "SUD" in name.upper().replace(".", "").replace(" ", "")


def table_vocabulary(lots):
    """every code the April 2000 table uses for a single district (split codes like 'RH-1/RH-2' left out)"""
    return {canon(z) for z in lots["zoning"].dropna().astype(str) if z and "/" not in z}
