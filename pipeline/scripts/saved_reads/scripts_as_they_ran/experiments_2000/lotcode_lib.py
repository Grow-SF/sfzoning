"""One spelling for a zoning code, whether printed on the map or written in the table."""
import truth

ALIAS = {"NO BEACH": "NORTH BEACH", "FILLMORE": "UPPER FILLMORE", "WEST PORTAL": "WEST PORTAL AVENUE", "CR-NC": "CRNC",
         "UPR MARKET": "UPPER MARKET", "24TH-NOE": "24TH-NOE VALLEY", "24TH-MISSION": "24TH MISSION", "HAYES": "HAYES-GOUGH"}


import re

_DASH = re.compile(r"^(NC|RH|RM|RC|M|C)([0-9]+|S)$")             # owner 2026-11-12: dashed and undashed spellings are one


ALIAS.update({"HAIGHT ST": "HAIGHT", "GOUGH": "HAYES-GOUGH"})   # 2026-11-13: the district part of a label that
                                                                #   prints an overlay after it; the other half of a split name


def canon(c):
    c = truth.norm(c).split(" & ")[0].strip()                   # "HAIGHT ST NCD & ALCOHOL RES.", "NC-2 & FINANCIAL RUSD":
    c = c.removesuffix(" NCD")                                  #   the district comes first, the overlay after the "&"
    c = ALIAS.get(c, c)
    return _DASH.sub(r"\1-\2", c)                                 # NC2 -> NC-2, NCS -> NC-S, RM3 -> RM-3
