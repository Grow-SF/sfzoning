"""One-off: provenance for the district reads made before it was recorded (through 2d1bf49).

Facts from the R26 run (docs/superpowers/notes, "Districts"): the per-region read was on for
1960, 1964, 1968 and the height codes of the four *-height series; it was off for 1972, 1975,
1980 and the use codes of the four *-zoning series (read before it existed).  1972, 1975 and
1980 were also read before 4d4ae58 (OS height districts never spread by plurality).  Reader A
(1921-1948) and today have no per-region read.  Existing provenance files are kept.

    uv run python scripts/backfill_district_provenance.py
"""
from sfzmaps.districts import assemble

REGION_READ = {s: True for s in ("1960", "1964", "1968", "1980-1990-height", "1990-1996-height",
                                 "1996-2000-height", "2000-digital-height")}
PRE_OS_FIX = "read before 4d4ae58 (OS height districts never spread by plurality)"
NOTES = {s: PRE_OS_FIX for s in ("1972", "1975", "1980")}

if __name__ == "__main__":
    for p in assemble.backfill_provenance(REGION_READ, NOTES):
        print(p)
