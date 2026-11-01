# Map category notes

How `pipeline/src/sfzmaps/districts/categories.py` places each zone on the housing scale, and
which placements are not confirmed against the Planning Code.

The scale (steps by homes allowed): `single_family` (1), `two_unit` (2), `three_unit` (3),
`homes_low` (1 per 500 sq ft or more), `homes_mid` (250-499), `homes_high` (150-249),
`homes_top` (under 150, form-based, or no limit). Blue and brown zones are placed by hand.
`zoning-districts.csv`'s `category` column is unchanged.

## Verification status

The SF Planning Code at codelibrary.amlegal.com returned HTTP 403 to an automated fetch on
2026-11-01, so no limit below was checked against it. Every row marked unverified is the
closest comparable zone, not a reading of the code. Source for everything else is the research
CSV (`docs/research/zoning-districts.csv`, `max_units_per_lot`).

| Zone | Step | Basis | Status |
|---|---|---|---|
| CCB, CRNC, CVR | homes_high | CSV "not read"; Chinatown districts, compared with RM-4 / RC-4 (1 per 200) | unverified |
| RED, SPD, RSD | homes_top | CSV "not read"; compared with the form-based mixed-use zones (UMU, MUR) | unverified |
| RED-MX | homes_top | CSV "form-based; not verified" | unverified |
| WMUG | homes_top | CSV "form-based; not verified"; follows UMU/MUG | unverified |
| RCD | homes_low | CSV "form-based; not verified"; compared with the NC districts (NC-2) | unverified |
| RTO-C | homes_top | CSV "not read (form-based expected)"; follows RTO | unverified |
| MR-MU, BR-MU, P70-MU, PPS-MU, S-MU | homes_top | per development agreement, no per-lot cap | unverified |
| MB-RA, HP-RA | homes_top | per redevelopment plan | unverified |
| TI-R, YBI-R, PM-R, TI-MU, YBI-MU, PM-MU1, PM-MU2 | homes_top | per development agreement | unverified |
| Named NCDs (NCD) | homes_low | CSV: varies (Broadway 1/400, Castro 1/600, Inner Clement 1/600); no confirmed limit | unverified |
| Named NCTs (NCT-MISSION etc.) | homes_top | CSV: NCT form-based, no density limit | from CSV |
| R-3.5 (1964-78) | homes_mid | CSV "unknown"; sits between R-3 (1 per 400) and R-4 (1 per 200) | unverified |
| R-4-C, R-5-C (1968-78) | homes_high, homes_top | CSV "presumably R-4 / R-5 density; not verified" | unverified |
| C-3-O, C-3-R, C-3-G, C-3-S (1968-85) | commercial_dense | blue by owner's rule; housing limit not verified in CSV | blue regardless |

Also by hand, not about limits: C-1 is `homes_mid` (CSV: never less than R-3, 1 per 400);
1921's C is `commercial_dense` (owner's ruling: it held downtown); MUO and CMUO are
`commercial_light` (Mixed-Use Office) while UMU, MUR and MUG share their research row and are
`homes_top`.

## Not placed

`RM-5` (1989-90 reads), `C-2-S`, `R-1-C`, `RM`, `RM-1(S)`, `MB-R-*`, `MB-NC-*`, `RNC` and other
codes that appear in `unknown_codes.json` have no research row and stay `unknown`. This task did
not change that.
