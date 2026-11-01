# Map Colours That Say What The Land Is For

## Why

The owner wants the map to take a position. The densest housing and vibrant mixed-use areas are the best uses of land, so they read as the most positive colour, green. Zones that ban most housing, such as RH-1, read as red. Today's palette is a neutral warm ramp, with one shade for all apartments and a separate violet for mixed-use.

## What the reader sees (the owner's decisions)

- A red-to-green housing scale: bright red, through orange and pale yellow, to deep green ("scale A" in the mockup). The densest housing and mixed-use are deep green. RH-1 is red.
- Places that are mostly office towers are blue, even where housing is allowed.
- Light commercial is light blue, dense commercial is dark blue, and industrial is brown.
- SSO is light blue. C-1 and WMUO differ, because C-1 allowed homes and WMUO doesn't.

## The scale

Seven steps, for housing and mixed-use alike, by how many homes the zone allows. Mixed-use has no colour of its own; this is how the map "pivots" to it. (Proposed by the author; the owner saw it and raised only C-1.)

| Step | Limit | Zones |
|---|---|---|
| 1 | One home | RH-1, RH-1(D), R-1, R-1-D, 1R |
| 2 | Two homes | RH-1(S), RH-2, R-2 |
| 3 | Three homes | RH-3 |
| 4 | 1 per 600-800 sq ft | RM-1, RM-2, NC-1, NC-2, NC-3, NC-S, RC-1, RC-2 |
| 5 | 1 per 400 sq ft | RM-3, R-3, RC-3, C-1 |
| 6 | 1 per 200 sq ft | RM-4, R-4, RC-4, SLR |
| 7 | 1 per 125, or no limit | R-5, 2R, NCT, RTO, UMU, MUR, MUG, RH-DTR |

Named NCDs go by their own limits. The builder places each code by its limit in the research data; this table gives the intent. Mockup colours were `#DB2D27 #EB9632 #F1EC9D #9DC471 #3F9C48 #18682F #00381C`.

The rule for blue: a zone is blue if it bans housing, or is mostly offices or big commercial. It's on the scale if it's mainly homes, or shops with homes.

- Dark blue: all of C-3, C-M.
- Light blue: C-2, SSO, WMUO, MB-O.
- Brown: all industrial categories, including 1921's LI, which allowed housing.
- 1921's C stays blue, although it allowed unlimited housing. It's dark or light per the builder's reading of what it was.

## Author's calls

- Parks and public land move to a quiet grey-tan, since green now means something else.
- "Unrestricted" moves from navy to a distinct purple, because navy would read as dense commercial.
- MUO and CMUO are "Mixed-Use Office" zones that allow housing. By the owner's rule they are light blue, not green. The other zones in their research row stay green.
- Every era uses the same rules, so 1921's 2R (no limit on homes) is deep green.
- Zones with an unread or per-agreement limit (CCB, CRNC, CVR, RED, RSD, SPD, CMUO, per-development-agreement zones) are checked against the Planning Code. If still unknown, they go to the step of the closest comparable zone and are listed in the PR.

## Goals

1. The site, legend and popup show the new colours and labels in every era, in the story and the explorer.
2. Steps are evenly spaced in OKLab for normal vision.
3. Colour-blind readers can tell the steps and uses apart. Scale A as mocked fails this for protanopia: a difference of 1.2 between its closest pair. Change lightness or hue as little as possible to fix it, keeping the character of the mockup, and meet the bars in `site/tests/palette.test.ts`. If those bars can't hold across all 12 or so categories together, keep adjacent steps and uses distinct first, and report the real numbers rather than quietly lowering the bars.
4. Story text matches what's on screen. Ten or so steps in `site/content/story.ts` name colours. The owner approved a site that takes a position with colour, not new advocacy in the text.

## Non-goals

- No change to which zone a lot belongs to, to the scans, or to cameras and step order.
- No new paid reads. Rebuilds use cached answers (`--cache-only` or `--republish`).
- No rewriting of the research history in `docs/research/zoning-districts.csv`; added columns or corrections with a source are fine.

## Anti-goals

- A green that means parks.
- A map where dark red and deep green look the same to a colour-blind reader.
- Colour that claims more than the data knows: an unverified zone placed confidently without being listed.
- A claim in the story text that the new groupings make false.

## Constraints and risks

- Moving C-1 out of commercial, and splitting apartments, changes the chart's groups. The 1960 story step says that year's biggest shifts include "less commercial land". Check that this and every other numeric claim still holds under the new groups. If the groups would falsify a true historic claim, keep the claim true, and tell the owner.
- The categories live in `pipeline/src/sfzmaps/districts/categories.py` and `shared.py`, `site/lib/palette.ts`, `stats.ts`, `Legend.tsx`. The shared table carries category numbers, so the shared files must be rebuilt, with `ARCHIVE_FORMAT` and `MANIFEST_VERSION` handled, and published through `site/scripts/publish-tiles.mjs` under new names. Never overwrite a published file.
- Author's call on the switch: publish new tile names, ship a new site build that reads them, and leave the old files untouched. Tabs already open stay on the old colours until reloaded, and must keep working.
- Keep AGENTS.md rules on faithfulness and credits. "both" view mixes colours with scan ink, and the palette tests cover that view.
- vadimg/sfzoning has no license. Use it only as a cross-check; copy nothing.

## Left to the builder

- Exact final colours, within the goals.
- How the mapping is stored, and category names.
- Whether to fold steps in the legend or chart, so long as all steps are shown on the map.
- Exact shades of blue, brown, grey-tan and purple.

## How we'll know

- Pipeline tests, site tests and e2e with published tiles pass.
- Rendered maps of several eras viewed by the builder, including "both" over solid-black districts.
- The owner checks the PR preview on a phone.
