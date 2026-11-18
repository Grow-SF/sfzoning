# Licensing

sfzoning — the history of San Francisco zoning. Copyright 2026 GrowSF.
Created by Steven Bacio for GrowSF (https://growsf.org).

This repository is licensed in two parts. Neither license grants any right to
the GrowSF name or logo (see [Trademark](#trademark)). The license texts
govern; this page explains what each one covers.

## Code

Apache License 2.0 (Apache-2.0). See `LICENSE`.

Covers all source code, scripts, tests, build and configuration files,
including everything under `pipeline/` and `site/` except the paths listed
under [Content and data](#content-and-data).

In plain terms (the license text governs): you may use, modify and distribute
the code, commercially or not, including in closed-source products. If you
distribute it, modified or not, include `LICENSE` and `NOTICE`, mark the files
you changed, and keep the attribution in `NOTICE`:

> Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org).

We ask, though the license does not require it, that websites and apps built
on this code show that credit line in their credits or footer.

**Separate works.** The content and data under [Content and data](#content-and-data)
and the material under [Third-party material](#third-party-material) are
separate works with their own licenses. They are not part of the code licensed
under Apache-2.0, even where the site bundles or loads them. If you use the
code for a purpose those licenses don't allow, for example commercially,
replace that material or get permission for it.

## Content and data

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). See `LICENSE-content`.

Covers:

- the story text (`site/content/`) and the site's written copy;
- research and documentation (`docs/`), including
  `docs/research/zoning-districts.csv`;
- the generated data published from this project: georeferenced scans, map
  tiles (PMTiles), district and parcel layers, statistics and the processed
  bundle (`pipeline/data/share/`), whether in this repository, in the
  project's Google Drive folder, or served by sfzoning.com. For the
  georeferenced scans and tiles, the license covers this project's
  contribution (the alignment and processing), not the underlying maps; see
  [Third-party material](#third-party-material).

Required attribution: when you share this content or anything adapted from it,
credit it as

> Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org),
> licensed CC BY-NC-SA 4.0.

with a link to https://sfzoning.com where the medium allows, and keep the
third-party credits below.

## Third-party material

Not covered by the licenses above; its own terms apply.

- **1948 zoning atlas imagery:** David Rumsey Map Collection, licensed
  CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/).
  Credit: "David Rumsey Map Collection, David Rumsey Map Center, Stanford
  Libraries". See https://www.davidrumsey.com/about.

- **Zoning maps 1921–2000s:** official maps of the City and County of San
  Francisco, from the archives of the San Francisco Planning Department,
  scanned by Steven Bacio. To the best of our knowledge they are in the
  public domain: they were adopted as part of the city's zoning law, the
  older ones were published without copyright notice or before 1931, and
  they are public records. GrowSF and Steven Bacio claim no copyright in the
  maps or in the scans, which are faithful reproductions, and the licenses
  above do not apply to them. Those licenses cover only what this project
  adds: the georeferencing, the district classification, the research and
  text, and the code. Please credit "San Francisco Planning Department" as
  the source of the maps.

- **DataSF datasets** (Zoning Map – Zoning Districts 3i4a-hu95; Parcels –
  Active and Retired acdm-wktn; Streets – Active and Retired 3psu-pn9h;
  Historical Zoning Districts - April 2000 cfhj-8c6b): Open Data Commons
  Public Domain Dedication and License (PDDL) 1.0.

- **Fonts:** Newsreader and Libre Franklin, SIL Open Font License 1.1,
  self-hosted via next/font.

- **Software dependencies** (npm and Python packages) are used under their
  own licenses, all compatible with Apache-2.0: mostly MIT, Apache-2.0, BSD
  and ISC, plus libvips (LGPL-3.0, via sharp) used as a separate, dynamically
  linked library.

## Trademark

"GrowSF"™ and the GrowSF logo are trademarks of GrowSF (U.S. trademark
application Serial No. 99827114 for GROWSF). The licenses above cover
copyright only and grant no trademark rights. You may (and, for the content,
must) name GrowSF to give credit, but you may not use the GrowSF name or logo
to name your project or to suggest that GrowSF made, endorses or is affiliated
with your version. Please give your fork its own name.

## Contributions

Contributions are accepted under the GrowSF Individual Contributor License
Agreement (`CLA.md`). See `CONTRIBUTING.md`.

## Not an official zoning record

This is historical research. The district maps and classifications were
produced partly by automated reading of scanned maps and may contain errors.
They are not an official statement of any property's zoning, past or present.
For official zoning, consult the San Francisco Planning Department. All
material is provided "as is", without warranty of any kind, as set out in
`LICENSE` and `LICENSE-content`.
