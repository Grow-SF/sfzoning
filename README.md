# sfzoning

The history of San Francisco zoning, 1921 to today: the city's archived zoning
maps aligned to real streets, what each district allowed, and a scrolling story
built on them.

By Steven Bacio for [GrowSF](https://growsf.org).

- `pipeline/`: the georeferencing pipeline. It aligns scanned historic maps to
  modern GIS data automatically, reads each era's zoning districts, and
  publishes map tiles. It's built for San Francisco, but the approach should
  carry over to other cities' historic maps. See `pipeline/README.md`.
- `site/`: the Next.js scrollytelling site. See `site/README.md`.
- `docs/`: design notes, plans and research, including the table of what each
  district allowed in each era.

## License

The project is licensed in two parts. See `NOTICE` for exactly what each part
covers; the license texts govern.

- **Code** (`pipeline/`, `site/` except the story text): open source under the
  [GNU AGPL v3.0](LICENSE). You can use and adapt it for your own city. If you
  distribute it, or run a modified version as a public website, you must
  publish your source code under the same license and keep the credit
  "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)"
  visible in your site's credits (an additional term under AGPL section 7).
- **Content and data** (story text, research, map tiles, district data):
  [CC BY-NC-SA 4.0](LICENSE-content). You can remix it for non-commercial
  purposes, sharing your version under the same license and crediting it as:

  > Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org),
  > licensed CC BY-NC-SA 4.0.

The content's non-commercial terms aren't an open source license, and the
content isn't covered by the AGPL even though the site bundles it. Using the
code commercially means replacing that content.

Some material isn't ours to license. The historic zoning maps are official
San Francisco maps from the Planning Department's archives; to the best of our
knowledge they're in the public domain, and we claim no rights in them or in
the scans. The 1948 atlas imagery (David Rumsey Map Collection,
CC BY-NC-SA 3.0) and the DataSF datasets (PDDL) keep their own terms. See
`NOTICE`.

## Trademark

GrowSF™ and the GrowSF logo are trademarks of GrowSF (U.S. application Serial
No. 99827114). The licenses cover copyright only. Credit GrowSF as above, but
don't use the GrowSF name or logo to name your version or to suggest that
GrowSF made or endorses it.

## Not an official zoning record

This is historical research. The district maps were produced partly by
automated reading of scanned maps and may contain errors. They are not an
official statement of any property's zoning. For official zoning, consult the
[San Francisco Planning Department](https://sfplanning.org).

## Contributing

Contributions are welcome under the
[GrowSF Individual Contributor License Agreement](CLA.md). See
[CONTRIBUTING.md](CONTRIBUTING.md).
