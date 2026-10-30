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

This project is open source in two parts. See `NOTICE` for exactly what each
part covers.

- **Code** (`pipeline/`, `site/`): [GNU AGPL v3.0](LICENSE). You can use and
  adapt it for your own city. If you distribute a modified version, or run one
  as a public website, you must publish your source code under the same
  license.
- **Content and data** (story text, research, map tiles, district data):
  [CC BY-NC-SA 4.0](LICENSE-content). You can remix it for non-commercial
  purposes, sharing your version under the same license and crediting it as:

  > Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org),
  > licensed CC BY-NC-SA 4.0.

Some material belongs to others and keeps its own terms: the 1948 atlas imagery
(David Rumsey Map Collection, CC BY-NC-SA 3.0) and the DataSF datasets (PDDL).
See `NOTICE`.

## Trademark

"GrowSF" and the GrowSF logo are trademarks of GrowSF. The licenses cover
copyright only. Credit GrowSF as above, but don't use the GrowSF name or logo
to name your version or to suggest that GrowSF made or endorses it.

## Contributing

Contributions are welcome under the [GrowSF Contributor License Agreement](CLA.md).
See [CONTRIBUTING.md](CONTRIBUTING.md).
