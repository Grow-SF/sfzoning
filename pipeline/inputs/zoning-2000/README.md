# zoning-2000

The April 2000 zoning table the 2000 solver reads. The lot index everywhere in the solver is the row order of
`april2000.geojson.gz`.

- `april2000.geojson.gz`: SF Planning's "Historical Zoning Districts - April 2000" (DataSF dataset `cfhj-8c6b`), the
  whole table as GeoJSON, gzip-compressed. Made by `uv run sfzmaps solver fetch-table`; `PROVENANCE.md` records the
  URL, fetch time, row count and the sha256 of the uncompressed bytes. Public domain (PDDL 1.0). Re-fetching can
  change the row order, so don't re-fetch without rebuilding everything indexed by lot.
- `PROVENANCE.md`: where the table came from, written by the fetch.
- `table-corrections.csv`: by-eye checks of lots whose table code disagrees with the printed map (columns `group,
  verdict, pass, A, B, lots, blklots`; `blklots` is space-separated). A verdict `A` means column A is the district the
  map prints for those lots. They are used only in scoring (`table.corrected_codes`), never as solver input.
