#!/usr/bin/env bash
# Run the same steps for several series side by side (run_series.sh each); every series that has no lots yet gets the
# lots every series without the table shares (1921's: the April 2000 table's lots with no codes, the same bytes for
# every such series).
#
#   scripts/all_years/run_many.sh "SERIES SERIES ..." STEP...
set -uo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
root="$here/data/solver/experiments/all-years"
series=$1
shift
for s in $series; do
  if [ ! -f "$root/$s/lots/lots.npz" ] && [ "$s" != "1996-2000-zoning" ]; then
    mkdir -p "$root/$s/lots"
    cp "$root/1921/lots/lots.npz" "$root/1921/lots/table.parquet" "$root/$s/lots/"
  fi
  "$here/scripts/all_years/run_series.sh" "$s" "$@" &
done
wait
