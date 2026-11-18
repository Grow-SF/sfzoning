#!/usr/bin/env bash
# Run series one after another (one lane), each through the steps named: several lanes side by side keep the machine's
# load in bounds (each series' walls use WALL_WORKERS processes, its solve four).  A series with no lots yet gets the
# lots every series shares (1921's: the April 2000 table's lots with no codes, the same bytes for every series).
#
#   scripts/all_years/lane.sh "SERIES SERIES ..." STEP...     (or @FILE: the series listed in FILE)
set -uo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
root="$here/data/solver"
series=$1
shift
case "$series" in @*) series=$(cat "${series#@}") ;; esac
for s in $series; do
  if [ ! -f "$root/$s/lots/lots.npz" ] && [ -f "$root/1921/lots/lots.npz" ] && [ "$s" != "1921" ]; then
    mkdir -p "$root/$s/lots"
    cp "$root/1921/lots/lots.npz" "$root/1921/lots/table.parquet" "$root/$s/lots/"
  fi
  "$here/scripts/all_years/run_series.sh" "$s" "$@"
done
