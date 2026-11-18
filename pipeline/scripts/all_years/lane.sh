#!/usr/bin/env bash
# Run series one after another (one lane), each through the steps named: several lanes side by side keep the machine's
# load in bounds (each series' walls use WALL_WORKERS processes, its solve four).
#
#   scripts/all_years/lane.sh "SERIES SERIES ..." STEP...
set -uo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
root="$here/data/solver/experiments/all-years"
series=$1
shift
case "$series" in @*) series=$(cat "${series#@}") ;; esac     # @FILE: the series listed in FILE
for s in $series; do
  if [ ! -f "$root/$s/lots/lots.npz" ] && [ "$s" != "1996-2000-zoning" ]; then
    mkdir -p "$root/$s/lots"
    cp "$root/1921/lots/lots.npz" "$root/1921/lots/table.parquet" "$root/$s/lots/"
  fi
  "$here/scripts/all_years/run_series.sh" "$s" "$@"
done
