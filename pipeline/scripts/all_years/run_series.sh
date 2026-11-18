#!/usr/bin/env bash
# Run solver steps for one series, each logged to data/solver/experiments/all-years/logs/<series>-<step>.log; stops at
# the first step that fails.  One thread for the maths libraries in every process.  $0: no model is asked.
#
#   scripts/all_years/run_series.sh SERIES STEP...      (steps: page lots prints text-chance labels walls solve)
set -uo pipefail
series=${1:?usage: run_series.sh SERIES STEP...}
shift
here=$(cd "$(dirname "$0")/../.." && pwd)
logs="$here/data/solver/experiments/all-years/logs"
mkdir -p "$logs"
export SFZMAPS_MAX_USD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
cd "$here"
for step in "$@"; do
  case "$step" in
    walls) args=(walls --workers 2) ;;
    solve) args=(solve final) ;;
    *) args=("$step") ;;
  esac
  start=$(date +%s)
  if uv run sfzmaps solver --series "$series" "${args[@]}" > "$logs/$series-$step.log" 2>&1; then
    echo "$series $step ok $(( $(date +%s) - start )) s"
  else
    echo "$series $step FAILED $(( $(date +%s) - start )) s (see $logs/$series-$step.log)"
    exit 1
  fi
done
