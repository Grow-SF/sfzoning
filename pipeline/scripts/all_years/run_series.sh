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
export SFZMAPS_MAX_USD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# the pinned toolchain's PROJ and GDAL data (pixi.toml), as a build's stages read them: the table's lots reproject the
# same way (2000's lots come out byte for byte the shipped ones)
pixi_env="$(cd "$here/.." && pwd)/.pixi/envs/default"
if [ -d "$pixi_env/share/proj" ]; then
  export PROJ_DATA="$pixi_env/share/proj" GDAL_DATA="$pixi_env/share/gdal" PROJ_NETWORK=ON
fi
cd "$here"
for step in "$@"; do
  case "$step" in
    walls) args=(walls --workers "${WALL_WORKERS:-2}") ;;
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
