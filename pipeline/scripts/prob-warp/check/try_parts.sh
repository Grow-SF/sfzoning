#!/bin/zsh
# For one candidate parts.json: the first p01 steps (regprep2, sides, distmeasure today) with SAME2000=1, into its own
# folder; then compare dm_today with the original.  usage: try_parts.sh NAME PARTS.json
source ${0:A:h}/rebuild_env.sh
N=$1; F=$2
T=$PW_DATA/rebuild-2026-11-13/try/$N
mkdir -p $T/scratch/stepview/city_m2 $T/out
cp $F $T/scratch/stepview/city_m2/parts.json
export SERIES=2000-digital-zoning SHEET=2000-digital-zoning-p01 SAME2000=1 PW_SCRATCH=$T/scratch AY_OUT=$T/out
cd $PW_PIPELINE
{ .venv/bin/python $PW_CODE/regprep2.py && .venv/bin/python $PW_CODE/sides.py && .venv/bin/python $PW_CODE/distmeasure.py today; } > $T/log 2>&1
.venv/bin/python - <<EOF
import numpy as np
a = np.load("$PW_RESCUED/allyears-run/2000-digital-zoning/2000-digital-zoning-p01/dm_2000-digital-zoning-p01_today.npz")
try:
    b = np.load("$T/out/dm_2000-digital-zoning-p01_today.npz")
except Exception as e:
    print("$N: no output", e); raise SystemExit
same = a["d"].shape == b["d"].shape and all(np.array_equal(a[k], b[k]) for k in a.files)
print("$N: points", b["d"].shape[0], "(original", a["d"].shape[0], ") identical" if same else ") differ")
EOF
