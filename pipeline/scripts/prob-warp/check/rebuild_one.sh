#!/bin/zsh
# Rebuild one sheet with these scripts: the p1 correction chain, then prob6.  The 2000 sheets run in SAME2000 mode,
# as the original runs did: ftpx 3.99 and the measured area from stepview/city_m2/parts.json (rebuilt by
# make_parts.py into the rebuild folder).  usage: rebuild_one.sh SERIES SHEET
source ${0:A:h}/rebuild_env.sh
mkdir -p $PW_RUN
if [[ $1 == 2000-digital-zoning ]]; then
  export SAME2000=1 PW_SCRATCH=$PW_DATA/rebuild-2026-11-13/scratch
  [ -f $PW_SCRATCH/stepview/city_m2/parts.json ] || { mkdir -p $PW_SCRATCH/stepview/city_m2 &&
    $PW_PIPELINE/.venv/bin/python ${0:A:h}/make_parts.py $PW_SCRATCH/stepview/city_m2/parts.json; }
fi
sh $PW_CODE/corr.sh $1 $2 && THREADS=3 sh $PW_CODE/prob.sh $1 $2
