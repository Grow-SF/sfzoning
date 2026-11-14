#!/bin/zsh
# Rebuild one sheet with these scripts:
#   1. the p1 correction chain (corr.sh)
#   2. its mirror entry with the p1 correction (build_mirror_ay.py)
#   3. prob6 (prob.sh), on the page's ink as printed (no text removal since 2026-11-13)
# The 2000 sheets run in SAME2000 mode, as the original runs did: ftpx 3.99 and the measured area from
# stepview/city_m2/parts.json (rebuilt by make_parts.py into the rebuild folder).  usage: rebuild_one.sh SERIES SHEET
set -e
CK=${0:A:h}
source $CK/rebuild_env.sh
zsh $CK/torchenv.sh                                  # the torch environment, made if missing
SERIES=$1 SHEET=$2
mkdir -p $PW_RUN
if [[ $SERIES == 2000-digital-zoning ]]; then
  export SAME2000=1 PW_SCRATCH=$PW_DATA/rebuild-2026-11-13/scratch
  [ -f $PW_SCRATCH/stepview/city_m2/parts.json ] || { mkdir -p $PW_SCRATCH/stepview/city_m2 &&
    $PW_PIPELINE/.venv/bin/python $CK/make_parts.py $PW_SCRATCH/stepview/city_m2/parts.json; }
fi
echo "1. p1 $SERIES $SHEET $(date +%H:%M:%S)"
sh $PW_CODE/corr.sh $SERIES $SHEET
echo "2. mirror $SERIES $SHEET $(date +%H:%M:%S)"
( cd $PW_PIPELINE
  [ -d $PW_MIRROR/work/$SERIES ] && [ ! -L $PW_MIRROR/work/$SERIES ] || \
    env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $PW_CODE/build_mirror_ay.py $SERIES --prepare
  env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $PW_CODE/build_mirror_ay.py $SERIES $SHEET )
echo "3. prob6 $SERIES $SHEET $(date +%H:%M:%S)"
THREADS=3 sh $PW_CODE/prob.sh $SERIES $SHEET
echo "done $SERIES $SHEET $(date +%H:%M:%S)"
