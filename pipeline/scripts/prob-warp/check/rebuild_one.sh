#!/bin/zsh
# Rebuild one sheet with these scripts, in the order flow.py ran them:
#   1. the p1 correction chain (corr.sh)
#   2. its mirror entry with the p1 correction (build_mirror_ay.py)
#   3. for the series with printed codes (1960 on; flow.py ran 1921-1948 as CORR_ONLY): Terra's boxes and the frame
#      line replayed at $0 (terra_replay.sh), then the years chain's pages (chain_pages.sh).  The code fit is resumed
#      from the chain's surviving checkpoint (FITS_FROM, default the rescued SHEET_c1) until the code lists are rebuilt;
#      set CODES=FILE and FITS_FROM= to fit from scratch.
#   4. prob6 (prob.sh), which reads those pages
# The 2000 sheets run in SAME2000 mode, as the original runs did: ftpx 3.99 and the measured area from
# stepview/city_m2/parts.json (rebuilt by make_parts.py into the rebuild folder).  usage: rebuild_one.sh SERIES SHEET
set -e
CK=${0:A:h}
source $CK/rebuild_env.sh
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
case $SERIES in
  1921|1929|1935|1948) echo "3. no printed codes: no chain pages" ;;
  *)
    echo "3. chain pages $SERIES $SHEET $(date +%H:%M:%S)"
    D=$PW_CHAIN/$SERIES/$SHEET
    [ -f $D/textboxes_gpt-6.1-terra.json ] && [ -f $D/frame_bottom.json ] || zsh $CK/terra_replay.sh $SERIES $SHEET
    if [ -z "$CODES" ]; then
      export FITS_FROM=${FITS_FROM-$PW_RESCUED/allyears-chain/$SERIES/${SHEET}_c1}
    fi
    zsh $CK/chain_pages.sh $SERIES $SHEET ;;
esac
echo "4. prob6 $SERIES $SHEET $(date +%H:%M:%S)"
THREADS=3 sh $PW_CODE/prob.sh $SERIES $SHEET
echo "done $SERIES $SHEET $(date +%H:%M:%S)"
