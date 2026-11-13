#!/bin/zsh
# The years chain's pages that prob6 reads, rebuilt with the stages and settings sheetAY.sh ran (it went on to the
# graph and labels, which prob6 does not read): raw graph, block outlines deleted (SHEET/noblock.png), walls
# (SHEET/walls.json), the zoning codes fitted and erased (SHEET_c1/clean.png).
# Inputs: the sheet's mirror entry with its p1 correction (build_mirror_ay.py), and Terra's boxes and the frame
# line in $PW_CHAIN/SERIES/SHEET (terra_replay.sh).  The code fit runs from scratch against CODES (the series' code
# list: codes.py, or 2000's codes_final.json); with FITS_FROM=DIR it is resumed from DIR/fits_raw.pkl, fitboxes.py's
# own checkpoint (RESUME: the fitting is skipped and only the erase runs, so the code list is not read).
# usage: [FITS_FROM=DIR | CODES=FILE] [NPROC=4] chain_pages.sh SERIES SHEET
set -e
source ${0:A:h}/rebuild_env.sh
export SERIES=$1 SHEET=$2
D=$PW_CHAIN/$SERIES/$SHEET; G=$D/raw; DC=$PW_CHAIN/$SERIES/${SHEET}_c1
mkdir -p $G $DC
cd $PW_PIPELINE
export SFZ_CORR=1 PYTHONPATH=$PW_CODE/../corrpatch SFZMAPS_DATA_DIR=$PW_MIRROR
PY=$PW_PIPELINE/.venv/bin/python
PAGE=$PW_MIRROR/work/$SERIES/$SHEET/page.png
stage() { echo "STAGE $SERIES $SHEET $1 $(date +%H:%M:%S)"; }
[ -f $D/textboxes_gpt-6.1-terra.json ] || { echo "FAILED $SHEET: no Terra boxes (terra_replay.sh)"; exit 1; }
[ -f $D/frame_bottom.json ] || { echo "FAILED $SHEET: no frame line (terra_replay.sh)"; exit 1; }
if [ -n "$FITS_FROM" ]; then
  [ -f $FITS_FROM/fits_raw.pkl ] || { echo "FAILED $SHEET: no $FITS_FROM/fits_raw.pkl"; exit 1; }
  echo '[]' > $DC/codes_not_read.json
  CODES=$DC/codes_not_read.json
fi
[ -n "$CODES" ] && [ -f $CODES ] || { echo "FAILED $SHEET: CODES (the series' code list) or FITS_FROM needed"; exit 1; }
WH=$($PY -c "import cv2; a = cv2.imread('$PAGE', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
echo '{}' > $D/stats.json
if [ ! -f $G/zgraph.json ]; then
  stage raw-graph
  cp $PAGE $G/clean.png
  echo '[]' > $G/fits3.json
  $PY $PW_CODE/vectorize.py $SERIES $SHEET 0,0 $G > $D/raw-graph.log 2>&1
  BLOCKS=1 $PY $PW_CODE/zonegraph.py $SERIES $G >> $D/raw-graph.log 2>&1
fi
if [ ! -f $D/noblock.png ]; then
  stage block-outlines
  $PY $PW_CODE/blocklines.py $SERIES $SHEET $G $D > $D/blocklines.log 2>&1
fi
if [ ! -f $D/walls.json ]; then
  stage walls
  WALLS_OUT=$D/walls.json $PY $PW_CODE/sidealign.py $D > $D/walls.log 2>&1
fi
if [ ! -f $DC/clean.png ]; then
  stage fit-erase${FITS_FROM:+-resumed}
  cp $D/textboxes_gpt-6.1-terra.json $DC/
  echo '{}' > $DC/stats.json
  [ -n "$FITS_FROM" ] && cp $FITS_FROM/fits_raw.pkl $DC/fits_raw.pkl && export RESUME=1
  CODES=$CODES CODES_ONLY=1 FITWORD_WHITE_UNIT=grow FITWORD_WALL_INSIDE=ignore SHEETFILES=$D PAGE=$D/noblock.png \
    NPROC=${NPROC:-4} $PY -u $PW_CODE/fitboxes.py $SERIES $SHEET 0,0,$WH $DC > $DC/fit.log 2>&1
  [ -f $DC/clean.png ] || { echo "FAILED $SHEET fit-erase: $(grep -m1 -E 'Error|Traceback' $DC/fit.log)"; exit 1; }
fi
stage done
