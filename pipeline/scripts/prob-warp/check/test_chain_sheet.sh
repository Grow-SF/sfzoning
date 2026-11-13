#!/bin/zsh
# Test the years chain + prob6 on one sheet whose letter-erased page (clean.png) and prob6 field both survive:
#   1. the p1 correction (corr.sh)
#   2. its mirror entry with the p1 correction (build_mirror_ay.py)
#   3. the chain's raw graph and block-outline stages only (as sheetAY.sh): noblock.png
#   4. prob6 (prob.sh), reading the rebuilt noblock.png and the surviving clean.png
# then compare the sheet's outputs with the rescued copy.  usage: test_chain_sheet.sh SERIES SHEET
set -e
source ${0:A:h}/rebuild_env.sh
SERIES=$1 SHEET=$2
CH=$PW_DATA/rebuild-2026-11-13/chain
export PW_CHAIN=$CH
mkdir -p $PW_RUN $CH/$SERIES/$SHEET/raw $CH/$SERIES/${SHEET}_c1
echo "1. p1 $(date +%H:%M:%S)"
sh $PW_CODE/corr.sh $SERIES $SHEET
echo "2. mirror $(date +%H:%M:%S)"
cd $PW_PIPELINE
[ -d $PW_MIRROR/work/$SERIES ] && [ ! -L $PW_MIRROR/work/$SERIES ] || \
  env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $PW_CODE/build_mirror_ay.py $SERIES --prepare
env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $PW_CODE/build_mirror_ay.py $SERIES $SHEET
echo "3. chain pages $(date +%H:%M:%S)"
D=$CH/$SERIES/$SHEET; G=$D/raw
(
  export SFZ_CORR=1 PYTHONPATH=$PW_CODE/../corrpatch SFZMAPS_DATA_DIR=$PW_MIRROR SERIES SHEET
  cp $PW_MIRROR/work/$SERIES/$SHEET/page.png $G/clean.png
  echo '[]' > $G/fits3.json
  .venv/bin/python $PW_CODE/vectorize.py $SERIES $SHEET 0,0 $G > $D/chain.log 2>&1
  BLOCKS=1 .venv/bin/python $PW_CODE/zonegraph.py $SERIES $G >> $D/chain.log 2>&1
  .venv/bin/python $PW_CODE/blocklines.py $SERIES $SHEET $G $D >> $D/chain.log 2>&1
)
cp -p $PW_RESCUED/allyears-chain/$SERIES/${SHEET}_c1/clean.png $CH/$SERIES/${SHEET}_c1/clean.png
echo "4. prob6 $(date +%H:%M:%S)"
THREADS=3 sh $PW_CODE/prob.sh $SERIES $SHEET
echo "compare $(date +%H:%M:%S)"
.venv/bin/python ${0:A:h}/compare_rebuild.py $PW_RESCUED/allyears-run/$SERIES/$SHEET $PW_RUN/$SERIES/$SHEET
