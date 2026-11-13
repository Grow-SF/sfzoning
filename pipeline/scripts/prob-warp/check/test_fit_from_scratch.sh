#!/bin/zsh
# The code fit from scratch on one sheet (chain_pages.sh with CODES, no checkpoint), in its own chain folder
# (rebuild-2026-11-13/chain_scratchfit), then compared with the chain's surviving checkpoint (compare_fits.py) and
# its clean.png with the rebuild's resumed one and the rescued one where they exist.  Needs the sheet's mirror entry
# and replayed Terra boxes (rebuild_one.sh makes both).  usage: test_fit_from_scratch.sh SERIES SHEET CODES.json
set -e
CK=${0:A:h}
source $CK/rebuild_env.sh
SERIES=$1 SHEET=$2
CH0=$PW_CHAIN
export PW_CHAIN=$PW_DATA/rebuild-2026-11-13/chain_scratchfit
mkdir -p $PW_CHAIN/$SERIES/$SHEET
cp $CH0/$SERIES/$SHEET/textboxes_gpt-6.1-terra.json $CH0/$SERIES/$SHEET/frame_bottom.json $PW_CHAIN/$SERIES/$SHEET/
unset FITS_FROM
CODES=${3:A} zsh $CK/chain_pages.sh $SERIES $SHEET
$PW_PIPELINE/.venv/bin/python $CK/compare_fits.py $PW_RESCUED/allyears-chain/$SERIES/${SHEET}_c1/fits_raw.pkl \
  $PW_CHAIN/$SERIES/${SHEET}_c1/fits_raw.pkl
for ref in $CH0 $PW_RESCUED/allyears-chain; do
  if [ -f $ref/$SERIES/${SHEET}_c1/clean.png ]; then
    cmp -s $ref/$SERIES/${SHEET}_c1/clean.png $PW_CHAIN/$SERIES/${SHEET}_c1/clean.png \
      && echo "clean.png identical to $ref" || echo "clean.png DIFFERS from $ref"
  fi
done
