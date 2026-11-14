#!/bin/zsh
# Fit one sheet:
#   1. its first correction (first_correction.sh)
#   2. its stand-in data folder entry, with that correction (stand_in_data_folder.py)
#   3. its probabilistic correction (probabilistic_correction.sh), on the page's ink as printed
# The 2000 sheets are fitted as the approved run fitted them (AS_APPROVED_2000): each sheet measured on its own area
# (measured_area_2000.py, written once to WARP_AREA_2000) and the scan's scale taken as 3.99 ft per px.
# usage: fit_one_sheet.sh YEAR SHEET
set -e
CK=${0:A:h}
source $CK/settings.sh
zsh $CK/make_torch_environment.sh
YEAR=$1 SHEET=$2
mkdir -p $WARP_FITS
if [[ $YEAR == 2000-digital-zoning ]]; then
  export AS_APPROVED_2000=1
  [ -f $WARP_AREA_2000 ] || $WARP_PIPELINE/.venv/bin/python $WARP_STEPS/measured_area_2000.py $WARP_AREA_2000
fi
echo "1. first correction $YEAR $SHEET $(date +%H:%M:%S)"
sh $WARP_STEPS/first_correction.sh $YEAR $SHEET
echo "2. stand-in data folder $YEAR $SHEET $(date +%H:%M:%S)"
( cd $WARP_PIPELINE
  [ -d $WARP_STAND_IN/work/$YEAR ] && [ ! -L $WARP_STAND_IN/work/$YEAR ] || \
    env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $WARP_STEPS/stand_in_data_folder.py $YEAR --prepare
  env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $WARP_STEPS/stand_in_data_folder.py $YEAR $SHEET )
echo "3. probabilistic correction $YEAR $SHEET $(date +%H:%M:%S)"
THREADS=3 sh $WARP_STEPS/probabilistic_correction.sh $YEAR $SHEET
echo "done $YEAR $SHEET $(date +%H:%M:%S)"
