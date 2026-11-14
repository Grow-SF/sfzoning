#!/bin/zsh
# The whole sheet warp, every year (README "Running it"):
#   1. fit every sheet: its first correction, its stand-in data folder entry, its probabilistic correction
#      (fit_every_year.sh -> fit_one_year.sh -> fit_one_sheet.sh, JOBS sheets at a time; finished fits are kept)
#   2. choose each sheet's correction, into the stand-in data folder the warps are made in (choose_correction.py)
#   3. 2000 p09 and p10: the correction with extra evidence (extra_evidence_2000.sh)
#   4. warp every corrected sheet: warped.tif, warped_paper.tif and footprints (warp_corrected_sheets.sh)
# usage: [JOBS=3] warp_every_year.sh [YEAR ...]      (default: the 13 years the warp corrects)
set -e
CK=${0:A:h}
source $CK/settings.sh
zsh $CK/make_torch_environment.sh
echo "1. fits $(date +%H:%M:%S)"
JOBS=${JOBS:-3} zsh $CK/fit_every_year.sh $@
echo "2. choice $(date +%H:%M:%S)"
(cd $WARP_PIPELINE && .venv/bin/python $WARP_STEPS/choose_correction.py)
echo "3. extra evidence, 2000 p09 and p10 $(date +%H:%M:%S)"
zsh $WARP_STEPS/extra_evidence_2000.sh
echo "4. warps $(date +%H:%M:%S)"
zsh $WARP_STEPS/warp_corrected_sheets.sh
echo "done $(date +%H:%M:%S)"
