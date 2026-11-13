#!/bin/zsh
# 2000 p06 and p07's prob6 fitted with every fitted letter taken out of the page (mirror_prob.py's "all text
# erased" fits, read from PW_ERASED_2000), rebuilt the way runs k4 and k5 made them on 2026-11-08/09:
#   1. k4 (erased2000/k4sheet.sh): every Terra box's text fitted and erased, not only the codes (fitboxes.py without
#      CODES_ONLY, against 2000's codes_final.json) on run k3's page with the block outlines deleted.  Run k3's
#      per-sheet pages are the years chain's 2000 pages (commit 68ae264: the chain reproduced run k3's 173 files
#      exactly), so the rebuild's chain pages are read.  -> $K/k4/pNN_c1/clean.png, checked against k4's surviving
#      checkpoint (compare_fits.py)
#   2. k5 (erased2000/regprep3_wmap.py, as it ran at 07:06 UTC 11-09 under the name regprep3.py): block-line centre
#      weights with that text taken out -> $K/regw_NN.npz
#   3. probfit (PERPOINT WMAP SHAPE, as erased2000/runprob.sh ran it) -> $K/confit_NN_prob6.npz; and p06 again with
#      LOWMEM, the run that survives (confit_06_prob6lm.npz, 08:22 UTC), as the check
# Needs rebuild_one.sh on both sheets first (p1's reg2 and probin, the mirror, the chain pages) and codes2000.sh.
# usage: erased2000.sh
set -e
CK=${0:A:h}
source $CK/rebuild_env.sh
K=$PW_DATA/rebuild-2026-11-13/erased-2000
CODES=$PW_DATA/rebuild-2026-11-13/codes2000/codes_final.json
SER=2000-digital-zoning
[ -f $CODES ] || zsh $CK/codes2000.sh
cd $PW_PIPELINE
k4() {
  local NN=$1 SHEET=$SER-p$1 D=$PW_CHAIN/$SER/$SER-p$1 DC=$K/k4/p$1_c1
  mkdir -p $DC
  if [ ! -f $DC/clean.png ]; then
    cp $D/textboxes_gpt-6.1-terra.json $DC/
    echo '{}' > $DC/stats.json
    SFZ_CORR=1 PYTHONPATH=$PW_CODE/../corrpatch SFZMAPS_DATA_DIR=$PW_MIRROR CODES=$CODES FITWORD_WHITE_UNIT=grow \
      FITWORD_WALL_INSIDE=ignore SHEETFILES=$D PAGE=$D/noblock.png NPROC=${NPROC:-4} \
      .venv/bin/python -u $PW_CODE/fitboxes.py $SER $SHEET 0,0,3300,5100 $DC > $DC/fit.log 2>&1
  fi
  echo "p$NN k4: $(.venv/bin/python $CK/compare_fits.py $PW_RESCUED/scratch-root-2000/k4/p$1_c1/fits_raw.pkl $DC/fits_raw.pkl | head -1)"
}
echo "1. k4 $(date +%H:%M:%S)"
k4 06 & k4 07 & wait
echo "2. k5 weights $(date +%H:%M:%S)"
for NN in 06 07; do
  mkdir -p $K/stepview/sheetsK/p$NN
  cp $PW_CHAIN/$SER/$SER-p$NN/noblock.png $K/stepview/sheetsK/p$NN/noblock.png
  cp $PW_RUN/$SER/$SER-p$NN/reg2_$SER-p$NN.npz $K/reg2_$NN.npz
  cp $PW_RUN/$SER/$SER-p$NN/probin_$SER-p$NN.npz $K/probin_$NN.npz
  env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python -W ignore $CK/run_as_ran.py --scratch $K \
    $CK/../erased2000/regprep3_wmap.py $NN | cut -c1-200
done
echo "3. probfit $(date +%H:%M:%S)"
( PROBDIR=$K/ PERPOINT=1 WMAP=1 SHAPE=1 THREADS=5 $PW_TPY $CK/../probfit_before_plus.py 06 prob6 800 > $K/probfit_prob6_06.log 2>&1 ) &
( PROBDIR=$K/ PERPOINT=1 WMAP=1 SHAPE=1 THREADS=5 $PW_TPY $CK/../probfit_before_plus.py 07 prob6 800 > $K/probfit_prob6_07.log 2>&1 ) &
wait
PROBDIR=$K/ PERPOINT=1 WMAP=1 SHAPE=1 LOWMEM=1 THREADS=5 $PW_TPY $CK/../probfit_before_plus.py 06 prob6lm 800 > $K/probfit_prob6lm_06.log 2>&1
echo "check $(date +%H:%M:%S)"
.venv/bin/python $CK/compare_rebuild.py $PW_RESCUED/scratch-root-2000 $K 2>&1 | grep -E "confit_06_prob6lm|==" || true
.venv/bin/python - $K <<'EOF'
import sys, numpy as np
K = sys.argv[1]
a = np.load(K + "/confit_06_prob6lm.npz"); b = np.load(K + "/confit_06_prob6.npz")
print("p06 LOWMEM vs not: largest field difference", float(np.abs(a["Gc"] - b["Gc"]).max()), "px")
EOF
