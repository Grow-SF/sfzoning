#!/bin/zsh
# Rebuild every prob-warp output from the pipeline's sheets, in the order the 2026-11-08/09 runs used:
#   1. the earlier "p1" correction, every fitted sheet of each series (corr_series.sh -> corr.sh)
#   2. the mirror data folder with the p1 corrections (build_mirror_ay.py)
#   3. the probabilistic fit, prob6, every sheet with p1's inputs (prob_all.sh -> prob.sh)
#   4. the choice between p1 and prob6, written into a copy of the mirror, data_ay2 (mirror_prob.py)
#   5. 2000 p09 and p10 on the "plus" fit (apply_plus.sh)
#   6. the corrected warps of every chosen sheet (warp_prob.sh -> warp1.sh -> warpK.py)
# Settings (environment; defaults in brackets):
#   PW_TPY       the torch environment's python (required; README)
#   PW_PIPELINE  the pipeline folder [this repo's]        PW_WORK    its work folder [PW_PIPELINE/data/work]
#   PW_RUN       outputs [PW_PIPELINE/data/prob-warp/run]  PW_MIRROR, PW_MIRROR2  the two mirrors [beside PW_RUN]
#   PW_CHAIN     the years chain's text-erased pages, read by prob6 [PW_PIPELINE/data/prob-warp/chain]
#   PW_ERASED_2000  2000 p06/p07's letter-erased prob6 fits [PW_RUN/../erased-2000]
# usage: rebuild.sh [SERIES ...]      (default: the 13 series of the original run)
set -e
A=${0:A:h}/allyears
P=${PW_PIPELINE:-${0:A:h:h:h}}
export PW_CODE=$A PW_PIPELINE=$P PW_RUN=${PW_RUN:-$P/data/prob-warp/run} PW_WORK=${PW_WORK:-$P/data/work}
: ${PW_TPY:?set PW_TPY to the torch environment python (README)}
SERIES=(${@:-1921 1929 1935 1948 1960 1964 1968 1972 1975 1980 1980-1990-zoning 1990-1996-zoning 2000-digital-zoning})
mkdir -p $PW_RUN
echo "1. p1 correction: ${#SERIES} series"
for s in $SERIES; do zsh $A/corr_series.sh $s; done
echo "2. mirror with p1"
for s in $SERIES; do
  [ -d $PW_RUN/$s ] && (cd $P && env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $A/build_mirror_ay.py $s | tail -1)
done
echo "3. prob6"
zsh $A/prob_all.sh
echo "4. choice"
(cd $P && .venv/bin/python $A/mirror_prob.py)
echo "5. 2000 p09, p10 plus"
zsh $A/apply_plus.sh
echo "6. warps"
zsh $A/warp_prob.sh
