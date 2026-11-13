#!/bin/zsh
# probabilistic correction on p06 and p07 in parallel: runprob.sh TAG [ENV=VAL ...]  -> k5/probfit_<TAG>_NN.log
S=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad
TAG=$1; shift
cd $S
for NN in 06 07; do
  (env "$@" THREADS=5 $S/torchenv/bin/python $S/probfit.py $NN $TAG 800 > $S/k5/probfit_${TAG}_$NN.log 2>&1) &
done
wait
cat $S/k5/probfit_${TAG}_06.log $S/k5/probfit_${TAG}_07.log | grep -v "^control points"
