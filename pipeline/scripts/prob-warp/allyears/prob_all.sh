#!/bin/zsh
# prob.sh for every sheet of every zoning series that has the old chain's inputs, PROB_P at a time (default 3,
# THREADS 3 each).  Log: run/prob_all.log.  usage: prob_all.sh
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
JOBS=()
for s in $A/run/*(/); do
  for d in $s/*(/); do
    [ -f $d/reg2_${d:t}.npz ] && [ -f $d/dm_sides_${d:t}.npz ] && JOBS+=("${s:t} ${d:t}")
  done
done
echo "START ${#JOBS} sheets $(date +%H:%M:%S)"
print -l $JOBS | xargs -P ${PROB_P:-3} -L 1 sh -c 'THREADS=3 sh '$A'/prob.sh $0 $1 | grep -E "^(FAILED|SKIP)|done"'
echo "ALL done $(date +%H:%M:%S)"
sysctl vm.swapusage
