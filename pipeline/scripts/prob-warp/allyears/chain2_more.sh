#!/bin/zsh
# a second queue over chain2_all.sh's job list, from the other end (sheets already taken are skipped by their lock).
# usage: chain2_more.sh [P]
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
echo "START more $(date +%H:%M:%S)"
awk '{l[NR]=$0} END {for (i = NR; i > 0; i--) print l[i]}' $A/run/chain2_jobs.txt | xargs -P ${1:-2} -L 1 zsh $A/chain2_1.sh
echo "ALL done more $(date +%H:%M:%S)"
