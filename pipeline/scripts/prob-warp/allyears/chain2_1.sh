#!/bin/zsh
# one sheet's sheetAY2.sh, logged to chain2/SERIES/SHEET.log; prints its last line.  A lock folder
# (chain2/SERIES/SHEET.lock) lets several queues share the job list: a sheet already taken is skipped.
# usage: chain2_1.sh SERIES SHEET
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
mkdir -p $A/chain2/$1
mkdir $A/chain2/$1/$2.lock 2>/dev/null || exit 0
zsh $A/sheetAY2.sh $1 $2 > $A/chain2/$1/$2.log 2>&1
tail -n 1 $A/chain2/$1/$2.log
