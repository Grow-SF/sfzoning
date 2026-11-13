#!/bin/zsh
# one sheet's chain, logged to chain/SERIES/SHEET.log; prints its last line; exits with the chain's status.
# usage: chain1.sh SERIES SHEET
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
mkdir -p $A/chain/$1
zsh $A/sheetAY.sh $1 $2 > $A/chain/$1/$2.log 2>&1
rc=$?
tail -1 $A/chain/$1/$2.log
exit $rc
