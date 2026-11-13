#!/bin/zsh
# Rebuild one sheet with the PR #58 scripts: the p1 correction chain, then prob6.  usage: rebuild_one.sh SERIES SHEET
source ${0:A:h}/rebuild_env.sh
mkdir -p $PW_RUN
sh $PW_CODE/corr.sh $1 $2 && THREADS=3 sh $PW_CODE/prob.sh $1 $2
