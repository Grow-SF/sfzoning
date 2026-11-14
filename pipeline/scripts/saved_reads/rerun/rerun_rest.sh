#!/bin/zsh
# PAID.  The re-check and the leader reads of rerun.sh, started beside its leader map instead of after it (they don't
# need it: the re-check asks about the published code-check cases).  Same scripts, settings and caps as rerun.sh;
# recorded into their own folders (codes2000-reread-*-fixed, leaders-sol-fixed-*), so two runs never write one file.
# usage: KEY=<the API key file> rerun_rest.sh [OUT]
set -e
: ${KEY:?set KEY to the API key file, e.g. pipeline/.env of the main checkout; its contents are never printed}
R=${0:A:h}
PKG=${R:h:h}
P=${SR_PIPELINE:-${PKG:h}}
OUT=${1:-$P/data/saved-reads/calls}
W=$P/data/saved-reads/work/rerun-rest
mkdir -p $W/codecheck $W/leaders
cd $P
git show prob-warp:pipeline/scripts/prob-warp/codes2000/answers/codecheck_cases_published.json > $W/codecheck/cases.json
SRC="saved-reads@$(git rev-parse --short saved-reads):pipeline/scripts/saved_reads/rerun"
LIVE=(uv run --env-file $KEY --project $P env -u SFZMAPS_DATA_DIR -u SFZ_CORR PYTHONPATH=$PKG python -W ignore -m saved_reads run --live)
echo "re-check and leader reads $(date +%H:%M:%S)"
SR_SOURCE=$SRC/reread.py THREADS=12 SFZMAPS_MAX_USD=20 $LIVE --out $OUT/codes2000-reread-terra-fixed --scratch $W $R/reread.py gpt-6.1-terra high all > $W/reread_terra.log 2>&1 &
SR_SOURCE=$SRC/reread.py THREADS=12 SFZMAPS_MAX_USD=10 $LIVE --out $OUT/codes2000-reread-sol-fixed --scratch $W $R/reread.py gpt-6.1-sol high all > $W/reread_sol.log 2>&1 &
for e in medium low; do
  for n in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
    SR_SOURCE=$SRC/leaders.py SFZMAPS_MAX_USD=1.5 $LIVE --out $OUT/leaders-sol-fixed-$e-p$n --scratch $W $R/leaders.py sheet 2000-digital-zoning-p$n $e \
      > $W/leaders-p$n-$e.log 2>&1 &
  done
done
wait
echo "  re-check terra: $(tail -n 1 $W/reread_terra.log)"
echo "  re-check sol: $(tail -n 1 $W/reread_sol.log)"
grep -h 'spent' $W/leaders-p*-*.log | sed 's/^/  leader read: /'
echo "done $(date +%H:%M:%S)"
