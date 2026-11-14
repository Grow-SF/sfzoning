#!/bin/zsh
# PAID.  Ask again, with the marks red as the prompts say, every read whose mark reached the model blue: the 2000 code
# reads (leader map and re-check, Terra and Sol, high effort) and Sol's leader reads (medium and low).  Same prompts,
# models, efforts, crops and cases as the original runs; only the mark's colour differs (see leaders2.py's header).
# Each run is recorded (saved_reads run --live) and capped on the meter (SFZMAPS_MAX_USD, Sol prices for every call):
#   leader map  Terra $25, Sol $25          re-check  Terra $20, Sol $10          leader reads  $1.50 per sheet and effort
# The re-check asks about the code-check page's cases as published, so case ids stay the ones the owner's answers name.
# Run check_marks.py first.  usage: KEY=<the API key file> rerun.sh [OUT]
set -e
: ${KEY:?set KEY to the API key file, e.g. pipeline/.env of the main checkout; its contents are never printed}
R=${0:A:h}                                                    # this folder (saved_reads/rerun)
PKG=${R:h:h}                                                  # the folder holding the saved_reads package
P=${SR_PIPELINE:-${PKG:h}}                                    # the pipeline folder
OUT=${1:-$P/data/saved-reads/calls-live}             # the paid runs' own records; record_all.sh replays them into calls/
W=$P/data/saved-reads/work/rerun
mkdir -p $W/leaders2 $W/codecheck $W/leaders
cd $P
cp ${PKG:h}/inputs/code-check-2000/codecheck_cases_published.json $W/codecheck/cases.json
SRC="$(git -C $R rev-parse --short HEAD):pipeline/scripts/saved_reads/rerun"      # the commit these scripts are at
LIVE=(uv run --env-file $KEY --project $P env -u SFZMAPS_DATA_DIR -u SFZ_CORR PYTHONPATH=$PKG python -W ignore -m saved_reads run --live)
spent() { grep -h 'spent' $1 | tail -n 1; }

echo "leader map $(date +%H:%M:%S)"
SR_SOURCE=$SRC/leaders2.py THREADS=12 SFZMAPS_MAX_USD=25 $LIVE --out $OUT/codes2000-terra-fixed --scratch $W $R/leaders2.py map gpt-6.1-terra $R/leaders2_v2.txt high > $W/map_terra.log 2>&1 &
SR_SOURCE=$SRC/leaders2.py THREADS=12 SFZMAPS_MAX_USD=25 $LIVE --out $OUT/codes2000-sol-fixed --scratch $W $R/leaders2.py map gpt-6.1-sol $R/leaders2_v2.txt high > $W/map_sol.log 2>&1 &
wait
echo "  terra: $(spent $W/map_terra.log); sol: $(spent $W/map_sol.log)"

echo "re-check $(date +%H:%M:%S)"
SR_SOURCE=$SRC/reread.py THREADS=12 SFZMAPS_MAX_USD=20 $LIVE --out $OUT/codes2000-terra-fixed --scratch $W $R/reread.py gpt-6.1-terra high all > $W/reread_terra.log 2>&1 &
SR_SOURCE=$SRC/reread.py THREADS=12 SFZMAPS_MAX_USD=10 $LIVE --out $OUT/codes2000-sol-fixed --scratch $W $R/reread.py gpt-6.1-sol high all > $W/reread_sol.log 2>&1 &
wait
echo "  terra: $(tail -n 1 $W/reread_terra.log); sol: $(tail -n 1 $W/reread_sol.log)"

echo "leader reads $(date +%H:%M:%S)"
for e in medium low; do
  for n in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
    SR_SOURCE=$SRC/leaders.py SFZMAPS_MAX_USD=1.5 $LIVE --out $OUT/leaders-sol-fixed-$e-p$n --scratch $W $R/leaders.py sheet 2000-digital-zoning-p$n $e \
      > $W/leaders-p$n-$e.log 2>&1 &
  done
  wait
  echo "  $e:"; grep -h 'spent' $W/leaders-p*-$e.log | sed 's/^/    /'
done
echo "done $(date +%H:%M:%S)"
