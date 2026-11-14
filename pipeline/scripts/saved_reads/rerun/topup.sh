#!/bin/zsh
# PAID.  Finish a re-check that stopped at its cap: the cases still unanswered are asked (answered ones come from the
# cache, free), recorded into the same folder as rerun_rest.sh's run.  usage: KEY=<the API key file> topup.sh MODEL CAP
set -e
: ${KEY:?set KEY to the API key file, e.g. pipeline/.env of the main checkout; its contents are never printed}
M=${1:?model, e.g. gpt-6.1-sol}; CAP=${2:?the most this run may spend, metered}
R=${0:A:h}
PKG=${R:h:h}
P=${SR_PIPELINE:-${PKG:h}}
W=$P/data/saved-reads/work/rerun-rest
cd $P
SR_SOURCE="saved-reads@$(git rev-parse --short saved-reads):pipeline/scripts/saved_reads/rerun/reread.py" THREADS=12 SFZMAPS_MAX_USD=$CAP \
  uv run --env-file $KEY --project $P env -u SFZMAPS_DATA_DIR -u SFZ_CORR PYTHONPATH=$PKG python -W ignore -m saved_reads run --live \
  --out $P/data/saved-reads/calls-live/codes2000-reread-${M#gpt-6.1-}-fixed --scratch $W $R/reread.py $M high all > $W/topup_${M#gpt-6.1-}.log 2>&1
tail -n 1 $W/topup_${M#gpt-6.1-}.log
