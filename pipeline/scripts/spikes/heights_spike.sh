#!/bin/zsh
# Heights spike, from pipeline/: the read step in the docTR overlay (S2 model C).  Usage: scripts/spikes/heights_spike.sh read
cd "${0:A:h}/../.." || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
exec uv run --with python-doctr --with torch==2.15.0 python scripts/spikes/heights_spike.py "$@"
