#!/bin/zsh
# Spike S2 end to end, from pipeline/: the crops, the synthetic crops, the four GPU trainings one after another (each
# its own exclusive turn), the CPU frames (one thread), the determinism and speed check, the scores and results.md.
#   scripts/spikes/s2_ctc/run_all.sh
# Inputs first (copied from the scratch folder, which macOS cleans after 3 days): data/solver/experiments/spike-s2/
# inputs/{full_fix1held,full_fix2held}.records, prior3.json, labs.pkl, font_real.npz.
set -e
cd "${0:A:h}/../../.."
OUT=data/solver/experiments/spike-s2
DOCTR=(uv run --with python-doctr --with torch==2.15.0 python)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $OUT/logs
uv run python scripts/spikes/s2_ctc/crops.py > $OUT/logs/crops.log 2>&1
uv run pytest -q -n 0 scripts/spikes/s2_ctc/test_s2_ctc.py > $OUT/logs/test.log 2>&1
uv run python scripts/spikes/s2_ctc/synth.py > $OUT/logs/synth.log 2>&1
scripts/spikes/s2_ctc/train_all.sh B C A D repro
$DOCTR scripts/spikes/s2_ctc/infer.py frames B C A D > $OUT/logs/infer.log 2>&1
$DOCTR scripts/spikes/s2_ctc/infer.py check C > $OUT/logs/check_C.log 2>&1
$DOCTR scripts/spikes/s2_ctc/infer.py check B > $OUT/logs/check_B.log 2>&1
$DOCTR scripts/spikes/s2_ctc/score.py > $OUT/logs/score.log 2>&1
uv run python scripts/spikes/s2_ctc/report.py > $OUT/logs/report.log 2>&1
echo "spike S2 done"
