#!/bin/zsh
# Spike S2 fix round 1 (review I3): training-variance replicates of C and B, the same recipe, data and steps as seed 0,
# seeds 1 and 2 (models/<reader>s<seed>.pt), each its own process and its own exclusive GPU turn; then their CPU frames.
# Usage (from pipeline/): scripts/spikes/s2_ctc/train_seeds.sh
set -e
cd "${0:A:h}/../../.."
LOGS=data/solver/experiments/spike-s2/logs
DOCTR=(uv run --with python-doctr --with torch==2.15.0 python)
mkdir -p $LOGS
for r in C B; do
  for s in 1 2; do
    $DOCTR scripts/spikes/s2_ctc/train.py $r --seed $s > $LOGS/train_${r}s$s.log 2>&1
    OMP_NUM_THREADS=1 $DOCTR scripts/spikes/s2_ctc/infer.py frames ${r}s$s > $LOGS/infer_${r}s$s.log 2>&1 &
  done
done
wait
echo "seeds done"
