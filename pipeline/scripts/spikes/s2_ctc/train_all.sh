#!/bin/zsh
# Spike S2: the GPU trainings one after another (B, then C, then A, then the direction classifier D, then the
# reproducibility check), each its own process and its own exclusive GPU turn, so a waiting process can take the GPU
# between them.  Usage (from pipeline/): scripts/spikes/s2_ctc/train_all.sh [readers...]
cd "${0:A:h}/../../.." || exit 1
LOGS=data/solver/experiments/spike-s2/logs
mkdir -p $LOGS
for r in ${@:-B C A D repro}; do
  uv run --with python-doctr --with torch==2.15.0 python scripts/spikes/s2_ctc/train.py $r > $LOGS/train_$r.log 2>&1 \
    || { echo "train $r failed"; exit 1; }
done
