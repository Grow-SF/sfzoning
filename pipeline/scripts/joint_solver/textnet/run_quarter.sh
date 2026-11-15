#!/bin/zsh
# train the detector without one quarter, predict over the page, score the held-out quarter.  usage: run_quarter.sh Q [ITERS]
T=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
{
  .venv/bin/python -W ignore $T/detector.py train $1 ${2:-8000}
  .venv/bin/python -W ignore $T/detector.py predict $1
  .venv/bin/python -W ignore $T/evaluate_detector.py $1
} > $T/run_$1.log 2>&1
echo "exit $?"
