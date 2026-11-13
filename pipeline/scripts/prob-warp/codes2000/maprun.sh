#!/bin/bash
# Terra-high v2 and Sol-high v2 leader reads on every 2000 use code, both models in parallel.
# Caps are on the meter, which prices every call at Sol's rates (Terra really costs half).
SP=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
ENV=/Users/sbuss/workspace/sfzoning/pipeline/.env
THREADS=12 SFZMAPS_MAX_USD=40 uv run --env-file $ENV python $SP/leaders2.py map gpt-6.1-terra $SP/leaders2/v2.txt high > $SP/leaders3_terra.log 2>&1 &
THREADS=12 SFZMAPS_MAX_USD=60 uv run --env-file $ENV python $SP/leaders2.py map gpt-6.1-sol $SP/leaders2/v2.txt high > $SP/leaders3_sol.log 2>&1 &
wait
tail -n 2 $SP/leaders3_terra.log $SP/leaders3_sol.log
