#!/usr/bin/env bash
# Extract Task 1..8 briefs, each followed by the plan's Global Constraints and Review Focus.
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd)
P="$W/../../../docs/superpowers/plans/2026-11-17-text-detector-text-vs-line.md"
for n in 1 2 3 4 5 6 7 8; do
  out="$W/task-$n-brief.md"
  {
    awk -v n="$n" '/^```/{f=!f} !f && /^### Task [0-9]+/{t=($0 ~ ("^### Task " n ":"))} t{print}' "$P"
    echo
    awk '/^## Global Constraints/{g=1} /^---$/{g=0} g{print}' "$P"
  } > "$out"
  echo "$out: $(wc -l < "$out" | tr -d ' ') lines"
done
