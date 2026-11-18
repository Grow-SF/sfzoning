#!/usr/bin/env bash
# Extract T0..T5 briefs (task text + the plan's Global constraints) into this workspace.
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd)
P="$W/../../../docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md"
for n in 0 1 2 3 4 5; do
  out="$W/task-$n-brief.md"
  {
    echo "# Brief: task T$n of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md"
    echo
    awk -v n="$n" '/^### T[0-9]/{intask=($0 ~ ("^### T" n ":"))} /^## /{intask=0} intask{print}' "$P"
    echo
    awk '/^## Global constraints/{g=1} /^## Tasks/{g=0} g{print}' "$P"
  } > "$out"
  echo "$out: $(wc -l < "$out" | tr -d ' ') lines"
done
