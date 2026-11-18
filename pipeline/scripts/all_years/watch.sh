#!/usr/bin/env bash
# Print each step log's last line once when it shows the step ended (a summary line) or failed: for watching runs.
shopt -s nullglob
cd "$(dirname "$0")/../../data/solver/experiments/all-years/logs" || exit 1
seen=""
while true; do
  for f in transfer-models.log *-walls.log *-solve.log *-page.log *-prints.log *-labels.log; do
    l=$(tail -n 1 "$f" | cut -c1-160)
    case "$l" in
      *Error*|*error*|*Traceback*|*seconds*|*"sheets laid"*|*prints:*|*'}'*|*joined*|*"solved in"*)
        key="$f:$l"
        case "$seen" in *"$key"*) ;; *) echo "$f: $l"; seen="$seen|$key" ;; esac ;;
    esac
  done
  sleep 10
done
