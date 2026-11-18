#!/usr/bin/env bash
# Resume paused (SIGSTOP) jobs one at a time as the machine frees up: every 30 s, when fewer than MAX of this
# worktree's python processes are running, the next job of the list gets SIGCONT (a job is a comma-separated list of
# pids: a process and its pool workers).  Prints each resume; exits when every job is resumed.
#
#   scripts/all_years/throttle.sh MAX JOB...
max=$1
shift
here=$(cd "$(dirname "$0")/../.." && pwd)
for job in "$@"; do
  while true; do
    running=$(ps -Ao stat,command | awk -v h="$here/.venv/bin/python" '$1 !~ /^T/ && index($0, h) {n++} END {print n+0}')
    [ "$running" -lt "$max" ] && break
    sleep 30
  done
  kill -CONT ${job//,/ } 2>/dev/null
  echo "$(date +%H:%M:%S) resumed $job (running before: $running)"
  sleep 60
done
