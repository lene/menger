#!/usr/bin/env bash
# Tests scripts/runtime-budget.sh with small CPU-burning commands instead of sbt runs.
# Run: scripts/test-runtime-budget.sh
set -uo pipefail
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runtime-budget.sh"
FAILURES=0
burn() { echo "python3 -c 'import time; e=time.process_time()+$1
while time.process_time()<e: pass'"; }

export RUNTIME_BUDGET_ROUNDS=3
RUNTIME_BUDGET_REFERENCE_CMD="$(burn 0.2)"
RUNTIME_BUDGET_SUBJECT_CMD="$(burn 0.6)"   # ratio ~3
export RUNTIME_BUDGET_REFERENCE_CMD RUNTIME_BUDGET_SUBJECT_CMD

if RUNTIME_BUDGET_MAX_RATIO=6 "$SCRIPT" > /dev/null; then echo "PASS: within the limit passes"
else echo "FAIL: within the limit passes"; FAILURES=$((FAILURES + 1)); fi

if RUNTIME_BUDGET_MAX_RATIO=1.5 "$SCRIPT" > /dev/null; then
    echo "FAIL: over the limit fails"; FAILURES=$((FAILURES + 1))
else echo "PASS: over the limit fails"; fi

out=$(RUNTIME_BUDGET_MAX_RATIO=6 "$SCRIPT")
if [ "$(grep -c '^round ' <<< "$out")" -eq 3 ] && grep -q '^median ratio' <<< "$out"; then
    echo "PASS: reports each round and the median"
else echo "FAIL: reports each round and the median"; FAILURES=$((FAILURES + 1)); fi

if (( FAILURES > 0 )); then echo "$FAILURES failure(s)"; exit 1; fi
echo "all passed"
