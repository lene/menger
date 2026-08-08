#!/bin/sh
# Suite: unit (Sprint 36 B1). Compile (fast-fail before the slower test run) + full test
# suite, extracted verbatim from the pre-push hook's former Phase 2 (compile) + Phase 3
# (sbt test). Adds failed-test-name extraction for qa-runner.sh's SUITE-line contract.
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_SCALA:-0}" -eq 0 ] && [ "${HAS_NATIVE:-0}" -eq 0 ]; then
  suite_skip unit "no Scala or native changes"
  exit 0
fi

echo "=== Compiling ==="
if ! sbt compile; then
  suite_fail unit 1 "compile"
  exit 1
fi

LOG=$(mktemp)
echo "=== Tests ==="
{ __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a sbt test --warn 2>&1; echo "SBT_TEST_RC=$?"; } | tee "$LOG"

if grep -q '^SBT_TEST_RC=0$' "$LOG"; then
  echo "sbt test: PASSED"
  suite_pass unit
  rm -f "$LOG"
  exit 0
fi

echo "sbt test: FAILED"
FAILED_LINES=$(grep '\*\*\* FAILED \*\*\*' "$LOG" | sed 's/^\[info\] *//; s/ \*\*\* FAILED \*\*\*.*//')
COUNT=$(printf '%s\n' "$FAILED_LINES" | grep -c . || true)
rm -f "$LOG"

if [ "$COUNT" -eq 0 ]; then
  suite_fail unit 1 "build-or-unknown"
else
  NAMES=$(printf '%s\n' "$FAILED_LINES" | paste -sd';' -)
  suite_fail unit "$COUNT" "$NAMES"
fi
exit 1
