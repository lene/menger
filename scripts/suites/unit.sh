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

# `sbt test` runs Project4DGpuSuite (real CUDA calls) alongside the plain unit tests, in
# one JVM run that can't be split without restructuring the suite (Sprint 36 D1) — same
# whole-suite gating memcheck.sh already applies even though only part of its work is
# GPU-bound.
gpu_preflight_or_skip unit || exit 1

echo "=== Compiling ==="
if ! sbt compile; then
  suite_fail unit 1 "compile"
  exit 1
fi

LOG=$(mktemp)
echo "=== Tests ==="
{ __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a sbt test --warn 2>&1; echo "SBT_TEST_RC=$?"; } | tee "$LOG"

# sbt's last write before exiting is often a bare ANSI erase-to-end-of-line sequence with no
# trailing newline (e.g. \x1b[0J), which then glues onto the front of the marker echo on the
# same captured line -- the anchored grep below sees "\x1b[0JSBT_TEST_RC=0", not
# "SBT_TEST_RC=0" at start-of-line, and misreports a passing run as failed. Strip ANSI CSI
# sequences (colors, cursor/erase codes) before matching.
if sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" | grep -q '^SBT_TEST_RC=0$'; then
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
