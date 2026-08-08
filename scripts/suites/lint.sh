#!/bin/sh
# Suite: lint (Sprint 36 B1). Scalafix check, split out of the former combined
# "scalafix then tests" phase. Kept immediately before unit.sh in every suite list —
# same BootServerSocket-contention reason the original phase ran them back to back
# (scalafix releases sbt's boot server socket before the test JVM needs it).
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_SCALA:-0}" -eq 0 ] && [ "${HAS_NATIVE:-0}" -eq 0 ]; then
  suite_skip lint "no Scala or native changes"
  exit 0
fi
if [ "${HAS_SCALA:-0}" -eq 0 ]; then
  suite_skip lint "no Scala changes (native-only push)"
  exit 0
fi

echo "=== Scalafix ==="
sbt "scalafix --check" --warn > /tmp/scalafix-prepush.log 2>&1
SCALAFIX_STATUS=$?
cat /tmp/scalafix-prepush.log
rm -f /tmp/scalafix-prepush.log

if [ "$SCALAFIX_STATUS" -eq 0 ]; then
  echo "scalafix: PASSED"
  suite_pass lint
else
  echo "scalafix: FAILED"
  suite_fail lint 1 "scalafix"
  exit 1
fi
