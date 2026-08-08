#!/bin/sh
# Suite: coverage (Sprint 36 B1). Coverage ratchet (.coverage_baseline: >=80% absolute,
# max 1% drop), extracted verbatim from the pre-push hook's run_coverage.
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_SCALA:-0}" -eq 0 ]; then
  suite_skip coverage "no Scala changes"
  exit 0
fi
if [ "${SKIP_COVERAGE:-0}" = "1" ]; then
  suite_skip coverage "SKIP_COVERAGE=1"
  exit 0
fi

echo "=== Checking Test Coverage ==="
BASELINE_FILE=".coverage_baseline"
COVERAGE_DROP_THRESHOLD="1.0"
COVERAGE_MIN="80"
COVERAGE_FLOOR="60"
REPORT_FILE="menger-app/target/scala-3.8.3/scoverage-report/scoverage.xml"

# Clean separately to avoid ClassNotFoundException with scoverage instrumented classes
sbt "project mengerApp" clean --warn
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a sbt "project mengerApp" "set coverageEnabled := true" test coverageReport --warn
COVERAGE_STATUS=$?

if [ "$COVERAGE_STATUS" -ne 0 ]; then
  echo "Coverage report: FAILED"
  suite_fail coverage 1 "coverage-run"
  exit 1
fi
if [ ! -f "$REPORT_FILE" ]; then
  echo "Coverage: Report file not found"
  suite_fail coverage 1 "report-missing"
  exit 1
fi

STATEMENT_RATE=$(grep -oP 'statement-rate="\K[0-9.]+' "$REPORT_FILE" | head -1)
if [ -z "$STATEMENT_RATE" ]; then
  echo "Coverage: Could not parse coverage rate"
  suite_fail coverage 1 "unparseable-rate"
  exit 1
fi

if [ -f "$BASELINE_FILE" ]; then
  BASELINE=$(cat "$BASELINE_FILE")
else
  BASELINE="$STATEMENT_RATE"
  echo "No baseline file found, using current coverage as baseline"
fi

DROP=$(echo "$BASELINE - $STATEMENT_RATE" | bc)
echo "Statement coverage - current: ${STATEMENT_RATE}% previous: ${BASELINE}%"

BELOW_FLOOR=$(echo "$STATEMENT_RATE < $COVERAGE_FLOOR" | bc)
if [ "$BELOW_FLOOR" -eq 1 ]; then
  echo "Coverage: FAILED - Below absolute floor of ${COVERAGE_FLOOR}%"
  suite_fail coverage 1 "below-floor"
  exit 1
elif [ "$(echo "$DROP > $COVERAGE_DROP_THRESHOLD" | bc)" -eq 1 ]; then
  BELOW_MIN=$(echo "$STATEMENT_RATE < $COVERAGE_MIN" | bc)
  if [ "$BELOW_MIN" -eq 1 ]; then
    echo "Coverage: FAILED - Dropped ${DROP}% and below ${COVERAGE_MIN}%"
    suite_fail coverage 1 "dropped-below-min"
    exit 1
  else
    echo "Coverage: WARNING - Dropped ${DROP}% but above ${COVERAGE_MIN}%"
  fi
else
  echo "Coverage: PASSED"
  echo "$STATEMENT_RATE" > "$BASELINE_FILE"
fi

suite_pass coverage
