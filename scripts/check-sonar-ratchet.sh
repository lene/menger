#!/bin/bash
# bash, not sh: uses `<(...)` process substitution to source the baseline file's
# key=value lines without a temp file, matching memcheck.sh's precedent for scripts
# that need a bashism.
# Sonar ratchet (Sprint 36 F2, O7). Sonar analysis runs via the SonarCloud GitHub App,
# not this repo's own CI — this script blocks until that app's check for the current SHA
# is complete, then ratchets against .sonar_baseline. Primary: SonarCloud Web API (needs
# SONAR_TOKEN). Fallback (missing/invalid token, or API error): parse the check-run
# conclusion via gh CLI — a coarser signal (pass/fail only, no per-metric ratings).
set -u
. ./standards/hooks/lib.sh

PROJECT_KEY=$(grep -oP 'sonar.projectKey=\K.*' sonar-project.properties)
BASELINE_FILE=".sonar_baseline"
SHA="${1:?usage: check-sonar-ratchet.sh <commit-sha>}"
REPO="${GITHUB_REPOSITORY:-lene/menger}"

wait_for_sonar_check() {
  gh api "repos/$REPO/commits/$SHA/check-runs" \
    --jq '.check_runs[] | select(.name | test("SonarCloud"; "i")) | select(.status == "completed")' \
    | grep -q .
}
RETRY_ATTEMPTS=10 RETRY_BASE_DELAY=15 retry_with_backoff "sonarcloud check-run" wait_for_sonar_check || {
  echo "check-sonar-ratchet: SonarCloud check never completed for $SHA — failing closed"
  exit 1
}

API_OK=0
if [ -n "${SONAR_TOKEN:-}" ]; then
  RESPONSE=$(curl -sf -u "$SONAR_TOKEN:" \
    "https://sonarcloud.io/api/measures/component?component=${PROJECT_KEY}&metricKeys=alert_status,sqale_rating,reliability_rating,security_rating") \
    && API_OK=1
fi

if [ "$API_OK" -eq 1 ]; then
  ALERT=$(echo "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(m["value"] for m in d["component"]["measures"] if m["metric"]=="alert_status"))')
  SQALE=$(echo "$RESPONSE"  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(m["value"] for m in d["component"]["measures"] if m["metric"]=="sqale_rating"))')
  RELIAB=$(echo "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(m["value"] for m in d["component"]["measures"] if m["metric"]=="reliability_rating"))')
  SECUR=$(echo "$RESPONSE"  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(m["value"] for m in d["component"]["measures"] if m["metric"]=="security_rating"))')

  [ -f "$BASELINE_FILE" ] || printf 'alert_status=%s\nsqale_rating=%s\nreliability_rating=%s\nsecurity_rating=%s\n' \
    "$ALERT" "$SQALE" "$RELIAB" "$SECUR" > "$BASELINE_FILE"

  # shellcheck disable=SC1090
  . <(sed -n 's/^\([a-z_]*\)=\(.*\)$/BASELINE_\1="\2"/p' "$BASELINE_FILE")

  FAIL=0
  ratchet_check sqale_rating "$SQALE" "$BASELINE_sqale_rating" down 0 || FAIL=1
  ratchet_check reliability_rating "$RELIAB" "$BASELINE_reliability_rating" down 0 || FAIL=1
  ratchet_check security_rating "$SECUR" "$BASELINE_security_rating" down 0 || FAIL=1
  if [ "$ALERT" = "ERROR" ] && [ "${BASELINE_alert_status:-OK}" = "OK" ]; then
    echo "check-sonar-ratchet: quality gate flipped OK -> ERROR"
    FAIL=1
  fi

  if [ "$FAIL" -eq 0 ]; then
    printf 'alert_status=%s\nsqale_rating=%s\nreliability_rating=%s\nsecurity_rating=%s\n' \
      "$ALERT" "$SQALE" "$RELIAB" "$SECUR" > "$BASELINE_FILE"
  fi
  exit "$FAIL"
fi

echo "check-sonar-ratchet: SONAR_TOKEN unavailable or API call failed — falling back to check-run conclusion"
CONCLUSION=$(gh api "repos/$REPO/commits/$SHA/check-runs" \
  --jq '[.check_runs[] | select(.name | test("SonarCloud"; "i"))][0].conclusion')
[ -f "$BASELINE_FILE" ] || echo "fallback_conclusion=$CONCLUSION" > "$BASELINE_FILE"
BASELINE_CONCLUSION=$(grep -oP 'fallback_conclusion=\K.*' "$BASELINE_FILE" 2>/dev/null || echo "success")
if [ "$BASELINE_CONCLUSION" = "success" ] && [ "$CONCLUSION" != "success" ]; then
  echo "check-sonar-ratchet (fallback): SonarCloud check regressed to '$CONCLUSION'"
  exit 1
fi
echo "fallback_conclusion=$CONCLUSION" > "$BASELINE_FILE"
exit 0
