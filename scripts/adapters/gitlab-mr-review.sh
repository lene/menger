#!/bin/sh
# Post AI review findings to a GitLab MR as a bot note. Idempotent.
# Updates the existing bot note if present; creates a new one otherwise.
#
# Usage: gitlab-mr-review.sh <findings-json>
#
# Env: GITLAB_API_TOKEN, CI_API_V4_URL, CI_PROJECT_ID, CI_MERGE_REQUEST_IID,
#      CI_COMMIT_SHA (for context), CI_JOB_URL (for context)
# Requires: curl, jq
set -eu

FINDINGS_FILE="${1:-review-findings.json}"
[ -f "$FINDINGS_FILE" ] || { echo "error: findings file not found: $FINDINGS_FILE" >&2; exit 1; }
command -v jq   > /dev/null || { echo "error: jq required" >&2;   exit 1; }
command -v curl > /dev/null || { echo "error: curl required" >&2; exit 1; }

[ -n "${GITLAB_API_TOKEN:-}" ]       || { echo "error: GITLAB_API_TOKEN not set" >&2;       exit 1; }
[ -n "${CI_API_V4_URL:-}" ]          || { echo "error: CI_API_V4_URL not set" >&2;          exit 1; }
[ -n "${CI_PROJECT_ID:-}" ]          || { echo "error: CI_PROJECT_ID not set" >&2;          exit 1; }
[ -n "${CI_MERGE_REQUEST_IID:-}" ]   || { echo "error: CI_MERGE_REQUEST_IID not set" >&2;   exit 1; }

BOT_MARKER="<!-- menger-ai-review-v1 -->"
NOTES_URL="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}/notes"

# --- Build note body ---
TOTAL=$(jq  '.stats.total'        "$FINDINGS_FILE")
AGREED=$(jq '.stats.agreed'       "$FINDINGS_FILE")
SINGLE=$(jq '.stats.single_model' "$FINDINGS_FILE")
CLAUDE_SUMMARY=$(jq  -r '.summaries.claude'   "$FINDINGS_FILE")
DS_SUMMARY=$(jq      -r '.summaries.deepseek' "$FINDINGS_FILE")
COMMIT_SHORT=$(printf '%.8s' "${CI_COMMIT_SHA:-unknown}")
JOB_LINK="${CI_JOB_URL:-}"

# Severity icons
sev_icon() {
  case "$1" in error) echo "🔴" ;; warning) echo "⚠️" ;; *) echo "ℹ️" ;; esac
}
models_label() {
  case "$1" in
    '["claude","deepseek"]') echo "Claude+DeepSeek" ;;
    '["claude"]')            echo "Claude" ;;
    '["deepseek"]')          echo "DeepSeek" ;;
    *)                       echo "$1" ;;
  esac
}

# Start building body in a temp file
BODY_FILE=$(mktemp)
trap 'rm -f "$BODY_FILE"' EXIT

cat >> "$BODY_FILE" << BODY_HEADER
${BOT_MARKER}
## 🤖 AI Code Review

_Claude + DeepSeek · commit \`${COMMIT_SHORT}\`${JOB_LINK:+ · [CI job](${JOB_LINK})}_

**${TOTAL} findings** — ${AGREED} agreed (both models), ${SINGLE} single-model

BODY_HEADER

if [ "$TOTAL" -gt 0 ]; then
  printf '| File | Line | | Category | Finding | Models |\n' >> "$BODY_FILE"
  printf '|------|------|---|----------|---------|--------|\n' >> "$BODY_FILE"

  jq -r '.findings[] |
    [ .file, (.line|tostring), .severity, .category, .message,
      (.models | join(",")) ]
    | @tsv' "$FINDINGS_FILE" | while IFS="$(printf '\t')" read -r file line sev cat msg models; do
    icon=$(sev_icon "$sev")
    mlabel=$(models_label "$(printf '["%s"]' "$(echo "$models" | sed 's/,/","/g')")")
    printf '| `%s` | %s | %s | %s | %s | %s |\n' \
      "$file" "$line" "$icon" "$cat" "$msg" "$mlabel" >> "$BODY_FILE"
  done
  printf '\n' >> "$BODY_FILE"
fi

printf '### Summaries\n\n' >> "$BODY_FILE"
printf '**Claude:** %s\n\n' "$CLAUDE_SUMMARY" >> "$BODY_FILE"
printf '**DeepSeek:** %s\n' "$DS_SUMMARY" >> "$BODY_FILE"

NOTE_BODY=$(cat "$BODY_FILE")

# --- Idempotency: find existing bot note ---------------------------------
EXISTING_NOTE_ID=$(curl -sf \
  -H "PRIVATE-TOKEN: $GITLAB_API_TOKEN" \
  "${NOTES_URL}?per_page=100" \
  | jq -r --arg marker "$BOT_MARKER" \
      '[.[] | select(.body | contains($marker))] | first | .id // empty' \
  2>/dev/null || echo "")

NOTE_PAYLOAD=$(jq -n --arg body "$NOTE_BODY" '{"body": $body}')

if [ -n "$EXISTING_NOTE_ID" ]; then
  echo "[gitlab-review] Updating existing note #${EXISTING_NOTE_ID}" >&2
  curl -sf -X PUT \
    -H "PRIVATE-TOKEN: $GITLAB_API_TOKEN" \
    -H "content-type: application/json" \
    "${NOTES_URL}/${EXISTING_NOTE_ID}" \
    -d "$NOTE_PAYLOAD" > /dev/null
else
  echo "[gitlab-review] Creating new bot note" >&2
  curl -sf -X POST \
    -H "PRIVATE-TOKEN: $GITLAB_API_TOKEN" \
    -H "content-type: application/json" \
    "${NOTES_URL}" \
    -d "$NOTE_PAYLOAD" > /dev/null
fi

echo "[gitlab-review] Done. ${TOTAL} findings posted to MR !${CI_MERGE_REQUEST_IID}"
