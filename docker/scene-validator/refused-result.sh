# shellcheck shell=bash
# Sourced by run-sandboxed.sh and run-render-window.sh: the one place the shell wrappers write
# the tagged result (contracts/process-result.json). They used to printf it by hand, claiming
# schema 1.0.0 while the validator said 1.1.0, and a quote in the message broke the JSON
# (review 2026-10-14, menger#82/#100). ProcessContractSuite runs both wrappers against the file.

RESULT_SCHEMA_VERSION="1.1.0"   # = SceneValidator.SchemaVersion
REFUSED_EXIT_CODE=2             # = SceneValidator.exitCodeFor(Refused)

# json_string <text>: <text> as a JSON string literal.
json_string() {
  local s="${1//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\t'/\\t}"
  s="${s//$'\r'/\\r}"
  printf '"%s"' "$s"
}

# refuse <message>: print the `refused` result on stdout and exit with its code.
refuse() {
  printf '{"tag": "refused", "messages": [%s], "findings": [], "schemaVersion": "%s"}\n' \
    "$(json_string "$1")" "$RESULT_SCHEMA_VERSION"
  exit "$REFUSED_EXIT_CODE"
}
