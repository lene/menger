#!/usr/bin/env bash
# Tests scripts/fetch-kiara-hdri.sh offline: file:// URLs with a small fixture and a pin
# overridden to its hash. Also checks that the script's pins agree with PROVENANCE.yaml.
# Run: scripts/test-fetch-kiara-hdri.sh
set -uo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPTS/fetch-kiara-hdri.sh"
PROVENANCE="$SCRIPTS/../PROVENANCE.yaml"
FAILURES=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SERVER="$WORK/server"
mkdir -p "$SERVER/4k"
echo "not really an HDRI" > "$SERVER/4k/kiara_2_sunrise_4k.hdr"
FIXTURE_HASH="$(sha256sum "$SERVER/4k/kiara_2_sunrise_4k.hdr" | awk '{print $1}')"
WRONG_HASH="$(printf '0%.0s' $(seq 64))"
NAME=kiara_2_sunrise_4k.hdr

export KIARA_HDRI_BASE_URL="file://$SERVER"
export KIARA_HDRI_PINS="4k $FIXTURE_HASH"

check() { # check <description> <condition-exit-status>
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); fi
}
run_fetch() { # run_fetch <dir> <args...>: sets OUT and RC
    local dir="$1"; shift
    OUT="$(KIARA_HDRI_DIR="$dir" "$SCRIPT" "$@" 2>&1)"; RC=$?
}
leftovers() { ls -A "$1" 2> /dev/null | grep -v "^$NAME$" | wc -l; }

# Fetch: file absent
DIR="$WORK/fetch"
run_fetch "$DIR" 4k
[ "$RC" -eq 0 ] && cmp -s "$DIR/$NAME" "$SERVER/4k/$NAME"
check "fetch: file downloaded and verified, exit 0" $?
[ "$(leftovers "$DIR")" -eq 0 ]; check "fetch: no temp file left behind" $?

# Default resolution is 4k
DIR="$WORK/default"
run_fetch "$DIR"
[ "$RC" -eq 0 ] && [ -f "$DIR/$NAME" ]; check "fetch: default resolution is 4k" $?

# Already there: no download (URL would fail)
DIR="$WORK/present"
run_fetch "$DIR" 4k
KIARA_HDRI_BASE_URL="file://$WORK/nowhere" run_fetch "$DIR" 4k
[ "$RC" -eq 0 ] && grep -q "already present" <<< "$OUT"
check "already there: no download, exit 0" $?

# Corrupt local copy: re-downloaded and verified
DIR="$WORK/corrupt"
mkdir -p "$DIR"; echo corrupt > "$DIR/$NAME"
run_fetch "$DIR" 4k
[ "$RC" -eq 0 ] && cmp -s "$DIR/$NAME" "$SERVER/4k/$NAME"
check "corrupt local copy: re-downloaded, exit 0" $?

# Corrupt local copy and a download that does not verify: nothing at the target, hashes named
DIR="$WORK/corrupt-mismatch"
mkdir -p "$DIR"; echo corrupt > "$DIR/$NAME"
KIARA_HDRI_PINS="4k $WRONG_HASH" run_fetch "$DIR" 4k
[ "$RC" -ne 0 ] && [ ! -e "$DIR/$NAME" ]; check "corrupt local copy, bad download: removed, exit non-zero" $?
grep -q "$WRONG_HASH" <<< "$OUT" && grep -q "$FIXTURE_HASH" <<< "$OUT"
check "corrupt local copy, bad download: message names expected and actual hash" $?

# Mismatch: download's hash differs from the pin
DIR="$WORK/mismatch"
KIARA_HDRI_PINS="4k $WRONG_HASH" run_fetch "$DIR" 4k
[ "$RC" -ne 0 ] && [ ! -e "$DIR/$NAME" ]; check "mismatch: nothing at the target, exit non-zero" $?
[ "$(leftovers "$DIR")" -eq 0 ]; check "mismatch: temp file removed" $?
grep -q "want $WRONG_HASH" <<< "$OUT" && grep -q "got $FIXTURE_HASH" <<< "$OUT"
check "mismatch: message names expected and actual hash" $?

# Bad resolution: exit non-zero before any download
DIR="$WORK/badres"
run_fetch "$DIR" 3k
[ "$RC" -ne 0 ] && [ ! -e "$DIR" ]; check "bad resolution: exit non-zero, nothing created" $?
grep -q "pinned resolutions: 4k" <<< "$OUT"; check "bad resolution: message lists pinned resolutions" $?

# Offline: download fails
DIR="$WORK/offline"
KIARA_HDRI_BASE_URL="file://$WORK/nowhere" run_fetch "$DIR" 4k
[ "$RC" -ne 0 ] && [ ! -e "$DIR/$NAME" ]; check "offline: nothing at the target, exit non-zero" $?
[ "$(leftovers "$DIR")" -eq 0 ]; check "offline: no temp file left behind" $?
grep -qi "curl" <<< "$OUT"; check "offline: curl's error passed through" $?

# Pins agree with PROVENANCE.yaml
unset KIARA_HDRI_PINS
script_pins="$(grep -E '^[0-9]+k [0-9a-f]{64}' "$SCRIPT" | tr -d '"' | awk '{print $1, $2}' | sort)"
provenance_pins="$(grep -E '^ +[0-9]+k: *[0-9a-f]{64}' "$PROVENANCE" | tr -d ':' | awk '{print $1, $2}' | sort)"
[ -n "$script_pins" ] && [ "$script_pins" = "$provenance_pins" ]
check "pins in the script and in PROVENANCE.yaml agree" $?

if (( FAILURES > 0 )); then echo "$FAILURES failure(s)"; exit 1; fi
echo "all passed"
