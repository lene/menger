#!/usr/bin/env bash
# Compare a menger caustics render (PFM) against a committed pbrt reference (PFM).
# Sprint 33, Task 33.1.
#
# Usage:
#   compare-caustics.sh <menger.pfm> <reference.pfm> [max_MSE] [max_FLIP]
#
# Exit 0 if both MSE and FLIP are within bounds, 1 otherwise. Prints the metrics.
# If thresholds are omitted, they are looked up in thresholds.txt by matching the
# reference basename (e.g. canonical-caustics_400x300 -> scene canonical-caustics,
# resolution 400x300).
#
# Metrics via imgtool (MSE, FLIP). FLIP is the primary perceptual gate; MSE is a
# coarse energy check. Both operate on the clipped-linear PFM pair, so the
# comparison is independent of the tone mapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMGTOOL="${IMGTOOL:-/usr/local/bin/imgtool}"
THRESHOLDS="$SCRIPT_DIR/thresholds.txt"

menger="${1:?menger PFM required}"
reference="${2:?reference PFM required}"
max_mse="${3:-}"
max_flip="${4:-}"

[ -f "$menger" ]    || { echo "MISSING menger image: $menger" >&2; exit 2; }
[ -f "$reference" ] || { echo "MISSING reference: $reference" >&2; exit 2; }

# Resolve thresholds from thresholds.txt if not given on the command line.
if [ -z "$max_mse" ] || [ -z "$max_flip" ]; then
  base="$(basename "$reference" .pfm)"                 # e.g. canonical-caustics_400x300
  scene="${base%_*}"; res="${base##*_}"
  line="$(grep -E "^[[:space:]]*${scene}[[:space:]]+${res}[[:space:]]" "$THRESHOLDS" | head -1 || true)"
  if [ -z "$line" ]; then
    echo "No thresholds for scene=$scene res=$res in $THRESHOLDS" >&2; exit 2
  fi
  max_mse="$(echo "$line" | awk '{print $3}')"
  max_flip="$(echo "$line" | awk '{print $4}')"
fi

extract() { grep -oE "$1 = [0-9.eE+-]+" | tail -1 | awk '{print $3}'; }

mse="$("$IMGTOOL" diff --metric MSE  --reference "$reference" "$menger" 2>/dev/null | extract MSE || true)"
flip="$("$IMGTOOL" diff --metric FLIP --reference "$reference" "$menger" 2>/dev/null | extract FLIP || true)"
: "${mse:=nan}" "${flip:=nan}"

printf 'compare: %s vs %s\n' "$(basename "$menger")" "$(basename "$reference")"
printf '  MSE  = %-12s (max %s)\n' "$mse"  "$max_mse"
printf '  FLIP = %-12s (max %s)\n' "$flip" "$max_flip"

ok=1
awk "BEGIN{exit !($mse  <= $max_mse)}"  2>/dev/null || ok=0
awk "BEGIN{exit !($flip <= $max_flip)}" 2>/dev/null || ok=0

if [ "$ok" = 1 ]; then echo "  PASS"; exit 0; else echo "  FAIL"; exit 1; fi
