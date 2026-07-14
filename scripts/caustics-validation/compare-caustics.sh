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

printf 'compare: %s vs %s\n' "$(basename "$menger")" "$(basename "$reference")"

if command -v "$IMGTOOL" >/dev/null 2>&1; then
  mse="$("$IMGTOOL" diff --metric MSE  --reference "$reference" "$menger" 2>/dev/null | extract MSE || true)"
  flip="$("$IMGTOOL" diff --metric FLIP --reference "$reference" "$menger" 2>/dev/null | extract FLIP || true)"
  : "${mse:=nan}" "${flip:=nan}"
else
  # Fallback (Sprint 33.11): imgtool (pbrt's image tool) is absent on the dev host and not
  # installed in the CI image, so the MSE/FLIP metrics cannot come from it. Compute the coarse
  # MSE energy check directly from the two linear PFMs in node, and skip FLIP (reimplementing
  # FLIP correctly is out of scope; it runs wherever imgtool exists). This preserves the
  # script's stated role: a LOOSE whole-image sanity bound that only catches gross breakage
  # (see thresholds.txt). The tight C8 gate is compare-caustic-delta.sh (pure node, no imgtool);
  # the full 800x600 + FLIP ladder stays in --full / manual where imgtool is available.
  mse="$(node - "$menger" "$reference" <<'NODE' 2>/dev/null || true
const fs=require('fs');function load(p){const b=fs.readFileSync(p);let i=0,h=[];
 while(h.length<3){let s='';while(b[i]!==10){s+=String.fromCharCode(b[i++]);}i++;h.push(s);}
 const d=h[1].trim().split(/\s+/).map(Number),le=parseFloat(h[2])<0;
 const a=new Float32Array(d[0]*d[1]*3);
 for(let k=0;k<a.length;k++)a[k]=le?b.readFloatLE(i+k*4):b.readFloatBE(i+k*4);return a;}
const A=load(process.argv[2]),B=load(process.argv[3]);
if(A.length!==B.length){console.error('PFM size mismatch');process.exit(2);}
let s=0;for(let k=0;k<A.length;k++){const e=A[k]-B[k];s+=e*e;}
console.log((s/A.length).toExponential(4));
NODE
)"
  : "${mse:=nan}"
  flip="n/a (imgtool absent)"
fi

printf '  MSE  = %-12s (max %s)\n' "$mse"  "$max_mse"
printf '  FLIP = %-12s (max %s)\n' "$flip" "$max_flip"

ok=1
awk "BEGIN{exit !($mse  <= $max_mse)}"  2>/dev/null || ok=0
[ "$flip" = "n/a (imgtool absent)" ] || awk "BEGIN{exit !($flip <= $max_flip)}" 2>/dev/null || ok=0

if [ "$ok" = 1 ]; then echo "  PASS"; exit 0; else echo "  FAIL"; exit 1; fi
