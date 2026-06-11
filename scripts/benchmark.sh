#!/usr/bin/env bash
# Performance benchmark: runs 4 representative scenes 3 times each, computes median
# frameMs per scene, then compares against perf-baseline.json.
#
# Usage:
#   ./scripts/benchmark.sh <menger-app-binary> [--update-baseline]
#
# Exit 0: all scenes within 15% of baseline (or --update-baseline was set).
# Exit 1: one or more scenes regressed >15%.
set -euo pipefail

BINARY=${1:?"Usage: $0 <menger-app-binary> [--update-baseline]"}
UPDATE_BASELINE=${2:-}
SCRIPT_DIR="$(cd "$(dirname "$0")" ; pwd)"
BASELINE_FILE="$SCRIPT_DIR/perf-baseline.json"
RUNS=3
THRESHOLD=1.15

if [ ! -x "$BINARY" ]; then
  echo "ERROR: binary not found or not executable: $BINARY" >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# label:args pairs
SCENES=(
  "sphere:--objects type=sphere"
  "sponge-volume-L4:--objects type=sponge-volume:level=4"
  "menger4d-L3:--objects type=menger4d:level=3"
  "sphere-IBL-accum:--objects type=sphere --accumulate"
)

echo "=== Menger Performance Benchmark ==="
echo "Binary: $BINARY"
echo "Runs per scene: $RUNS"
echo ""

RESULTS_FILE="$TMPDIR/results.json"
echo "{}" > "$RESULTS_FILE"

for entry in "${SCENES[@]}"; do
  label="${entry%%:*}"
  args="${entry#*:}"
  echo "--- Scene: $label ---"

  RUN_FRAMES=""
  for i in $(seq 1 $RUNS); do
    stats_file="$TMPDIR/${label}-run${i}.json"
    stderr_file="$TMPDIR/${label}-run${i}.err"
    # shellcheck disable=SC2086
    __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a "$BINARY" \
      $args \
      --headless \
      --timeout 0.5 \
      --stats-json "$stats_file" \
      2>"$stderr_file" || {
        echo "ERROR: scene $label run $i failed:" >&2
        cat "$stderr_file" >&2
        exit 1
      }
    if [ ! -f "$stats_file" ]; then
      echo "ERROR: stats file not written for $label run $i (app exited before first render)" >&2
      cat "$stderr_file" >&2
      exit 1
    fi
    frame_ms=$(python3 -c "import json; print(json.load(open('$stats_file'))['frameMs'])")
    echo "  run $i: ${frame_ms} ms"
    RUN_FRAMES="$RUN_FRAMES $frame_ms"
  done

  median=$(python3 -c "
import statistics
vals = [float(x) for x in '$RUN_FRAMES'.split()]
print(statistics.median(vals))
")
  echo "  median: ${median} ms"
  echo ""

  python3 -c "
import json
with open('$RESULTS_FILE') as f:
    d = json.load(f)
d['$label'] = float('$median')
with open('$RESULTS_FILE', 'w') as f:
    json.dump(d, f, indent=2)
"
done

if [ "$UPDATE_BASELINE" = "--update-baseline" ]; then
  cp "$RESULTS_FILE" "$BASELINE_FILE"
  echo "Baseline updated at $BASELINE_FILE:"
  cat "$BASELINE_FILE"
  exit 0
fi

if [ ! -f "$BASELINE_FILE" ]; then
  echo "No baseline at $BASELINE_FILE — run with --update-baseline first." >&2
  exit 1
fi

echo "=== Regression Check (threshold: ${THRESHOLD}x) ==="
python3 - "$RESULTS_FILE" "$BASELINE_FILE" "$THRESHOLD" << 'PYEOF'
import json, sys

results_file, baseline_file, threshold_str = sys.argv[1], sys.argv[2], sys.argv[3]
threshold = float(threshold_str)

with open(results_file) as f:
    measured = json.load(f)
with open(baseline_file) as f:
    baseline = json.load(f)

failed = []
for scene, ms in measured.items():
    if scene not in baseline:
        print(f"  {scene}: no baseline entry — skipping")
        continue
    base = baseline[scene]
    ratio = ms / base
    ok = ratio <= threshold
    print(f"  {'✅' if ok else '❌'} {scene}: {ms:.1f} ms (baseline {base:.1f} ms, {ratio:.2f}x)")
    if not ok:
        failed.append(scene)

print()
if failed:
    print(f"REGRESSION DETECTED in: {', '.join(failed)}")
    sys.exit(1)
else:
    print("All scenes within threshold. ✅")
PYEOF
