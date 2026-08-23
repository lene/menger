#!/usr/bin/env bash
# Performance benchmark: runs the representative scenes 3 times each, computes median
# frameMs per scene, then compares against perf-baseline.json.
#
# Usage:
#   ./scripts/benchmark.sh <menger-app-binary> [--update-baseline]
#
# Exit 0: all scenes within THRESHOLD of baseline (or --update-baseline was set).
# Exit 1: one or more scenes regressed beyond THRESHOLD.
#
# Performance budgets (arc42 §10 P1/P2):
#   P1 — single-frame interactive scenes render in < 5000 ms.
#   P2 — the fast primitive/curve scenes render in < 500 ms.
# Every scene below is a P1 case; the fast ones (glass-sphere, curve, tesseract) also hold P2.
# The absolute ceilings are enforced below alongside the relative ratio — a scene that
# stays within 1.3× baseline but exceeds its absolute ceiling fails the gate (F5).
set -euo pipefail

BINARY=${1:?"Usage: $0 <menger-app-binary> [--update-baseline]"}
UPDATE_BASELINE=${2:-}
SCRIPT_DIR="$(cd "$(dirname "$0")" ; pwd)"
BASELINE_FILE="$SCRIPT_DIR/perf-baseline.json"
RUNS=3
# Real baselines measured on the dev GPU in Sprint 33.13. The absolute budgets (P1 < 5 s,
# P2 < 500 ms) are the primary acceptance criterion — every scene here runs 30-100x under P1.
# This ratio is a secondary regression tripwire only. The primitive scenes render in ~50 ms,
# where a few ms of run-to-run jitter is a large relative swing (observed curve 1.17-1.23x with
# no code change), so the gate is set loose enough not to false-positive on noise.
THRESHOLD=1.30

if [ ! -x "$BINARY" ]; then
  echo "ERROR: binary not found or not executable: $BINARY" >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# label:args pairs (args exclude the common --headless/--save-name/--stats-json flags added below)
# _calibration (Sprint 36 D2) is a same-run baseline probe, not a tracked scene: cheapest
# possible render, measured identically to every other scene, popped out of the results
# before the per-scene comparison below and used to normalize each scene's ratio against
# this run's own machine speed — a slow/throttled/contended run and a fast/cool run should
# then report the same *adjusted* ratio even though their raw ms differ.
SCENES=(
  "_calibration:--objects type=sphere:pos=0,0,0:size=0.3"
  "glass-sphere:--objects type=sphere:pos=0,0,0:size=0.5:material=glass --plane y:-2"
  "diamond-sphere:--objects type=sphere:pos=0,0.5,0:size=0.3:material=diamond-dispersive:ior=2.42 --plane y:-2"
  "menger4d-L2:--objects type=menger4d:level=2:pos=0,0,0:size=0.8 --plane y:-2"
  "sierpinski4d-L2:--objects type=sierpinski4d:level=2:pos=0,0,0:size=0.8 --plane y:-2"
  "tesseract:--objects type=tesseract:pos=0,0,0:size=0.8 --plane y:-2"
  "curve:--objects type=curve:control-points=0,0,0,1,0,0,1,1,0,0,1,0:radius=0.05"
  "lsystem-tree-L3:--objects type=lsystem:preset=tree:level=3:size=0.8 --plane y:-2"
  "sphere-IBL-accum:--objects type=sphere --accumulation-frames 8"
  "caustics-glass:--objects type=sphere:ior=1.5 --caustics --caustics-photons 1000 --caustics-iterations 1 --plane y:-2"
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
      --allow-uniform-render \
      --save-name "$TMPDIR/${label}.png" \
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

# Absolute ceilings (F5): a scene can be within ratio but still over budget.
CEILING_P1 = 5000.0   # ms — every scene must stay under this
CEILING_P2 = 500.0    # ms — fast scenes (primitives/curves) must also stay under this
P2_SCENES = {"glass-sphere", "curve", "tesseract"}

with open(results_file) as f:
    measured = json.load(f)
with open(baseline_file) as f:
    baseline = json.load(f)

# Same-run calibration probe (Sprint 36 D2): >1.0 means this run's machine is slower
# than the baseline capture day (thermal throttling, GPU contention, weaker hardware).
# Dividing each scene's raw ratio by this normalizes that out.
calibration_ms = measured.pop("_calibration", None)
calibration_baseline_ms = baseline.pop("_calibration", None)
if calibration_ms is not None and calibration_baseline_ms is not None:
    calibration_ratio = calibration_ms / calibration_baseline_ms
    print(f"  calibration probe: {calibration_ms:.1f} ms (baseline {calibration_baseline_ms:.1f} ms, {calibration_ratio:.2f}x)")
    print()
else:
    calibration_ratio = 1.0
    print("  calibration probe: no baseline entry — ratios unadjusted")
    print()

failed = []
for scene, ms in measured.items():
    if scene not in baseline:
        print(f"  {scene}: no baseline entry — skipping")
        continue
    base = baseline[scene]
    raw_ratio = ms / base
    ratio = raw_ratio / calibration_ratio
    ok = ratio <= threshold
    ceiling = CEILING_P2 if scene in P2_SCENES else CEILING_P1
    abs_ok = ms <= ceiling
    status = '✅' if (ok and abs_ok) else '❌'
    print(f"  {status} {scene}: {ms:.1f} ms (baseline {base:.1f} ms, raw {raw_ratio:.2f}x, adjusted {ratio:.2f}x, ceiling {ceiling:.0f} ms)")
    if not ok:
        failed.append(f"{scene} (adjusted ratio {ratio:.2f}x > {threshold}x)")
    if not abs_ok:
        failed.append(f"{scene} (absolute {ms:.1f} ms > {ceiling:.0f} ms ceiling)")

print()
if failed:
    print(f"REGRESSION DETECTED in: {', '.join(failed)}")
    sys.exit(1)
else:
    print("All scenes within threshold. ✅")
PYEOF
