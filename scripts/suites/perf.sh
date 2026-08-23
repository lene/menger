#!/bin/sh
# Suite: perf (Sprint 36 F4, O7/D2). Wraps benchmark.sh's existing ratio/ceiling gate,
# then ratchets perf-baseline.json when a scene improves by a noise-clearing margin
# (>=10% faster than its stored baseline) — anything smaller is run-to-run jitter (see
# benchmark.sh's own comment: observed 1.17-1.23x swings with no code change) and left
# untouched, or the floor would chase noise on every green run. Depends on package.sh
# having already unpacked menger-app-<version>/, same as integration.sh.
set -u
. ./standards/hooks/lib.sh

gpu_preflight_or_skip perf || exit 1

VERSION=$(grep 'version :=' menger-app/build.sbt | cut -d '"' -f 2)
BINARY="./menger-app-${VERSION}/bin/menger-app"
LOG=$(mktemp)
echo "=== Running perf benchmark ==="
if ! ./scripts/benchmark.sh "$BINARY" > "$LOG" 2>&1; then
  cat "$LOG"
  suite_fail perf 1 "regression-or-ceiling"
  exit 1
fi
cat "$LOG"

IMPROVE_THRESHOLD=0.90
python3 - "$LOG" "scripts/perf-baseline.json" "$IMPROVE_THRESHOLD" << 'PYEOF'
import json, re, sys
log_file, baseline_file, improve_threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(baseline_file) as f:
    baseline = json.load(f)
changed = False
for m in re.finditer(r'^\s+.\s+(\S+): ([\d.]+) ms .*adjusted ([\d.]+)x', open(log_file).read(), re.M):
    scene, ms, ratio = m.group(1), float(m.group(2)), float(m.group(3))
    if ratio < improve_threshold and scene in baseline:
        print(f"perf ratchet: {scene} improved (adjusted {ratio:.2f}x < {improve_threshold}x) "
              f"— lowering baseline {baseline[scene]:.1f}ms -> {ms:.1f}ms")
        baseline[scene] = ms
        changed = True
if changed:
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
PYEOF

suite_pass perf
