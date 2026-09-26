#!/usr/bin/env bash
# Performance trend benchmark: renders representative scenes with the packaged binary and
# compares each scene's cost, normalised by a calibration render, against perf-baseline.json.
#
# Same concept as the in-process perf gates (io.github.lene.qa.RelativeBenchmark): every scene
# run is bracketed by calibration runs (cal, s1, cal, s2, ..., cal), and each round's ratio
# scene / mean(adjacent calibrations) cancels drift from throttling and background load. Over
# ROUNDS rounds, the median of those ratios (relative to the baseline) and a sign-test
# confidence interval decide: PASS if the whole interval is within THRESHOLD, FAIL if the whole
# interval is beyond it, INCONCLUSIVE otherwise or if the interval is too wide to trust.
#
# Usage:
#   ./scripts/benchmark.sh <menger-app-binary> [--update-baseline | --ratchet]
#
#   --update-baseline  store this run's median ratios as the new baseline
#   --ratchet          also lower a scene's baseline when it conclusively improved
#
# Exit 0: every scene passed (or the baseline was updated).
# Exit 1: a scene conclusively regressed, or is conclusively over its absolute budget.
# Exit 2: nothing failed, but at least one scene was inconclusive.
#
# Absolute budgets (arc42 §10 P1/P2) are kept as budget guards, not regression gates:
#   P1 — single-frame interactive scenes render in < 5000 ms.
#   P2 — the fast primitive/curve scenes render in < 500 ms.
# A scene fails its budget only if even its fastest round is over the ceiling.
set -euo pipefail

BINARY=${1:?"Usage: $0 <menger-app-binary> [--update-baseline | --ratchet]"}
MODE=${2:-}
SCRIPT_DIR="$(cd "$(dirname "$0")" ; pwd)"
BASELINE_FILE="$SCRIPT_DIR/perf-baseline.json"
ROUNDS=${BENCHMARK_ROUNDS:-7}   # 7 rounds: the sign-test interval is (min, max) at 98.4%
THRESHOLD=1.30                  # max slowdown vs baseline
RATCHET_THRESHOLD=0.90          # conclusive improvement needed to lower the baseline
MAX_SPREAD=2.0                  # interval wider than this: environment unstable, inconclusive

if [ ! -x "$BINARY" ]; then
  echo "ERROR: binary not found or not executable: $BINARY" >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# The calibration render: the cheapest possible scene, measured exactly like the others.
CALIBRATION="--objects type=sphere:pos=0,0,0:size=0.3"
# label:args pairs (args exclude the common --headless/--save-name/--stats-json flags)
SCENES=(
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

# Renders one scene headless and prints its frame time in ms. $1 = label, $2 = args.
# Callers capture stdout, which must be the number alone; xvfb-run merges the app's stderr
# into stdout, so all app output goes to a log file.
render_ms() {
  local stats_file="$TMPDIR/$1.json" log_file="$TMPDIR/$1.log"
  rm -f "$stats_file"
  # shellcheck disable=SC2086
  __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a "$BINARY" $2 \
    --headless --allow-uniform-render \
    --save-name "$TMPDIR/$1.png" --stats-json "$stats_file" \
    >"$log_file" 2>&1 || { echo "ERROR: scene $1 failed:" >&2; cat "$log_file" >&2; exit 1; }
  if [ ! -f "$stats_file" ]; then
    echo "ERROR: stats file not written for $1 (app exited before first render)" >&2
    cat "$log_file" >&2
    exit 1
  fi
  python3 -c "import json; print(json.load(open('$stats_file'))['frameMs'])"
}

echo "=== Menger Performance Benchmark ==="
echo "Binary: $BINARY"
echo "Rounds: $ROUNDS (each scene bracketed by calibration renders)"
echo ""

RAW="$TMPDIR/raw.tsv"   # label, scene ms, calibration ms before, calibration ms after
: > "$RAW"
for round in $(seq 1 "$ROUNDS"); do
  echo "--- Round $round/$ROUNDS ---"
  cal_before=$(render_ms _calibration "$CALIBRATION")
  for entry in "${SCENES[@]}"; do
    label="${entry%%:*}"
    ms=$(render_ms "$label" "${entry#*:}")
    cal_after=$(render_ms _calibration "$CALIBRATION")
    printf '%s\t%s\t%s\t%s\n' "$label" "$ms" "$cal_before" "$cal_after" >> "$RAW"
    echo "  $label: $ms ms (calibration $cal_before / $cal_after ms)"
    cal_before=$cal_after
  done
done
echo ""

python3 - "$RAW" "$BASELINE_FILE" "$MODE" "$THRESHOLD" "$RATCHET_THRESHOLD" "$MAX_SPREAD" << 'PYEOF'
import json, sys
from collections import defaultdict
from math import comb

raw_file, baseline_file, mode = sys.argv[1], sys.argv[2], sys.argv[3]
threshold, ratchet_threshold, max_spread = map(float, sys.argv[4:7])
CONFIDENCE = 0.95
CEILING_P1 = 5000.0   # ms, every scene
CEILING_P2 = 500.0    # ms, the fast scenes below
P2_SCENES = {"glass-sphere", "curve", "tesseract"}

ratios, frame_ms = defaultdict(list), defaultdict(list)
for line in open(raw_file):
    label, ms, cal_before, cal_after = line.split("\t")
    ratios[label].append(float(ms) / ((float(cal_before) + float(cal_after)) / 2))
    frame_ms[label].append(float(ms))

def median(values):
    s, n = sorted(values), len(values)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def interval(values):
    """Sign-test confidence interval for the median (same rule as RelativeBenchmark)."""
    s, n = sorted(values), len(values)
    ks = [k for k in range(1, n // 2 + 1)
          if 2 * sum(comb(n, i) for i in range(k)) / 2 ** n <= 1 - CONFIDENCE]
    k = ks[-1] if ks else 1
    return s[k - 1], s[n - k]

if mode == "--update-baseline":
    baseline = {label: round(median(r), 4) for label, r in ratios.items()}
    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"Baseline updated at {baseline_file} (scene / calibration render time):")
    print(json.dumps(baseline, indent=2))
    sys.exit(0)

with open(baseline_file) as f:
    baseline = json.load(f)
if "_calibration" in baseline:
    sys.exit("ERROR: perf-baseline.json holds milliseconds (old format); regenerate it with "
             "--update-baseline, which stores machine-independent ratios")

print(f"=== Regression check (limit {threshold}x baseline) ===")
failed, inconclusive, improved = [], [], {}
for label, scene_ratios in ratios.items():
    if label not in baseline:
        print(f"  {label}: no baseline entry, skipping")
        continue
    relative = [r / baseline[label] for r in scene_ratios]
    med, low, high = median(relative), *interval(relative)
    ceiling = CEILING_P2 if label in P2_SCENES else CEILING_P1
    fastest_ms = min(frame_ms[label])
    if high / low > max_spread:
        verdict = f"INCONCLUSIVE (too noisy: interval spans {high / low:.2f}x)"
        inconclusive.append(label)
    elif high <= threshold:
        verdict = "PASS"
    elif low > threshold:
        verdict = "FAIL"
        failed.append(f"{label} ({med:.2f}x baseline)")
    else:
        verdict = "INCONCLUSIVE (interval straddles the limit)"
        inconclusive.append(label)
    if fastest_ms > ceiling:
        verdict += f", OVER BUDGET ({fastest_ms:.0f} ms > {ceiling:.0f} ms)"
        failed.append(f"{label} (over {ceiling:.0f} ms budget)")
    if mode == "--ratchet" and high < ratchet_threshold:
        improved[label] = round(median(scene_ratios), 4)
    print(f"  {label}: {med:.2f}x baseline, CI [{low:.2f}, {high:.2f}] -> {verdict}")

if improved:
    for label, value in improved.items():
        print(f"perf ratchet: {label} improved conclusively, baseline {baseline[label]} -> {value}")
    baseline.update(improved)
    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")

print()
if failed:
    print(f"REGRESSION DETECTED in: {', '.join(failed)}")
    sys.exit(1)
if inconclusive:
    print(f"Inconclusive (too noisy to judge): {', '.join(inconclusive)}")
    sys.exit(2)
print("All scenes within limits.")
PYEOF
