#!/bin/sh
# Suite: perf-trend (Sprint 36 F4, O7/D2; renamed from `perf`). Runs benchmark.sh against the
# stored baseline, ratcheting a scene's baseline down when it conclusively improved.
# benchmark.sh brackets every scene render with calibration renders and judges each scene with
# a confidence interval (same concept as the in-process perf gates): exit 0 pass, 1 fail,
# 2 inconclusive -> SKIP, never a failure. Depends on package.sh having already unpacked
# menger-app-<version>/, same as integration.sh.
set -u
. ./standards/hooks/lib.sh

# A busy GPU makes timings meaningless: skip, in CI too.
gpu_preflight_or_skip perf-trend --skip-in-ci || exit 0

VERSION=$(grep 'version :=' menger-app/build.sbt | cut -d '"' -f 2)
BINARY="./menger-app-${VERSION}/bin/menger-app"
echo "=== Running perf trend benchmark ==="
./scripts/benchmark.sh "$BINARY" --ratchet
case $? in
  0) suite_pass perf-trend ;;
  2) suite_skip perf-trend "inconclusive: measurements too noisy to judge some scenes" ;;
  *) suite_fail perf-trend 1 "regression-or-budget"; exit 1 ;;
esac
