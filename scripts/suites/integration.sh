#!/bin/sh
# Suite: integration (Sprint 36 B1). 27-scenario integration suite, extracted verbatim.
# Depends on package.sh having already unpacked menger-app-<version>/ (same ordering as
# the original hook's Phase 5, which ran package immediately before this).
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_RENDERING:-0}" -eq 0 ] && [ "${HAS_NATIVE:-0}" -eq 0 ] && [ "${HAS_INTEGRATION:-0}" -eq 0 ]; then
  suite_skip integration "no rendering, native, or integration-script changes"
  exit 0
fi

gpu_preflight_or_skip integration || exit 1

VERSION=$(grep 'version :=' menger-app/build.sbt | cut -d '"' -f 2)
echo "=== Running integration tests ==="
if MAX_PARALLEL_JOBS=1 ./scripts/integration-tests.sh "./menger-app-${VERSION}/bin/menger-app"; then
  suite_pass integration
else
  suite_fail integration 1 "integration-tests"
  exit 1
fi
