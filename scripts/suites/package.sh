#!/bin/sh
# Suite: package (Sprint 36 B1). packageBin + unzip, extracted verbatim. integration.sh
# depends on this suite's unpacked menger-app-<version>/ directory — keep it immediately
# before integration.sh in every suite list, same ordering as the original hook's Phase 5.
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_RENDERING:-0}" -eq 0 ] && [ "${HAS_NATIVE:-0}" -eq 0 ] && [ "${HAS_INTEGRATION:-0}" -eq 0 ]; then
  suite_skip package "no rendering, native, or integration-script changes"
  exit 0
fi

echo "=== Building release package ==="
if ! sbt "mengerApp / Universal / packageBin" --warn; then
  suite_fail package 1 "packageBin"
  exit 1
fi
VERSION=$(grep 'version :=' menger-app/build.sbt | cut -d '"' -f 2)
if ! unzip -oq "./menger-app/target/universal/menger-app-${VERSION}.zip"; then
  suite_fail package 1 "unzip"
  exit 1
fi
suite_pass package
