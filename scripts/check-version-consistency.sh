#!/bin/sh
# Version consistency across the four version-carrying files (menger-specific):
# build.sbt, ci.yml, MengerCLIOptions.scala, docs/USER_GUIDE.md. docs/guide/user-guide.md
# is a separate, non-version-bearing doc, deliberately not checked here.
# Extracted from the pre-push hook (Sprint 28.2) so pre-commit and pre-push
# share one implementation. Tag availability stays in pre-push (release-time).
set -u

RED_TEXT=$(printf '\033[38;5;196m')
GREEN_TEXT=$(printf '\033[38;5;46m')
RESET_TEXT=$(printf '\033[0m')

STATUS=0

VERSION_SBT=$(grep -E 'version := ".*"' menger-app/build.sbt | cut -d \" -f 2)
VERSIONS_CI=$(grep -E '^[[:space:]]*DEPLOYABLE_VERSION:' .github/workflows/ci.yml | head -n 1 | cut -d \" -f 2)
VERSION_SOURCE=$(grep -E 'version\("menger v.* ' menger-app/src/main/scala/menger/MengerCLIOptions.scala | cut -d 'v' -f 3 | cut -d ' ' -f 1)
if [ "$VERSION_SBT" != "$VERSIONS_CI" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in ci.yml: ${RED_TEXT}${VERSIONS_CI}${RESET_TEXT}"
  STATUS=1
fi
if [ "$VERSION_SBT" != "$VERSION_SOURCE" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in MengerCLIOptions.scala: ${RED_TEXT}${VERSION_SOURCE}${RESET_TEXT}"
  STATUS=1
else
  echo "Version: ${GREEN_TEXT}${VERSION_SBT}${RESET_TEXT}"
fi
VERSION_INDEX=$(grep '^\*\*Version\*\*:' docs/USER_GUIDE.md | cut -d ' ' -f 2)
if [ "$VERSION_SBT" != "$VERSION_INDEX" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in docs/USER_GUIDE.md: ${RED_TEXT}${VERSION_INDEX}${RESET_TEXT}"
  STATUS=1
else
  echo "docs/USER_GUIDE.md version: ${GREEN_TEXT}${VERSION_INDEX}${RESET_TEXT}"
fi

# --- Toolchain pins advertised to the scene agent -----------------------------------------
# ManifestGenerator and CorpusExporter each hardcode the Scala and optix-jni versions they
# tell the agent it is generating for (AD-9's staleness signal). Nothing tied those constants
# to the build, and each suite asserted its own file's constant against a hand-copied literal,
# so a `scalaVersion` bump left all four copies stale and every test green (review round 2).
SCALA_BUILD=$(grep -E '^[[:space:]]*scalaVersion := ".*"' menger-app/build.sbt | head -n 1 | cut -d \" -f 2)
OPTIX_BUILD=$(grep -E 'optixJniDependency[[:space:]]*=' build.sbt | grep -oE '"[0-9]+\.[0-9]+\.[0-9]+"' | tail -n 1 | tr -d '"')

for TOOL_FILE in menger-app/src/main/scala/menger/tools/ManifestGenerator.scala \
                 menger-app/src/main/scala/menger/tools/CorpusExporter.scala; do
  SCALA_PIN=$(grep -E 'ScalaVersionPin = ".*"' "$TOOL_FILE" | cut -d \" -f 2)
  OPTIX_PIN=$(grep -E 'OptixJniVersionPin = ".*"' "$TOOL_FILE" | cut -d \" -f 2)
  if [ -n "$SCALA_BUILD" ] && [ "$SCALA_PIN" != "$SCALA_BUILD" ]; then
    echo "scalaVersion in menger-app/build.sbt: ${RED_TEXT}${SCALA_BUILD}${RESET_TEXT}, ScalaVersionPin in ${TOOL_FILE}: ${RED_TEXT}${SCALA_PIN}${RESET_TEXT}"
    STATUS=1
  fi
  if [ -n "$OPTIX_BUILD" ] && [ "$OPTIX_PIN" != "$OPTIX_BUILD" ]; then
    echo "optix-jni in build.sbt: ${RED_TEXT}${OPTIX_BUILD}${RESET_TEXT}, OptixJniVersionPin in ${TOOL_FILE}: ${RED_TEXT}${OPTIX_PIN}${RESET_TEXT}"
    STATUS=1
  fi
done
if [ "$STATUS" -eq 0 ]; then
  echo "Toolchain pins: ${GREEN_TEXT}scala ${SCALA_BUILD}, optix-jni ${OPTIX_BUILD}${RESET_TEXT}"
fi

exit $STATUS
