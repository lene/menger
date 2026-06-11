#!/bin/sh
# Version consistency across the four version-carrying files (menger-specific).
# Extracted from the pre-push hook (Sprint 28.2) so pre-commit and pre-push
# share one implementation. Tag availability stays in pre-push (release-time).
set -u

RED_TEXT=$(printf '\033[38;5;196m')
GREEN_TEXT=$(printf '\033[38;5;46m')
RESET_TEXT=$(printf '\033[0m')

STATUS=0

VERSION_SBT=$(grep -E 'version := ".*"' menger-app/build.sbt | cut -d \" -f 2)
VERSIONS_CI=$(grep -E 'DEPLOYABLE_VERSION.*' .gitlab-ci.yml | head -n 1 | cut -d ':' -f 2 | xargs)
VERSION_SOURCE=$(grep -E 'version\("menger v.* ' menger-app/src/main/scala/menger/MengerCLIOptions.scala | cut -d 'v' -f 3 | cut -d ' ' -f 1)
if [ "$VERSION_SBT" != "$VERSIONS_CI" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in .gitlab-ci.yml: ${RED_TEXT}${VERSIONS_CI}${RESET_TEXT}"
  STATUS=1
fi
if [ "$VERSION_SBT" != "$VERSION_SOURCE" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in MengerCLIOptions.scala: ${RED_TEXT}${VERSION_SOURCE}${RESET_TEXT}"
  STATUS=1
else
  echo "Version: ${GREEN_TEXT}${VERSION_SBT}${RESET_TEXT}"
fi
VERSION_USERGUIDE=$(grep '^\*\*Version\*\*:' docs/guide/user-guide.md | cut -d ' ' -f 2)
if [ "$VERSION_SBT" != "$VERSION_USERGUIDE" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in docs/guide/user-guide.md: ${RED_TEXT}${VERSION_USERGUIDE}${RESET_TEXT}"
  STATUS=1
else
  echo "docs/guide/user-guide.md version: ${GREEN_TEXT}${VERSION_USERGUIDE}${RESET_TEXT}"
fi
VERSION_INDEX=$(grep '^\*\*Version\*\*:' docs/USER_GUIDE.md | cut -d ' ' -f 2)
if [ "$VERSION_SBT" != "$VERSION_INDEX" ]; then
  echo "Version in build.sbt: ${RED_TEXT}${VERSION_SBT}${RESET_TEXT}, in docs/USER_GUIDE.md: ${RED_TEXT}${VERSION_INDEX}${RESET_TEXT}"
  STATUS=1
else
  echo "docs/USER_GUIDE.md version: ${GREEN_TEXT}${VERSION_INDEX}${RESET_TEXT}"
fi

exit $STATUS
