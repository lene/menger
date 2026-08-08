#!/bin/sh
# Suite: preflight (Sprint 36 B1). Environment validation (CUDA_HOME/OPTIX_ROOT — hard
# fail here, unlike optix-jni's warn-only check: menger's native build needs a working
# toolchain to compile at all), GitHub Actions workflow lint (HAS_CI only), Scala-version
# consistency, scripts/check-version-consistency.sh, and git-tag availability. Extracted
# verbatim from the pre-push hook's former Phase 1.
#
# Reads $QA_REMOTE (the git remote name pre-push receives as $1) to allow re-pushing an
# already-tagged version to origin/github (tagging is CI's job on merge to main).
set -u
. ./standards/hooks/lib.sh

RED_TEXT='\e[38;5;196m'
GREEN_TEXT='\e[38;5;46m'
RESET_TEXT='\e[0m'

STATUS=0

if [ "${HAS_SCALA:-0}" -gt 0 ] || [ "${HAS_NATIVE:-0}" -gt 0 ]; then
  echo "=== Validating Environment ==="

  if [ -z "${CUDA_HOME:-}" ]; then
    echo "${RED_TEXT}Error: CUDA_HOME is not set${RESET_TEXT}"
    suite_fail preflight 1 "cuda-home-unset"
    exit 1
  fi
  if [ ! -d "$CUDA_HOME" ]; then
    echo "${RED_TEXT}Error: CUDA_HOME points to non-existent directory: $CUDA_HOME${RESET_TEXT}"
    suite_fail preflight 1 "cuda-home-invalid"
    exit 1
  fi
  if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "${RED_TEXT}Error: nvcc not found at $CUDA_HOME/bin/nvcc${RESET_TEXT}"
    suite_fail preflight 1 "nvcc-missing"
    exit 1
  fi
  if [ -z "${OPTIX_ROOT:-}" ]; then
    echo "${RED_TEXT}Error: OPTIX_ROOT is not set${RESET_TEXT}"
    suite_fail preflight 1 "optix-root-unset"
    exit 1
  fi
  if [ ! -d "$OPTIX_ROOT" ]; then
    echo "${RED_TEXT}Error: OPTIX_ROOT points to non-existent directory: $OPTIX_ROOT${RESET_TEXT}"
    suite_fail preflight 1 "optix-root-invalid"
    exit 1
  fi
  if [ ! -f "$OPTIX_ROOT/include/optix.h" ]; then
    echo "${RED_TEXT}Error: optix.h not found at $OPTIX_ROOT/include/optix.h${RESET_TEXT}"
    suite_fail preflight 1 "optix-h-missing"
    exit 1
  fi

  echo "CUDA_HOME: ${GREEN_TEXT}$CUDA_HOME${RESET_TEXT}"
  echo "OPTIX_ROOT: ${GREEN_TEXT}$OPTIX_ROOT${RESET_TEXT}"
fi

if [ "${HAS_CI:-0}" -gt 0 ]; then
  if command -v actionlint >/dev/null 2>&1; then
    if actionlint .github/workflows/*.yml; then
      echo "actionlint: ${GREEN_TEXT}PASSED${RESET_TEXT}"
    else
      echo "actionlint: ${RED_TEXT}FAILED${RESET_TEXT}"
      STATUS=1
    fi
  else
    echo "actionlint not installed — skipping workflow lint (advisory; GitHub validates on push)"
  fi

  SCALA_VERSION=$(egrep 'scalaVersion := ".*"' menger-app/build.sbt | head -n 1 | cut -d \" -f 2)
  CI_SCALA_VERSION=$(egrep '^[[:space:]]*SCALA_VERSION:' .github/workflows/ci.yml | head -n 1 | cut -d \" -f 2)
  if [ "$SCALA_VERSION" != "$CI_SCALA_VERSION" ]; then
    echo "Scala in build.sbt: ${RED_TEXT}${SCALA_VERSION}${RESET_TEXT}, in ci.yml: ${RED_TEXT}${CI_SCALA_VERSION}${RESET_TEXT}"
    STATUS=1
  else
    echo "Scala version: ${GREEN_TEXT}${SCALA_VERSION}${RESET_TEXT}"
  fi
fi

if [ "$STATUS" -ne 0 ]; then
  suite_fail preflight 1 "workflow-lint-or-scala-version"
  exit 1
fi

if ! ./scripts/check-version-consistency.sh; then
  suite_fail preflight 1 "version-consistency"
  exit 1
fi

VERSION_SBT=$(egrep 'version := ".*"' menger-app/build.sbt | cut -d \" -f 2)
if [ "${NO_RELEASE:-0}" = "1" ]; then
  echo "Tag check skipped (${GREEN_TEXT}NO_RELEASE=1${RESET_TEXT})"
elif git tag | grep -q "^${VERSION_SBT}\$"; then
  echo "Tag ${RED_TEXT}${VERSION_SBT}${RESET_TEXT} already exists ($(git tag | xargs))"
  if [ "${QA_REMOTE:-}" != "origin" ] && [ "${QA_REMOTE:-}" != "github" ]; then
    suite_fail preflight 1 "tag-exists"
    exit 1
  else
    echo "Pushing anyway (tagging is handled by CI on merge to main)"
  fi
else
  echo "Tag ${GREEN_TEXT}${VERSION_SBT}${RESET_TEXT} is still available"
fi

suite_pass preflight
