#!/bin/sh
# Suite: native-checks (Sprint 36 B1). cppcheck + clang-tidy, extracted verbatim from the
# pre-push hook's run_cppcheck/run_clang_tidy. Those ran in parallel (background jobs) in
# the old hook; this v1 runs them sequentially (~1-2 min slower) — a stated, deliberate
# scope cut in the Sprint 36 Phase B plan. No CI job existed for either check before this
# suite (B4 adds one, closing that gap).
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_NATIVE:-0}" -eq 0 ]; then
  suite_skip native-checks "no native changes"
  exit 0
fi

STATUS=0
FAILED=""
CPP_FILES=$(git ls-files '*.cpp' | grep 'menger-geometry/src/main/native/')

# --- cppcheck ---
if ! command -v cppcheck >/dev/null 2>&1; then
  echo "cppcheck: not installed, skipping"
elif [ -z "$CPP_FILES" ]; then
  echo "cppcheck: skipped (no C++ source files)"
else
  echo "=== cppcheck ==="
  # shellcheck disable=SC2086
  if cppcheck --enable=warning,performance,portability --inline-suppr \
      --suppressions-list=.cppcheck-suppress --error-exitcode=1 --quiet \
      ${CPP_FILES}; then
    echo "cppcheck: PASSED"
  else
    echo "cppcheck: FAILED"
    STATUS=1
    FAILED="cppcheck"
  fi
fi

# --- clang-tidy ---
BUILD_DIR="menger-geometry/target/native/x86_64-linux/build"
if ! command -v clang-tidy >/dev/null 2>&1; then
  echo "clang-tidy: not installed, skipping"
elif [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
  echo "clang-tidy: skipped (compile_commands.json not found at ${BUILD_DIR} — run sbt compile first)"
elif [ -z "$CPP_FILES" ]; then
  echo "clang-tidy: skipped (no C++ source files)"
else
  echo "=== clang-tidy ==="
  GCC_VER=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
  GCC_ARGS=""
  if [ -d "/usr/include/c++/${GCC_VER}" ]; then
    GCC_ARGS="--extra-arg=-I/usr/include/c++/${GCC_VER} --extra-arg=-I/usr/include/x86_64-linux-gnu/c++/${GCC_VER}"
  fi
  # shellcheck disable=SC2086
  if clang-tidy -p "${BUILD_DIR}" --config-file=.clang-tidy ${GCC_ARGS} ${CPP_FILES}; then
    echo "clang-tidy: PASSED"
  else
    echo "clang-tidy: FAILED"
    STATUS=1
    FAILED="${FAILED:+$FAILED;}clang-tidy"
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  suite_pass native-checks
else
  suite_fail native-checks 1 "$FAILED"
fi
exit "$STATUS"
