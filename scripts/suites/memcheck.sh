#!/bin/bash
# bash, not sh: uses `local` and `printf %q` (bash-isms), matching the source hook.
# Suite: memcheck (Sprint 36 B1). Valgrind (host-side JNI leaks) + compute-sanitizer
# (device-side CUDA errors host valgrind cannot see), extracted verbatim from the
# pre-push hook's run_valgrind / run_compute_sanitizer / make_tool_java_home. No CI job
# existed for either check before this suite (B4 adds one, closing a real gap — see the
# Sprint 36 Phase B plan).
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_NATIVE:-0}" -eq 0 ]; then
  suite_skip memcheck "no native changes"
  exit 0
fi

gpu_preflight_or_skip memcheck || exit 1

# Shared with run_sanitizer_self_check() (Sprint 36 E2) — a future edit that weakens
# these flags weakens the self-check too, and the self-check will (correctly) stop
# detecting its own known defect, aborting the gate before it silently degrades.
CS_MEMCHECK_FLAGS="--tool memcheck --leak-check full"

# The tool must wrap the *forked test JVM*, not xvfb-run/sbt: valgrind does not trace
# child processes by default, so wrapping `xvfb-run -a sbt ...` instrumented only the
# dash wrapper script — it never saw menger native code (vacuous gate) and failed on a
# 9-byte leak inside /usr/bin/dash (false positive). This builds a fake java home whose
# bin/java execs the tool around the real JVM; sbt forks its test JVM through it via
# `Test / javaHome`.
make_tool_java_home() {
  local dir="$1"; shift   # remaining args: tool + its flags
  local real_java
  real_java=$(readlink -f "$(command -v java)")
  mkdir -p "$dir/bin"
  if [ "${1##*/}" = "compute-sanitizer" ]; then
    # compute-sanitizer exits 255 when the wrapped process made no CUDA calls (e.g.
    # sbt's `java -version` probe) — translate that specific case to 0, or every probe
    # JVM looks like a crashed one. %p in --log-file gives each tracked process
    # (--target-processes defaults to `all`) its OWN log file: without it the
    # CUDA-making JVM's report and a short-lived child's collide, the real "ERROR
    # SUMMARY" ends up on stdout, and the caller's anti-vacuity guard (which reads the
    # files) sees only the child's "terminated before first" and wrongly calls the
    # gate vacuous.
    : "${CS_LOG_DIR:?must be set when wrapping compute-sanitizer}"
    {
      echo '#!/bin/sh'
      printf 'logdir=%q\n' "$CS_LOG_DIR"
      # $@ already begins with the tool name (compute-sanitizer) — don't repeat it.
      printf ' %q' "$@"
      printf ' --log-file "$logdir/sanitizer.$$.%%p.log" %q "$@"\n' "$real_java"
      echo 'rc=$?'
      echo 'if [ "$rc" -eq 255 ] && grep -q "terminated before first instrumented API call" "$logdir"/sanitizer.$$.*.log 2>/dev/null; then exit 0; fi'
      echo 'exit "$rc"'
    } > "$dir/bin/java"
  else
    {
      echo '#!/bin/sh'
      printf 'exec'
      printf ' %q' "$@" "$real_java"
      printf ' "$@"\n'
    } > "$dir/bin/java"
  fi
  chmod +x "$dir/bin/java"
}

# --- Valgrind (host-side leaks in JNI native code) ---
run_valgrind() {
  echo "=== Valgrind ==="

  if ! command -v valgrind >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
      echo "Valgrind: FAILED - native changed on a CUDA-capable host but valgrind is not installed; the leak gate must not be silently skipped where it can run. Install valgrind."
      return 1
    fi
    echo "Warning: valgrind not installed and no GPU detected - skipping native memory checks"
    return 0
  fi

  echo "Running valgrind on menger-geometry JNI native code (VideoLoaderSuite, instrumented test JVM)..."

  # No --error-exitcode: JVMs under memcheck emit false uninitialised-value errors from
  # JIT code, so the exit code is noise. The gate below fails on the test result and on
  # the parsed leak summary instead. JVM and driver false positives are suppressed via
  # valgrind-suppressions.txt.
  local VGJH
  VGJH=$(mktemp -d /tmp/vg-javahome.XXXXXX)
  make_tool_java_home "$VGJH" valgrind \
    --leak-check=full \
    "--suppressions=$PWD/valgrind-suppressions.txt" \
    --log-file=/tmp/valgrind-detail-prepush.log

  __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a sbt \
    "set mengerGeometry / Test / javaHome := Some(file(\"$VGJH\"))" \
    "mengerGeometry / Test / testOnly menger.geometry.VideoLoaderSuite" --warn
  local STATUS_VG=$?
  rm -rf "$VGJH"

  if [ "$STATUS_VG" -ne 0 ]; then
    echo "Valgrind: FAILED (test run itself failed, see above)"
    return 1
  fi

  # Anti-vacuity guard: the detail log must show the instrumented JVM. Without it the
  # gate would be checking nothing.
  if ! grep -qE 'Command: .*/bin/java ' /tmp/valgrind-detail-prepush.log || \
     ! grep -q 'HEAP SUMMARY' /tmp/valgrind-detail-prepush.log; then
    echo "Valgrind: FAILED - no instrumented JVM found in /tmp/valgrind-detail-prepush.log; the gate would be vacuous"
    return 1
  fi

  # Attribution: FAIL only on leaks whose allocation stack passes through this repo's
  # native library (libmengergeometry). liboptixjni is a pinned Maven artifact — its
  # leaks are fixed in the optix-jni repo (Sprint 35 Ph2) and land here via pin bumps,
  # so they are warnings only. Third-party internals (JVM, glibc TLS/loader, ffmpeg,
  # CUDA driver) leak by design and are not actionable here at all. A leak caused by
  # *misusing* a library (e.g. a missing av_frame_free) still has one of our frames in
  # its stack.
  local LEAK_BLOCKS OPTIX_JNI_BLOCKS
  LEAK_BLOCKS=$(awk '
    /bytes in [0-9]+ blocks are (definitely|possibly|indirectly) lost/ {inblock=1; block=$0 "\n"; next}
    inblock && /^==[0-9]+== *$/ { if (block ~ /libmengergeometry/) printf "%s", block; inblock=0; next }
    inblock { block = block $0 "\n" }
    END { if (inblock && block ~ /libmengergeometry/) printf "%s", block }
  ' /tmp/valgrind-detail-prepush.log)
  OPTIX_JNI_BLOCKS=$(awk '
    /bytes in [0-9]+ blocks are (definitely|possibly|indirectly) lost/ {inblock=1; block=$0 "\n"; next}
    inblock && /^==[0-9]+== *$/ { if (block ~ /liboptixjni/) printf "%s", block; inblock=0; next }
    inblock { block = block $0 "\n" }
    END { if (inblock && block ~ /liboptixjni/) printf "%s", block }
  ' /tmp/valgrind-detail-prepush.log)
  if [ -n "$OPTIX_JNI_BLOCKS" ]; then
    echo "Valgrind: WARNING - leaks attributed to pinned liboptixjni (fixed in the optix-jni repo, not blocking):"
    echo "$OPTIX_JNI_BLOCKS"
  fi
  if [ -n "$LEAK_BLOCKS" ]; then
    echo "Valgrind: FAILED - leaks in menger native code (see /tmp/valgrind-detail-prepush.log):"
    echo "$LEAK_BLOCKS"
    return 1
  fi
  echo "Valgrind: PASSED"
}

# --- compute-sanitizer (device-side CUDA errors host valgrind cannot see) ---
run_compute_sanitizer() {
  echo "=== compute-sanitizer ==="

  if ! command -v compute-sanitizer >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
      echo "compute-sanitizer: FAILED - CUDA-capable host but compute-sanitizer is not on PATH (usually /usr/local/cuda/bin); it must run where it can."
      return 1
    fi
    echo "compute-sanitizer: skipped (not installed, no GPU detected)"
    return 0
  fi

  # Project4DGpuSuite creates/renders/disposes an OptiXRenderer — real CUDA traffic.
  # (mengerGeometry's only suite, VideoLoaderSuite, makes no CUDA calls, so sanitizing
  # it was a vacuous pass.) The suite's GPU tests are all tagged Slow, so we must NOT
  # exclude Slow here — that would drop every CUDA test and make the gate vacuous.
  # Instead RUNNING_UNDER_COMPUTE_SANITIZER makes the two perf-timing tests self-skip;
  # the correctness GPU tests run and make the instrumented CUDA calls the anti-vacuity
  # guard requires.
  echo "Running compute-sanitizer (memcheck) on the OptiX render path (Project4DGpuSuite, instrumented test JVM)..."

  local CSJH CS_LOG_DIR
  CSJH=$(mktemp -d /tmp/cs-javahome.XXXXXX)
  CS_LOG_DIR=$(mktemp -d /tmp/cs-logs.XXXXXX)
  make_tool_java_home "$CSJH" compute-sanitizer $CS_MEMCHECK_FLAGS --error-exitcode 1

  RUNNING_UNDER_COMPUTE_SANITIZER=true __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a sbt \
    "set mengerApp / Test / javaHome := Some(file(\"$CSJH\"))" \
    "mengerApp / Test / testOnly io.github.lene.optix.Project4DGpuSuite" --warn
  local STATUS_CS=$?
  rm -rf "$CSJH"

  # Logs of JVMs that made instrumented CUDA calls. Probe JVMs (e.g. sbt's
  # `java -version`) legitimately end with "terminated before first instrumented API
  # call" and prove nothing.
  local INSTRUMENTED_LOGS
  INSTRUMENTED_LOGS=$(grep -L 'terminated before first instrumented API call' "$CS_LOG_DIR"/sanitizer.*.log 2>/dev/null || true)

  # Anti-vacuity guard: at least one wrapped JVM must have made instrumented CUDA
  # calls — otherwise the gate checked nothing (previously a silent "PASSED" on
  # exactly this condition).
  if [ -z "$INSTRUMENTED_LOGS" ]; then
    if [ "$STATUS_CS" -ne 0 ]; then
      echo "compute-sanitizer: FAILED (test run itself failed, see above)"
    else
      echo "compute-sanitizer: FAILED - no CUDA calls were instrumented ($CS_LOG_DIR); the gate would be vacuous"
    fi
    return 1
  fi

  # Attribution: FAIL only on findings whose backtrace passes through this repo's
  # native library (libmengergeometry). Findings inside the pinned liboptixjni (or the
  # CUDA driver) are fixed in the optix-jni repo (Sprint 35 Ph2) and land here via pin
  # bumps — warnings only, kept visible.
  local fail=0 findings=0 f
  for f in $INSTRUMENTED_LOGS; do
    if grep -qE 'ERROR SUMMARY: [1-9][0-9]* error' "$f"; then
      findings=1
      if grep -q 'libmengergeometry' "$f"; then
        echo "compute-sanitizer: FAILED - findings in menger native code ($f):"
        cat "$f"
        fail=1
      else
        echo "compute-sanitizer: WARNING - findings attributed to pinned liboptixjni / CUDA driver (fixed in the optix-jni repo, not blocking): $(grep -E 'LEAK SUMMARY|ERROR SUMMARY' "$f" | paste -sd' ')"
      fi
    fi
  done

  if [ "$fail" -ne 0 ]; then
    return 1
  fi
  if [ "$STATUS_CS" -ne 0 ] && [ "$findings" -eq 0 ]; then
    echo "compute-sanitizer: FAILED (test run failed without sanitizer findings, see above)"
    return 1
  fi
  echo "compute-sanitizer: PASSED"
}

# --- Sanitizer self-check (Sprint 36 E2): proves the gate below can actually catch
# a real defect before its PASS is trusted. Runs as its own tiny standalone binary —
# no sbt/JVM/xvfb-run — so it adds seconds, not minutes, to the hook.
run_sanitizer_self_check() {
  echo "=== compute-sanitizer self-check ==="

  if ! command -v compute-sanitizer >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    echo "compute-sanitizer self-check: skipped (compute-sanitizer/nvcc not on PATH)"
    return 0
  fi

  local dir bin
  dir=$(mktemp -d /tmp/cs-selfcheck.XXXXXX)
  bin="$dir/defect"
  if ! nvcc -o "$bin" "$PWD/scripts/suites/sanitizer-self-check.cu" 2>"$dir/nvcc.log"; then
    echo "compute-sanitizer self-check: FAILED - could not build the known-defective sample:"
    cat "$dir/nvcc.log"; rm -rf "$dir"; return 1
  fi

  # No --error-exitcode: a non-zero exit is the *expected* outcome, and the
  # log content (not the exit code) is what gets asserted below.
  compute-sanitizer $CS_MEMCHECK_FLAGS "$bin" >"$dir/selfcheck.log" 2>&1

  if grep -qE 'ERROR SUMMARY: [1-9][0-9]* error' "$dir/selfcheck.log"; then
    echo "compute-sanitizer self-check: PASSED (gate correctly flagged the known defect)"
    rm -rf "$dir"; return 0
  fi

  echo "SANITIZER SELF-CHECK DID NOT DETECT KNOWN DEFECT — gate is not trustworthy, aborting"
  cat "$dir/selfcheck.log"; rm -rf "$dir"; return 1
}

run_sanitizer_self_check || {
  echo "memcheck: aborting before running the real suite — sanitizer self-check failed"
  exit 1
}

STATUS=0
FAILED=""
run_valgrind || { STATUS=1; FAILED="valgrind"; }
run_compute_sanitizer || { STATUS=1; FAILED="${FAILED:+$FAILED;}compute-sanitizer"; }

if [ "$STATUS" -eq 0 ]; then
  suite_pass memcheck
else
  suite_fail memcheck 1 "$FAILED"
fi
exit "$STATUS"
