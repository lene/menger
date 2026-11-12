#!/usr/bin/env bash
# Tests scripts/frame-budget.sh offline: the log parsing on canned logs, and whole runs with
# fake launchers instead of the app (no GPU; the process tests need xvfb-run and a private lock).
# Run: scripts/test-frame-budget.sh
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPTS/frame-budget.sh"
# shellcheck source=/dev/null
source "$SCRIPT"
set +e   # the sourced script's set -e would end this run at the first expected failure
FAILURES=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

report() { # report <description> <exit-status>
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); fi
}
check() { # check <description> <command...>: passes when the command succeeds
    local description="$1"; shift
    "$@" > /dev/null 2>&1
    report "$description" $?
}
frame_line() { # frame_line <timestamp> <k>
    echo "$1 INFO  menger.engines.WithAnimation - Rendering frame $2/3 (t=1.0)"
}
noise() { echo "12:00:00.000 INFO  menger.Main - Loading scene: x.scala"; }

expect_equal() { # expect_equal <description> <actual> <expected>
    local status=0
    [ "$2" = "$3" ] || status=1
    report "$1" "$status"
}
expect_ms() { # expect_ms <description> <log> <ms>
    local out status=0
    out="$(frame_ms_from_log "$2")" || status=$?
    [ "$status" -eq 0 ] && [ "$out" = "$3" ] || status=1
    report "$1" "$status"
}
expect_failure() { # expect_failure <description> <log>
    local status=0
    frame_ms_from_log "$2" > /dev/null && status=1
    report "$1" "$status"
}
expect_cell() { # expect_cell <description> <log> <cell>
    local status=0
    [ "$(timeout_cell 1800 "$2")" = "$3" ] || status=1
    report "$1" "$status"
}

# ---- log parsing ----

{ noise; frame_line 10:00:01.000 1; noise; frame_line 10:00:03.500 2; frame_line 10:00:04.250 3
  noise; } > "$WORK/normal.log"
expect_ms "normal: 750 ms from the 2nd to the 3rd line, not the cold 1st" "$WORK/normal.log" 750

{ frame_line 23:59:58.000 1; frame_line 23:59:59.900 2; frame_line 00:00:00.400 3; } \
  > "$WORK/midnight.log"
expect_ms "midnight: 23:59:59.900 -> 00:00:00.400 is 500 ms" "$WORK/midnight.log" 500

{ frame_line 10:00:01.000 1; frame_line 10:00:02.000 2; frame_line 10:00:02.300 3
  frame_line 10:00:09.000 4; } > "$WORK/extra.log"
expect_ms "extra lines: only the first three count" "$WORK/extra.log" 300

{ frame_line 10:59:58.000 1; frame_line 10:59:59.000 2; frame_line 11:00:01.250 3; } \
  > "$WORK/carry.log"
expect_ms "carry across the hour: 2250 ms" "$WORK/carry.log" 2250

{ noise; frame_line 10:00:01.000 1; frame_line 10:00:02.000 2; } > "$WORK/few.log"
expect_failure "too few lines: non-zero exit" "$WORK/few.log"

noise > "$WORK/none.log"
expect_failure "no frame lines: non-zero exit" "$WORK/none.log"

{ noise; frame_line 10:00:01.000 1; } > "$WORK/t1.log"
expect_cell "timeout during frame 1" "$WORK/t1.log" "> 1800 s (timed out at frame 1/3)"
{ frame_line 10:00:01.000 1; frame_line 10:00:05.000 2; } > "$WORK/t2.log"
expect_cell "timeout during frame 2" "$WORK/t2.log" "> 1800 s (timed out at frame 2/3)"
expect_cell "timeout before any frame" "$WORK/none.log" "> 1800 s (timed out before frame 1)"

echo "10:00:03.000 ERROR menger.engines.WithAnimation - Failed to build frame 2" > "$WORK/error.log"
check "an ERROR line is found" log_has_error "$WORK/error.log"
log_has_error "$WORK/normal.log"; expect_equal "a clean log has no error" "$?" 1

# ---- timeout classification: 124, or 137 only after the timeout ran out ----

check "status 124 is a timeout" stopped_by_timeout 124 0 1800
check "status 137 after the timeout is a timeout" stopped_by_timeout 137 1800000 1800
check "status 137 long after the timeout is a timeout" stopped_by_timeout 137 1830000 1800
stopped_by_timeout 137 5000 1800; expect_equal "status 137 after 5 s of 1800 is not a timeout (OOM killer)" "$?" 1
stopped_by_timeout 1 5000000 1800; expect_equal "status 1 is never a timeout" "$?" 1

# ---- sample times ----

expect_equal "uniform 1 0.5 samples 0 and 0.5" "$(uniform_times 1 0.5 | paste -sd' ')" "0 0.5"
expect_equal "uniform 180 6 samples 30 times" "$(uniform_times 180 6 | wc -l)" 30
expect_equal "uniform 180 6 ends at 174" "$(uniform_times 180 6 | tail -n 1)" 174
expect_equal "uniform 0.3 0.1: no accumulated float error" "$(uniform_times 0.3 0.1 | paste -sd' ')" "0 0.1 0.2"

# ---- argument validation (exit 2, the message names the problem) ----

touch "$WORK/Scene.scala"
FAKE="$WORK/fake"
echo '#!/bin/sh' > "$FAKE"; chmod +x "$FAKE"
mkdir -p "$WORK/empty" "$WORK/tex/assets/hdri"
touch "$WORK/tex/$DEFAULT_HDRI"

expect_usage_error() { # expect_usage_error <description> <message fragment> <args...>
    local description="$1" fragment="$2" out status=0
    shift 2
    out="$("$SCRIPT" "$FAKE" "$WORK/Scene.scala" "$@" 2>&1)" || status=$?
    if [ "$status" -eq 2 ] && grep -qF -- "$fragment" <<< "$out"; then status=0; else status=1; fi
    report "$description" "$status"
}
expect_usage_error "--timeout 0" "--timeout must be a positive integer" --at 1 --timeout 0
expect_usage_error "--timeout 1.5" "--timeout must be a positive integer" --at 1 --timeout 1.5
expect_usage_error "--width 0" "--width must be a positive integer" --at 1 --width 0
expect_usage_error "--height -5" "--height must be a positive integer" --at 1 --height -5
expect_usage_error "--uniform 0 1" "--uniform T must be a positive number" --uniform 0 1
expect_usage_error "--uniform 10 0" "--uniform STEP must be a positive number" --uniform 10 0
expect_usage_error "--uniform 10 abc" "--uniform STEP must be a positive number" --uniform 10 abc
expect_usage_error "--at with an empty entry" "bad sample time ''" --at 1,,2
expect_usage_error "--at with a glob" "bad sample time '*'" --at '*'
expect_usage_error "--at with a path" "bad sample time '../x'" --at ../x
expect_usage_error "--at empty" "no sample times given" --at ""
expect_usage_error "unknown flag is named" "unknown option: --bogus" --at 1 --bogus
expect_usage_error "no mode" "give --at or --uniform" --width 100
expect_usage_error "--out not writable" "--out file is not writable" \
  --at 1 --out "$WORK/no-such-dir/out.md"
expect_usage_error "--hdri absolute" "--hdri must be a relative path" --at 1 --hdri /etc/passwd
expect_usage_error "--hdri with .." "--hdri must be a relative path" --at 1 --hdri ../x.hdr

# ---- missing assets (exit 1) ----

run_script() { # run_script <args...>: sets OUT and RC
    OUT="$("$SCRIPT" "$@" 2>&1)"; RC=$?
}
run_script "$FAKE" "$WORK/Scene.scala" --at 1 --texture-dir "$WORK/empty"
[ "$RC" -eq 1 ] && grep -q "scripts/fetch-kiara-hdri.sh 4k" <<< "$OUT"
report "no HDRI: exit 1, names fetch-kiara-hdri.sh 4k" $?
run_script "$FAKE" "$WORK/Scene.scala" --at 1 --texture-dir "$WORK/tex" --hdri assets/hdri/other.hdr
[ "$RC" -eq 1 ] && grep -qF "assets/hdri/other.hdr is missing" <<< "$OUT"
report "--hdri names the missing env map" $?

# ---- whole runs with fake launchers ----

# The fake launcher behaves as FAKE_MODE says. t is its --start-t.
cat > "$FAKE" <<'EOF'
#!/bin/sh
t=""
while [ $# -gt 0 ]; do [ "$1" = --start-t ] && t="$2"; shift; done
frame() { echo "$1 INFO  x.WithAnimation - Rendering frame $2/3 (t=$t)" >&2; }
three_frames() { frame 10:00:00.000 1; frame 10:00:01.000 2; frame 10:02:01.000 3; }   # warm 120000 ms
hang() {
  [ -z "$IGNORE_TERM" ] || trap '' TERM
  echo $$ > "$FAKE_DIR/launcher.pid"
  frame 10:00:00.000 1
  sleep 60 &
  echo $! > "$FAKE_DIR/sleep.pid"
  wait
}
case "$FAKE_MODE" in
  cost) three_frames ;;
  hang) hang ;;
  hang_at) if [ "$t" = "$HANG_T" ]; then hang; else three_frames; fi ;;
  exit3) echo "boom" >&2; exit 3 ;;
  error) three_frames; echo "10:02:02.000 ERROR x.WithAnimation - frame 3 failed" >&2 ;;
esac
EOF
chmod +x "$FAKE"

gone() { ! kill -0 "$(cat "$1")" 2> /dev/null; }

if command -v xvfb-run > /dev/null; then
  export FAKE_DIR="$WORK" GPU_LOCK="$WORK/lock" KILL_GRACE_SECONDS=1
  run_fake() { # run_fake <mode> <args...>: sets OUT and RC
      local mode="$1"; shift
      rm -f "$WORK/launcher.pid" "$WORK/sleep.pid"
      OUT="$(FAKE_MODE="$mode" "$SCRIPT" "$FAKE" "$WORK/Scene.scala" --texture-dir "$WORK/tex" "$@" 2>&1)"
      RC=$?
  }

  run_fake cost --uniform 1 0.5 --out "$WORK/out.md"
  [ "$RC" -eq 0 ] && grep -qF "| Scene | 0 | 120000 |" <<< "$OUT" \
    && grep -qF "| Scene | 0.5 | 120000 |" <<< "$OUT" \
    && grep -qxF "Mean: 120000.0 ms/frame. Budget for 1 s at 30 fps (30 frames): 1.00 h." <<< "$OUT"
  report "uniform 1 0.5: both rows, exact Mean/Budget line" $?
  head -n 1 "$WORK/out.md" | grep -q "^Frame budget [0-9-]* [0-9:]*; GPU: .*; menger commit: .*; 1920x1080; timeout 1800 s; scene: Scene; hdri: $DEFAULT_HDRI$"
  report "--out starts with the provenance line" $?
  grep -qF "Mean: 120000.0 ms/frame" "$WORK/out.md"; report "--out has the table and the budget" $?

  HANG_T=0.5 run_fake hang_at --uniform 1 0.5 --timeout 1
  [ "$RC" -eq 0 ] && grep -qF "No budget: 1 of 2 samples timed out." <<< "$OUT"
  report "uniform, one of two samples times out: No budget line, exit 0" $?
  gone "$WORK/sleep.pid"; report "the hung launcher's child is gone" $?

  HANG_T=1 run_fake hang_at --at 1,2 --timeout 1
  [ "$RC" -eq 0 ] && grep -qF "| Scene | 1 | > 1 s (timed out at frame 1/3) |" <<< "$OUT" \
    && grep -qF "| Scene | 2 | 120000 |" <<< "$OUT"
  report "hang at the first t only: timeout row, then a measured row, exit 0" $?

  HANG_T=1 IGNORE_TERM=1 run_fake hang_at --at 1,2 --timeout 1
  [ "$RC" -eq 0 ] && grep -qF "| Scene | 1 | > 1 s (timed out at frame 1/3) |" <<< "$OUT" \
    && grep -qF "| Scene | 2 | 120000 |" <<< "$OUT"
  report "a launcher ignoring TERM still gives the timeout row" $?
  gone "$WORK/sleep.pid"; report "the TERM-ignoring launcher's child is gone" $?

  run_fake hang --at 1 --timeout 1
  [ "$RC" -eq 1 ] && grep -qF "every t timed out" <<< "$OUT"
  report "every t times out: exit 1" $?

  run_fake exit3 --at 1
  [ "$RC" -eq 1 ] && grep -qF "failed with exit status 3" <<< "$OUT" && grep -qF "boom" <<< "$OUT"
  report "a launcher exiting 3: exit 1, failure message and the log" $?

  run_fake error --at 1
  [ "$RC" -eq 1 ] && grep -qF "the app logged an ERROR" <<< "$OUT"
  report "3 frames plus an ERROR line: exit 1, not a measurement" $?
  ! grep -qE '^\| Scene \| 1 \|' <<< "$OUT"; report "...and no row for it" $?

  # An interrupted run kills the render before it exits and frees the lock.
  rm -f "$WORK/launcher.pid" "$WORK/sleep.pid"
  FAKE_MODE=hang "$SCRIPT" "$FAKE" "$WORK/Scene.scala" --texture-dir "$WORK/tex" --at 1 \
    --timeout 60 > /dev/null 2>&1 &
  SCRIPT_PID=$!
  for _ in $(seq 1 100); do [ -s "$WORK/sleep.pid" ] && break; sleep 0.1; done
  kill -TERM "$SCRIPT_PID"
  wait "$SCRIPT_PID"; STATUS=$?
  expect_equal "TERM: the script exits 130" "$STATUS" 130
  gone "$WORK/sleep.pid" && gone "$WORK/launcher.pid"; report "TERM: the render is gone with it" $?
  flock -n "$WORK/lock" true; report "TERM: the lock is free" $?

  # A child that outlives the render (Xvfb after a kill) must not keep the GPU lock.
  ORPHAN="$WORK/orphan"
  {
    echo '#!/bin/sh'
    # in a session of its own, so that nothing but the lock's fd decides whether it matters
    echo 'setsid sh -c '"'"'echo $$ > "'"$WORK"'/orphan.pid"; exec sleep 20'"'"' &'
    echo 'sleep 1'
    for k in 1 2 3; do
      echo "echo \"10:00:0$k.000 INFO  x - Rendering frame $k/3 (t=1.0)\" >&2"
    done
  } > "$ORPHAN"
  chmod +x "$ORPHAN"
  GPU_LOCK="$WORK/lock2" "$SCRIPT" "$ORPHAN" "$WORK/Scene.scala" --at 1 \
    --texture-dir "$WORK/tex" > /dev/null 2>&1
  flock -n "$WORK/lock2" true; report "orphaned child does not hold the GPU lock after the render" $?
  kill "$(cat "$WORK/orphan.pid")" 2> /dev/null
else
  echo "SKIP: xvfb-run is missing: the whole-run tests with fake launchers (uniform, timeouts," \
    "failures, interrupt, orphaned child)"
fi

echo
if [ "$FAILURES" -eq 0 ]; then echo "All frame-budget tests passed"; else echo "$FAILURES failed"; exit 1; fi
