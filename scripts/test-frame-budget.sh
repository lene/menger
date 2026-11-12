#!/usr/bin/env bash
# Tests frame_ms_from_log of scripts/frame-budget.sh offline, on canned logs.
# Run: scripts/test-frame-budget.sh
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPTS/frame-budget.sh"
set +e   # the sourced script's set -e would end this run at the first expected failure
FAILURES=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

report() { # report <description> <exit-status>
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); fi
}
frame_line() { # frame_line <timestamp> <k>
    echo "$1 INFO  menger.engines.WithAnimation - Rendering frame $2/3 (t=1.0)"
}
noise() { echo "12:00:00.000 INFO  menger.Main - Loading scene: x.scala"; }

expect_ms() { # expect_ms <description> <log> <ms>
    local out status=0
    out="$(frame_ms_from_log "$2")" || status=$?
    [ "$status" -eq 0 ] && [ "$out" = "$3" ] || status=1
    report "$1" "$status"
}
expect_cell() { # expect_cell <description> <log> <cell>
    local status=0
    [ "$(timeout_cell 1800 "$2")" = "$3" ] || status=1
    report "$1" "$status"
}
expect_failure() { # expect_failure <description> <log>
    local status=0
    frame_ms_from_log "$2" > /dev/null && status=1
    report "$1" "$status"
}

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

# Timed-out render: the cell names the last frame the log shows
{ noise; frame_line 10:00:01.000 1; } > "$WORK/t1.log"
expect_cell "timeout during frame 1" "$WORK/t1.log" "> 1800 s (timed out at frame 1/3)"
{ frame_line 10:00:01.000 1; frame_line 10:00:05.000 2; } > "$WORK/t2.log"
expect_cell "timeout during frame 2" "$WORK/t2.log" "> 1800 s (timed out at frame 2/3)"
expect_cell "timeout before any frame" "$WORK/none.log" "> 1800 s (timed out before frame 1)"

# Whole run with a launcher that logs frame 1 and then hangs: the row says so, the run ends
# with exit 0 for a mixed result, 1 when every t timed out.
if command -v xvfb-run > /dev/null; then
  HANG_BIN="$WORK/hang"; touch "$WORK/Scene.scala"
  printf '#!/bin/sh\necho "10:00:01.000 INFO  x - Rendering frame 1/3 (t=1.0)" >&2\nsleep 60\n' > "$HANG_BIN"
  chmod +x "$HANG_BIN"
  mkdir -p "$WORK/tex/assets/hdri"; touch "$WORK/tex/$HDRI_RELATIVE_PATH"
  status=0
  out="$(GPU_LOCK="$WORK/lock" "$SCRIPTS/frame-budget.sh" "$HANG_BIN" "$WORK/Scene.scala" \
    --at 1,2 --texture-dir "$WORK/tex" --timeout 1 2>&1)" || status=$?
  verdict=1
  if [ "$status" -eq 1 ] && grep -q "| Scene | 2 | > 1 s (timed out at frame 1/3) |" <<< "$out" \
    && grep -q "every t timed out" <<< "$out"; then verdict=0; fi
  report "all t time out: both rows listed, then exit 1" "$verdict"
fi

# Missing HDRI: exit 1 naming the fetch script
FAKE_BIN="$WORK/bin"; echo '#!/bin/sh' > "$FAKE_BIN"; chmod +x "$FAKE_BIN"
touch "$WORK/Scene.scala"; mkdir "$WORK/empty"
status=0
out="$("$SCRIPTS/frame-budget.sh" "$FAKE_BIN" "$WORK/Scene.scala" --at 1 \
  --texture-dir "$WORK/empty" 2>&1)" || status=$?
verdict=1
if [ "$status" -eq 1 ] && grep -q "scripts/fetch-kiara-hdri.sh 4k" <<< "$out"; then verdict=0; fi
report "no HDRI: exit 1, names fetch-kiara-hdri.sh 4k" "$verdict"

echo
if [ "$FAILURES" -eq 0 ]; then echo "All frame-budget tests passed"; else echo "$FAILURES failed"; exit 1; fi
