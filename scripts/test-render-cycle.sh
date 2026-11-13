#!/usr/bin/env bash
# Tests scripts/render-cycle.sh offline: argument checks, and whole runs with a fake launcher
# instead of the app (no GPU; the runs need xvfb-run, ImageMagick and a private lock).
# Run: scripts/test-render-cycle.sh
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPTS/render-cycle.sh"
# shellcheck source=/dev/null
source "$SCRIPT"   # for DEFAULT_HDRI; main does not run
set +e             # the sourced script's set -e would end this run at the first expected failure
FAILURES=0

TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT

report() { # report <description> <exit-status>
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); fi
}
holds() { "$@"; }   # a command to chain with &&, so that $? is the chain
expect_equal() { # expect_equal <description> <actual> <expected>
    local status=0
    [ "$2" = "$3" ] || { status=1; echo "  expected '$3', got '$2'"; }
    report "$1" "$status"
}

FAKE="$TEST_DIR/fake"
echo '#!/bin/sh' > "$FAKE"; chmod +x "$FAKE"
mkdir -p "$TEST_DIR/tex/assets/hdri"
touch "$TEST_DIR/tex/$DEFAULT_HDRI"
COMMON=(--texture-dir "$TEST_DIR/tex" --width 16 --height 9)

# ---- argument checks (exit 2, the message names the problem) ----

expect_usage_error() { # expect_usage_error <description> <message fragment> <args after BIN SCENE...>
    local description="$1" fragment="$2" out status=0
    shift 2
    out="$("$SCRIPT" "$FAKE" Scene "${COMMON[@]}" "$@" 2>&1)" || status=$?
    if [ "$status" -eq 2 ] && grep -qF -- "$fragment" <<< "$out"; then status=0; else status=1; fi
    report "$description" "$status"
}
expect_usage_error "no --seconds" "give --seconds" --out "$TEST_DIR/o"
expect_usage_error "no --out" "give --out" --seconds 1
expect_usage_error "--seconds 0" "--seconds must be a positive number" --out "$TEST_DIR/o" --seconds 0
expect_usage_error "--seconds abc" "--seconds must be a positive number" --out "$TEST_DIR/o" --seconds abc
expect_usage_error "--fps 0" "--fps must be a positive number" --out "$TEST_DIR/o" --seconds 1 --fps 0
expect_usage_error "--chunk 0" "--chunk must be a positive integer" --out "$TEST_DIR/o" --seconds 1 --chunk 0
expect_usage_error "--chunk 2.5" "--chunk must be a positive integer" --out "$TEST_DIR/o" --seconds 1 --chunk 2.5
expect_usage_error "--timeout-per-frame 0" "--timeout-per-frame must be a positive integer" \
    --out "$TEST_DIR/o" --seconds 1 --timeout-per-frame 0
expect_usage_error "less than one frame" "is less than one frame" --out "$TEST_DIR/o" --seconds 0.01 --fps 10
expect_usage_error "unknown option" "unknown option: --bogus" --bogus

# ---- missing prerequisites (exit 1, before any render) ----

run_script() { # run_script <args...>: sets RUN_OUT and RC
    RUN_OUT="$("$SCRIPT" "$@" 2>&1)"; RC=$?
}
run_script "$TEST_DIR/no-such-launcher" Scene --seconds 1 --out "$TEST_DIR/o" "${COMMON[@]}"
[ "$RC" -eq 1 ] && grep -qF "launcher not found or not executable" <<< "$RUN_OUT"
report "a missing launcher: exit 1" $?
run_script "$FAKE" Scene --seconds 1 --out "$TEST_DIR/o" --texture-dir "$TEST_DIR/tex" --hdri assets/hdri/other.hdr
[ "$RC" -eq 1 ] && grep -qF "assets/hdri/other.hdr is missing" <<< "$RUN_OUT"
report "a missing HDRI: exit 1, names it" $?
run_script "$FAKE" "$TEST_DIR/Missing.scala" --seconds 1 --out "$TEST_DIR/o" "${COMMON[@]}"
[ "$RC" -eq 1 ] && grep -qF "scene file not found" <<< "$RUN_OUT"
report "a .scala file that does not exist: exit 1" $?
if ! command -v ffmpeg > /dev/null; then
    run_script "$FAKE" Scene --seconds 1 --out "$TEST_DIR/o" "${COMMON[@]}" --video "$TEST_DIR/v.mp4"
    [ "$RC" -eq 1 ] && grep -qF "ffmpeg is not installed" <<< "$RUN_OUT"
    report "--video without ffmpeg: exit 1" $?
fi
touch "$TEST_DIR/file-not-dir"
run_script "$FAKE" Scene --seconds 1 --out "$TEST_DIR/file-not-dir/o" "${COMMON[@]}"
[ "$RC" -eq 1 ] && grep -qF "is not writable" <<< "$RUN_OUT"
report "an --out that cannot be created: exit 1, not writable" $?
[ ! -e "$TEST_DIR/o" ]; report "nothing was created before the checks passed" $?

# ---- whole runs with a fake launcher ----

# Writes --frames PNGs with the local numbering to --save-name and appends "start end frames" to
# a call log. FAKE_MODE acts on the call whose --start-t is FAKE_BAD_START: error (an ERROR line,
# exit 0), short (all frames but the last, exit 0), hang.
cat > "$FAKE" <<'EOF'
#!/bin/sh
frames=""; start=""; end=""; pattern=""
while [ $# -gt 0 ]; do
  case "$1" in
    --frames) frames="$2" ;; --start-t) start="$2" ;; --end-t) end="$2" ;; --save-name) pattern="$2" ;;
  esac
  shift
done
echo "$start $end $frames" >> "$FAKE_DIR/calls"
write=$frames
if [ "$start" = "$FAKE_BAD_START" ]; then
  case "$FAKE_MODE" in
    error) echo "10:00:00.000 ERROR x.WithAnimation - frame failed" >&2 ;;
    short) write=$((frames - 1)) ;;
    hang) sleep 60 & echo $! > "$FAKE_DIR/sleep.pid"; wait ;;
  esac
fi
i=0
while [ "$i" -lt "$write" ]; do
  cp "$FAKE_DIR/base.png" "$(printf "$pattern" "$i")"
  i=$((i + 1))
done
EOF
chmod +x "$FAKE"

calls() { wc -l < "$TEST_DIR/calls" 2> /dev/null || echo 0; }
call_starts() { cut -d' ' -f1 "$TEST_DIR/calls" | paste -sd' '; }
frame_names() { (cd "$1/frames" && ls) | paste -sd' '; }
mtimes() { stat -c %y "$1"/frames/f_0000[0-3].png "$1"/frames/f_0000[89].png; }

run_cycle() { # run_cycle <out dir> <args...>: sets RUN_OUT and RC
    local out_dir="$1"; shift
    RUN_OUT="$("$SCRIPT" "$FAKE" Scene --out "$out_dir" "${COMMON[@]}" "$@" 2>&1)"
    RC=$?
}

missing_tools=""
for tool in xvfb-run convert; do
    command -v "$tool" > /dev/null || missing_tools+=" $tool"
done
if [ -n "$missing_tools" ]; then
    echo "SKIP: missing tools:$missing_tools: the whole-run tests with a fake launcher"
else
    export FAKE_DIR="$TEST_DIR" GPU_LOCK="$TEST_DIR/lock" KILL_GRACE_SECONDS=1
    convert -size 8x8 xc:red "$TEST_DIR/base.png"
    GOOD="$TEST_DIR/good"
    EXPECTED_NAMES="f_00000.png f_00001.png f_00002.png f_00003.png f_00004.png f_00005.png f_00006.png f_00007.png f_00008.png f_00009.png"

    run_cycle "$GOOD" --seconds 1 --fps 10 --chunk 4
    expect_equal "T=1 s, 10 fps, chunk 4: exit 0" "$RC" 0
    expect_equal "...three calls" "$(calls)" 3
    expect_equal "...f_00000.png to f_00009.png exist, nothing else" "$(frame_names "$GOOD")" "$EXPECTED_NAMES"
    expect_equal "...the calls start at t = 0, 0.4 and 0.8" "$(call_starts)" "0.000000 0.400000 0.800000"
    expect_equal "...the last call ends at t = 0.9, with 2 frames" "$(tail -n 1 "$TEST_DIR/calls")" "0.800000 0.900000 2"
    holds [ ! -d "$GOOD/work/chunk_0" ] && holds [ -f "$GOOD/logs/chunk_0.log" ] && holds [ -f "$GOOD/logs/chunk_2.log" ]
    report "...work directories are gone, the chunk logs are kept" $?
    grep -q "progress: 10 of 10 frames" <<< "$RUN_OUT"
    report "...a progress line after every chunk" $?
    grep -qF "ffmpeg -y -framerate 10 -i $GOOD/frames/f_%05d.png -c:v libx264 -pix_fmt yuv420p" <<< "$RUN_OUT"
    report "...without --video the ffmpeg command is printed" $?

    run_cycle "$GOOD" --seconds 1 --fps 10 --chunk 4
    expect_equal "a rerun: exit 0" "$RC" 0
    expect_equal "...makes no call" "$(calls)" 3

    BEFORE="$(mtimes "$GOOD")"
    rm -f "$GOOD/frames/f_00005.png"
    run_cycle "$GOOD" --seconds 1 --fps 10 --chunk 4
    expect_equal "f_00005.png deleted, rerun: exit 0" "$RC" 0
    expect_equal "...exactly one more call" "$(calls)" 4
    expect_equal "...for chunk 1 (t = 0.4 to 0.7)" "$(tail -n 1 "$TEST_DIR/calls")" "0.400000 0.700000 4"
    expect_equal "...f_00005.png is back" "$(frame_names "$GOOD")" "$EXPECTED_NAMES"
    expect_equal "...frames of chunks 0 and 2 keep their modification times" "$(mtimes "$GOOD")" "$BEFORE"

    : > "$TEST_DIR/calls"
    BAD="$TEST_DIR/bad_error"
    FAKE_MODE=error FAKE_BAD_START=0.400000 run_cycle "$BAD" --seconds 1 --fps 10 --chunk 4
    expect_equal "an ERROR line in chunk 1: exit status 1" "$RC" 1
    expect_equal "...frames of chunk 0 exist" "$(frame_names "$BAD")" "f_00000.png f_00001.png f_00002.png f_00003.png"
    holds [ -d "$BAD/work/chunk_1" ] && holds [ -f "$BAD/logs/chunk_1.log" ] && holds [ -f "$BAD/work/chunk_1/l_00000.png" ]
    report "...work/chunk_1, its frames and its log are still there" $?
    grep -qF "chunk 1" <<< "$RUN_OUT" && grep -qF "frames 4 to 7" <<< "$RUN_OUT" \
        && grep -qF "logs/chunk_1.log" <<< "$RUN_OUT"
    report "...the message names the chunk, its frame range and the log" $?
    expect_equal "...chunk 2 was not tried" "$(calls)" 2

    run_cycle "$BAD" --seconds 1 --fps 10 --chunk 4
    expect_equal "the rerun after fixing the cause: exit 0" "$RC" 0
    expect_equal "...renders chunks 1 and 2 only" "$(calls)" 4
    expect_equal "...all frames exist" "$(frame_names "$BAD")" "$EXPECTED_NAMES"
    holds [ ! -d "$BAD/work/chunk_1" ] && ls -d "$BAD"/work/chunk_1.old.* > /dev/null 2>&1
    report "...the failed work directory was set aside, not deleted" $?

    : > "$TEST_DIR/calls"
    SHORT="$TEST_DIR/bad_short"
    FAKE_MODE=short FAKE_BAD_START=0.400000 run_cycle "$SHORT" --seconds 1 --fps 10 --chunk 4
    expect_equal "3 of 4 frames in chunk 1, exit 0: exit status 1" "$RC" 1
    grep -qF "first missing local frame 3 (global frame 7)" <<< "$RUN_OUT"
    report "...the message names the first missing frame" $?
    expect_equal "...no global frame of chunk 1" "$(frame_names "$SHORT")" "f_00000.png f_00001.png f_00002.png f_00003.png"

    HANG="$TEST_DIR/bad_hang"
    FAKE_MODE=hang FAKE_BAD_START=0.000000 run_cycle "$HANG" --seconds 0.1 --fps 10 --chunk 1 --timeout-per-frame 1
    expect_equal "a chunk that outlives its timeout: exit status 1" "$RC" 1
    grep -qF "timed out after 1 s" <<< "$RUN_OUT"
    report "...the message says so" $?
    ! kill -0 "$(cat "$TEST_DIR/sleep.pid")" 2> /dev/null; report "...the hung launcher's child is gone" $?

    # An interrupted run kills the render, keeps what exists and exits 130; the lock is free.
    INT="$TEST_DIR/interrupted"
    rm -f "$TEST_DIR/sleep.pid"
    FAKE_MODE=hang FAKE_BAD_START=0.400000 "$SCRIPT" "$FAKE" Scene --out "$INT" "${COMMON[@]}" \
        --seconds 1 --fps 10 --chunk 4 > /dev/null 2>&1 &
    SCRIPT_PID=$!
    for _ in $(seq 1 100); do [ -s "$TEST_DIR/sleep.pid" ] && break; sleep 0.1; done
    kill -TERM "$SCRIPT_PID"
    wait "$SCRIPT_PID"; STATUS=$?
    expect_equal "TERM during chunk 1: the script exits 130" "$STATUS" 130
    ! kill -0 "$(cat "$TEST_DIR/sleep.pid")" 2> /dev/null; report "...the render is gone with it" $?
    flock -n "$TEST_DIR/lock" true; report "...the GPU lock is free" $?
    expect_equal "...the frames of chunk 0 are kept" "$(frame_names "$INT")" "f_00000.png f_00001.png f_00002.png f_00003.png"
    holds [ -d "$INT/work/chunk_1" ]; report "...the work directory of chunk 1 is kept" $?

    : > "$TEST_DIR/calls"
    ONE="$TEST_DIR/one"
    run_cycle "$ONE" --seconds 0.5 --fps 10 --chunk 4
    expect_equal "T=0.5 s, 10 fps, chunk 4: exit 0" "$RC" 0
    expect_equal "...a chunk of 4 and a chunk of 1" "$(awk '{ print $3 }' "$TEST_DIR/calls" | paste -sd' ')" "4 1"
    expect_equal "...the single-frame chunk has --start-t equal to --end-t" \
        "$(tail -n 1 "$TEST_DIR/calls")" "0.400000 0.400000 1"

    if command -v ffmpeg > /dev/null; then
        run_cycle "$GOOD" --seconds 1 --fps 10 --chunk 4 --video "$TEST_DIR/out.mp4"
        holds [ "$RC" -eq 0 ] && holds [ -s "$TEST_DIR/out.mp4" ]
        report "--video out.mp4 with all frames present: the file exists" $?
    else
        echo "SKIP: ffmpeg is not installed: the --video case"
    fi
fi

echo
if [ "$FAILURES" -eq 0 ]; then echo "All render-cycle tests passed"; else echo "$FAILURES failed"; exit 1; fi
