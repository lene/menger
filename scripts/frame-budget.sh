#!/usr/bin/env bash
# Frame-time harness for the dimension ladder's render budget (menger#159).
#
# Renders 3 animation frames around each requested t and takes the warm frame's cost from the
# app's own log: the "Rendering frame k/N" lines carry a timestamp, so frame 3's start minus
# frame 2's start is what a video render pays per frame. Frame 1 is cold (JIT, first upload)
# and is not used. The engine is not touched.
#
# What the warm frame contains depends on the scene. The 3 frames of one measurement are
# 1/30 s apart. A static benchmark moment (Moment3D/Moment4D quantise their level to 0.25
# steps) draws the same geometry in all three, the engine sees an unchanged geometry
# (FrameChange) and skips the rebuild: the warm frame is render + save only. A scene whose
# geometry changes from frame to frame (G5's) pays the scene build and the 4D projection in the
# warm frame as well, which is what a video render pays.
#
# Usage:
#   scripts/frame-budget.sh BIN SCENE (--at T1,T2,... | --uniform T STEP)
#       [--width W] [--height H] [--texture-dir DIR] [--hdri RELPATH] [--timeout SECONDS]
#       [--out FILE]
#
#   BIN            the staged launcher, menger-app/target/universal/stage/bin/menger-app
#   SCENE          a scene file with `def scene(t: Float)`, e.g. scripts/frame-budget/Moment3D.scala
#                  (a value ending in .scala must be an existing file), or a scene name that
#                  --scene resolves, e.g. examples.dsl.ladder.DimensionLadder (passed on unchecked)
#   --at           measure at these t (seconds; non-negative numbers)
#   --uniform T S  measure at t = i*S for i = 0, 1, ... while i*S < T, then print the mean and
#                  the budget for an animation of T seconds at 30 fps: mean ms/frame x T x 30
#                  frames, in hours
#   --width/height render size (positive integers, default 1920x1080)
#   --texture-dir  where the scene's env map is resolved (default: the menger checkout this
#                  script is in; a worktree has no assets/hdri, so pass the main checkout)
#   --hdri RELPATH the env map the scene uses, relative to --texture-dir, checked before the
#                  first render (default assets/hdri/kiara_2_sunrise_4k.hdr)
#   --timeout S    kill a render that runs longer than S seconds (positive integer, default
#                  1800); its row reads "> S s (timed out at frame k/3)" and the run goes on with
#                  the next t. A render that logged all 3 frames before the kill counts as
#                  measured. The time waiting for the GPU lock does not count
#   --out FILE     also append the markdown to FILE, starting with a provenance line: date, GPU,
#                  commit, resolution, timeout, scene, hdri
#
# Every render runs under flock /tmp/menger-gpu.lock (GPU_LOCK overrides it, for the tests), and
# never outlives it: on INT/TERM/HUP the render's process group is killed before this script
# exits. The lock's file descriptor is closed for the render's children.
# Exit 1: the HDRI is missing, a render fails (non-zero exit or an ERROR line in the app log,
# even if 3 frames were logged), the log has fewer than 3 frame lines, or every t timed out.
# Exit 2: bad arguments.
set -euo pipefail
# awk would print "1,066667" in a decimal-comma locale, which the app rejects as a Float.
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LOCK="${GPU_LOCK:-/tmp/menger-gpu.lock}"   # overridable for the offline tests
KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-30}"
DEFAULT_HDRI=assets/hdri/kiara_2_sunrise_4k.hdr
FRAMES_PER_MEASUREMENT=3
FPS=30
DEFAULT_WIDTH=1920
DEFAULT_HEIGHT=1080
DEFAULT_TIMEOUT_SECONDS=1800
MS_PER_DAY=86400000
MS_PER_HOUR=3600000
MS_PER_SECOND=1000
TIMEOUT_EXIT=124
KILLED_EXIT=137
INTERRUPTED_EXIT=130
INTERRUPT_GRACE_SECONDS=1
NUMBER_PATTERN='^[0-9]+(\.[0-9]+)?$'
POSITIVE_INTEGER_PATTERN='^[1-9][0-9]*$'
ERROR_LINE_PATTERN='^[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} +ERROR '
# Runs as the render's shell under flock: records the start time and its own pid, which timeout
# then keeps (exec) and makes the leader of a process group of its own.
# shellcheck disable=SC2016
RENDER_WRAPPER='date +%s%3N > "$1"; echo $$ > "$2"; shift 2; exec timeout "$@"'

RENDER_PID=""
RENDER_PGID_FILE=""

# Prints the warm frame's cost in ms from a log: the timestamp of the 3rd "Rendering frame"
# line minus that of the 2nd, modulo a day (the log pattern has no date, so a render across
# midnight wraps). Returns 1 when the log has fewer than 3 such lines.
frame_ms_from_log() {
  local log_file="$1"
  local stamps
  stamps="$(grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} .*Rendering frame [0-9]+/' "$log_file" \
    | head -n "$FRAMES_PER_MEASUREMENT" | cut -c1-12)"
  [ "$(printf '%s\n' "$stamps" | grep -c .)" -eq "$FRAMES_PER_MEASUREMENT" ] || return 1
  printf '%s\n' "$stamps" | awk -v day="$MS_PER_DAY" '
    { split($0, p, /[:.]/); ms[NR] = ((p[1] * 60 + p[2]) * 60 + p[3]) * 1000 + p[4] }
    END { print (ms[3] - ms[2] + day) % day }'
}

# Prints "k/N" of the last "Rendering frame" line in a log; nothing when there is none.
last_frame_from_log() {
  { grep -E 'Rendering frame [0-9]+/[0-9]+' "$1" || true; } | tail -n 1 \
    | sed -E 's|.*Rendering frame ([0-9]+/[0-9]+).*|\1|'
}

# The table cell of a render killed after $1 seconds, for the log $2.
timeout_cell() {
  local frame
  frame="$(last_frame_from_log "$2")"
  if [ -n "$frame" ]; then echo "> $1 s (timed out at frame $frame)"
  else echo "> $1 s (timed out before frame 1)"; fi
}

# Whether a render that ended with exit status $1 after $2 ms was stopped by the timeout of $3
# seconds: timeout's own 124, or a 137 (killed) that came only after the timeout had run out.
# A 137 before that is something else, the OOM killer for one, and is a failure.
stopped_by_timeout() {
  local status="$1" elapsed_ms="$2" timeout_seconds="$3"
  [ "$status" -eq "$TIMEOUT_EXIT" ] \
    || { [ "$status" -eq "$KILLED_EXIT" ] && [ "$elapsed_ms" -ge "$((timeout_seconds * MS_PER_SECOND))" ]; }
}

# True when the app logged an ERROR line: a frame that failed to build or render does not
# stop the animation, so a log with 3 frame lines can still hold no measurement.
log_has_error() { grep -qE "$ERROR_LINE_PATTERN" "$1"; }

fail() { echo "ERROR: $*" >&2; exit 1; }

die_usage() {
  echo "ERROR: $*" >&2
  echo "Usage: see the header of scripts/frame-budget.sh, or run it with --help" >&2
  exit 2
}

usage() { # usage <exit status>
  sed -n '2,/^set -e/p' "${BASH_SOURCE[0]}" | sed '$d;s/^# \{0,1\}//' >&2
  exit "${1:-2}"
}

is_number() { [[ "$1" =~ $NUMBER_PATTERN ]]; }
is_positive_number() { is_number "$1" && awk -v n="$1" 'BEGIN { exit !(n > 0) }'; }
is_positive_integer() { [[ "$1" =~ $POSITIVE_INTEGER_PATTERN ]]; }

# Kills the render's process group: TERM, a moment to stop, then KILL, and waits for the lock
# holder (flock) to be gone. A render must never outlive the lock it ran under.
terminate_render() {
  local pgid=""
  [ -z "$RENDER_PGID_FILE" ] || pgid="$(cat "$RENDER_PGID_FILE" 2> /dev/null || true)"
  if [ -n "$pgid" ]; then kill -TERM -- "-$pgid" 2> /dev/null || true; fi
  sleep "$INTERRUPT_GRACE_SECONDS"
  if [ -n "$pgid" ]; then kill -KILL -- "-$pgid" 2> /dev/null || true; fi
  if [ -n "$RENDER_PID" ]; then
    kill "$RENDER_PID" 2> /dev/null || true
    wait "$RENDER_PID" 2> /dev/null || true
  fi
}

on_interrupt() {
  trap '' INT TERM HUP
  terminate_render
  exit "$INTERRUPTED_EXIT"
}

# Kills what is left of a finished render (a child that outlived it).
sweep_render_group() {
  local pgid
  pgid="$(cat "$RENDER_PGID_FILE" 2> /dev/null || true)"
  if [ -n "$pgid" ]; then kill -KILL -- "-$pgid" 2> /dev/null || true; fi
}

# Renders 3 frames from t. Sets CELL, the table cell, and MS, the warm frame's ms (empty when
# the render timed out before logging 3 frames). Exits 1 on any failure.
measure_ms() {
  local t="$1"
  local end_t log_file started_file status=0 started_ms elapsed_ms
  end_t="$(awk -v t="$t" -v n="$FRAMES_PER_MEASUREMENT" -v fps="$FPS" \
    'BEGIN { printf "%.6f", t + (n - 1) / fps }')"
  log_file="$WORK/t_${t}.log"
  started_file="$WORK/render.started"
  RENDER_PGID_FILE="$WORK/render.pgid"
  rm -f "$started_file" "$RENDER_PGID_FILE"
  # -o: the lock fd is not inherited by the app's children (Xvfb outlives a killed render and
  # would hold the lock); flock itself keeps the lock until the command exits. Run in the
  # background and waited for, so that a trapped signal is handled at once.
  flock -o "$GPU_LOCK" sh -c "$RENDER_WRAPPER" sh "$started_file" "$RENDER_PGID_FILE" \
    -k "$KILL_GRACE_SECONDS" "$TIMEOUT_SECONDS" \
    env __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a "$BIN" --headless \
    --scene "$SCENE" --texture-dir "$TEXTURE_DIR" --frames "$FRAMES_PER_MEASUREMENT" \
    --start-t "$t" --end-t "$end_t" --save-name "$WORK/f_%03d.png" \
    --width "$WIDTH" --height "$HEIGHT" > "$log_file" 2>&1 &
  RENDER_PID=$!
  wait "$RENDER_PID" || status=$?
  sweep_render_group
  RENDER_PID=""
  started_ms="$(cat "$started_file" 2> /dev/null || date +%s%3N)"
  elapsed_ms=$(( $(date +%s%3N) - started_ms ))

  if log_has_error "$log_file"; then
    cat "$log_file" >&2
    fail "the app logged an ERROR rendering $SCENE at t=$t (log above)"
  fi
  MS=""
  if stopped_by_timeout "$status" "$elapsed_ms" "$TIMEOUT_SECONDS"; then
    MS="$(frame_ms_from_log "$log_file" || true)"
    if [ -n "$MS" ]; then CELL="$MS"; else CELL="$(timeout_cell "$TIMEOUT_SECONDS" "$log_file")"; fi
    return 0
  fi
  [ "$status" -eq 0 ] \
    || { cat "$log_file" >&2; fail "render of $SCENE at t=$t failed with exit status $status (log above)"; }
  MS="$(frame_ms_from_log "$log_file")" \
    || { cat "$log_file" >&2; fail "fewer than $FRAMES_PER_MEASUREMENT 'Rendering frame' lines for t=$t (log above)"; }
  CELL="$MS"
}

emit() { echo "$1"; if [ -n "$OUT" ]; then echo "$1" >> "$OUT"; fi; }

# Sample times of --uniform: i*STEP for i = 0, 1, ... while i*STEP < TOTAL. By index, so the
# float error does not accumulate.
uniform_times() { # uniform_times <total> <step>
  awk -v total="$1" -v step="$2" \
    'BEGIN { for (i = 0; i * step < total - 1e-9; i++) printf "%.10g\n", i * step }'
}

need_values() { # need_values <count> <flag> <args after the flag...>
  local count="$1" flag="$2"
  shift 2
  [ "$#" -ge "$count" ] || die_usage "$flag needs $count value(s)"
}

set_times_at() {
  local entry
  local -a entries=()
  IFS=, read -ra entries <<< "$1" || true
  [ "${#entries[@]}" -gt 0 ] || die_usage "no sample times given"
  for entry in "${entries[@]}"; do
    is_number "$entry" || die_usage "bad sample time '$entry' in --at (a non-negative number is expected)"
  done
  TIMES=("${entries[@]}")
}

set_times_uniform() { # set_times_uniform <total> <step>
  is_positive_number "$1" || die_usage "--uniform T must be a positive number, got '$1'"
  is_positive_number "$2" || die_usage "--uniform STEP must be a positive number, got '$2'"
  TOTAL="$1"
  mapfile -t TIMES < <(uniform_times "$1" "$2")
  [ "${#TIMES[@]}" -gt 0 ] || die_usage "no sample times given"
}

parse_args() {
  case "${1:-}" in -h | --help) usage 0 ;; esac
  [ "$#" -ge 2 ] || usage 2
  BIN="$1"; SCENE="$2"; shift 2
  MODE=""; TOTAL=""; TIMES=()
  WIDTH=$DEFAULT_WIDTH; HEIGHT=$DEFAULT_HEIGHT; OUT=""; HDRI=$DEFAULT_HDRI
  TIMEOUT_SECONDS=$DEFAULT_TIMEOUT_SECONDS
  TEXTURE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --at)
        need_values 1 "$@"; [ -z "$MODE" ] || die_usage "give either --at or --uniform, once"
        MODE="at"; set_times_at "$2"; shift 2 ;;
      --uniform)
        need_values 2 "$@"; [ -z "$MODE" ] || die_usage "give either --at or --uniform, once"
        MODE="uniform"; set_times_uniform "$2" "$3"; shift 3 ;;
      --width)
        need_values 1 "$@"; is_positive_integer "$2" || die_usage "--width must be a positive integer, got '$2'"
        WIDTH="$2"; shift 2 ;;
      --height)
        need_values 1 "$@"; is_positive_integer "$2" || die_usage "--height must be a positive integer, got '$2'"
        HEIGHT="$2"; shift 2 ;;
      --timeout)
        need_values 1 "$@"; is_positive_integer "$2" || die_usage "--timeout must be a positive integer (seconds), got '$2'"
        TIMEOUT_SECONDS="$2"; shift 2 ;;
      --texture-dir) need_values 1 "$@"; TEXTURE_DIR="$2"; shift 2 ;;
      --hdri)
        need_values 1 "$@"
        case "$2" in /* | *..*) die_usage "--hdri must be a relative path without '..', got '$2'" ;; esac
        HDRI="$2"; shift 2 ;;
      --out) need_values 1 "$@"; OUT="$2"; shift 2 ;;
      *) die_usage "unknown option: $1" ;;
    esac
  done
  [ -n "$MODE" ] || die_usage "give --at or --uniform"
  if [ -n "$OUT" ]; then : >> "$OUT" 2> /dev/null || die_usage "--out file is not writable: $OUT"; fi
}

provenance_line() {
  local gpu commit
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -n 1)" || true
  [ -n "$gpu" ] || gpu="unknown"
  commit="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2> /dev/null)" || commit=unknown
  echo "Frame budget $(date '+%Y-%m-%d %H:%M'); GPU: $gpu; menger commit: $commit;" \
    "${WIDTH}x${HEIGHT}; timeout $TIMEOUT_SECONDS s; scene: $SCENE_NAME; hdri: $HDRI"
}

main() {
  parse_args "$@"
  [ -x "$BIN" ] || fail "launcher not found or not executable: $BIN (run: sbt mengerApp/stage)"
  case "$SCENE" in
    *.scala) [ -f "$SCENE" ] || fail "scene file not found: $SCENE" ;;
  esac
  [ -f "$TEXTURE_DIR/$HDRI" ] \
    || fail "$TEXTURE_DIR/$HDRI is missing; run scripts/fetch-kiara-hdri.sh 4k"

  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  trap on_interrupt INT TERM HUP
  local scene_name results="" t measured=0 timed_out=0
  case "$SCENE" in
    *.scala) scene_name="$(basename "$SCENE" .scala)" ;;
    *) scene_name="${SCENE##*.}" ;;
  esac
  SCENE_NAME="$scene_name"

  emit "$(provenance_line)"
  emit ""
  emit "| scene | t | warm ms/frame |"
  emit "|---|---|---|"
  for t in "${TIMES[@]}"; do
    measure_ms "$t"
    emit "| $scene_name | $t | $CELL |"
    if [ -n "$MS" ]; then
      results+="$MS"$'\n'; measured=$((measured + 1))
    else
      echo "t=$t timed out after $TIMEOUT_SECONDS s" >&2; timed_out=$((timed_out + 1))
    fi
  done
  [ "$measured" -gt 0 ] || fail "every t timed out after $TIMEOUT_SECONDS s"

  if [ "$MODE" = uniform ] && [ "$timed_out" -gt 0 ]; then
    emit ""
    emit "No budget: $timed_out of $((measured + timed_out)) samples timed out."
  elif [ "$MODE" = uniform ]; then
    local mean hours
    mean="$(printf '%s' "$results" | awk '{ s += $1 } END { printf "%.1f", s / NR }')"
    hours="$(awk -v m="$mean" -v t="$TOTAL" -v fps="$FPS" -v h="$MS_PER_HOUR" \
      'BEGIN { printf "%.2f", m * t * fps / h }')"
    emit ""
    emit "Mean: $mean ms/frame. Budget for $TOTAL s at $FPS fps ($(awk -v t="$TOTAL" -v f="$FPS" 'BEGIN { print t * f }') frames): $hours h."
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
