#!/usr/bin/env bash
# Shared by frame-budget.sh, experiment-sheet.sh and render-cycle.sh: rendering one app
# invocation under the GPU lock, with a timeout, in a process group of its own that never
# outlives the lock, plus the argument checks and the error helpers those scripts share.
# shellcheck disable=SC2034,SC2153  # results and settings are read by the sourcing scripts
# Sourced, never run. It sets no shell options; the sourcing script owns set -e.
#
# The sourcing script provides before the first render:
#   RENDER_STATE_DIR  a writable directory for the render's bookkeeping files
#   and, for the interrupt handling, `trap on_interrupt INT TERM HUP`.
# GPU_LOCK and KILL_GRACE_SECONDS may be overridden in the environment (the offline tests do).

# awk would print "1,066667" in a decimal-comma locale, which the app rejects as a Float.
export LC_ALL=C

RENDER_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LOCK="${GPU_LOCK:-/tmp/menger-gpu.lock}"   # overridable for the offline tests
KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-30}"
DEFAULT_HDRI=assets/hdri/kiara_2_sunrise_4k.hdr
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
RENDER_STATUS=""
RENDER_ELAPSED_MS=""

# Whether a render that ended with exit status $1 after $2 ms was stopped by the timeout of $3
# seconds: timeout's own 124, or a 137 (killed) that came only after the timeout had run out.
# A 137 before that is something else, the OOM killer for one, and is a failure.
stopped_by_timeout() {
  local status="$1" elapsed_ms="$2" timeout_seconds="$3"
  [ "$status" -eq "$TIMEOUT_EXIT" ] \
    || { [ "$status" -eq "$KILLED_EXIT" ] && [ "$elapsed_ms" -ge "$((timeout_seconds * MS_PER_SECOND))" ]; }
}

# True when the app logged an ERROR line: a frame that failed to build or render does not
# stop the animation, so a log with all its frame lines can still hold no result.
log_has_error() { grep -qE "$ERROR_LINE_PATTERN" "$1"; }

fail() { echo "ERROR: $*" >&2; exit 1; }

# SCRIPT_NAME is the sourcing script's file name, for the hint.
die_usage() {
  echo "ERROR: $*" >&2
  echo "Usage: see the header of scripts/$SCRIPT_NAME, or run it with --help" >&2
  exit 2
}

# usage <exit status>: prints the sourcing script's header comment (SCRIPT_FILE is its path).
usage() {
  sed -n '2,/^set -e/p' "$SCRIPT_FILE" | sed '$d;s/^# \{0,1\}//' >&2
  exit "${1:-2}"
}

is_number() { [[ "$1" =~ $NUMBER_PATTERN ]]; }
is_positive_number() { is_number "$1" && awk -v n="$1" 'BEGIN { exit !(n > 0) }'; }
is_positive_integer() { [[ "$1" =~ $POSITIVE_INTEGER_PATTERN ]]; }

need_values() { # need_values <count> <flag> <args after the flag...>
  local count="$1" flag="$2"
  shift 2
  [ "$#" -ge "$count" ] || die_usage "$flag needs $count value(s)"
}

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

# run_render <timeout seconds> <log file> <launcher> <app arguments...>
# Runs the app headless under the GPU lock and the timeout, its output in the log. Sets
# RENDER_STATUS (the exit status) and RENDER_ELAPSED_MS (wall clock from the moment the lock
# was taken). Judging the outcome (timeout, ERROR line, missing output) is the caller's.
run_render() {
  local timeout_seconds="$1" log_file="$2" launcher="$3"
  local started_file="$RENDER_STATE_DIR/render.started" started_ms status=0
  shift 3
  RENDER_PGID_FILE="$RENDER_STATE_DIR/render.pgid"
  rm -f "$started_file" "$RENDER_PGID_FILE"
  # -o: the lock fd is not inherited by the app's children (Xvfb outlives a killed render and
  # would hold the lock); flock itself keeps the lock until the command exits. Run in the
  # background and waited for, so that a trapped signal is handled at once.
  flock -o "$GPU_LOCK" sh -c "$RENDER_WRAPPER" sh "$started_file" "$RENDER_PGID_FILE" \
    -k "$KILL_GRACE_SECONDS" "$timeout_seconds" \
    env __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a "$launcher" --headless "$@" \
    > "$log_file" 2>&1 &
  RENDER_PID=$!
  wait "$RENDER_PID" || status=$?
  sweep_render_group
  RENDER_PID=""
  started_ms="$(cat "$started_file" 2> /dev/null || date +%s%3N)"
  RENDER_ELAPSED_MS=$(( $(date +%s%3N) - started_ms ))
  RENDER_STATUS="$status"
}

# One line of provenance for a report: date, GPU, commit, size, timeout, scene(s), hdri.
# Reads WIDTH, HEIGHT, TIMEOUT_SECONDS, SCENE_NAME and HDRI; PROVENANCE_LABEL names the report.
provenance_line() {
  local gpu commit
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -n 1)" || true
  [ -n "$gpu" ] || gpu="unknown"
  commit="$(git -C "$RENDER_LIB_DIR" rev-parse --short HEAD 2> /dev/null)" || commit=unknown
  echo "${PROVENANCE_LABEL:-Frame budget} $(date '+%Y-%m-%d %H:%M'); GPU: $gpu; menger commit: $commit;" \
    "${WIDTH}x${HEIGHT}; timeout $TIMEOUT_SECONDS s; scene: $SCENE_NAME; hdri: $HDRI"
}
