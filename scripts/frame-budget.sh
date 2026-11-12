#!/usr/bin/env bash
# Frame-time harness for the dimension ladder's render budget (menger#159).
#
# Renders 3 animation frames around each requested t and takes the warm frame's cost from the
# app's own log: the "Rendering frame k/N" lines carry a timestamp, so frame 3's start minus
# frame 2's start is what a video render pays per frame (scene build, 4D projection, render and
# save). Frame 1 is cold (JIT, first upload) and is not used. The engine is not touched.
#
# Usage:
#   scripts/frame-budget.sh BIN SCENE (--at T1,T2,... | --uniform T STEP)
#       [--width W] [--height H] [--texture-dir DIR] [--timeout SECONDS] [--out FILE]
#
#   BIN            the staged launcher, menger-app/target/universal/stage/bin/menger-app
#   SCENE          a scene file with `def scene(t: Float)`, e.g. scripts/frame-budget/Moment3D.scala
#   --at           measure at these t (seconds)
#   --uniform T S  measure at t = 0, S, 2S, ... below T, then print the mean and the budget for
#                  an animation of T seconds at 30 fps: mean ms/frame x T x 30 frames, in hours
#   --width/height render size (default 1920x1080)
#   --texture-dir  where the scene's env map is resolved (default: the menger checkout this
#                  script is in; a worktree has no assets/hdri, so pass the main checkout)
#   --timeout S    kill a render that runs longer than S seconds (default 1800); its row reads
#                  "> S s (timed out at frame k/3)" and the run goes on with the next t. The
#                  time waiting for the GPU lock does not count
#   --out FILE     also append the markdown to FILE
#
# Every render runs under flock /tmp/menger-gpu.lock (GPU_LOCK overrides it, for the tests): the GPU is shared with other tasks.
# Exit 1 when the HDRI is missing, a render fails, the log has fewer than 3 frame lines, or
# every t timed out.
set -euo pipefail
# awk would print "1,066667" in a decimal-comma locale, which the app rejects as a Float.
export LC_ALL=C

GPU_LOCK="${GPU_LOCK:-/tmp/menger-gpu.lock}"   # overridable for the offline tests
HDRI_RELATIVE_PATH=assets/hdri/kiara_2_sunrise_4k.hdr
FRAMES_PER_MEASUREMENT=3
FPS=30
DEFAULT_WIDTH=1920
DEFAULT_HEIGHT=1080
MS_PER_DAY=86400000
MS_PER_HOUR=3600000
DEFAULT_TIMEOUT_SECONDS=1800
KILL_GRACE_SECONDS=30
TIMEOUT_EXIT=124

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

fail() { echo "ERROR: $*" >&2; exit 1; }

# Renders 3 frames from t and prints the warm frame's ms. $1 = t. When the render times out it
# prints the table cell for that instead and returns $TIMEOUT_EXIT.
measure_ms() {
  local t="$1"
  local end_t log_file
  end_t="$(awk -v t="$t" -v n="$FRAMES_PER_MEASUREMENT" -v fps="$FPS" \
    'BEGIN { printf "%.6f", t + (n - 1) / fps }')"
  log_file="$WORK/t_${t}.log"
  local status=0
  flock "$GPU_LOCK" timeout -k "$KILL_GRACE_SECONDS" "$TIMEOUT_SECONDS" \
    env __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a "$BIN" --headless \
    --scene "$SCENE" --texture-dir "$TEXTURE_DIR" --frames "$FRAMES_PER_MEASUREMENT" \
    --start-t "$t" --end-t "$end_t" --save-name "$WORK/f_%03d.png" \
    --width "$WIDTH" --height "$HEIGHT" > "$log_file" 2>&1 || status=$?
  if [ "$status" -eq "$TIMEOUT_EXIT" ] || [ "$status" -eq "$((128 + 9))" ]; then
    timeout_cell "$TIMEOUT_SECONDS" "$log_file"
    return "$TIMEOUT_EXIT"
  fi
  [ "$status" -eq 0 ] \
    || { cat "$log_file" >&2; fail "render of $SCENE at t=$t failed (log above)"; }
  frame_ms_from_log "$log_file" \
    || { cat "$log_file" >&2; fail "fewer than $FRAMES_PER_MEASUREMENT 'Rendering frame' lines for t=$t (log above)"; }
}

usage() { sed -n '2,/^set -e/p' "$0" | sed '$d;s/^# \{0,1\}//' >&2; exit 2; }

emit() { echo "$1"; if [ -n "$OUT" ]; then echo "$1" >> "$OUT"; fi; }

uniform_times() { # uniform_times <total> <step>
  awk -v total="$1" -v step="$2" 'BEGIN { for (t = 0; t < total; t += step) printf "%g\n", t }'
}

main() {
  [ "$#" -ge 3 ] || usage
  BIN="$1"; SCENE="$2"; shift 2
  local mode="" times="" total=""
  WIDTH=$DEFAULT_WIDTH; HEIGHT=$DEFAULT_HEIGHT; OUT=""; TIMEOUT_SECONDS=$DEFAULT_TIMEOUT_SECONDS
  TEXTURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --at) mode="at"; times="${2:?--at needs a list}"; shift 2 ;;
      --uniform)
        mode="uniform"; total="${2:?--uniform needs T}"
        times="$(uniform_times "$total" "${3:?--uniform needs STEP}")"; shift 3 ;;
      --width) WIDTH="${2:?}"; shift 2 ;;
      --height) HEIGHT="${2:?}"; shift 2 ;;
      --texture-dir) TEXTURE_DIR="${2:?}"; shift 2 ;;
      --timeout) TIMEOUT_SECONDS="${2:?}"; shift 2 ;;
      --out) OUT="${2:?}"; shift 2 ;;
      *) usage ;;
    esac
  done
  [ -n "$mode" ] || usage
  [ -x "$BIN" ] || fail "launcher not found or not executable: $BIN (run: sbt mengerApp/stage)"
  [ -f "$SCENE" ] || fail "scene file not found: $SCENE"
  [ -f "$TEXTURE_DIR/$HDRI_RELATIVE_PATH" ] \
    || fail "$TEXTURE_DIR/$HDRI_RELATIVE_PATH is missing; run scripts/fetch-kiara-hdri.sh 4k"

  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  local scene_name results="" t ms measured=0 timed_out=0 status
  scene_name="$(basename "$SCENE" .scala)"

  emit "| scene | t | warm ms/frame |"
  emit "|---|---|---|"
  for t in ${times//,/ }; do
    status=0
    ms="$(measure_ms "$t")" || status=$?
    [ "$status" -eq 0 ] || [ "$status" -eq "$TIMEOUT_EXIT" ] || exit "$status"
    emit "| $scene_name | $t | $ms |"
    if [ "$status" -eq 0 ]; then
      results+="$ms"$'\n'; measured=$((measured + 1))
    else
      echo "t=$t timed out after $TIMEOUT_SECONDS s" >&2; timed_out=$((timed_out + 1))
    fi
  done
  [ "$measured" -gt 0 ] || fail "every t timed out after $TIMEOUT_SECONDS s"

  if [ "$mode" = uniform ] && [ "$timed_out" -gt 0 ]; then
    emit ""
    emit "No budget: $timed_out of $((measured + timed_out)) samples timed out."
  elif [ "$mode" = uniform ]; then
    local mean hours
    mean="$(printf '%s' "$results" | awk '{ s += $1 } END { printf "%.1f", s / NR }')"
    hours="$(awk -v m="$mean" -v t="$total" -v fps="$FPS" -v h="$MS_PER_HOUR" \
      'BEGIN { printf "%.2f", m * t * fps / h }')"
    emit ""
    emit "Mean: $mean ms/frame. Budget for $total s at $FPS fps ($(awk -v t="$total" -v f="$FPS" 'BEGIN { print t * f }') frames): $hours h."
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
