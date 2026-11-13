#!/usr/bin/env bash
# Canonical render of a looping animation (menger#185): the frames of one cycle in chunks, resumable,
# stopping at the first bad chunk, and an optional video.
#
# Usage:
#   scripts/render-cycle.sh BIN SCENE --seconds T --out DIR
#       [--fps 30] [--chunk 60] [--width 1920] [--height 1080] [--timeout-per-frame SECONDS]
#       [--texture-dir DIR] [--hdri RELPATH] [--video FILE]
#
#   BIN            the staged launcher, menger-app/target/universal/stage/bin/menger-app
#   SCENE          a scene name that --scene resolves (examples.dsl.ladder.DimensionLadder), or an
#                  existing .scala file
#   --seconds T    the cycle's length; F = round(T x fps) frames, frame g at t = g / fps, so the
#                  last frame is at T - 1/fps and a looping scene does not repeat its first frame
#   --out DIR      frames/f_%05d.png (numbered by the global frame index), logs/chunk_<c>.log, and
#                  work/ for the chunk being rendered
#   --fps N        frames per second (positive number, default 30)
#   --chunk N      frames per app invocation (positive integer, default 60)
#   --width/height render size (positive integers, default 1920x1080); frames are not checked
#   --timeout-per-frame S  a chunk of n frames is killed after n x S seconds (positive integer,
#                  default 3600); the time waiting for the GPU lock does not count
#   --texture-dir  where the scene's env map is resolved (default: the menger checkout this
#                  script is in; a worktree has no assets/hdri, so pass the main checkout)
#   --hdri RELPATH the env map the scene uses, relative to --texture-dir, checked before the
#                  first render (default assets/hdri/kiara_2_sunrise_4k.hdr)
#   --video FILE   when all frames exist, ffmpeg makes FILE (H.264, yuv420p) from them; without
#                  it the command is printed
#
# A chunk whose global frames all exist is skipped, so a rerun continues with the first
# incomplete chunk. A chunk is rendered into work/chunk_<c>/ and its frames are moved to their
# global names only when the app exited 0, logged no ERROR line and wrote all of them. Otherwise
# the run stops with exit 1, naming the chunk, its frame range, the first missing frame and the
# log, and leaves work/chunk_<c>/ in place. A rendered frame is never deleted; a work directory
# left by an earlier failure is moved aside (work/chunk_<c>.old.<time>) before the chunk is tried
# again. Every render runs under flock /tmp/menger-gpu.lock (GPU_LOCK overrides it, for the
# tests) and never outlives it: on INT/TERM/HUP the render's process group is killed and the
# script exits 130 with everything kept.
# Exit 1: a chunk failed, the launcher, scene file or HDRI is missing, or ffmpeg failed.
# Exit 2: bad arguments.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="${BASH_SOURCE[0]}"
SCRIPT_NAME="$(basename "$SCRIPT_FILE")"
# shellcheck source-path=SCRIPTDIR source=lib/render-lock.sh
source "$SCRIPT_DIR/lib/render-lock.sh"
DEFAULT_FPS=30
DEFAULT_CHUNK=60
DEFAULT_WIDTH=1920
DEFAULT_HEIGHT=1080
DEFAULT_TIMEOUT_PER_FRAME=3600
GLOBAL_FRAME_PATTERN='f_%05d.png'
LOCAL_FRAME_PATTERN='l_%05d.png'

parse_args() {
  case "${1:-}" in -h | --help) usage 0 ;; esac
  [ "$#" -ge 2 ] || usage 2
  BIN="$1"; SCENE="$2"; shift 2
  SECONDS_TOTAL=""; OUT_DIR=""; VIDEO=""; HDRI=$DEFAULT_HDRI
  FPS=$DEFAULT_FPS; CHUNK=$DEFAULT_CHUNK; WIDTH=$DEFAULT_WIDTH; HEIGHT=$DEFAULT_HEIGHT
  TIMEOUT_PER_FRAME=$DEFAULT_TIMEOUT_PER_FRAME
  TEXTURE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --seconds)
        need_values 1 "$@"; is_positive_number "$2" || die_usage "--seconds must be a positive number, got '$2'"
        SECONDS_TOTAL="$2"; shift 2 ;;
      --out) need_values 1 "$@"; OUT_DIR="$2"; shift 2 ;;
      --fps)
        need_values 1 "$@"; is_positive_number "$2" || die_usage "--fps must be a positive number, got '$2'"
        FPS="$2"; shift 2 ;;
      --chunk)
        need_values 1 "$@"; is_positive_integer "$2" || die_usage "--chunk must be a positive integer, got '$2'"
        CHUNK="$2"; shift 2 ;;
      --width)
        need_values 1 "$@"; is_positive_integer "$2" || die_usage "--width must be a positive integer, got '$2'"
        WIDTH="$2"; shift 2 ;;
      --height)
        need_values 1 "$@"; is_positive_integer "$2" || die_usage "--height must be a positive integer, got '$2'"
        HEIGHT="$2"; shift 2 ;;
      --timeout-per-frame)
        need_values 1 "$@"
        is_positive_integer "$2" || die_usage "--timeout-per-frame must be a positive integer (seconds), got '$2'"
        TIMEOUT_PER_FRAME="$2"; shift 2 ;;
      --texture-dir) need_values 1 "$@"; TEXTURE_DIR="$2"; shift 2 ;;
      --hdri)
        need_values 1 "$@"
        case "$2" in /* | *..*) die_usage "--hdri must be a relative path without '..', got '$2'" ;; esac
        HDRI="$2"; shift 2 ;;
      --video) need_values 1 "$@"; VIDEO="$2"; shift 2 ;;
      *) die_usage "unknown option: $1" ;;
    esac
  done
  [ -n "$SECONDS_TOTAL" ] || die_usage "give --seconds"
  [ -n "$OUT_DIR" ] || die_usage "give --out"
  TOTAL_FRAMES="$(awk -v t="$SECONDS_TOTAL" -v fps="$FPS" 'BEGIN { print int(t * fps + 0.5) }')"
  [ "$TOTAL_FRAMES" -ge 1 ] || die_usage "--seconds $SECONDS_TOTAL at $FPS fps is less than one frame"
}

check_environment() {
  [ -x "$BIN" ] || fail "launcher not found or not executable: $BIN (run: sbt mengerApp/stage)"
  case "$SCENE" in
    *.scala) [ -f "$SCENE" ] || fail "scene file not found: $SCENE" ;;
  esac
  [ -f "$TEXTURE_DIR/$HDRI" ] \
    || fail "$TEXTURE_DIR/$HDRI is missing; run scripts/fetch-kiara-hdri.sh 4k"
  if [ -n "$VIDEO" ]; then
    command -v ffmpeg > /dev/null || fail "ffmpeg is not installed (needed for --video)"
  fi
  mkdir -p "$OUT_DIR/frames" "$OUT_DIR/work" "$OUT_DIR/logs" 2> /dev/null && [ -w "$OUT_DIR" ] \
    || fail "$OUT_DIR is not writable"
}

# The t of global frame $1, with 6 decimals as the app is given it.
frame_t() { awk -v g="$1" -v fps="$FPS" 'BEGIN { printf "%.6f", g / fps }'; }

global_frame() { # global_frame <index>: the path
  local name
  # shellcheck disable=SC2059
  printf -v name "$GLOBAL_FRAME_PATTERN" "$1"
  echo "$OUT_DIR/frames/$name"
}

local_frame() { # local_frame <chunk work dir> <index>: the path
  local name
  # shellcheck disable=SC2059
  printf -v name "$LOCAL_FRAME_PATTERN" "$2"
  echo "$1/$name"
}

chunk_complete() { # chunk_complete <first frame> <count>
  local i
  for ((i = $1; i < $1 + $2; i++)); do [ -f "$(global_frame "$i")" ] || return 1; done
}

# Prints the first local frame of a chunk's work directory that is missing; nothing if all exist.
first_missing_local() { # first_missing_local <work dir> <count>
  local i
  for ((i = 0; i < $2; i++)); do
    [ -f "$(local_frame "$1" "$i")" ] || { echo "$i"; return 0; }
  done
}

# Stops the run for a bad chunk, leaving its work directory and log.
fail_chunk() { # fail_chunk <chunk> <first frame> <count> <work dir> <log> <reason> [missing local]
  local chunk="$1" g0="$2" n="$3" work="$4" log="$5" reason="$6" missing="${7:-}"
  echo "ERROR: chunk $chunk (frames $g0 to $((g0 + n - 1)) of $TOTAL_FRAMES) failed: $reason" >&2
  if [ -n "$missing" ]; then
    echo "  first missing local frame $missing (global frame $((g0 + missing)))" >&2
  fi
  echo "  log: $log" >&2
  echo "  partial output kept in: $work" >&2
  exit 1
}

# Moves a work directory left by an earlier failure out of the way; nothing is deleted.
set_aside_stale_work() { # set_aside_stale_work <work dir>
  [ ! -e "$1" ] || mv "$1" "$1.old.$(date +%s%N)"
}

promote_frames() { # promote_frames <work dir> <first frame> <count>: local files to global names
  local i
  for ((i = 0; i < $3; i++)); do mv -f "$(local_frame "$1" "$i")" "$(global_frame $(($2 + i)))"; done
  rmdir "$1"
}

render_chunk() { # render_chunk <chunk> <first frame> <count>
  local chunk="$1" g0="$2" n="$3" work log start_t end_t pattern reason="" missing
  work="$OUT_DIR/work/chunk_$chunk"
  log="$OUT_DIR/logs/chunk_$chunk.log"
  start_t="$(frame_t "$g0")"
  end_t="$(frame_t $((g0 + n - 1)))"
  set_aside_stale_work "$work"
  mkdir -p "$work"
  echo "chunk $chunk: frames $g0 to $((g0 + n - 1)), t = $start_t to $end_t"
  pattern="$work/$LOCAL_FRAME_PATTERN"
  run_render "$((n * TIMEOUT_PER_FRAME))" "$log" "$BIN" \
    --scene "$SCENE" --texture-dir "$TEXTURE_DIR" --frames "$n" \
    --start-t "$start_t" --end-t "$end_t" --save-name "$pattern" \
    --width "$WIDTH" --height "$HEIGHT"
  RENDERED_MS=$((RENDERED_MS + RENDER_ELAPSED_MS))
  RENDERED_FRAMES=$((RENDERED_FRAMES + n))

  missing="$(first_missing_local "$work" "$n")"
  if stopped_by_timeout "$RENDER_STATUS" "$RENDER_ELAPSED_MS" "$((n * TIMEOUT_PER_FRAME))"; then
    reason="timed out after $((n * TIMEOUT_PER_FRAME)) s"
  elif [ "$RENDER_STATUS" -ne 0 ]; then reason="the app exited with status $RENDER_STATUS"
  elif log_has_error "$log"; then reason="the app logged an ERROR"
  elif [ -n "$missing" ]; then reason="the app did not write all $n frames"
  fi
  [ -z "$reason" ] || fail_chunk "$chunk" "$g0" "$n" "$work" "$log" "$reason" "$missing"
  promote_frames "$work" "$g0" "$n"
}

print_progress() { # print_progress <frames done>
  awk -v done="$1" -v total="$TOTAL_FRAMES" -v ms="$RENDERED_MS" -v rendered="$RENDERED_FRAMES" \
    -v per_second="$MS_PER_SECOND" 'BEGIN {
      if (rendered == 0) { printf "progress: %d of %d frames\n", done, total; exit }
      mean = ms / rendered / per_second
      printf "progress: %d of %d frames, mean %.2f s/frame, about %.0f s left\n",
        done, total, mean, mean * (total - done)
    }'
}

finish_video() {
  local command=(ffmpeg -y -framerate "$FPS" -i "$OUT_DIR/frames/$GLOBAL_FRAME_PATTERN"
    -c:v libx264 -pix_fmt yuv420p "${VIDEO:-VIDEO.mp4}")
  if [ -z "$VIDEO" ]; then
    echo "all $TOTAL_FRAMES frames exist; to make a video:"
    echo "  ${command[*]}"
    return 0
  fi
  "${command[@]}" < /dev/null || fail "ffmpeg failed (command: ${command[*]})"
  echo "video: $VIDEO"
}

main() {
  parse_args "$@"
  check_environment
  RENDER_STATE_DIR="$(mktemp -d)"
  trap 'rm -rf "$RENDER_STATE_DIR"' EXIT
  trap on_interrupt INT TERM HUP

  local chunk g0 n frames_done=0
  RENDERED_MS=0
  RENDERED_FRAMES=0
  for ((g0 = 0, chunk = 0; g0 < TOTAL_FRAMES; g0 += CHUNK, chunk++)); do
    n=$((TOTAL_FRAMES - g0 < CHUNK ? TOTAL_FRAMES - g0 : CHUNK))
    if chunk_complete "$g0" "$n"; then
      echo "chunk $chunk: frames $g0 to $((g0 + n - 1)) exist, skipped"
    else
      render_chunk "$chunk" "$g0" "$n"
    fi
    frames_done=$((frames_done + n))
    print_progress "$frames_done"
  done
  finish_video
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
