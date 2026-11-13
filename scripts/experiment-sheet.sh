#!/usr/bin/env bash
# Experiment sheet (menger#186): renders every (scene, t) cell as one still and lays them out as
# a contact sheet, rows = scenes, columns = moments, with a table of the render times.
#
# Usage:
#   scripts/experiment-sheet.sh BIN --out DIR --at T1,T2,... --scenes NAME1,NAME2,...
#       [--labels L1,L2,...] [--width W] [--height H] [--timeout SECONDS]
#       [--texture-dir DIR] [--hdri RELPATH] [--force]
#
#   BIN            the staged launcher, menger-app/target/universal/stage/bin/menger-app
#   --out DIR      where everything goes: cells/<label>_t<t>.png and .log, sheet.png, times.md
#   --at           the moments (seconds; non-negative numbers), one column each
#   --scenes       scene names (examples.dsl.ladder.variants.Tint1) or existing .scala files,
#                  one row each, in the given order
#   --labels       one label per scene (letters, digits, _ and -); default: the part of the
#                  scene name after the last dot, or the file name without .scala
#   --width/height size of a cell (positive integers, default 640x360)
#   --timeout S    kill a render that runs longer than S seconds (positive integer, default
#                  1800); the cell reads "> S s". The time waiting for the GPU lock does not count
#   --texture-dir  where the scene's env map is resolved (default: the menger checkout this
#                  script is in; a worktree has no assets/hdri, so pass the main checkout)
#   --hdri RELPATH the env map the scenes use, relative to --texture-dir, checked before the
#                  first render (default assets/hdri/kiara_2_sunrise_4k.hdr)
#   --force        render every cell again; without it a cell whose image exists is kept
#
# A cell that times out, exits non-zero or logs an ERROR line is a failed cell: its log is kept,
# no image is kept (also not the one of an earlier run, under --force), the sheet shows a grey tile and times.md says "failed" (or "> S s"), and the
# run goes on. The sheet is written in any case; the exit status is 1 if any cell failed.
# Every render runs under flock /tmp/menger-gpu.lock (GPU_LOCK overrides it, for the tests), and
# never outlives it. The first run stamps width, height and hdri in run.params; a rerun into the
# same --out with different ones is refused (exit 2). times.md starts with a provenance line, as
# frame-budget.sh's tables do.
# Exit 1: a cell failed, or the launcher, a scene file, the HDRI or ImageMagick is missing.
# Exit 2: bad arguments, or parameters that differ from the first run in this --out.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="${BASH_SOURCE[0]}"
SCRIPT_NAME="$(basename "$SCRIPT_FILE")"
# shellcheck source-path=SCRIPTDIR source=lib/render-lock.sh
source "$SCRIPT_DIR/lib/render-lock.sh"
DEFAULT_WIDTH=640
DEFAULT_HEIGHT=360
DEFAULT_TIMEOUT_SECONDS=1800
SHEET_LABEL_WIDTH=240
SHEET_HEADER_HEIGHT=48
SHEET_POINTSIZE=24
SHEET_TEXT_FILL=white
SHEET_HEADER_BACKGROUND=white
SHEET_HEADER_FILL=black
FAILED_TILE_COLOUR=grey50
FAILED_TEXT=failed
LABEL_PATTERN='^[A-Za-z0-9_-]+$'

# Splits $1 at commas into the array named $2; an empty list or an empty entry is a usage error.
split_list() { # split_list <value> <array name> <flag>
  local -n target="$2"
  local entry
  target=()
  IFS=, read -ra target <<< "$1" || true
  [ "${#target[@]}" -gt 0 ] || die_usage "$3 needs at least one value"
  for entry in "${target[@]}"; do
    [ -n "$entry" ] || die_usage "$3 has an empty entry: '$1'"
  done
}

default_label() { # default_label <scene>
  case "$1" in
    *.scala) basename "$1" .scala ;;
    *) echo "${1##*.}" ;;
  esac
}

set_times() {
  local entry
  split_list "$1" TIMES --at
  for entry in "${TIMES[@]}"; do
    is_number "$entry" || die_usage "bad moment '$entry' in --at (a non-negative number is expected)"
  done
}

check_unique() { # check_unique <what> <values...>
  local what="$1" duplicate
  shift
  duplicate="$(printf '%s\n' "$@" | sort | uniq -d | head -n 1)"
  [ -z "$duplicate" ] || die_usage "$what '$duplicate' is given twice"
}

set_labels() {
  local label index
  if [ -z "$LABELS_ARG" ]; then
    LABELS=()
    for index in "${!SCENES[@]}"; do LABELS+=("$(default_label "${SCENES[$index]}")"); done
  else
    split_list "$LABELS_ARG" LABELS --labels
    [ "${#LABELS[@]}" -eq "${#SCENES[@]}" ] \
      || die_usage "--labels has ${#LABELS[@]} entries but --scenes has ${#SCENES[@]}"
  fi
  for label in "${LABELS[@]}"; do
    [[ "$label" =~ $LABEL_PATTERN ]] \
      || die_usage "bad label '$label' (letters, digits, _ and - only; pass --labels)"
  done
  check_unique "label" "${LABELS[@]}"
}

parse_args() {
  case "${1:-}" in -h | --help) usage 0 ;; esac
  [ "$#" -ge 1 ] || usage 2
  BIN="$1"; shift
  OUT_DIR=""; TIMES=(); SCENES=(); LABELS_ARG=""; FORCE=0; HDRI=$DEFAULT_HDRI
  WIDTH=$DEFAULT_WIDTH; HEIGHT=$DEFAULT_HEIGHT; TIMEOUT_SECONDS=$DEFAULT_TIMEOUT_SECONDS
  TEXTURE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --out) need_values 1 "$@"; OUT_DIR="$2"; shift 2 ;;
      --at) need_values 1 "$@"; set_times "$2"; shift 2 ;;
      --scenes) need_values 1 "$@"; split_list "$2" SCENES --scenes; shift 2 ;;
      --labels) need_values 1 "$@"; LABELS_ARG="$2"; shift 2 ;;
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
      --force) FORCE=1; shift ;;
      *) die_usage "unknown option: $1" ;;
    esac
  done
  [ -n "$OUT_DIR" ] || die_usage "give --out"
  [ "${#TIMES[@]}" -gt 0 ] || die_usage "give --at"
  [ "${#SCENES[@]}" -gt 0 ] || die_usage "give --scenes"
  check_unique "moment" "${TIMES[@]}"
  set_labels
}

check_environment() {
  local scene tool
  [ -x "$BIN" ] || fail "launcher not found or not executable: $BIN (run: sbt mengerApp/stage)"
  for scene in "${SCENES[@]}"; do
    case "$scene" in
      *.scala) [ -f "$scene" ] || fail "scene file not found: $scene" ;;
    esac
  done
  [ -f "$TEXTURE_DIR/$HDRI" ] \
    || fail "$TEXTURE_DIR/$HDRI is missing; run scripts/fetch-kiara-hdri.sh 4k"
  for tool in montage convert; do
    command -v "$tool" > /dev/null || fail "$tool is not installed (package: imagemagick)"
  done
}

cell_base() { echo "$OUT_DIR/cells/${1}_t${2}"; } # cell_base <label> <t>

# A cell is done when its image exists and its .cell holds the milliseconds. Decided from the
# .cell text as well as the image, so that the table, the sheet and the skip test cannot disagree.
cell_done() { # cell_done <base>
  [ -f "$1.png" ] && [[ "$(cat "$1.cell" 2> /dev/null)" =~ ^[0-9]+$ ]]
}

fail_cell() { # fail_cell <base> <text>: no image, the text in the table; returns 1
  rm -f "$1.png"
  echo "$2" > "$1.cell"
  return 1
}

# Renders one cell. Writes <base>.png on success and <base>.cell, the text for the table: the
# milliseconds, or the failure. Returns 1 for a failed cell, whose log is kept and whose image,
# from an earlier run, is removed.
render_cell() { # render_cell <scene> <label> <t>
  local scene="$1" base candidate
  base="$(cell_base "$2" "$3")"
  candidate="$WORK/cell.png"
  rm -f "$candidate"
  run_render "$TIMEOUT_SECONDS" "$base.log" "$BIN" \
    --scene "$scene" --texture-dir "$TEXTURE_DIR" --t "$3" --save-name "$candidate" \
    --width "$WIDTH" --height "$HEIGHT"
  if stopped_by_timeout "$RENDER_STATUS" "$RENDER_ELAPSED_MS" "$TIMEOUT_SECONDS"; then
    fail_cell "$base" "> $TIMEOUT_SECONDS s"; return 1
  fi
  if [ "$RENDER_STATUS" -ne 0 ] || log_has_error "$base.log" || [ ! -f "$candidate" ]; then
    fail_cell "$base" "$FAILED_TEXT"; return 1
  fi
  # errexit is off in here (the caller tests the status), so the move is checked explicitly
  mv -f "$candidate" "$base.png" || { fail_cell "$base" "$FAILED_TEXT"; return 1; }
  echo "$RENDER_ELAPSED_MS" > "$base.cell"
}

render_all_cells() {
  local row column base
  FAILED_CELLS=0
  for row in "${!SCENES[@]}"; do
    for column in "${!TIMES[@]}"; do
      base="$(cell_base "${LABELS[$row]}" "${TIMES[$column]}")"
      if [ "$FORCE" -eq 0 ] && cell_done "$base"; then
        echo "kept   ${LABELS[$row]} t=${TIMES[$column]} (image exists)"
      elif render_cell "${SCENES[$row]}" "${LABELS[$row]}" "${TIMES[$column]}"; then
        echo "done   ${LABELS[$row]} t=${TIMES[$column]}: $(cat "$base.cell") ms"
      else
        echo "FAILED ${LABELS[$row]} t=${TIMES[$column]}: $(cat "$base.cell") (log: $base.log)" >&2
        FAILED_CELLS=$((FAILED_CELLS + 1))
      fi
    done
  done
}

cell_text() { # the table text of a cell; "?" for an image whose time was not recorded
  cat "$(cell_base "$1" "$2").cell" 2> /dev/null || echo "?"
}

text_tile() { # text_tile <file> <width> <height> <background> <fill> <text>
  convert -size "${2}x${3}" "xc:$4" -fill "$5" -pointsize "$SHEET_POINTSIZE" \
    -gravity center -annotate 0 "$6" "$1"
}

# The tile of a cell: its image, or a grey tile with the failure text.
tile_for() { # tile_for <label> <t> <file>
  local base
  base="$(cell_base "$1" "$2")"
  if cell_done "$base"; then cp "$base.png" "$3"
  else text_tile "$3" "$WIDTH" "$HEIGHT" "$FAILED_TILE_COLOUR" "$SHEET_TEXT_FILL" "$(cell_text "$1" "$2")"; fi
}

# sheet.png: a header row of "t = T" tiles above the grid of cells (montage, rows = scenes), and
# a column of row labels to its left.
write_sheet() {
  local tiles="$WORK/tiles" row column index=0 file
  mkdir -p "$tiles"
  local -a cell_files=() header_files=() label_files=()
  text_tile "$tiles/corner.png" "$SHEET_LABEL_WIDTH" "$SHEET_HEADER_HEIGHT" \
    "$SHEET_HEADER_BACKGROUND" "$SHEET_HEADER_FILL" ""
  for column in "${!TIMES[@]}"; do
    file="$tiles/head_$column.png"
    text_tile "$file" "$WIDTH" "$SHEET_HEADER_HEIGHT" "$SHEET_HEADER_BACKGROUND" \
      "$SHEET_HEADER_FILL" "t = ${TIMES[$column]}"
    header_files+=("$file")
  done
  for row in "${!SCENES[@]}"; do
    file="$tiles/label_$row.png"
    text_tile "$file" "$SHEET_LABEL_WIDTH" "$HEIGHT" "$SHEET_HEADER_BACKGROUND" \
      "$SHEET_HEADER_FILL" "${LABELS[$row]}"
    label_files+=("$file")
    for column in "${!TIMES[@]}"; do
      file="$tiles/cell_$((index++)).png"
      tile_for "${LABELS[$row]}" "${TIMES[$column]}" "$file"
      cell_files+=("$file")
    done
  done
  montage "${cell_files[@]}" -tile "${#TIMES[@]}x" -geometry "${WIDTH}x${HEIGHT}+0+0" \
    "$tiles/grid.png"
  montage "$tiles/corner.png" "${header_files[@]}" -tile "$((${#TIMES[@]} + 1))x1" \
    -geometry "+0+0" "$tiles/head.png"
  montage "${label_files[@]}" -tile "1x" -geometry "+0+0" "$tiles/labels.png"
  convert "$tiles/labels.png" "$tiles/grid.png" +append "$tiles/body.png"
  convert "$tiles/head.png" "$tiles/body.png" -append "$OUT_DIR/sheet.png"
}

write_times() {
  local row column line
  SCENE_NAME="$(IFS=,; echo "${LABELS[*]}")"
  PROVENANCE_LABEL="Experiment sheet"
  {
    provenance_line
    echo ""
    line="| scene (ms)"
    for column in "${!TIMES[@]}"; do line+=" | t = ${TIMES[$column]}"; done
    echo "$line |"
    line="|---"
    for column in "${!TIMES[@]}"; do line+="|---"; done
    echo "$line|"
    for row in "${!SCENES[@]}"; do
      line="| ${LABELS[$row]}"
      for column in "${!TIMES[@]}"; do line+=" | $(cell_text "${LABELS[$row]}" "${TIMES[$column]}")"; done
      echo "$line |"
    done
  } > "$OUT_DIR/times.md"
}

main() {
  parse_args "$@"
  check_environment
  mkdir -p "$OUT_DIR/cells" || fail "cannot create $OUT_DIR/cells"
  check_run_params "$OUT_DIR/run.params" "$OUT_DIR" "width=$WIDTH" "height=$HEIGHT" "hdri=$HDRI"
  WORK="$(mktemp -d)"
  RENDER_STATE_DIR="$WORK"
  trap 'rm -rf "$WORK"' EXIT
  trap on_interrupt INT TERM HUP

  render_all_cells
  write_sheet
  write_times
  echo "sheet: $OUT_DIR/sheet.png; times: $OUT_DIR/times.md"
  [ "$FAILED_CELLS" -eq 0 ] || fail "$FAILED_CELLS cell(s) failed (logs in $OUT_DIR/cells)"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
