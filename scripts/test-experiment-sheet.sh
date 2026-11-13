#!/usr/bin/env bash
# Tests scripts/experiment-sheet.sh offline: argument checks, and whole runs with a fake launcher
# instead of the app (no GPU; the runs need xvfb-run, ImageMagick and a private lock).
# Run: scripts/test-experiment-sheet.sh
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPTS/experiment-sheet.sh"
# shellcheck source=/dev/null
source "$SCRIPT"   # for SHEET_LABEL_WIDTH and SHEET_HEADER_HEIGHT; main does not run
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

touch "$TEST_DIR/Scene.scala"
FAKE="$TEST_DIR/fake"
echo '#!/bin/sh' > "$FAKE"; chmod +x "$FAKE"
mkdir -p "$TEST_DIR/tex/assets/hdri"
touch "$TEST_DIR/tex/$DEFAULT_HDRI"
SCENES=examples.dsl.ladder.variants.Tint1,examples.dsl.ladder.variants.Tint2
MOMENTS=0,0.5,150
CELL_WIDTH=40
CELL_HEIGHT=30
COMMON=(--texture-dir "$TEST_DIR/tex" --width "$CELL_WIDTH" --height "$CELL_HEIGHT")

# ---- argument checks (exit 2, the message names the problem) ----

expect_usage_error() { # expect_usage_error <description> <message fragment> <args after BIN...>
    local description="$1" fragment="$2" out status=0
    shift 2
    out="$("$SCRIPT" "$FAKE" "${COMMON[@]}" "$@" 2>&1)" || status=$?
    if [ "$status" -eq 2 ] && grep -qF -- "$fragment" <<< "$out"; then status=0; else status=1; fi
    report "$description" "$status"
}
expect_usage_error "no --scenes" "give --scenes" --out "$TEST_DIR/o" --at 0
expect_usage_error "no --at" "give --at" --out "$TEST_DIR/o" --scenes "$SCENES"
expect_usage_error "no --out" "give --out" --at 0 --scenes "$SCENES"
expect_usage_error "--labels count differs from the scene count" \
    "--labels has 1 entries but --scenes has 2" \
    --out "$TEST_DIR/o" --at 0 --scenes "$SCENES" --labels A
expect_usage_error "two scenes with the same default label" "label 'Tint' is given twice" \
    --out "$TEST_DIR/o" --at 0 --scenes a.b.Tint,c.d.Tint
expect_usage_error "a label that is not a safe file name" "bad label 'a/b'" \
    --out "$TEST_DIR/o" --at 0 --scenes "$SCENES" --labels a/b,c
expect_usage_error "a bad moment" "bad moment '../x'" --out "$TEST_DIR/o" --at ../x --scenes "$SCENES"
expect_usage_error "an empty moment" "--at has an empty entry" --out "$TEST_DIR/o" --at 0,,1 --scenes "$SCENES"
expect_usage_error "a repeated moment" "moment '0' is given twice" --out "$TEST_DIR/o" --at 0,0 --scenes "$SCENES"
expect_usage_error "--timeout 0" "--timeout must be a positive integer" \
    --out "$TEST_DIR/o" --at 0 --scenes "$SCENES" --timeout 0
expect_usage_error "unknown option" "unknown option: --bogus" --out "$TEST_DIR/o" --bogus

# ---- missing prerequisites (exit 1, before any render) ----

run_script() { # run_script <args...>: sets RUN_OUT and RC
    RUN_OUT="$("$SCRIPT" "$@" 2>&1)"; RC=$?
}
run_script "$FAKE" --out "$TEST_DIR/o" --at 0 --scenes "$TEST_DIR/Missing.scala" "${COMMON[@]}"
[ "$RC" -eq 1 ] && grep -qF "scene file not found" <<< "$RUN_OUT"
report "a .scala file that does not exist: exit 1, scene file not found" $?
run_script "$FAKE" --out "$TEST_DIR/o" --at 0 --scenes "$SCENES" --texture-dir "$TEST_DIR/tex" \
    --hdri assets/hdri/other.hdr
[ "$RC" -eq 1 ] && grep -qF "assets/hdri/other.hdr is missing" <<< "$RUN_OUT"
report "a missing HDRI: exit 1, names it" $?
run_script "$TEST_DIR/no-such-launcher" --out "$TEST_DIR/o" --at 0 --scenes "$SCENES" "${COMMON[@]}"
[ "$RC" -eq 1 ] && grep -qF "launcher not found or not executable" <<< "$RUN_OUT"
report "a missing launcher: exit 1" $?
[ ! -e "$TEST_DIR/o" ]; report "nothing was created before the checks passed" $?

# ---- whole runs with a fake launcher ----

# Writes a small PNG to its --save-name and counts its calls, unless FAKE_MODE says that the
# cell (FAKE_BAD_SCENE at FAKE_BAD_T) goes wrong.
cat > "$FAKE" <<'EOF'
#!/bin/sh
out=""; scene=""; t=""
while [ $# -gt 0 ]; do
  case "$1" in --save-name) out="$2" ;; --scene) scene="$2" ;; --t) t="$2" ;; esac
  shift
done
echo "$scene $t" >> "$FAKE_DIR/calls"
if [ "$scene" = "$FAKE_BAD_SCENE" ] && [ "$t" = "$FAKE_BAD_T" ]; then
  case "$FAKE_MODE" in
    exit1) echo boom >&2; exit 1 ;;
    error) echo "10:00:00.000 ERROR x.WithAnimation - frame 0 failed" >&2
           convert -size 8x8 xc:red "$out"; exit 0 ;;
    hang) sleep 60 & echo $! > "$FAKE_DIR/sleep.pid"; wait ;;
  esac
fi
convert -size 8x8 xc:red "$out"
EOF
chmod +x "$FAKE"

calls() { wc -l < "$TEST_DIR/calls" 2> /dev/null || echo 0; }
image_count() { find "$1/cells" -name '*.png' | wc -l; }
sheet_size() { identify -format '%w %h' "$1/sheet.png"; }
data_rows() { grep -c '^| [^s]' "$1/times.md"; } # rows that are not the header
header_cells() { grep '^| scene' "$1/times.md" | tr '|' '\n' | grep -c 't = '; }

run_sheet() { # run_sheet <out dir> <args...>: sets RUN_OUT and RC
    local out_dir="$1"; shift
    RUN_OUT="$("$SCRIPT" "$FAKE" --out "$out_dir" --at "$MOMENTS" --scenes "$SCENES" "${COMMON[@]}" "$@" 2>&1)"
    RC=$?
}

missing_tools=""
for tool in xvfb-run montage convert identify; do
    command -v "$tool" > /dev/null || missing_tools+=" $tool"
done
if [ -n "$missing_tools" ]; then
    echo "SKIP: missing tools:$missing_tools: the whole-run tests with a fake launcher"
else
    export FAKE_DIR="$TEST_DIR" GPU_LOCK="$TEST_DIR/lock" KILL_GRACE_SECONDS=1
    GOOD="$TEST_DIR/good"
    EXPECTED_WIDTH=$((SHEET_LABEL_WIDTH + 3 * CELL_WIDTH))
    EXPECTED_HEIGHT=$((SHEET_HEADER_HEIGHT + 2 * CELL_HEIGHT))

    run_sheet "$GOOD"
    expect_equal "2 scenes x 3 moments: exit 0" "$RC" 0
    expect_equal "...6 images" "$(image_count "$GOOD")" 6
    holds [ -f "$GOOD/cells/Tint1_t0.png" ] && holds [ -f "$GOOD/cells/Tint2_t150.log" ]; status=$?
    report "...named <label>_t<t>.png, with a .log beside it; the label is the last name part" "$status"
    expect_equal "...sheet.png is 3 tiles wide and 2 high (plus label column and header row)" \
        "$(sheet_size "$GOOD")" "$EXPECTED_WIDTH $EXPECTED_HEIGHT"
    expect_equal "...times.md has 2 data rows" "$(data_rows "$GOOD")" 2
    expect_equal "...and 3 time columns" "$(header_cells "$GOOD")" 3
    head -n 1 "$GOOD/times.md" | grep -q "^Experiment sheet [0-9-]* [0-9:]*; GPU: .*; menger commit: .*; ${CELL_WIDTH}x${CELL_HEIGHT}; timeout 1800 s; scene: Tint1,Tint2; hdri: $DEFAULT_HDRI$"
    report "...times.md starts with the provenance line" $?
    grep -qE '^\| Tint1 \| [0-9]+ \| [0-9]+ \| [0-9]+ \|$' "$GOOD/times.md"
    report "...a row holds the milliseconds of each cell" $?
    expect_equal "...the fake launcher was called 6 times" "$(calls)" 6

    run_sheet "$GOOD"
    expect_equal "a rerun: exit 0" "$RC" 0
    expect_equal "...renders nothing" "$(calls)" 6
    expect_equal "...and keeps the times in the table" "$(data_rows "$GOOD")" 2
    grep -qE '^\| Tint1 \| [0-9]+ \| [0-9]+ \| [0-9]+ \|$' "$GOOD/times.md"
    report "...the rerun's table still has the milliseconds" $?

    run_sheet "$GOOD" --force
    expect_equal "--force: exit 0" "$RC" 0
    expect_equal "...renders all 6 again" "$(calls)" 12

    run_labelled="$TEST_DIR/labelled"
    run_sheet "$run_labelled" --labels Red,Green
    [ -f "$run_labelled/cells/Red_t0.png" ] && grep -q '^| Green ' "$run_labelled/times.md"
    report "--labels names the files and the rows" $?

    cases=(exit1 error hang)
    for mode in "${cases[@]}"; do
        : > "$TEST_DIR/calls"
        BAD="$TEST_DIR/bad_$mode"
        extra=()
        [ "$mode" = hang ] && extra=(--timeout 1)
        FAKE_MODE="$mode" FAKE_BAD_SCENE=examples.dsl.ladder.variants.Tint2 FAKE_BAD_T=0.5 \
            run_sheet "$BAD" "${extra[@]}"
        expect_equal "$mode in one cell: exit status 1" "$RC" 1
        expect_equal "...5 images, the failed cell has none" "$(image_count "$BAD")" 5
        holds [ ! -f "$BAD/cells/Tint2_t0.5.png" ] && holds [ -f "$BAD/cells/Tint2_t0.5.log" ]; status=$?
        report "...its log is kept, its image is not" "$status"
        expect_equal "...the other cells were still rendered" "$(calls)" 6
        expect_equal "...sheet.png still exists with its full size" "$(sheet_size "$BAD")" \
            "$EXPECTED_WIDTH $EXPECTED_HEIGHT"
        expected_text=failed
        [ "$mode" = hang ] && expected_text="> 1 s"
        grep -qE "^\| Tint2 \| [0-9]+ \| $expected_text \| [0-9]+ \|$" "$BAD/times.md"
        report "...times.md says '$expected_text' in that cell" $?
    done
    ! kill -0 "$(cat "$TEST_DIR/sleep.pid")" 2> /dev/null; report "the hung launcher's child is gone" $?
fi

echo
if [ "$FAILURES" -eq 0 ]; then echo "All experiment-sheet tests passed"; else echo "$FAILURES failed"; exit 1; fi
