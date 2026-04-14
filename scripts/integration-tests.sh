#!/bin/bash
# Integration tests for Menger OptiX renderer
# Usage: ./scripts/integration-tests.sh <menger-binary-path> [--update-references]
#
# Runs comprehensive integration tests and prints summary.
# Exit code: 0 if all pass, 1 if any fail.
#
# Options:
#   --update-references  Regenerate reference images instead of comparing
#
# Environment variables:
#   PARALLEL_MODE=true|false      Enable/disable parallel execution (default: true)
#   MAX_PARALLEL_JOBS=N           Max concurrent test categories (default: 2, reduces GPU contention)

# Parse arguments
UPDATE_REFERENCES=false
MENGER_BIN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --update-references)
            UPDATE_REFERENCES=true
            shift
            ;;
        *)
            MENGER_BIN="$1"
            shift
            ;;
    esac
done

if [ -z "$MENGER_BIN" ] || [ ! -x "$MENGER_BIN" ]; then
    echo "Usage: $0 <menger-binary-path> [--update-references]"
    exit 1
fi

# Configuration
DEFAULT_TIMEOUT=0.1
CAUSTICS_TIMEOUT=1.0
SCRIPT_DIR="$(dirname "$0")"
TEST_ASSETS_DIR="$SCRIPT_DIR/test-assets"
REFERENCE_DIR="$SCRIPT_DIR/reference-images"
DIFF_DIR="$SCRIPT_DIR/test-diffs"
IMAGE_DIFF_THRESHOLD=0.001  # 0.1% pixel difference tolerance
TEST_WIDTH=200              # Render at 1/4 width (1/16 pixels) for fast test runs
TEST_HEIGHT=150

# Test tracking (global counters - will be aggregated from parallel jobs)
PASSED=0
FAILED=0
FAILED_TESTS=""
IMAGE_COMPARISON_FAILURES=0

# Parallelization support
PARALLEL_MODE="${PARALLEL_MODE:-true}"  # Can be disabled with PARALLEL_MODE=false
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-2}"  # Number of test categories to run in parallel (2 for GPU stability)

# Export variables needed by subshells
export MENGER_BIN
export UPDATE_REFERENCES
export DEFAULT_TIMEOUT
export CAUSTICS_TIMEOUT
export SCRIPT_DIR
export TEST_ASSETS_DIR
export REFERENCE_DIR
export DIFF_DIR
export IMAGE_DIFF_THRESHOLD
export TEST_WIDTH
export TEST_HEIGHT

# Colors
RED='\e[38;5;196m'
GREEN='\e[38;5;46m'
YELLOW='\e[38;5;226m'
RESET='\e[0m'
export RED GREEN YELLOW RESET

# Create directories if they don't exist
mkdir -p "$REFERENCE_DIR"
mkdir -p "$DIFF_DIR"

# Check for ImageMagick
if ! command -v compare &> /dev/null; then
    echo -e "${RED}ERROR: ImageMagick 'compare' command not found${RESET}"
    echo "Please install ImageMagick: sudo apt-get install imagemagick"
    exit 1
fi

# Compare images using ImageMagick
# Returns 0 if images match within threshold, 1 otherwise
# Sets global variable COMPARISON_RESULT with the comparison message
compare_images() {
    local test_name="$1"
    local actual_image="$2"
    local reference_image="$3"
    local diff_image="$4"

    if [ ! -f "$reference_image" ]; then
        COMPARISON_RESULT="no reference"
        return 0
    fi

    # Use ImageMagick compare with AE (Absolute Error) metric
    # Returns number of different pixels
    local diff_pixels
    diff_pixels=$(compare -metric AE "$actual_image" "$reference_image" "$diff_image" 2>&1) || true

    # Get total pixel count
    local total_pixels
    total_pixels=$(identify -format "%[fx:w*h]" "$actual_image" 2>/dev/null) || total_pixels=1

    # Calculate percentage difference
    local diff_percent
    diff_percent=$(echo "scale=6; $diff_pixels / $total_pixels" | bc)

    # Compare against threshold
    local is_below_threshold
    is_below_threshold=$(echo "$diff_percent <= $IMAGE_DIFF_THRESHOLD" | bc)

    if [ "$is_below_threshold" -eq 1 ]; then
        COMPARISON_RESULT="match (diff: ${diff_percent}%)"
        rm -f "$diff_image"  # Remove diff image if test passed
        return 0
    else
        COMPARISON_RESULT="MISMATCH (diff: ${diff_percent}%, saved to: $diff_image)"
        ((IMAGE_COMPARISON_FAILURES++))
        return 1
    fi
}

# Run a test, track result, clean up
run_test() {
    local name="$1"
    shift

    # Generate temporary output filename (unique even in parallel mode)
    local temp_output="test_temp_$$_${RANDOM}_$(date +%N).png"
    rm -f "$temp_output"

    # Run test in headless mode with output file
    local test_passed=false
    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless --save-name "$temp_output" --width "$TEST_WIDTH" --height "$TEST_HEIGHT" "$@" >/dev/null 2>&1 && [ -f "$temp_output" ]; then
        test_passed=true
    fi

    if $test_passed; then
        # Sanitize test name for filename (replace spaces and special chars with underscores)
        local sanitized_name=$(echo "$name" | sed 's/[^a-zA-Z0-9-]/_/g' | sed 's/__*/_/g')
        local reference_file="$REFERENCE_DIR/${sanitized_name}.png"
        local diff_file="$DIFF_DIR/${sanitized_name}_diff.png"

        if [ "$UPDATE_REFERENCES" = true ]; then
            # Update reference image mode
            cp "$temp_output" "$reference_file"
            echo -e "  ${name} - reference updated ${YELLOW}⟳${RESET}"
            ((PASSED++))
        else
            # Normal test mode with image comparison
            local image_match=true
            COMPARISON_RESULT=""
            if [ -f "$reference_file" ]; then
                if ! compare_images "$name" "$temp_output" "$reference_file" "$diff_file"; then
                    image_match=false
                fi
            else
                COMPARISON_RESULT="no reference"
            fi

            if $image_match; then
                ((PASSED++))
                echo -e "  ${name} - ${COMPARISON_RESULT} ${GREEN}✓${RESET}"
            else
                ((FAILED++))
                FAILED_TESTS="$FAILED_TESTS\n  - $name (image mismatch)"
                echo -e "  ${name} - ${COMPARISON_RESULT} ${RED}✗${RESET}"
            fi
        fi
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name (execution failed)"
        echo -e "  ${name} - execution failed ${RED}✗${RESET}"
    fi

    # Clean up temporary file
    rm -f "$temp_output"
}

# Run a test that should FAIL (uses --timeout unless --headless is present)
run_test_should_fail() {
    local name="$1"
    shift

    # Don't use rm -f test_*.png in parallel mode - could delete other tests' files
    # These tests don't produce output anyway

    # Check if --headless is in the arguments (headless and timeout are mutually exclusive)
    local use_timeout=true
    for arg in "$@"; do
        if [ "$arg" = "--headless" ]; then
            use_timeout=false
            break
        fi
    done

    local cmd_result
    if $use_timeout; then
        __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --timeout $DEFAULT_TIMEOUT "$@" >/dev/null 2>&1
        cmd_result=$?
    else
        __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN "$@" >/dev/null 2>&1
        cmd_result=$?
    fi

    if [ $cmd_result -eq 0 ]; then
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name (expected failure but succeeded)"
        echo -e "  ${name} - expected failure but succeeded ${RED}✗${RESET}"
    else
        ((PASSED++))
        echo -e "  ${name} - failed as expected ${GREEN}✓${RESET}"
    fi
}

# Run a test that produces output file (uses --timeout unless --headless is present)
run_test_with_output() {
    local name="$1"
    local output_file="$2"
    shift 2

    rm -f "$output_file"

    # Check if --headless is in the arguments (headless and timeout are mutually exclusive)
    local use_timeout=true
    for arg in "$@"; do
        if [ "$arg" = "--headless" ]; then
            use_timeout=false
            break
        fi
    done

    local test_passed=false
    if $use_timeout; then
        if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --timeout $DEFAULT_TIMEOUT "$@" >/dev/null 2>&1 && [ -f "$output_file" ]; then
            test_passed=true
        fi
    else
        if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN "$@" >/dev/null 2>&1 && [ -f "$output_file" ]; then
            test_passed=true
        fi
    fi

    if $test_passed; then
        # Sanitize test name for filename (replace spaces and special chars with underscores)
        local sanitized_name=$(echo "$name" | sed 's/[^a-zA-Z0-9-]/_/g' | sed 's/__*/_/g')
        local reference_file="$REFERENCE_DIR/${sanitized_name}.png"
        local diff_file="$DIFF_DIR/${sanitized_name}_diff.png"

        if [ "$UPDATE_REFERENCES" = true ]; then
            # Update reference image mode
            cp "$output_file" "$reference_file"
            echo -e "  ${name} - reference updated ${YELLOW}⟳${RESET}"
            ((PASSED++))
        else
            # Normal test mode with image comparison
            local image_match=true
            COMPARISON_RESULT=""
            if [ -f "$reference_file" ]; then
                if ! compare_images "$name" "$output_file" "$reference_file" "$diff_file"; then
                    image_match=false
                fi
            else
                COMPARISON_RESULT="no reference"
            fi

            if $image_match; then
                ((PASSED++))
                echo -e "  ${name} - ${COMPARISON_RESULT} ${GREEN}✓${RESET}"
            else
                ((FAILED++))
                FAILED_TESTS="$FAILED_TESTS\n  - $name (image mismatch)"
                echo -e "  ${name} - ${COMPARISON_RESULT} ${RED}✗${RESET}"
            fi
        fi
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name (execution failed)"
        echo -e "  ${name} - execution failed ${RED}✗${RESET}"
    fi

    rm -f "$output_file"
}

# Run a test category in parallel mode (writes results to temp file)
run_category_parallel() {
    local category_name="$1"
    local category_func="$2"
    local result_file="$3"

    # Reset counters for this category
    PASSED=0
    FAILED=0
    FAILED_TESTS=""
    IMAGE_COMPARISON_FAILURES=0

    # Run the category function
    $category_func

    # Write results to file
    echo "$PASSED" > "${result_file}.passed"
    echo "$FAILED" > "${result_file}.failed"
    echo -e "$FAILED_TESTS" > "${result_file}.failed_tests"
    echo "$IMAGE_COMPARISON_FAILURES" > "${result_file}.image_failures"
}

# Aggregate results from parallel runs
aggregate_results() {
    local result_dir="$1"

    PASSED=0
    FAILED=0
    FAILED_TESTS=""
    IMAGE_COMPARISON_FAILURES=0

    for result_file in "$result_dir"/*.passed; do
        if [ -f "$result_file" ]; then
            PASSED=$((PASSED + $(cat "$result_file")))
        fi
    done

    for result_file in "$result_dir"/*.failed; do
        if [ -f "$result_file" ]; then
            FAILED=$((FAILED + $(cat "$result_file")))
        fi
    done

    for result_file in "$result_dir"/*.failed_tests; do
        if [ -f "$result_file" ]; then
            local tests=$(cat "$result_file")
            if [ -n "$tests" ]; then
                FAILED_TESTS="${FAILED_TESTS}${tests}"
            fi
        fi
    done

    for result_file in "$result_dir"/*.image_failures; do
        if [ -f "$result_file" ]; then
            IMAGE_COMPARISON_FAILURES=$((IMAGE_COMPARISON_FAILURES + $(cat "$result_file")))
        fi
    done
}

print_summary() {
    local total=$((PASSED + FAILED))
    echo ""
    echo "=== Integration Test Summary ==="
    if [ $FAILED -eq 0 ]; then
        echo -e "Passed: ${GREEN}${PASSED}/${total}${RESET}"
    else
        echo -e "Passed: ${PASSED}/${total}"
        echo -e "Failed: ${RED}${FAILED}/${total}${RESET}"
        echo -e "\nFailed tests:${FAILED_TESTS}"
    fi

    if [ $IMAGE_COMPARISON_FAILURES -gt 0 ]; then
        echo -e "\n${YELLOW}Image comparison failures: ${IMAGE_COMPARISON_FAILURES}${RESET}"
        echo -e "Review diff images in: ${DIFF_DIR}"
    fi

    if [ "$UPDATE_REFERENCES" = true ]; then
        echo -e "\n${YELLOW}Reference images updated in: ${REFERENCE_DIR}${RESET}"
    fi
}

# ============================================
# Test Categories
# ============================================

test_basic_objects() {
    echo "Basic OptiX Objects:"
    run_test "sphere" --objects type=sphere:size=0.5 --plane y:-2
    run_test "cube" --objects type=cube:size=0.5 --plane y:-2
    run_test "sponge-volume" --objects type=sponge-volume:level=1:size=0.5 --plane y:-2
    run_test "sponge-surface" --objects type=sponge-surface:level=1:size=0.5 --plane y:-2
    run_test "tesseract" --objects type=tesseract:size=0.5 --plane y:-2
}

test_multi_object() {
    echo "Multi-Object (IAS):"
    run_test "multiple spheres" --plane y:-2 \
        --objects type=sphere:pos=-1,0,0:size=0.5 \
        --objects type=sphere:pos=1,0,0:size=0.5
    run_test "multiple cubes" --plane y:-2 \
        --objects type=cube:pos=-1,0,0:size=0.5 \
        --objects type=cube:pos=1,0,0:size=0.5
    run_test "mixed sphere+cube" --plane y:-2 \
        --objects type=sphere:pos=-1,0,0:size=0.5 \
        --objects type=cube:pos=1,0,0:size=0.5
    run_test "object with color" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#FF0000
    run_test "object with IOR" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:ior=1.5
    run_test "sponge-volume instance" --plane y:-2 \
        --objects type=sponge-volume:pos=0,0,0:size=0.5:level=1
    run_test "cube-sponge instance" --plane y:-2 \
        --objects type=cube-sponge:pos=0,0,0:size=0.5:level=1
}

test_antialiasing() {
    echo "Antialiasing:"
    run_test "AA basic" --objects type=sphere --antialiasing --plane y:-2
    run_test "AA custom depth" --objects type=sphere --antialiasing --aa-max-depth 2 --plane y:-2
    run_test "AA custom threshold" --objects type=sphere --antialiasing --aa-threshold 0.05 --plane y:-2
}

test_lighting() {
    echo "Lighting:"
    run_test "point light" --objects type=sphere --light point:0,3,0:2.0 --plane y:-2
    run_test "directional light" --objects type=sphere --light directional:1,-1,-1:1.5 --plane y:-2
    run_test "colored light" --objects type=sphere --light point:2,2,2:1.5:#FF0000 --plane y:-2
    run_test "shadows" --objects type=sphere --shadows --light directional:1,-1,-1:2.0 --plane y:-2
}

test_scene_options() {
    echo "Scene Options:"
    run_test "custom camera" --objects type=sphere --camera-pos 0,2,5 --camera-lookat 0,0,0 --plane y:-2
    run_test "plane color solid" --objects type=sphere --plane y:-2 --plane-color '#808080'
    run_test "plane color checkered" --objects type=sphere --plane y:-2 --plane-color '#FFFFFF:#000000'
    run_test "plane material chrome" --objects type=sphere --plane y:-2 --plane-material chrome
    run_test "plane material gold" --objects type=sphere --plane y:-2 --plane-material gold
    run_test "plane material matte" --objects type=sphere --plane y:-2 --plane-material matte
    run_test_with_output "custom size + save" "test_size.png" \
        --objects type=sphere --width 400 --height 300 --headless --save-name test_size.png --plane y:-2
}

test_materials() {
    echo "Materials:"
    run_test "material preset glass" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=glass
    run_test "material preset chrome" --plane y:-2 \
        --objects type=cube:pos=0,0,0:size=0.5:material=chrome
    run_test "material preset matte" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=matte
    run_test "material with color override" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=metal:color=#FFD700
}

test_film_materials() {
    echo "Film Materials (Thin-Film Interference):"
    # Default Film preset (500nm, IOR 1.33)
    run_test "film sphere default" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film
    # Explicit thickness overrides
    run_test "film thickness 300nm" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film:film-thickness=300
    run_test "film thickness 700nm" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film:film-thickness=700
    # Coated metal (chrome with oily film)
    run_test "chrome with film coating 300nm" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=chrome:film-thickness=300
    # Three thicknesses side by side
    run_test "three film thicknesses" --plane y:-2 \
        --objects type=sphere:pos=-2,0,0:size=0.5:material=film:film-thickness=300 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film \
        --objects type=sphere:pos=2,0,0:size=0.5:material=film:film-thickness=700
    # Regression: zero thickness behaves like standard dielectric
    run_test "film zero thickness (regression)" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=glass
    # Standalone film-thickness without material preset
    run_test "standalone film-thickness parameter" --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:film-thickness=500
    # Film on tesseract edges
    run_test "tesseract with film edge material" --plane y:-2 \
        --objects type=tesseract:size=0.8:edge-material=film:edge-radius=0.025
}

test_textures() {
    echo "Textures:"
    run_test "texture on cube" --plane y:-2 \
        --texture-dir "$TEST_ASSETS_DIR" \
        --objects type=cube:pos=0,0,0:size=0.5:texture=test_checker.png
}

test_caustics() {
    echo "Caustics:"
    TIMEOUT=$CAUSTICS_TIMEOUT run_test "caustics minimal" \
        --objects type=sphere:ior=1.5 --caustics \
        --caustics-photons 1000 --caustics-iterations 1 --plane y:-2
}

test_tesseract() {
    echo "Tesseract (4D Hypercube):"
    run_test "tesseract default rotation" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8
    run_test "tesseract custom XW rotation" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-xw=45
    run_test "tesseract custom YW rotation" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-yw=30
    run_test "tesseract custom ZW rotation" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-zw=60
    run_test "tesseract all rotations" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-xw=30:rot-yw=20:rot-zw=10
    run_test "tesseract custom projection" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:eye-w=5.0:screen-w=2.0
    run_test "tesseract with color" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:color=#4488FF
    run_test "tesseract with material" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:material=glass
    run_test "tesseract transparent" --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:ior=1.5
    run_test "multiple tesseracts" --plane y:-2 \
        --objects type=tesseract:pos=-1,0,0:size=0.5 \
        --objects type=tesseract:pos=1,0,0:size=0.5
}

test_4d_sponges() {
    echo "4D Menger Sponges:"
    run_test "tesseract-sponge level 0" --plane y:-2 \
        --objects type=tesseract-sponge:level=0:pos=0,0,0:size=0.8
    run_test "tesseract-sponge level 1" --plane y:-2 \
        --objects type=tesseract-sponge:level=1:pos=0,0,0:size=0.8
    run_test "tesseract-sponge-2 level 0" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=0:pos=0,0,0:size=0.8
    run_test "tesseract-sponge-2 level 1" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1:pos=0,0,0:size=0.8
    run_test "tesseract-sponge with rotation" --plane y:-2 \
        --objects type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30:size=0.8
    run_test "tesseract-sponge with material" --plane y:-2 \
        --objects type=tesseract-sponge:level=1:material=glass:size=0.8
    run_test "tesseract-sponge-2 with color" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1:color=#FF4488:size=0.8
    run_test "tesseract-sponge with edges" --plane y:-2 \
        --objects type=tesseract-sponge:level=1:edge-material=chrome:edge-radius=0.015:size=0.8
    run_test "mixed 4D sponge + tesseract" --plane y:-2 \
        --objects type=tesseract-sponge:level=1:pos=-1.5,0,0:size=0.5 \
        --objects type=tesseract:pos=1.5,0,0:size=0.5
    run_test "mixed 4D sponge + 3D sphere" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1:pos=-1.5,0,0:size=0.5 \
        --objects type=sphere:pos=1.5,0,0:size=0.5

    # Fractional level tests (per-vertex alpha blending)
    run_test "fractional level 0.5 (tesseract-sponge)" --plane y:-2 \
        --objects type=tesseract-sponge:level=0.5:size=0.8
    run_test "fractional level 1.25 (tesseract-sponge)" --plane y:-2 \
        --objects type=tesseract-sponge:level=1.25:size=0.8
    run_test "fractional level 1.5 (tesseract-sponge)" --plane y:-2 \
        --objects type=tesseract-sponge:level=1.5:size=0.8
    run_test "fractional level 1.75 (tesseract-sponge-2)" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1.75:size=0.8
    run_test "fractional level 0.9 (tesseract-sponge-2)" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=0.9:size=0.8
    run_test "fractional level with material" --plane y:-2 \
        --objects type=tesseract-sponge:level=1.5:material=glass:size=0.8
    run_test "fractional level with rotation" --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1.3:rot-xw=30:rot-yw=20:size=0.8
    run_test "mixed fractional + integer levels" --plane y:-2 \
        --objects type=tesseract-sponge:level=1.5:pos=-1.2,0,0:size=0.5 \
        --objects type=tesseract-sponge:level=1:pos=1.2,0,0:size=0.5
}

test_3d_fractional_sponges() {
    echo "3D Fractional Sponges:"
    # SpongeByVolume fractional levels
    run_test "sponge-volume fractional level 0.5" --plane y:-2 \
        --objects type=sponge-volume:level=0.5:size=0.8
    run_test "sponge-volume fractional level 1.25" --plane y:-2 \
        --objects type=sponge-volume:level=1.25:size=0.8
    run_test "sponge-volume fractional level 1.5" --plane y:-2 \
        --objects type=sponge-volume:level=1.5:size=0.8
    run_test "sponge-volume fractional level 1.75" --plane y:-2 \
        --objects type=sponge-volume:level=1.75:size=0.8

    # SpongeBySurface fractional levels
    run_test "sponge-surface fractional level 0.5" --plane y:-2 \
        --objects type=sponge-surface:level=0.5:size=0.8
    run_test "sponge-surface fractional level 1.25" --plane y:-2 \
        --objects type=sponge-surface:level=1.25:size=0.8
    run_test "sponge-surface fractional level 1.5" --plane y:-2 \
        --objects type=sponge-surface:level=1.5:size=0.8
    run_test "sponge-surface fractional level 1.75" --plane y:-2 \
        --objects type=sponge-surface:level=1.75:size=0.8

    # Fractional levels with materials
    run_test "sponge-volume fractional with glass" --plane y:-2 \
        --objects type=sponge-volume:level=1.5:material=glass:size=0.8
    run_test "sponge-surface fractional with chrome" --plane y:-2 \
        --objects type=sponge-surface:level=1.5:material=chrome:size=0.8
    run_test "sponge-volume fractional with color" --plane y:-2 \
        --objects type=sponge-volume:level=1.3:color=#FF4488:size=0.8

    # Mixed fractional and integer levels
    run_test "mixed sponge-volume fractional + integer" --plane y:-2 \
        --objects type=sponge-volume:level=1.5:pos=-1.2,0,0:size=0.5 \
        --objects type=sponge-volume:level=1:pos=1.2,0,0:size=0.5
    run_test "mixed sponge-surface types different levels" --plane y:-2 \
        --objects type=sponge-surface:level=1.25:pos=-1.2,0,0:size=0.5 \
        --objects type=sponge-surface:level=1.75:pos=1.2,0,0:size=0.5

    # Edge cases
    run_test "sponge-volume very small fractional part" --plane y:-2 \
        --objects type=sponge-volume:level=1.01:size=0.8
    run_test "sponge-surface fractional near 1.0" --plane y:-2 \
        --objects type=sponge-surface:level=1.99:size=0.8
}

test_file_output() {
    echo "File Output:"
    run_test_with_output "save PNG" "test_output.png" \
        --objects type=sphere --headless --save-name test_output.png --plane y:-2
    run_test_with_output "save with AA" "test_aa.png" \
        --objects type=sphere --antialiasing --headless --save-name test_aa.png --plane y:-2
}

test_headless() {
    echo "Headless Mode:"
    run_test_with_output "headless sphere" "test_headless_sphere.png" \
        --objects type=sphere --headless --save-name test_headless_sphere.png --plane y:-2
    run_test_with_output "headless cube" "test_headless_cube.png" \
        --objects type=cube --headless --save-name test_headless_cube.png --plane y:-2
    run_test_with_output "headless tesseract" "test_headless_tesseract.png" \
        --objects type=tesseract --headless --save-name test_headless_tesseract.png --plane y:-2
    run_test_with_output "headless multi-object" "test_headless_multi.png" \
        --objects type=sphere:pos=-1,0,0 --objects type=cube:pos=1,0,0 \
        --headless --save-name test_headless_multi.png --plane y:-2
    run_test_should_fail "headless without save-name" --objects type=sphere --headless --plane y:-2
}

test_dsl_scenes() {
    echo "DSL Scenes:"
    run_test "DSL SimpleScene" --scene examples.dsl.SimpleScene
    run_test "DSL ThreeMaterials" --scene examples.dsl.ThreeMaterials
    run_test "DSL GlassSphere" --scene examples.dsl.GlassSphere
    run_test "DSL TesseractDemo" --scene examples.dsl.TesseractDemo
    run_test "DSL FilmSphere" --scene examples.dsl.FilmSphere
    run_test "DSL SpongeShowcase" --scene examples.dsl.SpongeShowcase
    run_test "DSL MengerShowcase" --scene examples.dsl.MengerShowcase
    run_test "DSL CausticsDemo" --scene examples.dsl.CausticsDemo
    run_test "DSL CustomMaterials" --scene examples.dsl.CustomMaterials
    run_test "DSL ComplexLighting" --scene examples.dsl.ComplexLighting
    run_test "DSL ReusableComponents" --scene examples.dsl.ReusableComponents
    run_test "DSL MixedMetallicShowcase" --scene examples.dsl.MixedMetallicShowcase
}

test_t_animation() {
    echo "t-Parameter Animation:"
    # Freeze-frame: animated scene evaluated at a fixed t value
    run_test "t-animation freeze-frame OrbitingSphere" \
        --scene examples.dsl.OrbitingSphere --t 0.5
    run_test "t-animation freeze-frame PulsingSponge" \
        --scene examples.dsl.PulsingSponge --t 1.0

    # Multi-frame animation: 3 frames for speed
    local temp_dir
    temp_dir=$(mktemp -d)
    local test_passed=false

    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless \
        --scene examples.dsl.OrbitingSphere \
        --frames 3 --start-t 0 --end-t 1 \
        --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --save-name "${temp_dir}/orbit_%04d.png" >/dev/null 2>&1; then
        # Verify all 3 frames were created
        local frame_count=0
        for f in "${temp_dir}"/orbit_*.png; do
            [ -f "$f" ] && ((frame_count++))
        done
        if [ "$frame_count" -eq 3 ]; then
            test_passed=true
        fi
    fi

    if $test_passed; then
        ((PASSED++))
        echo -e "  t-animation multi-frame (3 frames) ${GREEN}✓${RESET}"
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - t-animation multi-frame (execution or frame count failed)"
        echo -e "  t-animation multi-frame (3 frames) ${RED}✗${RESET}"
    fi

    rm -rf "$temp_dir"
}

test_colored_shadows() {
    echo "Colored Shadows:"

    # Basic colored shadow tests — verify the feature renders successfully
    run_test "colored shadow red sphere" --plane y:-2 --shadows --transparent-shadows \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#FF000080:ior=1.5

    run_test "colored shadow green sphere" --plane y:-2 --shadows --transparent-shadows \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#00FF0080:ior=1.5

    run_test "colored shadow blue sphere" --plane y:-2 --shadows --transparent-shadows \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#0000FF80:ior=1.5

    # Verify transparent-shadows produces different output than scalar shadows
    # Render same scene WITH and WITHOUT --transparent-shadows, then compare
    local temp_with="test_cshadow_with_$$_${RANDOM}.png"
    local temp_without="test_cshadow_without_$$_${RANDOM}.png"
    rm -f "$temp_with" "$temp_without"

    local rendered_both=true

    if ! __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless \
        --save-name "$temp_with" --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --plane y:-2 --shadows --transparent-shadows \
        --light directional:0,1,0:2.0 \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#FF000080:ior=1.5 \
        >/dev/null 2>&1 || [ ! -f "$temp_with" ]; then
        rendered_both=false
    fi

    if ! __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless \
        --save-name "$temp_without" --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --plane y:-2 --shadows \
        --light directional:0,1,0:2.0 \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#FF000080:ior=1.5 \
        >/dev/null 2>&1 || [ ! -f "$temp_without" ]; then
        rendered_both=false
    fi

    if $rendered_both && [ -f "$temp_with" ] && [ -f "$temp_without" ]; then
        # Compare the two images — they should DIFFER
        local diff_pixels
        diff_pixels=$(compare -metric AE "$temp_with" "$temp_without" /dev/null 2>&1) || true

        if [ "$diff_pixels" -gt 0 ] 2>/dev/null; then
            ((PASSED++))
            echo -e "  colored vs scalar shadows differ (${diff_pixels}px) ${GREEN}✓${RESET}"
        else
            ((FAILED++))
            FAILED_TESTS="$FAILED_TESTS\n  - colored vs scalar shadows differ (images identical)"
            echo -e "  colored vs scalar shadows differ - images identical ${RED}✗${RESET}"
        fi
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - colored vs scalar shadows differ (render failed)"
        echo -e "  colored vs scalar shadows differ - render failed ${RED}✗${RESET}"
    fi

    rm -f "$temp_with" "$temp_without"
}

test_parametric_caustics_comparison() {
    echo "Parametric caustics comparison:"

    local prim_out="test_prim_caustics_$$.png"
    local para_out="test_para_caustics_$$.png"

    TIMEOUT=$CAUSTICS_TIMEOUT xvfb-run -a $MENGER_BIN --headless \
        --save-name "$prim_out" --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --scene examples.dsl.GlassSphere > /dev/null 2>&1 || true

    TIMEOUT=$CAUSTICS_TIMEOUT xvfb-run -a $MENGER_BIN --headless \
        --save-name "$para_out" --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --scene examples.dsl.ParametricSphereCaustics > /dev/null 2>&1 || true

    if [ ! -f "$prim_out" ] || [ ! -f "$para_out" ]; then
        echo -e "  ${YELLOW}SKIP${RESET}: could not render one or both caustic images"
        rm -f "$prim_out" "$para_out"
        return
    fi

    local pass_threshold=5  # allow up to 5% pixel difference due to tessellation approximation
    local diff_pixels total_pixels diff_pct ok
    diff_pixels=$(compare -metric AE "$prim_out" "$para_out" /dev/null 2>&1) || diff_pixels=0
    total_pixels=$(identify -format "%[fx:w*h]" "$prim_out" 2>/dev/null) || total_pixels=1
    diff_pct=$(echo "scale=2; $diff_pixels * 100 / $total_pixels" | bc 2>/dev/null) || diff_pct=0
    ok=$(echo "$diff_pct <= $pass_threshold" | bc 2>/dev/null) || ok=0
    if [ "$ok" -eq 1 ]; then
        echo -e "  ${GREEN}PASS${RESET}: parametric sphere caustics match primitive (diff: ${diff_pct}%)"
    else
        echo -e "  ${RED}FAIL${RESET}: caustic mismatch between primitive and parametric sphere (diff: ${diff_pct}%)"
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - parametric sphere caustics: mismatch between primitive and parametric sphere (diff: ${diff_pct}%)"
    fi
    rm -f "$prim_out" "$para_out"
}

test_area_lights() {
    echo "Area Lights (Soft Shadows):"
    # Basic area light renders successfully
    run_test "area light basic" \
        --objects type=sphere --shadows \
        --light "area:0,2,2:0,-1,0:1.5:4:10" \
        --plane y:-2 --camera-pos 0,3,5 --camera-lookat 0,-1,0

    # Larger radius → broader penumbra
    run_test "area light large radius" \
        --objects type=sphere --shadows \
        --light "area:0,2,2:0,-1,0:3.0:8:10" \
        --plane y:-2 --camera-pos 0,3,5 --camera-lookat 0,-1,0

    # Colored area light
    run_test "area light colored orange" \
        --objects type=sphere --shadows \
        --light "area:0,2,2:0,-1,0:1.5:4:10:ff8800" \
        --plane y:-2 --camera-pos 0,3,5 --camera-lookat 0,-1,0

    # Area light with transparent sphere — colored soft shadow
    run_test "area light colored shadow" \
        --objects "type=sphere:color=#FF000066:ior=1.5" --shadows --transparent-shadows \
        --light "area:0,2,2:0,-1,0:1.5:4:10" \
        --plane y:-2 --camera-pos 0,3,5 --camera-lookat 0,-1,0

    # Verify soft shadow is visually distinct from hard point shadow
    local soft_png="test_area_soft_$$.png"
    local hard_png="test_area_hard_$$.png"
    local base_args="--objects type=sphere --shadows --plane y:-2 --camera-pos 0,3,5 --camera-lookat 0,-1,0"

    if $MENGER_BIN $base_args --light "area:0,2,2:0,-1,0:2.0:8:10" \
            --headless --save-name "$soft_png" 2>/dev/null ; then
        if $MENGER_BIN $base_args --light "point:0,2,2:10" \
                --headless --save-name "$hard_png" 2>/dev/null ; then
            local diff_pixels
            diff_pixels=$(compare -metric AE "$soft_png" "$hard_png" /dev/null 2>&1) || true
            if [ "${diff_pixels:-0}" -gt 100 ] 2>/dev/null; then
                ((PASSED++))
                echo -e "  soft vs hard shadow differ (${diff_pixels}px) ${GREEN}✓${RESET}"
            else
                ((FAILED++))
                FAILED_TESTS="$FAILED_TESTS\n  - soft vs hard shadow differ (images identical or compare failed)"
                echo -e "  soft vs hard shadow — images identical or compare failed ${RED}✗${RESET}"
            fi
        fi
    fi
    rm -f "$soft_png" "$hard_png"
}

test_error_handling() {
    echo "Error Handling:"
    run_test_should_fail "invalid object type" --objects type=invalid-type --plane y:-2
    run_test_should_fail "invalid multi-object type" \
        --objects type=invalid:pos=0,0,0:size=1 --plane y:-2
    run_test_should_fail "invalid material preset" \
        --objects type=sphere:pos=0,0,0:size=0.5:material=unobtanium --plane y:-2
    run_test_should_fail "tesseract invalid eye-w <= screen-w" \
        --objects type=tesseract:eye-w=1.0:screen-w=2.0 --plane y:-2
    run_test_should_fail "tesseract-sponge missing level" \
        --objects type=tesseract-sponge:pos=0,0,0 --plane y:-2
    run_test_should_fail "tesseract-sponge-2 missing level" \
        --objects type=tesseract-sponge-2:pos=0,0,0 --plane y:-2
    run_test_should_fail "tesseract-sponge negative level" \
        --objects type=tesseract-sponge:level=-1 --plane y:-2
    run_test_should_fail "headless with timeout" \
        --objects type=sphere --headless --save-name test.png --timeout 5 --plane y:-2
}

# ============================================
# Export functions for parallel execution
# ============================================

# Export all helper functions and test categories so they're available in background subshells
export -f compare_images
test_video_output() {
    echo "Video Output (ffmpeg):"

    # Skip entire category if ffmpeg is not available
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo -e "  ${YELLOW}SKIP${RESET}: ffmpeg not found on PATH — skipping video output tests"
        return
    fi

    local temp_dir
    temp_dir=$(mktemp -d)

    # Test 1: MP4 output with frame cleanup (default --keep-frames=false)
    # Use 1 frame to minimise GPU load during the hook run
    local mp4_out="${temp_dir}/test_orbit.mp4"
    local mp4_passed=false

    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless \
        --scene examples.dsl.OrbitingSphere \
        --frames 1 --start-t 0 --end-t 1 \
        --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --save-name "${temp_dir}/orbit_mp4_%04d.png" \
        --video "$mp4_out" >/dev/null 2>&1; then
        if [ -f "$mp4_out" ] && [ -s "$mp4_out" ]; then
            # Frames should have been cleaned up
            local remaining_frames=0
            for f in "${temp_dir}"/orbit_mp4_*.png; do
                [ -f "$f" ] && ((remaining_frames++))
            done
            if [ "$remaining_frames" -eq 0 ]; then
                mp4_passed=true
            fi
        fi
    fi

    if $mp4_passed; then
        ((PASSED++))
        echo -e "  video output MP4 (1 frame, frames cleaned up) ${GREEN}✓${RESET}"
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - video output MP4 (failed or frames not cleaned up)"
        echo -e "  video output MP4 (1 frame, frames cleaned up) ${RED}✗${RESET}"
    fi

    # Test 2: MKV output with --keep-frames
    local mkv_out="${temp_dir}/test_orbit.mkv"
    local mkv_passed=false

    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless \
        --scene examples.dsl.OrbitingSphere \
        --frames 1 --start-t 0 --end-t 1 \
        --width "$TEST_WIDTH" --height "$TEST_HEIGHT" \
        --save-name "${temp_dir}/orbit_mkv_%04d.png" \
        --video "$mkv_out" --keep-frames >/dev/null 2>&1; then
        if [ -f "$mkv_out" ] && [ -s "$mkv_out" ]; then
            # Frames should still exist
            local kept_frames=0
            for f in "${temp_dir}"/orbit_mkv_*.png; do
                [ -f "$f" ] && ((kept_frames++))
            done
            if [ "$kept_frames" -eq 1 ]; then
                mkv_passed=true
            fi
        fi
    fi

    if $mkv_passed; then
        ((PASSED++))
        echo -e "  video output MKV/hevc_nvenc (1 frame, --keep-frames) ${GREEN}✓${RESET}"
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - video output MKV/hevc_nvenc (failed or frame count wrong)"
        echo -e "  video output MKV/hevc_nvenc (1 frame, --keep-frames) ${RED}✗${RESET}"
    fi

    rm -rf "$temp_dir"
}

export -f run_test
export -f run_test_should_fail
export -f run_test_with_output
export -f run_category_parallel
export -f test_basic_objects
export -f test_multi_object
export -f test_antialiasing
export -f test_lighting
export -f test_scene_options
export -f test_materials
export -f test_film_materials
export -f test_textures
export -f test_caustics
export -f test_tesseract
export -f test_4d_sponges
export -f test_3d_fractional_sponges
export -f test_dsl_scenes
export -f test_t_animation
export -f test_file_output
export -f test_headless
export -f test_colored_shadows
export -f test_parametric_caustics_comparison
export -f test_area_lights
export -f test_error_handling
export -f test_video_output

# ============================================
# Main
# ============================================

main() {
    echo "=== Menger Integration Tests ==="
    # Clear OptiX disk cache before any renders to avoid corrupted-state failures
    # from previous runs. OptiX will rebuild the cache during the first render.
    rm -rf /var/tmp/OptixCache_lene 2>/dev/null || true
    echo "Binary: $MENGER_BIN"
    if [ "$UPDATE_REFERENCES" = true ]; then
        echo -e "Mode: ${YELLOW}UPDATE REFERENCES${RESET}"
    else
        echo "Mode: Test with image comparison"
    fi

    if [ "$PARALLEL_MODE" = "true" ]; then
        echo -e "Parallelization: ${GREEN}Enabled${RESET} (max $MAX_PARALLEL_JOBS concurrent categories)"
    else
        echo "Parallelization: Disabled"
    fi
    echo ""

    if [ "$PARALLEL_MODE" = "true" ]; then
        # Parallel execution mode
        local result_dir=$(mktemp -d)
        local job_count=0

        # Define all test categories
        local categories=(
            "test_basic_objects"
            "test_multi_object"
            "test_antialiasing"
            "test_lighting"
            "test_scene_options"
            "test_materials"
            "test_film_materials"
            "test_textures"
            "test_caustics"
            "test_tesseract"
            "test_4d_sponges"
            "test_3d_fractional_sponges"
            "test_dsl_scenes"
            "test_t_animation"
            "test_file_output"
            "test_headless"
            "test_colored_shadows"
            "test_parametric_caustics_comparison"
            "test_area_lights"
            "test_error_handling"
            "test_video_output"
        )

        # Run categories in parallel with job control
        for category in "${categories[@]}"; do
            # Wait if we've reached max parallel jobs
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                sleep 0.1
            done

            # Run category in background
            run_category_parallel "$category" "$category" "$result_dir/$category" &
            ((job_count++))
        done

        # Wait for all background jobs to complete
        wait

        # Aggregate results
        aggregate_results "$result_dir"

        # Cleanup
        rm -rf "$result_dir"
    else
        # Sequential execution mode (original behavior)
        test_basic_objects
        test_multi_object
        test_antialiasing
        test_lighting
        test_scene_options
        test_materials
        test_film_materials
        test_textures
        test_caustics
        test_tesseract
        test_4d_sponges
        test_3d_fractional_sponges
        test_dsl_scenes
        test_t_animation
        test_file_output
        test_headless
        test_colored_shadows
        test_parametric_caustics_comparison
        test_area_lights
        test_error_handling
        test_video_output
    fi

    print_summary

    [ $FAILED -eq 0 ]
}

main
