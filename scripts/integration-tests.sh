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
    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --headless --save-name "$temp_output" "$@" >/dev/null 2>&1 && [ -f "$temp_output" ]; then
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
    run_test "sphere" --optix --objects type=sphere:size=0.5 --plane y:-2
    run_test "cube" --optix --objects type=cube:size=0.5 --plane y:-2
    run_test "sponge-volume" --optix --objects type=sponge-volume:level=1:size=0.5 --plane y:-2
    run_test "sponge-surface" --optix --objects type=sponge-surface:level=1:size=0.5 --plane y:-2
    run_test "tesseract" --optix --objects type=tesseract:size=0.5 --plane y:-2
}

test_multi_object() {
    echo "Multi-Object (IAS):"
    run_test "multiple spheres" --optix --plane y:-2 \
        --objects type=sphere:pos=-1,0,0:size=0.5 \
        --objects type=sphere:pos=1,0,0:size=0.5
    run_test "multiple cubes" --optix --plane y:-2 \
        --objects type=cube:pos=-1,0,0:size=0.5 \
        --objects type=cube:pos=1,0,0:size=0.5
    run_test "mixed sphere+cube" --optix --plane y:-2 \
        --objects type=sphere:pos=-1,0,0:size=0.5 \
        --objects type=cube:pos=1,0,0:size=0.5
    run_test "object with color" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:color=#FF0000
    run_test "object with IOR" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:ior=1.5
    run_test "sponge-volume instance" --optix --plane y:-2 \
        --objects type=sponge-volume:pos=0,0,0:size=0.5:level=1
    run_test "cube-sponge instance" --optix --plane y:-2 \
        --objects type=cube-sponge:pos=0,0,0:size=0.5:level=1
}

test_antialiasing() {
    echo "Antialiasing:"
    run_test "AA basic" --optix --objects type=sphere --antialiasing --plane y:-2
    run_test "AA custom depth" --optix --objects type=sphere --antialiasing --aa-max-depth 2 --plane y:-2
    run_test "AA custom threshold" --optix --objects type=sphere --antialiasing --aa-threshold 0.05 --plane y:-2
}

test_lighting() {
    echo "Lighting:"
    run_test "point light" --optix --objects type=sphere --light point:0,3,0:2.0 --plane y:-2
    run_test "directional light" --optix --objects type=sphere --light directional:1,-1,-1:1.5 --plane y:-2
    run_test "colored light" --optix --objects type=sphere --light point:2,2,2:1.5:#FF0000 --plane y:-2
    run_test "shadows" --optix --objects type=sphere --shadows --light directional:1,-1,-1:2.0 --plane y:-2
}

test_scene_options() {
    echo "Scene Options:"
    run_test "custom camera" --optix --objects type=sphere --camera-pos 0,2,5 --camera-lookat 0,0,0 --plane y:-2
    run_test "plane color solid" --optix --objects type=sphere --plane y:-2 --plane-color '#808080'
    run_test "plane color checkered" --optix --objects type=sphere --plane y:-2 --plane-color '#FFFFFF:#000000'
    run_test_with_output "custom size + save" "test_size.png" \
        --optix --objects type=sphere --width 400 --height 300 --headless --save-name test_size.png --plane y:-2
}

test_materials() {
    echo "Materials:"
    run_test "material preset glass" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=glass
    run_test "material preset chrome" --optix --plane y:-2 \
        --objects type=cube:pos=0,0,0:size=0.5:material=chrome
    run_test "material preset matte" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=matte
    run_test "material with color override" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=metal:color=#FFD700
}

test_film_materials() {
    echo "Film Materials (Thin-Film Interference):"
    # Default Film preset (500nm, IOR 1.33)
    run_test "film sphere default" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film
    # Explicit thickness overrides
    run_test "film thickness 300nm" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film:film-thickness=300
    run_test "film thickness 700nm" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film:film-thickness=700
    # Coated metal (chrome with oily film)
    run_test "chrome with film coating 300nm" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=chrome:film-thickness=300
    # Three thicknesses side by side
    run_test "three film thicknesses" --optix --plane y:-2 \
        --objects type=sphere:pos=-2,0,0:size=0.5:material=film:film-thickness=300 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=film \
        --objects type=sphere:pos=2,0,0:size=0.5:material=film:film-thickness=700
    # Regression: zero thickness behaves like standard dielectric
    run_test "film zero thickness (regression)" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:material=glass
    # Standalone film-thickness without material preset
    run_test "standalone film-thickness parameter" --optix --plane y:-2 \
        --objects type=sphere:pos=0,0,0:size=0.5:film-thickness=500
    # Film on tesseract edges
    run_test "tesseract with film edge material" --optix --plane y:-2 \
        --objects type=tesseract:size=0.8:edge-material=film:edge-radius=0.025
}

test_textures() {
    echo "Textures:"
    run_test "texture on cube" --optix --plane y:-2 \
        --texture-dir "$TEST_ASSETS_DIR" \
        --objects type=cube:pos=0,0,0:size=0.5:texture=test_checker.png
}

test_caustics() {
    echo "Caustics:"
    TIMEOUT=$CAUSTICS_TIMEOUT run_test "caustics minimal" \
        --optix --objects type=sphere:ior=1.5 --caustics \
        --caustics-photons 1000 --caustics-iterations 1 --plane y:-2
}

test_tesseract() {
    echo "Tesseract (4D Hypercube):"
    run_test "tesseract default rotation" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8
    run_test "tesseract custom XW rotation" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-xw=45
    run_test "tesseract custom YW rotation" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-yw=30
    run_test "tesseract custom ZW rotation" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-zw=60
    run_test "tesseract all rotations" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:rot-xw=30:rot-yw=20:rot-zw=10
    run_test "tesseract custom projection" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:eye-w=5.0:screen-w=2.0
    run_test "tesseract with color" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:color=#4488FF
    run_test "tesseract with material" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:material=glass
    run_test "tesseract transparent" --optix --plane y:-2 \
        --objects type=tesseract:pos=0,0,0:size=0.8:ior=1.5
    run_test "multiple tesseracts" --optix --plane y:-2 \
        --objects type=tesseract:pos=-1,0,0:size=0.5 \
        --objects type=tesseract:pos=1,0,0:size=0.5
}

test_4d_sponges() {
    echo "4D Menger Sponges:"
    run_test "tesseract-sponge level 0" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=0:pos=0,0,0:size=0.8
    run_test "tesseract-sponge level 1" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1:pos=0,0,0:size=0.8
    run_test "tesseract-sponge-2 level 0" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=0:pos=0,0,0:size=0.8
    run_test "tesseract-sponge-2 level 1" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1:pos=0,0,0:size=0.8
    run_test "tesseract-sponge with rotation" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30:size=0.8
    run_test "tesseract-sponge with material" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1:material=glass:size=0.8
    run_test "tesseract-sponge-2 with color" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1:color=#FF4488:size=0.8
    run_test "tesseract-sponge with edges" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1:edge-material=chrome:edge-radius=0.015:size=0.8
    run_test "mixed 4D sponge + tesseract" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1:pos=-1.5,0,0:size=0.5 \
        --objects type=tesseract:pos=1.5,0,0:size=0.5
    run_test "mixed 4D sponge + 3D sphere" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1:pos=-1.5,0,0:size=0.5 \
        --objects type=sphere:pos=1.5,0,0:size=0.5

    # Fractional level tests (per-vertex alpha blending)
    run_test "fractional level 0.5 (tesseract-sponge)" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=0.5:size=0.8
    run_test "fractional level 1.25 (tesseract-sponge)" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1.25:size=0.8
    run_test "fractional level 1.5 (tesseract-sponge)" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1.5:size=0.8
    run_test "fractional level 1.75 (tesseract-sponge-2)" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1.75:size=0.8
    run_test "fractional level 0.9 (tesseract-sponge-2)" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=0.9:size=0.8
    run_test "fractional level with material" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1.5:material=glass:size=0.8
    run_test "fractional level with rotation" --optix --plane y:-2 \
        --objects type=tesseract-sponge-2:level=1.3:rot-xw=30:rot-yw=20:size=0.8
    run_test "mixed fractional + integer levels" --optix --plane y:-2 \
        --objects type=tesseract-sponge:level=1.5:pos=-1.2,0,0:size=0.5 \
        --objects type=tesseract-sponge:level=1:pos=1.2,0,0:size=0.5
}

test_3d_fractional_sponges() {
    echo "3D Fractional Sponges:"
    # SpongeByVolume fractional levels
    run_test "sponge-volume fractional level 0.5" --optix --plane y:-2 \
        --objects type=sponge-volume:level=0.5:size=0.8
    run_test "sponge-volume fractional level 1.25" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.25:size=0.8
    run_test "sponge-volume fractional level 1.5" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.5:size=0.8
    run_test "sponge-volume fractional level 1.75" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.75:size=0.8

    # SpongeBySurface fractional levels
    run_test "sponge-surface fractional level 0.5" --optix --plane y:-2 \
        --objects type=sponge-surface:level=0.5:size=0.8
    run_test "sponge-surface fractional level 1.25" --optix --plane y:-2 \
        --objects type=sponge-surface:level=1.25:size=0.8
    run_test "sponge-surface fractional level 1.5" --optix --plane y:-2 \
        --objects type=sponge-surface:level=1.5:size=0.8
    run_test "sponge-surface fractional level 1.75" --optix --plane y:-2 \
        --objects type=sponge-surface:level=1.75:size=0.8

    # Fractional levels with materials
    run_test "sponge-volume fractional with glass" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.5:material=glass:size=0.8
    run_test "sponge-surface fractional with chrome" --optix --plane y:-2 \
        --objects type=sponge-surface:level=1.5:material=chrome:size=0.8
    run_test "sponge-volume fractional with color" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.3:color=#FF4488:size=0.8

    # Mixed fractional and integer levels
    run_test "mixed sponge-volume fractional + integer" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.5:pos=-1.2,0,0:size=0.5 \
        --objects type=sponge-volume:level=1:pos=1.2,0,0:size=0.5
    run_test "mixed sponge-surface types different levels" --optix --plane y:-2 \
        --objects type=sponge-surface:level=1.25:pos=-1.2,0,0:size=0.5 \
        --objects type=sponge-surface:level=1.75:pos=1.2,0,0:size=0.5

    # Edge cases
    run_test "sponge-volume very small fractional part" --optix --plane y:-2 \
        --objects type=sponge-volume:level=1.01:size=0.8
    run_test "sponge-surface fractional near 1.0" --optix --plane y:-2 \
        --objects type=sponge-surface:level=1.99:size=0.8
}

test_file_output() {
    echo "File Output:"
    run_test_with_output "save PNG" "test_output.png" \
        --optix --objects type=sphere --headless --save-name test_output.png --plane y:-2
    run_test_with_output "save with AA" "test_aa.png" \
        --optix --objects type=sphere --antialiasing --headless --save-name test_aa.png --plane y:-2
}

test_headless() {
    echo "Headless Mode:"
    run_test_with_output "headless sphere" "test_headless_sphere.png" \
        --optix --objects type=sphere --headless --save-name test_headless_sphere.png --plane y:-2
    run_test_with_output "headless cube" "test_headless_cube.png" \
        --optix --objects type=cube --headless --save-name test_headless_cube.png --plane y:-2
    run_test_with_output "headless tesseract" "test_headless_tesseract.png" \
        --optix --objects type=tesseract --headless --save-name test_headless_tesseract.png --plane y:-2
    run_test_with_output "headless multi-object" "test_headless_multi.png" \
        --optix --objects type=sphere:pos=-1,0,0 --objects type=cube:pos=1,0,0 \
        --headless --save-name test_headless_multi.png --plane y:-2
    run_test_should_fail "headless without save-name" --optix --objects type=sphere --headless --plane y:-2
}

test_error_handling() {
    echo "Error Handling:"
    run_test_should_fail "invalid object type" --optix --objects type=invalid-type --plane y:-2
    run_test_should_fail "invalid multi-object type" \
        --optix --objects type=invalid:pos=0,0,0:size=1 --plane y:-2
    run_test_should_fail "invalid material preset" \
        --optix --objects type=sphere:pos=0,0,0:size=0.5:material=unobtanium --plane y:-2
    run_test_should_fail "tesseract invalid eye-w <= screen-w" \
        --optix --objects type=tesseract:eye-w=1.0:screen-w=2.0 --plane y:-2
    run_test_should_fail "tesseract-sponge missing level" \
        --optix --objects type=tesseract-sponge:pos=0,0,0 --plane y:-2
    run_test_should_fail "tesseract-sponge-2 missing level" \
        --optix --objects type=tesseract-sponge-2:pos=0,0,0 --plane y:-2
    run_test_should_fail "tesseract-sponge negative level" \
        --optix --objects type=tesseract-sponge:level=-1 --plane y:-2
    run_test_should_fail "headless with timeout" \
        --optix --objects type=sphere --headless --save-name test.png --timeout 5 --plane y:-2
}

# ============================================
# Export functions for parallel execution
# ============================================

# Export all helper functions and test categories so they're available in background subshells
export -f compare_images
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
export -f test_file_output
export -f test_headless
export -f test_error_handling

# ============================================
# Main
# ============================================

main() {
    echo "=== Menger Integration Tests ==="
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
            "test_file_output"
            "test_headless"
            "test_error_handling"
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
        test_file_output
        test_headless
        test_error_handling
    fi

    print_summary

    [ $FAILED -eq 0 ]
}

main
