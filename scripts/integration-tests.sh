#!/bin/bash
# Integration tests for Menger OptiX renderer
# Usage: ./scripts/integration-tests.sh <menger-binary-path> [--update-references]
#
# Runs comprehensive integration tests and prints summary.
# Exit code: 0 if all pass, 1 if any fail.
#
# Options:
#   --update-references  Regenerate reference images instead of comparing

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

# Test tracking
PASSED=0
FAILED=0
FAILED_TESTS=""
IMAGE_COMPARISON_FAILURES=0

# Colors
RED='\e[38;5;196m'
GREEN='\e[38;5;46m'
YELLOW='\e[38;5;226m'
RESET='\e[0m'

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

    # Generate temporary output filename
    local temp_output="test_temp_$$.png"
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

    rm -f test_*.png

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

    rm -f test_*.png
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
    echo ""

    test_basic_objects
    test_multi_object
    test_antialiasing
    test_lighting
    test_scene_options
    test_materials
    test_textures
    test_caustics
    test_tesseract
    test_4d_sponges
    test_file_output
    test_headless
    test_error_handling

    print_summary

    [ $FAILED -eq 0 ]
}

main
