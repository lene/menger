#!/bin/bash
# Integration tests for DSL scene loading
# Usage: ./scripts/test-dsl-scenes.sh <menger-binary-path> [--update-references]
#
# Tests the --scene CLI option with pre-compiled DSL scenes.
# Exit code: 0 if all pass, 1 if any fail.

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
SCRIPT_DIR="$(dirname "$0")"
REFERENCE_DIR="$SCRIPT_DIR/reference-images"
DIFF_DIR="$SCRIPT_DIR/test-diffs"
IMAGE_DIFF_THRESHOLD=0.001  # 0.1% pixel difference tolerance

# Test tracking
PASSED=0
FAILED=0
FAILED_TESTS=""

# Colors
RED='\e[38;5;196m'
GREEN='\e[38;5;46m'
YELLOW='\e[38;5;226m'
RESET='\e[0m'

# Create directories
mkdir -p "$REFERENCE_DIR"
mkdir -p "$DIFF_DIR"

# Check for ImageMagick
if ! command -v compare &> /dev/null; then
    echo -e "${RED}ERROR: ImageMagick 'compare' command not found${RESET}"
    echo "Please install ImageMagick: sudo apt-get install imagemagick"
    exit 1
fi

# Compare images using ImageMagick
compare_images() {
    local test_name="$1"
    local actual_image="$2"
    local reference_image="$3"
    local diff_image="$4"

    if [ ! -f "$reference_image" ]; then
        echo -e "    ${YELLOW}⚠${RESET} No reference image, skipping comparison"
        return 0
    fi

    local diff_pixels
    diff_pixels=$(compare -metric AE "$actual_image" "$reference_image" "$diff_image" 2>&1) || true

    local total_pixels
    total_pixels=$(identify -format "%[fx:w*h]" "$actual_image" 2>/dev/null) || total_pixels=1

    local diff_percent
    diff_percent=$(echo "scale=6; $diff_pixels / $total_pixels" | bc)

    local is_below_threshold
    is_below_threshold=$(echo "$diff_percent <= $IMAGE_DIFF_THRESHOLD" | bc)

    if [ "$is_below_threshold" -eq 1 ]; then
        echo -e "    ${GREEN}✓${RESET} Image matches (diff: ${diff_percent}%)"
        rm -f "$diff_image"
        return 0
    else
        echo -e "    ${RED}✗${RESET} Image mismatch (diff: ${diff_percent}%)"
        echo -e "      Diff saved to: $diff_image"
        return 1
    fi
}

# Run a DSL scene test
run_dsl_test() {
    local name="$1"
    local scene_name="$2"

    local temp_output="dsl_test_$$_${RANDOM}.png"
    rm -f "$temp_output"

    echo -e "${YELLOW}Testing DSL scene:${RESET} $name"
    echo "  Command: $MENGER_BIN --optix --scene $scene_name --headless --save-name $temp_output"

    local test_passed=false
    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --optix --scene "$scene_name" --headless --save-name "$temp_output" >/dev/null 2>&1 && [ -f "$temp_output" ]; then
        test_passed=true
    fi

    if $test_passed; then
        local sanitized_name=$(echo "$name" | sed 's/[^a-zA-Z0-9-]/_/g' | sed 's/__*/_/g')
        local reference_file="$REFERENCE_DIR/${sanitized_name}.png"
        local diff_file="$DIFF_DIR/${sanitized_name}_diff.png"

        if [ "$UPDATE_REFERENCES" = true ]; then
            cp "$temp_output" "$reference_file"
            echo -e "  ${YELLOW}⟳${RESET} Reference updated"
            ((PASSED++))
        else
            local image_match=true
            if [ -f "$reference_file" ]; then
                if ! compare_images "$name" "$temp_output" "$reference_file" "$diff_file"; then
                    image_match=false
                fi
            else
                echo -e "    ${YELLOW}⚠${RESET} No reference image"
            fi

            if $image_match; then
                ((PASSED++))
                echo -e "  ${GREEN}✓${RESET} Passed"
            else
                ((FAILED++))
                FAILED_TESTS="$FAILED_TESTS\n  - $name (image mismatch)"
                echo -e "  ${RED}✗${RESET} Failed"
            fi
        fi
    else
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name (execution failed)"
        echo -e "  ${RED}✗${RESET} Execution failed"
    fi

    rm -f "$temp_output"
    echo ""
}

# Run tests that should fail
run_dsl_test_should_fail() {
    local name="$1"
    local scene_name="$2"

    echo -e "${YELLOW}Testing DSL scene (should fail):${RESET} $name"
    echo "  Command: $MENGER_BIN --optix --scene $scene_name --headless --save-name test.png"

    if __GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a $MENGER_BIN --optix --scene "$scene_name" --headless --save-name test.png >/dev/null 2>&1; then
        ((FAILED++))
        FAILED_TESTS="$FAILED_TESTS\n  - $name (expected failure but succeeded)"
        echo -e "  ${RED}✗${RESET} Expected failure but succeeded"
    else
        ((PASSED++))
        echo -e "  ${GREEN}✓${RESET} Failed as expected"
    fi
    echo ""
}

# Print summary
print_summary() {
    local total=$((PASSED + FAILED))
    echo ""
    echo "=== DSL Scene Integration Test Summary ==="
    if [ $FAILED -eq 0 ]; then
        echo -e "Passed: ${GREEN}${PASSED}/${total}${RESET}"
    else
        echo -e "Passed: ${PASSED}/${total}"
        echo -e "Failed: ${RED}${FAILED}/${total}${RESET}"
        echo -e "\nFailed tests:${FAILED_TESTS}"
    fi

    if [ "$UPDATE_REFERENCES" = true ]; then
        echo -e "\n${YELLOW}Reference images updated in: ${REFERENCE_DIR}${RESET}"
    fi
}

# ============================================
# Main Tests
# ============================================

echo "=== DSL Scene Integration Tests ==="
echo "Binary: $MENGER_BIN"
if [ "$UPDATE_REFERENCES" = true ]; then
    echo -e "Mode: ${YELLOW}UPDATE REFERENCES${RESET}"
else
    echo "Mode: Test with image comparison"
fi
echo ""

# Test registered scenes (simple names)
echo "=== Registered Scenes ==="
run_dsl_test "glass-sphere (registry)" "glass-sphere"
run_dsl_test "menger-showcase (registry)" "menger-showcase"

# Test reflection-loaded scenes (fully-qualified names)
echo "=== Reflection-Loaded Scenes ==="
run_dsl_test "glass-sphere (reflection)" "examples.dsl.GlassSphere"
run_dsl_test "menger-showcase (reflection)" "examples.dsl.MengerShowcase"

# Error handling
echo "=== Error Handling ==="
run_dsl_test_should_fail "non-existent scene" "non-existent-scene"
run_dsl_test_should_fail "invalid class name" "invalid.package.InvalidClass"

print_summary

[ $FAILED -eq 0 ]
