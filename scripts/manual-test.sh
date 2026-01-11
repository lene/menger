#!/bin/bash
# Manual test script for Menger renderer
# Runs through all major features for visual verification

set -e

cd "$(dirname "$0")/.."

OUTPUT_DIR="test-output"
MENGER="./menger-app-0.4.1/bin/menger-app"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Menger Manual Test Suite ===${NC}\n"

# Step 1: Build executable
echo -e "${YELLOW}[1/2] Building executable...${NC}"
sbt stage
echo -e "${GREEN}✓ Build complete${NC}\n"

# Step 2: Create output directory
echo -e "${YELLOW}[2/2] Setting up test output directory...${NC}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}\n"

# Function to run a test
run_test() {
    local name="$1"
    local args="$2"
    echo -e "${BLUE}Testing:${NC} $name"
    echo "  Command: $MENGER $args"
    $MENGER $args
    echo -e "${GREEN}  ✓ Done${NC}\n"
}

echo -e "${BLUE}=== Static Render Tests ===${NC}\n"

# Basic Objects
echo -e "${YELLOW}--- Basic Objects ---${NC}"
run_test "Sphere (default)" "--save $OUTPUT_DIR/01-sphere.png"
run_test "Cube" "--objects type=cube --save $OUTPUT_DIR/02-cube.png"
run_test "Sponge Surface L2" "--objects type=sponge-surface:level=2 --save $OUTPUT_DIR/03-sponge-surface.png"
run_test "Sponge Volume L2" "--objects type=sponge-volume:level=2 --save $OUTPUT_DIR/04-sponge-volume.png"
run_test "Tesseract" "--objects type=tesseract --save $OUTPUT_DIR/05-tesseract.png"

# Materials
echo -e "${YELLOW}--- Materials ---${NC}"
run_test "Glass" "--objects type=sphere:material=glass --save $OUTPUT_DIR/06-glass.png"
run_test "Chrome" "--objects type=sphere:material=chrome --save $OUTPUT_DIR/07-chrome.png"
run_test "Diamond" "--objects type=sphere:material=diamond --save $OUTPUT_DIR/08-diamond.png"
run_test "Matte" "--objects type=sphere:material=matte --save $OUTPUT_DIR/09-matte.png"
run_test "Custom Color (red)" "--objects type=sphere:color=ff0000 --save $OUTPUT_DIR/10-red.png"
run_test "Transparent" "--objects type=sphere:ior=1.5:color=00ff0080 --save $OUTPUT_DIR/11-transparent.png"

# Multi-Object Scenes
echo -e "${YELLOW}--- Multi-Object Scenes ---${NC}"
run_test "Two Spheres" "--objects type=sphere:x=-1.5 type=sphere:x=1.5 --save $OUTPUT_DIR/12-two-spheres.png"
run_test "Mixed Objects" "--objects type=sphere:x=-1.5:material=glass type=cube:x=1.5:material=chrome --save $OUTPUT_DIR/13-mixed.png"
run_test "RGB Cubes" "--objects type=cube:x=-2:color=ff0000 type=cube:x=0:color=00ff00 type=cube:x=2:color=0000ff --save $OUTPUT_DIR/14-rgb-cubes.png"

# Lighting
echo -e "${YELLOW}--- Lighting ---${NC}"
run_test "Point Light" "--lights type=point:x=2:y=3:z=2:intensity=1.0 --save $OUTPUT_DIR/15-point-light.png"
run_test "Directional Light" "--lights type=directional:dx=-1:dy=-1:dz=-1:intensity=0.8 --save $OUTPUT_DIR/16-directional.png"
run_test "Colored Light" "--lights type=point:x=2:y=3:z=2:color=ff8800 --save $OUTPUT_DIR/17-colored-light.png"
run_test "Two Colored Lights" "--lights type=point:x=-3:y=3:z=2:color=ff0000 type=point:x=3:y=3:z=2:color=0000ff --save $OUTPUT_DIR/18-two-lights.png"

# Shadows
echo -e "${YELLOW}--- Shadows ---${NC}"
run_test "Shadows On" "--objects type=sphere:x=-1:y=0.5 type=cube:x=1 --shadows --save $OUTPUT_DIR/19-shadows-on.png"
run_test "Shadows Off" "--objects type=sphere:x=-1:y=0.5 type=cube:x=1 --no-shadows --save $OUTPUT_DIR/20-shadows-off.png"

# Antialiasing
echo -e "${YELLOW}--- Antialiasing ---${NC}"
run_test "No AA" "--no-aa --save $OUTPUT_DIR/21-no-aa.png"
run_test "AA On" "--aa --save $OUTPUT_DIR/22-aa-on.png"
run_test "AA Depth 4" "--aa --aa-depth 4 --save $OUTPUT_DIR/23-aa-depth4.png"

# Textures
echo -e "${YELLOW}--- Textures ---${NC}"
run_test "Textured Cube" "--objects type=cube:texture=scripts/test-assets/test_checker.png --save $OUTPUT_DIR/24-textured.png"

# Ground Plane
echo -e "${YELLOW}--- Ground Plane ---${NC}"
run_test "Solid Plane" "--plane-color solid:808080 --save $OUTPUT_DIR/25-plane-solid.png"
run_test "Checkered Plane" "--plane-color checkered:ffffff:000000 --save $OUTPUT_DIR/26-plane-checker.png"
run_test "No Plane" "--no-plane --save $OUTPUT_DIR/27-no-plane.png"

# Camera
echo -e "${YELLOW}--- Camera ---${NC}"
run_test "High Camera" "--camera eye=0,5,10:look=0,0,0:up=0,1,0 --save $OUTPUT_DIR/28-camera-high.png"
run_test "Close Camera" "--camera eye=0,1,3:look=0,0,0 --save $OUTPUT_DIR/29-camera-close.png"

# Image Size
echo -e "${YELLOW}--- Image Size ---${NC}"
run_test "HD 1920x1080" "--size 1920x1080 --save $OUTPUT_DIR/30-hd.png"
run_test "Square 512x512" "--size 512x512 --save $OUTPUT_DIR/31-square.png"

# Caustics
echo -e "${YELLOW}--- Caustics (experimental) ---${NC}"
run_test "Caustics" "--objects type=sphere:material=glass --caustics --caustics-photons 10000 --save $OUTPUT_DIR/32-caustics.png"

# Showcase
echo -e "${YELLOW}--- Showcase ---${NC}"
run_test "High Quality Render" "--objects type=sponge-surface:level=2:material=glass --aa --aa-depth 3 --shadows --lights type=point:x=3:y=5:z=3:intensity=1.2 --size 1920x1080 --save $OUTPUT_DIR/33-showcase.png"

echo -e "${BLUE}=== Static Tests Complete ===${NC}"
echo -e "Output files in: ${GREEN}$OUTPUT_DIR/${NC}"
echo ""
ls -la "$OUTPUT_DIR"/*.png | awk '{print $9, $5}' | column -t
echo ""

# Interactive tests
echo -e "${BLUE}=== Interactive Tests ===${NC}"
echo -e "${YELLOW}The following tests require manual interaction.${NC}"
echo -e "Controls: Mouse drag = rotate, Scroll = zoom, Q/ESC = quit\n"

interactive_tests=(
    "Basic sphere::"
    "Glass refraction:--objects type=sphere:material=glass"
    "Multi-object + shadows:--objects type=sphere:x=-1.5:material=glass type=cube:x=1.5 --shadows"
    "Sponge performance:--objects type=sponge-surface:level=2"
    "Tesseract (4D):--objects type=tesseract:color=00ffff"
    "Colored lights:--lights type=point:x=-3:y=3:z=2:color=ff0000 type=point:x=3:y=3:z=2:color=0000ff"
    "Checkered plane:--plane-color checkered:ffffff:000000"
)

echo "Available interactive tests:"
for i in "${!interactive_tests[@]}"; do
    IFS=':' read -r name args <<< "${interactive_tests[$i]}"
    echo "  $((i+1)). $name"
done
echo "  0. Skip interactive tests"
echo ""

while true; do
    read -p "Enter test number (0-${#interactive_tests[@]}): " choice
    
    if [[ "$choice" == "0" ]]; then
        echo -e "${YELLOW}Skipping interactive tests${NC}"
        break
    fi
    
    if [[ "$choice" -ge 1 && "$choice" -le "${#interactive_tests[@]}" ]]; then
        idx=$((choice - 1))
        IFS=':' read -r name args <<< "${interactive_tests[$idx]}"
        echo -e "\n${BLUE}Running:${NC} $name"
        echo "Command: $MENGER $args"
        echo -e "${YELLOW}Press Q or ESC to quit and return to menu${NC}\n"
        $MENGER $args || true
        echo ""
    else
        echo "Invalid choice. Enter 0-${#interactive_tests[@]}"
    fi
done

echo -e "\n${GREEN}=== Manual Test Suite Complete ===${NC}"
echo -e "Review rendered images in: ${BLUE}$OUTPUT_DIR/${NC}"
