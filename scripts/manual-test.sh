#!/bin/bash
# Manual test script for Menger renderer
# Runs through all major features for visual verification

set -e

cd "$(dirname "$0")/.."

OUTPUT_DIR="test-output"
MENGER="./menger-app/target/universal/stage/bin/menger-app"
# Timeout in seconds for each static render test
TIMEOUT=3

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

# Function to run a test with timeout
run_test() {
    local name="$1"
    local args="$2"
    echo -e "${BLUE}Testing:${NC} $name"
    echo "  Command: $MENGER $args -t $TIMEOUT"
    $MENGER $args -t $TIMEOUT
    echo -e "${GREEN}  ✓ Done${NC}\n"
}

echo -e "${BLUE}=== Static Render Tests ===${NC}\n"

# Basic Objects
echo -e "${YELLOW}--- Basic Objects ---${NC}"
run_test "Sphere (default)" "-o --objects type=sphere -s $OUTPUT_DIR/01-sphere.png"
run_test "Cube" "-o --objects type=cube -s $OUTPUT_DIR/02-cube.png"
run_test "Sponge Surface L1" "-o --objects type=sponge-surface:level=1 -s $OUTPUT_DIR/03-sponge-surface.png"
run_test "Sponge Volume L1" "-o --objects type=sponge-volume:level=1 -s $OUTPUT_DIR/04-sponge-volume.png"
run_test "Cube Sponge L1" "-o --objects type=cube-sponge:level=1 --max-instances 64 -s $OUTPUT_DIR/05-cube-sponge.png"

# Materials (use material= preset for transparency)
echo -e "${YELLOW}--- Materials ---${NC}"
run_test "Glass" "-o --objects type=sphere:material=glass -s $OUTPUT_DIR/06-glass.png"
run_test "Diamond" "-o --objects type=sphere:material=diamond -s $OUTPUT_DIR/07-diamond.png"
run_test "Chrome" "-o --objects type=sphere:material=chrome -s $OUTPUT_DIR/08-chrome.png"
run_test "Custom Color (red)" "-o --objects type=sphere:color=#ff0000 -s $OUTPUT_DIR/09-red.png"
run_test "Custom Color (green)" "-o --objects type=sphere:color=#00ff00 -s $OUTPUT_DIR/10-green.png"
run_test "Custom Color (blue)" "-o --objects type=sphere:color=#0000ff -s $OUTPUT_DIR/11-blue.png"

# Multi-Object Scenes (note: cannot mix spheres with cubes/meshes yet - TD-5)
echo -e "${YELLOW}--- Multi-Object Scenes ---${NC}"
run_test "Two Spheres" "-o --objects type=sphere:pos=-1.5,0,0 --objects type=sphere:pos=1.5,0,0 -s $OUTPUT_DIR/12-two-spheres.png"
run_test "Two Cubes" "-o --objects type=cube:pos=-1.5,0,0:material=glass --objects type=cube:pos=1.5,0,0 -s $OUTPUT_DIR/13-two-cubes.png"
run_test "RGB Cubes" "-o --objects type=cube:pos=-2,0,0:color=#ff0000 --objects type=cube:pos=0,0,0:color=#00ff00 --objects type=cube:pos=2,0,0:color=#0000ff -s $OUTPUT_DIR/14-rgb-cubes.png"

# Lighting
echo -e "${YELLOW}--- Lighting ---${NC}"
run_test "Point Light" "-o --objects type=sphere --light point:2,3,2:1.0 -s $OUTPUT_DIR/15-point-light.png"
run_test "Directional Light" "-o --objects type=sphere --light directional:-1,-1,-1:0.8 -s $OUTPUT_DIR/16-directional.png"
run_test "Colored Light (orange)" "-o --objects type=sphere --light point:2,3,2:1.0:ff8800 -s $OUTPUT_DIR/17-colored-light.png"
run_test "Two Colored Lights" "-o --objects type=sphere --light point:-3,3,2:1.0:ff0000 --light point:3,3,2:1.0:0000ff -s $OUTPUT_DIR/18-two-lights.png"

# Shadows (using homogeneous scenes - cannot mix spheres with cubes yet)
echo -e "${YELLOW}--- Shadows ---${NC}"
run_test "Shadows On (spheres)" "-o --objects type=sphere:pos=-1,0.5,0 --objects type=sphere:pos=1,0,0:size=0.5 --shadows -s $OUTPUT_DIR/19-shadows-on.png"
run_test "Shadows Off (spheres)" "-o --objects type=sphere:pos=-1,0.5,0 --objects type=sphere:pos=1,0,0:size=0.5 -s $OUTPUT_DIR/20-shadows-off.png"

# Antialiasing
echo -e "${YELLOW}--- Antialiasing ---${NC}"
run_test "No AA" "-o --objects type=sphere -s $OUTPUT_DIR/21-no-aa.png"
run_test "AA On" "-o --objects type=sphere --antialiasing -s $OUTPUT_DIR/22-aa-on.png"
run_test "AA Depth 4" "-o --objects type=sphere --antialiasing --aa-max-depth 4 -s $OUTPUT_DIR/23-aa-depth4.png"

# Ground Plane
echo -e "${YELLOW}--- Ground Plane ---${NC}"
run_test "Default Plane" "-o --objects type=sphere --plane y:-1 -s $OUTPUT_DIR/24-plane-default.png"
run_test "Solid Plane" "-o --objects type=sphere --plane y:-1 --plane-color 808080 -s $OUTPUT_DIR/25-plane-solid.png"
run_test "Checkered Plane" "-o --objects type=sphere --plane y:-1 --plane-color ffffff:000000 -s $OUTPUT_DIR/26-plane-checker.png"

# Camera
echo -e "${YELLOW}--- Camera ---${NC}"
run_test "High Camera" "-o --objects type=sphere --camera-pos 0,5,10 --camera-lookat 0,0,0 -s $OUTPUT_DIR/27-camera-high.png"
run_test "Close Camera" "-o --objects type=sphere --camera-pos 0,1,3 --camera-lookat 0,0,0 -s $OUTPUT_DIR/28-camera-close.png"

# Image Size
echo -e "${YELLOW}--- Image Size ---${NC}"
run_test "HD 1920x1080" "-o --objects type=sphere --width 1920 --height 1080 -s $OUTPUT_DIR/29-hd.png"
run_test "Square 512x512" "-o --objects type=sphere --width 512 --height 512 -s $OUTPUT_DIR/30-square.png"

# Caustics
echo -e "${YELLOW}--- Caustics (experimental) ---${NC}"
run_test "Caustics" "-o --objects type=sphere:material=glass --caustics --caustics-photons 10000 -s $OUTPUT_DIR/31-caustics.png"

# Showcase
echo -e "${YELLOW}--- Showcase ---${NC}"
run_test "High Quality Render" "-o --objects type=sponge-surface:level=1:material=glass --antialiasing --aa-max-depth 3 --shadows --light point:3,5,3:1.2 --width 1920 --height 1080 -s $OUTPUT_DIR/32-showcase.png"

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
    "Basic sphere:-o --objects type=sphere"
    "Glass refraction:-o --objects type=sphere:material=glass"
    "Two spheres + shadows:-o --objects type=sphere:pos=-1.5,0,0:material=glass --objects type=sphere:pos=1.5,0,0 --shadows"
    "Sponge surface L1:-o --objects type=sponge-surface:level=1"
    "Cube Sponge L1:-o --objects type=cube-sponge:level=1:color=#00ffff --max-instances 64"
    "Colored lights:-o --objects type=sphere --light point:-3,3,2:1.0:ff0000 --light point:3,3,2:1.0:0000ff"
    "Checkered plane:-o --objects type=sphere --plane y:-1 --plane-color ffffff:000000"
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
