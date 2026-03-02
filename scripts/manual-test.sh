#!/bin/bash
# Manual test script for Menger renderer
# Runs through all major features for visual verification

set -e

cd "$(dirname "$0")/.."

OUTPUT_DIR="test-output"
REFERENCE_DIR="scripts/reference-images"
DIFF_DIR="scripts/test-diffs"
MENGER="./menger-app/target/universal/stage/bin/menger-app"
IMAGE_DIFF_THRESHOLD=0.001  # 0.1% pixel difference tolerance

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_STATIC=false
UPDATE_REFERENCES=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            SKIP_STATIC=true
            shift
            ;;
        --update-references)
            UPDATE_REFERENCES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-i|--interactive] [--update-references]"
            echo "  -i, --interactive      Skip static tests, go directly to interactive menu"
            echo "  --update-references    Regenerate reference images instead of comparing"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== Menger Manual Test Suite ===${NC}\n"

if [ "$UPDATE_REFERENCES" = true ]; then
    echo -e "${YELLOW}Mode: UPDATE REFERENCES${NC}\n"
else
    echo -e "Mode: Test with image comparison\n"
fi

# Step 1: Build executable
echo -e "${YELLOW}[1/3] Building executable...${NC}"
sbt stage
echo -e "${GREEN}✓ Build complete${NC}\n"

# Step 2: Create output directory
echo -e "${YELLOW}[2/3] Setting up test output directory...${NC}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REFERENCE_DIR"
mkdir -p "$DIFF_DIR"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}\n"

# Step 3: Check for ImageMagick
echo -e "${YELLOW}[3/3] Checking for ImageMagick...${NC}"
if ! command -v compare &> /dev/null; then
    echo -e "${RED}WARNING: ImageMagick 'compare' command not found${NC}"
    echo "Image comparison will be skipped. Install with: sudo apt-get install imagemagick"
    IMAGEMAGICK_AVAILABLE=false
else
    echo -e "${GREEN}✓ ImageMagick available${NC}"
    IMAGEMAGICK_AVAILABLE=true
fi
echo ""

# Compare images using ImageMagick
# Returns 0 if images match within threshold, 1 otherwise
compare_images() {
    local test_name="$1"
    local actual_image="$2"
    local reference_image="$3"
    local diff_image="$4"

    if [ "$IMAGEMAGICK_AVAILABLE" = false ]; then
        return 0
    fi

    if [ ! -f "$reference_image" ]; then
        echo -e "    ${YELLOW}⚠${NC} No reference image found, skipping comparison"
        return 0
    fi

    # Use ImageMagick compare with AE (Absolute Error) metric
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
        echo -e "    ${GREEN}✓${NC} Image matches reference (diff: ${diff_percent}%)"
        rm -f "$diff_image"
        return 0
    else
        echo -e "    ${RED}✗${NC} Image differs from reference (diff: ${diff_percent}%, threshold: ${IMAGE_DIFF_THRESHOLD})"
        echo -e "      Diff saved to: $diff_image"
        return 1
    fi
}

# Skip static tests if -i/--interactive flag is set
if [ "$SKIP_STATIC" = true ]; then
    echo -e "${YELLOW}Skipping static tests (--interactive mode)${NC}\n"
else

# Function to run a test with headless mode
run_test() {
    local name="$1"
    local args="$2"

    # Extract output filename from args (look for -s argument)
    local output_file=""
    local found_save=false
    for arg in $args; do
        if [ "$found_save" = true ]; then
            output_file="$arg"
            break
        fi
        if [ "$arg" = "-s" ]; then
            found_save=true
        fi
    done

    echo -e "${BLUE}Testing:${NC} $name"
    echo "  Command: $MENGER $args --headless"

    if $MENGER $args --headless; then
        # Check if we should compare images
        if [ -n "$output_file" ] && [ -f "$output_file" ]; then
            # Sanitize test name for filename
            local sanitized_name=$(echo "$name" | sed 's/[^a-zA-Z0-9-]/_/g' | sed 's/__*/_/g')
            local reference_file="$REFERENCE_DIR/${sanitized_name}.png"
            local diff_file="$DIFF_DIR/${sanitized_name}_diff.png"

            if [ "$UPDATE_REFERENCES" = true ]; then
                # Update reference image mode
                cp "$output_file" "$reference_file"
                echo -e "  ${YELLOW}⟳${NC} Reference updated: $reference_file"
            else
                # Compare with reference
                compare_images "$name" "$output_file" "$reference_file" "$diff_file"
            fi
        fi
        echo -e "${GREEN}  ✓ Done${NC}\n"
    else
        echo -e "${RED}  ✗ Failed${NC}\n"
    fi
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

# Textures
echo -e "${YELLOW}--- Textures ---${NC}"
run_test "Textured Cube" "-o --texture-dir scripts/test-assets --objects type=cube:texture=test_checker.png -s $OUTPUT_DIR/12-textured-cube.png"
run_test "Textured Cube + Glass" "-o --texture-dir scripts/test-assets --objects type=cube:texture=test_checker.png:material=glass -s $OUTPUT_DIR/13-textured-glass.png"

# Multi-Object Scenes (note: cannot mix spheres with cubes/meshes yet - TD-5)
echo -e "${YELLOW}--- Multi-Object Scenes ---${NC}"
run_test "Two Spheres" "-o --objects type=sphere:pos=-1.5,0,0 --objects type=sphere:pos=1.5,0,0 -s $OUTPUT_DIR/14-two-spheres.png"
run_test "Two Cubes" "-o --objects type=cube:pos=-1.5,0,0:material=glass --objects type=cube:pos=1.5,0,0 -s $OUTPUT_DIR/15-two-cubes.png"
run_test "RGB Cubes" "-o --objects type=cube:pos=-2,0,0:color=#ff0000 --objects type=cube:pos=0,0,0:color=#00ff00 --objects type=cube:pos=2,0,0:color=#0000ff -s $OUTPUT_DIR/16-rgb-cubes.png"

# Lighting
echo -e "${YELLOW}--- Lighting ---${NC}"
run_test "Point Light" "-o --objects type=sphere --light point:2,3,2:1.0 -s $OUTPUT_DIR/17-point-light.png"
run_test "Directional Light" "-o --objects type=sphere --light directional:-1,-1,-1:0.8 -s $OUTPUT_DIR/18-directional.png"
run_test "Colored Light (orange)" "-o --objects type=sphere --light point:2,3,2:1.0:ff8800 -s $OUTPUT_DIR/19-colored-light.png"
run_test "Two Colored Lights" "-o --objects type=sphere --light point:-3,3,2:1.0:ff0000 --light point:3,3,2:1.0:0000ff -s $OUTPUT_DIR/20-two-lights.png"

# Shadows (using homogeneous scenes - cannot mix spheres with cubes yet)
echo -e "${YELLOW}--- Shadows ---${NC}"
run_test "Shadows On (spheres)" "-o --objects type=sphere:pos=-1,0.5,0 --objects type=sphere:pos=1,0,0:size=0.5 --shadows -s $OUTPUT_DIR/21-shadows-on.png"
run_test "Shadows Off (spheres)" "-o --objects type=sphere:pos=-1,0.5,0 --objects type=sphere:pos=1,0,0:size=0.5 -s $OUTPUT_DIR/22-shadows-off.png"

# Antialiasing
echo -e "${YELLOW}--- Antialiasing ---${NC}"
run_test "No AA" "-o --objects type=sphere -s $OUTPUT_DIR/23-no-aa.png"
run_test "AA On" "-o --objects type=sphere --antialiasing -s $OUTPUT_DIR/24-aa-on.png"
run_test "AA Depth 4" "-o --objects type=sphere --antialiasing --aa-max-depth 4 -s $OUTPUT_DIR/25-aa-depth4.png"

# Ground Plane
echo -e "${YELLOW}--- Ground Plane ---${NC}"
run_test "Default Plane" "-o --objects type=sphere --plane y:-1 -s $OUTPUT_DIR/26-plane-default.png"
run_test "Solid Plane" "-o --objects type=sphere --plane y:-1 --plane-color 808080 -s $OUTPUT_DIR/27-plane-solid.png"
run_test "Checkered Plane" "-o --objects type=sphere --plane y:-1 --plane-color ffffff:000000 -s $OUTPUT_DIR/28-plane-checker.png"

# Camera
echo -e "${YELLOW}--- Camera ---${NC}"
run_test "High Camera" "-o --objects type=sphere --camera-pos 0,5,10 --camera-lookat 0,0,0 -s $OUTPUT_DIR/29-camera-high.png"
run_test "Close Camera" "-o --objects type=sphere --camera-pos 0,1,3 --camera-lookat 0,0,0 -s $OUTPUT_DIR/30-camera-close.png"

# Image Size
echo -e "${YELLOW}--- Image Size ---${NC}"
run_test "HD 1920x1080" "-o --objects type=sphere --width 1920 --height 1080 -s $OUTPUT_DIR/31-hd.png"
run_test "Square 512x512" "-o --objects type=sphere --width 512 --height 512 -s $OUTPUT_DIR/32-square.png"

# 4D Tesseract
echo -e "${YELLOW}--- 4D Tesseract ---${NC}"
run_test "Tesseract (default)" "-o --objects type=tesseract -s $OUTPUT_DIR/33-tesseract.png"
run_test "Tesseract (rotated)" "-o --objects type=tesseract:rot-xw=45:rot-yw=30 -s $OUTPUT_DIR/34-tesseract-rot.png"
run_test "Tesseract (projection)" "-o --objects type=tesseract:eye-w=4.0:screen-w=2.0 -s $OUTPUT_DIR/35-tesseract-proj.png"
run_test "Tesseract (glass)" "-o --objects type=tesseract:material=glass -s $OUTPUT_DIR/36-tesseract-glass.png"
run_test "Tesseract (chrome)" "-o --objects type=tesseract:material=chrome -s $OUTPUT_DIR/37-tesseract-chrome.png"

# Tesseract Edge Rendering (Phase 4 feature)
echo -e "${YELLOW}--- Tesseract Edge Rendering ---${NC}"
run_test "Tesseract with chrome edges" "-o --objects type=tesseract:edge-radius=0.02:edge-material=chrome -s $OUTPUT_DIR/38-tesseract-chrome-edges.png"
run_test "Tesseract with cyan emissive edges" "-o --objects type=tesseract:edge-radius=0.03:edge-color=#00ffff:edge-emission=3.0 -s $OUTPUT_DIR/39-tesseract-cyan-edges.png"
run_test "Tesseract glass with gold edges" "-o --objects type=tesseract:material=glass:edge-material=gold:edge-radius=0.025 -s $OUTPUT_DIR/40-tesseract-glass-gold.png"
run_test "Tesseract rotated with edges" "-o --objects type=tesseract:rot-xw=45:rot-yw=30:edge-material=chrome:edge-radius=0.02 -s $OUTPUT_DIR/41-tesseract-rot-edges.png"
run_test "Tesseract film with emissive edges" "-o --objects type=tesseract:material=film:edge-color=#ff00ff:edge-emission=5.0:edge-radius=0.03 -s $OUTPUT_DIR/42-tesseract-film-magenta.png"

# Film and Parchment Materials
echo -e "${YELLOW}--- Film and Parchment Materials ---${NC}"
run_test "Film sphere" "-o --objects type=sphere:material=film -s $OUTPUT_DIR/43-film-sphere.png"
run_test "Parchment cube" "-o --objects type=cube:material=parchment -s $OUTPUT_DIR/44-parchment-cube.png"
run_test "Film with emission" "-o --objects type=sphere:material=film:emission=3.0 -s $OUTPUT_DIR/45-film-emissive.png"

# Thin-Film Interference (physically-based, Airy formula)
echo -e "${YELLOW}--- Thin-Film Interference ---${NC}"
run_test "Film 300nm (violet/blue)" "-o --objects type=sphere:material=film:film-thickness=300 --plane y:-2 -s $OUTPUT_DIR/83-film-300nm.png"
run_test "Film 500nm (green, default)" "-o --objects type=sphere:material=film --plane y:-2 -s $OUTPUT_DIR/84-film-500nm.png"
run_test "Film 700nm (red/orange)" "-o --objects type=sphere:material=film:film-thickness=700 --plane y:-2 -s $OUTPUT_DIR/85-film-700nm.png"
run_test "Three thicknesses side by side" "-o --objects type=sphere:pos=-2,0,0:material=film:film-thickness=300 --objects type=sphere:pos=0,0,0:material=film --objects type=sphere:pos=2,0,0:material=film:film-thickness=700 --plane y:-2 -s $OUTPUT_DIR/86-film-three-thicknesses.png"
run_test "Chrome with oil film 300nm" "-o --objects type=sphere:material=chrome:film-thickness=300 --plane y:-2 -s $OUTPUT_DIR/87-chrome-film-coated.png"
run_test "Film 2000nm (thick, averaged grey)" "-o --objects type=sphere:material=film:film-thickness=2000 --plane y:-2 -s $OUTPUT_DIR/88-film-2000nm.png"

# 4D Menger Sponges
echo -e "${YELLOW}--- 4D Menger Sponges ---${NC}"
run_test "TesseractSponge Level 0" "-o --objects type=tesseract-sponge:level=0 -s $OUTPUT_DIR/46-tesseract-sponge-l0.png"
run_test "TesseractSponge Level 1" "-o --objects type=tesseract-sponge:level=1 -s $OUTPUT_DIR/47-tesseract-sponge-l1.png"
run_test "TesseractSponge2 Level 1" "-o --objects type=tesseract-sponge-2:level=1 -s $OUTPUT_DIR/48-tesseract-sponge2-l1.png"
run_test "TesseractSponge2 Level 2" "-o --objects type=tesseract-sponge-2:level=2 -s $OUTPUT_DIR/49-tesseract-sponge2-l2.png"
run_test "TesseractSponge (rotated)" "-o --objects type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30 -s $OUTPUT_DIR/50-tesseract-sponge-rot.png"
run_test "TesseractSponge (glass)" "-o --objects type=tesseract-sponge:level=1:material=glass -s $OUTPUT_DIR/51-tesseract-sponge-glass.png"
run_test "TesseractSponge2 (chrome)" "-o --objects type=tesseract-sponge-2:level=1:material=chrome -s $OUTPUT_DIR/52-tesseract-sponge2-chrome.png"
run_test "TesseractSponge with chrome edges" "-o --objects type=tesseract-sponge:level=1:edge-material=chrome:edge-radius=0.015 -s $OUTPUT_DIR/53-tesseract-sponge-edges.png"
run_test "TesseractSponge2 glass + gold edges" "-o --objects type=tesseract-sponge-2:level=1:material=glass:edge-material=gold:edge-radius=0.02 -s $OUTPUT_DIR/54-tesseract-sponge2-glass-gold.png"
run_test "Mixed 4D sponges" "-o --objects type=tesseract-sponge:level=1:pos=-1.5,0,0:color=#FF4444 --objects type=tesseract-sponge-2:level=1:pos=1.5,0,0:color=#44FF44 -s $OUTPUT_DIR/55-mixed-4d-sponges.png"
run_test "4D sponge + 3D sphere" "-o --objects type=tesseract-sponge-2:level=1:pos=-1.5,0,0:material=glass --objects type=sphere:pos=1.5,0,0:material=chrome -s $OUTPUT_DIR/56-4d-sponge-3d-sphere.png"

# Fractional Levels (per-vertex alpha blending)
echo -e "${YELLOW}--- Fractional Levels (4D Sponges) ---${NC}"
run_test "TesseractSponge Level 0.5" "-o --objects type=tesseract-sponge:level=0.5 -s $OUTPUT_DIR/61-tesseract-sponge-l0.5.png"
run_test "TesseractSponge Level 1.25" "-o --objects type=tesseract-sponge:level=1.25 -s $OUTPUT_DIR/62-tesseract-sponge-l1.25.png"
run_test "TesseractSponge Level 1.5" "-o --objects type=tesseract-sponge:level=1.5 -s $OUTPUT_DIR/63-tesseract-sponge-l1.5.png"
run_test "TesseractSponge Level 1.75" "-o --objects type=tesseract-sponge:level=1.75 -s $OUTPUT_DIR/64-tesseract-sponge-l1.75.png"
run_test "TesseractSponge2 Level 0.9" "-o --objects type=tesseract-sponge-2:level=0.9 -s $OUTPUT_DIR/65-tesseract-sponge2-l0.9.png"
run_test "TesseractSponge2 Level 1.3" "-o --objects type=tesseract-sponge-2:level=1.3 -s $OUTPUT_DIR/66-tesseract-sponge2-l1.3.png"
run_test "Fractional level (glass)" "-o --objects type=tesseract-sponge:level=1.5:material=glass -s $OUTPUT_DIR/67-frac-glass.png"
run_test "Fractional level (rotated)" "-o --objects type=tesseract-sponge-2:level=1.4:rot-xw=30:rot-yw=20 -s $OUTPUT_DIR/68-frac-rotated.png"
run_test "Mixed fractional + integer" "-o --objects type=tesseract-sponge:level=1.5:pos=-1.5,0,0:color=#FF8844 --objects type=tesseract-sponge:level=1:pos=1.5,0,0:color=#4488FF -s $OUTPUT_DIR/69-mixed-frac-int.png"
run_test "Fractional with edges" "-o --objects type=tesseract-sponge:level=1.6:edge-material=chrome:edge-radius=0.015 -s $OUTPUT_DIR/70-frac-edges.png"

echo -e "${YELLOW}--- Fractional Levels (3D Sponges) ---${NC}"
run_test "SpongeByVolume Level 0.5" "-o --objects type=sponge-volume:level=0.5 -s $OUTPUT_DIR/71-sponge-volume-l0.5.png"
run_test "SpongeByVolume Level 1.25" "-o --objects type=sponge-volume:level=1.25 -s $OUTPUT_DIR/72-sponge-volume-l1.25.png"
run_test "SpongeByVolume Level 1.5" "-o --objects type=sponge-volume:level=1.5 -s $OUTPUT_DIR/73-sponge-volume-l1.5.png"
run_test "SpongeByVolume Level 1.75" "-o --objects type=sponge-volume:level=1.75 -s $OUTPUT_DIR/74-sponge-volume-l1.75.png"
run_test "SpongeBySurface Level 0.5" "-o --objects type=sponge-surface:level=0.5 -s $OUTPUT_DIR/75-sponge-surface-l0.5.png"
run_test "SpongeBySurface Level 1.25" "-o --objects type=sponge-surface:level=1.25 -s $OUTPUT_DIR/76-sponge-surface-l1.25.png"
run_test "SpongeBySurface Level 1.5" "-o --objects type=sponge-surface:level=1.5 -s $OUTPUT_DIR/77-sponge-surface-l1.5.png"
run_test "SpongeBySurface Level 1.75" "-o --objects type=sponge-surface:level=1.75 -s $OUTPUT_DIR/78-sponge-surface-l1.75.png"
run_test "3D Fractional (glass)" "-o --objects type=sponge-volume:level=1.5:material=glass -s $OUTPUT_DIR/79-3d-frac-glass.png"
run_test "3D Fractional (chrome)" "-o --objects type=sponge-surface:level=1.3:material=chrome -s $OUTPUT_DIR/80-3d-frac-chrome.png"
run_test "3D Mixed fractional + integer" "-o --objects type=sponge-volume:level=1.5:pos=-1.5,0,0:color=#FF8844 --objects type=sponge-volume:level=1:pos=1.5,0,0:color=#4488FF -s $OUTPUT_DIR/81-3d-mixed-frac-int.png"
run_test "3D Volume vs Surface frac" "-o --objects type=sponge-volume:level=1.5:pos=-1.5,0,0 --objects type=sponge-surface:level=1.5:pos=1.5,0,0 -s $OUTPUT_DIR/82-volume-vs-surface-frac.png"

# Caustics
echo -e "${YELLOW}--- Caustics (experimental) ---${NC}"
run_test "Caustics" "-o --objects type=sphere:material=glass --caustics --caustics-photons 10000 -s $OUTPUT_DIR/57-caustics.png"

# DSL Scenes
echo -e "${YELLOW}--- DSL Scenes ---${NC}"
run_test "DSL: SimpleScene" "-o --scene examples.dsl.SimpleScene -s $OUTPUT_DIR/90-dsl-simple.png"
run_test "DSL: ThreeMaterials" "-o --scene examples.dsl.ThreeMaterials -s $OUTPUT_DIR/91-dsl-three-materials.png"
run_test "DSL: GlassSphere" "-o --scene examples.dsl.GlassSphere -s $OUTPUT_DIR/92-dsl-glass-sphere.png"
run_test "DSL: TesseractDemo" "-o --scene examples.dsl.TesseractDemo -s $OUTPUT_DIR/93-dsl-tesseract.png"
run_test "DSL: FilmSphere" "-o --scene examples.dsl.FilmSphere -s $OUTPUT_DIR/94-dsl-film-sphere.png"
run_test "DSL: SpongeShowcase" "-o --scene examples.dsl.SpongeShowcase -s $OUTPUT_DIR/95-dsl-sponge-showcase.png"
run_test "DSL: MengerShowcase" "-o --scene examples.dsl.MengerShowcase -s $OUTPUT_DIR/96-dsl-menger-showcase.png"

# Animated DSL Scenes (t-parameter)
echo -e "${YELLOW}--- Animated DSL Scenes (t-parameter) ---${NC}"
run_test "OrbitingSphere t=0" "-o --scene examples.dsl.OrbitingSphere --t 0 -s $OUTPUT_DIR/100-orbiting-t0.png"
run_test "OrbitingSphere t=1.57" "-o --scene examples.dsl.OrbitingSphere --t 1.57 -s $OUTPUT_DIR/101-orbiting-t1.57.png"
run_test "OrbitingSphere t=3.14" "-o --scene examples.dsl.OrbitingSphere --t 3.14 -s $OUTPUT_DIR/102-orbiting-t3.14.png"
run_test "PulsingSponge t=0" "-o --scene examples.dsl.PulsingSponge --t 0 -s $OUTPUT_DIR/103-pulsing-t0.png"
run_test "PulsingSponge t=1.5" "-o --scene examples.dsl.PulsingSponge --t 1.5 -s $OUTPUT_DIR/104-pulsing-t1.5.png"
run_test "PulsingSponge t=3" "-o --scene examples.dsl.PulsingSponge --t 3 -s $OUTPUT_DIR/105-pulsing-t3.png"

# Showcase
echo -e "${YELLOW}--- Showcase ---${NC}"
run_test "High Quality Render" "-o --objects type=sponge-surface:level=1:material=glass --antialiasing --aa-max-depth 3 --shadows --light point:3,5,3:1.2 --width 1920 --height 1080 -s $OUTPUT_DIR/58-showcase.png"
run_test "Tesseract Showcase (glass with emissive edges)" "-o --objects type=tesseract:material=glass:edge-color=#00ffff:edge-emission=5.0:edge-radius=0.025:rot-xw=20:rot-yw=15 --antialiasing --shadows --width 1920 --height 1080 -s $OUTPUT_DIR/59-tesseract-showcase.png"
run_test "4D Sponge Showcase (glass with edges)" "-o --objects type=tesseract-sponge-2:level=2:material=glass:edge-material=chrome:edge-radius=0.015:rot-xw=30:rot-yw=20 --antialiasing --shadows --width 1920 --height 1080 -s $OUTPUT_DIR/60-tesseract-sponge-showcase.png"

echo -e "${BLUE}=== Static Tests Complete ===${NC}"
echo -e "Output files in: ${GREEN}$OUTPUT_DIR/${NC}"
if [ "$UPDATE_REFERENCES" = true ]; then
    echo -e "Reference images updated in: ${YELLOW}$REFERENCE_DIR/${NC}"
fi
echo ""
ls -la "$OUTPUT_DIR"/*.png | awk '{print $9, $5}' | column -t
echo ""

fi  # End of static tests block

# Interactive tests
echo -e "${BLUE}=== Interactive Tests ===${NC}"
echo -e "${YELLOW}The following tests require manual interaction.${NC}"
echo -e "Controls: Mouse drag = rotate, Scroll = zoom, Q/ESC = quit\n"

interactive_tests=(
    "Basic sphere:-o --objects type=sphere"
    "Glass refraction:-o --objects type=sphere:material=glass"
    "Water:-o --objects type=sphere:material=water"
    "Diamond:-o --objects type=sphere:material=diamond"
    "Chrome (metallic):-o --objects type=sphere:material=chrome"
    "Gold (metallic):-o --objects type=sphere:material=gold"
    "Copper (metallic):-o --objects type=sphere:material=copper"
    "Film (translucent):-o --objects type=sphere:material=film"
    "Film 300nm (violet/blue tint):-o --objects type=sphere:material=film:film-thickness=300 --plane y:-2"
    "Film 500nm (green tint, default):-o --objects type=sphere:material=film --plane y:-2"
    "Film 700nm (red/orange tint):-o --objects type=sphere:material=film:film-thickness=700 --plane y:-2"
    "Three film thicknesses:-o --objects type=sphere:pos=-2,0,0:material=film:film-thickness=300 --objects type=sphere:pos=0,0,0:material=film --objects type=sphere:pos=2,0,0:material=film:film-thickness=700 --plane y:-2"
    "Chrome with oily film coat:-o --objects type=sphere:material=chrome:film-thickness=300 --plane y:-2"
    "Parchment (semi-translucent):-o --objects type=cube:material=parchment"
    "Metal (blue):-o --objects type=sphere:material=metal:color=#4488ff"
    "Plastic (red):-o --objects type=sphere:material=plastic:color=#ff4444"
    "Matte (green):-o --objects type=sphere:material=matte:color=#44ff44"
    "Emissive sphere:-o --objects type=sphere:material=film:emission=5.0:color=#ff8800"
    "Textured cube:-o --texture-dir scripts/test-assets --objects type=cube:texture=test_checker.png"
    "Textured cube + glass:-o --texture-dir scripts/test-assets --objects type=cube:texture=test_checker.png:material=glass"
    "Two spheres + shadows:-o --objects type=sphere:pos=-1.5,0,0:material=glass --objects type=sphere:pos=1.5,0,0 --shadows"
    "Sponge surface L1:-o --objects type=sponge-surface:level=1"
    "Cube Sponge L1:-o --objects type=cube-sponge:level=1:color=#00ffff --max-instances 64"
    "Tesseract (4D hypercube):-o --objects type=tesseract"
    "Tesseract (4D rotated):-o --objects type=tesseract:rot-xw=45:rot-yw=30:rot-zw=15"
    "Tesseract (glass):-o --objects type=tesseract:material=glass"
    "Tesseract (chrome):-o --objects type=tesseract:material=chrome"
    "Tesseract with chrome edges:-o --objects type=tesseract:edge-material=chrome:edge-radius=0.02"
    "Tesseract with cyan glow edges:-o --objects type=tesseract:edge-color=#00ffff:edge-emission=5.0:edge-radius=0.03"
    "Tesseract glass + gold edges:-o --objects type=tesseract:material=glass:edge-material=gold:edge-radius=0.025"
    "Tesseract film + magenta glow:-o --objects type=tesseract:material=film:edge-color=#ff00ff:edge-emission=5.0:edge-radius=0.03"
    "Tesseract rotated with edges:-o --objects type=tesseract:rot-xw=45:rot-yw=30:edge-material=chrome:edge-radius=0.02"
    "TesseractSponge L1 (4D Menger):-o --objects type=tesseract-sponge:level=1"
    "TesseractSponge L1 rotated:-o --objects type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30"
    "TesseractSponge L1 glass:-o --objects type=tesseract-sponge:level=1:material=glass"
    "TesseractSponge2 L1 (surface 4D):-o --objects type=tesseract-sponge-2:level=1"
    "TesseractSponge2 L2:-o --objects type=tesseract-sponge-2:level=2"
    "TesseractSponge2 L1 chrome:-o --objects type=tesseract-sponge-2:level=1:material=chrome"
    "TesseractSponge L1 + chrome edges:-o --objects type=tesseract-sponge:level=1:edge-material=chrome:edge-radius=0.015"
    "TesseractSponge2 glass + gold edges:-o --objects type=tesseract-sponge-2:level=1:material=glass:edge-material=gold:edge-radius=0.02"
    "Mixed 4D sponges (red+green):-o --objects type=tesseract-sponge:level=1:pos=-1.5,0,0:color=#ff4444 --objects type=tesseract-sponge-2:level=1:pos=1.5,0,0:color=#44ff44"
    "4D sponge + 3D sphere:-o --objects type=tesseract-sponge-2:level=1:pos=-1.5,0,0:material=glass --objects type=sphere:pos=1.5,0,0:material=chrome"
    "TesseractSponge L0.5 (fractional):-o --objects type=tesseract-sponge:level=0.5"
    "TesseractSponge L1.5 (fractional):-o --objects type=tesseract-sponge:level=1.5"
    "TesseractSponge2 L1.3 (fractional):-o --objects type=tesseract-sponge-2:level=1.3"
    "Fractional L1.5 glass:-o --objects type=tesseract-sponge:level=1.5:material=glass"
    "Fractional L1.4 rotated:-o --objects type=tesseract-sponge-2:level=1.4:rot-xw=30:rot-yw=20"
    "Mixed fractional + integer:-o --objects type=tesseract-sponge:level=1.5:pos=-1.5,0,0 --objects type=tesseract-sponge:level=1:pos=1.5,0,0"
    "SpongeByVolume L0.5 (3D fractional):-o --objects type=sponge-volume:level=0.5"
    "SpongeByVolume L1.5 (3D fractional):-o --objects type=sponge-volume:level=1.5"
    "SpongeBySurface L1.25 (3D fractional):-o --objects type=sponge-surface:level=1.25"
    "SpongeBySurface L1.75 (3D fractional):-o --objects type=sponge-surface:level=1.75"
    "3D Fractional glass:-o --objects type=sponge-volume:level=1.5:material=glass"
    "3D Fractional chrome:-o --objects type=sponge-surface:level=1.3:material=chrome"
    "3D Mixed frac levels:-o --objects type=sponge-volume:level=1.5:pos=-1.5,0,0:color=#FF8844 --objects type=sponge-volume:level=1:pos=1.5,0,0:color=#4488FF"
    "3D Volume vs Surface frac:-o --objects type=sponge-volume:level=1.5:pos=-1.5,0,0 --objects type=sponge-surface:level=1.5:pos=1.5,0,0"
    "Colored lights:-o --objects type=sphere --light point:-3,3,2:1.0:ff0000 --light point:3,3,2:1.0:0000ff"
    "Checkered plane:-o --objects type=sphere --plane y:-1 --plane-color ffffff:000000"
    "DSL: SimpleScene (minimal sphere + directional light):-o --scene examples.dsl.SimpleScene"
    "DSL: ThreeMaterials (glass / chrome / gold presets side by side):-o --scene examples.dsl.ThreeMaterials"
    "DSL: GlassSphere (glass sphere with caustics on white floor):-o --scene examples.dsl.GlassSphere"
    "DSL: TesseractDemo (4D hypercube, glass, custom projection):-o --scene examples.dsl.TesseractDemo"
    "DSL: FilmSphere (thin-film interference at 300 / 500 / 700 nm):-o --scene examples.dsl.FilmSphere"
    "DSL: SpongeShowcase (volume / surface / cube sponge types):-o --scene examples.dsl.SpongeShowcase"
    "DSL: MengerShowcase (level-2 gold Menger sponge):-o --scene examples.dsl.MengerShowcase"
    "DSL: CausticsDemo (glass sphere with photon-mapped caustics):-o --scene examples.dsl.CausticsDemo"
    "DSL: CustomMaterials (Material() constructor and factory methods):-o --scene examples.dsl.CustomMaterials"
    "DSL: ComplexLighting (5-light warm/cool setup):-o --scene examples.dsl.ComplexLighting"
    "DSL: ReusableComponents (shared Materials + Lighting objects):-o --scene examples.dsl.ReusableComponents"
    "DSL: OrbitingSphere t=0 (sphere in circular orbit, start):-o --scene examples.dsl.OrbitingSphere --t 0"
    "DSL: OrbitingSphere t=1.57 (sphere in circular orbit, quarter-turn):-o --scene examples.dsl.OrbitingSphere --t 1.57"
    "DSL: PulsingSponge t=1 (sponge growing to level 1):-o --scene examples.dsl.PulsingSponge --t 1"
    "DSL: PulsingSponge t=2 (sponge growing to level 2):-o --scene examples.dsl.PulsingSponge --t 2"
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
        echo -e "${YELLOW}Press Ctrl+Q to quit and return to menu (ESC resets 4D view)${NC}\n"
        $MENGER $args || true
        echo ""
    else
        echo "Invalid choice. Enter 0-${#interactive_tests[@]}"
    fi
done

echo -e "\n${GREEN}=== Manual Test Suite Complete ===${NC}"
echo -e "Review rendered images in: ${BLUE}$OUTPUT_DIR/${NC}"
