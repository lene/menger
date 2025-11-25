# ImageMagick Methods for Measuring Sphere Diameter in Screenshots

This document describes various ImageMagick techniques for measuring sphere diameters in screenshots, particularly useful for validating the window resize behavior.

## Method 1: Edge Detection + Connected Components (Most Accurate)

```bash
# Convert to grayscale, find edges, then measure the largest connected component
convert screenshot.png -colorspace gray -edge 1 -threshold 50% -morphology close diamond:1 \
  -define connected-components:verbose=true \
  -connected-components 8 null: 2>&1 | \
  grep -E "^[[:space:]]*[0-9]+:" | \
  awk '{print $2}' | \
  sed 's/x/ /' | \
  awk '{if ($1 > max_width) max_width=$1; if ($2 > max_height) max_height=$2} END {print "Width:", max_width, "Height:", max_height}'
```

## Method 2: Histogram-Based Approach (For High Contrast)

```bash
# Assuming sphere is distinct from background
# First convert to binary (sphere=white, background=black)
convert screenshot.png -colorspace gray -threshold 50% binary.png

# Count white pixels in horizontal scan lines to find diameter
convert binary.png -scale 1x600! -depth 8 txt:- | \
  grep -c "white" | \
  awk '{print "Height in pixels:", $1}'

# Count white pixels in vertical scan lines
convert binary.png -scale 800x1! -depth 8 txt:- | \
  grep -o "white" | wc -l | \
  awk '{print "Width in pixels:", $1}'
```

## Method 3: Fuzz-Based Color Selection (If Sphere Has Distinct Color)

```bash
# Select pixels similar to sphere color, get bounding box
convert screenshot.png -fuzz 10% -fill red -opaque "rgb(100,100,100)" \
  -trim -format "%w x %h" info:
```

## Method 4: Profile Analysis (Simple but Effective)

```bash
# Extract horizontal profile through center and measure
convert screenshot.png -crop 800x1+0+300 -scale 800x1! -depth 8 txt:- | \
  awk -F'[,:]' '/gray\(255/{start=NR} /gray\(0/ && start {print NR-start; exit}'

# Extract vertical profile through center
convert screenshot.png -crop 1x600+400+0 -scale 1x600! -depth 8 txt:- | \
  awk -F'[,:]' '/gray\(255/{start=NR} /gray\(0/ && start {print NR-start; exit}'
```

## Method 5: Automated Script Combining Multiple Techniques

```bash
#!/bin/bash
# measure_sphere.sh

IMAGE=$1
TEMP_DIR=$(mktemp -d)

# Method 1: Edge detection
convert "$IMAGE" -colorspace gray \
  -edge 2 \
  -threshold 30% \
  -morphology close disk:2 \
  "$TEMP_DIR/edges.png"

# Find bounding box of largest object
convert "$TEMP_DIR/edges.png" \
  -define connected-components:area-threshold=1000 \
  -connected-components 4 \
  -auto-level \
  "$TEMP_DIR/components.png"

# Get measurements
MEASUREMENTS=$(convert "$TEMP_DIR/components.png" -trim -format "%w,%h,%X,%Y" info:)
WIDTH=$(echo $MEASUREMENTS | cut -d, -f1)
HEIGHT=$(echo $MEASUREMENTS | cut -d, -f2)

echo "Sphere diameter (width): $WIDTH pixels"
echo "Sphere diameter (height): $HEIGHT pixels"
echo "Circularity check: $(echo "scale=2; $WIDTH / $HEIGHT" | bc)"

rm -rf "$TEMP_DIR"
```

## Recommended Method for OptiX Sphere on Checkered Background

Given that we have a sphere rendered on a checkered background, the most reliable approach is:

```bash
# Function to measure sphere dimensions
measure_sphere() {
    local img=$1
    local name=$2

    # Convert to binary, isolate sphere, measure
    convert "$img" \
        -colorspace gray \
        -threshold 45% \
        -morphology open disk:2 \
        -morphology close disk:2 \
        -trim \
        -format "$name: Width=%w Height=%h AspectRatio=%[fx:w/h]\n" \
        info:
}

# Usage for our test sizes
measure_sphere screenshot_800x600.png "800x600"
measure_sphere screenshot_1600x600.png "1600x600"
measure_sphere screenshot_800x1200.png "800x1200"
```

This approach:
1. Converts to grayscale
2. Applies threshold to separate sphere from background
3. Cleans up with morphology operations (removes noise)
4. Trims to bounding box of the sphere
5. Reports dimensions and aspect ratio

## Key Metrics to Track

1. **Sphere Width/Height**: Actual pixel dimensions of the sphere
2. **Aspect Ratio**: Should always be ~1.0 for a perfect circle
3. **Scale Factor**: Compare dimensions between different window sizes

## Expected Results per Specification

| Window Size | Expected Sphere Diameter | Notes |
|------------|-------------------------|--------|
| 800x600 (base) | D pixels | Baseline measurement |
| 1600x600 | 2*D pixels | Width doubled → sphere doubles |
| 800x1200 | D pixels | Height doubled → sphere unchanged |

## Validation Criteria

- Aspect ratio should be 1.0 ± 0.02 (allowing for rasterization)
- Width resize: new_diameter / old_diameter ≈ new_width / old_width
- Height resize: new_diameter ≈ old_diameter (no change)

## Troubleshooting

If measurements are inconsistent:
1. Adjust threshold value based on sphere/background contrast
2. Increase morphology disk size for noisy images
3. Use `-blur 0x1` before edge detection for smoother edges
4. Consider using `-connected-components` with area threshold to filter small artifacts