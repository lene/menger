#!/bin/bash
# Manual test script for fractional level 4D sponges
# Tests the smooth transition between levels using alpha transparency

set -e

echo "=== Testing Fractional Level Support for 4D Sponges ==="
echo ""

# Test 1: Integer level (baseline)
echo "Test 1: Integer level 1.0 (baseline)"
sbt "run --optix --headless --objects type=tesseract-sponge:level=1:color=#FF4444 --save-name fractional-level-1.0.png --timeout 5"
echo ""

# Test 2: Fractional level 1.25 (75% transparent overlay)
echo "Test 2: Fractional level 1.25"
sbt "run --optix --headless --objects type=tesseract-sponge:level=1.25:color=#FF4444 --save-name fractional-level-1.25.png --timeout 5"
echo ""

# Test 3: Fractional level 1.5 (50% transparent overlay)
echo "Test 3: Fractional level 1.5"
sbt "run --optix --headless --objects type=tesseract-sponge:level=1.5:color=#FF4444 --save-name fractional-level-1.5.png --timeout 5"
echo ""

# Test 4: Fractional level 1.75 (25% transparent overlay)
echo "Test 4: Fractional level 1.75"
sbt "run --optix --headless --objects type=tesseract-sponge:level=1.75:color=#FF4444 --save-name fractional-level-1.75.png --timeout 5"
echo ""

# Test 5: Integer level 2.0 (next level)
echo "Test 5: Integer level 2.0 (next level)"
sbt "run --optix --headless --objects type=tesseract-sponge:level=2:color=#FF4444 --save-name fractional-level-2.0.png --timeout 5"
echo ""

# Test 6: TesseractSponge2 with fractional level
echo "Test 6: TesseractSponge2 with fractional level 1.5"
sbt "run --optix --headless --objects type=tesseract-sponge-2:level=1.5:color=#44FF44 --save-name fractional-level-sponge2-1.5.png --timeout 5"
echo ""

# Test 7: Multiple fractional levels in same scene
echo "Test 7: Multiple fractional levels in same scene"
sbt "run --optix --headless --objects type=tesseract-sponge:level=1.3:pos=-2,0,0:color=#FF4444 --objects type=tesseract-sponge:level=1.6:pos=0,0,0:color=#44FF44 --objects type=tesseract-sponge:level=1.9:pos=2,0,0:color=#4444FF --save-name fractional-level-multiple.png --timeout 10"
echo ""

echo "=== All tests complete ==="
echo "Generated images:"
ls -lh fractional-level-*.png
echo ""
echo "Expected behavior:"
echo "  - Level 1.0: Baseline (1,152 faces)"
echo "  - Level 1.25: Mix of level 2 (opaque) + level 1 (75% transparent)"
echo "  - Level 1.5: Mix of level 2 (opaque) + level 1 (50% transparent)"
echo "  - Level 1.75: Mix of level 2 (opaque) + level 1 (25% transparent)"
echo "  - Level 2.0: Full level 2 (55,296 faces)"
echo ""
echo "Visual check: Images should show smooth transition from level 1 to level 2"
