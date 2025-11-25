# Window Resize Bug Fix Attempts Documentation

## Executive Summary

This document comprehensively records all attempts to fix the OptiX sphere rendering window resize bug documented in `WINDOW_RESIZE_SPEC.md`. The primary issue is that spheres become distorted (elliptical) during window resizing instead of maintaining their circular shape. Unit tests pass but manual window resizing fails.

### Key Findings:
1. **Root Cause Identified**: OptiX renderer interprets FOV parameter as HORIZONTAL field of view, but we pass VERTICAL FOV value (45°)
2. **Failed Fix #1**: Adjusting FOV inversely to width scale made the problem worse (zoomed in instead of fixing distortion)
3. **Proposed Fix**: Change OptiXWrapper.cpp to interpret FOV as vertical (implemented but not yet tested due to compilation issues)

### Status: UNRESOLVED
- The bug remains unfixed in production
- A promising fix has been identified but not fully tested
- Multiple diagnostic tools and tests have been created for future debugging

## Problem Description

From `WINDOW_RESIZE_SPEC.md`:
- **Expected Behavior**:
  - Width resize: Sphere scales uniformly (stays circular, changes size)
  - Height resize: Sphere stays same size (stays circular)
  - No distortion, no black borders
- **Actual Behavior**:
  - Spheres become elliptical/distorted during resizes
  - Black borders appear when width is reduced
  - Unit tests pass but manual resize fails

## Investigation Timeline

### Phase 1: Initial Hypothesis and Logging

**Hypothesis**: FOV calculation mismatch between horizontal and vertical field of view.

**Actions Taken**:
1. Added comprehensive logging to `OptiXResources.scala`:

```scala
// Added to updateCameraAspectRatio method
logger.info(s"[OptiXResources] ====== UPDATE CAMERA ASPECT RATIO START ======")
logger.info(s"[OptiXResources] updateCameraAspectRatio called with ${width}x${height}")

val aspectRatio = width.toFloat / height.toFloat
val verticalFOV = 45f  // Fixed vertical FOV in degrees

logger.info(s"[OptiXResources] FOV calculations:")
logger.info(s"[OptiXResources]   - Aspect ratio: $aspectRatio (width/height = $width/$height)")
logger.info(s"[OptiXResources]   - Vertical FOV: $verticalFOV°")

logger.info(s"[OptiXResources] Calling updateImageDimensions(${width}, ${height})")
renderer.updateImageDimensions(width, height)

logger.info(s"[OptiXResources] Calling setCamera with vFOV=$verticalFOV°")
renderer.setCamera(eye, lookAt, up, verticalFOV)
logger.info(s"[OptiXResources] ====== UPDATE CAMERA ASPECT RATIO END ======")
```

### Phase 2: Failed Attempt #1 - FOV Scaling

**Approach**: Scale FOV inversely with window width to maintain sphere size

**Code Changes** (in `OptiXResources.scala`):
```scala
def updateCameraAspectRatio(width: Int, height: Int): Unit =
  val initialWidth = 800f
  val widthScale = width.toFloat / initialWidth
  val baseFOV = 45f
  val effectiveFOV = baseFOV / widthScale  // WRONG: This zooms in when width increases!

  renderer.setCamera(eye, lookAt, up, effectiveFOV)
```

**Result**: COMPLETE FAILURE
- User feedback: "you have achieved fuck all in fixing the resizing behaviour"
- Sphere zoomed IN when width increased (opposite of intended)
- Made the problem worse, not better

**Screenshots Provided**:
- `/home/lepr/2025-11-13-133743_800x600_scrot.png` - Initial (correct)
- `/home/lepr/2025-11-13-133903_1267x600_scrot.png` - Width increased (sphere distorted horizontally)
- `/home/lepr/2025-11-13-133913_682x600_scrot.png` - Width decreased (sphere stretched vertically + black border)
- `/home/lepr/2025-11-13-133922_682x885_scrot.png` - Height increased (sphere stretched vertically)

### Phase 3: FOV Convention Investigation

**Discovery**: Need to determine if OptiX interprets FOV as horizontal or vertical

**Test Code Created**: `FOVConventionTest.scala`
```scala
package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll

class FOVConventionTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  "OptiX FOV parameter" should "determine whether it's horizontal or vertical FOV" in {
    assume(OptiXRenderer.isLibraryLoaded, "OptiX library not loaded")

    val renderer = new OptiXRenderer()
    renderer.initialize() shouldBe true

    try {
      // Set up a simple green sphere at origin
      renderer.setSphere(0f, 0f, 0f, 1.0f)
      renderer.setSphereColor(0f, 1f, 0f, 1f)  // Green
      renderer.setPlane(2, true, -10f)  // Plane at z = -10

      // Camera looking at origin from z=5
      val eye = Array(0f, 0f, 5f)
      val lookAt = Array(0f, 0f, 0f)
      val up = Array(0f, 1f, 0f)
      val fov = 45f

      println("\nTesting FOV convention with 45° FOV at different aspect ratios...")
      println("If FOV is HORIZONTAL: sphere width in pixels should be constant")
      println("If FOV is VERTICAL: sphere height in pixels should be constant")
      println()

      // Test 1: Square aspect ratio (baseline)
      renderer.updateImageDimensions(400, 400)
      renderer.setCamera(eye, lookAt, up, fov)
      val img1 = renderer.render(400, 400)
      val (width1, height1) = measureSphere(img1, 400, 400)
      println(f"400x400 (1:1): Sphere width=$width1%3d pixels, height=$height1%3d pixels")

      // Test 2: Wide aspect ratio
      renderer.updateImageDimensions(800, 400)
      renderer.setCamera(eye, lookAt, up, fov)
      val img2 = renderer.render(800, 400)
      val (width2, height2) = measureSphere(img2, 800, 400)
      println(f"800x400 (2:1): Sphere width=$width2%3d pixels, height=$height2%3d pixels")

      // Test 3: Tall aspect ratio
      renderer.updateImageDimensions(400, 800)
      renderer.setCamera(eye, lookAt, up, fov)
      val img3 = renderer.render(400, 800)
      val (width3, height3) = measureSphere(img3, 400, 800)
      println(f"400x800 (1:2): Sphere width=$width3%3d pixels, height=$height3%3d pixels")

      val widthVariation = Math.max(Math.max(width1, width2), width3) -
                           Math.min(Math.min(width1, width2), width3)
      val heightVariation = Math.max(Math.max(height1, height2), height3) -
                           Math.min(Math.min(height1, height2), height3)

      println()
      println(s"Width variation:  $widthVariation pixels")
      println(s"Height variation: $heightVariation pixels")

      if (widthVariation < heightVariation) {
        println("✓ FOV is HORIZONTAL (width is more constant than height)")
      } else if (heightVariation < widthVariation) {
        println("✓ FOV is VERTICAL (height is more constant than width)")
      } else {
        println("✗ Neither width nor height is constant - unexpected behavior")
      }

    } finally {
      renderer.dispose()
    }
  }

  private def measureSphere(img: Array[Byte], width: Int, height: Int): (Int, Int) = {
    // Measure horizontal diameter at center row
    val centerY = height / 2
    var hStart = -1
    var hEnd = -1
    for (x <- 0 until width) {
      val idx = (centerY * width + x) * 4
      val g = img(idx + 1) & 0xFF  // Green channel
      if (g > 128) {  // Bright green
        if (hStart == -1) hStart = x
        hEnd = x
      }
    }
    val sphereWidth = if (hStart >= 0) hEnd - hStart + 1 else 0

    // Measure vertical diameter at center column
    val centerX = width / 2
    var vStart = -1
    var vEnd = -1
    for (y <- 0 until height) {
      val idx = (y * width + centerX) * 4
      val g = img(idx + 1) & 0xFF  // Green channel
      if (g > 128) {  // Bright green
        if (vStart == -1) vStart = y
        vEnd = y
      }
    }
    val sphereHeight = if (vStart >= 0) vEnd - vStart + 1 else 0

    (sphereWidth, sphereHeight)
  }
}
```

**Test Results**:
```
[OptiXWrapper] setCamera: dims=400x400 aspect=1 fov=45° ulen=0.414214 vlen=0.414214
[OptiXWrapper] setCamera: dims=800x400 aspect=2 fov=45° ulen=0.414214 vlen=0.207107
[OptiXWrapper] setCamera: dims=400x800 aspect=0.5 fov=45° ulen=0.414214 vlen=0.828427
```

**Analysis**:
- `ulen` (horizontal FOV tangent) stays CONSTANT at 0.414214
- `vlen` (vertical FOV tangent) VARIES with aspect ratio
- This proves OptiX interprets the FOV parameter as HORIZONTAL FOV
- We're passing 45° expecting it to be VERTICAL FOV

### Phase 4: Proposed Fix (Not Yet Tested)

**Root Cause**: FOV convention mismatch
- OptiX native code treats FOV as horizontal
- Scala code passes vertical FOV value

**Fix Implementation** (in `OptiXWrapper.cpp`):
```cpp
// BEFORE (treats FOV as horizontal):
float ulen = std::tan(fov * 0.5f * M_PI / 180.0f);
float vlen = ulen / aspect_ratio;

// AFTER (treats FOV as vertical):
float vlen = std::tan(fov * 0.5f * M_PI / 180.0f);  // Vertical FOV
float ulen = vlen * aspect_ratio;                    // Horizontal derived from vertical
```

**Status**: Fix implemented but not compiled/tested due to build system issues

## Test Scripts Created

### 1. Automated Window Resize Test
Location: `/tmp/test_window_resize.sh`

```bash
#!/bin/bash

echo "Starting OptiX window resize test..."

# Start the application with specific parameters
ENABLE_OPTIX_JNI=true sbt "run --optix --sponge-type sphere --radius 0.5 --color 00ff00" &
APP_PID=$!

# Wait for window to appear
sleep 10

# Find window ID
WINDOW_ID=$(xdotool search --name "Menger" | head -1)

if [ -z "$WINDOW_ID" ]; then
    echo "ERROR: Could not find application window"
    kill $APP_PID 2>/dev/null
    exit 1
fi

echo "Found window ID: $WINDOW_ID"

# Test sequence
echo "Initial size: 800x600"
sleep 3

echo "Test 1: Increase width to 1200x600"
xdotool windowsize $WINDOW_ID 1200 600
sleep 3

echo "Test 2: Decrease width to 600x600"
xdotool windowsize $WINDOW_ID 600 600
sleep 3

echo "Test 3: Increase height to 600x900"
xdotool windowsize $WINDOW_ID 600 900
sleep 3

echo "Test 4: Return to original 800x600"
xdotool windowsize $WINDOW_ID 800 600
sleep 3

echo "Test complete. Closing application..."
kill $APP_PID
```

### 2. Manual Test Launch Script
```bash
# Launch with green sphere for better visibility
ENABLE_OPTIX_JNI=true sbt "run --optix --sponge-type sphere --radius 0.5 --color 00ff00 --timeout 120"
```

## Diagnostic Outputs

### OptiXWrapper Debug Logs
The OptiXWrapper.cpp was modified to output detailed camera setup information:
```cpp
std::cout << "[OptiXWrapper] setCamera: dims=" << impl->image_width << "x" << impl->image_height
          << " aspect=" << aspect_ratio << " fov=" << fov << "° ulen=" << ulen
          << " vlen=" << vlen << std::endl;
```

### OptiXResources Logging
Comprehensive logging added to trace the entire resize flow from LibGDX through JNI to OptiX.

## Failed Approaches (DO NOT RETRY)

### ❌ 1. FOV Scaling Based on Window Width
**Why it failed**: Fundamentally wrong approach. Scaling FOV inversely with width causes zoom effects, not aspect ratio correction. This makes the sphere larger when width decreases, which is opposite of the desired behavior.

### ❌ 2. Adjusting FOV Without Understanding Convention
**Why it failed**: Without knowing whether OptiX uses horizontal or vertical FOV, any FOV adjustments are guesswork and likely to fail.

## Next Steps

1. **Complete the FOV convention fix**:
   - Rebuild native code with vertical FOV interpretation
   - Test with both unit tests and manual window resizing

2. **Alternative approaches if FOV fix fails**:
   - Investigate if OptiX has a separate aspect ratio parameter
   - Check if the projection matrix can be modified directly
   - Consider viewport clipping as a workaround

3. **Additional diagnostics needed**:
   - Capture actual projection matrices being used
   - Compare with working WindowResizeDiameterTest implementation
   - Profile the exact JNI calls during resize events

## File Locations

- Bug specification: `docs/WINDOW_RESIZE_SPEC.md`
- This document: `docs/WINDOW_RESIZE_FIX_ATTEMPTS.md`
- Test images: `optix-jni/fov_convention_*.ppm`
- Screenshots: `/home/lepr/2025-11-13-*.png`
- Modified source files:
  - `src/main/scala/menger/OptiXResources.scala`
  - `optix-jni/src/main/native/OptiXWrapper.cpp`
  - `optix-jni/src/test/scala/menger/optix/FOVConventionTest.scala`

## Conclusion

The window resize bug stems from a FOV convention mismatch between the Scala layer (expecting vertical FOV) and the OptiX native layer (interpreting as horizontal FOV). This causes spheres to become elliptical during window resizes. A fix has been identified and partially implemented but requires completion and testing.