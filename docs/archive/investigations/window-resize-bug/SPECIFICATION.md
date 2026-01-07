# Window Resize Specification for OptiX Sphere Rendering

## Overview

This document specifies the exact behavior required for window resize operations when rendering spheres using OptiX ray tracing.

## Requirements

### 1. Horizontal Resize (Width Changes)

When the window width changes:
- The sphere MUST scale uniformly in BOTH width and height directions
- Example: If window width doubles, the sphere's diameter must double in both horizontal and vertical directions
- The sphere MUST remain perfectly circular (no distortion into an ellipse)
- The scaling is proportional to the width change
- The sphere remains centered in the window, if it was centered before. The look-at point remains constant.
- The rendered image must fill the entire window (no black borders)
- The checkered background MUST also scale uniformly with the sphere

### 2. Vertical Resize (Height Changes)

When the window height changes:
- The sphere size MUST NOT change at all
- The sphere MUST remain exactly the same size as before the height change
- The sphere MUST remain perfectly circular
- The sphere remains centered in the window, if it was centered before. The look-at point remains constant.
- The rendered image must fill the entire window (no black borders)
- the checkered background does NOT scale with height changes, but the visible area of the background changes to fill the window

### 3. Display Requirements

- The rendered image MUST fill the entire window
- There MUST NOT be any black borders (no letterboxing or pillarboxing)
- The image must stretch to fill the window if necessary

### 4. Camera Behavior

- The look-at point MUST remain constant at the defined look-at point during all resize operations
- Currently the look-at point is (0,0,0) but this may change in the future
- The look-at point must not "wander" or shift during resize

### 5. Initial State

- The initial sphere size must match what the unit tests expect (baseline behavior at 800x600 reference dimensions)

## Current Problem

**Unit tests pass** - The automated tests in `WindowResizeDiameterTest.scala` correctly demonstrate
the expected behavior when calling `updateImageDimensions()`, `setCamera()` and `render()` directly.

**Manual window resizing fails** - When the user manually resizes the LibGDX window, the behavior does NOT match the requirements:

**Observed wrong behaviors:**
1. Sphere appears too large initially
2. Vertical resize scales the sphere vertically (WRONG - should not scale at all)
3. Horizontal resize scales the sphere horizontally only (WRONG - should scale uniformly in both directions)
4. Resizing below original size causes black borders (WRONG - must fill window)
5. when resizing, the look-at point shifts (WRONG - must remain constant)

**Task**: Implement correct resize behavior that matches requirements.

## Working Implementation (From Tests)

**OptiX JNI Layer (C++):**
Standard perspective projection - keeps horizontal FOV constant:
- `ulen = tan(fov/2)` - stays constant (e.g., 0.414214 for 45°)
- `vlen = ulen / aspect_ratio` - varies with aspect ratio

**Scala Layer (OptiXResources.updateCameraAspectRatio):**
- Uses **fixed vertical FOV = 45°**
- Passes actual window dimensions to `renderer.updateImageDimensions(width, height)`
- Calls `renderer.setCamera(eye, lookAt, up, verticalFOV)`

**Result with working code:**
- Horizontal resize: Sphere scales uniformly in both directions
- Vertical resize: Sphere size remains constant
- This is standard perspective projection behavior

## The Problem

When manually resizing the window, something in the Scala layer is NOT passing the correct dimensions or FOV to the OptiX JNI layer. The task is to identify what's different between:

1. **Test path** (works): `updateImageDimensions()` + `setCamera()` called directly
2. **Manual resize path** (broken): `OptiXEngine.resize()` → `OptiXResources.updateCameraAspectRatio()` → calls same methods

**Key insight**: The OptiX JNI implementation is CORRECT (tests pass). The bug is in what the Scala layer sends to it during manual resize.

## Investigation Notes

### Code Flow Analysis

**Manual Resize Path:**
1. `OptiXEngine.resize(width, height)` - src/main/scala/menger/engines/OptiXEngine.scala:99-102
2. `OptiXResources.updateCameraAspectRatio(width, height)` - src/main/scala/menger/OptiXResources.scala:96-116
   - Calls `renderer.updateImageDimensions(width, height)` (line 108)
   - Calls `renderer.setCamera(eye, lookAt, up, verticalFOV=45°)` (line 115)
3. `OptiXEngine.render()` - src/main/scala/menger/engines/OptiXEngine.scala:69-76
   - Gets `width = Gdx.graphics.getWidth` (current window width)
   - Gets `height = Gdx.graphics.getHeight` (current window height)
   - Calls `optiXResources.renderScene(width, height)` (line 74)
4. `OptiXResources.renderScene(width, height)` - src/main/scala/menger/OptiXResources.scala:93-94
   - Calls `renderer.render(width, height)`
5. C++ `OptiXWrapper::render(width, height, output)` - optix-jni/src/main/native/OptiXWrapper.cpp:391-440
   - Does NOT call `updateImageDimensions()` (see comment lines 397-398)
   - Sets `params.image_width = width` and `params.image_height = height` (lines 427-428)
   - Uses cached camera parameters from last `setCamera()` call

**Test Path:**
1. Test calls `renderer.updateImageDimensions(width, height)`
2. Test calls `renderer.setCamera(eye, lookAt, up, fov=45°)`
3. Test calls `renderer.render(width, height)`

### Root Cause Analysis

**How Camera and Rendering Work:**

1. `updateImageDimensions(w, h)` caches dimensions in `impl->image_width/height`
2. `setCamera(...)` calculates camera vectors using the CACHED dimensions:
   ```cpp
   aspect_ratio = impl->image_width / impl->image_height
   ulen = tan(fov/2)
   vlen = ulen / aspect_ratio
   ```
3. `render(w, h)` sets `params.image_width/height` which the SHADER uses for pixel→ray conversion
4. **CRITICAL**: If `params.image_width/height` don't match the aspect ratio the camera was configured for, you get distortion!

**The Bug:**

During initialization (`OptiXResources.initialize()` → `createCamera()`):
- `setCamera()` is called WITHOUT calling `updateImageDimensions()` first (OptiXResources.scala:57-63)
- This means the camera is configured with UNINITIALIZED dimensions (probably 0x0 or garbage)
- When `render()` is first called with actual window dimensions, there's a mismatch
- This causes initial distortion

Then when manually resizing:
- IF LibGDX doesn't call `resize()` before every `render()`, the camera aspect ratio won't be updated
- OR if `resize()` is called but dimensions change between `resize()` and `render()`, there's a mismatch

**Implemented Fix:**

Made `render()` robust by checking if dimensions changed and updating camera if needed (Option 2):
- Added `lastCameraWidth` and `lastCameraHeight` fields to track dimensions (src/main/scala/menger/engines/OptiXEngine.scala:54-55)
- In `render()`, check if current dimensions differ from last camera dimensions (lines 78-82)
- If changed, call `updateCameraAspectRatio()` with new dimensions
- This handles both initialization (0x0 → actualDimensions) and manual resize cases
- Only updates camera when dimensions actually change (efficient)

This ensures camera aspect ratio always matches the render dimensions, preventing distortion.

## Test Cases

See `optix-jni/src/test/scala/menger/optix/WindowResizeDiameterTest.scala` for automated tests that verify the expected behavior with programmatic dimension changes.

## Files Involved

- `/home/lepr/workspace/menger/optix-jni/src/main/native/OptiXWrapper.cpp` - Camera setup (lines 165-184)
- `/home/lepr/workspace/menger/src/main/scala/menger/OptiXResources.scala` - Render dimension calculations
- `/home/lepr/workspace/menger/src/main/scala/menger/engines/OptiXEngine.scala` - Window resize handling and display
