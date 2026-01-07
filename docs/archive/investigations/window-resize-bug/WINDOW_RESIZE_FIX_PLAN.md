# Window Resize Bug Fix Plan

## Problem Summary (Facts from WINDOW_RESIZE_SPEC.md)

### Expected behavior:
- When window WIDTH changes: Sphere should scale uniformly in BOTH dimensions proportional to width change
- When window HEIGHT changes: Sphere size should NOT change at all
- Sphere must remain circular (no distortion)
- Look-at point must remain constant
- Window must be filled (no black borders)

### Actual observed behavior during manual resize:
1. Sphere appears too large initially
2. Vertical resize scales the sphere vertically (should not scale)
3. Horizontal resize scales the sphere horizontally only (should scale uniformly)
4. Resizing below original size causes black borders
5. Look-at point shifts during resize

### Known fact:
Unit tests in `WindowResizeDiameterTest.scala` pass when calling methods directly.

## Hypothesis

### Primary hypothesis:
The bug is in `OptiXResources.updateCameraAspectRatio()` which calculates but doesn't use horizontal FOV.

### Evidence supporting this hypothesis:
- Test path calls `setCamera()` with constant 45° FOV and works
- Runtime path goes through `updateCameraAspectRatio()` which also passes 45° to `setCamera()` but fails
- The method calculates horizontal FOV but never uses it (line 116)

### Alternative possibilities:
1. The C++ JNI layer interprets FOV differently in different contexts
2. Image dimensions aren't properly updated before camera calculations
3. There's state contamination between resize events
4. The rendering pipeline has different behavior for programmatic vs user-initiated resizes
5. LibGDX window events provide different dimension values than expected

## Testing Strategy

### Phase 1: Hypothesis Testing
1. Add extensive logging to trace exact values passed through resize pipeline
2. Create automated test to reproduce manual resize behavior programmatically
3. Compare exact parameter values between working test path and failing runtime path
4. Test alternative FOV values to understand their effect on scaling
5. Verify dimensions are correctly propagated to C++ layer

### Phase 2: Root Cause Verification
1. Temporarily modify `updateCameraAspectRatio()` to pass different FOV values
2. Test if passing horizontal FOV for width changes produces correct behavior
3. Test if FOV calculation formula is correct
4. Verify C++ layer's FOV interpretation matches expectations
5. Rule out alternative hypotheses through targeted experiments

### Phase 3: Fix Implementation (only after verification)
1. Implement the verified solution
2. Add comprehensive unit tests for the resize behavior
3. Create integration tests that simulate actual window resize events
4. Add regression tests to prevent future breakage

### Phase 4: Validation
1. Run all existing tests to ensure no regression
2. Run automated resize verification script
3. Manual testing with interactive resize
4. Verify all specification requirements are met

## Automated Testing Approach

### Test script will:
- Start app with `ENABLE_OPTIX_JNI=true sbt "run --optix --sponge-type sphere"`
- Use `wmctrl` to get window ID
- Use `xdotool` to programmatically resize: 800x600 → 1600x600 → 800x1200
- Capture screenshots with `scrot` at each size
- Measure sphere diameter using ImageMagick
- Compare measurements to specification requirements
- Generate report showing actual vs expected behavior

## Success Criteria
- Quantitative measurements prove specification compliance
- All existing tests continue to pass
- New tests prevent regression
- Root cause is definitively identified, not just hypothesized

## Files of Interest

### Primary suspects:
- `src/main/scala/menger/OptiXResources.scala` (lines 97-118) - updateCameraAspectRatio method
- `src/main/scala/menger/engines/OptiXEngine.scala` (lines 78-82, 111) - resize handling

### Reference implementation (working):
- `optix-jni/src/test/scala/menger/optix/WindowResizeDiameterTest.scala`

### C++ JNI layer:
- `optix-jni/src/main/native/OptiXWrapper.cpp` (lines 145-180 setCamera, 183-190 updateImageDimensions)

## Notes

This plan emphasizes:
1. Testing the hypothesis before assuming it's correct
2. Considering alternative explanations
3. Verifying the fix through quantitative measurement
4. Adding tests to prevent regression