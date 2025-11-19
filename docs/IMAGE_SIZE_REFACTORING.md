# ImageSize Type Introduction - Implementation Plan

## Overview
Replace all `(width: Int, height: Int)` parameter pairs with a unified `ImageSize` case class for better type safety and API clarity.

## Decision: Case Class vs Named Tuple

**Chosen:** Case Class

**Rationale:**
- Validation: Enforce positive dimensions at construction
- Public API: Part of OptiXRenderer's public interface
- Extensibility: Can add methods (area, aspectRatio) later
- Consistency: Matches pattern of other value types (Light, RayStats, RenderResult)
- Named tuples in codebase are only for private types (ByteVec, StartEnd)

## Implementation Steps

### 1. Create ImageSize Type
**File:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`
- Add `case class ImageSize(width: Int, height: Int)` with validation
- Position: After `Light` object, before `RayStats`
- Include preconditions: `require(width > 0)`, `require(height > 0)`

### 2. Update ThresholdConstants
**File:** `optix-jni/src/test/scala/menger/optix/ThresholdConstants.scala`
- Change tuple constants to ImageSize:
  - `SMALL_IMAGE_SIZE = ImageSize(10, 10)`
  - `TEST_IMAGE_SIZE = ImageSize(400, 300)`
  - `STANDARD_IMAGE_SIZE = ImageSize(800, 600)`

### 3. Add Overloaded Methods to OptiXRenderer
**File:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`
- Add `def updateImageDimensions(size: ImageSize): Unit`
- Add `def render(size: ImageSize): Option[Array[Byte]]`
- Add `def renderWithStats(size: ImageSize): RenderResult`
- Keep existing native methods unchanged (JNI boundary)

### 4. Add Overloaded Helper Functions (ImageValidation.scala)
**File:** `optix-jni/src/test/scala/menger/optix/ImageValidation.scala`
Add overloads for 11 functions:
- `imageByteSize(size: ImageSize): Int`
- `getRGBAt(imageData: Array[Byte], size: ImageSize, x: Int, y: Int): RGB`
- `getCenterPixel(imageData: Array[Byte], size: ImageSize): RGB`
- `spherePixelArea(imageData: Array[Byte], size: ImageSize, bgThreshold: Int = 30): Int`
- `detectSphereCenter(imageData: Array[Byte], size: ImageSize): (Int, Int)`
- `estimateSphereRadius(imageData: Array[Byte], size: ImageSize): Double`
- `brightnessStdDev(imageData: Array[Byte], size: ImageSize): Double`
- `edgeBrightness(imageData: Array[Byte], size: ImageSize): Double`
- `brightnessGradient(imageData: Array[Byte], size: ImageSize): Double`
- `colorChannelRatio(imageData: Array[Byte], size: ImageSize, channel: Int): Double`
- `backgroundVisibility(imageData: Array[Byte], size: ImageSize): Boolean`

### 5. Add Overloaded Helper Functions (ShadowValidation.scala)
**File:** `optix-jni/src/test/scala/menger/optix/ShadowValidation.scala`
Add overloads for 7 functions:
- `Region.bottomCenter(size: ImageSize, fraction: Double = 0.25): Region`
- `Region.topCenter(size: ImageSize, fraction: Double = 0.25): Region`
- `Region.leftSide(size: ImageSize, fraction: Double = 0.25): Region`
- `Region.rightSide(size: ImageSize, fraction: Double = 0.25): Region`
- `regionBrightness(imageData: Array[Byte], size: ImageSize, region: Region): Double`
- `brightnessContrast(imageData: Array[Byte], size: ImageSize, region1: Region, region2: Region): Double`
- `detectDarkestRegion(imageData: Array[Byte], size: ImageSize, gridSize: Int = 5): Region`

### 6. Add Overloaded Matchers (ImageMatchers.scala)
**File:** `optix-jni/src/test/scala/menger/optix/ImageMatchers.scala`
Add overloads for 12 matchers:
- `haveValidRGBASize(size: ImageSize): Matcher[Array[Byte]]`
- `haveSmallSphereArea(size: ImageSize): Matcher[Array[Byte]]`
- `haveMediumSphereArea(size: ImageSize): Matcher[Array[Byte]]`
- `haveLargeSphereArea(size: ImageSize): Matcher[Array[Byte]]`
- `beRedDominant(size: ImageSize, tolerance: Double): Matcher[Array[Byte]]`
- `beGreenDominant(size: ImageSize, tolerance: Double): Matcher[Array[Byte]]`
- `beBlueDominant(size: ImageSize, tolerance: Double): Matcher[Array[Byte]]`
- `beGrayscale(size: ImageSize): Matcher[Array[Byte]]`
- `showGlassRefraction(size: ImageSize): Matcher[Array[Byte]]`
- `showWaterRefraction(size: ImageSize): Matcher[Array[Byte]]`
- `showDiamondRefraction(size: ImageSize): Matcher[Array[Byte]]`
- `showBackground(size: ImageSize): Matcher[Array[Byte]]`

### 7. Update Test Helper Methods
**File:** `optix-jni/src/test/scala/menger/optix/RendererFixture.scala`
- Add `def renderImage(size: ImageSize): Array[Byte]`

**File:** `optix-jni/src/test/scala/menger/optix/WindowResizeDiameterTest.scala`
- Add `def measureVerticalDiameter(img: Array[Byte], size: ImageSize): Int`
- Add `def measureHorizontalDiameter(img: Array[Byte], size: ImageSize): Int`

**File:** `optix-jni/src/test/scala/menger/optix/TestScenarios.scala`
- Update `imageDimensions: Option[(Int, Int)]` to `imageDimensions: Option[ImageSize]`
- Update all usages in `applyTo` and `render` methods

### 8. Refactor Test Call Sites
Update ~200+ call sites to use ImageSize:
- Replace `STANDARD_IMAGE_SIZE._1, STANDARD_IMAGE_SIZE._2` → `STANDARD_IMAGE_SIZE`
- Replace `TEST_IMAGE_SIZE._1, TEST_IMAGE_SIZE._2` → `TEST_IMAGE_SIZE`
- Replace literal pairs like `renderer.render(800, 600)` → `renderer.render(ImageSize(800, 600))`
- Update all matcher calls, validation calls, etc.

### 9. Main Engine Updates (if needed)
**File:** `src/main/scala/menger/engines/OptiXEngine.scala`
- Add `def resize(size: ImageSize): Unit` overload
- Add `def renderWithStats(size: ImageSize): Array[Byte]` overload

### 10. Verify Tests Pass
- Run full test suite: `sbt test --warn`
- Verify all 818 tests still pass
- Commit changes

## Strategy
- Keep all existing methods (backward compatibility)
- Add new overloads that delegate to existing methods
- Systematically refactor call sites to use ImageSize
- This is a non-breaking change that improves type safety

## Expected Impact
- ~200+ call sites updated
- ~30 new overloaded methods added
- Net reduction in code verbosity (`.width, .height` vs `._1, ._2`)
- Zero runtime performance impact (case class inlining)
