# Design: Code Review Fixes — Sprint 12 + Sprint 11 Open Issues

**Date:** 2026-03-05
**Branch:** feature/sprint-12
**Issues addressed:** H6, M12, M13, M14, Sprint 11.1 M1/M2/L3, Sprint 11.2 M1/L1-L6/B1-B3, Sprint 11.3/11.4 M1/M2/M3/L1

---

## Section 1 — Structural Scala Changes

### 1a. `SceneClassifier` object (H6, M12, M14 partial)

**Problem:** `classifyScene`, `isTriangleMeshType`, and `selectSceneBuilder` are duplicated between
`OptiXEngine` and `AnimatedOptiXEngine`. The animated copy has diverged: it lacks
`TesseractEdgeSceneBuilder` support, causing animated scenes with 4D edge rendering to silently
use the wrong builder (M12). The methods are ~50 lines per copy.

**Solution:** Create `menger/engines/SceneClassifier.scala`:

```scala
object SceneClassifier:
  def classify(specs: List[ObjectSpec]): SceneType = ...
  def isTriangleMeshType(objectType: String): Boolean = ...
  def selectSceneBuilder(
    sceneType: SceneType,
    textureDir: Option[String]
  )(using ProfilingConfig): Option[SceneBuilder] = ...
```

`selectSceneBuilder` carries the full `TesseractEdgeSceneBuilder` logic from `OptiXEngine`.
Both engines delete their private copies and call `SceneClassifier.*`.

`OptiXEngine` retains `selectMeshBuilder` (mixed-scene only) and `rebuildScene` (engine-specific).
`AnimatedOptiXEngine` retains `buildSceneFromConfigs` (animation-specific).

The `SceneType` enum stays in `OptiXEngine.scala` (or moves to `SceneClassifier.scala` — same
file is cleanest since they're tightly coupled).

**Effect on line count:**
- `OptiXEngine`: ~488 → ~450 lines (after 1a + 1c)
- `AnimatedOptiXEngine`: ~191 → ~165 lines

### 1b. `computeEffectiveMaxInstances` helper (M14, Sprint 11.3/11.4 L2)

**Problem:** Three nearly-identical blocks in `OptiXEngine` compute effective max instances:
lines ~279–288 (createMultiObjectScene mesh path), ~312–321 (createMultiObjectScene general
path), ~385–394 (rebuildScene mesh path).

**Solution:** Extract to:

```scala
private def computeEffectiveMaxInstances(
  builder: SceneBuilder,
  specs: List[ObjectSpec]
): Int
```

Uses `userSetMaxInstances` and `execution.maxInstances` from the enclosing class.
All three call sites replace the ~10-line block with a single call.

**Effect:** `OptiXEngine` reduces by ~20 more lines → ~430 lines total after 1a+1b.

### 1c. `sceneFunction(t)` Try wrapping (M13)

**Problem:** In `AnimatedOptiXEngine.render()`, `sceneFunction(t)` is called bare. A user scene
function that throws for a particular `t` crashes the whole application instead of skipping the
frame.

**Solution:** Wrap in `Try`:

```scala
Try(sceneFunction(t)) match
  case scala.util.Failure(e) =>
    logger.error(s"Scene function threw for frame $frame (t=$t): ${e.getMessage}", e)
    frameCounter.incrementAndGet()
  case scala.util.Success(dslScene) =>
    // existing render logic
```

Frame is skipped (counter incremented) so the animation continues to completion.

### 1d. Document `event.eyeW` sentinel (Sprint 11.3/11.4 L1)

Add a comment at `OptiXEngine.scala` line ~93 explaining that `Const.defaultEyeW` is used as a
sentinel value to distinguish "no eyeW change" from "explicit eyeW change", and noting the
limitation: a user cannot intentionally reset eyeW to exactly `defaultEyeW` via an event.

---

## Section 2 — Scala Input Layer Changes

### 2a. `KeyRotation` trait (Sprint 11.1 M1)

**Problem:** `GdxKeyHandler` and `OptiXKeyHandler` have identical `factor: Map[Key, Int]` and
`angle(delta, keys): Float`. Two copies with no compile-time coupling.

**Solution:** Create `menger/input/KeyRotation.scala`:

```scala
trait KeyRotation:
  protected def rotatePressed: KeyPressTracker
  protected def rotateAngle: Float

  protected val factor: Map[Key, Int] = Map(
    Key.Right -> -1, Key.Left -> 1,
    Key.Up    ->  1, Key.Down -> -1,
    Key.PageUp -> 1, Key.PageDown -> -1
  )

  protected def angle(delta: Float, keys: Seq[Key]): Float =
    delta * rotateAngle * keys.find(rotatePressed.isPressed).map(factor(_)).getOrElse(0)
```

Both `GdxKeyHandler` and `OptiXKeyHandler`:
- `extends KeyHandler with KeyRotation`
- Change `private val rotatePressed` / `private val rotateAngle` to `protected val`
- Delete local `factor` Map and `angle()` method

### 2b. Move `MouseButton.toGdxButton` to `LibGDXConverters` (Sprint 11.1 M2)

**Problem:** `GdxCameraHandler` defines `extension (button: MouseButton) def toGdxButton: Int`.
Converter logic belongs in `LibGDXConverters`, not in a handler class.

**Solution:** Add to `LibGDXConverters`:

```scala
def toGdxButton(button: MouseButton): Int = button match
  case MouseButton.Left        => Buttons.LEFT
  case MouseButton.Right       => Buttons.RIGHT
  case MouseButton.Middle      => Buttons.MIDDLE
  case MouseButton.Unknown(c)  => c
```

Update `GdxCameraHandler` call sites to `LibGDXConverters.toGdxButton(button)`.
Remove extension method from `GdxCameraHandler`.

### 2c. Extract `eyeW` scroll formula to `CameraHandler` (Sprint 11.3/11.4 M1)

**Problem:** Both handlers contain the identical expression:

```scala
Math.pow(Const.Input.eyeScrollBase, amountY.toDouble).toFloat + Const.Input.eyeScrollOffset
```

**Solution:** Add to `CameraHandler` trait in `InputHandler.scala`:

```scala
protected def computeEyeW(amountY: Float): Float =
  Math.pow(Const.Input.eyeScrollBase, amountY.toDouble).toFloat + Const.Input.eyeScrollOffset
```

Both `GdxCameraHandler.handleScroll` and `OptiXCameraHandler.handleScroll` call `computeEyeW(amountY)`.

### 2d. Document scroll/ESC return value asymmetries (Sprint 11.3/11.4 M2, M3)

Add comments at:
- `GdxCameraHandler.handleScroll`: explains returning `false` to let other handlers act on the event
- `OptiXCameraHandler.handleScroll`: explains returning `true` to consume all scroll events (single handler setup)
- `GdxKeyHandler.handleKeyPress` ESC case: explains returning `false` (allows system window close)
- `OptiXKeyHandler.handleKeyPress` ESC case: explains returning `true` (consumes, prevents unintended app exit)

### 2e. Rename `DragTracker._origin` backing var (Sprint 11.1 L3)

**Problem:** `private var _origin` uses an underscore prefix, which is not idiomatic Scala.

**Solution:** Rename to `private var dragOrigin`. The public accessor `def origin` is unchanged.

---

## Section 3 — C++/CUDA Fixes

### 3a. `emission` in `getTriangleMaterial` (Sprint 11.2 M1 — functional defect)

**Problem:** `getTriangleMaterial()` in `hit_triangle.cu` has 6 output parameters but no
`emission`. Both call sites in the transparent/Fresnel path pass `emission = 0.0f` (default),
so emissive triangle meshes silently show no glow in the transparent path.

**Solution:** Add `float& emission` as the 7th output parameter to `getTriangleMaterial()`.
Extract it from the instance material (same pattern as `roughness`, `metallic`, etc.).
Update both call sites and pass `emission` to `blendFresnelColorsAndSetPayload` /
`blendFresnelColorsRGBAndSetPayload`.

### 3b. Named constants in `computeThinFilmReflectance` (Sprint 11.2 L2)

Add to `OptiXData.h` in the `RayTracingConstants` namespace:

```cpp
constexpr float THIN_FILM_COSINE_CLAMP_MIN = 0.001f;  // Prevents cos(θ) ≤ 0 in Airy formula
constexpr float THIN_FILM_AIRY_DENOM_GUARD = 1e-8f;   // Prevents divide-by-zero in Airy denominator
constexpr float CIE_Y_INTEGRAL_NORM        = 106.5f;  // CIE 1931 Y colour-matching integral (D65)
```

Replace the three unnamed literals in `helpers.cu` `computeThinFilmReflectance()`.

### 3c. Misc C++ fixes (Sprint 11.2 L1, L5, L6, B1, B2)

- **L1:** Fix indentation regression in `calculateLighting()` comment block in `helpers.cu`
  (ambient comment lost 4-space indent).
- **L5:** Replace `255.0f` and `255.99f` in `hit_cylinder.cu` diffuse fallback with
  `COLOR_BYTE_MAX` / `COLOR_SCALE_FACTOR` constants (already defined in the codebase).
- **L6:** Add a comment in `__closesthit__cylinder()` before the `film_thickness` retrieval:
  _"film_thickness retrieved for future thin-film support; cylinder shader currently uses
  diffuse-only path — thin-film branch is not implemented for cylinders."_
- **B1:** Replace `// OPTION B: ...` framing in `hit_cylinder.cu` with a plain comment
  explaining the design (single-bounce metallic reflection at depth 0, diffuse fallback
  thereafter) without implying an unresolved alternative exists.
- **B2:** Rewrite the empty `__closesthit__cylinder_shadow` comment to clarify that the
  shadow payload is set by the any-hit shader, not this shader; this function is a no-op
  by design.

---

## Section 4 — Test and DSL Fixes

### 4a. Fix vacuous assertion in `FilmRenderSuite` (Sprint 11.2 L4)

**Problem:** Line 127: `filmChannelSpread should be >= 0.0` is trivially true. The inline
comment already acknowledges this is really a logging/observability test.

**Solution:** Remove the assertion. Keep the spread computation and `logger.info` call.
The test still runs and logs the channel spread physics data; the real correctness check
(`filmImage should not equal noFilmImage`) is in the test above it.

### 4b. `Plastic`/`Matte` DSL presets delegate to `OptixMaterial` (Sprint 11.2 B3)

**Problem:** `dsl/Material.scala` hard-codes `Plastic` and `Matte` preset values inline.
If `optix.Material` factory method defaults change, DSL presets silently diverge.

**Solution:**
1. Add named `val Plastic` and `val Matte` to `optix/Material.scala` companion:
   ```scala
   val Plastic = plastic(White)  // delegates to existing factory method
   val Matte   = matte(White)    // delegates to existing factory method
   ```
2. In `dsl/Material.scala`, delegate:
   ```scala
   val Plastic = fromOptix(OptixMaterial.Plastic)
   val Matte   = fromOptix(OptixMaterial.Matte)
   ```

Single source of truth now covers all named presets in `OptixMaterial`.

---

## Files Changed Summary

| File | Change |
|------|--------|
| `menger/engines/SceneClassifier.scala` (NEW) | SceneClassifier object + SceneType enum |
| `menger/input/KeyRotation.scala` (NEW) | Shared factor map + angle() trait |
| `menger/engines/OptiXEngine.scala` | Use SceneClassifier; extract computeEffectiveMaxInstances; add sentinel comment |
| `menger/engines/AnimatedOptiXEngine.scala` | Use SceneClassifier; wrap sceneFunction in Try |
| `menger/input/InputHandler.scala` | Add computeEyeW to CameraHandler trait |
| `menger/input/GdxKeyHandler.scala` | Mixin KeyRotation; make rotatePressed/rotateAngle protected |
| `menger/input/OptiXKeyHandler.scala` | Mixin KeyRotation; make rotatePressed/rotateAngle protected |
| `menger/input/GdxCameraHandler.scala` | Use LibGDXConverters.toGdxButton; use computeEyeW; add scroll return comment |
| `menger/input/OptiXCameraHandler.scala` | Use computeEyeW; add scroll/ESC return comments |
| `menger/input/LibGDXConverters.scala` | Add toGdxButton method |
| `menger/gdx/DragTracker.scala` | Rename _origin → dragOrigin |
| `menger/optix/Material.scala` | Add val Plastic, val Matte |
| `menger/dsl/Material.scala` | Delegate Plastic/Matte to OptixMaterial |
| `optix-jni/.../FilmRenderSuite.scala` | Remove vacuous assertion, keep logger.info |
| `optix-jni/.../shaders/helpers.cu` | Named thin-film constants; fix indentation |
| `optix-jni/.../shaders/hit_triangle.cu` | Add emission to getTriangleMaterial |
| `optix-jni/.../shaders/hit_cylinder.cu` | Named color constants; thin-film comment; B1/B2 comments |
| `optix-jni/.../OptiXData.h` | Add THIN_FILM_COSINE_CLAMP_MIN, THIN_FILM_AIRY_DENOM_GUARD, CIE_Y_INTEGRAL_NORM |

---

## Non-Changes (Explicitly Deferred)

- `OptiXEngine` staying above 400 lines (~430 after extraction): further reduction would require
  splitting `createMultiObjectScene`/`rebuildScene` into a separate class — larger refactor,
  deferred to a future sprint.
- Sprint 11.5 L1 (`parseFourDRotationValues` in CliValidation): already resolved — current code
  uses `ConverterUtils.parseFourDRotation`. No action needed.
- Sprint 11.3/11.4 M4 (missing Shift+Scroll tests for OptiXCameraHandler): `OptiXCameraHandler`
  is in `coverageExcludedPackages` due to LibGDX runtime dependency. Deferred.
