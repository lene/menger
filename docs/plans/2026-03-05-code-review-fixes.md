# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resolve all open issues from CODE_REVIEW.md (H6, M12, M13, M14) and all untracked open issues from Sprint 11 assessments in CODE_IMPROVEMENTS.md.

**Architecture:** Four groups: (1) engine-layer Scala refactoring — extract SceneClassifier object, computeEffectiveMaxInstances helper, and wrap sceneFunction in Try; (2) input-layer Scala refactoring — shared KeyRotation trait, MouseButton converter migration, eyeW formula extraction, DragTracker rename; (3) CUDA/C++ fixes — emission output in getTriangleMaterial, named thin-film constants, misc cylinder/helpers comments; (4) test and DSL fixes — remove vacuous assertion, delegate Plastic/Matte presets to OptixMaterial.

**Tech Stack:** Scala 3, sbt, CUDA/OptiX C++. Tests: ScalaTest, sbt test. Style: `sbt "scalafix --check"`. Build: `sbt compile`.

---

## Task 1: Create `SceneClassifier` object with tests (H6, M12)

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/SceneClassifier.scala`
- Create: `menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala`

**Context:** `classifyScene`, `isTriangleMeshType`, and `selectSceneBuilder` are private methods duplicated between `OptiXEngine.scala` (~lines 198–244) and `AnimatedOptiXEngine.scala` (~lines 159–170). The animated copy is missing `TesseractEdgeSceneBuilder` support. The `SceneType` enum is at line 38 of `OptiXEngine.scala` — it moves into `SceneClassifier.scala` so both engines share it.

**Step 1: Write the failing test**

Create `menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala`:

```scala
package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneClassifierSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig(enabled = false)

  private def spec(t: String) = ObjectSpec(objectType = t, x = 0, y = 0, z = 0, size = 1)

  "SceneClassifier.isTriangleMeshType" should "return true for cube" in:
    SceneClassifier.isTriangleMeshType("cube") shouldBe true

  it should "return true for sponge types" in:
    SceneClassifier.isTriangleMeshType("sponge-volume") shouldBe true
    SceneClassifier.isTriangleMeshType("sponge-surface") shouldBe true

  it should "return false for sphere" in:
    SceneClassifier.isTriangleMeshType("sphere") shouldBe false

  "SceneClassifier.classify" should "classify all-sphere scene as Spheres" in:
    val specs = List(spec("sphere"), spec("sphere"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.Spheres]

  it should "classify cube-sponge scene as CubeSponges" in:
    val specs = List(spec("cube-sponge"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.CubeSponges]

  it should "classify all-cube scene as TriangleMeshes" in:
    val specs = List(spec("cube"), spec("cube"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify sphere + cube as SimpleMixed" in:
    val specs = List(spec("sphere"), spec("cube"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify sphere + cube + sponge as ComplexMixed" in:
    val specs = List(spec("sphere"), spec("cube"), spec("sponge-volume"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.ComplexMixed]

  "SceneClassifier.selectSceneBuilder" should "return SphereSceneBuilder for Spheres" in:
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.Spheres(List(spec("sphere"))), textureDir = None
    )
    result shouldBe defined

  it should "return None for SimpleMixed" in:
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.SimpleMixed(List(spec("sphere")), "cube"), textureDir = None
    )
    result shouldBe None
```

**Step 2: Run test to verify it fails**

```
sbt "testOnly menger.engines.SceneClassifierSuite"
```
Expected: compilation error — `SceneClassifier` does not exist yet.

**Step 3: Create `SceneClassifier.scala`**

Create `menger-app/src/main/scala/menger/engines/SceneClassifier.scala`:

```scala
package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder

/** Scene classification and builder selection logic shared by OptiXEngine and AnimatedOptiXEngine. */
object SceneClassifier:

  def classify(specs: List[ObjectSpec]): SceneType =
    val types = specs.map(_.objectType.toLowerCase).toSet

    if types.contains("cube-sponge") then
      SceneType.CubeSponges(specs)
    else if types.forall(_ == "sphere") then
      SceneType.Spheres(specs)
    else if types.forall(isTriangleMeshType) then
      SceneType.TriangleMeshes(specs)
    else
      val hasSpheres  = types.contains("sphere")
      val meshTypes   = types.filter(isTriangleMeshType)

      if hasSpheres && meshTypes.size == 1 then
        SceneType.SimpleMixed(specs, meshTypes.head)
      else if hasSpheres && meshTypes.size > 1 then
        val all4DProjected = meshTypes.forall(ObjectType.isProjected4D)
        if all4DProjected then
          SceneType.SimpleMixed(specs, meshTypes.head)
        else
          SceneType.ComplexMixed(specs)
      else
        SceneType.ComplexMixed(specs)

  def isTriangleMeshType(objectType: String): Boolean =
    objectType == "cube" ||
    ObjectType.isSponge(objectType) ||
    ObjectType.isProjected4D(objectType)

  def selectSceneBuilder(
    sceneType: SceneType,
    textureDir: Option[String]
  )(using ProfilingConfig): Option[SceneBuilder] =
    sceneType match
      case SceneType.Spheres(_)       => Some(SphereSceneBuilder())
      case SceneType.CubeSponges(_)   => Some(CubeSpongeSceneBuilder())
      case SceneType.TriangleMeshes(specs) =>
        val all4DProjected  = specs.forall(s => ObjectType.isProjected4D(s.objectType))
        val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
        if all4DProjected && hasEdgeRendering then
          Some(TesseractEdgeSceneBuilder(textureDir))
        else
          Some(TriangleMeshSceneBuilder(textureDir))
      case SceneType.SimpleMixed(_, _) => None  // handled specially in createMultiObjectScene
      case SceneType.ComplexMixed(_)   => None
```

Also move `SceneType` enum from `OptiXEngine.scala` (lines 38–43) into `SceneClassifier.scala` — append it after the `SceneClassifier` object:

```scala
enum SceneType:
  case CubeSponges(specs: List[ObjectSpec])
  case Spheres(specs: List[ObjectSpec])
  case TriangleMeshes(specs: List[ObjectSpec])
  case SimpleMixed(specs: List[ObjectSpec], meshType: String)
  case ComplexMixed(specs: List[ObjectSpec])
```

And delete the `enum SceneType` block from `OptiXEngine.scala`.

**Step 4: Run test to verify it passes**

```
sbt "testOnly menger.engines.SceneClassifierSuite"
```
Expected: all 9 tests PASS.

**Step 5: Commit**

```
git add menger-app/src/main/scala/menger/engines/SceneClassifier.scala
git add menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala
git commit -m "feat(engines): extract SceneClassifier object with unit tests"
```

---

## Task 2: Update `OptiXEngine` to use `SceneClassifier` (H6 part 2)

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`

**Context:** `OptiXEngine` has private `classifyScene`, `isTriangleMeshType`, and `selectSceneBuilder` methods (lines ~198–244). These are replaced by `SceneClassifier.*` calls. The `SceneType` enum reference already works since it moved to the same package. The `selectSceneBuilder` call needs `execution.textureDir` passed explicitly.

**Step 1: Run full tests to establish baseline**

```
sbt test
```
Expected: all tests PASS (baseline).

**Step 2: Update `OptiXEngine.scala`**

a) Remove the three private methods: `classifyScene` (lines ~198–227), `isTriangleMeshType` (lines ~228–229), `selectSceneBuilder` (lines ~231–244).

b) In `createMultiObjectScene`, update the `classifyScene(specs)` call:
```scala
val result = SceneClassifier.classify(specs) match
```

c) In `createMultiObjectScene`, update the `selectSceneBuilder(sceneType)` call (in the `case sceneType =>` arm):
```scala
SceneClassifier.selectSceneBuilder(sceneType, execution.textureDir) match
```

d) In `rebuildScene`, update both calls similarly:
```scala
SceneClassifier.classify(specs) match
```
```scala
SceneClassifier.selectSceneBuilder(sceneType, execution.textureDir) match
```

e) `selectMeshBuilder` (lines ~246–254) references `execution.textureDir` — update it too to use `SceneClassifier.isTriangleMeshType` (already available in the object) if needed, or leave as-is since it's engine-specific mixed-scene logic.

**Step 3: Verify compile and tests pass**

```
sbt compile
sbt test
```
Expected: compile clean, all tests PASS.

**Step 4: Scalafix check**

```
sbt "scalafix --check"
```
Expected: no issues.

**Step 5: Commit**

```
git add menger-app/src/main/scala/menger/engines/OptiXEngine.scala
git commit -m "refactor(engines): OptiXEngine delegates to SceneClassifier"
```

---

## Task 3: Update `AnimatedOptiXEngine` to use `SceneClassifier` (H6 part 3, M12)

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala`

**Context:** `AnimatedOptiXEngine` has its own private `classifyScene` (lines ~159–166), `isTriangleMeshType` (lines ~168–170), and `selectSceneBuilder` (lines ~151–157). The `selectSceneBuilder` copy is missing `TesseractEdgeSceneBuilder` (M12). Replacing with `SceneClassifier.*` fixes both the duplication and the functional gap.

**Step 1: Update `AnimatedOptiXEngine.scala`**

a) Remove the three private methods: `classifyScene`, `isTriangleMeshType`, `selectSceneBuilder`.

b) Remove unused imports: `menger.engines.scene.SphereSceneBuilder`, `menger.engines.scene.TriangleMeshSceneBuilder` (now handled by SceneClassifier), `menger.engines.scene.SceneBuilder`.

   Add import: none needed — `SceneClassifier` is in the same package.

c) In `buildSceneFromConfigs`, update:
```scala
val sceneType = SceneClassifier.classify(specs)

sceneType match
  case SceneType.Spheres(_) =>
    SphereSceneBuilder().buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
  case SceneType.TriangleMeshes(_) =>
    SceneClassifier.selectSceneBuilder(sceneType, executionConfig.textureDir)
      .get.buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
  case SceneType.SimpleMixed(allSpecs, _) =>
    Try {
      val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
      val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
      if sphereSpecs.nonEmpty then
        SphereSceneBuilder()
          .buildScene(sphereSpecs, rendererWrapper.renderer, executionConfig.maxInstances).get
      if meshSpecs.nonEmpty then
        SceneClassifier.selectSceneBuilder(
          SceneType.TriangleMeshes(meshSpecs), executionConfig.textureDir
        ).get.buildScene(meshSpecs, rendererWrapper.renderer, executionConfig.maxInstances).get
    }
  case other =>
    SceneClassifier.selectSceneBuilder(other, executionConfig.textureDir) match
      case Some(builder) =>
        builder.buildScene(specs, rendererWrapper.renderer, executionConfig.maxInstances)
      case None =>
        Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))
```

Add missing import `menger.engines.scene.SphereSceneBuilder` back (still needed for the SimpleMixed sphere path).

**Step 2: Verify compile and tests pass**

```
sbt compile
sbt test
```
Expected: compile clean, all tests PASS.

**Step 3: Scalafix check**

```
sbt "scalafix --check"
```

**Step 4: Commit**

```
git add menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala
git commit -m "refactor(engines): AnimatedOptiXEngine delegates to SceneClassifier (fixes M12 TesseractEdge gap)"
```

---

## Task 4: Extract `computeEffectiveMaxInstances` in `OptiXEngine` (M14)

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`

**Context:** Three near-identical blocks compute effective max instances in `OptiXEngine` (~lines 279–288, 312–321, 385–394). Extract to a private helper.

**Step 1: Add the helper method to `OptiXEngine`**

Add after the `rebuildScene` method (or near the other private helpers):

```scala
private def computeEffectiveMaxInstances(builder: SceneBuilder, specs: List[ObjectSpec]): Int =
  if userSetMaxInstances then
    execution.maxInstances
  else
    val required = builder.calculateRequiredInstances(specs)
    if required > 0 && required > execution.maxInstances then
      val adjusted = Math.min(required * 2, menger.common.Const.maxInstancesLimit)
      logger.info(
        s"Auto-adjusting max instances: ${execution.maxInstances} → $adjusted (scene requires $required)"
      )
      adjusted
    else
      execution.maxInstances
```

**Step 2: Replace the three inline blocks**

In `createMultiObjectScene` mesh path (first occurrence, ~line 279):
```scala
val effectiveMaxInstances = computeEffectiveMaxInstances(meshBuilder, meshSpecs)
```

In `createMultiObjectScene` general path (second occurrence, ~line 312):
```scala
val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
```

In `rebuildScene` mesh path (third occurrence, ~line 385):
```scala
val effectiveMaxInstances = computeEffectiveMaxInstances(meshBuilder, meshSpecs)
```

Note: the `rebuildScene` general path (line ~411) calls `builder.buildScene(specs, renderer, execution.maxInstances)` — leave this as-is; it intentionally skips the auto-adjust logic on rebuild.

**Step 3: Verify compile and tests pass**

```
sbt compile
sbt test
```

**Step 4: Commit**

```
git add menger-app/src/main/scala/menger/engines/OptiXEngine.scala
git commit -m "refactor(engines): extract computeEffectiveMaxInstances helper in OptiXEngine"
```

---

## Task 5: Wrap `sceneFunction(t)` in `Try` (M13)

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala`

**Context:** In `render()` at line ~91, `sceneFunction(t)` is called bare. A user scene function that throws for a particular t value crashes the whole application instead of skipping the frame and continuing.

**Step 1: Update `render()` in `AnimatedOptiXEngine.scala`**

Find the block starting at `val dslScene = sceneFunction(t)` and wrap:

```scala
Try(sceneFunction(t)) match
  case scala.util.Failure(e) =>
    logger.error(
      s"Scene function threw for frame ${frame + 1}/${animConfig.frames} (t=$t): ${e.getMessage}", e
    )
    frameCounter.incrementAndGet()
  case scala.util.Success(dslScene) =>
    val configs = SceneConverter.convert(dslScene, causticsConfig)

    // Rebuild geometry
    val renderer = rendererWrapper.renderer
    renderer.clearAllInstances()
    buildSceneFromConfigs(configs).recover { case e: Exception =>
      logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
    }

    // Reconfigure planes per frame (supports animated plane changes)
    sceneConfigurator.configurePlanes(renderer, configs.planes)

    // Apply per-scene background color if set
    configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))

    // Update camera from this frame's scene
    cameraState.updateCamera(
      renderer, configs.camera.position, configs.camera.lookAt, configs.camera.up
    )
    cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))

    // Render
    val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
    renderResources.renderToScreen(rgbaBytes, width, height)

    // Save frame
    saveImage()

    frameCounter.incrementAndGet()
    ()
```

`Try` is already imported in the file (`import scala.util.Try`). Add `import scala.util.Failure` if not already present.

**Step 2: Verify compile and tests pass**

```
sbt compile
sbt test
```

**Step 3: Scalafix check**

```
sbt "scalafix --check"
```

**Step 4: Commit**

```
git add menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala
git commit -m "fix(engines): wrap sceneFunction(t) in Try to skip failing frames gracefully (M13)"
```

---

## Task 6: Add `event.eyeW` sentinel comment (Sprint 11.3/11.4 L1)

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`

**Step 1: Add comment at the sentinel check**

Find the line `if event.eyeW != Const.defaultEyeW then` (line ~93) and add the comment above it:

```scala
// Sentinel: Const.defaultEyeW signals "no eyeW change in this event" (rotation-only).
// A user cannot intentionally reset eyeW to exactly defaultEyeW via an event — if that
// becomes needed, replace the sentinel with Option[Float] in RotationProjectionParameters.
if event.eyeW != Const.defaultEyeW then
```

**Step 2: Verify compile**

```
sbt compile
```

**Step 3: Commit**

```
git add menger-app/src/main/scala/menger/engines/OptiXEngine.scala
git commit -m "docs(engines): document eyeW sentinel pattern limitation in OptiXEngine"
```

---

## Task 7: Create `KeyRotation` trait (Sprint 11.1 M1)

**Files:**
- Create: `menger-app/src/main/scala/menger/input/KeyRotation.scala`
- Modify: `menger-app/src/main/scala/menger/input/GdxKeyHandler.scala`
- Modify: `menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala`

**Context:** Both `GdxKeyHandler` and `OptiXKeyHandler` have identical `factor: Map[Key, Int]` (lines 83–87 / 66–70) and `angle(delta, keys): Float` (lines 89–90 / 72–73). Extract to a shared trait.

**Step 1: Create `KeyRotation.scala`**

```scala
package menger.input

import menger.common.Key
import menger.gdx.KeyPressTracker

/** Shared key-rotation angle calculation used by GdxKeyHandler and OptiXKeyHandler. */
trait KeyRotation:
  protected def rotatePressed: KeyPressTracker
  protected def rotateAngle: Float

  protected val factor: Map[Key, Int] = Map(
    Key.Right   -> -1, Key.Left  -> 1,
    Key.Up      ->  1, Key.Down  -> -1,
    Key.PageUp  ->  1, Key.PageDown -> -1
  )

  protected def angle(delta: Float, keys: Seq[Key]): Float =
    delta * rotateAngle * keys.find(rotatePressed.isPressed).map(factor(_)).getOrElse(0)
```

**Step 2: Update `GdxKeyHandler.scala`**

- Change `class GdxKeyHandler(...) extends KeyHandler:` to `class GdxKeyHandler(...) extends KeyHandler with KeyRotation:`
- Change `private val rotateAngle` to `protected val rotateAngle`
- Change `private val rotatePressed` to `protected val rotatePressed`
- Delete the local `private val factor = Map(...)` block (lines 83–87)
- Delete the local `private def angle(...)` method (lines 89–90)

**Step 3: Update `OptiXKeyHandler.scala`**

- Change `class OptiXKeyHandler(...) extends KeyHandler with LazyLogging:` to `... extends KeyHandler with KeyRotation with LazyLogging:`
- Change `private val rotateAngle` to `protected val rotateAngle`
- Change `private val rotatePressed` to `protected val rotatePressed`
- Delete the local `private val factor = Map(...)` block (lines 66–70)
- Delete the local `private def angle(...)` method (lines 72–73)

**Step 4: Verify compile and tests pass**

```
sbt compile
sbt "testOnly menger.input.*"
```
Expected: compile clean, existing key handler tests PASS.

**Step 5: Scalafix check**

```
sbt "scalafix --check"
```

**Step 6: Commit**

```
git add menger-app/src/main/scala/menger/input/KeyRotation.scala
git add menger-app/src/main/scala/menger/input/GdxKeyHandler.scala
git add menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala
git commit -m "refactor(input): extract KeyRotation trait to eliminate factor/angle duplication"
```

---

## Task 8: Move `MouseButton.toGdxButton` to `LibGDXConverters` (Sprint 11.1 M2)

**Files:**
- Modify: `menger-app/src/main/scala/menger/input/LibGDXConverters.scala`
- Modify: `menger-app/src/main/scala/menger/input/GdxCameraHandler.scala`
- Modify: `menger-app/src/test/scala/menger/input/LibGDXConvertersSuite.scala` (if exists, else check)

**Context:** `GdxCameraHandler.scala` defines `extension (button: MouseButton) def toGdxButton: Int` (lines 93–98). Converter logic belongs in `LibGDXConverters`.

**Step 1: Add `toGdxButton` to `LibGDXConverters.scala`**

Append after the existing `convertButton` method:

```scala
/** Convert domain MouseButton to LibGDX button code */
def toGdxButton(button: MouseButton): Int = button match
  case MouseButton.Left        => Buttons.LEFT
  case MouseButton.Right       => Buttons.RIGHT
  case MouseButton.Middle      => Buttons.MIDDLE
  case MouseButton.Unknown(c)  => c
```

**Step 2: Add test to `LibGDXConvertersSuite.scala`**

Read the existing suite first, then append:

```scala
"LibGDXConverters.toGdxButton" should "convert Left to Buttons.LEFT" in:
  LibGDXConverters.toGdxButton(MouseButton.Left) shouldBe com.badlogic.gdx.Input.Buttons.LEFT

it should "convert Right to Buttons.RIGHT" in:
  LibGDXConverters.toGdxButton(MouseButton.Right) shouldBe com.badlogic.gdx.Input.Buttons.RIGHT

it should "pass through Unknown button code" in:
  LibGDXConverters.toGdxButton(MouseButton.Unknown(99)) shouldBe 99
```

**Step 3: Update `GdxCameraHandler.scala`**

- Delete the extension method block (lines 93–98)
- Replace `button.toGdxButton` at line 51 with `LibGDXConverters.toGdxButton(button)`
- Replace `button.toGdxButton` at line 54 with `LibGDXConverters.toGdxButton(button)`

**Step 4: Verify compile and tests pass**

```
sbt compile
sbt "testOnly menger.input.LibGDXConvertersSuite"
```
Expected: all tests PASS.

**Step 5: Scalafix check**

```
sbt "scalafix --check"
```

**Step 6: Commit**

```
git add menger-app/src/main/scala/menger/input/LibGDXConverters.scala
git add menger-app/src/main/scala/menger/input/GdxCameraHandler.scala
git add menger-app/src/test/scala/menger/input/LibGDXConvertersSuite.scala
git commit -m "refactor(input): move MouseButton.toGdxButton to LibGDXConverters"
```

---

## Task 9: Extract `computeEyeW` to `CameraHandler` trait (Sprint 11.3/11.4 M1)

**Files:**
- Modify: `menger-app/src/main/scala/menger/input/InputHandler.scala`
- Modify: `menger-app/src/main/scala/menger/input/GdxCameraHandler.scala`
- Modify: `menger-app/src/main/scala/menger/input/OptiXCameraHandler.scala`

**Context:** Both camera handlers compute eye-W identically:
`Math.pow(Const.Input.eyeScrollBase, amountY.toDouble).toFloat + Const.Input.eyeScrollOffset`
(`GdxCameraHandler` line 64, `OptiXCameraHandler` line 76).

**Step 1: Add `computeEyeW` to `CameraHandler` trait in `InputHandler.scala`**

Inside the `CameraHandler` trait body, add:

```scala
import menger.common.Const

protected def computeEyeW(amountY: Float): Float =
  Math.pow(Const.Input.eyeScrollBase, amountY.toDouble).toFloat + Const.Input.eyeScrollOffset
```

If `Const` is not already imported in `InputHandler.scala`, add the import at the top.

**Step 2: Update `GdxCameraHandler.scala`**

In `handleScroll`, replace the inline formula with:
```scala
val eyeW = computeEyeW(amountY)
```

**Step 3: Update `OptiXCameraHandler.scala`**

In `handleScroll`, replace the inline formula with:
```scala
val eyeW = computeEyeW(amountY)
```

**Step 4: Verify compile and tests pass**

```
sbt compile
sbt "testOnly menger.input.*"
```

**Step 5: Commit**

```
git add menger-app/src/main/scala/menger/input/InputHandler.scala
git add menger-app/src/main/scala/menger/input/GdxCameraHandler.scala
git add menger-app/src/main/scala/menger/input/OptiXCameraHandler.scala
git commit -m "refactor(input): extract computeEyeW formula to CameraHandler trait"
```

---

## Task 10: Document scroll/ESC return value asymmetries + rename `DragTracker._origin` (Sprint 11.3/11.4 M2, M3; Sprint 11.1 L3)

**Files:**
- Modify: `menger-app/src/main/scala/menger/input/GdxCameraHandler.scala`
- Modify: `menger-app/src/main/scala/menger/input/OptiXCameraHandler.scala`
- Modify: `menger-app/src/main/scala/menger/input/GdxKeyHandler.scala`
- Modify: `menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala`
- Modify: `menger-app/src/main/scala/menger/gdx/DragTracker.scala`

**Step 1: Add scroll comments to `GdxCameraHandler.scala`**

In `handleScroll`, before the `if isShiftPressed then` line:
```scala
// Returns false: lets other handlers (e.g. camera zoom) still act on this scroll event.
```
And annotate the non-Shift branch `baseController.scrolled(amountX, amountY)` with:
```scala
// The LibGDX CameraInputController return value propagates naturally here.
```

**Step 2: Add scroll comment to `OptiXCameraHandler.scala`**

In `handleScroll`, change the final `true` return to be clearly documented:
```scala
true // Consume all scroll events: single-handler setup, no other handler should act on scroll.
```

**Step 3: Add ESC comments to key handlers**

In `GdxKeyHandler.handleKeyPress`, ESC case:
```scala
case Key.Escape =>
  resetCamera()
  false  // Don't consume: allows the OS/windowing system to handle window-close on Escape.
```

In `OptiXKeyHandler.handleKeyPress`, ESC case:
```scala
case Key.Escape =>
  onReset()
  true  // Consume: prevents unintended application exit; ESC only resets the 4D view.
```

**Step 4: Rename `_origin` in `DragTracker.scala`**

Change `private var _origin: ScreenCoords` to `private var dragOrigin: ScreenCoords`.
Change `def start(pos: ScreenCoords): Unit = _origin = pos` to `... = dragOrigin = pos`.
Change `def origin: ScreenCoords = _origin` to `def origin: ScreenCoords = dragOrigin`.

**Step 5: Verify compile and tests pass**

```
sbt compile
sbt "testOnly menger.input.*"
```

**Step 6: Commit**

```
git add menger-app/src/main/scala/menger/input/GdxCameraHandler.scala
git add menger-app/src/main/scala/menger/input/OptiXCameraHandler.scala
git add menger-app/src/main/scala/menger/input/GdxKeyHandler.scala
git add menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala
git add menger-app/src/main/scala/menger/gdx/DragTracker.scala
git commit -m "docs/refactor(input): document return value asymmetries, rename DragTracker._origin"
```

---

## Task 11: Add `emission` output to `getTriangleMaterial` in `hit_triangle.cu` (Sprint 11.2 M1)

**Files:**
- Modify: `optix-jni/src/main/native/shaders/hit_triangle.cu`

**Context:** `getTriangleMaterial()` is missing an `emission` output parameter. Both call sites in the Fresnel/thin-film path pass `emission = 0.0f` by default, so emissive transparent triangle meshes show no glow.

**Step 1: Read `hit_triangle.cu` to find exact line numbers**

Read the file and locate:
1. The `getTriangleMaterial` function signature
2. Its body where it sets the output parameters
3. Both call sites (Fresnel path and thin-film path)
4. The call to `blendFresnelColorsAndSetPayload` and/or `blendFresnelColorsRGBAndSetPayload`

**Step 2: Update `getTriangleMaterial` signature**

Add `float& out_emission` as the 7th parameter after `out_film_thickness`:

```cpp
static __device__ void getTriangleMaterial(
    const InstanceMaterial& mat,
    float2 uv, bool hasUVs,
    float4& out_color,
    float& out_ior,
    float& out_roughness,
    float& out_metallic,
    float& out_specular,
    float& out_film_thickness,
    float& out_emission          // NEW
)
```

**Step 3: Set `out_emission` in the function body**

Inside `getTriangleMaterial`, after the other output assignments, add:
```cpp
out_emission = mat.emission;
```

**Step 4: Update call sites**

At each call site:
1. Add `float mesh_emission;` to the local variable declarations (alongside `float mesh_ior, roughness, metallic, specular, film_thickness;`)
2. Pass `mesh_emission` as the last argument to `getTriangleMaterial(..., mesh_emission)`
3. Pass `mesh_emission` to `blendFresnelColorsAndSetPayload` / `blendFresnelColorsRGBAndSetPayload` (check their signatures — they already take an `emission` parameter from the sphere shader)

**Step 5: Build the native library**

```
sbt "optixJni/nativeCompile"
```
Expected: build succeeds, no C++ errors.

**Step 6: Run OptiX tests**

```
sbt "optixJni/test"
```
Expected: all GPU tests PASS. Emissive triangle meshes will now glow in transparent path.

**Step 7: Commit**

```
git add optix-jni/src/main/native/shaders/hit_triangle.cu
git commit -m "fix(shaders): add emission output to getTriangleMaterial (transparent triangle mesh emission now works)"
```

---

## Task 12: Named thin-film constants + misc C++ fixes (Sprint 11.2 L1, L2, L5, L6, B1, B2)

**Files:**
- Modify: `optix-jni/src/main/native/include/OptiXData.h`
- Modify: `optix-jni/src/main/native/shaders/helpers.cu`
- Modify: `optix-jni/src/main/native/shaders/hit_cylinder.cu`

**Step 1: Add named constants to `OptiXData.h`**

In the `RayTracingConstants` namespace (or wherever `HIT_POINT_RAY_TMIN` etc. live), add:

```cpp
// Thin-film interference constants (helpers.cu: computeThinFilmReflectance)
constexpr float THIN_FILM_COSINE_CLAMP_MIN = 0.001f;  // Prevents cos(θ) ≤ 0 in Airy formula
constexpr float THIN_FILM_AIRY_DENOM_GUARD = 1e-8f;   // Prevents divide-by-zero in Airy denominator
constexpr float CIE_Y_INTEGRAL_NORM        = 106.5f;  // CIE 1931 Y colour-matching integral (D65)
```

**Step 2: Update `helpers.cu` — replace magic numbers**

In `computeThinFilmReflectance`:
- Replace `0.001f` with `RayTracingConstants::THIN_FILM_COSINE_CLAMP_MIN`
- Replace `1e-8f` with `RayTracingConstants::THIN_FILM_AIRY_DENOM_GUARD`
- Replace `106.5f` with `RayTracingConstants::CIE_Y_INTEGRAL_NORM`

**Step 3: Fix indentation regression in `helpers.cu`**

In `calculateLighting()`, find the `// Add ambient lighting` comment block that lost its 4-space indent (it starts at column 0). Re-indent it to 4 spaces to match surrounding code.

**Step 4: Update `hit_cylinder.cu` — replace `255.0f`/`255.99f`**

Find the cylinder diffuse fallback path in `__closesthit__cylinder()`. Replace:
- `255.99f` with `RayTracingConstants::COLOR_SCALE_FACTOR` (verify this constant name against the codebase)
- `255.0f` with `RayTracingConstants::COLOR_BYTE_MAX` (verify constant name)

If the constants use a different namespace or name, grep the codebase:
```
grep -r "255.99\|COLOR_SCALE\|COLOR_BYTE" optix-jni/src/main/native/
```

**Step 5: Fix cylinder thin-film comment (L6)**

Before the `film_thickness` retrieval line in `__closesthit__cylinder()`, add:
```cpp
// Note: film_thickness is retrieved here for API consistency, but cylinder shaders use
// a diffuse-only lighting model that does not implement thin-film interference.
// Thin-film support for cylinders is deferred to a future sprint.
```

**Step 6: Fix `OPTION B` comment (B1)**

Find the `// OPTION B:` comment in `hit_cylinder.cu` and replace it with:
```cpp
// Single-bounce metallic reflection at ray depth 0, diffuse fallback for depth > 0.
// This avoids deep recursion while still supporting metallic appearance on cylinder edges.
```

**Step 7: Fix `__closesthit__cylinder_shadow` comment (B2)**

Find the empty or misleading comment in `__closesthit__cylinder_shadow` and replace with:
```cpp
// Shadow closest-hit: no-op by design.
// The shadow payload (0.0 = shadowed) is set by the any-hit shader (__anyhit__cylinder_shadow)
// before this closest-hit runs. This function exists only to satisfy the OptiX program group
// requirements; it does not need to modify payload state.
```

**Step 8: Build and test**

```
sbt "optixJni/nativeCompile"
sbt "optixJni/test"
```
Expected: clean build, all GPU tests PASS.

**Step 9: Commit**

```
git add optix-jni/src/main/native/include/OptiXData.h
git add optix-jni/src/main/native/shaders/helpers.cu
git add optix-jni/src/main/native/shaders/hit_cylinder.cu
git commit -m "fix(shaders): named thin-film constants, fix indentation, clarify cylinder comments"
```

---

## Task 13: Fix vacuous assertion in `FilmRenderSuite` (Sprint 11.2 L4)

**Files:**
- Modify: `optix-jni/src/test/scala/menger/optix/FilmRenderSuite.scala`

**Context:** Line 127: `filmChannelSpread should be >= 0.0` is trivially true. The inline comment already says this is a logging/observability test. The `logger.info` on line 124 is the real value here.

**Step 1: Remove the vacuous assertion**

Find the test `"produce more color variation than same sphere without film"`. Remove line 127:
```scala
filmChannelSpread should be >= 0.0  // trivially true — really a logging/observability test
```

The `logger.info` line 124 stays. The test still runs and logs the physics data. The real assertion (images differ) is in the preceding test.

**Step 2: Run the test to verify it still passes**

```
sbt "optixJni/testOnly menger.optix.FilmRenderSuite"
```
Expected: 8 tests PASS (the test that had the vacuous assertion now has only the logger.info and passes).

**Step 3: Commit**

```
git add optix-jni/src/test/scala/menger/optix/FilmRenderSuite.scala
git commit -m "test(film): remove vacuous filmChannelSpread >= 0.0 assertion, keep observability logging"
```

---

## Task 14: Delegate `Plastic`/`Matte` DSL presets to `OptixMaterial` (Sprint 11.2 B3)

**Files:**
- Modify: `optix-jni/src/main/scala/menger/optix/Material.scala`
- Modify: `menger-app/src/main/scala/menger/dsl/Material.scala`

**Context:** `dsl/Material.scala` lines 52–53 hard-code `Plastic` and `Matte` inline. `optix/Material.scala` has `plastic(White)` and `matte(White)` factory methods but no named `val` constants. Adding `val Plastic` and `val Matte` to OptixMaterial and delegating from DSL completes the single-source-of-truth for all material presets.

**Step 1: Add `Plastic` and `Matte` to `optix/Material.scala`**

In the `Material` companion object, after the `Parchment` val, add:

```scala
// Opaque presets
val Plastic = plastic(White)   // ior=1.5, roughness=0.3, metallic=0, specular=0.5
val Matte   = matte(White)     // ior=1.0, roughness=1.0, metallic=0, specular=0.0
```

**Step 2: Update `dsl/Material.scala`**

Replace lines 52–53:
```scala
val Plastic = Material(Color.White, ior = 1.5f, roughness = 0.3f, metallic = 0f, specular = 0.5f)
val Matte   = Material(Color.White, ior = 1.0f, roughness = 1.0f, metallic = 0f, specular = 0f)
```
with:
```scala
// Opaque presets — delegate to OptixMaterial to maintain single source of truth
val Plastic = fromOptix(OptixMaterial.Plastic)
val Matte   = fromOptix(OptixMaterial.Matte)
```

**Step 3: Verify tests pass**

```
sbt "testOnly menger.dsl.*"
```
Expected: all DSL tests PASS (including any MaterialSuite tests that check Plastic/Matte values).

**Step 4: Verify scalafix**

```
sbt "scalafix --check"
```

**Step 5: Commit**

```
git add optix-jni/src/main/scala/menger/optix/Material.scala
git add menger-app/src/main/scala/menger/dsl/Material.scala
git commit -m "refactor(dsl): Plastic/Matte presets delegate to OptixMaterial (single source of truth)"
```

---

## Task 15: Update `CODE_REVIEW.md` and run full test suite

**Files:**
- Modify: `CODE_REVIEW.md`

**Step 1: Run full test suite**

```
sbt test
```
Expected: all tests PASS, no regressions.

**Step 2: Update `CODE_REVIEW.md`**

Mark H6, M12, M13 as resolved. Update M14 status to reflect remaining line count (~430 lines, still above 400 but reduced from 488). Add resolutions for all Sprint 11 issues now tracked.

**Step 3: Update summary table in `CODE_REVIEW.md`**

Adjust the open/completed counts in the summary.

**Step 4: Commit**

```
git add CODE_REVIEW.md
git commit -m "docs: mark H6/M12/M13 resolved, update Sprint 11 open issue status in CODE_REVIEW.md"
```

---

## Execution Order Summary

| Task | Issues | Risk |
|------|--------|------|
| 1 — SceneClassifier + tests | H6 setup | Low |
| 2 — OptiXEngine uses SceneClassifier | H6 pt2 | Low |
| 3 — AnimatedOptiXEngine uses SceneClassifier | H6 pt3, M12 | Low |
| 4 — computeEffectiveMaxInstances | M14 | Low |
| 5 — sceneFunction Try wrapping | M13 | Low |
| 6 — eyeW sentinel comment | L1 | Trivial |
| 7 — KeyRotation trait | Sprint 11.1 M1 | Low |
| 8 — toGdxButton migration | Sprint 11.1 M2 | Low |
| 9 — computeEyeW extraction | Sprint 11.3 M1 | Low |
| 10 — Comments + DragTracker rename | M2, M3, L3 | Trivial |
| 11 — emission in getTriangleMaterial | Sprint 11.2 M1 | Medium (CUDA) |
| 12 — Named constants + misc C++ | Sprint 11.2 L1-L6, B1-B2 | Low (CUDA) |
| 13 — Remove vacuous assertion | Sprint 11.2 L4 | Trivial |
| 14 — Plastic/Matte delegation | Sprint 11.2 B3 | Low |
| 15 — Full test suite + CODE_REVIEW.md update | — | Trivial |
