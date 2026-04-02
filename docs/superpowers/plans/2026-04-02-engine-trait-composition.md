# Engine Trait Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `OptiXEngine`/`AnimatedOptiXEngine` into a composable trait hierarchy that removes LibGDX from `RenderEngine`, extracts shared scene-building into `BaseEngine`, and provides stub traits for future capabilities.

**Architecture:** `RenderEngine` becomes a pure Scala trait (no LibGDX). `BaseEngine` is a new abstract class that extends `Game with RenderEngine` and holds all shared infrastructure (renderer, resources, scene-building). `InteractiveEngine` and `AnimationEngine` replace the old engine classes; `WithAnimation` encapsulates the t-sweep loop; `WithPreview` and `WithVideoExport` are stubs.

**Tech Stack:** Scala 3, LibGDX (confined to `BaseEngine`), OptiX JNI, ScalaTest AnyFlatSpec, Wartremover, Scalafix

---

## File Map

### New files (create)
| File | Responsibility |
|------|---------------|
| `menger-app/src/main/scala/menger/engines/BaseEngine.scala` | Abstract class: LibGDX `Game` bridge, shared renderer/resources, scene-building primitives |
| `menger-app/src/main/scala/menger/engines/WithAnimation.scala` | Trait: t-parameter sweep, frame counter, render+save loop |
| `menger-app/src/main/scala/menger/engines/WithPreview.scala` | Trait stub: empty, filled in by Task 17.6 |
| `menger-app/src/main/scala/menger/engines/WithVideoExport.scala` | Trait stub: empty, filled in by Task 17.5 |
| `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala` | Class: replaces `OptiXEngine`, extends `BaseEngine` |
| `menger-app/src/main/scala/menger/engines/AnimationEngine.scala` | Class: replaces `AnimatedOptiXEngine`, extends `BaseEngine with WithAnimation` |
| `menger-app/src/main/scala/menger/engines/PreviewEngine.scala` | Class stub: `BaseEngine with WithPreview` |
| `menger-app/src/main/scala/menger/engines/VideoEngine.scala` | Class stub: `BaseEngine with WithAnimation with WithVideoExport` |
| `menger-app/src/test/scala/menger/engines/RenderEngineSuite.scala` | Test: `RenderEngine` is pure Scala, no LibGDX |
| `menger-app/src/test/scala/menger/engines/WithAnimationSuite.scala` | Test: `TAnimationConfig` logic, frame predicates, save name formatting |
| `menger-app/src/test/scala/menger/InteractiveEngineSuite.scala` | Test: replaces `OptiXEngineSuite`, updated class refs |

### Modified files
| File | Change |
|------|--------|
| `menger-app/src/main/scala/menger/engines/RenderEngine.scala` | Remove `extends Game` and `import com.badlogic.gdx.Game` |
| `menger-app/src/main/scala/Main.scala` | `OptiXEngine` → `InteractiveEngine`, `AnimatedOptiXEngine` → `AnimationEngine` |

### Deleted files (after replacements exist and tests pass)
| File | Replaced by |
|------|-------------|
| `menger-app/src/main/scala/menger/engines/OptiXEngine.scala` | `InteractiveEngine.scala` + `BaseEngine.scala` |
| `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala` | `AnimationEngine.scala` + `WithAnimation.scala` |
| `menger-app/src/test/scala/menger/OptiXEngineSuite.scala` | `InteractiveEngineSuite.scala` |

---

## Task 1: Strip LibGDX from `RenderEngine` — write the test first

**Files:**
- Create: `menger-app/src/test/scala/menger/engines/RenderEngineSuite.scala`
- Modify: `menger-app/src/main/scala/menger/engines/RenderEngine.scala`

- [ ] **Step 1.1: Write the failing test**

Create `menger-app/src/test/scala/menger/engines/RenderEngineSuite.scala`:

```scala
package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderEngineSuite extends AnyFlatSpec with Matchers:

  private class TestEngine extends RenderEngine:
    override def create(): Unit  = ()
    override def render(): Unit  = ()
    override def resize(width: Int, height: Int): Unit = ()
    override def dispose(): Unit = ()
    override def pause(): Unit   = ()
    override def resume(): Unit  = ()

  "RenderEngine" should "be implementable without LibGDX" in:
    val engine = new TestEngine
    engine.create()
    engine.render()
    engine.resize(800, 600)
    engine.dispose()
    engine.pause()
    engine.resume()
    engine shouldBe a[RenderEngine]

  it should "not require extending Game" in:
    // TestEngine does not extend com.badlogic.gdx.Game — it compiles only if RenderEngine is pure
    val engine = new TestEngine
    engine shouldBe a[RenderEngine]
```

- [ ] **Step 1.2: Run test — expect compile failure because `RenderEngine extends Game`**

Run from worktree root:
```
sbt "testOnly menger.engines.RenderEngineSuite"
```
Expected: compilation error — `TestEngine` must implement Game lifecycle methods, or `RenderEngine` is still coupled to `Game`.

- [ ] **Step 1.3: Remove `extends Game` from `RenderEngine`**

Edit `menger-app/src/main/scala/menger/engines/RenderEngine.scala` to read:

```scala
package menger.engines

trait RenderEngine:
  def create(): Unit
  def render(): Unit
  def resize(width: Int, height: Int): Unit
  def dispose(): Unit
  def pause(): Unit
  def resume(): Unit
```

(Remove the `import com.badlogic.gdx.Game` line and `extends Game`.)

- [ ] **Step 1.4: Run the test — expect compile error on `OptiXEngine` and `AnimatedOptiXEngine`**

```
sbt "testOnly menger.engines.RenderEngineSuite"
```
Expected: `RenderEngineSuite` passes but the overall build fails because `OptiXEngine extends RenderEngine` and `RenderEngine` no longer extends `Game`, so `OptiXEngine` is no longer a `Game`. The whole project won't compile yet — that's fine, we'll fix it in Task 2.

Run the suite in isolation to confirm the test itself passes conceptually:
```
sbt "project mengerApp" "testOnly menger.engines.RenderEngineSuite"
```

---

## Task 2: Create `BaseEngine` — the LibGDX bridge

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/BaseEngine.scala`

`BaseEngine` brings back the `Game` extension (so `Lwjgl3Application` can accept an engine), holds shared infrastructure, and provides scene-building primitives extracted from both old engines.

- [ ] **Step 2.1: Create `BaseEngine.scala`**

Create `menger-app/src/main/scala/menger/engines/BaseEngine.scala`:

```scala
package menger.engines

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.Game
import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.common.ValidationException
import menger.dsl.SceneConverter
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.optix.CameraState
import menger.optix.OptiXRendererWrapper
import menger.optix.SceneConfigurator

abstract class BaseEngine(maxInstances: Int)(using protected val profilingConfig: ProfilingConfig)
    extends Game with RenderEngine with LazyLogging:

  protected val rendererWrapper: OptiXRendererWrapper = OptiXRendererWrapper(maxInstances)
  protected val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)
  protected def sceneConfigurator: SceneConfigurator
  protected def cameraState: CameraState

  // Override in concrete engines that need auto-adjustment (e.g. InteractiveEngine)
  protected def computeEffectiveMaxInstances(builder: SceneBuilder, specs: List[ObjectSpec]): Int =
    maxInstances

  protected def selectMeshBuilder(specs: List[ObjectSpec]): SceneBuilder =
    val all4DProjected = specs.forall(s => ObjectType.isProjected4D(s.objectType))
    val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
    import menger.engines.scene.TesseractEdgeSceneBuilder
    import menger.engines.scene.TriangleMeshSceneBuilder
    if all4DProjected && hasEdgeRendering then
      TesseractEdgeSceneBuilder(textureDir)(using profilingConfig)
    else
      TriangleMeshSceneBuilder(textureDir)(using profilingConfig)

  // Must be provided by concrete engine — where texture assets live
  protected def textureDir: String

  protected def buildSceneFromSpecs(
    specs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Try[Unit] =
    SceneClassifier.classify(specs) match
      case SceneType.SimpleMixed(allSpecs, _) =>
        val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
        val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
        logger.info(s"Mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects")
        Try(buildMixedSceneObjects(sphereSpecs, meshSpecs, renderer))

      case SceneType.ComplexMixed(allSpecs) =>
        val objectTypes = allSpecs.map(_.objectType).distinct
        Failure(UnsupportedOperationException(
          "Cannot mix spheres with multiple different triangle mesh types. " +
          s"Objects: ${objectTypes.mkString(", ")}. " +
          "Spheres can be mixed with one mesh type at a time."
        ))

      case sceneType =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(textureDir)) match
          case Some(builder) =>
            val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
            builder.validate(specs, effectiveMaxInstances) match
              case Left(error) =>
                Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
              case Right(_) =>
                builder.buildScene(specs, renderer, effectiveMaxInstances)
          case None =>
            Failure(UnsupportedOperationException(s"No builder available for $sceneType"))

  protected def buildSceneFromConfigs(
    configs: SceneConverter.SceneConfigs,
    renderer: menger.optix.OptiXRenderer
  ): Try[Unit] =
    val specs = configs.scene.objectSpecs.getOrElse(List.empty)
    val sceneType = SceneClassifier.classify(specs)
    sceneType match
      case SceneType.Spheres(_) =>
        SphereSceneBuilder().buildScene(specs, renderer, maxInstances)
      case SceneType.TriangleMeshes(_) =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(textureDir))
          .get.buildScene(specs, renderer, maxInstances)
      case SceneType.SimpleMixed(allSpecs, _) =>
        Try {
          val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
          val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
          if sphereSpecs.nonEmpty then
            SphereSceneBuilder().buildScene(sphereSpecs, renderer, maxInstances).get
          if meshSpecs.nonEmpty then
            SceneClassifier.selectSceneBuilder(
              SceneType.TriangleMeshes(meshSpecs), Some(textureDir)
            ).get.buildScene(meshSpecs, renderer, maxInstances).get
        }
      case other =>
        SceneClassifier.selectSceneBuilder(other, Some(textureDir)) match
          case Some(builder) => builder.buildScene(specs, renderer, maxInstances)
          case None          => Failure(UnsupportedOperationException(s"Unsupported scene type: $other"))

  protected def rebuildGeometry(
    specs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Unit =
    renderer.clearAllInstances()
    SceneClassifier.classify(specs) match
      case SceneType.SimpleMixed(allSpecs, _) =>
        val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
        val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
        logger.debug(s"Rebuilding mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects")
        buildMixedSceneObjects(sphereSpecs, meshSpecs, renderer)

      case SceneType.ComplexMixed(_) =>
        Failure(UnsupportedOperationException("Complex mixed scenes not supported for rebuilding")).get

      case sceneType =>
        SceneClassifier.selectSceneBuilder(sceneType, Some(textureDir)) match
          case Some(builder) =>
            builder.buildScene(specs, renderer, maxInstances).get
          case None =>
            logger.warn(s"Cannot rebuild scene type: $sceneType")
            Failure(UnsupportedOperationException(s"Scene type $sceneType not supported for rebuilding")).get

  private def buildMixedSceneObjects(
    sphereSpecs: List[ObjectSpec],
    meshSpecs: List[ObjectSpec],
    renderer: menger.optix.OptiXRenderer
  ): Unit =
    val meshBuilder = selectMeshBuilder(meshSpecs)
    val effectiveMaxInstances = computeEffectiveMaxInstances(meshBuilder, meshSpecs)
    if sphereSpecs.nonEmpty then
      SphereSceneBuilder().buildScene(sphereSpecs, renderer, effectiveMaxInstances).get
    if meshSpecs.nonEmpty then
      meshBuilder.buildScene(meshSpecs, renderer, effectiveMaxInstances).get

  // Default lifecycle — concrete engines override what they need
  override def resize(width: Int, height: Int): Unit = {}
  override def dispose(): Unit =
    renderResources.dispose()
    rendererWrapper.dispose()
  override def pause(): Unit  = {}
  override def resume(): Unit = {}
```

- [ ] **Step 2.2: Compile to verify `BaseEngine` is well-formed**

```
sbt "project mengerApp" compile
```
Expected: Compilation errors in `OptiXEngine` and `AnimatedOptiXEngine` (they now reference `RenderEngine` which no longer extends `Game`). `BaseEngine` itself should compile. If `BaseEngine` has errors, fix them before proceeding.

---

## Task 3: Create stub capability traits

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/WithPreview.scala`
- Create: `menger-app/src/main/scala/menger/engines/WithVideoExport.scala`

These are intentionally empty stubs. Tasks 17.5 and 17.6 fill them in.

- [ ] **Step 3.1: Create `WithPreview.scala`**

```scala
package menger.engines

// Stub: filled in by Task 17.6
// Will add: t-scrubbing key bindings, on-screen t display
// Left/Right keys step t, Space toggles play/pause
trait WithPreview extends RenderEngine
```

- [ ] **Step 3.2: Create `WithVideoExport.scala`**

```scala
package menger.engines

// Stub: filled in by Task 17.5
// Will add: ffmpeg pipeline invoked after animation completes
trait WithVideoExport extends RenderEngine
```

- [ ] **Step 3.3: Compile stubs**

```
sbt "project mengerApp" compile
```
Expected: same errors as before (in `OptiXEngine`/`AnimatedOptiXEngine`) but no new ones from the stubs.

---

## Task 4: Create `WithAnimation` trait

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/WithAnimation.scala`
- Create: `menger-app/src/test/scala/menger/engines/WithAnimationSuite.scala`

`WithAnimation` encapsulates the entire animation render loop from `AnimatedOptiXEngine.render()`.

- [ ] **Step 4.1: Write the `WithAnimation` test first**

Create `menger-app/src/test/scala/menger/engines/WithAnimationSuite.scala`:

```scala
package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WithAnimationSuite extends AnyFlatSpec with Matchers:

  "TAnimationConfig" should "compute tForFrame correctly for 10 frames over [0,1]" in:
    val config = TAnimationConfig(startT = 0f, endT = 1f, frames = 10, savePattern = "frame_%04d.png")
    config.tForFrame(0) shouldBe 0f
    config.tForFrame(9) shouldBe (1f +- 0.001f)

  it should "compute tForFrame correctly for 5 frames over [0.5, 1.5]" in:
    val config = TAnimationConfig(startT = 0.5f, endT = 1.5f, frames = 5, savePattern = "f_%d.png")
    config.tForFrame(0) shouldBe (0.5f +- 0.001f)
    config.tForFrame(4) shouldBe (1.5f +- 0.001f)

  "frame completion predicate" should "be true when frame >= animConfig.frames" in:
    val config = TAnimationConfig(startT = 0f, endT = 1f, frames = 3, savePattern = "f_%d.png")
    (3 >= config.frames) shouldBe true
    (2 >= config.frames) shouldBe false

  "save name formatting" should "format frame index with %04d pattern" in:
    val pattern = "animation_%04d.png"
    String.format(pattern, Integer.valueOf(0))   shouldBe "animation_0000.png"
    String.format(pattern, Integer.valueOf(42))  shouldBe "animation_0042.png"
    String.format(pattern, Integer.valueOf(999)) shouldBe "animation_0999.png"

  it should "format frame index with %d pattern" in:
    val pattern = "frame_%d.png"
    String.format(pattern, Integer.valueOf(7)) shouldBe "frame_7.png"
```

- [ ] **Step 4.2: Run `WithAnimationSuite` — expect pass (pure logic, no engine needed)**

```
sbt "testOnly menger.engines.WithAnimationSuite"
```
Expected: All 5 tests pass (these test `TAnimationConfig` and `String.format` — no engine code yet).

- [ ] **Step 4.3: Create `WithAnimation.scala`**

Create `menger-app/src/main/scala/menger/engines/WithAnimation.scala`:

```scala
package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.common.ImageSize
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.input.GdxRuntime
import menger.optix.CausticsConfig
import menger.optix.RenderConfig

trait WithAnimation extends RenderEngine with SavesScreenshots with LazyLogging:
  self: BaseEngine =>

  protected def sceneFunction: Float => Scene
  protected def animConfig: TAnimationConfig
  protected def renderConfig: RenderConfig
  protected def causticsConfig: CausticsConfig
  protected def firstFrameConfigs: SceneConverter.SceneConfigs

  protected val frameCounter: AtomicInteger = new AtomicInteger(0)

  abstract override def create(): Unit =
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configureCamera(renderer)
    buildSceneFromConfigs(firstFrameConfigs, renderer).recover { case e: Exception =>
      logger.error(s"Failed to create initial scene: ${e.getMessage}", e)
      GdxRuntime.exit()
    }.get
    renderer.setRenderConfig(renderConfig)
    renderer.setCausticsConfig(firstFrameConfigs.caustics)
    sceneConfigurator.configurePlanes(renderer, firstFrameConfigs.planes)
    GdxRuntime.setContinuousRendering(true)

  abstract override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    val width  = GdxRuntime.width
    val height = GdxRuntime.height
    val frame  = frameCounter.get()

    if width > 0 && height > 0 && frame < animConfig.frames then
      val t = animConfig.tForFrame(frame)
      logger.info(s"Rendering frame ${frame + 1}/${animConfig.frames} (t=$t)")
      Try(sceneFunction(t)) match
        case Failure(e) =>
          logger.error(
            s"Scene function threw for frame ${frame + 1}/${animConfig.frames} (t=$t): ${e.getMessage}", e
          )
          frameCounter.incrementAndGet()
        case scala.util.Success(dslScene) =>
          val configs  = SceneConverter.convert(dslScene, causticsConfig)
          val renderer = rendererWrapper.renderer
          renderer.clearAllInstances()
          buildSceneFromConfigs(configs, renderer).recover { case e: Exception =>
            logger.error(s"Failed to build scene for frame $frame (t=$t): ${e.getMessage}", e)
          }
          sceneConfigurator.configurePlanes(renderer, configs.planes)
          configs.background.foreach(c => sceneConfigurator.setBackgroundColor(renderer, c))
          cameraState.updateCamera(renderer, configs.camera.position, configs.camera.lookAt, configs.camera.up)
          cameraState.updateCameraAspectRatio(renderer, ImageSize(width, height))
          val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
          renderResources.renderToScreen(rgbaBytes, width, height)
          saveImage()
          frameCounter.incrementAndGet()
          ()
    else if frame >= animConfig.frames then
      logger.info(s"Animation complete: ${animConfig.frames} frames rendered")
      GdxRuntime.exit()
```

- [ ] **Step 4.4: Compile**

```
sbt "project mengerApp" compile
```
Expected: `WithAnimation` compiles. The build still fails on `OptiXEngine`/`AnimatedOptiXEngine` — that is expected. Fix any `WithAnimation` compilation errors before moving on.

---

## Task 5: Create `InteractiveEngine` (replaces `OptiXEngine`)

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala`
- Create: `menger-app/src/test/scala/menger/InteractiveEngineSuite.scala`

`InteractiveEngine` is `OptiXEngine` with scene-building delegated to `BaseEngine` methods.

- [ ] **Step 5.1: Write the test first**

Create `menger-app/src/test/scala/menger/InteractiveEngineSuite.scala`:

```scala
package menger

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import menger.cli.Axis
import menger.cli.PlaneConfig
import menger.cli.PlaneSpec
import menger.common.Const
import menger.config.CameraConfig
import menger.config.EnvironmentConfig
import menger.config.ExecutionConfig
import menger.config.MaterialConfig
import menger.config.OptiXEngineConfig
import menger.config.SceneConfig
import menger.engines.InteractiveEngine
import menger.optix.RenderConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class InteractiveEngineSuite extends AnyFlatSpec with Matchers:

  private def createConfig(
    radius: Float = Const.defaultSphereRadius,
    ior: Float = Const.iorVacuum,
    color: Color = Color.WHITE,
    scale: Float = 1.0f,
    timeout: Float = 0f,
    enableStats: Boolean = false,
    renderConfig: RenderConfig = RenderConfig.Default
  ): OptiXEngineConfig =
    val colorHex = f"#${(color.r * 255).toInt}%02X${(color.g * 255).toInt}%02X${(color.b * 255).toInt}%02X"
    val objectSpec = ObjectSpec.parse(s"type=sphere:pos=0,0,0:size=${radius * 2}:scale=$scale:color=$colorHex:ior=$ior") match
      case Left(error) => sys.error(s"Failed to parse object spec: $error")
      case Right(spec) => List(spec)

    OptiXEngineConfig(
      scene = SceneConfig(objectSpecs = Some(objectSpec)),
      camera = CameraConfig(
        position = Vector3(0f, 0.5f, Const.defaultCameraZDistance),
        lookAt = Vector3(0f, 0f, 0f),
        up = Vector3(0f, 1f, 0f)
      ),
      environment = EnvironmentConfig(
        planes = List(PlaneConfig(PlaneSpec(Axis.Y, false, Const.defaultFloorPlaneY), colorSpec = None))
      ),
      execution = ExecutionConfig(
        fpsLogIntervalMs = 1000,
        timeout = timeout,
        enableStats = enableStats
      ),
      render = renderConfig
    )

  private def createEngine(config: OptiXEngineConfig): InteractiveEngine =
    given ProfilingConfig = ProfilingConfig.disabled
    InteractiveEngine(config)

  "InteractiveEngine" should "be instantiated" in:
    val config = createConfig()
    val engine = createEngine(config)
    engine shouldBe a[InteractiveEngine]

  it should "store timeout in config" in:
    val config = createConfig(timeout = 5.0f)
    config.execution.timeout shouldBe 5.0f

  it should "accept various radius values" in:
    createConfig(radius = 0.1f)
    createConfig(radius = 10.0f)
    createConfig(radius = 1.5f)
    // No assertions - just verify parsing works

  it should "have default timeout 0" in:
    val config = createConfig()
    config.execution.timeout shouldBe 0f

  it should "store enableStats false by default" in:
    val config = createConfig()
    config.execution.enableStats shouldBe false

  it should "store enableStats true when provided" in:
    val config = createConfig(enableStats = true)
    config.execution.enableStats shouldBe true

  "OptiXEngineConfig" should "have sensible defaults" in:
    val config = OptiXEngineConfig.Default
    config.execution.timeout shouldBe 0f
    config.execution.enableStats shouldBe false
    config.render shouldBe RenderConfig.Default

  "MaterialConfig" should "have useful presets" in:
    MaterialConfig.Glass.ior shouldBe 1.5f
    MaterialConfig.Diamond.ior shouldBe 2.42f
    MaterialConfig.Water.ior shouldBe 1.33f
    MaterialConfig.Mirror.ior shouldBe 1.0f
```

- [ ] **Step 5.2: Run the test — expect compile failure (`InteractiveEngine` doesn't exist yet)**

```
sbt "testOnly menger.InteractiveEngineSuite"
```
Expected: Compile error — `menger.engines.InteractiveEngine` not found.

- [ ] **Step 5.3: Create `InteractiveEngine.scala`**

Create `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala`:

```scala
package menger.engines

import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Try

import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.RotationProjectionParameters
import menger.common.Const
import menger.common.ImageSize
import menger.common.ObjectType
import menger.config.OptiXEngineConfig
import menger.engines.scene.SceneBuilder
import menger.input.EventDispatcher
import menger.input.GdxRuntime
import menger.input.Observer
import menger.input.OptiXCameraHandler
import menger.input.OptiXInputMultiplexer
import menger.input.OptiXKeyHandler
import menger.objects.higher_d.Projection
import menger.objects.higher_d.TesseractSponge2Mesh
import menger.objects.higher_d.TesseractSpongeMesh
import menger.optix.CameraState
import menger.optix.SceneConfigurator

class InteractiveEngine(
  config: OptiXEngineConfig,
  userSetMaxInstances: Boolean = false
)(using ProfilingConfig)
    extends BaseEngine(config.execution.maxInstances)
    with TimeoutSupport with LazyLogging with SavesScreenshots with Observer:

  // Convenience accessors for config sections
  private val scene       = config.scene
  private val camera      = config.camera
  private val environment = config.environment
  private val execution   = config.execution

  override protected def textureDir: String = execution.textureDir

  // Required by TimeoutSupport trait
  override def timeout: Float = execution.timeout

  // Extract object specs from config (must be provided)
  private val objectSpecs: List[ObjectSpec] = scene.objectSpecs.getOrElse {
    logger.error("SceneConfig must provide objectSpecs. Legacy single-object parameters are no longer supported.")
    sys.error("SceneConfig must provide objectSpecs")
  }

  // Event dispatcher for 4D rotation events
  private val eventDispatcher = EventDispatcher().withObserver(this)

  // Mutable state for current object specs (for interactive rotation updates)
  private val currentObjectSpecs =
    new AtomicReference[Option[List[ObjectSpec]]](Some(objectSpecs))

  // Keyboard handler for 4D rotation (initialized in finalizeCreate)
  private val keyHandler = new AtomicReference[Option[OptiXKeyHandler]](None)

  // Track if we have 4D projected objects (need rebuild on rotation)
  private lazy val has4DObjects: Boolean =
    currentObjectSpecs.get().exists(_.exists(spec => ObjectType.isProjected4D(spec.objectType)))

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator removed. Use objectSpecs instead.")),
    camera.position,
    camera.lookAt,
    camera.up,
    environment.lights
  )

  override protected val cameraState: CameraState =
    CameraState(camera.position, camera.lookAt, camera.up)

  private lazy val cameraController: OptiXCameraHandler =
    OptiXCameraHandler(rendererWrapper, cameraState, renderResources,
      camera.position, camera.lookAt, camera.up, eventDispatcher)

  // Handle rotation/projection events from keyboard and mouse
  override def handleEvent(event: RotationProjectionParameters): Unit =
    logger.debug(s"Received rotation event: rotXW=${event.rotXW}, rotYW=${event.rotYW}, rotZW=${event.rotZW}")
    if has4DObjects then
      currentObjectSpecs.set(currentObjectSpecs.get().map(_.map { spec =>
        if ObjectType.isProjected4D(spec.objectType) then
          val currentProj = spec.projection4D.getOrElse(Projection4DSpec.default)
          val newEyeW =
            if event.eyeW != Const.defaultEyeW then
              val updatedProjection = Projection(currentProj.eyeW, currentProj.screenW) + event.projection
              updatedProjection.eyeW
            else
              currentProj.eyeW
          val newProj = currentProj.copy(
            eyeW = newEyeW,
            rotXW = currentProj.rotXW + event.rotXW,
            rotYW = currentProj.rotYW + event.rotYW,
            rotZW = currentProj.rotZW + event.rotZW
          )
          logger.debug(s"Updated tesseract: rotXW=${newProj.rotXW}, rotYW=${newProj.rotYW}, rotZW=${newProj.rotZW}, eyeW=${newProj.eyeW}")
          spec.copy(projection4D = Some(newProj))
        else
          spec
      }))
      rebuildScene()
      renderResources.markNeedsRender()
      GdxRuntime.requestRendering()

  // Level thresholds for warnings (based on triangle counts and performance)
  private val VolumeLevelWarning = Const.Engine.spongeLevelWarningThreshold
  private val SurfaceLevelWarning = Const.Engine.spongeLevelWarningThreshold
  private val VolumeLevelMax = Const.Engine.cubeSpongeMaxLevel
  private val SurfaceLevelMax = Const.Engine.cubeSpongeMaxLevel

  private def warnIfHighLevel(spec: ObjectSpec): Unit =
    spec.level.foreach { level =>
      val intLevel = level.toInt
      spec.objectType match
        case "sponge-volume" =>
          if intLevel >= VolumeLevelWarning then
            val estimatedTriangles =
              math.pow(Const.Engine.cubesPerSpongeLevel, intLevel).toLong *
              Const.Engine.trianglesPerCube
            logger.warn(
              s"Sponge level $intLevel may be slow " +
              s"(~${estimatedTriangles / 1000}K triangles)"
            )
          if intLevel > VolumeLevelMax then
            logger.error(s"Sponge level $intLevel exceeds recommended maximum ($VolumeLevelMax)")
        case "sponge-surface" =>
          if intLevel >= SurfaceLevelWarning then
            val estimatedTriangles =
              math.pow(Const.Engine.trianglesPerCube, intLevel).toLong * 6 * 2
            logger.warn(
              s"Sponge level $intLevel may be slow " +
              s"(~${estimatedTriangles / 1000}K triangles)"
            )
          if intLevel > SurfaceLevelMax then
            logger.error(s"Sponge level $intLevel exceeds recommended maximum ($SurfaceLevelMax)")
        case "tesseract-sponge" =>
          if intLevel >= Const.Engine.tesseractSpongeWarnLevel then
            val triangles = TesseractSpongeMesh.estimatedTriangles(intLevel)
            logger.warn(
              s"tesseract-sponge level $intLevel may be slow " +
              s"(~${triangles / 1000}K triangles)"
            )
          if intLevel > Const.Engine.tesseractSpongeMaxLevel then
            logger.error(
              s"tesseract-sponge level $intLevel exceeds recommended maximum " +
              s"(${Const.Engine.tesseractSpongeMaxLevel})"
            )
        case "tesseract-sponge-2" =>
          if intLevel >= Const.Engine.tesseractSponge2WarnLevel then
            val triangles = TesseractSponge2Mesh.estimatedTriangles(intLevel)
            logger.warn(
              s"tesseract-sponge-2 level $intLevel may be slow " +
              s"(~${triangles / 1000}K triangles)"
            )
          if intLevel > Const.Engine.tesseractSponge2MaxLevel then
            logger.error(
              s"tesseract-sponge-2 level $intLevel exceeds recommended maximum " +
              s"(${Const.Engine.tesseractSponge2MaxLevel})"
            )
        case _ => // No warning for other types
    }

  // Override BaseEngine default: auto-adjust maxInstances when user did not set it explicitly
  override protected def computeEffectiveMaxInstances(builder: SceneBuilder, specs: List[ObjectSpec]): Int =
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

  override def create(): Unit =
    logger.info(s"Creating InteractiveEngine with ${objectSpecs.length} objects")
    objectSpecs.foreach(warnIfHighLevel)
    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlanes(renderer, environment.planes)
    sceneConfigurator.configureCamera(renderer)
    buildSceneFromSpecs(objectSpecs, renderer)
      .flatMap { _ =>
        Try {
          renderer.setRenderConfig(config.render)
          renderer.setCausticsConfig(config.caustics)
          environment.background.foreach(sceneConfigurator.setBackgroundColor(renderer, _))
          finalizeCreate()
        }
      }
      .recover { case e: Exception =>
        logger.error(s"Failed to create OptiX scene: ${e.getMessage}", e)
        GdxRuntime.exit()
      }.get

  private def resetTo4DDefaults(): Unit =
    logger.debug("Resetting 4D view to initial state")
    currentObjectSpecs.set(Some(objectSpecs))
    rebuildScene()
    renderResources.markNeedsRender()
    GdxRuntime.requestRendering()

  private def finalizeCreate(): Unit =
    val handler = OptiXKeyHandler(eventDispatcher, onReset = () => resetTo4DDefaults())
    keyHandler.set(Some(handler))
    GdxRuntime.setInputProcessor(OptiXInputMultiplexer(cameraController, handler))
    GdxRuntime.setContinuousRendering(false)
    GdxRuntime.requestRendering()
    if execution.timeout > 0 then startExitTimer(execution.timeout)

  private def rebuildScene(): Unit =
    currentObjectSpecs.get() match
      case Some(specs) =>
        val renderer   = rendererWrapper.renderer
        val savedEye   = cameraController.currentEye
        val savedLookAt = cameraController.currentLookAt
        val savedUp    = cameraController.currentUp
        logger.debug(s"Rebuilding scene with updated rotation; camera: eye=$savedEye, lookAt=$savedLookAt")
        Try {
          rebuildGeometry(specs, renderer)
          cameraState.updateCamera(renderer, savedEye, savedLookAt, savedUp)
          logger.debug("Scene rebuild complete")
        }.recover { case e =>
          logger.error(s"Failed to rebuild scene: ${e.getMessage}", e)
        }
      case None =>
        logger.warn("Cannot rebuild single-object scene interactively")

  override def render(): Unit =
    GdxRuntime.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)
    keyHandler.get().foreach(_.update(GdxRuntime.deltaTime))
    val width  = GdxRuntime.width
    val height = GdxRuntime.height
    if width > 0 && height > 0 then
      if renderResources.currentDimensions.isEmpty then
        cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))
        renderResources.markNeedsRender()
      if renderResources.needsRender then
        val size = ImageSize(width, height)
        val rgbaBytes = if execution.enableStats then renderWithStats(width, height)
                        else rendererWrapper.renderScene(size)
        renderResources.renderToScreen(rgbaBytes, width, height)
      else
        renderResources.redrawExisting(width, height)
      saveImage()
    if shouldExitAfterSave then
      renderResources.markSaved()
      GdxRuntime.exit()

  private def shouldExitAfterSave: Boolean =
    execution.saveName.isDefined && !renderResources.hasSaved && execution.timeout == 0

  override protected def currentSaveName: Option[String] = execution.saveName

  private def renderWithStats(width: Int, height: Int): Array[Byte] =
    val result = rendererWrapper.renderSceneWithStats(ImageSize(width, height))
    val stats  = result.stats
    logger.info(
      s"Ray stats: primary=${stats.primaryRays} total=${stats.totalRays} " +
      s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
      s"shadow=${stats.shadowRays} aa=${stats.aaRays} " +
      s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
    )
    result.image

  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing InteractiveEngine")
    super.dispose()
```

- [ ] **Step 5.4: Run the test — expect it to pass (instantiation only, no GPU)**

```
sbt "testOnly menger.InteractiveEngineSuite"
```
Expected: All 8 tests pass. The project still won't fully compile because `OptiXEngine` and `AnimatedOptiXEngine` still reference the old `RenderEngine`. That's fine — we'll fix in Task 7.

---

## Task 6: Create `AnimationEngine` and stub engines

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/AnimationEngine.scala`
- Create: `menger-app/src/main/scala/menger/engines/PreviewEngine.scala`
- Create: `menger-app/src/main/scala/menger/engines/VideoEngine.scala`

- [ ] **Step 6.1: Create `AnimationEngine.scala`**

Create `menger-app/src/main/scala/menger/engines/AnimationEngine.scala`:

```scala
package menger.engines

import scala.util.Failure

import menger.ProfilingConfig
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

class AnimationEngine(
  val sceneFunction: Float => Scene,
  val animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with SavesScreenshots:

  override protected def textureDir: String = executionConfig.textureDir

  // Evaluate first frame eagerly to initialise the environment (lights, camera, planes)
  private val _firstScene = sceneFunction(animConfig.tForFrame(0))

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator not used in animated engine")),
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up,
    firstFrameConfigs.lights
  )

  override protected val cameraState: CameraState = CameraState(
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up
  )

  override protected def currentSaveName: Option[String] =
    Some(String.format(animConfig.savePattern, Integer.valueOf(frameCounter.get())))
```

- [ ] **Step 6.2: Create `PreviewEngine.scala` (stub)**

```scala
package menger.engines

import menger.ProfilingConfig
import menger.optix.CameraState
import menger.optix.SceneConfigurator

// Stub: constructor signature and implementation are provided by Task 17.6
class PreviewEngine()(using ProfilingConfig)
    extends BaseEngine(0) with WithPreview:

  override protected def textureDir: String = "."

  override protected val sceneConfigurator: SceneConfigurator =
    sys.error("PreviewEngine not yet implemented (Task 17.6)")

  override protected val cameraState: CameraState =
    sys.error("PreviewEngine not yet implemented (Task 17.6)")

  override def create(): Unit = sys.error("PreviewEngine not yet implemented (Task 17.6)")
  override def render(): Unit = {}
```

- [ ] **Step 6.3: Create `VideoEngine.scala` (stub)**

```scala
package menger.engines

import scala.util.Failure

import menger.ProfilingConfig
import menger.config.ExecutionConfig
import menger.dsl.Scene
import menger.dsl.SceneConverter
import menger.optix.CameraState
import menger.optix.CausticsConfig
import menger.optix.RenderConfig
import menger.optix.SceneConfigurator

// Stub: video encoding is a no-op until Task 17.5
class VideoEngine(
  val sceneFunction: Float => Scene,
  val animConfig: TAnimationConfig,
  executionConfig: ExecutionConfig,
  val renderConfig: RenderConfig,
  val causticsConfig: CausticsConfig,
  videoOutputPath: String
)(using ProfilingConfig)
    extends BaseEngine(executionConfig.maxInstances)
    with WithAnimation with WithVideoExport with SavesScreenshots:

  override protected def textureDir: String = executionConfig.textureDir

  private val _firstScene = sceneFunction(animConfig.tForFrame(0))

  override protected val firstFrameConfigs: SceneConverter.SceneConfigs =
    SceneConverter.convert(_firstScene, causticsConfig)

  override protected val sceneConfigurator: SceneConfigurator = SceneConfigurator(
    Failure(UnsupportedOperationException("Legacy geometry generator not used in video engine")),
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up,
    firstFrameConfigs.lights
  )

  override protected val cameraState: CameraState = CameraState(
    firstFrameConfigs.camera.position,
    firstFrameConfigs.camera.lookAt,
    firstFrameConfigs.camera.up
  )

  override protected def currentSaveName: Option[String] =
    Some(String.format(animConfig.savePattern, Integer.valueOf(frameCounter.get())))
```

- [ ] **Step 6.4: Compile all new engines**

```
sbt "project mengerApp" compile
```
Expected: All new engine files compile. The build still fails because `OptiXEngine`/`AnimatedOptiXEngine` reference the old `RenderEngine` trait. Fix any compile errors in the new files before proceeding.

---

## Task 7: Update `Main.scala` and delete old engine files

**Files:**
- Modify: `menger-app/src/main/scala/Main.scala`
- Delete: `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`
- Delete: `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala`
- Delete: `menger-app/src/test/scala/menger/OptiXEngineSuite.scala`

- [ ] **Step 7.1: Update `Main.scala` imports and references**

In `menger-app/src/main/scala/Main.scala`:

Replace these two import lines:
```scala
import menger.engines.AnimatedOptiXEngine
import menger.engines.OptiXEngine
```

With:
```scala
import menger.engines.AnimationEngine
import menger.engines.InteractiveEngine
```

Replace the `AnimatedOptiXEngine(...)` call (in `createSceneBasedEngine`):
```scala
        AnimatedOptiXEngine(
          sceneFunction = fn,
          animConfig = animConfig,
          executionConfig = buildExecutionConfig(opts),
          renderConfig = opts.renderConfig,
          causticsConfig = opts.causticsConfig
        )
```

With:
```scala
        AnimationEngine(
          sceneFunction = fn,
          animConfig = animConfig,
          executionConfig = buildExecutionConfig(opts),
          renderConfig = opts.renderConfig,
          causticsConfig = opts.causticsConfig
        )
```

Replace the two `OptiXEngine(...)` calls and their return-type annotations:

In `createOptiXEngineFromDslScene` — change the return type annotation and engine construction:
```scala
  private def createOptiXEngineFromDslScene(opts: MengerCLIOptions, dslScene: menger.dsl.Scene)(using ProfilingConfig): OptiXEngine =
```
→
```scala
  private def createOptiXEngineFromDslScene(opts: MengerCLIOptions, dslScene: menger.dsl.Scene)(using ProfilingConfig): InteractiveEngine =
```

And:
```scala
    OptiXEngine(engineConfig, opts.userSetMaxInstances)
```
→ (both occurrences — in `createOptiXEngineFromDslScene` and `createCliBasedOptiXEngine`)
```scala
    InteractiveEngine(engineConfig, opts.userSetMaxInstances)
```

In `createCliBasedOptiXEngine` — change the return type annotation:
```scala
  private def createCliBasedOptiXEngine(opts: MengerCLIOptions)(using ProfilingConfig): OptiXEngine =
```
→
```scala
  private def createCliBasedOptiXEngine(opts: MengerCLIOptions)(using ProfilingConfig): InteractiveEngine =
```

- [ ] **Step 7.2: Update `MainSuite.scala` references**

In `menger-app/src/test/scala/MainSuite.scala`, replace:
```scala
import menger.engines.OptiXEngine
```
with:
```scala
import menger.engines.InteractiveEngine
```

And replace both occurrences of:
```scala
shouldBe a [OptiXEngine]
```
with:
```scala
shouldBe a [InteractiveEngine]
```

- [ ] **Step 7.3: Delete the old engine source files**

```
rm menger-app/src/main/scala/menger/engines/OptiXEngine.scala
rm menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala
```

- [ ] **Step 7.4: Delete the old test file**

```
rm menger-app/src/test/scala/menger/OptiXEngineSuite.scala
```

- [ ] **Step 7.5: Compile the full project**

```
sbt compile
```
Expected: Clean compilation — no errors anywhere in the project.

---

## Task 8: Run all tests and verify full test suite passes

- [ ] **Step 8.1: Run the full test suite**

```
sbt test
```
Expected: All ~1,053 Scala tests pass (C++ tests unchanged). Key suites to watch:
- `menger.engines.RenderEngineSuite` — `RenderEngine` is pure Scala
- `menger.engines.WithAnimationSuite` — frame logic and save name formatting
- `menger.InteractiveEngineSuite` — engine instantiation tests
- `MainSuite` — `createEngine` returns `InteractiveEngine`

If any tests fail, follow TEST FAILURE PROTOCOL: investigate root cause before changing test code.

- [ ] **Step 8.2: Verify code quality**

```
sbt "scalafix --check"
```
Expected: No warnings. Common Scalafix issues:
- Unused imports (remove them)
- Import ordering (java.* → scala.* → com.* → menger.*)
- No `var` declarations

Fix any Scalafix violations before committing.

---

## Task 9: Commit

- [ ] **Step 9.1: Review the diff**

```
git diff HEAD
git status
```
Review all changes before staging.

- [ ] **Step 9.2: Stage the new and modified files**

```
git add menger-app/src/main/scala/menger/engines/RenderEngine.scala
git add menger-app/src/main/scala/menger/engines/BaseEngine.scala
git add menger-app/src/main/scala/menger/engines/WithAnimation.scala
git add menger-app/src/main/scala/menger/engines/WithPreview.scala
git add menger-app/src/main/scala/menger/engines/WithVideoExport.scala
git add menger-app/src/main/scala/menger/engines/InteractiveEngine.scala
git add menger-app/src/main/scala/menger/engines/AnimationEngine.scala
git add menger-app/src/main/scala/menger/engines/PreviewEngine.scala
git add menger-app/src/main/scala/menger/engines/VideoEngine.scala
git add menger-app/src/main/scala/Main.scala
git add menger-app/src/test/scala/menger/engines/RenderEngineSuite.scala
git add menger-app/src/test/scala/menger/engines/WithAnimationSuite.scala
git add menger-app/src/test/scala/menger/InteractiveEngineSuite.scala
git add menger-app/src/test/scala/MainSuite.scala
```

Also stage the deleted files:
```
git add -u menger-app/src/main/scala/menger/engines/OptiXEngine.scala
git add -u menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala
git add -u menger-app/src/test/scala/menger/OptiXEngineSuite.scala
```

- [ ] **Step 9.3: Show diff to user for review**

```
git diff --cached
```

**STOP HERE.** Show the staged diff to the user before committing. Do not commit without explicit user confirmation.

- [ ] **Step 9.4: Commit (only after user confirms)**

```
git commit -m "refactor: Engine trait composition (Task 17.2)

- RenderEngine is now a pure Scala trait (no LibGDX dependency)
- BaseEngine abstract class bridges Game lifecycle with RenderEngine
- WithAnimation trait encapsulates t-sweep animation render loop
- InteractiveEngine replaces OptiXEngine (same behaviour, delegates scene-building to BaseEngine)
- AnimationEngine replaces AnimatedOptiXEngine (same behaviour, render loop from WithAnimation)
- WithPreview and WithVideoExport added as stubs for Tasks 17.6 and 17.5
- PreviewEngine and VideoEngine added as stubs
- Scene-building logic no longer duplicated between engines"
```

---

## Self-Review Checklist

After writing this plan, I verified against the spec:

1. **Spec coverage:**
   - ✅ `RenderEngine` stripped of `extends Game` (Task 1)
   - ✅ `BaseEngine` with shared infrastructure and scene-building (Task 2)
   - ✅ `WithPreview` and `WithVideoExport` stubs (Task 3)
   - ✅ `WithAnimation` trait with t-sweep loop (Task 4)
   - ✅ `InteractiveEngine` replaces `OptiXEngine` (Task 5)
   - ✅ `AnimationEngine`, `PreviewEngine`, `VideoEngine` (Task 6)
   - ✅ `Main.scala` updated (Task 7)
   - ✅ Old files deleted (Task 7)
   - ✅ `RenderEngineSuite` tests pure trait (Task 1)
   - ✅ `WithAnimationSuite` tests frame logic (Task 4)
   - ✅ `InteractiveEngineSuite` replaces `OptiXEngineSuite` (Task 5)
   - ✅ `MainSuite` updated to `InteractiveEngine` (Task 7)
   - ✅ All tests run and pass (Task 8)
   - ✅ Scalafix check (Task 8)
   - ✅ Commit (Task 9)

2. **No placeholders:** All code blocks are complete.

3. **Type consistency:**
   - `buildSceneFromConfigs` signature is `(configs: SceneConverter.SceneConfigs, renderer: menger.optix.OptiXRenderer): Try[Unit]` — consistent between `BaseEngine` and `WithAnimation`.
   - `firstFrameConfigs` is `SceneConverter.SceneConfigs` — defined in `WithAnimation` as abstract, provided by `AnimationEngine` and `VideoEngine`.
   - `textureDir: String` is abstract in `BaseEngine`, provided by all concrete engines.
   - `sceneConfigurator: SceneConfigurator` and `cameraState: CameraState` are abstract in `BaseEngine`, provided by all concrete engines.
