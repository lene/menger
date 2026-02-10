# Sprint 10: Scala DSL for Scene Description

**Sprint:** 10 - Scene Description Language (Scala DSL)
**Status:** Not Started
**Estimate:** 18-23 hours
**Branch:** `feature/sprint-10`
**Dependencies:** None (builds on existing infrastructure)

---

## Goal

Create a Scala DSL that allows concise, type-safe scene definitions that compile with the project and can be loaded via `--scene <classname>`.

## Success Criteria

- [ ] `--scene scenes.MyScene` loads and renders a scene defined in Scala DSL
- [ ] Case-class style syntax: `Scene(objects = List(Sphere(Glass)))`
- [ ] Concise syntax: `Sphere(Glass)` creates a glass sphere at origin
- [ ] Arrow syntax for camera: `Camera((0, 0.5, 3) -> (0, 0, 0))`
- [ ] Scene files can import definitions from other files (standard Scala imports)
- [ ] All current object types, materials, and lights are expressible
- [ ] Material factory shorthands: `Matte("#FF0000")`, `Plastic("#00FF00")`
- [ ] Texture support: `Cube((0,0,0), texture = "brick.png", size = 1.5)`
- [ ] Render settings as flat Scene fields: `shadows = true`, `antialiasing = true`
- [ ] Caustics configuration: `caustics = Caustics(photons = 100000)`
- [ ] Comprehensive tests for DSL parsing and scene generation
- [ ] Example scene files demonstrating DSL capabilities

---

## Scope

### In Scope (First Version)
- Objects: Sphere, Cube, Sponge (VolumeFilling/SurfaceUnfolding/CubeSponge)
- Materials: presets (Glass, Chrome, Gold, etc.) + custom definitions + factory shorthands
- Lights: Directional, Point (up to 8)
- Camera: position, lookAt, up; arrow syntax `(pos) -> (lookAt)`
- Plane: axis, position, color (solid) or colors (checkered)
- Textures: per-object texture assignment
- Render settings: shadows, antialiasing (flat on Scene)
- Caustics: photons, iterations, radius, alpha

### Deferred (Backlog)
- Animation keyframes (Sprint 12)
- Runtime evaluation (compile-time only for now)
- 4D DSL enhancements (tesseract support - after Sprint 11)
- Window/output settings (width, height, saveName, headless — CLI-only for now)

---

## DSL Design

### Example Scene

```scala
package scenes
import menger.dsl.*

object MyScene extends SceneDefinition:
  val scene = Scene(
    camera = Camera((0, 0.5, 3) -> (0, 0, 0)),

    lights = List(
      Directional((-1, 1, -1), intensity = 2.0),
      Point((0, 5, 0), intensity = 1.5, color = "#FFFFCC")
    ),

    objects = List(
      Sphere(Glass),                                    // material only, at origin
      Sphere((2, 0, 0), Chrome, size = 0.8),           // position, material, size
      Cube((-2, 0, 0), color = "#FF0000", size = 1.5), // color instead of material
      Cube((0, 0, -2), texture = "brick.png"),         // textured
      Sponge(VolumeFilling, level = 3, Glass, size = 2.0),
      Sponge((4, 0, 0), SurfaceUnfolding, level = 2, color = "#00FF00")
    ),

    plane = Plane(Y at -2, color = "#808080"),

    shadows = true,
    antialiasing = true,
    aaMaxDepth = 3,

    caustics = Caustics(photons = 100000, iterations = 10, radius = 0.1)
  )
```

### Positional Parameter Conventions

| Type | Positional args | Named args |
|---|---|---|
| `Camera` | `(pos) -> (lookAt)` or `(pos), (lookAt)` | `up` |
| `Sphere` | position?, material? | `size`, `color`, `texture` |
| `Cube` | position?, material? | `size`, `color`, `texture` |
| `Sponge` | position?, spongeType, material? | `level`, `size`, `color`, `texture` |
| `Plane` | axis spec (e.g. `Y at -2`) | `color`, `colors` |
| `Directional` | direction | `intensity`, `color` |
| `Point` | position | `intensity`, `color` |
| `Caustics` | *(none)* | `photons`, `iterations`, `radius`, `alpha` |

### Material Availability

**Presets:** `Glass`, `Water`, `Diamond`, `Chrome`, `Gold`, `Copper`, `Film`, `Parchment`, `Plastic`, `Matte`

**Factories:** `Matte("#color")`, `Plastic("#color")`, `Metal("#color")`, `Glass("#color")`

**Custom:** `Material(color = "#FF0000", roughness = 0.3, metallic = 0.8)`

**Modifications:** `Glass.copy(ior = 1.7)`

### Reusable Definitions

```scala
// scenes/common/Materials.scala
package scenes.common
import menger.dsl.*

object Materials:
  val MyCustomGlass = Glass.copy(ior = 1.7, color = Color("#E0E0FF"))
  val BrushedGold = Gold.copy(roughness = 0.3)

// scenes/common/Lighting.scala
object Lighting:
  val ThreePoint = List(
    Directional((-1, 1, -1), intensity = 1.0),
    Directional((1, 0.5, -1), intensity = 0.5),
    Directional((0, -1, 1), intensity = 0.3)
  )

// Usage:
import scenes.common.Materials.MyCustomGlass
object MyScene extends SceneDefinition:
  val scene = Scene(objects = List(Sphere(MyCustomGlass)))
```

---

## Tasks

### Task 11.0.1: Fix CI Warning Spam

**Status:** Not Started | **Estimate:** 1-1.5 hours | **Priority:** HIGH

Suppress JVM warnings from Java 21 about deprecated/restricted APIs in Scala 3.7.4, libGDX, LWJGL.

**Create `.sbtopts`:**
```bash
# Suppress Java 21 warnings for Scala 3 and native libraries
-J--add-opens=java.base/java.lang=ALL-UNNAMED
-J--add-opens=java.base/java.nio=ALL-UNNAMED
-J--add-opens=java.base/sun.nio.ch=ALL-UNNAMED
-J--enable-native-access=ALL-UNNAMED
-J-Djdk.internal.lambda.disableEagerInitialization=true
```

**Update `.gitlab-ci.yml`** - Add to global variables:
```yaml
variables:
  JAVA_OPTS: "--add-opens=java.base/java.lang=ALL-UNNAMED --enable-native-access=ALL-UNNAMED"
  SBT_OPTS: "-J--add-opens=java.base/java.lang=ALL-UNNAMED -J--enable-native-access=ALL-UNNAMED"
```

**Verify:** Run `sbt test` locally and in CI - logs should be clean.

---

### Task 11.0.2: Rename Sponge Types for Clarity

**Status:** Not Started | **Estimate:** 1 hour | **Priority:** MEDIUM

Rename confusing sponge type names before implementing DSL.

**Changes:**

3D Sponges - CLI names (backwards compatible):
- `sponge-volume` (keep), `sponge-surface` (keep), `cube-sponge` (keep)
- `sponge` (deprecated) → alias to `sponge-volume` with warning
- `sponge-2` (deprecated) → alias to `sponge-surface` with warning

3D Sponges - DSL names (new):
- `VolumeFilling`, `SurfaceUnfolding`, `CubeSponge`

4D Sponges:
- `tesseract-sponge` → rename to `tesseract-sponge-volume`
- `tesseract-sponge-2` → rename to `tesseract-sponge-surface`
- Keep old names as deprecated aliases with warnings

**Files to Modify:**
- `ObjectType.scala` - Add new enums, deprecation logic
- `MengerCLIOptions.scala` - Update help text
- `docs/USER_GUIDE.md` - Use new names
- Test files - Replace old names

---

### Step 11.1: Create Core DSL Types

**Status:** Not Started | **Estimate:** 2 hours

Create `Vec3`, `Color`, and `Material` types in `menger.dsl` package.

**Files to Create:**

**`Vec3.scala`** - 3D vector with tuple conversions:
```scala
package menger.dsl
import com.badlogic.gdx.math.Vector3 as GdxVector3
import menger.common.Vector

case class Vec3(x: Float, y: Float, z: Float):
  def toGdxVector3: GdxVector3 = GdxVector3(x, y, z)
  def toCommonVector: Vector[3] = Vector[3](x, y, z)

object Vec3:
  val Zero = Vec3(0f, 0f, 0f)
  val UnitX = Vec3(1f, 0f, 0f)
  val UnitY = Vec3(0f, 1f, 0f)
  val UnitZ = Vec3(0f, 0f, 1f)

  given Conversion[(Float, Float, Float), Vec3] = t => Vec3(t._1, t._2, t._3)
  given Conversion[(Int, Int, Int), Vec3] = t => Vec3(t._1.toFloat, t._2.toFloat, t._3.toFloat)
  given Conversion[(Double, Double, Double), Vec3] = t => Vec3(t._1.toFloat, t._2.toFloat, t._3.toFloat)
```

**`Color.scala`** - Color with hex parsing:
```scala
package menger.dsl
import menger.common.Color as CommonColor

case class Color(r: Float, g: Float, b: Float, a: Float = 1.0f):
  def toCommonColor: CommonColor = CommonColor(r, g, b, a)

object Color:
  val White = Color(1f, 1f, 1f)
  val Black = Color(0f, 0f, 0f)
  val Red = Color(1f, 0f, 0f)
  val Green = Color(0f, 1f, 0f)
  val Blue = Color(0f, 0f, 1f)

  def apply(hex: String): Color =
    val cleanHex = if hex.startsWith("#") then hex.substring(1) else hex
    require(cleanHex.length == 6 || cleanHex.length == 8)
    val r = Integer.parseInt(cleanHex.substring(0, 2), 16) / 255f
    val g = Integer.parseInt(cleanHex.substring(2, 4), 16) / 255f
    val b = Integer.parseInt(cleanHex.substring(4, 6), 16) / 255f
    val a = if cleanHex.length == 8 then Integer.parseInt(cleanHex.substring(6, 8), 16) / 255f else 1f
    Color(r, g, b, a)

  given Conversion[String, Color] = Color(_)
```

**`Material.scala`** - Material with presets and factories:
```scala
package menger.dsl
import menger.optix.Material as OptixMaterial

case class Material(
  color: Color = Color.White,
  ior: Float = 1.0f,
  roughness: Float = 0.5f,
  metallic: Float = 0.0f,
  specular: Float = 0.5f
):
  def toOptixMaterial: OptixMaterial =
    OptixMaterial(color.toCommonColor, ior, roughness, metallic, specular)

object Material:
  // Presets
  val Glass = Material(Color(1f, 1f, 1f, 0.02f), ior = 1.5f, roughness = 0f, metallic = 0f, specular = 1f)
  val Water = Material(Color(1f, 1f, 1f, 0.02f), ior = 1.33f, roughness = 0f, metallic = 0f, specular = 1f)
  val Diamond = Material(Color(1f, 1f, 1f, 0.02f), ior = 2.42f, roughness = 0f, metallic = 0f, specular = 1f)
  val Chrome = Material(Color(0.9f, 0.9f, 0.9f), ior = 1f, roughness = 0f, metallic = 1f, specular = 1f)
  val Gold = Material(Color(1f, 0.84f, 0f), ior = 1f, roughness = 0.1f, metallic = 1f, specular = 1f)
  val Copper = Material(Color(0.72f, 0.45f, 0.20f), ior = 1f, roughness = 0.2f, metallic = 1f, specular = 1f)

  // Factories
  def matte(color: Color) = Material(color, ior = 1f, roughness = 1f, metallic = 0f, specular = 0f)
  def plastic(color: Color) = Material(color, ior = 1.5f, roughness = 0.3f, metallic = 0f, specular = 0.5f)
  def metal(color: Color) = Material(color, ior = 1f, roughness = 0.1f, metallic = 1f, specular = 1f)
  def glass(color: Color) = Material(color.copy(a = 0.02f), ior = 1.5f, roughness = 0f, metallic = 0f, specular = 1f)
```

**Tests:** Create `Vec3Spec`, `ColorSpec`, `MaterialSpec` with basic validation tests (hex parsing, tuple conversion, preset properties, factory methods).

---

### Step 11.2: Create Light Types

**Status:** Not Started | **Estimate:** 1 hour

**`Light.scala`** - Light types with tuple support:
```scala
package menger.dsl
import menger.common.Light as CommonLight

sealed trait Light:
  def toCommonLight: CommonLight

case class Directional(
  direction: Vec3,
  intensity: Float = 1.0f,
  color: Color = Color.White
) extends Light:
  def toCommonLight: CommonLight = CommonLight.Directional(
    direction.toCommonVector, color.toCommonColor, intensity)

object Directional:
  // Overloads for tuple positions (Float, Int, Double)
  def apply(direction: (Float, Float, Float), intensity: Float = 1.0f, color: Color = Color.White): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3), intensity, color)
  // Similar for Int, Double tuples...

case class Point(
  position: Vec3,
  intensity: Float = 1.0f,
  color: Color = Color.White
) extends Light:
  def toCommonLight: CommonLight = CommonLight.Point(
    position.toCommonVector, color.toCommonColor, intensity)

object Point:
  // Overloads for tuple positions (similar pattern to Directional)
```

**Tests:** `LightSpec` - Verify tuple construction, defaults, conversion to CommonLight.

---

### Step 11.3: Create Scene Object Types

**Status:** Not Started | **Estimate:** 2.5 hours

**`SceneObject.scala`** - Objects (Sphere, Cube, Sponge):
```scala
package menger.dsl
import menger.ObjectSpec

sealed trait SceneObject:
  def pos: Vec3
  def size: Float
  def toObjectSpec: ObjectSpec

case class Sphere(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None
) extends SceneObject:
  def toObjectSpec: ObjectSpec = ObjectSpec(
    objectType = "sphere", x = pos.x, y = pos.y, z = pos.z, size = size,
    level = None, color = color.map(_.toCommonColor),
    ior = material.map(_.ior).getOrElse(ior),
    material = material.map(_.toOptixMaterial), texture = texture
  )

object Sphere:
  def apply(material: Material): Sphere = Sphere(pos = Vec3.Zero, material = Some(material))
  def apply(pos: Vec3, material: Material): Sphere = Sphere(pos, Some(material))
  def apply(pos: Vec3, material: Material, size: Float): Sphere = Sphere(pos, Some(material), size = size)
  // Tuple overloads for (Float, Float, Float), (Int, Int, Int)...

// Cube - similar structure to Sphere

enum SpongeType(val objectTypeName: String):
  case VolumeFilling extends SpongeType("sponge-volume")
  case SurfaceUnfolding extends SpongeType("sponge-surface")
  case CubeSponge extends SpongeType("cube-sponge")

case class Sponge(
  spongeType: SpongeType,
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None
) extends SceneObject:
  def toObjectSpec: ObjectSpec = ObjectSpec(
    objectType = spongeType.objectTypeName, x = pos.x, y = pos.y, z = pos.z,
    size = size, level = Some(level), color = color.map(_.toCommonColor),
    ior = material.map(_.ior).getOrElse(ior),
    material = material.map(_.toOptixMaterial), texture = texture
  )

object Sponge:
  // Overloads: spongeType + level + optional (position, material, color, size)
  def apply(spongeType: SpongeType, level: Float): Sponge = Sponge(spongeType, Vec3.Zero, level)
  // Additional overloads for position/material/color combinations...

export SpongeType.{VolumeFilling, SurfaceUnfolding, CubeSponge}
```

**Tests:** `SceneObjectSpec` - Verify construction patterns, ObjectSpec conversion, sponge types.

---

### Step 11.4: Create Camera and Plane Types

**Status:** Not Started | **Estimate:** 1.5 hours

**`Camera.scala`** - Camera with arrow syntax:
```scala
package menger.dsl
import menger.config.CameraConfig

case class Camera(
  position: Vec3 = Vec3(0f, 0f, 3f),
  lookAt: Vec3 = Vec3.Zero,
  up: Vec3 = Vec3.UnitY
):
  def toCameraConfig: CameraConfig =
    CameraConfig(position.toGdxVector3, lookAt.toGdxVector3, up.toGdxVector3)

object Camera:
  val Default = Camera()

  def apply(positionToLookAt: (Vec3, Vec3)): Camera =
    Camera(position = positionToLookAt._1, lookAt = positionToLookAt._2)

  extension (pos: Vec3)
    def ->(lookAt: Vec3): (Vec3, Vec3) = (pos, lookAt)

  // Tuple overloads for (Float, Float, Float), (Int, Int, Int)...
```

**`Plane.scala`** - Plane with axis syntax:
```scala
package menger.dsl
import menger.cli.{Axis, PlaneSpec, PlaneColorSpec}

case class AxisPosition(axis: Axis, positive: Boolean, value: Float)

sealed trait AxisHelper:
  def axis: Axis
  def at(value: Float): AxisPosition = AxisPosition(axis, value >= 0, value)
  def at(value: Int): AxisPosition = at(value.toFloat)

case object X extends AxisHelper { val axis = Axis.X }
case object Y extends AxisHelper { val axis = Axis.Y }
case object Z extends AxisHelper { val axis = Axis.Z }

case class Plane(
  axisPosition: AxisPosition,
  color: Option[Color] = None,
  checkered: Option[(Color, Color)] = None
):
  def toPlaneSpec: PlaneSpec =
    PlaneSpec(axisPosition.axis, axisPosition.positive, axisPosition.value)

  def toPlaneColorSpec: Option[PlaneColorSpec] = (color, checkered) match
    case (Some(c), _) => Some(PlaneColorSpec(c.toCommonColor, None))
    case (_, Some((c1, c2))) => Some(PlaneColorSpec(c1.toCommonColor, Some(c2.toCommonColor)))
    case _ => None

object Plane:
  def apply(axisPosition: AxisPosition, color: Color): Plane =
    Plane(axisPosition, Some(color))
  def apply(axisPosition: AxisPosition, color: String): Plane =
    Plane(axisPosition, Some(Color(color)))
  // Checkered overloads...
```

**Tests:** `CameraSpec` (arrow syntax, tuple conversion), `PlaneSpec` (axis syntax, color specs).

---

### Step 11.5: Create Scene, Caustics, and SceneDefinition

**Status:** Not Started | **Estimate:** 3 hours

**`Caustics.scala`** - Caustics config:
```scala
package menger.dsl
import menger.optix.CausticsConfig

case class Caustics(
  photons: Int = 100000,
  iterations: Int = 10,
  radius: Float = 0.1f,
  alpha: Float = 0.7f
):
  require(photons >= 1 && photons <= 10000000)
  require(iterations >= 1 && iterations <= 1000)
  require(radius > 0f && radius <= 10f)
  require(alpha > 0f && alpha < 1f)

  def toCausticsConfig: CausticsConfig =
    CausticsConfig(enabled = true, photons, iterations, radius, alpha)

object Caustics:
  val Disabled: Option[Caustics] = None
  val Default = Caustics()
  val HighQuality = Caustics(photons = 500000, iterations = 20, alpha = 0.8f)
```

**`Scene.scala`** - Complete scene with flat render settings:
```scala
package menger.dsl
import menger.config.OptiXEngineConfig

case class Scene(
  camera: Camera = Camera.Default,
  lights: List[Light] = List.empty,
  objects: List[SceneObject] = List.empty,
  plane: Option[Plane] = None,
  shadows: Boolean = false,
  antialiasing: Boolean = false,
  aaMaxDepth: Int = 2,
  aaThreshold: Float = 0.1f,
  caustics: Option[Caustics] = None
):
  require(lights.size <= 8, s"Maximum 8 lights allowed")
  require(objects.nonEmpty, "Scene must have at least one object")
  require(aaMaxDepth >= 1 && aaMaxDepth <= 4)
  require(aaThreshold >= 0f && aaThreshold <= 1f)

  def toOptiXEngineConfig: OptiXEngineConfig =
    // Convert objects, lights, plane to config structures
    // Return OptiXEngineConfig with all settings
```

**`SceneDefinition.scala`** - Trait for loadable scenes:
```scala
package menger.dsl

trait SceneDefinition:
  val scene: Scene
```

**`package.scala`** - Export all DSL types for convenient imports:
```scala
package menger

package object dsl:
  export menger.dsl.{Vec3, Color, Material, Light, Directional, Point}
  export menger.dsl.{SceneObject, Sphere, Cube, Sponge}
  export menger.dsl.SpongeType.{VolumeFilling, SurfaceUnfolding, CubeSponge}
  export menger.dsl.{Camera, Plane, Caustics, Scene, SceneDefinition}
  export menger.dsl.{X, Y, Z}
  export menger.dsl.Material.{Glass, Water, Diamond, Chrome, Gold, Copper}
  // Export factory methods with capitalized names
```

**Tests:** `CausticsSpec` (validation, defaults), `SceneSpec` (construction, validation, arrow syntax, OptiXEngineConfig conversion).

---

### Step 11.6: Create Scene Loader

**Status:** Not Started | **Estimate:** 2 hours

**`SceneLoader.scala`** - Load scenes by class name:
```scala
package menger.dsl
import scala.util.{Try, Success, Failure}
import menger.config.OptiXEngineConfig

object SceneLoader extends LazyLogging:
  def load(className: String): Either[String, OptiXEngineConfig] =
    for
      clazz <- loadClass(className)
      instance <- getInstance(clazz)
      scene <- extractScene(instance, className)
      config <- validateAndConvert(scene)
    yield config

  private def loadClass(className: String): Either[String, Class[?]] =
    Try(Class.forName(className)) match
      case Success(clazz) => Right(clazz)
      case Failure(e: ClassNotFoundException) =>
        Left(s"Scene class not found: '$className'")
      case Failure(e) => Left(s"Failed to load: ${e.getMessage}")

  private def getInstance(clazz: Class[?]): Either[String, Any] =
    // Try MODULE$ field for Scala objects, fallback to constructor

  private def extractScene(instance: Any, className: String): Either[String, Scene] =
    instance match
      case sd: SceneDefinition => Try(sd.scene).toEither.left.map(_.getMessage)
      case _ => Left(s"'$className' does not extend SceneDefinition")

  private def validateAndConvert(scene: Scene): Either[String, OptiXEngineConfig] =
    Try(scene.toOptiXEngineConfig).toEither.left.map(e => s"Invalid scene: ${e.getMessage}")
```

**Tests:** `SceneLoaderSpec` - Load simple/complex scenes, handle errors (not found, wrong type, init failure).

---

### Step 11.7: Add CLI Option for Scene Loading

**Status:** Not Started | **Estimate:** 1.5 hours

**Modify `MengerCLIOptions.scala`** - Add `--scene` option:
```scala
val sceneClass: ScallopOption[String] = opt[String](
  name = "scene",
  required = false,
  group = optixGroup,
  descr = "Load scene from Scala DSL class (e.g., scenes.MyScene)"
)

// Validation: --scene requires --optix, mutually exclusive with --objects
```

**Modify `Main.scala`** - Integrate scene loading:
```scala
private def createOptiXEngine(opts: MengerCLIOptions): OptiXEngine =
  val engineConfig = opts.sceneClass.toOption match
    case Some(className) =>
      SceneLoader.load(className) match
        case Right(config) =>
          // Apply CLI overrides for execution config
          config.copy(execution = ExecutionConfig(...), ...)
        case Left(error) =>
          System.err.println(s"Error loading scene: $error")
          sys.exit(1)
    case None =>
      // Original CLI-based configuration
      OptiXEngineConfig(...)

  OptiXEngine(engineConfig)
```

**Tests:** `SceneCLIOptionsSuite` - Verify `--scene` parsing, validation, mutual exclusion.

---

### Step 11.8: Create Example Scene Files

**Status:** Not Started | **Estimate:** 1 hour

Create example scenes in `menger-app/src/main/scala/scenes/`:

- `SimpleScene.scala` - Single glass sphere
- `ThreeSpheres.scala` - Three spheres with different materials
- `MengerShowcase.scala` - Sponge with surrounding spheres, checkered plane
- `scenes/common/Materials.scala` - Reusable custom materials
- `scenes/common/Lighting.scala` - Reusable lighting setups (ThreePoint, Dramatic, Soft)
- `CustomMaterialsDemo.scala` - Demo using custom materials and lighting imports

---

### Step 11.9: Add Texture Support to DSL

**Status:** Not Started | **Estimate:** 1.5 hours

**Modify `SceneObject.scala`** - Add `texture: Option[String] = None` parameter to Sphere, Cube, Sponge case classes and include in `toObjectSpec`.

**Tests:** `TextureDSLSpec` - Verify texture parameter in ObjectSpec conversion.

---

### Step 11.10: ~~Add Render Settings to DSL~~

**Status:** Merged into Step 11.5

Render settings are now flat fields on `Scene`. No separate step needed.

---

### Step 11.11: Update Documentation

**Status:** Not Started | **Estimate:** 1 hour

**`CHANGELOG.md`** - Add v0.6.0 section with DSL features.

**`README.md`** - Add "Scene Description Language (DSL)" section with example and usage.

**`TODO.md`** - Add deferred DSL features to backlog (animation, runtime eval, tesseract support, window settings).

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing: `sbt test --warn`
- [ ] Code compiles without warnings
- [ ] Example scenes render correctly
- [ ] Documentation updated (CHANGELOG, README, TODO)

---

## Summary

| Step | Task | Estimate | Files |
|------|------|----------|-------|
| 11.0.1 | Fix CI Warning Spam | 1-1.5h | .sbtopts, CI config |
| 11.0.2 | Rename Sponge Types | 1h | 3-4 files |
| 11.1 | Core DSL Types | 2h | 6 files |
| 11.2 | Light Types | 1h | 2 files |
| 11.3 | Scene Object Types | 2.5h | 2 files |
| 11.4 | Camera and Plane | 1.5h | 4 files |
| 11.5 | Scene, Caustics, SceneDefinition | 3h | 7 files |
| 11.6 | Scene Loader | 2h | 2 files |
| 11.7 | CLI Integration | 1.5h | 3 files |
| 11.8 | Example Scenes | 1h | 6 files |
| 11.9 | Texture Support | 1.5h | 2 files |
| 11.11 | Documentation | 1h | 3 files |
| **Total** | | **~19-20h** | ~40 files |

---

## Notes

### Design Decisions

1. **Compile-time only** - Scenes compile with project for type safety and IDE support
2. **Case-class style only** - No builder pattern, just overloaded `apply` methods
3. **Arrow syntax** - `Camera((0, 0.5, 3) -> (0, 0, 0))` uses Scala extension methods
4. **Flat render settings** - Shadows, AA directly on Scene (not nested object)
5. **Caustics as dedicated type** - Separate case class with 4 related parameters
6. **Tuple conversions** - Implicit conversions for concise syntax
7. **Standard imports** - Reusable definitions via normal Scala imports
8. **CLI-only execution settings** - Window size, save name stay as CLI flags

### Potential Issues

1. **Implicit conversions** - May cause ambiguity, use explicit types if needed
2. **Multiple overloads** - Int/Float/Double tuples increase compile time
3. **Error messages** - Need clear validation errors for DSL users

### Future Extensions

1. Animation keyframes (Sprint 12)
2. Tesseract / 4D support (after Sprint 11)
3. Procedural placement helpers
4. Scene composition utilities

---

## References

- Config classes: `menger-app/src/main/scala/menger/config/`
- ObjectSpec: `menger-app/src/main/scala/menger/ObjectSpec.scala`
- Material presets: `optix-jni/src/main/scala/menger/optix/Material.scala`
- Light types: `menger-common/src/main/scala/menger/common/Light.scala`
