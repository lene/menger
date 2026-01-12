# Sprint 11: Scala DSL for Scene Description

**Sprint:** 11 - Scene Description Language (Scala DSL)
**Status:** Not Started
**Estimate:** 18-23 hours
**Branch:** `feature/sprint-11`
**Dependencies:** None (builds on existing infrastructure)

---

## Goal

Create a Scala DSL that allows concise, type-safe scene definitions that compile with the project and can be loaded via `--scene <classname>`.

## Success Criteria

- [ ] `--scene scenes.MyScene` loads and renders a scene defined in Scala DSL
- [ ] Both block-style and case-class-style syntax work
- [ ] Concise syntax: `sphere(Glass)` creates a glass sphere at origin
- [ ] Scene files can import definitions from other files (standard Scala imports)
- [ ] All current object types, materials, and lights are expressible
- [ ] Texture support: `cube((0,0,0), "brick.png", size = 1.5f)`
- [ ] Render settings: `shadows()`, `antialiasing(4)`, `highQuality()`
- [ ] Comprehensive tests for DSL parsing and scene generation
- [ ] Example scene files demonstrating DSL capabilities

---

## Scope

### In Scope (First Version)
- Objects: sphere, cube, sponge (Volume/Surface/CubeSponge)
- Materials: presets (glass, chrome, gold, etc.) + custom definitions
- Lights: directional, point (up to 8)
- Camera: position, lookAt, up
- Plane: axis, position, color (solid or checkered)
- Textures: per-object texture assignment (`texture = "brick.png"`)
- Render settings: shadows, antialiasing configuration

### Deferred (Backlog)
- Caustics settings (algorithm issues, Sprint 4 deferred)
- Animation keyframes (Sprint 12)
- Runtime evaluation (compile-time only for now)
- Tesseract support (depends on Sprint 8)

---

## DSL Design

### Block-Style Syntax (Primary)

```scala
package scenes

import menger.dsl.*

object MyScene extends SceneDefinition:

  val scene = scene {
    camera(position = (0, 0.5, 3), lookAt = (0, 0, 0))
    
    light(Directional((-1, 1, -1), intensity = 2.0))
    light(Point((0, 5, 0), intensity = 1.5, color = "#FFFFCC"))
    
    sphere(Glass)                                    // at origin, default size
    sphere((2, 0, 0), Chrome, size = 0.8)           // with position, material, size
    cube((-2, 0, 0), color = "#FF0000", size = 1.5) // with color instead of material
    sponge(Volume, level = 3, color = "#00FF00")    // sponge with level
    
    plane(Y at -2, color = "#808080")
  }
```

### Case-Class Style (Alternative)

```scala
package scenes

import menger.dsl.*

object MyScene2 extends SceneDefinition:
  val scene = Scene(
    camera = Camera((0, 0.5, 3) -> (0, 0, 0)),
    lights = List(
      Directional((-1, 1, -1), intensity = 2.0),
      Point((0, 5, 0), intensity = 1.5)
    ),
    objects = List(
      Sphere(Glass),
      Sphere((2, 0, 0), Chrome, size = 0.8),
      Cube((-2, 0, 0), color = "#FF0000")
    ),
    plane = Some(Plane(Y at -2, color = "#808080"))
  )
```

### Reusable Definitions

```scala
// scenes/common/Materials.scala
package scenes.common

import menger.dsl.*

object Materials:
  val MyCustomGlass = Glass.copy(ior = 1.7, color = Color("#E0E0FF"))
  val BrushedGold = Gold.copy(roughness = 0.3)

// scenes/common/Lighting.scala  
package scenes.common

import menger.dsl.*

object Lighting:
  val ThreePoint = List(
    Directional((-1, 1, -1), intensity = 1.0),
    Directional((1, 0.5, -1), intensity = 0.5),
    Directional((0, -1, 1), intensity = 0.3)
  )

// Usage:
import scenes.common.Materials.MyCustomGlass
import scenes.common.Lighting.ThreePoint

object MyScene extends SceneDefinition:
  val scene = scene {
    lights(ThreePoint)
    sphere(MyCustomGlass)
  }
```

---

## Tasks

### Step 11.1: Create Core DSL Types

**Status:** Not Started
**Estimate:** 2 hours

Create the fundamental types for the DSL in `menger.dsl` package.

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/Vec3.scala`**

```scala
package menger.dsl

import com.badlogic.gdx.math.Vector3 as GdxVector3
import menger.common.Vector

/** 3D vector for DSL use with implicit conversions from tuples. */
case class Vec3(x: Float, y: Float, z: Float):
  def toGdxVector3: GdxVector3 = GdxVector3(x, y, z)
  def toCommonVector: Vector[3] = Vector[3](x, y, z)

object Vec3:
  val Zero: Vec3 = Vec3(0f, 0f, 0f)
  val UnitX: Vec3 = Vec3(1f, 0f, 0f)
  val UnitY: Vec3 = Vec3(0f, 1f, 0f)
  val UnitZ: Vec3 = Vec3(0f, 0f, 1f)
  
  /** Implicit conversion from tuple to Vec3 */
  given Conversion[(Float, Float, Float), Vec3] with
    def apply(t: (Float, Float, Float)): Vec3 = Vec3(t._1, t._2, t._3)
  
  /** Implicit conversion from Int tuple to Vec3 */
  given Conversion[(Int, Int, Int), Vec3] with
    def apply(t: (Int, Int, Int)): Vec3 = Vec3(t._1.toFloat, t._2.toFloat, t._3.toFloat)
  
  /** Implicit conversion from Double tuple to Vec3 */
  given Conversion[(Double, Double, Double), Vec3] with
    def apply(t: (Double, Double, Double)): Vec3 = Vec3(t._1.toFloat, t._2.toFloat, t._3.toFloat)
```

**`menger-app/src/main/scala/menger/dsl/Color.scala`**

```scala
package menger.dsl

import menger.common.Color as CommonColor

/** Color for DSL use with hex string parsing. */
case class Color(r: Float, g: Float, b: Float, a: Float = 1.0f):
  def toCommonColor: CommonColor = CommonColor(r, g, b, a)
  def copy(
    r: Float = this.r,
    g: Float = this.g,
    b: Float = this.b,
    a: Float = this.a
  ): Color = Color(r, g, b, a)

object Color:
  val White: Color = Color(1f, 1f, 1f)
  val Black: Color = Color(0f, 0f, 0f)
  val Red: Color = Color(1f, 0f, 0f)
  val Green: Color = Color(0f, 1f, 0f)
  val Blue: Color = Color(0f, 0f, 1f)
  
  /** Parse hex color string (with or without #) */
  def apply(hex: String): Color =
    val cleanHex = if hex.startsWith("#") then hex.substring(1) else hex
    require(cleanHex.length == 6 || cleanHex.length == 8,
      s"Hex color must be 6 or 8 characters, got: $hex")
    val r = Integer.parseInt(cleanHex.substring(0, 2), 16) / 255f
    val g = Integer.parseInt(cleanHex.substring(2, 4), 16) / 255f
    val b = Integer.parseInt(cleanHex.substring(4, 6), 16) / 255f
    val a = if cleanHex.length == 8 then
      Integer.parseInt(cleanHex.substring(6, 8), 16) / 255f
    else 1f
    Color(r, g, b, a)
  
  /** Implicit conversion from hex string to Color */
  given Conversion[String, Color] with
    def apply(hex: String): Color = Color(hex)
```

**`menger-app/src/main/scala/menger/dsl/Material.scala`**

```scala
package menger.dsl

import menger.optix.Material as OptixMaterial

/** Material definition for DSL use. */
case class Material(
  color: Color = Color.White,
  ior: Float = 1.0f,
  roughness: Float = 0.5f,
  metallic: Float = 0.0f,
  specular: Float = 0.5f
):
  def toOptixMaterial: OptixMaterial = OptixMaterial(
    color = color.toCommonColor,
    ior = ior,
    roughness = roughness,
    metallic = metallic,
    specular = specular
  )
  
  def copy(
    color: Color = this.color,
    ior: Float = this.ior,
    roughness: Float = this.roughness,
    metallic: Float = this.metallic,
    specular: Float = this.specular
  ): Material = Material(color, ior, roughness, metallic, specular)

object Material:
  // Dielectric presets (transparent materials with refraction)
  val Glass: Material = Material(
    color = Color(1f, 1f, 1f, 0.02f),
    ior = 1.5f,
    roughness = 0f,
    metallic = 0f,
    specular = 1f
  )
  
  val Water: Material = Material(
    color = Color(1f, 1f, 1f, 0.02f),
    ior = 1.33f,
    roughness = 0f,
    metallic = 0f,
    specular = 1f
  )
  
  val Diamond: Material = Material(
    color = Color(1f, 1f, 1f, 0.02f),
    ior = 2.42f,
    roughness = 0f,
    metallic = 0f,
    specular = 1f
  )
  
  // Metal presets
  val Chrome: Material = Material(
    color = Color(0.9f, 0.9f, 0.9f),
    ior = 1f,
    roughness = 0f,
    metallic = 1f,
    specular = 1f
  )
  
  val Gold: Material = Material(
    color = Color(1f, 0.84f, 0f),
    ior = 1f,
    roughness = 0.1f,
    metallic = 1f,
    specular = 1f
  )
  
  val Copper: Material = Material(
    color = Color(0.72f, 0.45f, 0.20f),
    ior = 1f,
    roughness = 0.2f,
    metallic = 1f,
    specular = 1f
  )
  
  // Factory methods
  def matte(color: Color): Material =
    Material(color, ior = 1f, roughness = 1f, metallic = 0f, specular = 0f)
  
  def plastic(color: Color): Material =
    Material(color, ior = 1.5f, roughness = 0.3f, metallic = 0f, specular = 0.5f)
  
  def metal(color: Color): Material =
    Material(color, ior = 1f, roughness = 0.1f, metallic = 1f, specular = 1f)
  
  def glass(color: Color): Material =
    Material(color.copy(a = 0.02f), ior = 1.5f, roughness = 0f, metallic = 0f, specular = 1f)
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/Vec3Spec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Vec3Spec extends AnyFlatSpec with Matchers:

  "Vec3" should "create from explicit values" in:
    val v = Vec3(1f, 2f, 3f)
    v.x shouldBe 1f
    v.y shouldBe 2f
    v.z shouldBe 3f

  it should "convert from Float tuple" in:
    val v: Vec3 = (1.0f, 2.0f, 3.0f)
    v shouldBe Vec3(1f, 2f, 3f)

  it should "convert from Int tuple" in:
    val v: Vec3 = (1, 2, 3)
    v shouldBe Vec3(1f, 2f, 3f)

  it should "convert from Double tuple" in:
    val v: Vec3 = (1.0, 2.0, 3.0)
    v shouldBe Vec3(1f, 2f, 3f)

  it should "convert to GdxVector3" in:
    val v = Vec3(1f, 2f, 3f)
    val gdx = v.toGdxVector3
    gdx.x shouldBe 1f
    gdx.y shouldBe 2f
    gdx.z shouldBe 3f

  it should "convert to CommonVector" in:
    val v = Vec3(1f, 2f, 3f)
    val common = v.toCommonVector
    common(0) shouldBe 1f
    common(1) shouldBe 2f
    common(2) shouldBe 3f

  it should "have Zero constant" in:
    Vec3.Zero shouldBe Vec3(0f, 0f, 0f)
```

**`menger-app/src/test/scala/menger/dsl/ColorSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ColorSpec extends AnyFlatSpec with Matchers:

  "Color" should "parse 6-digit hex without #" in:
    val c = Color("FF0000")
    c.r shouldBe 1f +- 0.01f
    c.g shouldBe 0f +- 0.01f
    c.b shouldBe 0f +- 0.01f
    c.a shouldBe 1f

  it should "parse 6-digit hex with #" in:
    val c = Color("#00FF00")
    c.r shouldBe 0f +- 0.01f
    c.g shouldBe 1f +- 0.01f
    c.b shouldBe 0f +- 0.01f

  it should "parse 8-digit hex with alpha" in:
    val c = Color("#FF000080")
    c.r shouldBe 1f +- 0.01f
    c.a shouldBe 0.5f +- 0.01f

  it should "convert from string implicitly" in:
    val c: Color = "#0000FF"
    c.b shouldBe 1f +- 0.01f

  it should "reject invalid hex" in:
    an[IllegalArgumentException] should be thrownBy Color("GGG")
    an[IllegalArgumentException] should be thrownBy Color("FF00")

  it should "have predefined constants" in:
    Color.White shouldBe Color(1f, 1f, 1f)
    Color.Black shouldBe Color(0f, 0f, 0f)
    Color.Red shouldBe Color(1f, 0f, 0f)
```

**`menger-app/src/test/scala/menger/dsl/MaterialSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MaterialSpec extends AnyFlatSpec with Matchers:

  "Material presets" should "have correct Glass properties" in:
    val glass = Material.Glass
    glass.ior shouldBe 1.5f
    glass.roughness shouldBe 0f
    glass.metallic shouldBe 0f
    glass.specular shouldBe 1f
    glass.color.a shouldBe 0.02f +- 0.01f

  it should "have correct Chrome properties" in:
    val chrome = Material.Chrome
    chrome.metallic shouldBe 1f
    chrome.roughness shouldBe 0f

  it should "have correct Gold properties" in:
    val gold = Material.Gold
    gold.metallic shouldBe 1f
    gold.color.r shouldBe 1f
    gold.color.g shouldBe 0.84f +- 0.01f

  "Material.copy" should "allow overriding properties" in:
    val customGlass = Material.Glass.copy(ior = 1.7f)
    customGlass.ior shouldBe 1.7f
    customGlass.roughness shouldBe 0f  // unchanged

  "Material factory methods" should "create matte materials" in:
    val m = Material.matte(Color.Red)
    m.roughness shouldBe 1f
    m.metallic shouldBe 0f
    m.specular shouldBe 0f

  it should "create plastic materials" in:
    val m = Material.plastic(Color.Blue)
    m.roughness shouldBe 0.3f
    m.metallic shouldBe 0f

  it should "create metal materials" in:
    val m = Material.metal(Color.White)
    m.metallic shouldBe 1f
    m.roughness shouldBe 0.1f

  "Material.toOptixMaterial" should "convert correctly" in:
    val dsl = Material.Glass
    val optix = dsl.toOptixMaterial
    optix.ior shouldBe dsl.ior
    optix.roughness shouldBe dsl.roughness
    optix.metallic shouldBe dsl.metallic
```

---

### Step 11.2: Create Light Types

**Status:** Not Started
**Estimate:** 1 hour

Create DSL types for light sources.

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/Light.scala`**

```scala
package menger.dsl

import menger.common.Light as CommonLight
import menger.common.LightType as CommonLightType

/** Base trait for DSL light definitions. */
sealed trait Light:
  def toCommonLight: CommonLight

/** Directional light (like sun). */
case class Directional(
  direction: Vec3,
  intensity: Float = 1.0f,
  color: Color = Color.White
) extends Light:
  def toCommonLight: CommonLight = CommonLight.Directional(
    direction = direction.toCommonVector,
    color = color.toCommonColor,
    intensity = intensity
  )

object Directional:
  /** Create from tuple (uses implicit conversion) */
  def apply(direction: (Float, Float, Float), intensity: Float, color: Color): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3), intensity, color)
  
  def apply(direction: (Float, Float, Float), intensity: Float): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3), intensity, Color.White)
  
  def apply(direction: (Float, Float, Float)): Directional =
    Directional(Vec3(direction._1, direction._2, direction._3), 1.0f, Color.White)
  
  /** Overloads for Int tuples */
  def apply(direction: (Int, Int, Int), intensity: Float, color: Color): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), intensity, color)
  
  def apply(direction: (Int, Int, Int), intensity: Float): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), intensity, Color.White)
  
  def apply(direction: (Int, Int, Int)): Directional =
    Directional(Vec3(direction._1.toFloat, direction._2.toFloat, direction._3.toFloat), 1.0f, Color.White)

/** Point light (like light bulb). */
case class Point(
  position: Vec3,
  intensity: Float = 1.0f,
  color: Color = Color.White
) extends Light:
  def toCommonLight: CommonLight = CommonLight.Point(
    position = position.toCommonVector,
    color = color.toCommonColor,
    intensity = intensity
  )

object Point:
  /** Create from tuple */
  def apply(position: (Float, Float, Float), intensity: Float, color: Color): Point =
    Point(Vec3(position._1, position._2, position._3), intensity, color)
  
  def apply(position: (Float, Float, Float), intensity: Float): Point =
    Point(Vec3(position._1, position._2, position._3), intensity, Color.White)
  
  def apply(position: (Float, Float, Float)): Point =
    Point(Vec3(position._1, position._2, position._3), 1.0f, Color.White)
  
  /** Overloads for Int tuples */
  def apply(position: (Int, Int, Int), intensity: Float, color: Color): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), intensity, color)
  
  def apply(position: (Int, Int, Int), intensity: Float): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), intensity, Color.White)
  
  def apply(position: (Int, Int, Int)): Point =
    Point(Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat), 1.0f, Color.White)
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/LightSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.LightType as CommonLightType

class LightSpec extends AnyFlatSpec with Matchers:

  "Directional" should "create with Vec3 direction" in:
    val light = Directional(Vec3(-1f, 1f, -1f), intensity = 2.0f)
    light.direction shouldBe Vec3(-1f, 1f, -1f)
    light.intensity shouldBe 2.0f
    light.color shouldBe Color.White

  it should "create from Float tuple" in:
    val light = Directional((-1f, 1f, -1f), intensity = 2.0f)
    light.direction shouldBe Vec3(-1f, 1f, -1f)

  it should "create from Int tuple" in:
    val light = Directional((-1, 1, -1), intensity = 2.0f)
    light.direction shouldBe Vec3(-1f, 1f, -1f)

  it should "use default intensity of 1.0" in:
    val light = Directional((-1, 1, -1))
    light.intensity shouldBe 1.0f

  it should "accept custom color" in:
    val light = Directional((-1, 1, -1), intensity = 1.0f, color = Color("#FF0000"))
    light.color.r shouldBe 1f +- 0.01f

  it should "convert to CommonLight" in:
    val light = Directional((-1f, 1f, -1f), intensity = 2.0f)
    val common = light.toCommonLight
    common.lightType shouldBe CommonLightType.Directional
    common.intensity shouldBe 2.0f

  "Point" should "create with Vec3 position" in:
    val light = Point(Vec3(0f, 5f, 0f), intensity = 1.5f)
    light.position shouldBe Vec3(0f, 5f, 0f)
    light.intensity shouldBe 1.5f

  it should "create from tuple" in:
    val light = Point((0, 5, 0), intensity = 1.5f)
    light.position shouldBe Vec3(0f, 5f, 0f)

  it should "convert to CommonLight" in:
    val light = Point((0f, 5f, 0f), intensity = 1.5f)
    val common = light.toCommonLight
    common.lightType shouldBe CommonLightType.Point
    common.intensity shouldBe 1.5f
```

---

### Step 11.3: Create Scene Object Types

**Status:** Not Started
**Estimate:** 2.5 hours

Create DSL types for scene objects (sphere, cube, sponge).

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/SceneObject.scala`**

```scala
package menger.dsl

import menger.ObjectSpec
import menger.optix.Material as OptixMaterial

/** Base trait for all DSL scene objects. */
sealed trait SceneObject:
  def pos: Vec3
  def size: Float
  def toObjectSpec: ObjectSpec

/** Sphere object. */
case class Sphere(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f
) extends SceneObject:
  def toObjectSpec: ObjectSpec = ObjectSpec(
    objectType = "sphere",
    x = pos.x,
    y = pos.y,
    z = pos.z,
    size = size,
    level = None,
    color = color.map(_.toCommonColor),
    ior = material.map(_.ior).getOrElse(ior),
    material = material.map(_.toOptixMaterial),
    texture = None
  )

object Sphere:
  /** Sphere with just material at origin */
  def apply(material: Material): Sphere =
    Sphere(pos = Vec3.Zero, material = Some(material))
  
  /** Sphere with position and material */
  def apply(pos: Vec3, material: Material): Sphere =
    Sphere(pos = pos, material = Some(material))
  
  def apply(pos: Vec3, material: Material, size: Float): Sphere =
    Sphere(pos = pos, material = Some(material), size = size)
  
  /** Sphere from tuple position */
  def apply(pos: (Float, Float, Float), material: Material): Sphere =
    Sphere(Vec3(pos._1, pos._2, pos._3), material)
  
  def apply(pos: (Float, Float, Float), material: Material, size: Float): Sphere =
    Sphere(Vec3(pos._1, pos._2, pos._3), material, size)
  
  def apply(pos: (Int, Int, Int), material: Material): Sphere =
    Sphere(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), material)
  
  def apply(pos: (Int, Int, Int), material: Material, size: Float): Sphere =
    Sphere(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), material, size)

/** Cube object. */
case class Cube(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f
) extends SceneObject:
  def toObjectSpec: ObjectSpec = ObjectSpec(
    objectType = "cube",
    x = pos.x,
    y = pos.y,
    z = pos.z,
    size = size,
    level = None,
    color = color.map(_.toCommonColor),
    ior = material.map(_.ior).getOrElse(ior),
    material = material.map(_.toOptixMaterial),
    texture = None
  )

object Cube:
  def apply(material: Material): Cube =
    Cube(pos = Vec3.Zero, material = Some(material))
  
  def apply(pos: Vec3, material: Material): Cube =
    Cube(pos = pos, material = Some(material))
  
  def apply(pos: Vec3, material: Material, size: Float): Cube =
    Cube(pos = pos, material = Some(material), size = size)
  
  def apply(pos: (Float, Float, Float), material: Material): Cube =
    Cube(Vec3(pos._1, pos._2, pos._3), material)
  
  def apply(pos: (Float, Float, Float), material: Material, size: Float): Cube =
    Cube(Vec3(pos._1, pos._2, pos._3), material, size)
  
  def apply(pos: (Int, Int, Int), material: Material): Cube =
    Cube(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), material)
  
  def apply(pos: (Int, Int, Int), material: Material, size: Float): Cube =
    Cube(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), material, size)

/** Sponge type variants. */
enum SpongeType(val objectTypeName: String):
  case Volume extends SpongeType("sponge-volume")
  case Surface extends SpongeType("sponge-surface")
  case CubeSponge extends SpongeType("cube-sponge")

/** Sponge object (Menger sponge variants). */
case class Sponge(
  spongeType: SpongeType,
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f
) extends SceneObject:
  def toObjectSpec: ObjectSpec = ObjectSpec(
    objectType = spongeType.objectTypeName,
    x = pos.x,
    y = pos.y,
    z = pos.z,
    size = size,
    level = Some(level),
    color = color.map(_.toCommonColor),
    ior = material.map(_.ior).getOrElse(ior),
    material = material.map(_.toOptixMaterial),
    texture = None
  )

object Sponge:
  /** Sponge at origin with level */
  def apply(spongeType: SpongeType, level: Float): Sponge =
    Sponge(spongeType = spongeType, level = level)
  
  def apply(spongeType: SpongeType, level: Float, color: Color): Sponge =
    Sponge(spongeType = spongeType, level = level, color = Some(color))
  
  def apply(spongeType: SpongeType, level: Float, material: Material): Sponge =
    Sponge(spongeType = spongeType, level = level, material = Some(material))
  
  /** Sponge with position */
  def apply(spongeType: SpongeType, pos: Vec3, level: Float): Sponge =
    Sponge(spongeType = spongeType, pos = pos, level = level)
  
  def apply(spongeType: SpongeType, pos: Vec3, level: Float, color: Color): Sponge =
    Sponge(spongeType = spongeType, pos = pos, level = level, color = Some(color))
  
  def apply(spongeType: SpongeType, pos: Vec3, level: Float, material: Material): Sponge =
    Sponge(spongeType = spongeType, pos = pos, level = level, material = Some(material))
  
  /** Tuple position overloads */
  def apply(spongeType: SpongeType, pos: (Float, Float, Float), level: Float): Sponge =
    Sponge(spongeType, Vec3(pos._1, pos._2, pos._3), level)
  
  def apply(spongeType: SpongeType, pos: (Float, Float, Float), level: Float, color: Color): Sponge =
    Sponge(spongeType, Vec3(pos._1, pos._2, pos._3), level, Some(color))
  
  def apply(spongeType: SpongeType, pos: (Int, Int, Int), level: Float): Sponge =
    Sponge(spongeType, Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), level)

// Export SpongeType variants for convenience
export SpongeType.{Volume, Surface, CubeSponge}
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/SceneObjectSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneObjectSpec extends AnyFlatSpec with Matchers:

  // === Sphere Tests ===

  "Sphere" should "create at origin with just material" in:
    val s = Sphere(Material.Glass)
    s.pos shouldBe Vec3.Zero
    s.material shouldBe Some(Material.Glass)
    s.size shouldBe 1.0f

  it should "create with position and material" in:
    val s = Sphere((2, 0, 0), Material.Chrome)
    s.pos shouldBe Vec3(2f, 0f, 0f)
    s.material shouldBe Some(Material.Chrome)

  it should "create with position, material, and size" in:
    val s = Sphere((1, 2, 3), Material.Gold, size = 0.5f)
    s.pos shouldBe Vec3(1f, 2f, 3f)
    s.size shouldBe 0.5f

  it should "convert to ObjectSpec correctly" in:
    val s = Sphere((1f, 2f, 3f), Material.Glass, size = 2.0f)
    val spec = s.toObjectSpec
    spec.objectType shouldBe "sphere"
    spec.x shouldBe 1f
    spec.y shouldBe 2f
    spec.z shouldBe 3f
    spec.size shouldBe 2.0f
    spec.material.isDefined shouldBe true

  // === Cube Tests ===

  "Cube" should "create at origin with material" in:
    val c = Cube(Material.Chrome)
    c.pos shouldBe Vec3.Zero
    c.material shouldBe Some(Material.Chrome)

  it should "create with color instead of material" in:
    val c = Cube(pos = Vec3(1f, 0f, 0f), color = Some(Color("#FF0000")))
    c.color.isDefined shouldBe true
    c.material shouldBe None

  it should "convert to ObjectSpec correctly" in:
    val c = Cube((1, 2, 3), Material.Gold, size = 1.5f)
    val spec = c.toObjectSpec
    spec.objectType shouldBe "cube"
    spec.x shouldBe 1f
    spec.size shouldBe 1.5f

  // === Sponge Tests ===

  "Sponge" should "create Volume type with level" in:
    val s = Sponge(Volume, level = 3)
    s.spongeType shouldBe Volume
    s.level shouldBe 3f
    s.pos shouldBe Vec3.Zero

  it should "create Surface type with position and level" in:
    val s = Sponge(Surface, (1, 2, 3), level = 2.5f)
    s.spongeType shouldBe Surface
    s.pos shouldBe Vec3(1f, 2f, 3f)
    s.level shouldBe 2.5f

  it should "create CubeSponge with color" in:
    val s = Sponge(CubeSponge, level = 2, color = Color("#00FF00"))
    s.spongeType shouldBe CubeSponge
    s.color.isDefined shouldBe true

  it should "convert Volume to correct ObjectSpec type" in:
    val s = Sponge(Volume, level = 3)
    s.toObjectSpec.objectType shouldBe "sponge-volume"

  it should "convert Surface to correct ObjectSpec type" in:
    val s = Sponge(Surface, level = 2)
    s.toObjectSpec.objectType shouldBe "sponge-surface"

  it should "convert CubeSponge to correct ObjectSpec type" in:
    val s = Sponge(CubeSponge, level = 2)
    s.toObjectSpec.objectType shouldBe "cube-sponge"

  it should "include level in ObjectSpec" in:
    val s = Sponge(Volume, level = 3.5f)
    s.toObjectSpec.level shouldBe Some(3.5f)
```

---

### Step 11.4: Create Camera and Plane Types

**Status:** Not Started
**Estimate:** 1.5 hours

Create DSL types for camera and ground plane.

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/Camera.scala`**

```scala
package menger.dsl

import menger.config.CameraConfig

/** Camera configuration for DSL use. */
case class Camera(
  position: Vec3 = Vec3(0f, 0f, 3f),
  lookAt: Vec3 = Vec3.Zero,
  up: Vec3 = Vec3.UnitY
):
  def toCameraConfig: CameraConfig = CameraConfig(
    position = position.toGdxVector3,
    lookAt = lookAt.toGdxVector3,
    up = up.toGdxVector3
  )

object Camera:
  val Default: Camera = Camera()
  
  /** Create camera with position -> lookAt syntax */
  def apply(positionToLookAt: (Vec3, Vec3)): Camera =
    Camera(position = positionToLookAt._1, lookAt = positionToLookAt._2)
  
  /** Arrow syntax helper: position -> lookAt */
  extension (pos: Vec3)
    def ->(lookAt: Vec3): (Vec3, Vec3) = (pos, lookAt)
  
  /** Create from tuple positions */
  def apply(position: (Float, Float, Float), lookAt: (Float, Float, Float)): Camera =
    Camera(
      position = Vec3(position._1, position._2, position._3),
      lookAt = Vec3(lookAt._1, lookAt._2, lookAt._3)
    )
  
  def apply(position: (Int, Int, Int), lookAt: (Int, Int, Int)): Camera =
    Camera(
      position = Vec3(position._1.toFloat, position._2.toFloat, position._3.toFloat),
      lookAt = Vec3(lookAt._1.toFloat, lookAt._2.toFloat, lookAt._3.toFloat)
    )
```

**`menger-app/src/main/scala/menger/dsl/Plane.scala`**

```scala
package menger.dsl

import menger.cli.Axis
import menger.cli.PlaneSpec
import menger.cli.PlaneColorSpec

/** Axis with position for plane definition. */
case class AxisPosition(axis: Axis, positive: Boolean, value: Float)

/** Axis helpers for DSL syntax: Y at -2 */
sealed trait AxisHelper:
  def axis: Axis
  def at(value: Float): AxisPosition = 
    AxisPosition(axis, positive = value >= 0, value)
  def at(value: Int): AxisPosition = at(value.toFloat)

case object X extends AxisHelper:
  val axis: Axis = Axis.X

case object Y extends AxisHelper:
  val axis: Axis = Axis.Y

case object Z extends AxisHelper:
  val axis: Axis = Axis.Z

/** Ground plane configuration. */
case class Plane(
  axisPosition: AxisPosition,
  color: Option[Color] = None,
  checkered: Option[(Color, Color)] = None
):
  require(color.isDefined || checkered.isDefined || (color.isEmpty && checkered.isEmpty),
    "Plane must have either solid color, checkered colors, or no color")
  
  def toPlaneSpec: PlaneSpec = PlaneSpec(
    axis = axisPosition.axis,
    positive = axisPosition.positive,
    value = axisPosition.value
  )
  
  def toPlaneColorSpec: Option[PlaneColorSpec] =
    (color, checkered) match
      case (Some(c), _) => Some(PlaneColorSpec(c.toCommonColor, None))
      case (_, Some((c1, c2))) => Some(PlaneColorSpec(c1.toCommonColor, Some(c2.toCommonColor)))
      case _ => None

object Plane:
  /** Plane with solid color */
  def apply(axisPosition: AxisPosition, color: Color): Plane =
    Plane(axisPosition, color = Some(color))
  
  /** Plane with solid color from string */
  def apply(axisPosition: AxisPosition, color: String): Plane =
    Plane(axisPosition, color = Some(Color(color)))
  
  /** Plane with checkered pattern */
  def apply(axisPosition: AxisPosition, checkered: (Color, Color)): Plane =
    Plane(axisPosition, checkered = Some(checkered))
  
  /** Plane with checkered pattern from strings */
  def apply(axisPosition: AxisPosition, color1: String, color2: String): Plane =
    Plane(axisPosition, checkered = Some((Color(color1), Color(color2))))
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/CameraSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CameraSpec extends AnyFlatSpec with Matchers:

  "Camera" should "have sensible defaults" in:
    val cam = Camera()
    cam.position shouldBe Vec3(0f, 0f, 3f)
    cam.lookAt shouldBe Vec3.Zero
    cam.up shouldBe Vec3.UnitY

  it should "create with explicit position and lookAt" in:
    val cam = Camera(position = Vec3(1f, 2f, 3f), lookAt = Vec3(0f, 0f, 0f))
    cam.position shouldBe Vec3(1f, 2f, 3f)
    cam.lookAt shouldBe Vec3.Zero

  it should "create from tuple positions" in:
    val cam = Camera((0, 0.5, 3), (0, 0, 0))
    cam.position shouldBe Vec3(0f, 0.5f, 3f)
    cam.lookAt shouldBe Vec3.Zero

  it should "create from Float tuple positions" in:
    val cam = Camera((0f, 0.5f, 3f), (0f, 0f, 0f))
    cam.position shouldBe Vec3(0f, 0.5f, 3f)

  it should "support arrow syntax" in:
    import Camera.->;
    val cam = Camera(Vec3(0f, 0.5f, 3f) -> Vec3.Zero)
    cam.position shouldBe Vec3(0f, 0.5f, 3f)
    cam.lookAt shouldBe Vec3.Zero

  it should "convert to CameraConfig" in:
    val cam = Camera((1, 2, 3), (0, 0, 0))
    val config = cam.toCameraConfig
    config.position.x shouldBe 1f
    config.position.y shouldBe 2f
    config.position.z shouldBe 3f
    config.lookAt.x shouldBe 0f
```

**`menger-app/src/test/scala/menger/dsl/PlaneSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.cli.Axis

class PlaneSpec extends AnyFlatSpec with Matchers:

  "AxisHelper" should "create Y at -2" in:
    val ap = Y at -2
    ap.axis shouldBe Axis.Y
    ap.value shouldBe -2f
    ap.positive shouldBe false

  it should "create X at 5" in:
    val ap = X at 5
    ap.axis shouldBe Axis.X
    ap.value shouldBe 5f
    ap.positive shouldBe true

  it should "create Z at 0" in:
    val ap = Z at 0
    ap.axis shouldBe Axis.Z
    ap.value shouldBe 0f
    ap.positive shouldBe true

  "Plane" should "create with solid color" in:
    val p = Plane(Y at -2, color = Color("#808080"))
    p.color.isDefined shouldBe true
    p.checkered shouldBe None

  it should "create with solid color from string" in:
    val p = Plane(Y at -2, "808080")
    p.color.isDefined shouldBe true

  it should "create with checkered pattern" in:
    val p = Plane(Y at -2, checkered = (Color.White, Color.Black))
    p.checkered.isDefined shouldBe true
    p.color shouldBe None

  it should "create with checkered pattern from strings" in:
    val p = Plane(Y at -2, "FFFFFF", "000000")
    p.checkered.isDefined shouldBe true

  it should "convert to PlaneSpec" in:
    val p = Plane(Y at -2, "808080")
    val spec = p.toPlaneSpec
    spec.axis shouldBe Axis.Y
    spec.value shouldBe -2f
    spec.positive shouldBe false

  it should "convert solid color to PlaneColorSpec" in:
    val p = Plane(Y at -2, color = Color.White)
    val colorSpec = p.toPlaneColorSpec
    colorSpec.isDefined shouldBe true
    colorSpec.get.isSolid shouldBe true

  it should "convert checkered to PlaneColorSpec" in:
    val p = Plane(Y at -2, checkered = (Color.White, Color.Black))
    val colorSpec = p.toPlaneColorSpec
    colorSpec.isDefined shouldBe true
    colorSpec.get.isCheckered shouldBe true
```

---

### Step 11.5: Create Scene and SceneBuilder

**Status:** Not Started
**Estimate:** 3 hours

Create the main Scene class and the builder for block-style DSL.

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/Scene.scala`**

```scala
package menger.dsl

import com.badlogic.gdx.graphics.{Color => GdxColor}
import com.badlogic.gdx.math.Vector3
import menger.ObjectSpec
import menger.cli.LightSpec
import menger.cli.LightType as CliLightType
import menger.cli.PlaneColorSpec
import menger.cli.PlaneSpec
import menger.config.CameraConfig
import menger.config.EnvironmentConfig
import menger.config.ExecutionConfig
import menger.config.OptiXEngineConfig
import menger.config.SceneConfig

/** Complete scene definition. */
case class Scene(
  camera: Camera = Camera.Default,
  lights: List[Light] = List.empty,
  objects: List[SceneObject] = List.empty,
  plane: Option[Plane] = None
):
  require(lights.size <= 8, s"Maximum 8 lights allowed, got ${lights.size}")
  require(objects.nonEmpty, "Scene must have at least one object")
  
  /** Convert to OptiXEngineConfig for rendering. */
  def toOptiXEngineConfig: OptiXEngineConfig =
    val objectSpecs = objects.map(_.toObjectSpec)
    
    val lightSpecs = lights.map { light =>
      light match
        case d: Directional =>
          LightSpec(
            CliLightType.DIRECTIONAL,
            Vector3(d.direction.x, d.direction.y, d.direction.z),
            d.intensity,
            GdxColor(d.color.r, d.color.g, d.color.b, d.color.a)
          )
        case p: Point =>
          LightSpec(
            CliLightType.POINT,
            Vector3(p.position.x, p.position.y, p.position.z),
            p.intensity,
            GdxColor(p.color.r, p.color.g, p.color.b, p.color.a)
          )
    }
    
    val planeSpec = plane.map(_.toPlaneSpec).getOrElse(
      PlaneSpec(menger.cli.Axis.Y, positive = true, -2f)
    )
    val planeColorSpec = plane.flatMap(_.toPlaneColorSpec)
    
    OptiXEngineConfig(
      scene = SceneConfig(objectSpecs = Some(objectSpecs)),
      camera = camera.toCameraConfig,
      environment = EnvironmentConfig(
        plane = planeSpec,
        planeColor = planeColorSpec,
        lights = lightSpecs
      ),
      execution = ExecutionConfig.Default
    )
```

**`menger-app/src/main/scala/menger/dsl/SceneBuilder.scala`**

```scala
package menger.dsl

import scala.collection.mutable.ListBuffer

/** Mutable builder for block-style DSL syntax. */
class SceneBuilder:
  private var _camera: Camera = Camera.Default
  private val _lights: ListBuffer[Light] = ListBuffer.empty
  private val _objects: ListBuffer[SceneObject] = ListBuffer.empty
  private var _plane: Option[Plane] = None
  
  // === Camera ===
  
  def camera(
    position: Vec3 = Vec3(0f, 0f, 3f),
    lookAt: Vec3 = Vec3.Zero,
    up: Vec3 = Vec3.UnitY
  ): Unit =
    _camera = Camera(position, lookAt, up)
  
  def camera(position: (Float, Float, Float), lookAt: (Float, Float, Float)): Unit =
    _camera = Camera(position, lookAt)
  
  def camera(position: (Int, Int, Int), lookAt: (Int, Int, Int)): Unit =
    _camera = Camera(position, lookAt)
  
  // === Lights ===
  
  def light(l: Light): Unit =
    require(_lights.size < 8, "Maximum 8 lights allowed")
    _lights += l
  
  def lights(ls: List[Light]): Unit =
    require(_lights.size + ls.size <= 8, s"Maximum 8 lights allowed, would have ${_lights.size + ls.size}")
    _lights ++= ls
  
  // === Objects ===
  
  /** Add a sphere to the scene. */
  def sphere(material: Material): Unit =
    _objects += Sphere(material)
  
  def sphere(pos: Vec3, material: Material, size: Float = 1.0f): Unit =
    _objects += Sphere(pos, material, size)
  
  def sphere(pos: (Float, Float, Float), material: Material): Unit =
    _objects += Sphere(pos, material)
  
  def sphere(pos: (Float, Float, Float), material: Material, size: Float): Unit =
    _objects += Sphere(pos, material, size)
  
  def sphere(pos: (Int, Int, Int), material: Material): Unit =
    _objects += Sphere(pos, material)
  
  def sphere(pos: (Int, Int, Int), material: Material, size: Float): Unit =
    _objects += Sphere(pos, material, size)
  
  /** Add a cube to the scene. */
  def cube(material: Material): Unit =
    _objects += Cube(material)
  
  def cube(pos: Vec3, material: Material, size: Float = 1.0f): Unit =
    _objects += Cube(pos, material, size)
  
  def cube(pos: (Float, Float, Float), material: Material): Unit =
    _objects += Cube(pos, material)
  
  def cube(pos: (Float, Float, Float), material: Material, size: Float): Unit =
    _objects += Cube(pos, material, size)
  
  def cube(pos: (Int, Int, Int), material: Material): Unit =
    _objects += Cube(pos, material)
  
  def cube(pos: (Int, Int, Int), material: Material, size: Float): Unit =
    _objects += Cube(pos, material, size)
  
  /** Cube with color instead of material */
  def cube(pos: Vec3, color: Color, size: Float): Unit =
    _objects += Cube(pos = pos, color = Some(color), size = size)
  
  def cube(pos: (Float, Float, Float), color: String, size: Float): Unit =
    _objects += Cube(pos = Vec3(pos._1, pos._2, pos._3), color = Some(Color(color)), size = size)
  
  def cube(pos: (Int, Int, Int), color: String, size: Float): Unit =
    _objects += Cube(pos = Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), color = Some(Color(color)), size = size)
  
  /** Add a sponge to the scene. */
  def sponge(spongeType: SpongeType, level: Float): Unit =
    _objects += Sponge(spongeType, level = level)
  
  def sponge(spongeType: SpongeType, level: Float, color: Color): Unit =
    _objects += Sponge(spongeType, level = level, color = Some(color))
  
  def sponge(spongeType: SpongeType, level: Float, color: String): Unit =
    _objects += Sponge(spongeType, level = level, color = Some(Color(color)))
  
  def sponge(spongeType: SpongeType, level: Float, material: Material): Unit =
    _objects += Sponge(spongeType, level = level, material = Some(material))
  
  def sponge(spongeType: SpongeType, pos: Vec3, level: Float): Unit =
    _objects += Sponge(spongeType, pos = pos, level = level)
  
  def sponge(spongeType: SpongeType, pos: Vec3, level: Float, color: Color): Unit =
    _objects += Sponge(spongeType, pos = pos, level = level, color = Some(color))
  
  def sponge(spongeType: SpongeType, pos: Vec3, level: Float, material: Material): Unit =
    _objects += Sponge(spongeType, pos = pos, level = level, material = Some(material))
  
  def sponge(spongeType: SpongeType, pos: (Float, Float, Float), level: Float): Unit =
    _objects += Sponge(spongeType, Vec3(pos._1, pos._2, pos._3), level)
  
  def sponge(spongeType: SpongeType, pos: (Float, Float, Float), level: Float, color: String): Unit =
    _objects += Sponge(spongeType, Vec3(pos._1, pos._2, pos._3), level, color = Some(Color(color)))
  
  def sponge(spongeType: SpongeType, pos: (Int, Int, Int), level: Float): Unit =
    _objects += Sponge(spongeType, Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), level)
  
  /** Add any SceneObject directly. */
  def add(obj: SceneObject): Unit =
    _objects += obj
  
  def add(objs: List[SceneObject]): Unit =
    _objects ++= objs
  
  // === Plane ===
  
  def plane(axisPosition: AxisPosition, color: Color): Unit =
    _plane = Some(Plane(axisPosition, color = Some(color)))
  
  def plane(axisPosition: AxisPosition, color: String): Unit =
    _plane = Some(Plane(axisPosition, color))
  
  def plane(axisPosition: AxisPosition, checkered: (Color, Color)): Unit =
    _plane = Some(Plane(axisPosition, checkered = Some(checkered)))
  
  def plane(axisPosition: AxisPosition, color1: String, color2: String): Unit =
    _plane = Some(Plane(axisPosition, color1, color2))
  
  // === Build ===
  
  def build(): Scene =
    require(_objects.nonEmpty, "Scene must have at least one object")
    Scene(
      camera = _camera,
      lights = _lights.toList,
      objects = _objects.toList,
      plane = _plane
    )
```

**`menger-app/src/main/scala/menger/dsl/package.scala`**

```scala
package menger

/** DSL for defining ray-traced scenes in Scala. */
package object dsl:
  
  /** Block-style scene builder function. */
  def scene(block: SceneBuilder ?=> Unit): Scene =
    val builder = SceneBuilder()
    block(using builder)
    builder.build()
  
  // Re-export all DSL types for convenient imports
  export menger.dsl.Vec3
  export menger.dsl.Color
  export menger.dsl.Material
  export menger.dsl.Material.{Glass, Water, Diamond, Chrome, Gold, Copper}
  export menger.dsl.Light
  export menger.dsl.Directional
  export menger.dsl.Point
  export menger.dsl.SceneObject
  export menger.dsl.Sphere
  export menger.dsl.Cube
  export menger.dsl.Sponge
  export menger.dsl.SpongeType
  export menger.dsl.SpongeType.{Volume, Surface, CubeSponge}
  export menger.dsl.Camera
  export menger.dsl.Plane
  export menger.dsl.X
  export menger.dsl.Y
  export menger.dsl.Z
  export menger.dsl.Scene
  export menger.dsl.SceneDefinition
```

**`menger-app/src/main/scala/menger/dsl/SceneDefinition.scala`**

```scala
package menger.dsl

/** Trait for scene definition objects that can be loaded via --scene flag. */
trait SceneDefinition:
  /** The scene to render. */
  val scene: Scene
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/SceneBuilderSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneBuilderSpec extends AnyFlatSpec with Matchers:

  "SceneBuilder" should "build a simple scene with one sphere" in:
    val s = scene {
      sphere(Glass)
    }
    s.objects should have size 1
    s.objects.head shouldBe a[Sphere]

  it should "build scene with camera" in:
    val s = scene {
      camera(position = (0, 0.5, 3), lookAt = (0, 0, 0))
      sphere(Glass)
    }
    s.camera.position shouldBe Vec3(0f, 0.5f, 3f)
    s.camera.lookAt shouldBe Vec3.Zero

  it should "build scene with lights" in:
    val s = scene {
      light(Directional((-1, 1, -1), intensity = 2.0f))
      light(Point((0, 5, 0)))
      sphere(Glass)
    }
    s.lights should have size 2

  it should "build scene with multiple objects" in:
    val s = scene {
      sphere(Glass)
      sphere((2, 0, 0), Chrome, size = 0.8f)
      cube((-2, 0, 0), "#FF0000", size = 1.5f)
    }
    s.objects should have size 3

  it should "build scene with sponge" in:
    val s = scene {
      sponge(Volume, level = 3)
    }
    s.objects should have size 1
    s.objects.head shouldBe a[Sponge]
    s.objects.head.asInstanceOf[Sponge].level shouldBe 3f

  it should "build scene with plane" in:
    val s = scene {
      sphere(Glass)
      plane(Y at -2, "#808080")
    }
    s.plane.isDefined shouldBe true

  it should "build scene with checkered plane" in:
    val s = scene {
      sphere(Glass)
      plane(Y at -2, "FFFFFF", "000000")
    }
    s.plane.get.checkered.isDefined shouldBe true

  it should "reject more than 8 lights" in:
    an[IllegalArgumentException] should be thrownBy:
      scene {
        for i <- 1 to 9 do light(Point((i, 0, 0)))
        sphere(Glass)
      }

  it should "reject empty scene" in:
    an[IllegalArgumentException] should be thrownBy:
      scene {
        camera((0, 0, 3), (0, 0, 0))
        // no objects
      }

  it should "support adding list of lights" in:
    val threePointLighting = List(
      Directional((-1, 1, -1), intensity = 1.0f),
      Directional((1, 0.5, -1), intensity = 0.5f),
      Directional((0, -1, 1), intensity = 0.3f)
    )
    val s = scene {
      lights(threePointLighting)
      sphere(Glass)
    }
    s.lights should have size 3
```

**`menger-app/src/test/scala/menger/dsl/SceneSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneSpec extends AnyFlatSpec with Matchers:

  "Scene (case class style)" should "create with explicit lists" in:
    val s = Scene(
      camera = Camera((0, 0, 3), (0, 0, 0)),
      lights = List(Directional((-1, 1, -1))),
      objects = List(Sphere(Glass), Cube((2, 0, 0), Chrome)),
      plane = Some(Plane(Y at -2, "808080"))
    )
    s.objects should have size 2
    s.lights should have size 1
    s.plane.isDefined shouldBe true

  it should "convert to OptiXEngineConfig" in:
    val s = scene {
      camera((0, 0.5, 3), (0, 0, 0))
      light(Directional((-1, 1, -1), intensity = 2.0f))
      sphere(Glass)
      cube((2, 0, 0), Chrome)
      plane(Y at -2, "808080")
    }
    
    val config = s.toOptiXEngineConfig
    
    // Check camera
    config.camera.position.x shouldBe 0f
    config.camera.position.y shouldBe 0.5f
    config.camera.position.z shouldBe 3f
    
    // Check objects
    config.scene.objectSpecs.isDefined shouldBe true
    config.scene.objectSpecs.get should have size 2
    
    // Check lights
    config.environment.lights should have size 1
    config.environment.lights.head.intensity shouldBe 2.0f
    
    // Check plane
    config.environment.plane.value shouldBe -2f

  it should "use default camera when not specified" in:
    val s = Scene(objects = List(Sphere(Glass)))
    s.camera shouldBe Camera.Default

  it should "reject empty objects list" in:
    an[IllegalArgumentException] should be thrownBy:
      Scene(objects = List.empty)

  it should "reject more than 8 lights" in:
    val tooManyLights = (1 to 9).map(i => Point((i, 0, 0))).toList
    an[IllegalArgumentException] should be thrownBy:
      Scene(lights = tooManyLights, objects = List(Sphere(Glass)))
```

---

### Step 11.6: Create Scene Loader

**Status:** Not Started
**Estimate:** 2 hours

Create the loader that finds and loads scene definitions by class name.

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/SceneLoader.scala`**

```scala
package menger.dsl

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.config.OptiXEngineConfig

/** Loads scene definitions by class name. */
object SceneLoader extends LazyLogging:

  /** 
   * Load a scene definition by fully qualified class name.
   * 
   * @param className Fully qualified class name (e.g., "scenes.MyScene")
   * @return Either an error message or the OptiXEngineConfig
   */
  def load(className: String): Either[String, OptiXEngineConfig] =
    logger.info(s"Loading scene: $className")
    
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
        Left(s"Scene class not found: '$className'. " +
          "Make sure the class exists and is compiled. " +
          s"Error: ${e.getMessage}")
      case Failure(e) =>
        Left(s"Failed to load scene class '$className': ${e.getMessage}")

  private def getInstance(clazz: Class[?]): Either[String, Any] =
    // Scala objects have a MODULE$ field containing the singleton instance
    Try {
      val field = clazz.getField("MODULE$")
      field.get(null)
    } match
      case Success(instance) => Right(instance)
      case Failure(_) =>
        // Try to instantiate as a class instead
        Try(clazz.getDeclaredConstructor().newInstance()) match
          case Success(instance) => Right(instance)
          case Failure(e) =>
            Left(s"Scene '${clazz.getName}' must be an object extending SceneDefinition, " +
              s"or a class with a no-arg constructor. Error: ${e.getMessage}")

  private def extractScene(instance: Any, className: String): Either[String, Scene] =
    instance match
      case sd: SceneDefinition =>
        Try(sd.scene) match
          case Success(scene) => Right(scene)
          case Failure(e) =>
            Left(s"Error evaluating scene in '$className': ${e.getMessage}")
      case _ =>
        Left(s"'$className' does not extend SceneDefinition. " +
          "Scene objects must extend menger.dsl.SceneDefinition trait.")

  private def validateAndConvert(scene: Scene): Either[String, OptiXEngineConfig] =
    Try(scene.toOptiXEngineConfig) match
      case Success(config) =>
        logger.info(s"Scene loaded successfully: ${scene.objects.size} objects, ${scene.lights.size} lights")
        Right(config)
      case Failure(e) =>
        Left(s"Invalid scene configuration: ${e.getMessage}")
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/SceneLoaderSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Test scene definitions (these would normally be in separate files)
object TestSimpleScene extends SceneDefinition:
  val scene = Scene(
    objects = List(Sphere(Glass))
  )

object TestComplexScene extends SceneDefinition:
  val scene = scene {
    camera((0, 0.5, 3), (0, 0, 0))
    light(Directional((-1, 1, -1), intensity = 2.0f))
    sphere(Glass)
    cube((2, 0, 0), Chrome)
    plane(Y at -2, "808080")
  }

object TestInvalidScene extends SceneDefinition:
  val scene: Scene = throw RuntimeException("Scene initialization failed")

class NotASceneDefinition:
  val scene = "not a scene"

class SceneLoaderSpec extends AnyFlatSpec with Matchers:

  "SceneLoader" should "load a simple scene by class name" in:
    val result = SceneLoader.load("menger.dsl.TestSimpleScene")
    result shouldBe a[Right[_, _]]
    result.toOption.get.scene.objectSpecs.get should have size 1

  it should "load a complex scene with all elements" in:
    val result = SceneLoader.load("menger.dsl.TestComplexScene")
    result shouldBe a[Right[_, _]]
    
    val config = result.toOption.get
    config.scene.objectSpecs.get should have size 2
    config.environment.lights should have size 1
    config.camera.position.y shouldBe 0.5f

  it should "return error for non-existent class" in:
    val result = SceneLoader.load("scenes.DoesNotExist")
    result shouldBe a[Left[_, _]]
    result.left.getOrElse("") should include("not found")

  it should "return error for class that doesn't extend SceneDefinition" in:
    val result = SceneLoader.load("menger.dsl.NotASceneDefinition")
    result shouldBe a[Left[_, _]]
    result.left.getOrElse("") should include("SceneDefinition")

  it should "return error when scene initialization fails" in:
    val result = SceneLoader.load("menger.dsl.TestInvalidScene")
    result shouldBe a[Left[_, _]]
    result.left.getOrElse("") should include("Error evaluating scene")
```

---

### Step 11.7: Add CLI Option for Scene Loading

**Status:** Not Started
**Estimate:** 1.5 hours

Add `--scene` CLI option and integrate with Main.scala.

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

Add after other OptiX options (around line 85):

```scala
val sceneClass: ScallopOption[String] = opt[String](
  name = "scene",
  required = false,
  group = optixGroup,
  descr = """Load scene from Scala DSL class.
            |  Format: fully.qualified.ClassName
            |  Example: --scene scenes.MyScene
            |  The class must extend menger.dsl.SceneDefinition""".stripMargin
)
```

Add validation (in the validation section):

```scala
// --scene requires --optix
validateOpt(sceneClass, optix) { (scene, isOptix) =>
  if scene.isDefined && !isOptix then
    Left("--scene requires --optix flag")
  else
    Right(())
}

// --scene is mutually exclusive with --objects
validateOpt(sceneClass, objects) { (scene, objs) =>
  if scene.isDefined && objs.exists(_.nonEmpty) then
    Left("--scene and --objects are mutually exclusive")
  else
    Right(())
}
```

**`menger-app/src/main/scala/Main.scala`**

Modify `createOptiXEngine` method:

```scala
private def createOptiXEngine(opts: MengerCLIOptions)(using ProfilingConfig): OptiXEngine =
  // Check if loading from scene file
  val engineConfig = opts.sceneClass.toOption match
    case Some(className) =>
      // Load from DSL scene class
      menger.dsl.SceneLoader.load(className) match
        case Right(config) => 
          // Apply CLI overrides for execution config
          config.copy(
            execution = ExecutionConfig(
              fpsLogIntervalMs = opts.fpsLogInterval(),
              timeout = opts.timeout(),
              saveName = opts.saveName.toOption,
              enableStats = opts.stats(),
              maxInstances = opts.maxInstances()
            ),
            render = opts.renderConfig,
            caustics = opts.causticsConfig
          )
        case Left(error) =>
          System.err.println(s"Error loading scene: $error")
          sys.exit(1)
    case None =>
      // Original CLI-based configuration
      OptiXEngineConfig(
        scene = SceneConfig(
          spongeType = opts.objectType.toOption.getOrElse("sphere"),
          level = opts.level(),
          material = MaterialConfig(opts.color(), opts.ior()),
          sphereRadius = opts.radius(),
          scale = opts.scale(),
          center = opts.center(),
          objectSpecs = opts.objects.toOption
        ),
        camera = CameraConfig(
          position = opts.cameraPos(),
          lookAt = opts.cameraLookat(),
          up = opts.cameraUp()
        ),
        environment = EnvironmentConfig(
          plane = opts.plane(),
          planeColor = opts.planeColor.toOption,
          lights = opts.light.toOption.getOrElse(List.empty)
        ),
        execution = ExecutionConfig(
          fpsLogIntervalMs = opts.fpsLogInterval(),
          timeout = opts.timeout(),
          saveName = opts.saveName.toOption,
          enableStats = opts.stats(),
          maxInstances = opts.maxInstances()
        ),
        render = opts.renderConfig,
        caustics = opts.causticsConfig
      )
  
  OptiXEngine(engineConfig)
```

Add import at top of Main.scala:

```scala
import menger.dsl.SceneLoader
```

#### Tests to Add

**`menger-app/src/test/scala/menger/cli/SceneCLIOptionsSuite.scala`**

```scala
package menger.cli

import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneCLIOptionsSuite extends AnyFlatSpec with Matchers:

  "--scene" should "be parsed when provided with --optix" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--scene", "scenes.MyScene"))
    opts.sceneClass.toOption shouldBe Some("scenes.MyScene")

  it should "default to None when not provided" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))
    opts.sceneClass.toOption shouldBe None

  it should "require --optix flag" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--scene", "scenes.MyScene"))

  it should "be mutually exclusive with --objects" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq(
        "--optix",
        "--scene", "scenes.MyScene",
        "--objects", "type=sphere"
      ))

  it should "allow other OptiX options" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix",
      "--scene", "scenes.MyScene",
      "--timeout", "5",
      "--shadows"
    ))
    opts.sceneClass.toOption shouldBe Some("scenes.MyScene")
    opts.timeout() shouldBe 5f
    opts.shadows() shouldBe true
```

---

### Step 11.8: Create Example Scene Files

**Status:** Not Started
**Estimate:** 1 hour

Create example scene files demonstrating DSL capabilities.

#### Files to Create

**`menger-app/src/main/scala/scenes/SimpleScene.scala`**

```scala
package scenes

import menger.dsl.*

/** Simple scene with a single glass sphere. */
object SimpleScene extends SceneDefinition:
  val scene = scene {
    sphere(Glass)
  }
```

**`menger-app/src/main/scala/scenes/ThreeSpheres.scala`**

```scala
package scenes

import menger.dsl.*

/** Three spheres with different materials. */
object ThreeSpheres extends SceneDefinition:
  val scene = scene {
    camera((0, 0.5, 4), (0, 0, 0))
    
    light(Directional((-1, 1, -1), intensity = 1.5f))
    
    sphere((-2, 0, 0), Glass)
    sphere((0, 0, 0), Chrome)
    sphere((2, 0, 0), Gold)
    
    plane(Y at -1, "404040")
  }
```

**`menger-app/src/main/scala/scenes/MengerShowcase.scala`**

```scala
package scenes

import menger.dsl.*

/** Menger sponge showcase with various materials. */
object MengerShowcase extends SceneDefinition:
  val scene = scene {
    camera((0, 1, 5), (0, 0, 0))
    
    light(Directional((-1, 2, -1), intensity = 1.2f))
    light(Point((3, 3, 3), intensity = 0.8f))
    
    // Central sponge
    sponge(Volume, level = 2.5f, material = Glass)
    
    // Surrounding spheres
    sphere((2.5, 0, 0), Chrome, size = 0.5f)
    sphere((-2.5, 0, 0), Gold, size = 0.5f)
    sphere((0, 2.5, 0), Diamond, size = 0.5f)
    
    plane(Y at -1.5, "FFFFFF", "808080")  // checkered
  }
```

**`menger-app/src/main/scala/scenes/common/Materials.scala`**

```scala
package scenes.common

import menger.dsl.*

/** Reusable custom material definitions. */
object Materials:
  /** Custom tinted glass */
  val TintedGlass = Glass.copy(color = Color("#E0E8FF"))
  
  /** Custom rose gold */
  val RoseGold = Gold.copy(color = Color("#B76E79"))
  
  /** Brushed metal with higher roughness */
  val BrushedMetal = Chrome.copy(roughness = 0.3f)
  
  /** Frosted glass effect */
  val FrostedGlass = Glass.copy(roughness = 0.4f)
```

**`menger-app/src/main/scala/scenes/common/Lighting.scala`**

```scala
package scenes.common

import menger.dsl.*

/** Reusable lighting setups. */
object Lighting:
  /** Classic three-point lighting */
  val ThreePoint: List[Light] = List(
    Directional((-1, 1, -1), intensity = 1.0f),   // key light
    Directional((1, 0.5, -1), intensity = 0.5f),  // fill light  
    Directional((0, -0.5, 1), intensity = 0.3f)   // back light
  )
  
  /** Dramatic single light from above */
  val Dramatic: List[Light] = List(
    Point((0, 5, 0), intensity = 2.0f)
  )
  
  /** Soft ambient-like lighting */
  val Soft: List[Light] = List(
    Directional((-1, 1, -1), intensity = 0.7f),
    Directional((1, 1, 1), intensity = 0.5f),
    Directional((0, 1, 0), intensity = 0.3f)
  )
```

**`menger-app/src/main/scala/scenes/CustomMaterialsDemo.scala`**

```scala
package scenes

import menger.dsl.*
import scenes.common.Materials.*
import scenes.common.Lighting.*

/** Demo using custom materials and lighting presets. */
object CustomMaterialsDemo extends SceneDefinition:
  val scene = scene {
    camera((0, 1, 5), (0, 0, 0))
    
    lights(ThreePoint)
    
    sphere((-2, 0, 0), TintedGlass)
    sphere((0, 0, 0), RoseGold)
    sphere((2, 0, 0), FrostedGlass)
    
    plane(Y at -1, "303030")
  }
```

---

### Step 11.9: Add Texture Support to DSL

**Status:** Not Started
**Estimate:** 1.5 hours

Add texture support to scene objects in the DSL.

#### Files to Modify

**`menger-app/src/main/scala/menger/dsl/SceneObject.scala`**

Add `texture` parameter to Sphere, Cube, and Sponge:

```scala
/** Sphere object. */
case class Sphere(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None  // NEW
) extends SceneObject:
  def toObjectSpec: ObjectSpec = ObjectSpec(
    objectType = "sphere",
    x = pos.x,
    y = pos.y,
    z = pos.z,
    size = size,
    level = None,
    color = color.map(_.toCommonColor),
    ior = material.map(_.ior).getOrElse(ior),
    material = material.map(_.toOptixMaterial),
    texture = texture  // NEW
  )

// Similar updates for Cube and Sponge...
```

**`menger-app/src/main/scala/menger/dsl/SceneBuilder.scala`**

Add texture overloads:

```scala
/** Add a cube with texture. */
def cube(pos: Vec3, texture: String, size: Float = 1.0f): Unit =
  _objects += Cube(pos = pos, texture = Some(texture), size = size)

def cube(pos: (Float, Float, Float), texture: String, size: Float): Unit =
  _objects += Cube(pos = Vec3(pos._1, pos._2, pos._3), texture = Some(texture), size = size)

def cube(pos: (Int, Int, Int), texture: String, size: Float): Unit =
  _objects += Cube(
    pos = Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat),
    texture = Some(texture),
    size = size
  )

/** Add a sponge with texture. */
def sponge(spongeType: SpongeType, level: Float, texture: String): Unit =
  _objects += Sponge(spongeType, level = level, texture = Some(texture))
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/TextureDSLSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextureDSLSpec extends AnyFlatSpec with Matchers:

  "Cube with texture" should "include texture in ObjectSpec" in:
    val c = Cube(pos = Vec3.Zero, texture = Some("checker.png"), size = 1.0f)
    c.texture shouldBe Some("checker.png")
    c.toObjectSpec.texture shouldBe Some("checker.png")

  "SceneBuilder" should "support cube with texture" in:
    val s = scene {
      cube((0, 0, 0), "brick.png", size = 1.5f)
    }
    s.objects.head.asInstanceOf[Cube].texture shouldBe Some("brick.png")

  it should "support sponge with texture" in:
    val s = scene {
      sponge(Volume, level = 2, "wood.png")
    }
    s.objects.head.asInstanceOf[Sponge].texture shouldBe Some("wood.png")

  "Sphere" should "support texture" in:
    val s = Sphere(pos = Vec3.Zero, texture = Some("earth.png"))
    s.texture shouldBe Some("earth.png")
```

#### Verification

```bash
sbt "project mengerApp" "testOnly menger.dsl.TextureDSLSpec"
```

---

### Step 11.10: Add Render Settings to DSL

**Status:** Not Started
**Estimate:** 1.5 hours

Add render settings (shadows, antialiasing) to the scene DSL.

#### Files to Create

**`menger-app/src/main/scala/menger/dsl/RenderSettings.scala`**

```scala
package menger.dsl

import menger.config.RenderConfig

/** Render settings configuration for DSL use. */
case class RenderSettings(
  shadows: Boolean = false,
  antialiasing: Int = 1,
  maxDepth: Int = 10
):
  require(antialiasing >= 1 && antialiasing <= 16,
    s"Antialiasing must be 1-16, got $antialiasing")
  require(maxDepth >= 1 && maxDepth <= 50,
    s"Max depth must be 1-50, got $maxDepth")

  def toRenderConfig: RenderConfig = RenderConfig(
    shadows = shadows,
    antialiasing = antialiasing,
    maxDepth = maxDepth
  )

object RenderSettings:
  val Default: RenderSettings = RenderSettings()
  val HighQuality: RenderSettings = RenderSettings(
    shadows = true,
    antialiasing = 4,
    maxDepth = 20
  )
  val Fast: RenderSettings = RenderSettings(
    shadows = false,
    antialiasing = 1,
    maxDepth = 5
  )
```

#### Files to Modify

**`menger-app/src/main/scala/menger/dsl/Scene.scala`**

Add render settings to Scene:

```scala
/** Complete scene definition. */
case class Scene(
  camera: Camera = Camera.Default,
  lights: List[Light] = List.empty,
  objects: List[SceneObject] = List.empty,
  plane: Option[Plane] = None,
  render: RenderSettings = RenderSettings.Default  // NEW
):
  // ... existing code ...

  /** Convert to OptiXEngineConfig for rendering. */
  def toOptiXEngineConfig: OptiXEngineConfig =
    // ... existing code ...
    OptiXEngineConfig(
      scene = SceneConfig(objectSpecs = Some(objectSpecs)),
      camera = camera.toCameraConfig,
      environment = EnvironmentConfig(
        plane = planeSpec,
        planeColor = planeColorSpec,
        lights = lightSpecs
      ),
      execution = ExecutionConfig.Default,
      render = render.toRenderConfig  // NEW
    )
```

**`menger-app/src/main/scala/menger/dsl/SceneBuilder.scala`**

Add render settings builder methods:

```scala
class SceneBuilder:
  // ... existing fields ...
  private var _render: RenderSettings = RenderSettings.Default

  // === Render Settings ===

  def shadows(enabled: Boolean = true): Unit =
    _render = _render.copy(shadows = enabled)

  def antialiasing(samples: Int): Unit =
    _render = _render.copy(antialiasing = samples)

  def maxDepth(depth: Int): Unit =
    _render = _render.copy(maxDepth = depth)

  def render(settings: RenderSettings): Unit =
    _render = settings

  def highQuality(): Unit =
    _render = RenderSettings.HighQuality

  // === Build ===

  def build(): Scene =
    require(_objects.nonEmpty, "Scene must have at least one object")
    Scene(
      camera = _camera,
      lights = _lights.toList,
      objects = _objects.toList,
      plane = _plane,
      render = _render  // NEW
    )
```

**`menger-app/src/main/scala/menger/dsl/package.scala`**

Add export for RenderSettings:

```scala
export menger.dsl.RenderSettings
```

#### Tests to Add

**`menger-app/src/test/scala/menger/dsl/RenderSettingsSpec.scala`**

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderSettingsSpec extends AnyFlatSpec with Matchers:

  "RenderSettings" should "have sensible defaults" in:
    val r = RenderSettings()
    r.shadows shouldBe false
    r.antialiasing shouldBe 1
    r.maxDepth shouldBe 10

  it should "validate antialiasing range" in:
    an[IllegalArgumentException] should be thrownBy RenderSettings(antialiasing = 0)
    an[IllegalArgumentException] should be thrownBy RenderSettings(antialiasing = 17)

  it should "validate maxDepth range" in:
    an[IllegalArgumentException] should be thrownBy RenderSettings(maxDepth = 0)
    an[IllegalArgumentException] should be thrownBy RenderSettings(maxDepth = 51)

  "RenderSettings.HighQuality" should "enable shadows and AA" in:
    val r = RenderSettings.HighQuality
    r.shadows shouldBe true
    r.antialiasing shouldBe 4

  "SceneBuilder" should "support shadows() method" in:
    val s = scene {
      shadows()
      sphere(Glass)
    }
    s.render.shadows shouldBe true

  it should "support antialiasing() method" in:
    val s = scene {
      antialiasing(4)
      sphere(Glass)
    }
    s.render.antialiasing shouldBe 4

  it should "support highQuality() shortcut" in:
    val s = scene {
      highQuality()
      sphere(Glass)
    }
    s.render.shadows shouldBe true
    s.render.antialiasing shouldBe 4

  it should "support render(settings) method" in:
    val s = scene {
      render(RenderSettings.Fast)
      sphere(Glass)
    }
    s.render.maxDepth shouldBe 5

  "Scene.toOptiXEngineConfig" should "include render settings" in:
    val s = scene {
      shadows()
      antialiasing(2)
      sphere(Glass)
    }
    val config = s.toOptiXEngineConfig
    config.render.shadows shouldBe true
    config.render.antialiasing shouldBe 2
```

#### Verification

```bash
sbt "project mengerApp" "testOnly menger.dsl.RenderSettingsSpec"
```

---

### Step 11.11: Update Documentation

**Status:** Not Started
**Estimate:** 1 hour

Update changelog, README, and add DSL documentation.

#### Files to Modify

**`CHANGELOG.md`** (add at top):

```markdown
## [0.6.0] - 2026-XX-XX

### Added
- **Scala DSL for Scene Description** - Define scenes using type-safe Scala code
  - Block-style syntax: `scene { sphere(Glass); cube((2,0,0), Chrome) }`
  - Case-class style: `Scene(objects = List(Sphere(Glass)))`
  - Load via CLI: `--scene scenes.MyScene`
  - Reusable materials and lighting setups via standard Scala imports
  - Full IDE support with autocompletion and type checking
```

**`README.md`** (add new section):

```markdown
## Scene Description Language (DSL)

Define scenes using type-safe Scala code:

```scala
// scenes/MyScene.scala
package scenes

import menger.dsl.*

object MyScene extends SceneDefinition:
  val scene = scene {
    camera((0, 0.5, 3), (0, 0, 0))
    light(Directional((-1, 1, -1), intensity = 2.0f))
    
    sphere(Glass)
    cube((2, 0, 0), Chrome)
    sponge(Volume, level = 3, color = "#00FF00")
    
    plane(Y at -2, "#808080")
  }
```

Run with:
```bash
sbt "run --optix --scene scenes.MyScene"
```

See `menger-app/src/main/scala/scenes/` for more examples.
```

**`TODO.md`** (add to backlog):

```markdown
## Backlog - Scene DSL

### Deferred Features
- Caustics settings in DSL (algorithm issues, Sprint 4 deferred)
- Runtime scene evaluation (currently compile-time only)
- Tesseract support (depends on Sprint 8)

### Future Enhancements
- Animation keyframes in DSL (Sprint 12)
- Scene composition (combining multiple scenes)
- Procedural object placement helpers
```

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing: `sbt test --warn`
- [ ] Code compiles without warnings: `sbt compile`
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] Example scenes render correctly
- [ ] CHANGELOG.md updated
- [ ] README.md updated with DSL documentation
- [ ] TODO.md updated with deferred features

---

## Summary

| Step | Task | Estimate | New Files |
|------|------|----------|-----------|
| 11.1 | Core DSL Types (Vec3, Color, Material) | 2h | 3 source + 3 test |
| 11.2 | Light Types | 1h | 1 source + 1 test |
| 11.3 | Scene Object Types | 2.5h | 1 source + 1 test |
| 11.4 | Camera and Plane Types | 1.5h | 2 source + 2 test |
| 11.5 | Scene and SceneBuilder | 3h | 3 source + 2 test |
| 11.6 | Scene Loader | 2h | 1 source + 1 test |
| 11.7 | CLI Integration | 1.5h | Modify 2 + 1 test |
| 11.8 | Example Scenes | 1h | 6 source |
| 11.9 | Texture Support | 1.5h | Modify 2 + 1 test |
| 11.10 | Render Settings DSL | 1.5h | 1 source + 1 test |
| 11.11 | Documentation | 1h | Modify 3 |
| **Total** | | **~18.5h** | ~28 files |

---

## Notes

### Design Decisions

1. **Compile-time only**: Scenes are compiled as part of the project for type safety and IDE support. Runtime evaluation can be added later if needed.

2. **Two syntax styles**: Block-style (`scene { }`) for convenience, case-class style for explicit control. Both produce the same `Scene` type.

3. **Tuple conversions**: Implicit conversions allow `(1, 2, 3)` instead of `Vec3(1f, 2f, 3f)` for conciseness.

4. **Material presets**: Common materials (Glass, Chrome, Gold) are pre-defined; custom materials via `Material(...)` or `.copy()`.

5. **Standard imports**: Reusable definitions use standard Scala `import` statements, no special DSL syntax needed.

### Potential Issues

1. **Implicit conversions**: May cause ambiguity in some cases. Can be resolved by using explicit types.

2. **Tuple vs Vec3**: Multiple overloads needed for Int/Float/Double tuples. May affect compile times.

3. **Error messages**: Scene validation errors need clear messages for DSL users.

### Future Extensions

1. **Texture support**: Add `texture: Option[String]` to objects
2. **Render settings**: Add `render { shadows(); antialiasing() }` block
3. **Animation**: Add keyframe support for position/rotation over time
4. **Procedural helpers**: Grid, circle, random placement utilities

---

## References

- Existing config classes: `menger-app/src/main/scala/menger/config/`
- ObjectSpec parsing: `menger-app/src/main/scala/menger/ObjectSpec.scala`
- Material presets: `optix-jni/src/main/scala/menger/optix/Material.scala`
- Light types: `menger-common/src/main/scala/menger/common/Light.scala`
