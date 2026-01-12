# Sprint 8: 4D Projection Foundation

**Sprint:** 8 - 4D Projection Foundation
**Status:** Not Started
**Estimate:** 12-15 hours
**Branch:** `feature/sprint-8`

---

## Goal

Render a tesseract (4D hypercube) projected to 3D via OptiX ray tracing using `--objects type=tesseract`.

## Success Criteria

- [ ] `--objects type=tesseract` renders a 4D hypercube projected to 3D
- [ ] 4D rotation parameters work: `rot-xw`, `rot-yw`, `rot-zw`
- [ ] 4D projection parameters work: `eye-w`, `screen-w`
- [ ] Default rotation (15° XW, 10° YW) shows visible 4D structure
- [ ] Materials work on tesseract (glass, chrome, etc.)
- [ ] All tests pass (~40 new tests)

---

## Background

### Existing 4D Infrastructure

The codebase already has extensive 4D support for LibGDX rendering:

| Class | Location | Purpose |
|-------|----------|---------|
| `Tesseract` | `menger-app/.../higher_d/Tesseract.scala` | 16 vertices, 24 faces, 32 edges |
| `Face4D` | `menger-app/.../higher_d/Face4D.scala` | 4D face (quad in 4D space) |
| `Projection` | `menger-app/.../higher_d/Projection.scala` | 4D→3D perspective projection |
| `Rotation` | `menger-app/.../higher_d/Rotation.scala` | 4D rotation matrices (XW, YW, ZW planes) |
| `Quad3D` | `menger-app/.../higher_d/Quad3D.scala` | Projected 3D quad |

### The Gap

Existing 4D code outputs LibGDX `Model` objects. We need to convert projected 4D geometry to `TriangleMeshData` for OptiX rendering.

### Key Insight: Material Support is Free

Investigation confirmed that material support requires **no additional work**. The existing infrastructure handles materials for any `TriangleMeshData`:
- `extractMaterialProperties()` extracts color/IOR from ObjectSpec
- `addTriangleMeshInstance()` applies materials per-instance
- Any class implementing `TriangleMeshSource` automatically gets material support

---

## Tasks

### Step 8.1: Add "tesseract" to Valid Object Types

**Status:** Not Started
**Estimate:** 0.5 hours

Add `"tesseract"` as a valid object type and create helper methods for 4D type classification.

#### Subtasks

- [ ] Add "tesseract" to `VALID_TYPES` set
- [ ] Add `HYPERCUBE_TYPES` set
- [ ] Add `isHypercube()` helper method
- [ ] Add unit tests

#### Files to Modify

**`menger-common/src/main/scala/menger/common/ObjectType.scala`**

```scala
// Add to VALID_TYPES set (around line 14):
val VALID_TYPES: Set[String] = Set(
  "sphere",
  "cube",
  "sponge-volume",
  "sponge-surface",
  "cube-sponge",
  "tesseract"  // NEW
)

// Add new section after SPONGE_TYPES (around line 29):
/**
 * Object types that represent 4D hypercube variants.
 */
val HYPERCUBE_TYPES: Set[String] = Set("tesseract")

// Add helper method (around line 67):
/**
 * Checks if the given object type is a 4D hypercube variant.
 *
 * @param objectType Object type string (case-insensitive)
 * @return true if the type is a hypercube variant, false otherwise
 */
def isHypercube(objectType: String): Boolean =
  HYPERCUBE_TYPES.contains(objectType.toLowerCase)
```

#### Tests to Add

**`menger-common/src/test/scala/menger/common/ObjectTypeSpec.scala`** (new file)

```scala
package menger.common

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ObjectTypeSpec extends AnyFlatSpec with Matchers:

  "ObjectType" should "recognize tesseract as valid" in:
    ObjectType.isValid("tesseract") shouldBe true

  it should "recognize tesseract (case insensitive)" in:
    ObjectType.isValid("TESSERACT") shouldBe true
    ObjectType.isValid("Tesseract") shouldBe true

  it should "classify tesseract as hypercube" in:
    ObjectType.isHypercube("tesseract") shouldBe true

  it should "not classify cube as hypercube" in:
    ObjectType.isHypercube("cube") shouldBe false

  it should "not classify sphere as hypercube" in:
    ObjectType.isHypercube("sphere") shouldBe false

  it should "not classify sponge-volume as hypercube" in:
    ObjectType.isHypercube("sponge-volume") shouldBe false

  it should "include tesseract in validTypesString" in:
    ObjectType.validTypesString should include("tesseract")
```

---

### Step 8.2: Verify Rotation.identity Exists

**Status:** Not Started
**Estimate:** 0.5 hours

Check if `Rotation.identity` exists in the existing 4D code. If not, add it.

#### Subtasks

- [ ] Check `Rotation.scala` for identity rotation
- [ ] Add `Rotation.identity` if missing
- [ ] Run existing 4D tests to verify nothing is broken

#### Files to Check/Modify

**`menger-app/src/main/scala/menger/objects/higher_d/Rotation.scala`**

If `Rotation.identity` doesn't exist, add to companion object:

```scala
object Rotation:
  /** Identity rotation (no transformation) */
  val identity: Rotation = Rotation(
    transformationMatrix = Matrix.identity[4],
    pivotPoint = Vector[4](0f, 0f, 0f, 0f)
  )
  
  // ... existing factory methods ...
```

#### Verification

```bash
sbt "testOnly *Tesseract*"
sbt "testOnly *Projection*"
sbt "testOnly *Rotation*"
sbt "testOnly *Face4D*"
```

---

### Step 8.3: Create TesseractMesh Class

**Status:** Not Started
**Estimate:** 4-5 hours

Create a new class that converts a projected 4D tesseract to `TriangleMeshData` for OptiX rendering.

#### Subtasks

- [ ] Create `TesseractMesh` case class
- [ ] Implement 4D rotation application
- [ ] Implement 4D→3D projection
- [ ] Convert projected quads to triangles with normals and UVs
- [ ] Handle center translation
- [ ] Add comprehensive unit tests

#### Files to Create

**`menger-app/src/main/scala/menger/objects/higher_d/TesseractMesh.scala`**

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector

/**
 * Generates a triangle mesh from a 4D tesseract projected to 3D.
 *
 * The tesseract (4D hypercube) has 16 vertices, 32 edges, 24 square faces,
 * and 8 cubic cells. This class projects it to 3D space for rendering.
 *
 * @param center   Center position in 3D space after projection
 * @param size     Size of the tesseract (side length in 4D, default 1.0)
 * @param eyeW     W-coordinate of 4D eye for projection (default 3.0)
 * @param screenW  W-coordinate of projection screen (default 1.5)
 * @param rotXW    Rotation angle in XW plane in degrees (default 15.0)
 * @param rotYW    Rotation angle in YW plane in degrees (default 10.0)
 * @param rotZW    Rotation angle in ZW plane in degrees (default 0.0)
 */
case class TesseractMesh(
  center: Vector3 = Vector3(0f, 0f, 0f),
  size: Float = 1.0f,
  eyeW: Float = 3.0f,
  screenW: Float = 1.5f,
  rotXW: Float = 15f,
  rotYW: Float = 10f,
  rotZW: Float = 0f
) extends TriangleMeshSource:

  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")

  private val tesseract = Tesseract(size = size)

  private val rotation: Rotation =
    if rotXW == 0f && rotYW == 0f && rotZW == 0f then
      Rotation.identity
    else
      Rotation(rotXW, rotYW, rotZW, Vector[4](0f, 0f, 0f, 0f))

  private val projection = Projection(eyeW, screenW)

  /** Project all 4D faces to 3D quads */
  private def projectedQuads: Seq[Quad3D] =
    tesseract.faces.map { face4d =>
      val rotatedFace = Face4D(
        rotation(face4d.a),
        rotation(face4d.b),
        rotation(face4d.c),
        rotation(face4d.d)
      )
      projection(rotatedFace)
    }

  override def toTriangleMesh: TriangleMeshData =
    val quads = projectedQuads
    val meshDataList = quads.map(quadToTriangleMesh)

    // Merge all quad meshes and translate to center position
    val merged = TriangleMeshData.merge(meshDataList)
    translateMesh(merged, center)

  /** Convert a 3D quad to two triangles with proper normals and UVs */
  private def quadToTriangleMesh(quad: Quad3D): TriangleMeshData =
    val v0 = quad(0)
    val v1 = quad(1)
    val v2 = quad(2)
    val v3 = quad(3)

    // Calculate face normal (cross product of two edges)
    val edge1 = new Vector3(v1).sub(v0)
    val edge2 = new Vector3(v3).sub(v0)
    val normal = edge1.crs(edge2).nor()

    // Vertex format: position(3) + normal(3) + uv(2) = 8 floats
    val vertices = Array(
      v0.x, v0.y, v0.z, normal.x, normal.y, normal.z, 0f, 0f,
      v1.x, v1.y, v1.z, normal.x, normal.y, normal.z, 1f, 0f,
      v2.x, v2.y, v2.z, normal.x, normal.y, normal.z, 1f, 1f,
      v3.x, v3.y, v3.z, normal.x, normal.y, normal.z, 0f, 1f
    )

    // Two triangles: (v0,v1,v2) and (v0,v2,v3)
    val indices = Array(0, 1, 2, 0, 2, 3)

    TriangleMeshData(vertices, indices, vertexStride = 8)

  /** Translate all vertices in the mesh by the center offset */
  private def translateMesh(mesh: TriangleMeshData, offset: Vector3): TriangleMeshData =
    if offset.x == 0f && offset.y == 0f && offset.z == 0f then
      mesh
    else
      val translated = mesh.vertices.clone()
      var i = 0
      while i < translated.length do
        translated(i) += offset.x
        translated(i + 1) += offset.y
        translated(i + 2) += offset.z
        i += mesh.vertexStride
      TriangleMeshData(translated, mesh.indices, mesh.vertexStride)
```

#### Tests to Add

**`menger-app/src/test/scala/menger/objects/higher_d/TesseractMeshSpec.scala`**

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractMeshSpec extends AnyFlatSpec with Matchers:

  // === Geometry Tests ===

  "TesseractMesh" should "generate correct vertex and triangle counts" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    // 24 faces x 4 vertices = 96 vertices
    data.numVertices shouldBe 96
    // 24 faces x 2 triangles = 48 triangles
    data.numTriangles shouldBe 48
    data.vertexStride shouldBe 8

  it should "generate valid vertex data (no NaN or Inf)" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    data.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }

  it should "generate valid index data (within vertex bounds)" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    data.indices.foreach { idx =>
      idx should be >= 0
      idx should be < data.numVertices
    }

  it should "generate normalized normals" in:
    val mesh = TesseractMesh()
    val data = mesh.toTriangleMesh

    for i <- 0 until data.numVertices do
      val nx = data.vertices(i * 8 + 3)
      val ny = data.vertices(i * 8 + 4)
      val nz = data.vertices(i * 8 + 5)
      val length = math.sqrt(nx * nx + ny * ny + nz * nz)
      length shouldBe 1.0 +- 0.001

  // === Transformation Tests ===

  it should "apply center translation correctly" in:
    val offset = Vector3(5f, -3f, 2f)
    val centered = TesseractMesh(center = Vector3(0f, 0f, 0f)).toTriangleMesh
    val translated = TesseractMesh(center = offset).toTriangleMesh

    val dx = translated.vertices(0) - centered.vertices(0)
    val dy = translated.vertices(1) - centered.vertices(1)
    val dz = translated.vertices(2) - centered.vertices(2)

    dx shouldBe offset.x +- 0.001f
    dy shouldBe offset.y +- 0.001f
    dz shouldBe offset.z +- 0.001f

  it should "scale geometry correctly" in:
    val small = TesseractMesh(size = 1.0f).toTriangleMesh
    val large = TesseractMesh(size = 2.0f).toTriangleMesh

    def boundingBoxSpan(mesh: menger.common.TriangleMeshData): Float =
      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 8))
      xs.max - xs.min

    val smallSpan = boundingBoxSpan(small)
    val largeSpan = boundingBoxSpan(large)

    largeSpan shouldBe (smallSpan * 2f) +- 0.1f

  // === 4D Rotation Tests ===

  it should "produce different geometry with XW rotation" in:
    val unrotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = TesseractMesh(rotXW = 45f, rotYW = 0f, rotZW = 0f).toTriangleMesh

    unrotated.vertices should not equal rotated.vertices

  it should "produce different geometry with YW rotation" in:
    val unrotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = TesseractMesh(rotXW = 0f, rotYW = 45f, rotZW = 0f).toTriangleMesh

    unrotated.vertices should not equal rotated.vertices

  it should "produce different geometry with ZW rotation" in:
    val unrotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rotated = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 45f).toTriangleMesh

    unrotated.vertices should not equal rotated.vertices

  it should "produce same geometry for 0 and 360 degree rotation" in:
    val rot0 = TesseractMesh(rotXW = 0f, rotYW = 0f, rotZW = 0f).toTriangleMesh
    val rot360 = TesseractMesh(rotXW = 360f, rotYW = 0f, rotZW = 0f).toTriangleMesh

    rot0.vertices.zip(rot360.vertices).foreach { case (a, b) =>
      a shouldBe b +- 0.01f
    }

  // === Projection Tests ===

  it should "produce larger projection with closer eye" in:
    val far = TesseractMesh(eyeW = 10f, screenW = 5f).toTriangleMesh
    val near = TesseractMesh(eyeW = 3f, screenW = 1.5f).toTriangleMesh

    def maxExtent(mesh: menger.common.TriangleMeshData): Float =
      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 8).abs)
      xs.max

    maxExtent(near) should be > maxExtent(far)

  it should "handle edge-case projection distances" in:
    val farMesh = TesseractMesh(eyeW = 100f, screenW = 99f)
    noException should be thrownBy farMesh.toTriangleMesh

  // === Default Value Tests ===

  it should "have visible 4D structure with default rotation" in:
    val mesh = TesseractMesh()
    mesh.rotXW shouldBe 15f
    mesh.rotYW shouldBe 10f
    val data = mesh.toTriangleMesh
    data.numTriangles shouldBe 48

  // === Validation Tests ===

  it should "reject invalid projection parameters (eyeW <= screenW)" in:
    an[IllegalArgumentException] should be thrownBy:
      TesseractMesh(eyeW = 1.0f, screenW = 2.0f)

  it should "reject negative projection parameters" in:
    an[IllegalArgumentException] should be thrownBy:
      TesseractMesh(eyeW = -1.0f, screenW = 1.0f)
```

---

### Step 8.4: Add ObjectSpec Support for 4D Parameters

**Status:** Not Started
**Estimate:** 1.5 hours

Extend `ObjectSpec` to parse 4D-specific parameters from the CLI.

#### Subtasks

- [ ] Add 4D fields to ObjectSpec case class
- [ ] Add parsing methods for `eye-w`, `screen-w`, `rot-xw`, `rot-yw`, `rot-zw`
- [ ] Add validation for 4D parameters
- [ ] Update `parse()` method to include new parameters
- [ ] Add unit tests

#### Files to Modify

**`menger-app/src/main/scala/menger/ObjectSpec.scala`**

Add to case class (around line 33):
```scala
case class ObjectSpec(
  objectType: String,
  x: Float = 0.0f,
  y: Float = 0.0f,
  z: Float = 0.0f,
  size: Float = 1.0f,
  level: Option[Float] = None,
  color: Option[Color] = None,
  ior: Float = 1.0f,
  material: Option[Material] = None,
  texture: Option[String] = None,
  // 4D projection parameters
  eyeW: Float = 3.0f,
  screenW: Float = 1.5f,
  rotXW: Float = 15f,
  rotYW: Float = 10f,
  rotZW: Float = 0f
)
```

Add parsing methods in companion object:
```scala
private def parseEyeW(kvPairs: Map[String, String]): Either[String, Float] =
  parseFloatParam(kvPairs, "eye-w", 3.0f, "4D eye W-coordinate (e.g., eye-w=3.0)")

private def parseScreenW(kvPairs: Map[String, String]): Either[String, Float] =
  parseFloatParam(kvPairs, "screen-w", 1.5f, "4D screen W-coordinate (e.g., screen-w=1.5)")

private def parseRotXW(kvPairs: Map[String, String]): Either[String, Float] =
  parseFloatParam(kvPairs, "rot-xw", 15f, "XW rotation angle in degrees (e.g., rot-xw=45)")

private def parseRotYW(kvPairs: Map[String, String]): Either[String, Float] =
  parseFloatParam(kvPairs, "rot-yw", 10f, "YW rotation angle in degrees (e.g., rot-yw=30)")

private def parseRotZW(kvPairs: Map[String, String]): Either[String, Float] =
  parseFloatParam(kvPairs, "rot-zw", 0f, "ZW rotation angle in degrees (e.g., rot-zw=15)")

private def parseFloatParam(
  kvPairs: Map[String, String],
  key: String,
  default: Float,
  description: String
): Either[String, Float] =
  kvPairs.get(key) match
    case Some(valueStr) =>
      Try(valueStr.toFloat).toEither.left.map { e =>
        s"Invalid $key value '$valueStr': ${e.getMessage}. Expected a valid $description"
      }
    case None => Right(default)

private def validate4DParams(
  objType: String,
  eyeW: Float,
  screenW: Float
): Either[String, Unit] =
  if ObjectType.isHypercube(objType) then
    if eyeW <= screenW then
      Left(s"eye-w ($eyeW) must be greater than screen-w ($screenW) for 4D projection")
    else if eyeW <= 0 || screenW <= 0 then
      Left("eye-w and screen-w must be positive values")
    else
      Right(())
  else
    Right(())
```

Update `parse()` method for-comprehension to include:
```scala
eyeW <- parseEyeW(kvPairs)
screenW <- parseScreenW(kvPairs)
rotXW <- parseRotXW(kvPairs)
rotYW <- parseRotYW(kvPairs)
rotZW <- parseRotZW(kvPairs)
_ <- validate4DParams(objType, eyeW, screenW)
```

And update the yield to include the new fields.

#### Tests to Add

**`menger-app/src/test/scala/menger/ObjectSpecSpec.scala`** (additions)

```scala
// === Tesseract Type Tests ===

"ObjectSpec" should "parse basic tesseract" in:
  val result = ObjectSpec.parse("type=tesseract")
  result shouldBe a[Right[_, _]]
  result.map(_.objectType) shouldBe Right("tesseract")

it should "parse tesseract with position and size" in:
  val result = ObjectSpec.parse("type=tesseract:pos=1,2,3:size=2.5")
  result.map(_.x) shouldBe Right(1f)
  result.map(_.y) shouldBe Right(2f)
  result.map(_.z) shouldBe Right(3f)
  result.map(_.size) shouldBe Right(2.5f)

it should "use default 4D rotation for tesseract" in:
  val result = ObjectSpec.parse("type=tesseract")
  result.map(_.rotXW) shouldBe Right(15f)
  result.map(_.rotYW) shouldBe Right(10f)
  result.map(_.rotZW) shouldBe Right(0f)

it should "parse tesseract with custom 4D rotation" in:
  val result = ObjectSpec.parse("type=tesseract:rot-xw=45:rot-yw=30:rot-zw=15")
  result.map(_.rotXW) shouldBe Right(45f)
  result.map(_.rotYW) shouldBe Right(30f)
  result.map(_.rotZW) shouldBe Right(15f)

it should "parse tesseract with projection parameters" in:
  val result = ObjectSpec.parse("type=tesseract:eye-w=5.0:screen-w=2.0")
  result.map(_.eyeW) shouldBe Right(5.0f)
  result.map(_.screenW) shouldBe Right(2.0f)

it should "use default projection parameters for tesseract" in:
  val result = ObjectSpec.parse("type=tesseract")
  result.map(_.eyeW) shouldBe Right(3.0f)
  result.map(_.screenW) shouldBe Right(1.5f)

// === Material Tests for Tesseract ===

it should "parse tesseract with color" in:
  val result = ObjectSpec.parse("type=tesseract:color=#FF5500")
  result.map(_.color.isDefined) shouldBe Right(true)

it should "parse tesseract with material preset" in:
  val result = ObjectSpec.parse("type=tesseract:material=glass")
  result.map(_.material.isDefined) shouldBe Right(true)

it should "parse tesseract with IOR" in:
  val result = ObjectSpec.parse("type=tesseract:ior=1.8")
  result.map(_.ior) shouldBe Right(1.8f)

// === Validation Tests ===

it should "reject invalid eye-w (not greater than screen-w)" in:
  val result = ObjectSpec.parse("type=tesseract:eye-w=1.0:screen-w=2.0")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("eye-w")

it should "reject invalid rotation value" in:
  val result = ObjectSpec.parse("type=tesseract:rot-xw=notanumber")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("rot-xw")

// === 4D params on non-4D types (should be ignored, no error) ===

it should "allow 4D params on non-4D types without error" in:
  val result = ObjectSpec.parse("type=cube:rot-xw=45")
  result shouldBe a[Right[_, _]]
```

---

### Step 8.5: Integrate TesseractMesh into OptiXEngine

**Status:** Not Started
**Estimate:** 2 hours

Modify `OptiXEngine` to recognize and render tesseract objects.

#### Subtasks

- [ ] Add import for TesseractMesh
- [ ] Update `isTriangleMeshType()` to include tesseract
- [ ] Add tesseract case to `createMeshForSpec()`
- [ ] Add tesseract case to `geometryGenerator` for single-object mode
- [ ] Add unit tests

#### Files to Modify

**`menger-app/src/main/scala/menger/engines/OptiXEngine.scala`**

Add import:
```scala
import menger.objects.higher_d.TesseractMesh
```

Update `isTriangleMeshType` (around line 182):
```scala
private def isTriangleMeshType(objectType: String): Boolean =
  objectType == "cube" || 
  ObjectType.isSponge(objectType) || 
  ObjectType.isHypercube(objectType)
```

Update `createMeshForSpec` (around line 322):
```scala
private def createMeshForSpec(spec: ObjectSpec): menger.common.TriangleMeshData =
  spec.objectType match
    case "cube" =>
      val cube = Cube(center = Vector3(0f, 0f, 0f), scale = spec.size)
      cube.toTriangleMesh
    case "sponge-volume" =>
      require(spec.level.isDefined, "sponge-volume requires level")
      val sponge = SpongeByVolume(center = Vector3(0f, 0f, 0f), scale = spec.size, level = spec.level.get)
      sponge.toTriangleMesh
    case "sponge-surface" =>
      given menger.ProfilingConfig = profilingConfig
      require(spec.level.isDefined, "sponge-surface requires level")
      val sponge = SpongeBySurface(center = Vector3(0f, 0f, 0f), scale = spec.size, level = spec.level.get)
      sponge.toTriangleMesh
    case "tesseract" =>
      TesseractMesh(
        center = Vector3(0f, 0f, 0f),
        size = spec.size,
        eyeW = spec.eyeW,
        screenW = spec.screenW,
        rotXW = spec.rotXW,
        rotYW = spec.rotYW,
        rotZW = spec.rotZW
      ).toTriangleMesh
    case other =>
      require(false, s"Unknown mesh type: $other")
      ???
```

Update `geometryGenerator` (around line 89):
```scala
private val geometryGenerator: Try[OptiXRenderer => Unit] = scene.spongeType match {
  case "sphere" => Try(_.setSphere(scene.center.toVector3, scene.sphereRadius))
  case "cube" => Try { renderer =>
    val cube = Cube(center = scene.center, scale = scene.sphereRadius * 2)
    val mesh = cube.toTriangleMesh
    renderer.setTriangleMesh(mesh)
  }
  case "sponge-volume" => Try { renderer =>
    val sponge = SpongeByVolume(center = scene.center, scale = scene.sphereRadius * 2, level = scene.level)
    val mesh = sponge.toTriangleMesh
    renderer.setTriangleMesh(mesh)
  }
  case "sponge-surface" => Try { renderer =>
    given menger.ProfilingConfig = profilingConfig
    val sponge = SpongeBySurface(center = scene.center, scale = scene.sphereRadius * 2, level = scene.level)
    val mesh = sponge.toTriangleMesh
    renderer.setTriangleMesh(mesh)
  }
  case "tesseract" => Try { renderer =>
    val mesh = TesseractMesh(
      center = scene.center,
      size = scene.sphereRadius * 2
      // Use default 4D rotation for single-object mode
    )
    renderer.setTriangleMesh(mesh.toTriangleMesh)
  }
  case _ => Failure(UnsupportedOperationException(scene.spongeType))
}
```

---

### Step 8.6: Update CLI Documentation

**Status:** Not Started
**Estimate:** 0.5 hours

Update CLI help text to document the tesseract type and its parameters.

#### Subtasks

- [ ] Update `--objects` description in MengerCLIOptions
- [ ] Verify help output shows tesseract info

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

Update the `--objects` description:
```scala
val objects: ScallopOption[List[ObjectSpec]] = opt[List[ObjectSpec]](
  name = "objects", required = false, group = optixGroup,
  descr = """Objects (repeatable): type=TYPE:pos=x,y,z:size=S[:options]
            |  Types: sphere, cube, sponge-volume, sponge-surface, cube-sponge, tesseract
            |  Options: level=L, color=#RGB, ior=I, material=M, texture=F
            |  Tesseract 4D options: eye-w=W, screen-w=W, rot-xw=DEG, rot-yw=DEG, rot-zw=DEG
            |  Example: --objects type=tesseract:pos=0,0,0:size=2:rot-xw=30:color=#FF0000""".stripMargin
)
```

---

### Step 8.7: Integration Tests and Manual Verification

**Status:** Not Started
**Estimate:** 2 hours

Create integration tests and perform manual visual verification.

#### Subtasks

- [ ] Create integration test file
- [ ] Run manual verification commands
- [ ] Verify material support works
- [ ] Verify different rotations produce different results
- [ ] Take screenshots for documentation

#### Files to Create

**`menger-app/src/test/scala/menger/engines/TesseractIntegrationSpec.scala`**

```scala
package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractIntegrationSpec extends AnyFlatSpec with Matchers:

  "Tesseract integration" should "parse tesseract ObjectSpec correctly" in:
    val result = ObjectSpec.parse("type=tesseract:pos=0,0,0:size=2.0")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract"
    spec.size shouldBe 2.0f

  it should "parse tesseract with 4D rotation" in:
    val result = ObjectSpec.parse("type=tesseract:pos=0,0,0:rot-xw=45:rot-yw=30")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.rotXW shouldBe 45f
    spec.rotYW shouldBe 30f

  it should "parse tesseract with material" in:
    val result = ObjectSpec.parse("type=tesseract:material=glass:color=#88AAFF")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.material.isDefined shouldBe true

  it should "classify tesseract as triangle mesh type" in:
    ObjectType.isHypercube("tesseract") shouldBe true

  it should "classify tesseract as valid type" in:
    ObjectType.isValid("tesseract") shouldBe true
```

#### Manual Verification Commands

```bash
# Basic tesseract with default rotation
sbt "run --optix --objects type=tesseract:pos=0,0,0:size=2:color=#4488FF --save-name tesseract-basic.png --timeout 3"

# Tesseract with no rotation (aligned to axes)
sbt "run --optix --objects type=tesseract:pos=0,0,0:size=2:rot-xw=0:rot-yw=0:color=#FF8844 --save-name tesseract-no-rotation.png --timeout 3"

# Tesseract with 45-degree XW rotation
sbt "run --optix --objects type=tesseract:pos=0,0,0:size=2:rot-xw=45:rot-yw=0:color=#44FF88 --save-name tesseract-rot45.png --timeout 3"

# Glass tesseract
sbt "run --optix --objects type=tesseract:pos=0,0,0:size=2:material=glass --save-name tesseract-glass.png --timeout 3"

# Chrome tesseract
sbt "run --optix --objects type=tesseract:pos=0,0,0:size=2:material=chrome --save-name tesseract-chrome.png --timeout 3"

# Multiple tesseracts (same mesh, different positions)
sbt "run --optix --objects type=tesseract:pos=-2,0,0:color=#FF0000 --objects type=tesseract:pos=2,0,0:color=#00FF00 --save-name tesseract-multi.png --timeout 3"

# Tesseract with sphere
sbt "run --optix --objects type=tesseract:pos=-2,0,0:size=1.5:color=#4488FF --objects type=sphere:pos=2,0,0:color=#FF4444 --save-name tesseract-with-sphere.png --timeout 3"
```

---

### Step 8.8: Update Documentation

**Status:** Not Started
**Estimate:** 0.5 hours

Update changelog, roadmap, and backlog.

#### Subtasks

- [ ] Update CHANGELOG.md with new features
- [ ] Update ROADMAP.md to mark Sprint 8 complete
- [ ] Add multiple instances to backlog in TODO.md
- [ ] Archive sprint documentation

#### Files to Modify

**`CHANGELOG.md`** (add at top):
```markdown
## [0.5.0] - 2026-XX-XX

### Added
- **Tesseract (4D Hypercube)** - Render 4D geometry projected to 3D via OptiX
  - `--objects type=tesseract` for 4D hypercube rendering
  - 4D projection parameters: `eye-w`, `screen-w` (default: 3.0, 1.5)
  - 4D rotation parameters: `rot-xw`, `rot-yw`, `rot-zw` (default: 15, 10, 0)
  - Full material support (glass, chrome, etc.)
  - Example: `--objects type=tesseract:pos=0,0,0:size=2:rot-xw=30:material=glass`
```

**`TODO.md`** (add to backlog):
```markdown
## Backlog - 4D Features

- Multiple tesseract instances with independent 4D rotations
  - Currently all tesseracts in a scene share the same mesh (projection/rotation parameters)
  - Enhancement: per-instance 4D rotation via modified ObjectSpec handling
  - Requires: generate separate mesh per unique (eyeW, screenW, rotXW, rotYW, rotZW) combination
  - Or: implement 4D rotation as shader-time transformation
```

**`ROADMAP.md`** - Update completed sprints table:
```markdown
| 8 | 4D Projection Foundation | ✅ Complete | [archive](docs/archive/sprints/) |
```

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing (Scala + C++): `sbt test --warn`
- [ ] Code compiles without warnings: `sbt compile`
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Manual verification screenshots captured
- [ ] Sprint documentation archived

---

## Summary

| Step | Task | Estimate | Files |
|------|------|----------|-------|
| 8.1 | Add "tesseract" to ObjectType | 0.5h | `ObjectType.scala`, new `ObjectTypeSpec.scala` |
| 8.2 | Verify Rotation.identity | 0.5h | `Rotation.scala` |
| 8.3 | Create TesseractMesh | 4-5h | New: `TesseractMesh.scala`, `TesseractMeshSpec.scala` |
| 8.4 | Add ObjectSpec 4D parameters | 1.5h | `ObjectSpec.scala`, `ObjectSpecSpec.scala` |
| 8.5 | Integrate into OptiXEngine | 2h | `OptiXEngine.scala` |
| 8.6 | Update CLI documentation | 0.5h | `MengerCLIOptions.scala` |
| 8.7 | Integration tests | 2h | New: `TesseractIntegrationSpec.scala` |
| 8.8 | Update documentation | 0.5h | `CHANGELOG.md`, `TODO.md`, `ROADMAP.md` |
| **Total** | | **12-15h** | |

---

## Notes

### Decisions Made

1. **Default 4D rotation:** 15° XW, 10° YW - ensures 4D structure is immediately visible
2. **Projection defaults:** eyeW=3.0, screenW=1.5 - matches existing LibGDX 4D rendering
3. **Material support:** Confirmed free - no additional work needed
4. **Multiple instances:** Deferred to backlog - all tesseracts in a scene share the same mesh for now

### Existing Code to Leverage

- `Tesseract` class: Already has 16 vertices, 24 faces, 32 edges
- `Projection` class: Already does 4D→3D perspective projection
- `Rotation` class: Already does 4D rotation in XW/YW/ZW planes
- `TriangleMeshData.merge()`: Can combine multiple quads into one mesh

### Potential Issues

1. **Degenerate faces:** Some 4D faces may project to zero-area triangles when viewed edge-on. Monitor for visual artifacts.
2. **Winding order:** May need adjustment if backface culling causes missing faces.
3. **Normal calculation:** Current implementation uses per-quad normals. May need per-triangle normals for non-planar projected quads.

---

## References

- Existing 4D code: `menger-app/src/main/scala/menger/objects/higher_d/`
- Triangle mesh pipeline: `menger-common/.../TriangleMeshData.scala`
- OptiX integration: `menger-app/src/main/scala/menger/engines/OptiXEngine.scala`
- Object parsing: `menger-app/src/main/scala/menger/ObjectSpec.scala`
