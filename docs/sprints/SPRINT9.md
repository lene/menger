# Sprint 9: TesseractSponge

**Sprint:** 9 - TesseractSponge
**Status:** Not Started
**Estimate:** 15-20 hours
**Branch:** `feature/sprint-9`
**Depends on:** Sprint 8 (TesseractMesh)

---

## Goal

Render 4D Menger sponges (`TesseractSponge` and `TesseractSponge2`) projected to 3D via OptiX ray tracing.

## Success Criteria

- [ ] `--objects type=tesseract-sponge:level=N` renders volume-based 4D sponge
- [ ] `--objects type=tesseract-sponge-2:level=N` renders surface-based 4D sponge
- [ ] Fractional levels work (e.g., `level=1.5` for animation support)
- [ ] 4D rotation parameters work: `rot-xw`, `rot-yw`, `rot-zw`
- [ ] Materials work on sponges (glass, chrome, etc.)
- [ ] Level limits enforced with warnings (level 3/4 max)
- [ ] All tests pass (~35 new tests)

---

## Background

### Existing 4D Sponge Implementations

The codebase has two TesseractSponge implementations for LibGDX rendering:

| Class | Approach | Level 1 Faces | Level 2 Faces | Level 3 Faces |
|-------|----------|---------------|---------------|---------------|
| `TesseractSponge` | Volume-based (3D Menger analog) | 1,152 | 55,296 | 2,654,208 |
| `TesseractSponge2` | Surface-based | 384 | 6,144 | 98,304 |

**TesseractSponge** (volume-based):
- Divides 4D space into 3×3×3×3 = 81 sub-cubes
- Keeps 48 sub-cubes where `|x|+|y|+|z|+|w| > 2`
- Face count: 48^level × 24

**TesseractSponge2** (surface-based):
- Subdivides each face into 3×3 = 9 sub-faces
- Removes center, adds 8 perpendicular "hole wall" faces
- Face count: 16^level × 24

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `TesseractSponge` | `menger-app/.../higher_d/TesseractSponge.scala` | Volume-based 4D sponge |
| `TesseractSponge2` | `menger-app/.../higher_d/TesseractSponge2.scala` | Surface-based 4D sponge |
| `Mesh4D` | `menger-app/.../higher_d/Mesh4D.scala` | Trait: `faces: Seq[Face4D]` |
| `Fractal4D` | `menger-app/.../higher_d/Fractal4D.scala` | Trait: adds `level: Float` |

### Architecture Decision: Refactor TesseractMesh

Sprint 8 creates `TesseractMesh` for projecting a `Tesseract` to `TriangleMeshData`. 

**This sprint refactors it** to accept any `Mesh4D`, enabling reuse for both sponge types:

```scala
// Before (Sprint 8):
case class TesseractMesh(size: Float, ...) extends TriangleMeshSource:
  private val tesseract = Tesseract(size = size)
  private def projectedQuads: Seq[Quad3D] = tesseract.faces.map(...)

// After (Sprint 9):
case class Mesh4DProjection(mesh4D: Mesh4D, ...) extends TriangleMeshSource:
  private def projectedQuads: Seq[Quad3D] = mesh4D.faces.map(...)

// TesseractMesh becomes a convenience wrapper:
object TesseractMesh:
  def apply(size: Float, ...): Mesh4DProjection = 
    Mesh4DProjection(Tesseract(size), ...)
```

---

## Tasks

### Step 9.1: Refactor TesseractMesh to Mesh4DProjection

**Status:** Not Started
**Estimate:** 2 hours

Rename and refactor `TesseractMesh` to accept any `Mesh4D` instead of creating its own `Tesseract`.

#### Subtasks

- [ ] Rename `TesseractMesh` to `Mesh4DProjection`
- [ ] Change constructor to accept `Mesh4D` instead of `size: Float`
- [ ] Create `TesseractMesh` companion object as convenience factory
- [ ] Update all usages in `OptiXEngine`
- [ ] Update tests

#### Files to Modify

**`menger-app/src/main/scala/menger/objects/higher_d/TesseractMesh.scala`**

Rename file to `Mesh4DProjection.scala` and refactor:

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
import menger.common.Vector

/**
 * Projects any 4D mesh to a 3D triangle mesh for OptiX rendering.
 *
 * Takes a Mesh4D (which provides faces: Seq[Face4D]) and projects each 4D face
 * to 3D using perspective projection, then converts to triangles.
 *
 * @param mesh4D   The 4D mesh to project (Tesseract, TesseractSponge, etc.)
 * @param center   Center position in 3D space after projection
 * @param eyeW     W-coordinate of 4D eye for projection (default 3.0)
 * @param screenW  W-coordinate of projection screen (default 1.5)
 * @param rotXW    Rotation angle in XW plane in degrees (default 15.0)
 * @param rotYW    Rotation angle in YW plane in degrees (default 10.0)
 * @param rotZW    Rotation angle in ZW plane in degrees (default 0.0)
 */
case class Mesh4DProjection(
  mesh4D: Mesh4D,
  center: Vector3 = Vector3(0f, 0f, 0f),
  eyeW: Float = 3.0f,
  screenW: Float = 1.5f,
  rotXW: Float = 15f,
  rotYW: Float = 10f,
  rotZW: Float = 0f
) extends TriangleMeshSource:

  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")

  private val rotation: Rotation =
    if rotXW == 0f && rotYW == 0f && rotZW == 0f then
      Rotation.identity
    else
      Rotation(rotXW, rotYW, rotZW, Vector[4](0f, 0f, 0f, 0f))

  private val projection = Projection(eyeW, screenW)

  /** Project all 4D faces to 3D quads */
  private def projectedQuads: Seq[Quad3D] =
    mesh4D.faces.map { face4d =>
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

    val merged = TriangleMeshData.merge(meshDataList)
    translateMesh(merged, center)

  private def quadToTriangleMesh(quad: Quad3D): TriangleMeshData =
    val v0 = quad(0)
    val v1 = quad(1)
    val v2 = quad(2)
    val v3 = quad(3)

    val edge1 = new Vector3(v1).sub(v0)
    val edge2 = new Vector3(v3).sub(v0)
    val normal = edge1.crs(edge2).nor()

    val vertices = Array(
      v0.x, v0.y, v0.z, normal.x, normal.y, normal.z, 0f, 0f,
      v1.x, v1.y, v1.z, normal.x, normal.y, normal.z, 1f, 0f,
      v2.x, v2.y, v2.z, normal.x, normal.y, normal.z, 1f, 1f,
      v3.x, v3.y, v3.z, normal.x, normal.y, normal.z, 0f, 1f
    )

    val indices = Array(0, 1, 2, 0, 2, 3)
    TriangleMeshData(vertices, indices, vertexStride = 8)

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


/**
 * Convenience factory for creating a projected Tesseract.
 * Maintains backward compatibility with Sprint 8 API.
 */
object TesseractMesh:
  def apply(
    center: Vector3 = Vector3(0f, 0f, 0f),
    size: Float = 1.0f,
    eyeW: Float = 3.0f,
    screenW: Float = 1.5f,
    rotXW: Float = 15f,
    rotYW: Float = 10f,
    rotZW: Float = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(
      mesh4D = Tesseract(size = size),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )
```

#### Update OptiXEngine

**`menger-app/src/main/scala/menger/engines/OptiXEngine.scala`**

Update import:
```scala
import menger.objects.higher_d.Mesh4DProjection
import menger.objects.higher_d.TesseractMesh
```

The `createMeshForSpec` calls to `TesseractMesh(...)` should continue to work unchanged due to the companion object.

#### Update Tests

**`menger-app/src/test/scala/menger/objects/higher_d/TesseractMeshSpec.scala`**

Rename to `Mesh4DProjectionSpec.scala` and update:

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Mesh4DProjectionSpec extends AnyFlatSpec with Matchers:

  // === TesseractMesh factory tests (backward compatibility) ===

  "TesseractMesh" should "generate correct vertex and triangle counts" in:
    val projection = TesseractMesh()
    val data = projection.toTriangleMesh

    data.numVertices shouldBe 96  // 24 faces × 4 vertices
    data.numTriangles shouldBe 48 // 24 faces × 2 triangles
    data.vertexStride shouldBe 8

  it should "accept size parameter" in:
    val small = TesseractMesh(size = 1.0f).toTriangleMesh
    val large = TesseractMesh(size = 2.0f).toTriangleMesh
    
    def boundingBoxSpan(mesh: menger.common.TriangleMeshData): Float =
      val xs = (0 until mesh.numVertices).map(i => mesh.vertices(i * 8))
      xs.max - xs.min

    boundingBoxSpan(large) shouldBe (boundingBoxSpan(small) * 2f) +- 0.1f

  // === Mesh4DProjection direct tests ===

  "Mesh4DProjection" should "accept any Mesh4D" in:
    val tesseract = Tesseract(size = 1.0f)
    val projection = Mesh4DProjection(tesseract)
    val data = projection.toTriangleMesh

    data.numTriangles shouldBe 48

  it should "work with TesseractSponge level 0" in:
    val sponge = TesseractSponge(0)
    val projection = Mesh4DProjection(sponge)
    val data = projection.toTriangleMesh

    data.numTriangles shouldBe 48  // Same as Tesseract at level 0

  it should "work with TesseractSponge level 1" in:
    val sponge = TesseractSponge(1)
    val projection = Mesh4DProjection(sponge)
    val data = projection.toTriangleMesh

    // 1,152 faces × 2 triangles = 2,304 triangles
    data.numTriangles shouldBe 2304

  it should "work with TesseractSponge2 level 1" in:
    val sponge = TesseractSponge2(1)
    val projection = Mesh4DProjection(sponge)
    val data = projection.toTriangleMesh

    // 384 faces × 2 triangles = 768 triangles
    data.numTriangles shouldBe 768

  // === Existing tests (from Sprint 8) ===
  // ... keep all existing TesseractMesh tests ...
```

---

### Step 9.2: Add Sponge Types to ObjectType

**Status:** Not Started
**Estimate:** 0.5 hours

Add `tesseract-sponge` and `tesseract-sponge-2` as valid object types.

#### Subtasks

- [ ] Add types to `VALID_TYPES`
- [ ] Add to `HYPERCUBE_TYPES` 
- [ ] Add `is4DSponge()` helper method
- [ ] Update `validTypesString` to include new types
- [ ] Add unit tests

#### Files to Modify

**`menger-common/src/main/scala/menger/common/ObjectType.scala`**

```scala
val VALID_TYPES: Set[String] = Set(
  "sphere",
  "cube",
  "sponge-volume",
  "sponge-surface",
  "cube-sponge",
  "tesseract",
  "tesseract-sponge",    // NEW
  "tesseract-sponge-2"   // NEW
)

val HYPERCUBE_TYPES: Set[String] = Set(
  "tesseract",
  "tesseract-sponge",    // NEW
  "tesseract-sponge-2"   // NEW
)

/**
 * Checks if the given object type is a 4D sponge variant.
 */
def is4DSponge(objectType: String): Boolean =
  val lower = objectType.toLowerCase
  lower == "tesseract-sponge" || lower == "tesseract-sponge-2"
```

#### Tests to Add

**`menger-common/src/test/scala/menger/common/ObjectTypeSpec.scala`** (additions)

```scala
it should "recognize tesseract-sponge as valid" in:
  ObjectType.isValid("tesseract-sponge") shouldBe true

it should "recognize tesseract-sponge-2 as valid" in:
  ObjectType.isValid("tesseract-sponge-2") shouldBe true

it should "classify tesseract-sponge as hypercube" in:
  ObjectType.isHypercube("tesseract-sponge") shouldBe true

it should "classify tesseract-sponge-2 as hypercube" in:
  ObjectType.isHypercube("tesseract-sponge-2") shouldBe true

it should "identify tesseract-sponge as 4D sponge" in:
  ObjectType.is4DSponge("tesseract-sponge") shouldBe true

it should "identify tesseract-sponge-2 as 4D sponge" in:
  ObjectType.is4DSponge("tesseract-sponge-2") shouldBe true

it should "not identify tesseract as 4D sponge" in:
  ObjectType.is4DSponge("tesseract") shouldBe false
```

---

### Step 9.3: Add Level Validation for 4D Sponges

**Status:** Not Started
**Estimate:** 1 hour

Add level requirement and limits for 4D sponge types.

#### Subtasks

- [ ] Update ObjectSpec to require level for 4D sponges
- [ ] Add level limit constants to Const
- [ ] Add validation in ObjectSpec.parse()
- [ ] Add unit tests

#### Files to Modify

**`menger-common/src/main/scala/menger/common/Const.scala`**

Add constants (find appropriate section):

```scala
object Engine:
  // ... existing constants ...
  
  // 4D sponge level limits
  val tesseractSpongeMaxLevel: Int = 3      // 2.6M faces at level 3
  val tesseractSponge2MaxLevel: Int = 4     // 1.5M faces at level 4
  val tesseractSpongeWarnLevel: Int = 2     // 55K faces - warn user
  val tesseractSponge2WarnLevel: Int = 3    // 98K faces - warn user
```

**`menger-app/src/main/scala/menger/ObjectSpec.scala`**

Update `validateSpongeLevel` to handle 4D sponges:

```scala
private def validateSpongeLevel(objType: String, level: Option[Float]): Either[String, Unit] =
  if ObjectType.isSponge(objType) && level.isEmpty then
    Left(s"Sponge object requires 'level' field. Add level=<number> to specification. " +
      s"Example: type=$objType:level=2")
  else if ObjectType.is4DSponge(objType) && level.isEmpty then
    Left(s"4D sponge object requires 'level' field. Add level=<number> to specification. " +
      s"Example: type=$objType:level=1")
  else if ObjectType.is4DSponge(objType) && level.isDefined then
    validate4DSpongeLevel(objType, level.get)
  else
    Right(())

private def validate4DSpongeLevel(objType: String, level: Float): Either[String, Unit] =
  val maxLevel = objType.toLowerCase match
    case "tesseract-sponge" => Const.Engine.tesseractSpongeMaxLevel
    case "tesseract-sponge-2" => Const.Engine.tesseractSponge2MaxLevel
    case _ => Int.MaxValue
  
  if level < 0 then
    Left(s"Level must be non-negative, got $level")
  else if level > maxLevel then
    Left(s"Level $level exceeds maximum ($maxLevel) for $objType. " +
      s"High levels generate millions of triangles and may crash.")
  else
    Right(())
```

#### Tests to Add

**`menger-app/src/test/scala/menger/ObjectSpecSpec.scala`** (additions)

```scala
// === 4D Sponge Level Tests ===

"ObjectSpec" should "require level for tesseract-sponge" in:
  val result = ObjectSpec.parse("type=tesseract-sponge:pos=0,0,0")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("level")

it should "require level for tesseract-sponge-2" in:
  val result = ObjectSpec.parse("type=tesseract-sponge-2:pos=0,0,0")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("level")

it should "parse tesseract-sponge with valid level" in:
  val result = ObjectSpec.parse("type=tesseract-sponge:level=2")
  result shouldBe a[Right[_, _]]
  result.map(_.level) shouldBe Right(Some(2f))

it should "parse tesseract-sponge-2 with valid level" in:
  val result = ObjectSpec.parse("type=tesseract-sponge-2:level=3")
  result shouldBe a[Right[_, _]]
  result.map(_.level) shouldBe Right(Some(3f))

it should "reject tesseract-sponge level above maximum" in:
  val result = ObjectSpec.parse("type=tesseract-sponge:level=4")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("maximum")

it should "reject tesseract-sponge-2 level above maximum" in:
  val result = ObjectSpec.parse("type=tesseract-sponge-2:level=5")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("maximum")

it should "accept fractional level for tesseract-sponge" in:
  val result = ObjectSpec.parse("type=tesseract-sponge:level=1.5")
  result shouldBe a[Right[_, _]]
  result.map(_.level) shouldBe Right(Some(1.5f))

it should "reject negative level for tesseract-sponge" in:
  val result = ObjectSpec.parse("type=tesseract-sponge:level=-1")
  result shouldBe a[Left[_, _]]
  result.left.getOrElse("") should include("non-negative")
```

---

### Step 9.4: Create TesseractSpongeMesh Factory Objects

**Status:** Not Started
**Estimate:** 1.5 hours

Create convenience factory objects for both sponge types, similar to `TesseractMesh`.

#### Subtasks

- [ ] Create `TesseractSpongeMesh` factory object
- [ ] Create `TesseractSponge2Mesh` factory object
- [ ] Add unit tests

#### Files to Create

**`menger-app/src/main/scala/menger/objects/higher_d/TesseractSpongeMesh.scala`**

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3

/**
 * Factory for creating projected TesseractSponge meshes.
 *
 * TesseractSponge is the volume-based 4D Menger sponge analog:
 * - Level 0: 24 faces (same as Tesseract)
 * - Level 1: 1,152 faces (48 × 24)
 * - Level 2: 55,296 faces (48² × 24)
 * - Level 3: 2,654,208 faces (48³ × 24) - MAXIMUM
 *
 * @param center   Center position in 3D space after projection
 * @param size     Size of the sponge (default 1.0)
 * @param level    Fractal recursion level (0-3, supports fractional)
 * @param eyeW     W-coordinate of 4D eye for projection (default 3.0)
 * @param screenW  W-coordinate of projection screen (default 1.5)
 * @param rotXW    Rotation angle in XW plane in degrees (default 15.0)
 * @param rotYW    Rotation angle in YW plane in degrees (default 10.0)
 * @param rotZW    Rotation angle in ZW plane in degrees (default 0.0)
 */
object TesseractSpongeMesh:
  def apply(
    center: Vector3 = Vector3(0f, 0f, 0f),
    size: Float = 1.0f,
    level: Float = 1.0f,
    eyeW: Float = 3.0f,
    screenW: Float = 1.5f,
    rotXW: Float = 15f,
    rotYW: Float = 10f,
    rotZW: Float = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(
      mesh4D = TesseractSponge(level),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )

  /** Estimated face count for a given level */
  def estimatedFaces(level: Int): Long =
    math.pow(48, level).toLong * 24

  /** Estimated triangle count for a given level */
  def estimatedTriangles(level: Int): Long =
    estimatedFaces(level) * 2
```

**`menger-app/src/main/scala/menger/objects/higher_d/TesseractSponge2Mesh.scala`**

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3

/**
 * Factory for creating projected TesseractSponge2 meshes.
 *
 * TesseractSponge2 is the surface-based 4D sponge:
 * - Level 0: 24 faces (same as Tesseract)
 * - Level 1: 384 faces (16 × 24)
 * - Level 2: 6,144 faces (16² × 24)
 * - Level 3: 98,304 faces (16³ × 24)
 * - Level 4: 1,572,864 faces (16⁴ × 24) - MAXIMUM
 *
 * @param center   Center position in 3D space after projection
 * @param size     Size of the sponge (default 1.0)
 * @param level    Fractal recursion level (0-4, supports fractional)
 * @param eyeW     W-coordinate of 4D eye for projection (default 3.0)
 * @param screenW  W-coordinate of projection screen (default 1.5)
 * @param rotXW    Rotation angle in XW plane in degrees (default 15.0)
 * @param rotYW    Rotation angle in YW plane in degrees (default 10.0)
 * @param rotZW    Rotation angle in ZW plane in degrees (default 0.0)
 */
object TesseractSponge2Mesh:
  def apply(
    center: Vector3 = Vector3(0f, 0f, 0f),
    size: Float = 1.0f,
    level: Float = 1.0f,
    eyeW: Float = 3.0f,
    screenW: Float = 1.5f,
    rotXW: Float = 15f,
    rotYW: Float = 10f,
    rotZW: Float = 0f
  ): Mesh4DProjection =
    Mesh4DProjection(
      mesh4D = TesseractSponge2(level, size),
      center = center,
      eyeW = eyeW,
      screenW = screenW,
      rotXW = rotXW,
      rotYW = rotYW,
      rotZW = rotZW
    )

  /** Estimated face count for a given level */
  def estimatedFaces(level: Int): Long =
    math.pow(16, level).toLong * 24

  /** Estimated triangle count for a given level */
  def estimatedTriangles(level: Int): Long =
    estimatedFaces(level) * 2
```

#### Tests to Add

**`menger-app/src/test/scala/menger/objects/higher_d/TesseractSpongeMeshSpec.scala`**

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSpongeMeshSpec extends AnyFlatSpec with Matchers:

  "TesseractSpongeMesh" should "generate correct triangle count at level 0" in:
    val mesh = TesseractSpongeMesh(level = 0f).toTriangleMesh
    mesh.numTriangles shouldBe 48  // 24 faces × 2

  it should "generate correct triangle count at level 1" in:
    val mesh = TesseractSpongeMesh(level = 1f).toTriangleMesh
    mesh.numTriangles shouldBe 2304  // 1,152 faces × 2

  it should "generate valid mesh data (no NaN)" in:
    val mesh = TesseractSpongeMesh(level = 1f).toTriangleMesh
    mesh.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }

  it should "apply 4D rotation" in:
    val unrotated = TesseractSpongeMesh(level = 1f, rotXW = 0f, rotYW = 0f).toTriangleMesh
    val rotated = TesseractSpongeMesh(level = 1f, rotXW = 45f, rotYW = 0f).toTriangleMesh
    unrotated.vertices should not equal rotated.vertices

  it should "apply center translation" in:
    val centered = TesseractSpongeMesh(level = 0f, center = Vector3(0f, 0f, 0f)).toTriangleMesh
    val translated = TesseractSpongeMesh(level = 0f, center = Vector3(5f, 0f, 0f)).toTriangleMesh
    
    val dx = translated.vertices(0) - centered.vertices(0)
    dx shouldBe 5f +- 0.01f

  it should "support fractional levels" in:
    val mesh = TesseractSpongeMesh(level = 0.5f).toTriangleMesh
    // Fractional level uses floor, so 0.5 -> level 0
    mesh.numTriangles shouldBe 48

  it should "estimate face counts correctly" in:
    TesseractSpongeMesh.estimatedFaces(0) shouldBe 24
    TesseractSpongeMesh.estimatedFaces(1) shouldBe 1152
    TesseractSpongeMesh.estimatedFaces(2) shouldBe 55296

  it should "estimate triangle counts correctly" in:
    TesseractSpongeMesh.estimatedTriangles(1) shouldBe 2304
    TesseractSpongeMesh.estimatedTriangles(2) shouldBe 110592
```

**`menger-app/src/test/scala/menger/objects/higher_d/TesseractSponge2MeshSpec.scala`**

```scala
package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSponge2MeshSpec extends AnyFlatSpec with Matchers:

  "TesseractSponge2Mesh" should "generate correct triangle count at level 0" in:
    val mesh = TesseractSponge2Mesh(level = 0f).toTriangleMesh
    mesh.numTriangles shouldBe 48  // 24 faces × 2

  it should "generate correct triangle count at level 1" in:
    val mesh = TesseractSponge2Mesh(level = 1f).toTriangleMesh
    mesh.numTriangles shouldBe 768  // 384 faces × 2

  it should "generate correct triangle count at level 2" in:
    val mesh = TesseractSponge2Mesh(level = 2f).toTriangleMesh
    mesh.numTriangles shouldBe 12288  // 6,144 faces × 2

  it should "generate valid mesh data (no NaN)" in:
    val mesh = TesseractSponge2Mesh(level = 1f).toTriangleMesh
    mesh.vertices.foreach { v =>
      v.isNaN shouldBe false
      v.isInfinite shouldBe false
    }

  it should "apply 4D rotation" in:
    val unrotated = TesseractSponge2Mesh(level = 1f, rotXW = 0f, rotYW = 0f).toTriangleMesh
    val rotated = TesseractSponge2Mesh(level = 1f, rotXW = 45f, rotYW = 0f).toTriangleMesh
    unrotated.vertices should not equal rotated.vertices

  it should "support fractional levels" in:
    val mesh = TesseractSponge2Mesh(level = 1.5f).toTriangleMesh
    // Fractional level uses floor, so 1.5 -> level 1
    mesh.numTriangles shouldBe 768

  it should "estimate face counts correctly" in:
    TesseractSponge2Mesh.estimatedFaces(0) shouldBe 24
    TesseractSponge2Mesh.estimatedFaces(1) shouldBe 384
    TesseractSponge2Mesh.estimatedFaces(2) shouldBe 6144
    TesseractSponge2Mesh.estimatedFaces(3) shouldBe 98304
```

---

### Step 9.5: Integrate Sponges into OptiXEngine

**Status:** Not Started
**Estimate:** 2.5 hours

Add 4D sponge support to OptiXEngine's mesh creation and scene setup.

#### Subtasks

- [ ] Update `isTriangleMeshType()` to include 4D sponges
- [ ] Add cases to `createMeshForSpec()`
- [ ] Add cases to `geometryGenerator` for single-object mode
- [ ] Add level warning logging
- [ ] Add unit tests

#### Files to Modify

**`menger-app/src/main/scala/menger/engines/OptiXEngine.scala`**

Add imports:
```scala
import menger.objects.higher_d.TesseractSpongeMesh
import menger.objects.higher_d.TesseractSponge2Mesh
```

Update `isTriangleMeshType` (around line 182):
```scala
private def isTriangleMeshType(objectType: String): Boolean =
  objectType == "cube" || 
  ObjectType.isSponge(objectType) || 
  ObjectType.isHypercube(objectType)  // Includes tesseract, tesseract-sponge, tesseract-sponge-2
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
    case "tesseract-sponge" =>
      require(spec.level.isDefined, "tesseract-sponge requires level")
      warn4DSpongeLevel("tesseract-sponge", spec.level.get)
      TesseractSpongeMesh(
        center = Vector3(0f, 0f, 0f),
        size = spec.size,
        level = spec.level.get,
        eyeW = spec.eyeW,
        screenW = spec.screenW,
        rotXW = spec.rotXW,
        rotYW = spec.rotYW,
        rotZW = spec.rotZW
      ).toTriangleMesh
    case "tesseract-sponge-2" =>
      require(spec.level.isDefined, "tesseract-sponge-2 requires level")
      warn4DSpongeLevel("tesseract-sponge-2", spec.level.get)
      TesseractSponge2Mesh(
        center = Vector3(0f, 0f, 0f),
        size = spec.size,
        level = spec.level.get,
        eyeW = spec.eyeW,
        screenW = spec.screenW,
        rotXW = spec.rotXW,
        rotYW = spec.rotYW,
        rotZW = spec.rotZW
      ).toTriangleMesh
    case other =>
      require(false, s"Unknown mesh type: $other")
      ???

private def warn4DSpongeLevel(spongeType: String, level: Float): Unit =
  val intLevel = level.toInt
  spongeType match
    case "tesseract-sponge" =>
      if intLevel >= Const.Engine.tesseractSpongeWarnLevel then
        val triangles = TesseractSpongeMesh.estimatedTriangles(intLevel)
        logger.warn(s"$spongeType level $intLevel generates ~${triangles / 1000}K triangles, may be slow")
    case "tesseract-sponge-2" =>
      if intLevel >= Const.Engine.tesseractSponge2WarnLevel then
        val triangles = TesseractSponge2Mesh.estimatedTriangles(intLevel)
        logger.warn(s"$spongeType level $intLevel generates ~${triangles / 1000}K triangles, may be slow")
    case _ => ()
```

Update `geometryGenerator` for single-object mode (around line 89):
```scala
private val geometryGenerator: Try[OptiXRenderer => Unit] = scene.spongeType match {
  case "sphere" => Try(_.setSphere(scene.center.toVector3, scene.sphereRadius))
  case "cube" => Try { renderer =>
    val cube = Cube(center = scene.center, scale = scene.sphereRadius * 2)
    renderer.setTriangleMesh(cube.toTriangleMesh)
  }
  case "sponge-volume" => Try { renderer =>
    val sponge = SpongeByVolume(center = scene.center, scale = scene.sphereRadius * 2, level = scene.level)
    renderer.setTriangleMesh(sponge.toTriangleMesh)
  }
  case "sponge-surface" => Try { renderer =>
    given menger.ProfilingConfig = profilingConfig
    val sponge = SpongeBySurface(center = scene.center, scale = scene.sphereRadius * 2, level = scene.level)
    renderer.setTriangleMesh(sponge.toTriangleMesh)
  }
  case "tesseract" => Try { renderer =>
    val mesh = TesseractMesh(center = scene.center, size = scene.sphereRadius * 2)
    renderer.setTriangleMesh(mesh.toTriangleMesh)
  }
  case "tesseract-sponge" => Try { renderer =>
    warn4DSpongeLevel("tesseract-sponge", scene.level)
    val mesh = TesseractSpongeMesh(center = scene.center, size = scene.sphereRadius * 2, level = scene.level)
    renderer.setTriangleMesh(mesh.toTriangleMesh)
  }
  case "tesseract-sponge-2" => Try { renderer =>
    warn4DSpongeLevel("tesseract-sponge-2", scene.level)
    val mesh = TesseractSponge2Mesh(center = scene.center, size = scene.sphereRadius * 2, level = scene.level)
    renderer.setTriangleMesh(mesh.toTriangleMesh)
  }
  case _ => Failure(UnsupportedOperationException(scene.spongeType))
}
```

---

### Step 9.6: Update CLI Documentation

**Status:** Not Started
**Estimate:** 0.5 hours

Update CLI help text to document the new 4D sponge types.

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

Update the `--objects` description:
```scala
val objects: ScallopOption[List[ObjectSpec]] = opt[List[ObjectSpec]](
  name = "objects", required = false, group = optixGroup,
  descr = """Objects (repeatable): type=TYPE:pos=x,y,z:size=S[:options]
            |  Types: sphere, cube, sponge-volume, sponge-surface, cube-sponge,
            |         tesseract, tesseract-sponge, tesseract-sponge-2
            |  Options: level=L, color=#RGB, ior=I, material=M, texture=F
            |  4D options: eye-w=W, screen-w=W, rot-xw=DEG, rot-yw=DEG, rot-zw=DEG
            |  Examples:
            |    --objects type=tesseract-sponge:level=2:rot-xw=30:color=#FF0000
            |    --objects type=tesseract-sponge-2:level=3:material=glass""".stripMargin
)
```

---

### Step 9.7: Integration Tests and Manual Verification

**Status:** Not Started
**Estimate:** 2.5 hours

Create integration tests and perform manual visual verification.

#### Subtasks

- [ ] Create integration test file
- [ ] Run manual verification commands
- [ ] Verify material support works
- [ ] Verify different levels produce different results
- [ ] Test performance at higher levels
- [ ] Take screenshots for documentation

#### Files to Create

**`menger-app/src/test/scala/menger/engines/TesseractSpongeIntegrationSpec.scala`**

```scala
package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import menger.objects.higher_d.TesseractSpongeMesh
import menger.objects.higher_d.TesseractSponge2Mesh
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TesseractSpongeIntegrationSpec extends AnyFlatSpec with Matchers:

  // === ObjectSpec Parsing Tests ===

  "TesseractSponge integration" should "parse tesseract-sponge with level" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract-sponge"
    spec.level shouldBe Some(1f)

  it should "parse tesseract-sponge-2 with level" in:
    val result = ObjectSpec.parse("type=tesseract-sponge-2:level=2")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.objectType shouldBe "tesseract-sponge-2"
    spec.level shouldBe Some(2f)

  it should "parse tesseract-sponge with 4D rotation" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.rotXW shouldBe 45f
    spec.rotYW shouldBe 30f

  it should "parse tesseract-sponge with material" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1:material=glass")
    result shouldBe a[Right[_, _]]
    
    val spec = result.toOption.get
    spec.material.isDefined shouldBe true

  it should "parse fractional level" in:
    val result = ObjectSpec.parse("type=tesseract-sponge:level=1.5")
    result shouldBe a[Right[_, _]]
    result.map(_.level) shouldBe Right(Some(1.5f))

  // === Type Classification Tests ===

  it should "classify tesseract-sponge as hypercube" in:
    ObjectType.isHypercube("tesseract-sponge") shouldBe true

  it should "classify tesseract-sponge-2 as hypercube" in:
    ObjectType.isHypercube("tesseract-sponge-2") shouldBe true

  it should "classify tesseract-sponge as 4D sponge" in:
    ObjectType.is4DSponge("tesseract-sponge") shouldBe true

  // === Mesh Generation Tests ===

  it should "generate different meshes for different levels" in:
    val level1 = TesseractSpongeMesh(level = 1f).toTriangleMesh
    val level2 = TesseractSpongeMesh(level = 2f).toTriangleMesh
    
    level2.numTriangles should be > level1.numTriangles

  it should "generate more triangles for TesseractSponge than TesseractSponge2" in:
    val sponge1 = TesseractSpongeMesh(level = 1f).toTriangleMesh
    val sponge2 = TesseractSponge2Mesh(level = 1f).toTriangleMesh
    
    sponge1.numTriangles should be > sponge2.numTriangles
```

#### Manual Verification Commands

```bash
# TesseractSponge level 1 (1,152 faces)
sbt "run --optix --objects type=tesseract-sponge:level=1:color=#4488FF --save-name sponge4d-l1.png --timeout 5"

# TesseractSponge level 2 (55,296 faces) - may be slow
sbt "run --optix --objects type=tesseract-sponge:level=2:color=#FF8844 --save-name sponge4d-l2.png --timeout 10"

# TesseractSponge2 level 1 (384 faces)
sbt "run --optix --objects type=tesseract-sponge-2:level=1:color=#44FF88 --save-name sponge4d2-l1.png --timeout 5"

# TesseractSponge2 level 2 (6,144 faces)
sbt "run --optix --objects type=tesseract-sponge-2:level=2:color=#FF44FF --save-name sponge4d2-l2.png --timeout 5"

# TesseractSponge2 level 3 (98,304 faces)
sbt "run --optix --objects type=tesseract-sponge-2:level=3:color=#FFFF44 --save-name sponge4d2-l3.png --timeout 10"

# Glass 4D sponge
sbt "run --optix --objects type=tesseract-sponge:level=1:material=glass --save-name sponge4d-glass.png --timeout 5"

# With 4D rotation
sbt "run --optix --objects type=tesseract-sponge:level=1:rot-xw=45:rot-yw=30:color=#FF0000 --save-name sponge4d-rotated.png --timeout 5"

# Compare both sponge types side by side
sbt "run --optix --objects type=tesseract-sponge:level=1:pos=-1.5,0,0:color=#FF0000 --objects type=tesseract-sponge-2:level=1:pos=1.5,0,0:color=#00FF00 --save-name sponge4d-compare.png --timeout 5"
```

---

### Step 9.8: Update Documentation

**Status:** Not Started
**Estimate:** 0.5 hours

Update changelog, roadmap, and TODO.

#### Files to Modify

**`CHANGELOG.md`** (add after Sprint 8 entry):
```markdown
## [0.5.0] - 2026-XX-XX

### Added
- **Tesseract (4D Hypercube)** - Render 4D geometry projected to 3D via OptiX
  - `--objects type=tesseract` for 4D hypercube rendering
  - 4D projection parameters: `eye-w`, `screen-w`
  - 4D rotation parameters: `rot-xw`, `rot-yw`, `rot-zw`
  
- **TesseractSponge (4D Menger Sponge)** - Render 4D fractal sponges
  - `--objects type=tesseract-sponge:level=N` - Volume-based 4D sponge (levels 0-3)
  - `--objects type=tesseract-sponge-2:level=N` - Surface-based 4D sponge (levels 0-4)
  - Fractional level support for smooth animations
  - Full material support (glass, chrome, etc.)
  - Example: `--objects type=tesseract-sponge:level=2:rot-xw=30:material=glass`

### Changed
- Refactored `TesseractMesh` to `Mesh4DProjection` for reuse with any 4D geometry
```

**`ROADMAP.md`** - Update completed sprints table:
```markdown
| 8 | 4D Projection Foundation | ✅ Complete | [archive](docs/archive/sprints/) |
| 9 | TesseractSponge | ✅ Complete | [archive](docs/archive/sprints/) |
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
| 9.1 | Refactor TesseractMesh → Mesh4DProjection | 2h | Rename + refactor `TesseractMesh.scala` |
| 9.2 | Add sponge types to ObjectType | 0.5h | `ObjectType.scala` |
| 9.3 | Add level validation for 4D sponges | 1h | `ObjectSpec.scala`, `Const.scala` |
| 9.4 | Create TesseractSpongeMesh factories | 1.5h | New: `TesseractSpongeMesh.scala`, `TesseractSponge2Mesh.scala` |
| 9.5 | Integrate into OptiXEngine | 2.5h | `OptiXEngine.scala` |
| 9.6 | Update CLI documentation | 0.5h | `MengerCLIOptions.scala` |
| 9.7 | Integration tests | 2.5h | New: `TesseractSpongeIntegrationSpec.scala` |
| 9.8 | Update documentation | 0.5h | `CHANGELOG.md`, `ROADMAP.md` |
| **Total** | | **11-13h** | |

---

## Notes

### Decisions Made

1. **Both sponge types supported:** `tesseract-sponge` and `tesseract-sponge-2`
2. **Level limits:** TesseractSponge max 3, TesseractSponge2 max 4
3. **Refactored architecture:** `Mesh4DProjection` accepts any `Mesh4D`
4. **Fractional levels:** Supported from the start (matches LibGDX behavior)

### Performance Considerations

| Type | Level | Faces | Triangles | Expected Render Time |
|------|-------|-------|-----------|---------------------|
| TesseractSponge | 1 | 1,152 | 2,304 | < 1 sec |
| TesseractSponge | 2 | 55,296 | 110,592 | 2-5 sec |
| TesseractSponge | 3 | 2,654,208 | 5.3M | 10-30 sec |
| TesseractSponge2 | 1 | 384 | 768 | < 1 sec |
| TesseractSponge2 | 2 | 6,144 | 12,288 | < 1 sec |
| TesseractSponge2 | 3 | 98,304 | 196,608 | 2-5 sec |
| TesseractSponge2 | 4 | 1,572,864 | 3.1M | 10-30 sec |

### Potential Issues

1. **Memory usage:** Level 3 TesseractSponge generates 5.3M triangles - may cause OOM on low-memory GPUs
2. **Face generation time:** High levels may take significant time to generate geometry before rendering
3. **Fractional levels:** Current implementation uses floor() - animation smoothing requires interpolation overlay (future enhancement)

---

## References

- TesseractSponge implementation: `menger-app/src/main/scala/menger/objects/higher_d/TesseractSponge.scala`
- TesseractSponge2 implementation: `menger-app/src/main/scala/menger/objects/higher_d/TesseractSponge2.scala`
- Existing test suites: `menger-app/src/test/scala/menger/objects/higher_d/TesseractSpongeSuite.scala`
- Sprint 8 plan: `SPRINT8.md`
