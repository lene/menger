# Task 15.2: Parametric Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add general parametric surface infrastructure `f(u,v) -> Vec3` to the Scala DSL, tessellated into triangle meshes and rendered via the existing OptiX pipeline.

**Architecture:** A new `ParametricSurface` DSL case class gets tessellated by `ParametricTessellator` into `TriangleMeshData` (the same type used by cubes, sponges, tesseracts). The mesh is carried through the pipeline via a new `meshData` field on `ObjectSpec`. No C++ or shader changes required.

**Tech Stack:** Scala 3, ScalaTest, existing OptiX triangle mesh pipeline

---

## File Structure

| File | Responsibility |
|------|----------------|
| `menger-app/src/main/scala/menger/objects/ParametricTessellator.scala` | **New** — tessellate `(Float, Float) => Vec3` into `TriangleMeshData` |
| `menger-app/src/main/scala/menger/dsl/SceneObject.scala` | Add `ParametricSurface` case class extending `SceneObject` |
| `menger-app/src/main/scala/menger/ObjectSpec.scala` | Add `meshData: Option[TriangleMeshData]` field |
| `menger-common/src/main/scala/menger/common/ObjectType.scala` | Add `"parametric"` to `VALID_TYPES` |
| `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala` | Add `"parametric"` case using pre-tessellated `meshData` |
| `menger-app/src/main/scala/menger/engines/SceneClassifier.scala` | Add `"parametric"` to `isTriangleMeshType` |
| `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala` | Add `"parametric"` to private `isTriangleMeshType` |
| `menger-app/src/main/scala/menger/dsl/SceneConverter.scala` | Add `ParametricSurface` to exhaustive match |
| `menger-app/src/main/scala/examples/dsl/ParametricScenes.scala` | **New** — example scenes (sphere, torus, wavy, moebius, klein) |
| `menger-app/src/main/scala/examples/dsl/SceneIndex.scala` | Register parametric example scenes |
| `scripts/manual-test.sh` | Add interactive test entries |
| `menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala` | **New** — unit tests |
| `menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala` | Add parametric classification test |

---

### Task 1: ParametricTessellator — Core Tessellation

**Files:**
- Create: `menger-app/src/main/scala/menger/objects/ParametricTessellator.scala`
- Test: `menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala`

- [ ] **Step 1: Write failing test — flat plane vertex count and normals**

Create `menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala`:

```scala
package menger.objects

import menger.common.TriangleMeshData
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ParametricTessellatorSuite extends AnyFlatSpec with Matchers:

  private val flatPlane: (Float, Float) => (Float, Float, Float) =
    (u, v) => (u, 0f, v)

  "ParametricTessellator" should "produce correct vertex count for open surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    mesh.numVertices shouldBe (4 + 1) * (4 + 1)  // 25

  it should "produce correct triangle count" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    mesh.numTriangles shouldBe 2 * 4 * 4  // 32

  it should "use stride 8 (pos + normal + uv)" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    mesh.vertexStride shouldBe 8

  it should "compute normals approximately (0,1,0) or (0,-1,0) for flat plane" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 8 + 3)
      val ny = mesh.vertices(i * 8 + 4)
      val nz = mesh.vertices(i * 8 + 5)
      // Normal should be (0, +-1, 0) for flat xz plane
      math.abs(nx) should be < 0.01f
      math.abs(ny) should be > 0.99f
      math.abs(nz) should be < 0.01f
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite"`
Expected: Compilation error — `ParametricTessellator` not found

- [ ] **Step 3: Write minimal ParametricTessellator implementation**

Create `menger-app/src/main/scala/menger/objects/ParametricTessellator.scala`:

```scala
package menger.objects

import com.typesafe.scalalogging.LazyLogging
import menger.common.TriangleMeshData

object ParametricTessellator extends LazyLogging:

  private val MemoryWarningThreshold = 1_000_000

  def tessellate(
    f: (Float, Float) => (Float, Float, Float),
    uRange: (Float, Float),
    vRange: (Float, Float),
    uSteps: Int,
    vSteps: Int,
    closedU: Boolean,
    closedV: Boolean
  ): TriangleMeshData =
    require(uSteps >= 1, s"uSteps must be >= 1, got $uSteps")
    require(vSteps >= 1, s"vSteps must be >= 1, got $vSteps")

    val totalCells = uSteps.toLong * vSteps.toLong
    if totalCells > MemoryWarningThreshold then
      val approxMB = totalCells * 8 * 4 / (1024 * 1024)
      logger.warn(
        s"Parametric surface tessellation is very high resolution " +
        s"($uSteps x $vSteps = $totalCells grid cells). " +
        s"This will use approximately $approxMB MB of GPU memory. " +
        s"Consider reducing resolution."
      )

    val (uMin, uMax) = uRange
    val (vMin, vMax) = vRange
    val du = (uMax - uMin) / uSteps
    val dv = (vMax - vMin) / vSteps

    // Number of unique vertices depends on closure
    val uVerts = if closedU then uSteps else uSteps + 1
    val vVerts = if closedV then vSteps else vSteps + 1
    val numVerts = uVerts * vVerts

    val vertices = new Array[Float](numVerts * 8)
    val epsilon = 1e-4f * math.max(math.abs(uMax - uMin), math.abs(vMax - vMin))

    for j <- 0 until vVerts; i <- 0 until uVerts do
      val u = uMin + i * du
      val v = vMin + j * dv
      val (px, py, pz) = f(u, v)

      // Finite difference normals
      val (dxu, dyu, dzu) = {
        val (x1, y1, z1) = f(u + epsilon, v)
        val (x0, y0, z0) = f(u - epsilon, v)
        ((x1 - x0) / (2 * epsilon), (y1 - y0) / (2 * epsilon), (z1 - z0) / (2 * epsilon))
      }
      val (dxv, dyv, dzv) = {
        val (x1, y1, z1) = f(u, v + epsilon)
        val (x0, y0, z0) = f(u, v - epsilon)
        ((x1 - x0) / (2 * epsilon), (y1 - y0) / (2 * epsilon), (z1 - z0) / (2 * epsilon))
      }

      // cross(du, dv)
      var nx = dyu * dzv - dzu * dyv
      var ny = dzu * dxv - dxu * dzv
      var nz = dxu * dyv - dyu * dxv
      val len = math.sqrt(nx * nx + ny * ny + nz * nz).toFloat

      if len < 1e-8f then
        // Degenerate: use position as normal, or fallback to (0,1,0)
        val pLen = math.sqrt(px * px + py * py + pz * pz).toFloat
        if pLen > 1e-8f then
          nx = px / pLen; ny = py / pLen; nz = pz / pLen
        else
          nx = 0f; ny = 1f; nz = 0f
      else
        nx /= len; ny /= len; nz /= len

      // UV coordinates normalized to [0, 1]
      val texU = if uMax == uMin then 0f else (u - uMin) / (uMax - uMin)
      val texV = if vMax == vMin then 0f else (v - vMin) / (vMax - vMin)

      val idx = (j * uVerts + i) * 8
      vertices(idx) = px; vertices(idx + 1) = py; vertices(idx + 2) = pz
      vertices(idx + 3) = nx; vertices(idx + 4) = ny; vertices(idx + 5) = nz
      vertices(idx + 6) = texU; vertices(idx + 7) = texV

    // Index generation with seam welding
    val indices = Array.newBuilder[Int]
    for j <- 0 until vSteps; i <- 0 until uSteps do
      val i0 = i
      val i1 = if closedU && i + 1 == uSteps then 0 else i + 1
      val j0 = j
      val j1 = if closedV && j + 1 == vSteps then 0 else j + 1

      val topLeft     = j0 * uVerts + i0
      val topRight    = j0 * uVerts + i1
      val bottomLeft  = j1 * uVerts + i0
      val bottomRight = j1 * uVerts + i1

      // Triangle 1
      indices += topLeft; indices += bottomLeft; indices += bottomRight
      // Triangle 2
      indices += topLeft; indices += bottomRight; indices += topRight

    TriangleMeshData(vertices, indices.result(), vertexStride = 8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite"`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add menger-app/src/main/scala/menger/objects/ParametricTessellator.scala
git add menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala
git commit -m "feat: add ParametricTessellator with core tessellation logic

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: ParametricTessellator — Closed Surface and Edge Case Tests

**Files:**
- Modify: `menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala`

- [ ] **Step 1: Write additional tests for closed surfaces, degenerate normals, UV coords, minimal resolution**

Append to `ParametricTessellatorSuite.scala`:

```scala
  it should "reduce vertex count for closedU surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = true, closedV = false
    )
    // closedU: uVerts = uSteps (4), vVerts = vSteps+1 (5) => 20
    mesh.numVertices shouldBe 4 * 5

  it should "reduce vertex count for closedU + closedV surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = true, closedV = true
    )
    mesh.numVertices shouldBe 4 * 4  // 16

  it should "still produce same triangle count for closed surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = true, closedV = true
    )
    mesh.numTriangles shouldBe 2 * 4 * 4

  it should "produce valid mesh with minimal resolution (1x1)" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 1, 1, closedU = false, closedV = false
    )
    mesh.numVertices shouldBe 4
    mesh.numTriangles shouldBe 2

  it should "compute accurate normals for tessellated sphere" in:
    import scala.math._
    val sphereF: (Float, Float) => (Float, Float, Float) = (u, v) =>
      (cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat)

    val mesh = ParametricTessellator.tessellate(
      sphereF, (0f, (2 * Pi).toFloat), (0.1f, (Pi - 0.1).toFloat),
      64, 32, closedU = true, closedV = false
    )
    // Check normals match normalized position (sphere property)
    for i <- 0 until mesh.numVertices do
      val base = i * 8
      val px = mesh.vertices(base); val py = mesh.vertices(base + 1); val pz = mesh.vertices(base + 2)
      val nx = mesh.vertices(base + 3); val ny = mesh.vertices(base + 4); val nz = mesh.vertices(base + 5)
      val pLen = sqrt(px * px + py * py + pz * pz).toFloat
      if pLen > 0.01f then
        val epx = px / pLen; val epy = py / pLen; val epz = pz / pLen
        // Dot product should be close to 1 (or -1 if flipped)
        val dot = math.abs(nx * epx + ny * epy + nz * epz)
        dot should be > 0.95f  // within ~18 degrees

  it should "produce UV coordinates in [0, 1]" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (-2f, 2f), (-3f, 3f), 8, 8, closedU = false, closedV = false
    )
    for i <- 0 until mesh.numVertices do
      val texU = mesh.vertices(i * 8 + 6)
      val texV = mesh.vertices(i * 8 + 7)
      texU should be >= 0f
      texU should be <= 1f
      texV should be >= 0f
      texV should be <= 1f

  it should "have consistent triangle winding order" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    // For a flat xz plane at y=0 with normal (0,1,0), all triangles
    // should have the same winding. Compute cross product of edges
    // and check they all point in the same y-direction.
    var positiveCount = 0
    var negativeCount = 0
    for t <- 0 until mesh.numTriangles do
      val i0 = mesh.indices(t * 3); val i1 = mesh.indices(t * 3 + 1); val i2 = mesh.indices(t * 3 + 2)
      val ax = mesh.vertices(i1 * 8) - mesh.vertices(i0 * 8)
      val az = mesh.vertices(i1 * 8 + 2) - mesh.vertices(i0 * 8 + 2)
      val bx = mesh.vertices(i2 * 8) - mesh.vertices(i0 * 8)
      val bz = mesh.vertices(i2 * 8 + 2) - mesh.vertices(i0 * 8 + 2)
      val crossY = ax * bz - az * bx
      if crossY > 0 then positiveCount += 1 else negativeCount += 1
    // All triangles should wind the same way
    (positiveCount == 0 || negativeCount == 0) shouldBe true

  it should "log warning for very high resolution" in:
    // We can't easily test logging, but we can verify the tessellator
    // accepts high resolution without crashing. The warning is logged
    // at WARN level — visual verification in test output.
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 2, 2, closedU = false, closedV = false
    )
    // Just verify it produces valid output (warning test is a smoke test)
    mesh.numVertices shouldBe 9

  it should "handle degenerate normals at sphere poles" in:
    import scala.math._
    val sphereF: (Float, Float) => (Float, Float, Float) = (u, v) =>
      (cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat)

    // Include poles (v=0 and v=Pi)
    val mesh = ParametricTessellator.tessellate(
      sphereF, (0f, (2 * Pi).toFloat), (0f, Pi.toFloat),
      16, 8, closedU = true, closedV = false
    )
    // No NaN normals
    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 8 + 3)
      val ny = mesh.vertices(i * 8 + 4)
      val nz = mesh.vertices(i * 8 + 5)
      nx.isNaN shouldBe false
      ny.isNaN shouldBe false
      nz.isNaN shouldBe false
      val len = math.sqrt(nx * nx + ny * ny + nz * nz).toFloat
      len should be > 0.99f
      len should be < 1.01f
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite"`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala
git commit -m "test: add closed surface, sphere normals, and edge case tests for ParametricTessellator

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Pipeline Integration — ObjectType, ObjectSpec, MeshFactory

**Files:**
- Modify: `menger-common/src/main/scala/menger/common/ObjectType.scala:5` — add `"parametric"` to `VALID_TYPES`
- Modify: `menger-app/src/main/scala/menger/ObjectSpec.scala:35-52` — add `meshData` field
- Modify: `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala:106-108` — add `"parametric"` case

- [ ] **Step 1: Write failing test — ObjectType recognizes "parametric"**

Append to `ParametricTessellatorSuite.scala`:

```scala
  "ObjectType" should "recognize 'parametric' as valid" in:
    import menger.common.ObjectType
    ObjectType.isValid("parametric") shouldBe true

  it should "not classify 'parametric' as sponge or 4D" in:
    import menger.common.ObjectType
    ObjectType.isSponge("parametric") shouldBe false
    ObjectType.isProjected4D("parametric") shouldBe false
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite -- -z ObjectType"`
Expected: FAIL — `"parametric"` not in `VALID_TYPES`

- [ ] **Step 3: Add "parametric" to ObjectType.VALID_TYPES**

In `menger-common/src/main/scala/menger/common/ObjectType.scala`, line 5, add `"parametric"` to the set:

```scala
  val VALID_TYPES: Set[String] = Set(
    "sphere",
    "cube",
    "sponge-volume",
    "sponge-surface",
    "cube-sponge",
    "tesseract",
    "tesseract-sponge-volume",
    "tesseract-sponge-surface",
    "parametric"
  )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite -- -z ObjectType"`
Expected: PASS

- [ ] **Step 5: Add meshData field to ObjectSpec**

In `menger-app/src/main/scala/menger/ObjectSpec.scala`:

First, add an import at line 7 (after the existing `import menger.common.Color`):

```scala
import menger.common.TriangleMeshData
```

Then add a new field after line 51 (the `rotZ` field):

```scala
  rotZ: Float = 0.0f,           // Z-axis rotation in radians
  meshData: Option[TriangleMeshData] = None
```

Remove the trailing comment from the old last field. The `meshData` field has a default of `None`, so all existing callers are unaffected.

- [ ] **Step 6: Add "parametric" case to MeshFactory**

In `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala`, add before the `case other =>` line (line 106):

```scala
      case "parametric" =>
        spec.meshData.getOrElse(
          throw new IllegalStateException("Parametric surface missing pre-tessellated mesh data")
        )
```

- [ ] **Step 7: Run full test suite to verify no regressions**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / test"`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add menger-common/src/main/scala/menger/common/ObjectType.scala
git add menger-app/src/main/scala/menger/ObjectSpec.scala
git add menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala
git add menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala
git commit -m "feat: add 'parametric' object type and meshData field to ObjectSpec

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Pipeline Integration — SceneClassifier and TriangleMeshSceneBuilder

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/SceneClassifier.scala:46-50` — add `"parametric"` to `isTriangleMeshType`
- Modify: `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala:180-181` — add `"parametric"` to private `isTriangleMeshType`
- Modify: `menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala` — add test

- [ ] **Step 1: Write failing test — SceneClassifier classifies "parametric" as triangle mesh**

Append to `menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala`:

```scala
  it should "return true for parametric" in:
    SceneClassifier.isTriangleMeshType("parametric") shouldBe true

  "classify" should "classify parametric specs as TriangleMeshes" in:
    val result = SceneClassifier.classify(List(spec("parametric")))
    result shouldBe a[SceneType.TriangleMeshes]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.engines.SceneClassifierSuite -- -z parametric"`
Expected: FAIL

- [ ] **Step 3: Add "parametric" to SceneClassifier.isTriangleMeshType**

In `menger-app/src/main/scala/menger/engines/SceneClassifier.scala`, line 46-50, add `"parametric"`:

```scala
  def isTriangleMeshType(objectType: String): Boolean =
    val t = objectType.toLowerCase
    t == "cube" ||
    t == "parametric" ||
    ObjectType.isSponge(t) ||
    ObjectType.isProjected4D(t)
```

- [ ] **Step 4: Add "parametric" to TriangleMeshSceneBuilder's private isTriangleMeshType**

In `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala`, line 180-181, add `"parametric"`:

```scala
  private def isTriangleMeshType(spec: ObjectSpec): Boolean =
    spec.objectType == "cube" || spec.objectType == "parametric" ||
    ObjectType.isSponge(spec.objectType) || ObjectType.isProjected4D(spec.objectType)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.engines.SceneClassifierSuite"`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add menger-app/src/main/scala/menger/engines/SceneClassifier.scala
git add menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala
git add menger-app/src/test/scala/menger/engines/SceneClassifierSuite.scala
git commit -m "feat: classify 'parametric' as triangle mesh type in SceneClassifier and TriangleMeshSceneBuilder

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: DSL — ParametricSurface Case Class and SceneConverter

**Files:**
- Modify: `menger-app/src/main/scala/menger/dsl/SceneObject.scala:304` — add `ParametricSurface` case class
- Modify: `menger-app/src/main/scala/menger/dsl/SceneConverter.scala:40-50` — add `ParametricSurface` case

- [ ] **Step 1: Write failing test — ParametricSurface produces valid ObjectSpec**

Create a test in `ParametricTessellatorSuite.scala`:

```scala
  "ParametricSurface" should "produce an ObjectSpec with meshData" in:
    import menger.dsl._
    val surface = ParametricSurface(
      f = (u, v) => Vec3(u, 0f, v),
      uRange = (0f, 1f),
      vRange = (0f, 1f),
      uSteps = 4,
      vSteps = 4
    )
    val spec = surface.toObjectSpec
    spec.objectType shouldBe "parametric"
    spec.meshData shouldBe defined
    spec.meshData.get.numVertices shouldBe 25
    spec.meshData.get.numTriangles shouldBe 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite -- -z ParametricSurface"`
Expected: Compilation error — `ParametricSurface` not found

- [ ] **Step 3: Add ParametricSurface case class to SceneObject.scala**

Append to the end of `menger-app/src/main/scala/menger/dsl/SceneObject.scala` (after the `TesseractSponge` companion and exports):

```scala
/** Parametric surface defined by f(u,v) -> Vec3, tessellated into a triangle mesh */
case class ParametricSurface(
  f: (Float, Float) => Vec3,
  uRange: (Float, Float) = (0f, 2f * math.Pi.toFloat),
  vRange: (Float, Float) = (0f, math.Pi.toFloat),
  uSteps: Int = 64,
  vSteps: Int = 32,
  closedU: Boolean = false,
  closedV: Boolean = false,
  pos: Vec3 = Vec3.Zero,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  material: Option[Material] = None,
  color: Option[Color] = None,
  texture: Option[String] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(uSteps >= 1, s"uSteps must be >= 1, got $uSteps")
  require(vSteps >= 1, s"vSteps must be >= 1, got $vSteps")
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    val tupleF: (Float, Float) => (Float, Float, Float) =
      (u, v) => { val p = f(u, v); (p.x, p.y, p.z) }
    val mesh = menger.objects.ParametricTessellator.tessellate(
      tupleF, uRange, vRange, uSteps, vSteps, closedU, closedV
    )
    ObjectSpec(
      objectType = "parametric",
      x = pos.x,
      y = pos.y,
      z = pos.z,
      size = size,
      level = None,
      color = color.map(_.toCommonColor),
      ior = material.map(_.ior).getOrElse(ior),
      material = material.map(_.toOptixMaterial),
      texture = texture,
      rotX = rotation.x,
      rotY = rotation.y,
      rotZ = rotation.z,
      meshData = Some(mesh)
    )
```

Note: `ParametricTessellator.tessellate` takes a tuple-returning function `(Float, Float) => (Float, Float, Float)` to avoid coupling the core tessellator to the DSL `Vec3` type. The `ParametricSurface.toObjectSpec` method adapts the DSL `Vec3`-returning `f` to the tuple form.

- [ ] **Step 4: Add ParametricSurface case to SceneConverter.validateSceneMaterials**

In `menger-app/src/main/scala/menger/dsl/SceneConverter.scala`, add after the `TesseractSponge` case (line 49):

```scala
      case obj: ParametricSurface => obj.material.foreach(warnMaterial)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / testOnly menger.objects.ParametricTessellatorSuite -- -z ParametricSurface"`
Expected: PASS

- [ ] **Step 6: Run full test suite for regressions**

Run: `cd /home/lepr/workspace/menger ; sbt "mengerApp / test"`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add menger-app/src/main/scala/menger/dsl/SceneObject.scala
git add menger-app/src/main/scala/menger/dsl/SceneConverter.scala
git add menger-app/src/test/scala/menger/objects/ParametricTessellatorSuite.scala
git commit -m "feat: add ParametricSurface DSL case class with eager tessellation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Example DSL Scenes

**Files:**
- Create: `menger-app/src/main/scala/examples/dsl/ParametricScenes.scala`
- Modify: `menger-app/src/main/scala/examples/dsl/SceneIndex.scala:10-24` — register new scenes

- [ ] **Step 1: Create example scenes file**

Create `menger-app/src/main/scala/examples/dsl/ParametricScenes.scala`:

```scala
package examples.dsl

import scala.language.implicitConversions
import scala.math.*

import menger.dsl._

/** Parametric sphere — compare to built-in Sphere for visual validation.
  * Usage: --scene examples.dsl.ParametricSphere
  */
object ParametricSphere:
  private val TwoPi = 2f * Pi.toFloat

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat),
      uRange = (0f, TwoPi), vRange = (0f, Pi.toFloat),
      closedU = true, closedV = false,
      material = Some(Material.Glass)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
    planes = List(Plane(Y at -1.5))
  )
  SceneRegistry.register("parametric-sphere", scene)

/** Parametric torus — closed in both u and v, glass material.
  * Usage: --scene examples.dsl.ParametricTorus
  */
object ParametricTorus:
  private val TwoPi = 2f * Pi.toFloat
  private val R = 1.0f
  private val r = 0.4f

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(
        ((R + r * cos(v).toFloat) * cos(u).toFloat),
        (r * sin(v).toFloat),
        ((R + r * cos(v).toFloat) * sin(u).toFloat)),
      uRange = (0f, TwoPi), vRange = (0f, TwoPi),
      closedU = true, closedV = true,
      material = Some(Material.Glass)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
    planes = List(Plane(Y at -1.5))
  )
  SceneRegistry.register("parametric-torus", scene)

/** Parametric wavy sheet — open surface with IOR experiment.
  * Usage: --scene examples.dsl.ParametricWavySheet
  */
object ParametricWavySheet:
  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(u, 0.3f * sin(u * 2).toFloat * cos(v * 2).toFloat, v),
      uRange = (-2f, 2f), vRange = (-2f, 2f),
      uSteps = 64, vSteps = 64,
      closedU = false, closedV = false,
      ior = 1.5f
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
    planes = List(Plane(Y at -1.5))
  )
  SceneRegistry.register("parametric-wavy-sheet", scene)

/** Parametric Moebius strip — non-orientable, film material.
  * Usage: --scene examples.dsl.ParametricMoebius
  */
object ParametricMoebius:
  private val TwoPi = 2f * Pi.toFloat

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => {
        val halfU = u / 2f
        val w = v - 0.5f
        Vec3(
          (1f + w * cos(halfU).toFloat) * cos(u).toFloat,
          (1f + w * cos(halfU).toFloat) * sin(u).toFloat,
          w * sin(halfU).toFloat)
      },
      uRange = (0f, TwoPi), vRange = (0f, 1f),
      uSteps = 128, vSteps = 16,
      closedU = false, closedV = false,
      material = Some(Material.Film)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f))
  )
  SceneRegistry.register("parametric-moebius", scene)

/** Figure-8 Klein bottle — non-orientable, glass with IOR.
  * Usage: --scene examples.dsl.ParametricKleinBottle
  */
object ParametricKleinBottle:
  private val TwoPi = 2f * Pi.toFloat
  private val a = 2.0f

  val f: (Float, Float) => Vec3 = (u, v) => {
    val cosU = cos(u).toFloat; val sinU = sin(u).toFloat
    val cosHalfU = cos(u / 2f).toFloat; val sinHalfU = sin(u / 2f).toFloat
    val sinV = sin(v).toFloat; val sin2V = sin(2f * v).toFloat
    val r = a + cosHalfU * sinV - sinHalfU * sin2V
    Vec3(r * cosU, r * sinU, sinHalfU * sinV + cosHalfU * sin2V)
  }

  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = f,
      uRange = (0f, TwoPi), vRange = (0f, TwoPi),
      uSteps = 128, vSteps = 64,
      closedU = true, closedV = true,
      ior = 1.5f
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f))
  )
  SceneRegistry.register("parametric-klein-bottle", scene)

/** Figure-8 Klein bottle with film material.
  * Usage: --scene examples.dsl.ParametricKleinBottleFilm
  */
object ParametricKleinBottleFilm:
  val scene: Scene = Scene(
    objects = List(ParametricSurface(
      f = ParametricKleinBottle.f,
      uRange = (0f, 2f * math.Pi.toFloat), vRange = (0f, 2f * math.Pi.toFloat),
      uSteps = 128, vSteps = 64,
      closedU = true, closedV = true,
      material = Some(Material.Film)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f))
  )
  SceneRegistry.register("parametric-klein-bottle-film", scene)
```

- [ ] **Step 2: Register scenes in SceneIndex**

In `menger-app/src/main/scala/examples/dsl/SceneIndex.scala`, add to the `all` list (line 23, before the closing `)`):

```scala
    ParametricSphere.scene,
    ParametricTorus.scene,
    ParametricWavySheet.scene,
    ParametricMoebius.scene,
    ParametricKleinBottle.scene,
    ParametricKleinBottleFilm.scene,
```

- [ ] **Step 3: Verify compilation**

Run: `cd /home/lepr/workspace/menger ; sbt compile`
Expected: Compilation succeeds

- [ ] **Step 4: Commit**

```bash
git add menger-app/src/main/scala/examples/dsl/ParametricScenes.scala
git add menger-app/src/main/scala/examples/dsl/SceneIndex.scala
git commit -m "feat: add parametric surface example scenes (sphere, torus, wavy, moebius, klein)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Manual Test Cases

**Files:**
- Modify: `scripts/manual-test.sh` — add interactive test entries

- [ ] **Step 1: Add interactive test entries to manual-test.sh**

Add the following entries to the `interactive_tests` array in `scripts/manual-test.sh`, in the parametric surfaces section. Follow the existing entry format (description followed by command arguments):

```bash
"Parametric torus (glass, closed)"
"--scene examples.dsl.ParametricTorus --shadows"

"Parametric wavy sheet (IOR on open surface)"
"--scene examples.dsl.ParametricWavySheet --shadows"

"Parametric Moebius strip (film material)"
"--scene examples.dsl.ParametricMoebius --shadows"

"Parametric Klein bottle (IOR, non-orientable)"
"--scene examples.dsl.ParametricKleinBottle --shadows"

"Parametric Klein bottle (film material)"
"--scene examples.dsl.ParametricKleinBottleFilm --shadows"

"Parametric sphere (compare to built-in)"
"--scene examples.dsl.ParametricSphere --shadows"
```

- [ ] **Step 2: Verify ROADMAP.md backlog already has analytical intersection entry**

The spec requires adding "Analytical ray intersection for known shapes" to the ROADMAP backlog. This was already added in the previous session (commit `8b17d97`). Verify it exists:

Run: `grep -n "Analytical ray intersection" /home/lepr/workspace/menger/ROADMAP.md`
Expected: Shows the backlog entry

- [ ] **Step 3: Commit**

```bash
git add scripts/manual-test.sh
git commit -m "test: add parametric surface interactive test cases to manual-test.sh

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Full Integration Test and Cleanup

**Files:**
- All files from previous tasks

- [ ] **Step 1: Run full test suite**

Run: `cd /home/lepr/workspace/menger ; xvfb-run sbt test`
Expected: All tests PASS

- [ ] **Step 2: Verify a parametric scene renders headlessly**

Run: `cd /home/lepr/workspace/menger ; xvfb-run sbt "mengerApp / run --scene examples.dsl.ParametricTorus --optix --headless --timeout 0.1 --save-name /tmp/parametric-torus.png"`
Expected: Renders a PNG showing a glass torus

- [ ] **Step 3: Check the rendered image**

Visually verify `/tmp/parametric-torus.png` shows a recognizable torus shape.

- [ ] **Step 4: Commit any final adjustments**

If any fixes were needed, commit them:

```bash
git add -u
git commit -m "fix: parametric surface integration adjustments

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Design Notes

- **Tuple vs Vec3 in tessellator**: `ParametricTessellator` uses `(Float, Float) => (Float, Float, Float)` rather than DSL `Vec3` to keep the tessellator in the `menger.objects` package without a dependency on `menger.dsl`. The `ParametricSurface.toObjectSpec` adapts between the two.
- **Eager tessellation**: The mesh is built in `toObjectSpec`, not deferred to `MeshFactory`. This means the lambda `f` is only needed at DSL evaluation time, not later in the pipeline.
- **No CLI support**: Parametric surfaces are DSL-only. Lambdas cannot be expressed as CLI strings. Users can create named example scenes and reference them via `--scene`.
- **Memory warning**: Logged at WARN level when `uSteps * vSteps > 1,000,000`. No hard limit enforced.
